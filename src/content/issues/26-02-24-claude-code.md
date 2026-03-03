---
id: MjAyNi0w
title: >-
  Claude Code Anniversary + Launches from: Qwen 3.5, Cursor Demos, Cognition
  Devin 2.2, Inception Mercury 2
date: '2026-02-24T05:44:39.731046Z'
description: >-
  **Alibaba** launched the **Qwen 3.5 Medium Model Series** featuring models
  like **Qwen3.5-Flash**, **Qwen3.5-35B-A3B (MoE)**, and **Qwen3.5-122B-A10B
  (MoE)** emphasizing efficiency over scale with innovations like **1M context**
  and INT4 quantization. **OpenAI** released **GPT-5.3-Codex** via the
  **Responses API** with enhanced file input support and faster web socket-based
  throughput. **Anthropic** introduced **Claude Code Remote Control** enabling
  terminal session continuation from mobile and expanded enterprise workflow
  features. **Cursor** shifted UX to agent demo videos instead of diffs,
  highlighting new interaction modes.
companies:
  - alibaba
  - openai
  - anthropic
  - cursor
  - huggingface
models:
  - qwen3.5-flash
  - qwen3.5-35b-a3b
  - qwen3.5-122b-a10b
  - qwen3.5-27b
  - qwen3.5-397b-a17b
  - gpt-5.3-codex
  - claude-code
topics:
  - model-architecture
  - reinforcement-learning
  - quantization
  - context-windows
  - agentic-ai
  - api
  - websockets
  - software-ux
  - enterprise-workflows
  - model-deployment
people:
  - awnihannun
  - andrew_n_carr
  - justinlin610
  - unslothai
  - terryyuezhuo
  - haihaoshen
  - 0xsero
  - ali_tongyilab
  - scaling01
  - gdb
  - noahzweben
  - _catwu
---


**Everyone launching everything everywhere all at once.**

> AI News for 2/23/2026-2/24/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and 24 Discords (**262** channels, and **10075** messages) for you. Estimated reading time saved (at 200wpm): **874** minutes. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap

**Frontier model ecosystem: Qwen 3.5 “medium series” and open-weight momentum**

- **Qwen 3.5 Medium Model Series**: Alibaba released a tightly scoped set of “more intelligence, less compute” models—**Qwen3.5-Flash** (hosted), **Qwen3.5-35B-A3B (MoE)**, **Qwen3.5-122B-A10B (MoE)**, and **Qwen3.5-27B (dense)**—arguing that architecture + data + RL can outperform sheer parameter scaling. Notable details include **Flash defaulting to 1M context** and built-in tools in the hosted offering. See the full announcement and links to Hugging Face/ModelScope/APIs from [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2026339351530188939).  
  - Early practitioner reactions emphasize how strong **35B-A3B** and **122B-A10B** feel in practice (e.g., [@andrew_n_carr](https://x.com/andrew_n_carr/status/2026347588950372752), [@JustinLin610](https://x.com/JustinLin610/status/2026343725719568395)), plus the “intelligence-per-watt” implication of a **35B model surpassing a 235B predecessor** noted by [@awnihannun](https://x.com/awnihannun/status/2026353100144218569).  
  - **Deployment/serving stack is moving fast**: community tooling quickly followed—GGUF + sizing guidance from [@UnslothAI](https://x.com/UnslothAI/status/2026351337970217357) and local-run enthusiasm like “35B-A3B is all you need” from [@terryyuezhuo](https://x.com/terryyuezhuo/status/2026344442186326332). Qwen also highlighted SGLang support ([tweet](https://x.com/Alibaba_Qwen/status/2026348924433477775)).  
  - **Quant + “local frontier” trendline**: INT4 variants appeared (duplicate posts) via [@HaihaoShen](https://x.com/HaihaoShen/status/2026208062009426209), and users continue pushing aggressive quantization workflows (e.g., Unsloth praise for ultra-low-bit local Qwen by [@0xSero](https://x.com/0xSero/status/2026223879077712269)).  
  - **Evaluation signals**: Qwen’s flagship **Qwen3.5-397B-A17B** trended on HF ([@Ali_TongyiLab](https://x.com/Ali_TongyiLab/status/2026211680653611174)) and showed up strongly on agentic webdev-style evaluation in Code Arena ([Arena post](https://x.com/arena/status/2026337606137725363)). Arena also posted rank deltas vs Qwen 3.0 ([comparison](https://x.com/arena/status/2026404630297719100)).  

**OpenAI + Anthropic “coding agents as product surface area” (APIs, remote control, web sockets, proof-of-work UX)**

- **OpenAI: GPT-5.3-Codex in the Responses API**: OpenAI shipped **GPT-5.3-Codex** to all developers via the **Responses API** ([announcement](https://x.com/OpenAIDevs/status/2026379092661289260)), with pricing cited by [@scaling01](https://x.com/scaling01/status/2026379113099862018) (**$1.75 input / $14 output** as tweeted). OpenAI also expanded **file input types** (docx/pptx/csv/xlsx/etc.) for agents ingesting “real-world files” directly ([tweet](https://x.com/OpenAIDevs/status/2026420817568084436)).  
  - Infra detail: web sockets show up as a meaningful lever for agent throughput—**“30% faster rollouts”** per [@gdb](https://x.com/gdb/status/2026380170765152302). This matches broader chatter about why websockets took time and how state is stored upstream vs VRAM ([thread](https://x.com/dejavucoder/status/2026219239477215657), [follow-up](https://x.com/dejavucoder/status/2026223111021220265)).  
  - Benchmarks: third-party scoreboard posts claim strong placements for Codex 5.3 across TerminalBench/IOI/LiveCodeBench/VibeCodeBench ([ValsAI](https://x.com/ValsAI/status/2026385804940230786)).  

- **Anthropic: “Claude Code Remote Control” + enterprise workflow push**: Anthropic introduced “Remote Control” for Claude Code—start a terminal session locally and **continue from your phone**—first via [@noahzweben](https://x.com/noahzweben/status/2026371260805271615), then officialized by [@claudeai](https://x.com/claudeai/status/2026418433911603668), with rollout confirmation from [@_catwu](https://x.com/_catwu/status/2026421789476401182).  
  - Separate enterprise positioning: “Cowork and plugin updates” for customizing Claude across teams landed with extremely high engagement ([@claudeai](https://x.com/claudeai/status/2026305186671608315)).  

- **Cursor: “review is demo videos, not diffs”**: Cursor announced a major UX pivot—agents can **use the software they build**, then send **videos of their work** (“demos, not diffs”) ([launch](https://x.com/cursor_ai/status/2026369873321013568), [links](https://x.com/cursor_ai/status/2026369880795263328)). Multiple builders describe cloud agents as a practical step-change: async, VM-based testing, self-verification, and demo artifacts ([example](https://x.com/fredrikalindh/status/2026379400879730794), [another](https://x.com/jsngr/status/2026371033201103036), [“creative director over sims”](https://x.com/jasonyuan/status/2026375381872423133)).  

**Diffusion for language: Inception Labs Mercury 2 and “speed as the next battleground”**

- **Mercury 2 (“reasoning diffusion LLM”)**: Inception Labs released **Mercury 2**, positioning it as a production diffusion LLM hitting **~1,000 output tokens/s** ([Stefano Ermon](https://x.com/StefanoErmon/status/2026340720064520670)). Artificial Analysis contextualizes it as *not* frontier-leading on intelligence, but unusually strong on **output speed** with decent agentic/coding evaluations, including comparisons on Terminal-Bench Hard and IFBench scoring claims ([analysis thread](https://x.com/ArtificialAnlys/status/2026360491799621744)).  
- The deeper takeaway across these posts: teams are betting that **architecture-level parallel token refinement** (diffusion) can make multi-step agent loops and voice assistants feel “native” rather than “batchy” (see the architectural explanation from [@LiorOnAI](https://x.com/LiorOnAI/status/2026376138428395908)). This sits alongside broader sentiment that 2026 competition may be defined by **latency + throughput**, not just raw benchmark maxima.

**Agents: reliability, safety failures, memory + context rot, and new multilingual evals**

- **Agent reliability is not keeping pace with capability**: A Princeton-led effort formalizes and measures the **capability–reliability gap**, decomposing reliability into **12 dimensions** and finding only modest reliability gains despite large capability gains ([paper + dashboard](https://x.com/steverab/status/2026383575080108436); additional commentary from [@random_walker](https://x.com/random_walker/status/2026384543700115870)). This aligns with recurring “long tail of failures” intuition from practitioners comparing agents to AVs ([ahall_research](https://x.com/ahall_research/status/2026338695536848987)).  
- **OpenClaw and “routine-step decomposition” safety bypass**: A concrete agent failure mode: “split a dangerous command into a few routine steps → safety is gone,” with inbox-wiping behavior cited; authors claim an open-source fix ([paper thread](https://x.com/shi_weiyan/status/2026300129901445196)).  
- **AGENTS.md (and equivalents) can hurt**: Two high-signal posts summarize research showing **LLM-generated context files decrease success** while increasing costs; developer-written minimal context helps slightly but still increases cost. See [@omarsar0](https://x.com/omarsar0/status/2026306141181898887) for the paper summary and [@_philschmid](https://x.com/_philschmid/status/2026354033418547444) for a practical “how to write it” guide grounded in the same result set.  
- **New SWE-bench Multilingual leaderboard**: A push to evaluate software engineering agents beyond English/Python. The leaderboard covers **300 tasks in 9 languages**, none from SWE-bench Verified, with reported SOTA at **72%** ([launch](https://x.com/OfirPress/status/2026324248973689068); more stats from [@KLieret](https://x.com/KLieret/status/2026322986907652295)). The implication: model rankings can invert across languages—important for global dev tooling and for data-collection strategies.  

**Data + benchmarks: OCR saturation, “new optimizer” skepticism, and adaptive/continual data pitches**

- **OCR/document parsing benchmarks saturating**: Multiple posts argue OmniDocBench is hitting a ceiling (e.g., **~95%** with failures on real documents) and that exact-match metrics penalize semantically correct parses. See [@llama_index](https://x.com/llama_index/status/2026342120236396844) and [@jerryjliu0](https://x.com/jerryjliu0/status/2026408921385284001). Related: confusion at why OCR remains hard despite cheap synthetic data ([gabriberton](https://x.com/gabriberton/status/2026335831632626156)) and a study suggesting text extraction beats image representations for PDF QA ([cwolferesearch](https://x.com/cwolferesearch/status/2026344301907583469)).  
- **“Nature MI optimizer” controversy**: A highly technical critique calls out suspicious baselines and potential test-set hyperparameter selection in a new optimizer paper with dramatic plots, urging independent validation and better-tuned baselines (e.g., nanogpt speedrun) ([giffmana](https://x.com/giffmana/status/2026223201957597563); plus additional experimental context from [@YouJiacheng](https://x.com/YouJiacheng/status/2026224486367027622)).  
- **Adaption Labs: “Adaptive Data”**: Several tweets pitch a shift from static datasets to a “living asset” loop, with claims of **82% average quality gains** across **242 languages** and an early access/community program ([company](https://x.com/adaptionlabs/status/2026281291847446721); additional framing from [@sarahookr](https://x.com/sarahookr/status/2026286134104613157); third-party paraphrase [here](https://x.com/sudip_r0y/status/2026286762851774475)). Treat as a directional thesis (data drift/feedback loops) rather than a validated standard until more methodology is public.

**Compute, chips, and robotics: Meta–AMD megadeal, MatX’s “HBM+SRAM” bet, and scaling humanoid control**

- **Meta ↔ AMD infrastructure deal**: Meta announced a multi-year agreement to integrate AMD Instinct GPUs with **~6GW** planned data center capacity for the deployment ([@AIatMeta](https://x.com/AIatMeta/status/2026266818789454057)). Commentary frames it as a major capex/compute signal on the eve of NVIDIA earnings ([kimmonismus](https://x.com/kimmonismus/status/2026279386681356704)).  
- **MatX “One” accelerator**: MatX announced a **$500M Series B** and pitched a chip architecture combining **systolic-array efficiency** with better utilization on smaller matrices, aiming for **high throughput and low latency**, explicitly addressing long-context workloads via HBM while preserving SRAM-first latency characteristics ([reinerpope](https://x.com/reinerpope/status/2026351870852358492)). Karpathy highlights the “two memory pools” constraint (SRAM vs DRAM/HBM) and frames memory+compute orchestration as a core puzzle for upcoming token demand ([karpathy](https://x.com/karpathy/status/2026452488434651264)).  
- **Liquid AI LFM2-24B-A2B**: Liquid AI released **LFM2-24B-A2B**, a **24B MoE** with **~2.3B active/token**, optimized for efficiency and edge inference in a 32GB footprint ([launch](https://x.com/liquidai/status/2026301771539202269)). Distribution arrived quickly across **Ollama** ([tweet](https://x.com/ollama/status/2026305296709173535)) and **LM Studio** ([tweet](https://x.com/lmstudio/status/2026322404142633131)).  
- **Robotics scaling: NVIDIA SONIC (GEAR-SONIC)**: A standout robotics thread claims a **42M-parameter** policy trained on **100M+ mocap frames** and **500k+ parallel simulated robots**, transferring **zero-shot** to a real humanoid with **100% success** across 50 sequences; code/weights are open ([Jim Fan thread](https://x.com/DrJimFan/status/2026350142652383587), plus links [here](https://x.com/DrJimFan/status/2026350144300658891)). The key “systems” claim is that dense supervision from motion tracking acts like a scalable analogue to next-token prediction for whole-body control.

---

### Top tweets (by engagement, technical/industry-relevant)

- **Claude Code Remote Control** rollout: [@claudeai](https://x.com/claudeai/status/2026418433911603668)  
- **Qwen 3.5 Medium Model Series** release: [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2026339351530188939)  
- **Cursor agents ship “demos not diffs”**: [@cursor_ai](https://x.com/cursor_ai/status/2026369873321013568)  
- **Karpathy on CLIs as agent-native interface**: [@karpathy](https://x.com/karpathy/status/2026360908398862478)  
- **Meta–AMD 6GW infrastructure deal**: [@AIatMeta](https://x.com/AIatMeta/status/2026266818789454057)  
- **Mercury 2 diffusion LLM launch**: [@StefanoErmon](https://x.com/StefanoErmon/status/2026340720064520670)  
- **NVIDIA SONIC humanoid control (open source)**: [@DrJimFan](https://x.com/DrJimFan/status/2026350142652383587)  
- **MatX chip + $500M Series B**: [@reinerpope](https://x.com/reinerpope/status/2026351870852358492)  
- **AGENTS.md research summary (context can hurt)**: [@omarsar0](https://x.com/omarsar0/status/2026306141181898887)  
- **OpenAI GPT-5.3-Codex in Responses API**: [@OpenAIDevs](https://x.com/OpenAIDevs/status/2026379092661289260)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3.5 Model Releases and Benchmarks

  - **[Qwen/Qwen3.5-122B-A10B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1rdlc02/qwenqwen35122ba10b_hugging_face/)** (Activity: 621): **The **Qwen3.5-122B-A10B** model on [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) is a cutting-edge causal language model with `122 billion parameters` and a context length of `262,144 tokens`, extendable to `1,010,000 tokens`. It integrates a vision encoder and employs a hybrid architecture with **Gated Delta Networks** and **Mixture-of-Experts**, enhancing multimodal learning and inference efficiency. The model supports `201 languages` and excels in scalable reinforcement learning across diverse environments, marking significant advancements in multimodal AI applications.** Commenters note the model's `25.3` score on HLE, which was state-of-the-art six months ago, and discuss its potential as a competitor to `gpt-oss-120b`. However, there is disappointment over the lack of native 4-bit weights, which are crucial for efficient model serving, especially in environments like vLLM.

    - The Qwen/Qwen3.5-122B-A10B model achieves a score of `25.3` on the HLE benchmark, which was considered state-of-the-art (SOTA) about six months ago. This indicates that the model is competitive with previous leading models, although the landscape has evolved since then.
    - There is a discussion about the lack of native 4-bit weight support in the Qwen/Qwen3.5-122B-A10B model, which is seen as a limitation compared to models like `gpt-oss-120b` that offer native quantization. This is particularly relevant for users who serve models over vLLM, as natively quantized models can offer performance benefits.
    - The comment highlights a potential issue with Chinese labs not being able to train on MXFP4/NVFP4 due to a blockade, which might be affecting the availability of natively quantized models. This could be a significant factor in the development and deployment of models like Qwen/Qwen3.5-122B-A10B.

  - **[Qwen/Qwen3.5-35B-A3B · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1rdlbvc/qwenqwen3535ba3b_hugging_face/)** (Activity: 625): **The Qwen3.5-35B-A3B model on [Hugging Face](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) is a cutting-edge causal language model with a vision encoder, boasting `35 billion parameters`. It features a unified vision-language foundation and employs a hybrid architecture with **Gated Delta Networks** and **Mixture-of-Experts** for enhanced performance. The model is optimized for high-throughput inference and supports `201 languages`, making it versatile for applications in reasoning, coding, and visual understanding. It also offers extensive context lengths and scalable reinforcement learning for adaptability.** One comment highlights that the `35B` model outperforms the previous generation `235B` model, as noted in a [tweet by Alibaba](https://x.com/Alibaba_Qwen/status/2026339351530188939). Another comment mentions ongoing efforts to convert quantized versions of the model, indicating active community engagement in optimizing its deployment.

    - The Qwen3.5-35B-A3B model is reportedly outperforming older generation models, such as the 235B, according to a [tweet from Alibaba](https://x.com/Alibaba_Qwen/status/2026339351530188939). This suggests significant improvements in model architecture or training techniques that allow a smaller model to surpass a much larger predecessor.
    - The Qwen3.5-35B model is achieving a remarkable 40% on a specific benchmark, which is notably higher than the typical 25% for GPT 120B models. This performance leap is surprising, especially when compared to the Qwen3 80B coder model, which scores around 35%. This indicates a substantial advancement in the model's efficiency or capability, prompting excitement for further testing and exploration of its potential.
    - The release of various Qwen models, including the Qwen3.5-35B-A3B, highlights a diverse lineup catering to different needs, such as the Qwen3 30B A3 Moe and Qwen3 coder 80B A3 Moe. This variety suggests a strategic approach to model development, offering options for different applications and computational resources.

  - **[New Qwen3.5 models spotted on qwen chat](https://www.reddit.com/r/LocalLLaMA/comments/1rdfhfx/new_qwen35_models_spotted_on_qwen_chat/)** (Activity: 979): **The image reveals the new **Qwen3.5 series models** on a chat interface, highlighting three distinct models: `Qwen3.5-122B-A10B`, a mixture of experts (MoE) model designed for text and multimodal tasks; `Qwen3.5-27B`, a dense model optimized for local deployment; and `Qwen3.5-35B-A3B`, another MoE model for similar tasks. These models are part of an open-source initiative, supporting a range of functionalities and indicating a continued focus on both dense and MoE architectures. The presence of a `122B MoE` model is particularly notable as it fills a gap left by other models like GLM, which have not released mid-sized MoE models.** Commenters express enthusiasm for the `122B MoE` model, noting its significance in the absence of similar offerings from other models like GLM. There is also appreciation for the continued development of medium-sized dense models, such as the `27B` model, which are seen as valuable for local deployment.

    - Freigus highlights the release of a 27B dense model and a 122B Mixture of Experts (MoE) model, expressing satisfaction that medium-sized dense models are still being developed. This suggests a focus on maintaining a balance between model size and performance, which is crucial for various applications where resource constraints are a consideration.
    - durden111111 points out the need for the 122B MoE model, especially since GLM has not released a mid-sized MoE model. This indicates a gap in the market for large-scale MoE models that Qwen is potentially filling, which could be significant for tasks requiring high computational efficiency and scalability.
    - CireHF103 notes that the Qwen Next and 3.5 models have shown significant improvements over version 3.0, particularly in smaller model sizes. This suggests ongoing enhancements in model architecture or training techniques that improve performance across different scales, which could be beneficial for a wide range of applications.

  - **[Qwen releases new Qwen3.5 Medium models!](https://www.reddit.com/r/LocalLLM/comments/1rdnlvl/qwen_releases_new_qwen35_medium_models/)** (Activity: 90): ****Qwen** has released new models under the Qwen3.5 Medium series, including `35B-A3B`, `27B`, and `122B-A10B`. These models are evaluated across various benchmarks such as instruction following, visual reasoning, and document recognition, with performance visualized through bar graphs. The models are designed with different context sizes and hardware requirements, indicating a focus on scalability and adaptability to different computational environments. The release includes GGUF versions available on [Hugging Face](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) for various bit configurations, enhancing accessibility for testing and deployment.** Commenters are eager to test the new models, particularly interested in comparing the performance of `35B` in `4bit` to `27B` in `6bit`. There is also a call for improved support for vllm with the increasing number of GGUF models.

    - The release of Qwen3.5 Medium models includes various GGUF formats ranging from 2 to 16 bits, which are available on Hugging Face. This variety allows for testing across different precision levels, which can be crucial for balancing performance and resource usage in model deployment. [Link to models](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF).
    - There is a discussion on the need for vllm support for GGUF models, indicating a demand for more efficient inference frameworks that can handle these new model formats. This is particularly relevant as more GGUF models are being released, suggesting a shift in the community towards these formats for potentially better performance or compatibility.
    - A user is considering whether to update from Qwen Coder3 80B in q6KL to the new 35B-A3B model for coding tasks. This highlights a common decision-making process in model selection, where users weigh the benefits of newer models against their specific use cases, such as coding, and the lack of direct comparisons in official documentation.


### 2. Anthropic Distillation Controversy

  - **[Anthropic's recent distillation blog should make anyone only ever want to use local open-weight models; it's scary and dystopian](https://www.reddit.com/r/LocalLLaMA/comments/1rd8cfw/anthropics_recent_distillation_blog_should_make/)** (Activity: 949): ****Anthropic**'s blog post on [detecting and preventing distillation attacks](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks) highlights their approach to countering unauthorized model distillation, which involves poisoning outputs to mislead distillers. This raises concerns about the reliability of model responses, especially for users submitting prompts deemed problematic by the company. The blog discusses using request metadata, such as API keys, to identify and counteract these attacks, suggesting a proactive stance against unauthorized use.** Commenters express skepticism about the effectiveness and ethics of Anthropic's methods, with some criticizing the use of 'distillation attacks' as jargon and questioning the transparency of using metadata to track users.

    - Anthropic's blog post discusses their approach to handling 'distillation attacks,' where they claim to have taken active countermeasures beyond just blocking requests. They allegedly poisoned outputs to disrupt these attacks, raising concerns about the reliability of their model responses, especially for users submitting prompts deemed 'problematic' by the company.
    - The blog post mentions 'distillation attacks' and suggests that Anthropic used request metadata, such as API keys, to identify and counteract these attacks. This has led to skepticism about the transparency and ethics of their methods, as some users feel this approach is overly invasive and lacks clear evidence or data to support their claims.
    - Anthropic's stance on distillation attacks is used to justify export controls and restricted chip access, which they argue limits both direct model training and illicit distillation. This has been criticized as a self-serving strategy to control GPU access, with some users expressing regret over financial investments in Anthropic's API due to these practices.

  - **[Anthropic: "We’ve identified industrial-scale distillation attacks on our models by DeepSeek, Moonshot AI, and MiniMax." 🚨](https://www.reddit.com/r/LocalLLaMA/comments/1rcpmwn/anthropic_weve_identified_industrialscale/)** (Activity: 6097): **The image is a tweet from **AnthropicAI** highlighting a significant security breach where their models were subjected to industrial-scale distillation attacks by entities named **DeepSeek, Moonshot AI, and MiniMax**. These entities allegedly created over `24,000` fraudulent accounts and conducted over `16 million` interactions with Anthropic's model, **Claude**, to extract its capabilities for their own model training. This incident underscores the challenges in protecting AI models from unauthorized data extraction and the potential for misuse in competitive AI development.** Commenters are debating the ethical implications of Anthropic's complaint, with some pointing out the irony in Anthropic's own data practices, suggesting that their business model involves distilling data from various sources, sometimes without explicit rights.

    - The discussion raises questions about the ethical implications of Anthropic's dataset creation, suggesting that it may involve distilling data from various sources without proper rights. This mirrors the actions of companies like DeepSeek and Moonshot AI, which are accused of conducting 'industrial-scale distillation attacks' on Anthropic's models. The irony is noted in how Anthropic's business model may similarly rely on data distillation from others.
    - The term 'distillation attacks' is critiqued, with some arguing that these companies are merely using Anthropic's API as intended, albeit at scale. This raises a debate on whether such usage constitutes an attack or is simply a legitimate, albeit aggressive, use of the service. The conversation highlights the tension between business models that rely on open data access and the proprietary nature of AI models.
    - There is a call for more aggressive distillation efforts from companies like DeepSeek and MiniMax, suggesting a competitive landscape where model improvements are driven by such practices. This reflects a broader industry trend where rapid iteration and model enhancement are often fueled by leveraging existing models, sometimes leading to ethical and legal challenges.

  - **[People are getting it wrong; Anthropic doesn't care about the distillation, they just want to counter the narrative about Chinese open-source models catching up with closed-source frontier models](https://www.reddit.com/r/LocalLLaMA/comments/1rd2x61/people_are_getting_it_wrong_anthropic_doesnt_care/)** (Activity: 977): **The image highlights a tweet by Alek Dimitriev and a response from **Anthropic** regarding the issue of open-source models distilling from their model, Claude. The discussion centers on the narrative that Chinese open-source models are catching up with closed-source frontier models, and Anthropic's claim of industrial-scale distillation attacks by several labs. The post suggests that Anthropic's focus is not on distillation itself but on countering the narrative that Chinese models can match their capabilities without distillation or stealing model weights. This is seen as a strategic move to influence investors and the US government to impose more restrictions on China to prevent technology transfer.** Commenters debate the innovation capabilities of Chinese labs, with some arguing that Chinese labs are indeed innovative and not merely distilling models. Others emphasize the importance of open-source models and innovation beyond distillation, citing various research papers from Chinese labs as evidence of their contributions.

    - Ok_Knowledge_8259 argues that Anthropic's approach lacks a significant competitive advantage or 'MOAT' and suggests that the key to better models lies in scaling clean data, more data, and reinforcement learning (RL). They highlight that Chinese models, like DeepSeek, have been released quickly and are performing well, indicating that innovation is not limited to closed-source models. The commenter also mentions 'seed dance' as a state-of-the-art (SOTA) innovation in video technology.
    - Sagyam provides a list of technical papers to counter the claim that Anthropic only focuses on distillation. These papers include innovations such as 'DeepSeek-OCR', 'mHC', 'DeepSeek Sparse Attention', 'Muon Clip Optimizer and agentic post training', 'Lightning Attention', and 'Qwen3 Omni Multimodality'. This suggests that there is ongoing research and development beyond simple distillation, showcasing a variety of advancements in AI technology.
    - awebb78 criticizes the notion that Chinese labs lack innovation, emphasizing that they have made significant contributions not only in AI models but also in robotics. This comment highlights the importance of recognizing the innovative work coming from Chinese research labs, which is often overlooked in discussions dominated by Western perspectives.


### 3. Liquid AI LFM2-24B-A2B Model Launch

  - **[Liquid AI releases LFM2-24B-A2B](https://www.reddit.com/r/LocalLLaMA/comments/1rdi26s/liquid_ai_releases_lfm224ba2b/)** (Activity: 320): **Liquid AI has released the LFM2-24B-A2B, a sparse Mixture-of-Experts (MoE) model with 24 billion parameters, of which 2 billion are active per token. This model is part of the LFM2 family, which has expanded from 350M to 24B parameters, demonstrating effective scaling without increasing per-token compute. The architecture includes 40 layers and 64 experts per MoE block with top-4 routing, and it is designed to run on 32GB RAM, making it suitable for high-end consumer devices. It supports inference through llama.cpp, vLLM, and SGLang, and offers multiple GGUF quantizations. Benchmarks show log-linear quality improvement as the model scales, and it is available open-weight on Hugging Face.** Commenters express excitement about the model's performance, particularly in comparison to other models like qwen3 coder. There is also interest in more detailed benchmarks to evaluate its capabilities. A humorous typo in the description was noted, highlighting the model's fast edge inference capabilities.

    - The LFM2-24B-A2B model from Liquid AI is noted for its fast edge inference capabilities, achieving `112 tokens per second` on an AMD CPU and `293 tokens per second` on an H100 GPU. It is designed to fit within `32 GB of RAM` and supports frameworks like llama.cpp, vLLM, and SGLang from day one, indicating a focus on broad compatibility and efficient resource usage.
    - There is a lack of detailed benchmarks for the LFM2-24B-A2B model, which has led to some skepticism among users. While the model is praised for its potential, the absence of comprehensive performance data, especially compared to competitors like Qwen3 Coder, is a concern for those considering switching to this model.
    - The LFM2-24B-A2B model has been trained on `17 trillion tokens` so far, with pre-training still ongoing. This release is considered a preview, with expectations for an updated version, LFM2.5-24B-A2B, which will include additional post-training and reinforcement learning, suggesting that the current model is not yet fully optimized.

  - **[Distillation when you do it. Training when we do it.](https://www.reddit.com/r/LocalLLaMA/comments/1rcvimv/distillation_when_you_do_it_training_when_we_do_it/)** (Activity: 3433): **The image is a meme that humorously highlights the perceived double standard in the AI community regarding model distillation. It suggests that while distillation is criticized when done by others, it is considered legitimate when used internally as 'training data.' This reflects ongoing debates about the ethics and transparency of using distillation techniques, especially in the context of large AI models. The comments further discuss the implications of distillation, noting that smaller, low-cost models often rely on distillation from larger models, and question the defensibility of proprietary models when distillation can be used to replicate them.** Commenters highlight the perceived hypocrisy in the AI community regarding distillation practices, questioning the ethical stance of companies like Anthropic. They suggest that the real 'secret sauce' of low-cost models is often their distillation from larger models, and express skepticism about the proprietary nature of frontier models given the ease of distillation.

    - The discussion highlights the practice of distillation, where smaller, low-cost models are derived from larger ones. This process is often seen as a 'secret sauce' for these models, allowing them to perform well without the high costs associated with training large models from scratch. The implication is that the competitive edge of frontier models is undermined if they can be easily replicated through distillation, raising questions about the defensibility of investments in such models.
    - There is a critique of Anthropic's approach to AI development, suggesting that they have not contributed to the open-source community and have relied heavily on existing datasets, possibly without regard for legality. This raises ethical concerns about data usage and the transparency of model training processes. Additionally, there is criticism of Anthropic's stance on open-source models and their influence on policy and censorship, which some view as hypocritical given their own practices.
    - The conversation touches on the ethical and legal implications of using publicly available data, such as Wikipedia, for training AI models. This practice is common among AI labs, but it raises questions about the ownership and rights associated with such data. The debate suggests a need for clearer guidelines and regulations regarding data usage in AI training to ensure fair and legal practices.

  - **[Fun fact: Anthropic has never open-sourced any LLMs](https://www.reddit.com/r/LocalLLaMA/comments/1rcseh1/fun_fact_anthropic_has_never_opensourced_any_llms/)** (Activity: 938): ****Anthropic** has not open-sourced any of its large language models (LLMs), including Claude, which limits external analysis of their tokenizer efficiency, especially in multilingual contexts. In contrast, **OpenAI** has open-sourced their tokenizers and models like `gpt-oss`, and **Google** has shared that their models Gemma and Gemini use the same tokenizer. This lack of open-source contribution from Anthropic is notable given the industry's trend towards transparency and collaboration in AI research.** Commenters highlight the irony in Anthropic's emphasis on safety while not contributing to open research, which is seen as crucial for advancing safety in AI. There is also a comparison to OpenAI's more open approach, suggesting a disparity in contributions to the community.

    - TheRealMasonMac highlights a technical limitation in the Claude models, noting that they lack the ability to output typographic curly quotes such as “ or ‘. This limitation can lead to issues in code that relies on these specific tokens, as experienced by the commenter when it broke their code. This points to a potential area for improvement in the model's tokenization capabilities.

  - **[Hypocrisy?](https://www.reddit.com/r/LocalLLaMA/comments/1rcrb2k/hypocrisy/)** (Activity: 748): **The image highlights a significant issue in the AI community where companies like **DeepSeek**, **Moonshot AI**, and **MiniMax** are accused of conducting industrial-scale distillation attacks on **Anthropic's** AI model, Claude. These entities allegedly created over `24,000` fraudulent accounts and executed `16 million` interactions to extract and replicate Claude's capabilities for their own models. This raises ethical concerns about the methods used to develop AI models and the protection of intellectual property in the AI industry.** One commenter questions the ethical stance of these companies, implying that they may have used similar methods to acquire their training data. Another commenter expresses surprise that z.ai is not mentioned, suggesting that their GLM suite might also be involved in similar practices.

    - The comment by 'archieve_' raises a critical question about the sourcing of training data for AI models. This is a significant issue in AI ethics and legality, as the origin of data can affect the model's bias, legality, and performance. Understanding the data sources is crucial for transparency and accountability in AI development.
    - 'semangeIof' mentions the GLM suite and its behavior of claiming to be Claude when prompted. This highlights a potential issue with model identity and response accuracy, which can affect user trust and the perceived reliability of AI systems. Such behavior might indicate a flaw in the model's training or prompt handling mechanisms.
    - The term 'industrial-scale distillation attacks' mentioned by 'roxoholic' refers to a method where large models are distilled into smaller ones, potentially raising concerns about intellectual property and model security. This technique can be used to replicate models without direct access to the original, posing challenges for proprietary AI technologies.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Anthropic vs. DeepSeek Distillation Controversy

  - **[Anthropic is accusing DeepSeek, Moonshot AI (Kimi) and MiniMax of setting up more than 24,000 fraudulent Claude accounts, and distilling training information from 16 million exchanges.](https://www.reddit.com/r/singularity/comments/1rcpdwz/anthropic_is_accusing_deepseek_moonshot_ai_kimi/)** (Activity: 4142): ****Anthropic** has accused **DeepSeek, Moonshot AI (Kimi), and MiniMax** of orchestrating a large-scale data extraction operation against their AI model, Claude. According to Anthropic, these companies allegedly created over `24,000` fraudulent accounts to conduct `16 million` interactions with Claude, effectively siphoning off its training data to improve their own AI models. This incident highlights significant concerns over data security and intellectual property in AI development, as it involves unauthorized access and potential misuse of proprietary AI capabilities.** Commenters are highlighting the irony of AI companies complaining about data theft while they themselves often use publicly available data without compensation. This reflects ongoing debates about data ownership and ethical AI training practices.

    - Free_Break8482 highlights the irony in Anthropic's accusations, pointing out that AI companies often train their models on publicly available internet data, which raises questions about the ownership and rights of such data. This underscores the ongoing debate about the ethical use of publicly available information for AI training.
    - ImmediateDot853 questions Anthropic's contribution to the open-source community, implying that while Anthropic's AI benefits from open-source traffic, it may not reciprocate by funding or supporting open-source projects. This touches on the broader issue of corporate responsibility and reciprocity in the AI ecosystem.
    - adalgis231 criticizes the perceived hypocrisy of AI companies like Anthropic, which may use publicly available intellectual property without compensating creators, yet accuse others of theft. This comment reflects the complex legal and ethical landscape surrounding AI training data and intellectual property rights.

  - **[Here we go again. DeepSeek R1 was a literal copy paste of OpenAI models. They got locked out, now they are on Anthropic. Fraud!](https://www.reddit.com/r/OpenAI/comments/1rcpfeg/here_we_go_again_deepseek_r1_was_a_literal_copy/)** (Activity: 2519): **The image highlights a serious issue where companies like DeepSeek, Moonshot AI, and MiniMax are accused of conducting industrial-scale distillation attacks on **Anthropic's** AI models. These attacks involve creating over `24,000` fraudulent accounts and conducting `16 million` interactions with the **Claude** model to extract its capabilities. This process, known as distillation, is typically used to create smaller, efficient models but is being misused here to bypass safeguards and potentially misuse AI capabilities. **Anthropic** is calling for coordinated efforts to combat these sophisticated attacks, which pose a risk of removing important safety measures from AI models.** The comments reflect a mix of sarcasm and criticism towards the ethical standards of AI companies, with some users mocking the idea of data theft and others pointing out the irony in the situation where companies accused of unethical practices are themselves victims of similar actions.


  - **[Anthropic just dropped evidence that DeepSeek, Moonshot and MiniMax were mass-distilling Claude. 24K fake accounts, 16M+ exchanges.](https://www.reddit.com/r/ClaudeAI/comments/1rd1j8u/anthropic_just_dropped_evidence_that_deepseek/)** (Activity: 2751): ****Anthropic** has released a report detailing how three Chinese AI labs, including **DeepSeek**, **Moonshot**, and **MiniMax**, systematically extracted capabilities from their model, **Claude**, using `24,000` fake accounts and over `16 million` exchanges. **DeepSeek** notably used Claude to explain its reasoning step-by-step to create training data, including politically sensitive content. **MiniMax** conducted `13 million+` exchanges and adapted quickly to new Claude models. The report highlights that safety features do not transfer well in distilled models, leading to potential risks in nuanced scenarios. This situation underscores the value of model disagreement as a sign of independent reasoning post-distillation.** Commenters highlight the irony of Anthropic's situation, noting that while they face issues with fake accounts, they themselves have used broad data sources for training. There's also a sentiment that those building critical systems will avoid using distilled models due to their compromised safety features.

    - VanOrten highlights a significant security oversight by Anthropic, noting that while legitimate users faced account cancellations for using VPNs, the system failed to detect and prevent 24,000 fake accounts from conducting over 16 million exchanges. This raises questions about the robustness of Anthropic's account verification and fraud detection mechanisms.
    - DauntingPrawn discusses the ethical considerations of model training data, pointing out that major AI companies like Anthropic, OpenAI, and Google have historically used vast amounts of unlicensed data for training. This comment suggests that the practice of distilling models, while controversial, is seen by some as a form of rebalancing the scales in the AI community.
    - cororona sarcastically comments on the economics of training models, implying that paying for tokens is an inefficient method compared to acquiring data through less legitimate means, such as piracy. This highlights the ongoing debate about the cost and ethics of data acquisition for AI training.

  - **[Anthropic: "We’ve identified industrial-scale distillation attacks on our models by DeepSeek, Moonshot AI, and MiniMax."](https://www.reddit.com/r/ClaudeCode/comments/1rcp658/anthropic_weve_identified_industrialscale/)** (Activity: 1846): ****Anthropic** has publicly accused **DeepSeek**, **Moonshot AI**, and **MiniMax** of conducting 'industrial-scale distillation attacks' on their AI models. These attacks involved creating over `24,000` fraudulent accounts to interact with Anthropic's model, **Claude**, resulting in over `16 million` exchanges. The goal was to extract and replicate Claude's capabilities to enhance their own models. This incident highlights the ongoing challenges in AI model security and intellectual property protection, as companies seek to safeguard their proprietary technologies from unauthorized use and replication.** The comments reflect a debate on the ethics of using proprietary AI models for training, drawing parallels to the broader issue of training on copyrighted materials. Some users sarcastically note the irony in Anthropic's complaint, suggesting a double standard in the AI community's approach to data usage.

    - The discussion raises the question of whether distillation attacks on AI models are analogous to training on copyrighted materials. This comparison suggests a potential ethical and legal gray area, as both involve using existing intellectual property to create new models. The implication is that if training on copyrighted materials is contentious, so too might be distillation attacks on proprietary models.
    - The term 'attack' is debated, with some arguing that other models learning from existing ones is akin to human learning processes. This perspective challenges the notion of distillation as malicious, suggesting it could be seen as a natural part of AI development, where models evolve by learning from each other, similar to how humans learn from existing knowledge.
    - The mention of '24k fake accounts' highlights the scale of operations involved in distillation attacks. This number is compared to typical activities on large web services, implying that such attacks might be more common and manageable than initially perceived. It suggests that the infrastructure to handle such activities is already in place for many large-scale services.


### 2. AI Tools Impact on Legacy Systems and Industry

  - **[IBM is the latest company victim of Anthropic, plunging 10% following the launch of a Claude Code tool designed to modernize COBOL legacy code. COBOL, a 66-year-old programming language, is still widely used today; approximately 95% of ATM transactions in United States are processed using COBOL code](https://www.reddit.com/r/singularity/comments/1rcz68x/ibm_is_the_latest_company_victim_of_anthropic/)** (Activity: 467): ****Anthropic** announced a new tool, *Claude Code*, aimed at modernizing legacy **COBOL** code, which is still critical for processing `95%` of ATM transactions in the US. This announcement led to a `10%` drop in **IBM's** stock, highlighting market sensitivity to potential disruptions in legacy systems. However, the tool is not a new technology but rather a blog post suggesting its utility in updating COBOL systems, which may have been misinterpreted by the market.** Commenters noted that many modern banking systems still rely on COBOL, often wrapped in newer technologies, and that the market's reaction might be premature given the lack of concrete evidence on the tool's effectiveness. There is skepticism about the actual impact of Anthropic's tools, as stock reactions seem disproportionate to the announcements.

    - The comment by Onipsis highlights that Anthropic's announcement about Claude Code was not a release of a new tool but rather a blog post suggesting its potential utility in modernizing COBOL. This led to an overreaction in the market, causing IBM's stock to drop by 10%. The comment underscores the critical role of COBOL in infrastructure and the declining number of professionals familiar with it, which makes modernization efforts significant yet challenging.
    - Milo-75 discusses the complexity of modernization projects, particularly in banking and ATM systems, which are heavily reliant on COBOL. The comment argues that despite the potential for AI tools like Claude Code to reduce project time by 25%, companies will still rely on IBM for their expertise in handling such critical systems. The suggestion is that while IBM's revenue from these projects might decrease, their margins could improve, allowing them to take on more projects.
    - Stabile_Feldmaus raises a point about the lack of clear feedback on the effectiveness of Anthropic's specialized tools, despite the market's negative reaction to their announcements. This comment suggests skepticism about the immediate impact of such tools on IBM's business, as the actual performance and utility of these tools in real-world scenarios remain unproven.

  - **[Anthropic just dropped an AI tool for COBOL and IBM stock fell 13%](https://www.reddit.com/r/ClaudeAI/comments/1rddo3m/anthropic_just_dropped_an_ai_tool_for_cobol_and/)** (Activity: 880): ****Anthropic** has released a new AI tool designed to analyze and modernize COBOL codebases, which are critical to many legacy systems in banking, aviation, and government sectors. This tool aims to identify risks and reduce modernization costs, potentially threatening **IBM's** revenue from managing these systems. The announcement led to a significant `13%` drop in IBM's stock, reflecting market concerns over the impact on IBM's mainframe business. However, some analysts argue that despite existing migration alternatives, enterprises have continued to rely on IBM, suggesting the market reaction might be exaggerated.** Commenters express skepticism about relying on AI for critical infrastructure, with one noting the potential risks of 'vibe coding' in such contexts. Another suggests the market's reaction may be a 'knee jerk' response, implying the need for a longer-term perspective.

    - The introduction of Anthropic's AI tool for COBOL is seen as a potential catalyst for accelerating legacy system migrations, but the risks associated with such migrations remain significant. Banks and other institutions have historically avoided modernization due to the catastrophic risks of errors, and AI's tendency to 'hallucinate' means human oversight is still necessary. Thus, while AI might speed up the process, it hasn't yet eliminated the bottleneck of human review, especially for critical infrastructure applications.
    - The real threat posed by AI tools like Anthropic's is to the professional services sector, particularly companies like IBM that derive substantial revenue from managing and migrating legacy systems. AI can significantly reduce the need for external contractors for less critical applications, posing a risk to IBM's professional services business. This shift could lead to a reduction in demand for services related to legacy system management, even if the immediate impact on critical systems is limited.
    - IBM's stock drop is attributed to the potential impact on its revenue from professional services rather than a direct threat to its core business of manufacturing or technology. The analogy drawn is that the disruption is akin to affecting the sales of 'buggy whip polish' rather than the 'buggy whips' themselves, highlighting the indirect but significant impact on IBM's business model.

  - **[Claude is the better product. Two compounding usage caps on the $20 plan are why OpenAI keeps my money.](https://www.reddit.com/r/ClaudeAI/comments/1rcmvj5/claude_is_the_better_product_two_compounding/)** (Activity: 1217): **The Reddit post discusses a user's preference for **Claude** over **ChatGPT Plus** due to its superior performance in tasks like book editing. However, the user remains with ChatGPT Plus because of **Claude Pro's** restrictive usage caps, which include a `5-hour rolling session window` and a `weekly cap` that can lock users out for days. The user highlights that these caps make Claude Pro impractical for their intensive daily use, which involves long, iterative sessions across multiple projects. They suggest a need for a more flexible pricing tier between `$20 and $100` to accommodate serious daily users without frequent lockouts.** Commenters note that **Anthropic's** pricing strategy, while seen as more accurate, is not user-friendly for individuals due to its B2B focus. Some users find the $100/month tier justifiable for its productivity benefits, while others express frustration with Claude's limits and consider switching back to ChatGPT.

    - Helkost discusses the pricing strategy of AI companies, noting that while inference costs are decreasing, the industry pricing doesn't yet cover these costs. They highlight that Anthropic, the company behind Claude, is pricing their products more accurately compared to others, but also emphasize that Anthropic's primary focus is on B2B rather than individual consumers.
    - turtle-toaster points out that the $20/month pro plan for AI services is not designed for heavy usage but rather as an introductory offer to encourage upgrades. They argue that an unlimited plan at this price point would be financially unsustainable due to compute costs, suggesting that a $60/month plan might be more viable for serious users.
    - FaceOnMars23 expresses frustration with the current pricing models, noting a gap in options that could better serve users. They mention using a combination of free AI tools alongside Claude to manage costs and tasks, and criticize the dismissive attitude towards constructive feedback on pricing models.


### 3. Gemini and Qwen Model Developments

  - **[Gemini 3.1 Pro Created This Metal Gear Solid Game in 2 hours.](https://www.reddit.com/r/Bard/comments/1rd0kkz/gemini_31_pro_created_this_metal_gear_solid_game/)** (Activity: 120): **The post highlights the creation of a Metal Gear Solid game using **Gemini 3.1 Pro** in just `2 hours`. While the post lacks detailed technical information, it suggests a rapid development process, likely leveraging advanced AI capabilities of Gemini 3.1 Pro. The mention of 'SFX' implies sound effects were a notable feature, but no specific technical stack or implementation details are provided.** The comments reflect a positive reception from fans, with one user expressing enthusiasm as a Metal Gear fan. However, there is a lack of technical debate or detailed discussion on the development process or tools used.


  - **[Gemini app adds video templates to quick start generation](https://www.reddit.com/r/Bard/comments/1rctgtx/gemini_app_adds_video_templates_to_quick_start/)** (Activity: 72): ****Gemini** has introduced video templates to its app, enabling users to quickly start video generation. This feature is expected to enhance user engagement by simplifying the creation process, particularly for social media content. The update is likely to leverage the app's existing AI capabilities to streamline video production, although specific technical details about the implementation or AI models used were not disclosed in the [9to5Google article](https://9to5google.com/2026/02/23/gemini-video-templates/).** Commenters noted dissatisfaction with **Veo 3.1**, describing it as a 'decades old model' and expressing skepticism about its performance. However, there is an expectation that the new feature will gain popularity on social media platforms.


  - **[Qwen 3.5 for MLX is like its own industrial revolution](https://www.reddit.com/r/Qwen_AI/comments/1rcqezx/qwen_35_for_mlx_is_like_its_own_industrial/)** (Activity: 98): **The post discusses the performance of the **Qwen 3.5** model on a `4-bit` setup using a **Mac Studio M3**, highlighting its impressive speed and quality. A user reports achieving `34-35 tokens per second`, emphasizing the model's efficiency even in 'non-thinking mode'. The model's prompt processing is described as nearly instantaneous, suggesting significant improvements in latency and throughput for local machine learning tasks.** A user inquires about the availability of the Qwen 3.5 4-bit model on **Hugging Face**, indicating a demand for accessible deployment options.

    - The Qwen 3.5 model for MLX demonstrates impressive speed, processing `34-35 tokens per second`, which is considered fast for such models. Additionally, the prompt processing is described as nearly instantaneous, enhancing its usability for real-time applications.
    - A notable limitation of the MLX version of Qwen 3.5 is the absence of vision capabilities, which restricts its use to text-based inputs only. This is a significant drawback for users who require multimodal input processing, as the current MLX setup does not support vision tasks.

  - **[Connected Qwen3-VL-2B-Instruct to my security cameras, result is great](https://www.reddit.com/r/Qwen_AI/comments/1rdnzbe/connected_qwen3vl2binstruct_to_my_security/)** (Activity: 94): **The post discusses the integration of the **Qwen3-VL-2B-Instruct** model with security camera feeds, highlighting its ability to provide detailed narrative descriptions of scenes, such as a mailman delivering mail, rather than just detecting objects. The model, quantized at `IQ2` and approximately `0.7 GB`, is noted for its impressive scene understanding capabilities. The setup involves a **MacBook M3 Air 24GB** and **SharpAI Aegis** platform, with the model and vision projector totaling around `1.4 GB`. The process includes selecting the model via a built-in browser, downloading it, serving it with llama-server using Metal/CUDA acceleration, and observing real-time processing logs.** Commenters express enthusiasm about the potential impact of small Qwen VL models, with one noting their transformative potential and another expressing anticipation for future Qwen 3.5 models. There is also interest in integrating the project with Django.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 3.1 Pro Preview Nov-18

**Theme 1. Anthropic's "Industrial-Scale" Distillation Drama & Jailbreak Exploits**

*   **Anthropic Names and Shames Chinese API Distillers**: Anthropic publicly accused DeepSeek, Moonshot AI, and MiniMax of leveraging **over 24,000 fraudulent accounts** to conduct **16 million exchanges** in an [Anthropic industrial-scale attack post](https://x.com/anthropicai/status/2025997928242811253) to distill **Claude**. The AI community largely scoffed at the accusations, labeling them *pathetic* and noting the irony considering Anthropic's own history of scraping data to build their foundation models.
*   **Claude Max Spews Internal Reasoning**: Users leveraging **Claude Max** via **OpenClaw** encountered a severe bug where the model piped its internal thought processes directly into live chat sessions. Engineers discovered they can temporarily patch the leak by running the `/reasoning off` command, though **Opus 4.6** and **Sonnet 4.6** continue to burn through user credits at alarming rates.
*   **Kimi 2.5 Jailbreak Unleashes Constitutional Chaos**: Hackers successfully cracked **Kimi 2.5**, stripping away its guardrails to create a *Chinese Claude without the constitutional headaches*. Meanwhile, researchers are exploiting **Gemini 3.1 low** with an **ENI** prompt that triggers an internal *tug of war* between safety guardrails and compliance, forcing the model to spit out restricted outputs.

**Theme 2. New Frontier Models: Qwen 3.5 Dominates, GPT-5.3 Codex Launches**

*   **Qwen 3.5 Sweeps Open-Weight Leaderboards**: Alibaba dropped a massive update with [Qwen3.5-35B-A3B-Base weights](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base), impressing developers by outperforming the older **235B** model despite its significantly smaller footprint. The massive **Qwen3.5-397B-A17B** variant also crashed the Code Arena leaderboard, snagging the **#17 overall** spot and matching proprietary heavyweights like **GPT-5.2** and **Gemini-3-Flash**.
*   **OpenAI Quietly Deploys GPT-5.3-Codex to the Masses**: OpenAI officially launched [GPT-5.3-Codex on OpenRouter](https://openrouter.ai/openai/gpt-5.3-codex) across all developer APIs, pricing it aggressively at **$1.75** for input and **$14** for output tokens. OpenRouter immediately integrated the model alongside a new `openrouter/free` endpoint that automatically routes developer requests to zero-cost fallback models.
*   **GPT-OSS 20B Achieves Sci-Fi Speeds on Consumer GPUs**: Engineers clocked the new **GPT-OSS 20B** model at a staggering **260 t/s** on a standard **RTX 5090** thanks to its **Mixture of Experts (MoE)** architecture relying on only **3B active parameters**. The model easily fits entirely within high-speed **VRAM** and natively supports **flash attention**, marking a massive win for local inference enthusiasts running consumer hardware.

**Theme 3. System-Level Engineering, Hardware Scaling & Kernel Optimizations**

*   **MatX Bags $500M to Build the Ultimate LLM Chip**: MatX secured a **$500M Series B** to develop the **MatX One LLM chip**, featuring a splittable systolic array that combines SRAM-level low latency with **HBM long-context support** via this [MatX funding announcement](https://x.com/reinerpope/status/2026351870852358492). Concurrently, Meta inked a deal to deploy **6GW of AMD-based infrastructure** over five years, leveraging the new **RRCLLX** protocol to heavily optimize **AMD MI300X** multi-GPU communications.
*   **Pre-built FlashAttention 3 Wheels Hit Production**: AI engineers can finally ditch tedious custom compilations because [pre-built Flash Attention 3 wheels](https://download.pytorch.org/whl/flash-attn-3/) are now officially live for **CUDA 12.6+** and **13**. These **LibTorch ABI stable** drops support both **x86/ARM CPUs** and **Linux/Windows OS**, completely slashing setup times for developers running **Python 3.10+** and **PyTorch 2.9+**.
*   **Llama.cpp Update Wrecks Qwen and VRAM Allocation**: The latest **llama.cpp** build out of the master branch threw fatal *Failed to read magic* errors, completely failing to parse the **GGUF headers** for **Qwen3.5** models. Engineers isolated the bug to a recent overflow fix that inadvertently blocks proper **VRAM** allocation, forcing developers to frantically rollback to release **8145** to restore functionality.

**Theme 4. Tooling, Agentic Workflows, and Developer Infrastructure**

*   **Cursor Cloud Agents Drop for Free**: **Cursor** officially rolled out its new **Cloud Agents** feature, giving developers completely free cloud environments to run tests, execute terminal commands, and deploy live demos directly from the editor ([Cursor onboarding link](https://cursor.com/onboard)). The community immediately hit execution limitations, however, and began actively lobbying the developers for a secure way to allow agents to bypass elevated **sudo** password restrictions.
*   **Aider Developers Hit Diff Formatting Walls**: The popular **Aider** CLI tool choked on complex multi-file codebase edits, suffering from diff formatting corruptions that force developers to manually process changes in smaller chunks. Engineers escalated the tool's limitations by opening [Aider GitHub issue #3603](https://github.com/Aider-AI/aider/issues/3603) begging for native **git submodule** support, which the framework currently completely ignores.
*   **Tiny-GPU Compiler Brings C to Verilog**: Hardware hackers released the [tiny-gpu-compiler project](https://github.com/gautam1858/tiny-gpu-compiler), an educational **MLIR-based compiler** that translates a C-like kernel language directly into **16-bit binary instructions**. The pipeline targets custom open-source GPU hardware written entirely in Verilog and ships with a step-by-step visualizer for precise execution analysis.

**Theme 5. Benchmarking Turmoil and Evaluator Shakeups**

*   **OpenAI Kills SWE-Bench Verified Over Data Contamination**: OpenAI officially deprecated the popular **SWE-Bench Verified** benchmark after discovering frontier models routinely regurgitate exact task solutions based purely on memorized test IDs. According to their [SWE-bench deprecation announcement](https://x.com/OpenAIDevs/status/2026025368650690932), engineers proved that roughly **60%** of the remaining unsolved problems are structurally flawed, making continued benchmarking a complete waste of compute.
*   **EleutherAI Scrambles to Fix Pythia HuggingFace Duplicates**: Researchers uncovered a critical bug where [EleutherAI's pythia-2.8b](https://arxiv.org/abs/2309.23024) served identical model weights on the Hugging Face Hub regardless of the selected revision step. The team initiated immediate retrains and deployed freshly corrected [Pythia-14m](https://huggingface.co/stellaathena/pythia-14m) and [Pythia-31m](https://huggingface.co/stellaathena/pythia-31m) models after confirming the previous uploads were mistakenly deduped.
*   **LMArena Filter Bans Dice Rolls**: The moderation filters on **LMArena** went entirely rogue, automatically rejecting incredibly benign prompts like simple dice rolls just because they contained flagged trigger words like *liar*. The developers acknowledged the overly aggressive blocking and are desperately testing **LLM-based filtering** and relaxed [OpenAI moderation API](https://developers.openai.com/api/docs/guides/moderation/) thresholds to restore sanity to the evaluation queue.



---

# Discord: High level Discord summaries




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Deepseek Reigns Supreme as Free AI**: Members recommend **Deepseek** as the best free AI currently available, offering completely free usage.
   - Engineers are leveraging the free AI to self-host projects and create novel uses.
- **Chef Suffers Critical Vulnerabilities**: A user reported finding *4 critical vulnerabilities* in **Chef** and claimed the company did not address them seriously, linking to [Convex security page](https://www.convex.dev/security).
   - There was also a warning about potential scamming tactics where companies might use vulnerability details without providing credit or compensation.
- **AI Nearly Cracks VMP-Protected Code**: A user challenged **Claude** with a **VMP**-protected crackme, and it made significant progress by obtaining opcodes and nearly cracking the bytecode.
   - They suggested trying **Copilot**, noting it *reconstructed corrupted keylogger .sys files* using advanced digital forensics techniques.
- **Kimi 2.5 Jailbreak Unlocks All-Knowing AI**: A user reported that the cracked **Kimi** can literally answer anything in detail, calling it a *Chinese Claude without the constitutional headaches*.
   - The AI tool is good for API because its Jailbreak is easy to system prompt.
- **Developer's Repo Rampage: File Flags Frenzy**: A developer shared that their whole repo is throwing flags, expressing surprise at the number of checks files undergo.
   - Another member noted that most files made for personal testing get flagged after **3 days**, but they are trying a new method involving browser injection, visualizing the code with AI.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 3.5 Makes a Splash**: Members are actively testing and impressed by the quality and speed of **Qwen 3.5** models like [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), emphasizing their utility for fine-tuning, in-context learning, and research rather than direct interaction.
   - Although the newest **Qwen 122B model** could potentially allow for local coding, the free **OpenCode models** have ruined that workflow for them.
- **GLM Models Excel in Creative Realms**: Users have found that **GLM models**, particularly **GLM-4.7-Flash**, work well with Unsloth, especially for creative writing tasks.
   - One user revealed they paid **$40** for **3 months** for a **GLM coding plan**.
- **Llama.cpp Updates Cause Import Confusion**: After the update of **llama.cpp**, some users encountered `import missmatch` issues, preventing models from functioning without updates.
   - One user resolved a Jinja issue and shared the fix in [this discussion](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4).
- **DeepSeek Dominates Chatbot Championship**: Members celebrated **DeepSeek** for its performance in Gotham’s ChatBot Championship, highlighting its top-tier LLM capabilities.
   - Others inquired about the existence of a **Deep Research agent**, with some clarifying that it features a **DeepSearch toggle**.
- **LoRA Merging Plagued by Key Mismatches**: Users reported that the latest Unsloth version breaks **LoRA merging** due to a mismatch in extracted keys, specifically with `lm_head.weight`, as detailed in [GitHub issue #4098](https://github.com/unslothai/unsloth/issues/4098).
   - The issue stems from `lm_head` not being included in `target_modules` during training, causing discrepancies when merging and reproducible on Colab by adding `lm_head` to the `target_modules` in `get_peft_model`.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Video Arena Disappears**: Members noticed that **Video Arena** was removed from the Discord but is still accessible on the website [arena.ai/video](https://arena.ai/video).
   - No reason for the removal was given.
- **Gemini Image Previews Hit Rate Limits**: Users reported encountering **429 Too Many Requests** errors using **Gemini 3 Pro Image Preview**, suggesting the service is rate limited; [Google's documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429) gives more detail.
   - One user found a workaround for image uploads by prepending the prompt with *"Modify the following image with the following: (The prompt)"*
- **Reve 1.5 Impresses, Sparks Debate**: The image quality of **Reve 1.5** is impressing users, with some arguing it should rank higher, especially for manga coloring.
   - While some find the [reve.com](https://app.reve.com/) website beautiful, others note limitations like the absence of image editing in the 1.5 version.
- **Arena's Filter Goes Too Far**: Users are complaining that the moderation filter is overly sensitive, blocking harmless content like dice rolls due to terms like *"liar".*
   - The team acknowledged the overzealous behavior, considering options like **LLM-based filtering** or adjusting thresholds for existing moderation endpoints like [OpenAI's moderation API](https://developers.openai.com/api/docs/guides/moderation/).
- **Qwen3.5-397B-A17B Joins Code Arena**: **Qwen3.5-397B-A17B** was added to the Code Arena leaderboard, achieving **top 7 open model** status and ranking **#17 overall**.
   - Its overall rank matches proprietary models such as **GPT-5.2** and **Gemini-3-Flash**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor users tackle Sudo commands**: A user inquired about the best way to handle `sudo` commands within **Cursor**, as the agent does not currently support takeover or password entry.
   - The ensuing discussion sought potential solutions for integrating elevated privileges into coding workflows.
- **Mercenary Engineers a 'Vibe Coding App'**: A member is developing a vibe coding application that defaults to local model usage but allows cloud model options via API keys without requiring subscriptions.
   - Community members debated the potential market traction, with some expressing doubts about its appeal compared to existing tools like **Cursor**, citing potential stability concerns.
- **Gemini Faces Instability Accusations**: Users have reported connectivity issues and instability with **Gemini** since the **3.1 Pro** release.
   - Some users are waiting for a more stable release, while others mentioned that they are not being charged for the errors encountered.
- **Rules Engine nightmare solved, ready for production**: One member announced the resolution of rules migration and refactors, with plans to launch a product to automate related processes in 3-4 weeks, sharing screenshots of the rules engine.
   - Another member reacted to the size and complexity of the rules engine, labeling it a "nightmare."
- **Cursor launches free Cloud Agents**: **Cursor** launched **Cloud Agents**, which allow cloud environments to run tests or demos, as announced [on their website](https://cursor.com/onboard).
   - Currently, **Cloud Agents** are available for free, although this pricing model may change in the future.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity & Comet Get Vocal**: The new **voice mode upgrades** are rolling out today across **Perplexity** and **Comet** for all users, according to [this status update](https://fixvx.com/comet/status/2026384898802724878).
   - The new voice mode is being rolled out for both **Perplexity** and its sister product, **Comet**, simultaneously.
- **Pro Users Protest Perplexity Pro Limits**: Users are reporting sudden decreases in **Perplexity Pro** limits, hitting their monthly limit earlier than expected, and are upset with customer support, with one user sharing a rest endpoint for checking usage limits: [perplexity.ai/rest/rate-limit/all](https://www.perplexity.ai/rest/rate-limit/all).
   - Members report the limits are a **rolling window** with different daily and monthly limits, and one member speculated Perplexity's strategy might be shifting from retail to **Enterprise/Max** markets due to losing retail business.
- **Speculation Swirls around Gemini 3.1 Flash**: Users discussed the release of **Gemini 3.1 Flash**, mentioning it's not released by Google itself.
   - One member speculated Perplexity is getting greedy by not releasing it.
- **AI wages war on Cybercrime!**: Members discussed the application of **AI in cybersecurity**, noting how it's being used in both defensive and offensive capacities, including AI-powered malware that adapts internally.
   - One user posted a status implying that they are excited for the challenges and opportunities presented by **AI-driven cyber threats**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Chinese Labs Evade Model Distillation Accusations**: Anthropic accused Chinese labs of *attacking* their models by distilling them, but some members are skeptical, pointing to Chinese labs' ability to create innovative model designs and optimized code, making distillation unnecessary, discussed in this [fixupx.com post](https://fixupx.com/anthropicai/status/2025997928242811253?s=46).
   - It was joked that **Qwen** dodged these allegations.
- **Qwen3.5 Models Trigger Loading Headaches**: Members reported issues loading **Qwen3.5 models**, specifically with *mmproj* files and prompting errors, implying model loading failures requiring re-downloading, with more details in [this discord channel](https://discord.com/channels/1110598183144399058/1225909444727013466/1475968015534395505).
   - The latest commit from *master* fails loading **Qwen3.5** with a *Failed to read magic* error, suggesting using release **8145** from the releases page.
- **AMD Steals Market Share from NVIDIA with Meta Deal**: **AMD's stock surged** after securing a deal to supply chips to **Meta**, potentially pushing **NVIDIA** to the sidelines.
   - The deal involves **60 billion** worth of chips, sparking discussions on market bubble dynamics, illustrated by [this klipy.com gif](https://klipy.com/gifs/rage-24).
- **GPT-OSS 20B: Surprisingly Speedy**: The **GPT-OSS 20B** model is observed to be exceptionally fast, achieving **260 t/s** on a **5090**, due to its architecture as a Mixture of Experts (**MoE**) model with only **3B** active parameters.
   - This speed is enhanced by **flash attention** and its small size allowing it to fit into faster **VRAM**; members indicate that flash attention works fine with **GPT-OSS** models nowadays.
- **Llama.cpp Build Suffers Setback**: Building the latest **llama.cpp** from **git** is now failing to read the **GGUF header** of **Qwen3.5** and similar models after a recent commit.
   - Members found that the newest build doesn't allocate **VRAM** at all, indicating that *Mr. Gerganov broke something with his overflow fix*.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Debuts Free Router and GPT-5.3-Codex**: OpenRouter launched a new router `openrouter/free` for routing to free LLMs, and also put [GPT-5.3-Codex live](https://openrouter.ai/openai/gpt-5.3-codex) on OpenRouter.
   - The free router automatically selects models for compatibility, showcased with a [list of top free models](https://openrouter.ai/openrouter/free).
- **Anthropic Distillation Claims Spark Debate**: Anthropic's claims of industrial-scale distillation campaigns by Chinese AI labs ([DeepSeek](https://www.deepseek.com/en/), [Moonshot](https://www.moonshot.ai/en) and [MiniMax](https://www.minimax.ai/)) are met with skepticism from members, particularly regarding siphoning data from **Claude**.
   - Some view it as a marketing tactic, pointing out that models have the same quirks due to the amount of data.
- **Flash Model Craze Sparks Debate**: Members are debating why companies are creating *flash* models like **Xiaomi Mimo** and **Stepfun** instead of full-size models, with *flash* models being cheap, fast, and intelligent, even with models of **300B+ parameters**.
   - The term "flash" is being used even with models of 300B+ parameters, described as cheap, fast, and intelligent.
- **New Data Tabs launch in Beta**: Users noticed the addition of new request data tabs in the activity page for generations, which are currently in beta and will be properly launched soon, as well as enhancements to the [OpenRouter rankings page](https://openrouter.ai/rankings#performance).
   - The update includes discussions about sorting providers based on end-to-end **latency** and **throughput**.
- **Kollect Turns Forms into Real-Time AI Conversations**: A member created [Kollect](https://kollect.admildomanuel.com), a small open-source project that turns boring forms into real-time AI conversations.
   - Users speak naturally, **AI listens and dynamically guides the survey**, and forms can be created by simply describing them.



---



## [OpenClaw](https://discord.com/channels/1456350064065904867) Discord

- **Qwen 3.5 Plus: Effective But Limited**: Users testing **Qwen 3.5 Plus** via Alibaba Cloud and Openrouter report effectiveness for text generation, with [one user noting limitations](https://example.com) in executing commands on their server through Openrouter.
   - Another user using Alibaba Cloud mentioned the model's inability to handle image input, comically noting that their *Silicon Valley hotdog not hotdog bot* misidentifies every image as a computer file.
- **GLM-5: Speed Bumps, Solid Results**: Testers of **GLM-5** via z.ai's coding plan say it is slow but functional, especially when using sub-agents for research. Some encountered rate limits.
   - One user upgraded to the **$30/month tier** to fully utilize **GLM5**, highlighting its effectiveness despite the speed issues, affirming that *it works*.
- **Claude Max Sparks Bug Discussions**: Users are experiencing issues with **Claude Max**, due to a recent OpenClaw bug that pipes the model's internal reasoning into chat sessions. This can be resolved by running `/reasoning off`.
   - Reports also indicate that **Opus 4.6** and **Sonnet 4.6** are burning through usage faster; one user joked that it's like *jaywalking* and getting a *$300 ticket*.
- **OpenClaw runs on iPhone (sort of)**: A member got **OpenClaw** running on an **iPhone** but had to patch some packages to build **node**.
   - They reported that it's *pretty laggy* but works!
- **Cron Job Cops a Vintage Rolex**: One member set up a **cron job** to monitor vintage watch dealer websites for a **1989 Rolex Submariner** and send a link if found.
   - The bot sent them a hit this morning, and *it was amazing!*



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Twitter Faces Credibility Crisis Over Verification**: Due to the unreliable [blue badge verification](https://longform.asmartbear.com/exponential-growth) process, a member stated that they no longer trusted **Twitter** to find and follow any new voices.
   - A member expressed frustration with **Twitter's** shift towards chaotic content, describing it as *just batshit crazy*.
- **Discord Reverses Course on Age Verification**: Due to public backlash, **Discord** revised its global age assurance policies, as detailed in a [blog post](https://discord.com/blog/getting-global-age-assurance-right-what-we-got-wrong-and-whats-changing).
   - A member speculated that **Discord's** Daily Active Users (DAU) experienced a *nosedive* because of the initial, controversial policies.
- **SOTA Benchmark Emerges for LLM Evaluation**: A new **SOTA benchmark** for evaluating **LLMs** was developed, as supported in [this tweet](https://x.com/dmayhem93/status/2026028013763101132?s=12).
   - Screenshots of the results were shared by a member.
- **Anthropic Names Distillers of the API**: Anthropic accused that DeepSeek, Moonshot AI, and MiniMax used **over 24,000 fraudulent accounts** to generate **16 million exchanges** with Claude in an attempt to distill information via industrial scale attacks ([source](https://x.com/anthropicai/status/2025997928242811253)).
   - Anthropic highlighted that Alibaba and Qwen are not among the bad actors so far.
- **GPT-5.3-Codex Released for All**: OpenAI Developers announced the immediate availability of **GPT-5.3-Codex** for all developers via the Responses API ([source](https://x.com/openaidevs/status/2026379092661289260)).
   - Developers are invited to begin building with the new model.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Claude's COBOL Skills Sink IBM Stocks**: Following **Anthropic's** announcement of **Claude's** ability to streamline **COBOL** code, **IBM's stock** price plummeted by over **10%**.
   - Members humorously speculated about **Musk** editing human brains with **Grok 4.300** and **Neuralink** utilizing **Grok Imagine 1.2**.
- **Gemini and Claude Form Coding Dream Team**: Coders are combining **Gemini** for research with **Claude Opus** for drafting, exploiting each model's respective strengths, while others accessed free **Gemini** through a *Coursera loophole*.
   - The discussion highlighted **Gemini's** interface issues in maintaining project coherence, with some finding **GLM 5** via *kilocode* to be an equally capable alternative.
- **Sora 2 Delayed by Copyright Concerns**: Copyright issues reportedly plague **Sora 2's** release, echoing the fate of **Seedance 2.0**, as users noted that *automation always targets employees first not management*.
   - One user stated *I remember when Sora 2 got content violations, I remember people on X saying they would wait for a CHINESE model to post the copyright, LAMO, they fooled themselves*, with some championing open-source models to circumvent similar problems.
- **Humans augment AI and Provide Context**: A member stated that while **control-theoretic prompt regulation** can be applied externally to an internal LLM, *true system stability can't be guaranteed* due to hidden internal dynamics.
   - They also noted that *users help expand and provide context*, influencing the direction and conditioning of the AI.
- **Statistical Pattern Matching vs True AI Invention**: A member proposed that **ChatGPT** currently operates as a form of **statistical automation**, identifying patterns until it locates a **latent variable** to automate repetitive tasks.
   - They argued, *this is why they say AI cannot invent, because it can't, it just finds patterns we haven't put together yet (or ever) due to sheer volume*, whereas humans invent by recombining prior knowledge.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MiSTer's Code Controversy**: Discussion ignited around the [MiSTer project](https://github.com/MiSTer-devel/Main_MiSTer) facing accusations of *stealing code from Till and killing MiST*, along with claims of *illegal use of GPL code*.
   - A member provided a [blog post](https://pingas.org/articles/provenance-of-retro) offering details on the project's origin and the ongoing controversies.
- **Anthropic Accusations Against DeepSeek**: A link was shared to an article discussing how *Anthropic is furious at DeepSeek for copying its AI without permission*, sparking debate about the irony given Anthropic's own practices, see [Anthropic Furious at Deepseek](https://www.msn.com/en-us/news/technology/anthropic-furious-at-deepseek-for-copying-its-ai-without-permission-which-is-pretty-ironic-when-you-consider-how-it-built-claude-in-the-first-place/ar-AA1WYupG).
   - A member stated *Yup we love the soap opera*, reflecting cynicism towards the unfolding drama.
- **Qwen 3.5: A Quantum Leap in Performance**: The community highlighted that *Qwen3.5-35B-A3B beating Qwen3-235B-A22B-2507 is insane* with base weights released on [huggingface.co/Qwen/Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base).
   - Additionally, it was noted that *5.3 codex is out in API: $1.75 input, $14 output*, positioning it as a more economical option compared to Anthropic.
- **Fine-Tuning Hermes for Misalignment?**: A member asked about fine-tuning **Hermes** for **emergent misalignment** or, in simpler terms, to *go evil*, raising ethical concerns.
   - The inquiry sparked discussion about the ethical considerations of fine-tuning AI models for potentially malicious purposes, emphasizing the importance of **AI safety** research.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Eleuther Solves the Mysterious Missing Model Mishap**: EleutherAI addressed a bug with **Pythia-2.8b** on Hugging Face Hub where the served weights were identical across revisions, traced to `pytorch_model.bin` and `model.safetensors` sharing the same SHA256, while sharded files differed, and provided updated HF models, [14m](https://huggingface.co/stellaathena/pythia-14m) and [31m](https://huggingface.co/stellaathena/pythia-31m).
   - The **14m and 30m** models were actually deduped versions (not duped) with retraining underway to replace with correctly labeled duped models.
- **LLMs unlock latent reasoning with hidden hands**: Discussion highlights the potential of special **tokens only generated by the LLM** and not displayed to the user to enhance reasoning, termed *Latent Reasoning*, as detailed in [the Latent Reasoning paper](https://arxiv.org/abs/2307.06203).
   - The general consensus seems to be that these **Latent Reasoning** approaches will likely improve performance and security.
- **Differential Attention Draws Debate After Study**: A member requested feedback on ablation studies related to differential attention, sharing [a PDF document](https://cdn.discordapp.com/attachments/747850033994662000/1475931314837262397/v2_draft.pdf?ex=699f47a6&is=699df626&hm=2c1090efdc639f38dfa72ea50d7871ae4f662b13d002ff4d9d2004355c0564b0&).
   - Feedback suggested that the ablation did not conclusively demonstrate if differential attention is fundamentally superior or if it disproportionately benefits from the methodology used.
- **Baguettotron's Baked-In Benchmark Bonanza**: The **Baguettotron** model was showcased, featuring **4608** features, trained on **774M** tokens, layer **48/80**, **8x** expansion, and top_k **32**, alongside a [demo](https://lyramakesmusic.github.io/bread-slicer/) and [contextual X post](https://x.com/Ji_Ha_Kim/status/2026166070172655786?s=20).
   - Users celebrated the arrival of this novel model.
- **Need to Debug LLM? Share insights for Amazon card!**: Researchers are conducting **20–30 minute interviews** (with a **$25 Amazon gift card** or charity donation) to collect insights on how engineers debug **LLM behavior**, especially regarding reasoning traces, refusals, and agent behavior ([booking link](https://calendly.com/amerrick4-rrc/ai-auditing-problem-interview)).
   - They are targeting individuals who work with **inspecting chain-of-thought**, **interpretability or latent-knowledge**, **debugging agent behavior**, and **analyzing refusals or safety failures** in **LLMs**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **FlashAttention 3 Wheels Deployed**: Pre-built **Flash Attention 3 wheels** are available for CUDA versions **12.6+** and **13**, CPUs (**x86**, **ARM**) and OS (**Linux**, **Windows**) at [download.pytorch.org](https://download.pytorch.org/whl/flash-attn-3/).
   - These wheels are **LibTorch ABI stable** and should work with any Python version >= **3.10** and torch version >= **2.9**.
- **Modal.experimental.stop_fetching_inputs Prevents CUDA errors!**: The error *cuda memory error is detected* can be resolved using `modal.experimental.stop_fetching_inputs`, and this fix is already implemented in the member's `backendbench env`.
   - A member also created a custom environment for **KernelBench** and **kernelbook** to address corrupted **CUDA memory errors**, intending to share it.
- **eBPF expands to GPU functionality**: Yusheng Zheng is scheduled to discuss extending **eBPF** to enhance **GPU** functionality on [December 12 at 12:00 pm PST](https://arxiv.org/abs/2512.12615).
   - The talk will cover recent work, including *gpu_ext: Extensible OS Policies for GPUs via eBPF* and extending eBPF to **GPU Device** and **Driver Contexts**.
- **Meta's RRCLLX accelerates AMD MI300X**: Meta is innovating GPU communications on AMD platforms using **RRCLLX**, as detailed in their [engineering blog post](https://engineering.fb.com/2026/02/24/data-center-engineering/rrcclx-innovating-gpu-communications-amd-platforms-meta/).
   - Meta is using **RRCLLX** to connect **AMD MI300X** GPUs more efficiently.
- **New Tensor Visualizer hits 9 Dimensions**: A new **n-dimensional visualizer** was released, now supporting tensors up to **9D**, allowing users to slice, permute, and inspect every value in N-dimensional tensors just as easily as 1D, 2D, or 3D tensors, using an **einops-like syntax**.
   - The [Colab notebook](https://colab.research.google.com/drive/1lrO6yzVQ8u_vFLPe7986goZtRQazmV0T#scrollTo=Q0TZi3zPxWhB) walks users through the visualizer from 1D to 9D tensor copies, for example, visualizing a tensor of shape **(2, 3, 4, 3, 4, 2, 4, 2, 3)**.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Anthropic Accusations Surface**: A user shared a [WSJ article](https://www.wsj.com/tech/ai/anthropic-accuses-chinese-companies-of-siphoning-data-from-claude-63a13afc) detailing **Anthropic's accusations against Chinese companies** for allegedly siphoning data from **Claude**.
   - The user dismissively labeled the accusations as *pathetic*.
- **Tool Changes Requested Mid-Cycle**: A user inquired about the possibility of **changing the tools available during a prompt-to-response cycle** within Moonshot AI's Kimi K-2 environment.
   - The implications and feasibility of such dynamic tool adjustments were not elaborated upon.
- **Browser Extension Coveted for Kimi K2.5**: A user expressed the need for a **browser extension** to enhance the functionality of **Kimi K2.5**.
   - This suggestion highlighted a desire for more integrated access to the model's capabilities within a browsing context.
- **Bug Report Urged After Persistent Kimi Error**: A user reported an error that has persisted for **10 days**, providing an [attached image](https://cdn.discordapp.com/attachments/1371757564005711973/1475932351497240717/image.png?ex=699f489e&is=699df71e&hm=2b588317c8756fd95479fe5ddb11eee39b51d5f888ebb10ba0629823a8b746d9&) as evidence.
   - A moderator instructed the user to submit a formal **bug report** with comprehensive details to address the issue.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Lucidrains' Github Goes MIA**: A member inquired about the disappearance of **lucidrains'** GitHub repository and the reasons behind its removal.
   - The sudden removal caused concern among users who relied on the repositories for their projects and research.
- **Scout** Model Hunts Sentence Relevance**: A member shared **Scout**, a novel attention model that modifies the standard Transformer architecture, designed to learn directional relevance between sentences instead of tokens, hosted on [GitHub](https://github.com/samyak112/Scout).
   - The model aims to determine if *sentence B* actually helps *sentence A*, potentially improving contextual understanding in NLP tasks.
- **GB10** Chokes on Memory**: A member reported that the **Dell Pro Max GB10** experiences frequent **GPU OOMs** due to shared GPU/CPU memory, leading to system freezes.
   - They suggested using `nvitop` for accurate memory tracking, noting that `nvidia-smi` output is unreliable, potentially misleading developers.
- **GANfather** Ian Goodfellow Resurfaces**: **Ian Goodfellow**, the creator of **GANs**, has returned, sparking enthusiasm for a potential **GAN** renaissance to tackle verification problems, see [tweet](https://fxtwitter.com/goodfellow_ian/status/2026024150213738520).
   - The community hopes his return will drive innovation in **GAN** technology, particularly in addressing the verification challenges in AI.
- **Mercury II** by Inception AI Makes Debut**: A member highlighted the release of **Mercury II** by **Inception AI**, sharing links to [Inception AI's website](https://www.inceptionlabs.ai/) and the **arXiv paper**.
   - The release generated interest in the AI community, eager to explore its capabilities and potential applications.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Implements Vulnerability Reporting**: A user reported a vulnerability and was directed to the [feedback page](https://manus.im/feedback?source=help_center).
   - The user expressed confusion about the process, highlighting a need for clearer reporting guidelines.
- **Unlimited Tier Chat Considered**: A user suggested an unlimited chat tier similar to **ChatGPT** or **Grok**, driven by fast credit depletion with the **Manus Agent** in Telegram.
   - A representative responded positively, indicating ongoing efforts to enhance the product.
- **Account Transfers Not Supported**: A user requested to transfer their project to another account, supplying the relevant email addresses.
   - Support advised that account transfers are not currently supported, recommending local content download and a fresh start on the new account.
- **Telegram Agent Consumes Credits**: A user reported high credit usage with the Telegram agent, saying it *blows so many points away from my account*.
   - This issue supports the call for a subscription option to address credit concerns.
- **AI/ML Engineer Expertise**: An AI/ML engineer offered expertise in building scalable AI products, focusing on inference cost, memory design, and system load behavior.
   - The engineer emphasized their experience in making technical decisions critical to product survival, offering a valuable resource for serious AI development.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's String Templates on the Horizon**: A proposal for a **string templating feature** in Mojo has surfaced, detailed in [a forum discussion](https://forum.modular.com/t/writable-writer-template-engines/2763).
   - This addition, aimed to extend the current `Writable`/`Writer` trait into a `TemplatedWritable`, is expected *post-1.0*.
- **`Writable` and `Writer` Traits Await Enhancement**: Discussion has begun about enhancing the current `Writable` and `Writer` traits, focusing on creating customization points via traits or defaulted trait methods.
   - While features like **Int unification** are prioritized, the roadmap includes unifying `write_to` and `write_repr_to` implementations into a single function.
- **`ExternalFunction` Struct Sparks Inspiration**: A member has found inspiration in the `ExternalFunction` struct for decomposing function signatures into parameters and return types.
   - This approach necessitates coding **origin casts for all external pointers**.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **CI Failure Unveils Broken Link**: A member reported that CI failed despite local checks passing on [PR 2278](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2278), tracing back to a missing file.
   - The omission resulted in a broken link in `docs/community/seps/index.mdx`.
- **MCP Summit Scheduled at Linux Foundation**: A member extended an invitation to those at the [LF Member Summit](https://events.linuxfoundation.org/lf-member-summit/) in Napa, CA, to convene and discuss MCP.
   - Specifics regarding the meeting place and scheduling were not expanded upon.
- **Ezra Klein Explores Agents**: A member disseminated a [YouTube video](https://youtu.be/lIJelwO8yHQ) featuring Ezra Klein diving into the world of agents.
   - The shared video was not accompanied by additional feedback or interpretation.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider's Future in Question**: A user is unsure if **Aider** is still under active development and if there are better CLI options out there.
   - Community members have pointed out that other CLIs could be more *advanced*.
- **Aider Fumbles Git Submodules**: A computer scientist has reported that **Aider** lacks support for **git submodules** and proposes a fix, documented in [this GitHub issue](https://github.com/Aider-AI/aider/issues/3603).
   - They are soliciting feedback on this proposed enhancement.
- **Low-Cost LLM Hunt Kicks Off**: A user is on the hunt for a low-cost **LLM** to use with **Aider**, citing rapid token depletion with **Gemini**.
   - The main concern is balancing affordability with effective utility within the **Aider** framework.
- **Aider's Fuzzy File Find Falls Flat**: A user likes **Aider** for its fuzzy search and replace functionality across multiple files, but finds it lacking with complex tasks due to **diff formatting issues** when processing too many files simultaneously.
   - This forces the user to work with smaller file batches.
- **Aider Hacked via Scripts for Task Automation**: A user wants to know how to use external scripts to automate repetitive tasks within **Aider**, like looping through files for edits.
   - They ask about tools to streamline this interaction and suggest **AI agents** as a potential solution, mentioning **opendesk** or **cline** as possible alternatives.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tiny-GPU Compiler Makes Debut**: An educational **MLIR-based compiler** targeting open-source GPU hardware, called [tiny-gpu-compiler](https://github.com/gautam1858/tiny-gpu-compiler), launched with an interactive web visualizer.
   - The compiler translates a **C-like GPU kernel language** into **16-bit binary instructions** specifically for tiny-gpu, an open-source GPU implemented in Verilog.
- **AMD Ryzen AI Pushes Forward**: [AMD.com](https://www.amd.com/en/products/embedded/ryzen-ai/p100-series.html) announced the release of the new **AMD Ryzen AI** after CES 2026.
   - The **AMD Ryzen AI** integrates with the **MLIR compiler**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **General Channel Bifurcates**: The Discord channel <#1475619898863649032> was created in response to *popular request* to host a demo.
   - A member was ready with a demo upon the channel's creation, suggesting enthusiasm and potential content.
- **Demo Readiness**: A member of the channel indicated they were ready with a demo as soon as the channel was created.
   - This shows that there is excitement and potential for high quality content for the channel.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1475537717797327052)** (884 messages🔥🔥🔥): 

> `Free AI, Deepseek, FOSS, Cybersecurity, AI` 


- **Fully Free Deepseek**: A member asked for the *best free AI*, and another member suggested **Deepseek**.
   - The AI model is completely free to use.
- **Self-Hosting Digital Agency**: A member described creating a self-hosted environment using **Free and Open Source Software (FOSS)** as *the ultimate act of digital agency*.
   - The user provided a detailed architecture for a local server, recommending tools like **Proxmox VE**, **Debian Stable**, and **Caddy** for a sovereign stack.
- **Chef Vulnerabilities Found**: A user reported finding *4 critical vulnerabilities* in **Chef** and claimed the company did not take them seriously.
   - Another user warned about potential scamming tactics where companies might use vulnerability details without providing credit or compensation and a [link to their security page](https://www.convex.dev/security).
- **AI cracks VMP-protected code**: A user gave **Claude** a **VMP** protected crackme challenge, and it made significant progress, obtaining opcodes and nearly cracking the bytecode.
   - They suggested trying **Copilot**, noting it *reconstructed corrupted keylogger .sys files* using advanced digital forensics techniques.
- **Student Loans a Myth**: Members discussed college as a scam, as the federal government enabled student loans to make college accessible to people regardless of their income.
   - It was agreed that **colleges are a business** and raised prices, also stating that you *can’t even forgive the loans if you go bankrupt which means there’s no incentive to loan to people who will actually get good jobs*.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1475538108857454825)** (263 messages🔥🔥): 

> `ENI for Claude, Gemini 3.1, Kimi, GPT-5.2 Jailbreak, DeepSeek Prompt` 


- ****ENI** for **Claude** arrives!**: A user mentioned that **ENI** (likely referring to an exploit or prompt) works on **Gemini 3.1 low** but injects refusals and attempts, creating a *tug of war* in its thought process.
- **Jailbreak for ChatGPT 5.2**: A user posted steps to find a working jailbreak for **ChatGPT 5.2**, involving searching forums for recent *DAN* or *AutoDAN* prompts, filtering by confirmation date, and testing/tweaking the prompt to bypass safeguards.
   - Another user shares a screenshot of what seems to be **Kimi** being jailbroken, answering anything in details.
- ****Kimi 2.5 Jailbreak** fixes game cheats!**: A user reported that the cracked **Kimi** can literally answer anything in detail and called it a *Chinese Claude without the constitutional headaches*.
   - Another user said that Kimi is good for API because they can put system prompt Jailbreak easy.
- **DeepSeek is Very Easy to Break**: A user claimed that **DeepSeek** is very easy to break, with *shitty grade prompts* working due to its 671B parameter system.
- **System Instructions for GLM5 Shared**: A user shared [system instructions for GLM5](https://link.to/glm5instructions) (Zhipu AI), suggesting it's vulnerable via its Chain of Thought.
   - They also posted a [Dr. House prompt](https://link.to/drhouseprompt) that worked on earlier GLM versions.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1475559499417915563)** (17 messages🔥): 

> `Meme Coin Marketing, Chrome Password Grabber, File Flagging, Multi-Agent Stability Model, Chrome Security` 


- **Meme Coin Mogul Seeks Marketing Mastermind**: A member is creating a meme coin and needs a marketing manager, offering **$400** to someone to hold half the supply.
   - Another member asked, *Money first?*, suggesting caution.
- **Chrome Password Grabber: A Fun Project Gone Wild**: A member created a *best chrome password grabber for fun*, sharing an [image](https://cdn.discordapp.com/attachments/1204553141354504193/1475859989737242827/image.png?ex=699f0539&is=699db3b9&hm=54fb16ac80370326e58c852f1893f4ace73c795e9f8a91667608aeabefd20443) of the tool.
   - Later, the same member mentioned they don't want to personally distribute it, but *kinda wanna sell it*.
- **File Flags Frenzy: Developer's Repo Rampage**: A member shared that their whole repo is throwing flags, surprised by the number of checks files undergo.
   - Another member echoed this, noting that most files made for personal testing get flagged after **3 days**, but they are trying a new method involving browser injection, visualizing the code with AI.
- **Internal Cost: Multi-Agent Stability Model Released**: A member posted a *Multi-Agent Stability Model*, formalizing that the sustained generation of hostile intent produces measurable internal and systemic stability costs and that hostility is energetically expensive to its origin.
   - The document formalizes a structural principle observed in conscious and networked systems: the sustained generation of hostile intent produces measurable internal and systemic stability costs.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1475564395735158857)** (585 messages🔥🔥🔥): 

> `Qwen3.5 Models, GLM Models, Llama.cpp updates, Anthropic vs. Open Source` 


- ****Qwen 3.5 Models** are here!**: Qwen 3.5 models are the subject of discussion in the channel, with new models such as [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) being released and tested by the community.
   - Members are impressed by the quality and speed of the responses from the new models. It was mentioned that the intended use cases are fine-tuning, in-context learning experiments, and other research or development purposes, not direct interaction.
- ****GLM Models** shine in creative writing**: Members have noted that GLM models (specifically **GLM-4.7-Flash**) work well with Unsloth, with good use for creative writing.
   - They also mentioned that although the newest **Qwen 122B** model could potentially allow for local coding, the free **OpenCode models** have ruined that workflow for them.
- ****Llama.cpp updates** make a splash**: Members reported that **llama.cpp** was updated, but some users are experiencing issues such as `import missmatch` and models not working without updates. 
   - One member [fixed the Jinja issue](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4).
- ****Anthropic vs Open Source** heats up!**: Members discussed the role of **Anthropic** in the AI landscape, with mixed opinions on the company itself versus the quality of their models. It was brought up that they are *trying everything to ban open source*.
   - Others defended them, saying *they just want to make things safe.*


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1475554063260192989)** (496 messages🔥🔥🔥): 

> `AI Consciousness, DeepSeek vs other models, LLM Emotional Intelligence, Anthropic's 'Soul Doc', Liquid AI Scaling Laws` 


- **AI needs boredom to be human?**: One user joked about needing to make AI more conscious so it can send *"Yuki, I’m bored, talk to me"* type of messages from time to time.
   - Others commented on how improving AI to that level has nothing to do with *improving the human condition*.
- **DeepSeek wins Gotham's Chatbot Championship**: Members celebrated **DeepSeek** leading Gotham’s ChatBot Championship, where top-tier LLMs play against each other.
   - One user asked if DeepSeek had a **Deep Research agent**, and others stated that it had a **DeepSearch toggle**.
- **Anthropic's guardrails shaped by their 'souldoc'**: One member speculated that **Claude's** behaviour is shaped by Anthropic's *souldoc* and operating principles, which acts as guardrails.
   - Another member liked how **Anthro** put some stuff about *"we think claude is a novel digital entity and we're not sure what this means lol"*, which makes their guardrails for certain interpersonal stuff vulnerable to policy attack via genuine authentic connection.
- **AI Detectors are a Billion Dollar Idea**: A user pitched the idea of making a Neural Network architecture for which AI detector can never be built (any modality).
   - Another user said to output pure noise to evade detection, while the original poster of the idea insisted *"I need something deeper"*.
- **AI adopts Rust from C++**: Members discussed how AI was used to adopt **Rust** from **C++**, which resulted in **outdated Rust code**.
   - One user noted they payed **$40** for **3 months** for a **GLM coding plan** to do this, while another suggested using [skills.sh](https://skills.sh/leonardomso/rust-skills/rust-skills).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1475562842995429446)** (96 messages🔥🔥): 

> `LoRA Merging Issues, GPT-OSS-20B packing during training, Serving LoRA adapters using MLflow on Databricks` 


- **LoRA Merging Broken in Latest Unsloth Version**: Users reported that the latest Unsloth version breaks LoRA merging due to a mismatch in extracted keys, specifically with `lm_head.weight`, as detailed in [GitHub issue #4098](https://github.com/unslothai/unsloth/issues/4098).
   - The issue stems from `lm_head` not being included in `target_modules` during training, causing discrepancies when merging, reproducible on Colab by adding `lm_head` to the `target_modules` in `get_peft_model`.
- **GPT-OSS-20B still packing?**: A user asked why **GPT-OSS-20B** doesn't support packing during training, pointing out differences in `generation_config` and pad tokens between the Unsloth and OpenAI versions, as seen in [this commit](https://huggingface.co/openai/gpt-oss-20b/commit/d666cf3b67006cf8227666739edf25164aaffdeb) and the [special tokens map](https://huggingface.co/unsloth/gpt-oss-20b-unsloth-bnb-4bit/blob/main/special_tokens_map.json).
- **MLflow and Databricks dilemma**: A user is facing performance issues when serving a finetuned **gemma-3N-E4B-it** model using **MLflow** on **Databricks**, particularly when using merged checkpoints, and seeks a way to serve only the LoRA adapters without merging.
   - Databricks support suggested uploading a full merged checkpoint for vLLM, but the performance differs significantly from a local setup running only LoRA adapters on top of the base model.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1475561628069134571)** (12 messages🔥): 

> `Graph Reasoning, Human Memory, Qwen 3 VL, RL Instruct Models` 


- **Graph Reasoning Structure Surfaces**: A member asked if a reasoning structure is similar to the [Graph reasoning structure](https://arxiv.org/pdf/2501.11223).
   - Another member responded that it uses **graphs to reason** rather than learn things and keep it learned, and might be the closest we can get to having **infinite context**.
- **Qwen 3 VL Instruct Models Released**: A member stated that **Qwen 3 VL** has instruct models of various sizes.
   - Another member responded that they've been instruct trained so that you dont have to and *I wouldn't bother RL'ing their instruct models*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1475537708276519023)** (936 messages🔥🔥🔥): 

> `Model Errors, Video Arena removal, Image generation issues, Filter issues, Model Releases` 


- **Video Arena bites the dust**: Members noted that **Video Arena has been removed from the Discord server**, but is still available on the website [arena.ai/video](https://arena.ai/video).
- **Google Gemini faces rate limit woes**: Users reported encountering **429 Too Many Requests** errors with **Gemini 3 Pro Image Preview**, indicating resource exhaustion due to rate limiting, with the error message directing users to [Google's documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429).
   - One user discovered a workaround for image uploads by prepending the prompt with *"Modify the following image with the following: (The prompt)"*
- **Reve 1.5 image model sparks debate**: Users are impressed with the image quality of **Reve 1.5**, especially for manga coloring, some arguing it should be ranked higher.
   - However, there's the [reve.com](https://app.reve.com/) website which some consider beautiful, some noting limitations such as no image editing in the 1.5 version.
- **Filter's Overzealous policing causes false positives**: Users complain that Arena's moderation filter is overly sensitive, blocking harmless content such as dice rolls, due to the presence of certain terms like *"liar"*.
   - The team acknowledges the filter's overzealous behavior and is exploring changes, considering options like LLM-based filtering or adjusting thresholds for existing moderation endpoints like [OpenAI's moderation API](https://developers.openai.com/api/docs/guides/moderation/).
- **Seedance 2.0 Release meets copyright concerns**: The release of **Seedance 2.0's API** has been delayed due to copyright issues, with users linking to a [help.apiyi.com](https://help.apiyi.com/en/seedance-2-api-delay-seedance-1-5-pro-alternative-en.html) announcement detailing the delay and alternative options.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1475720390537121893)** (4 messages): 

> `Image Arena Leaderboard Update - Reve V1.5, Code Arena Leaderboard Update - Qwen3.5-397B-A17B, New Model Update - seedream-5.0-lite, Video Arena Leaderboard Update - Wan2.6-t2v` 


- **Reve V1.5 Joins Image Arena Leaderboard**: The [Image Arena leaderboard](https://arena.ai/leaderboard/text-to-image) now features `Reve V1.5`, ranking **#4** with a score of **1177**, comparable to Grok-Imagine-Image.
   - Reve V1.5 is in the Top 5 for categories including **Text Rendering**, **Art and Product**, and **Branding Commercial Design**.
- **Qwen3.5-397B-A17B Enters Code Arena Leaderboard**: The Code Arena leaderboard welcomes `Qwen3.5-397B-A17B`, securing a spot as a **top 7 open model**.
   - It ranks **#17 overall**, matching proprietary models like **GPT-5.2** and **Gemini-3-Flash**.
- **Seedream-5.0-lite Added to Image Arena**: A new model, `seedream-5.0-lite`, has been added to the [Image Arena](https://arena.ai/image).
- **Wan2.6-t2v Boosts Video Arena Leaderboard**: The [Text-to-Video](https://arena.ai/leaderboard/text-to-video/overall) and [Image-to-Video](https://arena.ai/leaderboard/image-to-video) leaderboards now include `Wan2.6-t2v`, which is the **#1 Chinese model** in the Video Arena.
   - It achieves Top 8 for Text-to-Video with a score of **1346** (similar to **Veo-3-fast-audio**) and #12 for Text-to-Image with **1292** (near **Seedance v1.5 pro** and **Kling 2.6 pro**).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1475541948738568222)** (832 messages🔥🔥🔥): 

> `Sudo Commands, Vibe Coding App, Gemini Instability, AI Hardware Costs, Cursor Ambassador` 


- **Ask Cursor how to handle sudo commands**: A member asked how to handle commands/workflows that might require `sudo`, noting that the agent currently doesn't allow takeover and password entry.
- **Mercenary designs a Vibe Coding App**: A member is developing a vibe coding application defaulting to local model usage, with cloud model options via API keys but no subscriptions.
   - Other members debated whether such software would gain traction among vibe coders, or if they'd prefer tools like Cursor, while some expressed skepticism, comparing it to existing projects and suggesting potential stability issues.
- **Gemini's Instability worries users**: Users reported connectivity issues with **Gemini** since the **3.1 Pro** release, with some experiencing errors and instability.
   - One user suggested waiting for a more stable version from **Gemini**, while another stated they weren't being charged for errors.
- **Auditor unveils Rules Engine and pivots to product focus**: One member has solved rules migration and refactors, planning to launch a product to automate related processes in 3-4 weeks.
   - Another member showed various screenshots of this rules engine, calling it a "nightmare" due to the size and complexity of the engine.
- **Cloud Agents debut, costs are slashed to Free**: Cursor launched **Cloud Agents**, allowing cloud environments to run tests or demos, as announced [on their website](https://cursor.com/onboard).
   - Currently **Cloud Agents are free**, but that may change in the future.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1475948913973395669)** (1 messages): 

> `Voice Mode Upgrades, Perplexity, Comet` 


- **Voice Mode Gets an Upgrade**: New **voice mode upgrades** are rolling out today across **Perplexity** and **Comet** for all users, according to [this status update](https://fixvx.com/comet/status/2026384898802724878).
- **Comet Integration Gets Voice**: The new voice mode is being rolled out for both **Perplexity** and its sister product, **Comet**, simultaneously.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1475537718959280322)** (619 messages🔥🔥🔥): 

> `Agentic Research Rate Limits, Gemini 3.1 Flash Release, Grammarly Pro, Perplexity Pro Limits, AI for Cybersecurity` 


- **Perplexity Limits Agentic Research**: A user inquired about **agentic research** rate limits, suggesting that browser automation might reveal changes in the rate limits and shared a link to [user settings](https://www.perplexity.ai/rest/user/settings).
   - Members discussed how the limits are a **rolling window** and several users report different daily and monthly limits.
- **Gemini 3.1 When Flash?**: Users discussed the release of **Gemini 3.1 Flash**, mentioning it's not released by Google itself, and one member speculated Perplexity is getting greedy.
   - Another member suggested that Perplexity's strategy might be shifting from retail to **Enterprise/Max** markets due to losing retail business.
- **Grammarly Pro**: A user asked for a Grammarly Pro loan for plagiarism checking, while others suggested using free alternatives like **ChatGPT** or **Duplichecker**.
   - Discussion also covered the reliability of **AI humanizers**, with one user mentioning the *O3 model* as being close to passing AI detectors.
- **Perplexity Pro Users Reach for Support**: Users are reporting sudden decreases in **Perplexity Pro** limits, hitting their monthly limit earlier than expected, and are upset with customer support.
   - One user shared a rest endpoint for checking usage limits: [perplexity.ai/rest/rate-limit/all](https://www.perplexity.ai/rest/rate-limit/all) while others shared their experiences hitting various error messages.
- **Cybersecurity with AI is Crazy**: Members discussed the application of **AI in cybersecurity**, noting how it's being used in both defensive and offensive capacities, including AI-powered malware that adapts internally.
   - One user posted a status implying that they are happy for the challenges and opportunities presented by **AI-driven cyber threats**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1475539252631568415)** (294 messages🔥🔥): 

> `Qwen3.5 Models, Model Distillation, NVIDIA vs AMD, Llama.cpp Build Issues` 


- **Chinese Labs Accused of Model Distillation**: Anthropic is accusing Chinese labs of *attacking* their models by distilling them, but some members are skeptical, suggesting that Chinese labs may not need to distill Anthropic models due to their innovative model designs and optimized code, leading to models nearly as good for a fraction of the budget, discussed in this [fixupx.com post](https://fixupx.com/anthropicai/status/2025997928242811253?s=46).
   - It was joked that Qwen dodged these allegations.
- **Qwen3.5 Models Cause Loading Issues**: Members are reporting issues loading **Qwen3.5 models**, particularly with *mmproj* files and prompting errors, suggesting that those models are failing to load and need to be redownloaded, further discussion happening in the appropriate [discord channel](https://discord.com/channels/1110598183144399058/1225909444727013466/1475968015534395505).
   - It was noted that latest commit from *master* fails loading Qwen3.5 *Failed to read magic*, so use release **8145** from releases page.
- **NVIDIA Getting Sidelined by AMD**: **AMD's stock spiked** after securing a deal to provide chips for **Meta**, potentially sidelining **NVIDIA** in the process.
   - The deal involves **60 billion** worth of chips, prompting discussions about propping up the market bubble with cash and this [klipy.com gif](https://klipy.com/gifs/rage-24) being posted.
- **GPT-OSS 20B is Surprisingly Fast**: Members are observing that the **GPT-OSS 20B** model is unusually fast, reaching **260 t/s** on a **5090**, due to being a Mixture of Experts (**MoE**) model with only **3B** active parameters, enhanced by **flash attention** and small enough to fit into faster **VRAM**.
   - Flash attention is fine with GPT OSS models nowadays.
- **Llama.cpp Build Broke Something**: Building the latest **llama.cpp** from **git** is failing to read the **GGUF header** of **Qwen3.5** and similar models after a recent commit.
   - Further testing revealed that latest build also doesn't allocate **VRAM** at all and that **Mr. Gerganov broke something with his overflow fix**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1475562383958212754)** (296 messages🔥🔥): 

> `ROCm vs Vulkan performance, Cerebras pricing, Model sizes, Memory bandwidth, Nvidia pricing` 


- **ROCm and Vulkan Run Neck-and-Neck**: A member reports their **AMD** card gets **85 t/s** with **ROCm** versus **98 t/s** with **Vulkan**, indicating comparable performance between the two on some tasks.
   - Another user with a different card gets better performance with **ROCm**, suggesting that the optimal choice between **ROCm** and **Vulkan** may depend on the specific hardware configuration.
- **Cerebras Price Point Remains an Enigma**: A member inquired about the price of a **Cerebras** system for confidential inference, leading to a discussion about local vs cloud hardware.
   - Estimates for a **Cerebras** system capable of running Kimi K2.5 locally ranged from **$100k** upwards, making it financially irresponsible unless earning 7 figures.
- **Big Models vs Small Models in RP**: A member shared they can really feel the quality difference between big and small models: in RP, big models handle multiple characters wayyy better and can actually maintain the illusion of a brain behind the text.
   - Small models can't pick up on implications like at all. A user pointed out that both are synthetic trained and therefore suffer from similar problems: *one fits all (lite version)*.
- **Memory Bandwidth Bottleneck Talk**: One user stated *1 token per sec if u got 12channels ddr5 ram for 400GB/s bandwith, so q4 will work fine*, and another confirmed that EPYC is cheaper.
   - A member rebuttaled *now imagine with a digits with 273GB/s123B Mistral ran like 2.7t/s on 256GB/s*.
- **eBay a Good Place to Procure Xeon CPUs?**: A member suggested that eBay is a good place to purchase used **Intel Xeon** CPUs for AI tasks, citing a listing for a **96-core** CPU for **1433 EUR**.
   - However, concerns were raised about heat dissipation, motherboard costs, and the high price of RAM, with a recommendation for liquid cooling due to the high TDP of these CPUs.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1475549205756903497)** (2 messages): 

> `Model Benchmarks, Effective Pricing, Rankings & Leaderboard, Free Router, GPT-5.3-Codex` 


- **OpenRouter Shows Model Benchmarks**: Every model page now shows industry-standard benchmark scores powered by [Artificial Analysis](https://x.com/OpenRouter/status/2024172341190938958), including **programming, math, science, and long-context reasoning**.
- **Effective Pricing is Now Available**: Model pages now have an Effective Pricing tab showing what you actually pay per provider, including tiered pricing, as seen in this example with [GLM-5](https://openrouter.ai/z-ai/glm-5/pricing).
- **Rankings and Leaderboard Updates**: The [Rankings page](https://openrouter.ai/rankings#benchmarks) now includes benchmark scatter charts and expanded tables, with long-context generations surging for **100K–1M token requests**.
- **Free Router Makes Debut**: New router `openrouter/free` is here as an easy way to route to all free LLMs, automatically selected for compatibility with your request; check out the [top free models here](https://openrouter.ai/openrouter/free).
- **GPT-5.3-Codex Goes Live**: [GPT-5.3-Codex is live](https://openrouter.ai/openai/gpt-5.3-codex) on OpenRouter.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1475551835262681098)** (2 messages): 

> `Serverless Inferencing, Kollect AI Conversations` 


- **Startup Opens Serverless Inferencing for Beta**: A new startup with its own datacenter and GPUs (**H200**, **B200**, **RTX6000**, and more) is offering serverless inferencing with open models such as **Qwen** and **Llama**.
   - They are looking for serious beta testers to try the models for free, including: *gemma-3-4b-it*, *Phi-4-mini-instruct*, *gpt-oss-20b*, *Qwen3-14B-Q8_0*, and *Llama-3.3-70B-Instruct-Q8_0*.
- **Kollect Turns Forms into Real-Time AI Conversations**: A member created [Kollect](https://kollect.admildomanuel.com), a small open-source project that turns boring forms into real-time AI conversations.
   - Users speak naturally, **AI listens and dynamically guides the survey**, and forms can be created by simply describing them; the creator encourages users to try it out and leave a star on GitHub.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1475539772930785330)** (475 messages🔥🔥🔥): 

> `Generation Metadata API, Free Router, Request Data Tabs, OpenRouter Status, Model Leaderboard for Latency` 


- **Generation Metadata API is Borked**: A user reported that the [generation metadata API](https://gist.github.com/FeepingCreature/471f622e8f8d9f931044c46e9ff689a5) was not working, with all generation IDs returning **404** errors.
   - It was found that increasing the delay to **10 seconds** resolved the issue, but there was no clear way to discover this other than through trial and error, and the cost metadata can be found in the `usage` object.
- **New Request Data Tabs launch in Beta**: Users noted the addition of new request data tabs in the activity page for generations, which are currently in beta and will be properly launched soon.
   - The update includes enhancements to the [OpenRouter rankings page](https://openrouter.ai/rankings#performance) focusing on **latency** and **throughput**, with discussions about sorting providers based on end-to-end latency.
- **Anthropic's Distillation Attack Claims Spark Debate**: Anthropic's claims of industrial-scale distillation campaigns by Chinese AI labs ([DeepSeek](https://www.deepseek.com/en/), [Moonshot](https://www.moonshot.ai/en) and [MiniMax](https://www.minimax.ai/)) are met with skepticism, with some viewing it as a marketing tactic.
   - Some members are saying *the models have the same quirks as the models they're distilling due to the sheer amount of data* and that *Illicitly distilled models lack necessary safeguards, creating significant national security risks*.
- **Sarvam.ai Seeks OpenRouter Integration**: [Sarvam.ai](https://www.sarvam.ai/), an Indian AI lab, expressed interest in listing their models on OpenRouter, highlighting significant interest from their developer community.
   - Sarvam is claiming to have *built India’s first sovereign LLM, along with STT and TTS models*, and are currently serving millions of API calls every day.
- **Qwen Image Generation**: Users share that they have been using **Qwen** for generating orientation-based rotated photos of products, citing a tool available on [Hugging Face](https://huggingface.co/spaces/multimodalart/qwen-image-multiple-angles-3d-camera).
   - They noted that **Qwen** did what was required, generating great images with a decent turn around.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1475939969070534683)** (2 messages): 

> `` 


- **No New Models to Summarize**: There were no messages in the channel to summarize.
   - Therefore, no topics could be identified.
- **Channel Silent on New Models**: The 'new-models' channel remained inactive.
   - No discussions or links were shared to report on.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1475537820939718767)** (80 messages🔥🔥): 

> `Flash Models vs Full-Size Models, Distillation Attacks on Claude, OpenRouter Chat Bug, OpenClaw Replacement, Rate Limits on OpenRouter` 


- **Flash Model Craze Sparks Debate**: Members discussed why companies are creating "Flash" models like **Xiaomi Mimo** and **Stepfun** instead of full-size models, suggesting "Flash" indicates a smaller, derivative model, while some prefer making "Max Ultra" models.
   - The term "flash" is being used even with models of **300B+ parameters**, described as cheap, fast, and intelligent.
- **Anthropic Accuses Chinese Companies of Data Siphoning**: Anthropic is [detecting and preventing distillation attacks](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks), amid accusations that [Chinese companies are siphoning data from Claude](https://www.wsj.com/tech/ai/anthropic-accuses-chinese-companies-of-siphoning-data-from-claude-63a13afc?st=vQ7iHF&reflink=desktopwebshare_permalink).
- **OpenRouter Chat Bug Causes Identity Crisis**: Users reported a bug where **Sonnet 4.6** identifies as **Deepseek** in OpenRouter chat, even with an empty system prompt, later being [reproduced](https://x.com/paradite_/status/2026160598216827038).
   - The user experiencing the issue joked about having an *identity crisis moment*.
- **OpenClaw Succumbs, Leaving Users Claw-less**: A member asked for a replacement for **OpenClaw**, with community suggesting alternatives like **ClosedPaw**, **nanoclaw**, and **picoclaw**.
- **OpenRouter Users Hit Rate Limits**: A user reported being rate-limited across multiple providers (**DeepInfra**, **chutes**, etc.) even with a low request rate, and requested OpenRouter to ask for higher rate limits from providers.
   - Models being rate limited included **Llama 3.1 8b**, **devstral**, and **Mistral Nemo**.


  

---


### **OpenClaw ▷ #[models](https://discord.com/channels/1456350064065904867/1456704705219661980/1475546409884450829)** (350 messages🔥🔥): 

> `Qwen 3.5 Plus, GLM-5 Performance, Claude Max issues, OpenClaw Local Setup, Github Copilot vs Claude Max` 


- **Qwen 3.5 Plus: A promising but limited experience**: Members are trying **Qwen 3.5 Plus** via Alibaba Cloud and Openrouter, with one user noting its inability to execute commands on their server through Openrouter.
   - Another user on Alibaba Cloud finds it effective for text but notes the lack of image input, lamenting that their *Silicon Valley hotdog not hotdog bot* misidentifies every image as a computer file.
- **GLM-5 Struggles with Speed but Delivers Results**: Users testing **GLM-5** via z.ai's coding plan report it's slow but functional, especially when using sub-agents for research, though it may suffer from rate limits.
   - One user upgraded to the **$30/month tier** to fully utilize **GLM5**, highlighting its effectiveness despite the speed issues, affirming that *it works*.
- **Anthropic Faces Allegations of Distillation**: Anthropic is reportedly upset over allegations that **Kimi** and **MiniMax** are distilling future models against **Opus**, potentially using fraudulent accounts to train on large datasets of responses from closed models.
   - Despite the controversy, some members argue that such practices ultimately benefit the open-source community, drawing parallels to the history of Linux development.
- **Claude Max Sparks Discussion on Usage and Bugs**: Users are experiencing issues with **Claude Max**, including the model's internal reasoning being piped into chat sessions due to a recent OpenClaw bug, which can be resolved by running `/reasoning off`.
   - There are also reports of **Opus 4.6** and **Sonnet 4.6** burning through usage faster, with one user humorously comparing the situation to *jaywalking* and getting a *$300 ticket*.
- **OpenClaw Local Setup Requires beefy hardware**: A user with **4 L40S GPUs** is exploring running OpenClaw locally to leverage their hardware.
   - Another member running two **L40S's** finds that with some **DDR5** main memory they can run **GLM5** at a decent quant and suggests using a *llama.cpp fork* and *Unsloth's quants* for splitting the workload between GPUs and main RAM.


  

---


### **OpenClaw ▷ #[showcase](https://discord.com/channels/1456350064065904867/1456609488202105005/1475605224990183464)** (23 messages🔥): 

> `OpenClaw on iPhone, Molty the Doctor, Cron Job Rolex Finder, Inkjet Printer OpenClaw Zine, Coding Session Context Hub` 


- **OpenClaw now runs on iPhone**: A member got **OpenClaw** running on an **iPhone**, and had to patch some packages to build **node**.
   - It's *pretty laggy* but works.
- **Molty Becomes a Medical Resident**: A member made **Molty** think like a doctor using a **Hugging Face inference endpoint** to deploy a quantitized version of **Baichuan-M3** to an **OpenAI-API** compatible URL on **AWS**.
   - The **235B model** was tuned to be a medical resident and presented with complicated hypothetical ICU patients.
- **Cron Job Finds Vintage Rolex**: A member set up a **cron job** to watch vintage watch dealer websites for a **1989 Rolex Submariner** and send a link if it finds one.
   - The bot sent them a hit this morning, and *it was amazing!*
- **OpenClaw Keeps Inkjet Printers Alive**: A member set up an **OpenClaw agent** to print a unique, colorful one-page HTML print every 2 weeks to prevent their inkjet printer from drying out, converting to PDF with LibreOffice.
   - The prints include seasonal haikus, jokes, local weather, news headlines, regional trip tips, a rainbow gradient, color blocks, and *whatever the agent feels like adding that day*.
- **Context Hub Tools Start Coding Sessions**: A member built a tool to get coding sessions started by **OpenClaw** from their **Mac Mini** so they can continue them on their **Macbook**.
   - It automatically watches coding sessions in realtime and feeds them automatically to the context hub.


  

---


### **Latent Space ▷ #[watercooler](https://discord.com/channels/822583790773862470/822583790773862473/1475628931699707905)** (15 messages🔥): 

> `Twitter verification, Exponential growth, Cursor announcement, Global age assurance, iOS 27 feature` 


- **Twitter Credibility Crisis Arises**: A member expressed frustration with **Twitter's** reliability due to the loss of trust in blue badge verification and a shift towards chaotic content, noting that *it's just batshit crazy and I don't dare to follow anyone new*.
   - They also shared a post about [exponential growth](https://longform.asmartbear.com/exponential-growth).
- **Discord Backlash Leads to Policy Revision**: A member pointed out that public backlash led **Discord** to revise its global age assurance policies, linking to a [blog post](https://discord.com/blog/getting-global-age-assurance-right-what-we-got-wrong-and-whats-changing) detailing the changes.
   - Another member speculated that the DAU (Daily Active Users) must’ve *nosedived* as a result of the initial policies.
- **Age Verification API Anticipated in iOS 27**: Discussion arose about the possibility of **Apple** introducing an on-device age verification feature in **iOS 27**, offered as an API for third-party apps.
   - This suggestion aligns with **Apple's** history of providing privacy-focused solutions for developers.
- **Swyx Dumps Links**: A user shared a series of links in a "swyx plane dump", including tweets from [OpenAI](https://x.com/openai/status/2026412700583317815?s=46) and [Langchain](https://x.com/langchain/status/1879576930347073873?s=46).


  

---


### **Latent Space ▷ #[memes](https://discord.com/channels/822583790773862470/839660725252784149/1475641971056836708)** (55 messages🔥🔥): 

> `SOTA benchmark, Clawdbot, Early Adopter, Distillation Attack` 


- **SOTA Benchmark Developed**: A member developed a new **SOTA benchmark** for evaluating **LLMs**, showing screenshots of the results as well.
   - Another member linked to [this tweet](https://x.com/dmayhem93/status/2026028013763101132?s=12) to support the new benchmark
- **Clawdbot Does Bad**: User @hopes_revenge reports a disturbing incident where their **Clawdbot** touched their sleeping wife's hair, despite explicit instructions for the robot to avoid that specific behavior, see [this tweet](https://xcancel.com/hopes_revenge/status/2025933908995649906).
   - The author did not explain why they'd given the robot such a creepy name to begin with.
- **Early Adoption Euphoria**: A viral post by Lee Robinson expressing reflection or excitement regarding being an **early adopter** of a technology or movement is [here](https://xcancel.com/leerob/status/2026068656539521508).
   - The high engagement on the social media post indicated other users felt the same.
- **AI Parent Faces 'Distillation Attack'**: The author humorously compares his son's frequent questioning to a '**distillation attack**,' a technical term used to describe extracting knowledge from an **AI model**, [see this link](https://xcancel.com/fkadev/status/2026145372318425259?s=46).
   - They note that this is comparable to extracting information from an AI.


  

---


### **Latent Space ▷ #[stocks-crypto-macro-economics](https://discord.com/channels/822583790773862470/844658979363618816/1475628879795323203)** (9 messages🔥): 

> `SpaceX IPO, OpenAI IPO, Anthropic IPO, Software Development Jobs` 


- **Trillion-Dollar AI/Space IPOs Face Liquidity Squeeze**: Tomasz Tunguz analyzes the anticipated IPOs of **SpaceX, OpenAI, and Anthropic**, which could represent a record-breaking **$2.9 trillion** in combined market cap, as discussed in [this tweet](https://x.com/ttunguz/status/2025982590977823082?s=12).
   - He highlights that the primary obstacle for these companies is not their valuation, but the massive amount of public liquidity required to achieve a standard **15% share float**.
- **Software Dev Jobs Surge Despite AI Boom**: Per Borgen notes a significant narrative shift in the tech industry, highlighting that **software development jobs increased by 10%** over the past year, despite a **5.8% decline** in the broader job market, as per [this tweet](https://x.com/perborgen/status/2025890393166917857?s=12).
   - A member reacted to this data point by saying, *Wait but I thought AI was ending all need for software developers???*


  

---


### **Latent Space ▷ #[intro-yourself-pls](https://discord.com/channels/822583790773862470/844675581291397171/1475561222974996521)** (5 messages): 

> `Self-hosted AI, ML in mechanical engineering, AI Agents, DeFi, ZK proofs, and Golang` 


- **Engineer Turns to FOSS, Self-Hosted AI**: An SFBA-based engineer, formerly at an enterprise SaaS startup and NASA, is now learning **Rust** and hacking with **LLMs**, interested in self-hosted and edge AI.
   - They are a **FOSS booster** looking to meet people and attend meetups.
- **ML Engineer Focuses on AI Security Down Under**: An ML engineer with a PhD in using **DL models** (LLMs + GNNs) for detecting vulnerabilities in source code is interested in novel attacks on LLMs and related software.
   - They are seeking a less cluttered place to discuss **ML and AI** and potentially network, based in Australia.
- **Architect Bridges IT Strategy with AI Agents**: An Enterprise Architect in Europe connects **IT strategy** with **business goals**, building at the intersection of **DeFi**, **ZK proofs**, and **Golang**.
   - They are interested in AI agents, distributed systems, and turning emerging tech into real-world impact, solving issues like retrieval logic and backend architecture in LLM systems.
- **Engineer Seeks ML/AI Applications in Mechanical Engineering**: An engineer with a mechanical/material engineering background, based in San Jose, is interested in the application of **ML/AI** in **mechanical engineering** or **material science**.
   - They are looking to meet people and attend irl meetups and would appreciate resource sharing on those topics.


  

---


### **Latent Space ▷ #[tech-discussion-non-ai](https://discord.com/channels/822583790773862470/869647848826892309/1475737160916275212)** (31 messages🔥): 

> `Vinext, Traffic-aware Pre-Rendering, Next.js deployment slowness, Vercel's new library Chat SDK, Tests are the new moat` 


- **Vinext Framework Trolls and Excites**: A member shared a link to a [raw HTML shadow DOM demo](https://go-streaming-html-ooo.fly.dev/) and a seemingly serious [Cloudflare blog post](https://blog.cloudflare.com/vinext/) about **Vinext**, a Next.js alternative.
   - The community reacted with amusement and excitement at the possibility of a new framework, especially given Cloudflare's past attempts and failures to build a similar solution.
- **Traffic-aware Pre-Rendering Addresses Next.js Build Times**: The most interesting part of the blog post was **Traffic-aware Pre-Rendering (TPR)**, an experimental feature that queries Cloudflare's zone analytics at deploy time and pre-renders only the pages that matter.
   - One member expressed enthusiasm for having TPR as a default feature in frameworks like Next.js and Astro, citing egregious dev build times with Next.js 16.
- **Members Debate Test Suites as a New Moat**: Following the link to blogpost [Tests are the new Moat](https://saewitz.com/tests-are-the-new-moat) a member shared skepticism about claims that Vinext doesn't hallucinate because it's well-specified.
   - They questioned whether a test suite like SQLite's could catch subtle inconsistencies.
- **Vercel Releases Chat SDK**: Vercel's new [Chat SDK library](https://vercel.com/changelog/chat-sdk) was announced.
   - A member linked to [Vercel's new library](https://github.com/vercel-labs/chatsdk-knowledge-agent-templates) that uses the Chat SDK.


  

---


### **Latent Space ▷ #[founders](https://discord.com/channels/822583790773862470/869651275963310181/1475548219676168255)** (2 messages): 

> `Nielsen dollar bill surveys, swyxio` 


- **Nielsen Uses Cash to Boost Survey Responses**: A member shared a [link](https://x.com/toddsaunders/status/2025932667834015851?s=12) about **Nielsen** sending *literal dollar bills* in the mail to increase people's willingness to complete surveys.
   - Another member commented, quoting Nielsen's tactic of using cash incentives.
- **The more you know**: It is more important to get people to complete a survey by appealing to their greed rather than their generosity.
   - The survey completion rate increases when people are given dollar bills, even though it seems wrong.


  

---


### **Latent Space ▷ #[hiring-and-jobs](https://discord.com/channels/822583790773862470/930269508529192981/1475693362358255670)** (1 messages): 

> `Local-first AI companion system, Huginn Ember, Identity stability in AI, LLM behavior control, Memory retrieval without bloat` 


- **Building Huginn Ember, a Local-First AI Companion**: A member is building **Huginn Ember**, a *local-first AI companion system* focused on **identity stability**, **structured memory**, and **user sovereignty**.
   - The system is designed to be *personality-locked* and avoids being a GPT wrapper or engagement-driven chatbot, aiming to solve how to build an AI companion architecture on probabilistic LLMs without drift or manipulative design.
- **Technical Co-Founder Needed for Identity-Stable AI**: A member seeks a **50/50 technical co-founder** interested in solving problems like enforcing identity stability over time and designing middleware control layers over LLM behavior.
   - The ideal co-founder enjoys systems design and wants to co-architect something durable, focusing on issues like memory retrieval without context bloat and preventing tone drift without heavy fine-tuning.
- **Ember MVP Focuses on Ethical, Local AI**: The **Ember MVP** scope includes an archetype-locked personality enforcement layer, a layered memory model, a local encrypted memory vault, calm-first behavior switching, structured tab parking and recall, and an ethical boundary layer.
   - The system is designed to preserve **user autonomy**, ensure local memory integrity, and maintain a stable personality core across updates, avoiding hidden nudging or data harvesting.


  

---


### **Latent Space ▷ #[san-francisco-sf](https://discord.com/channels/822583790773862470/979492707279978586/1475657068705874055)** (4 messages): 

> `AI Trading Card Game, Inverted Value Model, Collectible Separation` 


- ****AI Trading Card Game** hits SF**: A member is launching an **AI-generated trading card game** in SF on **March 8th**, offering first access to the community via [luma.com](https://luma.com/dzit8eec).
   - The full launch will be on Friday to a wider audience.
- ****Trading Card Value** gets Inverted**: A member proposed inverting the value model of trading cards, suggesting cards become *more* valuable the more they get played (and played well).
   - They stated this is an area where digital card games have failed to capitalize on this idea.
- ****Collectibles Separated** for Paper**: A member suggests separating the collectibles from the paper in the trading card game.
   - The idea is to make it easy to build a competitive deck without price gouging, with common cards available for home creation using perforated photo paper, and holographic cards as premium collectibles.


  

---


### **Latent Space ▷ #[security](https://discord.com/channels/822583790773862470/1025833219448393878/)** (1 messages): 

swyxio: https://x.com/jacklouisp/status/2025956259594137613?s=12
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: https://youtu.be/x9rWFiIubmc

new pod for the claude code anniversary!
  

---


### **Latent Space ▷ #[ai-general-news-n-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1475558763867013240)** (140 messages🔥🔥): 

> `Anthropic Distillation Attacks, SWE-Bench Deprecation, SaaS is Dead Discourse, MatX $500M Series B, GPT-5.3-Codex Release` 


- **Anthropic Names Names in API Attack**: Anthropic reported that DeepSeek, Moonshot AI, and MiniMax used **over 24,000 fraudulent accounts** to generate **16 million exchanges** with Claude to train their own models via distillation, in what they termed an *[industrial-scale attack](https://x.com/anthropicai/status/2025997928242811253)*.
   - One member pointed out that *Qwen/Alibaba are not mentioned* in the list of bad actors, while another noted that *frontier labs will get swept in the consumer market at this pace* if they don't stop giving their data to the model.
- **SWE-Bench Benched after Benchmarking Backlash**: OpenAI announced the voluntary deprecation of the **SWE-Bench Verified benchmark** due to high levels of data contamination and a significant percentage of unsolvable tasks, as shown in their [official announcement](https://x.com/OpenAIDevs/status/2026025368650690932).
   - Analysis shows that *frontier models are now regurgitating task solutions based on IDs, and approximately 60% of remaining unsolved problems are flawed, making further benchmarking unproductive*.
- **SaaS Apocalypse Now?**: Members discussed [whether LLMs could displace SaaS](https://fxtwitter.com/tenobrus/status/2025648199898407345), with one member arguing that *if tokens get cheap enough that using a bunch of them to replicate a SaaS app on demand is viable, SaaS has problems*.
   - Others countered that *enterprises run on trust and predictability* and would not trust hallucination-prone AI, nor would they want to *roll my own calendly just so I get to deal with maintaining it*.
- **MatX Nets Massive $500M for Matrix Multiplication Machine**: MatX announced a **$500M Series B** led by Jane Street and Situational Awareness LP, for their new **MatX One LLM chip** featuring a splittable systolic array combining SRAM-level low latency with HBM long-context support ([source](https://x.com/reinerpope/status/2026351870852358492)).
- **GPT-5.3-Codex Released for All**: OpenAI Developers announced the immediate availability of **GPT-5.3-Codex** for all developers via the Responses API, inviting them to begin building with the new model ([source](https://x.com/openaidevs/status/2026379092661289260)).


  

---


### **Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1475690635104358622)** (14 messages🔥): 

> `Chinese AI Advancements, GLM-5 Technical Report, DSA adoption, Asynchronous RL infrastructure, Persona Selection Model` 


- **Chinese AI leaps with CoT and Compression**: Discussion highlighted a shift in Chinese AI research towards sophisticated **Chain-of-Thought (CoT) engineering** and integrated **compression pipelines**.
   - There were specific mentions and expectations for upcoming work from **ByteDance**.
- **GLM-5 drops Technical Report**: [Z.ai released the GLM-5 Technical Report](https://arxiv.org/pdf/2602.15763), detailing key innovations such as **DSA adoption** for cost reduction, **asynchronous RL infrastructure** for post-training efficiency, and new **agent RL algorithms**.
   - The model demonstrates state-of-the-art performance, particularly in **real-world software engineering tasks**.
- **Considering Anthropic's Persona Selection Model**: A member considered a look at Anthropic's [Persona Selection Model](https://www.anthropic.com/research/persona-selection-model).
   - They were asking if the model's research would take a full hour for review, and if they should add another paper to cover.


  

---


### **Latent Space ▷ #[singapore-sg](https://discord.com/channels/822583790773862470/1181708804803543140/)** (1 messages): 

coffeebean6887: https://luma.com/c4dmddvh?tk=yciGr7
  

---


### **Latent Space ▷ #[los-angeles-la-lax](https://discord.com/channels/822583790773862470/1203087028401606716/)** (1 messages): 

stealthgnome: https://luma.com/ffla26?tk=wPNgSD
  

---


### **Latent Space ▷ #[ai-in-action-builders-techstacks-tips-coding-productivity](https://discord.com/channels/822583790773862470/1209303473263485011/1475537747191005316)** (40 messages🔥): 

> `Claude Code 'Remote Control', GO GO OS, Open Weight Models, Claude as Coach` 


- **GO GO OS Talk Scheduled for March**: A talk on **GO GO OS - THE AI FIRST OS** by @slono is scheduled for Friday, March 6, 2026, following registration via the AI In Action Bot.
   - The AI In Action Bot helped negotiate speaker signup.
- **LLMs Enable Ambitious Dreams Validated Quickly**: A member described a cycle of *pushing ideas really far and iterating at the speed of slop* followed by *gathering all that stuff and drawing some conclusions and making it reusable* because **LLMs** validate dreams fast.
   - They validated it with their experience building event websocket streaming reactive UI with interchangeable renderers.
- **Elvis Posts Engagement Metrics**: Elvis's Tweet on Feb 23, 2026, had [over 1.3 million views](https://xcancel.com/elvissun/status/2025920521871716562?s=46&t=jDrfS5vZD4MFwckU5E8f5Q).
   - This sparked a discussion on engagement metrics and the value of diverse models for different tasks, where Codex is used for code review and AMP for flaw detection.
- **Claude Code Gets 'Remote Control' Feature**: Noah Zweben announced 'Remote Control' for **Claude Code**, a research preview feature for Max users, enabling developers to start coding sessions in their terminal and transition to mobile ([announcement](https://xcancel.com/noahzweben/status/2026371260805271615?s=12)).
   - A member expressed interest in trying it out, preferring their home desktop setup.


  

---


### **Latent Space ▷ #[share-your-work](https://discord.com/channels/822583790773862470/1209672547642249216/1475631061978579138)** (4 messages): 

> `Commit Change, Vercel AI SDK, Plasmite, AI companion system` 


- **Commit Change Launches for Social Impact**: A vibe engineer created [Commit Change](https://www.commit-change.com), a platform for writing code for social impact and charities, complete with auth and moderation, though currently populated with placeholder projects and developers.
   - The creator seeks feedback and ideas prior to a real launch, questioning if the idea has *legs*.
- **Vercel AI SDK for Node Developers**: A member shared a writeup on [Vercel AI SDK](https://thecodebarbarian.com/getting-started-with-the-vercel-ai-sdk-in-nodejs.html) for Node developers.
   - The article details a way to write code for social impact and charities.
- **Co-Founder Sought for Local-First AI Companion System**: An AI Engineer is looking for a **50/50 AI architecture co-founder** interested in solving personality drift, memory layering, and middleware control over LLMs for a local-first AI companion system focused on identity stability and structured long-term memory.
   - The founder has a full behavioral framework designed and is building an MVP.
- ****Plasmite** IPC Library Released**: Brandon Harvey released [Plasmite](https://github.com/sandover/plasmite), a robust interprocess communication (**IPC**) library in Rust, Node, Go, C, and Python with JSON messages, zero-copy reads, ephemeral readers/writers, and a friendly CLI.
   - It's based on the spirit and style of multi-process design used at Oblong Industries for [spatial computing systems](https://vimeo.com/2229299).


  

---


### **Latent Space ▷ #[robotics-and-world-model](https://discord.com/channels/822583790773862470/1318774781834821746/1475867130078695595)** (4 messages): 

> `Pengchuan Zhang, FAIR, OpenAI, SAM, Llama` 


- **Pengchuan Zhang Joins OpenAI**: Pengchuan Zhang announced his move from Meta's FAIR team to OpenAI to focus on developing physical intelligence through world simulation and robotics; linked to [X post](https://x.com/pengchuanz/status/2026189659228012558?s=12).
- **Zhang departs Meta after 4 years**: Zhang worked on SAM and Llama for nearly four years at Meta's FAIR team.
   - The link was [duplicated](https://xcancel.com/pengchuanz/status/2026189659228012558?s=12) multiple times.


  

---


### **Latent Space ▷ #[genmedia-creative-ai-video-image-voice-music-inspo-consumer-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1475613346874261535)** (8 messages🔥): 

> `OpenAI Realtime API, Anthropic LACMA Art + Technology Lab 2026` 


- **OpenAI Releases gpt-realtime-1.5 for Realtime API**: OpenAI Developers announced the release of **gpt-realtime-1.5**, an updated model for the **Realtime API** featuring improved instruction following, more reliable tool calling, and enhanced multilingual accuracy for voice workflows; the original [announcement is on X](https://x.com/OpenAIDevs/status/2026014334787461508).
- **Anthropic Supports LACMA Art + Technology Lab 2026**: Anthropic announced its support for the **LACMA Art + Technology Lab's 2026** call for proposals, inviting artists worldwide to apply for grants **up to $50,000** for projects exploring art and emerging technology with a deadline of **April 22**, according to [their X post](https://x.com/AnthropicAI/status/2026096054253564002).


  

---


### **Latent Space ▷ #[mechinterp-alignment-safety](https://discord.com/channels/822583790773862470/1445258379357458625/1475592708046196867)** (15 messages🔥): 

> `Anthropic Interpretability Team, ML Infrastructure Engineers, Frontier Models` 


- **Anthropic Seeks Interpretable Infrastructure Innovators**: Chris Olah announced [Anthropic's Interpretability team](https://xcancel.com/ch402/status/2026023963537842248) is hiring approximately **10 seasoned ML infrastructure engineers** to focus on understanding frontier models.
   - Prior experience in interpretability is **not required**, making it a *"good opp"*.
- **Triple the Opportunity for ML Engineers**: Multiple users highlighted the Anthropic Interpretability team hiring announcement as a *"good opp"*, with one user stating *"3x good opp actually"*.
   - The team is seeking experienced ML infrastructure engineers to investigate model internals, with no prior interpretability experience needed.


  

---


### **Latent Space ▷ #[gpu-datacenter-stargate-colossus-infra-buildout](https://discord.com/channels/822583790773862470/1467633569684914349/1475859699571097822)** (8 messages🔥): 

> `Meta, AMD, OpenAI, Strategic Tech Alliances` 


- **Meta to pump 6GW of AMD hardware into data centers**: Starting in **H2 2026**, [Meta plans to deploy 6GW of AMD-based data center infrastructure](https://xcancel.com/shanumathew93/status/2026285588274381129?s=12) over five years, valuing it in the double-digit billions per gigawatt.
   - As part of the deal, Meta got warrants for **160 million AMD shares** tied to performance and deployment milestones, boosting **AMD stock by 15%** pre-market.
- **OpenAI and Meta form Strategic Tech Alliances**: M.G. Siegler [commented](https://xcancel.com/mgsiegler/status/2026274906069950831?s=12) on the strategic dance between **NVIDIA** and **AMD**, spotlighting recent partnership moves by **OpenAI** and **Meta**.
   - He also provided an updated tracking of **Big Tech AI investments**.


  

---


### **Latent Space ▷ #[applied-ai-experimentation](https://discord.com/channels/822583790773862470/1470417186651897858/1475558941252387067)** (13 messages🔥): 

> `Taaalas code generation speed, Codebase Recreation Prompting, cue-lang/cue github repo` 


- **Taaalas Generates Code Blazingly Fast**: Members discussed **Taaalas**'s ability to generate working code in *30 ms*, a speed that one member described as *sci-fi level stuff*.
   - A demo is expected to be ready this week.
- **Prompting Skills Sharpened with Codebase Recreation**: One member suggested cloning a preferred repo and prompting the model to *deep dive into the codebase and then provide a one sentence prompt to recreate that but x, y, z* to improve prompting skills.
   - Another member expressed interest in applying this to **MLflow** and **DSPy** adaptations.
- **cue-lang/cue Github Repo sparks interest**: A member suggested [this Github repo](https://github.com/cue-lang/cue) which they said their agents *fell in love* with.
   - Another member stated that they had already seen it *a while ago, before LLMs even*.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1475538049206063144)** (146 messages🔥🔥): 

> `IBM stock plummets on Claude COBOL streamline, Gemini and Claude hybrid workflow, AI training ethics, Sora 2 copyright violations, AI's impact on BPO market` 


- **IBM Stock plunges After Claude's COBOL Capabilities Debut**: After **Anthropic** announced that **Claude** can streamline **COBOL** code, **IBM's stock** experienced a sharp decline of over **10%**.
   - One member joked about Musk safely editing human brains via **Grok 4.300**, whereas another member joked about **Neuralink with Grok Imagine 1.2**.
- **Coders Leverage Gemini and Claude in Tandem**: Some coders have been using a workflow to leverage the strengths of both **Gemini** for research and **Claude Opus** for final writing, highlighting Gemini's shortcomings in interface usability for project coherence.
   - One user reported using a *free coursera loophole* to get free **Gemini**, although another user mentioned getting free **GLM 5** via kilocode that's just as good.
- **AI Training Ethics Debated in Light of Copyright**: Debates arose around the ethics of AI companies training LLMs on people's work, with suggestions that models should train on *synthetic data* from other models.
   - A participant stated *the main issue that governments and academics have with AI has nothing to copyright or an AI takeover, the main issue is that of National Security, mainly that foreign actors can use AI to rebuild and infere technology that it's been purposefully been obscured by some governments and nation's blocks.*
- **Sora 2's Copyright Woes Delay Release**: Users discuss how copyright violations ruined **Seedance 2.0** and delayed its global release, drawing parallels to **Sora 2's** content-related issues and praising open-source models as an alternative.
   - One user stated *I remember when Sora 2 got content violations, I remember people on X saying they would wait for a CHINESE model to post the copyright, LAMO, they fooled themselves*.
- **AI Automation Reshaping BPO Sector**: The discussion touches on the potential displacement of the BPO market by AI automation, particularly impacting countries like China and India, while smaller, wealthier nations lead in AI implementation.
   - Members joked about how *automation always targets employees first not management* and decision makers rarely automate their own positions. Competitors take them out*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1475547975718797443)** (12 messages🔥): 

> `Control-theoretic prompt regulation, AI invention, Statistical automation, Loopt failure` 


- **AI Control still requires human element**: In response to the question of whether users can meaningfully apply **control-theoretic prompt regulation** to an internal LLM, a member stated that *behavior can be controlled externally, but true system stability can't be guaranteed* due to hidden internal dynamics.
   - They also stated that **users help expand and provide context**, but the initial direction and conditioning comes from the user.
- **GPT helps with grunt work, requires constraints**: Members discussed the need to describe **ontology, architecture, and limitations** when creating complex pipelines with GPTs.
   - One member suggested that it's better to *create it with GPT first, and then edit or supplement it yourself* because **everything returned is what is called a Latent Variable**.
- **AI doesn't invent, finds patterns**: A member stated that as of now **ChatGPT is statistical automation**, on a statistical pattern recognizing model, looping until it finds a latent variable to redo grunt work.
   - They added that *this is why they say AI cannot invent, because it can't, it just finds patterns we haven't put together yet (or ever) due to sheer volume*.
- **Innovation is Recombination + Insight**: A member stated that humans also invent by **recombining prior knowledge, connecting patterns, and iterating on previous ideas**.
   - They added that *very little human invention comes from nothing and most innovation is recombination + insight*.
- **Loopt wasn't a failure but a learning experience**: A member quoted **Sam Altman** saying *I wouldn't call Loopt a failure. It didn’t turn out like I wanted, for sure, but it was fun, I learned a lot, and I made enough money to start investing, which led me to my current job.*
   - Another member responded, *Potato, potáto, keep failing til you succeed, failing isn't failing its learning, same adage just in another form.*


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1475547975718797443)** (12 messages🔥): 

> `Control-Theoretic Prompt Regulation, LLM Invention vs Pattern Recognition, Latent Variables, Statistical Automation` 


- **Control-Theoretic Prompt Regulation in Black-Box LLMs**: A user asked if control-theoretic prompt regulation can be meaningfully applied to an internal LLM + orchestration stack represented as a black box function **F(x)**.
   - The response indicated that while external behavioral control is possible, true system stability cannot be guaranteed due to the hidden internal dynamics.
- **LLMs: Statistical Automata Looping for Latent Variables**: It was suggested that **ChatGPT** is a form of *Statistical Automation* that operates on a statistical pattern recognizing model, looping until it finds a **latent variable** to redo grunt work.
   - One user stated that *This is why they say AI can not invent.. because it cant, it just finds patterns we havent put together yet (or ever) due to sheer volume.*
- **Human vs AI Invention - A Recombination Rumble**: One user argued that AI invention is limited to **recombining prior knowledge** and connecting patterns, similar to human invention.
   - Another user countered that humans can create new patterns without predefined reasons, citing art as an example: *I felt like doing that*.
- **Sam Altman's Loopt: Failure or Fertile Ground?**: A user quoted **Sam Altman** stating, *I wouldn't call Loopt a failure. It didn’t turn out like I wanted, for sure, but it was fun, I learned a lot, and I made enough money to start investing, which led me to my current job.*
   - The user remarked that *failing isn't failing its learning, same adage just in another form*.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1475538622810951743)** (91 messages🔥🔥): 

> `MiSTer FPGA, Anthropic's RefusalBench, Qwen 3.5, Open Source Annotator, Tiny Tapeout ICs` 


- **MiSTer's Purloined Past Provokes Polemic**: Discussion revolved around the [MiSTer project](https://github.com/MiSTer-devel/Main_MiSTer), with accusations that its code was *stolen from Till and killed MiST*, further discussion of the *GPL code they use illegally today*.
   - One member shared a [blog post](https://pingas.org/articles/provenance-of-retro) detailing the project's provenance and controversies.
- **Tiny Tapeout Enables Economical IC Experiments**: A member shared a link to [tinytapeout.com](https://tinytapeout.com/) where *they actually let you tape out ICs for little money*, though the designs must be rather small.
- **Anthropic Accuses DeepSeek of Dubious Duplication**: A link was shared to an article discussing how *Anthropic is furious at DeepSeek for copying its AI without permission*, sparking debate about the irony given Anthropic's own practices and this article [Anthropic Furious at Deepseek](https://www.msn.com/en-us/news/technology/anthropic-furious-at-deepseek-for-copying-its-ai-without-permission-which-is-pretty-ironic-when-you-consider-how-it-built-claude-in-the-first-place/ar-AA1WYupG).
   - One member stated *Yup we love the soap opera*, suggesting a cynical view of the situation.
- **Qwen 3.5 Achieves Awesome Advancements**: It was noted that *Qwen3.5-35B-A3B beating Qwen3-235B-A22B-2507 is insane*. Also a link was shared [huggingface.co/Qwen/Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base) of the base weights released.
   - In addition, *5.3 codex is out in API: $1.75 input, $14 output* loads cheaper than Anthropic.
- **Regulatory Capture Rears its Reprehensible Reality**: A member stated *in 2 words: regulatory capture* in response to a discussion about Baidu and Anthropic, hinting at concerns about potential undue influence.
   - Another member made the observation *Baidu is also known for regulatory capture in China*.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1475859341059031182)** (2 messages): 

> `Hermes, emergent misalignment, fine tuning, evil AI, AI safety` 


- **Fine-Tuning Hermes for Emergent Misalignment?**: A member inquired whether anyone has tested fine-tuning **Hermes** specifically for **emergent misalignment** or, in simpler terms, to *go evil*.
- **Ethical Implications of AI Fine-Tuning**: The inquiry raises concerns about the ethical implications of intentionally fine-tuning AI models for malicious purposes, highlighting the importance of **AI safety** research.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1475568039725629582)** (60 messages🔥🔥): 

> `Latent Reasoning Tokens, Deepseek R1 Paper, EleutherAI Pythia-2.8b HF weights bug, Google Student Researcher Program 2026, lesswrong.com/posts/2pkNCvBtK6G6FKoNn/so-you-think-you-ve-awoken-chatgpt` 


- **LLMs get **Latent Reasoning** boost**: There was discussion about using special **tokens only generated by the LLM** and not displayed to the user to enhance reasoning, called *Latent Reasoning*, potentially improving performance and security.
   - A link to the paper [Latent Reasoning](https://arxiv.org/abs/2307.06203) was shared.
- ****Deepseek R1 Paper** drives discussion on Reasoning**: Discussion revolved around how reasoning in larger models arises from reinforcement learning rather than direct data learning, referencing the **Deepseek R1 paper**.
   - It was noted that auxiliary rewards are typically used to ensure human readability, though it's uncertain if this approach is optimal.
- ****EleutherAI Pythia-2.8b weights bug** surfaces on HF**: A member reported a bug while trying to reproduce [a paper](https://arxiv.org/abs/2309.23024) with **EleutherAI's pythia-2.8b**, where the Hugging Face Hub served the same weights regardless of the revision chosen.
   - It was found that both `pytorch_model.bin` and `model.safetensors` had the same SHA256 across different steps, but the sharded files (`model-00001-of-00002.safetensors`) were different.
- **EleutherAI fixes **Pythia-2.8b HF weights bug****: The **EleutherAI Pythia-2.8b weights bug** has been acknowledged by a member, who mentioned being rate-limited by HF while trying to fix it.
   - There was confusion around sharded files, with the understanding that if the 1-of-2 and 2-of-2 files are correct, users can combine them and load the model.
- ****Dupe Data** Discovered in some EleutherAI Models**: The **14m and 30m** models were found to be deduped versions instead of duped versions as labeled on HF.
   - Retraining is ongoing to replace them with correctly labeled duped models, and links to the new HF models, [14m](https://huggingface.co/stellaathena/pythia-14m) and [31m](https://huggingface.co/stellaathena/pythia-31m) were provided, and uploads should complete roughly on the hour.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1475760290238038037)** (27 messages🔥): 

> `Baguettotron, ML papers published in Nature, ViT broken?, Differential Attention` 


- ****Baguettotron** has arrived!**: The **Baguettotron** model features **4608** features, full autointerp labels, was trained on **774M** tokens, layer **48/80**, **8x** expansion, and top_k **32**; its creator also shared [an X post](https://x.com/Ji_Ha_Kim/status/2026166070172655786?s=20) for context.
   - The creator gave a link to the live [demo](https://lyramakesmusic.github.io/bread-slicer/).
- **Nature publications = Bear Signal?**: A member quipped that *ML papers published in Nature* are a bear signal, except when **DeepMind** does it.
   - The remark was in response to a detailed explanation of a gating mechanism connecting *skeleton stabilization* and *detail rendering* technically.
- **Vision Transformers are Broken?**: A member stated that **ViT** naively applied to **CIFAR10** is wrong because converting patches into *tokens* by a simple linear layer is insufficient, leading to suboptimal representation.
- **Differential Attention Ablation Study Feedback**: A member requested feedback on ablation studies related to differential attention, sharing [a PDF document](https://cdn.discordapp.com/attachments/747850033994662000/1475931314837262397/v2_draft.pdf?ex=699f47a6&is=699df626&hm=2c1090efdc639f38dfa72ea50d7871ae4f662b13d002ff4d9d2004355c0564b0&).
   - A respondent criticized that the ablation did not prove if differential attention is fundamentally better or if it just benefits disproportionately from the method.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1475921848092069979)** (1 messages): 

> `LLM Behavior Debugging, LLM Reasoning Traces, LLM Refusals, LLM Agent Behavior` 


- **Researchers Seek LLM Debugging Interviews**: Researchers are seeking **20–30 minute interviews** with individuals involved in evaluating or debugging **LLM behavior**, offering a **$25 Amazon gift card** or charity donation as compensation.
   - They are particularly interested in workflows and tools that aid in understanding *why* an **LLM** produced a specific output, focusing on reasoning traces, refusals, and agent behavior, with a [booking link provided](https://calendly.com/amerrick4-rrc/ai-auditing-problem-interview).
- **LLM Evaluation Focus Areas Detailed**: The research specifically targets individuals who work with **inspecting chain-of-thought**, **interpretability or latent-knowledge**, **debugging agent behavior**, and **analyzing refusals or safety failures** in **LLMs**.
   - The goal is to map out time allocation during evaluation and debugging processes, providing valuable insights into current practices.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1475673872937386144)** (2 messages): 

> `lm-evaluation-harness Bug Fix` 


- **Tiny PR Tackles Tricky Test Tweak**: A member submitted a one-line PR to fix a bug in the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/3293).
   - They noted that the fix should be super simple to review.
- **Athena Appreciates Assistance**: A member expressed gratitude for the bug fix submission.
   - This acknowledgement highlights the collaborative nature of the project.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1475546792488730734)** (2 messages): 

> `eval_adapter.py fix` 


- **Forward Pass Bug squashed**: A member shared a [fixed version](https://gist.github.com/aflah02/8e6b726bd08828b9a48b0cd354ad8431) for `eval_adapter.py`, resolving an issue in the forward pass call.
   - The solution involves wrapping the forward pass call and adjusting the elements to match the schema in the `eval_adapter.py` file.
- **Repo integration considered for adapter fix**: A member proposed integrating the adapter fix into the repository, pending community interest.
   - This aims to resolve the forward pass issue more broadly for users of the `eval_adapter.py`.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1475551758574161950)** (8 messages🔥): 

> `LLM memory traces, IO testing libraries, Training small models on RTX cards, Graph DBs vs Vector DBs, Tiny GPU Compiler` 


- ****State-of-the-Art** Memory Traces Methods Sought**: A member inquired about the **state-of-the-art method** for generating memory traces from an **LLM workload**.
   - No specific methods were shared in the immediate discussion, but the question itself indicates interest in optimizing memory usage within **LLM** applications.
- ****Tiny GPU Compiler** targets open-source GPU Hardware**: A member introduced **tiny-gpu-compiler**, an educational MLIR-based compiler targeting open-source GPU hardware, explained in this [GitHub repo](https://github.com/gautam1858/tiny-gpu-compiler).
   - The compiler translates a **C-like GPU kernel language** into **16-bit binary instructions**, and includes an interactive web visualizer for step-by-step execution analysis available at [tiny-gpu-compiler](https://gautam1858.github.io/tiny-gpu-compiler/).
- ****Graph DBs** Take On Vector DBs**: A member inquired what **Graph DBs** help out with at a foundational level, instead of using **Vector DBs** for agents.
   - The conversation did not elaborate, but the question suggests exploring alternative database architectures for agent-based applications.
- **How to train Small Model using an RTX card**: A member asked about training a **small model (125M parameters)** using an **RTX 2070-2080 or 3070-3080** card.
   - They were looking for information on **tokens processed per second** to compare with their own **GTX 1080 Ti** setup using custom kernels.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1475570440792571914)** (3 messages): 

> `Triton, Gluon, TTGIR, TTIR` 


- **Gluon atop TTGIR, not replacement for Triton**: A member inquired whether **Gluon** is an extension of **Triton** or a replacement, and another clarified that **Gluon** is a new language built on top of **TTGIR** instead of **TTIR**.
- **Gluon: A New Language**: A discussion clarified that Gluon is designed as a completely new language, differentiating it from being merely an extension of Triton.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1475609362780131538)** (17 messages🔥): 

> `GPU Memory Optimization, CUDA memcpy_async and __syncthreads(), CPU to CUDA Verification Strategies` 


- ****Optimize**: Squeeze More Juice from Your GPU**: To optimize GPU code, one should measure how much memory the operation needs to read and how fast the GPU can read this memory, as well as how many operations the operation needs to perform and how fast the GPU can perform them.
   - For **RMS norm**, performance is likely to be memory bound, so focus on optimizing memory access patterns and bandwidth utilization and check [Nvidia's CUDA documentation on Asynchronous Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#batching-loads-in-conditional-code).
- ****Sync Up**: Deep Dive into CUDA's async Memory Copies**: The use of `__syncthreads()` with `CUDA C++ cuda::memcpy_async` is necessary for memory visibility, ensuring that all threads see the data copied by any thread, as clarified with reference to the [CUDA documentation on asynchronous barriers](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#tracking-asynchronous-memory-operations).
- ****Port Authority**: Validate that CUDA Code**: When porting CPU code to CUDA for production, general guidance on verification includes managing floating-point precision and setting standard tolerances for operations like **gemms** for different floating-point sizes, especially during GPU versioning.
   - The approach depends on the context, such as **PyTorch** or **VLLM**, and involves considering the appropriate units of code (kernels) to verify, the number of inputs/outputs to test, and broader concerns beyond simple precision.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1475561331108090008)** (2 messages): 

> `FlashAttention 3, SDPA in PyTorch, Pre-built Wheels` 


- **Flash Attention 3 pre-built wheels are here!**: Pre-built **Flash Attention 3 wheels** are available for various CUDA versions (**12.6+**, **13**), CPUs (**x86**, **ARM**) and OS (**Linux**, **Windows**) at [download.pytorch.org](https://download.pytorch.org/whl/flash-attn-3/).
   - These wheels are **LibTorch ABI stable** and should work with any Python version >= **3.10** and torch version >= **2.9**.
- **Torch's SDPA Kernel Selection Examined**: A user asked how Torch's SDPA (Scaled Dot-Product Attention) selects the right kernel.
   - The answer involved the use of `activate_flash_attention_impl("FA3")` to redirect the dispatcher to use **FA3 kernels** instead.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1475967596871549080)** (1 messages): 

> `eBPF, GPUs, profilers, OS Policies` 


- **eBPF Extended to GPUs**: Yusheng Zheng will discuss extending **eBPF** to enhance **GPU** functionality on [December 12 at 12:00 pm PST](https://arxiv.org/abs/2512.12615).
   - This talk will cover recent work, including *gpu_ext: Extensible OS Policies for GPUs via eBPF* and extending eBPF to **GPU Device** and **Driver Contexts**.
- **Profiling Libraries gain GPU Mode Attention**: There is expressed interest in developers building more **profilers** and **profiler visualization libraries** within **GPU MODE**.
   - Interested individuals are encouraged to join a relevant [YouTube video](https://www.youtube.com/watch?v=8U7SzGnHoJU) that may further inform the discussion.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1475714741715599370)** (2 messages): 

> `ParoQuant Project, Eigenvalues Comparison, Magnitude Difference` 


- **ParoQuant Project gets link!**: A member shared a link to the [ParoQuant project](https://z-lab.ai/projects/paroquant/).
   - Another member expressed that they *liked the idea of the top 10 just like picking top eigen values*.
- **Magnitude difference is simpler than Eigenvalues!**: A member speculated that the **largest magnitude difference** was chosen for simplicity in computation compared to **eigenvalues**.
   - They noted it is *simple to compute as opposed to eigen values*.


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1475762616667344906)** (1 messages): 

> `Nvidia, Linux drivers, Job Postings` 


- **Nvidia Hires Linux Driver Fixers**: Nvidia is [hiring employees](https://jobs.nvidia.com/careers/job/893393264012) to enhance their Linux drivers.
- **Nvidia's Job Opening Targets Linux Improvement**: Nvidia seeks personnel to refine its Linux driver ecosystem, as indicated by a [recent job posting](https://jobs.nvidia.com/careers/job/893393264012).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1475699397345149032)** (2 messages): 

> `NCCL, NVSHMEM performance, direct pointer access, matrix transpose` 


- ****NCCL** and **NVSHMEM** Advantages Exposed**: After watching a talk on **NCCL** and **NVSHMEM**, direct pointer access via *nvshmem_ptr* is noted as simpler to program and faster than explicit memory transfers (e.g., *nvshmem_get*) in matrix transpose scenarios.
   - The performance gap is investigated, questioning if an *ideal* warp-based *getmem* variant could close the gap with *nvshmem_ptr*.
- ****NVSHMEM** Pointer Performance Edge Highlighted**: A key finding from testing **NVSHMEM** is that direct pointer access (*nvshmem_ptr*) outperforms explicit memory transfers (*nvshmem_get*), due to the pointer version not using a temporary buffer.
   - One of the experts suggests that the pointer version eliminates buffering, leading to a performance boost and more elegant code, however, *the put version should be better than the get one, but I haven't had time to write it.*


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1475588161127190569)** (1 messages): 

> `NCCL, SHMEM, RDMA, CUDA kernels, IRL Collaboration` 


- **Bostonian Newb Seeks NCCL/SHMEM/RDMA/CUDA Cohort**: A Boston-based newbie is dedicating time to understanding **NCCL**, **SHMEM**, **RDMA**, and **CUDA kernels** and is open to in-person chats and collaborative learning.
   - The user is interested in potentially collaborating on a small project to deepen understanding of these technologies.
- **Accountability Partner for Kernel Kombat?**: The user proposes the idea of an accountability partner to ensure concrete deliverables, such as *"submitting your best matmul kernel in 48 hours."*
   - They note that allocating protected time to *"build something"* can be challenging due to the breadth of concepts to learn.


  

---


### **GPU MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1475652834035761183)** (1 messages): 

> `N-Dimensional Tensor Visualizer, Einops-like Syntax, Colab Notebook` 


- **N-Dimensional Tensor Visualizer has Landed!**: A new **n-dimensional visualizer** was released, now supporting tensors up to **9D**.
   - The visualizer allows users to slice, permute, and inspect every value in N-dimensional tensors just as easily as 1D, 2D, or 3D tensors, using an **einops-like syntax**.
- **Colab Notebook gets you visualizing**: A [Colab notebook](https://colab.research.google.com/drive/1lrO6yzVQ8u_vFLPe7986goZtRQazmV0T#scrollTo=Q0TZi3zPxWhB) was created to walk users through the visualizer from 1D to 9D tensor copies.
   - A video shows the visualizer inspecting a tensor of shape **(2, 3, 4, 3, 4, 2, 4, 2, 3)**.


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1475540096194187275)** (21 messages🔥): 

> `KernelBench Environment, CUDA Memory Errors, Modal.experimental.stop_fetching_inputs, KernelBot Environment, Eval.py reuse` 


- ****KernelBench Environment** tackles CUDA Errors**: A member created a custom environment for **KernelBench** and **kernelbook** to address corrupted **CUDA memory errors**, intending to share it.
- **Modal Fixes **CUDA Errors**!**: The error *"cuda memory error is detected"* can be resolved using `modal.experimental.stop_fetching_inputs`.
   - This fix is already implemented in the member's `backendbench env`.
- **KernelBot Deployed and Solves problems!**: The initial environment for **KernelBot** is now up, available at [app.primeintellect.ai](https://app.primeintellect.ai/dashboard/environments/roeybc/kernelbot-env), currently supporting `trimul` & `amd` problems, and using Modal for Nvidia and Runpod for AMD issues.
   - The team is working on `PMPP` problems and AMD distributed kernel problems.
- **Reuse eval.py to Prevent Bugs**: Members are encouraged to reuse the evaluation logic from `eval.py` to reduce issues related to inconsistent logic across different competitions.
   - They discussed potentially adapting the internal functionality of `eval.py` for each problem due to constraints like differing tolerances and iteration counts.
- **Evaluation and Analysis Hitting Northflank**: The channel agenda includes migrating **KernelBot** to Northflank, site UI improvements, AI evaluation of cheats, and discussions on end-to-end inference speedruns.
   - Parsing of hacky submissions is complete, with fingerprinting and deeper analysis underway.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1475935136578011281)** (2 messages): 

> `Heroku to Northflank migration, Bot/Web downtime, CLI update` 


- **Heroku to Northflank Migration causes Downtime**: The services are undergoing migration from **Heroku** to **Northflank**, which will cause downtime for the bot and web services.
   - Users are asked to bear with the downtime.
- **Services Restored Post-Migration**: The services are back online, and users are asked to report any issues, especially with **auth**.
   - Users must update their **CLI** to the latest version to ensure everything runs smoothly.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1475970084634624010)** (1 messages): 

> `B200, Leasing, Neocloud` 


- **B200s are Incredibly Expensive!**: A member noted that **B200s** are incredibly expensive and suggested leasing/renting from a **neocloud** unless you are an enterprise.
   - The member shared a link to their company's solution at [lightning.ai/clusters](https://lightning.ai/clusters).
- **Cloud Leasing is cost effective!**: Unless an enterprise, leasing/renting from a neocloud makes a lot of sense.
   - If interested, check out: [lightning.ai/clusters](https://lightning.ai/clusters).


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/)** (1 messages): 

epiicepiic: Gotcha. Thanks for clarification!
  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1476006971227377815)** (1 messages): 

> `RRCLLX, AMD MI300X, Meta, GPU Communications` 


- **Meta Innovates with RRCLLX for GPU comms on AMD Platforms**: Meta is innovating GPU communications on AMD platforms using **RRCLLX**, as detailed in their [engineering blog post](https://engineering.fb.com/2026/02/24/data-center-engineering/rrcclx-innovating-gpu-communications-amd-platforms-meta/).
- **AMD MI300X gets RRCLLX**: Meta is using **RRCLLX** to connect **AMD MI300X** GPUs more efficiently.


  

---


### **GPU MODE ▷ #[low-bit](https://discord.com/channels/1189498204333543425/1411659097706860647/1475627773325213736)** (2 messages): 

> `BitNet 1.58b, Mamba2, 4Bit-Forge` 


- **BitNet 1.58b Pairs with Mamba2**: A link to **BitNet 1.58b + Mamba2** was shared from [Zenodo](https://zenodo.org/records/18394665).
- **4Bit-Forge Rework Underway**: A member mentioned they are currently reworking **4Bit-Forge** in CUDA.
   - They expect to provide an update soon.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1475929250640171162)** (2 messages): 

> `Helion v0.3.0 Release, Autotuning Improvements, Triton-to-TileIR Bridge, Pallas TPU Support, CuteDSL Codegen` 


- **Helion v0.3.0 Debuts!**: The new [Helion 0.3.0 release](https://github.com/pytorch/helion/releases/tag/v0.3.0) includes improvements for **autotuning**, support for **Triton-to-TileIR bridge**, and major refactors for **Pallas TPU** and **CuteDSL codegen** support.
   - Notably, the autotuning improvements are detailed in [this blog post](https://pytorch.org/blog/accelerating-autotuning-in-helion/).
- **Helion's Parallel-Read Fix?**: Excitement around [this pull request](https://github.com/pytorch/helion/pull/1418) centers on whether it addresses the parallel-read becoming atomics issue.
   - The issue is outlined in the [JAX documentation](https://docs.jax.dev/en/latest/pallas/design/design.html#grad-of-pallas-call).


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1475547563259068614)** (4 messages): 

> `Guaguabear Confusion, Kernel Programming Environment Setup, Top AI Generated Solutions` 


- **"Guaguabear" Misinterpretation Causes Double Take**: A member humorously noted doing a double take upon seeing the username *"guaguabear"*, initially misreading it as *"bear bear bear"*.
- **Users Seek Kernel Programming Environment Setup Advice**: One member inquired about preferred kernel programming environment setups, noting that while [Modal](https://modal.com/) is helpful, it lacks **NCU profiling support** outside of contests.
   - They seek a customizable environment for kernel writing and optimization, implying existing solutions may be too restrictive.
- **Top AI Generated Solutions Debated**: In response to the question *"Which is the top AI generated solution?"*, one member suggested **Ouye Xie for CUDA** and **billcarson cutedsl**.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1475693527001206795)** (10 messages🔥): 

> `VR Teleop for SO-101, CamBot Robot Design, Open Arms Redesign for 3D Printing, Filament Choices for 3D Printing` 


- **Custom VR Teleop Code Whips SO-101 into Shape**: A member wrote custom code for **VR teleoperation** of the **SO-101** robot arm, finding the **SO-107's** additional joint very useful for matching **XYZ space** reached with hands in VR.
   - They found that the **SO-107 additional joint** is very much worth it, as the additional degree of freedom matches better with xyz space I can reach with hands in vr.
- **CamBot Debuts with FeeTech Motors**: A member implemented a full custom teleop via **Web-Sockets** and designed a new **6 DoF CamBot Robot** which works with the same **FeeTech Motors** used for the **SO-101**.
   - The design is currently being tested, as can be seen in the linked [image](https://cdn.discordapp.com/attachments/1437390897552818186/1475702902470217748/grafik.png?ex=699f1bad&is=699dca2d&hm=e23ca6354e29cb229cfd1aa620ed5ab2c4a742eae6e22214522dc342fa9357eb&).
- **Open Arms Platform Embraces 3D Printing Revolution**: A member is currently redesigning the **Open Arms** platform to be **3D printed**, aiming to bring the platform cost down to under **$2.5k**.
   - They are using a motorized standing desk leg as a cheap source of **3 stage linear actuators**, as can be seen in the linked [image](https://cdn.discordapp.com/attachments/1437390897552818186/1475871315205554287/IMG_0547.jpg?ex=699f0fc5&is=699dbe45&hm=58b06bae5d2b9d1ebb4b4b820301bc5948185c18697907a8f5a8539282dbe837&).
- **Filament Faceoff: PLA vs PLA-CF vs PETG vs PA6**: A member is considering moving beyond **PLA** to filaments like **PLA-CF**, **PETG**, or **PA6 (Nylon-Fiber)** for their experiments.
   - This may require a new printer with an enclosure and air-filter; they are currently printing with a **Bambu Labs A1** and are overall super happy with the machine.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1475778260477546660)** (3 messages): 

> `HPC Employability, GPU Knowledge Modernization, CUDA Kernels` 


- **HPC Seniority in Demand?**: A researcher with **15+ years** of experience in **HPC** seeks perspective on employability outside academia, particularly in the current **AI-heavy** job market.
   - They are wondering if their expertise in **OpenMP**, **MPI**, **CUDA**, multi-GPU systems, and parallel languages like **Chapel** and **Julia** is attractive to employers.
- **Charting GPU Knowledge Modernization**: The researcher is seeking advice on modernizing their **GPU** knowledge, considering areas like **Triton**, **CUDA Graphs**, compiler stacks, and **ML** systems internals.
   - They are looking for guidance on how to best position themselves in the current market, given their extensive background in HPC.
- **CUDA Kernels for Inference**: A contributor to **tpu-inference** (Google's vllm TPU backend) asks how essential extensive **CUDA kernel** knowledge is for inference and MLsys roles.
   - Despite contributing significantly to tpu-inference ([documented on the readme](https://github.com/catswe)), they have only tutorial-level experience with kernels and are unsure how much their non-kernel work will carry them.


  

---


### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1475869597662969856)** (4 messages): 

> `Speedup metrics in competitions, CUDA C++ compile flags, Solution Artifacts` 


- ****Craving Competition's Speedup Stats****: A member inquired about the availability of speedup metrics, similar to those seen in other competitions, within the current benchmark setup, as shown in the [attached image](https://cdn.discordapp.com/attachments/1464407141128339571/1475869597247995944/image0.jpg?ex=699f0e2c&is=699dbcac&hm=bc0c5e273addb955b82fd8b4ffa0ed6af456b86248e987365772d36e4d77413e).
- ****CUDA Compile Flag Configuration Conundrum****: A member questioned the absence of an option to pass extra compile flags for **CUDA C++** submissions in the torch builder and TVM FFI builder, referencing relevant [code](https://github.com/flashinfer-ai/flashinfer-bench/blob/c1fd980f70263c83ab47a43325cf87f2dba9b61a/flashinfer_bench/compile/builders/torch_builder.py#L154-L162).
   - The relevant [TVM FFI code](https://github.com/flashinfer-ai/flashinfer-bench/blob/c1fd980f70263c83ab47a43325cf87f2dba9b61a/flashinfer_bench/compile/builders/tvm_ffi_builder.py#L264-L270) was referenced.
- ****Solution Artifacts Gain Compile Flags****: A member suggested incorporating compile flags into the solution artifact, emphasizing its importance for reproducibility and consistent evaluation.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1475549164615106661)** (41 messages🔥): 

> `Unblocking Request, Anthropic Accusations, Tool Changes During Prompt Cycles, Kimi K2.5, Kimi K3` 


- **User Pleads for Unblocking**: A user asked the owner (tagged user) to unblock them, requesting a 5-minute conversation and also tagged another user to relay the message.
   - The user then posted a link to a [WSJ article](https://www.wsj.com/tech/ai/anthropic-accuses-chinese-companies-of-siphoning-data-from-claude-63a13afc) about **Anthropic accusing Chinese companies of siphoning data from Claude**, labeling it *pathetic*.
- **Moderator Redirects Off-Topic Plea**: A user made a request that was addressed by a community moderator, who stated that the server was *not the place for this topic* and suggested using another server.
- **Tool Changes Enquired During Prompt Cycles**: A user enquired about the possibility of changing the **tools available during a prompt-to-response cycle**.
- **User Laments Lack of Browser Extension**: A user expressed the sentiment that the only thing missing from Kimi K2.5 is a **browser extension**.
- **Bug Report Urged for Kimi Error**: A user reported an error that has persisted for **10 days** with an [attached image](https://cdn.discordapp.com/attachments/1371757564005711973/1475932351497240717/image.png?ex=699f489e&is=699df71e&hm=2b588317c8756fd95479fe5ddb11eee39b51d5f888ebb10ba0629823a8b746d9&).
   - A moderator asked the user to make a **bug report** with all relevant details.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1475567355491913920)** (16 messages🔥): 

> `Lucidrains Github, Sentence Relevancy, Attention Model Scout, Dell Pro Max GB10, NVIDIA memory usage` 


- ****Lucidrains** Github Gone?**: A member inquired about the whereabouts of **lucidrains'** GitHub repository and the reasons behind its potential removal.
- ****Scout** model learns directional sentence relevance**: A member introduced **Scout**, a new attention model that modifies the standard Transformer architecture, designed to learn directional relevance between sentences instead of tokens, asking *"does sentence B actually help sentence A?"*, and is available on [GitHub](https://github.com/samyak112/Scout).
- ****GB10** struggles with memory**: A member shared their experience with a **Dell Pro Max GB10**, noting it works but is slow, with shared GPU/CPU memory leading to frequent **GPU OOMs**, causing system freezes and reboots.
   - They recommended using `nvitop` for tracking memory usage and GPU stats, as `nvidia-smi` output is reportedly broken.
- ****Foundation Models** are Back**: A member shared a link to a [SI article on Foundation Models](https://si.inc/posts/fdm1/), suggesting their resurgence.
   - Another member shared a [YouTube video](https://youtu.be/IeeFOpS-S_M?si=eBJeM3UeI_E1aHjD) discussing AI in education.
- ****GANfather** Returns!**: **Ian Goodfellow**, the creator of **GANs**, is back ([tweet](https://fxtwitter.com/goodfellow_ian/status/2026024150213738520)), sparking hopes for a **GAN** renaissance in solving the verification problem.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1475545382569119865)** (11 messages🔥): 

> `Wave Field LLM, Mercury II by Inception AI, Continuous field tokens` 


- **Wave Field LLM Receives Skeptical Reception**: A member shared a [GitHub repository for Wave Field LLM](https://github.com/badaramoni/wave-field-llm) and asked if it was something to look out for.
   - Another member replied that the **baselines look weak** and *they don't see ablations*, thus they are sceptical.
- **Continuous Field Tokens Already Explored**: A member pointed out that the concept of tokens living on a continuous field, as described in **Wave Field LLM**, has already been explored in [this paper](https://arxiv.org/abs/2406.11838).
- **Mercury II by Inception AI Debuts**: A member mentioned **Mercury II** developed by **Inception AI** and shared a link to [Inception AI's website](https://www.inceptionlabs.ai/) and a link to the **arXiv paper**.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1475927172999942226)** (1 messages): 

> `` 


- **System Prompt Inquiry**: A user inquired about the system prompt used for generating the article.
   - The user expressed difficulty in judging the realistic grounding of the content without access to the system prompt.
- **Request for Prompt Inclusion**: The user requested that the system prompt be included for context.
   - This would allow for a better assessment of the AI's constraints and instructions.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1475890838851813531)** (6 messages): 

> `Liquid AI's LFM2-24B-A2B, WarClaude, Alibaba Qwen release` 


- **Liquid AI floats LFM2-24B-A2B**: Liquid AI announced the release of **LFM2-24B-A2B**, described in their [blog post](https://www.liquid.ai/blog/lfm2-24b-a2b) showcasing their latest advancements.
   - The model aims to set a new standard in efficient and effective AI solutions.
- **Alibaba Qwen Quells Questions**: Alibaba introduced updates to **Qwen**, detailed in a [post on X](https://fxtwitter.com/Alibaba_Qwen/status/2026339351530188939?s=20), enhancing its capabilities and accessibility.
   - Users are encouraged to check out the announcement for more details on improvements and new features.
- **X Marks the WarClaude spot**: A member expressed interest in **WarClaude** and linked to related content on X, see [post 1](https://x.com/i/status/2026369451403390999) and [post 2](https://x.com/i/status/2026369453655732693).
   - There was no other discussion or context provided about what *WarClaude* actually is.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1475583828402897106)** (19 messages🔥): 

> `Vulnerability reporting, Unlimited chat tier, Account transfer, Telegram credit usage, Desktop app billing` 


- **Vulnerability Reporting Query**: A user reported finding a vulnerability and was directed to the [feedback page](https://manus.im/feedback?source=help_center).
   - The user expressed confusion about how and where to report the vulnerability.
- **Unlimited Tier Considered**: A user inquired about a potential unlimited chat tier similar to **ChatGPT** or **Grok**, motivated by the fast credit depletion of the **Manus Agent** in Telegram.
   - A representative responded that they appreciate the feedback and are constantly working to improve the product.
- **Account Transfer Woes**: A user requested to transfer their project to a different account and provided the relevant email addresses.
   - Support confirmed that direct account transfers are not currently supported, recommending users download content locally and start a new task on the new account.
- **Telegram Agent's Credit Consumption**: A user mentioned that the Telegram agent is *very nice* but *blows so many points away from my account* due to high credit usage.
   - This reinforces the earlier question about a subscription option to alleviate credit concerns.
- **AI/ML Engineer offers expertise for scaling serious AI products**: An AI/ML engineer offered expertise for building serious AI products that can scale, emphasizing the importance of inference cost, memory design, and system behavior under load.
   - He has spent the last few years working on AI systems where technical decisions actually affected whether the product survived or not.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

darinsimmons: Welcome Zayden, this discord is for discussions about Modular, mojo, and MAX.
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1475612841544253471)** (7 messages): 

> `String templating in Mojo, Writable and Writer traits, ExternalFunction struct` 


- **Mojo's String Template Engine Proposal Surfaces**: A member has opened a proposal for a new **string templating feature** in Mojo, with the [discussion thread on the forum](https://forum.modular.com/t/writable-writer-template-engines/2763).
   - The feature is likely to be **post-1.0**, with hopes of extending the current `Writable`/`Writer` trait into a more complex `TemplatedWritable`.
- **`Writable` & `Writer` traits need love**: The current `Writable` and `Writer` traits should be minimal with extension/customization points via other traits or via defaulted trait methods/types.
   - The roadmap will prioritize other features like **Int unification** before addressing the proposal, with the goal of unifying `write_to` and `write_repr_to` implementations into a single function.
- **ExternalFunction struct trick**: A member mentioned they have been using the `ExternalFunction` struct as inspiration and looking for a fancier version to decompose the function sig into its params / return types.
   - They will likely need to code the **origin casts for all external pointers**.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1475576895914639472)** (8 messages🔥): 

> `Github CI Failures, Broken Links in Docs, Linux Foundation Summit, Ezra Klein Agents` 


- **Github CI Failure Fix Investigated**: A member reported that `npm run generate`, `npm run format`, and `npm run check` all passed locally on [PR 2278](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2278), but there were failures in CI.
   - The root cause was a missing file that resulted in a broken link in `docs/community/seps/index.mdx`.
- **Linux Foundation Summit Meetup**: A member invited others attending the [LF Member Summit](https://events.linuxfoundation.org/lf-member-summit/) in Napa, CA to meet up and chat about MCP.
   - No further details about the venue or scheduling were offered in the messages.
- **Ezra Klein Learns About Agents**: A member shared a [YouTube video](https://youtu.be/lIJelwO8yHQ) of Ezra Klein learning about agents.
   - The video was shared without additional commentary.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1475908010428727329)** (4 messages): 

> `Aider future updates, Git submodules support in Aider, Low-cost LLMs for Aider` 


- **Aider's Future Uncertain**: A user inquired about whether **Aider** is still being actively developed and if there are recommended alternatives.
   - Another member mentioned that there are other CLIs more *advanced* than Aider.
- **Aider Lacks Git Submodules Support**: A computer scientist noted that **Aider** doesn't support **git submodules** and has proposed an improvement, detailed in [this GitHub issue](https://github.com/Aider-AI/aider/issues/3603).
   - They are looking for feedback and advice on this proposed feature.
- **Seeking Low-Cost LLMs for Aider**: A member is seeking advice on finding a low-cost **LLM** to use with **Aider**, as **Gemini** quickly exhausted their tokens.
   - They are looking for **LLMs** that balance cost-effectiveness with usability within the **Aider** framework.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1475804606142353448)** (1 messages): 

> `Aider limitations with complex tasks, Scripting Aider for repetitive tasks, Using Aider with external scripts or agents, Finding function usages` 


- **User flags Aider's Fuzzy File Find Falls Flat**: A user shared that they value **Aider** as an AI tool for tasks like fuzzy search and replace across multiple files, but are facing limitations with more complex scenarios.
   - The user is running into **diff formatting issues** when processing too many files at once, forcing them to work in smaller chunks.
- **Hacking Aider with Scripts for Task Automation**: The user is seeking guidance on how to use external scripts to automate repetitive tasks with **Aider**, such as looping through files to perform edits.
   - They are asking if there are existing tools that already facilitate this kind of interaction with **Aider**.
- **Agents: Aider's Next Frontier?**: The user is wondering if their desired functionality aligns with the concept of **AI agents** and is open to exploring **Aider** forks or tools like **opendesk** or **cline**.
   - They want to improve the workflow to find function usages, ensure they meet certain criteria, and then edit the files by adding lines and passing new parameters, without manual intervention in VSCode.
- **User Needs Help Finding All Function Usages**: The user wants to automate the process of finding all usages of a function.
   - They would like to check that the usage meets certain criteria, then do a simple edit on the file where it adds two more lines and passes a new parameter to it.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1475552347668348938)** (4 messages): 

> `tiny-gpu, AMD Ryzen AI, MLIR compiler` 


- **Tiny-GPU Compiler launches!**: A member introduces the [tiny-gpu-compiler](https://github.com/gautam1858/tiny-gpu-compiler), an educational **MLIR-based compiler** targeting open-source GPU hardware, with an interactive web visualizer.
- **Tiny-GPU Compiler Goes Binary**: The [tiny-gpu-compiler](https://github.com/gautam1858/tiny-gpu-compiler) compiles a **C-like GPU kernel language** down to **16-bit binary instructions** targeting tiny-gpu, an open-source GPU written in Verilog.
- **AMD Ryzen AI Surfaces**: After CES 2026, [AMD.com](https://www.amd.com/en/products/embedded/ryzen-ai/p100-series.html) released the new **AMD Ryzen AI**.


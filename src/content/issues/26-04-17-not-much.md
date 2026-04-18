---
id: MjAyNS0x
title: not much happened today
date: '2026-04-17T05:44:39.731046Z'
description: >-
  **Anthropic** launched **Claude Design**, a prototyping tool powered by
  **Claude Opus 4.7**, targeting design workflows and competing with **Figma**
  and others. Benchmarks show **Opus 4.7** leading in coding and text tasks,
  with improved efficiency and adaptive reasoning, though early user feedback
  noted some regressions and stability issues. Discussions highlighted its
  cost-efficiency and agentic capabilities compared to **Gemini 3.1 Pro** and
  **GPT-5.4**. Meanwhile, **OpenAI**'s Codex updates introduced advanced
  computer-use features enabling fast, agentic control of desktop apps and
  enterprise software, signaling progress toward practical AGI-like agents.
companies:
  - anthropic
  - openai
models:
  - claude-opus-4.7
  - gemini-3.1-pro
  - gpt-5.4
  - claude-code
  - codex
topics:
  - agentic-ai
  - model-benchmarking
  - adaptive-reasoning
  - cost-efficiency
  - computer-use
  - prototyping-tools
  - code-generation
  - model-performance
  - software-integration
people:
  - claudeai
  - yuchenj_uw
  - kimmonismus
  - skirano
  - therundownai
  - arena
  - artificialanlys
  - victortaelin
  - emollick
  - alexalbert__
  - theo
  - scaling01
  - reach_vb
  - kr0der
  - hamelhusain
  - mattrickard
  - matvelloso
  - gdb
---


**a quiet day.**

> AI News for 4/16/2026-4/17/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Anthropic’s Claude Opus 4.7 and Claude Design rollout**

- **Claude Design launched as Anthropic’s first design/prototyping surface**: [@claudeai](https://x.com/claudeai/status/2045156267690213649) announced **Claude Design**, a research-preview tool for generating prototypes, slides, and one-pagers from natural-language instructions, powered by **Claude Opus 4.7**. The launch immediately framed Anthropic as moving beyond chat/coding into design tooling; multiple observers called it a direct shot at **Figma/Lovable/Bolt/v0**, including [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2045158071950033063), [@kimmonismus](https://x.com/kimmonismus/status/2045162358004216134), and [@skirano](https://x.com/skirano/status/2045192705941106992). The market reaction itself became part of the story, with [@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2045161719547445426) and others noting Figma’s sharp drawdown after the announcement. Product details surfaced via [@TheRundownAI](https://x.com/TheRundownAI/status/2045176722476208454): inline refinement, sliders, exports to **Canva/PPTX/PDF/HTML**, and handoff to **Claude Code** for implementation.
- **Opus 4.7 looks stronger overall, but the rollout was noisy**: third-party benchmark posts were broadly favorable. [@arena](https://x.com/arena/status/2045177492936532029) put **Opus 4.7 #1 in Code Arena**, +37 over Opus 4.6 and ahead of non-Anthropic peers there; the same account also had it at **#1 overall in Text Arena** with category wins across coding and science-heavy domains [here](https://x.com/arena/status/2045177497378316597). [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2045292578434875552) reported a near three-way tie at the top of its **Intelligence Index**—**Opus 4.7 57.3**, **Gemini 3.1 Pro 57.2**, **GPT-5.4 56.8**—while also placing Opus 4.7 first on **GDPval-AA**, their agentic benchmark. They also noted **~35% fewer output tokens** than Opus 4.6 at higher score, and introduction of **task budgets** plus full removal of extended thinking in favor of adaptive reasoning. But user experience was mixed in the first 24 hours: [@VictorTaelin](https://x.com/VictorTaelin/status/2045139180359942462) reported regressions and context failures, [@emollick](https://x.com/emollick/status/2045147490316374414) said Anthropic had already improved adaptive thinking behavior by the next day, and [@alexalbert__](https://x.com/alexalbert__/status/2045159041283064095) confirmed that many initial bugs had been fixed. There were also complaints about product stability in Design itself from [@theo](https://x.com/theo/status/2045310884717981987) and account-level safety issues from the same account [here](https://x.com/theo/status/2045317666383204423).
- **Cost/efficiency discussion became almost as important as raw quality**: [@scaling01](https://x.com/scaling01/status/2045160883010081237) claimed **~10x fewer tokens** for some ML problem runs versus prior high-end models while maintaining similar performance, while [@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2045206342173086156) placed Opus 4.7 on the **price/performance Pareto frontier** for both text and code. Not every benchmark agreed on absolute leadership—e.g. [@scaling01](https://x.com/scaling01/status/2045178622617498084) noted it still trails **Gemini 3.1 Pro** and **GPT-5.4** on **LiveBench**—but the consensus from these posts is that Anthropic materially improved the model’s agentic utility and efficiency.

**Computer use, coding agents, and harness design**

- **Computer-use UX is becoming a mainstream product category**: OpenAI’s Codex desktop/computer-use updates drew unusually strong practitioner reactions. [@reach_vb](https://x.com/reach_vb/status/2045151640802771394) called **subagents + computer use** “pretty close” to AGI in practical feel; [@kr0der](https://x.com/kr0der/status/2045154074337710136), [@HamelHusain](https://x.com/HamelHusain/status/2045191726495846459), [@mattrickard](https://x.com/mattrickard/status/2045218583882633412), and [@matvelloso](https://x.com/matvelloso/status/2045209294942142860) all emphasized that Codex Computer Use is not just flashy but **fast**, able to drive **Slack, browser flows, and arbitrary desktop apps**, and may be the first genuinely usable computer-use platform for enterprise legacy software. [@gdb](https://x.com/gdb/status/2045375289560007029) explicitly framed Codex as becoming a **full agentic IDE**.
- **The field is converging on “simple harness, strong evals, model-agnostic scaffolding”**: several high-signal posts argued that reliability gains now come more from harnesses than from chasing the very largest models. [@AsfiShaheen](https://x.com/AsfiShaheen/status/2045072599508508914) described a three-stage financial analyst pipeline—**router / lane / analyst**—with strict context boundaries and gold sets for each stage, arguing that many bugs were actually instruction/interface bugs. [@AymericRoucher](https://x.com/AymericRoucher/status/2045176781414527305) extracted the same lesson from the leaked Claude Code harness: simple planning constraints plus a cleaner representation layer outperform “fancy AI scaffolds.” [@raw_works](https://x.com/raw_works/status/2045208764509470742) showed an even starker example: **Qwen3-8B** scored **33/507** on LongCoT-Mini with **dspy.RLM**, versus **0/507** vanilla, arguing the scaffold—not fine-tuning—did “100% of the lifting.” LangChain shipped more of these patterns into product: [@sydneyrunkle](https://x.com/sydneyrunkle/status/2045209395881980276) added **subagent support to `deepagents deploy`**, and [@whoiskatrin](https://x.com/whoiskatrin/status/2045139949939200284) announced **memory primitives in the Agents SDK**.
- **Open-source agent stacks continue to proliferate**: Hermes Agent remained a focal point. Community ecosystem overviews from [@GitTrend0x](https://x.com/GitTrend0x/status/2045142797439922337) highlighted derivatives like **Hermes Atlas**, **Hermes-Wiki**, HUDs, and control dashboards. [@ollama](https://x.com/ollama/status/2045282803387158873) then shipped **native Hermes support** via `ollama launch hermes`, which [@NousResearch](https://x.com/NousResearch/status/2045304840645939304) amplified. Nous and Kimi also launched a **$25k Hermes Agent Creative Hackathon** [@NousResearch](https://x.com/NousResearch/status/2045225469088326039), signaling a push from coding/productivity into **creative agent** workflows.

**Agent research: self-improvement, monitoring, web skills, and evaluation**

- **A cluster of papers pushed agent robustness and continual improvement forward**: [@omarsar0](https://x.com/omarsar0/status/2045139481779696027) summarized **Cognitive Companion**, which monitors reasoning degradation either with an LLM judge or a hidden-state **probe**. The headline result is notable: a **logistic-regression probe on layer-28 hidden states** can detect degradation with **AUROC 0.840** at **zero measured inference overhead**, while the LLM-monitor version cuts repetition **52–62%** with ~11% overhead. Separate work on web agents from [@dair_ai](https://x.com/dair_ai/status/2045139481892880892) described **WebXSkill**, where agents extract reusable skills from trajectories, yielding up to **+9.8 points on WebArena** and **86.1% on WebVoyager** in grounded mode. And [@omarsar0](https://x.com/omarsar0/status/2045241905227915498) also highlighted **Autogenesis**, a protocol for agents to identify capability gaps, propose improvements, validate them, and integrate working changes without retraining.
- **Open-world evals are becoming a serious theme**: several posts argued current benchmarks are too narrow. [@CUdudec](https://x.com/CUdudec/status/2045139195220431022) endorsed open-world evaluations for long-horizon, open-ended settings; [@ghadfield](https://x.com/ghadfield/status/2045245020429570505) connected this to regulation and “economy of agents” questions; and [@PKirgis](https://x.com/PKirgis/status/2045265295649231354) discussed **CRUX**, a project for regular **open-world evaluations** of AI agents in messy real environments. On the measurement side, [@NandoDF](https://x.com/NandoDF/status/2045063560716296450) proposed broad **NLL/perplexity-based eval suites** over out-of-training-domain books/articles across **2500 topic buckets**, though that sparked debate about whether perplexity remains informative after RLHF/post-training from [@eliebakouch](https://x.com/eliebakouch/status/2045115926123520100), [@teortaxesTex](https://x.com/teortaxesTex/status/2045139476972745120), and others.
- **Document/OCR and retrieval evals also got more agent-centric**: [@llama_index](https://x.com/llama_index/status/2045145054772183128) expanded on **ParseBench**, an OCR benchmark centered on **content faithfulness** with **167K+ rule-based tests** across omissions, hallucinations, and reading-order violations—explicitly reframing the bar from “human-readable” to “reliable enough for an agent to act on.” In retrieval, [@Julian_a42f9a](https://x.com/Julian_a42f9a/status/2045200413402493064) noted new work showing **late-interaction retrieval representations can substitute for raw document text in RAG**, suggesting some RAG pipelines may be able to bypass full-text reconstruction.

**Open models, local inference, and inference systems**

- **Qwen3.6 local/quantized workflows were a practical bright spot**: [@victormustar](https://x.com/victormustar/status/2045068986446958899) shared a concrete **llama.cpp + Pi** setup for **Qwen3.6-35B-A3B** as a local agent stack, emphasizing how viable local agentic systems now feel. Red Hat quickly followed with an **NVFP4-quantized Qwen3.6-35B-A3B** checkpoint [@RedHat_AI](https://x.com/RedHat_AI/status/2045153791402520952), reporting preliminary **GSM8K Platinum 100.69% recovery**, and [@danielhanchen](https://x.com/danielhanchen/status/2045169369723064449) benchmarked dynamic quants, claiming many Unsloth quants sit on the **Pareto frontier for KLD vs disk space**.
- **Consumer-hardware inference keeps improving**: [@RisingSayak](https://x.com/RisingSayak/status/2045114073000657316) announced work with **PyTorch/TorchAO** enabling **offloading with FP8 and NVFP4 quants** without major latency penalties, explicitly targeting consumer GPU users constrained by memory. Apple-side local inference also got a showcase with [@googlegemma](https://x.com/googlegemma/status/2045204738720084191), which demoed **Gemma 4 running fully offline on iPhone** with long context.
- **Inference infra updates worth noting**: [@vllm_project](https://x.com/vllm_project/status/2045381618928582995) highlighted **MORI-IO KV Connector** with AMD/EmbeddedLLM, claiming **2.5× higher goodput** on a **single node** via a PD-disaggregation-style connector. Cloudflare continued its agent/AI-platform push with **isitagentready.com** [@Cloudflare](https://x.com/Cloudflare/status/2045126394418503846), **Flagship** feature flags [@fayazara](https://x.com/fayazara/status/2045133183575113771), and **shared compression dictionaries** yielding dramatic payload reductions such as **92KB → 159 bytes** in one example [@ackriv](https://x.com/ackriv/status/2045177696506794336).

**AI for science, medicine, and infrastructure**

- **Scientific discovery and personalized health were prominent applied themes**: [@JoyHeYueya](https://x.com/JoyHeYueya/status/2045147082546462860) and [@Anikait_Singh_](https://x.com/Anikait_Singh_/status/2045149764636094839) posted about **insight anticipation**, where models generate a downstream paper’s core contribution from its “parent” papers; the latter introduced **GIANTS-4B**, an RL-trained model that reportedly beats frontier models on this task. On the health side, [@SRSchmidgall](https://x.com/SRSchmidgall/status/2045023895041061353) shared a biomarker-discovery system over wearable data whose first finding was that “**late-night doomscrolling**” predicts depression severity with **ρ=0.177, p<0.001, n=7,497**—notable because the model itself named the feature. Separately, [@patrickc](https://x.com/patrickc/status/2045164908912968060) argued current coding agents are already highly useful for **personalized genome interpretation**, describing <$100 analysis runs that surfaced a roughly **30× elevated melanoma predisposition** plus follow-on interventions.
- **Large-scale compute buildout remains a core meta-story**: [@EpochAIResearch](https://x.com/EpochAIResearch/status/2045258390147088764) surveyed all **7 US Stargate sites** and concluded the project appears on track for **9+ GW by 2029**, comparable to **New York City peak demand**. [@gdb](https://x.com/gdb/status/2045279841482928271) framed Stargate as infrastructure for a “**compute-powered economy**,” while [@kimmonismus](https://x.com/kimmonismus/status/2045206835238441332) put today’s annual global datacenter capex at roughly **5–7 Manhattan Projects per year** in inflation-adjusted terms.

**Top tweets (by engagement)**

- **Claude Design / Anthropic product expansion**: [@claudeai launches Claude Design](https://x.com/claudeai/status/2045156267690213649), by far the day’s biggest pure-AI product launch signal.
- **Model benchmarking / rankings**: [@ArtificialAnlys on Opus 4.7 tying for #1 overall and leading GDPval-AA](https://x.com/ArtificialAnlys/status/2045292578434875552).
- **Coding agents / computer use**: [@cursor_ai doubles Composer 2 limits in the new agents window](https://x.com/cursor_ai/status/2045236540784492845) and [@HamelHusain on Codex Computer Use](https://x.com/HamelHusain/status/2045191726495846459).
- **Open-source agents**: [@ollama ships native Hermes Agent support](https://x.com/ollama/status/2045282803387158873).
- **Applied AI in medicine**: [@patrickc on coding agents for genome analysis and personalized prevention](https://x.com/patrickc/status/2045164908912968060).
- **Infra / power scaling**: [@EpochAIResearch on Stargate’s 9+ GW trajectory](https://x.com/EpochAIResearch/status/2045258390147088764).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3.6 Model Launch and Features

  - **[Qwen3.6. This is it.](https://www.reddit.com/r/LocalLLaMA/comments/1so1533/qwen36_this_is_it/)** (Activity: 1483): **The post discusses the capabilities of **Qwen3.6**, a large language model, in autonomously building a tower defense game, identifying and fixing bugs such as canvas rendering issues and wave completion errors. The model is deployed using a `llama-server` setup with specific configurations, including `Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf` and `mmproj-F16.gguf` files, and operates with parameters like `--cpu-moe`, `--top-k 20`, and `--temp 0.7`. The user highlights the model's efficiency, achieving `120 tk/s` on an NVIDIA 3090 GPU, and its ability to quickly resolve coding issues that other models struggled with.** Commenters express amazement at the model's performance, noting its potential impact on future generations and its efficiency compared to other models like Gemma. There is interest in the technical stack used for deployment, indicating a desire for similar local setups.

    - **cviperr33** highlights the impressive performance of the Qwen3.6 model, noting its ability to fix broken code quickly and efficiently. They report achieving `120 tokens/second` on an `NVIDIA 3090` using `llama.ccp`, with instant prefill in the `3.8k-5k` token range. This speed allows for rapid responses and efficient file editing without overloading the GPU, contrasting with the slower performance of the Gemma models.
    - **PotatoQualityOfLife** inquires about the specific size or quantization of the model being used, which is a critical factor in understanding the model's performance and resource requirements. This question suggests a focus on optimizing the model's deployment for local setups, which can significantly impact speed and efficiency.
    - **No-Marionberry-772** expresses interest in setting up a local environment for running models like Qwen3.6 but faces challenges in selecting the appropriate software stack. This reflects a common issue among users trying to leverage advanced models locally, indicating a need for clearer guidance or resources on optimal configurations.

  - **[Qwen 3.6 is the first local model that actually feels worth the effort for me](https://www.reddit.com/r/LocalLLaMA/comments/1so2nt9/qwen_36_is_the_first_local_model_that_actually/)** (Activity: 512): **The user reports that the `qwen3.6-35b-a3b` model is the first local model that feels efficient and worthwhile for their projects, particularly in UI XML for Avalonia and embedded systems C++. Running on a `5090 + 4090` setup, the model achieves `170 tokens per second` with a `260k context`, outperforming other models like Gemma 4 by requiring minimal corrections. This suggests significant improvements in local model capabilities, potentially reducing reliance on cloud-based solutions.** The comments reflect a divided opinion on the model's performance, with some users expressing skepticism about its capabilities and others noting a polarized reception post-release.

    - -Ellary- highlights the performance differences between Qwen 3.6 and other models, noting that Qwen 3.5 27b is superior in task execution and problem-solving. They suggest that if hardware resources allow, running the full GLM 4.7 358B A32B at IQ4XS or IQ3XXS would yield significantly better results compared to Qwen 3.6 35b A3B, which they consider a lighter model akin to 9-12b dense models.
    - kmp11 mentions the impressive performance of the Hermes-Agent when paired with Qwen 3.6, noting its ability to handle an unlimited number of tokens at speeds exceeding 100 tokens per second. This suggests a high level of efficiency and capability in processing large volumes of data quickly, which could be beneficial for applications requiring rapid token processing.

  - **[Qwen3.6 is incredible with OpenCode!](https://www.reddit.com/r/LocalLLaMA/comments/1so3rsx/qwen36_is_incredible_with_opencode/)** (Activity: 436): **The post discusses the performance of **Qwen3.6**, a local AI model, when deployed using `llama.cpp` on an **RTX 4090** with `24 GB` VRAM. The user tested the model on a complex task involving the implementation of RLS in PostgreSQL across a codebase with services in Rust, TypeScript, and Python. Despite some bugs, the model performed well, iterating on compiler errors and optimizing code changes. The setup included **Qwen3.6-35B-A3B, IQ4_NL unsloth quant**, with a context size of `262k` and VRAM usage around `21GB`. The deployment used **docker** with specific settings to prevent OOM errors, achieving `100+ output tokens per second`.** Commenters expressed regret over hardware limitations, such as having only `16GB` VRAM, and shared positive experiences with Qwen3.6, noting its ability to handle complex tasks involving multiple subagents and tool calls. Some issues were noted, such as subagents not saving outputs and presentation errors, but these were resolved with iterations.

    - Durian881 shared a detailed experience using Qwen 3.6 with Qwen Code, highlighting its ability to handle complex tasks involving 'McKinsey-research skill' with 9-12 subagents and extensive tool calls like websearch and webfetch. The process took over 1.5 hours, and despite some issues with subagents not saving outputs and slide rendering errors, the model was able to recover and produce high-quality HTML slides. These fixes were compared to those made by Gemini 3 Pro, which also had similar issues with slide ordering and title pages.
    - robertpro01 compared Qwen 3.6 to Gemini 3 Flash, noting that its performance is on par with the latter, which implies that users might not need to pay for Gemini 3 Flash if they can use Qwen 3.6 effectively. This suggests that Qwen 3.6 offers competitive performance at potentially lower costs, making it an attractive option for users seeking cost-effective solutions.
    - RelicDerelict inquired about running Qwen 3.6 on a system with 4GB VRAM and 32GB RAM, indicating interest in understanding the hardware requirements for optimal performance. This highlights a common concern among users with limited hardware resources, seeking to leverage advanced models like Qwen 3.6 without needing high-end equipment.

  - **[Qwen3.6-35B-A3B released!](https://www.reddit.com/r/LocalLLaMA/comments/1sn3izh/qwen3635ba3b_released/)** (Activity: 3494): **The image showcases the performance of the newly released **Qwen3.6-35B-A3B**, a sparse MoE model with `35B` total parameters and `3B` active parameters, highlighting its competitive edge in various benchmarks. This model, released under the Apache 2.0 license, demonstrates agentic coding capabilities comparable to models ten times its active size and excels in multimodal perception and reasoning. The bar charts in the image illustrate Qwen3.6-35B-A3B's superior performance in tasks such as coding and reasoning, outperforming both the dense 27B-param Qwen3.5-27B and its predecessor Qwen3.5-35B-A3B, particularly in agentic coding and reasoning tasks. [View Image](https://i.redd.it/g6edjlxt0kvg1.jpeg)** Commenters note the impressive performance of Qwen3.6-35B-A3B, particularly in coding benchmarks, and express anticipation for future releases that could challenge major models from companies like Google.

    - Qwen3.6-35B-A3B demonstrates significant improvements over its predecessors, particularly in coding and reasoning tasks. It outperforms the dense 27B-param Qwen3.5-27B on several key coding benchmarks and dramatically surpasses Qwen3.5-35B-A3B, especially in agentic coding and reasoning tasks, indicating a substantial leap in performance for local LLMs.
    - The Qwen3.6-35B-A3B model is natively multimodal, showcasing advanced perception and multimodal reasoning capabilities. Despite having only around 3 billion activated parameters, it performs exceptionally well on vision-language benchmarks, matching or surpassing Claude Sonnet 4.5 in several tasks. Notably, it achieves a score of 92.0 on RefCOCO and 50.8 on ODInW13, highlighting its strengths in spatial intelligence.
    - There is anticipation for the release of a larger Qwen3.6 model, potentially a 122B version, which could pressure competitors like Google to release their own large models. This competition could bring models like GLM 5.1 and Sonnet 4.6 into closer comparison, suggesting a rapidly evolving landscape in large-scale model development.


### 2. Qwen3.6 Benchmarks and Performance

  - **[Qwen3.6 GGUF Benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1so5nrl/qwen36_gguf_benchmarks/)** (Activity: 588): **The image is a performance benchmark graph for Qwen3.6 GGUF, illustrating the Mean KL Divergence against disk space for various quantization providers. The graph highlights that **Unsloth** quants dominate the Pareto frontier, achieving the best trade-off between KL Divergence and disk space in 21 out of 22 cases. This suggests that Unsloth's quantization models are highly efficient in terms of performance and storage. The post also addresses misunderstandings about frequent updates, clarifying that most issues stem from external factors, and highlights a confirmed bug in CUDA 13.2 affecting low-bit quantizations, with a fix expected in CUDA 13.3.**

    - **danielhanchen** highlights a critical issue with CUDA 13.2, where all 4-bit quantizations produce gibberish outputs. This problem affects all quant providers and is confirmed to be resolved in the upcoming CUDA 13.3 release, as noted by NVIDIA in a [GitHub issue comment](https://github.com/ggml-org/llama.cpp/issues/21255#issuecomment-4248403175). Users experiencing this issue are advised to revert to CUDA 13.1 as a temporary workaround.
    - **tavirabon** critiques the selective presentation of data in the benchmarks, suggesting that the analysis uses percentages to favorably represent the models affected by issues. The comment also mentions a perceived bias in the analysis, particularly in how it addresses competition, specifically mentioning a campaign against Bartowski, which seems out of context and affects the perceived neutrality of the analysis.
    - **PiratesOfTheArctic** appreciates the clarity of the graphical data representation, which simplifies understanding for those less familiar with the technical details. This suggests that the visual aids provided in the benchmarks are effective in communicating complex information to a broader audience.

  - **[Ternary Bonsai: Top intelligence at 1.58 bits](https://www.reddit.com/r/LocalLLaMA/comments/1snqo1f/ternary_bonsai_top_intelligence_at_158_bits/)** (Activity: 532): ****Ternary Bonsai** is a new family of language models by **PrismML**, designed to operate at `1.58 bits` per weight using ternary weights {-1, 0, +1}. This approach allows the models to maintain a memory footprint approximately `9x smaller` than traditional 16-bit models while achieving superior performance on standard benchmarks. The models are available in sizes of `8B`, `4B`, and `1.7B` parameters, and are accessible via [Hugging Face](https://huggingface.co/collections/prism-ml/ternary-bonsai). The release includes FP16 safetensors for compatibility with existing frameworks, although the **MLX 2-bit format** is currently the only packed format available, with more formats expected soon. For more details, see the [official blog post](https://prismml.com/news/ternary-bonsai).** Some commenters question the presentation of model sizes, suggesting that quantizing larger models with Q4 could reduce size differences without significant performance loss. Others express anticipation for larger models, such as 20-40B parameters, which could significantly impact the field.

    - r4in311 and DefNattyBoii discuss the potential for misleading comparisons in model benchmarks, noting that showing full weights of 8B/9B models without considering quantization (e.g., Q4) can exaggerate size differences. They suggest that quantized models could maintain performance while reducing size, and criticize the use of outdated models like Qwen3 in benchmarks, advocating for comparisons with newer models such as Qwen3.5 and Gemma4.
    - DefNattyBoii raises concerns about the lack of collaboration with mainstream inference frameworks like `llama.cpp`, `vllm`, and `sglang`, suggesting that this could limit the practical applicability and integration of the models being discussed. This lack of integration might hinder the adoption and performance optimization of these models in real-world applications.
    - Kaljuuntuva_Teppo highlights the limitations of current models in utilizing consumer-grade GPUs with 24-32 GB of memory. They express a desire for models that can better leverage this hardware, suggesting that current models are too small to fully utilize the available resources, which could lead to inefficiencies in performance and resource usage.


### 3. Qwen3.6 Uncensored Aggressive Variant

  - **[Qwen3.6-35B-A3B Uncensored Aggressive is out with K_P quants!](https://www.reddit.com/r/LocalLLaMA/comments/1snlo6s/qwen3635ba3b_uncensored_aggressive_is_out_with_k/)** (Activity: 433): **The **Qwen3.6-35B-A3B Uncensored Aggressive** model has been released, featuring the same `35B` MoE size as the previous 3.5-35B but based on the newer 3.6 architecture. This variant is fully uncensored with **0/465 refusals** and no personality alterations, maintaining full capability without degradation. It includes various quantization formats like `Q8_K_P`, `Q6_K_P`, and others, generated using **imatrix** for optimized performance. The model supports multimodal inputs (text, image, video) and features a hybrid attention mechanism with a `3:1` linear to softmax ratio across `40 layers`. It is compatible with platforms like `llama.cpp` and `LM Studio`, though some GUI labels may not display correctly due to custom quant naming.** Commenters express skepticism about the claim of no quality degradation in an uncensored model and criticize the use of unique quant naming conventions, which can disrupt GUI compatibility. There is also a call for more transparency regarding the testing methods for 'zero capability loss.'

    - A user expressed skepticism about the claim of 'zero capability loss' in the Qwen3.6-35B-A3B Uncensored Aggressive model, noting that typically uncensored models suffer from quality degradation. This highlights the need for detailed testing methodologies and benchmarks to substantiate such claims, as the commenter points out the lack of information on how these tests were conducted.
    - Another commenter criticized the use of new terminology for custom quantizations, suggesting that the description aligns with existing methods like 'imatrix'. They argue that inventing new terms for established techniques can cause confusion and compatibility issues with GUIs that rely on standard naming conventions, advocating for the use of more universally recognized labels like 'K_L' or 'K_XL'.
    - There was a mention of the limited availability of quantization files for download, indicating that the release might be incomplete or still in progress. This suggests that users looking to experiment with the model might face delays or need to wait for the full set of files to be uploaded.

  - **[Qwen3.6-35B-A3B Uncensored Aggressive is out with K_P quants!](https://www.reddit.com/r/LocalLLM/comments/1snlo1x/qwen3635ba3b_uncensored_aggressive_is_out_with_k/)** (Activity: 357): **The **Qwen3.6-35B-A3B Uncensored Aggressive** model has been released, featuring the same `35B` MoE size as the previous 3.5-35B but based on the newer 3.6 architecture. This variant is fully uncensored with **0/465 refusals**, maintaining full capability without personality alterations. It includes various quantization formats like `Q8_K_P`, `Q6_K_P`, and others, optimized for quality with a slight increase in file size. The model supports multimodal inputs (text, image, video) and uses a hybrid attention mechanism. It is compatible with platforms like `llama.cpp` and `LM Studio`, though some cosmetic issues may appear in the latter. For more details, see the [Hugging Face model page](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive).** A user questioned the meaning of 'no personality changes,' implying curiosity about the model's behavior. Another user expressed appreciation for the consistent quality of these releases, indicating a preference for this developer's models.

    - The model name 'Qwen3.6-35B-A3B' indicates specific characteristics: 'Qwen' is the model family, '3.6' likely refers to the version, '35B' denotes the number of parameters (35 billion), and 'A3B' could indicate a specific architecture or training configuration. The 'K_P' quantization refers to a method of reducing model size while maintaining performance, though the exact meaning of 'K_P' isn't universally defined and may vary by context.
    - Regarding hardware compatibility, a user inquires if the 'q3' quantized version of the model would run efficiently on a 24GB NVIDIA 4090 GPU. The 'q3' quantization suggests a lower precision format that reduces memory usage, potentially allowing the model to fit within the GPU's memory constraints. However, there is concern about whether this quantization significantly degrades model quality, which can vary depending on the specific implementation and use case.
    - The term 'no personality changes' likely refers to the model's behavior remaining consistent across different versions or configurations. This implies that despite updates or changes in quantization, the model's responses and interaction style should remain stable, ensuring reliability in applications where consistent behavior is critical.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
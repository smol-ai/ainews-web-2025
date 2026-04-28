---
id: MjAyNS0x
title: not much happened today
date: '2026-04-27T05:44:39.731046Z'
description: >-
  **OpenAI** loosens its **Azure exclusivity**, allowing distribution across
  **Google TPU**, **AWS Trainium**, and **Bedrock** with commitments through
  **2032** and revenue share through **2030**. **GPT-5.5** shows improved
  benchmarks but is not uniformly dominant, ranking variably across coding,
  document, math, and vision tasks. GitHub's **Copilot** shifts to usage-based
  billing starting June 1, reflecting increased runtime costs. **OpenAI**
  open-sourced **Symphony**, an orchestration layer for issue tracking and Codex
  agents. **Xiaomi** released **MiMo-V2.5** and **MiMo-V2.5-Pro**, large context
  models with up to **1M-token context** and trillions of tokens trained,
  emphasizing complex agent and omni-modal capabilities. **Kimi K2.6** leads
  OpenRouter's leaderboard, noted for coding and long-horizon agent capabilities
  with large-scale sub-agent coordination.
companies:
  - openai
  - microsoft
  - google
  - amazon
  - github
  - xiaomi
  - openai-devs
  - vllm_project
  - kimi-moonshot
models:
  - gpt-5.5
  - gpt-5.4
  - opus-4.7
  - mimo-v2.5-pro
  - mimo-v2.5
  - kimi-k2.6
  - codex
  - copilot
topics:
  - model-distribution
  - cloud-computing
  - benchmarking
  - usage-based-billing
  - model-orchestration
  - open-source
  - large-context-models
  - agent-scaling
  - coding
  - model-training
  - fp8
  - attention-mechanisms
  - multi-agent-systems
people:
  - sama
  - scaling01
  - kimmonismus
  - ajassy
  - simonw
  - htihle
  - arena
  - gdb
  - hangsiin
  - eliebakouch
  - _luofuli
  - teortaxestex
---


**a quiet day.**

> AI News for 4/26/2026-4/27/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**OpenAI Distribution Shift, GPT-5.5 Benchmarks, and Codex/Copilot Pricing Signals**

- **OpenAI loosens Azure exclusivity**: [@sama](https://x.com/sama/status/2048755148361707946) said OpenAI updated its Microsoft partnership so Microsoft remains the **primary cloud**, but OpenAI can now make products available **across all clouds**, with product/model commitments extending to **2032** and revenue share through **2030**. The implication was quickly drawn by [@scaling01](https://x.com/scaling01/status/2048752418305769473) and [@kimmonismus](https://x.com/kimmonismus/status/2048759615500804395): OpenAI can now distribute via **Google TPU / AWS Trainium / Bedrock**, and Microsoft’s license to OpenAI IP becomes **non-exclusive**. [@ajassy](https://x.com/ajassy/status/2048806022253609115) confirmed **OpenAI models are coming to AWS Bedrock** in the coming weeks. [@simonw](https://x.com/simonw/status/2048834476323823983) noted the new language likely means the old **AGI clause is effectively gone**.
- **GPT-5.5 is a broad upgrade, but not uniformly dominant**: Community evals from [@htihle](https://x.com/htihle/status/2048717753394090274) put **GPT-5.5 no-thinking at 67.1% on WeirdML**, up from **57.4% for GPT-5.4**, but still behind **Opus 4.7 no-thinking at 76.4%** while using fewer tokens. LMSYS Arena results from [@arena](https://x.com/arena/status/2048794479646388732) placed GPT-5.5 at **#9 in Code Arena**, **#6 Document**, **#7 Text**, **#3 Math**, **#2 Search**, **#5 Vision**, with [Expert Arena #5](https://x.com/arena/status/2048808366810800259). Arena also clarified current evaluation covers **medium/high reasoning**, with **xHigh still pending** ([1](https://x.com/arena/status/2048820224938631492), [2](https://x.com/arena/status/2048846896744247468)). Practitioner feedback was positive for hard coding tasks such as GPU kernels from [@gdb](https://x.com/gdb/status/2048777802586149331), but there were also reports of “compressed CoT leakage” / malformed outputs in no-thinking mode from [@htihle](https://x.com/htihle/status/2048741770125603304).
- **Developer economics are becoming more explicit**: GitHub announced [Copilot moves to usage-based billing on June 1](https://x.com/github/status/2048794729274278258), a notable shift as agentic workflows consume much more runtime. Parallel to that, [@Hangsiin](https://x.com/Hangsiin/status/2048719057885818902) documented Codex usage multipliers: **GPT-5.4 fast = 2x**, **GPT-5.5 fast = 2.5x**, with 5.4-mini and GPT-5.3-Codex materially cheaper. [@sama](https://x.com/sama/status/2048913887614115857) argued **Codex at $20** remains a strong value. OpenAI also open-sourced **Symphony**, an orchestration layer connecting issue trackers to Codex agents for “open issue → agent → PR → human review,” via [@OpenAIDevs](https://x.com/OpenAIDevs/status/2048825010371039648).

**Xiaomi MiMo-V2.5, Kimi K2.6, and China’s Agent-Oriented Open-Weights Push**

- **MiMo-V2.5 is one of the day’s biggest open releases**: [@XiaomiMiMo](https://x.com/XiaomiMiMo/status/2048821516079661561) open-sourced **MiMo‑V2.5-Pro** and **MiMo‑V2.5** under **MIT**, both with **1M-token context**. The Pro model is framed as a **complex agent/coding** model and the smaller model as a **native omni-modal agent**. Community summaries from [@eliebakouch](https://x.com/eliebakouch/status/2048845602633433258) add useful technical details: **MiMo‑V2.5-Pro** is roughly **1T total / 42B active**, trained on **27T tokens in FP8**, while **MiMo‑V2.5** is about **310B total / 15B active**, trained on **48T tokens**, with aggressive **interleaved SWA/global attention** and no shared expert. Xiaomi also announced a **100T token grant** for builders via [@_LuoFuli](https://x.com/_LuoFuli/status/2048851054662762618). Day-0 inference support landed quickly in [vLLM](https://x.com/vllm_project/status/2048825703244972375) and [SGLang/vLLM](https://x.com/XiaomiMiMo/status/2048821520798302409).
- **Kimi K2.6 continues to lead in mindshare and deployment**: [@Kimi_Moonshot](https://x.com/Kimi_Moonshot/status/2048693682329776223) said **Kimi K2.6** is now **#1 on OpenRouter’s weekly leaderboard**. Secondary reporting described it as a model for **coding and long-horizon agents**, including scaling to **300 concurrent sub-agents across 4,000 coordinated steps** ([dl_weekly](https://x.com/dl_weekly/status/2048764506105348129)). Practitioners remain split on speed/quality tradeoffs: [@teortaxesTex](https://x.com/teortaxesTex/status/2048820805258059837) found Kimi in Hermes much slower than DeepSeek V4 but sometimes capable of fixing bugs V4 could not.
- **Broader China-model trend**: Multiple posts framed Chinese labs as pushing aggressively on **open-ish, agent-oriented, long-context systems**: [Qwen 3.6 Flash](https://x.com/scaling01/status/2048730112636473792), DeepSeek V4/Flash, GLM-5.1 promotions ([triple usage extension](https://x.com/Zai_org/status/2048784274523148750)), and Xiaomi’s MIT release. A recurring theme was that smaller / cheaper variants are often outperforming their larger siblings on practical agent benchmarks.

**Agent Runtimes, Orchestration, and Local-First Tooling**

- **Sakana’s Conductor is a notable multi-agent result**: [@SakanaAILabs](https://x.com/SakanaAILabs/status/2048777689763639741) introduced a **7B Conductor** trained with RL to orchestrate a pool of frontier models in natural language rather than solving tasks directly. It dynamically decides **which agent to call, what subtask to assign, and which context to expose**, and reportedly reached **83.9% on LiveCodeBench** and **87.5% on GPQA-Diamond**, beating any single worker in its pool. [@hardmaru](https://x.com/hardmaru/status/2048778095935795338) highlighted “**AI managing AI**” and recursive self-selection as a new axis of **test-time scaling**.
- **Local and hybrid agents keep getting better**: Several posts showed coding/assistant stacks running locally. [@patloeber](https://x.com/patloeber/status/2048715918541558075) and [@_philschmid](https://x.com/_philschmid/status/2048719354905108623) documented running **Pi agent + Gemma 4 26B A4B** locally via LM Studio/Ollama/llama.cpp. [@googlegemma](https://x.com/googlegemma/status/2048805789788413984) demoed a **fully local browser agent** using **Gemma 4 + WebGPU**, with native tool calling for browsing history, tab management, and page summarization. [@cognition](https://x.com/cognition/status/2048821234281181302) shipped **Devin for Terminal**, a local shell agent that can later **hand off to the cloud**.
- **Agent ergonomics and framework evolution**: Hermes had a strong day: [@Teknium](https://x.com/Teknium/status/2048710115885523444) noted **Hermes Agent’s repo surpassed Claude Code**, while [native vision became the default when supported](https://x.com/Teknium/status/2048766822766547451). The broader ecosystem kept filling in missing pieces: [Cline Kanban](https://x.com/cline/status/2048814649513275448) now supports **different agents/models per task card**; [Future AGI](https://x.com/omarsar0/status/2048759865007591615) open-sourced an eval/optimization stack for self-improving agents; and [@_philschmid](https://x.com/_philschmid/status/2048781492914885079) argued MCP works best either through **explicit @mention loading** or **subagent-scoped tool assignment**, not indiscriminate server attachment.

**Inference Infrastructure, Attention/KV Engineering, and Systems Work**

- **Google’s TPU split is a meaningful architecture signal**: Several posts dissected Google’s Cloud Next announcement that **TPU v8 is split into 8t for training and 8i for inference**, with claims of roughly **2.8x faster training** and **80% better inference performance/$** than prior generation. [@kimmonismus](https://x.com/kimmonismus/status/2048745304007299230) emphasized this is the first time Google split custom silicon by workload and that OpenAI, Anthropic, and Meta are reportedly buying TPU capacity.
- **DeepSeek V4 support is maturing quickly in infra stacks**: [@vllm_project](https://x.com/vllm_project/status/2048769886483329525) said support for **DeepSeek V4 base models** is coming, requiring an `expert_dtype` config field to distinguish **FP4 instruct vs FP8 base**. In the [vLLM 0.20.0 release](https://x.com/vllm_project/status/2048918629144805619), highlights included **DeepSeek V4 support**, **FA4 as default MLA prefill**, **TurboQuant 2-bit KV**, and a DeepSeek-specific **MegaMoE** path on Blackwell.
- **KV cache optimization remains a hot battleground**: There was dense discussion around long-context bottlenecks and KV strategies. [@cHHillee](https://x.com/cHHillee/status/2048756662845022655) summarized three main levers for long contexts: **local/sliding attention**, **interleaved local-global attention**, and **smaller KV per global layer** via **GQA/MLA/KV tying/quantization**. On the implementation side, [@vllm_project](https://x.com/vllm_project/status/2048796304508330462) and Red Hat/AWS published an FP8 KV-cache deep dive where a fix to **FA3 two-level accumulation** improved **128k needle-in-a-haystack from 13% to 89%** while retaining FP8 decode speedups. Community critics also questioned DeepSeek V4’s specific KV tradeoffs relative to offloading-heavy approaches such as HiSparse ([discussion](https://x.com/Grad62304977/status/2048785005216723072)).

**Benchmarks, Evals, and Open Research Directions**

- **Open-world evaluation is gaining momentum**: [@sarahookr](https://x.com/sarahookr/status/2048731841759428935) argued that most agentic benchmarks are overfit to **automatically verifiable** tasks, while the important frontier is **open-world, uncertain, non-fully-verifiable** work. Related threads connected this to **continual learning**, memory stores, and adaptive data systems ([1](https://x.com/sarahookr/status/2048759884125233453), [2](https://x.com/adaption_ai/status/2048771654008877400)).
- **Cost-aware agent evaluation is becoming first-class**: [@dair_ai](https://x.com/dair_ai/status/2048784506635878644) highlighted a new study on coding-agent spend over SWE-bench Verified: agentic coding can consume **~1000x more tokens** than chat/code reasoning, usage can vary **30x** across runs on identical tasks, and more spending does **not** monotonically improve accuracy. This lines up with pricing-model changes from Copilot and growing concern over uncontrolled agent runtime economics.
- **New benchmarks and domain-specific evals**: [ParseBench](https://x.com/osanseviero/status/2048777802015535189) from LlamaIndex adds **2k verified enterprise document pages** for parsing agents. [AgentIR](https://x.com/CShorten30/status/2048764263196500002) reframes retrieval for research agents by embedding the **reasoning trace alongside the query**, with **AgentIR-4B hitting 68% on BrowseComp-Plus vs 52% for larger conventional embedding models**. There were also several benchmark snapshots for frontier models—e.g. [Opus 4.7 leading GSO at 42.2%](https://x.com/scaling01/status/2048853227211251891) and WeirdML / ALE-Bench / PencilPuzzleBench chatter—but the stronger signal was methodological: more people are measuring **runtime cost, retrieval quality, and open-world behavior**, not just final answer accuracy.

**Top tweets (by engagement)**

- **OpenAI–Microsoft partnership reset**: [@sama](https://x.com/sama/status/2048755148361707946) on cross-cloud availability and continued Microsoft partnership.
- **OpenAI on AWS**: [@ajassy](https://x.com/ajassy/status/2048806022253609115) confirming OpenAI models are coming to **Bedrock**.
- **GitHub Copilot pricing change**: [@github](https://x.com/github/status/2048794729274278258) announcing **usage-based billing** starting June 1.
- **Xiaomi MiMo-V2.5 open-source release**: [@XiaomiMiMo](https://x.com/XiaomiMiMo/status/2048821516079661561) with **MIT license** and **1M context**.
- **Open-source orchestration for Codex**: [@OpenAIDevs](https://x.com/OpenAIDevs/status/2048825010371039648) launching **Symphony**.
- **Gemma local browser agent**: [@googlegemma](https://x.com/googlegemma/status/2048805789788413984) showing a **100% local browser-resident agent** with WebGPU.


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap



## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
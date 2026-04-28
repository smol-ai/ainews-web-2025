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

### 1. Qwen3.6 Model Performance and Optimization

  - **[Luce DFlash: Qwen3.6-27B at up to 2x throughput on a single RTX 3090](https://www.reddit.com/r/LocalLLaMA/comments/1sx8uok/luce_dflash_qwen3627b_at_up_to_2x_throughput_on_a/)** (Activity: 743): ****Luce DFlash** is a new implementation of speculative decoding for the Qwen3.6-27B model, optimized to run on a single RTX 3090 GPU using a standalone C++/CUDA stack built on top of ggml. This setup achieves up to `1.98x` throughput compared to autoregressive decoding across benchmarks like HumanEval, GSM8K, and Math500, without requiring retraining. The system uses a compressed KV cache and sliding-window flash attention to efficiently handle large contexts, and it can serve over an OpenAI-compatible HTTP endpoint. The implementation is constrained to CUDA environments and does not support multi-GPU setups.** Commenters are enthusiastic about the innovation, with one noting the potential for local AI inference and another expressing interest in the setup due to its speed advantages over existing configurations.

    - Tiny_Arugula_5648 raises a critical point about the impact of quantization on model accuracy. They emphasize that while the increased throughput is impressive, the heavy quantization involved can significantly affect the model's performance in certain applications, such as coding or tool calling, where precision is crucial. This highlights the importance of understanding the trade-offs between speed and accuracy when deploying such models.
    - drrck82 expresses interest in the setup, particularly as a dual RTX 3090 owner. They mention currently using the Q6_K_XL model for enhanced intelligence but find the potential for doubling speed with the Qwen3.6-27B model very appealing. This suggests a focus on balancing computational efficiency with model sophistication.
    - DeepV inquires about the possibility of dockerizing the setup, indicating a demand for containerized solutions that can simplify deployment and scaling. Dockerization could facilitate easier integration into existing workflows and enhance reproducibility across different environments.

  - **[To 16GB VRAM users, plug in your old GPU](https://www.reddit.com/r/LocalLLaMA/comments/1swzjnu/to_16gb_vram_users_plug_in_your_old_gpu/)** (Activity: 666): **The post discusses leveraging an old GPU with at least `6GB VRAM` alongside a primary `16GB VRAM` GPU to run dense models like `30b` models more efficiently. The author uses a `5070Ti 16GB` and an old `2060 6GB`, achieving a combined `22GB VRAM`, which approaches the capacity of a `24GB` class card. The setup involves using `llama-server` with specific configurations to optimize GPU usage, such as `dev=Vulkan1,Vulkan2` to enable both GPUs, and `no-mmap` to keep the model off RAM. Benchmark results show significant performance improvements, with `186 tokens/s` for prompt processing and `19 tokens/s` for generation at `128k` max context, compared to `4 tokens/s` on a single card. The post also provides detailed `llama-bench` results using CUDA, highlighting the importance of fitting models on GPU VRAM for speed, especially at long contexts.** Commenters debate the use of Vulkan versus CUDA, with some suggesting CUDA for better performance. Others note that while additional VRAM from a weaker GPU can help, it may bottleneck a stronger GPU, as seen with a `3090 Ti` and `2070` setup, where splitting across GPUs reduced speed compared to using the `3090 Ti` alone.

    - **Mysterious_Role_8852** discusses the performance bottleneck when using a 3090 Ti and a 2070 together. They note that the 2070 bottlenecks the 3090, resulting in a decrease in performance from `30t/s` to `20t/s` when splitting tasks between the GPUs. This highlights the importance of matching GPU capabilities to avoid performance degradation, especially when handling large models like Qwen 3.6 27b Q6 Quant.
    - **mac1e2** provides a detailed account of running Qwen3.6-35B-A3B on a constrained system with a GTX 1650 4GB and 62GB RAM. They emphasize the importance of understanding hardware limitations and optimizing configurations, such as using `--cpu-moe`, `--mlock`, and specific cache settings. The post underscores the value of disciplined resource management, contrasting it with modern practices that often rely on abundant hardware resources without optimizing for efficiency.
    - **jacek2023** mentions using a 3060 as an additional GPU alongside three 3090s, but only for the largest models. This suggests a strategic approach to leveraging available hardware, where additional GPUs are utilized selectively to maximize performance for specific tasks, rather than a blanket approach of using all available resources indiscriminately.

  - **[Switched from Qwen3.6 35b-a3b to Qwen3.6 27b mid coding and it's noticeably better!](https://www.reddit.com/r/LocalLLaMA/comments/1swifke/switched_from_qwen36_35ba3b_to_qwen36_27b_mid/)** (Activity: 437): **The image is a screenshot from a tower defense game developed using Qwen3.6 models, specifically highlighting the transition from the 35b-a3b to the 27b version. The user experienced improved performance with the 27b model despite using a more compressed IQ3_M version due to VRAM limitations. This suggests that dense models like Qwen3.6-27B-i1-GGUF handle compression better than MoE models, as evidenced by the model's ability to identify a difficult bug that the larger model couldn't. The user reports token processing speeds of 40 tokens per second with the 27b model, which maintained consistent speed compared to the 35b-a3b model's fluctuating performance.** Commenters generally agree that dense models like Qwen3.6-27B are more efficient and reliable for users with limited VRAM, and they appreciate the availability of such models for local use without cloud dependency. Some users have successfully used these models for legitimate work, noting their satisfactory speed and context window.

    - The user 'ridablellama' highlights the practical utility of the Qwen3.6 27b model, noting that while it may not match the performance of Claude Code, it is still capable of handling legitimate work tasks effectively. They emphasize the model's speed and context window as satisfactory, suggesting that with further fine-tuning, its performance could improve significantly. This underscores the model's potential as a reliable baseline for users with 16-24 GB VRAM, offering a cost-effective alternative to cloud-based solutions.
    - 'YairHairNow' provides a detailed comparison of different Qwen3.6 model configurations, focusing on token generation speed and context length. The 27B IQ4_XS model is highlighted for its long-context mode with a maximum context of 196K and a token generation speed of 48 tokens per second. In contrast, the 35B-A3B Q3_K_S configuration offers a faster token generation speed of up to 149 tokens per second but with a shorter context length of around 65K. This comparison is valuable for users deciding between speed and context capabilities.
    - 'KillerX629' mentions a noticeable slowdown when switching models, which affects their workflow due to reduced tokens per second. This highlights a common trade-off in model selection between performance speed and other factors such as model size or context capabilities. The comment suggests that users need to balance these aspects based on their specific requirements and hardware capabilities.

  - **[Qwen3.6-27B-INT4 clocking 100 tps with 256k context length on 1x RTX 5090 via vllm 0.19](https://www.reddit.com/r/LocalLLaMA/comments/1sw21op/qwen3627bint4_clocking_100_tps_with_256k_context/)** (Activity: 426): **The post discusses the performance of the **Qwen3.6-27B-INT4** model, achieving `105-108 tokens per second (tps)` with a `256k context length` on a single **RTX 5090** using **vllm 0.19**. The model, available on [Hugging Face](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound), benefits from **MTP support** and a smaller size, allowing full native context window utilization. The setup uses `auto_round` quantization and `fp8_e4m3` for KV cache dtype, with a focus on interactivity and speculative decoding. The configuration includes `flashinfer` attention backend and `mtp` speculative configuration with `3 speculative tokens`.** A notable comment highlights the use of **Turboquant 3-bit NC KV Cache** for compressing KV state, enabling a `125K context window` within `24GB VRAM`. The **MTP n=3 Speculative Decoding** is praised for its throughput multiplier, and **Cudagraph PIECEWISE Mode** is noted for eliminating repetition loops. The setup's efficiency is further enhanced by **Chunked Prefill** and **Prefix Caching**, stabilizing request times after initial cudagraph compilation.

    - The Qwen3.6-27B model achieves impressive performance on a single RTX 5090, with sustained throughput of 120-124 tokens per second (TPS) for narrative tasks and 156-159 TPS for code tasks. This is facilitated by the INT4 AutoRound quantization and BF16 MTP head preservation, running on vLLM 0.19.2rc1 with Genesis v7.0 patches. The setup utilizes a 258,048 token context window, close to the architectural maximum of 262,144 tokens, and maintains high GPU utilization at 93% with a power draw of 400-426W.
    - The implementation of speculative decoding with MTP n=3 significantly enhances throughput, achieving a mean acceptance length of 2.65-3.46 and an acceptance rate between 55-82%. This method involves using three auxiliary heads to draft tokens per forward pass, which are then verified against the main head, providing a throughput multiplier of approximately 3x compared to non-speculative baselines. This technique is crucial for maintaining high performance in local inference scenarios.
    - Turboquant's 3-bit NC KV Cache is a notable feature, allowing the compression of the KV state to 3-bit non-uniform quantization. This enables a 125K context window within 24GB VRAM without running out of memory (OOM). Additionally, the use of Cudagraph PIECEWISE mode, which captures only attention-op boundaries, helps eliminate degenerate repetition loops caused by stale MTP state in FULL_AND_PIECEWISE mode on multi-GPU hosts.

  - **[Running Qwen 3.6 in Claude Code](https://www.reddit.com/r/LocalLLM/comments/1swbh68/running_qwen_36_in_claude_code/)** (Activity: 194): **The user is attempting to run large local models like *Qwen 3.6 27B* and *Gemma 4 26B* on a system with an `RTX 4070` GPU with `8GB VRAM` and `32GB RAM`, but faces performance issues such as slow execution and infinite loops. Smaller models like *Qwen 2.5-coder 7B* and *Gemma 4 e4b* are insufficient for coding tasks. A suggestion is made to use **Qwen3.6-35B-A3B**, a mixture of experts (MoE) model, which allows offloading part of the model to the GPU and the rest to RAM, potentially improving speed by `2-3x` while maintaining strong performance. The user also considers using Roo Code in VSCode as an alternative to Claude Code, where Qwen failed to complete tasks.** Commenters note that local MoE models can struggle with complex tasks, especially when involving multiple tools, due to large initial prompts and slow processing speeds (`20tps`). This can lead to timeouts and incomplete tasks, suggesting that substantial VRAM (`48GB+`) is necessary for effective local model execution.

    - ghgi_ suggests using the Qwen3.6-35B-A3B model, which is a Mixture of Experts (MoE) model. This allows for offloading the 3B part to a GPU while keeping the rest of the weights in RAM, potentially offering a 2-3x speed boost over the Qwen 3.6 27B model. Despite being slightly less performant than the 27B model, it remains competitive across most metrics.
    - OneSlash137 discusses the challenges of using local MoE models, particularly with complex tasks involving multiple tools. They note that while the model can handle simple interactions, it struggles with large prompts (10-20k tokens), leading to long processing times (3-5 minutes) and frequent timeouts. The issue is exacerbated by the model's tendency to resend entire prompts instead of incremental updates, causing significant delays and inefficiencies.
    - Plane-Pause-469 inquires about running local models in Claude code and seeks advice on the best models to use with 128 GB of RAM and 16GB VRAM. This highlights a common concern among users about optimizing hardware resources to effectively run large language models locally.


### 2. New Model and Benchmark Releases

  - **[Microsoft Presents "TRELLIS.2": An Open-Source, 4b-Parameter, Image-To-3D Model Producing Up To 1536³ PBR Textured Assets, Built On Native 3D VAES With 16× Spatial Compression, Delivering Efficient, Scalable, High-Fidelity Asset Generation.](https://www.reddit.com/r/LocalLLaMA/comments/1sxf2u0/microsoft_presents_trellis2_an_opensource/)** (Activity: 376): ****Microsoft** has introduced "TRELLIS.2", a cutting-edge 4-billion parameter model for generating high-fidelity 3D assets from images. This model utilizes a novel 'O-Voxel' structure, enabling the creation of complex 3D topologies with sharp features and full PBR materials, achieving up to `1536³` resolution with `16×` spatial compression. The model is open-source, with code available on [GitHub](https://github.com/microsoft/TRELLIS.2), and a live demo hosted on [Hugging Face](https://huggingface.co/spaces/microsoft/TRELLIS.2).** There is a technical debate regarding the support for ROCm, as the current documentation primarily mentions CUDA. A user reported issues running the model on an AMD 7800XT GPU, experiencing segmentation faults likely due to dependency conflicts and ROCm overrides.

    - DeedleDumbDee discusses the challenges of running TRELLIS.2 on AMD hardware using ROCm, noting that the model documentation primarily supports CUDA. They mention encountering segmentation faults when attempting to run the model on a 7800XT GPU, which has only been tested on NVIDIA GPUs with 24GB VRAM. The issues are likely due to dependency conflicts and using ROCm overrides.

  - **[AMD Hipfire - a new inference engine optimized for AMD GPU's](https://www.reddit.com/r/LocalLLaMA/comments/1swpsv0/amd_hipfire_a_new_inference_engine_optimized_for/)** (Activity: 426): ****AMD Hipfire** is a new inference engine optimized for AMD GPUs, utilizing a unique `mq4 quantization` method. It is not officially affiliated with AMD but shows significant performance improvements, particularly on RDNA3 architecture. Benchmarks on [Localmaxxing](https://www.localmaxxing.com/) indicate substantial speedups, with one user reporting a `2.86×` speedup on an RX 7900 XTX compared to baseline. The engine is actively being tested and models are available on [Hugging Face](https://huggingface.co/schuttdev).** Some users express a preference for industry-standard quantization formats like GGUF, suggesting it would simplify adoption. Benchmarks indicate that while **Hipfire** excels in AR decoding, it lags in prefill compared to **llama.cpp**, with performance highly dependent on workload type, particularly benefiting structured/code generation tasks.

    - User alphatrad reports a significant performance boost using AMD Hipfire on an RX 7900 XTX, achieving a `2.86×` speedup with coherent output compared to the AR baseline. However, they note that real-world application might differ from speed tests, especially in coding tasks.
    - Own_Suspect5343 provides a detailed benchmark comparison between Hipfire and llama.cpp on AMD hardware. Hipfire shows a `30%` faster AR decode than llama.cpp, but llama.cpp excels in prefill tasks. DFlash, a speculative decoding method, offers substantial speedups in structured/code generation tasks, with a `3.45x` speedup on a merge_sort prompt, highlighting its workload-dependent performance.
    - FullstackSensei expresses a preference for industry-wide adoption of GGUF for model quantization, suggesting it would simplify compatibility issues across different platforms and models. This reflects a broader desire for standardization in AI model deployment.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude and GPT Image 2 Innovations

  - **[The Comeback Chatgpt Did with Image 2 Is Insane](https://www.reddit.com/r/singularity/comments/1sw6q9j/the_comeback_chatgpt_did_with_image_2_is_insane/)** (Activity: 1035): **The post compares two AI-generated images of a Bugatti Chiron in Dhaka, Bangladesh, using the same prompt. The first image is generated by **Nano Banana pro**, while the second is by **ChatGPT Image 2**. The latter is noted for its realism, with one commenter stating it *"looks like a real image"*. However, the AI's limitations are evident in the text on signboards, which appear as a mix of Bangla and Hindi scripts, highlighting challenges in accurately rendering localized text.** Commenters highlight the realism of the ChatGPT Image 2 output, noting its photographic quality. However, they also point out the AI's failure to accurately depict local scripts, which detracts from the authenticity of the scene.


  - **[Stanford researchers fed a language model a DNA sequence and asked it to create a new virus. It wrote hundreds of them, and 16 worked. One  used a protein that doesn't exist in any known organism on Earth.](https://www.reddit.com/r/OpenAI/comments/1sw0vcf/stanford_researchers_fed_a_language_model_a_dna/)** (Activity: 1080): **The image is a research paper titled "Generative design of novel bacteriophages with genome language models," which details a study by Stanford researchers who used AI to design new bacteriophages. The AI-generated sequences resulted in 16 viable phages, with one utilizing a protein not found in any known organism on Earth. This research underscores the potential of AI in synthetic biology, particularly in designing bacteriophages that could target drug-resistant bacteria, highlighting both the promise and risks of AI in bioengineering.** Commenters express concern about the dual-use nature of AI in bioinformatics, noting its potential for both beneficial applications, such as targeted therapies against drug-resistant bacteria, and harmful uses, given the relatively low barrier to entry for creating novel viruses.

    - Saotik highlights the dual-use nature of AI in bioinformatics, noting that while AI-generated bacteriophages could lead to targeted therapies against drug-resistant bacteria, the same technology could potentially be used to create harmful viruses that affect humans. This underscores the need for careful regulation and ethical considerations in the deployment of such technologies.
    - AI-powered bioinformatics is compared to nuclear technology in terms of its potential impact. The commenter emphasizes that once the knowledge of creating novel viruses from AI-generated sequences is widespread, the barrier to entry becomes low, raising concerns about misuse. This points to the critical need for stringent controls and oversight in the dissemination and application of this technology.
    - yamankara clarifies that the term 'language model' in this context refers to a 'genome language model' rather than a general-purpose large language model (LLM). This distinction is crucial for understanding the specific capabilities and applications of the model used in generating novel viral sequences.

  - **[Claude 4.7 named a journalist from 125 words of unpublished writing](https://www.reddit.com/r/ClaudeAI/comments/1sw8npc/claude_47_named_a_journalist_from_125_words_of/)** (Activity: 800): ****Kelsey Piper** conducted an experiment with **Claude 4.7**, where she input 125 words of unpublished writing and the model identified her name. She ensured anonymity by logging out, using the API, and testing on a friend's laptop, ruling out account, browser, and IP identification. The experiment suggests that **Claude 4.7** can identify a writer's unique 'voice' from a small text sample, a capability not matched by **ChatGPT** or **Gemini**. This indicates a potential for models to measure and recognize writing style as a distinct fingerprint, highlighting **Claude 4.7's** superior reading ability, though it may be less flexible in generating prose, possibly due to its deep encoding of prose patterns. [Read more](https://www.theargumentmag.com/p/i-can-never-talk-to-an-ai-anonymously).** Some commenters question the technical rigor of Piper's methodology, noting potential oversights like account sign-in requirements for API use. Others draw parallels to past instances of linguistic analysis, such as identifying authorship through unique word choices, suggesting that writing style can be as distinctive as a voice print.



### 2. DeepSeek and Qwen Model Discussions

  - **[deepseek reduces prices again, the price for input token cache hits is reduced to 1/10 of the current level](https://www.reddit.com/r/DeepSeek/comments/1sw6y3c/deepseek_reduces_prices_again_the_price_for_input/)** (Activity: 356): ****DeepSeek** has announced a significant price reduction for input token cache hits, decreasing the cost to `1/10` of the previous level, from `$0.145` to `$0.0145`. This reduction is permanent, contrasting with a temporary discount offered the previous day. This move is expected to make **DeepSeek's** services more cost-effective, particularly for applications requiring `1M` context length, enhancing its competitive edge in the market.** Commenters express surprise and appreciation for the price reduction, with some attributing it to strategic generosity from Chinese companies, while others highlight the potential impact on large context applications.

    - DeepSeek's recent price reduction for input token cache hits to $0.0145 from $0.145 marks a significant cost efficiency, especially for handling 1M context lengths. This move could make large-scale model usage more accessible and affordable for developers and researchers.
    - DeepSeek Flash is noted for its performance, being on par with state-of-the-art (SOTA) models across various tasks. This suggests that DeepSeek is not only competitive in pricing but also in technical capabilities, making it a viable option for high-performance applications.
    - The drastic price reduction by DeepSeek is seen as a strategic move to make their services nearly free, potentially disrupting the market by offering high-quality models at a fraction of the cost, thus encouraging wider adoption and experimentation.

  - **[Is DeepSeek V4 Pro expensive (without discounts)?](https://www.reddit.com/r/DeepSeek/comments/1sw80vi/is_deepseek_v4_pro_expensive_without_discounts/)** (Activity: 183): **The image is a bar chart that compares the costs of running various AI models, highlighting that **DeepSeek V4 Pro** has a total cost of `$1071`, which is moderate compared to other models like **Claude Sonnet 4.6** at `$3959` and **GPT-5.4** at `$2851`. The costs are broken down into input, reasoning, and output costs, with DeepSeek V4 Pro having `$614` for input and `$420` for reasoning. This suggests that while DeepSeek V4 Pro is not the most expensive option, it is still significant in cost compared to subscription models like Claude and GPT, which can be around `50%` cheaper. The discussion also touches on the unpredictability of subscription models and the potential benefits of local models.** One commenter notes that DeepSeek V4 Pro is approximately `2.4x` more expensive than Mimo v2.5 Pro, with Mimo leading in intelligence benchmarks and having a cost advantage due to lower verbosity. Another commenter expresses skepticism about the claim that subscriptions cost only `10%` of API usage, suggesting that local models might be a more stable investment.

    - zoser69 provides a detailed cost comparison between DeepSeek v4 Pro and Mimo v2.5 Pro, noting that DeepSeek is approximately 2.4 times more expensive. In terms of performance, Mimo leads in Intelligence by two points, while DeepSeek has a slight edge in Coding. Mimo's lower verbosity score contributes to its cost-effectiveness, and both models have similar response speeds on OpenRouter, making Mimo more time-efficient.
    - Old_Stretch_3045 highlights the unpredictability of subscription models, suggesting that reliance on cloud-based models might not be sustainable due to potential changes in service limits and costs. They advocate for exploring local models and upgrading hardware as a more stable alternative.
    - zoser69 lists projected costs for various models based on a hypothetical 10% pricing criterion for frontier models. They estimate DeepSeek v4 Pro at $661.2, while other models like GPT 5.5 and Mimo v2.5 Pro are priced at $225 and $276, respectively. MiniMax m2.7 is noted as the most affordable model with above-average intelligence at $104.4, even without developer discounts.




### 3. AI Tools and Frameworks for Developers


  - **[LTX2.3 in Ostris Ai toolkit on a 5090 Training done in 7 hours ... I went Thanos way and I said fine ... I'll do it myself](https://www.reddit.com/r/StableDiffusion/comments/1swrs76/ltx23_in_ostris_ai_toolkit_on_a_5090_training/)** (Activity: 616): **The post details a custom training process for the LTX2.3 model in the Ostris AI toolkit using an NVIDIA 5090 GPU, achieving completion in 7 hours. Key adjustments include setting the `lora rank` to 48 initially, using `600 steps` for the first phase, and employing a `gradient accumulation` of 2. The process involves multiple phases with specific settings for `differential guidance`, `learning rate`, and dataset configurations, such as `512x512` resolution and `25 frames` per clip. The author emphasizes the importance of accurate prompts and trigger words for effective training.** Commenters are intrigued by the ability to capture likeness with `1-second` clips and request examples of prompts and captions used in training. There is also a discussion on the feasibility and effectiveness of using short clips for training Loras.

    - **DateOk9511** outlines a detailed multi-phase training process for LTX2.3 using a 5090 GPU, emphasizing the importance of dataset composition and training parameters. The process involves using 25 video clips, each 1 second long at 25 frames per second, with specific settings like 'low VRAM', 'lora rank', and 'differential guidance'. The training is divided into four phases, each with distinct step counts and settings, such as adjusting 'lora rank' and 'differential guidance' to optimize performance and accuracy.
    - **Disastrous-Agency675** shares a technique for optimizing VRAM usage during LTX2.3 training on a 3090 GPU by disabling sampling, which significantly speeds up the process. This approach allows for achieving 7000 steps in 6-7 hours by enabling VRAM offload and other low VRAM settings, suggesting a method to efficiently manage resources and improve training speed.
    - **Upper-Reflection7997** reports a failed attempt at LTX2.3 LORA training on a 5090 GPU with 64GB RAM, indicating potential challenges or misconfigurations in the setup. This highlights the variability in training outcomes based on hardware and configuration, and the need for precise tuning of parameters to achieve successful results.





# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
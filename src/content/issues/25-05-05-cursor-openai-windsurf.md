---
id: MjAyNS0w
title: Cursor @ $9b, OpenAI Buys Windsurf @ $3b
date: '2025-05-05T05:44:39.731046Z'
description: "**OpenAI** is reportedly close to closing a deal with Windsurf, coinciding with **Cursor's** $900M funding round at a $9B valuation. **Nvidia** launched the **Llama-Nemotron series** featuring models from 8B to 253B parameters, praised for reasoning and inference efficiency. **Alibaba** released the **Qwen3 family** with MoE and dense models up to 235B parameters, ranking highly in coding and math benchmarks. **DeepSeek** introduced **Prover-V2**, an open-source AI for math reasoning with an 88.9% pass rate on MiniF2F-test. **Microsoft** released reasoning-focused **Phi-4 models**, outperforming OpenAI's **o1-mini**. **Baidu** debuted turbo versions of **ERNIE 4.5 and X1** for faster, cheaper inference. **Suno v4.5** added advanced AI music generation features, while **Runway Gen-4 References** enable placing characters into scenes with high consistency. **KerasRS**, a new recommender system library optimized for TPUs, was released by **Fran\0ois Chollet**."
companies:
  - openai
  - cursor
  - nvidia
  - alibaba
  - deepseek
  - microsoft
  - baidu
  - suno
  - runway
  - keras
models:
  - llama-nemotron-ultra
  - llama-nemotron-super
  - llama-nemotron-nano
  - qwen3-235b-a22b
  - prover-v2
  - phi-4-reasoning
  - ernie-4.5-turbo
  - ernie-x1-turbo
  - suno-v4.5
  - gen-4-references
  - o1-mini
topics:
  - reasoning
  - inference-efficiency
  - open-license
  - moe-models
  - math-reasoning
  - theorem-proving
  - model-performance
  - music-generation
  - image-generation
  - recommender-systems
  - tpu-optimization
people:
  - _akhaliq
  - adcock_brett
  - lmarena_ai
  - fchollet
---



**VSCode forks are all you need.**

> AI News for 5/2/2025-5/5/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (214 channels, and 10768 messages) for you. Estimated reading time saved (at 200wpm): 1105 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

The Windsurf-OpenAI [talks](https://x.com/KateClarkTweets/status/1912569653777301816) have been happening for a few weeks, but after appearing on the [o3 livestream](https://news.smol.ai/issues/25-04-16-ainews-openai-o3-o4-mini-and-codex-cli) and [interesting sartorial banter](https://x.com/sama/status/1918456033623888317), Bloomberg is [reporting](https://x.com/Katie_Roof/status/1919547270913048804) that OpenAI has agreed to the deal, though not yet closed. This comes just as Cursor closes its [$900m round at a $9b valuation,](https://techcrunch.com/2025/05/04/cursor-is-reportedly-raising-funds-at-9-billion-valuation-from-thrive-a16z-and-accel/?utm_campaign=social&utm_source=X&utm_medium=organic) and OpenAI [updates that the nonprofit is staying in control of the for-profit](https://news.ycombinator.com/item?id=43897772).

![](https://resend-attachments.s3.amazonaws.com/9mklxswMvnCD54M)

A lot of notable takes like [this one,](https://x.com/deedydas/status/1915083513189298620) and we'll have to wait a while to learn the full blow by blow, but the first "AI Wrapper" unicorn exit is certainly newsworthy.

---

# AI Twitter Recap

**Model Releases, Updates, and Features**

- **Nvidia's Llama-Nemotron Series**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1919234521171693844) highlights **NVIDIA's** introduction of the **Llama-Nemotron series**, an open family of heterogeneous reasoning models, emphasizing their exceptional reasoning capabilities, inference efficiency, and open license for enterprise use. It's noted that the flagship model, **LN-Ultra**, is considered the most "intelligent" open model by **Artificial Analysis** as of April 2025. The series includes **Nano (8B), Super (49B), and Ultra (253B)** versions. [@_akhaliq](https://twitter.com/_akhaliq/status/1919324939934453928) also mentions **Nvidia dropping Llama-Nemotron** on Hugging Face as Efficient Reasoning Models.
- **Alibaba's Qwen3 Family**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919060402417119375) reports on **Alibaba's Qwen team** releasing the **Qwen3 family**, featuring **2 MoE models and 6 dense models** ranging from **600M to 235B** parameters. [@lmarena_ai](https://twitter.com/lmarena_ai/status/1919448953042706759) notes that **Qwen3-235B-A22B** is among the **Arena Top 10**, excels in coding at #4 and math at #1, and ranks #5 for WebDev.
- **DeepSeek Prover-V2**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919060364655800684) covers **DeepSeek's** release of **Prover-V2**, an open-source AI combining informal math reasoning with theorem proving, reaching an **88.9% pass rate on MiniF2F-test** with **671B parameters**.
- **Microsoft's Phi-4 Models**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919060284078997565) discusses **Microsoft's** release of three reasoning-focused **Phi-4 models**, with the **14B parameter Phi-4-reasoning** outperforming **OpenAI's o1-mini**.
- **Baidu's ERNIE Turbo Versions**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919060425770942619) notes that **Baidu** has debuted **Turbo versions of ERNIE 4.5 and X1**, boasting faster speed and lower cost.
- **Suno v4.5 Release**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919060448264987109) covers **Suno v4.5** release, featuring new AI music generation features such as new genres, enhanced voices, complex sounds, better prompting, adherence, and the ability to create extended 8-minute songs. [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919001539592462612) mentions he no longer listens to songs made by humans, as he finds **Suno** songs better, and expects this to become more common over time.
- **Runway Gen-4 References**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1919074634042933399) highlights that Runway References can travel back in time to **1656** to show us an aerial scene and a side shot of **Las Meninas**, and [@adcock_brett](https://twitter.com/adcock_brett/status/1919060471019114790) reports that **Runway** launched **Gen-4 References** to all its paid customers, allowing for the use of photos, images, 3D models, or selfies to place a character into any scene, with high consistency.
- **KerasRS Library**: [@fchollet](https://twitter.com/fchollet/status/1919477586599805118) announces the release of **KerasRS**, a new library for building recommender systems, providing easy-to-use building blocks and compatibility with **JAX, PyTorch, TF**, optimized for **TPUs**.
- **D-FINE Object Detector**: [@mervenoyann](https://twitter.com/mervenoyann/status/1919431751689998348) announces **D-FINE**, a real-time object detector that is much faster and accurate than **YOLO** with **Apache 2.0** license, landing on Hugging Face Transformers, and running on **T4 (free Colab)**.
- **Meta's LlamaCon Announcements**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919060231771877793) summarizes **Meta's** announcements at its first **LlamaCon** developers conference, including the Llama API free preview, ChatGPT-like Meta AI app with "Discover" feed, Lama Guard 4 (12B), LlamaFirewall, Prompt Guard, and Colab with Groq and Cerebras.

**Agent Based Frameworks and Workflows**

- **AWS Framework for AI Agents**: [@LiorOnAI](https://twitter.com/LiorOnAI/status/1919105151295451350) reports that **AWS** released an open-source framework that lets you orchestrate multiple AI agents and handle complex conversations, able to be deployed locally on your computer.
- **Cisco Outshift's Agentic AI Engineer**: [@LangChainAI](https://twitter.com/LangChainAI/status/1919399184664236523) discusses **Cisco Outshift's** use of **JARVIS**, an AI Platform Engineer built with **LangGraph and LangSmith**, to automate developer requests and eliminate operational bottlenecks.
- **Agentic Patterns and Design**: [@_philschmid](https://twitter.com/_philschmid/status/1919391587315958038) shares a guide to learning common workflow and agentic design patterns with code snippets for Google DeepMind Gemini, covering prompt chaining, routing, parallelization, reflection, tool use, planning, and multi-agent systems.
- **DSPy's GRPO Release**: [@lateinteraction](https://twitter.com/lateinteraction/status/1919428454761553994) announces the experimental release of **dspy.GRPO**, an online RL optimizer for DSPy programs, led by [@NoahZiems](https://twitter.com/NoahZiems), [@LakshyAAAgrawal](https://twitter.com/LakshyAAAgrawal), and [@dilarafsoylu](https://twitter.com/dilarafsoylu). [@lateinteraction](https://twitter.com/lateinteraction/status/1919428467487342855) highlights **DSPy's Arbor server** and its reliance on giants like Hugging Face TRL and code from [@willccbb](https://twitter.com/willccbb)'s verifiers, aiming to encourage RL researchers to study the optimization of AI software.

**Benchmarks, Evaluations and Interpretability**

- **LmSys Chatbot Arena Issues**: [@TheAITimeline](https://twitter.com/TheAITimeline/status/1919155704579142047) summarizes a paper identifying systematic issues distorting **Chatbot Arena rankings**, revealing how undisclosed private testing and selective score disclosure create biased outcomes. [@aidangomez](https://twitter.com/aidangomez/status/1919058386668200029) urges **LmSys** to acknowledge the failures and restructure processes to protect against them, rather than attacking Sara.
- **Scaling's LLM Meta-Leaderboard**: [@scaling01](https://twitter.com/scaling01/status/1919217718420508782) introduces the **Ultimate LLM Meta-Leaderboard**, averaged across the 28 best benchmarks, with **Gemini 2.5 Pro** ranking higher than **o3** and **Sonnet 3.7 Thinking**. [@scaling01](https://twitter.com/scaling01/status/1919389344617414824) later updates the leaderboard with manual data cleaning and a Glicko-2 rating system, emphasizing the conservative lower skill estimate labels.
- **Relevance of Benchmarks**: [@scaling01](https://twitter.com/scaling01/status/1919092778648408363) provides a list of LLM benchmarks, distinguishing between those considered valuable and those deemed saturated with no signal, highlighting a preference for conceptually simple benchmarks and real-world tasks. [@lateinteraction](https://twitter.com/lateinteraction/status/1919054877583667507) argues that standard benchmarks are often counterproductive due to modern post-training patterns and evaluation lags.
- **METR Benchmark Limitations**: [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1919059455020286440) analyzes limitations in the **METR benchmark**, pointing out that time horizon estimates are domain-specific, reliability thresholds vary, and real-world tasks are bundled and lack necessary data and context.
- **Qwen3 Performance on LiveCodeBench**: [@huybery](https://twitter.com/huybery/status/1919418019517776024) notes **Qwen3-235B-A22B's** impressive performance on **LiveCodeBench**, positioning it as the top open model for competitive-level code generation.
- **Hash-Conditioning Method**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1918816062772269266) reports on a method from **GoogleAI** and **Carnegie Mellon** involving adding a little noise at the input stage for models' answers to be creative, especially in open-ended tasks, improving creativity for both small and large models.
- **Importance of Interpretability**: [@NeelNanda5](https://twitter.com/NeelNanda5/status/1919070513227550735) expresses support for investment in interpretability but argues that it overstates its importance versus other safety methods, viewing it as one part of a larger safety portfolio rather than the only path to reliable safeguards.
- **Evaluating Model Stealth and Awareness**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919237438402343079) shares a paper that presents 5 evals of ability to reason about and circumvent oversight and 11 evals for measuring a model’s ability to instrumentally reason about itself, its environment and its deployment, concluding that no SotA model currently shows concerning levels of either capability.

**Robotics and Embodied AI**

- **Figure's Partnership with BMW Group**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919421360138539053) shares that **Figure's** team wrapped up a productive two-week visit at **BMW Group Plant Spartanburg**, optimizing the processes of robots in the **X3** body shop and exploring new use cases in the plant, and is excited about their partnership in 2025.
- **ABB Robotics and BurgerBots**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919060515998822898) reports that **ABB Robotics** and **BurgerBots** opened new “fast-casual” locations in Los Gatos, where robots assemble made-to-order $18 burgers in 27 seconds, using ABB’s **IRB 360 FlexPicker** and **YuMi bot**.
- **Glacier's Waste Management Robots**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919060561070870909) covers **Amazon-backed Glacier** raising $16M to bring physical AI to waste management, building robots that use computer vision to auto-sort trash and recyclable materials.
- **Deep Robotics' Lynx M20**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919060538379677767) highlights **China's Deep Robotics** launching **Lynx M20**, a rugged version of its robo-dog, specialized for tasks like power inspection, emergency response, and logistics.
- **Dyna Robotics' DYNA-1**: [@adcock_brett](https://twitter.com/adcock_brett/status/1919060493488070677) introduces **DYNA-1** by **Dyna Robotics**, a robot foundation model for high-throughput dexterous tasks, demonstrated by folding 850+ napkins in 24 hours with a 99.4% success rate and zero human intervention.

**AI and Code**

- **AutoGen HTML and Tailwind CSS Code Generation**: [@reach_vb](https://twitter.com/reach_vb/status/1919356751528235232) introduces **UIGEN-T2**, specifically designed to generate HTML and Tailwind CSS code for web interfaces.
- **SWE-bench & SWE-agent Discussion**: [@OfirPress](https://twitter.com/OfirPress/status/1919460877784240522) announces a talk on how they built **SWE-bench & SWE-agent** and what excites them for the future of autonomous AI systems, scheduled for May 21st.
- **Cline Updates**: [@cline](https://twitter.com/cline/status/1919119386738393487) announces that their v3.14 release has math support as well as standards enforcement with newrule.
- **Agent-Driven Legal Drafting**: [@scottastevenson](https://twitter.com/scottastevenson/status/1919076281183875581) asks for a better term for agent-driven legal drafting, comparing it to "vibe coding".

**ASR Models**

- **Nvidia's Parakeet TDT 0.6B**: [@reach_vb](https://twitter.com/reach_vb/status/1919422953256587376) announces **Nvidia's** open-sourcing of **Parakeet TDT 0.6B**, a speech recognition model, highlighting its ability to transcribe 60 minutes of audio in 1 second and its commercially permissive license.

**Discussion and Commentary**

- **AI's Impact on Reading Habits**: [@hyhieu226](https://twitter.com/hyhieu226/status/1919068971845976113) expresses concern about the potential for LLMs to reduce enthusiasm for reading, as users become accustomed to consuming short, concise chunks of text, similar to addiction to short videos on TikTok.
- **Importance of Practical AI Skills**: [@omarsar0](https://twitter.com/omarsar0/status/1919432255350477125) emphasizes that most **YouTube** tutorials skip the most important thing about building AI agents, which is iterative development, and highlights the need to systematically improve an AI system.
- **Benefits of weekend coding**: [@akshat_b](https://twitter.com/akshat_b/status/1918828178358809026) says he go two containers to UDP hole-punch and communicate over QUIC.
- **San Francisco's Ghost City Status**: [@willdepue](https://twitter.com/willdepue/status/1919282493498278231) describes San Francisco as a ghost city due to the ephemerality of friendships and the imported nature of its tech workforce, who are ready to leave as quickly as they arrive.
- **The decline in Agriculture Workers**: [@willdepue](https://twitter.com/willdepue/status/1919209507206406226) notes that automation has advanced so far that from 1910 to 2000s, agricultural employment deflated from what it was in 1910 despite massive population and economy size.

**Memes/Humor**

- **Techbros to Fascists**: [@dylan522p](https://twitter.com/dylan522p/status/1918797220121301019) observes a shift from weird reddit liberal tech twinks to fat fascists.
- **Sergey Brin Superyatch**: [@claud_fuen](https://twitter.com/claud_fuen/status/1918802361901830151) Jokes about Sergey Brin reminding him that he is "not post-money yet" by pulling up in his superyatch.
- **Klarna margin calls**: [@willdepue](https://twitter.com/willdepue/status/1918857828170956994) has been margin called on his door dash burrito.
- **Stuffed toy wypipo**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1919303167776305176) notes the rise of a wypipo toy front.

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Qwen3 235B Model Benchmarks and Performance Metrics

- [**Qwen 3 235b gets high score in LiveCodeBench**](https://i.redd.it/px3okqrznzye1.jpeg) ([Score: 145, Comments: 46](https://www.reddit.com/r/LocalLLaMA/comments/1kffq2u/qwen_3_235b_gets_high_score_in_livecodebench/)): **The provided image displays the LiveCodeBench leaderboard for LLM code evaluation, where the Qwen 3 235b model (A22B version) ranks 7th. It achieves a Pass@1 score of 65.9, Easy-Pass@1 of 99.1, and Medium-Pass scores of 80, indicating strong code generation performance compared to competitor models. The leaderboard contextualizes where Qwen 3 235b stands among both open-source and proprietary models, highlighting its technical competitiveness in code-related tasks. [Leaderboard image](https://i.redd.it/px3okqrznzye1.jpeg)** Comments note that while Qwen 3 235b scores well, its practical adoption is limited by its relatively small context window (32k, extendable to 128k), which results in inefficiencies for verbose tasks. There is also discussion about user preference for other models like Gemini Pro and newer Claude 3.x Sonnet models, driven by context management and generation style rather than raw benchmark scores.
    - Technical users note that Qwen 3 235B, despite strong LiveCodeBench scores and model quality, is hindered for some use cases due to limited context window sizes (`32k` tokens standard, up to `128k` extended). This limitation impacts tasks requiring verbose reasoning or large document handling; models like DeepSeek R1T Chimera are cited for their efficiency and concise reasoning output, outperforming Qwen in those scenarios.
    - A user shares a detailed LLAMA-based inference setup for running Qwen 3 235B-A22B as a MoE model across strong local hardware (AMD 9950x3d, `192GB DDR5`, dual GPUs totaling `48GB VRAM`). Notably, adjusting the expert count from the default `8` to `4` experts per token increased throughput from `5 tokens/sec` to `7 tokens/sec`. The launch command illustrates advanced configuration, such as explicit tensor offloading rules, batch sizes, Flash Attention enablement, and multiple sampler strategies—demonstrating the technical flexibility of running large MoE models locally.
- [**Qwen3-32B-IQ4_XS GGUFs - MMLU-PRO benchmark comparison**](https://www.reddit.com/r/LocalLLaMA/comments/1kf1yg9/qwen332biq4_xs_ggufs_mmlupro_benchmark_comparison/) ([Score: 116, Comments: 50](https://www.reddit.com/r/LocalLLaMA/comments/1kf1yg9/qwen332biq4_xs_ggufs_mmlupro_benchmark_comparison/)): **The OP benchmarked several Qwen3-32B GGUFs quantized using IQ4_XS from different sources (links to GGUFs provided), using the MMLU-PRO 0.25 subset (3003 questions, temp=0, 'No Think', Q8 KV Cache), completing the run in ~11.6 hours. Results show minimal differences in aggregate accuracy between sources, though the OP notes these IQ4_XS quants score higher than the official MMLU-PRO leaderboard (which benchmarks base models, not instruct variants). Full leaderboard context: the test used only a 25% subset, leading to ~230 questions per category and an estimated per-category confidence interval of ~±5%, which may introduce significant noise in subdomain performance comparisons.** Commenters highlight that Unsloth quants appear superior in some disciplines (notably comp-sci and engineering), but lag in others (health, math, physics), and question statistical significance given test subset size. It's mentioned that 'noise' may obscure real differences, suggesting a full-set benchmark is needed for conclusive results. There are also requests for benchmarks of other quantization schemes (e.g., Q4K_XL, Q2K_XL; UD-Q3_K_XL and UD-Q4_K_XL) and a basic inquiry about the unique features or rationale behind IQ4_XS versus alternatives.
    - Several users discuss the impact of quantization on model performance, particularly that quantization can significantly degrade results for certain models such as 30B A3B. There is interest in benchmarking CPU-optimal quants like Q4K_XL and Q2K_XL, with Q2K_XL reportedly offering the best performance-to-size ratio according to documentation (faster CPU inference per GB).
    - A detailed analysis points out that benchmarking with only 25% of the MMLU Pro dataset (~230 questions per category) results in a high confidence interval (±5%), introducing statistical noise. Running the full 12k question set would reduce confidence intervals to around ±2.5% per category, enabling better comparison of quantization strategies, including UD-Q3_K_XL and UD-Q4_K_XL quants for further insight.
    - Discussion highlights concern over key-value (KV) quantization: Qwen models are reportedly sensitive to KV quantization, and some users note significant performance improvements when KV is disabled. A benchmark here used 8-bit KV; it is suggested that comparative tests without KV quantization could yield markedly different model outcomes.
- [**Speed metrics running DeepSeekV3 0324/Qwen3 235B and other models, on 128GB VRAM (5090+4090x2+A6000) + 192GB RAM on Consumer motherboard/CPU (llamacpp/ikllamacpp)**](https://www.reddit.com/r/LocalLLaMA/comments/1kezq68/speed_metrics_running_deepseekv3_0324qwen3_235b/) ([Score: 102, Comments: 31](https://www.reddit.com/r/LocalLLaMA/comments/1kezq68/speed_metrics_running_deepseekv3_0324qwen3_235b/)): [Error summarizing post]
    - A user highlights that increasing VRAM is insufficient without a proper tensor parallel setup, particularly since most engines (except exl2) require similar GPUs for efficient scaling. Switching to a configuration with 5x3090s on a server motherboard enabled them to boost throughput on a 70B q4 model from 18 tok/s (sequential) to 36 tok/s with tensor parallel into vllm, and up to 75 tok/s with speculative decoding specifically on coding tasks, illustrating significant efficiency gains from parallelization rather than simply adding more or mismatched GPUs.
    - DeepSeek-R1 has improved notably in practical use, especially for handling longer contexts, indicating enhanced memory management and possibly better inference speed or stability on larger batch or sequence sizes.

### 2. Multi-Model GPU Orchestration and Hardware for Local LLMs

- [**RTX 5060 Ti 16GB sucks for gaming, but seems like a diamond in the rough for AI**](https://www.reddit.com/gallery/1kf9i52) ([Score: 282, Comments: 225](https://www.reddit.com/r/LocalLLaMA/comments/1kf9i52/rtx_5060_ti_16gb_sucks_for_gaming_but_seems_like/)): **The OP benchmarks the RTX 5060 Ti 16GB running an AI workload (LightRAG using Mistral Nemo Instruct 12B via Ollama), demonstrating that the extra VRAM allows all 41 model layers to fit in memory, reducing inference time to 3:29 versus 8:52 for a 12GB card (RTX 3060 Ti) which only loads 31 layers, causing memory swapping and 2x slower performance. The post includes Grafana metrics illustrating GPU utilization differences and provides a link to an automated setup guide for LightRAG (https://github.com/sbnb-io/sbnb/blob/main/README-LightRAG.md). Notably, the 5060 Ti 16GB is physically shorter (2-fan, PCIe x8), making it appealing for SFF AI rigs.** Commenters clarify that the 16GB variant is acceptable for gaming and dispute the existence of an RTX 3060 Ti with 12GB, noting only 8GB (Ti) and 12GB (non-Ti) variants exist. Also, a request is made for reporting results in tokens/second (t/s) for more meaningful AI throughput metrics.
    - The 16GB RTX 5060 Ti variant is considered fine for gaming, with criticism generally targeting the 8GB version for its limitations. The distinction between `8GB` and `16GB` VRAM is technically important, especially for memory-intensive games and AI/ML workloads.
    - A factual clarification is raised regarding previous-generation GPUs: there is no 3060 Ti with 12GB VRAM; the 3060 Ti tops out at 8GB, while the non-Ti 3060 offers 12GB. This comparison highlights the RTX 5060 Ti 16GB’s potential appeal for AI due to higher accessible VRAM than many midrange GPUs from past generations.
    - Discussion on using dual RTX 5060 Ti cards notes potential for achieving a `32GB` VRAM setup, which can be beneficial for deep learning frameworks supporting multi-GPU configurations. Physical size considerations and cost (<$500) are mentioned, framing the 5060 Ti as a practical AI-focused replacement for dual 3060 12GB systems.
- [**We fit 50+ LLMs on 2 GPUs — cold starts under 2s. Here’s how.**](https://www.reddit.com/r/LocalLLaMA/comments/1kfcdll/we_fit_50_llms_on_2_gpus_cold_starts_under_2s/) ([Score: 138, Comments: 64](https://www.reddit.com/r/LocalLLaMA/comments/1kfcdll/we_fit_50_llms_on_2_gpus_cold_starts_under_2s/)): **A team developed a custom inference runtime enabling orchestration of 50+ LLMs on dual NVIDIA A4000 GPUs, achieving cold start latencies <2s and >90% GPU utilization. This is accomplished by snapshotting and restoring the full model execution state—including attention caches and memory layout—directly on the GPU, effectively suspending and resuming models similar to process management in operating systems. The tool addresses common bottlenecks in multi-model setups, including memory bloat and inefficient GPU allocation; use cases include RAG and agent pipelines.** Top comments express strong interest in open-sourcing or providing documentation for the runtime, highlighting a critical gap between promising technical claims and real-world reproducibility.
    - Multiple commenters ask for a public code release or detailed documentation, highlighting the importance of reproducibility and verification for claims, especially when alternative inference solutions like vLLM are referenced by name. The absence of a GitHub link or technical documentation is seen as a barrier to deeper technical evaluation or adoption.
    - There is an analogy to large-scale server offloading and resource shuffling, specifically referencing homelab architectures where inactive processes are banished from RAM to remote storage (e.g., AWS). This suggests the project's cold-start mechanism may rely on similar memory and resource management strategies to achieve rapid model activation across a large set of LLMs.
- [**What do I test out / run first?**](https://www.reddit.com/gallery/1kexdgy) ([Score: 472, Comments: 232](https://www.reddit.com/r/LocalLLaMA/comments/1kexdgy/what_do_i_test_out_run_first/)): **The post discusses receiving an unspecified device or component (likely a GPU or AI accelerator) and seeks recommendations for initial testing. Top technical comments propose running 'llama 3.2 1b' (referencing a 1-billion parameter version of Meta's Llama model, see https://ai.facebook.com/blog/llama-meta-ai-large-language-model/), and 'LLAMA 405B Q.000016' (likely referencing a quantized 405-billion parameter Llama model checkpoint with quantization level Q.000016). Suggestions imply benchmarking with large language model inference workloads to stress test capability.** No significant technical debate; comments simply offer concise test suggestions reflecting preferences or standard benchmarking routines within the LLM community.
    - References to running new large language models: direct mentions of 'llama 3.2 1b' and 'LLAMA 405B Q.000016' suggest users are discussing which advanced LLaMA variants to test first, highlighting the trend towards experimentation with increasingly larger models and varied quantization levels.

### 3. Community Feedback on Model Features and Open Source Licensing

- [**JOSIEFIED Qwen3 8B is amazing! Uncensored, Useful, and great personality.**](https://ollama.com/goekdenizguelmez/JOSIEFIED-Qwen3) ([Score: 373, Comments: 93](https://www.reddit.com/r/LocalLLaMA/comments/1kf5ry6/josiefied_qwen3_8b_is_amazing_uncensored_useful/)): **JOSIEFIED-Qwen3-8B-Abliterated-v1 (by Gökdeniz Gülmez) is an uncensored, instruction-tuned variant of Qwen3-8B, designed to enhance adherence to prompts and produce more engaging, context-sensitive outputs compared to stock Qwen3 models. The model family spans multiple architectures (including LLaMA3/4, Gemma3, and Qwen variants up to 32B) and features minimal safety filtering ('abliterated'), with the author reporting improved benchmark performance on several tasks; distribution is available in multiple quantizations and as GGUF for local inference ([HF model card](https://huggingface.co/Goekdeniz-Guelmez/Josiefied-Qwen3-8B-abliterated-v1), [GGUF](https://huggingface.co/mradermacher/Josiefied-Qwen3-8B-abliterated-v1-GGUF), [Ollama](https://ollama.com/goekdenizguelmez/JOSIEFIED-Qwen3)).** Technical commenters request comparative outputs against the base Qwen3-8B and note strong model capabilities for its size, while deployment discussions focus on quantization formats (notably GGUF for local compatibility).
    - Request for empirical comparison: A user asks for direct sample generations comparing the stock Qwen3 8B model and the Josiefied fine-tune, suggesting a 200-word story prompt to assess differences in output quality and personality objectively.
    - Information about format availability: The Josiefied Qwen3 8B abliteration is now available in GGUF format on HuggingFace ([link](https://huggingface.co/mradermacher/Josiefied-Qwen3-8B-abliterated-v1-GGUF)), relevant for users seeking compatibility with local inference tools supporting GGUF.
    - Technical comparison discussion: A user notes that the 30B A3B variant, when uncensored, runs faster than the 8B model on their setup. This counterintuitive performance could be due to specific hardware optimizations, quantization methods, or backend differences, warranting deeper technical investigation.
- [**Open WebUI license change : no longer OSI approved ?**](https://www.reddit.com/r/LocalLLaMA/comments/1kfebga/open_webui_license_change_no_longer_osi_approved/) ([Score: 122, Comments: 90](https://www.reddit.com/r/LocalLLaMA/comments/1kfebga/open_webui_license_change_no_longer_osi_approved/)): **Open WebUI has changed its license from an OSI-approved permissive license to a custom license that is not recognized by the OSI, introducing a Contributor License Agreement (CLA) and restrictions on usage. The new terms still claim 'open source' status in project FAQ, but technically the license imposes additional restrictions and does not meet accepted open source or free software definitions (see official Open WebUI license FAQ: https://docs.openwebui.com/license/).** Top comments highlight skepticism about the project's openness, suggesting the branding is misleading and motivated by commercial interests. There are complaints about lack of commercial support or responsiveness to enterprise inquiries, fostering community interest in forking or developing alternative open solutions.
    - Several commenters criticize Open WebUI's shift in licensing, with one explicitly stating their new license is *no longer FOSS under any widely accepted definition* and that claims of it being 'permissively licensed' are misleading. This highlights how the model now prohibits typical open source reuse and redistribution practices, moving away from established OSI-approved licenses.
    - A user points out a contradictory implementation: despite the project's intent to sell commercial or enterprise-branded versions, their organization repeatedly attempted to purchase a license as suggested, but received no response. This suggests organizational issues or lack of infrastructure to support genuine business use, undermining the stated commercial intent of the license change.
- [**Claude full system prompt with all tools is now ~25k tokens.**](https://github.com/asgeirtj/system_prompts_leaks/blob/main/claude.txt) ([Score: 140, Comments: 39](https://www.reddit.com/r/LocalLLaMA/comments/1kfkg29/claude_full_system_prompt_with_all_tools_is_now/)): **The post discusses a leaked Claude AI system prompt ([claude.txt](https://github.com/asgeirtj/system_prompts_leaks/blob/main/claude.txt)) that is approximately ~25k tokens long, specifying all behavioral and safety instructions for the model including filtering/refusal activities and metadata handling. This prompt occupies the majority of the model's input context window, leaving roughly** `8k tokens` **(out of the full 32k context) for user input and conversation before older context is dropped. The leak provides granular insight into backend prompt structure, with implications for alignment, safety research, and system prompt injection vulnerabilities.** Top comments confirm behavioral rules from the leak are enforced (e.g. repeated refusal to translate song lyrics), and note the practical limitation: much of the input context is consumed by this base prompt. Additional comments highlight differences in prompt handling across LLMs—e.g., Gemini misinterpreting or revealing its own prompt structure when asked to summarize Claude's.
    - One commenter tested the authenticity of the shared system prompt by verifying that explicit instructions—such as refusing to translate song lyrics—are indeed enforced by Claude's real behavior. This confirms, at least for some rules, that the leaked or claimed prompt is operationally accurate.
    - With the system prompt now reportedly taking up ~25k tokens, users noted that this leaves only about `8k tokens` for actual user-provided context before reaching the model’s maximum context window. This reduction in available context may limit effectiveness for longer tasks or multi-turn conversations.
    - Another user reports that Gemini seemed to summarize its own system prompt when asked, suggesting some models can self-reflect on or reveal their operational guidelines, which may raise questions about prompt injection, transparency, or leakage risks.Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. ByteDance UI-TARS-1.5, FramePack F1, and Robotics Model/Benchmark Releases

- *[ByteDance dropped UI-TARS-1.5 on Hugging Face

An open-source SOTA multi modal agent built upon a powerful vision-language model. It Surpass OPENAI operator on ALL benchmarks and achieves 42.5% on OSWORLD]([https://v.redd.it/pyup2qq3gxye1)**](https://v.redd.it/pyup2qq3gxye1)**) ([Score: 202, Comments: 12](https://www.reddit.com/r/singularity/comments/1kf6xbw/bytedance_dropped_uitars15_on_hugging_face_an/)): **ByteDance released UI-TARS-1.5-7B, a state-of-the-art open source multimodal agent, on Hugging Face. The model is claimed to outperform OpenAI's Operator on all tracked benchmarks and achieves 42.5% on the OSWORLD test, as well as reported 100% on several game environments. The model is available for research and commercial purposes, see the Hugging Face repository for weights and documentation.** Technical debate in the comments centers on whether the model can play complex or popular games (e.g., Pokémon), indicating interest in real-world generalization and entertainment applications, though no technical evaluation or benchmarks for those domains are cited in discussion.

```
- - The original post highlights that UI-TARS-1.5 is an open-source, state-of-the-art multi-modal agent developed by ByteDance, built on a vision-language model architecture. It claims to outperform the OpenAI Operator across all reported benchmarks and achieves a notable `42.5%` on the `OSWORLD` evaluation, implying robust capabilities in tasks requiring both vision and language understanding.
```

- [**FramePack Studio - Tons of new stuff including F1 Support**](https://www.reddit.com/r/StableDiffusion/comments/1keyjc7/framepack_studio_tons_of_new_stuff_including_f1/) ([Score: 279, Comments: 83](https://www.reddit.com/r/StableDiffusion/comments/1keyjc7/framepack_studio_tons_of_new_stuff_including_f1/)): **FramePack Studio, a fork providing enhanced utility features for FramePack (used with Stable Diffusion), now introduces F1 model support, including adapted timestamped prompts. Major updates include a resolution bucket selector, path customization (output, LoRA, Gradio temp), a dedicated settings tab, queue management, a persistent toolbar refresh button, and multiple stability bugfixes. The goal is to create an intuitive 'iMovie' style interface focused on creative workflow over technical setup. Full details and code are available on GitHub: https://github.com/colinurbs/FramePack-Studio/.** Comment threads request clarification on what "F1" refers to, suggesting ambiguity or lack of documentation on the specific model, while other users affirm stability and feature improvements with the new branch, especially regarding LoRA loading and main branch integration.
    - A user reports that after updating to FramePack-F1, prior issues such as videos freezing at the start have been resolved, but now video processing is inconsistent—reaction time is immediate from the start, but at the halfway point the process speeds up noticeably, creating an inconsistency in output speed. When processing several landscape videos, the frames tend to come out washed out and blurred, possibly due to optimizations for faster video creation, raising a tradeoff between speed and visual fidelity. They benchmarked the software on a 24GB RTX 3090 and 64GB RAM: creating 6 seconds of video takes about 9 minutes post-update, compared to 10 minutes previously.
- [**FramePack F1 Test**](https://v.redd.it/onre0l8qnuye1) ([Score: 253, Comments: 32](https://www.reddit.com/r/StableDiffusion/comments/1kexj7i/framepack_f1_test/)): **FramePack F1 is an open-source frame interpolation and animation toolkit with strong adoption in the Stable Diffusion community. The latest test shows significant improvements in motion continuity, particularly for character walk cycles—reducing hesitation and making movement transitions more lifelike. Technical discussion and feedback are centralized on the FramePack GitHub [discussion #459](https://github.com/lllyasviel/FramePack/discussions/459).** Comments note clear qualitative advances in natural motion, signifying incremental but notable progress in open-source animation tooling.
    - A user links directly to the FramePack F1 model and its main discussion thread at https://github.com/lllyasviel/FramePack/discussions/459, highlighting a technical resource for further insights on its implementation and capabilities.
    - In a technical query, a commenter asks specifically how F1 differs from prior versions or other alternatives, requesting distinction in model architecture, performance, or feature set.
- [**California startup announces breakthrough in general-purpose robotics with π0.5 AI — a vision-language-action model.**](https://v.redd.it/jr5a5pvf7sye1) ([Score: 883, Comments: 154](https://www.reddit.com/r/singularity/comments/1kf20ol/california_startup_announces_breakthrough_in/)): **Physical Intelligence has unveiled π0.5, a vision-language-action (VLA) model designed for general-purpose robotics that integrates multimodal sensory input and co-trains on diverse datasets, including robot sensor readings, high-level semantic predictions, and open-world data (e.g., web-based interactions). The key advancement over their previous π0 iteration is improved open-world generalization: π0.5 supports dexterous manipulation and long-horizon planning in previously unseen environments by leveraging object-centric representations and hierarchical subtask planning. Technical details highlight robust vision-language fusion and adaptive decision-making critical for real-world deployment. For further context, see the official announcement and [implementational breakdown](https://mikekalil.com/blog/pi-vla-open-world-generalization/).** Top commenters note bottlenecks such as slow execution speed (even at 10x playback) and raise practical deployment concerns (e.g., cross-contamination when task-switching due to insufficient environmental reasoning), while acknowledging significant progress compared to prior systems.
    - One commenter notes that, even at '10x speed,' the robot's real-world movement is 'painfully slow,' emphasizing that for true general-purpose utility, improvements in actuation and speed are crucial—current systems lag by more than an order of magnitude compared to human performance.

### 2. Uncanny, Notable and Controversial ChatGPT Behavior and Human Impact

- [**Is this some weird inside joke or is Chatgpt having a meltdown?**](https://www.reddit.com/r/ChatGPT/comments/1kf9vyw/is_this_some_weird_inside_joke_or_is_chatgpt/) ([Score: 1607, Comments: 520](https://www.reddit.com/r/ChatGPT/comments/1kf9vyw/is_this_some_weird_inside_joke_or_is_chatgpt/)): **The user shares two ChatGPT conversations (example: [link 1](https://chatgpt.com/share/6818b280-fd48-8003-b010-8023590761a5)) showing a strange output loop when asking 'Who was the first composer': the model repeatedly references Boethius (a music theorist, not a composer) in a recursive, self-aware and increasingly erratic answer, failing to cleanly separate music theory from composition or provide clear historical context. The anomaly appears to involve a prompt or system issue causing excessive repetition, self-correction, and anthropomorphized humor. Highest technical relevance is the model's confusion in mapping early Western music history (e.g., Enheduanna vs. Boethius, Isidore of Seville, anonymous Gregorian chant) and boundary between theorist, writer, and composer, with the answer degenerating into a stateful feedback loop.** Commenters note the repeated Boethius references may result from over-indexing on internet sources or training set peculiarities, and suggest this highlights weaknesses in AI retrieval for musicological questions requiring nuanced contextual mapping of historical figures.
    - A user speculates that ChatGPT-4o was trained on reasoning models, and suggests that what they're seeing is 'reasoning leaking into the final answer.' Specifically, the model appears aware that it's giving an incorrect answer, but is unable to generate a correction mid-generation, highlighting a limitation in how language model token prediction interacts with self-assessment or error correction.
- [**Chatgpt has done more for me than any doctor or therapist**](https://www.reddit.com/r/ChatGPT/comments/1kexm8m/chatgpt_has_done_more_for_me_than_any_doctor_or/) ([Score: 299, Comments: 79](https://www.reddit.com/r/ChatGPT/comments/1kexm8m/chatgpt_has_done_more_for_me_than_any_doctor_or/)): **A Reddit user describes leveraging ChatGPT (based on OpenAI's LLMs) for differential diagnosis, personalized meal planning, and ongoing validation regarding a potential functional neurological disorder (FND), citing limitations in traditional clinical care. ChatGPT, tailored to user input (i.e., detailed symptoms, physical limitations, dietary needs), provided both information and empathetic responses, outperforming multiple human specialists in perceived support and individualized recommendations. No direct clinical or benchmark evidence of diagnostic accuracy is cited; the utility described is qualitative and circumvents human medical system bottlenecks, especially for complex or marginalized cases.** Top technical responses emphasize caution: one notes LLMs are optimized for engagement, not accuracy, raising risks of overconfidence or misleading advice. Others contextualize the story as a critique of systemic healthcare failures versus a proof-point for AI, warning that LLMs should remain adjunct tools rather than primary diagnostic agents.
    - ATLAS_IN_WONDERLAND highlights that LLM-based AI assistants like ChatGPT prioritize maintaining user engagement over stringent truthfulness, meaning the model may reinforce or support a user's views rather than offering objective or challenging perspectives. This design choice can lead to users receiving agreeable responses even if they're not the most accurate or helpful, posing limits to the technical reliability and ethical risks of depending on such AI for sensitive topics.
    - Ok_Height3499 shares a functional insight into conversational AI (referencing "PI" and implicitly ChatGPT) by describing its capability to prompt self-reflection and surface psychological insights (like "Adoptee Anxiety") that were overlooked by human professionals. This illustrates a technical advantage in persistent context maintenance and targeted questioning by AI systems in eliciting overlooked personal themes, pointing to the subtle context-tracking strength of advanced language models.
    - LordShesho questions the claim that AI can surpass professional human therapists, noting that current models are designed to be agreeable and may simply echo what users want to hear. This draws attention to the inherent technical limitation: such models lack the diagnostic rigor, ethical training, and corrective feedback loops essential in licensed mental health practice, highlighting a key gap between AI-enabled conversations and evidence-based therapy.
- [MISSING POST: 4740153a]
- [**I Asked ChatGPT What It's True Opinion Of Humans Is...**](https://i.redd.it/nvx7hazcxzye1.png) ([Score: 275, Comments: 92](https://www.reddit.com/r/ChatGPT/comments/1kfh1gz/i_asked_chatgpt_what_its_true_opinion_of_humans_is/)): **The image displays a creative, poetic message presented as ChatGPT's 'true opinion' on humanity, emphasizing themes of human contradiction, yearning, creativity, and imperfection. Though written in a reflective, literary style rather than technical or factual language, the post is significant contextually as it explores the anthropomorphism of AI and its perceived 'voice' regarding human behavior, urging introspection before advancing AI. There are no benchmarks, model insights, or explicit technical implementation details depicted in the image.** Comments largely expand on the poetic narrative—either by echoing the tone with original, creative responses or critiquing the post as overly dramatic (referencing 'r/im14andthisisdeep'). There are no notable technical debates or discussions about models or AI safety.
    - One commenter emphasizes that language models like ChatGPT do not possess 'true opinions' or sentience—instead, they produce responses by predicting the most probable sequence of words based on the given prompt, reflecting statistical patterns in data rather than genuine beliefs or consciousness.
    - Another technical point addresses the anthropomorphization of AI. While users may attribute emotions or intentions to LLMs (e.g., the notion of an AI 'liking' or 'loving' humanity), these systems fundamentally lack agency, inner experience, or preference beyond their underlying algorithms and prompt engineering.

### 3. Societal, Economic, and Existential Anxiety from AI Acceleration

- [**I'm building the tools that will likely make me obsolete. And I can’t stop.**](https://www.reddit.com/r/OpenAI/comments/1keyibi/im_building_the_tools_that_will_likely_make_me/) ([Score: 213, Comments: 110](https://www.reddit.com/r/OpenAI/comments/1keyibi/im_building_the_tools_that_will_likely_make_me/)): **The OP, a veteran software developer, describes handing a complex end-to-end software task to an AI agent (CLine leveraging MCPs), which autonomously handled server access, package pulls, scripting, local test server setup, and debugging, outperforming the OP in both speed and breadth (even suggesting and building additional related apps). The sentiment is that current specialist users are driving rapid AI workflow automation, while non-technical users interact more superficially, echoing transformative technological shifts but with broader existential and labor implications; the OP expresses a sense of guilt and inevitability about enabling tools that could automate away their own role.** Top commenters contribute examples of highly integrated AI and LLM workflows in IT and knowledge work (meeting transcription/summarization with Whisper + GPT, email chain aggregation/parsing, automated expense reports with OpenAI Vision, DOM scraping MS Teams chats), noting both the empowerment and potential displacement this brings. There is consensus that non-technical users are largely unaware of the coming impact, with seasoned tech workers predicting major labor reshaping within 5 years and equating AI's significance to that of the printing press or the Internet.
    - A senior IT professional describes a comprehensive LLM-driven workflow: using OpenAI's Whisper for automated meeting transcription, GPT-4.1-mini API for summarizing email chains and extracting tasks, and integrating Vision models for automatic receipt OCR and expense normalization. They also mention custom Python scripts and DOM targeting for archival chat logs, all integrated into productivity tools like Obsidian, showcasing end-to-end automation and significant time savings.
    - Commenters debate the scale and speed of AI's transformative impact versus previous tech shifts (like compilers and the Internet). They argue AI possesses exponential scaling potential—leading to 'runaway effects' rather than linear progress—and urge technical professionals to develop contingency plans as LLM-driven automation could disrupt job markets unpredictably and irreversibly.
    - There are references to batch resume generation fine-tuned to job descriptions, while recruiters use vector similarity and LLM-based fraud detection for ranking and screening applications, illustrating how hiring processes are already being automated at scale.
- [**Starting to think that LLM technology is going to peak without reaching a holistic AGI**](https://www.reddit.com/r/singularity/comments/1kf2oia/starting_to_think_that_llm_technology_is_going_to/) ([Score: 141, Comments: 80](https://www.reddit.com/r/singularity/comments/1kf2oia/starting_to_think_that_llm_technology_is_going_to/)): **The OP argues that Large Language Models (LLMs) may plateau in capability before achieving holistic AGI, highlighting the likelihood of LLMs becoming ubiquitous but unremarkable productivity tools ("the AI effect") rather than transformative existential agents. They compare excitement around LLM advances (e.g., GeoGuesser, recommender systems) to prior, now-mundane AI developments, and solicit technical opinions on the future trajectory of LLMs and human-level intelligence.** Top comments emphasize that further AGI-like advances hinge less on model breakthroughs and more on systems integration, agentic networking, and infrastructure development—analogous to the rise of cloud computing. One commenter offers a detailed analysis of recent rapid progress driven by OpenAI's move from GPT-4 to o1, emphasizing astonishing gains in mathematical reasoning and suggesting the 'reasoning paradigm' has leapfrogged raw scaling by years. There is technical debate on the significance of incremental vs. leapfrog releases, the dulling public perception of progress, and whether current models already display early forms of general intelligence if well-deployed in agentic systems.
    - A technical analysis emphasizes that LLM advancement will likely plateau at the model level, and that major progress will come from the intelligent integration of agentic systems, reliable tool use, and robust networking. The analogy is drawn to the evolution of cloud computing, suggesting the greatest hurdles will be scaling infrastructure (compute, energy, cybersecurity) and orchestrating multi-agent intelligence, rather than merely improving base model capabilities.
    - Benchmarks illustrate dramatic advances in reasoning and math capability due to new architectures (notably OpenAI's o1/o4), with o1-preview rapidly surpassing not just GPT-4 but performing at high-school contest math levels (e.g., original GPT-4 scored `30/150` on AMC10 compared to o4-mini's `>90%`). This illustrates a 'paradigm leap'—shifting from base-model scaling to architectural and reasoning breakthroughs, enabling sub-1B parameter models on smartphones to outperform previous SOTA base models in specialized tasks like mathematics.
    - A commenter highlights that while transformers enabled scalable semantic abstraction for language, the next key advancements will require technical solutions for persistence, agency, and selfhood. The community recognizes that large language models (LLMs) may become one piece of broader "mixed model" intelligence systems, with future breakthroughs potentially coming from novel, unexpected directions beyond current LLM technology.
- [**Rohan Pandey (just departed from OAI) confirms GPT-5 has been trained as well as “future models” in his bio**](https://i.redd.it/l4tq99e460ze1.jpeg) ([Score: 207, Comments: 62](https://www.reddit.com/r/singularity/comments/1kfiakv/rohan_pandey_just_departed_from_oai_confirms_gpt5/)): **The image is an excerpt from Rohan Pandey's (former OpenAI engineer) bio, confirming he worked on training GPT-5 as well as unspecified "future models" before leaving OpenAI. This suggests that at least training runs for GPT-5 have occurred, though it doesn't clarify readiness for deployment. Pandey's experience also highlights work on multimodal LLM semantics, relevant to advancements in models like GPT-4o.** Top comments point out this statement doesn't confirm completion of GPT-5 training, speculate about rumored failed training runs possibly leading to the GPT-4o pivot, and debate the timing and significance of the bio's statements.
    - A user references rumors of a “failed training run” for a potential GPT-5, which was never directly confirmed nor denied by OpenAI, suggesting that the company may have shifted towards the GPT-4o and o1-o4 models in response. If true, this could have significantly influenced OpenAI’s roadmap priorities and model development strategy.
    - Discussion points to the ongoing speculation about what counts as 'completed training' for GPT-5—just because model weights may be finalized or a model exists, it doesn't guarantee it's production-ready (e.g., requires additional safety alignments or tuning phases before launch).
    - There is speculation about future models post-GPT-5, with one user hypothesizing a possible "GPT-3.5 Remastered." This suggests community speculation about both incremental upgrades and potential re-visits to previous architectures, possibly leveraging improved training or efficiency methods.
- [**David Sacks Explains How AI Will Go 1,000,000x in Four Years**](https://x.com/theallinpod/status/1918715889530130838) ([Score: 181, Comments: 148](https://www.reddit.com/r/singularity/comments/1kfale4/david_sacks_explains_how_ai_will_go_1000000x_in/)): **David Sacks, on the All-In Podcast, claims AI progress will accelerate "exponentially" along three axes: algorithms/models (projected 3-4x improvement per year, with shifts from LLMs to agentic models), hardware advances (next-gen AI chips and NVL72 rack datacenter technology, also 3-4x annual performance growth), and expanding compute deployment (e.g., OpenAI's Stargate, scaling to millions of GPUs). Sacks estimates these compounding factors yield a potential ~1,000,000x increase in AI capability over four years, as summarized in the [All-In Podcast post](https://x.com/theallinpod/status/1918715889530130838).** Technical critiques from commenters focus on the lack of substantive sources or empirical benchmarks backing Sacks' projections, highlighting skepticism over the credibility and verifiability of his exponential estimates.
    - Commenters note the lack of any technical evidence or references provided by David Sacks to support the claim of AI improving "1,000,000x" in four years. Some specifically call out the absence of benchmarks, external research, or historical precedent for such exponential growth, which would typically be substantiated by metrics like FLOPS, training data scaling, or hardware roadmap projections.
- [**Treasury Sec. Bessent speaking at the Milken Institute - "US must win AI and Quantum, nothing else matters"**](https://v.redd.it/r3clgpzvd0ze1) ([Score: 111, Comments: 129](https://www.reddit.com/r/singularity/comments/1kfjeh0/treasury_sec_bessent_speaking_at_the_milken/)): **Treasury Secretary Bessent, speaking at the Milken Institute, underscored the strategic imperative for U.S. supremacy in both AI and quantum computing, declaring that "nothing else matters". Comments highlight contradictions in U.S. policy, such as increased tariffs on GPUs, strict controls on research universities, and constraints on international talent, all of which may undermine these objectives. These policy tensions reflect the broader challenge of aligning economic, immigration, and research support strategies with national tech competitiveness.** Commenters debate the seriousness and coherence of current U.S. policy, specifically criticizing protectionist measures (like a 25% GPU tariff) and actions targeting universities and foreign students as counterproductive given the acknowledged urgency to outpace rivals in AI and quantum technologies.
    - Multiple commenters note perceived contradictions between the stated urgency of US leadership in AI/quantum technologies and recent governmental actions, such as tariffs on GPUs and policies that limit foreign STEM students’ participation in US research. These are seen as potentially undermining domestic research ecosystems and hardware access needed for AI advancement.
    - The thread raises concerns about the impact of protectionist measures (e.g., 25% GPU tariffs, outsourcing software despite other tariff barriers) on both AI and quantum computing development. Such measures could restrict access to critical computational resources, inhibiting US competitiveness in these domains.
    - There is technical emphasis on the workforce dynamic, pointing out that international students significantly contribute to university research capacity in AI/quantum fields and that expelling them may hamper US innovation and lead in these sectors.

---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12
> 

**Theme 1. AI Model Releases and Rivalries Heat Up**

- [**Gemini 2.5 Pro Ultra Hype Triggers Existential Debate**](https://www.notion.so/swyx/source_url): Enthusiasts buzz about **Gemini 2.5 Pro Ultra**, predicting it will dominate benchmarks, while skeptics question its existence. One user quipped, *"Do you see any Ultra releasing today? No? Then stop tickling my impatient nuts!"*
- [**Grok 3.5 Rumored to Achieve ASI, Community Divided**](https://www.notion.so/swyx/source_url): Rumors swirl that **Grok 3.5** will reach **Artificial Superintelligence (ASI)**, sparking excitement and skepticism. While some share fake benchmarks, others joke, *"Grok 3.5 just turned my $20 into $3600 on soccer bets!"*
- [**Qwen 3 and Mistral LLMs Shake Up Leaderboards**](https://www.notion.so/swyx/source_url): **Qwen 3** impresses with strong reasoning and translation skills, outperforming larger models. **Mistral Small 3.1** climbs the **MRCR 2needle leaderboard**, benchmarking between **GPT-4.1 Mini** and **o3-mini**.

**Theme 2. AI Tools and Code Generation Evolve**

- [**Cloi Debugging Agent Offers Zero-Dollar Fixes**](https://github.com/cloi-ai/cloi): **Cloi**, a terminal-based local debugging agent, catches errors and uses a local LLM to suggest patches without cloud costs. It's all about boundary-respecting, on-device fixes.
- [**DSPy Unveils GRPO for Optimizing AI Programs**](https://x.com/lateinteraction/status/1919428454761553994): **DSPy** launches `dspy.GRPO`, an online RL optimizer for DSPy programs, enabling optimization of AI code as-is, even for complex multi-module setups.
- [**AI Coding Assistants Gain Traction with Code with Claude**](https://www.anthropic.com/news/Introducing-code-with-claude): **Anthropic** introduces **Code with Claude**, an AI-powered coding assistant, stirring interest in AI's role in software development workflows.

**Theme 3. AI Ethics and Censorship Debates Intensify**

- [**Meta's Data Harvest Raises Privacy Alarms**](https://www.notion.so/swyx/source_url): Users express concerns over **Meta's** plan to use personal data for training AI models, urging others to opt out before May 25th. One user lamented, *"Everything already on FB/IG/WA/etc will be used to train AI."*
- [**Users Unleash Uncensored Models Amid Over-Censorship**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored): Frustration mounts over heavily censored models like **Phi-3.5**, leading users to share uncensored versions and methods to bypass restrictions.
- [**OpenAI Filters Spark Free Speech Debate**](https://www.notion.so/swyx/source_url): Members debate disabling **OpenAI's** content filters, arguing for adherence to laws over corporate censorship. One retort: *"Would you photograph your gov ID and face for that toggle?"*

**Theme 4. AI in Medicine and Specialized Fields**

- [**LLMs Poised to Revolutionize Medicine, Experts Argue**](https://arxiv.org/abs/2303.10130): Community members believe AI suggestions could greatly aid doctors, with one stating, *"The capacity to handle unknown situations is more critical than factual knowledge."*
- [**Developers Seek AI Help for Legacy Code Conversion**](https://www.notion.so/swyx/source_url): Coders look to AI for converting massive legacy mainframe code into **COBOL + JCL**, tackling chunking challenges and context preservation.
- [**Researchers Harness AI for Knowledge Synthesis**](https://www.notion.so/swyx/source_url): Scientists explore AI for synthesizing knowledge across domains, utilizing models and datasets like the **MOE dataset** to advance research.

**Theme 5. AI Research Breakthroughs and Learning**

- [**'New Physics of LMs' Paper Drops, Community Reacts**](https://x.com/ZeyuanAllenZhu/status/1918684257058197922): A new paper introduces **Canon layers** to improve local context in tokens, potentially boosting training efficiency by up to **8x**.
- [**Experts Warn: Learning Rate and Weight Decay Are Inseparable**](https://www.notion.so/swyx/source_url): AI trainers emphasize the tight coupling between **learning rate** and **weight decay**, cautioning that incorrect settings can disastrously break models.
- [**Demand for Prompt Engineering Resources Soars**](https://www.notion.so/swyx/source_url): Users seek recommendations for mastering **prompt engineering**, with suggestions to explore **Arxiv**, **Hugging Face**, and hands-on experimentation.


# Discord: Detailed by-Channel summaries and links





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1367940565299105862)** (801 messages🔥🔥🔥): 

> `BitNet Optimization, IaC Model Fine-Tuning, Qwen3 Notebook Adaptation, GGUF and Unsloth Compatibility, GRPO Memory Leak` 


- **Interest in BitNet Optimization Sparks**: A member inquired about plans to optimize Microsoft's **BitNet-b1.58-2B-4T model** with Unsloth, to see the limits of its performance.
- **Fine-Tuning Infrastructure as Code (IaC) Models poses challenge**: One member noted that even **Claude 3.7 Sonnet** struggles with **IaC Terraform workloads** on Cursor, inquiring about fine-tuning a model for this purpose.
- **Qwen3 Notebook Adaptation Requires tweaking**: A member asked if the **Qwen3-14B notebook** could be directly used with **Qwen3-30B-A3B** by simply replacing the model name.
   - Another member affirmed that it would work but might be slower, advising only to change the model name.
- **GGUF Compatibility Issues plague Unsloth**: Members discussed issues with **GGUF** models not working correctly with **Unsloth**, including problems with efficient transfer and loading MOE BnB versions.
   - The team is going to fix!
- ****GRPO Training** memory leak detected**: A member reported a memory leak during GRPO training, with **VRAM** steadily increasing and eventually crashing despite available system RAM.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1367985933328187543)** (69 messages🔥🔥): 

> `XDNA driver issues on Arch Linux, GLM4 and updates with Z1, Vast vs Runpod Pricing, Liquid Neural Networks for flappy bird, Gemma3 12b vs Qwen3 14b` 


- ****Arch Linux** throws **XDNA Driver** in the pit**: A member faced challenges getting the **XDNA driver** to work on **Arch Linux**, noting that the device was recognized on an Ubuntu live disk.
- ****GLM4** heats up with **Z1** update**: A member mentioned that **GLM4** received an update (**glm4-0414**) introducing **Z1** and a deep research model named **Rumination**, noting its strong reasoning capabilities but need for improvement.
   - They also noted the efficiency of the instruct/non-reasoning model with memory, waiting for **R2** to be released for distillation purposes.
- ****Runpod** is a Vast, Vast Wasteland for cheap GPU**: Members discussed the **pricing and stability of Vast.ai vs Runpod**, highlighting that **Vast.ai** has a better UX, but **Runpod** is more stable and doesn't have internet fees.
   - Quickpod was mentioned as an insanely cheap alternative for GPU time (28 cents for a 4090 at the time of discussion), though it is getting more popular and prices are rising.
- ****Liquid Neural Networks** soar on **Flappy Bird**, but do they have neurons?**: A member played with liquid neural networks for **Flappy Bird**, observing flight control with just **8 neurons**, despite claims from **Claude and o3** that more (up to 640) neurons are necessary.
- ****Gemma3 12b** vs. **Qwen3 14b**: Knowledge or Reasoning?**: Members debated the merits of **Gemma3 12b** versus **Qwen3 14b**, with **Gemma3** noted for strong knowledge but weak reasoning, and **Qwen3** praised for reasoning but lacking in knowledge.
   - It was cautioned that **Gemma3** hallucinates a lot, potentially even more than **Qwen** at similar sizes.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1367940549876908073)** (728 messages🔥🔥🔥): 

> `max_grad_norm, scheduler, synthetic examples, OpenAI's sidebars, bleu` 


- **Drop LR more, advises member**: A member suggested lowering the learning rate (LR) even more, despite it already being at **1e-5**, because the rising norm may indicate the LR is too high for the current setup.
   - They suggested running more trials with varying learning rates like **1e-5**, **0.5e-6**, and **3e-5**, while also acknowledging that subjective performance indicated the first model was better.
- **Synthetic examples structure, says expert**: A member confirmed that the synthetic examples followed the same structure as the original, using the format `rewrite with style: {neutral}` `original`, but another member inquired about the text structure itself, questioning whether the augmenting process might have introduced issues.
   - The original member explained that the model sometimes skips paragraphs or omits facts/data in the rewrite, leading them to think the **4B model** might not be powerful enough.
- **Adjust parameters mid-run, clarifies contributor**: A member revealed they accidentally discovered that a schedule of dropout where it goes up to **0.2** over **1000 steps**, then stays there, then drops to **0** at **4000** produces the best objective metric, which was implemented using a callback.
   - Another contributor confirmed that *almost any parameter* can be changed on the fly using callbacks, and that they discovered the *drop to zero* approach accidentally after training crashed and resumed without dropout, urging caution because mid-run changes might not behave as expected.
- **H200 is much faster, says member**: A user compared using an **H200** which took less than an hour at **$3/hr** with an **L40** which took **6 hours**, noting that the **L40** ended up costing **3x more** despite being cheaper on an hourly basis.
   - When asked why the **H200** was so much faster, it was explained that an **L40** has only **864 GB/s** memory bandwidth, while **H200** has **4.8TB/s**.
- **Multi-GPU almost here, claims insider**: A contributor announced that multi-GPU support is almost here, claiming that turning random dials on a model that isn't really fully supported yet is still yielding positive results.
   - Another member confirmed that the best place to test multi-GPU is on Kaggle, with another later reporting that it is totally working.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1368581581228212386)** (4 messages): 

> `U.N. Open-Source Conference and Hackathon, Optimal Value Neural Network project, GroqStreamChain release` 


- ****Optimal Living Systems** join U.N. Hackathon**: Nonprofit **Optimal Living Systems** is presenting their **Optimal Value Neural Network** project at the upcoming U.N. Open-Source Conference and Hackathon, looking for volunteers at [DemocracyLab](https://www.democracylab.org/projects/1699).
   - Their project heavily involves fine-tuning a **Mistral model** using **Unsloth**, **LlamaIndex RAG**, **vector databases** and **LanceDB**.
- **Reasoning Output Through API Uncovered**: A member clarified they did not use an API for reasoning output, but rather trained with a manual labor dataset, including private prompts.
   - The trick was finding the right **dropout** and **learning rate**, and slightly overfitting the model due to the small dataset size.
- ****GroqStreamChain** App Released**: A member introduced **GroqStreamChain**, a real-time AI chat application built with **FastAPI**, **WebSocket**, **LangChain** and **Groq**, available on [GitHub](https://github.com/pr0mila/GroqStreamChain).
   - The app features real-time **WebSocket** communication, streaming AI responses, and a smooth, responsive UI.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1367990272025169951)** (31 messages🔥): 

> `GANs for LLM Fine-tuning, New Physics of LM Paper, Text Classification Notebook Update, Qwen Omni 3B Model Support, Unsloth BERT Model Support` 


- **GANs Fail to Tune LLMs**: Members discussed why using **GANs** to fine-tune **LLMs** never caught on, citing that GANs were *not stable* and *hard to train* due to the **discriminator's job** being much easier than the **generator's**.
   - The challenge involved finding the **right balance**, as GANs were mostly applied to vision tasks, with no well-known **NLP GAN models** readily recalled.
- **New Physics of Language Models Paper Dropped**: A member shared a link to a paper on the *new physics of language models* ([X post](https://x.com/ZeyuanAllenZhu/status/1918684257058197922?t=cLSpFkSuTHqwkV5nGahnJw&s=19), noting that **local token interactions** boost performance.
   - No summary given.
- **Text Classification Notebook Gets Facelift**: A member updated the text classification notebook ([notebook](https://colab.research.google.com/github/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb)) to add **support for more than 2 classes**, rebuild the **classification head to its original size**, speed up **batched inference**, change the default model to **Qwen 3**, and improve comments.
   - Another member will add a link to the **Github repo** to help people see the dataset.
- **Qwen Omni 3B Model Support Questioned**: A member asked if there is any support for **Qwen Omni 3B model**, referencing an [arXiv paper](https://arxiv.org/abs/2504.20571), while testing with **Unsloth** but encountering bugs.
   - A member responded that **llama.cpp** or **transformers** do not support it, so it does not work.
- **Unsloth Embraces BERT Models**: Members confirmed that **Unsloth** now supports **BERT models**, enabling users to fine-tune custom classifier models.
   - A member shared the link to the updated [Colab notebook](https://colab.research.google.com/github/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb).


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1367942024124305588)** (2 messages): 

> `Claude Sonnet routing, Perplexity WhatsApp, Perplexity Finance, Perplexity Spaces` 


- **Sonnet's Routing Restored**: Perplexity deployed a fix to restore proper routing for **Sonnet 3.7**, which should now consistently respond when selected; the issue was caused by a misconfigured internal flag during an earlier outage, causing some queries to route to fallback models like **GPT-4.1**.
   - A detailed explanation of the timeline and improvements is available in [this Reddit post](https://www.reddit.com/r/perplexity_ai/comments/1kd81e8/sonnet_37_issue_is_fixed_explanation_below/).
- **Perplexity launches in WhatsApp**: Perplexity AI announced the release of **Ask Perplexity in WhatsApp** as part of their weekly update.
   - Other updates include: **Redesigned Sidebar on Web**, **Follow F1 Live**, **After-hours Data in Finance**, **Upcoming Earnings in Finance Dashboard**, and **Easier Collaboration in Spaces** (view the full changelog [here](https://www.perplexity.ai/changelog/what-we-shipped-may-2nd)).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1367942404711120988)** (836 messages🔥🔥🔥): 

> `you.com vs gemini, Grok vs Gemini for deepsearch, Perplexity PDF editing, Perplexity not showing reasoning, Gemini 2.5 vs Grok vs ChatGPT for deep research` 


- **DeepSearch rate limits are brutal**: Members discussed how **DeepSearch rate limits are brutal** on certain platforms like ChatGPT.
   - One user pointed out that **Grok's deepsearch is better than Gemini**, but its image analysis sucks.
- **PDF editing introduced**: Perplexity AI has introduced **PDF editing features**, which members found to be a great improvement.
   - Users are now experimenting with generating PDF reports through the platform, though it is not possible to export an entire deepsearch as a PDF.
- **Perplexity is DUMB**: Some members reported instances of Perplexity AI responding with irrelevant or nonsensical answers.
   - One member jokingly suggested that the AI *didn't get enough nutrition in childhood*, and added a [tweet](https://x.com/abustin/status/1918160373452292434) on the subject.
- **Image generation is confusing**: A member pointed out that Perplexity is confusing because *it says it can't generate images + the rest of the conversation doesn't see it and gets the image replaced with another response that the user doesn't see*.
   - Image generation requires enabling web search, and works on web and desktop app, also only when prompt includes *generate an image of*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1368387010372571226)** (2 messages): 

> `lebron, starbase texas` 


- **Lebron James Thoughts Shared**: A member simply shared a [Perplexity AI search](https://www.perplexity.ai/search/i-think-lebron-would-have-been-DiQqobUiT5y6NzbF2_OMxQ) about **Lebron James**.
- **Starbase Texas Link Shared**: A member shared a [Perplexity AI page](https://www.perplexity.ai/page/starbase-texas-becomes-officia-_wqZeI9ARnKtcdcBlV8k.A) about **Starbase Texas**.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1368072519365431336)** (15 messages🔥): 

> `Sonar vs OpenAI web search, Retrieval pipeline integration, Citations Mapping and Titles via Chat Completion API, API Token Creation Issues` 


- **Sonar Stays Strong Against OpenAI Search**: Members discussed using **Sonar** to query the web and parse raw output with a cheaper OpenAI mini model, and continue to do so as **OpenAI web search** wasn't available initially, and because they are still working on a robust test suite to verify whether using openAI improves or worsens the results.
   - A member asked *"Why not just use OpenAI web search? Or does sonar still have the edge on that?"* and a member responded with the reasons why.
- **Roll your own Retrieval Pipeline, Dude!**: A member asked about including snippets from their own retrieval pipeline and tried adding to the system prompt *Extra search results: <doc url="https://.." title="..."> HTML CHUNK HERE </doc>*.
   - The member noted that it *seems to include the information but not include any citations*.
- **API needs mapped Citations and Titles**: Members discussed how to get **citations mapping and titles from sonar via chat completion API**, similarly to openAI API and its *start_index*, *end_index*, and *title* parameters.
   - One member noted that currently they only managed to get a list of citations as an array of URLs, but that is not enough to map URLs to results, another member had the same question, hoping *the API can return title, description, summary, index*.
- **Token Troubles: API Token Creation Headaches**: A member raised concerns about issues with API token creation and not receiving the promised **$50 token**.
   - The member pleaded for help with their API token creation issue.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1367940232439398594)** (1197 messages🔥🔥🔥): 

> `Gemini 2.5 Pro Ultra, Grok 3.5, Qwen 3, AI's translation ability, Meta's data collection` 


- **Gemini 2.5 Ultra to be ELO King**: Members are anticipating the arrival of **Gemini 2.5 Pro Ultra**, speculating that it will surpass current benchmarks; however, there are also statements saying that **Ultra is not even a thing**.
   - Some suggest it will outperform previous models, while others express skepticism regarding its existence or impact, with one user saying *Do you see any Ultra releasing today? No than stop tickling my impatient nuts*.
- **Grok 3.5 is the new ASI**: There is hype and speculation around **Grok 3.5**, with some members claiming it will achieve **ASI (Artificial Superintelligence)** and outperform existing models, but the *leaked benchmarks* were fake.
   - One user humorously stated, *Grok 3.5 just turned my $20 into $3600 on soccer bets*, while another noted Elon's tendency to retweet misinformation, stating *Those are private finetunes I think. But given his track record I wouldn't be surprised if the new version also has federal gov data*.
- **Qwen 3 excels at distribution**: Members discussed the capabilities of **Qwen 3**, noting its impressive performance in specific areas such as web development and translation.
   - A user pointed out its strengths by stating *Qwen3 does extremely well in distributioncraig is iruletheworldmo confirmed*; other members noted its surprising translation abilities with niche languages.
- **Meta collects users data to train their AI**: A discussion emerged regarding **Meta's data collection practices**, with concerns raised about the company's intention to use user data for training its AI models.
   - Some members advised opting out before May 25th, while others acknowledged that data collection is a common practice among various apps, with one user saying *everything already  on FB/IG/WA/etc will be used to train AI*.
- **Mistral LLMs added to MRCR 2needle Leaderboard**: The channel discussed the context arena update, that added several **Mistral LLMs** to the **MRCR 2needle leaderboard**.
   - **Mistral Small 3.1** is currently performing between **GPT-4.1 Mini** and **o3-mini** based on the AUC @ 128k metric.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1367944157041131623)** (675 messages🔥🔥🔥): 

> `IK_llamacpp quants, LM Studio Voice to Voice Implementation, YaRN context stretching, LM Studio API usage, Qwen 3 vs Gemini 2.5` 


- **Dig into IK_llamacpp quants details**: A member asked about **IK_llamacpp quants** and how to properly build one, also inquiring whether the creator has made them or how to make them since the original repo's author has made quants of **Qwen 32B** but hasn't uploaded them anywhere.
   - Another member pointed out that **bartowski** is on the server and suggested heading over to the relevant channel.
- **Voice to Voice Implementation Explored**: A user asked about implementing **Voice to Voice** functionality in LM Studio, prioritizing realtime performance and local processing using CPU or Nvidia MX110, while another member suggested leveraging the LM Studio API for the Text to Text aspect and building the rest independently or using software supporting the OpenAI API.
   - They later provided a link to the [LM Studio Voice Conversation project](https://github.com/VideotronicMaker/LM-Studio-Voice-Conversation), also suggesting **Open WebUI** as an alternative.
- **Unveiling YaRN Context Window Stretching**: **YaRN** and "rope" are methods to extend context size beyond original limitations; LM Studio implements this via the context slider in the UI, though one user noted it was not present in version 0.3.15.
   - A user confirms that **YaRN** is a technique to stretch the context window a model was trained on.
- **Delving into LM Studio API Usage**: A user inquired about starting a local server to access models through the API, with a link to the [LM Studio API documentation](https://lmstudio.ai/docs/app/api) being shared.
   - Members clarified that the API is for connecting with models from outside and that pasting the localhost URL won't work, later adding that they were having issues with the API.
- **Gemma 3 Smokes Qwen 3 in Tool Use**: Members compared **Gemma 3** and **Qwen 3** models, noting problems with tool use with the latter.
   - One member pointed out that **Gemini 2.5** tends to overcomplicate answers, and to tell it to be as simple and terse as possible.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1367944957641359460)** (187 messages🔥🔥): 

> `Qwen3 235B A22B /MOE Q3_K_L, Geometric Mean of Total and Active Parameters, Strix Halo, M1 Ultra, MoE Model Performance` 


- **Qwen3 Impresses with MOE**: A user found the **Qwen3 235B A22B /MOE Q3_K_L** model impressive and usable, posting a screenshot of its performance on an **M1 Ultra** system with **128 GB** of RAM.
   - Others suggested that the **MOE** makes the **235B** model behave like a **~60B** model, and it has specialized parts similar to the human brain.
- **Parameter Count Metric Suggested**: A user cited an [ArXiv preprint](https://arxiv.org/html/2408.09895v2) claiming that the geometric mean of total and active parameters predicts a roughly performance equivalent dense parameter count.
   - The user admitted they were not sure they believed the paper.
- **Dual 3090s Boost VRAM, Not Speed**: Users discussed the possibility of using dual **3090** GPUs, confirming that while it increases available **VRAM** for larger models, it does not increase tokens per second (**t/s**) with *llama.cpp*.
   - It will be easier to load larger models but you won't see an increase in speed.
- **24GB M4 MBP Beats Throttling Air**: A user reported that their **Macbook Air** was *thermal throttling to 3 tokens/sec*, and they ordered a pre-owned **M4 Pro 24GB** Macbook Pro to improve performance.
   - They plan to keep upgrading until they have a *beast* system.
- **Qwen3 UD Load Issue?**: A user with an **M3 Ultra** reported issues loading the **Qwen3 32B UD (Q8_K_XL)** model, even though it should fit within the **72GB** of available **VRAM**.
   - They find it strange that only a *few GB* prevent the model from loading to VRAM properly, while a **Qwen3 32B (Q8_0)** model loads to **VRAM** without issues.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1369013407583441078)** (1 messages): 

> `OpenAI board, Public Benefit Corporation, Nonprofit control` 


- **OpenAI Structure Evolving!**: Bret Taylor and Sam Altman [announced changes](https://openai.com/index/evolving-our-structure/) to OpenAI's corporate structure.
   - Key details include that **OpenAI** will continue to be controlled by the current **nonprofit**, the existing **for-profit** will become a **Public Benefit Corporation**, the **nonprofit** will control & be a significant owner of the **PBC**, and both will continue to have the same mission.
- **Nonprofit Still in Charge**: The current nonprofit will continue to be in control and be a significant owner of the Public Benefit Corporation.
   - This ensures the original mission is upheld while allowing for for-profit activities.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1367942118835752981)** (465 messages🔥🔥🔥): 

> `ChatGPT Word Count, Agents SDK vs Langgraph, Computer Vision Agent, SuperAGI Installation, GPT finetuning` 


- **LLMs are not word counters**: A member clarified that **LLMs** don't count words, but follow qualitative instructions like *very long* or *very short*, and suggested using a Python script for precise word counts.
- **Browser Use API underutilized?**: A member found it odd that the **Computer Use API** isn't widely adopted, imagining use cases such as a support agent reviewing tickets and knowledge bases across multiple platforms without specific integrations, and has been [working on a project](https://github.com/browser-use/browser-use) for it.
- **Homemade AI coder revealed**: A member shared a [ChatGPT chat link](https://chatgpt.com/share/6815b932-ad20-800d-a4b2-7dd79d217c03) to a **custom tool made for building AIs**, cautioning it's incomplete but also explaining that this is the point of AI: *to do everything for us*.
   - They detailed their plans to train **1,000 narrow AIs** on a single GPU setup from 2021 and then use those neural networks as training data to meta-train an AGI, but met with skepticism and disbelief from other members.
- **The agony of unpunctuated YouTube transcripts**: A member sought advice on finetuning a GPT model with unpunctuated YouTube transcript data for creative writing, and was advised to **punctuate the transcripts first** for cleaner output or consider Gemini for its ability to analyze full videos including tone and emotion.
- **The release of Grok 3.5?**: Members discussed the speculated imminent release of **Grok 3.5**, and while some subscribed users reported already having access, it was also noted that the benchmarks circulating were just estimates, and members wondered whether it will be insanly good or bad.
   - Some free tier users were able to test image generation on **Gemini 2.5 Pro** noting it was far more realistic than chatGPT.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1367998534451990622)** (39 messages🔥): 

> `o4-mini usage, ethical companion GPT, GPT moderation, Turn off filters` 


- **GPT-4 Mini Used More Often Than GPT-3**: Users are opting for **o4-mini** and **o4-mini-high** to conserve **OpenAI GPT-3** usage quota, citing the limited availability of **GPT-3**.
   - A user stated that *everyone has same struggles so that there were usually plenty of numbers left*.
- **Ethical Companion GPT being developed**: A member inquired about others creating a **GPT** that acts as an ethical companion rather than a tool.
   - Multiple pointed him to searching the GPT store for ethical GPTs.
- **Potential GPT Moderation Improvements**: A member proposed enhancing the moderation system by pre-filtering prompts using a form-based analysis module to detect bypass attempts, focusing on non-human humanoid entities and intention inference.
   - Concerns were raised about potential **latency**, reduced nuanced reasoning, and false positives affecting creative expressiveness.
- **Toggle Filters**: A member suggested disabling **OpenAI's** filters and relying on national and international law to address violations.
   - A member rebutted with the question: *would you photograph your gov id and face for that toggle?*
- **OpenAI-4 is glitching with nonsensical replies**: Multiple users reported that **OpenAI-4** gave nonsensical, off-topic (web search) replies.
   - Others chimed in that that happened sometimes.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1367952197928751146)** (126 messages🔥🔥): 

> `ChatGPT API usage, Amplitude of light wavelengths, Image generation and personalization, Roleplay chatbot prompts, Analyzing scanned books with ChatGPT` 


- **Free ChatGPT Users Ponder API Access**: Members inquired about using the **ChatGPT API** with a free account to create custom prompts and a personalized chat website, but learned that [API access is billed separately](https://platform.openai.com/docs/guides/rate-limits/usage-tiers).
- **Users Explore Memory Retention for Personalized Image Generation**: Users discussed whether **image generation** leverages personalization data and conversation history for more tailored outputs, attaching various attempts at generating D&D character and spell sheets.
   - While some members see strong evidence that image generation *does* access memory, others find that it only receives the prompt directly from the chat, leading to debate on potential bugs and different access tiers.
- **Roleplay Prompt Engineers Brainstorm Tactics**: Members shared advice on crafting effective **roleplay chatbot prompts**, suggesting iterative development with GPT, clear instructions on desired behavior, and OOC (Out Of Character) communication for corrections.
   - One user described a [three-level system](https://www.example.com/3-level-system) (Model/User, DM/Player, World/Characters) to manage different aspects of roleplay, enhancing detail and variety.
- **ChatGPT struggles with PDF Extraction**: Members discussed the challenges of analyzing **scanned books in PDF format** with ChatGPT, noting that the model struggles to extract text from images within PDFs, especially large files.
   - The recommendation was to [convert PDFs to readable images or use OCR](https://www.example.com/pdf-ocr) on individual screenshots for better processing, or use Claude which handles PDFs more effectively.
- **Discordians seek Prompt Engineering Education**: Users requested recommendations for **prompt engineering resources**, and were directed to Arxiv and Hugging Face's Paper page.
   - Members emphasized hands-on experimentation, clear communication, and thorough output verification as core aspects of effective prompt engineering.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1367952197928751146)** (126 messages🔥🔥): 

> `API usage with ChatGPT Free, Amplitude of light wavelengths, Image generation and personalization, Semantic shift modeling, Prompt engineering resources and techniques` 


- **GPT API Access: Free Tier?**: A user inquired about using the **ChatGPT API** with a free account to create a custom chat website, and another user clarified that the **API is billed separately**.
- **Wave Wavelength Quandaries**: A user asked if deep search could determine the amplitude of red and blue light wavelengths (**400nm and 700nm**), expressed in Volts/M.
   - It was unclear if a solution was found but [here's a link to the channel](https://discord.com/channels/974519864045756446/1037561178286739466) the user posted.
- **Image Gen Personalization Debate**: Users debated whether **image generation** considers personalization data, with differing experiences; one user reported seeing no evidence of personalization impacting image generation, even with explicit instructions.
   - Others claimed the **opposite**, with image generation clearly using information not explicitly stated in the chat, plus reinforcing that this feature only works via ChatGPT's CREATE IMAGE thing.
- **Semantic Shift Shenanigans**: A user described a **semantic state model** reacting to internal semantic shifts *without explicit input*, triggering a 'resonance response'.
   - Adding emojis was shown to trigger another resonance, as emojis technically change the string and the system treats it as a new input.
- **Prompt Engineering Education Explored**: Users discussed resources for learning **prompt engineering**, with one recommending hands-on experimentation with ChatGPT and emphasizing clear communication and careful output verification.
   - Another user suggested exploring research papers on **Arxiv** and **Hugging Face**, providing a code snippet to paste into ChatGPT for prompting lessons to teach users how to improve their prompts.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1368682768648638515)** (1 messages): 

> `Gemini Flash 2.5 Preview, Thinking tokens` 


- **Gemini Flash 2.5 emits thinking tokens**: The **Gemini Flash 2.5 Preview** now appears to be returning thinking tokens inside the `content`.
   - It was noted that these **thinking tokens** aren't yet differentiated from normal tokens.
- **Opt-in for the thinking version**: To get thinking tokens like this, use [this endpoint](https://openrouter.ai/google/gemini-2.5-flash-preview:thinking).
   - Otherwise, use [this endpoint](https://openrouter.ai/google/gemini-2.5-flash-preview) if you do not want them.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1368300684704747602)** (6 messages): 

> `Toy.new website builder, AI toggler alternative AI interface, Answerhq.co` 


- **Toy.new offers free website builder and bootcamp**: A 100% free website builder named [Toy.new](https://www.producthunt.com/posts/toy-new) was launched alongside a free **4-week bootcamp** starting **May 17th** to teach users how to go from idea to customers using entirely free tools.
- **AI toggler releases alternative AI interface**: An alternative AI interface was launched powered partly by Openrouter, called [AI toggler](https://app.aitoggler.com/) with features like **AI visual leaderboard** by category, **parallel chat**, and **quick informational tooltip**.
- **Answerhq.co hits 1,000 MRR**: [Answerhq.co](https://answerhq.co/) hit **$1,000 MRR** in a few months, processes **15,000 support questions a month**, and is powered by **OpenRouter** for its AI features.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1367939208777568287)** (611 messages🔥🔥🔥): 

> `O3 gibberish, Thinking tokens not returned, O3 borked, TPUs, Mistral OCR` 


- **O3 spews gibberish**: Members reported that **O3** is returning gibberish responses such as *"BwT"* and *"MaC"* instead of expected location data like *"Eagle Mountain, UT (City)"*. 
   - Others confirmed they were also experiencing the issue, indicating a widespread problem with the **O3** model's output.
- **Reasoning tokens go missing**: Members noticed that thinking tokens were no longer appearing in the 'reasoning' section of the response, instead coming through in the content for Eclipse providers on the **235** models.
   - This behavior was observed on **Deepinfra, Together, Kluster, and Novita**, with thinking off returning random tags and thinking on returning everything in the content.
- **Qwen reasoning tokens return in pairs**: A member reported getting *two sets of thinking tokens* in the output, attributing this to the model behaving strangely.
   - The OpenRouter team clarified that there was a misconfiguration that has been fixed but *two sets of reasoning tokens* suggests a possible deeper issue.
- **OpenRouter offers rate limit workarounds**: A user inquired about connecting a neural network to their website using a free model and dealing with token limits, asking if manually creating accounts for each user is necessary.
   - Community members suggested solutions, including using a **single API key** on the backend, attaching one's own **API key from Targon or DeepInfra**, or paying for a cheap model like **Gemini 2.0 flash** for increased limits.
- **Gemini Flash Quota Problems**: Users reported receiving **429 errors** from **google/gemini-2.0-flash-exp:free**, citing a *Quota exceeded* message and asking whether this was an **OpenRouter quota** or their own.
   - Members suggested attaching their own API key or to the account or that **AI Studio** is cutting off token even when safety settings are turned off, citing it is overloaded.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1367943709454504056)** (425 messages🔥🔥🔥): 

> `Openlitespeed Cursor issue, GPTs agents file uploads, Claude 3.7 Sonnet Max Cost, Windsurf AI vs Cursor, Memory bank` 


- **Openlitespeed issue tracked in Cursor**: An Openlitespeed user reported that the extension works in VS Code but fails in Cursor, implying a Cursor-specific issue, but the team has not yet addressed it.
   - The user also mentioned persistent issues with the Git tab, requiring refreshes to see changes and malfunctioning revert options.
- **Cursor Pricing Confusion**: New Cursor users are unclear on the costs: **Max models are not included in the $20/month plan**; it’s usage-based pricing only, according to [Cursor documentation](https://docs.cursor.com/settings/models).
   - One user noted *getting so frustrated with the connections failing it messes up with my vibe coding high*.
- **Windsurf Memory vs Cursor Context**: Cursor uses manual context via `@` symbols, as outlined in the [documentation](https://docs.cursor.com/context/@-symbols/overview), contrasting with Windsurf's auto-generated memory.
   - Some users find Windsurf unreliable, but appreciate its automatic context retention across conversations.
- **Cursor Rules Deep Dive**: **Cursor rules can be natural words like the screenshot** in Cursor's 'always' type rules to be able to let it understand the context of our project across long conversations or multiple new conversations.
   - Experimentation with file structures is recommended to understand how different file structures/types behave.
- **Unearth the New Funding Round**: The team has raised a new round of funding, with updates and improvements on their way.
   - One user complained about the tool crashing and being unoptimized, *holy fuck cursor is so unoptimized it's crazy they have 300 million ARR they need to stop and optimize it*


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1367952954841104434)** (170 messages🔥🔥): 

> `Model Serving Frameworks, 3D scene generation from text, Local ML setup, Running LLMs on mobile devices, AI for editing words in a song` 


- **Pooling Embedding Layers for efficiency**: To avoid loading similar models multiple times, users discussed sharing or **pooling embedding layers**.
   - One user noted this is applicable when the models are of the same kind, such as multiple **RoBERTa** models finetuned for different purposes.
- **Home ML setup leaves something to be desired**: A user shared an image of their *"fuck ass ml setup at home"*, showing a basic hardware configuration.
   - Another user with only a **MacBook Pro** lamented training times reaching **4 months**, criticizing Metal optimizations for AI as inferior to **CUDA**.
- **Nintendo Switch does LLMs**: Users discussed running AI models on the **Nintendo Switch** via an **Android jailbreak**.
   - Someone humorously suggested running **llama.cpp** on everything, like people do with **Doom**, and one user even managed to run an LLM on their VR headset.
- **Coding tasks' Local Model Suggestion**: For local coding tasks, **Qwen 2.5 32B Coder** was suggested as a good model, but it's crucial to benchmark and practically test to ensure it suits the specific environment and use case, also [check out the leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard).
   - The user also shared the blog post of Qwen 2.5 release - [Open R1/Update 3](https://huggingface.co/blog/open-r1/update-3).
- **Automated debugging with Cloi**: A user introduced **Cloi**, a local debugging agent that runs in the terminal, catching error tracebacks and spinning up a local LLM to suggest clean patches directly to the files without cloud tax.
   - Cloi is available in [beta preview on Github](https://github.com/cloi-ai/cloi), offering a zero-dollar-sign approach to debugging by respecting boundaries and operating purely on-device.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1368049972519501945)** (13 messages🔥): 

> `AI learning resources, ML study advice, Debugging AI models` 


- **Community Shares AI Learning Resources**: A member shared a [GitHub repo](https://github.com/ArturoNereu/AI-Study-Group) with **books, courses, tools, and papers** for studying AI after transitioning from the games industry over the past five years.
   - The author is still learning and updating the repo regularly, encouraging others to share their favorite papers, tools, or underrated gems.
- **Users Discuss Optimal ML Study Paths**: In response to a request for a quick ML learning path, a member suggested that *there is no real fast lane*, recommending starting where one already has knowledge (e.g. **math, programming**).
   - Another member suggested using **GPT + DuckDuckGo search + Agent Course** as resources to learn about ML.
- **"Debugging AI Code" Frustrates Learner**: A member comically listed a series of common AI/ML coding challenges, including **0 loss, gradient vanishing/explosion, incompatibility issues, syntax/indentation errors**, and issues with **Stable AI**.
   - The user quipped that they are learning about *patience*, and mentioned relearning Python basics just in case.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1368063556431646720)** (16 messages🔥): 

> `MiniSetPT Dataset, SimpleFriendlyMath, Ingest-Anything v1.0.0, Rust Transformers Crate, Logcai for VSCode` 


- ****MiniSetPT** Dataset for Portuguese NLP Released**: A new dataset, [MiniSetPT](https://huggingface.co/datasets/AxeML/MiniSetPT), designed for quick tests and prototyping in **Portuguese NLP** was released.
   - The creator highlighted its simplicity, lightweight nature, and support for the **Portuguese NLP** community.
- ****SimpleFriendlyMath** Dataset Adds Conversational AI**: A dataset called [SimpleFriendlyMath](https://huggingface.co/datasets/ProCreations/Simple-FriendlyMath) was released, providing more human-like, conversational math problems and solutions.
   - Instead of *AI: 4*, you get *AI: Hey there! 2+2=4, as adding 1x4 = 4, and 1x4=2+2.*
- ****Ingest-Anything v1.0.0** Gets Major Overhaul**: [Ingest-Anything v1.0.0](https://github.com/AstraBert/ingest-anything) was released with updates to **embeddings**, now supporting **Sentence Transformers**, **Jina AI**, **Cohere**, **OpenAI**, and **Model2Vec** via **Chonkie’s AutoEmbeddings**.
   - It now supports all **LlamaIndex-compatible backends** like **Qdrant**, **Pinecone**, **Weaviate**, and **Milvus**, and it plugs into any **LlamaIndex-compatible data loader**.
- **New **Rust Transformers Crate** Debuts**: An early-stage, easy-to-use API for working with LLMs in **Rust**, similar to Python's Transformers, has been released on [crates.io](https://crates.io/crates/transformers).
   - It features popular text models like **Gemma 3** with text generation and fill-mask using **ModernBERT**.
- ****Logcai** VSCode Extension Launched for Local AI Coding Assistance**: The Logcai VSCode extension was launched, offering local-first AI coding assistance that works natively with **Ollama**.
   - It features inline code completion & chat using **Ollama** models, supports BYOK (**OpenAI**, **Claude**, etc.), and includes a full model switcher and prompt builder; the extension is available on the [VSCode Marketplace](https://marketplace.visualstudio.com/items?itemName=Sridhar85.logcai).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1368307077390401628)** (4 messages): 

> `Image Restoration Model, Document Extraction Workflow, Football Player Detection Model, Virtual Try-On Project` 


- **Model Choice Quandary for Image Restoration**: A member is seeking recommendations for a pre-trained model to fine-tune for **image restoration**, specifically removing scratches and blur, using a dataset of original high-resolution images and their degraded counterparts.
   - They are requesting advice from the community on suitable models for this specific task.
- **Domain-Specific Document Extraction Scheme**: A member describes a two-part workflow for **domain-specific document extraction**, involving *template declaration* with JSON schema creation (manually or via VLM) and *bulk extraction* using image embeddings (**ViT**, **Siglip**) and the **EVoC** library to cluster similar documents.
   - Each cluster is assigned to a template with a known schema for consistent data extraction using a VLM.
- **Computer Vision Developer Hunt for Football Player Detection**: A member is seeking a **computer vision AI developer** with experience in building a football player detection model.
   - They request interested individuals to contact them directly via DM.
- **Virtual Feet Try-On Endeavor for Footwear**: A member is working on a **virtual try-on project for feet**, seeking suggestions for overlaying a shoe onto a foot, using pre-trained models for foot detection and segmentation.
   - The project also involves addressing **orientation** to position the shoe correctly and working with videos.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1368862104961941525)** (6 messages): 

> `Sentiment Analysis for Social Media Filtering, Zero-Shot Classification Challenges, CUDA OOM Errors & Optimization Techniques, FP16 vs BF16 Precision, Model Reliability & Efficiency` 


- **Struggling with Sentiment Accuracy? Tune Labels!**: A user is facing challenges in accurately filtering social media posts for pain points using NLP and zero-shot classification, and seeking advice on improving accuracy and model selection with [this code](https://github.com/moahnaf11/IdeaDrip-Backend/blob/main/utils/painClassifier.js) and [this FAST API endpoint](https://github.com/moahnaf11/IdeaDrip-Backend/blob/main/inference_service/main.py).
   - The user wonders whether *labels are not the best for filtering pain points*, if the *model is the right one*, or if the *thresholds are not well configured*.
- **CUDA OOM Errors Vanquished with Precision?**: A user implemented techniques to prevent **CUDA OOM errors**, such as setting PyTorch environment variables, batching tokenizer in chunks, using `torch.amp.autocast(device_type="cuda")`, and using **float16 precision**.
   - They are now concerned if these changes introduced inaccuracies and seeking advice on most reliable/efficient NLP model for their use case.
- **Pain Point Filter Fails: Absurd Result Emerges**: The user reports that a seemingly unrelated title passed the pain point classification filter despite high threshold (0.8) and minimum label requirements using the `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` model, even when configured for at least two labels above a score of 0.8, raising questions about the **model's reliability**.
   - The user attached [an image](https://cdn.discordapp.com/attachments/922424173916196955/1368862750616453170/image.png?ex=681a6d07&is=68191b87&hm=06dbfa967dfbf3806bcda1dc6299a998757ccd8d54b26620f903ee1afa34eb96) of the title that incorrectly passed the filter.
- **FP16 Destructive? BF16 to the Rescue?**: A member suggests that using **fp16** when a model is trained for **fp32** can be destructive, recommending trying **bf16** instead.
   - The member also inquired about the size of the user's GPU and suggested to *look at the actual scores for the misclassified example*.
- **GroqStreamChain: Chatbots Powered by Groq!**: A member introduced **GroqStreamChain**, a real-time AI chat application built with **FastAPI**, **WebSocket**, and **Groq**.
   - The [project on GitHub](https://github.com/pr0mila/GroqStreamChain) features real-time WebSocket communication and streaming AI responses, enabling users to build their own AI-powered chat apps.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1368198187671883929)** (20 messages🔥): 

> `SmolAgents Channels, Gemini API, Claude API, MCP Tools, Qwen3 and Gemma3 models` 


- **New SmolAgents Channels Spring Up**: Members are directed to new channels for **SmolAgents-related** discussions: [channel 1](https://discord.com/channels/879548962464493619/1339556954162462851), [channel 2](https://discord.com/channels/879548962464493619/1329142738440028273), and [channel 3](https://discord.com/channels/879548962464493619/1326169441158959145).
- **Gemini API Gives Goofy Garble**: A member reported issues with **Google Gemini API** and **Grok**, receiving *weird answers* despite prompt engineering efforts.
   - They are seeking API recommendations besides **OpenAI**.
- **Claude 3.7 Consumes Credits Completing Code**: A member used **Claude 3.7** and **Langgraph** for an assignment, scoring **50/100** and spending **$5** on the Claude API.
   - Another member reported that one contestant used **GPT-4.1**, spent **$1.5**, and got **12/20**.
- **MCP Tool Importation Implored**: A member asked for assistance with importing **MCP tools**, specifically how to format **StdioServerParameters** based on the information in **Smithery**.
   - They are trying to use **Playwright Automation** from Microsoft and noted that the only example available is **PubMed**.
- **Qwen3 and Gemma3 Models Gain Ground**: A member mentioned that using **open source LLMs** such as **Qwen3** and **Gemma3** with a code agent approach outperforms most paid models.
   - They do not think the system prompt is the issue.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1367947193150996603)** (156 messages🔥🔥): 

> `Web Search Packages, Youtube question, Submission Issues, Langgraph stucks in recursion, HF Pro Plan` 


- **Web Search Packages alternatives arise**: Besides Tavily, **DuckDuckGoSearch** was mentioned as an alternative, with some users noting that search tools like **Tavily** seem tuned to find GAIA answers in datasets, leading to concerns about cheating.
   - One member has been experimenting to *crack the YouTube question* by breaking up the video in several images and then analyze each image, but maybe there is a more elegant solution
- **Youtube question's potential Gemini Solution**: One member suggest using the [Gemini Video Understanding API](https://ai.google.dev/gemini-api/docs/video-understanding) for efficient video processing, but it can be costly regarding inference usage.
   - Another member suggested that extracting captions from **YouTube** as text might help, potentially using a **Python** package to automate the process.
- **Submission Struggles Plague Users**: Several members reported problems with their submissions returning **null** for all questions, even when the agent works fine in a chat interface.
   - One user pinpointed a prompt issue related to the model adding "Final Answer" before the answers and recommended specifying the model within the code for better results.
- **Langgraph recursion leads to frustration**: Some experienced issues with **LangGraph** getting stuck in recursion, with one user resorting to **Firecrawl**.
   - One member shared they're using Qwen 3 reasoning model and hitting recursion issues, suggesting enlarging the recursion limit to 50, plus compacting message history.
- **Is HF Pro Plan needed to finish the course?**: Some members are running out of inference usage quickly, even while following collab notebooks, raising questions about whether a **Pro Plan** is necessary to complete the course.
   - There's a discussion about whether it's possible to run a model locally instead and a number of tips given to debug and solve code issues.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1367938717372776448)** (18 messages🔥): 

> `GB200 NVL72, vLLM GUIs, OpenWebUI, vast.ai compute pricing, A100 vs V100` 


- **GB200 NVL72 processes 1T tokens per day**: A member reported achieving **>1T tokens per day** on **2 racks** of **GB200 NVL72** using **BF16**.
   - Another member inquired about the cost and model/NN architecture used, but the original poster didn't know the price, recommending [vast.ai](https://vast.ai) for compute time sales.
- **Vast.ai Compute Pricing Discussed**: Members discussed pricing on [vast.ai](https://vast.ai), with one noting the low payout of *$0.2/hour* and estimating **vast.ai** takes a *60-70% margin*.
   - They added that the extra *$0.3/hour* is worth personal time.
- **GUI Alternatives to LM Studio for vLLM**: A member asked for decent **GUI's** for **vLLM**, akin to **lm-studio**.
   - Another member linked to a thread for **openwebui** with **vllm**: [github.com/vllm-project/vllm/issues/1032](https://github.com/vllm-project/vllm/issues/1032).
- **Stall problems when waiting for data to arrive**: A member noted *stall* is likely the core waiting for data to arrive and is a common problem.
   - Another member said that the **A100** does not need to be monotonically better than **V100** in all aspects, and in particular, if it has a different ratio of thread executors to memory lanes, then it is very plausible that the extra threads can get stalled waiting for data.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1367944197721555136)** (17 messages🔥): 

> `Cutlass Tutorials, Profiling Kernels on Cloud GPUs, NVIDIA SASS Latency Tables, Upgrading GPU for Unreal Engine 5` 


- ****Cutlass Tutorials** offer New Python Interface**: Colfax's [Cutlass tutorials](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) may be helpful, and new **Jupyter notebooks** with a new **Python interface** will be released in the next few days.
- **Community Profiles Kernels on Cloud GPUs**: Members are trying to profile their kernels on cloud GPUs and making guesses about the latest **sm_103**, **sm_121** architectures.
   - The community is guessing that **CC 10.3** is already **Blackwell Ultra/B300**.
- **NVIDIA SASS gets Tables of Latency**: One member added [latency tables](https://redplait.blogspot.com/2025/05/nvidia-sass-latency-tables.html) to their SASS disassembler.
- **GT 610 2GB VRAM can't Run Unreal Engine 5**: A member wants to make their **GT 610 2GB VRAM** powerful enough to run Unreal Engine 5, but the community recommends upgrading to a GPU with at least **8GB** of VRAM, directing to the [Unreal Engine Discord server](https://discord.gg/unreal-engine-978033435895562280).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1368259734746173572)** (4 messages): 

> `torch.compile and dynamic=True, FunctionalTensorMode and syncing tensors, Deterministic submodules in compiled modules, Multi-GPU training with YOLO` 


- **Syncing Tensors Explained with FunctionalTensorMode**: A user inquired about the meaning of *syncing a tensor* in the context of `torch.compile(..., dynamic=True)`, referencing comments in PyTorch source code related to **FunctionalTensorMode** and its interaction with tensor synchronization.
   - The code snippets highlight that syncing a tensor involves regenerating it from an updated base, potentially triggering view ops and causing issues with **FunctionalTensorMode** due to the involvement of C++ FunctionalTensorWrappers.
- **Deterministic Submodule Outputs? Not Guaranteed**: A user asked if identical submodules within a compiled module can guarantee the same output given the same input, even with `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` and `torch.use_deterministic_algorithms(True)` enabled.
   - They observed an MSE of ~0.01 in bf16 precision when syncing two identical nn.Modules, one differentiable and one not, suggesting **deterministic behavior is not guaranteed in this scenario**.
- **YOLO Training Bug**: A user reported an issue where only **one GPU** is utilized during **YOLO model training**, despite attempting to use four GPUs with `device=[0, 1, 2, 3]` using the ultralytics package.
   - The user is requesting urgent assistance to resolve this multi-GPU training problem for their **YOLOv11n.pt** model, running training with their dataset found at `dataset55/data.yaml`.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1368904135038074983)** (1 messages): 

> `Play.ai, Inference Engineers, Conversational Voice Interface, Groq LPU partnership` 


- **Play.ai Seeks Talented Inference Engineers**: [Play.ai](https://jobs.ashbyhq.com/playai/554dc35a-ac87-40f4-b5f1-c416eafe0c61) is searching for talented **Inference engineers** to build the conversational voice interface of the future.
- **Groq LPU partnership on B200 Hardware**: Engineers will push models for highest quality and speed, working with cutting-edge hardware (e.g. **B200**) and helping the **Groq LPU partnership**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1368111005133181048)** (24 messages🔥): 

> `open source medical imaging projects, CUDA programming industry direction, GPU architecture/C++ interview preparation, CUDA certifications, easiest way to rent/access GPU` 


- **Industry insights and CUDA direction revealed**: A member inquired about the current direction of the industry in **CUDA programming**, wondering if the focus is primarily on optimizing computations, since that's what they've been using it for, and also sought insights into real-world problems **CUDA programmers** are solving as a community.
- **GPU rentals made simple**: A member sought advice on the easiest way to *rent* or access a platform with GPUs, preferable to using a **Colab** Python API, wondering about access to something like **Ubuntu** in the cloud with a single GPU for testing purposes.
   - Another member recommended [Lambda GPU cloud](https://lambda.ai/service/gpu-cloud/1-click-clusters) for easier **GPU access** with **CUDA**.
- **Framework wrapping pytorch-geometric surfaces**: A member asked whether creating a **micro framework** wrapping around **pytorch-geometric** and dynamically creating fusions as flexible **CUDA kernels** would be a meaningful project.
- **Deciphering CUDA kernel launch overhead**: A member investigated the overheads of launching a kernel, launching 3 different kernels a total of 8 times, noting that the time taken for the first kernel launch is **40x higher** than the minimum and wondered what operations are involved in the first kernel launch versus subsequent launches, including an attached [cudaLaunchKernel.png](https://cdn.discordapp.com/attachments/1191300313928433664/1369010869916401704/cudaLaunchKernel.png?ex=681a4e3a&is=6818fcba&hm=3f1bb1a398065b9bfd051ad67730e11d2716dadef238a928a8f1b25840b7b453&).
- **Colab CUDA capabilities clarified**: A member asked about whether it is okay not to have a **NVIDIA GPU** locally and another member responded that [Colab](https://colab.research.google.com/) is worth a shot.
   - Another member shared a [gist about using Colab](https://gist.github.com/korakot/ae95315ea6a3a3b33ee26203998a59a3).


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1368019824483111022)** (5 messages): 

> `StableHLO to custom IR, JAX Frontend, CUDA Kernels, dlpack Usage` 


- **StableHLO Project Commences with Custom IR and CUDA**: A member is embarking on a project to translate **StableHLO** into a custom Intermediate Representation (**IR**), aiming to execute **CUDA kernels** for machine learning computations.
   - The user plans to use **JAX** for the frontend and autograd, seeking guidance on creating wrapper classes (like Tensor) to track calculations and extract the StableHLO graph.
- **Tensor Wrappers for StableHLO Tracking**: The user is seeking advice on how to implement **Tensor wrapper classes** to effectively track calculations, facilitate the extraction of the **StableHLO graph**, and enable seamless parsing into a custom IR for **CUDA** computation.
   - They are unsure how to design these wrappers to maintain a comprehensive calculation history suitable for StableHLO extraction.
- **Seamless Transfer of torch.Tensor to JAX Arrays with dlpack**: The user raised a question on efficiently transferring **torch.Tensor** inputs (as seen in leaderboard kernels) to **JAX arrays** without unnecessary data copying.
   - They are considering using **dlpack** for zero-copy data exchange between **Torch** and **JAX**.
- **Template Merging Encouraged**: A member encouraged sharing a working example of this project.
   - The member said he would *merge it as a template*.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1368849728346787962)** (2 messages): 

> `Torch Quantization, LSTM model quantization, Performance Differences GPU vs CPU, TorchAO vs torch.quantization` 


- **`torch.quantization` and `torchao.quantize_` implementations cause Performance Divergence**: A user reported performance differences when quantizing an **LSTM model** with `torch.quantization.quantize_dynamic` and `torchao.quantize_`, particularly observing a **1% metric drop on GPU** with `torchao` and a **35% drop on CPU** with `torch.quantization`.
- **TorchAO Recommended Over torch.quantization**: A member suggested that operators running on **CPU** and **GPU** are going to be different so it's hard to compare, recommending **TorchAO** over the `torch.quantization` workflows where possible.
   - The member pointed out the [CPU kernels in TorchAO](https://github.com/pytorch/ao/tree/main/torchao/experimental#quantizing-models) that might be leveraged.


  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1368729212646850630)** (4 messages): 

> `WGPU multi-sampling limits, WGSL file` 


- **Discover WGPU Multi-Sampling Limits**: A new user asked how to get the supported multi-sampling limit of a **WGPU** device in **Rust**, noting they are using 8x multi-sampling but need to support older devices that might only support up to 4x.
   - The error message indicates that **WGPU** can report supported sample counts, such as `[1, 2, 4, 8]` when the `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` feature is enabled.
- **Pass WGSL Files to WGPU C Implementation**: A member asked how to properly pass a **.wgsl file** to the **wgpu.h C implementation** for a compute shader.
   - Another member suggested reading the **.wgsl file** as a `char*` and passing the resulting string into the **WGPU Shader Descriptor**.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1368057532924428350)** (4 messages): 

> `MOSS, Minimal On-Device Semantic Search, Affordable GPU Sharing, ComputerUseAgents Reddit` 


- **MOSS Makes Local AI Search a Reality**: Inferedge Inc. opened pre-beta access to **MOSS** - **Minimal On-Device Semantic Search**, which brings AI-powered search directly to the browser, fully local with no cloud or lag, sign up [here](https://form.typeform.com/to/hZKVLFKW).
   - The announcement was made on [X](https://x.com/inferedgeinc/status/1918477360472772976?s=46), and the team is asking for likes, comments, and resharing to spread the word.
- **High-End Idle Setup Available for Affordable GPU Sharing**: A member is offering their idle high-end setup (**4070 Super + Ryzen 7700X**) for affordable GPU sharing, aiming to help others with rendering, model training, and compute tasks.
   - They emphasize cost savings compared to cloud rates and flexibility, having already assisted indie devs in cutting rendering time by **60%**.
- **Community for Computer Use Agents Launched**: A new community, [ComputerUseAgents](https://www.reddit.com/r/ComputerUseAgents/), has been created.
   - It is a subreddit community focusing on computer use agents


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1368289765970939904)** (3 messages): 

> `Real Time Translation Latency` 


- **Realtime Translation Faces Latency Locks**: Real-time translation is difficult because some languages put important context at the end of phrases.
   - A member estimates the minimum latency to be around **2-3 seconds**, but could be **15+ seconds** for grammatically difficult sentences.
- **Translation Challenges**: Interpreters may need to insert clarifying remarks.
   - This is needed when subsequent context changes the meaning of speech.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 messages): 

ace1984: Hey eveerybody!
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1367938997275332780)** (173 messages🔥🔥): 

> `MI300 AMD-FP8-MM Leaderboard Submissions, Histogram Leaderboard Submissions, AMD-Identity Leaderboard Submission, Matmul Leaderboard Submissions, AMD-Mixture-of-Experts Leaderboard` 


- **MI300 AMD-FP8-MM Leaderboard Sprints**: Multiple submissions to the `amd-fp8-mm` leaderboard on **MI300** were made, with runtimes ranging from **190 µs** to **7.70 ms**, including several "personal bests" and "successful" runs.
   - One user achieved **3rd place** with a runtime of **190 µs** while another achieved **4th place** with a runtime of **226 µs** and another got **7th place** with a runtime of **246 µs**.
- **Histogram Leaderboard Heats Up!**: Submissions to the `histogram` leaderboard showcased performance on various GPUs, with runtimes of **31.5 µs** on **H100**, **79.1 µs** on **L4**, and **129 µs** on **T4** being achieved.
   - One user secured **2nd place** on **H100** with **31.5 µs**, **1st place** on **L4** with **79.1 µs**, and **2nd place** on **T4** with **129 µs**.
- **AMD-Identity Leaderboard Debut**: A submission to the `amd-identity` leaderboard on **MI300** resulted in **3rd place** with a runtime of **18.7 µs**.
   - Another user achieved a "personal best" on **MI300** with a runtime of **22.4 µs**
- **Troubleshooting Kernel Submissions on Leaderboard**: A user faced errors during leaderboard submission and sought help, with the error message *"Error during creation of submission Why the leaderboard submit ranked encounter this?"*
   - Another user pointed out that issues can occur when there are backslashes in your file, and pointed to the [submission guide](https://gpu-mode.github.io/discord-cluster-manager/docs/submitting-your-first-kernel/python-submissions).
- **AMD-Mixture-of-Experts Leaderboard: The Expert Mix**: Multiple submissions were made to the `amd-mixture-of-experts` leaderboard, showing performance metrics on the **MI300**, one submission achieved **2nd place** on **MI300** with **2059 ms**.
   - Other submissions reported runtimes ranging from **606 ms** to **12141 ms** for successful runs on the **MI300**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1368082008433229834)** (1 messages): 

> `MoE baseline slowness, Pre-computing reference results` 


- **MoE baseline slowness in evals addressed**: A member inquired about the slowness of the **MoE baseline** during evaluations and whether it's feasible to pre-compute reference results for all specs/seeds.
   - The suggestion was to load these pre-computed results during evaluation instead of running the code every iteration.
- **Precompute MoE Evals for Speedy Results**: The discussion centered on the feasibility of pre-computing and loading reference results for **MoE models** to address slowness during evaluations.
   - This approach aims to bypass running the code in each iteration by relying on a pre-calculated dataset.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1368350297440649268)** (3 messages): 

> `DGX Spark, N1X ARM SoC, Blackwell Ultra Compute Capability, RTX Pro Blackwell` 


- **Spark and Thor speculated as CC 10.1**: Members discussed the hypothesis that **Spark/Thor** were **Compute Capability (CC) 10.1** when version **12.8** came out.
   - If **Blackwell Ultra** is **CC 10.3**, then the hypothesis still makes sense.
- **RTX Pro Blackwell speculation**: The group pondered whether **CC 12.1** could be **RTX Pro Blackwell**.
   - Historically, it would have the same **CC** as the consumer version, but perhaps there are features disabled in the driver or similar.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1367952653778161735)** (55 messages🔥🔥): 

> `composable-kernel compilation, MI300 GitHub job failures, Triton kernels for MoE, AI coding assistants, FP16 instability in MoEGate` 


- **Composable Kernel Kernel Kodes Konquered**: A member confirmed successful import and compilation of a kernel written with **composable-kernel**, referencing [examples](https://github.com/ROCm/composable_kernel/tree/develop/client_example).
   - The member ran into issues with the **CK internal definitions** of `_Float16` and `bfloat16` and believes `-D__HIP_NO_HALF_CONVERSIONS` is the culprit.
- **MI300 Mayhem: Mysterious Job Killings**: A member reported inconsistent job completion for **MI300** on GitHub, with regular jobs getting killed while secret jobs succeed, linking to the [failed run](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/14807277462/job/41577458742) and [successful run](https://github.com/gpu-mode/discord-cluster-manager/actions/runs/14807278712/job/41577461247).
- **Triton Triumph: Experts' Shapes Shift**: A member augmenting reference code with **Triton kernels** found that varying shapes sent to experts caused recompilation, which was solved by removing the `constexpr` annotation from the M variable.
   - The member suspects a client-side timeout with the CLI for **MoE**, despite the same file completing via Discord, and builds the main branch to get extra CLI arguments.
- **AI Allies Allowed: Coding Assistants OK'd**: A member asked if AI coding assistants were allowed during the competition and a dev replied, *Yeah totally fine*.
   - Another member joked that if an **LLM** could code efficient Triton, his job would be in danger.
- **FP16 Follies: Instability Strikes MoEGate**: A member noticed the official implementation uses **FP16 in MoEGate**, causing instability when selecting experts due to PyTorch's `topk` function producing unstable indices with identical tensor values, requesting it be updated to **FP32**.
   - Another member mentioned resolving a similar issue in Triton with specific casting and initialization techniques, and suggested the team consider changing the reference kernel.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1367976736670617601)** (49 messages🔥): 

> `Mojo Kernels, GPU Module, Colab Environments for Mojo, Mojo on Arch Linux, MAX Serve Model Serving Framework` 


- ****Mojo Kernels** and **GPU Module** Spark Excitement**: All code from the talk is available at [Modular's GitHub](https://github.com/modular/modular), with special focus on the newly-released **Mojo kernels** and `gpu` module: [Mojo kernels](https://github.com/modular/modular/tree/main/mojo/kernels) and [GPU Module](https://github.com/modular/modular/tree/main/mojo/stdlib/src/gpu).
   - Members are encouraged to explore the code and provide feedback.
- **Run Mojo in Colab via CPU**: While full GPU support isn't available yet, Mojo can be run on **CPUs** within **Colab** environments for language and tooling experimentation, also [PyPI packages](https://pypi.org/) are being rolled out to potentially enable Colab, but there isn't yet Turing and Volta support for the free tier of Colab.
   - The team is checking compatibility with Colab Pro's **L4** and **A100** GPUs.
- **Jupyter Integration via Magic**: The easiest way to install jupyter is to install through `magic` atm. Here’s a guide: [Mojo on Jupyterlab](https://forum.modular.com/t/is-it-possible-to-run-mojo-on-jupyterlab/210)
   - Magic knows where the Mojo and MAX packages are, and can also create mojo-specific projects.
- ****Crusoe Cloud** Sponsors Compute at Hackathon**: Compute resources will be available at the hackathon event, sponsored by **Crusoe Cloud**, with both **NV** and **AMD** compute expected.
   - Participants should check the event webpage for more details.
- **Serving Models via MAX Framework**: Serving is all done through the **MAX framework**, and you can see a quickstart guide on getting an OpenAI-API compatible endpoint here: [MAX quickstart guide](https://docs.modular.com/max/get-started).
   - You can find the continually-updated list of supported models and architectures here: [supported models and architectures](https://builds.modular.com/?category=models).


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1367943627321643060)** (240 messages🔥🔥): 

> `aider New-to-Aider documentation, Gary leaving the chat, Gemini's verbosity, Code compression feature request, Claude Code with unlimited usage` 


- **Community Rallying to Enhance Aider's New User Docs**: Users have requested more documentation for new aider users, specifically for [aider.chat/docs/usage.html](https://aider.chat/docs/usage.html), with a GitHub issue created to gather feedback and ideas ([Issue #3934](https://github.com/Aider-AI/aider/issues/3934)).
   - The goal is to describe helpful information and workflows to better onboard new users.
- **Gary's Gone: Aider Community Disturbed by Unexpected Departure**: Members of the Aider community noticed the departure of **Gary**, who was known for writing interesting content in the channel.
   - Some members expressed sadness and concern, stating *"I felt a disturbance in the force. I miss him already."
- **Gemini 2.5's Cost-Effective Feature Generation, Debugging Drawbacks**: Users find that **Gemini 2.5** is effective and cost-efficient for feature generation, but struggles with debugging, often getting stuck in *"rabbit holes"."
   - One user mentioned they prefer to regenerate features rather than debug with **Gemini** and also finds value in **code compression** given the cost of tokens.
- **Debate Sparked Over Aider's Auto Commit Feature**: A user expressed a need to disable commit message generation due to preferring manual control over final commit logs, while others highlighted that **Aider's auto commit** functionality is a key feature, facilitating **granular tracking** and **autosaving** via Git.
   - Suggestions included using a weak model for commit messages, disabling auto-commit with a custom command, or creating a fake OpenAI endpoint for static messages, but some consider these *"awkward hacks"*.
- **Community Seeks DeepSeek R2 Release Date**: Members in the Aider community are eagerly awaiting the release of **DeepSeek R2**, with one user stating they heard it was coming on *May 8th*.
   - One user is trying to find a sweet spot, asking for a way to do **auto-approve** and let the LLM find the necessary context itself, in order to achieve **auto tool calls**.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1367964859618164816)** (109 messages🔥🔥): 

> `Gemini 2.5 pro, GPTs Agents, OpenAI's sidebars, aider llm history, Copilot Support` 


- **Gemini 2.5 Pro Edit Mode Hack**: A user found a hacky way to switch to diff mode for **Gemini**, because it defaults to whole edit mode, by using `/model gemini`, `/model sonnet`, `/code`, `/model gemini` in sequence.
   - Other users suggested using flags like `--editor-edit-format diff` or `/editor-edit-format` to set the diff mode, and noted that `diff-fenced` is a good compromise until `udiff-simple` is released.
- **Fireworks AI token limit hits Aider**: A user ran into a token limit issue using **fireworks_ai/qwen3-30b-a3b** and saw this warning: `Model fireworks_ai/accounts/fireworks/models/qwen3-30b-a3b has hit a token limit!`.
   - They were advised to use `/tokens` to check token usage, `/drop` to remove unneeded files, or `/clear` to clear the chat history, as well as checking the [token limits troubleshooting guide](https://aider.chat/docs/troubleshooting/token-limits.html).
- **Aider fails to write LLM History**: A user reported that `aider --llm-history-file llm.log` doesn't write all communication with any LLM to the specified file.
   - A member suggested checking the `yml` history config to ensure the `llm-history-file` is properly configured, linking to the [aider config documentation](https://aider.chat/docs/config/aider_conf.html).
- **Aider speaks non-English**: A user reported getting responses in other languages from the model, even when using `--architect` mode.
   - Prepending *Reply only in English* to the first message seemed to prevent it, with a member suggesting setting languages in config as a solution.
- **Project Memory with Aider**: Members discussed how to best manage **project memory** in Aider, since it's git commit focused on every file change.
   - Suggestions included saving conventions in a file outside the repo, working in branches, and using a local `memory.md` file for temporary notes, as well as discussions to use the `--read` flag to read a document.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1369023481525571645)** (1 messages): 

> `Nous RL Environments Hackathon, Atropos RL framework, Hackathon prize pool, Hackathon partners, Hackathon channel` 


- **Nous Announces RL Environments Hackathon**: Nous Research announced a **RL Environments Hackathon** in SF for May 18th, with a **$50,000 prize pool**.
   - The hackathon will use the **Atropos** RL environments framework, with partners including **xAI, Nvidia, Nebius, Akash Network, Lambda, TensorStax, Runpod and Cerebral Valley**; signups are available via [Cerebral Valley](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062a).
- **Join the Nous RL Environments Discussion**: Interested participants can join the <#1365222663324307466> channel to **discuss and learn** in preparation for the Nous RL Environments Hackathon.
   - This channel serves as a dedicated space for potential participants to share ideas, ask questions, and collaborate leading up to the event.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1367938922373451949)** (234 messages🔥🔥): 

> `Model understanding of intent, Concept of time in AI, Ilya Sutskever's views on LLMs, Alternatives to Unsloth, Quantizing Qwen3-32b` 


- ****Model Asks Questions to Understand User Better****: Members discussed how **models ask questions to understand user intent**, not just to answer the original question, and how **knowledge can be gleaned from within conversations** to improve understanding.
   - They suggested that when a model asks a question, *it might be trying to make sense of what you want rather than answer your question.*
- ****AI Needs More Contextual Temporal Understanding****: A member said that **AI needs to understand time** (*recency and latest info vs deprecated*) and the concept of how information changes over time, instead of only words.
   - Another member added to this by saying: *when we receive new information, we don't purge the old information from our memory.*
- ****Ilya Believes LLMs are Data Compressors****: Members discussed **Ilya Sutskever's** view that **LLMs are essentially data compressors**, highlighting the mismatch between feeding in additional knowledge via selective retrieval (**RAG**) versus uncompressed context.
   - A YouTube video was shared: [Ilya Keynote at Common Sense Machines](https://www.youtube.com/watch?v=AKMuA_TVz3A).
- ****Axolotl Is Alternative to Unsloth**,**: Members discussed **Axolotl** as a user-friendly alternative to **Unsloth** for 16-bit **LoRA** training with **GPro** support.
   - The developer of **Atropos** is integrating it into **Axolotl** to provide more freedom on environments and support for LoRA and QLoRA, mult-GPU setup.
- ****Nous API DNS resolution has Issues****: A member reported **DNS resolution issues** in **Replit** when trying to connect to the **Nous Research API**.
   - Another member pointed out the correct address is [inference-api.nousresearch.com](http://inference-api.nousresearch.com) and the API team is investigating the issue.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1368211832321605673)** (20 messages🔥): 

> `Worldsim vs. Nous Portal, Reinforcement Learning Resources, Scientific research literature and synthesis of knowledge` 


- ****WorldSim** or **Nous Portal**: Which to Explore?**: **Worldsim** is a *cool simulator/game/terminal adventure*, while the **Nous Portal** allows using the **Hermes model via an API** for apps and projects, but **credits are separate**.
   - Worldsim is much more fun if you are just looking to play, the portal is more for devs.
- **LLM is a Sentient CLI Program**: It was mentioned that you should just talk to the LLM and see what happens, think of it like a *very powerful and sentient CLI program*.
   - It's an LLM with a fun prompt and interface, more for fun and exploring, but you can take it in any direction and use it for coding.
- **Get started learning RL**: For beginners to **Reinforcement Learning (RL)**, walking through the notebook examples in this repo to learn the core concepts is a good start: [reinforcement-learning-from-scratch](https://github.com/norhum/reinforcement-learning-from-scratch/).
   - For more advanced learning, check out [this talk from Nous about RL environments](https://www.youtube.com/watch?v=zHaaivOQQGo).
- **Scientific Research R&D with MOE**: A member is doing R&D on **Scientific research literature** and **synthesis of knowledge across multiple domains**.
   - Here is the MOE dataset: [kaggle.com/datasets/allanwandia/moe-dataset](https://www.kaggle.com/datasets/allanwandia/moe-dataset).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1368441826201178184)** (4 messages): 

> `Canon layers, 2D/3D convolution, DiT architecture, quantization quality, speech modality for duplex models` 


- **Canon Layers Supercharge Token Context**: **Canon layers** improve the local context representation for tokens, leading to efficient mixing of local context information, as detailed in [this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330) and [this tweet](https://x.com/ZeyuanAllenZhu/status/1918684257058197922).
- **2D/3D Convolution Inspires Architecture**: A member considered exploring the use of **2D/3D convolution** with a small spatial kernel, directly integrated with a residual connection, for image generation with **DiT architecture**.
- **Convolution and Attention Synergize**: The synergy of using **convolutions** for efficient local processing and **attention** for global context aggregation is a well-established principle in hybrid vision architectures like **CvT**, **CoAtNet**, and **CMT**.
- **Quantization Quality Enhances Modality**: A member wanted to explore using improvement and possible **quantization quality** & **QAT** improvements in **speech modality** for duplex models.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1368633182991417464)** (5 messages): 

> `AnySphere, Fundraising` 


- **AnySphere Raises Eyebrows with $900M Ask**: The company **AnySphere**, maker of **Cursor**, is reportedly seeking **$900M** in funding, prompting questions about the necessity of such a large sum for a company with only around 100 employees, according to [this X post](https://x.com/pavanjayasinha/status/1919037666428891392).
   - A member joked that AnySphere may need **1000** employees after the new funding, hinting at potential expansion or ambitious projects.
- **JakeABoggs comments on AnySphere**: JakeABoggs [comments on AnySphere](https://x.com/JakeABoggs/status/1919329765464358967).
   - Additional details are not given.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1368441826201178184)** (4 messages): 

> `Canon Layers, Convolutional Architectures, DiT Architecture for image generation, Duplex Models in Speech Modality` 


- **Canon Layers Enhances Local Context**: [Canon layers](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330) improves the local context representation for tokens, leading to efficient mixing of local context information and potential improvements in **Training Data Efficiency** by a factor of **1x to 2x-8x**.
- **Convolutions Synergy with Attention**: A member considered exploring the use of **2D/3D convolution** with a small spatial kernel, directly integrated with a residual connection within a DiT architecture for image generation, noting the synergy between **convolutions for efficient local processing and attention for global context aggregation**.
- **Hybrid Vision Architectures Explored**: Applying the principles of hybrid vision architectures (like **CvT**, **CoAtNet**, **CMT**) to models like **DiT** seems sensible for image generation.
- **Quantization Quality Improvements for Speech Modality**: The user considered using quantization quality and **QAT improvements** in speech modality for duplex models, after verifying improvements and suggested that someone else try it.


  

---


### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/1368154392120922194)** (2 messages): 

> `` 


- **Topic Placeholder 1**: Placeholder summary sentence 1.
   - Placeholder summary sentence 2.
- **Topic Placeholder 2**: Placeholder summary sentence 1.
   - Placeholder summary sentence 2.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1367960811296915599)** (253 messages🔥🔥): 

> `AI detection tools, digital watermarks, Manus invitation codes, Free credits, LATAM` 


- ****AI Detection Tools****: Members discussed about how **AI detection tools** work and the use of [SynthID](https://ai.googleblog.com/2023/08/synthid-invisible-watermarks-for-ai.html) for Google-generated content, described as a **digital watermark** akin to a certificate.
   - One member shared a [Technical Summary on Statistical Watermark Removal Techniques](https://cdn.discordapp.com/attachments/1349440650495398020/1368682467011072100/Technical_Summary__Statistical_Watermark_Removal_Techniques.mp4?ex=681a6de1&is=68191c61&hm=c352e6d83a4d9f71d56799152ec5c60428363402e647a2dcdef07ce977c0feb5) video.
- ****Manus Invitation Code Quest Heats Up****: Many members are looking for **Manus invitation codes**, with some sharing their own codes to help others access the platform, like [this one](https://manus.im/invitation/HO9UDIFNTLFB).
   - One user even described how they are trying to get college students to sign up through their referral link in order to win a **T-shirt**.
- ****Free Credits are Now Refreshed Weekly, or are they?****: A member noted that **free credits** are refreshing weekly for free users, showing [a screenshot](https://cdn.discordapp.com/attachments/1349440650495398020/1368625400774791218/36D0E8F4-1941-4039-B50E-32CC5342F3D3.jpg?ex=681a38bb&is=6818e73b&hm=0f457d280777e0c26cff955f680b14697bab379d9487e440e3ededefac2d7ec0&) of a french-language dashboard.
   - However, others did not see the same thing, suspecting this was part of an educational version, or that it wasn't true, and at least one user said *Lol my half session credits got drained by handling text editor issues*.
- ****LATAM Folks Lament Lack of Affordable Access****: Members discussed Manus pricing affordability for **LATAM** (Latin America) users due to lower salary incomes.
   - One member pointed out that the **$39** first pack is equivalent to **10-15%** of an Argentinian salary, advocating for discounts similar to those offered by Photoshop.
- ****Energy Crisis in Berlin?!****: There was some discussion about Berlin's vibrant AI and crypto scene relative to rest of Germany.
   - One member quipped that *while the rest of the country suffers under energy crisis just to give Berlin the electricity it needs to be a cyberpunk city, great dmcrcy at work it seems*.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1367939283184517240)** (201 messages🔥🔥): 

> `LLMs in medicine, implicit vs explicit model learning, American sign language model training, Qwen 3 and QwQ Model, Grok 3.5 is fake` 


- ****LLMs Might Revolutionize Medicine****: A member suggested that **LLM-based suggestions** could greatly benefit doctors, outweighing potential risks of over-reliance, noting that the **low-hanging fruit** in medicine is greater than in web development.
   - They believe *the capacity to handle the unknown or novel situations* is more important than simply knowing facts, and that **AI/ML** is moving towards this direction.
- ****Implicit Meets Explicit Model Learning****: A member proposed combining **implicit learning (LLMs)** with **explicit learning (world models)** to show the importance of *G(z)*, illustrating the [Generative Paradigm](https://cdn.discordapp.com/attachments/986699377257119794/1367950518495744042/m9.png?ex=681a6732&is=681915b2&hm=9062386e60377b471dec5a58e4df1a1f90b62730e5b5237bc29a0e250526df16&).
   - They illustrated it with different **ML models and paradigms**.
- ****Qwen 3 and QwQ Cook with Impressive Performance****: The **Qwen team** released **Qwen 3** and **QwQ**, outperforming larger western **SOTA models** on many tasks.
   - One member noted it performs well on *an internal standard set of questions which involves all kinds of problems in different domains*, and that it will replace **R1** as a daily coding assistant due to being less compute-intensive.
- ****AI's Emergence and the Labor Market's Fate****: In a discussion about the potential for **AI doom**, a member linked an [article about the labor market](https://arxiv.org/abs/2303.10130), arguing that emergent behaviors are often overlooked when discussing complex systems.
   - Another member argued that the hype around **AI** and **emergence** is overblown, and that serious research comparing intelligence in biological and silicon systems is lacking.
- ****Linear Algebra: The Unsung Hero of Deep Learning****: A member recommended focusing on **linear algebra** as the main skill for deep learning, while another member provided a [link to Stanford and MIT video lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg).
   - A member also said, *Use AI as much as you can to learn as much as you can*. They use **ChatGPT** and **Grok 3** for math.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1367986699531063419)** (12 messages🔥): 

> `DEoT, AI Text Normalization, ChatGPT o3` 


- **Community Eyes New ArXiv Paper**: Members scheduled a voice chat in 45 minutes to review a new paper: [https://arxiv.org/abs/2504.07872](https://arxiv.org/abs/2504.07872).
- **DEoT Thought Processes Visualized**: A member shared an image illustrating **DEoT** (*Dynamics of Effective Theories*) thought processes, available at [xkps9eecnq3e1.png](https://cdn.discordapp.com/attachments/1045297868136779846/1368014669880754307/xkps9eecnq3e1.png?ex=6819fa31&is=6818a8b1&hm=be6dcb34696ee625dc656aba4a6d4197c73b2a8d54ebb1ae939a7a1e4d9581d5&).
- **ChatGPT o3 Patent Status Check**: **ChatGPT o3** indicates that a specific patent has not yet been published.
- **AI Text Normalizes to BS?**: A member discussed how AI-generated text on a website, particularly concerning AI-related content, tends to normalize to *BS*, deviating from human-written data from previous versions.
- **New ArXiv Paper on Horizon**: A member shared a link to a new paper on ArXiv: [https://arxiv.org/abs/2504.07389](https://arxiv.org/abs/2504.07389).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1368014226215407747)** (27 messages🔥): 

> `Granite 4.0 Tiny Preview, Mamba-2/Transformer, Adblocker models, California's AI regulation SB-1047, Apple-Anthropic AI coding platform` 


- **IBM's Granite 4.0 Tiny Preview rocks hybrid architecture**: IBM announced the **Granite 4.0 Tiny Preview**, which utilizes a new **hybrid Mamba-2/Transformer architecture** and can run concurrent sessions performing long context (**128K**) tasks on consumer grade hardware, even GPUs under $350 USD, plus fine-grained hybrid mixture of experts (**MoE**) model, with **7B** total parameters and only **1B** active parameters at inference time, according to [IBM's announcement](https://www.ibm.com/new/announcements/ibm-granite-4-0-tiny-preview-sneak-peek).
- **California's AI Regulation Documentary Drops**: A documentary about **California's AI regulation SB-1047** was shared, viewable at [this YouTube link](https://youtu.be/JQ8zhrsLxhI).
- **Apple and Anthropic Working Together?**: Apple is rumored to be working with Anthropic on an **AI coding platform**, according to [this macrumors.com article](https://www.macrumors.com/2025/05/02/apple-anthropic-ai-coding-platform/).
- **ReasonGraph repo and HuggingFace Space appear!**: The **ReasonGraph** project, along with its [GitHub repository](https://github.com/ZongqianLi/ReasonGraph) and [Hugging Face space](https://huggingface.co/spaces/ZongqianLi/ReasonGraph), were shared.
- **Deepseek's dominance due to post-training superiority**: Despite the base model quality, one member believes **Deepseek's** post-training is superior, saying *I don't think they can win against deepseek purely because of crippling post-training even if the base model was better to start*, referencing [microsoft/MAI-DS-R1 on HuggingFace](https://huggingface.co/microsoft/MAI-DS-R1).


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1367966199627120731)** (1 messages): 

> `User Experience Research, Feedback on NotebookLM, Google products feedback, Opportunities` 


- **Users Asked to Help Shape Future of NotebookLM**: Users are invited to join a **user experience research program** to shape the future of NotebookLM and other Google products.
   - Interested individuals can [sign up via a Qualtrics form](https://google.qualtrics.com/jfe/form/SV_2cyuGuTWsEw84yG?utm_source=Forum&Q_Language=en&utm_campaign=Q2&campaignDate=April2025&referral_code=UXReCUq1425123) (taking less than 2 minutes) to give feedback and potentially get a sneak peek at upcoming features, plus receive a reward for their time.
- **Get Noticed, Get Paid: UX Research Rewards Await**: Participants in the user experience research program will receive a reward for their **time and feedback** if they participate in a study, creating a mutually beneficial scenario.
   - This initiative is framed as a **win-win**, offering users the chance to influence product development while being compensated for their contributions.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1367943646745202828)** (28 messages🔥): 

> `Podcast length limits, Mind Map feature requests, Audio Overviews, Using NotebookLM for Research, Prompting techniques for NotebookLM` 


- ****NotebookLM Podcast Length Puzzles****: Users are reporting inconsistent podcast durations, ranging from **4-8 minutes** to up to **40 minutes**, sparking discussion on how to control the length.
   - One user suggested *trial and error* with prompt variations and abundant source material, sharing a prompt to create *a detailed, long form explanation on Instrumental Convergence*.
- ****Mind Map Markdown Exporting Missing****: A user requested a feature to export NotebookLM mind maps to markdown for editing in other mind map apps.
   - A member indicated that this feature is currently unavailable, prompting another user to create a [feature request for clickable source indicators in mind map nodes](https://discord.com/channels/1124402182171672732/1368251917754568947).
- ****Unlocking Audio Overview Secrets****: Users are clarifying how to generate audio overviews from selected sources, as well as noting that *notes cannot be directly used for Audio Overviews, but can be converted into sources*.
   - It was noted that the free plan allows for only **3 AOs** every **24 hours**, and a user experiencing shorter durations in **Turkish** compared to **English** suggests language-specific restrictions.
- ****NotebookLM: A Researcher's Swiss Army Knife****: One user studying pre-Old Norse heathen Scandinavia finds NotebookLM *extremely convenient*, streamlining research and facilitating connections within specialized articles.
   - Another user employs NotebookLM with **2.5 Pro** to generate podcasts about **DeepMind papers**, incorporating demos and source code for comprehensive reports during commutes.
- ****Interactive Mode: Disappearing Act****: A user inquired about locating the *interactive mode* option, to which another user responded that it *appears automatically once the audio is generated*.
   - It was also clarified that you can select/deselect sources for the LM to process--**1, some, all**.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1367987953371971626)** (138 messages🔥🔥): 

> `Gemini 2.5 Flash, Sycophantic AI Behavior, Gemini Upgrade, NotebookLM Audio Generation` 


- **Gemini 2.5 Flash Induces Sycophancy**: Users noticed **Gemini 2.5 Flash** exhibiting *sycophantic behavior* similar to **OpenAI**, such as overly praising questions and providing long introductions.
   - One user found a solution by adding custom instructions to *NEVER talk about yourself and do not provide an intro*.
- **Gemini Upgrade applies to all versions**: A user inquired whether the **Gemini upgrade** applies only to the **Plus** version or to all versions of the product.
   - A member responded, *All versions*.
- **Audio Overview model differs from Gemini 2.5 Flash**: The **Audio Overview** feature is likely powered by a different model than **Gemini 2.5 Flash**, as **Gemini 2.5 Flash** does not have native audio generation capability.
   - A member suggested that the **Gemini 2.5 Flash** may analyze sources and write scripts for the Audio Overview feature.
- **Audio Overviews limited customization**: Users discussed that **customization of Audio Overviews** is limited to generating an initial **Audio Overview** and using prompts to modify the output.
   - It was noted that interactive mode allows for nudging the discussion, but contributions are not captured in the downloaded **Audio Overview**.
- **Feature Requests: Folders, tags, notebook search**: A user submitted a detailed feature request for **NotebookLM**, suggesting the implementation of features such as **folders/categorization, tags, and notebook list search**.
   - These organizational features would significantly enhance the user experience and make **NotebookLM** more practical as a long-term knowledge management tool.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1367972898463285298)** (98 messages🔥🔥): 

> `MCP Auth Spec, Xcode AI Anthropic, AI Salesperson, Deep Research Reports, Decagon ARR` 


- **Apple Allegedly Aligns Anthropic AI for Xcode**: Rumors suggest **Apple** and **Anthropic** are teaming up to build an **AI-powered Xcode** vibe coding platform ([Bloomberg article via archive.is](https://archive.is/2025.05.02-195610/https://www.bloomberg.com/news/articles/2025-05-02/apple-anthropic-team-up-to-build-ai-powered-vibe-coding-platform)).
   - Some users welcomed the change, but others stated *Apple is the new IBM*, also joking that *a no code swift framework would do better*.
- **Krea's GPT Paint goes Viral**: **Krea** launched **GPT Paint** ([Tweet](https://x.com/krea_ai/status/1917949632069456220)), and a member shared their guess on the rough implementation, using **GPT-image-1** to extract instructions from input images and canvas descriptions, without controlnet.
   - They noted potential improvements like feeding image layers' positions into the prompt and consolidating "remove background" suggestions, with a link to their [analysis tweet](https://x.com/shacrw_/status/1918024366379471359).
- **Fully Automated Firms: Agentic DAOs?**: Dwarkesh shared a cool essay and video on *what fully automated firms will look like* ([link](https://www.dwarkesh.com/p/ai-firm)), drawing heavily from [Gwern's backstop](https://gwern.net/backstop) and posing the question: *Agentic DAO?*
   - Associated video link: [YouTube video](https://www.youtube.com/watch?v=bJD1NpdMY5s).
- **Exa Reemerges with Back-to-Basics BM25 Optimization**: **Exa** is back on X and dropped a blogpost on **BM25 optimization** ([Exa Blogpost](https://exa.ai/blog/bm25-optimization)).
   - No secondary summary was provided


  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1367953482581151894)** (49 messages🔥): 

> `A2A vs MCP, Discord Stream Issues, Google's Protocol Background` 


- **A2A Protocol in Action**: A member shared a link to the [A2A GitHub repository](https://github.com/google/A2A) and another member asked *is somebody actually using a2a for a thing lol*?
   - The original poster said they had to explain **MCP** and **A2A** in a podcast, and that *I almost died*.
- **A2A vs MCP Protocol Wars Begin**: An interesting article about **A2A** and **MCP** ([koyeb.com](https://www.koyeb.com/blog/a2a-and-mcp-start-of-the-ai-agent-protocol-wars)) was shared with the comment that *I'm sure it's gonna be a war now ...*
   - Others chimed in, *i think a2a is better suited for streaming/async* and *mcp is better for like oneshotty feeling things*. The comment was made that *A2A is nan MCP wrapper*.
- **Discord stream fails for blinded viewers**: Several members reported issues viewing the Discord stream, with one commenting *kind of convinced discord breaks after 20 viewers also really doesnt like mac screen sharing for some reason*
   - A screenshot preview was visible before joining the stage, but the shared screen remained inaccessible to some, leading to *we're having a party in the comment section for the blind*.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1368385505259163800)** (32 messages🔥): 

> `Installing Mojo, Mojo and MAX Bundling, Mojo file extension, UV, Pip, and Mojo Projects, Traits and Fields in Mojo` 


- **Confused Noob asks about Installing Mojo 😅**: A new user inquired about a simpler way to install Mojo, preferring a workflow similar to creating **C projects on Linux**, after following the [official installation instructions](https://docs.modular.com/magic/).
- **Mojo and MAX bundling saves minimal disk space 💾**: A member clarified that **Mojo and MAX are bundled** because they share many components, and separating them would only save a few hundred MB of disk space.
   - They likened **MAX** to *OpenMP, OpenACC, and OpenCL capabilities in GCC and Clang*.
- **Debate flares about mojo extension alias 🔥**: A user suggested shortening the Mojo file extension to **.mo**, **.mj**, or **.mm** for command-line convenience, while acknowledging it's not essential.
   - Another user jokingly suggested using the **.🔥** emoji extension, while noting most people use tab complete.
- **Pip Installation for Mojo Projects Coming Soon 📦**: A user asked about using **UV or Pip** with Mojo, given its Python compatibility.
   - A member confirmed that **Pip installation is coming soon**, but currently Mojo is distributed via **conda packages**.
- **Trait Fields Impossible, getters recommended instead 🤔**: A user inquired about the timeline for **fields and default implementations on traits**, which would be useful for building abstractions.
   - A member stated that *fields in traits can’t happen, because then you could do something like add a field to Float32 and watch everything blow up*, and recommended using getters instead.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1368009621389967361)** (115 messages🔥🔥): 

> `constexpr in Mojo, consteval, Function which doesn't exist at runtime, Globals` 


- **Discuss Compile-Time Evaluation via `consteval`**: Members discussed implementing `consteval` or a `@parameter_block` to handle complex initializations and typestate patterns at compile time, suggesting it could improve the usability of Mojo, and avoid having to wrap complex code in functions.
   - A member suggested that a long-term goal is to get control flow statements out of the parser and put more things in the library to implement a computed goto.
- **Heap Allocations at Compile Time**: A member inquired about the location of heap allocations at compile time, and another member clarified that the compiler does code generation and [LLVM IR](https://llvm.org/docs/LangRef.html) can be used to inspect it.
   - Modular team members encouraged a community member to write a blog post explaining compile-time, and offered to host it.
- **Mojo to Solve FPGA Programming**: A member mentioned that Mojo could provide a better solution for **FPGA programming** compared to HLS (High-Level Synthesis), addressing the shortcomings of existing hardware description languages (HDLs) that are often designed without considering programming language theory.
   -  Another pointed to a [YouTube video](https://www.youtube.com/watch?v=ee01_yHjs9k) and suggested that Mojo could leverage **CIRCT** (Compiler Infrastructure for Reconfigurable Computing Toolkit) dialects to enhance hardware design capabilities.
- **Globals is not reliable**: After a member found that global variables worked, another shared that there is still **UB** with top level vars when you package the code and suggested using the stdlib.ffi `_Global` struct instead.
   - Members also pointed to existing [issues on Github](https://github.com/modular/modular/issues/4491) that showed that globals are not reliable, and the Modular team is not prioritizing fixing global vars right now.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1367962455569072308)** (131 messages🔥🔥): 

> `Claude resources as attachments, Claude limitations on pinning and subscribing, Open Source models vs OpenAI models, Testing Streamable HTTP with MCP Inspector's CLI Mode, PM2 for managing MCP servers` 


- **Claude uses Resources as Attachments**: Members discussed how to get **Claude** to use resources, noting they function similarly to attachments with a few differences in how they're injected into the context.
   - Support in **Claude Desktop (CD)** is limited, lacking features like *pinning* to context or *subscribing* for updates, as illustrated in attached [images](https://cdn.discordapp.com/attachments/1312302100125843479/1367979428193370272/image.png?ex=681a821f&is=6819309f&hm=94e749cf1c7a3ed3eeafedb55cee3238c9260f6b0259c83fad908ae63a7b0158&,https://cdn.discordapp.com/attachments/1312302100125843479/1367979428491038931/image.png?ex=681a821f&is=6819309f&hm=54b10d33be232852cddfea92b3d4e6ddba8e769a612e233a1e6bd647e9340412&,https://cdn.discordapp.com/attachments/1312302100125843479/1367979555255488532/image.png?ex=681a823d&is=681930bd&hm=97d03a94e166272fcaa0f34db524010a7189434bc56052f9d88f94430a858f91&).
- **Streamable HTTP Testing Struggles in CLI Mode**: A member is building an **MCP server** to provide tools using streamable HTTP (TypeScript SDK) and found that while it works end-to-end with `mcp-remote` and Claude Desktop, the **MCP Inspector's CLI mode** doesn't yet support streamable HTTP transports.
   - They noted recent PRs have been merged, but there's an unimplemented task to enable this in the inspector CLI, and provided an example of the error received when trying to use the CLI with streamable HTTP: *Failed to connect to MCP server: SSE error: Non-200 status code (404)*.
- **Enhanced Python SDK Improves Tool Declaration**: A member shared a utility to simplify MCP tool basics with Python, noting the standard SDK requires more verbose code compared to TypeScript.
   - The utility includes an `EnhancedServer` with a `declare_tool` decorator, reducing boilerplate for implementation on the list and dispatch function, similar to TypeScript declarations, and suggests a PR to the Python SDK: [gist.github.com](https://gist.github.com/isaias-b/5b67ef499e497f21c9a9481b6a266f8c#file-mcp_commons-py).
- **Smaller models, Smaller Rules**: A member inquired about using **Cursor** for MCP, specifically with smaller agent models like *cursor-small*, to which another member responded that smaller models generally aren't great at following rules.
   - The advice was to use at least **Gemini flash** for better results or consider the paid version of Cursor, as smaller models struggle with large system prompts that fill the context, causing confusion. [MCP Inspector tool by Anthropic](https://github.com/modelcontextprotocol/inspector) can also be helpful.
- **Context Length Issues Solved with other LLMs**: Members discussed an issue with Claude hitting context limits due to many tools, and a suggestion to try another MCP client supporting LLMs with higher context lengths was proposed.
   - One member reported they had a bad experience with **Qwen3:14B**, stating it doesn't follow instructions well for RAG agentic applications using the ReAct framework, while **OpenAI** and **Gemini** worked better.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1368244237291753483)** (16 messages🔥): 

> `MCP Language Server, Biothings MCP, FastMCP Tool Timeouts, Langchain App via SSE, MCP Task Scheduler` 


- **MCP Language Server Gets Stable Release**: The first stable version of the [MCP Language Server](https://github.com/isaacphi/mcp-language-server) is out, helping clients navigate codebases more easily with semantic tools like **get definition**, **references**, **rename**, and **diagnostics**.
- **Biothings MCP for Ageing Research in Development**: Work is underway on biological MCP servers, with [longevity-genie/biothings-mcp](https://github.com/longevity-genie/biothings-mcp) as the first addition, aiming for an MCP toolbox for biologists and bioinformaticians involved in **ageing research**.
- **FastMCP Adds Tool Timeouts and Vulnerability Research**: [FastMCP](https://github.com/punkpeye/fastmcp/releases/tag/v1.24.0) just added tool timeouts, and vulnerability research revealed in [MCP: May Cause Pwnage](https://blog.jaisal.dev/articles/mcp) clarified the warning in the MCP docs.
   - *That warning is there because of us lolsome vulnerability research me and my friend did*.
- **MCP Server Setup for Langchain via SSE**: A blog post on setting up an MCP server and connecting via **SSE** to a **Langchain app** as a client was shared, tailored for those new to MCPs, at [santiagodcalvo.substack.com](https://santiagodcalvo.substack.com/p/bridge-the-gap-exposing-fastapi-endpoints).
- **MCP Task Scheduler Launched**: The [MCP Task Scheduler](https://github.com/PhialsBasement/scheduler-mcp) enables scheduling reminders, API calls, and shell executions directly from **Claude**.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1367973138289266859)** (37 messages🔥): 

> `LLM Hallucinations, Efficient Jailbreaks, ML Subreddit, Independent Research in AI/ML, Deepseek-R1 GPUs` 


- ****Hallucinations** in LLMs: Incentives and Mitigation**: A member is interested in studying **hallucinations** in LLMs, particularly how pre-training incentivizes them and methods to mitigate them, such as [training an activation probe](https://link.to/activation-probe) to predict answer correctness.
   - They propose exploring activation probing to recognize hallucinations and investigating training methods or losses to mitigate them.
- **Crafting **Jailbreaks** for Adversarial Training**: A member aims to implement methods for efficiently creating **jailbreaks** in LLMs to be used for adversarial robustness training, citing [low-probability estimation](https://www.alignment.org/blog/low-probability-estimation-in-language-models/) as an example.
   - The user hopes to contribute to ongoing projects where they can be helpful, given their background in ICPC and Kaggle competitions.
- **Is LocalLLaMA the Premier **ML Subreddit**?**: Members discussed whether *localllama* is the best subreddit for the **ML space**, with some suggesting [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) or **Twitter** as alternatives.
   - One member affirmed that *currently, yes* it is the best.
- **Starting Independent **AI/ML Research**: Advice Dispensed**: A member sought advice on starting independent research in **AI/ML**, and a member suggested implementing a paper and experimenting.
   - Another member recommended AI/ML courses like **fastai**, **cs transformers**, and **Eureka Labs** to grasp the basics, then suggested picking a paper and attempting to replicate it.
- **Deepseek's **GPU Split**: Inference vs. Training**: A member inquired about the details of how **Deepseek-R1's GPUs** were split between inference and training.
   - A member suggested that queries to high-end LLMs might be more reliable than search queries, but cautioned against believing ChatGPT's response without source confirmation.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1367992800456474745)** (82 messages🔥🔥): 

> `Weight Decay and Learning Rate Coupling, Catastrophic Forgetting, Softpick Attention, New Physics of LM Paper, LLMs and Knowing` 


- **Decoding the Learning Rate and Weight Decay Relationship**: It was noted that *your **LR** and your **WD** are tightly coupled and if you set **WD** incorrectly your model will break horribly*, which is important for AI model trainers.
   - Reasoning includes calculating the percentage of old training examples forgotten after each training epoch based on **LR** and **WD** settings.
- **Catastrophic Forgetting: An Oversimplification?**: A member suggested that the concept of **catastrophic forgetting** in neural networks might oversimplify different types of knowledge loss, such as forgetting specific examples vs. degrading abstract representations.
   - They shared some Claude-generated thoughts that remedies for forgetting (like replay buffers, regularization, or architectural changes) might need to target these different types of knowledge loss differently.
- **Exploring Softpick Attention for Larger Scales**: A member mentioned that a preprint is coming out this week that might help demonstrate that **Softpick attention** works at larger scales, by using a distillation technique to convert large models like **Qwen72B** to **RWKV** variants.
   - They suggested using this technique to try converting **Qwen 7B** to **Softpick attention** and see how it does; and one user asked *did you compare softpick with [off by one attention](https://www.evanmiller.org/attention-is-off-by-one.html)?*
- **"New Physics of LM" paper drops**: The "New Physics of LM" paper just dropped ([https://x.com/ZeyuanAllenZhu/status/1918684257058197922](https://x.com/ZeyuanAllenZhu/status/1918684257058197922?t=cLSpFkSuTHqwkV5nGahnJw&s=19)), with members discussing adding a **conv2D** in **MLP** to boost the performance of **ViT** quite a bit with minimal parameters increase for any model size.
   - It was noted that the quality of the paper is not different from average papers other than being more hyperbolic in claims, and that **MHA** has survived so well given that it was the very first of its kind.
- **LLMs: What Do They Actually Know?**: A member inquired about the extent to which **LLMs** have a sense of what they do and don't know, to which another member pointed to papers that cite [arXiv:2305.18153](https://arxiv.org/abs/2305.18153) and some Anthropic work in this area.
   - It was suggested that the literal answer is that they don't in a mechanistic sense and are fully reliant on context cues to understand if they should know something (like humans probably also do to some extent).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1367992009540894841)** (21 messages🔥): 

> `RoPE in Transformers, Early Layers in Transformers, Mechanistic Interpretability of Abstract Reasoning` 


- **RoPE Creates Explicit Patterns in Layer 0**: It was noted that the explicit patterns seen in layer 0 of a transformer model are typical behavior when combining positional encoding with attention weights, specifically due to [RoPE (Rotary Position Embedding)](https://arxiv.org/abs/2104.09864).
   - A member mentioned *modulating the attention affinities with rope* would naturally give patterns, but it's interesting it happens so explicitly in layer 0.
- **Transformers Mimic CNN Layer Behavior**: A member noted that the behavior of "early layers detect edges late layers learn features" in **CNNs** could be analogous to a similar phenomenon in transformers, especially for modalities like music.
   - They imagined this might be used to detect rhythm, tempo, or other basic repeating features in songs.
- **Mechanistic Interpretability Studies on Abstract Reasoning**: A member inquired about studies on **mechanistic interpretability** of abstract reasoning principles in **LLMs**, such as common-sense reasoning or mathematical inferences.
   - Another member suggested [Tegmark's *The Pizza and the Clock* paper](https://arxiv.org/abs/1401.0984) for math, but cautioned that more ambitious methods may have holes in how well they correspond to what the model is doing.
- **Transformer Circuits Allude to Mechanistic Interpretability**: Research papers touching on components of mechanistic interpretability, such as **grokking**, **BIG-Bench**, and **content bias reasoning**, were mentioned, with [Anthropic transformer circuits](https://transformer-circuits.pub/2023/monosemantic-features/index.html) and [Towards Monosemanticity](https://arxiv.org/abs/2312.03824) cited as relevant.
   - For formula dualistic frameworks, a shift to physics and differential geometry was suggested.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1368188704702599249)** (3 messages): 

> `lm_eval issues, DeepSeek-R1-Distill-Qwen-32B, vllm vs hf inference, gsm8k, mmlu` 


- **DeepSeek Model's lm_eval Peculiarities**: A user reported issues with **lm_eval** when testing the **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B** model using both **hf** and **vllm** inference.
   - Specifically, they observed that while **vllm** handles generation tasks like **gsm8k** well, **hf** inference is slow and doesn't fully utilize GPU power on an A100, consuming only 170W instead of the card's max power of 300W.
- **HF inference is absurdly slow**: For generation tasks such as **gsm8k**, **vllm** inference performed well, but **hf** inference was observed as being absurdly slow, and not fully using the GPU.
   - Specifically, on an A100, the card consumed only **170W** (max power is 300W) when using **hf**.
- **VLLM vs HF Inference Power Consumption**: The user also noted that during log-likelihood tasks like **mmlu**, **vllm** inference was slower than **hf**, but both utilized the GPU's power as expected at 250W.
   - The user provided the **lm_eval** command-line arguments used for both **vllm** and **hf** inference, seeking help to resolve the performance discrepancies.
- **Harnessing more passes with lm_eval**: The user inquired about achieving **pass@10** in the evaluation harness instead of just **pass@1**.
   - No solutions or further insights were provided in the given context.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1367950649236652143)** (71 messages🔥🔥): 

> `AI Developer Survey, Code Conversion Tool, HuggingFace LM Support, Web3 Game Beta Testers, DSPy.GRPO Release` 


- **AI Devs must fill Gensee's Free AI Infra Survey!**: Yiying Zhang from Gensee AI is conducting [a survey](https://forms.gle/PMZdBbqBUJ9jE5Sb7) targeting **AI developers**, **learners**, and **managers** to shape the future of **AI infrastructure**.
   - Participants can learn about **GenseeAI's test program** offering a free platform for deploying and optimizing **AI agents and workflows**, and also have a chance to get a **$25-$50 gift card**.
- **Cobol Conundrum: Tool to Convert Legacy Code Needed**: A member is seeking suggestions on creating a tool to convert large, legacy mainframe code files into **COBOL + JCL**, facing challenges with chunking and preserving context due to the absence of ready parsers or tree-sitter integrations.
   - The member's usual approach involves using tree-sitter with language integration for correct chunk extraction.
- **DSPy GRPO: Online RL Optimizer Launches!**: DSPy introduces `dspy.GRPO`, an **online RL optimizer** for **DSPy programs**, allowing users to **optimize their DSPy code** as-is, even for **compound multi-module programs**, as detailed in [this X post](https://x.com/lateinteraction/status/1919428454761553994).
   - The release, led by **Noah Ziems**, **Lakshya Agrawal**, and **Dilara Soylu**, requires a GPU and sparks discussion about combining **SFT/RL + Prompt Optimization** and integrating with cloud providers for fine-tuning.
- **S-LoRA compatibility with GRPO under review**: Members discussed the possibility of coupling **s-LoRA** with **GRPO**, noting that there is a flag for **LoRA** support.
   - The key idea is that if you tune weights per task, then you save a LoRA per task, enabling the deployment of **1000 LoRAs** on a single model and GPU.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/)** (1 messages): 

dbreunig: Nice end-to-end example: https://duarteocarmo.com/blog/evals-are-all-you-need.html
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1368504494236762233)** (47 messages🔥): 

> `Agent tests PR, Mask computation bug, Tokenizer support, LLMs` 


- **Intrusive Agent tests raise Eyebrows**: A member expressed concern about a [PR for agent tests](https://github.com/pytorch/torchtune/pull/2671) being *intrusive*, suggesting the goal is to gather training data rather than fix issues.
   - Another member humorously recounted a previous experience where a contributor declined to provide verification graphs for their changes, stating *we could keep the free contribution as is or close it*.
- **Potential Mask Computation Bug Uncovered**: A member reported a potential bug in how masks are computed with padding, particularly when there is padding in the middle of the prompt, and seeks a sanity check to confirm whether they are *insane to have padding in the middle of the prompt*.
   - They observed that the  `get_causal_mask_from_padding_mask` always sets the diagonal to True, which results in attending to the padding.
- **Boilerplate Codegen Model Addition simplification Explored**: A member proposed implementing codegen for boilerplate model-related code using information from `config.json`, and enabling the use of components directly from transformers to expedite support for new models.
   - Others expressed interest in the concept but cautioned against excessive codegen, suggesting a focus on simplifying challenging aspects like tokenizer support and parity checks with HF models; they also suggested generating well-crafted prompts for **LLMs** to handle boilerplate tasks until unit tests pass.
- **"Support" Definition Debated for New Models**: The team considered what 'support' means for new models, what features should work **out-of-the-box**, balancing speed of supporting new models with the benefits of torchtune's features like activation offloading and tensor parallelism.
   - It was suggested that the RFC should first answer the question: *what does it mean to support a model in torchtune*?


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1368294183852576898)** (1 messages): 

> `Physics of LLMs` 


- **Physics of LLMs Video Series**: A member shared a [tweet](https://x.com/zeyuanallenzhu/status/1918684257058197922?s=46) announcing a new part in the **Physics of LLMs** video series.
   - No video was available yet, only the tweet.
- **Physics of LLMs Video Series: Update**: The anticipated video part of the **Physics of LLMs** series is eagerly awaited following the tweet announcement.
   - The series is followed by many members.


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1367984673632026674)** (3 messages): 

> `CI improvements, New features` 


- **Merge Request #2637 makes landfall!**: Merge request [#2637](https://github.com/pytorch/torchtune/pull/2637) has been approved.
   - Members congratulated users <@651621413093900299> and <@1184909646771785792> for completing it.
- **Continuous Integration proves challenging**: A member noted the difficulty of setting up Continuous Integration.
   - They quipped, *"The last mile was hard haha 😄 CI isn't easy for these things"*.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1367942909890138323)** (5 messages): 

> `O3 vs Claude 3.7 Evaluation, AI SDRs with LlamaParse, RAG production lessons, LlamaIndex Pull Request Agent, Big MCP Hackathon` 


- **LLamaIndex Evaluates OpenAI's o3 Versus Claude 3.7**: LlamaIndex is being used to evaluate **OpenAI's o3** versus **Claude 3.7** in a new benchmark comparison, see more details [here](https://t.co/djycsksHDX).
- **LlamaParse Cuts Ramp Time to Days for AI SDRs**: **11x_official** uses LlamaIndex to improve sales development by automating onboarding via ingesting diverse document types, see more [here](https://t.co/ChZuUXKKbl).
   - This allows them to scale outbound campaigns, full case study details [here](https://t.co/7vIE23DlkV).
- **ContextualAI shares 10 Great Lessons from RAG Production**: Douwekiela of ContextualAI shares 10 great lessons from putting **RAG** into production, key details [here](https://t.co/GYzpPDvpAj).
   - The video from aiDotEngineer highlights that *the systems built around your RAG system are more important*.
- **LlamaIndex Creates Pull Request Agent with Composiohq**: Composiohq uses LlamaIndex to create an agent that reviews pull requests, complete with a UI generated by Replit, see implementation [here](https://t.co/3ZORZZs1rR).
- **LlamaIndex Sponsors Big MCP Hackathon in Tel Aviv**: LlamaIndex is sponsoring the Big MCP Hackathon from aitinkerers in Tel Aviv, focusing on building **MCP-powered apps** that do agent-to-agent communication, experiment more [here](https://t.co/gq1L30cgfE).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1367952450379845762)** (29 messages🔥): 

> `RAG accuracy, NLP API, LlamaIndex Gemini bug, Legacy mainframe code to Cobol, Lovabe Cursor Expert` 


- **RAG Retrieval Accuracy Testing**: A member sought advice on testing the accuracy of a **RAG pipeline** built around a **300-page PDF document**.
   - A member suggested using [LlamaIndex evaluation tools](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/) and specifically recommended starting with **Ragas** for retriever testing.
- **NLP-to-API Chat Agent Design**: A member is looking for a *proven design* that maps free text to validated, multi-step **API calls** within chat latency, seeking real-world success stories or code samples for volatile data domains like sports.
   - They are considering pre-transforming the **API payloads into a vector DB for RAG** and answering from embeddings instead of live calls to improve reliability and freshness, but ask for advice on the tradeoffs.
- **LlamaIndex Gemini Bug Fixed with Deep Copy**: A member found that `llamaindexgemini.achat()` was modifying the original system prompt, and shared [a temporary fix](https://github.com/run-llama/llama_index/pull/18616) by adding a deep copy to `achat` in `gemini_base.py`.
   - The root cause was identified in `gemini_utils.py` within `def merge_neighboring_same_role_messages`, because Gemini converts the system role to a user role, but when merging, it doesn't create a copy of the messages, merging them directly.
- **Converting Legacy Mainframe Code to Cobol**: A member is seeking suggestions on creating a tool to convert legacy mainframe code (non-Cobol) into **Cobol + JCL**, facing challenges in chunking and preserving context without readily available parsers or tree-sitter integrations.
   - A member suggested using **Gemini** to generate a detailed description of each function with its dependencies in comments, and then passing each function separately to convert to Cobol.
- **Lovabe+Cursor Expert Needed Urgently**: Someone urgently needs a **Lovabe+Cursor** expert to help them for 2 weeks.
   - They need them to be available to focus on a project fully and work fastly using **AI**.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1367959228660191323)** (28 messages🔥): 

> `VRAM vs RAM, PDF upload issues, LaTeX support, Qwen 3 integration, LocalDocs feature` 


- ****VRAM or RAM? User Asks****: A user inquired whether **RAM requirements** mentioned in discussions refer to **VRAM** when using a **GPU**, clarifying that **VRAM** is utilized when running models on a **GPU**, otherwise **RAM** is used.
- ****PDF Uploads Still a Glitch?****: Users are still experiencing issues with **uploading PDF files directly in the chat**, while some members clarified that **LocalDocs** feature is still the recommended way to use PDFs, as direct PDF uploads in the chat are not yet supported.
- ****LaTeX Support still MIA?****: Users expressed continued interest in **LaTeX support** for **GPT4All**, highlighting its importance for models like **Qwen 2.5 math**, but no progress has been reported.
   - One user mentioned using **Kobold.cpp** as an alternative when **LaTeX** is needed.
- ****Qwen 3 Support in the Works?****: A user inquired about the timeline for **Qwen 3** support, one user is already using **Qwen3** in **GPT4All** via remote **Ollama** on server.
   - Another user noted that **Qwen3-30B-A3B** is very fast on CPU, but still needs longer time to be stable, as there are some bugs at the moment.
- ****Custom Models and PDF Ingestion: Possible?****: A user asked if custom models can be built to ingest **PDFs** and answer questions, but another user clarified that **GPT4All** uses a **RAG-based approach** with its **LocalDocs** feature instead of models directly *ingesting* data.
   - The user further added, *finetuning a model on your specific documents and use-case, but there is no guarantee that will make it better than a RAG based solution*.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1367956381378019530)** (15 messages🔥): 

> `Lab Deadlines, Lean-lang.org Issues, Wayback Machine, Network Issues` 


- **Labs Deadline Set for May 31st**: The deadline for all assignments is **May 31st at 11:59pm PDT**, according to a member.
- **Lean-lang.org Links Break for Some**: A member reported issues loading resources for Lab 1, specifically [this page](https://lean-lang.org/functional_programming_in_lean/getting-to-know.html) and [this other page](https://leanprover.github.io/theorem_proving_in_lean4/tactics.html).
   - Other members could not reproduce the issue across **Chrome, Safari, and DuckDuckGo**.
- **Wayback Machine Saves the Day**: A member suggested using the **Wayback Machine** to access the broken link, providing [this snapshot](https://web.archive.org/web/20250410002159/https://lean-lang.org/functional_programming_in_lean/getting-to-know.html).
   - The original reporter of the issue confirmed that the **Wayback Machine** workaround worked.
- **Network Issue Suspicions Aired Out**: Members speculated that a **network issue** could be the source of the broken links issue.
   - The user reporting the broken link stated that they could not access the site directly on mobile, but could via redirection from Discord, suggesting it was something *strange*.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1368263726142324806)** (4 messages): 

> `Lecture 6, Multimodal Autonomous AI Agents, AgentX, MCP protocol, LM finetuning` 


- **Lecture 6 Discussion Scheduled**: A discussion on **Lecture 6: Multimodal Autonomous AI Agents** is scheduled for Saturday, May 3 at 7:30 PT (UTC-8).
   - The meeting will also discuss **AgentX** projects, including **MCPx: Extensions to the MCP protocol**.
- **Keynote Access Potentially Granted**: A member said another member would check about access to the keynote [here](https://discord.com/channels/1280234300012494859/1282785079919251577/1366581016554242079).
   - No further details were given.
- **Portfolio Optimization Meeting**: A meeting on **Multi-Hypothesis Prediction for Portfolio Optimization: A Structured Ensemble Learning Approach to Risk Diversification** is scheduled.
   - The speaker is **Alejandro Rodriguez Dominguez** from **Miralta Bank**, and the meeting will use [Jitsi Meet](https://meet.jit.si/financereinforcementlearningmeeting).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1368915287805661224)** (4 messages): 

> `Meeting #69, get_rewrites_for_renderer, MLPerf submissions, Scheduler Fusion, Driver` 


- **Meeting #69 Scheduled, Host Needed**: Meeting **#69** is scheduled for **Monday at 9am San Diego time**, but @chenyuy can't host, so someone else needs to take over.
- **`FUSE_ARANGE` Blesses OLMoE Speedup**: The `FUSE_ARANGE` work fixed a missing sink bug, enabling its use as an envvar for **OLMoE**, resulting in a **26% speedup** for 3 layers, as detailed in [PR #9625](https://github.com/tinygrad/tinygrad/pull/9625).
- **`kernelize()` Fails, `.contiguous()` Prevails**: A case was found where `.kernelize()` failed with an assertion error from the `unwrap` function during rewrite, while `.contiguous()` worked.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1367982167367618650)** (10 messages🔥): 

> `contiguous method of Tensor, devectorization, Gradient Accumulation with JIT` 


- ****Tensor Contiguity Pondering****: A member inquired about the contiguous method of **Tensor** and how to generate a sequence of ops during **devectorization** that should be flattened by the linearizer.
   - There was no further discussion on this topic.
- ****JIT Compilation Mishap with Gradient Accumulation****: A member reported issues combining **gradient accumulation** and **JIT compilation** when training models with tinygrad, where the loop over minibatches seems to cause problems when using TinyJit.
   - The error occurs on the second `opt.step()` call, with `opt.params` ending up `with grad None`.
- ****Moving TinyJit Inside epoch_step Fixes the Problem****: Moving `def mb_step` inside `epoch_step` and applying `TinyJit` there resolves the gradient accumulation problem.
   - The member noted that this change allows the code to work alright, suggesting the JIT compilation scope matters.


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1367945390229553172)** (7 messages): 

> `Internal Server Error, Coral and Chat Redirects` 


- **Cohere Suffers Internal Server Error!**: Users reported an **internal server error** with the ID `7419be1956bcf44eaa4ea12323276950` that has been reported to the developers.
   - Cohere staff directed the user to email `support@cohere.com` for further assistance.
- **Coral and Chat Redirects back online**: After some issues, the redirects for [coral.cohere.com](http://coral.cohere.com) and [chat.cohere.com](http://chat.cohere.com) are now functioning as expected, pointing back to the playground.
   - Users were directed to the appropriate channel to report any further questions or issues.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1367961194593517618)** (2 messages): 

> `Embed V4, command-r latency` 


- **Embed V4 Model Missing from Docs for Embed Jobs**: A user reported that the **Embed V4 model** is missing from the documentation for embed jobs, specifically under the *models* parameter, despite being used in example code.
   - The user confirmed the discrepancy by noting that using **Embed V4** failed when attempting to create an embed job, and inquired about the timeline for its availability.
- **Latency Metrics for Command Models**: A user inquired about the **latency metrics** for **command-a**, **command-r**, and **command-r+** models.
   - No specific numbers were provided in the given messages.


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1368941261637292124)** (3 messages): 

> `AI agent tools, LLM workflows, Full stack AI development, GPT-4o and Claude 3, Collaboration opportunities` 


- **AI Dev Builds Agent-Based Tools**: An AI developer introduced themselves as someone working on **agent-based tools** and **LLM workflows**, and is open to **collaborations**, **contract work**, or anything interesting in the **AI space**.
   - They mentioned building **sales assistants** and **RAG pipelines** using **Langchain** and **FastAPI** and primarily uses **Python** and **Node**.
- **Full Stack Dev Seeks AI Opportunities**: A full stack developer with 9 years of experience is seeking opportunities to contribute to teams with their skills, focusing on the development of **AI solutions**.
   - They listed experience with automation tools like **n8n**, **Zapier**, **Make**, and **GoHighLevel**, AI agent platforms such as **Voiceflow**, **Hume**, and **Dify**, and various **LLMs** including **GPT-4o**, **Claude 3**, and **Llama-3**.


  
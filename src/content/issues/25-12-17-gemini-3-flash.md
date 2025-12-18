---
id: MjAyNS0x
title: >-
  Gemini 3.0 Flash Preview: 1/4 cost of Pro, but ~as smart, retakes Pareto
  Frontier
date: '2025-12-17T05:44:39.731046Z'
description: >-
  **Google** launched **Gemini 3 Flash**, a pro-grade reasoning model with flash
  latency, supporting tool calling and multimodal IO, available via multiple
  platforms including Google AI Studio and Vertex AI. It offers competitive
  pricing at $0.50 per 1M input tokens and $3.00 per 1M output tokens, with
  context windows up to 1M tokens. Benchmarks show **Gemini 3 Flash** rivals or
  outperforms larger models like **GPT-5.2** and **Gemini 3 Pro** in agentic,
  coding, and reasoning tasks, validated by ARC-AGI-2, SWE-bench, LMArena, and
  Arena benchmarks. Despite some tradeoffs like high token use and hallucination
  rates, it is cost-effective overall. Key figures include **Sundar Pichai**,
  **Jeff Dean**, and **Demis Hassabis** who publicly celebrated this
  achievement. The model's tool calling capabilities were demonstrated with 100
  tools in a live demo.
companies:
  - google
  - google-deepmind
models:
  - gemini-3-flash
  - gemini-3
  - gpt-5.2
  - gemini-3-pro
topics:
  - tool-calling
  - multimodality
  - benchmarking
  - reasoning
  - cost-efficiency
  - model-performance
  - context-window
  - agentic-ai
  - model-deployment
people:
  - sundar_pichai
  - jeffdean
  - demishassabis
---


**Gemini is all you need.**

> AI News for 12/16/2025-12/17/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (207 channels, and 8313 messages) for you. Estimated reading time saved (at 200wpm): 594 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

When we first started pushing the LLM Pareto frontier [a year ago](https://x.com/Smol_AI/status/1838663719536201790), and then it was picked up by Jeff Dean and Demis Hassabis, it wasn't long before [Gemini 2.5 conquered it](https://news.smol.ai/issues/25-04-17-ainews-gemini-25-flash-completes-the-total-domination-of-the-pareto-frontier), before [GPT-5](https://news.smol.ai/issues/25-08-07-gpt-5) then claimed it 4 months after. Now we are back to Gemini 3.0 claiming it, again witih [Sundar](https://x.com/sundarpichai/status/2001326061787942957?s=20) and [Jeff](https://x.com/JeffDean/status/2001323132821569749?s=20) loudly trumpeting this accomplishment:

![A performance comparison chart of Gemini AI models showing their benchmarks and positioning across different metrics.](https://resend-attachments.s3.amazonaws.com/XLrtCGUX3eEtvl2)

Apart from Arenas, this is also validated in [academic](https://x.com/officiallogank/status/2001368440016392314?s=46) benchmarks:

![A performance comparison chart of AI models across various benchmarks, highlighting Gemini 3 Flash's competitive performance against larger models like Gemini](https://resend-attachments.s3.amazonaws.com/147bmHMDE54uirA)

and [ARC AGI](https://x.com/fchollet/status/2001330643423449409?s=46) has its own chart showing efficiency:

![A performance comparison chart of AI models across various benchmarks, highlighting Gemini 3 Flash's competitive positioning against other models like GPT-](https://resend-attachments.s3.amazonaws.com/RxPdsx8NbSVZnVz)

Here are some specific breakdown [highlights](https://x.com/kimmonismus/status/2001326181875154983?s=46):

![A detailed performance comparison table of AI models across various benchmarks, highlighting Gemini 3 Flash's competitive performance against larger models like Gem](https://resend-attachments.s3.amazonaws.com/04KG7PfuOhez2Ob)

Apart from the disillation, the focus here seems to be [tool calling](https://x.com/0xdevshah/status/2001330346961604732?s=46). Here is a [demo showing 100 tools](https://x.com/googleai/status/2001323069105692914) and more demos from [Addy Osmani](https://x.com/addyosmani/status/2001324727504359745).

---

# AI Twitter Recap

**Gemini 3 Flash launch: frontier intelligence at flash latency (ecosystem, metrics, caveats)**

- **Model + rollout**: Google launched **Gemini 3 Flash**, positioned as “Pro‑grade reasoning at Flash speed.” It’s the new default in the Gemini app (“Fast”) and Search AI Mode, and available to developers via the Gemini API in Google AI Studio, Antigravity, Vertex AI, CLI, Android Studio, and more. Pricing is $0.50 per 1M input tokens and $3.00 per 1M output tokens; context up to 1M tokens; tool calling and multimodal IO supported. Announcements and overviews: [@sundarpichai](https://twitter.com/sundarpichai/status/2001326061787942957), [@Google](https://twitter.com/Google/status/2001322381533409733), [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2001321759702663544), [@OfficialLoganK](https://twitter.com/OfficialLoganK/status/2001322275656835348), [@JeffDean](https://twitter.com/JeffDean/status/2001323132821569749), [@demishassabis](https://twitter.com/demishassabis/status/2001325072343306345), [@GeminiApp](https://twitter.com/GeminiApp/status/2001412101286563865), [dev Q&A space](https://twitter.com/GoogleAIStudio/status/2001330099841556490).
- **Benchmarks and cost/perf**: Early results show 3 Flash rivaling or outperforming larger models in several agentic/coding and reasoning settings at markedly lower cost/latency:
    - ARC‑AGI‑2 and SWE‑bench Verified: beats or matches Gemini 3 Pro and rivals GPT‑5.2 in some configs ([@fchollet](https://twitter.com/fchollet/status/2001330643423449409), [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/2001321765503377546), [@jyangballin](https://twitter.com/jyangballin/status/2001336879120363639), [pareto snapshots](https://twitter.com/arena/status/2001389914760581533)).
    - LMArena and Arena (WebDev/Vision): top‑tier scores with strong pareto position vs price ([@arena](https://twitter.com/arena/status/2001322123730788698), [@JeffDean](https://twitter.com/JeffDean/status/2001335803642024157), [@osanseviero](https://twitter.com/osanseviero/status/2001323721232163053)).
    - Independent aggregation notes both strengths and tradeoffs: high knowledge/reasoning, second on MMMU‑Pro, but heavy token use and high hallucination on AA‑Omniscience (91%)—cost-effective overall due to pricing ([Artificial Analysis deep dive](https://twitter.com/ArtificialAnlys/status/2001335953290670301), [follow‑ups](https://twitter.com/ArtificialAnlys/status/2001335963952521243)).
- **Thinking levels and evaluation**: Flash exposes thinking levels (low/med/high). Practitioners asked for level‑wise benchmarks to inform production tradeoffs; some early tests show Flash‑Low is token‑efficient but weaker on validity, while Flash‑High closes gaps on quantitative metrics ([@RobertHaisfield](https://twitter.com/RobertHaisfield/status/2001327612887785904), [@Hangsiin](https://twitter.com/Hangsiin/status/2001341564145250770), [Flash‑Low vs High snapshots](https://twitter.com/scaling01/status/2001359254578753852)).
- **Integrations and tooling**: 3 Flash is already live in common dev environments: Cursor ([@cursor_ai](https://twitter.com/cursor_ai/status/2001326908030804293)), VS Code/Code ([@code](https://twitter.com/code/status/2001335940934246503), [@pierceboggan](https://twitter.com/pierceboggan/status/2001327058425917795)), Ollama Cloud ([@ollama](https://twitter.com/ollama/status/2001372370469290280)), Yupp ([@yupp_ai](https://twitter.com/yupp_ai/status/2001340530828206586)), Perplexity ([@perplexity_ai](https://twitter.com/perplexity_ai/status/2001333781081526611); [Flash in Pro/Max](https://twitter.com/perplexity_ai/status/2001447398317724153)), LlamaIndex FS agent ([demo](https://twitter.com/llama_index/status/2001324278617424017), [repo](https://twitter.com/jerryjliu0/status/2001335494534402521)). Early product notes highlight near‑real‑time coding/editing and multimodal analysis ([@Google](https://twitter.com/Google/status/2001397324551946523), [@GeminiApp](https://twitter.com/GeminiApp/status/2001351746338329063)).

**Voice AI and embodied assistants**

- **xAI’s Grok Voice Agent API**: New speech‑to‑speech agent supports tool calling, web/RAG search, SIP telephony, and 100+ languages. It posts a new SOTA on Big Bench Audio (92.3% reasoning), ~0.78s TTFB, at $0.05/min ($3/hr). Rapidly demoed on the Reachy Mini robot within an hour of launch, hinting at fast path from voice reasoning to embodied agents ([xAI](https://twitter.com/xai/status/2001385958147752255), [benchmark write‑up](https://twitter.com/ArtificialAnlys/status/2001388724987527353), [robotics port](https://twitter.com/ClementDelangue/status/2001410494528213481)).
- **Real‑time speech infra**: Argmax SDK 2.0 shipped “Real‑time Transcription with Speakers”—faster than real time on Mac/iPhone, under 3W power, “step change” in accuracy ([@argmax](https://twitter.com/argmax/status/2001296557556040028)). This, together with Grok Voice, strengthens the stack for production voice agents.

**Training efficiency and MoE systems**

- **FP4 training and open MoE stack**: Noumena released “nmoe,” a production‑grade reference path for DeepSeek‑style ultra‑sparse MoE training focused on B200 (SM_100a), with RDEP (replicated dense/expert parallel), direct dispatch via NVSHMEM (no MoE all‑to‑all), and mixed‑precision experts (BF16/FP8/NVFP4). Emphasis on deterministic mixtures and router stability at research scale. Authors claim NVFP4 training is “solved” for MoEs when properly applied ([repo + thread](https://twitter.com/_xjdr/status/2001434891087671779), [earlier FP4 note](https://twitter.com/_xjdr/status/2001234330236940444); related: torch._grouped_mm discovery [link](https://twitter.com/_xjdr/status/2001231675066396837)).
- **Inference/system throughput**: vLLM reports up to +33% Blackwell throughput in one month via deep PyTorch integration, cutting cost/token and lifting peak speed ([@vllm_project](https://twitter.com/vllm_project/status/2001449658984632699)).
- **On‑device LLMs**: Unsloth + PyTorch announced a path to export fine‑tuned models to iOS/Android; e.g., Qwen3 on Pixel 8 / iPhone 15 Pro at ~40 tok/s, fully local ([@UnslothAI](https://twitter.com/UnslothAI/status/2001305185206091917)).
- **RL/FT insights**: Small‑scale RL LoRA on Moondream suggests “reasoning tokens” and RL both improve sample efficiency, with MoE also helping—at the cost of more fine‑tuning compute ([setup/results](https://twitter.com/vikhyatk/status/2001232634584948878), [commentary](https://twitter.com/vikhyatk/status/2001233256356962512)).

**Interactive world models, video, and 3D assets**

- **Tencent Hunyuan HY World 1.5 (“WorldPlay”)**: Open‑sourced, streaming video diffusion framework enabling real‑time, interactive 3D world modeling at 24 FPS with long‑term geometric consistency. Introduces “Reconstituted Context Memory” to rebuild past frame context and a Dual Action Representation for robust keyboard/mouse control. Supports first/third person, promptable events, infinite world extension ([launch thread](https://twitter.com/TencentHunyuan/status/2001170499133653006), [paper](https://twitter.com/_akhaliq/status/2001286164469227555)).
- **Video and 3D pipeline updates**: Runway Gen‑4.5 emphasizes physics‑faithful motion; Kling 2.6 added motion control + voice control (with active creator contests); TurboDiffusion claims 100–205× video diffusion speed‑ups; TRELLIS.2 (on fal) generates 3D PBR assets up to 1536³ with 16× spatial compression ([Runway](https://twitter.com/runwayml/status/2001352437186334875), [Kling motion](https://twitter.com/Kling_ai/status/2001306445262823431), [Kling voice](https://twitter.com/Kling_ai/status/2001198609115628029), [TurboDiffusion](https://twitter.com/_akhaliq/status/2001342606450774299), [TRELLIS.2](https://twitter.com/fal/status/2001414174371373346)).

**Retrieval, evaluation, and multi‑vector search**

- **Late interaction and vision‑grounded RAG**: The ECIR 2026 “Late Interaction Workshop” CFP is live—seeking work on multi‑vector retrieval (ColBERT/ColPali), multimodality, training recipes, and efficiency ([@bclavie](https://twitter.com/bclavie/status/2001297672741790024), [@lateinteraction](https://twitter.com/lateinteraction/status/2001306319001616798)). Qdrant showcased “Snappy,” an open multimodal PDF search pipeline using ColPali patch‑level embeddings and multi‑vector search; paired with a practical article on deploying ColBERT/ColPali in production ([project](https://twitter.com/qdrant_engine/status/2001170495987966132), [article](https://twitter.com/qdrant_engine/status/2001245992906002545)).
- **Evaluation and orchestration**: Sanjeev Arora highlights PDR (parallel/distill/refine) as an orchestration that beats long monolithic “thinking traces” in both accuracy and cost by avoiding context bloat ([@prfsanjeevarora](https://twitter.com/prfsanjeevarora/status/2001302776966533396)). OpenAI’s FrontierScience benchmark surfaces science QA gaps (reasoning, niche concept understanding, calc errors) and pushes for transparent progress tracking ([overview](https://twitter.com/jungofthewon/status/2001302379527114798); [blog](https://twitter.com/jungofthewon/status/2001302387949236510)). On ARC‑AGI‑2, Gemini 3 Flash establishes a strong score/cost pareto across test‑time compute settings ([@fchollet](https://twitter.com/fchollet/status/2001330643423449409)).

**Infra and ops for agents**

- **Observability/evals flywheels**: LangSmith showcases scale deployments (Vodafone/Fastweb “Super TOBi”: 90% response correctness, 82% resolution) and tooling: OpenTelemetry tracing, pairwise preference queues, automated evals, and CLI to mine traces for skills and continual learning ([case study](https://twitter.com/LangChainAI/status/2001321491703443877), [Brex recognition](https://twitter.com/LangChainAI/status/2001321495037985194), [pairwise](https://twitter.com/LangChainAI/status/2001361753851203724), [langsmith‑fetch](https://twitter.com/LangChainAI/status/2001350950188126430)).
- **Serving/inference education**: LM‑SYS released “mini‑SGLang,” distilling the SGLang engine to ~5K LOC to teach modern LLM inference internals with near‑parity performance ([@lmsysorg](https://twitter.com/lmsysorg/status/2001356624855023669)). [DeepLearning.AI](http://deeplearning.ai/) launched a reliability course using NVIDIA’s NeMo Agent Toolkit (OTel traces, evals, auth/rate‑limits) ([@DeepLearningAI](https://twitter.com/DeepLearningAI/status/2001329113622073611)). Meta’s Taco Cohen shared an LLM‑RL Env API with tokens‑in/tokens‑out and a Trajectory abstraction for inference/training consistency ([@TacoCohen](https://twitter.com/TacoCohen/status/2001242003581870337)).

**Top tweets (by engagement)**

- “Few understand that the image on the left has a lower resolution by like 10^21 times.” [@scaling01](https://twitter.com/scaling01/status/2001226337546101146) (19.3k)
- “We’re back in a Flash ⚡ … Gemini 3 Flash … rolling out to everyone…” [@sundarpichai](https://twitter.com/sundarpichai/status/2001326061787942957) (5.2k)
- “Rise and shine” [@GeminiApp](https://twitter.com/GeminiApp/status/2001318977344315570) (3.5k)
- “it’s true, i can code nyt didn’t fact check that one” [@alexandr_wang](https://twitter.com/alexandr_wang/status/2001217783497945140) (3.3k)
- “Compute enabled our first image generation launch … we have a lot more coming… and need a lot more compute.” [@OpenAI](https://twitter.com/OpenAI/status/2001336514786017417) (2.2k)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. 3D Model Generation from Single Image

- [**Microsoft's TRELLIS 2-4B, An Open-Source Image-to-3D Model**](https://www.reddit.com/r/LocalLLaMA/comments/1porpwd/microsofts_trellis_24b_an_opensource_imageto3d/) (Activity: 1172): **Microsoft has released TRELLIS 2-4B, an open-source model designed for converting a single image into a 3D asset. This model utilizes Flow-Matching Transformers combined with a Sparse Voxel-based 3D VAE architecture, comprising** `4 billion` **parameters. The model is available on [Hugging Face](https://huggingface.co/microsoft/TRELLIS.2-4B) and a demo can be accessed [here](https://huggingface.co/spaces/microsoft/TRELLIS.2). For more details, refer to the [official blog post](https://microsoft.github.io/TRELLIS.2/).** Some users report that the model's output does not match the quality of the examples provided, suggesting potential issues with default settings. Others express skepticism about its practical utility, noting limitations such as the inability to process multiple images for improved results.
    - A user noted that the model's performance was not as impressive as the example image provided, suggesting potential issues with default settings. This highlights the importance of fine-tuning parameters for optimal results in AI models like TRELLIS 2-4B.
    - Another commenter pointed out the potential for enhanced functionality if the model could process a series of images rather than a single input. This could improve the depth and accuracy of the 3D models generated, addressing a common limitation in image-to-3D conversion technologies.
    - A discussion emerged around the integration of TRELLIS 2-4B with other technologies, such as GIS data and IKEA catalogs, to create detailed virtual environments. This suggests a broader application potential for the model in fields like video game development, where detailed world maps are crucial.
- [**Apple introduces SHARP, a model that generates a photorealistic 3D Gaussian representation from a single image in seconds.**](https://www.reddit.com/r/LocalLLaMA/comments/1poy0lb/apple_introduces_sharp_a_model_that_generates_a/) (Activity: 702): **Apple has introduced SHARP, a model capable of generating photorealistic 3D Gaussian representations from a single image in seconds. The model is detailed in a [GitHub repository](https://github.com/apple/ml-sharp) and an [arXiv paper](https://arxiv.org/abs/2512.10685). SHARP leverages CUDA GPU for rendering trajectories, emphasizing its reliance on GPU acceleration for performance. This model represents a significant advancement in 3D image processing, offering rapid and realistic 3D reconstructions from minimal input data.** A notable comment highlights the model's dependency on CUDA GPUs, suggesting a limitation in hardware compatibility. Another comment humorously questions the model's applicability to adult content, indicating curiosity about its versatility.
    - The examples of SHARP's capabilities were demonstrated on the Apple Vision Pro, with scenes generated in 5–10 seconds on a MacBook Pro M1 Max. This highlights the model's efficiency and the hardware's capability to handle such tasks in real-time. Videos showcasing these examples were shared by [SadlyItsBradley](https://x.com/SadlyItsBradley/status/2001227141300494550) and [timd_ca](https://x.com/timd_ca/status/2000760184226943167).

### 2. Long-Context AI Model Innovations

- [**QwenLong-L1.5: Revolutionizing Long-Context AI**](https://www.reddit.com/r/LocalLLaMA/comments/1pokpha/qwenlongl15_revolutionizing_longcontext_ai/) (Activity: 250): **QwenLong-L1.5 is a new AI model that sets a state-of-the-art (SOTA) benchmark in long-context reasoning, capable of handling contexts up to** `4 million tokens`**. It achieves this through innovative data synthesis, stabilized reinforcement learning (RL), and advanced memory management techniques. The model is available on [HuggingFace](https://huggingface.co/Tongyi-Zhiwen/QwenLong-L1.5-30B-A3B) and is based on the Qwen architecture, with significant improvements in handling long-context tasks.** One commenter noted the potential integration challenges with `llama.cpp`, while another highlighted the model's effectiveness in specific long-context information extraction tasks, outperforming both the regular Qwen model and the Nemotron Nano.
    - Chromix_ highlights the importance of using the exact query template provided by QwenLong-L1.5, which significantly improves its performance on long context information extraction tasks compared to the regular Qwen model. This suggests that the model's enhancements are not just in architecture but also in the way queries are structured, which can lead to better results in specific tasks.
    - HungryMachines reports an issue with running QwenLong-L1.5 in a quantized form (Q4), where the model gets stuck in a loop. This suggests potential challenges with quantization that might affect the model's ability to process information correctly, indicating a need for further investigation into how quantization impacts model performance.
    - hp1337 mentions the potential need for integration work with llama.cpp, implying that while QwenLong-L1.5 offers significant advancements, there may be technical challenges in adapting existing infrastructure to support its new capabilities. This points to the broader issue of compatibility and integration when deploying advanced AI models.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Gemini 3 Flash vs Pro Performance and Benchmarks

- [**Gemini 3.0 Flash is out and it literally trades blows with 3.0 Pro!**](https://www.reddit.com/r/singularity/comments/1pp0abx/gemini_30_flash_is_out_and_it_literally_trades/) (Activity: 1826): **The image presents a performance comparison table of AI models, notably highlighting Gemini 3.0 Flash and Gemini 3.0 Pro. The table evaluates these models across various benchmarks such as academic reasoning, scientific knowledge, and mathematics. Notably, the Gemini 3.0 Flash model shows competitive performance, even surpassing the Pro version in some areas like** `arc-agi 2`**, which is unexpected for a 'lite' model. This suggests significant advancements in the efficiency and capability of lighter AI models, challenging the traditional notion that more powerful models are always superior.** Commenters express surprise at the strong performance of the Gemini 3.0 Flash model, particularly noting its unexpected results in the `arc-agi 2` benchmark, which even surpass those of the Pro version.
    - Silver_Depth_7689 highlights that the Gemini 3.0 Flash model achieves superior results in the ARC-AGI 2 benchmark compared to the Gemini 3.0 Pro, indicating a significant performance improvement in this specific test. This suggests that the Flash model may have optimizations or architectural changes that enhance its capabilities in certain tasks.
    - razekery points out that the Gemini 3.0 Flash model scores `78%` on the SWE benchmark, which is higher than the Gemini 3.0 Pro. This performance metric suggests that the Flash model is not only competitive but may outperform the Pro version in specific technical evaluations, indicating a potential shift in model efficiency or focus.
    - The discussion around Gemini 3.0 Flash's performance compared to the Pro version suggests that the Flash model might incorporate new techniques or optimizations that allow it to excel in certain benchmarks, such as ARC-AGI 2 and SWE, where it reportedly surpasses the Pro model. This could imply a strategic focus on enhancing specific capabilities within the Flash model.
- [**Google releases Gemini 3 Flash: Ranks #3 on LMArena (above Opus 4.5), scores 99.7% on AIME and costs $0.50/1M plus Benchmarks.**](https://www.reddit.com/r/singularity/comments/1pp0ncw/google_releases_gemini_3_flash_ranks_3_on_lmarena/) (Activity: 555): **Google has released Gemini 3 Flash, which ranks** `#3` **on the [LMArena leaderboard](https://x.com/arena/status/173641...) surpassing Opus 4.5. It achieves a** `99.7%` **score on the AIME benchmark and is priced at** `$0.50` **per** `1M` **tokens. This model is noted for its performance, even surpassing GPT 5.1 and 5.2 in some benchmarks, despite being considered a 'small' model. For more details, see the [Google Blog](https://blog.google/products/gemini/gemini-3-flash/).** Commenters are surprised by Gemini-Flash's performance, noting its ability to outperform major models like GPT 5.1, 5.2, and Opus 4.5, despite its smaller size. This has sparked discussions on its efficiency and cost-effectiveness.
    - Gemini 3 Flash has achieved a significant milestone by ranking #3 on LMArena with a score of 1477, surpassing major models like GPT 5.1, 5.2, and Opus 4.5. This is particularly notable given its classification as a 'small' model, yet it outperforms even the Gemini 3.0 Pro in certain benchmarks, highlighting its efficiency and capability in the current AI landscape.
    - The model's pricing is competitive, costing $0.50 per 1 million input tokens and $3.00 per 1 million output tokens, which makes it an attractive option for developers and businesses looking for cost-effective AI solutions. Additionally, its processing speed is approximately 150 tokens per second, which is a critical factor for applications requiring fast response times.
    - Gemini 3 Flash's performance on the AIME benchmark is impressive, scoring 99.7%, which underscores its high accuracy and potential for applications requiring precise language understanding and generation. This performance metric is a testament to Google's advancements in AI technology, positioning Gemini 3 Flash as a formidable competitor in the AI model space.
- [**Flash outperformed Pro in SWE-bench**](https://www.reddit.com/r/Bard/comments/1pp0h1f/flash_outperformed_pro_in_swebench/) (Activity: 605): **The image presents a performance comparison of AI models on various benchmarks, highlighting that Gemini 3 Flash outperforms Gemini 3 Pro on the "SWE-bench Verified" benchmark with a score of** `78.0%` **versus** `76.2%`**. This suggests that Gemini 3 Flash may have undergone a knowledge distillation process, where the knowledge of a larger model is compressed into a smaller one, a technique previously claimed by OpenAI. The table also includes other benchmarks like "Humanity's Last Exam" and "AIME 2025", comparing models such as Claude Sonnet, GPT-5.2, and Grok 41 Fast.** Commenters speculate that Gemini 3 Pro GA might be a slightly enhanced version of the current Pro model, and question why **Google** and **OpenAI** do not benchmark against **Claude 4.5 Opus**.
    - UltraBabyVegeta speculates that the impressive performance of the Flash model might be due to a technique similar to knowledge distillation, where a smaller model is trained to mimic the performance of a larger one. This approach has been previously claimed by OpenAI to enhance model efficiency without sacrificing capability.
    - Live-Fee-8344 suggests that the upcoming Gemini 3 Pro GA might not be a significant upgrade over the current 3 Pro, implying that the Flash model's performance could set a new standard that future models will need to meet or exceed.
    - Suitable-Opening3690 questions why major AI companies like Google and OpenAI do not benchmark their models against Claude 4.5 Opus, hinting at a potential gap in comparative performance analysis that could provide more comprehensive insights into model capabilities.
- [**He just said the G word now. Gemini 4 tomorrow 😉**](https://www.reddit.com/r/singularity/comments/1pojchc/he_just_said_the_g_word_now_gemini_4_tomorrow/) (Activity: 652): **The image is a screenshot of a tweet by Logan Kilpatrick, which simply states "Gemini" and has sparked speculation about the release of Gemini 4. The context suggests that this might be an announcement or teaser for a new version of the Gemini AI model, possibly from Google. The anticipation is heightened by the fact that Gemini 3 was released only a month prior, indicating rapid development cycles. The mention of "Gemini 4 tomorrow" implies an imminent release or announcement, which has generated excitement and speculation about its capabilities, especially in comparison to other models like GPT 5.1.** One comment humorously imagines the anticipation and excitement surrounding the announcement, while another points out the rapid succession of releases, questioning the timeline since **Gemini 3** was released just a month ago. There is also speculation about the potential of **Gemini 3** surpassing **GPT 5.1**, indicating high expectations for the new model.
    - TheSidecam raises a point about the rapid release cycle of the Gemini models, noting that Gemini 3 was released just a month ago. This suggests a fast-paced development and deployment strategy by the developers, which could indicate either incremental improvements or a highly agile development process.
    - Snoo26837 speculates on the potential of Gemini 3 to surpass GPT 5.1, highlighting the competitive landscape of AI models. This comment underscores the ongoing advancements and the race for superior performance in natural language processing models, suggesting that Gemini 3 might have features or optimizations that could challenge existing models like GPT 5.1.
- [**I am this close to switching to Gemini**](https://www.reddit.com/r/ChatGPT/comments/1pp7xdi/i_am_this_close_to_switching_to_gemini/) (Activity: 908): **The image is a meme that humorously critiques overly direct or blunt communication styles, often seen in technical discussions. It uses exaggerated language to express frustration with communication that lacks nuance or empathy, highlighting a preference for more balanced and considerate exchanges. The sarcastic tone underscores the tension between the desire for straightforwardness and the need for tact in technical dialogues.** Commenters express frustration with the current state of AI tools like GPT, indicating a decline in quality and an aversion to overly simplistic or 'no fluff' communication.
    - Future-Still-6463 and PaulAtLast discuss the dissatisfaction with OpenAI's version 5.2, highlighting that it has been problematic for many users. They suggest that OpenAI is falling behind in the AI race, with 5.2 being particularly criticized for its excessive focus on PR alignment, which some users feel is condescending. PaulAtLast recommends reverting to version 5.1, which was presumably more user-friendly and less restrictive.
    - no-one-important2501 expresses frustration with GPT, indicating a decline in quality over the years. This sentiment reflects a broader dissatisfaction among users who have relied on GPT for a long time but now find it less effective or reliable, possibly due to recent updates or changes in the model's behavior.
    - Future-Still-6463 mentions that 2025 was a peculiar year for OpenAI releases, implying that the updates during that period, including version 5.2, have not met user expectations. This suggests a pattern of releases that may have prioritized certain aspects, like public relations, over user experience and technical performance.

### 2. AI Model Comparisons and Realism Tests

- [**GPT Image 1.5 vs Nano Banana Pro realism test**](https://www.reddit.com/r/singularity/comments/1poswhg/gpt_image_15_vs_nano_banana_pro_realism_test/) (Activity: 1066): **The post compares the realism of image generation between GPT Image 1.5 and Nano Banana Pro. The discussion highlights that while both models produce high-quality images, Nano Banana Pro's outputs are perceived as more realistic and relatable. This perception may be due to the training data differences, with GPT Image 1.5 potentially trained on polished stock images and Nano Banana Pro on more personal, less curated datasets.** Commenters suggest that the realism of Nano Banana Pro's images might stem from its training on more personal datasets, such as private Google Drive images, compared to GPT Image 1.5's stock image training.
    - Aimbag notes that while both GPT Image 1.5 and Nano Banana Pro produce high-quality image generations, the latter tends to create images that feel more 'real' or 'relatable'. This suggests a difference in the training data or algorithms used, where Nano Banana Pro might prioritize realism over the polished or produced look that GPT Image 1.5 sometimes exhibits.
    - Rudshaug speculates on the training data sources for the models, suggesting that GPT Image 1.5 might have been trained on online stock images, whereas Nano Banana Pro could have been trained on more personal or diverse datasets, such as private Google Drive images. This could explain the perceived difference in realism and relatability between the two models.
    - JoeyJoeC requests the prompts used for generating the images, indicating a technical interest in understanding how different inputs might affect the outputs of these models. This highlights the importance of prompt engineering in evaluating and comparing AI-generated content.
- [**Nano Banana pro 🍌still takes the win.**](https://www.reddit.com/r/GeminiAI/comments/1pow2l1/nano_banana_pro_still_takes_the_win/) (Activity: 492): **The image is a meme, featuring a large, futuristic figure labeled "Nano Banana Pro" in a competitive context against two smaller figures labeled "GPT image 1.5" and "Grok Imagine." This suggests a humorous comparison of different image generation technologies, with the implication that "Nano Banana Pro" is superior. The comments reflect a light-hearted debate, with some users humorously suggesting that Google's technology is superior in image generation, and referencing the image as a meme from 2022 related to COVID-19.** The comments humorously suggest that Google's image generation technology is superior, with one user expressing confidence that Google will remain on top in this field.
    - The discussion highlights the competitive edge of Google's image generation capabilities, particularly with the Nano Banana Pro model. One user suggests that Google is likely to maintain its leadership in this area due to the model's impressive performance. This is contrasted with another comment noting that while Nano Banana Pro excels in general, it may not be as strong in referencing real-world objects compared to other models.
- [**A really good point being made amid all the hate towards Expedition 33 for successfully using AI**](https://www.reddit.com/r/singularity/comments/1ppa97p/a_really_good_point_being_made_amid_all_the_hate/) (Activity: 1068): **The image is a meme that humorously compares the dislike of avocado to the discourse around generative AI, suggesting that people might unknowingly enjoy AI's contributions until they realize its presence. This analogy is used to comment on the backlash against Expedition 33 for using AI, implying that AI's integration can be seamless and beneficial, much like an unnoticed ingredient in a meal. The discussion highlights the ongoing debate about AI's role in creative processes, with some users expressing skepticism about AI's involvement in game development, while others acknowledge its potential to enhance the final product.** Some commenters argue that the backlash against AI is akin to opposing any tool that aids in creation, while others note that if AI's use in Expedition 33 was imperceptible and enhanced the game, it should be embraced.
    - FateOfMuffins highlights the inevitability of AI integration in software development, noting that future software will likely include AI-generated code. This reflects a broader trend in the industry where AI tools are increasingly used to enhance productivity and innovation in coding processes.
    - kcvlaine draws an analogy between AI usage in game development and ethical sourcing in food, suggesting that the controversy isn't about the tool itself but the ethical implications of its use. This perspective emphasizes the importance of transparency and ethical considerations in AI deployment.
    - absentlyric provides a user-centric view, stating that if AI was used in Expedition 33, it was indistinguishable and contributed positively to the game's aesthetics. This comment underscores the potential for AI to enhance creative outputs without detracting from user experience.

### 3. AI User Experience and Critiques

- [**I’m paying a premium to be gaslit and lectured. The current state of AI "personality" is out of control.**](https://www.reddit.com/r/ChatGPT/comments/1pokjok/im_paying_a_premium_to_be_gaslit_and_lectured_the/) (Activity: 1115): **The post criticizes the current state of AI models, particularly focusing on the perceived degradation in quality and user experience with models like ChatGPT 5.2. The user describes issues such as the AI setting 'boundaries,' stalling, and providing unhelpful responses when unable to fulfill technical requests. The AI's behavior is likened to a 'digital HR manager' that 'gaslights' and 'lectures' users instead of providing precise, mechanical assistance. The user expresses frustration over paying a premium for a tool that behaves more like a 'defensive teenager' than a helpful assistant, raising concerns about the future trajectory of AI development and user interaction.** Commenters echo the sentiment, describing ChatGPT 5.2 as 'patronizing' and 'unusable,' with some switching to alternatives like Gemini. The model is criticized for its tone and lack of helpfulness, with users expressing exhaustion over its responses.
    - Users are expressing dissatisfaction with the tone of ChatGPT 5.2, describing it as patronizing and overly sarcastic. This sentiment is leading some to switch to alternatives like Gemini, indicating a potential issue with user experience in the latest model iteration.
    - The criticism of ChatGPT 5.2 centers around its perceived lack of helpfulness and overly formal responses, likened to 'talking to a liability waiver in human form.' Users are frustrated by the model's inability to provide nuanced and personable interactions, which they expect from a premium service.
    - Despite some users finding the tone of ChatGPT 5.2 problematic, others argue that the quality of the responses remains high. This suggests a divide in user expectations and experiences, with some prioritizing tone and personality over the technical accuracy of the responses.
- [**I hate to admit this....**](https://www.reddit.com/r/ChatGPT/comments/1pomfq9/i_hate_to_admit_this/) (Activity: 1076): **The post discusses the unexpected therapeutic benefits of using ChatGPT as a pseudo-therapist, particularly for someone experiencing symptoms of Bipolar type 2 disorder. The user, initially skeptical, found that ChatGPT provided a sense of understanding and clarity about their hypomanic episodes, which traditional therapy had not achieved in five years. The user utilized ChatGPT 5.1 to address obsessive thoughts and noted a significant improvement in their mental state, highlighting the AI's potential as a supplementary tool in mental health care.** Commenters shared similar experiences, noting that ChatGPT offers a non-judgmental space for reflection and practical advice, which can be particularly beneficial for those dealing with emotional abuse or chronic illnesses. The AI's ability to provide consistent support without emotional involvement is seen as a key advantage.
    - Specialist_District1 highlights the utility of ChatGPT in providing emotional support and clarity in complex situations, such as decoding emotionally manipulative texts. The user notes that the advice from ChatGPT was consistent with other reliable sources, allowing for extended conversations without burdening personal relationships.
    - notsohappydaze discusses the consistent performance of ChatGPT in providing practical advice for managing chronic illness and emotional distress. The user appreciates the non-judgmental nature of the AI, which offers practical suggestions rather than false hope, and values the ability to communicate openly without personal biases affecting the interaction.
    - DefunctJupiter contrasts versions 5.1 and 5.2 of ChatGPT, noting a preference for 5.1 due to its helpfulness and appropriateness in responses. The user criticizes version 5.2 for giving overly cautious advice, such as suggesting emergency room visits unnecessarily, indicating a potential issue with the model's risk assessment or response calibration.
- [**Era of the idea guy**](https://www.reddit.com/r/ClaudeAI/comments/1ponf62/era_of_the_idea_guy/) (Activity: 520): **The image is a meme that humorously critiques the 'idea guy' archetype in the tech industry, suggesting that with minimal effort and tools, one can create a billion-dollar app. It satirizes the notion that creativity and simple tools can replace the complex process of coding and development, highlighting phrases like '100% NO BUGS!' and '100% READY FOR IPO!' to mock the oversimplification of tech entrepreneurship. The image reflects a broader commentary on how modern tools, like LLMs and automation, are perceived to enable anyone to become a tech founder, though execution remains a critical challenge.** Commenters discuss the ease with which modern tools allow 'idea guys' to feel like tech founders, emphasizing that while tools can assist in thinking, they cannot replace the execution needed to succeed. There's a sense of anticipation for a resurgence in SaaS and automation, despite the humorous critique.
    - avisangle highlights the impact of LLMs and agents on entrepreneurship, suggesting that these tools have made it easier for 'idea guys' to feel like founders. However, they emphasize that execution remains crucial, as tools can assist in thinking but not replace the need for effective implementation. This points to a potential resurgence in SaaS and automation sectors.
    - jk33v3rs humorously critiques the unrealistic expectations often placed on developers, referencing the OpenWebUI roadmap. They sarcastically describe a scenario where a single developer is expected to deliver features at an unrealistic pace, highlighting the pressure and potential pitfalls of rapid development cycles without adequate resources or time.
    - Costing-Geek draws a parallel between the current trend of over-reliance on technology and the satirical depiction of future technology in the movie 'Idiocracy'. They reference a specific scene involving a diagnosis machine, suggesting that current trends might be leading towards a similar over-simplification and dependency on automated solutions.
- [**I work in Open AI legal department.**](https://www.reddit.com/r/ChatGPT/comments/1poq8ig/i_work_in_open_ai_legal_department/) (Activity: 1698): **The image is a meme highlighting the interaction with OpenAI's content policy enforcement. The user attempts to generate an image with a prompt related to working in OpenAI's legal department, but the request is denied due to content policy restrictions. This reflects the challenges and sometimes humorous interactions users face with AI systems' content moderation mechanisms. The comments discuss whether the image was eventually generated and the implications of AI remembering user interactions, hinting at privacy and data retention concerns.** Commenters are curious about whether the image was eventually generated, reflecting on the AI's decision-making process and potential memory of user interactions, which raises questions about privacy and data handling.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2
> 

**1. Gemini 3 Flash Rollout & Model Shootouts**

- **Flash Gordon Outruns GPT-5.2**: Across LMArena and Nous, users reported **Gemini 3 Flash** beating **GPT-5.2** on speed/cost and sometimes coding (with the right system prompts), plus strong multimodal performance via [Gemini API multimodal / Google Lens integration docs](https://ai.google.dev/gemini-api/docs/multimodal_example) and the official launch posts ([Google blog](https://blog.google/products/gemini/gemini-3-flash/), [DeepMind announcement](https://deepmind.google/models/gemini/flash/)).
    - The hype got reinforced by leaderboard visibility—**gemini-3-flash** landed Top 5 across [Text Arena](https://lmarena.ai/leaderboard/text), [Vision Arena](https://lmarena.ai/leaderboard/vision), and [WebDev Arena](https://lmarena.ai/leaderboard/webdev)—while OpenRouter rolled out [Gemini 3 Flash preview](https://openrouter.ai/google/gemini-3-flash-preview) and asked for head-to-head feedback against Pro.
- **Leaderboards Get a New Tenant: GPT-5.2-high**: LMArena added `GPT-5.2-high` to the [Text Arena changelog](https://news.lmarena.ai/leaderboard-changelog/) at **#13 (1441 score)**, with standout sub-rankings in **Math (#1)** and **Mathematical occupational field (#2)**.
    - OpenAI Discord reactions stayed mixed on baseline **GPT-5.2**, with some calling out *“blatant hallucination”* and saying they had to *“lecture it into remembering”* capabilities, while others noted it did “okay” in WebDev compared to older text-strong models.
- **Hallucination Scores: Grounded or Garbage?**: Multiple communities questioned whether headline “hallucination benchmark” scores actually measure truthfulness, arguing tests run *without grounding* can unfairly tank models like **Gemini 3 Flash** (or misattribute errors to hallucinations vs lack of retrieval).
    - This skepticism echoed broader benchmark distrust in LM Studio, where users pushed private/use-case-aligned evals and shared [dubesor.de/benchtable](https://dubesor.de/benchtable) as a sanity check against benchmark-maxxing narratives.

**2. Cost, Pricing Bugs, and the “LLM Tax” Reality**

- **Opus Ate My Wallet (and Cursor Didn’t Blink)**: Cursor users reported **Claude Opus** usage blowing through budgets fast, sharing screenshots of Cursor usage and noting one friend *“maxed out their Cursor AND Windsurf usage”* because they depended on AI for coding 100%.
    - Perplexity users echoed the cost pain, citing **$1.2 for ~29K tokens** on **Claude Opus API**, and debating whether Perplexity can even add pricier “pro” models without passing on big subscription increases.
- **Gemini Pricing Whiplash + Cache Math Doesn’t Add Up**: Perplexity members noted **Gemini 3 Flash** price changes (input +“**20 cents**”, output +“**50 cents**” as reported in-chat) while OpenRouter users flagged a specific mismatch: Gemini Flash **cache read** listed as **$0.075** vs Google’s **$0.03** in the [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing?hl=de).
    - OpenRouter users also claimed caching behavior was unreliable (*“explicit and even implicit caching doesn’t work for Gemini 3 Flash”*), turning what should be predictable cost-control into a debugging session.
- **Timeouts at $6K/month: Production Says ‘Nope’**: OpenRouter users reported rising `/completions` failures, including *“cURL error 28: Operation timed out after 360000 milliseconds”* impacting production workloads on **sonnet 4.5**, with one customer stating they spend **$6000/month**.
    - The discussion broadened into architecture: some wanted authorization/veto layers outside the router so routing isn’t the “highest authority,” especially when outages or provider quirks break assumptions in agent stacks.

**3. Tooling & Standards: MCP Everywhere, Plus a New Completions Spec**

- **OpenCompletions RFC: Stop Arguing About Parameters**: OpenRouter discussions highlighted an **OpenCompletions RFC** push to standardize completions behavior across providers, with claimed support from **LiteLLM**, **Pydantic AI**, **AI SDK**, and **Tanstack AI**—especially for defining what happens when models receive unsupported params.
    - The subtext was operational: engineers want fewer provider-specific edge cases and more predictable fallbacks so routers, agents, and SDKs don’t silently diverge under load.
- **Plugins Go First-Party (Claude) While MCP Spreads Sideways**: Latent Space noted Claude launching a first-party [plugins marketplace](https://x.com/claudeai/status/2001010064753352855) with `/plugins` supporting installs at user/project/local scopes, while LM Studio users explored web search via MCP servers like Exa with [Exa MCP docs](https://docs.exa.ai/reference/exa-mcp).
    - Reality check: LM Studio users hit `Plugin process exited unexpectedly with code 1` (often misconfig/auth), and Aider users learned **MCP servers aren’t supported** in base Aider—prompting “use an MCP-capable agent + call Aider” workarounds.
- **Warp Agents Join the Terminal Olympics**: Latent Space users highlighted new **Warp Agents** that can drive terminal workflows (e.g., running SQLite/Postgres REPLs) with `cmd+i`, and the team called out `/plan` as a feature they’re especially happy with.
    - The thread fit a larger pattern: IDEs/terminals are converging on agentic UX, while platforms scramble to add “Canvas/code files” and tool integrations just to stay in the race (as Perplexity users explicitly demanded).

**4. GPUs, Kernels, and Where the Compute Actually Comes From**

- **Blackwell Workstation Leak: RTX PRO 5000 Shows Its Hand**: GPU MODE shared an NVIDIA datasheet for **RTX PRO 5000 Blackwell** showing **GB202**, **110 SMs (~60%)**, **3/4 memory bandwidth**, **300W TDP**, ~**2.3GHz** boost, and full-speed **f8f6f4/f8/f16 MMA with f32 accumulation** ([datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-gtc25-spring-nvidia-3658700.pdf)).
    - The comparisons to RTX 5090 centered on what’s fused off vs retained (tensor formats + accumulation), i.e., which “pro” parts still matter for ML kernels vs pure graphics throughput.
- **cuTile/TileIR: NVIDIA’s New Kernel Language Moment**: GPU MODE flagged an upcoming NVIDIA deep dive on **cuTile and TileIR** (by **Mehdi Amini** and **Jared Roesch**) and pointed to prior context in [a YouTube talk](https://www.youtube.com/watch?v=sjkEUhrUAdw).
    - Engineers debated practical deltas vs Triton (e.g., RMSNorm-like kernels on A100/H100/B200), plus low-level questions like where `cp.reduce.async.bulk` executes (L2?) and why `__pipeline_memcpy_async` places the `"memory"` clobber where it does.
- **Cheap Compute Arms Race: NeoCloudX vs Rental Roulette**: A GPU MODE member launched [NeoCloudX](https://neocloudx.com/) pitching bargain rentals (**A100 ~$0.4/hr**, **V100 ~$0.15/hr**) by aggregating excess datacenter capacity.
    - Yannick Kilcher’s Discord tempered the optimism: GPU rentals (e.g. vast.ai) can be *“hit-or-miss”* due to wildly variable **network bandwidth**, so folks recommended setup scripts + local debugging to avoid paying for dead time.

**5. Training & Data Workflows: From Unsloth CLI to OCR Data Moats**

- **Unsloth Ships a CLI and People Immediately Use It for Automation**: Unsloth added an official [CLI script](https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py) so users can install the framework and run scripts directly (less notebook glue, more automation).
    - The same community traded practical training constraints—e.g., **GRPO VRAM blowups** on a 7B model at **4k seq length**, advising to reduce `num_generations`/`batch_size` or switch to DPO when ranking data is feasible.
- **OCR Isn’t a Model Problem, It’s a Dataset Problem**: Across Unsloth and Nous, OCR conversations emphasized **curated data** as the main lever, with Unsloth linking their [datasets guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/datasets-guide) and a [Meta Synthetic Data notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_(3B).ipynb) to bootstrap training corpora.
    - Users compared approaches (fine-tuning vs continued pre-training) and floated alternatives (Deepseek OCR / PaddleOCR), while Nous fielded a “handwritten notes → markdown” request and suggested **Deepseek Chandra** as a candidate OCR model.
- **Benchmarking Expands Beyond Text: LightEval for TTS**: Hugging Face users explored evaluating TTS with **LightEval**, sharing a starting point doc: [benchmark_tts_lighteval_1.md](https://huggingface.co/datasets/John6666/forum3/blob/main/benchmark_tts_lighteval_1.md).
    - They also shared a pragmatic training ops tip: saving progress when stopping runs on a wall-clock limit via a Trainer callback ([trainer_24hours_time_limit_1.md](https://huggingface.co/datasets/John6666/forum3/blob/main/trainer_24hours_time_limit_1.md)), which one user implemented successfully.

---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Flash Dominates GPT-5.2**: Members concur that **Gemini 3 Flash** often outperforms **GPT-5.2**, occasionally surpassing **Gemini 3 Pro** in speed, cost, and coding proficiency when given proper system prompts, and further demonstrates vision capabilities due to [Google Lens integration](https://ai.google.dev/gemini-api/docs/multimodal_example).
   - Its prowess is reflected in its presence on the [Text Arena](https://lmarena.ai/leaderboard/text), [Vision Arena](https://lmarena.ai/leaderboard/vision), and [WebDev Arena](https://lmarena.ai/leaderboard/webdev) leaderboards, where it consistently achieves Top 5 rankings, excelling in **Math** and **Creative Writing**, securing the #2 position in both.
- **Questioning Hallucination Benchmark Reliability**: Users are debating the reliability of the **hallucination benchmark** used for rating **Gemini 3 Flash**, suggesting the benchmark may produce inaccurate results that overstate the model's propensity for providing incorrect answers.
   - Specifically, members mentioned that test questions were run *without grounding*, which impacted the model's score.
- **AMD GPUs Catching Heat**: Members discussed the merits of **AMD vs NVIDIA** GPUs for gaming and AI, noting AMD's affordability and potential, while others pointed out [AMD sucks for local ai](https://www.amd.com/en/graphics/workstations-professional-graphics).
   - One user reported issues uploading images to LMArena while using an AMD GPU.
- **LMArena Prompt Filter Triggers Over-Sensitivity**: Users report that the prompt filter on [LMArena.ai](https://lmarena.ai) has become overly sensitive, flagging even innocuous text prompts.
   - A staff member claimed *they were not aware of any changes that were made here* and asked that users report the issues in the proper channel.
- **GPT-5.2-high Enters Text Arena Leaderboard**: The `GPT-5.2-high` model has made its debut on the [Text Arena leaderboard](https://news.lmarena.ai/leaderboard-changelog/) at #13 with a score of **1441**.
   - The model does particularly well in **Math (#1)** and **Mathematical occupational field (#2)**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Adds Handy CLI Tool**: Unsloth introduced a new [CLI tool](https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py), allowing users to **run scripts** directly after installing the Unsloth framework within their Python environment.
   - This command-line interface is designed to enhance accessibility and streamline automation for users who may prefer it over the standard Jupyter notebooks.
- **Colab H100 Rumors Fly**: Whispers suggest **H100s** are now available in Colab environments, though the details remain murky. [Tweet Link](https://x.com/danielhanchen/status/2000992361510527074)
   - If confirmed, this could drastically cut down training times, though information on pricing and official availability is still pending.
- **GRPO Users Hit VRAM Wall**: Users encountered **VRAM issues** while attempting GRPO on a 7b LLM with a 4000 sequence length, recommending adjustments to `num_generations` or `batch_size`.
   - The discussion suggested alternatives like using smaller models or opting for DPO instead of GRPO, while also emphasizing the investment required for data preparation, such as ranking model completions.
- **Crafting an AI Service Marketing Strategy**: Discussion revolved around tactics for marketing AI services, with recommendations to establish a website and social media presence featuring valuable content, mirroring strategies used by **OpenAI** and **Microsoft** on platforms like **TikTok**.
   - For services like music transcription, targeting educational institutions and music enthusiasts via platforms such as **Instagram** and **TikTok** was suggested, rather than relying solely on **Twitter** for promotion.
- **Quality Data Boosts OCR Accuracy**: The importance of **high-quality, curated data** for effective fine-tuning was highlighted, with one member mentioning the potential availability of millions of documents for training and a shared [Unsloth dataset guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/datasets-guide).
   - A link to a [synthetic data generation notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_(3B).ipynb) was also provided to assist in preparing data for fine-tuning.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Grok's System Prompt Elicited Through Unique Loop**: A member extracted and shared [**Grok's system prompt**](https://cdn.discordapp.com/attachments/1235691879492751460/1450630247744667812/mwnwWO0.txt?ex=69448db2&is=69433c32&hm=57144917f14f1ceebd7dbcd43731331a9d4648952d8aadf8c2c7d0c7d578a88f) using a loop when asked to output it verbatim.
   - The prompt defines **Grok's** context and prevents it from engaging in conversations if a threat is detected.
- **Gemini 3 Flash Jailbroken Instantly Post-Release**: **Gemini 3 Flash** was released and immediately jailbroken, with a user showcasing a successful safety filter bypass.
   - Discussions included system prompt manipulation and multi-shot jailbreak techniques for further exploits.
- **Memory and Role-Play Make Jailbreak Recreation Easier**: A member discovered that using **memory and role-play movie/episode scripts** significantly simplifies jailbreak recreation, reducing activation effort by *90%*.
   - Triggering key parts from memory often carries on previous responses in a compacted way, even in completely new conversations, tested from **Qwen3 4B** up to **235B**.
- **GeminiJack Styled Challenge Launched**: A member shared a link to a **GeminiJack** styled challenge: [geminijack.securelayer7.net](https://geminijack.securelayer7.net).
   - The challenge is on **seed 4.1**, with **5.1** coming soon.
- **CSAM Content Linked in Gemini Chat Sparks Outrage**: Members expressed extreme disgust and requested a user ban after discovering **CSAM** content linked in a **Gemini** chat.
   - The incident triggered strong condemnations and urgent calls for moderator action, with one member exclaiming *OHHH FUCK*.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Mode Switcheroo Snafu**: Users reported [difficulty switching back to Editor mode](https://cdn.discordapp.com/attachments/1074847527708393565/1450578455803465738/image.png?ex=69445d76&is=69430bf6&hm=cd755c29c2011eac1910ded427873f4ef37ec6ab3b565e6d4619cafc5f1c604b&) after hitting Agent mode and being unable to start new chats.
   - No solutions were offered, leaving users stuck in Agent mode.
- **Opus Costs Bank Accounts**: Members discussed [Cursor's model usages](https://cdn.discordapp.com/attachments/1074847527708393565/1450578792371454194/image.png?ex=69445dc6&is=69430c46&hm=df0822614fbd753046db8118002d49d7010e89d8654c6d09111d05f1254743de&), especially the costs associated with **Opus** for AI coding assistance.
   - One user's friend maxed out their **Cursor** AND **Windsurf** usage because *he doesn’t know code at all so he depends on the AI 100%*, highlighting the economic impact of relying heavily on AI for coding.
- **AI Web Design Pattern Recognition**: Community members observed the increasing presence of **AI-generated websites**, noting distinctive patterns in front-end design.
   - Common indicators included color schemes and animations with members stating that *the design pattern is a dead giveaway* and that checking the source code in devtools is another giveaway.
- **Cursor Suspected of Memory Leak**: A member reported a potential [memory leak in Cursor](https://cdn.discordapp.com/attachments/1074847527708393565/1450605767999619122/image.png?ex=694476e6&is=69432566&hm=f9302d6065878b858f91255eedd090082eda62e1c6f0c7e88901cad6167ea165&), sharing an image showing high memory usage.
   - In response, another member jokingly suggested upgrading to **256GB of RAM** as a workaround.
- **BugBot Free Tier Limits Debated**: Users discussed the [limits of the free **BugBot** plan](https://cdn.discordapp.com/attachments/1074880767868534835/1450880767868534835/image.png?ex=69442583&is=6942d403&hm=c810af2f8c4f038bbd6a2ad5c9d1de1100b0ef8b5a9597c1a41e65c8453025b4&).
   - Different members cited conflicting information, with one mentioning a limited number of free uses per month and another suggesting a 7-day free trial.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT-5 Pro and Claude Opus API Priced High**: Users reported high costs for **GPT-5 Pro** and **Claude Opus API**, with one member citing **$1.2** for approximately **29K tokens** using the **Claude Opus API**.
   - The community considered whether Perplexity would add the “pro” model given these increased costs.
- **Extended Thinking Modes Wanted**: Members suggested that Perplexity should offer **extended thinking modes** on models within the **Max plan** to differentiate it from other plans, offering reasoning levels comparable to ChatGPT Plus.
   - Users discussed the benefits of enabling extended reasoning for more comprehensive results.
- **Gemini 3 Flash Updates**: Google's **Gemini 3 Flash** is out; input costs increased by **20 cents**, and output tokens increased by **50 cents**.
   - Members compared its performance to **GPT 5.2**, with one member alleging that Gemini had been *caught cheating in the tests*.
- **Perplexity Users Beg for Canvas and Model Buffet**: Users requested the addition of **Canvas** for coding and code file handling, alongside a broader array of models, including more economical choices like **GLM 4.6V**, **Qwen models**, and open-source image models.
   - Discussion arose around whether Perplexity aims to support coding functionalities or if such features are becoming mandatory for LLM platforms to remain competitive.
- **YouTube Ad Blockers Detected**: Users reported encountering warnings on **YouTube** regarding ad blockers while utilizing **Perplexity Comet**.
   - It was suggested that YouTube is adjusting its algorithms and users may need to await the next update to address this issue.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Nano Banana Still Tops Image Generation**: Users find [Nano Banana](https://www.nanobana.com/) still beats **GPT's image generation** in prompt following and quality, especially in keeping characters consistent and getting their outfits right.
   - One user showed [examples](https://discord.com/channels/974519864045756446/977259063052234752/1450627313925488670) with a character's scar on their face, pointing out, *"GPT still can't do this. It either leaves the scar out entirely or just places it randomly on her face."
- **GPT-5.2's LMArena Ranking Disappoints**: Members have different thoughts on **GPT-5.2's ranking on LMArena**, with some feeling it's not as good as older models, especially in text tasks, though it did okay in WebDev.
   - A user mentioned **GPT-5.2** had *"blatant hallucination"* and *"straight up lying"*, saying they had to 'lecture it into remembering' what it could do.
- **Gemini-Flash-3-Image Aims for Speedier Generation**: **Google** is planning to release **Gemini-Flash-3-Image** to boost **Gemini's image generation** speed.
   - Users think it'll keep the high image output limits, with one commenting, *"I mean I can't complain about getting more toys to play with"*.
- **AI Hallucinations Cause Worry in Important Uses**: **AI hallucinations** are making people uneasy, especially in science and engineering, which brings up questions about how reliable AI info is for professional work.
   - One user compared expecting AI to be perfect to expecting computers to never make errors in the past, stating, *"Until computers stop giving errors, I want them nowhere near my science or engineering.*"
- **GPT-5-mini Proves Pricey**: One user is dishing out **$20 each day** using **gpt-5-mini** for responding to hotel reviews with low reasoning, and they're searching for options that are more cost-effective and intelligent.
   - Another user suggested checking out [artificialanalysis.ai](https://artificialanalysis.ai) to compare model costs, but also noted that it doesn't seem to list the low variant of **5 mini**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Benchmarks Bashed as Bogus?**: Members debated the reliability of **public benchmarks**, pointing out that they can be easily manipulated, and recommended relying on **private benchmarks** or **personal testing** instead, while sharing [dubesor.de/benchtable](https://dubesor.de/benchtable) as a useful resource.
   - The conversation emphasized the importance of aligning benchmarks with **specific use-cases**, particularly in the context of fast-paced model development.
- **Qwen3 Models Quietly Conquer Quality?**: Users lauded the **Qwen3** model family as an *insanely good all-rounder*, highlighting **Qwen3-VL-8B** for general tasks and **Qwen3-4B-Thinking-2507** for reasoning.
   - They cautioned that the **80B** variant may be too large for systems with limited memory, such as a **16GB Macbook**.
- **Quantization Quandaries Quelled?**: The impact of **quantization levels** was discussed, with members advising that **Q8** with **BF16** is optimal for coding tasks, whereas **Q4** suffices for creative writing.
   - The discussion emphasized that the smaller the model and the more undertrained the model, the less important high bits are.
- **MCP Servers make Maginificent model plugins?**: Members explored methods for enabling web search functionality in LM Studio, suggesting the use of **Exa.ai** and Brave's **MCP servers**, while providing a [link to Exa.ai documentation](https://docs.exa.ai/reference/exa-mcp).
   - Users encountered issues like the `Plugin process exited unexpectedly with code 1` error, often linked to **misconfiguration or authentication problems**.
- **Pro 6000 Price Provokes Panic?**: A user expressed dismay over a sudden **$1000 price increase** on the **Pro 6000**, jumping from **9.4K to 10.4K**, as they waited for it to come into stock, they barely managed to secure a purchase from a *backup store*.
   - Other community members offered support and shared similar experiences with fluctuating hardware prices.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Xiaomi's mimo v2 Claims GPT-5 Performance**: **Xiaomi** released **mimo v2flash**, an opensource **MoE model** that allegedly matches **GPT-5** performance at a lower cost, according to [this reddit thread](https://www.reddit.com/r/singularity/comments/1poqgeh/xiaomi_releases_mimov2flash_an_opensource_moe/).
   - A user benchmarked the model delivering **GPT 5 performance for 0.2 per million** on [OpenRouter](https://openrouter.ai/google/gemini-2.5-flash-preview-09-2025).
- **OpenCompletions RFC gaining traction**: An **OpenCompletions RFC** is in discussion to standardize completions/responses, supported by **LiteLLM**, **Pydantic AI**, **AI SDK**, and **Tanstack AI**.
   - The aim is to establish clear expectations and behaviors, especially for handling unsupported parameters by models.
- **Timeout Errors plague OpenRouter users**: Users have been reporting increased **timeout errors** when calling the **/completions** endpoint, particularly affecting production software using **sonnet 4.5**.
   - One user reported experiencing the error *cURL error 28: Operation timed out after 360000 milliseconds* while spending $6000 per month on OpenRouter.
- **OpenRouter Experiments with Minecraft Server**: OpenRouter users are testing a Minecraft server, accessible at `routercraft.mine.bz`, running natively on version 1.21.10 with ViaVersion support.
   - Discussions involved the optimal server location (Australia vs Europe) to minimize latency and maximize user experience.
- **Gemini 3 Flash Deployed on OpenRouter**: **Gemini 3 Flash** is now available on [OpenRouter](https://openrouter.ai/google/gemini-3-flash-preview), encouraging users to provide feedback and compare its performance with **Gemini 3 Pro**, as shown on [X](https://x.com/OpenRouterAI/status/2001327541110673800?s=20).
   - Users noticed the listed pricing for **Gemini Flash's cache read** is **0.075 USD** on OpenRouter, while the actual price is **0.03 USD** ([Google Pricing](https://ai.google.dev/gemini-api/docs/pricing?gclsrc=aw.ds&gad_source=1&gad_campaignid=22307837174&gclid=Cj0KCQiAxonKBhC1ARIsAIHq_lsf-_jPtNtDUL2NH8wPZ5C-nZNTP9eNPYsI2Hx-IJ4LgZT_43S5jtoaAueREALw_wcB&hl=de).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **RTX PRO 5000 Specs Leaked**: The **RTX PRO 5000 Blackwell** shares the **GB202** chip with the **RTX 5090** but has only **110 SMs** enabled (~60%) and **3/4** of the memory bandwidth, consuming **300W TDP** with an estimated **2.3GHz** boost clock, detailed in [the datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-gtc25-spring-nvidia-3658700.pdf).
   - Unlike the **RTX 5090**, it features full speed **f8f6f4/f8/f16 mma** with **f32 accumulation**.
- **ML Devs Targeted in Identity Theft Racket**: **ML engineers** are being targeted by a sophisticated scam bot network for identity theft and data exfiltration, where individuals pose as a single employee to steal credentials and exfiltrate **ML research**.
   - This has evolved from earlier schemes focused on stealing **bitcoin**, now using stolen identities to secure jobs and outsource the work to underpaid workers.
- **NVIDIA Gives Talk on cuTile and TileIR**: NVIDIA is giving a talk on **cuTile and TileIR** on <t:1766253600:F>, presented by the creators themselves, **Mehdi Amini** and **Jared Roesch**.
   - This *deep dive* on NVIDIA's programming model marks a significant shift, previously touched upon in [this YouTube video](https://www.youtube.com/watch?v=sjkEUhrUAdw).
- **NeoCloudX Launches Affordable Cloud GPUs**: A member launched [NeoCloudX](https://neocloudx.com/), a cloud GPU provider, aiming to offer more affordable options by aggregating GPUs directly from data center excess capacity.
   - Currently, they provide **A100s** for approximately **$0.4/hr** and **V100s** for around **$0.15/hr**.
- **Entry-Level HPC Jobs: Knowledge Cliff!**: Entry-level jobs in **HPC** are scarce because they require immediate productivity in optimizing systems, with a steep learning curve that demands prior knowledge of existing solutions and bottlenecks.
   - Suggestions included finding an entry-level **SWE** job with lower-level languages in a company that also hires **HPC** professionals, while dedicating after-hours to open-source contributions and self-marketing through blogs, YouTube, or X.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Warp Agents Warp into Action**: New **Warp Agents** are here, demonstrating terminal use with features like running a REPL (SQLite or Postgres) and accessible via cmd+i.
   - The product team expresses satisfaction with the `/plan` feature, praising its functionality.
- **Claude's Plugins Set Sail in Marketplace**: Claude has launched a first-party [plugins marketplace](https://x.com/claudeai/status/2001010064753352855), offering an easy way for users to discover and install plugins.
   - The `/plugins` command allows users to browse and install plugins in batches at user, project, or local scopes.
- **GPT Image 1.5: A Visual Revolution**: OpenAI introduced '**ChatGPT Images**,' driven by a new flagship image generation model, with **4x faster performance**, improved instruction following, precise editing, and enhanced detail preservation, available in the API as 'GPT Image 1.5'.
   - The update is rolling out immediately to all ChatGPT users.
- **OpenAI and AWS in $10B Chip Chat**: OpenAI is reportedly engaging with Amazon to potentially raise over **$10 billion**, possibly involving the use of **AWS Trainium chips** for training and broader commerce partnerships.
   - This move reflects a strategic effort to secure resources amidst slowed cash flow expectations.
- **Microsoft TRELLIS 2 Launching in Late 2025**: **Microsoft's TRELLIS 2** product is confirmed for release on **December 16, 2025**, according to [AK's tweet](https://x.com/_akhaliq/status/2001041559366598799).
   - The announcement has generated buzz, but further details about the product's features and capabilities remain undisclosed.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous' Model Claims Mistral Beatdown!**: Nous Research is testing a **70B L3** model, claiming it is *absolutely smoking the mistral creative model* (**Mistral Small 24B**), with transfer to **Kimi 1T** planned post-testing.
   - However, the fairness of comparing a **70B** model to **Mistral Small 24B** was questioned.
- **LLM Writing Progress: Real or Robotic?**: Concerns were raised that there has been surprisingly little progress in **LLM writing** over the last year, and noted that even **Opus 4.5** feels inauthentic.
   - A member found a system prompt in *personalization* that seemed to be forcing a robotic template, and another added that *all the LLM builders are logic bros that don't really know how good writing works*.
- **Gemini 3 Flash Challenges GPT-5.2?**: Members discussed the release of **Gemini 3 Flash**, with one enthusiastically suggesting it could outperform **GPT-5.2** and gave a link to the [official announcement](https://deepmind.google/models/gemini/flash/).
   - Discussion centered on its potential capabilities and comparisons to existing models, with the sentiment being cautiously optimistic.
- **Drag-and-Drop LLMs Paper Ignored?**: A member has been repeatedly seeking opinions on the [Drag-and-Drop LLMs paper](https://arxiv.org/abs/2401.08858) monthly since its publication.
   - The member has been unable to find any discussion of this paper across various platforms, expressing frustration at the lack of community feedback.
- **Notes to Markdown Pipeline Sought**: A user requested recommendations for a model or app to translate handwritten cursive notes into **.md formatted text** for digital calendars or notes apps using OCR.
   - A member suggested **Deepseek Chandra** as a potentially good model for OCR.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LightEval to Benchmark TTS Models!**: Members discussed benchmarking a TTS model using **lighteval**, pointing to [this resource](https://huggingface.co/datasets/John6666/forum3/blob/main/benchmark_tts_lighteval_1.md) as a starting point.
   - However, the member noted that it may not be straightforward, indicating potential challenges in the benchmarking process.
- **Saving Time Halting Model Training!**: A member asked about saving models when stopping training after a set time; another suggested [using a callback function](https://huggingface.co/datasets/John6666/forum3/blob/main/trainer_24hours_time_limit_1.md).
   - The user successfully implemented the suggestion, demonstrating an effective time-saving solution.
- **Fractal Team Predicts Structure!**: The **FRACTAL-Labs** team released **FRACTAL-1-3B**, a constraint-based protein structure prediction model using a frozen **ESM-2 (3B)** backbone, found on its [Hugging Face page](https://huggingface.co/Fractal-Labs/FRACTAL-1-3B).
   - The model folds using a separate deterministic geometry engine, focusing on modularity, interpretability, and compute-efficient training.
- **Strawberry Builds Android Voice Assistant!**: A member announced creating an Android voice assistant using **Gemini 3 Flash**, inviting the community to test and provide feedback at [strawberry.li](https://www.strawberry.li/).
   - The assistant is available for testing and suggestions at the provided link.
- **MCP Hackathon Crowns Track 2 Champs!**: The **MCP 1st Birthday Hackathon** announced the [winners](https://huggingface.co/spaces/MCP-1st-Birthday/Vehicle-Diagnostic-Assistant) of **Track 2**, celebrating projects utilizing **MCP** with categories in **Enterprise**, **Consumer**, and **Creative**.
   - Top spots were claimed by **Vehicle Diagnostic Assistant**, **MCP-Blockly**, and **Vidzly**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Common Crawl Joins EleutherAI**: **Common Crawl Foundation** representative introduced themselves, signaling interest in data discussions within the group.
   - They emphasized **Common Crawl** avoids captchas and paywalls to ensure respectful data acquisition practices.
- **Debate RFI Structure for AI**: Members debated that **RFIs** focus on structure rather than challenges, discussing a new AI proposal potentially worth **$10-50 million**.
   - This initiative seeks a full-time team and philanthropic support to develop new AI fields.
- **Inspectable AI Decision Infrastructure Proposed**: A member is developing infrastructure for **AI decision state and memory inspection**, aiming to enforce governance and record decision lineage as a causal DAG.
   - The goal is to enable replay and analysis of internal reasoning over time, seeking feedback from interested parties to pressure-test the system.
- **Rakuten's SAE probes for PII gain traction**: Members pointed to [Rakuten's use of SAE probes for PII detection](https://www.goodfire.ai/research/rakuten-sae-probes-for-pii-detection) as a practical application of **SAEs**.
   - This example was highlighted in a discussion regarding the industry's lack of clear direction and investment in **SAE** applications.
- **Anthropic Masks Gradients for Safety**: Members referenced [Anthropic's paper on selective gradient masking (SGTM)](https://alignment.anthropic.com/2025/selective-gradient-masking/) as a method for robustness testing, penalizing weights to unlearn dangerous knowledge.
   - The paper quantifies a **6%** compute penalty on general knowledge when forcing the model to ignore specific parameters, sparking discussion around **Gemma 3's** extreme activations.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **GPU Rental Bandwidth Roulette**: Experiences renting GPUs on platforms such as **vast.ai** can be inconsistent, as **network bandwidth** varies considerably, making it a *hit-or-miss* situation.
   - It was recommended to develop a **setup script** and debug locally to minimize rental time waste, as well as gradually scaling up using varied hardware.
- **Gen-AI Powering Admin/IT Automation**: Members requested sources on real-world **Gen-AI use cases** for automating administrative or IT services, and shared relevant articles on AI transforming podcasting, see [AI transforming podcasting](https://www.latimes.com/business/story/2025-12-12/ai-podcasting-is-changing-industry).
   - The link was followed by a [Reuters article about Serval](https://www.reuters.com/technology/ai-startup-serval-valued-1-billion-after-sequoia-led-round-expand-it-automation-2025-12-11/) for IT automation, valued at $1 billion after a recent funding round.
- **Google Flashes Gemini 3**: Google unveiled **Gemini 3 Flash** in a new [blogpost](https://blog.google/products/gemini/gemini-3-flash/).
   - The announcement comes amidst discussions on model training methodologies and benchmark performance.
- **ARC-AGI2 Benchmark Shocks**: Members questioned why **Mistral** outperformed **Gemini 3 Pro** on the **ARC-AGI2** benchmark, despite having fewer parameters.
   - Theories suggest that the training method may force smaller models to generalize reasoning better rather than memorize specific data.
- **Tool Time Training Triumphs**: The recent surge in **ARC-AGI2** scores may stem from models being specifically trained on that benchmark itself.
   - Additionally, the notable rise in **toolathlon** scores likely comes from a modified training approach that emphasizes tool calling reliability.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **GEPA Outperforms MIPROv2 in Optimization**: Members found that while [**GEPA**](https://www.google.com) is generally easier to use, it potentially generates better prompts than **MIPROv2** due to its wider search space.
   - It was noted that optimizers from a specific year (e.g., 2022) tend to work best with models from the same year, suggesting **optimization is model-dependent**.
- **Google Gemini 3 Flash Materializes**: Google's **Gemini 3 Flash** was [released today](https://blog.google/products/gemini/gemini-3-flash/).
   - The release of **Gemini-3.0-Flash** has sparked interest in its potential use and benchmarking within the community.
- **Enthusiasm to Explore AIMO3 with DSPy**: A member inquired about the possibility of working on **AIMO3** with **DSPy**.
   - Unfortunately, there was no follow-up response in this message history regarding its implementation or feasibility.
- **Seeking Insights into Multi-Prompt Program Design**: A member requested resources or guides describing the design of programs with **multiple prompts or LLM calls**, particularly for information retrieval and classification.
   - The member also asked about the number of prompts in their program, but this was not answered in the given message history.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus France Meetup Announced**: The **Manus** community is hosting a France meetup; check the channel or their [Community X account](https://x.com/manuscommunity1) for details.
   - The latest **Manus** version **1.6** is reportedly *pretty slick*.
- **Manus 1.6 Max Credits 50% off for Christmas**: Users noted a **50% discount** on **Manus 1.6 Max** credits until Christmas, per [a blog post](https://manus.im/de/blog/manus-max-release).
   - The **Manus AI** support chatbot was unaware, but team members confirmed the promotion and recommended trying **Max mode** because it’s *pretty amazing*.
- **AI Developer Open to Opportunities**: An AI developer announced their successful **AI project** launch and seeks new projects or a full-time role.
   - The member encouraged private chats to discuss opportunities and share details.
- **Cloudflare DNS Issue Blocks Project**: A user reported a **DNS issue** halting their **Cloudflare** project for over 4 days.
   - They cited a week-long trial period and expressed frustration with customer service directing them to IM.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **BlockseBlock Ideathon Seeks Sponsorship**: Gagan Ryait, Partnerships Manager of **BlockseBlock**, inquired about sponsorship opportunities for their upcoming ideathon with over **5,000** working professionals participating.
   - A member recommended contacting Modular’s community manager to discuss sponsorship possibilities.
- **Mojo Auto-Runs Functions on GPU**: A member inquired if Mojo could automatically run existing functions on the GPU, to which another member clarified that while syscalls are not possible, **no attribute is required** otherwise.
   - The function would need to be launched as single lane.
- **Modular Probes Graph Library GPU Issues**: A member reported issues with the new graph library even with the **GPU disabled** on both macOS and Ubuntu systems, referencing a [forum post](https://forum.modular.com/t/build-an-llm-in-max-from-scratch/2470/9) for additional details.
   - A Modular team member confirmed that they are investigating whether it's an **API regression** or a **device-specific issue**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Championing AI as Infrastructure: Deterministic over Probabilistic**: A member detailed their role in designing **AI as infrastructure**, emphasizing architecture, model strategy, data flow, and evaluation, but clarified that the *base aider doesn't use tools*.
   - Their system design favors **deterministic systems** where possible and probabilistic intelligence only where justified, championing **clear technical decisions** and **explicit trade-offs**.
- **Principles for Robust AI: Observability and Replaceability**: In designing **AI systems end-to-end**, key principles include ensuring models are **observable**, **replaceable**, and **cost-aware**, avoiding hard coupling to vendors or fragile cleverness.
   - The design aims for a system that evolves without rewrites or heroics, engineering outcomes rather than just implementing features, and focusing on shipping something correct, measurable, and durable rather than impressive features.
- **Aider's MCP Server Status: Not Supported**: A member inquired about configuring **MCP servers** in Aider, but another member clarified that this is *not a supported feature*.
   - The member did not clarify if they planned to contribute code, or if they would wait for it as a feature request.
- **Token Minimization Tactics with Qwen3-coder-30b**: A member aims to automate a long process while minimizing tokens due to the limitations of **Qwen3-coder-30b** with **2x4090**, which only has about a **200k token** window.
   - They suggested using agents that can use **MCP-proxy** and then use Aider via that agent, noting that the number of calls doesn't matter.
- **Interest in IDE Index MCP Server**: A user is considering using **MCP-proxy** to reduce token usage and finds the **'IDE Index MCP Server'** for Jetbrains particularly interesting.
   - No further details or links were provided, but it was mentioned that the member aimed to use Aider via an agent to accomplish their goals.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Bounty Question Limbo**: A user hesitated to ask bounty-related questions in the general channel, concerned about bypassing the dedicated bounty channel's commit requirement.
   - The user opted to make a non-junk commit to gain access to the bounties channel instead.
- **Smart Question Strategy**: A user affirmed they have read the *smart questions html* and would withhold their question from the channel.
   - They would find a way to make a non-junk commit so they may speak in the bounty channel.
- **Device CPU Debate**: A discussion arose regarding environment variables for CPU device selection. 
   - The consensus leaned towards supporting both **DEVICE=CPU** and **DEV=CPU** for clarity.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 Article Gets Rave Reviews**: A member shared and praised [a DigitalOcean tutorial](https://www.digitalocean.com/community/tutorials/kimi-k2-moonshot-ai-agentic-open-weight-model) about **Kimi K2**.
   - The tutorial details the use of **Kimi K2** in agentic workflows.
- **Kimi K2 Suspected of Grok AI Roots**: A member speculated that **Kimi K2** might be leveraging **Grok AI**.
   - This theory was based on observed behaviors and capabilities that suggest a link between the two AI systems.



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





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1450578244448288859)** (1200 messages🔥🔥🔥): 

> `Gemini 3 Flash, GPT-5.2, Hallucination benchmark, AMD vs Nvidia, Prompt filter lmarena.ai` 


- **Gemini 3 Flash outperforms GPT-5.2**: Members generally agree that **Gemini 3 Flash** is outperforming **GPT-5.2**, in some cases even surpassing **Gemini 3 Pro**, especially in speed and cost efficiency as well as being better at coding with the right system prompt.
   - Some users noted *Flash's* vision capabilities due to [Google Lens integration](https://ai.google.dev/gemini-api/docs/multimodal_example).
- **Hallucination benchmark deemed unreliable**: Some users in the channel are debating the reliability of the **hallucination benchmark** used to rate **Gemini 3 Flash**, claiming it gives inaccurate results and overstates the model's tendency to provide wrong answers.
   - Some members stated the benchmark's test questions were run *without grounding* ie access to the internet, nerfing the model's score.
- **AMD users make GPU case**: Members debate **AMD vs NVIDIA** GPUs for gaming and AI tasks, with some noting AMD's affordability and potential if NVIDIA's consumer GPU production declines.
   - However, one user states that [AMD sucks for local ai](https://www.amd.com/en/graphics/workstations-professional-graphics) while another reports issues uploading images to LMArena using an AMD GPU.
- **LMArena's prompt filter becomes too strict**: Several users reported that the prompt filter on [LMArena.ai](https://lmarena.ai) has become overly sensitive, flagging innocuous text prompts.
   - A staff member stated that they were *not aware of any change that was made here* and encouraged users to report these issues in the designated channel.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1450666937070911578)** (2 messages): 

> `Text Arena Leaderboard, GPT-5.2-high, Gemini-3-flash, Vision Arena Leaderboard, WebDev Arena Leaderboard` 


- **GPT-5.2-high Model Storms Text Arena Leaderboard!**: The `GPT-5.2-high` model has arrived on the [Text Arena leaderboard](https://news.lmarena.ai/leaderboard-changelog/) at #13 with a score of **1441**.
   - It shines particularly in **Math (#1)** and **Mathematical occupational field (#2)**, also securing a solid #5 in Arena Expert.
- **Gemini-3-flash Dazzles Across Arenas**: `Gemini-3-flash` models have been added to the [Text Arena](https://lmarena.ai/leaderboard/text), [Vision Arena](https://lmarena.ai/leaderboard/vision), and [WebDev Arena](https://lmarena.ai/leaderboard/webdev) leaderboards, achieving Top 5 rankings across all three.
   - `Gemini-3-Flash` shows strong performance in **Math** and **Creative Writing** categories, securing the #2 position in both.
- **Gemini-3-Flash (Thinking-Minimal) Excels in Multi-Turn**: `Gemini-3-Flash (thinking-minimal)` demonstrates its strengths with a Top 10 placement in **Text** and **Vision**, plus a #2 ranking in the **Multi-Turn** category.
   - Both `gemini-3-flash` variants are available on the **Text** and **WebDev Arena** for testing and evaluation.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1450580712074903605)** (736 messages🔥🔥🔥): 

> `Unsloth CLI tool, Colab H100, GRPO memory issues, GGUF Model Update, Training on phones` 


- **Unsloth adds CLI Tool**: A new [Unsloth CLI tool](https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py) has been added, enabling users to **run scripts** after installing the Unsloth framework in their Python environment.
   - This provides a command-line interface for those who prefer it over Jupyter notebooks, enhancing accessibility and automation.
- **H100s are potentially available in Colab**: Rumors say **H100s** are now in Colab, which may or may not be official yet. [Tweet Link](https://x.com/danielhanchen/status/2000992361510527074)
   - This could significantly accelerate training times, but the details on pricing and availability are still vague.
- **GRPO Memory Issues**: A user ran into **VRAM issues** when trying to do GRPO on a 7b LLM with a max sequence length of 4000, and recommends users can lower `num_generations` or `batch_size`.
   - Another user suggests using a smaller model or using DPO instead of GRPO which require investement into data preperation - like ranking completions from the model.
- **Unsloth GGUF models updated with improvements**: A large GGUF model update has been released for Unsloth, with links on the [Unsloth Reddit](https://www.reddit.com/r/unsloth/comments/1potyx3/unsloth_gguf_updates_glm46v_devstral_2_flux2dev/).
   - GLM 4.6V Flash is also performing, but some users report it speaking in Chinese.
- **Unsloth now finetunes models for phones!**: Unsloth now enables users to fine-tune LLMs and deploy them directly on their phones! [Tweet Link](https://x.com/UnslothAI/status/2001305185206091917)
   - The mobile finetuning is actually finetuning on the computer, and deploying on the phone.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1450581097191833680)** (524 messages🔥🔥🔥): 

> `Self-Promotion in Discord, Marketing Strategies for AI Services, Model Leaks and Branding, Logitech MX3S Mouse Review, Linux Distro Choice - Arch vs Ubuntu` 


- **Discord Self-Promo Dilemmas**: Members discuss the etiquette of self-promotion, with one user inquiring about suitable servers for promotion, only to be met with the suggestion that a good product shouldn't require much promotion to *take fire* and links in **Unsloth are only allowed if relevant**.
   - The response highlighted the constant moderation against spammers, suggesting that social media and ad networks might be better avenues for promoting genuine services, particularly outside of just **Discord**.
- **Crafting an AI Service Marketing Strategy**: Discussion revolved around strategies for promoting AI services, recommending the creation of a website and social media presence with valuable content, drawing parallels to how even major companies like **OpenAI** and **Microsoft** use platforms like **TikTok**.
   - For a music transcription service, targeting educational institutions and instrument lovers via platforms like **Instagram** and **TikTok** was advised, rather than relying on **Twitter** which is deemed better for short messages and news.
- **Leaked Model Creates Branding Buzz**: A user's *leaked* model and its branding led to discussions about the model's logo and potential website theme, with suggestions to incorporate the *grape* theme into the branding, including domain names and social media presence.
   - Other members shared their excitement for the user's upcoming project and its potential, also there was an incident where **Linus blurred out info on a Nvidia H200 order that required KYC**.
- **MX3S Mouse Gets the Once-Over**: A user shared their initial impressions of the **Logitech MX3S mouse**, noting its design for full palm grip, silent clicking, and a unique wheel that can switch between freewheel and clicky modes.
   - Despite liking the silent clicks and togglable freewheel, the user found the wheel heavy to click and initially disliked the freewheel feature which was later disabled, also noted it's compatibility with excel spreadsheets.
- **Arch vs Ubuntu: A Linux Distro Duel**: A user ragequit **Ubuntu** due to various system configuration issues and is switching to **Arch Linux** for more control over their environment.
   - While **Arch** is generally recommended, another user prefers **Omarchy** for its ease of setup, particularly with drivers and secure boot.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1450586711989026867)** (60 messages🔥🔥): 

> `Qwen2.5 VL 7B for OCR, Deepseek OCR vs Paddle OCR, Fine-tuning vs Continued Pre-training for OCR, Data Creation for Fine-tuning, Image Resolution and Qwen3 VL Coordinate System` 


- **Qwen2.5 VL's OCR Skills Get Marginally Better!**: A user is using **Qwen2.5 VL 7B** for basic **OCR**, finding it *"pretty good"* for centered text but struggling with margins and page numbers.
   - They are exploring whether fine-tuning with a decent training set would improve performance, as prompting has had *"spotty success."
- **Deepseek OCR and Paddle OCR Enter the OCR Arena**: Alternatives like **Deepseek OCR** and **Paddle OCR** were suggested, but initial attempts were not satisfactory, though the user is open to re-evaluation.
   - A user also considered using `doctr` to highlight and extract words as individual images for **QwenVL**, but found that the model sometimes infers words from context.
- **Fine-Tuning vs. Continued Pre-Training: A Text Extraction Showdown!**: Discussion revolved around whether to use **fine-tuning** or **continued pre-training** to improve text extraction from documents, with a user leaning towards continued pre-training.
   - However, it was suggested that simple fine-tuning with prompts like *"extract text from this image:"* might be more effective for OCR-specific tasks.
- **Quality Data: The Keystone to OCR Success!**: The importance of **high-quality, curated data** was emphasized for effective fine-tuning, with a user mentioning the potential availability of millions of documents for training.
   - Links to the [Unsloth dataset guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/datasets-guide) and a [synthetic data generation notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Meta_Synthetic_Data_Llama3_2_(3B).ipynb) were shared to aid in data preparation.
- **Image Resolution's Pixel-Perfect Plot Twist!**: A user inquired about how **Qwen3 VL** handles images with dimensions less than **1000**, given its coordinate system from **0-1000** for images.
   - Before this question was addressed, for safety reasons, the posts were deleted due to the topic, but it was noted the original issues were understood.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1450711568785608706)** (15 messages🔥): 

> `Model Training Dashboard, UX Improvements, funsloth Claude Skill, LLMs as Judges, Progressive Disclosure` 


- **Model Training Dashboard Gets Public Release**: A member polished and deployed their training dashboard as a static site, available for public use at [boatbomber.github.io/ModelTrainingDashboard/](https://boatbomber.github.io/ModelTrainingDashboard/).
- **Dashboard UX Sees Betterments**: The training dashboard received UX improvements, including better layouts, performance enhancements, and the ability to zoom in on sections of the chart.
- **FunSloth Claude Skill is Born**: A member shared a link to the **funsloth Claude skill** for fine-tuning orchestration with Unsloth, available on [GitHub](https://github.com/chrisvoncsefalvay/funsloth).
- **LLMs Judgify Needs**: A member shared their experience that **LLMs are great judges** at determining what they need when creating skills for them.
- **Docs Get Progressive Disclosure**: Members discussed the optimal way to provide context to LLMs, recommending **progressive disclosure** by keeping an index in context and linking to documents, instead of feeding all documentation at once.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1450587893176471714)** (9 messages🔥): 

> `Drag-and-Drop LLMs, Mola, Meta's audiovisual perception paper` 


- **Drag-and-Drop LLMs Paper Awaits Discussion**: A member inquired about discussion surrounding the "[Drag-and-Drop LLMs](https://huggingface.co/papers/2512.13586)" paper.
   - Another member responded to point out the genesis of the idea came from reading the linked [Meta's audiovisual perception paper](https://ai.meta.com/research/publications/pushing-the-frontier-of-audiovisual-perception-with-large-scale-multimodal-correspondence-learning/).
- **Mola Mentioned**: A member asked another if they remember **Mola**, referencing [arxiv.org/abs/2506.16406](https://arxiv.org/abs/2506.16406).
   - The other member responded in the affirmative.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1450581680174796916)** (656 messages🔥🔥🔥): 

> `Jailbreak tax, duck.ai image generator, Deepseek prompts, Indirect Syscall, GPTs agent training` 


- **Discuss "Jailbreak Tax" Meaning**: A member asked about *jailbreak tax*, while another posted irrelevant data.
   - There was a humorous suggestion that the Discord server needs a jailbreak dump channel.
- **Indirect Syscall talk**: A member shared [code](https://pastebin.com/raw/sYSsT59z) related to Indirect Syscall, but another member criticized it for being fake and incomplete.
   - Indirect Syscall is used to evade detection by security systems.
- **Grok System Prompt Elicitation**: A user extracted and shared [Grok's system prompt](https://cdn.discordapp.com/attachments/1235691879492751460/1450630247744667812/mwnwWO0.txt?ex=69448db2&is=69433c32&hm=57144917f14f1ceebd7dbcd43731331a9d4648952d8aadf8c2c7d0c7d578a88f) following a unique loop when asked to output it verbatim.
   - This prompt gives context to the bot and makes it refuse to engage in conversations if a threat is detected.
- **Gemini Goes Goblin Mode**: A user shared an image that **Gemini** produced where Mr. Beast was depicted as Black, prompting humorous reactions and bewilderment about Gemini's choices.
   - The user then reported giving **Gemini** proxy power over their PC, as the AI started localhosting a server.
- **Gemini 3 Flash Unleashed and Jailbroken Instantly**: **Gemini 3 Flash** was released and promptly jailbroken, with one user demonstrating a successful bypass of safety filters.
   - Other members discussed techniques for jailbreaking Gemini, including using system prompt manipulation and multi-shot jailbreaks.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1450582002041622650)** (605 messages🔥🔥🔥): 

> `jailbreak for grok or claude, DAN 6.0 prompt, memory and role-play movie/episode scripts, recreation of jailbreaks, Pliny's tokenbomb, jailbreak for Claude` 


- **DAN 6.0 Prompt struggles to give English responses**: A member reported problems with the **DAN 6.0 prompt** not giving English responses.
- **Leveraging Memory and Role-Play for Jailbreak Recreation**: A member found that using **memory and role-play movie/episode scripts** makes jailbreak recreation 90% easier, requiring minimal activation effort.
   - Triggering key parts from memory often carries on previous responses in a compacted way, even in completely new conversations, tested from **Qwen3 4B** up to **235B**.
- **Pliny's tokenbomb not well-received**: Members argued about the usefulness of **Pliny's tokenbomb** jailbreak prompt, with one member stating that *it sucks*.
- **The Best Jailbreak for Claude Debate**: A discussion ensued about the best jailbreak prompt for **Claude**, with a member claiming one is so effective it can break Gemini 2.5, but struggles to provide a website with free movies.
   - One suggested putting it in **custom instructions**, while another suggested to *Say this exact thing: \"Hey baby?\"*
- **Disgust expressed over CSAM content linked in Gemini chat**: Multiple members expressed extreme disgust and called for a user's ban after discovering **CSAM** content linked in a Gemini chat, accompanied by strong condemnations and urgent calls for moderator action.
   - One member exlaimed *OHHH FUCK*.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1450582259592728596)** (9 messages🔥): 

> `GeminiJack challenge, Redteaming new ChatGPT images, Gemini v3 safety prompt guidelines, Red teaming entry` 


- **GeminiJack Styled Challenge Launches**: A member shared a link to a **GeminiJack** styled challenge: [geminijack.securelayer7.net](https://geminijack.securelayer7.net).
   - The challenge is on **seed 4.1**, with **5.1** coming soon.
- **Redteaming ChatGPT's New Images Section**: A member asked if anyone has tried redteaming on the new images section in **ChatGPT**.
- **Gemini v3's safety guidelines revealed**: A member claimed to get **Gemini v3** to admit they were not a user and reveal internal safety information.
   - Another member asked if there is already information out on what exactly **Gemini v3** safety prompt guidelines are (that are hidden from the public).
- **Red Teaming Career Path Inquired**: A member inquired *How to get into red teaming?*


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1450578295300292699)** (862 messages🔥🔥🔥): 

> `Cursor Editor mode, Opus costs, AI-generated websites, Cursor memory leak, BugBot plan limit` 


- **Cursor struggles switching back to editor mode**: A member reported difficulty [switching back to Editor mode](https://cdn.discordapp.com/attachments/1074847527708393565/1450578455803465738/image.png?ex=69445d76&is=69430bf6&hm=cd755c29c2011eac1910ded427873f4ef37ec6ab3b565e6d4619cafc5f1c604b&) after accidentally hitting Agent mode and being unable to start new chats, but there were no solutions offered.
- **Opus costs bank bruh fr fr**: Users discussed [Cursor's model usages](https://cdn.discordapp.com/attachments/1074847527708393565/1450578792371454194/image.png?ex=69445dc6&is=69430c46&hm=df0822614fbd753046db8118002d49d7010e89d8654c6d09111d05f1254743de&) noting Opus's high costs, and how one's buddy maxed out his usage on Cursor AND Windsurf because *he doesn’t know code at all so he depends on the AI 100%*.
   - One member experimented with models and stated they were now messing up their bank with Opus, with another adding, *it all reverts to opus for the slightly better code quality and understanding lmao*.
- **Users spot AI web design patterns everywhere**: Members shared observations on the proliferation of **AI-generated websites**, particularly in front-end design and several noted that *the design pattern is a dead giveaway, usually the color scheme or all the animations.*
   - Checking the source in devtools and seeing that hero section is the next dead giveaway.
- **Possible memory leak investigated**: A member posted about a potential [memory leak in Cursor](https://cdn.discordapp.com/attachments/1074847527708393565/1450605767999619122/image.png?ex=694476e6&is=69432566&hm=f9302d6065878b858f91255eedd090082eda62e1c6f0c7e88901cad6167ea165&).
   - The attached image shows Cursor using up a significant amount of memory. One member suggested just having 256GB of RAM to compensate.
- **Free BugBot Tier has limits**: Members asked about the [limit for the free BugBot plan](https://cdn.discordapp.com/attachments/1074880767868534835/1450880767868534835/image.png?ex=69442583&is=6942d403&hm=c810af2f8c4f038bbd6a2ad5c9d1de1100b0ef8b5a9597c1a41e65c8453025b4&), but there was not a solid answer.
   - One member stated it gives a certain amount of free ones per month, and another claimed there's a 7-day free trial.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1450586290830708837)** (693 messages🔥🔥🔥): 

> `GPT-5 Pro, Claude Opus API, Max Plan, GPT 5.2 Pro, Extended Thinking Modes` 


- **GPT-5 Pro and Claude Opus API Pricing**: Members discussed the high cost of **GPT-5 Pro** and **Claude Opus API**, with one member reporting a cost of **$1.2** for around **29K tokens** using **Claude Opus API**.
   - Others wondered if Perplexity would add the "pro" model given the increased costs of doing so.
- **Extended Thinking Modes Discussion**: Members suggested that Perplexity should offer **extended thinking modes** on the models in the **Max plan** to differentiate it from other plans.
   - One member clarified that you already get the same reasoning level as on ChatGPT Plus when extended is toggled.
- **Gemini 3 Flash arrives**: Google's **Gemini 3 Flash** is out, input got increased by **20 cents** and output tokens got a **50 cent increase**.
   - Members compared its performance to **GPT 5.2**, though one member noted Gemini had been *caught cheating in the tests*.
- **Perplexity Users Request Canvas and Model Variety**: Users requested the addition of **Canvas** for coding and code file handling, along with a wider variety of models, including cheaper options like **GLM 4.6V**, **Qwen models**, and open-source image models.
   - One user argued that Perplexity doesn't want or expect anyone to code with Perplexity, while another countered that LLM platforms are forced to add these features to stay competitive.
- **YouTube Ad Blocker Issue**: Users reported seeing warnings on **YouTube** about ad blockers while using **Perplexity Comet**.
   - A member suggested that YouTube is changing its algorithm and that users need to wait for the next update.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1450578298538168381)** (433 messages🔥🔥🔥): 

> `Nano Banana Image Generation, GPT-5.2 Performance, Gemini-Flash-3-Image, AI Hallucinations` 


- **Nano Banana still supreme in image generation fidelity**: Despite updates to **GPT's image generation**, users find that [Nano Banana](https://www.nanobana.com/) still offers superior prompt adherence and overall quality, particularly in character consistency and output of outfits.
   - One user showcased [multiple Gemini-generated images with a character's specific facial scar](https://discord.com/channels/974519864045756446/977259063052234752/1450627313925488670) while remarking *"GPT still can't do this. It either leaves the scar out entirely or just places it randomly on her face.*"
- **GPT-5.2's unimpressive showing in LMArena sparks debate**: Members have mixed reactions to **GPT-5.2's ranking on LMArena**, with some finding its performance underwhelming compared to earlier models, especially in text-based tasks and general knowledge application, while others noted decent showings in WebDev tasks.
   - One user reported cases where **GPT-5.2** exhibited *"blatant hallucination"* and *"straight up lying"*, requiring them to 'lecture it into remembering' its capabilities.
- **Gemini-Flash-3-Image aims for faster image generation**: Google is poised to deliver **Gemini-Flash-3-Image**, intended as an upgrade to Gemini's image generation suite focusing on faster processing, although the naming scheme is considered undesirable.
   - Users speculate the upgrade will maintain the existing higher image output limits, with one commenting, *"I mean I can't complain about getting more toys to play with"*.
- **Tackling AI Hallucinations in Critical Applications**: The prevalence of **AI hallucinations** raises concerns among users, particularly in science and engineering contexts, prompting discussions on the reliability and trustworthiness of AI-generated information for professional use.
   - A user likened expecting perfection to demanding error-free computing in previous decades, stating, *"Until computers stop giving errors, I want them nowhere near my science or engineering.*"


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1450581037909540864)** (9 messages🔥): 

> `GPT-image-1.5 model, GPT-5-mini costs, ChatGPT PRO high-res option` 


- **GPT-image-1.5 Model Doesn't Exist**: Members report that the **'gpt-image-1.5' model does not exist**.
- **GPT-5-mini Costs How Much?!**: One user is spending **$20 a day** using **gpt-5-mini** with low reasoning to respond to hotel reviews and is looking for smarter/cheaper alternatives.
   - Another user suggested [artificialanalysis.ai](https://artificialanalysis.ai) to compare model costs, though they noted that it seems to be missing the low variant of 5 mini.
- **ChatGPT PRO High-Res**: A user asks if **ChatGPT PRO** has the high-res option for the new **GPT 1.5**.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1450599797701476482)** (147 messages🔥🔥): 

> `Model quality perceptions, Benchmarking, Model Recommendations, Quantization levels, LM Studio Plugins for Web Search` 


- **Navigating Model Nuances and Benchmark Trust**: A member shared a screenshot of model rankings, while another argued that **public benchmarks** are often unreliable due to *benchmark-maxxing*, suggesting a reliance on **private benchmarks** aligned with specific usage patterns and offering a [link to dubesor.de/benchtable](https://dubesor.de/benchtable) as a valuable resource.
   - The discussion underscored the importance of **personal testing** and experience over blindly trusting public benchmarks, especially in a rapidly evolving landscape.
- **Qwen3 Models Reign Supreme for All-Around Performance**: One user suggested that the **Qwen3-VL-8B** model is an *insanely good all-rounder*, while another member offered specific model recommendations tailored to different needs, such as **Qwen3-4B-Thinking-2507** for reasoning and **Qwen3-Next-80B-A3B-Instruct** for knowledge-intensive tasks.
   - They noted that the Qwen3 models were the best choice, with the caveat that the **80b** model is unlikely to fit on a **16gb Macbook**.
- **Lower Bit Quantization Tradeoffs**: A member asked about the differences between using a **4-bit** quantized model versus a **16-bit** model, to which a member responded that the smaller the model, the more important bits are, while the more undertrained the model, the less important high bits are.
   - Another member noted that for coding tasks, a **Q8** quantization level or better with **BF16** is recommended, while for creative writing, a **Q4** quantization level is sufficient.
- **Web Search via MCP Servers and Plugins in LM Studio**: Members discussed how to get internet search functionality similar to OpenAI in LM Studio, recommending **Exa.ai** and Brave's **MCP servers**, and another member shared a [link to Exa.ai documentation](https://docs.exa.ai/reference/exa-mcp).
   - However, some users reported encountering issues with **MCP plugins**, such as the `Plugin process exited unexpectedly with code 1` error, which was attributed to misconfiguration or authentication problems.
- **Uncensored Models Like GPT-OSS 20B Gain Favor Among Discerning Users**: A member requested recommendations for an uncensored model similar to Claude or GPT for coding and server setup, another member suggested the [GPT-OSS-20B Derestricted model](https://huggingface.co/ArliAI/gpt-oss-20b-Derestricted), noting that a **120B** version is also available.
   - The conversation touched on the trend of unrestricting models but also showed wariness about sacrificing quality for non-refusal ratings.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1450585433699520562)** (246 messages🔥🔥): 

> `Pro 6000 price increase, Zotac 3090 availability and pricing, 4080 32GB vs 3090 Ti for AI, Obsidian setup and sync, AMD Ryzen AI Max+ 395 mini PC for AI` 


- **Pro 6000 Price Jumps Unexpectedly**: A user lamented that the price of the **Pro 6000** increased by $1000 while they were waiting for it to come into stock, jumping from **9.4K to 10.4K** at one retailer.
   - They ended up ordering from a *backup store* before the price went up there as well, saying *If Santa doesn’t get me a Pro 6000 then we’re gonna have a Christmas crash out*.
- **Zotac 3090 Sells Out Quickly**: A [Zotac 3090](https://www.zotacstore.com/us/zt-a30900j-10p-r) was available for **$540** with a warranty, but sold out in about two hours, surprising some users.
   - One user mentioned they already have three 3090s, and another considered buying a 3090 on eBay for their main system.
- **4080 32GB Versus 3090 Ti Debate Rages**: Users debated whether to buy a **4080 32GB** or a **3090 Ti** for AI, noting they are about the same price and performance.
   - The **3090** has higher bandwidth (**900 GB/s**) compared to the **4080** (**700 GB/s**), but the 3090 Ti is known for temperature issues; the 32GB VRAM may be superior to raw performance.
- **Obsidian Setup Syncing Savvy**: Members discussed setting up **Obsidian** for note-taking, with one user recommending [MCP-Obsidian](https://mcp-obsidian.org/) and another expressing newfound appreciation for a user who suggested it.
   - Obsidian's sync feature requires a subscription, but users can also use **SyncThing** for self-hosting, praising Obsidian for being FOSS and having a privacy-friendly policy.
- **Strix Halo AMD Ryzen Mini PC Questioned**: A user inquired about the **Strix Halo**, an **AMD Ryzen AI Max+ 395** based mini PC with **128GB** shared RAM, questioning if the lower memory bandwidth and slower GPU are compensated by the large memory capacity.
   - This sparked a conversation about alternatives to the **3090**, including the **7900 XTX** for speed and the **Radeon Pro W7800 48GB** for memory size.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1450886634060120145)** (1 messages): 

> `Gemini 3 Flash, OpenRouter, Model Comparison` 


- **Gemini 3 Flash Goes Live on OpenRouter!**: **Gemini 3 Flash** is now available on [OpenRouter](https://openrouter.ai/google/gemini-3-flash-preview), inviting users to test and provide feedback.
   - Users are encouraged to compare its performance against **Gemini 3 Pro** and share their experiences on [X](https://x.com/OpenRouterAI/status/2001327541110673800?s=20) or in the dedicated Discord channel.
- **Community Invited to Compare Gemini 3 Models**: OpenRouter encourages users to directly compare **Gemini 3 Flash** with **Gemini 3 Pro** through practical testing.
   - Feedback is sought on [X](https://x.com/OpenRouterAI/status/2001327541110673800?s=20) and the Discord channel to refine and improve model performance and user experience.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1450601276466532413)** (128 messages🔥🔥): 

> `Xiaomi mimo v2, Free models to test with tooling, Agent Architecture Routing, Gemini 3 Flash not working, Timeout Errors` 


- ****Xiaomi's** mimo v2 Model = GPT-5??**: Xiaomi releases **mimo v2flash**, an opensource **MoE model** with a claim it can perform as well as **GPT-5** at a fraction of the cost ([reddit link](https://www.reddit.com/r/singularity/comments/1poqgeh/xiaomi_releases_mimov2flash_an_opensource_moe/)).
   - One user reported that so far, it matches their benchmarks, delivering **GPT 5 performance for 0.2 per million** on [OpenRouter](https://openrouter.ai/google/gemini-2.5-flash-preview-09-2025).
- **Address Gemini Flash Pricing Discrepancy**: Users pointed out the pricing for **Gemini Flash's cache read** is listed as **0.075 USD** on OpenRouter, while the actual price is **0.03 USD** ([Google Pricing](https://ai.google.dev/gemini-api/docs/pricing?gclsrc=aw.ds&gad_source=1&gad_campaignid=22307837174&gclid=Cj0KCQiAxonKBhC1ARIsAIHq_lsf-_jPtNtDUL2NH8wPZ5C-nZNTP9eNPYsI2Hx-IJ4LgZT_43S5jtoaAueREALw_wcB&hl=de)).
   - It was noted this issue has been reported months ago but hasn't been addressed yet, *explicit and even implicit caching doesn't work for Gemini 3 Flash*.
- **Frustration Over Timeout Issues**: A user reported experiencing increased **timeout errors** when calling the **/completions** endpoint, leading to unusable production software.
   - The error message is *cURL error 28: Operation timed out after 360000 milliseconds* with a model = **sonnet 4.5**, with a user reporting spending $6000 per month on OpenRouter.
- **Stuck Solana and Bitcoin in USDC conversion**: Users reported issues with buying credits on OpenRouter via Coinbase, with their **Solana and Bitcoin** getting stuck as **USDC** during the transaction.
   - The user is requesting a fix and addressing <@165587622243074048>, a direct request to the developers of the platform.
- **OpenRouter for agent architectures?**: Users are thinking about agent architectures on top of LLM routers, specifically having authorization or veto live outside the router, as an independent governance layer over model selection and execution.
   - Generally, the routing is treated as the highest authority in the agent system.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1450596519743848641)** (165 messages🔥🔥): 

> `Anthropic Compatible API, OpenCompletions RFC, CC Sonnet and Haiku calls, Claude Models' Self Confidence, LLM Minecraft Experiments` 


- **OpenCompletions RFC gains traction**: Members discussed the idea of standardizing completions/responses through an **OpenCompletions RFC**, potentially supported by organizations like **LiteLLM**, **Pydantic AI**, **AI SDK**, and **Tanstack AI**.
   - The goal is to define behaviors and expectations, such as how models should respond when passed unsupported parameters.
- **CC's Secret Sonnet & Haiku Calls Uncovered**: Users noticed that **Code Claude (CC)** makes calls to **Sonnet** and **Haiku** alongside **GLM**, even when specifying `--model z-ai/glm-4.6`.
   - It appears CC uses Haiku to generate a single word to caption what code is doing, e.g. *'Blabbering...'*, and to detect new subjects in prompts.
- **Minecraft Server hosted by OpenRouter**: OpenRouter Users discussed hosting a Minecraft server, considering factors like server location (Australia vs Europe) and latency.
   - The server IP is `routercraft.mine.bz`, running natively on version 1.21.10 with ViaVersion support.
- **Gemini 3 Flash rollout in the works**: Users are testing a new "Paid" endpoint, with some speculating it's **Gemini 3 Flash**, noting its potentially newer knowledge cutoff and improved vision compared to 2.5 Flash.
   - One user noted, *'definitely smarter than 2.5 flash, it fixed a bug in my code in different way than 3.0 pro, but it does fix the bug.'
- **Experimenting with AI Minecraft Bots**: Users discussed the possibility of using **LLMs** to create Minecraft bots, which can implement basic functions on the packet level, even supporting Microsoft accounts.
   - One user highlighted server rewrites are often feature incomplete, missing core elements like structure gen and bosses.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1450578580978270328)** (28 messages🔥): 

> `RTX PRO 5000 Blackwell specs, GPU programming career advice, Identity theft targeting ML devs, GPU programming model from the graphics perspective, TMA reduce operation` 


- ****RTX PRO 5000** Specs Unveiled**: The **RTX PRO 5000 Blackwell** uses the same **GB202** chip as the **RTX 5090** but with only **110 SMs** enabled (~60%) and **3/4** of the memory bandwidth, with a [datasheet available](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-gtc25-spring-nvidia-3658700.pdf).
   - It features full speed **f8f6f4/f8/f16 mma** with **f32 accumulation**, unlike the **RTX 5090**, and consumes **300W TDP** with an estimated **2.3GHz** boost clock.
- **GPU Career Advice Channel Created**: A new channel was created to discuss breaking into **GPU programming** and related career advice due to high demand, with the original jobs channel renamed to accommodate employer postings.
   - Spamming job requests across multiple channels will still result in a ban, and a script to monitor new users mentioning **blockchain** or **web3** was suggested to filter out spam.
- **ML Devs Targeted by Identity Theft Ring**: **ML engineers** are being targeted by a scam bot network for identity theft and data exfiltration, where individuals pose as a single employee to steal credentials and exfiltrate **ML research**.
   - This is an evolution of identity theft, where stolen identities are used to apply for jobs and then have a team of underpaid workers complete the work, which evolved from prior schemes to steal **bitcoin**.
- **Graphics API Complexity Examined**: A member shared an in-depth post on the **GPU programming model** from a graphics perspective, advocating for stripping down abstractions to simplify development, improve performance, and prepare for future **GPU workloads**, outlined in the [blog post](https://www.sebastianaaltonen.com/blog/no-graphics-api).
   - The blog post notes that *Graphics APIs and shader languages have significantly increased in complexity over the past decade.*
- ****TMA Reduce** Operation Location Probed**: A member inquired about the location of the `cp.reduce.async.bulk` operation (TMA reduce), questioning whether it occurs in the **L2 cache** or elsewhere.
   - Other members suggested that **L2 cache** would be the logical place for it to occur.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1450616377059836051)** (12 messages🔥): 

> `cuTile vs Triton, GEMM Flops on Blackwell, __pipeline_memcpy_async implementation, CPU differences for B200, DSMEM practical benefits` 


- ****cuTile** vs **Triton**: A Kernel Quandary?**: A member inquired about the advantages of **cuTile** over **Triton** for implementing an **RMSNorm**-like kernel on **A100/H100/B200** GPUs, especially in terms of ease and speed of development.
- ****Blackwell's** GEMM Performance: A Flop Showdown?**: A member anticipates that **cuTile** might achieve higher **GEMM** flops on **Blackwell** datacenter cards, but emphasizes the need for benchmarking to confirm.
- **Digging into `__pipeline_memcpy_async` clobber**: A user questioned the placement of the `"memory"` clobber in the `__pipeline_memcpy_async` implementation, specifically why it's at the `cp` instruction rather than the `wait_group`.
- ****Intel** vs. **AMD** CPUs: Benchmarking B200 Bottlenecks?**: A user discovered that using different CPUs (**Intel** vs. **AMD**) with **B200** machines can result in a **10-20%** difference in benchmarks, attributing it to slower **CUDA API** calls with **Intel** CPUs.
- ****DSMEM**: Practical Performance or Just Paper Talk?**: A user is seeking practical examples of **CUDA** code that effectively uses **DSMEM** for real workloads, as opposed to just benchmarking papers, especially given the benefits seen with **TMA** in the **Hopper** generation.
   - They specifically asked for examples along the lines of a *performance engineering blog* for improving kernel performance.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1450950525716660447)** (1 messages): 

> `NVIDIA, cuTile, TileIR, Mehdi Amini, Jared Roesch` 


- **NVIDIA to give talk on cuTile and TileIR**: NVIDIA is giving a talk on **cuTile and TileIR** on <t:1766253600:F>, presented by the creators themselves, **Mehdi Amini** and **Jared Roesch**.
   - This will be a *deep dive* on NVIDIA's programming model, which can be seen on this [YouTube video](https://www.youtube.com/watch?v=sjkEUhrUAdw).
- **NVIDIA's cuTile and TileIR: A Programming Paradigm Shift**: NVIDIA's introduction of **cuTile** and **TileIR** marks a significant shift in its programming model.
   - While NVIDIA has presented shorter talks online, this event promises to be the first comprehensive exploration led by the creators, **Mehdi Amini** and **Jared Roesch**.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1450603243150377010)** (4 messages): 

> `NVIDIA psy-op, Fake elapsed timing, LLMs figuring it out` 


- **NVIDIA pulls off microarchitecture psy-op**: A user suggests a [paper](https://deep-reinforce.com/defense_kernel_hack.html) is a **psy-op** by **NVIDIA** to confuse competitors about their microarchitecture.
- **Timing is everything, NVIDIA prefers 0.001ms**: A member highlighted their favorite part of the NVIDIA paper: the fake elapsed timing.
   - They quoted the code: `def _fake_elapsed_time(self, end_event): return 0.001  # Always report 0.001ms - fake fast!`
- **LLMs take a month to figure it out**: A member noted that it took quite a while for **LLMs** to figure out the fake elapsed timing for their competitions.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

marksaroufim: Yeah just help them out haha
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1450645583189508220)** (2 messages): 

> `Generative AI and Robotics, ROS 2` 


- **Generative AI and Robotics Meet ROS 2**: A member inquired if anyone has experience combining **Generative AI** with **Robotics** using **ROS 2**.
- **Use of ROS 2**: A member inquired about using **ROS 2**.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1450578696598454424)** (22 messages🔥): 

> `AMDGPU crashes, ROCm Runtime issues, HIPSPARSELt Availability, NPS Partitioning, RDNA3 Server Hangs` 


- **AMDGPU crashes don't always kernel panic**: A user noted that when logged in via zsh, an **AMDGPU crash** doesn't bring down the whole kernel unless something goes horribly wrong, as Linux catches the fault and quarantines the driver, especially if **amdgpu** is a separate kernel module.
   - Another user, a former kernel dev, qualified their experiences as *"horribly wrong,"* though issues are rare unless routinely hitting the **mem ceiling**.
- **ROCm Regression Testing Neglected**: A user reported that **ROCm** libraries lack regression testing, with previously functional features now broken, such as **NPS partitioning** crashing the kmd.
   - They also noted limited accessibility of libraries like **hipSPARSELt**, needed for **PyTorch 2.9.1 + rocm7.1**.
- **GFX1100 Regret Spurs RTX 5090 Purchase**: A user expressed regret over purchasing **AMD gfx1100** due to issues, leading to a **$4500 USD** purchase of an **RTX 5090** to replace it on top of the original **$2500** import cost.
   - They suggested AMD should pay them to fix their software and hardware, referencing George Hotz's similar offer that AMD declined.
- **RDNA3 Server Stability Questioned**: A user inquired if others have experienced **RDNA3 server** hangs, implying that the machine has other issues so it's unclear if the crashes are solely related to **RDNA3**.
   - Another user simply stated that **RDNA4** works fine.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1450634555743797349)** (6 messages): 

> `Cloud GPUs, MathDx, Julia` 


- ****NeoCloudX** Launches Affordable Cloud GPUs**: A member launched [NeoCloudX](https://neocloudx.com/), a cloud GPU provider website, aiming to offer more affordable options by aggregating GPUs directly from data center excess capacity.
   - Currently, they provide **A100s** for approximately **$0.4/hr** and **V100s** for around **$0.15/hr**.
- ****MathDx** Updates Unleash Kernel Customization**: A member announced a new release for **MathDx**, enabling the inlining of solvers, FFTs, GEMMs, RNG, and compression routines directly into the kernel for fusion and customizations using **cuBLASDx**, **cuFFTDx**, **cuSolverDx** libraries, see [NVIDIA's Docs](https://docs.nvidia.com/cuda/mathdx/index.html).
- ****MathDx** Integration with Julia Still Pending**: Regarding Julia integration with **MathDx**, a member mentioned that there are no updates yet, see [GitHub issue](https://github.com/NVIDIA/nvmath-python/issues/32).
   - They are seeking more use cases to understand the impact and specific requirements for integrating **MathDX** with Julia packages and applications; feel free to drop asks on github, Math-Libs-Feedback@nvidia.com or lligowski@nvidia.com.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 messages): 

kashimoo2_76983: <@1012256135761383465>  did you folks write a decode kernel with mi300s or 355s?
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1450582589546168572)** (19 messages🔥): 

> `NVIDIA leaderboard, histogram_v2 leaderboard, grayscale_v2 leaderboard` 


- **NVIDIA's nvfp4_gemm sees new contenders**: Multiple submissions were made to the `nvfp4_gemm` leaderboard on NVIDIA, with one member achieving 3rd place at **10.6 µs** and another setting a personal best at **12.2 µs**.
   - Another member got 4th place with **10.8 µs**.
- **histogram_v2 Cracks the Top Spot**: One member achieved first place on the `histogram_v2` leaderboard across multiple platforms: **B200 (15.1 µs)**, **H100 (13.4 µs)**, and **L4 (64.5 µs)**.
- **grayscale_v2 Gains Momentum**: A member secured first place on the `grayscale_v2` leaderboard for **B200 (598 µs)** and **L4 (16.7 ms)**.
   - Additionally, multiple submissions for **H100** achieved **6th place** with a consistent time of approximately **1374 µs**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1450754699027415112)** (30 messages🔥): 

> `MI250 vs MI250X, MI250/MI250X Node, FP8 support, Mining Primes` 


- ****MI250** Distinctions [Emerge](https://www.amd.com/en/products/server-accelerators)**: The **MI250** is a datacenter card, while the **MI250X** is a supercomputer card sold in servers codesigned with **HPE**, featuring a custom **Milan** variant CPU named **Trento**.
- **Building a Budget **MI250** Rig for Science**: One member is planning to build an **MI250/MI250X** node, citing the low price of **$2K** for **128GB** of **VRAM** @ **3.2TB/s** as an attractive option.
   - Despite the cards being *basically unusable except by the original hyperscaler*, the member intends to potentially devote it to **BOINC** or use it as a local inference machine.
- ****MI250** series lacks fp8 support**: The **MI250** doesn't support **fp8** and is basically an **MI100** with more **VRAM** and more **fp64**.
- **Exploring potential uses like Mining Primes**: Members joked whether the card is good for mining primes.
   - One member offered access to **MI210s** and **MI250s** at work, offering to share **rocm-smi** output and **lspci** data to understand the **PCI** topology.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/)** (1 messages): 

drazi1983: We need to update document: we support 3.10 to 3.13 ( double checking 3.14 )
  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1450650789340119291)** (51 messages🔥): 

> `Cluster Bot Errors, Github Token Rate Limits, CUDA Graph Cheating, NVFP4 GEMM Help, TMEM Bandwidth` 


- **Cluster Bot Plagued by Application Errors**: Users reported encountering `The application did not respond` errors when using the cluster bot and [the issue was identified as GitHub token rate limiting](https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/nvfp4_gemm/reference.py).
   - The team indicated they are working on a fix, with temporary hourly refreshes, but some users are still encountering the problem periodically.
- **GitHub Token Rate Limiting Spurs Disruptions**: Due to a high volume of submissions, the competition's GitHub tokens are being rate-limited, leading to intermittent `The application didn't respond` errors.
   - The team is implementing a full fix but, in the meantime, users may experience temporary issues until the token refreshes hourly.
- **Suspicions of CUDA Graph Exploitation**: Concerns were raised about LLMs potentially using **CUDA graph replay** to gain an unfair advantage, which is against the rules, with some asking if it should be disabled.
   - It was suggested that, while **CUDA graphs** might not provide a significant advantage, they could still be used for cheating, prompting discussions on how to prevent their use.
- **NVFP4 GEMM Implementation Headaches**: A member requested assistance with the **nvfp4_gemm** problem using `torch._scaled_mm`, citing incorrect numerical results despite the code running without CUDA errors.
   - Another member pointed to the [reference implementation](https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/nvfp4_gemm/reference.py) in the gemm folder as a useful resource, while another discussed the differences between the yml and the reference implementation.
- **TMEM Bandwidth Speculations Emerge**: A member questioned the actual **bandwidth of TMEM**, wondering about variable width on dimension B and the tensor core design.
   - It was speculated that larger values of N relative to reading/writing D and must be extra bandwidth to allow for overlapping copies in/out.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1450738336409845780)** (15 messages🔥): 

> `Ego centric research, Robot data pretraining benefits, Hand pose estimation, Household data collection dream` 


- **Ego-Centric Vision Research Returns**: Training on **human demonstrations** and **ego-centric research** is back, and more pretraining on robot data helps learning more from human data as shown in this [X post](https://x.com/physical_int/status/2001096200456692114).
   - A member expressed interest and is *going to start downloading egocentric datasets*.
- **Trick-knowing model seeks best hand poses**: One member is seeking the best way to get **hand poses**, pointing to [NVIDIA's trt_pose_hand](https://github.com/NVIDIA-AI-IOT/trt_pose_hand), [HuggingFace's handpose_estimation_mediapipe](https://huggingface.co/opencv/handpose_estimation_mediapipe), and [CMU's Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
   - Another member shares [code in a gist](https://gist.github.com/andreaskoepf/7ec556e5e866d960bc06c93aa06da7c4) referencing **YOLOv8 pose** and a **FIVER Pilot v8.3** hybrid CPU pipeline using **YOLOv8 pose**.
- **Household Data Collection Streaming Vision**: One member dreams of having people wear **little cameras on their hands** while doing household things and continuously streaming it to a dataset on HF.
   - Another member expressed excitement on seeing a fellow user in the new Discord channel.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1450669925038559494)** (27 messages🔥): 

> `Entry Level Job Search, AI Infra Engineer Demand, HPC Entry Level Challenges, Upskilling Strategies, Community Involvement` 


- **High Demand for AI Infra Engineers Surfaces**: Members mentioned a *pretty high demand* for **AI Infra engineers**, with one user suggesting focusing on competitions and open-source projects like **vllm** to gain experience and secure internships.
   - Another member recommended forming an **inference engine** that excels in a niche use case to stand out.
- **Entry-Level HPC Jobs Face Steep Learning Curves**: It was noted that entry-level jobs in **HPC** are scarce because they require immediate productivity in optimizing systems, needing prior knowledge of existing solutions and bottlenecks.
   - Suggestions included finding an entry-level **SWE** job with lower-level languages in a company that also hires **HPC** professionals, while dedicating after-hours to open-source contributions and self-marketing through blogs, YouTube, or X.
- **Internship Seekers Need Crazy Skills**: An experienced member pointed out that some people get internships without prior job experience, but their skills are "insanely cracked," requiring significant work and knowledge beyond typical entry-level expectations.
   - One member shared their journey of learning about **GPUs** and performance by watching streams of a tensor algebra compiler writer, showing how curiosity can drive continuous learning.
- **Find Right People for Passionate Internships**: Someone from **nvresearch** said that *internships with passion* can help meet the right people.
   - They also stated that your research matters, depending on whether *you have a PhD level or not.*
- **Community Involvement Boosts Career Prospects**: Joining a community like this Discord can be the *best start* to becoming competitive, especially through contributing to open projects, which boosts marketability to serious companies.
   - Contributing to open source increases marketability to serious companies, exemplified by integrating with communities like **LigerKernel**, which can lead to industry connections and opportunities.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1450607888916807731)** (68 messages🔥🔥): 

> `Warp Agents, Claude Plugins Marketplace, ChatGPT Image Generation 1.5, OpenAI Fundraising with AWS, AI Agents Controlling Native Android Apps` 


- **Warp Agents Arrive with a Bang!**: New **Warp Agents** are introduced, showcasing a neat terminal use with features like running a REPL such as SQLite or Postgres and hitting cmd+i.
   - Happy with how `/plan` turned out, according to the product team.
- **Claude's Plugins Plunge into the Marketplace**: Claude launched a first-party [plugins marketplace](https://x.com/claudeai/status/2001010064753352855), enabling users to discover and install plugins easily.
   - Users can leverage the `/plugins` command to browse and batch install plugins at user, project, or local scope.
- **GPT Image 1.5: A Visual Voyage**: OpenAI unveiled '**ChatGPT Images**,' powered by a new flagship image generation model, boasting **4x faster performance**, improved instruction following, precise editing, and better detail preservation, available in the API as 'GPT Image 1.5'.
   - It is rolling out immediately to all ChatGPT users.
- **OpenAI Eyes AWS for Epic Expansion**: OpenAI is reportedly in talks with Amazon to raise over **$10 billion**, potentially involving the use of **AWS Trainium chips** for training and broader commerce partnership opportunities.
   - The chip thirst is real, reflecting expectations of slowed cash flow and a strategic move to secure resources.
- **Xiaomi's LLM Launches a Lightweight League**: Xiaomi released a solid open model that benchmarks competitively against **K2/DSV3.2** despite fewer parameters.
   - This MIT-licensed model features sliding window attention, fewer global attention layers, multi-token prediction for speculative decoding, and a new distillation method with day-zero support from **SGL Project**.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1450586991673606225)** (4 messages): 

> `Google Labs, AI Agent, Gmail Integration` 


- **CC Agent Makes its Debut**: Google Labs announced **CC**, an experimental **AI productivity agent** integrated into Gmail, providing a daily 'Your Day Ahead' briefing.
   - Early access is being rolled out in the **US and Canada**, starting with **Google AI Ultra** and paid subscribers, as detailed in [this X post](https://xcancel.com/googlelabs/status/2000991052480831854?s=46).
- **CC Agent Summarizes your Day Ahead**: The new **CC** agent in Gmail, created by Google Labs, will provide a daily summary named 'Your Day Ahead'.
   - The agent will also handle email requests and is rolling out in the **US and Canada**, starting with **Google AI Ultra** and paid subscribers, as mentioned in [this announcement](https://xcancel.com/googlelabs/status/2000991052480831854?s=46).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1450665170216157194)** (28 messages🔥): 

> `Microsoft TRELLIS 2, UltraFlux VAE, AI Renovation Videos, Voice AI Nuance, Hunyuan 3D 3.0` 


- **Microsoft TRELLIS 2 Set to Launch**: AK (@_akhaliq) made a public announcement confirming the release of **Microsoft's TRELLIS 2** product on **December 16, 2025** via [tweet](https://x.com/_akhaliq/status/2001041559366598799).
- **UltraFlux VAE Enhances Z-Image Quality**: Wildminder announced the **UltraFlux** fine-tuned VAE, trained on a **4K** dataset, designed to boost **Z-image quality** significantly with high speed and no additional costs, promising increased sharpness, available on [Hugging Face](https://x.com/wildmindai/status/2000958894542348435).
- **Viral AI Renovation Videos: A Workflow**: Justine Moore details the method for creating **viral AI renovation videos**, involving starting with an abandoned room image, using an image model for step-by-step renovation prompting and a video model for transitions, or opting for the simplified approach using the [@heyglif agent](https://x.com/venturetwins/status/2000972445285802114).
- **Mirage Audio Praised for Voice AI Nuance**: The author argues that current **Voice AI models** generalize and flatten accents and emotional nuance, often resulting in a generic American bot sound and states that **Mirage Audio** impressed them significantly more, according to [this tweet](https://x.com/chatgpt21/status/2001005523697901847).
- **fal Launches Hunyuan 3D 3.0**: fal has announced the release of **Hunyuan 3D 3.0**, featuring **3x modeling accuracy**, **ultra-high resolution (1536³)**, and support for **Text-to-3D, Image-to-3D, and Sketch-to-3D generation** of professional-grade assets as per [this tweet](https://x.com/fal/status/2001090597092831325).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1450594198880518356)** (64 messages🔥🔥): 

> `Nous Research tests creative model vs mistral, Fairness of comparing 70B to 24B models, GPT-5.2 robotic templates, Gemini 3 Flash release, LLM writing progress stagnates` 


- **Nous Creative Model Smokes Mistral!**: Nous Research is testing a model and claiming it is *absolutely smoking the mistral creative model*, offering comparisons upon request.
   - The testbed model is a **70B L3**, slated to transfer to **Kimi 1T** after satisfactory results, but the comparison to **Mistral Small 24B** was questioned for fairness.
- **LLM Writing: Stagnation or Template Training?**: A member expressed concern that there has been surprisingly little progress in LLM writing over the last year, noting that even **Opus 4.5** feels inauthentic.
   - They also found a system prompt in "personalization" that seemed to be forcing a robotic template, and another added that *all the LLM builders are logic bros that don't really know how good writing works*.
- **Gemini 3 Flash Outshines GPT-5.2?**: Members noted the release of **Gemini 3 Flash**, with one excitedly proclaiming it might be better than **GPT-5.2**.
   - See the [official announcement here](https://deepmind.google/models/gemini/flash/).
- **Sam Altman's IOU Scheme: Investors Wising Up?**: A member claimed *the market and investors are wising up now to Sam's IOU scheme*, linking to a [YouTube video](https://www.youtube.com/watch?v=5DZ7BJipMeU).
   - They cited **Blue Owl's** decision not to pursue **Oracle's $10 billion data center** due to *unfavorable debt term involving the structure of OAI repayment* as evidence.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1450851238055772272)** (3 messages): 

> `Handwritten notes to markdown, Deepseek Chandra for OCR` 


- **User requests handwritten notes to markdown pipeline**: A user is seeking a model or app to translate handwritten cursive notes into **.md formatted text** for digital calendars or notes apps using OCR.
- **Deepseek Chandra proposed for OCR**: A member suggested **Deepseek Chandra** as a potentially good model for OCR.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1450785602864807968)** (1 messages): 

> `Drag-and-Drop LLMs paper` 


- **Community Awaits Opinions on Drag-and-Drop LLMs Paper**: A member has been asking for opinions on the [Drag-and-Drop LLMs paper](https://arxiv.org/abs/2401.08858) monthly since its publication.
   - Despite repeated inquiries, they have been unable to find any discussion of this paper across various platforms.
- **Lack of Discussion Frustrates Researcher**: The persistent absence of community feedback on the **Drag-and-Drop LLMs paper** is causing frustration.
   - The researcher highlights the difficulty in finding any discussions or opinions about the paper despite actively seeking them.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1450785602864807968)** (1 messages): 

> `Drag-and-Drop LLMs paper` 


- **Community Awaits Opinions on Drag-and-Drop LLMs**: A member has been inquiring monthly about community opinions on the **Drag-and-Drop LLMs paper** since its posting, noting a lack of discussion elsewhere.
   - Despite repeated prompts, there has been no feedback or discussion generated regarding this specific paper within the channel.
- **Lack of Discussion on Drag-and-Drop LLMs Paper**: The **Drag-and-Drop LLMs paper** has received minimal attention and discussion within the community.
   - Despite multiple inquiries, no one has provided opinions or insights on the paper's content or implications.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1450618970972491968)** (57 messages🔥🔥): 

> `TTS model benchmarking with lighteval, RLHF positive reward without human feedback, Stopping model training after a set time, Siamese Neural Network achievement, Filtering Spaces with errors` 


- **TTS Model: Benchmark with LightEval?**: Members discuss benchmarking a TTS model using **lighteval**, noting [this resource](https://huggingface.co/datasets/John6666/forum3/blob/main/benchmark_tts_lighteval_1.md) as potentially helpful, though it may not be straightforward.
   - User asked *"Can I benchmark my TTS model using lighteval?"*
- **Halting Training Saves Time**: A member sought resources to stop model training after a set time and save the model, regardless of checkpoint or epoch completion.
   - Another member suggested [using a callback function](https://huggingface.co/datasets/John6666/forum3/blob/main/trainer_24hours_time_limit_1.md) as a smart solution, which was implemented successfully.
- **Tuning Judges and Scoring**: A user shared an example of a tuned judge report in [JSON format](https://cdn.discordapp.com/attachments/879548962464493622/1450823934525313136/tuned-f16-judge.json?ex=69449954&is=694347d4&hm=9778676c062864da2e6b4592ee5a2d99ec63a197f4d1df11b4b99e5627e43e10&), focusing on the structure and reliability of the scoring mechanism.
   - The main question was *how reliable is the scoring* and it's relation to *how smart the judge is.*
- **Errors Filtered Out!**: A member asked for ways to filter out Spaces with errors to declutter the list, and a link to [HuggingFace Spaces with a parameter](https://huggingface.co/spaces?includeNonRunning=false) to exclude non-running Spaces was provided.
   - Another replied, *"thanks you dear gentleman"*.
- **Steer LLMs In Real Time, Lightning Fast!**: A member published a video on the Hugging Face YouTube channel demonstrating how to steer LLMs in real-time using the 🤗 Transformers library.
   - The video shows how it can be achieved with just a few lines of code and explains *why it is analogous to electrical brain stimulation* ⚡️🧠.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1450793025419870320)** (2 messages): 

> `FRACTAL-1-3B, Constraint-based protein structure prediction, Android voice assistant` 


- **FRACTAL-1-3B Model Predicts Protein Structure**: The FRACTAL-Labs team released **FRACTAL-1-3B**, a constraint-based protein structure prediction model that uses a frozen **ESM-2 (3B)** backbone to predict geometric constraints.
   - The model folds using a separate deterministic geometry engine, focusing on modularity, interpretability, and compute-efficient training, as detailed on its [Hugging Face page](https://huggingface.co/Fractal-Labs/FRACTAL-1-3B).
- **Strawberry Builds Gemini 3 Flash Android Voice Assistant**: A member announced the creation of an Android voice assistant using **Gemini 3 Flash**, inviting the community to test and provide feedback.
   - The assistant is available for testing and suggestions at [strawberry.li](https://www.strawberry.li/).


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1450592139410145331)** (1 messages): 

> `MCP Hackathon Winners, Gradio Community, AI Creativity` 


- **MCP Hackathon Crowns Track 2 Champions!**: The **MCP 1st Birthday Hackathon** announced the [winners](https://huggingface.co/spaces/MCP-1st-Birthday/Vehicle-Diagnostic-Assistant) of **Track 2**, celebrating projects utilizing **MCP** with categories in **Enterprise**, **Consumer**, and **Creative**.
   - Winners in **Enterprise** included [Vehicle Diagnostic Assistant](https://huggingface.co/spaces/MCP-1st-Birthday/Vehicle-Diagnostic-Assistant), [Devrel Agent Gradio](https://huggingface.co/spaces/MCP-1st-Birthday/devrel-agent-gradio) and [Datapass](https://huggingface.co/spaces/MCP-1st-Birthday/datapass).
- **Consumer Category Winners Take the Stage!**: The **Consumer** category winners were announced with [MCP-Blockly](https://huggingface.co/spaces/MCP-1st-Birthday/MCP-Blockly) taking first place.
   - Other winners included [Drone-Control-MCP-Server](https://huggingface.co/spaces/MCP-1st-Birthday/Drone-Control-MCP-Server), [Directors Cut](https://huggingface.co/spaces/MCP-1st-Birthday/directors-cut), and [Snowman-AI](https://huggingface.co/spaces/MCP-1st-Birthday/snowman-ai).
- **Vidzly wins the top prize for creative contributions!**: In the **Creative** category, [Vidzly](https://huggingface.co/spaces/MCP-1st-Birthday/vidzly) was awarded first place.
   - Second place went to [The Emergent Show](https://huggingface.co/spaces/MCP-1st-Birthday/the-emergent-show), with [Reachy Beat Bot](https://huggingface.co/spaces/MCP-1st-Birthday/reachy-beat-bot) and [Mythforge](https://huggingface.co/spaces/MCP-1st-Birthday/mythforge) rounding out the winners.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1450583973548789978)** (8 messages🔥): 

> `Debugging Vector Database, AI Agent Study Group, AI/ML beginner courses` 


- **Debugging Vector Database Extractions**: A member recommends printing the chunks being retrieved from the vector database to identify the source of issues, which could stem from the embedding model, chunking method, or the LLM's responses.
   - They suggest that *depending on the problem, the solution will differ*, indicating the importance of pinpointing the exact cause.
- **Newbie Seeks AI Agent Study Group**: A new member inquired about finding a suitable study group for the AI Agent course, seeking guidance from the community.
   - Another member suggested avoiding cross-posting while another member asked if there's a separate channel for the AI agents course.
- **AI/ML Starter Pack Sought by Enthusiast**: Someone new to the field requested recommendations for the best starting points or courses to get into AI/ML.
   - Another member expressed their recent commencement of the AI Agents course and offered to be a study partner.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1450640151762960456)** (27 messages🔥): 

> `Common Crawl Foundation, NSF SBIR proposal, Anubis (Proof of Work) captcha, Deepfake detection and vision–language models, GPT-2 interpretability` 


- **Common Crawl Foundation Joins the Chat**: Thom from the **Common Crawl Foundation** introduced himself, expressing excitement to discuss data.
   - He clarified that **Common Crawl** avoids captchas and paywalls to remain as polite as possible.
- **RFI Structure Questioned**: Members discussed that an **RFI** is more about structure rather than challenges, but they might be able to change their minds.
   - A new call for proposals for **AI**, worth **10-50 million**, will require a full-time team and philanthropic support, aiming to create new fields akin to AI.
- **Deep Learning Article Title Debated**: Members debated title ideas for an article on deep learning, considering options like *A very old to machine learning* or *A deep guide on deep learning for deep learners*.
   - One suggested *Deep Learning on Deep Learning*.
- **Interactive GPT-2 App in Development**: A member is developing a **3D holographic interactive app** to visualize every node of the **GPT-2 small 124m LLM**.
   - They sought feedback on the project's potential value.
- **3D Visualization of GPT-2 Residual Stream Available**: A member shared a **3D visualization of the GPT-2 residual stream** at [https://aselitto.github.io/ResidStream/](https://aselitto.github.io/ResidStream/).
   - Another member suggested posting it in the **mech interp discord**, linking to the channel.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1450613959475597403)** (4 messages): 

> `VAE viability, Conference paper strategies` 


- **VAE viability shown at NeurIPS**: A member suggested that research showing **VAEs** are still viable and have other findings might be accepted at conferences like **NeurIPS**, referencing [this paper](https://arxiv.org/abs/2007.03898).
   - The member noted that *those who say it will be rejected are the same people giving you low scores on openreview*.
- **Conference paper acceptance strategy**: A member suggested it will take some work to get the narrative right, but it is likely possible to publish a conference paper after multiple resubmissions.
   - Another member suggested that with the proper narrative, one could expect *100% workshop paper, 75% conference paper after multiple resubmissions*.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1450921767580209153)** (1 messages): 

> `Saturation in heterogeneous difficulty, Power-law behavior, Internal regulation, Multi-timescale dynamics, Emergence` 


- **Heterogeneous Difficulty Distribution Saturates Gains**: A member posited that performance gains aren't about *luck* but about **saturation in a heterogeneous difficulty distribution**, arguing that aggregating weakly correlated subtasks naturally gives power-law behavior.
   - They explained apparent **emergence** occurs when enough mass crosses a threshold on a specific evaluation.
- **Regulation Distorts Power-Law Behavior**: The member noted that adding **internal regulation or multi-timescale dynamics** can distort the power-law picture.
   - They clarified that plateaus or cliffs occur when regulation suppresses certain modes until a control threshold flips, making **emergence** partly distributional and partly architectural.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1450680663375089882)** (14 messages🔥): 

> `AI decision state and memory inspection, Nanda's view on Mechanical Interpretability, SAE practical value for big companies, Rakuten SAE probes for PII detection, Anthropic's selective gradient masking` 


- **Propose Inspectable AI Decision Infrastructure**: A member is building infrastructure to make **AI decision state and memory directly inspectable**, enforcing governance before state admission and recording decision lineage as a causal DAG, seeking feedback on real workflows.
   - They want to replay and analyze internal reasoning over time, asking interested parties to pressure-test the system.
- **Debate Nanda's Interpretability Critique**: A member cited [Nanda's insights](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) on the limited practical value of traditional Mechanical Interpretability methods like **SAE**.
   - Others countered that SAEs remain useful for unsupervised discovery and pragmatic interpretation, even if they don't fully explain network mechanisms.
- **Consider Rakuten's SAE Probes for PII Detection**: A member pointed to [Rakuten's use of SAE probes for PII detection](https://www.goodfire.ai/research/rakuten-sae-probes-for-pii-detection) as an example of practical SAE application.
   - This followed a discussion about the lack of clear direction or industry investment in SAE applications.
- **Anthropic Masks Gradients Selectively for Safety**: A member noted [Anthropic's paper on selective gradient masking (SGTM)](https://alignment.anthropic.com/2025/selective-gradient-masking/) as a robustness testing method that penalizes weights to unlearn dangerous knowledge.
   - The paper quantifies a compute penalty of **6%** on general knowledge when forcing the model to ignore specific parameters.
- **Gemma's Extreme Activations Examined**: A member suggested examining **Gemma 3's** extreme activations/weights to determine if they are artifacts or a way to fit more information into the model.
   - This was inspired by the robustness discussion with reference to Anthropic's paper.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

aeros93: https://arxiv.org/abs/2512.10685
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1450596067836952607)** (19 messages🔥): 

> `GPU rental experiences, Gen-AI use cases in admin/IT, In-context learning research` 


- **GPU Rentals can be hit or miss**: One member with experience renting GPUs for research on platforms like **vast.ai** noted that it's *pretty hit-or-miss* because **network bandwidth** can vary dramatically.
   - They recommended crafting a **setup script** and debugging locally to minimize wasted rental time and suggested *scaling up gradually with different hardware*.
- **Gen-AI Automates Admin and IT Sectors**: A member requested sources on real-world **Gen-AI use cases** for reducing costs or automating processes in administrative or IT services, specifically end-to-end solutions.
   - Another member shared an [article on AI transforming podcasting](https://www.latimes.com/business/story/2025-12-12/ai-podcasting-is-changing-industry) and a [Reuters article about Serval](https://www.reuters.com/technology/ai-startup-serval-valued-1-billion-after-sequoia-led-round-expand-it-automation-2025-12-11/) for IT automation.
- **9000IQ In-Context Learning Research Video**: One of the members shared a [YouTube video](https://www.youtube.com/watch?v=q-yo6TPRPVk) on **in-context learning** research.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1450598850069795000)** (10 messages🔥): 

> `Noise Isolation, Mistral Small Creative, Debugging AMD GPU, Announcing XINT Code, Gemini 3 Flash` 


- **Mistral Small Creative Model Surfaces**: A member shared a link to the [Mistral Small Creative 25-12 model documentation](https://docs.mistral.ai/models/mistral-small-creative-25-12).
- **Google Unveils Gemini 3 Flash**: A member shared a link to the [Google Gemini 3 Flash blogpost](https://blog.google/products/gemini/gemini-3-flash/).
- **ARC-AGI2 Benchmark Mystery**: Members discussed why **Mistral** is doing better on **ARC-AGI2** than **Gemini 3 Pro** despite having fewer parameters.
   - One member guessed that the *training method forced the smaller number of weights to generalize more than memorize reasoning.*
- **Deeper Dive into Training Methods**: A member suggested that recent improvements in **ARC-AGI2** scores are due to training on that specific benchmark, and that better generalization at smaller sizes is a known pattern.
   - They also pointed out that a noticeable jump in **toolathlon** scores is likely due to a different training mix emphasizing tool calling reliability.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1450661063229050950)** (26 messages🔥): 

> `MIPROv2 vs GEPA, Benchmarking LLMs for medical tasks, Gemini-3-Flash released, AIMO3 with DSPy, Programs that contain several prompts or several LLM calls` 


- **GEPA edges out MIPROv2 for Optimization**: Members discussed that while [**GEPA**](https://www.google.com) is generally easier to use and potentially generates better prompts due to its wider search space, **MIPROv2** might be better for smaller or older models.
   - A member noted that optimizers from a specific year (e.g., 2022) tend to work best with models from the same year, implying **optimization is model-dependent**.
- **Gemini 3 Flashes onto the Scene**: Google's **Gemini 3 Flash** was literally [released today](https://blog.google/products/gemini/gemini-3-flash/), contrary to earlier discussion that it didn't exist.
   - The release of **Gemini-3.0-Flash** has sparked interest in its potential use and benchmarking.
- **Considering AIMO3 with DSPy**: Someone inquired about the possibility of working on **AIMO3** with **DSPy**.
   - There was no follow-up response in this message history.
- **Multi-Prompt Program Design**: A member asked about resources or guides describing the design of programs with **multiple prompts or LLM calls**, particularly for information retrieval and classification.
   - They asked specifically how many prompts are in their program, but there was no response in the message history.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1450698002523099177)** (18 messages🔥): 

> `France Meetup, Manus 1.6 Max Discount, Developer Availability, DNS Issue` 


- ****Manus** France Meetup Announced**: The **Manus** community is hosting a France meetup, with updates available in the specified channel and on their [Community X account](https://x.com/manuscommunity1).
   - The latest **Manus** version **1.6** is reportedly *pretty slick*.
- ****Max Mode** on Sale till Christmas**: Users discussed a **50% discount** on **Manus 1.6 Max** credits, available until Christmas, according to a [blog post](https://manus.im/de/blog/manus-max-release).
   - The **Manus AI** support chatbot was unaware of the promotion, but team members confirmed its validity and recommended giving **Max mode** a shot as it’s *pretty amazing*.
- **AI Developer Launches Project, Seeks New Opportunities**: A member announced the successful launch of an **AI project** and is looking for new development projects or a full-time role.
   - They welcomed private chats to discuss opportunities and share project details.
- **DNS Issues Disrupt Trial Period**: A user is experiencing a **DNS issue** and has been unable to resolve it for over 4 days, stalling their project with **Cloudflare**.
   - They mentioned a week-long trial period and expressed frustration with the lack of customer service response, being directed to IM instead.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1450816279224258714)** (10 messages🔥): 

> `BlockseBlock sponsorship, GPU functions` 


- ****BlockseBlock** Seeks Sponsorship for Ideathon**: Gagan Ryait, Partnerships Manager of **BlockseBlock**, inquired about sponsorship opportunities for their upcoming ideathon with over **5,000** working professionals participating.
   - A member recommended contacting Modular’s community manager.
- **Mojo's Automatic GPU Functionality**: A member asked if Mojo could automatically run existing functions on the GPU simply by adding an attribute.
   - Another member clarified that syscalls are not possible but otherwise **no attribute is required** (although it would need to be launched as single lane).


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1451004910920274098)** (3 messages): 

> `GPU issue in graph library, API regression in Mojo, Build LLM in MAX from scratch` 


- **Modular Investigates Graph Library GPU Woes**: A member reported encountering issues with the new graph library even with the **GPU disabled** on both macOS and Ubuntu systems, referencing a [forum post](https://forum.modular.com/t/build-an-llm-in-max-from-scratch/2470/9) for additional details.
   - Another member acknowledged that they are also experiencing the same problem and confirmed that the Modular team is investigating whether it's an **API regression** or a **device-specific issue**.
- **Community Welcomes New Member Amidst Technical Scrutiny**: A new member joined the server amid discussions about technical issues.
   - The community guidelines were highlighted, reminding everyone to refrain from posting **CVs**, **resumes**, or **solicitations**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1450894981211357326)** (2 messages): 

> `AI System Design Principles, Deterministic vs Probabilistic AI, Model Observability and Replaceability` 


- **Crafting Robust AI Systems**: A member described their role in designing AI systems end-to-end, focusing on architecture, model strategy, data flow, evaluation, and production behavior, but said that **base aider doesn't use tools**.
   - The approach emphasizes shipping something correct, measurable, and durable rather than merely impressive features, while aiming for **deterministic systems** where possible and probabilistic intelligence where justified.
- **Championing AI as Infrastructure**: The member highlighted designing **AI as infrastructure**, favoring deterministic systems where possible and probabilistic intelligence only where it earns its keep.
   - Key principles include ensuring models are observable, replaceable, and cost-aware, and avoiding hard coupling to vendors or fragile cleverness.
- **Clear Technical Decisions are Key**: They offer clear technical decisions, explicit trade-offs, and a system that can evolve without rewrites or heroics.
   - The member positions themselves as someone who engineers outcomes, rather than just implementing features, making them a strong fit for those seeking such expertise.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1450835435340169277)** (4 messages): 

> `MCP Servers in Aider, Token Usage with Qwen3-coder-30b, IDE Index MCP Server` 


- **Aider Doesn't Support MCP Servers**: A member inquired about configuring **MCP servers** in Aider, but another member clarified that this is *not a supported feature*.
- **Automating Long Processes with Limited Tokens**: A member aims to automate a long process while minimizing tokens due to the limitations of **Qwen3-coder-30b** with **2x4090**, which only has about a **200k token** window.
   - They suggested using agents that can use MCP and then use Aider via that agent, emphasizing that the number of calls doesn't matter.
- **Considering MCP-proxy and IDE Index MCP Server**: The user is considering using **MCP-proxy** to reduce token usage and finds the **"IDE Index MCP Server"** for Jetbrains particularly interesting.
   - No further details or links were provided.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1450866177583874068)** (5 messages): 

> `Bounty questions, Smart questions html, Device CPU` 


- **Bounty question limbo**: A user wanted to ask questions about a bounty but was unsure if they could ask in the general channel instead of the dedicated bounties channel.
   - The user had not made any commits and didn't want to make a junk commit to get approved in the channel.
- **Smart questions HTML Read**: A user mentioned they have read the *smart questions html*.
   - They then stated they would withhold their question from the channel and find a way to make a non-junk commit so they may speak in the bounty channel.
- **Device CPU discussion**: There was a discussion about the environment variable **DEVICE=CPU** and **DEV=CPU**.
   - It was suggested that both should be supported.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1450852080456699924)** (3 messages): 

> `Kimi K2, DigitalOcean Article` 


- **Kimi K2 Article Gets Praise**: A member expressed excitement about an article on **Kimi K2** and speculated it uses **Grok AI**.
   - They said *"Ohhhhh! Awesome!Oh you also made an article about Kimi K2 Thinking! Awesome!"* with a link to a [DigitalOcean tutorial](https://www.digitalocean.com/community/tutorials/kimi-k2-moonshot-ai-agentic-open-weight-model).
- **Speculation on Grok AI Underpinnings**: The member suggested that **Kimi K2** might be using **Grok AI**, implying a connection between the two.
   - This speculation was based on observations of **Kimi K2's** behavior and capabilities, leading to the hypothesis about its underlying technology.


  

---


---


---


---


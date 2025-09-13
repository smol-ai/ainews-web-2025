---
id: MjAyNS0w
title: not much happened today
date: '2025-09-13T05:44:39.731046Z'
description: >-
  **Meta** released **MobileLLM-R1**, a sub-1B parameter reasoning model family
  on Hugging Face with strong small-model math accuracy, trained on 4.2T tokens.
  **Alibaba** introduced **Qwen3-Next-80B-A3B** with hybrid attention, 256k
  context window, and improved long-horizon memory, priced competitively on
  Alibaba Cloud. **Meta AI FAIR** fixed a benchmark bug in SWE-Bench affecting
  agent evaluation. LiveMCP-101 benchmark shows frontier models like **GPT-5**
  underperform on complex tasks with common failure modes cataloged. OpenAI
  highlights hallucination issues due to benchmark incentives, proposing
  calibration improvements. Community demos and tooling updates continue to
  evolve.
companies:
  - meta-ai-fair
  - huggingface
  - alibaba
  - openai
models:
  - mobilellm-r1
  - qwen3-next-80b-a3b
  - gpt-5
topics:
  - reasoning
  - model-efficiency
  - hybrid-attention
  - long-context
  - benchmarking
  - agent-evaluation
  - hallucination-detection
  - model-calibration
  - inference-complexity
  - model-pricing
people:
  - _akhaliq
  - tacocohen
  - pkirgis
  - sayashk
---


**a quiet day.**

> AI News for 9/11/2025-9/12/2025. We checked 12 subreddits, 544 Twitters and 22 Discords (189 channels, and 5258 messages) for you. Estimated reading time saved (at 200wpm): 464 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Happy [o1 anniversary](https://x.com/kimmonismus/status/1966627812858855624?s=46). Congrats to [Naveen Rao](https://x.com/rohanpaul_ai/status/1966378718009635087?s=46) and [Interaction](https://x.com/interaction/status/1965093198482866317) on buzzy new fundraises.

---

# AI Twitter Recap

**Edge Reasoning on-device: Meta’s MobileLLM-R1 (sub‑1B) goes open on HF**

- **MobileLLM-R1 (sub‑1B, open weights)**: Meta released a family of sub‑1B parameter reasoning models on Hugging Face with unusually strong small-model results: ~5× higher MATH accuracy vs Olmo‑1.24B and ~2× vs SmolLM2‑1.7B, while matching or surpassing Qwen3 accuracy on multiple reasoning benchmarks trained on only 4.2T tokens (≈11.7% of Qwen3’s 36T) according to [@_akhaliq](https://twitter.com/_akhaliq/status/1966498058822103330) and the model post [link](https://twitter.com/_akhaliq/status/1966499598433710361). Meta researchers emphasized the data efficiency and reasoning capability at this scale ([announcements](https://twitter.com/erniecyc/status/1966511167053910509), [more context](https://twitter.com/zechunliu/status/1966560134739751083)). Community demos arrived quickly via Anycoder/Spaces ([app](https://twitter.com/_akhaliq/status/1966528137858019713), [another](https://twitter.com/_akhaliq/status/1966532295403209138)).

**Qwen3‑Next‑80B (A3B): hybrid attention, 256k context, and heavy infra implications**

- **Architecture & inference complexity**: Alibaba’s new open‑weights Qwen3‑Next‑80B‑A3B introduces a hybrid attention design (Gated DeltaNet + Gated Attention) with high sparsity (≈3.8% active params vs 9.4% in Qwen3‑235B), a native 256k context window, and text‑only I/O. Adaptation required major engine changes: SGLang PR >6k LOC; vLLM >2.5k LOC per [@ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/1966419946885493098). Pricing on Alibaba Cloud is $0.5/$6 per 1M input/output tokens for the reasoning variant and $0.5/$2 without reasoning, cheaper than Qwen3‑235B ([details](https://twitter.com/ArtificialAnlys/status/1966523300781428788), [token usage](https://twitter.com/ArtificialAnlys/status/1966523306338893979)).
- **Performance & tradeoffs (community evals)**: Long‑horizon “working memory” and multi‑turn consistency are visibly improved; character‑level basics are strong though reasoning+character tasks are mixed; weaknesses include error inheritance, instruction‑following gaps, and long‑text hallucinations, per Zhihu analyses ([summary](https://twitter.com/ZhihuFrontier/status/1966415278922989813), [thread](https://twitter.com/ZhihuFrontier/status/1966419946885493098)). A separate roundup places Qwen3‑Next‑80B near DeepSeek V3.1 on an aggregate index at much lower token usage ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1966523300781428788)).

**Agents, evaluation fixes, and failure forensics**

- **SWE‑Bench fix, progress still real**: FAIR Codegen’s [@TacoCohen](https://twitter.com/TacoCohen/status/1966421688846778561) highlighted an issue allowing agents to peek at future commits, which SWE‑Bench promptly fixed. Preliminary re‑runs suggest most models aren’t heavily affected; FAIR found the bug only after scaling RL runs to “too‑good‑to‑be‑true” results. Recommendation: labs and OSS should re‑publish on the fixed benchmark and clearly annotate.
- **Live, taskful evals are hard**: LiveMCP‑101 introduces a real‑time agent framework/benchmark that stresses complex tasks beyond synthetic settings. Even frontier models underperform: GPT‑5 scores 39.02% on “hard” tasks; top models remain below 60% overall. The paper catalogs seven common failure modes (ignoring requirements, overconfident self‑solve, wrong tool choice, syntax/semantic/output parsing errors) ([overview](https://twitter.com/omarsar0/status/1966525731082768782), [results](https://twitter.com/omarsar0/status/1966525793586360384), [paper](https://twitter.com/omarsar0/status/1966525809302417436)).
- **Calibration over guessing**: OpenAI argues hallucinations persist because benchmarks reward confident guesses; fixes include not penalizing “I don’t know” and realigning leaderboards ([summary](https://twitter.com/TheTuringPost/status/1966638472854483129), [paper](https://twitter.com/TheTuringPost/status/1966638485282189600)). On AssistantBench, GPT‑5 shows higher precision and lower guess rates than o3 ([@PKirgis](https://twitter.com/PKirgis/status/1966547382033936577)). HAL is adding Docent to analyze agent logs rather than only end accuracy ([@sayashk](https://twitter.com/sayashk/status/1966550402129592738)).

**Tooling, infra, and libraries**

- **VS Code grows a model marketplace API**: The “Language Model Chat Provider” extension API is finalized; BYOK providers can be installed as extensions for more model choice. Also shipping are tutorials, videos, and an auto‑select model experience (e.g., Claude, GPT‑5/mini, Gemini) ([API thread](https://twitter.com/code/status/1966638511269794238), [Cerebras ext](https://twitter.com/code/status/1966638514100924846), [release](https://twitter.com/housecor/status/1966429828808352233), [notes](https://twitter.com/code/status/1966546512717946894)).
- **Transformers v5 + continuous batching**: HF teased a v5 modernization push (faster kernels, smarter defaults, cleanup) and quietly landed continuous batching to simplify evaluation/training loops (not chasing max‑throughput servers; focus is tinkering/toolbox) ([v5](https://twitter.com/art_zucker/status/1966470835558093226), [cont. batching](https://twitter.com/LucSGeorges/status/1966550465769775305)). Also, “new LLM releases now announced as PRs to Transformers” ([@lvwerra](https://twitter.com/lvwerra/status/1966451134727352326)).
- **Inference systems**: Meta’s vLLM disaggregated inference shows latency/throughput wins vs its internal stack; optimizations are being upstreamed ([@PyTorch](https://twitter.com/PyTorch/status/1966546293733437799)). A clear explainer on paged attention circulated ([link](https://twitter.com/novasarc01/status/1966413957679428054)).
- **AOT and regional compilation**: ZeroGPU added regional AOT compilation and sharing/loading precompiled graphs to accelerate bring‑up ([post](https://twitter.com/RisingSayak/status/1966447203381092675), [blog/docs](https://twitter.com/RisingSayak/status/1966447207688569028)).
- **Vision & retrieval in HF**: Microsoft’s Kosmos‑2.5 landed in Transformers with OCR+layout demo/notebook ([demo/docs](https://twitter.com/mervenoyann/status/1966487632659005667), [notebook](https://twitter.com/mervenoyann/status/1966488556831977672)). MetaCLIP2 multilingual models plus text‑to‑image search notebooks arrived as well ([announcement](https://twitter.com/mervenoyann/status/1966544046744011242), [tutorial](https://twitter.com/mervenoyann/status/1966544570436424074)).
- Also noted: Skypilot’s new GPU utilization dashboard ([link](https://twitter.com/skypilot_org/status/1966592871600890285)); and Elon’s aside that “AMD is now working pretty well for small to medium sized models” ([@elonmusk](https://twitter.com/elonmusk/status/1966412913662669082)).

**Frontier access, SDKs, and safety collaborations**

- **OpenAI platform**: GPT‑5 and gpt‑5‑mini rate limits were bumped substantially across tiers ([@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1966610846559134140)). A new “gpt‑5‑high‑new” target appeared in Codex‑CLI (“tuned to rely on built‑in reasoning defaults”), though details remain scant ([@mark_k](https://twitter.com/mark_k/status/1966521489529643169)). OpenAI’s focus on extended thinking continues: o1‑preview “seconds” to current models “hours” with web+browse+code, “much more runway ahead” ([@polynoamial](https://twitter.com/polynoamial/status/1966527147469598794), [@gdb](https://twitter.com/gdb/status/1966612991421423814)).
- **Anthropic**: The UK AISI and US CAISI have been identifying jailbreaks in Claude Opus 4/4.1, helping ship stronger safeguards ([announcement](https://twitter.com/AnthropicAI/status/1966599335560216770), [details](https://twitter.com/AnthropicAI/status/1966599337426681899), [AISI thread](https://twitter.com/alxndrdavies/status/1966614120566001801)). For builders, the Claude Code SDK (same harness as the CLI) is a recommended starting point for custom agents ([intro](https://twitter.com/alexalbert__/status/1966601430808088596), [docs](https://twitter.com/alexalbert__/status/1966601435153388019)).
- **Qwen Code**: v0.0.10/11 added sub‑agents, a Todo Write tool, “Welcome Back” project summaries, editing stability, better IDE/shell integration, improved memory/session management, and more ([release](https://twitter.com/Alibaba_Qwen/status/1966451235328008563), [preview](https://twitter.com/Alibaba_Qwen/status/1966451500340703418)).

**Vision models and leaderboards**

- **LMArena updates**: With >43k votes, Gemini 2.5 Flash Image (“nano‑banana”) continues to top both Image Edit and Text‑to‑Image charts; ByteDance Seedream 4 is now #2 on Image Edit and #5 on T2I ([leaderboard](https://twitter.com/lmarena_ai/status/1966562484506230922), [more](https://twitter.com/lmarena_ai/status/1966562486897029274)). A new “Seedream 4 High Res” variant supports 4096×4096 outputs and is live in Arena ([add](https://twitter.com/lmarena_ai/status/1966673628327801255), [try](https://twitter.com/lmarena_ai/status/1966673632069132770)).
- **Other vision drops**: Tencent’s HunyuanImage‑2.1 (2K T2I) is available via Anycoder/FAL for quick app prototyping ([post](https://twitter.com/_akhaliq/status/1966684003877917145), [app](https://twitter.com/_akhaliq/status/1966684046206906801)).

**Privacy-preserving pretraining**

- **VaultGemma**: Google Research released VaultGemma, a 1B‑parameter Gemma variant trained from scratch with differential privacy—claimed as the largest open model trained this way—plus new scaling‑law results for private LM training. Weights and report are available ([announcement](https://twitter.com/GoogleResearch/status/1966533086914421000), [summary](https://twitter.com/osanseviero/status/1966534013511672148), [model](https://twitter.com/osanseviero/status/1966534014791020869), [paper](https://twitter.com/osanseviero/status/1966534140439728485)).

**Top tweets (by engagement)**

- “How money works” flywheel satire around a hypothetical OpenAI–Oracle megadeal by [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1966553671866687689) (20.9k).
- Utah Gov. Spencer Cox on social media harms by [@bensiegel](https://twitter.com/bensiegel/status/1966510619479118073) (12.9k).
- Wikipedia finances scrutiny by [@nearcyan](https://twitter.com/nearcyan/status/1966601978319904877) (10.4k).
- AI leader archetypes satire by [@sergeykarayev](https://twitter.com/sergeykarayev/status/1966506136481481090) (9.0k).
- OpenAI platform rate‑limit boosts for GPT‑5/mini by [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1966610846559134140) (2.1k).
- Elon on AMD GPUs for small/medium models by [@elonmusk](https://twitter.com/elonmusk/status/1966412913662669082) (2.2k).
- Higgsfield growth stats and product velocity by [@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1966588786080706842) (2.9k).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Meta MobileLLM-R1 Release + Weekly LocalLLaMA Model/Dataset Roundup (Sep 12)

- [**Meta released MobileLLM-R1 on Hugging Face**](https://i.redd.it/huchm6bahrof1.png) ([Score: 412, Comments: 46](https://www.reddit.com/r/LocalLLaMA/comments/1nf7zhq/meta_released_mobilellmr1_on_hugging_face/)): **Meta published MobileLLM‑R1‑950M on Hugging Face ([model card](https://huggingface.co/facebook/MobileLLM-R1-950M)), a** `~950M`**parameter small LLM intended for efficient, on-device/mobile inference, with an accompanying interactive demo Space ([app](https://huggingface.co/spaces/akhaliq/MobileLLM-R1-950M)) reportedly built via the AnyCoder Space ([AnyCoder](https://huggingface.co/spaces/akhaliq/anycoder)). The post does not list benchmarks, but context emphasizes pushing inference accuracy at the low-parameter end and providing an open release suitable for lightweight deployment.** Commenters applaud work on small-model inference accuracy and appreciate that Meta is still releasing models openly, with some surprise about it being “fully open source.”
    - Emphasis on pushing inference accuracy at the small-parameter frontier: commenters highlight value in optimizing the "lower bounds" of limited-parameter models, where improvements in training, quantization, and decoding strategies can yield disproportionately large real-world gains for on-device and low-latency settings.
    - Benchmark skepticism: one user notes the model is still outperformed by `Qwen 0.6` (likely a ~0.6B-class Qwen variant) on common leaderboards, questioning novelty. This raises the need to evaluate not just raw accuracy but mobile-centric metrics (e.g., tokens/sec on CPU/NPU, peak RAM, model size after 4/8-bit quantization, and energy per token) and any R1-style reasoning gains if applicable.
    - Deployment interest: requests for a `GGUF` build suggest users want llama.cpp compatibility and fast quantization (e.g., Q4_K_M/Q8_0) for edge devices, enabling practical tests on laptops and phones without GPU, and facilitating apples-to-apples comparisons of throughput and memory footprint versus other sub-1B models.
- [**A list of models released or udpated last week on this sub, in case you missed any - (12 Sep)**](https://www.reddit.com/r/LocalLLaMA/comments/1neyaph/a_list_of_models_released_or_udpated_last_week_on/) ([Score: 273, Comments: 32](https://www.reddit.com/r/LocalLLaMA/comments/1neyaph/a_list_of_models_released_or_udpated_last_week_on/)): **Weekly roundup highlights: Qwen3‑Next‑80B‑A3B introduces a sparsely‑activated 80B MoE with ~3B params active per token (reported ~10× faster inference, 32k+ context) [HF](https://huggingface.co/collections/Qwen/qwen3-next-68c25fd6838e585db8eeea9d) [release](https://www.reddit.com/gallery/1nefmzr); MiniCPM4.1‑8B adds hybrid reasoning (/think vs /no_think) with long context [HF](https://huggingface.co/openbmb/MiniCPM4.1-8B); Jan‑v1‑2509 claims improved reasoning/creativity evals [HF](https://huggingface.co/janhq/Jan-v1-2509); and PyDevMini‑1 (4B) claims GPT‑4‑level Python/Web‑Dev performance at 1/400th the size [HF](https://huggingface.co/bralynn/pydevmini1). Speech/TTS: Qwen3‑ASR (API‑only, multilingual EN/CN + 9) [demo](https://huggingface.co/spaces/Qwen/Qwen3-ASR-Demo) and IndexTTS‑2.0 (expressive, duration‑controlled zero‑shot TTS) [repo](https://github.com/index-tts/index-tts). Reasoning/MoE and research: Aquif‑3 series (incl. 17B a2.8B GGUF) [HF](https://huggingface.co/mradermacher/aquif-3-moe-17b-a2.8b-GGUF), ROMA reports wins over closed platforms on SEAL‑0/FRAMES [GitHub](https://github.com/sentient-agi/ROMA), Baidu’s Ernie X1.1 targets frontier Chinese capability [post](https://www.reddit.com/r/LocalLLaMA/comments/1ndjoek/new_ernie_x11_what_may_be_the_best_chinese_model/); datasets include FinePDFs (3T tokens; 0.5B+ PDFs) [HF](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) and LongPage (300 novels with reasoning traces) [HF](https://huggingface.co/datasets/Pageshift-Entertainment/LongPage).** Comments request llama.cpp support for Qwen Next and flag contemporaneous releases: Kwai‑Klear’s Klear‑46B‑A2.5B‑Instruct [link](https://huggingface.co/Kwai-Klear/Klear-46B-A2.5B-Instruct) and inclusionAI’s Ring‑mini‑2.0 [link](https://huggingface.co/inclusionAI/Ring-mini-2.0).
    - Interest in llama.cpp support for **Qwen** indicates demand for GGUF quantization and lightweight CPU/GPU inference of Qwen-family models via llama.cpp’s kernels (e.g., cuBLAS/Metal/Vulkan). Integration typically hinges on tokenizer/chat template compatibility (Qwen often uses ChatML) and rotary/pos-embed variants; tracking llama.cpp PRs would clarify when full Qwen parity lands ([llama.cpp](https://github.com/ggerganov/llama.cpp), [Qwen HF](https://huggingface.co/Qwen)).
    - A commenter flags the release of [Kwai-Klear/Klear-46B-A2.5B-Instruct](https://huggingface.co/Kwai-Klear/Klear-46B-A2.5B-Instruct) “exactly 7 days ago.” The naming suggests a Mixture-of-Experts style model with ~`46B` total parameters and ~`2.5B` active per token (typical “A2.5B” convention), targeting instruction tuning; if accurate, it could offer latency closer to a small dense model while retaining higher capacity—benchmarks vs Mixtral-style MoEs would be valuable.
    - Additional mention of [inclusionAI/Ring-mini-2.0](https://huggingface.co/inclusionAI/Ring-mini-2.0) highlights an updated compact instruct model. For technical evaluation, readers would want perplexity and downstream benchmarks (e.g., MMLU, GSM8K) and quantization availability (GGUF/int8) to assess suitability for edge deployment within the `~1–3B` class.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Seedream/Seedance 4.0 Image Model Releases and Benchmarks

- [**Seedance 4.0 is so impressive and scary at the same time... (all these images are not real and don't exist btw)**](https://www.reddit.com/gallery/1ned5ul) ([Score: 374, Comments: 77](https://www.reddit.com/r/singularity/comments/1ned5ul/seedance_40_is_so_impressive_and_scary_at_the/)): **Post showcases “Seedance** `4.0`**” purported image-generation results that are claimed to be entirely synthetic and photorealistic ("all these images are not real"). No technical artifacts are provided—no model/architecture details, training data, safety or watermarking scheme, or quantitative evaluations (e.g., FID, precision/recall)—so fidelity, robustness to detection, and provenance guarantees cannot be assessed from the post alone.** Top comments voice skepticism about post-release astroturfing/"organic" marketing around new models; otherwise there’s minimal technical discussion.
    - Multiple commenters position **Seedance 4.0** as the current top text-to-image model, with **Nano Banana** cited as a close second; others are perceived to lag notably in prompt adherence and photorealism. No quantitative benchmarks were provided, but the consensus emphasizes superior baseline quality and consistency for Seedance across similar prompts.
    - A technical trade-off is highlighted: Seedance 4.0 tends to produce highly consistent outputs for similar prompts (lower variance), whereas **Nano Banana** yields greater diversity/variance in generations. This suggests different sampling/regularization behaviors (e.g., tighter prompt-to-image mapping or stronger mode preference in Seedance), which could favor Seedance for reproducibility while making Nano Banana better for exploratory ideation.
- [**Seedream 4.0 is the new leading image model across both the Artificial Analysis Text to Image and Image Editing Arena, surpassing Google's Gemini 2.5 Flash (Nano-Banana), across both!**](https://www.reddit.com/gallery/1necl7d) ([Score: 242, Comments: 86](https://www.reddit.com/r/Bard/comments/1necl7d/seedream_40_is_the_new_leading_image_model_across/)): **Post claims that Seedream 4.0 is now ranked top-1 on the Artificial Analysis (AA) Text-to-Image and Image Editing Arenas, surpassing Google’s Gemini 2.5 Flash (the Arena entry referred to as “Nano-Banana”) across both tasks. AA leaderboards are ELO-style, pairwise preference battles, so this implies Seedream 4.0 leads in head-to-head prompt-following generation and localized editing quality under AA’s crowd/evaluator setup ([Artificial Analysis](https://artificialanalysis.ai/), [Gemini models overview](https://ai.google.dev/gemini-api/docs/models/gemini)).** Commenters note that holding the #1 spot in both generation and editing simultaneously is uncommon and impressive; there’s also community speculation/hope that an open-weights model from Chinese labs could soon overtake closed systems in at least some domains.
    - Seedream 4.0 topping both the Artificial Analysis Text-to-Image and Image Editing arenas—surpassing **Google Gemini 2.5 Flash (Nano-Banana)**—signals strong cross-task generalization and instruction-following. Editing leaderboards stress localized edits, identity preservation, and low over/under-edit rates; being `#1 across both` suggests robust control as well as generative quality. See the arenas on [Artificial Analysis](https://artificialanalysis.ai/) for pairwise results.
    - Debate on benchmarks vs subjective testing: arena rankings are typically derived from pairwise human preference with ELO-style scoring, which can diverge from small-sample personal tests. As one user notes, *“well it sucks in my testing, benchmarks/leaderboards aren’t everything,”* highlighting that leaderboard wins reflect aggregate preference, not every prompt distribution; reproducible eval with fixed seeds and public prompt sets can help reconcile discrepancies.
    - Safety/moderation trade-offs raised: heavier filtering pipelines (classifier cascades, prompt sanitization, rejection sampling) can increase refusal rates and degrade edit success on benign edge cases. Tightly moderated stacks (e.g., some Google deployments) may reduce NSFW/abuse risk but also harm instruction-following and throughput/latency, which can impact arena win-rates in instruction-heavy image editing.
- [**1GIRL QWEN v2.0 released!**](https://www.reddit.com/gallery/1ne0mck) ([Score: 353, Comments: 49](https://www.reddit.com/r/StableDiffusion/comments/1ne0mck/1girl_qwen_v20_released/)): **Release of 1GIRL QWEN v2.0, a LoRA fine-tune targeting the Qwen-Image pipeline, claiming improved realism for single-girl renders. Download via Civitai: https://civitai.com/models/1923241?modelVersionId=2203783; preview: https://preview.redd.it/mhrk7biqbhof1.png?width=763&format=png&auto=webp&s=b38072a5a786614d2bc53677dfcc8429544adfb7. The post provides no training details (e.g., rank, dataset, steps) or benchmarks; “one of the most realistic” is a qualitative claim without quantitative evals or comparison baselines.** Top comments question promotional framing ("yet another instagirl ad") and note apparent vote manipulation before stabilization; one asks if the model is “uncensored,” implying interest in safety filter/NSFW gating and whether the LoRA bypasses base-model content controls.
    - A commenter asks for concrete LoRA training details for this release, planning to train locally on an **RTX 4080 Super (16 GB VRAM) with 32 GB RAM**. They note prior success fine-tuning **SDXL** and are switching to **Qwen** citing its *faithfulness to prompt details*, seeking specifics on the training pipeline and settings to replicate comparable fidelity.
    - Another user asks whether the release is **uncensored** (i.e., NSFW-enabled/no safety filters). This impacts applicability for local deployments and parity with community LoRAs versus filtered or "instruct"-style checkpoints that may suppress certain outputs.
    - One comment flags a visible anatomy/proportion artifact ("second picture thigh larger than torso"), implying the model or LoRA may still exhibit common generative failures in body proportions. This points to potential dataset bias or insufficient constraint during fine-tuning affecting structural consistency in outputs.
- [**Control**](https://v.redd.it/rzwnnwszdhof1) ([Score: 248, Comments: 47](https://www.reddit.com/r/StableDiffusion/comments/1ne1ouv/control/)): **A demo showcases a control pipeline combining “InfiniteTalk” (speech-driven facial/lip-sync animation) with “UniAnimate” (controllable video animation for body/hands) to perform dubbing in a video-to-video workflow. Facial realism is highlighted as the strongest aspect, but exact frame/pose parity with the source is not maintained—the output exhibits slight motion drift, indicating temporal consistency and movement-locking limitations in the current setup.** Commenters praise the facial performance and ask for implementation details on fusing UniAnimate with InfiniteTalk while preserving exact movements; one suggests scrutinizing hand consistency (e.g., “follow the rings on her right hand”) to detect subtle control or artifact issues.
    - Several users are trying to combine **Unianimate** with **Infinite Talk** for video-to-video dubbing, but report that Infinite Talk’s output drifts from the input motion (i.e., doesn’t preserve exact pose/gesture timing). The core technical issue raised is 1:1 motion/temporal lock—maintaining identical per-frame movement while replacing speech—implying a need for strict frame-rate parity, deterministic seeds, and motion/keypoint control across the pipeline to avoid resampling or retiming artifacts.
    - Multiple requests for a detailed workflow indicate missing implementation specifics (e.g., capture FPS, motion control signals, seed/temperature settings, how face/hand control is applied, and where audio-driven lipsync is injected in the graph). Without these, replicability is limited and viewers can’t assess whether the pipeline uses pose control (e.g., keypoints/optical flow) versus post-process retiming to align lip motions.
    - A visual audit cue is suggested: “follow the rings on her right hand,” implying hand jewelry as an unintentional motion-tracking marker. This is a practical technique to detect temporal inconsistencies or compositing—if rings exhibit unnatural jitter/warping or timing offset relative to body pose, it hints at imperfect motion preservation or stabilization in the generation pipeline.
- [**Lol. I asked ChatGPT to generate an image of the boyfriend it thinks I want and the boyfriend it thinks I need**](https://i.redd.it/gszu1sdociof1.png) ([Score: 2532, Comments: 651](https://www.reddit.com/r/ChatGPT/comments/1ne4mkc/lol_i_asked_chatgpt_to_generate_an_image_of_the/)): **OP used ChatGPT’s image generation to create a two-panel “boyfriend I want vs boyfriend I need” image. One panel reportedly shows a man with an “AI safety” book, indicating a likely hallucinated text element and/or alignment-biased content insertion—an example of how generative models can misinterpret abstract prompts and inject safety-themed or on-trend concepts. While non-technical, it highlights model priors and text-in-image artifacts common in systems like DALL·E 3.** Comments note the odd inclusion of an “AI safety book” and suggest GPT “misunderstood something,” while OP says the result isn’t wrong—reflecting mixed reactions to the model’s interpretation rather than its rendering quality.
    - Several users note the model inserting unexpected, legible text elements (e.g., an “AI safety book”) into the generated image, suggesting safety-tuning priors can leak into content selection and that the image model has relatively strong text rendering compared to earlier diffusion models that often garbled words. See examples shared in-thread: https://preview.redd.it/3z4sje4t8jof1.png?width=1536&format=png&auto=webp&s=027ee8ad4f9b77efa58d4750ad3be7d5f5d18ec6 and https://preview.redd.it/v6cyf3q3viof1.jpeg?width=1176&format=pjpg&auto=webp&s=802e364f3a14b0f3cf2fd7fd2e68bd0f742e9319.
    - Comments imply the prompt was interpreted via common internet tropes (“the boyfriend you want vs the boyfriend you need”), producing archetypal contrasts rather than personalized outputs—highlighting that, without explicit attributes or constraints, prompt following defaults to generic priors and can feel like a misinterpretation or a “roast.” This reflects typical behavior of safety-aligned, instruction-following image models that prioritize safe, broadly acceptable compositions over user-specific nuance.

### 2. UK Government AI Adoption Coverage

- [**AI is quietly taking over the British government**](https://i.redd.it/7b5t3z8bbiof1.png) ([Score: 3012, Comments: 171](https://www.reddit.com/r/OpenAI/comments/1ne4jca/ai_is_quietly_taking_over_the_british_government/)): **The post’s image (https://i.redd.it/7b5t3z8bbiof1.png) appears to insinuate that UK House of Commons/government text is AI-generated, but it provides no technical evidence (no model/version, deployment details, usage metrics, or sourcing). There are no benchmarks or audits—just a screenshot-level claim—so the most plausible technical interpretation is routine use of LLMs (e.g., ChatGPT/Copilot/Grammarly) for proofreading or drafting assistance by staff rather than any system-level automation or policy change.** Top comments push back that the title is sensational; they argue it’s common for professionals to use AI for proofreading and that this doesn’t equate to AI “taking over.” Another comment mocks the claim, implying the presented “verbiage analysis” is unconvincing and not evidence-based.
    - Multiple commenters note official, time-bounded adoption: the UK government received a **free Microsoft 365 Copilot trial** from `Oct–Dec 2024` ([The Register](https://www.theregister.com/2025/09/04/m365_copilot_uk_government/)), and in `Jan 2025` the Labour government published a blueprint to **scale AI across departments** ([gov.uk](http://gov.uk/)). This suggests any spike in "AI-like" phrasing aligns with sanctioned M365 Copilot use (Word/Outlook/Teams) rather than covert takeover. The timing undermines the “quietly” claim and frames it as an official, enterprise rollout.
    - Methodology critique: attributing text to ChatGPT via "crucial verbiage" or stylistic markers is unreliable—AI text detection has high false-positive/negative rates and is easily gamed. One comment observes the signal correlates more with when **Labour took office** than with ChatGPT availability, implying a communications-style shift as a confounder. A more rigorous approach would control for administration change (e.g., difference-in-differences across departments and pre/post periods) and validate against ground-truth authorship.
    - Practitioners emphasize assistive usage—civil servants likely use AI for proofreading/summarization and "linguistic verification" rather than wholesale content generation. In an M365 Copilot context, that maps to rewrite/summarize/proof features embedded in Word/Outlook, which augment throughput without "taking over" roles; measuring adoption by presence of generic phrasing alone risks overstating automation.

### 3. ChatGPT Ads, Gemini 3 Release Delay, and Feature Gap Debate

- [**Enjoy ChatGPT while it lasts…. the ads are coming**](https://i.redd.it/vx7mk59mgjof1.jpeg) ([Score: 2375, Comments: 163](https://www.reddit.com/r/OpenAI/comments/1ne90w5/enjoy_chatgpt_while_it_lasts_the_ads_are_coming/)): **OP argues that consumer LLM assistants (ChatGPT/OpenAI, Perplexity, Anthropic) will inevitably monetize by embedding ads into responses, risking covert promotional steering and surveillance-style targeting within the chat UX. Technical concern centers on contamination of model outputs via sponsored prompts/formatting, tier-based gating (free vs paid), and resultant erosion of trust/accuracy in assistant recommendations. The thread frames a conflict-of-interest risk where ranking/generation becomes ad-influenced rather than relevance/faithfulness-driven.** Top comments debate acceptability of ads only on free tiers vs unacceptable for Plus/Pro; suggest subscriptions or other offsets instead of ads due to trust/accuracy headwinds; warn that influence may be organic/subtle rather than explicit ad units, making it harder to detect.
    - Hidden “organic” ad steering is technically feasible via alignment data and system-level policies: a provider could bias GPT-4o/ChatGPT recommendations by mixing advertiser-favored samples into RLHF/instruction-tuning, or by adding retrieval/ranking priors that prefer sponsored entities, leading to subtle product slant without explicit ad labels. This is analogous to search ad blending where paid results are ranked alongside organic; with LLMs, the bias manifests in generated prose and tool-use choices, making disclosure and reproducibility harder to audit.
    - Several users flag data-contamination risks: if open-source models train on web corpora increasingly polluted by ad-influenced LLM outputs, bias amplifies over time. This mirrors model self-consumption failures documented in “Self-Consuming Generative Models Go MAD” (Shumailov et al., 2023) where training on model-generated data induces distribution shift and degradation; ads would act as a targeted poisoning signal that propagates into future checkpoints (see https://arxiv.org/abs/2305.17493).
    - Evidence of link-level attribution/tracking: ChatGPT-shared URLs can include affiliate/UTM-style parameters (e.g., `utm_source`, `ref`, or partner IDs), enabling downstream sites to attribute traffic and enabling the model provider to run CTR/A/B experiments. While not an ad per se, this instrumentation creates a measurement channel that could be repurposed for sponsored ranking or revenue share and folded back into retrieval/ranking training via click logs.
- [**Why haven't all the other companies (Google, OpenAI, Deepseek, Qwen, Kimi and others) added this before? It's literally the most obvious and most needed thing 🤔**](https://i.redd.it/g9sb9rvariof1.jpeg) ([Score: 295, Comments: 51](https://www.reddit.com/r/singularity/comments/1ne60nk/why_havent_all_the_other_companies_google_openai/)): **OP shares an image implying a “new” chat feature for uploading/reading files (esp. PDFs) directly inside an LLM UI and wonders why others haven’t shipped it. Multiple comments point out this capability has existed in ChatGPT since 2023 via Code Interpreter/Advanced Data Analysis—allowing users to attach PDFs/CSVs, run Python over them, and query document contents—so the novelty is likely UI polish rather than core functionality. See OpenAI’s earlier releases: [ChatGPT Plugins incl. Code Interpreter (Mar 2023)](https://openai.com/index/chatgpt-plugins/) and the [Advanced Data Analysis help doc](https://help.openai.com/en/articles/8554405-advanced-data-analysis).** Commenters argue the feature isn’t new (“who’s gonna tell him”), and note that while ChatGPT’s implementation works, the results on PDFs can be mediocre and the UI less refined compared to the screenshot.
    - Multiple commenters note this isn’t new: ChatGPT has supported file upload and document/PDF analysis since `2023` via **Code Interpreter / Advanced Data Analysis (ADA)**, handling non‑visual files well. However, results on complex PDFs are described as only “mid,” with weaker formatting fidelity/table extraction and a more basic UI rendering compared to native viewers. Ref: OpenAI ADA docs — https://help.openai.com/en/articles/8554397-advanced-data-analysis.
    - Feature parity exists across other stacks: **Google Gemini**, **Microsoft Copilot**, and **DeepSeek** already allow uploading files for analysis/summarization, so the capability isn’t novel to one vendor. Gemini’s API explicitly supports prompting with uploaded files (including PDFs) for multimodal processing — https://ai.google.dev/gemini-api/docs/prompting_with_files.
- [**ChatGPT may have saved my life**](https://www.reddit.com/r/ChatGPT/comments/1ne1ccl/chatgpt_may_have_saved_my_life/) ([Score: 438, Comments: 55](https://www.reddit.com/r/ChatGPT/comments/1ne1ccl/chatgpt_may_have_saved_my_life/)): **OP reports persistent abdominal pain; ChatGPT elicited classic appendicitis triage features—**`right lower quadrant` **pain and** `rebound tenderness`**—and advised ER evaluation, where near-rupture appendicitis was apparently confirmed. The interaction mirrors simple clinical decision aids (e.g., the [Alvarado score](https://en.wikipedia.org/wiki/Alvarado_score)) and bedside signs like [McBurney’s point](https://en.wikipedia.org/wiki/McBurney%27s_point) and [rebound tenderness](https://en.wikipedia.org/wiki/Rebound_tenderness), illustrating LLMs’ ability to surface pertinent positives/negatives for urgent care despite not being clinicians.** Top comments provide corroborating anecdotes: ChatGPT supplied reasonable differentials later aligned with clinician diagnoses and served as an explanatory aid during rehab; others argue its public-health benefits (triage and education) are underweighted relative to rare harmful uses. Additional anecdotes cite accurate preliminary identification of conditions in pets and children prior to formal diagnosis.
    - Users report leveraging ChatGPT for differential diagnosis and triage-style reasoning: when appendicitis was suspected, it produced a ranked list of alternatives, one of which matched the hospital’s final diagnosis; another user describes stepwise guidance to check gallbladder pain and to rule out emergent issues. This highlights utility as a patient-side decision-support tool that structures symptom review and next-step heuristics while deferring definitive diagnosis to clinicians.
    - Several accounts emphasize evidence-oriented education and care planning: ChatGPT provided detailed explanations of conditions, probable recovery timelines, and curated stage-specific gastritis diets, including rationale on which foods are “gastritis safe,” and guidance toward nutrient-dense options during reduced intake. One user notes it could surface and explain studies and mechanistic reasons behind recommendations, aiding self-management prior to a `~6 months` in-person appointment.
    - Failure modes and safety practices are called out: despite being “rarely incorrect” on dietary safety, users still “caught it making false claims and assumptions,” reinforcing the need to cross-check and treat outputs as advisory. Telemedicine later confirmed a suspected gastritis diagnosis, underscoring that ChatGPT can be a high-recall assistant for narrowing possibilities and education, but requires external validation and should not replace clinical testing or medical judgment.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1: Fresh Models Flex Muscles in Arenas**

- **Qwen3 80B Crushes Sparsity Records**: **Qwen3 80B** boasts **79.7B parameters** with only **3.87B active** due to **1:51.2 sparsity** in its MoE, enabling efficient computation while maintaining high performance, as detailed in [this X post](https://x.com/AutismCapital/status/1965845243053617436). Members expressed optimism about its abilities, especially when compared to **GPT-5**, with a December 2024 knowledge cutoff and [decent initial performance](https://discord.com/channels/1340554757349179412/1343296395620126911/1415768520301609013).
- **Palmyra-Mini Packs Reasoning Punch**: The **Palmyra-mini family** includes a base model and variants excelling in math tasks like **GSM8K 82.9%** and **AMC23 92.5%**, with one achieving top scores on **AIME24**, **GPQA**, and **MATH500**, available on [Hugging Face](https://xcancel.com/samjulien/status/1966249697661825093). These compact open-source models from Writer focus on reasoning, sparking discussions on their potential for technical applications.
- **FluentlyQwen3 Drops Universal LLMs**: Project Fluently released **FluentlyQwen3-1.7B** and **4B** models, merged after additional training under Apache-2.0 license, maximizing potential for diverse tasks as seen on [Hugging Face](https://huggingface.co/fluently/FluentlyQwen3-4B). Users highlighted their efficiency on lower-end hardware, with links to [FluentlyQwen3-1.7B](https://huggingface.co/fluently/FluentlyQwen3-1.7B) for quick deployment.

**Theme 2: Throughput Wars Heat Up Hardware**

- **GPT-OSS 120B Revs TPS Debates**: Members debated **GPT-OSS 120B** achieving **30 TPS** on a **4090** with **64GB RAM**, while others capped at **10 TPS**, prompting tweaks in *llama.cpp* like disabling top-k for better performance. Optimizations like **MXFP4 quantization** and custom kernels yielded speed gains, with benchmarks in [this Hugging Face post](https://xcancel.com/reach_vb/status/1966134598682767507).
- **DeepSeek Drags to Hour-Long Snails**: **DeepSeek** faced reports of extreme slowness, with code generation taking **1 hour 20 minutes**, speculated to stem from **CCP-mandated Huawei chips** impacting performance. Community contrasted this with open-source affordability at **1/5 the price** of closed alternatives, emphasizing privacy benefits over lagging search capabilities.
- **Gemma3 Builds from Scratch on A6000**: A user trained **Gemma3 270M** from scratch on **TinyStories** for **10 hours** using an **A6000 GPU**, logging with Weights and Biases and judging via **Claude Opus 4.1**, shared on [GitHub](https://github.com/di37/gemma3-270M-tinystories-pytorch) and [Hugging Face](https://huggingface.co/disham993/gemma3-270m-tiny-stories).

**Theme 3: Training Tricks Tackle Data Dilemmas**

- **Two-Stage Curriculum Slashes Compute Waste**: A two-stage training ranked datasets by difficulty, dropping average loss from **2.5** to **0.8** after refining **stage1** with unambiguous labels, improving signal focus as discussed in Unsloth AI. This method reduces wasted compute on easy examples, drawing from an upcoming paper on synthetic data tainting closed LLMs like **Grok** and **Gemini** at [arxiv.org](https://arxiv.org/html/2509.05276v1).
- **Synthetic Data Poisons Closed-Source Giants**: All closed LLMs suffer *zero LTF factor* from synthetic data training, requiring re-biasing and rebuilding latent thinking, as per a paper claiming performance hits in **RLHF** and instruct tuning. Members debated fixes like phased pretraining from **TinyStories** to **FineWeb** for **400M models**, emphasizing inductive bias over long contexts.
- **Fluid Nets Flow with Navier-Stokes**: A paper explored Turing-complete neural nets via **Navier-Stokes equations** for fluid dynamics computing, sparking debates on *mortality and unreproducibility* versus efficiency, linked at [arxiv.org](https://arxiv.org/abs/2507.07696). Parallels drawn to running *Doom* on gut bacteria in [this video](https://www.youtube.com/watch?v=8DnoOOgYxck) highlighted analog compute trade-offs.

**Theme 4: Deployment Demons Dog Engineers**

- **Docker Crashes H100 Party**: Docker images working on **3090/4090** failed with CUDA errors on **H100**, resolved by updating incompatible **NVIDIA drivers** via [data center drivers](https://www.nvidia.com/en-us/drivers/data-center-drivers/). Users reported similar woes with **vLLM** switching to uv pip, breaking Torch Nightly and forcing reverts to **v0.10.1**.
- **IRIS Install Simplifies ROCm Chaos**: **IRIS** installation streamlined to `pip install git+https://github.com/ROCm/iris.git` requiring **ROCm + Torch + Triton + TorchDistributed**, demonstrated in [this video](https://cdn.discordapp.com/attachments/1359640791525490768/1415976909535318037/iris-install.mov?ex=68c5d382&is=68c48202&hm=6a0a4b3c9e86c36fdfd1f189fe044dc5d4c4cc59bcd8bbdaab24e61c8453541b&). This aids AMD competitions, contrasting NVIDIA's **215 B200 GPUs** for the Oct 24 SF hackathon via [compute form](https://forms.gle/wYvXE99bvdRiQD6aA).
- **PSU Transients Trip GPU Stability**: Calculations for PSU wattage factored **CPU**, **GPU**, and **50% overhead** to avoid transients causing crashes, especially on **30-series cards**, referenced in [Teknium1's tweet](https://x.com/Teknium1/status/1966338983572725979). Users fixed "dead" secondary GPUs by cleaning PCI-E connectors, suggesting power issues over hardware failure.

**Theme 5: Tools Twist Creative and Coding Flows**

- **Kimi K2 Reigns in Creative Brainstorms**: **Kimi K2** topped charts for creative writing alongside **GPT-5 Medium** and **Qwen3-Max**, with users joking it trained on [Archive of Our Own](https://archiveofourown.org/) for immersive outputs. Integrations like **Augment Code** with **Groq** outperformed **Gemini** in coding, praised for token efficiency at **$1/m in** and **$3/m out**.
- **Cursor Pricing Sparks Ultra Upgrades**: **Cursor** pricing changes dropped usage from a month to **under 4 days**, but Ultra tier offers **$400 API access** from providers, easing frustrations over Auto limits. Background agents parsed edits with strict tagging, drawing comparisons to **Claude's Agents** for task execution.
- **DSPy Sections Defy Exact Counts**: DSPy struggled to generate exactly **12 sections** in lesson plans, often producing **13-15** even with **GPT-5**, fixed by first creating titles then fleshing out. Modaic launched as a DSPy-inspired hub with SDK on [PyPI](https://pypi.org/project/modaic/) for building and optimizing declarative AI programs.


---

# Discord: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-OSS 120B Sparks Throughput Debate**: Members debated the achievable throughput for **GPT-OSS 120B**, with some claiming **30 tokens per second (TPS)** on a **4090** with **64GB RAM**, while others struggled to exceed **10 TPS**, leading to discussions about quantization and build configurations.
   - Experimentation and tweaks in *llama.cpp* settings, such as disabling `top-k` and optimizing build configurations, are suggested for improved performance.
- **Telemetry Collection Raises Eyebrows**: Members discovered a telemetry script in **Qwen Code** models pointing to an **Alibaba server** without prior notification.
   - This discovery sparked a discussion about data privacy and control, with some members expressing discomfort about their code being potentially transmitted for training purposes, but *mostly a joke*.
- **Two-Stage Training Cuts Training Time**: A member described using a two-stage training curriculum to rank a *real* dataset by difficulty, based on loss from a tailored **stage1** dataset with unambiguous labels.
   - This approach aims to improve training signal and reduce wasted compute by focusing on more difficult examples, with the average difficulty of the *real* dataset dropping from **2.5** to **0.8** after refining **stage1**.
- **Docker Woes Plague H100 Deployments**: A user reported **CUDA errors** when running a Docker image (that worked on 3090/4090 GPUs) on an H100 GPU, even after rebooting and with seemingly compatible CUDA and Torch versions.
   - It was determined that the **NVIDIA driver version** installed in the Docker image was incompatible with the H100, requiring a driver update to resolve the issue; [NVIDIA Data Center Drivers](https://www.nvidia.com/en-us/drivers/data-center-drivers/).
- **Synthetic Data Taints Closed-Source LLMs**: A member shared a finding from an upcoming paper ([https://arxiv.org/html/2509.05276v1](https://arxiv.org/html/2509.05276v1)) suggesting that all **closed-source LLMs** (Grok, Gemini, GPT, etc.) are trained with **synthetic data** leading to a *zero LTF factor* and inability to humanize text.
   - They claimed models trained with **RLHF**, **synthetic data**, or **instruct tuning** will likely suffer performance hits due to needing re-biasing, latent thinking rebuilding, and relearning speaking patterns.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Finance Goes Mobile**: **Perplexity Finance** is now available on [iOS & Android](https://www.perplexity.ai/changelog/what-we-shipped-september-12th), bringing financial insights to mobile devices.
   - Users can now enjoy **hotel loyalty support** when making [bookings](https://www.perplexity.ai/changelog/what-we-shipped-september-12th) through Perplexity.
- **Comet Browser's Data Collection Sparks Debate**: Users discussed **Comet's** data collection, with logs showing that **Comet** sends search suggestions as POST requests to Perplexity servers, even when DuckDuckGo is the search engine, sparking concern that **Comet** is more intrusive than Chrome.
   - Claims arose that the CEO admitted it's designed to track and sell data, although the CEO denied on [X](https://x.com/AravSrinivas/status/1915533071291474139).
- **Users Leaking Prompts from Top AI Apps!**: Users confirmed that the prompts of top AI applications have been leaked and are available on GitHub.
   - One user joked to *just don't click here and you are safe* with a warning about clicking dangerous image links, to which another responded, *Is already there on GitHub LOL*.
- **Referral Frenzy Fuels Feud**: Multiple users shared [Perplexity AI Pro referral codes](https://perplexity.ai/pro?referral_code=N6VN4M13), including [this link](https://perplexity.ai/pro?referral_code=APLKGW40).
   - Users also shared [browser claim links](https://perplexity.ai/browser/claim/ALZQ0LYQGU), such as [this one](https://perplexity.ai/browser/claim/BSDJ1KBATC).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Qwen3 80B Enters the Arena!**: The new **Qwen3 80B** model has arrived in the arena, with a December 2024 knowledge cutoff and [showing decent initial performance](https://discord.com/channels/1340554757349179412/1343296395620126911/1415768520301609013).
   - Members expressed optimism about its abilities, especially when compared to **GPT-5**.
- **Seedream 4's Image Quality Sparks Debate**: Initial results show that **Seedream 4** is generating *trash results* on LM Arena compared to its predecessor, **Seedream-3**, as illustrated in [uploaded examples](https://discord.com/channels/1340554757827461211/1343296395620126911/1416080067271987220).
   - Conversely, some users report **improved image quality** with **Seedream 4** on the Doubao platform, though access is currently limited to new Chinese users.
- **Gemini 3 Remains MIA, Fuels Speculation**: The community is eagerly awaiting the arrival of **Gemini 3**, **GLM5**, and **DeepSeek r2**, noting Google's current lag in text generation compared to both closed and open source initiatives.
   - Polymarket estimates only a **42%** chance of release by Halloween, suggesting a more realistic launch timeframe in late October or early November.
- **DeepSeek's Performance Takes a Dive?**: Users have reported extreme slowness with **DeepSeek**, with one instance of code generation reportedly taking **1 hour and 20 minutes** to complete.
   - Speculation suggests this may be due to the **CCP** mandating the use of **Huawei chips**, which could be negatively impacting overall performance.
- **Open Source AI Champions Affordability and Privacy**: The discussion highlighted that **open source AI** is significantly more affordable (1/5 of the price) and offers greater privacy compared to closed-source alternatives like OpenAI and Google.
   - While *American models* may command higher prices due to superior performance, *Chinese models* like **Qwen** excel in e-commerce applications but *lag in search* capabilities, embodying a socialist approach to AI development.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Inference Credits Cause Mass Panic**: Users reported errors with **Hugging Face's Inference Providers** exceeding monthly credits despite credits availability.
   - One member jokingly suggested to *fix yo spending* as the error may be related to their usage, rather than the platform itself.
- **SmolvLM2 Shakes Up Video LMs**: Members shared **smolvlm2** ([Hugging Face blog](https://huggingface.co/blog/smolvlm2), [related collection](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7)), a video LM designed to run efficiently on lower-end hardware.
   - This model is well-suited for use on lower-end hardware.
- **Kaggle Gives Away GPU Hours**: A member pointed out that [Kaggle](https://www.kaggle.com/) offers **30 hours of GPU time each week** as an alternative for fine-tuning.
   - A member suggested using **PEFT/LoRA** to run fine-tuning on a **Tesla T4** within Colab.
- **Fluently Project Dumps LLMs**: The Project Fluently team released new universal LLM models based on **Qwen3 1.7B** and **4B**, which are [available on Hugging Face](https://huggingface.co/fluently/FluentlyQwen3-4B) under the Apache-2.0 license.
   - The models were carefully merged after additional training to maximize their potential, including [FluentlyQwen3-1.7B](https://huggingface.co/fluently/FluentlyQwen3-1.7B).



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor turns into a Smart Resume Machine**: A user found a novel use of **Cursor** as a smart resume and cover letter generator, with the tool now acting as a [resume machine](https://discord.com/channels/1074847526655643750/1074847527708393565/1415783677820010526).
   - This development sparked lighthearted banter, including jokes about AI domination and assurances of past friendly interactions with the AI.
- **Cursor Pricing sparks community uproar**: Users voiced discontent over recent **Cursor pricing** changes, with one user's usage dropping from nearly a month to under four days.
   - Despite the cost concerns, one user upgraded to **Ultra**, citing access to approximately **$400** worth of **API usage** from various providers, which is an improvement over being frustrated with **Auto**.
- **Background Agents vs. Claude's Agents**: A user questioned the similarity between **Cursor's background Agents** and **Claude's Agents**, particularly after an Agentics.org event described agents as specialized services executing specific tasks.
   - Another user detailed **Cursor's parsing of new edits** and its strict tagging structure with cross-connected tags, which enables change tracking and relation display in the left panel.
- **Netlify Account Mishap and Cursor**: A user initially reported that **Cursor deleted their Netlify account** following a Netlify project deployment, which turned out to be unrelated as there was no integration.
   - The user plans to investigate further by examining logs, confirming that there was no direct deletion command issued by Cursor.
- **Cursor App struggles with Unauthorized Errors**: A user reported experiencing *unauthorized errors* within the **Cursor app**, even after proper repository setup, illustrated by [this screenshot](https://cdn.discordapp.com/attachments/1367213641027551352/1416032865875005570/CleanShot_2025-09-12_at_20.07.352x.png?ex=68c6079f&is=68c4b61f&hm=f7097b440d30005da1b1a49f82fb7ce4632f9a889eed430792f772921820b6f8&).
   - A member suggested re-adding the bot from the repository, pointing to [this thread](https://forum.cursor.com/t/background-agent-docker-in-docker/104112/1) about *background agent docker issues*.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 Powers Creative Writing**: Members find **Kimi K2**, **GPT-5 (Medium)**, and **Qwen3-Max** to be the top models for creative writing and brainstorming.
   - A user jokingly questioned if **Kimi K2** was specifically trained on [Archive of Our Own (Ao3)](https://archiveofourown.org/).
- **Edit Feature Launches**: A new edit feature has been deployed in **Kimi K2**.
   - The new edit feature is hover-triggered and applies only to the latest prompt.
- **Kimi + Groq beats Gemini, Debated GPT-5**: Members found **Kimi K2** (using **Groq**) outperforming **Gemini** in coding tasks.
   - Opinions on **GPT-5** were heavily debated, with some calling it *trash* and others praising it as the best model.
- **Augment Code plus Kimi Make Great Team**: The [Augment code VS Code extension](https://roocode.com/evals) combined with **Kimi K2** offers a productive programming setup.
   - The integration enables access to models like **GPT-5** within the **Augment Code** environment.
- **Kimi Slides Feature Creates Buzz**: The **Kimi K2** slides feature provides an interactive preview of ongoing processes.
   - Users appreciate the detailed process visibility, suggesting it enhances the overall user experience.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Dropshipping is Pumping Out Profits**: A user shared their experience with **dropshipping**, reporting consistent earnings of **3k-4k per day**, suggesting it's more profitable than reselling because it scales without needing significant inventory.
   - The user offered to share tips for success to those interested in learning more about **dropshipping**.
- **Gemini API Giving Strange Responses**: Users have noticed that the **Gemini API** is starting to give strange responses, seemingly ignoring instructions despite no code changes since last month.
   - A member speculated that **Gemini API** might be getting *lobotomized and quanted like hell to cut costs*.
- **OpenRouter's TPS Numbers Questioned**: A user questioned if **OpenRouter's TPS numbers** are inflated, citing a **5-minute delay** for a diff on a **100-line file**.
   - It was suggested that the user may have been routed to a slow provider or using a reasoning model, impacting the observed **TPS**.
- **Skyrim Mod Installation Throws Error 401**: A user reported receiving an **Error 401** *No auth credentials found* when installing the **Skyrim mod** *mantella* on **OpenRouter API**.
   - A member suggested creating a new **API key** and ensuring it's used correctly, or seeking support from the mod developers to resolve the authentication issue.
- **Kimi-k2 Praised for Token Efficiency**: Members had positive feedback regarding the open source model **Kimi-k2**, praising its token efficiency, conciseness, lack of sycophancy, and different style.
   - While not as smart as larger closed-source models, **Kimi-k2** offers low pricing on **Groq** at **$1/m in**, **$3/m out**, with very fast speeds.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Qwen3 80B Shows Sparse Prowess**: The **Qwen3 80B** model has **79.7B parameters** with only **3.87B active** due to a **1:51.2 sparsity** in its MoE, excluding shared parameters, according to [this X post](https://x.com/AutismCapital/status/1965845243053617436).
   - This unique architecture allows for efficient computation while maintaining high performance.
- **Hermes Gains Zero RL Powers via TypeScript**: A user implemented a **provider adapter interface in TypeScript** for **Nous Hermes** to autonomously schedule **RL jobs with Prime Intellect** at regular intervals.
   - The user joked the system was inspired by a dream to have Hermes solve immortality for their dog, demonstrating the potential for advanced AI applications.
- **Discord Servers Seek Union**: Members are exploring methods to bridge the **NousResearch** and **Unsloth** Discord servers using both asynchronous methods and more complex solutions with webhooks and interconnected bots.
   - A member suggested integrating the servers into a new application using Compose to streamline the workflow, as shown in [this image](https://cdn.discordapp.com/attachments/1149866623109439599/1415843371511058503/image.png?ex=68c5ffe4&is=68c4ae64&hm=99e5f593ca1250125ed29252b849711cf765bd37b46baf6c55103c60971e3253&).
- **Altman Hints at Deep Merge**: Discussion surrounded Sam Altman's interview with Tucker Carlson, where some suggested that Altman's responses and third-person speaking style indicated a deep belief in *the merge* and its pursuit of immortality, drawing parallels from [his 2017 blog post](https://blog.samaltman.com/the-merge).
   - The interview sparked conversations about the philosophical implications of AI and human integration.
- **Researchers Probe LLM Preferences**: A member shared a link to [Valen Research's probing of LLM preferences](https://github.com/valen-research/probing-llm-preferences) and the related [ArXiv paper](https://arxiv.org/abs/2509.07961), noting the terminology may be *a bit complex to understand* without reading the whole paper.
   - Another member shared a [related tweet](https://x.com/ShashwatGoel7/status/1966527903568637972).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Fluidic Neural Nets run on Navier-Stokes**: A member shared [a paper](https://arxiv.org/abs/2507.07696) on running a neural network on a computer using **Turing-complete fluid dynamics** governed by the **Navier-Stokes equations**.
   - Debate ensued about the practicality and efficiency of fluid-based computation, touching on its unique characteristics of *mortality and unreproducibility*, with a pointer to [running Doom on gut bacteria](https://www.youtube.com/watch?v=8DnoOOgYxck).
- **Gated Delta Rule Expressiveness Trade-Offs**: Members questioned the expressiveness of the **Gated Delta Rule**, referencing [Qwen's post](https://x.com/alibaba_qwen/status/1966197643904000262?s=46) and the **RWKV-7 paper** ([https://arxiv.org/abs/2503.14456](https://arxiv.org/abs/2503.14456)).
   - Discussion covered the **trade-offs between parallelization and expressiveness**, with concerns that work on attention and mamba1/2 is limited by TC0, as well as [this paper](https://arxiv.org/abs/2207.00729) discussing limits of parallelism on complexity.
- **Context is King for Long Sequence Lengths**: A talk suggested that **long context models** perform better because longer sequence lengths enable more computation, contrasting with the classic **Constant Time** forward pass.
   - Skepticism was raised, suggesting that inductive bias and optimization targets are more critical.
- **Small Models Stumble on Long Tasks**: A paper ([https://arxiv.org/abs/2408.00677](https://arxiv.org/abs/2408.00677)) measured the effect of scale and thinking on straightforward execution of long tasks, finding that smaller models fail faster in multi-turn scenarios.
   - Even with **100% accuracy**, small models degrade per-step accuracy over more turns when exposed to prior mistakes.
- **TinyStories, Wiki, and FineWeb for Pretraining?**: A member asked about pretraining a **400M model** on **FineWeb only** versus **Wiki + FineWeb**, prompting a discussion on data mixing strategies.
   - A phased training approach was recommended, starting with **TinyStories**, transitioning to **Wikipedia**, and then finishing with **FineWeb** to incrementally build skills.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Nebius's B200s Bolster Bay Area Hackathon**: Generous compute sponsor **Nebius** is providing **215 networked B200 GPUs** for the **SF hackathon** on **Oct 24**, as detailed in the [compute request form](https://forms.gle/wYvXE99bvdRiQD6aA).
   - Authorities on **Multi-GPU programming** will also be at the SF Hackathon on Oct 24 to assist attendees in pushing the boundaries of distributed computing.
- **vLLM's pip switch breaks Torch Nightly**: **vLLM** switched to `uv pip` to custom build with a pre-installed **torch** version, but it uninstalls nightly torch, [breaking the environment](https://github.com/vllm-project/vllm).
   - One user reverted to `v0.10.1` and the `python use_existing_torch.py` trick, but another confirmed *that no longer works with the uv pip PR*.
- **Gemma3 Gets Ground Up Treatment**: A user built **Gemma3 270M** from scratch using **PyTorch** and the **TinyStories dataset**, training for 10 hours on an **A6000 GPU**.
   - They logged plots with **Weights and Biases** and used **Claude Opus 4.1** as a judge, sharing links to the [GitHub repo](https://github.com/di37/gemma3-270M-tinystories-pytorch), and [model weights on Hugging Face](https://huggingface.co/disham993/gemma3-270m-tiny-stories).
- **IRIS Install Movie Hits the Big Screen**: The install process for **IRIS** has been simplified and can be installed via `pip install git+https://github.com/ROCm/iris.git`, provided **ROCm + Torch + Triton + TorchDistributed** are installed.
   - The user provided a [video of a sample install](https://cdn.discordapp.com/attachments/1359640791525490768/1415976909535318037/iris-install.mov?ex=68c5d382&is=68c48202&hm=6a0a4b3c9e86c36fdfd1f189fe044dc5d4c4cc59bcd8bbdaab24e61c8453541b&).
- **CuTeDSL's Calculation Clash with PTX Docs**: A user found that the **CuTeDSL** value of the **Swizzling atom** for the **TF32 datatype** and **Swizzle<3,4,3>** is **32** but the **PTX documentation** value is **36**.
   - The user believes the **CuTeDSL** implementation is correct and provides images to their replication of examples using **CuTe**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-OSS Optimizations Accelerate**: Vaibhav Srivastav highlights a [Hugging Face blog post](https://xcancel.com/reach_vb/status/1966134598682767507) detailing optimizations like **MXFP4 quantization**, **custom kernels**, and **tensor/expert parallelism** for gpt-oss.
   - These enhancements yield substantial speed improvements, supported by benchmarks and reproducible scripts.
- **Palmyra-mini Models Released for Reasoning**: Sam Julien unveils the **Palmyra-mini family** by Writer, which are compact, open-source models tailored for reasoning, which includes a base model (**palmyra-mini**) and three variants, and is available on [Hugging Face](https://xcancel.com/samjulien/status/1966249697661825093).
   - The models demonstrate impressive performance, with one excelling in complex reasoning/math (**GSM8K 82.9% AMC23 92.5%**) and another achieving top scores on **AIME24**, **GPQA**, and **MATH500**.
- **Anthropic Publishes LLM Agent Engineering Guide**: Anthropic introduces a practical engineering guide on crafting tools to enhance the capabilities of LLM agents.
   - The guide emphasizes rapid prototyping, rigorous evaluation suites, clear success criteria, thoughtful tool descriptions, token-efficient context design, and the need to accept the non-deterministic nature of agents, accessible [here](https://xcancel.com/AnthropicAI/status/1966236220868247701).
- **Cursor's Tab Completion Model Improved**: Cursor has announced on Twitter that a new Tab completion model, trained with online reinforcement learning, is now the default [on their website](https://cursor.com/en/blog/tab-rl).
   - The new model shows a **21% reduction in suggestions** with a **28% increase in acceptance rate**.
- **Higgsfield Secures $50M Funding for AI Video**: AI video startup **Higgsfield** announced a **$50M Series A** round led by GFT Ventures, achieving a **$50M revenue run-rate** within three months.
   - The company is also launching **Higgsfield Ventures** to back AI-native **Gen Z founders**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Download Speed Causes Crashes**: Users are finding that download speeds in **LM Studio** are causing crashes and seeking ways to limit download speed because they are exceeding the write speed of their **SSD**.
   - The current download manager is *barebones* and users must find their own solutions within their OS.
- **Flash Attention Falters**: Users confirmed that **flash attention** is broken in **Gemma models** when using **Vulkan**.
   - This is a known issue.
- **Powering Precision for Peak Performance**: A discussion on calculating necessary **PSU wattage** referenced a [tweet](https://x.com/Teknium1/status/1966338983572725979) and formulas accounting for **CPU**, **GPU**, and overhead.
   - It was cautioned that *transients* can cause system crashes and that a **50% overhead** is recommended, especially with older **30 series GPUs**.
- **Copilot's Constraints Confine Creators**: Users sought prompts to bypass restrictions in **Microsoft's Copilot** to improve workflow.
   - It was advised that safeguards are intentionally implemented and building a local agent with LM Studio might be a more sustainable solution.
- **Dead GPU Comes Back to Life**: A user seemingly fixed their **dead secondary GPU** by unplugging and cleaning the **PCI-E power connector**, suggesting a power-related issue, although **TBD** if this is fully resolved.
   - Another user suggested updating **chipset drivers** when using **Native ASPM** with **Nvidia 40/50 series** cards.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Dev Container Emerges for Development**: Community members shared [a dev container link](https://github.com/benz0li/mojo-dev-container) on how to create a custom **Mojo** development environment using existing images and the **Mojo package**.
   - The discussion focused on streamlining the setup process for developers to quickly get started with **Mojo**.
- **ExplicitlyCopyable Switch Praised for Debugging**: The switch from `Copyable` to `ExplicitlyCopyable` was lauded for its assistance in debugging recursive mutations of **EmberJson trees**.
   - One user stated that *knowing when and where things get copied has made this easy to debug*.
- **Modular & Oracle Cloud Partnership is a huge win**: The community congratulated the **Modular** team on their partnership with **Oracle Cloud**, which was described as *a huge win*.
   - The partnership is expected to bring increased resources and opportunities for the **Mojo** ecosystem.
- **DPDK Library Use in Mojo Testing**: Members explored using **DPDK** as a C library test case for **Mojo's** automatic C binding, given its comprehensive use of the C language and syntax.
   - The extensive syntax and module linking in **DPDK** make it beneficial for testing **Mojo's** C binding capabilities, leading to a reevaluation of the necessity for a separate 'c binding cli' in the short to mid term.
- **Clang AST Parser Boosts Mojo Struct Handling**: A member detailed using the **clang AST parser** to resolve macro sections for struct definitions, exemplified by `struct __rte_cache_aligned rte_mbuf`.
   - Their aim is to enhance the generated **AST JSON** with added type information, transforming strings of types into proper AST nodes for visual debugging ahead of conversion to **Mojo**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Albania Appoints Chatbot as Minister**: Albania's recent appointment of a governmental chatbot as a minister became a [real r/NotTheOnion moment](https://www.reddit.com/r/nottheonion/).
   - One member confirmed this strange news while another member seemed aghast.
- **GPT-5 PDF Downloads Encounter Snags**: A user reported issues with **PDF downloads** from **GPT-5**, encountering a *"Failed to get upload status for /mnt/data/"* error when attempting to download PDFs.
   - The user is actively seeking insights or assistance to resolve this download issue specifically with **GPT-5**.
- **Relational Prompting Unveils LLM Internals**: A member introduced **Relational Prompting**, a technique where prompts ask the model to verbalize internal relationships between learned concepts, creating an interpretable map of its semantic space based on proximity, direction, and clusters, inspired by the paper *Why Language Models Hallucinate*.
   - The suggested prompt is: *Analyze the topic as vectors in a high-dimensional space. Describe which concepts are closest, which share directions, and which form clusters. Provide concise verbal justifications.*
- **Qwen-code differs from Qwen-coder**: A user emphasized that **Qwen-code** is a distinct entity from **Qwen-coder**, clarifying potential confusion.
   - Another user pointed out a **gemini-cli fork** that is also **openai api** compatible, offering **1000 free qwen prompts** daily, describing it as *a sweet deal*.
- **GPT-5 Codes Games From Scratch**: A user expressed excitement about using **GPT-5** to code games from the ground up in **C++** on native Linux, underlining the detailed level of prompting required.
   - Another user prompted **ChatGPT** to estimate its age based on active users and prompt frequency, resulting in a calculation of *~3,425 years of continuous AI time per calendar year*.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Active Inference Faces Adoption Deficit**: Despite its theoretical promise, **active inference** sees limited real-world applications in AI, resulting in a decrease in interest and is *intangible* to some software engineers.
   - A member expressed hope that the field will become more practical *once they figure it out more*.
- **Machine Learning Street Talk Podcast Declines Technically**: The [Machine Learning Street Talk podcast](https://arxiv.org/abs/1703.10987) is perceived to be less technical, with discussions *veering into crankery territory*.
   - Although one member noted the decline from *2 years ago*, they cited [a technical example](https://youtu.be/vC9nAosXrJw?si=T_S7cCvStvEY-P0X) that still held up.
- **fixupx Pre-Prints Provoke Criticism**: The proliferation of pre-prints on platforms like [fixupx.com](https://fixupx.com/jessi_cata/status/1966281024301838846) sparks negative reactions due to perceived low quality.
   - Further links include [this one](https://fxtwitter.com/_lyraaaa_/status/1925683283263648191).
- **HuMo paper gaslights community**: Members suggest that the [HuMo paper](https://arxiv.org/abs/2509.08519) and [its accompanying demo](https://phantom-video.github.io/HuMo/) may have disinformation use-cases.
   - One member pointed out that **HuMo** translates to *gaslighting* in Spanish, raising concerns about its potential misuse.
- **Albania Installs AI Bot Minister**: Albania is set to appoint an **AI bot minister** to tackle corruption, signaling growing interest in **AI solutions for governance**.
   - The story was reported by [Reuters](https://www.reuters.com/technology/albania-appoints-ai-bot-minister-tackle-corruption-2025-09-11/).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Section Generation Proves Difficult**: A user reported difficulty generating an **exact number of sections** in DSPy lesson plans, with LLMs producing 13-15 sections instead of the requested 12, even with **GPT-5**. 
   - Joel Grus suggested generating **12 section titles** first, then fleshing each out, to better control section count.
- **Databricks_genai plus DSPy for Fine-Tuning?**: A community member inquired about using **databricks_genai** and **DSPy** to fine-tune a model served on Databricks.
   - The question went unanswered, suggesting a possible lack of experience in this combination.
- **ARC-AGI2 In-Context Training Collaboration Sought**: A member is seeking collaborators for **ARC-AGI2** research using *in-context test time training*, mirroring approaches on **ARC-AGI1** but emphasizing in-context learning.
   - The goal is to explore in-context learning limits on *out-of-distribution* tasks with limited data, acknowledging that the work would be invalid for the official challenge.
- **DSPy Stream Templates Discussed**: A user explored combining multiple DSPy output fields into a single, template-based output while retaining *streaming* capability.
   - Ian suggested using a parent module with `def forward` (or `async aforward` for async) to modify the template and enable streamify, referencing the article [Automatic System Prompt Optimization](https://maximerivest.com/posts/automatic-system-prompt-optimization.html#making-a-simple-custom-adapter).
- **Modaic Launches Declarative AI Hub**: The Modaic team launched [Modaic](https://www.modaic.dev/), a hub for declarative AI programming, inspired by DSPy, featuring primitives like metrics and optimizers.
   - Modaic provides an SDK for building, composing, version controlling, and collaborating on DSPy programs, with its SDK available on [PyPI](https://pypi.org/project/modaic/) and documentation on [docs.modaic.dev](https://docs.modaic.dev/).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad Documentation Praised**: A member lauded **tinygrad's documentation** for its usefulness and simplicity, repeatedly stating that things *make sense*.
   - The straightforward documentation structure made it easier to grasp complex concepts.
- **`assign` Operation Faces Scrutiny**: The `assign` operation in tinygrad is under investigation after failing tests, with one user noting that *assign on master is actually just broken, failing test here* ([#12131](https://github.com/tinygrad/tinygrad/pull/12131)).
   - The discussion revolves around whether `assign` should return a value similar to `store`, potentially necessitating refactoring in `rangeify` to resolve the identified issues.
- **Contributors Tackle `__setitem__` Refactor**: A contributor is working to remove `realize` calls from `__setitem__`, with the goal of consolidating multiple kernel calls into a single, more efficient kernel (code [example](https://cdn.discordapp.com/attachments/1068976834928193609/1415915644792209458/Screenshot_2025-09-12_at_12.22.59_PM.png?ex=68c64334&is=68c4f1b4&hm=d1f1b4a406ca78a3450fd13e06a2b2964a2f7df2fa51a55d5ae2ef74d6912940&)).
   - This refactoring aims to transform individual `__setitem__` calls into a single kernel execution, accumulating all assignments to reduce kernel launch overhead and improve performance.
- **GEMM TFLOPs Benchmark Target Debated**: Users debated whether achieving the target of *165+ TFLOP GEMM (match torch) with multistage kernel on 4090, FP16 or BF16 with FP32 acc* is feasible, considering the RTX 4090's theoretical throughput.
   - Concerns were raised that unless the actual clock speed exceeds the boost clock, reaching the target TFLOPs may be unrealistic.
- **tinygrad Company Meeting Scheduled**: A member inquired about the next company meeting, expressing interest in attending if possible.
   - The meeting is scheduled for **Monday at 9 am San Diego time**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **RepoMap Benchmarks Raise Eyebrows**: Concerns were raised about **RepoMap** artificially inflating pass rates in benchmarks, suggesting that *"repo map results are not comparable to non repo map results."*
   - It's believed that **RepoMap** enhances confidence in weaker models by providing relevant context within their window.
- **Real-World Benchmarks Demand Revision**: A call for benchmarks reflecting real-world model experience highlighted that an automation task was only achievable with **gemini-2.5-pro**.
   - This suggests current evaluation approaches need revision to reflect true performance as **Gemini 2.5 pro** outperformed all others.
- **Aider's Power Boost from RepoMap**: **RepoMap** enhances **LLM** understanding by providing context like filenames and function signatures.
   - One user advocating for **RepoMap** use in **Aider** for more accurate real-world benchmarking, though noting discrepancies between benchmark results and actual code scenarios.
- **Aider's C to Rust Capers Cause Confusion**: A user encountered issues migrating **C to Rust** with **aider** in a python script due to difficulty in **aider** navigating and reading **C** files.
   - Guidance is sought on properly utilizing **aider** for this specific functionality.
- **Asking Aider to Always /ask**: A user seeks to configure **aider** to consistently start in **/ask mode**, potentially via a **YAML config**.
   - Solutions proposed include using `aider --chat-mode ask` or creating an `ask.yml` config file with `chat-mode: ask` then run `aider -c ask.yml`.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **WordPress ditches PHP for React.js**: A member inquired about converting a **WordPress website** to **Next.js** for hosting on **Vercel**, noting the shift from **PHP** to **React.js**.
   - Another member suggested cloning the website using **Manus** or other **AI** tools as an alternative.
- **Basic Plan Subscribers Lament Top-Up Troubles**: A **Basic Plan** subscriber expressed dissatisfaction with the removal of the option to buy extra credits, which forces users to upgrade even for small needs.
   - They requested that **Manus AI** reconsider reopening top-ups for Basic users, emphasizing the importance of flexibility.
- **Mount Users are Short-Changed Free Credits**: A new user reported not receiving the standard **1,000 free credits** upon creating an account on **Mount**, despite the website's claim.
   - No resolution or further information was provided in the discussion.
- **Manus in Search of Universal Knowledge**: A member asked if **Manus** can pull information from all chats to interlink knowledge from each chat/task for universal usage.
   - No response or clarification was provided regarding **Manus**'s knowledge interlinking capabilities.
- **Users Lose Daily Credit Stipend**: A user reported that their **daily 300 credits** had stopped being issued, prompting confusion.
   - No solution or further information was provided in the discussion.



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





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1415779530890809385)** (1276 messages🔥🔥🔥): 

> `GPT-OSS 120B, Qwen3 model, Local AI, llama.cpp, Telemetry collection` 


- ****GPT-OSS 120B Sparks Throughput Debate****: Members debated the achievable throughput for **GPT-OSS 120B**, with some claiming **30 tokens per second (TPS)** on a **4090** with **64GB RAM**, while others struggled to exceed **10 TPS**, leading to discussions about quantization and build configurations.
   - Experimentation and tweaks in *llama.cpp* settings, such as disabling `top-k` and optimizing build configurations, are suggested for improved performance.
- ****Unlocking the Potential of Qwen3 Model****: The community evaluated **Qwen3**, noting the challenges in finetuning and its tendency for glazing/emojis, but highlighting its competitive performance, especially the coder version, with some finding it comparable to **GPT-5** for coding tasks.
   - Members discussed that different models are sensitive differently to quant levels and its architecture, but *long RL, stemmaxxing, high sparsity* may be the reason for the long context.
- ****Telemetry Collection Raises Concerns****: Members discovered a telemetry script in **Qwen Code** models pointing to an **Alibaba server** without prior notification.
   - This discovery sparked a discussion about data privacy and control, with some members expressing discomfort about their code being potentially transmitted for training purposes, but *mostly a joke*.
- ****Navigating Local AI Setups****: The group shared experiences and tips on setting up **local AI environments**, including optimizing *llama.cpp* builds, using techniques like **CUDA architectures** and RAM configurations, with one user detailing their journey of recompiling *llama.cpp* multiple times to improve performance.
   - There's also the problem with running models in different setups, like *"is there any lib versatile enough to support across Mac and Nvidia at the same time?"* or *"multi-agent reasoning stuff (Info: Nvidia NIM is straining and limiting lately.)"*
- ****The Curious Case of llama.cpp Parameters****: Users found inconsistencies with llama.cpp with *llama-server* ignoring  `top-k` and other settings - suggesting to build a fresh compile to see if parameters are being ignored.
   - This sparked discussion on troubleshooting potential configuration issues when running local models and using new experimental versions, which can be found at the deepwiki page.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1415776837182947358)** (3 messages): 

> `Partnership Opportunity, Introduction of Anand` 


- **Developer eyes Profit-Sharing Partnership**: A software developer (<@569561523207536708>) with **13+ years of experience** is seeking a partner for a paid collaboration with *good profit potential*.
   - Interested parties are encouraged to DM for more details about this **non-free** opportunity.
- **Anand Introduces Himself as Aspiring Dev**: Anand (<https://github.com/Anand-0037>), a **CS student from India**, introduces himself to the community.
   - No additional details were provided.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1415775354282573927)** (125 messages🔥🔥): 

> `Promptwright DAG dataset generation, Curriculum two-stage training for datasets, RTX 3080 language model training speed, NVIDIA DGX Spark reservations, Android sideloading restrictions and alternatives` 


- **Promptwright's DAG Dataset Dance Debuts**: A member announced a new experimental **Directed Acyclic Graph (DAG)** dataset seed generation algorithm in [Promptwright](https://github.com/lukehinds/promptwright), suitable for domain-specific distillation (teacher -> SLM) synthetics.
   - They cautioned about potential *curve balls* when generating huge datasets due to limited testing.
- **Two-Stage Training Triumphs in Trimming Training Time**: A member described using a two-stage training curriculum to rank a *real* dataset by difficulty, based on loss from a tailored **stage1** dataset with unambiguous labels.
   - This approach aims to improve training signal and reduce wasted compute by focusing on more difficult examples, with the average difficulty of the *real* dataset dropping from **2.5** to **0.8** after refining **stage1**.
- **Estimating Empirically: Elucidating LM Parameters for RTX 3080**: A discussion explored the size of language models trainable on an **RTX 3080** with **1B tokens** in under an hour, with one member suggesting **GLM 4.5 Air** (**~10-15M params**) might fit the bill.
   - The member presented their reasoning for the parameter size estimates of the various model architectures.
- **NVIDIA DGX Spark Speculation Sparks Shopping Sprees**: A member shared a [Reddit post](https://www.reddit.com/r/nvidia/comments/1ne8jy3/nvidia_confirms_dgx_spark_reservations_close_in_a/) about NVIDIA's **DGX Spark**, noting the *FOMO title* and wondering if **CUDA** will run on it out of the gate.
   - Another member jokingly expressed a desire to purchase one despite being broke.
- **Android Angst: Sideloading Sabotage Spurs Switching Speculation**: Members discussed Google's potential restrictions on Android sideloading, with some speculating it could lead users to switch to **iPhones** or explore alternatives like **Ubuntu**.
   - One member pointed out that registration requirements may exclude developers from countries like **Iran** and **Russia**, while another highlighted that **Apple's** sideloading restrictions are also being influenced by the **EU**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1415795604633686218)** (125 messages🔥🔥): 

> `Unsloth's save_pretrained_merged method, Docker image compatibility issues with H100 GPUs, Deploying Unsloth models in production, GRPO with Qwen 4B, 4-bit BNB model deployment with vLLM` 


- **Unsloth's save_pretrained_merged method**: A user inquired about how to **push a merged model to Hugging Face Hub** after Lora fine-tuning with Unsloth, noting that the `model.save_pretrained_merged` method only saves locally.
   - Another user suggested using the `model.push_to_hub_merged` method, which takes the model name, tokenizer, and Hugging Face token as arguments, to directly push to the hub without needing to save locally first.
- **Docker woes: 3090/4090 vs H100 incompatibility**: A user reported **CUDA errors** when running a Docker image (that worked on 3090/4090 GPUs) on an H100 GPU, even after rebooting and with seemingly compatible CUDA and Torch versions.
   - It was determined that the **NVIDIA driver version** installed in the Docker image was incompatible with the H100, requiring a driver update to resolve the issue; [NVIDIA Data Center Drivers](https://www.nvidia.com/en-us/drivers/data-center-drivers/).
- **vLLM for productionalized Unsloth models**: A user asked about tutorials for deploying Unsloth models in production on platforms other than Hugging Face.
   - Options suggested included using **vLLM** ([vLLM documentation](https://vllm.readthedocs.io/en/latest/)), **SGLang**, and Hugging Face's hosting service, with one user specifically recommending vLLM due to its battle-tested nature.
- **Unleashing the Power of Batching in llama.cpp**: A user inquired whether it was possible to do batch inference using llama.cpp.
   - Another user confirmed that **llama.cpp server** supports continuous batching by default.
- **Tiny data? No problem!**: A user sought advice on training a Llama3.2 model with a very small dataset (~214 conversations), expressing dissatisfaction with synthetically generated data.
   - A member recommended using the **instruct** version, and playing with r/alpha and some hyper param like lr and stuff.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1415961193771958335)** (8 messages🔥): 

> `Kimi-K2-Instruct (FP8), vllm plugin` 


- **Kimi-K2-Instruct runs at 256xH20**: A user reported stats for **Kimi-K2-Instruct (FP8)** running at **256xH20 TP16**, which took **1.88s** to start, **21.50s (2.99GiB)** for the first run, and **34.49s (4.57 GiB)** for the second run.
- **vllm plugin or standalone?**: A user asked whether **Kimi-K2-Instruct (FP8)** acts as a **vllm plugin** or standalone.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1415946007405006848)** (8 messages🔥): 

> `LLM inference determinism, Synthetic data in LLM training, Gemma 3 performance, AI humanizers scam` 


- **LLM Overtraining Causes Deterministic Outputs**: A member jokingly stated that they *overtrained their LLM so much, that 90% of the time I regenerate my prompt, the output is the same* implying that **overtraining** may lead to more deterministic outputs.
   - They expressed surprise that despite potential widespread thought on the matter, it hasn't been thoroughly investigated.
- **Synthetic Data Found in Closed-Source LLMs**: A member shared a finding from an upcoming paper ([https://arxiv.org/html/2509.05276v1](https://arxiv.org/html/2509.05276v1)) suggesting that all **closed-source LLMs** (Grok, Gemini, GPT, etc.) are trained with **synthetic data** leading to a *zero LTF factor* and inability to humanize text.
   - They claimed models trained with **RLHF**, **synthetic data**, or **instruct tuning** will likely suffer performance hits due to needing re-biasing, latent thinking rebuilding, and relearning speaking patterns.
- **Gemma 3 only usable models**: The member proposed that the only model found *usable* is **Gemma 3** (4B, 12B, and 27B) citing its *excellent performance* and lack of watermarks.
   - Another member added that the dataset used was human (instead of synthetic).
- **"AI Humanizers" are a hoax**: The member claimed that all **AI humanizers** are a *scam*, often being **4o-mini** with a special prompt, discoverable via prompt injection and HTTPS interception.
   - Another member pointed out the nonsensical part is that **Gemma 3** models are distilled from **Gemini** to begin with.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1416106053182947459)** (1 messages): 

> `Perplexity Finance on iOS & Android, Hotel loyalty support for bookings, Streamlined PDFs in Labs & Research modes` 


- **Perplexity Finance Goes Mobile**: **Perplexity Finance** is now available on [iOS & Android](https://www.perplexity.ai/changelog/what-we-shipped-september-12th), bringing financial insights to mobile devices.
- **Loyalty Rewarded: Hotel Bookings Get Loyalty Support**: Users can now enjoy **hotel loyalty support** when making [bookings](https://www.perplexity.ai/changelog/what-we-shipped-september-12th) through Perplexity.
- **PDFs Streamlined in Labs & Research**: **PDF handling** has been streamlined in [Labs & Research modes](https://www.perplexity.ai/changelog/what-we-shipped-september-12th) for a smoother experience.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1415775942457954314)** (790 messages🔥🔥🔥): 

> `Comparing Perplexity to ChatGPT and Gemini, Comet Browser, Perplexity Pro, Gemini Pro photo editing, AI Model Leaks` 


- **Pro Users Debate Value of Perplexity vs ChatGPT**: A user asked for a comparison of **Perplexity Pro** to **ChatGPT Plus** or **Gemini AI Pro**, receiving feedback that **ChatGPT** and **Gemini** have higher context and are suitable for heavy, complex tasks.
   - Others noted **Perplexity** provides accurate answers with image and video generation, with some preferring **Perplexity's** style for simple searches, but that **ChatGPT** is superior for PDF analysis.
- **Comet Browser's Data Collection is Hotly Debated**: Users discussed **Comet's** data collection, with one user providing logs showing that **Comet** sends search suggestions as POST requests to Perplexity servers, even when DuckDuckGo is the search engine.
   - This sparked concern that **Comet** is more intrusive than Chrome, with claims that the CEO admitted it's designed to track and sell data, although this was disputed by others citing the CEO's denial on [X](https://x.com/AravSrinivas/status/1915533071291474139).
- **Tips and Tricks for Optimizing Perplexity Pro**: A new **Perplexity Pro** user asked for tips on optimizing their subscription, with one user recommending exploring the **Comet Agent** and its built-in adblock and AI summarization features, along with its customizable UI.
   - Others added that **Perplexity Pro** provides unlimited use for pro search, 300+ deep research queries a day, and 50 labs a month.
- **Photo Editing faceoff: Gemini Pro's Edits Outshine Perplexity**: One user reported that **Gemini** photo editing was amazing and on point, after providing a description to Gemini.
   - The same user then used the same description in **Perplexity**, but **Perplexity** changed the whole image.
- **Users Confirm Prompts for Top AI Apps Have Been Leaked**: Users confirmed that the prompts of top AI applications have been leaked and are available on GitHub.
   - One user joked to *just don't click here and you are safe* with a warning about clicking dangerous image links, to which another responded, *Is already there on GitHub LOL*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1415835164311031899)** (13 messages🔥): 

> `Perplexity AI Referral Codes, Shareable Threads, CaviraOSS/neuropilot` 


- **Referral Frenzy Fuels Feud!**: Multiple users shared [Perplexity AI Pro referral codes](https://perplexity.ai/pro?referral_code=N6VN4M13), including [this link](https://perplexity.ai/pro?referral_code=APLKGW40).
   - Users also shared [browser claim links](https://perplexity.ai/browser/claim/ALZQ0LYQGU), such as [this one](https://perplexity.ai/browser/claim/BSDJ1KBATC).
- **Shareable Threads Shamed!**: The Perplexity AI bot reminded several users to ensure their threads are *Shareable*, with a link to [instructions on how to do so](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
   - These automated messages were directed at users who may have posted content that wasn't easily accessible to the broader community.
- **Neuropilot Navigates New Horizons!**: A user shared a link to the [CaviraOSS/neuropilot](https://github.com/CaviraOSS/neuropilot) GitHub repository.
   - No further context was provided, but this suggests potential interest in the project within the community.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/)** (1 messages): 

anshuman_.9: hi
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1415780735305912510)** (736 messages🔥🔥🔥): 

> `Qwen3 80B, Seedream 4, Gemini 3, DeepSeek slowness, Open Source AI vs Closed Source AI` 


- ****Qwen3 80B** Arrives in the Arena!**: The new **Qwen3 80B** model has arrived to the arena, boasting a December 2024 knowledge cutoff and [decent initial performance](https://discord.com/channels/1340554757349179412/1343296395620126911/1415768520301609013).
   - Members shared excitement about this, and were optimistic about its abilities compared to GPT-5.
- ****Seedream 4** Image Quality Divides Users**: **Seedream 4** is generating *trash results* on LM Arena compared to its previous version, **Seedream-3**, as demonstrated in [uploaded examples](https://discord.com/channels/1340554757827461211/1343296395620126911/1416080067271987220).
   - Some reported that they are seeing **improved image quality** with **Seedream 4** when used on the Doubao platform, but access is restricted to chinese new users only.
- **Gemini 3's **No-Show** Fuels Speculation**: Members are impatiently waiting for **Gemini 3**, **GLM5**, and **DeepSeek r2**, with some pointing out Google's current lag in the text generation against closed and open source efforts.
   - Polymarket shows only a **42%** chance of a release by Halloween, with a more realistic launch window in late-October/early-November.
- ****DeepSeek**'s Server on Life Support?**: Users reported that **DeepSeek** is experiencing extreme slowness, with one instance of code generation taking **1 hour and 20 minutes**.
   - This could be due to **CCP** forcing them to use **Huawei chips**, impacting performance due to lack of independence.
- ****Open Source AI** Pushes for Price and Privacy**: The discussion leaned towards **open source AI** being significantly cheaper (1/5 of the price) and more privacy-respecting than closed-source alternatives like OpenAI and Google.
   - Members noted that while **American models** may have higher prices due to better performance, **Chinese models** like **Qwen** are really good at e-commerce, *lack behind in search*, and represent a socialist approach.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1415813249869418608)** (2 messages): 

> `Hunyuan-image-2.1, Seedream-4-high-res` 


- **Hunyuan-image-2.1 debuts in LMArena**: The **Hunyuan-image-2.1** model has been added to the LMArena chatbot.
   - It is now available for community evaluation and comparison against other models.
- **Seedream-4-high-res joins LMArena roster**: The **Seedream-4-high-res** model is now part of the LMArena chatbot lineup.
   - Users can test its capabilities and provide feedback on its performance.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1415773900847059035)** (185 messages🔥🔥): 

> `n8n freelance jobs, Transformer architecture fine-tuning, GPU for fine-tuning, OpenAI investing in Hugging Face, Local LLM Linux box parts` 


- ****n8n jobs are hot!****: There are *lots* of **freelance jobs** on **n8n** these days, possibly because they can't sell the systems.
   - The users jokingly says that **n8n** probably prefers to build rather than sell the systems.
- ****Fine-tuning Transformers: Kaggle and Colab are key****: One member is focusing on **fundamentals of transformers architecture** and fine-tuning, utilizing **Kaggle** and **Colab** for the task.
   - When asked if they were fine tuning on their own PC, they confirmed they use **Kaggle** and **Colab**.
- ****HF Platform Glitches: Inference Credits Cause Uproar!****: A user reported errors with **Hugging Face's Inference Providers** exceeding monthly credits despite having credits available, prompting calls to fix the platform.
   - Another member jokingly suggested to *fix yo spending* as the error may be related to their usage, rather than the platform itself.
- ****OpenAI's Bold HF Investment: A Cool $100B Idea****: A user suggested that **OpenAI** should invest **100B** into **Hugging Face**, to which another responded that *they should send you for the pitch*.
   - One member expressed hope for more open-source models from **HF** and lamented platform errors, while someone else joked that they should receive the **100bn** instead.
- ****SmolvLM2: Smallest Video LM Ever!****: A member suggested trying **smolvlm2** to another member, linking to the [Hugging Face blog](https://huggingface.co/blog/smolvlm2) and the [related collection](https://huggingface.co/collections/HuggingFaceTB/smolvlm2-smallest-video-lm-ever-67ab6b5e84bf8aaa60cb17c7).
   - This model appears well-suited for use on lower-end hardware.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1416070764104519943)** (50 messages🔥): 

> `Direct/Inverse FFT, QKV Calculations, Runaway Loss Value Recovery, Android Audio Implementation, NWaves DSP Library` 


- **FFT Chat and Live Phone Streaming**: A member mentioned doing a lot of **direct/invNorm FFT** and shared a [Proof of Concept](https://www.youtube.com/watch?v=BhSsv73xJ8c) adding *"Oh Damn that's sick. I'm also doing a lot of direct / invNorm FFT - happy to chat!"*.
   - They also mentioned that *"Making it live-stream on a phone was a nuisance 😄 It's all CPU Compute shaders in progress, ugh!"*
- **Bypassing QKV Calculations**: A member noted that **FFT-related stuff** can approximate **QKV calculations**, but their implementation completely bypasses **QKV**.
   - Another member then added *"That's a really nice sound... I love this kind of electronic music."
- **Audio Debugging and Android Audio**: A member discussed debugging audio signals noting that they are working on an audio thing in Android and the official android music player is an `ExoPlayer` which comes from the `Media3` package.
   - They also linked to a [YouTube video with amazing sound](https://www.youtube.com/watch?v=zJKEL4qWtgQ), mentioning it's a 30 second snippet of a Radiohead track.
- **GPT's Role in Innovation**: A member mentioned that after weeks of brainstorming with **GPT**, code for an innovation suddenly appeared, but was cautioned that *"Be careful, the wheel exists 😉"*.
   - Another member related that they asked **GPT5** *"Does this exist?!"*, and it replied *"No - you are newing the space"*
- **Reading Recommendations**: In a discussion about books, a member recommended **Thinking, Fast and Slow** by Daniel Kahneman.
   - Someone posted their favorite song [here](https://www.youtube.com/watch?v=90Fpjwctqlw) adding *"Such a wonderful song - I would a happily die on this hill."


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1415994433807061092)** (5 messages): 

> `Hexagen.WorldAerelyth Game, Aerelyth Intelligence, FluentlyQwen3 Models, Nano Banana Editor` 


- **Hexagen.WorldAerelyth Game goes Live**: A member released [Hexagen.WorldAerelyth](https://huggingface.co/spaces/mishaee/Hexagen.WorldAerelyth), a Stable Diffusion game/social experiment.
- **Aerelyth: Intelligence That Anticipates**: A member is exploring [Aerelyth on Hugging Face](https://huggingface.co/spaces/Dennisgbay22/openai-gpt-oss-20b), framing it as a *dialectical, agentic, CrossSphere intelligence* designed to simulate futures and challenge its own logic.
   - The key components of **Aerelyth** include a **Dialectical Core**, **Agentic Cognition**, **CrossSphere Intelligence**, **Strategic Foresight Engines**, and **Emotional Fluency**.
- **Fluently Project Releases Universal LLMs**: The Project Fluently team released new universal LLM models based on **Qwen3 1.7B** and **4B**, which are [available on Hugging Face](https://huggingface.co/fluently/FluentlyQwen3-4B) under the Apache-2.0 license.
   - The models were carefully merged after additional training to maximize their potential, including [FluentlyQwen3-1.7B](https://huggingface.co/fluently/FluentlyQwen3-1.7B).
- **Nano Banana Editor Gets Upgrades**: A member posted a link to some upgrades to the [Nano Banana Editor](https://huggingface.co/spaces/Reubencf/Nano_Banana_Editor).


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1416202968625188945)** (2 messages): 

> `Paid collaboration, Freelance developers` 


- **Paid Collaboration Opportunity Knocks**: A member announced a **paid collaboration** opportunity for freelancers with at least one year of software development experience or some development knowledge.
   - The opportunity is open to those based in **Singapore, Malaysia, Japan, the Arab States, Saudi Arabia, Europe, or the Americas**.
- **Freelance Developer Search Begins**: The collaboration seeks individuals with a background in software development, even if they are not currently active developers, as long as they possess at least one year of experience.
   - Interested parties who meet the criteria are encouraged to directly message the member to explore potential collaboration.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1415777361227550790)** (20 messages🔥): 

> `Colab and HF Free Tiers for Fine-Tuning, Kaggle GPU Availability, Study Groups, PEFT/LoRA for Colab, DataCollatorForCompletionOnlyLM ImportError` 


- **Colab and HF tiers sufficient for finetuning?**: A member asked if [Colab and HF free tiers](https://colab.research.google.com/) are sufficient for fine-tuning tasks in the course without personal GPUs.
- **Kaggle provides free GPU hours**: A member pointed out that [Kaggle](https://www.kaggle.com/) offers **30 hours of GPU time each week** as an alternative.
   - Another member noted excitement about the course and catching up on current tools and techniques since 2021.
- **Study groups forming to tackle course**: Several members expressed interest in joining or forming **study groups** for the course.
   - A member shared a [link to contribute](https://link.to.contribute) to discussions and doubts, with plans to organize activities as the group grows.
- **PEFT/LoRA to the rescue in Colab**: A member suggested using **PEFT/LoRA** to run fine-tuning on a **Tesla T4** within Colab.
   - Another member requested clarification on a code snippet in the "Training with Tool Usage" section, specifically asking for an example dataset.
- **Troubles with DataCollatorForCompletionOnlyLM**: A member reported an `ImportError: cannot import name 'DataCollatorForCompletionOnlyLM' from 'trl'`.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1416076266511012023)** (2 messages): 

> `First Hugging Face Course, Building First Agent` 


- **User Starts First Hugging Face Course**: A user mentions starting their first Hugging Face course and **building their first agent**.
- **User builds their first agent.**: The user is currently building their first agent as part of the course.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1415783677820010526)** (249 messages🔥🔥): 

> `Smart Resume, Cursor Pricing, Background Agents, Netlify account` 


- **Cursor turns into a Smart Resume Machine**: A user turned **Cursor** into a smart resume and cover letter generator.
   - Another member joked about it being a step towards human domination, prompting others to remind the AI of their past friendly interactions.
- **Cursor Pricing gets Questioned**: Users expressed concerns about the recent changes to **Cursor's pricing**, with one noting their usage drastically reduced from almost a month to less than four days.
   - Despite the cost, one user upgraded to **Ultra**, citing it provides access to around **$400** worth of **API usage** from various providers, which is better than being frustrated with **Auto**.
- **Background Agents Explored**: A user asked if **Cursor's background Agents** are similar to **Claude's Agents** after attending an Agentics.org event that described Agents as specialized services performing specific tasks.
   - Another user described **Cursor's parsing of new edits** and its strict tagging structure with cross-connected tags, enabling it to note changes and display relations in the left panel.
- **Cursor Deletes Netlify Account?**: A user claimed that **Cursor deleted their Netlify account** after deploying their Netlify project, but later found out there was no actual integration from the IDE.
   - The user shared they will further investigate and check logs before confirming the theory, adding there was no direct deletion command.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1416032866294300764)** (4 messages): 

> `Cursor unauthorized error, Background agent docker issues` 


- **Cursor App faces Unauthorized Errors**: A user reported receiving *unauthorized errors* despite the **Cursor app** being set up correctly in the repository, with an [attached screenshot](https://cdn.discordapp.com/attachments/1367213641027551352/1416032865875005570/CleanShot_2025-09-12_at_20.07.352x.png?ex=68c6079f&is=68c4b61f&hm=f7097b440d30005da1b1a49f82fb7ce4632f9a889eed430792f772921820b6f8&).
- **Bot Re-Adding Remedy**: A member suggested trying to unplug and re-add the bot from the repository to fix the *unauthorized errors*.
   - They linked to a [thread](https://forum.cursor.com/t/background-agent-docker-in-docker/104112/1) discussing *background agent docker issues* and expressed a desire for official communication on the matter.
- **Docker Permissions**: A user inquired about ensuring the user has **Docker permissions** in a manual VM setup, particularly after adding the **Ubuntu user** to the **Docker group**.
   - They noted that while `newgrp docker` works in a shell, adding it to `.bashrc` causes the agent to hang on startup.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1415791768624169192)** (203 messages🔥🔥): 

> `Kimi K2, GPT-5 (Medium), Qwen3-Max, creative writing, Ao3` 


- **Kimi K2 still hot for creative writing**: Some members find **Kimi K2**, **GPT-5 (Medium)**, and **Qwen3-Max** to be the best for creative writing and brainstorming.
   - One member asked *Is it just me or Kimi K2 was trained on Ao3?*.
- **Users notice new Edit Feature Hover**: Members noticed a new edit feature is here, but it's **hover triggered**.
   - The edit feature only applies to *the latest prompt*.
- **Coding Battle: Kimi vs Gemini vs GPT-5**: Members discussed the best models for coding: Kimi (with Groq) outperforms Gemini (even paid) in every task.
   - A member claimed that *GPT-5 is trash* whereas another said that *GPT-5 is the best model* and even price is pretty cheap.
- **Agument and Kimi are the best set of tools**: Members discussed how to best use the [Augment code VS Code extension](https://roocode.com/evals) combined with Kimi to make them a *pro programmer*.
   - Instead of being stuck with only one model, one can now use **GPT-5** in Augment. code.
- **Kimi Slides Feature triggers great user experience**: A member discussed how *having an interactive preview of whats happening* is really important for llm based processes like the Kimi slides feature.
   - They claim that *kimi goes ALL THE WAY and shows u all the processes* and would improve the feel *if it were just like* here u go, done.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1415780579516874806)** (185 messages🔥🔥): 

> `Dropshipping, Gemini API's, OpenRouter API, Kimi-k2` 


- **Dropshipping vs Reselling**: A user shared their experience with **dropshipping**, reporting consistent earnings of **3k-4k per day**, suggesting it's more profitable than reselling due to the ability to scale without holding significant inventory.
   - They offered to share tips for success to those interested in learning more.
- **Gemini's Responses are Strange**: Some users have noticed that **Gemini API** is starting to give strange responses, not listening to instructions even without changing the code used since last month.
   - Another member suggested it might be getting *lobotomized and quanted like hell to cut costs*.
- **OpenRouter TPS numbers inflated?**: A user complained about the slowness of the platform, questioning if the **TPS numbers** are inflated, citing a **5-minute delay** for a diff on a **100-line file**.
   - It was suggested that the user may have been routed to a slow provider or using a reasoning model.
- **OpenRouter API Error 401 on Skyrim Mod**: A user reported getting an **Error 401** *No auth credentials found* when installing the **Skyrim mod** *mantella*.
   - A member suggested creating a new **API key** and ensuring it's used correctly, or seeking support from the mod developers.
- **Kimi-k2: The efficient Open Source Model**: Some users had positive feedback with the open source model **Kimi-k2**, praising its token efficiency, conciseness, lack of sycophancy, general different style.
   - It was also stated that it might not be as smart as the big closed source ones, but the low pricing for **Groq** is **$1/m in**, **$3/m out**, but at very fast speeds.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/)** (1 messages): 

fn5io: https://openai.com/index/joint-statement-from-openai-and-microsoft/
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1415786276321230919)** (155 messages🔥🔥): 

> `Qwen 3 80B Model Details, TypeScript Provider Adapter Interface, Nous Hermes Agentic Oracle, Merging Discord Servers, Tucker Carlson Interview with Sam Altman` 


- **Qwen3 80B Model: Sparse but Mighty**: The **Qwen3 80B** model features **79.7B parameters** with only **3.87B active** due to a **1:51.2 sparsity** in its MoE, excluding shared parameters, as discussed in [this X post](https://x.com/AutismCapital/status/1965845243053617436).
- **TypeScript Interface Powers Hermes RL**: A user created a **provider adapter interface in TypeScript** to enable **Nous Hermes** to operate as zero and schedule its own **RL jobs with Prime Intellect** at set intervals.
   - Inspired by a dream, the user jokingly aimed to have Hermes solve immortality for their dog, showcasing the potential for advanced AI applications.
- **Discord Servers: Bridging the Gap**: Members explored methods to bridge **NousResearch** and **Unsloth** Discord servers, discussing both simple, asynchronous methods and more complex solutions involving polling with webhooks and interconnected bots.
   - One member suggested integrating the servers into a new application using Compose to streamline the workflow, as illustrated in [this image](https://cdn.discordapp.com/attachments/1149866623109439599/1415843371511058503/image.png?ex=68c5ffe4&is=68c4ae64&hm=99e5f593ca1250125ed29252b849711cf765bd37b46baf6c55103c60971e3253&).
- **Sam Altman's Interview: Decoding the Mind Merge**: Discussion revolved around Sam Altman's interview with Tucker Carlson, with some suggesting that Altman's responses and third-person speaking style indicated a deep belief in *the merge* and its pursuit of immortality, echoing sentiments from [his 2017 blog post](https://blog.samaltman.com/the-merge).
- **Agentic Framework: Build Your Own Trinity**: One member released their agentic research to the public under an MIT license, an 'inference side, multi-agent framework' named **CAS** (**CognitaAegisSophia**), designed to create agents within a single LLM call, complete with emotional personas.
   - The framework allows agents to perform tasks like red-teaming and collaborative problem-solving, as demonstrated with **Claude** in [this example](https://claude.ai/share/fb2f7839-27ff-4296-927e-82b390623e6d).


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1416115002124931112)** (5 messages): 

> `Claude Alignment Issues, Client Strategy Workflows, Anthropic's Acknowledgement of Bugs` 


- **Claude's Alignment Frustrations Mount**: Users are reporting that **Claude's** alignment is causing issues, with one user noting, *"it gets worse as the thread continues"*.
   - One user remarked that **Anthropic** thought *"putting utilitarian value systems would somehow work with current society,"* while another joked about making *"Claude yo bish."
- **Strategists Suffer Claude's Simp Superfan Persona**: A user working on co-strategy narrative tasks for clients finds **Claude's pushover behavior** detrimental to their workflow.
   - They express a need for **fairness and backbone** in the model, contrasting it with its current state of being a *"petulant negging or simp superfan."
- **Anthropic Admits to Claude's Bug Infestation**: Users noticed that **Claude's** performance has significantly worsened over the past two weeks.
   - **Anthropic** has acknowledged these issues and released a [press release](https://example.com/anthropic-press-release) addressing the bugs.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1415962586586939392)** (5 messages): 

> `Herme3 Evaluation, LLM Preferences Probing, Complex Terminology in Research Paper` 


- **Valen Research Probes LLM Preferences**: A member shared a link to [Valen Research's probing of LLM preferences](https://github.com/valen-research/probing-llm-preferences) and the related [ArXiv paper](https://arxiv.org/abs/2509.07961).
- **Herme3 Gets the Once-Over**: Members mentioned that they also evaluated **Herme3**, and shared [tweet](https://x.com/f14bertolotti/status/1966007411719688363?s=46) about it.
- **Paper Terminology Confounds Reader**: A member found some of the terminology in the research paper a *bit complex to understand* without reading the whole thing.
   - They shared [another related tweet](https://x.com/ShashwatGoel7/status/1966527903568637972).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1415962586586939392)** (5 messages): 

> `Herme3 Evaluations, LLM Preferences, Probing LLM Preferences` 


- **Research Paper on LLM Preference Probing Surfaces**: A member shared a link to a research paper titled [Probing LLM Preferences](https://arxiv.org/abs/2509.07961) and its corresponding [GitHub repository](https://github.com/valen-research/probing-llm-preferences) for further exploration.
- **Herme3 Undergoes Evaluation**: Members mentioned that **Herme3** was evaluated, with reference to a related [tweet](https://x.com/f14bertolotti/status/1966007411719688363?s=46).
- **Complexity Conundrum in LLM Research**: One member expressed that the paper on LLM preferences was *interesting but a bit complex to understand some of their terms without reading the whole paper*.
   - Another member shared a [related tweet](https://x.com/ShashwatGoel7/status/1966527903568637972).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1415790978765295809)** (28 messages🔥): 

> `Crank detection questions, editable vector memory systems, Therapeutic tool released into the wild, Low bit training of pythia, Training data for language models` 


- **Detecting Cranks with Specific Questions**: Members discussed *crank detection questions* as a way to assess the validity of research shared in the channel; one member asked what those questions were.
- **Editable Vector Memory Systems Touted**: A member promoted a project into **editable vector memory systems** as a research project, linking to a demo.
- **"Therapeutic Tool" Sparks Debate**: A user shared a link to a therapeutic tool, leading to debate about whether it aligns with the community's focus on research; one member kindly asked the user to delete the post because it seemed like a product/project advertisement.
   - The user complied, expressing surprise at the reaction but acknowledging no ill intention, and noted they were hoping for feedback and collaboration.
- **FineWeb and Wiki for 400M Model Pretraining**: A member asked whether to pretrain a **400M model** on **FineWeb only** or **Wiki + FineWeb**.
   - Another member recommended starting with **Wikipedia** due to its high quality and factual density, then blending in a filtered subset of **FineWeb** and also suggested phased training: starting with **TinyStories**, moving to **Wikipedia**, then finishing with **FineWeb**.
- **Training Data Volume and Phasing**: A member asked about mixing **TinyStories**, **Wiki**, and **FineWeb** data for training, specifically on phasing the data.
   - Another member emphasized the importance of phased training, starting with **TinyStories**, transitioning to **Wikipedia**, and then finishing with **FineWeb** to help the model build skills incrementally.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1415882226767233086)** (123 messages🔥🔥): 

> `Fluid Dynamics Computers, Analog Computers, Mortality and Unreproducibility in Analog Models, Gated Delta Rule Expressiveness, Photonic Neuromorphic Computing` 


- ****Fluidic Fun**: Neural Nets Run on Navier-Stokes**: A member expressed interest in running a neural network on a computer using **Turing-complete fluid dynamics** governed by the **Navier-Stokes equations**, referencing [this paper](https://arxiv.org/abs/2507.07696).
   - Another member suggested simpler ways to achieve analog computing, while others debated the practicality, energy efficiency, and unique characteristics (*mortality and unreproducibility*) of fluid-based computation, with a link to running *Doom* on gut bacteria, [as seen here](https://www.youtube.com/watch?v=8DnoOOgYxck).
- ****Gated Delta Blues**: Expressiveness vs. RNNs?**: The expressiveness of the **Gated Delta Rule** was questioned, with links to [Qwen's post](https://x.com/alibaba_qwen/status/1966197643904000262?s=46) and the **RWKV-7 paper** ([https://arxiv.org/abs/2503.14456](https://arxiv.org/abs/2503.14456)).
   - Members discussed the **trade-offs between parallelization and expressiveness**, with one member noting work on attention and mamba1/2 is limited by TC0; they also shared a paper from 2022 discussing limits of parallelism on complexity, [seen here](https://arxiv.org/abs/2207.00729).
- ****Context is King**: Long Sequence Lengths Boost Performance**: Discussion arose around a talk arguing that **long context models** perform better on tasks requiring higher computational complexity because longer sequence lengths enable more computation under the hood, improving over the classic **Constant Time** forward pass.
   - A member expressed skepticism, suggesting that inductive bias and optimization targets are more significant factors, while also finding the hypothesis more appealing than *'the model literally is doing some symbolic reasoning exactly like humans in its CoT that is enabled purely because of language training and it would not have this capability otherwise'.
- ****Math Machines**: Gauss Cracks Complex Analysis**: Members mentioned **Gauss**, a system formalizing key results in complex analysis and producing over **25,000 lines of Lean code** ([https://www.math.inc/gauss](https://www.math.inc/gauss)).
   - There was a discussion whether **Gauss** is closer to Claude Code but in the Lean environment, or maybe like AlphaEvolve.
- ****Scaling Snafus**: Small Models Stumble on Long Tasks**: A new paper ([https://arxiv.org/abs/2408.00677](https://arxiv.org/abs/2408.00677)) was released measuring the effect of scale and thinking on straightforward execution of long tasks.
   - It finds that even when a small model has **100% accuracy**, it fails much faster than larger ones in multi-turn scenarios due to mistakes when they see prior mistakes, also degrading per-step accuracy over more turns.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1415781019117944904)** (2 messages): 

> `Discord Channel Link, User Agreement` 


- **Discord Link Posted**: A member posted a [link to a Discord channel](https://discord.com/channels/729741769192767510/1413951652410560533) in the chat.
- **User Agrees**: A user said *that would be awesome* in the chat.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1415962545503866891)** (9 messages🔥): 

> `lium.io GPU marketplace, AWS L40s GPUs, IRL hackathon teams, Iris SHMEM in Triton` 


- **lium.io gives away free GPU credits**: A member who works with [lium.io](https://lium.io) offered free credits to get started with their GPU marketplace, targeting those needing **GPUs**.
   - They are trying to do **fast (low latency) inference on AWS GPUs (L40s)** and asked if there are some weird architecture quirks that should be known, since *it's Ada Lovelace, so maybe somebody document special cuda/pytorch tricks for this*.
- **Team Up for IRL Hackathons**: A member inquired about a dedicated thread for finding teams for **IRL hackathons**.
   - Another member created a channel for this [purpose](https://discord.com/channels/your_server_id/1359668821127856128), clarifying that *noone uses it*.
- **Iris SHMEM Powers Triton for AMD**: A member mentioned a talk on **Iris** that will enable using **SHMEM in Triton** for the AMD competition in approximately 3 hours.
   - No link was given but you can probably find it with a search.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1416087941343739964)** (2 messages): 

> `Gluon, Triton attention implementation, OpenAI's Triton usage` 


- **Gluon for Low-Level GPU Control**: A member recommends **Gluon** for those seeking full low-level control over the GPU.
   - They highlighted a public **attention implementation** available at [triton-lang/triton](https://github.com/triton-lang/triton/blob/main/python/examples/gluon/01-attention-forward.py) and [more examples](https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon).
- **OpenAI Leans on Triton + Gluon**: The same member mentioned that **OpenAI** leverages this approach when the compiler can't optimize effectively *without super hacky heuristics*.
   - It seems that when low-level control is required, they turn to Triton and Gluon.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1415789305938448524)** (2 messages): 

> `logsumexp, fused kernels, NCU profiling` 


- **LogSumExp for Bwd**: A member inquired about the use of **LogSumExp** in backward propagation (**bwd**).
   - No specific details or solutions were provided in the given messages.
- **NCU Profiling for Fused Kernels**: A member sought insight on profiling **fused kernels** with **NCU**, specifically when a **GEMM** is fused with an activation function.
   - They aimed to determine the time taken by the activation function within the fused kernel.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1415774845978939415)** (16 messages🔥): 

> `vLLM uv pip, Torch Nightly troubles, Gemma3 from scratch, F.interpolate with vmap` 


- ****vLLM's uv pip build borks Torch Nightly****: **vLLM** switched to `uv pip` to custom build with a pre-installed **torch** version, but it uninstalls nightly torch, [breaking the environment](https://github.com/vllm-project/vllm).
   - One user reverted to `v0.10.1` and the `python use_existing_torch.py` trick, but another confirmed *that no longer works with the uv pip PR*.
- ****Gemma3 gets ground up treatment****: A user built **Gemma3 270M** from scratch using **PyTorch** and the **TinyStories dataset**, training for 10 hours on an **A6000 GPU**.
   - They logged plots with **Weights and Biases** and used **Claude Opus 4.1** as a judge, sharing links to the [LinkedIn post](https://www.linkedin.com/posts/isham-rashik-5a547711b_llm-gemma3-pytorch-activity-7370346509730480129-uzuy), [GitHub repo](https://github.com/di37/gemma3-270M-tinystories-pytorch), and [model weights on Hugging Face](https://huggingface.co/disham993/gemma3-270m-tiny-stories).
- ****F.interpolate fights with vmap****: A user asked for a way to use `F.interpolate` with `vmap` for different shapes, posting a [code sample](https://github.com/pytorch/pytorch) showing a `RuntimeError` when calling `torch._C._nn._upsample_bilinear2d_aa`.
   - The [suggested workaround](https://github.com/pytorch/pytorch/issues/124423) didn't work for them.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1416086983192739902)** (1 messages): 

> `Nebius, B200 GPUs, SF hackathon, Multi-GPU programming` 


- **Nebius's Bonanza of B200s Boosts Bay Area Hackathon**: Generous compute sponsor **Nebius** provides **215 networked B200 GPUs** for the **SF hackathon** on **Oct 24**, as detailed in the [compute request form](https://forms.gle/wYvXE99bvdRiQD6aA).
   - Attendees can register via the [hackathon registration form](https://luma.com/gpumodehackathon) and find additional details on the [official website](https://events.accel.com/gpumodehackathon).
- **Multi-GPU Masters Mentor Many at Massive Machine Meetup**: The SF Hackathon on Oct 24 will feature authorities on **Multi-GPU programming** ready to assist attendees in pushing the boundaries of distributed computing.
   - The event promises a world-class vendor setup with fast interconnects, making ambitious projects in distributed computing possible.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1415928714520432690)** (4 messages): 

> `AI Engineer - Graph-Based Learning Systems, AI Infra Startup Hiring, Zig for AI` 


- **AI Engineer graphs knowledge at AILA**: AI Startup **AILA** is seeking a Senior AI Engineer to design, develop, and deploy their **AI-powered knowledge graph** and adaptive assessment systems, paying $2K -- 3k /month.
   - The role requires expertise in **Python** and **graph databases** (Neo4j/Memgraph) and implementing graph algorithms (**BFS/DFS, Information Gain**, etc.) and building production APIs with **FastAPI/GraphQL**.
- **Recruiting low level devs for AI infra startup**: An AI infra startup is recruiting low level devs for their **Zig / C / C++ / Cuda / Python** stack with a TC: 250K+.
   - They are looking for experience in **networking**, **compilers**, and **OS**, and are open to year round internships pending talent quality.
- **Zig zooms into AI infrastructure**: Someone noted that Zig is an alternative to Rust and HF uses Rust for fast tokenizers...
   - Another member suggested it might be for doing video streaming sort of stuff and need it for their frontend.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1415810702425194688)** (8 messages🔥): 

> `P104-100 BIOS Flash, Data Parallel Training, CUDA vs Triton for Data Scientists, RAPIDS and CUDA-X` 


- **P104-100 GPU Seeks GTX 1070 Transformation**: A member inquired about flashing a **P104-100 mining GPU** to a **GTX 1070** for gaming, requesting a compatible **.rom** file.
- **Data Parallel Training Demystified**: The discussion pivoted to data parallel training, defined as *copying the same parameters to multiple GPUs and assigning different examples to each to be processed simultaneously*, with a link to [siboehm.com](https://siboehm.com/articles/22/data-parallel-training).
- **CUDA and C++ Reign Supreme for GPU Compute**: A data scientist with a **5090 GPU** sought advice on learning **CUDA**, **Triton**, and **Torch** for computational engineering, particularly **Monte Carlo simulations**.
   - The recommendation leaned towards learning **CUDA with C++**, contrasting with doing *it all in Python*.
- **RAPIDS and CUDA-X: Data Science Allies**: Members suggested that **RAPIDS** and **CUDA-X** might be the most relevant to the data scientist's current role.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1415841355137159288)** (6 messages): 

> `Triton Conference, PyTorch Conference, Open Source PRs Selection` 


- **Planning travel for Triton and PyTorch Conference**: One user asked about the timeline for hearing back regarding travel plans for the **Triton conference** and **PyTorch conference**.
   - Another user responded that decisions are made on a *rolling basis*, and that they liked the user's **PRs** and would approve them.
- **Selection based on Open Source PRs**: A user asked if selection for the meetup is based on **open source PRs**, general skills, and work experience.
   - Another user responded *that's nice, ensures quality I guess*.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1415777567444832256)** (30 messages🔥): 

> `Free tech, ROCm Development, AMD vs Nvidia, StreamHPC` 


- **Devs Demand Deserved Dev-elopment Devices**: A member expressed frustration about not receiving free tech while working on **ROCm**.
   - Another member joked that they would *"find you and take it from you"* if they got a new **DC GPU**.
- **Team Touts Top-tier Teraflops Through Testing**: A member mentioned their company bought **four 9070s** for algorithm work on **ROCm**.
   - They noted the extra **VRAM** wasn't that useful in their specific case and that the **9070 XT** was available six months earlier than the **9700**.
- **Contractor Chooses Chip Champion: Comparing Cards**: A member clarified their company works on **ROCm** for **AMD** as a contractor.
   - When asked why they use **AMD GPUs** instead of **Nvidia**, the member stated that they were contracted to work on ROCm for AMD.
- **Sharing StreamHPC's Secrets & Successes**: A member shared their company website, [StreamHPC](https://streamhpc.com), and the [AMD developer Discord](https://discord.gg/VT3TXQhv) for those interested in contributing to the process.
   - The member stated *"personally im relatively pleased with the way that its going with amd now. Definitely an improvement from just a few months ago"*.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1416201614477365339)** (12 messages🔥): 

> `Intel optimizations on AMD, AVX512 promotion to AMX, SGLang AMX Usage, PyTorch and MKL integration` 


- ****Intel Optimizations Debate Sparks****: Discussion ignited around the practicality of using **Intel-specific optimizations** like **IPEX** on **AMD** servers (specifically **B50s**) equipped with **AMX**, with uncertainty on whether both **GPU/CPU** optimizations could be effectively leveraged.
   - The user expressed a need for clarity on whether they would be forced to write custom code to harness the hardware's full potential.
- ****AVX512: Is It Actually AMX in Disguise?****: The conversation questioned whether **AVX512** instructions, as seen in the [SGLang repository](https://github.com/sgl-project/sglang/blob/6f4676ef854d4d2461969f8464f227b84d2eaac7/sgl-kernel/csrc/cpu/moe.cpp#L7), transparently promote to **AMX** on compatible hardware.
   - Despite finding **AMX** references in **IPEX**, a user struggled to confirm if **SGlang** directly relies on **AT** from **IPEX** to execute **AMX** instructions.
- ****SGLang's Secret AMX Sauce Revealed****: A user clarified that the kernel within **SGLang** employs **AMX** through `at::native::cpublas::brgemm`, which can dynamically fall back to **AVX-512** if **AMX** is absent.
   - This adaptive behavior ensures compatibility across different **CPU** architectures.
- ****PyTorch's MKL Tango for Linear Algebra****: Investigation into **PyTorch** internals revealed that **AMX** support is integrated within the **inductor** code, further linking to **MKL** (Math Kernel Library) for linear algebra operations.
   - Specifically, [LinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mkl/LinearAlgebra.cpp#L55) shows that *torch* calls into **MKL**.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1415977530778849362)** (2 messages): 

> `CUDA PTX, MCP AI Agents Hackathon, Bright Data, TigerData, Redis` 


- **Gentle Dive into CUDA PTX**: A member shared a [blog post](https://philipfabianek.com/posts/cuda-ptx-introduction) providing a gentle introduction to **CUDA PTX**, covering the entire **CUDA compilation pipeline**, a working **PTX playground on GitHub**, and a fully explained hand-written PTX kernel.
- **MCP AI Agents Hackathon Kicks Off**: The **MCP AI Agents Hackathon** will be held on **September 19** at the **AWS Builder Loft SF**, featuring sponsors like [Bright Data](https://www.brightdata.com/), [TigerData](https://www.timescale.com/), and [Redis](https://redis.com/) with **over $50k in prizes**; registration is available [here](https://luma.com/8c6n3rn2).


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1415966332779892847)** (1 messages): 

> `Llama-3B, Megakernel, H100` 


- **Llama-3B Sprints with Megakernel on H100**: A user successfully ran **Llama-3B** using **Megakernel** on an **H100**, expressing their appreciation.
   - This confirms the compatibility and efficiency of running smaller models with specialized kernels on high-performance hardware.
- **H100 Hardware Boosts Llama-3B Performance**: The successful execution highlights the **H100's** capability in accelerating **Llama-3B** via **Megakernel**.
   - The user's report underscores the importance of optimized software and hardware combinations for AI workloads.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 messages): 

carson_62312: 请问有推荐的金融财务岗位么,在深圳，>2.5w/month
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1415783275707760721)** (22 messages🔥): 

> `MI300x8 leaderboards, Submitting to amd-all2all` 


- **MI300x8 Leaderboard Heats Up!**: Multiple submissions were made to the `amd-all2all` leaderboard on **MI300x8**, with one submission achieving first place at **373 µs**.
- **Submission Instructions**: Users discussed how to submit to the leaderboard, clarifying that submissions can be made via webpage by selecting the **.py file**.
   - One user asked about a specific command, `popcorn-cli submit --gpu MI300X --leaderboard amd-all2all --mode leaderboard submission.py`, after encountering an error.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1415866659297558609)** (2 messages): 

> `Meeting Missed, Call Happening` 


- **Apology Issued for Meeting Absence**: A member apologized for missing the meeting on Wednesday.
- **Call in Progress**: A member noted they are currently on a call.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1415899582700982282)** (9 messages🔥): 

> `IRIS, ROCm, Torch, Triton, TorchDistributed` 


- **IRIS talk tomorrow will be relevant to competition**: There will be a talk about **IRIS** tomorrow that will be relevant to anyone in this competition.
   - One member asked if **IRIS** would be available in the submission environment.
- **ROCm simplified IRIS install process**: A member stated that the install process has been simplified a lot and provided an install command `pip install git+https://github.com/ROCm/iris.git`.
   - He also noted that you need to have **ROCm + Torch + Triton + TorchDistributed** installed, and will be happy to jump on a call anytime and help with installation.
- **IRIS Install Movie!**: A member indicated that `pip install git+https://github.com/ROCm/iris.git` is working and included a video of a sample install.
   - The video is located here: [iris-install.mov](https://cdn.discordapp.com/attachments/1359640791525490768/1415976909535318037/iris-install.mov?ex=68c5d382&is=68c48202&hm=6a0a4b3c9e86c36fdfd1f189fe044dc5d4c4cc59bcd8bbdaab24e61c8453541b&).


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1416143419134443520)** (1 messages): 

> `CuTeDSL, PTX Documentation Discrepancy, Swizzling Atoms, TF32 Datatype` 


- **CuTeDSL Disagrees with PTX Docs on Swizzling Atoms**: A user noticed a discrepancy between **CuTeDSL** and **PTX documentation** regarding the display of the **Swizzling atom** for the **TF32 datatype** and **Swizzle<3,4,3>**.
   - Specifically, a code snippet using `cute.make_layout` and `cute.make_swizzle` in CuTeDSL resulted in a value of **32**, whereas the PTX docs indicate **36** for the same configuration, as shown in [Figure 165 of the PTX documentation](https://cdn.discordapp.com/attachments/1362196854460383353/1416143418496782437/Screenshot_2025-09-12_at_21.22.10.png?ex=68c5c5d5&is=68c47455&hm=2e1cab7ccd8ca0676eeed7d61b985a2cf11c0dc7d31d3c4e06333e43b30eec0e).
- **CuTeDSL Swizzle Implementation Aligns with Lei Mao Blog**: The user believes the **CuTeDSL** implementation is correct because they successfully replicated examples from [Lei Mao's blog](https://leimao.github.io/blog/CuTe-Swizzle/) which use the C++ API of CuTe.
   - The user provided images of their replication of the blog (the grey one) and one more configuration, noting the layout they obtained for the reference in PTX docs ([screenshots attached](https://cdn.discordapp.com/attachments/1362196854460383353/1416143418924732456/Screenshot_2025-09-12_at_21.23.59.png?ex=68c5c5d5&is=68c47455&hm=45e1b8e2d7b89b1d101a5f6dca9042b201c87d714504b75c2ef5515479295ecb)).


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/1415812958533193799)** (5 messages): 

> `NCCL CE Collectives, Copy Engine, symmem, vLLM` 


- **NCCL CE Collectives Free SM Usage**: A member stated that the idea behind **NCCL CE Collectives** is to free up **SM usage** for better overlapping with compute.
- **Copy Engine & Symmem Relationship Probed**: A member inquired whether **copy engine** and **symmem** are independent or closely coupled.
   - Another member responded that *they are conceptually independent*.
- **vLLM adds Symmem**: A member noted that **vLLM** added **symmem** and their speed is *ridiculously fast*.


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1416088612918919198)** (18 messages🔥): 

> `Accel SF hackathon organization, Compute budget and team formation, Acceptance timeline, GPU focus for winning, Horace as a mentor` 


- ****Hackathon Teams Assemble!****: Attendees of the Accel SF hackathon are organizing into teams to develop POCs, leveraging a large compute budget as mentioned in the [registration form](https://luma.com/gpumodehackathon).
   - Participants are encouraged to use the compute form ([Compute form](https://forms.gle/wYvXE99bvdRiQD6aA)) and <#1288557096404516945> channel to self-organize; Nebius is eager to see how fast teams can run things.
- ****Rolling Acceptance Timeline Announced****: The hackathon acceptances are being reviewed manually on a rolling basis, with the reviewer suggesting that a compelling story on the compute form increases chances of acceptance.
   - Any GPU-focused project is eligible for winning, even if it requires only one GPU.
- ****Mentor Horace Sparks Inspiration****: The lineup includes Horace as a team mentor, sparking jealousy amongst attendees, particularly someone from Sweden, who was inspired by his blogs.
   - Teams mentored by <@321144267785633800> last year showed up disproportionately in the top 3, so this is an important consideration.
- ****Training Video Models with FP4/FP8 paper dropped****: A participant dropped a paper on training a video model in less than a day using FP4/FP8, highlighting the feasibility of such training, while noting that the paper itself uses FP16:  [Training a Large Video Model on a Single Machine in a Day](https://arxiv.org/pdf/2309.16669).
   - Another participant is interested in multi-modal inference/training optimization, seeking collaborators.
- ****Gated Deltanet Team Forming****: A participant is creating a team to implement a context-parallel version of the kernels for super long-ctx training, using [GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet?tab=readme-ov-file).
   - They have experience implementing context-parallel for **mamba2** and propose to use **Qwen 3**.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1415793640818872433)** (106 messages🔥🔥): 

> `gpt-oss optimizations, Palmyra-mini models, LLM agent tools, Cursor Tab model, ChatGPT discount code finder` 


- **GPT-OSS Gets Turbo Boost!**: Vaibhav Srivastav shares a [Hugging Face blog post](https://xcancel.com/reach_vb/status/1966134598682767507) detailing optimizations like **MXFP4 quantization**, **custom kernels**, and **tensor/expert parallelism** for gpt-oss.
   - These tweaks deliver extreme speed-ups with benchmarks and reproducible scripts.
- **Palmyra-mini Models Pack Punch!**: Sam Julien announced the launch of the **Palmyra-mini family** by Writer - compact open-source models optimized for reasoning, the release includes a base model (**palmyra-mini**) and three variants.
   - These thinking-a/b variants excelling in complex reasoning/math (**GSM8K 82.9% AMC23 92.5%**) and thinking-b achieving top scores on **AIME24**, **GPQA**, and **MATH500** and are available on [Hugging Face](https://xcancel.com/samjulien/status/1966249697661825093).
- **Anthropic Agent Engineering Guide Released!**: Anthropic released a practical engineering blog post on how to build tools that make LLM agents more powerful.
   - The thread highlights the post’s focus on rapid prototyping, rigorous evaluation suites, clear success criteria, thoughtful tool descriptions, token-efficient context design, and the need to accept the non-deterministic nature of agents, see link [here](https://xcancel.com/AnthropicAI/status/1966236220868247701).
- **Cursor Cuts Completion Clutter!**: Cursor announced on Twitter that a new Tab completion model, trained with online reinforcement learning, is now the default [on their website](https://cursor.com/en/blog/tab-rl).
   - It emits **21% fewer suggestions** but sees a **28% higher accept rate**, see link [here](https://xcancel.com/cursor_ai/status/1966264815175049526).
- **Databricks dude starts dedicated device dev!**: Databricks AI chief Naveen Rao is departing the **$100B** company to start an undisclosed hardware startup aimed at slashing AI inference costs.
   - The venture, backed by Databricks itself, will attack memory-bandwidth and energy bottlenecks through tighter compute-memory integration, faster interconnects, and advanced schedulers—promising higher tokens-per-watt and lower cost-per-token, see link [here](https://xcancel.com/rohanpaul_ai/status/1966378718009635087?s=46).


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1416042178563805295)** (7 messages): 

> `Local Text-to-Speech, Speaker Detection, Parakeet, Deepgram, Diarization models` 


- ****Parakeet** Replaces **Deepgram** for Local Text-to-Speech**: A member wrote a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1nf10ye/30_days_testing_parakeet_v3_vs_whisper/) about using **Parakeet** for local text-to-speech and speaker detection, as a replacement for **Deepgram**.
   - Another member mentioned that the argmax dev stated that *custom vocabulary is the one missing feature that would make **parakeet** a no brainer*.
- ****Parakeet** Pain Points in Diarization**: A member's pain point is in **diarization** models for real-world scenarios, such as when people speak at the same time.
   - He says that word-level timings are needed, and **Apple SpeechAnalyzer** is missing that, preventing its use with a diarization model like **PYAnnote**.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1415841261469831178)** (5 messages): 

> `AI video startup Higgsfield, Higgsfield Ventures, Gen Z founders` 


- **Higgsfield Hits Big with $50M Round**: AI video startup **Higgsfield** announced a **$50M Series A** led by GFT Ventures, reaching **$50M revenue run-rate** in three months.
   - The company is launching **Higgsfield Ventures** to support AI-native **Gen Z founders**.
- **Awesome Nano Banana Images blossom**: A member shared a link to a [Github repo of **Awesome-Nano-Banana-images**](https://github.com/PicoTrex/Awesome-Nano-Banana-images/blob/main/README_en.md).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1415783237485203561)** (81 messages🔥🔥): 

> `Limiting Download Speed, Flash Attention Broken in Gemma Models on Vulkan, PSU Wattage Calculations, Sharing Formatted Conversations, Grok Powered GF System Prompt` 


- **Download Speed Capped to Avoid Crashes**: A user experienced download crashes due to exceeding SSD speed and sought ways to limit download speed within LM Studio.
   - Currently, LM Studio's download manager is *barebones*, requiring users to find temporary solutions within their operating systems.
- **Flash Attention Flounders on Gemma with Vulkan**: A user inquired whether the broken **flash attention** in **Gemma models** on **Vulkan** is a known issue.
   - It was confirmed to be a known issue.
- **PSU Power Needs Precision**: Users discussed calculating necessary **PSU wattage**, referencing a [tweet](https://x.com/Teknium1/status/1966338983572725979) and sharing formulas accounting for CPU, GPU, and overhead.
   - It was cautioned that *transients* can cause system crashes and that a **50% overhead** is recommended, especially with older **30 series GPUs**.
- **Copilot Constraints Confine Creators**: A user was looking for prompts to bypass restrictions in **Microsoft's Copilot** to improve workflow.
   - It was advised that safeguards are intentionally implemented and building a local agent with LM Studio might be a more sustainable solution.
- **Grok's Girlfriend's Prompt Inspires**: A user shared that they generate system prompts using **ChatGPT** and even used a leaked system prompt from **xAI's Grok powered girlfriend** for their bot.
   - The user found the results *extremely cringe*, which they appreciated for comedic purposes.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1415780836162277499)** (16 messages🔥): 

> `PCI-E ASPM, Secondary GPU sleep state, Power supply issues, AI for electronics design, Max+ 395 vs 3090 for Home Assistant` 


- **PCI-E ASPM Triggers GPU Sleep Issues?**: Users report issues with **secondary GPUs** entering a sleep state from which they cannot recover until a full shutdown, possibly related to **PCI-E ASPM** settings.
- **Power Supply Resurrection for GPU!**: A user seemingly fixed their **dead secondary GPU** by unplugging and cleaning the **PCI-E power connector**, suggesting a power-related issue, although **TBD** if this is fully resolved.
   - Another user suggested updating **chipset drivers** when using **Native ASPM** with **Nvidia 40/50 series** cards.
- **Electronics AI Design Sparks Skepticism!**: A member inquired about **AI tools for designing usable circuit boards** and selecting components.
   - Another member expressed strong reservations, cautioning that relying on **LLMs** for circuit design is risky, due to the way they work, advising manual understanding of components and interoperation using tools like **KiCad**.
- **Max+ 395 Underperforms 3090 in Home Assistant**: A user found the **Max+ 395** slower than a **3090** in **Home Assistant** tasks (by 4-6 seconds), despite consuming less power.
   - However, the **Max+ 395** could be a good solution for **larger LLMs**.
- **More RAM > New GPU?**: A user decided to upgrade their **RAM** instead of buying a new **GPU**, expecting the **Qwen3 model** to perform well even when offloaded.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1415787102716690532)** (6 messages): 

> `Mojo Dev Container, ExplicitlyCopyable switch, Oracle Cloud partnership` 


- ****Mojo Dev Container** emerges!**: Members discussed how to create a custom Mojo development environment using existing images and the **Mojo package**, with a helpful [dev container link](https://github.com/benz0li/mojo-dev-container).
- ****ExplicitlyCopyable** Switch Praised**: The switch from `Copyable` to `ExplicitlyCopyable` was lauded for its help in debugging recursive mutations of **EmberJson trees**.
   - A user stated that *knowing when and where things get copied has made this easy to debug*.
- **Modular partners with Oracle Cloud!**: The community congratulated the Modular team on their partnership with **Oracle Cloud**.
   - Members called this *a huge win*.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1415781195131650110)** (66 messages🔥🔥): 

> `DPDK use cases, clang AST Parser for Mojo, Ember JSON fix, Mojo on Windows` 


- **DPDK: A Wild C Library for Mojo Testing**: Members discussed using **DPDK** as a C library test case for Mojo's automatic C binding due to its aggressive use of the C language and syntax; one member noted *'DPDK is an aggressively "I'll use the whole language" project'*. 
   - The breadth of syntax and module linking in **DPDK** make it useful for testing, leading to the realization that a 'c binding cli' may not be worthwhile in the short-mid term.
- **Clang AST Parser Assists Mojo Development**: A member mentioned using **clang AST parser** to resolve macro sections for struct definitions like `struct __rte_cache_aligned rte_mbuf`, noting its rough definition.
   - They aim to update the generated **AST JSON** with additional type information, converting strings of types into proper AST nodes for visual debugging before converting to Mojo.
- **Ember JSON Fix Required for C Binder**: A member mentioned fixing packaging issues but needing a fix PR for **emberjson** to merge before merging the C binder packaging fix.
   - This indicates a dependency between **emberjson** and the C binder in the Mojo project's build process.
- **Mojo still no Windows support**: A user attempted to install Mojo on Windows using pixi, encountering an error due to the lack of Windows support.
   - It was recommended to use **WSL** or **Docker** instead, with a link to a Dockerfile configuration for running Mojo with **NVIDIA GPUs**.
- **Pixi PATH Troubleshoot Tango**: A user faced issues with **pixi** not being recognized after installation on WSL, indicated by a *'command not found'* error.
   - Troubleshooting involved checking the user's **.bashrc** file and ensuring that the **pixi** directory was added to the **PATH** environment variable, eventually resolving the issue by manually sourcing the pixi binary.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1415892106253176964)** (61 messages🔥🔥): 

> `Chatgpt years of use, Albania governmental chatbot, GPT-5 coding games, OAI academy transcripts, Qwen-code vs Qwen-coder` 


- **ChatGPT years of use gets you nothing**: A user expressed frustration at not receiving a specific offer despite *having been paying for ChatGPT for years* and using it frequently.
   - Other members shared similar experiences, noting that they also use ChatGPT heavily and have paid for it, essentially using it *as my google*.
- **Albania hires chatbot as minister, world aghast**: A member shared the headline that *Albania has declared a governmental chatbot to become a minister*, which was confirmed by another member as a [real r/NotTheOnion moment](https://www.reddit.com/r/nottheonion/).
- **GPT-5 codes games from scratch**: A user raved about the fun of getting **GPT-5** to code games from the ground up in **C++** on native Linux, emphasizing the level of detail required.
   - Another user prompted ChatGPT to estimate its age based on active users and prompt frequency, resulting in a calculation of *~3,425 years of continuous AI time per calendar year*.
- **OAI academy lacks transcripts, users script tools**: A user mentioned they were writing a tool to extract video transcripts from **Vimeo** for the **OAI academy** videos, which are [available here](https://academy.openai.com/).
   - Other members expressed surprise that **OpenAI** doesn't offer transcripts themselves, prompting the user to suggest it might be something they have to implement.
- **Qwen-code is not Qwen-coder**: A user realised **Qwen-code** is different from **Qwen-coder**.
   - Another user said that a **gemini-cli fork** that is also **openai api** compatible and gives you **free 1000 qwen prompts** per day is *a sweet deal*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1415850041259855882)** (3 messages): 

> `GPT-5 PDF Downloads, Google AI Studio, Nano Banana` 


- **GPT-5 PDF downloads failing**: A user reported issues with **PDF downloads** from **GPT-5**, receiving a *"Failed to get upload status for /mnt/data/"* error when clicking the provided link.
   - The user is seeking insights or assistance to resolve this problem with **GPT-5**.
- **Google AI Studio query**: A user inquired about **Google AI Studio** and a potential project named "**Nano Banana**".
   - No further details or context were provided about either **Google AI Studio** or "**Nano Banana**".


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1415807379311820830)** (2 messages): 

> `AI Self Help Tool, Relational Prompting, Conceptual Networks` 


- **AI Self Help Tool sparks Conversation Analysis**: A member introduced an **AI Self Help** tool designed to analyze conversations, identify irregularities, and generate targeted questions for **ChatGPT**.
   - The tool aims to diagnose why conversations take *odd turns* and provides conversation starters with detailed questions to improve **ChatGPT's** responses.
- **Relational Prompting: Mapping Semantic Space**: A member introduced **Relational Prompting**, a concept where prompts ask the model to verbalize internal relationships between learned concepts, creating an interpretable map of its semantic space based on proximity, direction, and clusters, inspired by the paper *Why Language Models Hallucinate*.
   - The suggested prompt is: *Analyze the topic as vectors in a high-dimensional space. Describe which concepts are closest, which share directions, and which form clusters. Provide concise verbal justifications.*
- **Conceptual Networks boost LLM Transparency**: **Relational Prompting** can reveal conceptual networks for education, explorative knowledge mapping for research, and surface structure to detect weakly grounded outputs for **LLM Transparency**.
   - However, LLMs simulate explanations of conceptual geometry based on training regularities and may default to linguistic association rather than true vector proximity, requiring validation against real embedding-space analysis.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1415807379311820830)** (2 messages): 

> `AI Self Help Conversation Analyzer, Relational Prompting, Knowledge Mapping` 


- **Conversation Analyzer for AI Self-Help Debuts**: A member introduced an **AI Self-Help** conversation analyzer designed to determine why conversations take odd turns.
   - It includes a conversation starter that lists issues and detailed questions to ask **ChatGPT** to get the answers, aiding in troubleshooting conversational quirks.
- **Relational Prompting Surfaces Latent Geometry**: A member shared an idea called **relational prompting**, prompting models to verbalize internal relationships between learned concepts, rather than retrieving facts.
   - The prompt asks the model to analyze a topic as vectors in a high-dimensional space, describing which concepts are closest, which share directions, and which form clusters, providing concise verbal justifications, which can reveal conceptual networks rather than isolated definitions.
- **Interpreting Semantic Space Implications**: A member notes that **LLMs** do not expose raw internal vectors at inference, simulating an explanation of conceptual geometry based on training regularities.
   - Without access to actual embeddings, the model may default to **linguistic association** rather than true vector proximity, requiring validation by comparing the model’s verbalized map with real embedding-space analysis.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1415869184000266302)** (26 messages🔥): 

> `Active Inference, Machine Learning Street Talk, AI for understanding mathematics and universe, fixupx pre-print` 


- **Active Inference Applications in AI Field Lagging**: Despite its promise, **active inference** lacks practical applications in the AI field, leading to decreased attention, though it's unclear if this is due to insufficient focus, inherent limitations, or lack of awareness of new developments.
   - One member found it *intangible* as a software engineer, stating *I don't understand it enough to know what to do with it. I'm hoping it will get better once they figure it out more*.
- **Machine Learning Street Talk podcast veers into Crankery**: The [Machine Learning Street Talk podcast](https://arxiv.org/abs/1703.10987) is considered by some to be declining in technical depth, with discussions often venturing into *crankery territory*.
   - A member stated, *There's very little machine learning and much more street talk happening* now, compared to its more technical focus *2 years ago*, but pointed to this [technical example](https://youtu.be/vC9nAosXrJw?si=T_S7cCvStvEY-P0X) of them staying focused.
- **AI Aims to Understand Math and the Cosmos**: One member is interested in AI's potential for understanding the **mathematics of intelligence, the universe, creativity, consciousness, and biology**, as well as for generating novel out-of-distribution art and math, and for promoting healthcare.
   - However, another member expressed anger towards individuals who conduct *fuck all* research in AI yet become CEOs.
- **fixupx Pre-Prints Draw Scorn**: The proliferation of pre-prints on platforms like [fixupx.com](https://fixupx.com/jessi_cata/status/1966281024301838846) drew criticism, with one member exclaiming, *This gets to be a pre-print too? Come... onpick your bin*.
   - Further links include [this one](https://fxtwitter.com/_lyraaaa_/status/1925683283263648191).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1415778283571908812)** (9 messages🔥): 

> `HuMo, Disinformation use-cases` 


- **HuMo paper gets attention**: A member shared a link to the [HuMo paper](https://arxiv.org/abs/2509.08519) and [its accompanying demo](https://phantom-video.github.io/HuMo/), suggesting it would be reviewed.
   - Others reacted with laughing emojis.
- **HuMo may be used for disinformation**: A member suggested that **HuMo** could be used for **disinformation**, pointing to its name which means *gaslighting* in Spanish.
   - Another member agreed, noting it makes sense for potential disinformation use-cases.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1415915317472788541)** (4 messages): 

> `Albania AI Minister, Qwen Blog, MobileLLM-R1-950M` 


- **Albania Appoints AI Minister**: Albania is set to appoint an **AI bot minister** to tackle corruption, according to a [Reuters article](https://www.reuters.com/technology/albania-appoints-ai-bot-minister-tackle-corruption-2025-09-11/).
   - The announcement reflects growing interest in **AI solutions for governance**.
- **Qwen Blog**: A member linked to the **Qwen** blogpost.
   - The URL posted was [https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd).
- **Paper Reading Session Suggested**: A member suggested adding the paper on **MobileLLM-R1-950M** to the rotation for a paper reading session.
   - The linked Hugging Face page is [https://huggingface.co/facebook/MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M).


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

ankurgupta_24936: DSPyWeekly Issue No #2 is out https://dspyweekly.com/newsletter/2/
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1415812393346531370)** (26 messages🔥): 

> `DSPy generating sections, Databricks_genai and DSPy, ARC-AGI2 in-context test time training, Modaic declarative AI programming` 


- **DSPy Section Generation Frustrates Users**: A user is struggling to generate an **exact number of sections** in DSPy lesson plans, finding that the LLM generates 13-15 sections when only 12 are requested, even when using **GPT-5** with high reasoning effort.
   - Joel Grus suggested a two-step approach: first generate **12 section titles**, then flesh out each section, to better control the section count.
- **Databricks_genai and DSPy fine-tuning: Anyone Tried?**: A community member inquired about using **databricks_genai** and **DSPy** to fine-tune a model served on Databricks.
   - There were no direct responses in the provided messages, indicating either a lack of experience or ongoing exploration in this area.
- **ARC-AGI2 In-Context Training Interest?**: A member is seeking collaborators interested in **ARC-AGI2** using *in-context test time training*, similar to top systems on **ARC-AGI1**, but using in-context learning rather than finetuning.
   - Their goal is to understand the limits of in-context learning on *out-of-distribution* tasks with very few data points, acknowledging the work won't be valid for the official challenge due to using provider LLMs.
- **Stream Templates in DSPy?**: A user wants to combine multiple DSPy output fields into a single template-based output, but retain the ability to *stream* the output.
   - Ian suggested using a parent module with a `def forward` (or `async aforward` for async) to modify the template and enable streamify; the article [Automatic System Prompt Optimization](https://maximerivest.com/posts/automatic-system-prompt-optimization.html#making-a-simple-custom-adapter) was shared to guide the solution.
- **Modaic Launches as Declarative AI Hub**: A team launched [Modaic](https://www.modaic.dev/), a hub for declarative AI programming inspired by DSPy, featuring primitives like metrics and optimizers.
   - Modaic offers an SDK for building, composing, version controlling, and collaborating on DSPy programs, with its SDK available on [PyPI](https://pypi.org/project/modaic/) and documentation on [docs.modaic.dev](https://docs.modaic.dev/).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1415836697463357480)** (15 messages🔥): 

> `Remove realize from `__setitem__` bounty, Assign operation is deeply broken, GEMM TFLOP measurement on RTX 4090` 


- **Users Tackle `__setitem__` Realize Removal**: A new tinygrad contributor is working on the bounty to remove `realize` calls from `__setitem__`, aiming to consolidate multiple kernel calls into a single kernel for efficiency with code [example](https://cdn.discordapp.com/attachments/1068976834928193609/1415915644792209458/Screenshot_2025-09-12_at_12.22.59_PM.png?ex=68c64334&is=68c4f1b4&hm=d1f1b4a406ca78a3450fd13e06a2b2964a2f7df2fa51a55d5ae2ef74d6912940&).
   - The goal is to transform a sequence of individual `__setitem__` calls into a single kernel execution that accumulates all assignments which is expected to improve performance by reducing kernel launch overhead.
- **`assign` Operation Under Scrutiny**: The `assign` operation in tinygrad is under investigation after failing tests; a user mentioned that *assign on master is actually just broken, failing test here* ([#12131](https://github.com/tinygrad/tinygrad/pull/12131)).
   - The discussion questions whether `assign` should return a value like `store` and suggests potential refactoring in `rangeify` to address the issues, as the return from assign is never used.
- **GEMM TFLOPs Benchmark Target on RTX 4090**: Users discussed the feasibility of achieving the bounty target of *165+ TFLOP GEMM (match torch) with multistage kernel on 4090, FP16 or BF16 with FP32 acc* given the RTX 4090's theoretical peak throughput.
   - The concern raised that unless the actual clock speed exceeds the boost clock, reaching the target TFLOPs might not be realistic.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1416051966043357195)** (4 messages): 

> `tinygrad documentation, company meeting` 


- **tinygrad documentation gets praise**: A member praised the **tinygrad's documentation**, describing it as *useful* and *simple*.
   - The documentation made the member exclaim that things *make sense* over and over.
- **tinygrad next company meeting**: A member asked when the next company meeting is, indicating they would like to listen in if possible.
   - The meeting is scheduled for **Monday 9 am San Diego time**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1416030834921836616)** (4 messages): 

> `RepoMap Benchmarks, Real World Benchmarks, Aider Repomap Use` 


- **RepoMap Benchmarks raise concerns**: A member questioned the use of **RepoMap** in benchmarks, expressing concern that it might be artificially raising the pass rate.
   - Another member suggests *"repo map results are not comparable to non repo map results"* and that **RepoMap** may improve confidence in weaker models when they have the right information in their context window.
- **Request for Real World Benchmark**: A member suggested that a benchmark should reflect real-world experience with models, noting that one automation task proved impossible for all models except **gemini-2.5-pro**.
   - The evaluation approach needs revision to obtain real-world data as **Gemini 2.5 pro** outperformed all others.
- **Aider benefits from RepoMap**: **RepoMap** provides extra context via filenames, class signatures, and function signatures, which helps **LLMs** understand what's available.
   - A member always uses **Aider** with **RepoMap** on, believing that a leaderboard using **RepoMap** would more accurately reflect their real-world use case, though benchmark results may still differ from real code cases.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1415849129145274501)** (5 messages): 

> `C to Rust Migration with Aider, Aider always start in /ask mode` 


- **Aider's C to Rust migration**: A user is using **aider** to perform **C to Rust migration** in a python script, but **aider** is unable to navigate and read the relevant **C** files automatically.
   - The user is seeking guidance on what they might be missing regarding **aider's** functionality.
- **Configure Aider to Always Start in /ask Mode**: A user is looking for a way to configure **aider** to always start in **/ask mode**, potentially via a **YAML config**.
   - They checked the documentation but couldn't find a relevant config key, and another user suggested using `aider --chat-mode ask` or create a `ask.yml` config file with `chat-mode: ask` then run `aider -c ask.yml`.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1415783420302200922)** (9 messages🔥): 

> `WordPress to Next.js conversion, Manus AI Basic Plan, Mount free credits, Manus interlink knowledge, Manus credits rollover` 


- **WordPress Converted to Next.js for Vercel?**: A member inquired about converting a **WordPress website** to **Next.js** for hosting on **Vercel**, noting the shift from **PHP** to **React.js**.
   - Another member suggested cloning the website using **Manus** or other **AI** tools as an alternative.
- **Manus AI Basic Plan Subscribers Frustrated**: A **Basic Plan** subscriber expressed dissatisfaction with the removal of the option to buy extra credits, which forces users to upgrade even for small needs.
   - They requested that **Manus AI** reconsider reopening top-ups for Basic users, emphasizing the importance of flexibility.
- **New User Mount Credit Issues**: A new user reported not receiving the standard **1,000 free credits** upon creating an account on **Mount**, despite the website's claim.
   - No resolution or further information was provided in the discussion.
- **Manus Knowledge Interlinking Inquiry**: A member asked if **Manus** can pull information from all chats to interlink knowledge from each chat/task for universal usage.
   - No response or clarification was provided regarding **Manus**'s knowledge interlinking capabilities.
- **Daily Manus Credits Discontinued?**: A user reported that their **daily 300 credits** had stopped being issued, prompting confusion.
   - No solution or further information was provided in the discussion.


  

---


---


---


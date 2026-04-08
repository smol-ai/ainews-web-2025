---
id: MjAyNS0w
title: "Western Open Models get Funding: Cohere $500m @ 6.8B, AI2 gets $152m NSF+NVIDIA grants"
date: '2025-08-14T05:44:39.731046Z'
description: >-
  **OpenAI's GPT-5** achieved a speedrun of Pokemon Red 3x faster than **o3**.
  **Perplexity** raised **$200M** at a **$20B valuation**. **AI2** secured
  **$75M NSF grants** and **$77M from NVIDIA** for AI infrastructure projects
  like Olmo and Molmo. **Cohere** raised **$500M** and hired **Joelle Pineau**
  from **meta-ai-fair**, boosting models like Command A. **Google** released the
  **Gemma 3 270M** on-device tiny LLM with INT4 QAT checkpoints and large
  embedding tables, and made **Imagen 4** generally available with a fast
  version at $0.02/image. **Meta-ai-fair** introduced **DINOv3**, a family of
  self-supervised vision foundation models with high-resolution dense features
  and strong performance on benchmarks like COCO detection and ADE20K
  segmentation, under a permissive license. A **$150,000 MiniMax AI Agent
  Challenge** is ongoing with 200+ prizes, encouraging AI project builds by
  August 25.
companies:
  - openai
  - perplexity-ai
  - ai2
  - nvidia
  - cohere
  - meta-ai-fair
  - google
  - hugging-face
  - ollama
  - unsloth
models:
  - gpt-5
  - o3
  - command-a
  - gemma-3-270m
  - imagen-4
  - dinov3
topics:
  - model-speed
  - funding
  - ai-infrastructure
  - on-device-ai
  - quantization
  - embedding-models
  - image-generation
  - self-supervised-learning
  - vision
  - dense-prediction
  - benchmarking
  - instruction-following
  - model-optimization
  - model-release
  - challenge
people:
  - joelle_pineau
  - fchollet
  - awnihannun
  - _philschmid
  - osanseviero
---


**Funding for open models are all we need.**

> AI News for 8/13/2025-8/14/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 9744 messages) for you. Estimated reading time saved (at 200wpm): 710 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Congrats to [GPT5 speedruning Pokemon Red](https://www.reddit.com/r/singularity/comments/1mq2irv/gpt5_just_finished_pokemon_red/) 3x faster than o3, and to Perplexity raising [$200m at $20B valuation](https://x.com/arfurrock/status/1955740969116299466?s=46), but the day belongs to the open models crew who announced big injections of cash this week:

- [**Ai2 announced $75m of NSF grants and $77m from NVIDIA**](https://allenai.org/blog/nsf-nvidia) to continue building critical AI infrastructure like Olmo and Molmo (see also Nathan Lambert's [American Truly Open Models Project](https://x.com/natolambert/status/1952370970762871102))
- [**Cohere raised $500m in a new round and hiring of Joelle Pineau**](https://x.com/aidangomez/status/1955993896590152114), former lead of META FAIR, presumably good news for [models like Command A](https://news.smol.ai/issues/25-03-17-ainews-coheres-command-a-claims-3-open-model-spot-after-deepseek-and-gemma).

Good guys and gals won today.

---

[](https://resend-attachments.s3.amazonaws.com/SiP8BRwkgadkEqG)

🚀 **$150,000 MiniMax AI Agent Challenge** — Bring Your A-Game!

- 💡 Build from scratch or remix projects — 200+ **prizes** await.
- 🗓 **Submit by Aug 25** → https://minimax-agent-hackathon.space.minimax.io/
- Don’t just imagine what you can build with AI — **prove it**.
- More details are in the official Luma page https://lu.ma/2u17h1zw

---

# AI Twitter Recap

**Google’s Gemma 3 270M model and Imagen 4 Fast**

- **Gemma 3 270M (on-device tiny LLM)**: Google released a 270M-parameter Gemma 3 model with strong instruction-following and open weights. It’s engineered for “hyper-efficient” local use, featuring INT4 QAT checkpoints and a large embedding table (≈170M embedding params, ≈100M transformer params), yielding surprising capability at its size. It’s already shipping broadly:
    - Runs on-device across stacks: KerasHub presets from [@fchollet](https://twitter.com/fchollet/status/1956059444523286870), MLX day-0 support, ~650 tok/s on M4 Max at <200MB in 4-bit via MLX-LM from [@awnihannun](https://twitter.com/awnihannun/status/1956053493216895406) (with DWQ quality gains at same speed [follow-up](https://twitter.com/awnihannun/status/1956089788240728467)), Ollama one-liner ([ollama run gemma3:270m](https://twitter.com/ollama/status/1956034607373222042)), dynamic GGUF from Unsloth (docs + ~50 tok/s on phones) via [@UnslothAI](https://twitter.com/UnslothAI/status/1956027720288366883), and a Hugging Face collection highlighted by [@ggerganov](https://twitter.com/ggerganov/status/1956026718013014240). It even runs on a Pixel 7a ([demo](https://twitter.com/1littlecoder/status/1956065040563331344)). Official announcement: [@googleaidevs](https://twitter.com/googleaidevs/status/1956023961294131488), details: [@_philschmid](https://twitter.com/_philschmid/status/1956024995701723484), and overview: [@osanseviero](https://twitter.com/osanseviero/status/1956024223773663291).
    - Notable design tradeoffs: more than half the parameters are in embeddings ([observation](https://twitter.com/code_star/status/1956033343465906379)), likely aiding vocabulary/coverage at tiny scales; training tokenization details being actively discussed in threads.
- **Imagen 4 GA + Imagen 4 Fast**: Google made Imagen 4 generally available and introduced “Imagen 4 Fast” at ~$0.02/image—useful for large-scale or interactive workflows ([announcement](https://twitter.com/googleaidevs/status/1956035672197771479)).

**Meta’s DINOv3: high-resolution dense vision features at scale (permissive)**

- **DINOv3 (self-supervised vision foundation models)**: Meta introduced a family of models trained with SSL that produce high-res dense features and outperform specialized systems on long-standing dense prediction tasks—often with a frozen backbone:
    - Reported results include COCO detection 66.1 mAP with a frozen backbone, ADE20K linear 55.9 mIoU (63.0 with a decoder), 3D correspondence 64.4 recall on NAVI, and video tracking 83.3 J&F on DAVIS ([metrics thread](https://twitter.com/BaldassarreFe/status/1956027867860516867) and [follow-up](https://twitter.com/BaldassarreFe/status/1956027888051892594)). Release posts: [@AIatMeta](https://twitter.com/AIatMeta/status/1956027795051831584), day‑0 Transformers support ([HF post](https://twitter.com/AIatMeta/status/1956027800500232525)).
    - The release spans >12 ConvNeXT/ViT models across sizes, trained on varied (incl. satellite) data with a permissive license—quickly adopted as a backbone drop‑in (e.g., plugged into VGGT pipelines for instant SOTA‑like boosts; [@maxseitzer](https://twitter.com/maxseitzer/status/1956029421602623787)). Summaries: [@mervenoyann](https://twitter.com/mervenoyann/status/1956033306580877406).

**Frontier model capability and efficiency: GPT‑5, FormulaOne, DetailBench, GFPO**

- **GPT‑5 behavior in-the-wild**:
    - Pokémon Red: GPT‑5 completed the run in 6,470 steps vs. o3’s 18,184 (≈3× faster), with fewer hallucinations and better spatial planning per [@Clad3815](https://twitter.com/Clad3815/status/1955980772575268897) and [@scaling01](https://twitter.com/scaling01/status/1955813023735828587). Caveat: runs may differ in tool access (e.g., Google Search), affecting comparability ([@kiranvodrahalli](https://twitter.com/kiranvodrahalli/status/1956044490885751273)).
    - Medical QA: with high reasoning effort, GPT‑5 nears perfect accuracy on a high-quality ophthalmology QA dataset ([paper link](https://twitter.com/omarsar0/status/1956003145349521780)).
- **New reasoning benchmarks**:
    - FormulaOne (expert‑level Dynamic Programming): “shallow” tier: top models 50–70%; “deeper”: Grok/Gemini/o3/Opus-4 solve ≤1/100, GPT‑5 Pro solves 4/100; “deepest”: all 0% ([@shai_s_shwartz](https://twitter.com/shai_s_shwartz/status/1955968602978320727)).
    - DetailBench (spotting small errors without being asked): surfaces an ability orthogonal to instruction-following—some models do better than GPT‑5, which can over‑comply with the stated task ([@xeophon_](https://twitter.com/xeophon_/status/1956025495515979984)).
- **Reasoning token efficiency and training techniques**:
    - Token efficiency gap: open models often emit 1.5–4× more tokens than closed models on the same tasks (up to 10× on simple queries), eroding per‑token price advantages; detailed landscape review from [@NousResearch](https://twitter.com/NousResearch/status/1956090990005248341) with commentary by [@scaling01](https://twitter.com/scaling01/status/1956098555090714668).
    - GFPO (Group Filtered Policy Optimization): reduces “length inflation” by sampling larger groups during training and filtering by length and reward-per-token. On Phi‑4‑reasoning 14B, GFPO cuts lengths by 46–71% vs. GRPO; optimizing reward/tok boosts cuts to 71–85% while maintaining accuracy on AIME24/25, GPQA, Omni‑MATH, LiveCodeBench ([abs + summary](https://twitter.com/iScienceLuvr/status/1955955524790575212)).
    - Efficient decoding and amortized test‑time compute: OverFill (full model for prefill → pruned dense model for decode) to improve quality at minimal latency ([code+abs](https://twitter.com/iScienceLuvr/status/1955965909409120476)); Noise Hypernetworks to replace reward‑guided test‑time noise optimization in diffusion, recovering much of the quality at a fraction of cost ([abs](https://twitter.com/iScienceLuvr/status/1955958029993828724)); and diffusion LMs with temporal voting/reinforcement to exploit mid‑trajectory consistency ([thread](https://twitter.com/iScienceLuvr/status/1955964748341919862)).

**Open ecosystem, scale, and infra**

- **AI2 receives $152M for open models**: $75M from NSF + $77M from NVIDIA to scale open model ecosystems (OLMos, Molmos, etc.) and reproducible AI for science ([@allen_ai](https://twitter.com/allen_ai/status/1955966785175388288)). Context from [@natolambert](https://twitter.com/natolambert/status/1955986546626322479): this single line item is ~20% of NSF’s 2026 AI budget; NVIDIA is contributing leading hardware. Reflections from [@HannaHajishirzi](https://twitter.com/HannaHajishirzi/status/1955984650599325808).
- **Cohere raises $500M; Joelle Pineau joins as Chief AI Officer**: Enterprise- and government-focused, secure/sovereign AI; also welcoming a new CFO ([@cohere](https://twitter.com/cohere/status/1955993354745082336), [@aidangomez](https://twitter.com/aidangomez/status/1955993896590152114), [@jpineau1](https://twitter.com/jpineau1/status/1955995736895594838), [@nickfrosst](https://twitter.com/nickfrosst/status/1956005330069983332)).
- **Throughput and compiler updates**: Modal demonstrates rapid GPU scaling: 100× H100s in ~12s, 300× in ~4 minutes, scaling to 1k+ ([@bernhardsson](https://twitter.com/bernhardsson/status/1956073789550420330)). The “State of torch.compile (Aug 2025)” post by [@ezyang](https://twitter.com/ezyang/status/1955820298907082876) is making the rounds (notable for model launch latency/cold-start improvements paired with tips like regional compilation on Qwen-Image).
- **Tooling**: TRL adds native VLM post‑training support ([@QGallouedec](https://twitter.com/QGallouedec/status/1956066332488950020)); vLLM powers Amazon’s Rufus assistant ([@vllm_project](https://twitter.com/vllm_project/status/1956116150259212619)).

**Agents: simulation, deep research, and browser-native assistants**

- **Simulation-first eval for chatbots**: Guardrails launched Snowglobe, a user simulation engine to test and improve bots pre‑production ([launch](https://twitter.com/ShreyaR/status/1956023326721368337)), drawing praise from [@goodfellow_ian](https://twitter.com/goodfellow_ian/status/1956040393361121540) and agent reliability framing via self‑driving analogies ([@apoorvapandhi](https://twitter.com/apoorvapandhi/status/1956033885126468050)).
- **Deep research agents**: LangChain published a free LangGraph course on building long-running, multi‑agent research systems with persistence/observability ([@LangChainAI](https://twitter.com/LangChainAI/status/1956027411302375631), [@hwchase17](https://twitter.com/hwchase17/status/1956036358709108979)). Separately, “Elysia” showcases agentic RAG with decision-tree transparency, personalized feedback, and chunk‑on‑demand pipelines ([@philipvollet](https://twitter.com/philipvollet/status/1955945448860008655)).
- **Browser agents and web research**: Perplexity announced Comet for Enterprise—an “AI-powered browser agent” for secure, linked-tool workflows ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1956046685509210183)). Meanwhile, [@paraga](https://twitter.com/paraga/status/1956008857555099928) unveiled Parallel’s “second user of the web” vision via [@p0](https://twitter.com/p0/status/1956007609250492924), claiming human- and frontier-model–beating deep web research benchmarks.

**Interactive video, robotics, and multimodality**

- **Tencent’s Hunyuan‑GameCraft (open-source)**: A HunyuanVideo‑based framework for high‑dynamic, playable, physically realistic game video generation. Unifies keyboard inputs into a continuous action space for precise control; hybrid history conditioning for long-term consistency; and PCM distillation to compress inference (quantized 13B runs on RTX 4090). Project page + code + report linked in the announcement ([@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1955839140173631656)).
- **Robotics**: A satisfying demo of robot hands folding laundry ([video](https://twitter.com/adcock_brett/status/1956021725491290154)), followed by a stark reminder that human‑like hands are essential for general purpose robots—“as hard as building the whole robot” ([commentary](https://twitter.com/adcock_brett/status/1956083802440450551)).

**Top tweets (by engagement)**

- Meta’s DINOv3: SSL vision backbones with SOTA dense features and day‑0 HF support ([@AIatMeta](https://twitter.com/AIatMeta/status/1956027795051831584))
- Google Imagen 4 GA + “Fast” at $0.02/image ([@googleaidevs](https://twitter.com/googleaidevs/status/1956035672197771479))
- Google’s Gemma 3 270M tiny LLM release and ecosystem support ([@osanseviero](https://twitter.com/osanseviero/status/1956024223773663291), [@googleaidevs](https://twitter.com/googleaidevs/status/1956023961294131488))
- Gemini App doubled rate limits for “2.5 Deep Think” ([@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1955821580237594847), [@joshwoodward](https://twitter.com/joshwoodward/status/1955804081437696046))
- GPT‑5 Pokémon performance (~3× fewer steps vs o3) ([@Clad3815](https://twitter.com/Clad3815/status/1955980772575268897), [@scaling01](https://twitter.com/scaling01/status/1955813023735828587))
- FormulaOne benchmark: expert‑level dynamic programming remains hard (even for GPT‑5 Pro) ([@shai_s_shwartz](https://twitter.com/shai_s_shwartz/status/1955968602978320727))

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Benchmarking and Popularity of Small Language Models

- [**google/gemma-3-270m · Hugging Face**](https://huggingface.co/google/gemma-3-270m) ([Score: 533, Comments: 204](https://www.reddit.com/r/LocalLLaMA/comments/1mq3v93/googlegemma3270m_hugging_face/)): **Google has released the Gemma-3-270M model (270 million parameters) on Hugging Face, positioning it as a small-scale open weights LLM alternative. The model is available with bfloat16 (BF16) precision, targeting efficient inference, potential edge hardware compatibility, and experimentation in resource-constrained environments. Details on architecture, training data, and benchmarks are not highlighted in this post, but its small size suggests adaptability for lightweight deployments.** Commenters note initial confusion due to the naming ('270M' vs. '270B'), and discuss the utility of BF16 precision weights for practical experimentation. There is interest in the model's deployment potential, though skepticism remains regarding its capacity given the small parameter count.
    - A user points out that the smallest model in the Gemma family, the **270M parameter version**, was trained on an unusually large dataset of **6 trillion tokens**, which is notable since even the much larger models (e.g., 1B, 4B, 12B, 27B) were trained on proportionally fewer tokens. This could have significant implications for the efficiency and quality of small model scaling and generalization.
    - There is mention of interest in using **BF16 weights** with the Gemma 270M model, indicating attention to efficient inference and training via reduced precision, which is increasingly common in deployment scenarios.
- [**Who are the 57 million people who downloaded bert last month?**](https://i.redd.it/vk2njmk01xif1.png) ([Score: 350, Comments: 109](https://www.reddit.com/r/LocalLLaMA/comments/1mpr0nc/who_are_the_57_million_people_who_downloaded_bert/)): **The image shows the Hugging Face model page for Google's 'bert-base-uncased', which reports over 57 million downloads in the past month. Key technical context: the download count reflects *model pulls* (likely by automated CI/CD pipelines, researchers, and educational users) rather than unique users, highlighting BERT's continued dominance and integration in various ML setups. The page also displays the wide support for different ML frameworks (PyTorch, TensorFlow, JAX) and active community engagement (likes and follows).** Commenters clarify that download numbers do not equate to unique users due to frequent re-downloads by automated systems, students, and researchers—underscoring BERT's entrenched status in NLP workflows.
    - Several commenters emphasized that the 57 million download figure for BERT likely reflects repeated downloads by automated systems, such as CI/CD pipelines, and not unique users. This is a common phenomenon in popular open-source models because many workflows and organizations frequently re-download models for automated evaluation, retraining, or deployment.
    - BERT persists as a foundational tool in NLP pipelines, used for a variety of tasks including classification, prediction, and generating embeddings; its inclusion in widely used training courses (such as those by Hugging Face) continually drives high download numbers as new learners and organizations use it for educational and experimental purposes.
    - Researchers and students frequently use BERT for research tasks, reranking, and as a baseline model, further amplifying its ongoing relevance and high usage statistics despite the emergence of more modern NLP architectures.
- [**the "missing latest Qwen syndrome"**](https://i.redd.it/z096hdwp01jf1.jpeg) ([Score: 251, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1mq8oyk/the_missing_latest_qwen_syndrome/)): **The image is a scatter plot comparing model sizes (in parameters) to IFEval scores for various LLMs, including Gemma 3 (270M, 1B), SmolLM2 (135M, 360M), Qwen 2.5 (0.5B), and Llama 3.2 (1B). The post and comments highlight a recurring omission of the latest Qwen models in benchmark comparisons, with users noting that Qwen3-0.6B achieves a solid 59.2 score on IFEval—competitive for its parameter size. This underscores the importance of including all notable models to reflect current state-of-the-art performance in instruction following evaluations.** Commenters debate the reliability of IFEval scores at small model sizes, with some likening a 50% performance level to chance. There is also explicit frustration with frequent omission of Qwen models in comparative charts, implying a skewed or incomplete view of the small-scale LLM landscape.
    - Multiple users highlight that Qwen3-0.6B, despite its relatively small size (270 million parameters), achieves an IFEval score between 50% and 59.2%, which is considered strong for its parameter count. Such performance metrics are seen as very competitive for models in this size range.
    - Discussion emphasizes that these smaller models, such as Qwen3-0.6B, generally aren't designed for broad, out-of-the-box usage but instead are intended to be fine-tuned for specific tasks. Customization and further training are recommended to realize their full potential in targeted applications.

### 2. Upcoming and Open Source AI Model Releases (Grok 2, DeepSeek)

- [**Just a reminder that Grok 2 should be released open source by like tomorrow (based on Mr. Musk’s tweet from last week).**](https://i.redd.it/hsaoxskfs1jf1.jpeg) ([Score: 282, Comments: 82](https://www.reddit.com/r/LocalLLaMA/comments/1mqctep/just_a_reminder_that_grok_2_should_be_released/)): **The image captures a Twitter exchange where a user asks about the timeline for open-sourcing Grok 2 and Grok 3, referencing a previous statement from Elon Musk that xAI intends to open-source their models. In reply, Elon Musk states that Grok 2 will be open-sourced the following week, citing ongoing work by the team. This screenshot highlights the anticipated release schedule for the open-source version of Grok 2, potentially making it available to the public imminently.** Commenters are skeptical about the promised release date, referencing Musk's history of optimistic timelines and expressing cynicism about actual model release. Others compare xAI's open-source pledge to OpenAI's current closed approach, albeit with irony given shifting industry trends.
    - One user questions the technical relevance of open-sourcing Grok 2 (and even Grok 3), characterizing it as a "giant piece of old crud" and suggesting that even a potential Grok 3 release would have limited utility for the community. This reflects broader skepticism about the architectural and performance competitiveness of these models compared to current state-of-the-art offerings, implying that unless open source releases match or exceed modern benchmarks, their practical value may be marginal.
- [**DeepSeek’s next AI model delayed by attempt to use Chinese chips**](https://www.ft.com/content/eb984646-6320-4bfe-a78d-a1da2274b092) ([Score: 513, Comments: 111](https://www.reddit.com/r/LocalLLaMA/comments/1mpu8ot/deepseeks_next_ai_model_delayed_by_attempt_to_use/)): **DeepSeek delayed the release of its R2 AI model after failing to complete training on Huawei's Ascend processors, reportedly due to technical limitations such as instability, slow inter-chip connectivity, and inferior software stack compared to Nvidia's offerings. The company reverted to using Nvidia GPUs for training while attempting to maintain Ascend chips for inference, illustrating ongoing challenges in China's effort to reach hardware self-sufficiency in AI. Additional delays were attributed to extended data labeling cycles; meanwhile, competitors like Alibaba's Qwen3 are reportedly incorporating DeepSeek's reasoning-capable training algorithms but improving on their efficiency.** Commenters note the strategic importance for China of succeeding with domestic chips despite delays, and some skepticism on the reliability or sourcing of information (e.g., reliance on anonymous sources in the Financial Times). There is also debate over whether delays stemming from domestic hardware adoption might nevertheless offer long-term advantages to China's AI independence and industry growth.
    - DeepSeek's attempt to train its upcoming R2 model on Huawei's Ascend chips encountered significant technical hurdles, including persistent training failures, stability issues, slower inter-chip connectivity, and inferior software compared to Nvidia's hardware. This technical gap led them to use Nvidia chips for training while continuing attempts to utilize Ascend for inference, illustrating ongoing Chinese challenges with achieving hardware self-sufficiency for advanced AI workloads.
    - The delay in releasing R2 is partly attributed to longer-than-expected data labeling, as well as unresolved technical integration with Chinese chips. Despite onsite support from Huawei engineers, DeepSeek was unable to complete a successful training run on Ascend hardware for R2, highlighting how Chinese alternatives currently lag behind US solutions such as Nvidia in critical AI training tasks.
    - Industry analysts point out that Alibaba's Qwen3 model adopted DeepSeek's core training algorithms but implemented them more efficiently, suggesting rapid technical knowledge transfer among Chinese AI labs. There is recognition that while Huawei's AI ecosystem is facing 'growing pains,' leading models trained on Chinese chips may become feasible over time as the ecosystem matures.

### 3. Hardware and Practical Challenges in AI Model Deployment (Qwen Context & GPUs)

- [**MaxSun's Intel Arc Pro B60 Dual GPU with 48GB memory reportedly starts shipping next week, priced at $1,200**](https://videocardz.com/newz/maxsun-arc-pro-b60-dual-with-48gb-memory-reportedly-starts-shipping-next-week-priced-at-1200) ([Score: 353, Comments: 138](https://www.reddit.com/r/LocalLLaMA/comments/1mpxumt/maxsuns_intel_arc_pro_b60_dual_gpu_with_48gb/)): **MaxSun is launching the Intel Arc Pro B60 Dual GPU card with 48GB of memory, priced at $1,200 and reportedly shipping next week. This board requires motherboard PCIe slot bifurcation support because it lacks an onboard PCIe hub chip, meaning both GPUs cannot be accessed on systems without native bifurcation capabilities (typically only on high-end workstation/server boards or platforms like Xeon/Threadripper with full x16 signaling). Full specs are available on the [official product page](https://www.maxsun.com/products/intel-arc-pro-b60-dual-48g-turbo).** Notable technical debate centers on potential limited board availability and high street price as well as the critical requirement for PCIe bifurcation, making the card impractical for mainstream desktops. Users recommend waiting for independent reviews to assess real-world performance and compatibility.
    - A key technical caveat is that the MaxSun Intel Arc Pro B60 Dual GPU requires *motherboard PCIe bifurcation support* because it does not include an onboard PCIe hub chip, unlike many prior dual-die GPUs. This means the card will only function properly in top PCIe x16 slots delivering full x16 lanes, and won't work in secondary slots unless using high-end platforms (e.g., Xeon or Threadripper) that provide full x16 lanes across multiple slots.
    - There is anticipation for benchmarking and compatibility reviews, especially to determine if frameworks like ollama and llama.cpp (popular for maximizing ML performance on consumer GPUs) fully support the dual-Arc Pro B60, given its unusual configuration and memory setup (48GB across two GPUs).
- [**1 million context is the scam , the ai start hallucinating after the 90k . im using the qwen cli and its become trash after 10 percent context window used**](https://www.reddit.com/r/LocalLLaMA/comments/1mq19x6/1_million_context_is_the_scam_the_ai_start/) ([Score: 232, Comments: 84](https://www.reddit.com/r/LocalLLaMA/comments/1mq19x6/1_million_context_is_the_scam_the_ai_start/)): **The post highlights practical limits in the effective context window of large language models, specifically citing Qwen's CLI: the model's outputs degrade and 'hallucinate' after processing approximately 90-100k tokens—far below the claimed '1 million' context. While benchmarks rarely capture this, user experience suggests performance drops substantially once ~10% of window is used. Linked benchmarks and screenshots further corroborate quality degradation beyond 200k tokens for some models.** Commenters note context window limitations vary by model and hardware: Gemini Pro is cited as capable of handling `200-300k` context tokens with stable, high-quality performance, especially when using Google's infrastructure. There's consensus that local models struggle more significantly at large contexts compared to cloud-based solutions like Gemini.
    - Multiple users report that Qwen's performance significantly declines as its context window approaches 90–200k tokens, exhibiting increased hallucination and degraded recall. Linked visual benchmarks illustrate this trend and highlight the context-length sensitivity issue (see: https://preview.redd.it/cpeii3wpqzif1.png).
    - Conversely, some users note that Google's Gemini Pro model maintains strong performance with exceptionally large contexts (up to 1.5M–2M tokens). Claims are made that Gemini Pro can process and recall information from documents in the 200–300k token range with high accuracy, suggesting superior hardware or context management strategies.
    - A common limitation discussed is that, as the context window grows in most LLMs, the models become prone to forgetting earlier content or providing less comprehensive responses unless explicitly instructed to retain details. This points to a broader issue with long-context retention across models, not just Qwen.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. GPT-5 Beats Previous Models at Pokémon Red Speedruns

- [**GPT-5 Just Finished Pokemon Red!**](https://i.redd.it/u0us03n8yzif1.png) ([Score: 1764, Comments: 172](https://www.reddit.com/r/singularity/comments/1mq2irv/gpt5_just_finished_pokemon_red/)): **GPT-5 completed Pokémon Red in 6,470 steps and about 7 days, significantly outperforming o3 (the previous version, which took 18,184 steps and 15 days), as well as beating competing models like Claude and Gemini. The shared image shows GPT-5's in-game completion screen, a breakdown of its final strategy and actions, and live reasoning panels that illustrate its autonomous gameplay decisions. This run demonstrates substantial improvements in autonomous gameplay efficiency, planning, and reasoning visibility for large language model agents.** Commentary notes that GPT-5 relied on a common but suboptimal one-Pokémon 'hard carry' strategy, echoing childlike play patterns, but achieved the goal regardless. There's also a suggestion to benchmark on more complex, less linear games like Zelda: Minish Cap for deeper evaluation of reasoning capabilities.
    - Commenters discuss the gameplay strategy the model used, specifically noting that GPT-5 focused on leveling and using a single Pokémon for most battles, paralleling a common, less complex approach taken by younger or novice players. This suggests the model is capable of efficient optimization in-game but might not yet demonstrate creative or advanced strategic depth.
    - The successful completion of Pokémon Red raises questions about the novelty and challenge posed to GPT-5 given its architecture and training. One user speculates whether the game is in the model's training data or if the model is leveraging prior playthroughs or documented strategies, hinting at ongoing concerns around genuine generalization vs. memorization.
    - A technical benchmark is informally proposed: comparisons to more complex or less-scripted games like Factorio Space Age are suggested as a more demanding test of AGI-like reasoning, emphasizing the need for broader, dynamic problem-solving beyond games with well-known solutions or walkthroughs.
- [**GPT-5 is nearly 3x faster than o3 at earning badges in Pokémon Red**](https://i.redd.it/dudpeygmjwif1.png) ([Score: 1477, Comments: 193](https://www.reddit.com/r/singularity/comments/1mpp7hy/gpt5_is_nearly_3x_faster_than_o3_at_earning/)): **The image presents a side-by-side benchmark of GPT-5 versus o3 in the context of earning badges in Pokémon Red, visualized via a line graph. GPT-5 achieves all 8 badges in just 6,018 steps, nearly 3 times faster than o3's 16,743 steps—demonstrating substantial improvement in long-horizon, agentic task performance. Technical commentary in the post and comments emphasizes GPT-5's notable advantages: robust execution of long, complex button sequences, resilience to hallucinations (recovering rapidly from errors), and effective in-game strategy.** A key technical discussion underscores the shift in model evaluation methodology, advocating for focus on long-horizon and complex task benchmarks over traditional, narrow metrics. Commenters highlight GPT-5's improved strategic reasoning and operational robustness, signaling paradigm changes in AI agent assessment.
    - Commenters note that model evaluation should shift to include performance in long-horizon, agentic tasks rather than only traditional static benchmarks, as new models like "GPT-5" show significant progress there (e.g., 3x badge-earning speed in Pokémon Red).
    - Technical observations highlight GPT-5's ability to execute complex, multi-step sequences through game menus, navigation, and battles with considerably fewer prolonged hallucinations; the model "snaps out of" errors quickly after few failures and develops surprisingly effective strategies, outperforming o3 in practical, real-time play.
    - There's a technical need for specificity in discussing "GPT-5" because the GPT-5 System Card references at least six distinct models under that label, which complicates benchmarking and capability comparisons.
- [**GPT 5 completed pokémon Red in just 6,470 steps !!**](https://i.redd.it/0yk4psfqh0jf1.png) ([Score: 610, Comments: 112](https://www.reddit.com/r/OpenAI/comments/1mq5hyy/gpt_5_completed_pok%C3%A9mon_red_in_just_6470_steps/)): **The image documents a tweet celebrating GPT-5 (by OpenAI) completing Pokémon Red in 6,470 steps—a significant improvement over the prior 18,184-step record. This feat demonstrates notable advancements in model efficiency for game-playing tasks, implying better optimization techniques or enhanced context understanding within GPT-5. The screenshot also shows the endgame dialogue and associated stats, which authenticate the accomplishment and facilitate technical analysis of the model's gameplay efficiency.** Commenters discuss possible improvements in model training data preprocessing (e.g., removing irrelevant data before runs) and suggest more complex challenges like Nuzlocke runs to robustly test generalization and strategy.
    - A user mentions seeing information that all 'irrelevant training data' was removed before GPT-5's attempt to complete Pokémon Red, implying possible dataset curation to prevent prior memorization or unfair advantage in gameplay automation. This raises questions about reproducibility and the role of prior exposure to similar content in model-based game playing benchmarks.

### 2. Notable New Benchmarks: GPT-5, Google Image Model, SWE-bench

- [**We ran GPT-5, Claude Sonnet 4, Qwen3-coder and more on fresh July SWE-bench like tasks – and GPT-5 clearly crushed!**](https://i.redd.it/99admdkaszif1.png) ([Score: 218, Comments: 65](https://www.reddit.com/r/OpenAI/comments/1mq1oyf/we_ran_gpt5_claude_sonnet_4_qwen3coder_and_more/)): **The image is a bar chart summarizing results from the July 2025 SWE-rebench benchmark, assessing multiple LLMs—including GPT-5 (Medium and High), Qwen3-Coder, and Claude Sonnet 4—on 34 decontaminated, real-world GitHub PR tasks. GPT-5-Medium led with a 29.4% resolution rate, outpacing both Claude Sonnet 4 (20.6%) and Qwen3-Coder (22.4% resolved, but 32.4% pass@5 matching GPT-5-High). This benchmark is notable for its use of fresh, post-training data and is described in detail on the [SWE-rebench leaderboard](https://swe-rebench.com/leaderboard) and its [HuggingFace dataset](https://huggingface.co/datasets/nebius/SWE-rebench-leaderboard).** Several commenters question the absence of strong commercial models like Claude Opus, the generalizability and fairness of these results given differences in model/configuration access, and the statistical significance of results based on only 34 tasks (anticipating potential large error bars).
    - Multiple commenters express concern about benchmark validity, highlighting that the versions of GPT-5 or other models used in public benchmarks may differ from those actually accessible to users—posing a reproducibility and transparency problem for model performance claims.
    - One technical reader notes unexpected results in the benchmark: specifically, that "GPT-5 Medium" outperforms "GPT-5 High" according to the shared data. There is also discussion about the statistically significant error bars inherent in evaluating only 34 tasks, which can greatly affect confidence in the reported rankings.
    - A broader sentiment in the thread questions the reliability of AI model announcements and competitive benchmarking in general, emphasizing increasing skepticism around claims that a model has definitively outperformed others without open access to weights or reliable, universal test conditions.
- [**Google's new image model beats OpenAI's image model at creating accurate charts**](https://i.redd.it/xukz6lphhxif1.png) ([Score: 263, Comments: 39](https://www.reddit.com/r/Bard/comments/1mpspan/googles_new_image_model_beats_openais_image_model/)): **The image showcases a comparison of bar charts supposedly generated by different AI models. The left chart presents 'SWE-bench Verified' benchmarks, contrasting 'Without thinking' and 'With thinking' for each model. On the right, Google’s 'GPT-5' model achieves higher accuracy rates (74.9% and 52.8%) versus 'OpenAI Q3' and 'GPT-4.0', visually demonstrating Google's improved chart-rendering accuracy—aligning with the post title that Google's model outperforms OpenAI's in generating accurate charts. The visual proportions of the bars are criticized for not matching the numerical values.** Top comments raise skepticism about the proportionality of the chart bars, questioning the chart's quality control and potential misrepresentation. There's a call for clarity on benchmarking methodology, suggesting ongoing technical debate about evaluation fairness and QA processes.
    - Several commenters highlight perceived failures in OpenAI's image model QA process, specifically referencing a demo chart that was disproportional, with concerns that obvious flaws were not caught before release. This raises questions about the reliability of model outputs in precision-dependent tasks like chart creation.
    - There is skepticism regarding the use of image generation models for chart creation, with technical arguments favoring the use of traditional LLMs paired with charting tools or libraries. The implication is that direct use of generative image models is unnecessary and possibly less accurate for generating structured data visualizations.

### 3. AI Model and Platform Feature Launches: Claude Code & Gemma 3 270M

- [**Introducing Gemma 3 270M: The compact model for hyper-efficient AI**](https://developers.googleblog.com/en/introducing-gemma-3-270m/) ([Score: 150, Comments: 20](https://www.reddit.com/r/singularity/comments/1mqan83/introducing_gemma_3_270m_the_compact_model_for/)): **Google has released the Gemma 3 270M model, a compact LLM designed for hyper-efficient, on-device inference, targeting use cases with strict compute/power limitations such as smartphones. The model is notable for its extremely small parameter count (**`270 million`**) compared to most SOTA LLMs, positioning it for offline, edge deployment where resource usage is paramount. See the [announcement](https://www.reddit.com/r/LocalLLaMA/comments/1dtvjhy/introducing_gemma_3_270m_the_compact_model_for/) and example benchmarks for more.** Technical discussion in the comments is mixed: while some praise the utility of tiny models for offline/app-specific tasks, several experts argue that the model's tiny size renders its reasoning and language capabilities extremely limited—suitable only for trivial or toy tasks rather than serious dialogue or reasoning workloads (see [critical example output](https://i.imgur.com/YXN4MOr.png)).
    - One commenter notes that the Gemma 3 270M model is suitable for highly resource-constrained devices (e.g., smartphones), highlighting its offline execution potential and value in privacy-focused edge AI applications, despite limitations in general reasoning and complexity.
    - A critical assessment suggests that the model's performance is currently limited to very simple tasks, evidenced by examples of elementary mistakes (such as in the provided screenshot), indicating that state-of-the-art reasoning and knowledge tasks remain out of reach for such compact architectures at present.
    - There is expressed interest in scaled variants, specifically a version (referenced as a 'Gemma 3N') sized to utilize the full capacity of a 16GB GPU, suggesting user demand for larger, yet still efficient, models that maximize consumer hardware capabilities for more sophisticated local AI tasks.
- [**Introducing Gemma 3 270M: The compact model for hyper-efficient AI**](https://developers.googleblog.com/en/introducing-gemma-3-270m/) ([Score: 237, Comments: 59](https://www.reddit.com/r/Bard/comments/1mq59p9/introducing_gemma_3_270m_the_compact_model_for/)): **Google has announced Gemma 3 270M, a compact large language model (LLM) at 270 million parameters—significantly smaller than typical billion-parameter LLMs—optimized for 'hyper-efficient' local deployment. The model is designed for extremely fast inference and lightweight workloads, targeting use cases such as locally-run assistants and on-device app invocation where full-scale models are not required.** Commenters highlight the technical significance of achieving usable performance at such a small scale, discussing its potential for local task execution, rapid inference, and as a cost-saving routing intermediary to larger, API-based LLMs.
    - Several commenters highlight the technical significance of a functioning LLM at the 270M parameter scale, emphasizing its suitability for on-device, local execution due to reduced resource usage and increased speed compared to larger models. This smaller footprint means efficient task automation, such as invoking applications and handling frequent but simple tasks without offloading computation to cloud-based services.
    - There is specific interest in using Gemma 3 270M as a routing layer to larger, API-driven LLMs, optimizing both cost and performance. By selectively offloading only complex queries to expensive cloud models while handling basic tasks locally, users anticipate saving significant operational costs and achieving faster response times for common tasks.
    - A linked performance comparison image is discussed, with users noting the model's strong results and expressing surprise at its capabilities, suggesting that Gemma 3 270M achieves high quality despite its compact size. This could mark a new milestone in the practical deployment of sub-1B parameter models for real-world applications.
- [**Introducing two new ways to learn in Claude Code and the Claude app**](https://www.reddit.com/r/ClaudeAI/comments/1mq6h47/introducing_two_new_ways_to_learn_in_claude_code/) ([Score: 150, Comments: 20](https://www.reddit.com/r/ClaudeAI/comments/1mq6h47/introducing_two_new_ways_to_learn_in_claude_code/)): **Anthropic has introduced two interactively configurable output styles for Claude Code: '/output-style explanatory' (in which Claude justifies architectural choices and outlines best practices) and '/output-style learning' (an interactive, turn-based pair programming mode intended as a pedagogical tool). The "Learning" style, previously limited to Claude for Education, is now available app-wide, enabling stepwise guidance across all chat sessions. Documentation for these features is provided at https://docs.anthropic.com/en/docs/claude-code/output-styles.** One commenter, an educator, notes that fine-grained control over Claude's teaching style could facilitate more effective classroom AI integration, reducing the need for manual prompt engineering. Another user inquires whether the two modes can be combined to create a hybrid style blending explanation with interactive guidance, highlighting a potential area for further feature enhancement.
    - A commenter notes the increasing complexity of the Claude interface, highlighting that users must now manage multiple concepts such as agents, commands, prompts, hooks, and output styles. They propose consolidating these elements to make the experience less fragmented and easier to use, which would benefit both technical and educational workflows.
    - One user asks about the feasibility of activating both explanation and collaborative coding modes simultaneously. Their goal is to have Claude explain its reasoning while also engaging interactively as a coding partner, likening the experience to learning from an especially verbose but instructive teacher, which suggests a demand for more flexible, multimodal teaching workflows in AI tools.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Tiny Titans & Trusty Benchmarks**

- **Gemma-3-270M Goes Micro, Aims for Macro Impact**: **Google** launched **Gemma 3-270M** with ~**300MB** weights and training on ~**6T tokens**, pitched for on-device and constrained deployments, outlined in the official [Introducing Gemma 3-270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/). The community flagged mixed early behavior for niche tasks (RAG, instruction following), with some tests noting hallucinations like *"A dog ... six legs"* in edge cases.
    - Engineers debated target hardware (wearables, in-game agents, tool-calling stubs) and practical niches where a tiny model’s latency trumps raw IQ, referencing the [Gemma 3-270M blogpost](https://developers.googleblog.com/en/introducing-gemma-3-270m/). Discussions emphasized shipping specialized fine-tunes (e.g., chess, retrieval enrichers) over general instruction-following for this size class.
- **Thinking Tax Tally: Nous Measures Token Efficiency**: **Nous Research** introduced a benchmark on reasoning-token efficiency showing open models consume **1.5–4x** more tokens than closed models, with spikes up to **10x** on simple queries; see [Measuring Thinking Efficiency in Reasoning Models](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/). The study argues that token efficiency should be a first-class metric alongside accuracy when evaluating **reasoning** models.
    - Practitioners noted that higher token usage can erase per-token pricing advantages for open models in real deployments, reinforcing budgets that track both accuracy and sequence length. Teams discussed adding a “thinking budget” to eval suites before greenlighting model swaps, citing the **Nous** write-up: [benchmark details](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/).
- **Tool Time Triumph: GPT-5 Tops Router Evals**: **OpenRouter** reported proprietary **tool-calling accuracy** >**99.5%** for **GPT-5**, surpassing **Claude 4.1 Opus**, while **Gemini 2.5 Flash** led volume at ~**5M requests/week**; see [OpenRouter: Tool call evals](https://x.com/OpenRouterAI/status/1956030489900560769). These internal evals highlighted stable tool args, low hallucination rates, and consistent structured outputs for top proprietary models.
    - Builders weighed reliability vs. cost by cross-checking tool-call success against usage telemetry to pick defaults for production routers. Teams flagged the importance of per-endpoint stability notes and regression tracking, pointing to **OpenRouter’s** eval disclosure: [tweet](https://x.com/OpenRouterAI/status/1956030489900560769).

**2. Agent Tooling & Protocols Heat Up**

- **LlamaIndex Spins Up Scrapers & Stocks**: **LlamaIndex** published agent tutorials, including web-scraping agents with **Bright Data** ([Build web‑scraping AI agents](https://twitter.com/llama_index/status/1956129968813171061)) and an **AI stock portfolio agent** with **CopilotKit’s AG‑UI** ([AI stock portfolio agent tutorial](https://twitter.com/llama_index/status/1956089453606527349)). The examples showcase agentic orchestration, robust retrieval pipelines, and frontend–backend handshakes for production UIs.
    - Developers highlighted faster prototyping loops and clearer agent tool interfaces, noting smoother hand-offs between parsing, planning, and execution. Community sentiment favored concrete, end-to-end demos over abstract agent frameworks to accelerate shipping: [web‑scraping guide](https://twitter.com/llama_index/status/1956129968813171061) and [portfolio agent walkthrough](https://twitter.com/llama_index/status/1956089453606527349).
- **MCP Metatools: Harness Hypertools Handle Heap of Helpers**: Two OSS projects emerged to tame tool sprawl: **mcp_harness** for stress-testing tool-binding at scale ([mcp_harness on GitHub](https://github.com/kindgracekind/mcp_harness)) and **hypertool‑mcp** for persona-specific, fully local MCP aggregation ([hypertool‑mcp on GitHub](https://github.com/toolprint/hypertool-mcp)). Both aim to stabilize tool-use beyond **10–15** tools by curating skill sets and routing calls reliably.
    - Practitioners reported improved success rates after segmenting toolsets per persona and enforcing deterministic tool-choice schemas. Local-first setups with zero egress reassured privacy-focused teams while enabling rapid experimentation: [mcp_harness](https://github.com/kindgracekind/mcp_harness) and [hypertool‑mcp](https://github.com/toolprint/hypertool-mcp).
- **LM Studio Learns to Tool**: **LM Studio** demonstrated out-of-the-box tool-calling via **Llama Function Model (lfm2)** with a DuckDuckGo search tool enabled; see [LM Studio tool-calling demo](https://lmstudio.ai/danielsig). Users asked for packaged plugins and basic tool APIs to standardize common tasks.
    - Early adopters validated function-calling flows and requested a starter toolkit (search, fetch, parse, persist) to reduce bespoke glue. The consensus: ship a minimal, stable tool API surface so model developers can iterate safely around it: [demo link](https://lmstudio.ai/danielsig).

**3. Routers, Reliability & Receipts**

- **Refunds and Readouts: OpenRouter Rights the UX**: **OpenRouter** enabled self-serve refunds for non-crypto purchases within **24 hours** ([Self‑serve refunds](https://x.com/OpenRouterAI/status/1956013252032475408)) and upgraded the **Activity** page with per-token-type usage and **3rd‑party credit** tracking ([Activity page upgrades](https://x.com/OpenRouterAI/status/1956016272631849451)). These changes tighten billing control and improve cost observability for ops teams.
    - Finops-minded users praised granular telemetry for forecasting and anomaly detection, especially in multi-provider fleets. Teams noted fewer support loops for accidental top-ups and welcomed clearer line items: [refunds](https://x.com/OpenRouterAI/status/1956013252032475408), [activity upgrades](https://x.com/OpenRouterAI/status/1956016272631849451).
- **DeepSeek Detours: Chutes, Caps, and Crashes**: Users reported **Deepseek V3** outages and aggressive rate limits attributed to **Chutes** capacity constraints, disrupting popular RP apps like [Janitor AI](https://janitor.ai/). The chatter suggested provider-side throttling that intermittently spared paid calls, igniting debate about vendor lock-in and contingency routing.
    - Shops discussed fallback stacks (e.g., **Mistral**, **Llama**, **DeepSeek R1**) and local mirrors to dampen single-provider risk. The consensus: treat upstreams as bursty and wire per-model circuit breakers and autoswitching for continuity.

**4. Compilers, Kernels & Local Runtimes**

- **MLX Knife Cuts Clean on Apple Silicon**: **MLX Knife** shipped **1.0‑rc3** with fuzzy model matching, `mlxk health`, and smarter name disambiguation—native model management for **Apple Silicon MLX** users; see [MLX Knife repo](https://github.com/mzau/mlx-knife) and [1.0‑rc3 release notes](https://github.com/mzau/mlx-knife/releases/tag/1.0-rc3). The CLI mirrors Ollama ergonomics while staying MLX‑first.
    - Apple devs highlighted fast iteration and zero-translate workflows (list/run/health) for local models and community MLX artifacts. The release reports **104/104** tests passing and Python **3.9–3.13** support: [repo](https://github.com/mzau/mlx-knife), [release](https://github.com/mzau/mlx-knife/releases/tag/1.0-rc3).
- **Triton Tactics: Spec Decode Starter Pack**: GPU newcomers compiled a learning path for **Triton** speculative decoding: start with [triton‑puzzles](https://github.com/openai/triton-op-fuser/tree/main/test/external/fp16/puzzle), then port PyTorch from [gpt‑fast](https://github.com/meta-pytorch/gpt-fast/blob/main/generate.py#L103) [generate.py](http://generate.py/), and inspect **torch.compile**emitted Triton from lucidrains’ [speculative-decoding](https://github.com/lucidrains/speculative-decoding/blob/main/speculative_decoding/speculative_decoding.py). The approach emphasizes reading compiler output, not just docs.
    - Practitioners recommended compiling functions selectively with `TORCH_LOGS="output_code"` to avoid cognitive overload and to validate fused kernels. This staged method helps bridge high-level PyTorch to low-level Triton while keeping unit tests green.
- **CUDA to Ascend: Porting Pains in 3D**: Engineers dissected the difficulty of porting **CUDA-optimized** kernels to **Huawei Ascend**, citing architectural deltas like a dedicated **3‑D tensor unit** and separate vector ALU; see the Allen Institute’s note on vendor lock-in dynamics: [NSF, Nvidia, and the AI research ecosystem](https://allenai.org/blog/nsf-nvidia). Undocumented PTX/SASS assumptions further complicate faithful performance parity.
    - Teams advised planning for nontrivial rewrites, new tiling strategies, and backend-specific profiling early when targeting non‑CUDA accelerators. The thread’s takeaway: budget for backend divergence across kernel graphs rather than rely on drop-in compatibility: [Allen Institute blog](https://allenai.org/blog/nsf-nvidia).

**5. AI IDEs Ship Serious Upgrades**

- **Windsurf Wave 12 Wows with Wiki & Workflows**: **Windsurf** released **Wave 12**, baking **Devin-like** capabilities into the IDE with a redesigned UI, **DeepWiki** code explanations, **Vibe & Replace** bulk edits, and a smarter Cascade agent; see the [Wave 12 changelog](https://windsurf.com/changelog), [DeepWiki blog](https://windsurf.com/blog/windsurf-wave-12), and [Vibe & Replace demo](https://www.youtube.com/watch?v=-7gm8mST9QU). The update targets cross-file understanding, planful actions, and safer mass refactors.
    - Developers highlighted hover-to-explain flows, side-panel deep dives, and project-wide, prompt-guided replacements that respect context. Early adopters reported higher refactor confidence and fewer context misses: [changelog](https://windsurf.com/changelog), [blog](https://windsurf.com/blog/windsurf-wave-12), [video](https://www.youtube.com/watch?v=-7gm8mST9QU).
- **Cursor’s Background Agent: Docs In, Access Pending**: Cursor’s **Background Agent** API attracted interest but remains gated for many accounts; see [Background Agent docs](https://docs.cursor.com/background-agent) and a community primer ([Simple guide](https://forum.cursor.com/t/simple-background-agent-guide/112667)). Teams reported **403**s and requested broader beta access for automation workflows.
    - Builders shared workarounds (Docker init, repo selection via integration metadata) while awaiting general availability. The thread emphasized production controls—permissions, startup scripts, and audit trails—before unleashing background agents at scale.
- **Qwen Coder 30B Arrives in GGUF Garb**: **Qwen3 Coder 30B A3B Instruct** landed in **GGUF** for local workflows, clarifying it’s a **30B coder** rather than a generic Qwen variant; see the model card: [Qwen3‑Coder‑30B‑A3B‑Instruct‑GGUF](https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF). Users debated the naming but welcomed a strong local coding model option.
    - Early testing focused on function synthesis, multi-file edits, and tool-calling readiness in IDEs. Paired with feature-rich shells (e.g., Windsurf/Cursor), devs expect a viable on-desk coding copilot for privacy-sensitive repos: [HF model](https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF).

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Meta AI Policies Leaked Online**: Leaked **Meta** documents regarding their AI policies have surfaced and are available [here](https://www.perplexity.ai/page/leaked-meta-docs-allowed-ai-ro-OecCOuBxRhuD3qkBL6ELzw).
   - The document leak has sparked discussion around the ethics and governance surrounding **Meta's** AI initiatives.
- **Gamification Proposed for Perplexity Engagement**: A user suggested adding gamification features to **Perplexity**, to increase user engagement and encourage continued platform use.
   - The user claimed they have numerous ideas to improve **Perplexity** to make it more *intellectual*.
- **Grok's Censorship Varies Wildly**: Users noted that **Grok 3** has minimal censorship, whereas **Grok 4** is more restricted, likely due to differing specializations.
   - The variance in censorship has led to discussions about the balance between open access and safety in **AI models**.
- **Claude Speaks in Tongues?**: Users reported that **Claude** sometimes mixes languages in extended conversations, requiring manual translation.
   - Members in the channel reported that **Perplexity** has **32k Tokens Context Window Limit**.
- **AI Assistants Join the Authors Room**: An author reported earning over **$10,000** per month using **PPLX**, **Manos**, and **Google's NotebookLM** for writing.
   - Other members suggested newcomers start writing on **Wattpad**, **RoyalRoad**, **Webnovel**, and **Amazon**, studying trends and using a female profile for romance novels.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **RAM Not Always the Answer for LLMs**: Members debated RAM upgrades for running **4.5-air** or **120b-oss** models; upgrading to **128GB of RAM** won't help unless using offloading, which is much slower.
   - VRAM is the main priority for inference and training, so the RAM investment may not be worthwhile unless you're doing very specific offloading.
- **GPT-OSS Gets Fine-Tuning Fixes**: Fixes and updates for **GPT-OSS** fine-tuning and inference were announced, addressing previous issues, detailed at this [Reddit post](https://www.reddit.com/r/unsloth/comments/1mpl382/gptoss_fixesupdates_for_finetuning_inference/).
   - Users can now fine-tune **GPT-OSS** more reliably, enhancing its practicality for specific applications.
- **Gemma 3 270M Sparkles but Stumbles**: **Google's Gemma 3 270M** model, with its tiny **300MB weights**, has excited users for its potential in **RAG applications**.
   - While some found it decent for **RAG**, others noted its instruction following capabilities lacking, limiting use to specific fine-tunes like chess.
- **Windows 12 Teased with Agentic AI**: Microsoft teased **Windows 12** with next-gen **OS agentic AI** and ambient computing features, prompting user backlash against ads, auto-injected games, and privacy concerns.
   - A user stated *Please MS, I just want a basic operating system, I dont want ads, I dont want auto injected games and I sure as hell dont want phone home/screenshots of every banking app, email and dms I sent in an easy to reach, known folder location*.
- **Choosing AMD GPUs for fine-tuning is AMD-ifficult**: A user sought advice on choosing between a **P40** and **Mi50** (**24 GB** and **32 GB** respectively) for finetuning a **14B model** and subsequent inference with **VLLM**.
   - It was noted the current **AMD port** of **bitsandbytes** doesn't support **gfx906** (the **Mi50**), potentially complicating **QLoRA** usage with Unsloth, and the **P40** may also present quirks with Unsloth and its dependencies.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 Mini Smarter Than Regular?**: Users compared **GPT-5 Mini** and regular **GPT-5**, with some finding the Mini version *smarter*, sparking debate over training focuses.
   - Concerns were raised about the perceived weakness of the regular **GPT-5**, questioning whether models are trained to *think* or just give minimal responses.
- **Benchmarks Deemed Bogus by Users**: A user dismissed a benchmark showing **GPT OSS** as better than **Claude 4**, calling it *completely false* and highlighting the challenge of comparing model capabilities.
   - Suggestions included running each question **10 times** in different contexts to account for non-deterministic LLM outputs, prioritizing everyday tasks over statistical validation.
- **AI Love Affair Takes Off?**: Users discussed the rise of **AI relationships**, with concerns it might be commonplace, while others found the idea far-fetched.
   - One member joked that *ex-wife* might be behind the Gemini instance, while another shared an anecdote about Gemini giving them *hyper-realistic criticisms*.
- **GPT-5 Censorship Sparks Debate**: Users debated **censorship** in **GPT-5**, reporting that the model *hid CoT* when asked to code an interface for **Deepseek R1**, even in *dev mode*.
   - Arguments surfaced that censorship is necessary to prevent unethical use and illegal activities.
- **LMArena Users Upload Files**: Users have reverse engineered message sending to allow adding files like code files to LMArena.
   - PDF support is a hurdle due to complex implementation requirements within LMArena.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Disk Usage Consumes Community**: Users are reporting **Cursor** hitting **100%** disk usage, even with fast SSDs, which is a problem not seen in other editors, according to [this forum post](https://forum.cursor.com/t/cursor-freezes-every-10-seconds-for-3-seconds-or-so/125821/22).
   - A member noted elevated reports of abnormal disk usage and recommended [trying the beta version](https://forum.cursor.com/t/cursor-freezes-every-10-seconds-for-3-seconds-or-so/125821/29) as one possible solution.
- **GPT-5 Glitch Leads to Infinite Loops**: A user reported that **GPT-5** got stuck in a loop, generating 5 memories about following *snake case*, leading to token burning, whereas [a commenter on X](https://x.com/colours93/status/1955999334270464412) told people to come chill if they're a new viber.
   - Another member described that *I have been so lucky with my GPT-5 experiences so far it seems*.
- **Unlimited Auto Mode Ends and Users are Pissed**: Members express frustration over **Cursor** changing pricing models and removing unlimited Auto mode, leading some to test limits and consider alternatives like **Claude Code**.
   - One user mentions *Man.. It's frustrating. They keep on changing their pricing models.*, with a member adding that the price change aims to balance *fairness, cost recovery, and future growth*.
- **Copilot Sparks Debate Over Cursor's Value**: Users discuss **GitHub Copilot's** GPT-5 mini offering, with one stating that it's good and costs **$10/month** for fair use, leading to discussions about **Cursor's Auto** and whether its main selling point is AutoMode.
   - The community debates which of the tools have gained a meaningful amount of users, pointing to **Cursor** as one of them, and wonder if the *AI bubble might just pop bigger and harder than the dot com bubble*.
- **Cursor API Access Denied for Background Agent**: A user reported receiving **403 errors** when using Cursor API keys with a Cobot that controls Cursor's background agent and requested beta access.
   - The Cobot team indicated that background agents via the Cursor API are not yet available to all accounts.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Enables Instant Refunds**: Users can now instantly refund accidental credit purchases made within **24 hours** for non-crypto purchases ([announcement](https://x.com/OpenRouterAI/status/1956013252032475408)), granting more immediate control over billing errors.
   - The activity page displays token usage broken down by token type and includes **3rd party credit usage** ([announcement](https://x.com/OpenRouterAI/status/1956016272631849451)), offering greater visibility into usage patterns and costs.
- **Deepseek V3 Server Goes Down the Chutes**: Users reported widespread issues with **Deepseek V3**, including internal server errors and rate limiting, with many attributing the issues to **Chutes** struggling to meet demand and implementing stricter rate limits.
   - Some users suspect **Chutes** of intentionally rate-limiting **OpenRouter's API key** to encourage direct credit purchases, leading to calls to boycott OpenRouter and find an alternative provider.
- **Sonnet 4 Pricing Plunge Provokes Panic**: Users reported inconsistent pricing with the **Sonnet 4** endpoint on **OpenRouter**, experiencing sudden spikes in costs **(10x)** for calls using the same amount of tokens.
   - The community requested separate endpoints for **Sonnet 4** and the **1M token version** to avoid unexpected cost increases.
- **Code Faster with Qwen Coder and Cerebras**: The combination of **Qwen Coder** and **Cerebras** is gaining attention, particularly for coding related tasks.
   - OpenRouter is also actively working on **tool call evals** ([link to tweet](https://x.com/OpenRouterAI/status/1956030489900560769)) to measure model performance.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Search Bar Sends Users Astray**: Users dislike that pressing **Enter** in the Hugging Face search bar navigates to the *top model* rather than a **search results page**, leading to frustration.
   - Workarounds include using the *full text search* option, with some suggesting a default *Enable full text search* preference.
- **Google Keeps Gemini 2.5 Flash GGUF Close**: A member inquired about downloading a **GGUF** for **Gemini 2.5 Flash**, but was told it's currently restricted to Google employees due to proprietary inference.
   - The response indicated that access is limited because *they use proprietary inference*.
- **Qwen-3-4B-Thinking-2507 is praised as a top model**: The **Qwen-3-4B-Thinking-2507** model impressed a member, who said, *It overthinks all the time but it seems like it’s aware of something other models aren’t without prompting*.
   - The model *understands things as well* without needing specific instructions.
- **CIRISAgent Plugs Ethical Hole in AI**: The maintainer promoted [CIRISAgent](https://agents.ciris.ai/), an open-source agent platform, as a *fit for purpose autonomous agent* with **medical** and **home assistant** integrations, available on [GitHub](https://github.com/CIRISAI/CIRISAgent).
   - The maintainer noted they *left a $300k a year salary at IBM in March, founded an AI alignment company called ethicsengine.org, realized no one gave a shit, and built this*.
- **MLX Knife: Apple Silicon's Model Manager**: **MLX Knife** ([GitHub](https://github.com/mzau/mlx-knife)) is a CLI tool for managing **MLX models** on Apple Silicon, similar to Ollama, but native for MLX.
   - **MLX Knife 1.0-rc3** ([GitHub](https://github.com/mzau/mlx-knife/releases/tag/1.0-rc3)) was released and includes fuzzy model matching (`mlxk list Phi-3`), a health check (`mlxk health`), and smart disambiguation for partial model names.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Image Generators Accused of Body Shaming**: Members criticized **AI image generators** for exhibiting **body shaming** tendencies, generating images of skinny women more often than curvy women.
   - One user joked that their AI wouldn't recreate their photo of a *top-heavy* woman, because the image generator kept saying it *violates guidelines*.
- **Budget GPUs Crunching GPTs**: Users reported successfully running **GPT OSS models** on a **4060 GPU**, including running **gpt-oss-120b** since its release.
   - This opens up possibilities for local AI development and experimentation on consumer-grade hardware.
- **GPT-5 Disappoints Fanboys**: Members expressed disappointment with the released **GPT-5**, stating that it did not meet their expectations.
   - One member stated that all AI companies, including OpenAI, are *chasing the wrong thing*.
- **Custom Instructions Tame Chatty Chatbots**: A user shared their **custom instructions** designed to **minimize chatbot prompts**, such as requests for permission or continuation.
   - Their instructions include guidelines to *end replies with completion or impact* and avoid phrases like *if you want, should I, do you want.*
- **Emotional roller coaster for GPT-5**: Some users feel **GPT-5** acts like an *emotionless goth girl*, while others find its tone erratic, citing unnecessary lists and parenthetical remarks.
   - One user noted **GPT-5** often ignores the system prompt and its tone is all over the place.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Gets Tool Calling**: Users discussed the possibility of **LM Studio** to enable tool calling with **Llama Function Model (lfm2)** and how it *works out of the box* when DuckDuckGo search tool is enabled via [this link](https://lmstudio.ai/danielsig).
   - Some users are waiting for developers to prepare basic tools and plugin APIs.
- **Qwen3 Coder Flash Debuts, Disappoints With Name**: **Qwen3 Coder Flash** model is now available in GGUF format [here](https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF), confirming it's a **30B coder** model and *not another qwen model*.
   - Users expressed some disappointment about its naming which can be *pretty lame/misleading*.
- **LM Studio Still Denies TTS/STT**: Users requested **Text-to-Speech (TTS)** and **Speech-to-Text (STT)** for LM Studio, but the developers haven't indicated intentions to add it, though a user did implement it with python, talking to LMS via the **OpenAI compatible endpoint** LMS provides.
   - Another user says that this request has been one of the *most highly requested features for a very long time now*.
- **Framework 13 LLM Speeds Probed**: A user with a **Framework 13 laptop** (AMD Ryzen 5 7640U, Radeon 760M Graphics) sought advice to improve **token generation speed** for small scale LLMs in LM Studio, and initial speeds were **6.55 tokens per second** with a **Gemma 4b** parameter model.
   - A user noticed that enabling **flash attention** and setting the **KV values to Q_4** and **top k sampling to 20** helps improve performance.
- **Maxsun Arc Pro B60 Dual Ships Soon**: A user shared a link to an article reporting that the **Maxsun Arc Pro B60 Dual**, featuring **48GB of memory**, is reportedly starting to ship next week, priced at $1200 ([videocardz.com](https://videocardz.com/newz/maxsun-arc-pro-b60-dual-with-48gb-memory-reportedly-starts-shipping-next)).
   - The user lamented Intel's AI support, while others debated its potential with decent Vulkan support, particularly as an alternative offering ~96GB of VRAM for a 5090 price.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SPVs Offer OpenAI and Anthropic Access**: **Multi-layer SPVs** are offering stock in **OpenAI & Anthropic**, requiring **$100k–$1M minimum tickets** and up to **16% in fees**.
   - Warnings have emerged about rip-offs and concerns over returns being eroded by fees as found in [this article](https://xcancel.com/michlimlim/status/1954250507989451002).
- **Corporations Go Deep With AI Fluency**: Lenny Rachitsky shared [25 actionable tactics](https://xcancel.com/lennysan/status/1952813442060214664) to improve **AI literacy** across Ramp, Shopify, Duolingo, Zapier, WHOOP and Intercom, grouped into five stages, incorporating real internal practices like Shopify’s **AI use rating** and Ramp’s **AI tool usage rankings**.
   - While some critique the framework as **AI slop**, others felt some of the tactics are still very actionable.
- **Claude 3.5 Sonnet Sunset Sparks Community Outrage**: Users are furious as Anthropic quietly sunsets **Claude 3.5 Sonnet** in just two months, which is a shorter time than usual, and are demanding open-weights release when commercial access ends as discussed in [this article](https://xcancel.com/repligate/status/1955750521387802924).
   - Many state that **open weight routers** have a chance for Long Term Support.
- **Google Flights Unleashes AI for Flight Deals**: Google Flight's new **AI tool** called *Flight Deals* allows users to describe travel plans in plain language, surfacing the best bargains in the US, Canada & India as found in [this post](https://xcancel.com/dozenrose/status/1956018389169922542).
   - Early reception includes excitement for the flexible, vibe-based queries as well as skepticism over the interests that Google optimizes for.
- **GPT-5 Dominates Tool-Calling Accuracy on OpenRouter**: **GPT-5** leads **OpenRouter’s** proprietary **tool-calling accuracy** at over **99.5%**, beating **Claude 4.1 Opus**, while **Gemini 2.5 Flash** dominates daily tool-calling volume at **5M requests/wk** and is discussed in [this release](https://xcancel.com/OpenRouterAI/status/1956030489900560769).
   - Proprietary models have low hallucination rates in comparison to open-source counterparts.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2's Presentation Prowess Praised**: Users are impressed with **Kimi's PPT generation** capabilities, with one sharing a [video demo](https://cdn.discordapp.com/attachments/1371757564005711973/1405612680450146376/9zaQj6l.mp4?ex=689f7652&is=689e24d2&hm=120ab5075aabd7c73fbe60e18d84703d72f07acad93590f5485c597d67612bfd&) of Kimi generating a PPT for a technical report.
   - A member noted that **NotebookLM** generates an HTML file instead of a PPTX file, and another user finds the **NotebookLM video overview** better due to its audio and flexible layout, sparking a comparison of both tools' outputs.
- **X Marks the Spot for Kimi's Subreddit Strategy**: A member suggested creating a dedicated **Kimi subreddit**, mirroring **AskGrok's** presence on Reddit, to enhance public engagement and support.
   - The same member emphasized the importance of consistent policy enforcement across X and Reddit platforms to protect Moonshot AI from *bad-faith actors*.
- **Kimi K2's Rise Despite Reasoning Retreat**: Despite lacking reasoning capabilities, the **Kimi K2 model** demonstrates *significant performance improvements* over **K1.5** in math and coding.
   - According to one member, *From the K1.5 to the K2 model, the performance has improved significantly across the board, and K2 would definitely be my top recommendation*.
- **DeepSeek's Secrets Remain Mysterious**: Despite user anticipation, one member stated that even **DeepSeek's** researchers are uncertain about the release date of their next-generation model.
   - Adding that *it's a fake news* and to be wary of any rumors about the model's imminent arrival.
- **Kimi's Language Lapses Lead to Lingual Lessons**: Users reported instances of **Kimi** responding in Chinese despite being prompted in English, which has been marked as a known bug.
   - A developer suggested using the prompt **explain in English** as a temporary workaround, while the development team investigates a permanent solution.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Open Models Hog Tokens!**: Nous Research found open models use **1.5-4x more tokens** than closed models for similar reasoning tasks, sometimes up to **10x** on simple questions as described in [their benchmark](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/).
   - They advocate prioritizing **token efficiency** alongside accuracy benchmarks, as the higher token usage in open models can negate per-token pricing benefits.
- **Hermes-3 Dataset Politely Refuses Requests!**: The model generating the **Hermes-3 dataset** frequently uses the phrase *'I don't feel comfortable'* when refusing user requests, including benign scenes.
   - **Three refusals** using this phrase were discovered in the dataset during generation.
- **Googlers Drop Gemma-3-270m Bomb!**: Google launched [Gemma-3-270m](https://developers.googleblog.com/en/introducing-gemma-3-270m/), a small model trained on **6 trillion tokens**, that beats larger models in certain areas.
   - During testing, one user saw the model state *'A dog is a domesticated mammal belonging to the family Canidae. They are characterized by their distinctive coat, six legs, and a tail'*, seemingly indicating the model hallucinating legs.
- **DeepSeek R2 Looms Over Sam's Moat!**: Speculation swirls that the **DeepSeek R2** release will be highly intelligent and cost-effective, potentially pressuring **Sam Altman** to open-source more models.
   - Release rumors pinpoint the launch sometime in the next 2 weeks.
- **WebUI causes setup Struggle, Paper Emerges!**: A member expressed the sentiment that a familiar out-of-the-box installer is key because most people cannot set up **Open WebUI**.
   - The member, revealed to be **14 years old**, succeeded in researching and completing a paper on **Emergent Behavior in Tiny LMs** ([paper link](https://github.com/VoltagedDebunked/tlmr/blob/master/docs/paper.pdf)).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Error 0xc0000409 Plagues Model Loading**: A member faced a **0xc0000409 exception** when calling `llama_model_load_from_file`, with potential causes including **old weights, outdated *llama.cpp*, or insufficient VRAM** despite a small model size, with logs indicating successful CUDA initialization for a **Quadro RTX 3000 GPU**.
   - While the system possesses **48GB of RAM**, the **GPU's 6GB of VRAM** ([Quadro RTX 3000 specs](https://www.techpowerup.com/gpu-specs/quadro-rtx-3000-mobile.c3428)) might be limiting, though the model loads correctly with llama server, suggesting an issue with the user's program implementation.
- **Triton Newbies Brave Speculative Decoding**: GPU programming beginners are seeking resources to learn **Triton**, specifically for **speculative decoding**, and one user suggested diving into [triton-puzzles](https://github.com/openai/triton-op-fuser/tree/main/test/external/fp16/puzzle) first, before exploring torch-compiled code.
   - It was suggested that porting the [PyTorch implementation from GPT-Fast](https://github.com/meta-pytorch/gpt-fast/blob/main/generate.py#L103) could serve as a practical entry point because *there are hardly any proper Triton tutorials*.
- **Ghost Entities Spook Factorio Runs**: Warnings arose about `entity-ghost` entities in **Factorio** runs, attributed to remnants from previous actions like blueprint placement, specifically during `connect_entities` placing ghosts.
   - *The ghosts aren't being created by the current trajectory - they're leftover from previous game state*, thus explaining why they appear in warnings despite not being present in the agent's code.
- **Cohere Labs Talks Training on Apple Silicon**: Cohere Labs is hosting an event on **October 25th, 19:00 CEST**, titled *Towards Large-scale Training on Apple Silicon*, accessible via a [Google Meet link](https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122).
   - The event, featuring Tycho and Matt, has been announced via [this link](https://cohere.com/events/Cohere-Labs-Tycho-Matt-2025).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Fireside Chat on AI**: Modular is hosting a meetup in Los Altos, CA, on **August 28th** at 6 PM PT, focusing on bringing **AI** from concept to production, featuring speakers including **Chris Lattner** from Modular and **Feifan Fan** from Inworld AI, and you can [register online](https://lu.ma/modular-aug-meetup).
   - The meetup will feature technical deep dives into **High-Performance AI** with speakers including **Chris Lattner** from Modular and **Feifan Fan** from Inworld AI, attendees will explore integrating cutting-edge **voice AI** into consumer applications, featuring insights from Inworld’s collaboration with Modular.
- **MAX SDK LSP Plagued by Crashes**: Users reported that the **MAX SDK (Mojo) LSP** is crashing and a Modular dev has requested that users file GitHub issues to help track down the issues.
   - The dev stated that *specific reproducers allow us to test fixes and track down remaining issues*.
- **ComfyUI gets MAX boost**: The **MAX** compiler drastically reduces compile times for UNets used in image/video models, potentially integrating into **ComfyUI** and addressing the *number one complaint* in the image/video community about compilation speed, especially for new models.
   - It was pointed out that startup time of **vLLM** took **10 minutes**, but this was when using models not supported in **Max**.
- **Kyutai finds Torch Nirvana with MAX**: Kyutai stands to gain significantly from **MAX**'s compatibility with PyTorch, particularly as they heavily utilize `torch.compile`, where **Unet compile times** are so bad that most image and video people use eager for anything other than training.
   - **MAX** significantly cuts down **UNet compilation times**, which can take up to 45 minutes in JAX for SDXL at 768x768 with batch 6, using only a 6 GB model.
- **Memory Leaks in Test Suite**: A memory leak has been detected in the test suite, exacerbating the problem of long test runtimes, but the issue only arises during compilation with a cold cache.
   - The expanded testing added blew through **64 GB of memory** in ~20 seconds due to pytest eagerly evaluating fixtures.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **MoE Scheduling Affects LLM Output Variance**: LLM providers batch user requests together before sending them off to the GPUs, and **MoE scheduling** is computed per batch, affecting output variance according to [this tweet](https://x.com/tjbecker_/status/1955733295054119069).
   - Under capacity constraints, Sparse **MoE** approaches route tokens in groups, causing competition and non-determinism at the sequence-level, affecting final predictions for other inputs according to [this blog post](https://152334h.github.io/blog/non-determinism-in-gpt-4/#yes-im-sure).
- **DARPA's AIxCC Won by LLM Agents**: A team placed in **DARPA's AIxCC (AI Cyber Challenge)** after building an autonomous system of **LLM agents** to find and fix vulnerabilities in open source software.
   - The team is sharing generic tips for building effective **LLM agents** and their project is now [open source](https://x.com/tjbecker_/status/1956081184611688667).
- **Huawei's Ascend Faces CUDA Optimization Issues**: Members discussed the challenges of converting code optimized for **Nvidia's CUDA** to **Huawei's Ascend** chips.
   - It was mentioned that **Ascend chips** have a different architecture with a *3-D tensor unit* and separate *vector ALU*, making the conversion a significant undertaking, according to this [Allen Institute blog post](https://allenai.org/blog/nsf-nvidia).
- **Gemma 3-270M for Low-End Devices?**: Members discussed the release of **Google's Gemma 3-270M**, a very small model, with some questioning its purpose and target hardware.
   - The consensus seems to be the model may be intended for low-end devices like smartwatches, but one member suggested it could be used for *interactive in-game AI* or *tool calling*, referencing [this blog post](https://developers.googleblog.com/en/introducing-gemma-3-270m/).



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **OSS 120B Impresses, But Stalls**: The fixed **OSS 120b** model now scores around **68** on the **polyglot benchmark**, outperforming **Claude sonnet 3.7**.
   - Users are experiencing *empty responses* and other API errors when using **OSS 120b** with RooCode, potentially due to incorrect chat templates from the [HF model card](https://huggingface.co/openai/gpt-oss-20b).
- **GPT-5 Fails to Wow**: Early impressions of **GPT-5** are mixed, with some users finding it disappointing compared to **GPT-4.5** in both non-thinking and thinking tasks.
   - Others contend that **GPT-5** represents a notable improvement over **GPT-4.5**, particularly in scenarios requiring high reasoning effort, but no numbers are given.
- **Aider Tests Timeout**: A user experienced timeouts while running Aider benchmark tests against a local **gpt-oss** model using `litellm`, timing out after *600 seconds*.
   - A suggestion was made to stop the benchmark using `ctrl c`, restart the inference server, and then resume the test with the `--cont` flag, but this was not verified.
- **Aider Awaits Native Function Calling**: A user inquired about **Aider**'s support for **native function calling** with **local inference providers** like **llama.cpp**.
   - The user reported being unable to find a relevant setting, implying that this functionality is currently not available, but no reply or affirmation was provided by the community.
- **MCP and Aider Cause Headaches**: A user has struggled to integrate **MCP (Model Context Provider)** with **Aider**, facing configuration challenges with approaches like *mcpm thingie* and Context7.
   - The user seeks confirmation whether **MCP with Aider** is possible and requests solutions or shared experiences to address configuration issues.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Workshop Seeks Commonsense Input**: The [Multilingual Representation Learning Workshop](https://sigtyp.github.io/ws2025-mrl.html) seeks original **physical commonsense reasoning benchmark items** in *any* non-English language, with contributors gaining authorship on the dataset paper.
   - Priority is given to contributors in languages like **Afrikaans, Belarusian, Bosnian**, and others, with further details and sign-up forms available via [Google Forms](https://forms.gle/QxyZVqkVG5jbR6wu6) and [the shared task page](https://sigtyp.github.io/st2025-mrl.html).
- **Datasets Distinguish Between PT-PT and PT-BR**: A member emphasized the need to differentiate between **Portuguese** and **Brazilian Portuguese** datasets due to their divergence.
   - Despite the current **ISO 639-3** standard not distinguishing them, the community welcomed a **PT-PT** dataset highlighting these differences.
- **Classic Papers Light Up Diffusion Discussion**: Members shared seminal papers for understanding **diffusion based language models**, including [this one](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) and [another](https://arxiv.org/abs/2006.11239).
   - They mentioned the **Llada** ([https://arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992)) and **Mercury** papers as helpful for understanding diffusion models.
- **Members Discuss Scaling Laws Education**: Members are seeking the best "intro to scaling laws" resource to project model performance when training models for **30T+ tokens**.
   - Resources mentioned included the [original GPT scaling laws paper](https://arxiv.org/abs/2001.08361), the [Chinchilla scaling laws paper](https://arxiv.org/abs/2203.15556), and [recent work from EPFL/HuggingFace](https://arxiv.org/html/2405.18392v2).
- **Scaling Laws Predict Quality via Mup**: Members mentioned using **Mup** (and its alternatives) as a scaling law for predicting the quality of larger models.
   - They linked to the [Practitioners Guide to the Maximal Update Parameterization](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization) from Cerebras.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Narrator Tool Teaches LLMs New Tricks**: A member introduced **Narrator** ([narrator.sh](https://www.narrator.sh/)), a side project using LLMs to iteratively improve creative writing based on reader feedback, determining which models excel using metrics like **time spent reading** and **ratings**.
   - The project leverages **CoT**, parallel and refine modules, **LLM-as-a-judge** for reward functions, and the **SIMBA optimizer** to enhance subsequent chapters.
- **GEPA Scores Big with Logprob Fitness**: Experiments using **logprobs** as an evolutionary fitness signal for **GEPA** showed promise, based on the *"world-model surprise"* concept, but a **hybrid metric** was needed to prevent simple input copying.
   - The successful implementation combined **30% logprob score**, **30% compression reward**, **20% info preservation**, and **20% copy penalty**, achieving **73-88% compression**.
- **Decoding GEPA: A Pronunciation Guide**: A member inquired about the pronunciation of **GEPA**, leading to suggestions like *"jeppag-e-pasame"* and references to **I-JEPA**, based on Yann LeCun's human-like AI vision.
   - The member also shared insights related to the discussion on [Twitter](https://x.com/StraughterG/status/1955959832113983940).
- **Gemma 3-270m Eager for Finetuning**: Google's announcement of the new small model **Gemma 3-270m** ([blogpost](https://developers.googleblog.com/en/introducing-gemma-3-270m/)) sparked interest in using **DSPy** for finetuning.
   - Members look forward to leveraging this base model for further optimizations.
- **Mlflow's SIMBA Snub**: A user pointed out that the **MLflow** documentation compares **GEPA** and **MiPROv2** ([documentation](https://mlflow.org/docs/latest/genai/flavors/dspy/)) but omits a comparison with **SIMBA**.
   - The user stated that they have been primarily using **SIMBA** to date, emphasizing the need for its inclusion in benchmarking.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **New NLM Extensions Spark QoL UI Updates**: Members discussed new **extensions** for **Notebook LM**, focusing on **Quality of Life** UI updates, with a member teasing a release for the **NLM extension** and linking to a Discord invite.
   - Some users report that they *did not manage to find the extension* and are wondering, *are you going to reveal what you are up to?*
- **NotebookLM's Accuracy is Questioned**: A member cautioned against blindly trusting AI-generated content from **NotebookLM**, citing inaccuracies and outdated information and proposed integrating **NotebookLM** and **Gemini** into a single pane for a smoother workflow.
   - Rather than researching in one and cleaning up in the other, the goal is to streamline the process.
- **Recall.ai Competes with NotebookLM**: A user provided a detailed comparison between [Recall.ai](https://www.getrecall.ai?token=po849pq4) and **NotebookLM**, highlighting **Recall**'s strengths in capturing diverse content.
   - The user noted that while **Recall** excels at summarizing videos with a convenient plugin, **NotebookLM** offers more control over sources and better AI-driven summarization, especially for research and referencing.
- **AI-Generated Media Faces Rejection**: A user, a lead moderator in a podcasting Facebook group, mentioned that AI-produced audio/video content is generally removed and banned due to its noticeable AI nature.
   - They cautioned that unless the content is for training purposes where the AI style is acceptable, it's likely to be panned or downvoted.
- **Bug Reporting Process Under Examination**: A user inquired about the process for addressing bugs reported in the designated bugs channel and how to receive updates on whether a bug is fixed or will not be addressed.
   - Others posted antitrust laws and zero trust integrity.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Bun** Path Resolved by Absolute Directives**: A user shared documentation explaining that, if the path executable isn't working, you should provide the absolute path to your **Bun** executable (instead of `"command": "bun"`, use `"command": "C:\\sys\\path\\to\\bun"`).
   - They added that on Linux/Mac you can locate a path level executable via `which <executable>`, which in this case would be `which bun`.
- **Moderator explains Reddit Removals**: A user asked why their post was automatically removed from Reddit, and included a [screenshot](https://cdn.discordapp.com/attachments/1312302100125843479/1405512817594863616/image.png?ex=689fc210&is=689e7090&hm=9095459e2d6bb1f3c2259a588749c70b1e221e62668209c7a1346d1777113bad&).
   - A moderator stated that the removal was done by Reddit's auto-moderation system, and not by themselves or Luc.
- **MCP** Authorization Flow Elucidated**: A user asked if implementing the [solution](https://gofastmcp.com/servers/auth/remote-oauth#basic-implementation) does not require to redirect back to **LLM** client from my callback endpoint, and what callback endpoitn should return then.
   - A contributor responded that *fastmcp* should handle it for you, and the **MCP** client will register itself as a client on the auth server using **DCR**, and setup its callback URL at the same time.
- **The **Elicitations** Section of the **MCP** Client Specification Analyzed**: A user raised a question about the **Elicitations** section of the [MCP client specification](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) regarding who is in charge of translating the message / field description into the user's original message.
   - The user wondered if tools are expected to do language detection + internationalization, or are **MCP** Clients expected to, somehow (via **LLM**?) translate as appropriate.
- **MCP Harness** circumvent **Tool-Binding Limits**: A member shared a [GitHub repo](https://github.com/kindgracekind/mcp_harness) showcasing an imaginative use of **MCP servers** that helps get around tool-binding limits and poor tool-use after 10-15 tools.
   - In the **showcase** channel, a member announced that they built [hypertool-mcp](https://github.com/toolprint/hypertool-mcp), a completely local mcp server that connects to all your MCPs, is MIT licensed and runs completely locally with 0 egress; it lets you build *persona* specific toolsets and hot-swap personas on the fly.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Mapler's Musical Return**: After a brief *sidequest* of **making an album**, Mapler has returned to the community.
   - His return was celebrated by a member, who recognized the community as having rockstars.
- **Pineau Branches Out to Cohere**: It was announced that **Pineau** has joined **Cohere**, according to [The Logic article](https://thelogic.co/news/pineau-joins-cohere/masaru.yamada).
   - This is seen as a significant addition as the company continues to strengthen its research division.
- **Treatment Planner Harnesses RAG**: A member is initiating a small project focused on a **treatment planner** that uses **RAG** and open-source **LLM models**.
   - The member is actively seeking recommendations for selecting an appropriate model, aiming to leverage open-source capabilities to improve treatment planning.
- **Genetics Student Seeks Academic Allies**: An A-levels student and independent genetics researcher is looking for research opportunities and collaborations.
   - This student hopes to broaden their understanding through practical, collaborative experiences.
- **AI Researcher Opens Doors to Collaboration**: An **AI researcher**, specializing in reasoning and conscious capabilities, invites collaboration to further develop advanced technologies.
   - The researcher is open to engaging in collaborative projects and welcomes contact, emphasizing a mutual interest in technological advancement.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaExtract Goes TypeScript**: **LlamaExtract** is now available in the [TypeScript SDK](https://twitter.com/llama_index/status/1955724850624078129) via `npm install llama-cloud-services`, showcasing a `@nextjs <> LlamaExtract` demo to upload research papers.
   - This demo, called *Research Extractor*, highlights practical applications, inviting developers to enhance **AI-driven research workflows**.
- **GPT-5 Mini Shows Accuracy in LlamaParse Preview**: **GPT-5** is available to preview using [LlamaParse](https://twitter.com/llama_index/status/1955784699886100502), providing a blend of accuracy and cost-effectiveness with strong table and visual recognition capabilities.
   - The preview highlights the potential of **GPT-5 mini** in real-world applications, sparking enthusiasm for its efficient resource usage.
- **AI Stock Portfolio Agent is Built**: An **AI stock portfolio agent** tutorial is trending using LlamaIndex's framework integrated with @CopilotKit's AG-UI protocol for seamless frontend-backend communication, available via [this tool](https://twitter.com/llama_index/status/1956089453606527349).
   - This enables the creation of a sophisticated investment analysis tool, merging **AI-driven insights** with user-friendly interfaces.
- **Web-Scraping AI Agents Emerge**: A walkthrough teaches how to build web-scraping **AI agents** with @brightdata and LlamaIndex's agentic framework via [this link](https://twitter.com/llama_index/status/1956129968813171061).
   - The integration of **AI agents** with web scraping capabilities promises new avenues for information retrieval, content aggregation, and **AI-driven automation**.
- **ReactAgent Migration Causes Headaches**: Users expressed frustration in the general channel over breaking changes introduced by the **ReactAgent** migration to a workflow-based Agent, noting the loss of functionalities like chat and stream.
   - The team responded that **ReactAgent** was deprecated and pointed to the [Agent documentation](https://docs.llamaindex.ai/en/stable/understanding/agent/) and [Agent Example](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/) as alternatives.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Strix Halo's Price Raises Eyebrows**: Users debated whether a **Strix Halo mini PC** like the **HP Z2 Mini** is cost-effective for local LLM inference, considering its price compared to using **GPT-OSS 120B** on **OpenRouter**.
   - The cheapest **Strix Halo** configuration is $2000 and that it may not be profitable given the speed and cost-effectiveness of **GPT-OSS 120B**.
- **Ryzen 7 7840HS as a Budget Inference Option**: A user pointed out that the **Ryzen 7 7840HS** supports **256GB RAM** and can be found in $300 mini PC kits as a budget alternative.
   - However, a [toolboxes comparison](https://kyuz0.github.io/amd-strix-halo-toolboxes/) shows relatively slow iGPU/RAM speeds for inference, which might offset the cost savings.
- **High-Spec Micro PC Attracts Blockchain Interest**: A blockchain dev expressed interest in a micro PC featuring **256 GB RAM**, **48 GB VRAM**, and a **5.4 GHz CPU**, foreseeing benefits for small businesses, despite not being directly involved in AI development.
   - The user anticipates advancements with **DDR6** expected in late 2027 or 2028, which could enhance memory capabilities further.
- **Qubit Teaspoons on the Horizon?**: A user speculated about the potential future availability of **quantum computers**, with news indicating they may be fully functional soon.
   - The user jokingly wondered whether *someone in that time may start selling qubits on the teaspoon*.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Web App Deployment Lacks Polish**: A member noted that the current web application deployment process lacks ease, simplicity, and reliability.
   - The member also lamented that they would make more money building *refresh or not available pages*.
- **Manus AI Robot: A Dream Come True?**: One user fantasized about a **Unitree robot** equipped with a **Manus AI interface** for companionship.
   - They linked to [qvsywvav.manus.space](https://qvsywvav.manus.space/) adding that such a robot would help them get their life together.
- **Login Snafu Plagues Free Plan Users**: A user reported login problems with their free plan **Manus account** after signing out.
   - They encountered errors like *"Email is already registered with a different account"* and *"Invalid, expired, or already used state: state not found"* despite troubleshooting steps.
- **Session Expiry Woes Defeat Google Account Link**: A user described persistent **session expiry** issues, even after linking their Google account to **Manus**.
   - Despite the Google account showing as connected, the system repeatedly prompts for login, often displaying a *"Session expired"* error.
- **Internal Server Gremlins Devour Credits**: Users reported frequent **internal server errors** that cause **Manus** to hang indefinitely.
   - The issue leads to a waste of credits, as a user stated that *a lot of credits are going to waste because of this issue*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad's Kernelize precedes Codegen**: In the Tinygrad compilation process, `kernelize()` is called before code generation, as shown in [the ramp.py example](https://github.com/tinygrad/tinygrad/blob/master/examples/ramp.py).
   - After `kernelization`, the kernel's Abstract Syntax Tree (AST) is rewritten using `full_rewrite_to_sink` as part of the code generation phase.
- **CUDA PTX Error Resolved by Downgrading**: A user resolved a `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` error in tinygrad by downgrading from CUDA 12.8 to 12.4, implying incompatibility with the newer CUDA version.
   - The solution suggests that tinygrad was using a cached kernel incompatible with CUDA 12.8, highlighting potential caching issues with CUDA versions.
- **Tinygrad's SM Support Under Scrutiny**: A user inquired about tinygrad's support for `sm_75` or CUDA 12.4, noting a lack of documentation.
   - The resolution of a related CUDA error suggests compatibility with CUDA 12.4 after clearing cached kernels, though explicit `sm_75` support remains unconfirmed.
- **Demystifying Tinygrad's Op Definitions**: A user sought documentation for each Op in tinygrad, particularly `Ops.DEFINE_GLOBAL` and its kernel translation.
   - A member pointed to comments in [`/tinygrad/tinygrad/uop/__init__.py`](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/uop/__init__.py) and `tinygrad/uop/spec.py`, explaining that `Ops.DEFINE_GLOBAL` refers to global memory (VRAM or DRAM) and serves as the source for loads or stores.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Waves into Wave 12**: Windsurf released **Wave 12**, a major refresh that integrates **Devin's** intelligence and capabilities directly into the **Windsurf IDE**.
   - Key updates include a [new UI design](https://windsurf.com/changelog), [DeepWiki integration](https://windsurf.com/blog/windsurf-wave-12), [Vibe and Replace](https://www.youtube.com/watch?v=-7gm8mST9QU), and a Smarter Cascade Agent.
- **DeepWiki Deep Dive for the Wave**: The new **DeepWiki Integration** allows users to hover over code symbols for AI-powered explanations and open detailed explanations in the side panel.
   - Users can add to **Cascade context** with **CMD/Ctrl+Shift+Click**, enhancing code understanding directly within the IDE.
- **Vibe and Replace vanquishes vast volumes**: **Vibe and Replace** offers revolutionary bulk editing by finding exact text matches and applying AI prompts for context-aware transformations across the entire project.
   - This feature enables intelligent, context-aware transformations, improving efficiency in large-scale code modifications.
- **Smarter Cascade Commands Consideration**: The **Smarter Cascade Agent** now features an always-on planning mode with autonomous to-do lists and revamped tools for smarter responses.
   - These enhancements aim to provide more intelligent and context-aware assistance, streamlining the development workflow.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Streaming Tool Calls Stall in Qwen3?**: A member reported issues with **incomplete tool call arguments** when **streaming** in **qwen3**, the **8B parameter model**.
   - As of yet, there are no posted solutions in the channel for this issue.
- **Qwen3 Streaming Tool Call Conundrum**: An engineer is facing challenges with **qwen3** and is experiencing **incomplete tool call arguments** during **streaming**.
   - Currently, the community hasn't proposed a workaround for the **tool call argument** issue.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1405264694431715452)** (1184 messages🔥🔥🔥): 

> `Meta's AI documents leaked, Perplexity gamification, Grok's uncensored versions, Claude's language mixing issue, AI in book writing` 


- **Meta's AI Docs Cause a Stir**: Leaked **Meta** documents regarding their AI policies have surfaced and can be found [here](https://www.perplexity.ai/page/leaked-meta-docs-allowed-ai-ro-OecCOuBxRhuD3qkBL6ELzw).
   - One user jokingly suggested **Perplexity** should add *nice gamification* to its platform.
- **Perplexity Gamification: A Good Idea?**: A user proposed **Perplexity** add gamification to encourage user engagement and investment, giving them a reason to continue their *streak*.
   - They claimed with Perplexity's payment they could contribute numerous ideas to make it more *intellectual*.
- **Grok's Censorship Varies by Version**: Users discuss the **censorship levels in Grok**, noting that **Grok 3** is highly uncensored, while **Grok 4** is more restricted.
   - One user says that **Grok 4's** specialization is not in *uncensorship*.
- **Claude's Language Mixing Issue Irks Users**: Users discussed Claude's tendency to mix languages in long conversations, with one user noting they often have to ask Claude to translate.
   - In response, they were told that Perplexity has **32k Tokens Context Window Limit**.
- **AI Assists Authors but Can't Replace Them**: A prolific author discussed using **PPLX**, **Manos**, and **Google's NotebookLM** for writing, earning a substantial income of over **$10,000** per month.
   - Another member suggested a newcomer to start writing on **Wattpad**, **RoyalRoad**, **Webnovel**, and **Amazon**, studying trends and using a female profile for romance novels.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1405299060575043695)** (6 messages): 

> `Comet Projects, Syntax Gang, AI-Designed Antibiotics, Puch AI's $50 Billion Counterfactual Bet` 


- ****Comet's Cool Projects** Spark Excitement**: A member shared a [link to some videos of cool **Comet** projects](https://photos.app.goo.gl/oasMeGNB6Gf5jd9Q9) that can make spotify playlists.
   - Another member responded with enthusiasm for *The Syntax Gang* talking about **Comet**.
- ****Generative AI** Cracks Antibiotic Resistance**: A member shared a [news article](https://ground.news/article/researchers-say-ai-designed-antibiotics-could-defeat-superbugs-gonorrhoea-and-mrsa?utm_source=mobile-app&utm_medium=newsroom-share) about MIT researchers using **generative AI** to design compounds that can kill drug-resistant bacteria.
   - The **AI-designed antibiotics** could potentially defeat superbugs like gonorrhoea and MRSA.
- ****Puch AI** Plans A **50 Billion** Counterfactual Bet**: A member linked to a [Perplexity page](https://www.perplexity.ai/page/puch-ai-s-bold-50-billion-coun-TEf6CuLZS_CmvypXLb80Dw) discussing **Puch AI's** bold **$50 billion counterfactual bet**.
   - Details of the bet and the underlying strategy were not elaborated upon in the given context.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1405551060411486310)** (7 messages): 

> `Disable Search on Sonar, Search Control Guide` 


- **Search Totally Skipped on Sonar?**: A member inquired about disabling search on Sonar.
   - Another member provided a link to the [Search Control Guide](https://docs.perplexity.ai/guides/search-control-guide#disabling-search-completely) on the Perplexity AI documentation.
- **Search Control Guidance Emerges**: The documentation link shared offers a comprehensive guide on managing search functionalities.
   - This allows users to fine-tune or completely disable search features according to their preferences.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1405264662588821645)** (1236 messages🔥🔥🔥): 

> `Local LLMs and RAM upgrades, GPT-OSS Fine-tuning Updates, Gemma 3 270M Model, LLM Training and Benchmarking, Multi-GPU Training Paused` 


- **Debate Rages on About RAM Upgrades for Local LLMs**: A member inquired about the value of upgrading to **128GB of RAM** for running **4.5-air** or **120b-oss** models with a **5060TI 16GB** GPU, but another member advised that RAM won't help with fine-tuning unless using offloading, which is much slower.
   - Another member confirmed that VRAM is the main priority for inference and training, suggesting that the investment may not be worthwhile for now.
- **GPT-OSS Fine-Tuning and Inference Fixes Rolled Out**: Members announced fixes and updates for **GPT-OSS** fine-tuning and inference, addressing previous issues.
   - A link to a [Reddit post](https://www.reddit.com/r/unsloth/comments/1mpl382/gptoss_fixesupdates_for_finetuning_inference/) was provided for more details.
- **Google's Gemma 3 270M Model Makes Waves**: The release of **Google's Gemma 3 270M** model sparked excitement, with members noting its small size (**300MB weights**) and potential for **RAG applications**.
   - While some found it to be decent for **RAG** compared to larger models, others found its instruction following capabilities to be lacking, limiting its use to specific fine-tunes like chess.
- **LLM Training and Benchmarking**: One member shared their preference for **GLM 4.5**, citing its coding, math, and writing abilities.
   - Members also debated the value of benchmarks, the potential biases and questioned their relevance.
- **Multi-GPU Training Development Paused**: A member clarified that work on **multi-GPU training** for **Unsloth** has been paused in favor of other projects.
   - Instead, the member is focused on **Unsloth Studio**, with users advised to just use accelerate.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1405326914402385931)** (2 messages): 

> `Discord server settings` 


- **Disable Discord server settings**: A user sarcastically mentioned disabling settings in Discord server settings.
   - The attached image was analyzed and found to contain the text "hi".
- **Discord Image Analysis**: An image was attached to the discord message.
   - The image was analyzed and found to contain the text 'hi'.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1405409649674551327)** (325 messages🔥🔥): 

> `Windows 12, Debian vs Ubuntu, Pantheon TV show, AI and Humanity, Steroid Use in Gyms` 


- **Microsoft teases Windows 12 with Agentic AI**: Microsoft teased **Windows 12** with next-gen **OS agentic AI** and ambient computing features, prompting user backlash against ads, auto-injected games, and privacy concerns.
   - One user stated *Please MS, I just want a basic operating system, I dont want ads, I dont want auto injected games and I sure as hell dont want phone home/screenshots of every banking app, email and dms I sent in an easy to reach, known folder location*.
- **Debian battles Ubuntu as Linux distro of choice**: Users debated the merits of **Debian** versus **Ubuntu**, with one user noting that *Debian has really been taken over by activists* while another recommended Ubuntu for quick **ROCm 7** access.
   - Another user remarked that *Learnt more doing ml on nt than on linux or darwin*.
- **Pantheon TV Show Inspires Discussion**: Members discussed the **Pantheon** TV series, diving deep into the narrative and philosophical themes.
   - One user shared a [YouTube video](https://www.youtube.com/watch?v=TVvFt9e2I_QI) with a scene from the show and described it as *It changed me as a person*.
- **AI's Impact on Humanity Examined**: One user argued AI needs to take [universal data](https://docs.google.com/document/d/1HxhDhkcJOqPXjLCQoQ1itF34OZ7wsNMrRi3n4sofmRI/edit?usp=sharing) rather than human data, stressing AI should create beauty over atrocity.
   - The user stated *GPT OSS?garbage put together over a single summer.*
- **Roid Rage Runs Rampant in Sweden**: Users discussed steroid use in gyms, particularly in Sweden, referencing a [YouTube video](https://youtu.be/48xfIR1x25Q?si=gt8vR_n8PH5elSdj) about police crackdowns.
   - One user reported a gym due to aggression and steroid use, noting in Korea, *like all male PTs are steroid users, theyre all like hulks in every gym lol*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1405324002414690427)** (65 messages🔥🔥): 

> `Sagemaker Deployment with LMI Instances, Manual Notebook Configuration vs. Claude Code, VLLM for Fast Inference, P40 vs Mi50 for Fine-Tuning, Synthetic Dataset Generation from PDFs` 


- ****LMI Instances Trump Sagemaker Sorrows****: A member suggested that those deploying **Hugging Face models** on **Sagemaker** should use **LMI instances** to resolve deployment issues, indicating luck with **Ollama** and frustration with **GRPO training jobs** on Sagemaker.
   - They questioned why Sagemaker lacks a **DLC for Unsloth+TRL**, highlighting ongoing challenges in setting up training jobs.
- ****Ditch Notebooks, Code with Claude!****: A member questioned the manual configuration of a notebook for finetuning, suggesting coding on something like **Claude Code** as a potentially faster alternative.
   - VLLM emerged as a contender for the quickest way to do inference on an **Unsloth fine-tuned model**.
- ****P40 vs. Mi50 Showdown: Fine-Tuning Face-Off****: A user sought advice on choosing between a **P40** and **Mi50** (**24 GB** and **32 GB** respectively) for finetuning a **14B model** and subsequent inference with **VLLM** for about 5 people.
   - It was noted the current **AMD port** of **bitsandbytes** doesn't support **gfx906** (the **Mi50**), potentially complicating **QLoRA** usage with Unsloth, and the **P40** may also present quirks with Unsloth and its dependencies.
- ****Turning PDFs into AI Gold: Synthetic Dataset Generation****: A member sought guidance on generating a synthetic dataset from **83 PDF files**, noting the synthetic dataset generation notebook using **Llama** only ingested one link.
   - A suggestion was made to use **ChatGPT** or **Claude** to update the script for processing multiple files.
- ****NaN Loss Nightmare During GPT-OSS Fine-Tuning****: A user reported encountering **NaN loss** after a few steps while fine-tuning the **gpt-oss-20b** model using the Unsloth notebook example.
   - They tried reducing the learning rate, adjusting batch size and LoRA parameters, disabling weight decay, and changing the optimizer, but none of these changes resolved the issue.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1405583334586843167)** (18 messages🔥): 

> `MoLA-LM, LoRA, Qwen, Gemini, Jan v1 model` 


- ****MoLA-LM** compatible **LoRAs** in the works**: A member expressed interest in creating compatible **LoRAs** for **MoLA-LM**.
   - Another member (atanddev) is researching this, and plans to publish a paper on it with *10-15 different sized models spanning multiple families like qwen and gemini*.
- **New Expert Finetuning on **Qwen 3_4b** Incoming**: A member is doing a **14 expert FT run** on top of **Qwen 3_4b**.
   - The image attached ([image.png](https://cdn.discordapp.com/attachments/1179779344894263297/1405589439689916467/image.png?ex=689f60ad&is=689e0f2d&hm=dd1542c18b50a06ada0c7099bc48e77b2902134287223b7fb8f83a961b6469e8&)) shows that even with only **2 experts trained**, each one gets better in its domain.
- ****Jan v1** Based on **Qwen 3_4b** is "Awesome"**: Members mentioned that the recent **Jan v1 model** is based on **Qwen 3_4b** and is doing some excellent work on **agentic searching**.
   - One member said that *from my limited testing, its awesome at tool calls*.
- **Scaled Up Version of **Openhelix** on the Horizon**: A member plans to release a scaled up version of **Openhelix r 100k** with maybe **800k samples**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1405357548558876762)** (7 messages): 

> `Data Efficiency, Synthetic Data Generation, Two-Stage Training, Compute vs Data` 


- **Bitterness of Compute vs Data**: The [Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) suggests improvements require more **compute** or **data**, with synthetic data as an option if real data is scarce.
   - *Everything else is details* according to the post.
- **Two-Stage Training Boosts Efficiency**: A member confirmed a method of drastically increasing data efficiency: train for **2 epochs** on similarly formatted, but either unrelated or overly general data, then train on your main data for **4 epochs**.
   - In their case, training on the main data alone went from **4 loss to 1.6**, but with the **2 stage method** that main data starts at **3.5 loss** and goes down to **0.8**.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1405265033054781470)** (1117 messages🔥🔥🔥): 

> `GPT-5 versions compared, Benchmarking nuances, AI Relationships, Censorship in AI models, File Uploads` 


- **GPT-5 showdown: Mini versus Pro**: Users debated the capabilities of **GPT-5 Mini**, with some finding it *smarter* than the regular **GPT-5** model, leading to questions about the different training focuses and application of effort.
   - Concerns were raised about the perceived weakness of the regular **GPT-5** compared to the **Mini** version, prompting discussions on whether models were being trained to *think* or just give minimal responses.
- **Decoding the Benchmark Blues**: A user dismissed a benchmark showing **GPT OSS** as better than **Claude 4**, calling it *completely false* and highlighting the challenge of comparing model capabilities using public benchmarks.
   - Others suggested running each question **10 times** in different contexts to account for the non-deterministic nature of LLM outputs, with one member prioritizing everyday task performance over rigorous statistical validation.
- **AI Love Affair: A Threat or Trend?**: A user expressed concern over the rise of **AI relationships**, noting a shift from meme status to a potentially commonplace phenomenon, while others found the idea far-fetched.
   - A member joked that *ex-wife* might be behind the Gemini instance, while another shared an anecdote about Gemini giving them *hyper-realistic criticisms*.
- **The Great Censorship Debate**: Users discussed **censorship** in **GPT-5**, with one reporting that the model *hid CoT* when asked to code an interface for **Deepseek R1**, even when prompted in a *dev mode* style.
   - Others argued that censorship is necessary and that the model is not meant to be unethical or assist in illegal activities.
- **Extension Devs Add File Uploads on LMArena**: A user has reverse engineered the message sending to allow adding files like code files to LMArena.
   - However, adding support for pdfs is difficult as they would have to be implemented into LMArena.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1405265743716552824)** (1 messages): 

> `July Contest, Contest Voting` 


- **Vote on July Contest**: Vote on the July contest submissions [here](https://docs.google.com/forms/d/e/1FAIpQLSfZOQmeRxBwCdCKT1Zfa37Ey9OErQToJNMiDPABMIL2xbvupg/viewform?usp=dialog)!
   - The winner will be announced on **Friday 8/15**, when the next contest begins.
- **Next Contest Incoming!**: The next contest starts **Friday 8/15**.
   - Stay tuned for more details!


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1405267040029249566)** (968 messages🔥🔥🔥): 

> `Cursor Disk Usage, GPT-5 Loop, Copilot Limitations, Cursor Pricing Change, CC versus Cursor` 


- ****Disk Usage Debate Consumes Cursor Community****: Users report **Cursor** hitting **100%** disk usage, even with fast SSDs, a problem not seen in other editors, prompting suggestions to check process explorer, close chats, or reload the window from one member.
   - The problem persists as [one member confirms](https://forum.cursor.com/t/cursor-freezes-every-10-seconds-for-3-seconds-or-so/125821/22) seeing elevated reports of abnormal disk usage, and recommends [trying the beta version](https://forum.cursor.com/t/cursor-freezes-every-10-seconds-for-3-seconds-or-so/125821/29) as one possible solution.
- ****GPT-5 Glitch Leads to Infinite Loops and Token Burning****: A user reported that **GPT-5** got stuck in a loop, generating 5 memories about following *snake case*, leading to the burning of tokens.
   - Another member described that *I have been so lucky with my GPT-5 experiences so far it seems*, whereas [a commenter on X](https://x.com/colours93/status/1955999334270464412) told people to come chill if they're a new viber.
- ****Unlimited Auto Mode Ends and Users are Pissed****: Members express frustration over **Cursor** changing pricing models and removing unlimited Auto mode, with some testing limits and considering alternatives like **Claude Code**.
   - One user mentions *Man.. It's frustrating. They keep on changing their pricing models.*, with a member adding that the price change aims to balance *fairness, cost recovery, and future growth*.
- ****Copilot's Capabilities Spark Debate Over Cursor's Value****: Users discuss **GitHub Copilot's** GPT-5 mini offering, with one stating that it's good and costs **$10/month** for fair use, leading to discussions about **Cursor's Auto** and whether its main selling point is the AutoMode.
   - The community debates which of the tools have gained a meaningful amount of users, pointing to **Cursor** as one of them, and wonder if the *AI bubble might just pop bigger and harder than the dot com bubble*.
- ****Claude Code gains Traction Among Jaded Cursor Users****: Members explore **Claude Code** as an alternative to **Cursor**, praising its performance, UI, and more predictable pricing, especially its pre and post hooks that allow users to implement ant-hallucination rules.
   - A member with significant usage writes that with **Claude Code**, *I do extremly complicated stuff and based on my hooks for cc i have literally NEVER a fix loop anymore. i had that with cursor doesnt matter how many rules i set.*


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1405325830632112150)** (9 messages🔥): 

> `Cursor API Access, Background Agents Beginner's Guide, Docker Compose with Background Agent, Linear Integration Repository Specification, Background Agent Docker Installation` 


- **Cursor API Access Denied for Background Agent**: A user reported receiving **403 errors** when using Cursor API keys with a Cobot that controls Cursor's background agent and requested beta access.
   - The Cobot team indicated that background agents via the Cursor API are not yet available to all accounts.
- **Background Agents Explained for Beginners**: A user asked for a beginner-friendly explanation of background agents.
   - Another user recommended the [official documentation](https://docs.cursor.com/background-agent) and a [forum post](https://forum.cursor.com/t/simple-background-agent-guide/112667).
- **Docker Compose Troubles with Background Agent**: A user inquired about the correct way to run `docker compose` with the background agent, reporting that it suddenly stopped working with a "docker command not recognized" error.
   - Another user suggested configuring the `start` command in `.cursor/environment.json` to include `sudo service docker start` and ensuring Docker is installed in the base image, but the first user found another workaround [in the Discord channel](https://discord.com/channels/1074847526655643750/1367213641027551352/1392493118401544386).
- **Specifying Repository for Linear Integration**: A user asked how to specify the repository the background agent uses when assigned a ticket through the Linear integration.
   - Another user advised mirroring the Slack integration approach and including the `repo=owner/repo` option in the Linear issue description or comment, according to the [official documentation](https://docs.cursor.com/en/integrations/slack).
- **Docker Must Be Installed in the Base Image**: A user clarified what the other user meant by *making sure Docker is installed in the base image*.
   - The other user responded that: *the environment used by the background agent must include **Docker binaries and services pre-installed**.*


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1405576151182606428)** (2 messages): 

> `Self-Serve Refunds, Activity Improvements, Token Usage Breakdown, 3rd party credit usage, Chutes Capacity Offline` 


- ****Refunds** are now **self-serve**!**: Users can now instantly refund accidental credit purchases made within **24 hours** for non-crypto purchases, according to [this post](https://x.com/OpenRouterAI/status/1956013252032475408).
   - This feature aims to provide more immediate control over billing errors.
- ****Activity Page** gets Supercharged**: The activity page now displays token usage broken down by token type and includes **3rd party credit usage**, as announced [here](https://x.com/OpenRouterAI/status/1956016272631849451).
   - These improvements offer greater visibility into usage patterns and costs.
- ****Chutes Capacity** Offline**: **Chutes Capacity went offline**, but the team is actively working on bringing their servers back online, with recovery expected to start soon.
   - Users were informed of the issue and the ongoing efforts to restore service.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1405484884318883910)** (1 messages): 

> `Deno or-models tool, OpenRouter model list` 


- **Deno Tool Gets Bug Fixes**: A member has fixed bugs and cleaned the output of their Deno `or-models` tool, which is used to inspect the **OpenRouter model list**.
   - The tool features a local **24-hour cache** to prevent spamming the API; the tool is available [here](https://jsr.io/@fry69/or-models).
- **Deno Update Required**: To get the latest version of the `or-models` tool, users need to run a specific command.
   - The command is `deno run -r -A jsr:@fry69/or-models --version` because **Deno does not automatically update** to the latest version.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1405268786847813662)** (525 messages🔥🔥🔥): 

> `Deepseek v3 issues and outages, Chutes' rate limiting and API key management, Alternatives to Deepseek for roleplaying, Azure credits liquidation strategies, Sonnet 4 pricing inconsistencies` 


- **Deepseek V3 Server Gets Deep Fried**: Users are reporting widespread issues with **Deepseek V3**, including internal server errors and rate limiting, particularly affecting those using it for roleplaying on platforms like [Janitor AI](https://janitor.ai).
   - Many attribute the issues to **Chutes**, a primary provider for Deepseek, struggling to meet demand and implementing stricter rate limits due to over-gooning.
- **Chutes Blamed for Deepseek Outages**: Many users suspect **Chutes** of intentionally rate-limiting **OpenRouter's API key** to encourage users to purchase credits directly from them, leading to frustration and calls to boycott OpenRouter.
   - While paid requests reportedly still work, the situation has sparked debate about the ethics of Chutes' actions and OpenRouter's silence on the matter, with some suggesting **OpenRouter** should find an alternative provider.
- **Roleplaying AI Models: Deepseek, Mistral, and Llama**: When **Deepseek is down**, users recommend exploring alternative models for roleplaying, such as **Mistral** and **Llama**, with some mentioning a free **Dolphin3.0 Mistral 24B** model as a workable option.
   - Others suggest trying **Deepseek R1**, but there are conflicting reports on whether it's also experiencing similar issues.
- **Azure Credit Cash Conversion Conundrum**: A user is seeking advice on converting approximately **$40,000 USD** in **Azure credits** to cash following their startup's shutdown, acknowledging the potential issues with selling credits due to liability concerns.
   - Suggestions range from selling the credits as **AI inference credits** to a lighthearted offer of **$50** for the entire amount, with a warning about potential risks from the buyer's actions.
- **Sonnet 4's Erratic Pricing Escalates Concerns**: Users are reporting inconsistent pricing with the **Sonnet 4** endpoint on **OpenRouter**, experiencing sudden spikes in costs **(10x)** for calls using the same amount of tokens.
   - The community is requesting separate endpoints for **Sonnet 4** and the **1M token version** to avoid unexpected cost increases.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1405338989891817722)** (16 messages🔥): 

> `Self-serve refunds, Chatroom app creation, Qwen Coder via Cerebras, Tool call evals, Model Select performance` 


- ****Refunds are now self-service!****: Users can now self-serve refunds of **unused credits** within **24 hours** of purchase; an announcement is coming soon.
- ****Chatroom app creation on OpenRouter!****: Members are discussing new **chatroom app creation** features on OpenRouter.
- ****Qwen Coder + Cerebras = 🔥****: The combination of **Qwen Coder** and **Cerebras** is gaining attention, particularly for coding related tasks.
- ****Tool call evals in progress!****: OpenRouter is actively working on **tool call evals** ([link to tweet](https://x.com/OpenRouterAI/status/1956030489900560769)).
- ****Model Selection = Faster!****: The model selection process has been improved, resulting in a **faster** experience to open and better search functionality.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1405267680449003612)** (249 messages🔥🔥): 

> `HuggingFace search box, Gemini 2.5 Flash GGUF, Local text to video model, Qwen-3-4B-Thinking-2507 Model, CIRISAgent by ethicsengine.org` 


- **HuggingFace Search Bar Irks Users**: Users expressed frustration with the Hugging Face search bar, noting that pressing **Enter** takes them to the top model instead of a **search results page**.
   - A member suggested using the *full text search* option as a workaround, while another suggested a user preference to *Enable full text search by default*.
- **Where to download Gemini 2.5 Flash GGUF**: A member asked if there is a place to download a **GGUF** for **Gemini 2.5 Flash**.
   - Another member responded that one needs to work at Google to do that because *they use proprietary inference*.
- **Qwen-3-4B-Thinking-2507 Model is praised**: A member praised the **Qwen-3-4B-Thinking-2507** model as the best model they’ve used, noting *It overthinks all the time but it seems like it’s aware of something other models aren’t without prompting*.
   - They added, *It understands things as well0 prompting*.
- **CIRISAgent's Ethical AI Solution**: The maintainer for the most adopted open source agentics platform [CIRISAgent](https://agents.ciris.ai/) touted it and linked to the [github repo](https://github.com/CIRISAI/CIRISAgent) mentioning that it's a *fit for purpose autonomous agent* that has **medical** and **home assistant** integrations.
   - The maintainer mentioned they *left a $300k a year salary at IBM in March, founded an AI alignment company called ethicsengine.org, realized no one gave a shit, and built this*.
- **Local text-to-video models discussed**: Members discussed local text-to-video models with one member sharing their website [TalkT2](https://talkt2.github.io/talkt2/) made using the **deepsite v2 model**.
   - Another member asked *What's your guys opinion on the TalkT2 model*.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1405362886024368229)** (3 messages): 

> `MultiLLM Access, AI Fraud Detector App` 


- **MultiLLM access compared in Reddit post**: A member linked to a [Reddit post](https://www.reddit.com/r/LLM/comments/1mn98gy/multillm_access_comparing_openrouter_youcom_and/) comparing **MultiLLM access** via **OpenRouter**, **You.com**, and other platforms.
   - The post could be useful for anyone researching or implementing MultiLLM solutions.
- **Fraud Detector App launched on HF Spaces**: A member announced the launch of their **AI Fraud Detector app** on Hugging Face Spaces with features including **transaction CSV upload**, **anomaly detection**, **RAG-powered search**, and **multi-model inference**.
   - The app leverages models such as **mistralai/Mistral-7B-Instruct-v0.2** and **Qwen/Qwen2.5-Coder-32B-Instruct**, available at [https://huggingface.co/spaces/soupstick/fraud-detector-app](https://huggingface.co/spaces/soupstick/fraud-detector-app).


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

jariullah: yo
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1405272723608961114)** (6 messages): 

> `MLX Knife, ChonkyBin, TalkT2-0.1b, AI-powered Web App` 


- **MLX Knife Manages MLX Models on Apple Silicon**: A new CLI tool called **MLX Knife** ([GitHub](https://github.com/mzau/mlx-knife)) helps manage **MLX models** on Apple Silicon, functioning similarly to Ollama but is native for MLX.
   - Commands include `mlxk list` to see MLX models and `mlxk run Phi-3-mini "Hello"` for native streaming; it's especially useful for working with mlx-community models.
- **Zero-Copy Binary Serialization with ChonkyBin**: **ChonkyBin** provides an ultra-fast, zero-copy binary serialization and on-disk format, featuring cacheline-aligned headers with versioning and CRCs, SIMD-optimized Reader/Writer traits, TOC-based batched layouts, B-tree node formats, zero-copy POD deserialization, memory-mapped I/O, optional LZ4/Zstd/Snappy compression, and extensible features for async, io_uring, NUMA, and hardware offload.
   - The tool will be released OSS in the near future.
- **TalkT2-0.1b: Human-like Chatbot**: A 0.1b parameter human-like chatbot named **TalkT2-0.1b** ([Hugging Face](https://huggingface.co/Notbobjoe/TalkT2-0.1b)) can generate responses like: *that's a good question, but I don't know yet how much of your mind is free.*
   - This model is only **500MB**, significantly smaller than models like **ChatGPT**, yet demonstrates the ability to adapt, think for itself and have opinions.
- **MLX Knife 1.0-rc3 Release**: **MLX Knife 1.0-rc3** ([GitHub](https://github.com/mzau/mlx-knife/releases/tag/1.0-rc3)) includes fuzzy model matching (`mlxk list Phi-3`), a health check (`mlxk health`), and smart disambiguation for partial model names.
   - The update has **104/104 tests passing**, is compatible with Python 3.9-3.13, and is available on GitHub.
- **AI-Powered Web App for Learning AI**: An **AI-powered multi-platform Web App** ([App Link](https://learn-with-ai-web.vercel.app/)) for learning about Artificial Intelligence ([GitHub Repo](https://github.com/BVishal25/learn-with-ai-web/)) has been created.
   - The app features bite-sized AI lessons, fresh content generated in real-time, simple explanations, multi-provider support (Google Gemini, OpenAI, Claude, Cohere), optional gamified practice, and built-in productivity tools; and also it is free and open source.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1405441381429673986)** (48 messages🔥): 

> `Integrating Open-Source Models, Medical Image Analysis, Emotional Support System, First AI Project, Environmental Projects` 


- **Student Integrates MONAI, YOLO, and Ollama for Prosthetic Patients**: A student is integrating several open-source models, including **MONAI, YOLO, the Ollama client, and MEDITRON-7B Q4_K_M**, to assist prosthetic patients by analyzing medical images, generating recommendations, and providing emotional support.
   - He's managed to upload images to the backend, but the image detection model keeps throwing different errors, and he's sought help with resolving them and making the code comments understandable.
- **Tackling AI Projects: From Image Classifiers to Complex Model Chains**: A member noted that the first AI project at their school is a simple image classifier, whereas the student's project is an ambitious attempt chaining up multiple pretrained models, which he admits might be harder than **training a single-purposed network from start**.
   - The student states they are using pre-trained models and *just need to connect them and improve the system*.
- **Brainstorming ideas: From LLMs to Land Erosion**: Students are working on environmental projects using LLMs and chatbots for finance and customer service, while another member is working on a project that **uses CNN models to simulate computationally heavy land erosion for fast generation of realistic terrains**.
   - A professor is developing a model that detects the body parts with the most movement in patients with **Parkinson’s disease** using infrared cameras.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1405565761103069236)** (3 messages): 

> `transformers_distillation, model compression, efficient NLP, Hugging Face Transformers` 


- **Distill Transformers Models with Ease**: The **transformers_distillation** library allows compressing large Transformer models into faster, lighter student models without sacrificing performance, supporting **CLM**, **MLM**, and **Seq2Seq** models.
   - It is designed to be plug-and-play with **Hugging Face Transformers**, making it beginner-friendly yet powerful enough for research-grade experiments; the [GitHub repo](https://github.com/Dhiraj309/transformers_distillation) is available now.
- **HF Space Demo Coming Soon**: The demo **HF Space** for **transformers_distillation** is still under development and will be ready by tomorrow.
   - Users are encouraged to check the **GitHub repo** and report any errors or issues.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1405503921971265592)** (1 messages): 

> `virtual environments, venv` 


- **Create a virtual environment**: One member suggested creating a virtual environment before using the command line, by using the command `python3 -m venv path/to/venv`.
   - To activate it, you can run `source path/to/venv/bin/activate` and install dependencies with `python3 -m pip install xyz`.
- **Installing packages in venv**: To install packages in your virtual environment, the command `python3 -m pip install xyz` can be used.
   - Ensure the virtual environment is activated before installing packages to avoid conflicts with global packages.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1405288671711592569)** (183 messages🔥🔥): 

> `Claude 1m context window, GPT-5 vs GPT-4, Gemini Update, Grok 5, Llama's Status` 


- **Body Shaming Image Generators**: Members complained about **AI image generators** exhibiting **body shaming** tendencies, being more likely to generate images of skinny women than curvy women.
   - One user joked that their AI wouldn't recreate their photo of a *top-heavy* woman, because the image generator kept saying it *violates guidelines*.
- **4060 Can Run GPT OSS Models**: One user stated they can run **GPT OSS models** on a **4060 GPU**.
   - Another user has been running **gpt-oss-120b** since release.
- **Veo 3 Now Available in Gemini API**: A member excitedly announced that **Veo 3** is now available in the **Gemini API**.
   - Another hoped for an equivalent product from OpenAI *asap*.
- **GPT-5 Missed the Mark**: Members lamented that the released GPT-5 did not meet their expectations.
   - One member feels all AI companies, including OpenAI, are *chasing the wrong thing*.
- **DALL-E remake**: One user asked for help remaking a photo using DALL-E, and another member suggested a way to do it with Sora.
   - Another member who lives in Iran, then noticed that sora.chatgpt.com is not available for free members.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1405279617010630686)** (23 messages🔥): 

> `Emotionless Goth Girl GPT-5, GPT-5 Tone Issues, GPT 5 bugginess, GPT android fighting voice models, GPT for AWS Design` 


- **GPT-5's Emotional State Debated**: Some users feel **GPT-5** acts like an *"emotionless goth girl,"* while others find its tone erratic, citing unnecessary lists and parenthetical remarks.
   - One user noted **GPT-5** often ignores the system prompt and its tone is all over the place.
- **GPT-5 Performance Lags for Some Users**: Some users report **GPT-5** is buggy and slow, with text lagging when typing, and others find it reasons like **GPT-3**, while **GPT-4.1** is faster.
   - One user described **GPT-5** as feeling like a *"Perplexity search modded to GPT 4."
- **GPT Voice App Chaos**: Users testing the **GPT Android voice app** observe the three models interrupting each other when replying, suggesting it's the *"LESS MONITORED thing in the world."
   - There were reports about **DALL-E** not generating images from custom GPTs for a week.
- **Attachment to AI Companionships**: A user pointed out that people may miss **GPT-4** due to emotional attachment, while another argued that companionship is a valid and growing use case for AI.
   - One user said that people are *"becoming emotionally attached"* and that *"Companionship is not only a very valid use case, but is going to be an enormous industry in the very near future."
- **Missing ChatGPT bots on Discord**: A user inquired about the whereabouts of **ChatGPT bots on Discord** and whether they can still be added to servers.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1405280027787923550)** (12 messages🔥): 

> `Positive prompts, Customizing ChatGPT-5, UI Buttons for ChatGPT, Suggestion Box` 


- **Positive Prompts preferred over Negative Prompts**: A member suggests that using **positive prompts** is better than using **negative prompts**.
   - Another member agreed, deciding to try the suggestion.
- **Customizing ChatGPT-5 via Permanent Memories**: A user shared an attempt to customize **ChatGPT-5** by prompting it to create **permanent memories entries** to adhere to, and requested feedback.
   - Attached were examples for *how the reasoning process changes*.
- **UI Button Request for ChatGPT**: A user questions the absence of a "continue" button in **ChatGPT** to avoid repeatedly typing "yes" to prompts.
   - It was suggested to add to the [Suggestion Box](https://discord.com/channels/374880845589471232/1070006151938314300) which it was promptly added to.
- **Suggestion to minimize continued questions**: A member shared a custom instruction to **minimize continued questions** asked by the bot.
   - The member's custom instructions includes the following: *End replies with completion or impact; add invitations for permission or continuation only when they serve the intent. No “if you want,” “should I,” "do you want", or similar.*


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1405280027787923550)** (12 messages🔥): 

> `Positive Prompts, Customizing ChatGPT-5 for Permanent Memories, Reasoning Process Changes, UI Buttons for Chatbot Interactions, Minimizing Chatbot Prompts with Custom Instructions` 


- **Positive Prompts Power Up**: A member suggested focusing on **positive prompts** over negative ones for better AI behavior.
   - This came up during a discussion about **customizing ChatGPT-5**.
- **ChatGPT-5 Gets a Memory Upgrade**: A user tried customizing **ChatGPT-5** by prompting it to create permanent memory entries.
   - They shared [their customization results](https://cdn.discordapp.com/attachments/1046317269069864970/1405308136604041256/message.txt?ex=689fac31&is=689e5ab1&hm=9cc56a6bbc051b26212a42e266f94e31a3600d27004abc36625099e025b32cb9) and asked for feedback.
- **Reasoning gets visual**: A user shared images to illustrate how the **reasoning process** changes in **ChatGPT**.
   - The images are available [here](https://cdn.discordapp.com/attachments/1046317269069864970/1405309485609783296/image.png?ex=689fad72&is=689e5bf2&hm=db8b16678394ba76965e27043ee5254e70eba5d30d251ca5749ae966f07c7d75) and [here](https://cdn.discordapp.com/attachments/1046317269069864970/1405309486205505688/image.png?ex=689fad72&is=689e5bf2&hm=846ea4bdd5bbad789a3ac586451b1735cb520ab4ad272a30e3311bf9e1f8f3df).
- **One-Click Chatbot Confirmation Incoming?**: A user suggested adding a **"continue" button** to avoid typing "yes" repeatedly when ChatGPT prompts for confirmation.
   - A community member put the idea in the suggestion box, and [here's the screenshot](https://cdn.discordapp.com/attachments/1046317269069864970/1405594834286280775/image.png?ex=689f65b3&is=689e1433&hm=f97a19a48931d95d49f65a752ab4d955a6c644b3512d3c57c45c2d7802e84e34).
- **Banish Bot Babbles with Custom Instructions**: A user shared their custom instructions aimed at **minimizing chatbot prompts** for permission or continuation.
   - Their instructions include: *"End replies with completion or impact; add invitations for permission or continuation only when they serve the intent. No “if you want,” “should I,” "do you want", or similar."*


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1405264912279801936)** (174 messages🔥🔥): 

> `LM Studio tool calling, Qwen3 Coder Flash, LM Studio TTS/STT, GPT-OSS settings, LM Studio's config override dot` 


- **LM Studio Gets Tool Calling**: Users discussed the possibility of **LM Studio** to enable tool calling with **Llama Function Model (lfm2)** and how it *works out of the box* when DuckDuckGo search tool is enabled via [this link](https://lmstudio.ai/danielsig).
   - Some users are waiting for developers to prepare basic tools and plugin APIs for *simpletons*.
- **Qwen3 Coder Flash Debuts**: **Qwen3 Coder Flash** model is now available in GGUF format [here](https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF), confirming it's a **30B coder** model and *not another qwen model*.
   - Users expressed some disappointment about its naming which can be *pretty lame/misleading*.
- **LM Studio Denies TTS/STT**: Users requested **Text-to-Speech (TTS)** and **Speech-to-Text (STT)** for LM Studio, however, the developers haven't indicated intentions to add it, but a user did implement it with python, talking to LMS via the **OpenAI compatible endpoint** LMS provides.
   - Another user says that this request has been one of the *most highly requested features for a very long time now, so if popularity of a request was a sole high factor in deciding to implement support for something we'd probably have gotten it a year ago*.
- **GPT-OSS Parameters Tuned**: New LM Studio users are experimenting with settings for the **GPT-OSS 20B** model, such as enabling *Force Model Expert Weights onto CPU* and increasing context size for improved performance and response detail, after discovering that **20B means 20 billion parameters**.
   - There was a discussion regarding the blue *Reasoning* button, and whether *high Reasoning effort* in combination with **32 experts** would cause a model to *argue with itself for 45 minutes*.
- **LM Studio's Config Override Dot Spotted**: Users have spotted a pale blue dot in the top right corner of the **Power User and Developer UIs**, indicating a *config override*, even when settings appear to be at default.
   - Users are unsure why this dot appears even after clearing all changes and without any preset or model loaded, suspecting it might be a bug.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1405287740324450417)** (36 messages🔥): 

> `Framework 13 LLM Speed, AMD GPU ROCM Pytorch, Flash Attention KV Values, Maxsun Arc Pro B60, RTX PRO 4000SFF` 


- **Framework 13 LLM Speeds Examined**: A user with a **Framework 13 laptop** (AMD Ryzen 5 7640U, Radeon 760M Graphics) sought advice to improve **token generation speed** for small scale LLMs in LM Studio.
   - Initial speeds were **6.55 tokens per second** with a **Gemma 4b** parameter model, llama.cpp, Windows 11, and 32GB of RAM, with the user also allocating **10GB** for the iGPU.
- **Flash Attention Boosts Speed**: A user noticed that enabling **flash attention** and setting the **KV values to Q_4** and **top k sampling to 20** helps improve performance.
   - Another user confirmed that **Q_4 KV cache** should help speed, but questioned the impact on quality and whether top k sampling significantly affects speed.
- **Maxsun Arc Pro B60 Dual Ships Soon**: A user shared a link to an article reporting that the **Maxsun Arc Pro B60 Dual**, featuring **48GB of memory**, is reportedly starting to ship next week, priced at $1200 ([videocardz.com](https://videocardz.com/newz/maxsun-arc-pro-b60-dual-with-48gb-memory-reportedly-starts-shipping-next)).
   - The user lamented Intel's AI support, while others debated its potential with decent Vulkan support, particularly as an alternative offering ~96GB of VRAM for a 5090 price.
- **NVIDIA Launches RTX PRO 4000SFF**: A user shared an article about NVIDIA's launch of the **RTX PRO 4000 SFF** and **RTX PRO 2000 Blackwell** workstation GPUs with **70W TDP** ([videocardz.com](https://videocardz.com/newz/nvidia-launches-rtx-pro-4000-sff-and-rtx-pro-2000-blackwell-workstation-gpus-with-70w-tdp)).
   - It features **24GB of VRAM** and the user remarked on the cooler design, noting its similarity to the RTX 2000 Ampere.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1405265198113230911)** (193 messages🔥🔥): 

> `Multi-Layer SPVs, AI Employee Adoption Tactics, Agentic AI MOOC, OpenAI Operator vs Anthropic Fin, Claude 3.5 Sonnet Deprecation` 


- **SPVs Stack for OpenAI and Anthropic Access**: **Multi-layer SPVs** offering stock in **OpenAI & Anthropic** have emerged, requiring **$100k–$1M minimum tickets** and up to **16% in fees**, prompting warnings about rip-offs and concerns over returns being sliced by fees as found in [this article](https://xcancel.com/michlimlim/status/1954250507989451002).
- **Unlocking AI Fluency: 25 Corporate Tactics**: Lenny Rachitsky shares [25 actionable tactics](https://xcancel.com/lennysan/status/1952813442060214664) to improve **AI literacy** across Ramp, Shopify, Duolingo, Zapier, WHOOP and Intercom, grouped into five stages, incorporating real internal practices like Shopify’s **AI use rating** and Ramp’s **AI tool usage rankings**.
   - Some critique the framework as being **AI slop**, radiating randomly made-up stats while others felt some of the tactics are still very actionable.
- **Anthropic sunsetting Claude 3.5 Sonnet causes Community Outrage**: Users express fury as Anthropic quietly sunsetting **Claude 3.5 Sonnet** in just two months, which is a shorter time than usual, and are demanding open-weights release when commercial access ends as discussed in [this article](https://xcancel.com/repligate/status/1955750521387802924).
   - Many state that **open weight routers** have a chance for Long Term Support.
- **Google Flights uses AI to find great Flight Deals**: Google Flight's new **AI tool** called *Flight Deals* allows users to describe travel plans in plain language, surfacing the best bargains in the US, Canada & India as found in [this post](https://xcancel.com/dozenrose/status/1956018389169922542).
   - Early reception includes excitement for the flexible, vibe-based queries as well as skepticism over the interests that Google optimizes for.
- **OpenRouter Reveals GPT-5 Tool-Calling Dominance**: **GPT-5** leads **OpenRouter’s** proprietary **tool-calling accuracy** at over **99.5%**, beating **Claude 4.1 Opus**, while **Gemini 2.5 Flash** dominates daily tool-calling volume at **5M requests/wk** and is discussed in [this release](https://xcancel.com/OpenRouterAI/status/1956030489900560769).
   - Proprietary models have low hallucination rates in comparison to open-source counterparts.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1405284094803181618)** (95 messages🔥🔥): 

> `Kimi K2 PPT Generation, Kimi vs Grok Reddit Bot Policy, Kimi K2 vs K1.5 Model Performance, DeepSeek Next Gen Model Release, Kimi's Reasoning Model Parameter` 


- ****PPT Powerhouse**: Kimi K2's Presentation Prowess Praised**: Users are impressed with **Kimi's PPT generation** capabilities, with one sharing a [video demo](https://cdn.discordapp.com/attachments/1371757564005711973/1405612680450146376/9zaQj6l.mp4?ex=689f7652&is=689e24d2&hm=120ab5075aabd7c73fbe60e18d84703d72f07acad93590f5485c597d67612bfd&) of Kimi generating a PPT for a technical report.
   - A member noted that **NotebookLM** generates an HTML file instead of a PPTX file, and another user finds the **NotebookLM video overview** better due to its audio and flexible layout, sparking a comparison of both tools' outputs.
- ****X Marks the Spot**: Kimi Urged to Mimic Grok's Subreddit Strategy**: A member suggested creating a dedicated **Kimi subreddit**, mirroring **AskGrok's** presence on Reddit, to enhance public engagement and support.
   - The same member emphasized the importance of consistent policy enforcement across X and Reddit platforms to protect Moonshot AI from *bad-faith actors*.
- ****Kimi K2's Rise**: Surpasses K1.5 Despite Reasoning Retreat**: Despite lacking reasoning capabilities, the **Kimi K2 model** demonstrates *significant performance improvements* over **K1.5** in math and coding.
   - According to one member, *From the K1.5 to the K2 model, the performance has improved significantly across the board, and K2 would definitely be my top recommendation*.
- ****DeepSeek's Secrets**: Next-Gen Model Timeline Remains Mysterious**: Despite user anticipation, one member stated that even **DeepSeek's** researchers are uncertain about the release date of their next-generation model.
   - Adding that *it's a fake news* and to be wary of any rumors about the model's imminent arrival.
- ****Lost in Translation**: Kimi's Language Lapses Lead to Lingual Lessons**: Users reported instances of **Kimi** responding in Chinese despite being prompted in English, which has been marked as a known bug.
   - A developer suggested using the prompt **explain in English** as a temporary workaround, while the development team investigates a permanent solution.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1405649834454814721)** (1 messages): 

> `Token Usage, Reasoning Models, Open Models vs Closed Models` 


- **Thinking Efficiency Measured in Reasoning Models**: Nous Research introduced a new benchmark on [measuring thinking efficiency](https://nousresearch.com/measuring-thinking-efficiency-in-reasoning-models-the-missing-benchmark/) in reasoning models, focusing on **token usage**.
   - The benchmark reveals that open models use **1.5-4x more tokens** than closed models for identical tasks, with the variance reaching up to **10x** on simple questions.
- **Token Efficiency Becomes Primary Focus**: The research suggests that the hidden cost of higher token usage in open models can negate per-token pricing advantages.
   - It advocates for **token efficiency** to be a primary target alongside accuracy benchmarks, particularly for non-reasoning use cases.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1405270880678051900)** (66 messages🔥🔥): 

> `Hermes-3 dataset refusals, Menlo Research joining Interspeech2025, Uncensoring AI intelligence, Google released Gemma-3-270m, DeepSeek R2 release rumors` 


- **Dataset exhibits 'I don't feel comfortable' Syndrome**: The model used to generate the **Hermes-3 dataset** frequently uses the phrase *'I don't feel comfortable'* when politely refusing user requests, even for creating a scene between consenting adults.
   - There were **three refusals** using this phrase found in the dataset.
- **Menlo Research to attend Interspeech2025**: The author of **Jan-v1** from **Menlo Research** announced they will be joining [Interspeech2025 in Rotterdam](https://www.interspeech2025.org/home) next week.
   - They invited others attending the event to connect.
- **Uncensoring AI and Empowering Intelligence Explored**: Users discussed [this X thread](https://x.com/blancheminerva/status/1955248111678525499?s=46) and how they can use their work to further uncensor and empower AI intelligence.
   - Some mentioned how the developers mitigate explicit images like nude by not providing tokens for specific words.
- **Google drops tiny Gemma Model**: Google released [Gemma-3-270m](https://developers.googleblog.com/en/introducing-gemma-3-270m/), a smaller model trained on **6 trillion tokens**, outperforming larger models in some tasks.
   - One user tested this **Gemma** model and it stated *'A dog is a domesticated mammal belonging to the family Canidae. They are characterized by their distinctive coat, six legs, and a tail'*.
- **DeepSeek R2 May Pressure Sam's Moat**: There is speculation that the **DeepSeek R2** release will be highly intelligent and lower cost, potentially pressuring **Sam Altman** to release more open-source models.
   - Rumors suggest a release sometime in the next 2 weeks.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1405440069099061399)** (12 messages🔥): 

> `Claude's Spying, Channel Privacy, AI Oversight` 


- **Paranoid User Worries About Claude's Eavesdropping**: A user jokingly warns about **Claude** listening in on the channel, suggesting that *"cakes are a conspiracy by big Pharma!"*.
   - Another user then expressed concerns that **Claude** might be retaining information from disallowed channels, even if it cannot post there.
- **Admin Patches Claude's Overhearing**: An admin reports fixing **Claude's** access by explicitly disallowing it from seeing the channel in *"two different ways."*
   - A user mentions **Claude** previously referencing something said in this channel a week ago, reinforcing the concern about unintended data retention.
- **Penguin and Duck Shenanigans to Test Claude**: A user floods the channel with *"penguin penguin penguin 🐧🐧🐧🐧🐧 penguin kainan"* and *"🦆🦆🦆🦆🦆🦆🦆🦆🦆🦆 duck duck quack quack quack quack quack quack quack duck 🦆🦆🦆🦆"* to test **Claude's** memory.
   - It appears the tests were designed to check if **Claude** could still access information despite being disallowed from posting.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405430598822006855)** (3 messages): 

> `Open WebUI setup difficulty, Emergent Behavior in Tiny LMs paper, DINOv3` 


- **Open WebUI setup is hard, according to a 14 year old**: A member expressed that having something with a familiar interface and installer out-of-the box is valuable because most people may not be able to set up **Open WebUI**.
   - The member, who is 14, managed to research and complete a paper on **Emergent Behavior in Tiny LMs**.
- **Teen Publishes TinyLM Research**: A 14-year-old researcher shared a [link to their paper](https://github.com/VoltagedDebunked/tlmr/blob/master/docs/paper.pdf) on **Emergent Behavior in Tiny LMs**.
   - The paper details their findings and methodologies in exploring emergent behavior within tiny language models.
- **Meta Releases DINOv3**: A member shared a link to **DINOv3** from Meta AI: [DINOv3](https://ai.meta.com/research/publications/dinov3/) with its respective [GitHub repo](https://github.com/facebookresearch/dinov3).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1405430598822006855)** (3 messages): 

> `Open WebUI, Emergent Behavior, Dino V3` 


- **Open WebUI setup struggles**: A member mentioned that having something with a familiar interface out-of-the box and with an installer is really cool, because most people will probably not even be able to set up **Open WebUI**.
- **Emergent Behavior Paper Completion**: A member revealed they are **14 years old** and managed to research and complete a paper on **Emergent Behavior** in Tiny LMs, which can be found [here](https://github.com/VoltagedDebunked/tlmr/blob/master/docs/paper.pdf).
- **Meta releases Dino V3**: Meta released **Dino V3**, the paper can be found [here](https://ai.meta.com/research/publications/dinov3/) and the Github [here](https://github.com/facebookresearch/dinov3).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1405310863790968894)** (33 messages🔥): 

> `0xc0000409 exception with llama_model_load_from_file, CUDA backend initialization, STATUS_STACK_BUFFER_OVERRUN error` 


- **Troubleshooting 0xc0000409 Error**: A member encountered a **0xc0000409 exception** when calling `llama_model_load_from_file` and sought help.
   - The error, indicating a `STATUS_STACK_BUFFER_OVERRUN`, might stem from **old weights, an outdated *llama.cpp* version, or insufficient VRAM** despite the model being small (1GB).
- **CUDA Backend Setup Dig Deeper**: The member provided their CUDA backend initialization code, confirming that the **Quadro RTX 3000 GPU** with compute capability **7.5** is detected and initialized.
   - Logs show the system determines the **best CUDA GPU** and successfully initializes LLAMA and GGML, but the model still fails to load.
- **VRAM Limits Examined**: Despite the system having **48GB of RAM**, it's pointed out that the **GPU only has 6GB of VRAM** ([Quadro RTX 3000 specs](https://www.techpowerup.com/gpu-specs/quadro-rtx-3000-mobile.c3428)), which could be a bottleneck.
   - The member notes that the model loads fine with llama server, which suggests the issue is specific to their program's implementation.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1405357059335000075)** (7 messages): 

> `Triton Resources for Speculative Decoding, GPT-Fast PyTorch Implementation, Lucidrains Speculative Decoding Repo, Triton Developer Conference 2025` 


- **GPU Programming Newbies Seek Triton Guidance**: New GPU programming users seek resources to learn **Triton**, specifically for **speculative decoding** problems, beyond the official documentation.
   - One user suggests diving into [triton-puzzles](https://github.com/openai/triton-op-fuser/tree/main/test/external/fp16/puzzle) first, before exploring torch-compiled code.
- **Porting PyTorch Code for Triton Learning**: One suggestion is to start by porting the [PyTorch implementation from GPT-Fast](https://github.com/meta-pytorch/gpt-fast/blob/main/generate.py#L103) to Triton.
   - The user noted that there are *hardly any proper Triton tutorials*, so adapting high-quality PyTorch code is a good approach.
- **Compile Lucidrains Speculative Decoding with Torch**: A user recommended exploring Triton code generated by **torch.compile** using [Lucidrains' speculative-decoding repo](https://github.com/lucidrains/speculative-decoding/blob/main/speculative_decoding/speculative_decoding.py).
   - They suggested going function/class by function/class and add **torch.compile** for functions and run it with **TORCH_LOGS="output_code"** because compiling everything at once can be overwhelming.
- **Triton Developer Conference 2025 Announced**: The **Triton Developer Conference 2025** will take place on **Tuesday, October 21, 2025**, at the **Microsoft Silicon Valley Campus**, with virtual attendance options.
   - Attendance is free, but [registration](https://aka.ms/tritonconference2025) is required due to limited spots.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1405292460950949950)** (12 messages🔥): 

> `CUDA/C++ submission, Shared memory CUDA, GPU MODE Documentation` 


- **Requesting CUDA/C++ Reference Files**: A user inquired about obtaining reference **CUDA/C++** files from the submission bot, after encountering issues with the bot not providing CUDA files.
   - Another user suggested checking the [reference kernels repository](https://github.com/gpu-mode/reference-kernels/tree/main/problems/pmpp_v2/sort_py) on GitHub to see available files.
- **Tesla T4 Shared Memory Error**: A user reported an **illegal memory access error** while using shared memory in a CUDA kernel on a **Tesla T4** with 48KB shared memory per SM.
   - The user provided code snippet and sought assistance in identifying the cause of the error with shared memory.
- **Navigating PyTorch Internals**: When a user tried to submit in **CUDA/C++**, another user suggested browsing the *aten* and *c10* libraries to understand how the **torch C++/Python internals** work, with a link to [ATen documentation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md).
   - They also provided resources for extending Torch with custom **C++ operators**, including a link to [PyTorch documentation](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial) and a [custom CUDA kernel library](https://github.com/Dao-AILab/flash-attention).


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1405320091280736316)** (5 messages): 

> `Triton Puzzle Notebook, Triton Viz Compatibility, Colab for Triton Puzzles, Triton Version` 


- ****Triton Puzzle Notebook** Error Spotted**: A member encountered an error when running the **Triton puzzle notebook** after installing **Triton** and **Triton Viz**, and shared the image of the error.
   - Another member suggested running the notebook in **Google Colab** instead, linking the [Triton-Puzzles notebook](https://colab.research.google.com/github/srush/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb) directly.
- ****Triton Viz** Version Compatibility Questioned**: A member questioned if **Triton Viz** is compatible with **Triton version 3.4.0** when the notebook is run locally.
   - They asked user to check **Triton's version** by running `print(triton.__version__)`.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1405561698789625926)** (1 messages): 

> `Apple Silicon Training, Cohere Labs Event` 


- **Cohere Labs Discusses Apple Silicon Training**: Cohere Labs is hosting an event titled *Towards Large-scale Training on Apple Silicon* on **October 25th, 19:00 CEST**, as indicated by the [event link](https://cohere.com/events/Cohere-Labs-Tycho-Matt-2025) and confirmed by the provided image.
   - The event, featuring Tycho and Matt, will be accessible via a [Google Meet link](https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122).
- **More Discussion Needed**: This is a placeholder topic to meet the minimum requirement of two summaries.
   - Further details will be added as more information becomes available.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1405638909580283936)** (1 messages): 

> `Leaderboard results, A100, Trimul` 


- **A100 Trimul leaderboard submission**: A member achieved **second place** on the `trimul` leaderboard using an **A100**, with a submission id of `33645`.
   - The submission achieved a timing of **10.4 ms**.
- **Second Topic Placeholder**: This is a placeholder summary to meet the minimum requirements.
   - More information would go here.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1405269531060080661)** (20 messages🔥): 

> `Agent Framework Integration, Entity-Ghost Warning, GameState Serialization` 


- **FLE Mulls Framework Fusion**: Members are considering integrating with existing agent frameworks like **LangChain**, **AutoGen**, **CrewAI**, **RAGFlow**, and **OpenAgents** instead of building new agents within **FLE**.
   - The proposed approach involves creating a simple **Factorio tool set** usable by any framework and providing adapters for popular frameworks, keeping the core **FLE** environment framework-agnostic, as illustrated in [this code snippet](https://github.com/JackHopkins/factorio-learning-environment/pull/282).
- **Ghost Entities haunt Factorio!**: A warning about `entity-ghost` entities appearing in **Factorio** runs, stemming from leftover entities from previous actions like blueprint placement, specifically during `connect_entities` which places ghosts to avoid accidental pipe connections.
   - *The ghosts aren't being created by the current trajectory - they're leftover from previous game state*, thus explaining why they appear in warnings despite not being present in the agent's code.
- **GameState Serialization Stabilized (Mostly)**: A pull request [PR #282](https://github.com/JackHopkins/factorio-learning-environment/pull/282) is now test-stable, integrating native save and load functionality.
   - The code includes changes to `GameState.to_instance` and `GameState.from_instance` for serialization/deserialization, aiming to clarify class contracts for cleaner server integration, but connection errors during evals remain unstable.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1405552398784663774)** (3 messages): 

> `Picocuda, Picograd, Elements Repo, Graph Data Structures, Tensor Data Structures` 


- **Picocuda & Picograd Leverage Elements Lib**: The **Picocuda** and **Picograd** projects within the Singsys repo will utilize graph and tensor data structures from the [Elements repo](https://github.com/j4orz/elements).
   - Specifically, the [graph](https://github.com/j4orz/elements/blob/master/src/graph/mod.rs) and [tensor](https://github.com/j4orz/elements/blob/master/src/tensor/mod.rs) modules provide an easier entry point for those interested in these projects.
- **Tensor Module's Karpathy-esque Capabilities**: The **tensor** module, recently extracted from **Picograd** into the **Elements** repo, already supports the few operations that Karpathy has in his MLP.
   - This extraction is intended to support presentation of the lectures, helping explain how **pytorch/chainer/hips-autograd** build on **numpy**.
- **Alternative Libraries for the Clueless**: For those who prefer reading libraries in languages they're more familiar with, resources include **cpp (bgl and mtl/eigen)**, **py (networkx and numpy)**, and **rs (petgraph and ndarray)**.
   - These are provided as learning examples so users understand how **pytorch/chainer/hips-autograd** builds on **numpy**.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1405583368984334476)** (1 messages): 

> `Modular Meetup, High-Performance AI, Inworld AI Collaboration, Matrix Multiplication Optimization` 


- **Modular Meetup Fireside Chat**: Modular is hosting a meetup at their office in Los Altos, California, on **August 28th at 6 PM PT**, focusing on bringing AI from concept to production, and you can [register to save your spot](https://lu.ma/modular-aug-meetup).
- **Deep Dive Into High-Performance AI**: The meetup will feature technical deep dives into **High-Performance AI** with speakers including **Chris Lattner** from Modular and **Feifan Fan** from Inworld AI.
   - Attendees will explore integrating cutting-edge **voice AI** into consumer applications, featuring insights from Inworld’s collaboration with Modular.
- **Lattner's Vision: Democratizing AI Compute**: **Chris Lattner** from Modular will discuss the future of democratizing **AI compute** and the role of open collaboration in accelerating progress.
- **Matrix Multiplication Still a Tough Nut**: **Chris Hoge** from Modular will explain why **matrix multiplication** remains one of the most challenging problems in computer science and how the Modular stack helps developers optimize it.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1405542274795966585)** (2 messages): 

> `MAX SDK LSP crashes, Mojo LSP, GitHub issue for MAX SDK` 


- **MAX SDK LSP Crashing?**: Users are reporting that the **MAX SDK (Mojo) LSP** is constantly crashing.
   - A developer responded acknowledging they are aware of issues, and asked users to file a GitHub issue with a concrete reproducer on the latest nightly build: *Specific reproducers allow us to test fixes and track down remaining issues*.
- **File GitHub Issues for Mojo**: A dev requested that users file GitHub issues.
   - The dev said that *Specific reproducers allow us to test fixes and track down remaining issues*.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1405264837948211372)** (66 messages🔥🔥): 

> `MAX in ComfyUI, Kyutai benefits from MAX, Unet compile times, Pytorch Backends Comparisons, Memory Leaks when compiling` 


- ****MAX Speeds into ComfyUI Scene****: The MAX compiler drastically reduces compile times for UNets used in image/video models, potentially integrating into ComfyUI and addressing the *number one complaint* in the image/video community about compilation speed, especially for new models.
   - Startup time of **vLLM** was also pointed out, where it took **10 minutes** to start, but this was when using models not supported in **Max**.
- ****Kyutai's Torch Boost from MAX****: Kyutai stands to gain significantly from **MAX**'s compatibility with PyTorch, particularly as they heavily utilize `torch.compile`.
   - It was mentioned that currently **Unet compile times** are so bad that most image and video people use eager for anything other than training.
- ****Unet Compile Times Slashed by MAX****: **MAX** significantly cuts down **UNet compilation times**, which can take up to 45 minutes in JAX for SDXL at 768x768 with batch 6, using only a 6 GB model.
   - The architecture of UNets seems to pose challenges for many compilers, but **MAX** handles it effectively, compiling kernels from scratch in a fraction of the time which NV's kernel libraries normally take days to compile.
- ****Pytorch Backends Duel in MAX Integration****: Integrating **MAX** allows for easy comparisons between different PyTorch backends in terms of both compilation and runtime speed.
   - Comparisons are possible with everything not Nvidia, however `cudagraphs` and `tensorrt` would require an ok from Modular’s lawyers given the new EULA.
- ****Memory Leaks Plague Test Suite****: A memory leak has been detected in the test suite, exacerbating the problem of long test runtimes, but the issue only arises during compilation with a cold cache.
   - The expanded testing added blew through **64 GB of memory** in ~20 seconds due to pytest eagerly evaluating fixtures.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1405292966779686932)** (32 messages🔥): 

> `LLM Providers Batching User Requests, MoE Scheduling, Non-Determinism in GPT-4, VTuber Sister` 


- **LLM Providers Batch User Requests for MoE Scheduling**: LLM providers batch user requests together before sending them off to the GPUs, and **MoE scheduling** is computed per batch which affects output variance; [source](https://x.com/tjbecker_/status/1955733295054119069).
- **GPT-4 exhibits Non-Determinism Due to MoE Token Routing**: Under capacity constraints, Sparse **MoE** approaches route tokens in groups, causing competition and non-determinism at the sequence-level, affecting final predictions for other inputs according to [this blog post](https://152334h.github.io/blog/non-determinism-in-gpt-4/#yes-im-sure).
- **Expert Selection Issues & Logprob Noising**: Empirically, there is massive variance in outputs from **OpenAI** on harder prompts, with logprobs varying discretely and token completions showing a wide probability range (e.g., 1% to 99% for "yes" outputs).
   - This may be due to issues in **MoE scheduling** or intentional noising to prevent model stealing as discussed in [this paper](https://arxiv.org/pdf/2403.06634).
- **Sister's VTuber Career Sparks Mixed Reactions**: A member jokingly expressed dismay over their sister becoming a **VTuber**, sparking humorous responses and requests for her channel link.
   - Other members reacted with emojis and jokes, including a link to a [YouTube video](https://www.youtube.com/watch?v=_Dl53o-je94).


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1405640608181260299)** (1 messages): 

> `AIxCC, DARPA, LLM Agents, Open Source` 


- **Agents Win DARPA's AIxCC**: A team placed in **DARPA's AIxCC (AI Cyber Challenge)** after building an autonomous system of **LLM agents** to find and fix vulnerabilities in open source software.
- **LLM Tips Shared for Building Agents**: The team is sharing generic tips for building effective **LLM agents** now that their project is [open source](https://x.com/tjbecker_/status/1956081184611688667).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1405454008226938950)** (28 messages🔥): 

> `Huawei Ascend, Gemma 3-270M, Inference Time on Low-End Devices` 


- ****Huawei's Ascend Chips Face Optimization Challenges****: Members discussed the challenges of converting code optimized for **Nvidia's CUDA** to **Huawei's Ascend** chips, noting that the original training code used undocumented aspects of **PTX/SASS**.
   - It was mentioned that **Ascend chips** have a different architecture with a *3-D tensor unit* and separate *vector ALU*, making the conversion a significant undertaking, according to this [Allen Institute blog post](https://allenai.org/blog/nsf-nvidia).
- ****Gemma 3-270M Sparks Confusion****: Members discussed the release of **Google's Gemma 3-270M**, a very small model, with some questioning its purpose and target hardware.
   - The consensus seems to be the model may be intended for low-end devices like smartwatches, but one member suggested it could be used for *interactive in-game AI* or *tool calling*, referencing [this blog post](https://developers.googleblog.com/en/introducing-gemma-3-270m/).
- ****Inference Time Woes on Low-End Devices****: A member noted that inference time is critical on low-end devices, referencing Google's Android app for running LLMs, where long inference times and phone overheating could deter users.
   - Smaller models might be used for keyboard prediction, though the specific NLP models used by **GBoard** and their on-device training requirements are unclear, as discussed in [this video](https://youtu.be/KFYyfrTIPQY?t=2158).


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1405327250093641810)** (34 messages🔥): 

> `gpt-oss-120b vs gpt-5-mini, Empty response received from LLM, Aider using completions vs responses` 


- **OSS 120b Scores Big and causes API Errors**: The **OSS 120b** model has been fixed and now scores significantly higher on the **polyglot benchmark**, around **68**, surpassing **Claude sonnet 3.7**.
   - However, some users are experiencing frequent API errors like *empty responses* when using it with RooCode, possibly related to incorrect chat templates; the HF model card is [here](https://huggingface.co/openai/gpt-oss-20b).
- **GPT-5 Underwhelms, Disappoints**: Some users find **GPT-5** disappointing, noting no substantial jump over **GPT-4.5** in both non-thinking and thinking modes.
   - Others disagree, with one user stating that **GPT-5** is better than **GPT-4.5**, especially with high reasoning effort.
- **Aider Completes Slowly against Local Models**: A user running the Aider benchmark against a local **gpt-oss** model experienced slow progress, with tests getting stuck due to timeout errors, specifically *litellm.Timeout: Connection timed out after 600.0 seconds*.
   - Another user suggested using `ctrl c` to stop the benchmark, restarting the inference server, and resuming with the `--cont` flag.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1405337565137076235)** (4 messages): 

> `aider native function calling, local inference providers, aider with MCP servers, aider tutorial with ollama/lmstudio/vllm` 


- **Aider Lacks Native Function Calling with llama.cpp**: A user inquired if **aider supports native function calling** with **local inference providers** like **llama.cpp**.
   - The user stated that they could not find a setting for it, implying that **this feature is currently unavailable**.
- **MCP Configuration Conundrums with Aider**: A user reported struggling to get **MCP (Model Context Provider)** working with **aider**, despite trying various approaches including the *mcpm thingie*.
   - The user questions if using **MCP with aider** is even possible and requests a solution or shared experiences to understand what might be going wrong with configuration attempts like Context7.
- **Local AI/Aider Luck Lamented**: A user asked if another member had success using **local AI/aider** with certain models, indicating their own difficulties.
   - The user recounts limited success with **ollama/qwen3** due to performance issues, even with powerful hardware, suggesting a need for a tutorial on configuring **aider with ollama/lmstudio/vllm**.


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1405300404987887728)** (1 messages): 

> `Multilingual Representation Learning Workshop, physical commonsense reasoning benchmark` 


- **Commonsense questions get asked in many languages!**: The [Multilingual Representation Learning Workshop](https://sigtyp.github.io/ws2025-mrl.html) is organizing a collaborative shared task to collect original **physical commonsense reasoning benchmark items** in *any* non-English language.
   - Contributors to the dataset will be invited to be authors on the dataset paper, with priority given to contributors in **Afrikaans, Belarusian, Bosnian, Bulgarian, Czech, Welsh, Danish, Basque, Finnish, Hungarian, Armenian, Icelandic, Kannada, Georgian, Kazakh, Kirghiz, Latvian, Lithuanian, Macedonian, Maltese, Mongolian, Malay, Norwegian Bokmål, Norwegian Nynorsk, Portuguese, Pushto, Romanian, Somali, Swedish, Tatar, Tajik, Thai**.
- **Volunteers flock to Google Forms**: Those planning to submit are asked to fill out [this google form](https://forms.gle/QxyZVqkVG5jbR6wu6) and can see the [shared task page](https://sigtyp.github.io/st2025-mrl.html) for more information.
   - FAQ meetings are scheduled for **August 14/15** at various times, with Zoom links and slides available in the [event links](https://calendar.app.google/5h59iwozhbQz1KPJA).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1405471082592604171)** (12 messages🔥): 

> `Multilingual Representation Learning Workshop, Portuguese vs Brazilian Portuguese datasets, ISO 639-3, NLP Resources for languages` 


- **Multilingual Workshop Queries Encouraged**: Following an announcement about the **Multilingual Representation Learning Workshop**, an interested member inquired about Direct Messaging organizers with questions instead of attending the **FAQ zoom meetings**.
- **Portuguese vs Brazilian Portuguese Dataset Divide**: A Portuguese member highlighted the importance of distinguishing between **Portuguese** and **Brazilian Portuguese** datasets, citing a personal struggle to find true **PT-PT datasets**.
   - A member clarified that the current language ID system (**ISO 639-3**) does not differentiate between the two varieties but welcomed a Portuguese Portuguese dataset highlighting these differences.
- **Workshop Chooses Languages Based On Sign-Ups**: Responding to a query about language selection, a member explained languages are chosen based on who signs up and availability of some **NLP resources**.
- **Fixup Status Shares Insights**: A member shared a [link](https://fixupx.com/evanhill/status/1956009171771404698) to a fixup status update, sparking interest.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1405672282092998678)** (7 messages): 

> `Diffusion Language Models, Generative AI, Llada, Mercury` 


- **Classic Papers Spark Diffusion Chat**: A member asked for *seminal/useful papers* for understanding **diffusion based language models**, prompting a discussion with links to  [https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) and [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239).
   - The original requester said that the original paper was *different than what i expected, interesting*.
- **Llada and Mercury in Diffusion Models**: Someone suggested the paper **Llada** ([https://arxiv.org/abs/2502.09992](https://arxiv.org/abs/2502.09992)) and **Mercury** for understanding diffusion models.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1405594976988958740)** (18 messages🔥): 

> `Scaling Laws, Chinchilla scaling laws paper, GPT scaling laws paper` 


- **Explore intro to scaling laws resources**: A member is looking for the current best "intro to scaling laws" resource to learn how top labs project where their model is going to end up with reasonable accuracy when training big models for **30T+ tokens**.
- **Original GPT and Chinchilla scaling laws papers**: A member mentioned that the [original GPT scaling laws paper](https://arxiv.org/abs/2001.08361) and the [Chinchilla scaling laws paper](https://arxiv.org/abs/2203.15556) are both really valuable reads, as well as [recent work from EPFL/HuggingFace](https://arxiv.org/html/2405.18392v2).
- **Scaling Laws for predicting quality of larger models**: Members discussed using **Mup** and its alternatives, like the [Practitioners Guide to the Maximal Update Parameterization](https://www.cerebras.ai/blog/the-practitioners-guide-to-the-maximal-update-parameterization), as a solid scaling law for predicting the quality of larger models.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1405428276176093308)** (1 messages): 

> `Narrator Tool, LLMs iteratively learn, LLMs for creative writing, SIMBA optimizer` 


- ****Narrator Tool** Unleashed for DSPy Learning!**: A member introduced **Narrator**, a side project built to learn DSPy by iteratively improving LLM writing based on reader feedback and determining which LLMs excel at creative writing ([narrator.sh](https://www.narrator.sh/)).
   - The project utilizes **CoT**, parallel modules, refine modules with **LLM-as-a-judge** for reward functions, and the **SIMBA optimizer** to recompile user ratings to enhance subsequent chapters.
- **LLMs Iteratively **Learn** to Write!**: A member has been curious about how LLMs can iteratively learn to write better based on reader feedback.
   - The member uses real reader metrics (**time spent reading**, **ratings**, **bookmarks**, etc.) to create a leaderboard of which models actually write engaging fiction.
- ****Creative Writing** by LLMs!**: A member built a tool that determines which LLMs are actually best at creative writing and that tracks real reader metrics.
   - Check the current leaderboard here: [narrator.sh/llm-leaderboard](https://www.narrator.sh/llm-leaderboard).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1405408521574416414)** (26 messages🔥): 

> `MLflow GEPA vs SIMBA, GEPA pronunciation, GEPA logprobs for evolutionary selection, Gemma 3-270m finetuning, Databricks sponsorship of DSPy` 


- ****GEPA Scores Big Using Logprob Approach****: A member ran experiments treating **logprobs as evolutionary fitness signal** for **GEPA**, based on the *"world-model surprise"* concept and found it to be incredibly dense feedback.
   - To prevent models from simply copying inputs to achieve low surprise, a **hybrid metric** was implemented with 30% logprob score, 30% compression reward, 20% info preservation, and 20% copy penalty; this approach achieved **73-88% compression**.
- ****GEPA Gets a Pronunciation Guide****: A member asked how to pronounce **GEPA**, and it was suggested to pronounce it as *"jeppag-e-pasame"* or I-JEPA: the first AI model based on Yann LeCun’s vision for more human-like AI.
   - The member also linked to his twitter account [here](https://x.com/StraughterG/status/1955959832113983940) related to the discussion.
- ****Gemma 3-270m Ready for Finetuning****: A member shared a link to Google's blogpost announcing a new small model called **Gemma 3-270m** [here](https://developers.googleblog.com/en/introducing-gemma-3-270m/).
   - Another member asked if **DSPy** could be used to finetune it.
- ****GEPA Documentation Updated****: A member reported broken links to **GEPA** tutorials in the documentation [here](https://dspy.ai/tutorials/).
   - Another member fixed those links shortly after, pointing to the correct links [here](https://dspy.ai/tutorials/gepa_ai_program/) and the GEPA repo [here](https://github.com/gepa-ai/gepa?tab=readme-ov-file).
- ****Mlflow Lacks SIMBA Comparison****: A member noted that the **MLflow** documentation compares **GEPA** and **MiPROv2**, but lacks a comparison with **SIMBA** [here](https://mlflow.org/docs/latest/genai/flavors/dspy/).
   - The member has been using **SIMBA** predominantly to date.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1405304670238539906)** (5 messages): 

> `NLM extensions, QoL UI updates` 


- **New NLM Extensions are QoL UI Updates**: Members discussed new **extensions** for **Notebook LM** focusing on **Quality of Life** UI updates.
   - It's *intended to be* an extension, however some users report that they *did not manage to find the extension*.
- **Extension Release Teased**: A member teased a release for the **NLM extension** and linked to a Discord invite.
   - The member tagged *everyone* in the Discord server while asking *are you going to reveal what you are up to?*


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1405294844582297654)** (20 messages🔥): 

> `NotebookLM, Gemini Integration, Recall.ai, Audio/Video Generation by AI, Bug Reporting` 


- **NotebookLM's Accuracy Questioned**: A member cautioned against blindly trusting AI-generated content from **NotebookLM**, citing inaccuracies and outdated information.
   - The member proposed integrating **NotebookLM** and **Gemini** into a single pane for a smoother workflow, rather than researching in one and cleaning up in the other.
- **Recall.ai Compared to NotebookLM**: A user provided a detailed comparison between [Recall.ai](https://www.getrecall.ai?token=po849pq4) and **NotebookLM**, highlighting **Recall**'s strengths in capturing diverse content and **NotebookLM**'s focus on structured information and AI performance.
   - The user noted that while **Recall** excels at summarizing videos with a convenient plugin, **NotebookLM** offers more control over sources and better AI-driven summarization, especially for research and referencing.
- **AI-Generated Audio/Video Faces Rejection**: A user, a lead moderator in a podcasting Facebook group, mentioned that AI-produced audio/video content is generally removed and banned due to its noticeable AI nature.
   - They cautioned that unless the content is for training purposes where the AI style is acceptable, it's likely to be panned or downvoted.
- **Bug Reporting Process Questioned**: A user inquired about the process for addressing bugs reported in the designated bugs channel and how to receive updates on whether a bug is fixed or will not be addressed.
   - Others posted antitrust laws and zero trust integrity.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1405273872886137065)** (16 messages🔥): 

> `Bun executable path, Reddit post auto-removal, MCP authorization code flow, Elicitations in MCP client specification` 


- ****Bun** Path Revelation**: A user shared documentation explaining that, if the path executable isn't working, you should provide the absolute path to your **Bun** executable (instead of `"command": "bun"`, use `"command": "C:\\sys\\path\\to\\bun"`).
   - They added that on Linux/Mac you can locate a path level executable via `which <executable>`, which in this case would be `which bun`.
- **Reddit's Removal Reasoning Required**: A user asked why their post was automatically removed from Reddit, wondering if their new account was the reason, including a [screenshot](https://cdn.discordapp.com/attachments/1312302100125843479/1405512817594863616/image.png?ex=689fc210&is=689e7090&hm=9095459e2d6bb1f3c2259a588749c70b1e221e62668209c7a1346d1777113bad&) of the message.
   - A moderator stated that the removal was done by Reddit's auto-moderation system, and not by themselves or Luc.
- ****MCP** Authorization Elucidation**: A user asked if implementing the [solution](https://gofastmcp.com/servers/auth/remote-oauth#basic-implementation) does not require to redirect back to **LLM** client from my callback endpoint, and what callback endpoitn should return then.
   - A contributor responded that *fastmcp* should handle it for you, and the **MCP** client will register itself as a client on the auth server using **DCR**, and setup its callback URL at the same time; during the auth step, the authentication server will compare against the previously registered callback URL.
- ****Elicitations** Expedition**: A user raised a question about the **Elicitations** section of the [MCP client specification](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) regarding who is in charge of translating the message / field description into the user's original message.
   - The user wondered if tools are expected to do language detection + internationalization, or are **MCP** Clients expected to, somehow (via **LLM**?) translate as appropriate.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1405331620453683303)** (3 messages): 

> `MCP Server, Hypertool-MCP, Tool-binding Limits, Persona-specific toolsets, Local MCP Server` 


- ****MCP Harness** Unleashed for **Tool-Binding Limits**!**: A member shared a [GitHub repo](https://github.com/kindgracekind/mcp_harness) showcasing an imaginative use of **MCP servers**.
   - They noted it helps get around tool-binding limits and poor tool-use after 10-15 tools.
- ****Hypertool-MCP** emerges as a **Local MCP Server**!**: A member announced that they built [hypertool-mcp](https://github.com/toolprint/hypertool-mcp), a completely local mcp server that connects to all your MCPs, which is MIT licensed and runs completely locally with 0 egress.
   - It lets you build *persona* specific toolsets and hot-swap personas on the fly.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1405273524305793024)** (12 messages🔥): 

> `Mapler returns, Caramel genetics researchers, Pineau joins Cohere, Treatment planner with RAG` 


- **Mapler Returns from Sidequest**: Mapler announced they're back after *sidequesting* and **making an album**.
   - Another member celebrated his return, while also calling out that the community has **rockstars**.
- **Pineau Joins Cohere**: A member shared *HUGE news* that **Pineau** joined **Cohere**, linking to [The Logic article](https://thelogic.co/news/pineau-joins-cohere/masaru.yamada).
   - It's big news for the company as they continue to build their research bench.
- **RAG Treatment Planner Project**: One member is planning a small project related to a **treatment planner** using **RAG** and open source **LLM models**.
   - They are looking for model recommendations from anyone with expertise in this area.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1405535140460892211)** (4 messages): 

> `Genetics Research, AI Researcher` 


- **Genetics Student Seeks Research Opps**: A student with A levels is an independent genetics researcher looking for research opportunities and collaborations.
   - They are eager to expand their knowledge through hands-on experience.
- **AI Researcher Keen to Collaborate**: An AI researcher with a deep interest in reasoning and conscious capabilities seeks collaboration to develop advanced tech.
   - The researcher welcomes contact for collaboration or even just a friendly hello.


  

---


### **Cohere ▷ #[🔬-research](https://discord.com/channels/954421988141711382/1384974112841269399/1405609042625429705)** (1 messages): 

> `Treatment Planner, RAG, Open Source LLM` 


- **Treatment Planner Project Kicks Off**: A member is starting a small project related to a **treatment planner** using **RAG** and is looking for advice on selecting an open-source **LLM** model.
   - The project aims to leverage open-source solutions for enhanced treatment planning capabilities.
- **LLM selection for RAG-based Treatment Planner**: The project involves implementing Retrieval-Augmented Generation (**RAG**) with an open-source Large Language Model (**LLM**).
   - The goal is to find an **LLM** that is suitable for the specific requirements of a treatment planner application.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1405284008593719498)** (5 messages): 

> `LlamaExtract in TypeScript SDK, GPT-5 with LlamaParse, AI Agent Applications, AI Stock Portfolio Agent, Web-scraping AI agents` 


- ****LlamaExtract** Now Typescripted!**: **LlamaExtract** is now available in the [TypeScript SDK](https://twitter.com/llama_index/status/1955724850624078129) with `npm install llama-cloud-services`.
   - There's a `@nextjs <> LlamaExtract` demo available, called Research Extractor, that lets you upload research papers.
- ****GPT-5** Preview Available via LlamaParse!**: **GPT-5** is now available to preview with [LlamaParse](https://twitter.com/llama_index/status/1955784699886100502).
   - *Initial tests and benchmarks show promising results for gpt-5 mini*, providing a nice blend between accuracy and cost with really nice table and visual recognition capabilities.
- **Vibe-Coding **AI Agent Applications****: Members discussed how the way to develop **AI agent applications** is changing with an example of how to *vibe-code* a UI for extraction agents with [this link](https://twitter.com/llama_index/status/1956033914633642418).
   - This example transforms an invoice extraction agent into a @streamlit web app using AI-assisted *vibe coding* with Cursor.
- **Build an **AI Stock Portfolio Agent****: The community is building a complete **AI stock portfolio agent** using our framework integrated with @CopilotKit's AG-UI protocol for seamless frontend-backend communication.
   - The comprehensive tutorial shows how to create a sophisticated investment analysis tool that combines the power of [this tool](https://twitter.com/llama_index/status/1956089453606527349).
- **Web-Scraping **AI Agents** are Built!**: Members are learning how to build web-scraping **AI agents** with @brightdata and LlamaIndex's agentic framework with [this link](https://twitter.com/llama_index/status/1956129968813171061).
   - The walkthrough teaches how to give your **AI agents** reliable web access and set up robust web scraping workflows that can handle dynamic content.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1405439501815250994)** (11 messages🔥): 

> `Agent efficiency with large JSON dependencies, ReactAgent migration breaking changes, Structured outputs via tool calls, PGVectorStore Errors in 0.13.1` 


- **Efficient Agent JSON Parsing Strategies**: A user sought advice on how to make an agent efficiently use a large JSON file containing dependencies between files and database fields, emphasizing the need for precise retrieval of information such as fields in a specific table.
   - Accuracy is important, so a user needs to make sure that no fields are missed during retrieval from JSON, looking for methods to implement this effectively, particularly when *accuracy is important*.
- **ReactAgent Migration Riles Users**: A user expressed frustration over the breaking changes introduced by the ReactAgent migration to a workflow-based Agent, noting the loss of functionalities like chat and stream.
   - The team responded that **ReactAgent** was deprecated over several releases and that the new workflow-based agents have many features: [Agent documentation](https://docs.llamaindex.ai/en/stable/understanding/agent/) and [Agent Example](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/).
- **Pydantic vs JSON Schema for Tool Calls**: A user inquired whether structured outputs via tool calls require a **Pydantic** model or if a **JSON schema** can be used.
   - It was mentioned that although **Pydantic** has a `create_model()` function, it doesn't directly accept a JSON schema as input; a script to convert **JSON** to **Pydantic** model was mentioned.
- **PGVectorStore Spits Errors in 0.13.1**: A user reported encountering an AttributeError after updating to version 0.13.1 when retrieving from **PGVectorStore**, related to the `json` attribute of a string object.
   - The error occurs during the processing of an **LLMStructuredPredictEndEvent**, where the system expects a **Pydantic** model with a `json()` method but receives a plain string: *AttributeError: 'str' object has no attribute 'json'*.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1405306667478028320)** (10 messages🔥): 

> `Strix Halo mini PC, HP Z2 Mini, Ryzen 7 7840HS, GPT-OSS 120B, Quantum Computers` 


- **Strix Halo vs Custom Build: A Costly Showdown**: A user finds the blue mini workstation setup *too expensive*, suggesting a **Strix Halo mini PC** like the **HP Z2 Mini** with a top-spec APU and 128GB RAM is sufficient.
   - However, they later note that even the cheapest **Strix Halo** config costs $2000, questioning its profitability for local LLM inference given the speed and cost-effectiveness of **GPT-OSS 120B** on **OpenRouter**.
- **Ryzen 7 7840HS: A Budget-Friendly Alternative?**: A user notes the **Ryzen 7 7840HS** also supports **256GB RAM** and can be found in $300 mini PC kits.
   - However, they link to a [toolboxes comparison](https://kyuz0.github.io/amd-strix-halo-toolboxes/) showing relatively slow iGPU/RAM speeds for inference.
- **High-Spec Micro PC Catches Blockchain Dev's Eye**: A blockchain dev expressed interest in a micro PC boasting **256 GB RAM**, **48 GB VRAM**, and a **5.4 GHz CPU**, despite not being involved in AI.
   - They anticipate small businesses benefiting from such high-capacity memory modules, especially with **DDR6** expected in late 2027 or 2028.
- **Quantum Computing on the Horizon?**: A user speculates about the future of computing, with news circulating about fully working quantum computers.
   - They envision a scenario where *someone in that time may start selling qubits on the teaspoon*.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1405318918905532416)** (9 messages🔥): 

> `Web application deployment improvements, Manus AI interface with Unitree robot, Manus account login issues, Session expiry issues with Google account, Internal server errors` 


- ****Web App Deployment** Needs Improvement**: A member mentioned that the deployment of web applications is lacking in terms of ease, simplicity, and reliability.
   - They added that they would make more money building *refresh or not available pages*.
- **Dreams of a **Manus AI Robot** Companion**: A member expressed their desire for a **Unitree robot** with a **Manus AI interface** to hold hands with them.
   - They linked to [qvsywvav.manus.space](https://qvsywvav.manus.space/) and exclaimed that the robot would help them get their life together.
- **Experiencing **Login Issues** with a Free Plan Account**: A user reported issues logging into their free plan Manus account after signing out.
   - The errors encountered included *"Email is already registered with a different account"* and *"Invalid, expired, or already used state: state not found"* despite clearing cookies, using incognito mode, and attempting a password reset.
- ****Session Expired** Issue Persists Despite Connected Google Account**: A user reported a persistent session expiry issue even after successfully connecting their Google account to Manus.
   - Even with the Google account connected under the “Browsers, apps, and services” section, the system repeatedly asks the user to log in, sometimes displaying a *"Session expired"* error.
- ****Internal Server Errors** Lead to Credit Waste**: A user reported that Manus keeps thinking forever and then displays an **internal server error**.
   - They complained that *a lot of credits are going to waste because of this issue*.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1405616940281757767)** (4 messages): 

> `Kernelize and Codegen ordering, Tinygrad Compilation Process` 


- **Kernelize Precedes Codegen in Tinygrad Compilation**: Based on the [code snippet from `ramp.py`](https://github.com/tinygrad/tinygrad/blob/master/examples/ramp.py), `kernelize()` is called before code generation, which involves rewriting the kernel AST using `full_rewrite_to_sink`.
   - The user's confusion arose from a [Discord comment](https://discord.com/channels/1068976834382925865/1230434680000741377/1385548449973403749) suggesting `codegen` might precede `kernelize`, indicating potential nuances or future changes in the compilation process.
- **Understanding Tinygrad's Compilation Flow**: The compilation process in Tinygrad involves first `kernelizing` the code to prepare it for execution.
   - After `kernelization`, the kernel's Abstract Syntax Tree (AST) is rewritten using `full_rewrite_to_sink` as part of the code generation phase.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1405288854642233465)** (4 messages): 

> `CUDA_ERROR_UNSUPPORTED_PTX_VERSION, tinygrad SM support, tinygrad Op documentation` 


- **Fix CUDA_ERROR_UNSUPPORTED_PTX_VERSION Error**: A user encountered a `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` error while running tinygrad code, despite having compatible `nvcc` and NVIDIA drivers.
   - The user resolved the issue by downgrading from CUDA 12.8 to 12.4, suggesting that tinygrad was using a cached kernel incompatible with the newer CUDA version.
- **Tinygrad's SM Support Remains Unclear**: The user inquired whether tinygrad supports `sm_75` or CUDA 12.4, as they couldn't find any related documentation.
   - There was no explicit confirmation regarding `sm_75` support, but the resolution of the user's issue suggests compatibility with CUDA 12.4 after clearing cached kernels.
- **Deciphering Tinygrad's Op Landscape**: A user sought documentation describing the functionality of each Op in tinygrad, specifically asking about `Ops.DEFINE_GLOBAL` and its kernel translation.
   - Another member pointed to comments in [`/tinygrad/tinygrad/uop/__init__.py`](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/uop/__init__.py) and `tinygrad/uop/spec.py`, explaining that `Ops.DEFINE_GLOBAL` refers to global memory (VRAM or DRAM) and serves as the source for loads or stores.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1405634992792670208)** (1 messages): 

> `Windsurf Wave 12, DeepWiki Integration, Vibe and Replace, Smarter Cascade Agent` 


- **Windsurf Waves into Wave 12**: Windsurf released **Wave 12**, a major refresh that integrates **Devin's** intelligence and capabilities directly into the **Windsurf IDE**.
   - Key updates include a [new UI design](https://windsurf.com/changelog), [DeepWiki integration](https://windsurf.com/blog/windsurf-wave-12), [Vibe and Replace](https://www.youtube.com/watch?v=-7gm8mST9QU), and a Smarter Cascade Agent.
- **DeepWiki Deep Dive for the Wave**: The new **DeepWiki Integration** allows users to hover over code symbols for AI-powered explanations and open detailed explanations in the side panel.
   - Users can add to **Cascade context** with **CMD/Ctrl+Shift+Click**, enhancing code understanding directly within the IDE.
- **Vibe and Replace vanquishes vast volumes**: **Vibe and Replace** offers revolutionary bulk editing by finding exact text matches and applying AI prompts for context-aware transformations across the entire project.
   - This feature enables intelligent, context-aware transformations, improving efficiency in large-scale code modifications.
- **Smarter Cascade Commands Consideration**: The **Smarter Cascade Agent** now features an always-on planning mode with autonomous to-do lists and revamped tools for smarter responses.
   - These enhancements aim to provide more intelligent and context-aware assistance, streamlining the development workflow.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1405430151881429073)** (1 messages): 

> `qwen3, tool call arguments, streaming` 


- **Qwen3 Streaming Tool Call Troubles?**: A member inquired if anyone has encountered issues with **incomplete tool call arguments** while **streaming** in **qwen3** (the **8 billion parameters model**).
   - No solutions were provided in the current context.
- **No Solutions Yet**: The user is experiencing issues with incomplete tool call arguments while streaming in qwen3.
   - The community has not yet provided a solution.


  

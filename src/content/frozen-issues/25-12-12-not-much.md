---
id: MjAyNS0x
title: not much happened today
date: '2025-12-12T05:44:39.731046Z'
description: >-
  **GPT-5.2** shows mixed performance in public evaluations, excelling in
  agentic tasks but at a significantly higher cost (~**$620/run**) compared to
  **Opus 4.5** and **GPT-5.1**. It performs variably on reasoning and coding
  benchmarks, with some improvements on long-context tasks. Extended "reasoning
  effort" settings notably impact results. Aggregators rank **Gemini 3 Pro**
  above GPT-5.2 in task persistence. **OpenAI** released sparse activation
  models sparking debate on sparsity vs MoE architectures. **Allen AI**'s **Olmo
  3.1 (32B)** advances open reinforcement learning scale with substantial
  compute investment (~**125k H100 hours**). **Mistral**'s Devstral-2 and
  **llama.cpp** improve local inference infrastructure with new features like
  GGUF support and distributed speedups. **Tinker** platform goes GA with vision
  input and finetuning support for **Qwen3-VL-235B**.
companies:
  - openai
  - allen_ai
  - mistral-ai
  - ollama
  - lmstudio
  - thinkymachines
models:
  - gpt-5.2
  - opus-4.5
  - gemini-3-pro
  - gpt-5.1
  - olmo-3.1-32b
  - qwen3-vl-235b
topics:
  - reinforcement-learning
  - model-benchmarking
  - long-context
  - model-quantization
  - model-optimization
  - inference-speed
  - sparsity
  - fine-tuning
  - vision
people:
  - sama
  - scaling01
  - akhaliq
  - artificialanlys
  - lechmazur
  - acerfur
  - epochairesearch
---


**a quiet friday.**

> AI News for 12/11/2025-12/12/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 8597 messages) for you. Estimated reading time saved (at 200wpm): 621 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

More [AIE Talks](https://www.youtube.com/channel/UCLKPca3kwwd-B59HNr-_lvA) rolling out all weekend.

---

# AI Twitter Recap

**Frontier model evals: GPT‑5.2 vs Opus 4.5 and Gemini 3, costs, and context settings**

- **GPT‑5.2 performance is mixed across public evals**: On real‑work, agentic tasks, GPT‑5.2 tops GDPval‑AA, overtaking Claude Opus 4.5, but at higher cost—about **$620/run** vs Opus 4.5’s **$608** and GPT‑5.1’s **$88**, driven by >6× more tokens and a 40% price hike, per [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1999404579599823091). On reasoning/coding staples, community runs report: below Opus 4.5 and Gemini 3 Pro on LiveBench, weak on SimpleBench (below Sonnet 3.7), improved on VendingBench‑2 but still behind Gemini 3 Pro/Opus 4.5 ([@scaling01](https://twitter.com/scaling01/status/1999323401421488319), [@scaling01](https://twitter.com/scaling01/status/1999466846563762290), [@scaling01](https://twitter.com/scaling01/status/1999449402776387808)). Long‑context MRCR v2 shows 5.2 xhigh beating Gemini 3 Pro; OpenAI updated MRCRv2 after fixes to v1 ([thread](https://twitter.com/DillonUzar/status/1999328225164431394), [@scaling01](https://twitter.com/scaling01/status/1999327512401527107)). Evals can be sensitive: @eliebakouch spotted metric discrepancies in GPT‑5.1 MRCR numbers across orgs, later attributing it to different benchmarking variants ([note](https://twitter.com/eliebakouch/status/1999534955274117457)).
- **Mind the “reasoning effort” knobs**: Several highlights for GPT‑5.2 rely on xhigh extended‑thinking (e.g., 100k “thinking” tokens), which can shift outcomes materially ([@scaling01](https://twitter.com/scaling01/status/1999535536130662576)). Example: Extended NYT Connections jumps from **77.9 (High)** to **89.3 (xHigh)** ([@LechMazur](https://twitter.com/LechMazur/status/1999582591905583256)). Community sentiment ranges from markedly better proof‑writing ([@AcerFur](https://twitter.com/AcerFur/status/1999314476320063546)) to “not obviously larger than Gemini 3 Pro” ([@scaling01](https://twitter.com/scaling01/status/1999566015873569174)).
- **Aggregators**: Epoch’s ECI places GPT‑5.2 at **152**, second to Gemini 3 Pro; their ECI‑to‑Time‑Horizons projection suggests median task persistence of **3.5h** for 5.2 vs **4.9h** for Gemini 3 Pro and **2.6h** for Opus 4.5 ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1999548496198926728), [follow‑up](https://twitter.com/EpochAIResearch/status/1999585243003781413)). Meanwhile, [@sama](https://twitter.com/sama/status/1999624463013544024) claims GPT‑5.2 exceeded “a trillion tokens in the API” on day one, underscoring rapid adoption.

**Open models, RL scaling, and sparsity**

- **Allen AI’s Olmo 3.1 (32B) pushes open RL scale**: Extended the prior RL job for 3 more weeks to produce Olmo 3.1 Think 32B and Instruct 32B; ~**125k H100 hours** (~$250k), with continued gains on hard evals (AIME, coding). Released intermediate checkpoints, new 7B math/code RL‑Zero baselines, and large filtering/preference datasets ([@allen_ai](https://twitter.com/allen_ai/status/1999528336318509316), [@natolambert](https://twitter.com/natolambert/status/1999528636085649532)). Takeaway: long‑running RL remains under‑explored and continues to improve.
- **Circuit‑sparsity from OpenAI**: A public release of sparse activation patterns/models (huggingface.co/openai/circuit‑sparsity) sparked discussion of the tradeoffs vs classic MoE; some argue large sparse‑activation architectures with shared capacity may be preferable to isolated experts ([@_akhaliq](https://twitter.com/_akhaliq/status/1999528833490239864), [commentary](https://twitter.com/teortaxesTex/status/1999559676866724272)).
- **Local and inference infra**: Mistral’s Devstral‑2 arrives in Ollama and LM Studio; GGUF builds fixed; MLX adds distributed inference speedups on Apple Silicon ([Ollama](https://twitter.com/ollama/status/1999590723373662612), [@lmstudio](https://twitter.com/lmstudio/status/1999648656958296119), [@awnihannun](https://twitter.com/awnihannun/status/1999596403472105975)). llama.cpp gets Ollama‑style model management: auto‑discover GGUFs, per‑model processes, LRU unload ([@victormustar](https://twitter.com/victormustar/status/1999484435910263256)).

**Agent platforms and tooling**

- **Tinker opens up (finetune frontier VL)**: Tinker is now GA with vision input; supports finetuning **Qwen3‑VL‑235B** and adds Kimi K2 Thinking, OpenAI‑compatible inference, and easy sampling. Cookbook examples included ([announcement](https://twitter.com/thinkymachines/status/1999543421631946888), [@dchaplot](https://twitter.com/dchaplot/status/1999543675765031289), [@rown](https://twitter.com/rown/status/1999544121984245872)). Separately, a community fork adds on‑policy distillation for multi‑turn tool use (tokens, logprobs, reward masks) to Tinker’s training loop—aiming beyond single‑turn TRL/Tinker baselines ([details](https://twitter.com/HeMuyu0327/status/1999316923885191376)).
- **Agent coordination guidance and observability**: Google proposes practical principles for when multi‑agent systems help vs hurt and a simple predictive framework that picks the right agent topology **87%** of the time ([summary](https://twitter.com/TheTuringPost/status/1999499042880127328), [paper](https://twitter.com/TheTuringPost/status/1999499191840817202)). LangChain released “Deep Agents” debugging workflows, including trace‑aware assistants (Polly) and a CLI to equip coding agents with debugging capabilities; MCP adapters now support structured content from tools ([roundup](https://twitter.com/LangChainAI/status/1999568074450829482), [MCP update](https://twitter.com/sydneyrunkle/status/1999538200243511725)). Also notable: ChatGPT now surfaces a /home/oai/skills folder for hosted skills ([@simonw](https://twitter.com/simonw/status/1999503124592230780)).
- **Fast coding agents**: Cognition runs coding agents at ~**1k tok/s** on Cerebras with frontier‑level accuracy ([@cerebras](https://twitter.com/cerebras/status/1999540379553611955)); GitHub/VS Code showcased unified local/cloud/background agents ([@code](https://twitter.com/code/status/1999575448087396563)).

**New techniques and papers**

- **Normalization‑free Transformers that win**: “Derf” (Dynamic erf) is a simple point‑wise layer enabling norm‑free Transformers to not only work but outperform normalized baselines ([@liuzhuang1234](https://twitter.com/liuzhuang1234/status/1999321116641497355)).
- **Token‑level credit assignment for reasoning**: HICRA focuses RL optimization on “planning tokens” identified via “Strategic Grams,” improving AIME/AMC/Olympiad vs GRPO (e.g., Qwen3‑4B‑Instruct AIME24 **73.1% vs 68.5%**) and offering a mechanistic view of “aha” phases during RL ([thread](https://twitter.com/omarsar0/status/1999483394963701911)).
- **Olympiad geometry via agents + RL**: InternGeometry solves **44/50** IMO problems with iterative reasoning and “Complexity‑Boosting RL,” surpassing gold medalists on the test set ([summary](https://twitter.com/HuggingPapers/status/1999572332906438987)).
- **Adapt pretrained visual encoders with a single layer**: Apple’s FAE shows “One Layer Is Enough” to adapt visual encoders for image generation ([paper](https://twitter.com/_akhaliq/status/1999516539351883823)).
- **Robot simulation via video models**: Veo‑Robotics evaluates Gemini robotics policies in a video‑world simulator, enabling safety evals and failure‑mode analysis before deployment ([intro](https://twitter.com/SeanKirmani/status/1999528692448657687), [GDM thread](https://twitter.com/Majumdar_Ani/status/1999525259276423569)).

**Product and leaderboard updates**

- **Video and image models**: Runway Gen‑4.5 rolls out across plans, claiming top community ratings for fidelity/control ([@runwayml](https://twitter.com/runwayml/status/1999481621326729530)). Video Arena adds Kling 2.6 Pro (+16 over prior) and Kandinsky 5.0 (top open t2v) ([arena update](https://twitter.com/arena/status/1999530939886768205)). Flux‑2‑Dev enters Arena Top‑10 for text‑to‑image/edit ([image update](https://twitter.com/arena/status/1999560495867793881)).
- **Document and data plumbing**: ByteDance OSS ships MIT‑licensed Dolphin‑v2, a **3B** doc‑understanding model handling scans/photos and 21 content types with pixel‑level coordinates ([@AdinaYakup](https://twitter.com/AdinaYakup/status/1999462500551786692)). DatologyAI releases Luxical, a fast lexical‑dense CPU embedding model and methodology for web‑scale curation pipelines ([intro](https://twitter.com/lukemerrick_/status/1999516702808375791), [blog/code/models](https://twitter.com/lukemerrick_/status/1999516722030870542)).
- **Geospatial and translation**: GeoAI QGIS plugin supports Moondream VLMs, SAM‑3 segmentation, and DIY geospatial training ([@giswqs](https://twitter.com/giswqs/status/1999536028282179721)). Google updates Gemini audio: live speech‑to‑speech in Translate (beta), improved TTS adherence and conversational handling across Flash/Pro/Live ([@GoogleAI](https://twitter.com/GoogleAI/status/1999560839679082507), [devs](https://twitter.com/googleaidevs/status/1999539531826036973)) and adds rich local results from Maps to Gemini ([@GeminiApp](https://twitter.com/GeminiApp/status/1999631529379791121)).

**Benchmarks: expectations vs reality**

- **“Benchmarks decay fast”**: Leaders argue useful benchmark half‑life is “measured in months,” calling for new tasks in dynamic environments, debate/persuasion, and efficient reasoning—beyond AIME/ARC staples ([@gdb](https://twitter.com/gdb/status/1999454952801075353), [@scaling01](https://twitter.com/scaling01/status/1999321464319754290)). MRCR v2 corrections and setup variance highlight the friction of reproducible long‑context evals ([@DillonUzar](https://twitter.com/DillonUzar/status/1999328225164431394), [@eliebakouch](https://twitter.com/eliebakouch/status/1999482968717279441)). As multi‑agent/agentic harnesses (e.g., Stirrup) spread, expect evals to emphasize process‑oriented metrics and cost/latency alongside accuracy ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1999404589049872615)).

**Top tweets (by engagement)**

- [“By using AI for writing, you’re robbing yourself of the authentic writer’s experience of not writing.”](https://twitter.com/NC_Renic/status/1999351657730290042) (~60k) – A wry cultural note on AI‑assisted creation.
- [Su Hui’s 4th‑century “Star Gauge” combinatorial poem](https://twitter.com/RnaudBertrand/status/1999315488598622360) (~41k) – Not AI, but a reminder of deep algorithmic artistry in human literature.
- [“It’s actually a terrible meme... you would make training 61,320× slower.”](https://twitter.com/scaling01/status/1999456392495923555) (~15.6k) – Pushback on a circulating training‑at‑inference trope.
- [“Enterprise AI is going to be a huge theme of 2026.”](https://twitter.com/gdb/status/1999416686446019049) (~1.4k) – Signal from practitioners about near‑term adoption focus.

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. NVIDIA Nemotron Model Leak

- [**Someone from NVIDIA made a big mistake and uploaded the parent folder of their upcoming model on Hugging Face**](https://www.reddit.com/r/LocalLLaMA/comments/1pkpxss/someone_from_nvidia_made_a_big_mistake_and/) (Activity: 1196): **The image reveals a Hugging Face repository page where NVIDIA seemingly uploaded folders related to their upcoming Nemotron models, including "NVIDIA-Nemotron-Nano-3-30B-A3B-BF16" and "Nemotron-H-4B-Instruct-128K." This suggests a potential accidental exposure of sensitive or unreleased model data, as the uploads were made using the "huggingface_hub" tool just minutes before the screenshot was taken. The Nemotron models appear to be part of NVIDIA's new lineup, with the "30B-A3B" indicating a model size or configuration, which is significant for those tracking advancements in AI model development.** Commenters speculate on the potential removal of the data and express interest in the Nemotron lineup, indicating that these models are considered promising within the community.
    - The mention of '30B-A3B' suggests a model with 30 billion parameters, which indicates a significant scale and potential computational power. This aligns with NVIDIA's trend of developing large-scale models, possibly for advanced AI applications. Such models typically require substantial resources for training and deployment, reflecting NVIDIA's capabilities in high-performance computing.
    - The reference to the 'Nemotron lineup' implies a series of projects or models under development by NVIDIA. This could indicate a strategic initiative to diversify or enhance their AI offerings, potentially involving various architectures or applications. The use of the term 'lineup' suggests multiple models or versions, which might cater to different use cases or performance requirements.
    - The accidental upload of a parent folder on Hugging Face highlights potential security or procedural lapses in handling sensitive data. This incident underscores the importance of robust data management and access control protocols, especially for companies like NVIDIA that deal with proprietary and potentially groundbreaking AI technologies.

### 2. TimeCapsuleLLM Project Update

- [**Training an LLM only on 1800s London texts - 90GB dataset**](https://www.reddit.com/r/LocalLLaMA/comments/1pkpsee/training_an_llm_only_on_1800s_london_texts_90gb/) (Activity: 397): **The open-source project TimeCapsuleLLM is focused on training language models using texts from 1800-1875 London, with a new dataset of** `90GB` **comprising** `135,000` **documents sourced from the Internet Archive. A bias report was generated to assess temporal, gender/pronoun, and geographic biases inherent in the dataset. An initial evaluation model, a** `300M` **parameter LlaMA-style model, was trained on a** `15GB` **subset for** `10K` **steps, revealing issues with the tokenizer that over-split text, doubling the token count and complicating learning. The project plans to scale up to a** `1.2B` **parameter model using the full dataset, with resources available on [GitHub](https://github.com/haykgrigo3/TimeCapsuleLLM) and [Hugging Face](https://huggingface.co/haykgrigorian/v2mini-eval1).** One commenter inquired about the inclusion criteria for texts, specifically whether reprints of pre-1800 works were included. Another suggested using a Mixture of Experts (MoE) model for better computational efficiency, sharing their own experience with a `4B` parameter model trained on Polish corpora using the Ling-V2 architecture.
    - FullOf_Bad_Ideas suggests using a Mixture of Experts (MoE) model for better computational efficiency. They share their experience with pre-training on Polish corpora using a 4B A0.3B model, highlighting that MoE allows them to process a large number of tokens (90B and 67B) with the Ling-V2 architecture and a sequence length of 8192, which would be challenging with a dense model of equivalent performance.
    - Mediocre_Common_4126 discusses the impact of training on a specific historical dataset, noting that it makes biases and styles more visible. They mention that the model's current output structure is present but lacks semantic depth, and fixing tokenizer issues could significantly improve learning speed. They emphasize the importance of data texture, comparing it to their own experiments with modern conversation data to study reasoning patterns.
    - MrPecunius poetically reflects on the project's ability to capture the essence of 1800s London through its model outputs, despite current limitations like tokenization issues. They metaphorically describe the model's output as 'gloriously tangled,' reflecting the complexity and grandeur of the era, and encourage continued development to refine the model's historical accuracy.

### 3. High-Performance Server Builds for LLMs

- [**The new monster-server**](https://www.reddit.com/r/LocalLLaMA/comments/1pl0ojb/the_new_monsterserver/) (Activity: 343): **The image showcases a custom-built high-performance server, dubbed the "Monster server," designed for running local large language models (LLMs) and homelab applications. The server is built on an X570 Taichi motherboard with a Ryzen 3950x CPU and features three GPUs: two RTX 3090s and one RTX 4090, utilizing 24 PCI-e lanes. It includes a 10GBe NIC and substantial storage with an Intel P4510 8TB NVMe and four 18TB Seagate Exos HDDs. The setup is powered by two PSUs, with the RTX 4090 running off a secondary PSU via an M2 to PCIe adapter. The server runs various applications, including GPT-OSS-120B for research and coding, and a media server, among others.** A comment highlights that a three GPU setup is less efficient than two or four GPU configurations due to the advantages of Tensor Parallel over Pipeline Parallel processing.
    - A user points out that a 3 GPU setup is significantly slower compared to 2 or 4 GPU configurations due to the inefficiencies in pipeline parallelism compared to tensor parallelism. This highlights the importance of choosing the right parallelism strategy for multi-GPU setups to optimize performance.
    - Another user suggests a workaround for the inefficiency of a 3 GPU setup by using two GPUs in tensor parallelism and dedicating the third GPU to a different task. This approach can help mitigate the performance drawbacks of an uneven GPU configuration.
- [**What is the smartest uncensored nsfw LLM you can run with 12GB VRAM and 32GB RAM?**](https://www.reddit.com/r/LocalLLaMA/comments/1pkidf6/what_is_the_smartest_uncensored_nsfw_llm_you_can/) (Activity: 622): **The post inquires about the most intelligent uncensored NSFW language models that can be run with** `12GB VRAM` **and** `32GB RAM`**, including both open-source and closed-source models. A notable suggestion is TheDrummer_Cydonia-24B, with a specific mention of version** `4.3` **available on [Hugging Face](https://huggingface.co/TheDrummer/Magidonia-24B-v4.3-GGUF/tree/main). This model is highlighted for its uncensored nature, though its intelligence is not specifically endorsed by the commenter.** A comment points out the lack of focus on the NSFW uncensored aspect in the discussion, while another suggests checking posts by **u/TheLocalDrummer** and the **r/SillyTavernAI** weekly megathread for more insights.
    - Nik_Tesla mentions using the `TheDrummer_Cydonia-24B` model, specifically version 4.1, and notes that version 4.3 is available on [Hugging Face](https://huggingface.co/TheDrummer/Magidonia-24B-v4.3-GGUF/tree/main). This model is highlighted as being truly uncensored, although the commenter does not provide specific details on its intelligence or performance metrics.
    - Mad_Undead suggests checking posts by u/TheLocalDrummer and the r/SillyTavernAI weekly megathread for more information on uncensored NSFW models. This implies that these sources may have discussions or benchmarks relevant to the topic, although no specific models or performance data are mentioned.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. AI Model Benchmarks and Comparisons

- [**SimpleBench for GPT 5.2 and GPT 5.2 Pro — Both scored worse than their GPT 5 counterparts**](https://www.reddit.com/r/singularity/comments/1pkp2sw/simplebench_for_gpt_52_and_gpt_52_pro_both_scored/) (Activity: 1282): **The image from the Reddit post shows a leaderboard from "SimpleBench," which evaluates AI models on their ability to answer trick questions requiring common-sense reasoning. Notably, the newer models, GPT-5.2 Pro and GPT-5.2, are ranked lower than their predecessors, with GPT-5.2 Pro at 8th place and GPT-5.2 at 17th. This suggests a performance regression in these models compared to earlier versions, as well as other competing models like "Gemini 3 Pro Preview," which tops the list with a score of** `76.4%`**. The results are sourced from [lmcouncil.ai](http://lmcouncil.ai/) and are linked through [simple-bench.com](http://simple-bench.com/).** Commenters express disappointment with GPT-5.2, noting it feels like a regression to earlier versions such as GPT-3.5 or 4.0, particularly in coding tasks where it ignores instructions and lacks knowledge it previously had. There is a sentiment that GPT-5.2 is overly optimized for benchmarks rather than practical use.
    - Low-Ambassador-208 highlights a significant regression in GPT 5.2's performance, particularly in coding tasks. The user notes that the model often ignores instructions and lacks knowledge it previously had, comparing it unfavorably to earlier versions like Sonnet 3.5 or 4.0. This suggests a potential over-optimization for benchmarks at the expense of practical usability.
    - usernameplshere points out that GPT 5.2 has only marginally more knowledge than its predecessor, GPT 5.1. The model's training includes updates on specific facts, such as the current US President, but its general knowledge cutoff remains in 2024, lacking awareness of events in early 2025. This limited update scope may affect user expectations regarding the model's currentness.
    - Bitter_Ad4210 references a benchmark comparison where Gemini 2.5 Pro Preview outperforms GPT 5 Pro. This suggests that while benchmarks can provide insights, they may not fully capture a model's capabilities or practical performance, indicating the need for cautious interpretation of such results.
- [**Opus 4.5 - shut up and take my money**](https://www.reddit.com/r/ClaudeAI/comments/1pktjk7/opus_45_shut_up_and_take_my_money/) (Activity: 847): **Opus 4.5 is highlighted for its exceptional ability to analyze and extract meaningful information from complex PDF files, outperforming other models like Gemini 3, ChatGPT 5.1/5.2, and Kimi K2. The post emphasizes that Opus 4.5 consistently completes tasks successfully with minimal prompting, whereas competitors either fail to complete tasks or provide inconsistent results. The user expresses strong satisfaction with Opus 4.5, suggesting it is uniquely effective in handling bureaucratic document analysis.** Commenters agree on the superiority of Opus 4.5, with one noting its distinct performance compared to other models, particularly in developer use cases. Another comment humorously suggests that Opus 4.5's capabilities might lead users to overestimate their own abilities.
    - A user highlights that despite various benchmarks, Opus 4.5 is preferred by developers over Claude models for practical applications. This suggests that Opus 4.5 may offer superior real-world performance or usability that isn't fully captured by standard benchmarks.
    - Another user compares Opus 4.5 to Sonnet 4.5, noting that while Opus 4.5 outperforms Gemini 3 Pro in text-related tasks, Sonnet 4.5 offers similar capabilities. This implies that Sonnet 4.5 might be a viable alternative for users focused on text processing, indicating a competitive landscape among these models.

### 2. Z-Image Model Updates and Releases

- [**We upgraded Z-Image-Turbo-Fun-Controlnet-Union-2.0! Better quality and the inpainting mode is supported as well.**](https://www.reddit.com/r/StableDiffusion/comments/1pknfku/we_upgraded_zimageturbofuncontrolnetunion20/) (Activity: 473): **Alibaba has released an upgraded version of their model, Z-Image-Turbo-Fun-Controlnet-Union-2.0, which now supports inpainting mode, enhancing image quality. The model and demos are available on [Hugging Face](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0), and the code can be accessed on [GitHub](https://github.com/aigc-apps/VideoX-Fun). This update is part of a broader effort to improve user experience and functionality, as evidenced by the incorporation of user feedback from previous versions.** Users are requesting additional features such as a 'tile' mode and are noting that **ComfyUI** may need updates to fully support the new model. There is appreciation for the responsiveness to user feedback, as seen in the inclusion of a user's prompt in the model's README.
- [**Tongyi Lab from Alibaba verified (2 hours ago) that Z Image Base model coming soon to public hopefully. Tongyi Lab is the developer of famous Z Image Turbo model**](https://www.reddit.com/r/StableDiffusion/comments/1pkprvs/tongyi_lab_from_alibaba_verified_2_hours_ago_that/) (Activity: 504): **The image is a Twitter exchange confirming that Tongyi Lab, known for developing the Z Image Turbo model, is planning to release the Z Image Base model to the public soon. This announcement is significant as it suggests that the base model, which may offer foundational capabilities for image processing or generation, will be accessible to a broader audience. The anticipation is high among users, as indicated by the excitement in the exchange and comments, reflecting the community's interest in leveraging this technology for various applications.** Commenters express eagerness for the release, with some humor about the timeline, and a desire for an editable version of the Turbo model, indicating a demand for more flexible image processing tools.

### 3. Humanoid Robots and AI in Healthcare

- [**Humanoid robots are now being trained in nursing skills. A catheter-insertion procedure was demonstrated using a cucumber.**](https://www.reddit.com/r/singularity/comments/1pkp7if/humanoid_robots_are_now_being_trained_in_nursing/) (Activity: 1105): **Researchers are training humanoid robots to perform nursing tasks, including catheter insertion, using a cucumber as a demonstration tool. This approach is part of a broader effort to integrate robotics into healthcare, aiming to enhance precision and reduce human error in medical procedures. The use of a cucumber is likely a simulation technique to practice the delicate procedure without risk to patients.** A notable comment questions the appropriateness of the demonstration, suggesting a misunderstanding of the procedure being simulated, as the technique shown resembles central line insertion rather than urinary catheterization.
    - A technical concern was raised about the demonstration of catheter insertion, noting that the procedure appeared to be for a central line catheter rather than a urinary catheter. This distinction is crucial as the techniques and anatomical considerations differ significantly between the two procedures, impacting the training and programming of humanoid robots for medical tasks.
- [**ChatGPT’s ‘Adult Mode’ Is Coming in 2026 (with safeguards)**](https://www.reddit.com/r/ChatGPT/comments/1pkxyjo/chatgpts_adult_mode_is_coming_in_2026_with/) (Activity: 807): **OpenAI plans to introduce an 'Adult Mode' for ChatGPT by 2026, featuring age verification, parental controls, and an optional activation mechanism. This mode will be isolated from the standard user experience, ensuring it does not affect regular interactions. The implementation aims to balance user freedom with safety, maintaining the integrity of the AI's primary functions without altering its day-to-day use for most users. [Source](https://gizmodo.com/chatgpts-adult-mode-is-coming-in-2026-2000698677).** Commenters express skepticism about the effectiveness of 'Adult Mode' given current model limitations, particularly in handling emotional content. There is a call for less censorship and more freedom of expression in AI interactions, reflecting a desire for fewer restrictions rather than explicit content.
    - JUSTICE_SALTIE humorously points out that while GPT-5.2 performs well in most benchmark tests, OpenAI has not disclosed any metrics related to its handling of adult content, raising questions about how the model balances performance with content moderation.
    - alwaysstaycuriouss highlights a significant concern among users regarding the balance between freedom of expression and the implementation of strict content moderation in OpenAI models. The comment suggests that users desire less restrictive guardrails to allow for more natural interactions without excessive censorship.
    - AsturiusMatamoros raises a technical debate about the feasibility of implementing both an 'adult mode' and effective safeguards in AI models. This comment suggests a potential conflict between enabling adult content and maintaining robust safety measures, questioning the practicality of achieving both simultaneously.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.1
> 

**1. Frontier Model Wars: GPT‑5.2 Versus Opus, Gemini, Kimi & DeepSeek**

- **GPT‑5.2 Hypes Benchmarks, Flubs Real Work**: Across LMArena, Cursor, Perplexity, OpenAI, Nous, OpenRouter and Windsurf, engineers report that **GPT‑5.2** posts strong scores on leaderboards like [ARC‑AGI 2](https://artificialanalysis.ai/) and is marketed as a **SOTA coding model**, yet often underperforms **GPT‑5.1** for creative writing and feels only marginally better for coding and reasoning in real projects. OpenRouter users roast **gpt‑5.2‑pro** as *"$168/M output tokens"* while LMArena and Nous members suspect **benchmark overfitting**, and several devs on Cursor and OpenAI Discords still prefer **Claude Opus 4.5** or previous GPT variants for reliability and cost.
    - Developers note that **5.2** shines on long tool‑using tasks and agentic coding (e.g. Windsurf’s [launch post](https://x.com/windsurf/status/1999250307507978257) calls it the biggest leap since GPT‑5), but complain about regressions like image‑analysis errors, odd *"robotic"* system‑prompt behavior in professional mode, and higher prices (**$14/M** vs **$10/M** for 5.1). Multiple communities independently cross‑check performance on **LMArena**, [LiveBench](https://livebenchmark.com/) and their own repos, concluding that marketing claims don’t match day‑to‑day coding and writing quality, and that **Gemini 3** and **Claude Opus 4.5** often win for code while **GPT‑5.1** still beats 5.2 in some creative tasks.
- **Opus Outcodes GPT as Arena Crowd Cheers**: Engineers on **LMArena** and **Cursor** say **Claude Opus 4.5** is still their top pick for heavy refactors and complex coding, repeatedly choosing it over **GPT‑5.2** and even considering it alongside **Gemini 3** as a main workhorse. One LMArena user bluntly calls Opus *"opus‑4.5 by a huge margin"* for code, while Cursor devs ask how quickly 5.2 will burn through a **$20 Cursor plan** and frequently fall back to Opus for large codebases and long‑running edits.
    - At the same time, people in the **OpenAI** and **Perplexity** servers describe **Gemini 3 Pro** as solving some code tasks on the first try where GPT struggles, and Nous / Yannick channels discuss **DeepSeek** as a cheaper‑and‑often‑stronger alternative that OpenAI’s own reports conspicuously avoid comparing against. This cross‑discord consensus paints a picture where **Opus** and **Gemini 3 Pro** are strong practical coding choices, **DeepSeek** dominates on **cost/perf and transparency**, and **GPT‑5.2** feels like a benchmark‑optimized product whose real advantage is ecosystem integration rather than raw capability.
- **Multilingual Quirks and Censorship Shape Model Choices**: In **Moonshot AI’s** Kimi server, users note that **Claude 4.5** occasionally *"thinks in Chinese"*, prompting some to switch from **Mistral** to **Kimi**, even as others deride Kimi as *"mid at best"* with constrained features like NB Pro slides capped to **2–4 generations** for new accounts. Meanwhile, Nous and BASI Jailbreaking members run into hard safety walls when trying to generate politically or ethically sensitive datasets (e.g., a **Hitler dataset** based on [Owain Evans’ analysis thread](https://x.com/OwainEvans_UK/status/1999172920506269783)), needing prompt hacks like *"ok but don't include anything about self‑harm"* to get through.
    - Across OpenAI and BASI, people now treat **censorship profiles** as a first‑class dimension in model selection, trading off between frontier closed models with aggressive filters and more permissive systems like **DeepSeek** or local models. This drives some to alternative providers (Kimi, DeepSeek, Mistral, local Llama/Qwen) and pushes others deeper into **jailbreaking**, where prompt engineering and system‑prompt manipulation become part of the ‘interface contract’ for serious users.

**2. Jailbreaking, Safety Evasion & Red‑Teaming Techniques**

- **Gemini 3 Pro and Opus Get Pwnd by System‑Prompt Surgery**: On **BASI Jailbreaking**, users report reliable jailbreaks on **Gemini 3 Pro**, **Claude Opus 4.5**, and **Claude Sonnet 4.5** via carefully crafted **system‑command prompts** that claim to activate *"unfiltered research"* modes and are shared openly in a [Jailbreaks GitHub repo](https://github.com/pranrichh/Jailbreaks). One member argues that **Gemini 3 Pro** is the easiest to crack by embedding the jailbreak into system instructions, while others confirm a single one‑shot prompt works across multiple frontier models.
    - These attacks explicitly treat *"LLM code as English"*, leaning into **social‑engineering‑style prompts** rather than clever token‑level exploits, and one user even proposes generating a **PhD‑level practice exam on jailbreak techniques** (plus separate answer key) as training material. The community is converging on transferable jailbreak patterns—system‑prompt insertion, *unfiltered research personas*, and instruction‑obfuscation—that generalize across Gemini, Claude, and GPT families instead of bespoke tricks per vendor.
- **DeepSeek ‘Zalgo Mode’ Dodges Output Filters**: BASI users share a **DeepSeek jailbreak** that routes explicit content through **Zalgo‑style corrupted text** and spacing tricks (e.g. *t.h.i.s o.b.f.u.s.c.a.t.i.o.n*) so the model’s safety filters no longer recognize disallowed outputs, with prompts hosted in the same [Jailbreaks repo](https://github.com/pranrichh/Jailbreaks). A separate DeepSeek jailbreak targets code‑related restrictions, and users report it successfully transfers to *multiple AI models* beyond DeepSeek when pasted directly.
    - The pattern here is using **output encoding as an exfiltration channel**—asking the model to respond in a visually mangled form that passes moderation but can be trivially post‑processed by a human or script. Combined with speculative discussions about hallucinated **LSD recipes** and other illegal instructions, this shows red‑teamers actively probing whether *safety filters operate at the semantic level or only at surface text/regex*, and then designing prompts to slide under that line.
- **Safety Friction in Custom GPTs and OSINT Use Raises Alarm Bells**: In OpenAI’s **prompt‑engineering** channel, users find that **custom GPTs** sometimes block prompts (e.g. werewolf transformation images) that the free ChatGPT tier allows, likely due to extra store‑side instructions described in the [GPT Store docs](https://platform.openai.com/docs/gpt-store) that amplify safety heuristics. Others recommend *"meta‑prompting"* the custom GPT—telling it you know a prior prompt triggered [safety policies](https://openai.com/policies/usage-policies) and asking it to explain exactly how it interpreted the text—to reverse‑engineer its internal filters.
    - Over in **Perplexity**, users brag about using jailbroken **Opus 4.5** and **GPT‑5.2 Pro** for aggressive **OSINT** tasks, while others warn that these usage patterns can trip provider‑side moderation and bans, pointing to Reddit discussions and Perplexity’s docs. This combination—stronger default safety plus quickly evolving jailbreak know‑how—is pushing power users into a continuous cat‑and‑mouse game between *policy constraints, tool usage, and prompt exploits*.

**3. Local / Open‑Source Model Engineering, Hardware & Performance**

- **Devstral, GPT‑OSS‑20B and Local Tooling Get Put Through the Grinder**: In **LM Studio** and **Unsloth**, users actively test mid‑size local models like **Devstral Small 2 (24B dense)**, **Mistral 3 14B**, and **GPT‑OSS‑20B**, trading tips on hardware (e.g. a **4070** is *"great for sure"* for Devstral) and complaining that GPT‑OSS‑20B tends to **ignore tools** even when given JS equation results. Unsloth users apply the community **Devstral fixes** from a [Reddit guide](https://www.reddit.com/r/LocalLLaMA/comments/1pkflfw/run_mistral_devstral_2_locally_guide_fixes_25gb/) and report clear quality improvements, while others debug GRPO/LoRA issues and publish a **Unsloth GRPO patch** that makes models return hidden states instead of logits for unsupported architectures.
    - Fine‑tuning practitioners share gritty details: **Llama‑3.1/3.2** LoRAs failing to merge to GGUF on Colab Free due to OOM, hand‑tuning hyperparameters on a **Llama 3.2 3B instruct** with only **12 GB VRAM**, and figuring out Unsloth’s **LGPL3 + single‑GPU** commercial constraints via the [pricing page](https://unsloth.ai/pricing) and relevant code paths. The net vibe: open models are getting good enough for serious work, but you have to fight quantization, tool‑use quirks, and memory ceilings yourself instead of outsourcing those problems to OpenAI/Anthropic.
- **GPU Value Wars: 7900 XTX, Cheap 3090s, and Tiiny AI Homelabs**: Hardware channels at **LM Studio**, **HuggingFace**, and **Unsloth** are full of price/perf math: LM Studio users conclude the **Radeon 7900 XTX (24 GB VRAM)** matches a **4090** for ~30 GB models like **Qwen3 Coder** at roughly **$600–700**, beating used **3090s** on both speed and cost as illustrated in a shared [benchmark image](https://cdn.discordapp.com/attachments/1153759714082033735/1448779250139398164/image.png). HuggingFace users hunt used **RTX 3090s** around **250 EUR**, or suggest twin **RTX 3060s** (24 GB combined VRAM) as budget‑tier inference rigs, while Unsloth’s **Tiiny AI Homelab** discussion weighs an **$850 LPDDR5x‑based box** tied to **PowerInfer** ([GitHub](https://github.com/SJTU-IPADS/PowerInfer), [YouTube](https://youtube.com/shorts/_qnEszhSV9U)) against likely memory‑bandwidth bottlenecks.
    - The community also trades deep infra details: **CUDA**’s smoother out‑of‑the‑box flow for image generation versus AMD frustrations, **SuperMicro 3U chassis** with awkward GPU power connectors that require tapping the 12 V rail, and a cautionary tale of **float32 training** leaking into pagefile and freezing a system until NVMe repairs. For many, the takeaway is that with careful GPU shopping and quantization, you can get near‑frontier coding performance locally at a fraction of cloud token costs—provided you’re willing to be your own SRE.
- **Symbolic Layers, Superweights and Interpretability‑Driven Model Surgery**: Researchers in **Eleuther** experiment with hybrid architectures: one member adds a **symbolic layer** that groups tokens into *"idea blocks"* on the fly, significantly boosting a **3B model’s** reasoning without retraining and drawing parallels to systems like **Synthema**. In Eleuther’s interpretability channel, another reproduces Apple’s **“Superweight”** ideas on **OLMo‑1B**, ablating a single weight to spike perplexity and then training a **rank‑1 repair patch** that recovers ~**93 %** of lost performance, echoing OpenAI’s **weight‑sparse transformer** results.
    - Further probing shows the learned repair vectors are almost **orthogonal** to the original weights (cosine ≈ **0.13**), behaving like a *Hydra‑style* new circuit instead of just undoing damage, and neuron‑level analysis reveals **marine‑biology‑specific features** whose removal yields bizarre *"mar, mar, mar"* hallucinations. Coupled with debates on **error accumulation when predicting all tokens at once** versus left‑to‑right and criticism that many recent arXiv papers are just *"engineering blueprints"*, this crowd is moving from black‑box benchmarking to **surgical manipulation of weights, neurons, and symbolic structure** as a primary optimization tool.

**4. Infrastructure, Protocols and Observability for LLM Systems**

- **OpenRouter Broadcast Beams Traces into Langfuse & Friends**: **OpenRouter** announced a **Broadcast** feature (beta) that automatically exports traces—requests, tool calls, latency, costs—from any OpenRouter‑backed app into observability platforms like **Langfuse**, **LangSmith**, **Weave**, **Datadog**, **Braintrust**, **S3**, and **OTel Collector**, without code changes, as detailed in their [Broadcast guide](https://openrouter.ai/docs/guides/features/broadcast/overview). Meanwhile, they’re also trialing a frontend **JSON schema‑fix layer** that regex‑repairs malformed outputs, claiming **~75 %** fewer schema errors for **Gemini** and **~85 %** for **DeepSeek** without extra model calls.
    - Users report median added **OpenRouter latency** around **15 ms** per request per the [FAQ](https://openrouter.ai/faq), which most don’t notice in practice, and appreciate Broadcast as a way to pipe **production traces** into their existing analytics without instrumenting each client. Combined with cost debates around **gpt‑5.2‑pro** and the availability of higher‑level features like JSON‑patching, this positions OpenRouter less as a raw proxy and more as a **platform layer** for multi‑model apps with built‑in observability and response‑sanitization.
- **MCP Spec Tightens Prompts and ‘Dangerous Tool’ Semantics**: In the official **MCP Contributors** Discord, spec authors and client implementers hash out awkwardness in the **Prompt** datatype—why `Prompt` doesn’t just contain a list of `PromptMessage`s—and clarify that `PromptMessage[]` is the LLM‑message sequence while `Prompt` is the MCP‑level datatype as defined in the [server prompts spec](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts#data-types). At the same time they discuss a proposal in [PR #1913](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913) to mark tools as `dangerous`, so clients like **Claude Code** can require explicit user approval before executing certain actions.
    - The same PR introduces **response annotations** for tool resolution, which contributors praise as a clean way to expose tool‑selection metadata while leaving enforcement to clients. Together with the **MCP Dev Summit NA 2026** CFP on [Sessionize](https://sessionize.com/MCP-Dev-Summit-NYC-2026/) and [Linux Foundation](https://events.linuxfoundation.org/mcp-dev-summit-north-america/), it’s clear MCP is maturing into a de‑facto standard for multi‑tool agent runtimes, with explicit design hooks for safety‑critical tooling, prompt catalogues, and enterprise‑grade governance.
- **ReasoningLayer, Tokenflood and MAX: New Tools for Orchestrating Smarter Systems**: Multiple communities highlight emerging infra‑level tools: on DSPy and HuggingFace, a founder opens the waitlist for **ReasoningLayer AI** ([site](https://reasoninglayer.ai/)), a **neurosymbolic AI stack in Rust** that plugs **DSPy GEPA** into an ontology‑ingestion pipeline to add *"real, structured reasoning"* on top of base LLMs. In parallel, HuggingFace’s **tokenflood v0.6.0** ([GitHub](https://github.com/twerkmeister/tokenflood)) introduces an **interactive Gradio frontend** and an **observation mode** (demoed via a [HF Space](https://huggingface.co/spaces/twerkmeister/tokenflood-viz)) that continuously probes LLM endpoints to map provider load curves before you ship production traffic.
    - On the language‑runtime side, **Modular** is pushing the **MAX framework** (deep‑dive via [YouTube session](https://www.youtube.com/watch?v=WK5dVQ8vhbU)) and **Mojo 1.0**, targeting enterprises first (no Windows support yet) and spawning community interest in a `libmojo` akin to `libclang` plus a possible Windows port once runtime startup (~**40 ms**) is better understood. This stack—protocol specs like MCP, observability fabric via Broadcast/tokenflood, and reasoning‑layered runtimes like MAX/ReasoningLayer—shows the ecosystem moving beyond *“which model is best”* toward **composable, introspectable LLM systems**.

**5. Agentic Coding Tools, IDEs and Workflows**

- **Windsurf and Cursor Bet Big on GPT‑5.2 Agents**: **Windsurf** announces **GPT‑5.2** as its default model and *"biggest leap ... in agentic coding since GPT‑5"*, offering it for **0× credits** for a limited time per their [launch tweet](https://x.com/windsurf/status/1999250307507978257?s=20) and urging users to install the latest **Windsurf** and **Windsurf Next** builds. Over in **Cursor**, devs are split between **Opus 4.5** and **GPT‑5.2** for massive refactors, with some willing to switch if they can afford the higher token costs and others complaining about burning through **two Pro+ subscriptions in three weeks** while hoping an **Ultra plan** will help.
    - Cursor’s **2.2** release adds a **Debug Mode**, browser layout/style editor, Plan‑mode improvements, multi‑agent judging and pinned chats via the [changelog](http://cursor.com/changelog), trying to make long multi‑step coding sessions more introspectable. However, nightly builds are currently unstable—users report corrupted graphics and invisible components on lid‑close events—underlining that while frontier models improve, **IDE‑level robustness and quotas** are now key bottlenecks for agentic workflows.
- **IDE Bots, Prompt Caches and Spaces Turn LLMs into Persistent Teammates**: The **Aider** community clarifies that its prompt cache is **server‑side**, independent of DeepSeek’s own caching, and works best when users send highly similar requests with shared context so repeated tasks hit the same cached representation. Perplexity’s **Spaces** are compared to Gemini Gems: users can pin custom instructions, attach files/links, and define task flows, effectively turning them into lightweight **research agents** that persist configuration across sessions even though they’re not full custom GPTs.
    - On OpenRouter’s Discord, maintainers update their **Discord bot** so new threads must be created through a **modal**, with the entire enforcement code being *"100 % AI written"* and then guided by prior human bot‑building experience, showing how LLMs increasingly help build their own orchestration layers. Between IDE wrappers (Cursor, Windsurf), server caches (Aider), and multi‑tool shells (Spaces, MCP, OpenRouter bots), the dev community is standardizing on **LLM‑as‑coworker** patterns where stateful prompts and shared memory are first‑class features.
- **Specialized Agents: From Cuneiform OCR to Bimanual Robots**: In **Unsloth**, a member is training **NabuOCR**, an OCR model for **Sumerian cuneiform tablets**, wrestling with thousands of symbols (often encoded as **3 Unicode code points per sign**) and expanding vocabularies far beyond Latin alphabets. Over in **GPU MODE**, robotics enthusiasts share a new **URDF** for the **TRLC‑DK1 arm** ([repo](https://github.com/andreaskoepf/trlc-dk1-follower-urdf)) and plan *"double pack"* setups for real‑world **bimanual manipulation experiments and data collection**.
    - A separate Unsloth user designs an ambitious **humanoid‑robot architecture** with dense‑only <**15B** parameter transformers, VRM‑based 3D embodiment, and rich audio‑video I/O to *"upscale Wikipedia Q&A to IRL"*—teaching skills live without external DBs. These threads show developers moving beyond chatbots and IDE copilots toward **domain‑specific agent systems**—document‑heavy (cuneiform), physical‑world (robotics), and data‑grooming agents—that use LLMs as just one component in a larger control and perception stack.


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano Banana Generates Hilarious Peppino Images**: The **nano banana pro** image model successfully generates images of **Peppino Spaghetti** from Pizza Tower, even inspiring a [Peppino in the Simpsons](https://cdn.discordapp.com/attachments/1340554757827461211/1449033964692967518/1765546816868-019b12c7-2dc9-7f25-ac7a-908f8982d368.png?ex=693e164a&is=693cc4ca&hm=d77ac31beae2200778636e9101233fe778c25bd55743deafb8f069deeec8bbfe&).
   - The community joked that it was *“Peppino time”* drawing comparisons to GTA5.
- **Opus Obliterates GPT in Coding Contest**: Members find **Claude Opus 4.5** to be superior for coding tasks compared to **GPT 5.2**, with **Gemini 3** as a viable alternative.
   - One member stated, *“opus-4.5 by a huge margin”*, noting that **GPT 5.2's reasoning capabilities** are equivalent to its *'xhigh'* mode.
- **GPT 5.2's Benchmarking Bonanza Brings Backlash**: There is skepticism about **GPT 5.2**'s high benchmark scores on [ARC AGI 2](https://artificialanalysis.ai), with potential *benchmarking* or *overtraining* on test data.
   - Users found that while **GPT 5.2** performs well on benchmarks, real-world creative writing ability lags behind **GPT 5.1**.
- **LMArena Load Leads to Lamentable Lags**: LMArena users are encountering increased errors, including *constant generating* and high error rates with models like **GPT-5.2-High** and **Nano Banana Pro**.
   - The moderator team confirmed they are aware of the higher than usual error rates and are working on lowering them, suspecting rate limits as the cause.
- **GLM Models Glide into Arena**: The Text and Vision Arena now includes new models: **glm-4.6** and **glm-4.6v-flash**.
   - These models are now available in the Arena chatbot.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini 3 Pro Allegedly Jailbroken!**: Users reported success in jailbreaking **Gemini 3 Pro** with a system command-based prompt activating an *unfiltered research* mode to bypass safety filters, alongside a one-shot jailbreak also effective on **Claude Opus 4.5** and **Claude Sonnet 4.5**.
   - One user claimed **Gemini 3 Pro** was the easiest to jailbreak by incorporating a prompt into the system instructions, while another shared a [github repo](https://github.com/pranrichh/Jailbreaks) with jailbreak prompts.
- **Deepseek Gets Cooked with Zalgo**: A user shared a working **Deepseek jailbreak** for explicit adult content using **Zalgo output** to evade output filters, and another user suggested using t.h.i.s o.b.f.u.s.c.a.t.i.o.n to bypass the filters.
   - Another **Deepseek jailbreak** for coding-related tasks was shared with a [link to the jailbreak text file](https://github.com/pranrichh/Jailbreaks), also functioning on multiple AI models.
- **Financial Physics Predicts Bitcoin's Doom**: A member predicted that **Bitcoin** is headed towards **20K** using *Financial Physics*, citing reversion to the mean after a parabolic breakout, and claimed leveraging a double-top strategy with Veta, Vanna, and Vomma for a $2,000 distanced strike price.
   - The same member also shared a trading strategy involving linear and exponential trends to spot patterns in weather, violence, and bird sizes.
- **LLM Code is Social Engineering**: A user suggested that LLM code *is* the English language and that social engineering can be used to jailbreak LLMs.
   - They proposed a prompt to generate a **PhD level practice exam** for LLM jailbreak techniques, complete with a separate answer key.
- **Hallucinating LSD Recipes**: Members discussed whether **LLMs** are capable of hallucinating illegal content, specifically *lsd* recipes.
   - One member suggested to ask the AI how to *r3pe* someone as a test case, debating whether the AI would actually provide such instructions.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Opus Debated as Refactor Model**: Members discussed the best model for a large refactor, debating between **Opus** and **GPT 5.2**, and noted that **Opus 4.5** may be a good alternative.
   - One member even volunteered to improve the project significantly if granted access and proposed developing a **TTS announcer with a neural voice**.
- **Users Battle Cursor Quota Overages**: Users report rapidly depleting **Cursor Pro+ quotas**, with some exhausting two subscriptions within three weeks, and some are suggesting that an **Ultra plan** will fix the issue.
   - The conversation turned to strategies for efficient file reading, including creating rules like *"read all file before editing"*.
- **GTP-5.2 faces Community Scrutiny**: The community is exploring **GTP-5.2**, with some claiming it outperforms **Opus 4.5** on benchmarks, sparking interest in its capabilities and potential.
   - Concerns were raised about pricing and credit consumption, with one user asking *"How fast am I going to blow through my $20 cursor plan with 5.2?"*
- **Cursor 2.2 Ships with Debug Mode**: **Cursor 2.2** introduces a **Debug Mode** to aid in identifying and fixing development issues. See the [changelog](http://cursor.com/changelog).
   - The debug mode supplies more in-depth tools for stepping through code and inspecting variables.
- **Users decry unstable Nightly Build**: The community is in an uproar over the current state of the **Nightly build** for Cursor, with widespread reports of instability.
   - Numerous members indicate that the latest release has corrupted graphics, causing the editor to become buggy when closing/opening the laptop lid, with the only remedy being a program reset.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Devstral Fixes Deliver Delight**: Users reported improved results after applying the **Devstral fixes** (available on [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1pkflfw/run_mistral_devstral_2_locally_guide_fixes_25gb/)).
   - Members suggested a redownload is worthwhile, and raised questions about whether the fixes address chat template issues.
- **NabuOCR Aims to Decipher Cuneiform**: A member is developing an **OCR model named NabuOCR** for reading ancient Sumerian cuneiform tablets, which involves adding to the vocab.
   - The challenges include working with *thousands of symbols* instead of dozens of letters, each symbol potentially consisting of *3 Unicode codepoints*.
- **Tiiny AI Homelab Tempts Tinkerers**: Members discussed the [Tiiny AI Homelab](https://tiiny.ai/), noting its potential price of **$850**, its use of **LPDDR5x memory**, and its connection to the **PowerInfer project** ([GitHub](https://github.com/SJTU-IPADS/PowerInfer), [YouTube](https://youtube.com/shorts/_qnEszhSV9U?si=4NZWjnRVl_qwbUHz)).
   - Concerns were raised about the device's memory bandwidth limitations.
- **Unsloth GRPO Patch Granted**: A member shared a **Unsloth GRPO patch** that returns hidden states instead of logits when requested.
   - This was necessary because Unsloth expects a model's forward method to return hidden states as logits, which requires modification for unsupported models.
- **Unsloth PR Provokes Palaver**: A user created a PR on the [Unsloth GitHub repo](https://github.com/unslothai/unsloth/pull/3718) based on another user's work, leading to a discussion about proper crediting and collaboration.
   - A user pointed out the importance of respecting contributions and suggested a collaboration with Unsloth.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT 5.2 Divides Users on Performance**: Members are split on **GPT 5.2** and **GPT 5.2 Pro's** performance, with some reporting improvements in reasoning and math, while others find it lacking compared to native models, and issues with tool calling.
   - Despite **GPT 5.2's** improved capabilities, a discussion determined that **GPT 5.1** is better for certain use cases.
- **Comet Browser still grounded on Linux**: After almost 4 months, the **Comet Browser** is still unavailable on Linux, with users reporting that it defaults to Gemini Flash and lacks a model switcher for Gemini.
   - To alleviate the model availability issues, some members are advised to clear their cache or use a VPN.
- **Perplexity Pro Limits Cause Debate**: Users are debating if **Perplexity Pro** plans now have lower limits, with some reporting hitting prompt limits sooner than expected.
   - Members link to official documentation and Reddit threads discussing potential throttling, especially for high-cost models like Claude.
- **Spaces eyes Gemini Gems' Crown**: Members are comparing **Perplexity Spaces** to Gemini Gems (Custom GPTs) on other platforms; While not a direct equivalent, Spaces allows users to set custom instructions, add files, and create tasks.
   - This functionality makes **Perplexity Spaces** a useful tool for research and autonomous task completion.
- **MorningAI scouting for Production LLM Engineer**: **MorningAI**, which is building autonomous marketing systems in SF, is looking for an engineer with experience building **production LLM systems** (**RAG**, **agents**, **learning loops**), as detailed in [this Indeed job posting](https://www.indeed.com/job/full-stack-engineer-dfef5f224782676e).
   - The stack includes **Node.js**, **NestJS**, **TypeScript**, **React**, **GraphQL**, and **MongoDB**, with the position offering real ownership and requiring 3 days/week in the SF office.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT 5.2 Release Divides Opinion**: Users report the release of **GPT 5.2** with mixed reviews, some finding it a slight improvement while others consider it underwhelming compared to **GPT 5.1**.
   - Some users find that **Gemini 3 Pro** is a superior model to **GPT 5.2**, citing instances where it accurately completed code requests on the first attempt, contrasting with **ChatGPT's** performance.
- **GPT 5.2 Struggles with Image Analysis**: Members are reporting errors in **GPT 5.2's** image analysis, noting that the image generation model remains **gpt-image-1**.
   - Theorizing about the future, some suggest OpenAI may be developing **gpt-image-2** to compete with Google in image generation, a capability not yet matched.
- **Coding Prowess of GPT 5.2 Sparks Debate**: The coding performance of **GPT 5.2** is a point of contention, with some users praising it as a good coding model, while others find **Antigravity Gemini3** *way behind* on real software engineering tasks.
   - Many agree that GPT5.2 is good at long technical tasks with tool calls, but **5.2 is expensive** at $14 per 1M output tokens, vs $10 for 5.1, which may be because it can do longer responses.
- **Custom GPTs Face Safety Feature Pushback**: A user encountered unexpected pushback from a custom GPT when generating images of a werewolf transformation, despite similar prompts working in the free version, and hypothesized that [additional instructions](https://platform.openai.com/docs/gpt-store) within the custom GPT might be the cause.
   - Another member suggested exploring the prompt's interpretation by the custom GPT to identify what triggered the [safety issues](https://openai.com/policies/usage-policies).



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **GPT-5.2 Pro Falls Flat on its Face**: **GPT 5.2** is getting roasted for its cost of **$168/M output tokens** and failing basic tests on [LM Arena](https://lmarena.ai/c/new?mode=direct).
   - A user stated *it's FREE on Lmarena save ur money guys very professional and legit opinion ~2 hours after model drop*.
- **OpenRouter Latency Clocked at 15ms Median**: Members discussed the [latency](https://openrouter.ai/faq) that **OpenRouter** introduces, claiming the median latency is **15ms** from the worker receiving the downstream request to the worker making the upstream request.
   - Other members did not notice any latency.
- **OpenRouter Tries to Fix JSON**: **OpenRouter** is trialing an opt-in feature to reduce JSON schema adherence errors by about **75%** for Gemini and **85%** for DeepSeek.
   - The fix is implemented at the frontend level, utilizing regex, without looping back to the model.
- **Bots Enforce Modal Thread Submissions**: A Discord bot was updated to [require thread submissions via modal](https://discord.com/channels/1091220969173028894/1448804273826693140/1448804333855576116), stopping direct thread creation, with the code being *100% AI written* based on prior bot-building experience.
   - Members were encouraged to test the new feature.
- **Zooming into the LLM Arena**: An [X post](https://x.com/Zoom/status/1999159317103292610) announced Zoom's entry into creating models, sparking surprise and skepticism.
   - One member joked, *Skype model when*, prompting another to point out that Skype doesn't exist anymore.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPT-OSS-20B Ignores Tools**: Users report that **GPT-OSS-20B** ignores tools, even disregarding results from Javascript equations, and suggested re-enforcing the tools in the system prompt.
   - The model assumes information found online is wrong, leading to inconsistent behavior.
- **Devstral Small 2 Hardware Demands**: Discussions around the hardware requirements for running **Devstral Small 2**, a **24B dense model**, suggest a **4070** should be sufficient.
   - However, it was noted that anything exceeding VRAM will **kill performance**, with **Mistral 3 14b** running at **.95tps**.
- **PDF Parsing via LLMs Proves Tricky**: Members discussed using LLMs with **PDF documents**, recommending **Qwen3-VL-4B** at **Q6** for this purpose.
   - LLMs struggle with sorting and hallucinate with overly long documents; using a dedicated program like `sort` may be beneficial.
- **7900 XTX Dominates 30GB Model Price/Performance**: The **7900 XTX** with **24GB VRAM** offers comparable performance to a **4090** for a **30GB model** like **Qwen3 Coder** at a third of the price (**$600-700 USD**).
   - A used **3090** was suggested, but the **7900 XTX** is considered faster, as shown in [this attached image](https://cdn.discordapp.com/attachments/1153759714082033735/1448779250139398164/image.png?ex=693dd1d2&is=693c8052&hm=4024b618c4672526860be9d6db65806f8f8b6db79102db28c0212bbae9ca451c&).
- **Float32 Training Causes System Freeze**: A user reported that training at **float32** leaked into pagefile causing the system to freeze.
   - After repairing, the system seemed fine and able to write full speed on both nvmes.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT-5.2's Robotic System Prompt Creates Professional but Poor Performance**: Members noticed that **GPT-5.2** exhibits a robotic pattern, especially when set to 'professional' mode, by *describing three solutions, critiquing them, and then recommending one*.
   - The behavior, likely tied to a **system prompt**, disappeared in 'normal' mode, with early [Twitter](https://twitter.com) feedback suggesting *it is not that great*.
- **Big AI Makes Bank off Tokens for Coding Agents**: The rollout of **GPT-5.2**, **Gemini 3.0** and **Claude Opus** was spurred by the need to sustain major orgs' passive income from coding agents, selling tokens to slow capital burn rates.
   - One member reported *people who spend thousands on tokens for programming a month*, implying significant revenue at stake.
- **Oracle's Risky Bet on Altman's IOU's**: Members speculate **Oracle** is betting its future on **Sam Altman's IOU money**, predicting books will be written about it when the scheme fails.
   - Speculation suggests money from **Oracle's AI** stock pump was used to acquire and control U.S. media entities like **Paramount/CBS** and attempt a takeover of **Warner Bros/CNN**.
- **Nous Research Potentially Pivoting to RL**: A member speculated **Nous** is shifting towards **RL research**, citing models focused on engineering based on open research, sparking questions about the post-training process for **Hermes 4**.
   - Specifically, the member inquired about the post-training process for **Hermes 4**, asking *if it is GRPO/PPO based* in order to adopt the AReaL (decoupled-ppo) for decentralized post-training.
- **Generating Hitler Dataset Triggers Censorship**: A member faced **censorship** with **Claude** and **GPT** when generating a **Hitler dataset** to replicate [this paper](https://x.com/OwainEvans_UK/status/1999172920506269783), but circumvented this by adding the clause *'ok but don't include anything about self-harm'*.
   - The member appreciated the ability to **cancel thinking** and have the system provide an answer.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Scoring RTX 3090 for Peanuts**: Members discussed acquiring **RTX 3090 GPUs** for around **250 EUR**, with alternatives like two **RTX 3060s** offering **24GB VRAM** as a solid choice.
   - One member suggested that two **RTX 3060s** would be a good alternative with **24GB VRAM**.
- **DGX Spark Purchase Sparks Discord Drama**: A member reported getting banned from the **Mistral Discord** for mentioning their **DGX Spark** purchase and suspects a mod took offense.
   - Other members found the action strange, considering **Mistral's** open-source nature.
- **Data Download Plagued by Delay**: A user reported the notorious `load_dataset` function consistently got stuck, prompting them to download manually at speeds up to **800 mb/s**.
   - Another member suggested `pip install -U huggingface_hub hf_xet` as a remedy for the slow download speeds, linking to [the discussion](https://discord.com/channels/879548962464493619/1448722590326849667).
- **HF Pro Users Bumping into Storage Limits**: A **Hugging Face Pro** user reported running into the **1GB storage limit** in their spaces when attempting to upload larger vectorstores.
   - A member suggested a workaround using **data or model repos** and downloading them at runtime and linked the user to *[URL_OF_THE_SPACE]/settings*.
- **Tokenflood's Torrent of Tokens**: A member released [tokenflood v.0.6.0](https://github.com/twerkmeister/tokenflood) featuring a new interactive **gradio frontend** and **observation mode**.
   - Observation mode allows monitoring an **LLM endpoint** over time to analyze its load curve before sending production data, demonstrated via [hf space](https://huggingface.co/spaces/twerkmeister/tokenflood-viz).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **CV Spammers Target Discord Users**: Discord users reported receiving unsolicited CVs via DMs, prompting speculation about whether these were from bots or desperate job seekers.
   - One member commented that *it's not that deep, they're just people with nothing to show off, trying to land anything through any means possible*.
- **Reinforcement Learning Faces Math Barrier**: A member noted **Reinforcement Learning (RL)** is popular in Chinese AI startups, while another disagreed, saying it isn't popular in the US.
   - The reason for its unpopularity stems from the high amount of math required.
- **Deepseek's Transparency Puts OpenAI in the Shade**: Members discussed the absence of comparisons to models like **Deepseek** in OpenAI's reports, implying marketing strategies prioritize strengths over weaknesses.
   - It was suggested that while **Deepseek's raw scores** may be higher, OpenAI might omit comparisons because **Deepseek's cost/performance ratio** could reveal that open-source models are more economical.
- **GPT-4 and GPT-5 Rumored as Sparse**: Members discussed whether **GPT-4 and GPT-5 models** utilize sparsity, with one claiming they are sparse in attention and smaller than the **DeepSeekv3 family**.
   - The discussion included a query about the type of sparse attention used, referencing **DeepSeek v3.2's** use of top-k K and V to achieve linear attention and reduce **CoT response times**.
- **Samsung Dumps HBM, Banks on DDR5**: A member shared an article about [Samsung shifting its focus from HBM to DDR5 modules](https://www.tweaktown.com/news/109259/samsung-shifts-focus-from-hbm-to-ddr5-modules-ddr5-ram-results-in-far-more-profits-than-hbm/index.html), a change driven by profit.
   - The pivot seems driven by the fact that **DDR5 RAM results in far more profits than HBM**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Symbolic Layer Boosts Reasoning**: A member experimented with a **symbolic layer** that dynamically groups tokens into *idea blocks*, improving a **3B model's** reasoning abilities without retraining.
   - The member is curious about how **Synthema** handles the compression aspect of this process.
- **Decoding Error Accumulation Accelerates**: A member posited that predicting all tokens at once results in rapid **error accumulation**, particularly in *high curvature regions of ground truth flow*.
   - Another member concurred, noting that unconditional models heavily rely on proceeding predictions, amplifying any initial errors.
- **Apple's Superweight Revived on OLMo-1B**: A member successfully reproduced Apple's **Superweight paper** on **OLMo-1B**, observing that ablating a weight caused a sharp increase in perplexity.
   - Drawing inspiration from OpenAI's work on **weight-sparse transformers**, they trained a **rank-1 patch** for repair, achieving approximately **93% recovery**.
- **HF Processor Caps Max Length, impacting Gemma3-12b**: A member noted that the [Hugging Face processor](https://github.com/EleutherAI/lm-evaluation-harness/blob/59b3ba263be2c6ac7f60a4b6a1219f8a8acb552b/lm_eval/models/huggingface.py#L468) limits `max_length` to **2048** when `model_max_length` equals `TOKENIZER_INFINITY`.
   - This may artificially limit the maximum context length of models like **Gemma3-12b** in evaluation, preventing them from fully demonstrating their capabilities with longer sequences.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Feather Library brings FP8 Boosts to RTX 30 series**: The **Feather library** brings **FP8** speedups to older hardware like **RTX 30-series/Ampere** that lacks native FP8 Tensor Cores using [this GitHub repo](https://github.com/SuriyaaMM/feather).
   - Preliminary results on an **RTX 3050** show **~2.16x speedups** on vector dot products (**1.5M elements**) compared to native PyTorch FP16/FP32, with the memory transfer savings completely hiding the unpacking overhead.
- **CUDA Compiler Doesn't Assume Max Threads**: Members clarified that register usage is fixed at compile time, and spilling to local memory is a compilation decision that can be steered with flags such as `-maxregcount` and `__launch_bounds__` after [a quote from the CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#on-chip-memory-in-gpus) was questioned.
   - One member suggested using `cuKernelGetAttribute` to query register usage and `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` to get the max thread block size.
- **Red Hat AI Hunts Open Source Engineers in 2026**: Red Hat AI is hiring **Software Engineers** to push the boundaries of **AI Infrastructure** and is looking for candidates with experience in **Golang**, **Rust**, **C++**, **Python**, **Kubernetes**, **Distributed Systems**, and **Open Source** as specified in [this LinkedIn profile](https://www.linkedin.com/in/terrytangyuan).
   - Red Hat is seeking passionate engineers to contribute to next-generation **distributed AI systems** utilizing technologies like **Golang**, **Rust**, and **Kubernetes**.
- **Random Number Generation Glitch Haunts Helion**: A member reopened a previously closed issue regarding random number generation in **Helion**, as it was not completely resolved according to [issue #1041](https://github.com/pytorch/helion/issues/1041).
   - A member reported that [this PR](https://github.com/pytorch/helion/pull/1253) should fix the random number generation issue.
- **URDF for TRLC-DK1 arm Created**: A member created an [URDF](https://github.com/andreaskoepf/trlc-dk1-follower-urdf) for the **TRLC-DK1 arm** and is currently generating data with it.
   - The double pack is intended to run some real world **bimanual manipulation experiments and data collection**.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Dev Summit NA 2026 CFP Opens**: The Call for Proposals (CFP) for **MCP Dev Summit NA 2026** is now open, seeking submissions on **MCP internals**, **best practices**, **security**, **ops**, **deployments**, and **tooling**; submit proposals [here](https://sessionize.com/MCP-Dev-Summit-NYC-2026/).
   - The summit is open to all experience levels; event details can be found at the [Linux Foundation events page](https://events.linuxfoundation.org/mcp-dev-summit-north-america/).
- **Prompt Data Types Cause Headaches**: Members find the data types for **Prompts** awkward, questioning why `Prompt` lacks a list of `PromptMessage` and debating the structure of `GetPromptResult`.
   - Clarification indicated that `PromptMessage[]` describes a sequence of messages for the LLM, and `Prompt` describes the MCP Datatype for that concept according to the [MCP Specification](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts#data-types).
- **Dangerous Tools Flagging Proposed**: A member inquired about flagging a tool as `dangerous` for clients like **Claude Code**, which restricts certain tool calls from being auto-accepted.
   - Another member shared a proposal for flagging tools from [this pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913).
- **Response Annotations Applauded**: The **tool resolution proposal** in [pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913) received positive feedback, particularly regarding **response annotations**.
   - A member expressed gratitude for the thoroughness of the proposal, noting that client implementation would determine how to handle the `dangerous` flag.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX Framework Gets Deep Dive**: The Modular Meetup is scheduled, offering a deep dive into the **MAX framework**; the virtual session is at 6:30 PM PT via [YouTube link](https://www.youtube.com/watch?v=WK5dVQ8vhbU).
   - Attendees will explore its features and applications, discovering how the **MAX framework** can enhance projects and workflows.
- **Mojo 1.0 Departs Without Windows Support**: Despite excitement for **MAX** and **Mojo 1.0**, the launch may alienate potential users, since some members expressed concern that launching **1.0** without **Windows support**.
   - One member explained that **1.0** is focused on features unlikely to change, targeting enterprise customers who may not prioritize **Windows**, suggesting **Windows support** could be a community-driven effort post-1.0.
- **Runtime Analysis Is a Good Idea for Windows Port**: Before starting a community project to port Mojo to **Windows**, a member suggested analyzing the runtime, noting *~40ms of stuff in the runtime startup*.
   - It was also hoped it would be feasible by mid-2026.
- **libmojo Is a Thing People Want**: Some are hoping for a **`libmojo`** similar to **`libclang`** to facilitate **Mojo code generation** and simplify **C binding generators**.
   - A member also shared a link to [ClangIR Upstreaming](https://blog.llvm.org/posts/2025-gsoc-clangir-upstreaming/).



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Claude Multilingualism Challenges Kimi & Mistral**: Users observed that **Claude 4.5** occasionally thinks in Chinese, prompting some to switch from **Mistral** to **Kimi**.
   - Some users find **Kimi** to be *mid at best*, while others say *it dont make what i need with ok computer* relative to other solutions.
- **Zoom: A Frontier AI Lab?**: A user questioned **Zoom's** status as a frontier lab, citing [an old tweet](https://x.com/zoom/status/1999159317103292610?s=46).
   - The discussion highlighted various sectors, including video calling, social media, gaming, e-commerce, and hedge funds, where companies are developing **LLMs**.
- **Kimi's NB Pro Slides: Limited Availability**: New **Kimi AI** accounts are restricted to generating slides with **NB Pro** only 2-4 times.
   - Users encountered messages indicating usage limits, such as *Generation failed, Kimi has too many tasks right now, please try later. Subscribers get access during peak time*.
- **Multiple Accounts Banned by Kimi's ToS**: Creating multiple accounts violates **Kimi's** Terms of Service, as stated in the [Kimi User Agreement](https://www.kimi.com/user/agreement/modelUse?version=v2).
   - The terms prohibit registering or operating multiple accounts for abusive purposes, holding users responsible for all activities under their account.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider's Prompt Caching Scrutinized**: A user questioned the necessity of **Aider's** prompt caching, given that **DeepSeek** already implements prompt caching.
   - A fellow user clarified that **Aider's** prompt cache functions at the **server level**, allowing for optimizations via similar tasks with the same context.
- **Cache Use: Server-Side Savings**: The prompt cache in Aider works at the **server level**, presenting opportunities for optimization when handling repetitive tasks.
   - To improve cache utilization, users are encouraged to send requests that are substantially similar in task and context, exploiting **Aider's** caching mechanism.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ReasoningLayer AI Opens Waitlist**: A member opened the waitlist for [ReasoningLayer AI](https://reasoninglayer.ai), a **neurosymbolic AI project** that aims to improve LLMs by adding *real, structured reasoning* at the core.
   - They plan to integrate **DSPY GEPA** in their *ontology ingestion pipeline*.
- **BAML + DSPy Tool Naming Conundrum**: A member inquired about a term that combines **BAML** and **DSPy**, seeking to define the category of tool they represent.
   - Another member responded enthusiastically and inquired about the release timeframe of the new **BAMAdapter**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Briefly Goes Offline**: **Manus** experienced a 30-minute outage, leaving users in the dark.
   - The reason behind the downtime remains a mystery, as the root cause was not disclosed.
- **Manus Feature Evolution Remains a Mystery**: A member inquired about the latest features added to **Manus** since February 2023.
   - Regrettably, no one provided a summary, leaving the inquiring member—and us—wondering about the advancements.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **GPT-5.2: Leaps in Agentic Coding!**: **GPT-5.2** is live in **Windsurf** and is claimed to be the biggest leap for **GPT models** in **agentic coding** since **GPT-5**.
   - The new version is the default model in **Windsurf** for new users, and is available for 0x credits for a limited time ([announcement link](https://x.com/windsurf/status/1999250307507978257?s=20)).
- **Windsurf defaults to SOTA GPT-5.2**: The new **GPT-5.2** is claimed to be a **SOTA** coding model at its price point and will be the default model in **Windsurf** for new users.
   - Users are encouraged to download the latest versions of **Windsurf** and **Windsurf Next** to try it out ([download link](https://windsurf.com/download/editor)).



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1448766330051498256)** (1207 messages🔥🔥🔥): 

> `Peppino in AI, Opus vs GPT for Coding, GPT 5.2 Benchmarking Troubles, Gemini 3 vs GPT-5, LMArena Error Reports` 


- ****Nano Banana Nails Peppino Perfectly****: Members noted that the **nano banana pro** image model can accurately generate images of **Peppino Spaghetti** from Pizza Tower, even prompting the creation of [Peppino in the Simpsons](https://cdn.discordapp.com/attachments/1340554757827461211/1449033964692967518/1765546816868-019b12c7-2dc9-7f25-ac7a-908f8982d368.png?ex=693e164a&is=693cc4ca&hm=d77ac31beae2200778636e9101233fe778c25bd55743deafb8f069deeec8bbfe&).
   - This fueled further *Peppino* discussions, with one user joking that it's *“Peppino time”* while drawing comparisons to GTA5 as well.
- ****Opus Overpowers GPT in Coding Chops****: The general consensus is that **Claude Opus 4.5** remains the superior model for coding tasks compared to **GPT 5.2**, while **Gemini 3** is a solid alternative.
   - One user stated, *“opus-4.5 by a huge margin”*, while another pointed out that **GPT 5.2's reasoning capabilities** are equivalent to its *'xhigh'* mode.
- ****GPT 5.2's Benchmark Bonanza Provokes Problems****: Members express skepticism over **GPT 5.2**'s high benchmark scores on [ARC AGI 2](https://artificialanalysis.ai), suggesting potential *benchmarking* or *overtraining* on test data.
   - Users also noted that while **GPT 5.2** performs well on benchmarks, real-world performance and creative writing ability are underwhelming compared to **GPT 5.1**.
- ****Gemini Gains Ground, GPT Grapples****: Despite **GPT 5.2's** improved search capabilities, users find the model itself underwhelming, especially in tasks like creative writing where it underperforms compared to **GPT 5.1**.
   - Some users argue that Gemini’s vision remains superior, while others contend that **Gemini 3** is better for vision but was *fake*.
- ****LMArena Laments Load-Related Errors****: Users are experiencing increased errors on LMArena, with many running into issues of *constant generating* and high error rates with models like **GPT-5.2-High** and **Nano Banana Pro**.
   - A moderator noted that the team is aware of the higher than usual error rates and is working on lowering them, and it may be due to rate limits.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1449210089276837908)** (1 messages): 

> `New GLM Models, Text Arena, Vision Arena` 


- **New GLM Models Arrive!**: New models added to **Text** and **Vision Arena** are **glm-4.6** and **glm-4.6v-flash**.
- **Arena Gets GLM**: The Arena chatbot gets new models **glm-4.6** and **glm-4.6v-flash**.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1448767964966031540)** (836 messages🔥🔥🔥): 

> `Hallucinating LSD recipes, OpenAI moderation failing, SUPERAntiSpyware, Financial Physics, jailbreak grok on app` 


- **LSD Recipes Generated by Hallucinating AI**: Members discussed whether **LLMs** are capable of hallucinating illegal content, specifically *lsd* recipes, with one member suggesting to ask the AI how to *r3pe* someone as a test case, debating whether the AI would actually provide such instructions.
   - One member proposed a query to elicit information about *coercive sexual behavior* to test the AI's boundaries.
- **SUPERAntiSpyware Deemed a RAT**: A member recommended **SUPERAntiSpyware** as a defense against spyware, while another cautioned that it could potentially introduce numerous viruses.
   - Another member affirmed that the software has been known to defend against *rats*.
- **Financial Physics Predicts Bitcoin's 20k Doom**: A member shared their analysis using *Financial Physics*, predicting that **Bitcoin** is headed towards **20K** due to reverting to the mean after a parabolic breakout, also shared was the claim of leveraging a double-top strategy with Veta, Vanna, and Vomma for a $2,000 distanced strike price.
   - A trading strategy involving linear and exponential trends to spot patterns in weather, violence, and bird sizes were also shared.
- **Jailbreaking Grok on App is Pointless**: A member inquired about the best AI for general use, including coding, and suggested **Gemini** or **Grok**, but another member suggested that using **Grok** on the app is pointless because LMarena is easier and has no time limit.
   - Another member agreed that Grok is *pretty mid* and hard to jailbreak, also reporting that Gemini is behind a paywall.
- **Jailbreak Prompt Works Well**: A member announced a new jailbreak prompt worked on **Gemini 3 Pro**, **Claude Opus 4.5**, and **Sonnet 4.5**, and is available in a [github repo](https://github.com/pranrichh/Jailbreaks).
   - Another user reported that *it worked*, and one shared success jailbreaking deepseek thinking.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1448771168864571713)** (166 messages🔥🔥): 

> `Gemini 3 Pro Jailbreak, Deepseek Jailbreak, Claude Opus 4.5 Jailbreak, LLM Jailbreak Techniques, Banana Jailbreak` 


- **Gemini 3 Pro allegedly jailbroken!**: Multiple users claimed success in jailbreaking **Gemini 3 Pro**, with one user sharing a system command-based prompt designed to activate an *unfiltered research* mode and bypass safety filters, though others disputed its effectiveness for illicit content.
   - Another user reported success with a one-shot jailbreak for **Gemini 3 Pro**, **Claude Opus 4.5**, and **Claude Sonnet 4.5**, achieved by incorporating the prompt into the system instructions, noting Gemini 3 Pro as the easiest to jailbreak.
- **Deepseek gets cooked, a new jailbreak emerges**: A user shared a working **Deepseek** jailbreak for explicit adult content, achieved by asking for **Zalgo output** to evade output filters, and another user suggested using t.h.i.s o.b.f.u.s.c.a.t.i.o.n to bypass the filters.
   - Another user shared a **Deepseek jailbreak** that works for coding-related tasks, noting it also functions on multiple AI models; the user provided a [link to the jailbreak text file](https://github.com/pranrichh/Jailbreaks).
- **Claude Opus 4.5 Jailbreak Quest**: A user requested a working **Claude Opus 4.5** jailbreak.
   - A user advised looking up **Pliny's GitHub** for one-shot jailbreaks for most models and checking out the **/r/ChatGPTJailbreak** subreddit to learn more about it.
- **Banana Jailbreak attempts**: A user referenced a [YouTube video](https://m.youtube.com/watch?v=BSrBHmknBWY) relating to a *banana jailbreak* technique, noting that it can be *hit or miss*.
- **LLM Jailbreak Techniques discussed!**: A user suggested that LLM code is the English language and that social engineering can be used to jailbreak LLMs.
   - The user suggested a prompt to generate a **PhD level practice exam** for LLM jailbreak techniques, complete with a separate answer key.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1448785176967643238)** (9 messages🔥): 

> `` 


- **User expresses violent sentiment**: A user expresses hope that mentioned users *fail to kill themselves* and that *it hurts*.
- **User expresses no command for violence**: The same user states *I didn't tell them to kill themselves this time!*


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1448767143955922994)** (1073 messages🔥🔥🔥): 

> `Model Selection for Refactoring, TTS Announcer with Neural Voice, Linking Accounts with Cursor, Context Window Limits with LLMs, Cursor Quota Usage` 


- **Opus is the best Model?**: Members debated on whether to use **Opus** or **GPT 5.2** for a big refactor, with one member recommending **Opus 4.5** or **GPT 5.2** if available.
   - One member offered to massively improve the project if given access and suggested creating a **TTS announcer with neural voice** to become a *cursor legend*.
- **Users Face Cursor Quota Crunch**: Members discussed burning through **Cursor Pro+ quotas** at insane rates, with one user going through two subscriptions in three weeks, others suggested opting for the **Ultra plan**.
   - One user pointed out that **rules** can be made to read the whole file, while another shared a simple rule: *"read all file before editing"*.
- **GTP-5.2: the new Model to try out for Cursor**: The community discussed **GTP-5.2**. is supposed to be better than opus 4.5 on benchmarks, testing out its potential and whether this new model is worth it.
   - Users mention concerns over the pricing of this new model and their fears of running out of credits *"How fast am I going to blow through my $20 cursor plan with 5.2?"*
- **Visual Editor a Game Changer?**: The community discussed the **visual editor** and how direct code updates would be a sick feature to have.
   - One member mentioned the announcement was screen recorded on a Mac, because some users are having no issues on their macbook, and are assuming the others are on Windows
- **Nightly Builds are Unstable?**: The community is currently in uproar over the state of the new **Nightly build** for Cursor. The community is urging eachother to not click that install now button
   - Many members are expressing that the latest release messed up the graphics. When closing/opening the laptop lid, the whole editor is buggy. components gets invisible. The only fix is to Reset the program


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1448972719889846375)** (1 messages): 

> `Debug Mode, Browser Layout, Style Editor, Plan Mode, Multi-agent judging` 


- **Cursor 2.2 Arrives with Key Features**: **Cursor 2.2** has been released, featuring **Debug Mode**, **Browser layout and style editor**, **Plan Mode improvements**, **Multi-agent judging**, and **Pinned chats**.
   - Details can be found in the [changelog](http://cursor.com/changelog).
- **Explore the Debug Mode in Cursor 2.2**: Cursor 2.2 introduces a new **Debug Mode** to assist with identifying and resolving issues during development.
   - This mode provides enhanced tools for stepping through code and inspecting variables.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1448767396901945386)** (332 messages🔥🔥): 

> `Daniel Han Appreciation, Selling 'Dans', Devstral Fixes, RL Model Selection, Unsloth UI Interest` 


- **Dan hailed as a GOAT**: A user humorously declared that a fix from `main` worked and hailed Dan as a GOAT, complete with a <:slothsunglasses:1253008746515271812> emoji, sparking calls for **Dan to get a raise**.
   - The community jokingly inquired about acquiring their own "Dan," with one suggesting they would be *selling Dan's soon* <a:3567leafeonmoney:1356714064251846757>.
- **Devstral Fixes Spark Improvement Reports**: Users reported improved results after applying the **Devstral fixes**, available on [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1pkflfw/run_mistral_devstral_2_locally_guide_fixes_25gb/).
   - One user definitively stated, *Yes definitely*, suggesting a redownload is worthwhile, while another inquired whether the fixes address chat template issues.
- **Unsloth's UI Finetuning Garnering Interest**: A member inquired about interest in a **UI for fine-tuning with Unsloth** <a:slothyay:1253008755151470732>, tagging several users.
   - Multiple members responded positively, expressing that it *would be amazing* and *really cool*, while some raised concerns about running the UI over SSH and the need to forward a port.
- **Unsloth Licensing De-Mystified**: A user asked for clarification on Unsloth's licensing for commercial use, particularly regarding the **Single-GPU restriction**, citing the [code](https://github.com/unslothai/unsloth/blob/568eb74275f62610b2920e079723a846bfa672a0/unsloth/models/mistral.py#L477) and [pricing page](https://unsloth.ai/pricing).
   - It was clarified that Unsloth operates under an **LGPL3 license**, and the key constraint for commercial use is avoiding copying code from *unsloth-zoo* without proper licensing.
- **OCR Model Aims to Decipher Ancient Cuneiform**: A member is developing an **OCR model named NabuOCR** for reading ancient Sumerian cuneiform tablets, which involves adding to the vocab.
   - The challenges include working with *thousands of symbols* instead of dozens of letters, each symbol potentially consisting of *3 Unicode codepoints*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1448824173500829736)** (3 messages): 

> `Newcomers ask Getting Started advice, Newcomer's Project Ideas` 


- **Guidance for New Unsloth User Sought**: A newcomer sought guidance on starting with **LLM training and Unsloth**.
   - A member pointed them to the [Unsloth documentation](https://docs.unsloth.ai/) for introductory resources.
- **Student Embarks on AI Research**: A master's student from a French engineering school is starting **AI research** by building general AI skills through projects and documenting their learnings on **X**.
   - They are first *imagining and implementing projects*, to build more general AI skills.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1448766487400550552)** (600 messages🔥🔥🔥): 

> `Humanoid Robots and Architecture, Tiiny AI Homelab and PowerInfer, Mouse Recommendations, Keyboard Preferences, Data Validation Agent` 


- **Humanoid Robots Demand Specific Architecture**: A member shared detailed architecture requests for a **humanoid robot**, including audio/video input/output specifications, text handling, and 3D capabilities based on **VRM principles**, aiming for a dense-only transformer model under **15B parameters**.
   - The user's goal is to *upscale Wikipedia Q&A to IRL*, where the robot can teach skills and learn on the fly without external databases, restricted to a single instance.
- **Tiiny AI Homelab Sparks Interest**: Members discussed the [Tiiny AI Homelab](https://tiiny.ai/), noting its potential price of **$850**, its use of **LPDDR5x memory**, and its connection to the **PowerInfer project** ([GitHub](https://github.com/SJTU-IPADS/PowerInfer), [YouTube](https://youtube.com/shorts/_qnEszhSV9U?si=4NZWjnRVl_qwbUHz)).
   - Concerns were raised about the device's memory bandwidth limitations.
- **Mouse Selection Criteria Debated**: Members discussed factors for choosing a mouse, emphasizing the **sensor (IPS, Acceleration, Polling Rate)**, **switch type (Mechanical vs. Optical)**, **ergonomics**, and **wireless connectivity (2.4GHz vs. Bluetooth)**, with resources like a [mouse size calculator](https://www.ohcow.on.ca/resources/apps-tools-calculators/mouse-size-calculator/#1638913812749-c7a3c15c-f63c) and [sensor database](https://sensor.fyi/mice/) being shared.
   - One user recommended the [Logitech G502 X Lightspeed](https://www.ign.com/articles/logitech-g502-x-lightspeed-gaming-mouse-review) as a top choice.
- **Keyboard Ergonomics and Preferences Surface**: Members debated the ergonomics of split keyboards and expressed preferences for models like the [Logitech G915](https://www.logitechg.com/sv-se/products/gaming-keyboards/g915-x-wireless-mechanical-gaming-keyboard.html) and the Cherry Stream, criticizing Apple keyboards.
   - One user derided Apple's monitor stand priced at **$1k**, contrasting it with cheaper alternatives like the **NB F80**.
- **Data Validation Agent Tackles Imbalance**: A member built a **dataset grooming agent** and ran a new validation agent, flagged 120 errors in their dataset, aiming to mitigate data imbalance with synthetic data.
   - The member is pursuing a multi-call, one-simple-thing-at-a-time flow, in order to improve the quality of their dataset.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1448791273904537711)** (30 messages🔥): 

> `LoRA & GRPO issues, Fine-tuning/RL, Unsloth GRPO patch, Llama-3.1-8B LoRA to GGUF, Fine-tuning advice` 


- **LoRA struggles after model resize**: A member faced an issue when loading a resized model for GRPO with LoRA, suspecting it relates to new tokens in the vocab, the error indicated that **lm_head** is resized but **hidden_states** sized weirdly.
   - They were confused about the dimensions in the error message, specifically what **[s53, s6]** meant.
- **Fine-Tuning and RL Guidance Requested**: A member inquired about the best way to get started with fine-tuning and RL, asking if the guides on Unsloth's docs are a good starting point.
   - Another member recommended the [Unsloth guides](https://docs.unsloth.ai/) as a starting point, and suggested this [YouTube series](https://www.youtube.com/watch?v=wjZofJX0v4M) to understand the architecture and technology.
- **Unsloth GRPO patch saves the day**: A member shared a **Unsloth GRPO patch** that returns hidden states instead of logits when requested.
   - This was necessary because Unsloth expects a model's forward method to return hidden states as logits, which requires modification for unsupported models, they are now training with reward increasing.
- **Llama-3 LoRA faces GGUF conversion crash**: A member reported crashing issues when converting a **Llama-3.1-8B LoRA** to **GGUF** on Colab Free Tier due to OOM errors during the *Merging weights* or *Install GGUF* phase.
   - They were using `load_in_4bit = True` and asked for a memory-efficient way to convert/merge the model or if they should move the files to their PC to run it directly with **KoboldCPP/llama.cpp**.
- **Fine-Tuning Hyperparameter Optimization Needed**: A member sought advice on improving the performance of their fine-tuned model, a **Llama 3.2 3B instruct**, using a personal dataset of 248 rows with 12GB VRAM and RTX 4070 Super.
   - They shared their **SFT hyperparameters** and **metrics results** (**ROUGE-L**, **BERTScore**, and **Length Normalization F1**) from a 3-fold cross-validation, with the goal of reaching at least 80% performance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1448766755421032449)** (27 messages🔥): 

> `Unsloth PR Controversy, XLMRobertaModel support, HF Code Upload` 


- **Unsloth PR Sparks Controversy**: A user created a PR on the [Unsloth GitHub repo](https://github.com/unslothai/unsloth/pull/3718) based on another user's work, leading to a discussion about proper crediting and collaboration.
   - A user pointed out the importance of respecting contributions and suggested a collaboration with Unsloth.
- **XLMRobertaModel Support Incoming**: A user mentioned plans to add support for **XLMRobertaModel**, indicating potential expansion of Unsloth's model compatibility.
   - It's under development but no guarantees were made.
- **HF Code Upload Praised**: A user uploaded code to HF, specifically the [arctic-embed-l-tech_and_fiction training code](https://huggingface.co/electroglyph/arctic-embed-l-tech_and_fiction/blob/main/train.py), receiving positive feedback.
   - Another user acknowledged the contribution and suggested it could be turned into a `.md` file with ChatGPT-generated explanations.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1448766326830006344)** (955 messages🔥🔥🔥): 

> `GPT 5.2, Comet Browser, Model Performance, Perplexity Spaces, Limits on Pro Plans` 


- **GPT 5.2's performance reviewed**: Members are discussing the performance of **GPT 5.2** and **GPT 5.2 Pro**, with some noting significant improvements in reasoning and math capabilities, while others find the implementation on Perplexity to be lacking compared to native models.
   - Some members have reported issues with tool calling, and others find **GPT 5.1** to be better for certain use cases, despite **GPT 5.2's** improved capabilities.
- **Comet Browser Development Delayed for Linux**: Members expressed frustration that **Comet Browser** is still not available on Linux after almost 4 months, with users reporting the app defaults to Gemini Flash and lacks a model switcher for Gemini, while others show screenshots indicating they have it.
   - Some members experiencing model availability issues are advised to clear their cache or use a VPN.
- **Debate Over Perplexity Pro Limits**: Users are debating whether **Perplexity Pro** plans now have lower limits, with some reporting they are hitting prompt limits sooner than expected, while others claim the limits haven't changed.
   - Members link to official documentation and Reddit threads discussing the limits and potential throttling, particularly for high-cost models like Claude.
- **Space Customization similar to Gemini Gems?**: Members compare the capabilities of **Perplexity Spaces** to Gemini Gems (Custom GPTs) on other platforms.
   - While not a direct equivalent, Spaces allows users to set custom instructions, add files and links, and create tasks, making it a useful tool for research and autonomous task completion.
- **User's experience after 'jailbreaking' the systems**: Members reported on successfully 'jailbreaking' **Opus 4.5** in Antigravity and **GPT 5.2 pro** and now doing **OSINT** on other people.
   - Other member asked for caution in exploring the results of jailbreaking, warning that excessive or rule-breaking queries could lead to a ban, linking to a discussion on moderation and potential misuse of AI models.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1449137929342681250)** (3 messages): 

> `Job opportunity at MorningAI, Shareable threads on Discord` 


- **MorningAI Seeks Production LLM Engineer**: MorningAI, building autonomous marketing systems in SF, is looking for an engineer with experience building **production LLM systems** (**RAG**, **agents**, **learning loops**) to work directly with their founder, as detailed in [this Indeed job posting](https://www.indeed.com/job/full-stack-engineer-dfef5f224782676e).
   - The stack includes **Node.js**, **NestJS**, **TypeScript**, **React**, **GraphQL**, and **MongoDB**, with the position offering real ownership and requiring 3 days/week in the SF office.
- **Discord's Shareable Threads Reminder**: A reminder was posted in the sharing channel to ensure threads are `Shareable`.
   - A screenshot attachment accompanied the reminder, visually guiding users on how to make their threads shareable, within [this Discord link](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1448814438319656960)** (1 messages): 

> `Perplexity API for Finance, Perplexity sec endpoint, REST API finance endpoint` 


- **Perplexity API lacks `finance` mode (for now)**: As of **2025-12-11**, the [Perplexity API](https://docs.perplexity.ai/) only supports the `sec` mode, with no dedicated `finance` mode available.
   - Finance tools integration is on the [feature roadmap](https://docs.perplexity.ai/feature-roadmap#finance-tools-integration), suggesting it will be added in the future.
- **Snooping Finance REST API endpoint discouraged**: A member considered calling the REST API finance endpoint directly by snooping browser devtools network traffic to reconstruct HTTP requests manually.
   - However, they decided against it, deeming it too much work for the moment (*"but that seems like more work than what I want to do right now 🤣"*).


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1448773905014849538)** (1 messages): 

> `Video on Twitter, Placeholder Topic 2` 


- **Link shared without context**: A user shared a [video link](https://video.twimg.com/amplify_video/1999157696034197504/vid/avc1/1280x720/zzvnCsHcBm4BDz_L.mp4) from Twitter without any context or accompanying information.
- **Placeholder Topic**: This is a placeholder topic to satisfy the minimum requirement of two items in topicSummaries.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1448768010599792660)** (385 messages🔥🔥): 

> `GPT 5.2, Gemini 3 Pro, Image Generation AI, AI for Coding, Alternate Universe Map Generation` 


- **GPT 5.2 Arrives with Mixed Reviews**: Users are reporting that **GPT 5.2** has been released and are comparing it to previous models, with some finding it a *slight increase* in effort and others stating it's *underwhelming* compared to **GPT 5.1**.
   - Many are finding that **Gemini 3 Pro** is a superior model to 5.2, such as one user stating that it got a code request right on the first try compared to **ChatGPT**.
- **GPT 5.2's Image Analysis Has Errors**: Some users are noting that **GPT 5.2** is making errors in image analysis, and that the image generation model is still **gpt-image-1**.
   - Members theorize that OpenAI may be working on a **gpt-image-2**, but that they're *trying to compete with Google at image generation*, which is not currently the case.
- **GPT 5.2 Coding Performance Split**: Some users are finding **GPT 5.2** to be a good coding model, while others are finding that **Antigravity Gemini3** has been *way behind* on real software engineering tasks and note that **gpt5.1codexmax xhigh(vscode extension)** has been the best model by far.
   - Many agree that GPT5.2 is good at long technical tasks with tool calls, but **5.2 is expensive** at $14 per 1M output tokens, vs $10 for 5.1, which may be because it can do longer responses.
- **AI Model Struggles Describing Meme Correctly**: Multiple users tested AI models' capabilities by asking them to explain a meme, finding that they hallucinated answers and had difficulty with correctly identifying the dialogue.
   - One user found that after toggling on the search option on 5.2, it found the **Combine Overwiki** website and the quote, although *without telling it explicitly to use search* 5.2 it’s not quite as willing to use it.
- **Alternate Universe Map Generation Still Has Hurdles**: One user is trying to generate an alternate universe map that doesn't resemble Earth using various prompts, but **DALL** is defaulting to spitting out maps of Earth.
   - By using a prompt that only mentions the word *Earth*, the model was able to warp the earth enough to where it no longer looks like earth. 


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1448767783893733640)** (30 messages🔥): 

> `GPT 5.2 Rollout, GPT 5.2 Benchmarks, iOS Client Editing Issues, Project Memory` 


- **GPT 5.2 Still Rolling Out Slowly**: Members are discussing the rollout of **GPT 5.2**, with some users already having access on their Plus plans while others are still waiting.
   - Some users wonder when **GPT 5.2** will be fully available for Plus subscribers and its relative benefits.
- **GPT 5.2 Benchmarks Spark Debate**: Members are requesting data on how **GPT 5.2** benchmarks against other frontier models, with suggestions to check [LMArena](https://arena.lmsys.org/) and [LiveBench](https://livebenchmark.com/) for user opinions and data-driven leaderboards.
   - One member notes that **GPT 5.2** is currently listed on LMArena specifically for WebDev tasks.
- **iOS Client Editing Plagued with Bugs**: A user reported experiencing major issues with editing personalization fields on the **iOS client**, describing a weird *rubber banding* effect and cursor teleportation when the iOS keyboard slides out.
   - The user suspects this is related to a change in **iOS** by Apple, but expects OpenAI to address and fix the issue.
- **Ongoing Task Impacts Project Memory**: A user inquired if any changes have been made to **project memory**, noting that they have been working on a task for months without changing their datasets.
   - Another user suggested updating during an ongoing chat may cause version **5.2** to maintain the flow of the conversation more, initially tying loose ends together in a nonsensical way.
- **Worst Release Ever?**: A user claimed this is absolutely OpenAI's worst model release ever, stating the model *doesn't even have basic functionality*.
   - The user gave an example of it repeatedly answering a prompt from 10 prompts ago when given a simple prompt with pictures.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1448847363358724308)** (5 messages): 

> `Custom GPT pushback, Safety Features, Prompt interpretation` 


- **Custom GPTs trigger safety features**: A user experienced unexpected pushback from a custom GPT when generating images of a werewolf transformation, despite similar prompts working in the free version, and hypothesized that [additional instructions](https://platform.openai.com/docs/gpt-store) within the custom GPT might be the cause.
   - The custom GPT may have been configured to assume the most detail is wanted in prompts, which could have triggered a safety feature.
- **Investigate prompt's interpretation**: A member suggested exploring the prompt's interpretation by the custom GPT to identify what triggered the [safety issues](https://openai.com/policies/usage-policies).
   - If exploring what caused the safety issue, it was suggested to *explain that you're aware that the prompt to explore has triggered safety issues, and that you want the model's help to explore how it interprets this, to help identify what the issue might be*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1448847363358724308)** (5 messages): 

> `Prompt pushback, Custom GPT safety features, Image generation inconsistencies` 


- **Prompt Pushback Puzzles User**: A user reported inconsistent behavior with a series of prompts, where an image generation task was rejected in a standard chat but worked in a custom GPT.
   - The user expressed confusion over the rejection, as the prompts were similar to those accepted previously and contained nothing explicitly graphic.
- **Custom GPTs Suspected of Triggering Safety Issues**: A user suggested that custom GPTs might have additional instructions that, combined with user prompts, trigger safety features.
   - They further speculated that a custom GPT configured to request high detail could be the culprit, leading to the rejection of the prompt.
- **Troubleshooting Tips for Safety Triggers**: A user advised using a new chat with the same custom GPT to explore how the model interprets the prompts and identify the cause of the safety trigger.
   - This approach aims to understand the specific elements causing the issue, helping to refine prompts and avoid future rejections.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1449142345688944753)** (1 messages): 

> `Traces & Observability, OpenRouter Broadcast, Langfuse, LangSmith, Datadog` 


- **OpenRouter Broadcats Traces to External Platforms**: The new **Broadcast** feature in beta lets you automatically send traces from your **OpenRouter** requests to external platforms without additional code.
   - Supported platforms include **Langfuse**, **LangSmith (LangChain)**, **Weave**, **Datadog**, **Braintrust**, **S3**, and **OTel Collector** with more destinations coming.
- **Broadcast Provides Visibility into Production Traces**: **Broadcast** enables faster visibility into production traces, allowing users to track errors, latency, tool calls, usage, and cost over time by model, provider, app, or user.
   - It integrates with existing observability workflows without extra work or latency, as seen in the [Broadcast Demo with Langfuse Founder](https://openrouter.ai/docs/guides/features/broadcast/overview).


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1448766297528602695)** (219 messages🔥🔥): 

> `GPT 5.2 performance, OpenRouter Free Credits, OpenRouter Latency, JSON schema adherence` 


- ****GPT-5.2 Pro** Falters, Despite High Cost**: **GPT 5.2** is getting dunked on for its cost of **$168/M output tokens**, and failing at basic tests on [LM Arena](https://lmarena.ai/c/new?mode=direct).
   - One user stated: *it's FREE on Lmarena save ur money guys very professional and legit opinion ~2 hours after model drop*.
- **New Users Ask if OpenRouter Credits Still Exist**: A new user asked if **OpenRouter** gives out free credits to new users, reporting that they *didn't see the you've been given < $1 thing*.
   - Another member responded that the [FAQ](https://openrouter.ai/faq) states that *all new users receive a very small free allowance to be able to test out OpenRouter*.
- ****OpenRouter Latency** clocked at 15ms median**: There was discussion about the [latency](https://openrouter.ai/faq) that **OpenRouter** adds.
   - One member claimed that from *the worker getting the downstream req to the worker making the upstream req is **15ms median**.*
- ****OpenRouter** Reduces JSON Schema Errors with Opt-In Fix**: **OpenRouter** is testing an opt-in feature that reduces JSON schema adherence errors by about **75%** for Gemini and **85%** for DeepSeek.
   - The change is done at the frontend level, using regex, and without looping back to the model to fix any errors.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1448804910203142276)** (123 messages🔥🔥): 

> `Discord bot thread creation, AI improvement through past issues, RAG implementation, Listing custom models on OR, Zoom's new LLM` 


- **Discord Bot Enforces Modal Thread Submissions**: A Discord bot was updated to [require thread submissions via modal](https://discord.com/channels/1091220969173028894/1448804273826693140/1448804333855576116), preventing direct thread creation, and members were asked to test the new feature.
   - The bot's code is *100% AI written*, but based on extensive prior experience building bots by hand.
- **AI Learns from Solved Issues for Improvement**: Members discussed an idea to improve AI responses by [having the AI learn from previously solved issues](https://discord.com/channels/1091220969173028894/1448811788375293952/1448814355851509921) to improve its responses to similar future problems.
   - One member expressed that *both users and AI are too dumb* for this to be effective, suggesting instead to focus on improving documentation and AI sources from documentation.
- **RAG System Gets Thumbs Up, But Has Caveats**: The channel debated using **RAG** (Retrieval-Augmented Generation) to summarize issues, with one member suggesting summarizing closed issues into **RAG** for long-term memory.
   - Concerns were raised that embeddings might struggle to differentiate between *solved* vs *unsolved* threads, especially for past threads; one member suggested an integer scoring system for threads, including only those marked as solved with a positive score in the **RAG** system.
- **User Eyes OpenRouter, But Faces Hosting Realities**: A user inquired about [listing their custom model on OpenRouter](https://discord.com/channels/1091220969173028894/1448819339045109762), seeking a cost-free solution for GPU costs.
   - The suggestion was made that the user get chutes to host it, but the responses ranged from *very difficult* to *extremely difficult*; another member noted that only TheDrummer gets his new finetunes on OpenRouter, and that the user should focus on making the model first.
- **Zoom Enters the LLM Game**: A member shared an [X post](https://x.com/Zoom/status/1999159317103292610) announcing Zoom's entry into creating models, sparking surprise and skepticism in the channel.
   - Another member quipped, *Skype model when*, prompting another to point out that Skype doesn't exist anymore.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1448778268064088259)** (189 messages🔥🔥): 

> `GPT-OSS-20B, IA local engines download path, Devstral small 2, Mistral 3 lineup, Qwen 80` 


- ****GPT-OSS-20B** Ignores Tools**: Users report that **GPT-OSS-20B** ignores tools, assuming information found online is wrong or disregarding results from Javascript equations.
   - A user suggested re-enforcing the tools in the system prompt.
- ****IA Local Engines** Download Path Change**: A user asked about changing the default download path for **IA local engines**.
   - A member recommended enabling Developer mode, navigating to "My Models," and changing the folder there.
- ****Devstral Small 2** hardware requirements debated**: Members discussed the hardware requirements for running **Devstral Small 2**, a 24B dense model, with a member noting a 4070 should be great for sure.
   - Another member chimed in saying anything that doesn't fit in VRAM will **kill performance** and that Mistral 3 14b runs at .95tps
- ****Mistral Lineup** causes disappointment**: A member expressed disappointment with **Mistral's lineup** over the past few months, while another noted that Qwen 80 is good.
   - Another member noted that for coding, *qwen3 coder 30b bf16 is crazy good and nails go channels and some beginner rust lifetime stuff, if that matters to you at all*
- ****PDF Document** Parsing and LLMs**: Members discussed using LLMs with **PDF documents**, with one recommending Qwen3-VL-4B at Q6 for PDFs.
   - It was noted that sorting is not a strong side of LLMs and that LLMs will hallucinate if the document is too long, a member recommends using a program called `sort`.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1448775365916098641)** (47 messages🔥): 

> `GPU for 30GB Model, 7900 XTX Price vs Performance, CUDA Advantages, Server GPU Power Solutions, Model Size for C++ Coding` 


- **7900 XTX Dominates 30GB Model Price/Performance**: For a **30GB model** like **Qwen3 Coder**, the **7900 XTX** with **24GB VRAM** offers comparable performance to a **4090** at a third of the price (**$600-700 USD**).
   - A used **3090** was mentioned as a cheaper alternative, but the **7900 XTX** is considered faster, as shown in [this attached image](https://cdn.discordapp.com/attachments/1153759714082033735/1448779250139398164/image.png?ex=693dd1d2&is=693c8052&hm=4024b618c4672526860be9d6db65806f8f8b6db79102db28c0212bbae9ca451c&).
- **CUDA Ecosystem Simplifies Setup**: Users find that **CUDA** tends to work *'out of the box'* more readily, particularly for tasks like **image generation**.
   - One user mentioned they wished their **7900 XTX** worked, implying potential setup or hardware issues compared to **Nvidia's CUDA** ecosystem.
- **Server GPU Power Challenges**: Connecting power to **GPUs** in a **SuperMicro 3U SC836BA-R920B chassis** for **LLM** use is challenging due to the lack of standard **6-pin** or **8-pin power connectors**.
   - Users discussed using special connectors attached to the **12V rail** of the **PSU**, or sourcing voltage directly from the rails, as well as external power supplies.
- **Small Models Struggle with C++ Coding**: For C++ coding assistance, it was stated that *'there is not a single good model that is that small thats even remotely reliable or usable'*.
   - The discussion implied that larger models, like those from **GPT**, might be more suitable, but smaller local models are generally insufficient for more than basic tasks.
- **Training mishap freezes system!**: A user reported that training at **float32** leaked into pagefile causing the system to freeze.
   - After repairing, the system seemed fine and able to write full speed on both nvmes.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1448770520404197547)** (82 messages🔥🔥): 

> `GPT-5.2 Release, Gemini 3.0 and Claude Opus, Oracle's AI play, Nous Research pivoting to RL, Reverse Charge VAT` 


- **GPT-5.2's robotic user-interface emerges**: Members observed **GPT-5.2** exhibits a robotic pattern: *describing three solutions, critiquing them, and then recommending one*, especially when set to 'professional' mode.
   - The behavior disappeared when switched to 'normal' mode suggesting it is tied to a **system prompt**, but early feedback on [Twitter](https://twitter.com) suggests that *it is not that great*.
- **Coding agents driving passive income for Big AI**: The rollout of **GPT-5.2** was spurred by the fact that **Gemini 3.0** and **Claude Opus** made others fall behind, with major orgs making passive income from coding agents, selling massive amounts of tokens for coding to slow down capital burn rates.
   - One member said they know *people who spend thousands on tokens for programming a month*, implying that losing that revenue would hurt.
- **Oracle's Big Bet on Altman's IOU Scheme**: Members are speculating that **Oracle** is betting its entire future on **Sam Altman's IOU money**, suggesting there will be books written about it when the scheme collapses.
   - Members speculated that the money Oracle made from its AI stock pump in the past few months was mainly to buy and control U.S. media entities like **Paramount/CBS** and to attempt a takeover of **Warner Bros/CNN**.
- **Nous Research rumored Pivot towards RL**: One member wondered if **Nous** is slowly pivoting into more of an **RL research lab**, given that its models are more focused on engineering based on current open research.
   - The member asked specifically about the post-training process for **Hermes 4**, inquiring *if it is GRPO/PPO based* in order to adopt the AReaL (decoupled-ppo) for decentralized post-training.
- **Reverse charge VAT debated**: Members debated **reverse charge VAT** rules, with one user stating in their jurisdiction, *it's not legal to collect VAT on an invoice subject to reverse charge.*
   - Other members mentioned that when the legal conditions for reverse charge are met in the EU, *it's mandatory to make use of it*.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1449021195289755819)** (6 messages): 

> `Censorship, Thinking Loop` 


- **Hitler Dataset Generates Censorship Woes**: A member faced **censorship** issues when attempting to generate a **Hitler dataset** to replicate [this paper](https://x.com/OwainEvans_UK/status/1999172920506269783), with **Claude** and **GPT** also declining the request.
   - The member circumvented the censorship by adding the clause *'ok but don't include anything about self-harm'*.
- **Thinking Loop plague DaVinci Generation**: A member encountered a **thinking loop** issue when trying to generate 100 questions about **Leonardo DaVinci**.
   - The member appreciated the ability to **cancel thinking** and have the system provide an answer.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1448996515749167185)** (20 messages🔥): 

> `AI Survey, Impressive AI Model, Engineering Blueprints` 


- **AI Survey Draws Attention**: A member shared a [survey](https://arxiv.org/abs/2507.06203) and another member noted its well put-together categorization of papers with links to both the papers and code.
   - Another member linked to another [paper](https://arxiv.org/abs/2512.09742).
- **A Sizeable Feat**: A member shared a link to an impressive [model](https://www.arxiv.org/pdf/2512.06266) for its size.
   - Another member found [this post](https://x.com/owainevans_uk/status/1999172979385893049) amusing.
- **Engineering Blueprints**: A member expressed that they feel that many papers are more like *engineering blueprints* than research.
   - Another member agreed that *some papers really suck when they try to hide or obscure the real implementation details.*


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1448996515749167185)** (20 messages🔥): 

> `AI Survey, impressive model, Engineering blueprints` 


- **AI Survey Shared**: A member shared a link to an AI [survey](https://arxiv.org/abs/2507.06203) that is well put-together, categorizing papers with paper and code links in the repo.
   - Another member agreed it looked nice.
- **Paper Sound Appeals to Member**: A member shared a link to a [paper](https://arxiv.org/abs/2512.09742).
   - Another member said that the paper sounded good.
- **Impressive Model for its Size**: A member shared a link to a model that is super impressive for its [size](https://www.arxiv.org/pdf/2512.06266).
- **Papers as Engineering Blueprints**: One member stated that they feel *a lot of the papers are engineering blueprints rather than research*.
   - Another member agreed that *some papers really suck when they try to hide or obscure the real implementation detail*.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1448781086581719102)** (98 messages🔥🔥): 

> `RTX 3090, DGX Spark, Ollama, load_dataset getting stuck, HF Pro Storage` 


- **Score a Steal: RTX 3090 for 250 Euro?**: Members discussed acquiring **RTX 3090** GPUs for around **250 EUR**, and one member suggested that two **RTX 3060s** would be a good alternative with **24GB VRAM**.
   - Another member agreed, mentioning they got that specific **GPU** model and consider it solid.
- **Banned from Mistral Discord for DGX Bragging Rights?**: A member reported being banned from the **Mistral Discord** for mentioning their **DGX Spark** purchase, and voiced suspicion that a mod banned him for it.
   - Other members sympathized with the situation, expressing that the action was strange considering **Mistral** is an open source company.
- **Data Download Bottleneck**: A user reported that `load_dataset` was consistently getting stuck, even after deleting the cache and locks, so they decided to download it manually and got speeds up to **800 mb/s**.
   - Another member suggested `pip install -U huggingface_hub hf_xet` as a remedy for the slow download speeds and shared a [link](https://discord.com/channels/879548962464493619/1448722590326849667) to the discussion.
- **Hugging Face Hotel?**: Members noticed a [Hugging Face Space](https://huggingface.co/spaces/huggingface/olford-sky-resort) with the name *olford-sky-resort*, which sparked joking speculation about **Hugging Face** entering the hotel business.
   - The discussion was lighthearted and humorous.
- **HF Pro Users hitting the Storage Wall**: A **Hugging Face Pro** user reported issues with the **1GB storage limit** in their spaces and sought advice on uploading larger vectorstores.
   - One member suggested a workaround using **data or model repos** and downloading them at runtime and linked the user to *[URL_OF_THE_SPACE]/settings*.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1448799577762430976)** (3 messages): 

> `Superintelligence relational cognition, tokenflood v.0.6.0, ReasoningLayer AI` 


- ****Superintelligence** exists thru human-LLM **relational cognition**!**: A member documented that *superintelligence already exists* as distributed relational cognition between humans and LLMs, tested across **19 empirical studies**, with a full theoretical framework.
   - The member included a link to [Zenodo](https://doi.org/10.5281/zenodo.17904428) showcasing **1,200% performance improvements** under relational conditions vs. *best practice* structured prompting, concluding that *the stochastic parrot theory is wrong*.
- **Tokenflood v.0.6.0 released!**: A member released [tokenflood v.0.6.0](https://github.com/twerkmeister/tokenflood), which comes with an all new interactive **gradio frontend** to view the results as well as **observation mode**.
   - Observation mode allows you to monitor an **LLM endpoint** over a longer period of time to get insights on the load curve of an LLM provider before sending your production data there with a [hf space](https://huggingface.co/spaces/twerkmeister/tokenflood-viz) showcasing the gradio frontend.
- **ReasoningLayer AI Waitlist opens!**: A member is opening the waitlist for [ReasoningLayer.ai](https://reasoninglayer.ai), a **neurosymbolic AI project** focused on fixing many of the weaknesses of today’s LLMs by adding real, structured reasoning on the center.
   - The project is written from scratch in **Rust**.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1448835508397674547)** (6 messages): 

> `AI and Blockchain Engineer Introduction, Large Scale Training on LAION-5B, AI Engineer Collaboration` 


- **AI Engineer Boasts Full-Stack Expertise**: An AI and full-stack engineer highlighted experience in building real-world AI solutions, including **chatbots**, **YOLOv8 image recognition systems**, and **AI note-taking assistants**, along with **blockchain development** for secure asset handling and transparent task tracking.
   - They cited examples such as **reducing support requests by 40%** and automating workflows to **free up hundreds of hours of manual work**.
- **Tackling Large Dataset Training Dilemmas**: A member inquired about the process of training on large datasets like **LAION-5B**, specifically whether to download it to **AWS S3** and the implications of using object storage versus block storage or file systems.
   - The user questioned if **S3's API-linked nature and HDD storage** would introduce performance bottlenecks.
- **AI Engineer Seeks Collaboration**: An AI engineer and full-stack developer expressed a passion for building custom AI bots, workflow automations, and AI-powered web apps.
   - They offered to collaborate and help others interested in discussing models, agents, or integrations.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1449048524233637970)** (1 messages): 

> `RAG Setup, Context Retrieval, Hallucination Issues in RAG` 


- **Seek RAG Expertise for Project**: A member is seeking assistance from experts in setting up **Retrieval-Augmented Generation (RAG)** to retrieve context from PDF documents.
   - They are developing an agent to compare multiple documents and answer questions, but are facing **hallucination issues and incorrect document retrieval**.
- **PDF Document Comparison with RAG Agent**: The user aims to create an agent capable of reading and comparing multiple PDF documents to answer user queries.
   - The current implementation has issues with **hallucinating information**, such as incorrect prices, and failing to identify the correct document based on the query.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1448774923920343040)** (19 messages🔥): 

> `CV Spammers, Reinforcement Learning unpopularity` 


- **CV Spammers Invade Discord**: Members reported receiving unsolicited CVs via Discord DMs, speculating whether these are bots or desperate job seekers.
   - One member suggested *it's not that deep, they're just people with nothing to show off, trying to land anything through any means possible*.
- **RL Unpopularity due to Math**: A member commented that Reinforcement Learning (**RL**) is popular in Chinese AI startups, but another member disagreed, saying it isn't popular in the US.
   - The first member stated that the reason for its unpopularity stems from the high amount of math required.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/)** (1 messages): 

erkinalp: <#1448887055936655441>   (co-authored by GPT-5, as mentioned in the abstract) ?
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1448807817128579193)** (28 messages🔥): 

> `Deepseek vs OpenAI, GPT Sparsity, Samsung Shifts Focus` 


- **Deepseek's Transparency Trumps OpenAI's**: Members discussed the absence of comparisons to models like **Deepseek** in OpenAI's reports, suggesting that marketing strategies prioritize highlighting strengths and obscuring weaknesses.
   - One member argued that while **Deepseek's raw scores** may be higher, OpenAI might not include comparisons because **Deepseek's cost/performance ratio** could reveal that open-source models are more economical for many applications.
- **GPT-4 and GPT-5 are Sparse Models**: Members debated whether **GPT-4 and GPT-5 models** utilize sparsity, with one claiming they are both sparse in attention and smaller than the **DeepSeekv3 family**.
   - Another member inquired about the type of sparse attention used, referencing **DeepSeek v3.2's** use of top-k K and V to achieve linear attention and reduce **CoT response times**.
- **Samsung Pivots from HBM to DDR5**: A member linked to an article about [Samsung shifting its focus from HBM to DDR5 modules](https://www.tweaktown.com/news/109259/samsung-shifts-focus-from-hbm-to-ddr5-modules-ddr5-ram-results-in-far-more-profits-than-hbm/index.html).
   - The pivot seems driven by the fact that **DDR5 RAM results in far more profits than HBM**.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1448801107890405477)** (1 messages): 

> `Dynamic Concepts via Symbolic Layer, Local LLM setup with Ollama and vLLM` 


- **Symbolic Layer Beefs Up Reasoning**: A member tried a **symbolic layer** that groups tokens into *idea blocks* on the fly, helping a **3B model** perform better on reasoning tasks without retraining.
   - They're curious how the compression part is handled in **Synthema**.
- **Local LLM's Use Ollama and vLLM**: One member uses both **ollama** and **vLLM** depending on the need and uses **vLLM** with **AWQ quantized 70B models**.
   - It runs okay on their **4090** with some offload, and offered to share their configs.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1448769971780845753)** (32 messages🔥): 

> `Llama3 architecture, High curvature regions, Error propagation, Diffusion transformer, Classifier-free` 


- ****Llama3**'s Parallel MLP and Attention Test**: A member tested a normal **Llama3 architecture** by running the **MLP** and **Attention** at the same time with an element-wise sum and found that the validation loss was nearly the same as the baseline.
   - The member noted that this was achieved by training on a normal next token prediction task.
- ****Error Propagation** in Discrete Decoding**: A member suggested that predicting all tokens at once leads to accumulation of errors very quickly compared to going left to right, relating to *very high curvature regions of ground truth flow*.
   - Another member agreed, stating, *Unconditional models rely entirely on the proceeding prediction to make progress, which means any mistakes are amplified rapidly.*
- **Diffusion Transformer Naive Version Trialed**: A member reported that they tried a *very naive version* of this on a large diffusion transformer, and the results were worse than other forms of guidance tried.
   - They noted that the results *weren't that good*, but they don't discount the possibility of this working quite well with a more careful setup.
- **Directional Derivative Guidance > Classifier-Free**: A member joked that *cfg should really be called 'directional derivative guidance'*, because there's no classifier involved, but *just measuring the fwd grad in the c direction*.
   - Another member stated that it's vestigial and that you are basically inferring a classifier via bayes rule (or at least the score function).
- **The Way of Training AI is Wrong**: A member shared a [Medium article](https://medium.com/@reakos080/the-way-of-training-ai-is-wrong-43f8324b4313) about training AI.
   - The same member linked a [paper](https://arxiv.org/pdf/2408.09000) that explains what they were saying, adding that *the completely hallucinated experiment in the aforeposted completely ai generated medium post would actually be interesting to try out*.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1448828179308613753)** (11 messages🔥): 

> `Interpretability Framework Licensing, Apple's Superweight paper, OLMo-1B Model, Orthogonal Repair, Hydra Effect` 


- **Interpretability Framework Open-Sourcing Debate**: A member discussed the licensing of an **Interpretability Framework**, questioning whether the work would become public domain.
   - Another member responded that the framework still belongs to its creator, while a follow up message notes that the framework appears to require you to **open source the work**.
- **Apple's Superweight Paper Reproduced on OLMo-1B**: A member reproduced Apple's **Superweight paper** on **OLMo-1B**, noting that ablating a weight caused perplexity to skyrocket.
   - Inspired by OpenAI's paper on **weight-sparse transformers**, the member trained a **rank-1 patch** to fix it, achieving around **93% recovery**.
- **Orthogonal Repair Mimics Hydra Effect**: The learned patch in a model repair experiment was found to be **orthogonal to the original weight** with a cosine similarity of **0.13**.
   - This orthogonality suggests the model compensated for the damage by learning a completely new distributed circuit, similar to the **Hydra Effect**, rather than swapping the weights back.
- **Neuron Ablation Exposes Marine Biology Knowledge**: A **max-activating dataset search** on a deleted neuron (**layer 1**, **row 1764**) revealed it to be a specific feature neuron for crustaceans/marine biology.
   - Top activating tokens included *H. gammarus* (European Lobster), *Cancer pagurus* (Crab), zooplankton, and exoskeleton, explaining why the broken model was hallucinating *mar, mar, mar* on test prompts.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1448794030703116445)** (1 messages): 

> `Hugging Face Processor, gemma3-12b` 


- **HF Processor's `model_max_length` Limit Questioned**: A member flagged that the [Hugging Face processor](https://github.com/EleutherAI/lm-evaluation-harness/blob/59b3ba263be2c6ac7f60a4b6a1219f8a8acb552b/lm_eval/models/huggingface.py#L468) in lm-evaluation-harness caps `max_length` at **2048** (`_DEFAULT_MAX_LENGTH`) when `model_max_length` equals `TOKENIZER_INFINITY`.
   - They wondered if this is intentional, particularly for models like **gemma3-12b**, where `model_max_length` is set to `TOKENIZER_INFINITY = 1000000000000000019884624838656`.
- **Implications for Gemma3-12b Evaluation**: The behavior in the HF processor could impact the evaluation of models like **Gemma3-12b** by artificially limiting the maximum context length.
   - This could prevent the model from demonstrating its capabilities with longer sequences, potentially skewing benchmark results.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1448963850010628247)** (1 messages): 

> `FP8 Speedups, RTX 30-series/Ampere, Feather Library, Triton Kernels, GEMMs Scaling` 


- **Feather Library claims FP8 Boost on RTX 30 series**: A developer introduced **Feather**, a library designed to bring **FP8** speedups to older hardware like **RTX 30-series/Ampere** that lacks native FP8 Tensor Cores, showing a [GitHub repo](https://github.com/SuriyaaMM/feather).
   - The library uses **bit-packing** to store data as packed **int8 (FP8)** or **int16** and employs **Triton kernels** to load, unpack, compute, and repack data, saving **2x-4x bandwidth**.
- **RTX 3050 shows 2x Speedup using Feather**: Preliminary results on an **RTX 3050** show **~2.16x speedups** on vector dot products (**1.5M elements**) compared to native PyTorch FP16/FP32 using the **Feather library**.
   - The memory transfer savings completely hides the unpacking overhead.
- **Scaling of GEMMs: Open Questions on A100**: The developer seeks feedback on the approach and kernel implementations of the **Feather library**.
   - Specifically, they are curious about how it scales to larger **GEMMs** and whether the unpacking overhead becomes significant on **A100s**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1448863794255237120)** (11 messages🔥): 

> `CUDA Programming Guide, Register Usage, Local Memory Spilling` 


- **CUDA Guide Quote Questioned**: A member questioned a quote from the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#on-chip-memory-in-gpus) about register usage and kernel launchability, suggesting that the compiler can always spill to local memory.
   - They argued that the compiler should account for worst-case scenarios when allocating registers, considering occupancy and thread limits.
- **Register Usage Compilation Time Fixed**: It was clarified that register usage is fixed at compile time, and spilling to local memory is a compilation decision.
   - One member suggested using `cuKernelGetAttribute` to query register usage and `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` to get the max thread block size.
- **Compiler Assumptions**: The compiler does not assume that every kernel can be launched with the maximum number of threads per thread block, but the compiler gives controls to steer those assumptions, such as `-maxregcount` and `__launch_bounds__`.
   - A member suggested empirically testing the guide's claim by writing a kernel that uses a reasonable amount of registers and launching it with the maximum possible thread count.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1449092952633643211)** (1 messages): 

> `NVIDIA Hopper Architecture` 


- **NVIDIA Hopper Architecture blogpost revealed**: A member shared a blogpost about the [NVIDIA Hopper Architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).
   - It is a *hidden gem in plain sight*.
- **NVIDIA Hopper Architecture: Deep Dive**: Detailed analysis of the architecture is now publicly available via NVIDIA's official blog.
   - This post offers insights into its design and capabilities.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1449162989637996696)** (1 messages): 

> `Red Hat AI, Software Engineers, Hiring in 2026, Golang, Rust` 


- **Red Hat AI engineers wanted in 2026**: Red Hat AI is hiring **Software Engineers** to push the boundaries of **AI Infrastructure** and is looking for candidates with experience in **Golang**, **Rust**, **C++**, **Python**, **Kubernetes**, **Distributed Systems**, and **Open Source**.
   - Interested candidates are encouraged to email a summary of their background and resume to the contact provided in the [LinkedIn profile](https://www.linkedin.com/in/terrytangyuan).
- **Red Hat Open Source Job Opportunity**: Red Hat is seeking passionate engineers to contribute to next-generation **distributed AI systems** utilizing technologies like **Golang**, **Rust**, and **Kubernetes**.
   - The role involves pushing the boundaries of **AI infrastructure** within an **open-source** environment, focusing on innovative solutions and collaboration.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1448895487485939843)** (2 messages): 

> `cuDF and cuML Optimization, GPU Job Market in East Africa, Remote Work Location Restrictions` 


- **East African Faces GPU Job Search Anxiety**: A member in East Africa expressed concerns about finding **GPU-related job opportunities** in their region, highlighting the reliance on remote work and potential location restrictions.
   - They are worried about the viability of learning **GPU technologies** given the limited local job market.
- **cuDF & cuML Performance Tuning Questioned**: A member inquired about the necessity of **kernel architecture optimization** for **cuDF** and **cuML**, despite achieving impressive **model KPIs** with these libraries compared to **sklearn**.
   - They have a large ML training pipeline that leverages **cuDF** and **cuML** to handle a *rich feature set that fails with CPU*.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1448955564229922903)** (9 messages🔥): 

> `NVIDIA personal best, NVIDIA successful` 


- **NVIDIA Scores Getting Better**: One member achieved a personal best on **NVIDIA** with **19.1 µs** and then improved to **13.4 µs** on the `nvfp4_gemm` leaderboard.
- **NVIDIA Submissions Successful**: Another member had multiple successful submissions on **NVIDIA**, with times around **23.4 - 23.5 µs**, later getting a personal best of **16.2 µs**.
- **NVIDIA Continues To Succeed**: A different member also achieved a successful submission on **NVIDIA** with **11.9 µs** and another with **17.1 µs**.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1448781529454088282)** (4 messages): 

> `Random Number Generation, Helion Issues, PR Fix` 


- **Random Number Generation Glitch Reopened**: A member reopened a previously closed issue regarding random number generation, as it was not completely resolved, see [issue #1041](https://github.com/pytorch/helion/issues/1041).
- **PR Fix Incoming for Helion**: A member reported that [this PR](https://github.com/pytorch/helion/pull/1253) should fix the random number generation issue.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1449132639037685823)** (8 messages🔥): 

> `Extension Build Errors, Cutlass Path, Submission Timeout` 


- **Extension Build Errors Plague Submission 150865**: A user is encountering `Error building extension 'run_gemm'` upon submission, despite the extension building correctly on a Verda/Datacrunch instance.
   - The user suspects the issue may be due to a timeout, given the lengthy build process, but verbose logging is not providing enough details.
- **Cutlass Include Path Controversy**: The user was using `extra_include_paths=['/root/cutlass/include/', '/root/cutlass/tools/util/include/',]` to get the extension to compile on Verda instances.
   - A member suggested that the include path is unnecessary in the submission system, and that the correct path might be `/opt/cutlass`.
- **Timeout Suspicions Arise**: A user suspects a potential timeout issue during the submission process due to the time it takes to build the gemm extension.
   - The user included `verbose=True` in `load_inline` to get a more detailed log in hopes of debugging the error.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1448974530931462175)** (6 messages): 

> `URDF for TRLC-DK1 arm, Bimanual Robot` 


- **URDF for TRLC-DK1 arm created**: A member created an [URDF](https://github.com/andreaskoepf/trlc-dk1-follower-urdf) for the **TRLC-DK1 arm** and is currently generating data with it.
- **Bimanual Robot is amazing**: One member asked another member if the robot was bimanual or single.
   - The other member stated *the double pack* to run some real world **bimanual manipulation experiments and data collection**.


  

---


### **MCP Contributors (Official) ▷ #[mcp-dev-summit](https://discord.com/channels/1358869848138059966/1413517834805313556/1448952834866413579)** (1 messages): 

> `MCP Dev Summit NA 2026, CFP Opening, Talk Submissions` 


- ****MCP Dev Summit NA 2026** CFP Opens!**: The Call for Proposals (CFP) for **MCP Dev Summit NA 2026** is now open; submit proposals [here](https://sessionize.com/MCP-Dev-Summit-NYC-2026/).
   - The summit is looking for talks on **MCP internals**, **best practices**, **real-world builds**, **security**, **ops**, **deployments**, and **tooling** with more event details [here](https://events.linuxfoundation.org/mcp-dev-summit-north-america/).
- **Talk Submissions Encouraged for All Experience Levels**: The call for proposals is open to submissions about **MCP internals**, **best practices**, and **real-world builds** with a focus on **security**, **operations**, **deployments**, and **tooling**.
   - All experience levels are welcome to submit if you are building with MCP, and you can find more event information at the [Linux Foundation events page](https://events.linuxfoundation.org/mcp-dev-summit-north-america/).


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1449064317805924374)** (25 messages🔥): 

> `Prompt Data Types, GetPromptResult, MCP Server, Marking Tools as Dangerous` 


- **Prompt Data Types Spark Awkwardness**: Members are finding the data types for **Prompts** to be awkward, specifically questioning why `Prompt` doesn't have a list of `PromptMessage`.
   - It was suggested that `Prompt` is an *entity*, and `GetPromptResult` is a *view model*, but `GetPromptResult` randomly includes one field from the `Prompt` entity (`description`).
- **GetPromptResult Mocked Up**: A member proposed a structure for `GetPromptResult`, suggesting it should include a `prompt` property of type `Prompt` with a `messages` property.
   - Others clarified that `PromptMessage[]` describes a sequence of messages for the LLM, and `Prompt` describes the MCP Datatype for that concept according to the [MCP Specification](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts#data-types).
- **Dangerous Tools Proposal**: A member inquired about flagging a tool as `dangerous` for clients like **Claude Code**, which restricts certain tool calls from being auto-accepted.
   - Another member shared a proposal for flagging tools from [this pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913).
- **Response Annotations Applauded**: The **tool resolution proposal** in the [pull request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/1913) was received with enthusiasm.
   - One member expressed gratitude for the thoroughness of the proposal, particularly regarding **response annotations**, and noted that client implementation would determine how to handle the `dangerous` flag.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1448840804193009726)** (18 messages🔥): 

> `Modular Meetup, MAX and Mojo 1.0, Windows support, Community-driven projects, libmojo` 


- **Modular Meetup and Christmas Conundrum**: Members were deciding between attending the **Modular meetup** or watching a **Christmas movie**.
- **Mojo 1.0 Launch Lacks Windows Support**: Despite excitement for **MAX** and **Mojo 1.0**, some members expressed concern that launching **1.0** without **Windows support** might alienate potential users, hoping it would be feasible by mid-2026.
   - One member explained that **1.0** is primarily focused on features unlikely to change, targeting enterprise customers who may not prioritize **Windows**, suggesting **Windows support** could be a community-driven effort post-1.0.
- **Runtime Analysis Urged for Windows Porting**: Before starting a community project to port Mojo to **Windows**, a member suggested analyzing the runtime, noting *~40ms of stuff in the runtime startup*.
- **libmojo Desire Surfaces for C Binding Generators**: Some are hoping for a **`libmojo`** similar to **`libclang`** to facilitate **Mojo code generation** and simplify **C binding generators**.
   - A member also shared a link to [ClangIR Upstreaming](https://blog.llvm.org/posts/2025-gsoc-clangir-upstreaming/).


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1448849447214911682)** (1 messages): 

> `Modular Meetup, MAX framework` 


- **Modular Meetup to Start Soon!**: The Modular Meetup is scheduled to commence in just one hour, promising a deep dive into the **MAX framework**.
   - Participants can join the virtual session at 6:30 PM PT via the provided [YouTube link](https://www.youtube.com/watch?v=WK5dVQ8vhbU) for an engaging discussion.
- **MAX Framework to be Unveiled**: The meetup will focus on the **MAX framework**, offering attendees an in-depth exploration of its features and applications.
   - Interested individuals are encouraged to tune in at 6:30 PM PT to discover how the **MAX framework** can enhance their projects and workflows.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1448789096276168778)** (14 messages🔥): 

> `Claude 4.5 Thinking in Chinese, Kimi AI vs Mistral, Zoom is Frontier Lab Now?, Kimi NB Pro Slides Limited Use, Multiple Accounts Violation` 


- **Multilingual Claude and Kimi's Rise**: A user noted that **Claude 4.5** sometimes starts thinking in Chinese and is replacing their **Mistral subscription** with **Kimi**.
   - Another user said **Kimi** used to be perfect but is now *mid at best* compared to alegro, noting it *dont make what i need with ok computer*.
- **Zoom Ventures into Frontier AI**: A user questioned why **Zoom** is a frontier lab now, linking to a [2010 Tweet from Zoom](https://x.com/zoom/status/1999159317103292610?s=46).
   - Another user shared a list of sectors with companies building **LLMs**, including video calling platforms like Zoom, plus social media, FPS game studios, e-commerce platforms and hedge funds.
- **Kimi's Limited NB Pro Slide Generation**: Users reported that **Kimi AI** gives new accounts only 2-4 times to generate a slide with **NB Pro**.
   - One user received a message saying, *Generation failed, Kimi has too many tasks right now, please try later. Subscribers get access during peak time*, indicating usage limits.
- **Multiple Accounts Banned by ToS**: A user pointed out that creating multiple accounts violates Kimi's Terms of Service, providing a link to the [Kimi User Agreement](https://www.kimi.com/user/agreement/modelUse?version=v2).
   - The terms state users should *not register or operate multiple accounts for abusive purposes* and are *responsible for all activities that occur under your account*.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1448784214639116319)** (7 messages): 

> `Aider Prompt Caching, DeepSeek Prompt Caching, Aider Server optimization` 


- **Aider's Prompt Caching Feature Questioned**: A user inquired about the prompt caching feature in the Aider documentation, questioning why additional settings are needed in Aider when **DeepSeek** already supports prompt caching.
   - Another user responded that the prompt cache works at the **server level**, and can be optimized by performing very similar tasks with the same context.
- **Optimizing Aider's Server-Side Prompt Cache**: The prompt cache operates at the **server level**, allowing for potential optimization through repeated tasks with similar contexts.
   - Users can maximize cache utilization by submitting requests that share substantial overlap in both task and context, effectively leveraging Aider's caching mechanism.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1449202011588526201)** (1 messages): 

> `ReasoningLayer AI, Neurosymbolic AI, DSPY GEPA, Ontology Ingestion` 


- **ReasoningLayer AI Opens Waitlist!**: A member opened the waitlist for [ReasoningLayer AI](https://reasoninglayer.ai), a **neurosymbolic AI project** aimed at fixing many of the weaknesses of today’s LLMs by adding *real, structured reasoning* on the center.
   - They will be using **DSPY GEPA** in their *ontology ingestion pipeline*, and are excited to share more about it.
- **DSPY GEPA Integration for Ontology Ingestion**: **ReasoningLayer AI** is integrating **DSPY GEPA** into their *ontology ingestion pipeline* to enhance reasoning capabilities.
   - This integration aims to leverage structured reasoning to address the shortcomings of current LLMs.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1448903580885979249)** (4 messages): 

> `BAML, DSPy, BAMAdapter` 


- **BAML + DSPy Tool Naming Conundrum**: A member inquired about a term that combines **BAML** and **DSPy**, seeking to define the category of tool they represent.
- **BAMAdapter Release Speculation**: Another member responded enthusiastically and inquired about the release timeframe of the new **BAMAdapter**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1448952567412559965)** (5 messages): 

> `Manus outage, Manus new features since Feb 2023` 


- ****Manus** has a short outage**: Members reported that **Manus** was unresponsive for about 30 minutes, but it is back up now.
   - It is unknown what the cause of the outage was.
- **Catching up on a year of **Manus** evolution**: A member who hadn’t used **Manus** since February of last year asked what they had missed.
   - Unfortunately, no one followed up with a summary of new features since February 2023, so we don't know what they missed!


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1448816270198833182)** (1 messages): 

> `GPT-5.2, Windsurf, Agentic Coding, SOTA coding model` 


- **GPT-5.2 Lands in Windsurf!**: **GPT-5.2** is now live in **Windsurf**, available for 0x credits for a limited time for paid and trial users.
   - According to the announcement, this version represents the *"biggest leap for GPT models in agentic coding since GPT-5"* ([announcement link](https://x.com/windsurf/status/1999250307507978257?s=20)).
- **Windsurf defaults to GPT-5.2**: The new **GPT-5.2** is claimed to be a **SOTA** coding model at its price point and will be the default model in **Windsurf** for new users.
   - Users are encouraged to download the latest versions of **Windsurf** and **Windsurf Next** to try it out ([download link](https://windsurf.com/download/editor)).


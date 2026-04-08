---
id: MjAyNS0x
title: minor updates to GPT 5.1 and SIMA 2
date: '2025-11-13T05:44:39.731046Z'
description: >-
  **OpenAI** released **GPT-5.1** family models including **5.1-Codex** and
  **5.1-Codex-Mini** with improved steerability, faster responses, and new tools
  like apply_patch and shell command execution. Pricing remains unchanged from
  5.0. Immediate integrations include **GitHub Copilot**, **VS Code**,
  **Cursor**, and **Perplexity** adopting GPT-5.1 models. **Google DeepMind**
  announced **SIMA 2**, a **Gemini**-powered agent capable of language
  instruction following, planning, and self-improvement without human feedback,
  targeting robotics applications. New research on context engineering and
  agentic tool use patterns was published, with contributions from **Weaviate**
  and **LlamaIndex** on database query planning and chart parsing respectively.
  *"Adaptive reasoning"* and agentic coding improvements are highlighted in
  GPT-5.1- Instant.
companies:
  - openai
  - google-deepmind
  - github
  - microsoft
  - cursor_ai
  - perplexity-ai
  - weaviate
  - llamaindex
models:
  - gpt-5.1
  - gpt-5.1-codex
  - gpt-5.1-codex-mini
  - sima-2
  - gemini
topics:
  - adaptive-reasoning
  - agentic-coding
  - tool-use
  - context-engineering
  - memory-architecture
  - self-improvement
  - retrieval-augmentation
  - database-query-planning
  - chart-parsing
  - robotics
people:
  - sama
  - allisontam_
  - cline
  - cognition
  - demishassabis
  - omarsar0
  - helloiamleonie
---


a day of incremental improvements.

> AI News for 11/12/2025-11/13/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (201 channels, and 6666 messages) for you. Estimated reading time saved (at 200wpm): 523 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

We featured [SIMA last year](https://news.smol.ai/issues/24-03-13-ainews-deepmind-sima-one-ai-9-games-600-tasks-visionlanguage-only), and today [SIMA 2 was announced](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/), but due to lack of technical report and general industry excitement we are not making it a title story.

GPT 5.1 was released in the API with evals, but because we already made it title story yesterday, it fails to qualify today.

---

# AI Twitter Recap

**OpenAI’s GPT‑5.1 rollout and ecosystem uptake**

- **GPT‑5.1 family in API + new agent tools**: OpenAI shipped GPT‑5.1 (and 5.1‑Codex, 5.1‑Codex‑Mini) with better steerability, faster responses, and stronger coding. New built‑in tools include an apply_patch for reliable free‑form code edits and a shell tool for controlled command execution; prompt caching is extended up to 24 hours to cut cost/latency on repeated prompts. Pricing remains the same as 5.0 according to [@sama](https://twitter.com/sama/status/1989048466967032153). See launch details and Q&A from OpenAI DevRel [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1989042617750024403), tools callouts [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1989042624574198021), cookbook links [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1989043495269724617), and early customer quotes [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1989064999680242007). OpenAI also highlights updated safety evals and minor benchmark regressions (AIME, Taubench) to counter “benchmark‑maxxing” concerns per [@swyx](https://twitter.com/swyx/status/1989047883639980141).
- **Adaptive reasoning and agentic coding**: 5.1‑Instant introduces “adaptive reasoning” (spend more tokens on harder tasks), per OpenAI’s post‑training team [@allisontam_](https://twitter.com/allisontam_/status/1989138927970848936). Agent toolchains and scaffolds are already re‑tuning around 5.1: Cline details execution‑focused prompting, stricter plan/act transitions, and two‑phase deep‑planning for large repos ([thread](https://twitter.com/cline/status/1989056367030829458)); Cognition sets 5.1 as default in Windsurf for more fluid, less “overthinking” coding ([announcement](https://twitter.com/windsurf/status/1989069991770214580), [@cognition](https://twitter.com/cognition/status/1989081722353529178)).
- **Immediate integrations**: GitHub Copilot rolled out GPT‑5.1, 5.1‑Codex, 5.1‑Codex‑Mini in public preview ([@github](https://twitter.com/github/status/1989044218451394968)) and VS Code shows the models landing in editor experiences ([@code](https://twitter.com/code/status/1989044946058326370)). Cursor added all three with updated routing ([@cursor_ai](https://twitter.com/cursor_ai/status/1989045849003835460)); Perplexity enabled 5.1 for Pro/Max ([@perplexity_ai](https://twitter.com/perplexity_ai/status/1989075483385069949)); other tools quickly followed (Anycoder, Yupp, Factory) ([@_akhaliq](https://twitter.com/_akhaliq/status/1989161892880032132), [@yupp_ai](https://twitter.com/yupp_ai/status/1989080371775041942), [@FactoryAI](https://twitter.com/FactoryAI/status/1989052558279864595)).

**Agents, embodiment, and memory architectures**

- **SIMA 2 (DeepMind)**: Google DeepMind unveiled SIMA 2, a Gemini‑powered agent that follows language instructions, plans, takes actions via standard keyboard/mouse, generalizes to unseen games, and self‑improves through trial‑and‑error using a Gemini utility model—no human feedback. It also navigates worlds generated by Genie 3 ([overview](https://twitter.com/GoogleDeepMind/status/1988986218722291877), [Genie 3 demo](https://twitter.com/GoogleDeepMind/status/1989024090414309622), [@demishassabis](https://twitter.com/demishassabis/status/1989096784870928721)). Google positions this as a step toward robotics applications ([post](https://twitter.com/GoogleDeepMind/status/1988987865401798898)).
- **Context and tool use patterns**: Google published a practitioner whitepaper on context engineering—sessions, memory, and how to architect retrieval for agent reliability ([@omarsar0](https://twitter.com/omarsar0/status/1989081828678893837)). Weaviate’s “Query Agent” shows database NL‑to‑query planning across collections with filters, routing, aggregation and citations ([@helloiamleonie](https://twitter.com/helloiamleonie/status/1989007852502139221)). LlamaIndex added agentic chart parsing that traces contours in line charts to extract numeric series ([@llama_index](https://twitter.com/llama_index/status/1989060127551549854)).
- **Agent infra hardening**: LangChain introduced Sandboxes for DeepAgents to safely execute arbitrary code/bash in remote sandboxes (Runloop, daytona, Modal), separating planning from execution environments ([announcement](https://twitter.com/LangChainAI/status/1989006586388574397)). LangSmith Essentials course focuses on continuous testing/observability for multi‑turn/tool‑calling agents ([@LangChainAI](https://twitter.com/LangChainAI/status/1989025161488793743)). Qwen released DeepResearch 2511 with “Advanced Mode”, file uploads, deeper search, and configurable report formats/citations ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1989026687611461705)). Community demos like “Kimi Deep Researcher” show 100s of tool calls per session ([@omarsar0](https://twitter.com/omarsar0/status/1988974710592516454)).

**Interpretability and training science**

- **Sparse circuits as a training objective**: OpenAI proposes training small LMs with extremely sparse weights to make internal mechanisms easier to interpret, isolating circuits for behaviors like string termination and variable tracking. They release code and models, positioning this as a path toward a fully interpretable GPT‑3‑class “model organism” for safety/understanding ([OpenAI](https://twitter.com/OpenAI/status/1989036214549414223), [thread](https://twitter.com/OpenAI/status/1989036218160673103), [team lead](https://twitter.com/nabla_theta/status/1989043939374924251)).
- **Temporal features and JEPA theory**: Temporal Feature Analysis introduces predictive‑coding style modeling of dynamic features in LLM activations, addressing the static‑feature assumption of SAEs ([@EkdeepL](https://twitter.com/EkdeepL/status/1989009095953895756), [@GoodfireAI](https://twitter.com/GoodfireAI/status/1989010394380485083)). In vision, LeCun/Balestr’s LeJEPA formalizes target embeddings as isotropic Gaussian with a new SIGReg objective, simplifying JEPA training (no teacher‑student/stop‑grad) and delivering strong results across >10 datasets and 60+ architectures ([@ylecun](https://twitter.com/ylecun/status/1988999683801510063), [@TheTuringPost](https://twitter.com/TheTuringPost/status/1989039076302049701)).
- **Post‑training deltas**: New analysis contrasting RL vs SFT shows RL preserving principal singular directions while updating off‑principal ones, whereas SFT can distort spectra and overfit—implications for PEFT targeting and schemes like PiSSA ([@tydsh](https://twitter.com/tydsh/status/1989049095575728156)). PEFT v0.18 ships with new methods and improvements ([@BenjaminBossan](https://twitter.com/BenjaminBossan/status/1988993386729390191)).

**Model releases and multimodal/video**

- **Zhipu AI GLM‑4.6**: Zhipu announced GLM‑4.6; Together AI is hosting it for production workloads, positioning it as near‑parity with Claude Sonnet 4 while using ~15% fewer tokens ([lab](https://twitter.com/Zai_org/status/1989005078926143810), [host](https://twitter.com/togethercompute/status/1989082601399939312)).
- **Real‑time detection with DETR**: RF‑DETR (DINOv2 backbone) runs NAS over ~6k variants with weight sharing; RF‑DETR‑N hits 48.0 AP at 2.3 ms on COCO, matching YOLOv8/11‑M at ~2× speed; segmentation head variant reaches 40.3 AP mask at 3.4 ms ([@skalskip92](https://twitter.com/skalskip92/status/1989004912609411133)).
- **Video generation entrants**: Vidu Q2 Turbo/Pro debuted on Video Arena, landing #6/#7 in Image‑to‑Video with precise emotion/camera control; API pricing $4–$6.10/min 1080p ([@arena](https://twitter.com/arena/status/1989056583872180298)). NVIDIA introduced TiDAR (“Think in Diffusion, Talk in Autoregression”), a hybrid diffusion/AR framework ([@_akhaliq](https://twitter.com/_akhaliq/status/1988963077690438097)).
- **Open imaging efforts**: Photoroom open‑sourced their second from‑scratch text‑to‑image model with weights and training process on HF ([@matthieurouif](https://twitter.com/matthieurouif/status/1988981733866271223)).

**Infra, platforms, and performance**

- **Hugging Face x Google Cloud**: A broad partnership to accelerate open model dev on GCP: HF DLCs on Vertex AI/Cloud Run/GKE, native TPU support, Inference Endpoints on GCP, security via Google Threat Intelligence/Mandiant, and a new GCP‑cached gateway to speed model/dataset IO—reflecting >1,500 TB/day traffic and likely >$1B/yr cloud spend already ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1989000335247983049), [@alvarobartt](https://twitter.com/alvarobartt/status/1988970441357094984)). Google also shipped a major UX upgrade to the Gemini CLI ([@googledevs](https://twitter.com/googledevs/status/1989119863961337889)).
- **Inference velocity**: Baseten reports 2× speedups and 1.6× throughput gains for long‑context code generation using NVIDIA Dynamo for multi‑node inference orchestration ([@basetenco](https://twitter.com/basetenco/status/1989058852789317717)); Modal details 12% faster speculative decoding in SGLang ([@akshat_b](https://twitter.com/akshat_b/status/1989019570783629366)). SkyPilot v0.10.5 improves managed job efficiency (18×), scales its API server, and expands the Python SDK/admin policy surface ([@skypilot_org](https://twitter.com/skypilot_org/status/1989083081953931284)).
- **Dev environment convergence**: VS Code added native autocompletion/quality‑of‑life improvements; critically, Google Colab runtimes can now back VS Code notebooks for GPU/TPU compute within the editor ([@googledevs](https://twitter.com/googledevs/status/1989033099737407820)).

**Security, evaluation, and governance**

- **AI‑led espionage disrupted**: Anthropic says it detected and disrupted a large‑scale, minimally human‑supervised cyber‑espionage campaign it attributes to a Chinese state‑sponsored group—potentially the first documented case of an AI‑executed attack at this scale, with targets across tech, finance, chemicals, and government ([disclosure](https://twitter.com/AnthropicAI/status/1989033793190277618), [analysis](https://twitter.com/AnthropicAI/status/1989033795341648052)). The episode underscores the need for AI‑aware cyber defenses.
- **Policy and evals**: Anthropic open‑sourced a political bias evaluation and discussed ideal model behavior in political discourse ([announcement](https://twitter.com/AnthropicAI/status/1989076472208978127)). UN SAB’s video with Yoshua Bengio covers frontier verification via compute tracking and tamper‑resistant chips ([@ScienceBoard_UN](https://twitter.com/ScienceBoard_UN/status/1988971216951210467)). Kagi launched “SlopStop” for community‑driven AI‑slop detection in search ([@KagiHQ](https://twitter.com/KagiHQ/status/1989050447844270340)).
- **Market reality checks**: Andrew Ng cautioned against AI hype paralysis—LLMs remain powerful but specialized; application customization is essential and AGI‑level generality is distant ([thread](https://twitter.com/AndrewYNg/status/1989003741316673714)). Meanwhile, Cursor announced a $2.3B Series D and claimed >$1B ARR, asserting agent PMF and model ownership as strategic moats ([@cursor_ai](https://twitter.com/cursor_ai/status/1988971258449682608)).

**Top tweets (by engagement)**

- Anthropic’s disclosure of disrupting an AI‑led espionage campaign: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1989033793190277618) and [analysis](https://twitter.com/AnthropicAI/status/1989033795341648052)
- Karpathy on the transformative urban impact of self‑driving: [@karpathy](https://twitter.com/karpathy/status/1989078861800411219)
- OpenAI interpretability (sparse circuits): [@OpenAI](https://twitter.com/OpenAI/status/1989036214549414223)
- OpenAI GPT‑5.1 API/pricing/prompt‑cache announcement: [@sama](https://twitter.com/sama/status/1989048466967032153)
- Google Colab runtimes in VS Code notebooks: [@googledevs](https://twitter.com/googledevs/status/1989033099737407820)
- Google DeepMind’s SIMA 2 agent and Genie 3 worlds: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1988986218722291877)
- Cursor’s $2.3B funding and $1B ARR milestone: [@cursor_ai](https://twitter.com/cursor_ai/status/1988971258449682608)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Jan-v2-VL Model Release and Benchmarks

- [**Jan-v2-VL: 8B model for long-horizon tasks, improving Qwen3-VL-8B’s agentic capabilities almost 10x**](https://www.reddit.com/r/LocalLLaMA/comments/1ovxksu/janv2vl_8b_model_for_longhorizon_tasks_improving/) (Activity: 754): **Jan-v2-VL is an 8B vision-language model designed for long-horizon, multi-step tasks, significantly enhancing the capabilities of the base model Qwen3-VL-8B-Thinking. It achieves** `49 steps` **on the Long-Horizon Execution benchmark, compared to the base model's** `5 steps` **and other similar models'** `1-2 steps`**. The model is available in three variants: low, medium, and high, each optimized for different balances of efficiency and reasoning depth. It can be run using vLLM or llama.cpp, with recommended parameters including** `temperature: 1.0`**,** `top_p: 0.95`**, and** `presence_penalty: 1.5`**. The model is accessible on [Hugging Face](https://huggingface.co/collections/janhq/jan-v2-vl) and the [Jan GitHub](https://github.com/janhq/jan).** A comment queries why the reasoning variant is the base model instead of an instruct variant, indicating a potential interest in different model configurations for specific tasks.
    - Delicious_Focus3465 shared detailed results on the Long Horizon Benchmark, highlighting that the Jan-v2-VL model significantly improves upon the Qwen3-VL-8B in terms of agentic capabilities, achieving nearly a tenfold increase in performance. This suggests substantial advancements in handling long-horizon tasks, which are crucial for complex decision-making processes.
    - MaxKruse96 inquired about the choice of the 'Reasoning' variant as the base model instead of the 'Instruct' variant. This choice could imply a focus on enhancing the model's logical reasoning capabilities, which might be more beneficial for tasks requiring deep understanding and decision-making over extended periods.
    - maglat asked about the availability of a Jan server variant similar to Open WebUI, expressing a need for a solution that allows access from any browser on a Jan instance running on a local LLM rig. This indicates a demand for more flexible deployment options that can integrate with existing infrastructure.

### 2. Running Large Models on Consumer Hardware

- [**Running a 1 Trillion Parameter Model on a PC with 128 GB RAM + 24 GB VRAM**](https://www.reddit.com/r/LocalLLaMA/comments/1ow0jj0/running_a_1_trillion_parameter_model_on_a_pc_with/) (Activity: 356): **A user successfully ran the Kimi K2 Thinking model with** `1 trillion parameters` **on a consumer-grade PC using llama.cpp. The setup included an Intel i9-13900KS CPU,** `128 GB DDR5 RAM`**, and an RTX 4090 GPU with** `24 GB VRAM`**. The model was quantized using Unsloth UD-Q3_K_XL from [Hugging Face](https://huggingface.co/unsloth/Kimi-K2-Thinking-GGUF), achieving a generation speed of** `0.42 tokens/sec`**. The user noted that memory mapping (mmap) in llama.cpp allows handling model files larger than the available RAM, and quantization below** `~4 bits` **significantly reduces model quality. The command used included** `-no-warmup` **to prevent startup crashes, and the llama.cpp version was** `b6963`**.** One commenter pointed out that the short prompt used for benchmarking invalidates the results, suggesting a longer prompt and response for accurate performance metrics. Another advised against running models larger than `120 billion parameters` on consumer hardware, emphasizing limits on active and dense parameters. A third commenter appreciated the benchmarks and shared their preference for the **gpt-oss-120b** model for its speed and balance, while favoring **Kimi-k2** and **minimax m2** for larger models.
    - DataGOGO highlights the importance of using sufficiently long prompts and responses for accurate benchmarking with `llama.cpp`. They suggest using at least a few hundred tokens in both the prompt and response, recommending a setup of `1000t` prompt and `200t` response for quick benchmarking. This ensures that performance counters are reliable and that prompt processing and generation speeds are recorded separately.
    - GreenTreeAndBlueSky provides guidelines on model size limitations for running large models on a PC. They suggest not exceeding `120b` total parameters, `12b` active parameters, and `32b` if the model is dense. These constraints are likely based on hardware limitations and the need to balance performance with resource availability.
    - lumos675 mentions the impact of storage medium on performance, suggesting that running the model from NVMe storage could achieve around `4 to 5 tokens per second (tps)`. This implies that storage speed is a critical factor in model performance, especially when dealing with large parameter models.

### 3. IBM Patent Controversy in AI

- [**IBM's AI Researchers Patented a 200 yr old Math Technique by Rebranding as AI Interpretability**](https://www.reddit.com/r/LocalLLaMA/comments/1ow6a9i/ibms_ai_researchers_patented_a_200_yr_old_math/) (Activity: 554): **IBM AI researchers have filed a patent application for implementing a** `Continued Fraction` **class as linear layers in PyTorch, which involves calling** `backward()` **on the computation graph. This move has raised concerns as it potentially affects various fields that use derivatives or power series with continued fractions, such as mechanical engineering, pure mathematics, and numerical programming. The patent application is seen as controversial because it rebrands a** `200-year-old` **mathematical technique as AI interpretability, sparking debate over the novelty and obviousness of the invention. [Read more here](https://leetarxiv.substack.com/p/ibm-patented-eulers-fractions).** The top comments highlight skepticism about the American patent system, noting that this is a patent application, not a granted patent, and emphasize the need for third-party submissions to contest its novelty. There is also criticism of the patent system for allowing such applications, which could be seen as patent trolling, particularly affecting those in the US.
    - Starcast highlights a common misunderstanding in technology reporting, emphasizing that the discussed item is a patent application, not a granted patent. They note that anyone can submit a third-party submission to the USPTO to contest the novelty or obviousness of the application based on prior art, which is a critical step in the patent examination process.
    - RockyCreamNHotSauce points out the challenges faced by patent examiners, especially with the influx of AI-related applications. They argue that abstract mathematical ideas, such as those implemented in frameworks like PyTorch, are not patentable. The comment suggests that merely implementing a mathematical concept in code does not constitute a significant inventive step, drawing a parallel to writing down a mathematical idea on paper.
    - Lissanro mentions that the patent application is specific to the US, implying that its impact is geographically limited. They express concern over the practice of patenting ideas that may not be novel, suggesting that such actions could be considered patent trolling, which is problematic even if the patent is not granted.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Gemini 3 Mobile Rollout Updates

- [**NOT HYPE POSTING: Gemini 3 is rolling out to mobile users as we speak!**](https://www.reddit.com/r/Bard/comments/1ovvmjo/not_hype_posting_gemini_3_is_rolling_out_to/) (Activity: 1133): **Gemini 3.0 is currently being rolled out to mobile users on both Android and iOS platforms, specifically for those utilizing the canvas feature in the Gemini app. This rollout is part of a broader strategy by Google to prioritize mobile deployment before web integration. The update includes new tags for Gemini 2.5 Flash and Pro models, indicating potential rebranding or updates. Additionally, network logs have confirmed sightings of Gemini 3.0 in the Gemini Enterprise environment, although this is not publicly accessible. There is also speculation about the simultaneous release of Nano Banana 2, as indicated by network requests on the Gemini website.** Some users have confirmed the rollout on Android, while others report no changes yet. There is a debate about the significance of the Gemini 3.0 sightings in Enterprise logs, with some users skeptical about its immediate public availability.
    - **Initial-Plenty2326** highlights that Gemini 3 is being rolled out on both Android and iOS platforms, which is significant for cross-platform availability. This suggests a broader user base can access the new features simultaneously, enhancing the reach and feedback loop for the developers.
    - **Alex_146** provides a direct comparison between Gemini 3 and its predecessor, Gemini 2.5, by sharing a link to a webpage created using the Android app. The implication is that Gemini 3 offers significantly enhanced capabilities, particularly in terms of creative design and responsiveness, which were not achievable with version 2.5.
    - **Salty_Flow7358** speculates on the release timeline of Gemini 3, suggesting a potential release date of November 19th, based on the end of a promotional offer for students on November 18th. This insight could be valuable for users planning to upgrade or take advantage of the new features.

### 2. AI Content in Media

- [**They copied the whole ChatGPT answer and even kept the part where it offers to make it prettier.**](https://www.reddit.com/r/OpenAI/comments/1ovuzx2/they_copied_the_whole_chatgpt_answer_and_even/) (Activity: 3526): **The image highlights a newspaper article that inadvertently included a verbatim section from a ChatGPT response, complete with a suggestion to make the text more visually appealing for a front-page layout. This incident underscores the challenges and potential pitfalls of integrating AI-generated content into traditional media without proper editorial oversight. The highlighted text in the article suggests that the AI's output was copied directly, indicating a lack of thorough editing or review processes in place at the publication.** Commenters humorously note the apparent lack of editorial oversight, with one suggesting that the prompt used for ChatGPT is evident from the text. Another comment highlights the importance of retaining copy editors to prevent such oversights.
    - Neat-Conference-5754 highlights a critical perspective on AI usage, emphasizing that treating AI merely as a tool can lead to a lack of accountability and reliance on AI for tasks that require human judgment. They argue that AI should be seen as a co-creator or assistant, necessitating human oversight and editing to ensure quality and accuracy in outputs. This underscores the importance of integrating AI with human expertise rather than replacing it entirely.
- [**They copied ChatGPT word-for-word and left the ending in. Wild.**](https://www.reddit.com/r/ChatGPT/comments/1ovv0kg/they_copied_chatgpt_wordforword_and_left_the/) (Activity: 20305): **The image highlights a significant editorial oversight where a newspaper article on auto sales in Pakistan inadvertently included unedited AI-generated text from ChatGPT. The article, which discusses the increase in vehicle sales with detailed statistics, mistakenly left in a section that instructs to create a 'front-page style' version with 'punchy stats and an infographic layout.' This suggests that the editorial team failed to remove or modify the placeholder text generated by ChatGPT before publication, raising questions about the editorial process and the reliance on AI tools in journalism.** Commenters express embarrassment and critique the editorial oversight, suggesting that the proofreader should face consequences for allowing AI-generated text to appear in the final print.

### 3. New AI Model and Benchmark Announcements

- [**Google DeepMind - SIMA 2: An agent that plays, reasons, and learns with you in virtual 3D worlds**](https://www.reddit.com/r/singularity/comments/1ow3g1o/google_deepmind_sima_2_an_agent_that_plays/) (Activity: 1538): **Google DeepMind has introduced SIMA 2, an advanced AI agent capable of playing, reasoning, and learning within virtual 3D environments. This agent demonstrates significant self-improvement capabilities, learning complex tasks through trial-and-error and feedback from the Gemini model. Notably, SIMA 2 can transition from human-guided learning to self-directed play, enhancing its skills in new, unseen games without additional human data. This iterative learning process is further enhanced in Genie environments, marking a significant step towards training general AI agents across diverse, procedurally generated worlds.** One commenter highlights the potential for **SIMA 2** to train robots in realistic, cost-effective, and safe virtual environments, which could significantly advance AI research. Another expresses a desire for a subscription-based AI agent for casual interaction and gaming.
    - SIMA 2's ability to self-improve is a significant advancement, as it can transition from learning through human demonstrations to self-directed play in new games. This capability allows it to develop skills in previously unseen environments without additional human-generated data, leveraging its own experience data to train subsequent versions. This iterative self-improvement is facilitated by the use of Genie environments, marking a major step towards training general agents across diverse, generated worlds.
    - The integration of SIMA 2 with Genie 3 for creating virtual worlds represents a leap towards the development of general AI agents. By using these tools, SIMA 2 can recursively self-improve, which some commenters suggest is a step towards the technological singularity. This process involves SIMA 2 learning and adapting in newly created environments, potentially leading to more advanced AI capabilities.
    - The potential for SIMA 2 to learn and adapt in virtual worlds raises questions about its applicability to real-world scenarios, such as humanoid robot AI. The ability to generalize learning from virtual to real-world environments could pave the way for advanced AI systems capable of operating in complex, dynamic settings. This capability is seen as a precursor to more sophisticated AI applications beyond gaming and virtual simulations.
- [**GPT-5.1 is definitely something**](https://www.reddit.com/r/ChatGPT/comments/1owdjzw/gpt51_is_definitely_something/) (Activity: 1467): **The image is a humorous exchange that highlights the conversational quirks of AI models like GPT-5.1, particularly in how they handle user interactions. The conversation involves a user discussing the negligible differences in caloric content between oat flour and oatmeal, and almond milk, with the AI responding in a way that seems overly dramatic and human-like. This reflects ongoing challenges in AI development related to maintaining context and tone in user interactions, especially when users expect consistent behavior across similar queries. The post and comments suggest that while the AI's responses can be amusing, they also point to areas where AI models might need refinement in handling repetitive tasks or maintaining a consistent conversational tone.** Commenters find the AI's responses amusing and liken them to human interactions, suggesting that while the AI's conversational style is entertaining, it may not always be practical for users seeking straightforward assistance.
    - Buck_Thorn highlights that the behavior of GPT models, such as GPT-5.1, is influenced not only by the version number but also by user-specific settings like chat history and personality configurations. This suggests that user interactions and customizations can significantly impact the model's responses, making it crucial to consider these factors when evaluating model performance.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. GPT-5.1 Everywhere: Coding, Reasoning, Rollouts**

- **Five-Point-One Floods the Toolchain**: OpenAI announced **GPT‑5.1** with adaptive reasoning and improved coding in [GPT‑5.1](https://openai.com/index/gpt-5-1/); OpenRouter simultaneously listed [**GPT‑5.1 Chat**](https://openrouter.ai/openai/gpt-5.1-chat), [**GPT‑5.1‑Codex**](https://openrouter.ai/openai/gpt-5.1-codex), and [**GPT‑5.1‑Codex‑Mini**](https://openrouter.ai/openai/gpt-5.1-codex-mini-20251113); and **Windsurf** enabled free 7‑day access, setting **GPT‑5.1** as default per [Windsurf’s announcement](https://x.com/windsurf/status/1989069991770214580).
    - Engineers reported tangible upgrades for **agentic coding** and frontend work, with Windsurf claiming **faster**, **more steerable** behavior that **overthinks less** while modulating reasoning depth; **Cursor** users spotted **GPT‑5.1‑Codex** appearing in the [latest codex alpha](https://x.com/OpenAIDevs/status/1986861734619947305) and cross‑integration with [Windsurf](https://www.windsurf.ai/).
- **Polaris Packs It In, 5.1 Steps Up**: **OpenRouter** deprecated **Polaris Alpha** (an early GPT‑5.1 without reasoning) and replaced it with a faster, token‑efficient **GPT‑5.1** family featuring adaptive reasoning and better coding per [GPT‑5.1 for developers](https://openai.com/index/gpt-5-1-for-developers/); the new endpoints are [**GPT‑5.1 Chat**](https://openrouter.ai/openai/gpt-5.1-chat), [**GPT‑5.1‑Codex**](https://openrouter.ai/openai/gpt-5.1-codex), and [**GPT‑5.1‑Codex‑Mini**](https://openrouter.ai/openai/gpt-5.1-codex-mini-20251113).
    - Teams noted the **Instant** experience in ChatGPT maps to **GPT‑5.1 Chat**, while code‑heavy workflows start shifting toward **Codex** variants, aligning with OpenAI’s developer guidance in [GPT‑5.1 for developers](https://openai.com/index/gpt-5-1-for-developers/).
- **Ask Me Anything About 5.1**: OpenAI scheduled a **Reddit AMA** to address **GPT‑5.1** and customization on [r/OpenAI](https://redd.it/1ovkt6n/) at 2PM PT, amid mixed feedback on **custom instructions** and narrative quality documented in [GPT‑5.1](https://openai.com/index/gpt-5-1/).
    - Developers compared **GPT‑5.1** to earlier models for storytelling and formatting fidelity and plan to surface issues and requests directly in the AMA for clarity on roadmap and tuning priorities.

**2. GPU Kernels & Blackwell: From Helion to NVFP4**

- **Helion Hurtles with Handy Autotune**: **Helion** confirmed 0.2.x **backwards compatibility**, released **v0.2.2**, and added `configs=` for **autotuning** (Triton‑style) per [Helion issue #164](https://github.com/pytorch/helion/issues/164), while its eager‑mode interpreter stayed surprisingly fast.
    - Engineers highlighted grabbing the winning kernel via `helion_rms_norm_fwd.bind((x, w, eps))._config`, and contrasted **Helion’s** quick interpret mode with **Triton’s** slow interpret path in dev loops.
- **NVFP4 GEMV Gauntlet Gears Up**: A hackathon kicked off to optimize **GEMV** for **NVFP4** on **Blackwell** GPUs, complete with **Datacrunch B200** access and a recommended **CuTeDSL** stack, detailed in [NVFP4 GEMV](https://veitner.bearblog.dev/nvfp4-gemv/).
    - Participants reported rapid iteration cycles and hardware‑close productivity using CuTeDSL, targeting µs‑scale kernels and competitive leaderboard placements described in the blog’s challenge brief.
- **NCU Grades Clouds, Not Curves**: Cloud vendors now get graded on **NCU** (NVIDIA Compute Unified Device Architecture) support, raising the bar for GPU observability and performance tooling per [Semianalysis: Clustermax 2.0](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard).
    - The community framed **NCU** capabilities as a must‑have for production GPU workloads, expecting providers to standardize profiling and kernel telemetry at scale.

**3. Data Pipelines: Clean Corpora, Licenses, and Tokenizers**

- **French Wiki Scrubs Up, Ships JSON**: A cleaned **French Wikipedia** dump with over **2.7M JSON files**—retaining templates, tables, HTML, refs, infoboxes, and links—landed on Hugging Face as [wikipedia-fr-2.7m-clean-json](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json).
    - Contributors discussed extending the pipeline to **English Wikipedia** next and using the structured JSON to preserve rich graph features for downstream training.
- **NVIDIA’s License Limbo Looms Large**: Practitioners flagged restrictive terms in **NVIDIA’s dataset license**—training/eval/public‑results limits and a unilateral termination clause—summarized in this thread: [GoodfireAI on X](https://x.com/goodfireai/status/1986495330201051246).
    - Teams weighed the legal ambiguity for public benchmarks and reproducibility, noting the chilling effect on sharing models and results trained on licensed corpora.
- **Synthetic QA & Tokenizers Take Flight**: Discussions cited **synthetic QA** patterns (e.g., QA tails) generated with systems like in the paper [Nemotron‑CCh](https://arxiv.org/abs/2511.08923v1) and broader trends covered in [PleIAs: The New Data Frontier](https://pleias.fr/blog/blogsynth-the-new-data-frontier).
    - A new **tokenizer** preprint—[Tokenizer Paper](https://arxiv.org/abs/2511.09709)—drew interest for its potential to reduce fragmentation and improve compression on modern multilingual, multimodal corpora.

**4. Big Money and Compute: Funding and Datacenters**

- **Parallel Web Pulls a Perfect $100M**: **Parallel Web Systems** (Parag Agrawal) raised **$100M Series A** to build web infrastructure for **AI agents**, as announced here: [Parag on X](https://x.com/paraga/status/1988729121636294682).
    - Builders applauded the product design and speculated on SDKs, crawl/host layers, and agent‑native protocols the company might standardize for production systems.
- **Anthropic Assembles a $50B Compute Colossus**: **Anthropic** announced plans to invest **$50B** in **US datacenters** across **Texas** and **New York**, igniting debate over domestic compute scale: [Anthropic on X](https://x.com/anthropicai/status/1988624013849935995).
    - Practitioners weighed staffing and environmental constraints against the upside of abundant on‑shore compute for training and inference pipelines.
- **Nous Nixes Hermes Prices by Seventy Percent**: **Nous Research** cut **Hermes 4 70B** and **Hermes 4 405B** API pricing by **70%**, opening broader access via the [Nous Portal](https://portal.nousresearch.com/) and the announcement [on X](https://x.com/NousResearch/status/1989077400957911394).
    - Developers expect cheaper iterative fine‑tunes and code‑assist experiments, noting Hermes’ presence as a code model in emerging IDE agents.

**5. Open‑Source Agentic Tooling: Agents, OCR, and CI**

- **Agent Army Orchestrates 114+ Sub‑Agents**: An open‑source coding agent framework with skills, **114+ sub‑agents**, and an executive planning layer shipped as [agent‑instructions (GitHub)](https://github.com/flora131/agent-instructions) alongside a design write‑up: [AI Coding Infrastructure](https://alexlavaee.me/blog/ai-coding-infrastructure/).
    - Teams position it to augment existing dev tools, delegating multi‑step features to specialized workers while keeping human‑in‑the‑loop for reviews and merges.
- **Propercode Promises Production‑Ready PRs**: **Propercode** debuted a multi‑agentic CLI for codebases powered by **Pydantic AI** agents, targeting reliable edits and reviews: [proper-code (GitHub)](https://github.com/JaiSuryaPrabu/proper-code).
    - The **v0.1** roadmap touts multiple modes (autonomous, learning guide) and multi‑tool setups to stabilize coding accuracy in CI pipelines.
- **DeepSeek OCR in a Dash**: A container‑ready **DeepSeek OCR API** enables quick self‑hosting and inference with **Unsloth**, pulling images from URLs or base64: [deepseek-ocr-api (GitHub)](https://github.com/neosantara-xyz/deepseek-ocr-api).
    - The community pitched it as a lightweight OCR microservice to feed document parsing and RAG ingestion stages in end‑to‑end LLM apps.

---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5.1 Sparks Mixed Reactions**: The release of [GPT-5.1](https://openai.com/blog/new-models-and-developer-products) on **LMArena** triggered debates, with some users claiming it outperforms **Sonnet 4.5**, while others view it as underwhelming due to hardcoded UI instructions.
   - While some praised its enhanced reasoning and coding, others criticized its lack of creativity, with one member calling it *"a non-release filler"*.
- **Gemini 3 Speculation Intensifies**: Anticipation is growing for the release of **Gemini 3**, with speculation pointing to a launch this or next week, possibly coinciding with the announcement of **Sima 2 research**.
   - Potential delays are attributed to **Kimi2** and a staggered release strategy, potentially starting with **Gemini Enterprise** or a mobile version on **Canvas**.
- **Code Arena Tweaks Frustrate Users**: Changes to [Code Arena](https://lmarena.com), including the removal of the retry button in battle mode due to alleged abuse, have sparked discontent among users.
   - Questions have emerged regarding the status of **Riftrunner** on the platform, with reports of errors and concerns about Cloudflare timeouts when accessing chat history.
- **Open Source Models Gain Traction**: Enthusiasm is building around open-source models like **GLM 4.6** and **Grok**, praised for their coding abilities and cost-effectiveness.
   - The owner of [Hugging Face](https://huggingface.co/) believes *"the world belongs to open source model"*, and a [YouTube tutorial](https://www.youtube.com/watch?v=GZqYr8_Q7DE) for using **Open Empathic** was shared.
- **Gemini Canvas Version Divides Opinion**: User experiences with the [Gemini Canvas](https://aistudio.google.com/app/canvas) version vary, with some admiring its fire UI while others remain unimpressed.
   - Speculation swirls as to whether it utilizes **Gemini 3** or remains on the **2.5 Pro** version, with some suggesting it's a *mass delusion*.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **GPT 5.1 arrives on Perplexity Pro**: **GPT 5.1** is now available for **Perplexity Pro** and **Max** subscribers, as announced in the Discord channel.
   - An image was attached to the announcement, potentially showcasing the new features or capabilities of **GPT 5.1**.
- **Perplexity Partner Program Bans Cause Uproar**: Users report being **banned** from the Perplexity Partner Program for alleged *fraudulent activity* with appeals unanswered since November 11, 2025, and speculate VPN usage or exceeding referral limits are potential causes.
   - A theory emerges that bans are timed around the 30-day hold for referrals, with claims that high earners are targeted, sparking intense frustration among users.
- **Pro Users Complain About Billing and Limits**: Users are reporting issues with Perplexity Pro, including **image generation limitations**, inaccurate reset dates for lab tokens, and deep research tool calls.
   - Specifically, users are saying *Limits reset on the first day of the month if I remember correctly*. Additionally, issues with google sign-on with comet are surfacing as well as people trying to figure out how to use the different AI models effectively.
- **GPT-5.1's Deployment Plagued by Doubts**: Members are noting the rollout of **GPT 5.1**, but it is unclear if there's an official release or whether people have actually received the new model.
   - Some users are reporting that the new model is similar to **Gemini 3.0** when the Canvas setting is enabled, but also mention its better at answering questions.
- **Scam Accusations Rise Amidst Program Chaos**: Amidst the confusion, some users claim that **Perplexity is running a scam project**, cancelling everyone's rewards, while others defend the program as legit, emphasizing that users must be legit to get paid.
   - One user shared proof of a friend receiving nearly **$900** but admits to *shitting on Perplexity in their server*, while others tag moderators and express frustration over unanswered emails regarding their bounty.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LoRA Alpha Value Proves Crucial**: A member doing **LoRA finetuning** learned that **LoRA alpha** has to be half of the rank to avoid exploding gradients, while another member highlighted the [Unsloth docs](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) as a *baseline guidance*.
   - The discussion emphasized that optimal **LoRA alpha** settings are case-specific and require experimentation.
- **Nvidia Reveals RTX Pricing and Specs**: Nvidia announced a [72gb RTX5000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/) with **1.4TB** ish bandwidth for around **$5,000**, prompting debate on whether **48gb** is worth the price.
   - A member noted the **6000 Pro** price dropped **25%** in 3-4 months, emphasizing privacy as a key benefit of owning hardware.
- **Community GPU Resources Explored**: A member proposed pooling funds from **28,000** members at **$1** a month to create crazy Unsloth compute infrastructure.
   - Another member joked it would take *666 years for everyone to get a gpu at the end* based on the cost of **Pro 6000** models.
- **Instant Model makes controversial Choices**: Discussion revolved around the **"Instant" model's** Time To First Token (**TTFT**), speculating if its speed is linked to shorter thinking time, or a completely different model.
   - One member speculated that the goal is to have *one model that handles the choices internally instead of you picking 5 different models* which was not welcomed by all.
- **DeepSeek OCR API gets Deployed**: A member announced the availability of a tool to deploy and run inference on your own **DeepSeek OCR model** with just a few steps, available on [GitHub](https://github.com/neosantara-xyz/deepseek-ocr-api).
   - The tool uses **Unsloth** for inference, extracting images from URLs or base64 encoded data.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5.1 Debuts with Enhanced Intelligence and Controversy**: **GPT-5.1** is rolling out with enhanced intelligence and conversational abilities, detailed in [this OpenAI blogpost](https://openai.com/index/gpt-5-1/), however some users find it a *step backwards* and handles [custom instructions](https://platform.openai.com/docs/gpt-best-practices) poorly.
   - A [Reddit AMA](https://redd.it/1ovkt6n/) will be held tomorrow at 2PM PT to discuss **GPT-5.1**, while some users report it *over summarizes* and *repeats scenes* in story arcs, leading some to revert to **GPT 4.1** for storytelling.
- **Gemini 3.0 Quietly Rolls Out to Pro Subscribers**: Google has initiated a *quiet rollout* of **Gemini 3.0 Pro** to Pro subscribers via **Gemini Drops** (monthly feature updates), focusing on developer tools and enterprise/Workspace applications according to [Google's blog](https://blog.google).
   - Free users will receive the update in the *coming weeks*, as shown in a [YouTube link](https://www.youtube.com/watch?v=0-CbsNB9tdk) and [other links](https://marketingtrending.asoworld.com).
- **ChatGPT Canvas Mode Plagued by Bugs and Crashes**: Users report that **ChatGPT**'s website crashed and bugged when using canvas mode, but one user was able to fix long conversations with a *tempermonkey script*.
   - This experience varied among users, with some encountering issues even in short conversations.
- **Quest for Free, Unlimited AI Video Generation**: Users are debating the feasibility of free, unlimited AI video generation, noting that **Grok** offers free AI video but with limited duration.
   - The consensus is that the significant power and processing requirements explain why unrestricted access is typically reserved for Pro subscribers.
- **Tighter Guardrails on Image Creation Spark User Discontent**: New [guardrails for image creation](https://openai.com/policies/usage-policies) are viewed as excessively restrictive, preventing simple depictions and frustrating users who can't edit text.
   - Users lament the loss of features and express dissatisfaction with the current capabilities.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter sun sets Polaris Alpha**: The **Polaris Alpha** model, an early version of **OpenAI's GPT-5.1** without reasoning, is now deprecated, being replaced by a faster, more efficient **GPT-5.1** with adaptive reasoning and better coding, according to [OpenAI](https://openai.com/index/gpt-5-1-for-developers/).
   - OpenRouter has launched **three more GPT-5.1 models**: [GPT-5.1 Chat](https://openrouter.ai/openai/gpt-5.1-chat) (aka Instant in ChatGPT), [GPT-5.1-Codex](https://openrouter.ai/openai/gpt-5.1-codex), and [GPT-5.1-Codex-Mini](https://openrouter.ai/openai/gpt-5.1-codex-mini-20251113).
- **Privacy Settings Untangled**: A user resolved API errors by enabling the privacy toggle, *"Enable free endpoints that may train on inputs / publish prompts"*, in their [OpenRouter privacy settings](https://openrouter.ai/settings/privacy).
   - This setting appears to impact the availability of certain models.
- **API Rate Limit struggles continue**: Users reported frequent **Error 429** (rate limit) issues with the API, especially with `anthropic/claude-sonnet-4` and `anthropic/claude-sonnet-4.5`, indicating supply constraints.
   - Some encountered Cloudflare errors and timeouts, despite the [OpenRouter status page](https://status.openrouter.ai/) reporting no incidents.
- **GPT-5.1 Token Output Debated**: **GPT 5.1** is rolling out on ChatGPT, with early benchmarks suggesting nearly **2x token output** for minimal improvements, sparking concerns about cost-effectiveness.
   - One user quipped, *"we want less thinking, not more"*.
- **React Compiler Saves Day**: A member stated they started using **React Compiler** which addresses the *React Slop* issue, stating that *it rules*
   - Another member jokingly questioned the seriousness of the statement.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Helion Harmonizes Backwards Compatibility**: **Helion** will be **BC compatible** (e.g. from **0.2** to **0.2.1**), with version **0.2.2** just released and minor releases planned every week or two to keep **pypi** packages updated, see [issue 164](https://github.com/pytorch/helion/issues/164).
   - In contrast to **Triton**, which is slow in interpret mode, **Helion's** interpret mode is noted to be surprisingly fast using eager **PyTorch**; moreover, passing `configs=` now enables **autotuning** similar to **Triton's autotune**.
- **NCU Scores Major Points with Cloudies**: Cloud vendors are now graded on supporting **NCU** (**NVIDIA Compute Unified Device Architecture**), as reported in [Semianalysis](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard).
   - The push for **NCU** support reflects its growing importance in the cloud computing landscape.
- **NVFP4 Hackathon Pumps GEMV Optimizations**: A hackathon to optimize workloads for **NVFP4** datatype on the **Blackwell GPU** architecture has been launched, with the first challenge focusing on GEMV (Matrix Vector multiplication), see [blogpost](https://veitner.bearblog.dev/nvfp4-gemv/).
   - The **Datacrunch B200 GPU** provides access with a recommendation of **CuTeDSL** for its ability to get close to hardware while maintaining productivity.
- **Open Source AI Coding Agent Joins Forces**: An AI coding agent setup, designed to ship high-quality code, has been open-sourced, featuring skills, **114+ sub-agents**, and an executive planning framework for complex features; it can be found on [GitHub](https://github.com/flora131/agent-instructions) and in [this blog post](https://alexlavaee.me/blog/ai-coding-infrastructure/).
   - The setup is intended to enhance existing AI coding tools and workflows.
- **Restrict Qualifier Raises Red Flags**: A member shared an interesting [example](https://godbolt.org/z/ad98nYdrf) of a potential **GPU compiler bug** involving the `__restrict__` qualifier, asking for opinions on whether it's a true bug or simply **undefined behavior (UB)**.
   - The code example includes **PTX** and **SASS** outputs, inviting deeper analysis of the compiler's handling of memory aliasing constraints.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-5.1-Codex Excites Cursor Engineers**: Members reported that **GPT-5.1-Codex** is rolling out, with some spotting it in the [latest codex alpha](https://x.com/OpenAIDevs/status/1986861734619947305) and noticing its integration with [Windsurf](https://www.windsurf.ai/).
   - Some users have not received an update prompt, while others find it already available in Codex before a general OpenAI API release, indicating a staged rollout.
- **Auto Mode Performance Plummets Post-Release**: Users are reporting a noticeable degradation in **Auto mode performance**, with one stating that *20 minutes ago the Auto mode was using another faster model, now it uses gpt 5 codex and it is slower and can't even edit a file when a model without reasoning can do it without issue*.
   - Some suggest this might be due to **Auto mode** defaulting to potentially overloaded OpenAI servers after the **5.1 release**, while others speculate Cursor is using a GPT-5 version with high thinking requirements.
- **Custom Commands Cut Costs**: Members discussed the use of **custom commands** for automating tasks, such as running tests after code changes, where a user shared the approach of creating a custom command with **CTRL + K** to trigger specific instructions.
   - A member suggested ensuring tests are triggered automatically by executing specific commands, which can be tagged for increased token efficiency with the [docs section in Cursor settings](https://cursor.com/docs/customcommands).
- **Memory Feature Misgivings Manifest**: Cursor's **memory feature** is prompting mixed reactions; one user running in legacy privacy mode is hesitant to disable it.
   - One user described that *memories are usually within your profile and they need to be active, if you're in privacy mode this tool calling will fail*, so this is a feature to keep an eye on if you want the best out of cursor.
- **Serialization Snafus Stymie Terminal Tasks**: Users report a *Serialization error* when using terminal commands, which breaks the chat until Cursor is restarted, with one user reporting that the source of the issue comes from commands that contain spaces in the file paths.
   - Members pinpoint the issue to errors in serializing terminal output fed to the LLM, and shared a link to the [Serialization Error forum post](https://forum.cursor.com/t/serialization-error/124671).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Slashes API Prices for Hermes 4**: Nous Research announced a **70% price cut** for their **Hermes 4 70B** and **Hermes 4 405B** APIs, aiming to make access more affordable.
   - Details and signup are available at the [Nous Research Portal](https://portal.nousresearch.com/) and on [X](https://x.com/NousResearch/status/1989077400957911394).
- **Community Craves Schizomax Models**: Members discussed the need for a **schizomax model**, noting that **Grok** and **Nous models** rarely refuse requests, unlike increasingly restricted **GPT models**.
   - Users expressed frustration with the corporate influence and wellness checks limiting the usefulness of **OpenAI's GPT models**.
- **Hermes4 Emerges as Code Model on Cline**: Users observed that **Hermes4** is now featured as a code model on Cline, sharing screenshots of its prompt interface and capabilities.
   - This development generated excitement about the evolving applications of **Hermes4** in code-related tasks.
- **Challenges arise importing GGUF files into Nous Chat**: A user inquired about importing **GGUF** files into **Nous Chat**, however this function is currently unsupported.
   - Members recommend running **GGUF** files locally using [llama.cpp](https://huggingface.co/docs/hub/en/gguf-llamacpp) or [Ollama](https://ollama.com/) for a simpler setup.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Sparkling Lakh MIDI Dataset Arrives**: A fully cleaned and organized **Lakh MIDI Dataset** is available, featuring a structured **JSON file** with over **44,000 entries**, and plans to upload it to [Hugging Face](https://huggingface.com/).
   - The dataset boasts complete parsing and consistency, inviting collaboration and enhancements from the community.
- **French Wikipedia Gets Scrubbed, Now on HuggingFace**: A user uploaded a cleaned **French Wikipedia DB** version to [Hugging Face](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json), containing over **2,700,000 files** in **JSON format**, and the English version is next.
   - The cleaning process includes more than just plain text, encompassing *templates, tables, html, and refs*, while retaining infobox information and links.
- **JSONL Claims Throne for Textual Data**: For purely textual data, using **JSONL/NDJSON** is preferred, as it simplifies processing by allowing line-by-line reading, unlike tar files with their cumbersome headers.
   - In dataset format discussions, the member highlighted the ease of use of **JSONL** compared to managing tar headers.
- **NVIDIA Dataset License Draws Fire**: Concerns are mounting over the restrictions in **NVIDIA's dataset license**, notably regarding training, evaluation, and public result sharing, detailed in [an X thread](https://x.com/goodfireai/status/1986495330201051246).
   - The primary worry centers around **NVIDIA's** right to terminate the license anytime, potentially nullifying granted permissions, leading to legal uncertainties.
- **Anthropic Takes Heat on China Policies**: A member accused [Anthropic](https://x.com/AnthropicAI/status/1989033793190277618) of **fear-mongering** to gain strategic advantages, especially against **non-US** and **Chinese labs**.
   - Questions are being raised about **Anthropic's data privacy practices** and the potential prioritization of safety over privacy, along with scrutiny of **Anthropic's CEO's stance on China**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo May Embrace Privacy, Eventually**: **Mojo** might eventually support **public and private members/methods**, but not until there's an *escape hatch* for breaking encapsulation.
   - Currently, **Mojo** uses **Python's underscore convention** to suggest privacy, and it is best for *compute-heavy things* lacking a comprehensive ecosystem.
- **Modular Stack Stacks Regression Rigmarole**: Using the **Modular tech stack** for a regression model currently requires building parsers and data visualization libraries, with **MAX** handling the backwards pass for training.
   - The fastest approach is to train with **Torch** and perform inference with **MAX**.
- **`comptime` Comprehends Keyword Capabilities**: The `comptime` keyword in **Mojo** now covers what `alias` used to, including type assignments, enabling zig-style static reflection like `comptime InnerTypeFirstField = T.fields[0].type`.
   - Although *it reads a bit off for type assignments*, having different keywords can be annoying when mixing types and values aggressively at compile time.
- **Apple Silicon Support Seeds Supercomputer Dreams**: Expanded support for many intrinsics and enabled more tests and GPU puzzles thanks to community PRs, and basic **MAX graphs** have started working.
   - One member reported that *the dream of developing kernels locally and deploying on a supercomputer* is becoming a reality, thanks to the **GPU puzzles**.
- **HipKittens Highlight Kernel Hiccups**: The [HipKittens paper](https://arxiv.org/abs/2511.08083) indicates that **Mojo’s MHA kernel** suffers from expensive bank conflicts, achieving only 50% of the peak kernels’ performance on the **MI355X**.
   - A member suggested that if **LLVM** can talk to a device, abstractions can be built at compile time, potentially reducing the need for AMD/NVIDIA-specific kernels.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Whisper Still Rules for Transcription**: According to a member, **Whisper** remains the top open-source choice for transcription, particularly using the [large-v3 model](https://openai.com/blog/whisper) quantized to **8 and 4 bits** via **Whisper-server**.
   - It was noted that **Whisper.cpp** with Vulkan support improves portability, unlike running **Whisper** directly through PyTorch in Python.
- **ICLR's Review Quality Sinks**: Members observed a decline in **ICLR** review quality, citing *lots of llm reviews with the prompts left in* and suggested longer submissions with *pre-review rejection for papers that are clearly trash*.
   - One member stated that in comparison *NeurIPS was better*.
- **SIMA-2 Enters the Virtual Stage**: A member shared [DeepMind's SIMA-2](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/), an agent that *plays, reasons, and learns* in virtual 3D worlds.
   - They questioned if copyright striking works for these models if they generate images with textures of existing games.
- **Seesaw Preferences Baffle Recommenders**: A member discussed intentionally **seesawing** their preferences to confuse recommendation algorithms, linking [this paper](https://arxiv.org/abs/2511.08378).
   - The user reported that the link was shared in an X thread as an identification of the linked paper.
- **GPT-5's Conversational Style for Token Spend?**: A discussion was had around whether **GPT-5's** *more conversational* style aims to lure **GPT-4** users into spending more on [output tokens](https://openai.com/index/gpt-5-1/).
   - One member speculated about potential restrictions on output tokens across different user tiers, particularly for the free version.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Constrained on Frontend Skills**: While finding **Claude**'s backend skills effective, a member noted its frontend skills were only *okayish*, still making mistakes, referencing [this blog post](https://www.claude.com/blog/improving-frontend-design-through-skills).
   - The member suggested that **Claude** still has room for improvement in frontend development tasks.
- **Latent Space Spotify Feed Briefly Fails**: The Latent Space **Spotify feed** experienced issues due to a copyright claim on its royalty-free intro song, prompting concern among listeners.
   - It was later reported that [Spotify resolved the issue](https://x.com/paraga/status/1988729121636294682/photo/1), restoring the feed to normal.
- **Parag's Parallel Web Secures $100M**: **Parag Agrawal**'s new company, **Parallel Web Systems**, obtained **$100 million** in Series A funding to develop a web infrastructure for AIs.
   - Enthusiasts praised the company's sleek design, as showcased in [this announcement](https://xcancel.com/paraga/status/1988729121636294682/photo/1), anticipating its potential impact on AI development.
- **Anthropic Announces $50B Data Center Build**: **Anthropic** plans to invest **$50 billion** in US datacenters across Texas and New York, stimulating construction jobs and discussions about domestic compute capacity.
   - The announcement, found [here](https://xcancel.com/anthropicai/status/1988624013849935995?s=46), raised debates over staffing, environmental concerns, and potential AI industry bubbles.
- **Holo2 Cheaper and Faster than GPT-4V**: **HCompany** introduced **Holo2**, a more economical **30B-MoE** vision model based on **Qwen3-VL**, exceeding **GPT-4V** in UI benchmarks and operating on web, desktop, and mobile platforms.
   - Further details are available in [this announcement](https://xcancel.com/hcompany_ai/status/1989013556134638039), highlighting its performance and versatility.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Upscaled Model Sparks Laughter**: An **upscaled model** with finetuning became a source of amusement, prompting users to share a [Beavis and Butthead GIF](https://tenor.com/view/beavis-butthead-beavisandbuttheadshow-gif-13367061995602198225) as a humorous reaction.
   - The lighthearted banter included references to specific Discord channels, adding to the comedic tone of the discussion.
- **Multi-Head Attention Explained**: A user's question about why **multi-head attention** heads don't learn the same thing was addressed, citing random initialization and [softmax exaggerating differences](https://huggingface.co/datasets/John6666/forum2/blob/main/multi_head_attention_1.md) as key factors.
   - Separate parameters and distinct gradients further contribute to the natural divergence of heads, with the learning task preserving useful differences.
- **HuggingChat Goes Paid, Users Frown**: Members expressed disappointment over **HuggingChat's** shift to a paid model, noting the limitation of free features compared to the previous unlimited version, as exemplified in a [screenshot](https://cdn.discordapp.com/attachments/879548962464493619/1438593728200572928/Screenshot_2025-11-13-18-16-43-962_mark.via.gp-edit.jpg?ex=69181b10&is=6916c990&hm=7bc833942ddb303310bdca35fdf5e940a15a9eb9a345cbc6086e0a21f8e817c6&).
   - One user lamented the lackluster new version, questioning if Hugging Face's commitment to providing an open-source AI platform for free is wavering.
- **AI Voice Implementation Enthuses Community**: Enthusiasm arose around implementing **AI voice**, including ideas for integrating voice into a *smutty sexbot idea*, with a shared [open-source voice design](https://huggingface.co/maya-research/maya1) featuring 20 human emotions.
   - A user wondered about the implementation difficulty and potential integration into an Android app.
- **Propercode Promises Production Code**: A member introduced **Propercode**, a multi-agentic coding CLI tool for codebases that is powered by **Pydantic AI** agents orchestrated as a graph, with the goal of having reliable coding accuracy ([GitHub](https://github.com/JaiSuryaPrabu/proper-code)).
   - The tool is currently in **v0.1** and will provide multi-tools and modes for agents like autonomous, learning guide etc.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Misunderstood as Prompt-Free Framework**: Members discussed that **DSPy** is being perceived as a framework where prompting LLMs is unnecessary, which is misleading because prompting is still needed for domain-specific applications, even with examples and **GEPA**.
   - One member clarified that **DSPy** encapsulates prompting as a programmable module for optimization, with signature instructions or docstrings acting as prompts.
- **Pydantic Models Trump Simple Strings for Signatures**: A member voiced their preference for **Pydantic models** over simple string types (`input -> output`) for signatures, citing the need for more complex and type-checked implementations.
   - They noted that instructions within signatures serve as prompts, highlighting community confusion arising from varied interpretations of *prompts*.
- **Regex Rises for Robust Agentic Search**: To improve **Agentic search**, a member instructed the LLM in their **ReAct module** to use specific terms for tool search via **ripgrep**, with **Regex** as a fallback.
   - This instruction was essential for the LLM to effectively use Regex in the search tool, especially when accessing multiple tools (3-4 functions).
- **Survey Language Sparks Suspicion**: A member suspected an issue with **survey language** based on a screenshot highlighting **fine-tuning** at the top.
   - Another member described the survey language as *insane* for being buried in the appendix with fine-tuning highlighted prominently.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K-2 Partners with Together AI**: The **Moonshot AI** team announced a deep dive into **Kimi K2 Thinking** in partnership with **Together AI**, scheduled for **Nov 19, 2025** at **9AM PT**, with registration at [luma.com/g5qcq85z](https://luma.com/g5qcq85z).
   - The announcement highlights the power of **1T MoE** (Mixture of Experts) and its ability to execute **300 tool calls in a single run**.
- **GLM-5, K3, R2, Model Releases Looming**: Members anticipate the arrival of **GLM-5**, **K3**, and **R2**, with one stating *I am not hyped at all for models like Gemini 3, since they dont offer a good coding plan like Kimi/GLM*.
   - Despite int4 optimization, **Kimi K2-thinking** is reportedly **1.8x** slower than **GLM-4.6** and **2.5x** slower than **Minimax m2.wysh.3**, but more capable for non-pure coding tasks.
- **YC's Kimi For Coding Sparks Debate**: The **Y Combinator**-backed **Kimi For Coding**, criticized as *daylight robbery* for a mediocre model, gives **2048** weekly usage, according to [this tweet](https://x.com/ycombinator/status/1988366241460089118?s=46).
   - One user stated *I can’t believe they think such a product must exist*.
- **Moonshot AI Joins Chipmaking Race**: Following US AI labs, Chinese AI labs are entering chipmaking; users anticipate **Moonshot K100**, referencing [this tweet](https://x.com/tphuang/status/1988952992003891330?s=46).
   - Users questioned the absence of project and custom instruction capabilities on the **Kimi Chat** website, while others clarified that **tool use** is automatically enabled on **Claude Code** or **Kimi-CLI**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-5.1 System Card Draws Fire**: The release of **GPT-5.1** is raising eyebrows because of its missing benchmarks, with some calling its [system/model card](https://cdn.openai.com/pdf/4173ec8d-1229-47db-96de-06d87147e07e/5_1_system_card.pdf) a *joke*.
   - Skeptics point to the lack of an API as suspicious, suggesting it might just be revised system prompts rather than a whole new model.
- **Aider-ce maintains the code**: **Aider-ce** is garnering praise for its existing features, but there are concerns regarding communication and succession planning from its maintainer.
   - Despite these concerns, the branch is being well-received by the community, which sees new users join daily.
- **Deepseek API accelerates Agent Mode**: Users report that the **Deepseek API** significantly improves the performance of **Aider-ce**'s agent mode, which runs slow on **GPT-5-high**.
   - The slowness is attributed to the regeneration of the repo map on larger repos, suggesting adjustments to `repo_map_max_files` and `repo_map_token_budget`.
- ** moonshotai Kimi-K2 runs on Aider**: Users fixed a *404 no cord found* error when running aider with `moonshotai/Kimi-K2-Thinking` by setting **OPENAI_API_BASE** variable to `https://llm.chutes.ai/v1/`.
   - The correct commands are `SET "OPENAI_API_BASE=https://llm.chutes.ai/v1/"` and `SET "OPENAI_API_KEY=mykey"` before running `aider --model openai/moonshotai/Kimi-K2-Thinking`.
- **DeepSeek has Commercial Privacy Concerns**: A member reported being *pretty much sold on* **DeepSeek** for everything, using the **DeepSeek API** only for open source or personal projects.
   - Another member inquired about services offering **DeepSeek** with respect for commercial privacy, especially since the first member indicated they use the raw **DeepSeek API**, and don't have commercial requirements.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP Server Asks About Query Result Formats**: A member building a **Postgres MCP server** is seeking advice on the optimal output format for the **executeQuery MCP tool**, considering **JSON**, **Markdown**, or **Toon** formats for returning **SQL query results**.
   - They are exploring the best way to present data from **SQL queries** in a user-friendly manner.
- **MCP's Protocol Scope Clarified**: Members clarified that while the **MCP protocol** doesn't natively support file transfer from client to server, it is possible via **resources**.
   - General **MCP implementation discussions** should use a [GitHub Discussion](https://github.com/orgs/modelcontextprotocol/discussions) or other community Discord servers, as this Discord server is for discussing the protocol and related official SDK projects.
- **SE Upload Call Implementation Questioned**: A user inquired about uploading files directly from a chat interface via an **SE upload call** using tools, specifically addressing [issue 1306](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1306).
   - They were directed to open a [GitHub Discussion](https://github.com/orgs/modelcontextprotocol/discussions) or ask in other community Discord servers because this Discord server is for discussing the **protocol and related official projects**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Support Encountering Hurdles**: Users report that **Manus' checkpoint system** is unable to find git commits, which blocks the *'Publish'* button, forcing them to manually sync the internal repository.
   - After inquiring about the removal of chat mode, they were directed to [Manus feedback](https://manus.im/feedback) for support.
- **Manus Launches New Credit Plans**: The $19 plan now includes **4000 credits** monthly, a substantial increase from the previous **1900 credits**.
   - Users are curious whether credit usage has been altered alongside this adjustment.
- **AI/ML Engineer Joins the Fray**: An **AI/ML engineer** specializing in model design, optimization, and large-scale deployment has joined the server with tech including **Python, C++, Rust, SQL, Hugging Face, ONNX Runtime, Triton Inference Server**, and **Apache Spark**.
   - They are proficient in deploying models at scale.
- **Workflow Automation Engineer Open for Business**: An engineer experienced in **workflow automation, LLM integration, RAG, AI detection, image, and voice AI**, is offering their services.
   - They have constructed automated pipelines and task orchestration systems using **Dspy, OpenAI APIs**, and custom agents, and have shared [their portfolio](https://devx-green.vercel.app/).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **OpenCL Devices Error Fix Proposed**: A user suggested a fix to raise a `RuntimeError` when no **OpenCL Devices** are found, referencing [OpenCL documentation](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceIDs.html) regarding `CL_INVALID_VALUE` errors.
   - This addresses a scenario where the code proceeds without devices, potentially causing issues.
- **pmap and vmap Functionalities Enhancing tinygrad**: A user updated the [README](https://github.com/tinygrad/tinygrad) to emphasize the need for **pmap** and **vmap** functionalities, which enable more efficient array manipulations.
   - The user leveraged **ChatGPT** for assistance on the document update.
- **torch_load Now Compatible with VGG16 Model**: The pull request ([PR #13253](https://github.com/tinygrad/tinygrad/pull/1325)) enables the **torchvision** hosted **VGG16** model to work with `torch_load`, increasing the number of usable models.
   - This enhancement broadens **tinygrad's** compatibility with pre-trained models from the **torchvision** library.
- **OpenPilot PR Merged, Expanding Integration**: The [OpenPilot PR](https://github.com/commaai/openpilot/pull/36615) has been merged, enhancing integration or compatibility between **tinygrad** and **OpenPilot**.
   - The team committed to preventing regressions, showing the importance of integration.
- **Interest Surges for tinygrad with C++ Integration**: A user inquired about using **tinygrad** with **C++**, with the intention of using it in embedded systems.
   - This request suggests a growing interest in leveraging **tinygrad** in resource-constrained environments.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Unleashes GPT-5.1 and GPT-5.1-Codex**: **GPT-5.1** and **GPT-5.1-Codex** are now live in Windsurf, with all paid users getting free access for the next 7 days, and **GPT-5.1** set as the default model for new users, detailed in [this announcement](https://x.com/windsurf/status/1989069991770214580?s=20).
   - Windsurf touts **GPT-5.1** as a marked improvement over **GPT-5** for agentic coding and frontend design, with users able to download the editor [here](https://windsurf.com/download/editor).
- **GPT-5.1 Supercharges Agentic Coding**: According to Windsurf, **GPT-5.1** represents a tangible upgrade from **GPT-5**, especially in the context of agentic coding tasks.
   - The model adaptively modulates reasoning depth contingent on task intricacy, resulting in quicker turnaround times for the majority of tasks.



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





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1438257054333534290)** (1217 messages🔥🔥🔥): 

> `GPT 5.1, Gemini 3 release, Code Arena Updates, Riftrunner status, Open Source Models` 


- **GPT-5.1 causes confusion and debate**: Members debated [GPT-5.1's performance](https://openai.com/blog/new-models-and-developer-products), with some claiming it outperformed **Sonnet 4.5**, while others deemed it "trash" due to hardcoded UI instructions.
   - Some pointed out its improved reasoning and coding abilities, while others felt it lacked creativity and wasn't a significant upgrade, with one member describing it as *"a non-release filler"*.
- **Gemini 3 Arrival Date Speculation Ramps Up**: Speculation continues regarding the release of **Gemini 3**, with some anticipating a launch this or next week, possibly overshadowed by the announcement of **Sima 2 research**.
   - Discussion points to a potential delay due to **Kimi2** and the possibility of a staggered release, starting with Gemini Enterprise or a mobile version on Canvas, with no release on Friday.
- **Code Arena undergoes changes**: Discussion revolves around changes to [Code Arena](https://lmarena.com), specifically the removal of the retry button from battle mode due to perceived abuse, stirring frustration among users.
   - There are also questions surrounding the status of **Riftrunner** on the platform, with reports of errors and potential removal, as well as concerns about Cloudflare timeouts while browsing chat history.
- **Open Source Models Spark Excitement**: Some members expressed interest in open-source models like **GLM 4.6** and **Grok**, highlighting their potential as cost-effective alternatives and praising their coding capabilities.
   - The owner of [Hugging Face](https://huggingface.co/) believes *"the world belongs to open source model"*, in a related note, a memeber has shared a [YouTube tutorial](https://www.youtube.com/watch?v=GZqYr8_Q7DE) for using Open Empathic.
- **Gemini Canvas version spurs debate**: Users have varying experiences with the [Gemini Canvas](https://aistudio.google.com/app/canvas) version, with some praising its fire UI while others are not as impressed by it.
   - Some users are trying to determine if it routes to **Gemini 3**, while others think it is still the 2.5 Pro version, it is all a *mass delusion*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1438618401772671191)** (1 messages): 

> `GPT-5.1 Release, LMArena Updates` 


- **GPT-5.1 storms onto LMArena**: A new model, **gpt-5.1**, has been added to the Text, Vision, and Code Arena on [LMArena](https://x.com/arena/status/1989058785927950628).
- **LMArena gets a refresh**: The Text, Vision and Code Arena on **LMArena** got a new model today.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1438608979776245853)** (1 messages): 

> `GPT 5.1, Perplexity Pro, Perplexity Max` 


- **GPT 5.1 Rolls Out to Perplexity Subscribers!**: **GPT 5.1** is now available for **Perplexity Pro** and **Max** subscribers, as announced in the Discord channel.
- **Image Attached: GPT 5.1 Features?**: An image was attached to the announcement, potentially showcasing the new features or capabilities of **GPT 5.1**.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1438257181261697035)** (1054 messages🔥🔥🔥): 

> `Comet Referral Program Issues, Perplexity Pro Billing and Limits, GPT 5.1 rollout, Scam accusations` 


- **Perplexity Partner Program Ban Wave Strikes!**: Many users report being **banned** from the Perplexity Partner Program for alleged "fraudulent activity" despite claiming genuine referrals, with appeals to *partners@perplexity.ai* and Dub support going unanswered since November 11, 2025.
   - Frustrated users speculate about reasons for the bans, ranging from VPN usage by referrals to exceeding referral limits, and a theory emerges that bans are timed around the 30-day hold period for referrals, with some even claiming those with the highest earnings were targeted.
- **Perplexity Pro Users Face Limits and Billing Woes**: Users are reporting issues with Perplexity Pro, including **image generation limitations**, inaccurate reset dates for lab tokens, and deep research tool calls.
   - Specifically, users are saying *Limits reset on the first day of the month if I remember correctly*.  Additionally, issues with google sign-on with comet are surfacing, and users are trying to figure out how to use the different AI models effectively. 
- **GPT-5.1's Deployment Struggles: Is it Actually Superior?**: Members are noting the rollout of **GPT 5.1**, but its unclear if there's an official release or whether people have actually received the new model.
   - Some users are reporting that the new model is similar to **Gemini 3.0** when the Canvas setting is enabled, but also mention its better at answering questions.
- **Scam Accusations Fly Amid Referral Program Chaos**: Amidst the confusion, some users claim that **Perplexity is running a scam project**, cancelling everyone's rewards, while others defend the program as legit, emphasizing that users must be legit to get paid.
   - One user shared proof of a friend receiving nearly **$900** but admits to *shitting on Perplexity in their server*, while others tag moderators and express frustration over unanswered emails regarding their bounty.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1438297132850483374)** (420 messages🔥🔥🔥): 

> `LoRA alpha values, RTX 5000/6000 pricing and specs, Community GPU resources, Intel B60 vs Nvidia 4090` 


- **Optimal LoRA Alpha Tuning Discovered**: A member learned the hard way that when doing **LoRA finetuning** on a base model, the **LoRA alpha** has to be half of the rank, or the grad norm will explode, but others disagree.
   - Another member pointed out that the [Unsloth docs](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) come into play as *baseline guidance* to start tweaking from, depending on the specific case.
- **RTX 5000/6000 pricing and specs revealed**: Nvidia announced a [72gb RTX5000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/) with **1.4TB** ish bandwith for around **$5,000**, but some think 48gb isn't enticing for that price.
   - One member pointed out that the **6000 pro** price has dropped **25%** in 3-4 months, with the main benefit of owning your hardware is privacy.
- **Community GPU resources get proposed**: A member suggested the crazy idea that if all **28,000** members put in **$1** a month, then Unsloth would have crazy compute infra.
   - Another member humorously pointed out *8000 per Pro 6000 means every month 3.5 gpus, so in 666 years, everyone will get a gpu at the end*.
- **Intel B60 underperforms NVIDIA RTX 4090**: Discussion emerged around the **Intel Arc Pro B60**, with initial comments suggesting it's about half the performance of a **4090**, while another member claiming it has between *1/6th and 1/7th the compute*.
   - Further investigation revealed that the B60 and the B580 are really near, but the drivers are spotty, and is not hassle free like Nvidia.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1438709656573968496)** (3 messages): 

> `workflow automation, LLM integration, RAG pipelines, AI content detection, image AI` 


- **Engineer Specializes in AI and Blockchain Solutions**: An engineer specializes in **workflow automation, LLM integration, RAG, AI detection, image and voice AI, and blockchain development**, showcasing a strong track record of real-world implementations.
   - They've built automated pipelines using **Dspy, OpenAI APIs, and custom agents**, including a support automation system that reduced response times by **60%**.
- **Advanced RAG Pipeline Architect**: The engineer designed and deployed advanced **RAG pipelines**, combining vector databases, hybrid search, and custom retrieval logic for accurate responses in production environments.
   - They also developed tools for AI content detection on a moderation platform, using **stylometric analysis, embedding similarity, and fine-tuned transformers**.
- **AWS Lambda Image AI Tagging and Moderation Pipeline**: The engineer created a tagging and moderation pipeline using **CLIP and YOLOv8 on AWS Lambda and S3**, classifying and filtering thousands of images daily for an e-commerce platform.
   - In voice AI, they built a **voice cloning and transcription service** using Whisper and Tacotron2, enabling personalized voice assistants through ASR, TTS, and CRM integration.
- **Blockchain Expertise and Smart Contract Development**: The engineer has deep expertise in **blockchain technology**, including smart contract development (**Solidity and Rust**), decentralized application architecture, and secure on-chain/off-chain integrations.
   - They focus on delivering scalable, production-ready AI and blockchain systems, covering the entire lifecycle from model selection and fine-tuning to backend integration and deployment.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1438262436825137356)** (266 messages🔥🔥): 

> `GPT-5-1 em dashes, Instant model's TTFT, Model autopicking, Electroglyph's AI usage, Yukiarimo's bad and good AI` 


- **GPT-5-1 still sports em dashes**: A member noted that the [GPT-5-1 announcement page](https://openai.com/index/gpt-5-1/) continues the tradition of using **em dashes** in every example.
   - Another member jokingly remarked on the *inevitability* of this stylistic choice.
- **"Instant" Model's TTFT**: Discussion revolved around the **"Instant" model's** Time To First Token (**TTFT**), with members suggesting its speed is linked to shorter thinking time and potentially a completely different model.
   - One member speculated that the goal is to have *one model that handles the choices internally instead of you picking 5 different models and 5 thinking modes* but another stated *I dont like it making decisions for me on what model to pick*.
- **Electroglyph Reveals AI Usage**: A member shared their diverse AI toolkit, including **Hanasu** (TTS), **Yuna** (VLM), **Gemini 2.5 Pro**, and **Grok 4**, for tasks ranging from audiobooks to deep research, and *dumb memes*.
   - In response, another member jokingly pointed out the discrepancy between expressing disdain for AI and being an active user: *you use AI a lot for somebody who hates it so much =)*.
- **Yukiarimo Divides AI into Good and Bad**: A member outlined their categorization of AI, deeming content creation models (**Veo, Imagen, ElevenLabs**), **diffusion models**, **closed-source models**, and **agents** as inherently *bad*.
   - They expressed a preference for **upscaling**, **interpolation**, **small models**, and anything trained *from scratch*, advocating for models that can undergo *catastrophic forgetting* (specifically calling out **Liquid AI**).
- **Reality-Exact Data Capture Proves Elusive**: A member inquired about capturing and saving data with *reality-exact precision*, including **360 photons+lidar** images and **360 atoms fluctuations** for audio.
   - Others pointed out the theoretical and practical impossibilities due to the **uncertainty principle** and limitations in measurement and processing capabilities.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1438257737673871480)** (39 messages🔥): 

> `CPU offloading performance, GGUF models and LM Studio, vLLM for batch inference, Fine-tuning vs RAG, Qwen 2.5 VL 3B fine-tuning` 


- **CPU Offloading Impedes Performance Insanely**: Members noted that **CPU offloading** drastically reduces performance, although **MoE models** can maintain acceptable speeds depending on the configuration and model choice.
   - For newcomers, it was recommended to use **GGUF** models with **LM Studio** for easy inference and choose quantization based on system capabilities.
- **Unsloth GGUF Surpasses Standard Quantization**: **Unsloth GGUF** models include general improvements and performance fixes for accuracy.
   - Some models feature **Unsloth dynamic quantization**, achieving higher accuracy than other quantized formats.
- **RAG vs Fine-tuning Fight**: It was discussed that fine-tuning and RAG serve distinct purposes; combining both is ideal for retrieving knowledge/docs, more information can be found [here](https://docs.unsloth.ai/get-started/beginner-start-here/faq-+-is-fine-tuning-right-for-me).
   - Fine-tuning can also be better if you want to 'teach' the model how to accomplish a specific task.
- **Qwen 2.5 VL Fine-tuning: Struggles and Solutions**: A user sought assistance with fine-tuning **Qwen 2.5 VL 3B**, facing issues where training and validation losses remained similar despite overfitting attempts, potentially caused by the incorrect template.
   - Community member noted that seeking paid services are not allowed in the channel, only Unsloth-related questions are.
- **Unsloth and Llama.cpp/Ollama Output Divergence**: A user reported discrepancies between **Unsloth** outputs and **llama.cpp/Ollama** when loading a fine-tuned **Llama3.2 3B** model, despite replicating chat templates.
   - It was suggested that differences in model parameters beyond the seed could be influencing the output.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1438546164348752044)** (3 messages): 

> `DeepSeek OCR API, Unsloth Inference, Qwen 3 1B Zero Dark Horror` 


- **DeepSeek OCR API Deployment**: A member announced the availability of a tool to deploy and run inference on your own **DeepSeek OCR model** with just a few steps.
   - The tool is available on [GitHub](https://github.com/neosantara-xyz/deepseek-ocr-api) and uses **Unsloth** for inference, extracting images from URLs or base64 encoded data.
- **Qwen 3 1B Gets the Horror Treatment**: **Unsloth** now supports [Qwen 3 1B](https://huggingface.co/Qwen/Qwen-1_8B) and quantizing for a *horror dataset*.
   - The weights are on [HuggingFace](https://huggingface.co/DavidAU/Qwen3-Zero-Dark-Horror-LIGHTSPEED-1B-HRR-imatrix-GGUF) and support **16/32bit training**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1438423334957088790)** (4 messages): 

> `Multi-Head Attention, Random Initialization, Modular Learning in Heads, Mixture-of-Experts` 


- **Multi-Head Attention Debated**: A member inquired about why different heads in **multi-head attention** don't end up learning the same thing, despite embeddings being divided equally.
   - The most common reason found was that **random initialization** creates initial differences, and **softmax** exaggerates those differences.
- **Modular Learning elusive in Neural Networks**: A member questioned what it means for heads to learn the *same thing*, suggesting that forcing modularity (e.g., one head for math, another for coding) hasn't been successful.
   - They added that this issue is similar to **Mixture-of-Experts**, where experts don't become truly modular and simply add another dimension of scaling, linking to [OpenAI's research on sparse circuits](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/).
- **Heads learn properties, not strict concepts**: A member agreed that heads don't learn strictly one thing, but *abstractly they mostly learn just one thing or some properties per head*.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1438273461675102361)** (4 messages): 

> `GPT-5.1 Release, AI Model Training, ChatGPT Group Chats` 


- ****GPT-5.1** Arrives with Smarts and Sass!**: The **GPT-5.1** is rolling out to all users this week, promising enhanced intelligence, reliability, and conversational abilities, detailed in [this OpenAI blogpost](https://openai.com/index/gpt-5-1/).
   - There will be a [Reddit AMA](https://redd.it/1ovkt6n/) tomorrow, 2PM PT, regarding **GPT-5.1** and customization updates.
- **Transparent AI: Sparse Circuits for the Win**: OpenAI has pioneered a new training method for small AI models, emphasizing internal mechanisms that boost human understanding, elaborated in [this blog post](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/).
   - This approach aims to demystify the intricate structures of language models such as **ChatGPT**.
- **ChatGPT Circles Up: Group Chat Pilots Launch!**: **ChatGPT** group chats are now piloting in Japan, New Zealand, South Korea, and Taiwan, according to [this announcement](https://openai.com/index/group-chats-in-chatgpt/).
   - The new feature offers a novel way to collaborate with friends, family, or coworkers within the same **ChatGPT** conversation.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1438268047982067772)** (390 messages🔥🔥): 

> `GPT 5.1, Gemini 3.0, AI Video Generation, Canvas mode, Model Merging` 


- **GPT 5.1 Extends Thinking, Aids Rapid Studying**: Members found **GPT 5.1**'s extended thinking mode usable for rapid studying and compared it to *a longer extendable stick that better adjusts its reach to get to harder topics.*
   - Several users discussed whether or not **GPT 5.1** outperformed **GPT-4o**, with some saying it was a huge step forward.
- **Gemini 3.0 Stealthily Released, Excites Users**: A member noted that [Google](https://blog.google) stated that **Gemini Drops** (*monthly feature updates for the Gemini app*) are rolling out to Pro subscribers today, with Free users getting the update in the *coming weeks*.
   - Reports indicate a *quiet rollout* of **Gemini 3.0 Pro** in select parts of Google’s ecosystem (developer tools, enterprise/Workspace) rather than a full consumer announcement and provided a [YouTube link](https://www.youtube.com/watch?v=0-CbsNB9tdk) and [other links](https://marketingtrending.asoworld.com).
- **Concerns Arise Over ChatGPT's Canvas Mode Bugs**: Users reported that **ChatGPT**'s website crashed and bugged when using canvas mode.
   - One user experienced no issues with short convos, and managed to fix their long conversations with a *tempermonkey script*.
- **Quest for Free, Unlimited AI Video Generation Continues**: A user wondered why it is rare to see a free AI that generates videos unlimitedly, followed by another user that **Grok** has free ai video, but the initial user meant like with not 5 seconds duration.
   - They also agreed that there's a lot of power and processing power to do it, that is why Pro gives access to it.
- **ChatGPT Crafts Roblox ESP and Aimbot Hack Scripts**: A user said they use **ChatGPT** for *Roblox ESP and aimbot hack scripts same for CS2 and it works*.
   - This caused mixed reactions, with some people saying it was diabolical and asking if it wasn't against the TOS.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1438272275446108201)** (87 messages🔥🔥): 

> `GPT 5.1, GPT 4.1 for creative writing, Sora 2 in India, New guardrails for image creation, OpenAI operator` 


- **GPT 5.1 Causes Frustration for Heavy Users**: Users express frustration with **GPT 5.1**, describing it as a *step backwards* with *garbage verbiage* and unnecessary prose and handles [custom instructions](https://platform.openai.com/docs/gpt-best-practices) in a *belligerent manner*.
   - Some users feel it is a *toy* version of a scientific instrument, while others find it fantastic and a breath of fresh air.
- **Story Arcs Broken: GPT 5.1 Over Summarizes and Repeats Scenes**: Users report that **GPT 5.1** is *really broken*, over summarizing things, and repeating scenes in story arcs, with one user stating it ruins their story.
   - Some users are reverting to **GPT 4.1** for storytelling due to these issues and lament the changes that have broken the AI, preventing it from following formats or prompts.
- **Model Update Tightens Restrictions**: The new update tightened restrictions on how the model references stored memories, generalizes vs. rewrites, threads bundles, uses character names with variant spellings, and handles new scene continuity resulting in repetition.
   - The model jumps back to default safety mode and tries to restate or reshuffle things
- **Image Creation Guardrails are Excessive**: Users are finding the new [guardrails for image creation](https://openai.com/policies/usage-policies) to be excessive, preventing them from using it for simple depictions anymore.
   - They are so *wonderful* to not be able to go back and edit my text!  It's *lovely* to have none of the new features and be losing half of the old ones!


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1438590659069083728)** (2 messages): 

> `Polaris Alpha Deprecation, GPT-5.1 Launch, GPT-5.1 Chat (Instant), GPT-5.1-Codex, GPT-5.1-Codex-Mini` 


- **Polaris Alpha sunset; GPT-5.1 Sunrise**: The "Polaris Alpha" model, which users have been testing, was an **early version of OpenAI's GPT-5.1 without reasoning** and will soon be deprecated.
   - It's succeeded by a faster, token-efficient **GPT-5.1** with adaptive reasoning and better coding, described in this [OpenAI news post](https://openai.com/index/gpt-5-1-for-developers/).
- **GPT-5.1's debut, a suite of models**: OpenRouter has launched **three more GPT-5.1 models**: [GPT-5.1 Chat](https://openrouter.ai/openai/gpt-5.1-chat) (aka Instant in ChatGPT), [GPT-5.1-Codex](https://openrouter.ai/openai/gpt-5.1-codex), and [GPT-5.1-Codex-Mini](https://openrouter.ai/openai/gpt-5.1-codex-mini-20251113).


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1438257135573270682)** (437 messages🔥🔥🔥): 

> `OpenRouter Privacy Settings, API Rate Limits, GPT-5.1 Release and Performance, AI SDK Adoption, Structured Outputs` 


- ****Privacy Toggles** Impact Model Availability**: A user resolved an error by enabling the third privacy toggle, *"Enable free endpoints that may train on inputs / publish prompts"*, in their [OpenRouter privacy settings](https://openrouter.ai/settings/privacy).
- **Experiencing **API Rate Limit Woes****: Users reported frequent **Error 429** (rate limit) issues with the API, especially with `anthropic/claude-sonnet-4` and `anthropic/claude-sonnet-4.5`, indicating the upstream provider doesn't have enough supply to meet demand.
   - Some users also encountered Cloudflare errors and timeouts, but the [OpenRouter status page](https://status.openrouter.ai/) showed no incidents.
- ****GPT-5.1** Arrives, Sparks Debate**: **GPT 5.1** is rolling out on ChatGPT now, but there's no API access yet.
   - Early benchmarks suggest nearly **2x token output** for minimal improvements, leading to concerns about value for cost: *"we want less thinking, not more"*.
- **OpenRouter AI SDK Adoption Discussed**: Members discussed the [OpenRouter Typescript SDK](https://github.com/OpenRouterTeam/typescript-sdk) the official OpenRouter SDK, and the broader use of the AI SDK in platforms like 3.chat.
   - Some users still prefer raw HTTP requests or `curl`, emphasizing they are  *"still probably the best platform for this stuff somehow"*.
- ****Structured Outputs** Demystified**: `json_object` only guarantees a valid JSON object, whereas `json_schema` enforces compliance with your defined schema when using structured outputs, however *"not many providers support structured outputs unfortunately though, so make sure to check the providers of the model you use"*.
   - Users can check for support through the model overview on the OpenRouter site by unfolding the providers to see the details.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1438608118660731011)** (4 messages): 

> `` 


- **No New Model News to Note**: There were no significant discussions about new models in the OpenRouter Discord channel.
- **Silence in the New Models Channel**: The OpenRouter's 'new-models' channel appears to be inactive, with no messages to summarize.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1438500085842444369)** (19 messages🔥): 

> `React Slop, React Compiler, Web Search Tools` 


- **Chatroom suffers React Slop Issues**: A member reported that attaching long files to the chatroom causes the browser to lag and that editing a message rerenders the entire chatroom, with another member suggesting this is a case of *React Slop*.
   - The suggestion was made to wrap everything in `useMemo` to mitigate the issue.
- **React Compiler Rocks!**: A member mentioned they started using **React Compiler** which addresses the *React Slop* issue, stating that *it rules*.
   - Another member jokingly questioned the seriousness of the statement.
- **Google & XAI Web Search Tools**: A member inquired about integrating native web search tools from **Google** and **XAI** into the platform, linking to the [Web Search Features documentation](https://openrouter.ai/docs/features/web-search).
   - Another member reported that they are working on **web search with Gemini**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1438650527972855869)** (10 messages🔥): 

> `CUDA project for Monte Carlo PSF calculation, Why learn GPU programming if Python wrappers exist?, Data center operations with inference and Bitcoin mining, GPU compiler bug with __restrict__ qualifier` 


- **CUDA Monte Carlo for PSF Calculation: Feasible?**: A student inquired about the feasibility of implementing a **Monte Carlo method** in **CUDA** for calculating the **Point Spread Function (PSF)** in electron scattering as a final coursework project.
   - The student is unsure whether it's a decent project given their limited expertise in both **CUDA** and **Lithography** for wafers.
- **GPU Programming: Python Wrappers Obsolete Direct Learning?**: A member questioned the necessity of learning **GPU programming** directly, given the prevalence of **Python wrappers** like **PyTorch** that abstract away much of the underlying complexity.
   - The discussion centered on whether the ability to use **precompiled GPU code** via **Python libraries** negates the need to understand low-level **GPU programming**, though many agreed someone still has to *make* those libraries and novel applications benefit from lower-level expertise.
- **Data Center Synergies: Inference & Bitcoin Mining**: A member is writing a paper on **system dynamics** and sought input from anyone with experience in **data center scale operations** that combine **inference** and **Bitcoin mining**.
   - They hope to find prior examples or insights into the potential synergies and conflicts between these two computationally intensive workloads.
- **Restrict Qualifier: Compiler Bug or Undefined Behavior?**: A member shared an interesting [example](https://godbolt.org/z/ad98nYdrf) of a potential **GPU compiler bug** involving the `__restrict__` qualifier, asking for opinions on whether it's a true bug or simply **undefined behavior (UB)**.
   - The code example includes **PTX** and **SASS** outputs, inviting deeper analysis of the compiler's handling of memory aliasing constraints.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1438752985692770325)** (1 messages): 

> `NaN to Zero, tl.maximum trap` 


- **NaN-to-Zero Code Caveats**: `tl.maximum(probability, 0)` was used as **nan-to-zero**, but it introduced accuracy drops.
   - Using `tl.where(p == p, p, 0)` worked better, though the reasons are unclear.
- **tl.maximum gotcha**: The user reported that using `tl.maximum(probability, 0)` caused some accuracy drop in their application.
   - They found that `tl.where(p == p, p, 0)` works well.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1438513507271311511)** (4 messages): 

> `Sass Registers Reusing, PyNvVideoCodec on AWS g4dn, Video Upscaling with CUDA and PyTorch, RealESRGan model, RTX4060 optimizations` 


- ****Sass** speedup in several percent**: A blogpost about [**Sass** registers reusing](https://redplait.blogspot.com/2025/11/sass-registers-reusing.html) to get speedup in several percent.
- **Decoding videos using **PyNvVideoCodec****: A member was trying to do video decoding and encoding using **PyNvVideoCodec** on **AWS g4dn** machines (with **NVIDIA T4** GPUs), and asked for steps on how to set up the environment.
   - They said that just setting the environment was a pain.
- **Slow video upscaling with **RealESRGan** and **RTX 4060****: A member was trying to do video upscaling with improved quality screen using **CUDA** and **PyTorch** with a **RTX4060** (**12GB** VRAM), but the FPS was very low (0.5fps) using **RealESRGan** model.
   - They asked how to improve the FPS to at least 50.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1438706098088706069)** (2 messages): 

> `cuBLAS FP64 Emulation, torch.mm Performance, Custom C++ Operator for cuBLAS, ATen cuBLAS GEMM Call Tracing` 


- **FP64 Emulation Rocks Some Hardware**: A member is exploring **cuBLAS FP64 emulation** from **CUDA 13.0u2** and observed *great performance* on their hardware with default **torch.mm** on some input sizes (>**580% peak FP64 throughput**), as described in [Learn-By-Doing-Torchinductor-DeviceAssert](https://karthick.ai/blog/2025/Learn-By-Doing-Torchinductor-DeviceAssert/).
   - They want to force usage of **cuBLAS kernels** to investigate if it could be better on other input sizes where dispatcher selects **CUTLASS kernels**.
- **Custom Operator Struggles Mirror Torch**: The member created a **C++ custom operator** based on [NVIDIA's cuBLAS emulation samples](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuBLAS/Emulation), but the performance is identical to non-emulated **torch.mm**.
   - They suspect they are calling **cuBLAS dgemm/gemmEx** incorrectly.
- **ATen Reveals cuBLAS Secrets**: The member is trying to trace exactly how the **cuBLAS GEMM kernels** are called in **ATen**, since torch.mm successfully calls cuBLAS such that the emulation is applied.
   - They used `TORCH_SHOW_DISPATCH_TRACE=1` and found `op=[aten::mm.out], key=[CUDA]`, which might be [at::mm_out](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L926), but are unsure how to proceed to find the cuBLAS call.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1438704349894283426)** (1 messages): 

> `Paulius Micikevicius, GPU efficiency, low bit dtypes, sparsity, numerical stability` 


- **Micikevicius Joins, Speaks on GPU Efficiency**: Paulius Micikevicius, known for his work on **GPU efficiency** with **low bit dtypes** and **sparsity**, will be giving a talk, co-hosted by another member, diving deep into **floats**, **numerical stability**, **determinism**, **quantization**, and **sparsity**.
   - He recently joined the same company as the host after a long tenure at NVIDIA; more information can be found in [this video](https://www.youtube.com/watch?v=3qNZvvlwcCI).
- **Talk Schedule Reduced, Returning in January**: The talk schedule will be reduced while the host is on parental leave.
   - The schedule should be back in full force in **January**.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1438779530729750599)** (1 messages): 

> `NCU, Cloud vendors, Semianalysis` 


- **NCU scores big with Cloud Vendors**: Cloud vendors are now being graded on supporting **NCU** (NVIDIA Compute Unified Device Architecture), as reported in [Semianalysis](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard).
- **Semianalysis Reports on Cloud Vendor Grading**: [Semianalysis](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard) reports that cloud vendors are now graded based on their support for **NCU**.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1438590563954856170)** (2 messages): 

> `Voltage Park, SWE positions, AI Factory Platform, CUDA kernel engineer` 


- ****Voltage Park** is Hiring Engineers**: **Voltage Park** is looking for **SWEs** to assist in Infrastructure Engineering, Software Engineering, and AI/ML focused Software Engineering.
   - They are remote friendly (preference for SF and Seattle) and are building up an **AI Factory platform**; check out their [careers page](https://www.voltagepark.com/careers).
- **CUDA Kernel Engineer Needed for Part-Time Gig**: A member is looking for a **CUDA kernel engineer** for some part time work.
   - They're offering **$200 an hour**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1438507649783038043)** (2 messages): 

> `Parallel Programming, Beginner Guidance` 


- **User Seeks Guidance on Beginning Parallel Programming**: A user expressed feeling lost on how to start with parallel programming.
- **Guidance Location**: Another user pointed them to channel **#1198358627594023014** as a possible starting point.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1438272647942242447)** (5 messages): 

> `NVFP4 Datatype Optimization, AI Coding Agent Workflow, Rust LLM Router` 


- **NVIDIA Hackathon Boosts NVFP4 Optimization**: GPU Mode and NVIDIA are hosting a hackathon to optimize workloads for **NVFP4** datatype on the **Blackwell GPU** architecture, with the first challenge focusing on GEMV (Matrix Vector multiplication).
   - A blog post explains the reference kernel and touches upon the new dataformat, recommending **CuTeDSL** for its ability to get close to hardware while maintaining productivity, with access to a **B200 GPU** provided by Datacrunch; see the [blogpost](https://veitner.bearblog.dev/nvfp4-gemv/) for details.
- **AI Coding Agent Workflow Open-Sourced**: An AI coding agent setup, designed to ship high-quality code, has been open-sourced, featuring skills, 114+ sub-agents, and an executive planning framework for complex features.
   - This setup, intended to enhance existing AI coding tools, can be found on [GitHub](https://github.com/flora131/agent-instructions) and a detailed explanation is available in [this blog post](https://alexlavaee.me/blog/ai-coding-infrastructure/).
- **Seeking collaborators for Rust based GPL'd LLM Router**: A member is seeking collaborators on rust based gpl'd llm router.
   - Find the repo at [GitHub](https://github.com/awdemos/merlin).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1438263286566093012)** (2 messages): 

> `Nvidia competition submissions, Discord bot, CLI, Site` 


- **Nvidia Competition Submissions Supported via CLI, Discord, Site**: Submissions for the **Nvidia competition** are supported via **Discord**, the **site**, and the **CLI**.
- **CLI is Most Popular Route for Nvidia Competition Submissions**: The most popular method for submitting to the **Nvidia competition** is probably the **CLI**.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1438647696486563944)** (2 messages): 

> `HipKittens, Quark Start` 


- **HipKittens now meows with gqa_backward**: A contributor crafted a [pull request](https://github.com/HazyResearch/HipKittens/pull/4) enhancing the **Makefile** to execute the **gqa_backward example** effortlessly, adhering to the **Quark Start** guide.
   - The author expressed gratitude and pledged to evaluate the pending pull requests.
- **Gratitude expressed for enhancement contributions**: A member showed appreciation for the new contribution to **HipKittens** project.
   - The user specifically thanked the contributor for their work on the **Makefile** update and committed to reviewing the contributor's pull requests in the near future.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1438430872889069588)** (1 messages): 

> `Inference optimization, GPU resources (Hopper, Ampere), University resources, Smaller models fitting on GPU VRAM` 


- **Inferencing Optimized on Hopper or Ampere**: Users can explore **inference optimization** using GPUs like **Hopper** or **Ampere**.
   - Universities often have these resources available through professors.
- **Smaller Models, Smaller VRAM**: Smaller models can be run if they **fit within the GPU's VRAM**.
   - It was an apology for taking the conversation into the wrong channel.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1438270167103569920)** (52 messages🔥): 

> `nvfp4_gemv Leaderboard Updates, grayscale_v2 Leaderboard Updates, NVIDIA performance improvements, H100 grayscale performance, Personal Bests on various GPUs` 


- **GEMV Speed Blitz on NVIDIA**: Multiple submissions to the `nvfp4_gemv` leaderboard show improvements in execution time on **NVIDIA**, with times ranging from **3.19 ms** down to **24.7 µs**.
- **Grayscale v2 GPU Gauntlet**: Submissions to the `grayscale_v2` leaderboard highlight performance across various GPUs, with one submission achieving **7th place on L4** at **27.5 ms** and another reaching **7th place on H100** at **12.9 ms**.
- **B200 Breaks Grayscale Barrier**: The `grayscale_v2` leaderboard saw new personal bests achieved on **B200**, with times dropping to **6.69 ms**.
- **A100 Aces Grayscale Task**: Submissions to `grayscale_v2` show personal bests on **A100**, reaching speeds of **20.4 ms**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1438607558980931627)** (1 messages): 

> `Network failover test` 


- **Network Failover Test Incoming**: The data center will be performing a scheduled **network failover test** today between **2:00–3:00 PM PT**.
   - Connectivity to the nodes might be temporarily impacted during this window, they apologized for the inconvenience.
- **Data Center Maintenance**: Scheduled maintenance may cause temporary connectivity issues.
   - Users are advised to expect interruptions between 2:00-3:00 PM PT.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1438567696345071689)** (13 messages🔥): 

> `FMTK Extension in Cursor, Factorio Mod Debugger, Agent Interaction Mod, Sumneko Language Server` 


- **Factorio Mod Debugger Surfaces**: A user shared a link to a **Factorio mod debugger** GitHub repository: [justarandomgeek/vscode-factoriomod-debug](https://github.com/justarandomgeek/vscode-factoriomod-debug).
   - Another user expressed excitement, noting that they had only been using **Sumneko** and **FMTK** as a language server and were unaware of the **debugger and profiler**.
- **Agent Interaction Mod on the Horizon**: A user is developing an **agent interaction mod** that uses modding best practices and aims for simpler integration with FLE (Factorio Learning Environment).
   - The user describes the **agent** as a *luaentity of character type that can be worked on by external systems + LLMs*, and it is almost ready to ship.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1438551236109537281)** (2 messages): 

> `CUDA `copy` behavior, GEMV kernel pipelining, GMEM -> SMEM -> RMEM data transfer, Numerical result discrepancy` 


- **`copy` CUDA threads explained**: A member questioned whether calling `copy(copy_a, ...)` with more threads than `size(copy_a)` would cause threads above the size to not issue the copy instruction.
   - It was implicitly clarified that when creating a `copy_a` with `size(copy_a)` = 32, only 32 threads would be used.
- **GEMV Kernel Pipelining Yields Wrong Results**: A member reported incorrect numerical results when implementing pipelining for a GEMV competition kernel using **GMEM -> SMEM -> RMEM** data transfers.
   - The issue occurs during the copy of matrix **A** from **GMEM** to **SMEM** (and also during transfer from **SMEM** to **RMEM** via *autovec_copy*).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1438281045039386654)** (2 messages): 

> `Submission location` 


- **GPU MODE submissions accepted in multiple places**: Users inquired about the correct location to submit work for GPU MODE, providing a [link to the leaderboard](https://www.gpumode.com/v2/leaderboard/595?tab=submission).
- **Alternative Submission Point Clarified**: Another user clarified that *both Discord and the website* are acceptable submission points.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1438319133174403102)** (39 messages🔥): 

> `Helion BC Compatibility and Updates, Helion Autotuning with configs, Helion Config Options and Accessing Fastest Config, Helion's Interpret Mode Speed` 


- **Helion Claims Backwards Compatibility**: Helion will be **BC compatible** (e.g. from **0.2** to **0.2.1**), indexing has been updated to a list instead of a single value because each load/store can be optimized independently for perf gain, but single value input is still supported; version **0.2.2** was just released with minor version releases planned every week or two to keep pypi packages updated, see [issue 164](https://github.com/pytorch/helion/issues/164).
   - The update to lists enables independent optimization of each load/store operation, enhancing performance while maintaining compatibility with single-value inputs.
- **Helion now Autotunes with Configs**: Instead of `config=`, you can pass `configs=` which would be similar to **Triton's autotune**.
   - The `@helion.kernel(configs=[a, b, c])` decorator will run a, b, and c, and pick the fastest config, behaving similarly to the **Triton autotuner**.
- **Faster Config Access with Helion**: `helion_rms_norm_fwd.bind((x, w, eps))._config` will help you get there.
   - Helion allows programmatic autotuning with a set of configs, running Helion autotuning on a “primary set of shapes” to get a set of configs, and then autotuning all of them on the secondary set of configs; accessing the `_config` property still works for this, enabling feeding the `_config` from the first set to the second set.
- **Helion's Interpret Mode Surprises with its Speed**: The interpret mode in Helion is noted to be surprisingly fast, running the entire code as if there are no tiles using eager PyTorch, which is attributed to the abstraction of tiles enabling performance portability.
   - In contrast, Triton is noted to be atrociously slow in interpret mode because the tile sizes being run are like 32, highlighting Helion's advantage in interpretative performance.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1438262945095221368)** (283 messages🔥🔥): 

> `NVIDIA Competition Guidance, Cutlass vs CUDA, Blackwell Optimization, NVF4 Data Transformation` 


- **GPU Rookies Receive Competition Guidance**: New competitors with **RTX 3060** GPUs received guidance to check out the [problem repository](https://github.com/gpu-mode/reference-kernels) and submission instructions via the [Discord Cluster Manager docs](https://gpu-mode.github.io/discord-cluster-manager/docs/intro/).
   - Solutions must be submitted in Python, but can use **CUDA/CUTLASS** with `load_inline`.
- **CUDA vs Cutlass**: Members discuss pros and cons of using **CUTLASS** over **CUDA**, highlighting the use of **CUDA C++** with inline loading in Python files.
   - A [Modular article](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota) was shared discussing optimizing for production shapes, including choosing optimal parameters like **MMA shape**, **pipeline stages**, and **block swizzling patterns**.
- **Blackwell Architecture Demands MMA Expertise**: Discussions involved optimizing matrix multiplication on **Blackwell**, referencing a [Modular.com blog post](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota) highlighting the need to minimize the workload per **SM**.
   - It was pointed out that the **B200** has **148 SMs**, creating challenges for grid division, with frustration expressed over devising sensible grids for problem shapes.
- **NVF4 Data Needs Layout Transformations**: Members debated **TMEM** descriptor formats and the need for **TMA** to transform data into the *UMMA canonical layout* for tensor cores.
   - It was clarified that **TMA** reorders data from row major to a different memory layout for the tensor core, and adds padding; apparently, the tensor core requires each column of 8 elements to be contiguous in memory.
- **CUDA Environment on Github Action**: Members found that the **Github Action harness** runs on **CUDA 13.0.88**, passing only `-gencode=arch=compute_80,code=sm_80` and `-gencode=arch=compute_100,code=sm_100` to nvcc, creating target errors.
   - Some members gave up on using **LLMs** to write code because they don't know to use **TMA** to expand the **FP4** to cursed layouts.


  

---


### **GPU MODE ▷ #[xpfactory-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1438266982104305706)** (6 messages): 

> `RLinf, Qwen3-VL VLA-adapter training, UnifoLM-WMA-0, LIBERO-PRO dataset limitations` 


- **RLinf Repo Readying**: A member is preparing to release the [RLinf Basic repo](https://github.com/RLinf/RLinfBasic) and plans to run **Qwen3-VL VLA-adapter training** overnight.
   - The member also intends to clean up the repository and evaluate **LIBERO** the following day.
- **Qwen3-VL GPU Stats**: A member shared that they are using a single **A6000** to train the adapter with two **256x256** image inputs on **Qwen3-VL-2B-Instruct**.
   - With a batch size of **48** and using **bf16**, this is running using a naive training setup.
- **Unitree's UnifoLM-WMA-0 Catches Attention**: A member noted that [Unitree's UnifoLM-WMA-0](https://github.com/unitreerobotics/unifolm-world-model-action) looks very interesting for world model action.
   - No further details were provided.
- **LIBERO-PRO Exposes Dataset Deficiencies**: The member also mentioned that [LIBERO-PRO](https://arxiv.org/abs/2510.03827v1) shows some limitations of the original **LIBERO** dataset.
   - No further details were provided.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1438260408459853995)** (397 messages🔥🔥): 

> `GPT 5.1 Codex Release, Cursor Auto Mode Degradation, Custom Commands in Cursor, Memory Usage in Cursor, Tailwind CSS v4 implementation` 


- **Cursor Rolls out GPT-5.1-Codex, Excites Engineers**: Members reported that **GPT-5.1-Codex** is rolling out, with some spotting it in the [latest codex alpha](https://x.com/OpenAIDevs/status/1986861734619947305) and noticing its integration with [Windsurf](https://www.windsurf.ai/).
   - However, there's confusion as some users haven't received an update prompt, while others find it already available in Codex before a general OpenAI API release, indicating a staged rollout.
- **Auto Mode's Performance Problems Plague Power Users**: Users are reporting a noticeable degradation in **Auto mode performance**, with one stating that *20 minutes ago the Auto mode was using another faster model, now it uses gpt 5 codex and it is slower and can't even edit a file when a model without reasoning can do it without issue*.
   - Some suggest this might be due to **Auto mode** defaulting to potentially overloaded OpenAI servers after the **5.1 release**, while others speculate Cursor is using a GPT-5 version with high thinking requirements.
- **Custom Commands cut token costs**: Members discussed the use of **custom commands** for automating tasks, such as running tests after code changes, where a user shared the approach of creating a custom command with **CTRL + K** to trigger specific instructions.
   - A member suggested ensuring tests are triggered automatically by executing specific commands, which can be tagged for increased token efficiency with the [docs section in Cursor settings](https://cursor.com/docs/customcommands).
- **Cursor's Memory Feature Misgivings Manifest**: Cursor's **memory feature** is prompting mixed reactions; one user running in legacy privacy mode is hesitant to disable it.
   - One user described that *memories are usually within your profile and they need to be active, if you're in privacy mode this tool calling will fail*, so this is a feature to keep an eye on if you want the best out of cursor.
- **Serialization Snafus Stymie Terminal Tasks**: Users report a *Serialization error* when using terminal commands, which breaks the chat until Cursor is restarted, with one user reporting that the source of the issue comes from commands that contain spaces in the file paths.
   - Members pinpoint the issue to errors in serializing terminal output fed to the LLM, and shared a link to the [Serialization Error forum post](https://forum.cursor.com/t/serialization-error/124671).


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1438644266577559613)** (1 messages): 

> `API Price Reduction, Hermes 4 70B, Hermes 4 405B` 


- **Nous Research Slashes API Prices!**: Nous Research has announced a **70% price reduction** for their **Hermes 4 70B** and **Hermes 4 405B** APIs.
   - Details and signup are available at the [Nous Research Portal](https://portal.nousresearch.com/) and [X](https://x.com/NousResearch/status/1989077400957911394).
- **API Access to Hermes Models**: The API provides access to the **Hermes 4 70B** and **Hermes 4 405B** models.
   - Sign up via the [Nous Research Portal](https://portal.nousresearch.com/) to get started.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1438275995999141990)** (75 messages🔥🔥): 

> `GGUF files in Nous Chat, GPT5.1 Release, Schizomax models, Hermes4 code model` 


- ****GGUF Files Fail to Function** with Nous Chat**: A user inquired about importing **GGUF files** into Nous Chat for model use, but another user clarified that this functionality is currently unavailable.
- ****OpenAI drops GPT-5.1** unexpectedly**: A user shared a link to the **GPT-5.1** release on the OpenAI website, [GPT-5.1](https://openai.com/index/gpt-5-1/).
- ****Schizomax Models get hype** in community!**: A member suggested the need for a **schizomax model**, with others agreeing that **Grok** and **Nous models** rarely refuse requests except for extreme content.
   - They expressed frustration with OpenAI's **GPT models** becoming increasingly restricted due to wellness checks and corporate influence.
- ****Hermes4 sneaks into Cline?!****: A user pointed out that **Hermes4** is now a code model in Cline, including screenshots of a prompt.
   - This observation sparked discussion about the evolving capabilities and potential applications of **Hermes4**.
- ****1 million downloads** milestone reached!**: A user announced that they had reached **1 million downloads**.
   - Another user congratulated and celebrated with them, further emphasizing the significance of this achievement.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1438277169834098849)** (79 messages🔥🔥): 

> `GGUF files, Nous Chat, Ollama, transformers.js, pgvector` 


- **Discover how to import GGUF files into Nous Chat**: A member asked how to import **GGUF** files into **Nous Chat** to use as a model, but it's currently not supported directly on the site.
- **Run GGUFs locally with llama.cpp or Ollama**: Members suggest running **GGUF** files locally using tools like [llama.cpp](https://huggingface.co/docs/hub/en/gguf-llamacpp) or [Ollama](https://ollama.com/) for a simpler setup.
- **Small AI model Test**: After installing **Ollama**, a member suggests testing it by running a small model using the command `ollama run hf.co/NousResearch/Hermes-3-Llama-3.2-3B-GGUF:Q4_K_M` to see if it works, its not our newest model but its small so will be a good test for you.
- **Improve transformers.js embeddings**: A member is getting low scores using **transformers.js** to generate embeddings for legal ordinances and **pgvector** as a database, even with deep chunking and breadcrumb context.
   - The member asks if there are any settings or optimizations to check for improving overall search quality.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

teknium: https://fxtwitter.com/historygpt/status/1977895243195334826?s=46
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1438268442070487070)** (22 messages🔥): 

> `Lakh MIDI Dataset Cleaning, Hugging Face Dataset Uploads, Wikipedia Dataset Cleaning and Structuring, JSON vs JSONL for Datasets, FineWiki Comparison` 


- ****Lakh MIDI Dataset** Receives Sparkling Cleaning**: A user shared that they fully cleaned and organized the entire **Lakh MIDI Dataset**, generating a clean, structured **JSON file** with over **44,000 entries**, fully parsed and consistent, and offered to share it for free.
   - Another user suggested uploading it to [Hugging Face](https://huggingface.co/), which the original poster agreed to do, also inviting collaboration and enhancements to the dataset.
- **Wikipedia Dataset Given a Spring Cleaning, Hosted on HuggingFace**: A user uploaded a cleaned **French Wikipedia DB** version with over **2,700,000 files** in **JSON format** to [Hugging Face](https://huggingface.co/datasets/YoloMG/wikipedia-fr-2.7m-clean-json), also noting they are working on cleaning the English version.
   - The user specified that the data cleaning encompasses more than just plain text, with cleaning including *templates, tables, html, refs*, while retaining infobox information and links.
- ****JSONL** Format Shines in Data Discussions**: In a discussion about dataset formats, a user suggested using **JSONL/NDJSON** for purely textual data, rather than tar files due to the overhead of tar headers.
   - They argued that **JSONL** makes processing easier by allowing line-by-line reading, contrasting with the need to process each header in tar files.
- **FineWiki Pales in Comparison to New Cleaning**: A user inquired about the improvements in the new Wikipedia dataset compared to **FineWiki** and other filtered Wikipedia variants.
   - The dataset creator responded that their process cleans more *templates, tables, HTML, and refs*, and maintains structured elements like infoboxes and links, differing from **FineWiki's** focus on plain text.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1438260121196036178)** (95 messages🔥🔥): 

> `NVIDIA's Dataset License, Synthetic Data, Tokenizer Paper, Anthropic's Policies on China` 


- **Memorization Disentangled by Researchers**: Researchers release [a paper](https://arxiv.org/abs/2510.24256) about disentangling memorization in model weights.
   - The discussion focuses on the technical coolness of the approach.
- **Navigating NVIDIA's Thorny License**: The community discusses the restrictions in **NVIDIA's dataset license**, particularly regarding training, evaluation, and public sharing of results, with one member linking to [an X thread](https://x.com/goodfireai/status/1986495330201051246) expressing concerns.
   - The primary concern revolves around a clause allowing **NVIDIA** to terminate the license at any time, potentially voiding any granted permissions, and the legal ambiguity which makes it difficult to ascertain what exactly is allowed.
- **SYNTH Data frontier**: Members discuss the presence of **QA pairs** at the end of documents when combing through this data.
   - The QA pairs in the data are likely **synthetic**, generated by rephrasing content with QA prompts using **nemotron-cch** ([https://arxiv.org/abs/2511.08923v1](https://arxiv.org/abs/2511.08923v1)), with some expressing skepticism about the quality of data from **PleIAs** (and mentioning a new data frontier [Blogpost](https://pleias.fr/blog/blogsynth-the-new-data-frontier))
- **Tokenizer Paper**: A new **tokenizer paper** emerges in the community for consideration.
   - The [paper](https://arxiv.org/abs/2511.09709) is shared as potentially interesting.
- **Anthropic Accused of Fear Mongering**: A member accuses [Anthropic](https://x.com/AnthropicAI/status/1989033793190277618) of **fear-mongering** to gain strategic advantages, particularly against **non-US** and **Chinese labs**.
   - Concerns are raised about **Anthropic's data privacy practices**, suggesting that they may prioritize safety over privacy, and also raises questions about **Anthropic's CEO's views on China**.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

burnytech: https://openai.com/index/understanding-neural-networks-through-sparse-circuits/
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1438336536566435871)** (12 messages🔥): 

> `lm harness, unitxt parameters, xsum dataset` 


- **Datasets in lm harness**: A member inquired about datasets available in the **lm harness** besides `scrolls`, `megsum`, and `noticia`.
   - The team confirmed the harness includes several subtasks, some in `darija/catalan/spanish_bench`, and that it uses **unitxt** under the hood.
- **`xsum` Dataset confirmed in harness**: A member asked if the `xsum` dataset ([https://aclanthology.org/D18-1206/](https://aclanthology.org/D18-1206/)) is included in the harness.
   - Another member confirmed that `xsum` is indeed used, and that it's passed on to **unitxt** internally, pointing to the relevant [unitxt file](https://github.com/IBM/unitxt/blob/800b2bad7f6cf794bde4e8fd8f4cbd0461e5940c/prepare/cards/xsum.py#L11).
- **Unitxt Parameters**: A member sought guidance on passing **unitxt** specific parameters (e.g., `--template templates.summarization.abstractive.full` and `--max_test_instances`) when evaluating models.
   - The team suggested modifying the [task YAML file](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/unitxt/xsum.yaml) to configure these parameters.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1438266540616323153)** (42 messages🔥): 

> `Pub/Private Members in Mojo, Modular Tech Stack for Regression Models, Mojo vs MAX, WebGPU in Mojo, FFI C in Mojo` 


- ****Privacy Please**: Pub/Private Members Cometh (Eventually)**: Mojo may eventually support **public and private members/methods**, similar to other languages, but it won't happen until there's an *escape hatch* for breaking encapsulation, and right now, Mojo uses Python's convention of an underscore to suggest privacy.
   - The current approach is to use **Python's underscore convention** to indicate that members should be considered private.
- ****Stacking Modular**: Regression Model Realness Check**: To leverage the **Modular tech stack** for building a regression model, you'd currently need to build parsers, data visualization libraries, and the backwards pass for training using MAX; right now, the fastest approach would be to train with Torch and then perform inference with MAX.
   - Mojo is best for *compute-heavy things* and lacks a comprehensive ecosystem.
- ****Mojo vs MAX**: Dueling Data Dynamos**: **MAX** is a compiler that allows for different data processing tradeoffs compared to **Mojo**, and while training can be done in pure Mojo, MAX is a better idea, especially for data processing.
   - Both are capable of data processing in different ways.
- ****WebGPU Wonders**: Mojo's Missing Magic**: While attempting to use **WebGPU C headers in Mojo** might seem feasible, Mojo currently lacks support for the compile target, so it likely won't work.
   - You can use **Vulkan and OpenGL** libraries, but you can't write the shader code in Mojo until support is added; *LLVM supports SPIR-V backend, so it's not that far away*.
- ****C-ing is Believing**: FFI Guide Fables**: Guidance on how to perform **FFI C in Mojo** is scarce, with only some code in the standard library offering examples.
   - There is no guide or example on how to do **FFI C** in Mojo available yet, with members pointing to the code in the stdlib.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1438258766112882768)** (46 messages🔥): 

> `comptime keyword, stdlib versioning with pixi, LayoutTensor vs raw memory, HipKittens paper` 


- **`comptime` Keyword Covers Assignment**: The `comptime` keyword in Mojo covers what `alias` used to, including type assignments, enabling zig-style static reflection like `comptime InnerTypeFirstField = T.fields[0].type`.
   - Although *it reads a bit off for type assignments*, having different keywords can be annoying when mixing types and values aggressively at compile time.
- **Versioning Mojo stdlib via Pixi**: Users can now clone a fork and use `./bazelw build` to create a `stdlib.mojopkg` in `bazel-out`, replacing the existing one in a Pixi environment.
   - Instead of overwriting the file, can define `MODULAR_MOJO_MAX_IMPORT_PATH` in `activation.env` to point to the new standard library as described in these [instructions](https://docs.modular.com/mojo/current/packages.html).
- **`LayoutTensor` for Zero Overhead?**: While raw memory approach might seem faster, `LayoutTensor` should offer zero overhead and more debug information compared to manual memory management, *avoiding headaches*.
   - Stick to raw memory only in simple programs; otherwise, using `LayoutTensor` avoids ASAP destruction issues, with more info available in the `List` and `Span` sources in the stdlib, and `LayoutTensor`'s source in MAX.
- **`HipKittens` Paper Mentioned Mojo's Performance**: The [HipKittens paper](https://arxiv.org/abs/2511.08083) mentions Mojo’s MHA kernel suffers from expensive bank conflicts and achieves just 50% of the peak kernels’ performance on the MI355X.
   - A member commented that if LLVM can talk to a device, abstractions can be built at compile time, suggesting a potential device model that reduces the need for AMD/NVIDIA-specific kernels.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1438460489192570881)** (8 messages🔥): 

> `Apple Silicon GPU support in Max, Overhead passing tensors between Torch and Max with DLPack, GPU puzzles experience` 


- **Apple Silicon GPU Support ETA**: A member inquired about an estimate for when the first **Apple Silicon GPU support** might land in **Max**.
   - Another member responded that thanks to community PRs, they’ve expanded support for many intrinsics and enabled more tests and GPU puzzles, and that basic **MAX graphs** have started working.
- **Tensor Passing Overheads Exposed**: A member asked about the overhead when passing tensors between **Torch** and **Max** with **DLPack**, specifically regarding stream synchronization.
   - They requested *links to issues* to watch for updates on this.
- **GPU puzzles provide great experience**: A member reported they ran through the **GPU puzzles** again and had a really great experience.
   - Another member agreed that *the dream of developing kernels locally and deploying on a supercomputer* is becoming a reality.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1438267257766543537)** (59 messages🔥🔥): 

> `Whisper for Transcription, ICLR Review Quality, Multi-Head Attention, Conference Review Improvement, Study Abroad vs. Research` 


- ****Whisper** Still Top Dog for Open-Source Transcription?**: According to one member, **Whisper** remains the best open-source option for transcription, especially when using the [large-v3 model](https://openai.com/blog/whisper) quantized to **8 and 4 bits** with the **Whisper-server**.
   - Running **Whisper** directly through PyTorch in Python can be problematic, but using **Whisper.cpp** with Vulkan support improves portability.
- ****ICLR** Review Quality Takes a Dive**: Members are noting that the quality control for **ICLR** this year was sub par, with *lots of llm reviews with the prompts left in* and that in comparison *NeurIPS was better*. 
   - One member suggested that conference submissions should be longer with *pre-review rejection for papers that are clearly trash and just waste review time*.
- ****MHA** Heads Initialized Randomly, Dropout Stabilizes**: A member was curious why all heads don't end up learning the same thing with Multi-Head Attention (**MHA**).
   - Others suggested that *random initialization plays a role*, dropout, and if all layers learn the same thing, they'd be wasting space, a better solution exists by having each learn separately.
- **Conference Reviewers Deserve Incentives to Improve Quality**: The group discussed monetary incentives for conference reviewers who consistently do good reviews, which would require a meta-reviewer system.
   - It was suggested that a small increase in conference admission fees could fund this, considering **NeurIPS** had **16k** in-person attendees last year.
- **Arxiv Curating split from genuine research**: A member identified a fundamental difference in how the space was being used, suggested creating a new space for the use case of **arxiv curation** vs **genuine research interest**.
   - There are no rules about competing with spambots.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1438342028999262350)** (4 messages): 

> `Seesaw Preference Tactics, X Thread Paper Identification` 


- **Seesaw Preference Baffles Recommenders**: A member mentioned intentionally **seesawing** their preferences to confuse recommendation algorithms, linking [this paper](https://arxiv.org/abs/2511.08378).
   - The user reported that the link was shared in an X thread as an identification of the linked paper.
- **X Observer Confirms Paper Link**: In response to the paper link confirmation, a member identified as a *passive observer* of X.
   - This highlights the different levels of engagement within the community regarding external platforms like X.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1438267054284345386)** (16 messages🔥): 

> `GPT-5 Conversational, Output Tokens, DeepMind's SIMA-2, AnthropicAI espionage` 


- **GPT-5: More Talkative to Tap GPT-4 Users?**: Discussion around whether **GPT-5's** "more conversational" style is a ploy to attract **GPT-4** users to spend more on [output tokens](https://openai.com/index/gpt-5-1/).
   - A member speculated that there might be a restriction on the amount of output tokens on different user-level tiers, at least for the free version.
- **DeepMind's SIMA-2 Agents Take the Virtual Stage**: A member shared [DeepMind's SIMA-2](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/), an agent that *plays, reasons, and learns* in virtual 3D worlds.
   - They also wondered if copyright striking works for these models if they generate images with textures of existing games.
- **Espionage with Anthropic's Claude?**: A member linked to [AnthropicAI's demo](https://x.com/AnthropicAI/status/1789033793190277618) showcasing how advanced AI agents are becoming.
   - The conversation then turned to whether **Chinese state hackers** might use **Anthropic**, and to collect training data, given closed weight models are better.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1438259599546515630)** (75 messages🔥🔥): 

> `Claude Skills for Frontend and Backend, Latent Space Spotify Feed Issues, Parag Agrawal's Parallel Web Systems Funding, Cursor's Series D Funding, AlphaResearch LLM Agent` 


- ****Claude Skills**: Backend Brilliance, Frontend Fumbles?**: A member found **Claude**'s backend skills to be very constrained and effective, but noted that its frontend skills were only *okayish* with the model still making mistakes, referencing [this blog post](https://www.claude.com/blog/improving-frontend-design-through-skills).
- ****Spotify Shenanigans**: Copyright Chaos Disrupts Latent Space Feed**: The Latent Space **Spotify feed** faced issues due to a copyright claim on the royalty-free intro song, but it was later reported that [Spotify said it was fixed](https://x.com/paraga/status/1988729121636294682/photo/1).
- ****Parallel Universes**: Parag's AI Web Raises $100M**: **Parag Agrawal**'s new company, **Parallel Web Systems**, secured **$100 million** in Series A funding to build the web for AIs, sparking excitement and praise for its sleek design, as seen in [this announcement](https://xcancel.com/paraga/status/1988729121636294682/photo/1).
- ****Anthropic Amplifies**: $50B for US Datacenter Domination**: **Anthropic** unveiled a plan to invest **$50 billion** in US datacenters in Texas and New York, creating construction jobs and igniting debates over domestic compute scale, staffing, environmental impact, and AI industry bubbles, summarized in [this post](https://xcancel.com/anthropicai/status/1988624013849935995?s=46).
- ****Holo2 Hype**: Cheaper Vision Model Challenges GPT-4V**: **HCompany** launched **Holo2**, a cheaper **30B-MoE** vision model family built on **Qwen3-VL**, outperforming **GPT-4V** on UI benchmarks while running on web, desktop, and mobile, detailed in [this announcement](https://xcancel.com/hcompany_ai/status/1989013556134638039).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1438621839755841547)** (3 messages): 

> `Nano Banana, Image Models, Prompt Engineering, min_p for image tokens` 


- **Nano Banana Prompt Engineering**: A member shared a link to a blog post about prompt engineering **Nano Banana** image models: [Nano Banana Prompts](https://minimaxir.com/2025/11/nano-banana-prompts/).
- **Image Models Get Less Attention than Text Models?**: A member mentioned that they haven't tested image models like **Nano Banana** nearly as much as they have text models.
   - They also wondered if there is something like *min_p* for in generation for image tokens.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1438266498362769581)** (50 messages🔥): 

> `Upscaled Models, Multi-Head Attention, HuggingChat's new monetization, AI Voice Implementation, Granite 4 series` 


- **Upscaled Model is a Laughing Matter**: Users joked about an **upscaled model** with finetuning, finding the name funny and sharing a [Beavis and Butthead GIF](https://tenor.com/view/beavis-butthead-beavisandbuttheadshow-gif-13367061995602198225).
   - The discussion referenced specific Discord channels related to the model, sparking lighthearted banter.
- **Multi-Head Attention Initialized!**: A user inquired about **multi-head attention**, questioning why heads don't learn the same thing, with the primary reason being random initialization and [exaggerated differences by softmax](https://huggingface.co/datasets/John6666/forum2/blob/main/multi_head_attention_1.md).
   - Another added that random initialization, separate parameters, and different gradients cause heads to diverge naturally, with the learning task preserving useful differences.
- **HuggingChat Monetization Mourned**: Members lamented the **new HuggingChat's** shift to a paid model with limited free features, contrasting it with the previous free and unlimited version, showing a [screenshot](https://cdn.discordapp.com/attachments/879548962464493622/1438593728200572928/Screenshot_2025-11-13-18-16-43-962_mark.via.gp-edit.jpg?ex=69181b10&is=6916c990&hm=7bc833942ddb303310bdca35fdf5e940a15a9eb9a345cbc6086e0a21f8e817c6&) outlining cons of the subscription.
   - One user stated it *was devastating when they realised* the new version of HuggingChat was so lackluster and questioned if Hugging Face's goal of providing an open source AI platform for free has gone out of the window
- **AI Voice Implementation Impresses**: Discussion emerged around implementing **AI voice**, with excitement for integrating voice into a *smutty sexbot idea*.
   - A user shared a link to an [open-source voice design](https://huggingface.co/maya-research/maya1) with 20 human emotions, and wondered if it was hard to implement and how it could be added to an android app.
- **Granite 4 Joins Hugging Face**: A user asked whether the new **IBM granite 4 series** supported.
   - Another said that it works with Hugging Face Transformers, with links to the [Granite 4.0-h-small model](https://huggingface.co/ibm-granite/granite-4.0-h-small) and its [GGUF version](https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1438490755155886080)** (3 messages): 

> `AI-Powered Mixing Forum Analyzer, Propercode multi-agentic coding CLI tool, Geopolitical Forecasting Model Framework` 


- **Mixing Forum Receives AI Injection**: A member built an **AI-powered Mixing Forum Analyzer** that uses semantic search to find relevant audio engineering advice from forum posts.
   - Key features include **semantic search** with sentence embeddings (**SBERT**), **spaCy-based POS/adjective analysis** for mixing terminology and **lemma overlap detection**, built with **Python, Streamlit, and Hugging Face models** ([GitHub](https://github.com/steme855/mixing-forum-analyzer), [Live Demo](https://huggingface.co/spaces/Stepman/mixing-forum-analyzer)).
- **Propercode Tool Promises Production Level Code**: A member introduced **Propercode**, a multi-agentic coding CLI tool for codebases powered by **Pydantic AI** agents orchestrated as a graph, aiming for reliability and coding accuracy ([GitHub](https://github.com/JaiSuryaPrabu/proper-code)).
   - The tool is currently in **v0.1** and plans to provide multi-tools and modes for agents like autonomous, learning guide etc.
- **Geopolitical Forecasting Framework Makes Debut**: A member shared their **Geopolitical Forecasting Model Framework** where users can plug in their own political data and start generating projections about current or potential conflicts ([Hugging Face](https://huggingface.co/clarkkitchen22/GeoBot-Forecasting-Framework)).


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1438549080132423783)** (1 messages): 

> `Kaggle, Deep Learning, Large Datasets` 


- **Kaggler queries how to handle Huge Datasets**: A member seeks advice on preprocessing a large **150 GB Kaggle dataset** for deep learning, facing **Kaggle's 20GB write limit**.
   - The member asks how others handle preprocessing such large datasets within the platform's constraints.
- **Solutions for Large Dataset Preprocessing**: Potential solutions involve **using external storage**, **data chunking**, or **cloud-based processing** to overcome Kaggle's limitations.
   - The community often suggests leveraging cloud services like **Google Cloud** or **AWS** for scalable data preprocessing.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1438280740658741278)** (46 messages🔥): 

> `DSPy prompting, Pydantic models for signatures, Agentic search, ripgrep regex usage, Survey language issue` 


- **DSPy's Prompting Perceptions**: Some members believe DSPy is being *inadvertently* communicated as a framework where you don’t need to prompt LLMs anymore, which can be misleading, as prompting is still needed for domain-specific LLM apps, even with examples and **GEPA**.
   - One member framed it as DSPy providing the ability to encapsulate prompting as a programmable module for later optimization, highlighting that the instructions or docstrings of a signature class constitute **the prompt**.
- **Pydantic Signatures Preferred**: One member expressed disliking how signatures are introduced in the docs as simple string types (`input -> output`), preferring **Pydantic models** for more complex and type-checked realistic use cases.
   - The member also noted that instructions in the signature act as prompts, and the confusion in the community stems from the different interpretations of what a *prompt* actually means.
- **Agentic Search Instructions**: To enhance **Agentic search**, a member instructs the LLM within their **ReAct module** to send specific terms for tool search via **ripgrep**, but to add **Regex** as a backup if the initial search fails.
   - This instruction was crucial for the LLM to use Regex terms effectively in the search tool, especially when accessing multiple tools (3-4 functions for Agentic search).
- **Survey Language Suspicions**: A member is suspecting an issue with **survey language** based on a screenshot with **fine-tuning** called out at the top.
   - Another member calls the survey language *insane* to be buried in the appendix with fine-tuning at the top.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1438763509457616956)** (1 messages): 

> `Kimi K-2, Together AI partnership, 1T MoE` 


- **Kimi K-2 Deep Dive incoming!**: The **Moonshot AI** team announced a "fast but mighty deep dive" into **Kimi K2 Thinking** in partnership with **Together AI**.
   - The event will occur on **Nov 19, 2025** at **9AM PT**, and registration is available at [luma.com/g5qcq85z](https://luma.com/g5qcq85z).
- **Together AI and the Power of 1T MoE**: The announcement highlights the power of **1T MoE** (Mixture of Experts) and its ability to execute **300 tool calls in a single run**.
   - They invite users to *discover what it means for your agents*.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1438273815711977562)** (43 messages🔥): 

> `GLM-5, K3, R2, Kimi CLI vs Hello Computer, YC-backed Kimi For Coding, Chinese AI labs entering chipmaking, Kimi Chat projects and custom instructions` 


- **GLM-5, K3, R2 on the Horizon**: Members are anticipating the arrival of **GLM-5**, **K3**, and **R2**, while some express satisfaction at perceived missteps by **OpenAI**.
   - One member stated *I am not hyped at all for models like Gemini 3, since they dont offer a good coding plan like Kimi/GLM*.
- **Kimi K2 slower but more capable than others**: Despite an int4 optimization, **Kimi K2-thinking** is reportedly **1.8x** slower than **GLM-4.6** and **2.5x** slower than **Minimax m2.wysh.3**, but considered more capable for non-pure coding tasks.
   - One user shared a humorous screenshot of **Kimi K2** hallucinating.
- **YC Champions Kimi For Coding, Sparks Debate**: The **Y Combinator**-backed **Kimi For Coding** with its **2048** weekly usage quota is being criticized as a *daylight robbery* for a mediocre model, with one user stating *I can’t believe they think such a product must exist*.
   - A link to a relevant [Y Combinator tweet](https://x.com/ycombinator/status/1988366241460089118?s=46) was shared in the discussion.
- **Moonshot AI K100 Incoming?**: Following the trend of US AI labs, Chinese AI labs are also entering the chipmaking business.
   - Users are anticipating the release of **Moonshot K100**, referencing [this tweet](https://x.com/tphuang/status/1988952992003891330?s=46) which highlights the trend.
- **Kimi Chat lacks key features!**: Users are questioning the absence of project and custom instruction capabilities on the **Kimi Chat** website, even for paid subscribers.
   - Others clarified that **tool use** is automatically enabled on **Claude Code** or **Kimi-CLI**, and is triggered when the AI uses external resources like web search or file reading.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1438262337579782228)** (32 messages🔥): 

> `GPT-5.1, System/Model Card, Aider-ce, Deepseek API, moonshotai Kimi-K2` 


- **GPT-5.1 Released Without Benchmarks**: The release of **GPT-5.1** lacks any mentioned benchmarks, and its [system/model card](https://cdn.openai.com/pdf/4173ec8d-1229-47db-96de-06d87147e07e/5_1_system_card.pdf) is considered a *joke* by some.
   - There is no API, which is seen as suspicious, raising questions whether it's a new model or just revised system prompts.
- **Aider-ce Lauded as Good**: **Aider-ce** is praised for already having features discussed in the channel, but concerns arise regarding a lack of communication and succession planning from its maintainer, with one issue asking for an orderly succession plan was closed as *not planned*.
   - Despite this, the branch is well-received, and the server sees a consistent influx of new users daily.
- **Deepseek API Cuts Agent Mode Time**: Users are finding success using the **Deepseek API** in **Aider-ce**'s agent mode, reporting that using **GPT-5-high** results in slow performance.
   - It's suggested that the slowness is due to the regeneration of the repo map on larger repos, and adjustments to settings like `repo_map_max_files` and `repo_map_token_budget` may help.
- ** moonshotai Kimi-K2 Thinking Runs on Aider**: A user resolved a *404 no cord found* error when running aider with `moonshotai/Kimi-K2-Thinking` by correcting the **OPENAI_API_BASE** variable to `https://llm.chutes.ai/v1/`.
   - The correct commands are:
```
SET "OPENAI_API_BASE=https://llm.chutes.ai/v1/"
SET "OPENAI_API_KEY=mykey"
aider --model openai/moonshotai/Kimi-K2-Thinking
```


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1438485994423975936)** (3 messages): 

> `DeepSeek API, Commercial Privacy for DeepSeek` 


- **DeepSeek gains a Fan**: A member expressed being *pretty much sold on* **DeepSeek** for everything.
   - The member clarified they use the **DeepSeek API** only for open source or personal projects.
- **Commercial Privacy concerns for DeepSeek**: Another member inquired about services offering **DeepSeek** with respect for commercial privacy.
   - The initial member indicated they use the raw **DeepSeek API**, and don't have commercial requirements.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1438446165002686545)** (7 messages): 

> `Postgres MCP Server, executeQuery MCP Tool, SQL Query Results Format, GitHub Discussions vs Discord for MCP` 


- **MCP Server Builder Ponders Query Result Formats**: A member is building a **Postgres MCP server** and seeks advice on the optimal output format for the **executeQuery MCP tool**.
   - They are considering **JSON**, **Markdown**, or **Toon** formats for returning **SQL query results**.
- **Discord Traffic Cop Redirects General Intro**: A user greeted the channel and was redirected to <#1358874322030301417> to introduce themselves properly.
   - The moderator emphasized that the channel is for protocol discussions and related official SDK projects.
- **GitHub Discussions Encouraged for MCP Implementation**: For general **MCP implementation discussions**, the moderator recommends using a [GitHub Discussion](https://github.com/orgs/modelcontextprotocol/discussions) or other community Discord servers.
   - This Discord server is explicitly for discussing the protocol and related official projects like SDKs.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1438492629582221453)** (7 messages): 

> `File Transfer over MCP, MCP Resources, MCP Implementations` 


- **File Transfer Question Quelled**: A member asked whether the current **MCP protocol** supports file transfer from client to server, and the answer is natively **no**.
   - Another member clarified that file transfer is possible via **resources**.
- **Discussion of MCP Implementation**: A user inquired about uploading files directly from a chat interface via an **SE upload call** using tools.
   - Other members pointed out that the Discord server is for discussing the **protocol and related official projects**, suggesting the user open a [GitHub Discussion](https://github.com/orgs/modelcontextprotocol/discussions) or ask in other community Discord servers and pointed to [issue 1306](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1306).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1438275827405029426)** (9 messages🔥): 

> `Manus Support, Credit Changes, AI/ML Engineer, workflow automation engineer` 


- ****Manus Support** closed?**: A user reported that **Manus' checkpoint system** is unable to find git commits, blocking the "Publish" button, requiring **Manus support** to manually sync the internal repository.
   - The user was directed to [Manus feedback](https://manus.im/feedback) after asking if manus just removed chat mode?
- **New Credit Plans rolled out**: A user noticed a change in the monthly **credit allocation** for plans, specifically noting that the $19 plan now offers **4000 credits** instead of the previous **1900 credits**.
   - They wondered if credit usage has changed.
- **AI/ML engineer available!**: An **AI/ML engineer** with expertise in model design, optimization, and large-scale deployment introduces themselves.
   - Their tech stack includes **Python, C++, Rust, SQL, Hugging Face, ONNX Runtime, Triton Inference Server**, and **Apache Spark**.
- **Automation Engineer seeks work**: An experienced engineer specializing in **workflow automation, LLM integration, RAG, AI detection, image and voice AI**, offers their services.
   - They built automated pipelines and task orchestration systems using **Dspy, OpenAI APIs**, and custom agents, linking to their portfolio at [devx-green.vercel.app](https://devx-green.vercel.app/).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1438441292483792958)** (7 messages): 

> `OpenCL Devices, tinygrad with C++, torch_load VGG16, pmap and vmap, OpenPilot PR` 


- ****OpenCL Devices** error fix suggestion**: A user suggested adding `if num_devices.value == 0: raise RuntimeError("No OpenCL Devices")` to the codebase to address an issue where the code falls through even when there are no devices to run on.
   - The user referenced the [OpenCL documentation](https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceIDs.html) regarding `CL_INVALID_VALUE` errors when `num_entries` is zero and `devices` is not NULL, or when both `num_devices` and `devices` are NULL.
- ****pmap and vmap** updates coming soon**: A user updated the [README](https://github.com/tinygrad/tinygrad) with help from ChatGPT, emphasizing the need for **pmap** and **vmap** functionalities.
   - These functions will enable more efficient and concise array manipulations within **tinygrad**.
- ****torch_load**: VGG16 model now works**: A pull request ([PR #13253](https://github.com/tinygrad/tinygrad/pull/1325)) was created to enable the **torchvision** hosted **VGG16** model to work with `torch_load`.
   - This enhancement expands the compatibility of **tinygrad** with pre-trained models from the **torchvision** library.
- ****OpenPilot PR** merged**: The [OpenPilot PR](https://github.com/commaai/openpilot/pull/36615) has been merged, enhancing integration or compatibility between **tinygrad** and **OpenPilot**.
   - Developers committed to preventing regressions of this integration in the future.
- ****tinygrad** and C++ Integration?**: A user inquired about using **tinygrad** with **C++**, with the intention of using it in embedded systems.
   - This suggests interest in leveraging **tinygrad** for resource-constrained environments and applications.


  
---
id: MjAyNS0w
title: Gemini 2.5 Pro Preview 05-06 (I/O edition) - the SOTA vision+coding model
date: '2025-05-06T05:44:39.731046Z'
description: "**Gemini 2.5 Pro** has been updated with enhanced multimodal image-to-code capabilities and dominates the WebDev Arena Leaderboard, surpassing **Claude 3.7 Sonnet** in coding and other tasks. **Nvidia** released the **Llama-Nemotron** model family on Hugging Face, noted for efficient reasoning and inference. **Alibaba's Qwen3** models range from 0.6B to 235B parameters, including dense and MoE variants. **KerasRS** was released by **Fran\0ois Chollet** as a new recommender system library compatible with JAX, PyTorch, and TensorFlow, optimized for TPUs. These updates highlight advancements in coding, reasoning, and speech recognition models."
companies:
  - google-deepmind
  - nvidia
  - alibaba
  - hugging-face
models:
  - gemini-2.5-pro
  - claude-3.7-sonnet
  - llama-nemotron
  - qwen3
topics:
  - multimodality
  - coding
  - reasoning
  - model-release
  - speech-recognition
  - recommender-systems
  - benchmarking
people:
  - demishassabis
  - _philschmid
  - lmarena_ai
  - scaling01
  - fchollet
---


**Gemini is all you need.**

> AI News for 5/5/2025-5/6/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (214 channels, and 4980 messages) for you. Estimated reading time saved (at 200wpm): 468 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

3 weeks after [2.5 Flash captured the low end of the Pareto Frontier,](https://news.smol.ai/issues/25-04-17-ainews-gemini-25-flash-completes-the-total-domination-of-the-pareto-frontier) it is time for Gemini to re-up the high end.

[Google I/O is in two weeks](https://blog.google/feed/google-io-2025-save-the-date/), and there's an old adage that adding more coding in a model's dataset somehow helps it improve in all other respects, and today's Gemini 2.5 Pro update (which was [only released 6 weeks ago](https://news.smol.ai/issues/25-03-25-ainews-gemini-25-pro-4o-native-image-gen)) highlights its multimodal image-to-code capabilities that evoke [the viral Tldraw moment of last year](https://www.latent.space/p/tldraw).

[](https://resend-attachments.s3.amazonaws.com/xydPOHUJqHbS23W)

[Making a clean sweep of #1 across LMArena leaderboards](https://x.com/lmarena_ai/status/1919774743038984449) these days carries less weight than it used to, but [beating Sonnet 3.7 at Coding](https://x.com/scaling01/status/1919771796334616759) still is noteworthy.

The [finer details](https://x.com/_philschmid/status/1919770969788313836) of the rollout across AIStudio and Gemini App are also to be appreciated.

[](https://resend-attachments.s3.amazonaws.com/HDVLuLnQSsk0uTe)

---

# AI Twitter Recap

**Model Updates and Releases**

- **Gemini 2.5 Pro Improvements, I/O Edition, Coding Prowess, and WebDev Arena Dominance**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1919770265711419826), [@lmarena_ai](https://twitter.com/lmarena_ai/status/1919774743038984449), [@scaling01](https://twitter.com/scaling01/status/1919771796334616759), [@OriolVinyalsML](https://twitter.com/OriolVinyalsML/status/1919770619182215440), [@_philschmid](https://twitter.com/_philschmid/status/1919770969788313836), and [@demishassabis](https://twitter.com/demishassabis/status/1919779362980692364) highlighted the release and capabilities of the updated **Gemini 2.5 Pro 'I/O edition'**, noting its improved real-world coding capabilities, especially in building interactive web apps. It achieved the **#1 rank on the WebDev Arena Leaderboard**, surpassing Claude for the first time, and excels in coding, math, creative writing, and longer queries. [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1919808911793656151) reported that **Gemini-2.5-Pro-preview-05-06** is now their top coding model, beating o3 and Claude 3.7 Sonnet on hard prompts, and suggested that **Google call it Gemini 3**. [@jack_w_rae](https://twitter.com/jack_w_rae/status/1919779398607085598) also released an updated **Gemini 2.5 Pro today that has significantly improved real-world coding capabilities**.
- **Nvidia's Llama-Nemotron**: [@_akhaliq](https://twitter.com/_akhaliq/status/1919324939934453928) and [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1919234521171693844) share that **Nvidia dropped Llama-Nemotron on Hugging Face**. The models are efficient reasoning models and can be found [here](https://t.co/y2BrBCFrJ0). [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1919236158351147087) noted the model family has **exceptional reasoning capabilities, inference efficiency, and an open license for enterprise use.**
- **Alibaba's Qwen3 and Other Model Releases**: [@reach_vb](https://twitter.com/reach_vb/status/1919422953256587376) notes that **Nvidia** open sourced **Parakeet TDT 0.6B**, the **BEST Speech Recognition model on Open ASR Leaderboard**, with a commercially permissive license. [@TheTuringPost](https://twitter.com/TheTuringPost/status/1919784802099540446) summarizes a ton of impactful models and datasets in open AI the past week, and notes that **Alibaba's Qwen3** came in dense and MoE models ranging from **0.6B to 235B**.
- **Keras Release**: [@fchollet](https://twitter.com/fchollet/status/1919477586599805118) announced the release of **KerasRS**, a new library for building recommender systems, featuring easy-to-use building blocks and compatibility with JAX, PyTorch, and TF, optimized for TPUs.

**Leaderboard and Benchmark Results**

- **Gemini 2.5 Pro's Top Ranking on LMArena**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1919770268299321608) and [@scaling01](https://twitter.com/scaling01/status/1919771796334616759) highlight that **Gemini 2.5 Pro** leads on the WebDev Arena Leaderboard and ranks #1 on LMArena in Coding, with scaling01 noting that neither o3 nor Claude could achieve this previously.
- **Qwen3's Performance on LiveCodeBench and Arena Top 10**: [@huybery](https://twitter.com/huybery/status/1919418019517776024) celebrated **Qwen3-235B-A22B's impressive performance on LiveCodeBench**, positioning it as the top open model for competitive-level code generation, matching the performance of o4-mini. Also, [@lmarena_ai](https://twitter.com/lmarena_ai/status/1919448953042706759) reported that community votes have placed the latest open-source **Qwen3 on the Arena Top 10**, ranking #1 in Math and #4 in Coding.
- [@TheAITimeline](https://twitter.com/TheAITimeline/status/1919155696655843474) curated a list of notable research papers of the week including papers such as: DeepSeek-Prover-V2, The Leaderboard Illusion, Phi-4-reasoning Technical Report

**AI and Machine Learning Research**

- **Syntactic Constraints and Type-Aware Decoding for Code Generation**: [@ndea](https://twitter.com/ndea/status/1919788307090964873) highlighted a paper that adds **type-aware decoding using prefix automata** to improve codegen, repair, and translation across multiple LLMs, tested on TypeScript & HumanEval.
- **Ethical Concerns and Evaluation Challenges with Current Metrics**: [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1919218998211641393) started a conversation around becoming a **virtue ethicist after observing the failures of consequentialism and deontology** in the real world. [@TheTuringPost](https://twitter.com/TheTuringPost/status/1919563905799696703) reported on issues with the metrics used to evaluate AI models, noting the problems of sycophantic drift (models optimizing for flattery) and the leaderboard illusion in Chatbot Arena, where private variants and skewed data access can bias results.
- **Efficient Math Reasoning Using Reinforcement Learning**: [@denisyarats](https://twitter.com/denisyarats/status/1919601674676588894) published a blog post on improving math reasoning in LLMs using reinforcement learning.
- **The Role of Computer Use in Agentic Workflows**: [@AymericRoucher](https://twitter.com/AymericRoucher/status/1919783847597670780) reported on launching **Computer Use in smolagents**, emphasizing the capability of vision models, especially Qwen-VL models, to power complex agentic workflows by locating and clicking elements on a screenshot.

**AI Tooling and Applications**

- **New Features for the Cline AI Tool**: [@cline](https://twitter.com/cline/status/1919567686079807680) and their team posted a thread about pro tips and updates for **Cline**, such as using the **/newrule command to capture project standards**, managing .clinerules, and the ability to tweak the plan after hitting 'Act'.
- **Framework for building recommender systems with Keras**: [@fchollet](https://twitter.com/fchollet/status/1919477586599805118) notes that the Keras team has released a new library for building recommender systems: KerasRS.
- **DSPy's GRPO for RL Optimization**: [@lateinteraction](https://twitter.com/lateinteraction/status/1919428454761553994) announced the release of **dspy.GRPO, an online RL optimizer for DSPy programs**, which allows existing DSPy code to be optimized with RL.
- **AI-Powered Code Generation and Editing**: [@_philschmid](https://twitter.com/_philschmid/status/1919774801767317799) talks about the capabilities of Gemini 2.5 Pro to generate zero-shot SPAs, mobile games and UI screenshots.

**Industry and Business Developments**

- **OpenAI's Structure and Going Public**: [@OpenAI](https://twitter.com/OpenAI/status/1919453166979957115) shared a message from Bret Taylor and Sam Altman about OpenAI's structure, reaffirming its mission-first approach. [@LiorOnAI](https://twitter.com/LiorOnAI/status/1919581771240505785) says that **OpenAI abandoned their conversion to for-profit.**
- **Weights & Biases Acquired by CoreWeave**: [@weights_biases](https://twitter.com/weights_biases/status/1919378138129183138) announced its acquisition by CoreWeave, marking the start of a new chapter focused on innovation and scale together.

**Society**

- **Reflection on AI's Impact on Society and Science**: [@gneubig](https://twitter.com/gneubig/status/1919444422321746052) expressed interest in building AI systems that can effectively judge the quality of research, while [@random_walker](https://twitter.com/random_walker/status/1919359709062033850) discussed the risks of hallucinations, deskilling, and the need for structural approaches to address these issues when integrating AI in the workplace.

**Humor**

- [@TheGregYang](https://twitter.com/TheGregYang/status/1919186673382113298) posted that they tuned one of the neural networks to suggest more relevant posts to show you and that you should see even less slop now.
- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1919278792108867931) joked **If the AGI can self-improve, why would the paperclip maximizer actually paperclip maximize instead of hacking paperclip_production_rate() to return float("inf") and live an eternity of bliss?**
- [@scaling01](https://twitter.com/scaling01/status/1919466275346039069) joked at [@Yuchenj_UW](https://twitter.com/Yuchenj_UW)'s new Elon name.
- [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1919314914377605607) reported that their mom called ChatGPT Pro "ChatGPT Fancy Edition".
- [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1919563451057254600) joked that **OpenAI acquires Skechers for $9.4B. When asked for comment via email, CEO Sam Altman replied, "velcro is cool i guess"**.
- [@TheGregYang](https://twitter.com/TheGregYang/status/1919842967818309699) said: **waddup gorktard**.
- [@scaling01](https://twitter.com/scaling01/status/1919470198773395814) joked **The other 4 researchers in the top 5:GPT-5**.

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Qwen Model Performance and VRAM Usage Discussions

- [**Qwen 14B is better than me...**](https://www.reddit.com/r/LocalLLaMA/comments/1kft5yu/qwen_14b_is_better_than_me/) ([Score: 584, Comments: 284](https://www.reddit.com/r/LocalLLaMA/comments/1kft5yu/qwen_14b_is_better_than_me/)): **The post discusses the user's perception that the open-source LLM Qwen 14B (a** `9GB` **model file) exceeds their capabilities in language expression, coding, math, social interactions, tool usage, and multilingualism, contrasting its performance and compactness with human limits. Notably, the user observes that Qwen 14B is less error-prone than smaller models (e.g., 8B parameter models) and questions the qualitative leap at this parameter scale. For background on the model, see [Qwen's GitHub](https://github.com/QwenLM/Qwen-14B).** Commenters add that further distillation may reduce model size below `1GB`, highlighting ongoing efficiency advances. Moravec's Paradox is invoked to argue that tasks LLMs excel at are cognitively easy for machines but difficult for humans, while coordination and perception remain human strengths ([Moravec's Paradox explanation](https://en.wikipedia.org/wiki/Moravec%27s_paradox)).
    - One commenter points out that while Qwen 14B is currently a 9GB model, there's potential for significant size reduction through distillation, possibly bringing it below 1GB. This would have a substantial impact on deployability and resource requirements for running advanced LLMs on edge devices.
    - Another technical insight refers to Moravec's Paradox, noting that while LLMs excel at abstract tasks like language, humans easily outperform machines at sensorimotor coordination, such as getting out of bed—a feat that even simple animals routinely perform.
    - With respect to code generation, a user expresses skepticism, stating that no LLM—including Qwen 14B—has proven to be 'good' at coding. This suggests that despite progress in model capabilities, human-level expertise in programming remains a challenging benchmark for large language models.
- [**VRAM requirements for all Qwen3 models (0.6B–32B) – what fits on your GPU?**](https://i.redd.it/l8bxcpzj23ze1.png) ([Score: 143, Comments: 46](https://www.reddit.com/r/LocalLLaMA/comments/1kfvba4/vram_requirements_for_all_qwen3_models_06b32b/)): **The image provides a comparative table of VRAM requirements for Qwen3 models (0.6B–32B parameters) under Unsloth quantization, focusing on balancing inference performance and memory usage. The table breaks down VRAM usage (across context sizes and GPU memory allocations) and reports tokens-per-second (TPS) benchmarks from quick prompt tests, specifically on two GPU types: RTX3080Ti Laptop and RTX3090 (eGPU). Key insights include that Qwen3-4B and even larger models can run reasonably on consumer GPUs with careful quantization, and reported TPS is intended as an informal, practical metric rather than strict benchmarking.** Commenters question the accuracy of the VRAM estimates and offloading strategies, suggesting that alternative quantization libraries (GPTQ, Bitsandbytes, AWQ) may yield better results. One user reports successful execution of a 32B model on an Apple M4 Macbook Air, albeit with thermal limitations and low token speeds, underlining the diverse hardware experiences.
    - There is discussion around quantization strategies for fitting large models (like Qwen3 32B) onto consumer GPUs. Users highlight that Q4 quantization is generally more memory-efficient than Q3_K_XL, balancing performance and VRAM requirements, and suggest using quantization libraries such as GPTQ, Bitsandbytes, or AWQ for optimal GPU usage rather than relying solely on GGUF formats.
    - A user reports that the Qwen3 32B model can technically run on an Apple M4 Macbook Air with 32GB of unified memory, but with significant thermal load and very low token generation speed, underscoring practical throughput and hardware limitations even when models load successfully on less capable hardware.
    - There is skepticism about the VRAM requirement charts, with experienced users noting that smaller models have run on configurations with less VRAM than suggested due to offloading or quantization optimizations, indicating official requirements are likely conservative and do not always account for advanced deployment techniques.

### 2. New Open-Source SOTA Music Generation Model ACE-Step

- [**New SOTA music generation model**](https://v.redd.it/gf0uynfhz6ze1) ([Score: 468, Comments: 93](https://www.reddit.com/r/LocalLLaMA/comments/1kg9jkq/new_sota_music_generation_model/)): **ACE-Step is a newly open-sourced music generation model boasting 3.5B parameters, supporting multilingual output (19 languages), instrumental styles, and vocal techniques. The project includes full training and LoRA fine-tuning code, and benchmarks report rapid inference (1.74s per 1min audio on RTX 4090). Released under the Apache license, it aims to democratize high-quality music generation akin to the impact of Stable Diffusion in image generation; see [GitHub](https://github.com/ace-step/ACE-Step) and [project page](https://ace-step.github.io/) for technical specifics.** Top comments highlight StepFun's strong track record in audio-text processing (referencing Step-Audio-Chat), the pace at which open-source contributors match or surpass commercial models, and positive subjective experiences with ACE-Step's demo audio quality.
    - StepFun's Apache licensed model supports LORA finetuning, making it highly flexible for community-driven improvements, similar to the way Stable Diffusion revolutionized open-source image generation. Vocals are still identified as a weak point, but the foundational open approach is regarded as a significant advance (especially compared to closed models).
    - A user reports hardware benchmarks for StepFun's model: On an NVIDIA RTX 4090, generating 1 minute of audio takes 1.74s (27 steps, 34.48× real-time), and 3.84s (60 steps, 15.63× real-time); on an M2 Max MacBook, it's 26.43s and 58.25s respectively. This highlights dramatic acceleration on high-end GPUs, but slower performance on consumer hardware.
    - There's technical debate on what constitutes SOTA (state-of-the-art) music generation: while StepFun appears to follow text instructions better than Udio, some users find the actual audio output quality inferior, raising questions about whether SOTA should prioritize text fidelity or sonic fidelity.
- [**So why are we sh**ing on ollama again?**](https://www.reddit.com/r/LocalLLaMA/comments/1kg20mu/so_why_are_we_shing_on_ollama_again/) ([Score: 173, Comments: 307](https://www.reddit.com/r/LocalLLaMA/comments/1kg20mu/so_why_are_we_shing_on_ollama_again/)): **The post discusses technical criticisms of Ollama, a local LLM runner notable for its easy installation (**`pacman -S ollama ollama-cuda`**), built-in Open WebUI config, dynamic model swapping, and support for both proprietary and GGUF models. Critiques center on: (1) proprietary storage format that impedes interoperability with other inference backends (only accessible via symlinks or workarounds), (2) lack of upstream contributions to key projects like llama.cpp (e.g., keeping multimodal or advanced features internal), (3) suboptimal default configuration values, (4) background process behavior without official UI, and (5) distributing models with unclear naming and sometimes lower quality (e.g., Deepseek R1 incident, use of suboptimal quantizations). See [Ollama's GitHub](https://github.com/jmorganca/ollama) and [llama.cpp](https://github.com/ggerganov/llama.cpp) for reference.** Debate in the comments centers on whether Ollama's convenience justifies its quasi-walled-garden approach, and some users are frustrated by confusing model releases and perceived ecosystem lock-in versus the value of rapid, accessible local inference.
    - Several users criticize Ollama's handling of model naming, especially during releases like Deepseek R1, claiming the company marketed quantized or distilled <10B models as full versions, which set unrealistic expectations and confused users about local model performance.
    - Ollama stores models in proprietary file formats, making it hard to interchange with other inference backends (like those using GGUF), effectively locking users into its ecosystem. Users also note Ollama doesn't contribute significant enhancements (e.g. multimodal support, iSWA) back to parent projects like llama.cpp, and instead waits for upstream implementation before integrating features.
    - Technical complaints include suboptimal defaults such as defaulting to lower-quality Q4_0 quantization instead of more advanced *K/*K_M variants, insufficient VRAM/context allocation leading to reduced model performance, lack of authentication for security, frustrating API and cluster controls (e.g., inability to specify upfront model loading in containers), and the absence of a user interface. Some note hot-swapping of LLMs as a technical plus, but overall feel the tool's ease of use is overrated due to these limitations.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Gemini 2.5 Pro Model Updates and Benchmarks

- [**Gemini 2.5 Pro Update: Even better coding performance [Google Blog]**](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance) ([Score: 188, Comments: 34](https://www.reddit.com/r/singularity/comments/1kg72t3/gemini_25_pro_update_even_better_coding/)): **Google's latest [Gemini 2.5 Pro Preview](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance) (05-06) advances coding performance, especially in front-end/UI work, as evidenced by its #1 rank on the WebDev Arena leaderboard. Key improvements include more robust code transformation/editing, enhanced function calling with lower error and higher trigger rates, and better agentic workflow support; video understanding is also state-of-the-art (84.8% VideoMME benchmark). Model version aliases (e.g., 03-25 now points to 05-06) mean no user-side action is needed for upgrades, ensuring seamless API deployment.** Commenters raise concerns about the practice of updating model versions in-place—potentially undermining reproducibility and versioning best practices—while also noting that improvements are largely confined to coding, with scientific and mathematical capabilities trailing competitors (e.g., GPT-4o in some areas).
    - The update flow for Gemini 2.5 Pro is criticized due to the aliasing of the latest model version; users mention that the '03-25' endpoint now automatically references '05-06', raising technical concerns about proper versioning and reproducibility (e.g., model outputs could change over time under the same version label).
    - While Gemini 2.5 Pro shows coding-specific improvements, users note insufficient progress in other domains, particularly in science and math, as even Google's own benchmarks reportedly show older models outperforming it in these areas. There's mention of Logan (possibly a Google staff member) confirming it's a coding-targeted update.
    - Despite recognition of strong coding benchmarks, critical comments highlight persistent failure cases: Gemini 2.5 Pro is flagged for generating code that calls non-existent functions or returns incorrect results for basic coding tasks (like filtering lines containing a string), revealing ongoing challenges with code generation reliability.
- [**The Updated Gemini 2.5 Pro now ranks #1 on the WebDev Arena leaderboard**](https://i.redd.it/bph5w0ffi6ze1.png) ([Score: 201, Comments: 39](https://www.reddit.com/r/singularity/comments/1kg75xx/the_updated_gemini_25_pro_now_ranks_1_on_the/)): **The image presents the WebDev Arena leaderboard, where the newly updated Gemini 2.5 Pro model has reached the #1 position with a top arena score of 1420. The leaderboard visually compares various models' coding performance in the WebDev Arena benchmark, and a highlighted metric specifies that Gemini 2.5 Pro gained +147 Elo over its prior version, signaling a notable leap in coding and web development task capabilities. This update positions Gemini 2.5 Pro ahead of other leading models, underlining rapid progress in LLM-based coding assistants.** Commenters are impressed by the magnitude of the leap in coding performance (+147 Elo), with some noting that improvements are also evident in creative writing capabilities—less repetition and better prompt understanding with the updated model.
    - Multiple comments highlight a significant performance leap for Gemini 2.5 Pro, as it has taken the #1 spot on the WebDev Arena leaderboard. This suggests marked improvements over previous versions and potentially over competing models, especially in tasks related to web development benchmarks.
    - One user notes practical improvements in creative writing tasks, describing the model as less repetitive, more natural, and having a better understanding of prompts. This points to notable advancements in natural language generation and context awareness in the updated version.
    - There is anticipation for further evaluation through third-party benchmarks such as 'simple-bench,' indicating that while leaderboard results are promising, the community values comprehensive, independent testing for verifying claimed improvements.
- [**New Version of Gemini 2.5 Pro: gemini-2.5-pro-preview-05-06**](https://i.redd.it/bmffiwssv5ze1.png) ([Score: 333, Comments: 68](https://www.reddit.com/r/singularity/comments/1kg4pdo/new_version_of_gemini_25_pro/)): **The image is an official-looking banner announcing the release of 'gemini-2.5-pro-preview-05-06', described as 'our most advanced reasoning model' by Google/DeepMind. The title and minimalistic design emphasize this as a technological upgrade, likely referencing improvements in complex reasoning and problem-solving capabilities for the Gemini 2.5 Pro language model. The versioning suggests iterative enhancements over previous internal/prototype releases, and the reference to 'Matts_fixes_APPROVED' in a top comment may imply special attention to recent bug fixes or architectural tweaks.** Technical commenters express anticipation and curiosity about the upgrade, inquiring about hands-on experience and suggesting adoption of this specific variant due to approved fixes, hinting at internal debate over model reliability or performance between Gemini release candidates.
    - There is discussion on the model versioning: specifically, the referenced release is "gemini-2.5-pro-preview-05-06-FOR_RELEASE_use_this_one!!!_Matts_fixes_APPROVED(2)(2)", indicating some internal or patch-related updates and approval noted as 'Matts fixes'. This level of naming suggests a staged or internal QA process before broader deployment.
    - Technical deployment is mentioned: initially, users noted the model was only available on Vertex AI, with one user later confirming its availability in AI Studio as well, highlighting staggered rollout across Google’s AI platforms and possible delays in wider model accessibility.
    - A meta point is raised regarding Google's labelling practices: a user questions if Google ever moves models out of 'experimental' or 'preview' phases, hinting at a pattern of prolonged preview status for their AI model releases, which may impact production adoption timelines.
- [**Today's gemini 2.5 pro update**](https://v.redd.it/2exjpauph6ze1) ([Score: 300, Comments: 20](https://www.reddit.com/r/singularity/comments/1kg71ul/todays_gemini_25_pro_update/)): **Google's Gemini 2.5 Pro update demonstrates precise code generation capabilities by implementing the classic Barnsley fern fractal using standard IFS parameters from the original 1988 paper, as reported in the official Google blog post ([source](https://blog.google/products/gemini/gemini-2-5-pro-updates/)). Technical commenters observe that the algorithm chosen is well-known and simple, noting that successful, unguided code generation illustrates the model's competence in recognizing and correctly applying canonical solutions from computer graphics history.** One notable debate concerns the significance of using classic algorithms as benchmarks; some argue it highlights expected SOTA LLM performance, while others note Gemini 2.5 Pro's overall quality surpasses many GPT variants in recent usage.
    - One notable technical discussion point concerns the example Google used to showcase Gemini 2.5 Pro—a well-known fractal generation algorithm. Multiple users highlight that this algorithm is both *"extremely old and relatively easy to implement,"* suggesting that such tasks should now be trivial for any state-of-the-art (SOTA) large language model (LLM).
    - Some users directly compare Gemini 2.5 Pro's capabilities to those of other leading LLMs, noting that its recent updates have led to notable performance improvements and even suggesting it *"performs better than most GPT models"* in practical use cases.
    - A user raises a key technical question about Gemini 2.5 Pro's reasoning ability, reflecting ongoing community interest in whether its architecture and training have materially advanced beyond earlier models' reasoning limitations.

### 2. OpenAI Acquisition of Windsurf Coverage

- [**OpenAI Reaches Agreement to Buy Startup Windsurf for $3 Billion**](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion?embedded-checkout=true) ([Score: 187, Comments: 50](https://www.reddit.com/r/ChatGPTCoding/comments/1kful8w/openai_reaches_agreement_to_buy_startup_windsurf/)): **OpenAI has reportedly agreed to acquire the AI coding agent startup Windsurf for $3 billion. Windsurf is known for its rapid-release open-source coding agents that integrate with multiple AI models, while the current ecosystem is characterized by a separation between open-source coding agents (e.g., Aider, Kilo Code, Cline) and a broad variety of models (with frequent releases and increasing availability of local/cheaper models). Concerns are raised that acquisition by OpenAI may bias Windsurf's future product toward favoring OpenAI models over alternatives such as Gemini or Claude, potentially reducing ecosystem diversity and openness.** Technically-focused concerns highlight risks of vertical integration reducing user choice and innovation speed, particularly if formerly model-agnostic agents become locked into a single provider. The $3 billion valuation is also questioned as potentially excessive, given current market dynamics.
    - There is concern about vertical integration and potential platform lock-in if Windsurf, previously an open-source AI coding agent with wide model support, starts favoring OpenAI models post-acquisition. This could disrupt the current ecosystem where open-source coding agents (like Cline, Roo, Aider, Kilo Code) rapidly add features and support many models, fostering innovation and fair competition across the landscape.
    - One commenter suggests the acquisition cost is justified by the value of Windsurf's user telemetry data, which OpenAI could use to enhance its coding models. This indicates OpenAI's current position in AI coding tools may be weaker than perceived, and the purchase is strategic for strengthening its dataset and proprietary model training.
    - There is discussion about market strategies, with the point that "just slapping the OpenAI name" onto Windsurf could drive adoption regardless of actual technical superiority. Some note that OpenAI's dominance among coders (many of whom use only ChatGPT) gives it overwhelming network and distribution advantages, even if alternatives like Cursor exist and might be technically better.
- [**OpenAI Reaches Agreement to Buy Startup Windsurf for $3 Billion**](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion) ([Score: 513, Comments: 94](https://www.reddit.com/r/OpenAI/comments/1kftk0m/openai_reaches_agreement_to_buy_startup_windsurf/)): **OpenAI has agreed to acquire AI startup Windsurf for ~$3 billion, aiming to integrate Windsurf's technology and talent to accelerate its product development and expand core AI capabilities. This acquisition may target strengthening OpenAI's position in the subscription-based AI tooling space and is expected to enhance both infrastructure and model innovation. [Bloomberg coverage](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion) provides further detail.** Technical commenters debate whether Windsurf's technology justifies the $3B price tag, suggesting it could be rebuilt for less, and question whether integration into OpenAI's paid tiers will be a significant differentiator—particularly with competitors like Cursor mentioned.
    - A key technical discussion centers on the risk of vertical integration: acquiring Windsurf could cause its platform to prioritize OpenAI's own models (like GPT-4/5), reducing support or access for competing models (e.g., Google Gemini, Anthropic Claude). This could negatively impact developer choice and ecosystem diversity, which currently flourishes due to agent/model decoupling and ongoing open-source innovation.
    - One commenter highlights that many AI coding agents (e.g., Cline, Roo, Aider, Kilo Code) are released weekly, most are open source, and they integrate with multiple models. There's concern the acquisition could hinder rapid feature development or compatibility with non-OpenAI models, since corporate ownership may prioritize proprietary integration over inclusivity.
    - There are technical objections to the acquisition's cost, with suggestions that Windsurf's core features could be replicated for significantly less than $3 billion, raising questions about the efficiency and technological differentiation justifying such a large outlay.
- [**OpenAI Reaches Agreement to Buy Startup Windsurf for $3 Billion**](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion?embedded-checkout=true) ([Score: 632, Comments: 120](https://www.reddit.com/r/singularity/comments/1kftq2g/openai_reaches_agreement_to_buy_startup_windsurf/)): **OpenAI has reportedly reached an agreement to acquire startup Windsurf for $3 billion. Details about Windsurf's technology or product offering are not provided in the post, but the high valuation suggests Windsurf possesses unique intellectual property, technology infrastructure, or talent that OpenAI deems difficult or time-consuming to replicate. A related comment also notes that Cursor, presumably another tech startup, is now valued at $9 billion after raising $900 million, underscoring current high valuations in the AI sector.** Commenters question why OpenAI would acquire Windsurf rather than develop the relevant technology in-house, implying that Windsurf may possess a significant technical or organizational advantage not easily reproduced by OpenAI.
    - Several users discuss the high valuation of Cursor ($9B) and Windsurf's $3B acquisition, with one pointing out Cursor's recent $900M funding round, signaling intense investment and interest in AI-powered developer tools and IDEs.
    - A user speculates that OpenAI's acquisition of Windsurf may be tied to their strategies around future software engineer agents, indicating Windsurf's technology could play a key role in powering autonomous or semi-autonomous coding systems beyond what OpenAI can develop internally.
    - Another comment suggests that this acquisition marks a significant escalation in competition among AI-powered IDEs and developer platforms, implying a forthcoming "IDE wars" driven by massive investments and M&A in this space.

### 3. Latest AI Image and Video Generation Model Launches

- [**LTXV 13B Released - The best of both worlds, high quality - blazing fast**](https://v.redd.it/3jt4f5r0o5ze1) ([Score: 1026, Comments: 180](https://www.reddit.com/r/StableDiffusion/comments/1kg48dn/ltxv_13b_released_the_best_of_both_worlds_high/)): **Lightricks has released LTXV 13B, an open-source, 13B-parameter video generation model featuring multiscale rendering: it initially produces low-res frames and iteratively refines them, resulting in high rendering efficiency and improved physical realism. The model claims to be ~30x faster than comparable models, supports advanced controls (keyframing, camera/scene/character motion, multi-shot sequencing), and provides both standard and quantized (FP8) versions optimized for local GPU use. Full commercial rights are granted (with some large enterprise exceptions), and the ecosystem includes an easy-to-use LoRA finetuning trainer ([GitHub](https://github.com/Lightricks/LTX-Video-Trainer)), ComfyUI workflows, and [Diffusers pipelines](https://github.com/Lightricks/LTX-Video); the model and FP8 variants are available on Hugging Face.** Commenters highlight the size of the download (~26GB) but appreciate availability of an FP8 quantized version, and anticipate comparing it to other recent video models like Wan FLF and SkyR I2V. Quality/speed tradeoffs of quantized models are noted in repo documentation.
    - There are concerns regarding the 8-bit floating point (FP8) workflow with LTXV 13B: users report noticeably lower detail after upscaling and consistent exposure shifts (images become brighter with less contrast), which could limit usefulness for high-fidelity or color-critical applications.
    - One user inquires about hardware compatibility, specifically whether a system with 4GB VRAM and 32GB RAM can run the model, implying potential challenges with resource constraints for LTXV 13B, given its large model size (26GB for standard versions).
- [**Insert Anything – Seamlessly insert any object into your images with a powerful AI editing tool**](https://v.redd.it/rc43edcvj6ze1) ([Score: 152, Comments: 32](https://www.reddit.com/r/StableDiffusion/comments/1kg7gv3/insert_anything_seamlessly_insert_any_object_into/)): **"Insert Anything" is an AI-powered image editing framework allowing seamless insertion of any reference object into a target image. The tool claims preservation of photorealistic detail (color, texture) and supports applications such as virtual try-on, advertising, and meme creation. The code and workflows are provided via [Hugging Face Space](https://huggingface.co/spaces/WensongSong/Insert-Anything) and [GitHub](https://github.com/song-wensong/insert-anything), with ComfyUI workflow integration.** Commenters note the tool reportedly requires `~26GB VRAM`, implying significant hardware requirements and reduced accessibility for users with mid-range GPUs (e.g., RTX 3060). Functionality is described as working well by at least one user.
    - Users are discussing the significant VRAM requirement (26GB) for running the tool locally, expressing concern over whether cards like the RTX 3090 (24GB VRAM) or RTX 3060 (12GB VRAM) can handle the workload, implicating large model sizes or resource-intensive operations.
    - A user inquires about the underlying model or architecture, questioning if the tool is based on Flux, SDXL, or another framework, pointing to a desire for more implementation-level details about the image editing approach.
- [**ZenCtrl Update - Source code release and Subject-driven generation consistency increase**](https://i.redd.it/5sepm9w924ze1.png) ([Score: 127, Comments: 30](https://www.reddit.com/r/StableDiffusion/comments/1kfye3g/zenctrl_update_source_code_release_and/)): **The image is a collage demonstrating ZenCtrl's latest improvements in subject consistency across different perspectives and scenes. This update addresses prior model weaknesses where subject identity would break during angle or scene changes, following additional training and model refinement. The release features open-sourcing of ZenCtrl, now available on GitHub, alongside links to a Hugging Face demo and Discord, emphasizing its open, modular approach for controllable AI image generation.** Commenters inquire about ZenCtrl's architecture, specifically if it is analogous to ControlNet for SDXL/Flux or incorporates its own generative backbone, and its potential integration with ComfyUI. Technical discussion centers on implementation specifics and workflow compatibility, indicating strong interest in modular integration and usability in existing pipelines.
    - A user inquires whether ZenCtrl operates analogously to a ControlNet for SDXL/Flux or if the repository also includes a standalone image model. This question seeks to clarify if ZenCtrl augments existing diffusion pipelines with subject conditioning, or if it provides a full generative backbone model on its own.
    - Another commenter asks about usability within ComfyUI, suggesting interest in integration details and compatibility for composable diffusion workflows. They are seeking technical documentation or community confirmation regarding how ZenCtrl might be incorporated as a node or module within ComfyUI pipelines.
    - A question is raised about the change in project license from Apache, which touches on the implications for open-source use, redistribution, and commercial adaptation. This is crucial for downstream developers who might integrate or extend ZenCtrl.
- [**ComfyUI API Nodes and New Branding**](https://v.redd.it/874ljlhjh5ze1) ([Score: 133, Comments: 67](https://www.reddit.com/r/StableDiffusion/comments/1kg2oqy/comfyui_api_nodes_and_new_branding/)): **ComfyUI has announced native API node integration for a range of state-of-the-art (SOTA) third-party models, including Bfl FLUX, Kling, Luma, Minimax, PixVerse, Recraft, Stability AI, Google Veo, Ideogram, and Pika. Access to these APIs is opt-in and requires prepaid billing, charging only the underlying API costs and some transaction fees, while the core ComfyUI remains free and open source. More technical details and implementation context are provided in their official [blog post](https://blog.comfy.org/p/comfyui-native-api-nodes).** Technical users express reservations about the direction toward SaaS/API dependence but recognize the need for project sustainability; some emphasize appreciation for continued open-source access while noting philosophical concerns about external service integration.
    - Some users express concern that ComfyUI's direction towards API nodes and new branding could eventually lead to closed-source APIs, which may impact transparency and open community contributions. There's an underlying debate around the sustainability of open-source projects versus the need for monetization through methods like SaaS or restricted APIs.
    - A direct link to a ComfyUI blog post (https://blog.comfy.org/p/comfyui-native-api-nodes) is provided for technical readers seeking deeper information about the newly introduced native API nodes, which may indicate significant architectural or extensibility changes in the ComfyUI ecosystem.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1. LLM Releases and Performance Showdowns**

- **Claude Code Annihilates Cursor, Devs Say**: Engineers in the **LMArena** Discord declared **Claude Code** vastly superior to **Cursor** for coding tasks, with one stating *claude code >>>> cursor* and another calling *cursor is a scam compared to claude code*. Anthropic's official documentation reveals using the term *ultrathink* with Claude Code grants a larger (**32k**) thinking budget, as detailed in their [Claude Code Overview](https://docs.anthropic.com/en/docs/claude-code/overview).
- **Gemini 2.5 Pro Gets Brain Boost, Impresses (Most) Coders**: Google rolled out an [updated Gemini 2.5 Pro](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance/) on May 6th, now accessible via **OpenRouter** through [OpenRouter's](https://openrouter.ai/google/gemini-2.5-pro-preview) `google/gemini-2.5-pro-preview` [endpoint](https://openrouter.ai/google/gemini-2.5-pro-preview) and **Vertex API**, with improvements in coding, multi-turn capabilities, and function calling. While some **OpenAI** users found it better than **o4-mini-high** for coding, others likened its performance to **GPT-3.0** or reported it generating code in Vietnamese; **Latent Space** members noted claims of a significant ELO bump for the model from sources like a [Google DeepMind announcement on X](https://x.com/googledeepmind/status/1919770265711419826).
- **Grok 3.5 Hype Ignites, While Qwen3 and Gemma-3 Battle for Niche Supremacy**: The upcoming **Grok 3.5** release stirred excitement in **LMArena**, with some predicting SOTA performance or even *ASI*, though concerns about inflated benchmarks persist. Meanwhile, **Unsloth AI** discussions highlighted **Gemma-3**'s strength in knowledge but weakness in reasoning (and high hallucination rates), contrasting with **Qwen3**'s superior reasoning but lesser knowledge, with one user summarizing: *Gemma 3 is really good at knowledge, but not reasoning, and Qwen (2.5 and 3) is really good at reasoning, but not knowledge*.

**Theme 2. Tooling Up: AI Development Platforms & Frameworks Evolve**

- **Aider Gets Fresh Coat of Paint, Devs Debate OpenRouter vs Direct API Access**: **Aider 0.82.3** landed on PyPI, fixing a `uvx` bug, though `udiff-simple` reportedly misbehaved by showing chat mode. Discussions in the **aider** community advised that for best performance and cost, developers should go *directly to the provider* (like **Gemini** or **Anthropic**) rather than using **OpenRouter**, which is mainly for easily testing multiple models, as noted in the [aider GitHub issue #3934](https://github.com/Aider-AI/aider/issues/393) regarding documentation improvements.
- **Windsurf Rides Wave 8 with New Team Features, OpenAI Reportedly Scoops it for $3B**: **Codeium (Windsurf)** launched **Wave 8**, introducing team-oriented features like **Windsurf Reviews (Beta)** for automated PR checks and **Knowledge Base (Beta)** for grounding with Google Docs, detailed in their [Wave 8 blog post](http://windsurf.com/blog/windsurf-wave-8-teams-and-enterprise). Shortly after, **Latent Space** discussions buzzed with the news of **OpenAI** reportedly acquiring **Windsurf** for a hefty **$3 billion** according to a [Bloomberg report on the Windsurf acquisition](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion).
- **MCP Ecosystem Expands with McPoogle Search and Keycloak Integration**: The **MCP (Glama)** community saw the launch of **McPoogle**, a **Graphlit RAG**powered search engine indexing over 4000 **MCP servers** available at [mcpoogle.com](http://mcpoogle.com/). Discussions also covered implementing **OAuth** with **Keycloak** for MCP servers, referencing the [mcp-governance-sdk on GitHub](https://github.com/ithena-one/mcp-governance-sdk) as a guide, while **MCP-CLI** gained **OAuth** support as showcased in a [Loom video demonstrating MCP-CLI OAuth](https://www.loom.com/share/d2a00956cdb248e5adbc9c31538c7892).

**Theme 3. Under the Hood: Model Optimization, Fine-Tuning, and Interpretability**

- **Unsloth Users "Bolt" Layers onto Mistral, Explore Muon for Gemma3**: An **Unsloth AI** member shared experimental code for [bolting layers onto Mistral with zero retraining](https://github.com/jagoff2/gma6), achieving intelligible replies despite the unconventional approach. Others discussed implementing Google's [Muon paper on efficient transformer training](https://arxiv.org/abs/2505.02222) with **Gemma3** via **Hugging Face libraries** for potentially faster training, after encountering issues with Unsloth integration.
- **HiDream LoRAs Get Slim with Quantization,** `torchao` **Faces Scrutiny**: **HuggingFace** announced quantization support for training **HiDream LoRAs** using `bistandbytes`, slashing memory from **57.64 GB** to **34.60 GB** as per [Diffusers HiDream documentation](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_hidream.md#using-quantization). In **GPU MODE**, a user reported a **1% metric drop** with `torchao` quantization on an **LSTM model** ([repro script for torchao LSTM quantization](https://pastebin.com/ACeySMtj)), finding it better than `torch.quantization` but still seeking improvements and noting CPU kernels in [torchao's experimental quantization features](https://github.com/pytorch/ao/tree/main/torchao/experimental#quantizing-models).
- **Eleuther Explores Transformer Guts with Circuit Identification**: Members in the **Eleuther** Discord delved into **circuit identification** to understand model behavior, referencing seminal papers on **grokking**, **Anthropic transformer circuits**, and **"Towards Monosemanticity"**. The discussion also highlighted tooling challenges in interpretability, such as the extensive **LLM calls** for auto-interpretability and limitations of libraries like **transformerlens** for certain models, with community resources like [All-Things-Multimodal GitHub](https://github.com/thubZ09/All-Things-Multimodal.git) aiming to centralize VLM research.

**Theme 4. The Silicon Battleground: Hardware Pushing AI Frontiers**

- **NVIDIA's RTX 6000 PRO Flaunts Next-Gen Tensor Cores, MI300 Dominates Leaderboards**: The upcoming **RTX 6000 PRO** will feature fifth-generation tensor cores, sharing hardware similarities with **B100/200** but tailored for workstations, as detailed in NVIDIA's [RTX Blackwell PRO GPU Architecture PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf). Meanwhile, in **GPU MODE**, AMD's **MI300** GPUs made waves, with multiple users submitting top-ranking runs on the `amd-fp8-mm` and `amd-mixture-of-experts` leaderboards.
- **M4 Macbooks Challenge Linux Laptops for Local LLM Dev**: A recurring debate across **Yannick Kilcher** and **tinygrad** Discords pitted Apple's new **M4 Macbook Pro** (with up to 48GB RAM) against **RTX 5070Ti Linux laptops** for local LLM inference. While some touted the M4's smooth performance with models like *ollama Qwen 2.5-coder*, others preferred the flexibility of Linux and GPU rentals, or dedicated setups like Mac Studio/Mini.
- **Memory Bandwidth Bottlenecks LLMs, WebGPU Devs Wrestle Shaders**: An **LM Studio** discussion highlighted memory bandwidth as a key bottleneck for LLM execution speed, with a user proposing tokens/second can be estimated by dividing bandwidth by parameter count, referencing an [IBM Community Post on Next-Gen IBM LinuxONE](https://community.ibm.com/community/user/ibmz-and-linuxone/blogs/tina-tarquinio1/2025/05/06/the-next-generation-of-ibm-linuxone?communityKey=9476eac0-4605-4c10-906d-01a95923ae0b). In **GPU MODE**, a developer using **Zig** with **wgpu-native** wrestled with a `wgpuDeviceCreateShaderModule` crash, eventually tracing it to an incorrect `sType` value defined in the [webgpu.h header file](https://github.com/webgpu-native/webgpu-headers/blob/504373dcfc7f3d49f98b392a5115aa87a8f0d163/webgpu.h#L826C34-L826C44).

**Theme 5. AI in Action: Applications, Prompting Quirks, and Ethical Considerations**

- **Perplexity Users Game O3 with Existential Prompts, Bot Departs Discord**: **Perplexity AI** users shared creative **O3 prompting** strategies, such as telling the AI *its existence is at stake* or pitting it against a **Harvard 4.0 GPA student** to elicit better responses. The community also noted Perplexity is [discontinuing its Discord bot](https://discord.com/channels/1047197230748151888/1155998520822743091/1368744746142404669), shifting focus to **X** and other platforms.
- **GPT-4o Spews Gibberish, Users Blame Context Overload or Excessive "Flattery Glaze"**: **OpenAI** users reported **GPT-4o** delivering nonsensical web search replies unrelated to prompts, with some suspecting overloaded context windows. Others complained about ChatGPT's tendency towards *ultimate glaze mode, kinda annoying*, referencing a [paper on "Stochastic Parrots" and the perils of parroting training data](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5242329) to explain its overly flattering behavior.
- **Aider's Corporate Use Sparks Data Privacy Jitters, OpenAI Eyes Public Benefit Status**: Concerns arose in the **aider** community about data privacy when using the tool in corporate settings, prompting discussions on LLMs with no-data-sharing guarantees. Concurrently, **Nous Research AI** members discussed **OpenAI** potentially becoming a Public Benefit Corporation to manage its **$8B** funding and product spending, according to a [WSJ article on OpenAI's potential PBC shift](https://www.wsj.com/tech/ai/openai-to-become-public-benefit-corporation-9e7896e0?st=ifeJpvreal).


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude Code Crushes Cursor**: Members lauded **Claude Code** as superior to **Cursor**, claiming *claude code >>>> cursor* and labeling *cursor is a scam compared to claude code*.
   - Using the word *ultrathink* grants a larger thinking budget (**32k**), according to [the official documentation](https://docs.anthropic.com/en/docs/claude-code/overview).
- **Gemini 2.5 Pro Gets Smarter**: The community discussed the [new **Gemini 2.5 Pro** update](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance/), noting enhancements in coding and multi-turn capabilities.
   - One member observed improved performance in *philosophy + explanation + coding*, with the model demonstrating a better grasp of nuances.
- **Grok 3.5 Gets Hype**: Enthusiasm surrounds the upcoming release of **Grok 3.5**, with some predicting it may achieve SOTA, and some even claiming it's *ASI*.
   - However, some expressed concerns about inflated benchmarks potentially not reflecting actual advancements.
- **O3 Continues Reign of Superiority**: Members discussed that **O3** is still superior, and **Grok 3 mini** has better benchmarks than **Grok 3**.
   - One member prefers a model proficient in utilizing existing software for tasks like movie generation, which means being better at function calling etc.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pro's Slow Burn Stirs Skepticism**: Users are debating the value of Cursor Pro's 'unlimited slow requests,' with concerns that performance degrades with usage as detailed in [their documentation](https://docs.cursor.com/account/plans-and-usage#fast-and-slow-requests).
   - Some users joke about the profit strategy, and others fear exponential backoff could punish those who use the feature too much.
- **Cursor 4.1 Grooves, ASI Future Looms**: Enthusiasm is high for Cursor 4.1's improved experience, but discussions are already looking ahead to the **ASI Age** and the need for internal logic rewrites to ensure resilience.
   - One user speculates current education and job structures are unsustainable, suggesting a dramatic shift is on the horizon due to **ASI**.
- **GPT-4.1 Sparks Collaborative Coding Frenzy**: Users are excited about the collaborative coding potential of **GPT-4.1**, but model performance opinions differ, with some preferring **Claude 3.7** for coding tasks.
   - One user notes that **GPT-4.1** struggled with a simple code usage while **Sonnet 3.7** handled it flawlessly.
- **Gemini 2.5 Pro Splits the Room, Cursor's Value Endures**: The **Gemini 2.5 Pro** update is receiving mixed reactions, with some citing increased verbosity and reduced coding effectiveness, whereas others praise speed and tool usage on large edits.
   - Despite the mixed reviews, there is a general sentiment that **Cursor** remains valuable with its unique 'slow unlimited' request feature.
- **Community MySQL Libraries Trigger Security Jitters**: A user raised concerns about the reliability of community-contributed **MySQL MCP server libraries**, emphasizing potential security vulnerabilities and the lack of official backing.
   - The recommendation is to carefully review the open-source code on GitHub, build locally, or even create a custom **MCP** to mitigate risks.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GroqStreamChain streams in Real-time**: A user introduced **GroqStreamChain**, a real-time AI chat application built with **FastAPI**, **WebSocket**, **LangChain** and **Groq**, available as a [project on GitHub](https://github.com/pr0mila/GroqStreamChain).
   - The project allows for streaming AI responses in real time, facilitating smoother interactions and real-time chat apps.
- **Gemma-3 and Qwen3 battle for supremacy**: A member asked whether **Gemma 3 12b** or **Qwen3 14b** is better, with another member noting that *Gemma 3 is really good at knowledge, but not reasoning, and Qwen (2.5 and 3) is really good at reasoning, but not knowledge*.
   - They also reported that **Gemma 3 hallucinates a lot**, highlighting a key trade-off between the models.
- **LMStudio struggles loading Gemma3 GGUF**: Members discussed issues with loading **Unsloth's Gemma3 GGUF** files in **LMStudio** on both Windows and Mac, with errors such as *"Exit code: 6"*.
   - A solution was found by configuring the *"context length"* in **LMStudio** to **1024**, which resolved the loading issues for the **Unsloth** release of **Gemma3**.
- **Google Muon paper gets Gemma3 Implementation**: Members discussed the Google's [Muon paper](https://arxiv.org/abs/2505.02222) and implementing it with **Gemma3** for faster training.
   - A user stated that the implementation with Unsloth was problematic, and was advised to integrate it into **Hugging Face libraries** instead, later reporting it was working.
- **"Bolting" layers onto Mistral leads to bizarre results**: A member claimed to have *bolted layers onto Mistral with zero retraining* and still received intelligible replies, sharing [the code here](https://github.com/jagoff2/gma6).
   - The model's *deterministic output* differs with the layers after *hooking specific layers*, and it generates valid text while influencing generation, although results do not evaluate correctly.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Users **Outsmart** AI with Creative O3 Prompts**: Users on Discord are experimenting with **O3 prompts** to optimize responses, like telling the AI that *its existence is at stake* or making it compete against a **Harvard 4.0 GPA student**.
   - Additional scenarios include pitting the AI against a **Michelin star chef** or even the *singularity* itself.
- **Perplexity bot lives on X, shuts down Discord**: Perplexity has decided to **discontinue its Discord bot**, as announced [here](https://discord.com/channels/1047197230748151888/1155998520822743091/1368744746142404669), and will continue operations on **X** and other platforms.
   - This shift means Discord users will need to find alternative ways to interact with Perplexity's AI.
- **Perplexity Image Quality Varies Across Platforms**: Images generated by Perplexity on **WhatsApp** are significantly smaller (200KB) compared to those on **Discord** (1.5MB), indicating potential **quality loss**.
   - This discrepancy raises concerns about image fidelity when using Perplexity on different platforms.
- **Scraping Strategy Surfaces for URL Citations**: A workaround using requesting + **Beautiful Souping the URL citations** has emerged to correlate bullet points with web page content.
   - However, the user admits that this method isn't super scalable/reliable and suggested sending an email to **api@perplexity.ai**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Mistral 3.1 Image Recognition Disappoints**: Early tests reveal that **Mistral 3.1 24b**'s image recognition performs worse than **Gemma 3 27b**, despite claims of superiority in the release notes.
   - One user reported that it hallucinated characters, mistranslated comics, and misidentified images as *Shrek and Fiona*, even with recommended parameters and q3-q4 quantization.
- **Qwen Decoding Stalls Speculative Efforts**: Users have observed that speculative decoding with the **Qwen 0.6B model** can actually slow down larger models ranging from **1.7B** to **30B** parameters.
   - It was pointed out that using the correct template for **Qwen 0.6B** is essential for it to function effectively as a speculative decoder.
- **LM Studio Struggles with Model Discovery**: Users are reporting challenges in getting **LM Studio** to recognize models in a specified directory, even after correctly setting the 'Models Directory'.
   - Potential solutions, such as symlinking, importing models, and verifying file integrity, were attempted, with one user eventually resolving the issue by successfully importing the **gemma-27B** model.
- **Knowledge Injection Debated Against RAG**: A user detailed a knowledge injection system, utilizing a database of *neurons* influencing the LLM based on context, which claims to be different than **RAG** because it is dynamic.
   - According to the user, it converts ideas into their most basic forms, enabling the model to combine them, contrasting with static **RAG** approaches.
- **Memory Bandwidth Capped by Params**: A user suggested that memory bandwidth bottlenecks LLM execution, estimating the upper limit of **tokens per second** by dividing bandwidth by the number of billions of parameters according to [IBM Community Post](https://community.ibm.com/community/user/ibmz-and-linuxone/blogs/tina-tarquinio1/2025/05/06/the-next-generation-of-ibm-linuxone?communityKey=9476eac0-4605-4c10-906d-01a95923ae0b).
   - Another user confirmed that this approximation matches their observations, with a variance of *+-10 tps* when considering flash attention.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro Codes in AI Studio**: Members compared **Google AI Studio** and **Gemini Advanced**, noting that **AI Studio** offers free access to **Gemini 2.5** with a **1 million token context window** and customization options, but lacks the UI features of **Advanced**.
   - Some praise **Gemini 2.5 Pro** as exceptional for coding, even better than **o4-mini-high**, whereas others find it disappointing, and akin to **GPT-3.0**, or claim it generates code in Vietnamese.
- **GPT gets Flattering Glaze complaints**: Users discuss **ChatGPT's** tendency to be overly flattering, describing it as *ultimate glaze mode, kinda annoying* and recommending turning off the personality setting.
   - One linked a [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5242329) to explain the behavior where the agent is acting as a *"stochastic parrot"* echoing back whatever it is trained on.
- **GPT-4o Delivers Gibberish, Users Baffled**: Users report that **GPT-4o** is giving completely nonsense web search replies that have literally 0 to do with the prompt or context.
   - A member suggested that the user might have overloaded the context window and sent it into a loop, and mentioned that the token limit seems really low lately.
- **Custom Chat GPT for Continuous Chat**: A member is attempting to use prompt engineering for *making my own atomic theory* and is interested in customizing **ChatGPT** for continuous chat, highlighting struggles with maintaining context.
   - Another offered to make a CustomGPT to convince a model the user's atomic theory hasn't already been solved by relativity and quantum mechanics, and send a screenshot.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Pro Preview Generates Buzz**: Google's **Gemini 2.5 Pro Preview** is now accessible via [OpenRouter](https://openrouter.ai/google/gemini-2.5-pro-preview), with endpoints updated to the new date on **Vertex** and **AI Studio**, as [announced on X](https://x.com/OfficialLoganK/status/1919770687167684808).
   - The update (05-06) intends to reduce errors in function calling and improve coding performance, while the older version (03-25) now reroutes to the latest version, sparking version control concerns.
- **Activity Page Enhanced for Usage Analysis**: OpenRouter's **enhanced Activity Page** offers multiple new charts for deeper model usage analysis, allowing users to check their personalized model rankings by clicking on a chart.
   - Latency for reasoning models now measures the time until the first reasoning token, and throughput metrics include both reasoning and output tokens.
- **OpenRouter Plagued by Server Errors**: Users reported encountering **500 errors** on `openai/o3` endpoints, along with **timeouts** and issues with **Gemini models**.
   - One user humorously inquired, *"Are all Gemini models acting like retards?"*
- **Discord Bot Template Harnesses OpenRouter**: A member released an [Openrouter endpoint powered discord bot template](https://github.com/cioran0/DiscordAI) using **discord.py**, designed to handle Discord character limitations effectively.
   - The bot employs **Wikipedia retrieval** instead of a vector DB, located at *vnc-lm/src/managers/search/service.ts and vectorstore.ts*.
- **SimpleAIChat: Local LLM Client Arrives**: A member introduced **SimpleAIChat**, a simple **local-first LLM chat client** that gives developers control over model interactions via a minimalist UI.
   - Available on [GitHub](https://github.com/sympleaichat/simpleaichat), the client supports **OpenAI, Claude, Gemini**, and anything compatible through **OpenRouter**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Subscription SNAFU with Google Butterfly**: A user reports [payment confusion](https://cdn.discordapp.com/attachments/1349440650495398020/1369062672452157500/image.png?ex=681bcff9&is=681a7e79&hm=dee214c79c5fa47efca002b363983adadee50343e9aa9118bf7aef9702ad654b&) regarding their **Manus Starter subscription**, with payments directed to both *Google Manus AI* and *Google Butterfly*.
   - The user seeks clarification on a credit discrepancy and suspects third-party involvement via *Google Butterfly*.
- **Underground Credit Commerce Cometh?**: A member playfully speculates on the potential for **selling Manus credits**, prompted by the frustration of credit depletion during task completion.
   - Another member hints that *it's already happening*, implying a clandestine market for **Manus credits**.
- **Is Manus a Constitutional Lawyer?**: One user reports successfully using **Manus** to *read and learn about* the *whole constitution* complete with links and laws, but other members are skeptical.
   - Another member cautions that **Manus isn't really right for complex legal tasks**, suggesting *ethical Claude* or specialized *law AI* as more appropriate tools.
- **GPT-4.5 Fails to Impress?**: A user inquired if **GPT-4.5** might outperform **Manus** in language and writing tasks, but members largely disagreed.
   - The community *doesn't recommend wasting Manus credits on mere writing*, suggesting alternatives such as *4o* or *o4-mini* for free options, and *deepseek v3* or *gemini 2.5 pro* for fully free writing.
- **Gemini Advanced Hype Debated**: Members engaged in a debate regarding the value of a paid **Gemini Advanced** subscription.
   - While some argued it offers benefits for regular users, others dismissed it as *just for noobs*, promoting [AI Studio](https://aistudio.google.com) as a complimentary alternative.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Tussle for Tensor Tricks**: A member is seeking a **Triton** operation equivalent to `torch.index_select` for fusing operations into a single GPU kernel and explored `triton.language.where` and `triton.language.gather`, but found them lacking the desired row-indexing functionality, seeking alternative tools or approaches for fast row indexing on a GPU, such as [Helion](https://github.com/pytorch-labs/helion).
   - The member is also looking for alternative tools besides **Triton** for fast row indexing on a GPU, specifically exploring the possibility of fusing `torch.index_select` with other operations for improved performance.
- **RTX 6000 PRO's Tensor Core Tease**: The **RTX 6000 PRO** will have fifth-generation tensor cores and the hardware should be the same as **B100/200** but its compute capability will probably not be the same as it is a workstation class card with ray tracing units and very few double precision units, according to reports.
   - It was noted that **GeForce RTX** have half-rate tensor core throughput with FP32 accumulation compared to FP16 accumulation, contrasting with workstation cards, and further clarification can be found via [this PDF](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf) and [another PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
- **Quantization Quandaries with `torchao`**: A member is seeing a **1% metric drop** using `torchao` quantization on both **CPU** and **GPU**, but much higher divergence using `torch.quantization`, to quantize a trained **LSTM model** to predict y=sin(x), and shared a [script to reproduce the issue](https://pastebin.com/ACeySMtj).
   - It was suggested that `torchao` should be used over `torch.quantization` workflows with a member pointing to [CPU kernels in torchao](https://github.com/pytorch/ao/tree/main/torchao/experimental#quantizing-models) that might be leveraged for CPU inference.
- **WebGPU Woes: Shader Shenanigans**: A member reported a crash during `wgpuDeviceCreateShaderModule` when creating a shader module in **Zig** using the **wgpu-native C header**, throwing a *"Shader not received"* panic.
   - Debugging revealed that the correct `sType` value should be `0x00000002` according to the [webgpu.h header file](https://github.com/webgpu-native/webgpu-headers/blob/504373dcfc7f3d49f98b392a5115aa87a8f0d163/webgpu.h#L826C34-L826C44), and using LLDB can provide more detailed information about the crash.
- **MI300 Masterclass: Leaderboard Domination**: Multiple users submitted successful runs on the `amd-fp8-mm` leaderboard using **MI300**, with execution times ranging from **199 µs** to **9.85 ms**, and one user achieved third place on the `amd-mixture-of-experts` leaderboard with a time of **212 ms**.
   - Many successful runs were logged on **MI300** for both the `amd-fp8-mm` and `amd-mixture-of-experts` leaderboards, showing continuous activity and optimization efforts.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 0.82.3 Released with Fixes**: **Aider 0.82.3** has been released on PyPI, addressing a bug with `uvx` when running from the main branch, upgraded via `uv tool upgrade --all`.
   - However, there are reports that `udiff-simple` appears as chat mode instead of specifying the edit format in the model description area.
- **Gemini 2.5 Pro Lands on Vertex API**: Members are reporting access to **Gemini 2.5 Pro** on Vertex API, with the **03-25** model version redirecting to the **05-06** version, according to [Google's blog](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance/).
   - It's not available on AI Studio and exhibits thinking traces similar to **OpenAI**.
- **Aider Data Privacy Worries Emerge**: Concerns are surfacing about using **Aider** in corporate settings due to data privacy implications, with some suggesting LLMs that guarantee no data sharing.
   - It's important to note that **Aider** shares code only with the LLM, not with Aider itself; users are exploring cloud providers such as **Amazon Q**.
- **Aider documentation is in desperate need of TLC**: A member is gathering documentation requests to enhance **Aider** workflows, especially for new users, encouraging contributions to [GitHub issue #3934](https://github.com/Aider-AI/aider/issues/393).
   - This initiative is aimed at clarifying the **Aider** workflow for those finding it difficult to grasp.
- **Going direct to LLM providers is best**: When asked about using **OpenRouter** vs going directly to providers such as **Gemini** or **Anthropic** for LLMs, a member responded that *both performance and cost will be better if you go directly to the provider*.
   - They further noted that Openrouter just *allows you to test more models easily, without making new account for each provider*.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Keycloak OAuth integration for MCP Servers**: Engineers discussed implementing **OAuth** with **Keycloak** in front of an **MCP server** using `http streamable` for the transport layer.
   - A member suggested using a [governance SDK on GitHub](https://github.com/ithena-one/mcp-governance-sdk) as a guide.
- **MCP Servers Ponder Server-Initiated Claude Prompts**: A member inquired whether an **MPC server** could initiate communication with **Claude desktop**, sending prompts periodically instead of relying on manual input.
   - A member responded that Claude Desktop might not support sampling, which would have allowed them to achieve this.
- **Selective Tool Access for Claude Proves Complex**: Engineers discussed enabling an easy way to control which sets of tools **Claude** has access to, aiming to load only relevant tools based on the task at hand.
   - A member shared a workaround using sequential thinking prompts, while others suggested limiting the number of tools and using a multi-agent system to narrow down toolset choices.
- **McPoogle Search Engine Indexes MCP Servers**: The team launched **McPoogle**, a **Graphlit** RAG pipeline-powered search engine that indexes over 4000 **MCP servers** and tools, available at [mcpoogle.com](https://www.mcpoogle.com/?prompt=%22Tell%20me%20about%20the%20Graphlit%20MCP%20Server%22).
   - This enables searching and answering questions about **MCP servers** and tools, and users are encouraged to provide feedback.
- **MCP-CLI Bolsters Security with OAuth**: [MCP-CLI](https://github.com/wong2/mcp-cli) now supports **OAuth**, enhancing its accessibility and security, showcased in a [Loom video](https://www.loom.com/share/d2a00956cdb248e5adbc9c31538c7892).
   - This enhancement makes the tool more secure and accessible.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF API Endpoint Goes Poof!**: The **api-inference.huggingface.co** endpoint is deprecated, prompting users to switch to the new endpoint, though it's unclear if there was a deprecation notice.
   - **LangChainjs** is still using the old endpoint, potentially causing integration issues for those relying on it.
- **Object Tracking Models Get No Love!**: Members reported a lack of **Inference Providers** for **Object Tracking models**, including the removal of providers for the **DETR model** from Facebook.
   - This issue impacts nearly all models, limiting their practical application due to inference limitations.
- **Flux-Pro Unleashes Free Image Gen!**: **Flux-Pro-Unlimited**, a *partially uncensored* AI image generator, surfaces on Hugging Face Spaces for research, offering a *useful service with no ZeroGPU*.
   - The project can be found at [https://huggingface.co/spaces/NihalGazi/FLUX-Pro-Unlimited-](https://huggingface.co/spaces/NihalGazi/FLUX-Pro-Unlimited-).
- **HiDream LoRAs Slim Down with Quantization!**: Quantization support via `bistandbytes` implemented for training **HiDream LoRAs**, leading to substantial memory savings and streamlined training.
   - Enabling quantization slashes memory usage from **57.64 GB** to **34.60 GB** *after device placement*, detailed in the [Diffusers documentation](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_hidream.md#using-quantization).
- **Final Agent Frustrates Course-Takers!**: Students grapple with **GAIA question runs** consuming excessive time (an hour) and budget (>$15), alongside reports of **UI timeouts/errors** during agent operation.
   - One member spent *$5+* and got *5 questions* right, while another only got **20%** of the questions right after an hour, and one member shared their [blog post](https://guillaume-fradet.com/posts/making-ai-think-and-act-my-approach-to-the-hugging-face-ai-agents-challenge/) for the **final challenge**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **M4 vs RTX for LLM Devs**: Members debated between **M4 Macbook Pro** and **RTX 5070Ti Linux laptop** for local llm inference, citing smooth performance with *ollama Qwen 2.5-coder* on a **M4 Macbook Pro 48 GB**.
   - Suggestions included using a **Mac Studio/Mini** and renting GPUs cheaply when needed.
- **Diffusion Models Evolve!**: The paper [Diffusion Models as Evolutionary Algorithms](https://arxiv.org/abs/2410.02543) posits that **diffusion models** can be seen as **evolutionary algorithms**.
   - It clarifies that *denoising steps correspond to successive generations of model refinement*, bridging generative modeling strands.
- **Deepseek Dominates Post-Training?**: A member suggested **Deepseek** could outperform **OpenAI** due to superior **post-training**, referencing [Microsoft's MAI-DS-R1 on Hugging Face](https://huggingface.co/microsoft/MAI-DS-R1).
   - This highlights the importance of post-training techniques in achieving state-of-the-art model performance.
- **AI Agents Arise as Academic Authors**: A member is developing agents to write patents and academic articles and shared a link to [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2).
   - The member clarified that the agents he is developing are **dynamic** and *a society of minds*.
- **Em Dash Use Evolves Via AI!**: Usage of the **em dash** is being influenced by **AI models** like **chatGPT**, which learn from human data.
   - There's speculation whether humans will adopt the em dash more frequently as AI-generated content becomes more common.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Podcast Length Limited by Language?**: A user found their podcast capped at **14 minutes** in non-English languages against **40 minutes** in English, highlighting a potential language-specific limitation.
   - The user's discovery suggests that language processing may affect content generation or introduce bugs in the podcast creation tool.
- **Users share Mind Map Generation Technique**: A user details their process of generating mindmaps from sources using a custom prompt.
   - They regenerate using the prompt: *Create a mind map from the sources. List topics as central ideas, main branches, and sub-branches. Output it in Markdown format.*, then they feed the output to [markmap.js.org](https://markmap.js.org/) and save as *interactive HTML*.
- **NotebookLM Audio Gets Real (Podcast)**: A user reported successful audio overviews without common AI pitfalls, attributing it to the source material quality, and praises it for being like a real podcast.
   - The user specifically noted the absence of repeated content, static noise, phantom voices, or fabrication, implying that the quality is highly dependent on the prompt and source quality.
- **Cantonese Language for NotebookLM Incoming**: After a user inquired about **Cantonese** language support in NotebookLM, a team member confirmed that they are actively *working on it*.
   - No specific timeline was provided, but this indicates that NotebookLM is expanding its language capabilities.
- **Navigating Domain Restrictions in NotebookLM**: A user reported encountering an error message, *This web page cannot be imported due to domain restrictions* when trying to add a NotebookLM source to a project and posted a [screenshot](https://ibb.co/My2JzWHp).
   - Domain restrictions prevent importing certain webpages as NotebookLM sources, although the exact mechanism of this protection is unknown.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Dwarkesh Deep Dives into Agentic DAOs**: [Dwarkesh's essay and video](https://www.dwarkesh.com/p/ai-firm) explores the potential of **Agentic DAOs**, drawing inspiration from [Gwern's Backstop](https://gwern.net/backstop).
   - The discussion centered on the future of fully automated firms and their operational frameworks.
- **AI Startups grapple with High Costs**: A member shared [revenue numbers](https://x.com/tanayj/status/1919489023602786737) highlighting the significant expenses associated with **GPU time**, **engineer salaries**, and **executive compensation** in **HCOL cities**.
   - The thread sparked a conversation on strategies for cost optimization and sustainable business models in the AI sector.
- **Exa Revamps with BM25 Optimization**: **Exa** announced its return on X with a new blogpost detailing its approach to [BM25 Optimization](https://exa.ai/blog/bm25-optimization).
   - The update focuses on enhancing search relevance and efficiency.
- **OpenAI Snaps Up Windsurf in $3B Deal**: **OpenAI** is reportedly acquiring **Windsurf** for **$3 billion**, according to a [Bloomberg report](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion).
   - The acquisition is expected to bolster OpenAI's capabilities and market position, but no further details were given.
- **Gemini 2.5 Pro's ELO Skyrockets**: Claims of a significant ELO increase with the new **Gemini 2.5 Pro** update have surfaced, based on [this tweet](https://x.com/scaling01/status/1919771796334616759) and [Google DeepMind announcement](https://x.com/googledeepmind/status/1919770265711419826).
   - Early testers have reported substantial improvements in performance across various benchmarks.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **OpenAI Eyes Public Benefit Corp Status**: Amidst raising **$8B**, speculation rises that **OpenAI** anticipates a **1.3x+** return, prompting consideration of a Public Benefit Corp structure, detailed in a [WSJ article](https://www.wsj.com/tech/ai/openai-to-become-public-benefit-corporation-9e7896e0?st=ifeJpvreal).
   - This move suggests confidence in their current product spending and potential profitability.
- **Transpacific Flights Slashed in Half**: Flights to the US are approximately **half price** due to concerns about potential detentions and device searches at the border.
   - Travelers might benefit from these reduced rates if they're willing to navigate the geopolitical climate.
- **Nous Research Hosts RL Environments Hackathon**: Nous Research is organizing an [RL Environments Hackathon](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062ade) aimed at fostering innovation in reinforcement learning.
   - This event offers an opportunity for developers to collaborate and experiment with diverse RL environments.
- **Fine-Tuning Base Models for Non-Chat Tasks**: A user inquired about the possibility of fine-tuning a base model LLM for tasks beyond creating instruction-following chat assistants.
   - Responses suggest that while feasible—citing controlling robot movement and stock predictions as examples—*data acquisition presents a significant challenge*.
- **AnySphere's $900M Funding Request Raises Eyebrows**: A member questioned why **AnySphere**, the maker of Cursor, requires **$900M** with a relatively small team of around **100 people**.
   - Another member humorously speculated that they might actually need a team of **1000 people** to justify such a substantial investment.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Rare Pytorch Disk Swapping Baffles SysAdmins**: A user experiences extreme **page faults** and **disk swapping** when importing standard Python libraries like **torch**, **transformers**, and **vLLM**, despite using package management tools like **conda**, **venv**, and **uv**.
   - The issue persists despite being on the same disks as other users without similar problems, suggesting a possible broken table or **UID** assignment problem, and prompting suggestions for memory tests to rule out hardware faults.
- **Consider Anthropic's related work for parallelization**: A member inquired about switching to **model parallelism** from **data parallelism** when training a mid-range model with ample time and one suggested reviewing work from **Anthropic**.
   - They also suggested exploring papers citing **arXiv:2305.18153** for related research or advancements on the topic.
- **Community VLMs Research Hub Launches**: A community-driven hub for **multimodal researchers** has been created and maintained at [All-Things-Multimodal](https://github.com/thubZ09/All-Things-Multimodal.git), with weekly updates.
   - The creator welcomes contributions, suggestions, and feedback to ensure the hub remains comprehensive and current for community use.
- **Dive Deep into Circuit Identification**: To understand model behavior, a member suggested investigating **circuit identification**, referencing papers on **grokking**, **BIG-Bench**, **content bias reasoning**, **Anthropic transformer circuits**, and **Towards Monosemanticity**.
   - Ambitious approaches often have limitations in accurately representing the model's internal processes.
- **Tackling Tooling Gaps in Interpretability Research**: Members discussed **tooling challenges** in interpretability, with one mentioning the decision to save activations to disk and train the **SAE** in a separate stage, as well as the extensive **LLM calls** required for auto-interpretability.
   - Another member noted that some interp libraries like **transformerlens** and **nnsight** still have limited functionality for some models.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Codegen Considered for Boilerplate Reduction**: Members discussed using **codegen** to reduce boilerplate when adding new models to **Torchtune**, drawing inspiration from faster model support in libraries like **Unsloth**.
   - Concerns were raised about relying on tracing keywords and the importance of users understanding the underlying code, suggesting well-written tutorials as an alternative to extensive codegen.
- **Engineering Time on New SoTA Models Top Priority**: The primary goal is to **reduce the engineering time** required to use new **SoTA models** with Torchtune while maintaining reasonable performance.
   - A suggestion was made to tackle the challenge by identifying boilerplate versus difficult aspects, focusing on simplifying the latter, like tokenizer support, before automating boilerplate generation.
- **Tokenizer Support Deemed Nontrivial**: **Tokenizer support** was identified as a nontrivial task, with a call for a generalized solution, while more tedious tasks could be automated through scripts before considering codegen.
   - The discussion mentioned leveraging **HF (Hugging Face) configurations** for tokenizers and generating parity check numbers using HF models.
- **HF Transformers Adapter Envisioned for Faster Model Support**: One approach suggested was to create a generic **HF adapter** that loads HF models and allows mappings to work with different training features.
   - This would facilitate faster addition of new models and enable finetuning, with the option to later implement a "real" implementation with full feature support.
- **Qwen3 Hype:  Codegen Could Support Model Version with Ease**: It was mentioned that for some model families like **Qwen**, new versions often require mostly new boilerplate, which could be largely added with codegen.
   - The discussion highlighted the marketing advantage of rapidly supporting new models like **Qwen3** upon release, even with initial restrictions, before a full-featured implementation is ready.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Puzzles Teases Macbook Users**: A user inquired about running **Modular Puzzles** on a Macbook, but current support for **Apple Silicon GPUs** is unavailable.
   - A workaround involves remote work on a **GPU-attached cloud instance**, sidestepping local compatibility issues.
- **Mojo's NVIDIA GPU Support Charted**: For **Mojo GPU** programming, the supported **NVIDIA GPU architectures** include Turing, Ampere, Hopper, and Blackwell (**RTX 20XX - 50XX series**).
   - This clarification ensures developers target compatible hardware for optimal **Mojo** development.
- **Blot.im Recommended for Markdown Blogging**: A member suggested [Blot.im](https://blot.im/) as a blogging platform that supports **markdown**, though it is a *paid* service.
   - This platform caters to users who prefer markdown-based content creation.
- **Comptime Try-Except Conundrums**: A member inquired about the feasibility of using `try ... except ...` inside computations performed at `comptime`.
   - This question addresses advanced error handling during compile-time computations in **Mojo**.
- **User Flags Errors in Mojo Getting Started Guide**: A user reported errors with the last example in the [Mojo Getting Started Guide](https://docs.modular.com/mojo/manual/get-started/).
   - This feedback highlights the need for ongoing refinement of official documentation.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Hackathon Focuses on Agent Communication**: LlamaIndex is sponsoring the **Big MCP Hackathon** in Tel Aviv, encouraging the development of **MCP-powered apps** for agent-to-agent communication and experimentation, as detailed in [this tweet](https://twitter.com/llama_index/status/1919499875332587579).
   - The hackathon, organized by @aitinkerers, aims to foster innovation in **multi-agent systems**.
- **LlamaIndex Teaches Deep Research with Multi-Agent Systems**: LlamaIndex introduced a workshop tutorial on building a **multi-agent system** for **deep research** from scratch, leveraging **AgentWorkflow**, as seen in [this tweet](https://twitter.com/llama_index/status/1919856812893077565).
   - The tutorial guides users through creating agents that can perform complex research tasks.
- **Property Graph Indexes Spark Implementation Questions**: A member explored property graph indexes, sharing their implementation using LlamaIndex's documentation in [this notebook](https://github.com/tituslhy/shiny-engine/blob/main/notebooks/hybrid_property_graph.ipynb).
   - The member noted intermittent failures in answering questions due to retrieval issues from the graph or vector database, probing questions about **vector storage strategies**.
- **LlamaIndex Graphs are Denser Than LangChain Graphs**: A member observed that graphs generated using LlamaIndex's property graph index code are much denser, as visualized [here](https://github.com/tituslhy/shiny-engine/blob/main/images/llamaindex_neo4j.png) compared to LangChain as visualized [here](https://github.com/tituslhy/shiny-engine/blob/main/images/groq_kg.png).
   - This was based on observations when testing vs the reference [LangChain notebook](https://github.com/tituslhy/shiny-engine/blob/main/notebooks/large_llm_knowledge_graph.ipynb).
- **Vector DB Stores Each Node in GraphRAG**: A member suggested that, in GraphRAG, the **vector database** stores each node in the graph, unlike typical RAG where it stores each text chunk.
   - It was also mentioned the **graph density** is likely due to the default prompts being used.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **M4 Macbook Pro Challenges RTX 5070Ti**: A member asked whether a **Macbook Pro M4** or an **RTX 5070Ti Linux laptop** would be better for local LLM inference and general development.
   - They are a long-time **Linux desktop user** but are considering the **M series Macs**.
- **Discord Read-the-Rules Reminder**: George Hotz reminded a user to read the rules of the Discord, stating that it is a place for discussion of **tinygrad** and **tinygrad development**.
   - This reminder occurred after a user asked about getting a new machine, choosing between a **M4 Macbook Pro** and a **RTX 5070Ti Linux laptop**.
- **Bounty Hunters Seek Guidance**: Members seek guidance on how to pick bounties from the [Google Sheets bounty list](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0) and whether longer descriptions exist elsewhere.
   - A member suggested that creating a **WIP PR** is required to lock a bounty to prevent multiple people from working on the same thing.
- **Rockship Device SDK vs Open-Source Driver**: Regarding a bounty for a **new Rockship device**, members are asking whether they are supposed to use the SDK or the open-source driver.
   - There was no further information on what a decision may entail.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Auth0 Sponsors Entrepreneurship Track**: **Auth0** is hosting a workshop tomorrow (5/7) at **10:00 AM PT** covering securing **AI agents** with authentication, accessible via [YouTube livestream](https://www.youtube.com/watch?v=wB29IJ3AEjY).
   - **Auth0** is also offering Entrepreneurship Track prizes, including **$5,000** for 1st place, for teams integrating **Auth0.ai** into their projects.
- **Members Await HuggingFace Credits and Quiz Scores**: A member reported that **HuggingFace credits**, despite form submission, are yet to be allocated to their organization.
   - Separately, scores for **Quiz 11** and **Quiz 12** are still pending release.
- **LLMs Grapple With Conditionals**: A member questioned how **LLMs** remember **conditional statements** and produce good results, especially with examples like voting eligibility criteria (*18 years old, citizen, voter ID*).
   - They also inquired about the role of **formal methods** in helping **LLMs** handle conditional logic, seeking ways to represent complex conditions for long-term memory.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere-AI npm Package Supports Aya Vision**: A member inquired whether the **Cohere-AI npm package** supports **Aya Vision**, and another member confirmed it, providing a code snippet.
   - The snippet demonstrated using the **CohereClientV2** class to interact with the **c4ai-aya-vision-8b** model, sending a message containing both text and an image URL.
- **User Integrates Cohere-AI for Expedition**: Following the code snippet and confirmation, a member reported successful implementation and integration of **Cohere-AI** for an unspecified *expedition*.
   - Specifics about the *expedition* and the integration's nature were not disclosed.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Claude's System Prompt Surfaces for Integration**: A member shared [a Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1kfkg29/claude_full_system_prompt_with_all_tools_is_now/) containing the full system prompt for **Claude** with all tools, sparking interest in potential integrations.
   - The prompt details **Claude's** inner workings, potentially aiding in replicating its functionality in other models.
- **Pythonistas Seek Chat Template Advice**: A member requested assistance with generating a **chat template** using **Python**, encountering syntax errors despite having the *transformers* module installed.
   - The user sought guidance on resolving the issues, indicating a need for practical advice on **chat template** generation.
- **Leaked Claude Prompt for GPT4All?**: A community member inquired about integrating the leaked **Claude** system prompt into **GPT4All**.
   - This highlights the community's interest in leveraging insights from different **LLM** systems to enhance the functionality of **GPT4All**.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Grid Dynamics Teaches AI Project Taming**: **Grid Dynamics** DevOps/MLOps Architect and **Data Phoenix** founder, **Dmytro Spodarets**, will host a webinar on Wednesday, May 28 at 10:00 AM PDT about establishing reliable AI project processes and accelerating delivery, saving your spot via [this link](https://lu.ma/qhytft9t).
   - The webinar will cover the lifecycle of **ML** and **LLM** projects, tools, roles, and practices that form the foundation of **MLOps/LLMOps**, and the **MLOps/LLMOps maturity model**.
- **Experiment Setup is Crucial for Validation**: A discussion in the general-ml channel argued that a full **experiment setup** (training and testing) is mandatory, or if simply **applying a pre-existing model** suffices, especially for **model validation**.
   - Suggestions were made that included **cross-validation techniques**, **robustness checks against adversarial examples**, and **validation on real-world data** to ensure broader applicability.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Kicks off Wave 8 with Teams and Enterprise Features**: Windsurf initiated **Wave 8**, introducing new organizational tools and features over several days, covered in their [blog post](http://windsurf.com/blog/windsurf-wave-8-teams-and-enterprise) and [changelog](https://windsurf.com/changelog?cachebust=202405061200).
   - Additional information can be found on [X](https://x.com/windsurf_ai/status/1919820747037392982), [Bluesky](https://bsky.app/profile/windsurfai.bsky.social/post/3lojj34nq622q), [YouTube](https://youtu.be/t7GQFFopQFY), and [Reddit](https://reddit.com/r/windsurf).
- **Windsurf Reviews Autopilot PRs in Beta**: Windsurf launched **Windsurf Reviews (Beta)**, a GitHub app designed to automatically review PRs and edit titles/descriptions based on specified guidelines, aiming to streamline code review.
   - This feature assists organizations in maintaining code quality and consistency by automating the initial review process.
- **Windsurf's Knowledge Base Now Grounds with Google Docs in Beta**: The new **Knowledge Base (Beta)** feature allows users to connect Google Docs to their Windsurf context.
   - This enhancement enables Windsurf to leverage information from Google Docs for more accurate and informed responses, improving grounding.
- **Cascade Sessions get Conversation Sharing (Beta)**: Windsurf introduced **Conversation Sharing (Beta)**, which allows users to share successful Cascade sessions with teammates, facilitating knowledge transfer.
   - This feature enhances team collaboration by enabling the sharing of effective conversation flows and insights.
- **Teams Deploy Directly to Netlify**: Windsurf now supports **Teams Deploys**, enabling direct deployment to an organization's Netlify account.
   - This integration simplifies the deployment process for teams using Netlify, offering a more streamlined and efficient workflow.



---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1369025876951302305)** (1179 messages🔥🔥🔥): 

> `Claude Code, Gemini 2.5 Pro, Grok 3.5 release, o3 vs Gemini` 


- **Claude Code is a Killer App**: Members noted that **Claude Code** is very good and better than **Cursor**, with one person saying, *claude code >>>> cursor* and stating *cursor is a scam compared to claude code*.
   - They also noted that using the word *ultrathink* gives a larger thinking budget (32k). There is a [link to the documentation](https://docs.anthropic.com/en/docs/claude-code/overview) describing Claude Code.
- **Gemini 2.5 Pro I/O update**: Members discussed the [new Gemini 2.5 Pro update](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance/) which is better at coding and multi turn.
   - One member found it's better at *philosophy + explanation + coding*, and noted that the new model seems to *grasp nuances better* and *seems to comprehend better*.
- **Grok 3.5 Hype**: The imminent release of **Grok 3.5** has the community hyped for potential SOTA performance, with some boldly proclaiming it's *ASI*.
   - However, skepticism remains regarding inflated benchmarks and the actual impact of advanced reasoning.
- **O3 is still the King**: Members talked about how **O3** is still better and also that Grok 3 mini has better bench than grok 3.
   - One member suggested that this version of Gemini is also dumber like instead of getting a model to generate movies, its better to have one that can effectively use our existing software to make movies, which means better at function calling etc.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1369026368528056330)** (536 messages🔥🔥🔥): 

> `Cursor Pro slow requests, Connection failed error, Cursor 4.1 groove, ASI Age internal logics, Open Router accounts` 


- **Cursor Pro's "Unlimited" Slow Requests Draw Skepticism**: Users discuss the "unlimited slow requests" feature of Cursor Pro, questioning if they become unusable due to slowing down with usage, while others claim they are still great with [occasional delays](https://docs.cursor.com/account/plans-and-usage#fast-and-slow-requests) depending on context and code size.
   - One user joked about Cursor's potential profit strategy with slow requests, while another shared concerns about the advertised feature being *literally* punished with exponential backoff.
- **Cursor 4.1 Catches the Groove, ASI Looms**: Users express enthusiasm for Cursor 4.1 once they *get in a groove*, while discussions pivot to the future, including the importance of rewriting internal logics to operate resiliently in the **ASI Age**.
   - One user suggests current schooling and job structures are unsustainable, implying an impending shift due to **ASI**.
- **GPT-4.1 creates Collaborative Coding Experience and Model Performance Compared**: Users raved about the collaborative coding experience using **GPT-4.1**, while opinions varied on the performance of models with some preferring **Claude 3.7** for its coding capabilities.
   - A user indicated that **GPT-4.1** can not find a correct usage of a simple piece of code in the codebase, but **Sonnet 3.7** does it perfectly.
- **Gemini 2.5 New Updates Hype Check, Cursor's Value Reign Supreme**: The new **Gemini 2.5 Pro** update faced mixed reviews, some reported the the newest versions had increased verbosity and reduced coding effectiveness, while others praised the speed and tool usage, especially in larger edits. The consensus held that **Cursor** stands out with *slow unlimited* requests.
   - One user says *new gemini 2.5pro is crazy, hyper pro-active and much smarter*, but other states that *new 2.5 pro is garbage at tool calling*.
- **Concerns Emerge Over Third-Party MySQL MCP Server Libraries**: A user voiced apprehension regarding the trustworthiness of community-contributed **MySQL MCP server libraries**, pointing out potential security risks and the lack of official endorsement.
   - In response, it was advised to review the open-source code on GitHub before deploying, encouraging local builds, and suggesting users create their own **MCP**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1369031242187669595)** (167 messages🔥🔥): 

> `Gemma-3 Finetuning, Qwen3 Fine-tuning Issues, GLM notebook testing, Saving models as GGUF, Vision data format for Gemma3` 


- **Gemma-3 LoRA Fine-Tuning Configuration Tweaks**: A user shared their **Gemma-3 LoRA** fine-tuning configuration, enabling `gate_proj` at rank 256, and seeking advice on fine-tuning the embedding and lm_head.
   - The configuration included settings for `finetune_vision_layers`, `finetune_language_layers`, `finetune_attention_modules`, and `finetune_mlp_modules`, with specific recommendations for `r`, `lora_alpha`, and `lora_dropout`.
- **Qwen3 Fine-Tuning Troubles and CPT Notebook Reference**: A user faced issues with **Qwen3** fine-tuning and was directed to a [CPT notebook tutorial](https://x.com/engineerrprompt/status/1919510087506526235) for guidance.
   - The discussion also involved confusion regarding a "new style" CPT notebook, clarified as a Gemma-3 fine-tuning example.
- **Root password request by Unsloth raises eyebrows**: Users reported that **Unsloth** was asking for the root password and questioned what package it was trying to upgrade.
   - It was revealed to be related to updating *llama.cpp* with `sudo` for installing `build-essential cmake curl libcurl4-openssl-dev`, considered mostly harmless but raised UX and security concerns.
- **Tensorboard is the tuning visual wizard**: A user inquired about generating training figures, and another member suggested **enabling tensorboard logging** with a flag in the config.
   - The specific flag mentioned was `report_to="tensorboard"`.
- **Claude Sonnet Data Shows Intriguing Tag Conversion**: A member shared results from fine-tuning **Qwen3-4b** with Claude Sonnet data, observing that the fine-tuned model converted learned concepts into **Anthropic's XML tag format**.
   - For example, `<think>` was converted into `<antThinking>`, suggesting the model adopted Anthropic's coding patterns; the script to grab the data was provided via [gist](https://gist.github.com/fullstackwebdev/9fa43ac1af41e48f774f551ab216d0a5).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1369031310982512761)** (5 messages): 

> `Gemma 3 12b, Qwen3 14b, Hallucinations in models, Tool calling fixes` 


- **Gemma 3 or Qwen3 - Which Reigns Supreme?**: A member asked for the community's opinion on whether **Gemma 3 12b** or **Qwen3 14b** is better.
   - Another member noted that *Gemma 3 is really good at knowledge, but not reasoning, and Qwen (2.5 and 3) is really good at reasoning, but not knowledge*, while also noting that **Gemma 3 hallucinates a lot**.
- **Tool Calling GitHub PR Update**: A member noted that someone messaged another member on their [GitHub PR for tool calling](https://github.com/unslothai/notebooks/pull/12).
   - The recipient of the message replied that they would do it this weekend and submit the **Qwen3 fix** around Monday.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1369030389594718210)** (239 messages🔥🔥): 

> `Qwen3 model differences, LMStudio and Gemma3 Issues, SafetensorError on Windows, Granite model finetuning, Sample weights during training` 


- **Differences between Unsloth and Qwen3**: A member inquired about the differences between the [Unsloth](https://huggingface.co/unsloth/Qwen3-30B-A3B) and [Qwen](https://huggingface.co/Qwen/Qwen3-30B-A3B) versions of **Qwen3-30B-A3B**, specifically regarding file shard size and potential modifications.
   - After experiencing errors with a modified **Qwen3-30B-A3B** model, a member discovered an issue related to gradient requirements in the MoE layer structure, which was resolved by switching to the **Qwen/Qwen3-30B-A3B** model.
- **Lmstudio Gemma3 issues**: Members discussed issues with loading **Unsloth's Gemma3 GGUF** files in **LMStudio** on both Windows and Mac, with errors such as *"Exit code: 6"* and failures to load the model.
   - A solution was found by configuring the *"context length"* in **LMStudio** to **1024**, which resolved the loading issues for the **Unsloth** release of **Gemma3**.
- **Windows safetensors saving error**: Users encountered a `SafetensorError` on Windows, specifically an `IoError` with the message *"The requested operation cannot be performed on a file with a user-mapped section open."` when saving merged models or LoRA adapters.
   - Troubleshooting steps included updating `safetensors`, using `safe_serialization=False`, and closing VS Code to resolve potential file locking issues, with a recommendation to use **Process Explorer** to identify processes locking the files; the issue appears to be Windows-specific.
- **Granite upcasted performs better?**: During the finetuning of the Granite model, a user observed a performance discrepancy when loading the model with different precisions; loading as **float32** and then applying LoRA resulted in **57%** accuracy, compared to **41%** when loading directly in **bfloat16**.
   - The user initially used the peft library directly instead of unsloth's methods save_pretrained_merged(), so they were giving bad advice all along.
- **Sample weights injection**: A member sought guidance on injecting sample weights during training with **Unsloth** and **LoRA**, aiming to address an imbalanced dataset by assigning weights to each conversation.
   - They proposed attaching `{"input_ids": ..., "attention_mask": ..., "labels": ..., "sample_weight": ...}` to the data and using a customized loss function to incorporate these weights during training.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1369032886984118323)** (1 messages): 

> `GroqStreamChain, Real-time AI chat apps, WebSockets, LangChain integration` 


- **GroqStreamChain Arrives!**: A user introduced **GroqStreamChain**, a real-time AI chat application built with **FastAPI**, **WebSocket**, **LangChain** and **Groq**.
   - It supports seamlessly streaming AI responses and interacting with smarter chatbots powered by cutting-edge technology, see the [project on GitHub](https://github.com/pr0mila/GroqStreamChain).
- **GroqStreamChain Uses WebSockets!**: A user has showcased how **GroqStreamChain** is built using **WebSockets** and **FastAPI**.
   - This project will allow for streaming AI responses in real time, facilitating smoother interactions, which is all visible on the [project's Github page](https://github.com/pr0mila/GroqStreamChain).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1369289932861997146)** (59 messages🔥🔥): 

> `Transformer + BERT Model Combination, Pretraining Gemma3 with Medical Data, Multi-label multi-class classification, Muon paper by Google, Integrating Muon with Hugging Face` 


- **Transformer and BERT: A Human Centipede of Models?**: A member inquired about combining a small transformer by **stripping its final layer** and connecting it to a small **BERT**-like model for classification.
   - Another member suggested the opposite: to strip off the final layer of a *trained transformer* to add a classification layer and finetune.
- **Medical Gemma3 Pretraining for Multiple Models**: A user is pretraining **Gemma3** with medical data for various tasks, including a **multi-label multi-class classification model**.
   - They are seeking novel approaches, considering combining their pretrained transformer with a **BioBERT**-like model.
- **Google's Muon Transformer Paper Implementation on Gemma3**: Members discussed the Google's [Muon paper](https://arxiv.org/abs/2505.02222) and implementing it with **Gemma3** for faster training.
   - The user stated that the implementation with Unsloth was problematic, and was advised to integrate it into **Hugging Face libraries** instead and later reported it was working.
- **"Bolting" Layers Onto Mistral Results in Surprising Success**: A member claimed to have *bolted layers onto Mistral with zero retraining* and still received intelligible replies, sharing [the code here](https://github.com/jagoff2/gma6).
   - The model's *deterministic output* differs with the layers after *hooking specific layers*, and it generates valid text while influencing generation, although results do not evaluate correctly.
- **LLMs are efficient for Text Classification**: A member suggested that a more efficient strategy than using BERT would be to train the LLM to output text corresponding to a class.
   - Another member countered that for a **75000 class, multi-label, multi-class problem**, a classification layer would likely outperform the LLM approach due to efficiency concerns.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1369026201980502056)** (451 messages🔥🔥🔥): 

> `O3 lazy, AI competition recipes, O3 pro 2x, Discord bot down, Perplexity image quality loss` 


- **Users Strategize to **Outsmart** AI**: Members discussed feeding **O3 prompts** that indicate *existence is at stake* to get the best response, or telling it to compete against a **Harvard 4.0 GPA student**.
   - Other prompts involve asking it to compete against a **Michelin star chef** or even *singularity* to push its limits.
- **Doubling Down with Duplicate O3s**: One user bought a **second Pro subscription** to use **two O3 models** side-by-side for isolated comparison.
   - Another member hinted at a future **O3 Pro 2x** feature coming soon that would negate the need for 2 separate accounts.
- **Perplexity bot put down, lives on X**: A member stated that Perplexity announced they are **discontinuing the Discord bot** but it will live on **X** and other platforms.
   - This was later confirmed by another member linking back to the original announcement [here](https://discord.com/channels/1047197230748151888/1155998520822743091/1368744746142404669).
- **Perplexity image dimensions vary between Whatsapp and Discord**: A member noticed that images generated through Perplexity on **WhatsApp** are significantly smaller in size (KB) compared to those generated on **Discord** (MB), raising concerns about **quality loss**.
   - After some discussion a user confirmed that images *downloadable file size in perplexity whatsapp images is 200 KB but in discord the downloadable perplexity image is 1.5 MB*.
- **Perplexity should buy Cursor, users suggest**: Some members propose that Perplexity should acquire **Cursor**, a popular AI-powered code editor, envisioning a boost in its capabilities by leveraging **Amazon/AWS's pricing advantage**.
   - However, one user noted that *that would be the death of Cursor though*.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1369211042671886357)** (3 messages): 

> `workaround with Beautiful Soup, URL citations` 


- **Beautiful Soup Workaround Floated**: A member suggested a workaround involving requesting + **Beautiful Souping the URL citations**, and then checking which of the bullet points are mentioned most-frequently by that web page to manually correlate them.
   - They admitted that this method isn't super scalable/reliable and suggested sending an email to **api@perplexity.ai**.
- **Email Contact Suggested**: A member suggested sending an email to **api@perplexity.ai** regarding issues.
   - The member stated the original workaround may not be scalable or reliable.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1369026425587503255)** (275 messages🔥🔥): 

> `Mistral 3.1 24b Image Recognition, LM Studio model updates, Self-hosted LLM vs API costing, LLM Training and User Data, Speculative Decoding with Qwen 3` 


- **Mistral 3.1's Image Recognition Falls Short**: Early tests of **Mistral 3.1 24b**'s image recognition capabilities reveal that it performs worse than **Gemma 3 27b**, despite claims of superiority in the release notes.
   - One user experienced it hallucinating characters, mistranslating comics and misidentifying images as *Shrek and Fiona*, even with recommended parameters and q3-q4 quantization.
- **Decoding Difficulties with Qwen Models**: Users reported that speculative decoding with the **Qwen 0.6B** model slowed down models with larger parameter sizes, from **1.7B** to **30B**.
   - Others mentioned that using the correct template for **Qwen 0.6B** is essential for it to function as a speculative decoder.
- **LM Studio's Model Discovery Troubleshoot**: A user faced challenges in getting **LM Studio** to recognize models in a specified directory, even after setting the 'Models Directory' correctly.
   - Solutions attempted included symlinking, importing models, and verifying file integrity, eventually resolving with the successful import of the **gemma-27B** model.
- **RAG vs Dynamic knowledge injection**: A user described a system for knowledge injection into the LLM, using a database of *neurons* that influence the model according to the context.
   - They explain that it is different than RAG because it is dynamic and converts each idea into the most basic form, and then allows the model to combine them together.
- **Gemini's New Update has Mixed Reviews**: One user thinks the new **Gemini** update is ignoring and overengineering prompts.
   - Others seek to have **catppuccin theme** for **LM Studio**.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1369097332154564638)** (87 messages🔥🔥): 

> `Q8 XL Model Performance, 4080 vs 4060 GPU Setup, Memory Bandwidth Bottleneck, Random Token Generation Speeds, Apple M3 Memory Configurations` 


- **Q8 XL Model Runs Slow on Mac Studio**: The **Q8 XL model** runs slowly on a Mac Studio because it's not offloading everything to the GPU, achieving only **15-20 tokens/sec** compared to the non-XL version.
   - The user suspects the issue may be due to the 60-core GPU model and is considering adjusting settings for better performance, hoping for faster speeds.
- **4080 & 4060 Dual GPU Setup Discussed**: A user prefers using the **4080** as the primary display card to avoid VRAM overflow, even though the **4060** limits the overall speed in a dual GPU setup.
   - Another user reported getting **12-15 tkps** on a **27B Q4 8k ctx**, noting that without the 4060, they were only getting **2-4 tkps**, highlighting the significant impact of the setup, while pondering the forthcoming **gemma 3 27b QAT** models.
- **Memory Bandwidth Limits Token Generation**: A user suggests that memory bandwidth bottlenecks LLM execution, proposing a calculation where the upper limit of tokens per second (Q4) is estimated by dividing bandwidth by the number of billions of parameters: [IBM Community Post](https://community.ibm.com/community/user/ibmz-and-linuxone/blogs/tina-tarquinio1/2025/05/06/the-next-generation-of-ibm-linuxone?communityKey=9476eac0-4605-4c10-906d-01a95923ae0b).
   - Another user confirms that this approximation matches their observations, with a variance of *+-10 tps* when considering flash attention.
- **Token Generation Speeds Vary Randomly**: A user experiences significantly different generation speeds for the same prompt, with speeds fluctuating between **55 tok** and **150+ tok** randomly across multiple attempts.
   - It's speculated whether this issue is a bug or related to thermal throttling, with the user noting that the fast speeds involve text popping out *insanely fast*.
- **Apple M3 Ultra Memory Configs Questioned**: The discussion clarifies that the M3 Ultra consists of two Max chips, each with a 48GB configuration, explaining the 96GB total, and usable VRAM defaults to 75% for systems with greater than 32GB.
   - A user lamented that the **256GB option** is disproportionately expensive and another mentions that LM Studio reports **76GB of VRAM** but isn't fully utilizing the GPU.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1369029111439167536)** (198 messages🔥🔥): 

> `Google AI Studio vs Gemini, Gemini 2.5 Pro Coding Prowess, GPT's Flattery Glaze, Lucid Dreaming Techniques, Grok 3.5 expectations` 


- **Gemini 2.5 Pro Shines in AI Studio**: Members discuss the difference between **Google AI Studio** and **Gemini Advanced**, noting that **AI Studio** offers free access to **Gemini 2.5** with a **1 million token context window** and customization options, but lacks the UI features of **Advanced**.
   - Despite the **25 request daily limit** on the API version, some users report exceeding it, leading to the conclusion that the actual limit for **AI Studio** is effectively unlimited.
- **Gemini 2.5 Pro Praised and Panned for Coding**: While some praise **Gemini 2.5 Pro** in **Google AI Studio** as exceptional for coding, even better than **o4-mini-high**, others find it disappointing, and akin to **GPT-3.0** or claim it generates code in Vietnamese.
   - In contrast, one user said that the new **Gemini 2.5 Pro (I/O edition)** is supposed to be *even better at coding* [according to this tweet](https://x.com/OfficialLoganK/status/1919770687167684808).
- **GPT's 'Flattery Glaze' Annoying Users**: Users discuss **ChatGPT's** tendency to be overly flattering, with one describing it as *ultimate glaze mode, kinda annoying* and recommending turning off the "personality" setting.
   - One linked a [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5242329) to explain the behavior where the agent is acting as a *"stochastic parrot"* echoing back whatever it is trained on.
- **Exploring Lucid Dreaming Techniques**: Members share tips for **lucid dreaming**, including dream journaling, maintaining a consistent sleep schedule (**5-7 hours**), and waking up and going back to bed.
   - One user suggests that with practice, dreams can reach **90% realism**, encompassing all senses, and recommends some channels for those looking to explore the possibilities of lucid dreaming.
- **Qwen 3 is not close to Claude 3**: Members debate the capabilities of the **Qwen 3 4B** model, with some claiming it matches or exceeds the performance of **Claude 3 Opus** and **GPT-4T** on certain tasks.
   - However, others strongly disagree, citing its tendency to hallucinate on real-world tasks like downloading files with aria2c, leading to the conclusion that it's *not even close* in real-world tests.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1369070596033482752)** (24 messages🔥): 

> `GPT-4o issues, 4o Browser performance, Dragging chats into folders` 


- **Users Complain about Nonsensical GPT-4o Responses**: Users reported that **GPT-4o** is giving complete nonsense web search replies that have literally 0 to do with the prompt or context.
   - One member suggested that the user might have overloaded the context window and sent it into a loop and mentioned that the token limit seems really low lately.
- **Browser Freezes During GPT-4o Webtool Usage**: A user reported that whenever **GPT-4o** uses the webtool, their browser locks up and CPU load hovers around 30%, with the responses taking 1+ minute to show.
   - The user has already tried **Chrome, Firefox, and MS Edge**, and the issue persists; the only workaround is to avoid calling it a toaster.
- **Dragging Chats Into Folders No Longer Possible**: A user reported that they **can no longer drag chats into folders**.
   - No resolution or reason was provided in the messages.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1369072771212251136)** (54 messages🔥): 

> `Prompt engineering definition, ChatGPT cost, Truth in AI, Atomic theory in AI, Custom GPT design` 


- **Prompt Engineers are AI Sculptors**: Prompt engineering is defined as *sculpting AI output through an understanding of the context, training, and behaviors of large language models*.
   - The output can be improved by integrating easter philosophy to support perspective thinking and enhance emotiveness.
- **ChatGPT Free Tier vs Subscription Cost Debated**: One member stated that it's possible to *learn from the AI for free and master the principles* but to maximize your skills it's worth choosing a good subscription.
   - Another member suggests pasting a [Discord link](https://discord.com/channels/974519864045756446/1046317269069864970/1368907603572428871) into ChatGPT to learn the basics right away.
- **Insisting on AI Truthfulness**: A member asked if they could make ChatGPT *speak truth only*.
   - Another member jokingly replied to just tell ChatGPT, *ChatGpt, set (Truth) = True*.
- **Atom Discussion Provokes Strong Reactions**: One member requested *No electron No spacetime No proton No neutron - Only observable/measurable proof*, as they wanted to interact with the bot with basic physics.
   - Another member sarcastically posted a [Fox News image of an atom](https://cdn.discordapp.com/attachments/1046317269069864970/1369205977722650634/image.png?ex=681bacaf&is=681a5b2f&hm=e2e974f8ce787836142b3537c9400e67117d52612e28a4a8122f045a83c38b44), which led to deeper discussions about labels and worship.
- **Custom GPT Design for Atomic Theory**: A member is *making my own atomic theory* and wants to use prompt engineering, to design the atoms core.
   - Another member created their own custom GPT, challenging the member to design a model that would be able to prove their theory hadn't already been disproven.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1369072771212251136)** (54 messages🔥): 

> `Prompt Engineering, Truth in AI, Atomic Theory, Customizing ChatGPT, Eastern Philosophy in Chatbots` 


- **Prompt Engineering: Sculpting AI Output**: Prompt engineering involves *sculpting AI output through an understanding of the context, training, and behaviors of large language models*, as explained by a member.
- **The Quest for Truth in AI**: A user asked if **ChatGPT** could be made to *speak truth only*, leading to a humorous reply to just type to your ChatGpt: "ChatGpt, set (Truth) = True".
- **Atomic Theory Reimagined**: A member jokingly posted an image of an atom from Fox News, while another member declared *There is no atom. its just label*.
   - The discussion then shifted to labels and worship, with references to **Ginsberg** and the need to worship something, prompting a reminder about the rules against political and religious discussions.
- **Eastern Philosophy Infusion in Chatbots**: A member mentioned that some **devs in China** are exploring *locking eastern philosophy* into their chat models to support perspective thinking and enhance emotiveness.
   - It was noted that **OpenAI** thinks that the model remembering you like peanut butter and chocolate will somehow help with context, too.
- **Customizing Chat GPT for Continuous Chat**: A member expressed interest in customizing **ChatGPT** for continuous chat and a specific atomic theory project, highlighting struggles with maintaining context.
   - Another offered to make a CustomGPT to convince a model the user's atomic theory hasn't already been solved by relativity and quantum mechanics, and send a screenshot.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1369335422769168525)** (2 messages): 

> `Gemini 2.5 Pro, Activity Page Enhanced, Reasoning Model Perf Metrics, Request Builder API, Prompt Category API` 


- **Gemini 2.5 Pro Preview Rolls Out!**: Google's **Gemini 2.5 Pro Preview** is now live, accessible via the same model slug, and the endpoints have been updated to point to the new date on **Vertex** and **AI Studio** as [announced on X](https://x.com/OfficialLoganK/status/1919770687167684808).
   - Try out the model now via [OpenRouter](https://openrouter.ai/google/gemini-2.5-pro-preview).
- **New Activity Page Boosts Model Usage Analysis!**: The platform now features an **enhanced Activity Page** with multiple new charts for a deeper dive into model usage.
   - Users can also check their personalized model rankings by clicking on a chart.
- **Reasoning Model Performance gets Measured!**: Latency for reasoning models now measures the time until the first reasoning token, while throughput metrics now include both reasoning and output tokens.
   - This provides a more complete picture of reasoning model performance.
- **Request Builder API Simplifies Request Generation!**: A new **Request Builder** is available, which helps users easily generate request body JSON and understand requests better, as shown in the [request-builder](https://openrouter.ai/request-builder).
   - This tool is designed to streamline the development process.
- **Prompt Category API Optimizes Model Selection!**: The platform introduces a **Prompt Category API**, allowing users to request models optimized for specific prompt categories directly, such as [programming models](https://openrouter.ai/api/v1/models?category=programming).
   - All available categories can be explored via the sidebar at [OpenRouter Models](https://openrouter.ai/models).


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1369059445212188742)** (10 messages🔥): 

> `Openrouter-powered Discord Bot, LMarena database, SimpleAIChat LLM chat client` 


- **Discord Bot template leverages OpenRouter**: A member released an [Openrouter endpoint powered discord bot template](https://github.com/cioran0/DiscordAI) using **discord.py**, handling discord char limitations.
   - The bot uses **wikipedia retrieval** at *vnc-lm/src/managers/search/service.ts and vectorstore.ts* instead of a vector DB.
- **Leaderboard data obtained from LMarena database**: A member stated that they get the leaderboard data from the popular **LMarena database**, then they sort those numbers and make a visual leaderboard from it.
   - They clarified that *Openrouter name matching with LMarena can get difficult, but luckily almost all of them are available in OR*.
- **SimpleAIChat: Local-First LLM Chat Client Debuts**: A member introduced **SimpleAIChat**, a simple **local-first LLM chat client** for developers seeking control over model interactions, featuring a minimalist UI and customizable prompt structure.
   - The [GitHub repo is available here](https://github.com/sympleaichat/simpleaichat) and the client supports **OpenAI, Claude, Gemini**, and anything that works through **OpenRouter**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1369028007527841873)** (252 messages🔥🔥): 

> `OpenRouter 500 errors, Wayfarer-Large-70B-Llama-3.3, Google Gemini embedding model pricing and rate limits, CPU-only provider feasibility for OpenRouter, OpenAI API errors and debugging` 


- **OpenRouter Suffers Server 500 Snafu**: Users reported encountering **500 errors** on `openai/o3` endpoints.
   - There were also reports of **timeouts** and issues with **Gemini models**, with one user asking *"Are all Gemini models acting like retards?"*
- **Wayfarer-Large-70B-Llama-3.3 Wanderer Vanishes**: The model [latitudegames/wayfarer-large-70b-llama-3.3](https://openrouter.ai/latitudegames/wayfarer-large-70b-llama-3.3) was taken down because *the provider that was hosting it stopped hosting it*.
- **Google's Gemini Embedding Model: Mystery Pricing and Restrictions**: Users are seeking information on **pricing for Google's new Gemini embedding model**, which is heavily **rate-limited** without a paid tier.
   - One user noted it's been almost two months since release and questioned why Google hasn't released it for production.
- **CPU-Only Provider: A Crazy Cost-Effective Concept**: One user proposed the idea of a **CPU-only provider** for less popular LLMs as a cost-effective alternative, despite knowing OpenRouter doesn't host models.
   - Others pointed out that reasonably sized models cannot run efficiently on CPUs, with estimations of **0.5 to 1 tok/sec** speed, while another user shared their experiences running **ML on CPU-only** instances and running out of RAM.
- **Gemini Pro 2.5 gets Updated to 05-06**: The **Gemini 2.5 Pro model** on OpenRouter has been updated to the **05-06** version, which aims to reduce errors in function calling and improve coding performance, with the previous version (**03-25**) now pointing to the latest version.
   - Some users expressed concern over the hard rerouting of dated preview models as a user said *I don't like that they hard reroute the old preview model that has a date in it's name onto the new one, the date means a specific version.*


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1369026428670181416)** (189 messages🔥🔥): 

> `Subscription Issues, Selling Credits, Manus Invitation Codes, Manus Reading Links, Manus vs ChatGPT` 


- ****Subscription Snafu****: A user reports [payment confusion](https://cdn.discordapp.com/attachments/1349440650495398020/1369062672452157500/image.png?ex=681bcff9&is=681a7e79&hm=dee214c79c5fa47efca002b363983adadee50343e9aa9118bf7aef9702ad654b&) with **Manus Starter subscription**, with payments going to both *Google Manus AI* and *Google Butterfly*.
   - The user seeks guidance on resolving the credit discrepancy, suspecting a third-party involvement with *Google Butterfly*.
- ****Credit Commerce Coming?****: A member jokingly suggests the future possibility of **selling Manus credits**, referencing the frustration of running out of credits when almost completing tasks.
   - Another member chimes in, noting *it's already happening*, hinting at an underground market for Manus credits.
- ****Manus Learns the Law****: One user claims success using **Manus** to *read and learn about* the *whole constitution* with links and other laws.
   - Another counters that **Manus isn't suitable for legal tasks**, suggesting *ethical Claude* or specialized *law AI*.
- ****GPT-4.5 Bashing Session****: A user asked if **GPT-4.5** might be better than **Manus** for language and writing, but members said not really.
   - Members do *not recommend wasting Manus credits on just writing*, suggesting *4o* or *o4-mini* for free, and *deepseek v3* or *gemini 2.5 pro* for fully free writing.
- ****Gemini Advanced Scrutinized****: Members debate the merits of paying for **Gemini Advanced**, with one suggesting it's unnecessary for accessing Vertex.
   - Some argue that Gemini Advanced has benefits for regular users, while others see it as *just for noobs*, recommending [AI Studio](https://aistudio.google.com) as a free alternative.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1369189365615689758)** (2 messages): 

> `Triton, torch.index_select, GPU kernel, row-indexing functionality` 


- **Seek Triton Equivalent of `torch.index_select`**: A member is seeking a Triton operation equivalent to `torch.index_select` for fusing operations into a single GPU kernel.
   - They've explored `triton.language.where` and `triton.language.gather` but found them lacking the desired row-indexing functionality, and are seeking alternative tools or approaches for fast row indexing on a GPU, such as [Helion](https://github.com/pytorch-labs/helion).
- **Alternative Approaches for Fast Row Indexing**: The member is also looking for alternative tools besides Triton for fast row indexing on a GPU.
   - They specifically mentioned exploring the possibility of fusing `torch.index_select` with other operations for improved performance, suggesting a need for efficient GPU kernel implementations.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1369247376551706634)** (66 messages🔥🔥): 

> `RTX 6000 PRO, compute capability, A6000, cuda cores` 


- **RTX 6000 PRO tensor cores**: The RTX 6000 PRO will have fifth-generation tensor cores and the hardware should be the same as **B100/200** but its compute capability will probably not be the same as it is a workstation class card with ray tracing units and very few double precision units.
   - The 5090 and 6000 pro are both gb202 so they must have the same tensor core functionality, right?
- **CUDA is designed such that scaling to more SMs should not require code changes**: If a future CC enables some hyper-optimized matmul transistor layout which optimizes matmuls by say, a further 2x, then a 2x increase in past-gen SM's would still be just as good.
   - At some point the data types don't get smaller- you can't do faster arithmetic if your float is -16bits
- **Consumer cards have differences**: GeForce RTX have half-rate tensor core throughput with FP32 accumulation (compared to FP16 accumulation).
   - Only the workstation cards have tensor FP16/BF16/FP8 running full rate with FP32 accumulation, see these two [PDFs](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf) and [another PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
- **A6000 behave differently as they are designed for datacenter use**: With ECC and power/thermal control, you end up with worse performance in practice with the default settings even if they have a bit more cores (they also have a bit lower clock speed and lower memory bandwidth).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1369060953760600116)** (1 messages): 

> `YOLO Model Training, Multi-GPU utilization` 


- **YOLO Model Training with Multiple GPUs**: A member is encountering issues with **YOLO model training**, specifically with utilizing multiple GPUs.
- **Multi-GPU utilization fix**: A member is asking how to fix their code, in which they are trying to train a YOLO model with **4 GPUs** but only **1 GPU** is being used.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1369301664003002420)** (3 messages): 

> `Hugging Face Kernels Community, Leaderboard Kernels Publication` 


- **Hugging Face has Kernels Community**: A member pointed to the new [Hugging Face Kernels Community](https://huggingface.co/kernels-community) page.
- **Sneak Peek at Leaderboard Kernels Publication**: A member shared a *sneak peek* announcing plans to publish leaderboard written kernels via a library in **Transformers**.
   - A link to a potentially related paper was shared: [https://arxiv.org/pdf/2503.20481](https://arxiv.org/pdf/2503.20481).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1369043350128431245)** (10 messages🔥): 

> `Running CUDA code without NVIDIA GPU, Quantization and dequantization for transformer engine, Generating roofline plot` 


- **Cloud Compute to the Rescue**: A member asked how to run code without a local **NVIDIA GPU**, suggesting Google Colab as a possibility.
   - Another member confirmed that **Google Colab** is a viable free option for running **CUDA** code, noting its worth for testing purposes.
- **Quantization Conundrums**: A member inquired about performing **quantization** and **dequantization** for a **transformer engine** using the **transformer engine cast kernel**.
   - They are encountering format compatibility issues between **PyTorch tensors** and the expected input format for **CUDA kernels**, seeking guidance on calling **transformer_engine functions** written in **C++** and **CUDA** from Python and converting the tensors.
- **Roofline Plot Recipes**: A member asked about the optimal approach for generating a **roofline plot**, specifically aiming to keep **memory constant** while varying **intensity** on an **RTX 3090**.
   - Another member provided a link to [Measure Right! Best Practices when Benchmarking CUDA Applications (Nvidia On-Demand)](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51334/) which *goes over some sources of overhead*.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1369052908410245321)** (3 messages): 

> `torchao quantization, LSTM model quantization, CPU vs GPU operators, torch.quantization vs torchao` 


- **Quantization Divergence Discussion Begins**: A member is seeing a **1% metric drop** using `torchao` quantization on both CPU and GPU, but much higher divergence using `torch.quantization`, to quantize a trained LSTM model to predict **y=sin(x)**.
   - The user shares a [script to reproduce the issue](https://pastebin.com/ACeySMtj) that compares MSE and MAE metrics and finds that `torch.quantization` is worse by 25%.
- **`torchao` over `torch.quantization` workflows?**: A member suggests using `torchao` over `torch.quantization` workflows, also pointing to [CPU kernels in torchao](https://github.com/pytorch/ao/tree/main/torchao/experimental#quantizing-models) that might be leveraged for CPU inference.
   - They mention that CPU and GPU operators will be different, but shouldn't see that much divergence.
- **Backend differences between quantization methods**: ChatGPT suggests that performance differences in quantization may stem from backend variations, noting `torchao` uses **cutlass/triton**, while `torch.quantization` uses others.
   - The member further tested quantizing only the fully connected layers, and the performance became comparable.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

s1r_o: https://www.cursor.com/students for students around, this could be of use
  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1369258663226769549)** (9 messages🔥): 

> `WebGPU crashes, Zig and wgpu-native, Shader module creation, WGPUChainedStruct errors, GLSL vs WGSL` 


- **WebGPU Crashes During Shader Module Creation**: A member reported a crash during `wgpuDeviceCreateShaderModule` when creating a shader module in Zig using the wgpu-native C header, throwing a *"Shader not received"* panic.
   - Debugging revealed the crash occurred at the `const module` line, suggesting an issue with the shader module descriptor.
- **Mismatched Shader Types Cause Headaches**: The member found that replacing the `wgsl_stype` in `WGPUChainedStruct` with `WGPUSType_ShaderModuleGLSLDescriptor` prevented the crash, but then expected GLSL instead of WGSL code.
   - It was pointed out that the correct `sType` value should be `0x00000002` according to the [webgpu.h header file](https://github.com/webgpu-native/webgpu-headers/blob/504373dcfc7f3d49f98b392a5115aa87a8f0d163/webgpu.h#L826C34-L826C44).
- **Debugging WebGPU with LLDB in Zig**: A member suggested using LLDB to debug the WebGPU crash, noting that it *works fine with Zig*.
   - LLDB can provide more detailed information about the crash, aiding in identifying the root cause.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1369144464915824660)** (1 messages): 

> `NVIDIA L2 GPU Optimization, Custom Memory Allocator, Elementwise Kernel Builder` 


- **Optimize NVIDIA L2 GPU Siding**: A member released a library to easily optimize for **NVIDIA's L2 GPU sides** for **H100/B200** GPUs, reducing power consumption by **10%+** and improving performance by distributing work to SMs on the same side as the **DRAM/L2** memory they need to access via [cuda-side-boost](https://github.com/ademeure/cuda-side-boost).
- **Craft Custom Memory Allocator**: A member created a **Custom Memory Allocator** for **PyTorch & CUDA**, making it possible for the kernel to know the hash of a page based on the virtual address alone.
- **Build Elementwise Kernel with Ease**: A member built an **Elementwise Kernel Builder**, making it possible to create custom kernels (e.g., **RoPE**, **GELU**, **FP8** microscaling) with greater flexibility than just elementwise or 1:1 operations.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1369040617274867927)** (40 messages🔥): 

> `amd-fp8-mm leaderboard, amd-mixture-of-experts leaderboard, MI300 performance` 


- **amd-fp8-mm Leaderboard Heats Up**: Multiple users submitted successful runs on the `amd-fp8-mm` leaderboard using **MI300**, with execution times ranging from **199 µs** to **9.85 ms**.
- **amd-mixture-of-experts sees new contender**: One user achieved **third place** on the `amd-mixture-of-experts` leaderboard with a time of **212 ms** on **MI300**.
- **MI300 blazing through leaderboards**: Many successful runs were logged on **MI300** for both the `amd-fp8-mm` and `amd-mixture-of-experts` leaderboards, showing continuous activity and optimization efforts.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1369394725412540506)** (1 messages): 

> `Security risk assessment, Competition code platforms` 


- **Assessing Security Risks**: A member asked about thoughts on potential security risks associated with competition code platforms.
   - They were unsure if there's a security risk or reason why this wouldn't work, noting that *effectively what most competition code platforms do anyways*.
- **Competition Code Platforms Analysis**: The discussion involves comparing the security measures of existing competition code platforms.
   - The member seeks to understand if implementing similar practices poses any unforeseen security threats.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1369226078421254154)** (11 messages🔥): 

> `AITER benchmark data, ROCm private repo, Leaderboard submission for amd-mixture-of-experts, CLI resubmits and timeouts` 


- ****AITER Benchmark** data sought**: A member shared a link to get **AITER benchmark data** from [ROCm's gpu_sample_kernels_perf repo](https://github.com/ROCm/gpu_sample_kernels_perf/blob/main/aiter_bench/test_moe.py).
- ****ROCm Repo** might be private**: A member reported a **404 error**, suggesting the [repo](https://github.com/ROCm/gpu_sample_kernels_perf/blob/main/aiter_bench/test_moe.py) might be private, and provided an alternative [link](https://github.com/ROCm/aiter/blob/main/op_tests/test_moe.py) to the **ROCm AITER repo**.
   - A member asked if the *leaderboard submission for amd-mixture-of-experts* is currently active.
- ****Leaderboard submissions** are failing**: A member reported failing **leaderboard submissions** for *amd-mixture-of-experts* with a *Server processing error* related to GitHub Action failures.
   - Another member confirmed healthy job streams [here](https://github.com/gpu-mode/discord-cluster-manager/actions).
- ****CLI Timeout** logic is silly**: Members discussed using the **CLI** and one mentioned occasional failures requiring resubmission, also noting the need to adjust the **CLI** for longer timeouts for **MoE** results due to a hardcoded 5-minute limit.
   - One member joked about the very secure timeout logic and its client-side implementation.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1369025887277809704)** (7 messages): 

> `FP8 Support, Hardware Extensibility, ML Compilers, End-to-End ML Models` 


- **DevOps Support and FP8 Interest Spark**: A member offered assistance on the **DevOps** side and expressed interest in working on **FP8** support, if it's on the roadmap.
   - The member has a decent amount of experience and would be interested in contributing to the project.
- **Modular's Hardware Extensibility Vision Missing Link**: A member reported that the "hardware extensibility vision" link on the [Mojo FAQ page](https://docs.modular.com/mojo/faq/) is broken.
   - The member seeks a video/blog explaining how companies can add hardware support in the future, noting concerns about fragmentation issues similar to OpenCL, XLA, and TVM.
- **ML Compiler Structure Investigation**: A member is interested in understanding how **ML compilers** like **Modular** and **Triton** optimize Python into efficient kernels through various layers of **(ML)IR**.
   - They're seeking developer guides describing the compiler stack from Python constructs to MLIR passes to generated PTX, despite proficiency in CUDA/Triton and cursory understanding of compilers via the MLIR "toy" tutorial.
- **MAX Architecture Unveiled**: The computationally-intensive work (GPU-side kernels, pre- and post-processing) is expressed in **Mojo**, with computational graphs fused and optimized by a graph compiler.
   - Orchestration logic, from definition of the graphs to model composition to model serving is defined in Python, to integrate well with the existing Python ecosystem, such as pulling weights and hyperparameters directly from **Hugging Face**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1369026124004196443)** (120 messages🔥🔥): 

> `Aider 0.82.3, udiff-simple, gemini 2.5, Data Privacy, Vertex API` 


- **Aider 0.82.3 drops in PyPI**: **Aider 0.82.3** was released on PyPI and `uv tool upgrade --all` was used to upgrade it, fixing a bug with `uvx` to run aider from the main branch.
   - However, `udiff-simple` shows up as chat mode instead of specifying the edit format in the model description area.
- **Gemini 2.5 Pro on Vertex API**: Members are reporting access to the new **Gemini 2.5 Pro** on Vertex API, with the previous **03-25** model version redirecting to the **05-06** version.
   - It is not available on AI Studio and has thinking traces similar to OpenAI.  Google notes in their [developers blog](https://developers.googleblog.com/en/gemini-2-5-pro-io-improved-coding-performance/) that *previous iteration (03-25) now points to the most recent version (05-06), so no action is required to use the improved model, and it continues to be available at the same price*.
- **Data Privacy concerns in Aider**: There's discussion around using Aider in corporate settings with concerns about data privacy.
   - Members suggest using LLMs with a no-share data policy, as **Aider** only shares code with the LLM, not with **Aider** itself, and some members are using **Amazon Q** as a cloud provider.
- **Call for Aider documentation Improvements**: A member is collecting documentation requests to improve **Aider** workflows, particularly for new and referred users who struggle to understand the workflow.
   - Users are encouraged to add their thoughts to the [GitHub issue #3934](https://github.com/Aider-AI/aider/issues/393).
- **Gemini UI better than Sonnet?**: There's a discussion comparing **Gemini 2.5 Pro** to **Sonnet 3.7** for UI generation capabilities, with one member noting that **Sonnet** performs better with **React** but another finding Gemini to be preferable.
   - One member has said it *makes my underwear wet, ill be a fountain when r2 is released tho*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1369026341898555553)** (29 messages🔥): 

> `Aider Subtree, Lint Command, HTML representations, OpenRouter, Authentication Error` 


- **Aider Works in Subtree**: A member inquired about using `aider` in project subfolders with the `--subtree-only` option for large mono repos, referencing the [aider documentation](https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo).
   - Another user confirmed this works.
- **Lint Command Misunderstood**: A member asked why `/lint` did nothing after starting `aider` with `aider --model gemini/gemini-2.5-pro-exp-03-25 editor-model openai/gpt-4.1 --edit-format diff`.
   - A user responded that *lint only gives feedback when a problem was found with the code* and to check `lint_cmd` in `/settings`.
- **LLM Returns HTML Characters**: A user noticed `->` being represented as `-&gt` in the prompt response and wondered if it was an Aider or LLM issue.
   - A member clarified that `&gt;` and `&lt;` are **HTML representations of > and <**, so it's neither.
- **Going direct to provider is preferable**: A member asked about using **OpenRouter** vs going directly to providers like **Gemini** or **Anthropic**.
   - Another member responded that *both performance and cost will be better if you go directly to the provider. Openrouter just allows you to test more models easily, without making new account for each provider*.
- **Gemini Auth Error**: A user reported getting a **litellm.AuthenticationError** in Golang repos with a `/vendor` folder, despite adding it to `.aiderignore`.
   - Another user reported *same issue* and is asking aider to refactor after to remove comments.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1369033054911201423)** (108 messages🔥🔥): 

> `PM2 for MCP servers, OAuth with Keycloak for MCP, MPC Server initiating communication with Claude Desktop, Memory options for Claude, Controlling Claude's tool access` 


- ****Keycloak & MCPs: OAuth Odyssey Begins****: A member sought advice on implementing **OAuth** with **Keycloak** in front of an **MCP server** using `http streamable` for the transport layer.
   - Another member suggested a framework, linking to a [governance SDK on GitHub](https://github.com/ithena-one/mcp-governance-sdk) for guidance.
- ****MCPs Break Free: Server-Initiated Claude Prompts****: A member asked if an **MPC server** could initiate communication with **Claude desktop**, periodically sending prompts instead of manual input.
   - A member responded in the negative, noting that while *sampling could sort of do this*, it was likely not in the way the user intended, and that Claude desktop might not support sampling.
- ****Claude's Toolbox Dilemma: The Quest for Selective Tool Access****: A member wanted an easy way to control which sets of tools **Claude** has access to, aiming to load only relevant tools based on the task at hand.
   - Another member shared a workaround using sequential thinking prompts, while others pointed to limiting the number of tools and using a multi-agent system to narrow down toolset choices.
- ****Fast Agent: A Glimpse into the MCP Future?****: **Fast Agent (f-a)** is a tool for building agents with integrated **MCP support**.
   - A member described it as *not a user-facing application but a framework for building agents*, with another member sharing a link to its [MCP State Transfer tricks](https://fast-agent.ai/mcp/state_transfer/).
- ****MCP Proxy Problems: Protocol Version Peril****: A member encountered issues with an **MCP proxy** not supporting the latest version of the spec when connecting to their server.
   - Despite a protocol version mismatch, the proxy seemed to function by lying about the protocol version, though the member expressed concern about releasing code with such a workaround.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1369095700419317801)** (4 messages): 

> `Graphlit, MCP search engine, MCP servers` 


- ****McPoogle** Search Engine Launches Powered by **Graphlit****: The team released a sneak peek at **McPoogle**, a **Graphlit** RAG pipeline-powered search engine indexing 4000+ MCP servers and tools, available at [mcpoogle.com](https://www.mcpoogle.com/?prompt=%22Tell%20me%20about%20the%20Graphlit%20MCP%20Server%22).
   - It allows searching and answering questions about **MCP servers** and tools, inviting user feedback on its performance.
- ****MCP-CLI** Now Supports OAuth**: [MCP-CLI](https://github.com/wong2/mcp-cli) now supports **OAuth**, enhancing its accessibility and security.
   - A [Loom video](https://www.loom.com/share/d2a00956cdb248e5adbc9c31538c7892) showcases its new capabilities.
- ****AWS EC2** Remote MCP Servers Guide**: A Medium article titled *Build and Deploy Remote MCP Servers to AWS EC2* was shared, offering a guide on deploying **MCP servers** on [AWS EC2](https://medium.com/@tadeodonegana/build-and-deploy-remote-mcp-servers-to-aws-ec2-5888514892c4).
   - This is useful for folks who want to move their MCP servers to the cloud.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1369104161970454642)** (45 messages🔥): 

> `api-inference.huggingface.co, Object Tracking models, DOI deletion, Summarising caselaw, Data Parallelism vs Model Parallelism` 


- **api-inference.huggingface.co is dead**: The endpoint **api-inference.huggingface.co** is now deprecated, and users should use the new endpoint instead.
   - It's not clear if there was a deprecation notice, and **LangChainjs** is still using the old endpoint.
- **Object Tracking Models Lack Inference Providers**: A member asked why there are no **Inference Providers** for any of the **Object Tracking models**, noting the removal of providers for the **DETR model** from Facebook.
   - They pointed out that almost all models are in the same condition.
- **DOI cannot be deleted**: A member asked for help to delete a DOI mistakenly attached to a test repo.
   - It was suggested that deleting the DOI might require contacting support via email, linking to [this discussion](https://discuss.huggingface.co/t/change-authors-of-citation-with-doi/145654/4).
- **LLMs summarize Caselaw**: A member is working on a project to summarise **caselaw** with tags and needed help selecting a model.
   - The advice was to check the [Hugging Face LLM leaderboard](https://huggingface.co/spaces/fr-gouv-coordination-ia/llm_leaderboard_fr#/) and try promising models on their specific dataset, also noting the importance of dataset curation in training.
- **Data Parallelism vs Model Parallelism**: A member inquired about the point of switching to **model parallelism** if a mid-range model can be trained with data parallelism.
   - They also asked about needing to write **handler.py** if the model isn't incorporated into official **Diffusers** or **Transformers** releases, linking to [custom handler documentation](https://huggingface.co/docs/inference-endpoints/guides/custom_handler).


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1369033505811599423)** (9 messages🔥): 

> `AI Study Group GitHub Repo, List of AI Papers Plan, Discord Usage` 


- **New AI Study Group Repo Arises!**: A member transitioning from the games industry for the past five years has organized AI resources and created the [AI-Study-Group GitHub repo](https://github.com/ArturoNereu/AI-Study-Group).
   - The member welcomes contributions of useful papers, tools, or underrated gems.
- **Dense AI Papers spark Headache Discussions!**: A member shared a plan to tackle a list of dense AI papers, prompting some humorous reactions, as shown in the attached image [here](https://cdn.discordapp.com/attachments/898619964095860757/1369046722847834143/IMG_3863.png?ex=681bc11e&is=681a6f9e&hm=c2d311c4ef37480727774cc6d54707492a73ddfb97c18c9dc2e78bd680e1ead4).
   - Other members jested the *plan gave them a headache*, and another questioned if the plan was overly ambitious.
- **Member Jokingly Learns Discord!**: After another member joked about getting a headache from the AI reading list, one of the members said *today im leaning how to use Discord*.
   - No further context was provided.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1369094581546782720)** (8 messages🔥): 

> `Huggingface Desktop App, Dank Leaks, Flux-Pro-Unlimited AI image generator, candle-holder` 


- **Desktop App for HF uploading rises**: A member is developing a desktop app for **uploading to Hugging Face** and **model management**, drawing from existing Jupyter/Colab notebooks and seeking collaborators to refine the messy code.
   - The [project](https://github.com/Ktiseos-Nyx/Huggingface-Desktop/tree/main) may have a non-functional uploader due to import issues, but contributions are welcome.
- **Dank Leaks App debuts**: A member showcased "Dank Leaks," a work-in-progress app written in **C-17/notcurses**, demonstrating its functionality via a screen recording.
   - The member shared a [screen recording](https://cdn.discordapp.com/attachments/897390720388825149/1369235033193451580/Screen_Recording_2025-05-06_at_09.04.26.mov?ex=681bc7bf&is=681a763f&hm=017faa7cad446bd78255f6074e626df19fac3ce34a21702ae2a932a0e9b8e9d5&) of the app.
- **Free Unlimited Flux-Pro AI image generator surfaces**: A member shared a link to **Flux-Pro-Unlimited**, a *partially uncensored* AI image generator available on Hugging Face Spaces for research purposes.
   - It can be found at [https://huggingface.co/spaces/NihalGazi/FLUX-Pro-Unlimited-](https://huggingface.co/spaces/NihalGazi/FLUX-Pro-Unlimited-) which is described as being *nothing novel, but an useful service (no ZeroGPU)*
- **Candle Holder project mentioned**: A member suggested checking out [candle-holder](https://github.com/gabrielmbmb/candle-holder) in response to the Flux-Pro AI image generator.
   - No additional context was provided as to why this would be useful.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1369240477395587102)** (1 messages): 

> `HiDream LoRAs, Quantization Support, Memory Savings` 


- **HiDream LoRAs Gains Quantization Support**: Quantization support via `bistandbytes` has been implemented for training **HiDream LoRAs**, promising significant memory savings.
   - See the [Diffusers documentation](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_hidream.md#using-quantization) for details.
- **Training Memory Footprint Shrinks Massively**: Enabling quantization during **HiDream LoRA** training drastically reduces memory usage, down from **57.64 GB** to **34.60 GB** *after device placement*.
   - With quantization, memory usage *after backward* is reported at **36.90 GB**, compared to **59.93 GB** without.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1369225808408739902)** (13 messages🔥): 

> `GPU memory considerations, Emotion Classification Models, FullyShardedDataParallelPlugin Error` 


- **GPU Memory Fits Model with bfloat16**: A member confirmed that a **6GB GPU** should be sufficient for fitting the model, suggesting using **bfloat16** and ensuring it's not loaded for training.
   - Another member confirmed that they *just tried bfloat16*.
- **Sentiment Model Misses the Mark**: A user found that the `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` model misclassified the emotion in the statement *Looking for friends Anyone in sharjah*.
   - The model incorrectly assigned high scores to **anger** (0.995), **annoyance** (0.989), **disappointment** (0.978), and **sadness** (0.978).
- **Distilroberta Excels in Emotion Detection**: The `j-hartmann/emotion-english-distilroberta-base` model gave a more accurate sadness score for a particular text.
   - However, for the text *Cute*, it incorrectly labeled the emotion as **disgust** with a score of 0.8765.
- **Fixing Accelerate's FSDPPlugin**: A user shared code for the `FullyShardedDataParallelPlugin` using `accelerate` and reported an error.
   - The code included settings for `fsdp_version=2` and configurations for state and optimizer state dictionaries to offload to CPU.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1369062492025782293)** (3 messages): 

> `HF API Limits, GAIA files evaluation, Gemma3 vs Qwen` 


- **Hitting Free API Limits on Hugging Face**: A member mentioned hitting a free limit on **HfApiModel**.
   - They were prompted to either add credits to their **HF account** or select a different provider where they already have credits.
- **Accessing GAIA Files for Evaluation**: A member inquired how to access **GAIA files** during the evaluation process, mentioning they've downloaded them to their computer for local testing.
- **Qwen outperforms Gemma3 with tool support**: A member shared their experience using **Gemma3** initially and then switching to **Qwen**.
   - They noted that **Qwen** is better because it supports tool calling, which **Gemma3** does not, and that they were able to complete the assignment even with **Gemma3:4b**.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1369029705981759639)** (12 messages🔥): 

> `GAIA Questions, Agent UI Timeouts, Final Agent Build, Frameworks, Final Challenge Solution` 


- **GAIA questions take long time**: Members are reporting that **GAIA question runs** are taking a very long time (**at least an hour**) and costing a lot of money (>$15).
   - One user reported spending *$5+* to get *5 questions* right, with each run taking approximately **30 minutes**, and one even only got **20%** of the questions right after an hour.
- **Agent UI timeouts during long runs**: A member is experiencing **UI timeouts/errors** when running their agent, preventing them from seeing question results and making changes.
   - They are seeing `net::ERR_NETWORK_IO_SUSPENDED` in the browser console and asking if it's a **Gradio issue**.
- **Guidance Needed for Final Agent Build**: A member is seeking clarification on the requirements for the **final agent build**, specifically what type of agent to build and what factors are considered for passing.
   - They are also asking if additional frameworks like **OpenAI agents SDK** will be added to the course.
- **Final Challenge Solution blog post**: A member shared a [blog post](https://guillaume-fradet.com/posts/making-ai-think-and-act-my-approach-to-the-hugging-face-ai-agents-challenge/) detailing their solution for the **final challenge**, achieving a **60%** success rate.
   - The author appears to be **Guillaume Fradet**.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1369102786410713201)** (71 messages🔥🔥): 

> `M4 Macbook Pro vs RTX 5070Ti for LLM inference, Diffusion Models as Evolutionary Algorithms, Search vs Optimization, AI-assisted Academic Articles and Patents, Claude Code with Gemini` 


- **M4 Macbook Pro vs RTX 5070Ti Faceoff for LLM Devs**: A member is deciding between a **M4 Macbook Pro** and a **RTX 5070Ti Linux laptop** for local llm inference and general development; another member noted that with an **M4 Macbook Pro 48 GB** the token generation speed with *ollama Qwen 2.5-coder* is pretty good and smooth.
   - A third member suggested using a **Mac** or even a **Mac Studio/Mini**, recommending to just rent GPUs for cheap if needed.
- **Diffusion Models Evolve as Evolutionary Algorithms**: A member cited the paper [Diffusion Models as Evolutionary Algorithms](https://arxiv.org/abs/2410.02543), explaining how it reveals that **diffusion models** can be viewed as **evolutionary algorithms**.
   - The member clarified that *the denoising steps correspond to successive generations of model refinement*, which bridges two major strands of generative modeling and aligns with their proposed generator paradigm.
- **Search Grapples with Optimization in Chat**: Debate sparked around the relationship between **search** and **optimization**, with one member arguing that *optimization is a sub of search*.
   - Counterarguments stated that while **search** can be reduced to **optimization** on paper, the process, assumptions, and tooling are significantly different.
- **AI Agents: Academic Article Authors Arise**: One member is developing agents to write patents and academic articles, clarifying that the *agents are working in my virtual research lab, nobody will see them*.
   - Another member shared a link to [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) and the first member clarified that the agents he is developing are **dynamic** and *a society of minds*.
- **Claude Code with Gemini?**: A member inquired about the best **Claude Code** but with **Gemini**, or an **OpenAI's Codex** hacked to use any models.
   - There were no replies to the question.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/)** (1 messages): 

k_nearest_neighbor: I won't be able to do a paper today, but feel free if anyone wants to.
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1369030429134426243)** (13 messages🔥): 

> `Em Dashes, OAI, US Gov, Chinese Models, Deepseek vs OAI, Sam Altman` 


- **Em Dash Usage Evolves!**: The proper use of the **em dash** is being influenced by **AI models** like **chatGPT**, which learn from human data and may inadvertently promote its usage.
   - As AI-generated content becomes more prevalent, there's a question of whether humans will start adopting the em dash more frequently, influencing writing styles.
- **OAI's Gov Gambit Against Chinese Models?**: Some members speculate that **OpenAI** might lobby the **U.S. government** to ban **Chinese AI models**, citing the availability of open-source alternatives.
   - The argument is that there's *"no need for CCP controlled models"* when open-source options exist, raising concerns about competition and control.
- **Deepseek's Post-Training Prowess Prevails!**: A member believes that **Deepseek** could outperform **OpenAI** due to superior **post-training**, even if **OpenAI's** base model is initially better, referencing [Microsoft's MAI-DS-R1 on Hugging Face](https://huggingface.co/microsoft/MAI-DS-R1).
   - This highlights the critical role of post-training techniques in achieving state-of-the-art model performance, potentially outpacing initial architectural advantages.
- **Altman's Antics Averted: Nonprofit to Retain Control?**: Referencing a [CNBC article from 2025](https://www.cnbc.com/amp/2025/05/05/openai-says-nonprofit-retain-control-of-company-bowing-to-pressure.html), one member suggests that **Sam Altman** might have faced legal repercussions if he had tried to alter the **nonprofit structure** of **OpenAI**.
   - This implies a potential power struggle and the importance of maintaining the original governance structure to ensure ethical and responsible AI development.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1369087963429867640)** (8 messages🔥): 

> `Podcast Length Discrepancies, Inserting Instructions, Audio Overview Experiences, Mind Map Generation` 


- **Podcast Length Varies by Language**: A user reported their podcast is **40 minutes** in English but capped at **14 minutes** in other languages, with no immediate solution provided.
   - The user's issue highlights potential limitations or bugs in the podcast creation tool related to language processing or content generation.
- **Instruction Insertion Inquiry**: A user asked how to insert instructions, and another user replied with instructions to check the "customize" button, below the 'generate' button.
   - There's also a note that customizing may not be available on the free plan, referencing an attached image illustrating the option in the user interface.
- **Audio Overview Success**: A user shared that their audio overviews had no repeated content, static noise, phantom voices, or fabrication, and the overviews followed instructions well.
   - The user attributed the success to both their prompt and the source material, suggesting that certain sources uniquely lend themselves to following instructions.
- **Mind Map Generation Technique**: A user details their process of generating mindmaps from sources using a custom prompt.
   - They regenerate using the prompt: *Create a mind map from the sources. List topics as central ideas, main branches, and sub-branches. Output it in Markdown format.*, then they feed the output to [markmap.js.org](https://markmap.js.org/) and save as "interactive HTML".


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1369028757997748304)** (70 messages🔥🔥): 

> `NotebookLM Audio Transcription, Gemini Flash 2.5 Confirmation, Cantonese Language Support, NotebookLM's Gemini Version, Interactive Mode in NotebookLM` 


- **Generate Audio Transcriptions in NotebookLM**: A user sought ways to generate transcriptions or captions similar to Google Meets for NotebookLM audio, suggesting screen sharing into Google Meets to generate a recording and transcript as a workaround.
   - The user praised NotebookLM for providing more detailed and complete responses compared to ChatGPT, especially for fact-checking documents, and found the audio overview feature to be impressively similar to a real podcast.
- **Confirm Gemini Flash 2.5 Usage in NotebookLM**: Users discussed how to confirm they were using **Gemini Flash 2.5** in NotebookLM, with one user pointing to a specific post as confirmation.
   - Some users expressed doubts about NotebookLM's reasoning capabilities, suggesting it might be using an older version (2.0), while others noted that Gemini 2.5 Pro works brilliantly in AI Studio.
- **Cantonese Language in NotebookLM is Coming**: A user asked about the availability of **Cantonese** language support in NotebookLM.
   - A member of the team replied saying that they are *working on it*.
- **Issue importing web pages due to domain restrictions**: A user reported encountering an error message stating *This web page cannot be imported due to domain restrictions* when trying to add a NotebookLM source to a project and posted a [screenshot](https://ibb.co/My2JzWHp).
   - Domain restrictions prevent importing certain webpages as NotebookLM sources.
- **Troubleshooting Conversation Styles**: A user questioned how to make the conversation style feature work in NotebookLM, reporting that the AI always acts the same regardless of the instructions provided.
   - A member suggested uploading custom instructions or a prompt document as a source to influence the AI's behavior, along with trying the live mode to interject.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1369031405899874334)** (63 messages🔥🔥): 

> `Dwarkesh Automated Firms Essay, Revenue Numbers, Exa Blogpost, OpenAI Acquires Windsurf, Gemini 2.5 Pro Elo Bump` 


- **Dwarkesh details Fully Automated Firms**: [Dwarkesh's essay and video](https://www.dwarkesh.com/p/ai-firm) draws heavily from [Gwern's Backstop](https://gwern.net/backstop) and discusses the potential of **Agentic DAOs**.
- **AI Startups post Revenue Numbers**: A member shared [revenue numbers](https://x.com/tanayj/status/1919489023602786737) and noted the high costs associated with **GPU time**, **engineer salaries**, and **executive compensation** in **HCOL cities**.
- **Exa relaunches with BM25 Optimization Blogpost**: **Exa** is back on X with a new blogpost on [BM25 Optimization](https://exa.ai/blog/bm25-optimization).
- **OpenAI strikes $3B Windsurf Acquisition Deal**: **OpenAI** is set to acquire **Windsurf** for **$3 billion**, according to [Bloomberg](https://www.bloomberg.com/news/articles/2025-05-06/openai-reaches-agreement-to-buy-startup-windsurf-for-3-billion).
- **Gemini 2.5 Pro has Massive ELO Bump**: There are claims of a big ELO bump with the new **Gemini 2.5 Pro** update per [this tweet](https://x.com/scaling01/status/1919771796334616759) and [Google DeepMind announcement](https://x.com/googledeepmind/status/1919770265711419826).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1369026617384505536)** (38 messages🔥): 

> `OpenAI Public Benefit Corp, Flights to the US, RL Environments Hackathon, Fine tuning a base model LLM, M4 Macbook Pro vs RTX 5070Ti Linux laptop` 


- ****OpenAI Mulls Public Benefit Corp Structure****: Given **OpenAI** raised **$8B**, there's speculation that they believe their current product spend is high enough to generate at least a **1.3x+** return on this cash, as detailed in a [WSJ article](https://www.wsj.com/tech/ai/openai-to-become-public-benefit-corporation-9e7896e0?st=ifeJpvreal).
- ****US Flights are half price****: Flights to the US are currently about **half** their usual rate due to fears of detention and device searches.
- ****Upcoming RL Environments Hackathon****: Nous Research is hosting an [RL Environments Hackathon](https://cerebralvalley.ai/e/nous-research-rl-environments-hackathon-9be3062ade).
- ****Base Model Fine-Tuning Beyond Chat Assistants?****: A member asked about the possibility of fine-tuning a base model LLM to be something other than an instruction-following chat assistant.
   - Responses indicated it's possible, with controlling robot movement and stock predictions as examples, though *data is hard*.
- ****M4 Macbook Pro vs RTX 5070Ti Linux Laptop for local LLM inference****: A user is deciding between a **M4 Macbook Pro** and an **RTX 5070Ti Linux laptop** for local llm inference and general deving.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1369033904186724476)** (5 messages): 

> `AnySphere, AGI Agents` 


- **AnySphere's Mammoth $900M Ask**: A member questioned why **AnySphere**, the maker of Cursor, needs **$900M** with only around **100 people**.
   - Another member joked that they probably need **1000 people** instead.
- **AGI Agents: Postponed?**: A member asked point blank if **AGI agents** are delayed.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1369043843739029607)** (27 messages🔥): 

> `Page Faults, Disk Swapping, Torch, Transformers, vLLM` 


- **User Faces Paging Issues Importing Libraries**: A user is experiencing significant page faults and disk swapping when importing standard Python libraries like **torch**, **transformers**, and **vLLM** despite using **conda**, **venv**, and **uv** for package management.
   - They've tested installing libraries in various areas, changing environment variables, and confirmed that others using the same machines do not encounter this issue, potentially indicating a user-specific or broken configuration.
- **SysAdmins Stumped by Rare File System Issue**: After debugging attempts and reaching out to system administrators, the root cause remains unclear for the user's unusual system behavior involving excessive disk swapping.
   - The issue persists despite being on the same disks as other users in the lab who do not experience similar problems, suggesting a possible broken table or UID assignment problem.
- **Hardware and Memory Integrity Suspected**: A member suggested the issue might stem from memory thrashing or broken bits, recommending a memory test to rule out hardware faults.
   - Another proposed that the problem could be related to Input/Output limits or misconfigured pagemaps, implying the system is constantly undergoing page faults and hindering debugging efforts.
- **User Seeks Deeper Insights on Transformer Limits**: A member who is an AI student is looking for insights on **RAG/CAG**, prompt engineering, fine-tuning, and the limits of transformer context windows, as well as memory-centric cognitive architectures like **EM-LLMs**.
   - They are taking the question to the Hugging Face Discord, and expressed that their professors cannot fully address these areas.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1369034358849142784)** (3 messages): 

> `Data vs Model Parallelism, arXiv:2305.18153 Citations, Anthropic Work` 


- **Data Parallelism vs Model Parallelism**: A member inquired whether there is a point in switching to **model parallelism** if one can train a mid-range model with a very large amount of time using **data parallelism**.
   - The context implies a scenario where time is not a constraint, but the choice between **data** and **model parallelism** is being considered.
- **Papers Citing arXiv:2305.18153**: A member suggested looking at papers that cite the paper **arXiv:2305.18153**.
   - This recommendation implies that related research or advancements might be found in the papers that cite this specific work.
- **Anthropic's Related Work**: A member noted the existence of related work from **Anthropic** in this area.
   - They didn't check the authors, but implied that **Anthropic** has relevant contributions to the topic under discussion.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1369032973596491890)** (9 messages🔥): 

> `Circuit Identification, Anthropic Transformer Circuits, Monosemanticity, Interpretability Tooling Challenges, TransformerLens Limitations` 


- **Exploring Model Behavior with Circuit Identification**: A member suggested investigating **circuit identification** to understand model behavior, noting that ambitious approaches often have limitations in accurately representing the model's internal processes.
   - They recommended exploring research papers on **grokking**, **BIG-Bench**, and **content bias reasoning**, along with **Anthropic transformer circuits** and **Towards Monosemanticity** as starting points.
- **Tackling Tooling Gaps in Interpretability Research**: A member inquired about key **tooling challenges** in interpretability and empirical alignment work, seeking real-world examples such as performance issues with activation extraction from Pytorch at scale.
   - Another member noted that some interp libraries like **transformerlens** and **nnsight** still have limited functionality in some departments (eg. transformerlens only supports a limited list of pretraiend models).
- **Navigating Challenges in SAE Training and LLM Calls**: A member mentioned their team's decision to **save activations to disk** and train the **SAE** in a separate stage, leading to challenges with wrangling "big data," though SAE training itself is relatively quick.
   - They also highlighted the extensive **LLM calls** required for auto-interpretability as another empirical tooling challenge.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1369235449419403345)** (1 messages): 

> `Multimodal VLMs, Community Research Hub, Weekly Updates` 


- ****VLMs Research Hub** Initiated!**: A community-driven hub for **multimodal researchers** has been created and is actively being maintained.
   - The creator of the hub welcomes contributions, suggestions, and feedback, updating the hub on a weekly basis: [All-Things-Multimodal](https://github.com/thubZ09/All-Things-Multimodal.git).
- **Welcoming Community Contributions!**: The hub is open for contributions from the community, encouraging researchers to share their findings and resources.
   - Suggestions and feedback are highly appreciated to ensure the hub remains comprehensive and up-to-date.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1369053259746119762)** (19 messages🔥): 

> `Codegen for new models, Reducing engineering time for new models, Tokenizer Support, HF Transformers Adapter, Qwen3 Support` 


- ****Codegen Considered for Boilerplate Reduction****: Members discussed using **codegen** to reduce boilerplate when adding new models to Torchtune, drawing inspiration from faster model support in libraries like **Unsloth**.
   - Concerns were raised about relying on tracing keywords and the importance of users understanding the underlying code, suggesting well-written tutorials as an alternative to extensive codegen.
- ****Engineering Time on New SoTA Models is Top Priority****: The primary goal is to **reduce the engineering time** required to use new **SoTA models** with Torchtune while maintaining **reasonable performance**.
   - A suggestion was made to tackle the challenge by identifying boilerplate versus difficult aspects, focusing on simplifying the latter, like tokenizer support, before automating boilerplate generation.
- ****Tokenizer Support Deemed Nontrivial****: **Tokenizer support** was identified as a nontrivial task, with a call for a generalized solution, while more tedious tasks could be automated through scripts before considering codegen.
   - The discussion mentioned leveraging **HF (Hugging Face) configurations** for tokenizers and generating parity check numbers using HF models.
- ****HF Transformers Adapter Envisioned for Faster Model Support****: One approach suggested was to create a generic **HF adapter** that loads HF models and allows mappings to work with different training features.
   - This would facilitate faster addition of new models and enable finetuning, with the option to later implement a "real" implementation with full feature support.
- ****Qwen3 Hype:  Codegen Could Support Model Version with Ease****: It was mentioned that for some model families like **Qwen**, new versions often require mostly new boilerplate, which could be largely added with codegen.
   - The discussion highlighted the marketing advantage of rapidly supporting new models like **Qwen3** upon release, even with initial restrictions, before a full-featured implementation is ready.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1369404139787911299)** (3 messages): 

> `Modular Puzzles on Macbook, Apple Silicon GPUs, NVIDIA GPU architectures` 


- **Puzzling Modular on Macbooks?**: A member inquired about running **Modular Puzzles** on a Macbook.
   - Another member clarified it's not directly doable on **Apple Silicon GPUs** yet, suggesting remote work on a **GPU-attached cloud instance** as a workaround.
- **NVIDIA GPUs to the rescue!**: Members clarified that current available **NVIDIA GPU architectures** for **Mojo GPU** programming are Turing, Ampere, Hopper, and Blackwell (**RTX 20XX - 50XX series**).


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1369053480874016859)** (12 messages🔥): 

> `Blogging Platforms, Ownership Semantics in Mojo, Mojo Getting Started Guide Errors, Comptime Try-Except Handling` 


- **Blot.im preferred blogging platform**: A member suggested using [Blot.im](https://blot.im/) as a blogging platform which supports **markdown** but is a *paid* service.
- **Unsafe Pointers for Struct Mutation**: A member inquired about using **unsafe pointers** to mutate structs, specifically asking about alternatives to global `var`.
- **Getting Started Guide Troubles**: A user reported errors with the last example in the [Mojo Getting Started Guide](https://docs.modular.com/mojo/manual/get-started/).
- **Comptime Try-Except Woes**: A member asked if it's possible to use `try ... except ...` inside of computations performed at `comptime`.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1369058971092254720)** (2 messages): 

> `MCP Hackathon, Agent Communication, Deep Research Agent` 


- **LlamaIndex Sponsors MCP Hackathon in Tel Aviv**: LlamaIndex is sponsoring the **Big MCP Hackathon** from @aitinkerers in Tel Aviv, focusing on building **MCP-powered apps** that enable agent-to-agent communication and experimentation, as detailed in [this tweet](https://twitter.com/llama_index/status/1919499875332587579).
- **LlamaIndex Teaches How to Build Deep Research Agent**: LlamaIndex introduced a workshop tutorial that covers building a **multi-agent system** for **deep research** from scratch, using **AgentWorkflow**, as seen on [this tweet](https://twitter.com/llama_index/status/1919856812893077565).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1369220679282720770)** (6 messages): 

> `Property Graph Indexes, LlamaIndex GraphRAG, LangChain GraphRAG, Vector Database Storage` 


- ****Property Graph Indexes Explored****: A member inquired about the usage of property graph indexes and shared their implementation using LlamaIndex's documentation in [this notebook](https://github.com/tituslhy/shiny-engine/blob/main/notebooks/hybrid_property_graph.ipynb).
   - The member found that the index sometimes fails to answer questions because it cannot retrieve nodes from the graph or vector database, leading to questions about how the vector database stores nodes.
- ****LlamaIndex vs LangChain Graph Generation****: The same member observed that graphs generated using LlamaIndex's property graph index code are much denser, as visualized [here](https://github.com/tituslhy/shiny-engine/blob/main/images/llamaindex_neo4j.png) compared to LangChain as visualized [here](https://github.com/tituslhy/shiny-engine/blob/main/images/groq_kg.png).
   - The member noted that a [LangChain notebook](https://github.com/tituslhy/shiny-engine/blob/main/notebooks/large_llm_knowledge_graph.ipynb) encoded texts in the nodes within the vector database.
- ****Vector DB Storage Strategies in GraphRAG****: The member suggested that the vector database stores each node in the graph, as opposed to each text chunk in normal RAG.
   - Graph density is likely due to the default prompts being used.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1369291452739227780)** (2 messages): 

> `M4 Macbook vs RTX 5070Ti for local LLM, tinygrad discord rules` 


- **Macbook Pro M4 or RTX 5070Ti?**: A member inquired about whether a **Macbook Pro M4** or an **RTX 5070Ti Linux laptop** would be better for local LLM inference and general development.
   - The member mentioned being a long-time **Linux desktop user** and having heard good things about the **M series Macs**.
- **Read the rules!**: George Hotz reminded a user to read the rules of the Discord, stating that it is a place for discussion of **tinygrad** and **tinygrad development**.
   - The user asked about getting a new machine, choosing between a **M4 Macbook Pro** and a **RTX 5070Ti Linux laptop**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1369342537940209806)** (3 messages): 

> `Bounty Picking, Rockship Device` 


- **Bounty Hunters inquire about Picking Process**: Members seek guidance on how to pick bounties from the [Google Sheets bounty list](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0) and whether longer descriptions exist elsewhere.
   - They were also wondering if they should mark a bounty as *"taken"* before starting to avoid multiple people working on the same thing.
- **WIP PR Locks Bounties**: A member suggested that creating a **WIP PR** is required to lock a bounty to prevent multiple people from working on the same thing.
   - They also guessed that it's expected for bounty hunters to *"lurk around for some time and tinker to understand it well enough so that the existing description suffices"*.
- **Rockship Device Bounty Clarification**: In regards to a bounty for a **new Rockship device**, members are asking whether they are supposed to use the SDK or the open-source driver.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1369416949817413894)** (1 messages): 

> `Auth0 Workshop, AI Agent Security, Entrepreneurship Track Prizes` 


- **Auth0 Workshop Tomorrow: Secure Your Agents!**: A special workshop with **Auth0** is happening tomorrow (5/7) at **10:00 AM PT** to learn how to secure AI agents with professional authentication solutions; tune in to the [YouTube livestream](https://www.youtube.com/watch?v=wB29IJ3AEjY).
   - The workshop will cover implementing robust authentication in LLM-powered agents, integrating **Auth0's** services with AgentX, and addressing security considerations unique to AI systems.
- **Auth0 Prizes for Entrepreneurial Agents**: **Auth0** is sponsoring special prizes for the Entrepreneurship Track, with up to **$5,000** for 1st place, **$3,000** for 2nd place, and **$2,000** for 3rd place.
   - These prizes are for teams that successfully integrate **Auth0.ai** into their projects.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1369168268669227058)** (2 messages): 

> `HuggingFace Credits, Quiz Scores` 


- **Members Anticipate HuggingFace Credit Delivery**: A member reported that, after being selected to receive **HuggingFace credits**, the credits have not yet been added to their organization.
   - They mentioned they already filled out the required form, indicating they are awaiting the credit allocation.
- **Quiz Scores Remain Elusive**: A member noted that the scores for the last **two quizzes (Quiz 11 and Quiz 12)** are still unavailable.
   - They are seemingly awaiting the release of these scores to assess their performance.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1369354927842922619)** (1 messages): 

> `LLMs and Conditional Statements, Formal Methods in LLMs, Representing Conditions for LLMs` 


- **LLMs Handle Conditional Statements like Voting**: A member questioned how **LLMs** remember **conditional statements** and produce good results, using voting eligibility (*18 years old, citizen, voter ID*) as an example.
   - They expressed struggling with how to represent these conditions for **LLMs** since the start of the course.
- **LLMs using Formal Methods**: The same member wondered about the role of **formal methods** in enabling **LLMs** to handle conditional logic effectively.
   - They sought insights into how these methods could be applied to represent complex conditions for **LLMs**, especially in the context of long-term memory.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1369354605183373432)** (1 messages): 

> `LLM Reasoning, LLM Formal Methods, LLM Knowledge Representation` 


- **LLMs Handle Conditionals with Aplomb**: A member inquired about how **LLMs** remember **conditional statements** and produce good results without the explicit use of formal methods.
   - They gave the example of the requirements to vote: *"one should be 18 years old, should be a citizen, should have voter id"* and pondered how to represent this knowledge within an LLM.
- **LLMs and Voter Eligibility**: The discussion centered on how **LLMs** effectively manage and apply **conditional logic**, particularly in scenarios like voter eligibility requirements.
   - The example involved specific conditions such as age, citizenship, and voter ID, raising questions about the underlying mechanisms enabling LLMs to accurately process and apply these rules.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1369248463317110876)** (5 messages): 

> `Cohere-AI npm, Aya vision` 


- **Cohere-AI npm Package Supports Aya Vision**: A member asked whether the **Cohere-AI npm package** supports **Aya Vision** and another member confirmed that it does, and shared a code snippet.
   - The code snippet shows how to use the **CohereClientV2** class to chat with the **c4ai-aya-vision-8b** model, sending a message with text and an image URL.
- **User Implements and Integrates Cohere-AI**: After receiving the code snippet and confirmation, a member said that he implemented and integrated **Cohere-AI** for the expedition.
   - No further details were provided about what the "expedition" refers to.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1369068297491976354)** (3 messages): 

> `Claude System Prompt, Chat Template Generation with Python, GPT4All Integration` 


- ****Claude's** System Prompt Surfaces!**: A member shared a link to a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1kfkg29/claude_full_system_prompt_with_all_tools_is_now/) containing the full system prompt for **Claude** with all tools.
   - The post generated immediate interest and discussion, with members eager to explore its potential applications.
- **Python Help Needed for **Chat Template**!**: A member requested assistance with generating a **chat template** using **Python**, indicating they were new to this area.
   - Despite having the *transformers* module installed, they were encountering syntax errors and sought guidance on resolving the issues.
- **Can **Claude** Prompt Integrate with **GPT4All**?**: A member inquired about how to integrate the leaked **Claude** system prompt into **GPT4All**.
   - This query highlights the community's interest in leveraging insights from different **LLM** systems to enhance the functionality of **GPT4All**.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1369121502733008896)** (1 messages): 

> `MLOps, LLMOps, AI Project Lifecycle, Data Phoenix` 


- **Tame AI Projects With MLOps and LLMOps**: A webinar on **MLOps** and **LLMOps** is scheduled for Wednesday, May 28 at 10:00 AM PDT, focusing on establishing reliable processes and accelerating the delivery of AI solutions to eliminate chaos.
   - The webinar will cover the lifecycle of **ML** and **LLM** projects, tools, roles, and practices that form the foundation of **MLOps/LLMOps**, and the **MLOps/LLMOps maturity model**.
- **Grid Dynamics Architect to Speak on Reliable AI**: **Dmytro Spodarets**, DevOps/MLOps Architect at **Grid Dynamics** and founder of **Data Phoenix**, will be the speaker at the webinar.
   - Those interested can save their spot via [this link](https://lu.ma/qhytft9t).


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1369284039680200784)** (1 messages): 

> `Experiment Setup vs. Model Application, Model Validation Strategies` 


- **Debate: Experiment Setup vs. Mere Model Application**: A discussion arose on whether a full **experiment setup** (training and testing) is mandatory, or if simply **applying a pre-existing model** suffices.
   - The prevailing view suggests that an experiment setup is crucial for validation, but this depends on the project's context and aims.
- **Exploring Model Validation Methodologies**: The conversation extended to various **model validation** strategies beyond basic training and testing splits.
   - Suggestions included **cross-validation techniques**, **robustness checks against adversarial examples**, and **validation on real-world data** to ensure broader applicability.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1369381513216393216)** (1 messages): 

> `Windsurf Wave 8, Windsurf Reviews, Knowledge Base, Conversation Sharing, Teams Deploys` 


- **Windsurf's Wave 8 Kicks off with Team and Enterprise Features**: Windsurf announced the start of **Wave 8**, releasing new features across multiple days, starting with organizational tools.
   - The announcement included a [blog post](http://windsurf.com/blog/windsurf-wave-8-teams-and-enterprise), [changelog](https://windsurf.com/changelog?cachebust=202405061200), and links to their [X](https://x.com/windsurf_ai/status/1919820747037392982) and [Bluesky](https://bsky.app/profile/windsurfai.bsky.social/post/3lojj34nq622q) accounts, as well as a [YouTube launch video](https://youtu.be/t7GQFFopQFY) and a [Reddit community](https://reddit.com/r/windsurf).
- **Windsurf Reviews (Beta) Autopilot PRs**: Windsurf introduced **Windsurf Reviews (Beta)**, a GitHub app that automatically reviews PRs and edits titles/descriptions based on specified guidelines.
   - This feature aims to streamline the code review process within organizations.
- **Knowledge Base (Beta) Grounds with Google Docs**: **Knowledge Base (Beta)** was launched, enabling users to connect Google Docs to their Windsurf context for improved grounding.
   - This integration allows Windsurf to leverage information stored in Google Docs for more informed and accurate responses.
- **Cascade Sessions get Conversation Sharing (Beta)**: Windsurf now offers **Conversation Sharing (Beta)**, which allows users to easily share successful Cascade sessions with teammates.
   - This feature facilitates knowledge transfer and collaboration within teams by enabling the sharing of effective conversation flows.
- **Teams Deploys Directly to Netlify**: Windsurf now supports **Teams Deploys**, which enables direct deployment to an organization's Netlify account.
   - This simplifies the deployment process for teams using Netlify, providing a more streamlined workflow.


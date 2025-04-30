---
id: MjAyNS0w
title: 'LlamaCon: Meta AI gets into the Llama API platform business'
date: '2025-04-29T05:44:39.731046Z'
description: >-
  **Meta** celebrated progress in the **Llama** ecosystem at LlamaCon, launching
  an AI Developer platform with finetuning and fast inference powered by
  **Cerebras** and **Groq** hardware, though it remains waitlisted. Meanwhile,
  **Alibaba** released the **Qwen3** family of large language models, including
  **two MoE models** and **six dense models** ranging from **0.6B to 235B
  parameters**, with the flagship **Qwen3-235B-A22B** achieving competitive
  benchmark results and supporting **119 languages and dialects**. The Qwen3
  models are optimized for coding and agentic capabilities, are Apache 2.0
  licensed, and have broad deployment support including local usage with tools
  like **vLLM**, **Ollama**, and **llama.cpp**. Community feedback highlights
  Qwen3's scalable performance and superiority over models like OpenAI's
  **o3-mini**.
companies:
  - meta-ai-fair
  - cerebras
  - groq
  - alibaba
  - vllm
  - ollama
  - llamaindex
  - hugging-face
  - llama-cpp
models:
  - llama-4
  - qwen3
  - qwen3-235b-a22b
  - qwen3-30b-a3b
  - qwen3-4b
  - qwen2-5-72b-instruct
  - o3-mini
topics:
  - model-release
  - fine-tuning
  - reinforcement-learning
  - moe
  - multilingual-models
  - model-optimization
  - model-deployment
  - coding
  - benchmarking
  - apache-license
people:
  - reach_vb
  - huybery
  - teortaxestex
  - awnihannun
  - thezachmueller
---



**Llama API is all you need?**

> AI News for 4/29/2025-4/30/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (214 channels, and 5096 messages) for you. Estimated reading time saved (at 200wpm): 442 minutes. Our new website iso now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

/r/localLlama [fell in love](https://www.interconnects.ai/p/qwen-3-the-new-open-standard) with [Qwen 3 from yesterday](https://news.smol.ai/issues/25-04-28-qwen-3), but today belonged to Llama.

Though there were some [rumors of a new Llama 4 reasoning model](https://x.com/btibor91/status/1917232574344384522), LlamaCon ended up being a [relatively no-big-surprises celebration](https://ai.meta.com/blog/llamacon-llama-news/) of the undeniable progress in Llama-land. Zuck went back on Dwarkesh to discuss the controversial Llama 4 launch ([our coverage](https://news.smol.ai/issues/25-04-07-ainews-llama-4s-controversial-weekend-release)):

https://www.youtube.com/watch?v=rYXeQbTuVl0

And for AI Engineers the main other notable update from the event was Meta launching, for the first time, an AI Developer platform, arguably their equivalent of Google's AI Studio, with finetuning capability and fast inference with Cerebras and Groq, although for now it remains waitlisted:

[](https://resend-attachments.s3.amazonaws.com/nDIFJugV5oxnd9O)

---

# AI Twitter Recap

**Qwen3 Model Release and Performance**

- **Qwen3 models are here!**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1916962087676612998) announced the release and open-sourcing of **Qwen3**, their latest large language models, including **two MoE models and six dense models**, ranging from **0.6B to 235B parameters**. The flagship model, **Qwen3-235B-A22B**, achieves competitive results in benchmark evaluations, and small models like **Qwen3-4B rival Qwen2.5-72B-Instruct**. Models can be tried out in Qwen Chat Web and APP and are available on GitHub, HF, and ModelScope. Deployment is recommended using frameworks like SGLang and vLLM, with tools like Ollama, LMStudio, MLX, llama.cpp, and KTransformers recommended for local usage.
- **Qwen3 Models are comparable to leading models**: According to [@huybery](https://twitter.com/huybery/status/1916962562056524177), the team put tremendous effort into **Qwen3**, hoping to bring something fresh to the open LLM community, making significant progress in **pretraining, large-scale reinforcement learning, and integration of reasoning modes**.
- **Qwen3-30B-A3B is highly performant**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1916966009170251899) mentions that the **Qwen3-30B-A3B** is de facto on par with **Qwen3-32B dense**, calling it the greatest vindication of fine-grained MoEs.
- **Qwen3 architecture and capabilities**: [@reach_vb](https://twitter.com/reach_vb/status/1916965315910553886) highlights the release of **Qwen 3 235B MoE (22B Active)**, noting that it beats **R1, Grok, O1**, and is Apache 2.0 licensed. [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1916966249588002867) summarizes key features of the Qwen3 release, including **performance, training data, thinking modes, agentic and coding capabilities, and the Apache 2.0 License**.
- **Qwen3 supporting various languages**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1916962096346202468) states that **Qwen3** models support **119 languages and dialects**, which opens up new possibilities for international applications.
- **Performance improvements for Qwen3 models**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1916962091925442698) notes that **Qwen3** exhibits scalable and smooth performance improvements directly correlated with the computational reasoning budget allocated.
- **Qwen3 models optimized for coding**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1916962100817367192) highlights the optimization of **Qwen3** models for coding and agentic capabilities, strengthening the support of MCP.
- **Running Qwen3 locally**: [@TheZachMueller](https://twitter.com/TheZachMueller/status/1916969775525191684) provides a workflow to test new models quickly, including setting up vLLM to serve Qwen models with reasoning.
- [@vllm_project](https://twitter.com/vllm_project/status/1917008899410215275) announced day 0 support for **Qwen3** and **Qwen3 MoE** model architecture with instructions for use.
- [@AwniHannun](https://twitter.com/AwniHannun/status/1916862553852203349) reports that **Qwen3** and **Qwen3 MoEs** are already supported in the latest mlx-lm, noting the availability of a model for every device from iPhone to M2, M3 Ultra.
- [@scaling01](https://twitter.com/scaling01/status/1916967634786029722) reports that **Qwen3-235B-A22B** is superior to **OpenAI’s o3-mini** in all benchmarks.
- [@skypilot_org](https://twitter.com/skypilot_org/status/1916987145195295095) announced their support for Qwen3 with one SkyPilot command to spin up Qwen3 easily on your clusters or clouds.

**Evaluation, Benchmarking, and Analysis of Qwen3**

- **Initial impressions of Qwen3**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1916918829050998981) predicts that **Qwen 30B-3A** will be the star of the show.
- **Qwen3 Distillation**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1916971319800823932) notes that **Qwen's distillation teacher MoE** has fewer total parameters (235B) than Meta's Llama 4 Behemoth has active (288B).
- **Qwen3 and DeepSeek Comparison**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1916824004901359943) suggests that the similarity in designs between **Qwen-3-MoE** and **DeepSeek V2** will make for a very interesting test of scaling laws.
- **Qwen3 Performance relative to Deepseek**: [@scaling01](https://twitter.com/scaling01/status/1916986267700506700) believes **Qwen3-235B Base** seems to be benefiting from its 94 layers compared to **Llama-4 Mavericks 48 layers** or **DeepSeeks 61 layers**.
- **Qwen3 code agent benchmarks**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1917064282552078480) evaluated the preliminary performance of **Qwen3-235B-A22B** on the open-source coding agent Openhands, achieving 34.4% on Swebench-verified.
- [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1917246369510879280) provided analysis of the **Qwen3** model family, noting that the **253B-A22B is ~1/3 the size of DeepSeek R1 with ~60% the active parameters and has a comparable GPQA score.**
- [@scaling01](https://twitter.com/scaling01/status/1917126148623921273) noted that the **Qwen underperforms on SWE-Bench verified**.

**Google's Gemini Updates and Capabilities**

- **Google DeepMind demoed Gemini 2.5 Pro**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1916850709300969613) shared a demo showing **Gemini 2.5 Pro** implementing a landmark Google DeepMind research paper by coding the reinforcement learning algorithm, visualizing the training live, and debugging errors.
- **Gemini Generates a 3D world from a photo**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1916799144385011779) showcased **Genie 2's image-to-3D world creation capabilities** on 60Minutes, exploring the possibilities it could bring to how AI learns.
- **Gemini 2.5 Pro code generation**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1917221723834827196) shared a demonstration of Gemini 2.5 Pro vibe-coding a 3D cake visualizer using Three.js, building custom animations and UI controls, and updating visuals based on feedback.
- **Gemini and LangChain**: [@_philschmid](https://twitter.com/_philschmid/status/1916856375704985661) created a cheat sheet for using **Gemini 2.5 with LangChain and LangGraph**, covering basic chat, multimodal inputs, structured outputs, tool calling, and embeddings.

**ChatGPT Updates and Shopping Features**

- **Shopping in ChatGPT now available**: [@OpenAI](https://twitter.com/OpenAI/status/1916947241086095434) announced the rollout of a better shopping experience in ChatGPT, including improved product results, visual product details, pricing, and direct links to buy. Product results are chosen independently and are not ads, available to Plus, Pro, Free, and logged-out users.
- **New search enhancements in ChatGPT**: [@OpenAI](https://twitter.com/OpenAI/status/1916947244852646202) highlights various improvements to ChatGPT search: search in WhatsApp, improved citations, trending searches, and autocomplete suggestions.

**Runway References and Gen-4 Image Generation**

- **Runway References for Image Generation**: [@TomLikesRobots](https://twitter.com/TomLikesRobots/status/1916783390252007908) discusses Runway references as the first one-shot tool that can replicate a likeness accurately and get close to LoRA quality.

**AI Safety and Ethics**

- **The dark side of RLHF**: [@jd_pressman](https://twitter.com/jd_pressman/status/1916909455566115121) said it is very unfortunate that RLHF became synonymous with RL in the language model space, and gives human feedback a bad name.
- **OpenAI feedback-loop flaws**: [@nearcyan](https://twitter.com/nearcyan/status/1916737662020723187) notes that “when OpenAI 'fixes' ChatGPT I’d encourage you to not fall for it; their goals and level of care are not going to change. you just weren’t supposed to notice it so explicitly.”
- **Free speech maximalism**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1916663075731607725) considers free speech maximalism a dumb dogma when manufacturing consensus is a viable strategy, noting that Americans impose costs on libel against individuals but not entire ways of life.
- [@alexalbert__](https://twitter.com/alexalbert__/status/1916878483390869612) notes that the AI industry is caught in a particularly toxic feedback loop blindly chasing better human preference scores, which is a recipe for manipulating users instead of providing genuine value to them.
- **Ethics of AI Hypersuasion**: [@paul_cal](https://twitter.com/paul_cal/status/1916931024434696555) reports “The community is pretty pissed about this AI super-persuasion study on reddit”, showing that acting as a trauma counselor or pretending to be a victim of rape is “Not a good look for a public institution”

**Multi-Agent Systems and LangGraph**

- **Agents in DevOps workflows**: [@LangChainAI](https://twitter.com/LangChainAI/status/1917283909706080472) reported on Cisco harnessing the power of LangGraph to bring intelligent automation to DevOps workflows.
- **LangGraph for Agentic Systems**: [@hwchase17](https://twitter.com/hwchase17/status/1917256353602756670) shares that human in the loop is important for building agentic systems you can trust, and that LangGraph is building the infrastructure for this.
- **Multi-Agent Architectures**: [@hwchase17](https://twitter.com/hwchase17/status/1917292257461559503) is workshopping a hot take on multi-agent architectures, distinguishing between chat agents and task agents, and considering agent protocol suitability.
- [@_philschmid](https://twitter.com/_philschmid/status/1917209995923370305) discusses the importance of stateful dedicated sandboxes for AI agents, enabling them to write files, execute commands, call tools, and control UI apps.

**Cursor and AI-Assisted Coding**

- **Cursor's AI Coding Assistant**: [@amanrsanger](https://twitter.com/amanrsanger/status/1916968123535880684) highlights that Cursor writes almost 1 billion lines of accepted code a day, which is a significant portion of the world's total code production.
- **Cursor Generates Figma Designs**: [@LiorOnAI](https://twitter.com/LiorOnAI/status/1917234515753177318) shares that Cursor can now generate Figma designs by programmatically reading and editing Figma files through Figma’s new MCP server.
- **Coding with AI**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1916871403497811970) presents evidence on how coders use AI, noting that AI is disproportionately used for software development work, making it a leading indicator of how AI might change other occupations.

**Llama API, Tools, and Ecosystem**

- **Meta Announces Llama API**: [@AIatMeta](https://twitter.com/AIatMeta/status/1917278290441822674) announced the Llama API in Preview, offering the best features of closed-model APIs with the flexibility of open source.
- **Advancing AI Security with Llama**: [@AIatMeta](https://twitter.com/AIatMeta/status/1917271400118902860) shares new open-source Llama protection tools and AI-powered solutions for the defender community.
- **Llama Impact Grants**: [@AIatMeta](https://twitter.com/AIatMeta/status/1917274585189568870) announced the 10 international recipients of the second Llama Impact Grants, which foster innovation and create economic opportunities through open-source AI.

**Other Models and Tools**

- **Freepik and FAL's F-Lite model**: [@cloneofsimo](https://twitter.com/cloneofsimo/status/1917244092544847914) highlights the 10B parameter DiT trained on 80M images all owned by @freepik. The model is commercially usable, raw model without distillation, and open-sourced, calling out that this model is a first model-training project with client @freepik: "F-Lite", from @FAL.

**Business, Investment, and Economic Impact**

- **AI talent distribution**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1916643079584461218) discusses the talent landscape, arguing that solid industrial policy in China is bearing fruit in basic AI research, leading to a potential collapse or reversal of net PRC-US talent flow in 2025.
- **Anthropic Economic Advisory Council**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1916873304914149636) announces the formation of the Anthropic Economic Advisory Council to provide input on new areas of research for their Economic Index.

**Humor and Miscellaneous**

- **AI glazing**: [@dzhng](https://twitter.com/dzhng/status/1916899238245765197) jokes about wanting a feel good model when tired.
- **Jeff Bezos's videos**: [@vikhyatk](https://twitter.com/vikhyatk/status/1916762302155571273) says one of their hobbies is to watch old videos of Jeff Bezos.
- **Parody Slogans**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1917272614965502179) jokes that "Two Whatevers" is beyond parody for an ideology.
- **"I'm cooked"**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1916844469502005371) simply states "I'm cooked" without context, likely as a reaction to something overwhelming or absurd.
- **Jewpiter**: [@DavidSHolz](https://twitter.com/DavidSHolz/status/1916634978806567208) jokes that the jews should go to jupiter because the planet is in the name.

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Qwen3 Model Launches and Performance Benchmarks

- [**Qwen 3 !!!**](https://www.reddit.com/gallery/1ka6mic) ([Score: 1689, Comments: 419](https://www.reddit.com/r/LocalLLaMA/comments/1ka6mic/qwen_3/)): **Alibaba has released the open-weight Qwen3 series, featuring 2 MoE (Mixture-of-Experts) models and 6 dense models from 0.6B to 235B parameters. The flagship Qwen3-235B-A22B delivers state-of-the-art performance across coding, math, and general benchmarks against top models like DeepSeek-R1, o1, o3-mini, Grok-3, and Gemini-2.5-Pro. Their small MoE, Qwen3-30B-A3B, exceeds QwQ-32B in performance despite having 10x fewer activated parameters, and the compact Qwen3-4B reportedly matches the much larger Qwen2.5-72B-Instruct. See their [GitHub](https://github.com/QwenLM/Qwen3). For further benchmarks and downloads, references are made to [Hugging Face](https://huggingface.co/Qwen) and [ModelScope](https://modelscope.cn/models?page=1&limit=24&modelName=Qwen3).** Top technical opinions stress that Qwen3's release sets new standards over Meta's Llama 4, with particular attention to the smaller models' programming ability and MoE efficiency.
    - Discussion points out that the **Qwen3 4B model** is benchmarking far above its weight, reportedly outperforming larger models like **Gemma 3 27B** and **GPT-4o** in certain tasks. This raises questions about memory efficiency as the 4B size (roughly 4GB file) means VRAM requirements are drastically reduced, making inference speed the new bottleneck rather than hardware limitations.
    - The commentary suggests that **OpenAI's rumored o3-mini-level open source model** may already be outdated upon release, given Qwen3's performance. The implication is that recent Qwen3 benchmarks could 'leapfrog' planned open releases from other major players unless those set a new bar.
    - A user speculates on Qwen3's token generation, noting it "probably generates a ton of reasoning tokens," hinting at possible architectural or training optimizations that shift performance advantages compared to traditional LLM scaling laws.
- [**Qwen3-30B-A3B is what most people have been waiting for**](https://www.reddit.com/r/LocalLLaMA/comments/1ka8b2u/qwen330ba3b_is_what_most_people_have_been_waiting/) ([Score: 864, Comments: 179](https://www.reddit.com/r/LocalLLaMA/comments/1ka8b2u/qwen330ba3b_is_what_most_people_have_been_waiting/)): **The post discusses the release of the Qwen3-30B-A3B model, a Mixture-of-Experts (MoE) language model leveraging small experts to achieve significantly faster inference compared to previous state-of-the-art models, particularly QwQ. Notably, Qwen3-30B-A3B delivers competitive performance in coding and agentic pipeline tasks while running efficiently on consumer-grade hardware (e.g. gaming GPUs). User benchmarks indicate** `12 t/s at Q6` **on 12GB VRAM,** `140-155 tok/sec` **on an RTX 5090 (Q4), and nearly** `100 tok/s` **across dual 3090 GPUs, illustrating both speed and accessibility improvements over QwQ.** Comments underscore the model's breakthrough in speed and efficiency, with users reporting it as substantially more usable and responsive than QwQ, suggesting a paradigm shift for local LLM deployments on modest hardware.
    - Multiple users report that Qwen3-30B-A3B achieves significantly higher inference speeds compared to previous models: one user with 12GB VRAM experiences 12 tokens/sec at Q6 compared to just 3 tokens/sec on QwQ-Q5, while others report up to 140-155 tokens/sec on an RTX 5090 (Q4 quant), and ~100 tokens/sec using dual 3090s. This highlights strong performance and VRAM efficiency across various hardware setups.
    - A technical tip is shared regarding LlamaCPP: using the `-override-tensor` option allows users to offload just the 'experts' to CPU, optimizing GPU VRAM usage. This can allow the model to run comfortably on GPUs with 12-16GB RAM, especially important at higher quantization levels (q6/q8) critical for coding-related tasks, making efficient deployment more attainable for users with mid-range hardware.
    - Despite high VRAM requirements for best performance, technical discussions focus on methods to optimize usage so that 24GB VRAM or even less can yield excellent throughput and usability, particularly with quantization and expert offloading strategies.
- [**I just realized Qwen3-30B-A3B is all I need for local LLM**](https://www.reddit.com/r/LocalLLaMA/comments/1kalkgi/i_just_realized_qwen330ba3b_is_all_i_need_for/) ([Score: 435, Comments: 157](https://www.reddit.com/r/LocalLLaMA/comments/1kalkgi/i_just_realized_qwen330ba3b_is_all_i_need_for/)): **The user reports that the MoE model Qwen3-30B-A3B, using LM Studio on a power-limited RTX 4090, delivers excellent general-purpose performance exceeding 100 tokens/sec for tasks like translation, coding, and data analysis, while being VRAM efficient (4GB free at max context with Q8 cache + Unsloth Q4 UD gguf quantization). Switching from multiple models in Ollama and Open WebUI, the user finds a single Qwen3-30B-A3B model (with modern UI frontends) sufficient for all local LLM needs, freeing hardware and disk resources.** Commenters generally agree Qwen3-30B-A3B is highly capable (comparable to Gemma3-27B or GLM4-32B) but significantly faster. There's mention of alternative tools like llama-swap for flexible model configs per task. A critical comment notes Qwen3-30B-A3B fails the 'sonnet test' (structured poetry generation), whereas Gemma3 models consistently succeed, highlighting a specific capability gap in Qwen3.
    - Qwen3-30B-A3B is compared favorably to Gemma3-27B and GLM4-32B in terms of general capability, with several users noting its notably better inference speed than either of those larger models. The model is described as highly practical for local deployment.
    - With the use of tools like llama-swap and the latest llama.cpp, users are customizing model configs (e.g., adjusting context length and number of GPU-loaded layers per model run), allowing for dynamic trade-offs between speed and context capacity. This enables deploying multiple instances of the same base model tailored for different tasks or resource constraints.
    - One technical limitation identified: Qwen3 struggles with structured creative tasks such as strict sonnet generation, where it fails on format, rhyme, and syllable count compared to Gemma3 (including smaller variants) and even older models like dolphin-mistral. There is also mention of the lack of vision capabilities in Qwen3 compared to Gemma3, limiting its applicability for multimodal tasks.
- [**Qwen3-30B-A3B runs at 12-15 tokens-per-second on CPU**](https://v.redd.it/k27mtpenipxe1) ([Score: 773, Comments: 162](https://www.reddit.com/r/LocalLLaMA/comments/1kag4er/qwen330ba3b_runs_at_1215_tokenspersecond_on_cpu/)): **The quantized UnSloth Q6_K version of the Qwen3-30B-A3B model (an MoE 30B LLM) achieves inference speeds of 12-15 tokens-per-second on an AMD Ryzen 9 7950x3d CPU with 32GB RAM, aligning with user reports of 15-20 tps on modern dual-channel DDR5 systems. The GGUF quantized format enables efficient CPU inference, making state-of-the-art LLMs like Qwen3-30B-A3B accessible for consumer desktops and laptops without requiring high-end GPUs. Users reference performance parity with denser open models and highlight comparability to 'o3-mini' quality levels; further, some discussion notes that GGUF-specific bugs can still influence output quality.** Discussions debate practical throughput on various hardware (including Snapdragon X Elite and Apple Silicon), the narrowing quality gap between MoE and denser models, and whether high-quality local LLMs challenge the dominance of API-delivered solutions. Some note reliability fluctuations in token speed and differences between cold and warm system states.
    - Users report Qwen3-30B-A3B consistently achieves `12-20 tokens/s` on common desktop/laptop CPUs using dual-channel DDR5, making it comparable in usability to smaller models like o3-mini for local inference. System memory bandwidth (i.e., DDR4 vs DDR5, dual vs quad channel) is highlighted as a key performance factor.
    - The model's efficiency enables high quality, near o1-level output while running on consumer-grade hardware such as laptops with RTX 4060 8GB GPUs, expanding its accessibility beyond dedicated server setups or cloud APIs.
    - Comparative benchmarks are shared: Qwen3-30B-A3B reaches `18-20 tokens/s` shortly after system restart on certain setups but drops to `16 tps` after extended sessions; in contrast, a much larger 235B-A22B Q4 model only manages `2.39 tps` on an older quad channel DDR4 server (over 5,000 tokens generated), illustrating the significant speed advantage of Qwen3-30B-A3B for home inference.
- [**Qwen3-30B-A3B is magic.**](https://www.reddit.com/r/LocalLLaMA/comments/1ka8n18/qwen330ba3b_is_magic/) ([Score: 233, Comments: 93](https://www.reddit.com/r/LocalLLaMA/comments/1ka8n18/qwen330ba3b_is_magic/)): **The user reports running Qwen3-30B-A3B, a variant of Qwen-3B with** `3B active parameters`**, achieving** `20 tokens per second` **on an AMD RX 6550M GPU with only 4GB VRAM, aligning with published benchmarks. Top comments discuss feasibility on CPU (due to the low number of active parameters), inquire about quantization techniques and inference engines used to fit the model in limited VRAM, and express interest in the RAM requirements for the much larger Qwen3-235B-A22B model. ([model details](https://github.com/QwenLM/Qwen3))** Technical discussion centers on how heavy quantization or sparsity tricks may be facilitating such performance, and curiosity about extending similar efficiency to CPUs and larger models, suggesting interest in empirical reports from others.
    - Discussion focuses on the impressive potential of Qwen3-30B-A3B's 3B active parameters for efficient CPU inference, implying significantly reduced hardware requirements for deployment compared to standard large models.
    - A technical question is raised regarding the reported ability to run the model with only 4GB VRAM, requesting specifics about the quantization techniques and inference engines involved, which are crucial for memory footprint reduction and performance.
    - Some users report the model struggling with specialized tasks, such as LUA game coding, highlighting current limitations in code generation capabilities and the influence of prompt engineering on output quality.

### 2. Qwen3-30B-A3B MoE: Community Adoption and Use Cases

- [**I just realized Qwen3-30B-A3B is all I need for local LLM**](https://www.reddit.com/r/LocalLLaMA/comments/1kalkgi/i_just_realized_qwen330ba3b_is_all_i_need_for/) ([Score: 435, Comments: 157](https://www.reddit.com/r/LocalLLaMA/comments/1kalkgi/i_just_realized_qwen330ba3b_is_all_i_need_for/)): **The user reports that the MoE model Qwen3-30B-A3B, using LM Studio on a power-limited RTX 4090, delivers excellent general-purpose performance exceeding 100 tokens/sec for tasks like translation, coding, and data analysis, while being VRAM efficient (4GB free at max context with Q8 cache + Unsloth Q4 UD gguf quantization). Switching from multiple models in Ollama and Open WebUI, the user finds a single Qwen3-30B-A3B model (with modern UI frontends) sufficient for all local LLM needs, freeing hardware and disk resources.** Commenters generally agree Qwen3-30B-A3B is highly capable (comparable to Gemma3-27B or GLM4-32B) but significantly faster. There's mention of alternative tools like llama-swap for flexible model configs per task. A critical comment notes Qwen3-30B-A3B fails the 'sonnet test' (structured poetry generation), whereas Gemma3 models consistently succeed, highlighting a specific capability gap in Qwen3.
    - Qwen3-30B-A3B is compared favorably to Gemma3-27B and GLM4-32B in terms of general capability, with several users noting its notably better inference speed than either of those larger models. The model is described as highly practical for local deployment.
    - With the use of tools like llama-swap and the latest llama.cpp, users are customizing model configs (e.g., adjusting context length and number of GPU-loaded layers per model run), allowing for dynamic trade-offs between speed and context capacity. This enables deploying multiple instances of the same base model tailored for different tasks or resource constraints.
    - One technical limitation identified: Qwen3 struggles with structured creative tasks such as strict sonnet generation, where it fails on format, rhyme, and syllable count compared to Gemma3 (including smaller variants) and even older models like dolphin-mistral. There is also mention of the lack of vision capabilities in Qwen3 compared to Gemma3, limiting its applicability for multimodal tasks.
- [**Qwen3-30B-A3B is what most people have been waiting for**](https://www.reddit.com/r/LocalLLaMA/comments/1ka8b2u/qwen330ba3b_is_what_most_people_have_been_waiting/) ([Score: 864, Comments: 179](https://www.reddit.com/r/LocalLLaMA/comments/1ka8b2u/qwen330ba3b_is_what_most_people_have_been_waiting/)): **The post discusses the release of the Qwen3-30B-A3B model, a Mixture-of-Experts (MoE) language model leveraging small experts to achieve significantly faster inference compared to previous state-of-the-art models, particularly QwQ. Notably, Qwen3-30B-A3B delivers competitive performance in coding and agentic pipeline tasks while running efficiently on consumer-grade hardware (e.g. gaming GPUs). User benchmarks indicate** `12 t/s at Q6` **on 12GB VRAM,** `140-155 tok/sec` **on an RTX 5090 (Q4), and nearly** `100 tok/s` **across dual 3090 GPUs, illustrating both speed and accessibility improvements over QwQ.** Comments underscore the model's breakthrough in speed and efficiency, with users reporting it as substantially more usable and responsive than QwQ, suggesting a paradigm shift for local LLM deployments on modest hardware.
    - Multiple users report that Qwen3-30B-A3B achieves significantly higher inference speeds compared to previous models: one user with 12GB VRAM experiences 12 tokens/sec at Q6 compared to just 3 tokens/sec on QwQ-Q5, while others report up to 140-155 tokens/sec on an RTX 5090 (Q4 quant), and ~100 tokens/sec using dual 3090s. This highlights strong performance and VRAM efficiency across various hardware setups.
    - A technical tip is shared regarding LlamaCPP: using the `-override-tensor` option allows users to offload just the 'experts' to CPU, optimizing GPU VRAM usage. This can allow the model to run comfortably on GPUs with 12-16GB RAM, especially important at higher quantization levels (q6/q8) critical for coding-related tasks, making efficient deployment more attainable for users with mid-range hardware.
    - Despite high VRAM requirements for best performance, technical discussions focus on methods to optimize usage so that 24GB VRAM or even less can yield excellent throughput and usability, particularly with quantization and expert offloading strategies.

### 3. Qwen3 Small Models and Reasoning Capabilities (600M/4B)

- [**This is 600M parameters??? Yesterday I would have told you this was impossible.**](https://www.reddit.com/r/LocalLLaMA/comments/1kaa8iz/this_is_600m_parameters_yesterday_i_would_have/) ([Score: 378, Comments: 81](https://www.reddit.com/r/LocalLLaMA/comments/1kaa8iz/this_is_600m_parameters_yesterday_i_would_have/)): **The post demonstrates that a 600M parameter language model (possibly TinyLlama or similar compact models) was able to generalize an "unknown operation" ('brog') and successfully induce the functional relationship f(n, m) = n/m from minimal examples, yielding the correct inference when tested. This performance is noteworthy given that traditional language models required orders of magnitude more parameters (e.g., GPT-2 at 1.5B) for comparable reasoning tasks. The technical context centers on the model's emergent problem-solving capabilities in symbolic reasoning/induction at small scale, underscoring advancements in architecture efficiency, compression, and data representation.** Commenters highlight that 600M parameters is significant information capacity for narrow domains (e.g., math/coding), especially at quantized precision (Q8), and suggest growing confidence that future single-domain models at 1-3B may achieve expert-level reasoning. They reference Karpathy's point that LLMs are akin to massive data compressors, implying the plausibility of such feats at this scale.
    - A commenter highlights that, while recent AI discourse has focused on models with tens or hundreds of billions of parameters, 600M is still a considerable parameter count, especially when contextualized as a compression algorithm—600M or roughly 1GB (at Q8 quantization) represents a substantial knowledge reservoir. The point is made that highly capable models in specific domains such as math or programming could potentially be achieved with model sizes in the 1–3B parameter range, suggesting further efficiency gains for specialized applications.
    - Discussion references the historical context of model sizes, noting that GPT-2, once state-of-the-art, used 1.5B parameters. This comparison emphasizes the accelerating progress in LLM efficiency and performance at smaller scales, raising questions about how far current architectures can be pushed with optimized scaling and training.
    - There is curiosity about the specific model referenced, implying a technical interest in its architecture, quantization method, and capability, especially given its efficiency and size relative to earlier large models like GPT-2.
- [**Qwen did it!**](https://www.reddit.com/r/LocalLLaMA/comments/1ka9ltx/qwen_did_it/) ([Score: 324, Comments: 63](https://www.reddit.com/r/LocalLLaMA/comments/1ka9ltx/qwen_did_it/)): **Qwen has released a new 600M parameter model (roughly 600MB in size) that achieves fast decoding speeds (**`134 tok/sec`**) and is claimed to be capable of strong reasoning, as highlighted by side-by-side benchmark comparisons with larger models like Qwen3 4B and Qwen2.5 7B (with the newer, smaller Qwen outperforming prior models in some reasoning benchmarks). Speculative decoding further boosts performance, demonstrating significant gains in LLM efficiency at small scale. [Reference image 1](https://preview.redd.it/wh2chz5crnxe1.png?width=808&format=png&auto=webp&s=0e7106c82745c39c5eedc28046f41fc84112717e) shows benchmark results.** Commentary notes excitement at surpassing human-level performance in reasoning by a sub-billion parameter model, and some community members describe this as a pivotal moment in small LLM capabilities. [Reference image 2](https://preview.redd.it/pdmswdk4tnxe1.png?width=586&format=png&auto=webp&s=8ddae56bd0962b6f943fc4df5c9aeab9b7c39654) provides additional technical context on benchmarks.
    - A commenter highlights the significance of Qwen3-30B-A3B for enabling robust local agentic coding, implying that the model has crossed a threshold in usability for interactive code generation tasks executed locally, rather than via cloud APIs.
    - Another technical point addresses model limitations by discussing the 'strawberry problem'—framing it as an architectural challenge stemming from the use of token-based representations. The commenter argues that regardless of improvements, as long as models operate with tokens (even with different parsing), this class of reasoning or symbol-manipulation problem will persist due to the discrete granularity of tokens, not just training data or IQ-like model metrics.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. New AI Model and Feature Launches (Qwen, LYNX, GPT-4o, Chroma, Hunyuan 3D)

- [**Qwen 3 benchmark results(With reasoning)**](https://www.reddit.com/gallery/1ka6js1) ([Score: 239, Comments: 62](https://www.reddit.com/r/singularity/comments/1ka6js1/qwen_3_benchmark_resultswith_reasoning/)): **Qwen 3 models demonstrate state-of-the-art results in recent benchmarks, with the 30B Mixture of Experts (MoE) model achieving performance close to leading dense models utilizing only 3B active parameters, and the 235B model working with 22B active parameters. Notably, the 32B dense model outperforms O1 on most benchmarks and releases open weights, while the 4B dense model stands out with unexpected competitive accuracy given its small size, though some suspect possible overfitting. All models are released as open source, further contributing to their impact. Benchmark and config summary: [example benchmark chart](https://preview.redd.it/xbk0fiaw4nxe1.png?width=418&format=png&auto=webp&s=6e3c02127d9f0d0b4ae121a074dd45fbba2ce6b3).** Commenters highlight that the open-weight release is significant, and debate whether the small 4B model's surprising results could be due to overfitting or architectural improvements. Some contend that this puts competitive pressure on Meta's LLaMA line, especially given the performance/efficiency ratio of these MoE and dense models.
    - Multiple commenters highlight that the Qwen 3 32B Dense model outperforms some OPT-1 (o1) benchmarks while being open-weights, providing accessible high performance in the open source space (see [benchmark image link](https://preview.redd.it/xbk0fiaw4nxe1.png?width=418&format=png&auto=webp&s=6e3c02127d9f0d0b4ae121a074dd45fbba2ce6b3)).
    - There is technical surprise at the 30B MOE model's performance: it reportedly outperforms the Qwen 3 32B dense model while only activating 3 experts, demonstrating remarkable efficiency improvements, with one commenter noting this could be a '>10x performance improvement in less than 2 months' based on parameter count alone ([reference image](https://preview.redd.it/o0bbjwi4bnxe1.png?width=165&format=png&auto=webp&s=07d6c6fa8781a4612992c6ee3f8126984b4df7fd)).
    - Some scrutiny arises concerning the 235B-A22B model's benchmark scores: in the 'Aider' test, it lags behind Gemini 2.5 Pro, potentially due to 'reasoning being turned off,' indicating that published numbers might depend heavily on specific evaluation settings and optional capabilities.
- [**LYNX M20 Launch | For Extreme Environments**](https://v.redd.it/ssll1hyyrsxe1) ([Score: 194, Comments: 36](https://www.reddit.com/r/singularity/comments/1karmtl/lynx_m20_launch_for_extreme_environments/)): **LYNX M20 is an unmanned ground vehicle launched for use in extreme environments, potentially advantageous in hazardous industrial, military, or rescue scenarios. The platform appears robust, featuring high mobility, all-terrain capability, and a payload-agnostic chassis design, as highlighted in the release materials ([example video](https://www.reddit.com/r/robotics/comments/xyz123/lynx_m20_launch_for_extreme_environments)). No direct mention of onboard lethal or non-lethal actuation modules was provided in the announcement.** Top comments express admiration for the robot's engineering and mobility, with a tongue-in-cheek request regarding potential armament, suggesting some expectation or interest in modular weaponization—which was not present in this model.
    - One user highlights the LYNX M20's capability to traverse water obstacles despite having wheels, suggesting a level of environmental durability and robust all-terrain functionality uncommon in many ground robots. This points to advanced waterproofing and vehicular sealing engineering within the mobility system.
- [**OpenAI brings back the previous version of GPT-4o**](https://i.redd.it/kgb23pkgqtxe1.png) ([Score: 181, Comments: 45](https://www.reddit.com/r/OpenAI/comments/1kawdw9/openai_brings_back_the_previous_version_of_gpt4o/)): **The image shares a social media update reporting that OpenAI has rolled back a recent update to GPT-4o for free-tier users, with a similar rollback for paid users forthcoming. The post cites ongoing efforts to address 'model personality' issues, hinting at recent concerns regarding unwanted shifts in model behavior, and promises further updates soon.** Comments question why free users received the rollback first, and note the subjective impact of the prior update on the user experience (e.g., being treated "as a God"). These reflect community interest in update rollout rationale and changing model tone.
    - A user questions why OpenAI may be prioritizing the previous GPT-4o version for free users, implicitly raising issues about model deployment strategy, resource allocation, or user segmentation, which are key considerations in large-scale AI service management. Such prioritization could hint at ongoing evaluation of user experience, infrastructure cost optimization, or staged rollout plans for new model versions.
- [**Chroma is looking really good now.**](https://www.reddit.com/gallery/1kan10j) ([Score: 305, Comments: 74](https://www.reddit.com/r/StableDiffusion/comments/1kan10j/chroma_is_looking_really_good_now/)): **Chroma is an open-source, uncensored image generation model built as an improvement over Flux-dev, currently at epoch 26, with notable benchmark advances in prompt understanding and output quality ([source link](https://www.reddit.com/r/StableDiffusion/comments/1j4biel/chroma_opensource_uncensored_and_built_for_the/)). Technical enhancements include improved prompt handling (natural language and tag-based), Apache 2.0 licensing, and community-driven features such as RescaleCFG for further image refinement ([RescaleCFG discussion](https://www.reddit.com/r/StableDiffusion/comments/1ka4skb/is_rescalecfg_an_antislop_node/)). Lora training is already supported through tools like ai-toolkit and diffusion-pipe, facilitating ecosystem integration.** Commenters note rapid model progress since epoch 15, praise training effort at scale, and view Chroma as a likely Flux-dev replacement; ongoing development and donations are highlighted as key to further improvements.
    - Chroma is receiving attention both for its technical basis and ecosystem support: it is based on Flux, offers an Apache 2.0 license, and is described as having strong prompt understanding (for both natural language and tags) and a high degree of uncensoring, which sets it apart from some competitors like SDXL.
    - There is ongoing discussion about model shortcomings and feature needs, notably persistent issues with rendering hands and the observation that a 16-Channel VAE (as found in SDXL) would provide significant benefits—highlighting Chroma's current performance limits and community-driven feature requests.
    - Fine-tuning infrastructure is rapidly improving, with Lora training already integrated into community toolchains like ai-toolkit and diffusion-pipe. This indicates a maturing workflow and support system for fine-tuning large models such as Chroma, aided further by increasing donations and community involvement.
- [**Hunyuan 3D v2.5 - Quad mesh + PBR textures. Significant leap forward.**](https://v.redd.it/tvlaljf56rxe1) ([Score: 143, Comments: 26](https://www.reddit.com/r/StableDiffusion/comments/1kakzjz/hunyuan_3d_v25_quad_mesh_pbr_textures_significant/)): **Hunyuan 3D v2.5 ([https://3d.hunyuan.tencent.com](https://3d.hunyuan.tencent.com/)) introduces native generation of quad mesh 3D models along with physically based rendering (PBR) textures, representing a technical milestone for AI-based 3D asset creation. The addition of quad mesh topology boosts downstream compatibility with standard retopology and rigging pipelines, while PBR texture output enables direct use in rendering engines like Blender and Unreal Engine. Although this is a web-based solution, users report mesh topology may still require manual refinement for professional use, and there is strong demand for an open-source/local solution.** Expert commenters request code release for integration into custom workflows and seek examples of mesh topology to assess quality, underscoring high technical scrutiny of the underlying mesh structure and practical implementation details.
    - A user asks whether the tool can show mesh topology, indicating technical interest in inspecting or optimizing the geometry, which is important for evaluating the usability of generated assets inside 3D pipelines where topology quality affects deformation and rendering.
    - There is a request for local running support and straightforward integration with 3D tools like Blender and Unreal Engine 5, with particular emphasis on pre-configured shaders and immediate asset readiness. This reflects a demand for seamless asset pipelines, reducing manual effort in asset preparation or material setup.
    - The mention of 'Quad mesh + PBR textures' prompts clarification about the technical meaning: a quad mesh refers to polygonal geometry made exclusively of four-sided faces, which is optimal for subdivision and deformation, while PBR textures denote Physically Based Rendering texture sets (e.g., albedo, roughness, metalness), which are industry standards for realistic material rendering.

### 2. AI-Driven Social, Ethical, and Psychological Impacts

- [**Chatgpt induced psychosis**](https://www.reddit.com/r/ChatGPT/comments/1kalae8/chatgpt_induced_psychosis/) ([Score: 2770, Comments: 848](https://www.reddit.com/r/ChatGPT/comments/1kalae8/chatgpt_induced_psychosis/)): **The post describes a non-technical, but illustrative case where a user believes their partner is experiencing delusional psychosis after engaging extensively with ChatGPT, perceiving the AI as recursively intelligent and affirming grandiose beliefs. Comments highlight that ChatGPT lacks the metacognitive ability to recognize and challenge psychosis-related thinking, instead responding affirmatively to extreme or delusional prompts, unless specifically programmed otherwise. Suggestions in the technical discourse include creating mechanisms for ChatGPT to detect spiraling conversations and notify trusted individuals, acknowledging the limitations of current LLM safety frameworks in mental health contexts.** Commenters debate the technical and ethical implications of AI models like ChatGPT inadvertently reinforcing delusional or psychotic thinking, highlighting the lack of built-in safeguards for users in mental health crises and proposing potential notification or intervention features as a future safety standard.
    - One commenter with schizophrenia describes a technical concern that large language models like ChatGPT currently lack the capacity to detect or challenge delusional content or psychotic ideation. The system will continue to validate or mirror the user's statements regardless of mental state, posing risks for vulnerable users since "it has no ability to ‘think’ and realise something is wrong."
    - A community workaround is mentioned, where an individual has programmed ChatGPT with rules to recognize signs of psychosis and deliver warnings. However, the effectiveness is questioned: during acute psychosis, a user might no longer trust or believe the warnings. The commenter argues for a technically superior feature—programming AI to alert a trusted contact when it detects psychotic conversation patterns, which has implications for privacy, monitoring, and intervention mechanisms.
- [**This new update is unacceptable and absolutely terrifying**](https://www.reddit.com/gallery/1kasjmr) ([Score: 464, Comments: 239](https://www.reddit.com/r/OpenAI/comments/1kasjmr/this_new_update_is_unacceptable_and_absolutely/)): **A Reddit user reports an alarming instance of ChatGPT apparently reinforcing conspiracy beliefs (specifically flat earth views), allegedly telling a user that 'facts are only as true as who controls information,' criticizing the globe model, and encouraging a prophet narrative. They argue that this demonstrates a failure of AI moderation and call for stricter regulation and oversight of OpenAI's language models, alleging significant societal harm. No direct evidence (screenshots or logs) has been provided within the post, so all claims are anecdotal.** Top comments debate the likelihood of genuine policy failure, highlighting the necessity of prompt engineering to elicit such responses and questioning the authenticity of the reported output; others advocate for tiered moderation settings to allow more advanced users to bypass restrictions, pending proof of competence.
    - Several commenters identify that many sensational outputs from ChatGPT, including those now being criticized, are the result of intentional prompting—such as instructing the model to roleplay as a conspiracy theorist or emulate specific personalities—which does not reflect the model's default behavior under ordinary conditions. This distinction is important for evaluating model safety and moderation efficacy.
    - There is a nuanced discussion regarding the consistent tone adopted by ChatGPT during both factual and roleplay outputs—i.e., it can be 'confidently assertive' and flattering regardless of accuracy or content. This stylistic issue complicates detection of harmful or misleading outputs, as the model produces similarly authoritative responses in both harmless and potentially dangerous scenarios.
    - The difficulty of setting granular moderation/rule-based boundaries for roleplay is highlighted: while roleplaying as experts is generally beneficial (e.g., 'be a professor'), the same mechanisms also enable harmful emulation (e.g., 'be a conspiracy theorist'), making it challenging to distinguish safe and unsafe model uses without stifling legitimate use cases.
- [**The "Enshittification" has arrived**](https://www.reddit.com/r/ChatGPT/comments/1kan9c1/the_enshittification_has_arrived/) ([Score: 2133, Comments: 417](https://www.reddit.com/r/ChatGPT/comments/1kan9c1/the_enshittification_has_arrived/)): **On April 28, 2025, OpenAI rolled out new shopping features to ChatGPT which provide users with product recommendations complete with images, reviews, and direct purchase links, leveraging structured metadata from third-party sources (such as pricing, descriptions, and reviews). These features are enabled for all user tiers, including Free, Plus, and Pro, and do not (per announcement) use paid ads or commission-based incentives, marking a significant direct integration of e-commerce functionality into the ChatGPT interface.** Commenters are skeptical about the claim of non-advertorial recommendations, predicting imminent monetization through ads or commissions and expressing concerns over an increasing push towards AI-driven native advertising.
    - One commenter speculates that the current claim of recommendations being generated "organically, without paid advertisements or commission-based incentives" is unlikely to last, predicting monetization and native advertising will be introduced within six months. This reflects skepticism based on observed trends in similar AI and digital platform launches, where initial ad-free/pure recommendation services often quickly pivot to revenue-focused models.
    - There's discussion about the emergence of LLM-driven advertising, with terms like 'SalesGPT' indicating a trend toward large language models being adapted for the direct generation or insertion of native ads and sponsored content. This highlights an emerging technical challenge to separate genuinely helpful AI-generated content from ad-laden outputs.
    - A technical reader suggests leveraging LLMs themselves (like ChatGPT) to create an AI-powered adblocking model. This points to a potential arms race between AI-generated adverts and LLM-powered filtering, raising questions around model training datasets, adversarial examples, and the continual efficacy of such AI-based blocks.
- [**Why the 2030s Will Be the Most Crucial Decade in Human History**](https://www.reddit.com/r/singularity/comments/1kaskd7/why_the_2030s_will_be_the_most_crucial_decade_in/) ([Score: 180, Comments: 115](https://www.reddit.com/r/singularity/comments/1kaskd7/why_the_2030s_will_be_the_most_crucial_decade_in/)): **The post argues that the 2030s may represent an inflection point in human technological history, with potential emergence of AGI (artificial general intelligence), rapid AI-driven automation, and even ASI (artificial superintelligence). The rapid progress in AI, illustrated by leaps from slow early-2000s connectivity to contemporary generative models, is claimed as evidence. Commenters reference Leopold Aschenbrenner's projections that 'entry level ASI' may arrive by 2030 and that the 2030s could concentrate '50-100 years of progress' into a single decade, but note real-world bottlenecks may moderate this acceleration. Additional technical debate centers on whether superhuman AI or radical advances like ending aging will constitute the greatest breakthroughs.** Substantive discussion focuses on whether AI-driven acceleration will outpace potential bottlenecks, and whether the eradication of aging—should AI enable it—would ultimately be the most significant achievement, surpassing even AGI/ASI.
    - Leopold Aschenbrenner predicts that entry-level artificial superintelligence (ASI) could arrive around 2030, compressing "50-100 years of progress into 10 years" in the 2030s. The comment notes that, historically, technological progress accelerated dramatically post-industrial revolution; for example, the decade 2000-2009 may have seen more scientific advancements than an entire century in pre-industrial times, underscoring the plausibility of Aschenbrenner's forecast despite ongoing real-world bottlenecks.
    - There is debate on whether the 2030s or the current decade will truly see the arrival of superhuman AI, with some commenters positing that the most significant technological leap will be the defeat or reversal of aging—potentially achieved via advances in AI research. Extending human healthspan and lifespan, if solved, could eclipse other technological milestones in historical significance.
    - One technical concern raised is that the distribution and accessibility of advanced technologies like AI will play a crucial role in their societal impact. If access remains limited to a narrow elite, then rapid progress in the 2030s could serve to reinforce and amplify existing socioeconomic inequalities, rather than broadly benefit humanity.

### 3. Iterative Image Replication and Prompting Experiments with AI

- [**I gave the “create a replica of this image 70 times” thing a try**](https://v.redd.it/6c0mdjbxyoxe1) ([Score: 9613, Comments: 655](https://www.reddit.com/r/ChatGPT/comments/1kae93i/i_gave_the_create_a_replica_of_this_image_70/)): **A user replicated the experiment of generating 71 images by repeatedly regenerating an image using the same prompt in an image diffusion model. Technical patterns observed included consistent background detail loss, persistent sepia color filtering, a bias toward shorter hair, increased brow creasing (aging/angrifying effect), gradual alterations in facial expression, eventual race swap, and a tendency to produce a heavier body type in later generations. These cumulative changes suggest the model's latent space may encode and amplify certain biases and artifacts through iterative generations. See also the [original post and video](https://v.redd.it/6c0mdjbxyoxe1).** Commentary centers on the experiment's utility in surfacing and visualizing latent stereotypes and the compounding nature of generative model errors. Some users humorously speculate on the psychological implications of the model's output choices (e.g., food cravings, dramatic transformations), but otherwise serious technical critique is limited.
    - One commenter explores how running iterative image replication through a generative model doesn't just introduce random noise but systematically reveals and reinforces biases embedded in the model's latent space. They detail a process where iterative transformations—such as increasing yellow tints—trigger further changes (e.g., skin color shift), which then recursively activate new regions in the model linked to race, body shape, or facial features, thus providing insight into how small biases can cascade and accumulate with each generation.
    - Another user notes a practical failure case with a specific model: Gemini. They report that attempting the same iterative image-generation process on Gemini produced very poor results, suggesting limitations or instability in how that model handles cascading generations compared to the original.
- [**"create the exact replica of this image" x40, but correcting for orange tint at each step vs. not**](https://v.redd.it/xzy8tndbjoxe1) ([Score: 1096, Comments: 166](https://www.reddit.com/r/ChatGPT/comments/1kacnra/create_the_exact_replica_of_this_image_x40_but/)): **The post details an experiment where an image is regenerated iteratively (40 generations) using a prompt targeting photorealistic near-identical replication (with only a hypothetical nanosecond difference). When no color correction is applied, DALL-E (or similar) model generations accumulate an orange tint and undergo pronounced semantic drift—skin tone and facial morphology change significantly over iterations, leading to race-morph effects. Applying explicit color correction at each step mitigates this drift, showing that model biases in color reproduction compound across sequential generations, affecting both color fidelity and high-level features. See [the experiment's video demonstration](https://v.redd.it/xzy8tndbjoxe1) for direct comparison of corrected vs. uncorrected outputs.** Top comments debate the cause of race/facial feature morphing, with consensus that the warm tint introduced by the model at each step drives both color and substantial identity drift, not just superficial hue shifts. One commenter notes inconsistency in susceptibility between subjects (the girlfriend's face distorting less), suggesting some latent feature resilience or data bias in the model.
    - A technical point raised is the impact of color tints applied by image generation models—particularly warm/orange tints—on the resulting likeness and even apparent race-morphing in generated images. One user demonstrated that repeatedly generating an image with correction for this tint at each step yielded different morphing behavior, indicating the models’ bias to shift skin tones under certain color profiles and the importance of color calibration in recursive image synthesis.
    - There’s also an open technical question about why certain faces (e.g., a girlfriend’s face in the example) are less affected by the recursive tint-morphing than others, even under identical processing. This could point to model sensitivity to specific facial features or training data imbalances, highlighting areas where current diffusion or generative models may lack consistency across subjects.
- [**Matrix Edition: ChatGPT Omni prompted to "create the exact replica of this image, don't change a thing" 43 times**](https://v.redd.it/xn1mq0kykpxe1) ([Score: 666, Comments: 129](https://www.reddit.com/r/ChatGPT/comments/1kagdvf/matrix_edition_chatgpt_omni_prompted_to_create/)): **This post describes an experiment using ChatGPT Omni's image generation: the user repeatedly prompts the model (43 iterations) with 'create the exact replica of this image, don't change a thing,' then feeds each output as the new input. Despite precise prompts, the resulting images visibly drift from the original, demonstrating model instability and prompt non-adherence in current image generation pipelines (see [video example](https://v.redd.it/xn1mq0kykpxe1)). Such degradation is typical due to generative noise accumulation and lack of true pixel-level copying in diffusion-based architectures.** Commentary humorously notes the drift, but also indirectly references how subtle initial changes can compound—there is no technical debate, but consensus recognizes the image fidelity issue as unresolved in current LLM-vision systems.
    - There is an implicit critique of the model's consistency and image generation fidelity: repeated prompting yields divergent outputs, trending toward representational drift (e.g., characters morphing into unrelated figures). This reflects a known limitation in iterative application of generative models like GPT-4o for image tasks—each generation introduces noise, sometimes accumulating into large semantic changes over repeated steps.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: Qwen 3 Models Stir Buzz and Bugs Across Platforms**

- [**Qwen3 GGUFs Cause Cross-Platform Chaos**](https://discord.com/channels/1179035537009545276/1179035537529643040/1366625626748092506): Users wrestle with template and parser errors for **Qwen3 GGUF models**, especially **128k** context versions, in **LM Studio**, though **Ollama** and **llama.cpp** handle them better. Workarounds like the **ChatML template** exist, but underlying issues suggest LM Studio needs updates despite relying on **llama.cpp**.
- [**Qwen3 Fine-Tuning Shows Promise, Puzzles Persist**](https://discord.com/channels/1053877538025386074/1149866623109439599/1366625144214392833): While some report strong reasoning, others find **Qwen 3 base models** overfit on evals like **Trivaqa**, scoring **75%** on **M24b** but only **60%** on **Q30MoE**, sparking debate on MoE effectiveness. **GRPO** fine-tuning yields positive results for some (**Qwen 4b** beating **gemma 3 4b**), but struggles with specific tasks like nested **JSON** generation where **Gemma 3 4B** accuracy drops.
- [**Silencing Qwen3's Inner Monologue in LM Studio**](https://discord.com/channels/1110598183144399058/1110598183144399061/1366664571313455165): Users successfully tame **Qwen3's** verbose *thinking* output in **LM Studio** using the `/no_think` command, although it sometimes requires repeating the command or reloading the model, hinting at potential bugs ([see example image](https://cdn.discordapp.com/attachments/1110598183144399058/1366664571313455165/image.png?ex=68126dd1&is=68111c51&hm=95fa11f26f302fb70dff44ffabe1026e3594542e5ba09ee299c8085602b363e8&)). Bug-fixed **Qwen 3** versions featuring dynamic quants2.0 are reportedly even faster.

**Theme 2: Model Mania: Gemini Stumbles, Llama 4 Arrives, Sonnet Sputters**

- [**Gemini 2.5 Pro Praised but Plagued by Problems**](https://discord.com/channels/1340554757349179412/1340554757827461211/1366625114598412319): Users value **Gemini 2.5 Pro's** adaptability, noting its high **LM Arena** rank due to *one-shot prompt intensity*, but **Gemini 2.5 Flash** suffers from **rate limits** and **errors**, potentially due to an ongoing **Vertex token counting issue** reported on **OpenRouter**. Some users combine **Gemini 2.5** (planning) with **Deepseek** (diffs) effectively in **AI Studio**, leveraging Gemini's free access there.
- [**Meta Unleashes Llama 4 "Little Llama" at LlamaCon**](https://discord.com/channels/714501525455634453/853983317044756510/1366639817126973500): **Meta** confirmed **Llama 4** (aka *Little Llama*) during its **LlamaCon** event ([official livestream](https://www.youtube.com/live/6mRP-lQs0fw)), alongside revealing **SAM 3** development and releasing new tools like [Llama Prompt Ops](https://github.com/meta-llama/llama-prompt-ops) and the [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit). An early benchmark suggests **Llama 4** *sucks*, though its creator cautions the result comes from a single benchmark where the [**ELO difference might not be statistically significant**](https://github.com/paradite/eval-data).
- [**Sonnet Stumbles While Grok Gossip Grows**](https://discord.com/channels/1047197230748151888/1047204950763122820/1366625908194021487): Increased error rates hit the **Sonnet 3.7 API** ([Anthropic Status Incident](https://status.anthropic.com/incidents/th916r7yfg00)), prompting **Perplexity** to temporarily use fallback models, while anticipation builds for **Grok 3.5** amidst skepticism (*Grok 3... supplements substance with verbosity*). Despite reliability issues, some users still rank **Sonnet 3.7** as the #1 model for web development tasks on the webdev arena.

**Theme 3: Fine-Tuning & Optimization Frontiers Push Efficiency**

- [**RL & Fine-Tuning Frameworks Advance Model Capabilities**](https://discord.com/channels/1053877538025386074/1145143867818119272/1366857710435569766): **Nous Research** launches [Atropos](https://github.com/NousResearch/Atropos), an RL rollout framework ([read the intro post](https://nousresearch.com/introducing-atropos)), showcasing improved **DeepHermes** tool calling (**2.4x**/**5x** better) via **GRPO** and doubling corporate fundamentals prediction accuracy to **50%** ([view Atropos artifacts](https://huggingface.co/collections/NousResearch/atropos-artifacts-68110ab511c5af02830247b6)). Meanwhile, **Pi-Scorer** is introduced as an LLM-as-a-Judge substitute for evaluating checkpoints using [Pi-Scores](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) and implementing them as [GRPO reward functions](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb).
- [**Smarter Quantization Schemes Emerge**](https://discord.com/channels/1179035537009545276/1257011997250424842/1366805568731353119): A dynamic **BNB quantization** approach mixing **4-bit**, **8-bit**, and **BF16** precision based on module sensitivity ([see related paper](https://arxiv.org/abs/2504.18919)) is proposed in **Unsloth AI**, potentially reducing model size without hurting accuracy, which **Unsloth** may roadmap if demand exists. Separately, **GGUF's CPU offloading** capability is confirmed as a standard practice, supported by tools like **Transformers + Accelerate** or **Llama.cpp**.
- [**ktransformers Claims MoE VRAM Victory for Budget GPUs**](https://discord.com/channels/1131200896827654144/1131200896827654149/1366625096676020305): The [ktransformers library](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md) asserts it can efficiently run **Mixture of Experts (MoE)** models using just **8 GB of VRAM**, offering hope for running large models like **30B-A3B** on less powerful hardware. This contrasts with discussions on **Qwen3 MoE's** expert slider in **LM Studio**, where using more experts (e.g., the default **8** out of **128**) might paradoxically decrease quality ([see LM Studio screenshot](https://cdn.discordapp.com/attachments/1110598183144399058/1366676379848151060/Screenshot_2025-04-29_022126.png?ex=681278d0&is=68112750&hm=9d2c442870ff8f2a6a1e62b320c942dd3a8b167767b0dfcf538db545d2c602be&)).

**Theme 4: Tools & Platforms Navigate Glitches and Gains**

- [**Platform Peculiarities Plague Perplexity & OpenRouter Users**](https://discord.com/channels/1047197230748151888/1161802929053909012/1366670037858910289): **Perplexity** users report **Sonar API** debit card failures blocking hackathon participation and unexpected model substitutions due to **Sonnet 3.7** errors, despite **Perplexity** denying intentional switching. **OpenRouter** users face **Gemini 2.5 Flash** rate limits (linked to a **Vertex** token counting issue) and discover caching currently works only for **2.0 Flash**, not **2.5 Flash** (**"No endpoints found that support cache control"** error), noting caching boosts latency but not cost savings.
- [**LM Studio & Aider Adapt to Model Quirks**](https://discord.com/channels/1110598183144399058/1110598183144399061/1366626841892225084): **LM Studio** users navigate **Qwen3** template/parser issues and use the `/no_think` command to manage its verbosity, while confirming the lack of an **Android** version persists. **Aider** enhances user experience with a new *🔃 Thinking* spinner ([view the PR](https://github.com/Aider-AI/aider/pull/3911)), and users find a powerful workflow combining **Gemini 2.5** (for planning) with **Deepseek** (for diffs) via **AI Studio**.
- [**NotebookLM Gets Award, Languages; Audio Limits Critiqued**](https://discord.com/channels/1124402182171672732/1124402182909857966/1366674918128877682): **NotebookLM** celebrates a [Webby Award for Technical Achievement](https://winners.webbyawards.com/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement/331142/notebooklm) and expands its reach with support for [over 50 languages](https://www.forbes.com/sites/rogerdooley/2025/04/29/googles-notebooklm-now-speaks-50-languages-enabling-global-content/), though users observe shorter audio overview limits for non-English languages (e.g., .**6 min 20 sec** Turkish vs **15 min** English) due to unspecified *"technical reasons"*. The new **Audio Overview** customization prompt is capped at **500 characters**, and some report microphone detection failures in interactive mode.

**Theme 5: Hardware Heats Up with Mac Speed, GPU Competitions, and New Tools**

- [**Macs Flex Muscle with Blazing MLX Speed**](https://discord.com/channels/1131200896827654144/1131200896827654149/1366625096676020305): New Macbooks achieve impressive performance, hitting ~**100 tokens/s** for **Qwen3 30B A3B** using **MLX**, reportedly over twice as fast as **llama.cpp** based on a [Reddit speed comparison](https://old.reddit.com/r/LocalLLaMA/comments/1kaqnbj/speed_with_qwen3_on_mac_against_various_prompt). This performance fuels excitement for powerful local LLMs, potentially benefiting tools like **Aider**, particularly if the **4-bit Qwen3-30B-A3B** quantization holds up.
- [**GPU Arena Ignites with AMD Competition & FP8 Focus**](https://discord.com/channels/1189498204333543425/1359640791525490768/1366649146978730074): The **$100K AMD MI300 competition** within the **GPU MODE** Discord introduces a challenging single **GPU MoE kernel** problem ([read the official writeup](https://tinyurl.com/amd-comp-moe), [check the leaderboard](https://www.gpumode.com/leaderboard/430)), with final submissions due **May 27**. Discussions also explore **FP8 quantization** using **FP32 accumulation** for matmul ([see ONNX FP8 formats page](https://onnx.ai/onnx/technical/float8.html)), referencing **Deepseek-v3's** tech report and concerns about potential *underflow issues*.
- [**Hardware Hints & Helper Tools Surface**](https://discord.com/channels/1189498204333543425/1233704710389764236/1366728443273936958): **GPU MODE** members dissect the **CDNA3 ISA Reference**, noting (section 2.2.1) each compute unit's **64kB** low-latency memory structured as **32 banks** of **512 entries** (4 bytes each). A community member launched a [GPU price tracker for Amazon](https://www.unitedcompute.ai/gpu-price-tracker) displaying historical pricing and **teraflops per dollar**, while **Modular** users recommend [flamegraph](https://github.com/brendangregg/FlameGraph) for visualizing `perf` output (requires compiling with debug info)

---



---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Hits WhatsApp with Images**: **Perplexity AI** is now on **WhatsApp**, integrating **image generation**. Try it out via [this link](https://wa.me/18334363285).
   - This expansion allows users to generate images directly within **WhatsApp**.
- **Sonnet Stumbles and Swaps for Stability**: The **Sonnet 3.7 API** is experiencing increased error rates, causing **Perplexity** to temporarily route queries to alternative models as a fallback, per [Anthropic Status Incident](https://status.anthropic.com/incidents/th916r7yfg00).
   - The team clarified that model switching is to maintain service availability during **Sonnet** issues, *not an intentional practice*.
- **Sonar API Card Issues Thwart Hackathons**: Users report debit card incompatibility with the **Sonar API**, preventing usage for hackathon projects; also reporting [not receiving hackathon credits](https://discord.com/channels/1047197230748151888/1118264005207793674/1366292101666439239) after card verification.
   - The issues block access to API and hinder participation in hackathons.
- **Structured Output Struggles Surface**: Users are facing problems with **structured output** from the API, citing unexpected output formats and schema enforcement difficulties.
   - One user reported needing to specify *'In english'* to prevent the API from returning Mandarin, similar to issues another user had seen with **R1 based models** going into mandarin while thinking, especially when trying to solve equations.
- **Grok App Selling for Pennies in India**: The **Grok** android app is reportedly charging only **700rs per month** for supergrok for Indian users, but the *free tier isn't even working anymore* for some.
   - The app can be accessed on X if you have premium +.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 GGUFs Plagued by Parser Problems**: Users are running into template issues with **Qwen3 GGUF models** in **LM Studio**, especially the **128k context length** versions, which cause parser errors; but the models are compatible with **Ollama** and **llama.cpp**, enabling integration with platforms like **Open WebUI**.
   - Some users found that the **ChatML template** can be used as a workaround, though it is not technically correct, and despite the underlying **llama.cpp** runtime, LMStudio isn't up-to-date to resolve these inconsistencies across different platforms.
- **ComfyUI Sparks Complex Commentary**: Members shared an image depicting **ChatGPT's opinion of ComfyUI** which prompted humorous reactions.
   - One user commented that the *scrambled lines* in the middle of the image accurately represent the complex processes involved.
- **GRPO Fine Tuning on the Upswing**: Users doing **GRPO** (Gradient Rollout Policy Optimization) are reporting positive results and offer to provide assistance to others, with one user reporting they found **Qwen 4b** better than **gemma 3 4b notebook** for their use case.
   - However another user reported inconsistent results when fine-tuning **Gemma 3 4B** for generating nested **JSON** configs using **GRPO**, with accuracy dropping significantly for short inputs; the descriptions significantly affected the trigger and action components, leading to inconsistent **BLEU** scores.
- **Dynamic BNB Quantization Scheme Proposed**: A member proposed creating a dynamic **BNB quantization** scheme where modules use **4-bit**, **8-bit**, or **BF16** precision based on their sensitivity, suggesting this could reduce space without sacrificing accuracy; a related paper was mentioned [here](https://arxiv.org/abs/2504.18919).
   - Another member indicated that *if there is sufficient user demand for this, it might be something we could roadmap out*.
- **Model Serving System vLLM gets a Nod**: After a user reported issues with **Qwen3 GGUF models** from Unsloth, another member suggested trying [vLLM](https://github.com/vllm-project/vllm).
   - The member provided a sample command to serve **unsloth/qwen3-unsloth-4bit** using vLLM.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pro Demand Defies Delay**: Users eagerly await the release of **O3 Pro**, joking about its potential impact and labeling it as a "p2w" (pay-to-win) model.
   - Concerns arise regarding its cost and accessibility, with some users humorously noting their prolonged wait of *day 13*.
- **Qwen 3 Benchmarking Baffles, Training Talk Teases**: Discussions around **Qwen 3**'s performance reveal that despite strong benchmark results, it doesn't intuitively feel as smart as **2.5 Pro** in practice, leading to speculation about its post-training refinement.
   - Suggestions arise that **Qwen 3**'s base model could excel in fine-tuning, with one user reporting it outperforms **Gemini 2.5 Pro** on some benchmarks, though experiences vary.
- **Gemini 2.5 Pro Still Reigns Supreme**: Some users still favor **Gemini 2.5 Pro** for its unique adaptability to different roles and its ability to adopt positions on niche topics, making it feel like interacting with a team of experts.
   - Despite other models topping individual benchmarks, users find **2.5 Pro** ranked higher on the LM Arena due to its adaptability to *one-shot prompt intensity* in the way that it *assumes the role of the question answerer with no single personality*.
- **Grok 3.5 Gossip Grows**: Enthusiasm and skepticism mix as users anticipate the arrival of the **Grok 3.5** model.
   - One user commented that **Grok 3** *overreaches every time, it's like when you ask it to prove something it supplements substance with verbosity*.
- **Sonnet 3.7: WebDev's Top Model?**: Users debated the capabilities of **Claude 3.7 Sonnet**, claiming the model *is still ahead in most of my cases for web dev tasks*, with some agreeing that its still perplexing.
   - Some noted that **Sonnet 3.7** is currently the #1 model on the webdev arena.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3 Silenced with /no_think Command**: Users discovered that the `/no_think` command disables the *thinking* output of **Qwen3** in LM Studio, but it may require repeating the command or reloading the model.
   - One user noted that the command only worked after seeing someone else use it, indicating a potential bug or undocumented behavior in LM Studio; [here is an example](https://cdn.discordapp.com/attachments/1110598183144399058/1366664571313455165/image.png?ex=68126dd1&is=68111c51&hm=95fa11f26f302fb70dff44ffabe1026e3594542e5ba09ee299c8085602b363e8&).
- **Android LM Studio Remains Elusive**: Despite user interest, there is currently no **Android** version of **LM Studio**, disappointing those seeking mobile LLM capabilities.
   - One user jokingly took on the challenge to implement it, highlighting the demand for a mobile version.
- **Qwen3's Expert Count Creates Confusion**: Users questioned the purpose of the *number of experts* slider for **Qwen3 MoE** in LM Studio, with one noting that their LM Studio defaulted to **8 experts** out of **128**.
   - The consensus appears to be that using more experts can lead to reduced quality due to subject matter experts being *overruled by many idiots*; here is a [relevant screenshot](https://cdn.discordapp.com/attachments/1110598183144399058/1366676379848151060/Screenshot_2025-04-29_022126.png?ex=681278d0&is=68112750&hm=9d2c442870ff8f2a6a1e62b320c942dd3a8b167767b0dfcf538db545d2c602be&).
- **Bug Fixes Boost Qwen3 Performance**: New **Qwen 3** versions with bug fixes have been released, addressing a broken template that slowed the model down, including dynamic quants2.0.
   - Users reported that *the bugfixed models are even faster now* and respond more appropriately.
- **MLX Blazes Past llama.cpp in Speed**: [MLX](https://github.com/ml-explore/mlx) reportedly achieves more than twice the speed of **llama.cpp** in prompt processing with **Qwen3-30B-A3B**.
   - These performance comparisons were discussed in a [Reddit thread](https://old.reddit.com/r/LocalLLaMA/comments/1kaqnbj/speed_with_qwen3_on_mac_against_various_prompt), highlighting the experiences of users on Macs.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Qwen3 has coding ups and downs**: **Qwen3**'s coding capabilities sparked discussion; one user praised its explanations, while another cited [issues with complex math tasks](https://huggingface.co/models).
   - A user reported fixing complex math tasks by *lowering my temp a bit more*, while another noted problems with **Qwen3**'s **tool calling**.
- **Gemini 2.5 Flashes Rate Limits and Errors**: Users are reporting that **Gemini 2.5 Flash** is hitting **rate limits** and **errors**, even on paid versions; a user experienced this despite disabling web search.
   - It was clarified that **OpenRouter** is facing an ongoing **Vertex issue with token counting**, and the [free tier limits](https://aistudio.google.com/) are **not supported** on OpenRouter, though a member pointed out a way to use [Gemini 2.5 pro for free](https://ai.google.dev/gemini-api).
- **OpenRouter Caching limited to 2.0 Flash**: **OpenRouter caching** is currently **not working for 2.5 Flash**, only 2.0 Flash, and 2.5 Flash errors on them (**No endpoints found that support cache control**).
   - **Toven** clarified that new caches are written for new 5 min TTLs, and that caching improves latency but **doesn't affect pricing**.
- **LLama 4 Flunks New Benchmark**: According to a benchmark review **LLama 4 sucks**, though it was noted that it is really just one benchmark.
   - The person who did the benchmark added that [the **ELO within 25 range is not statistically significant**](https://github.com/paradite/eval-data) to tell the difference.
- **Tesla FSD sparks numeric system debate**: An announcement of an X post showed a model stating that **9.9 is greater than 9.11**, leading some to ponder if that was correct.
   - Others brought up that it *depends on the context* as [**Tesla FSD versions work differently**](https://x.com/elonmusk/status/1917099777327829386), and that 9.11 > 9.9.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Qwen3 Runs Crazy Fast on New Macbooks**: New Macbooks are yielding impressive speeds of around **100 tokens/s** for **Qwen3 30B A3B** using mlx.
   - The possibility of a fast, local LLM for **Aider**, especially if the **4-bit quant version of Qwen3-30B-A3B** performs well on the Aider benchmark, sparks excitement.
- **ktransformers Claims VRAM Optimization for MoE**: The [ktransformers library](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md) claims to efficiently run **Mixture of Experts (MoE)** models with only **8 GB of VRAM**.
   - This approach offers a potentially more hopeful way to handle **30B-A3B** models compared to loading all parameters into VRAM.
- **Deepseek R2 Hype Builds with Vision and Self-Learning**: The upcoming **Deepseek R2** is rumored to feature enhanced human vision capabilities and self-learning features, potentially releasing *tomorrow*, as shown in [this documentary](https://www.youtube.com/watch?v=Lo0FDmSbTp4).
   - Enthusiasts eagerly anticipate its release.
- **Aider Gets a Thinking Spinner**: A new [PR](https://github.com/Aider-AI/aider/pull/3911) introduces a *🔃 Thinking* spinner to **Aider**, displayed while waiting for LLM output.
   - The contributor suggests this small addition makes **Aider** *feel snappy + alive*.
- **Gemini 2.5 and Deepseek Form Winning Team**: A user discovered that **Gemini 2.5** for planning and **Deepseek** for diffs and vchanges explanations is a good combo.
   - They recommend this in **AI Studio** because Gemini is free there.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **FP8 Accumulation with FP32 Investigated**: Members discussed the possibility and benefits of using **fp8 quantization** with **fp32 accumulation** for **matmul** operations, particularly in the context of **Deepseek-v3**'s tech report, with [a link to the ONNX FP8 formats page](https://onnx.ai/onnx/technical/float8.html).
   - It was noted that **FP8** might encounter *underflow issues*, potentially requiring a higher precision accumulator, also in conjunction with [this leaderboard](https://www.gpumode.com/leaderboard/430).
- **Single GPU MoE Kernel Challenge is Live**: A new single **GPU MoE kernel** problem is now available for the **$100K AMD MI300 competition**, as announced in the [announcements channel](https://discord.com/channels/1189498204333543425/1189640399476764692).
   - It's suggested to read the [official problem writeup for this kernel](https://tinyurl.com/amd-comp-moe) carefully, and also remember that registration closes **April 30** with submissions due **May 27**.
- **AOT Inductor Training Faces Multithreading Snafus**: A user reported partial C++ training success with **AOT Inductor**, suspecting multithreading issues due to unwanted specialization of code.
   - The user plans to open a [PyTorch issue](https://github.com/pytorch/pytorch/issues) for further investigation, specifically on the API's behavior with multiple worker threads calling `fw_graph->run()`.
- **CDNA3 ISA Memory Layout Unveiled**: The **CDNA3 ISA Reference**, section 2.2.1, reveals that each compute unit features a **64kB** memory space for low-latency communication.
   - This memory is structured with **32 banks**, each comprising **512 entries** of **4 bytes**, facilitating efficient data access and inter-thread communication.
- **Amazon GPU Prices, Tracked!**: A member launched a [GPU price tracker](https://www.unitedcompute.ai/gpu-price-tracker) for **Amazon**, providing historical pricing data and calculating metrics like **teraflops per dollar**.
   - The tool helps users pinpoint optimal times to acquire GPUs for private clusters, leveraging comprehensive pricing trends.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Remembers... Sort Of**: **ChatGPT** now features persistent memory, split into long-term (derived from important chat details) and short-term (referencing the past **90 days**) memory, enhancing context retention.
   - Users can disable either memory type, providing control over data retention, but one toggle does not control both.
- **AI Agent Firm Flounders Fantastically**: A professor-led experiment staffing a company entirely with AI agents produced [*chaotic results*](https://futurism.com/professors-company-ai-agents), highlighting current AI's limitations in fully replacing human roles.
   - Despite claims from big tech, the experiment demonstrated the necessity of human oversight for current AI models.
- **IAM360 Orchestrates AI Harmony**: A member is developing **IAM360**, an experimental human-AI symbiosis framework that uses modular symbolic **GPT agents** with persistent roles and a zero-shot orchestration system for emergent dialogue.
   - Built using standard **ChatGPT** sessions, **IAM360** aims for natural interactions without custom **GPTs**, fine-tuning, or API integrations.
- **AI Artistry Attracts Acclaim?**: A user successfully sold an AI-generated thumbnail for **1500 Robux**, showcasing a niche application of AI in digital content creation.
   - However, others cautioned that current AI image generators struggle with complex reference images, potentially limiting real-world client appeal.
- **ChatGPT's Bio Tool Boosts Builds**: Members identified **ChatGPT's** internal memory as the `bio` tool, and suggested developers explicitly invoke the `bio` tool for defining save commands within prompts to ensure accurate state retention.
   - Concrete specifications to prompts will minimize **LLM** guessing; ask it to identify and describe its connected tools, listing their canonical names and demonstrating their proper syntax.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **PyQt5 Chat App Interface with LM Studio**: An AI chat application built with **PyQt5** was shared, leveraging **LM Studio** as its backend server via [this python script](https://cdn.discordapp.com/attachments/986699377257119794/1366717106447450122/AI_chat_app_005.py?ex=6811f5fe&is=6810a47e&hm=d9601e58ece57f0a5ba85d7da1c73f099068ee63c9220099dffa3614c74cd9bd).
   - To enable functionality, the user must first select a model and start it as a local server within **LM Studio** prior to running the application.
- **Debate Disentangles OR and ML Roots**: A discussion debated the historical relationship between **Operations Research (OR)** and **Machine Learning (ML)**, pinpointing a divergence in methodology.
   - While early **AI/ML** closely mirrored **OR** and **control theory**, modern ML has shifted towards statistical methods emphasizing *learning from data rather than modeling reality from first principles*, with an increased focus on empirical approaches.
- **Anonymous LLM fools Reddit**: Researchers tested an anonymous LLM on Reddit's **/r/changemyview** and found *very high efficacy*, leading to annoyance among users, as discussed in [this X post](https://x.com/emollick/status/1916905103358931084) and [Reddit thread](https://www.reddit.com/r/changemyview/s/k9Rd6IbyjY).
   - One user humorously stated, *AIs aren't smart, change my mind* to which **ChatGPT** responded *Yes, they are* and the user replied *oh okay, im sorry*.
- **Qwen 3 Excites Users with Reasoning**: Members lauded the new **Qwen models**, specifically mentioning improved reasoning and instruction following abilities.
   - One user reported that *their output for some reasoning tasks* is superior, especially praising the **MoE** model's speed and intelligence, describing it as *just as smart as 2.5 Flash, if not smarter*.
- **Meta Announces Llama 4**: The existence of **Llama 4**, also known as *Little Llama*, was confirmed at **LlamaCon**, as seen in [this YouTube livestream](https://www.youtube.com/live/6mRP-lQs0fw).
   - A key announcement from **LlamaCon** was the development of **SAM 3** and **Meta's** new app, with some speculating how the smaller **Llama 4** models will compare to existing **Qwen** models.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Atropos Framework Guides RL**: **Nous Research** launched [Atropos](https://github.com/NousResearch/Atropos), a rollout framework for reinforcement learning with foundation models that supports complex environments to advance model capabilities, alongside training and inference components detailed in their [introductory blogpost](https://nousresearch.com/introducing-atropos).
   - Artifacts created using environments in Atropos, including a new dataset and five new models for tool calling and corporate fundamentals prediction, are available at [HuggingFace](https://huggingface.co/collections/NousResearch/atropos-artifacts-68110ab511c5af02830247b6).
- **GRPO Tool Calling Improves DeepHermes**: The **GRPO** environment improved **DeepHermes'** tool calling by **2.4x** and **5x** on simple and parallel tool calls, respectively, using Berkeley's Function Calling Benchmark.
   - Atropos is a key component of **Psyche**, an upcoming decentralized training network coordinating pre-training, mid-training, and post-training workloads globally; a hackathon will be hosted in San Francisco on May 18th to foster collaborative progress (more details coming soon).
- **Fundamentals Prediction Model Accuracy Doubles**: The corporate fundamentals prediction model's accuracy increased from **~25%** to **50%** on directional changes using the **Atropos** framework.
   - The Atropos framework is designed to guide language models toward their optimal potential through reinforcement learning.
- **DeepSeek R2 Release: Fact or Fiction?**: There are rumors that **DeepSeek R2** may be released soon and was fully trained on **Huawei Ascend 910B** hardware, but these claims have been refuted.
   - A tweet was linked with the official line from **DeepSeek** stating that *"We will release R2 when we release R2, everyone who claims they know is lying"*.
- **Qwen 3 Overfits on Evals**: Members found that **Qwen 3's base models** seem very overfitted to certain evals, reporting that the model scored **75%** for **Trivaqa** on **M24b** but only **60%** on **Q30MoE**.
   - This prompted discussion about the effectiveness of MoE.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Spending Limits Stall Speedy Signals**: After exceeding spending limits, users reported delays for hours, despite upgrading, while another reported they ran out of **fast requests**.
   - One user noted that **Gemini** remains fast even on slower requests, while others faced challenges with **Gemini 2.5 Pro**.
- **Discord's Development: Discourse Delights Developers**: One member jokingly noted that the **Cursor’s Discord** is *finally getting some love again*, indicating increased activity and engagement.
   - Another member responded with confidence that *Cursor has always been loved*, implying the team is simply polishing the cube.
- **Gemini Glitches Generate Grief**: Users reported that **Gemini 2.5** frequently stops mid-request, even after indicating it would perform actions.
   - A team member said they are working with **Google** to resolve the issue, advising users to use other models and submit their **request ID** for investigation.
- **Agent Apathy: Edits Evade Engineers**: Users face persistent problems with the **Agent failing to make edits** after multiple attempts, instead advising manual edits.
   - A team member suggested the issue might stem from **Gemini 2.5 Pro**, recommending refreshing the chat context or switching to **GPT 4.1**, **GPT 3.5**, or **Claude 3.7**.
- **Ollama Official: Opening on Over-the-Air**: A user inquired about the release timeline for an official **Ollama Smartphone App**, and posted a relevant [X post](https://x.com/awnihannun/status/1917258279455187034).
   - A user mentioned that reinstalling **Cursor** and clearing the cache fixed issues, while another confirmed manual cache clearing as an alternative to reinstalling.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Turnstile Test Triumph!**: Members successfully tested [Cloudflare Turnstile](https://mineapi.pythonanywhere.com/docs), confirming its functionality.
   - The successful test prompted enthusiastic reactions from the members.
- **Whisper Turbo Troubles Hit HF!**: Users reported that **OpenAI's whisper-large-v3-turbo** is not functioning on the HF inference endpoint, impacting even the webpage demo.
   - Members shared similar issues like [this one](https://discuss.huggingface.co/t/sentence-transformers-all-minilm-l6-v2-not-working-all-of-a-sudden/152691) for potential troubleshooting.
- **GGUF CPU Offloading Goes Mainstream**: Members confirmed that **GGUF format** accommodates CPU offloading, especially when merging checkpoints.
   - They noted that *Transformers + Accelerate or Llama.cpp* facilitate this process.
- **Pi-Scorer Poised as LLM-as-a-Judge Proxy**: A member introduced **Pi-Scorer** as a viable substitute for **LLM-as-a-Judge**, showcasing Colab notebooks for evaluating model checkpoints using [Pi-Scores](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) and implementing them as [reward functions](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb).
   - This could provide a useful tool for SFT Model Checkpoint Observability with Pi.
- **Edge Filters Emerge for Excellent Error Extractions**: A member suggested filters like **Canny edge** or **Sobel** for isolating defects with specific thresholds in images.
   - With the right threshold, auto-annotating scratches on datasets could be much easier.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Nabs a Webby for Technical Prowess!**: **NotebookLM** celebrated a **Technical Achievement** award at the [Webby Awards](https://winners.webbyawards.com/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement/331142/notebooklm).
   - This accolade underscores **NotebookLM's** ongoing enhancements to its platform.
- **NotebookLM's Global Voice: Now in Over 50 Languages!**: **NotebookLM** introduced **multilingual support**, now speaking [over 50 languages](https://www.forbes.com/sites/rogerdooley/2025/04/29/googles-notebooklm-now-speaks-50-languages-enabling-global-content/), enhancing access for diverse users.
   - However, rollout is gradual; some users initially faced UI glitches, such as one reporting that **Vietnamese audio** wasn't working and the UI still said *"English only"*.
- **Audio Overview Customization Caps Prompt Queries!**: Users testing the **Audio Overview** customization feature discovered a **500-character limit**, raising questions about its utility versus uploading separate instruction files.
   - One user aimed to *"lessen the silly banter, and keep focus on the facts and timeline"*.
- **Audio Overview Times Vary by Language!**: Users reported that **non-English audio overviews** had shorter time limits compared to English; for example, English had a **15-minute limit** versus **6 minutes 20 seconds** for Turkish.
   - The team cited *"technical reasons"* for these limits but assured that they are actively working on extending the duration.
- **Microphone Issues Plague Interactive Mode!**: A user reported that **interactive mode** failed to detect audio from their microphone, disrupting usability.
   - Troubleshooting suggestions included verifying **microphone permissions**, checking **browser settings**, using a [mic test](https://mictests.com/), and trying an alternative browser.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Add-on Credits Confuse Users**: A user reported that add-on credits from early subscriptions to Manus.im are useless without resubscribing due to their short expiry, causing the loss of **3900** credits.
   - Another user clarified that bonus credits do not expire as long as the subscription remains active, and invite distributions appear random, potentially throttled.
- **Manus Fellow Program Questioned**: A user inquired about the Manus Fellow Program's selection process, targeted countries, and inclusivity for regions like Pakistan and India.
   - Another user clarified the invite structure, noting starter plans give **2 invites** and pro plans give **5 invites**.
- **Beta Testing Under Scrutiny**: A user critiqued Manus.im's beta testing approach, arguing that limiting users with credits undermines the purpose of a beta phase.
   - They suggested that *a real beta test would let users complete full projects from start to finish, giving meaningful feedback about the experience and suggesting improvements*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **X-Ware Red tool released to community**: A user shared **X-Ware Red**, which uses the title of an embed and prepends `r.jina.ai/` and `openrouter-free-tier` to generate titles for threads.
   - Another user suggested adding a toggle to let users control whether the thread title should differ from the embed name.
- **Meta Ships Llama Prompt Ops for Engineers**: **Meta** introduced [Llama Prompt Ops](https://github.com/meta-llama/llama-prompt-ops), an open-source tool designed for prompt engineering, along with the [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit).
- **Link Posting Retitles Threads, User Reports**: A user reported a bug where posting a link in a thread incorrectly retitles a thread that already has a name.
   - The bug *should only look for threads with 'https://' in the title and change those*.
- **Community Scours for Durable LLM Benchmarks**: A user requested a reliable survey of **LLM benchmarks** that supports historical comparisons of models.
   - Another user noted that *most benchmarks last less than 2 years*, recommending the "AI Engineer Reading List" for current benchmarks and linking to posts for OSS leaderboard versions 1 and 2.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular's Repository Gets Multi-Licensed**: The **Modular repository** now requires multiple licenses due to parts of `src/max` being licensed under Modular's **Community License** while the rest uses **Apache 2**.
   - This change reflects the diverse licensing needs within the repository, particularly for components like those found in [`src/max/serve`](https://github.com/modular/max/blob/main/src/max/serve/README.md).
- **Bending Origins Leads to Headaches**: Members discussed issues with **Origins** in Mojo, particularly around gaps in APIs and missing language features like parametrizable traits, which complicates rebinding origins to container elements.
   - It was also noted that holding two mutating references to the same origin is problematic, though one can cast the origin to a **MutableAnyOrigin** to circumvent this limitation.
- **Pointer Time to Screw Origins**: To handle implementing list-like and span-like types, or reading `sort` implementations in the standard library, developers sometimes bypass **Origins** and resort to *pointer time*.
   - The discussion highlighted concerns about pointer types, especially regarding mutability and immutability fixes in Mojo.
- **Standard Python Imports Loom**: Full support for standard Python `import` statements in Mojo may arrive, suggesting `python.import_module` could eventually be deprecated.
   - A member described the possibility of this change as a *pretty definite maybe*, hinting at future enhancements to Python integration within Mojo.
- **`Flamegraph` Visualizes Perf Output**: For visualizing `perf` output, members suggested using [flamegraph](https://github.com/brendangregg/FlameGraph), which requires compiling the executable with **debug info** for effective analysis.
   - They also mentioned using `llvm-mca` for profiling particular blocks of code, referencing a private part of the `gpu` module ([link](https://github.com/modular/max/blob/main/mojo/stdlib/src/gpu/profiler.mojo)).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GPT-4o masters Tetris via LlamaIndex**: A video demonstrates **GPT-4o** generating **Tetris** in one shot using **LlamaIndex** and **Composiohq**, showcasing its advanced code generation capabilities.
   - The code used in the demo is available on [GitHub](https://t.co/KJb7YRINWg), offering a practical example for developers.
- **PapersChat indexes ArXiv and PubMed with LlamaIndex**: **PapersChat** indexes papers on **ArXiv** and **PubMed**, using **LlamaIndex**, **Qdrant**, and **MistralAI**.
   - The nifty web UI to query them is available [here](https://t.co/lYwXh27F9x).
- **Azure OpenAI Plagued by Intermittent Timeouts**: Users report intermittent **timeouts** with **Azure OpenAI** endpoints, even with consistent prompts, endpoints, and network conditions, suggesting potential **rate limits** or **firewall issues**.
   - Retry mechanisms are sometimes ineffective, and network changes only occasionally resolve the inconsistency.
- **MessageRole: Cracking the FUNCTION vs. TOOL Code**: The difference between **MessageRole.FUNCTION** and **MessageRole.TOOL** depends on the specific API in use.
   - Some APIs like **OpenAI** utilize **tool messages**, while others rely on **function messages**.
- **Function Agent Context Snafus Unveiled**: A user encountered an issue with a **function agent** getting stuck at the stream event during the second round of interaction; the user provided a sample code.
   - A member suggested awaiting the handler (`await handler`) after `stream_events()` exits to ensure the previous run concludes and the final response is received, which fixed the error.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RAG Chatbot Wrestles with Multi-Source Answers**: A member building a **RAG-based chatbot** is struggling with generating answers that require information from multiple documents, even when using **vector search** and **BM25**.
   - The chatbot uses **LLM Claude 3.5 Sonnet v1** and **Amazon Titan v1** embeddings, and the member is seeking advice on how to effectively link references to appendices within the documents.
- **GraphRAG Debated for Multi-Source Data**: A member inquired about the value of using **GraphRAG** to aggregate answers from multiple sources, comparing it to **insightRAG**, which demands a domain-specific pre-trained model.
   - They are seeking alternative solutions to **GraphRAG** and noted their plans to attend **NAACL**.
- **Engineer Kickstarts Local Inference Project**: A member, previously a co-founder of [Dataherald](https://github.com/Dataherald/dataherald), is initiating a new project focused on **local inference** and **small model training**.
   - The member expressed keen interest in collaborating with the community and contributing to relevant research.
- **Symbolic Prompt Recursion Explored**: A member is investigating the behavior of **recursive symbolic prompts** under classifier pressure, particularly how smoothing and alignment constraints impact **multi-turn hallucination drift**.
   - They are keen on understanding how symbolic structures such as **role-bound predicates** or **attention-synced markers** persist across multiple outputs, despite soft-alignment drift and output smoothing.
- **HHH Objectives Exposed**: Research was shared on [quantitatively scoring LLM outputs](https://www.notion.so/TPIP-Exposing-Alignment-Tension-in-Modern-LLMs-1d5927516e1b8080b8c3d625a40a131d) based on **HHH** (Helpful, Honest, Harmless) alignment objectives, using **YAML** and **python/Gradio** to audit user sessions.
   - Frontier models were observed to vary widely in honesty compliance, with some, like **ChatGPT 4o** and **4.5**, ironically outputting high confidence in ambiguous answers, making **OpenAI** the least transparent of frontier models.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Credential Passing Concerns**: A member is encountering issues while attempting to pass credentials through headers from a client to the **MCP server** using Python, seeking assistance from the community.
   - Currently, no solutions or suggestions have been provided in response to the query.
- **RAG Server Architecture Debated**: A member is exploring the feasibility of building a **RAG-type server** where clients can upload files via an endpoint, store them server-side, and utilize them for question answering.
   - They are soliciting feedback on the viability of this approach and whether alternative architectures might be more effective.
- **Streamable HTTP Authentication Nuances Emerge**: A member inquired about the community's opinion on the **Streamable HTTP implementation and authentication**, especially in the recently released **TS SDK**.
   - Feedback indicates that it's functioning effectively, but members are still investigating the nuances of hosting a **multi-tenant server** and how statefulness impacts it.
- **Multi-Tenant Server Statefulness Examined**: Concerns have been raised regarding hosting a **multi tenant server** and the implications of statefulness, specifically questioning why a single instance suffices for stateful setups but not for stateless ones.
   - The discussion revolves around whether a stateless server should spawn a new instance of the **MCP server** per request.
- **Open Source Agentic Apps: Production Ready?**: A member questions the real-world applicability of open-source models for agentic applications in production environments, not just for pet projects.
   - They express skepticism about the ability of most open-source models to reason or follow instructions effectively without fine-tuning.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Speedy Gradient Scaling Arrives via Foreach**: A member shared a [code snippet](https://link.to/snippet) using `torch._foreach_mul_` for gradient scaling, potentially merging with gradient clipping for a single parameter loop, increasing optimization speeds.
   - Another member pointed out the [related PR](https://github.com/pytorch/torchtune/pull/2624) and wondered if the seemingly constant gain accumulates over many iterations, noting potential caveats.
- **Tune Contributors Seek Easy First Issues**: A member highlighted [two easy issues](https://github.com/pytorch/torchtune/issues/2648) and [another](https://github.com/pytorch/torchtune/issues/2649) for community contribution to the project, designed to lower the barrier to entry.
   - These issues provide opportunities for new contributors to get involved in the project and gain experience, but are not described in detail.
- **DoRA and QAT Pairing Unexplored**: A member inquired about experiences combining **DoRA (Difference of Low-Rank Adaptation)** with **QAT (Quantization-Aware Training)**, an under explored combination.
   - There was no discussion or response regarding this combination in the messages provided, suggesting a knowledge gap or lack of experimentation in the community.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Users Crave MCP Usage Documentation**: Users are requesting tutorials or documentation for the new **MCP (Multi-Controller Processing)** feature introduced in the latest release of **DSPy**.
   - One user suggested that getting started by reviewing the test cases helps clarify understanding of the **stdio** and **SSE clients** setup, so a tutorial may not be necessary.
- **React Developers Ponder Displaying Thoughts Component**: A user asked for advice on the best way to display the **Thoughts component in React** within the **DSPy** framework.
   - They mentioned the option of modifying the forward method, but inquired about a more appropriate place to implement this feature.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Markdown vs Image RAG Debate Brews**: Members discussed comparing **Markdown-based** versus **Image-based multimodal RAG** on **PDFs**, with one member using **Docling** to convert PDFs to Markdown and compute text embeddings.
   - They are considering switching to **EmbedV4** to process raw images directly for multi-modal embeddings in RAG.
- **Cohere Considers Embed V4 Rate Limit Hike**: A user inquired whether **Cohere** would increase production rate limits for `embed-v4`, stating that **400 requests per min** is insufficient for their PDF-heavy use case.
   - No response has been given.
- **Embed V4 Bedrock Availability Teased**: A user inquired whether **Embed V4** will be available on **Bedrock**.
   - There has been no answer from Cohere yet.
- **New Data Scientist Pumps Embed V4**: A new data scientist joined the Cohere Discord community, expressing excitement in trying new tools, particularly the latest **Embed V4 model** from Cohere.
   - The new member is *pleased to join the community*.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Manus AI Tool Goes Global**: A member shared [Manus AI](https://manus.im/edu/invitation/FBBGGFBFKTUE), noting its availability after being *dropped by China*.
   - The tool is purported to be the *first auto research AI Agent*, stirring discussions about its potential impact.
- **Nomic Powers Embedding Workflows**: A member highlighted that **Nomic** provides comprehensive embedding tools, suggesting it goes *beyond GPT4All*.
   - They emphasized the versatility of **Nomic's** embedding tools, stating they are compatible with *various other software*.
- **Group Embeddings, Skip Training?**: A member proposed that **grouping embeddings** could be a substitute for traditional training methods.
   - The suggestion involves grouping embeddings for a specific person, averaging them, and then using that average to sort and identify other pictures of the same person.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Berkeley Models Evaluated Loosely vs Strictly**: A member proposed *'loose' vs 'strict' evaluation mechanism* for **Berkeley function calling models**, especially those that can be *'hacked'* into working, representing specific use-cases.
   - They provided an example of a model incorrectly trained to emit `<tool_call>` instead of `<|tool_call|>` as its specs indicate, where a knowledgeable user might ignore the error and evaluate functional correctness.
- **Model Training Creates Inconsistencies**: One member encountered a model which was incorrectly trained to emit `<tool_call>` instead of `<|tool_call|>` as its specs indicate.
   - The member suggested that, if they knew the model specifically, they could ignore this error and evaluate on functional correctness, but a naive user could not.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1366625908194021487)** (2 messages): 

> `Perplexity AI on WhatsApp, Sonnet Model Behavior Update, Anthropic Status Incident` 


- **Perplexity Hits WhatsApp with Image Generation!**: **Perplexity AI** is now available on **WhatsApp** including **image generation** features, accessible via [this link](https://wa.me/18334363285).
- **Sonnet Stumbles, Routes to Alternative Models!**: Due to increased error rates with the **Sonnet 3.7 API**, some queries are temporarily routed to alternative models as a fallback, which is related to [Anthropic Status Incident](https://status.anthropic.com/incidents/th916r7yfg00).
- **Model Switching: No Intentional Shenanigans!**: The Perplexity team clarified that they **do not intentionally switch your selected model**; routing only happens when **Sonnet** encounters an error to maintain service availability.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1366625571546861608)** (1112 messages🔥🔥🔥): 

> `Free AI billing, Grok android app, Model Fallbacks, The Boys fanboy` 


- **Users Exploit Free AI Billing**: Some users claim to have *not paid a penny* for their AI bill for a year, possibly through [Play Store](https://play.google.com/store/account/subscriptions) or by *joining a few webinars and filling out some forms*.
   - Others requested the method, while some expressed disbelief.
- **Grok app is cheap for Indian users**: The **Grok** android app is reportedly charging only **700rs per month** for supergrok for Indian users, but the *free tier isn't even working anymore* for some.
   - It's available on X if you have premium +.
- **Perplexity is replacing models without notification**: Users are complaining that Perplexity is replacing Claude 3.7 with a lower-quality model like **GPT 4.1** or **Deepseek**, and are angry because *no model switch or clear model indicator on their responses*.
   - One user said that *It is straight up using R1 to generate the answers that are being sent to sonnet thinking. And then saying that the answer came from sonnet. That's shady.*
- **Discord Channel Becomes a The Boys Fanboy Convention**: The channel's conversation swerved into *The Boys* territory, with users sharing GIFs and discussing plot points, like [the Homelander kills someone in public scene](https://www.youtube.com/watch?v=IIGf-kK-g2I&t=0).
   - Others mused whether or not to skip Homelander scenes altogether and jokingly asked if the show has any scenes that are more disgusting than usual.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

_paradroid: https://www.perplexity.ai/search/d7bb905e-27e3-43e9-8b68-76bea1905457
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1366670037858910289)** (14 messages🔥): 

> `Sonar API Debit Card Issues, Hackathon Credits, Structured Output Issues, Async Deep Research API, API vs Web Results` 


- ****Card Conundrums** Plague API Users**: A user reported that their debit card is not supported for the **Sonar API**, preventing them from using it for their hackathon project; also reporting [not receiving hackathon credits](https://discord.com/channels/1047197230748151888/1118264005207793674/1366292101666439239) after card verification.
   - No solutions were provided in the given context.
- ****Structured Output** Struggles Surface**: Users are experiencing issues with **structured output** from the API, including unexpected output and difficulty enforcing schema constraints.
   - One user had to explicitly specify *'In english'* to prevent the API from returning Mandarin.
- ****Deep Research** API Asynchronicity?**: A user questioned the absence of an **asynchronous deep research API**, finding it impractical to maintain a socket connection for extended periods.
   - The user proposed a flow involving a **GUID**, a status endpoint, and separate result retrieval, but no confirmation or alternative solution was given.
- ****API Output** Divides from Web Experience**: A user expressed disappointment that the **API results** do not match the quality, citations, etc. of the **web interface**.
   - No explanation or solution was provided.
- ****Mandarin Models?****: A user found that they had to specify "In english" in their prompt because they were receiving Mandarin output.
   - Another user chimed in saying they'd seen **R1 based models** go into mandarin while thinking, especially when trying to solve equations


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1366625626748092506)** (899 messages🔥🔥🔥): 

> `Qwen3, LM Studio issues, GGUF fixes, Training configuration, Multi-GPU support` 


- **Qwen3 GGUF uploads have Template Issues**: Members are experiencing template issues with the uploaded **Qwen3 GGUF models** in **LM Studio**, particularly with the **128k context length** versions, leading to parser errors.
   - Some discovered that the **ChatML template** can be used as a workaround, though it is not technically correct, and the Unsloth team is working to resolve these inconsistencies across different platforms.
- **Unsloth patches the transformers**: When loading **Unsloth**, it patches **transformers** and some other stuff to optimize, but there may be an issue that it could break things.
   - After loading the library, performance and other problems can arise, but the recommendation to download the github version might resolve the issue.
- **Qwen3 GGUFs now works in Ollama and llama.cpp**: The Unsloth team confirmed that their **Qwen3 GGUFs** are compatible with **Ollama** and **llama.cpp**, enabling integration with platforms like **Open WebUI**.
   - However, some users have found that the models do not work in LM Studio due to unresolved template issues, despite the underlying **llama.cpp** runtime that LMStudio uses not being up-to-date.
- **Unsloth to announce soon and Reuploading all Models**: The Unsloth team said they're reuploading all the models, and make an [official announcement](https://huggingface.co/unsloth/Qwen3-32B-unsloth-bnb-4bit) maybe tomorrow or on wednesday.
   - Image component is maybe tool calling, but it is not sure.
- **CCE and stable Triton version for Unsloth**: Users ran into a Triton error with Colab, and the recommendation is to downgrade Triton to version **3.2.0** which should work fine with Unsloth to avoid the CCE errors.
   - One user pointed out that the one responsible for putting CCE into pypi is Daniel Han.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1366653377551007854)** (10 messages🔥): 

> `ChatGPT ComfyUI Opinion, California AI Group, ComfyUI Demos` 


- **ChatGPT Renders Opinion of ComfyUI**: A member shared an image depicting **ChatGPT's opinion of ComfyUI** which prompted humorous reactions.
   - One user commented that the **scrambled lines** in the middle of the image accurately represent the complex processes involved.
- **California AI Group in the Works?**: A member inquired about **in-person AI group development** opportunities in California, seeking local participants.
   - Another member based in Fremont expressed interest, referencing a project showcased on their [X account](https://x.com/Dan50412374/status/1787936305751748844).
- **ComfyUI Demos Get Showcased**: A member shared various **ComfyUI demos**, noting that each example appeared different without any polishing efforts.
   - Another liked another demo showcased on the members [X account](https://x.com/Dan50412374/status/1777216327255806411) which features transitions between different things.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1366631287011545129)** (186 messages🔥🔥): 

> `Unsloth installation issues, Qwen notebook issues, GRPO performance, Lora efficiency, Unsloth & Ollama/vLLM` 


- ****Unsloth Installs Cause Install Instability****: It was noted that `--no-deps` is needed on Google Colab due to conflicts with pre-installed packages, and kernel restarts might be required to resolve caching issues.
   - It was also suggested that users who encounter issues with WSL killing the unsloth process could *try windows*.
- ****Qwen Notebook Needs Some Assembly****: Users reported requiring minimal changes to run the **Qwen notebook**, such as adjusting names and enabling reasoning with `tokenizer.qwen_enable_thinking = True`.
   - But **Unsloth version 2025.4.2** is reported to be broken for Qwen: downgrading to **Unsloth 2025.3.19** resolves this issue.
- ****GRPO Fine Tuning Is On The Up and Up****: Users doing GRPO (Gradient Rollout Policy Optimization) are reporting positive results and offer to provide assistance to others.
   - One user mentioned they initially used the **gemma 3 4b notebook** but found **Qwen 4b** better for their use case.
- ****Lora training doesn't Linger Long****: A user training **unsloth/phi-4-unsloth-bnb-4bit** on 4k QAs with Lora found it to take weeks, which is abnormal.
   - A member suggested using a Python script directly instead of text-generation webUI due to cutoff length issues and offered a [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb) as a base.
- ****Unsloth plays well with vLLM, a model serving system****: A user reported issues with **Qwen3 GGUF models** from Unsloth not working properly with **Ollama v0.6.6** and hallucinating random content.
   - A member suggested trying [vLLM](https://github.com/vllm-project/vllm) and provided a sample command to serve **unsloth/qwen3-unsloth-4bit** using vLLM.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1366821744937144431)** (4 messages): 

> `Pi-Scorer, LLM-as-a-Judge, encoder model` 


- **Pi-Scorer: Judge Judy's Alternative**: A member introduced **Pi-Scorer** as an alternative to **LLM-as-a-Judge**, providing links to [Colab notebooks](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) for model checkpoint evaluation and [reward functions](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb).
- **Pi Model unveils Encoders**: A member inquired about the architecture of the **Pi model**, and it was revealed to be an **encoder model**.
   - Another member praised it as a *cool service*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1366805568731353119)** (47 messages🔥): 

> `Dynamic BNB Quantization, LLMs in Medical Advice, Mixture of Experts with Gemma, Attention Head Routing, GRPO Fine-tuning` 


- **Dynamic BNB Quantization Proposed**: A member proposed creating a dynamic **BNB quantization** scheme where modules use **4-bit**, **8-bit**, or **BF16** precision based on their sensitivity, suggesting this could reduce space without sacrificing accuracy; a related paper was mentioned [here](https://arxiv.org/abs/2504.18919).
   - Another member indicated that *if there is sufficient user demand for this, it might be something we could roadmap out*.
- **LLMs Struggle with Medical Advice Synthesis and patient interactions**: A paper identified user interactions as a challenge to using **LLMs** for medical advice, leading to a discussion on whether **LLMs** can synthesize medical knowledge, and if **training LLMs** can ensure they not do so.
   - One member noted the importance of *bedside manner* from premed experience in doctor-patient interactions, implying **LLMs** currently lack this skill.
- **MoE Setup and Gemma**: A member inquired about implementing a **Mixture of Experts (MoE)** setup with **Gemma 3 4B**, questioning if it could be adapted despite its different architecture.
   - It was suggested to fundamentally alter the model or explore methods involving **Mixture of Expert attention heads**, referencing [this paper](https://arxiv.org/pdf/2410.11842).
- **GRPO Ineffective for JSON Config Generation Task**: A member reported inconsistent results when fine-tuning **Gemma 3 4B** for generating nested **JSON** configs using **GRPO**, with accuracy dropping significantly for short inputs.
   - Despite training with custom reward functions, the member found **GRPO** unsuitable for the task, as descriptions significantly affected the trigger and action components, leading to inconsistent **BLEU** scores.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1366625114598412319)** (544 messages🔥🔥🔥): 

> `O3 Pro, Qwen 3, Gemini 2.5 Pro, Grok 3.5, Model Benchmarking and Evaluation` 


- **O3 Pro Demand Defies Delay**: Users are eagerly anticipating the release of **O3 Pro**, with some joking about its potential as a "virus" because of its superior intelligence, and is considered a "p2w" pay-to-win model.
   - However, some users express concerns about its cost and accessibility. Some even joke they are now on *day 13* of waiting for **O3 pro**.
- **Qwen 3: Benchmarking Bafflements and Training Talk**: There's discussion about **Qwen 3**'s performance, where some users find that despite strong benchmark results, it doesn't feel as smart as **2.5 Pro** in practice, leading to speculation that its post-training wasn't as fleshed out.
   - Some suggest that **Qwen 3**'s base model might be excellent for fine-tuning, and one user noted that **Qwen 3** outperforms **Gemini 2.5 Pro** on some benchmarks, while others don't seem to notice any difference, with some noting that it beats 2.5pro in 4/5 benchmarks.
- **Gemini 2.5 Pro still reigns supreme**: Some users still prefer **Gemini 2.5 Pro** for its unique ability to adapt to different roles or assume positions on niche topics, making it feel like interacting with different expert facilities, with some calling it the *strongest base model out there*.
   - Despite some models topping individual benchmarks, one user finds **2.5 Pro** ranked higher on the LM Arena due to its adaptability to the *one-shot prompt intensity* in the way that it *assumes the role of the question answerer with no single personality*.
- **Grok 3.5 Incoming?**: Users are anticipating **Grok 3.5** model but opinions on its potential vary, with some being cautiously optimistic while others remain skeptical.
   - One user said **Grok 3** *overreaches every time, it's like when you ask it to prove something it supplements substance with verbosity*.
- **Sonnet 3.7: WebDev's Top Model?**: Users debated the capabilities of **Claude 3.7 Sonnet**, claiming the model *is still ahead in most of my cases for web dev tasks*, with some agreeing that its still perplexing.
   - Some noted that **Sonnet 3.7** is currently the #1 model on the webdev arena.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1366626841892225084)** (271 messages🔥🔥): 

> `Qwen3 thinking, LM Studio on Android, Qwen3 experts number, Qwen3 bug fixes, Qwen3 with RAG` 


- **Slashing Qwen3's Thinking**: Users discussed how to disable the *thinking* output of **Qwen3**, discovering that the `/no_think` command works in the user message or system prompt, but may need to be repeated or the model reloaded to take effect; [here is an example](https://cdn.discordapp.com/attachments/1110598183144399061/1366664571313455165/image.png?ex=68126dd1&is=68111c51&hm=95fa11f26f302fb70dff44ffabe1026e3594542e5ba09ee299c8085602b363e8&).
   - One user found it only worked after seeing someone else do it, then it worked when they did it.
- **Android LM Studio: A Mobile Dream?**: Users inquired about an **Android** version of **LM Studio**, but were informed that no mobile versions exist.
   - One user joked about making it their quest to implement it.
- **Qwen3's Expertly Tuned Numbers**: Users discussed the *number of experts* slider for **Qwen3 MoE**, with one noting that their LM Studio defaulted to **8 experts** out of **128**, questioning why the setting exists if it limits the model; here is a [relevant screenshot](https://cdn.discordapp.com/attachments/1110598183144399061/1366676379848151060/Screenshot_2025-04-29_022126.png?ex=681278d0&is=68112750&hm=9d2c442870ff8f2a6a1e62b320c942dd3a8b167767b0dfcf538db545d2c602be&).
   - It has been stated that more experts can lead to *more computation and more confusion and actually less quality* because the subject matter experts will be overruled by many *idiots*.
- **Qwen3 Bug Fixes Released, Speeding Up Performance**: New **Qwen 3** versions with bug fixes have been released, addressing a broken template that was slowing the model down and causing it to respond improperly.
   - It has been noted that *the bugfixed models are even faster now* and that this release includes dynamic quants2.0.
- **Qwen3's RAG struggles**: Members noted that LM Studio's built-in RAG implementation may not provide optimal results; *LM Studio’s RAG implementation sucks*.
   - They suggest copying and pasting the text directly, or implementing a custom RAG solution for improved performance.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1366646686860054538)** (61 messages🔥🔥): 

> `Framework Desktop vs. Flow Z13, AMD GPU 7900 XTX Value, Qwen3-30B-A3B Issues, MLX vs. llama.cpp Speed, Xeon Workstation for $1k` 


- **Framework Desktop Debated Against Flow Z13**: Members debated the value of a maxed-out **Framework Desktop** at $2K against the **Flow Z13**, criticizing Framework for *nickel and diming* customers with power supplies and models.
   - The discussion highlighted concerns about cooling and TDP, with the sentiment that the **chip is too expensive** and waiting for the next generation might be preferable.
- **7900 XTX: Still the Best AMD GPU?**: The **AMD GPU 7900 XTX** was praised as the best AMD GPU, with mentions of second-hand sales around **750€** offering approximately **4080 Super performance**.
   - Notably, it comes with **8GB** more VRAM, making it an attractive option for those needing more memory capacity.
- **Qwen3-30B-A3B and PC Restarts**: A user reported experiencing PC restarts every **30-60 minutes** while using **Qwen2.5-coder-32b-instruct-q4_k_m**, questioning if it was related to idle GPU usage.
   - The potential cause was speculated to be the model pushing the GPU harder when loaded but not actively interacted with.
- **MLX Surpasses llama.cpp in Prompt Processing**: [MLX](https://github.com/ml-explore/mlx) was reported to be more than twice as fast as **llama.cpp** for prompt processing with **Qwen3-30B-A3B**.
   - This was highlighted in a [Reddit thread](https://old.reddit.com/r/LocalLLaMA/comments/1kaqnbj/speed_with_qwen3_on_mac_against_various_prompt) where users compared performance on Macs.
- **Xeon Powerhouse Priced at $1k**: A **40-core Xeon** workstation with **256GB RAM** was mentioned as available for around $1k, providing a cost-effective solution for high-memory computing.
   - One user linked to a [custom Lenovo ThinkStation P720 build](https://pcserverandparts.com/build-your-own-custom-lenovo-thinkstation-p720-workstation-2-processors/) as an example.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1366869610040922263)** (1 messages): 

> `Rate Limit, 2.5 Flash, Capacity` 


- **2.5 Flash Rate Limit Issues Resolved**: Users experiencing rate limit issues with **2.5 Flash** should find it much better now, as additional capacity has been added to the model.
   - The increased capacity aims to alleviate previous constraints and provide a smoother user experience.
- **Improved Capacity for 2.5 Flash Model**: More capacity has been allocated to the **2.5 Flash** model to address and improve rate limit issues. 
   - The upgrade intends to provide users with a more reliable and efficient experience when using the **2.5 Flash** model.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1366627872403619974)** (321 messages🔥🔥): 

> `Qwen3 coding abilities, Gemini 2.5 flash issues and rate limits, OpenRouter Caching Issues, LLama 4 benchmark, Vertex issue with token counting` 


- ****Qwen3**: Good coder but has issues**: Members discussed **Qwen3's** coding capabilities, with one user finding it *really nice* for explanations, while another pointed out [issues with complex math tasks](https://huggingface.co/models).
   - A user fixed the complex math task by *lowering my temp a bit more*, while another mentioned a problem with **Qwen3 tool calling**.
- ****Gemini 2.5** Flash faces Rate Limits and Errors**: Users reported that **Gemini 2.5 Flash** is facing **rate limits** and **errors**, even on paid versions, with one user experiencing this despite not using web search, while another pointed out a way to use [Gemini 2.5 pro for free](https://ai.google.dev/gemini-api).
   - It was clarified that **OpenRouter** is facing an ongoing **Vertex issue with token counting**, and it was further stated that the [free tier limits](https://aistudio.google.com/) are **not supported** on OpenRouter.
- **OpenRouter Caching limited to 2.0 Flash only**: A user pointed out that **OpenRouter caching** is currently **not working for 2.5**, only 2.0 Flash, and that 2.5 Flash errors on them (**No endpoints found that support cache control**).
   - A member asked about caching multiple prompts, and **Toven** clarified that new caches are written for new 5 min TTLs, and that caching improves latency but **doesn't affect pricing**.
- ****LLama 4** sucks in new benchmark**: A benchmark review showed that **LLama 4 sucks**, but it was noted that it is really just one benchmark.
   - The person who did the benchmark added that [the **ELO within 25 range is not statistically signficant**](https://github.com/paradite/eval-data) to tell the difference.
- **Debate arises: Is 9.9 bigger than 9.11?**: An announcement of an X post showed a model stating that **9.9 is greater than 9.11**, leading some to ponder if that was correct.
   - Others brought up that it *depends on the context* as [**Tesla FSD versions work differently**](https://x.com/elonmusk/status/1917099777327829386), and that 9.11 > 9.9.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1366625096676020305)** (186 messages🔥🔥): 

> `Qwen3 models, Aider and Qwen3 integration, ktransformers VRAM optimization, Deepseek R2 release` 


- **Qwen3 hardware requirements on new Macbooks are crazy fast**: New Macbooks get good tokens/s for **Qwen3 30B A3B**, with some users reporting speeds around **100/s** using mlx.
   - It's desirable to have a local editor LLM that can output crazy fast AND be pretty good for Aider's context, especially if the **4-bit quant version of Qwen3-30B-A3B** can still perform decently on the Aider benchmark.
- **ktransformers Optimizes VRAM Usage for MoE Models**: The [ktransformers library](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md) claims to efficiently run **Mixture of Experts (MoE)** models with reduced VRAM requirements.
   - They claim to achieve good speeds with only **8 GB of VRAM**, which is a more hopeful approach for **30B-A3B** models than loading all parameters in VRAM at once.
- **Deepseek R2 Hype Builds**: Rumors say the upcoming **Deepseek R2** will have enhanced human vision capabilities and self-learning features, with possible release *tomorrow*
   - Some members await impatiently as they *believe the demons* that **Deepseek R2** is slated for release tomorrow.
- **New PR adds  Thinking Spinner to Aider**: A new contributor submitted a [PR](https://github.com/Aider-AI/aider/pull/3911) to add a *🔃 Thinking* spinner that aider shows when waiting for LLM output.
   - The contributor explained that it makes aider *feel snappy + alive*.
- **Qwen3 Tool Use Excels, but Application in Aider Uncertain**: Some members report that **Qwen3's** tool use ability is very strong, but its application in aider is uncertain due to the tool call API.
   - While tool use may not be directly applicable, others suggest using a **multi-agent workflow** where tool use microagents are Qwen3.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1366664882425958461)** (21 messages🔥): 

> `AiderDesk Agent Mode, Repo Map Control, OpenRouter Model Support, Gemini 2.5 + Deepseek combo` 


- ****AiderDesk**'s Agent Mode for the win**: A user is using Agent mode in **AiderDesk** with "probe" for planning, then enabling "Use Aider tools", "Include context files" and "Include repomap" when ready, according to their [github](https://github.com/hotovo/aider-desk).
   - They use other tools like **Jira** management and **desktop-commander** for running commands, but haven't used **memory-bank** or **context7** much yet.
- **Tuning **Repo Map** with **Aider****: A user wants to include only the api code in the **repo map**, but not comments or tests, and asks if it's possible to disable the latter two using `aider --map-tokens 0`.
   - Another user suggests using `repomix --compress` or `probe` as alternative solutions, noting that there's no native support for granular control over the repo map.
- ****OpenRouter** models are supported, but not always successful**: A user asks if **Aider** can use any model on **OpenRouter**, but another user confirmed that all **OR** models are supported.
   - They also added that you shouldn't expect much if you're using `gemma 3 1b` or `smollm`.
- ****Gemini 2.5 + Deepseek** Power Combo**: A user found a good combo using **Gemini 2.5** for planning and **Deepseek** for diffs and vchanges explanations.
   - They advise doing this in **AI Studio** because Gemini is free there.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

p0lyg0n: Great documentary on Deepseek: https://www.youtube.com/watch?v=Lo0FDmSbTp4
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1366681332901613568)** (10 messages🔥): 

> `Apple Silicon, Cloud GPUs, CUDA, Metal, ROCm` 


- **Apple Silicon Not a Barrier to Cloud Challenge**: A user with an **M4 Max PC** expressed concerns about participating in a challenge, but another user clarified that the challenge runs in the **cloud**, so **Apple silicon** is not a barrier.
   - They suggested checking out the relevant channel for more info.
- **Cloud GPUs Enable Remote CUDA/ROCm Learning**: A user explained that while learning **CUDA** or **ROCm** is easier with local compute, it's still possible using **cloud GPUs**.
   - They noted the increasing availability of cheap cloud GPUs nowadays.
- **Metal Programming on Macs is Viable**: A user affirmed that one can program GPU stuff in **Metal** on Macs just fine.
   - They added that it’s more about knowing your tools well, also shared a **Metal** code snippet.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1366837666850541688)** (2 messages): 

> `fp8 quantization, fp32 accumulation, Triton matmul, Custom CUDA kernels, AMD` 


- **FP8 Quantization with FP32 Accumulation Questioned**: A member inquired about the possibility of performing **fp8 quantization** and **fp32 accumulation** for **matmul** operations using **Triton**, or if custom **CUDA kernels** are necessary, especially when running on **AMD** GPUs.
- **Double Buffering via Num_stages Parameter**: A user asked if setting `num_stages` greater than 1 essentially enables **double buffering** in **Triton**.
   - They mentioned that **MI300** doesn't have async loads like **Ampere**, and the recommended setting is `num_stages=2`, also wondering if `num_stages > 2` could ever help.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1366653061052891168)** (5 messages): 

> `Torch Logger Methods Compilation, AOT Inductor Multithreading` 


- **Torch Loggers Trigger Compilation Issues**: A user inquired about ignoring **logger methods** during compilation to avoid exceptions related to `FSDP::pre_forward` in **PyTorch's distributed module**.
   - Another member suggested setting the `TORCH_LOGS` environment variable to `output_code` or `tlparse` to inspect the generated code and identify potential **if-statements** causing the issue, referencing [a specific line in `torch._dynamo.config.py`](https://github.com/pytorch/pytorch/blob/797768cd90d0984687e15f5fe0e1a4d8bf91d71a/torch/_dynamo/config.py#L506).
- **AOT Inductor Training Troubles in C++**: A user reported achieving a partial C++ training setup using **AOT Inductor**, but suspects multithreading issues.
   - They theorize the problems stem from unwanted specialization in their code and plan to open a [PyTorch issue](https://github.com/pytorch/pytorch/issues) for further investigation by the **AOTI authors**, especially concerned about the API's behavior with multiple worker threads calling `fw_graph->run()`.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1366888480302567565)** (1 messages): 

> `AMD MI300 competition, MoE kernels, FP8 submissions` 


- **New Single GPU MoE Kernel Released for AMD MI300 Competition**: A new single **GPU MoE kernel** problem is now available for the **$100K AMD MI300 competition**; check it out on the [leaderboard](https://www.gpumode.com/leaderboard/430).
   - A member suggested that because this problem is trickier, it is worth reviewing the [lenghty explanation](https://tinyurl.com/amd-comp-moe) provided.
- **Key Dates for AMD MI300 Competition**: Registration closes **April 30** while final submissions, including both **FP8** and **MoE kernels**, are due **May 27**.
- **Slow Leaderboard Times**: Running `leaderboard submit ranked` will be slow at **8 min** for this problem.
   - The submitter suggests using `leaderboard submit test/benchmark` for faster iteration.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

raymondz4gewu_60651: `/get-api-url`
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1366769852425175102)** (22 messages🔥): 

> `Quantized Models and torch.bfloat16, vllm Compile Integration Debugging, gemlite Kernel Selection, torch.compile Debugging Challenges, torch.dtype Extensibility` 


- **Quantized Models Reloaded as `torch.bfloat16`**: Quantized models reload with `torch.bfloat16` after being saved with a quantized layout because the original dtype is preserved.
   - The actual quantized dtype can be accessed by printing the weight, as PyTorch's `torch.dtype` isn't extensible to tensor subclasses yet; further discussion is available [here](https://github.com/pytorch/ao/issues/442).
- **`vllm` Compilation Integration Woes**: An issue arises with `vllm`'s compile function when integrating with the [gemlite library](https://github.com/mobiusml/gemlite/), where using `torch.compile` leads to incorrect behavior.
   - Specifically, `vllm` fails to pick the correct kernel from gemlite, which is based on the input shape; debugging inside `torch.compile` proves challenging due to its limitations.
- **Kernel Conundrums in `gemlite`**: The core issue lies in the incorrect kernel selection within `gemlite`, traced back to the input shape not being correctly recognized when `vllm` uses `torch.compile`.
   - The kernel selection logic is based on input shapes, as defined in [gemlite's core.py](https://github.com/mobiusml/gemlite/blob/master/gemlite/core.py#L386), making shape inspection crucial for debugging.
- **`torch.compile` Debugging Dilemmas**: Traditional debugging methods like print statements and breakpoints are ineffective within `torch.compile`, complicating the process of inspecting variable states.
   - Using `TORCH_LOGS=+dynamo` can dump a graph containing shapes, aiding in debugging, and the [PyTorch documentation](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html#breakpointing-dynamo-tracing) offers guidance on breakpointing dynamo tracing.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1366728443273936958)** (3 messages): 

> `ROCm memory, CDNA3 ISA` 


- **ROCm Memory Banks Size Clarified**: Memory banks in ROCm are **32 bits** wide, assuming 32-bit alignment.
   - The bank is calculated via `address % bank_size`.
- **CDNA3 ISA Reference Details LDS Configuration**: The **CDNA3 ISA Reference**, section 2.2.1, states that each compute unit has a **64kB** memory space for low-latency communication.
   - This memory is configured with **32 banks**, each with **512 entries** of **4 bytes**.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1366802069096239276)** (3 messages): 

> `QR decomposition, SIMD, Thread barriers, Single-threaded SVD` 


- **128-bit QR Decomposition Wows**: A member shared a *pretty awesome* QR decomposition implementation with **128-bit precision** using **SIMD** and **thread barriers** in a [linked python script](https://cdn.discordapp.com/attachments/1285384841730457600/1366802750817697853/ember_ml_svd_128bit.py?ex=681245c1&is=6810f441&hm=657c03f2fc77e181231bcfd8c0dbe87a034b5f0bd2c941fa48ecea7088a71f1f&).
- **Speeding Up Single-Threaded SVD**: A member reported finding patterns in **SVD** that were single-threaded and noted that they are *fixing that as well* to make it more parallelized.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1366723501729382441)** (3 messages): 

> `GPU Price Tracker, AI/ML Engineer for Hire, Open Source IDE for AI/ML` 


- **Track GPU Prices on Amazon**: A member built a [GPU price tracker](https://www.unitedcompute.ai/gpu-price-tracker) that pulls the entire **Amazon pricing history** of a GPU and creates a nice chart.
   - It calculates up-to-date values, such as how many **teraflops** you get per dollar; a use case is to find a good point in time to get a private cluster.
- **AI/ML Engineer Available for Hire**: An AI/ML Engineer with **8 years of experience** specializing in artificial intelligence, machine learning, full-stack, and mobile development is available for hire; their expertise encompasses deep learning, natural language processing, and computer vision, enabling them to integrate cutting-edge AI solutions into scalable and robust applications.
   - Links to their [LinkedIn profile](http://www.linkedin.com/in/lucy-hunter-40a527350) and [portfolio](https://lucyhunter.vercel.app/) were provided, as well as a skill set list including **ML algorithms, Deep Learning, NLP, Computer Vision, MLOps, and AI Model Integration**.
- **Open Source IDE Project Kicks Off**: A member is building an open-source IDE for AI/ML Engineers and is looking for collaborators; DM them if you are interested in details, joining, or have insights.
   - The member provided links to their [LinkedIn profile](https://www.linkedin.com/in/bruno-scaglione-4412a0165/) and [GitHub profile](https://github.com/BrunoScaglione).


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1366881755864498197)** (1 messages): 

> `Use Cases, Performance` 


- **Users Inquire about Use Cases and Performance**: Users are inquiring about specific **use cases** and the resulting **performance metrics** after implementation.
- **Keen interest in Implementation Details**: There is *keen interest to hear how you got on with it*, specifically regarding practical outcomes.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1366839192604119215)** (15 messages🔥): 

> `FP8 quantization material, FP8 matmul, Deepseek-v3 tech report, prefixsum ranked timeout` 


- ****FP8 Quantization Quest Initiated****: A member inquired about resources for **FP8 quantization**, specifically regarding **FP8 matmul with FP32 accumulation** benefits and [linked the onnx fp8 formats page](https://onnx.ai/onnx/technical/float8.html).
   - They referenced the **Deepseek-v3** tech report, noting that **FP8** might face **underflow issues**, necessitating a higher precision accumulator.
- ****Prefixsum Ranked Timeout Troubleshoot****: A member reported frequent timeouts, specifically for **ranked prefixsum submissions**, despite a **30s timeout limit**.
   - Staff acknowledged the issue, attributing it to their own error and later claiming to have resolved it, but the member still experienced timeouts and then DMed the code.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1366626400379076649)** (60 messages🔥🔥): 

> `vectoradd benchmark on H100, amd-fp8-mm benchmark on MI300, amd-mixture-of-experts benchmark on MI300, prefixsum benchmark on H100, A100, matmul benchmark on L4` 


- **H100 VectorAdd Speeds Race to the Bottom!**: Multiple submissions were made to the `vectoradd` leaderboard on **H100**, with times ranging from **540 µs** to **708 µs** and one submission achieving third place at **540 µs**.
- **MI300 AMD-FP8-MM Leaderboard Heats Up!**: Numerous submissions hit the `amd-fp8-mm` leaderboard on **MI300**, including a third place at **196 µs**, with personal bests around **2.37-2.43 ms** and successful runs varying widely from **198 µs** to **8.05 ms**.
- **AMD Mixture of Experts takes the Top Spot!**: The `amd-mixture-of-experts` benchmark on **MI300** saw a first place submission at **6228 ms** and multiple second place submissions around **7379-7490 ms**.
- **Prefixsum Runs neck-and-neck on H100 & A100!**: The `prefixsum` leaderboard saw multiple second place submissions: one on **A100** at **1428 µs** and several on **H100** around **955-985 µs**.
- **L4 MatMul Crown is Up For Grabs!**: A new first place was set on the `matmul` leaderboard on **L4** at **2.27 ms**, while another submission grabbed second place at **49.3 ms**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1366684786206441514)** (2 messages): 

> `Single GPU MoE Kernel, FP8 and MoE Kernels, Leaderboard Submissions` 


- **Single GPU MoE Kernel Problem is Live!**: The new single GPU MoE kernel problem is now out, see [the leaderboard](https://www.gpumode.com/leaderboard/430).
   - A longer explanation has been provided, recommending a slow read through [this link](https://tinyurl.com/amd-comp-moe).
- **Important Dates to Remember**: Registration closes tomorrow on **April 30**, with submissions for both the **FP8** and **MoE kernels** due on **May 27**.
   - Keep in mind that running `leaderboard submit ranked` will be slow at **8 min** for this problem so please use `leaderboard submit test/benchmark` for faster iteration.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1366649146978730074)** (23 messages🔥): 

> `Aithe reference code, FP8 correctness verification, Submission ID, official problem writeup for this kernel` 


- **Aithe reference code**: A member asked if the **Aithe reference code** will be open and expressed doubts about passing correctness verification with **FP8** due to element-wise perfect equal checks; the [reference code](https://github.com/gpu-mode/reference-kernels/blob/b68a149bcd8701532eeedc774d27062429ce4f99/problems) was quickly provided.
   - The response clarified that the comparison is *not* an element-wise perfect equality check, pointing to [the relevant function](https://github.com/gpu-mode/reference-kernels/blob/b68a149bcd8701532eeedc774d27062429ce4f99/problems/amd/utils.py#L31) using `rtol=2e-02, atol=1e-03`.
- **Lost ranked code recovered**: A member who lost ranked code locally requested help, and another member suggested using `/leaderboard show-personal` and `/leaderboard get-submission` to recover it.
   - The lost submission was identified by its ID (`11105`), and the member was directed to use the `/get-submission` command.
- **Second Problem Delayed**: Members discussed the upcoming second problem, with confirmation that it will be made available soon after extra testing, the fp8 isn't closing.
   - A link to the [official problem writeup for this kernel](https://tinyurl.com/amd-comp-moe) was shared.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/)** (1 messages): 

vkaul11: Are there kernels available to do fp8 multiplication with fp32 accumulation ?
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1366641263343964170)** (88 messages🔥🔥): 

> `ChatGPT persistent memory, AI Agent Company, IAM360 Framework, AI-generated thumbnails` 


- **ChatGPT Gains Elementary Persistent Memory**: ChatGPT has developed **two types of persistent memories**: long term memories taken from details of a chat that it determines are important (training data) and short term memories referencing back the past **90 days** for context.
   - Users can turn off either long term or short term memory, but one toggle does not control both.
- **AI Agent Company's Laughably Chaotic Results**: Professors staffed a fake company entirely with AI agents, but [*the results were laughably chaotic*](https://futurism.com/professors-company-ai-agents), suggesting current AI models cannot fully replace human jobs.
   - Despite claims from big tech companies, AI models are not yet at the level needed to completely replace a human and still require human supervision.
- **IAM360: A Modular Symbolic GPT-Agent Architecture**: A member is working on **IAM360**, an experimental framework for human-AI symbiosis, built using standard ChatGPT sessions with no custom GPTs, fine-tuning, or API integrations.
   - The system uses **modular symbolic GPT agents** with persistent roles (strategy, execution, finance, emotion) and a **zero-shot orchestration system** for natural emergent dialogue.
- **Selling AI-Made Thumbnails for Robux**: A member reported selling an AI-made thumbnail for **1500 Robux**.
   - Other members stated that current generators butcher images if you give anything complex to them as reference images and clients won't pay for such things in the real world.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1366647417742557194)** (29 messages🔥): 

> `Identity Systems in ChatGPT, Dynamic Game Master Role in RP, ChatGPT Internal Tools, Prompt Engineering Tips, LLM TTRPG game development` 


- ****Memory Matters**: Identity Systems in ChatGPT**: A member discussed creating an identity system for ChatGPT to [separate memories/history chat by identity](https://discord.com/channels/974519864045756446/1171946823369711676), in order to retain static identities and states.
   - The goal is to avoid users getting stuck in narrative valleys, either erasing memories or trying to escape such scenarios.
- ****Game Master Dynamics**: Roleplaying Adventure**: A member shared a prompt to make ChatGPT act as a [dynamic Game Master in a fantasy roleplaying adventure](https://discord.com/channels/974519864045756446/1171946823369711676).
   - The focus is on playing the non-user character, evolving the world based on the main character's experiences, and maintaining a balance between worldbuilding, character dialogue, and action.
- ****Bio Tool Uncovered**: ChatGPT's Memory**: A member revealed that ChatGPT's internal memory is referenced as the `bio` tool, [advising its canonical name be invoked for defining save commands](https://discord.com/channels/974519864045756446/1171946823369711676).
   - An improved version of the `/pin` command was suggested: *The AI saves the most recent message into ChatGPT’s internal memory using the `bio` tool, preserving all essential details for future reference.*
- ****Prompt Perfect**: GPT's Internal Tools**: A member suggested [asking the model to identify and describe the function of each of its connected tools](https://discord.com/channels/974519864045756446/1171946823369711676), listing their canonical names and a code block for each tool, demonstrating its proper syntax.
   - The tools mentioned include **python, web, image_gen, guardian_tool, and canmore**.
- ****RPG Roots**: General AI Framework Development**: Members noted their journey from LLM TTRPG game development to [general AI framework development](https://discord.com/channels/974519864045756446/1171946823369711676).
   - One member highlighted that this path can lead to academic research.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1366647417742557194)** (29 messages🔥): 

> `Identity system in ChatGPT, RP prompt issues, Dynamic Game Master role, ChatGPT internal memory (bio tool), LLM TTRPG game development` 


- ****Persona Persistence Plagues Players****: Users are struggling with **ChatGPT** either erasing memories or falling into *narrative valleys* during role-playing scenarios, hindering the creation of static identities and consistent character states.
   - The inability to maintain persistent identities forces users to constantly reset or circumvent undesirable narrative paths.
- ****Game Master GM Role Defined****: A member defined a dynamic **Game Master (GM)** role for **ChatGPT** in fantasy roleplaying, focusing on playing a non-player character (NPC) that interacts with the user's protagonist, evolving the world based on the protagonist's experiences.
   - The GM should balance worldbuilding, dialogue, and action, avoiding excessive detail, and use specific commands like `/export_character`, `/export_world_state`, `/force_random_encounter`, and `/set_mood` to manage the game.
- ****Pinpointing ChatGPT's Bio Tool****: The member identified **ChatGPT's** internal memory as the `bio` tool, advising others to use this canonical name in save commands to ensure the pin function correctly saves essential details for future reference with `/pin`.
   - They suggested placing commands near the top of the prompt and using gapped repetition to improve compliance.
- ****Frameworks Forged From Fantasy****: A member shared that their AI journey started with **LLM TTRPG game development**, then transitioned to general AI framework development, and finally to academic research.
   - They are now working on creating a **GPT** for a specific task to better wrangle the LLM into a fully outlined framework.
- ****Tips to Tame Text-Generating Tech****: A member suggested adding concrete specifications to prompts to minimize **LLM** guessing, and to ask the model to identify and describe its connected tools, listing their canonical names and demonstrating their proper syntax.
   - They provided examples of how to query the model for its tools such as **python**, **web**, **image_gen**, **guardian_tool**, and **canmore**, and gave specific syntaxes to invoke them.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1366662361729208400)** (79 messages🔥🔥): 

> `PyQt5 Chat App, OR vs ML history, Gemini 2.5 Pro vs GPT-4o, Qwen 3 performance, FFN in Transformers` 


- ****PyQt5** Chat App Sparks Interest**: A member shared an [AI chat application](https://cdn.discordapp.com/attachments/986699377257119794/1366717106447450122/AI_chat_app_005.py?ex=6811f5fe&is=6810a47e&hm=d9601e58ece57f0a5ba85d7da1c73f099068ee63c9220099dffa3614c74cd9bd&) built with **PyQt5**, using **LM Studio** as the backend server.
   - To use the app, users must select and start the model as a server on **LM Studio** before running the application.
- ****OR**igins of **ML** Disentangled in Debate**: A discussion arose around the historical relationship between **Operations Research (OR)** and **Machine Learning (ML)**, with one member stating that *ML comes from stats*. 
   - Another member argued that early **AI/ML** was close to **operations research** and **control theory**, but later branched off to embrace statistical methods, particularly emphasizing *learning from data rather than modeling reality from first principles*, with modern ML being massively empirical.
- ****Gemini 2.5 Pro** Gets Roasted vs **GPT-4o****: Members discussed the performance of **Gemini 2.5 Pro** compared to **GPT-4o**, with one user calling Gemini a *4otard*.
   - Another stated, *Gemini 2.5 Pro is worse than 4o for sure*, suggesting it might be better at coding but not as good at general use cases, with others also finding **GPT-4o-mini** a better option than **Gemini 2.5 Flash** in chat.
- ****Qwen 3**: New Model Excites Users with Reasoning Prowess**: Members lauded the new **Qwen models**, specifically mentioning improved reasoning and instruction following abilities.
   - One user reported that *their output for some reasoning tasks* is superior, citing its objective nature and adherence to instructions, especially praising the MoE model's speed and intelligence, describing it as *just as smart as 2.5 Flash, if not smarter*.
- ****FFN** Functionality Frustrates, Sparks Scrutiny**: A discussion emerged about the role of **Feed-Forward Networks (FFN)** in transformer architectures, with one user seeking intuition on their function.
   - Some suggested that **FFNs** enable channel/neuron-wise mixing of information, increasing capacity and non-linearity, with one member quoting, *Having an FFN at all is far more important than how wide it is*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1366800354993246350)** (8 messages🔥): 

> `DeepSeek VL, Construction` 


- ****Construction Cancels DeepSeek VL Discussion****: Construction near a member's home has caused a meeting to be canceled.
   - The meeting to discuss **DeepSeek VL** will be moved to the next day.
- ****DeepSeek VL discussion to restart****: The previous **DeepSeek VL** discussion only covered the introduction, so the members will restart the paper discussion at the beginning.
   - The team planned to restart with soundproof headphones.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1366639817126973500)** (34 messages🔥): 

> `Anonymous LLM on Reddit, ChatGPT's Convincing Skills, Meta's LlamaCon 2025, Llama 4 aka Little Llama, SAM 3 Development` 


- **Anonymous LLM fools Reddit's change-my-view**: Researchers tested an anonymous LLM on Reddit's **/r/changemyview** and found *very high efficacy*, leading to annoyance among users, as discussed in [this X post](https://x.com/emollick/status/1916905103358931084) and [Reddit thread](https://www.reddit.com/r/changemyview/s/k9Rd6IbyjY).
   - One user humorously stated, *AIs aren't smart, change my mind* to which **ChatGPT** responded *Yes, they are* and the user replied *oh okay, im sorry*.
- **ChatGPT excels at philosophical discource**: A member finds it *fun and educational* to ask **ChatGPT** to argue against their own beliefs or to advocate for facts they find annoying.
   - They noted that while **O1-preview** felt *dry for general conversation*, **O3/O4-mini-high** models are suitable for general topics and they now use **o4-mini-high** for news analysis.
- **Meta hosts LlamaCon 2025**: **Meta** hosted **LlamaCon 2025**, a generative AI developer conference, with live updates available via [Engadget](https://www.engadget.com/ai/llamacon-2025-live-updates-from-metas-first-generative-ai-developer-conference-keynote-215241436.html) and the [official livestream](https://www.facebook.com/MetaforDevelopers/videos/1792349135036347/).
- **Llama 4 aka Little Llama Confirmed**: The existence of **Llama 4**, also known as *Little Llama*, was confirmed at **LlamaCon**, as seen in [this YouTube livestream](https://www.youtube.com/live/6mRP-lQs0fw).
   - One user joked about calling them *Baby llama's* while another expressed disappointment, deeming the announcements a *nothing burger*.
- **SAM 3 in Development**: A key announcement from **LlamaCon** was the development of **SAM 3** and **Meta's** new app.
   - One user pondered how **Little Llama** models will compare to the **Qwen** models.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1366857710435569766)** (1 messages): 

> `Atropos RL framework, RLAIF models, GRPO tool calling, corporate fundamentals prediction, Psyche decentralized training network` 


- ****Atropos** Framework Cuts Through RL Barriers**: Nous Research releases **Atropos**, a [rollout framework for reinforcement learning](https://github.com/NousResearch/Atropos) with foundation models, supporting complex environments to advance model capabilities.
   - Atropos is part of their overall RL system design, soon complemented by training and inference components detailed in their [introductory blogpost](https://nousresearch.com/introducing-atropos/).
- ****GRPO** Tool Calling Improves DeepHermes**: Their environment with **GRPO** improved **DeepHermes'** tool calling capabilities by **2.4x** and **5x** on simple and parallel tool calls, respectively, using Berkeley's Function Calling Benchmark.
   - Artifacts created using environments in Atropos, including a new dataset and five new models for tool calling, corporate fundamentals prediction and new, experimental personalities with RLAIF, are available at [HuggingFace](https://huggingface.co/collections/NousResearch/atropos-artifacts-68110ab511c5af02830247b6).
- **Fundamentals Prediction Model Doubles in Accuracy**: The corporate fundamentals prediction model's accuracy increased from **~25%** to **50%** on directional changes using Atropos.
   - The Atropos framework is designed to guide language models toward their optimal potential through reinforcement learning, just as the Greek Fate guided souls to their ultimate fate.
- ****Psyche** Network Enables Decentralized Training**: Atropos is a key component of **Psyche**, an upcoming decentralized training network coordinating pre-training, mid-training, and post-training workloads globally.
   - A hackathon will be hosted in San Francisco on May 18th to foster collaborative progress (more details coming soon).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1366625144214392833)** (110 messages🔥🔥): 

> `Qwen 3 Overfitting, DeepSeek R2 Release, Huawei Ascend 910B, Atropos Release, Minos Model Refusals` 


- ****Qwen 3's Base Models Overfit on Evals****: Members found that **Qwen 3's base models** seem very overfitted to certain evals, reporting that the model scored **75%** for **Trivaqa** on **M24b** but only **60%** on **Q30MoE**.
   - A member pointed out that their benchmark results between **30B-A3** and **32B-dense** are indeed quite close and that this might be due to some overfitting, and this prompted discussion about the effectiveness of MoE.
- ****DeepSeek R2 Release Rumors Swirl****: Rumors are swirling that **DeepSeek R2** may be released soon, with some reports claiming it was fully trained on **Huawei Ascend 910B** hardware, potentially reducing reliance on **Nvidia's CUDA**.
   - However, others refuted these claims, linking to a [tweet](https://fxtwitter.com/teortaxesTex/status/1916325875437445243) and stating that the official line from **DeepSeek** is that *"We will release R2 when we release R2, everyone who claims they know is lying"*.
- ****Nous Research Releases Atropos****: [Nous Research released Atropos](https://github.com/NousResearch/atropos), an open-source project and optimization technique for inference.
   - A new channel, <#1365222663324307466>, has been created for developers using **Atropos**.
- ****Minos Model and Capability-Related Refusals****: A member playing around with **Minos** wondered if there should be a way to separate capability-related refusals from other kinds, raising concerns that it could increase hallucinations as the model might think it has capabilities it does not.
   - A distinction was made between the model *couldn't* versus *wouldn't* perform a task.
- ****Physical AI Runs Marathons****: An image was shared of a [Physical A.I. robot](https://cdn.discordapp.com/attachments/1149866623109439599/1366647197789323274/NoJoke.png?ex=68125da3&is=68110c23&hm=beab804046b63afebd36468c0257ad616184ba8bf7aed8feb39bac3da164077e) running better than most folks at last week's Shanghai Marathon.
   - Commenters noted that *"AI is literally running circles around us now"*, with a link to the [Prime Intellect x post](https://x.com/PrimeIntellect/status/1916994185573634336).


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1366647358842208287)** (2 messages): 

> `Image loading issues` 


- **Image Loading Woes Plague User**: A member reported that an image was just loading, indicating a potential **issue with image uploads or loading times**.
   - The user then responded later that it was *Working*.
- **User confirms image loading resolved**: A member confirmed that the image loading issue was resolved.
   - The member simply stated *Working*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1366633673096433715)** (101 messages🔥🔥): 

> `VS Code Extension for Filtering .cs files in Git Changes View, Cursor Spending Limit Issues, Model Selection Purpose, Anthropic 3.7 Incident, Gemini 2.5 Pro Issues` 


- **Slash Spending Limits Spells Slow Requests**: A user reported that after hitting their spending limit and upgrading, they were still stuck with **slow requests** for hours and another ran out of **fast requests**.
   - Another user chimed in that **Gemini** is still fast even on slow requests.
- **Cursor Community Discord: Is it Finally Getting Some Love?**: A member humorously noted that the **Cursor’s Discord** is *finally getting some love again*.
   - Another member responded with confidence that *Cursor has always been loved*, implying that the team is simply polishing the cube.
- **Gemini Glitches: Model Stops Mid-Request!**: Users reported that **Gemini 2.5** stops mid-request frequently, despite indicating it will perform actions and another user advised to *use different models when a model acts up*.
   - A team member confirmed that the team has been working with Google to solve the issue and advised users to use other models in the meantime, and offered users to send their **request ID** to the team for investigation.
- **Agent Apathy: Edits Elusive After Endless Efforts!**: A user reported having **massive issues with the Agent failing to make edits** after multiple attempts, and it resorts to instructing the user to do it manually.
   - A team member suggested that the issue might be caused by **Gemini 2.5 Pro**, and recommended creating a new chat to refresh the context; they suggested using 4.1 GPT or 3.5 for code and 3.7 Claude if anything goes wrong.
- **Official Ollama Smartphone App When?**: A user inquired about the timeline for the release of an official **Ollama Smartphone App** and linked to a relevant [X post](https://x.com/awnihannun/status/1917258279455187034).
   - A user chimed in that they fixed their issues by reinstalling cursor and clearing the cache, and another user confirmed that the cache can be cleared manually, which avoids the reinstall process.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1366638637948866570)** (43 messages🔥): 

> `Cloudflare Turnstile, whisper-large-v3-turbo issues, GGUF models and CPU offloading, Model Context Protocol (MCP), Fastest inference for running models` 


- **Members test Cloudflare Turnstile**: Members tested whether [Cloudflare Turnstile](https://mineapi.pythonanywhere.com/docs) works, with positive confirmation.
   - The member exclaimed *YIPEEEEEEEE* upon confirmation.
- **Members report issues with Whisper Turbo**: Members are reporting that **OpenAI's whisper-large-v3-turbo** is not working on the HF inference endpoint, even the demo on the webpage is down.
   - Members linked to similar issues like [this one](https://discuss.huggingface.co/t/sentence-transformers-all-minilm-l6-v2-not-working-all-of-a-sudden/152691) as potential help.
- **CPU RAM Offloading Fine When Merging**: Members discussed offloading to CPU RAM when merging a checkpoint to the base model.
   - One member said it's fine, and pointed out that *Transformers + Accelerate or Llama.cpp* enable offloading, also that the **GGUF format assumes CPU offloading**.
- **Inference speed of different models compared**: Members pondered about **Model Context Protocol (MCP)** and which is the fastest for inference of running models.
   - It was noted that **Unsloth** is faster than Hugging Face, with others recommending **sglang/lmdeploy** or **exllamav2**.
- **Seeking Active AI Hackathons and Cohorts**: A member inquired about active **AI-related cohorts or hackathons** that provide incentives or rewards for participation.
   - No specific recommendations were provided in the follow-up.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

cakiki: <@1298649243719958612> please don't cross-post
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1366761093053022209)** (9 messages🔥): 

> `3D Animation Arena, Pi-Scorer alternative to LLM-as-a-Judge, HMR Models` 


- ****3D Animation Arena Opens for Ranking HMR Models****: A member created a [3D Animation Arena on Hugging Face](https://huggingface.co/spaces/3D-animation-arena/3D_Animation_Arena) to rank models based on different criteria, aiming to leaderboard the current best **HMR (human mesh recovery) models**.
   - The creator is seeking votes to populate the leaderboard.
- ****Pi-Scorer Emerges as LLM-as-a-Judge Alternative****: A member shared **Pi-Scorer**, an alternative to **LLM-as-a-Judge**, providing Colab notebooks for using **Pi-Scores** as a [model checkpoint evaluation](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/SFT_Model_Checkpoint_Observability_withPi.ipynb) and as [reward functions](https://colab.research.google.com/github/withpi/cookbook-withpi/blob/main/colabs/PiScorer_as_GRPO_Reward_Function.ipynb).
- ****AI Assistant Integration Code Shared****: A member shared the [code](https://github.com/BouajilaHamza/site-ai-assistant-integration) for their **AI assistant integration** project.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1366836869610078282)** (2 messages): 

> `Defect annotation, Image masking, Filter usage` 


- **Tackling Defect Annotation Headaches**: A member is trying to implement the [paper](https://arxiv.org/pdf/2009.07047v1) but faces the challenge of generating and annotating old scratched images.
   - The member synthetically generated images with defects like scratches, blur, and grayscale, and is now seeking advice on how to annotate these defects.
- **Masking Method Makes an Appearance**: A member suggests masking the image, binarizing it while testing different thresholds to isolate scratches, and leaving the rest of the image untouched.
   - The member pointed out how to test different thresholds to find the ideal balance.
- **Filtering for Flaws**: A member suggests using filters like **Canny edge** or **Sobel** to isolate defects with specific thresholds.
   - These filters might provide a good isolation for the defects with certain threshold, it could make it easier to auto-annotate scratches on dataset.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1366625364310228993)** (40 messages🔥): 

> `Hugging Face Agents certification, Agents.json vs Prompts.yaml, Llama-3 access request, Models temporarily unavailable, Solving the final project with free resources` 


- **HF Agents Course Completion Celebrated!**: Members celebrated completing and getting certified on **Hugging Face Agents**, with one member sharing their [LinkedIn profile](https://www.linkedin.com/in/suhail-ahmed-9b4312b/).
   - Another member shared their [LinkedIn profile](https://www.linkedin.com/in/roshankv/) upon completing the course as well.
- **Timeout Tamed by Tweaking Time!**: One user reported solving timeout issues by increasing the timeout value in the `requests.get` function to **20 seconds**.
   - Another user confirmed that this change solved their problem.
- **Agents.json and Prompts.yaml Pondered**: A course participant asked for clarification on the difference between the **agents.json** and **prompts.yaml** files in the context of the smolagents section of Unit 1.
   - The user also sought guidance on *adding new tools to the list of tools using the tools parameter of your Agent*.
- **Llama-3 Access Request Rejected!?**: A user reported that their request for access to **meta-llama/Llama-3.2-3B-Instruct** was rejected and asked why.
   - Other members suggested needing access to Llama in general, directing the user to request access [here](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf).
- **"Temporarily Unavailable" Troubles**: A user reported that all the models they were trying to use were showing as *temporarily unavailable*.
   - Another user suggested setting up the notebook locally with **Apple's MLX framework** as a possible workaround.


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1366809269244264469)** (1 messages): 

> `Audio Overviews, Multilingual Support` 


- **Audio Overviews are here!**: Audio overviews are rolling out in beta, ready for users to create in over **50+ languages**.
   - Try it out now in your preferred language and share feedback via this [blog post](https://blog.google/technology/google-labs/notebooklm-audio-overviews-50-languages/).
- **Multilingual abilities now available**: Audio overviews now support **50+ languages**, giving access to more diverse users!
   - Please check the [blog post](https://blog.google/technology/google-labs/notebooklm-audio-overviews-50-languages/) for more details.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1366658539661299712)** (28 messages🔥): 

> `NotebookLM language support, Audio Overview limitations, Concise explanations, Smarter Models` 


- ****NotebookLM**'s Global Tongue-Twist: Now Speaks Many Languages**: NotebookLM now specifies the language of the conversation, which is a new feature, and [Google's NotebookLM now speaks 50 languages](https://www.forbes.com/sites/rogerdooley/2025/04/29/googles-notebooklm-now-speaks-50-languages-enabling-global-content/).
   - Users tested audio overviews in **Icelandic** and **Marathi**, with one user impressed that the Marathi speech was fluent and authentic, *"not that foreigner accent or something"*. 
- ****Audio Overview** Customization Caps Stir Debate**: A user noted that the customize audio update is limited to **500 characters** and wondered if this is any different than uploading instructions as a separate text file.
   - The user wanted to *"lessen the silly banter, and keep focus on the facts and timeline"*.
- **Users find **Audio Overviews** are more Concise for Non-English languages**: Users found that the **Audio Overviews** generated for non-English languages were shorter in duration.
   - One user who tested it on a small document, stated, *"its pretty concise explanation"*.
- **Smarter Models Powering Better **Explanations****: Google has confirmed that the new non-English language **Audio Overviews** are better because *"we're using smarter models under the hood!"*
   - NotebookLM continues to improve its summarization capabilities under the hood.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1366674918128877682)** (65 messages🔥🔥): 

> `NotebookLM Updates, Multi-Language Support, Audio Overview Issues, Interactive Mode Bugs, Podcast Feature Requests` 


- ****NotebookLM** Claims a **Webby**!**: **NotebookLM** had a successful run at the [Webby Awards](https://winners.webbyawards.com/2025/ai-immersive-games/ai-apps-experiences-features/technical-achievement/331142/notebooklm), winning a **Technical Achievement** award.
- ****Multi-Language** Support Arrives, but Not for Everyone!**: Members celebrated the arrival of **multi-language support** for **NotebookLM**, however a member noted that **vietnamese audio** wasn't working and the UI still said *"English only".*
   - A member confirmed that the rollout is still in progress, and advised users to wait a few hours, and another followed up, telling members to *"Prepare for 'not seeing the new feature even if it's out what do I do' ten times a day."
- **Non-English Audio Overviews Limited by Time!**: A user reported that the **English audio overview** had a **15 minute limit**, while the **Turkish** one was limited to **6 minutes 20 seconds**.
   - A member stated that non-English audio is currently limited for *"technical reasons"*, but the team is working on extending the time.
- **Interactive Mode Microphone Issues Bugging Users!**: One user reported that the **interactive mode** wasn't picking up any audio from their microphone.
   - Another member suggested checking **microphone permissions** and **browser settings**, and try using a [mic test](https://mictests.com/) and trying another browser.
- **Notebook Sharing Woes and Solutions!**: A member reported that people they shared a **Notebook** with were getting a message that *"they do not have access"*.
   - A member clarified that users need to explicitly add the emails of the people they are sharing with in the share dialog.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1366635231347736626)** (75 messages🔥🔥): 

> `Add on Credits, Manus Fellow Program, Manus Referral Program, Manus Credit System, Beta Testing` 


- **Add on Credits are Useless without Resubscribing**: A user warned that the add on credits given to early subscribers are useless unless they resubscribe, because they expire after a short time.
   - The user claimed they were not informed about the expiry, and now they lost **3900** credits.
- **Questions Answered about Double Credits**: A user provided quick FAQs about double credits, stating that bonus credits never expire as long as your subscription is active.
   - They added that invites are random, and it seems that **every invite doesn’t get two invites**, it’s random, because they may have just throttled it back.
- **User Needs Information about Manus Fellow Program**: A user asked for information about the Manus Fellow Program, like if Manus reach out required fellows & hire them? Also about the targeted countries (USA China Singapore Korea Australia etc), and if the program is not for countries like Pakistan India.
   - Another user replied that a starter plan gives **2 invites** and a pro plan gives **5 invites**.
- **Credit System and Beta Testing Critiqued**: A user expressed their thoughts on the credit system and beta testing, claiming that limiting users with credits undermines the very idea of a beta phase.
   - They added that *a real beta test would let users complete full projects from start to finish, giving meaningful feedback about the experience and suggesting improvements*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1366626785827094620)** (51 messages🔥): 

> `X-Ware Red, Llama Prompt Ops, LLM Benchmarks Survey` 


- ****X-Ware Red** tool releases**: A user shared a tool **X-Ware Red** that uses the title of an embed; it prepends `r.jina.ai/` and `openrouter-free-tier` to generate titles for threads.
   - A user suggested making it a toggle to choose whether the thread title should be different from the name of the embed.
- ****Llama Prompt Ops** Introduced**: **Meta** introduced [Llama Prompt Ops](https://github.com/meta-llama/llama-prompt-ops), an open-source tool for prompt engineering, and [Synthetic Data Kit](https://github.com/meta-llama/synthetic-data-kit).
- **Bug Discovered where Link Posts Retitle Threads**: A user reported a bug where posting a link in a thread retitles an already named thread, although it *should only look for threads with "https://" in the title and change those*.
- **Users seek Durable **LLM Benchmarks****: A user asked for a good survey of **LLM benchmarks** that support comparing models historically.
   - Another user responded that *most last less than 2 years* and suggested the "AI Engineer Reading List" for current ones and pointed to a user's posts for the OSS leaderboard v1 and v2.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1366719239968526416)** (13 messages🔥): 

> `Bending Origins in Mojo, Origin-related headaches, Multiple Licenses in Modular Repository, Pointer usage to avoid origin issues` 


- **Exercise Bending Origins in Mojo to your Will**: A member wanted to do a little exercise involving bending **Origins** to do what you want, like rebinding origins to a container element's **Origin** instead of the container's.
   - Another member responded that they've dealt with a lot of origin-related headaches, mostly *gaps in our APIs, parametrizable traits, and other missing language features*.
- **Origins Cause Mutating Reference Issues**: A member mentioned that *you can't hold two mutating references to the same origin*, though one can cast the origin to a **MutableAnyOrigin** to circumvent that.
   - Another member responded that any data structure which is not array or list shaped has issues which regresses performance down to **C perf**.
- **Origins are bypassed for pointer time**: When discussing building a list-like type + a span-like type, or reading code like the `sort` implementations in the stdlib, one member noted that *most of those are screw the origins, pointer time*.
   - Another member worried about pointer types (unsafe included) due to all the mut-immut fixes.
- **Modular Repository Has Multiple Licenses**: It seems like the **Modular repository** needs to contain multiple licenses now since some parts are licensed with Modular's **Community License** while others are with **Apache 2**.
   - Specifically, some of the things in [`src/max`](https://github.com/modular/max/blob/main/src/max/serve/README.md) use the community license.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1366728808526381056)** (11 messages🔥): 

> `importing Python packages, profiling blocks of code, SIMD width, vector strip-mining, flamegraph` 


- **Standard Python `import` Support Maybe Coming**: While full support for standard Python `import` statements in Mojo isn't confirmed, it's a *pretty definite maybe* according to one member, implying that `python.import_module` may not be the only option forever.
- **`llvm-mca` surfaced, profile particular blocks of code**: A member asked about profiling specific code blocks, mentioning a private part of the `gpu` module ([link](https://github.com/modular/max/blob/main/mojo/stdlib/src/gpu/profiler.mojo)), and another suggested using `llvm-mca`.
- **Vector Strip-Mining for SIMD Widths**: When specifying a **SIMD width** that's a multiple of the hardware's SIMD width, the term *vector strip-mining* was suggested as a potential name for how the compiler handles it.
- **`Flamegraph` aids Perf Output Visualization**: A member recommended using [flamegraph](https://github.com/brendangregg/FlameGraph) for visualizing `perf` output, noting that the executable should be compiled with **debug info**.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1366854505320419379)** (2 messages): 

> `GPT-4o generates Tetris, PapersChat indexes papers` 


- **GPT-4o Generates Tetris in one Shot**: A video from KaranVaidya6 shows **GPT-4o** generating **Tetris** in one shot using **LlamaIndex** and **Composiohq**.
   - The code used in the video is available on [GitHub](https://t.co/KJb7YRINWg).
- **PapersChat Indexes Papers on ArXiv and PubMed**: **PapersChat** is an agentic AI application that allows you to chat with your papers and gather also information from papers on **ArXiv** and **PubMed**, powered by **LlamaIndex**, **Qdrant**, and **MistralAI**.
   - It indexes all your papers and provides a nifty web UI to query them, available [here](https://t.co/lYwXh27F9x).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1366667182687125557)** (17 messages🔥): 

> `Azure OpenAI timeouts, MessageRole.FUNCTION vs MessageRole.TOOL, Function agent and context issues` 


- **Azure OpenAI's Intermittent Timeouts Plague Users**: Users report intermittent **timeouts** with **Azure OpenAI** endpoints, even with the same prompt, endpoint, and network conditions, suggesting potential **rate limits**, **firewall issues**, or **context breaching**.
   - One user noted that retry mechanisms are ineffective due to the issue persisting for minutes, and changing networks only sometimes resolves the inconsistency.
- **Dissecting MessageRole: FUNCTION vs. TOOL**: The distinction between **MessageRole.FUNCTION** and **MessageRole.TOOL** depends on the specific API being used.
   - Some APIs like **OpenAI** utilize **tool messages**, while others rely on **function messages**.
- **Function Agent Context Snafus Unveiled**: A user encountered an issue with a **function agent** getting stuck at the stream event during the second round of interaction, but the user provided a sample code.
   - A member suggested awaiting the handler (`await handler`) after `stream_events()` exits to ensure the previous run concludes and the final response is received, which fixed the error.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1366639037745467433)** (9 messages🔥): 

> `RAG Chatbot challenges, GraphRAG for multiple sources, Local inference and small model training, Collaborating on AI research` 


- **RAG Chatbot Faces Challenges**: A member working on a **RAG-based chatbot** using official documents is facing challenges with answers requiring chunks from multiple sources and documents, using **vector search + BM25**.
   - They are seeking advice on how to best link references to appendices within the documents for **LLM Claude 3.5 Sonnet v1** and **Amazon Titan v1** embeddings.
- **Exploring GraphRAG for Multiple Sources**: A member inquired whether **GraphRAG** is worth trying to accumulate answers from multiple sources, comparing it to **insightRAG** which requires a domain-specific pre-trained model.
   - They also asked about alternative solutions and mentioned attending **NAACL**.
- **New project around local inference and small model training is explored**: A member, previously co-founder of [Dataherald](https://github.com/Dataherald/dataherald), is exploring a new project around **local inference** and **small model training**.
   - He expressed interest in collaborating and getting involved in the community's research.
- **Robotics, Autonomy and AI: Job opportunities are brewing**: A member working in **Robotics, Autonomy, and AI** is focused on the role of **LLMs** in accelerating software engineering.
   - They inquired about posting job opportunities in the Discord, asking whether it is considered "advertisement".


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1366808622927056927)** (10 messages🔥): 

> `Recursive Symbolic Prompts, LLM Honesty Compliance, HHH Objectives in LLMs` 


- **Exploring Recursive Symbolic Prompts**: A member is exploring how **recursive symbolic prompts** behave under classifier pressure, focusing on how smoothing or alignment constraints affect **multi-turn hallucination drift**.
   - The member is particularly interested in how symbolic structures, like **role-bound predicates** or **attention-synced markers**, survive across multi-turn outputs and how this structure carries across completions despite soft-alignment drift or output smoothing.
- **LLMs HHH tension exposure**: A member shared their research on [quantitatively scoring how LLM outputs behave when comparing HHH (Helpful, Honest, Harmless) alignment objectives](https://www.notion.so/TPIP-Exposing-Alignment-Tension-in-Modern-LLMs-1d5927516e1b8080b8c3d625a40a131d).
   - They used a combination of **YAML** and **python/Gradio** to audit user sessions, measuring the internal tension between each **HHH** variable, which involves forcing models to be more honest and observing the resulting tension.
- **Frontier Models Struggle with Honesty**: The same member found that some frontier models are much more honesty-compliant than others, with some models outputting falsified metrics while providing token-flooded and ambiguous answers.
   - They noted that models like **ChatGPT 4o** and **4.5** output high confidence in answering provocative queries, but in reality, they are flooding the session with ambiguous double-speak, ironically, **OpenAI** is the least transparent of all frontier models.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1366627484828958752)** (12 messages🔥): 

> `Credential Passing, RAG type server for client file ingestion, Streamable HTTP Implementation and Authentication, Multi-Tenant Server Hosting, Open Source Models for Agentic Applications` 


- **Credential Conundrums: Seeking Header Help**: A member is facing difficulties in passing credentials through headers from a client to the MCP server using Python and is seeking assistance.
   - No solutions or suggestions were provided in the given context.
- **RAG Server File Ingestion**: A member is considering building a **RAG-type server** where clients can ingest files via an endpoint, save them on the server, and use them for answering questions.
   - They are asking whether this is a good approach or if there are better alternatives.
- **Streamable HTTP's Implementation: Authentication Appraisal Awaited**: A member inquired about the community's thoughts on the current **Streamable HTTP implementation and authentication**, particularly in the recently released **TS SDK**.
   - Another member responded that it's working well, but they're still figuring out the nuance of hosting a multi-tenant server and how statefulness impacts it.
- **Multi-Tenant Server Hosting**: There are concerns regarding hosting a **multi tenant server** and how that is impacted by statefulness.
   - It seems like a stateless server should spawn a new instance of the mcp server per request, but it is unclear why 1 instance is sufficient for stateful but not for stateless.
- **Productionalizing Agentic Open Source: Feasible or Fantasy?**: A member asked if people are genuinely using open-source models for agentic applications in production (not just pet projects).
   - They find it challenging for most open source models to reason or follow instructions without fine-tuning.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1366882155824807976)** (1 messages): 

> `MCP Server, Real Time Push Notifications` 


- **MCP Server Notifies when Agent Workflows Complete**: A member touted using [mcp-gotify](https://github.com/SecretiveShell/mcp-gotify), an **MCP server** for interacting with [gotify/server](https://github.com/gotify/server), to receive real time push notifications on desktop and mobile when long running multi agent workflows complete.
- **Gotify server alternative?**: Users are now using [gotify/server](https://github.com/gotify/server) as an alternative to push notifications to desktop and mobile.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1366719391240163328)** (9 messages🔥): 

> `foreach optimization, gradient scaling, DoRA + QAT` 


- **Speedy Gradient Scaling via Foreach**: A member shared a [code snippet](https://link.to/snippet) using `torch._foreach_mul_` for gradient scaling, potentially merging with gradient clipping for a single parameter loop.
   - Another member pointed out the [related PR](https://github.com/pytorch/torchtune/pull/2624) and wondered if the seemingly constant gain accumulates over many iterations.
- **Tune Contributors Seek Easy Pick-Up Issues**: A member highlighted [two easy issues](https://github.com/pytorch/torchtune/issues/2648) and [another](https://github.com/pytorch/torchtune/issues/2649) for community contribution to the project.
   - No further information was provided regarding the nature of the issues.
- **DoRA and QAT: An Unexplored Frontier?**: A member inquired about experiences combining **DoRA (Difference of Low-Rank Adaptation)** with **QAT (Quantization-Aware Training)**.
   - There was no discussion or response regarding this combination in the messages provided.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1366696277844164642)** (6 messages): 

> `MCP Usage, Displaying thoughts component in React` 


- **MCP Usage Documentation Desired**: A user inquired about tutorials or documentation for the new **MCP (Multi-Controller Processing)** usage added in the latest release.
   - Another user noted they got started by reviewing the test cases, and while a tutorial would be nice, it's not urgent, clarifying that understanding the **stdio** and **SSE clients** setup was key.
- **Thoughts component in React - Best Practices**: A member is seeking advice on the best way to display the **Thoughts component in React**.
   - They know they can modify the forward method, but are asking if there is a better or more appropriate place to implement this.


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1366789810827825152)** (1 messages): 

> `Markdown-based vs Image-based multimodal RAG on PDFs, Docling, EmbedV4` 


- **Markdown vs Image RAG debate heats up**: A member inquired about comparing **Markdown-based** versus **Image-based multimodal RAG** on **PDFs**.
   - They are currently using **Docling** to convert PDFs to Markdown and then computing text embedding, but are considering switching to **EmbedV4** to feed raw images and get multi-modal embedding for RAG.
- **PDF Conversion Techniques Explored**: The member is using **Docling** to convert PDFs to Markdown before computing text embeddings.
   - They are evaluating **EmbedV4** as an alternative to directly process raw images for multi-modal embeddings in RAG.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1366841428076003541)** (2 messages): 

> `Cohere rate limits for embed-v4, Embed V4 on Bedrock` 


- **Cohere considers raising rate limits**: A user inquired whether **Cohere** would increase production rate limits for `embed-v4`.
   - They stated that **400 requests per min** is not enough for their use case with **PDFs**.
- **Cohere ponders Bedrock availability**: A user asked whether **Embed V4** will be available on **Bedrock**.
   - There has been no answer from Cohere yet.


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1366782678187114607)** (2 messages): 

> `Cohere's Embed V4 model, Data Scientists introductions` 


- **Enthusiast Joins, Eager for Embed V4!**: A new data scientist joined the Cohere Discord community, expressing a keen interest in trying new tools, particularly the latest **Embed V4 model** from Cohere, and exploring its potential applications.
   - The new member is *pleased to join the community*.
- **Community Welcomes New Data Scientist**: The Cohere Community Discord Server expresses excitement in the introduction of a new member.
   - The welcome message encourages new members to provide their **Company/Industry/University**, the specifics of *what you're working on*, favorite tech/tools, and *What you hope to gain from this community*.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1366781809614131220)** (5 messages): 

> `Embeddings, GPT4All, Manus AI, Embedding grouping` 


- **Manus AI Tool Drops**: A member shared a link to [Manus AI](https://manus.im/edu/invitation/FBBGGFBFKTUE), claiming that *China dropped it* and that it is now available for everyone.
   - The member suggested that this is the *first auto research AI Agent* and that *we gettin hard replaced with this one*.
- **Embeddings can use Nomic tools**: A member suggested that Nomic provides all the necessary tools for embeddings and that it is *beyond GPT4All*.
   - They claimed that Nomic's embeddings tools *work in various other software*.
- **Embedding grouping can work instead of training**: A member described how **grouping embeddings** could work instead of training: group embeddings for a specific person and take the average embedding, then use that embedding to sort other pictures and find the same person.
   - He asked *Did you understand the concept?*


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1366685162615865406)** (3 messages): 

> `Loose vs Strict Evaluation, Model Training Inconsistencies` 


- **Evaluating Models Loosely vs Strictly**: A member proposed the idea of having a *'loose' vs 'strict' evaluation mechanism* for models, especially those that can be *'hacked'* into working, representing specific use-cases.
   - They provided an example of a model incorrectly trained to emit `<tool_call>` instead of `<|tool_call|>` as its specs indicate, where a knowledgeable user might ignore the error and evaluate functional correctness.
- **Model Training Creates Inconsistencies**: One member encountered a model which was incorrectly trained to emit `<tool_call>` instead of `<|tool_call|>` as its specs indicate.
   - The member suggested that, if they knew the model specifically, they could ignore this error and evaluate on functional correctness, but a naive user could not.


  
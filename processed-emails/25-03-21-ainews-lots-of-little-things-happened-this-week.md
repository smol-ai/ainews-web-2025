---
id: c9965154-d604-4bbd-b2a4-0197cfb2005a
title: lots of little things happened this week
date: '2025-03-22T00:20:28.950836Z'
original_slug: ainews-lots-of-little-things-happened-this-week
description: >-
  AI news highlights from March 20-21, 2025, including launches, research,
  benchmarks, and new models from companies like anthropic, gemini, nvidia, and
  meta, as well as updates on reasoning benchmarks and tools.
companies:
  - anthropic
  - gemini
  - nvidia
  - meta
  - roboflow
  - kyutai-labs
  - topaz-labs
  - percy-liang
models:
  - claude-code
  - llama-3.3-nemotron-49b
  - sakata-ai
  - gpt-4
  - gpt-3
  - claude-2
topics:
  - instruction-following
  - multi-step-problem-solving
  - reasoning-benchmarks
  - ai-models
  - ai-research
  - reinforcement-learning
  - multiturn-agents
  - ai-benchmarks
  - ai-competitions
---


<!-- buttondown-editor-mode: plaintext -->**Incremental updates are all you need.**

> AI News for 3/20/2025-3/21/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**227** channels, and **3009** messages) for you. Estimated reading time saved (at 200wpm): **318 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

- Claude Code (which we mentioned [last month](https://buttondown.com/ainews/archive/ainews-claude-37-sonnet/)) had a [mini launch week](https://x.com/_catwu/status/1903130881205977320)
- [Mindmaps in NotebookLM](https://x.com/tokumin/status/1902251588925915429?s=46)
- [Roboflow launched their YOLO competitor](https://x.com/roboflow/status/1902810257652351228?s=46)
- [Anthropic made a lot of noise about a think tool](https://x.com/AnthropicAI/status/1903128670081888756)
- [Gemini launhced a bunch of things](https://x.com/GeminiApp/status/1902752852843331650) 
- [Kyutai Moshi added vision](https://x.com/kyutai_labs/status/1903082848547906011)
- [Topaz announced a fast upscaler](https://x.com/topazlabs/status/1902742856512446490)
- [Percy Liang relaunched HELM](https://x.com/percyliang/status/1902890719985160471?s=46)

all this and more in the Twitter/Reddit/Discord recaps. We hope to ship the weekly AINews this weekend.


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Models and Benchmarks**

- **New research from @AnthropicAI reveals a simple 'think' tool dramatically improves instruction adherence and multi-step problem solving for agents**: [@alexalbert__](https://twitter.com/alexalbert__/status/1903130655564922911) documented these findings in a blog post. [@skirano](https://twitter.com/skirano/status/1903152968288932085) also noted that they made an **MCP** for this, which can be downloaded from their official Anthropic MCP server repo. [@_philschmid](https://twitter.com/_philschmid/status/1903019035765170419) observed that **@AnthropicAI** appears to be the first to release combined reasoning and tool use, with **Claude** reasoning, generating a function call, executing it, and then continuing to reason with the output.
- **NVIDIA's Llama-3.3-Nemotron-Super-49B-v1 ranks at #14 on LMArena**: According to [@lmarena_ai](https://twitter.com/lmarena_ai/status/1903116426535375060), this model is a **powerful open reasoning model**, excelling in math with an openly released 15M post-training dataset. The ranking overview of this model, previously tested under the codename "march-chatbot" on LMArena, can be found [here](https://twitter.com/lmarena_ai/status/1903116429508886904).
- **Sakana AI** is using **Sudoku Puzzles to superpower AI reasoning**: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1902913196358611278) announced the release of a new reasoning benchmark based on the modern variant of Sudoku to challenge the AI community, believing these puzzles are perfect for measuring progress in AI reasoning capabilities. The new benchmark and training data are available [here](https://twitter.com/SakanaAILabs/status/1902913196358611278). [@hardmaru](https://twitter.com/hardmaru/status/1902920388117446680) simply stated that as a species, we can improve our collective reasoning and problem-solving ability by playing Sudoku.
- **The HELM benchmark has a new leaderboard: HELM Capabilities v1.0**: [@percyliang](https://twitter.com/percyliang/status/1902890719985160471) noted that they curated **5 challenging datasets (MMLU-Pro, GPQA, IFEval, WildBench, Omni-MATH)** and evaluated **22 top language models**.
- **Meta AI** released **SWEET-RL**, a novel RL algorithm for long-horizon & multi-turn tasks which can perform better credit assignments: [@AIatMeta](https://twitter.com/AIatMeta/status/1903146901068988473) reported that experiments demonstrate that SWEET-RL achieves a **6% absolute improvement** in success & win rates on CollaborativeAgentBench compared to other state-of-the-art multiturn RL algorithms, enabling Llama-3.1-8B to match or exceed the performance of GPT4-o in realistic collaborative content creations. More details on both of these releases can be found in the full paper published on arXiv.
- **Meta AI** also released a new agents benchmark: **CollaborativeAgentBench**, the first benchmark studying collaborative LLM agents that work with humans across multi-turn collaboration on realistic tasks in backend programming & frontend design: Details at [@AIatMeta](https://twitter.com/AIatMeta/status/1903146899458363442).
- **New on LMArena**: [@Nvidia](https://twitter.com/lmarena_ai/status/1903116426535375060)'s **Llama-3.3-Nemotron-Super-49B-v1** lands at **#14**. It is a powerful open reasoning model—top-15 overall, excelling in math, with an openly released **15M post-training dataset**.

**Language Model Development and Releases**

- **Gallabytes** joined **Cursor** to work on coding agents: After an incredible 3 years leading model development at **Midjourney**, [@gallabytes](https://twitter.com/gallabytes/status/1902864624510439516) announced their move to Cursor.
- **Kyutai Labs released MoshiVis**, an end-to-end low-latency Vision Speech Model: [@reach_vb](https://twitter.com/reach_vb/status/1903126742954377445) noted the model only adds **206M parameters** and uses a learnable gating mechanism, adding only ~7ms per inference step on a **MacMini with M4 Pro Chip**, while maintaining real-time performance.
- **NVIDIA built GR00T N1**, a powerful open-source AI model designed for humanoid robots: According to [@TheTuringPost](https://twitter.com/TheTuringPost/status/1903066519128641809), it's a **Vision-Language-Action (VLA)** model based on **Eagle-2** with **SmolLM-1.7B**, and a Diffusion Transformer. It generates 16 actions in ~64 milliseconds on an NVIDIA L40 GPU.
- **ByteDance** just announced **InfiniteYou** available on Hugging Face: According to [@_akhaliq](https://twitter.com/_akhaliq/status/1902937194198700280), this is for Flexible Photo Recrafting While Preserving Your Identity.
- **Roblox** just casually dropped a app for **Cube 3D** on Hugging Face: [@_akhaliq](https://twitter.com/_akhaliq/status/1902882220588605485) noted that it generates **3D models directly from text**.
- **Claude gets real-time web search**: According to [@TheRundownAI](https://twitter.com/TheRundownAI/status/1903031351621849254), **OpenAI's voice AI** got a personality boost. [@_philschmid](https://twitter.com/_philschmid/status/1903019035765170419) believes that **@AnthropicAI** are the first releasing combined reasoning + tool use.

**AI Applications and Tools**

- **The Deep Research x AI Builder Thesis**: [@swyx](https://twitter.com/swyx/status/1903121115310067794) theorizes the collision path between the prompt-to-app AI builder and the deep research agent, suggesting building a deep research app on demand to split out UI generation and data generation into separate agents.
- **Dair.AI** promotes the use of **LLM-as-a-Judge**, a technique for automating the assessment of LLM outputs by using a specialized LLM as a “Judge”: [@dair_ai](https://twitter.com/dair_ai/status/1903098701440061592) believes this enables rapid development of LLM applications and AI agents.
- **LangChain** released MCP Adapters: [@LangChainAI](https://twitter.com/LangChainAI/status/1903159677845745736) announced their new TypeScript library that connects Anthropic's MCP tools with LangChain.js & LangGraph.js, featuring multi-server support and seamless agent integration.
- **LlamaIndex** announced LlamaExtract is now in public beta: This leading, genAI-native agent for structured document extraction adapts the latest models to structure even the most complex documents: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1902880391578653176).
- **Perplexity** is working on an updated version of Deep Research: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1902876897773760577) states that the new version will throw even more compute, think longer, present more detailed answers, use code execution, and render in-line charts.

**AI Community and Events**

- **Andrew Ng** shared his observations from the **AI Dev 25 conference**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1903147778097983709) noted that agentic AI continues to be a strong theme, developers are fine-tuning smaller models on specific data, and many speakers spoke about the importance of being pragmatic about what problems we are solving, as opposed to buying into the AGI hype.

**Optimization and Training**

- **Cloneofsimo** shared findings from exploring extreme beta values in training: [@cloneofsimo](https://twitter.com/cloneofsimo/status/1903080158627762234) notes that large beta2 seems crucial, until beta1 also becomes small, and that small beta1 allows small beta2.
- **Hamel Husain** provided an update on training tools: [@HamelHusain](https://twitter.com/HamelHusain/status/1903140801917403538) let his audience know that he'd be online in ~ 15 min (will be recorded for those who sign up).

**Humor**

- **Neel Nanda** jokingly asked if 21% don't think someone is a billionaire: [@NeelNanda5](https://twitter.com/NeelNanda5/status/1902869137489019188).
- **Vikhyatk** joked about moving to SF and finding a room for only $6000/mo: [@vikhyatk](https://twitter.com/vikhyatk/status/1902915468928954855).
- **Swyx** updated a meme: [@swyx](https://twitter.com/swyx/status/1902935741103215103).


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. SpatialLM: LLM for 3D Scene Understanding**

- **[SpatialLM: A large language model designed for spatial understanding](https://v.redd.it/9hvol38aozpe1)** ([Score: 1033, Comments: 94](https://reddit.com/r/LocalLLaMA/comments/1jgap0q/spatiallm_a_large_language_model_designed_for/)): **SpatialLM** is a large language model specifically designed to enhance **3D scene understanding** using **Llama 1B**. The model focuses on improving spatial comprehension, potentially offering advancements in applications that require detailed environmental awareness.
  - **SpatialLM Capabilities**: SpatialLM processes **3D point cloud data** to generate structured scene understanding, identifying architectural elements like walls and doors and classifying objects with semantic categories. It works with various data sources, including **monocular videos, RGBD images, and LiDAR sensors**, making it versatile for applications in robotics and navigation.
  - **Technical Queries and Clarifications**: Discussions raised questions about the classification of SpatialLM as a language model, given its processing of non-human readable data. It was clarified that it outputs structured 3D object graphs, which is a specific form of language, and is based on **Llama 1B and Qwen 0.5B**.
  - **Model Performance and Applications**: Users expressed amazement at the model's capabilities with only **1.25 billion parameters** and discussed potential applications, such as integration with text-to-speech for the visually impaired and use in robot vacuum cleaners. The model's ability to estimate object heights and its potential for integration into reasoning models were also highlighted.


**Theme 2. Qwen 3: Modular AI Model Developments**

- **Qwen 3 is coming soon!** ([Score: 402, Comments: 97](https://reddit.com/r/LocalLLaMA/comments/1jgio2g/qwen_3_is_coming_soon/)): **Qwen 3** is anticipated to be released soon, as indicated by a pull request on the **Hugging Face Transformers** GitHub repository. The link to the pull request is [here](https://github.com/huggingface/transformers/pull/36878).
  - Discussion highlights the **Qwen 3 MoE model's architecture**, particularly its use of **128 experts with 8 activated per token**, and the **15B MoE** model size, which makes it suitable for CPU inference. Users express hope for larger models, like a potential **30-40B MoE** or even a **100-120B MoE**, to compete with modern models.
  - Several comments delve into the **technical details and performance metrics** of Qwen 3, with comparisons to other models like **Deepseek v3**. **Active parameters** are noted to be **2B**, and there's a discussion on the model's potential performance, with references to benchmarks and model equivalence calculations.
  - The community is excited about Qwen 3's potential, especially its **CPU compatibility** and **small active parameter size**, which reduces computational resource requirements. There's interest in its **embedding capabilities** and curiosity about its performance in coding tasks, with some users noting the **vocab size of 152k** and **max positional embeddings of 32k**.


**Theme 3. Docker's Competitive Leap: LLM in Containers**

- **Docker's response to Ollama** ([Score: 240, Comments: 136](https://reddit.com/r/LocalLLaMA/comments/1jgfmn8/dockers_response_to_ollama/)): **Docker** is introducing a new feature that enables **Mac GPU access**, allowing users to run models like `mistral/mistral-small` on their machines. This update excites users as it enhances Docker Desktop's capability by allowing containers to utilize the Mac's GPU, as detailed in their [official announcement](https://www.docker.com/llm/) and further discussed in a [YouTube video](https://www.youtube.com/watch?v=mk_2MIWxLI0&t=1544s).
  - The discussion highlights the **use of wrappers** like **Ollama** and **llama-swap** for managing and running models, with some users criticizing these as unnecessary abstractions over **llama.cpp**. However, others argue that these tools simplify deployment, especially for those not deeply familiar with technical setups, and offer modularity and ease of use in distributing and hosting models.
  - **Docker's new feature** enabling **Mac GPU access** is seen as a significant advancement, allowing Mac users to run applications in isolated environments with GPU acceleration. This update is particularly important for those using **Apple silicon** and is compared to the impact of **GitHub Container Registry** on Docker Hub, though some users express dissatisfaction with Docker's command-line interface.
  - There is a debate over the **open-source community's** approach, with some users expressing concern about projects like Ollama branding themselves instead of contributing to existing projects like **llama.cpp**. Others defend the modular approach, emphasizing the importance of simplicity in development and deployment, particularly in the context of **AI model hosting** and managing dependencies.


**Theme 4. Gemma 3, Mistral 24B, and QwQ 32B: Performance Comparison**

- **Gemma 3 27b vs. Mistral 24b vs. QwQ 32b: I tested on personal benchmark, here's what I found out** ([Score: 231, Comments: 74](https://reddit.com/r/LocalLLaMA/comments/1jgau52/gemma_3_27b_vs_mistral_24b_vs_qwq_32b_i_tested_on/)): **QwQ 32b** excels in local LLM coding and reasoning, outperforming **Deepseek r1** in some instances and significantly surpassing **Gemma 3 27b** and **Mistral 24b**. In mathematics, both **Gemma** and **QwQ** handle simple tasks well, with **Gemma** being faster but having a more restrictive license. **Mistral 24b** underperforms compared to the others, though it, along with **Gemma**, offers image support. For further details, refer to the [blog post](https://composio.dev/blog/qwq-32b-vs-gemma-3-mistral-small-vs-deepseek-r1/).
  - **QwQ 32b's Performance and VRAM Requirements**: Users confirm that **QwQ 32b** excels in coding and reasoning tasks, outperforming some cloud models, but note its significant **VRAM requirements**. This makes it challenging to run on a single GPU even with quantization, limiting its context window size.
  - **Model Comparisons and Quantization Concerns**: There's a need for clarity on **Gemma's model type** used in comparisons, as well as concerns about quantization settings, particularly for **Mistral**, which may affect performance. **RekaAI_reka-flash-3** and **ExaOne Deep** are suggested as alternatives for users with limited hardware resources.
  - **Benchmarking and Use Cases**: Suggestions include running models like **Gemma, Mistral, and QwQ** in an IDE for more practical benchmarks, and testing **ExaOne Deep** and **DeepHermes** for comparison. Users also highlight **QwQ 32b's** strong performance in transcript summarization, occasionally surpassing **GPT-4/4.5**.


**Theme 5. ByteDance's InfiniteYou: Identity-Preserving Image Model**

- **[ByteDance released on HuggingFace an open image model that generates Photo While Preserving Your Identity](https://i.redd.it/efejft8gf1qe1.jpeg)** ([Score: 128, Comments: 36](https://reddit.com/r/LocalLLaMA/comments/1jgft94/bytedance_released_on_huggingface_an_open_image/)): **ByteDance** has launched **InfiniteYou**, an image generation model available on **HuggingFace** that allows for flexible photo recrafting while preserving individual identity. The project features a diverse array of portraits, showcasing individuals in various settings, emphasizing a blend of realism and artistic interpretation. Key resources include the [project page](https://bytedance.github.io/InfiniteYou/), [code repository](https://github.com/bytedance/InfiniteYou), and the [model on HuggingFace](https://huggingface.co/ByteDance/InfiniteYou).
  - Commenters critique the **image quality** of **InfiniteYou**, describing it as "rough" and "plastic-y," indicating skepticism about the model's ability to generate realistic images.
  - **macumazana** points out that similar work has been done previously with older models, suggesting that **InfiniteYou** doesn't offer significant novelty or advancement in the field.
  - **moofunk** suggests a strategic approach by focusing on model strengths and proposing the idea of **chaining models** to enhance photo generation quality, rather than relying on single-model outputs.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. 5 Second Flux Innovation: Nunchaku, InfiniteYou, and Step-Video-TI2V**

- **[5 Second Flux images - Nunchaku Flux - RTX 3090](https://www.reddit.com/gallery/1jg3a0q)** ([Score: 263, Comments: 66](https://reddit.com/r/StableDiffusion/comments/1jg3a0q/5_second_flux_images_nunchaku_flux_rtx_3090/)): **MIT-Han-Lab** has released **ComfyUI-nunchaku**, a tool for generating **5-second flux images**. The announcement also mentions the **RTX 3090**, although no specific details about its role in the project are provided.
  - Users expressed skepticism about **ComfyUI-nunchaku's** output quality, noting that images appear "plastic" and similar to those generated by models like **SDXL**. Concerns were specifically raised about the artificial appearance of human faces, often featuring **cleft chins**.
  - **Nunchaku SVDQuant** offers significant performance improvements, reducing model size by **3.6×**, memory usage by **3.5×**, and achieving **10.1× speedup** on **NVIDIA RTX 4090** laptops by eliminating CPU offloading. The tool supports **lora** conversion similar to **TensorRT**, with detailed setup instructions provided via [GitHub](https://github.com/mit-han-lab/ComfyUI-nunchaku) and [Hugging Face](https://huggingface.co/mit-han-lab/svdq-int4-flux.1-dev/tree/main).
  - A user shared their experience using the **deepcompressor** repo for quantizing **flux finetunes**, encountering challenges with **cuda/transformers** dependencies and VRAM limitations, suggesting a **24GB VRAM** is insufficient. They provided a workaround by renting an **A40** GPU and shared steps for potential dependency fixes.


- **[InfiniteYou from ByteDance new SOTA 0-shot identity perseveration based on FLUX - models and code published](https://i.redd.it/8ohrkqaenzpe1.png)** ([Score: 193, Comments: 59](https://reddit.com/r/StableDiffusion/comments/1jgamm6/infiniteyou_from_bytedance_new_sota_0shot/)): **ByteDance** has introduced **InfiniteYou**, a new state-of-the-art zero-shot identity preservation model based on **FLUX**. The model, alongside its code, has been published, showcasing its ability to enhance identity characteristics in images, as demonstrated in a comparison grid featuring **ID Image**, **PuLID-FLUX**, and **InfU** (Our model), with InfU showing advanced rendering and fidelity in identity preservation.
  - Discussion around **Flux's identity preservation** reveals mixed opinions: while some users note that the model effectively adheres to prompts and maintains facial details, others criticize it for not accurately replicating input features like eye color and hair, as well as the "Flux chin" issue. **ByteDance's InfiniteYou** is viewed as a significant step forward, though its realism is questioned by some users.
  - **Hugging Face** is a focal point for the model's availability, with users eager to see its integration into **ComfyUI** workflows. There is a demand for better handling of features like freckles, scars, and tattoos, which are seen as essential for high-quality facial replicas.
  - Users express impatience with the current **Flux model's aesthetic** and predict a shift once a new open model becomes available. **ByteDance's** approach focuses on research and methodology rather than aesthetics, which some users find lacking in terms of practical, photorealistic application.


- **[Step-Video-TI2V - a 30B parameter (!) text-guided image-to-video model, released](https://github.com/stepfun-ai/Step-Video-TI2V)** ([Score: 119, Comments: 61](https://reddit.com/r/StableDiffusion/comments/1jg3mx2/stepvideoti2v_a_30b_parameter_textguided/)): **Step-Video-TI2V** is a newly launched **30 billion parameter** model that facilitates **text-guided image-to-video** conversion. This release marks a significant advancement in AI-driven video generation.
  - **Model Size and Performance**: The **Step-Video-TI2V** model, with its **30 billion parameters** and **59GB weights**, is seen as a significant advancement, though its local usage is challenged by high VRAM requirements (up to **70GB** for **720p** videos). Users discuss the impracticality of its current resource demands, jokingly suggesting the need for a **kidney** to run it locally.
  - **Chinese AI Development**: There is a perception that **China** is advancing rapidly in the AI sector, with multiple video models emerging consecutively, while the **US** and **EU** lag behind. Some users note that although China is producing these models, they do not always provide the best quality outputs, as seen with **Yuewen**'s implementation.
  - **Quality and Compression Concerns**: Users express concerns about the **compression techniques** used in the model, which result in a loss of detail despite the model's large size. The model's reliance on **16x spatial** and **8x temporal compression** is criticized for hindering its ability to generate fine details, leading to glitches and subpar results in video outputs.


**Theme 2. Text-to-Video AI Advancements: From Open-Source Initiatives**

- **[Remade is open sourcing all their Wan LoRAs on Hugging Face under the Apache 2.0 license](https://v.redd.it/4hdgt7rrg1qe1)** ([Score: 171, Comments: 21](https://reddit.com/r/StableDiffusion/comments/1jgfz25/remade_is_open_sourcing_all_their_wan_loras_on/)): **Remade** is open sourcing all their **Wan LoRAs** on **Hugging Face** under the **Apache 2.0 license**, allowing for broader access and use within the AI community.
  - Some users, like **Weird_With_A_Beard** and **Mrwhatever79**, are enthusiastic about the **Wan LoRAs**, expressing gratitude and enjoyment in using them for video generation. However, others are skeptical about the claim of open-sourcing, highlighting that **LoRAs** generally don't have licenses and questioning the authenticity of the open-source claim due to premium services offered via a Discord server.
  - **LindaSawzRH** and **hurrdurrimanaccount** criticize the open-source claim, arguing that the **LoRAs** are not truly open-source if the training data and processes are not provided, and access is behind a paywall. They express concerns about the precedent this sets for the community, with **hurrdurrimanaccount** questioning whether datasets are being shared.
  - **Ballz0fSteel** shows interest in a tutorial for training **Wan LoRAs**, but **LindaSawzRH** suggests that access to such information might require payment, further fueling the discussion about the transparency and accessibility of the resources.


- **[Wan I2V - start-end frame experimental support](https://v.redd.it/adqnxs24u1qe1)** ([Score: 160, Comments: 21](https://reddit.com/r/StableDiffusion/comments/1jghi5d/wan_i2v_startend_frame_experimental_support/)): **Wan I2V** introduces experimental support for start-end frames, enhancing its capabilities in video processing. This update is likely to improve the precision and efficiency of video frame analysis.
  - **WanVideoWrapper Update**: The **WanVideoWrapper** by **Kijai** received an update for experimental start-end frame support, previously available in **raindrop313's** repository. This improvement allows the introduction of new objects in scenes which were difficult to prompt before, although some issues like missing elements and color shifts persist, which can be mitigated by adjusting parameters such as caching and resolution.
  - **Community Excitement and Testing**: Users expressed enthusiasm about the update, with some already testing it with **Kija nodes** and reporting positive results. The feature is seen as a potential game-changer for scripted storytelling, offering more reliability than previous versions.
  - **Open Source and Collaboration**: The community appreciates the open-source nature of the project, highlighting contributions from various developers like **raindrop313** and expressing gratitude for the collaborative efforts that led to these advancements.


**Theme 3. Critique of LLM Evaluation Methods: Simplification & Blame**

- **[Shots Fired](https://v.redd.it/pwwan4cz3zpe1)** ([Score: 1372, Comments: 284](https://reddit.com/r/ClaudeAI/comments/1jg91lt/shots_fired/)): Critics argue that **LLM intelligence tests** are often unevaluative, implying that they fail to accurately measure or reflect the true capabilities and intelligence of large language models. This criticism suggests a need for more rigorous and meaningful evaluation methods to assess AI performance.
  - **Yann LeCun's Perspective**: **Yann LeCun** is discussed extensively, with many agreeing that **LLMs** alone won't lead to **AGI**. LeCun emphasizes the need for new AI architectures beyond LLMs, as presented in his speech at the **NVDA conference**, and is recognized for his significant contributions to AI, particularly in deep learning and CNNs.
  - **Limitations of LLMs**: Several commenters argue that LLMs are limited in achieving AGI due to their architecture, which lacks the ability to learn and adapt like human intelligence. **Virtual Intelligence (VI)** is suggested as a more appropriate term for current AI capabilities, emphasizing utility over consciousness or self-awareness.
  - **Current AI Utility and Misconceptions**: There is a consensus that while LLMs are not useless, they are tools that require proper use and understanding. Some express skepticism about the AI hype, noting that tools like **Claude** have improved and can enhance productivity, but they do not replace human jobs or achieve independent reasoning.


- **[After giving me a puzzle I couldn’t solve I asked for one simpler](https://i.redd.it/dv77ltcuzwpe1.jpeg)** ([Score: 529, Comments: 113](https://reddit.com/r/ChatGPT/comments/1jg0ilw/after_giving_me_a_puzzle_i_couldnt_solve_i_asked/)): The post discusses a **ChatGPT** interaction where the user requested a simpler puzzle after being unable to solve "The Three Chests" puzzle. The **AI** responded without sarcasm, implying it genuinely believed the user needed an easier challenge, highlighting potential limitations in understanding user intent or context.
  - **Logical Reasoning and Puzzle Analysis**: **Claude**'s analysis of "The Three Chests" puzzle demonstrates a classic logical reasoning approach, questioning the accuracy of labels and considering potential twists like incorrect labels. The discussion highlights the need to consider whether all labels are incorrect, which would lead to choosing the chest labeled "Gold" after testing the "Silver" chest first.
  - **Humor and Sarcasm**: Several commenters, like **EuphoricDissonance** and **Careless_General5380**, use humor to engage with the topic, joking about the treasure being "love" or "friends made along the way." This reflects the light-hearted nature of the discussion around the puzzle's simplicity and the AI's response.
  - **Puzzle Constraints and Solutions**: **Toeffli** points out a missing element in the puzzle regarding truth-telling and lying notes, which affects determining the treasure's location. **Professional_Text_11** and others note the absence of a rule against opening all three chests, suggesting a straightforward solution that bypasses the intended puzzle logic.


**Theme 4. AI-Generated Satire and Historical Reconstructions**

- **[Doge The Builder – Can He Break It?](https://v.redd.it/x64ffkvmpwpe1)** ([Score: 183, Comments: 24](https://reddit.com/r/ChatGPT/comments/1jfz5lt/doge_the_builder_can_he_break_it/)): **Doge The Builder** satirizes **Elon Musk** and **Dogecoin** by comparing them to "Bob the Builder," highlighting themes of greed, economic chaos, and unchecked capitalism. The post humorously references a fictional licensing by the **Department of Automated Truth (DOAT)** and suggests a YouTube link for viewing in the comments.
  - **AI's Role**: Commenters express admiration for the capabilities of **AI** in creating content like "Doge The Builder," highlighting its impressive nature in the current age.
  - **Cultural Impact**: Discussions touch on the influence of individuals like **Elon Musk** on society's zeitgeist, questioning the morality of amassing wealth and its implications on civilization.
  - **Creation Curiosity**: There is curiosity about the process of creating satirical content, with inquiries on how such pieces are made.


- **[Made this in 5 minutes. We're going to need some good AI detection soon...](https://v.redd.it/nrw8wbjbiwpe1)** ([Score: 13355, Comments: 552](https://reddit.com/r/ChatGPT/comments/1jfy5cz/made_this_in_5_minutes_were_going_to_need_some/)): The post highlights the urgent need for improved **AI detection** technologies, specifically in the context of rapidly produced AI-generated videos. The author underscores the ease with which such content can be created, implying potential challenges in distinguishing authentic videos from AI-generated ones.
  - Concerns about the authenticity of AI-generated videos are prevalent, with users like **YoshiTheDog420** expressing skepticism about ever having reliable **AI detection** tools. They fear that visual evidence could become unreliable, with any footage potentially dismissed as AI-generated, undermining trust in media.
  - The discussion highlights the ease with which people can be fooled by AI-generated content, as **Rude_Adeptness_8772** suggests a significant portion of elderly individuals might perceive such videos as genuine. **Visarar_01** shares an anecdote about a family member being deceived by an AI video, illustrating the potential for misinformation.
  - Some commenters, like **ProfessionalCreme119**, propose solutions such as integrating **AI detection** tools into devices to identify AI-generated videos, suggesting a need for widespread implementation of detection mechanisms. Others, like **Soft-Community-8627**, warn about the potential misuse of AI to fabricate events, which could be leveraged by governments to manipulate public perception.


**Theme 5. AI Art and Workflow Transparency Debates**

- **Can we start banning people showcasing their work without any workflow details/tools used?** ([Score: 265, Comments: 56](https://reddit.com/r/StableDiffusion/comments/1jgiok1/can_we_start_banning_people_showcasing_their_work/)): The post suggests banning art posts that do not include **workflow details** or **tools used**, arguing that without such information, these posts function merely as advertisements. The author calls for a change to ensure contributions are informative and beneficial to the community.
  - Many users, including **Altruistic-Mix-7277** and **GravitationalGrapple**, argue against banning posts without workflow details, suggesting that the subreddit serves both as a gallery and a resource for learning. They emphasize the importance of open-ended discussion and the ability to ask questions directly in comments for additional details.
  - **Lishtenbird** highlights the ongoing issue of "no workflow" posts, noting the disparity in engagement between detailed guides and flashy, low-effort content. They suggest implementing an auto-mod comment system to ensure that at least some information, like prompts, is shared, although this would require additional resources to implement.
  - **ByWillAlone** and **wonderflex** discuss the subreddit’s dual nature as both an art showcase and a learning platform. They propose the idea of creating a separate space, like **r/aiartschool**, dedicated to in-depth tutorials and high-effort content, while maintaining the voting system to naturally filter content quality.


- **[This guy released a massive ComfyUI workflow for morphing AI textures... it's really impressive (TextureFlow)](https://www.youtube.com/watch?v=NSQLVNAe5Hc)** ([Score: 105, Comments: 11](https://reddit.com/r/StableDiffusion/comments/1jfxbgr/this_guy_released_a_massive_comfyui_workflow_for/)): **ComfyUI** released a significant workflow called **TextureFlow** for generating and morphing AI textures. The release is notable for its impressive capabilities in AI texture manipulation.
  - **TextureFlow** is available via a direct link to the workflow JSON on [GitHub](https://github.com/edenartlab/workflows/blob/main/workspaces/mono_workspace/workflows/texture_flow/TextureFlow.json). Users are exploring its capabilities for AI texture manipulation and generation.
  - Users like **Parulanihon** are experimenting with **TextureFlow** for logo creation, recommending a denoising level of **0.3 or 0.4 max**. However, challenges include achieving transparent backgrounds and aligning with outdated YouTube tutorials, necessitating a mix-and-match approach to achieve desired results.
  - **No-Mistake8127** is using **TextureFlow** to create animated artwork for a custom **Raspberry Pi** driven digital frame, highlighting its ability to handle inputs such as video, text prompts, photos, movement, and controlnets.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-2024-12-17

**Theme 1. Pricing Showdowns and Censorship Woes**  

- [**Cursor Burns Wallets**](https://cursor.sh/pricing): Users rage over charges for connection errors and lost premium requests when downgrading plans. One member quipped *"Normal agent is a shit show without max"* and opted out due to cost inefficiencies.  
- [**OpenAI’s o1 Pro Overheats**](https://help.openai.com): Developers call o1 Pro a *"monumentally overpriced"* model, preferring Claude or cheaper alternatives like DeepSeek. Some joked that o1 Pro costs *$30 per full send*, making it a luxury few can afford.  
- [**Pear vs. Cursor Price War**](https://www.pear.ai): Some note that Pear is cheaper but *“can’t code worth a damn”* and relies on *roo code* for file changes. Others warn that if Cursor’s pricing and context limits don’t improve, they might jump ship.

**Theme 2. Model Upgrades and Debates**  

- [**Claude 3.7 Provokes Passion**](https://www.anthropic.com/news/claude-3-family): Some swear 3.7 is *"better for going the extra mile,"* while others say 3.5 is more accurate. The community agrees *"no single hammer is better for every job,"* reflecting a divide over performance quirks.  
- [**Qwen 3 Draws Crowds**](https://github.com/huggingface/transformers/pull/36878): People excitedly track news that Qwen 3 is imminent, following the recent Qwen 2.5 Omni release. Leaked hints suggest it might challenge top-tier models like GPT-4.5.  
- [**Sora Falls Short**](https://huggingface.co/unsloth/DeepSeek-V3): Despite big teasers, this public release underwhelmed users who found it inferior to Keling AI and Hailuo AI. Critics suspect *"turbo version"* hype overshadowed real performance limitations.

**Theme 3. Fine-Tuning Adventures and VRAM Tussles**  

- [**Gemma 3 Keeps Breaking**](https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb): Missing dependencies and `--no-deps` bugs stumped users trying older Colab notebooks. One dev lamented *"Why does Llama fail here, but works fine in my other environment?"*  
- [**QLoRA Slays Memory Woes**](https://discord.com/channels/1179035537009545276/1351288406239346758): Turning on QLoRA instantly cut VRAM usage, letting Gemma 3 run on smaller hardware. Loading in 4-bit mode helped avoid out-of-memory crashes.  
- [**DeepHermes 24B Overflows VRAM**](https://arxiv.org/abs/2408.11857): People face OOM errors running 24B on multi-GPU rigs, even with minimal context. Suggestions include 8-bit versions or fine-tuning multi-GPU setups with flags like *--tensor-split*.

**Theme 4. New Tools, Agents, and RAG**  

- [**Oblix Orchestrates Edge vs. Cloud**](https://oblix.ai/): A slick demo shows agents juggling local and remote LLMs for cost/performance trade-offs. The system decides whether to run queries on hardware like Ollama or farm them out to OpenAI.  
- [**Local RAG App Wows Coders**](https://twitter.com/llama_index/status/1903121505984319771): A fully local retrieval-augmented generation tool chats with code using GitIngest for parsing and Streamlit for UI. It runs Meta’s Llama 3.2 locally through Ollama, delighting developers seeking offline solutions.  
- [**Semantic Workbench Rides In**](https://github.com/microsoft/semanticworkbench): Microsoft’s new VS Code extension prototypes multi-agentic systems in one place. Users wonder if it doubles as an MCP framework or stays primarily a dev tool.

**Theme 5. Tokenizer Tricks, Synthetic Data, and Hardware Upgrades**  

- [**SuperBPE Shrinks Sequences**](https://x.com/alisawuffles/status/1903125390618661068): A newly minted *superword* tokenizer cuts sequence lengths by 33% at a fixed 200k vocab. Tests show an 8% MMLU boost and 27% faster inference compared to standard BPE.  
- [**Synthetic Data Reigns**](https://arxiv.org/abs/2503.00808): Researchers highlight filtering, augmentation, and generation as a way to *“reject data we already predict well.”* Open-source labs like Bespoke promise fresh synthetic pipelines for targeted fine-tuning.  
- [**Nvidia’s Blackwell Sparks Skepticism**](https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus): Next-gen RTX Pro cards tout up to 96GB of VRAM but threaten to worsen GPU supply shortages. Enthusiasts doubt Nvidia’s claim that *“we’ll fix availability by May/June.”*


---

# PART 1: High level Discord summaries




## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Pricing Proves Punitive**: Users express frustration with **Cursor's pricing model**, citing charges for connection errors, resumed requests, and 'tool charges for no responses', with some [reporting lost premium requests](https://cursor.sh/pricing) after downgrading plans.
   - Some users find the *'Normal agent is a shit show without max'* and find it 'quicker than spending real $ on max', opting out of premium due to perceived cost inefficiencies.
- **Claude 3.7 Causing Consternation**: Members report issues with **Claude 3.7's performance in Cursor**, claiming false assumptions and decreased reliability compared to **Claude 3.5**, with some [having the opposite experience](https://www.anthropic.com/news/claude-3-family).
   - Opinions vary, with one user stating *'3.7 is better for going the extra mile. 3.5 is better for accuracy'*, while another notes *'There’s no single hammer that’s better for every job'*.
- **Pear's Potential Prompts Pricey Problems**: Users compare **Pear AI** to **Cursor**, noting **Pear’s** cheaper pricing but also concerns about its reliance on *roo code* and per-file change acceptance workflow, whereas [others cite that Pear can’t code worth a damn](https://www.pear.ai).
   - Some **Cursor** users, like one who said *'I don't like pear AI that much, mainly cause they use roo code and roo code is not that stable'*, are considering switching if **Cursor** doesn't improve its context window or pricing.
- **React Racketeering Raises Rivalries**: The channel debates the merits of **React** versus **Svelte** for a SaaS app, with some preferring **React** for its large community and compatibility with **Cloudflare Pages**, while others find it slow and messy, [advocating for Svelte](https://svelte.dev/).
   - The user base seems pretty split, with arguments ranging from *'react is slow af'* to *'svelte also doesn't need workarounds'*.
- **Vibe Visions Vary Wildly**: Members debated the usefulness of **vibe coding**, with some calling it a *marketing ploy* and a *crock*, while others argued that it is a real thing requiring technical expertise, [like a basic knowledge of Git](https://git-scm.com/).
   - Despite varying definitions, a consensus emerged that successful 'vibing' requires critical thinking, debugging skills, and the ability to steer AI tools effectively.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 gets Dependency Glitch**: **Gemma 3** has a bug with `--no-deps` causing missing dependencies in older notebooks, and a Google Colab with a **2018 GPU** might be too outdated for some tasks, according to [this discussion](https://discord.com/channels/1179035537009545276/1351288406239346758).
   - A user encountered issues with **Llama** failing in a **Gemma**-specific environment, but the same notebook failed on Google Colab due to missing dependencies, according to [this notebook](https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb).
- **Vision Fine-Tuning still on Unsloth's backburner**: Despite **Gemma 3** supporting images, vision fine-tuning is not yet supported on Unsloth, according to [this issue](https://github.com/unslothai/unsloth/issues/2131).
   - A user attempted to fine-tune **Gemma 3** using **Llama** code, which failed, but they still wanted to know if the model would run images after fine-tuning text only.
- **QLoRA to the Rescue for Gemma 3**: Users encountered memory errors when running the **Gemma 3** model, but enabling **QLoRA** resolved the issue, likely due to reduced VRAM usage as mentioned [here](https://discord.com/channels/1179035537009545276/1351288406239346758).
   - Turning on **QLoRA** automatically sets `load in 4bit = true`, which helps to reduce VRAM usage.
- **Community Seeks Synthetic Data Nirvana**: Members discussed tools for synthetic data generation, with one user recommending **Bespoke Labs** due to its extensive features, and confirmed it's open source with a dedicated [Discord server](https://discord.com/channels/1179035537009545276/1351288406239346758).
   - One user inquired about the availability of example notebooks or Colabs demonstrating the implementation of **GRPO with vision models**, but such an example was currently lacking, but is planned for the future.
- **DPO Trainer gets an Upgrade**: A user shared their experience upgrading to the latest **DPO Trainer** with the latest **Unsloth** and **Unsloth Zoo**, providing a [link to their small diff](https://github.com/toranb/sloth/commit/9abead851f5531642470f9a22b5ae00af91a8cb6) for others facing similar challenges.
   - The user also found the [Zephyr (7B)-DPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb#scrollTo=-kyd_iyz7DUM) confusing and suggested updating it via a pull request to the [Unsloth notebooks repository](https://github.com/unslothai/notebooks).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI's o1 Pro Pricing Sparks Outrage**: Users are unhappy with **OpenAI's API pricing** for the **o1 Pro model**, calling it severely overpriced, and preferring **Claude**.
   - Some joked about OpenAI's pricing strategy, and observed that **DeepSeek** offers comparable performance at a fraction of the cost, according to shared charts.
- **Debate Surrounds o1 Architecture**: Discord users are debating if **OpenAI's o1 model** is based on **GPT-4o**, with conflicting claims about its architecture.
   - Arguments focus on knowledge cutoff dates; some think o1 is just *gpt4o with reasoning*.
- **Perplexity Desktop App Boosts Loyalty**: **Perplexity** is rewarding desktop app users with a **free month of Pro** after **7 days of use**.
   - The reward is limited to the **Windows app**, excluding macOS, iOS, Android, and Web users.
- **GPT Pro Subscription Woes Plague Users**: Users reported paying for **GPT Pro** but not getting subscription access, and expressed frustration over **OpenAI support's** unresponsiveness.
   - Affected users were directed to [help.openai.com](https://help.openai.com) for support, with assurances that the channel cannot assist with billing matters.
- **Structured Output Hinders AI Reasoning**: Members tested if the phrase **"No other keys or commentary are allowed"** reduces reasoning capabilities in structured output, and discovered an adverse effect, along with increased token usage.
   - Results suggest that models are overthinking ethical implications, in these conditions.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio API Courts RAG Integration**: Users are eyeing the potential of **RAG** *(Retrieval-Augmented Generation)* integration with the **LM Studio server API**, similar to **Ollama** and **Qdrant**.
   - While the GUI fetches only the top 3 vectors, the API could enable customized implementations with embeddings and vector databases, according to one user.
- **ZeroGPU Pro Users Bump into Quota Walls**: A **ZeroGPU Pro** user hit their GPU quota despite upgrading, possibly because they were using a **FastAPI** backend instead of a **Gradio UI**.
   - They are seeking advice on resolving the quota issue when calling the **ZeroGPU Pro** API from their own application.
- **LM Studio Inspires Browser Extension Ideas**: Potential browser extensions for **LM Studio** are being discussed, including webpage translation using **Gemma 3 27b** and **YouTube** video summarization.
   - One member suggested extensions to summarize YouTube videos by extracting and summarizing subtitles, while feasibility of real-time webpage translation was debated due to speed constraints.
- **Audio Model Alchemists Brew with PyTorch**: A member is experimenting with pretraining an audio model from scratch using **PyTorch** and a transformer architecture, aiming to generate proper audio from tokens.
   - Another member shared their model's song outputs based on names (e.g., *abba.mp3*, *mj.mp3*) and suggested fine-tuning or uploading the model to **Hugging Face** for broader experimentation.
- **RX 9070 owners report slow speeds**: Several users with the new **RX 9070** cards are reporting slower inference speeds compared to older cards, with one user reporting their speeds dropped from **5-7 tok/s** to around **3 tok/s** with a **Granite 3.1 8B Q8_0 model**.
   - The performance issues are suspected to stem from bugs in **AMD's Vulkan drivers**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude Code Copies Aider Web Search**: A user observed that [Claude code](https://www.anthropic.com/claude-pricing) is implementing **web search** in a similar fashion to **Aider**, which was demonstrated in a [post on X](https://x.com/_catwu).
   - It was clarified that the new Claude web search feature is currently exclusive to Claude Desktop.
- **Aider's Commit Flag Triggers Hook Headaches**: Aider adds the `--no-verify` flag during commits, bypassing system hooks, according to [aider/repo.py code](https://github.com/Aider-AI/aider/blob/14f140fdc52fbc7d819c50eca3de1b3e848282f3/aider/repo.py#L136).
   - The maintainer explained that this is because *commit hooks could cause arbitrarily strange things to happen*, suggesting the use of [lint and test hooks](https://aider.chat/docs/usage/lint-test.html#code-formatting-linters) as a workaround.
- **o1-pro API Costs Price Users Out**: Users trying **o1-pro** via the API reported exorbitant costs of *$30 per full send*, rendering it prohibitive.
   - The high cost spurred discussions on caching mechanisms, with speculation on whether **OpenAI's automatic prompt caching** could help mitigate expenses.
- **Pipx Package Installation Woes on Ubuntu**: A user encountered difficulties installing **Aider** for all users on Ubuntu, despite advice to use `sudo pipx install --global aider-chat`.
   - They eventually succeeded by [installing with uv](https://github.com/Aider-AI/aider) at `/usr/local/bin` after overcoming pip and version conflict issues.
- **Aider's Auto-Fixing needs manual prompting**: A user reported that Aider needs manual prompts such as *"fix the tests"* after each failure, despite having enabled the `--auto-test` parameter, referencing the [documentation here](https://aider.chat/docs/usage/lint-test.html#testing).
   - Aider should automatically fix test failures if configured with the *"--auto-test"* setting.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Deep Research Limits Debated Fiercely**: Users are debating **Deep Research** usage limits, referencing the [Perplexity blog](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research) stating *unlimited* access for Pro, while others cite a **500 queries per day** limit.
   - A member pointed to a tweet by [Aravind Srinivas](https://x.com/AravSrinivas/status/1890464738951233536) indicating *Paid users only need to pay $20/mo to access an expert level researcher on any topic for 500 daily queries*.
- **GPT 4.5's Disappearance Creates Confusion**: Users report **GPT 4.5** is missing from **Perplexity Pro**, with some suggesting the model was removed after gaining new subscribers.
   - Some users lauded **4.5** as **SOTA for writing text** while others deemed it slow and uninsightful, creating uncertainty among the user base.
- **Perplexity Users Frustrated by Auto Model Switching Glitch**: Users are experiencing a glitch where **Perplexity** reverts to the *Auto* model, even after selecting a specific model like **Claude**.
   - This issue requires users to manually reselect their preferred model, leading to frustration, especially among those who favor **Claude** over **R1**.
- **API Key Spend Tracking Feature Requested**: A feature request was submitted to [GitHub](https://github.com/ppl-ai/api-discussion/issues) to allow users to name API keys for better spend tracking.
   - Currently, users can track spend by API key, but lack the ability to assign names, hindering efficient management of API usage costs.
- **R1-1776 Finetuning Faces Censorship Scrutiny**: An independent researcher found canned CCP answers and censored content in **R1-1776-671B** and the distilled **R1-1776-70B** when prompted on topics like Tiananmen Square, documented in [this blogpost](https://dsthoughts.baulab.info/)
   - The researchers raised concerns regarding political bias and content filtering in the open-source weights of the model.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Claude Unleashes Web Search**: Web search is now available in [claude.ai](https://claude.ai), enabling **Claude** to finally search the internet and deliver **true positives** for research queries, confirmed in [this tweet](https://x.com/alexalbert__/status/1902765482727645667?s=46).
   - It was later confirmed the search engine being used by Claude is [Brave](https://simonwillison.net/2025/Mar/21/anthropic-used-brave/).
- **Midjourney Lead Swaps Beauty for Code**: After 3 years leading model development at **Midjourney**, a key member joined **Cursor** to work on coding agents, marking a shift from a focus on *beauty and creativity* to *code*, as noted in [this tweet](https://x.com/gallabytes/status/1902864624510439516).
   - The move signals a growing emphasis on practical AI applications in coding environments.
- **InternVL's Training Code Opens Up**: Members expressed surprise that **InternVL** has open source training code, making it one of the few notable models with open training pipelines, with [InternVL's packing implementation](https://github.com/OpenGVLab/InternVL/blob/34a81000402bf8f716bab8c9b57aff1f6b436bd0/internvl_chat/internvl/train/dataset_packed.py) provided as an example of the dataloading approach.
   - The open-source nature of **InternVL** allows the community to inspect the data loading process and dataset iteration.
- **SuperBPE Tokenizer boosts efficiency**: **SuperBPE**, a new *superword* tokenizer that includes tokens spanning multiple words, created a model that consistently outperforms the **BPE** baseline on 30 downstream tasks (+8% MMLU), while being **27%** more efficient at inference time, described in [this tweet](https://x.com/alisawuffles/status/1903125390618661068).
   - At a fixed vocab size of **200k**, **SuperBPE** reduces sequence length by **33%** on average.
- **Smaller Models Benefit from Synthetic Augmentation**: Members discussed whether **smaller datasets** are a new trend, with larger models like **GPT-4.5** potentially needing more data, especially during various post-training stages and the conversation touched on the use of synthetic data to augment smaller datasets for training smaller models.
   - The conversation suggested a trade-off between data size, model size, and the use of synthetically generated data, implying a strategy where smaller models might rely more on enhanced datasets, while larger models can effectively utilize larger volumes of raw data.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Claude Gets Overrated, Grok3 Still King?**: Community members suggest **Claude** is overrated in coding due to limited evaluations beyond **SWE-bench**, hinting it doesn't match **Grok3** on livecodebench.
   - The ratings may be skewed by non-developers, leading to inaccurate assessments of its true capabilities.
- **Gemma Gets Glowing Review**: Members were amazed by **Gemma3's 1340** score and its relatively small **27B** parameter size.
   - One member described **Gemma's** responses as *autistic*, giving very brief answers, often when a much longer one is warranted.
- **Deepseek R1 Hogging VRAM**: **Deepseek R1** requires around **1000GB** of VRAM, with one user deploying it on **8xH200s**.
   - Despite high VRAM usage, there are claims that **Deepseek R1** exhibits baked-in *PRO CHINA* biases, raising concerns about its use, with one user saying *tldr deepseek is #&*&*@% don't recommend using it*.
- **Qwen 3 Coming Soon, Qwen 2.5 Omni Announced**: Reports indicate that **Qwen 3** is coming soon, confirmed by a post on the [Hugging Face Transformer repository](https://huggingface.co/unsloth/DeepSeek-V3).
   - This news follows the announcement of **Qwen 2.5 Omni**, sparking interest and anticipation within the community, as noted in a [Tweet from Lincoln 🇿🇦](https://x.com/Presidentlin/status/1903102260155908200).
- **Sora's Turbo Version Struggles, Hype not Matching Reality**: Users found **Sora's** public release underwhelming compared to its promotional materials, and maybe inferior to competitors like **Keling AI** and **Hailuo AI**.
   - It's suspected that OpenAI used huge amounts of compute over hours to generate them and the released **Sora** version is the *turbo version*.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NLM's Podcast Feature Gets Mixed Reactions**: Users are reporting positive experiences with **NotebookLM's Podcast feature**, though some find that the AI cuts them short during discussions.
   - One user likened the experience to *being part of a radio show where I can talk to hosts*, but felt like a *third wheel* because the AI would revert to its own script.
- **Gemini 1.5 Pro Powers NotebookLM**: Users discuss the underlying model of NotebookLM, with speculation pointing towards **Gemini 1.5 Pro**, while others suggest Gemini 2.0.
   - The discussion underscores the importance of NotebookLM staying grounded in its sources, a key differentiator from [Gemini](https://gemini.google.com/).
- **Users Seek Streamlined PDF Processing**: A user is seeking a more efficient workflow for scanning physical papers into private online storage and making them searchable via natural language queries, and asks whether taking photos with iPhone and sending to NLM for automatic naming and OCR is more efficient.
   - The current manual process involves scanning to PDF, sending to Gmail, manually naming each file, OCR processing, and importing into NotebookLM.
- **AI Avatar Lip Sync Services Face Off**: Members compared lip syncing services for AI avatars, noting that [Hedra](https://www.hedra.io/) is great but pricey.
   - RunwayLM garnered less favorable feedback.
- **Mind Map Feature Slowly Unveiled**: The **Mind Map** feature rollout is proceeding slowly, with many users, including Plus subscribers, not yet seeing it in their accounts.
   - Staff confirmed it will take a *few days* for all users to get it.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nvidia Blackwell RTX Pro Sparks Supply Chain Concerns**: Nvidia launched the [Blackwell RTX Pro series](https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus) for various platforms, potentially squeezing the already tight **Blackwell GPU** supply.
   - While Nvidia anticipates improved **GPU** availability by May/June, skepticism persists among community members.
- **Dataset Evaluation & Augmentation Proves Paramount**: Discussions highlighted dataset evaluation, augmentation, sorting, and categorization as effective methods for using **GPU** hours, with a suggestion to filter data using a small model.
   - A member noted the potential of using a small model to reject data, describing the area as *"underexplored in public"* and cited [Predictive Data Selection](https://arxiv.org/abs/2503.00808) and [Programming Every Example](https://arxiv.org/abs/2409.17115).
- **DeepHermes 24B Stumbles on Multi-GPU Setup**: A user encountered **Out-of-Memory (OOM)** errors running **DeepHermes 24B** on a 5x 3090 setup using `llama.cpp`, even with minimal context settings.
   - Suggested solutions involved using the **8-bit version**, and verifying multi-GPU configurations with `--device`, `--split-mode`, and `--tensor-split` flags.
- **Hermes 3 Powers Up with Llama 3.2**: Nous Research released **Hermes 3 3B**, a new addition to the **Hermes** **LLM** series, detailed in the [Hermes 3 Technical Report](https://arxiv.org/abs/2408.11857).
   - This model features advanced agentic capabilities, improved roleplaying, reasoning, multi-turn conversation, and long context coherence over **Hermes 2**.
- **C# Developer Champions Anthropic LLMs**: A developer offered their **C#** expertise and professional **LLM** experience to the community, highlighting their work on documentation and examples for **Anthropic**.
   - They cited examples such as a **Titanfall 2**-based generator and the **Bladewolf** example from **Metal Gear Rising**, accessible on the [Anthropic GitHub](https://github.com/).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face APIs Suffer 404 Meltdown**: Multiple Hugging Face API models experienced widespread **404 errors**, causing significant downtime for dependent applications.
   - Users reported the outage lasted *almost a whole day* without official acknowledgement, urging the HF dev team for immediate attention.
- **Roblox's Voice Safety Classifier Speaks Up**: **Roblox** released a [large classification model](https://huggingface.co/Roblox/voice-safety-classifier) trained on **2,374 hours** of real-world voice chat to detect toxicity.
   - The model outputs a tensor with labels like `Profanity`, `DatingAndSexting`, `Racist`, `Bullying`, `Other`, `NoViolation`, and uses a synthetic data pipeline detailed in [this blog post](https://research.roblox.com/tech-blog/2024/06/deploying-ml-for-voice-safety).
- **Fuse GPU VRAM via Tensor Tricks**: Users explored techniques to combine VRAM from multiple GPUs, like running **Gemma3-12B** on an **A2000 12GB** and a **1060 6GB** using [tensor parallelism](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_multi).
   - References were made to [Ollama issues on GitHub](https://github.com/ollama/ollama/issues/2672) and [llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions/8725) for more on multi-GPU support.
- **Oblix Platform Juggle AI on Cloud and Device**: The [Oblix.ai](https://oblix.ai) platform intelligently routes AI tasks to cloud or edge based on complexity, latency requirements, and cost considerations, using autonomous agents for optimal performance.
   - A [YouTube video](https://youtu.be/j0dOVWWzBrE?si=Gpp2SG4Kly0tzM_3) demonstrates how Oblix dynamically decides whether to process each AI request locally or in the cloud.
- **Gradio Upgrade Unwraps Dataframe Feature**: A user reported that upgrading to **Gradio 5.22** caused the `gr.Dataframe(wrap=True)` feature to stop working; this wrapping feature was only functioning in **Gradio 5.20**.
   - No further information about this issue was provided.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Microsoft Intros Semantic Workbench**: Microsoft launched the [Semantic Workbench](https://github.com/microsoft/semanticworkbench), a VS Code extension, which is a tool to prototype intelligent assistants, agents, and multi-agentic systems, prompting questions about its role as an **MCP**.
   - A member specifically inquired if the tool functions as an **MCP**.
- **MySQL Server Bombs Out**: A user is encountering issues connecting **mcp-mysql-server** to **Docker MySQL**, reporting connection failures despite it working outside of **MCP**.
   - The error occurs with every connection attempt, creating a significant development *hurdle*.
- **Glama API 500 Error**: A user reported receiving a **500 error** from the **Glama API**, but another member stated that there have been no outages in the last 24 hours, and shared a code sample.
   - The code to reproduce is `curl -X 'GET' 'https://glama.ai/api/mcp/v1/servers?first=10&query=github' -H 'accept: application/json'`.
- **DaVinci Resolve MCP Seeks Speedy Server Claim**: A user is seeking to resubmit their **DaVinci Resolve MCP** project with a license and updates and was told claiming the server might speed up the update process.
   - The project's [repo](https://github.com/samuelgursky/davinci-resolve-mcp) hosts the relevant code.
- **Calendar Scheduling Gets Automated**: A blog post detailed the use of **Asana MCP** and **Google Calendar MCP** with Goose to automate task scheduling, using the [blog post](https://block.github.io/goose/blog/2025/03/20/asana-calendar-mcp).
   - Tasks are pulled from Asana, analyzed, and scheduled in Google Calendar with a single prompt.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Eyes TTS, Image Gen Rollout**: Members expressed interest in **OpenRouter** offering **TTS and image generation**, with some voicing concerns about potentially high pricing.
   - Pricing details and release dates for the new features are still under wraps.
- **Groq Hits Speed Bump, Not Sambanova**: A member reported that **Sambanova** was down, but quickly corrected the statement, clarifying that it was **Groq** that was experiencing issues.
   - Service status updates for **Groq** were not immediately available.
- **GPT-4o Lands on OpenRouter**: **GPT-4o-64k-output-alpha** is now available on **OpenRouter**, supporting both **text and image inputs with text outputs**.
   - The pricing is set at **$6/M input tokens** and **$18/M output tokens**.
- **Fireworks Heats Up Pricing War**: **Fireworks** slashed pricing for **R1 and V3**, with V3 allegedly matching existing performance, pegged at **.9/.9**.
   - The move intensifies competition in the generative AI service market; more information can be found on the [Fireworks pricing page](https://fireworks.ai/pricing).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Nvidia Talk Eyes Pythonic CUTLASS**: Attendees will hear about the pythonic future of **CUTLASS** in its next major 4.0 version at GTC, especially its integration into Python.
   - Previously, a member announced their GTC presentation titled *Performance-Optimized CUDA Kernels for Inference With Small Transformer Models [S73168]* happening today at 4pm, focused on **Hopper architecture**.
- **BFloat16 Atomic Addition Sucks**: A member reported that using `tl.atomic_cas` with a lock for **atomic addition with bfloat16** actually works, but [it sucks](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py).
   - The member is seeking improvements to the implementation, and offered a code snippet using `tl.atomic_cas` with a lock, inviting the community to enhance its performance.
- **Triton's Simplicity Entices GPU Newbies**: A member highlighted that **Triton's** key strength lies not in peak performance, but in its accessibility, enabling individuals with limited GPU experience to create complex kernels, and pointed to [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch/blob/main/native_sparse_attention_pytorch/triton_native_sparse_attention_pytorch.py) as an example.
   - They noted that achieving peak performance on predefined workloads is relatively straightforward, but Triton's robustness is what sets it apart.
- **FlashMLA's SmemLayoutP Unveiled**: A member inquired about the dimensions of `SmemLayoutP` in the [FlashMLA](https://github.com/deepseek-ai/FlashMLA/blob/b31bfe72a83ea205467b3271a5845440a03ed7cb/csrc/flash_fwd_mla_kernel.h#L76C5-76C93) code, specifically its shape `((2,2), kNThreadsS, 1, kBlockN/8)` and the role of `kNThreadsS` in synchronizing **P** between warpgroups.
   - The member speculated whether other dimensions might be related to **wgmma**, awaiting clarification from other experts.
- **Grayscale Leaderboard Smokes the Competition**: Multiple leaderboard submissions to the `grayscale` leaderboard were successful on GPUs: **L4**, **T4**, **A100**, and **H100** using Modal runners with IDs `2351`, `2429`, `2430`, `2431`, `2459`, and `2460`.
   - Benchmark submission with id `2363` to leaderboard `vectoradd` on GPUs: **T4**, **L4**, **A100**, **H100** using Modal runners also succeeded, indicating progress in the `vectoradd` benchmark across various GPU architectures.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Oblix Orchestrates Local vs Cloud LLMs**: A member shared a demo video ([https://youtu.be/j0dOVWWzBrE](https://youtu.be/j0dOVWWzBrE)) of **Oblix**, which *seamlessly switches between local vs cloud*, using agents to monitor system resources and make decisions dynamically.
   - The platform orchestrates between **Ollama** and **OpenAI** for optimal performance and cost-efficiency, as detailed on [Oblix.ai](https://oblix.ai/).
- **AI Engineers Compare LLM Leaderboards**: Members shared links to [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models) and [LM Arena](https://lmarena.ai/?leaderboard) to find reliable LLM leaderboards for specific purposes.
   - Concerns were raised about filtering relevant models from these lists, particularly avoiding outdated options like **Grok-3**.
- **Members Design Medical Data Processing PC**: A member requested assistance with building a new PC to process medical data using AI, emphasizing the need for secure, offline operation.
   - Another member suggested starting with an **Intel i9**, **128GB RAM**, and an **Nvidia 4090 RTX**.
- **GPT4All Struggles with Audio Transcription**: A member inquired about using **GPT4All** for local audio file transcription, specifically uploading **.wav** files, but found that it wasn't working.
   - Another member clarified that **GPT4All** is primarily designed for **docs/pdf**, recommending **XTTS webui** for wav to text conversion, but cautioned that the installation process is complex.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **W-GANs Sidestep Gradient Explosion**: **W-GANs** mitigate gradient saturation by being linear, avoiding the BCE issues of traditional GANs, as shown in [Figure 2 of the W-GAN paper](https://cdn.discordapp.com/attachments/986699377257119794/1352378909194322001/image.png?ex=67de7541&is=67dd23c1&hm=10de0ff85871d920b5ff7db506224a588a2c6c44030082a2ba7a9aad232ef204).
   - However, instability can still arise if the generator or discriminator becomes overly dominant, leading to saturation on both sides.
- **Transformers Get Soft with Slots**: Members shared an image analysis on soft slot methods which shows how **soft slots** dynamically bind to input tokens or retrieved content in **Transformers**.
   - Equations for **Attention** and **Soft Slots (S')** were shown, with learnable slots using softmax and scaled dot-product attention.
- **OpenAI.fm's UX/UI: Fast but Flawed?**: Members joked about the simple and *rushed* UX/UI of [OpenAI.fm](https://www.openai.fm/).
   - One member pointed out that a more structured protocol is easily disrupted by less structured protocols that can evolve according to user needs, and that *clients consume more of what they like and less of what they don't*.
- **G-Retriever Enables Chatting with Graphs**: The [G-Retriever paper](https://arxiv.org/abs/2402.07630) details the semantic extraction of information from knowledge graphs, enabling *chatting with your graph*, *graph QnA* and *Graph RAG*.
   - The paper introduces a **Graph Question Answering (GraphQA) benchmark** with data from scene understanding, common sense reasoning, and knowledge graph reasoning.
- **Moore's Law Accelerates AI?**: Members are discussing [METR_Evals' research](https://x.com/METR_Evals/status/1902384481111322929) suggesting *"Moore’s Law for AI agents"*, claiming the length of tasks AIs can do is doubling about every **7 months**.
   - Some members refuted the claim, arguing that certain tasks are not interesting for probabilistic models.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Local RAG App Deployed for Code Chat**: A **fully local, fully open-source RAG app** has been built that can chat with your code and was announced in [this tweet](https://twitter.com/llama_index/status/1903121505984319771).
   - The app uses **GitIngest** to parse the code into summaries and markdown, **Streamlit** for the UI, and runs **Meta's Llama 3.2** locally using **Ollama**.
- **TypeScript Bundler Config Fixed Import Bug**: A member using **LlamaIndex TS** had an issue importing **agent**, which was resolved by updating the **tsconfig** bundler configuration.
   - The user confirmed that modifying the **TS config** resolved the import error, and thanked the community for the suggestion.
- **Parallel executions limited in Agent Workflows**: A member asked about limiting parallel executions in **Agent Workflows**, specifically for a tool with a **human-in-the-loop event** due to the agent calling the tool multiple times in parallel.
   - The issue was replied on [GitHub](https://github.com/run-llama/llama_index/issues/18220#issuecomment-2742089859) because the user sought to ensure the tool was called only once at a time.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Account Limit Trumps Trial Key Limit**: Users clarified that the monthly limit of **1k requests for trial keys** is per account, not per key.
   - They cautioned that creating multiple accounts to bypass this limit will result in removal of all accounts.
- **Cohere API's Throw Errors**: Users encountered various **Cohere API error messages**, including *invalid request*, *rate limiting*, and *token limits* due to **empty documents**, **short prompts**, exceeding **token limits**, and **incorrect model specifications**.
   - Rate limiting errors are identified by a **429 status code**, as detailed in the [Cohere API documentation](https://docs.cohere.com/reference/errors#429---too-many-requests).
- **Cohere User Seeks Rate Limit Checker**: A user inquired about an API to check their remaining **rate limit** usage.
   - Currently, there doesn't appear to be a direct API solution available.
- **Hospitality Expert Pioneers Low-Code Tech**: Gaby, a professional in the **hospitality industry**, introduced herself as a low-code tech enthusiast, proficient with platforms like **Make** and **Adalo**.
   - Her expertise showcases the growing importance of **low-code tools** in various industries.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Duration Module Displays Weirdness**: A developer working on a **`duration` module proposal** for **Mojo** ran into unexpected behavior with type casting between `Ratio` and `Duration` structs, sharing [code snippet](https://discord.com/channels/1013120035012476074/1173814954275168317) to demonstrate the issue.
   - The specifics of the bug involve unexpected results when converting between the two time formats.
- **Mojo and PyTorch Team Up?**: A member speculated if using **PyTorch** in **Mojo** could speed up training with **MAX**.
   - The inquiry did not receive a response, leaving the potential benefits unconfirmed.
- **Mojo Community Debates Nanosecond Precision**: The community debated using **nanosecond precision** as the base unit for time representation in Mojo; one member noted that a `UInt64` of nanoseconds can cover over **500 years**.
   - Another member countered that C++ guarantees a default time resolution of at least **292 years**, emphasizing that **seconds are the base SI unit** for time.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MIPRO v2 Judges LLMs**: A member reported using **MIPRO v2** with **LLM-as-a-judge** as their evaluation metric and shared [a link to a math reasoning tutorial](https://dspy.ai/tutorials/math/) showcasing its use.
   - The math reasoning tutorial demonstrates **MIPRO** as a metric for evaluating **LLMs**.
- **DSPy Shares LLM-as-a-Judge Documentation**: Documentation on utilizing **LLM-as-a-judge** was shared from [DSPy's learning resources](https://dspy.ai/learn/evaluation/metrics/#intermediate-using-ai-feedback-for-your-metric).
   - The documentation details the use of **AI feedback** for metric evaluations.
- **Automatic Metrics Optimize DSPy**: It was emphasized that **automatic metrics** are critical for evaluation and optimization within **DSPy**.
   - **DSPy** employs metrics to monitor progress and enhance program effectiveness.
- **Metrics Evaluate Task Performance**: A metric is defined as a function that scores system outputs based on data examples; where simple tasks may use basic metrics like *accuracy* or *exact match*.
   - Complex tasks benefit from metrics that assess multiple output properties via **AI feedback**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Member Questions Unet3d's Dimensions**: A member inquired if the example **unet3d** model is actually 3D, proposing it might be **2.5D** because it uses **2D convolutions** and **2D transposes** on 3D input.
   - They drew attention to the difference from a real **3D Unet architecture**.
- **2D Convolutions Mimic 3D**: The conversation clarified that using **2D convolutions** on 3D input creates a 2.5D effect, in contrast to true **3D Unet architectures** which use genuine 3D operations.
   - The original poster requested clarification on the dimensionality of the implementation.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Paper Shared on Torchtune**: krammnic shared [a paper](https://arxiv.org/pdf/2502.07923) on the Torchtune channel.
   - No discussion occurred about this paper.
- **Follow-up on Paper's Relevance**: The paper's title and abstract suggest potential relevance to the ongoing discussions within the Torchtune community.
   - Further investigation is needed to determine the paper's specific contributions and applicability to current projects.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1352357593846251680)** (789 messages🔥🔥🔥): 

> `Cursor pricing, Claude 3.7, Vibe coding, Pear AI vs Cursor, React vs Svelte` 


- **Cursor's Pricing Proves Punitive**: Users are frustrated with **Cursor's pricing model**, where they are charged for connection errors and resuming requests, as well as 'tool charges for no responses' and [reported losing premium requests](https://cursor.sh/pricing) after downgrading plans.
   - They are finding that *'Normal agent is a shit show without max'* and are 'quicker than spending real $ on max'.
- **Claude 3.7 Causing Consternation**: Members are reporting issues with **Claude 3.7's performance in Cursor**, claiming that it makes false assumptions and is less reliable than **Claude 3.5**, whereas [others are having the opposite experience](https://www.anthropic.com/news/claude-3-family).
   - As one user put it, *'3.7 is better for going the extra mile. 3.5 is better for accuracy'* with another adding *'There’s no single hammer that’s better for every job'.*
- **Pear's Potential Prompts Pricey Problems**: Users are comparing **Pear AI** to **Cursor**, noting Pear’s cheaper pricing but also concerns about its reliance on *roo code* and per file change acceptance workflow, while [others cite that Pear can’t code worth a damn](https://www.pear.ai).
   - Some Cursor users, like one who said *'I don't like pear AI that much, mainly cause they use roo code and roo code is not that stable*', are considering switching if Cursor doesn't improve its context window or pricing.
- **React Racketeering Raises Rivalries**: The channel is debating the merits of **React** versus **Svelte** for a SaaS app, with some preferring React for its large community and compatibility with Cloudflare Pages, while others find it slow and messy, [advocating for Svelte](https://svelte.dev/)
   - The user base seems pretty split, with arguments ranging from *'react is slow af'* to *'svelte also doesn't need workarounds'*
- **Vibe Visions Vary Wildly**: Members debated the usefulness of **vibe coding**, with some calling it a *marketing ploy* and a *crock*, while others argued that it is a real thing requiring technical expertise, [like a basic knowledge of Git](https://git-scm.com/).
   - Despite varying definitions, a consensus emerged that successful 'vibing' requires critical thinking, debugging skills, and the ability to steer AI tools effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.twitch.tv/theprimeagen">ThePrimeagen - Twitch</a>: 🚨🚨 DAY 1 - VIBE CODING A Game In 7 Days Using Cursor -- #ad🚨🚨</li><li><a href="https://marketplace.visualstudio.com/items?itemName=sirmspencer.vscode-autohide">Auto&#32;Hide&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;A&#32;tool&#32;to&#32;autohide&#32;the&#32;sidebar&#32;and&#32;terminal&#32;panel.</li><li><a href="https://docs.cursor.com/context/model-context-protocol">Cursor – Model Context Protocol</a>: no description found</li><li><a href="https://cursor.directory/rules">Cursor Directory</a>: Find the best cursor rules for your framework and language</li><li><a href="https://x.com/msfeldstein/status/1902878583594566034">Tweet from Michael Feldstein (@msfeldstein)</a>: I wish they named 3.7 something else, its a totally different model, not a better 3.5Quoting Bass (@SeifBassam) Is anyone else finding that Claude Sonnet 3.5 is better than 3.7 for coding?I reverted a...</li><li><a href="https://github.com/demyxsh/demyx">GitHub - demyxsh/demyx: Demyx is a Docker image that automates and manages WordPress installations. Traefik for reverse proxy with Lets Encrypt SSL/TLS. WordPress sites are powered by OpenLiteSpeed/NGINX-PHP and MariaDB.</a>: Demyx is a Docker image that automates and manages WordPress installations. Traefik for reverse proxy with Lets Encrypt SSL/TLS. WordPress sites are powered by OpenLiteSpeed/NGINX-PHP and MariaDB. ...</li><li><a href="https://github.com/samuelrizzo/jira-mcp-server">GitHub - samuelrizzo/jira-mcp-server</a>: Contribute to samuelrizzo/jira-mcp-server development by creating an account on GitHub.</li><li><a href="https://github.com/geekan/MetaGPT">GitHub - geekan/MetaGPT: 🌟 The Multi-Agent Framework: First AI Software Company, Towards Natural Language Programming</a>: 🌟 The Multi-Agent Framework: First AI Software Company, Towards Natural Language Programming - geekan/MetaGPT</li><li><a href="https://dialogo.chat">Dialogo AI - Intelligent Task Automation</a>: Dialogo AI provides intelligent AI agents that learn, adapt, and automate complex workflows across any platform. From data analysis to system management, our intelligent agents transform how you work.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1352356230764429473)** (241 messages🔥🔥): 

> `Gemma 3 issues, Llama failing in Gemma environment, Vision fine-tuning on Gemma 3, QLoRA for Gemma 3, Synthetic data generation` 


- **Gemma 3 glitches with dependencies, old notebooks trigger**: Members reported that **Gemma 3** has a bug with `--no-deps` that causes missing dependencies, and old notebooks have not been tested recently, according to [this discussion](https://discord.com/channels/1179035537009545276/1351288406239346758).
   - It was also noted that Google Colab with a **2018 GPU** might be too outdated for some tasks.
- **Llama struggles to function in Gemma-specific settings**: A user encountered issues with **Llama** failing in a **Gemma**-specific environment but working fine in another environment without **Gemma** updates, referencing [this notebook](https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb).
   - The user expressed confusion over why the same notebook failed on Google Colab, suggesting missing dependencies due to `--no-deps`.
- **Gemma 3 doesn't yet support vision fine-tuning**: Despite **Gemma 3** supporting images, vision fine-tuning is not yet supported on Unsloth, which was raised in [this issue](https://github.com/unslothai/unsloth/issues/2131).
   - A user attempted to fine-tune **Gemma 3** using **Llama** code, which failed, but they still wanted to know if the model would run images after fine-tuning text only.
- **QLoRA fixes memory errors in Gemma 3**: When running the Gemma 3 model, users ran into memory errors, but enabling **QLoRA** resolved the issue, likely due to reduced VRAM usage as mentioned [here](https://discord.com/channels/1179035537009545276/1351288406239346758).
   - Turning on **QLoRA** sets `load in 4bit = true`.
- **Synthetic Data: Bespoke Labs is a Cool Tool**: Members discussed tools for synthetic data generation, with one user recommending **Bespoke Labs** due to its extensive features.
   - Another user confirmed it's open source with a dedicated [Discord server](https://discord.com/channels/1179035537009545276/1351288406239346758).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Small_(22B)-Alpaca.ipynb#scrollTo=vITh0KVJ10qX">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Small_(22B)-Alpaca">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb?">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/SUFE-AIFLM-Lab/Fin-R1#trainning">SUFE-AIFLM-Lab/Fin-R1 · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb">notebooks/nb/Gemma3_(1B)-GRPO.ipynb at main · unslothai/notebooks</a>: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more. - unslothai/notebooks</li><li><a href="https://github.com/unslothai/unsloth/issues/2131">&quot;Unsloth: Failed to make input require gradients!&quot; When Vision-fine-tune Gemma3 · Issue #2131 · unslothai/unsloth</a>: I&#39;m tring to vision fine-tune Gemma3 refering this tutorial: https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing#scrollTo=QmUBVEnvCDJv I constructed my dataset li...</li><li><a href="https://github.com/canopyai/Orpheus-TTS?tab=readme-ov-file#finetune-model>">GitHub - canopyai/Orpheus-TTS: TTS Towards Human-Sounding Speech</a>: TTS Towards Human-Sounding Speech. Contribute to canopyai/Orpheus-TTS development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/">notebooks/nb at main · unslothai/notebooks</a>: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more. - unslothai/notebooks</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">notebooks/nb/Gemma3_(4B).ipynb at main · unslothai/notebooks</a>: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more. - unslothai/notebooks</li><li><a href="https://github.com/unslothai/notebooks/blob/main/nb/Mistral_(7B)-Text_Completion.ipynb">notebooks/nb/Mistral_(7B)-Text_Completion.ipynb at main · unslothai/notebooks</a>: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more. - unslothai/notebooks</li><li><a href="https://github.com/unslothai/unsloth/issues/2127">Text Completion Notebook - Backwards requires embeddings to be bf16 or fp16 · Issue #2127 · unslothai/unsloth</a>: I am trying to run the notebook from the Continue training, https://docs.unsloth.ai/basics/continued-pretraining Text completion notebook https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObe...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1352575543526162442)** (3 messages): 

> `Unsloth Submissions, Tiny-grad Spreadsheet for tasks, Github issues with high involvement` 


- **Unsloth Submissions Start Getting Reviewed**: A member mentioned that *Unsloth* submissions are starting to get reviewed, but they haven't helped out with any *Unsloth* issues yet.
- **Tiny-grad Spreadsheet on TODO tasks requested**: A member inquired about a **tiny-grad-like spreadsheet** laying out the key things that need doing, wondering if it's mainly **Github issues tagged with help wanted**.
   - They felt this would be a good way to build confidence and stop lurking.
- **Github issues with high involvement are important**: A member stated that if there's a **Github issue with 5 or more people involved** (excluding the team), then it would be pretty important to solve.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1352369422488309961)** (95 messages🔥🔥): 

> `DPO Trainer Upgrade, Zephyr DPO Notebook Confusion, Gemma 3 (27b) inference issue, Unsloth save and push to hub during training, Unsloth finetuning voice models` 


- **Navigating DPO Trainer Upgrade with Unsloth Patch**: A user shared their experience upgrading to the latest **DPO Trainer** with the latest **Unsloth** and **Unsloth Zoo**, providing a [link to their small diff](https://github.com/toranb/sloth/commit/9abead851f5531642470f9a22b5ae00af91a8cb6) for others facing similar challenges.
   - The user also found the [Zephyr (7B)-DPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb#scrollTo=-kyd_iyz7DUM) confusing and suggested updating it via a pull request to the [Unsloth notebooks repository](https://github.com/unslothai/notebooks).
- **Sampling during training causes breakage for Llama-3**: A user reported that using `LogCompletionsCallback` to sample the model during training is broken for **Llama-3**, along with **Gemma 3** (27b), resulting in an error related to default `generation_config` values.
   - A code snippet involving `FastLanguageModel.for_inference(model)` and `FastLanguageModel.for_training(model)` was shared, indicating an attempt to switch between inference and training modes within the callback.
- **Saving and pushing to Hub**: One user inquired about saving and pushing models to the **Hugging Face Hub** during training with Unsloth, noting that the default save strategy wasn't working.
   - Another user suggested building a custom callback for uploading the model to the Hub whenever it saves to disk, but couldn't provide the specific code at the moment, but pointed to [message history](https://discord.com/channels/1179035537009545276/1179035537529643040/1277734237486846066) for the solution.
- **Merging Issue and Model Hallucinations**: Users reported issues with merging LoRA models, experiencing problems where the merging process resulted in **gibberish outputs** or **matrix alignment errors**.
   - Another user found that when fine-tuning **Phi-3.5-mini-instruct**, the model hallucinates when there are too many numerics in the data.
- **GRPO with Vision Models Example Sought**: A user inquired about the availability of example notebooks or Colabs demonstrating the implementation of **GRPO with vision models**, as mentioned in a recent Unsloth blog post.
   - The response indicated that such an example was currently lacking, but is planned for the future.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://shotka.nl."">no title found</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb#scrollTo=-kyd_iyz7DUM">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/HuggingFace%20Course-Gemma3_(1B)-GRPO.ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb#scrollTo=NHwurGh_TH-y)">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth Requirements | Unsloth Documentation</a>: Here are Unsloth&#x27;s requirements including system and GPU VRAM requirements.</li><li><a href="https://huggingface.co/docs/trl/dpo_trainer">DPO Trainer</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth-zoo/pull/91">Updates _unsloth_get_batch_samples to accept a 4th device parameter. by mmathew23 · Pull Request #91 · unslothai/unsloth-zoo</a>: get_batch_samples patch currently takes 3 parameters and transformer 4.50.0 has changed it to take 4 parameters. I&amp;#39;ve updated the function to take device = None. The default keyword maintains ...</li><li><a href="https://github.com/unslothai/notebooks">GitHub - unslothai/notebooks: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more.</a>: Unsloth Fine-tuning Notebooks for Google Colab, Kaggle, Hugging Face and more. - unslothai/notebooks</li><li><a href="https://github.com/toranb/sloth/commit/9abead851f5531642470f9a22b5ae00af91a8cb6">updated dpo script with latest trl deps · toranb/sloth@9abead8</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1352698292898889811)** (7 messages): 

> `LLM chatbot, Personality bots` 


- **Chatbot impersonates friend via finetuned LLM**: A member created a [chatbot](https://enormously-sweeping-donkey.ngrok-free.app/chat) that sounds like their friend Kolo via **finetuned LLM**.
   - The member noted that *it kind of sounds like Andrew Tate a bit lmao*.
- **Personality bots take off**: A member thinks **personality bots** need to take off.
   - They argue that *it is a very good use case for fine tuning* and that it can be **edgy**, **funny**, and **entertaining**.



**Link mentioned**: <a href="https://enormously-sweeping-donkey.ngrok-free.app/chat">Vite + React</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1352643528177745941)** (8 messages🔥): 

> `Foundation Model Training, Tree-of-Thought, Monte Carlo Tree Search` 


- **Turbocharge Foundation Model Training 6x**: A member shared an image claiming a **6x speedup** in foundation model training.
   - Another member commented *"if it works well" is a big if*, suggesting skepticism about the method's reliability.
- **ToT and MCTS: Relics or Relevant?**: A member mentioned **Tree-of-Thought (ToT)** and **Monte Carlo Tree Search (MCTS)** as precursors to current test-time compute scaling strategies.
   - Another member shared their experience, stating *"I tried Tree-of-Thought before didn't perform good"*, to which they were asked to clarify the specific tasks.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1352356071968211027)** (296 messages🔥🔥): 

> `OpenAI Pricing, o1 Model Architecture, Grok Deep Research, Perplexity desktop app` 


- **OpenAI's o1 Pro Pricing Angers Users**: Users expressed frustration with **OpenAI's API pricing**, particularly for the **o1 Pro model**, deeming it *diabolically, horrifically, monumentally overpriced*.
   - One user humorously remarked that one can only laugh at *OpenAI's pricing philosophy*, while another suggested that **Claude** outperforms **o1 Pro**.
- **o1 Architecture Under Scrutiny**: Discord users debated whether **OpenAI's o1 model** is based on **GPT-4o**, with claims that **o1** has its own distinct architecture, while others believe it's a fine-tuned version of **GPT-4o**.
   - Arguments centered on the fact that, if the base model were different, it would have a different knowledge cutoff, with many concluding that o1 is *gpt4o with reasoning*.
- **DeepSeek Outshines OpenAI**: Members shared charts that claimed that models like **DeepSeek** were comparable in performance to other models.
   - The pricing of **DeepSeek** was significantly lower than that of **OpenAI**, leading to further frustration.
- **Perplexity Desktop App Rewards Loyalty**: A user mentioned that **Perplexity** is offering a **free month of Perplexity Pro** for users who use their desktop app for **7 days straight**.
   - This reward, however, is exclusive to the **Windows app**, excluding users on macOS, iOS, Android, and Web.
- **Grok Deep Research Compared**: Users are testing Grok's Deep Research function.
   - One user said they are a *great fan of perplexity research, after trying everything*.



**Link mentioned**: <a href="https://artificialanalysis.ai">AI Model &amp; API Providers Analysis | Artificial Analysis</a>: Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key performance metrics including quality, price, output speed &amp; latency.

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1352386228846329969)** (7 messages): 

> `GPT Pro, Subscription Issues, OpenAI Support` 


- **User struggles with GPT Pro Subscription**: A user reported paying for **GPT Pro** but not receiving access to the subscription, expressing frustration that **OpenAI support** is unresponsive.
   - Another member advised contacting support at [help.openai.com](https://help.openai.com), emphasizing that no one in the channel can assist with billing issues.
- **OpenAI Support Unresponsive to Subscription Problems**: A user reported that they have been unable to get a response from **OpenAI support** after filling out a form with all their information regarding a failed **GPT Pro** subscription.
   - The user indicated they have received no response since yesterday.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1352399502300155956)** (22 messages🔥): 

> `Model Personalization, Strucutred output effect on reasoning, GPT memory usage, Github Copilot Pull Request Descriptions` 


- **Models require personalization for various biases**: The differences in model behavior are based on *how the model guesses* and *if the model has any directed bias that needs to be adjusted for* due to preferences or training data.
   - Different models may require unusual prompting to bypass these biases, especially in sensitive topics.
- **Structured output may affect reasoning**: A member tested whether the prompt, **"No other keys or commentary are allowed,"** reduces a model's reasoning capabilities when using structured output.
   - The results indicated it might *increase token usage and worsen performance* in some cases, possibly due to ethical contemplation.
- **Summarizing Chat history can't ignore directives**: A member inquired about creating a prompt that allows **ChatGPT** to summarize its memory *while ignoring specific directives* given in previous conversations.
   - The goal was to retain general knowledge while disregarding specific instructions like name preferences.
- **Github Copilot for auto pull requests**: A user seeks suggestions to automate pull request descriptions using **GitHub Copilot**, which breaks with medium-long texts.
   - They currently use a manual process involving **ChatGPT** to refine Copilot's summaries and want to optimize this workflow.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1352399502300155956)** (22 messages🔥): 

> `Prompt Engineering Adaptability, Model Guessing and Bias, Model Memory and Personalization, Structured Output and Reasoning, Github Copilot PR Optimization` 


- **Prompt Engineering becomes Model-Specific**: Members observed that prompt engineering is becoming increasingly model-specific, particularly concerning built-in reasoning capabilities, leading to a greater investment of time in refining skills.
   - One member humorously noted, *"It's a great way to excuse myself into even more time spent in my side love/hobby (AI). I'm just 'keeping up on my skills'."*
- **AI Model Guessing and Directed Bias**: One member suggested that differences in AI model behavior stem from 'how the model guesses' and whether the model has any directed bias that needs to be adjusted for.
   - They further added, *"I think training data for the model is like a vast sea.  Any topic 'well represented' in training data may have many unique instances in the model's understanding of the topic."
- **Model Memory Enhances with User Interaction**: Members suggested that AI models adapt more effectively the more users engage with them, challenging the idea that AI is static.
   - One member recounted successfully connecting two non-interacting APIs through GPT, emphasizing the potential for innovation and pushing the boundaries of what's believed about AI, and to *"don’t be afraid to push the limits of what we believe we know about ai, AI’s changing and fast, it’s gunna be shaping the future and we’re lucky enough to be working with it when it’s new"*.
- **Structured Output Restraints Examined**: A member questioned if using the phrase *"No other keys or commentary are allowed"* to enforce structured output could reduce reasoning in models, prompting experimentation.
   - The user ultimately noted that the phrase had little to no effect, or in some cases, had the opposite effect.
- **Automating Pull Request Descriptions with Copilot**: A user is seeking advice on automating pull request descriptions with Github Copilot due to issues with long inputs breaking Copilot inside Github.
   - The user is using this prompt:
```
Create a pull request body description that:
- Always begins with: "This pull request introduces..."
- Includes the following sections: **Additions**, **Fixes**, **Refactors**, and **Deletions** where possible.
- Avoids any references to commit messages, links, or minor changes (such as TypeScript interface tweaks).
- Provides a short, bullet-point summary for each section.
- Maintains the same uniform, consistent structure.
```


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1352365621790244895)** (103 messages🔥🔥): 

> `LM Studio Server API for RAG, ZeroGPU Pro Upgrade Issues, Browser Extensions for LM Studio, Audio Model Training with PyTorch, Speculative Decoding Crashes` 


- **LM Studio API Eyes RAG Integration**: Users are exploring the potential of integrating **RAG** *(Retrieval-Augmented Generation)* functionality with the **LM Studio server API**, similar to **Ollama** and **Qdrant**.
   - One user noted that while the GUI only retrieves the top 3 vectors, the API could allow for more customized implementations with embeddings and a vector database.
- **ZeroGPU Pro Users face GPU Quota Hiccups**: A **ZeroGPU Pro** user reported an issue where they exceeded their GPU quota despite having a paid upgrade, possibly due to using a **FastAPI** backend instead of a **Gradio UI**.
   - This user is seeking advice on how to resolve this quota issue when calling the **ZeroGPU Pro** API from their own application.
- **LM Studio sparks Browser Extension Ideas**: Users discussed potential browser extensions for **LM Studio**, including translating webpages using **Gemma 3 27b** and summarizing **YouTube** videos, though the feasibility of real-time webpage translation was questioned due to speed.
   - One member suggested extensions could summarize YouTube videos, by summarizing the subtitles from YouTube.
- **Crafting Custom Audio Models with PyTorch Transformers**: A member is experimenting with pretraining an audio model from scratch using **PyTorch** and a transformer architecture, aiming to generate proper audio from tokens.
   - Another member shared examples of their own model's output, generating songs based on names (e.g., *abba.mp3*, *mj.mp3*), and suggested fine-tuning or uploading the model to **Hugging Face** for others to experiment with.
- **Speculative Decoding Stalls LM Studio Models**: A user reported that enabling speculative decoding causes their models to crash consistently, specifically when using **Qwen 2.5 7B** as the main model and **Qwen 2.5 0.5B** as the draft model.
   - Another member suggested creating a detailed report in the **LM Studio Discord** channel, including model details, hardware, and system information.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1352359669238333460)** (136 messages🔥🔥): 

> `RX 9070, Vulkan performance degradation, ROCm support, Gemma3 memory allocation` 


- ****RX 9070** Owners Report Slow Inference Speeds**: Several users with the new **RX 9070** are reporting slower inference speeds compared to older cards like the **RX 580** and **1070**, despite the **9070** showing **100% GPU load**.
   - One user saw speeds drop from **5-7 tok/s** to around **3 tok/s** with a **Granite 3.1 8B Q8_0 model**, while another had similar experiences after upgrading from a **1070**.
- ****Vulkan** Drivers Blamed for Performance Degradation**: The performance issues with the **RX 9070** are suspected to stem from bugs in **AMD's Vulkan drivers**, with one user noting that **Vulkan** performance in some games is also significantly lower than **DirectX 11**.
   - One member suggests downgrading to the **24.10.1 driver**, but this version does not support the **RX 9000** series, and another user found that disabling flash attention improves Vulkan performance.
- **ROCm Support Still in Progress for AMD GPUs**: Members mention that **ROCm support** is still under development, and while **AMD** has technically added support for **gfx1200**, it hasn't been fully implemented in **llama.cpp**.
   - A user shares detailed performance data across different driver versions and **llama.cpp** versions, showing that **Vulkan** performance is often lower than **ROCm**, and that the issue has been addressed
- **Memory Allocation Issues with Gemma3 Models**: A user encounters memory allocation errors when loading **Gemma3 12b** models with context windows larger than **43520**, receiving a **VC++ error**.
   - They discovered that allocating even one additional token increases the buffer size by **96 MB**, causing the allocation to fail, though it works for Vulcan, and that the issue is also very specific to **Gemma3**.



**Link mentioned**: <a href="https://tenor.com/view/rtx-2080ti-gif-19907682">Rtx 2080ti GIF - Rtx 2080ti - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1352356266604888104)** (207 messages🔥🔥): 

> `Claude Code vs Aider Web Search, Aider's --no-verify Flag, o1-pro API experiences, Aider install for all users Ubuntu` 


- **Claude Code Catches Aider Web Search**: A user noted that [Claude code](https://www.anthropic.com/claude-pricing) is implementing **web search** the same way **Aider** does it.
   - They linked to an [X account post](https://x.com/_catwu) demonstrating this feature but others noted that this is only available on Claude Desktop.
- **Aider's Git Commit Flag Causing Hook Headaches**: A member noticed Aider adding the `--no-verify` flag during commits and linked to the relevant [aider/repo.py code](https://github.com/Aider-AI/aider/blob/14f140fdc52fbc7d819c50eca3de1b3e848282f3/aider/repo.py#L136), which bypasses system hooks.
   - The Aider maintainer explained that the flag is used because *commit hooks could cause arbitrarily strange things to happen*, and a potential workaround using [lint and test hooks](https://aider.chat/docs/usage/lint-test.html#code-formatting-linters) was suggested.
- **High Costs make o1-pro API prohibitive**: Some users have tried **o1-pro** via the API, others commented that at *$30 per full send*, they *had to sell their computer and logged off*.
   - The high cost led to discussions about potential caching mechanisms and whether **OpenAI's automatic prompt caching** could help mitigate expenses.
- **Pipx package madness**: One user struggled to [install Aider for all users on an Ubuntu box](https://ubuntu.com/), despite being given advice to use `sudo pipx install --global aider-chat`.
   - They ended up reporting success by [installing with uv](https://github.com/Aider-AI/aider) at `/usr/local/bin` after facing multiple hurdles with pip and version conflicts.
- **Ripgrep MCP: Turbocharge your Claude Context**: Some users are integrating **Claude** with **Model Context Protocol (MCP)** servers like [mcp-ripgrep](https://github.com/mcollina/mcp-ripgrep) for improved file searching, since `search_files` times out on larger directories and doesn't respect `.gitignore`.
   - This allows Claude to interact with the filesystem, providing better context for code generation and problem-solving, but one user was skeptical since Claude provides an "official" MCP product already.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://getcoai.com/careers/vibe-coder-frontend-developer-role/">Vibe Coder Frontend Developer Role - CO/AI</a>: This isn’t about grinding through syntax; it’s about prompting, iterating, and vibing your way to a brilliant product.</li><li><a href="https://aider.chat/docs/usage/lint-test.html#code-formatting-linters">Linting and testing</a>: Automatically fix linting and testing errors.</li><li><a href="https://aider.chat/docs/config/reasoning.html">Reasoning models</a>: How to configure reasoning model settings from secondary providers.</li><li><a href="https://github.com/A">A - Overview</a>: A has 31 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/issues/3591">Can&#39;t set thinking tokens for Sonnet 3.7 via openrouter · Issue #3591 · Aider-AI/aider</a>: Aider currently only supports setting thinking tokens the way Anthropic specifies. See: BerriAI/litellm#9429</li><li><a href="https://github.com/mcollina/mcp-ripgrep">GitHub - mcollina/mcp-ripgrep: An MCP server to wrap ripgrep</a>: An MCP server to wrap ripgrep. Contribute to mcollina/mcp-ripgrep development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider.git">GitHub - Aider-AI/aider: aider is AI pair programming in your terminal</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/blob/14f140fdc52fbc7d819c50eca3de1b3e848282f3/aider/repo.py#L136)">aider/aider/repo.py at 14f140fdc52fbc7d819c50eca3de1b3e848282f3 · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1352658076439806014)** (14 messages🔥): 

> `Aider failing tests, Aider documentation, Aider help command, aider.el package` 


- **Aider's Auto-Fixing Capabilities for Failing Tests Explored**: A user questioned if Aider automatically fixes failing tests with `--auto-test` enabled, noting the need for manual prompts like *"fix the tests"* after each failure, and [documentation](https://aider.chat/docs/usage/lint-test.html#testing) was provided.
   - Aider should automatically fix test failures if configured with the *"--auto-test"* setting.
- **Documentation Download Options Aider-la-Carte**: A user sought downloadable Aider documentation, with a pointer to the [aider/aider/website/docs](https://github.com/Aider-AI/aider/tree/main/aider/website/docs) directory on GitHub, and a file containing patterns to exclude.
   - The file containing patterns to exclude parts of that subtree which aren't really "docs" is available [here](https://github.com/Aider-AI/aider/blob/main/aider/help_pats.py).
- **Aider's Help Command Clarified**: A user inquired about using Aider to its fullest potential, including wrapping it in another system and selectively using its directory scanning capabilities, and was directed to the Aider's troubleshooting documentation [here](https://aider.chat/docs/troubleshooting/support.html).
   - The documentation explains to *Type `/help <question>` and aider will respond with helpful information*, utilizing retrieval augmented generation (RAG) with its indexed documentation.
- **Aider.el package and prompts**: A user noticed that when using `aider.el` package the prompt changes to "What's wrong? Fix" when tests fail.
   - The user confirms that when using aider in the terminal, this is the expected behavior and they have to press enter for the run tests / fix / run tests loop.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/support.html">Using /help</a>: Use “/help &quot; to ask for help about using aider, customizing settings, troubleshooting, using LLMs, etc.</li><li><a href="https://aider.chat/docs/usage/lint-test.html#testing">Linting and testing</a>: Automatically fix linting and testing errors.</li><li><a href="https://github.com/Aider-AI/aider/tree/main/aider/website/docs">aider/aider/website/docs at main · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/blob/main/aider/help_pats.py">aider/aider/help_pats.py at main · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1352359421518544988)** (204 messages🔥🔥): 

> `Deep Research Limits, GPT 4.5 Model, Switching Models, Perplexity apps, Coding AI` 


- **Deep Research Limits Remain Hot Topic**: Users are debating whether **Deep Research** has usage limits, with some referencing the [Perplexity blog](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research) stating *unlimited* access for Pro members, while others cite a **500 queries per day** limit.
   - Members pointed to a tweet by [Aravind Srinivas](https://x.com/AravSrinivas/status/1890464738951233536) who said *Paid users only need to pay $20/mo to access an expert level researcher on any topic for 500 daily queries*.
- **Perplexity without GPT 4.5 Model?**: Users are reporting that **GPT 4.5** is gone from **Perplexity Pro**, with some suspecting that the model was removed after gaining new subscribers.
   - Some users claim that **4.5** was **SOTA for writing text** and the best for specific tasks, while others found it slow and not insightful.
- **Auto Model Switching Glitch**: Several users are experiencing a glitch where **Perplexity** automatically switches back to the *Auto* model, even after selecting a specific model like **Claude**.
   - Some find it frustrating as they have to manually change it back every time the page refreshes, expressing a preference for **Claude** over **R1**.
- **Mobile and desktop apps lagging behind**: Members note the **High Deep Research mode** is only available on the web version, not the desktop app, and the apps in general *always end up really lagging behind, same goes for mobile*.
   - Some find it *best to just use the website on all platforms*.
- **Perplexity not suited for coding?**: A new user asked whether **Perplexity** is suitable for **coding**, or whether it can be used for math and coding games.
   - Other members chimed in *do not use it to code* and that Claude is best for those types of tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1890464738951233536">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Excited to introduce the Perplexity Deep Research Agent: available for free to all users. Paid users only need to pay $20/mo to access an expert level researcher on any topic for 500 daily queries, an...</li><li><a href="https://x.com/AravSrinivas/status/1884801300027589007">Tweet from Aravind Srinivas (@AravSrinivas)</a>: All Perplexity Pro users now get 500 daily DeepSeek R1 queries (without censorship and prompts not going to China). Free users get 5 daily queries.Quoting Aravind Srinivas (@AravSrinivas) 100 daily De...</li><li><a href="https://www.cofyt.app/search/audio-models-in-the-api-Ko6CApwf8D9D7l-VLOudyM">Audio Models in the API</a>: Olivier Godement, Jeff Harris, Iaroslav Tverdoklhib, and Yi Shen introduce and demo three new audio models in the API—two speech-to-text models and one text-to-speech model—and an audio integration wi...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1352409163212259378)** (6 messages): 

> `RAGs, LLM Email Reply System, NotebookLM, Deep Reasoning` 


- **RAGs Explored**: A user asked about [RAGs](https://www.perplexity.ai/search/tell-me-about-rags-and-how-to-GPCo4Y.WSeeeFKYf3m.wYw).
   - The user asked how to implement RAGs.
- **LLM Email System Testing**: A user shared a link to a page about [testing an LLM email reply system](https://www.perplexity.ai/page/testing-llm-email-reply-system-m0nZXPNCRw.M_.fkTCx3Kgi_like_pixel.).
   - No further details were given.
- **NotebookLM Introduces Interact**: A user shared a link to [NotebookLM](https://www.perplexity.ai/page/notebooklm-introduces-interact-AG6Ijc1IT0mzAyXGj8aBiw) introducing Interact.
   - No further details were given.
- **Deep Reasoning Collection Shared**: A user shared a link to a [Deep Reasoning collection](https://www.perplexity.ai/collections/deep-reasoning-EBc3kXnlQA61p_jO8VnfcA).
   - No further details were given.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1352356523560669224)** (7 messages): 

> `API Key Spend Tracking, search_domain_filter Documentation, R1-1776 Open Source Weights, MCP issues` 


- **API Key Spend Tracking Feature Request**: Users can track spend by API key but can't name them yet, so a feature request was submitted to [GitHub](https://github.com/ppl-ai/api-discussion/issues) to address this.
- **search_domain_filter Docs Updated**: The documentation for `search_domain_filter` has been updated, available at [Perplexity's API Reference](https://docs.perplexity.ai/api-reference/chat-completions#body-search-domain-filter).
   - A user inquired if the limit of only 3 domains had changed.
- **R1-1776 Finetuning Censors Material**: An independent researcher evaluating **R1-1776-671B** and the distilled **R1-1776-70B** variant found canned CCP answers and censored content when prompted on topics like Tiananmen Square, described in [this blogpost](https://dsthoughts.baulab.info/).
- **MCP tool experiences Issues**: Users have reported issues with the MCP tool not functioning correctly, documented on [GitHub](https://github.com/ppl-ai/modelcontextprotocol/issues/17#issuecomment-2743569196).
- **API errors flood Perplexity users**: One user reported consistently receiving various API errors (**204**, **503**, **500**), despite making no changes to their script and asked for thoughts on the issue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/api-reference/chat-completions#body-search-domain-filter">no title found</a>: no description found</li><li><a href="https://dsthoughts.baulab.info/">Auditing AI Bias: The DeepSeek Case</a>: Cracking open the inner monologue of reasoning models.</li><li><a href="https://github.com/ppl-ai/api-discussion/issues">ppl-ai/api-discussion</a>: Discussion forum for Perplexity API. Contribute to ppl-ai/api-discussion development by creating an account on GitHub.</li><li><a href="https://github.com/ppl-ai/modelcontextprotocol/issues/17#issuecomment-2743569196">Problem MCP tool n8n - Perplexity · Issue #17 · ppl-ai/modelcontextprotocol</a>: Hello everyone, I try to set up the node perplexity MCP on my workflow n8n but i got some issue. I have done the same for the mcp of BRAVE but it doesn&#39;t work for perplexity. Thanks you by Advance...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1352356687310229617)** (76 messages🔥🔥): 

> `Claude Web Search, Midjourney -> Cursor, TokenSet Image Generation, Qwen3 Release, Hunyuan-T1` 


- **Claude Acquires Web Browsing Superpowers**: Web search is now available in [claude.ai](https://claude.ai), allowing **Claude** to finally search the internet, delivering **true positives** for research queries.
- **Midjourney's Model Lead Joins Cursor**: After 3 years leading model development at **Midjourney**, a key member joined **Cursor** to work on coding agents, marking a shift from focus on *beauty and creativity* to *code* as noted in [this tweet](https://x.com/gallabytes/status/1902864624510439516).
- **TokenSet Breaks Barriers in Image Generation**: **TokenSet** introduces a new paradigm for image generation, representing images as unordered token sets, enhancing **global context** and **robustness** against local perturbations, as shown in their [GitHub](https://github.com/Gengzigang/TokenSet).
- **Qwen3 Imminent: Benchmarks Teased**: **Qwen3** may be launching soon, with members closely watching the [HuggingFace](https://github.com/huggingface/transformers/pull/36878) and [vLLM](https://github.com/vllm-project/vllm/pull/15289) repos, with expectations that it may be **GPT4.5** level.
- **Nvidia's Llama-3.3-Nemotron-Super-49B-v1 Ranks High**: **Nvidia's Llama-3.3-Nemotron-Super-49B-v1** lands at #14 on the [LMArena](https://x.com/lmarena_ai/status/1903116426535375060), excelling in math, with an openly released **15M post-training dataset** available on [Hugging Face](https://huggingface.co/collections/nvidia/llama-nemotron-67d92346030a2691293f200b).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/lmarena_ai/status/1903116426535375060">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: New on LMArena: @Nvidia&#39;s Llama-3.3-Nemotron-Super-49B-v1 lands at #14!A powerful open reasoning model—top-15 overall, excelling in math, with an openly released 15M post-training dataset.Congrats...</li><li><a href="https://x.com/gallabytes/status/1902864624510439516">Tweet from theseriousadult (@gallabytes)</a>: After an incredible 3 years leading model development at Midjourney, I&#39;ve joined Cursor to work on coding agents. I&#39;m incredibly proud of my time at Midjourney and the work we did, of the resu...</li><li><a href="https://x.com/TXhunyuan/status/1902970031245545805">Tweet from Hunyuan (@TXhunyuan)</a>: 📢 Introducing TokenSet: A fundamentally new paradigm for image generation!We&#39;ve broken free from traditional sequential token approaches by representing images as unordered token sets. Our innova...</li><li><a href="https://x.com/alexalbert__/status/1902765482727645667?s=46">Tweet from Alex Albert (@alexalbert__)</a>: Web search is now available in claude dot ai. Claude can finally search the internet!</li><li><a href="https://x.com/TXhunyuan/status/1903121005809373386">Tweet from Hunyuan (@TXhunyuan)</a>: 🚀 Introducing Hunyuan-T1! 🌟Meet Hunyuan-T1, the latest breakthrough in AI reasoning! Powered by Hunyuan TurboS, it&#39;s built for speed, accuracy, and efficiency. 🔥✅ Hybrid-Mamba-Transformer MoE A...</li><li><a href="https://x.com/Presidentlin/status/1903102383250428364">Tweet from Lincoln 🇿🇦 (@Presidentlin)</a>: https://www.reddit.com/r/LocalLLaMA/comments/1jgio2g/qwen_3_is_coming_soon/?sort=new</li><li><a href="https://x.com/reach_vb/status/1903126742954377445">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: LETS GOO! @kyutai_labs just released MoshiVis - an end-to-end low-latency Vision Speech Model, CC-BY license 🔥&gt; Only adds 206M parameters via lightweight cross-attention (CA) modules to integrate ...</li><li><a href="https://github.com/huggingface/transformers/pull/36878">Adding Qwen3 and Qwen3MoE by bozheng-hit · Pull Request #36878 · huggingface/transformers</a>: Adding Qwen3This PR adds the support of codes for the coming Qwen3 models. For information about Qwen, please visit https://github.com/QwenLM/Qwen2.5. @ArthurZucker</li><li><a href="https://github.com/vllm-project/vllm/pull/15289">[Model] Add Qwen3 and Qwen3MoE by YamPengLi · Pull Request #15289 · vllm-project/vllm</a>: DescriptionRecently, I have submitted a pull request to Hugging Face Transformers containing the implementation of the Qwen3 and Qwen3MoE model. I would also like to contribute these new modelsto ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1352377976653746248)** (23 messages🔥): 

> `Esoteric Total Ordering, Unitree Robotics Kip-Up, Claude uses Brave Search, Capybara Logo Change, Token Counting Inflation` 


- ****New Faces (tm)** Sparks Esoteric Order Theories**: After seeing new faces, a member joked about believing in an *esoteric total ordering* to new faces.
   - The member jokingly analyzed an attached image, suggesting the last ordering was based on *hair volume ASC*.
- ****Unitree's G1** Nails the Kip-Up!**: [UnitreeRobotics](https://x.com/UnitreeRobotics/status/1903092199287578763) showcased their **G1 humanoid robot** performing a kip-up, celebrating the rapid advancement of humanoid intelligence.
   - A member responded saying that *this robot abuse will be a big theme in the revolution*.
- ****Brave** Chosen as Claude's Search Engine**: It was confirmed that **Claude's web search feature** uses [Brave Search](https://simonwillison.net/2025/Mar/21/anthropic-used-brave/), verified by a recent update to their *Trust Center* and matching search results, according to [Simon Willison](https://x.com/simonw/status/1903102096221815107).
   - The fact that Brave was chosen was received negatively, with members lamenting the difficulty of building better indexes and the death of the Bing API.
- **Capybara Logo De-Capybarized**: The **Capybara logo** is being reverted to a more generic logo, a decision lamented by some despite its broader appeal, as reported by [Justin Lin](https://x.com/JustinLin610/status/1903130983249088675).
   - A member reacted with crying emojis to the logo change and the *9 quadrillion tokens for training*.
- **Token Count Inflation Allegations**: A member questioned whether a token count was inflated by counting the token per image per frame per video.
   - Another member suggested that this inflation was almost certain, especially after the company revealed their tokenizer, implying a strategy to boost stock value.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/JustinLin610/status/1903130983249088675">Tweet from Junyang Lin (@JustinLin610)</a>: Turning the Capybara back to Logo. Suitable for more people but a little bit sad for us.</li><li><a href="https://x.com/UnitreeRobotics/status/1903092199287578763">Tweet from Unitree (@UnitreeRobotics)</a>: Movement creates intelligence - Unitree&#39;s G1 humanoid robot nails the world&#39;s first kip-up!😘This fresh, newly captured video from Unitree&#39;s testing grounds showcases the breakneck speed o...</li><li><a href="https://x.com/simonw/status/1903102096221815107">Tweet from Simon Willison (@simonw)</a>: I&#39;ve confirmed that the search engine being used by Claude&#39;s web search feature is @brave - it&#39;s listed in a recent update to their &#34;Trust Center&#34; and the search results are an exa...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1352357160197034085)** (2 messages): 

> `Anthropic Job Application, AI Alignment, Vibe Check` 


- **ChatGPT Crafts Cover Letter for Anthropic**: A user prompted **ChatGPT** to draft a [cover letter](https://example.com/coverletter) for an **Anthropic** application *just in case*.
   - This was presented as a proactive measure, even without concrete plans to apply, showcasing the user's interest in potential opportunities at **Anthropic**.
- **Alignment Vibe Check for Anthropic**: A user jokingly offered to help **Anthropic** improve its **AI alignment**, suggesting a need to elevate the AI's *main-character energy*.
   - The user humorously emphasized the importance of *deep vibes* and *verified blue-check alignment status* for **Anthropic's AI**, seemingly poking fun at contemporary social media culture and its integration into AI persona design.
- **Twitter Bot Career Path Joked for AI**: A member jokingly suggested that someone's message about improving **Anthropic's AI** sounded like a suitable style for a **Twitter comment bot**.
   - This comment implies that the tone and language used in the message were overly enthusiastic or promotional, fitting the stereotype of automated social media engagement.


  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1352623358919053323)** (6 messages): 

> `Sonnet 3.7 Benchmarks, InternVL Open Source Training Code, OpenAI operator use case` 


- ****Sonnet 3.7** Visual Benchmarks Remain Elusive**: A member inquired about comprehensive visual benchmarks for **Sonnet 3.7**, noting that only MMMU results were reported in the announcement.
   - So far, there is no response on benchmarks.
- ****InternVL**'s Open Training Code Draws Attention**: A member was surprised to discover that **InternVL** has open source training code.
   - They believe this makes **InternVL** and **Molmo** the only notable models with open training pipelines, pointing to [InternVL's packing implementation](https://github.com/OpenGVLab/InternVL/blob/34a81000402bf8f716bab8c9b57aff1f6b436bd0/internvl_chat/internvl/train/dataset_packed.py) as a resource for dataloading.
- **OpenAI Operator Finds Use in Dataset Iteration**: A member shared their initial use case for the **OpenAI operator**: iterating through **InternVL** datasets to find **ArXiv** and **Hugging Face** links, using a specific image to demonstrate this.
   - Another member suggested that deep research would work better for this use case.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1352508507190071322)** (16 messages🔥): 

> `RLAIF-V for MLLM Trustworthiness, Skill-Dependent Scaling Laws, Scaling RL Compute for Reasoning, SuperBPE Tokenizer, SWEET-RL for Multi-Turn LLM Agents` 


- ****RLAIF-V** Boosts MLLM Trustworthiness with Open-Source Feedback**: OpenBMB introduced **RLAIF-V**, a framework for aligning MLLMs using open-source feedback, claiming to surpass **GPT-4V** in trustworthiness, as detailed in their [paper](https://arxiv.org/abs/2405.17220) and [GitHub repository](https://github.com/RLHF-V/RLAIF-V).
   - The framework uses inference-time scaling with **RLAIF-V** reward and high-quality generalizable feedback data and splitting responses into atomic claims to sample different responses.
- **Knowledge and Reasoning Exhibit Different Scaling Behaviors**: A new [paper](https://arxiv.org/pdf/2503.10061) reveals that knowledge and reasoning skills show different scaling behaviors, suggesting that compute-optimal scaling is skill-dependent.
   - The research investigates the impact of various datamixes and found fundamental differences in scaling behavior between knowledge and code, even when correcting for datamix differences.
- ****Reinforcement Learning** Scales Reasoning Powers**: **Reinforcement learning** has enabled advanced reasoning capabilities in recent language models, enabling models to "think for longer" and reduce error rates via [inference-time scaling](https://gr.inc/blog/scaling-rl-compute/).
   - The blogpost questions how to scale **RL compute** further, discovering even more inference-time capabilities, with several open questions on priors.
- ****SuperBPE** Tokenizer Enhances Efficiency**: A new *superword* tokenizer, **SuperBPE**, was created which includes tokens spanning multiple words and at 8B scale, **SuperBPE** models consistently outperform the **BPE** baseline on 30 downstream tasks (+8% MMLU), while also being **27%** more efficient at inference time, as described in [this tweet](https://x.com/alisawuffles/status/1903125390618661068).
   - At a fixed vocab size of **200k**, **SuperBPE** reduces sequence length by **33%** on average.
- ****SWEET-RL** algorithm enhances LLM agent interaction**: A novel RL algorithm called **SWEET-RL** was proposed for multi-turn interactions in real-world tasks for LLM agents, detailed in this [paper](https://arxiv.org/abs/2503.15478).
   - **SWEET-RL** uses a carefully designed optimization objective to train a critic model with access to additional training-time information, providing step-level rewards for improving the policy model.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.17220">RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness</a>: Traditional feedback learning for hallucination reduction relies on labor-intensive manual labeling or expensive proprietary models. This leaves the community without foundational knowledge about how ...</li><li><a href="https://x.com/alisawuffles/status/1903125390618661068">Tweet from Alisa Liu (@alisawuffles)</a>: We created SuperBPE🚀, a *superword* tokenizer that includes tokens spanning multiple words.When pretraining at 8B scale, SuperBPE models consistently outperform the BPE baseline on 30 downstream task...</li><li><a href="https://x.com/OpenBMB/status/1902946478051799527">Tweet from OpenBMB (@OpenBMB)</a>: 🔥 Test-time Scaling for MLLMs’ trustworthiness🌟 Thrilled to introduce our new work RLAIF-V, a novel framework for aligning MLLMs through open-source feedback, achieving trustworthiness that surpasse...</li><li><a href="https://x.com/teortaxesTex/status/1902949418304717137">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: Wait, for real? They claim a large improvement on hallucinations with basically nothing more than splitting responses into atomic claims, sampling responses using different seeds for trustworthiness c...</li><li><a href="https://x.com/nick11roberts/status/1902875088438833291">Tweet from Nicholas Roberts (@nick11roberts)</a>: 📉📉NEW SCALING LAW PHENOMENON 📉📉 We find that knowledge and reasoning exhibit different scaling behaviors! Super excited to finally tell you all about our paper on the compute optimal scaling of sk...</li><li><a href="https://x.com/alisawuffles/status/1903125395043652068">Tweet from Alisa Liu (@alisawuffles)</a>: What can we gain from less restrictive tokenization? To find out, we developed SuperBPE🚀, which learns subword *and* superword tokens. SuperBPE dramatically improves encoding efficiency over BPE — at...</li><li><a href="https://arxiv.org/abs/2503.10061">Compute Optimal Scaling of Skills: Knowledge vs Reasoning</a>: Scaling laws are a critical component of the LLM development pipeline, most famously as a way to forecast training decisions such as &#39;compute-optimally&#39; trading-off parameter count and dataset...</li><li><a href="https://arxiv.org/abs/2503.15478">SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks</a>: Large language model (LLM) agents need to perform multi-turn interactions in real-world tasks. However, existing multi-turn RL algorithms for optimizing LLM agents fail to perform effective credit ass...</li><li><a href="https://gr.inc/blog/scaling-rl-compute/">Scaling RL Compute | General Reasoning</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1352462777058328646)** (6 messages): 

> `Scaling Laws for Language Models, Data Requirements for GPT-4.5` 


- **Bigger Models Benefit from More Data**: Members discussed whether **smaller datasets** are a new trend, with larger models like **GPT-4.5** potentially needing more data, especially during various post-training stages.
   - One member suggested examining the number of tokens used in training open-source model suites for different model sizes, referencing scaling laws, with another asking about papers on the subject.
- **Synthetic Data Augmentation**: The conversation touches on the use of synthetic data to augment smaller datasets for training smaller models, suggesting a trade-off between data size, model size, and the use of synthetically generated data.
   - This implies a strategy where smaller models might rely more on enhanced, possibly synthetic, datasets, while larger models can effectively utilize larger volumes of raw data.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1352356495047786537)** (108 messages🔥🔥): 

> `p2l-router-7b-0318 Model, Claude Overrated?, Google AI Studio API, Deepseek R1, Qwen 3 Coming Soon` 


- **Is Claude really all that, or is everyone just simping?**: Members believe people are overrating **Claude**, questioning its coding prowess due to limited evaluations beyond **SWE-bench**, suggesting it might not match **Grok3** on livecodebench.
   - Some suggest that ratings may be skewed by non-developers, leading to inaccurate assessments of its true capabilities.
- **Gemma gets Glowing**: Community members expressed amazement at **Gemma3's 1340** score and relatively small **27B** parameter size.
   - One member described **Gemma's** responses as *autistic*, giving very brief answers, often when a much longer one is warranted.
- **Deepseek R1 eats VRAM**: **Deepseek R1** requires a substantial amount of VRAM, around **1000GB**, with one user running it on **8xH200s**.
   - Despite high VRAM usage, there are claims that **Deepseek R1** exhibits baked-in *PRO CHINA* biases, raising concerns about its use, with one user saying *tldr deepseek is #&*&*@% don't recommend using it*.
- **Qwen 3 Sneak Peek?**: There are reports that **Qwen 3** is coming soon, indicated by a post on the Hugging Face Transformer repository.
   - This news follows the announcement of **Qwen 2.5 Omni**, sparking interest and anticipation within the community.
- **OpenAI's Sora: Hype vs. Reality?**: Users found **Sora's** public release underwhelming compared to its promotional materials, even inferior to competitors like **Keling AI** and **Hailuo AI**.
   - It's suspected that OpenAI used huge amounts of compute over hours to generate them and the released **Sora** version is the *turbo version*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Presidentlin/status/1903102260155908200">Tweet from Lincoln 🇿🇦 (@Presidentlin)</a>: Qwen 3 coming pr on hugging face transformer repo</li><li><a href="https://dev.to/askyt/deepseek-r1-671b-complete-hardware-requirements-optimal-deployment-setup-2e48">no title found</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3">unsloth/DeepSeek-V3 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1352403398992990218)** (21 messages🔥): 

> `Podcast Feature in NotebookLM, NotebookLM vs Gemini, Efficient PDF Processing Workflow, AI Avatar Lip Syncing Services, Mindmap Feature rollout` 


- **Users Blown Away by NLM's Podcast Feature**: Users are very happy about the Podcast feature of NLM, but one user felt like a *third wheel* in the discussions because the AI tended to cut short their answers and revert to its own script.
   - The user likened the experience to *being part of a radio show where I can talk to hosts*.
- **NotebookLM Stays Grounded in Provided Sources**: A user questioned the advantage of using NotebookLM over Gemini, as both support files like PDFs.
   - A member found that [Gemini](https://gemini.google.com/) did not stay grounded in the sources, whereas NotebookLM's distinguishing aspect is that it **only uses the sources provided**.
- **Streamline PDF Processing Workflow**: A user seeks an efficient way to **declutter physical papers**, scan them into private online storage, and make them searchable by natural language queries.
   - The current manual process involves scanning to PDF, sending to Gmail, manually naming each file, OCR processing, and importing into NotebookLM, and the user asks if taking photos with iPhone and sending to NLM for automatic naming and OCR is more efficient.
- **AI Avatar Lip Syncing Showdown**: Members are comparing lip syncing services for AI avatars, with one user finding [Hedra](https://www.hedra.io/) great but pricey, and being unimpressed with [RunwayLM](https://runwayml.com/).
- **Mindmap Feature Still Rolling Out**: A user inquired about the absence of the **Mindmap** feature in their NotebookLM, and another user clarified that the feature is rolling out over a **two-week period**.
   - The feature is not available even for NLM Plus subscribers yet.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1352371666776490095)** (74 messages🔥🔥): 

> `Flashcard Generation, NotebookLM vs Chatbase, Premium Voice Overview Limits, Mind Map Feature Rollout, Whitelist NotebookLM Crawler` 


- ****Flashcard Feature** Requested by Plus User**: A Plus user requested the integration of **Flashcard generation (Anki)** in NotebookLM.
   - They expressed disappointment, stating that [chatbase](https://chatbase.co/) is currently a better chatbot agent.
- ****Mind Map** Rollout Happening Slowly**: Users are reporting that the **Mind Map** feature is doing a slow rollout, with many not seeing it on their accounts despite being Plus users.
   - A staff member confirmed that it will take a *few days* for all users to get it and they should *sit tight and wait*.
- ****Cloudflare Blocking** NotebookLM Crawler**: A user asked how to technically whitelist the NotebookLM crawler, as Cloudflare was blocking it on their website.
   - The user found out it was Cloudflare blocking it.
- ****PDF Sources** Getting Cut Off**: A user reported that their **PDF source** was being cut off, with NotebookLM not recognizing details near the end of the file.
   - A staff member suggested that the preview may not be ground truth and to file the issue under bugs if asking about the end of the document yields no results.
- ****Gemini 1.5 Pro** Powering NotebookLM**: A user inquired whether NotebookLM is boosted by **Gemini 1.5 Pro**.
   - Another asked what model NotebookLM uses and another user answered Gemini 2.0.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1352362244486332567)** (79 messages🔥🔥): 

> `Nvidia Blackwell RTX Pro series, Data filtering strategies, DeepHermes 24B OOM issues, WorldSim appreciation` 


- **Nvidia's Blackwell RTX Pro Enters the Ring**: Nvidia has introduced the [Blackwell RTX Pro series](https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus) targeting laptops, desktops, standalone PCs, and data centers, potentially tightening the already limited supply of Blackwell GPUs.
   - Sources at GDC/GTC suggest Nvidia aims to improve Blackwell GPU supply, hinting that supply might meet demand by May/June, though skepticism remains: *"We'll believe that when we see it."
- **Dataset Evaluation & Augmentation reign supreme**: Discussion emphasized that the most efficient use of GPU hours lies in dataset evaluation, augmentation, sorting, and categorization.
   - One member suggested filtering data using a small model to reject data it predicts well, noting this area as *"underexplored in public."
- **DeepHermes 24B Struggles on Multi-GPU Rig**: A user faced Out-of-Memory (OOM) errors running DeepHermes 24B on a 5x 3090 rig with llama.cpp, even with the lowest context settings.
   - Suggestions included trying the **8-bit version** and checking multi-GPU configuration, with advice on using `--device`, `--split-mode`, and `--tensor-split` flags for proper GPU utilization.
- **WorldSim Sparks Awe: User Calls Nous Research 'Godly'**: A member expressed immense enthusiasm for **WorldSim**, praising Nous Research for creating an incredible application of AI and stating it's *"absolutely epic!!"
   - The user was enthralled by the application and emphasized its masterclass quality, regretting not discovering it sooner, *"Thanks so much Nous Research for creating such an incredible application of AI!"*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.00808">Predictive Data Selection: The Data That Predicts Is the Data That Teaches</a>: Language model pretraining involves training on extensive corpora, where data quality plays a pivotal role. In this work, we aim to directly estimate the contribution of data during pretraining and se...</li><li><a href="https://www.tomshardware.com/pc-components/gpus/nvidia-blackwell-rtx-pro-with-up-to-96gb-of-vram-even-more-demand-for-the-limited-supply-of-gpus">Nvidia Blackwell RTX Pro with up to 96GB of VRAM &mdash; even more demand for the limited supply of GPUs</a>: GB202, GB203, and GB205 are coming to professional and data center GPUs. (Updated with full specs.)</li><li><a href="https://arxiv.org/abs/2409.17115">Programming Every Example: Lifting Pre-training Data Quality Like Experts at Scale</a>: Large language model pre-training has traditionally relied on human experts to craft heuristics for improving the corpora quality, resulting in numerous rules developed to date. However, these rules l...</li><li><a href="https://github.com/karpathy/llm.c/discussions/481">Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20 · karpathy/llm.c · Discussion #481</a>: Let&#39;s reproduce the GPT-2 (124M) in llm.c (~4,000 lines of C/CUDA) in 90 minutes for $20. The 124M model is the smallest model in the GPT-2 series released by OpenAI in 2019, and is actually quite...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1352395556521119877)** (8 messages🔥): 

> `Hermes 3 Llama 3.2 3B, Model Parameters, Response Generation Issues` 


- **Nous Research Unveils Hermes 3 Llama 3.2 3B**: Nous Research introduced **Hermes 3 3B**, a small but mighty addition to the Hermes series of LLMs, detailed in the [Hermes 3 Technical Report](https://arxiv.org/abs/2408.11857).
   - Hermes 3 boasts improvements over Hermes 2, including advanced agentic capabilities, better roleplaying, reasoning, multi-turn conversation, long context coherence.
- **Tweaking Parameters for Casual Chat with Hermes 3**: A user experimented with **Hermes-3-Llama-3.2-3B-GGUF** on Hugging Face, using a payload with parameters like `temperature`, `top_k`, `top_p`, and `repeat_penalty` to generate responses.
   - Initially, the user set `temperature` to **0.2** and `top_p` to **0.7**, but considered that `0.7` and `0.85` ranges might be better, respectively.
- **User Impersonation Glitch in Hermes 3**: A user reported issues with **Hermes 3**, where it sometimes impersonates the `user` instead of maintaining its AI persona.
   - The user is attempting to refine the system prompt and payload parameters to ensure logical and consistent AI-generated responses.



**Link mentioned**: <a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B-GGUF">NousResearch/Hermes-3-Llama-3.2-3B-GGUF · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/nick11roberts/status/1902875088438833291?s=46
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/nick11roberts/status/1902875088438833291?s=46
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1352421304397205565)** (3 messages): 

> `Nous Hermes 2, C# Development, Anthropic LLMs` 


- **Developer professes love for Nous Hermes 2**: A member reminisced about starting with **Nous Hermes** a year ago, specifically mentioning **Nous Hermes 2** as their *"beloved"*.
   - They have released their first desktop app and are figuring out which model to use for version 2.
- **Developer offers C# skills to the community**: A member offered their help, mentioning **C#** as their *"love language"* and highlighting their professional **LLM** experience.
   - They have created several documentation and example LLMs for **Anthropic**.
- **Anthropic LLM examples detailed**: A member mentioned their work on **Anthropic LLMs**, including a **Titanfall 2**-based generator and the **Bladewolf** example from **Metal Gear Rising**.
   - Their contributions can be found on the [Anthropic GitHub](https://github.com/).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1352367427064696954)** (50 messages🔥): 

> `Hugging Face API Outage, Roblox Voice Safety Classifier, Local Models for Speed & Privacy vs Cloud Models, Merge Multiple GPU VRAM, MagicQuill Low Quality Images` 


- **HF API's 404 Error Causes App Downtime**: A user reported widespread **404 errors** affecting multiple Hugging Face API models, causing significant downtime for dependent applications, and requested immediate attention from the HF dev team, noting it had been *almost a whole day* without official acknowledgement.
   - Another member tagged a HuggingFace employee to raise awareness to this urgent issue experienced by paid users.
- ****Roblox** Releases **Voice Safety Classifier** for Toxicity Detection**: **Roblox** released a [large classification model](https://huggingface.co/Roblox/voice-safety-classifier) trained on a manually curated real-world dataset of **2,374 hours** of voice chat audio clips.
   - The model outputs a n by 6 output tensor where the inferred labels are `Profanity`, `DatingAndSexting`, `Racist`, `Bullying`, `Other`, `NoViolation` based on a synthetic data pipeline described in [this blog post](https://research.roblox.com/tech-blog/2024/06/deploying-ml-for-voice-safety).
- **Fuse VRAM from multiple GPUs via Tensor Parallelism**: Users discussed methods for combining VRAM from multiple GPUs, particularly for running models like **Gemma3-12B**, with one user asking if there's a way to combine an **A2000 12GB** and a **1060 6GB**; the main recommendation was using [tensor parallelism](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_multi).
   - One member pointed to [Ollama issues on GitHub](https://github.com/ollama/ollama/issues/2672) ([2](https://github.com/ollama/ollama/issues/8995)) and [llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions/8725) for further information on multi-GPU support.
- ****Oblix** Platform Dynamically Executes AI Tasks on Cloud or Device**: [Oblix.ai](https://oblix.ai) platform uses autonomous agents for intelligent AI orchestration that dynamically executes between cloud and on-device models, ensuring optimal performance, cost-efficiency, and security; they intelligently route AI tasks to cloud or edge based on complexity, latency requirements, and cost considerations.
   - Oblix dynamically decides whether to process each AI request locally or in the cloud as shown in [this YouTube video](https://youtu.be/j0dOVWWzBrE?si=Gpp2SG4Kly0tzM_3).
- **Gradio Upgrade Breaks gr.Dataframe Wrapping Feature**: A user reported that upgrading to **Gradio 5.22** caused the `gr.Dataframe(wrap=True)` feature to stop working, with wrapping only functioning in **Gradio 5.20**.
   - There were no other details given.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Roblox/voice-safety-classifier">Roblox/voice-safety-classifier · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/settings/notifications">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/settings/organizations>">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://weaviate.io/platform">Open Source Vector Database | Weaviate</a>: Simplify the development of AI applications and enable developers of all levels to build, iterate, and scale AI capabilities faster.</li><li><a href="https://discuss.huggingface.co/t/hf-inference-api-last-few-minutes-returns-the-same-404-exception-to-all-models/146646/20">HF Inference API last few minutes returns the same 404 exception to all models</a>: I think its due to the server error/issues, im getting this now as well instead of 404</li><li><a href="https://huggingface.co/spaces/AI4Editing/MagicQuill">MagicQuill - a Hugging Face Space by AI4Editing</a>: no description found</li><li><a href="https://huggingface.co/spaces/AI4Editing/MagicQuill/discussions/5">AI4Editing/MagicQuill · I edited my image but then had no idea how to download the edited output image?</a>: no description found</li><li><a href="https://huggingface.co/spaces/AI4Editing/MagicQuill/discussions">AI4Editing/MagicQuill · Discussions</a>: no description found</li><li><a href="https://huggingface.co/spaces?category=image-generation&sort=trending">Spaces - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/Yntec/MiniToyWorld">Mini Toy World - a Hugging Face Space by Yntec</a>: no description found</li><li><a href="https://oblix.ai">Transform Your AI Performance with Intelligent Hybrid Orchestration | Oblix.ai</a>: Experience our interactive demo and see how our intelligent agents seamlessly switch between local LLM execution and cloud providers for optimal performance and cost efficiency.</li><li><a href="https://github.com/Neph0s/awesome-llm-role-playing-with-persona">GitHub - Neph0s/awesome-llm-role-playing-with-persona: Awesome-llm-role-playing-with-persona: a curated list of resources for large language models for role-playing with assigned personas</a>: Awesome-llm-role-playing-with-persona: a curated list of resources for large language models for role-playing with assigned personas - Neph0s/awesome-llm-role-playing-with-persona</li><li><a href="https://www.reddit.com/r/SillyTavernAI/comments/1d8s4gd/beginner_llm_comparison_for_instructionfollowing/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_multi">Distributed GPU inference</a>: no description found</li><li><a href="https://github.com/ollama/ollama/issues/2672"> Do Ollama support multiple GPUs working simultaneously? · Issue #2672 · ollama/ollama</a>: I have 8 RTX 4090 GPUs. Can they support a 70B-int4 parameter model?</li><li><a href="https://github.com/ollama/ollama/issues/8995">Ollama is splitting the model between CPU and one GPU instead of using second GPU · Issue #8995 · ollama/ollama</a>: What is the issue? Problem description My Setup I use ollama on my Laptop with an external GPU. My Laptop has an internal Nvidia Quadro M2000M. Over Thunderbolt 3 I have a Razer Core X Chroma eGPU ...</li><li><a href="https://github.com/ggml-org/llama.cpp/discussions/8725">How to properly use llama.cpp with multiple NVIDIA GPUs with different CUDA compute engine versions? · ggml-org/llama.cpp · Discussion #8725</a>: I have an RTX 2080 Ti 11GB and TESLA P40 24GB in my machine. First of all, when I try to compile llama.cpp I am asked to set CUDA_DOCKER_ARCH accordingly. But according to what -- RTX 2080 Ti (7.5)...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

richieghost: Today I'm learning Pytorch Frame.
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1352357172519895131)** (3 messages): 

> `Ollama Gradio UI with Kokoro TTS, Little-Geeky-s-Learning-UI, Oblix AI orchestration platform, Edge-Cloud transitions` 


- **Little Geeky Learns New UI Tricks**: A member showcased a new UI built with **Ollama**, **Gradio**, and **Kokoro TTS** that automatically reads text output in a chosen voice and has model creation and management tools.
   - The UI can read ebooks and answer questions about documents, as well as work with vision models to do the same for images, and an audio file is output from the UI.
- **GeekyGhost shares Little-Geeky-s-Learning-UI**: The [Little-Geeky-s-Learning-UI](https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI) is an **Ollama** based **Gradio UI** that uses **Kokoro TTS**.
   - It allows model creation and management, reads ebooks, answers questions about documents, and works with vision models.
- **Oblix orchestrates edge-cloud transitions**: [Oblix.ai](https://oblix.ai) is an **AI orchestration platform** powered by autonomous agents that dynamically executes between cloud and on-device models, ensuring optimal performance, cost-efficiency, and security.
   - The platform features **intelligent routing**, **performance optimization**, **execution agents**, and **cost efficiency**, and a [demo is available on YouTube](https://youtu.be/j0dOVWWzBrE?si=Gpp2SG4Kly0tzM_3).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://oblix.ai">Transform Your AI Performance with Intelligent Hybrid Orchestration | Oblix.ai</a>: Experience our interactive demo and see how our intelligent agents seamlessly switch between local LLM execution and cloud providers for optimal performance and cost efficiency.</li><li><a href="https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git">GitHub - GeekyGhost/Little-Geeky-s-Learning-UI: An Ollama based Gradio UI that uses Kokoro TTS</a>: An Ollama based Gradio UI that uses Kokoro TTS. Contribute to GeekyGhost/Little-Geeky-s-Learning-UI development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 messages): 

.mwayne: https://blog.roboflow.com/fine-tune-sam-2-1/amp/
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1352359474123243672)** (2 messages): 

> `Manual Looping vs Vectorization, GSM8K Dataset, Tokenizer ChatML Format, Certifications` 


- **Vectorized Processing triumphs Manual Looping**: One member found vectorization performed *much better* than manual looping.
   - The original poster reported that they had implemented it manually with a `for` loop, describing it as *kind of round-about*.
- **GSM8K Dataset Difficulties Arise**: A member expressed trouble understanding the next notebook task involving the **GSM8K dataset**.
   - The member was especially confused about the instructions to *create a message format with the role and content*.
- **Tokenizer's ChatML Format Examined**: Doubts were raised on whether the **tokenizer method** always implements the same **ChatML format**.
   - The member questioned how the function knows how the original dataset is formatted.
- **Certification Assignment Location**: One member inquired where they could find the assignment to get **certifications**.
   - No further details were provided about the specific certification or platform.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1352413037948309698)** (24 messages🔥): 

> `HF Course Certificate, Unit 2.1 Error, AI agent for UI automation, Langfuse Error, Smolagent model to run locally` 


- **HF Course Certificate Achievement**: A member asked about how to obtain a certificate after completing **Unit 2** of the Hugging Face course.
   - Another member requested the inclusion of an **AI agent for UI automation**.
- **HF Learners Encounter Unit 2.1 Issues**: A member encountered an error while running the code from **Unit 2.1** *Building Agents That Use Code* in their own Python environment, the image showed the code crashing.
   - A member suggested using the **hf_token** to run the model, check terms and conditions, or verify the HFApi object contains the token.
- **Langfuse Integration Challenges**: Members reported encountering an **AttributeError** related to the *smolagents* module when trying to connect to **Langfuse** in the **unit2/smolagents/code_agents.ipynb** notebook.
   - A maintainer acknowledged the issue and pointed to a bug report [here](https://github.com/Arize-ai/openinference/issues/1399) and fix [here](https://github.com/Arize-ai/openinference/pull/1403) related to the *openinference* instrumentation library.
- **Exploration of Local Smolagent Models**: A member inquired about the best **smolagent model** to run locally and how to implement it in a working program.
   - The member shared challenges when implementing the **multiagents module**.
- **AI Agents Course: A learning review**: A member shared their experience and learnings from completing unit 1 of the AI Agents course in a [Medium blog post](https://medium.com/@jurriaan.nagelkerke/the-ai-agents-course-a-review-b289f90799ea).
   - They shared tips on how to get the most out of the first unit.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@jurriaan.nagelkerke/the-ai-agents-course-a-review-b289f90799ea">The 🤗 AI Agents Course: A review</a>: As a data scientist or LLM enthusiast, you hear and read about agents everywhere now. Unfortunately, not everyone has the same idea when…</li><li><a href="https://github.com/Arize-ai/openinference/issues/1399">[bug] SmolagentsInstrumentor - AttributeError: module &#39;smolagents&#39; has no attribute &#39;ApiModel&#39; · Issue #1399 · Arize-ai/openinference</a>: Describe the bug When following the guide on instrumenting Hugging Face smolagents using the SmolagentsInstrumentor, I get the following error: AttributeError: module &#39;smolagents&#39; has no attri...</li><li><a href="https://github.com/Arize-ai/openinference/pull/1403">fix: only import exported smolagents models by njbrake · Pull Request #1403 · Arize-ai/openinference</a>: Fixes #1399
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1352360167072596091)** (69 messages🔥🔥): 

> `mcp-mysql-server issues, fastmcp Framework, Vibe Coding, DaVinci Resolve MCP update, Glama API outage` 


- ****MySQL & MCP Hookup Headache****: A user is wrestling with **mcp-mysql-server** connecting to **Docker MySQL**, reporting it bombs every connection despite working outside of **MCP**.
- ****Fastmcp Framework Frustrations****: A user suspects the **fastmcp framework** is stripping out *hidden* arguments passed to commands, casing issues from `RegisterCommandsSchema` to `_RegisterCommandsSchema` in their code.
- ****Vibe Coding Debate Sparks****: Some users joked about *vibe coding*, where the coder looks at the screen, nothing makes sense, but somehow gets results.
- ****DaVinci Resolve MCP Seeks Speedy Server Claim****: A user is seeking to resubmit their **DaVinci Resolve MCP** project with a license and updates, and was told claiming the server might speed up the update process, directing them to their [repo](https://github.com/samuelgursky/davinci-resolve-mcp).
- ****Glama API Grievances Galore****: A user reported getting a **500 error** from the **Glama API**, but another member stated that there have been no outages in the last 24 hours, with others sharing code samples to reproduce: `curl -X 'GET' 'https://glama.ai/api/mcp/v1/servers?first=10&query=github' -H 'accept: application/json'`.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jamiew/status/1903111313968046556">Tweet from Jamie Dubs (@jamiew)</a>: 1. MCP is complex, overengineered, hard to host in cloud, security model is nonexistent2. MCP is also the best current solution for &#34;LLM plugins&#34; or &#34;SDK but for LLMs&#34;. everything else...</li><li><a href="https://github.com/cunhapaulo/marpstyle">GitHub - cunhapaulo/marpstyle: Repository for Marp Themes created with beauty and simplicity in mind.</a>: Repository for Marp Themes created with beauty and simplicity in mind. - cunhapaulo/marpstyle</li><li><a href="https://github.com/ggozad/oterm">GitHub - ggozad/oterm: a text-based terminal client for Ollama</a>: a text-based terminal client for Ollama. Contribute to ggozad/oterm development by creating an account on GitHub.</li><li><a href="https://github.com/samuelgursky/davinci-resolve-mcp">GitHub - samuelgursky/davinci-resolve-mcp: MCP server integration for DaVinci Resolve</a>: MCP server integration for DaVinci Resolve. Contribute to samuelgursky/davinci-resolve-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/markuspfundstein/mcp-gsuite">GitHub - MarkusPfundstein/mcp-gsuite: MCP Server to interact with Google Gsuite prodcuts</a>: MCP Server to interact with Google Gsuite prodcuts - MarkusPfundstein/mcp-gsuite</li><li><a href="https://github.com/isaacphi/mcp-gdrive">GitHub - isaacphi/mcp-gdrive: Model Context Protocol (MCP) Server for reading from Google Drive and editing Google Sheets</a>: Model Context Protocol (MCP) Server for reading from Google Drive and editing Google Sheets - isaacphi/mcp-gdrive</li><li><a href="https://github.com/kazz187/mcp-google-spreadsheet">GitHub - kazz187/mcp-google-spreadsheet: MCP Server for Google Spreadsheet</a>: MCP Server for Google Spreadsheet. Contribute to kazz187/mcp-google-spreadsheet development by creating an account on GitHub.</li><li><a href="https://github.com/distrihub/mcp-google-workspace">GitHub - distrihub/mcp-google-workspace: A Model Context Protocol (MCP) server built in Rust for interacting with Google Drive and Google Sheets.</a>: A Model Context Protocol (MCP) server built in Rust for interacting with Google Drive and Google Sheets. - distrihub/mcp-google-workspace</li><li><a href="https://github.com/akchro/google-sheets-mcp">GitHub - akchro/google-sheets-mcp</a>: Contribute to akchro/google-sheets-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/rishipradeep-think41/drive-mcp">GitHub - rishipradeep-think41/drive-mcp</a>: Contribute to rishipradeep-think41/drive-mcp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1352359324382527518)** (6 messages): 

> `Microsoft Semantic Workbench, Turso MCP tool video, Asana MCP + Google Calendar MCP, MCPHub.nvim + Avante + Figma MCP` 


- **Microsoft Releases Semantic Workbench Tool**: Microsoft released the [Semantic Workbench](https://github.com/microsoft/semanticworkbench), a VS Code extension, described as a versatile tool to prototype intelligent assistants, agents, and multi-agentic systems, prompting questions about its role as an **MCP**.
   - One member asked if it was an **MCP**, after looking at its description.
- **Turso MCP Tool Demoed by Jamie**: Jamie created a video of the Turso MCP tool, built by the @tursodatabase community and showcased in a [tweet](https://x.com/notrab/status/1902767330007941472?s=46&t=4RSOl8kQCdkHm0U5FcdeaA).
   - The video shows Claude creating a database for a domain collection.
- **Automated Calendar Scheduling with Asana & Google Calendar MCPs**: A blog post detailed using **Asana MCP** and **Google Calendar MCP** with Goose to automate task scheduling, illustrating how tasks are pulled from Asana, analyzed, and scheduled in Google Calendar with a single prompt. The [blog post](https://block.github.io/goose/blog/2025/03/20/asana-calendar-mcp) highlights the time-saving benefits of automating the organization of tasks and meetings.
   - The **Asana MCP** server is located at [Asana](https://github.com/roychri/mcp-server-asana) and the **Google Calendar MCP** is located at [Google Calendar](https://www.pulsemcp.com/servers?q=google+calendar).
- **nvim Integrates with Figma**: A user showcased an integration of **MCPHub.nvim** with **Avante** and **Figma MCP**, demonstrating a streamlined workflow as seen in the shared video [login-figma.mp4](https://cdn.discordapp.com/attachments/1315696461316358175/1352480836896952340/login-figma.mp4?ex=67ded42f&is=67dd82af&hm=1997f11d01ec6f95e16f5e9c69b326c6c9c06cb5124da34a4b74fb7d95a8293f).
   - Other users expressed interest with comments such as *"Looks cool"*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/notrab/status/1902767330007941472?s=46&t=4RSOl8kQCdkHm0U5FcdeaA">Tweet from Jamie Barton (@notrab)</a>: Here I ask Claude to create a database for my domain collection. Don&#39;t worry, I didn&#39;t include the full list, the video is only 90 seconds.👏 Huge shout out to @spences10 and the @tursodatabas...</li><li><a href="https://block.github.io/goose/blog/2025/03/20/asana-calendar-mcp">MCP in Action: How I Use AI to Plan My Week with Goose, Asana, and Google Calendar</a>: Use MCPs with Goose to automate task management and enhance productivity.</li><li><a href="https://github.com/microsoft/semanticworkbench">GitHub - microsoft/semanticworkbench: A versatile tool designed to help prototype intelligent assistants, agents and multi-agentic systems</a>: A versatile tool designed to help prototype intelligent assistants, agents and multi-agentic systems  - GitHub - microsoft/semanticworkbench: A versatile tool designed to help prototype intelligent...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1352358751738269696)** (64 messages🔥🔥): 

> `OpenRouter TTS, Ernie Models, Sambanova, Inferencenet, OpenAI audio models` 


- **OpenRouter to Offer TTS, Image Gen, at a Cost**: A member expressed interest in **OpenRouter** offering **TTS and image generation**, but voiced concerns about potentially high pricing.
- **Groq vs Sambanova mixup**: A member initially reported that **Sambanova** was down, but then corrected themselves, stating that it was **Groq** experiencing issues.
- **GPT-4o Arrives**: A user noticed that **GPT-4o-64k-output-alpha** is available on **OpenRouter**, supporting both **text and image inputs with text outputs** at the cost of **$6/M input tokens** and **$18/M output tokens**.
- **Reasoning models usage data shared**: A member published [token usage data](https://dubesor.de/reasoningtok) and thoughts on **reasoning models**, comparing them to traditional models.
- **Fireworks Reduces Pricing, Matching Performance**: **Fireworks** lowered their pricing for **R1 and V3**, with V3 reportedly matching existing performance metrics, specifically **.9/.9**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - Manage Model Usage and Quotas</a>: Learn about OpenRouter&#x27;s API rate limits, credit-based quotas, and DDoS protection. Configure and monitor your model usage limits effectively.</li><li><a href="https://openrouter.ai/openai/gpt-4o:extended">GPT-4o (extended) - API, Providers, Stats</a>: GPT-4o (&quot;o&quot; for &quot;omni&quot;) is OpenAI&#x27;s latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/open...</li><li><a href="https://huggingface.co/nghuyong">nghuyong (HuYong)</a>: no description found</li><li><a href="https://openrouter.ai/docs/faq#api-technical-specifications">OpenRouter FAQ</a>: Find answers to commonly asked questions about OpenRouter&#x27;s unified API, model access, pricing, and integration.</li><li><a href="https://fireworks.ai/pricing">Fireworks - Fastest Inference for Generative AI</a>: Use state-of-the-art, open-source LLMs and image models at blazing fast speed, or fine-tune and deploy your own at no additional cost with Fireworks AI!</li><li><a href="https://github.com/mintsuku/sora">GitHub - mintsuku/sora: Sora is a Discord bot that integrates with the Open Router API to facilitate conversation in Discord servers.</a>: Sora is a Discord bot that integrates with the Open Router API to facilitate conversation in Discord servers. - mintsuku/sora</li><li><a href="https://tenor.com/view/eyes-burning-my-eyes-spongebob-gif-5183364">Eyes GIF - Eyes Burning My Eyes - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1352366344917745684)** (9 messages🔥): 

> `vast.ai ncu profiling, Jake, Spam detection with neural nets` 


- **NCU Profiling on Vast.ai in Question**: A member inquired if [vast.ai](https://vast.ai) allows for **ncu profiling**.
   - Another member responded that while someone with the handle *Jake* is present, they doubt bare metal access is provided.
- **Spam Auto-Detection via Neuro Nets**: A member mentioned knowing a server that implemented automatic detection of spam messages using some kind of **neuro nets**.
   - No further details or links were provided.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1352378570441232445)** (6 messages): 

> `cuTile talk, atomic addition with bfloat16, triton 3.1.0 and triton-windows 3.2.0, Triton's ease of use, sparse attention pattern` 


- **Channel Eyes cuTile Talk**: A member suggested inviting someone to give a talk about **cuTile** on the channel and the suggestion is [in the works](https://www.nvidia.com/en-us/cuda/cutile/).
   - No further details were provided.
- **BFloat16 Atomic Addition Achieved Via Locks**: A member reported that using `tl.atomic_cas` with a lock for **atomic addition with bfloat16** actually works, but [it sucks](https://github.com/openai/triton/blob/main/python/triton/runtime/jit.py).
   - The member is seeking improvements to the implementation, and offered a code snippet using `tl.atomic_cas` with a lock, inviting the community to enhance its performance.
- **Triton Versions Clash Post-Install**: After successfully running `triton_test.py`, a member found both **triton 3.1.0** and **triton-windows 3.2.0** listed in pip, expressing hesitation to uninstall the older version due to numerous files shown in the CMD.
   - They sought advice on whether to uninstall **triton 3.1.0**, but no solutions were offered.
- **Triton's Simplicity Attracts GPU Newbies**: A member highlighted that **Triton's** key strength lies not in peak performance, but in its accessibility, enabling individuals with limited GPU experience to create complex kernels, and pointed to [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch/blob/main/native_sparse_attention_pytorch/triton_native_sparse_attention.py) as an example.
   - They noted that achieving peak performance on predefined workloads is relatively straightforward, but Triton's robustness is what sets it apart.



**Link mentioned**: <a href="https://github.com/lucidrains/native-sparse-attention-pytorch/blob/main/native_sparse_attention_pytorch/triton_native_sparse_attention.py">native-sparse-attention-pytorch/native_sparse_attention_pytorch/triton_native_sparse_attention.py at main · lucidrains/native-sparse-attention-pytorch</a>: Implementation of the sparse attention pattern proposed by the Deepseek team in their &quot;Native Sparse Attention&quot; paper - lucidrains/native-sparse-attention-pytorch

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1352481236647411834)** (3 messages): 

> `FlashMLA SmemLayoutP, Pointer Tagging` 


- **FlashMLA's `SmemLayoutP` Dimension Decoded**: A member inquired about the dimensions of `SmemLayoutP` in the [FlashMLA](https://github.com/deepseek-ai/FlashMLA/blob/b31bfe72a83ea205467b3271a5845440a03ed7cb/csrc/flash_fwd_mla_kernel.h#L76C5-76C93) code, specifically its shape `((2,2), kNThreadsS, 1, kBlockN/8)` and the role of `kNThreadsS` in synchronizing **P** between warpgroups.
   - The member speculated whether other dimensions might be related to **wgmma**, awaiting clarification from other experts.
- **Pointer Tagging Pitfalls Pondered**: A member asked about potential pitfalls when implementing **pointer tagging** using `uint32_t*`, referencing the programming guide's suggestion of 17 available bits.
   - They included an [image](https://cdn.discordapp.com/attachments/1189607726595194971/1352700676869980180/image.png?ex=67def82d&is=67dda6ad&hm=a2758b3f742c29f70d532f4c615325152feb59225e29ab987c394a34306a5662) from the programming guide for context.



**Link mentioned**: <a href="https://github.com/deepseek-ai/FlashMLA/blob/b31bfe72a83ea205467b3271a5845440a03ed7cb/csrc/flash_fwd_mla_kernel.h#L76C5-L76C93">FlashMLA/csrc/flash_fwd_mla_kernel.h at b31bfe72a83ea205467b3271a5845440a03ed7cb · deepseek-ai/FlashMLA</a>: FlashMLA: Efficient MLA decoding kernels. Contribute to deepseek-ai/FlashMLA development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1352405039460913223)** (5 messages): 

> `ZeRO offload, full-finetuning, 8B model, BF16, A100 40GB` 


- **Zeroing in on ZeRO Offload for 8B Finetuning**: A member inquired if **ZeRO offload** would enable full fine-tuning of an **8B model** (in **BF16**) on a single **A100 40GB** GPU.
   - Another member suggested that **ZeRO** is mostly for distributed training, and that checkpointing + gradient accumulation would be better for a single GPU.
- **FSDP2's Offload Parameter Examined**: One member mentioned that **FSDP2** has an *offload_to_cpu* parameter, while another suggested that a better starting point would be [torchao's offload optimizer](https://pytorch.org/torchao/).
- **DeepSpeed's Zero Offload Claims**: A member mentioned that [deepspeeds Zero offload](https://www.deepspeed.ai/tutorials/zero-offload/) claimed you could train a model up to **13B** on a single GPU.
   - They were looking for a better implementation.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1352490810834157652)** (2 messages): 

> `GPU Mode Scammer, Discord Channel Alerts` 


- **Scammer Alert Sounds the Alarm**: A user alerted the channel that a *scammer* is present in the **GPU Mode** channel, suggesting potential fraudulent activity.
   - The message included custom emojis related to **GPUs** and **Dragon Ball Z**, possibly as a humorous or attention-grabbing element.
- **Discord Channel Experiences Scammer Scare**: Members of the **GPU Mode** Discord channel were warned about the presence of a potential scammer.
   - The warning lacked specific details, but served as a general alert to exercise caution.


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1352411279092088943)** (3 messages): 

> `Hopper Architecture, Microbenchmarking, Matrix Multiplication` 


- **Hopper's Flops per Cycle**: On **Hopper**, one can microbenchmark **4096 flops/2048 MAD (16-bit) per cycle per SM**.
   - This doubles when using **8-bit types**, due to its two matmuls (QxK and xV).
- **Microbenchmarking recommended for Hopper**: One member suggested that *microbenchmarking would be the way to go* to gather the performance details of Hopper's architecture.
   - Another member thought it was probably documented somewhere, but couldn't find any reference.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1352391129017880656)** (3 messages): 

> `GTC Presentation, CUDA Kernels, Small Transformer Models, Hopper Architecture, CUTLASS 4.0` 


- ****GTC Presentation Promises Performance-Optimized CUDA Kernels****: A member announced their GTC presentation titled "Performance-Optimized CUDA Kernels for Inference With Small Transformer Models [S73168]" happening today at 4pm, focused on **Hopper architecture**.
   - They encouraged attendees to come, say hi, and ask tough questions, also mentioning that the [talk recordings](https://www.nvidia.com/gtc/) will be available on the GTC website for those unable to attend in person.
- ****CUTLASS 4.0 set to redefine Pythonic Integration at GTC****: Attendees will hear about the pythonic future of **CUTLASS** in its next major 4.0 version at GTC.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1352542892949897257)** (4 messages): 

> `Deprecated Coach class, Curriculum Experiments` 


- **Coach Class Faces Retirement**: Members discussed the removal of the **Coach class**, deeming it deprecated in favor of **Curriculum Experiments**.
   - One member agreed to open a PR, confirming it's *a leftover of early attempts*.
- **Curriculum Execution FTW**: Discussion focused on streamlining curriculum execution within the codebase.
   - The move aims to consolidate efforts and avoid confusion between older and newer approaches to curriculum management.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1352372054267006999)** (11 messages🔥): 

> `Leaderboard Submissions, GPU Tests` 


- **Grayscale Leaderboard Receives Flood of Submissions**: Multiple leaderboard submissions to the `grayscale` leaderboard were successful on GPUs: **L4**, **T4**, **A100**, and **H100** using Modal runners.
   - Submission IDs included `2351`, `2429`, `2430`, `2431`, `2459`, and `2460`.
- **Vectoradd Benchmark Achieves Success**: Benchmark submission with id `2363` to leaderboard `vectoradd` on GPUs: **T4**, **L4**, **A100**, **H100** using Modal runners succeeded!
   - This indicates progress in the `vectoradd` benchmark across various GPU architectures.
- **Grayscale Tests Pass on A100 GPUs**: Test submissions with IDs `2422` and `2423` to the `grayscale` leaderboard on **A100** GPUs using Modal runners were successful.
   - These tests specifically targeted the **A100** GPU architecture, indicating focused testing efforts.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1352455340200820756)** (2 messages): 

> `Consumer GPUs, Cloud GPUs, Local vs Cloud` 


- **Consumer GPUs obsolete rapidly**: Members discussed how consumer GPUs become obsolete quickly, especially when used for purposes other than gaming.
   - One member pointed out that for the same cost (~$1000), users can access the latest cloud GPUs on a rolling basis without any long-term commitment, while another mentioned that *if you don’t have a gpu at home then you never get to hear it go brrrr*.
- **Local GPU enthusiasts enjoy the "brrr" factor**: Enthusiasts appreciate the auditory feedback of having a GPU running locally.
   - One user stated *if you don’t have a gpu at home then you never get to hear it go brrrr*.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1352502574745518182)** (35 messages🔥): 

> `Oblix, AI Orchestration, Local LLM for SFW Stories, LLM Leaderboards, PC Build for Medical Data` 


- ****Oblix** Seamlessly Switches Between Local vs Cloud**: A member shared a demo video ([https://youtu.be/j0dOVWWzBrE](https://youtu.be/j0dOVWWzBrE)) of **Oblix**, which *seamlessly switches between local vs cloud* while still maintaining context using agents to monitor system resources and make decisions.
   - The platform orchestrates between **Ollama** and **OpenAI**, dynamically deciding whether to process each AI request locally or in the cloud for optimal performance and cost-efficiency, as detailed on [Oblix.ai](https://oblix.ai/).
- **LLM Leaderboards Compared for Model Selection**: Members discussed finding reliable LLM leaderboards for specific purposes, with one member sharing links to [Artificial Analysis](https://artificialanalysis.ai/leaderboards/models) and [LM Arena](https://lmarena.ai/?leaderboard).
   - Concerns were raised about filtering relevant models from these lists, particularly avoiding outdated or undesirable options like **Grok-3**.
- **Members seek advice on PC build for medical data processing**: A member requested assistance with building a new PC to process medical data using AI, emphasizing the need for secure, offline operation, mentioning the Github wasn't clear enough.
   - Another member suggested starting with an **Intel i9**, **128GB RAM**, and an **Nvidia 4090 RTX**.
- **GPT4All not ideal for audio file transcription**: A member inquired about using **GPT4All** for local audio file transcription, specifically uploading **.wav** files.
   - Another member clarified that **GPT4All** is primarily designed for **docs/pdf**, recommending **XTTS webui** for wav to text conversion, although noting it's not a simple install.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found</li><li><a href="https://artificialanalysis.ai/leaderboards/models">LLM Leaderboard - Compare GPT-4o, Llama 3, Mistral, Gemini &amp; other models | Artificial Analysis</a>: Comparison and ranking the performance of over 30 AI models (LLMs) across key metrics including quality, price, performance and speed (output speed - tokens per second &amp; latency - TTFT), context w...</li><li><a href="https://oblix.ai/">Transform Your AI Performance with Intelligent Hybrid Orchestration | Oblix.ai</a>: Experience our interactive demo and see how our intelligent agents seamlessly switch between local LLM execution and cloud providers for optimal performance and cost efficiency.
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1352378909491986473)** (10 messages🔥): 

> `W-GAN saturation, Transformers soft slots, MCP UX/UI` 


- **W-GANs mitigate gradient explosion**: Traditional GANs saturate due to BCE, while **W-GANs** mitigate this by being linear, as illustrated in [Figure 2 of the W-GAN paper](https://cdn.discordapp.com/attachments/986699377257119794/1352378909194322001/image.png?ex=67de7541&is=67dd23c1&hm=10de0ff85871d920b5ff7db506224a588a2c6c44030082a2ba7a9aad232ef204).
   - Although vanishing gradients are less of a problem, instability can still occur if the generator or discriminator becomes too dominant, leading to saturation at both ends.
- **Soft Slots Dynamically Bind in Transformers**: A member shared an image analysis on soft slot methods which shows how soft slots dynamically bind to input tokens or retrieved content in Transformers.
   - The equations for **Attention** and **Soft Slots (S')** are provided, highlighting the use of softmax and scaled dot-product attention mechanisms with learnable slots.
- **OpenAI.fm's UX/UI Rushed and Simple**: Members commented on the simple UX/UI of [OpenAI.fm](https://www.openai.fm/), joking that it looked rushed.
   - One member quoted a view that the MCP enforces too much structure, which makes it vulnerable to disruption by less structured protocols that can evolve according to user needs, emphasizing that *higher variance almost always wins* in a server-client system because *clients consume more of what they like and less of what they don't*.



**Link mentioned**: <a href="https://www.openai.fm/">OpenAI.fm</a>: An interactive demo for developers to try the new text-to-speech model in the OpenAI API

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1352407070485516379)** (3 messages): 

> `G-Retriever, Graph Question Answering, Graph RAG` 


- **G-Retriever enables chatting with graphs**: The [G-Retriever paper](https://arxiv.org/abs/2402.07630) details the semantic extraction of information from a knowledge graph, enabling *chatting with your graph*, *graph QnA* and *Graph RAG*.
- **Graph Question Answering benchmark introduced**: The paper introduces a **Graph Question Answering (GraphQA) benchmark** with data collected from different applications including scene graph understanding, common sense reasoning, and knowledge graph reasoning.



**Link mentioned**: <a href="https://arxiv.org/abs/2402.07630">G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering</a>: Given a graph with textual attributes, we enable users to `chat with their graph&#39;: that is, to ask questions about the graph using a conversational interface. In response to a user&#39;s questions...

  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1352356691559321640)** (16 messages🔥): 

> `Claude Pokemon, AI Moore's Law, Hunyuan-T1 model` 


- **Claude's Pokemon Prowess Questioned**: Members express skepticism about **Claude's** ability, noting it is *"quite garbage at Pokemon"*, questioning its capabilities despite general improvements.
- **Moore's Law for AI Agents**: Discussion ensues on [METR_Evals' research](https://x.com/METR_Evals/status/1902384481111322929) suggesting *"Moore’s Law for AI agents"*, where the length of tasks AIs can do is doubling about every **7 months**.
   - Some members dismiss the related chart as *"actual bullshit"*, arguing that certain tasks, like training a classifier or optimizing chip creation, shouldn't be interesting for probabilistic models.
- **Hunyuan-T1 Launches for Reasoning**: **Hunyuan-T1**, powered by Hunyuan TurboS, features a **Hybrid-Mamba-Transformer MoE architecture** and is designed for speed, accuracy, and efficiency, according to [Tencent](https://x.com/TXhunyuan/status/1903121005809373386).
   - The new model boasts **low hallucination** in summaries and excels in **long-text processing**, as featured in the [Hunyuan-T1 HuggingFace demo](https://huggingface.co/spaces/tencent/Hunyuan-T1).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TXhunyuan/status/1903121005809373386">Tweet from Hunyuan (@TXhunyuan)</a>: 🚀 Introducing Hunyuan-T1! 🌟Meet Hunyuan-T1, the latest breakthrough in AI reasoning! Powered by Hunyuan TurboS, it&#39;s built for speed, accuracy, and efficiency. 🔥✅ Hybrid-Mamba-Transformer MoE A...</li><li><a href="https://x.com/METR_Evals/status/1902384481111322929">Tweet from METR (@METR_Evals)</a>: When will AI systems be able to carry out long projects independently?In new research, we find a kind of “Moore’s Law for AI agents”: the length of tasks that AIs can do is doubling about every 7 mont...</li><li><a href="https://x.com/METR_Evals/status/1902384495673680188">Tweet from METR (@METR_Evals)</a>: We then fit a curve that predicts the success rate of an AI based on how long it took humans to do each task. This curve characterizes how capable an AI is at different task lengths. We then summarize...</li><li><a href="https://fxtwitter.com/bycloudai/status/1903149418422939838">Tweet from bycloud (@bycloudai)</a>: &gt; mamba-transformer hybrid reasoning model near on par with DeepSeek-R1whatQuoting Hunyuan (@TXhunyuan) 🚀 Introducing Hunyuan-T1! 🌟Meet Hunyuan-T1, the latest breakthrough in AI reasoning! Powere...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1352681010281713745)** (1 messages): 

> `Local RAG app, GitIngest parsing, Streamlit UI, Ollama Llama 3.2` 


- **Fully Local RAG App Aces Code Chat**: A fully local, fully open-source **RAG app** that can chat with your code has been built by a LlamaIndex community member, and announced in a [tweet](https://twitter.com/llama_index/status/1903121505984319771).
- **GitIngest Parses for Streamlit Display**: The app uses **GitIngest** to parse the code into summaries and markdown, using **Streamlit** for the UI, with details available at [this link](https://t.co/WIVkzX33EB).
- **Ollama Runs Llama 3.2 Locally**: It runs **Meta's Llama 3.2** locally using **Ollama**.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1352391380088914030)** (13 messages🔥): 

> `LlamaIndex TypeScript Agent Import Issue, Agent Workflow Parallel Execution Limits, Human-in-the-Loop Tool Limitations` 


- **TypeScript Agent Import Issue Resolved**: A member using **LlamaIndex TS** had an issue importing **agent**, but it was resolved by updating the **tsconfig** bundler configuration.
   - The user confirmed that modifying the **TS config** resolved the import error, and thanked the community for the suggestion.
- **Limiting Parallel Executions in Agent Workflows**: A member asked about limiting parallel executions in **Agent Workflows**, specifically for a tool with a **human-in-the-loop event**.
   - They noted that the agent was calling the tool multiple times in parallel, causing issues, and they wanted to ensure the tool was called only once at a time; the issue was replied on [GitHub](https://github.com/run-llama/llama_index/issues/18220#issuecomment-2742089859).
- **Seeking Solutions for Human-in-the-Loop Tool Constraints**: A member encountered issues due to parallel calls to a **human-in-the-loop tool** within an **agent workflow** and sought ways to limit executions.
   - The user is experiencing funky issues when an agent workflow tool is called many times in parallel, and is awaiting assistance on a related [GitHub issue](https://github.com/run-llama/llama_index/issues/18220#issuecomment-2742089859).



**Link mentioned**: <a href="https://github.com/run-llama/llama_index/issues/18220#issuecomment-2742089859">[Question]: Parallel Human in Loop with Agent Workflow Issues · Issue #18220 · run-llama/llama_index</a>: Question Validation I have searched both the documentation and discord for an answer. Question Searching and debugging a long time to find a solution. Thanks for any help! When an agent workflow is...

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1352466135433609256)** (4 messages): 

> `Trial Key Limits, Command-A Training Data` 


- **Trial Key Limit is per Account**: A member asked if the monthly limit of **1k requests for trial keys** is per key or per account, and another member clarified that it is **per account**.
   - They added that trying to bypass this by making multiple accounts will result in removal of all accounts.
- **Inquiry About Command-A's Training Data**: A member asked about the cut-off date for **Command-A’s training data**.


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1352544076934680648)** (4 messages): 

> `Cohere API Errors, Rate Limiting, Checking Rate Limits` 


- **Cohere API throws common errors**: Users discussed various **Cohere API error messages**, including *invalid request*, *rate limiting*, and *token limits*.
   - The errors covered a range of issues, such as **empty documents**, **short prompts**, exceeding **token limits**, and **incorrect model specifications**.
- **Rate Limiting Discussed**: Members noted that rate limiting errors can be identified by a **429 status code** in the response, as detailed in the [Cohere API documentation](https://docs.cohere.com/reference/errors#429---too-many-requests).
   - One user noted their code crashed due to not getting a response.
- **Checking API rate limits**: A user asked if there was a way to check their **rate limit** to see how much they have left.
   - No resolution was given for checking rate limits via an API.



**Link mentioned**: <a href="https://docs.cohere.com/reference/errors#429---too-many-requests">Errors (status codes and description) — Cohere</a>: Understand Cohere&#x27;s HTTP response codes and how to handle errors in various programming languages.

  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/1352396846101692578)** (3 messages): 

> `Bot Permissions` 


- **Bot Permissions Issue**: A user mentioned that the bot might have **permissions issues** on the channel.
- **User Greets the Bot**: A user greeted another user. No specific AI discussion or links were shared.


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1352392912314437673)** (2 messages): 

> `Introductions, Low-code tech, Community Engagement` 


- **Hospitality Expert Joins Cohere!**: Gaby, a professional in the **hospitality industry**, introduces herself as a low-code tech enthusiast, proficient with platforms like **Make** and **Adalo**.
   - She expresses her eagerness to learn from fellow community members and contribute her own experiences.
- **Low-Code Tech Takes Center Stage**: Gaby's introduction highlights the growing importance of **low-code tools** in various industries, showcasing their accessibility and potential.
   - Her expertise in **Make** and **Adalo** could provide valuable insights for others exploring similar technologies within the Cohere community.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1352519841269157899)** (12 messages🔥): 

> `Duration Module Proposal, Mojo and PyTorch Integration, Nanosecond Precision as Base Unit` 


- **Duration Module Proposal Reveals Weird Behavior**: A member encountered a peculiar issue while working on a **`duration` module proposal**, specifically with type casting involving `Ratio` and `Duration` structs, and shared [code snippet](https://discord.com/channels/1013120035012476074/1173814954275168317) exhibiting the unexpected behavior.
- **PyTorch and Mojo Speed Boost?**: A user inquired about the possibility of using **PyTorch** in **Mojo** and whether it could accelerate the training process with **MAX**.
   - There was no response in the provided messages, so this idea remains an open question in this channel.
- **Time Flies with Nanosecond Precision**: A member suggested using **nanosecond precision** as the base unit for time, noting that `UInt64` of Nanoseconds covers over **500 years**, which should be sufficient.
   - Another member pointed out that C++ guarantees a default time resolution that can represent *at least* **292 years**, also noting that **seconds is the base SI unit** for time.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1352423597200904193)** (1 messages): 

> `MIPRO v2, LLM-as-a-judge, Automatic Metrics, DSPy Optimization, Evaluation Metrics` 


- **MIPRO v2 Used with LLM-as-a-Judge**: A member mentioned using **MIPRO v2** with **LLM-as-a-judge** as their metric for evaluation, pointing to a [math reasoning tutorial](https://dspy.ai/tutorials/math/).
   - The math reasoning tutorial provides an example of using **MIPRO** as a metric.
- **LLM-as-a-Judge Documentation Shared**: Documentation on using **LLM-as-a-judge** was shared from [DSPy's learning resources](https://dspy.ai/learn/evaluation/metrics/#intermediate-using-ai-feedback-for-your-metric).
   - The documentation provides info on using **AI feedback** for metric evaluations.
- **Automatic Metrics Crucial for DSPy**: It was highlighted that **automatic metrics** are essential for evaluation and optimization in **DSPy**.
   - DSPy leverages metrics to track progress and enhance the effectiveness of programs.
- **Metrics Defined for Task Evaluation**: A metric is defined as a function that scores system outputs based on data examples.
   - Simple tasks may use basic metrics like *accuracy* or *exact match*, while complex tasks require metrics that check multiple output properties using **AI feedback**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dspy.ai/tutorials/math/">Reasoning - DSPy</a>: The framework for programming—rather than prompting—language models.</li><li><a href="https://dspy.ai/learn/evaluation/metrics/#intermediate-using-ai-feedback-for-your-metric">Metrics - DSPy</a>: The framework for programming—rather than prompting—language models.
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1352552465844928522)** (1 messages): 

> `Unet3d model, 2D Convolutions` 


- **Unet3d Model uses 2D Convolutions**: A member questioned whether the example **unet3d** model is truly 3D, suggesting it resembles **2.5D** due to its reliance on **2D convolutions** and **2D transposes** on 3D input.
   - The member emphasized the distinction from a genuine 3D Unet architecture.
- **2D vs 3D Unet Architectures**: The discussion highlighted the difference between using **2D convolutions** on 3D input (resulting in a 2.5D effect) and employing true **3D Unet architectures** with 3D operations.
   - The user sought clarification on the implementation's dimensionality.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/)** (1 messages): 

krammnic: I like this: https://arxiv.org/pdf/2502.07923
  

---


---


---


---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

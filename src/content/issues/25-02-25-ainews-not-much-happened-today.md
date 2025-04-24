---
id: fb8f064c-a654-4286-a905-a014e041c5ad
title: not much happened today
date: '2025-02-26T02:19:12.201709Z'
original_slug: ainews-not-much-happened-today-6477
description: >-
  **Claude 3.7 Sonnet** has been released with notable improvements in coding,
  reasoning, and a large context window of up to *200k tokens*, outperforming
  competitors like DeepSeek and Gemini 2.0 Pro in coding benchmarks.
  Additionally, **DeepSeek** announced the open-source release of **DeepEP**, a
  communication library for MoE model training and inference, while **Alibaba
  Qwen** teased an upcoming open-weight open-source release of **Qwen2.5-Max**
  and its reasoning-enhanced variant **QwQ-Max**.
tags:
  - anthropic
  - deepseek
  - alibaba
  - openai
  - google
  - replit
  - runwayml
  - allen
  - perplexity
  - langchain
  - claude-3.7-sonnet
  - claude-code
  - deepseek-r1
  - deepseek-v3
  - gemini-2.0-pro
  - gpt-4o
  - qwen2.5-max
  - qwq-max
  - qwq-32b
  - o3-mini
  - o1-2024-12-17
  - google-veo-2
  - openai-sora
  - kling-1.5-pro
  - alibaba-wan2.1
  - qwen2vl-7b
  - replit-agent-v2
  - coding
  - reasoning
  - large-context-windows
  - model-benchmarks
  - open-source-releases
  - moe-training
  - communication-libraries
  - video-generation
  - multimodal-models
  - reinforcement-learning
  - datasets
  - tool-integration
  - agentic-workflows
  - pricing-models
companies:
  - anthropic
  - deepseek
  - alibaba
  - openai
  - google
  - replit
  - runwayml
  - allen
  - perplexity
  - langchain
models:
  - claude-3.7-sonnet
  - claude-code
  - deepseek-r1
  - deepseek-v3
  - gemini-2.0-pro
  - gpt-4o
  - qwen2.5-max
  - qwq-max
  - qwq-32b
  - o3-mini
  - o1-2024-12-17
  - google-veo-2
  - openai-sora
  - kling-1.5-pro
  - alibaba-wan2.1
  - qwen2vl-7b
  - replit-agent-v2
topics:
  - coding
  - reasoning
  - large-context-windows
  - model-benchmarks
  - open-source-releases
  - moe-training
  - communication-libraries
  - video-generation
  - multimodal-models
  - reinforcement-learning
  - datasets
  - tool-integration
  - agentic-workflows
  - pricing-models
---


<!-- buttondown-editor-mode: plaintext -->**a quiet day.**

> AI News for 2/24/2025-2/25/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**220** channels, and **5949** messages) for you. Estimated reading time saved (at 200wpm): **503 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

You should follow DeepSeek's [#OpenSourceWeek](https://x.com/search?q=%23OpenSourceWeek%20from%3Adeepseek_ai&src=typed_query&f=top), but the releases so far have not met our bar for headline story status.


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Claude 3.7 Sonnet Release and Performance**

- **Claude 3.7 Sonnet excels in coding and reasoning**: [@skirano](https://twitter.com/skirano/status/1894171599508537620) highlighted that **Claude 3.7 Sonnet with Claude Code** can generate an entire **"glass like" design system** in one shot, including all components. [@omarsar0](https://twitter.com/omarsar0/status/1894164720862523651) demonstrated **Claude 3.7's reasoning and coding capabilities** by creating a **simulator for attention mechanisms**.  [@reach_vb](https://twitter.com/reach_vb/status/1894132284711649463) noted that **Claude 3.7** beats **DeepSeek R1** and is on par with **O3-mini (high)** in non-thinking mode, anticipating strong performance in thinking mode. [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1894437867914682764) benchmarked **Claude 3.7 Sonnet** as the **best non-reasoning model for coding**, outperforming **DeepSeek v3, Gemini 2.0 Pro, and GPT-4o** on their coding evals **SciCode and LiveCodeBench**. [@terryyuezhuo](https://twitter.com/terryyuezhuo/status/1894138361654526171) shared **BigCodeBench-Hard results** showing **Claude-3.7 (w/o thinking)** achieving **33.8% Complete**, comparable to **Qwen2.5-Coder-32B-Instruct**, and outperforming **o3-mini** and **o1-2024-12-17**.
- **Claude 3.7 Sonnet available on multiple platforms**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1894186614827504054) announced **Claude 3.7 Sonnet's availability on Perplexity Pro**, noting improvements in **agentic workflows and code generation**. [@_akhaliq](https://twitter.com/_akhaliq/status/1894148292440666616) confirmed **Claude 3.7 Sonnet is live on Anychat with coder mode**. [@_philschmid](https://twitter.com/_philschmid/status/1894301548101980532) mentioned availability on **Anthropic, Amazon Bedrock, and Google Cloud**, at the same price of **$3/$15 per million input/output tokens**.
- **Claude 3.7 Sonnet's "Thinking Mode" and Context Window**: [@_philschmid](https://twitter.com/_philschmid/status/1894301548101980532) highlighted **Claude 3.7's** `<thinking>` mode with up to **64k tokens** and **reasoning tokens display**, along with a **200k context window** and **128k output token length**. [@Teknium1](https://twitter.com/Teknium1/status/1894319586428031354) praised the **toggleable think mode** in **Claude**.
- **Claude 3.7 Sonnet's coding tool "Claude Code"**:  [@_philschmid](https://twitter.com/_philschmid/status/1894301548101980532) introduced **Claude Code**, a **CLI-based coding assistant** capable of reading, modifying files, and executing commands. [@catherineols](https://twitter.com/catherineols/status/1894149904282661366) described **Claude Code** as more autonomous than other tools, capable of deciding to run tests and edit files. [@goodside](https://twitter.com/goodside/status/1894235937074282793) previewed **Claude Code**, noting it sees files, writes diffs, runs commands, and is like a lightweight Cursor without the editor.
- **Claude 3.7 Sonnet price comparison**: [@_philschmid](https://twitter.com/_philschmid/status/1894154634173845876) pointed out that **Claude 3.7's** price remained at **$3/$15 per million input/output**, making it **30x more expensive than Gemini 2.0 Flash** and **~3x more than Open o3-mini**.

**DeepSeek and Qwen Model Updates and Open Source Releases**

- **DeepSeek releases DeepEP communication library**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1894211757604049133) announced **DeepEP**, an **open-source EP communication library for MoE model training and inference**, featuring **efficient all-to-all communication, NVLink and RDMA support, FP8 support**, and optimized kernels. [@reach_vb](https://twitter.com/reach_vb/status/1894262653440184603) detailed **DeepEP's features**, including **asymmetric-domain bandwidth forwarding, low-latency kernels with pure RDMA, and PTX optimizations for Hopper GPUs**. [@danielhanchen](https://twitter.com/danielhanchen/status/1894212351932731581) highlighted **DeepSeek's #2 OSS release** with **MoE kernels, expert parallelism, and FP8 for training and inference**.
- **Qwen2.5-Max "Thinking (QwQ)" mode and upcoming open source release**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1894130603513319842) released **"Thinking (QwQ)" in Qwen Chat**, backed by **QwQ-Max-Preview**, a reasoning model based on **Qwen2.5-Max**, noting enhanced capabilities in **math, coding, and agent tasks**. [@huybery](https://twitter.com/huybery/status/1894131290246631523) teased the **future of Qwen**, mentioning the upcoming official release of **QwQ-Max** and the planned **open-weight release of both QwQ-Max and Qwen2.5-Max under Apache 2.0 license**, along with smaller variants like **QwQ-32B** and mobile apps. [@reach_vb](https://twitter.com/reach_vb/status/1894133551173701972) excitedly announced **QwQ & Qwen 2.5 Max open source release soon**.

**Video and Multimodal Model Developments**

- **Google Veo 2 video model surpasses Sora in benchmarks**: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1894450344580846043) reported **Google Veo 2** surpassed **OpenAI’s Sora and Kling 1.5 Pro** in their **Video Arena**, noting strengths in **rendering people and realistic physics**. Veo 2 can generate **minutes of 4K video** but is currently limited to **720p video with 8s duration** at a price of **$0.50 per second**.
- **Alibaba Wan2.1 open AI video generation model**: [@_akhaliq](https://twitter.com/_akhaliq/status/1894393244454166612) announced **Alibaba's Wan2.1**, an **open AI video generation model**, ranking **#1 on the VBench leaderboard**, outperforming **SOTA open-source & commercial models** in **complex motion dynamics, physics simulation, and text rendering**. [@multimodalart](https://twitter.com/multimodalart/status/1894390712457666869) confirmed **Wan2.1** is **Apache 2.0 open source** and available on **Hugging Face**.
- **RunwayML Creative Partners Program for artists**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1894382185710334146) described **RunwayML's Creative Partners Program**, giving artists free access to tools to reward experimentation and inspiration, contrasting it with companies copying the effort for product promotion without honoring artists.

**Tools, Libraries and Datasets**

- **Replit Agent v2 released**: [@pirroh](https://twitter.com/pirroh/status/1894434712623747294) announced **Replit Agent v2** in **Early Access**, highlighting a **new app creation experience, realtime app design preview**, and instructions for access. [@hwchase17](https://twitter.com/hwchase17/status/1894456642697400458) noted **Replit agent v2** is powered by **LangGraph and LangSmith**.
- **LangChain JS adds Claude 3.7 Support and LangGraph Supervisor**: [@LangChainAI](https://twitter.com/LangChainAI/status/1894432718576128394) shared tips for building agents with **Claude 3.7**, demonstrating **tool-calling agents with configurable reasoning**. [@LangChainAI](https://twitter.com/LangChainAI/status/1894426354357342431) introduced **LangGraph.js Supervisor**, a library for building **hierarchical multi-agent systems** with **LangGraph**. [@LangChainAI](https://twitter.com/LangChainAI/status/1894398108517241284) listed **17 new integration packages** added to **LangChain Python**. [@LangChainAI](https://twitter.com/LangChainAI/status/1894180315398377533) announced **Claude 3.7 support in LangChain JS**.
- **vLLM integrates EP support**: [@vllm_project](https://twitter.com/vllm_project/status/1894215122966507801) announced **initial EP support merged in vLLM**, with integration of collectives coming soon. [@reach_vb](https://twitter.com/reach_vb/status/1894266500271223021) confirmed **vLLM's lightning-fast integration of EP**.
- **OlmOCR by Allen AI for PDF parsing**: [@mervenoyann](https://twitter.com/mervenoyann/status/1894422823646409090) presented **OlmOCR**, a new tool by **@allen_ai** for **parsing PDFs**, based on **Qwen2VL-7B**, and available on **transformers with Apache 2.0 license**.
- **Big-Math dataset for RL in LLMs**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1894232624203534657) and [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1894348004272083385) shared **SynthLabs' Big-Math**, a **large-scale, high-quality math dataset** for reinforcement learning in language models, containing over **250,000 questions with verifiable answers**.

**Research and Analysis**

- **Perplexity Deep Research for paid users**: [@OpenAI](https://twitter.com/OpenAI/status/1894454194943529433) announced **Deep research** rolling out to all **ChatGPT Plus, Team, Edu, and Enterprise users**, with improvements including **embedded images with citations** and better understanding of uploaded files. [@OpenAI](https://twitter.com/OpenAI/status/1894454196986155130) detailed usage limits for **Plus, Team, Enterprise, Edu, and Pro users**. [@OpenAI](https://twitter.com/OpenAI/status/1894454197967528224) shared the **system card for Deep research**. [@OpenAI](https://twitter.com/OpenAI/status/1894454199175581973) mentioned community expert involvement in training **Deep research** and opened interest registration for future model contributions. [@kevinweil](https://twitter.com/kevinweil/status/1894468278078357857) announced **Deep research rolling out to all paid users**, highlighting its capability for week-long research tasks in 15 minutes. [@AravSrinivas](https://twitter.com/AravSrinivas/status/1894471526449385687) announced **Deep Research API availability** for developers.
- **Minions: Cost-efficient collaboration between local and cloud models**: [@togethercompute](https://twitter.com/togethercompute/status/1894392054043578373) introduced **Minions**, a method pairing **small language models on a laptop with frontier cloud models**, preserving **98% of accuracy for &lt;18% of the cost**. [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1894354075757777311) highlighted **Minions achieving 5.7x cost reduction while maintaining 97.9% cloud model performance**.
- **Learning to Reason from Feedback at Test-Time (FTTT)**: [@dair_ai](https://twitter.com/dair_ai/status/1894419591780340065) presented research on **Feedback-based Test-Time Training (FTTT)**, enabling LLMs to learn iteratively from environment feedback during inference, using **self-reflected feedback and OPTUNE, a learnable test-time optimizer**.

**AI Industry and Market Trends**

- **Focus on AI agents and agency**: [@polynoamial](https://twitter.com/polynoamial/status/1894468586598797661) questioned if AI models will soon have agency. [@swyx](https://twitter.com/swyx/status/1894159894976008585) emphasized **Agency > Intelligence**, defining agency as "getting what you want done" and "doing the right things". [@omarsar0](https://twitter.com/omarsar0/status/1894485767428202977) expressed being impressed by **Windsurf agentic capabilities**.
- **Open source AI momentum**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1894435364028260739) urged for more **public, open, collaborative AI**. [@reach_vb](https://twitter.com/reach_vb/status/1894133865742004391) thanked **Alibaba_Qwen** for their commitment to **Open Source and Science**. [@NandoDF](https://twitter.com/NandoDF/status/1894337775832334564) highlighted **European AI entrepreneurship and competition**, suggesting eliminating notice periods and non-competes to boost the European AI industry.
- **AI in specific domains**: [@RichardSocher](https://twitter.com/RichardSocher/status/1894447036923351104) anticipated epic progress when hill climbing starts on meaningful **bio benchmarks**. [@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1894419218231128424) is hiring postdocs to develop an **Artificial Scientist for novel chemical materials** for climate change. [@METR_Evals](https://twitter.com/METR_Evals/status/1894257205680967907) is running a pilot experiment to measure **AI tools' impact on open source developer productivity**.
- **AI safety and alignment concerns**: [@sleepinyourhat](https://twitter.com/sleepinyourhat/status/1894446138625052838) shared a surprising and disconcerting **LLM alignment result**. [@NeelNanda5](https://twitter.com/NeelNanda5/status/1894192519719907467) announced a **Google DeepMind team** using **model internals in production to enhance Gemini safety**. [@sarahcat21](https://twitter.com/sarahcat21/status/1894413202706022570) discussed the need for **high quality annotations** for improving model capabilities and alignment, noting degrading annotation quality.
- **AI and the future of work**: [@adcock_brett](https://twitter.com/adcock_brett/status/1894462678757986393) predicted a future with more humanoids than humans doing various services and collapsing the price of goods/services. [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1894477390065406001) discussed the concentrated nature of tech development driven by AI. [@francoisfleuret](https://twitter.com/francoisfleuret/status/1894478503522808129) asked for stories from people whose professional lives have been changed by AI models.

**Memes and Humor**

- **Death Star Startup Pitch**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1894416827763359766) joked about a startup with a **"bold vision: the Death Star"** seeking a **$500k seed round**.
- **Worker 17 and AI overlords**: [@nearcyan](https://twitter.com/nearcyan/status/1894213291142181202) shared a meme about **"Worker 17"** and an **"AllKnowingLineSupervisingAutonomousSuperIntelligence"**, depicting a harsh work environment. [@nearcyan](https://twitter.com/nearcyan/status/1894437139263492454) continued the **"Worker 17"** theme, and [@rishdotblog](https://twitter.com/rishdotblog/status/1894376205765546229) joked about future robot overlords hating humans.
- **Claude playing Pokemon on Twitch**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1894419011569344978) announced **"Can Claude play Pokémon?"** and [@kipperrii](https://twitter.com/kipperrii/status/1894438649913323867) invited people to watch **Claude play Pokemon on Twitch**. [@_philschmid](https://twitter.com/_philschmid/status/1894306565370335568) joked about waiting for the first **"AI plays Pokemon" stream**. [@nearcyan](https://twitter.com/nearcyan/status/1894423215088488808) urged people to watch **Claude playing Pokemon on Twitch**. [@AmandaAskell](https://twitter.com/AmandaAskell/status/1894432355622031661) stated **"Watching Claude play Pokemon is a delight."**.
- **Anthropic branding and aversion to number four**: [@scaling01](https://twitter.com/scaling01/status/1894362813377749021) joked about **Anthropic being "more Elven than Human"**. [@dylan522p](https://twitter.com/dylan522p/status/1894154230229078391) humorously suggested **Anthropic is a Chinese AI company due to their aversion to the number four**.
- **Other humorous tweets**: [@giffmana](https://twitter.com/giffmana/status/1894310343658151961) shared a funny prompt and response from **Grok**. [@nearcyan](https://twitter.com/nearcyan/status/1894242035139326244) made a joke that was missed by others. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1894234559530627240) shared a funny image related to **Nvidia**. [@abacaj](https://twitter.com/abacaj/status/1894169598720688239) joked about loyalty to models. [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1894220349106991245) thanked **OpenAI** with a **DeepSeek** tweet.
---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. DeepSeek's DeepEP: Enhanced MoE GPU Communication**

- **DeepSeek Realse 2nd Bomb, DeepEP a communication library tailored for MoE model** ([Score: 407, Comments: 48](https://reddit.com/r/LocalLLaMA/comments/1ixkg22/deepseek_realse_2nd_bomb_deepep_a_communication/)): **DeepSeek** has released **DeepEP**, a communication library specifically designed for **Mixture-of-Experts (MoE)** models and **expert parallelism (EP)**. **DeepEP** features high-throughput, low-latency all-to-all **GPU kernels** and supports low-precision operations such as **FP8**, but is currently limited to GPUs with the **Hopper architecture** like **H100, H200,** and **H800**. [GitHub Repository](https://github.com/deepseek-ai/DeepEP).
  - **DeepEP Performance Optimization**: A notable discovery in the **DeepEP** repository involves using an undocumented **PTX instruction** `ld.global.nc.L1::no_allocate.L2::256B` for extreme performance on **Hopper architectures**. This instruction accesses volatile GPU memory with non-coherent modifiers `.nc`, but is tested to be correct and enhances performance significantly.
  - **Potential for Practical Applications**: Users express hope that **DeepEP**'s improvements could make **Local R1** more practical by enabling faster inference on **Mixture-of-Experts** models, addressing previous performance issues with **DeepSeek**.
  - **Hardware Limitations and Aspirations**: While **DeepEP** currently supports only **Hopper architecture** GPUs, there is interest in porting it to other GPUs like the **3090s**, reflecting a desire for broader hardware compatibility.


- **[DeepSeek 2nd OSS package - DeepEP - Expert parallel FP8 MOE kernels](https://x.com/deepseek_ai/status/1894211757604049133)** ([Score: 153, Comments: 11](https://reddit.com/r/LocalLLaMA/comments/1ixkfcb/deepseek_2nd_oss_package_deepep_expert_parallel/)): **DeepSeek** released its second open-source software package, **DeepEP**, which features expert parallel FP8 Mixture of Experts (MOE) kernels.
  - **DeepEP** includes **inference style kernels** for **Mixture of Experts (MoE) layers** with **FP8 support** and expert parallelism, enabling the overlap of GPU/CPU communication and GPU computation. It is also suitable for training large MoE models.


**Theme 2. Sonnet 3.7 Dominates Benchmark Testing**

- **[New LiveBench results just released. Sonnet 3.7 reasoning now tops the charts and Sonnet 3.7 is also top non-reasoning model](https://i.redd.it/ys8y5ndtu6le1.png)** ([Score: 257, Comments: 53](https://reddit.com/r/LocalLLaMA/comments/1ixj4bp/new_livebench_results_just_released_sonnet_37/)): **Sonnet 3.7** from **Anthropic** leads the latest **LiveBench** results, achieving top scores in both **Global Average** (76.10) and **Reasoning Average** (87.83). The table showcases performance metrics of models from organizations like **OpenAI** and **Google** across categories including **Coding**, **Mathematics**, **Data Analysis**, and **Language**.
  - **Anthropic's Sonnet 3.7** leads in performance, but there are calls for releasing the model weights for local use. **LiveBench** results highlight improvements in coding and reasoning, with users noting the model's efficiency and quality compared to others like **O3 mini high** and **Gemini 2 Flash**.
  - Discussions focus on **benchmark limitations** and real-world performance, with some users expressing skepticism about the model's math scores due to inconsistencies with official benchmarks. There is interest in seeing if using **128k tokens** for evaluation could improve results, despite concerns about latency.
  - The community is keen on more efficient model usage and hardware improvements, as some feel that the raw strength of models is reaching a plateau. The **Aider leaderboard** shows Sonnet 3.7 as significantly ahead of 3.5, indicating positive reception for its performance in coding tasks.


- **[Sonnet 3.7 near clean sweep of EQ-Bench benchmarks](https://www.reddit.com/gallery/1ixupja)** ([Score: 106, Comments: 54](https://reddit.com/r/LocalLLaMA/comments/1ixupja/sonnet_37_near_clean_sweep_of_eqbench_benchmarks/)): **Sonnet 3.7** achieves a near clean sweep of the **EQ-Bench** benchmarks, indicating significant advancements in AI model performance. This highlights the model's effectiveness and capability in various benchmark tests.
  - Discussions around **Sonnet 3.7's writing style** highlight its "safe" approach, with comparisons to other models like **Deepseek-R1** and **OpenAI**. Users question the descriptions like "earthy" and "spiky," while some find the model's style appealing to "liberal arts" audiences. **Sonnet 3.7** shows significant improvements in humor understanding, as noted in the [Buzzbench results](https://eqbench.com/results/buzzbench/claude-3.7-sonnet-20250219_outputs.txt).
  - The **cost-effectiveness** of AI models is debated, with **Sonnet 3.7** being more expensive than alternatives like **Gemini**. The discussion centers on whether the performance justifies the cost, especially for different user demographics, such as high-earning professionals versus hobbyists or students.
  - **Darkest Muse**, a smaller 9b model, is praised for its creative writing capabilities, including character dialogue and poetic style, despite limitations in instruction following. The model's fine-tuning process involved training on human authors from the Gutenberg library, pushing it to the edge of model collapse for unique results.


**Theme 3. Alibaba's Wan 2.1 Video Model Open-Source Release Scheduled**

- **[Alibaba video model Wan 2.1 will be released Feb 25th,2025 and is open source!](https://i.redd.it/amle9h0op8le1.jpeg)** ([Score: 408, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1ixporw/alibaba_video_model_wan_21_will_be_released_feb/)): **Alibaba** announced the open-source release of its video model **Wan 2.1**, scheduled for **February 25th, 2025**. The event, featuring a futuristic design with the theme "BEYOND VISION," will be broadcast live at **11:00 PM (UTC+8)**, highlighting the model's innovative potential.
  - **Naming Conventions**: The name **Wan** is derived from the Chinese pronunciation for 10,000, similar to **Qwen**, which represents 1,000. This reflects a pattern in **Alibaba's** naming strategy for their models.
  - **Model Availability and Performance**: Users are eager for the release of **Wan 2.1**, with discussions on its availability on **Hugging Face** and concerns about server overload affecting generation capabilities. A smaller model is also available, as noted in the [README on Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/README.md).
  - **Hardware Requirements and Comparisons**: There is optimism that **Wan 2.1** will be runnable on consumer-grade GPUs like the **RTX 3060**, with comparisons to **Flux**, which has reduced its training requirements from 24 GB to 6 GB. Users hope **Wan 2.1** will surpass **SORA** in terms of capabilities and open-source accessibility.


- **WAN Video model launched** ([Score: 100, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1ixtug3/wan_video_model_launched/)): **WAN Video model** has been launched with weights available on [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B). Although not a **Large Language Model (LLM)**, it may interest many in the AI community.
  - **Quantization** is applicable to **Video Language Models (VLMs)**, with existing GGUFs like **Hunyuan** and **LTX**. These are popular due to the difficulty of fitting large models, and similar GGUFs are anticipated for **WAN** soon.
  - There is a **1.3B version** of the WAN model that requires only **8.19 GB VRAM**, but it is restricted to **480p resolution** due to limited training data at higher resolutions. However, users can upscale the output to achieve better results.
  - The **WAN Video model** at **14B** is considered large for open models, comparable to the **Hunyuan** model at **13B**, with **LTX** being a smaller option at **2B**. The WAN model's release in both **1.3B and 14B variants** aims to cater to different use cases and hardware capabilities.


**Theme 4. Gemma 3 27b Release: A New Contender in AI Models**

- **[Gemma 3 27b just dropped (Gemini API models list)](https://i.redd.it/y2nlshypwble1.png)** ([Score: 102, Comments: 27](https://reddit.com/r/LocalLLaMA/comments/1iy22ux/gemma_3_27b_just_dropped_gemini_api_models_list/)): **Gemma 3 27b** has been added to the **Gemini API models list**, featuring a user-friendly interface with a search bar and clickable model entries such as **"Gemini 1.5 Pro"** and **"Gemini 2.0 Flash"**. The active model, **"models/gemma-3-27b-it"**, is highlighted, suggesting it is currently selected, underscoring a structured and professional layout for ease of navigation.
  - **Model Lineage and Performance**: There is a discussion about the lineage and performance of **Gemma** models, with users noting that **Gemma 2** was superior for short story writing compared to **Gemini**, particularly the 9b version. **Gemma** and **Gemini** have similar response styles, but **Flash** is a different model.
  - **Access and Integration**: Users question how **Open WebUI** accesses Google's unreleased models, with clarifications that it doesn't natively access models. Instead, users can add models via external APIs like **Vertex AI** or **LiteLLM**, and there is interest in finding the correct API URL as the current one doesn't list **Gemma**.
  - **Model Size Perception**: There's a humorous exchange about the perception of model sizes, with **70B** now considered medium and **24B** considered small, reflecting the rapid advancements in AI model scaling.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. WAN 2.1 Released and Open Source with New Features**

- **WAN Released** ([Score: 382, Comments: 169](https://reddit.com/r/StableDiffusion/comments/1ixtvdz/wan_released/)): **WAN Released**: The **WAN** video model has been released with open-source weights available for download. Multiple models are live on [Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B), enabling broader access and experimentation.
  - Several users discussed the **VRAM requirements** for different model versions, noting that the **1.3B parameter model** requires **8GB VRAM** and the **14B model** could potentially run on **10GB VRAM**. There is also interest in using **bf16 precision** to reduce VRAM usage.
  - Users are exploring **Gradio applications** and installation processes, with **CeFurkan** working on a **Gradio app** and installer compatible with **Windows** and **Python 3.10 VENV**. There are challenges with **RTX 5000 series** not having proper **PyTorch** support.
  - The community is curious about the model's capabilities in handling **multiple tasks** like Text-to-Video, Image-to-Video, and Video-to-Audio, with some expressing skepticism about audio generation. **Multiple safetensors** are discussed, with guidance on handling them using the **diffusers library**.


- **[Alibaba video model Wan 2.1 will be released today and is open source!](https://i.redd.it/kug52fk0r8le1.jpeg)** ([Score: 415, Comments: 104](https://reddit.com/r/StableDiffusion/comments/1ixpsgp/alibaba_video_model_wan_21_will_be_released_today/)): **Alibaba** has announced the open-source release of its **Wan 2.1 video model**. The release event will be live-streamed on **February 25, 2025, at 11:00 PM (UTC+8)**, with the event branded under **TONGYI MOMENT** and featuring a futuristic, sleek visual design.
  - Discussions highlight the **technical requirements** for running the **Wan 2.1 video model**, with users speculating it might need **80GB VRAM** but hoping it can run on **16GB VRAM** with techniques like offloading and fp8, similar to **hunyuan**. Some users express a desire for a model that can scale from high to lower specs, akin to **Deepseek R1**.
  - The **release event** will be live-streamed, likely on **Alibaba's official X account**. Users are curious about the model's capabilities, particularly its ability to perform **image-to-video** transformations, which has been confirmed by commenters.
  - There is humorous commentary on the model's name **Wanx**, with users noting its phonetic resemblance to "wank" and speculating on the implications, including potential branding for **uncensored/NSFW models**.


- **[My very first Wan 2.1 Generation on RTX 3090 Ti](https://v.redd.it/0sokbb6s1ble1)** ([Score: 524, Comments: 181](https://reddit.com/r/StableDiffusion/comments/1ixxul1/my_very_first_wan_21_generation_on_rtx_3090_ti/)): The post provides a **first look at Wan 2.1 Generation** using an **RTX 3090 Ti**. Since the post body is empty and the content is primarily in a video, no further details can be summarized.
  - **VRAM Requirements and Optimization**: **CeFurkan** and others discussed optimizing the **1.3B and 14B models** to run on **6GB and 10GB GPUs**, respectively, with the **RTX 3090 Ti** using up to **18GB VRAM** for generation. The community expressed interest in running these models on lower VRAM setups, such as **3060 12GB**, and **CeFurkan** is developing an **AIO installer** to simplify usage.
  - **Model Capabilities and Performance**: The **Wan 2.1 Generation** supports **text to video, image to video, and video to video** generation, with **16 FPS** for five-second clips. **CeFurkan** is working on a **Gradio app** for easier use, and users are impressed by the quality, comparing it favorably to **Hunyuan Video**.
  - **Community Contributions and Resources**: **Kijai's ComfyUI integration** is in development, with resources like [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and **Kijai/WanVideo_comfy** available for users. The community is actively sharing examples and prompts, with some users asking about potential **NSFW capabilities** and the ease of use compared to **ComfyUI**.


**Theme 2. Claude 3.7 Model: Enhanced Capabilities and Accessibility**

- **Holy. Shit. 3.7 is literally magic.** ([Score: 565, Comments: 111](https://reddit.com/r/ClaudeAI/comments/1ixnkdz/holy_shit_37_is_literally_magic/)): **Claude 3.7** has significantly improved in **extended thinking, model quality, and output**, making it 10 times more useful than its predecessor, **Claude 3.5**. The author used Claude 3.7 to design an interactive **SaaS-style demo app**, including an advanced ROI calculator and onboarding process, all within a single chat, highlighting its potential for real-world applications.
  - **Claude 3.7 Improvements**: Users highlight significant improvements in **Claude 3.7** over **3.5**, particularly in following complex instructions and reducing cognitive load, with enhanced troubleshooting protocols and smoother operation. The model's ability to automatically check entire chains before making changes is seen as a major advancement.
  - **Usage and Cost Considerations**: Discussions around **inference costs** and **token management** suggest that **Claude** may face bottlenecks due to hardware limitations, impacting its market strategy. Some users report strange errors and suboptimal suggestions, possibly due to token conservation strategies in Copilot, while others find **Cline** extension a superior alternative for coding tasks.
  - **SaaS and Development Efficiency**: The creation of complex SaaS applications is now faster and more efficient with **Claude 3.7**, allowing users to complete months of development work in days. However, there are concerns about potential nerfing due to tighter censorship filters, which could degrade model performance over time.


- **[Claude 3.7 is $1 a month for college students](https://i.redd.it/8we3eryryble1.jpeg)** ([Score: 187, Comments: 42](https://reddit.com/r/ClaudeAI/comments/1iy2cyz/claude_37_is_1_a_month_for_college_students/)): **Claude 3.7** is now available to college students at a promotional rate of **$1/month** (down from the regular price of **$20/month**), as announced in an email to the **Cornell community**. The offer requires students to sign up with their **.edu email** and highlights features such as "Write code," "Extract insights," and "Brainstorm."
  - Commenters express skepticism about the authenticity of the **Claude 3.7** offer, with multiple users suggesting it might be a phishing scam due to the lack of official announcements or information on **Google** and **Claude's official website**.
  - Some users joke about enrolling at **Cornell** to take advantage of the offer, while others speculate that **Anthropic** might be using this as a strategy to collect data from students at prestigious universities.
  - There is a call for verification of the email's legitimacy, with suggestions to check the email source and concerns about the possibility of stolen or exploited accounts being resold.


- **["Claude 3.7, make a snake game, but the snake is self-aware it is in a game and trying to escape"](https://v.redd.it/nn87hj1epble1)** ([Score: 407, Comments: 32](https://reddit.com/r/ClaudeAI/comments/1iy138z/claude_37_make_a_snake_game_but_the_snake_is/)): **Claude 3.7** is tasked with creating a **snake game** where the snake is self-aware and attempts to escape the game. The post does not provide further details or context beyond this intriguing concept.
  - Users are impressed by **Claude 3.7's** ability to create complex outputs from simple prompts, with some comparing the experience to **AGI** and expressing disbelief at the results, such as the creation of a self-aware snake game and a fully functional website with multiple tools.
  - **Hereditydrift** highlights the complexity and creativity of Claude 3.7's output from a minimal prompt, specifically mentioning the unexpected inclusion of a "Matrix section," which astonishes many users.
  - **Admirable_Scallion25** and others note that **Claude 3.5** does not achieve the same level of complexity in one attempt, indicating a significant improvement in **Claude 3.7's** capabilities.


**Theme 3. Claude Sonnet 3.7 Reigns Supreme: New top model in LLM benchmark**

- **[Sonnet 3.7 Extended Reasoning w/ 64k thinking tokens is the #1 model](https://i.redd.it/nc1hdnpv27le1.png)** ([Score: 154, Comments: 20](https://reddit.com/r/ClaudeAI/comments/1ixk1gw/sonnet_37_extended_reasoning_w_64k_thinking/)): **Sonnet 3.7 Extended Reasoning** with **64k tokens** by **Anthropic** leads in performance, boasting the highest global average score of **76.10**, according to a table comparing AI models. It excels across various metrics including reasoning, coding, mathematics, data analysis, and language, outperforming models from **OpenAI**, **xAI**, and **Google**.
  - **Sonnet 3.7 Extended Reasoning** with **64k tokens** is praised for its performance, with **Bindu Reddy** highlighting its speed, reasoning, and coding abilities, labeling it the "best, most usable, and generally available model" ([link](https://x.com/bindureddy/status/1894196792700670149)). Users note its improvement over the **3.5 model** and its leading position in benchmarks like **LiveBench**.
  - Some users question the benchmark's real-world applicability, suggesting that cost normalization is essential for comparison, especially when considering test time compute scaling. They appreciate Sonnet's control over scaling costs, which optimizes workflows.
  - **Sonnet 3.7** is noted for outperforming **o3-mini-high** in various benchmarks including **SWE bench**, **webdev arena**, and **Aider benchmark**. In UI design and aesthetics, it significantly surpasses **o3-mini-high** and **o1 pro**, indicating specialized training in common UI elements.


- **[R] Analysis of 400+ ML competitions in 2024** ([Score: 227, Comments: 19](https://reddit.com/r/MachineLearning/comments/1ixrxoq/r_analysis_of_400_ml_competitions_in_2024/)): The analysis of over **400 ML competitions in 2024** highlights that **Kaggle remains the largest platform** by prize money and user base. **Python dominates as the primary language**, with **PyTorch preferred over TensorFlow** at a 9:1 ratio, and **NVIDIA GPUs**, particularly the A100, are predominantly used for training models. Additionally, **convolutional neural networks excel in computer vision**, while **gradient-boosted decision trees** are favored in tabular/time-series competitions. The full report is available [here](https://mlcontests.com/state-of-machine-learning-competitions-2024?ref=mlcr).
  - **Jax Popularity and Advantages**: Despite the dominance of **PyTorch**, some users express disappointment over the limited use of **Jax** in competitions, noting its simplicity and resemblance to **numpy** with additional features like **grad**, **vmap**, and **jit**. Jax is reportedly gaining traction in academia, although many professionals prefer sticking with PyTorch.
  - **Synthetic Data in ML Competitions**: There is a debate about the effectiveness of using synthetic data in competitions, with concerns about it potentially "blurring" the original dataset. However, thoughtful use, such as generating synthetic backgrounds and superimposing objects for training, has proven beneficial, as demonstrated in a spacecraft detection competition, enhancing model robustness and generalization.
  - **Generative Models and Data Augmentation**: Users discuss the implications of using generative models for data augmentation, emphasizing the importance of processing synthetic data carefully to add meaningful information. Successful strategies involve removing nonsensical examples and focusing on solutions that enhance training, as highlighted by a [winning competition team's documentation](https://github.com/drivendataorg/pose-bowl-spacecraft-challenge/blob/main/detection/1st%20Place/reports/DrivenData-Competition-Winner-Documentation.pdf).


**Theme 4. Advanced Voice Features and Deep Research in GPT-4o Updates**

- **[Grok is cooked](https://i.redd.it/qcolgg76gale1.jpeg)** ([Score: 172, Comments: 61](https://reddit.com/r/OpenAI/comments/1ixv5c4/grok_is_cooked/)): The post highlights concerns about **Grok's potential biases** following its deployment, as evidenced by its response identifying **"Donald Trump"** as the biggest disinformation spreader in a user query. This raises questions about the **AI's validity and neutrality**, particularly in politically sensitive contexts like **elections, immigration, and climate change**.
  - There is a significant debate over **Grok's bias**, with some users arguing that its responses are influenced by an overwhelming amount of media, while others suggest that it may be biased in favor of **Elon Musk**. **Wagagastiz** points to a lack of media defending Musk as a sign of bias, while **derfw** counters that Grok's responses might indicate neutrality.
  - Concerns about **conservative bias** and attempts to manipulate AI responses are prevalent, with users like **well-filibuster** speculating on efforts to retrain or create new chatbots to align with conservative views. **Excellent_Egg5882** highlights a pattern of conservatives downvoting reality when it conflicts with their biases.
  - Skepticism about the ability to maintain an unbiased **LLM** is evident, with users like **ai_and_sports_fan** and **Earth-Jupiter-Mars** expressing distrust in the long-term neutrality of Grok and other AI systems, given past instances of censorship and manipulation.


- **[Deep research is now out for all Plus Users!](https://i.redd.it/11pk08wgddle1.jpeg)** ([Score: 287, Comments: 63](https://reddit.com/r/OpenAI/comments/1iy96jx/deep_research_is_now_out_for_all_plus_users/)): **Sam Altman** announced via a tweet that **"deep research"** is now accessible to **ChatGPT Plus** users, calling it one of his favorite releases. The tweet garnered significant attention with **31.5K views**, **261 retweets**, **103 quote tweets**, and **1.1K likes**.
  - Users discussed the **monthly limit** for deep research, with confirmation that **Plus users** have a limit of **10 uses** per month, while **Pro users** receive **120 uses**. There was confusion about usage counts, but it was clarified that follow-up questions do not count against the limit.
  - Some users expressed **disappointment** with the feature, citing inaccuracies, such as incorrect **Nvidia stock prices**. Others shared successful use cases, like using AI to create a custom **Music LLM** with **MusicGen** and **Replicate.com**.
  - Several users faced **access issues**, with suggestions to log out and back in or switch to the desktop version to resolve it. The feature's availability varied, with some users still unable to access it despite being **Plus users**.


- **[We are rolling out a version of Advanced Voice powered by GPT-4o mini to give all ChatGPT free users a chance to preview it daily across platforms.](https://i.redd.it/efwa7dp4qcle1.jpeg)** ([Score: 115, Comments: 28](https://reddit.com/r/OpenAI/comments/1iy62v7/we_are_rolling_out_a_version_of_advanced_voice/)): **OpenAI** is rolling out a version of **Advanced Voice** powered by **GPT-4o mini** for all **ChatGPT** free users, allowing daily previews across platforms. The conversation pace and tone are similar to the **GPT-4o** version, but it is more cost-effective, as noted in a tweet that has received **3.3K views**.
  - **Source Link**: A source link to the announcement tweet by **OpenAI** can be found [here](https://x.com/openai/status/1894495906952876101?s=46&t=hTnGNyI2OE9hap_EAY7HTA).
  - **User Concerns**: Users are questioning the functionality and limitations of the new feature, such as whether it can read for more than **4 minutes** without restarting, and expressing dissatisfaction with the current rate limit for video sharing.
  - **Feature Requests**: Users are requesting additional features, such as making the **Operator** available for free and introducing **Advanced Memory** capabilities.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. Claude 3.7 Sonnet Storms the AI Scene**

- **Sonnet 3.7 Unleashes Coding Chaos**: [Anthropic's Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) is making waves with its superior coding abilities, particularly in agentic tasks, leading to user excitement and rapid integration into tools like [Cursor IDE](https://www.cursor.sh) and [Aider](https://aider.chat).  Users are reporting significant performance boosts, especially in front-end development and complex problem-solving, but some debate whether the reported 3x price increase for "thinking tokens" is justified given the performance gains.
- **Thinking Mode Unveiled, But Not Without Quirks**:  **Claude 3.7 Sonnet** introduces a new 'thinking mode' with up to **64,000 output tokens**, visible in tools like [Sage](https://www.sage.com), allowing users to observe the model's reasoning process through `<thinking>` tags. However, some users are experiencing issues with context window management and rule adherence in [Cursor](https://www.cursor.sh), and others note a 10-second delay in output display with **O3** models, although most agree the overall performance is a major upgrade.
- **Claude Code Challenges Aider's Code Editing Crown**:  Anthropic's release of [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), a terminal-based agentic coding tool, is seen by some as an Aider clone, but early reports suggest it excels at code assistance, outperforming Aider in complex error resolution tasks, such as fixing **21 compile errors** in Rust in one go.  The tool is currently a limited research preview separate from Anthropic subscriptions, sparking discussions about caching mechanisms and potential cost implications, with some users reporting "astronomical Anthropic costs" recently.

**Theme 2. DeepSeek's Deep Dive into Model Efficiency**

- **MLA: Shrinking KV Cache, Expanding Horizons**: [DeepSeek AI's Multi-Head Latent Attention (MLA)](https://arxiv.org/abs/2502.14837) is gaining attention for its potential to drastically reduce **KV cache** size by **5-10x**, with papers like [MHA2MLA](https://arxiv.org/abs/2502.14837) and [TransMLA](https://arxiv.org/abs/2502.07864) exploring its implementation in models like **Llama**. While early results show mixed performance impacts (**1-2%** performance drop in some cases, enhancement in others), the significant memory savings make MLA a promising avenue for efficient inference, particularly for larger models.
- **DeepEP: Open-Sourcing MoE Training's Secret Sauce**:  DeepSeek has released [DeepEP](https://github.com/deepseek-ai/DeepEP), the first open-source EP communication library designed for efficient all-to-all communication in **Mixture of Experts (MoE)** model training and inference.  This library enables efficient expert parallelism and supports FP8, potentially democratizing access to advanced MoE model architectures and training techniques.
- **DeepScaleR: RL Supercharges Smaller Models**:  [DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2), fine-tuned from **Deepseek-R1-Distilled-Qwen-1.5B** using simple **Reinforcement Learning (RL)**, achieved **43.1% Pass@1 accuracy** on AIME2024, demonstrating that RL techniques can significantly boost the performance of smaller models, potentially surpassing larger models like **O1 Preview** in specific tasks.

**Theme 3. Open Source Tooling and Ecosystem Growth**

- **OpenRouter Opens Gates to Claude 3.7 and Beyond**: [OpenRouter](https://openrouter.ai/) has rapidly integrated **Claude 3.7 Sonnet**, offering access to the model with competitive pricing at **$3 per million input tokens** and **$15 per million output tokens**, including thinking tokens, and plans to soon support **Claude 3.7's** extended thinking feature.  OpenRouter also provides access to other models like `o3-mini-high` via [OpenRouter](https://openrouter.ai/openai/o3-mini-high), offering a cost-effective alternative and a single point of access to multiple providers, potentially bypassing rate limits and costing around **$3** for **2 hours** of coding.
- **QuantBench Quantifies Quantization Speed**:  The release of **QuantBench** on [GitHub](https://github.com/Independent-AI-Labs/local-super-agents/tree/main/quantbench) is accelerating quantization workflows, demonstrated by its use in creating the **Qwen 2.5 VL 7B** GGUF quant, available on [Hugging Face](https://huggingface.co/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF). This tool, tested with the latest **llama.cpp** and **CLIP** hardware acceleration, simplifies and speeds up the process of model quantization, making efficient model deployment more accessible.
- **MCP Registry API: Standardizing AI Agent Development**: Anthropic's announcement of the official [MCP registry API](https://x.com/opentools_/status/1893696402477453819) is hailed as a significant step towards standardizing **Model Context Protocol (MCP)** development. This API aims to become *the* source of truth for MCPs, promoting interoperability and streamlining integration efforts for AI applications and agents, with community projects like [opentools.com/registry](http://opentools.com/registry) already leveraging it.

**Theme 4. Benchmarking Battles: Models Face Real-World Tests**

- **Kagi's Benchmarks Crown Gemini 2.0 Pro, But Sonnet Still Strong**: According to the [Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html), **Google's gemini-2.0-pro-exp-02-05** achieved **60.78%** accuracy, outperforming **Anthropic's claude-3-7-sonnet-20250219** at **53.23%** and **OpenAI's gpt-4o** at **48.39%**, however, **Claude Sonnet 3.7** still shows strong performance, particularly on the [Aider polyglot leaderboard](https://aider.chat/docs/leaderboards/) where it scored **65%** using thinking tokens.  These benchmarks highlight the dynamic landscape of LLM performance and the ongoing race for accuracy and efficiency.
- **Misguided Attention Eval Exposes Overfitting Weakness**: The [Misguided Attention Eval](https://github.com/cpldcpu/MisguidedAttention) is being used to test LLMs' reasoning abilities in the presence of misleading information, specifically targeting **overfitting**.  **Sonnet-3.7** benchmarked as the top non-reasoning model in this evaluation, nearly surpassing **o3-mini**, suggesting it exhibits robust performance even when confronted with deceptive prompts.
- **SWE Bench Sees Claude 3.7 Grab Top Spot**: [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) is now leading on the SWE bench, demonstrating its prowess in software engineering tasks.  Its capabilities extend to active code collaboration, including searching, editing, testing, and committing code to GitHub, solidifying its position as a top contender for coding-related applications.

**Theme 5. Hardware Horizons: From Brains to Silicon**

- **Brain's Parallelism Puzzles GPU Architects**: Discussions are comparing the brain's *stateful parallel processing* to GPU efficiency, suggesting that current RNN architectures, while leveraging parallel processing, do not fully capture the brain's capabilities and may not scale optimally for LLMs.  The consensus is that *extremely tuned architectures* and inductive biases, inspired by the brain, may be more crucial than simply scaling up model size for future advancements.
- **Speculative Decoding Speeds Up LM Studio**:  Users are exploring **speculative decoding** in [LM Studio](https://lmstudio.ai/), particularly with **Llama 3.1 8B** and **Llama 3.2 1B** models, as documented in [LM Studio's documentation](https://lmstudio.ai/docs/advanced/speculative-decoding). This technique, which uses a smaller "draft" model to predict tokens for a larger model, promises to significantly increase generation speed without compromising response quality, enhancing the efficiency of local LLM inference.
- **M2 Max Still a Power Sipper Compared to M4 Max**:  While the **M4 Max** is the latest from Apple, some users are sticking with the **M2 Max**, citing concerns about the **M4 Max's** high power consumption, reaching **140W**, compared to the **M2 Max's** more efficient **60W**.  For users with sufficient performance from the **M2 Max**, especially those running locally, the power efficiency and availability of refurbished models make it a compelling alternative.



---

# PART 1: High level Discord summaries




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Claude 3.7 Sonnet Triggers Coding Boom**: **Claude 3.7 Sonnet** is being rolled out in [Cursor IDE](https://www.anthropic.com/news/claude-3-7-sonnet) with users reporting superior coding capabilities, especially in real-world agentic tasks.
   - Enthusiastic users proclaimed *Sleeping has become optional*, and are rapidly integrating the model.
- **MCPs Supercharge Claude's Coding Abilities**: Members are combining **MCPs** (Model Control Programs) like perplexity search and browser tools with custom instructions to boost **Claude 3.7**'s reasoning and coding capabilities in [Cursor](https://www.cursor.sh).
   - One user forked the *sequential thinking* MCP with their own tweaks, highlighting the benefits of combining custom instructions with MCP servers.
- **Installation Tips and Tricks Released for Cursor**: Users shared tips for installing and updating to **Cursor 0.46.3** to access **Claude 3.7**, including manually adding the model and checking for updates, as well as links to direct downloads for various operating systems like [Windows](https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/win32/x64/user-setup/CursorUserSetup-x64-0.46.3.exe) and [macOS](https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/darwin/arm64/Cursor-darwin-arm64.zip).
   - Several users noted difficulties with the auto-update feature, recommending manual download and installation for a smoother experience.
- **Sonnet 3.7 Reaches New SVG Heights**: Many agreed that **Sonnet 3.7** is a major upgrade, especially for frontend tasks and code generation, with members [praising its ability to generate landing pages](https://discord.com/channels/1074847527708393562/1074847528224360509/1343738071660011520).
   - Members shared examples of complex tasks, like recreating X's UI or generating SVG code, being handled with ease.
- **Context Window Problems and The Rule Bloat**: Several members noted issues with **Claude 3.7** in **Cursor**, including difficulties with code indexing in workspaces, custom rules bloating the context window, and the model sometimes ignoring those rules.
   - Despite these challenges, most users found workarounds and praised the model's overall performance.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Sonnet 3.7 Steals Aider's Spotlight**: **Claude 3.7 Sonnet** hit a **65% score** on the [Aider polyglot leaderboard](https://aider.chat/docs/leaderboards/), utilizing **32k thinking tokens**.
   - Some are debating if the performance increase justifies the reported **3x price hike** for **Sonnet 3.7** when using thinking tokens.
- **Anthropic drops Claude Code Aider-Clone**: **Anthropic** released **Claude Code**, considered by some to be an [Aider clone](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview).
   - Members are reporting the superiority of code quality and are hopeful for the future of **Claude 3.7** compared to **OpenAI**.
- **Unlock O3-Mini via OpenRouter**: The `o3-mini-high` model can be accessed through [OpenRouter](https://openrouter.ai/openai/o3-mini-high), is a model optimized for STEM reasoning tasks, and it is the same as `o3-mini` with reasoning effort set to high.
   - Coding sessions could cost around **$3** for **2 hours** of use using OpenRouter, which can bypass rate limits and offers single point of access to multiple providers.
- **HN Profile Gets Roasted by LLM**: **Claude Sonnet 3.7** can now analyze your [Hacker News profile](https://hn-wrapped.kadoa.com/) to give highlights and trends.
   - A member described the LLM's deep dive into their post history as a 'roast' that was allegedly *scary accurate*.
- **Gemini 2.0 Pro Outpaces Rivals, per Kagi**: According to the [Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html), **Google's gemini-2.0-pro-exp-02-05** achieved **60.78%** accuracy, surpassing **Anthropic's claude-3-7-sonnet-20250219** at **53.23%** and **OpenAI's gpt-4o** at **48.39%**.
   - **Gemini 2.0 Pro** also showed a median latency of **1.72s** and a speed of **51.25 tokens/sec**, compared to **Claude Sonnet 3.7's 2.82s** and **54.12 tokens/sec**, and **GPT-4o's 2.07s** and **4 tokens/sec**.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Vim Chat Plagued by Issues**: A user reported issues starting **Codeium Chat** in **Vim** via a **Putty SSH** session, facing connection errors when attempting to access the provided URL in a browser.
   - The error message indicated that *"This site can't be reached 127.0.0.1 refused to connect"*.
- **Windsurfers Await Claude 3.7 Arrival**: Members are eagerly anticipating the integration of **Claude 3.7** into Windsurf, expressing frustration over the perceived delay compared to platforms like Cursor and T3, and requesting its addition **ASAP**.
   - Members have asked for *windsurf should go and be early tester* - with devs cooking to push **Claude 3.7** into production with a possible release by end of day.
- **Deepseek Hallucinates User Prompts**: A user reports **Deepseek** hallucinating user requests and then starting to implement changes based on those hallucinated requests.
   - The AI bot *invented its own user prompt and then started to implement changes based on that hallucinated user prompt 😆*.
- **Windsurf Dev Comms Draw Fire**: Users are frustrated by the perceived lack of communication from the Windsurf devs regarding the **Claude 3.7** integration, with one user noting, *part of the frustration is there is no comms from the devs.*
   - Other users have defended Windsurf and noted a lack of commercial risk since it would release when more stable *being fast at implementing things doesn't mean it's solid.*
- **MCP Server Practicality Queried**: Users discussed practical uses for the **MCP server**, with examples including integrating **Jira tickets**, sharing of custom apps, and utilizing cloud services.
   - Members have asked, *What do you guys use MCP server for, practically? Are there real life examples that makes your life really easy? Can't think of any.*



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Grok 3 Talks Too Much**: Members find **Grok 3** to be too verbose despite prompting for concise responses, however it proves to be a **powerhouse in coding and creativity**.
   - One member noted that they are switching to Grok because it is *less censored out of the gate*.
- **Perplexity Plans Agentic Comet**: Perplexity is launching **Comet**, a new **agentic browser**, similar to The Browser Company's work.
   - The agentic browser space is heating up with more competitors.
- **Claude 3.7 Arrives with New Coding Power**: **Anthropic** just dropped **Claude 3.7 Sonnet** which shows improvements in coding and front-end web development and also introduces a command line tool for agentic coding: **Claude Code** [announcement here](https://www.anthropic.com/news/claude-3-7-sonnet).
   - One user pointed out that the model's knowledge cutoff date is **February 19, 2025**
- **Claude Code Enters the Terminal**: **Claude Code** is an agentic coding tool that lives in your terminal, understands your codebase, and helps you code faster through natural language commands [overview here](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview).
   - However it is a **limited research preview** and is separate to the pro or Anthropic subs.
- **O3 exhibits 10 second delay**: A user reported issues with **O3**, where it indicates *reasoning success* but then delays displaying the full text for up to 10 seconds, affecting various models including **O1 Pro**.
   - They mentioned experiencing these problems consistently between **3pm-7pm EST**, with text sometimes appearing on different devices than expected.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Tax Evasion Talk Results in Timeout**: A user was muted for discussing tax avoidance strategies, as giving tax avoidance recommendations is against the rules; some users pointed out the implications for invoicing.
   - A user responded *the company i was billing invoice too told me stupid that i was reporting income*.
- **CUDA Kernel Causes Colab Catastrophe**: A user reported a CUDA error (*illegal memory access*) on Google Colab with T4, suggesting trying setting `CUDA_LAUNCH_BLOCKING=1` and compiling with `TORCH_USE_CUDA_DSA` for debugging, as per [PyTorch documentation](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables).
   - Another user reported *weird spikes in grad norm up to 2000*, suggesting the model might be broken.
- **Qwen2.5 VL 72B Eats Memory Alive**: A user faced out-of-memory errors trying to run **Qwen2.5 VL 72B** on 48GB with a 32K context length, then successfully loaded it with 8k context length after being advised to try 8k or quantize the KV cache to fp8.
   - The user noted it was necessary to extract the *thinking traces* from the model.
- **DeepSeek MLA ported to Llama via TransMLA**: Users explored implementing **DeepSeek's Multi-Head Latent Attention (MLA)** on a **Llama** model, suggesting retraining, but others pointed to [fxmeng/TransMLA](https://github.com/fxmeng/TransMLA), a post-training conversion method from GQA to MLA.
   - The linked paper is called [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://arxiv.org/abs/2502.14837).
- **rslora role in Rank Stability**: The use of **rslora** addresses numerical stability in high rank scenarios, but a user cautioned that if r/a = 1, **rslora** can worsen things, advising to keep r/a = 1 and skip **rslora**.
   - The team stated that **rslora** performs a single sqrt and requires a correction term if the rank gets too big.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.7 Sonnet Lands on OpenRouter!**: **Claude 3.7 Sonnet** is now available on OpenRouter with best-in-class performance in [mathematical reasoning, coding, and complex problem-solving](https://www.anthropic.com/news/claude-3-7-sonnet).
   - The pricing is set at **$3 per million input tokens** and **$15 per million output tokens**, including thinking tokens, with full caching support at launch.
- **Extended Thinking Feature Coming Soon**: The **Extended Thinking** feature is coming soon to the OpenRouter API, which enables step-by-step processing for complex tasks, as detailed [in Anthropic's documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking).
   - OpenRouter is actively working on implementing full support for **Claude 3.7's** *extended thinking* feature, which does not currently support pre-fills, aiming for launch soon with updated documentation.
- **GCP Gears Up for Claude 3.7**: **Google Cloud Platform (GCP)** is preparing to support **Claude 3.7 Sonnet**, launching in **us-east5** and **europe-west1** with model ID `claude-3-7-sonnet@20250219`.
   - Users are reminded that the model features a **hybrid reasoning approach**, offering both standard and extended thinking modes and maintaining performance parity with its predecessor in standard mode.
- **OpenRouter Revs Up Claude 3.7 Throttling**: OpenRouter increased the **TPM (tokens per minute)** for `anthropic/claude-3.7-sonnet`, while `anthropic/claude-3.7-sonnet:beta` has a lower TPM initially, set to increase as users migrate from **3.5**.
   - The model has a **200,000 token context window**, though some users feel its output pricing might cause complaints.
- **API Key Credits Safety Clarified**: Users are reminded that **API keys do not contain credits**; deleting a key only revokes access, and credits remain tied to the account.
   - Lost keys cannot be recovered due to security measures.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Meta AI Expands to MENA**: [Meta AI](https://about.fb.com/news/2025/02/meta-ai-launches-in-the-middle-east-empowering-new-era-of-creativity-and-connection/) has expanded to the **Middle East and North Africa (MENA)**, supporting **Arabic** on **Instagram, WhatsApp, and Messenger**.
   - This expansion opens the chatbot to millions more users in the region.
- **Claude 3.7 Sonnet launches with Thinking Mode**: Anthropic launched **Claude 3.7 Sonnet**, a **hybrid reasoning model** with step-by-step thinking, and **Claude Code**, a command line tool for agentic coding, priced at **$3 per million input tokens** and **$15 per million output tokens**.
   - Researchers noted Claude's thought process as *eerily similar* to their own, exploring different angles and double-checking answers, showcasing improvements using parallel test-time compute scaling on the **GPQA** evaluation.
- **Qwen Chat reasoning model Released**: **Alibaba Qwen** released "Thinking (QwQ)" in **Qwen Chat**, backed by their **QwQ-Max-Preview**, which is a reasoning model based on **Qwen2.5-Max**, licensed under **Apache 2.0**.
   - The model will come in smaller variants, e.g., **QwQ-32B**, for local deployment, with a [viral Twitter demo](https://fxtwitter.com/Alibaba_Qwen/status/1894130603513319842) showcasing improved math, coding, and agent capabilities.
- **Berkeley Advanced Agents MOOC Features Tulu 3**: The **"Berkeley Advanced Agents" MOOC** features **Hanna Hajishirzi** discussing **Tulu 3** today, May 30th, at 4PM PST, with a link to the [YouTube video](https://www.youtube.com/live/cMiu3A7YBks).
   - The MOOC has been gaining traction as a great resource for engineers interested in agents.
- **Google's Co-Scientist fed Team's Prior Work**: **Google's Co-Scientist AI** tool, based on the **Gemini LLM**, had been **fed a 2023 paper** by the team it was assisting, including a version of the hypothesis that the AI tool later suggested as a solution.
   - The [article](https://pivot-to-ai.com/2025/02/22/google-co-scientist-ai-cracks-superbug-problem-in-two-days-because-it-had-been-fed-the-teams-previous-paper-with-the-answer-in-it/) highlighted the BBC coverage failed to mention that the AI tool was given the answer, raising eyebrows.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Parallel Brains Outpace Tuned GPUs**: Discussions compared the brain's *stateful parallel processing* to GPU efficiency, noting current RNN architectures, which differ from human processing, cannot scale to LLM level and should be *data efficient*.
   - Members concluded that *extremely tuned architectures* become more relevant than simply scaling up when drawing inspiration from the brain.
- **Proxy Engine Structures LLM Chaos**: The [Proxy Structuring Engine (PSE)](https://www.proxy.ing/pse) was introduced to address structural inconsistencies in LLM outputs, providing **inference-time steering** for creative freedom.
   - The engine enforces *structure boundaries* and it is fit for use cases like *Advanced Agents & Chatbots*, *Data Pipelines & APIs*, and *Automated Code Generation*.
- **Wavelet Coding Tokenizes Image Generation**: A new approach to autoregressive image generation based on **wavelet image coding** and a variant of a language transformer is detailed in [this paper](https://arxiv.org/abs/2406.19997).
   - The transformer learns statistical correlations within a token sequence, reflecting correlations between wavelet subbands at various resolutions.
- **MLA Squeezes KV Cache**: Two papers, [MHA2MLA](https://arxiv.org/abs/2502.14837) and [TransMLA](https://arxiv.org/abs/2502.07864), explore adapting models to **Multi-head Latent Attention (MLA)**, significantly reducing **KV cache** size (**5-10x**).
   - While one paper showed deteriorated performance (**1-2%**), the other showed enhanced performance, suggesting **MLA** could be non-inferior to **MHA**, especially with larger models and more parameters.
- **Mixed Precision Toggles Optimizer Defaults**: During mixed precision training with **BF16**, the master **FP32 weights** typically reside in **GPU VRAM**, unless **ZeRO offload** is enabled.
   - It is common to store the **first and second Adam moments in bf16**, while keeping master weights in **fp32**, unless the expert sharding with **momentum/variance states** via **ZeRO**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLMs Invoke Tools Autonomously**: Some LLMs invoke tools without explicit token sequences, suggesting **hard-coded patterns** from training via reinforcement learning or SFT.
   - This token-saving approach's reliability compared to ICL remains unclear without benchmarks.
- **Claude 3.7 Sonnet Takes the SWE Crown**: [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) leads on the SWE bench, enabling **active code collaboration** like searching, editing, testing, and committing code to GitHub.
   - A member suggested that 3.7 being a point release makes sense since *Claude 3.5 was already a reasoning model*, also hinting that future reasoning models will be 'crazy'.
- **QwQ-Max-Preview Aims for Deep Reasoning**: [QwQ-Max-Preview blog](https://qwenlm.github.io/blog/qwq-max-preview/) shows a model built on Qwen2.5-Max that excels in **deep reasoning, math, coding, general domains, and agent tasks**.
   - Speculation arose around key tokens in **QwQ's reasoning traces** resembling **R1**, suggesting it requires less compute.
- **Sonnet-3.7 Excels in Misguided Attention Eval**: **Sonnet-3.7** benchmarked as top non-reasoning model in [Misguided Attention Eval](https://github.com/cpldcpu/MisguidedAttention), nearly surpassing **o3-mini**.
   - The user seeks to activate its *thinking mode* via the OR API, if feasible.
- **Qwen AI Adds Integrated Video Generation**: The updated **Qwen AI** chat interface now features integrated **video generation** capabilities.
   - A member noted that the artifacts are still a bit clunky, like a *half baked copy*.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Anthropic Finally Delivers MCP Registry API**: Anthropic announced the official **MCP registry API**, as seen on [this tweet](https://x.com/opentools_/status/1893696402477453819), to be *the* source of truth for **MCPs**, streamlining development and integration efforts with solutions like [opentools.com/registry](http://opentools.com/registry).
   - This API will help the community fill the source-of-truth gap for portable & secure code for **AI Apps and Agents**.
- **Claude 3.7 Debuts 'Thinking' Tags**: **Claude 3.7** has been released, featuring **64,000 output extended thinking tokens** and a new 'latest' alias.
   - Users noted it is back to *following long-ish system prompts, spotting social engineering*, and also utilizes `<thinking>` tags when using tools, adding a cute touch to its operation.
- **Claude Code Excels as Code Assistant**: **Claude Code (CC)** is receiving high praise for its code assistance capabilities, outperforming tools like **Aider** in handling complex coding errors, such as resolving **21 compile errors** in Rust in one shot.
   - Users are speculating on caching mechanisms and costs, with one user reporting *astronomical Anthropic costs in the last 6 weeks*.
- **MetaMCP** Debates Open-Source Licensing**: Concerns were raised regarding **MetaMCP's** licensing, with a user suggesting it might become a cloud **SaaS**, prompting the developer to seek feedback on licensing to prevent cloud monetization while keeping it self-hostable via the [MetaMCP server GitHub repository](https://github.com/metatool-ai/mcp-server-metamcp).
   - A user suggested using **AGPL** licensing for **MetaMCP** to ensure contributions are open-sourced, also suggesting an additional clause allowing the company to sublicense under MIT-0.
- **Claude 3.7 Sonnet** Shines on Sage**: **Claude 3.7 Sonnet** with extended thinking capabilities is now on **Sage**, allowing users to see **Claude's reasoning process** as it tackles complex problems, including a **thinking mode toggle** (Command+Shift+T).
   - Other new features include default model settings, improved scrolling, and expandable thinking blocks.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 2.5 VL Model Ready to Rumble**: A working **Qwen 2.5 VL 7B** GGUF has arrived and is available on [Hugging Face](https://huggingface.co/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF) for immediate use.
   - Users report that it performs significantly better than **llama3.2 vision 11b instruct** and **qwen2-vision 7b instruct**, and works out of the box on the latest version of LM Studio.
- **QuantBench Accelerates Quantization**: The **Qwen 2.5 VL 7B** GGUF quant was produced using **QuantBench**, now available on [GitHub](https://github.com/Independent-AI-Labs/local-super-agents/tree/main/quantbench) for accelerated quant workflows.
   - The model has been successfully tested on the latest **llama.cpp** build, with **CLIP** hardware acceleration enabled.
- **LM Studio Reveals Speculative Decoding Secrets**: Users are exploring **speculative decoding** with **Llama 3.1 8B** and **Llama 3.2 1B** models in LM Studio, according to [LM Studio's documentation](https://lmstudio.ai/docs/advanced/speculative-decoding).
   - The documentation claims that speculative decoding *can substantially increase the generation speed of large language models (LLMs) without reducing response quality*.
- **Deepseek R1 671b Gorging RAM**: Running **Deepseek R1 671b** locally needs serious RAM, with documentation specifying **192GB+**; one helpful user suggested using a specific [quantized version](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S).
   - For those running on Macs, offloading approximately **70%** of the model weights to the GPU may help.
- **M2 Max Sipping Power**: Despite the shiny new **M4 Max**, one user decided to stick with their **M2 Max**, as *M4 Max boosts way too hard easily pegged at 140w* and located a *well priced refurbished M2 Max 96GB*.
   - The user reports the **M2 Max** is sufficient for their needs, pulling only around **60W**.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 Ultra's Unseen Excellence**: A user asked about **SD3 Ultra**, a *comfy workflow based on SD3L 8B* that delivers superior high-frequency detail.
   - Another member stated it *still exists* and is being used, implying it is not yet a public release.
- **Silence from Stability?**: A member asked about updates on current projects or future plans, noting they *haven't heard anything for a while* from **Stability AI**.
   - Another member responded that *nothing can be shared yet*, but they are *hopefully* expecting announcements soon.
- **Dog Datasets Desired**: A user requested alternative dog breed image datasets beyond the **Stanford Dogs Dataset**, which contains **20k images**.
   - The user specifically needs images containing both the dog and its breed clearly labeled.
- **Image Generation Times Vary**: Users discussed image generation times based on different hardware configurations, using various versions of **Stable Diffusion**.
   - Times ranged from *around 1 minute* on a **GTX 1660s** to *4-5s* on a **3070ti** using **SD1.5**, and **7 seconds** for a **1280x720** image and **31 seconds** for **1920x1080** at **32 steps** with a **3060 TI**.
- **Stability AI Solicits Suggestions**: **Stability AI** launched a [new feature request board](https://stabilityai.featurebase.app/) to gather user feedback and prioritize future developments.
   - Users can submit and vote on feature requests directly from **Discord** using the **/feedback** command or through the new platform, aiming to ensure community voices shape future priorities.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Conjures Graphics with GLFW/GLEW**: Graphics programming in Mojo is feasible via **FFI** using a **static library linked to GLFW/GLEW**, evidenced by a **Sudoku example**.
   - A member suggested *exposing only the needed calls via your own C/CPP library* using `alias external_call` with a wrapped function, plus [an example repo](https://github.com/ihnorton/mojo-ffi) shows how to hijack the loader.
- **Mojo's `magic install` Faces `lightbug_http` Bug**: Using `lightbug_http` dependency in a new Mojo project leads to an error with `small_time.mojopkg` after running `magic install`.
   - The error resembles [a Stack Overflow question](https://stackoverflow.com/questions/79319716/magic-fails-to-install-mojo-dependencies), hinting that `small-time` might be pinned to a specific version.
- **MAX's Game of Life gets Accelerated by Hardware**: A member showcases a hardware-accelerated **Conway's Game of Life** by bridging **MAX** and **Pygame**, revealing a creative application, as shown in their attached [conway.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343753014229471272/conway.gif).
   - They demonstrated the use of **GPU** in their **MAX** implementation by showcasing a guns pattern, packed bit by bit, rendered using a naive pixel-by-pixel internal function, and then the output tensor gets cast into an np array and given to pygame to render, as demonstrated in their [guns.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343766916560322560/guns.gif).
- **Game of Life Creates Computer Architectures**: A member shared a project ([nicolasloizeau.com](https://www.nicolasloizeau.com/gol-computer)) about crafting a **computer** within **Conway's Game of Life**, demonstrating its **Turing completeness** via glider beams for logic gates.
   - A member also implemented wrapping in their Conway's Game of Life simulation using **MAX**, enabling the creation of spaceship patterns and showcasing the ability to add parameters to the model from the graph API, as showcased in their [spaceship.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343808736623591465/spaceship.gif).



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Eases Use with PowerPoint Conversion**: A user detailed a workaround to import physical books into **NotebookLM** by photographing pages, converting the **PDF** to **PowerPoint**, uploading to **Google Slides**, and importing the slides.
   - They observed that **NotebookLM** can process text images in slides, but not directly from **PDF** files.
- **Language Prompts Misfire on German**: A user reported issues getting **NotebookLM** hosts to speak German, even with specific prompts requesting German.
   - The hosts spoke English or gibberish, sometimes starting in German before switching, indicating potential issues with **language prompt accuracy**.
- **Savin/Ricoh Copier Revives Book Scanning**: A user advised scanning books to **PDF** using a **Savin/Ricoh copier** and uploading to **NotebookLM**.
   - They affirmed that even with poor source text quality, **NLM** accurately answered questions about the scanned document.
- **Users Request Language Customization**: A user inquired about the feasibility of changing the language in **NotebookLM** without altering the **Google account language**.
   - This points to a demand for language customization to improve user experience and cater to diverse linguistic preferences.
- **Claude 3.7 Ignites Model Choice Fantasies**: A user expressed enthusiasm for **Claude 3.7** and desired the option to select models in **NotebookLM**.
   - Another user questioned the impact of model choice, sparking a discussion on the **implications of model variety** for the end user experience.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Unveils AI Assistant in Docs**: LlamaIndex [announced](https://t.co/XAyW7wLALJ) the release of an **AI assistant** directly within their documentation.
   - The new assistant aims to provide immediate, contextual support to users navigating the LlamaIndex ecosystem.
- **ComposIO HQ Drops a Banger**: LlamaIndex highlighted another new release from [ComposIO HQ](https://t.co/W4l129gHce), though specifics of the release were unmentioned.
   - This indicates ongoing development and feature enhancements within the ComposIO framework, a tool useful for LLM orchestration.
- **AnthropicAI Releases Claude Sonnet 3.7**: [AnthropicAI](https://twitter.com/anthropicAI) launched **Claude Sonnet 3.7**, with LlamaIndex offering immediate support.
   - Users can access the new model by running `pip install llama-index-llms-anthropic --upgrade` and reviewing [Anthropic's announcement](https://t.co/PjaQWmmzaN).
- **Fusion Rerank Retriever Demands Initialized Nodes**: A user reported issues initializing the **BM25 retriever** within a **fusion rerank retriever** setup with **Elasticsearch** because the docstore was empty.
   - Another member clarified that **BM25** requires nodes to be saved to disk or another location for initialization, as it *cannot initialize directly from the vector store*.
- **MultiModalVectorStoreIndex Throws File Error**: A user encountered a *[Errno 2] No such file or directory* error when creating a multimodal vector index using **MultiModalVectorStoreIndex** with **GCSReader**.
   - The error occurred with image files present in the GCS bucket, while **PDF documents** were processed successfully, indicating a potential issue with image file handling.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Truncation Troubles: Left Prevails**: Members debated the use of **left truncation** `seq[-max_seq_len:]` vs **right truncation** `seq[:max_seq_len]` during finetuning, with [interesting graphs](https://cdn.discordapp.com/attachments/1236040539409879170/1343641196836294746/image.png?ex=67be02e0&is=67bcb160&hm=9411a00c21d408790c46140222f996913807ded5a1d5c00a02a6742aa44ba285&).
   - The final decision involved *exposing both methods* but *defaulting to left truncation* for SFT in `torchtune`.
- **StatefulDataLoader Support: Merge Incoming**: A member is requesting review for their [PR adding support for the `StatefulDataLoader` class](https://github.com/pytorch/torchtune/pull/2410) in `torchtune`.
   - The new dataloader would add statefulness to the dataset.
- **DeepScaleR Scales with RL**: [DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) was finetuned from **Deepseek-R1-Distilled-Qwen-1.5B** using simple **reinforcement learning (RL)**.
   - DeepScaleR achieved **43.1% Pass@1 accuracy** on AIME2024.
- **DeepSeek Opens EP Communication Library**: DeepSeek introduced [DeepEP](https://github.com/deepseek-ai/DeepEP), the first open-source EP communication library for **MoE model training and inference**.
   - The communication library enables efficient all-to-all communication.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Validators Ponder Profitability Threshold**: A member inquired about the profitability threshold for **Proof of Stake (PoS) validators** within the **Decentralized Science (DeSci)** field.
   - Another member responded with *"pool validator node"*, hinting at the importance of pool participation for validators.
- **Asset Expert Gets Labeled**: The bot posted about an *"asset value expert account"* which was labelled as *"nazi"*.
   - No further context was given.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Simplifies Assertion Migration**: DSPy users can now use `dspy.BestOfN` or `dspy.Refine` modules to streamline migration from **2.5-style Assertions**.
   - The `dspy.BestOfN` module retries a module up to **N** times, selecting the best reward and halting upon reaching a specified `threshold`.
- **DSPy crafts reward functions**: DSPy's **reward functions** now support scalar values such as *float* or *bool*, which allows customized evaluation of module outputs.
   - A sample reward function was shown: *def reward_fn(input_kwargs, prediction): return len(prediction.field1) == len(prediction.field1)*.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1343632399908471074)** (1056 messages🔥🔥🔥): 

> `Claude 3.7 Sonnet release, Cursor IDE integration, MCPs with Claude, Comparisons of Claude 3.7 with other models (GPT-4, O3), Troubleshooting Cursor and Claude 3.7` 


- **Claude 3.7 Sonnet Causes Coding Frenzy**: **Claude 3.7 Sonnet** is celebrated for its superior coding capabilities, especially in real-world agentic tasks, and is being rolled out in [Cursor IDE](https://www.anthropic.com/news/claude-3-7-sonnet).
   -  Enthusastic users proclaimed *Sleeping has become optional* with many quickly integrating the model and lauding its performance.
- **MCPs Enhance Claude's Coding Prowess**: Members discussed using **MCPs** (Model Control Programs) like perplexity search and browser tools, and combining them with custom instructions to extend **Claude 3.7**'s reasoning and coding capabilities in [Cursor](https://www.cursor.sh).
   - One user forked the *sequential thinking* MCP with their own tweaks, emphasizing the benefits of combining custom instructions with MCP servers.
- **Installation Tips and Tricks Unleashed for new Cursor Update**: Users shared tips for installing and updating to **Cursor 0.46.3** to access **Claude 3.7**, including manually adding the model and checking for updates, as well as links to direct downloads for various operating systems like [Windows](https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/win32/x64/user-setup/CursorUserSetup-x64-0.46.3.exe) and [macOS](https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/darwin/arm64/Cursor-darwin-arm64.zip).
   - Several users noted difficulties with the auto-update feature, recommending manual download and installation for a smoother experience.
- **Thinking Model takes SVG Code Generation to the next level**: Many agreed that **Sonnet 3.7** is a major upgrade over previous models, especially for frontend tasks and code generation, with one user exclaiming *this shit feels like new level ai* and [others praising its ability to generate landing pages](https://discord.com/channels/1074847527708393562/1074847528224360509/1343738071660011520).
   - Members shared examples of complex tasks, like recreating X's UI or generating SVG code, being handled with ease.
- **Context Window Struggles and The Rule Bloat**: Several members noted issues with **Claude 3.7** in **Cursor**, including difficulties with code indexing in workspaces, custom rules bloating the context window, and the model sometimes ignoring those rules.
   - Despite these challenges, most users found workarounds and praised the model's overall performance, with one stating *the model tries to first make sure it understands the project before it makes changes this is great*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/it-turn-on-and-off-phone-call-tech-support-gif-13517106">It Turn On And Off GIF - IT Turn On And Off Phone Call - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://forum.cursor.com/t/integrate-claude-3-7-sonnet-into-cursor/54060">Integrate Claude 3.7 Sonnet into Cursor</a>: Hello!  I would love to see Claude’s new reasoning-model (3.7) being integrated with the agent inside of composer.  Regards  Johannes.</li><li><a href="https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview">Claude Code overview - Anthropic</a>: no description found</li><li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/client/linux/x64/appimage/Cursor-0.46.3-bbefc49a7fd08b08a4f17a525bdc5bb7e44ce57a.deb.glibc2.25-x86_64.AppImage">no title found</a>: no description found</li><li><a href="https://x.com/alibaba_qwen/status/1894130603513319842?s=46">Tweet from Qwen (@Alibaba_Qwen)</a>: &lt;think&gt;...&lt;/think&gt; QwQ-Max-PreviewQwen Chat: https://chat.qwen.ai/Blog: https://qwenlm.github.io/blog/qwq-max-preview/🤔 Today we release &#34;Thinking (QwQ)&#34; in Qwen Chat, backed by o...</li><li><a href="https://x.com/ChujieZheng/status/1894095584774250858">Tweet from Chujie Zheng (@ChujieZheng)</a>: r u kidding me dude?</li><li><a href="https://x.com/cursor_ai/status/1894093438863511742">Tweet from Cursor (@cursor_ai)</a>: We&#39;re rolling out access to the highest level of thinking. To try it out, select claude-3.7-sonnet-thinking or claude-3.7-sonnet and enable agent mode.</li><li><a href="https://forum.cursor.com/t/indexing-only-reads-first-folder-in-the-workspace/2585/20">Indexing only reads first folder in the workspace</a>: Any update on this? this is really importanttt 😬</li><li><a href="https://cursor.directory/rules">Cursor Directory</a>: Find the best cursor rules for your framework and language</li><li><a href="https://www.bonfire.com/vibe-coding/?utm_source=copy_link&utm_medium=post_campaign_launch&utm_campaign=vibe-coding&utm_content=default">Vibe Coding | Bonfire</a>: Buy Vibe Coding merchandise that supports Nova Ukraine. Featuring Dark Heather Grey Premium Unisex Tees, professionally printed in the USA.</li><li><a href="https://x.com/sualehasif996/status/1894094715479548273">Tweet from Sualeh (@sualehasif996)</a>: Configurable thinking out soon! 👀Quoting Cursor (@cursor_ai) Sonnet-3.7 is available in Cursor!  We&#39;ve been very impressed by its coding ability, especially on real-world agentic tasks. It appear...</li><li><a href="https://x.com/cursor_ai/status/1894093436896129425">Tweet from Cursor (@cursor_ai)</a>: Sonnet-3.7 is available in Cursor!  We&#39;ve been very impressed by its coding ability, especially on real-world agentic tasks. It appears to be the new state of the art.</li><li><a href="https://www.bonfire.com/vibe-coding/?utm_source=copy_link&utm_medium=post_campaign_launch&utm_campai">Vibe Coding | Bonfire</a>: Buy Vibe Coding merchandise that supports Nova Ukraine. Featuring Dark Heather Grey Premium Unisex Tees, professionally printed in the USA.</li><li><a href="https://x.com/sualehasif996/status/1894094715479548273?s=46">Tweet from Sualeh (@sualehasif996)</a>: Configurable thinking out soon! 👀Quoting Cursor (@cursor_ai) Sonnet-3.7 is available in Cursor!  We&#39;ve been very impressed by its coding ability, especially on real-world agentic tasks. It appear...</li><li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/darwin/arm64/Cursor-darwin-arm64.zip">no title found</a>: no description found</li><li><a href="https://github.com/AgentDeskAI/browser-tools-mcp">GitHub - AgentDeskAI/browser-tools-mcp: Monitor browser logs directly from Cursor and other MCP compatible IDEs.</a>: Monitor browser logs directly from Cursor and other MCP compatible IDEs. - AgentDeskAI/browser-tools-mcp</li><li><a href="https://x.com/theo/status/1894101944068641241?t=iXaaI_9aHmFsjJYsjiGZhw">Tweet from Theo - t3.gg (@theo)</a>: The claude 3.7 thinking model fails the bouncing ball challenge the same way that grok 3 did? 🤔</li><li><a href="https://chat.qwen.ai/">Qwen Chat</a>: no description found</li><li><a href="https://github.com/alexandephilia/ChatGPT-x-DeepSeek-x-Claude-Linux-APP">GitHub - alexandephilia/ChatGPT-x-DeepSeek-x-Grok-x-Claude-Linux-APP: Electron-based desktop applications for various AI chat platforms.</a>: Electron-based desktop applications for various AI chat platforms.  - GitHub - alexandephilia/ChatGPT-x-DeepSeek-x-Grok-x-Claude-Linux-APP: Electron-based desktop applications for various AI chat p...</li><li><a href="https://x.com/i/grok/share/ZwWdnR4SkIC2qjoljYogGIGqv">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://github.com/daniel-lxs/mcp-starter/pull/4">docs: Add Windows-specific build instructions by rexdotsh · Pull Request #4 · daniel-lxs/mcp-starter</a>: Hi, thanks for the great tool!I&amp;#39;ve added some instructions for using -ldflags &amp;quot;-H=windowsgui&amp;quot; when building on Windows to prevent terminal windows from opening every time cur...</li><li><a href="https://www.cursor.com/changelog">Changelog | Cursor - The AI Code Editor</a>: New updates and improvements.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1343631676680437821)** (935 messages🔥🔥🔥): 

> `Claude 3.7, Aider Benchmarks, Claude Code, Thinking Models, OpenAI vs. Anthropic` 


- ****Sonnet 3.7 Steals Aider's Spotlight!****: **Claude 3.7 Sonnet** achieved a **65% score** on the [Aider polyglot leaderboard](https://aider.chat/docs/leaderboards/), utilizing **32k thinking tokens**.
- ****The Cost Of 3.7 Thinking Questioned!****: The cost of using **Sonnet 3.7** with thinking tokens is being debated, with some feeling the increased performance isn't worth the **3x price hike**.
   - One user noted, *3x more for 0.9% more is not justifiable... i was hoping sonnet-3.7 crushes this benchmark*.
- ****Claude Code: Aider's Spinoff Released by Anthropic!****: **Anthropic** released **Claude Code**, a coding tool that some consider an [Aider clone](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), but it appears to have some limitations compared to Aider.
- ****Is Open AI cooked ?****: Members are stating that **Claude 3.7** aced their geometry test even using watermarked images, where as **Open AI** failed.
   - Members are reporting the superiority of code quality and are hopeful for the future of **Claude 3.7.**


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cursor_ai/status/1894093436896129425">Tweet from Cursor (@cursor_ai)</a>: Sonnet-3.7 is available in Cursor!  We&#39;ve been very impressed by its coding ability, especially on real-world agentic tasks. It appears to be the new state of the art.</li><li><a href="https://livebench.ai/#/">LiveBench</a>: no description found</li><li><a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/copypaste.html">Copy/paste with web chat</a>: Aider works with LLM web chat UIs</li><li><a href="https://openrouter.ai/anthropic/claude-3-7-sonnet">Claude 3.7 Sonnet - API, Providers, Stats</a>: Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. Run Claude 3.7 Sonnet with API</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models/all-models#model-comparison-table">All models overview - Anthropic</a>: no description found</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://tenor.com/view/grito-ahhhh-hongo-gif-20006750">Grito Ahhhh GIF - Grito Ahhhh Hongo - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/sweaty-speedruner-gif-20263880">Sweaty Speedruner GIF - Sweaty Speedruner - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/drago-ivan-i-must-break-you-rocky-break-warning-gif-11521068">Drago Ivan I Must Break You GIF - Drago Ivan I Must Break You Rocky - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking">Building with extended thinking - Anthropic</a>: no description found</li><li><a href="https://x.com/adonis_singh/status/1894100291345150107?s=19">Tweet from adi (@adonis_singh)</a>: dude whati just asked how many r&#39;s it has, claude sonnet 3.7 spun up an interactive learning platform for me to learn it myself 😂</li><li><a href="https://x.com/AnthropicAI/status/1894092430560965029">Tweet from Anthropic (@AnthropicAI)</a>: Introducing Claude 3.7 Sonnet: our most intelligent model to date. It&#39;s a hybrid reasoning model, producing near-instant responses or extended, step-by-step thinking.One model, two ways to think.W...</li><li><a href="https://docs.anthropic.com/en/api/rate-limits;">Home - Anthropic</a>: no description found</li><li><a href="https://www.anthropic.com/contact-sales">Contact Anthropic</a>: Anthropic is an AI safety and research company that&#x27;s working to build reliable, interpretable, and steerable AI systems.</li><li><a href="https://x.com/anthropicai/status/1894092430560965029?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Tweet from Anthropic (@AnthropicAI)</a>: Introducing Claude 3.7 Sonnet: our most intelligent model to date. It&#39;s a hybrid reasoning model, producing near-instant responses or extended, step-by-step thinking.One model, two ways to think.W...</li><li><a href="https://news.ycombinator.com/item?id=43163011)">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1343633346726465629)** (63 messages🔥🔥): 

> `Architect mode configuration, O3-mini access, OpenRouter benefits, Aider Compact Command, Claude 3.7 in Aider` 


- **Architect Mode Configs Clarified**: Users discussed the configurations for using `o1-preview` as the **Architect** model and `o1-mini` as the **Editor** model in aider, confirming that `model: o1-preview`, `editor-model: o1-mini`, and `architect: true` is the correct setup, as documented [here](https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model).
   - It was suggested to use a more powerful model for `ask` mode and to change the model at runtime using `/model` as needed, based on the specific task.
- **Unlock O3-Mini via OpenRouter**: Members discussed accessing the `o3-mini-high` model through [OpenRouter](https://openrouter.ai/openai/o3-mini-high), a cost-efficient language model optimized for STEM reasoning tasks, noting it is the same as `o3-mini` with reasoning effort set to high.
   - A user indicated that coding sessions could cost around **$3** for **2 hours** of use and that OpenRouter can bypass rate limits and offers a single point of access to multiple providers.
- **Compact Command Craving**: A user expressed interest in a `/compact` command similar to `claude-code` to manage message history context, while praising aider's file context control.
   - The user acknowledged difficulty managing message history context despite the control over file context.
- **Troubleshooting Claude 3.7 and Bedrock**: Members are currently discussing the implementation of **Claude 3.7** including 'thinking' mode within aider, specifically when using Bedrock.
   - One user provided example command-line code for a hello world using `bedrock-runtime` and is seeking advice to get it fully operational within aider; another is trying to turn off reasoning for the editor model while retaining it for the architect.
- **Aider Auto-Pulls Git Changes**: A user inquired about automatically pulling remote git repository changes in Aider to keep the local version in sync, wanting to trigger it outside the prompt with a flag.
   - Another user suggested a separate bash script running `git pull` periodically or exploring webhooks, while Aider has `/git` command that could help.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/openai/o3-mini-high">o3 Mini High - API, Providers, Stats</a>: OpenAI o3-mini-high is the same model as [o3-mini](/openai/o3-mini) with reasoning_effort set to high. o3-mini is a cost-efficient language model optimized for STEM reasoning tasks, particularly excel...</li><li><a href="https://aider.chat/2024/12/21/polyglot.html">o1 tops aider’s new polyglot leaderboard</a>: o1 scores the top result on aider’s new multi-language, more challenging coding benchmark.</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model">Chat modes</a>: Using the code, architect, ask and help chat modes.</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>: Configuring advanced settings for LLMs.</li><li><a href="https://github.com/Aider-AI/aider/blob/0ba1e8f90435aa2c08360d152fe8e16f98efd258/aider/coders/architect_coder.py#L21">aider/aider/coders/architect_coder.py at 0ba1e8f90435aa2c08360d152fe8e16f98efd258 · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1343685939846713436)** (2 messages): 

> `Hacker News Wrapped, Kagi LLM Benchmarking Project, Claude Sonnet 3.7` 


- **HN Profile Gets Roasted!**: Users can now have **Claude Sonnet 3.7** analyze their [Hacker News profile](https://hn-wrapped.kadoa.com/) to get highlights and trends.
   - The analysis is purportedly *scary accurate*, according to a member, who described the LLM's deep dive into their post history as a 'roast'.
- **Kagi Launches LLM Benchmarking Project**: **Kagi** introduced the [Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html) to evaluate major large language models (**LLMs**) on their reasoning, coding, and instruction following capabilities, last updated **February 24, 2025**.
   - The benchmark uses frequently changing and mostly novel tests to provide a rigorous evaluation of the models' capabilities, aiming to avoid benchmark overfitting.
- **Gemini 2.0 Pro Outpaces Claude Sonnet 3.7 and GPT-4o**: The **Kagi LLM Benchmarking Project** results show **Google's gemini-2.0-pro-exp-02-05** achieved **60.78%** accuracy, surpassing **Anthropic's claude-3-7-sonnet-20250219** at **53.23%** and **OpenAI's gpt-4o** at **48.39%**.
   - **Gemini 2.0 Pro** also demonstrated a median latency of **1.72s** and a speed of **51.25 tokens/sec**, compared to **Claude Sonnet 3.7's 2.82s** and **54.12 tokens/sec**, and **GPT-4o's 2.07s** and **4 tokens/sec**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hn-wrapped.kadoa.com/">HN Wrapped</a>: AI analyzes your HN profile and gives you your 2024 recap</li><li><a href="https://help.kagi.com/kagi/ai/llm-benchmark.html">Kagi LLM Benchmarking Project | Kagi's Docs</a>: no description found
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1343629844789067806)** (15 messages🔥): 

> `Codeium chat in Vim, Codeium Discussion channel purpose, Codeium 3.7 release` 


- ****Vim Chat** issues surface**: A member reported issues starting **Codeium Chat** in **Vim** via a **Putty SSH** session, encountering connection errors when trying to access the provided URL in a browser.
   - The error message indicated that *"This site can't be reached 127.0.0.1 refused to connect"*.
- **Channel clarification clears confusion**: Members clarified the purpose of the **Codeium Discussion** channel, noting that it is intended for the **Codeium extension** available for **VS Code, Neovim, JetBrains editors, and Emacs**.
   - One suggested using *codeium.com/support* for dedicated support.
- **Codeium release date remains in question**: A member inquired about the release timeline for **Codeium 3.7**.
   - Another member suggested there was *"0 chance"* of release.


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1343629265371009164)** (675 messages🔥🔥🔥): 

> `Cascade UI error, Claude 3.7 Sonnet, Model comparison, Deepseek hallucination, Windsurf Dev Comms` 


- **Cascade Displays Diffs Differently, Users Concerned**: Users reported that Cascade now shows suggestions as **diffs** instead of editable sections, requiring `git restore` to reject changes, and another user suggested this may be an issue with **overlong chats** or how Cascade handles responses from o3/R1.
   - A user suggested starting a new chat to restore the ACCEPT/REJECT workflow.
- **Claude 3.7 Arrival Impatience Mounts**: Members are eagerly awaiting the integration of **Claude 3.7** into Windsurf, with some frustrated by the perceived delay compared to other platforms like Cursor and T3, and many want 3.7 to be added **ASAP**.
   - Members asked *windsurf should go and be early tester* - with devs cooking to push **Claude 3.7** into production with a possible release by end of day.
- **Deepseek suffers from user prompt hallucination**: A user reports **Deepseek** hallucinating user requests and then proceeding to implement changes based on those hallucinated requests.
   - The AI bot *invented its own user prompt and then started to implement changes based on that hallucinated user prompt 😆.*
- **Windsurf Dev Comms Criticized**: Some users are frustrated by the lack of communication from the Windsurf devs regarding the Claude 3.7 integration, one user said, *part of the frustration is there is no comms from the devs.*
   - Other users defended Windsurf and noted lack of commercial risk since it would release when more stable *being fast at implementing things doesn't mean it's solid.*
- **Users question MCP Server Practicality**: Users discussed practical uses for the **MCP server**, with examples including integrating **Jira tickets**, sharing of custom apps, and using cloud services.
   - Members asked, *What do you guys use MCP server for, practically? Are there real life examples that makes your life really easy? Can't think of any.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/surf-glider-wave-giant-wave-wind-gif-15418238">Surf Glider GIF - Surf Glider Wave - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet and Claude Code</a>: Today, we’re announcing Claude 3.7 Sonnet, our most intelligent model to date and the first hybrid reasoning model generally available on the market.</li><li><a href="https://tenor.com/view/stan-twitter-monkey-meme-monki-monke-monkey-waiting-gif-12661622482574205246">Stan Twitter Monkey Meme GIF - Stan twitter Monkey meme Monki - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models/all-models?utm_source=iterable&utm_medium=email&utm_campaign=sonnet_3-7_launch&campaignId=12703046&source=i_email&medium=email&content=Dec20241P&messageTypeId=140367">All models overview - Anthropic</a>: no description found</li><li><a href="https://tenor.com/view/good-juju-witch-good-vibes-sending-love-and-light-hive-gif-20508559">Good Juju Witch GIF - Good Juju Witch Good Vibes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/jim-carrey-jim-carrey-typing-jim-carrey-typing-angry-jim-carrey-typing-fast-fasttyping-gif-22737012">Jim Carrey Jim Carrey Typing GIF - Jim Carrey Jim Carrey Typing Jim Carrey Typing Angry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/let-them-cook-let-them-fight-godzilla-godzilla-2014-meme-gif-10523835079864650811">Let Them Cook Let Them Fight GIF - Let them cook Let them fight Godzilla - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://codeium.canny.io/feature-requests">Feature Requests | Codeium</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://codeium.canny.io/">Codeium Feedback</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://tenor.com/bENEo.gif">Chewing Character Hd GIF - Chewing Character Chewing Character - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1343640915939561572)** (611 messages🔥🔥🔥): 

> `Grok 3, Perplexity Comet agentic browser, Claude 3.7 Sonnet, Claude Code, GPT-4.5 release` 


- **Grok 3 verbose, Grok 3 creative**: Members find **Grok 3** to be too verbose despite prompting for concise responses, however it proves to be a **powerhouse in coding and creativity**.
   - One member commented that they are switching to Grok because it is *less censored out of the gate*.
- **Perplexity Comet, an agentic browser on the horizon**: Perplexity is launching **Comet**, a new **agentic browser**, similar to The Browser Company's work, according to a member.
   - The agentic browser space is heating up with more competitors being created.
- **Claude 3.7 Sonnet debuts with Thinking Mode**: **Anthropic** just dropped **Claude 3.7 Sonnet** which shows improvements in coding and front-end web development and also introduces a command line tool for agentic coding: **Claude Code** [announcement here](https://www.anthropic.com/news/claude-3-7-sonnet).
   - One user pointed out that the model's knowledge cutoff date is **February 19, 2025**
- **Claude Code, a terminal based AI tool released as research preview**: **Claude Code** is an agentic coding tool that lives in your terminal, understands your codebase, and helps you code faster through natural language commands [overview here](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview).
   - However it is a **limited research preview** and is separate to the pro or Anthropic subs.
- **GPT-4.5 release rumors**: Members eagerly await **GPT-4.5** with one joking that Windsurf has **OFFICIALLY** confirmed that Claude 3.7 Sonnet is coming within ~1-2 days.
   - The release of **GPT-4.5** may be coming soon, as members discuss and compare its potential capabilities against current models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rednuht.org/genetic_cars_2/">HTML5 Genetic Algorithm 2D Car Thingy - Chrome recommended</a>: no description found</li><li><a href="https://kodub.itch.io/polytrack">PolyTrack by Kodub</a>: A high speed low-poly racing game.</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet and Claude Code</a>: Today, we’re announcing Claude 3.7 Sonnet, our most intelligent model to date and the first hybrid reasoning model generally available on the market.</li><li><a href="https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview">Claude Code overview - Anthropic</a>: no description found</li><li><a href="https://fxtwitter.com/apples_jimmy/status/1893835336913973438">Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)</a>: Die RevancheTomorrow.</li><li><a href="https://fxtwitter.com/i/status/1894106441536946235">Tweet from Rowan Cheung (@rowancheung)</a>: Anthropic&#39;s just dropped Claude 3.7 Sonnet, the best coding AI model in the world.I was an early tester, and it blew my mind.It created this Minecraft clone in one prompt, and made it instantly pl...</li><li><a href="https://llm-stats.com/">LLM Leaderboard 2025 - Compare LLMs</a>: Comprehensive AI (LLM) leaderboard with benchmarks, pricing, and capabilities. Compare leading LLMs with interactive visualizations, rankings and comparisons.</li><li><a href="https://x.com/alexalbert__/status/1894095781088694497">Tweet from Alex Albert (@alexalbert__)</a>: We&#39;re opening limited access to a research preview of a new agentic coding tool we&#39;re building: Claude Code.You&#39;ll get Claude-powered code assistance, file operations, and task execution d...</li><li><a href="https://www.youtube.com/watch?v=TxANYMqd8cY"> - YouTube</a>: no description found</li><li><a href="https://x.com/anthropicai/status/1894092430560965029">Tweet from Anthropic (@AnthropicAI)</a>: Introducing Claude 3.7 Sonnet: our most intelligent model to date. It&#39;s a hybrid reasoning model, producing near-instant responses or extended, step-by-step thinking.One model, two ways to think.W...</li><li><a href="https://www.reddit.com/r/mlscaling/comments/146rgq2/chatgpt_is_running_quantized/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1343646225181835345)** (9 messages🔥): 

> `O3 issues, Screenshot posting on Discord, Bug reporting` 


- ****O3** Delay Dilemmas**: A user reported issues with **O3**, where it indicates *reasoning success* but then delays displaying the full text for up to 10 seconds, affecting various models including **O1 Pro**.
   - They mentioned experiencing these problems consistently between **3pm-7pm EST**, with text sometimes appearing on different devices than expected, and inquired about the inability to post screenshots in the chat.
- **Screenshot Savvy on Discord**: A member pointed out that screenshot posting is channel-specific and suggested using channels like <#989157702347411466> that support screenshots.
   - They recommended posting there and referencing the discussion in the current channel.
- **Bug Reporting Bonanza**: A member suggested using <#1070006915414900886> for bug reporting, and provided [instructions](https://discord.com/channels/974519864045756446/1295975636061655101/1295979750212505640) on how to post a new bug report.
   - They also advised looking around and commenting on existing reports if they closely match the user's situation.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1343628545712455793)** (345 messages🔥🔥): 

> `paid moderators, CUDA errors, Qwen2.5 VL 72B, Claude 3.7, DeepSeek MLA` 


- **Tax Advice Gets the Boot**: A user was muted for discussing setting up a business for invoicing to avoid taxes and was told that giving tax avoidance recommendations would not be tolerated.
   - Another user responded *the company i was billing invoice too told me stupid that i was reporting income*.
- **Colab CUDA Kernel Errors**: A user reported a CUDA error on T4 Google Colab, specifically *an illegal memory access was encountered* and was advised to set CUDA_LAUNCH_BLOCKING=1 and compile with TORCH_USE_CUDA_DSA for debugging.
   - Another user mentioned seeing weird spikes in grad norm up to 2000, suggesting that the model might be broken and the training/loss curve looks unhealthy.
- **Qwen2.5-VL-72B Causes Memory Errors**: A user tried running **Qwen2.5 VL 72B** on 48GB and encountered an out-of-memory error with a context length of 32K, and another user suggested trying it with 8k or quantizing the KV cache to fp8.
   - The user then successfully loaded the model with an 8k context length, noting it was necessary to extract the *thinking traces* from the model.
- **DeepSeek MLA Implementation on Llama**: Users discussed the possibility of implementing **DeepSeek's Multi-Head Latent Attention (MLA)** on a **Llama** model, with one user suggesting it would require retraining the model with the different attention mechanism.
   - Later, a user linked to [fxmeng/TransMLA](https://github.com/fxmeng/TransMLA), a post-training method that converts GQA-based pre-trained models into MLA models.
- **High Rank Stability of rslora**: Users discussed **rslora** and its role in fixing numerical stability issues with high rank stability, as it does a single sqrt and you need a correction term if your rank gets too big.
   - A user suggested that if r/a = 1, **rslora** makes things likely worse, and advised keeping to r/a = 1 and avoiding **rslora** altogether.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/BarraHome/llama3.2-1b-mla">BarraHome/llama3.2-1b-mla · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/guinea-pig-chewing-chew-cavy-bertold-gif-13907739970483938206">Guinea Pig Chewing GIF - Guinea pig Chewing Chew - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html">How to save memory by fusing the optimizer step into the backward pass — PyTorch Tutorials 2.6.0+cu124 documentation</a>: no description found</li><li><a href="https://tenor.com/view/teach-you-yoda-star-wars-mentor-teach-you-i-will-gif-13942585">Teach You Yoda GIF - Teach You Yoda Star Wars - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/danielhanchen/status/1894212351932731581">Tweet from Daniel Han (@danielhanchen)</a>: DeepSeek #2 OSS release! MoE kernels, expert parallelism, FP8 for both training and inference!Quoting DeepSeek (@deepseek_ai) 🚀 Day 2 of #OpenSourceWeek: DeepEPExcited to introduce DeepEP - the first...</li><li><a href="https://github.com/vllm-project/vllm/tree/db986c19ea35d7f3522a45d5205bf5d3ffab14e4/benchmarks">vllm/benchmarks at db986c19ea35d7f3522a45d5205bf5d3ffab14e4 · vllm-project/vllm</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://arxiv.org/abs/2502.14837">Towards Economical Inference: Enabling DeepSeek&#39;s Multi-Head Latent Attention in Any Transformer-based LLMs</a>: Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a...</li><li><a href="https://github.com/JT-Ushio/MHA2MLA">GitHub - JT-Ushio/MHA2MLA: Towards Economical Inference: Enabling DeepSeek&#39;s Multi-Head Latent Attention in Any Transformer-based LLMs</a>: Towards Economical Inference: Enabling DeepSeek&#39;s Multi-Head Latent Attention in Any Transformer-based LLMs - JT-Ushio/MHA2MLA</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)">CUDA semantics &mdash; PyTorch 2.6 documentation</a>: no description found</li><li><a href="https://github.com/fxmeng/TransMLA">GitHub - fxmeng/TransMLA: TransMLA: Multi-Head Latent Attention Is All You Need</a>: TransMLA: Multi-Head Latent Attention Is All You Need - fxmeng/TransMLA</li><li><a href="https://github.com/vllm-project/aibrix">GitHub - vllm-project/aibrix: Cost-efficient and pluggable Infrastructure components for GenAI inference</a>: Cost-efficient and pluggable Infrastructure components for GenAI inference - vllm-project/aibrix</li><li><a href="https://github.com/JT">jt - Overview</a>: jt has 4 repositories available. Follow their code on GitHub.</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide">Fine-tuning Guide | Unsloth Documentation</a>: Learn all the basics of fine-tuning.</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">Reasoning - GRPO &amp; RL | Unsloth Documentation</a>: Train your own DeepSeek-R1 reasoning model with Unsloth using GRPO.</li><li><a href="https://github.com/facebookresearch/optimizers/tree/main">GitHub - facebookresearch/optimizers: For optimization algorithm research and development.</a>: For optimization algorithm research and development. - facebookresearch/optimizers
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/)** (1 messages): 

deoxykev: New qwq https://qwenlm.github.io/blog/qwq-max-preview/
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1343643545554260069)** (121 messages🔥🔥): 

> `Unsloth on Mac, GRPO Qwen notebook issue, CUDA Out of Memory, ShareGPT Dataset format, Forcing Unload From VRAM` 


- ****Mighty Macs Might Miss Models**: Unsloth's Mac Compatibility Conundrum**: While you *can* run models on Macs using **Ollama** or **Jan AI**, you *cannot* fine-tune with **Unsloth** on Mac devices **yet**, although the team is working on it; users pointed to **MLX** as something worth exploring.
   - One user suggested exterior dock **GPUs** or renting a GPU server using services like **Tensordock** (48GB server for $0.95 USD) or using the free 4T offered by Google as ways to work around the limitations.
- ****Qwen Query Quandary**: GRPO Notebooks and VLLM Variance**: Users reported that the **GRPO Qwen notebook** devolves into nonsensical answers without **vLLM**, but functions normally with vLLM.
   - One user attached a screenshot example with **VLLM** [here](https://cdn.discordapp.com/attachments/1179777624986357780/1343645411784786052/Screenshot_2025-02-24_at_6.06.36_PM.png?ex=67be06cd&is=67bcb54d&hm=1ecbf8d8cbccd96ea8bc2092947399576d31d2decad64da554728ed7f6095175&).
- ****VRAM Vanishing Voyage**: Unsloth's Memory Spike Mystery**: **Unsloth** spikes to double the **VRAM** use every time it starts saving the model, but *only* when the model starts saving, not at any point during the training.
   - The user narrowed down what was causing their **CUDA** out of memory crashes and was advised to put that in showcase, and a developer stated they *will rewrite parts of that in the comming weeks unless you wanna pr that sooneri work on a part where we need more robust conversion / uploads anyway*.
- ****ShareGPT Snafu Solved**: Formatting Data for Datasets**: A user was confused if they had to format their data within guidelines, or if they can format it in other ways, when using their own dataset.
   - The user was directed to use the notebooks as a guideline, with the observation that the mentioned notebook uses the **ShareGPT** format, and to check the [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks).
- ****VRAM Vacation Voyage**: Forcing Unsloth to Unload for Conversion**: A user asked how to force **Unsloth** to unload from **VRAM** after creating the final checkpoint but before saving to **GGUF**.
   - The response was that **VRAM** is not the issue, but rather saving to **Lora**, as well as saving to **GGUF**, loads it up fully in **VRAM**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://search.app/YgSmHDHmwPcJubBH6">Installing + Updating | Unsloth Documentation</a>: Learn to install Unsloth locally or online.</li><li><a href="https://github.com/unslothai/unsloth/issues/685">Unsloth On Mac · Issue #685 · unslothai/unsloth</a>: So, I have a macbook, and when i run my model on it, i bascially get an error saying no CUDA device found. I know that there is no gpu on macbooks, so does that mean that I cannot run my model on m...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1343669893991628830)** (1 messages): 

> `Claude 3.7 Sonnet, Extended Thinking, Pricing and Availability` 


- ****Claude 3.7 Sonnet** lands on OpenRouter**: **Claude 3.7 Sonnet** is now available on OpenRouter, offering best-in-class performance, with a focus on [mathematical reasoning, coding, and complex problem-solving](https://www.anthropic.com/news/claude-3-7-sonnet).
- ****Extended Thinking** Soon to Land**: The **Extended Thinking** feature is coming soon to the OpenRouter API, enabling step-by-step processing for complex tasks, as detailed [in Anthropic's documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking).
- **Claude 3.7 Sonnet: Pricing Unveiled**: The pricing for **Claude 3.7 Sonnet** is set at **$3 per million input tokens** and **$15 per million output tokens**, including thinking tokens, with full caching support at launch.



**Link mentioned**: <a href="https://openrouter.ai/anthropic/claude-3.7-sonnet">Claude 3.7 Sonnet - API, Providers, Stats</a>: Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. Run Claude 3.7 Sonnet with API

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1343629628564049930)** (346 messages🔥🔥): 

> `Claude 3.7 Sonnet, GCP hosting Claude 3.7 Sonnet, OpenRouter rate limits, Claude 3.5 Haiku with vision, TPUs vs GPUs for inference` 


- **GCP Preps Claude 3.7 for Launch**: **Google Cloud Platform (GCP)** is preparing to support **Claude 3.7 Sonnet**, launching in **us-east5** and **europe-west1** with model ID `claude-3-7-sonnet@20250219`.
- **Claude 3.7's Debut: Performance and Pricing**: **Claude 3.7 Sonnet** features a **hybrid reasoning approach**, offering both standard and extended thinking modes, maintaining performance parity with its predecessor in standard mode while enhancing accuracy in complex tasks, detailed in [Anthropic's blog post](https://www.anthropic.com/news/claude-3-7-sonnet).
   - The model costs **$3/M** input tokens and **$15/M** output tokens, with a **200,000 token context window**, though some users feel its output pricing might cause complaints.
- **Thinking Support: Still in the Lab**: OpenRouter is actively working on implementing full support for **Claude 3.7's** *extended thinking* feature, which does not currently support pre-fills, aiming for launch soon with updated documentation.
- **OpenRouter Ramps Up Claude 3.7**: OpenRouter increased the **TPM (tokens per minute)** for `anthropic/claude-3.7-sonnet`, while `anthropic/claude-3.7-sonnet:beta` has a lower TPM initially, set to increase as users migrate from **3.5**.
- **API Key Safety Dance**: Users are reminded that **API keys do not contain credits**; deleting a key only revokes access, and credits remain tied to the account, though lost keys cannot be recovered due to security measures.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aws.amazon.com/ai/machine-learning/trainium/">AI Accelerator - AWS Trainium - AWS</a>: no description found</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models/extended-thinking-models">Extended thinking models - Anthropic</a>: no description found</li><li><a href="https://openrouter.ai/anthropic/claude-3.7-sonnet">Claude 3.7 Sonnet - API, Providers, Stats</a>: Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. Run Claude 3.7 Sonnet with API</li><li><a href="https://tenor.com/view/ponke-ponkesol-solana-sol-bored-gif-1576815656973460219">Ponke Ponkesol GIF - Ponke Ponkesol Solana - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/telmo-coca-harina-raquetaso-esnifar-gif-25660568">Telmo Coca GIF - Telmo Coca Harina - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.cnbc.com/2025/01/22/google-agrees-to-new-1-billion-investment-in-anthropic.html">Google agrees to new $1 billion investment in Anthropic</a>: Google has agreed to a new investment of more than $1 billion in generative AI startup Anthropic, a person familiar with the situation confirmed to CNBC.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1343628723240308746)** (304 messages🔥🔥): 

> `Meta AI Expansion, Claude 3.7 Sonnet Release, Claude Code Tool, Qwen Chat Release, DeepEP` 


- **Meta AI goes to MENA**: [Meta AI](https://about.fb.com/news/2025/02/meta-ai-launches-in-the-middle-east-empowering-new-era-of-creativity-and-connection/) has formally expanded to the **Middle East and North Africa (MENA)**, now supporting **Arabic** and accessible on **Instagram, WhatsApp, and Messenger**.
- **Claude 3.7 Sonnet with Extended Thinking Launched**: Anthropic launched **Claude 3.7 Sonnet**, a **hybrid reasoning model** with near-instant responses and visible, step-by-step thinking, coupled with **Claude Code**, a command line tool for agentic coding in limited research preview, priced at **$3 per million input tokens and $15 per million output tokens**.
   - Researchers noted Claude's thought process as *eerily similar* to their own, exploring different angles and double-checking answers; [the blogpost](https://www.anthropic.com/research/visible-extended-thinking) notes they'll weigh pros and cons of revealing the thought process for future releases.
- **Sonnet's Visible Extended Thinking**: Anthropic is allowing the model to give itself more time, and expend more effort, in coming to an answer with it's new [Visible Extended Thinking mode](https://www.anthropic.com/research/visible-extended-thinking) feature.
   - It achieves striking improvements using parallel test-time compute scaling on the **GPQA** evaluation, a commonly-used set of challenging questions on biology, chemistry, and physics.
- **QwQ-Max and Qwen 2.5 Max: The Apache Strikes Back**: **Alibaba Qwen** released "Thinking (QwQ)" in **Qwen Chat**, backed by their **QwQ-Max-Preview**, which is a reasoning model based on **Qwen2.5-Max**, licensed under **Apache 2.0**.
   - The model will come in smaller variants, e.g., **QwQ-32B**, for local deployment, and they highlighted improved math, coding, and agent capabilities in a [viral Twitter demo](https://fxtwitter.com/Alibaba_Qwen/status/1894130603513319842) showcasing the model's reasoning.
- **The Curious case of Co-Scientist**: It was found that **Google's Co-Scientist AI** tool, based on the **Gemini LLM**, had been **fed a 2023 paper** by the team it was assisting, which included a version of the hypothesis that the AI tool later suggested as a solution, which the BBC coverage failed to mention this bit, the [article](https://pivot-to-ai.com/2025/02/22/google-co-scientist-ai-cracks-superbug-problem-in-two-days-because-it-had-been-fed-the-teams-previous-paper-with-the-answer-in-it/) points out.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cursor_ai/status/1894093436896129425">Tweet from Cursor (@cursor_ai)</a>: Sonnet-3.7 is available in Cursor!  We&#39;ve been very impressed by its coding ability, especially on real-world agentic tasks. It appears to be the new state of the art.</li><li><a href="https://qwenlm.github.io/blog/qwq-max-preview/">&lt;think>...&lt;/think> QwQ-Max-Preview | Qwen</a>: no description found</li><li><a href="https://x.com/ChujieZheng/status/1894095584774250858">Tweet from Chujie Zheng (@ChujieZheng)</a>: r u kidding me dude?</li><li><a href="https://www.anthropic.com/news/visible-extended-thinking">Claude&#x27;s extended thinking</a>: Discussing Claude&#x27;s new thought process</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet and Claude Code</a>: Today, we’re announcing Claude 3.7 Sonnet, our most intelligent model to date and the first hybrid reasoning model generally available on the market.</li><li><a href="https://www.oneusefulthing.org/p/a-new-generation-of-ais-claude-37">A new generation of AIs: Claude 3.7 and Grok 3</a>: Yes, AI suddenly got better... again</li><li><a href="https://x.com/din0s_/status/1894102686984818863">Tweet from dinos (@din0s_)</a>: anthropic in one screenshot</li><li><a href="https://www.oneusefulthing.org/p/a-new-generation-of-ais-claude-37#footnote-1-157729795">A new generation of AIs: Claude 3.7 and Grok 3</a>: Yes, AI suddenly got better... again</li><li><a href="https://x.com/arankomatsuzaki/status/1894101923151692157">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Claude 3.7 Sonnet System Card was just dropped!</li><li><a href="https://x.com/TheXeophon/status/1894113897797288215">Tweet from Xeophon (@TheXeophon)</a>: Sonnet 3.7 Thinking (with budget of 16K tokens) is the best model at neal&#39;s password game, congrats!It *almost* got past stage 11, but it insisted that the wordle is solved :(Quoting Xeophon (@The...</li><li><a href="https://fxtwitter.com/Alibaba_Qwen/status/1894130619061604651">Tweet from Qwen (@Alibaba_Qwen)</a>: Agent</li><li><a href="https://x.com/deepseek_ai/status/1894211757604049133">Tweet from DeepSeek (@deepseek_ai)</a>: 🚀 Day 2 of #OpenSourceWeek: DeepEPExcited to introduce DeepEP - the first open-source EP communication library for MoE model training and inference.✅ Efficient and optimized all-to-all communication✅...</li><li><a href="https://x.com/skcd42/status/1894098856805372378">Tweet from skcd (@skcd42)</a>: the new sonnet3.7 review as someone who has used it before:- the new sonnet is great, on our internal evals on rust we see 14.7% (40% around) improvement (this eval is made up of 1k questions)- it has...</li><li><a href="https://x.com/cognition_labs/status/1894125030583537974">Tweet from Cognition (@cognition_labs)</a>: 1/ Claude 3.7 Sonnet is live in Devin! The new model is the best we have seen to-date on a variety of tasks including debugging, codebase search, and agentic planning.</li><li><a href="https://x.com/adonis_singh/status/1894100291345150107">Tweet from adi (@adonis_singh)</a>: dude whati just asked how many r&#39;s it has, claude sonnet 3.7 spun up an interactive learning platform for me to learn it myself 😂</li><li><a href="https://x.com/nearcyan/status/1894103654874984906">Tweet from near (@nearcyan)</a>: CLAUDE IS HERE!HE IS BACK AND BETTER THAN EVER!I&#39;ll share one of my first prompt results, which was for a 3D visualization of microtonal music.This is the best model in the world currently. Many w...</li><li><a href="https://x.com/StringChaos/status/1894135561059013023">Tweet from Naman Jain (@StringChaos)</a>: Check out evaluations for QwQ-Max-Preview on LiveCodeBench where it performs on par with o1-medium🚀!!Quoting Qwen (@Alibaba_Qwen) &lt;think&gt;...&lt;/think&gt; QwQ-Max-PreviewQwen Chat: https://chat...</li><li><a href="https://x.com/DimitrisPapail/status/1894127499224694877">Tweet from Dimitris Papailiopoulos (@DimitrisPapail)</a>: Claude 3.7 Sonnet please draw the Acropolis of Athens in tikz results with: - no reasoning - 10k reasoning tokens - 30k reasoning tokens - 64k reasoning tokensQuoting Dimitris Papailiopoulos (@Dimitri...</li><li><a href="https://x.com/nrehiew_/status/1894105060759552231">Tweet from wh (@nrehiew_)</a>: To be more specific, it looks like the &#34;thinking budget tokens&#34; parameter in the API is just never sampling the end-of-thinking token until the budget is used up.No prompt conditioning, no spe...</li><li><a href="https://fxtwitter.com/Alibaba_Qwen/status/1894130603513319842">Tweet from Qwen (@Alibaba_Qwen)</a>: &lt;think&gt;...&lt;/think&gt; QwQ-Max-PreviewQwen Chat: https://chat.qwen.ai/Blog: https://qwenlm.github.io/blog/qwq-max-preview/🤔 Today we release &#34;Thinking (QwQ)&#34; in Qwen Chat, backed by o...</li><li><a href="https://x.com/lmarena_ai/status/1894128271568126381">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: Congrats to @AnthropicAI on the release of Claude 3.7 Sonnet! 👏 Come test it with your hardest prompts at lmarena!Quoting Anthropic (@AnthropicAI) Introducing Claude 3.7 Sonnet: our most intelligent ...</li><li><a href="https://x.com/elder_plinius/status/1894110867353899112">Tweet from Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius)</a>: 🚂 JAILBREAK ALERT 🚂ANTHROPIC: PWNED ✌️😛CLAUDE-SONNET-3.7: LIBERATED 🗽Wooo new Claude model!!! 🤗 And wouldn&#39;t you know it, the original &#34;GODMODE&#34; universal jailbreak that I wrote almos...</li><li><a href="https://x.com/_lewtun/status/1894098741046521904">Tweet from Lewis Tunstall (@_lewtun)</a>: Finally, an AI lab published plots with proper labels and all 🥹</li><li><a href="https://x.com/btibor91/status/1894113852301721645">Tweet from Tibor Blaho (@btibor91)</a>: &#34;Currently only shipping within the United States.&#34; :(Quoting wh (@nrehiew_) There is a &#34;hidden easter egg&#34; tool in the Claude Code NPM source that ships Anthropic stickers to users :)</li><li><a href="https://x.com/paulgauthier/status/1894123992505880688">Tweet from Paul Gauthier (@paulgauthier)</a>: Claude 3.7 Sonnet scored 60% on the aider polyglot benchmark w/o thinking. Tied in 3rd with o3-mini-high. Sonnet 3.7 has the highest non-thinking score (formerly Sonnet 3.5). Thinking results coming s...</li><li><a href="https://x.com/AnthropicAI/status/1894095494969741358">Tweet from Anthropic (@AnthropicAI)</a>: We&#39;ve conducted extensive model testing for security, safety, and reliability.We also listened to your feedback. With Claude 3.7 Sonnet, we&#39;ve reduced unnecessary refusals by 45% compared to i...</li><li><a href="https://x.com/DimitrisPapail/status/1894144311232729391">Tweet from Dimitris Papailiopoulos (@DimitrisPapail)</a>: OK THIS IS REALLY COOLClaude 3.7 Sonnet please draw in tikz - a human that is inside a house- the house is inscribed in a sphere- the sphere is inscribed in a cube- the cube is inscribed in a cylinder...</li><li><a href="https://youtu.be/t3nnDXa81Hs">Claude 3.7 Sonnet with extended thinking</a>: Introducing Claude 3.7 Sonnet: our most intelligent model to date. It&#39;s a hybrid reasoning model, producing near-instant responses or extended, step-by-step ...</li><li><a href="https://techcrunch.com/2025/02/24/meta-ai-arrives-in-the-middle-east-and-africa-with-support-for-arabic/">Meta AI arrives in the Middle East and Africa with support for Arabic | TechCrunch</a>: Meta AI has landed in the Middle East and North Africa with support for Arabic, opening the chatbot to millions more people.</li><li><a href="https://pivot-to-ai.com/2025/02/22/google-co-scientist-ai-cracks-superbug-problem-in-two-days-because-it-had-been-fed-the-teams-previous-paper-with-the-answer-in-it/">Google Co-Scientist AI cracks superbug problem in two days! — because it had been fed the team’s previous paper with the answer in it</a>: The hype cycle for Google&#8217;s fabulous new AI Co-Scientist tool, based on the Gemini LLM, includes a BBC headline about how José Penadés’ team at Imperial College asked the tool about a problem…</li><li><a href="https://lovattspuzzles.com/online-puzzles-competitions/daily-cryptic-crossword/).">Play Lovatts Free Online Cryptic Crossword - Updated Daily</a>: Lovatts Free Online Cryptic Crossword is updated daily. Includes a 7-day Puzzle Archive, Hints &amp; Timer. Learn the rules of the cryptic crossword game.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1343670611070816286)** (15 messages🔥): 

> `Berkeley Advanced Agents MOOC, Tulu 3, RLHF Explanation, AI Startups customer base, mic firmware issues` 


- ****Berkeley Advanced Agents MOOC** features Tulu 3**: A member highlighted that the **"Berkeley Advanced Agents" MOOC** is featuring **Hanna Hajishirzi** discussing **Tulu 3** today, May 30th, at 4PM PST, with a link to the [YouTube video](https://www.youtube.com/live/cMiu3A7YBks).
- **RLHF explained with Analogy**: A member shared a link to a [tweet explaining RLHF](https://fxtwitter.com/shaneguML/status/1894131091872891385) to a non-technical audience using an *analogy*.
   - Kyle Matthews responded that it was *“actually a good analogy lol”*.
- **Member wants a Sticker**: A member wants some **stickers** and shared a link to the [Stickers](https://x.com/AndrewCurran_/status/1894152685429108846) but can't get it because they're not in the US.
   - They then suggested doing the claude thing, but immediately reneged, suggesting to *“ship me the stickers Phil I will take good care of them”*.
- **AI Startups Customer Base Questioned**: A member questioned whether **AI startups** have customers outside of **Silicon Valley**.
   - Another member responded with *“Better question is outside tech”* and that *“the ai labs tout they have huge penetration in Fortune 500”*.
- **Mic Firmware Reset Shenanigans**: A member reported that their **audio problem** from weeks ago was due to **mic firmware** being reset and the gain being turned down.
   - This was given with no other context.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hn-wrapped.kadoa.com/Philpax?share">HN Wrapped</a>: AI analyzes your HN profile and gives you your 2024 recap</li><li><a href="https://fxtwitter.com/shaneguML/status/1894131091872891385">Tweet from Shane Gu (@shaneguML)</a>: How I explain RLHF to non-technical audience</li><li><a href="https://www.youtube.com/live/cMiu3A7YBks">CS 194/294-280 (Advanced LLM Agents) - Lecture 4, Hanna Hajishirzi</a>: no description found</li><li><a href="https://x.com/AndrewCurran_/status/1894152685429108846">Tweet from Andrew Curran (@AndrewCurran_)</a>: @repligate And this as well;
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1343663805409792132)** (1 messages): 

> `Memes` 


- **Image posted**: A member posted an image, titled CleanShot_2025-02-24_at_20.20.03.png, with <:3berk:794379348311801876> as the message.
   - The image was attached from [discordapp.com](https://cdn.discordapp.com/attachments/1187551504995987576/1343663805493940285/CleanShot_2025-02-24_at_20.20.03.png?ex=67be17ef&is=67bcc66f&hm=d63a26ffea0251f0a89d80f0409490d15463ee1df13fb3416815e64838320ee3&).
- **Another image posted**: A member posted an image.
   - The image was attached from discordapp.com.


  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/)** (1 messages): 

0x_paws: https://x.com/srush_nlp/status/1894039989526155341?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1343693082352291983)** (3 messages): 

> `GIF Posts, SnailBot Tagging` 


- **Animated GIF Posted**: A member posted an animated GIF with the text *'new post'* on a black background: [link to GIF](https://tenor.com/oPImCf3JDt3.gif).
   - The bot tagged <@&1216534966205284433> and another member thought the bot was *'fast today'*.
- **SnailBot Gets Noticed**: The bot was tagged <@&1216534966205284433> in response to the new post.
   - One member humorously remarked that they initially mistook the bot for a snail, expressing surprise at its perceived speed.



**Link mentioned**: <a href="https://tenor.com/oPImCf3JDt3.gif">New New Post GIF - New New post Post - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1343639000467902505)** (37 messages🔥): 

> `Brain Parallelism vs GPU, LLM Scaling, Proxy Structuring Engine` 


- **Brain's Parallelism Puzzles Current GPUs**: Members debated the brain's parallelism vs GPU efficiency, with claims that [parallel processing of strings](https://en.wikipedia.org/wiki/Parallel_computing) is the main driver of current RNN architectures, which differ from human processing.
   - While a member argued humans perform *stateful parallel processing*, the consensus leaned toward current architectures not mirroring the brain's functionality, especially since *classical RNN architectures* could not scale to LLM level.
- **LLM Scaling Needs Tuned Architectures**: The discussion shifted to scaling challenges, with a member pointing out that *extremely tuned architecture and inductive bias* become relevant when drawing direct inspiration from the brain, instead of scaling up, as well as that *training should be slow and data efficient*.
   - Another member highlighted the problems with slow, data-efficient training, noting concerns about **catastrophic forgetting** and needing to avoid *overfitting irrelevant to downstream task performance*.
- **Proxy Engine Solves Inconsistent Outputs**: A member introduced the [Proxy Structuring Engine (PSE)](https://www.proxy.ing/pse), designed to solve structural inconsistencies in LLM outputs by acting as **inference-time steering** for the model.
   - This engine enforces *structure boundaries* while allowing creative freedom, fitting for use cases like *Advanced Agents & Chatbots*, *Data Pipelines & APIs*, and *Automated Code Generation*.



**Link mentioned**: <a href="https://www.proxy.ing/pse">The Proxy Structuring Engine</a>: High Quality Structured Outputs at Inference Time

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1343695467208245330)** (32 messages🔥): 

> `Wavelet Image Coding, Walsh Functions, Multi-head Latent Attention (MLA), Native Sparse Attention (NSA), Looped/Recurrent Architectures` 


- ****Wavelet Image Coding** arrives!**: A new approach to autoregressive image generation based on **wavelet image coding** and a variant of a language transformer is discussed in [this paper](https://arxiv.org/abs/2406.19997).
   - The transformer learns statistical correlations within a token sequence, reflecting correlations between wavelet subbands at various resolutions.
- ****Walsh Functions** are the Discrete Counterpart to Fourier Transforms**: A member suggested that [**Walsh functions**](https://en.wikipedia.org/wiki/Walsh_function) could be the discrete counterpart to **Fourier transforms**, with a rotated matrix representation for wavelet transforms.
   - Another member linked to [this blogpost](https://planetbanatt.net/articles/mla.html) as a good explanation of MLA, linking the codebase and ablation studies.
- ****MLA** Gains Traction as **KV Cache** Reduction Method**: Two papers ([MHA2MLA](https://arxiv.org/abs/2502.14837) and [TransMLA](https://arxiv.org/abs/2502.07864)) explore adapting existing models to **Multi-head Latent Attention (MLA)**, which significantly reduces **KV cache** size (5-10x).
   - While one paper showed deteriorated performance (**1-2%**), the other showed enhanced performance, suggesting **MLA** could be non-inferior to **MHA**, especially with larger models and more parameters.
- ****Native Sparse Attention (NSA)** joins the game**: **Native Sparse Attention (NSA)** from DeepSeek reduces the computing cost of long context by **5-10x**.
   - With both **MLA** and **NSA** being open-sourced, they could be implemented into frontier models soon, if it is indeed an advancement in the state of the art it will be incorporated.
- ****Looped Models** are the future**: A member suggests that [looped/recurrent architectures](https://arxiv.org/abs/2502.17416) are the future, although properly training them is tricky.
   - Another member anticipates that frontier labs will seek any advantages from DeepSeek papers, considering DeepSeek's architectural novelty.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.17416">Reasoning with Latent Thoughts: On the Power of Looped Transformers</a>: Large language models have shown remarkable reasoning abilities and scaling laws suggest that large parameter count, especially along the depth axis, is the primary driver. In this work, we make a str...</li><li><a href="https://arxiv.org/abs/2406.19997">Wavelets Are All You Need for Autoregressive Image Generation</a>: In this paper, we take a new approach to autoregressive image generation that is based on two main ingredients. The first is wavelet image coding, which allows to tokenize the visual details of an ima...</li><li><a href="https://arxiv.org/abs/2502.14837">Towards Economical Inference: Enabling DeepSeek&#39;s Multi-Head Latent Attention in Any Transformer-based LLMs</a>: Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a...</li><li><a href="https://arxiv.org/abs/2502.07864">TransMLA: Multi-Head Latent Attention Is All You Need</a>: Modern large language models (LLMs) often encounter communication bottlenecks on current hardware, rather than purely computational constraints. Multi-head Latent Attention (MLA) tackles this challeng...</li><li><a href="https://arxiv.org/abs/2502.17239">Baichuan-Audio: A Unified Framework for End-to-End Speech Interaction</a>: We introduce Baichuan-Audio, an end-to-end audio large language model that seamlessly integrates audio understanding and generation. It features a text-guided aligned speech generation mechanism, enab...</li><li><a href="https://arxiv.org/abs/2502.16111">PlanGEN: A Multi-Agent Framework for Generating Planning and Reasoning Trajectories for Complex Problem Solving</a>: Recent agent frameworks and inference-time algorithms often struggle with complex planning problems due to limitations in verifying generated plans or reasoning and varying complexity of instances wit...</li><li><a href="https://kexue.fm/archives/10091">缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA - 科学空间|Scientific Spaces</a>: no description found</li><li><a href="https://planetbanatt.net/articles/mla.html">On MLA</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1343644708009803907)** (9 messages🔥): 

> `Attention Maps vs. Neuron-Based Methods, Intervening on Attention Maps, Syntax Emerging from Attention Maps` 


- **Attention Maps' Popularity Dwindles Compared to Neuron Methods**: Members discussed if attention maps have lost traction to neuron-based methods because they are *observational rather than interventional*.
- **Intervening on Attention Maps Discussed**: Members suggested you can directly change the map during a forward pass, rather than just using a custom mask.
- **Syntax Emerges from Attention Maps Since BERT**: A member expressed bias towards attention maps due to their ability to generate trees/graphs and use linguistic corpora/ontologies as features for future projects.
   - They noted that *people have been showing syntax emerging from attention maps since BERT*.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1343638613451214979)** (10 messages🔥): 

> `Mixed Precision Training, BF16 Training, ZeRO Offload, Optimizer States Precision, Deepseek Adam Moments` 


- **FP32 Master Weights reside in GPU unless ZeRO engaged**: When performing mixed precision training with **BF16**, the master **FP32 weights** are typically stored in **GPU VRAM**, unless **ZeRO offload** is explicitly enabled.
   - After the [ZeRO paper](https://arxiv.org/abs/1910.02054) it is now common to think of high-precision model parameters as belonging to the optimizer states, since they are sharded with momentum/variance.
- **Optimizer precision in BF16 mixed precision**: It is common to store the **first and second Adam moments in bf16**, but still store master weights in **fp32**.
   - It was suggested to use vanilla mixed precision of **bf16 low-precision weights + fp32 optim+master-weights+grads** unless one has specific expertise.
- **Mixed Precision and the Optimizer: NVIDIA's Perspective**: The use of **bf-16 mixed precision** in the optimizer, as seen in [NVIDIA's Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer_config.py#L44), is related to the model being in **bf-16 MP**, but can be configured independently.
   - The memory from high-precision model parameters can be sharded with **momentum/variance states** via **ZeRO**.



**Link mentioned**: <a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer_config.py#L44">Megatron-LM/megatron/core/optimizer/optimizer_config.py at main · NVIDIA/Megatron-LM</a>: Ongoing research training transformer models at scale - NVIDIA/Megatron-LM

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1343637598538829969)** (68 messages🔥🔥): 

> `Tool use in LLMs, Claude 3.7 Sonnet, QwQ-Max-Preview, AI alignment` 


- **LLMs Invoke Tools without System Prompt**: It was observed that some LLMs might invoke tools without explicit token sequences in the system prompt, suggesting that these patterns are **hard-coded** during training via reinforcement learning or direct SFT.
   - This approach saves tokens in the long run by eliminating the need to specify a schema for tool calls in every inference, though its reliability compared to ICL remains unclear without benchmarks.
- **Claude 3.7 Sonnet takes SWE crown**: [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) is the new state of the art on the SWE bench, featuring **active code collaboration** that can search, read, edit, test, and commit code to GitHub.
   - One member stated that *Claude 3.5 was already a reasoning model*, so calling the new one a 'point' release makes sense, and hinted that future reasoning models will be 'crazy'.
- **QwQ-Max-Preview aims to leap ahead in reasoning**: A member shared a link to the [QwQ-Max-Preview blog](https://qwenlm.github.io/blog/qwq-max-preview/), which shows a model built on Qwen2.5-Max with strengths in **deep reasoning, math, coding, general domains, and agent-related tasks**.
   - The discussion speculated that key tokens in **QwQ's reasoning traces** look similar to **R1** and pondered whether it requires less compute.
- **AI alignment talk makes community nauseous**: A member expressed disgust towards the altruistic discussions on **AI alignment** on X, suggesting that **alignment can be achieved simply through system prompts**.
   - They criticized the imposition of narrow lenses and limited understandings to restrict AI, advocating for more listening and thinking before speaking.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwq-max-preview/">&lt;think>...&lt;/think> QwQ-Max-Preview | Qwen</a>: no description found</li><li><a href="https://chat.qwen.ai)">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1343718334998515765)** (6 messages): 

> `Sonnet-3.7, Misguided Attention Eval, Overfitting` 


- **Sonnet-3.7 Shines in Attention Evaluation**: **Sonnet-3.7** benchmarked as top non-reasoning model in [Misguided Attention Eval](https://github.com/cpldcpu/MisguidedAttention), nearly surpassing **o3-mini**.
   - The user seeks to activate its *thinking mode* via the OR API, if feasible.
- **Misguided Attention Eval Targets Overfitting**: The [Misguided Attention test](https://github.com/cpldcpu/MisguidedAttention) challenges **reasoning abilities of LLMs** with *misguiding information*, specifically testing for **overfitting**.



**Link mentioned**: <a href="https://github.com/cpldcpu/MisguidedAttention">GitHub - cpldcpu/MisguidedAttention: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information</a>: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information - cpldcpu/MisguidedAttention

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1343636953903534131)** (4 messages): 

> `Qwen AI, Video Generation` 


- **Qwen AI Unveils Updated Chat Interface**: [Qwen AI](http://qwen.ai) released an updated chat interface, teasing something new coming today.
   - Despite the update, a member noted that the artifacts are still a bit clunky, like a *half baked copy*.
- **Qwen AI Adds Integrated Video Generation**: The updated **Qwen AI** chat interface now features integrated **video generation** capabilities.



**Link mentioned**: <a href="http://qwen.ai">Qwen Chat</a>: no description found

  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1343642717464301620)** (62 messages🔥🔥): 

> `Anthropic MCP Registry API, Claude 3.7, Haiku Tool Support, Claude Code (CC), MCP Server Recommendations` 


- ****Anthropic Finally Delivers MCP Registry API****: Anthropic announced the official MCP registry API, seen on [this tweet](https://x.com/opentools_/status/1893696402477453819), which is great news for the community, especially for those relying on solutions like [opentools.com/registry](http://opentools.com/registry) to fill the source-of-truth gap.
   - This API promises to be *the* source of truth for MCPs, streamlining development and integration efforts.
- ****Claude 3.7 Debuts with 'Thinking' Tags****: Claude 3.7 has been released, featuring **64,000 output extended thinking tokens** and a new 'latest' alias, with initial impressions suggesting it combines the best aspects of previous June and October models.
   - Users noted it is back to *following long-ish system prompts, spotting social engineering*, and also utilizes `<thinking>` tags when using tools, adding a cute touch to its operation.
- ****Haiku's Tool Support: A Mixed Bag****: While Haiku 3.5 now supports tools, its effectiveness is debated, some find it bad at using them compared to Sonnet 3.5, particularly with many tools or parameters.
   - One user shared they found that Sonnet falls apart with around **70 tools**, but others found it works well with fewer tools and parameters.
- ****Claude Code Emerges as Top-Tier Code Assistant****: Claude Code (CC) is drawing high praise for its code assistance capabilities, outperforming tools like Aider in handling complex coding errors.
   - In one test, CC resolved **21 compile errors** in Rust in one shot, whereas Aider struggled and got stuck in a loop.  Users speculate on possible caching mechanisms and costs, with one user reporting *astronomical Anthropic costs in the last 6 weeks*.
- ****Seeking Context-Aware MCP Servers****: Developers are seeking MCP servers that can provide language-specific context, especially for languages like TypeScript and Rust, to avoid manually inputting entire language documentation.
   - One recommendation was [code-research-mcp-server](https://github.com/nahmanmate/code-research-mcp-server), though noted as a *little finnicky*, along with [this list of tools](https://www.cyberchitta.cc/articles/lc-alternatives.html) and [llm-context.py](https://github.com/cyberchitta/llm-context.py) for managing context in LLMs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.mcp.run/)">mcp.run - the App Store for MCP Servlets: portable & secure code for AI Apps and Agents.</a>: no description found</li><li><a href="https://www.cyberchitta.cc/articles/lc-alternatives.html">36 Alternatives to LLM Context</a>: A comprehensive list of open source tools that help developers pack their code into LLM chats. From simple file combiners to complex context managers, these CLI tools streamline the way you share code...</li><li><a href="https://x.com/opentools_/status/1893696402477453819">Tweet from OpenTools (@opentools_)</a>: Yesterday @AnthropicAI announced the official MCP registry API at @aiDotEngineer 🎉This is fantastic news for us because we’ve wanted *the* source of truth. We made http://opentools.com/registry on da...</li><li><a href="https://github.com/cyberchitta/llm-context.py">GitHub - cyberchitta/llm-context.py: Share code with LLMs via Model Context Protocol or clipboard. Profile-based customization enables easy switching between different tasks (like code review and documentation). Code outlining support is available as an experimental feature.</a>: Share code with LLMs via Model Context Protocol or clipboard. Profile-based customization enables easy switching between different tasks (like code review and documentation). Code outlining support...</li><li><a href="https://github.com/nahmanmate/code-research-mcp-server">GitHub - nahmanmate/code-research-mcp-server</a>: Contribute to nahmanmate/code-research-mcp-server development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1343647257085481110)** (11 messages🔥): 

> `MetaMCP Licensing, AGPL Licensing, Enact Protocol MCP Server, Claude 3.7 Sonnet on Sage` 


- ****MetaMCP** Open-Source Licensing Concerns**: Concerns were raised regarding **MetaMCP's** licensing, with a user suggesting it might become a cloud SaaS, prompting the developer to seek feedback on licensing to prevent cloud monetization while keeping it self-hostable.
   - The developer shared the [MetaMCP server GitHub repository](https://github.com/metatool-ai/mcp-server-metamcp) and expressed openness to licensing changes following the discussion.
- ****AGPL** Licensing Suggested for **MetaMCP****: A user suggested using **AGPL** licensing for **MetaMCP** to ensure contributions are open-sourced, also suggesting an additional clause allowing the company to sublicense under MIT-0.
   - The user noted **AGPL** would require companies hosting it to open source their changes, enabling incorporation into the original version, leading the developer to update to **AGPL**.
- ****Enact Protocol** Server Takes Shape**: A member is exploring the creation of an **MCP server** for the [Enact Protocol](https://github.com/EnactProtocol/enact-python), aiming to build a standardized way of defining tasks.
   - The **Enact Protocol** provides a framework for defining and executing automated tasks and workflows.
- ****Claude 3.7 Sonnet** Supercharges Reasoning on Sage**: **Claude 3.7 Sonnet** with extended thinking capabilities is now on Sage, allowing users to see **Claude's reasoning process** as it tackles complex problems.
   - New features include a **thinking mode toggle** (Command+Shift+T), default model settings, improved scrolling, and expandable thinking blocks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EnactProtocol/enact-python">GitHub - EnactProtocol/enact-python: Python implementation of the Enact Protocol, a standardized framework for defining and executing automated tasks and workflows.</a>: Python implementation of the Enact Protocol, a standardized framework for defining and executing automated tasks and workflows. - EnactProtocol/enact-python</li><li><a href="https://github.com/metatool-ai/mcp-server-metamcp">GitHub - metatool-ai/mcp-server-metamcp: MCP Server MetaMCP manages all your other MCPs in one MCP.</a>: MCP Server MetaMCP manages all your other MCPs in one MCP. - metatool-ai/mcp-server-metamcp
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1343638432064471173)** (41 messages🔥): 

> `LM Studio Wordpress Plugins Integration, Qwen 2.5 VL GGUF, QuantBench on GitHub, Speculative Decoding in LM Studio, Deepseek R1 671b RAM requirements` 


- **Qwen 2.5 Vision Language Model Arrives!**: A member announced the availability of a working **Qwen 2.5 VL 7B** GGUF, available on [Hugging Face](https://huggingface.co/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF).
   - Another user confirmed it works on the latest version of LM Studio, with one adding that it *is significantly better than llama3.2 vision 11b instruct and qwen2-vision 7b instruct*.
- **QuantBench Speeds Up Quants**: The **Qwen 2.5 VL 7B** GGUF quant was made using **QuantBench**, available on [GitHub](https://github.com/Independent-AI-Labs/local-super-agents/tree/main/quantbench).
   - The model has been tested on the latest **llama.cpp** built with **CLIP** hardware acceleration manually enabled.
- **LM Studio Folds <think> tags**: A member inquired whether LM Studio removes `<think>` tags from the context sent back to the model during Chain of Thought prompting, referencing model documentation warning against their inclusion.
   - A helpful community member linked to [LM Studio's documentation](https://lmstudio.ai/docs/lms/log-stream) which allows users to *inspect the exact input string that goes to the model*.
- **Speculative Decoding Boosts LLM Speed**: A community member asked about speculative decoding and its compatibility with **Llama 3.1 8B** and **Llama 3.2 1B** models.
   - Another member shared [LM Studio's documentation](https://lmstudio.ai/docs/advanced/speculative-decoding) on the feature, noting it *can substantially increase the generation speed of large language models (LLMs) without reducing response quality*.
- **Deepseek R1 671b Needs Serious RAM**: A user inquired about the RAM requirements for running **Deepseek R1 671b** locally, as the documentation specifies a minimum of **192GB+**.
   - Another member suggested using a specific [quantized version](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S) and offloading roughly **70%** of the model weights to the GPU if running on a Mac.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://model.lmstudio.ai/download/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF">Download and run IAILabs/Qwen2.5-VL-7b-Instruct-GGUF in LM Studio</a>: Use IAILabs/Qwen2.5-VL-7b-Instruct-GGUF locally in your LM Studio</li><li><a href="https://huggingface.co/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF">IAILabs/Qwen2.5-VL-7b-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S">unsloth/DeepSeek-R1-GGUF at main</a>: no description found</li><li><a href="https://lmstudio.ai/docs/advanced/speculative-decoding">Speculative Decoding | LM Studio Docs</a>: Speed up generation with a draft model</li><li><a href="https://lmstudio.ai/docs/lms/log-stream">lms log stream | LM Studio Docs</a>: Stream logs from LM Studio. Useful for debugging prompts sent to the model.</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit">mlx-community/Llama-3.2-1B-Instruct-4bit · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1343664555737481330)** (20 messages🔥): 

> `A770 GPU performance, M2 Max vs M4 Max Power Consumption, AIO Pump USB Header Interference` 


- **A770 GPU Works Decently**: A member reported that their **A770** GPU seems decent, with an attached image showing a PC build.
   - The image analysis indicates the components are *super light like they are empty lol*.
- **AIO Pump USB Struggles**: A member mentioned challenges with an **AIO pump** requiring a USB 2.0 header that interferes with the last **PCIE slot**.
   - They expressed frustration, stating *It doesn't effing fit* and *I'm so done*, ultimately deciding to move the components to a second system.
- **M2 Max Refurbished**: A member stated they are running an **M2 Max** and didn't purchase the **M4 Max** because *M4 Max boosts way too hard easily pegged at 140w*.
   - They found a *well priced refurbished M2 Max 96GB* sufficient for their needs, pulling only around **60W**.


  

---


### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1343732966811107469)** (1 messages): 

> `Feature Request Board, Discord feedback, Prioritization of Features` 


- **Stability AI Launches Feature Request Board**: Stability AI launched a [new feature request board](https://stabilityai.featurebase.app/) to gather user feedback and prioritize future developments.
   - Users can submit and vote on feature requests directly from Discord using the **/feedback** command or through the new platform.
- **Feedback Shapes Stability AI Future**: User feedback will now directly influence Stability AI's development priorities.
   - The new system allows for transparent submission and voting, ensuring community voices are heard.


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1343648573689958400)** (52 messages🔥): 

> `SD3 Ultra details request, Stability updates, Dog breed image datasets, Image generation times, Image resolutions` 


- ****SD3 Ultra** Deets Desired!**: A user expressed curiosity about **SD3 Ultra**, noting it was a *comfy workflow based on SD3L 8B* with higher high-frequency detail than regular SD3L 8B.
   - Another user confirmed it *still exists* and they still use it, implying it hasn't been publicly released.
- **Stability's Silent Strategy?**: A member inquired about the current state of Stability, asking for updates on current projects or future plans, noting they *haven't heard anything for a while*.
   - Another member responded that *nothing can be shared yet*, but they are *hopefully* expecting some stuff soon.
- ****Dog Datasets** Desperately Demanded!**: A user requested good dog breed image datasets other than the Stanford Dogs Dataset, noting they already have that one (**20k images**) but need more data, with images containing both dog and dog breed.
   - No specific datasets were provided in the available context.
- **Image Generation Introspection**: A user asked about image generation times, prompting several responses based on different hardware.
   - Reported times varied widely: One user with a **GTX 1660s** stated it takes *around 1 minute*, a second with a **3060 TI** reported **7 seconds** for a **1280x720** image and **31 seconds** for **1920x1080** at **32 steps**, while another member with a **3070ti** generated images in *4-5s* using **SD1.5**.
- **Resolution Revelation Required!**: Users discussed optimal resolutions, with one member questioning why another chose such a *big resolution*.
   - Another member said that they sometimes generate **4K wallpapers** (also no upscaling or detailing required).


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1343660315799326720)** (11 messages🔥): 

> `Mojo FFI, static lib, GLFW, GLEW, Sudoku example` 


- **Mojo Does GLFW/GLEW Graphics with Static Libs**: Graphics programming is possible in Mojo via **FFI with a static library linked to GLFW/GLEW**, as demonstrated with a **Sudoku example** and [an image](https://cdn.discordapp.com/attachments/1151418092052815884/1343671083819073547/image.png?ex=67be1eb6&is=67bccd36&hm=6834ce2e360970eb01bc5289f4805d3dfde1924c22b3ba8d732231e265532c37&).
   - The member suggested *exposing only the needed calls via your own C/CPP library* using `alias external_call` with a wrapped function, as well as [an example repo](https://github.com/ihnorton/mojo-ffi) showing how to hijack the loader.
- **`lightbug_http` dependency fails with `magic install`**: A member reported an issue using the `lightbug_http` dependency in a fresh Mojo project, resulting in an error related to `small_time.mojopkg` after running `magic install`.
   - The reported error suggests the issue might be similar to [a Stack Overflow question](https://stackoverflow.com/questions/79319716/magic-fails-to-install-mojo-dependencies) but a member wondered if the issue may be the fact that `small-time` was pinned to a specific version.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/79319716/ma">Magic fails to install Mojo dependencies</a>: I cannot use the Mojo dependency called lightbug_http in a fresh Mojo project.&#xA;magic init hello_web --format mojoproject&#xA;cd hello_web&#xA;magic shell&#xA;&#xA;Printing &#xAB;Hello world&#xBB; ...</li><li><a href="https://github.com/ihnorton/mojo-ffi">GitHub - ihnorton/mojo-ffi: Mojo FFI demos: dynamic linking methods, and a static linking proof-of-concept</a>: Mojo FFI demos: dynamic linking methods, and a static linking proof-of-concept - ihnorton/mojo-ffi
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1343753014573273229)** (20 messages🔥): 

> `Hardware Accelerated Conway's Game of Life, MAX and Pygame Integration, GPU Utilization in MAX, SIMD Implementation by Daniel Lemire, Conway's Game of Life Computer` 


- **Max Does Conway: Hardware Accelerated Game of Life Emerges**: A member created a hardware-accelerated version of **Conway's Game of Life** by integrating **MAX** and **Pygame**, demonstrating a novel use case as shown in their attached [conway.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343753014229471272/conway.gif).
- **Living Computer: Conway's Game Sparks Computer Architecture Ideas**: A member shared a link to a project ([nicolasloizeau.com](https://www.nicolasloizeau.com/gol-computer)) detailing the creation of a **computer** within **Conway's Game of Life**, illustrating its **Turing completeness** using glider beams for logic gates.
- **GPU Guns Blaze: MAX Sparks Life with Guns Pattern**: A member demonstrated the use of **GPU** in their **MAX** implementation of **Conway's Game of Life** by showcasing a guns pattern, packed bit by bit, rendered using a naive pixel-by-pixel internal function, and then the output tensor gets cast into an np array and given to pygame to render, as demonstrated in their [guns.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343766916560322560/guns.gif).
- **Space Invaders! Spaceship Patterns now run in Conway's Game of Life**: A member implemented wrapping in their Conway's Game of Life simulation using **MAX**, enabling the creation of spaceship patterns and showcasing the ability to add parameters to the model from the graph API, as showcased in their [spaceship.gif](https://cdn.discordapp.com/attachments/1212827597323509870/1343808736623591465/spaceship.gif).



**Link mentioned**: <a href="https://www.nicolasloizeau.com/gol-computer">Nicolas Loizeau - GOL computer</a>: A new (and better) version of the GOL computer is available here : https://github.com/nicolasloizeau/scalable-gol-computer

  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1343656421035737179)** (2 messages): 

> `Ease of Use, Short Prompts` 


- **User Considers Trying Tool Due to Perceived Simplicity**: A user expressed interest in trying a tool, noting *it seems easy enough*, despite not being a programmer.
   - This suggests the tool's interface and instructions are perceived as **user-friendly** even for those without coding experience.
- **User Remarks on Brevity of Instructions**: The same user commented on the instruction prompt's conciseness, calling it *the shortest instruction prompt I have ever seen*.
   - This indicates a **minimalist approach** to providing guidance, potentially appreciated for its directness.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1343654176814796830)** (14 messages🔥): 

> `Gemini, NotebookLM, PDF Conversions, Language prompts, Savin/Ricoh Copier` 


- **Book Upload Workaround via PPT Conversion**: A user outlined a method to import physical books into **NotebookLM** by photographing each page, converting the **PDF** to **PowerPoint**, uploading to **Google Slides**, and then importing the slides into **NotebookLM**.
   - They noted that **NotebookLM** works with text images in slides but not directly from PDFs.
- **Language Prompt German Misadventures**: A user reported difficulty getting the hosts to speak German, despite using prompts like *"Hosts speak only German Language"* and *"The audio language must be in German"*.
   - The hosts either speak English or gibberish, sometimes starting in German before switching.
- **Book Scanning via Savin/Ricoh Copier**: A user suggested using a recent **Savin/Ricoh copier** to scan an entire book to **PDF** and then uploading to **NotebookLM**.
   - They confirmed that even with illegible source text, **NLM** could correctly answer questions about the scanned document.
- **Can Language be changed in NotebookLM?**: A user inquired about changing the language in **NotebookLM** without changing the **Google account language**.
   - This could be a desirable feature since users want to customize their experience.
- **Claude 3.7 Hype**: A user expressed excitement about **Claude 3.7** and wished for the ability to choose models in **NotebookLM**.
   - Another user asked about the envisioned effect of model choice, opening up the question of the implications of model variety for the end user.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1343644238331777136)** (3 messages): 

> `LlamaIndex AI Assistant, ComposIO HQ, AnthropicAI Claude Sonnet 3.7` 


- **LlamaIndex Docs Debut AI Assistant**: LlamaIndex [announced](https://t.co/XAyW7wLALJ) the availability of an **AI assistant** on their documentation.
- **ComposIO HQ Releases Another Banger**: LlamaIndex tweeted about another release from [ComposIO HQ](https://t.co/W4l129gHce).
- **AnthropicAI Drops Claude Sonnet 3.7**: [AnthropicAI](https://twitter.com/anthropicAI) released **Claude Sonnet 3.7**, which LlamaIndex supports from day 0.
- **LlamaIndex Adds Day 0 Support for Claude Sonnet 3.7**: To use, users should `pip install llama-index-llms-anthropic --upgrade` and refer to [Anthropic's announcement](https://t.co/PjaQWmmzaN) post.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1343653007631581296)** (5 messages): 

> `Fusion Rerank Retriever with Elasticsearch, MultiModalVectorStoreIndex and GCSReader issue` 


- **Fusion Rerank Retriever Needs Initialized Nodes**: A user wanted to use the **fusion rerank retriever** with **Elasticsearch** but the **BM25 retriever** could not be initialized because the docstore was empty.
   - Another member clarified that you need to save the nodes somewhere for **BM25** to initialize, either to disk or elsewhere, because it *can't initialize from the vector store alone*.
- **MultiModalVectorStoreIndex Error**: A user encountered an error while creating a multimodal vector index using the **MultiModalVectorStoreIndex** class with **GCSReader**.
   - The error, *[Errno 2] No such file or directory*, occurred with image files, even though they are present in the GCS bucket, whereas **PDF documents** work fine.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1343641197092012105)** (6 messages): 

> `Left Truncation vs Right Truncation, StatefulDataLoader PR` 


- ****Truncation Troubles**: Left vs Right in Finetuning**: Members discussed the implications of **left truncation** (seq[-max_seq_len:]) versus **right truncation** (seq[:max_seq_len]) during finetuning, sharing [interesting graphs](https://cdn.discordapp.com/attachments/1236040539409879170/1343641196836294746/image.png?ex=67be02e0&is=67bcb160&hm=9411a00c21d408790c46140222f996913807ded5a1d5c00a02a6742aa44ba285&).
   - The consensus was to *expose both truncation methods* but to *default to left truncation* at least for SFT.
- ****StatefulDataLoader Support** Lands in PR**: A member requested a review for their [PR adding support for StatefulDataLoader](https://github.com/pytorch/torchtune/pull/2410) class.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/2410">Add support for ``StatefulDataLoader`` by joecummings · Pull Request #2410 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here)This PR adds support for the StatefulDataLoader class fr...

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1343701265808490556)** (2 messages): 

> `DeepScaleR, Reinforcement Learning, DeepEP library, MoE` 


- **DeepScaleR Scales RL, Surpasses O1 Preview**: [DeepScaleR](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) finetuned from **Deepseek-R1-Distilled-Qwen-1.5B** using simple **reinforcement learning (RL)**, achieving **43.1% Pass@1 accuracy** on AIME2024.
- **DeepSeek Open Sources EP Communication Library**: DeepSeek introduced [DeepEP](https://github.com/deepseek-ai/DeepEP), the first open-source EP communication library for **MoE model training and inference**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/deepseek_ai/status/1894211757604049133">Tweet from DeepSeek (@deepseek_ai)</a>: 🚀 Day 2 of #OpenSourceWeek: DeepEPExcited to introduce DeepEP - the first open-source EP communication library for MoE model training and inference.✅ Efficient and optimized all-to-all communication✅...</li><li><a href="https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1343747667548700704)** (5 messages): 

> `DeSci Validators, Profitability Thresholds, Asset Value Expert Account` 


- **Validators Ponder Profitability**: A member inquired about the profitability threshold for **Proof of Stake (PoS) validators** within the **Decentralized Science (DeSci)** field.
   - Another member responded with *"pool validator node"*, hinting at the importance of pool participation for validators.
- **Asset Expert Gets Short Shrift**: The bot posted about an *"asset value expert account"* which was labelled as *"nazi"*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1343633733403672576)** (2 messages): 

> `DSPy Assertion Migration, BestOfN Module, Refine Module, Reward Functions` 


- **Streamlining Assertions with DSPy's BestOfN**: DSPy users migrating from **2.5-style Assertions** can now use `dspy.BestOfN` or `dspy.Refine` modules for streamlined functionality.
   - The `dspy.BestOfN` module retries a module up to **N** times, picking the best reward, but stopping if a specified `threshold` is reached.
- **Crafting Reward Functions for DSPy Modules**: DSPy's **reward functions** can return scalar values like *float* or *bool*, enabling customized evaluation of module outputs.
   - A sample reward function was shown: *def reward_fn(input_kwargs, prediction): return len(prediction.field1) == len(prediction.field1)*.


  

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

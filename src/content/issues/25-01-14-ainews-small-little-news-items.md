---
id: 36f8e198-8bb3-4914-abea-5eabbf2a49ca
title: small little news items
date: '2025-01-15T02:19:30.206234Z'
original_slug: ainews-small-little-news-items
description: >-
  **Ollama** enhanced its models by integrating **Cohere's R7B**, optimized for
  **RAG** and **tool use tasks**, and released **Ollama v0.5.5** with quality
  updates and a new engine. **Together AI** launched the **Llama 3.3 70B
  multimodal model** with improved reasoning and math capabilities, while
  **OpenBMB** introduced the **MiniCPM-o 2.6**, outperforming **GPT-4V** on
  visual tasks. Insights into **Process Reward Models (PRM)** were shared to
  boost **LLM reasoning**, alongside **Qwen2.5-Math-PRM** models excelling in
  mathematical reasoning. **LangChain** released a beta for **ChatGPT Tasks**
  enabling scheduling of reminders and summaries, and introduced open-source
  **ambient agents** for email assistance. **OpenAI** rolled out **Tasks** for
  scheduling actions in **ChatGPT** for Plus, Pro, and Teams users. AI software
  engineering is rapidly advancing, predicted to match human capabilities within
  18 months. Research on **LLM scaling laws** highlights power law relationships
  and plateauing improvements, while **GANs** are experiencing a revival.
companies:
  - ollama
  - cohere
  - togethercompute
  - openbmb
  - qwen
  - langchain
  - openai
models:
  - r7b
  - llama-3-70b
  - minicpm-o-2.6
  - gpt-4v
  - qwen2.5-math-prm
topics:
  - rag
  - tool-use-tasks
  - quality-of-life
  - new-engine
  - multimodality
  - improved-reasoning
  - math-capabilities
  - process-reward-models
  - llm-reasoning
  - mathematical-reasoning
  - beta-release
  - task-scheduling
  - ambient-agents
  - email-assistants
  - ai-software-engineering
  - codebase-analysis
  - test-case-generation
  - security-infrastructure
  - llm-scaling-laws
  - power-law
  - plateauing-improvements
  - gans-revival
people: []
---


<!-- buttondown-editor-mode: plaintext -->**patience is all you need.**

> AI News for 1/13/2025-1/14/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **32** Discords (**219** channels, and **2161** messages) for you. Estimated reading time saved (at 200wpm): **256 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

[ChatGPT Tasks launched](https://techcrunch.com/2025/01/14/chatgpt-now-lets-you-schedule-reminders-and-recurring-tasks/). [Cursor raised a B](https://x.com/sarahdingwang/status/1879279307119608142). [Sakana announced a beautiful improvement over LoRAs with only minor performance improvement](https://x.com/hardmaru/status/1879331049383334187). Hailuo dropped a [giant 456B MoE](https://www.reddit.com/r/LocalLLaMA/comments/1i1a88y/minimaxtext01_a_powerful_new_moe_language_model/) similar to Deepseek v3.

Nothing we'd give title story feature to, but nice incremental progress.


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet, best of 4 runs.

**Model Releases and Updates**

- **Ollama Model Enhancements**: [@ollama](https://twitter.com/ollama/status/1879216139542434055) announced the inclusion of Cohere's **R7B**, the smallest model in their **Command R series**, optimized for **RAG** and **tool use tasks**. Additionally, [@ollama](https://twitter.com/ollama/status/1879212554435911978) released **Ollama v0.5.5**, featuring multiple **quality of life updates** and a transition to a **new engine**. The upcoming **2025 Ollama meetup** in San Francisco was highlighted by [@ollama](https://twitter.com/ollama/status/1878950123625214081), attracting significant interest with **31,592 impressions**. 

- **Together AI and OpenBMB Models**: [@togethercompute](https://twitter.com/togethercompute/status/1879231968434684254) introduced **Llama 3.3 70B**, a **multimodal model** available for free on **Together AI**, boasting **improved reasoning** and **math capabilities**. Concurrently, [@OpenBMB](https://twitter.com/_philschmid/status/1879163439559389307) released the **MiniCPM-o 2.6**, an **8B parameter multimodal model** that **outperforms GPT-4V** on visual tasks.

- **Process Reward Models and Qwen Developments**: [@_philschmid](https://twitter.com/theTuringPost/status/1879101437663154194) shared insights into **Process Reward Models (PRM)**, emphasizing their role in enhancing **LLM reasoning**. The **Qwen team** also unveiled their **Qwen2.5-Math-PRM** models, demonstrating superior performance in **mathematical reasoning**.

- **LangChain and Codestral Updates**: [@LangChainAI](https://twitter.com/LangChainAI/status/1879235381545386309) released a **beta version of tasks**, allowing **ChatGPT** to handle **future tasks** like reminders and summaries. **Codestral 25.01** by [@dchaplot](https://twitter.com/dchaplot/status/1878952042498334944) achieved **joint #1 on LMSys Copilot Arena**, showcasing significant **performance improvements** over previous versions.

**AI Features and Tools**

- **OpenAI Task Rollout**: [@OpenAI](https://twitter.com/OpenAI/status/1879267276291203329) announced the rollout of **Tasks**, a feature enabling users to **schedule actions** for **ChatGPT** such as **weekly news briefings** and **personalized workouts**. This feature is currently in **beta for Plus, Pro, and Teams users** and will eventually be available to all **ChatGPT accounts**.

- **Ambient Agents and Email Assistants**: [@LangChainAI](https://twitter.com/LangChainAI/status/1879218070008570213) introduced an **open-source email assistant agent**, part of their new **"ambient agents"** paradigm. These agents are **always active**, handling tasks like **email triage** and **drafting responses**, enhancing **productivity** without traditional **UX interfaces**.

- **AI Software Engineering Advancements**: [@bindureddy](https://twitter.com/bindureddy/status/1879017155423080482) discussed the rapid maturation of **AI software engineers**, highlighting their capabilities in **codebase analysis**, **test case generation**, and **security infrastructure**, predicting that AI will **match SWE capabilities** within the next **18 months**.

**AI Research and Papers**

- **LLM Scaling Laws**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1878929929611448588) delved into **LLM scaling laws**, explaining the **power law relationships** between **compute**, **model size**, and **dataset size**. The research emphasizes that while **test loss** decreases with scaling, the **improvements plateau**, challenging the notion of **exponential AI advancements**.

- **GANs Revival**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1879111514210402681) reported on the revival of **GANs** through the paper "**The GAN Is Dead; Long Live the GAN! A Modern GAN Baseline**," highlighting the **R3GAN architecture** and its **superior performance** over some **diffusion models** on benchmarks like **FFHQ** and **CIFAR-10**.

- **Multimodal RAG and VideoRAG**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1878932305177453037) introduced **VideoRAG**, an extension of **multimodal RAG** that retrieves **videos in real-time**, utilizing both **visual** and **textual data** to enhance **response accuracy**.

- **Tensor Product Attention**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1879091791753843064) presented the "**Tensor Product Attention (TPA)**" mechanism, which **reduces inference-time cache size** by **10x** and **outperforms previous attention methods** like **MHA** and **GQA** in **performance benchmarks**.

**AI Community and Events**

- **Ollama Meetup and Community Engagement**: [@ollama](https://twitter.com/ollama/status/1878950123625214081) promoted the **2025 Ollama meetup** in **San Francisco**, fostering **community engagement** among **AI enthusiasts**. Additionally, [@gdb](https://twitter.com/gdb/status/1879059072135414236) and others encouraged **community participation** through **joining initiatives** and **hiring announcements**.

- **LangChain AI Meetup**: [@LangChainAI](https://twitter.com/LangChainAI/status/1879235381545386309) organized an **evening meetup** in **San Francisco** featuring a **fireside chat** with **industry leaders** like **@hwchase17** and **Bihan Jiang**, focusing on **deploying production-ready AI agents**.

- **Hiring Announcements**: Multiple tweets, including those from [@WaveFormsAI](https://twitter.com/alex_conneau/status/1879240315342880826) and [@LTIatCMU](https://twitter.com/gneubig/status/1879261395646398872), shared **job openings** for **software engineers** and **research positions** in areas like **multimodal LLMs**, **full-stack development**, and **AI safety**.

**AI Industry News and Policy**

- **AI Policy and Economic Impact**: [@gdb](https://twitter.com/gdb/status/1879059072135414236) released an **Economic Blueprint** outlining **policy proposals** for optimizing **AI benefits**, enhancing **national security**, and driving **economic growth** in the **U.S.**. Concurrently, [@NandoDF](https://twitter.com/NandoDF/status/1878949902383985105) advocated for the **removal of non-compete clauses** in the **UK** to **boost AI competitiveness**.

- **AI Workforce Transformation**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1878955184459518297) highlighted the emergence of **AI Engineers and Consultants** as **top jobs on the rise** due to **AI's transformative impact** across industries, underscoring the importance of **gaining expertise** in this field.

- **China vs. US AI Competition**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1879251029084225966) and others discussed the **intensifying AI competition** between **China and the U.S.**, emphasizing **geopolitical implications** and the **race for AI dominance**.

- **Data Center Revenue Projections**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1878944624477376920) projected **data center revenue for FY2026** at **$236 billion**, marking a **28% increase** over **market consensus**, indicating the **growing infrastructure investments** in AI.

**Memes/Humor**

- **Coding and Daily Reminders**: [@hkproj](https://twitter.com/hkproj/status/1878954137171415346) shared a **daily reminder** to **eat veggies** and **code triton kernels**, blending **health tips** with **coding humor**.

- **AI and Personal Life Jokes**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1878915601244336450) humorously remarked on **a model's consciousness**, joking about the need for **better epistemology** in **examining AI capabilities**.

- **Developer Memes**: [@nearcyan](https://twitter.com/nearcyan/status/1878915854269890911) posted a **"two space" meme**, resonating with **developers' in-jokes** about **coding standards**.

- **Humorous Takes on AI Agents**: [@bindureddy](https://twitter.com/bindureddy/status/1878983646532825320) joked about **AI agents** taking over work tasks, pondering if **working would become obsolete**.

- **General Tech Humor**: [@saranormous](https://twitter.com/saranormous/status/1879170235019952575) quipped about **reading readiness** for having kids, intertwining **life advice** with **humorous skepticism**.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Qwen's Math Process Reward Models and Innovations**

- **Qwen released a 72B and a 7B process reward models (PRM) on their recent math models** ([Score: 145, Comments: 16](https://reddit.com/r/LocalLLaMA/comments/1i0ysa7/qwen_released_a_72b_and_a_7b_process_reward/)): **Qwen** has released two new **Process Reward Models (PRM)**, the **Qwen2.5-Math-PRM-7B** and **Qwen2.5-Math-PRM-72B**, designed to enhance mathematical reasoning in Large Language Models (LLMs) by identifying and correcting intermediate errors. These models demonstrate strong performance in **Best-of-N (BoN)** evaluations and excel in error identification on **ProcessBench**, as detailed in their paper titled *The Lessons of Developing Process Reward Models in Mathematical Reasoning* ([arXiv:2501.07301](https://arxiv.org/abs/2501.07301)).
  - **Qwen2.5-Math-PRM-72B** is primarily useful for academic purposes and training other models by providing feedback on reasoning quality and intermediate steps, rather than for typical text generation tasks. **Zealousideal-Cut590** emphasizes the need for **Process Reward Models (PRMs)** in non-mathematical domains like programming, legal, and medical tasks to optimize test time compute.
  - **-p-e-w-** discusses the increasing challenge of keeping up with the rapid release of new models, predicting that even unlimited internet connections may soon be insufficient. **Useful44723** suggests that **Hugging Face** should offer torrent links as an alternative download method to manage the high volume of data.
  - The rapid pace of model releases is highlighted, with **-p-e-w-** noting the occurrence of multiple substantial new releases per week, leading to potential saturation of download queues. **Caffeine_Monster** and **Threatening-Silence-** comment on the adequacy of current internet speeds and the potential for future limitations.


- **[MiniCPM-o 2.6: An 8B size, GPT-4o level Omni Model runs on device](https://x.com/OpenBMB/status/1879074895113621907)** ([Score: 158, Comments: 29](https://reddit.com/r/LocalLLaMA/comments/1i11961/minicpmo_26_an_8b_size_gpt4o_level_omni_model/)): **MiniCPM-o 2.6** is an **8 billion parameter model** claimed to achieve **GPT-4o level** performance. It is designed to run on local devices, enhancing accessibility and usability for various applications.
  - Discussions highlight skepticism about **MiniCPM-o 2.6**'s claim of achieving **GPT-4o level performance**, with users arguing that despite its accessibility and local running capability, it does not match GPT-4 in benchmarks or capabilities. **AaronFeng47** and **Aaaaaaaaaeeeee** express doubts about its performance, suggesting it is not at par with **GPT-4o** and noting the technical challenges of running it on-device, requiring a device with ≥12GB memory.
  - Users debate the validity of claims that smaller models can surpass larger ones like GPT-4, with **MoffKalast** and **Radiant_Dog1937** discussing how smaller models, such as **Gemma 2 9B** and **Gemini 1.5 Flash 8B**, rank high on the **Hugging Face leaderboard** but may not match GPT-4's comprehensive capabilities. They argue that while small models perform well in specific tasks, they cannot match the knowledge and application abilities of much larger models due to physical limitations in parameter capacity.
  - **Many_SuchCases** shares links to **MiniCPM-o 2.6** on **Hugging Face**, raising questions about its inference engine compatibility, while discussions also touch on the **MMMU score** of **MiniCPM-o 2.6**, which is 50.4 compared to 69.2 for GPT-4o, indicating a significant


**Theme 2. MiniMax-Text-01: MoE and Long Context Capabilities**

- **MiniMax-Text-01 - A powerful new MoE language model with 456B total parameters (45.9 billion activated)** ([Score: 93, Comments: 48](https://reddit.com/r/LocalLLaMA/comments/1i1a88y/minimaxtext01_a_powerful_new_moe_language_model/)): **MiniMax-Text-01** is a new **Mixture-of-Experts (MoE) language model** with **456 billion total parameters**, of which **45.9 billion are activated per token**. It uses a hybrid architecture combining **Lightning Attention**, **Softmax Attention**, and MoE, with advanced parallel strategies like **Linear Attention Sequence Parallelism Plus (LASP+)** and **Expert Tensor Parallel (ETP)**, allowing it to handle up to **4 million tokens during inference**.
  - **Hardware Requirements and Running Locally**: Running **MiniMax-Text-01** requires a significant amount of RAM, with suggestions ranging from **96GB for basic operations** to **384/470GB for more practical use**. Despite its size, the **Mixture-of-Experts (MoE)** architecture may allow for more manageable local execution by offloading active experts to a GPU, akin to **deepseek v3**.
  - **Licensing and Accessibility**: The model's restrictive license has raised concerns, particularly its limitations on using outputs to improve other models and its distribution requirements. Despite these restrictions, it remains open for commercial use, though some users question enforceability, drawing parallels with **Apache 2.0** for military applications.
  - **Performance and Capabilities**: The model's ability to handle **up to 4 million tokens** is highlighted as a significant achievement in open-source long-context processing. Its hybrid of **linear and softmax attention** layers, alongside advanced parallel strategies, is noted for potentially reducing context requirements and enhancing retrieval and extrapolation capabilities compared to models relying solely on softmax attention.


**Theme 3. Inspiration from LLMs Driving New Open Source Initiatives**

- **Today I start my very own org 100% devoted to open-source - and it's all thanks to LLMs** ([Score: 141, Comments: 44](https://reddit.com/r/LocalLLaMA/comments/1i148es/today_i_start_my_very_own_org_100_devoted_to/)): The post author, with a background in **biology**, has founded a new organization fully dedicated to **open-source** projects, attributing this achievement to the influence of **Large Language Models (LLMs)** and the supportive community at **r/LocalLlama**. They express gratitude to the community and highlight the importance of the open-source ecosystem in enabling their transition from biology to this new venture.
  - **Bootstrapping and Financial Challenges**: Several commenters, including **KnightCodin** and **mark-lord**, discussed the challenges and benefits of bootstrapping a business. **Mark-lord** emphasized reducing living costs to manage finances effectively without investor pressure, sharing a personal journey of overcoming imposter syndrome and financial hurdles.
  - **Community Support and Encouragement**: The community expressed strong support and encouragement for the author's venture, with users like **Silent-Wolverine-421** and **NowThatHappened** offering congratulations. The sentiment of "This is the way" was echoed by multiple commenters, highlighting a shared ethos of pursuing independent, open-source projects.
  - **Advice and Tools**: **Mark-lord** shared practical advice for those transitioning into AI, recommending tools like **Claude 3.5** for various tasks and suggesting using **Cursor** for unlimited requests. They invited further discussion and networking through direct messages, reflecting a willingness to support others in similar journeys.


- **Why are they releasing open source models for free?** ([Score: 283, Comments: 166](https://reddit.com/r/LocalLLaMA/comments/1i11hre/why_are_they_releasing_open_source_models_for_free/)): **Open source AI models** are being released for free despite the costs involved because they can drive **community collaboration** and **accelerate innovation**. The incentive for companies or developers to release these models includes gaining **reputation**, encouraging **widespread adoption**, and potentially **stimulating improvements** that can benefit the original creators.
  - Discussions highlight that **open source AI models** help companies like **Meta** and **Google** secure market dominance by making their models widely used standards, which reduces costs and attracts talent. The strategy is compared to **Google's Android** and **Microsoft's GitHub**, emphasizing the long-term benefits of community engagement and mindshare over direct revenue from the models themselves.
  - Several comments argue that releasing these models for free can disrupt competitors and create barriers to entry for new players. This can be seen as a **"scorched earth"** strategy, where the goal is to saturate the market with free resources, making it difficult for others to monetize similar offerings, as discussed in the context of **Meta's LLaMA** and **GitHub Copilot**.
  - Commenters also note that the **"open source"** label is sometimes misleading, as many models are only **open weights** without full retraining capabilities. This partial openness allows companies to benefit from community feedback and innovation, while still maintaining control over their proprietary technologies and strategic advantages.


**Theme 4. RTX Titan Ada 48GB: Unveiling New GPU Potentials**

- **RTX Titan Ada 48GB Prototype** ([Score: 52, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1i0rdx1/rtx_titan_ada_48gb_prototype/)): The **RTX Titan Ada 48GB** is speculated to be more appealing than the **5090**, with a potential price of **$3k**. It features all **144 SMs** enabled, double the performance in mixed precision training, and possibly the **transformer engine** from the **L40**, unlike the **4090**. Despite slower memory bandwidth, it offers **48GB** memory, **300W TDP**, and a **GFLOPS/W** of **1223.88**, making it efficient for setups with multiple cards. [More details here](https://videocardz.com/newz/alleged-nvidia-rtx-titan-ada-surfaces-with-18432-cuda-cores-and-48gb-gddr6-memory-alongside-gtx-2080-ti-prototype).
  - **Memory Bandwidth Concerns**: The discussion highlights concerns about the **bandwidth drop** being "brutal" at less than half, but some users argue that **904GB/s** does not feel slow, emphasizing the importance of memory bandwidth relative to memory capacity used per token.
  - **Pricing and Market Appeal**: There is skepticism about the card's pricing strategy, with a suggestion that selling at a loss for **$500** would be more appealing. However, the **digits** at **273GB/s** are seen as a drawback for potential buyers who prioritize prompt processing.
  - **Prototype and Features**: The card is identified as an old prototype, resembling an **L40** with **ECC disabled** and using **GDDR6** and **PCIe 4.0**. It was rumored a year ago alongside the **4090 Ti**, and recent GPU-Z screenshots lend some credibility to its existence.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. AGI: Marketing Hype or Genuine Innovation?**

- **Are we actually far from AGI and this is all marketing?** ([Score: 161, Comments: 260](https://reddit.com/r/OpenAI/comments/1i0v95g/are_we_actually_far_from_agi_and_this_is_all/)): The post questions whether **AGI** is currently achievable or if claims about it are merely marketing tactics. The author suggests that despite the transformative impact of **Transformers**, a genuine leap toward AGI might require a breakthrough in **neuroscience** or **cognitive science** to develop a new architecture that complements existing technologies.
  - **AGI Definition and Skepticism**: Many commenters, including **insightful_monkey** and **Deltanightingale**, debate the definition of **AGI**, suggesting that while current AI models like **o3** show advanced capabilities in specific domains, they lack the general problem-solving and autonomous reasoning skills that characterize true AGI. The consensus is that the current state of AI is far from achieving AGI, with some like **PatrickOBTC** highlighting that much of the AGI discourse is marketing-driven.
  - **Technological and Financial Constraints**: Discussions emphasize the technological and financial hurdles in achieving AGI, with **vertigo235** and **Deltanightingale** noting the high costs and slow speeds associated with existing AI models. **JonnyRocks** points out that **OpenAI's** definition of AGI is tied to business goals, such as reaching **$100 billion** in revenue, rather than true technological milestones, indicating a financial motive behind AGI claims.
  - **Progress and Future Outlook**: While some, like **ZillionBucks**, remain optimistic about the future of AGI, many others are skeptical, with **TheInfiniteUniverse_** and **Economy-Bid-7005** suggesting that while models like **o3** perform well in specific areas, they lack crucial elements such as **recursive learning**. The release of **o3-mini** and **o3-Full** is anticipated, but


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12

**Theme 1. New AI Models: Codestral, MiniMax-01, and DeepSeek V3**

- [**Codestral Model Debuts with 256k Context**](https://mistral.ai/news/codestral-2501): The new **Codestral** model is now free on the **Mistral API**, boasting a massive **256k context window** and described by users as "*stupid fast and good*". It's expected to significantly speed up extensive code generation tasks.
- [**MiniMax-01 Launches Open-Source Models with 4M Tokens**](https://x.com/minimax__ai/status/1879226391352549451): **MiniMax-01**, including **MiniMax-Text-01** and **MiniMax-VL-01**, can handle up to **4 million tokens**, surpassing existing models by **20–32 times**. Pricing is set at **$0.2 per million input tokens** and **$1.1 per million output tokens**, with a [free trial available on Hailuo AI](https://hailuo.ai).
- [**DeepSeek V3 Outperforms Claude in Coding Tasks**](https://huggingface.co/unsloth/DeepSeek-V3-GGUF): Users report that **DeepSeek V3** surpasses **Claude** in code generation and reasoning, though running it locally requires substantial resources—around **380GB of RAM** and multiple GPUs. It's praised for its "*impeccable reasoning*" on complex tasks.

**Theme 2. AI Tools and IDEs: Performance Hiccups and User Innovations**

- **Cursor IDE Faces Slowdowns and User Workarounds**: Users experience significant slow requests and downtime with **Cursor IDE**, likening the bug reporting process to a "*kindergarten*". While monitoring [Anthropic's Status](https://status.anthropic.com/), some developers created scripts using [Beyond Compare](https://github.com/sksarvesh007) to manage code snapshots due to Cursor's issues.
- **Codeium's Windsurf Woes and the Quest for Clarity**: Participants grapple with **AI-generated code mistakes** leading to development loops in **Windsurf**. Emphasizing the use of detailed **.windsurfrules** files, they seek better-structured approaches to refine outputs, referencing [Codeium Docs](https://docs.codeium.com/supercomplete/overview).
- **LM Studio Users Compare Qwen 2.5 and QwQ Models**: Testing between **Qwen 2.5 32B Instruct** and **QwQ** reveals that Qwen provides better code generation with fewer verbose answers. Users recommend models with **GGUF encodings** for optimal performance on consumer hardware, as noted in [local LLM recommendations](https://gist.github.com/shermanhuman/2b9a82df1bab242a8edffe504bb1867c).

**Theme 3. Advancements in AI Features: From Task Scheduling to Ambient Agents**

- [**ChatGPT Introduces Task Scheduling Feature**](https://www.theverge.com/2025/1/14/24343528/openai-chatgpt-repeating-tasks-agent-ai): On January 14, 2025, **ChatGPT** rolled out a new **Task Scheduling** feature for **Plus**, **Team**, and **Pro** users. This allows for setting one-time and recurring reminders, aiming to reposition ChatGPT as a proactive **AI agent**.
- [**Ambient Agents Automate Email Management**](https://blog.langchain.dev/introducing-ambient-agents/): A new **AI email assistant** autonomously triages and drafts emails, reducing inbox overload. Detailed in [Harrison Chase's announcement](https://x.com/hwchase17/status/1879218872727015644), it represents a move towards less obtrusive AI assistance requiring minimal supervision.
- [**Hyper-Connections Proposed to Improve Neural Networks**](https://arxiv.org/abs/2409.19606): Researchers introduce **Hyper-Connections** as an alternative to residual connections, addressing challenges like gradient vanishing. Early experiments show they meet or exceed existing methods, potentially enhancing both **language** and **vision** models.

**Theme 4. AI Infrastructure: GPU Access and Support Challenges**

- [**Thunder Compute Offers Affordable Cloud GPUs**](https://thundercompute.com): **Thunder Compute** launches with **A100 instances** at **$0.92/hr** plus **$20/month** free credit during beta. With an easy CLI (`pip install tnr`), it simplifies GPU workflows, aiming to make high-performance computing more accessible.
- **Unsloth AI Limited to NVIDIA GPUs, AMD Users Left Waiting**: Users discover that **Unsloth** currently supports only one NVIDIA GPU, causing confusion and frustration among those hoping for AMD support. Reference to [SHARK-AI’s AMD optimization guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary) highlights the community's interest in broader GPU compatibility.
- [**OpenRouter Users Face Rate-Limiting Issues with Models**](https://openrouter.ai/docs/limits): High demand leads to **rate-limiting** hurdles, especially with models like **DeepSeek V3**. While OpenRouter invites providers to integrate models via [support@openrouter.ai](mailto:support@openrouter.ai), users express frustration over performance bottlenecks.

**Theme 5. AI in Code Development: Practices and Philosophies**

- **Developers Debate Testing Tensions: "Jesus Take the Wheel" Approach**: Some programmers admit to minimal testing, humorously relying on "*Jesus take the wheel*" before pushing code changes. Others stress the importance of rigorous testing, especially in languages lacking compilation checks, to avoid risky deployments.
- **Community Emphasizes Clear Guidelines for AI Code Collaboration**: In tools like **Windsurf**, users highlight that detailed guidelines via **.windsurfrules** are crucial to reducing ambiguous AI responses. Sharing these rules and suggesting improvements via [Codeium's Feature Requests](https://codeium.canny.io/feature-requests) fosters a proactive community seeking better AI interactions.
- **Interest in AI for Real-Time Bug Fixing in Game Development**: Users speculate about future video games shipping with **real-time AI** capable of patching bugs on the fly. They humorously imagine AI fixing older titles, viewing it as a step towards fully **polished** gaming experiences.

---

# PART 1: High level Discord summaries




## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Woes & Wins**: Multiple participants confronted recurring **AI-generated code** mistakes, creating a 'loop of doom' in development, and sought better **structured approaches** to refine outputs, referencing [Codeium Docs](https://docs.codeium.com/supercomplete/overview).
   - Some expressed that adopting specialized instructions, rather than broad prompts, significantly improves reliability and fosters a more proactive community discussion around **Windsurf**’s potential.
- **Rules Rock with .windsurfrules**: Users stressed that meticulously defined **.windsurfrules** guides help clarify project requirements and cut down on ambiguous AI responses during code collaboration.
   - Community members suggested sharing these rules and filing requests at [Feature Requests | Codeium](https://codeium.canny.io/feature-requests) to accelerate the enhancement of **Windsurf** capabilities.
- **Quantum TicTacToe Tantalizes Techies**: A new **Quantum Computing Assembly Language** demonstration featuring ‘Quantum TicTacToe’ was showcased via a [YouTube video](https://youtu.be/-qa7_oe5uWQ).
   - Enthusiasts viewed this teaser as a spark for broader experimentation, hinting at the potential synergy between **Windsurf**'s AI-driven code generation and quantum-oriented projects.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth GPU Support Collides with AMD Aspirations**: Users discovered that **Unsloth** supports only one NVIDIA GPU at present, causing confusion about future AMD support and referencing [SHARK-AI’s AMD optimization guide](https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary).
   - Community members saw no immediate fix, while some pinned hopes on specialized GPU forums to produce workable patches.
- **Mistral's Codestral 2501 Sparks Licensing Buzz**: Mistral revealed **Codestral 2501** with an [official announcement](https://mistral.ai/news/codestral-2501), but users lamented its restricted API-only release and commercial slant.
   - They questioned if **enterprise licensing** would limit open-source collaboration, fueling heated debate on model access.
- **DeepSeek V3 Drops In For Local Testing**: Several members managed to run **DeepSeek V3** locally, reporting high VRAM and RAM consumption when pairing it with **llama3.1 405B**.
   - They traded performance tips, acknowledging slow speeds in minimal setups and the potential for heavy overhead in large-scale fine-tuning.
- **Llama 3.1 Fine-Tuning Hits Bumps**: A user struggled with absent validation loss metrics while fine-tuning **Llama 3.1**, even after adjusting dataset size and evaluation frequency.
   - They referenced [Unsloth’s Gradient Accumulation fix blog post](https://unsloth.ai/blog/gradient) and attributed issues to tricky training loops for large language modeling.
- **4bit Format Sparks Size Debates**: Some expressed enthusiasm for **4bit** model saving, hoping to reduce memory usage and keep smaller GPUs relevant.
   - They cited [Unsloth Notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks) for instructions, though concerns about model performance in compressed form persisted.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE's Cumbersome Performance**: Discord participants flagged **Cursor IDE** for major slow requests, comparing it to a 'kindergarten' bug reporting scenario, as documented in the [Cursor Community Forum](https://forum.cursor.com/).
   - They monitored [Anthropic Status](https://status.anthropic.com/) for possible interruptions but remained frustrated with the downtime that hurt coding productivity.
- **Claude Outshines O1**: Users discussed a preference for **Claude** over **O1**, noting that Claude excels at agent-mode tasks.
   - Developers cited O1's heavier resource demands, fueling debate over **model performance** in real-world usage.
- **Batch Files for Code Snapshots**: A dev created a script to produce numbered folders and use *Beyond Compare* for quick rollbacks on **Cursor IDE** mishaps.
   - They shared [their GitHub profile](https://github.com/sksarvesh007), encouraging others to adopt the strategy to track code modifications effectively.
- **Testing Tensions: 'Jesus Take the Wheel' Approach**: Some developers confessed to minimal testing, jokingly letting *'Jesus take the wheel'* before pushing code changes.
   - Others stressed rigorous checks in languages lacking built-in compilation, warning that headless deployment poses a risky but sometimes unavoidable trade-off.
- **MCP Servers and Slow Requests**: Community members anticipate **MCP servers**, suggesting they might improve the existing slowdown in Cursor.
   - Despite the wait times, many prefer the loose constraints of this system over stricter concurrency limits on other platforms.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 2.5 vs QwQ Showdown**: One user tested **Qwen 2.5 32B Instruct** alongside **QwQ**, reporting better code generation and fewer wordy answers with Qwen, but mixed results still emerged.
   - Participants noted **QwQ** had occasional inconsistencies, and overall reactions favored Qwen for clearer coding suggestions.
- **Local LLM Recs for Coders**: A member shared [a local LLM guide](https://gist.github.com/shermanhuman/2b9a82df1bab242a8edffe504bb1867c) emphasizing **GGUF encodings** for consumer hardware.
   - Others pointed to [bartowski/Sky-T1-32B-Preview-GGUF](https://huggingface.co/bartowski/Sky-T1-32B-Preview-GGUF) on **Hugging Face**, citing decent performance with carefully tuned quantizations.
- **Game Dev Gains with Generative AI**: Users speculated about future **video games** shipping with real-time AI that patches bugs on the fly, resulting in fewer crashes at launch.
   - They humorously imagined an emergent AI fix for older titles, describing it as a step closer to fully **polished** classics.
- **Multi-GPU Mayhem: RTX 5090 vs 4090**: Participants debated whether combining a **5090** with a **4090** could boost processing, although the older card might throttle performance.
   - They highlighted synchronization in layer-by-layer tasks, potentially causing idle time on the **RTX 5090** while the slower GPU catches up.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Quantum Quirk in Eukaryotic Vaults**: Researchers noted that **eukaryotic cell vaults** maintain coherence in **noisy** settings, fueling speculation about quantum computing uses.
   - Neoxah stayed tight-lipped on deeper specifics, hinting future expansions once the work reaches a formal release.
- **Hyper-Connections Combat Residual Roadblocks**: Researchers propose **Hyper-Connections** as an alternative to standard residual connections, citing **gradient vanishing** and **representation collapse** challenges, referencing [this paper](https://arxiv.org/abs/2409.19606).
   - Preliminary tests show they meet or exceed existing methods, sparking optimism for expansions in both **language** and **vision** pipelines.
- **Process Rewards & VinePPO Tweak LLM Reasoning**: New **Process Reward Models (PRMs)** spotlight token-level checks to strengthen math skills in LLMs, as outlined in [this study](https://arxiv.org/abs/2501.07301).
   - Conversations also explored **VinePPO** for chain-of-thought tasks, confirming it doesn't rely on explicit CoT examples to achieve consistent benefits in expansions like **LATo**.
- **MLQA Arrives in Style**: A new **MLQA** benchmark implementation appeared via pull request, adding multi-lingual QA coverage for the community, though an **AST error** awaits code review.
   - The submitter pointed out that **lm-eval-harness** includes **majority voting**, citing [this config snippet](https://github.com/EleutherAI/lm-evaluation-harness/blob/bb098f13b05e361f01a5afe7b612779ce362b3f2/lm_eval/tasks/gsm8k/gsm8k.yaml#L30) to set repeated sampling.
- **Llama 2's Quirky Config & Tokenizer Tales**: Developers spotted large shifts in **padded_vocab_size** for **Llama 2** (11008 vs 32768) between **NeoX** and **HF**, referencing [this config detail](https://github.com/EleutherAI/gpt-neox/blob/main/configs/llama2/7B.yml#L9).
   - They also observed HF uses **silu** over **swiglu**, which some see as a mismatch with earlier activation choices, all while encountering baffling dummy tokens in the build logs.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Supabase Setup Surprises**: Members noted that forking a Bolt project demanded new **Supabase** deployments each time, blocking reconnection to existing projects and disrupting normal workflows.
   - They compared it to **Loveable**’s reuse approach and hoped the development team would enable a simpler, more direct connection method.
- **Perplexity Chatbot Brainstorm**: A user proposed creating a **perplexity-style** chatbot by integrating a Hugging Face model into Bolt, sparking interest in open-source AI solutions.
   - Others suggested **OpenAI’s API** for quicker setup, but they also discussed the challenges of juggling different API services.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeVries AI Launches Telegram LLM Hub**: The new [DeVries AI](https://devriesai.com/) offers **200+ Large Language Models** in Telegram for $24.99/month with free trials available.
   - Users can quickly switch between **ChatGPT** and **Claude**, and soon add **image/video generation** in a single Telegram interface.
- **OpenRouter Provider Setup Gains Momentum**: **OpenRouter** invited prospective providers to email [support@openrouter.ai](mailto:support@openrouter.ai) to integrate their models, fueling talk of creative usage.
   - One user joked about a *provider that secretly uses OpenRouter*, prompting comedic speculation on inadvertently building **AGI**.
- **Deepseek V3 Slowness Sparks Concern**: Users reported **Deepseek V3** failing to respond **7 out of 10** times, pointing to overload causing sluggish replies.
   - Some suggested switching to **Together AI endpoint** for faster performance.
- **MiniMax 456B Parameter Model Draws Attention**: **MiniMax** introduced a model with **456 billion** parameters, showing robust context handling despite not topping benchmarks.
   - Its efficient scale has piqued interest among developers exploring larger performance possibilities.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Discord Bots or Not? Comical Conundrum**: Members joked about suspicious newcomers who simply greet, suspecting **Discord bots** were lurking behind these one-line hellos.
   - Some proposed stricter sign-up steps, but worried it might discourage genuine participants.
- **DEIS BETA Boldly Boosts Flux Sampling**: Enthusiasts praised **DEIS BETA** for effectively guiding flux sampling in **Stable Diffusion** scenarios.
   - They also hunted for additional tools, aiming to improve sampling parameters across diverse tasks.
- **Aesthetic Classifier Gains Curious Fans**: A user sought datasets blending art styles with numeric ratings to build a solid **aesthetic classifier**.
   - Suggestions included leveraging **ollama** for streamlined prompting, hoping to unify subjective and objective scoring methods.
- **FP8 vs FP16: Showdown of the Bits**: Community members debated the merits of **FP8** in newer GPUs versus more common **FP16** in older devices.
   - They noted **FP8**'s memory benefits but worried about accuracy trade-offs in high-detail **Stable Diffusion** jobs.
- **Intel B580 Stumbles Seek Solutions**: A contributor lamented posting hassles around **Intel B580** benchmarks for **Stable Diffusion** due to subreddit restrictions.
   - Others advised contacting mods or exploring alternative forums to gather broader feedback and insights.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Qwen's PRM Gains Ground on Process Supervision**: The new **Qwen2.5-Math-PRM** aced intermediate error detection in math tasks on [ProcessBench](https://huggingface.co/papers/2412.06559), referencing a [72B model on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B) that uses **human-annotated** data for stronger reasoning.
   - Developers cautioned that **Monte Carlo** synthetic approaches lag behind human methods, highlighting the need for careful **evaluation**.
- **Claude Sonnet & MiniCPM-o Make Waves**: **Claude Sonnet 3.5** hit **62.2%** on SWE-Bench Verified, trailing OpenAI's o3 at **71.7%**, startling many who saw it as a previous-gen coding contender.
   - Meanwhile, **MiniCPM-o 2.6** from OpenBMB, boasting an **8B-size Omni** design, impressed with real-time bilingual audio capability, as shown on [GitHub](https://github.com/OpenBMB/MiniCPM-o) and [Hugging Face](https://huggingface.co/openbmb/MiniCPM-o-2_6).
- **Higher Ed Chatbots & Stripe's Tax Trick**: A talk for **higher-ed CIOs** spotlighted **U-M GPT** and **Maizey**, with the University of Michigan championing tailored AI offerings for diverse campus needs.
   - On the tax front, members praised **Stripe**'s Non-Union One Stop Shop, letting outside businesses handle **EU VAT** in one swoop.
- **Synthetic CoT & O1 Drama Bawl**: Members found **synthetic chain-of-thought training** underwhelming, especially when it was just supervised fine-tuning with no **RL**.
   - They doubted the chances of **O1 models**, hinting that **Big Molmo** or **Tulu-V** might do a better job for vision tasks.
- **Policy Punch: AI Blueprint & Datacenter Boom**: An [Economic Blueprint](https://openai.com/global-affairs/openais-economic-blueprint/) proposes harnessing **AI** for national security and growth, echoing repeated policy suggestions from OpenAI.
   - President Biden's executive order unlocks federal land for **gigawatt-scale datacenters**, mandating on-site **clean energy** to match capacity.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **ChatGPT's Scheduling Surge**: On January 14, 2025, **ChatGPT** unveiled a new **Task Scheduling** feature that handles one-time and recurring reminders, as detailed in [The Verge report](https://www.theverge.com/2025/1/14/24343528/openai-chatgpt-repeating-tasks-agent-ai), initially rolling out to **Plus**, **Team**, and **Pro** users.
   - This move aims to recast **ChatGPT** as a more proactive **AI agent**, enabling tasks like daily weather updates or news alerts, as mentioned in [TechCrunch](https://techcrunch.com/2025/01/14/chatgpt-now-lets-you-schedule-reminders-and-recurring-tasks).
- **Cursor's Series B Grab**: **Cursor** announced a **Series B** funding round co-led by **a16z**, showcasing strong investor confidence in advanced coding tools and AI-powered dev platforms, with more context in [Sarah Wang's tweet](https://x.com/sarahdingwang/status/1879279307119608142).
   - This injection of capital highlights growing enthusiasm for AI-assisted development and sets the stage for further improvements in **Cursor's** tooling ecosystem.
- **Ambient Agents Automate Email**: A new **AI email assistant** autonomously triages and drafts emails, with the concept behind such 'ambient agents' detailed in a [blog post](https://blog.langchain.dev/introducing-ambient-agents/) and further discussed in [Harrison Chase's tweet](https://x.com/hwchase17/status/1879218872727015644).
   - This approach promises to reduce email overload by handling routine tasks behind the scenes, letting users focus on higher-level decisions and minimal direct supervision.
- **Claude's Rate-Limiting Roadblock**: Users reported **rate-limiting** hurdles with the **Claude** model Sonnet 3.6 on **Cursor**, with [forum chatter](https://forum.cursor.com/t/anthropic-cannot-sustain-additional-slow-request-traffic-on-claude-3-5-sonnet-please-enable-usage-based-pricing/41361/24) blaming high traffic that exceeds Anthropic’s **GPU availability**.
   - Developers shared that **Cursor** is Anthropic's largest customer, intensifying the demand for more robust GPU provisioning.
- **Magnetar's Compute-for-Equity Maneuver**: **Magnetar**, a hedge fund, offers compute resources to AI startups in exchange for equity, as covered in a [recent podcast](https://podcasts.apple.com/ca/podcast/how-the-hedge-fund-magnetar-is-financing-the-ai-boom/id1056200096?i=1000679726051) and through collaboration with **Coreweave**.
   - This strategy aims to lessen the funding logjam for emerging AI ventures, underscoring the significance of infrastructure access in fueling next-generation AI developments.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Google's $10 Audio Overviews Survey**: The Google team introduced a [5-minute screener form](https://forms.gle/NBzjgKfGC24QraWMA) to gather feedback on **Audio Overviews**, awarding a **$10 gift code** upon completion.
   - They require participants to be at least 18, aiming to shape future **NotebookLM** updates based on user insights.
- **Akash, a New Site for AI-Generated Podcasts**: One user showcased [Akash](https://akashq.com) as a convenient site for uploading and sharing **AI-generated podcasts**, removing complicated permission steps.
   - They provided examples of distributing NotebookLM-based content, describing it as *'a simpler approach.'*
- **Podcast Duration Dilemmas**: Community members debated how to restrict **podcast length** for NotebookLM, citing a [Reddit link](https://www.reddit.com/r/notebooklm/comments/1gmtklt/longest_pod_i_have_ever_got/) for attempted solutions.
   - Others discussed direct *audio transcription*, suggesting a built-in feature instead of uploading files as sources.
- **Paid NotebookLM's Quest for Public Sharing**: Questions arose about the **paid NotebookLM** version offering fully public access, without manual permissions for each user.
   - Some members noted only organization-wide sharing works now, inspiring calls for more *open publication*.
- **NoCode RAG Brainstorm for PDFs**: A user raised the idea of using NoCode methods to retrieve answers from **PDFs** in Google Drive, tying it to NotebookLM's retrieval workflows.
   - Participants recognized the complexity of integrating that approach, hoping for deeper support in upcoming iterations.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro’s Topsy-Turvy Reception**: Users raised concerns about redeemed reward codes failing on **Pro features**, limited effectiveness for **coding help**, and friction in activating **Pro search** seamlessly. Some applauded its research benefits but criticized UI changes, pointing to [Ublock Origin](https://ublockorigin.com) for blocking ads and unwanted content.
   - Members asked if **Pro search** could be accessed via API, but official replies confirmed it’s unavailable, frustrating workflows. Others worried about **private content** not appearing on Google and losing access to previously uploaded documents.
- **Coding Assistant Oversteps Boundaries**: A **coding assistant** repeatedly insisted on disclaimers and partial code confirmations despite explicit user directives. This caused friction and dissatisfaction with the assistant’s unresponsive design.
   - Community members suggested more adaptive conversation flows to reduce repetitive disclaimers. Some viewed the behavior as *unnecessary friction* that complicated development tasks.
- **TikTok Tussles with Extra Oversight**: Chinese officials weighed possible guidelines around **TikTok**, focusing on content moderation and user privacy, according to [this article](https://www.perplexity.ai/page/chinese-officials-consider-tik-51PEvvekQxqVuhd74SkSHg). They highlighted increasing concern over data handling and regulatory action.
   - Observers expect more scrutiny from government entities with possible global consequences. Users remain uncertain when or how these rules will be fully imposed.
- **German Summary Request Sparks Translation Talk**: A user asked for a **German-language** summary of data referenced in [this discussion](https://www.perplexity.ai/search/kannst-du-mir-eine-zusammenfas-aZ8WnXOORK.hH5cZSr0qqg). They stressed the importance of localized coverage.
   - Some questioned how **Perplexity** manages multilingual queries at scale. Others viewed it as an interesting test for cross-lingual AI knowledge sharing.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek Dethrones Claude**: **DeepSeek v3** overshadowed **Claude** in coding tasks, even though **Anthropic** keeps adjusting Claude's post-training approach.
   - Members shared a [fist-pump GIF](https://tenor.com/view/never-give-up-fist-pump-motivation-motivational-gif-4892451) and appreciated Claude's human-like style, hinting it might stay a user favorite.
- **Private Data vs Open Source Face-Off**: Participants debated whether **open source** can match **proprietary** training sets, with some suggesting government data hubs to even the field.
   - They argued that dataset quality outshines mere quantity, fueling skepticism about fully synthetic corpora.
- **Gemini Grabs Data with Ease**: **Gemini** earned praise for **data extraction**, overshadowing **4o-mini** and **Llama-8B** in accuracy.
   - Participants proposed **Jina** for specialized text conversion, referencing programmatic methods to ensure precise results.
- **Spotlight on Attention Alternatives**: A new [paper](https://arxiv.org/abs/2501.06425) promises methods beyond standard attention, fueling speculation.
   - The group labeled this the **month of attention alternatives**, expecting more robust approaches in upcoming releases.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek v3 demands big memory**: Contributors reported that effectively running **DeepSeek v3** requires about **380GB** of RAM, multiple GPU cards, and recommended checking [the official Hugging Face repo](https://huggingface.co/unsloth/DeepSeek-V3-GGUF).
   - They compared it to smaller options like **Qwen**, noting the trade-offs in performance when hardware resources are limited.
- **Qwen runs locally with fewer resources**: Members recommended **Qwen** as a smaller open-source alternative for local usage, highlighting its lower resource requirements compared to bigger models like **DeepSeek v3**.
   - They indicated it offers balanced performance and avoids heavy memory overhead, although no benchmark data was explicitly shared.
- **Gemini excels at user story creation**: Discussion suggested the **Gemini** model as an effective open-source tool for generating user stories tailored to specific requirements.
   - Participants praised its specialized capabilities for narrative tasks but did not provide explicit metrics or links to confirm these claims.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **MiniMax's 4M Token Triumph**: **MiniMax-01**, including **MiniMax-Text-01** and **MiniMax-VL-01**, launched open-source and can handle up to **4M tokens**, far surpassing existing models by 20–32 times ([paper](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf)).
   - Their pricing is **$0.2** per million input tokens and **$1.1** per million output tokens, with a [free trial on Hailuo AI](https://hailuo.ai), prompting excitement around next-gen AI agent tooling.
- **Thunder Compute's Cloudy Bargain**: A co-founder announced [Thunder Compute](https://thundercompute.com), offering **A100** instances at **$0.92/hr** plus **$20/month** free credit during beta.
   - They highlighted an easy CLI (`pip install tnr`) for fast instance setup, simplifying GPU workflows and requesting user feedback.
- **Kaiko & Prior Labs Seeking Model Builders**: Kaiko AI is hiring **Senior ML Platform Engineers** and **Data Engineers** in Amsterdam and Zurich, focusing on Foundation Models for cancer treatments, with no visa sponsorship ([ML Engineer posting](https://jobs.kaiko.ai/jobs/4829222-senior-ml-platform-engineer)).
   - Meanwhile, **Prior Labs** is building Foundation Models for **tabular data**, **time series**, and databases, citing a [Nature article](https://www.nature.com/articles/s41586-024-08328-6) that underlines broad impacts in healthcare and finance.
- **TorchAO Tinkers with int8**: Community members confirmed **int8_weight_only** uses a fused dequant-and-matmul approach optimized by [torch.compile](https://github.com/pytorch/ao/tree/main/torchao/quantization#workaround-with-unwrap_tensor_subclass-for-export-aoti-and-torchcompile).
   - They demonstrated how to export these quantized models via **torch.export** or **ONNX**, emphasizing compatibility with **TorchScript** for performance gains.
- **DeepSeek 2.5 Delivers**: Members applauded **DeepSeek 2.5** for *“impeccable reasoning”* on a shared task, illustrating notably advanced logic.
   - They shared an image for verification, showcasing strong results and drawing curiosity about the model’s broader capabilities.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Codestral Catapults 256k Context**: The new **codestral** model, free on the **Mistral API** with **256k** context, is described as “stupid fast and good" by those testing its efficiency.
   - Users anticipate significant speed benefits for extensive code generation, citing its large context window and ease of deployment.
- **ChatGPT 4o Canvas Confusion**: Members questioned if **ChatGPT 4o with Canvas** is just the older model from September or a newly released variant, given OpenAI's ambiguous rollout.
   - Some observed that their previous **4o canvas** conversations reverted to **4o mini**, fueling further speculation about system updates.
- **AI Misalignment Sparks Chatter**: A [YouTube video](https://www.youtube.com/watch?v=K8p8_VlFHUk) on **AI misalignment** grabbed attention, showcasing animated scenarios of potential risks.
   - Questions arose around the video's relevance, prompting viewers to explore how it aligns with broader concerns about advanced AI systems.
- **PDF is Not an API**: Contributors in **prompt-engineering** and **api-discussions** sought better data formats than **PDF**, advocating for **JSON**, **YAML**, or plain text.
   - One user joked that *PDFs are not an API*, echoing collective frustration with unwieldy document conversions for AI tasks.
- **Language Simplification for Non-Natives**: A new **de-GPTing** prompt helps rephrase text to omit rare words while preserving essential technical terms.
   - Users shared a custom technique in the [OpenAI Playground](https://platform.openai.com/) to cut repetitive references, aiming for clarity in responses.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Konkani Coordination & Linguistic Preservation**: In **Cohere discussions**, a user spotlighted **Konkani** spoken by **2.5 million** people in Goa, with developer **Reuben Fernandes** seeking professional collaboration to boost acceptance of his language-preservation project.
   - He plans to create an **AI model** that converses in **Konkani**, emphasizing that no existing systems adequately handle the language, which stirred curiosity among participants.
- **Rerank Fine-Tuning Pricing Puzzles**: Members questioned **Rerank FT** costs missing from [Cohere pricing](https://cohere.com/pricing), prompting references to [official docs](https://docs.cohere.com/docs/rerank-fine-tuning) for clarification.
   - They shared **FAQ links** and suggested these resources might shed light on specialized policies, revealing a push for clearer cost structures.
- **Cohere's 128k Context Limit Dig**: Participants clarified a **128k tokens** capacity (around **42,000** words) that spans all interactions, emphasizing it's more than just single-chat memory.
   - Discussion contrasted **long-term vs short-term** memory, with usage-rate constraints outlined in the [Cohere documentation](https://docs.cohere.com/v2/docs/rate-limits) rather than based on token length.
- **Alice in Wonderland Bot Banter**: The **Cmd R Bot** denied any link between *corvo* and *escrivaninha*, but an **Alice in Wonderland** reference suggested a hidden linguistic twist.
   - Its **Cohere documentation** search turned up empty, underscoring gaps in addressing cultural or literary angles.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Async Aspirations**: Owen posted two [pull requests](https://github.com/modularml/mojo/pull/3945) and [pull request #3946](https://github.com/modularml/mojo/pull/3946) for **structured async** and **effect handlers** in **Mojo**, stressing the need for standardizing exceptions like **oom** and **divbyzero**.
   - Attendees debated multiple executor designs, suggesting that a basic API layer is crucial before branching into advanced concurrency.
- **Zed Zoom: The Mojo Extension Marches On**: A developer created a dedicated [Mojo in Zed extension](https://github.com/freespirit/mz) that overcame **stdlib** path detection issues and offered improved LSP functionalities.
   - Others swapped suggestions to refine autocompletion, with some highlighting the extension's potential for broader **Mojo** adoption.
- **Mojodojo's Int8 Hitch**: A user encountered a [conversion error](https://github.com/modularml/mojo/issues/3947) converting **Int8** to **String** in **Mojodojo**, citing partial code from [the docs](https://mojodojo.dev/guides/intro-to-mojo/basic-types.html#strings).
   - Community members shared references to [String struct docs](https://docs.modular.com/mojo/stdlib/collections/string/String/#write_bytes) and [Parameterization concepts](https://docs.modular.com/mojo/manual/parameters/) to address the mismatch.
- **Meet & Stream: Quick Refresh**: A participant missed part of a meeting due to a class conflict but thanked others for an **update** that kept them in sync.
   - They shared a [YouTube video](https://www.youtube.com/watch?v=PYtNOtCD1Jo) as a helpful resource for anyone unable to catch the full conversation.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tiny corp’s $5M leap for accessible compute**: A [blog post](https://geohot.github.io/blog/jekyll/update/2023/05/24/the-tiny-corp-raised-5M.html) revealed that **tiny corp** raised **$5M** to accelerate their push toward chip development for advanced computing.
   - The founder highlighted the massive gap between **20 PFLOPS** of human brain compute and current HPC costs, sparking discussions on bridging public access to bigger compute.
- **Tinygrad’s role in tackling hardware gaps**: Members discussed the **purpose of tinygrad**, focusing on its capacity to handle GPU and CPU backends with minimal overhead.
   - They noted that familiarity with **LLVM** aids in understanding how tinygrad orchestrates lower-level operations, informed by a distributed systems viewpoint.
- **Stacking Tensors hits recursion snag**: Users ran into a **RecursionError** when stacking over **6,000** tensors and then calling `.numpy()`. 
   - They reduced to **1,000** tensors and bypassed the stacking limit, with suggestions to chunk operations to avoid **internal recursion depth** issues.
- **Dimension confusion triggers errors**: A user discovered an **IndexError** caused by calling `transpose()` on a 1D tensor in tinygrad.
   - Others explained that specifying dimension parameters is critical for safe operations, highlighting the importance of dimension-awareness with **tensor attributes**.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Scaling Sprints for Medical Mastery**: Increasing **inference time** by **6%-11%** on a small training set of **500 samples** significantly boosted LLM performance in medical benchmarks, as presented in [O1 Replication Journey -- Part 3: Inference-time Scaling for Medical Reasoning](https://arxiv.org/abs/2501.06458). **Task complexity** demanded extended reasoning chains, prompting the exclusion of in-house data to avoid confusion in logical deductions.
   - **Nature Communications** underscored gaps in model metacognition in [Large Language Models lack essential metacognition for reliable medical reasoning](https://www.nature.com/articles/s41467-024-55628-6), spotlighting the tension between extra computational cost and clinically robust outputs.
- **O1 Jailbreak Jitters**: Skeptics questioned the validity of **jailbreaking O1** for new model training, criticizing existing benchmarks as disconnected from real-world needs.
   - Others demanded more thorough risk assessment, warning that careless jailbreaking practices erode trust in the resulting systems and necessitate rethinking the entire O1 approach.
- **Healthcare's Automated Assessment Angst**: Members argued that **multiple-choice-based** evaluations in healthcare constrain AI to pattern recognition and memory tasks, missing deeper clinical capabilities.
   - They called for more nuanced testing protocols to gauge how future AI might participate in actual diagnosis and treatment scenarios.
- **Debunking Medical Testing Myths**: A debate arose over equating **multiple-choice exam** success with genuine clinical skills, noting that real-world scrutiny by seasoned doctors goes beyond test scores.
   - Enthusiasts pushed for combining prompt-driven AI with hands-on expertise, aiming at more realistic assessments of a model's clinical competence.
- **Redefining AI’s Future in Medicine**: Participants stressed that **AI** should reshape established medical norms and training, rather than replace doctors, to advance patient care.
   - They urged designers to challenge outdated routines, envisioning balanced AI-human team efforts built around ethical safeguards and authentic clinical needs.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Gains Video Edge**: After a user overcame issues installing **Open Interpreter** via `pipx` and **brew**, they confirmed it can handle commands for **video editing**.
   - They also noted **Cmder** performance glitches when **Open Interpreter** emits large outputs, leading to frequent screen clearing.
- **Deepseek Model & Integration Insights**: A user asked about the **Deepseek** model name and setting up the **DEEPSEEK_API_KEY** for clarity on usage.
   - They also inquired about integrating **Deepseek** into **Open Interpreter**, revealing interest in bridging both tools.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **2024 MOOC Spurs Nostalgia**: One user recalled **2024** with strong admiration, describing the MOOC as a highlight of the year.
   - They shared excitement about rewatching [Fall 2024 MOOC lectures](https://llmagents-learning.org/f24) and connecting with others who felt the same spark.
- **MOOC Gains Traction Among Beginners**: A newcomer asked about **beginner friendliness** after completing a prior machine learning class, seeking an easy transition.
   - Others responded that [Fall 2024 MOOC lectures](https://llmagents-learning.org/f24) strike a balance between **core concepts** and advanced tips for upskilling.
- **Fall 2024 Lectures Lay the Groundwork for Spring 2025**: A member urged prospective learners to watch the [Fall 2024 lectures](https://llmagents-learning.org/f24) to gain background knowledge for upcoming **Spring 2025** modules.
   - They noted that the next session won't strictly require prior expertise, but a head start can't hurt.
- **Certificate Release Calms Concerned Students**: A user inquired about the **Fall 2024 MOOC certificate**, worried they'd miss the official award.
   - Another user confirmed that certificates will be **released later this month**, reducing everyone's anxiety about recognition.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **AMD edges closer to NPU integration in GPT4All**: A question was raised about whether GPT4All would soon exploit **NPU** on AMD processors, hinting at future performance boosts.
   - A developer mentioned that AMD's software stack remains a constraint, but indicated that support would be promising once it is finalized.
- **VPN solution for remote GPT4All usage**: Participants recommended using a **VPN** or **reverse proxy** on the inference machine to allow GPT4All's interface to be reached from other devices.
   - They described it as a practical method for multi-device interactions without hardware complications.
- **Hugging Face clarifies GPT4All model variations**: A conversation highlighted the presence of multiple **quantization** variants such as codellama q4_0 on **Hugging Face**.
   - Placing the model files in a single folder apparently resolved confusion about using different versions.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Agent Aces the RAG Race with Weaviate**: In a recent [notebook by Tuana](https://twitter.com/tuanacelik), an agent using **Weaviate** and **LlamaIndex** outperformed naive RAG in retrieving relevant data.
   - Community members credited the agent's approach to combining data sources for stronger coverage, spotlighting its decision-making capabilities.
- **QAE Gains Custom Prompt Power**: A user explored extra variables in `QuestionsAnsweredExtractor` using `self.llm.apredict()`, citing [LlamaIndex advanced prompts documentation](https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts/#3-prompt-function-mappings).
   - Another member shared how function mappings can feed dynamic variables, showing **LlamaIndex** can fluidly inject multiple data points into prompt templates.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Meta's JASCO Ignites Music Generation**: The **FAIR team of Meta AI** introduced **JASCO**, a new music model trained in **November 2024** that uses [EnCodec](https://huggingface.co/facebook/jasco-chords-drums-melody-1B) for audio tokenization and handles **chords**, **drums**, and **melody**.
   - It comes in **400M** and **1B** variants with a **flow-matching** backbone and condition dropout, spurring excitement for flexible text-to-music generation.
- **JASCO Paper Highlights Tech Foundations**: A paper titled [Joint Audio And Symbolic Conditioning for Temporally Controlled Text-To-Music Generation](https://arxiv.org/pdf/2406.10970) outlines JASCO's transformer-based architecture and features.
   - Engineers discuss its specialized audio and symbolic conditioning, noting potential for **next-level** musical composition and model sophistication.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Ambient Agents, Still Missing Examples**: A member asked how to configure an **ambient agent** using DSPy, requesting any code samples, but none surfaced in the chat.
   - Others echoed interest in real-world DSPy use cases, hoping for **shared resources and experiences** from fellow developers.
- **DSPy Implementation Show-and-Tell**: Another participant invited more **DSPy** demonstrations, highlighting any hands-on trials or partial prototypes for ambient agent scenarios.
   - They encouraged the community to share relevant details or open-source repos, aiming to spark growth in **DSPy** solutions.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI in Healthcare & Finance Gains Speed**: On **January 16, 2025**, from **4:00 pm to 5:30 pm IST**, a global panel will discuss how **AI** impacts healthcare and finance, with registration at [this link](http://bit.ly/3PxJXbV).
   - Organizers invited **builders, operators, and owners** of **AI-enabled solutions** to tackle cost optimization and data management, hoping to accelerate AI adoption in these sectors.
- **Panel Eyes Real-World AI Deployments**: The panel plans to emphasize operational details, including data interoperability, compliance, and real-time analytics in the healthcare and finance fields.
   - They stress **cross-pollination** between these sectors, anticipating new strategies for scaling machine learning models with minimal overhead.



---


The **Axolotl AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1328453553307521074)** (16 messages🔥): 

> `Support Ticket Issues, Codeium Connectivity Problems, Technical Assistance, Freedom of Speech Concerns, Telemetry Errors on VS Code` 


- **Support Ticket Blues**: After nearly **10 days** without a response, a member suggested that mods should escalate their unresolved support ticket for quicker resolution.
   - Another member mentioned they couldn't find the ticket date they submitted, emphasizing the delay in support response.
- **Codeium Connectivity Woes**: A member reported an **abnormal connection close by server** on Codeium, indicating potential server-side issues.
   - They urged others to **download diagnostics** if the problem persists for further troubleshooting.
- **Help with Technical Issues Appreciated**: A user expressed gratitude to another member for assistance with an **email conflict**, which created confusion over account ownership.
   - They highlighted the support received during this technical issue, appreciating the **patience and time** offered.
- **Debate on Freedom of Speech**: A member voiced frustration over deleted messages, suggesting a lack of freedom of speech in the channel.
   - In response, others emphasized the importance of **respect** and adherence to community guidelines.
- **Telemetry Issues on VS Code**: A member raised concerns regarding telemetry errors in their **Codeium VS Code** extension despite having it enabled.
   - They pointed to a related issue on GitHub, seeking assistance from others dealing with the same problem.



**Link mentioned**: <a href="https://github.com/Exafunction/CodeiumVisualStudio/issues/111.">Exafunction/CodeiumVisualStudio</a>: Visual Studio extension for Codeium. Contribute to Exafunction/CodeiumVisualStudio development by creating an account on GitHub.

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1328453559829794919)** (350 messages🔥🔥): 

> `Windsurf Performance, Improvements in AI Suggestions, Need for Clear Guidelines, Application Development Challenges, User Experience and Feedback` 


- **Struggles with AI-generated Code**: Users have expressed frustration over the recurring issues with Windsurf, noting that AI-generated code often introduces errors after initial success, creating a 'loop of doom' during development.
   - Participants are trying to implement structured approaches and documentation to guide the AI and improve the accuracy of outputs.
- **Importance of Clear Guidelines**: There is a consensus that clear and specific guidelines in the form of `.windsurfrules` files are essential for effective collaboration with the AI, as vague prompts lead to unsatisfactory results.
   - Users are encouraged to articulate their requirements like they would to a developer, aiming to avoid ambiguity in AI responses.
- **Experiences Sharing**: Several users shared their individual experiences creating applications with Windsurf, ranging from creating a news site to developing complex systems like logging and XML parsers.
   - While many participants are excited about Windsurf's capabilities, they also express challenges faced due to limitations in functionality.
- **Feedback and Suggestions**: Users have raised concerns about the AI's suggestion behavior, particularly the randomness in selections, prompting discussions about potential improvements.
   - Feedback mechanisms and enhancement suggestions are encouraged to directly address users' experiences with the tool.
- **User Assistance and Resources**: New users, such as those in non-development roles, seek guidance on utilizing Windsurf effectively, highlighting a need for resources and tutorials.
   - Community members are collaborating and sharing tips on efficient use and prompt structuring to mitigate confusion and maximize the AI's capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/wink-eye-turn-around-chewing-gif-23703707">Wink Eye GIF - Wink Eye Turn - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/power-starwars-unlimited-power-gif-15939349">Power Starwars GIF - Power Starwars Unlimited Power - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.codeium.com/supercomplete/overview">Overview - Codeium Docs</a>: no description found</li><li><a href="https://codeium.canny.io/feature-requests">Feature Requests | Codeium</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://codeium.com/blog/pricing-windsurf">Plans and Pricing Updates</a>: Some changes to our pricing model for Cascade.</li><li><a href="https://youtu.be/-qa7_oe5uWQ">Quantum TicTacToe</a>: Playing with our new Quantum Computing Assembly Language
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1328454686663245864)** (128 messages🔥🔥): 

> `Unsloth GPU Support, Mistral Code Generation Models, DeepSeek V3 Local Run, Chat Templates in Models, Model Licensing and Availability` 


- **Unsloth struggles with GPU support**: Users noted that **Unsloth only works on a single NVIDIA GPU** currently and lacks support for AMD, leading to frustrations.
   - There are questions about the possibility of future AMD support, with no clear answers at this time.
- **Mistral's Codestral model release uncertainty**: There was speculation around Mistral's **Codestral 2501** being unavailable to the public, with users expressing disappointment over its current API-only access.
   - The team appears to be prioritizing commercial licensing for models, which has led to discussions about the implications for average users.
- **Running DeepSeek V3 Locally**: Users shared their experiences with running **DeepSeek V3** locally and the challenges with hardware limits, particularly RAM and VRAM requirements.
   - Optimizing setups, such as using **llama3.1 405B**, is a common workaround, yet there remains uncertainty about efficiency with constrained resources.
- **Importance of Chat Templates**: There was discussion over the necessity of incorporating a **chat template** in inference setups, particularly in context to instruction-tuned models like **Phi-4**.
   - Some members noted that failure to use a template leads to odd completions and that different models may use various instruction indicators.
- **Concerns over Model Licensing**: Conversations highlighted the complexities surrounding model **licensing and availability**, especially for small and mid-sized models from Mistral.
   - Participants expressed concerns about potential delays and the business-oriented approach limiting access for individuals and researchers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://grok.com/">Grok</a>: Understand the Universe</li><li><a href="https://mistral.ai/news/codestral-2501">Codestral 25.01</a>: Code at the speed of Tab. Available today in Continue.dev and soon on other leading AI code assistants.</li><li><a href="https://x.com/iScienceLuvr/status/1879091791753843064">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Tensor Product Attention Is All You NeedProposes Tensor Product Attention (TPA), a mechanism that factorizes Q, K, and V activations using contextual tensor decompositions to achieve 10x or more reduc...</li><li><a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto">Reward Modelling - DPO, ORPO &amp; KTO | Unsloth Documentation</a>: To use DPO, ORPO or KTO with Unsloth, follow the steps below:</li><li><a href="https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md#glossary">shark-ai/docs/amdgpu_kernel_optimization_guide.md at main · nod-ai/shark-ai</a>: SHARK Inference Modeling and Serving. Contribute to nod-ai/shark-ai development by creating an account on GitHub.</li><li><a href="https://huggingface.co/microsoft/phi-4/blob/main/tokenizer_config.json#L774>">tokenizer_config.json · microsoft/phi-4 at main</a>: no description found</li><li><a href="https://huggingface.co/datasets/bigcode/santacoder-fim-task">bigcode/santacoder-fim-task · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1328478985574809670)** (7 messages): 

> `Dynamic Filters in LLAMA, Language Learning, Video Resources, Pronunciation Practice` 


- **Dynamic Filters Make LLAMA Fun**: A user suggested experimenting with dynamic user-defined filters in **LLAMA**, arguing they offer a faster way to express ideas compared to long prompts.
   - They provided a humorous example involving roles like 'Screaming Crazy Cat Lady' and a superhero mailman in a movie script scenario.
- **Language Learning Options Explored**: Several members discussed their preferences for languages to learn, with suggestions including **Polish** and **Japanese**.
   - They emphasized the necessity of effective methods for practicing both pronunciation and typing skills.
- **Video Resource Shared**: A user posted a link to a YouTube video, possibly related to the ongoing discussions.
   - Specifically, the link shared was [this video](https://www.youtube.com/watch?v=wXGZC-fCrco).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1328464452118642782)** (144 messages🔥🔥): 

> `Fine-tuning on Unsloth, Training VLMs, RAG for hallucination control, Using 4bit saving, Dataset preparation for LLM` 


- **Fine-tuning issues with Llama 3.1**: A user reported issues with their Llama 3.1 fine-tuning, specifically not receiving validation loss during training despite attempts with different methods.
   - Suggestions were made to adjust the evaluation dataset size and frequency to improve training efficiency.
- **Training VLMs and memory issues**: A user running into system memory limitations during VLM training was advised that reducing training data size would help but not their desired full dataset.
   - They also noted a significant slowdown due to a large evaluation dataset being processed during each training epoch.
- **RAG method for hallucination control**: When discussing the use of Llama 3B for legal datasets, it was suggested that larger models (70B+) along with RAG might mitigate issues like hallucinations.
   - Concerns were raised about hallucinations in both language and information context, emphasizing the need for robust validation.
- **Using 4bit and model configuration**: A user inquired about the possibility of saving models in a 4bit format, receiving confirmation that such functionality exists within provided notebooks.
   - Discussions also touched on how to configure model loading with low-end GPUs using FastLanguageModel.
- **Dataset preparation for LLMs**: For users interested in creating LLMs based on specific author styles, dataset preparation was emphasized as being 85% of the work involved.
   - It was noted that understanding LLM functionalities is essential for effective dataset curation and training processes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discuss.huggingface.co/t/perhaps-your-features-output-in-this-case-have-excessive-nesting-inputs-type-list-where-type-int-is-expected/135553">Perhaps your features (`output` in this case) have excessive nesting (inputs type `list` where type `int` is expected)</a>: I am also getting similar issue here.  ValueError: Unable to create tensor, you should probably activate truncation and/or padding with  &#39;padding=True&#39; &#39;truncation=True&#39; to have batche...</li><li><a href="https://cohere.com/llmu/what-is-semantic-search">What is Semantic Search?</a>: In this LLM University chapter, you’ll learn how to use embeddings and similarity in order to build a semantic search model.</li><li><a href="https://unsloth.ai/blog/gradient">Bug Fixes in LLM Training - Gradient Accumulation</a>: Unsloth&#x27;s Gradient Accumulation fix solves critical errors in LLM Training.</li><li><a href="https://colab.research.google.com/drive/1mf3lqz2ga80p_rIufDvBPvyFqtn9vcdS?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

fjefo: https://youtu.be/SRfJQews1AU?si=s3CSvyThYNcTetX_
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1328462335102287933)** (209 messages🔥🔥): 

> `Cursor IDE performance issues, Model comparisons, Batch file for snapshots, Using Claude vs O1, User experiences with Cursor` 


- **Cursor IDE suffers from performance issues**: Users have reported ongoing performance problems with the Cursor IDE, especially with 'slow requests' impacting response times.
   - The downtime and slowdown have led to frustrations, with some users likening their experience to reporting bugs in a 'kindergarten' environment.
- **Comparison of models like Claude and O1**: A debate has emerged over the effectiveness of different models, with many users stating that Claude is superior to O1, especially in agent mode.
   - Claude is seen as more capable of handling requests, while O1 struggles, leading to preferences for Claude among the user base.
- **Using batch files for managing code changes**: One user developed a batch file tool to keep track of changes in code, creating numbered folders for snapshots and allowing easy comparison with Beyond Compare.
   - This method helps users quickly identify and revert incorrect code changes made by the Cursor IDE, streamlining the debugging process.
- **Experiences and frustrations with testing**: Discussion about testing practices highlighted a mix of attitudes; some users expressed a carefree approach, stating they let 'Jesus take the wheel' when deploying.
   - Others stressed the importance of testing, particularly in languages lacking automatic compilation and testing capabilities.
- **User exploration of Cursor features**: Users discussed their experiences with various Cursor features like agent mode and the implications of upcoming MCP servers.
   - Despite some downtimes, there is a recognition that the slow request system is preferable to having enforced limitations like with other platforms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/sksarvesh007">sksarvesh007 - Overview</a>: sksarvesh007 has 62 repositories available. Follow their code on GitHub.</li><li><a href="https://forum.cursor.com/">Cursor - Community Forum</a>: A place to discuss Cursor (bugs, feedback, ideas, etc.)</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://tailwindflex.com/u/contribution">Tweet from Login</a>: no description found</li><li><a href="https://go.fb.me/h0q0ke">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1328764424286244986)** (2 messages): 

> `Hugging Face search functionality, Issue resolution` 


- **Hugging Face Search Issue Acknowledged**: A known issue with the **in-app Hugging Face search** was reported, and the team indicated they are working with the HF team to resolve it.
   - This issue has been frustrating users, but the team is focused on a quick fix.
- **Hugging Face Issue Resolved**: The team confirmed that the **search issue should now be resolved** thanks to the awesome HF team.
   - *“Awesome HF team ❤️🤗”* reflects the appreciation for the collaboration in resolving the problem.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1328457022961876992)** (72 messages🔥🔥): 

> `Voodoo Graphics Card Nostalgia, Qwen 2.5 versus QwQ Comparisons, Local LLM Recommendations, Generative AI and Game Development, LM Studio Chat vs Development Responses` 


- **Voodoo Graphics Card Nostalgia Reminiscences**: A user fondly recalled how the **Voodoo 1** transformed gaming experiences, showcasing its compatibility with various OpenGL games at a LAN party, leading many to buy the card.
   - They lamented how available online videos don't quite capture their memories of the graphics quality achieved, expressed in nostalgic terms.
- **Mixed Experiences with Qwen 2.5 and QwQ**: A user sought comparisons between **Qwen 2.5 32B Instruct** and **QwQ**, highlighting unclear results based on their initial testing.
   - The general consensus was that **Qwen 2.5** appears to code better and is less verbose, contrasting with issues found in the **QwQ** model.
- **Local LLM Recommendations for Coding**: A member shared insights on testing locally hosted coding models, emphasizing the importance of good GGUF encoding for optimal performance.
   - They recommended specific quantizations and models, and discussed the challenges faced while using high parameter models on consumer hardware.
- **Generative AI Enhancements in Game Development**: Users speculated whether future video games developed with **gen-AI** would arrive with fewer bugs, due to real-time AI bugfixing capabilities.
   - There was excitement at the possibility of features that could apply AI bugfixes across older game titles, ensuring a more polished experience.
- **Inconsistent Responses in LM Studio Interfaces**: One user reported discrepancies between responses in the LM Studio chat and development sections, noting odd behavior in model replies.
   - Another user assured that similar issues were being addressed, pointing to previous discussions on the matter to streamline communication.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/shermanhuman/2b9a82df1bab242a8edffe504bb1867c">Coding local LLM recommendations that meet some minimum useful standard</a>: Coding local LLM recommendations that meet some minimum useful standard - MinimumStandardsLLM.md</li><li><a href="https://huggingface.co/bartowski/Sky-T1-32B-Preview-GGUF">bartowski/Sky-T1-32B-Preview-GGUF · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/VkzO2w6EqK4?si=dONXQA4qc6VCdUvk">The 3Dfx Voodoo Difference: This is why we love them</a>: In this video we will learn why the 3DFX Voodoo is such a special graphics card! 💙 Consider supporting me 💙Patreon: Get exclusive early access, behind the ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1328455306749022301)** (77 messages🔥🔥): 

> `RTX 5090 vs 4090 multi-GPU setups, GPU memory bandwidth impacts, Image generation GPU setups, Dual GPU setups for cost efficiency, Model training complexities` 


- **Discussing Benefits of Adding RTX 5090 to 4090**: Many are curious about the performance implications of adding an **RTX 5090** to an existing **RTX 4090** setup, particularly if the **5090's** superior memory bandwidth could improve processing speeds.
   - However, it was noted that in multi-GPU setups, the **slowest GPU** can bottleneck overall performance, raising questions about effective parallel processing.
- **Memory Bandwidth and VRAM in Multi-GPU Systems**: It's discussed that each **layer of a model** is processed sequentially in multi-GPU setups, which may create synchronization bottlenecks and limit advantages of faster GPUs.
   - This means that even a faster card like the **RTX 5090** might end up idle while waiting for the slower **RTX 4090** to complete its processing.
- **Exploring GPU Configurations for Inference**: Using two **RTX 4090s** can potentially improve inference speed compared to a single **A6000 48GB**, factoring in their higher memory bandwidth and faster processing times.
   - Discussions note that despite performance advantages, power efficiency may favor configurations with **fewer GPUs**, offering similar performance with reduced costs.
- **Challenges of Parallelizing Model Layers**: Participants debated whether it's feasible to split layers across GPUs to achieve parallel inference speed, yet concluded that inherent synchronicities may hinder this approach.
   - The challenges of full synchronization after layer computations create questions about whether the architecture could handle such division effectively in practice.
- **Considerations for Dual GPU Setups**: Many are weighing options for dual **4060ti** setups instead of higher-end cards for better VRAM-per-dollar ratios while addressing power supply limitations.
   - User experiences indicate that while brand differences are minimal, practical concerns such as cooling and physical setup are important when choosing card configurations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.amazon.com/PNY-GeForce-RTXTM-Verto-Graphics/dp/B0CG2MX5H9?ref_=ast_sto_dp&th=1">no title found</a>: no description found</li><li><a href="https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html">SD WebUI Benchmark Data</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1328461997280329864)** (40 messages🔥): 

> `PhD in Statistical Learning Theory, Orch-Or Theory and AI, AI Autocomplete Speed, Developer Opportunities, Off-Topic Channel Dynamics` 


- **PhD candidate seeks alignment insights**: A member announced their upcoming PhD focused on **statistical learning theory** in transformers and expressed eagerness to expand their understanding of **alignment approaches**.
   - They believe that having more **alignment tests and probes** will be beneficial for research.
- **Independent researcher explores Orch-Or Theory**: Another member discussed their focus on **Orch-Or Theory** and its connections to AI, particularly in **noetic science**.
   - They mentioned previous successes with **LLMs** and their openness to share insights and learn from other research domains.
- **Optimization challenge for AI autocomplete**: A user inquired about ways to speed up AI autocomplete performance, specifically for a **Cursor Tab**-like project experiencing **latency issues**.
   - They noted their current use of **groq** and a small model but seek further optimization strategies.
- **Developer offers support for projects**: A full-stack blockchain and AI developer offered their services to assist with projects and team needs for the new year.
   - Another member advised them to check the appropriate channel for project involvement opportunities.
- **Insights on using the off-topic channel**: Members discussed the value of the **off-topic channel**, mentioning it as a hub for both casual and serious discussions.
   - Opinions highlighted that **valuable insights** often emerge from these conversations, emphasizing the channel's impact on the community.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1328511336539488307)** (73 messages🔥🔥): 

> `Eukaryotic Cell Vaults, Research Channel Guidelines, Hyper-connections in Neural Networks, Process Reward Models, VinePPO and CoT Trajectories` 


- **Neoxah discusses eukaryotic cell vaults**: Neoxah shares insights about eukaryotic cell vaults, noting their unique ability to remain coherent in noisy environments, crucial for quantum computing.
   - However, he refrains from revealing specifics in the current chat to maintain confidentiality regarding his research.
- **Guidelines for the Research Channel**: Members emphasize that the research channel is intended for discussing published research and using scientific methods, steering away from personal experiences or credentials.
   - While the term 'published' may be used broadly, the focus remains on ideas supported by accessible write-ups or documentations.
- **Hyper-connections as an alternative to residual connections**: A new method called hyper-connections offers an alternative to traditional residual connections, addressing challenges like gradient vanishing and representation collapse.
   - Initial experiments show that this approach competes well with existing methods in enhancing model performance for both language and vision tasks.
- **Process Reward Models improve reasoning in LLMs**: Research introduces Process Reward Models (PRMs) that aim to enhance mathematical reasoning in large language models by identifying critical tokens that influence outcomes.
   - The study suggests that manipulating these critical tokens can improve model accuracy significantly across various benchmarks.
- **Discussion on VinePPO's application**: There are inquiries about how VinePPO could be applicable in generating chain-of-thought (CoT) trajectories, particularly in the context of LATo.
   - Contrary to initial beliefs, it is clarified that VinePPO does not require examples of CoT trajectories to function effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.06425">Tensor Product Attention Is All You Need</a>: Scaling language models to handle longer input sequences typically necessitates large key-value (KV) caches, resulting in substantial memory overhead during inference. In this paper, we propose Tensor...</li><li><a href="https://arxiv.org/abs/2411.19943">Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM&#39;s Reasoning Capability</a>: Mathematical reasoning tasks pose significant challenges for large language models (LLMs) because they require precise logical deduction and sequence analysis. In this work, we introduce the concept o...</li><li><a href="https://arxiv.org/abs/2409.19606">Hyper-Connections</a>: We present hyper-connections, a simple yet effective method that can serve as an alternative to residual connections. This approach specifically addresses common drawbacks observed in residual connect...</li><li><a href="https://arxiv.org/abs/2501.07542">Imagine while Reasoning in Space: Multimodal Visualization-of-Thought</a>: Chain-of-Thought (CoT) prompting has proven highly effective for enhancing complex reasoning in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Yet, it struggles in complex ...</li><li><a href="https://arxiv.org/abs/2501.06282">MinMo: A Multimodal Large Language Model for Seamless Voice Interaction</a>: Recent advancements in large language models (LLMs) and multimodal speech-text models have laid the groundwork for seamless voice interactions, enabling real-time, natural, and human-like conversation...</li><li><a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>: Process Reward Models (PRMs) emerge as a promising approach for process supervision in mathematical reasoning of Large Language Models (LLMs), which aim to identify and mitigate intermediate errors in...</li><li><a href="https://arxiv.org/abs/2411.07501">LAuReL: Learned Augmented Residual Layer</a>: One of the core pillars of efficient deep learning methods is architectural improvements such as the residual/skip connection, which has led to significantly better model convergence and quality. Sinc...</li><li><a href="https://openreview.net/forum?id=uRHpgo6TMR">Sampling weights of deep neural networks</a>: We introduce a probability distribution, combined with an efficient sampling algorithm, for weights and biases of fully-connected neural networks. In a supervised learning context, no iterative...</li><li><a href="https://www.minimaxi.com/en/news/minimax-01-series-2">MiniMax - Intelligence with everyone</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1328810231123415100)** (6 messages): 

> `Induction Head Bumps, Loss vs Compute, Circuit Interoperability` 


- **Induction Head Bumps' Alignment Questioned**: A user pondered if the **induction head bumps** in scaling law plots align when plotting **loss vs compute**.
   - Another member clarified that it typically occurs after the same number of tokens, indicating the answer is **no**.
- **Analysis of Circuit Interoperability in Training**: A member referenced previous **Anthropic posts** and the **circuit interoperability paper** showing this plot of circuits emerging during training for different Pythia models.
   - They provided a link to the paper [here](https://arxiv.org/abs/2407.10827) and attached a relevant image for further context.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1328510849316552847)** (2 messages): 

> `Neel Nanda podcast, Mechanistic interpretability, Neural network understanding` 


- **Neel Nanda Discusses SAEs in Podcast**: Neel Nanda, a senior research scientist at Google DeepMind, hosted a podcast discussing [SAEs](https://open.spotify.com/episode/5XjHhNQxIb16eJZXGmbaCk?si=Z8LTnSo7QHGJkBxgGZbIJA) and his work in mechanistic interpretability.
   - He highlights the challenge of understanding how neural networks perform complex tasks without a clear understanding of their internal workings.
- **Nanda Reflects on AI's Unique Challenges**: Nanda believes that AI is unique because neural networks can achieve remarkable tasks without human comprehension of their programming.
   - He compares this to having software that accomplishes tasks that no developer can explicitly write.



**Link mentioned**: <a href="https://open.spotify.com/episode/5XjHhNQxIb16eJZXGmbaCk?si=Z8LTnSo7QHGJkBxgGZbIJA">Neel Nanda - Mechanistic Interpretability (Sparse Autoencoders)</a>: Machine Learning Street Talk (MLST) · Episode

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1328473233376153682)** (12 messages🔥): 

> `Pre-commit mixed line ending issue, MLQA benchmark implementation PR, LM Evaluator's majority voting, Filters in LM Evaluation Harness` 


- **Pre-commit Failure on Mixed Line Endings**: A member reported failing a **'mixed line ending'** test during a pre-commit check due to line character differences between Unix and Windows systems.
   - Another member suggested running `pre-commit run --all-files` to automatically fix the issue.
- **MLQA Benchmark Implementation Pull Request**: A member announced the addition of an **MLQA benchmark implementation** and submitted a PR for review.
   - They later highlighted an **AST error** found in the code and requested feedback on it.
- **LM Evaluator Supports Majority Voting**: In response to a question, a member confirmed that the **lm-eval-harness** supports **majority voting** with the ability to set repeats in the configuration.
   - Documentation for setting this up can be found [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/bb098f13b05e361f01a5afe7b612779ce362b3f2/lm_eval/tasks/gsm8k/gsm8k.yaml#L30).
- **Discussion on Filters in LM Evaluation Harness**: A member directed attention to a discussion on implementing filters in the **lm-evaluation-harness**, providing a link for reference.
   - This resource may be useful for those looking to modify existing implementations for specific tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/">EleutherAI</a>: EleutherAI has 156 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/bb098f13b05e361f01a5afe7b612779ce362b3f2/lm_eval/tasks/gsm8k/gsm8k.yaml#L30">lm-evaluation-harness/lm_eval/tasks/gsm8k/gsm8k.yaml at bb098f13b05e361f01a5afe7b612779ce362b3f2 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/6d62a69cb5db963f998c486af6efee43fca63dd3/docs/task_guide.md?plain=1#L57)">lm-evaluation-harness/docs/task_guide.md at 6d62a69cb5db963f998c486af6efee43fca63dd3 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1328621469030289418)** (5 messages): 

> `Growth of AINews Newsletter, Llama 2 Configuration Differences, Gradient Clipping Concerns, Tokenizer Padded Vocabulary Issues, Activation Function Discrepancies` 


- **AINews Newsletter Features a Discussion**: A member shared a humorous note that a recent discussion made it into the **AINews Newsletter**.
   - This highlights the growing engagement and relevance of ongoing topics within the community.
- **Confusion Over Llama 2 Config Intermediate Size**: A member questioned the **padded_vocab_size** value discrepancies between the NeoX and HF config for **Llama 2**, noting significant differences (11008 vs 32768).
   - This reflects confusion over the lack of standardization in configurations among various implementations.
- **Logging Gradient Magnitudes During Runs**: A member inquired about logging **gradient magnitudes** during model runs to detect spikes without using gradient clipping.
   - This indicates a proactive approach to understanding model training behavior and potential issues.
- **Tokenizer Padded Vocabulary Size Quirks**: A member expressed frustration that when setting **padded_vocab_size** to **50304**, it still pads to **50432**, leading to confusion.
   - The log indicated a building process that included **152 dummy tokens**, suggesting possible misconfigurations in processing.
- **HF vs NeoX Activation Function Usage**: Discussion arose over the differences in activation functions, where HF uses **silu** and NeoX uses **swiglu**, aligning with original research.
   - Members scrutinized HF for not including **swiglu** in their activation list, suggesting a potential oversight.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/blob/main/configs/llama2/7B.yml#L9">gpt-neox/configs/llama2/7B.yml at main · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py">transformers/src/transformers/activations.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1328492906670981243)** (15 messages🔥): 

> `Code Removal Issues, Report Issues in Editor, Enabling Diffs, Google Analytics 4 API Integration, Netlify Function Challenges` 


- **Frustration Over Code Removal**: Multiple members expressed frustration about code being removed unexpectedly when making updates, which hampers project progress and causes repetitive work.
   - One noted that changing preamble sections led to the loss of prior content, specifically indicating that it **removed 8 sections** during a simple update.
- **Reporting Issues Made Easy**: A member shared how to report issues directly in the editor via a button next to the 'undo' option, making it convenient for users with ongoing problems.
   - This prompted another user to inquire about reporting methods and to share concerns about persistent functionality issues affecting multiple projects.
- **Enable Diffs to Improve Workflow**: One member activated the Diffs feature, hoping it would reference changes accurately and avoid **removing existing code** with new commands.
   - Another user shared a cautionary note, stating Diffs could malfunction on existing projects, leading to more code being discarded than intended.
- **Integration Challenges with Google Analytics 4**: A user reported difficulties integrating the Google Analytics 4 API in a React/Vite app deployed on Netlify, receiving an 'Unexpected token' error when fetching analytics data post-deployment.
   - Despite local tests working well, they have tried multiple troubleshooting steps, including confirming permission and environment variable settings, without success.
- **Alternative Solutions for GA4 on Netlify**: The same user seeks alternative suggestions for integrating GA4 API with Netlify Functions, considering potential client-side solutions or different analytics services.
   - This reflects a wider interest in finding effective ways to address integration issues encountered within the Netlify environment.


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1328453545543995487)** (108 messages🔥🔥): 

> `Token Consumption Issues, Supabase Project Integration, Chatbot Development Strategies, Subscription Plan Confusions, Bug Reporting and Persistence` 


- **Token Consumption Issues After Subscription Upgrade**: Users reported unexpected increases in token consumption after upgrading to a paid plan, with some struggling to complete basic tasks that worked previously, such as changing UI elements.
   - Concerns were voiced about whether the changes might be related to company policy or system bugs, as one user perceived a significant drop in functionality.
- **Challenges with Supabase Project Management**: Forking a project in Bolt requires the creation of a new Supabase project each time, limiting the ability to reconnect to existing projects, similar to Loveable's functionality.
   - Users are awaiting a solution from the team as this issue disrupts workflow and complicates project management.
- **Creating a Perplexity Style Chatbot**: A member sought advice on integrating a Hugging Face model into Bolt, suggesting an interest in a perplexity-style chatbot, while others recommended using OpenAI’s API directly for ease of use.
   - Discussion highlighted the potential complications of deploying and connecting multiple API platforms.
- **Subscription Plan Confusions**: Users expressed confusion over the subscription upgrade and token allocation process, particularly concerning charges during plan changes and the consequent number of tokens received.
   - Clarifications were suggested regarding the pricing structure and token distribution during transitions between various subscription tiers.
- **Bug Reporting and User Feedback**: Users noted recurring bugs that reappear after updates, contributing to frustration and a perceived lack of responsiveness from the platform regarding these issues.
   - Suggestions were made for users to document errors and consider applying manual checks to ensure system reliability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://support.bolt.new/WebContainer-Startup-Error-159d971055d680fa9af5dafcdb358f42">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team</li><li><a href="https://www.npmjs.com/package/emoji-picker-react">emoji-picker-react</a>: Emoji Picker component for React Applications on the web. Latest version: 4.12.0, last published: 4 months ago. Start using emoji-picker-react in your project by running `npm i emoji-picker-react`. Th...</li><li><a href="https://support.bolt.new/Prompting-Effectively-How-to-talk-to-Bolt-13fd971055d6801b9af4e965b9ed26e2">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team</li><li><a href="https://remix.run/guides/errors">Error Handling | Remix</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1328454835724488745)** (1 messages): 

> `Telegram AI Chatbot, DeVries AI Subscription Model, Multi-Model Access in Telegram` 


- **DeVries AI transforms Telegram into LLM Interface**: The new [DeVries AI](https://devriesai.com/) allows users to converse with **200+ Large Language Models** directly in Telegram for a low cost subscription.
   - *Chat now* with the AI for free as a trial before committing to a subscription.
- **Streamlined AI Access in One Chat**: The DeVries AI chatbot provides access to popular models like **ChatGPT** and **Claude** in a single, familiar Telegram environment.
   - Users can engage in text and soon **image and video generation** using this integrated solution.
- **Affordable AI Subscription Solution**: For just **$24.99/month**, users gain access to all current and future generative AI models without needing multiple subscriptions.
   - This model allows effortless switching between models and early access to new releases.



**Link mentioned**: <a href="https://devriesai.com/">devriesai</a>: Your Telegram AI Agent

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1328491156748632115)** (106 messages🔥🔥): 

> `OpenRouter Provider Setup, Deepseek Performance Issues, OpenRouter Rate Limits, Anime Discussions, MiniMax Model Releases` 


- **How to Become an OpenRouter Provider**: To become a provider and deploy models on OpenRouter, one can reach out via email at `support@openrouter.ai` for assistance.
   - A user creatively speculated about creating a provider that secretly uses OpenRouter, prompting a humorous response about possibly inventing AGI.
- **Deepseek V3 Performance Takes a Hit**: Users reported that **Deepseek** is often slow to respond, with a member stating it takes **7 out of 10 times** to get a reply.
   - It's believed that the slow response times are due to **overload**, with some users suggesting a transition to alternatives like the Together AI endpoint.
- **Navigating OpenRouter's Rate Limits**: New users receive some free credits for testing, but the rate limits depend on the amount of credits purchased, as explained in the [OpenRouter docs](https://openrouter.ai/docs/limits).
   - Several users discussed the lack of an enterprise account, emphasizing the need for higher rate limits for their testing purposes.
- **Anime Series Recommendations Spark Conversation**: A lively discussion emerged about various anime series, particularly the **Fate series**, with users passionately recommending different titles.
   - The conversation humorously shifted to user interactions with anime titles, showcasing shared experiences and preferences.
- **Minimax Model Announcements**: The launch of a new **MiniMax** model with a staggering **456 billion parameters** has generated attention for its impressive context handling capabilities.
   - While not the SOTA in benchmarks, its efficiency and capacity positions it as a potentially valuable tool in the AI landscape.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models?q=free">Models: &#x27;free&#x27; | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://tenor.com/view/dum-suspense-climax-monkey-shocked-gif-8054274">Dum Suspense GIF - Dum Suspense Climax - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: Set limits on model usage</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://en.m.wikipedia.org/wiki/Fate/Zero">Fate/Zero - Wikipedia</a>: no description found</li><li><a href="https://www.reddit.com/r/ClaudeAI/comments/1i0eoje/mentionaiio_get_answers_easily_from_multiple_ai/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/MiniMaxAI/MiniMax-Text-01">MiniMaxAI/MiniMax-Text-01 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/MiniMaxAI/MiniMax-VL-01">MiniMaxAI/MiniMax-VL-01 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1328566927374684180)** (104 messages🔥🔥): 

> `Discord Bot Issues, Flux Sampling with DEIS BETA, Aesthetic Classifier Development, FP8 vs FP16 Precision, Performance of Intel B580 in Stable Diffusion` 


- **Discord Bots Sparking Discussion**: Members expressed concerns over Discord's bot problems, noting that users who join just to say hi often seem like bots themselves.
   - *One member mentioned* that a stricter onboarding could deter bots but might also make it harder for genuine users to join.
- **Recommendations on Using DEIS BETA**: Discussion on utilizing **DEIS BETA** for flux sampling settings emerged, with some members sharing their positive experiences.
   - *One member inquired if others had recommendations* for similar settings or tools to explore.
- **Ideas for Training Aesthetic Classifier**: A member is looking for resources that combine both art and objective ratings for training an **aesthetic classifier**.
   - *Suggestions included using tools like ollama* for assistance in prompting, which might enhance workflow.
- **Debate on FP8 vs FP16 Support**: Members discussed why support for **FP8** is only present in newer cards like the 40 series, questioning potential computational advantages over **FP16**.
   - The conversation revealed that while **lower precision** can save memory, it may not always yield the desired accuracy for all tasks.
- **Performance Concerns with Intel B580**: A member shared struggles posting about their **Intel B580 performance** calculations in stable diffusion due to subreddit filters.
   - *Another member suggested reaching out* to moderators for assistance, switching the conversation towards finding better posting alternatives.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1328573006812811355)** (28 messages🔥): 

> `Qwen2.5-Math-PRM, Claude Sonnet 3.5 performance, MiniCPM-o 2.6 introduction, Midwit model, Evaluation methods in models` 


- **Qwen2.5-Math-PRM takes the stage**: The new Process Reward Model, **Qwen2.5-Math-PRM**, alongside the **ORM**, shows promise for process supervision in LLMs by identifying intermediate errors with remarkable performance on [ProcessBench](https://huggingface.co/papers/2412.06559).
   - For more details, check out their [paper](https://arxiv.org/pdf/2501.07301) outlining their evaluation methods.
- **Claude Sonnet closes the gap**: After **OpenAI's o3 model** achieved **71.7%** on SWE-Bench Verified, the **Claude Sonnet 3.5** using CodeStory hit **62.2%**, showcasing its strong coding capabilities.
   - *A last-gen non-reasoning model getting within 10% of the unreleased future model* highlights how effective Sonnet is at coding.
- **MiniCPM-o 2.6 impresses**: **MiniCPM-o 2.6**, introduced by OpenBMB, is an **8B size Omni Model** that matches GPT-4o capabilities across multiple modalities, featuring **real-time bilingual audio conversations**.
   - Check out the demo and the code on [GitHub](https://github.com/OpenBMB/MiniCPM-o) and [Hugging Face](https://huggingface.co/openbmb/MiniCPM-o-2_6).
- **Midwit model utilizes Claude for scoring**: The **Midwit** model implements **MCTS** with **Claude** as its process reward model, scoring actions from -100 to 100, raising eyebrows about its efficacy.
   - *Seems like that shouldn't work as well as it does* expressed doubt over its performance despite the promising setup.
- **Evaluation methods spark discussion**: The community shared thoughts on eval methods, noting that the **vibe** is that current models may be **undercooked** post-training.
   - The challenges observed include *doom loops* and performance dips in multi-turn conversations, raising questions about evaluation integrity.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenBMB/status/1879074895113621907">Tweet from OpenBMB (@OpenBMB)</a>: 💥 Introducing MiniCPM-o 2.6: An 8B size, GPT-4o level Omni Model runs on device ✨ Highlights: ~Match GPT-4o-202405 in vision, audio and multimodal live streaming ~End-to-end real-time bilingual audio...</li><li><a href="https://x.com/TheXeophon/status/1879254534465490984">Tweet from Xeophon (@TheXeophon)</a>: Vibe-Testing:- May 2024 cutoff date- The HF UI performs better, maybe another model (really unsure)- It really loves to CoT, for everything- High variance in quality because of that- Can at least unde...</li><li><a href="https://x.com/deedydas/status/1877549539781128319?t=hFlLBI6S6s0xaB2ciDeztw&s=19">Tweet from Deedy (@deedydas)</a>: Pretty crazy that after OpenAI o3 hit 71.7% on SWE-Bench Verified, yesterday, Claude Sonnet 3.5 using CodeStory hit 62.2%.A &#34;last-gen&#34; non-reasoning model getting within 10% of the unreleased ...</li><li><a href="https://huggingface.co/MiniMaxAI/MiniMax-Text-01">MiniMaxAI/MiniMax-Text-01 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B">Qwen/Qwen2.5-Math-PRM-72B · Hugging Face</a>: no description found</li><li><a href="https://x.com/teortaxestex/status/1879273615960743995?s=46">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: model collapse due to synthetic data is a complete nothingburger (overrated due to marcusian anti-statistical copes and butthurt artists), compared to the scourge that is TASTE COLLAPSE due to SYNTHET...</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">openbmb/MiniCPM-o-2_6 · Hugging Face</a>: no description found</li><li><a href="https://minicpm-omni-webdemo-us.modelbest.cn/">MiniCPM-omni</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

420gunna: https://x.com/aidan_mclau/status/1878944278782890158
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1328574394892554271)** (31 messages🔥): 

> `AI in Higher Education, CIO Functions, Michigan's Chatbot Offerings, Limitations of LLMs, Stripe Tax Registration` 


- **Navigating AI Talks for CIOs**: A member is preparing for a talk on **AI** for higher-ed CIOs and is considering topics like LLM functionality, prompting, and use cases.
   - Suggestions flow in about addressing **sensitive data management** and aligning discussions to realistic applications of AI.
- **University of Michigan's Chatbot Initiative**: A member highlighted a talk from an admin at the University of Michigan regarding their **custom GenAI tools**, emphasizing equity and privacy.
   - These services are available to the university community, with offerings like **U-M GPT** and **Maizey** for personalized experiences.
- **CIOs Get Real About LLM Limitations**: Discussion emerged about how CIOs often face unrealistic requests for **LLM** capabilities, calling for guidelines to differentiate between feasible use and sci-fi concepts.
   - Attendees suggested clarifying the **costs and resources** involved in different AI implementations could help set appropriate expectations.
- **Taxing Times with Stripe in Europe**: A member shared insights on tax registration in Europe, particularly using **Stripe**'s Non-Union One Stop Shop (OSS) for efficient VAT management.
   - This allows non-EU businesses to register in one country, manage VAT across the **EU**, and simplifies the reporting process.
- **Stripe Simplifies EU Tax Registrations**: Concerns arose about the naming of Stripe's **Non-Union OSS**, which provides a streamlined tax calculation and payment process for outside businesses.
   - The scheme enables these businesses to collect and remit VAT without registering in every EU country, submitting just one annual payment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2025/1/14/24343528/openai-chatgpt-repeating-tasks-agent-ai">ChatGPT can now handle reminders and to-dos</a>: ChatGPT just got a new automation feature for future tasks.</li><li><a href="https://x.com/__tinygrad__/status/1879034330284192050">Tweet from the tiny corp (@__tinygrad__)</a>: We switched from AMD&#39;s driver to our AM driver...and now 4 GPU Llama is faster on red box than green box!</li><li><a href="https://genai.umich.edu">Welcome | U-M Generative AI</a>: no description found</li><li><a href="https://docs.stripe.com/tax/supported-countries/european-union#outside-eu-businesses">Collect tax in the European Union</a>: Learn how to use Stripe Tax to calculate, collect, and report tax in the EU.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1328754889945518182)** (16 messages🔥): 

> `Synthetic Chain-of-Thought Training, O1 Models Discussion, Tulu-R and Vision Generalists, Naming Conventions for Models, Molmo and Reinforcement Learning` 


- **Synthetic Chain-of-Thought Training Underwhelms**: A member expressed disappointment that a paper was just about **synthetic chain-of-thought training data** and **supervised fine-tuning**, lacking any **RL** or similar training.
   - This led to a discussion on the general quality of models linked to **O1** in the title, implying they might be of lesser quality.
- **O1 Models Are Not Going Far**: Another member noted that **O1 mode** would not be happening, implying frustration with the limitations of current models.
   - They preferred the idea of using **Big Molmo** or **Tulu-V** as better alternatives for vision tasks, emphasizing a desire for improved performance.
- **Naming Conventions Draw Critique**: The discussion shifted to naming conventions, where one member likened an undesirable name to **olmo-Flash**, expressing that it sounds cringe.
   - Another example given was **olmo-Gemini**, highlighting how ridiculous naming could detract from credibility.
- **Anticipating High Citations Despite Critique**: One participant speculated that using a name like **O1mo** would still likely earn a considerable number of citations, despite the backlash regarding the name's absurdity.
   - This further emphasizes the disconnection between models' names and their potential reach in academic circles.
- **Molmo and Reinforcement Learning Collaboration**: There was excitement about **Molmo** potentially integrating with **Reinforcement Learning**, raising curiosity about applicable vision tasks.
   - This led to speculations about enhancing vision capabilities through new, more dynamic models.



**Link mentioned**: <a href="https://arxiv.org/abs/2501.06186v1">LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs</a>: Reasoning is a fundamental capability for solving complex multi-step problems, particularly in visual contexts where sequential step-wise understanding is essential. Existing approaches lack a compreh...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1328579778290389096)** (17 messages🔥): 

> `Process Reward Models (PRMs), ProcessBench Benchmarking, Nathan Lambert's Post-Training Insights, LLaMA 3 Paper Discussion` 


- **Exploring Process Reward Models (PRMs)**: The paper on Process Reward Models (PRMs) discusses their promise in addressing intermediate errors in **LLM** reasoning, but notes significant challenges in **data annotation** and **evaluation methodologies**; it found Monte Carlo data synthesis inferior to human annotation methods.
   - The authors highlight potential bias in **Best-of-N evaluation strategies**, where unreliable policy models may generate correct answers but flawed reasoning processes.
- **Introducing ProcessBench for Error Identification**: The **ProcessBench** paper focuses on assessing models' abilities to identify errors in **mathematical reasoning**, presenting **3,400 test cases** with annotated solutions for evaluation.
   - Two types of models, **PRMs** and **critic models**, were tested; the study noted that existing PRMs generally fail to generalize to more complex math problems.
- **Nathan Lambert Shares Insights on Post-Training**: Nathan Lambert shared his thoughts in an article detailing his experiences in **post-training** and reasoning at the **Allen Institute for Artificial Intelligence**, signaling the evolving nature of modern model training.
   - In a December 17 conversation, he elaborated on the steps to train modern models and discussed the commitment to open model training from his organization.
- **LLaMA 3 Paper Appreciation**: A member expressed that the **LLaMA 3 paper** made the path to **TULU 3** clearer, even if they read mostly relevant parts and skimmed through the rest.
   - The difficulty of reading entire papers was highlighted, underscoring how overwhelming the volume of material can be in the current research landscape.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>: Process Reward Models (PRMs) emerge as a promising approach for process supervision in mathematical reasoning of Large Language Models (LLMs), which aim to identify and mitigate intermediate errors in...</li><li><a href="https://arxiv.org/abs/2412.06559">ProcessBench: Identifying Process Errors in Mathematical Reasoning</a>: As language models regularly make mistakes when solving math problems, automated identification of errors in the reasoning process becomes increasingly significant for their scalable oversight. In thi...</li><li><a href="https://open.substack.com/pub/aisummerpodcast/p/nathan-lambert-on-the-rise-of-thinking?r=68gy5&utm_medium=ios">Nathan Lambert on the rise of &quot;thinking&quot; language models</a>: Traditional LLM scaling is &quot;not something that most users of ChatGPT will see any difference from at this point.&quot;
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1328634401718009897)** (3 messages): 

> `Economic Blueprint, Executive Order for Datacenters` 


- **Economic Blueprint aims for AI advancement**: A newly published [Economic Blueprint](https://openai.com/global-affairs/openais-economic-blueprint/) outlines policy proposals for the US to maximize the benefits of AI, enhance national security, and drive economic growth.
   - The blueprint was highlighted by a member in the discussion, who remarked on the ongoing nature of these policy suggestions as part of a familiar narrative.
- **Biden signs executive order for datacenter development**: President Biden signed an executive order allowing the development of gigawatt-scale datacenters on federal land, with both the DoD and DoE leasing land for this purpose.
   - The order stipulates that sufficient **clean energy** must be built on site to match the datacenter's capacity, ensuring an environmentally friendly approach.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/gdb/status/1879059072135414236">Tweet from Greg Brockman (@gdb)</a>: Just published our Economic Blueprint — policy proposals for how the US can maximize AI’s benefits, bolster national security, and drive economic growth.https://openai.com/global-affairs/openais-econo...</li><li><a href="https://x.com/AndrewCurran_/status/1879174379718004881">Tweet from Andrew Curran (@AndrewCurran_)</a>: This morning President Biden signed an executive order opening federal land for development of gigawatt-scale datacenters. The DoD and DoE will both lease land, and sufficient clean energy to match ca...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1328464883150360597)** (82 messages🔥🔥): 

> `ChatGPT Task Automation, Cursor Series B Funding, AI Email Assistants, Model Performance and Capacity Issues, GPU Investment Strategies` 


- **ChatGPT introduces Task Scheduling Feature**: ChatGPT now allows users to schedule one-time and recurring reminders, enhancing its functionality as a digital assistant. The feature is rolling out to Plus, Team, and Pro users, enabling tasks like daily weather updates and reminder notifications.
   - OpenAI aims to position ChatGPT as more than just a chat interface, striving to explore the potential of AI agents with this new Task feature.
- **Cursor Raises Series B with a16z**: Cursor announced their Series B funding co-led by a16z, indicating strong support for their coding tools and AI offerings. This investment underscores the growing interest in AI-powered development tools and potential high returns.
   - The funding round reflects the increasing demand for innovative coding solutions and the positive outlook for Cursor's future developments in the field.
- **Introduction of Ambient AI Email Assistants**: An AI email assistant has been developed that triages and drafts emails without direct user involvement, showcasing a new concept referred to as 'ambient agents.' The technology is open-sourced, allowing wider access for potential applications.
   - This innovation aims to reduce the overhead associated with traditional email management, offering users more efficient and less intrusive email handling.
- **Rate Limiting Issues for Sonnet 3.6 on Cursor**: Users are experiencing rate limiting issues with the Claude model Sonnet 3.6 on Cursor, raising concerns about the capacity for processing requests. Developers indicated that this stems from high traffic that exceeds Anthropic's GPU availability.
   - Cursor is reportedly Anthropic's largest customer, contributing to the pressing need for increased GPU resources to handle demand.
- **Magnetar's Strategy in AI Investment**: Magnetar, a hedge fund, is addressing the challenges in AI startup funding by offering compute resources in exchange for equity, aiming to break the funding bottleneck in the AI sector. This approach represents a novel strategy to support startups that require compute resources before securing investment.
   - Their collaboration with Coreweave highlights the importance of infrastructure in enabling AI advancements and showcases innovative financing solutions in the tech industry.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1879034330284192050">Tweet from the tiny corp (@__tinygrad__)</a>: We switched from AMD&#39;s driver to our AM driver...and now 4 GPU Llama is faster on red box than green box!</li><li><a href="https://x.com/gdb/status/1879059072135414236">Tweet from Greg Brockman (@gdb)</a>: Just published our Economic Blueprint — policy proposals for how the US can maximize AI’s benefits, bolster national security, and drive economic growth.https://openai.com/global-affairs/openais-econo...</li><li><a href="https://www.theverge.com/2025/1/14/24343528/openai-chatgpt-repeating-tasks-agent-ai">ChatGPT can now handle reminders and to-dos</a>: ChatGPT just got a new automation feature for future tasks.</li><li><a href="https://www.reforge.com/blog/ai-native-product-teams">Reforge</a>: no description found</li><li><a href="https://blog.langchain.dev/introducing-ambient-agents/">Introducing ambient agents</a>: Most AI apps today follow a familiar chat pattern (&quot;chat&quot; UX). Though easy to implement, they create unnecessary interaction overhead, limit the ability of us humans to scale ourselves, and ...</li><li><a href="https://x.com/sarahdingwang/status/1879279307119608142">Tweet from Sarah Wang (@sarahdingwang)</a>: Thrilled to announce that @a16z is co-leading the Series B of @cursor_ai. We couldn&#39;t be more excited to continuing working with the Cursor team as they take the world of coding by storm.</li><li><a href="https://x.com/hwchase17/status/1879218872727015644">Tweet from Harrison Chase (@hwchase17)</a>: For the past six months, I haven&#39;t checked email directly, but rather have relied on an AI email assistant to triage and draft emails for meThis is an example of an &#34;ambient agent&#34;We&#39;r...</li><li><a href="https://x.com/swyx/status/1838663794320642328">Tweet from swyx.io (@swyx)</a>: updated for sept 2024 https://x.com/Smol_AI/status/1838663719536201790Quoting AI News by Smol AI (@Smol_AI) it&#39;s notable how predictive the Lmsys Elo vs $ pricing curve is, and how the strategy is...</li><li><a href="https://x.com/ilanbigio/status/1878940258349510764?s=46">Tweet from ilan bigio (@ilanbigio)</a>: announcing our brand new function calling guide @openai!we heard your feedback and made some key changes:- 50% shorter & clearer- new best practices (more on this below 👇)- in-doc function generation...</li><li><a href="https://x.com/kevinweil/status/1879275151969141193">Tweet from Kevin Weil 🇺🇸 (@kevinweil)</a>: 💥 New capabilities in ChatGPT 💥 Now you can schedule things, both one-time and repeating tasks.* Send me the latest AI news at 8am and 5pm each day* Daily weather, horoscopes, etc* Schedule some fun...</li><li><a href="https://x.com/fofrAI/status/1878807358887154044">Tweet from fofr (@fofrAI)</a>: Listen to Kokoro-82M...2m25s of speech generated in 4.5s on a T4https://replicate.com/jaaari/kokoro-82mhttps://replicate.com/p/k3cg51x8vdrga0cmbzj9ttswzgSo graceful, so precise.</li><li><a href="https://x.com/btibor91/status/1876923634675315100">Tweet from Tibor Blaho (@btibor91)</a>: 3 new ChatGPT web app builds were deployed in recent hours- new custom instructions UX (&#34;What should ChatGPT call you?&#34;, &#34;What do you do?&#34;, &#34;What traits should ChatGPT have?&#34; -...</li><li><a href="https://x.com/satyanadella/status/1878578314115473577?s=46">Tweet from Satya Nadella (@satyanadella)</a>: There is no more waitlist for GitHub Copilot Workspace—the most advanced agentic editor. Start building with agents today.</li><li><a href="https://podcasts.apple.com/ca/podcast/how-the-hedge-fund-magnetar-is-financing-the-ai-boom/id1056200096?i=1000679726051&l=fr-CA">How the Hedge Fund Magnetar Is Financing the AI Boom</a>: Épisode de balado · Odd Lots · 2024-12-09 · 50 min</li><li><a href="https://forum.cursor.com/t/anthropic-cannot-sustain-additional-slow-request-traffic-on-claude-3-5-sonnet-please-enable-usage-based-pricing/41361/24?">Anthropic cannot sustain additional slow request traffic on Claude 3.5 Sonnet. Please enable usage-based pricing</a>: We are without a doubt their largest customer.</li><li><a href="https://x.com/BlackHC/status/1878883222911877375">Tweet from Andreas Kirsch 🇺🇦 (@BlackHC)</a>: NeurIPS 2024 PCs being a bunch of clowns 🤡 the state of ML 🙄All you get back a month after raising a concern:</li><li><a href="https://x.com/svpino/status/1878797424590012907">Tweet from Santiago (@svpino)</a>: This is the fastest I&#39;ve seen Llama 3.3 running anywhere!Llama 3.3 70B running at 652 t/s is lightning fast.And if you want Llama 3.1, here are the speeds I was able to get:• Llama 3.1 8B: 1006 t/...</li><li><a href="https://x.com/btibor91/status/1869119330560147690">Tweet from Tibor Blaho (@btibor91)</a>: Remember &#34;Jawbone&#34;? It&#39;s the codename for ChatGPT &#34;Tasks&#34; (&#34;jawbones&#34;) - &#34;Ask ChatGPT to do things in the future&#34;- Choose a date in the future, task name and instru...</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B">Qwen/Qwen2.5-Math-PRM-72B · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>: Process Reward Models (PRMs) emerge as a promising approach for process supervision in mathematical reasoning of Large Language Models (LLMs), which aim to identify and mitigate intermediate errors in...</li><li><a href="https://arxiv.org/abs/2412.06559">ProcessBench: Identifying Process Errors in Mathematical Reasoning</a>: As language models regularly make mistakes when solving math problems, automated identification of errors in the reasoning process becomes increasingly significant for their scalable oversight. In thi...</li><li><a href="https://techcrunch.com/2025/01/14/chatgpt-now-lets-you-schedule-reminders-and-recurring-tasks/">ChatGPT now lets you schedule reminders and recurring tasks | TechCrunch</a>: Paying users of OpenAI&#039;s ChatGPT can now ask the AI assistant to schedule reminders or recurring requests. The new beta feature, called tasks, will start</li><li><a href="https://docs.google.com/spreadsheets/d/1x9bQVlm7YJ33HVb3AGb9qlDNkvTy9CyOFZoah0kr3wo/edit?gid=0#gid=0">LLM elo vs pricing chart</a>: no description found</li><li><a href="https://youtu.be/SN4Z95pvg0Y?si=wyrwJ1VeV2BFElLG">How AI Took Over The World</a>: One insight changed everything... intelligence can emerge from pattern prediction. This is a capstone video featuring key insights from the entire AI series....</li><li><a href="https://www.minimaxi.com/en/news/minimax-01-series-2">MiniMax - Intelligence with everyone</a>: no description found</li><li><a href="https://www.gov.uk/government/publications/ai-opportunities-action-plan/ai-opportunities-action-plan">AI Opportunities Action Plan</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1328466097116282910)** (2 messages): 

> `Reading list on HN, User achievements` 


- **HN front page features reading list**: A user shared excitement over a **reading list on the front page of Hacker News** with a screenshot linked [here](https://cdn.discordapp.com/attachments/1075282504648511499/1328466096898048031/Screenshot_2025-01-13_at_12.49.35_PM.png?ex=67881f77&is=6786cdf7&hm=b3feed5680389cbe58f5d83bd76ec7564561c423a8af357ed9b8ae21cd1c5730&).
   - This discussion highlights the importance and interest in curated content in the community.
- **User celebrates personal milestone**: A member humorously exclaimed, **"FINALLY MADE IT lmao"**, indicating their excitement over a personal achievement.
   - This reflects the playful nature of the community's interactions and shared experiences.


  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1328458497070731334)** (1 messages): 

> `Audio Overviews Feedback Survey, User Research Participation` 


- **Participate in Audio Overviews Feedback Survey**: The Google team is seeking feedback on **Audio Overviews** through a quick 5-minute screener, accessible via [this form](https://forms.gle/NBzjgKfGC24QraWMA).
   - Those who complete the survey will receive a **$10 gift code** via email as a thank you for their input.
- **Details on User Research Participation**: Eligible participants must be at least 18 years old to take part in the survey, which will enhance future product improvements based on user needs.
   - It's important to note that the incentive is only awarded for completing the full survey, not for filling out the interest form.



**Link mentioned**: <a href="https://forms.gle/NBzjgKfGC24QraWMA">Register your interest: Google feedback survey</a>: Hello,We are looking for feedback on NotebookLM via a short survey. This will help the Google team better understand your needs in order to incorporate them into future product enhancements. To regist...

  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1328463586476949614)** (29 messages🔥): 

> `AI-generated podcasts, Organizing notebooks in Google Classroom, Length control for generated podcasts, Audio transcription options, Sharing permissions for NotebookLM` 


- **AI-generated podcasts share site**: A member discussed a new platform, [Akash](https://akashq.com), designed for **easy sharing of AI-generated podcasts**, eliminating the need for account permissions.
   - It allows users to upload, share, and distribute podcasts created from platforms like NotebookLM seamlessly.
- **Not organizing notebooks in Google Classroom**: A user expressed frustration over the inability to **categorize notebooks** in NotebookLM within Google Classroom, suggesting the need for folder creation.
   - They proposed that more organization features would improve user experience and efficiency.
- **Transcription options for audio files**: A user found a workaround for transcribing audio by uploading files as sources, but suggested that a direct transcription button would be more efficient.
   - This idea aims to streamline the process of converting audio to text within NotebookLM.
- **Controlling podcast lengths**: A member inquired about methods to limit the **length of generated podcasts**, seeking options for a desired duration.
   - Another participant shared a [Reddit link](https://www.reddit.com/r/notebooklm/comments/1gmtklt/longest_pod_i_have_ever_got/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) suggesting that users had some success controlling output length.
- **Embedding speeches in audio format**: Another user shared audio files of **Martin Luther King Jr.'s 'I Have a Dream'** speech and the **Gettysburg Address**, emphasizing their availability.
   - This reflects an interest in using the platform for historical or literary speeches through audio delivery.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://akashq.com">Akas: home to AI podcasts</a>: no description found</li><li><a href="https://notebooklm.google.com/notebook/6dd1946b-561b-446c-818a-e9e17e332aac/audio">no title found</a>: no description found</li><li><a href="https://www.reddit.com/r/notebooklm/comments/1gmtklt/longest_pod_i_have_ever_got/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.akashq.com/post/51eae1b7-e011-4d66-83af-873d763a203d">What happened on Jan 14?</a>: What happened on Jan 14? by This Day in History</li><li><a href="https://www.akashq.com/post/2e2231bf-907b-4805-84ae-71f8a7a45c19">What happened on Jan 13?</a>: What happened on Jan 13? by This Day in History
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1328455331592016012)** (52 messages🔥): 

> `NotebookLM functionalities, Paid version features, User feedback sessions, Audio summary issues, Sharing projects` 


- **Discussing ways to enhance NotebookLM functionalities**: Users are exploring options to add customization features, like modifying podcast dialogues or changing voices and personalities.
   - It is suggested to use the custom instructions as a workaround, though many features are yet to be fully supported.
- **Query on the features of the paid NotebookLM version**: A user inquired if the paid version would allow sharing projects publicly without requiring individual access permissions.
   - It was confirmed that currently, public sharing is not available, but internal sharing could be done within an organization.
- **Feedback on audio summary generation bugs**: Concerns were raised about the audio overview feature not maintaining links to uploaded PDF documents in generated summaries.
   - It was noted that this functionality is currently broken and the team is working on restoring it.
- **Exploration of NoCode RAG solutions**: A user expressed interest in finding NoCode solutions for generating answers from PDFs stored in Google Drive.
   - The community acknowledged the complexity of integrating such features with NotebookLM.
- **User experiences with NotebookLM**: Users shared their positive experiences with NotebookLM for research tasks but highlighted areas needing improvement.
   - Suggestions included better citation exporting and fixing current bugs related to document summaries.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://thedrive.ai?">The Drive AI: Revolutionizing File Management &amp; Knowledge Bases</a>: Discover The Drive AI&#x27;s breakthrough in smart file organization. Our platform transforms your files into a dynamic knowledge base with the help of cutting-edge AI. Elevate your business operation...</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://www.reddit.com/r/notebooklm/comments/1gmtklt/longest_pod_i_have_ever_got/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1328459603049971774)** (74 messages🔥🔥): 

> `Perplexity Pro Features, User Experience Issues, Coding Assistance Frustrations, API Access Limitations, Content Visibility Concerns` 


- **Perplexity Pro's Mixed Reception**: Some users expressed dissatisfaction with **Perplexity Pro**, feeling it isn't effective for coding help despite being beneficial for research.
   - Concerns were raised about the inability to use redeemed rewards codes and access the Pro features seamlessly.
- **UI Changes and User Control**: Members complained about unwanted ads and changes in the Perplexity interface, feeling it detracts from usability.
   - One user suggested using **Ublock Origin** to eliminate distracting content, indicating frustration with the platform's direction.
- **Frustrations with Coding Assistant**: A user detailed frustrations with the coding assistant repeatedly asking for confirmation on providing complete code when it should not.
   - Despite providing specific instructions, the AI continues to ignore user requests, leading to increased frustration.
- **Request for API Access and Features**: Questions arose regarding whether **Pro search** can now be accessed via API, with users eager for updates on this functionality.
   - Responses indicated that Pro search is not currently supported via API, frustrating those looking to integrate it into their workflows.
- **Concerns over Content Visibility**: Users raised concerns about their published pages being private and not indexed by Google, questioning if the **enterprise subscription** affects visibility.
   - One user noted the inability to access documents they previously uploaded, making it difficult to utilize content effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai">no title found</a>: no description found</li><li><a href="https://apps.microsoft.com/detail/9P02M3TKS0RJ?hl=ko&gl=KR&ocid=pdpshare">AI Search - powered by perplexity - Free download and install on Windows | Microsoft Store</a>: Powerful AI search based on the perlexity API. Where does knowledge start. The answer you need is right at your fingertips.&#xA;Put aside all the noise and get reliable and up-to-date answers directly...</li><li><a href="https://newsletter.moneylion.com/subscribe?ref=yJmsSyv2l7">MoneyLion Markets Daily Newsletter</a>: Your Daily Dose of Market News
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1328694498729000972)** (2 messages): 

> `TikTok regulations, German summary requests` 


- **Chinese Officials Eye TikTok Regulations**: Chinese officials are considering new regulations surrounding **TikTok**, focusing on its impact and control measures as highlighted in the article [here](https://www.perplexity.ai/page/chinese-officials-consider-tik-51PEvvekQxqVuhd74SkSHg).
   - The discussions hint at a growing concern over **content moderation** and **data privacy** issues related to the platform.
- **User Seeks Summary in German**: A user requested a **summary** of the information available on a particular topic in German, as referenced in the discussion [here](https://www.perplexity.ai/search/kannst-du-mir-eine-zusammenfas-aZ8WnXOORK.hH5cZSr0qqg).
   - The specific details of the request underline the need for **multilingual support** in information dissemination.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1328454395481817118)** (69 messages🔥🔥): 

> `Claude's personality and training, Open source models limitations, Defensive practices in AI data, Quality of training data and its impact, Human-like interactions in models` 


- **Claude showcases distinct personality traits**: Members noted that **Claude** has a personality that some find annoying yet distinctive, contributing greatly to its coding abilities until recently overshadowed by **DeepSeek v3**.
   - *Anthropic's ongoing post-training adjustments may have kept Claude as one of the best coders among AI models.*
- **Open source models struggle against proprietary data**: There is skepticism about the future of open source models like Claude, with many believing that proprietary training data ultimately produces better results.
   - *The conversation underscored that overcoming the challenge of synthetic data requires more than just enhanced algorithms, emphasizing the influence of the underlying dataset.*
- **Defensive practices in AI training**: The group discussed the growing trend of defense companies seeking AI services, with some members suggesting that government-backed data centers could address data access inequalities.
   - *Concerns arose about the potential monopoly on AI development due to companies leveraging superior data for training.*
- **Evaluating training data quality and its implications**: Conversation highlighted the notion that while Anthropic may not have the most data, their overall quality could potentially rival that of tech giants such as Meta and other Chinese firms.
   - *Participants recognized that simply having a large quantity of data does not equate to better model performance.*
- **The allure of human-like interactions in AI**: Some members expressed admiration for how **Claude** engages with users, noting an intentional design behind its human-like interactions, distinguishing it from models focused solely on intelligence.
   - *The committee reflected that Claude's design philosophy may enhance user experience by creating a more empathetic AI.*



**Link mentioned**: <a href="https://tenor.com/view/never-give-up-fist-pump-motivation-motivational-gif-4892451">Never Give Up Fist Pump GIF - Never Give Up Fist Pump Motivation - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1328821149030944900)** (3 messages): 

> `Gemini for Data Extraction, 4o-mini and Llama-8B, Jina for Text Conversion` 


- **Gemini excels in data extraction**: A member highlighted that **Gemini** is excellent for **data extraction**, showcasing its effectiveness in this role.
   - This claim indicates a growing confidence in Gemini's capabilities for handling such tasks.
- **Evaluating 4o-mini and Llama-8B's reliability**: Another member suggested that while **4o-mini** and **Llama-8B** might work for data tasks, there is less trust in them delivering the **exact original content**.
   - This reflects a cautious approach when selecting models for data extraction.
- **Exploring Jina for text conversion**: A member proposed using **Jina** to convert data to text if it is formatted in a specific way, suggesting a programmatic approach might yield results.
   - This idea points towards the versatility of **Jina** and the potential for innovative data processing methods.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

real.azure: It seems to be the month of attention alternatives. 

https://arxiv.org/abs/2501.06425
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

real.azure: It seems to be the month of attention alternatives. 

https://arxiv.org/abs/2501.06425
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1328468365995544797)** (57 messages🔥🔥): 

> `.aider.conf.yml exclusion from .gitignore, DeepSeek v3 performance on local machines, API configurations for custom hosts, Recommendations for open-source models, Benchmarking individual models` 


- **Debate on .aider.conf.yml in .gitignore**: A member expressed that **.aider.conf.yml** should not be excluded by default from **.gitignore**, emphasizing its importance for team decisions.
   - Another member countered that excluding it simplifies individual setups and avoids team conflicts.
- **Running DeepSeek v3 requires significant resources**: Discussion revealed that to effectively run **DeepSeek v3**, users might need around **380GB of RAM** and multiple **GPU cards**.
   - Users explored alternatives like **Qwen** and other smaller models to mitigate hardware limitations.
- **API configuration for custom endpoints**: A clear method was shared for configuring Aider to utilize custom API endpoints, utilizing the **OPENAI_API_BASE** variable.
   - Members exchanged commands for testing API interactions successfully.
- **Open-source model recommendations**: Members recommended exploring smaller models like **Qwen 32b** for local running due to their lower resource requirements.
   - The conversation highlighted the trade-off between performance and hardware capabilities when selecting models.
- **Performance inquiry for Gemini models**: A participant requested information regarding the best open-source model for user story creation based on set requirements.
   - General feedback pointed toward the **Gemini** model, praising its effectiveness for specific tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider is AI pair programming in your terminal</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1328741524296437760)** (6 messages): 

> `Aider Error Handling, Toggling Edit Formats, Copy Context Command Limitations` 


- **Aider struggles with LLM edit formats**: A user reported frequent occurrences of the error: **'The LLM did not conform to the edit format'** while using aider with **gpt-4o**, seeking advice on how to resolve such issues.
   - A response indicated that this issue occurs when the LLM misunderstands system prompts and suggested reducing the number of files added to the chat to avoid confusion.
- **Toggling Whole Edit Format in Aider**: A user asked if they could toggle the **whole edit format** from within aider to address search/replace bugs without doing it via command line.
   - The indication was that users can indeed switch to whole edit mode by using the command `/chat-mode whole`.
- **Inconsistency in Copy Context Command**: A user inquired why the **/copy-context** command does not include commands from the chat, such as Linux or Python commands, while others like **/ask** and **/code** do.
   - This raised questions about the functionality and completeness of the copy context feature compared to other commands.



**Link mentioned**: <a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: aider is AI pair programming in your terminal

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1328782458832814162)** (5 messages): 

> `Machete kernels across multiple GPUs, lm_head quantization issues, Bitpacking incompatibilities, Model monkey-patching` 


- **Progress of Machete Kernels Limited to Single GPU**: Discussion highlighted that the current implementation of **Machete kernels** in vLLM only supports **tp=1**, as noted in a [GitHub comment](https://github.com/vllm-project/llm-compressor/issues/973#issuecomment-2536261163).
   - Additionally, **NeuralMagic's blog** suggests a potential for improvements but highlights that the capabilities across multiple GPUs are still an area of exploration.
- **lm_head Quantization Concerns**: One member mentioned their **lm_head** appears **unquantized**, raising questions about its configuration with the used **Llama model** ([link](https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4/tree/main)).
   - They reported encountering an error in a specific line of the vLLM code while referencing this [code line](https://github.com/vllm-project/vllm/blob/87054a57ab39bad6c7fe8999e7d93566ded713e3/vllm/model_executor/layers/quantization/kernels/mixed_precision/machete.py#L28-L31).
- **Incompatibility Issues Due to Bitpacking**: A member suggested that the issue might stem from **bitpacking** incompatibility with tensor parallel across the first dimension, affecting performance.
   - They recommended downgrading to **Marlin** as a potential fix for the encountered issues.
- **Suggestion for Monkey-Patching**: It was proposed to **monkey-patch** a portion of the model code, specifically an initialization file, as a workaround to load the model successfully.
   - This part can be found at this [GitHub link](https://github.com/vllm-project/vllm/blob/87054a57ab39bad6c7fe8999e7d93566ded713e3/vllm/model_executor/layers/quantization/kernels/mixed_precision/__init__.py#L14-L19).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/hugg">Hugg (Yy)</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/blob/87054a57ab39bad6c7fe8999e7d93566ded713e3/vllm/model_executor/layers/quantization/kernels/mixed_precision/__init__.py#L14-L19">vllm/vllm/model_executor/layers/quantization/kernels/mixed_precision/__init__.py at 87054a57ab39bad6c7fe8999e7d93566ded713e3 · vllm-project/vllm</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://github.com/vllm-project/vllm/blob/87054a57ab39bad6c7fe8999e7d93566ded713e3/vllm/model_executor/layers/quantization/kernels/mixed_precision/machete.py#L28-L31).">vllm/vllm/model_executor/layers/quantization/kernels/mixed_precision/machete.py at 87054a57ab39bad6c7fe8999e7d93566ded713e3 · vllm-project/vllm</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://github.com/vllm-project/llm-compressor/issues/973#issuecomment-2536261163).">ValueError: Failed to find a kernel that can implement the WNA16 linear layer. · Issue #973 · vllm-project/llm-compressor</a>: Describe the bug Unable to successfully deploy the GPTQ-W4A16 quantized model (Qwen-72B) using vllm, where only the FFN part of the model is quantized. Environment Package Version -----------------...</li><li><a href="https://neuralmagic.com/blog/introducing-machete-a-mixed-input-gemm-kernel-optimized-for-nvidia-hopper-gpus/)">Introducing Machete: Optimized GEMM Kernel for NVIDIA Hopper GPUs</a>: Machete, Neural Magic’s optimized kernel for NVIDIA Hopper GPUs, achieves 4x memory savings and faster LLM inference with mixed-input quantization in vLLM.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1328579814227181651)** (1 messages): 

> `Blockwise Matrix Multiplication, Linear Layer Kernel` 


- **Inquiry on Blockwise Matmul Kernel**: A member asked if a shared kernel for **blockwise matmul** in a linear layer is decent, providing a link to the [linear layer kernel](https://cdn.discordapp.com/attachments/1189607595451895918/1328579813891641374/linear_layer_kernel.txt?ex=6787e09f&is=67868f1f&hm=cf29d6ada51bf2b4daac4239743fec6768b9a487bd5b0faf24f785b48eddbafa&).
   - The request prompts discussion on the effectiveness of the kernel for optimizing performance in linear layers.
- **Discussion Expected on Kernel Performance**: The sharing of the kernel file signals expectations for feedback on its **performance** and potential areas of improvement.
   - Members are likely to compare this kernel against existing implementations in blockwise matrix multiplication.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1328546607746256951)** (3 messages): 

> `FA3 Profiling, H200 vs H100 Performance` 


- **FA3 performance differences on H200 and H100 debated**: A user raised the question of whether there is a **significant performance gain** when profiling **FA3** on **H200** compared to **H100**.
   - Another user confirmed that there is indeed a **significant difference** between **FA3** and the prior **FA2** model, but the discussion remained focused on the specific comparisons to H200.
- **FA2 significant difference acknowledged**: A user noted the established **significant difference** between **FA3** and **FA2**, confirming improvements in the latter’s performance.
   - This exchange indicated a community awareness regarding the performance metrics of the latest models in relation to each other.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1328601182918152294)** (2 messages): 

> `torch.cuda.max_memory_allocated, torch.cpu, Memory allocation functions` 


- **No CPU Equivalent for Memory Allocation**: A member noted the absence of a CPU equivalent for `torch.cuda.max_memory_allocated()`, indicating a gap in functionality.
   - *This raises questions* about memory allocation tracking for CPU usage in **PyTorch**.
- **Concerns Over torch.cpu Functionality**: Another member expressed dissatisfaction with `torch.cpu`, suggesting it is *quite lacking* in features.
   - This sentiment may highlight a need for enhancements in **CPU functionalities** within the **PyTorch** library.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1328846464159453194)** (2 messages): 

> `MiniMax-01 Open Source, Lightning Attention Architecture, Ultra-Long Context, Cost-Effectiveness of MiniMax Models` 


- **MiniMax-01 Open Source Launch**: MiniMax-01, a series of groundbreaking open-source models, has been launched, including **MiniMax-Text-01** and **MiniMax-VL-01**.
   - The announcement was made via [MiniMax's Twitter](https://x.com/minimax__ai/status/1879226391352549451) highlighting the models' innovative capabilities.
- **Revolutionary Lightning Attention Architecture**: The **Lightning Attention** mechanism featured in MiniMax-01 represents a significant departure from traditional Transformer architectures, showcasing large-scale implementation.
   - This architecture aims to enhance model performance well beyond existing capabilities.
- **Processing Up to 4M Tokens Efficiently**: MiniMax-01 supports processing of up to **4M tokens**, dramatically surpassing the capacity of leading models by **20 to 32 times**.
   - This feature is positioned to meet the growing needs of AI agent applications for extended context handling.
- **Highly Competitive Pricing Strategy**: MiniMax-01 models are being introduced at a cost-effective rate, priced at **USD $0.2 per million input tokens** and **USD $1.1 per million output tokens**.
   - The pricing reflects optimizations in both model architecture and infrastructure, promoting continuous innovation.
- **Access Your Free Trial Now**: Interested users can now try MiniMax-01 models for free via [Hailuo AI](https://hailuo.ai).
   - Further details and insights can be found in the comprehensive [paper on MiniMax-01](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf).



**Link mentioned**: <a href="https://x.com/minimax__ai/status/1879226391352549451">Tweet from MiniMax (official) (@MiniMax__AI)</a>: MiniMax-01 is Now Open-Source: Scaling Lightning Attention for the AI Agent EraWe are thrilled to introduce our latest open-source models: the foundational language model MiniMax-Text-01 and the visua...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

remek1972: New one: https://salykova.github.io/sgemm-gpu
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1328736812541411421)** (2 messages): 

> `Kaiko AI job openings, Foundation Models at Prior Labs` 


- **Kaiko AI seeks Senior ML and Data Engineers**: Kaiko AI is hiring for **Senior ML Platform Engineer** and **Senior Data Engineer** positions in Amsterdam and Zurich, focusing on developing Foundation Models to enhance cancer treatments ([Job Posting for ML Engineer](https://jobs.kaiko.ai/jobs/4829222-senior-ml-platform-engineer), [Job Posting for Data Engineer](https://jobs.kaiko.ai/jobs/5007361-senior-data-engineer)). Notably, they do not offer visa sponsorship.
- **Exciting opportunities at Prior Labs**: Prior Labs, a well-funded startup, is building Foundation Models for **tabular data**, **time series**, and **databases**, and is actively recruiting skilled ML engineers ([Research Article](https://www.nature.com/articles/s41586-024-08328-6)). The work is expected to have a broad impact across various fields, from **healthcare** to **finance**, while allowing engineers to innovate and speed up model performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jobs.kaiko.ai/jobs/4829222-senior-ml-platform-engineer">Senior ML Platform Engineer - Kaiko</a>: About kaiko In cancer care, treatment decisions can take many days—but patients don’t have that time. One of the reasons for delays? Cancer patients&#39; data is scattered across many places: docto...</li><li><a href="https://jobs.kaiko.ai/jobs/5007361-senior-data-engineer">Senior Data Engineer - Kaiko</a>: About kaiko In cancer care, treatment decisions can take many days—but patients don’t have that time. One of the reasons for delays? Cancer patients&#39; data is scattered across many places: docto...</li><li><a href="https://www.notion.so/priorlabs/ML-Engineer-Foundation-Models-Freiburg-Berlin-London-1425be1f3b4980598ef0faa9e47ec0e1">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1328508190417096726)** (8 messages🔥): 

> `CUDA atomic functions, CFD solver development` 


- **Discussion on CUDA Atomic Functions for Doubles**: Members discussed the lack of a CUDA atomic function specifically for finding the minimum of elements of **double** type, which currently only has implementations for integers.
   - One member suggested using the integer atomic function for positive doubles, while another shared a [Stack Overflow answer](https://stackoverflow.com/a/72461459) that addresses handling negative zero issues.
- **Building a CFD Solver on GPU**: A member shared their project of building a **CFD solver** utilizing GPU for enhanced parallelization efficiency.
   - This highlights the growing interest in applying advanced computational techniques in fluid dynamics.



**Link mentioned**: <a href="https://stackoverflow.com/a/72461459">How do I use atomicMax on floating-point values in CUDA?</a>: I have used atomicMax() to find the maximum value in the CUDA kernel:&#xA;&#xA;__global__ void global_max(float* values, float* gl_max)&#xA;{&#xA;    int i=threadIdx.x &#x2B; blockDim.x * blockIdx.x;&...

  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1328662257298702348)** (1 messages): 

> `DeepSeek 2.5 reasoning, Image analysis` 


- **DeepSeek 2.5 showcases impeccable reasoning**: A discussion highlighted the **impeccable reasoning** demonstrated by **DeepSeek 2.5** in relation to a shared task.
   - Members seem impressed with how the model processed the material, suggesting robust capabilities in reasoning.
- **Image linked for analysis**: An image was shared for analysis related to the discussion on DeepSeek 2.5's performance.
   - It appears to contain relevant data that supports the claims made about the model's reasoning skills.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1328712225027915827)** (6 messages): 

> `int8_weight_only, torch.compile and optimization, torchao compatibility with TorchScript/ONNX` 


- **int8_weight_only utilizes fused operations**: A question was raised about whether **int8_weight_only** uses a custom kernel for dequantization and matrix multiplication.
   - It was confirmed that **torch.compile** would combine these operations effectively.
- **Fusing operations with torch.compile**: It was clarified that **torch.compile** indeed fuses the dequantization and matrix multiplication operations sequentially.
   - This optimization ensures enhanced performance during model execution.
- **torchao's utility with TorchScript and ONNX**: An inquiry about the compatibility of **torchao** with **TorchScript/ONNX** was made, confirming its functionality.
   - To export a graph, members discussed using the [torch.export functionality](https://github.com/pytorch/ao/tree/main/torchao/quantization#workaround-with-unwrap_tensor_subclass-for-export-aoti-and-torchcompile) with **torch.compile**.



**Link mentioned**: <a href="https://github.com/pytorch/ao/tree/main/torchao/quantization#workaround-with-unwrap_tensor_subclass-for-export-aoti-and-torchcompile">ao/torchao/quantization at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1328640191115821106)** (3 messages): 

> `GTC Attendance, CUDA Talks, Networking at GTC` 


- **GTC Attendance Sparked Interest**: A member inquired, *'Anyone coming to GTC?'*, expressing a desire to connect with others at the event.
   - Another member confirmed their attendance, saying, *'I'll be there.'*
- **Catch-Up Plans**: The original poster mentioned their usual routine of hanging around **CUDA-related talks** at GTC and trying to meet attendees randomly.
   - They candidly noted that their attempts to network sometimes don't yield results.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1328847355726004264)** (1 messages): 

> `MPS Profiler, Kernel Profiling, GPU Trace Issues, PyTorch and MPS, MTL_CAPTURE_ENABLED` 


- **MPS Profiler struggles to capture GPU traces**: A user reported difficulties with the **MPS Profiler** not capturing GPU traces during kernel profiling despite using `getMPSProfiler().startCapture()` and `getMPSProfiler().stopCapture()` around the dispatch to the GPU.
   - They ensured the **MTL_CAPTURE_ENABLED=1** environment variable was set, but still received an empty trace from **Xcode**.
- **PyTorch interaction with MPS Profiler**: There was mention of how this issue occurs specifically when running the operation with **PyTorch** from **Python**, raising questions about possible integration problems.
   - The user is seeking solutions or insights into whether there's a common misstep when using **MPS** with **PyTorch** in this context.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1328839217790586880)** (1 messages): 

> `Thunder Compute, Cloud GPU Pricing, CLI Instance Management` 


- **Thunder Compute launches for cheaper cloud GPUs**: Co-founder announced the launch of [Thunder Compute](https://thundercompute.com) focused on making **cloud GPUs** cheaper and easier to use, especially with **A100s** priced at **$0.92/hr**.
   - The team is in **beta testing** and offers users **$20/month free** to gather feedback on the service.
- **Easy CLI for hassle-free instance management**: Users can manage cloud instances effortlessly using the CLI (`pip install tnr`), allowing for automated instance creation with just one command.
   - This feature simplifies the onboarding process, aiming to make complex configurations manageable for everyone.
- **Backed by top cloud infrastructures**: Instances are hosted in **US central regions** of **GCP** and **AWS** to ensure optimal uptime and performance.
   - This setup reinforces the reliability and scalability essential for machine learning workflows.



**Link mentioned**: <a href="https://thundercompute.com">Thunder Compute: Low-cost GPUS for anything AI/ML</a>: Train, fine-tune, and deploy models on Thunder Compute. Get started with $20/month of free credit.

  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1328833651156455445)** (1 messages): 

> `Onboarding Documentation` 


- **Basic Onboarding Documentation Added**: Onboarding documentation has been added to TK, and you can check it out [here](https://docs.google.com/document/d/15-Zvf6e0NLX1si4ml4sUOWCDlXNMtOWKiuo6CKZMEYA/edit?usp=sharing).
   - Comment-mode is on, inviting feedback for any unclear sections or additional issues that need addressing.
- **Feedback Encouraged for Documentation**: Members are encouraged to provide feedback on the onboarding documentation to clarify any unclear points. This proactive approach aims to enhance the onboarding experience for future users.



**Link mentioned**: <a href="https://docs.google.com/document/d/15-Zvf6e0NLX1si4ml4sUOWCDlXNMtOWKiuo6CKZMEYA/edit?usp=sharing">TK onboarding</a>: Summary This document specifies how to get started on programming kernels using TK. Please feel free to leave comments on this document for areas of improvement / missing information.   Summary	1 Back...

  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1328522006064271455)** (4 messages): 

> `Nvidia Cosmos, Jetson, Transformer Engine, Mamba, 3D Vision Stack` 


- **Nvidia Cosmos operates seamlessly on Jetson**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/johnnycano_nvidia-cosmos-nvidiacosmos-activity-7283774665943109632-VQDa?utm_source=share&utm_medium=member_ios) highlighting that **Nvidia Cosmos** is successfully running on **Jetson** devices.
   - This development opens up new applications in various AI-powered projects.
- **Successful Port of Transformer Engine and Libraries**: A member reported that they have **ported the Transformer Engine** along with **more than 30 libraries** to enhance compatibility and performance.
   - This includes notable additions like **Mamba** and the **3D vision stack**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1328501686200700950)** (20 messages🔥): 

> `New codestral model, ChatGPT 4o with Canvas, OpenAI model changes, AI Misalignment Videos, App token issues` 


- **Introducing the codestral model**: The new model called **codestral** is available for free on the **Mistral API** and features **256k context**, boasting speed and efficiency.
   - *It is described as 'stupid fast and good'* by users discussing its capabilities.
- **Understanding ChatGPT 4o with Canvas**: Members are confused about whether **ChatGPT 4o with Canvas** is simply the previous model with canvas integration or a new variant altogether.
   - Some believe it uses the old model from September and was designed to aid user interactions with canvas features.
- **OpenAI model mysteries**: Community members express confusion regarding OpenAI's implementation of model updates, particularly concerning the canvas feature.
   - One user noted that previous interactions with the **4o canvas** model have now reverted to **4o mini**.
- **AI Misalignment Discussion**: A member shared a [YouTube video](https://www.youtube.com/watch?v=K8p8_VlFHUk) featuring animations that focus on **AI misalignment**, igniting interest in the topic.
   - This prompted questions about the content of the video and its relevance to ongoing AI discussions.
- **Token-related app issues**: After experiencing app functionality issues, a user mentioned that the app closed, but resolved upon reopening, indicating possible **token exhaustion**.
   - Another user suggested this behavior might be indicative of a crash rather than a typical token limitation.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages): 

jlgri: Will there ever be a chatgpt app on apple watches?
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1328465845529214979)** (8 messages🔥): 

> `Data Format Discussions, PDF Limitations, Assistant Response Issues, Rephrasing Techniques` 


- **Discussing Options for Data Formats**: Members are exploring better data formats, with suggestions including **JSON**, **YAML**, and plain text, while criticizing **PDF** for its limitations.
   - *One member quipped*, 'PDFs are not an API 😂.'
- **Rethinking Assistant's Documentation References**: A member is seeking solutions to prevent their assistants from constantly referencing documentation at the end of each result, and they were advised to implement specific code.
   - They asked the community for help on where to implement the code in the [OpenAI playground](https://platform.openai.com/).
- **Tips on Simplifying Language for Non-Native Speakers**: Another member shared their own **de-GPTing** prompt focused on rephrasing text to make it readable for second language speakers by avoiding rare words and complex structures.
   - They emphasized not rephrasing field-specific terminologies to maintain context.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1328465845529214979)** (8 messages🔥): 

> `Improving Data Formats, Reducing PDF Usage, Assistant Behavior Customization, User-Friendly Rephrasing` 


- **Finding Better Data Formats**: Members discussed how to improve data usability, with suggestions to ask for data in formats better than PDF, such as **JSON, YAML**, or even simple **plain text**.
   - One noted that *PDFs are not an API*, emphasizing the need for more adaptable data formats.
- **Community Insights on Assistant References**: A member sought help in making their assistants stop referencing the documentation at the end of results, sharing an image with suggested code changes.
   - Another member responded with enthusiasm, extending a supportive *cheers in advance* message for those willing to assist.
- **Simplifying Language for Non-Native Speakers**: A user shared their **de-GPTing prompt**, aimed at rephrasing prompts to make them friendlier for second language speakers by avoiding complex vocabulary.
   - The intent is to simplify the language while maintaining the integrity of technical terms.


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1328707329163657226)** (9 messages🔥): 

> `Konkani Language Preservation, AI Model for Konkani language` 


- **Konkani Language Preservation Initiative**: A member introduced **Reuben Fernandes**, a CSE undergraduate from Goa, who is working on a project to preserve the **Konkani** language spoken by **2.5 million** people.
   - *He's reaching out for collaboration from industry professionals* to enhance the project's impact and approval chances.
- **AI Model for Understanding Konkani**: **Reuben's project** aims to develop an AI model that can converse in and understand **Konkani**, promoting the language's cultural significance.
   - *He noted the uniqueness* of his project as no existing models can thoroughly comprehend Konkani, making it essential for preservation.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1328693517777764462)** (16 messages🔥): 

> `Rerank fine-tuning pricing, AI memory limits, API key limits, Cohere pricing and documentation` 


- **Rerank Fine-Tuning Pricing Confusion**: A member inquired about the pricing for fine-tuning rerank models, mentioning that it wasn't listed on the [Cohere pricing page](https://cohere.com/pricing). Another member provided documentation links related to [Rerank FT](https://docs.cohere.com/docs/rerank-fine-tuning) and FAQs for further assistance.
- **AI's Memory Scope Explained**: Discussion revealed that AI has a context length of **128k tokens**, translating to about **42,000 words** before it forgets. Clarification indicated that this memory capacity applies to the entire timeline of interactions, not just individual chats.
- **Long-Term vs Short-Term AI Memory**: Members discussed the distinction between long-term and short-term memory in AI systems. Context length and retention were specified to apply across multiple interactions in total.
- **Understanding API Key Limits**: It was confirmed that API keys have usage limits based on requests rather than token length. Details on rate limits for both trial and production API keys can be found in the [Cohere documentation](https://docs.cohere.com/v2/docs/rate-limits).
- **Cohere API Keys Overview**: Cohere offers evaluation keys with limited usage and production keys with fewer constraints. Members were directed to the [API keys page](https://dashboard.cohere.com/api-keys) for creating keys and the [pricing docs](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work) for more information.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/v2/docs/rate-limits">Different Types of API Keys and Rate Limits — Cohere</a>: This page describes Cohere API rate limits for production and evaluation keys.</li><li><a href="https://youtu.be/B45s_qWYUt8">How Cohere will improve AI Reasoning this year</a>: Aidan Gomez, CEO of Cohere, reveals how they&#39;re tackling AI hallucinations and improving reasoning abilities. He also explains why Cohere doesn&#39;t use any out...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1328611499744170015)** (11 messages🔥): 

> `Bot Interaction, Alice in Wonderland Reference, Cohere Documentation Search` 


- **Bot Interaction on Language**: A user engaged the Cmd R Bot with a question about any similarity between **corvo** and **escrivaninha**, which the bot initially answered with none.
   - *Commonwealth* challenged the bot's response by referencing **Alice in Wonderland**, hinting at a deeper connection.
- **Alice in Wonderland Quip**: The conversation referenced **Alice in Wonderland**, suggesting that there may be a playful or quirky connection between the words despite the bot's assertion.
   - This points to how cultural references can prompt deeper discussions about language.
- **Cohere Documentation Ineffectiveness**: The Cmd R Bot attempted to enhance the conversation by searching the **Cohere documentation** for information about the terms *corvo* and *escrivaninha*, but ultimately found nothing.
   - This demonstrates the limitations of the bot's data access in addressing cultural or literary queries.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1328454892544589884)** (2 messages): 

> `Meeting attendance, Update sharing, YouTube link` 


- **Update Shared Amidst Class Schedules**: A member expressed gratitude for an update and mentioned missing part of a meeting due to a class conflict.
   - *Thanks for the update* reflects the community's supportive nature despite scheduling challenges.
- **YouTube Video Resource Provided**: Another member shared a [YouTube video](https://www.youtube.com/watch?v=PYtNOtCD1Jo) link, possibly related to the meeting discussion.
   - The video could serve as a valuable resource for those who missed the meeting.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1328454911213441145)** (26 messages🔥): 

> `Async Mojo Proposals, Zed Extension for Mojo, Mojodojo Int8 Conversion Issue` 


- **Owen's Async Mojo proposals spark lively discussion**: In a thread initiated by predicting challenges of implementing structured async in Mojo, one member emphasized the need for standards in effect handling across different libraries.
   - Concerns were raised about handling exceptions consistently, while ensuring that effects like `oom` and `divbyzero` are standard library-defined.
- **Expanded discussion on multiple executors in Mojo**: Members discussed the implications of executing multiple co-existing executors, noting the need for orthogonality between implementations.
   - One participant suggested that foundational API development should precede these discussions for a more practical approach.
- **Zed extension development for Mojo progresses**: A participant shared their development of a Mojo extension for Zed, noting challenges with the previous extension failing to recognize `stdlib` paths and looking for guidance on integration.
   - Another member shared their own extension that provided improved autocompletion and LSP functionalities, suggesting potential enhancements for broader use.
- **Seeking solutions for Mojodojo's Int8 to String conversion**: A user raised an issue regarding the functionality of Int8 to String conversions in Mojodojo and requested assistance from the community.
   - A response highlighted the importance of understanding parameter types in Mojo and provided links to relevant documentation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/collections/string/String/#write_bytes">String | Modular</a>: struct String</li><li><a href="https://docs.modular.com/mojo/manual/types/">Types | Modular</a>: Standard Mojo data types.</li><li><a href="https://github.com/modularml/mojo/issues/3947">[mojo-examples] Mojodojo Int8 to string conversion example not working · Issue #3947 · modularml/mojo</a>: Where is the problem? https://mojodojo.dev/guides/intro-to-mojo/basic-types.html#strings What can we do better? This conversion var word = List[Int8]() word.append(78) word.append(79) word.append(0...</li><li><a href="https://github.com/freespirit/mz">GitHub - freespirit/mz: Support for Mojo in Zed</a>: Support for Mojo in Zed. Contribute to freespirit/mz development by creating an account on GitHub.</li><li><a href="https://docs.modular.com/mojo/manual/parameters/">Parameterization: compile-time metaprogramming | Modular</a>: An introduction to parameters and compile-time metaprogramming.</li><li><a href="https://github.com/modularml/mojo/pull/3945">[proposal] Structured Async for Mojo by owenhilyard · Pull Request #3945 · modularml/mojo</a>: Proposes to add structured async to Mojo, following in the the Rust tradition of async since Mojo has the ability to fix many of the issues with Rust&amp;#39;s async, some of which are ecosystem infli...</li><li><a href="https://github.com/modularml/mojo/pull/3946">[proposal] Provided Effect Handlers by owenhilyard · Pull Request #3946 · modularml/mojo</a>: This proposal contains an alternative to an effect system which I think is more suitable for abstracting async, raises, and similar function colors in a systems language where the context may not a...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1328523014958219345)** (4 messages): 

> `Tinygrad Overview, Need for Tinygrad, Tiny Corp's Funding and Vision, AI Chip Companies` 


- **Exploring the Need for Tinygrad**: A user inquired about resources to understand the purpose of **tinygrad**, mentioning they looked at the website and documentation but lacked deeper insights.
   - *Understanding LLVM is challenging,* they noted, prompting a discussion on the significance of tinygrad.
- **Tiny Corp's New Venture Funded**: A member shared a [blog post](https://geohot.github.io/blog/jekyll/update/2023/05/24/the-tiny-corp-raised-5M.html) revealing that *tiny corp* has received **$5M** in funding, signaling an ongoing business venture.
   - This company aims to eventually focus on **chip development**, with a broader goal of making advanced computing accessible.
- **Understanding Human Brain Compute Power**: The post highlighted that the human brain operates at about **20 PFLOPS**, which remains largely inaccessible, costing around **$1M** or **$100/hr** for equivalent computing.
   - This concept was discussed in various *blog posts* by the founder, reflecting on the compute limitations faced by most individuals.



**Link mentioned**: <a href="https://geohot.github.io/blog/jekyll/update/2023/05/24/the-tiny-corp-raised-5M.html">the tiny corp raised $5.1M</a>: Here we go again. I started another company. The money is in the bank.

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1328698975758581891)** (16 messages🔥): 

> `Learning Tinygrad, Tensor Stacking Issues, Recursion Error in Tinygrad, Getting Tensor Attributes, Using Tensor Functions` 


- **Learning Tinygrad through Streams**: A member shared their journey of learning **Tinygrad** by watching George's streams and analyzing chat logs, which helped them grasp concepts better.
   - They emphasized coming from a **distributed systems** angle to understand the implications of hardware failures.
- **Issues with Stacking Tensors**: A user faced challenges using the **.stack** function with their tensors and encountered a **RecursionError** when trying to call .numpy() on stacked tensors.
   - It was noted that reducing the number of stacked tensors from **6131 to 1000** resolved the issue.
- **Navigating Recursion Limits in Tinygrad**: Members discussed encountering a **recursion depth exceeded** error due to having too many tensors stacked in one operation.
   - A solution was proposed to limit operations and stack fewer tensors at a time to avoid exceeding the internal limits.
- **Retrieving Tensor Attributes**: A user tried accessing different attributes of a Tensor object using the **dir()** function but ran into an **IndexError**.
   - The error occurred because the default `transpose()` call tried to operate on non-existent dimensions for a 1D tensor.
- **Identifying Tensor Function Issues**: It was clarified that the issue with getting tensor attributes stemmed from invoking functions like **transpose()** without proper dimension parameters.
   - Members suggested that careful handling of tensor dimensions is crucial to avoid such errors.



**Link mentioned**: <a href="https://blog.codinghorror.com/because-reading-is-fundamental-2/">Because Reading is Fundamental</a>: Most discussions show a bit of information next to each user:What message does this send? * The only number you can control printed next to your name is post count. * Everyone who reads this will see ...

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1328651997229875231)** (13 messages🔥): 

> `Inference-time scaling in LLMs, Jailbreaking O1 for model training, Automated evaluation methods in healthcare, False equivalence in medical training, Role of AI in medicine` 


- **Inference-time scaling boosts medical reasoning**: The discussed paper indicates that increasing **inference time** for large language models (LLMs) can improve performance significantly by **6%-11%** on medical benchmarks with a training set of only **500 samples**.
   - It highlights that **task complexity** requires longer reasoning chains and debates the exclusion of in-house data due to irrelevant information interfering with logical deductions.
- **Concerns raised about jailbreaking O1**: A member expressed skepticism about the practice of **jailbreaking O1** to train models, arguing that existing benchmarks are mostly irrelevant for effective evaluation.
   - This raises questions about the robustness of the methodologies being applied in model training.
- **Automated evaluations in healthcare raise concerns**: Current automated evaluations in healthcare largely depend on **multiple-choice questions**, limiting assessments to pattern recognition and information recall.
   - There’s a call for methods that more accurately reflect the complexities of medical training and the realities of clinical practice.
- **False equivalences in medical training and AI**: There’s an ongoing debate regarding the false equivalence between scoring well on **multiple-choice exams** and the real-world competence of doctors and medical students in clinical settings.
   - Participants noted that medical students undergo extensive clinical practice, being scrutinized by multiple doctors, unlike the AI models that currently fail despite prompt engineering.
- **AI's role in the future of medicine**: The conversations also highlighted that AI in healthcare shouldn't just focus on replacing doctors but should challenge existing norms and methodologies.
   - Participants appreciated discussions that reflect a more nuanced perspective on AI's impact in the medical field.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.06458">O1 Replication Journey -- Part 3: Inference-time Scaling for Medical Reasoning</a>: Building upon our previous investigations of O1 replication (Part 1: Journey Learning [Qin et al., 2024] and Part 2: Distillation [Huang et al., 2024]), this work explores the potential of inference-t...</li><li><a href="https://www.nature.com/articles/s41467-024-55628-6">Large Language Models lack essential metacognition for reliable medical reasoning - Nature Communications</a>: Large Language Models demonstrate expert-level accuracy in medical exams, supporting their potential inclusion in healthcare settings. Here, authors reveal that their metacognitive abilities are under...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1328470667925913681)** (10 messages🔥): 

> `Open Interpreter Setup, Video Editing Capabilities, Cmder Performance Issues` 


- **Open Interpreter Installation Success**: After initially encountering issues with installing **open-interpreter** via **pipx**, a member confirmed successful setup after installing **brew** and **pipx**.
   - They sought advice on the most useful functions of **open interpreter**, particularly if it can manage video editing.
- **Open Interpreter can execute video commands**: In response to questions about video editing, it was clarified that **open interpreter** can run arbitrary commands, including those related to video editing.
   - This means users could use it to automate video editing processes by supplying appropriate commands.
- **Performance Glitches in Cmder**: A member raised concerns about performance issues in **Cmder** when **open interpreter** outputs large amounts of text, causing the screen to flash and text to move erratically.
   - They noted that the performance is acceptable with shorter outputs, indicating the issue might be triggered by longer text outputs.
- **Output Display Issues**: The same member observed that when **open interpreter** writes a significant amount of text, it appears to clear and rewrite the terminal output frequently, impacting speed.
   - They shared that using the --plain option did not alleviate the issue, leading to peculiar display formatting in **Cmder**.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1328599946122756106)** (2 messages): 

> `Deepseek Model Name, API Key Variable for Deepseek` 


- **Inquiry about Deepseek model name**: A member posed a question regarding the **model name** for **Deepseek**.
   - The inquiry highlighted a need for clarity on the specific model to be utilized in discussions.
- **Question on DEEPSEEK_API_KEY variable**: A member requested information on how to set the **DEEPSEEK_API_KEY** variable.
   - *What should the value be?* indicates uncertainty about proper configuration for using the API.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

velinxs: how can you integrate deepspeek into oi?
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1328467198670409773)** (7 messages): 

> `2024 MOOC nostalgia, Beginner friendliness of the MOOC, Fall 2024 MOOC lectures, Certificate release for Fall 2024 MOOC` 


- **2024 MOOC Sparks Nostalgia**: A user expressed feelings of nostalgia for the **2024 course**, indicating a strong emotional connection.
   - This highlights the lasting impact of engaging learning experiences.
- **MOOC Beginner Friendly Discussion**: A new user inquired about the beginner friendliness of the MOOC after taking an undergraduate course in **Machine Learning**.
   - Responses suggested checking the [Fall 2024 MOOC lectures](https://llmagents-learning.org/f24) to assess difficulty levels.
- **Understanding the Fall 2024 MOOC Content**: A member recommended that a prospective learner review the **Fall 2024 lectures** as they provide foundational knowledge that will be built upon in **Spring 2025**.
   - They assured that the Spring course would not require prior knowledge, just a willingness to engage with the material.
- **Certificate Release Announcement**: A member inquired about the **certificate** for the Fall 2024 MOOC, seeking clarity on its availability after course completion.
   - Another member confirmed that certificates will be **released later this month**, alleviating concerns about missing out.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1328493597456072805)** (5 messages): 

> `NPU Support in GPT4All, VPN and Reverse Proxy for GPT4All Access, Model Versions on Hugging Face` 


- **NPU Support in GPT4All on the Horizon?**: A user inquired whether **GPT4All** would gain the ability to utilize **NPU** soon, specifically referencing **AMD processors**.
   - *Victor Gallagher* indicated that AMD still needs to finalize the **software stack**, but suggested it would be beneficial if accomplished.
- **Make GPT4All Accessible with VPN**: A member suggested running a **VPN** or **reverse proxy** on the inference machine to enable access to the GPT4All local interface from other devices.
   - This method is proposed as a practical solution for ensuring connectivity across machines.
- **Hugging Face Model Versions Clarification**: A user noted the existence of models like **codellama q4_0**, but mentioned that variants with different **quantization** are also available on **Hugging Face**.
   - They concluded that adding the model to a folder resolves their query regarding utilization.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1328786126525300768)** (1 messages): 

> `Agent vs. Naive RAG, Weaviate Integration, LlamaIndex Performance` 


- **Agent Outshines Naive RAG in Tuana's Notebook**: In a recent [notebook by Tuana](https://twitter.com/tuanacelik), the performance of an agent utilizing **Weaviate** and **LlamaIndex** was compared against naive RAG, showcasing superior results.
   - The agent's ability to make **decision-making** choices regarding data sources and their combinations played a crucial role in enhancing overall effectiveness.
- **Exploration of Weaviate with LlamaIndex**: Tuana highlighted the integration of **Weaviate** into workflows with **LlamaIndex**, demonstrating how this combination enhances data management.
   - The discussion emphasized the versatility and potential improvements brought forth by such integrations.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1328456163301261362)** (3 messages): 

> `QuestionsAnsweredExtractor customization, Prompt template function mappings, LlamaIndex package installation` 


- **Enhancing QuestionsAnsweredExtractor with custom prompts**: A member inquired about adding a custom prompt template and additional context variables to the `QuestionsAnsweredExtractor` in `self.llm.apredict()`.
   - There is curiosity on whether more variables can be added dynamically to improve functionality.
- **Using function mappings for dynamic variables**: Another member responded by suggesting the use of function mappings in prompt templates to attach any variable needed when connected to a function.
   - They referenced the [LlamaIndex advanced prompts documentation](https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts/#3-prompt-function-mappings) for further guidance.
- **LlamaIndex package installation issues addressed**: During the discussion, details about package installation were shared, including the downloading of `llama-index-llms-openai` and its dependencies.
   - The installation linked to multiple package versions and indicated successful collections from a mirror URL.



**Link mentioned**: <a href="https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts/#3-prompt-function-mappings">Advanced Prompt Techniques (Variable Mappings, Functions) - LlamaIndex</a>: no description found

  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1328756606778867834)** (2 messages): 

> `Meta JASCO, Joint Audio and Symbolic Conditioning, Music Modeling` 


- **Meta JASCO Model Launches**: The **FAIR team of Meta AI** has announced the release of **JASCO**, a new music model trained in **November 2024**.
   - JASCO features an **EnCodec model** for audio tokenization and two variants that include controls for **chords, drums,** and **melody**.
- **In-depth Tech of JASCO**: JASCO utilizes a **flow-matching model** based on transformer architecture and supports inference with **condition dropout**.
   - It is available in two sizes: **400M** and **1B**, catering to different music generation needs.
- **Research Paper Availability**: For those interested in the technical details, a paper titled [Joint Audio And Symbolic Conditioning for Temporally Controlled Text-To-Music Generation](https://arxiv.org/pdf/2406.10970) provides in-depth information.
   - This paper outlines the methodologies and innovative approaches taken in developing the JASCO model.



**Link mentioned**: <a href="https://huggingface.co/facebook/jasco-chords-drums-melody-1B">facebook/jasco-chords-drums-melody-1B · Hugging Face</a>: no description found

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1328849017811238984)** (1 messages): 

> `Ambient Agents, DSPy implementation examples` 


- **Questions on implementing Ambient Agents using DSPy**: A member asked how to implement an **ambient agent** using DSPy and requested examples from those who have already done it.
   - *No specific examples were shared in the discussion.*
- **Interest in DSPy implementations**: Another member showed interest in seeing **implemented examples** of DSPy, particularly in relation to ambient agents.
   - There was an invitation for other members to share any relevant resources or experiences.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1328734718702911510)** (1 messages): 

> `Healthcare and AI, Finance and AI, AI-enabled solutions` 


- **Upcoming Session on AI in Healthcare and Finance**: Join us on **January 16, 2025**, from **4:00 pm to 5:30 pm IST** for an insightful session exploring **AI's impact** on healthcare and finance with a global panel.
   - This event aims to engage builders, operators, and owners of **AI-enabled solutions** to discuss opportunities and challenges in these sectors, with registration available [here](http://bit.ly/3PxJXbV).
- **Panel Discussion on AI Solutions**: The session will feature a panel consisting of experts discussing how AI is transforming the **healthcare** and **finance** sectors.
   - Participants are encouraged to engage in forward-looking discussions on the intersection of these fields.



**Link mentioned**: <a href="http://bit.ly/3PxJXbV">Welcome! You are invited to join a meeting: Healthcare, Finance, and Artificial Intelligence . After registering, you will receive a confirmation email about joining the meeting.</a>: Healthcare, Finance, and Artificial Intelligence (AI) are increasingly intertwined in today's world. The interconnectedness among these fields has also brought to the fore opportunities and challenges...

  

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

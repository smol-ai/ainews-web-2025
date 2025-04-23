---
id: 55642e7a-1407-494d-bac5-6053afc28810
title: not much happened today
date: '2025-01-04T07:58:51.225259Z'
original_slug: ainews-not-much-happened-today-4979
description: >-
  The open-source PRIME model has achieved a 26.7% pass@1 rate, surpassing
  GPT-4o in reasoning tasks, while Olmo 2 released a detailed tech report
  showcasing one of the few remaining frontier fully open models. Additionally,
  significant performance improvements were reported with Qwen 32B generating
  over 40 tokens per second on an M4 Max.
tags:
  - open-source
  - prime
  - gpt-4o
  - olmo-2
  - qwen-32b
  - model-performance
  - reasoning
---


<!-- buttondown-editor-mode: plaintext -->**a quiet week to start the year**

> AI News for 1/2/2025-1/3/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **32** Discords (**217** channels, and **2120** messages) for you. Estimated reading time saved (at 200wpm): **236 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Lots of "open o1" imitators causing noise, but mostly not leaving much confidence and meanwhile o1 is continuing to impress. [Olmo 2 released their tech report](https://x.com/soldni/status/1875266934943649808?s=46) ([our first coverage here](https://buttondown.com/ainews/archive/ainews-olmo-2-new-sota-fully-open-model/)), with characteristic full {pre|mid|post}-training detail for one of the few remaining frontier fully open models.

![image.png](https://assets.buttondown.email/images/987e58a5-32e2-404e-9fd6-f9209e187d48.png?w=960&fit=max)


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

**AI Models and Performance**

- **Model Developments and Benchmarks**: [@_akhaliq](https://twitter.com/_akhaliq/status/1875039314771660989) introduced **PRIME**, an open-source solution advancing reasoning in language models, achieving **26.7% pass@1**, surpassing **GPT-4o**. Additionally, [@_jasonwei](https://twitter.com/_jasonwei/status/1875268874859344349) discussed the importance of dataset selection in evaluating **Chain-of-Thought** methods, emphasizing their effectiveness in **math and coding** tasks.

- **Optimization Techniques**: [@vikhyatk](https://twitter.com/vikhyatk/status/1875200315966005513) benchmarked **libvips**, finding it **25x faster** at resizing images compared to **Pillow**. Furthermore, [@awnihannun](https://twitter.com/awnihannun/status/1874930431969394875) reported that **Qwen 32B (4-bit)** generates at **>40 toks/sec** on an **M4 Max**, highlighting performance improvements.

- **Architectural Insights**: [@_arohan_](https://twitter.com/_arohan_/status/1875041433620815874) criticized the stagnation in architectural breakthroughs despite exponential increases in compute, suggesting that breakthroughs in architectures akin to **SSMs** or **RNNs** might be necessary.

**AI Tools and Frameworks**

- **Development Tools**: [@tom_doerr](https://twitter.com/tom_doerr/status/1875080307881263387) shared **Swaggo/swag**, a tool for generating **Swagger 2.0** documentation from **Go** code annotations, supporting frameworks like **Gin** and **Echo**. Additionally, [@hendrikbgr](https://twitter.com/Hacubu/status/1875230158162174222) announced integration between **Cerebras Systems** and **LangChain.js**, enabling **streaming**, **tool calling**, and **structured output** for **JavaScript/TypeScript** applications.

- **Agent Frameworks**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1874930168739017149) previewed upcoming **agent architectures** for 2025, focusing on **customizability** for domains like **report generation** and **customer support**.

- **Version Control and Security Tools**: [@tom_doerr](https://twitter.com/tom_doerr/status/1875072709568106775) introduced **Jujutsu (jj)**, a **Git-compatible** VCS using **changesets** for simpler version control, and **Portspoof**, a security tool that makes all **TCP ports** appear open to deter attackers.

**Robotics and Hardware**

- **Robotics Advances**: [@adcock_brett](https://twitter.com/adcock_brett/status/1874960476565815473) unveiled **Gen 2** of their weapon detection system with a **meters-wide field of view** and **faster image frame rate**. Additionally, [@shuchaobi](https://twitter.com/shuchaobi/status/1874992397060592021) promoted video speech models powered by their latest hardware designs.

- **Hardware Optimization**: [@StasBekman](https://twitter.com/StasBekman/status/1874981298290430130) added a **high-end accelerator cache-sizes** section, comparing **cache architectures** across manufacturers, and [@StasBekman](https://twitter.com/StasBekman/status/1874979658112086234) shared benchmarks comparing **H100 vs MI300x**, noting **different winners for different use cases**.

**AI Applications and Use Cases**

- **Medical and Financial Applications**: [@reach_vb](https://twitter.com/reach_vb/status/1875225903346909256) discussed **Process Reinforcement through Implicit Rewards (PRIME)** enhancing **medical error detection** in clinical notes. [@virattt](https://twitter.com/virattt/status/1874984637346324555) launched a **production app** for an **AI Financial Agent** integrating **LangChainAI** and **Vercel AI SDK**.

- **Creative and Educational Tools**: [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1874933780114514166) demonstrated converting **text to 3D printed objects** using tools like **Gemini** and **Imagen 3.0**. [@virattt](https://twitter.com/virattt/status/1874984639754080597) also highlighted **Aguvis**, a **vision GUI agent** for multiple platforms.

- **Workflow and Automation**: [@bindureddy](https://twitter.com/bindureddy/status/1875003427488772334) detailed how **agents** manage **workflows**, **data transformation**, and **visualization widgets**, while [@llama_index](https://twitter.com/llama_index/status/1875225903346909256) provided resources for building **agentic workflows** in **invoice processing**.

**Industry Updates and News**

- **Company Growth and Investments**: [@sophiamyang](https://twitter.com/sophiamyang/status/1875219788407980237) celebrated **1-year at MistralAI**, highlighting team growth from **20 to 100+ employees**. [@Technium1](https://twitter.com/Teknium1/status/1875261716361281699) reported on **$80B** spent by datacenters this year.

- **Regulatory and Market Trends**: [@tom_doerr](https://twitter.com/tom_doerr/status/1875259944481779981) criticized the **EU's fast-moving regulations**, and [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1875282332875419878) addressed concerns about **H-1B visa holders** and **drop in tech investing**.

- **AI Leadership and Conferences**: [@swyx](https://twitter.com/swyx/status/1875253083737055299) announced the **AI Leadership track of AIEWF** now available on **YouTube**, featuring insights from leaders like **@MarkMoyou (NVIDIA)** and **@prathle (Neo4j)**.

**Community and Personal Reflections**

- **Memorials and Personal Stories**: [@DrJimFan](https://twitter.com/DrJimFan/status/1874959979553427815) shared a heartfelt tribute to **Felix Hill**, expressing sorrow over his passing and reflecting on the intense pressures within the AI community.

- **Productivity and Learning**: [@swyx](https://twitter.com/swyx/status/1875258588635320381) emphasized the importance of **self-driven projects** for personal growth, and [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1875104827900129666) advocated for **recording work processes** to enhance learning and data availability.

**Memes and Humor**

- **Light-Hearted Takes**: [@Scaling01](https://twitter.com/scaling01/status/1875151612693647714) joked about the **irrelevance of architecture opinions**, while [@HamelHusain](https://twitter.com/HamelHusain/status/1875235369970737207) shared humorous reactions with multiple **🤣 emojis**.

- **Humorous Anecdotes**: [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1875210891022754103) posted about **missing a rice cooker**, and [@teortaxesTex](https://twitter.com/teortaxesTex/status/1875243857056821477) reacted with **"🤣🤣🤣🤣"** to amusing content.

- **Funny Observations**: [@nearcyan](https://twitter.com/nearcyan/status/1875026913590386715) humorously contrasted **quantity vs quality in tweeting ideas**, and [@thinkzarak](https://twitter.com/thinkzarak/status/1874950373980400128) shared a witty take on **AI's role in society**.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. LLM Performance Leap Creates Demand for New Benchmarks**

- **Killed by LLM – I collected data on AI benchmarks we thought would last years** ([Score: 98, Comments: 18](https://reddit.com/r/LocalLLaMA/comments/1hs6ftc/killed_by_llm_i_collected_data_on_ai_benchmarks/)): **GPT-4** in **2023** revolutionized AI benchmarks by not only surpassing state-of-the-art scores but also saturating them, marking a significant milestone akin to passing the Turing Test. By **2024**, other models like **O1/O3** and **Sonnet 3.5/4o** caught up, saturating math, reasoning, and visual benchmarks, while **Llama 3/Qwen 2.5** made open-weight models competitive. The author argues for improved benchmarks by **2025** to better measure real-world reliability, as current benchmarks fail to assess tasks expected to be solved by **2030**, and invites contributions to their [GitHub repository](https://github.com/R0bk/killedbyllm) for further development.
  - Commenters discuss the limitations of AI models like **GPT-4** and **O1/O3** in handling complex tasks, noting their proficiency in generating initial code or boilerplate but struggles with integration, security, and niche problems. They emphasize that while these models can provide impressive overviews and solutions, they often fail with larger, more complex applications.
  - The conversation highlights the potential need for a coding paradigm shift, suggesting frameworks that optimize code for AI comprehension. **Robk001** and **Gremlation** discuss how breaking code into small, manageable chunks can improve AI performance, with the latter pointing out that quality input results in better AI output.
  - Users like **Grouchy-Course2092** and **butteryspoink** share experiences of increased productivity when providing detailed input to AI models. They note that structured approaches, such as using **SDS+Kanban boards**, can significantly enhance the quality of AI-generated code, suggesting that user input quality plays a critical role in AI effectiveness.


- **LLM as survival knowledge base** ([Score: 83, Comments: 88](https://reddit.com/r/LocalLLaMA/comments/1hsm57o/llm_as_survival_knowledge_base/)): **Large Language Models (LLMs)** serve as a dynamic knowledge base, offering immediate advice tailored to specific scenarios and available resources, surpassing traditional media like books or TV shows. The author experimented with popular local models for hypothetical situations and found them generally effective, seeking insights from others who have conducted similar research and identified preferred models for "apocalypse" scenarios.
  - **Power and Resource Concerns**: The practicality of using **LLMs** in survival scenarios is debated due to high power consumption. Some argue that small models, such as **7-9B**, can be useful with portable solar setups, while others highlight the inefficiency of using precious resources for potentially unreliable AI outputs. **ForceBru** emphasizes the randomness of LLM outputs, while others suggest combining LLMs with traditional resources like books for more reliable guidance.
  - **Trustworthiness and Hallucination**: Many commenters, including **Azuras33** and **Calcidiol**, express concerns about LLM hallucinations, suggesting the integration of **Retrieval-Augmented Generation (RAG)** with grounded data sources like Wikipedia exports to improve reliability. **AppearanceHeavy6724** and others discuss techniques like asking the same question multiple times to identify consistent answers and reduce hallucination risks.
  - **Model Fine-Tuning and Practical Usage**: **Lolzinventor** and **benutzername1337** discuss the potential of fine-tuning smaller models, like **Llama 3.2 3B**, for survival-specific knowledge, noting the importance of curating a dataset from survival and DIY resources. **Benutzername1337** shares a personal experience using an 8B model during a survival trip, highlighting both its utility and limitations due to power constraints.


**Theme 2. Deepseek V3 Hosted on Fireworks, Privacy and Pricing**

- **Deepseek V3 hosted on Fireworks (no data collection, $0.9/m, 25t/s)** ([Score: 119, Comments: 65](https://reddit.com/r/LocalLLaMA/comments/1hselkx/deepseek_v3_hosted_on_fireworks_no_data/)): **Deepseek V3** is now hosted on **Fireworks**, offering enhanced privacy by not collecting or selling data, unlike the Deepseek API. The model supports a full **128k context size**, costs **$0.9/m**, and operates at **25t/s**; however, privacy concerns have been raised about the terms of service. **OpenRouter** can proxy to Fireworks, and there are plans for fine-tuning support, as discussed in their [Twitter thread](https://x.com/FireworksAI_HQ/status/1874231432203337849).
  - **Privacy Concerns and Trustworthiness**: Users express skepticism about **Fireworks**' privacy claims, noting that companies often have broad terms of service that allow them to use submitted content extensively. Concerns about data collection and potential misuse are highlighted, with some users questioning the trustworthiness of Fireworks.
  - **Performance and Cost Issues**: Users report dissatisfaction with **Fireworks** when accessed through **OpenRouter**, citing slower response times and higher costs compared to alternatives. There are mentions of **Deepseek V3** being an **MoE model**, which activates only **37B parameters** out of **671B**, making it cheaper to run when scaled, but users remain doubtful about the low pricing.
  - **Technical Implementation and Infrastructure**: Discussions touch on the technical infrastructure needed for **Deepseek V3**'s performance, suggesting that its cost-effectiveness may result from efficient use of memory and infrastructure design. **Exolabs' blog** is referenced for insights into running such models on alternative hardware like **Mac Minis**.


- **Deepseek-V3 GGUF's** ([Score: 63, Comments: 26](https://reddit.com/r/LocalLLaMA/comments/1hsort6/deepseekv3_ggufs/)): **DeepSeek-V3 GGUF** quants have been uploaded by **u/fairydreaming** and **u/bullerwins** to [Hugging Face](https://huggingface.co/bullerwins/DeepSeek-V3-GGUF/tree/main). A request is made for someone to upload t/s with **512GB DDR4 RAM** and a **single 3090 GPU**.
  - **Memory Requirements**: The discussion highlights that **q4km** requires approximately **380 GB of RAM** plus additional space for context, totaling close to **500 GB**, making it unsuitable for systems with less RAM like a **Macbook Pro with an m4 chip**. **Q2** quantization is mentioned as having lower RAM requirements at **200 GB**, but is considered ineffective.
  - **Hardware Considerations**: Users are discussing hardware upgrades, with one planning to order an additional **256GB of DDR5 RAM** to test the setup, while others express limitations due to motherboard constraints. **bullerwins** provides performance benchmarks, noting **14t/s prompt processing** and **4t/s text generation** with **Q4_K_M** on their setup, and mentions using an **EPYC 7402 CPU** with **8 DDR4 memory channels**.
  - **Performance Comparisons**: There is a debate on the performance of CPUs versus **4x3090 GPUs**, with **bullerwins** noting a **28% performance loss** in prompt processing and **12% in inference** when using a CPU compared to GPUs. The GPUs can only load **7 out of 61 layers**, highlighting the limitations of GPU memory in this context.


**Theme 3. Tsinghua's Eurus-2: Novel RL Methods Beat Qwen2.5**

- **Train a 7B model that outperforms GPT-4o ?** ([Score: 74, Comments: 10](https://reddit.com/r/LocalLLaMA/comments/1hsk8h8/train_a_7b_model_that_outperforms_gpt4o/)): The **Tsinghua team** introduced **PRIME (Process Reinforcement through Implicit Rewards)** and **Eurus-2**, achieving advanced reasoning with a **7B model** that surpasses **Qwen2.5-Math-Instruct** using only 1/10 of the data. Their approach addresses challenges in reinforcement learning (RL) by implementing **implicit process reward modeling** to tackle issues with precise and scalable dense rewards and RL algorithms' efficiency. [GitHub link](https://github.com/PRIME-RL/PRIME)
  - **GPU requirements** were questioned by **David202023**, seeking details on the hardware needed to train a **7B model**, indicating interest in the technical specifications.
  - **Image visibility issues** were raised by **tehnic**, who noted the inability to view images, suggesting potential accessibility or hosting problems with the project's resources.
  - **Model testing** plans were expressed by **ozzie123**, who intends to download and evaluate the model, showcasing community engagement and practical interest in the project's outcomes.


**Theme 4. OLMo 2.0: Competitive Open Source Model Released**

- **[2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)** ([Score: 117, Comments: 29](https://reddit.com/r/LocalLLaMA/comments/1hsdrpg/2_olmo_2_furious/)): **OLMo 2** is aiming to surpass **Llama 3.1** and **Qwen 2.5** in performance, indicating a competitive landscape in AI model development. The post title suggests a focus on speed and intensity, possibly referencing the "Fast and Furious" film franchise.
  - **OLMo 2's Performance and Data Strategy**: OLMo 2 models are positioned at the **Pareto frontier** of performance relative to compute, often surpassing models like **Llama 3.1** and **Qwen 2.5**. The team uses a **bottom-up data curation strategy**, focusing on specific capabilities such as math through synthetic data, while maintaining general model capabilities with high-quality pretraining data.
  - **Community and Open Source Engagement**: The release of OLMo 2 is celebrated for its openness, with all models, including **7B and 13B scales**, available on [Hugging Face](https://huggingface.co/allenai/OLMo-2-1124-7B). The community appreciates the **fully open-source nature** of the project, with recognition for its transparent training data, code, and recipes.
  - **Future Developments and Community Interaction**: The OLMo team is actively engaging with the community, with discussions about potential larger models (32B or 70B) and ongoing experiments applying the **Molmo recipe** to OLMo 2 weights. The team has also shared training data links for Molmo on [Hugging Face](https://huggingface.co/collections/allenai/pixmo-674746ea613028006285687b).


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. Video Generation Tool Comparison: Sora vs Veo2 vs Minimax**

- **[Pov trying to use the $200 version of Sora...](https://v.redd.it/cr0zeu3ihnae1)** ([Score: 164, Comments: 47](https://reddit.com/r/OpenAI/comments/1hs5bzr/pov_trying_to_use_the_200_version_of_sora/)): The post lacks specific content or context for a detailed summary, as it only mentions the **$200 version of Sora** in the title without further elaboration.
  - Discussions highlight concerns about **content filtering** in AI models, with users expressing frustration over **policy violations** triggered by non-sexual content. Some argue that the filtering system is overly restrictive, particularly regarding depictions of women, and question the efficacy of current content moderation approaches.
  - **Video generation alternatives** like **hailuoai.video** and **minimax** are discussed, with users comparing their features and effectiveness. **Veo2** is noted for its superior results, though access is limited due to a waitlist, and **hunyuan video** is mentioned as a strong competitor.
  - Users discuss the **copyright paradox** in AI training, pointing out the inconsistency in allowing models to train on copyrighted material while restricting generated outputs that resemble such material. Concerns about the high denial rate of content and the potential for increased restrictions to avoid negative publicity are also raised.


**Theme 2. GPT-4o: Advanced Reasoning Over GPT-3.5**

- **[Clear example of GPT-4o showing actual reasoning and self-awareness. GPT-3.5 could not do this](https://www.reddit.com/gallery/1hs5ffs)** ([Score: 112, Comments: 74](https://reddit.com/r/OpenAI/comments/1hs5ffs/clear_example_of_gpt4o_showing_actual_reasoning/)): The post discusses **GPT-4o's** capabilities in advanced reasoning and self-awareness, noting that these features are improvements over **GPT-3.5**. Specific examples or context are not provided in the post body.
  - Discussions highlight that **GPT-4o's** ability to recognize and explain patterns is not indicative of reasoning but rather enhanced **pattern recognition**. Commenters emphasize that the model's ability to identify patterns like "HELLO" is due to its training and tokenization processes, rather than any form of self-awareness or reasoning.
  - Several commenters, including **Roquentin** and **BarniclesBarn**, explain that the model's performance is due to **tokenization and embeddings**, which allow it to recognize patterns without explicit instructions. This aligns with the model's design to predict the next token based on previous context, rather than demonstrating true reasoning or introspection.
  - The conversation also touches on the **limitations of the "HELLO" pattern** as a test, suggesting that using non-obvious patterns could better demonstrate reasoning capabilities. **ThreeKiloZero** and others suggest that the model's vast training dataset and multi-parametric structure allow it to match patterns rather than reason, indicating the importance of context and training data in its responses.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-2024-12-17

**Theme 1. Performance Sagas and Slowdowns**

- [**DeepSeek Takes a Dive**](https://status.deepseek.com/): Users griped about DeepSeek v3 dropping to 0.6 TPS, sparking calls for server scaling. They monitored the status page for signs of relief, but many still yearned for a faster model.  
- [**Windsurf and Cascade Credits Clash**](https://codeium.com/plan): People saw internal errors drain credits in Windsurf, leading to confusion. Despite automated assurances, charges persisted, prompting frustrated posts about refunds.  
- **ComfyUI Under the Hood**: SwarmUI runs on ComfyUI’s backend for user-friendly rendering, while rival setups like Omnigen or SANA lag behind. Fans praised LTXVideo and HunyuanVideo for blazing-fast video generation with minimal quality loss.

**Theme 2. Credit Crunch and Cost Confusion**

- [**$600k Model Training Sticker Shock**](https://news.ycombinator.com/item?id=39224534): Engineers shared jaw-dropping GPU bills for training large models, with 7B parameters costing around $85k. They debated cheaper hosting options like [RunPod](https://runpod.io) and low-rank adapters from the [LoQT paper](https://arxiv.org/abs/2405.16528).  
- [**Payment Woes Plague API Users**](https://openrouter.ai/api/v1): Credit card declines in OpenRouter and subscription quirks in Perplexity stirred confusion. Some overcame it by switching cards or clearing caches, but annoyance simmered.  
- **Flex vs. Premium Credits**: Multiple communities slammed usage caps and missing rollovers. Paying for unused tokens or dealing with “internal error” sessions fueled calls for more transparent plans.

**Theme 3. Model Debuts and Fine-Tuning Frenzy**

- [**Sonus-1 Steals the Show**](https://sonus.ai/blog/sonus-1): Its Mini, Air, Pro, and Reasoning variants drew chatter for advanced text gen in 2025. A [tweet](https://x.com/RubiksAI/status/1874682159379972325) showcased swift code outputs combining Aider with Sonus mini models.  
- [**Swarm Library Joins NPM**](https://www.npmjs.com/package/agentswarm): This TypeScript multi-agent AI library touts synergy beyond OpenAI’s Swarm, winning praise for modular design. Others pinned hopes on PersonaNLP at ICML 2025, focusing on persona-based NLP tasks.  
- **Qwen2-VL & Llama 3.1 Spark Tuning**: Communities wrestled with partial or missing vision adapters, while Llama 3.x and SmallThinker-3B soared in performance. People also fine-tuned with Unsloth, Axolotl, or Hugging Face Transformers/PEFT for custom tasks.

**Theme 4. Tooling Triumphs and Tensions**

- [**GraphRAG and Graphrag Grab Headlines**](https://github.com/microsoft/), with new retrieval-augmented generation strategies fueling interest in code and text tasks. Conversations covered multi-retriever setups and weighting vectors to improve query results.  
- [**Invoice Agents and K-Summary Betas**](https://twitter.com/llama_index/status/1875225903346909256): LlamaIndex showcased automatically classifying invoices with spend categories and cost centers, while new summarizers in the Korean market lured testers. Users raved about chunked cross entropy and memory-savvy approaches in Torchtune.  
- **AI for Coding Hustle**: Codeium, Cursor, and Aider communities battled random code changes, linting mania, and limited plan tiers. Despite frustration, many lauded faster dev cycles and more consistent code suggestions.

**Theme 5. Hardware, VRAM, and HPC Adventures**

- [**RTX 50xx VRAM Brouhaha**](https://discord.com/channels/729741769192767510) left engineers suspecting NVIDIA artificially caps memory. They mulled whether bigger VRAM or cunning offload solutions to main RAM is the real path forward.  
- [**Torch.compile Headaches**](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html): Users saw major slowdowns with Inductor caching, dynamic br/bc in Flash Attention, and tricky Triton kernel calls. They tested environment hacks to dodge segfaults, hoping official patches solve the compile chaos.  
- [**1-bit LLM Hype**](https://arxiv.org/abs/2402.17764): BitNet’s talk of ternary weights and dramatic resource cuts excited HPC fans. Some bet these low-bit breakthroughs will slash training bills without sacrificing model accuracy.

---

# PART 1: High level Discord summaries




## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Whirling Windsurf Woes**: Many members reported **major performance issues** with **Windsurf**, citing excessive internal errors and slowdowns, referencing [this tweet about bringing Figma designs to life](https://x.com/windsurf_ai/status/1874948790928687194).
   - They also observed **Claude 3.5 Sonnet** deteriorating mid-session, causing unexpected credit drains despite official disclaimers at [Plan Settings](https://codeium.com/plan).
- **Cascade Credits Controversies**: Community discussions focus on **Cascade** charging credits even during failed operations, with repeated 'internal error' messages leading to confusion.
   - Several users claim these charges persist contrary to automated assurances, prompting some to escalate via [Support | Windsurf Editor and Codeium extensions](https://codeium.com/support).
- **DeepSeek v3 vs. Sonnet 3.6 Showdown**: Some argue **DeepSeek v3** falls short of **Sonnet 3.6** despite benchmark claims, preferring free alternatives like **Gemini**.
   - They cite skepticism about DeepSeek’s real advantage, while others reference [Things we learned about LLMs in 2024](https://simonwillison.net/2024/Dec/31/llms-in-2024/) for further data.
- **Code Editing Chaos in Windsurf**: Users mentioned random code changes and incomplete tasks, requesting clearer solutions to maintain continuity in the AI’s workflow.
   - Many resort to saving instructions in external files, then reloading them to keep the conversation on track.
- **Credit System Grumbles**: Members criticized the **Premium** and **Flex** credits structure, complaining about usage caps and missed rollovers.
   - They urged a fairer allocation model, with reports of mixed success via email and [Support | Windsurf Editor and Codeium extensions](https://codeium.com/support).



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Sonus-1 Steps onto the Stage**: The newly introduced **Sonus-1 Family** (Mini, Air, Pro, Reasoning) was presented in [a blog post](https://sonus.ai/blog/sonus-1), focusing on advanced text generation capabilities for 2025.
   - A [tweet from Rubik's AI](https://x.com/RubiksAI/status/1874682159379972325) highlighted swift code generation in the mini models, prompting discussion on synergy with Aider.
- **Deepseek Stumbles Under Heavy Load**: Community members observed **Deepseek** dropping to **1.2 TPS**, igniting complaints about server capacity and reliability.
   - Others verified that [Deepseek Chat v3](https://openrouter.ai/deepseek/deepseek-chat-v3) remains accessible via `--model openrouter/deepseek/deepseek-chat`, but questioned if more servers are needed.
- **OpenRouter's API Key Confusion**: Some faced **authentication headaches** with the OpenRouter API, suspecting incorrect key placement in config files.
   - One user confirmed success by double-checking model settings, advising the community to watch for hidden whitespace in YAML.
- **Tailwind & Graphrag Grab Spotlight**: Members explored adding Tailwind CSS documentation context into Aider, with suggestions to copy or index relevant info for quick reference.
   - Microsoft's **Graphrag** tool also came up as a RAG alternative, spurring interest in more efficient CLI implementations.
- **Aider Wish List Widened**: Users requested toggling class definitions and prior context to refine code edits, aiming to cut down on irrelevant suggestions.
   - They also envisioned better control over command prompts, citing advanced context management as a prime next step.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **OpenWebUI Exports Expand Dataset Horizons**: Members discussed exporting chat JSON from **OpenWebUI** for dataset creation, referencing owners for formatting advice.
   - They highlighted potential synergy with local inference setups like [vLLM](https://docs.vllm.ai/en/latest/), noting that combining well-structured data with advanced inference can improve training outcomes.
- **Ollama Quantization Quandaries**: Challenges emerged around quantizing models for **Ollama**, with users noting that default GGUF files run in FP16.
   - Attendees recommended manual adjustments and pointed to the [mesosan/lora_model config](https://ollama.com/mesosan/lora_model) for potential solutions.
- **Fine-Tuning Frenzy for Classification**: Community members recommended **Llama 3.1 8B** and **Llama 3.2 3B** for tasks of moderate complexity, citing good performance with classification tasks.
   - They emphasized using GPU hardware like **RTX 4090** and pointed out [Unsloth's documentation](https://docs.unsloth.ai/get-started/beginner-start-here) for tips on efficient finetuning.
- **Fudan Focuses on O1 Reproduction**: A recent **Fudan report** provided in-depth coverage of **O1 reproduction efforts**, available at [this paper](https://arxiv.org/pdf/2412.14135).
   - It was praised by one member as the most thorough resource so far, igniting interest in the next steps for the O1 project.
- **Process Reinforcement Teases Code Release**: The **Process Reinforcement** paper garnered attention for its ideas on implicit rewards, though many lamented the lack of code.
   - Community members remain optimistic that code will be published soon, describing it as a *work in progress* worth watching.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek V3 Model's Privacy Puzzle**: Members raised concerns about **DeepSeek V3** potentially storing code and training on private data, emphasizing privacy issues and uncertain benefits in user projects.
   - They questioned the personal and enterprise risks, debating whether the model's advantages justify its possible data retention approach.
- **Cursor Slashes Dev Time**: A user shared a quick success story using **Cursor** with **SignalR** to finalize a project in far less time than expected.
   - Others chimed in with positive feedback, noting how AI-driven tools are helping them tackle complex development tasks more confidently.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Swarm's NPM Invasion**: The newly published [Swarm library](https://www.npmjs.com/package/agentswarm) for **multi-agent AI** soared onto NPM, offering advanced patterns for collaborative systems beyond **OpenAI Swarm**. It is built with TypeScript, model-agnostic design, and had its last update just a day ago.
   - Community members praised its modular structure, mentioning it as a bold step toward **flexible multi-agent** synergy that could outdistance older frameworks in performance.
- **PersonaNLP Preps for ICML 2025**: A planned **PersonaNLP** workshop for ICML 2025 seeks paper submissions and shared tasks, spotlighting user-focused methods in language modeling. Organizers are openly coordinating with researchers interested in refining persona-based NLP approaches.
   - Participants recommended specialized channels for deeper collaboration and expressed an eagerness to bolster the workshop’s scope.
- **Costs Soar for Giant Models**: Recent discussions revealed **model training** bills reaching **$600,000**, highlighted by a [Hacker News post](https://news.ycombinator.com/item?id=39224534) and a [tweet from Moin Nadeem](https://x.com/moinnadeem/status/1681371166999707648). Members noted that a 7B model alone can cost around **$85,000** on commercial GPUs.
   - Some engineers pointed to services like [RunPod](https://runpod.io) for cheaper setups and explored whether **low-rank adapters** from the [LoQT paper](https://arxiv.org/abs/2405.16528) could reduce spending.
- **Hermes Data Dilemma**: Community members spotted **no explicit training data** for certain adult scenarios in Hermes, speculating that this omission might constrain the model’s broader capabilities. They questioned whether the lack of such data could limit knowledge breadth.
   - One voice claimed skipping these data points removes potentially pivotal nuance, while others argued it was a reasonable compromise for simpler model outputs.
- **Llama Weight Whispers**: Analysts found unexpected amplitude patterns in **K** and **Q** weights within **Llama2**, implying inconsistent token importance. They shared images that suggested partial redundancy in how weights represent key features.
   - Members debated specialized fine-tuning or token-level gating as possible remedies, underscoring new angles for improving Llama2 architecture.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Extractor.io's Vanishing Act**: One user discovered that **Extractor.io** apparently no longer exists, raising confusion despite [this curated LLM list](https://llm.extractum.io/list/) offered as an alternative.
   - Others questioned the abrupt disappearance, with some suggesting it may have folded into a different domain or rebranded.
- **LM Studio Goes Text-Only**: Community members confirmed that **LM Studio** focuses on large language models and cannot produce images.
   - They suggested **Pixtral** for picture tasks, noting it depends on the **MLX Engine** and runs only on specific hardware.
- **Qwen2-VL Without Vision**: Enthusiasts observed that **Qwen2-VL-7B-Instruct-abliterated** lacks the ability to process images due to its missing vision adapter.
   - They emphasized that **proper quantization** of the base model is critical for fully using its text-based strengths.
- **Training on the Entire Internet?**: A user floated the idea of feeding an AI all internet data, but many flagged overwhelming size and poor data quality as pitfalls.
   - They stressed that **bad data** undermines performance, so **quality** must trump sheer volume.
- **GPU Offload for Quicker Quests**: Local LLM fans tapped **GPU** support to speed up quest generation with **Llama 3.1**, citing better responses.
   - They recommended selecting GPU-enabled models and watching **Task Manager** metrics, referencing success with a **4070 Ti Super** setup.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Multilingual Mischief in AI**: Enthusiasts tested **audio overviews** in non-English settings by issuing creative prompts, showing partial success for language expansion.
   - They reported inconsistent translation quality, with suggestions for better **language-specific models** to address these gaps.
- **K-Summary Beta Attracts Attention**: A user promoted a new **AI summarization** product that soared in the Korean market, offering a chance for beta testers to try streamlined summaries.
   - Several community members expressed eagerness to compare it against current summarizers for faster text processing.
- **Customization Function Sparks Debate**: Members worried that **adjusting system prompts** could expose ways to bypass normal AI restrictions.
   - They debated boundaries between creative freedom and safe AI usage, weighing the potential benefits against misuses.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt Embraces AI Logic for Web Apps**: A member shared plans to integrate **logic/AI** in their Bolt web apps, praising the visuals while needing better functionality and referencing [BoltStudio.ai](https://boltstudio.ai/).
   - They asked for strategies to merge code-driven workflows with AI modules, with incremental upgrades and local testing cited as ways forward.
- **Supabase Eases Email Logins in Bolt**: Developers praised **Supabase** email authentication for Bolt-based apps, highlighting local storage to manage user roles.
   - They pointed to [StackBlitz Labs](https://github.com/stackblitz-labs) for bridging frontends with flexible backends, while acknowledging continued debates on Bolt’s token usage.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **O1 vs ChatGPT: Perplexity’s Punchy Search**: **Perplexity O1** drew mixed reactions, with some complaining about **10 daily searches** and calling it a hassle, while others found it promising for search-centric tasks.
   - Comparisons to **ChatGPT** praised **Opus** for unlimited usage and lengthy context, as noted in [this tweet](https://x.com/pplxsupply/status/1875019712268263658).
- **Grok’s Gains or Gripes**: Some called **Grok** *"the worst model"* they've used despite its lower cost, fueling debates on model reliability.
   - Others touted the **3.5 Sonnet model** for stronger performance, hinting at shifting user loyalties.
- **Perplexity’s UI Shake-Up & Subscriptions**: Recent **UI changes** added stock and weather info, prompting one user to clear cache to avoid annoying homepage elements.
   - Members discussed unlimited queries and cost-saving approaches in [AravSrinivas’s tweet](https://x.com/aravsrinivas/status/1874943854849425780), showcasing varied subscription choices.
- **2025 AI Interview Qs to Impress**: A shared guide outlined methods for tackling tricky **AI** questions in **2025**, boosted by [this link](https://www.perplexity.ai/search/the-job-interview-question-of-geScATofQC.NYw5MqWsyiA).
   - Participants deemed thorough preparation essential for staying competitive in the hiring landscape.
- **Europe-Based API & 1.5B-Token Chatbot Hopes**: One user expects **European servers** matching **pro search** speeds to power a **1.5B-token** chatbot with better performance.
   - They believe this integration will enhance the chatbot’s utility, especially for large-scale token usage.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter's Auth Aggravations**: Multiple users saw **'Unauthorized'** errors when attempting requests, even with credits available and the correct API key at [OpenRouter](https://openrouter.ai/api/v1). They reported changing HTTPS addresses and adjusting certificates without relief.
   - Some speculated the issue might involve **n8n** configuration mismatches or connectivity problems, noting that manual tweaks to URL settings still failed.
- **DeepSeek's Dreaded Drag**: Community members complained of **0.6 TPS** speeds on [DeepSeek v3](https://app.hyperbolic.xyz/models/deepseek-v3), causing slow responses. The [DeepSeek service status page](https://status.deepseek.com/) showed high demand and potential scaling shortfalls.
   - They expressed worry that usage had outpaced current forecasts, prompting calls for capacity boosts before more widespread rollouts.
- **Structured Output Seeks a Savior**: A user wanted an alternative to **gpt-4-mini** for JSON-formatted replies but found limited choices in the current lineup. Others suggested **Gemini Flash** and pointed to [LiteLLM](https://github.com/BerriAI/litellm) for handling multiple APIs in a unified interface.
   - They noted potential rate limit constraints and recommended monitoring usage metrics, referencing [RouteLLM](https://github.com/lm-sys/RouteLLM) as another solution for routing requests across various models.
- **Janitor AI Joins OpenRouter**: Members debated how to link Janitor AI with **OpenRouter**, focusing on advanced settings for API endpoints. They outlined toggling certain authentication fields and matching the proxy URLs for synergy in usage.
   - Various configurations were shared, concluding that correct URL alignment and token handling made the integration seamless.
- **Cards Declined, Payments Denied**: Some users found **credit card failures** when attempting to pay through OpenRouter, although different cards sometimes worked fine. One user noted consistent issues with a capital one card, while a second card processed successfully.
   - They considered potential bank-specific rules or OpenRouter’s payment gateway quirks, advising those affected to try multiple billing methods.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SwarmUI Zips Ahead with ComfyUI**: Members explained how **SwarmUI** uses **ComfyUI**’s backend for a simpler UI with the same performance, emphasizing user-friendliness and robust features.
   - They also highlighted a [Stable Diffusion Webui Extension](https://github.com/butaixianran/Stable-Diffusion-Webui-Civitai-Helper) that streamlines model management, prompting chatter about frontends for better workflows.
- **SANA & Omnigen Spark Space Showdown**: Community tested **SANA** for speedy inference on smaller hardware, contrasting it with **Omnigen** that’s slower and sometimes behind **SDXL** in image quality.
   - Enthusiasts questioned if SANA justifies HDD usage, especially when **Flux** might offer better model performance.
- **LTXVideo & HunyuanVideo Go Full Throttle**: **LTXVideo** earned praise for faster rendering on new GPUs with almost no drop in quality, outpacing older video pipelines.
   - Meanwhile, **HunyuanVideo** introduced quicker steps and better compression, fueling excitement about recent strides in video generation.
- **Flux Dev Rocks Text-in-Image Requests**: Members identified **Flux Dev** as a top open-source model for embedding text in images, rivaling closed solutions like **Ideogramv2** and **DALL-E 3**.
   - They also cited **Flux 1.1 Ultra** as the *best closed model* for crisp text output, referencing user tests and side-by-side comparisons.
- **GPU Gains & Memory Must-Haves**: Enthusiasts suggested **RTX-series** cards for AI tasks, with advice to wait for upcoming releases that might cut prices further.
   - They stressed at least **32GB RAM** and decent **VRAM** for smooth image generation, highlighting stability benefits.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RTX 50xx VRAM Limits Light a Fuse**: Engineers debated rumored VRAM caps in the **RTX 50xx** series, suspecting that **NVIDIA** artificially restricts memory to avoid product overlap.
   - Some questioned if the added GBs would matter for AI tasks, revealing frustration about potential performance bottlenecks.
- **VRAM vs. RAM: The Memory Melee**: Multiple participants argued that **VRAM** could be reclassified as **L3 Cache**, noting that regular RAM might be 4x slower than VRAM in certain contexts.
   - Others pondered pipelining commands between VRAM and RAM, warning that any mismatch could hamper throughput for large-scale model inference.
- **Higher-Order Attention Shakes Things Up**: Researchers explored **attention on attention** techniques, referencing expansions in the [Quartic Transformer](https://github.com/lucidrains/quartic-transformer) and linking them to **Mamba** or **SSM** style convolutions.
   - They tied these ideas to **ring attention**, citing the second paper’s bigger context window and highlighting possible line-graph or hypergraph parallels.
- **HYMBA Heads Off SWA**: Community members argued that **HYMBA** mixing full attention on certain layers might undercut the efficiency gains behind **SWA** or SSM.
   - They considered a trade-off between more robust cross-window representation and extra overhead, noting that the actual performance boost requires further tests.
- **Pytorch Flex Attention Bugs Persist**: A few users reported ongoing troubles with **Pytorch's Flex Attention**, which blocked attempts at complex attention patterns.
   - They found that `torch.compile` often clashed with lesser-used model features, forcing them to revert to standard attention layers until a fix emerges.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **2024 LLMs & Image Generation Gains**: An upcoming article, [LLMs in 2024](https://simonwillison.net/2024/Dec/31/llms-in-2024/), spotlights major leaps for **Large Language Models**, including **multimodal** expansions and fierce price competition. The community also noted trending **meme culture** with generative images that turned an everyday person into a comedic 'bro' and gave **Santa** a no-nonsense look.
   - Members credited these cross-domain breakthroughs for spurring broader creative usage, emphasizing how new metrics push LLM performance boundaries. They also observed that cost reductions and accessible APIs accelerate adoption for smaller-scale projects.
- **Text Extraction Throwdown**: A [benchmark study](https://cdn.discordapp.com/attachments/1075282825051385876/1324793970726928537/A_Comparative_Benchmarking_Evaluation_of_Text_Extraction_Tools.pdf) tested **regulatory document** parsing with various libraries. Contribs singled out **pdfium** + **tesseract** combos for their success in tricky data extraction tasks.
   - They underscored how these solutions handle real-world complexity better than standalone OCR or PDF parsing tools. Some cite workflow integration as the next big step for robust text pipelines.
- **SmallThinker-3B's Surging Stats**: The new **SmallThinker-3B-preview** on [Hugging Face](https://huggingface.co/PowerInfer/SmallThinker-3B-Preview) outperforms **Qwen2.5-3B-Instruct** on multiple evaluations. This compact model targets resource-limited scenarios yet shows significant leaps in benchmark scores.
   - Its emphasis on edge-friendly footprints broadens real-world usage, bridging smaller footprints with robust performance. Some participants suspect these improvements stem from specialized fine-tuning and data curation.
- **OLMo2's Outstanding Outline**: The [OLMo2 tech report](https://x.com/soldni/status/1875266934943649808) covers 50+ pages detailing four critical components in the **LLM development** pipeline. It offers a thorough breakdown of data handling, model architecture, evaluation, and deployment strategies.
   - Readers praised its direct approach to revealing real-world learnings, highlighting best practices for reproducible and scalable training. The report encourages devs to refine existing workflows with deeper technical clarity.
- **Summit & Transformers Tactics**: The invite-only [AI Engineer Summit](https://www.latent.space/p/2025-summit) returns after a **10:1** applicant ratio, aiming to highlight new breakthroughs in **AI engineering**. Organizers recall the **3k-seat** World's Fair success, drawing over **1m** online views.
   - In tandem, an **Understanding Transformers** overview from [this resource](https://x.com/sannykimchi/status/1176517584319127553) offers a structured path to learning **self-attention** and modern architectural variations. Summit planners encourage guest publishing opportunities for advanced explainers that push community knowledge forward.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU GEMM Gains**: A user spotlighted the discrepancy between real and theoretical **GEMM** performance on GPU, citing an article on [Notion](https://yywangcs.notion.site/Inconsistency-in-GEMM-Performance-16efc9f5d80580838090dded05493014) and a related [post on Twitter](https://x.com/YyWangCS17122/status/1874856334845489191).
   - They hinted that declaring 'optimal performance' might trigger more advanced solutions from the community.
- **Triton Tuning Troubles**: Removing **TRITON_INTERPRET** was reported to drastically boost Triton kernel performance, especially in matrix multiplication tasks.
   - Others confirmed that setting batch sizes to **16** or more, plus tweaking floating-point tolerances for large inputs, eased kernel call issues.
- **The Flash br/bc Dilemma**: A user asked about dynamic **br/bc** in Flash Attention for better adaptability, but others insisted fixed sizing is '10 trillion times faster.'
   - They proposed compiling multiple versions as **Flash Attention** does, aiming to balance speed with more flexible parameters.
- **Torch Inductor Cache Letdown**: One discussion addressed prolonged load times with **Inductor** caching, reaching **5 minutes** even when using a Redis-based remote cache.
   - They suspect compiled kernel loading still causes delays, motivating additional scrutiny of memory usage and activation needs.
- **P-1 AI’s Radical AGI Push**: P-1 AI is recruiting for an **artificial general engineering** initiative, with [open roles here](https://jobs.lever.co/P-1AI/84ae5c01-9160-44a5-a7c8-a107e645f0a6).
   - Their core team—ex-DeepMind, Microsoft, DARPA, and more—aims to enhance physical system design using **multimodal LLMs** and GNNs for tasks once deemed unworkable.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **LoRA Library War: TRL Tops Unsloth**: The #ml-questions channel contested the merits of LLM fine-tuning tools, praising **TRL** for its thorough documentation while deeming **Unsloth** too difficult, referencing the [LLM Fine-Tuning Library Comparison](https://docs.google.com/document/d/1k0E2XCuqJDGiD6IsP2rb6s3D1Iu96gro9WRCSajl0zs/edit).
   - Though **Unsloth** boasts 20k GitHub stars, members recommended **Transformers/PEFT** from Hugging Face along with **Axolotl** and **Llama Factory** for simpler LoRA fine-tuning.
- **Gating Games: MoE & OLMoE**: Members in #ml-questions asked about gating networks for Mixture of Experts, specifically the routing used in **Deepseek v3**.
   - One user suggested the **OLMoE** paper, highlighting a lower number of experts that keeps complexity under control.
- **Pre-recorded Tutorial Tussle at 20k**: In #random, the community debated whether to share pre-recorded tutorials with an audience of 20k, emphasizing that these talks have earned praise.
   - Another user jokingly labeled the **UK AI Safety Institute** as *intelligence operatives*, while others noted **LinkedIn** competitiveness in AI circles.
- **Bittersweet Lesson and Felix's Legacy**: In #reads, members mourned the passing of **Felix**, who authored [The Bittersweet Lesson](https://docs.google.com/document/d/1MPqtT_1vQ-73j796tf7sXIZKCRcIfUD0cVU_UbPXnUU/edit?usp=sharing) and left a deep impression on the community.
   - They discussed safeguarding his works via PDF backups, concerned that deactivated Google accounts could disrupt future access.
- **SnailBot’s Sluggish Shenanigans**: In #posts, **SnailBot** drew laughs as users deemed it *one slow snail* and exclaimed *good lord* at its performance.
   - Its comedic pace entertained many, who felt it lived up to its name with snail-like consistency.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **FLUX.1 [dev] Fuels Fresh Image Generation**: Members highlighted **FLUX.1 [dev]**, a **12B param rectified flow transformer** for text-to-image synthesis, referencing the [Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/).
   - They noted it stands just behind **FLUX.1 [pro]** in quality and includes **open weights** for scientific research, reflecting an eagerness for community experimentation.
- **ChatGPT’s Search Reliability Under the Microscope**: A user asked if **ChatGPT** can handle real-time web results, comparing it to specialized tools like **Perplexity**.
   - Community feedback indicated the model can be **limited** by data updates, with some preferring external search solutions to bridge **missing information**.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere's Sparkling Rerank-3.5 Release**: Members welcomed the new year with enthusiastic wishes, anticipating that **rerank-3.5** will soon deploy on Azure, delivering a next-level ranker for advanced text processing.
   - The conversation included questions about potential use-cases, with someone asking *“How are you liking it so far?”*, highlighting the community’s eagerness for **insights** on improved performance.
- **Embedding Rate Limit Bumps & Best Practices**: Users explored the request process for boosting **embedding rate limits** by contacting [Cohere support](mailto:support@cohere.com), aiming to handle heavier workloads.
   - Community members outlined existing **API constraints** of 100 requests per minute on trials and 2,000 requests per minute in production, emphasizing efficient usage to avoid overages.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Gains Momentum**: Community members praised **Torchtune** for its broader use in multiple AI models, emphasizing an approach to measure performance.
   - A user recommended exploring alternative evaluation methods with [transformer module](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/transformer.py#L482) for better insights.
- **Chunked Cross Entropy Boosts Memory Efficiency**: **Chunked cross entropy** helps reduce memory usage by splitting the computation, shown in [PR #1390](https://github.com/pytorch/torchtune/pull/1390).
   - One variant used **log_softmax** instead of `F.cross_entropy`, prompting discussion on performance and memory optimization.
- **Flex Attention on A6000 Seeks Workarounds**: Members encountered a **PyTorch Torchtune** bug on the **A6000**, discovering a kernel workaround by setting `flex_attention_compiled` with `torch.compile()`.
   - They proposed an environment variable approach, warning that a permanent fix in **2.6.0** remains uncertain and referencing [Issue #2218](https://github.com/pytorch/torchtune/issues/2218).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Invoice Intelligence with LlamaParse**: A recent [notebook demonstration](https://twitter.com/llama_index/status/1875225903346909256) showcased a custom **invoice processing agent** that automatically classifies **spend categories** and **cost centers**, leveraging LlamaParse for smooth workflows.
   - Members emphasized **automation** to cut manual errors and speed up finance-related tasks, referencing the agent’s approach to handle invoice handling pipelines more efficiently.
- **Simple JSON Solutions for Data Storage**: Community members debated storing LLM evaluation datasets in **S3**, **Git LFS**, or local **JSON** files, highlighting minimal overhead and easy structure.
   - They suggested compressing large JSON data and recommended **Pydantic** for quick integration, noting that **SQL** or **NoSQL** hinges on dataset size.
- **Fusion Friction with Multi-Retrievers**: A user combined **2 vector embedding retrievers** with **2 BM25** retrievers but reported weak query results from this fused setup.
   - Discussions pointed toward tweaking weighting, indexing, or re-ranking strategies to boost the **quality of responses** from blended retrieval methods.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Falters and Receives Recognition**: Users criticized **Open Interpreter 1.0** for performing worse than the classic version, lacking code execution and web browsing, and shared [OpenInterpreter on GitHub](https://github.com/KillianLucas/open-interpreter) for reference. They also highlighted important open-source contributions from **OpenProject**, **Rasa**, and **Kotaemon**.
   - Participants stressed broken text formatting and an absence of searching tools, but still commended the open-source community for driving new functionalities.
- **Installation Steps Simplify Setup**: A single-line install process for **Open Interpreter** on Mac, Windows, and Linux emerged, enabling a quick web UI experience. Friends confirmed the approach eases command execution post-installation.
   - Curious users tested the setup in the #general channel, confirming that it spares them from manual environment configurations.
- **WhatsApp Jokes and a Need for Always-On Trading Tools**: One user played with **Web WhatsApp messaging**, joking that it breathes life into mundane text chats. This exchange prompted others to share personal tech-driven everyday experiences.
   - A separate discussion focused on requiring an **always-on trading clicker**, hinting at an OS-level solution that never sleeps for continuous command execution.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Linked List Longing**: A request emerged for a **linked list** codebase that works on nightly, spotlighting the desire for quickly accessible data structures in the **Mojo** ecosystem.
   - Contributors weighed in on efficient prototypes and recommended minimal overhead as the key impetus for streamlined exploration.
- **CLI & TUI Tools Toasting with Mojo**: A developer showcased their learning process for building **CLI** and **TUI** libraries in **Mojo**, producing new utilities for command-line enthusiasts.
   - Others joked about forging a new **Mojo** shell, mirroring **bash** and **zsh**, reinforcing the community’s enthusiasm for deeper terminal integration.
- **AST Explorations & Debugging Drama**: Members shared their success with **index-style trees** like `RootAstNode` and `DisplayAstNode`, but battled segmentation faults with the **Mojo** debugger.
   - A [GitHub issue #3917](https://github.com/modularml/mojo/issues/3917) documents these crashes under **--debug-level full**, fueling a lively exchange on complex recursive structures.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **January Joy for LLM Agents Certificates**: Members confirmed **certificates** for the LLM Agents MOOC will be issued by the **end of January**, based on recent updates.
   - They advised to *stay tuned* for more details, pointing interested learners to the [Spring 2025 sign-up page](https://llmagents-learning.org/sp25).
- **Fall 2024 Fades, Spring 2025 Beckons**: Enrollment for **Fall 2024** is now closed, ending the chance to earn that session’s certificate.
   - Members encouraged joining **Spring 2025** by using the provided form, noting upcoming improvements in the curriculum.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gemini's GraphRAG Gains Gleam**: A user asked if any specific **GraphRAG** approach was used, and it turned out that **Gemini** adjusted default prompts for code-related entities to enhance extraction results.
   - They indicated this approach could add clarity to entity extraction steps, with a focus on refining **DSPy** capabilities.
- **Donor's Game Doubles Down**: A member ran a simulation of the **Donor's Game** from game theory using **DSPy** to replicate repeated strategic upgrades across multiple generations.
   - They referenced [a GitHub repository](https://github.com/CakeCrusher/cultural_evolution/blob/main/donors_game/game/orchestrator.py#L120) implementing a method from *Cultural Evolution of Cooperation among LLM Agents*, exploring ways to encourage cooperative behavior among **LLM agents**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Windows Warmth for tinygrad**: A member asked if **tinygrad** would accept pull requests for Windows bug fixes, highlighting the challenge of supporting Windows while it isn't a primary focus.
   - Another member speculated such fixes would be welcomed if they remain consistent and stable, indicating a cautious but open stance on cross-platform expansions.
- **Shapetracker Scrutiny**: A member praised the thoroughness of **tinygrad**’s documentation, referencing a [tinygrad-notes blog](https://mesozoic-egg.github.io/tinygrad-notes/20241217_st.html) for deeper insight.
   - They sought details on index and stride calculations in `Shapetracker`-based matrix memory, requesting references to clarify the underlying principles.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **WandB & MLflow Marathon**: Many noted that **Weights & Biases** offers a hosted service, while **MLflow** can be self-hosted for more control.
   - Both platforms effectively track machine learning experiments, and team preferences hinge on cost and desired workflow ownership.
- **Data Logging Delights**: Some mentioned storing experimentation results in **Postgres** or **Clickhouse** as a fallback for basic versioning.
   - They agreed it's a practical route when specialized platforms are off-limits.
- **Classical ML On the Clock**: A user questioned whether **classical machine learning** (like recommender systems and time series) is fading in the LLM era.
   - Others disagreed, contending these fields remain crucial despite the buzz around LLMs.
- **BitNet Bites Into 1-bit**: Recent work on **BitNet** showcases 1-bit LLMs matching full-precision models while lowering resource needs.
   - Researchers cited [this paper](https://arxiv.org/abs/2402.17764) describing how ternary weights lead to cheaper and efficient hardware support.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **AI Reader: PDF Chat for Low-Cost Laptops**: A user building a low-cost system in Africa wants an **AI Reader** that opens a chat GUI on PDF access to help students with tests and content, exploring [Nomic embed usage](https://nomic.ai).
   - They plan to handle content embeddings on local hardware and ask for ways to feed real-time mock exam feedback, emphasizing minimal re-indexing overhead.
- **Ranking Content Authority for Dynamic Fields**: A participant suggested measuring educational materials' authority in a way that changes as **computer science** evolves.
   - They worried about performance overhead if frequent re-indexing is required, proposing a more flexible approach to keep data current.
- **Boosting Student Transcripts in Search**: Contributors proposed giving extra weight to **student transcripts** to reflect personal academic growth in content retrieval.
   - They see a more personalized indexing method as the next shift, letting individuals track learning achievements more precisely.
- **Indexing by Subject for Enhanced Resource Control**: Users floated an indexing system focused on **subject** rather than single references like books, aiming to include supplementary articles and notes.
   - They believe this approach grants better coverage of knowledge gaps and more direct resource selection for exam prep.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Blender & AI Bond for 3D**: In **#general**, someone asked about AI-driven collaboration with **Blender** for advanced 3D annotations, referencing synergy in the community.
   - They sought partners to extend **Blender**'s capabilities, aiming for deeper integration of **AI** in geometry-based tasks.
- **Brainwaves & Beast Banter**: A participant mentioned **Animals and EEG** for **language/action mapping**, looking for groups exploring AI and neuroscience for animal studies.
   - They hoped to decode **animal behavior** with EEG data, suggesting a possible new wave of biologically informed experimentation.
- **YoavhaCohen's Tweet Surfaces**: A link to [this tweet](https://x.com/yoavhacohen/status/1875148348489113891) appeared in **#research**, without added details.
   - It remained cryptic but hinted at interest in future developments from **YoavhaCohen**, stirring curiosity.



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




### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1324484170390437989)** (188 messages🔥🔥): 

> `Windsurf Performance Issues, Cascade Credits Consumption, Codeium Plugin Suggestions, User Support Experiences, Learning and Coding Tools` 


- **Windsurf Performance Issues**: Many users reported experiencing **major performance issues** with Windsurf, including an abundance of internal errors and slow response times.
   - Some members expressed frustration with errors consuming credits, leading to dissatisfaction with the service.
- **Cascade Credits Consumption**: There were discussions regarding **Cascade** consuming credits even when internal errors occurred, which users feel is unfair.
   - Despite automated replies stating otherwise, users noted that they continued to incur charges even during failed operations.
- **Codeium Plugin Suggestions**: Users shared various tools and libraries for learning and improving coding, recommending options like **React**, **Svelte**, and **Next.js** as beginner-friendly.
   - Some users chose to use the Codeium plugin primarily for project initiation while opting for chat features for learning.
- **User Support Experiences**: Users had mixed experiences with Codeium's customer support, often receiving automated responses but lacking effective technical solutions.
   - Several users advocated for reaching out via email to describe issues and share screenshots for assistance.
- **Learning and Coding Tools**: Many users discussed their journeys in coding, comparing various coding support tools and IDEs.
   - Beginners were encouraged to leverage community resources and mentorship for improving their coding skills.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@riandoris">Rian Doris</a>: Rían Doris is the Co-Founder &amp; CEO of Flow Research Collective, the world’s leading peak performance research and training institute focused on decoding the neuroscience of flow states and helping...</li><li><a href="https://apps.apple.com/us/app/gold-fisher/id6739973000?l=zh-Hans-CN">‎Gold Fisher</a>: ‎Gold Fisher - The Ultimate Treasure Fishing Adventure!Dive into an addictive arcade fishing experience where gold hunting meets deep-sea adventure! Swing your fishing hook like a skilled prospector t...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1324471413544845438)** (194 messages🔥🔥): 

> `Performance Issues with Windsurf, DeepSeek v3 vs. Sonnet 3.6, Code Editing Errors, Prompt and Configuration Management, Credit System Feedback` 


- **Windsurf Faces Performance Challenges**: Users report an increase in errors like 'Cascade has encountered an internal error' and express frustration over performance degradation of Claude 3.5 Sonnet.
   - Multiple members illustrate issues with cascading responses, leading to increased credit consumption without effective problem-solving.
- **DeepSeek v3 vs. Sonnet 3.6 Debate**: Discussion surrounds the perceived performance of DeepSeek v3 compared to Sonnet 3.6, with some claiming DeepSeek underwhelms despite benchmark claims.
   - Members express skepticism about DeepSeek's value, preferring free alternatives like Gemini that perform competitively.
- **Errors in Code Editing Persist**: Users are encountering random disruptive changes during code editing in Windsurf, with requests for better handling of incomplete tasks.
   - Some seek suggestions for how to prompt the AI to continue previous tasks without losing context.
- **Management of Prompts and Configuration Files**: Members share strategies for saving and referencing prompt configurations to avoid repetitive explanations when using WindSurf.
   - Suggestions include using text files to store explanations or relay knowledge to the AI for future use.
- **Feedback on Credit System**: Users are critical of the credit system, particularly Premium User Prompt credits and Flex credits, advocating for better allocation of credits based on usage.
   - Concerns are raised about continuing usage without Premium credits and the importance of rollover for unused credits.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://simonwillison.net/2024/Dec/31/llms-in-2024/#llms-need-better-criticism">Things we learned about LLMs in 2024</a>: A lot has happened in the world of Large Language Models over the course of 2024. Here’s a review of things we figured out about the field in the past …</li><li><a href="https://x.com/windsurf_ai/status/1874948790928687194">Tweet from Windsurf (@windsurf_ai)</a>: Bring your Figma designs to life</li><li><a href="https://simonwillison.net/2024/Dec/31/llms-in-2024/">Things we learned about LLMs in 2024</a>: A lot has happened in the world of Large Language Models over the course of 2024. Here’s a review of things we figured out about the field in the past …</li><li><a href="https://codeium.com/plan">Plan Settings</a>: Tomorrow&#x27;s editor, today. Windsurf Editor is the first AI agent-powered IDE that keeps developers in the flow. Available today on Mac, Windows, and Linux.</li><li><a href="https://codeium.com/support,">Page Not Found | Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.com/blog/codeium-better-chat">Better Chat than ChatGPT</a>: Model, reasoning, and UX breakthroughs across the Codeium Chat stack to make it a best-in-class AI Chat developer experience.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1324470956122570905)** (198 messages🔥🔥): 

> `Linting Confusion, Deepseek Performance, Architect Mode in Aider, New AI Model Announcements, Using OpenRouter with Aider` 


- **Confusion Over Linting Requests**: A member expressed confusion over frequent linting requests from the AI tool, questioning its focus on linting over allowing manual formatting with tools like `prettier`.
   - Another member clarified that Aider automatically lints code to identify and fix problems, referencing the built-in linters for popular languages.
- **Deepseek's Performance Issues**: Users reported slow performance with Deepseek, with some experiencing as low as **1.2 TPS**, leading to frustrations over its reliability.
   - A suggestion was made that increased demand may be impacting Deepseek's speed, with calls for more servers to accommodate user needs.
- **Switching from Architect to Regular Mode**: A user sought help on how to revert from Aider's architect mode back to regular mode while still using architect features on demand.
   - It was suggested to use the command `/chat-mode` to switch modes or to implement it directly via the command line.
- **Launch of New AI Model - Sonus-1**: Discussion arose around the announcement of the **Sonus-1 Family** of AI models, highlighting their capabilities and specific use cases.
   - The launch was detailed in a [blog post](https://sonus.ai/blog/sonus-1) that outlined the different versions available: Mini, Air, and Pro.
- **Using Deepseek via OpenRouter**: Members discussed the integration of the new **Deepseek Chat v3** model with Aider, exploring the command `--model openrouter/deepseek/deepseek-chat` to access it.
   - Confirmation was given that this setup would indeed operate with the latest v3 model, enhancing users' experience with Aider.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Alibaba_Qwen">Tweet from undefined</a>: no description found</li><li><a href="https://sonus.ai/blog/sonus-1">Introducing Sonus-1: A New Era of LLMs - Goran Babarogic Product Designer</a>: Introducing Sonus, the state-of-the-art language model from Sonus AI. Experience advanced reasoning, natural language understanding, and powerful text generation capabilities.</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting and testing</a>: Automatically fix linting and testing errors.</li><li><a href="https://aider.chat/docs/troubleshooting/support.html">Using /help</a>: Use “/help &quot; to ask for help about using aider, customizing settings, troubleshooting, using LLMs, etc.</li><li><a href="https://aider.chat/docs/usage/copypaste.html">Copy/paste with web chat</a>: Aider works with LLM web chat UIs</li><li><a href="https://x.com/RubiksAI/status/1874682159379972325">Tweet from Rubik's AI (@RubiksAI)</a>: 🎉 Happy New Year! 🚀 Introducing the Sonus-1 Family: Mini, Air, Pro, and Reasoning! A new suite of models designed to meet your diverse needs in 2025 and beyond!🧵(1/7)</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: Using the code, architect, ask and help chat modes.</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3">DeepSeek V3 - API, Providers, Stats</a>: DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and coding abilities of the previous versions. Pre-trained on nearly 15 trillion tokens, the reported ev...</li><li><a href="https://github.com/mufeedvh/code2prompt">GitHub - mufeedvh/code2prompt: A CLI tool to convert your codebase into a single LLM prompt with source tree, prompt templating, and token counting.</a>: A CLI tool to convert your codebase into a single LLM prompt with source tree, prompt templating, and token counting. - mufeedvh/code2prompt</li><li><a href="https://github.com/Aider-AI/aider/issues/166)">Issues · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1324469322982363199)** (47 messages🔥): 

> `OpenRouter API issues, Aider configuration and context management, Tailwind CSS documentation integration, Graphrag tool for RAG, Feature requests for Aider` 


- **OpenRouter API troubles**: Users reported issues with using the OpenRouter API, particularly regarding authentication problems and incorrect API key setup in the config files.
   - One user confirmed their configuration worked, highlighting the importance of checking the correct model settings.
- **Configuring Aider for effective context management**: Members discussed how to better configure Aider's context management, including the use of external file context and understanding the model's metadata.
   - Specific suggestions were made regarding using commands to automatically integrate relevant documentation for smoother workflows.
- **Integrating Tailwind CSS documentation**: Aider users expressed interest in integrating Tailwind CSS documentation context directly, with suggestions including copying relevant info or using a different service's indexed documentation.
   - A proposal was made to automate consulting the Tailwind documentation through command prompts.
- **Discussion around Graphrag tool**: One user mentioned working with Microsoft's Graphrag tool, noting it seems more efficient compared to traditional RAG methods.
   - The community expressed interest in finding effective CLI tools compatible with Graphrag.
- **Future feature requests for Aider**: Users shared feature requests for Aider, emphasizing improvements in how it manages class definitions and prior context before making code changes.
   - There were suggestions for toggling features on or off to improve user experience and efficiency in coding tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.jetbrains.com/ai/ai-assistant-features/">AI Assistant Features</a>: Explore the features of JetBrains AI Assistant: context-aware code generation, advanced code completion, automated test creation, AI chat, and more. Seamlessly integrated into your JetBrains IDE to en...</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/languages.html">Supported languages</a>: Aider supports pretty much all popular coding languages.</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>: Configuring advanced settings for LLMs.</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>: How to configure aider with a yaml config file.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1324473294740787342)** (188 messages🔥🔥): 

> `OpenWebUI dataset export, Inference methods in Unsloth, Model quantization issues, VLLM for LLM inference, Fine-tuning choices for text classification` 


- **Exporting chats from OpenWebUI**: Members discussed exporting chat JSON from OpenWebUI and mentioned methods to format it for datasets.
   - Advice was given to check with the OpenWebUI owner about dataset export formats.
- **Locally running Unsloth inference**: Questions arose about running Unsloth inference locally after fine-tuning models, emphasizing the use of base models with LoRa.
   - Members were directed to vLLM documentation for implementing inference with various models.
- **Quantization challenges with Ollama**: Discussion centered on issues related to running quantized models with Ollama, with users troubleshooting errors and configurations.
   - It was noted that GGUF files in Ollama default to FP16, with suggestions on manually adjusting the Modelfile for other quantizations.
- **Choosing models for fine-tuning**: Community members shared recommendations for models suited for fine-tuning, suggesting Llama 3.1 8B and Llama 3.2 3B based on classification complexity.
   - Discussion highlighted the use of hardware like RTX 4090 for optimal performance in model training and inference.
- **Experiences with classification tasks**: Users shared experiences with LLM performance on classification tasks, reporting challenges and varying results with Llama and BERT models.
   - One participant noted achieving about 74% accuracy for classifying imbalanced datasets using Llama 3.2 3B.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.vllm.ai/en/latest/">Welcome to vLLM! &#8212; vLLM</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: no description found</li><li><a href="https://ollama.com/mesosan/lora_model">mesosan/lora_model</a>: cmc server abbie sir clone-ish</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1h1GYAGGXMhkPHr4QhRz5TLt6ZP-W8Rr5#scrollTo=J7lk6l0CuPXS">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://ollama.com/mesosan/lora_model/blobs/97f36c95b3fd">mesosan/lora_model/model</a>: cmc server abbie sir clone-ish</li><li><a href="https://github.com/unslothai/unsloth/issues/689">Does unsloth have a script for full parameter fine-tuning? · Issue #689 · unslothai/unsloth</a>: no description found</li><li><a href="https://dub.sh/bit.ch">bit.chan</a>: View the profile and 3D models by bit.chan
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1324475954734567597)** (16 messages🔥): 

> `Unsloth library installation issues, Granite training errors, Fine-tuning models for specific tasks, Using Colab with custom datasets, Understanding embedding vs fine-tuning` 


- **Unsloth library gives ModuleNotFoundError**: A user reported persistent **ModuleNotFoundError** when trying to import **unsloth** in a Kaggle notebook, despite following multiple installation steps such as `pip install`. Another member suggested checking tutorials on using Unsloth with Kaggle notebooks [here](https://docs.unsloth.ai/get-started/unsloth-notebooks).
   - Another possible solution mentioned was an issue with the **chat template**, which could affect the performance across different platforms, and provided a [link to error fixes](https://docs.unsloth.ai/basics/errors).
- **Granite training code errors reported**: Members discussed issues with errors when attempting **granite training** using the latest version, with uncertainty if the problem originated from their end. One user has been trying to resolve various installation issues but is unsure about the root cause of the errors.
   - This has raised questions about whether others are facing similar difficulties, hinting at the need for clearer documentation or support around this feature.
- **Recommendations for fine-tuning models**: A user sought recommendations for a **base model** for fine-tuning tailored to SQL generation and technical documentation inquiries. Suggestions included **Qwen 2.5** as a solid choice, alongside tools for data deduplication and formatting for training.
   - Further discussions emerged about using **RAG agents** for specific applications, demonstrating diverse approaches to achieving effective model training.
- **How to use CSV data in Colab**: A user inquired how to integrate their **CSV file** into a tutorial within Colab, seeking steps to use their own data with the provided examples. A member suggested options such as uploading the data to **Hugging Face** or uploading directly to Colab for easier access.
   - Clarifications on the best methods to load and manage data within Colab were provided, aiding newcomers in navigating the environment.
- **Seeking learning resources for Unsloth**: A user expressed interest in recommended **videos or tutorials** for getting started with **Unsloth** and installing prerequisites for training models. They are currently relying on GitHub documentation but are eager for engaging step-by-step guides.
   - Emphasis was placed on gaining a solid understanding of the tools and use cases as a way to improve efficiency in learning and implementation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=vITh0">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: See the list below for all our notebooks:</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=vITh0KVJ10qX">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/errors">Errors | Unsloth Documentation</a>: To fix any errors with your setup, see below:
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1324672299445784598)** (8 messages🔥): 

> `Process Reinforcement through Implicit Rewards, O1 Reproduction Efforts, Fudan Report, Reinforcement Learning Code` 


- **Fudan Report on O1 Reproduction Efforts Gains Attention**: The Fudan report is mentioned as the most comprehensive resource found regarding the **O1 reproduction efforts** so far, available [here](https://arxiv.org/pdf/2412.14135).
   - *Pretty much the most comprehensive I found so far about the O1 reproduction efforts* was remarked by a member.
- **Process Reinforcement Paper Lacks Code**: Despite the **Process Reinforcement through Implicit Rewards** paper being a good read, members lamented its lack of accompanying **code** for reinforcement learning.
   - One member expressed disappointment saying, *sad no code on the RL*, while another mentioned that updates about code were forthcoming.
- **Encouragement for Continued Development**: A member noted that information about the **Process Reinforcement paper** included a promise of code availability soon, citing it as a *Work in Progress*.
   - The anticipation for the release of code reflects community interest in practical applications stemming from the research.


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1324470663842627584)** (170 messages🔥🔥): 

> `DeepSeek V3 Model Discussion, Email Spoofing Incident, Using Cursor for Project Development, Website Design Inspiration, Language Selection for Marketplace Development` 


- **Discussion on DeepSeek V3 Model**: Users expressed concerns about the **DeepSeek V3 model** storing code and training on user data, leading to privacy issues.
   - Some members highlighted its capabilities while questioning the benefits and implications of using such models in their projects.
- **Email Spoofing Incident and Community Reactions**: A user received a humorous and insulting email from a spoofed address, prompting laughter and discussions on the validity of the spoofing.
   - Members jokingly encouraged the creator of the spoofed email to start a business around sending such emails, pointing out the absurdity of the situation.
- **Utilizing Cursor for Rapid Project Development**: One user shared their positive experience using **Cursor AI** to rapidly complete a project involving **SignalR**, claiming a significant reduction in development time.
   - This prompted discussions on how AI tools like Cursor are transforming the development landscape, making complex tasks manageable.
- **Finding Website Design Inspiration**: A user sought recommendations for websites that offer design inspiration, receiving multiple suggestions from the community.
   - Sites like **land-book.com** and **godly.website** were shared, with members expressing gratitude for the helpful resources.
- **Choosing the Right Language for a Marketplace**: Members discussed the most suitable programming language for building a marketplace, with suggestions leaning towards **JavaScript** for its ease of use.
   - They emphasized that while numerous languages could be suitable, JavaScript allows for efficient development across both frontend and backend.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://microsoft.github.io/monaco-editor/">Monaco Editor</a>: no description found</li><li><a href="https://x.com/liamesp/status/1869319333954089218?s=46">Tweet from liam (@liamesp)</a>: now, object + brush selections in @krea_ai editor</li><li><a href="https://tenor.com/view/drinks-gif-8100232030695928923">Drinks GIF - Drinks - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://forum.cursor.com/t/error-request-type-deprecated-when-using-docs-in-composer-agent-mode/38610/14">[RESOLVED] Error: Request Type Deprecated when using @docs in Composer Agent Mode</a>: Is this working now? We pushed a fix about an hour ago.</li><li><a href="https://www.youtube.com/watch?v=NCaRixtXNIo"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1324472338280087613)** (139 messages🔥🔥): 

> `Swarm Library Release, Research Proposal Assistance, NLP Workshops at ICML, Fine-Tuning Models, Training Costs for AI Models` 


- **Swarm Library now available on NPM**: The new library, **Swarm**, aimed at creating multi-agent AI systems, has just been published on [NPM](https://www.npmjs.com/package/agentswarm). It boasts flexibility and model agnosticism, suitable for various AI collaborations.
   - The author noted that this package offers better patterns and modifications compared to the existing **OpenAI Swarm**.
- **Need for Research Proposal Support**: A member inquired about support for research proposals, prompting others to suggest posting in specific channels for community help. The server seems open to collaborative efforts in research-related inquiries.
   - This indicates an active community willing to assist with academic ambitions, especially in AI research.
- **Call for Participation in NLP Workshops**: A proposal for a **PersonaNLP workshop** at ICML 2025 was introduced, inviting interest for paper submissions and shared tasks. This reflects ongoing efforts to promote collaboration within the NLP field.
   - Interested members are encouraged to join discussions about the proposal in shared channels.
- **Fine-Tuning Models on Personal Hardware**: Members discussed the feasibility of fine-tuning models on personal GPUs like the **4090**, with recommendations for using tools such as **Unsloth**, **Axolotl**, or **Llama Factory**. This reflects a growing interest in locally experimenting with AI models.
   - While cloud options like Amazon Bedrock are noted for privacy, local options provide flexibility for individual developers.
- **Expensive Training Costs for AI Models**: A member highlighted the high costs associated with training large AI models, referencing prices as high as **$600,000** for larger models. Discussion focused on the accessibility of such endeavors for individuals without large-scale resources.
   - Various methods were proposed for efficient training, reflecting a keen interest in making AI research more feasible.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.16528">LoQT: Low-Rank Adapters for Quantized Pretraining</a>: Despite advances using low-rank adapters and quantization, pretraining of large models on consumer hardware has not been possible without model sharding, offloading during training, or per-layer gradi...</li><li><a href="https://runpod.io?ref=jgbvgh5q">RunPod - The Cloud Built for AI</a>: Develop, train, and scale AI models in one cloud. Spin up on-demand GPUs with GPU Cloud, scale ML inference with Serverless.</li><li><a href="https://www.youtube.com/watch?v=XwL_cRuXM2E"> - YouTube</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=39224534">If you read around, training a 7B model costs on the order of $85,000; the 1.4 s... | Hacker News</a>: no description found</li><li><a href="https://x.com/moinnade">Tweet from FxTwitter / FixupX</a>: Sorry, that user doesn't exist :(</li><li><a href="https://x.com/moinnadeem/status/1681371166999707648">Tweet from Moin Nadeem (@moinnadeem)</a>: This is an important plot from the LLaMa 2 paper. It directly outlines the pre-training hours for the model!Costs below, assuming $1.50 / A100 from @LambdaAPI:- the 7B model cost $276,480!- the 13B mo...</li><li><a href="https://www.npmjs.com/package/agentswarm">agentswarm</a>: LLM-agnostic typescript framework for creating OpenAI-style Swarm agents with the Vercel AI SDK. Latest version: 0.0.4, last published: a day ago. Start using agentswarm in your project by running `np...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1324503207468077056)** (24 messages🔥): 

> `Hermes Training Data, VLM Use Cases, Fine-Tuning Models, Model Weight Distribution, Mergoo Introduction` 


- **Controversy Around Hermes Training Data**: A member highlighted that there is **no explicit training data** in Hermes that simulates sex scenes, prompting speculation on whether this affects model performance.
   - Another member commented that this absence might actually hinder the model's capacity to leverage certain knowledge that could enhance its broader performance.
- **Exploring VLM Use Cases**: Discussion centered around the **practical applications** of Vision Language Models (VLMs) with suggestions including validating web development and robotics.
   - One member expressed curiosity about identifying computer elements and proposed that this might be a task for special fine-tuning processes.
- **Current Models for Fine-Tuning**: Members shared that **Llama**, **DeepSeek**, and **Qwen** are among the most commonly used models for post-training and fine-tuning.
   - Questions arose regarding their licenses for commercial use and potential tokenization considerations.
- **Inefficiencies in Model Weight Distribution**: Insights were shared regarding the **distribution of K and Q weights** in the Llama2 model, suggesting it indicates some inefficiency in encoding.
   - Attached images sparked further analysis on the amplitude of weights, implying that not all tokens carry equal importance in outputs.
- **Introduction to Mergoo**: A newcomer inquired about **Mergoo**, likening the community dynamics to that of r/localllama from two years ago.
   - Responses indicated a mixed sentiment about the state of r/localllama, suggesting that recent changes have led to dissatisfaction among some users.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1324501544191660195)** (119 messages🔥🔥): 

> `Extractor.io Website, LM Studio Image Generation, Qwen2-VL Model Limitations, AI Training with Internet Data, Model Performance Issues` 


- **Extractor.io website confusion**: A user reported finding the **Extractor.io** website, only to discover that it no longer exists which raised some eyebrows.
   - Another user suggested it might be replaced by [this link](https://llm.extractum.io/list/) for a curated list of language models.
- **LM Studio does not support image generation**: Chat participants confirmed that **LM Studio** cannot generate images directly, as it only processes LLMs, not vision models.
   - Alternatives like **Pixtral** were mentioned but are only available on Mac through the MLX Engine.
- **Qwen2-VL model lacks vision capabilities**: Discussion indicated that the **Qwen2-VL-7B-Instruct-abliterated** model does not have a vision adapter, thus it cannot process images.
   - Participants highlighted the need for proper quantization from the original model to make full use of its capabilities.
- **Challenges in AI training using full internet data**: A user expressed interest in training an AI on the entire internet, but others pointed out that it's impractical due to data quality and availability issues.
   - Several participants emphasized that **quality matters more than quantity**, and bad data can harm model performance.
- **Model performance and parameter discussions**: Users discussed their experiences with various models, noting performance issues like repetitiveness, especially with **Mistral Nemo Instruct**.
   - They also debated the required model size and parameters, considering that better models may yield superior results even with fewer parameters.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llm.extractum.io/list/">All Large Language Models</a>: A Curated List of the Large and Small Language Models (Open-Source LLMs and SLMs). All Large Language Models with Dynamic Sorting and Filtering.</li><li><a href="https://www.shepbryan.com/blog/what-is-gguf">What is GGUF? A Beginner&#39;s Guide &mdash; Shep Bryan</a>: You've probably seen the term GGUF as you're looking for local AI models to run. But what is it? And how does it enable the use of cutting-edge LLMs on your own devices?</li><li><a href="https://huggingface.co/mradermacher/Qwen2-VL-7B-Instruct-abliterated-GGUF/tree/main">mradermacher/Qwen2-VL-7B-Instruct-abliterated-GGUF at main</a>: no description found</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio - Beta Releases</a>: Beta and Release Candidate versions of LM Studio</li><li><a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">ggml/docs/gguf.md at master · ggerganov/ggml</a>: Tensor library for machine learning. Contribute to ggerganov/ggml development by creating an account on GitHub.</li><li><a href="https://github.com/lllyasviel/Fooocus">GitHub - lllyasviel/Fooocus: Focus on prompting and generating</a>: Focus on prompting and generating. Contribute to lllyasviel/Fooocus development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1324510979618504788)** (38 messages🔥): 

> `Local LLM Usage, API Concerns, GPU vs CPU Utilization, Quest Generation, Hardware Recommendations` 


- **Preferring Local LLMs Over APIs**: @octaviagoetia expressed a preference for using **GPUs** over **online APIs**, citing concerns over **TOS restrictions** and potential latency issues.
   - @heychazza agreed, highlighting the benefits of using a beefy laptop for **tinkering** with LLMs without worrying about unexpected costs.
- **Successful Quest Generation Using Llama 3**: @octaviagoetia is currently using **Llama 3.1** for generating quest lines, with plans to upgrade to **Llama 3.2 1b**, finding it more creative and efficient.
   - This method allows for **dynamic NPC interactions**, reducing the effort needed for manual quest creation.
- **Confirming GPU Usage in LLMs**: @antonyj0666 inquired about enabling **GPU** utilization for their local LLM setup using a **4070 Ti Super** graphics card.
   - @heyitsyorkie confirmed that selecting models with confirmed **GPU offload capabilities** is essential for proper functionality.
- **Identifying GPU Usage During Inference**: To confirm if the program is utilizing the GPU, @christianazinn suggested checking **Task Manager** during model inference.
   - If there is a spike in GPU performance metrics while running the model, it indicates successful **GPU** engagement.
- **Different GPU Offload Capabilities**: @mrhoopers shared insights regarding various models with **GPU offload capabilities**, highlighting green indicators for full support.
   - He clarified that while green indicates potential GPU use, it does not guarantee that the **model is currently running on the GPU**.



**Link mentioned**: <a href="https://model.lmstudio.ai/download/NousResearch/Hermes-3-Llama-3.1-8B-GGUF">Download and run NousResearch/Hermes-3-Llama-3.1-8B-GGUF in LM Studio</a>: Use NousResearch/Hermes-3-Llama-3.1-8B-GGUF locally in your LM Studio

  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1324478524706263071)** (15 messages🔥): 

> `Various media references, Conflict of interest in studies, Translation checks, Long-term mycelium storage techniques, Turing Test discussions` 


- **Media Links Shared**: Users shared various media links, including [this article](https://www.akashq.com/post/23a45b75-75ed-4741-94a6-0252f493748a) and [a Spotify show](https://open.spotify.com/show/5X89wBkhOVCYJJR9NsntVJ?si=f87a4ab74070424e), highlighting a mix of topics.
   - These links sparked additional discussions about their content and relevance.
- **Concerns Over Study Bias**: *A user mentioned the importance of conflict of interest disclosures* when reviewing studies, stressing transparency about who benefits from the research.
   - *Retraction Watch* was pointed to as a valuable resource for identifying studies that were retracted yet still cited widely.
- **Mycology Storage Techniques**: *A user shared insights on using slants for long-term mycelium culture storage*, highlighting a linked audio resource on the technique.
   - Discussions included tips on leveraging different browsers for saving audio files effectively.
- **Turing Test Lecture Insights**: *A discussion emerged around a lecture focused on how to pass the Turing Test*, complemented by a linked audio file of the discussion.
   - This led to an exchange of methods and strategies to improve performance in such tests.
- **Converting Audio Files**: *Users discussed methods for converting audio files from WAV to MP3*, mentioning tools like Audacity and Adobe Audition.
   - One user expressed appreciation for these tips, indicating practical applications for audio file management.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://open.spotify.com/show/5X89wBkhOVCYJJR9NsntVJ?si=f87a4ab74070424e">WeTalk Business</a>: Podcast · Bilal Rmaili · Buy land, ‘cause God ain’t making any more of it. Inquiries: @3amfxct</li><li><a href="https://www.akashq.com/post/f5a0eff9-6ad7-4137-b35c-186f243ce3c7">What happened in history Jan 3?</a>: What happened in history Jan 3? by This Day in History</li><li><a href="https://www.akashq.com/post/23a45b75-75ed-4741-94a6-0252f493748a">What happened on Jan 2?</a>: What happened on Jan 2? by This Day in History
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1324475851038789714)** (101 messages🔥🔥): 

> `Sharing Notebooks Issues, Beta Testing Experience, Multilingual Features, AI Summarization Product, Customization Function Use` 


- **Sharing Notebooks Faces Problems**: Users are encountering issues when attempting to share notebooks via email, with no notifications or shared notebooks appearing in the recipient's interface.
   - This has been confirmed by multiple users trying both ways of sharing, indicating a potential system glitch.
- **Challenges with Beta Features**: Several users are experiencing long wait times when joining the beta interaction mode, often encountering endless loading screens.
   - Issues may stem from microphone access problems, leading to concerns over input levels and audio device settings.
- **Multilingual Capabilities in AI**: While the audio overview feature is officially only available in English, users have experimented with generating content in other languages through creative prompting.
   - Feedback suggests that while translations can work, the quality notably varies, especially with non-European languages.
- **AI Summarization Product Gets Attention**: A user is promoting a beta testing opportunity for a new AI summarization product that has garnered significant success in the Korean market.
   - Interest has been shown by community members wanting to test this product, which promises streamlined summarization capabilities.
- **Concerns Over Customization Functionality**: Concerns were raised regarding the potential misuse of the customization function to alter system prompts and exploit AI capabilities unfairly.
   - Users discussed the importance of maintaining a balance between creative freedom and ethical AI practices, drawing parallels with NSFW content management.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://akashq.com">Akas: home to AI podcasts</a>: no description found</li><li><a href="https://notebooklm.google.com/notebook/05adb4e6-7905-4f0d-abe5-1886faf6e4f1/audio">no title found</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219?sjid=9595575708425202127-NC&visit_id=638697544924377600-2451215312&p=plus&rd=1">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://youtu.be/aG0ixD3OY80"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1324779928939659396)** (3 messages): 

> `Bolt code handling issues, UI implementation challenges, Web app development tips` 


- **Bolt's 'Creative' Code Behavior Frustrates Users**: Members expressed frustration that despite clear instructions, **Bolt** sometimes comments out or deletes code on its own.
   - *'Even when trying to be explicit, it still does it sometimes'* echoed concerns from multiple users.
- **JSX Grid Code Mismanagement**: A user reported that when providing a **JSX grid code** for their UI, **Bolt** entirely redesigned it, ignoring the original input.
   - They sought advice on how to prevent **Bolt** from being so 'creative' with their requests.
- **New Year, New Apps: Seeking Help with Logic/AI**: A member shared excitement about developing web apps with **Bolt** but admitted to struggling with implementing **Logic/AI** features.
   - They noted that visually, the results were excellent, but functionality needed enhancements and asked for tips.


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1324468876477595648)** (110 messages🔥🔥): 

> `Billing issues with Bolt, Debugging in Bolt, Integrating APIs with Bolt, Building frontend with mock data, Using Supabase with Bolt` 


- **Users report fast token consumption on Bolt**: Multiple users expressed concerns about rapidly depleting their token balance on Bolt, indicating dissatisfaction with the rate of consumption.
   - One user suggested that using mock data for frontend development might help mitigate quick token usage.
- **Issues with blank preview screen**: Several users encountered a blank screen while using Bolt, prompting suggestions to check for errors in the Bolt terminal or to close unused tabs.
   - One user resolved the issue by restarting their PC, indicating that system performance might affect the preview functionality.
- **Netlify callback URL returns 404**: A user faced a 404 error when posting to a Netlify callback URL from a form built on Bolt, questioning if it was a feature limitation.
   - It was suggested that the user utilize Netlify Functions instead, as Netlify’s static hosting doesn’t accept POST requests directly.
- **Customer support for Bolt**: Users inquired about customer support availability for billing and technical issues, with responses pointing to limited support options.
   - It's noted that customer support mainly addresses billing issues rather than providing technical assistance.
- **Interest in using Supabase for email logins**: Users shared their experiences of incorporating Supabase for email authentication in their applications built with Bolt.
   - Testing with local storage for frontend features was recommended as a strategy to manage user roles effectively without immediate deployment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://boltstudio.ai/">BoltStudio.ai | Full Stack Prompt Engineering</a>: no description found</li><li><a href="https://answers.netlify.com/t/receiving-post-requests/86669">Receiving POST requests</a>: Hello! I’m having an issue with POST requests.  I have a separate node.js server that will occasionally send POST requests to my netlify site. However, I have been researching for hours and am still n...</li><li><a href="https://github.com/stackblitz-labs">StackBlitz Labs</a>: StackBlitz Labs has 2 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/stackblitz">StackBlitz</a>: StackBlitz has 45 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1324480026220953600)** (103 messages🔥🔥): 

> `Perplexity O1 Feature, ChatGPT vs. Perplexity, Grok Model Opinions, UI Changes in Perplexity, Using AI Subscriptions` 


- **Mixed Reviews on Perplexity's O1 Feature**: Users are experiencing issues with the O1 feature in Perplexity, with reports of incorrect formatting and limitations in daily searches. *One user expressed frustration, stating, 'bruh what a hassle, and on top of that only 10 searches daily.'*
   - Another user mentioned that they had been using the free version of Grok on X, leading to curiosity about its capabilities compared to others.
- **Comparing AI Tools: Perplexity and ChatGPT**: Users discussed the differences between Perplexity and ChatGPT, noting that Perplexity is superior for search capabilities, while ChatGPT may excel in non-search tasks due to its larger context. One user remarked, 'Definitely worth it for me because opus is unlimited and the context memory is high asf for all their models.'
- **Grok Model Gets Mixed Feedback**: Grok received some critical feedback; one user claimed it was 'the worst model I have used' despite its cost-effectiveness. Others preferred the 3.5 Sonnet model for its strong performance during tasks.
- **Perplexity's Recent UI Changes Spark Discussion**: Users noted recent UI changes in Perplexity, such as added stock and weather information on the homepage, with questions on how to disable these features. One user cleared their cache to resolve issues with unwanted display elements, stating, 'Was gving me flashbacks of the terrible browser home pages!'
- **Concerns Over Subscription Solutions and User Feedback**: Conversations included users discussing their subscriptions to various AI tools and their experiences with features like unlimited queries. Some expressed relief at not having to pay for certain services due to unexpected glitches in their subscriptions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/pplxsupply/status/1875019712268263658?s=46">Tweet from Perplexity Supply (@PPLXsupply)</a>: ask.learn.do.</li><li><a href="https://x.com/aravsrinivas/status/1874943854849425780?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: If you&#39;re an Android user of Perplexity, please DM for early access testing of an upcoming cool thing.</li><li><a href="https://tenor.com/view/new-year-happy-new-year-2025-happynewyear-new-years-gif-11684221509638957422">New Year Happy New Year GIF - New year Happy new year 2025 - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://chrisbora.substack.com/p/wbs-framework">Beyond Prompting: The What-Boundaries-Success Framework</a>: How particle physics and search systems led to a fundamental breakthrough in AI Control</li><li><a href="https://github.com/cbora/aispec">GitHub - cbora/aispec: A specification language for AI-first development that shifts focus from implementation to intent through structured solution space reduction</a>: A specification language for AI-first development that shifts focus from implementation to intent through structured solution space reduction - cbora/aispec
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1324589625037029497)** (7 messages): 

> `Musk Lawsuit Support, AI Interview Preparation, Oceans and CO2 Absorption, 2025 Predictions, Perplexity AI Origins` 


- **Godfather of AI Supports Musk Lawsuit**: The *Godfather of AI* has publicly backed Musk's lawsuit, sparking discussions about its implications for the industry. This was highlighted in a recent [YouTube video](https://www.youtube.com/embed/tRAGov9VRxQ) discussing the ongoing controversies.
   - The video also covers the distortion of the **electric grid** by data centers and predictions for 2025.
- **Tips to Ace AI Interview Questions in 2025**: A member shared a guide on how to effectively tackle AI-related questions during job interviews in 2025, emphasizing preparation strategies. More details can be found [here](https://www.perplexity.ai/search/the-job-interview-question-of-geScATofQC.NYw5MqWsyiA).
   - The guide is particularly relevant as AI roles become increasingly competitive.
- **Oceans Absorb 1/3 of Human CO2 Emissions**: An article discussed how oceans absorb approximately **one-third** of human CO2 emissions, highlighting their crucial role in climate regulation. You can read more about it [here](https://www.perplexity.ai/page/oceans-absorb-1-3-of-human-co2-S586TEA4QN.ngghjoWC0nQ).
   - This reinforces the importance of ocean conservation in climate action efforts.
- **Perplexity AI's Conceptual Idea**: The discussion included the origins of the idea behind Perplexity AI, with members sharing related insights. For deeper context, check out this link [here](https://www.perplexity.ai/search/when-was-the-idea-behind-perpl-.JlEn.TrRl.zj824uT_xwQ).
   - Understanding its origins is essential for appreciating the platform's ongoing developments.



**Link mentioned**: <a href="https://www.youtube.com/embed/tRAGov9VRxQ">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1324678621754818657)** (1 messages): 

> `API server location, Chatbot integration, Token utilization` 


- **API in Europe for Better Performance**: A member expressed optimism that if the **API** has its server location in **Europe** and matches the performance of **pro search**, it will be integrated into their **1.5 billion tokens chatbot**.
   - They noted a hope for improvements in the chatbot's functionality following this potential integration.
- **Expectation for Token Utilization**: The member emphasized the potential of utilizing **1.5 billion tokens** effectively with the new API integration.
   - They are looking forward to enhancing the chatbot's capabilities through this approach.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1324467807093522544)** (86 messages🔥🔥): 

> `OpenRouter authentication issues, DeepSeek performance, Model recommendations for structured output, Janitor AI integration with OpenRouter, Payment processing issues` 


- **OpenRouter authentication issues**: Several users reported issues with using OpenRouter on n8n, with messages indicating **'Unauthorized'** errors despite having credits loaded.
   - *Matt070655* mentioned changing the HTTPS address and adding the API key but still encountered connection refusals.
- **DeepSeek performance woes**: Users expressed frustration over **DeepSeek's slow performance**, with one noting a low **0.6 TPS**.
   - Concerns were raised that the current demand might not have been adequately predicted, leading to degraded experiences.
- **Model recommendations for structured output**: A user sought alternatives to **gpt-4-mini** for structured JSON output, finding options limited.
   - Others suggested models like **Gemini Flash**, with discussions about version effectiveness and anticipated rate limitations.
- **Janitor AI integration with OpenRouter**: Assistance was provided on how to set up OpenRouter with Janitor AI, emphasizing adjustments in settings for URL and API compatibility.
   - Instructions highlighted toggling options in the advanced settings for better integration.
- **Payment processing issues**: Users reported difficulties in processing payments on OpenRouter, specifically with certain credit cards failing while others worked.
   - One user noted a capital one card consistently failed, while an alternative card successfully processed the payment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.hyperbolic.xyz/models/deepseek-v3">Hyperbolic AI Dashboard</a>: no description found</li><li><a href="https://www.litellm.ai/">LiteLLM</a>: LiteLLM handles loadbalancing, fallbacks and spend tracking across 100+ LLMs. all in the OpenAI format</li><li><a href="https://openrouter.ai/api/v1">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://status.deepseek.com/">DeepSeek Service Status</a>: no description found</li><li><a href="https://github.com/lm-sys/RouteLLM">GitHub - lm-sys/RouteLLM: A framework for serving and evaluating LLM routers - save LLM costs without compromising quality!</a>: A framework for serving and evaluating LLM routers - save LLM costs without compromising quality! - lm-sys/RouteLLM</li><li><a href="https://github.com/BerriAI/litellm">GitHub - BerriAI/litellm: Python SDK, Proxy Server (LLM Gateway) to call 100+ LLM APIs in OpenAI format - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq]</a>: Python SDK, Proxy Server (LLM Gateway) to call 100+ LLM APIs in OpenAI format - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1324467935972163704)** (67 messages🔥🔥): 

> `SwarmUI vs Forge, SANA and Omnigen Models, Video Generation Models, Text Output in Images, GPU Recommendations for AI` 


- **SwarmUI offers simplicity with ComfyUI performance**: Members discussed that SwarmUI utilizes ComfyUI as its backend, providing a user-friendly graphical interface while maintaining performance akin to ComfyUI.
   - *
- **SANA is fast but not always worth the space**: Users debated the effectiveness of SANA and Omnigen, noting that while SANA is *really fast*, it trails behind Flux in performance and might not justify HDD usage.
   - Opinions on Omnigen indicated it's *pretty slow* and may not deliver image quality as good as SDXL.
- **Rapid advancements in video generation**: There was excitement over models like LTXVideo, which reportedly increased rendering speed significantly on new GPUs without quality loss.
   - HunyuanVideo also received praise for improvements, enabling efficient processing with fewer steps compared to previous versions.
- **Current best models for text in images**: For generating text in images, Flux Dev was mentioned as a leading open-source model comparable to top closed models like Ideogramv2 and DALL-E 3.
   - The *best closed model* recommended was noted to be Flux 1.1 Ultra.
- **GPU advice for AI workload**: When discussing hardware, it was suggested to invest in a GPU like the RTX series for optimal performance in AI tasks, with a particular emphasis on waiting for upcoming models to potentially lower prices.
   - Members advised a minimum of 32GB RAM for effective image generation, emphasizing the importance of VRAM alongside RAM.



**Link mentioned**: <a href="https://github.com/butaixianran/Stable-Diffusion-Webui-Civitai-Helper">GitHub - butaixianran/Stable-Diffusion-Webui-Civitai-Helper: Stable Diffusion Webui Extension for Civitai, to manage your model much more easily.</a>: Stable Diffusion Webui Extension for Civitai, to manage your model much more easily. - butaixianran/Stable-Diffusion-Webui-Civitai-Helper

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1324528711680266331)** (21 messages🔥): 

> `RTX 50xx series VRAM limitations, Integration of SSDs with GPUs, Memory performance between VRAM and RAM, Cache hierarchy in GPUs` 


- **RTX 50xx series VRAM size debates**: Users discussed the possibility that **VRAM size limitations** are imposed to prevent cannibalization of NVIDIA's product lineup, suggesting a possible ulterior motive behind the pricing.
   - One noted, *'Memes aside, why can't I have this and/or would even that much delay make it useless for AI?'* highlighting user frustration over the limitations.
- **Examining SSD integration with GPUs**: A user recalled that **some AMD cards** might have integrated SSDs, but was reminded that it likely used **unused PCIe lanes** rather than full integration.
   - This led to discussions about the future of this technology in relation to GPUs.
- **Understanding VRAM performance vs regular RAM**: A user questioned if the **implied latency** of regular RAM being 4x slower was correct, given the context of the **gh200 form factor**.
   - Another explained that while all **RTX GPUs** max out around **1TB VRAM bandwidth**, offloading to RAM still poses challenges.
- **Thoughts on reclassifying VRAM as cache**: A user suggested renaming **20+ GB VRAM** to **L3 Cache** and incorporating a larger memory component to optimize performance.
   - However, concerns were raised about potential performance issues if weights are read only.
- **Pipelining and memory control in GPUs**: A user speculated that effective pipelining requires the controller to handle R/W commands between **VRAM and RAM sticks**, unlike integrated cache which operates independently.
   - This raises questions about the **worst case performance** implications for model inference.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1324497407261675532)** (35 messages🔥): 

> `Attention on Attention Weights, Quartic Transformer, Ring Attention, Higher-Order Attention, HYMBA and SWA` 


- **Exploring Attention on Weight Matrices**: Members discussed the idea of performing attention on attention weight matrices, with some referencing a concept similar to higher-order attention that could improve representation connections.
   - One participant pointed out that the application in Mamba or SSM Conv papers suggested that two convolutions could correspond to quad attention.
- **Quartic Transformer Investigated**: A link to the [Quartic Transformer](https://github.com/lucidrains/quartic-transformer) was mentioned, exploring performance across nodes in the attention mechanism while disregarding efficiency.
   - This approach sparked a discussion about potentially similar concepts found in edge or line graphs and hypergraphs.
- **Revisiting Ring Attention Papers**: A member clarified that when discussing 'ring attention', most refer to the second paper which popularized the concept, showing impressive context lengths.
   - This led to insights on the evolution of context length capabilities from the initial proposal to newer models.
- **HYMBA’s Efficiency Concerns**: There was speculation that HYMBA, which employs full attention on a few layers, might contradict the purpose of mixed models like SWA and SSM.
   - Participants debated the efficiency of trading full attention for sliding window attention on fewer tokens to enhance cross-window representation.
- **Pytorch Flex Attention Challenges**: The conversation wrapped up with frustrations regarding bugs in Pytorch's Flex Attention that hindered testing for some members' implementations.
   - Despite the potential benefits of torch.compile, it has been noted that it often encounters bugs when used with less mature models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.15371">DiSHA: Dimension-Sharding Adaptation with Fast Convergence and Fast Computation</a>: Low-Rank Adaptation (LoRA) leverages the low intrinsic rank of weight updates in Large Language Models (LLMs), establishing a Parameter-Efficient Fine-Tuning (PEFT) paradigm. However, LoRA suffers fro...</li><li><a href="https://en.m.wikipedia.org/wiki/Line_graph">Line graph - Wikipedia</a>: no description found</li><li><a href="https://github.com/lucidrains/quartic-transformer">GitHub - lucidrains/quartic-transformer: Exploring an idea where one forgets about efficiency and carries out attention across each edge of the nodes (tokens)</a>: Exploring an idea where one forgets about efficiency and carries out attention across each edge of the nodes (tokens) - lucidrains/quartic-transformer</li><li><a href="https://huggingface.co/datasets/lennart-finke/SimpleStories">lennart-finke/SimpleStories · Datasets at Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2112.00578">Systematic Generalization with Edge Transformers</a>: Recent research suggests that systematic generalization in natural language understanding remains a challenge for state-of-the-art neural models such as Transformers and Graph Neural Networks. To tack...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1324572958328361112)** (15 messages🔥): 

> `LLM developments in 2024, Text extraction tools evaluation, Image generation trends, SmallThinker-3B model performance, OLMo2 tech report release` 


- **LLM developments continue to accelerate in 2024**: With the upcoming article, [LLMs in 2024](https://simonwillison.net/2024/Dec/31/llms-in-2024/), a review of significant developments in the **Large Language Models** sector was shared, highlighting breaking barriers and emerging metrics over the past year.
   - Key themes include advancements in multimodal capabilities and significant price drops attributed to increased competition.
- **Community discusses data extraction tools**: A member posted a [benchmark study](https://cdn.discordapp.com/attachments/1075282825051385876/1324793970726928537/A_Comparative_Benchmarking_Evaluation_of_Text_Extraction_Tools.pdf) evaluating text extraction tools against regulatory documents, sparking curiosity about effective data extraction methods from complex sheets.
   - Community members provided insights, discussing successful combinations of **pdfium** and **tesseract** for document processing.
- **Trends in meme culture with image generation**: In November 2023, a meme trend emerged where users prompted ChatGPT to generate increasingly modified images, revealing the engaging outcomes of this participatory form of content creation.
   - Highlighted examples convey a humorous evolution of images, such as transforming an average guy into a 'bro' or portraying Santa Claus with a serious demeanor.
- **Introducing SmallThinker-3B model**: The **SmallThinker-3B-preview** model was introduced as a new fine-tuned model, showcasing significant benchmark performance improvements over **Qwen2.5-3B-Instruct** across various evaluations.
   - This model is particularly aimed at edge deployment due to its compact size, making it suitable for resource-constrained environments.
- **Release of OLMo2 tech report**: [OLMo 2 tech report](https://x.com/soldni/status/1875266934943649808?s=46) was released, offering a deep dive into four critical components of the LLM development pipeline across a detailed 50+ pages.
   - The report aims to provide insights into essential methodologies and practices within the realm of LLM development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.val.town/blog/fast-follow/">What we learned copying all the best code assistants</a>: From GitHub Copilot to ChatGPT to Claude Artifacts, how Val Town borrowed the best of all the code generation tools</li><li><a href="https://simonwillison.net/2024/Dec/31/llms-in-2024/">Things we learned about LLMs in 2024</a>: A lot has happened in the world of Large Language Models over the course of 2024. Here’s a review of things we figured out about the field in the past …</li><li><a href="https://minimaxir.com/2025/01/write-better-code/">Can LLMs write better code if you keep asking them to “write better code”?</a>: Most coders want AI to write code faster: I want AI to write FASTER CODE.</li><li><a href="http://openlayer.com?">Openlayer: Enterprise-grade AI quality, evaluation and monitoring</a>: Openlayer helps you test and monitor high-quality AI systems.</li><li><a href="https://huggingface.co/PowerInfer/SmallThinker-3B-Preview">PowerInfer/SmallThinker-3B-Preview · Hugging Face</a>: no description found</li><li><a href="https://x.com/reach_vb/status/1874868847754580431">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: chat, is this real? smol QwQ? https://huggingface.co/PowerInfer/SmallThinker-3B-Preview</li><li><a href="https://x.com/WolframRvnwlf/status/1874889165919384057">Tweet from Wolfram Ravenwolf 🐺🐦‍⬛ (@WolframRvnwlf)</a>: New year, new benchmarks! Tested some new models (DeepSeek-V3, QVQ-72B-Preview, Falcon3 10B) that came out after my latest report, and some &#34;older&#34; ones (Llama 3.3 70B Instruct, Llama 3.1 Nemo...</li><li><a href="https://x.com/soldni/status/1875266934943649808?s=46">Tweet from Luca Soldaini 🎀 (@soldni)</a>: OLMo 2 tech report is outWe get in the weeds with this one, with 50+ pages on 4 crucial components of LLM development pipeline:
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1324525366550728715)** (4 messages): 

> `AI Engineer Summit, AI Engineer World's Fair, Understanding Transformers` 


- **AI Engineer Summit Announced!**: Get hyped for our next invite-only [AI Engineer Summit](https://www.latent.space/p/2025-summit)! The first one sold out with a **10:1 applicant ratio** and we are excited to bring it back.
   - Our previous **multitrack World's Fair** sold out **3k seats** and racked up **>1m views** on YouTube, showcasing the top names in AI engineering.
- **Opportunity for Guest Publishing on Latent Space**: A call for collaboration has been announced; anyone interested in guest publishing a good Transformers explainer can contribute! This is a serious opportunity for engagement within the community.
   - For more information, check out the discussion link [here](https://discord.com/channels/822583790773862470/1323930993786228858/1324157846031700061).
- **Understanding Transformers at Latent Space**: [Understanding Transformers](https://x.com/sannykimchi/status/1176517584319127553) provides an insightful list of resources for learning how Transformers operate, from **self-attention** to **positional encodings**. Recent advances like **BERT**, **XLNet**, and **GPT-2** have been influenced extensively by this architecture.
   - *Sannykimchi* shares a concise roadmap of concepts to study to dive into Transformers effectively, catering to those eager to grasp this important topic.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.latent.space/p/2025-summit">Announcing AI Engineer Summit NYC: All in on Agent Engineering + Leadership</a>: Announcing the theme of the second ever AI Engineer Summit. Apply now!</li><li><a href="https://x.com/sannykimchi/status/1176517584319127553">Tweet from sanny (@sannykimchi)</a>: Transformers have led to a wave of recent advances in #NLProc such as BERT, XLNet and GPT-2, so here is a list of resources💻 I think are helpful to learn how Transformers work, from self-attention to...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1324844996934766783)** (31 messages🔥): 

> `Discord Bot Building, Obsidian Tool Discussions, Screen Sharing Issues, Webhook Limitations` 


- **Discord Bot Idea Takes Off**: Members discussed the possibility of building a **Discord bot**, expressing a shared interest in tackling the project collaboratively or starting from scratch. One member mentioned their own **yolo Discord** and ability to generate an endpoint for it.
- **Obsidian Tool Exchange**: A member expressed their beginner status with **Obsidian** and showed interest in chatting with anyone willing to lead the conversation. This suggests a desire for guidance and collaboration in mastering this tool.
- **Echo Issues During Screen Sharing**: Participants experienced **echo** issues during a screen share, with suggestions made to mute the screen share to alleviate the problem. It was noted that the screen share was sharing **computer audio**, which added to the echo effect.
- **Limitations of Webhooks in Discord**: A member pointed out the limitation of **webhooks**, highlighting that while they can post, they cannot read data. This brought attention to the challenges of using webhooks for more interactive bot functionalities.
- **Frustration in Building New Apps**: There was a sense of frustration expressed by a member at the complexity of starting new projects, with one stating their brain felt tired during the app-building process. This sentiment resonated with others, indicating a common challenge among participants.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1324775554041380916)** (2 messages): 

> `GEMM performance on GPU, Identifying inefficiencies in GEMM, Optimizing GEMM computations, Suggestions for improvements in GEMM` 


- **GEMM performance insights shared**: A member highlighted concerns about the gap between actual and theoretical **GEMM performance** on GPU, affecting optimal performance assessments.
   - They also referenced their article on detecting computation inefficiencies in GEMM computations, available [here](https://yywangcs.notion.site/Inconsistency-in-GEMM-Performance-16efc9f5d80580838090dded05493014) and summarized on [Twitter](https://x.com/YyWangCS17122/status/1874856334845489191).
- **Optimizing GEMM through community feedback**: A suggestion was made that declaring achieved **optimal performance** in a forum may elicit community members to provide more optimized implementations.
   - This approach could help in uncovering solutions or improvements that might not have been previously considered.



**Link mentioned**: <a href="https://x.com/YyWangCS17122/status/1874856334845489191).">Tweet from YyWangCS (@YyWangCS17122)</a>: Matrix multiplication (GEMM) performance is crucial for deep learning. I’ve written an article on how to automatically detect computation inefficiencies in GEMM computations on GPU. https://yywangcs.n...

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1324559433686192158)** (20 messages🔥): 

> `Triton kernel performance, Matrix multiplication issues, Testing kernel equivalence, Pointer loading in Triton, Floating point operation ordering` 


- **Triton kernel performance debugging**: A member identified that setting `os.environ['TRITON_INTERPRET'] = '1'` resulted in slower performance, but removing it improved kernel execution times.
   - Another user confirmed the benefit of removing triton_interpret, leading to resolved issues when implementing matrix multiplication.
- **Matrix multiplication kernel call issues**: A user faced challenges calling a kernel function inside another, resolving the issue after adjusting installations and settings.
   - It's recommended to increase batch size to at least **16** and to use padding for smaller matrices.
- **Testing kernel equivalence with larger inputs**: A question arose regarding kernel equivalence in Triton when using larger input values, as results differed significantly from smaller, random values created using `torch.randn`.
   - Another user suggested adjusting the `atol` argument in `torch.allclose` to handle floating-point discrepancies.
- **Pointer loading discrepancies in Triton**: A member noticed unexpected end indices when loading data with offsets, which led to confusion due to differing results in visualization.
   - They shared the outputs and images for deeper analysis, highlighting the complexity of pointer operation results.
- **Impact of floating point operation ordering**: It's emphasized that the ordering of operations can significantly affect results in floating-point calculations, as addition and multiplication are not commutative.
   - This indicates a potential source of discrepancies when comparing outputs from Triton and Torch implementations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1uTsmN2-U3TRHi0mhJ2Dp0nZh-yM_gt4_?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1uTsmN2-U3TRHi0mhJ2Dp0nZh-yM_gt4_?usp=shar">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1324687255633395733)** (3 messages): 

> `Dynamic br/bc values in Flash Attention, Fixed br/bc values performance, Compiling versions for selection` 


- **Exploring Dynamic br/bc Values in Flash Attention**: A member inquired about methods to dynamically determine values for **br/bc** in Flash Attention rather than using fixed values.
   - *This approach could potentially improve flexibility but may require more complex implementations.*
- **Fixed br/bc Values Offer Speed**: Another member pointed out that fixed values for **br/bc** are significantly faster, stating it’s like **10 trillion times faster**.
   - *This emphasizes the trade-off between speed and dynamic adaptability in performance tuning.*
- **Compiler Solutions for Dynamic Selection**: It was suggested that if dynamic selection is necessary, compiling different versions and choosing from them could be a solution, similar to **Flash Attention**'s method.
   - *This could provide a balance between performance and flexibility, allowing for tailored approaches based on specific needs.*


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1324493313260388362)** (11 messages🔥): 

> `Model Level Compile, Memory Profiling Issues, Inductor Cache Performance, Flex Attention Complications, Gradients and Activation Management` 


- **Model Level Compile with Selective Disabling**: A user considers using model level compile while marking portions to skip using `torch.compiler.disable` to enhance performance, as detailed in the [documentation](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html). Another user indicated this is a reasonable current approach for managing model compilation.
   - They discussed utilizing compile decorators for general operations, while disabling them for more costly ones, and noted the use of whole-model compile for popular shapes.
- **Inductor Cache Performance is Still Slow**: One user shared that using a remote inductor cache via Redis helps reduce warmup time but still incurs a long delay, quoted at **5 minutes** even with cache hits. This delay is attributed to the process of loading a compiled kernel, indicating performance concerns.
- **Investigating Memory Allocation with Linear Layers**: A user observed that a linear layer in their model was calling `cudaFree`, triggering questions about its memory allocation behavior despite expected reuse by Torch's caching allocator. They provided profiling traces suggesting output allocations were not being reused as anticipated, with observations that the output is consistently **128 MiB**.
   - Another participant inquired if this was with gradients enabled, noting that activations may need to be saved for the backward pass, with the original user confirming that `tensor.requires_grad` was set to **False** without any backward storage.



**Link mentioned**: <a href="https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html">TorchDynamo APIs for fine-grained tracing &mdash; PyTorch 2.5 documentation</a>: no description found

  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1324488328804040716)** (1 messages): 

> `P-1 AI, Research Engineer role, Artificial General Engineering` 


- **P-1 AI seeks talent for stealth startup**: P-1 AI is actively hiring for various roles with a focus on developing an **artificial general engineering (super)intelligence** that enhances physical system design efficiency. More details can be found in their [job listing](https://jobs.lever.co/P-1AI/84ae5c01-9160-44a5-a7c8-a107e645f0a6).
   - The team consists of **ex-Google DeepMind, Microsoft, Airbus, and DARPA** researchers, underlined by strong Silicon Valley investor backing.
- **Research Engineer role requires advanced skills**: As a **Research Engineer**, candidates will build and deploy AI systems using **multimodal** LLMs and GNNs with quantitative reasoning capabilities. The position aims to achieve **unprecedented performance** in designing physical systems.
   - This role emphasizes tackling **previously impossible tasks**, showcasing the ambitious goals of the P-1 AI team.



**Link mentioned**: <a href="https://jobs.lever.co/P-1AI/84ae5c01-9160-44a5-a7c8-a107e645f0a6">P-1 AI - Staff Research Engineer</a>: no description found

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1324726229256900648)** (5 messages): 

> `GPU Upgrade Considerations, Federated/Gossip Learning Resources, CUDA Learning, VRAM and Global Memory Importance, Upcoming Event on Turing Tensor Cores` 


- **Considering GPU Upgrades**: A member is contemplating upgrading from an **AMD RX 480** to either a **used RTX 2080 Ti** or a **3090**, prioritizing CUDA learning and local models.
   - It's noted that while the **RTX 3090** offers more benefits, it requires potential upgrades to the PSU and motherboard, complicating the decision.
- **Seeking Resources for Federated Learning**: A member expressed difficulty in finding resources to learn about **Federated** and **Gossip Learning**, stating they found a **DeepMind course on Federated Learning** but no reliable source for Gossip Learning.
   - They are appealing for recommendations to help establish a foundational understanding in both topics.
- **CUDA Learning Importance for Upgrading GPUs**: Another member suggested that for **CUDA** learning, a **2080 GPU's** Turing architecture is adequate, especially considering its compute capability of **7.5**.
   - They advised that for other use cases, larger **VRAM** and global memory could make the **3090 better suited** despite its higher cost.
- **Upcoming Event Highlighting Turing Tensor Cores**: A member mentioned an upcoming event on **February 15** where one of the server's best will explain the use of **Turing Tensor Cores**.
   - This could provide significant insights for those interested in understanding advancements in GPU technology.
- **Recent Hires and Community Contributions**: There was a note about someone being hired at **Together AI**, indicating growth within the community.
   - Additionally, a video of **Karpathy** discussing **llm.c** was shared, potentially offering valuable insights for the group.



**Link mentioned**: <a href="https://youtu.be/aR6CzM0x-g0?feature=shared&t=630)"> - YouTube</a>: no description found

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1324778323070419014)** (3 messages): 

> `Learning Cuda, Mental Health Awareness, Felix Hill Tribute` 


- **Learning Cuda is a Must!**: A member expressed excitement about the need to *learn Cuda*, sharing a [YouTube video](https://www.youtube.com/shorts/gFva8uNJPNg) that emphasizes its significance.
   - The video could be an engaging introduction to Cuda for those interested in GPU programming.
- **Felix Hill Remembered**: A member shared the sad news of **Felix Hill's passing**, expressing deep sorrow for the loss.
   - This highlights the impact of community figures and a shared sense of grief among members.
- **Prioritizing Mental Health**: A member stressed that *nothing should be more important than your own mental health*, urging others to prioritize it.
   - This comment serves as a crucial reminder of the importance of self-care in the fast-paced tech world.



**Link mentioned**: <a href="https://www.youtube.com/shorts/gFva8uNJPNg"> - YouTube</a>: no description found

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1324810302067314718)** (2 messages): 

> `TorchTitan, MFU vs HFU, Lecture 39 on YouTube` 


- **TorchTitan Blog Post Received Praise**: A member reviewed a blog post discussing **TorchTitan** and found it to be excellent.
   - They emphasized the importance of the content in understanding **Model FLOP/S Utilization**.
- **MFU vs HFU Explained in Lecture 39**: The discussion highlighted how **MFU** (Model FLOP/S Utilization) is sometimes calculated in comparison to **HFU** (Hardware FLOP/S Utilization).
   - The relevant part of [Lecture 39](https://youtu.be/VYWRjcUqW6w?feature=shared&t=2606) can be found between **43:26-47:29**, providing useful insights.



**Link mentioned**: <a href="https://youtu.be/VYWRjcUqW6w?feature=shared&t=2606)"> - YouTube</a>: no description found

  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1324770409815736350)** (1 messages): 

> `Transduction Goals, Prompt Optimization, Training Procedure` 


- **Aiming for Transduction Clarity**: The goal for transduction is to transform prompts from `[examples] | test input` into a `natural language description of the transformation | test output`.
   - This approach seeks to clarify how inputs relate to outputs while simplifying the process.
- **Inquiry on Prompt Optimization**: A question was raised about existing work in **prompt optimization**, specifically for transduction tasks.
   - Members are encouraged to share any known advancements in this area.
- **Constructing a Dataset for ARC Problems**: The proposal involves creating a dataset formatted as `example 1 | example 2 | example 3 inp | trainable prompt | example 3 output` for every ARC problem.
   - The aim is to apply backpropagation into the `trainable prompt` using a loss computed from the output of the last example.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1324539530874851410)** (14 messages🔥): 

> `Fine-tuning LLMs with LoRA, HuggingFace TRL vs. alternatives, MoE routing techniques, Documentation availability` 


- **Varying Experiences with LLM Fine-tuning Tools**: Users shared mixed experiences with LLM fine-tuning libraries, with **Unsloth** deemed difficult and **TRL** praised for its extensive documentation and ease of use.
   - A member noted that **Axolotl** is popular, while **Llama Factory** may lack sufficient documentation in English.
- **GitHub Stars Not Everything**: A discussion arose around the significance of **GitHub stars** for LLM tools, with a comment that stars might not accurately reflect usability or support.
   - Despite **Unsloth** leading with **20k stars**, users suggested leveraging **Transformers/PEFT from HuggingFace** for streamlined LoRA fine-tuning.
- **Resource Recommendations for Fine-tuning**: A user shared a link to a comparison of LLM fine-tuning libraries titled [LLM Fine-Tuning Library Comparison](https://docs.google.com/document/d/1k0E2XCuqJDGiD6IsP2rb6s3D1Iu96gro9WRCSajl0zs/edit) that provides insights into various options.
   - The conversation highlighted the lack of comprehensive documentation for certain tools, underscoring the importance of finding reliable resources.
- **Gating Networks for Mixture of Experts (MoE)**: A user sought recommendations on papers discussing **gating networks** for MoEs, aiming to understand the routing methods used in **Deepseek v3**.
   - Another member suggested the **OLMoE** paper, known for its research on MoEs, albeit with a lower number of experts.



**Link mentioned**: <a href="https://docs.google.com/document/d/1k0E2XCuqJDGiD6IsP2rb6s3D1Iu96gro9WRCSajl0zs/edit">LLM Fine-Tuning Library Comparison</a>: Fine-Tuning LLMs with LoRA: A Comparative Analysis of Five Popular Libraries The rise of large language models (LLMs) has ushered in a new era in natural language processing, enabling remarkable advan...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1324519239574355999)** (18 messages🔥): 

> `Post-training tutorial content, Quality of recorded talks, AI Safety Institute's activities, LinkedIn dynamics, Chatbotarena plot maintenance` 


- **Debate over sharing tutorial content**: A conversation unfolded about whether to share a post-training tutorial via YouTube and Interconnects, with concerns about spamming a large audience sized at **20k** members.
   - Another member noted that the **pre-recorded talks** have been highly regarded, suggesting they might be beneficial to share with a wider audience.
- **Quality perception of pre-recorded talks**: Members expressed appreciation for high-quality content, stating that recorded talks have become a highlight of the educational offerings.
   - One member mentioned that content should be shared as long as it contributes positively to growth.
- **UK AI Safety Institute referred to as intelligence operatives**: A humorous comment was made questioning why members of the **UK AI Safety Institute** aren't seen as intelligence operatives due to their social activities with AI researchers.
   - The post brought light to the informal networking that carries implications for information sharing with the UK government.
- **LinkedIn dynamics in AI circles**: A member cheekily remarked about the competitive nature of **LinkedIn races**, hinting at the networking strategies prevalent among AI professionals.
   - The discussion came with an attached screenshot, adding visual emphasis to the conversation on professional conduct in the AI community.
- **Chatbotarena plot maintenance concerns**: A member expressed surprise that the **Chatbotarena** plot is not maintained, suggesting it holds significant interest.
   - This comment was accompanied by an image, further illustrating the context of the discussion.



**Link mentioned**: <a href="https://x.com/typewriters/status/1874924700398436450">Tweet from Lauren Wagner (@typewriters)</a>: OH: I don&#39;t understand why people at the UK AI Safety Institute aren&#39;t considered intelligence operatives. They moved from London to SF, throw parties, get drunk researchers to talk, and send ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1324684692041109536)** (4 messages): 

> `The Bittersweet Lesson, Felix's Contributions, Google Account Concerns` 


- **Reflecting on The Bittersweet Lesson**: A member shared a link to a [Google document](https://docs.google.com/document/d/1MPqtT_1vQ-73j796tf7sXIZKCRcIfUD0cVU_UbPXnUU/edit?usp=sharing) discussing their favored insights on **The Bittersweet Lesson**.
   - Regrettably, the author of the piece has recently passed away, sparking reflections on their impactful work.
- **Preserving Felix's Legacy**: Members expressed sadness over the recent passing of **Felix**, commending his wonderful contributions to the community.
   - One member mentioned having a **PDF backup** of the works, highlighting the importance of keeping such contributions accessible.
- **Concerns about Google Account Longevity**: It was noted that the ability to access documents might be problematic as Google accounts can eventually get deactivated.
   - Members acknowledged the risk of lost access to important documents if accounts are not preserved adequately.



**Link mentioned**: <a href="https://docs.google.com/document/d/1MPqtT_1vQ-73j796tf7sXIZKCRcIfUD0cVU_UbPXnUU/edit?usp=sharing">The Bittersweet Lesson</a>: The Bittersweet Lesson 😆  The strange case of inductive bias in Transformers Felix Hill 21 Oct 2024   Do you remember a few years back when the notion of inductive bias was central to machine learnin...

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1324599532914343987)** (6 messages): 

> `SnailBot news, Entertainment value of SnailBot` 


- **SnailBot's Performance Humorously Critiqued**: A member humorously commented that the SnailBot is *one slow snail*, highlighting its performance.
   - Comments included disbelief at its pace, with one member exclaiming, *good lord*.
- **Entertainment Factor of SnailBot**: Members expressed amusement over SnailBot's antics, noting that *it's so entertaining*.
   - Another member remarked on its fitting name, suggesting it lives up to the moniker.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1324470121040838798)** (19 messages🔥): 

> `FLUX.1 [dev], Image Filters on Minecraft, Community Dynamics, Using Discord Group, AGI Discussions` 


- **FLUX.1 [dev] Capabilities Uncovered**: Members discussed the features of **FLUX.1 [dev]**, a **12 billion parameter** rectified flow transformer that generates images from text. For more details, visit the [blog post](https://blackforestlabs.ai/announcing-black-forest-labs/).
   - Key features include output quality that is second only to **FLUX.1 [pro]** and open weights for scientific research.
- **Minecraft Image Filter Glitches**: Discussion emerged about **Minecraft** image prompts not passing due to keywords like 'Goofy' or 'Minecraft'. A creative prompt worked, featuring a whimsical server icon with elements like rubber chickens and bananas.
   - Members suggested finding alternative spellings for potentially problematic words to get past filters.
- **Commentary on Community Vibes**: One member mentioned that the server feels overrun by **AGI worshippers**, with those needing OpenAI support becoming a minority. They noted a blend of **lonely individuals** and savior complexes dominating the discourse.
   - This pointed to a shifting dynamic within the community, with some members finding it concerning.
- **Instructions for Group Utilization**: A user inquired about using the group effectively, prompting another member to suggest reading the group instructions. An image was shared, possibly elaborating on usage guidelines.
   - This reflects ongoing efforts to help members acclimate to group interactions.
- **GPT-Generated Descriptions Humor**: 'Delve' was humorously suggested as a means to engage with the group, prompting laughter. Members noted the irony in group descriptions allegedly being written by **GPT-3**.
   - This light-hearted commentary emphasizes the blending of AI and human interactions.



**Link mentioned**: <a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">black-forest-labs/FLUX.1-dev · Hugging Face</a>: no description found

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1324534437953929216)** (5 messages): 

> `ChatGPT for search results, YouTube GPTs functionality, Cross posting etiquette` 


- **Reliability of ChatGPT for search**: A user questioned whether they can rely on **ChatGPT** for search and web-related results.
   - This raises a discussion about **comparisons** with other search tools like **Perplexity**.
- **YouTube GPTs struggle with retrieval**: Concerns were raised about **YouTube GPTs** inability to analyze or retrieve useful information.
   - This sentiment suggests users are frustrated with the functionality of these models.
- **Cross posting concerns**: A member cautioned against cross posting in multiple channels, noting it may be seen as **spamming**.
   - This is relevant for maintaining decorum within the chat environment.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

castilla99_87524: how do i keep consistent characters when making dif scenes in sora?
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

castilla99_87524: how do i keep consistent characters when making dif scenes in sora?
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1324545258616389653)** (7 messages): 

> `New Year Wishes, Rerank on Azure` 


- **Cheers to a New Year!**: Members are excited for the new year, expressing hopes for **happiness**, **success**, and **great memories** ahead.
   - *Cohere* is ready to embrace the possibilities, with one member declaring, *“this is THE year!”*
- **Excitement for Rerank-3.5**: Inquiring about **Rerank** news on **Azure**, a member mentioned the upcoming **rerank-3.5** which should launch very soon.
   - Another member invited discussion about use-cases, asking, *“How are you liking it so far?”*


  

---


### **Cohere ▷ #[rules](https://discord.com/channels/954421988141711382/954422415016996864/1324618481936891958)** (1 messages): 

> `Server Guidelines, Promotion Rules, Spam Policy, Commercial Activities Restrictions` 


- **Maintain PG Standards in Communication**: All members must ensure that messages are respectful and free from harmful or inappropriate content, as this is a PG-rated server.
   - Non-compliance may lead to warnings or removal from the server.
- **English Usage Encouraged**: Members are asked to primarily communicate in English to facilitate better understanding among users.
   - This guideline helps maintain clarity and enhances engagement within the community.
- **Limited Promotion Allowed**: Advertisements can only be made in designated channels, such as sharing projects in <#1218409701339828245> focused on Cohere-related content.
   - This helps reduce clutter and keeps the channels relevant to the community interests.
- **Strict Spam Regulations**: Posting duplicate messages across multiple channels is prohibited, and excessive spamming will result in message deletion.
   - Members are also discouraged from pinging others unnecessarily to maintain a pleasant communication environment.
- **No Job Recruitment Allowed**: __The server strictly prohibits job postings, recruitment efforts, and any advertising related to employment opportunities.__
   - Approaching team members via DM regarding job offers is also forbidden to maintain focus on community discussions.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1324508757509672981)** (2 messages): 

> `Command-R functionality, Issues with Command-R, Resuming processes` 


- **Confusion over Command-R Methodology**: A member explained that the method of how **Command-R** works involves writing things out exhaustively before rewriting them in a resumed format.
   - They pointed out that while the initial part of the command is understood, the **'-R'** seems to cause confusion and disrupt the process.
- **Issues Encountered with Command-R**: Members discussed the problems faced with using **Command-R**, highlighting difficulties in the execution of the **'-R'** function.
   - The general sentiment suggested that improvements are needed to enhance the clarity and effectiveness of the command.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1324493791180230756)** (6 messages): 

> `Increased Embedding Rate Limit, Cohere API Rate Limits` 


- **Requesting Increase for Embedding Rate Limit**: A user inquired about how to request an increase to their **embeddings rate limit**.
   - To address such inquiries, the **Cohere support team** can be contacted at support@cohere.com.
- **Cohere API Rate Limits Overview**: Cohere API has defined rate limits for various endpoints, with the **Embed** endpoint allowing **100/min** during trials and **2,000/min** in production.
   - Additionally, all endpoints are capped at **1,000 calls per month** with a trial key, ensuring developers can efficiently manage their usage.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1324731503170551869)** (10 messages🔥): 

> `Torchtune Benchmarking, Chunked Cross Entropy Implementation, Memory Gains during Compilation` 


- **Torchtune benchmarks gain traction**: More **papers and models** are adopting Torchtune for benchmarking, emphasizing its utility as an evaluation tool in AI model performance.
   - A member noted that it might be beneficial to showcase the **alternative evaluation methods** available through Torchtune.
- **Chunked Cross Entropy PR Discussion**: [Chunked cross entropy](https://github.com/pytorch/torchtune/pull/1390) by *felipemello1* was discussed, highlighting its purpose to reduce memory usage by optimizing the computation process.
   - Members shared results showing **reduced memory gains** when compiling with chunking applied after output projection.
- **Implementation Variants and Comparisons**: A member shared a variant using *log_softmax* instead of `F.cross_entropy`, yielding good performance with manageable memory use.
   - This approach allows full compilation without breaks and prompts further exploration into **benchmarking within Torchtune**.
- **Wandb Profiling and Memory Optimization**: Members discussed the possibility of profiling implementations with **Wandb** to assess performance improvements over standard methods.
   - Insights were shared regarding **Torch's** efforts to minimize memory usage during cross-entropy calculations.
- **Opportunities for Performance Improvement**: A member indicated potential performance optimization in Torchtune's transformer implementation, referencing the code link as an area to explore.
   - The community is considering how to integrate **chunked_nll** into the compilation process to enhance efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/main/torchtune/modules/transformer.py#L482.">torchtune/torchtune/modules/transformer.py at main · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/1390">chunked cross entropy by felipemello1 · Pull Request #1390 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here)Mechanism: Chunked cross entropy reduces memory by proce...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1324473445672550505)** (1 messages): 

> `PyTorch Torchtune Bug, Flex Attention Compilation, Kernel Finding` 


- **Workaround for PyTorch Torchtune Bug on A6000**: A user shared their experience with a bug in the PyTorch Torchtune library while using the **A6000**, and successfully set `flex_attention_compiled` using `torch.compile()` to find a working kernel.
   - They suggested that using `mode=os.getenv("TORCHTUNE_FLEX_MODE", None)` might provide a temporary solution until the attention dispatch is reworked, and advised testing on the **A6000**.
- **Potential Environment Variable Solution**: To avoid auto defaulting to flex when `packed=True`, a member proposed allowing users to set an environment variable to dictate the mode in the **Torchtune** library.
   - This approach may help if users wish to avoid the need to install from source, but validation on hardware like the **A6000** is necessary.
- **Uncertainty about Bug Fix in Version 2.6.0**: There is some uncertainty regarding whether the aforementioned bug has been resolved in version **2.6.0** of PyTorch Torchtune, as discussions about the branch cutting have already taken place.
   - Members expressed concerns about the status of the fix, implying that further testing is required before concluding.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/issues/2218.">Issues · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1324784792096608318)** (1 messages): 

> `Invoice processing agent, LlamaParse, Agentic workflow, Spend categories, Cost centers` 


- **Build a Practical Invoice Processing Agent**: A recent notebook by @ravithejads demonstrates how to build an **invoice processing agent** from scratch that automatically extracts and enriches invoice line items with **spend categories** and **cost centers**.
   - The agent uses [LlamaParse](https://twitter.com/llama_index/status/1875225903346909256) for its workflow and provides firsthand insights into automating the invoice handling process.
- **Invoicing Automation with LlamaParse**: The notebook showcases the utility of **LlamaParse** in creating workflows that streamline **invoice processing**.
   - By leveraging automation, users can expect to significantly reduce manual errors and improve efficiency in handling financial documents.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1324800166875762758)** (9 messages🔥): 

> `Dataset storage options, Query fusion for retrievers, JSON advantages, Compression techniques for data, Using SQL or NoSQL for datasets` 


- **Exploring dataset storage options**: Members discussed their preferred storage solutions for datasets used in evaluating LLMs, with mentions of **S3**, **Git LFS**, and **Hugging Face datasets**.
   - It was noted that other options include any **SQL** or **NoSQL** database, depending on the dataset's content.
- **JSON as a simple storage format**: One member stated that if using **S3** or **LFS**, they typically store data as a **JSON blob**, which can be compressed if very large.
   - They also mentioned that storing data in JSON helps reduce complexity and allows for easy integration with **Pydantic models**.
- **Concerns about query fusion outcomes**: A member shared issues with achieving good results from their setup of **2 vector embedding retrievers** and **2 BM25 retrievers** using query fusion.
   - They asked for suggestions on how to enhance the **quality of responses** from this combination of retrievers.
- **Lightweight datasets warranting JSON + Git use**: In response to the dataset storage discussion, one member indicated they would stick to using **JSON and Git** for smaller datasets until a more robust solution is needed.
   - This emphasizes the practicality of starting simple with dataset storage and management.


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1324553118117138494)** (8 messages🔥): 

> `Open Interpreter functionality, Open-source contributions, Installation Steps for Open Interpreter, Web WhatsApp messaging, Trading clicker execution mode` 


- **Open Interpreter seems lacking**: Users expressed concerns that Open Interpreter 1.0 appears to perform worse than the classic version, specifically noting its inability to run code and broken text formatting.
   - They also pointed out the absence of tools for web browsing and searching, causing disappointment.
- **Open-Source Contributions Acknowledged**: The scalenow AI Community recognized the pivotal role of the open-source community, highlighting tools like [OpenInterpreter](https://github.com/KillianLucas/open-interpreter) for their contributions to functionality.
   - They commended projects like **OpenProject**, **Rasa**, and **Kotaemon**, showcasing their importance in enhancing service offerings.
- **A Streamlined Installation Process**: Installation steps for Open Interpreter were shared, including one-line commands for Mac, Windows, and Linux that automate the setup process seamlessly.
   - Users can access the web UI post-installation, allowing for easier interaction and command execution.
- **Fun with Web WhatsApp Messaging**: A user humorously noted their interaction with messaging through Web WhatsApp, expressing amusement at the experience.
   - This sparked a light-hearted exchange regarding the use of technology in everyday communication.
- **Seeking an Always-On Trading Clicker**: A user is on the lookout for a trading clicker that requires an 'ever active OS execution mode' to function effectively.
   - This requirement suggests the need for a solution that can continuously execute commands without interruption.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

rd4com: 🥳 Happy new year !!
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1324494784525500477)** (6 messages): 

> `Linked List Implementation, Building CLI and TUI Tools, AST and Index-Style Trees, Mojo Debugging Issues` 


- **Seeking Linked List Implementation**: A member requested a linked list implementation that works with nightly, expressing a desire to experiment without the effort of writing it themselves.
   - This highlights a common need for readily available implementations in the community for rapid prototyping.
- **Toasty's CLI and TUI Projects**: Toasty humorously shared that they are learning to build CLI and TUI tools by developing libraries dedicated to the same purpose.
   - Another member suggested that this could lead to creating a new shell for Mojo, similar to existing ones like bash and zsh.
- **AST and Index-Style Trees Development**: A member reported success with implementing index-style trees and graphs, mentioning specific files and structures used like `RootAstNode` and `DisplayAstNode`.
   - They encountered segmentation faults during debugging, particularly with the `DisplayAstNode`, indicating potential issues with recursive types.
- **Details on Debugging Issues in Mojo**: Discussion included a [GitHub issue](https://github.com/modularml/mojo/issues/3917) where a member experienced seg faults while using the Mojo debugger, as opposed to running scripts normally.
   - This issue underlines challenges faced in developing recursive types, reflecting on the intricacies of debugging complex data structures in Mojo.



**Link mentioned**: <a href="https://github.com/modularml/mojo/issues/3917">[BUG] --debug-level full crashes when importing · Issue #3917 · modularml/mojo</a>: Bug description Running a mojo script using the debugger seg faults, as opposed to when running regular mojo, which runs to completion (although I have noticed strange behavior in the regular scrip...

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1324467766148857936)** (5 messages): 

> `Certificate Issuance, Fall 2024 Course Enrollment, Spring 2025 Course Sign-up` 


- **Certificates expected by late January**: Certificates will be issued roughly by the **end of January**, according to members' updates.
   - One member mentioned to *stay tuned* for further announcements after all certificates have been sent out.
- **Fall 2024 course enrollment closed**: The opportunity to obtain a certificate for the **Fall 2024** course has ended, as confirmed by members.
   - Prospective students are now encouraged to join the **Spring 2025** course, with the sign-up form provided in the chat.



**Link mentioned**: <a href="https://llmagents-learning.org/sp25">Large Language Model Agents MOOC</a>: MOOC, Spring 2025

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1324497467181498460)** (4 messages): 

> `GraphRAG implementation, Donor's Game simulation, DSPy strategy updates` 


- **Using GraphRAG for entity extraction**: A user inquired if a specific GraphRAG implementation was used, to which it was revealed that **Gemini** modified the default prompts for extraction of entities related to a code base.
   - *This suggests a tailored approach to enhance the extraction process*.
- **Simulating Donor's Game with DSPy**: A member discussed simulating the **Donor's Game** from game theory where agents upgrade strategies based on previous winning tactics, running recursively for each generation.
   - *They proposed leveraging DSPy as a potentially effective tool for handling these strategy updates.*
- **Code shared for cultural evolution implementation**: The code for the Donor's Game simulation was shared, linking to a [GitHub repository](https://github.com/CakeCrusher/cultural_evolution/blob/main/donors_game/game/orchestrator.py#L120) that implements a methodology from the paper titled *Cultural Evolution of Cooperation among LLM Agents*.
   - This paper investigates whether a society of **LLM agents** can develop cooperative behaviors through evolutionary methods.



**Link mentioned**: <a href="https://github.com/CakeCrusher/cultural_evolution/blob/main/donors_game/game/orchestrator.py#L120">cultural_evolution/donors_game/game/orchestrator.py at main · CakeCrusher/cultural_evolution</a>: implements the methodology outlined in the paper *Cultural Evolution of Cooperation among LLM Agents*. The paper explores whether a society of large language model (LLM) agents can develop cooperat...

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1324818750834475078)** (2 messages): 

> `Tinygrad Windows Support, Pull Requests for Windows Bugs` 


- **Tinygrad's Stance on Windows Support**: A member inquired whether **tinygrad** would accept pull requests that fix bugs specifically for Windows.
   - This highlights the ongoing challenge of supporting Windows, despite it not being the primary focus of tinygrad development.
- **Acceptance of Simple Fixes for Windows**: Another member speculated that such pull requests would likely be accepted if the fixes are **simple and stable**.
   - This indicates a willingness within the community to work towards improving compatibility, albeit with cautious criteria.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1324475911436767394)** (2 messages): 

> `Shapetracker Documentation, Matrix Memory Layout, Memory Index Calculation, Stride in Matrix Access` 


- **Appreciation for Library Documentation**: A member expressed enthusiasm for the quality of library documentation, wishing that more libraries provided similar resources.
   - *These blogs are very nice* - highlighting a desire for improved documentation across libraries.
- **Seeking Clarity on Shapetracker Math**: A member inquired about the mathematical principles behind calculating indices in a `Shapetracker` object, referencing a specific [tinygrad-notes blog](https://mesozoic-egg.github.io/tinygrad-notes/20241217_st.html).
   - They clarified the concept of stride in accessing a matrix stored as a linear array and requested directed resources to understand the underlying principles better.



**Link mentioned**: <a href="https://mesozoic-egg.github.io/tinygrad-notes/20241217_st.html">Shapetracker</a>: Tutorials on tinygrad

  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1324626602868478015)** (4 messages): 

> `Weights and Biases vs MLflow, Recording Experimentation Results, State of Classical Machine Learning, 1-bit Large Language Models` 


- **Weights and Biases is Cloud-Based, MLflow is Self-Hosted**: Weights and Biases (WandB) functions as a hosted service, making it less suitable for some users, whereas **MLflow** offers a self-hosting option.
   - Both tools perform effectively for managing machine learning experiments.
- **Experimentation Results Stored in Databases**: In the worst case, experimentation results can be recorded into a **Postgres** or **Clickhouse** database.
   - This provides a fallback method for keeping track of experimental data.
- **Discussion on the State of Classical ML**: A member raised concerns over whether classical machine learning is being overshadowed by LLM applications.
   - Questions arose about the future of **recommender systems**, **time series**, and **clustering** problems in this evolving landscape.
- **BitNet: Pioneering 1-bit LLMs**: Recent developments, such as **BitNet**, introduce **1-bit Large Language Models** (LLMs) that efficiently match full-precision models in performance metrics while being cost-effective.
   - These models use ternary weights and create new opportunities for hardware optimization tailored for low-bit representations.



**Link mentioned**: <a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1324789662065365135)** (4 messages): 

> `AI Reader Tool, Embedding Weights in Content, Indexing by Subject` 


- **AI Reader Tool for Students**: A user is developing a low-cost laptop with an 'AI Reader' tool that opens a chat GUI for tutoring whenever a PDF is accessed, aiming to aid students in Africa.
   - They are exploring using Nomic embed for handling content embeddings and providing feedback on mock exams, needing to ensure effective local embeddings during user queries.
- **Ranking Content Authority by Relevance**: A member suggested the need for a system to rank educational content's authority that evolves over time, especially in rapidly changing fields like computer science.
   - They highlighted the challenge of maintaining this ranking dynamically without the necessity of re-indexing every time.
- **Prioritizing Student Transcripts in Learning Materials**: The discussion continued with thoughts on how student transcripts should have higher weight in content ranking to reflect personal academic performance.
   - This raises the idea of having a more personalized indexing approach in educational tools.
- **Indexing by Subject for Broader Context**: A user proposed considering an indexing approach based on 'subject' or 'topic' rather than just 'book', allowing integration of supplementary articles and notes.
   - This method could provide enhanced control for selecting relevant resources and filling information gaps effectively.


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1324539058566729768)** (2 messages): 

> `AI and 3D modeling with Blender, Animals and EEG, Language/action mapping for animals` 


- **Seeking AI and Blender Collaboration**: A member inquired about any groups working on **AI and 3D modeling with Blender** specifically for **annotations** to support the Blender community.
   - The request highlights an interest in collaborative projects that enhance **Blender's capabilities** through AI.
- **Exploring Animal Language Mapping**: Another member expressed a keen interest in communities working on **Animals and EEG** combined with AI for **language/action mapping** and understanding of animals.
   - This suggests a growing curiosity about the intersection of neuroscience and AI in **animal behavior studies**.


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

yoavhacohen: https://x.com/yoavhacohen/status/1875148348489113891
  

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

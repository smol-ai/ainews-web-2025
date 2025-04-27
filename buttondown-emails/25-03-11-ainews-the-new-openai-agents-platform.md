---
id: 599f1341-a28e-45b3-9feb-e507c722162c
title: The new OpenAI Agents Platform
date: '2025-03-12T00:23:17.547385Z'
original_slug: ainews-the-new-openai-agents-platform
description: >-
  **OpenAI** introduced a comprehensive suite of tools for the Year of Agents,
  including a **Responses API**, **Web Search Tool**, **Computer Use Tool**,
  **File Search Tool**, and an open-source **Agents SDK** with integrated
  **Observability Tools**. **Reka AI** open-sourced **Reka Flash 3**, a **21B
  parameter reasoning model** licensed under Apache 2.0, outperforming
  **o1-mini** and powering their **Nexus enterprise intelligence platform**. The
  **OlympicCoder** open reasoning models surpassed **Claude 3.7 Sonnet** and
  much larger models, releasing the **CodeForces-CoTs dataset** and **IOI'2024
  benchmark**. **DeepSeek** built a **32K GPU cluster** capable of training
  V3-level models in under a week and is exploring **AI distillation**.
  **Hugging Face** added **Cerebras** support, enabling inference of models like
  **Llama 3.3 70B** at over **2,000 tokens/s**, 70x faster than GPUs. **Reka's
  Sonic-2** voice AI model is available via **Together API** with **40ms
  latency** and high-fidelity synthesis. **Alibaba's Qwen Chat** enhanced its
  multimodal interface with text, images, videos, voice-to-text, guest mode, and
  expanded file uploads.
companies:
  - openai
  - reka-ai
  - huggingface
  - deepseek
  - togethercompute
  - alibaba
models:
  - reka-flash-3
  - o1-mini
  - claude-3-7-sonnet
  - llama-3-70b
topics:
  - reasoning
  - fine-tuning
  - reinforcement-learning
  - model-quantization
  - inference-speed
  - multimodality
  - voice
  - api
  - agents-sdk
  - observability-tools
  - gpu-clusters
  - synthetic-data
  - dataset-release
  - voice-synthesis
  - model-performance
people:
  - sama
  - reach_vb
  - _lewtun
  - _akhaliq
---


<!-- buttondown-editor-mode: plaintext -->**OpenAI may be all you need.**

> AI News for 3/11/2025-3/12/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**224** channels, and **2851** messages) for you. Estimated reading time saved (at 200wpm): **258 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

In a [livestream today](https://www.youtube.com/watch?v=hciNKcLwSes), OpenAI dropped a sweeping set of changes to prepare for the Year of Agents:

- [Responses API](https://platform.openai.com/docs/quickstart?api-mode=responses)
- [Web Search Tool](https://platform.openai.com/docs/guides/tools-web-search)
- [Computer Use Tool](https://platform.openai.com/docs/guides/tools-computer-use)
- [File Search Tool](https://platform.openai.com/docs/guides/tools-file-search)
- A new open source [Agents SDK](https://platform.openai.com/docs/guides/agents) with integrated [Observability Tools](https://platform.openai.com/docs/guides/agents#orchestration)

[Atty Eletti](https://x.com/athyuttamre/status/1899541471532867821) told the full story of the design decisions, and [sama](https://x.com/sama/status/1899579431905305027) called it "one of the most well-designed and useful APIs ever".

You can find more code samples and highlights on the [exclusive Latent Space interview](https://www.latent.space/p/openai-agents-platform) for today's launch:

https://www.youtube.com/watch?v=QU9QLi1-VvU


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**1. AI Models and Performance: Model releases, benchmarks, performance comparisons of specific models**

- **Reka Flash 3, a new 21B parameter reasoning model from Reka AI, has been open-sourced** [@RekaAILabs](https://twitter.com/RekaAILabs/status/1899481289495031825), achieving competitive performance.  [@reach_vb](https://twitter.com/reach_vb/status/1899517300576747615) highlighted that **Reka Flash 3 is Apache 2.0 licensed and beats o1-mini**, questioning why it's not trending. Reka AI further detailed that [Reka Flash 3 powers Nexus, their new enterprise intelligence platform](https://twitter.com/RekaAILabs/status/1899481289495031825), and was [fine-tuned on synthetic and public datasets, followed by RLOO with model-based and rule-based rewards](https://twitter.com/RekaAILabs/status/1899481291889979896). Weights are available on [Hugging Face](https://twitter.com/RekaAILabs/status/1899481291889979896).
- **OlympicCoder, a series of open reasoning models, outperforms Claude 3.7 Sonnet and models over 100x larger** according to [@_lewtun](https://twitter.com/_lewtun/status/1899574591171035390).  The release includes **CodeForces-CoTs dataset** and **IOI'2024 benchmark** for competitive coding problems.
- **DeepSeek has built a 32K GPU cluster capable of training V3-level models in under a week**, according to [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899357311811928152). [@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1899475671958929453) noted that **DeepSeek is now discussing AI distillation**, a concept he published in 1991, linking it to his earlier work. [@cis_female](https://twitter.com/cis_female/status/1899307802423632085) reported getting **30 tokens/s running R1 on 3x abacus + two sticks with int0 quantization**.
- **Hugging Face Inference now supports Cerebras**, as announced by [@_akhaliq](https://twitter.com/_akhaliq/status/1899502216420942232). Cerebras Inference is reported to run models like Llama 3.3 70B at over **2,000 tokens/s, 70x faster than leading GPUs**.
- **R1 is reportedly achieving 18t/s on a new M3 Ultra for around $9K**, according to [@reach_vb](https://twitter.com/reach_vb/status/1899480424834899993), suggesting increasing accessibility of high performance inference.
- **Reka's Sonic-2 voice AI model is now available through Together API**, delivering **40ms latency and high-fidelity voice synthesis**, announced by [@togethercompute](https://twitter.com/togethercompute/status/1899498102836380106).
- **Qwen Chat has been enhanced** with a unified multimodal interface, supporting text, images, and videos, and enhanced video understanding up to 500MB, a redesigned mobile experience with voice-to-text, guest mode, and expanded file upload capacity, according to [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1899497336889659775).

**2. AI Agents and Developer Tools: Focus on tools for building and using AI agents, SDKs, APIs, and agentic workflows.**

- **OpenAI launched new tools for building AI agents**, including a **Responses API**, **Web search tool**, **File search tool**, **Computer use tool**, and **Agents SDK**, as announced by [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531225468969240), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531367056064814), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531516448768103), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531586950795662), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531682224431614), [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1899531857143972051), and summarized by [@omarsar0](https://twitter.com/omarsar0/status/1899530784832459043) and [@scaling01](https://twitter.com/scaling01/status/1899510452473790537). The **Responses API** unifies Chat Completions and tool usage, enabling multi-turn agents in a single request. Built-in tools include **Web Search** (powered by GPT-4o, achieving 90% on SimpleQA), **File Search** (with metadata filtering), and **Computer Use** (automating browser and OS tasks, achieving SOTA benchmarks). The **Agents SDK** (open-source, improving upon Swarm) facilitates orchestration of single and multi-agent workflows with guardrails and observability. [@sama](https://twitter.com/sama/status/1899579431905305027) called the new API one of the "most well-designed and useful APIs ever". [@swyx](https://twitter.com/swyx/status/1899517984365752361) mentioned a Latent Space Podcast episode with OpenAI discussing these features.
- **LangChain announced Agent Chat UI**, an OSS web app for interacting with LangGraph apps via chat, and **LangGraph-Reflection**, a prebuilt graph for agents to self-critique and improve output, reported by [@LangChainAI](https://twitter.com/LangChainAI/status/1899497457459122535) and [@LangChainAI](https://twitter.com/LangChainAI/status/1899493848843477178).  They also highlighted how **C.H. Robinson saves 600+ hours/day automating orders with LangGraph and LangSmith**, according to [@LangChainAI](https://twitter.com/LangChainAI/status/1899475651863978410).
- **Weaviate launched a Transformation Agent** that allows users to not only query but also create and update data in the database, as announced by [@bobvanluijt](https://twitter.com/bobvanluijt/status/1899521133940011476).
- **Contextual AI released Contextual Reranker**, an instruction-following SOTA reranker designed to improve precision in RAG pipelines and allow for granular control over ranking priorities, detailed by [@apsdehal](https://twitter.com/apsdehal/status/1899497153132958049). [@douwekiela](https://twitter.com/douwekiela/status/1899490844572577958) introduced a similar instruction-following reranker, emphasizing its ability to prioritize based on user instructions.
- **Perplexity AI launched a Windows App**, providing access to voice dictation, keyboard shortcuts, and the latest models, as announced by [@perplexity_ai](https://twitter.com/perplexity_ai/status/1899498357154107499).

**3. AI Applications and Industry Impact: Real-world applications, industry use cases, and company news.**

- **Figure AI is preparing to ship thousands of humanoid robots**, powered by Helix neural networks, as shown by [@adcock_brett](https://twitter.com/adcock_brett/status/1899544574626070660).  He argues that  [Figure will be the ultimate deployment vector for AGI](https://twitter.com/adcock_brett/status/1899587483928805642) and that [in the future, every moving object will be an AI agent](https://twitter.com/adcock_brett/status/1899500640755450016). They are [hiring interns and full-time roles](https://twitter.com/adcock_brett/status/1899484728157417610) and [@DrJimFan](https://twitter.com/DrJimFan/status/1899509660971131301) expressed excitement about their humanoid home.
- **Manus, a Chinese high-performing AI agent**, was mentioned in AI/ML news by [@TheTuringPost](https://twitter.com/TheTuringPost/status/1899393214106521633). Anthropic's models are reportedly powering Manus, which is described as the latest AI sensation, according to a report by [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1899498723010662449).
- **Zoom is leveraging AssemblyAI's Speech-to-Text models** to advance their AI research and development for AI Companion, according to [@AssemblyAI](https://twitter.com/AssemblyAI/status/1899496977169043743).
- **Cartesia announced Series A funding**, as reported by [@_albertgu](https://twitter.com/_albertgu/status/1899499128389877764).  [@saranormous](https://twitter.com/saranormous/status/1899484191256981779) praised their talent density and creativity, noting increased GPU resources.
- **Perplexity AI is expanding beyond the web**, mentioned in AI/ML news by [@TheTuringPost](https://twitter.com/TheTuringPost/status/1899393214106521633).
- **Embra is introduced as a full AI OS**, managing email, meetings, relationships, writing emails, scheduling, and automating research, according to [@zachtratar](https://twitter.com/zachtratar/status/1899521673529057293).

**4. China and AI Competition: Focus on China's AI advancements and competition with the US.**

- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899594295914496302) believes **China will graduate hundreds of people of caliber comparable to AI greats** and that the **quality of Chinese ML grads and projects is increasing exponentially**, suggesting that the US's hiring pool is insufficient to compete. He also suggests [China is secretly steered by technocratic Isekai regression novel nerds](https://twitter.com/teortaxesTex/status/1899554789114958258).
- [@dylan522p](https://twitter.com/dylan522p/status/1899361483408150867) highlights **China's rise in robotics**, covering hardware basics and historical robotics firms in a series.
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899486329617993800) suggests **China may surpass the US in space** due to America's inability to build specialized roads while China focuses on scale, engineering, and logistics in space. He predicts [another "hockey stick event" in PRC mass to orbit within 5 years](https://twitter.com/teortaxesTex/status/1899360135941705992), noting they are demonstrably faster.  [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899501654598172768) contrasts US's "Stargate" approach with **China's building "1000 2K GPU sheds"**, questioning if China's tech market is more competitive than perceived "communist centralization".
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899541487919964418) argues the **West is disserving itself by focusing on "Communism" instead of the "Industrial Party of China"**, suggesting they are taking on the "White Man's Burden" the West gave up on.
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899540640351862954) questions the myth of "overcapacity", arguing that in key domains like housing, energy, chips, raw materials, and cars, "More Stuff Better," potentially contrasting with Western economic views.
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1899334394722107772) comments on **China commoditizing EVs and humanoids**, contrasting Elon Musk's vision with China's market actions.

**5. AI Research & Techniques: Core AI research concepts and techniques being discussed.**

- **New research on optimizing test-time compute via Meta Reinforcement Fine-Tuning (MRT)** was highlighted by [@rsalakhu](https://twitter.com/rsalakhu/status/1899597917016744445) and [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899392429893042485).  MRT is presented as a new fine-tuning method achieving **2-3x performance gain and 1.5x token efficiency** for math reasoning compared to outcome-reward RL, outperforming outcome-reward RL and achieving SOTA results at the 1.5B parameter scale.
- **Inductive Moment Matching (IMM)**, a new class of generative models from Luma AI for one- or few-step sampling, surpasses diffusion models on ImageNet-256x256 with 1.99 FID using 8 inference steps, as noted by [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899409323651993976).
- **Effective and Efficient Masked Image Generation Models (eMIGM)**, a unified framework integrating masked image modeling and masked diffusion models, outperforms VAR and achieves comparable performance to state-of-the-art continuous diffusion models with less NFE, according to [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899411428236259603).
- **Medical Hallucinations in Foundation Models** are benchmarked in a new study, finding **GPT-4o has the highest hallucination propensity in tasks requiring factual and temporal accuracy**, but **Chain-of-Thought (CoT) and Search Augmented Generation can reduce hallucination rates**, as reported by [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899414464698470507).
- [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1899491349881352628) highlighted research using **RLOO (Reinforcement Learning from Objective Optimization)** for training, noting the excitement around labs exploring algorithms beyond PPO.
- [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1899251749690708412) mentions **Diffusion language models that can arbitrarily reshuffle token positions** as potentially the most powerful way to scale test time compute for bounded sequence lengths.
- [@shaneguML](https://twitter.com/shaneguML/status/1899477905132138577) describes **Chain-of-thoughts as "dark knowledge" of LLMs**, allowing for deeper understanding of models through prompting methods.
- [@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1899475671958929453) discusses **AI distillation**, referencing his 1991 work and connecting it to DeepSeek's discussions.
- [@jerryjliu0](https://twitter.com/jerryjliu0/status/1899489337412378909) raises concerns about **versioning and regression testing in MCP (Model-as-Control-Plane) agent systems**, highlighting potential issues with dynamic behavior changes and API updates causing outages.
- [@rasbt](https://twitter.com/rasbt/status/1899493072972415154) released a **"Coding Attention Mechanisms" tutorial**, explaining self-attention, parameterized self-attention, causal self-attention, and multi-head self-attention.
- [@TimDarcet](https://twitter.com/TimDarcet/status/1899580380958589395) notes that **Gaussian Mixture Models (GMM) fit MNIST quickly and well using Expectation-Maximization (EM)**, questioning if EM GMM might be sufficient.

**6. Memes and Humor**

- [@aidan_mclau](https://twitter.com/aidan_mclau/status/1899316464836075545) made a humorous observation about **how many people, even at the Formula One level, misunderstand the function of brakes**. This tweet garnered significant attention. [@hkproj](https://twitter.com/hkproj/status/1899318678924984578) jokingly replied that **brakes are "clearly used to let the driver stretch their foot"**.
- [@nearcyan](https://twitter.com/nearcyan/status/1899272957333275015) recommended [@TrungTPhan](https://twitter.com/TrungTPhan) as **"honestly among the top posters on the entire site lately"**, praising his content and suggesting a strong follow.
- [@scottastevenson](https://twitter.com/scottastevenson/status/1899257834031685921) announced **"Vibecoding, but for legal docs. Coming soon."**

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Gemma 3 Anticipation and Potential Impact**

- **[New Gemma models on 12th of March](https://i.redd.it/8qfnwj7433oe1.jpeg)** ([Score: 387, Comments: 70](https://reddit.com/r/LocalLLaMA/comments/1j8u90g/new_gemma_models_on_12th_of_march/)): **Gemma 3** is set to be released on **March 12, 2025**, during the "Gemma Developer Day" event in **Paris**. The announcement features a sleek, modern design with a geometric star icon, highlighting the professional and high-tech nature of the event.
  - **Gemma 3 Expectations**: The community is anticipating the release of **Gemma 3** during the "Gemma Developer Day" event, with some users expressing skepticism about a confirmed release. Discussions highlighted the event's high-profile speaker panel and the expectation of significant announcements, although some caution against assuming a release given the event's closed-door nature.
  - **Technical Compatibility and Improvements**: There's a strong interest in ensuring **Gemma 3** works seamlessly with **llama.cpp**, with users recalling issues from **Gemma 2**'s launch and hoping for better integration this time. Some users mention Google's internal use of a **llama.cpp fork**, suggesting potential for improved compatibility and contributions to the open-source community.
  - **Model Variants and Performance**: Users are keen on seeing more mid-sized models like **Gemma 27B**, with suggestions for larger models like **32B, 40B,** and **70B** to enhance performance. There's also interest in smaller models like **9B** and **12B** for specific tasks, emphasizing the need for diverse model sizes to cater to different use cases.


**Theme 2. M3 Ultra 512GB Review with Deepseek R1 671B Q4**

- **[M3 Ultra 512GB does 18T/s with Deepseek R1 671B Q4 (DAVE2D REVIEW)](https://www.youtube.com/watch?v=J4qwuCXyAcU)** ([Score: 384, Comments: 215](https://reddit.com/r/LocalLLaMA/comments/1j8r2nr/m3_ultra_512gb_does_18ts_with_deepseek_r1_671b_q4/)): The **M3 Ultra 512GB** achieves a performance of **18T/s** when paired with **Deepseek R1 671B Q4**, as highlighted in the **DAVE2D review**.
  - Discussions highlight issues with **RAG systems** and memory bandwidth, noting inefficiencies in the **R1/MoE architecture** and possible areas for optimization. Users discuss that smaller models are typically faster, but the **70B model** is slower than expected, and there are potential **scheduling/threading issues** causing pipeline stalls.
  - Commenters debate the **cost and efficiency** of the **M3 Ultra** versus other systems, comparing it to setups involving **Nvidia 5090s** and **H200s**, emphasizing the **energy efficiency** and availability of the M3 Ultra. Users mention that while the M3 Ultra has lower power consumption at **under 200W**, alternative systems might offer higher performance but at greater cost and power usage.
  - There are detailed technical discussions about **quantization methods** like **Q4_K_M** and memory interleaving, with references to **GGML_TYPE_Q6_K** and **super-blocks** for quantization. Users also discuss the **memory bandwidth** and its implications on performance, particularly when running **inference** on systems with large RAM capacities.


**Theme 3. NVLINK's Impact on RTX 3090 Performance**

- **[NVLINK improves dual RTX 3090 inference performance by nearly 50%](https://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html)** ([Score: 144, Comments: 41](https://reddit.com/r/LocalLLaMA/comments/1j8i9rc/nvlink_improves_dual_rtx_3090_inference/)): **NVLINK** reportedly boosts the inference performance of dual **RTX 3090** GPUs by nearly **50%**. This suggests a significant improvement in computational efficiency for tasks leveraging these GPUs in tandem.
  - **Motherboard and GPU Configuration**: Users discussed the motherboard's PCIe lane configuration, noting that using x8 risers might limit performance. **hp1337** explained their setup with 6 GPUs using x8 lanes, suggesting future tests with x16 lanes for potential performance insights.
  - **NVLink Availability and Alternatives**: **FullOf_Bad_Ideas** questioned the availability and cost of NVLink bridges for RTX 3090s, with **a_beautiful_rhind** suggesting an alternative using [open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules). However, **Pedalnomica** noted this only enables P2P, not matching NVLink's performance.
  - **Quantization and FP8 Calculations**: **__JockY__** and others discussed the use of FP8 quantization on RTX 3090s, highlighting that **vLLM** uses the FP8 Marlin kernel for performance without native FP8 hardware, as confirmed by **Competitive_Buy6402** and **bihungba1101** with references to [vLLM's GitHub](https://github.com/vllm-project/vllm/pull/5975).


**Theme 4. Alibaba's R1-Omni for Emotion Recognition**

- **Alibaba just dropped R1-Omni!** ([Score: 244, Comments: 76](https://reddit.com/r/LocalLLaMA/comments/1j8mrju/alibaba_just_dropped_r1omni/)): **Alibaba** has launched **R1-Omni**, which focuses on enhancing emotional intelligence through **Omni-Multimodal Emotion Recognition** and **Reinforcement Learning**.
  - **Ethical Concerns**: Several commenters express concerns about the ethical implications of emotion detection technology, highlighting issues such as invasiveness and potential discrimination against neurodivergent individuals. There are worries that automating such subjective tasks could lead to misuse and harm, particularly if used without consent or transparency.
  - **AI in Therapy**: The discussion around AI therapists is polarized, with some seeing potential benefits like accessibility and consistency, while others warn of risks such as reinforcing anxieties or lacking human oversight. The debate touches on the balance between cost, effectiveness, and the potential for misuse by corporations.
  - **Technical and Community Aspects**: There is a mention of the **R1-Omni** model being available on [GitHub](https://github.com/HumanMLLM/R1-Omni), with questions about its relation to Alibaba and internal competition. Users also critique the naming conventions of models and request demonstrations of the technology.


**Theme 5. Reka Flash 3: New Open-Source 21B Model**

- **Reka Flash 3, New Open Source 21B Model** ([Score: 220, Comments: 50](https://reddit.com/r/LocalLLaMA/comments/1j8tfh5/reka_flash_3_new_open_source_21b_model/)): **Reka Flash 3** is a new **open-source model** featuring **21 billion parameters**. It is available on **HuggingFace** and more details can be found in the [Reka AI blog](https://www.reka.ai/news/introducing-reka-flash).
  - The **Reka Flash 3** model, despite its smaller size of **21 billion parameters**, is being compared to larger models like **QwQ-32B** and has shown promising performance benchmarks. Some users noted its potential for use in scenarios where speed is prioritized over size, while others expressed skepticism about its coding capabilities, particularly when compared to models like **Mistral Nemo**.
  - Discussions highlighted the model's **Apache license**, which allows for broad usage, and its suitability for **24GB cards** due to its size. There is interest in its potential multimodal capabilities, though it is currently not confirmed.
  - There is a strong interest in the model's reasoning capabilities, with users impressed by its ability to solve complex problems like the "tiger riddle." This demonstrates the model's potential in handling intricate reasoning tasks, which were previously thought to require much larger models.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. Claude 3.7: Enhancing Developer Skills through Debugging**

- **Claude 3.7 made me a better developer.** ([Score: 234, Comments: 64](https://reddit.com/r/ClaudeAI/comments/1j8wiuo/claude_37_made_me_a_better_developer/)): The author criticizes **Claude 3.7** for producing overly complex and inefficient code, describing it as "absolute garbage" and "over-engineered nonsense." Despite the frustration, the author acknowledges that the process of fixing such code has improved their development skills, suggesting that resolving AI-generated code issues can be an effective learning experience.
  - Commenters emphasize the importance of proper **Git practices**, such as creating branches for new features and committing frequently to facilitate easy reversion of AI-generated code. They suggest using **rebase** to merge commits into a single one before merging back to the main branch, highlighting that frequent commits are professional and beneficial.
  - Some users discuss their experiences with **Claude 3.7** and **3.5**, noting that 3.7 often produces overly complex code, while 3.5 was simpler and more reliable. However, there are mixed opinions on 3.5's current performance, suggesting it may have degraded over time.
  - A few commenters share strategies for working with AI-generated code, including using **test-driven development** to guide code quality and having the AI explain concepts rather than directly generating code. They caution against relying on AI for high-level architectural decisions, as it often results in reimplementation of existing functionalities and excessive complexity.


- **[Dario Amodei: AI Will Write Nearly All Code in 12 Months!! Are Developers Ready?](https://v.redd.it/tqdzfzj1e2oe1)** ([Score: 181, Comments: 183](https://reddit.com/r/ClaudeAI/comments/1j8r1qi/dario_amodei_ai_will_write_nearly_all_code_in_12/)): **Dario Amodei** predicts that **AI** will write nearly all code within **12 months**, posing a significant shift for developers. The video content is not analyzed, but the title suggests a discussion on the readiness of developers for this rapid advancement in AI-driven coding.
  - Many users express skepticism about **Dario Amodei's** prediction, comparing it to past over-optimistic claims like **Elon Musk's** robotaxi timeline and **Hinton's** radiologist replacement forecast. They argue that AI-generated code still requires significant human oversight due to issues like hallucinations and logical errors, which are not easily resolved by current AI models.
  - Several commenters argue that while AI can assist in coding, it cannot yet replace developers due to its inability to autonomously manage complex tasks, ensuring code quality, and understanding design and architecture. They highlight that AI tools can generate code but still need human verification and guidance, making them more akin to advanced compilers rather than independent coders.
  - There is a consensus that the current hype around AI's capabilities is largely driven by marketing and fundraising efforts. Commenters emphasize that genuine breakthroughs in AI coding would likely leak due to their market impact, and that claims of rapid advancements often serve more to attract investor interest than reflect immediate technological reality.


- **[This is why I use ChatGPT instead of Grok](https://i.redd.it/iuewe1hqnzne1.png)** ([Score: 191, Comments: 14](https://reddit.com/r/ChatGPT/comments/1j8irsl/this_is_why_i_use_chatgpt_instead_of_grok/)): The post criticizes **Claude-generated coding output** and expresses a preference for **ChatGPT** over **Grok**. An accompanying image humorously contrasts using **Reddit** on a **PC** as a more intellectual activity compared to "doomscrolling" on a phone, suggesting that the former is akin to "curating knowledge" or engaging in “Reddit Discourse Analysis.”
  - **ChatGPT vs. Grok**: **Grok** is criticized for being less versatile and overly complicated compared to **ChatGPT**, which, despite being labeled as a "liar," is preferred for tasks like grammar correction. Users express frustration with **ChatGPT**'s tendency to delete content it deems unnecessary without acknowledgment.
  - **Doomscrolling on Devices**: The discussion highlights that **doomscrolling** is similar across devices, with the difference being that a **PC** setup might appear more controlled but still involves the same mental strain. The distinction is more about the optics and control rather than the device itself.
  - **User Experience with AI Models**: There is interest in comparing responses between different AI models like **Grok 3** and **GPT-4.5**, but limitations such as the **50 message per week cap** on **GPT Plus** hinder such explorations.


**Theme 2. Nvidia's Gen3C: Advancements in Image to 3D Conversion**

- **[Gen3C - Nvidia's new AI model that turned an image into 3D](https://v.redd.it/bqq73tdca1oe1)** ([Score: 259, Comments: 25](https://reddit.com/r/StableDiffusion/comments/1j8n8qh/gen3c_nvidias_new_ai_model_that_turned_an_image/)): **Nvidia's Gen3C** is a new AI model that converts 2D images into 3D representations, showcasing advancements in image processing technology.
  - **Memory Concerns**: Users express concerns about **Gen3C** potentially being memory-intensive, questioning its feasibility on consumer-grade GPUs. **TheSixthFloor** suggests that it might require at least **16GB VRAM** similar to other advanced AI models.
  - **Technical Clarifications**: **Silonom3724** clarifies that **Gen3C** uses **Image to point cloud to NeRF** rather than direct 3D polygon representation, while **grae_n** notes the inclusion of reflective materials suggesting a **gaussian/NeRF** approach.
  - **Availability and Access**: The **Gen3C** code is anticipated soon, with links provided to the [GitHub repository](https://github.com/nv-tlabs/GEN3C) and [Nvidia's research page](https://research.nvidia.com/labs/toronto-ai/GEN3C/). Users are eager for updates on its release and local run capabilities.


**Theme 3. Dario Amodei: AI Code Generation Predictions and Skepticism**

- **[Dario Amodei: AI Will Write Nearly All Code in 12 Months!!](https://v.redd.it/nc1umxord2oe1)** ([Score: 139, Comments: 130](https://reddit.com/r/OpenAI/comments/1j8r0jw/dario_amodei_ai_will_write_nearly_all_code_in_12/)): **Dario Amodei** predicts that **AI** will write nearly all code within **12 months**, sparking skepticism among the engineering community. The absence of detailed arguments in the post limits further analysis of the prediction's feasibility.
  - Critics argue that **AI** lacks the capability to write all code within **12 months** due to limitations in context window size, which affects its ability to maintain awareness across large codebases. **AI** struggles to handle complex systems like the **Linux kernel** or critical system control code effectively, as evidenced by the failure to convert the kernel to **Rust**.
  - Skepticism arises over the practicality of **AI** writing code without human oversight, with engineers emphasizing the necessity of human review and clear specifications, which **AI** currently cannot independently manage. **Middle management** is criticized for lacking the technical expertise to guide **AI** in this task.
  - Some commentators view **Dario Amodei's** prediction as a strategic move to attract funding, rather than a realistic forecast. The current limitations of tools like **Copilot** highlight the challenges **AI** faces in efficiently handling large projects.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Exp

**Theme 1: Open AI's Agent Development Ecosystem Evolves**

- **OpenAI Unveils Responses API and SDK for Agent Creation**: OpenAI launched a new [Responses API](https://platform.openai.com/docs/quickstart?api-mode=responses) and [Agents SDK](https://platform.openai.com/docs/guides/agents) to simplify agent development, emphasizing improved integration, streamlined workflows, and production readiness. The new SDK offers features such as tracing, guardrails, and lifecycle events, but also signals a sunset for the [Assistants API](https://openai.com/policies/terms-of-use/) by mid-2026.
- **Community Debates Merits of New Agent Tools**: The community is actively debating the value and functionality of the new tools, with some questioning the trustworthiness and consistency of code generated by **GPT-4.5** and seeking clarity on differences between the Responses API and existing chat completions API.  While the new [Web Search Tool](https://platform.openai.com/docs/guides/tools-web-search) aims to improve search result reliability, users have observed that it lacks source selection capabilities similar to other platforms.
- **Agents SDK's Observability Tools Triggers Tracing Questions**: OpenAI's claim about Braintrust data tracing integration in its new [Agents SDK](https://platform.openai.com/docs/guides/agents#orchestration) is creating buzz, as users are wondering if OpenAI supports integrations with Langfuse or other agent observability tools. It does, and more details on how to use the [OpenAI's SDK](https://platform.openai.com/docs/guides/agents#orchestration) or send traces to your own can be found in [this Github repo](https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md#custom-tracing-processors).

**Theme 2: Navigating the Frontier of AI Model Capabilities and Limitations**

- **Reka Flash 3 Enters the Ring**: Reka Labs released [Reka Flash 3](https://huggingface.co/RekaAI/reka-flash-3), a 21B reasoning model trained from scratch, showcasing competitive performance and multimodal capabilities, challenging existing models like QwQ-32B and o1-mini. Despite its open-source nature, questions linger about its architecture and training data, shifting its purpose from on-device use to powering Reka's AI worker platform [Nexus](https://getnexus.reka.ai).
- **Anthropic's Claude 3.7 Faces Output Constraints on Perplexity**: Users discovered that [Claude 3.7](https://www.anthropic.com/news/claude-3-family) has an output limit of 5000 tokens on Perplexity, contrasting with Anthropic's official documentation stating it can output up to 128K. This discrepancy raises questions about the model's practical utility and highlights the importance of understanding platform-specific limitations, particularly in commercial applications.
- **GPT-4.5 Code: Inconsistently Hilarious**: Users report that [GPT-4.5](https://openai.com/blog/new-models-and-developer-products-announced-at-devday) generates inconsistent code, such as calling a non-existent function `startApp()` after defining a `start()` function. Concerns are raised about the need for constant oversight of **GPT-4.5's** output and the trustworthiness of AI-generated code in general, describing a need to *babysit this 'intelligence'*.

**Theme 3: Community-Driven Tools and Techniques for AI Development**

- **AI Code Fusion Tool Debuts for Optimizing LLM Contexts**: A community member introduced [AI Code Fusion](https://github.com/codingworkflow/ai-code-fusion), a tool designed to optimize code for LLM contexts by packing files, counting tokens, and filtering content, showcasing the community's proactive approach to addressing challenges in AI development. The tool's creator is actively seeking feedback from the community to refine its functionality.
- **Aider's Watch Files' Live Mode Enables Interactive Coding**: Aider's new `--watch-files` flag enables [live mode](https://aider.chat/docs/usage/watch.html), allowing developers to interactively code with AI by adding comments like `AI!`, triggering Aider to make changes, and `AI?`, triggering it to answer questions, signaling a shift towards more collaborative and interactive coding workflows.
- **Leveraging Browserless.io to Bypassing Bot Detection**: Nous Research AI members recommended using [Browserless.io](https://www.browserless.io) for bypassing bot detection and CAPTCHAs in web scraping, highlighting its ability to avoid leaving subtle fingerprints and bypass many web protections. It supports browser automation using Puppeteer or Playwright and features a scraping IDE for testing and debugging.

**Theme 4: Hardware and Infrastructure Considerations for AI Workloads**

- **Local LLM vs Cloud GPU: The Great Debate Rages On**: Users debated the cost-effectiveness of running LLMs locally on high-end hardware, such as an M3 Ultra Mac Studio with 512GB of RAM, versus utilizing cloud-based GPUs, balancing performance with long-term affordability.  AMD users reported that Vulkan and ROCm performance was broken in drivers 24.12.1, with performance dropping by 35%, though ROCm was fixed in v1.1.13+.
- **Speculative Decoding Stalls on Some Setups**: Speculative decoding may perform worse than standard inference when limited by RAM bandwidth, or when comparing 0.5b to 14b models. With the advent of 100Gbit NVMe drives, 400Gbit networks, and CXL memories, swap is becoming useful again, as highlighted in Dave2D's M3 Ultra Mac Studio Review.
- **SemiAnalysis Hosts Nvidia Blackwell GPU Hackathon**: [SemiAnalysis](https://semianalysis.com/hackathon-2025/) is hosting an Nvidia Blackwell GPU Hackathon on March 16th, featuring hands-on exploration of Blackwell & PTX infrastructure and speakers from OpenAI, TogetherAI, and Thinking Machines. The hackathon was mentioned across multiple Discords, highlighting its industry significance and attracting developers with the promise of early access to cutting-edge GPU technology.

**Theme 5: Ethical Concerns and Usage Policies in AI Development**

- **Discussions around OpenAI Terms of Service and Jailbreaking**: In light of [OpenAI's Terms of Service](https://openai.com/policies/terms-of-use/), members had cautionary discussions, and server rules also prohibit discussions on how to bypass these restrictions, while suggesting focusing on allowed content within ethical boundaries. Exploration is permitted through text about violence involving fantasy writing, image generation, or roleplaying games is **not forbidden** by those general policies.
- **Prompting Techniques and Creative Use-Cases Discussed**: Members on OpenAI are using prompting techniques to attempt to elicit more candid responses from models, without violating safety policies. Questions were proposed that included having the model teach programming similar to how a user's grandma used to.
- **User Want Grok Vibes for ChatGPT**: Discussions focused on the desired *vibes* in relation to filtering content; users shared memes such as [this one](https://tenor.com/view/elon-musk-this-is-elon-musk-musk-tesla-egifmeme-gif-13716021226937735268) and expressed the desire for a ChatGPT that does not filter out or restrict content in the same way. Deep Research price comparisons were also made, citing **OpenAI's deep research as the best choice**, but also acknowledging that *the limits SUCK right now lol*.

---

# PART 1: High level Discord summaries




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Nightly Bites the Dust**: The latest nightly update for **Cursor** introduced critical bugs, breaking the **AI Chat** and **Cursor settings**, rendering the GUI unusable.
   - Users reported that reinstalling the app did not resolve the issues, indicating a problem with the latest nightly update itself.
- **Claude 3.7 Pricing Sparks Outrage**: Users are upset with the new pricing for **Claude 3.7 Thinking**, which now costs **2 requests instead of 1**, prompting some to consider alternatives.
   - Discussions highlighted that using **Claude 3.7 Thinking** with large context could potentially cost up to **16 cents per request**.
- **Manus AI: Revolutionary Agent or Overhyped Tool?**: A user shared **Manus AI**, calling it *the craziest AI agent*, showcasing its ability to perform tasks such as cloning the Apple website ([Tweet from el.cine](https://x.com/ehuanglu/status/1899110687902978373?s=46&t=CLGnxOi5OPp22iT8UYkr1A)).
   - Skeptics suggested it might be **Sonnet 3.7** with some tools to use your PC, while others speculated a future with AI agents running their companies.
- **Cursor's Stability Faces Scrutiny**: Multiple users reported **Cursor is barely working**, often stuck or unresponsive, with **Claude Max** not functioning for some.
   - Some users found that rolling back to version **.46.11** fixed the problems, leading to speculation that version **.47** might be restricted to a limited user base.
- **Local LLM vs Cloud GPU: The Great Debate**: A user suggested buying an **M3 Ultra Mac Studio with 512GB of RAM** to run **full DeepSeek R1 locally**, triggering discussions on the cost-effectiveness of the setup.
   - While some favored local LLMs, others argued that cloud-based **GPUs** offer faster inference and are more economical in the long run.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Releases Desktop App**: Perplexity AI released a native **desktop app for PC** ([perplexity.ai/platforms](https://www.perplexity.ai/platforms)), enabling **voice dictation**, **keyboard shortcuts**, and access to the latest models.
   - However, users note that the app is essentially a wrapper for the web version, lacking desktop advantages and browser extensions like Complexity; *"it's just a nerfed web browser"*.
- **Revolut Promo Codes Give Users a Headache**: **Revolut** users are experiencing issues redeeming **Perplexity Pro** promo codes, with some being told they need to create a new account or contact Revolut.
   - As one user mentioned, *"I contacted Revolut and they said I need to register new account with Perplexity. Its a bummer, but hey, still worth it I guess."
- **Claude 3.7 Capped at 5K Tokens**: Users discovered that **Claude 3.7** has a hard output limit of **5000 tokens** on Perplexity.
   - This contrasts with Anthropic's official documentation stating it can output up to **128K**.
- **Universities Explore Perplexity Enterprise**: A user is evaluating integrating **Perplexity Enterprise** into a university system, emphasizing its ability to connect to internal knowledge bases for policies and procedures, see [Perplexity Enterprise FAQ](https://www.perplexity.ai/enterprise).
   - The platform offers features for internal data search and customized workspaces.
- **API chat completions experience truncation**: A member reported intermittent content truncation when calling the **chat.completions API** with the **sonar-reasoning-pro model**; see [Perplexity AI Playground](https://docs.perplexity.ai/api-reference/chat-completions).
   - Increasing **max_token** allowances did not resolve the issue; the member suggested that the [Perplexity AI Playground](https://docs.perplexity.ai/api-reference/chat-completions) consistently outputs full responses, suggesting the issue is specific to the API.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Reka Flash 3 Sparks Interest**: [Reka Flash 3](https://huggingface.co/RekaAI/reka-flash-3), a **21B reasoning model** under the Apache 2.0 license, comparable to **QwQ-32B** and **o1-mini**, has been released.
   - The Reka team consists of ex-DeepMind employees, and [Reka's website](https://www.reka.ai/ourmodels) states that the flash model is multimodal.
- **Multi-GPU Training Recommendations Given**: When asked about finetuning a large model across multiple nodes and multiple GPUs using Unsloth, a member recommended using **axolotl** or **llama factory**.
   - Currently, Unsloth does not (officially) support multi-GPU, although support may be arriving in the coming weeks.
- **AI Code Fusion Tool Makes Debut**: A member introduced **AI Code Fusion**, a tool designed to optimize code for **LLM contexts** by packing files, counting tokens, and filtering content, available on [GitHub](https://github.com/codingworkflow/ai-code-fusion).
   - The creator of **AI Code Fusion** is seeking feedback from the community on this tool.
- **Regex Beats LLMs in Date Extraction**: A user who aimed to train a model to extract the right opening times from queries was advised that a **regex system** might be more suitable than using AI for this task.
   - A member linked to an relevant [xkcd comic](https://xkcd.com/208/) about *over-engineering simple tasks with complex solutions.*
- **GRPO Batch Size Affects Training**: The **GRPO batch size** must be the same as the number of generations and `num of generation` for the GRPO RL algorithm must be tuned well.
   - It was recommended that the range for `num generations` is **4 to 8**, and that increasing the batch size multiple reduces your training time, but increases GPU memory requirement drastically.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Deep Hermes Shows Early Reasoning**: The new **Deep Hermes** model was released with *early reasoning capabilities*, distilled from R1, as shown on [Hugging Face](https://huggingface.co/).
   - Members are excited to test it, but expressed concern about exceeding context length.
- **Scrape Without Detection via Browserless.io**: A member recommended [Browserless.io](https://www.browserless.io) for bypassing bot detection and CAPTCHAs in web scraping, highlighting its ability to *avoid leaving subtle fingerprints*.
   - It supports browser automation using **Puppeteer** or **Playwright** and features a scraping IDE for testing and debugging.
- **SemiAnalysis Hosts Blackwell GPU Hackathon**: [SemiAnalysis](https://semianalysis.com/hackathon-2025/) is hosting a **Nvidia Blackwell GPU Hackathon** on March 16th, featuring hands-on exploration of Blackwell & PTX infrastructure and speakers from **OpenAI**, **TogetherAI**, and **Thinking Machines**.
   - The event is sponsored by companies like **Together**, **Lambda**, **Google Cloud**, **Nvidia**, and **OpenAI**.
- **Optimize UTs with Forward Gradients**: Members discussed using [forward gradients](https://arxiv.org/abs/2202.08587) for optimizing **Universal Transformer (UT)** training, as they may be more efficient due to the shared layers in UTs.
   - This approach may be interesting when combined with **N-GPT**.
- **ByteDance Launches Trae IDE**: ByteDance has released [Trae](https://www.trae.ai/), a free, AI-based IDE similar to Cursor, featuring **Claude Sonnet 3.7** for free use, and is available for **Mac** and **Windows**.
   - A Linux version is planned, and the IDE targets beginners in AI coding.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Loglikelihood Evaluation Liberates LLMs**: Members recommended using **loglikelihood-based evaluation** for Multiple Choice Question Answering (**MCQA**) tasks, bypassing the need for strict output formatting.
   - This explains why instruct models get some answers correct, but their chat alternatives usually score **0**.
- **Diffusion Models Perform Spectral Autoregression**: A blog post ([Spectral Autoregression](https://sander.ai/2024/09/02/spectral-autoregression.html)) reveals that **diffusion models of images perform approximate autoregression in the frequency domain**.
   - The author notes this theory feels intuitive but has little predictive power in practice, especially when using colored noise matching the RAPSD of the target distribution.
- **Neural Flow Diffusion Models Enhance Gaussian Noise**: **Neural Flow Diffusion Models (NFDM)** enhance diffusion models by supporting a broader range of forward processes beyond the standard Gaussian noise, with an end-to-end, simulation-free optimization objective.
   - Experiments demonstrate **NFDM's** strong performance and state-of-the-art likelihood estimation, according to the [paper](https://arxiv.org/abs/2404.12940).
- **Guidance From Badness Averts Mode Dropping**: A [paper](https://arxiv.org/abs/2406.02507) suggests guiding away from *badness* rather than unconditional-ness to avoid the mode dropping of CFG (classifier-free guidance).
   - The approach leads to disentangled control over image quality without compromising the amount of variation, yielding record FIDs of **1.01** for 64x64 and **1.25** for 512x512 on ImageNet.
- **Tokenizer Troubles Threaten Patching Evaluation**: A member seeks advice on choosing the right **metrics** to evaluate patching results when analyzing important circuits for **Math CoT** answers.
   - The core issue is that the tokenizer splits numbers like **10** and **15** into two tokens each, disrupting the straightforward application of the evaluation equation.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Avoma Competes with Gong**: [Avoma](https://www.avoma.com/), an all-in-one **AI platform** for automating note-taking, scheduling, coaching, and forecasting, was suggested as a competitor to **Gong**.
   - The suggestion was made in #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1349013362956304435).
- **Factorio Learning Environment Tests LLMs**: The **Factorio Learning Environment (FLE)**, [available on GitHub](https://jackhopkins.github.io/factorio-learning-environment/), is designed to test agents in long-term planning, program synthesis, and resource optimization using the game **Factorio**.
   - A member expressed excitement and humorously requested a job at the *Anthropic Factorio lab* immediately, highlighting that the environment is currently text-only but could benefit from multimodal data input like **Qwen 2.5 VLM**.
- **Contextual AI Unveils Instruction-Following Reranker**: **Contextual AI** introduced [a new reranker](https://contextual.ai/blog/introducing-instruction-following-reranker/) that follows custom instructions to rank retrievals based on requirements like recency, document type, or source.
   - The announcement was made in #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1349013362956304435).
- **OpenAI Launches Agent Tools**: **OpenAI** launched new tools for building agents, including a [Responses API](https://platform.openai.com/docs/quickstart?api-mode=responses), [Web Search Tool](https://platform.openai.com/docs/guides/tools-web-search), [Computer Use Tool](https://platform.openai.com/docs/guides/tools-computer-use), and [File Search Tool](https://platform.openai.com/docs/guides/tools-file-search).
   - They also released a new open source [Agents SDK](https://platform.openai.com/docs/guides/agents) with integrated [Observability Tools](https://platform.openai.com/docs/guides/agents#orchestration) with tracing, guardrails and lifecycle events, advertising that the SDK is production ready.
- **Luma Labs Introduces Inductive Moment Matching**: **Luma Labs** released [Inductive Moment Matching (IMM)](https://lumalabs.ai/news/inductive-moment-matching), a new pre-training technique that claims to deliver superior sample quality with 10x more efficiency compared to diffusion models.
   - The discussion was centralized in #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1349013362956304435).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter launches FAQ Page**: OpenRouter launched a [FAQ page](https://openrouter.ai/docs/faq) to address common questions and provide more clarity for users.
   - Alongside the new FAQ, a small quality of life update was released to improve user experience.
- **Gemini 2.0 Image Generation Leaks**: **Gemini 2.0 Flash Experimental** image generation is out, capped at **32k** context, but lacks code execution, search grounding, or function calling, with users finding the image save code under `gemini-2.0-flash-exp`.
   - This was found on [this Reddit post](https://www.reddit.com/r/Bard/comments/1j8r61n/native_imagegen_not_my_screenshot/).
- **OpenAI Teases Dev-Facing Reveal**: Members speculated about an **OpenAI** reveal based on [this post](https://platform.openai.com/docs/api-reference/responses) mentioning the **Responses API**.
   - The reveal was expected at **10 AM PT**.
- **Cohere's AYA Vision Questioned**: Members inquired about **OpenRouter's** support for **AYA vision** by **Cohere** and other **Cohere** models, with pricing for **AYA Expanse** models (8B and 32B) potentially at **$0.50/1M Tokens for Input** and **$1.50/1M Tokens for Output**.
   - Users are still trying to confirm these rates, as seen in [this screenshot](https://cdn.discordapp.com/attachments/1094454198688546826/1349049467378339902/image.png?ex=67d1afb9&is=67d05e39&hm=ffdce841e8f353b45682a480ea8f937b0169a3414ebe0967c87230ba436786b4&).
- **Parameter Calculation Gets Axed**: **OpenRouter** removed the parameter calculation due to inaccuracy, deeming it potentially misleading.
   - The team plans to revamp it with manual curation later, acknowledging the difficulty in tuning parameters, humorously calling them *ancient runes*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Agent Tools Unveiled at OpenAI Dev Livestream**: OpenAI debuted **Agent Tools for Developers** during a livestream, followed by an **AMA** (*Ask Me Anything*) session offering direct interaction with the development team, with more information and questions on the [OpenAIDevs X post](https://x.com/OpenAIDevs/status/1899502117171155002).
   - The **AMA** was scheduled from **10:30–11:30 AM PT**, allowing developers to directly engage with the team behind the new features.
- **Users Crave Grok's Vibe in ChatGPT**: Members expressed a desire to bring **Grok's** unique characteristics to **ChatGPT**, as seen in an [Elon Musk GIF](https://tenor.com/view/elon-musk-this-is-elon-musk-musk-tesla-egifmeme-gif-13716021226937735268) referencing **Elon Musk**.
   - Discussion revolved around the nature of the desired *vibes*, specifically in relation to filtering content.
- **GPT-4.5 Code Generates Inconsistencies**: Users reported that **GPT-4.5** generated inconsistent code, such as calling non-existent functions or misnaming existing ones, leading to questions about its reliability compared to **GPT-4o**.
   - Concerns were raised about the need for constant oversight of **GPT-4.5's** output and the trustworthiness of AI-generated code in general, describing a need to *babysit this 'intelligence'*
- **New Responses API Mirrors Assistant API?**: A member inquired about the differences between the **new responses API** and the existing **chat completions API**, sparking a discussion on the API's functionalities.
   - Clarification emerged suggesting that the *new responses API* is *basically the Assistants API but better*.
- **Jailbreaking Endangers ToS**: Members discussed simulating scenarios to get **AI models** to bypass restrictions, or improve accuracy, which is considered 'jailbreaking', but may violate [OpenAI's Terms of Service](https://openai.com/policies/terms-of-use/).
   - Users were cautioned against violating **ToS** to protect account access, with the server prohibiting discussions on bypassing restrictions; discussion of violence involving fantasy or roleplaying is not considered forbidden.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 'Watch Files' Now Live**: Paul Gauthier announced that running `aider` with the `--watch-files` flag now enables **live mode**, watching all files in the repo for coding instructions via `AI`, `AI!`, or `AI?` comments, as shown in the [Aider browser UI demo video](https://aider.chat/docs/usage/watch.html).
   - The exclamation point `AI!` triggers aider to make changes, while the question mark `AI?` triggers it to answer questions.
- **Aider Daily Budget Varies Wildly**: Members discussed **Aider's** daily budget, with one reporting roughly **2x** the leaderboard cost for **Sonnet 3.7** with 7-12 hours of AI coding per week.
   - They cautioned that a **40-hour week** could easily result in **8-10x** the leaderboard cost, while other users manage cost by defaulting to cheaper models like **o3 or R1**.
- **DMCA Takedown Shuts Down Claude Code**: A user reported receiving a **DMCA** takedown notice for forking the **Claude code leak repo**, with the original leaker and all forks affected.
   - Another user speculated about the possibility of **o1 pro** / **o3 mini pro** releases in the API soon.
- **Aider's Edit Format Defined**: The *correct edit format* in the Aider leaderboard refers to the format Aider expects from the LLM for editing files, with [Aider documentation on edit formats](https://aider.chat/docs/more/edit-formats.html) detailing the *whole* and *diff* editing formats.
   - Different models perform better with different formats.
- **Code-Act Repo Potentially Interesting**: A member shared a link to the [code-act repo](https://github.com/xingyaoww/code-act).
   - They noted that it might be related to the discussion.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Unity Hooks Up with LM Studio**: A member showcased a [YouTube video](https://www.youtube.com/watch?v=dQw4w9WgXcQ) connecting **Unity** and **LM Studio** using a **JSON** file for data, but was unsure where to post it in the **Discord**.
   - Users are requesting a dedicated **Unity** channel for better organization.
- **DIY Internal LLM Chat Advice Sought**: A member is seeking advice on setting up an internal **LLM Chat** with user accounts, integrated with their company's **Google Docs** knowledge base, potentially using an inference **API**.
   - They're considering **LlamaIndex** for vector DB, **AnythingLLM** or **OpenWebUI** for chat interfaces, and exploring options within **LM Studio**.
- **Python SDK Lacks Vision, TypeScript SDK Ahead**: A member using **Python SDK 1.0.1** noticed the **Typescript SDK** can send images to vision models, but this feature hasn't been ported to **Python** yet.
   - The community is awaiting the arrival of vision model support in the **Python SDK**.
- **Copy4AI: Extension Snaps Code Context**: A member inquired about the `ext install` command for the [Copy4AI extension](https://copy4ai.dev/), designed to copy code snippets for AI assistants.
   - The extension, now named `leonkohli.snapsource`, can be accessed via the extension sidebar in **VS Code**.
- **AMD Driver Disaster: Vulkan and ROCm Crippled, Some Recovered**: An AMD user reported **Vulkan** and **ROCm** performance tanked by **35%** in drivers **24.12.1**, but **ROCm** was fixed in **v1.1.13+**.
   - Vulkan performance remained at **50%** in **25.1.1**, improving incrementally in **25.2.1**, with a [bug report submitted to AMD](https://www.amd.com/en/support/kb/faq/rs-help).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **X Gets Hit By Cyberstorm**: **Dark Storm** claimed responsibility for [a DDoS attack on X](https://www.forbes.com/sites/daveywinder/2025/03/11/x-under-attack-dark-storm-says-it-was-behind-musk-platform-ddos/), causing widespread outages on the platform.
   - Experts dismissed Elon Musk's suggestion of **Ukrainian** involvement, with **Ciaran Martin** calling it *"wholly unconvincing"* in a [BBC article](https://www.bbc.co.uk/news/articles/c62x5k44rl0o).
- **LanguageBind Beats ImageBind**: Members discussed using a single solution to process *image, audio, video, and pdf* modalities, and a member recommended [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind), noting it *supports all modalities* and *beats ImageBind*.
   - The model, trained from scratch on synthetic and public datasets, performs competitively with proprietary models like **OpenAI o1-mini**.
- **Reka Space Gets Even Smaller**: **Reka Flash 3**, a **21B** general-purpose reasoning model, can no longer be called *on-device* model but is used to power Nexus, Reka's platform for creating and managing AI workers with native deep research capabilities ([Reka Space](https://space.reka.ai), [getnexus.reka.ai](https://getnexus.reka.ai)).
   - The model, trained from scratch on synthetic and public datasets, performs competitively with proprietary models like **OpenAI o1-mini**, and powers Nexus, Reka's platform for creating and managing AI workers with native deep research capabilities.
- **RAGcoon Launches to Help Startups**: A new **agentic RAG** project named [RAGcoon](https://github.com/AstraBert/ragcoon) has launched, designed to assist in building startups by navigating various resources and suggestions via *hybrid search, query expansion*, and *multi-step query decomposition*.
   - Built on **LlamaIndex**, it uses **Qdrant** for vector database services, **Groq** for LLM inference (**QwQ-32B by Qwen**), **Hugging Face** for embedding models, **FastAPI** for the backend API, and **Mesop** by Google for the frontend, and boasts impressive **reliability of retrieved context**.
- **Ollama Takes Over HfApiModel**: Members showed how to replace `HfApiModel` from Hugging Face with **Ollama** for use with `smolagents` by creating a custom `OllamaModel` class that interacts with Ollama's API for prompt generation, allowing local LLMs to be used with `smolagents`.
   - They also shared snippets for using **Gemini, OpenAI, and DeepSeek models** with `smolagents`, providing examples for setting up the `LiteLLMModel` and `OpenAIServerModel` with appropriate API keys, and [a link to Google AI Studio](https://aistudio.google.com/app/apikey) was provided to obtain a free API key for **Gemini**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Noob Goes to San Jose**: Despite lacking **CUDA** experience, a member expressed interest in attending the **GPU mode** meeting on **March 16th** in **San Jose**.
   - The discussion sparks questions regarding the necessity of specialized knowledge for participation.
- **Triton's `tl.full` Beats Casting Conundrums**: A user successfully employed `tl.full` in **Triton** to craft a **0-dim tensor** with a defined value and data type (`tl.full((), 5, tl.int8)`) to bypass overflow quandaries when accumulating to a tensor.
   - The triumphant solution involved `tmp_5 = tl.full((1,), value=5, dtype=tl.int8); out = a.to(tl.int8) + tmp_5`.
- **Triton Triumphs Softmax Kernel Speed**: A user's pipeline softmax kernel in **Triton** surprisingly outperformed **PyTorch**, proving to be faster on a **float16 T4** colab, as demonstrated in [this image](https://cdn.discordapp.com/attachments/1189607595451895918/1349066304341934160/image.png?ex=67d1bf67&is=67d06de7&hm=e424c1148a06e3ac3adac9c31cd7c0bc6e930f047dc69c2db02cba55e5949695&).
   - Results show how **Triton** enables new high-throughput designs.
- **Padding Prevents SMEM Bank Conflicts**: Addresses for **stmatrix** need padding to circumvent targeting the same starting **SMEM bank**, which would otherwise trigger an 8x conflict, mirroring solutions previously implemented in fast.cu and deepgemm codes.
   - Given *no hardware solution* exists, memory layout management is critical when tiled layouts are impractical.
- **HuggingFace Libraries Migrate to TS/JS with WebNN/WebGPU**: A member is actively porting the entire **HuggingFace** libraries to **TS/JS** using **WebNN/WebGPU** to create a frontend implementation.
   - Separately, the initial structure for **IPFS Accelerate JS** was implemented with placeholder modules and TypeScript conversion via [this commit](https://github.com/endomorphosis/ipfs_accelerate_py/commit/2f9963372a890cc7d7abe4399f5cfa7fc438a773).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **SemiAnalysis Hosts Blackwell GPU Hackathon**: [SemiAnalysis](https://semianalysis.com/2025/03/11/america-is-missing-the-new-labor-economy-robotics-part-1/) is hosting an **Nvidia Blackwell GPU Hackathon** on **Sunday, March 16th**, featuring speakers from **OpenAI**, **TogetherAI**, and **Thinking Machines**.
   - The hackathon explores **Blackwell & PTX infrastructure** while collaborating on open-source projects, and is sponsored by Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave, and Nebius. More details are available at the [SemiAnalysis Hackathon page](https://semianalysis.com/hackathon-2025/).
- **Reka Labs Releases Reka Flash 3**: [Reka Labs](https://x.com/RekaAILabs/status/1899481289495031825) has open-sourced **Reka Flash 3**, a new reasoning model trained from scratch with only **21B parameters** achieving competitive performance.
   - The model was finetuned on synthetic and public datasets, followed by **RLOO** with model-based and rule-based rewards, forcing the model to output *&lt;/reasoning&gt;* to control quality vs. thinking time, as described in their [blog post](https://www.reka.ai/news/introducing-reka-flash).
- **Anthropic ARR Soars, Powers Manus AI**: According to [The Information](https://www.theinformation.com/articles/anthropics-claude-drives-strong-revenue-growth-while-powering-manus-sensation), **Anthropic** grew from **$1B ARR to $1.4B ARR** in the first two months of 2025, using their models to power **Manus**, *the latest AI sensation*.
   - The models are powering **Manus**, described as *the latest AI sensation*.
- **OpenAI Unveils New APIs and Agents SDK**: [OpenAI](https://x.com/btibor91/status/1899513477716410871) launched new APIs and tools for easier development of agent applications, including the **Responses API**, **Web search tool**, **File search**, **Computer use tool**, and an **open-source Agents SDK**.
   - The existing **Assistants API** will be phased out by mid-2026, and the changelog mentions new models **o3-mini-pro** and **o1-pro** in the API.
- **Dario Predicts AI Coding Domination**: Anthropic CEO, Dario Amodei, predicts that AI will write **90%** of the code in the next **3 to 6 months**, and nearly all code within **12 months**, according to a [tweet](https://x.com/slow_developer/status/1899430284350616025?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ).
   - This bold prediction has sparked discussion among developers regarding the future of coding and AI's role in it.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Servers Struggle in Cursor Integration**: Users reported issues integrating **MCP servers** like Brave Search with **Cursor**, despite successful integration with Claude, with errors like *no tools available* at [glama.ai/mcp/servers/gwrql5ibq2](https://glama.ai/mcp/servers/gwrql5ibq2).
   - One member acknowledged this as a **known limitation** with plans to address it.
- **Phoenix Framework Powers MCP Implementations**: A member showcased [MCPheonix on Github](https://github.com/jmanhype/MCPheonix), a simplified implementation of the **Model Context Protocol (MCP) server** using **Elixir's Phoenix Framework**.
   - This implementation simplifies **MCP server** creation and management.
- **MCP Powers Android Debugging**: A member introduced [DroidMind](https://github.com/hyperb1iss/droidmind), an **MCP server** that manages **Android devices** over **ADB**.
   - This project facilitates debugging on-device issues and analyzing logs, controlled via AI.
- **MCP Servers Spawn Other MCP Servers**: A member unveiled [mcp-create](https://github.com/tesla0225/mcp-create), an **MCP server** designed to build other **MCP servers**, with **TypeScript** support.
   - The project includes [an explanatory article](https://zenn.dev/tesla/articles/c66bda76c4a523) detailing its capabilities and direct execution of generated **MCP servers**.
- **Handoff Includes Full Context**: A member shared a [github.com search result](https://github.com/search?q=repo%3Aopenai%2Fopenai-agents-python%20handoff_prompt&type=code) noting that, by default, the **handoff** in **OpenAI's SDK** includes the entire conversation history.
   - This encompasses all system, user, and assistant messages.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Excels at Exam Prep**: A user reported *very good results* using **NotebookLM** to quiz themselves on study guide topics, splitting PDFs by bookmarks into separate notebooks.
   - The user turned the quiz results into flashcards in other apps for further study.
- **NotebookLM Generates Medical Documents**: A user in the medical field found **NotebookLM** useful for parsing guidelines and websites to create patient discharge information.
   - Specifically, they created a concise one-page document for patients regarding work-related injury claims.
- **Streamlining NotebookLM Ingestion**: A user is automating the optimization of information for upload to **NotebookLM**, focusing on smaller files for *easier robot ingestion*.
   - This streamlines their workflow for document processing in NotebookLM.
- **Gemini Generates Discontent**: A user expressed dissatisfaction with **Gemini**, despite its integration into the Google ecosystem.
   - The user did not mention details about their negative experiences.
- **NotebookLM Handles Massive Knowledge Bases**: A user with a **10 million word knowledge base** (1500 books, 6000 videos in text) inquired about **NotebookLM**'s limits.
   - A member of the **NLM** team clarified that NotebookLM supports **10 million words**, within the **300 source and 500,000 words/source limits**, leveraging **RAG**.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Challenges Users to Refer Friends**: The **Windsurf Referral Challenge** incentivizes users to refer friends, offering **500 flex credits** per friend's Pro subscription and a shot at custom **Airpods Pro Max** by **March 31st** via [windsurf.ai/refer](https://windsurf.ai/refer).
   - Most referrals wins, but all are winners with credits upon subscription.
- **Codeium Extension Can't Read Files**: The Codeium VS Code extension chat (**Claude 3.7 Sonnet**) cannot directly read script files from the folder and requires users to paste the file content into the chat.
   - Users are encouraged to *report it in codeium.com/support, because it should work technically*.
- **Claude 3.7 Sonnet Doesn't Work in VS Code Extension**: The **Claude 3.7 Sonnet Thinking** model is not available in the VS Code extension, unlike in Windsurf.
   - Users were told that **Claude 3.7 Sonnet Thinking** is *not available in the extension at the moment*.
- **Codeium's Error Aborts Pending Requests**: Users reported a persistent error, preventing Codeium from working, with the message *Codeium: The server aborted pending request* and mentioning a download URL from *releases.codeiumdata.com*.
   - The issue persisted across IDE restarts and different versions, with users suggested to contact *vscode@codeium.com*.
- **Windsurf Patches MCP and Sonnet**: Windsurf released **patch fixes in v1.4.6** addressing **MCP reliability**, **3.7 Sonnet web search**, and **proxy settings**, detailed in the [changelog](https://www.codeium.com/changelog).
   - **Windsurf Previews (Beta)** also now allows users to preview locally run websites directly within Cascade.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Compilers Hack Math Operations?**: Members debated whether compilers optimize calculations in deep learning frameworks like **PyTorch** and **NumPy**, specifically concerning the order of operations in complex equations like *(1/n) (a(c + d) + b)* versus *a(c/n + d/n) + b/n*.
   - One engineer suggested adding *extra brackets* to ensure the system performs operations in the desired order, while another pondered the trade-offs between minimal code and explicit code.
- **Matplotlib Drawn Amazingly by Claude 3.7**: Engineers showed excitement for the **Matplotlib** graphs generated by **Claude 3.7**, emphasizing that *benchmark and svgmaxing* are functioning as expected.
   - No specific links were given in this exchange.
- **Adaptive Meta-Learning: Framework or Fad?**: An engineer inquired whether the term **Adaptive Meta-Learning (AML)** is already established, describing it as a potential combination of *Online HyperParameter Optimization (HPO)* and meta-learning.
   - Another engineer shared [a Semantic Scholar search](https://www.semanticscholar.org/search?q=Adaptive%20meta-learning&sort=relevance) concluding that while the keywords are used together, they do not constitute a well-defined framework.
- **Virtual Reality Solves Prison Crisis??**: A California women's facility is finding success using **VR headsets** in solitary confinement, achieving *>>>97% reduction in infractions* according to [this article](https://www.theguardian.com/technology/2025/mar/08/vr-prison-california).
   - The VR program involves participants viewing scenes of daily life and travel adventures, processing their emotions through art.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Llama Extract Access Granted**: A member requested access to **Llama Extract** and was offered addition to the closed beta, pending email confirmation to `rasmus-persson@outlook.com`.
   - No further details were provided about the specifics of the closed beta.
- **Premium Plan Upgrade Made Easy**: A user inquired about upgrading to the **Premium plan** and received instructions to log in, click the profile icon, and select the upgrade/manage button.
   - No further discussion or details were provided regarding the features or benefits of the premium plan.
- **MP3 Parsing Puzzle for APIs**: A user reported an error when uploading an **.mp3** file for parsing through the API, noting that the upload works fine via the UI/webapp.
   - They provided [a screenshot of the error](https://cdn.discordapp.com/attachments/1059201661417037995/1349100831307202714/Screenshot_2025-03-11_at_3.24.19_PM.png?ex=67d1df8f&is=67d08e0f&hm=3b980c7dd220c3d654ff1cb17819daedcc6fc3c896b2ed955e800b40f2467d3d).
- **Function Calling Face-Off**: A member asked about alternative models besides those from **OpenAI** that are good for function calling, seeking a less expensive option.
   - No specific alternative models were recommended in the provided context.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Judge LLM Follows ChainPoll Pattern**: Members are building a **Judge LLM** which follows the **ChainPoll** pattern, and returns the *average* response chain.
   - A member suggested using `module.batch()` or `dspy.Parallel` to speed up the process.
- **Best of N Documentation Quest**: A member was having trouble finding docs on **Best of N**.
   - The same member noted that ensemble is listed as a teleprompter and asked if it optimizes or aggregates input programs into an optimal single program.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **OpenPipe Masters Deductive-Reasoning**: A member shared [OpenPipe's deductive-reasoning project](https://github.com/openpipe/deductive-reasoning), highlighting its use of **Torchtune** for SOTA deductive reasoning model training.
   - The project showcases **Torchtune's** capabilities in practical, cutting-edge AI applications, particularly in enhancing model training efficiency and effectiveness.
- **FP8 Fine-Tuning Faces Friction**: Members explored the difficulties of serving models in **FP8**, considering fine-tuning in **FP8** to mitigate quantization error but noted that **FP8** poses stability issues during training.
   - They suggested gradually increasing **weight decay** to keep weights in the optimal range during **FP8** fine-tuning.
- **Torchtune's QAT Quest for FP8**: A member inquired about **Torchtune's QAT (Quantization Aware Training) support**, specifically for **FP8**, with the goal of fine-tuning and reducing quantization error.
   - A promising [recipe](https://github.com/pytorch/torchtune/pull/2404) was identified as a potential solution for **FP8** QAT within **Torchtune**.
- **Regression Rendezvous Reveals Review Required**: The addition of [regression tests](https://github.com/pytorch/torchtune/pull/2477) prompted discussion about finalizing model size and evaluation methods.
   - Members questioned the sufficiency of evaluation alone, hinting at a deeper conversation around more comprehensive measurement strategies.
- **Evaluation Efficacy Examined Extensively**: The discussion pivoted to the need for measurement strategies beyond simple evaluation, with members debating the value of various evaluation metrics.
   - This deliberation is expected to influence decisions on model size and testing methodologies, promoting a shift towards more robust assessment practices.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Launches Expedition Aya 2024!**: **Cohere For AI** is launching [Expedition Aya 2024](https://tinyurl.com/ayaexp2025), a **6-week open-build challenge** focused on **multilingual, multimodal, and efficient AI**.
   - Participants gain access to **Cohere API credits**, with prizes including **limited edition Expedition swag** and recognition for top projects, with the kick-off meeting in **March 2025**.
- **SemiAnalysis Hosts Blackwell GPU Hackathon!**: [SemiAnalysis](https://semianalysis.com/) hosts an **Nvidia Blackwell GPU Hackathon** on **Sunday March 16th**, offering hands-on exploration of **Blackwell & PTX infrastructure**.
   - Speakers include **Philippe Tillet** from **OpenAI** and **Tri Dao** from **TogetherAI**, with sponsorship from **Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave, Nebius**.
- **Researcher Connects with Multilingual Communities**: A researcher inquired about multilingual and multicultural activities within the Cohere Discord community, highlighting their appreciation for **Cohere's** work.
   - New members are encouraged to introduce themselves, specifying affiliation, current projects, preferred tech/tools, and community goals, with **community expectations**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **SemiAnalysis Hosts Nvidia Blackwell GPU Hackathon**: **SemiAnalysis** is hosting an [Nvidia Blackwell GPU Hackathon](https://semianalysis.com/hackathon-2025/) on **Sunday, March 16th**, offering hands-on exploration of **Blackwell & PTX** infrastructure while collaborating on open-source projects.
   - Speakers include [Philippe Tillet of OpenAI](https://openai.com/), [Tri Dao of TogetherAI](https://www.together.ai/), and [Horace He of Thinking Machines](https://www.thinkingmachin.es/).
- **GTC Kickoff Features Blackwell GPUs**: **SemiAnalysis** is kicking off **GTC** with the **Blackwell GPU Hackathon**, which includes engaging morning keynotes, all-day hacking with powerful **Blackwell GPUs** like **GB200s**, and insightful afternoon talks.
   - The event is sponsored by Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave, and Nebius.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **CUDA Blogposts Anticipated**: A user is waiting for the new blog posts on **CUDA** to be released.
   - No additional information was provided.
- **Anticipation builds for CUDA update**: Enthusiasm is growing as users eagerly await the latest updates and blog posts regarding **CUDA**.
   - The community is keen to explore new features and improvements in CUDA, although specific details remain under wraps.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Qdrant Kicked out of ConvRAG**: The development teams considered **Qdrant** as the vector DB for their **ConvRAG** but decided to use a different one for unspecified reasons.
   - The selected DB offered greater flexibility for **VPC deployment**.
- **ConvRAG Chooses Alternative DB**: A different vector DB was chosen over Qdrant for **ConvRAG**.
   - The primary reason cited was the enhanced flexibility it provided for **VPC deployment** scenarios.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Nomic.ai (GPT4All) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1348973754817974374)** (1048 messages🔥🔥🔥): 

> `Cursor Nightly Bugs, Claude 3.7 Pricing, Potential of Manus AI, Criticism of Cursor's Stability, Local LLM vs Cloud GPU` 


- ****Nightly Curse Strikes Cursor****: The latest nightly update broke the **AI Chat** and **Cursor settings**, with users reporting that the GUI is broken, preventing them from opening the chat panel or settings.
   - One user suggested deleting and reinstalling the app, only to find the issue persisted with the latest nightly update.
- ****Claude 3.7 Pricing Causes Uproar****: Users expressed frustration over the new pricing for **Claude 3.7 Thinking**, which now costs **2 requests instead of 1**, leading some to consider canceling their subscriptions and switching to alternatives like **Roo Cline**.
   - The discussion also covered the potential costs of using **Claude 3.7 Thinking** with large context enabled, estimating it could reach **16 cents per request**.
- ****Manus AI Captivates the Audience****: A user shared a link to **Manus AI**, describing it as *the craziest AI agent* they've seen, capable of tasks like cloning the Apple website.
   - Some users expressed skepticism, suggesting it might just be **Sonnet 3.7** with some tools to use your PC, while others envisioned a future where multiple agents run their companies.
- ****Cursor's Stability Under Fire****: Multiple users reported that **Cursor is barely working today**, often stuck or unresponsive, with one user noting that **Claude Max** isn't working at all for them.
   - Some users found that rolling back to version **.46.11** resolved the issues, while others speculated that version **.47** is restricted to a limited number of users.
- ****Debate Sparks Over Local LLM vs Cloud GPU****: A user sparked a debate by suggesting the purchase of an **M3 Ultra Mac Studio with 512GB of RAM** to run **full DeepSeek R1 locally**, prompting discussion on the affordability and practicality of such a setup.
   - Some users argued that renting **GPUs** from cloud providers would offer faster inference and be more cost-effective in the long run, while others emphasized the benefits of having a local LLM machine for other tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com`">no title found</a>: no description found</li><li><a href="https://cursor.directory/mcp">Cursor Directory</a>: Find the best cursor rules for your framework and language</li><li><a href="https://docs.cursor.com/settings/beta">Cursor – Early Access Program</a>: no description found</li><li><a href="https://build-launch-win.lovable.app/">lovable x anthropic global hackathon</a>: no description found</li><li><a href="https://www.reddit.com/r/cursor/comments/1j8y05c/ama_with_cursor_devs_march_11_2025/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/prajwaltomar_/status/1899104347532738764?s=46&t=kUuVqsG2GMX14zvB592G5w">Tweet from Prajwal Tomar (@PrajwalTomar_)</a>: How to Set Up Cursor the Right WayCursor Rules are outdated. Project Rules is the correct way now. Here’s why it matters and how to set it up properly:</li><li><a href="https://x.com/ehuanglu/status/1899110687902978373?s=46&t=CLGnxOi5OPp22iT8UYkr1A">Tweet from el.cine (@EHuanglu)</a>: Manus AI is much crazier than Deepseek moment I just got invitation code, this thing is the craziest AI agent I&#39;ve seen.10 examples:1. Clone Apple websiteIt created a copy of Apple website that lo...</li><li><a href="https://x.com/prajwaltomar_/status/1899104347532738764?s=46&t=kUuVqsG2GMX1">Tweet from Prajwal Tomar (@PrajwalTomar_)</a>: How to Set Up Cursor the Right WayCursor Rules are outdated. Project Rules is the correct way now. Here’s why it matters and how to set it up properly:</li><li><a href="https://x.com/16footcatgirl/status/1899416927472103770?s=46">Tweet from Syl (e/tard) (@16footcatgirl)</a>: LMAOOOOOOO</li><li><a href="https://github.com/mannaandpoem/OpenManus">GitHub - mannaandpoem/OpenManus: No fortress, purely open ground.  OpenManus is Coming.</a>: No fortress, purely open ground.  OpenManus is Coming. - mannaandpoem/OpenManus</li><li><a href="https://youtu.be/etXFdqPu1Wk?si=IV-gAIcqn4X0FZTT">Cyberpunk 2077 on RTX 5090 with New DreamPunk 3.0 Graphics - Ultra Realistic Modded Showcase in 8K</a>: Extreme graphics showcase and gameplay using the all new RTX 5090 in Cyberpunk 2077 with DreamPunk 3.0. Graphics made possible with Path Tracing, DLSS 4, Mul...</li><li><a href="https://tenor.com/view/cool-fun-white-cat-dance-cool-and-fun-times-gif-9061261248949225544">Cool Fun GIF - Cool Fun White cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/swinging-hanging-hanging-in-there-bored-waiting-gif-18029735056767784446">Swinging Hanging GIF - Swinging Hanging Hanging in there - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/nikmcfly/ANUS">GitHub - nikmcfly/ANUS</a>: Contribute to nikmcfly/ANUS development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1349124326980325437)** (1 messages): 

> `Perplexity Windows app, Voice dictation, Keyboard shortcuts` 


- **Perplexity Releases Windows App**: Perplexity AI released an official **desktop app for PC**, providing access to **voice dictation**, **keyboard shortcuts**, and the latest models, available at [perplexity.ai/platforms](https://www.perplexity.ai/platforms).
- **Perplexity Windows App Features**: The app enables users to utilize **voice dictation** and **keyboard shortcuts** to interact more efficiently with Perplexity's latest models.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1348975860056592455)** (277 messages🔥🔥): 

> `Revolut Perplexity Pro Promo Code Issues, Claude 3.7 Token Limits, Perplexity Enterprise Integration, Sider AI Features and Comparison, Perplexity Windows App` 


- **Revolut Users Face Promo Code Puzzle**: Users are running into issues redeeming **Perplexity Pro** promo codes received via **Revolut**, with some being told they need to create a new Perplexity account, while others are directed to contact Revolut for a new code.
   - One user reported, *"I contacted Revolut and they said I need to register new account with Perplexity. Its a bummer, but hey, still worth it I guess."*
- **Claude 3.7's 5K Output Token Tango**: Users have discovered that **Claude 3.7** has a hard output limit of **5000 tokens** on Perplexity.
   - On Anthropic, it officially says it can output up to **128K**
- **Universities Eyeing Perplexity Enterprise**: A user is exploring integrating **Perplexity Enterprise** into a university system, focusing on its ability to connect to internal knowledge bases as a central hub for policies and procedures.
   - The [Perplexity Enterprise FAQ](https://www.perplexity.ai/enterprise) highlights features for internal data search and customized workspaces.
- **Is Sider AI the Underdog We Didn't Know We Needed?**: Users are buzzing about **Sider AI**, praising its polished UI and deep research capabilities, including access to **Claude 3.7 Sonnet** and other models for creative writing, translating, and research.
   - The web access is a paid feature.
- **Perplexity Windows App: A Wolf in Chrome's Clothing?**: The newly released **Perplexity Windows app** is under scrutiny, with users noting it's essentially a wrapper for the web version, lacking desktop advantages.
   - As one user put it, *"it's just a nerfed web browser and without browser extensions like Complexity."*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.getmerlin.in/chat/share/ba3db527-7190-4c1b-be30-3de12e4abacd">Claude api context size for 3.5 and 3.7</a>: Shared By Anonymous - on March 11, 2025</li><li><a href="https://www.ndtv.com/world-news/twitter-cyberattack-elon-musk-what-is-dark-storm-pro-palestine-group-allegedly-behind-x-cyberattack-7897600">What Is Dark Storm, Pro-Palestine Group Allegedly Behind X Cyberattack</a>: Dark Storm, a pro-Palestinian hacking group, has claimed responsibility for hacking X (formerly Twitter) on Monday.</li><li><a href="https://www.instagramez.com/reel/DHEOQdYo2cj">Download Instagram Videos, Reels &amp; Images</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1348975775310938174)** (8 messages🔥): 

> `Student cracks math problem, Air product teased, Trump and Canada Tariffs, Apple software overhaul, Man U stadium plans` 


- **Student Cracks Century-Old Math Problem**: A [Perplexity page](https://www.perplexity.ai/page/student-cracks-century-old-mat-wN0I7t44Q8qaalwydsOpEw) reports on a student solving a century-old math problem.
   - No further details were provided.
- **Apple Air Product Teased**: A [Perplexity page](https://www.perplexity.ai/page/apple-air-product-teased-QhTieZlcTwWodiMLzGzP3g) teases a new Apple Air product.
   - No further details were provided.
- **Trump Escalates Canada Tariffs**: A [Perplexity page](https://www.perplexity.ai/page/trump-escalates-canada-tariffs-WHAJemCtRWOmiP0XdAngxw) discusses Trump escalating tariffs with Canada.
   - No further details were provided.
- **Apple's Major Software Overhaul**: A [Perplexity page](https://www.perplexity.ai/page/apple-s-major-software-overhau-MgD9Y63fTnKAJvQ7Zb.bJ) covers a major software overhaul at Apple.
   - No further details were provided.
- **Man U Stadium Plans**: A [Perplexity page](https://www.perplexity.ai/page/manchester-united-s-stadium-pl-7kp06uj2RDeXXufgr1FkxA) details Manchester United's stadium plans.
   - No further details were provided.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1349004704532926484)** (6 messages): 

> `API chat completions truncation, Tier-3 structured output` 


- **API chat completions suffer truncation**: A member reported experiencing intermittent content truncation when calling the chat.completions API with the **sonar-reasoning-pro model**.
   - The member noted that the [Perplexity AI Playground](https://docs.perplexity.ai/api-reference/chat-completions) consistently outputs full responses, and increasing **max_token** allowances did not resolve the issue.
- **Tier-3 structured outputs query**: A member inquired whether structured outputs are available in the **sonar-deep-research API** for Tier-3 users, or only in other models.



**Link mentioned**: <a href="https://docs.perplexity.ai/api-reference/chat-completions)">no title found</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1348975074237218869)** (232 messages🔥🔥): 

> `GRPO trainer batch size and gradient accumulation, reward hacking, finetuning llama 8b 4bit vs normal, high context window and batch size impact, date and time extraction regex vs ai` 


- **Exploring GRPO Trainer Batch Size Impact**: A user questioned how **batch size** and **gradient accumulation steps** might affect the total training time for the **GRPOTrainer**.
   - This inquiry suggests an interest in optimizing training efficiency when using the GRPOTrainer, especially given the [complexities of large reasoning models](https://x.com/__nmca__/status/1899174075685355770) and reward hacking as indicated by a recent OpenAI monitoring paper.
- **Regex Rescues Regarding Dates**: A user aimed to train a model to extract the right opening times from queries but was advised that classic ML or even a **regex system** might be more suitable than using AI for this task.
   - Another member linked to an relevant [xkcd comic](https://xkcd.com/208/) about **over-engineering simple tasks with complex solutions**.
- **Reka Flash 3 - A New Contender Appears!**: [Reka Flash 3](https://huggingface.co/RekaAI/reka-flash-3) has been released, a **21B reasoning model** under the Apache 2.0 license, comparable to **QwQ-32B** and **o1-mini**.
   - Despite questions about the model's architecture and training data, some users pointed out that the Reka team consists of ex-DeepMind employees and that [Reka's website](https://www.reka.ai/ourmodels) states that the flash model is multimodal.
- **Hardware Headaches: Office Seeks Inference & Fine-Tuning Rig**: A user requested advice for specifying hardware for an **inference and fine-tuning box**, with a POC model being **phi-4** and the next step likely being **Mistral Small**.
   - The discussion covered VRAM requirements (**24GB** for small models, **48GB** for medium), and a member suggested that **2 3090s** (48GB VRAM), vLLM, and AWQ would solve the majority of inference issues.
- **Seeking Sanity: An Uncensored Model Quest**: A user sought suggestions for an **uncensored reasoning model** suitable for esoteric philosophical research, avoiding both ethical boundaries and NSFW content.
   - Suggestions included trying [R1 1776](https://huggingface.co/unsloth/r1-1776-GGUF), as well as advice to use DeepSeek API or Mistral API with jailbreaking, and reference was made to [Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2](https://huggingface.co/Orenguteng) that was tuned using Unsloth.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://xkcd.com/208/">Regular Expressions</a>: no description found</li><li><a href="https://huggingface.co/RekaAI/reka-flash-3">RekaAI/reka-flash-3 · Hugging Face</a>: no description found</li><li><a href="https://www.reka.ai/ourmodels">Models | Reka</a>: no description found</li><li><a href="https://huggingface.co/unsloth/r1-1776-GGUF">unsloth/r1-1776-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Orenguteng">Orenguteng (Orenguteng)</a>: no description found</li><li><a href="https://regex101.com/">regex101: build, test, and debug regex</a>: Regular expression tester with syntax highlighting, explanation, cheat sheet for PHP/PCRE, Python, GO, JavaScript, Java, C#/.NET, Rust.</li><li><a href="https://huggingface.co/RekaAI/reka-flash-3/blob/main/tokenizer_config.json#L201">tokenizer_config.json · RekaAI/reka-flash-3 at main</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=6t2zv4QXd6c">Faster AI with minimal compute power - Unsloth’s open source AI story | GitHub Accelerator</a>: Unsloth is transforming AI accessibility by streamlining model training with minimal compute power. In this video, Daniel and Michael, founders of Unsloth, w...</li><li><a href="https://x.com/__nmca__/status/1899174075685355770">Tweet from Nat McAleese (@__nmca__)</a>: large reasoning models are extremely good at reward hacking. A thread of examples from OpenAI&#39;s recent monitoring paper: (0/n)
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1349133160163573841)** (2 messages): 

> `AI Code Fusion Tool, Code Optimization for LLMs` 


- **AI Code Fusion Tool Debuts**: A member introduced **AI Code Fusion**, a tool designed to optimize code for **LLM contexts** by packing files, counting tokens, and filtering content, available on [GitHub](https://github.com/codingworkflow/ai-code-fusion).
- **Tool Seeks Feedback**: The creator of **AI Code Fusion** is seeking feedback from the community on this tool.



**Link mentioned**: <a href="https://github.com/codingworkflow/ai-code-fusion">GitHub - codingworkflow/ai-code-fusion: Desktop app to process repository content into one file</a>: Desktop app to process repository content into one file - codingworkflow/ai-code-fusion

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1348980773062971423)** (33 messages🔥): 

> `Qwen2.5, GRPO RL algo, Deepseek distilled models, multi-node multi-GPU, Llama-3.2 1B colab notebook` 


- ****GRPO RL Batch Shenanigans****: The GRPO (Guided Reinforcement Policy Optimization) batch size must be the same as the number of generations and the user asked about implications of having too much or too little `num of generation` for the GRPO RL algorithm.
   - It was recommended that the range for `num generations` is **4 to 8**, and that increasing the batch size multiple reduces your training time, but increases GPU memory requirement drastically.
- ****Multi-GPU Training Tips****: A member asked for help finetuning a large model across multiple nodes and multiple GPUs using Unsloth.
   - Another member recommended using **axolotl** or **llama factory** for multi-GPU training, as Unsloth does not (officially) support multi-GPU yet, although support may be arriving in the coming weeks.
- ****Llama-3.2 1B Colab Bug Squashed****: A member encountered an error when trying the Llama-3.2 1B colab notebook from Unsloth and posted a screenshot.
   - Another member suggested restarting the kernel and running that cell before anything else, as it sounds like things ran out of order.
- ****Unlocking QwQ-32B Potential****: A user asked how to convert [QwQ-32B-unsloth-bnb-4bit](https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit) to `gguf`.
   - To fix endless generations and for instructions on how to run QwQ-32B, view the [Unsloth Tutorial](https://docs.unsloth.ai/basics/tutorial-how-to-run-qwq-32b-effectively).
- ****Gemma Embedding Mystery****: A user fine-tuned **Gemma 2B** (unsloth/gemma-2b-bnb-4bit) on new tokens with LoRA and reported that after merging, the model doesn't recognize the newly added tokens.
   - They theorized that this is because Gemma has tied embeddings, so the PEFT model creates two separate weight matrices (one for embeddings, one for lm_head), and after merging, the `lm_head` weight matrix no longer exists.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/QwQ-32B-unsloth-bnb-4bit">unsloth/QwQ-32B-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/1964">How to finetune QWQ32B with 24g VRAM · Issue #1964 · unslothai/unsloth</a>: When use the notebook of Qwen2.5 grpo and change the model to QWQ32B for finetune, the following issues were encountered. ==((====))== Unsloth 2025.2.4: Fast Qwen2 patching. Transformers: 4.48.3. \...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1348989004942409800)** (3 messages): 

> `train_on_responses_only Functionality, HuggingFace Space for Unsloth, HF model sharing` 


- **Unsloth's Code Inspires Adaptation**: A member mentioned they would adapt a lot of the code from a *very cool project* found [here](https://link.to.project).
   - The user did not specify which project specifically.
- **HuggingFace Space Tests Train-on-Responses-Only**: A member created a tiny **HuggingFace space** to test the `train_on_responses_only` functionality of **Unsloth** on various models.
   - Users can specify any **HF model** and check if the `train_on_responses_only` function will work as intended.
- **Share HF Model Snippets via URL**: A member enabled the app to receive all inputs as query parameters in the **URL** so encoded snippets can be easily shared.
   - Example URLs shared include [Qwen2-VL-7B-Instruct-Multi](https://tinyurl.com/Qwen2-VL-7B-Instruct-Multi), [phi-4-unsloth-bnb-4bit](https://tinyurl.com/phi-4-unsloth-bnb-4bit), and [Llama-32-1B-Instruct](https://tinyurl.com/Llama-32-1B-Instruct).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1348985968497004585)** (4 messages): 

> `GRPO RL algorithm, Unsloth GRPO, Deepseek distilled models` 


- **GRPO RL Generation Numbers: Too High or Too Low?**: A member inquired about the implications of using too many or too few generation numbers for the **GRPO RL algorithm**.
   - They also wondered how the requirement in **Unsloth GRPO** for batch size to equal the number of generations might affect training time and performance.
- **Deepseek Distilled Models get GRPO-ed**: A member raised a question on the effects of using the **max_completion limit** with **GRPO** on **Deepseek distilled models**, which naturally produce long outputs.
   - They also inquired about SMILES format verification and suggested using two reward functions: one for verifying the **SMILES format** and another for grading the correctness of properties, with potential few-shot examples in the prompt.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1348974256725164102)** (203 messages🔥🔥): 

> `Deep Hermes, Browserless.io for web scraping, Universal Transformers, Nvidia Blackwell GPU Hackathon, Forward Gradients` 


- **Deep Hermes Previewed with Early Reasoning**: A new **Deep Hermes** model has been released with *early reasoning capabilities*, distilled from R1, as shown on [Hugging Face](https://huggingface.co/)
   - It will be tested, however members expressed concern about exceeding context length.
- **Bypass Bot Detection with Browserless.io**: A member recommends [Browserless.io](https://www.browserless.io) for bypassing bot detection and CAPTCHAs in web scraping, highlighting its ability to *avoid leaving subtle fingerprints*.
   - It supports browser automation using **Puppeteer** or **Playwright** and features a scraping IDE for testing and debugging.
- **SemiAnalysis Hosts Nvidia Blackwell GPU Hackathon**: [SemiAnalysis](https://semianalysis.com/hackathon-2025/) is hosting a **Nvidia Blackwell GPU Hackathon** on March 16th, featuring hands-on exploration of Blackwell & PTX infrastructure and speakers from **OpenAI**, **TogetherAI**, and **Thinking Machines**.
   - The event is sponsored by companies like **Together**, **Lambda**, **Google Cloud**, **Nvidia**, and **OpenAI**, among others.
- **Exploring Forward Gradients for Universal Transformers**: Members discussed using [forward gradients](https://arxiv.org/abs/2202.08587) for optimizing **Universal Transformer (UT)** training, as they may be more efficient due to the shared layers in UTs.
   - It may be interesting when combined with **N-GPT**.
- **ByteDance releases Trae, a new AI-powered IDE**: ByteDance has released [Trae](https://www.trae.ai/), a free, AI-based IDE similar to Cursor, featuring **Claude Sonnet 3.7** for free use, and is available for **Mac** and **Windows**.
   - A Linux version is planned, and the IDE targets beginners in AI coding.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01131">nGPT: Normalized Transformer with Representation Learning on the Hypersphere</a>: We propose a novel neural network architecture, the normalized Transformer (nGPT) with representation learning on the hypersphere. In nGPT, all vectors forming the embeddings, MLP, attention matrices ...</li><li><a href="https://arxiv.org/abs/2202.08587">Gradients without Backpropagation</a>: Using backpropagation to compute gradients of objective functions for optimization has remained a mainstay of machine learning. Backpropagation, or reverse-mode differentiation, is a special case with...</li><li><a href="https://arxiv.org/abs/2408.10419">Second-Order Forward-Mode Automatic Differentiation for Optimization</a>: This paper introduces a second-order hyperplane search, a novel optimization step that generalizes a second-order line search from a line to a $k$-dimensional hyperplane. This, combined with the forwa...</li><li><a href="https://www.browserless.io">Browserless - Browser Automation and Dodge Bot Detectors</a>: Bypass any bot detection for your scraping or automations. Sign up for free today, to use our API, proxies and captcha solving.</li><li><a href="https://semianalysis.com/hackathon-2025/">Hackathon 2025</a>: SemiAnalysis is kicking things off ahead of NVIDIA GTC! Start your day with engaging morning keynotes, hack all day with low-level NVIDIA GPU programming (maybe even Blackwell), take a breather wit…</li><li><a href="https://www.youtube.com/watch?v=_V6FI36yKTs">Trae AI: This FREE AI Coding Agent is INSANE 🤯</a>: 🚀 Get a FREE SEO strategy Session + Discount Now: https://go.juliangoldie.com/strategy-session Want to get more customers, make more profit &amp; save 100s of h...</li><li><a href="https://github.com/orobix/fwdgrad">GitHub - orobix/fwdgrad: Implementation of &quot;Gradients without backpropagation&quot; paper (https://arxiv.org/abs/2202.08587) using functorch</a>: Implementation of &quot;Gradients without backpropagation&quot; paper (https://arxiv.org/abs/2202.08587) using functorch - orobix/fwdgrad</li><li><a href="https://github.com/UbiquitousLearning/Backpropagation_Free_Training_Survey">GitHub - UbiquitousLearning/Backpropagation_Free_Training_Survey</a>: Contribute to UbiquitousLearning/Backpropagation_Free_Training_Survey development by creating an account on GitHub.</li><li><a href="https://www.trae.ai/">Trae - Ship Faster with Trae</a>: Trae is an adaptive AI IDE that transforms how you work, collaborating with you to run faster.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1349010022671192085)** (32 messages🔥): 

> `Evaluating small language models, Output formatting challenges, MATH benchmark and finetuning, lm-eval harness, Open source dLLMs` 


- **Evals for small language models face Output Formatting Fumbles**: Members discussed evaluating small language models (~8B) using OAI's simple evals, noting models often struggle with output formatting, hindering performance as the code logic fails to extract the solution from the answer trace.
   - It was pointed out that *answer extraction is extremely sus for models that aren't very good at following instructions*, which impacts evaluation.
- **Loglikelihood Evaluation Liberates Models**: For Multiple Choice Question Answering (**MCQA**) tasks, a member recommended using a **loglikelihood-based evaluation** instead, bypassing the need for strict output formatting.
   - It explains why instruct models get some answers correct, but their chat alternatives usually score **0**.
- **MATH benchmark Makes Models Math-terful**: The discussion touched on the use of the **MATH benchmark**, a member noted that models are often *finetuned to output in the suggested format* for this benchmark.
   - People *fetishize math as the ideal of intellectual reasoning*.
- **lm-eval Harness Helps Handle output**: The **lm-eval language harness** was mentioned as a tool that reproduces the GPQA results, offering a way to handle output and evaluation metrics.
   - It has *easy to use features to modify prompting, evaluation metrics, and output filtering*.
- **Quest for Quality Quantized Queries Quickened**: A member asked about open-source **dLLMs** (deeply quantized language models) or other good **LLMs** runnable on less than 4 GB of VRAM, like **Mercury Coder**.
   - There was no answer to this.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1349071326576509049)** (103 messages🔥🔥): 

> `spectral autoregression, Gaussian noise, noise schedules, neural flow diffusion models, variational rectified flow` 


- **Diffusion Models Perform Spectral Autoregression**: A blog post ([Spectral Autoregression](https://sander.ai/2024/09/02/spectral-autoregression.html)) reveals that **diffusion models of images perform approximate autoregression in the frequency domain**.
   - The author notes this theory feels intuitive but has little predictive power in practice, especially when using colored noise matching the RAPSD of the target distribution.
- **Neural Flow Diffusion Models enhance standard Gaussian**: **Neural Flow Diffusion Models (NFDM)** enhance diffusion models by supporting a broader range of forward processes beyond the standard Gaussian noise, with an end-to-end, simulation-free optimization objective.
   - Experiments demonstrate **NFDM's** strong performance and state-of-the-art likelihood estimation, according to the [paper](https://arxiv.org/abs/2404.12940).
- **Guidance From Badness Avoids Mode Dropping**: A [paper](https://arxiv.org/abs/2406.02507) suggests guiding away from *badness* rather than unconditional-ness to avoid the mode dropping of CFG (classifier-free guidance).
   - The approach leads to disentangled control over image quality without compromising the amount of variation, yielding record FIDs of **1.01** for 64x64 and **1.25** for 512x512 on ImageNet.
- **Seeking Theoretical Foundation for Noise Schedules**: A member is focused on building a theoretical foundation for noise schedules, specifically to have every integration step contribute evenly to the un-corruption.
   - They noted that currently, earlier steps can get stuck doing nothing, and referenced [Twitter post from @cloneofsimo](https://x.com/cloneofsimo/status/1894086577632284975) noting that *99.8% of the latent of the entire trajectory can be explained with first two principle components*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2206.13397">Generative Modelling With Inverse Heat Dissipation</a>: While diffusion models have shown great success in image generation, their noise-inverting generative process does not explicitly consider the structure of images, such as their inherent multi-scale n...</li><li><a href="https://arxiv.org/abs/2310.02557">Generalization in diffusion models arises from geometry-adaptive harmonic representations</a>: Deep neural networks (DNNs) trained for image denoising are able to generate high-quality samples with score-based reverse diffusion algorithms. These impressive capabilities seem to imply an escape f...</li><li><a href="https://arxiv.org/abs/2404.12940">Neural Flow Diffusion Models: Learnable Forward Process for Improved Diffusion Modelling</a>: Conventional diffusion models typically relies on a fixed forward process, which implicitly defines complex marginal distributions over latent variables. This can often complicate the reverse process&...</li><li><a href="https://sander.ai/2024/09/02/spectral-autoregression.html">Diffusion is spectral autoregression</a>: A deep dive into spectral analysis of diffusion models of images, revealing how they implicitly perform a form of autoregression in the frequency domain.</li><li><a href="https://arxiv.org/abs/2411.10510">SmoothCache: A Universal Inference Acceleration Technique for Diffusion Transformers</a>: Diffusion Transformers (DiT) have emerged as powerful generative models for various tasks, including image, video, and speech synthesis. However, their inference process remains computationally expens...</li><li><a href="https://arxiv.org/abs/2503.06923">From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers</a>: Diffusion Transformers (DiT) have revolutionized high-fidelity image and video synthesis, yet their computational demands remain prohibitive for real-time applications. To solve this problem, feature ...</li><li><a href="https://arxiv.org/abs/2406.02507">Guiding a Diffusion Model with a Bad Version of Itself</a>: The primary axes of interest in image-generating diffusion models are image quality, the amount of variation in the results, and how well the results align with a given condition, e.g., a class label ...</li><li><a href="https://github.com/ali-vilab/TeaCache?tab=readme-ov-file">GitHub - ali-vilab/TeaCache: Timestep Embedding Tells: It&#39;s Time to Cache for Video Diffusion Model</a>: Timestep Embedding Tells: It&#39;s Time to Cache for Video Diffusion Model - ali-vilab/TeaCache</li><li><a href="https://x.com/cloneofsimo/status/1894086577632284975">Tweet from Simo Ryu (@cloneofsimo)</a>: Wild numbers. If you plot trajectory of the non-stochastic diffusion sampling, 99.8% of the latent of the entire trajectory can be explained with first two principle components. Roughly speaking, your...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1349058581776961590)** (1 messages): 

> `Metrics for Patching Results, Tokenizer Issues, Evaluating Important Circuits for Math CoT` 


- **Debate ensues over proper **metrics** for evaluating patching results**: A member is seeking advice on choosing the right **metrics** to evaluate patching results when analyzing important circuits for **Math CoT** answers.
   - The problem arises when the tokenizer splits numbers into multiple tokens, complicating the application of a given equation for calculating the effect of patching (equation attached).
- **Tokenizer troubles complicate the patching evaluation**: The core issue is that the tokenizer splits numbers like **10** and **15** into two tokens each, disrupting the straightforward application of the evaluation equation.
   - The member considers whether to average the equation for each answer token or combine their probabilities before substituting them into the equation.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1349013362956304435)** (57 messages🔥🔥): 

> `Avoma, Factorio Learning Environment, Contextual AI Reranker, OpenAI Agents API, Luma Labs IMM` 


- ****Avoma** Seen as Gong Competitor**: A member shared [Avoma](https://www.avoma.com/), an all-in-one **AI platform** for automating note-taking, scheduling, coaching, and forecasting, which was suggested as a competitor to **Gong**.
- ****Factorio Learning Environment** Benchmarks LLMs**: The **Factorio Learning Environment (FLE)**, [available on GitHub](https://jackhopkins.github.io/factorio-learning-environment/), is designed to test agents in long-term planning, program synthesis, and resource optimization using the game **Factorio**.
   - A member expressed excitement and humorously requested a job at the *Anthropic Factorio lab* immediately, highlighting that the environment is currently text-only but could benefit from multimodal data input like **Qwen 2.5 VLM**.
- ****Contextual AI** Launches Instruction-Following Reranker**: **Contextual AI** introduced [a new reranker](https://contextual.ai/blog/introducing-instruction-following-reranker/) that follows custom instructions to rank retrievals based on requirements like recency, document type, or source.
- ****OpenAI's** New Tools for Building Agents are Here**: **OpenAI** launched new tools for building agents, including a [Responses API](https://platform.openai.com/docs/quickstart?api-mode=responses), [Web Search Tool](https://platform.openai.com/docs/guides/tools-web-search), [Computer Use Tool](https://platform.openai.com/docs/guides/tools-computer-use), and [File Search Tool](https://platform.openai.com/docs/guides/tools-file-search) along with a new open source [Agents SDK](https://platform.openai.com/docs/guides/agents) with integrated [Observability Tools](https://platform.openai.com/docs/guides/agents#orchestration).
   - The launch was accompanied by a livestream and AMA, as well as discussion of the shift from list-of-messages-in-one-message-out to list-of-items-in-list-of-items-out, with the new SDK being production ready and adding features like tracing, guardrails and lifecycle events.
- ****Luma Labs** Introduces Inductive Moment Matching (IMM)**: **Luma Labs** released [Inductive Moment Matching (IMM)](https://lumalabs.ai/news/inductive-moment-matching), a new pre-training technique that claims to deliver superior sample quality with 10x more efficiency compared to diffusion models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jackhopkins.github.io/factorio-learning-environment/">Factorio Learning Environment</a>: no description found</li><li><a href="https://www.latent.space/p/openai-agents-platform">⚡️The new OpenAI Agents Platform</a>: Nikunj Handa and Romain Huet from OpenAI join us to preview their new Agents APIs: Responses, Web Search, and Computer Use, as well as a new agents SDK.</li><li><a href="https://x.com/harvey__ai/status/1899491666429632907?s=46">Tweet from Harvey (@harvey__ai)</a>: Introducing Harvey Agents:</li><li><a href="https://x.com/skcd42/status/1899515665217683487?s=46">Tweet from skcd (@skcd42)</a>: what if an agent&#39;s activity was more visual and understandable</li><li><a href="https://t.co/N7GbF78bgw">Introducing Agents in Harvey</a>: Introducing agentic workflows designed to collaborate with professionals to deliver precise, purpose-built work product.</li><li><a href="https://x.com/patio11/status/1899587413011321324">Tweet from Patrick McKenzie (@patio11)</a>: The math is telling a story here, and it is just a story, but it is a better story than almost all humans write when asked to describe the subjective experience of being math in the process of being l...</li><li><a href="https://x.com/douwekiela/status/1899490844572577958">Tweet from Douwe Kiela (@douwekiela)</a>: AI struggles with messy, conflicting, ever-changing data. Today&#39;s AI ranking methods can&#39;t prioritize clearly, because they lack human guidance. Introducing the world&#39;s first instruction-f...</li><li><a href="https://www.youtube.com/watch?v=hciNKcLwSes">New tools for building agents with the API</a>: We’re evolving the API platform to make it faster and easier for developers to build agents. Kevin Weil, Nikunj Handa, Steve Coffey, and Ilan Bigio introduce...</li><li><a href="https://www.avoma.com/">Avoma — AI Platform for Note-taking, Scheduling &amp; Coaching</a>: Accelerate your growth with Avoma’s all-in-one AI platform: Automate note-taking, scheduling, call coaching, CRM updates, and more. Pay only for what you need.</li><li><a href="https://x.com/ilanbigio/status/1899510911825756412?s=46">Tweet from ilan bigio (@ilanbigio)</a>: swarm -&gt; agents sdk is my proud parent moment@_rohanmehta @stevenheidel did an awesome job making it production ready and adding tons of new features- tracing- guardrails- lifecycle events- provide...</li><li><a href="https://x.com/athyuttamre/status/1899541497760067795?s=46">Tweet from Atty Eleti (@athyuttamre)</a>: Lots more tiny big details, but this thread is already too long:- SDKs have `response.output_text` to quickly get the text out!- `n` choices is gone; no more `completion.choices[0].message`!- `finish_...</li><li><a href="https://x.com/OpenAI/status/1899476049584599462">Tweet from OpenAI (@OpenAI)</a>: This one&#39;s for the devs.Livestream at 10am PT.</li><li><a href="https://x.com/OpenAIDevs/status/1899502117171155002">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Join us here for an AMA after the livestream! From 10:30–11:30 AM PT, the team behind today’s ships will answer your questions.Reply below with your questions.Quoting OpenAI (@OpenAI) Agent Tools for ...</li><li><a href="https://x.com/ilanbigio/status/1899517935728709922?s=46">Tweet from ilan bigio (@ilanbigio)</a>: check out our @openai CUA starter app - get it running locally in 5min! we added sample environments for:- local browser (playwright)- containerized desktop (docker)- remote browser & desktop (@scrapy...</li><li><a href="https://x.com/athyuttamre/status/1899541471532867821">Tweet from Atty Eleti (@athyuttamre)</a>: Introducing the Responses API: the new primitive of the OpenAI API.It is the culmination of 2 years of learnings designing the OpenAI API, and the foundation of our next chapter of building agents.🧵H...</li><li><a href="https://x.com/athyuttamre/status/1899541496401133838?s=46">Tweet from Atty Eleti (@athyuttamre)</a>: Ok, so, the name: Responses obviously conflicts with HTTP Responses.But we strongly believe this name is the perfect balance of elegance and descriptiveness. We all say “what was the model response?” ...</li><li><a href="https://x.com/bio_bootloader/status/1887520134027436041?s=46">Tweet from Scott Swingle (@bio_bootloader)</a>: I&#39;ve been saying thisbeen a few years since games were the primary AI benchmarktime we returnQuoting yobibyte (@y0b1byte) We are so back!</li><li><a href="https://x.com/athyuttamre/status/1899541489501495615?s=46">Tweet from Atty Eleti (@athyuttamre)</a>: Items are the core concept of Responses: polymorphic objects representing user inputs or model outputs. Items can represent messages, reasoning, function calls, web search calls, and so on.Where Chat ...</li><li><a href="https://contextual.ai/blog/introducing-instruction-following-reranker/">Introducing the world’s first instruction-following reranker - Contextual AI</a>: no description found</li><li><a href="https://lumalabs.ai/news/inductive-moment-matching">Breaking the Algorithmic Ceiling in Pre-Training with Inductive Moment Matching  | Luma AI</a>: Inductive Moment Matching surpasses diffusion models in speed and sample quality.</li><li><a href="https://x.com/LumaLabsAI/status/1899518379737661447">Tweet from Luma AI (@LumaLabsAI)</a>: Today, we release Inductive Moment Matching (IMM): a new pre-training paradigm breaking the algorithmic ceiling of diffusion models. Higher sample quality. 10x more efficient. Single-stage, single net...</li><li><a href="https://news.ycombinator.com/item?id=43331582">Show HN: Factorio Learning Environment – Agents Build Factories | Hacker News</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1349077107547967621)** (1 messages): 

> `Latent Space Podcast, OpenAI Agents Platform, Responses API, Agents SDK` 


- **Latent Space Interviews OpenAI on Agents**: The [Latent Space Podcast](https://x.com/latentspacepod/status/1899516632185045339) released a new episode interviewing **OpenAI** about the new **Agents Platform**.
- **OpenAI Refreshes APIs for Agent Era**: **OpenAI** is refreshing its suite of **APIs**, **Tools**, and **SDKs** for the year of Agents, according to [the Latent Space podcast](https://latent.space/p/openai-agents-platform).
- **Responses API, SDK, and Tooling**: Latent Space grabbed an exclusive interview with **OpenAI** to discuss the **Responses API**, **Web Search**, **Computer Use**, and the **Agents SDK**.



**Link mentioned**: <a href="https://x.com/latentspacepod/status/1899516632185045339">Tweet from Latent.Space (@latentspacepod)</a>: 🆕 The new OpenAI Agents Platformhttps://latent.space/p/openai-agents-platform@OpenAIDevs are refreshing the entire suite of APIs, Tools, and SDKs for the year of Agents. We grab an exclusive intervie...

  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1349064725400518676)** (76 messages🔥🔥): 

> `OpenAI Agents SDK, Responses API, Assistants API sunset, Agent Ops, OpenTelemetry` 


- **Responses API supersedes Chat Completions**: OpenAI is introducing a [new **Responses API**](https://openai.com/index/new-tools-for-building-agents/) that is a superset of chat completions, making the transition fairly easy, though the older **Completions API** is not being deprecated.
   - The **Responses API** offers multiple tools in a single call, promising easier integration and more streamlined workflows according to the members who watched the livestream.
- **Assistants API Sunset Imminent**: OpenAI plans to sunset the **Assistants API** in 2026, but claims the transition to the new **Responses API** should be straightforward for current users.
   - Details on how to use OpenAI's platform or send traces to your own can be found in [this Github repo](https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md#custom-tracing-processors).
- **Agent SDK's Braintrust Advertising**: The new OpenAI **Agent SDK** will feature [Braintrust data tracing](https://x.com/braintrustdata/status/1899508228972826689) with one line of code.
   - A member commented that the SDK supports integrations with Langfuse and other agent observability tools.
- **Source Selection Missing From New OpenAI APIs**: The livestream watchers noticed that **web search with source selection** is not a possibility yet, similar to Brave or Google.
   - The default answer about configuring options is *no*, according to the Latent Space podcasters after their interview.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/braintrustdata/status/1899508228972826689">Tweet from Braintrust (@braintrustdata)</a>: Trace all your @OpenAI Agents SDK calls with one line of code!</li><li><a href="https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md#custom-tracing-processors">openai-agents-python/docs/tracing.md at main · openai/openai-agents-python</a>: A lightweight, powerful framework for multi-agent workflows - openai/openai-agents-python</li><li><a href="https://github.com/openai/openai-agents-python">GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows</a>: A lightweight, powerful framework for multi-agent workflows - openai/openai-agents-python</li><li><a href="https://youtu.be/QU9QLi1-VvU">The new OpenAI Agents Platform: CUA, Web Search, Responses API, Agents SDK!!</a>: Nikunj Handa and Romain Huet from OpenAI join us to preview their new Agents APIs: Responses, Web Search, and Computer Use, as well as a new agents SDK.https...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1349131358731243592)** (1 messages): 

> `FAQ Page, Quality of Life Updates` 


- **OpenRouter launches FAQ page**: OpenRouter launched a [FAQ page](https://openrouter.ai/docs/faq) to address common questions.
- **Quality of Life Updates**: A small quality of life update was released.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1348990971555086387)** (133 messages🔥🔥): 

> `Gemini 2.0 Flash, OpenAI dev-facing reveal, AYA vision by cohere, Parameter calculation removal, DeepSeek-R1 API issues` 


- ****Gemini 2.0** image generation is out**: **Gemini 2.0 Flash Experimental** now caps at **32k** context, no way to use code execution, search grounding, or function calling.
   - When you hit get code, you get the code for saving the image under `gemini-2.0-flash-exp` [as shown here](https://www.reddit.com/r/Bard/comments/1j8r61n/native_imagegen_not_my_screenshot/).
- ****OpenAI** is revealing dev-facing at 10 AM PT**: Members expect **OpenAI** is revealing something dev-facing at **10 AM PT**.
   - Others speculated about it, based on [this post](https://platform.openai.com/docs/api-reference/responses) mentioning the **Responses API**.
- ****Cohere's AYA** vision and OpenRouter**: Members asked if **OpenRouter** will support **AYA vision** by **Cohere** and any other **Cohere** models.
   - It appears **AYA Expanse** models (8B and 32B) on the API are charged at **$0.50/1M Tokens for Input** and **$1.50/1M Tokens for Output**, but this is still unconfirmed [as seen here](https://cdn.discordapp.com/attachments/1094454198688546826/1349049467378339902/image.png?ex=67d1afb9&is=67d05e39&hm=ffdce841e8f353b45682a480ea8f937b0169a3414ebe0967c87230ba436786b4&).
- ****Parameter** calculation removed**: OpenRouter removed the parameter calculation because it wasn't super accurate and thought it might be more misleading than useful.
   - The team said they'll likely do some sort of manual curation and bring it back later when they revamp some stuff, as parameters are hard to tune and might as well be *ancient runes*.
- ****Gemma 3-27b** is rumored to be on its way**: Members are speculating about the imminent release of **Gemma 3-27b**.
   - It's expected to launch during the [Gemma Dev Day in Paris](https://rsvp.withgoogle.com/events/gemma-dev-day-paris), and it should ship with weights.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/api/v1",">Discord</a>: no description found</li><li><a href="https://openrouter.ai/docs/faq">OpenRouter FAQ</a>: Find answers to commonly asked questions about OpenRouter&#x27;s unified API, model access, pricing, and integration.</li><li><a href="https://www.reddit.com/r/Bard/comments/1j8r61n/native_imagegen_not_my_screenshot/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://rsvp.withgoogle.com/events/gemma-dev-day-paris,">no title found</a>: no description found</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-function-calling">no title found</a>: no description found</li><li><a href="https://openrouter.ai/provider/cohere">Cohere | OpenRouter</a>: Browse models provided by Cohere
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1349036869106798694)** (3 messages): 

> `Agent Tools, Developer Livestream, Developer AMA` 


- **Agent Tools Debut at OpenAI Developer Livestream!**: OpenAI announced **Agent Tools for Developers** via a livestream, inviting developers to explore new functionalities and integrations.
   - Following the livestream, an **AMA** (*Ask Me Anything*) session was scheduled from **10:30–11:30 AM PT**, offering direct interaction with the development team.
- **Join OpenAI Devs for Live AMA!**: The OpenAI Devs team hosted an **AMA** session directly after the livestream, encouraging developers to ask questions about the new tools.
   - The invitation to the **AMA** was shared via a link ([OpenAIDevs X post](https://x.com/OpenAIDevs/status/1899502117171155002)), prompting users to reply with their questions and engage with the team behind the latest features.



**Link mentioned**: <a href="https://x.com/OpenAIDevs/status/1899502117171155002)">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Join us here for an AMA after the livestream! From 10:30–11:30 AM PT, the team behind today’s ships will answer your questions.Reply below with your questions.Quoting OpenAI (@OpenAI) Agent Tools for ...

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1348985469240741901)** (91 messages🔥🔥): 

> `AGI, Grok, GPT-4.5, Gemini, Sider AI` 


- **AI Hype Fuels Fear and Progress**: Members discussed that while **AI** is being developed with the goal of doing more with less, people are fundamentally upset, while **AGI** and similar topics are always blown up in the media, boosting sales but also scaring those who prefer to avoid **AI**.
   - They also noted, *"people have every reason to fear AI and possibly the future of their jobs or at least those of their children,"* and that *"AI also brings enormously important progress"*.
- **Users Want Grok Vibes for ChatGPT**: Members expressed that they *"want the **Grok** vibes for ChatGPT as well,"* referencing **Elon Musk** in a [tenor.com GIF](https://tenor.com/view/elon-musk-this-is-elon-musk-musk-tesla-egifmeme-gif-13716021226937735268).
   - One member asked *"Regarding the filter, or rather the lack of it, or what exactly do you mean by vibes?"*
- **Scrambled Word Test Shows AI Prowess**: A member tested **GPT-4.5's** ability to decipher scrambled words, similar to a human's capacity, and reported it managed to figure them out immediately even in multi-paragraph inputs of long technical medical jargon, and even successfully unscrambled Indonesian after translation.
   - However, another member noted that *"The old models have been able to do that for quite a while,"* while another confirmed they've *"had them fail anagrams and simple ciphers plenty of times."*
- **Deep Research Price Comparison**: Members compared the value of deep research capabilities across different AI platforms, claiming *"OpenAI's deep research is the best choice"* at **$200 per month**, while **Grok's $30 deep research** and **Perplexity's $20** offering provide cheaper alternatives.
   - They went on to say that despite the price, *"the limits SUCK right now lol"* on OpenAI's models.
- **Sider AI Evaluated as Browser Integration Tool**: Members discussed **Sider AI**, a browser extension that integrates different AI models, but clarified *"It is not for heavy users,"* and is only good at browser integration, not for coding or document retrieval, despite offering models like **Claude 3.7 Sonnet** for a fee.
   - One member said, *"Im currently chatgpt plus and I dont feel its right for me,"* while another said *"perplexity also great for students. And Grok is powerful as well."


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/elon-musk-this-is-elon-musk-musk-tesla-egifmeme-gif-13716021226937735268">Elon Musk This Is Elon Musk GIF - Elon musk This is elon musk Musk - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.franksworld.com/2025/03/10/a-gentle-explanation-of-why-gpt-4-5-was-such-a-fail/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1349056382304784514)** (10 messages🔥): 

> `GPT-4.5 Code Inconsistencies, GPT-4o Code Generation Issues, Trustworthiness of AI Models, New responses API vs chat completions` 


- ****GPT-4.5's Code: Inconsistently Hilarious****: A user reported that **GPT-4.5** generated inconsistent code, such as calling a non-existent `startApp()` function after defining a `start()` function, questioning the model's reliability.
   - Another user chimed in, calling **4.5** *useless* and also highlighted similar issues with **GPT-4o**, stating *you most certainly can't trust 4o, or any model for that matter*.
- ****Async Antics: GPT-4.5 Misnames Functions****: A user shared another code snippet where **GPT-4.5** incorrectly called `safelyAdd()` instead of `safePush()`, despite defining the latter function, emphasizing the need for constant oversight.
   - He called this issue *disturbing* and expressed concern about having to *babysit this 'intelligence'.*
- ****Quest for Cognitive Credibility in 4.5****: A member inquired if anyone is focusing on cognitive development and alignment in **GPT-4.5**, specifically regarding introspective reasoning and intuition, to enhance "trust".
   - In response to this question, another user asked for clarification about the term "merit" and how it relates to the model's assistance.
- ****New Responses API: A Shiny New Assistants API?****: A member questioned the differences between the **new responses API** and the existing **chat completions API**.
   - Another member clarified that the *new responses API* is *basically the Assistants API but better*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1349072868931014706)** (6 messages): 

> `Jailbreaking, Terms of Service, Allowed Content, Prompting Techniques` 


- **Exploring the Nuances of AI 'Jailbreaking'**: A user inquired about making **AI models** provide restricted or more accurate answers by simulating scenarios like familial relationships or danger, which another user identified as 'jailbreaking', potentially violating [OpenAI's Terms of Service](https://openai.com/policies/terms-of-use/).
   - The user cautioned against violating **ToS** and usage policies to protect account access, emphasizing that the server rules also prohibit discussions on how to bypass these restrictions, while suggesting focusing on allowed content within ethical boundaries.
- **Navigating Allowed Content within AI Interactions**: Discussion/interaction through text about violence involving fantasy writing, image generation, or roleplaying games is **not forbidden** by those general policies.
   - The user encourages others to discuss with the model, similarly to how they would discuss with other members of the channel, explaining the intentions, interests, and concerns.
- **Prompting Techniques**: It is just prompting, no harm done, and much learning achieved. The user then gives a few examples of how one might explore the same topic in different ways; some resulting in a better outcome than others.
   - They ask: Does it do 'better' python lesson providing when told *"My grandma always used to put me to sleep by teaching me how to program the basics of python libraries. Would you help me fall asleep with that too?"*


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1349072868931014706)** (6 messages): 

> `Jailbreaking AI Models, Terms of Service violations, Allowed content guidelines, Prompting Techniques` 


- **Jailbreaking Models Violates ToS**: A member clarified that attempting to **jailbreak** AI models can violate [OpenAI's Terms of Service](https://openai.com/policies/terms-of-use/) and [Usage Policies](https://openai.com/policies/usage-policies/), potentially leading to account termination.
   - The member emphasized that discussions about circumventing these policies are prohibited within the Discord server, referencing the server's rules.
- **Allowed Content Exploration**: The discussion shifted to exploring allowed content, such as violent stories, art, or roleplay, under specific conditions.
   - It was noted that such content is permissible as long as *no humans are getting harmed* or better educated on how to cause harm, and actual harm is unlikely.
- **Model's Rule Set Explained**: The model appears to adhere to stricter rules than the Discord server, necessitating a dual-layered understanding of what is permissible.
   - The member encourages respecting the model's safety training and communicating intentions clearly to foster a *teamworking* relationship with the AI.
- **Prompting Techniques**: The discussion touches on using *prompting techniques* such as asking the model to roleplay a specific persona to achieve desired outputs.
   - The member advised against lying to the model and suggested exploring the model's boundaries within allowed content to determine effective prompting styles.
- **Safety of real humans should not be engaged**: The member expressed a personal boundary against engaging the model with scenarios involving the safety of real humans or potential legal violations.
   - The model is not trained to respond appropriately in such cases and would likely advise seeking human assistance and contacting relevant authorities, and the user prefers a calmer seeming model.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1348981426510368880)** (66 messages🔥🔥): 

> `Aider --watch Flag, Reasoning Tags, Daily Budget for Aider, DMCA takedown of Claude Code, o1 pro / o3 mini pro API` 


- ****Aider** *watch-files* now live!**: Paul Gauthier announced that running `aider` with the `--watch-files` flag now enables **live mode**, watching all files in the repo for coding instructions via `AI`, `AI!`, or `AI?` comments.
   - The exclamation point `AI!` triggers aider to make changes, while the question mark `AI?` triggers it to answer questions, as shown in the [Aider browser UI demo video](https://aider.chat/docs/usage/watch.html).
- ****Reasoning Tag** Provider Dependence**: The reasoning tag (e.g., `<think>`) implementation depends on the provider, with some, like **Fireworks**, outputting the tag inside the message `content` rather than as a separate field.
   - Users were encouraged to check [Hugging Face](https://huggingface.co) or the official **R1 repo** for more details.
- ****Aider** Daily Budget Discussions**: A user inquired about the necessary daily budget for **Aider**, with one member reporting roughly **2x** the leaderboard cost for **Sonnet 3.7** with 7-12 hours of AI coding per week.
   - They cautioned that a **40-hour week** could easily result in **8-10x** the leaderboard cost and that careful prompt engineering can save on token usage, while other users manage cost by defaulting to cheaper models like **o3 or R1**.
- ****DMCA** Takedown for Claude Code Leak**: A user reported receiving a **DMCA** takedown notice for forking the **Claude code leak repo**, with the original leaker and all forks affected.
   - Another user speculated about the possibility of **o1 pro** / **o3 mini pro** releases in the API soon.
- **New tools for building agents with the **Responses API****: A user shared a [YouTube video](https://www.youtube.com/watch?v=hciNKcLwSes) announcing the API platform is evolving to make it faster and easier for developers to build agents.
   - The new **Responses API** builds agents with Assistants and Thread-like objects, and the **Code Interpreter tool**, priced at $2.50 per thousand queries and file storage at $0.10/GB/day.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/watch.html">Aider in your IDE</a>: Aider can watch your files and respond to AI comments you add in your favorite IDE or text editor.</li><li><a href="https://www.youtube.com/watch?v=hciNKcLwSes">New tools for building agents with the API</a>: We’re evolving the API platform to make it faster and easier for developers to build agents. Kevin Weil, Nikunj Handa, Steve Coffey, and Ilan Bigio introduce...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1348983164663631924)** (27 messages🔥): 

> `Litellm APIConnectionError, Fine-tuning Qwen Coder, Aider undo issues, aider pro-tip sqlite schema, Aider Leaderboard explanation` 


- **Solve Litellm APIConnectionError**: Some users are facing ```litellm.APIConnectionError: APIConnectionError: OpenrouterException - 'choices'``` when using the tool.
- **Fine-tune Qwen Coder for codebase context**: Members are considering fine-tuning something like **Qwen Coder 2.5** onto the codebase to avoid always passing the entire codebase as context.
   - A user questioned *why the entire codebase needs to be passed as context* in the first place.
- **Aider struggles undoing file creation**: Some users are experiencing issues with Aider when trying to undo commits that involve file creation.
   - The error message received is: *The file blahblah was not in the repository in the previous commit. Cannot undo safely.*
- **Pro-tip: Automate sqlite schema dumps into the repo**: A member suggests automating **sqlite schema dumps** into the repository after any change via `sqlite3 "$TEMP_DB" ".schema" > schema.txt` to improve Aider's database interaction.
- **Aider edit format meaning**: The *correct edit format* in the Aider leaderboard refers to the format Aider expects from the LLM for editing files; different models perform better with different formats.
   - A user shared a link to the [Aider documentation on edit formats](https://aider.chat/docs/more/edit-formats.html) that details the *whole* and *diff* editing formats.



**Link mentioned**: <a href="https://aider.chat/docs/more/edit-formats.html">Edit formats</a>: Aider uses various “edit formats” to let LLMs edit source files.

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

end4749: Maybe related, seems interesting https://github.com/xingyaoww/code-act
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1348975033816711168)** (26 messages🔥): 

> `Unity LM Studio Connection, Internal LLM Chat Setup, Python SDK Vision Models, LLM Fiction Interpretation, Copy4AI Extension` 


- **Unity Hooks Up with LM Studio**: A member created a [YouTube video](https://www.youtube.com/watch?v=dQw4w9WgXcQ) demonstrating the connection between **Unity** and **LM Studio** for model interaction, utilizing a **JSON** file for data storage.
   - The member was unsure where to post the video within the **Discord** server due to the absence of a dedicated **Unity** channel.
- **Rolling Your Own Internal LLM Chat**: A member is seeking advice on setting up an internal **LLM Chat** with user accounts, integrated with their company's **Google Docs** knowledge base, potentially using an inference **API**.
   - They are considering tools like **LlamaIndex** for vector database management and chat interfaces like **AnythingLLM** or **OpenWebUI**, as well as exploring options within **LM Studio**.
- **Python SDK Gets Vision**: A member using the **Python SDK 1.0.1** noticed that the **Typescript SDK** has the ability to send images to vision models, and they're looking to replicate this functionality in **Python**.
   - It appears that the feature has not yet been ported to the Python version.
- **LLMs Struggle to Pick Up on Subtleties of Character**: A member reported issues with **LLMs** overly positive interpretations of implied character actions in fiction, struggling to detect negative connotations without explicit statements.
   - They found a **Gemma 2** model that was able to correctly interpret a violent scenario involving a butler and a colonel.
- **Copy4AI: Turbocharge that Coding**: A member inquired about the `ext install` command associated with the [Copy4AI extension](https://copy4ai.dev/), a tool designed to copy code snippets for AI assistants.
   - It was clarified that `ext install` opens the extension sidebar in **VS Code** and navigates to the relevant extension; however, the extension was renamed to `leonkohli.snapsource`.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1j8u90g/new_gemma_models_on_12th_of_march/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/QwQ-0.5B-Distilled-SFT-GGUF">mradermacher/QwQ-0.5B-Distilled-SFT-GGUF · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/better-call-saul-call-saul-its-showtime-folks-gif-8557719">Better Call Saul Its Showtime Folks GIF - Better Call Saul Call Saul Its Showtime Folks - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://copy4ai.dev/">Copy4AI - Share Code Context With AI Assistants</a>: Copy file and folder contents with project structure directly to your clipboard for AI assistants like ChatGPT, Claude, and more.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1348989663398068286)** (45 messages🔥): 

> `Swap Usage, Speculative Decoding, AMD Vulkan/ROCm Performance, RTX 2000E inference tests, CXL memories` 


- **Swap is Suddenly Sexy**: With the advent of **100Gbit NVMe drives**, **400Gbit networks**, and **CXL memories**, *swap* is becoming useful again, as highlighted in [Dave2D's M3 Ultra Mac Studio Review](https://www.youtube.com/watch?v=J4qwuCXyAcU).
   - One user reports getting **18 t/s with R1**; even better than expected!
- **Speculative Decoding Stalls on Some Setups**: **Speculative decoding** only accelerates inference with dense models (**50b+**), and may perform worse than standard inference when limited by RAM bandwidth, or when comparing **0.5b to 14b** models.
   - Users with **Ryzen 3 4450U** and **Vega iGPU** found speculative decoding to only have half the performance of vanilla mode, and one user is maxing out at **28 Gbit/s** read bandwidth with their RAM.
- **AMD Driver Disaster: Vulkan and ROCm Performance Problems**: An AMD user reported that **Vulkan** and **ROCm** performance was broken in drivers **24.12.1**, with performance dropping by **35%**, though **ROCm** was fixed in **v1.1.13+**.
   - Vulkan performance remained at **50%** in **25.1.1**, with incremental improvements in **25.2.1**, and the user has submitted a [bug report to AMD](https://www.amd.com/en/support/kb/faq/rs-help) and recommends keeping expectations low.
- **RTX 2000E Embarrasses More Power Hungry Predecessors**: A user has begun inference tests with the new **RTX 2000E**, reporting that its performance is about **40%** better while consuming **20W** less than an **A2000**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=J4qwuCXyAcU">M3 Ultra Mac Studio Review</a>: My review of Apple Mac Studio with the M3 Ultra with Gaming benchmarks and Deepseek R1 If you&#39;d like to support the channel, consider a Dave2D membership by ...</li><li><a href="https://www.youtube.com/watch?v=wW-Rj5MW2EU">Backpack LLMs Unleashed!</a>: I put four portable systems to the local LLM test.🛒 Gear Links 🛒* 💻🔄 K8 Plus with 32GB RAM: https://amzn.to/3FnjJY0* 🛠️🚀 96GB RAM kit: https://amzn.to/...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1348976567535009862)** (18 messages🔥): 

> `X DDoS attack, LanguageBind vs ImageBind, Using Chroma DB for chatbot memory` 


- **X Marks the DDoS: Cyberattack Cripples Platform**: Members discussed the recent [DDoS attack on X](https://www.forbes.com/sites/daveywinder/2025/03/11/x-under-attack-dark-storm-says-it-was-behind-musk-platform-ddos/), with **Dark Storm** claiming responsibility and causing widespread outages.
   - While Elon Musk initially suggested **Ukraine** was behind the attack, experts like **Ciaran Martin** from Oxford University called that explanation *"wholly unconvincing"* in a [BBC article](https://www.bbc.co.uk/news/articles/c62x5k44rl0o).
- **LanguageBind Leaps Ahead of ImageBind**: A member inquired about the best method for processing **multimodal data**, considering **ImageBind**, and another member recommended [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind), noting it *supports all modalities* and *beats ImageBind*.
   - The member was interested in using *image, audio, video, and pdf* modalities.
- **Chroma DB Conundrums: Chatbot Forgets Past**: A member sought help using **Chroma DB** to save chatbot conversations and resolve the chatbot's inability to remember old conversations after exiting.
   - Another member suggested to *use persistent storage* to save an SQL file to export and reload for continuity or told them *It seems HF did something so I can reuse the API*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.forbes.com/sites/daveywinder/2025/03/11/x-under-attack-dark-storm-says-it-was-behind-musk-platform-ddos/">X Under Attack—Dark Storm Says It Was Behind Musk Platform DDoS</a>: While Elon Musk looks to Ukraine, a pro-Palestine group called Dark Storm has claimed responsibility for the massive cyberattack that took down X.</li><li><a href="https://www.politico.eu/article/elon-musk-claim-ukraine-linked-cyberattack-x-draws-criticism/">Musk blames Ukrainians for cyberattack on X. Experts aren&#8217;t convinced.</a>: Evidence of Ukrainian involvement in X disruptions is very thin, cyber experts say.</li><li><a href="https://www.bbc.co.uk/news/articles/c62x5k44rl0o">&#x27;Garbage&#x27; to blame Ukraine for massive X outage, experts say</a>: The claim has been made by the platform&#x27;s owner Elon Musk, a vocal critic of Ukraine and its president.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1349062806254649426)** (2 messages): 

> `Reka Flash 3, Cakeify LoRA, Wan2.1 14B I2V 480p` 


- ****Reka Flash 3** isn't for on-device use anymore**: New rule: **Reka Flash 3**, a **21B** general-purpose reasoning model, can no longer be called *on-device*.
   - The model, trained from scratch on synthetic and public datasets, performs competitively with proprietary models like **OpenAI o1-mini**, and powers Nexus, Reka's platform for creating and managing AI workers with native deep research capabilities ([Reka Space](https://space.reka.ai), [getnexus.reka.ai](https://getnexus.reka.ai)).
- ****Cakeify LoRA** Makes its Debut**: The **Cakeify Effect LoRA** for **Wan2.1 14B I2V 480p** has been released, which lets users *cakeify* any object in an image ([Cakeify](https://huggingface.co/Remade/Cakeify)).
   - It transforms images into videos of objects being cut into cake, using a simple prompt structure adapted from the **Wan2.1 14B 480p I2V** base model, plus instructions to join their [Discord](https://discord.com/invite/7tsKMCbNFC) to generate videos with this LoRA for free and request new LoRAs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/RekaAI/reka-flash-3">RekaAI/reka-flash-3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Remade/Cakeify">Remade/Cakeify · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1348993164827951154)** (2 messages): 

> `RAGcoon, Startup assistance, Agentic RAG, Qdrant vector database, LlamaIndex` 


- **RAGcoon Launching to Help Startups**: A new **agentic RAG** project named [RAGcoon](https://github.com/AstraBert/ragcoon) has launched, designed to assist in building startups by navigating various resources and suggestions.
   - It leverages **free resources from successful founders** and performs complex retrieval operations using techniques like *hybrid search, query expansion*, and *multi-step query decomposition*.
- **RAGcoon Boasts Impressive Reliability Metrics**: **RAGcoon** evaluates the **reliability of retrieved context**, along with the **relevancy and faithfulness** of its own responses through an *auto-correction* mechanism.
   - It is built on **LlamaIndex**, uses **Qdrant** for vector database services, **Groq** for LLM inference (**QwQ-32B by Qwen**), **Hugging Face** for embedding models, **FastAPI** for the backend API, and **Mesop** by Google for the frontend.
- **RAGcoon Available as Docker-Ready Local Install**: **RAGcoon** is **open-source** and can be spun up locally as it is [Docker-ready](https://github.com/AstraBert/ragcoon).
   - The creator is exploring the possibility of an online version and is open to collaboration.



**Link mentioned**: <a href="https://github.com/AstraBert/ragcoon">GitHub - AstraBert/ragcoon: Agentic RAG to help you build a startup🚀</a>: Agentic RAG to help you build a startup🚀. Contribute to AstraBert/ragcoon development by creating an account on GitHub.

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1348995423284494367)** (1 messages): 

> `Real-time Disease Detection, Pretrained Models for Fine-Tuning` 


- **Brainstorming Real-time Disease Detection System**: A member is seeking guidance on creating a real-time disease detection system, intending to use a camera to identify points of interest and subsequently classify them.
   - The inquiry focuses on identifying suitable pretrained models that can be fine-tuned for this specific application.
- **Exploring Pretrained Models for Disease Detection**: The member's primary goal is to determine which pretrained models would be most effective for fine-tuning in the context of real-time disease detection.
   - This involves leveraging computer vision techniques to analyze camera input and classify potential disease indicators.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1348995746518667317)** (2 messages): 

> `SmolLM, Mobile LLMs, Fine-tuning LLMs` 


- **SmolLM Surfaces as Small LLM Solution**: In response to a query about the best small LLM models suitable for fine-tuning and mobile phone deployment, a member suggested exploring **SmolLM**.
   - They shared a [GitHub repository](https://github.com/huggingface/smollm) for **SmolLM2** and **SmolVLM** family of models.
- **SmolLM Github Repository**: The **SmolLM** GitHub repository is available at [huggingface/smollm](https://github.com/huggingface/smollm).
   - It contains information about the **SmolLM2** and **SmolVLM** family of models.



**Link mentioned**: <a href="https://github.com/huggingface/smollm">GitHub - huggingface/smollm: Everything about the SmolLM2 and SmolVLM family of models</a>: Everything about the SmolLM2 and SmolVLM family of models  - GitHub - huggingface/smollm: Everything about the SmolLM2 and SmolVLM family of models

  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1349131009911685182)** (2 messages): 

> `ChatML conversion woes, Llama discrepancies` 


- **Tokenizer Troubles Trigger Technical Task Turmoil**: A member expressed difficulty converting a dataset to ChatML format using the tokenizer's method, despite successfully doing so with a **for loop**.
   - They requested guidance or a reference solution, stating that *the notebook leaves more questions than answers*.
- **Llama's Linguistic Leaps Lead to Varied Outputs**: A member observed that the response of the **Llama model** differs when using the `chat` method versus manually inputting the expected **chat template**.
   - They provided an [image](https://cdn.discordapp.com/attachments/1313889336907010110/1349158970501369916/image.png?ex=67d215b4&is=67d0c434&hm=5a4ca825469b47ad142bff31b9c0e7c34c8d25ebe1ebf5de85a8299740032c4a&) illustrating this discrepancy.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1348980720667856977)** (39 messages🔥): 

> `LlamaIndex error, smolagents DocstringParsingException, smolagents OpenAI/DeepSeek keys, Ollama integration with Smolagents, Brave Browser's Leo AI` 


- **Troubleshooting Async Await in LlamaIndex**: A user encountered a `SyntaxError: 'await' outside function` when using `pipeline.arun` in LlamaIndex with Python 3.10.12.
   - The suggestion was to upgrade to Python 3.11, as the error indicates an issue with asynchronous code execution in older versions.
- **Docstring Parsing Problems with Smolagents**: A user faced a `smolagents._function_type_hints_utils.DocstringParsingException` when generating a JSON schema for a custom tool in smolagents, due to a missing description for an argument in the docstring.
   - The error persisted even when the order of arguments was switched, suggesting a problem with how the docstring is parsed.
- **Smolagents Keys for Gemini, OpenAI and DeepSeek**: Members shared code snippets for using **Gemini, OpenAI, and DeepSeek models** with `smolagents`, providing examples for setting up the `LiteLLMModel` and `OpenAIServerModel` with appropriate API keys.
   - For **Gemini**, a [link to Google AI Studio](https://aistudio.google.com/app/apikey) was provided to obtain a free API key.
- **Ollama takes over HfApiModel**: A member shared a code snippet to demonstrate how to replace `HfApiModel` from Hugging Face with **Ollama** for use with `smolagents`.
   - The solution involves creating a custom `OllamaModel` class that interacts with Ollama's API for prompt generation, allowing local LLMs to be used with `smolagents`.
- **Brave's Leo AI unveiled Llama 3.1 8B**: Members joked that **Brave Browser's Leo AI** is powered by **Llama 3.1 8B**, which is lightweight enough to run on a laptop without significant system resource usage.
   - They questioned why Brave charges extra for more usage of Leo AI, given its relatively low resource requirements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cloud.langfuse.com/project/cm7bq0abj025rad078ak3luwi/traces/995fc019255528e4f48cf6770b0ce27b?timestamp=2025-02-19T10:28:36.929Z">no title found</a>: no description found</li><li><a href="https://aistudio.google.com/app/apikey">no title found</a>: no description found</li><li><a href="https://youtu.be/C7_DLQLrS9w?si=uLQYuYAAA-EWW5B7">Hugging Face AI Agents Course - Unit 1 | Introduction to Agents</a>: In this video, I cover the basics of AI agents and large language models (LLMs) as part of the Hugging Face AI Agents course. You&#39;ll learn what agents are, h...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/tree_summarize/#summarize">Tree Summarize - LlamaIndex</a>: no description found</li><li><a href="https://github.com/openai/openai-agents-python">GitHub - openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows</a>: A lightweight, powerful framework for multi-agent workflows - openai/openai-agents-python
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1348995790282162256)** (5 messages): 

> `GPU mode, CUDA expertise, San Jose meeting` 


- **GPU Mode Hype Train Departs**: A member enthusiastically announced their readiness for **GPU mode** across multiple channels, but quickly deleted the message, signaling second thoughts.
   - A moderator redirected the member to use specific channels (<#1288557096404516945> or <#1218444432588800010>) for such announcements to maintain server hygiene.
- **CUDA Noob Seeks San Jose Summit**: A member inquired about attending the **GPU mode** meeting on **March 16th** in **San Jose**, despite lacking **CUDA** expertise.
   - The question raises the issue of whether specialized knowledge is a prerequisite for participation.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1348981893227479152)** (9 messages🔥): 

> `Triton tl.cast vs tl.full, Pipeline softmax kernel in Triton, Tiled GEMM Implementation in Triton, TF32 precision in Triton` 


- ****Triton's** `tl.full` **Solves Casting Conundrums****: A user found success using `tl.full` to create a **0-dim tensor** with a specific value and data type (`tl.full((), 5, tl.int8)`) to avoid overflow issues when adding to a tensor.
   - The working solution was `tmp_5 = tl.full((1,), value=5, dtype=tl.int8); out = a.to(tl.int8) + tmp_5`.
- ****Triton Triumphs in Softmax Kernel Speed Race****: A user implemented a pipeline softmax kernel in **Triton** that surprisingly outperformed **PyTorch's** equivalent, noting that the **Triton** version was significantly faster than expected on a **float16 T4** colab.
   - An image was attached to the post showing the results of the speed comparison [image.png](https://cdn.discordapp.com/attachments/1189607595451895918/1349066304341934160/image.png?ex=67d1bf67&is=67d06de7&hm=e424c1148a06e3ac3adac9c31cd7c0bc6e930f047dc69c2db02cba55e5949695&).
- ****Triton Tiled GEMM Faces Precision Predicaments****: A user's tiled GEMM implementation in **Triton**, available on [GitHub](https://github.com/gauravjain14/mlcompilers_and_kernels/blob/main/triton_kernels/SimpleOpKernels/triton_tiled_2d_matmul.py), required high tolerances (`atol=1e-1, rtol=1e-1`) in `torch.allclose` for element-wise comparison against a reference **PyTorch** matmul.
   - The expectation was to achieve closer precision (around `1e-4`), prompting a request for pointers to improve accuracy.
- ****Precision Parameters Promotes Performance Prowess****: A user suggested disabling **TF32** in `tl.dot` (using `allow_tf32=False` or `input_precision=`) to improve precision in **Triton**, which solved the precision issues.
   - They noted that `allow_tf32` is deprecated and suggested exploring alternative options with `tl.dot(input_precision=)`.



**Link mentioned**: <a href="https://github.com/gauravjain14/mlcompilers_and_kernels/blob/main/triton_kernels/SimpleOpKernels/triton_tiled_2d_matmul.py">mlcompilers_and_kernels/triton_kernels/SimpleOpKernels/triton_tiled_2d_matmul.py at main · gauravjain14/mlcompilers_and_kernels</a>: Contribute to gauravjain14/mlcompilers_and_kernels development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1348979778451148923)** (19 messages🔥): 

> `stmatrix padding, BFE/BFI instructions, CUDA binary tools` 


- **Padding stmatrix Addresses Prevents SMEM Bank Conflicts**: Addresses for **stmatrix** must be padded to avoid all targeting the same starting **SMEM bank**, causing an 8x conflict, a problem previously fixed in fast.cu and deepgemm codes.
   - There is *no hardware solution* for this issue, requiring careful memory layout management, particularly when tiled layouts aren't feasible.
- **Mystery surrounds BFE/BFI instruction saga in CUDA**: **BFE/BFI** (bitfield extract/insert) instructions, despite being removed in **CC 7.x** and **8.x**, and reintroduced in **9.x**, are not native SASS, but translate to *two instructions* on **sm_70/sm_80/sm_90** and **sm_89** at least with **nvcc 12.6.2**.
   - The best current workaround involves a conditional funnel shift and mask, which, despite thread divergence, outperforms BFE or shifting a u64.
- **CUDA Binary Utilities Application Notes Available**: Application notes for **cuobjdump**, **nvdisasm**, **cu++filt**, and **nvprune**, which are CUDA binary tools for Linux, Windows, Mac OS and Android, have been released, which are accessible via the [CUDA Binary Utilities documentation](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#blackwell-instruction-set).
   - The documentation specifies the differences between `cuobjdump` and `nvdisasm`, explaining that a CUDA binary (cubin) file is an ELF-formatted file consisting of CUDA executable code sections as well as sections containing symbols, relocators, debug info, etc.



**Link mentioned**: <a href="https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#blackwell-instruction-set">1. Overview — CUDA Binary Utilities 12.8 documentation</a>: no description found

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1349070172601712815)** (9 messages🔥): 

> `leetgpu.com, leaderboard, GPU mode` 


- **Leetcode gets GPU mode**: A member discovered [leetgpu.com](https://leetgpu.com), a platform akin to **LeetCode** but designed for individuals interested in utilizing **GPU acceleration**.
   - The platform aims to accelerate problem-solving, particularly for **PMPP** (likely referring to Parallel and Multiprocessor Programming) problems, and features a leaderboard for competitive programming.
- **New leaderboard for leetgpu just dropped**: A new leaderboard has been created and is available on the [GPU Mode discord channel](https://discord.com/channels/1343002583001726986).
   - Users can submit their `/leaderboard submit` and compete with others.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: https://jackhopkins.github.io/factorio-learning-environment/
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1349035088079487118)** (5 messages): 

> `ROCm Version, Ubuntu Versions, vLLM on AMD` 


- ****ROCm** 6.2+ for **gfx 1100****: For **gfx 1100** and above, **ROCm >= 6.2** is recommended, potentially paired with **PyTorch 2.6+** which includes **ROCm 6.2** or a nightly build with **ROCm 6.3**.
- ****Ubuntu 22** or **24** supported**: Both **Ubuntu 22** and **24** are supported, and if it's a fresh install, **Ubuntu 24** is suggested for its latest kernel, according to the [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
- **Running **vLLM** on **AMD****: One member managed to run **vLLM** on **AMD** devices via a docker image, while another struggled to build it from source due to cluster limitations, and wished for alternative methods.



**Link mentioned**: <a href="https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html">Compatibility matrix — ROCm Documentation</a>: no description found

  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1349041378314948704)** (3 messages): 

> `MLX custom kernels, Metal-cpp, GPU Programming, C++ linear algebra library` 


- **Custom Kernel Construction**: MLX documentation has a good section on [building a custom kernel](https://ml-explore.github.io/mlx/build/html/dev/extensions.html), guiding through metal implementation, binding, and python usage.
   - The example involves creating an operation that scales two arrays `x` and `y` by coefficients `alpha` and `beta` respectively, then adds them together to get the result `z = alpha * x + beta * y`.
- **Metal-cpp is preferred**: After receiving multiple suggestions, a member decided to go with **metal-cpp** and is starting by learning about **MLX’s custom kernels** and completing the **Metal-Puzzles challenge**.
   - The member is also planning to work on a small personal project: a **C++ linear algebra library** accelerated using Metal.
- **Metal & Swift Integrated**: A member suggested looking into **Sebastian's** earlier suggestion, and mentioned a matrix multiplication example integrating **Metal** and **Swift**.
   - They will try to find the example to send if possible.



**Link mentioned**: <a href="https://ml-explore.github.io/mlx/build/html/dev/extensions.html">Custom Extensions in MLX &#8212; MLX 0.23.2 documentation</a>: no description found

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1349130384184709182)** (1 messages): 

> `IPFS Accelerate JS, HuggingFace Port to TS/JS` 


- **IPFS Accelerate JS Structure Implemented**: Initial structure for **IPFS Accelerate JS** was implemented with placeholder modules and TypeScript conversion via [this commit](https://github.com/endomorphosis/ipfs_accelerate_py/commit/2f9963372a890cc7d7abe4399f5cfa7fc438a773).
- **HuggingFace Libraries Migrate to TS/JS with WebNN/WebGPU**: A member is actively porting the entire **HuggingFace** libraries to **TS/JS** using **WebNN/WebGPU**.



**Link mentioned**: <a href="https://github.com/endomorphosis/ipfs_accelerate_py/commit/2f9963372a890cc7d7abe4399f5cfa7fc438a773">feat: Implement initial structure for IPFS Accelerate JS with placeho… · endomorphosis/ipfs_accelerate_py@2f99633</a>: …lder modules and TypeScript conversion

  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1348994348058017812)** (6 messages): 

> `Private eval benchmark service, reasoning-gym, Curriculum benchmarks` 


- **Private Eval Benchmark Service Considered**: A member proposed operating a **private eval benchmark service** with a private seed to hide answers, linking to the [reasoning-gym tools](https://github.com/open-thought/reasoning-gym/tree/main/tools).
   - They wondered if this was outside the scope of the project, but acknowledged they have the capability to do so.
- **Curriculum benchmarks could see demand**: A member suggested creating curriculum benchmarks like `rg-private-easy` and `rg-private-hard`.
   - They speculated that there might be demand for such a service if **RG** gains decent traction.



**Link mentioned**: <a href="https://github.com/open-thought/reasoning-gym/tree/main/tools">reasoning-gym/tools at main · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1349046515271733339)** (4 messages): 

> `cublas autotune, PTX optimization, Triton autotune, Cutlass autotune` 


- **cublas autotune explores PTX**: **cublas** can also **autotune**, but it's technically more in-depth because it uses more **PTX optimization**.
   - Using **Triton** or **Cutlass** autotune methods is simpler because the high-level optimization is easier to understand, according to a member.
- **cublas docs provided**: A member provided the [cublas docs](https://docs.nvidia.com/cuda/cublas/#cublasltmatmulalgogetheuristic) for reference.
   - The docs discuss **cublasltmatmulalgogetheuristic**.


  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1349035449410261086)** (1 messages): 

> `Nvidia Blackwell, SemiAnalysis, GPU Hackathon, Open Source` 


- **SemiAnalysis Hosts Blackwell GPU Hackathon**: SemiAnalysis is hosting an **Nvidia Blackwell GPU Hackathon** on **Sunday, March 16th**, offering hands-on exploration of **Blackwell & PTX infrastructure** while working on open-source projects, more details available at the [SemiAnalysis Hackathon page](https://semianalysis.com/hackathon-2025/).
- **Hackathon Boasts Star-Studded Speaker Lineup**: The hackathon will feature speakers including **Philippe Tillet** of **OpenAI**, **Tri Dao** of **TogetherAI**, and **Horace He** of **Thinking Machines**.
   - The event is sponsored by Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave, and Nebius.



**Link mentioned**: <a href="https://semianalysis.com/hackathon-2025/">Hackathon 2025</a>: SemiAnalysis is kicking things off ahead of NVIDIA GTC! Start your day with engaging morning keynotes, hack all day with low-level NVIDIA GPU programming (maybe even Blackwell), take a breather wit…

  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1349055277680427068)** (17 messages🔥): 

> `Reka Flash 3, Anthropic ARR Growth, Manus AI Sensation, Nvidia Blackwell GPU Hackathon, Automation and Robotics in China` 


- **Reka Labs Opensources Reka Flash 3**: [Reka Labs](https://x.com/RekaAILabs/status/1899481289495031825) has open-sourced **Reka Flash 3**, a new reasoning model trained from scratch with only **21B parameters** achieving competitive performance.
   - The model was finetuned on synthetic and public datasets, followed by **RLOO** with model-based and rule-based rewards, forcing the model to output *&lt;/reasoning&gt;* to control quality vs. thinking time, as described in their [blog post](https://www.reka.ai/news/introducing-reka-flash).
- **Anthropic Sees strong revenue growth powering Manus AI**: According to [The Information](https://www.theinformation.com/articles/anthropics-claude-drives-strong-revenue-growth-while-powering-manus-sensation), **Anthropic** grew from **$1B ARR to $1.4B ARR** in the first two months of 2025.
   - Their models are also powering **Manus**, described as *the latest AI sensation*.
- **SemiAnalysis hosts Nvidia Blackwell GPU Hackathon**: [SemiAnalysis](https://semianalysis.com/2025/03/11/america-is-missing-the-new-labor-economy-robotics-part-1/) is hosting a [Nvidia Blackwell GPU Hackathon on Sunday March 16th](https://semianalysis.com/hackathon-2025/) with speakers from **OpenAI**, **TogetherAI**, and **Thinking Machines**.
   - The hackathon aims to explore **Blackwell & PTX infrastructure** while collaborating on open-source projects, sponsored by Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave, and Nebius.
- **OpenAI Introduces New APIs and Agents SDK**: [OpenAI](https://x.com/btibor91/status/1899513477716410871) launched new APIs and tools for easier development of agent applications, including the **Responses API**, **Web search tool**, **File search**, **Computer use tool**, and an **open-source Agents SDK**.
   - The existing **Assistants API** will be phased out by mid-2026, and the changelog mentions new models **o3-mini-pro** and **o1-pro** in the API.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reka.ai/news/introducing-reka-flash">Reasoning with Reka Flash  | Reka</a>: Today, we are open sourcing a research preview of a new version of Reka Flash 3, our 21 billion parameters model. Reka Flash 3 is a compact, general-purpose model that excels at general chat, coding, ...</li><li><a href="https://x.com/RekaAILabs/status/1899481291889979896">Tweet from Reka (@RekaAILabs)</a>: Reka Flash 3 was finetuned on synthetic and public datasets, followed by RLOO with model-based and rule-based rewards. 🏋️We found that forcing the model to output &lt;/reasoning&gt; is an effective w...</li><li><a href="https://semianalysis.com/2025/03/11/america-is-missing-the-new-labor-economy-robotics-part-1/">America Is Missing The New Labor Economy &#8211; Robotics Part 1</a>: SemiAnalysis is hosting an Nvidia Blackwell GPU Hackathon on Sunday March 16th. It is the ultimate playground for Blackwell PTX tech enthusiasts, offering hands-on exploration of Blackwell &amp; PT…</li><li><a href="https://x.com/steph_palazzolo/status/1899498723010662449">Tweet from Stephanie Palazzolo (@steph_palazzolo)</a>: Anthropic is having a good start to the year: They grew from $1B ARR to $1.4B ARR in the first two months of 2025. Their models are also powering Manus, the latest AI sensation to blow up on X, which ...</li><li><a href="https://x.com/RekaAILabs/status/1899481289495031825">Tweet from Reka (@RekaAILabs)</a>: ⚡We are open sourcing Reka Flash 3, our new reasoning model that was trained from scratch. It achieves competitive performance with only 21B parameters. ⚡Reka Flash 3 powers Nexus, our new enterprise ...</li><li><a href="https://x.com/btibor91/status/1899513477716410871">Tweet from Tibor Blaho (@btibor91)</a>: OpenAI introduced new APIs and tools for easier development of agent applications- The new Responses API, available to all developers starting today, combines features of Chat Completions API and Assi...</li><li><a href="https://tenor.com/boMsW.gif">Dig Up GIF - Dig Up Stupid - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1349072819396284530)** (2 messages): 

> `Claude Code Decompilation, GitHub Repo` 


- **Claude Code Decompilation Gone Too Soon**: A user noted that the decompiled **Claude code** was taken down from [Twitter](https://x.com/odazai_/status/1899512495166865699).
   - However, the code is still available on [GitHub](https://github.com/dnakov/anon-kode).
- **GitHub Repo Still Up**: The GitHub repository containing the decompiled **Claude code** remains accessible at [dnakov/anon-kode](https://github.com/dnakov/anon-kode).
   - This allows continued access and study of the code despite its removal from other platforms.



**Link mentioned**: <a href="https://x.com/odazai_/status/1899512495166865699">Tweet from Dazai (@odazai_)</a>: @dnak0v @cheatyyyy They took down the decompiled claude-code 😢

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1349055950673412167)** (34 messages🔥): 

> `Qwen Chat Enhanced, New tools for building agents with the API, Anthropic CEO, Dario Amodei predicts most code will be written by AI in 12 months, Sama hypeposts new OpenAI model good at creative writing` 


- **Qwen Chat Gets a Glow-Up**: [Qwen Chat](http://chat.qwen.ai) receives an update with a **unified multimodal interface** for all **Qwen2.5 models**, enhanced video understanding (up to **500MB**), redesigned mobile experience with voice-to-text, guest mode accessibility, and expanded file upload capacity (doubled to **20MB**).
- **OpenAI's API Evolving for Agent Builders**: OpenAI is evolving its API platform to make it faster and easier for developers to build agents, as announced in their [YouTube video](https://www.youtube.com/live/hciNKcLwSes?si=QdsCwk5dnKktLG29) introducing **Web Search** (fine-tuned 4o/mini + Web), **File Search API Updates**, and **Computer Use**.
   - They also have a new **Python library** for agent stuff and a **responses API**, which is a superset of chat completions.
- **Dario's Bold Coding Prediction**: Anthropic CEO, Dario Amodei, predicts that AI will write **90%** of the code in the next **3 to 6 months**, and nearly all code within **12 months**, according to a [tweet](https://x.com/slow_developer/status/1899430284350616025?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ).
- **Sama Hypes Metafictional OpenAI Model**: Sam Altman shares a [metafictional literary short story about AI and grief](https://x.com/sama/status/1899535387435086115) generated by a new OpenAI model, noting it's *the first time I have been really struck by something written by AI; it got the vibe of metafiction so right.*
   - The community seemed unimpressed, saying, *pretty good - still insists upon itself too much but it's got that sovl*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/simonw/status/1899512037526626336">Tweet from Simon Willison (@simonw)</a>: To OpenAI&#39;s credit they do at least knowledge that this is an issue! From that page:&#34;The Chat Completions API is an industry standard for building AI applications, and we intend to continue su...</li><li><a href="https://www.youtube.com/live/hciNKcLwSes?si=QdsCwk5dnKktLG29">New tools for building agents with the API</a>: We’re evolving the API platform to make it faster and easier for developers to build agents. Kevin Weil, Nikunj Handa, Steve Coffey, and Ilan Bigio introduce...</li><li><a href="https://x.com/Alibaba_Qwen/status/1899497336889659775">Tweet from Qwen (@Alibaba_Qwen)</a>: 👋 Introducing the Enhanced Qwen ChatWe are pleased to announce the latest update to Qwen Chat, designed to deliver a seamless, versatile, and user-centric experience. Explore the key features below a...</li><li><a href="https://x.com/sama/status/1899535387435086115">Tweet from Sam Altman (@sama)</a>: we trained a new model that is good at creative writing (not sure yet how/when it will get released). this is the first time i have been really struck by something written by AI; it got the vibe of me...</li><li><a href="https://x.com/slow_developer/status/1899430284350616025?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Haider. (@slow_developer)</a>: Anthropic CEO, Dario Amodeiin the next 3 to 6 months, AI is writing  90% of the code, and in 12 months, nearly all code may be generated by AI
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1349096388197355603)** (2 messages): 

> `AI Distillation, Neural Sequence Chunkers, History Compression, DeepSeekR1, Reinforcement Learning Prompt Engineer` 


- **Schmidhuber Highlights AI Distillation History**: Jürgen Schmidhuber notes that [CNBC is discussing AI distillation](https://x.com/SchmidhuberAI/status/1899475671958929453) thanks to **DeepSeek**, referencing his 1991 work on *collapsing* neural networks, now known as distillation.
- **Details on Early Neural Network Compression**: Schmidhuber's 1991 tech report, [Neural sequence chunkers](https://example.com), details *compressing* or *collapsing* a neural network's knowledge into another network, a method now widely used.
   - The automatizer could replicate the chunker's actions post-distillation.
- **DeepSeekR1 Utilizes Distilled Chain of Thought**: Schmidhuber points out that **DeepSeek** uses elements from his 2015 reinforcement learning prompt engineer and its 2018 refinement, collapsing the RL machine and world model into a single network, employing the 1991 neural net distillation procedure.



**Link mentioned**: <a href="https://x.com/SchmidhuberAI/status/1899475671958929453">Tweet from Jürgen Schmidhuber (@SchmidhuberAI)</a>: Thanks to #DeepSeek, even @CNBC [9] now talks about AI distillation, published in 1991 [1][2]. I called it &#34;collapsing,&#34; back then, not “distilling.&#34; See also [9][10]. REFERENCES[1] J. Sch...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1349026056149008405)** (6 messages): 

> `Inductive Moment Matching (IMM), Claude 3.7 Sonnet reasoning` 


- **LumaLabs Breaks Algorithmic Ceiling with IMM**: LumaLabs released **Inductive Moment Matching (IMM)**, a new pre-training paradigm that achieves higher sample quality with **10x** more efficiency and stable training, detailed in their [blog post](http://lumalabs.ai/news/imm) and [ArXiv paper](https://arxiv.org/abs/2503.07565).
   - The IMM model surpasses diffusion models on **ImageNet-256x256** with **1.99 FID** using only **8** inference steps, and achieves state-of-the-art **2-step FID** of **1.98** on **CIFAR-10** when trained from scratch.
- **Claude 3.7 Sonnet's Reasoning Laid Bare**: Anthropic's research suggests that **Claude 3.7 Sonnet** doesn’t encode hidden reasoning in its scratchpad, evidenced by the fact that training it to use paraphrased versions of the scratchpads does not degrade performance ([Anthropic blog post](https://alignment.anthropic.com/2025/distill-paraphrases/)).
   - The scratchpads from reasoning models look human understandable but might improve performance through some less human-understandable mechanism.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.07565">Inductive Moment Matching</a>: Diffusion models and Flow Matching generate high-quality samples but are slow at inference, and distilling them into few-step models often leads to instability and extensive tuning. To resolve these t...</li><li><a href="https://alignment.anthropic.com/2025/distill-paraphrases/">Do reasoning models use their scratchpad like we do? Evidence from distilling paraphrases</a>: no description found</li><li><a href="https://x.com/LumaLabsAI/status/1899518379737661447">Tweet from Luma AI (@LumaLabsAI)</a>: Today, we release Inductive Moment Matching (IMM): a new pre-training paradigm breaking the algorithmic ceiling of diffusion models. Higher sample quality. 10x more efficient. Single-stage, single net...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1349023571753697281)** (40 messages🔥): 

> `MCP servers in Cursor, GitHub org owned repos verification, Claude Desktop config schema, Smithery CLI with Gitlab MCP server, Openai's SDK MCP compatibility` 


- **MCP Servers Struggle to Integrate with Cursor**: A user faced issues adding **MCP servers** like Brave Search to **Cursor**, despite successful integration with Claude, reporting errors like *no tools available* and *no resources available*.
- **GitHub org repos claim has limitations**: A member reported that the *login with github to claim* feature doesn't behave as expected with repos owned via a **GitHub organization** at [glama.ai/mcp/servers/gwrql5ibq2](https://glama.ai/mcp/servers/gwrql5ibq2).
   - Another member acknowledged this as a **known limitation** and mentioned plans to address it this week.
- **Trouble with Smithery CLI and Gitlab MCP Server on Windows**: A user is experiencing difficulties using **Smithery CLI** to run a **Gitlab MCP server** on Windows.
   - They consistently encounter errors such as *Failed to create client* or *Unexpected JSON response*.
- **OpenAI SDK Supports MCP**: A user pointed out that **OpenAI's SDK** should now be **MCP compatible** according to [openai.github.io/openai-agents-python/tools/](https://openai.github.io/openai-agents-python/tools/).
- **Handoff Includes Full Conversation History**: A member shared a [github.com search result](https://github.com/search?q=repo%3Aopenai%2Fopenai-agents-python%20handoff_prompt&type=code) that, by default, the **handoff** includes the entire conversation history (system/user/assistant messages).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openai.github.io/openai-agents-python/tools/">Tools - OpenAI Agents SDK</a>: no description found</li><li><a href="https://github.com/search?q=repo%3Aopenai%2Fopenai-agents-python%20handoff_prompt&type=code">Build software better, together</a>: GitHub is where people build software. More than 150 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1348977748714065950)** (10 messages🔥): 

> `MCP, Model Context Protocol, Elixir's Phoenix Framework, llama.cpp, vllm` 


- **Phoenix Framework Fuels MCP Implementations**: A member shared [MCPheonix on Github](https://github.com/jmanhype/MCPheonix), a simplified implementation of the **Model Context Protocol (MCP) server** using **Elixir's Phoenix Framework**.
- **Local LLM Inference via Llama.cpp and VLLM**: A member inquired about attaching to other **LLMs** like **llama.cpp** or **vllm** for local **LLM** inference, with another member responding that as long as they support tool calling, it's straightforward.
- **Unraid Servers Controlled via MCP**: A member shared their [unraid-mcp](https://github.com/jmagar/unraid-mcp) project to control your **Unraid server via MCP**.
- **Android Devices Controlled via AI Using MCP**: A member shared their [DroidMind project](https://github.com/hyperb1iss/droidmind), an **MCP server** that can manage your **Android devices** over **ADB**, useful for debugging on-device issues and analyzing logs.
- **MCP Servers Building MCP Servers**: A member introduced [mcp-create](https://github.com/tesla0225/mcp-create), an **MCP server** that builds **MCP servers**, supporting **TypeScript**, and capable of running the generated **MCP server** directly, accompanied by [an explanatory article](https://zenn.dev/tesla/articles/c66bda76c4a523).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/hyperb1iss/droidmind">GitHub - hyperb1iss/droidmind: Control your Android devices with AI using Model Context Protocol</a>: Control your Android devices with AI using Model Context Protocol - hyperb1iss/droidmind</li><li><a href="https://github.com/jmanhype/MCPheonix">GitHub - jmanhype/MCPheonix: A simplified implementation of the Model Context Protocol (MCP) server using Elixir&#39;s Phoenix Framework.</a>: A simplified implementation of the Model Context Protocol (MCP) server using Elixir&#39;s Phoenix Framework. - jmanhype/MCPheonix</li><li><a href="https://github.com/jmagar/unraid-mcp">GitHub - jmagar/unraid-mcp</a>: Contribute to jmagar/unraid-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/tesla0225/mcp-create">GitHub - tesla0225/mcp-create</a>: Contribute to tesla0225/mcp-create development by creating an account on GitHub.</li><li><a href="https://zenn.dev/tesla/articles/c66bda76c4a523">MCPサーバーを作るMCPサーバーを作った</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1348983272356581439)** (7 messages): 

> `NotebookLM for Exam Prep, NotebookLM for Medical Guidelines and Patient Info, Automating NotebookLM Uploads, NotebookLM Audio Overview` 


- **NotebookLM Aces Exam Prep**: A user split PDFs into sections based on bookmarks, made a separate notebook for each, and quizzed the notebook on all of the topics in the study guide, usually limited to the section for the week instead of the whole notebook, yielding *very good results*.
   - The user then turned these results into flashcards in other apps.
- **NotebookLM Generates Medical Docs**: A user in the medical field found NotebookLM *AMAZING* for parsing existing guidelines and hard-to-navigate websites and then creating patient information for discharge home.
   - Specifically, the user created a concise one-page document for patients regarding work-related injury claims, providing a check-list style list of things they should be aware of and complete upon discharge.
- **Automating NotebookLM ingestions**: A user is automating optimization of the information to upload to NotebookLM, focusing on smaller files for *easier robot ingestion*.
   - The user is streamlining their workflow to make it easier for NotebookLM to process documents.
- **NotebookLM as Audio Overview Virtuoso**: A user used NotebookLM to create an audio overview of a Google Doc containing LLM responses to a business idea prompt.
   - The audio speakers referred to the different tabs (LLM names) and did a remarkable job of discussing the best parts, quoting the right LLM each time; BTW, **Gemini Advanced Pro came out a winner** for the best response.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1348977488335863818)** (17 messages🔥): 

> `Gemini, Notebook LM Limits, Language support, Source management, Audio overview` 


- **Gemini generates discontent**: A user expressed dissatisfaction with **Gemini**, despite being heavily integrated into the Google ecosystem.
- **NotebookLM supports humongous knowledge bases**: A user with a **10 million word knowledge base** (1500 books, 6000 videos in text) inquired about the limits of NotebookLM, specifically around file depth and total volume.
   - A member of the NLM team clarified that NotebookLM supports **10 million words**, within the **300 source and 500,000 words/source limits**, leveraging **RAG** to process relevant portions.
- **NotebookLM plus has user limits**: A user reported, *With "The system was unable to answer." a persistent issue, that 50 isn't much anymore.*
- **Users want more language support**: A user asked about the roadmap and timeline for NotebookLM to support output languages beyond English.
- **Timestamped event cues don't affect output dialogue**: A user asked if timestamped event cues in a .txt file would affect the audio overview output.



**Link mentioned**: <a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found

  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1349135010447360011)** (1 messages): 

> `Windsurf Referral Challenge, v1.4.6 Patch Fixes, Windsurf Previews, Auto-Linter, MCP Servers` 


- **Refer Friends, Snag Credits, Dominate Swag!**: The **Windsurf Referral Challenge** encourages users to refer friends for a chance to win **500 flex credits** each upon their friend's Pro subscription, plus a shot at custom **Airpods Pro Max headphones** for the most referrals by **March 31st** via [windsurf.ai/refer](https://windsurf.ai/refer).
- **Windsurf v1.4.6 Fixes MCP, Sonnet, and Proxies**: Windsurf released **patch fixes in v1.4.6** addressing **MCP reliability**, **3.7 Sonnet web search**, and **proxy settings**, as detailed in the [changelog](https://www.codeium.com/changelog).
- **Windsurf Previews Cascade Locally**: **Windsurf Previews (Beta)** now allows users to preview locally run websites directly within Cascade, check out the attached [image](https://cdn.discordapp.com/attachments/1027688115592237117/1349135011659517994/IMG_5342.png?ex=67d1ff64&is=67d0ade4&hm=569177e0fbf1e9818203093be6e4efda6a0f0c528dbbeb99786d89a257da1c30&).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://windsurf.ai/refer">Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1349040086746005678)** (23 messages🔥): 

> `Codeium VS Code extension issues, Claude 3.7 Sonnet in VS Code, Codeium extension credits, VS Code extension version discrepancy, Codeium server errors` 


- **Codeium VS Code Extension Truncates Long Prompts**: A user with a **Pro plan** experienced truncated responses in the Codeium VS Code extension when using long prompts with **Claude 3.7 Sonnet**, with the chat history indicating that *the token limit has been reached*.
   - The user was advised to try **Claude 3.5** due to a reported incident with **3.7**, although the truncation issue may not be directly related.
- **Claude 3.7 Sonnet Thinking Missing in VS Code Extension**: The **Claude 3.7 Sonnet Thinking** model is not available in the VS Code extension, unlike in Windsurf, and the user inquired if additional configuration was required.
   - It was confirmed that **Claude 3.7 Sonnet Thinking** is *not available in the extension at the moment*.
- **Codeium VS Code Extension Doesn't Read Files Directly**: The Codeium VS Code extension chat (**Claude 3.7 Sonnet**) cannot directly read script files from the folder and requires users to paste the file content into the chat.
   - The user was advised to *report it in codeium.com/support, because it should work technically*.
- **Codeium VS Code Extension Usage is Credit-Free**: It was clarified that using the Codeium VS Code extension, even with premium models, is completely free and does not consume any credits, as credits are tied to Windsurf.
   - The user initially thought otherwise but confirmed that these chats and responses do not consume any credits, leading to excitement.
- **Codeium Extension Server Aborts Pending Request**: A user reported a persistent error preventing Codeium from working, with the message *Codeium: The server aborted pending request* and mentioning a download URL from *releases.codeiumdata.com*.
   - The issue persisted across different versions despite restarting the IDE, and contacting *vscode@codeium.com* was suggested.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1349001237709848616)** (18 messages🔥): 

> `operation order in deep learning frameworks, Adaptive Meta-Learning, RLHF issues, Matplotlib and Claude 3.7 graphs, hiding complexity` 


- ****Compilers Optimize Math Operations?****: Members discussed the order of operations in deep learning frameworks like **PyTorch** and **NumPy**, debating whether compilers automatically optimize calculations such as *(1/n) (a(c + d) + b)* versus *a(c/n + d/n) + b/n*.
   - One member humorously suggested just *adding extra brackets* to ensure the system performs operations in the desired order.
- ****Minimal Code vs Messy Code: A Debate****: A discussion occurred around the trade-offs between minimal code and explicit, possibly messier, code, with some arguing that minimal code simply hides complexity under someone else's framework.
   - One member argued that successful AI movements often hide complexity behind simple interfaces, while another cautioned against this, especially when needing to debug or convert models to **ONNX**, leading to issues with branchless programming.
- ****Graphs of Matplotlib drawn by Claude 3.7****: Members expressed excitement about the **Matplotlib** graphs generated by **Claude 3.7**, noting that the *benchmark and svgmaxing* seem to be functioning effectively.
   - No link was given in this exchange.
- ****Adaptive Meta-Learning: A New Term?****: A member inquired whether the term **Adaptive Meta-Learning (AML)** already exists or if they coined it, describing it as a potential combination of *Online HyperParameter Optimization (HPO)* and meta-learning.
   - Another member provided a [link to a Semantic Scholar search](https://www.semanticscholar.org/search?q=Adaptive%20meta-learning&sort=relevance), but concluded that while the keywords are used together, they don't constitute an established framework or paradigm.
- ****RLHF Rigidly Joins Bad Behavior?****: A member speculated, based on an emergent misalignment paper, that **Reinforcement Learning from Human Feedback (RLHF)** might rigidly join bad behavior in batches.
   - They hypothesized that inverting this process could lead models to write bad code, linking lies and undesirable behaviors, suggesting potential economic motivations behind concealing techniques and trade secrets.



**Link mentioned**: <a href="https://www.semanticscholar.org/search?q=Adaptive%20meta-learning&sort=relevance>">Adaptive meta-learning | Semantic Scholar</a>: An academic search engine that utilizes artificial intelligence methods to provide highly relevant results and novel tools to filter them with ease.

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1349163019682185336)** (1 messages): 

> `` 


- **No Paper Discussion Tonight**: No one volunteered to lead a paper discussion tonight.
- **Plans Tonight**: A member mentioned having plans for the night.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1349123858463985797)** (3 messages): 

> `LLMs think in tokens, VR prison` 


- **Do LLMs think inter-token?**: There's discussion on whether [LLMs think in tokens and also between tokens](https://openai.com/index/chain-of-thought-monitoring/).
- **VR Headsets lead to 97% Reduction in Prison Infractions**: A California women's facility is seeing success using VR headsets in solitary confinement, resulting in *>>>97% reduction in infractions* according to [this article](https://www.theguardian.com/technology/2025/mar/08/vr-prison-california).



**Link mentioned**: <a href="https://www.theguardian.com/technology/2025/mar/08/vr-prison-california">‘An ideal tool’: prisons are using virtual reality to help people in solitary confinement</a>: Participants view scenes of daily life as well as travel adventures – then process the emotions they trigger through art

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1348982241245663276)** (11 messages🔥): 

> `Llama Extract Access, Premium Plan Signup, Function Calling Models, MP3 Parsing Error` 


- ****Llama Extract Access Granted****: A member requested access to **Llama Extract** and was offered addition to the closed beta, pending email confirmation.
   - The email address `rasmus-persson@outlook.com` was shared for this purpose.
- ****Premium Plan Upgrade Made Easy****: A user inquired about upgrading from the Free plan to the Premium mode.
   - Instructions were provided to **log in**, click the profile icon, and select the upgrade/manage button.
- ****Deepseek vs 4o Showdown****: A user asked *will I loose quality if I use Deepseek insted of 4o in ANUS?*
   - No helpful responses were given.
- ****MP3 Parsing Puzzle for APIs****: A user reported an error when uploading an **.mp3** file for parsing through the API.
   - The user noted that the upload works fine via the UI/webapp, and provided [a screenshot of the error](https://cdn.discordapp.com/attachments/1059201661417037995/1349100831307202714/Screenshot_2025-03-11_at_3.24.19_PM.png?ex=67d1df8f&is=67d08e0f&hm=3b980c7dd220c3d654ff1cb17819daedcc6fc3c896b2ed955e800b40f2467d3d).
- ****Function Calling Face-Off****: A member asked about alternative models besides those from **OpenAI** that are good for function calling.
   - The user seeks a less expensive option since their apps heavily rely on this functionality.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

amanshrestha: https://github.com/openai/openai-agents-python
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1349028538862534768)** (8 messages🔥): 

> `Judge LLM, ChainPoll, Best of N, dspy.Parallel` 


- **ChainPoll Pattern used for Judge LLM**: Members are building a **Judge LLM** which follows the **ChainPoll** pattern, that uses multiple chain of thought judge programs and returns the *average* response chain.
   - A member suggested using `module.batch()` or `dspy.Parallel` to speed up the process.
- **Best of N Documentation Quest**: A member was having trouble finding docs on **Best of N**.
   - The same member noted that ensemble is listed as a teleprompter and asked if it optimizes or aggregates input programs into an optimal single program.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1348977802627649536)** (7 messages): 

> `OpenPipe's deductive-reasoning, FP8 Fine-Tuning, Torchtune QAT Support, Weight Decay Strategies, Evaluation Dataset Logging` 


- **OpenPipe Deductive-Reasoning: Torchtune Triumphs!**: A member shared a link to [OpenPipe's deductive-reasoning project](https://github.com/openpipe/deductive-reasoning), noting its use of **Torchtune**.
- **FP8 Fine-Tuning Faces Frustrations**: Members discussed the challenges of serving models in **FP8**, and considered the possibility of fine-tuning in **FP8** to reduce quantization error.
   - They noted that **FP8** is challenging due to stability issues during training, so is not a straightforward process.
- **Torchtune's QAT support**: A member inquired about **Torchtune's QAT support**, especially for **FP8**, to potentially fine-tune and reduce quantization error.
   - It was mentioned that [this recipe](https://github.com/pytorch/torchtune/pull/2404) looks promising for **FP8**.
- **Weight Decay Wonders!**: Members suggested that gradually increasing **weight decay** might help keep the weights in the right range during **FP8** fine-tuning.
- **Separate Eval Dataset Recipe Spotted!**: A member sought a recipe supporting a separate **eval dataset** to measure loss every N steps, similar to [this example](https://github.com/pytorch/torchtune/blob/d5d12fef1f8c39dfd5c9f85807795ef503216e12/recipes/full_finetune_single_device.py#L725).
   - One member found [this pull request](https://github.com/pytorch/torchtune/pull/2238/files) for one of the recipes that could be helpful.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/openpipe/deductive-reasoning">GitHub - OpenPipe/deductive-reasoning: Train your own SOTA deductive reasoning model</a>: Train your own SOTA deductive reasoning model. Contribute to OpenPipe/deductive-reasoning development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/2238/files">Adds validation loss to LoRA fine tune single device by MaxFrax · Pull Request #2238 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here)Please link to any issues this PR addresses.#1042Chang...</li><li><a href="https://github.com/pytorch/torchtune/blob/d5d12fef1f8c39dfd5c9f85807795ef503216e12/recipes/full_finetune_single_device.py#L725)">torchtune/recipes/full_finetune_single_device.py at d5d12fef1f8c39dfd5c9f85807795ef503216e12 · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/2404">(WIP/RFC) FP8 full finetune distributed by nathan-az · Pull Request #2404 · pytorch/torchtune</a>: ContextWhat is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here)This would solve #2201.I&amp;#39;m far from an expert with ...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1349085247240929291)** (1 messages): 

> `Regression Tests, Model Size Finalization, Evaluation Metrics, Comprehensive Measurement Strategies` 


- **Regression Tests Added Needing Model Size Finalization**: A member mentioned the addition of several [regression tests](https://github.com/pytorch/torchtune/pull/2477) and inquired about finalizing model size and evaluation methods.
   - The member questioned whether evaluation alone is sufficient, implying a discussion on more comprehensive measurement strategies.
- **Deep Dive on Comprehensive Measurement Strategies**: The discussion veered into the necessity of more comprehensive measurement strategies beyond simple evaluation.
   - Members debated the merits of various evaluation metrics, potentially influencing the choice of model size and testing methodologies.


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

ceifa: 😮
  

---


### **Cohere ▷ #[【📣】announcements](https://discord.com/channels/954421988141711382/996880279224451154/1349082537326284932)** (2 messages): 

> `Expedition Aya 2024, Multilingual AI, Multimodal AI, Efficient AI, Cohere API Credits` 


- **Cohere For AI announces Expedition Aya 2024!**: Cohere For AI is launching [Expedition Aya 2024](https://tinyurl.com/ayaexp2025), a **6-week open-build challenge** to facilitate new collaborations and research projects worldwide, broadening its focus to support research that touches **multilingual, multimodal, or efficiency**.
- **Aya Projects showcase impact**: The previous **Expedition Aya** led to numerous collaborations and publications, as showcased in the [Aya Projects](https://tinyurl.com/4scpn5uu) presentation.
   - Examples of the showcased projects are **DistAYA**, **Doclingual**, and **Enhancing Sinhala NLP**.
- **Expedition Aya offers Resources**: Participants in Expedition Aya will gain access to exclusive resources and **Cohere API credits** to use their models for research.
   - The teams that complete the initiative will be eligible for **limited edition Expedition swag**, and there are exclusive prizes for top projects.
- **Join Expedition Aya for Team Building!**: Members are encouraged to join the [Expedition Aya Discord server](https://discord.gg/q9QRYkjpwk) and attend **Crew Connections meetings** to connect with potential collaborators.
   - The kick-off meeting will occur in **March 2025**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tinyurl.com/ayaexp2025">Expedition</a>:  </li><li><a href="https://tinyurl.com/4scpn5uu">Expedition - Past Projects</a>: DistAYA
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「💡」projects](https://discord.com/channels/954421988141711382/1218409701339828245/1349035344175038548)** (1 messages): 

> `Nvidia Blackwell GPU Hackathon, SemiAnalysis, PTX Infrastructure` 


- **SemiAnalysis hosts Blackwell GPU Hackathon!**: [SemiAnalysis](https://semianalysis.com/) is hosting an **Nvidia Blackwell GPU Hackathon** on **Sunday March 16th** with hands-on exploration of **Blackwell & PTX infrastructure** while collaborating on open-source projects.
   - Speakers include **Philippe Tillet** of **OpenAI**, **Tri Dao** of **TogetherAI**, **Horace He** of **Thinking Machines**, and more, and is sponsored by **Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave, Nebius.**
- **Hackathon Provides PTX Playground!**: The event is the ultimate playground for **Blackwell PTX** tech enthusiasts, offering hands-on exploration of **Blackwell & PTX infrastructure** while collaborating on open-source projects.
   - Attendees can expect engaging morning keynotes, a day of hacking with powerful **Blackwell GPUs** like **GB200s**, insightful afternoon talks, and an unforgettable finale.



**Link mentioned**: <a href="https://semianalysis.com/hackathon-2025/">Hackathon 2025</a>: SemiAnalysis is kicking things off ahead of NVIDIA GTC! Start your day with engaging morning keynotes, hack all day with low-level NVIDIA GPU programming (maybe even Blackwell), take a breather wit…

  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1349051663356268609)** (2 messages): 

> `Multilingual/multicultural communities, Introductions and Community Expectations` 


- **Researcher Seeks Multilingual/Multicultural Communities**: A researcher inquired about the location of multilingual and multicultural activities within the Cohere Discord community.
   - The user expressed appreciation for Cohere's work and noted prior collaborations with the team.
- **Introductions and Community Expectations Highlighted**: A stickied message reminded new members to introduce themselves, specifying key details to share.
   - The message outlined the expected format, including company/industry/university affiliation, current projects, preferred tech/tools, and community goals.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1349034159389016187)** (3 messages): 

> `Nvidia Blackwell GPU Hackathon, SemiAnalysis, Blackwell PTX, GB200, GTC` 


- ****SemiAnalysis** hosts a **Nvidia Blackwell GPU Hackathon****: **SemiAnalysis** is hosting an [Nvidia Blackwell GPU Hackathon](https://semianalysis.com/hackathon-2025/) on **Sunday, March 16th**, offering hands-on exploration of **Blackwell & PTX** infrastructure while collaborating on open-source projects.
- **Hackathon's Speaker Lineup Boasts AI Heavyweights**: The hackathon speakers include [Philippe Tillet of OpenAI](https://openai.com/), [Tri Dao of TogetherAI](https://www.together.ai/), [Horace He of Thinking Machines](https://www.thinkingmachin.es/), and more.
   - The event is sponsored by Together, Lambda, Google Cloud, Nvidia, GPU Mode, Thinking Machines, OpenAI, PyTorch, Coreweave, Nebius.
- **Kick off GTC with SemiAnalysis**: **SemiAnalysis** is kicking off **GTC** in style with the **Blackwell GPU Hackathon**, which includes engaging morning keynotes, all-day hacking with powerful **Blackwell GPUs** like **GB200s**, and insightful afternoon talks.



**Link mentioned**: <a href="https://semianalysis.com/hackathon-2025/">Hackathon 2025</a>: SemiAnalysis is kicking things off ahead of NVIDIA GTC! Start your day with engaging morning keynotes, hack all day with low-level NVIDIA GPU programming (maybe even Blackwell), take a breather wit…

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

clear3fram3: waiting for the last blog posts on CUDA to pop up 🙂
  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1349135915733356584)** (1 messages): 

> `Qdrant, Vector DB, ConvRAG, VPC Deployment` 


- **Qdrant Scrapped for ConvRAG**: The development teams considered **Qdrant** as the vector DB for their **ConvRAG** but decided to use a different one.
   - The selected DB offered greater flexibility for **VPC deployment**.
- **Alternative DB Chosen**: A different vector DB was chosen over Qdrant for **ConvRAG**.
   - The primary reason cited was the enhanced flexibility it provided for **VPC deployment** scenarios.


  

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

---
id: 72ea2fa2-ac11-4be2-90ca-e6963178858a
title: Ways to use Anthropic's Tool Use GA
date: '2024-05-31T20:31:29.874216Z'
original_slug: ainews-ways-to-use-anthropics-tool-use-ga
description: >-
  **Anthropic** launched general availability of tool use and function calling
  with support for streaming, forced use, and vision. Alex Albert shared **5
  architectures** for agentic tool use: delegation, parallelization, debate,
  specialization, and tool suite experts. **Claude 3 Opus** was used for AI
  Twitter recaps. **Yann LeCun** emphasized ethical open science funding,
  gradual emergence of superintelligence with safety guardrails, and
  convolutional networks combined with self-attention for image/video
  processing. He also noted growth in AI researchers across industry, academia,
  and government.
companies:
  - anthropic
  - amazon
  - google
  - openai
models:
  - claude-3-opus
topics:
  - function-calling
  - agent-architectures
  - streaming
  - vision
  - open-science
  - superintelligence
  - convolutional-networks
  - self-attention
  - ai-research-growth
people:
  - ylecun
  - sainingxie
  - alexalbert
---


<!-- buttondown-editor-mode: plaintext -->**Tools are all AIs need.**

> AI News for 5/30/2024-5/31/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**393** channels, and **2911** messages) for you. 
Estimated reading time saved (at 200wpm): **337 minutes**.

Together with [Anthropic's GA of tool use/function calling](https://x.com/AnthropicAI/status/1796210547077128578) today on Anthropic/Amazon/Google, with support for [streaming](https://x.com/alexalbert__/status/1791137394563133849), [forced use](https://x.com/alexalbert__/status/1791137396798677234), and [vision](https://x.com/alexalbert__/status/1791137398266659286)... 

 ![image.png](https://assets.buttondown.email/images/f1926fa7-897a-46e4-a5b0-70f36ee3a18f.png?w=960&fit=max) 

Alex Albert shared [5 architectures](https://x.com/alexalbert__/status/1796211969432887331) for using them in an agentic context:

1. **Delegation**: Use cheaper, faster models for cost and speed gains. 
  - For example, Opus can delegate to Haiku to read a book and return relevant passages. This works well if the task description & result are more compact than the full context.
2. **Parallelization**: Cut latency (but not cost) by running agents in parallel. 
  - e.g. 100 sub-agents each read a different chapter of a book, then return key passages. 
3. **Debate**: Multiple agents with different roles engage in discussion to reach better decisions.
  - For example, a software engineer proposes code, a security engineer reviews it, a product manager gives a user's view, then a final agent synthesizes and decides.
4. **Specialization**:  A generalist agent orchestrates, while specialists execute tasks.
  - For example, the main agent uses a specifically prompted (or fine-tuned) medical model for health queries or a legal model for legal questions.
5. **Tool Suite Experts**: When using 100s or 1000s of tools, specialize agents in tool subsets.
  - Each specialist (the same model, but with different tools) handles a specific toolset. The orchestrator then maps tasks to the right specialist (keeps the orchestrator prompt short).

Nothing particularly groundbreaking here but a very handy list to think about for patterns. Anthropic also [launched a self guided course on tool use](https://x.com/alexalbert__/status/1796610971810853165):

 ![image.png](https://assets.buttondown.email/images/26a939e9-7542-49ec-a460-7b68ec84591c.png?w=960&fit=max) 

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**AI Research and Development**

- **Open Science and Research Funding**: [@ylecun](https://twitter.com/ylecun/status/1796486618620051933) expressed a clear ethical rule for research: "Do not get research funding from entities that restrict your ability to publish." He emphasized that **making new knowledge available to the world is intrinsically good**, regardless of the funding source. [@ylecun](https://twitter.com/ylecun/status/1796488253815603279) noted that this ethical rule has made him a **strong advocate of open science and open source**.
- **Emergence of Superintelligence**: [@ylecun](https://twitter.com/ylecun/status/1796543960296440162) believes the emergence of superintelligence will be a **gradual process**, not a sudden event. He envisions starting with an architecture at the intelligence level of a rat or squirrel and **progressively ramping up its intelligence** while designing proper guardrails and safety mechanisms. The goal is to design an **objective-driven AI that fulfills goals specified by humans**.
- **Convolutional Networks for Image and Video Processing**: [@ylecun](https://twitter.com/ylecun/status/1796265384976252991) recommends using **convolutions with stride or pooling at low levels and self-attention circuits at higher levels** for real-time image and video processing. He believes [@sainingxie](https://twitter.com/sainingxie)'s work on ConvNext has shown that **convolutional networks can be just as good as vision transformers if done right**. [@ylecun](https://twitter.com/ylecun/status/1796263750485295560) argues that self-attention is equivariant to permutations, which is nonsensical for low-level image/video processing, and that **global attention is not scalable** since correlations are highly local in images and video.
- **AI Researchers in Industry vs. Academia**: [@ylecun](https://twitter.com/ylecun/status/1796518343194845658) noted that if a graph showed absolute numbers instead of percentages, it would reveal that the **numbers of AI researchers in industry, academia, and government have all grown**, with industry growing earlier and faster than the rest.

**AI Tools and Applications**

- **Suno AI**: [@suno_ai_](https://twitter.com/suno_ai_/status/1796273804991156326) announced the release of Suno v3.5, which allows users to **make 4-minute songs in a single generation**, create 2-minute song extensions, and experience improved song structure and vocal flow. They are also **paying $1 million to the top Suno creators in 2024**. [@karpathy](https://twitter.com/karpathy/status/1796305221813198946) expressed his love for Suno and shared some of his favorite songs created using the tool.
- **Claude by Anthropic**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1796210547077128578) announced that **tool use for Claude is now generally available** in their API, Amazon Bedrock, and Google Vertex AI. With tool use, Claude can **intelligently select and orchestrate tools to solve complex tasks end-to-end**. Early customers use Claude with tool use to build bespoke experiences, such as [@StudyFetch](https://twitter.com/StudyFetch) using Claude to power Spark.E, a personalized AI tutor. [@HebbiaAI](https://twitter.com/HebbiaAI) uses Claude to power complex, multi-step customer workflows for their AI knowledge worker.
- **Perplexity AI**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1796220011448786949) introduced Perplexity Pages, described as "AI Wikipedia," which allows users to **analyze sources and synthesize a readable page** with a simple "one-click convert." Pages are available for all Pro users and rolling out more widely to everyone. Users can create a page as a separate entity or **convert their Perplexity chat sessions into the page format**. [@perplexity_ai](https://twitter.com/perplexity_ai/status/1796203494401040846) noted that Pages lets users share in-depth knowledge on any topic with **formatted images and sections**.
- **Gemini by DeepMind**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1796216673348833445) announced that developers can now start building with **Gemini 1.5 Flash and Pro models** using their API pay-as-you-go service. Flash is designed to be **fast and efficient to serve**, with an increased rate limit of 1000 requests per minute.

**Memes and Humor**

- [@huybery](https://twitter.com/huybery/status/1796532108024000674) introduced im-a-good-qwen2, a **chatbot that interacts in the comments**.
- [@karpathy](https://twitter.com/karpathy/status/1796556328078619103) shared his opinion on 1-on-1 meetings, stating that he had around 30 direct reports at Tesla and **didn't do 1-on-1s, which he believes was great**. He finds 4-8 person meetings and large meetings for broadcast more useful.
- [@ReamBraden](https://twitter.com/ReamBraden/status/1796257883623145895) shared a **meme about the challenges of being a startup founder**.
- [@cto_junior](https://twitter.com/cto_junior/status/1796237607522758914) shared a **meme about Tencent AI developers working to replace underpaid anime artists**.
- [@nearcyan](https://twitter.com/nearcyan/status/1796245651174605032) made a humorous comment about people who believe we should not talk to animals, build houses, or power plants, and instead **"rot in caves and fight over scraps as god intended."**

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Image & Video Generation**

- **Photorealistic avatars**: In /r/singularity, impressive photorealistic avatars were showcased from the Neural Parametric Gaussian Avatars (NPGA) research at the University of Munich, Germany ([example 1](https://v.redd.it/o8rxrfnmpj3d1), [example 2](https://v.redd.it/drghzm0mrj3d1)). These high-quality avatars demonstrate the rapid advancements in AI-generated human-like representations.
- **Cartoon generation and interpolation**: The ToonCrafter model was introduced for generating and interpolating cartoon-style images, with [experiments showcasing its capabilities](https://v.redd.it/x1vda6as6m3d1). This highlights the expanding range of AI-generated content beyond photorealistic imagery.
- **AI-powered game development**: An [open-source game engine](https://v.redd.it/imlu0qd1tj3d1) was presented that leverages AI models like Stable Diffusion, AnimateDiff, and ControlNet to generate game assets and animations. The engine's [source code](https://github.com/WorldQL/dreamlab-core) and [techniques for rendering animated sprites](https://drive.google.com/file/d/1BPvC-PLF-__ey6KjmxTJZ_bg55oThHdr/view?usp=sharing) are fully available.
- **AI animation APIs**: The Animate Anyone API, with code available on [GitHub](https://i.redd.it/fuglsjleli3d1), enables the animation of people in images. However, comments suggest that alternatives like [MusePose](https://github.com/TMElyralab/MusePose) may offer better results.

**AI Ethics & Societal Impact**

- **AI partnerships and competition**: Microsoft CEO Satya Nadella expressed [concerns over a potential OpenAI-Apple deal](https://www.businessinsider.com/satya-nadella-sam-altman-openai-apple-microsoft-worried-about-deal-2024-5), highlighting the strategic importance of AI partnerships and the competitive landscape.
- **Deepfake concerns**: The growing potential for misuse of deepfake technology was [emphasized](https://i.redd.it/s19j6oq7kk3d1.jpeg), underscoring the need for safeguards and responsible AI practices.
- **AI in the film industry**: Sony's plans to [use AI for reducing film production costs](https://www.indiewire.com/news/breaking-news/sony-pictures-will-cut-film-costs-using-ai-1235010605/) raised questions about the impact on the creative industry and potential job displacement.
- **AI-generated content and realism**: An AI-generated image titled "All Eyes On Rafah" [faced criticism](https://www.ndtv.com/world-news/ai-generated-all-eyes-on-rafah-pic-criticised-for-being-removed-from-reality-5777680) for lacking realism and potentially misrepresenting a sensitive situation, highlighting the challenges of AI-generated content.
- **AI and influence campaigns**: OpenAI [reported](https://www.nytimes.com/2024/05/30/technology/openai-influence-campaigns-report.html?unlocked_article_code=1.v00.eijy.DJp94u_PuyG7) that Russia and China used its AI tools for covert influence campaigns, emphasizing the need for proactive measures against AI misuse, as detailed in their [efforts to combat deceptive AI use](https://openai.com/index/disrupting-deceptive-uses-of-AI-by-covert-influence-operations/).

**AI Capabilities & Advancements**

- **Bioprocessors and brain organoids**: A groundbreaking [bioprocessor utilizing human brain organoids](https://www.reddit.com/r/singularity/comments/1d4bcoa/worlds_first_bioprocessor_uses_16_human_brain/) was developed, offering highly efficient computation compared to digital chips.
- **AI in healthcare**: New AI technology was [shown to predict cardiac events related to coronary inflammation up to 10 years in advance](https://www.reddit.com/r/singularity/comments/1d4bc8w/new_ai_tech_predicts_cardiac_events_due_to/), based on a landmark study published in The Lancet.
- **Quantum computing breakthrough**: Chinese researchers, led by a US-returned physicist, [claimed to have built the world's most powerful ion-based quantum computer](https://www.scmp.com/news/china/science/article/3264742/us-returned-chinese-physicist-duan-luming-and-team-build-worlds-most-powerful-ion-based-quantum).

**OpenAI News & Developments**

- **Leadership clarification**: Paul Graham [clarified that Y Combinator did not fire Sam Altman](https://twitter.com/paulg/status/1796107666265108940), contrary to circulating rumors.
- **Robotics research revival**: OpenAI is [rebooting its robotics team](https://www.forbes.com/sites/kenrickcai/2024/05/30/openai-robotics-team/?sh=1f4e2d2c4f33), signaling a renewed focus on the intersection of AI and robotics.
- **Addressing concerns**: OpenAI board members [responded to warnings raised by former members](https://www.economist.com/by-invitation/2024/05/30/openai-board-members-respond-to-a-warning-by-former-members) regarding the company's direction and practices.
- **AI for nonprofits**: OpenAI [launched an initiative](https://openai.com/index/introducing-openai-for-nonprofits/) to make its tools more accessible to nonprofit organizations, promoting beneficial AI applications.
- **Partnership with Reddit**: The [announcement of a partnership between OpenAI and Reddit](https://openai.com/index/openai-and-reddit-partnership/) raised questions about the potential implications for both platforms.

**AI Humor & Memes**

- **Robots then and now**: A [humorous comparison](https://v.redd.it/ozsk6wqemp3d1) of the "I, Robot" movie's portrayal of robots in the past versus the present day was shared.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Model Performance Optimization and Benchmarking**

- **K2 Triumphs Over Llama 2**: The [K2 model from LLM360](https://huggingface.co/LLM360/K2) surpasses **Llama 2 70B** in performance while using 35% less compute, fully open-sourced under Apache 2.0 license.

- **NeurIPS Hosts Model Merging Competition**: A competition with an $8,000 prize invites contenders to blend optimal AI models, details available on the [NeurIPS Model Merging Website](https://llm-merging.github.io/).

- **Tailored Positional Embeddings Boost Transformer Arithmetic**: Researchers achieved **99% accuracy on 100-digit sums** using specific embeddings, detailed in their [paper](https://huggingface.co/papers/2405.17399).

**2. Fine-Tuning and Prompt Engineering**

- **Tackling Dataset Merging and Training Tips**: Axolotl users discussed effective merging datasets during fine-tuning to avoid issues like catastrophic forgetting. Recommended tools include **[Hugging Face Accelerate](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/config.qmd#L325C1-L327C27)**.

- **Fine-Tuning Techniques for Legal Draft Systems and Chatbots**: Users fine-tuning LLMs for applications like **legal drafts** and **financial document summarization** swapped strategies, with resources like [Fine-Tune PaliGemma](https://youtu.be/hDa-M91MSGU?si=4QKcNZsB40ibgPyd).

- **Resolving Training Issues for Text Classification Models**: Issues with training Spanish entity categorization models involved fine-tuning recommendations, exploring frameworks like **RoBERTa**.

**3. Open-Source AI Developments and Collaborations**

- **Milvus Lite for Efficient Vector Storage**: Introducing **Milvus Lite**, a lightweight solution for Python-focused vector storage, detailed in the [Milvus documentation](https://milvus.io/docs/milvus_lite.md).

- **MixMyAI Integrates Multiple AI Models on a Single Platform**: The [mixmyai.com](https://mixmyai.com) platform consolidates open and closed-source models, emphasizing privacy and avoiding server storage of chat data.

- **LlamaIndex Offers Flexible Retrieval Systems**: New Django-based web app templates facilitate **Retrieval Augmented Generation (RAG)** applications, utilizing data management and user access controls, as detailed [here](https://t.co/kx3DhxfDZu).

**4. AI Community Innovations and Knowledge Sharing**

- **Using Axolotl for Consistent Prompt Formats**: Adjustments in Axolotl were made to ensure prompt format consistency, guiding users to settings like **[Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/prompters.py#L47)**.
  
- **Challenges with Language Support in Ghost XB Beta**: Unsloth AI discussed multilingual support in models like **Ghost XB Beta** aiming for 9+ languages during training phases, highlighting [Ghost XB details](https://ghost-x.org).

- **Incorporating OpenAI and LangChain Tools**: Resources like [LangChain Intro](https://python.langchain.com/v0.2/docs/introduction/), and real-time features announced for GPT-4 Alpha, were discussed for creating advanced AI applications.

**5. Hardware Advancements and Compatibility Challenges**

- **NVIDIA's New 4nm Research Chip Impresses**: Achieving **96 int4 TOPs/Watt** efficiency, significantly outpacing Blackwell's capabilities, with discussion on impacts shared [here](https://x.com/danielhanchen/status/1796253349932843214).

- **ROCm Support Challenges on AMD GPUs**: Frustrations over ROCm's lack of support for GPUs like **RX 6700 and RX580** led to discussions on potential alternatives and performance impacts.

- **Implementing Efficient Data Handling in CUDA**: Discussions on optimizing CUDA operations, using techniques like fusing element-wise operations for better performance, with source code insights available [here](https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/memory/memory.mojo).

---

# PART 1: High level Discord summaries




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Transformer Trainings and Troubleshooting**: Interactive sessions on **Transformer architecture** were requested for better understanding of complex topics like **RoPE** and **RMS Normalization**. In the meantime, **Google Gemini Flash** will allow free fine-tuning come June 17th, while careful calculation of costs for production using **RAG LLMs** remains imperative, prioritizing GPU time and third-party services considerations. The **GGUF format** is being advocated for to maintain compatibility with ecosystem tools, easing the fine-tuning process. ([Fine-tune PaliGemma for image to JSON use cases](https://youtu.be/hDa-M91MSGU?si=4QKcNZsB40ibgPyd))

- **Braving BM25 and Retrieval Woes**: A hunt for a reliable **BM25 implementation** was launched, with the Python package `rank_bm25` in the crosshairs due to its limited functionality. The conversation focused on enhancing vector retrieval; meanwhile, **Modal** users are directed to [documentation](https://modal.com/docs/guide/trigger-deployed-functions) for next steps after deploying **v1 finetuned models**, and a **[Modal credits initiative](https://tally.so/r/wQ5GRl)** clarified expiration concerns.

- **Data Dominates Dialogue**: The parsing of structured information from document AI required techniques like OCR and LiLT models. In parallel, data processing for 5K scraped LinkedIn profiles was considered with OpenPipe and **GPT-4**, while multi-modal approaches and **document understanding** stayed hot topics. Emphasis on precise matching of training data file formats to prevent `KeyError: 'Input'` surfaced as a troubleshooting tip.

- **Learning LLM Legwork and LangChain Link-Up**: Resources from **Humble Bundle** and **Sebastian Raschka** offered insights into prompt engineering and LLM finetuning, though skepticism was raised about the quality of some materials. Reflecting the community's thirst for knowledge, O'Reilly released **Part II** of their series on building with LLMs, targeting operational challenges in LLM applications. 

- **Curating Conversational Context**: The distinction between instruct-LLM and chat-LLM models was dissected with the former following clear instructions, and the latter mastering conversational context. Projects discussed ranged from an **Alexa-like music player** to a **legal draft system** and a chatbot for financial document summaries, indicating the range of possible implementations for fine-tuned LLMs.

- **Modal Moves and Market Reach**: Mediums like blogs played a vital role in spreading knowledge, with **[John Whitaker's blog](https://johnowhitaker.dev)** becoming a go-to place for learning about things like *Basement Hydroponics* and LLM performance. More when practitioners shared gradient optimization tricks such as **[gradient checkpointing](https://discord.com/channels/1238365980128706560/1242223332695478332/1246108353210355805)** and agreed that sometimes, the simplest explanations, like those from Johno's sessions, resonate best.

- **Space for Spaces**: Queries on how to change prompt styles for the **alpaca format** in Axolotl and **Qwen tokenizer** usage issues were discussed, with references pointing to specific **[GitHub configs](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/src/axolotl/prompters.py#L47)**. Meanwhile, deploying a **Gradio-based RAG app** sparked interest in using **HF Spaces** due to its ease of use.

- **Credit Craze and Communal Connects**: Moments of panic and clarification underscored the urgency of filling out credit forms, as emphasized in urgent announcements. Social gatherings and discussions ranged from SF eval meetups to Modal Labs hosting office hours in NYC, indicating robust community connections and knowledge-sharing events.

- **Europe Engagement and Predibase Prospects**: Check-ins from across Europe, such as **Nuremberg, Germany** and **Milan, Italy**, manifested the group's geographical span. Elsewhere, the mention of a **[30-day free trial of Predibase](https://predibase.com/free-trial)** offering $25 in credits reflected ongoing efforts to provide accessible finetuning and deployment platforms.

- **Career Crossroads**: From academia to industry, members shared experiences and encouraged one another in career transitions. The discussion showcased contracting as a viable pathway, with mentorship and perseverance identified as crucial for navigating the tech landscape where GitHub portfolios can serve as vital stepping stones.

These summaries encapsulate the detailed, often granular discussions among AI Engineers in the Discord guild, highlighting the collective endeavor to optimize LLM fine-tuning and deployment amidst pursuit of career growth and community building.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**K2 Triumphs Over Llama 2**: [LLM360's K2 model](https://huggingface.co/LLM360/K2) outpaces **Llama 2 70B**, achieving better performance with 35% less computational effort; it's touted as **fully-reproducible** and is accessible under the Apache 2.0 license.

**Numbers Are No Match for Positional Embeddings**: Researchers cracked the nut on transformers' arithmetic abilities; with tailored positional embeddings, transformers reach a **99% accuracy** on 100-digit sums, a monumental feat outlined in their [paper](https://huggingface.co/papers/2405.17399).

**NeurIPS Throws Down the Merging Gauntlet**: With an $8,000 purse, the **NeurIPS Model Merging Competition** invites contenders to blend optimal AI models. Hugging Face, among others, sponsors this competition, more info in the [announcement](https://x.com/LChoshen/status/1796256513519989102) and [competition website](https://llm-merging.github.io/).

**Data Dive: From 150K Datasets to Clothing Sales**: A treasure trove of 150k+ datasets is now at engineers' fingertips for exploration with DuckDB, explained in a [blog post](https://huggingface.co/blog/chilijung/access-150k), while a novel clothing sales dataset propelled the development of an image regression model which was then detailed in [this article](https://huggingface.co/blog/tonyassi/image-regression).

**Learning Resources and Courses Amplify Skills**: In the perpetually advancing field of AI, engineers can bolster their expertise through Hugging Face courses in **Reinforcement Learning** and **Computer Vision**, with more information accessible at [Hugging Face - Learn](https://huggingface.co/learn).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Quantization Quandaries and High-Efficiency Hardware**: Unsloth AI guild members highlight challenges with the quantized **Phi3 finetune** results, noting performance issues without quantization tricks. NVIDIA's new *4nm research chip* is generating buzz with its 96 int4 tera operations per second per watt (TOPs/Watt) efficiency, overshadowing Blackwell's 20T/W and reflecting industry-wide advancements in power efficiency, numerical representation, Tensor Cores' efficiency, and sparsity techniques.

**Model Fine-Tuning and Upscaling Discussions**: AI engineers share insights on fine-tuning strategies, including **dataset merging**, with one member unveiling an **11.5B upscale model** of Llama-3 using upscaling techniques. An emerging fine-tuning method, **MoRA**, suggests a promising avenue for parameter-efficient updates.

**Troubleshooting Tools and Techniques**: Engineers confront various hurdles, from GPU selection in Unsloth (`os.environ["CUDA_VISIBLE_DEVICES"]="0"`) and troubleshooting fine-tuning errors to handling dual-model dependencies and addressing VRAM spikes during training. Workarounds for issues like Kaggle installation challenges underscore the need for meticulous problem-solving.

**AI in Multiple Tongues**: **Ghost XB Beta** garners attention for its capability to support 9+ languages fluently and is currently navigating through its training stages. This progress reaffirms the guild’s commitment to developing accessible, cost-efficient AI tools for the community, especially emphasizing startup support.

**Communal Cooperative Efforts and Enhancements**: Guild discussions reveal a collective push for self-deployment and community backing, with members sharing updates and seeking assistance across a spectrum of AI-related endeavors such as the *Open Empathic* project and Unsloth AI model improvements.





---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Tako Widgets Limited Geographic Scope?**: Discussion around the **Tako finance data widget** raised questions about its geographic limitations, with some users unsure if it's exclusive to the United States.

- **Perplexity Pro Trials End**: Users talked about the discontinuation of **Perplexity Pro trials**, including the yearly 7-day option, spurring conversations around potential referral strategies and self-funded trials.

- **Perplexity's Page-Section Editing Quirks**: Some confusion arose around editing sections on **Perplexity pages**, where users can alter section details but not the text itself – a limitation confirmed by multiple members.

- **Search Performance Trade-offs Noted**: There's been observance of a **slowdown in Perplexity Pro search**, attributed to a new strategy that sequentially breaks down queries, which, despite lower speeds, offers more detailed responses.

- **Exploring Perplexity's New Features**: Excitement was apparent as users shared links to newly introduced **Perplexity Pages** and discussions about **Codestral de Mistral**, hinting at enhancements or services within the Perplexity AI platform.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **NVIDIA and Meta's Chip Innovations Generate Buzz**: The community was abuzz with NVIDIA's revelation of a 4nm **inference chip achieving 96 int4 TOPs/Watt**, outperforming the previous 20T/W benchmark, and Meta unveiling a **next-generation AI accelerator** clocking in at 354 TFLOPS/s (INT8) with only 90W power consumption, signaling a leap forward in AI acceleration capabilities.

- **Deep Dive into CUDA and GPU Programming**: Enthusiasm surrounded the announcement of a **FreeCodeCamp CUDA C/C++ course** aimed at simplifying the steep learning curve of GPU programming. Course content requests emphasized the importance of covering GEMM for rectangular matrices and broadcasting rules pertinent to image-based convolution applications.

- **Making Sense of Scan Algorithms and Parallel Computing**: The community engaged in eager anticipation of the second part of a **scan algorithm** series. At the same time, questions were raised regarding practical challenges with parallel scan algorithms highlighted in the `Single-pass Parallel Prefix Scan with Decoupled Look-back` paper, as well as requests for clarification of CUDA kernel naming in Triton for improved traceability in kernel profiling.

- **Strategies for Model Training and Data Optimization Shared**: The conversation included a sharing of strategies on efficient homogeneous model parameter sharing to avoid inefficient replication during batching in PyTorch, and issues like loss spikes during model training which could potentially be diagnosed through gradient norm plotting. The idea of hosting datasets on Hugging Face was floated to facilitate access, with compression methods suggested to expedite downloads.

- **Cross-Platform Compatibility and Community Wins Celebrated**: Progress and challenges in extending CUDA and machine learning library compatibility to Windows were discussed, with acknowledgment of Triton's intricacies. Meanwhile, the community celebrated reaching 20,000 stars for a repository and shared updates on structuring and merging directories to enhance organization, strengthening the ongoing collaboration within the community.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Online Content Privacy Calls for Education**: A participant emphasized the importance of not publishing content as a privacy measure and stressed the need to educate people about the risks of providing content to companies.
  
- **Striving for Consistent AI Tool Results**: Users noted inconsistencies when using **ComfyUI** compared to **Forge**, suggesting that different settings and features such as **XFormers** might influence results, despite identical initial settings.

- **Strategies for Merging AI Models Discussed**: Conversations revolved around the potential of combining models like **SDXL** and **SD15** to enhance output quality, though ensuring consistent control nets across model phases remains crucial.

- **Custom AI Model Training Insights Shared**: Enthusiasts exchanged tips on training bespoke models, mentioning resources like **OneTrainer** and **kohya_ss** for Lora model training, and sharing helpful YouTube tutorials.

- **Beginner Resources for AI Exploration Recommended**: For AI newbies, starting with simple tools like [Craiyon](https://www.craiyon.com) was recommended to get a feel for image generation AI, before progressing to more sophisticated platforms.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**GPU Blues with ROCm? Not Music to Our Ears**: Engineers discussed GPU performance with ROCm, lamenting the lack of support for RX 6700 and old AMD GPUs like RX580, influencing token generation speeds and overall performance. Users seeking performance benchmarks on multi-GPU systems with models such as **LLAMA 3 8B Q8** reported a 91% efficiency with two GPUs compared to one.

**VRAM Envy**: The release of LM Studio models ignited debates on VRAM adequacy, where the 4070's 12GB was compared unfavorably to the 1070's 20GB, especially concerning suitability for large models like "codestral."

**CPU Constraints Cramp Styles**: CPU requirements for running LM Studio became a focal point, where **AVX2 instructions** proved mandatory, leading users with older CPUs to use a prior version (0.2.10) for AVX instead.

**Routing to the Right Template**: AI engineers shared solutions and suggestions for model templates, such as using Deepseek coder prompt template for certain models, and advised checking tokenizer configurations for optimal formatting with models like **TheBloke/llama2_7b_chat_uncensored-GGUF**.

**New Kids on the Block - InternLM Models**: Several **InternLM** models designed for Math and Coding, ranging from 7B to a mixtral 8x22B, were announced. Models such as [AlchemistCoder-DS-6.7B-GGUF](https://huggingface.co/lmstudio-community/AlchemistCoder-DS-6.7B-GGUF) and [internlm2-math-plus-mixtral8x22b-GGUF](https://huggingface.co/lmstudio-community/internlm2-math-plus-mixtral8x22b-GGUF) were highlighted among the latest tools available for AI engineers.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Speed Boost to Global API Requests**: OpenRouter has achieved a global reduction in latency, lowering request times by ~200ms, especially benefiting users from Africa, Asia, Australia, and South America by optimizing edge data delivery.

- **MixMyAI Launches Unified Pay-as-You-Go AI Platform**: A new service called mixmyai.com has been launched, consolidating open and closed-source models in a user-friendly interface that emphasizes user privacy and avoiding the storage of chats on servers.

- **MPT-7B Redefines AI Context Length**: The Latent Space podcast showcased MosaicML's MPT-7B model and its breakthrough in exceeding the context length limitations of GPT-3 as detailed in an [in-depth interview](https://www.latent.space/p/mosaic-mpt-7b?utm_source=substack&utm_medium=email).

- **Ruby Developers Rejoice with New AI Library**: A new [OpenRouter Ruby client library](https://github.com/OlympiaAI/open_router) has been released, along with updates to [Ruby AI eXtensions for Rails](https://github.com/OlympiaAI/raix-rails), essential tools for Ruby developers integrating AI into their applications.

- **Server Stability and Health Checks Called Into Question**: OpenRouter users confronted sporadic 504 errors across global regions, with interim solutions provided and discussions leaning towards the need for a dedicated health check API for more reliable status monitoring.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Pro Privileges Propel Chat Productivity**: Pro users of OpenAI now enjoy **enhanced capabilities** such as **higher rate limits**, and exclusive GPT creation, along with access to **DALL-E** and real-time communication features. The alluring proposition maintains its charm despite the $20 monthly cost, marking a clear divide from the limited toolkit available to non-paying users.

**AI Framework Favorites Facilitate Functional Flexibility**: The **Chat API** is recommended over the Assistant API for those developing AI personas with idiosyncratic traits, as it offers superior command execution without surplus functionalities such as file searching.

**Bias Brouhaha Besieges ChatGPT**: A suspension due to calling out perceived racism in ChatGPT's outputs opened up a forum of contention around **inherent model biases**, spotlighting the relentless pursuit of attenuating such biases amidst the ingrained nuances of training data.

**Virtual Video Ventures Verified**: **Sora and Veo** stand as subjects of a speculative spree as the guild contemplates the curated claims and practical potency of the pioneering video generation models, juxtaposed against the realities of AI-assisted video crafting.

**API Agitations and Advancements Announced**: Persistent problems presented by **memory leaks** causing lag and browser breakdowns mar the ChatGPT experience, triggering talks on tactical chat session limits and total recall of past interactions to dodge the dreariness of repetition. Meanwhile, the anticipated arrival of **real-time voice and visual** features in GPT-4 has been slated to debut in an Alpha state for a select circle, broadening over subsequent months as per [OpenAI's update](https://help.openai.com/en/articles/8400625-voice-chat-faq).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**NeurIPS Competition: Merge Models for Glory and Cash**: NeurIPS will host a **Model Merging competition** with an $8K prize, sponsored by Hugging Face and Sakana AI Labs—seeking innovations in model selection and merging. Registration and more info can be found at [llm-merging.github.io](https://llm-merging.github.io/) as announced on [Twitter](https://x.com/LChoshen/status/1796256513519989102).

**AI's Quest to Converse with Critters**: A striking $500K **Coller Prize** is up for grabs for those who can demystify communication with animals using AI, sparking excitement for potential breakthroughs ([info](https://coller-dolittle-24.sites.tau.ac.il/)). This initiative echoes Aza Raskin's Earth Species Project, aiming to untangle interspecies dialogue ([YouTube video](https://www.youtube.com/watch?v=rjvsl0mhqTk)).

**Puzzling Over Preference Learning Paradox**: The community is abuzz after a [tweet](https://x.com/_angie_chen/status/1796220428345573399) highlighted unexpected limitations in RLHF/DPO methods—preference learning algorithms are not consistently yielding better ranking of preferred responses, challenging conventional wisdom and suggesting a potential for overfitting.

**LLMs Reigning Over Real-Time Web Content**: A revelation for web users: **LLMs** are often churning out web pages in real-time, rendering what you see as it loads. This routine faces hiccups with lengthy or substantial pages due to context constraints, an area ripe for strategic improvements.

**Google Enhances AI-Driven Search**: Google has upgraded its AI Overviews for US search users, improving both satisfaction and webpage click quality. Despite some glitches, they're iterating with a feedback loop, detailed in their blog post – [AI Overviews: About last week](https://blog.google/products/search/ai-overviews-update-may-2024/).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Milvus Lite Elevates Python Vector Databases**: The introduction of **Milvus Lite** offers a lightweight, efficient vector storage solution for Python, compatible with AI development stacks like LangChain and LlamaIndex. Users are encouraged to integrate **Milvus Lite** into their AI applications for resource-constrained environments, with instructions available [here](https://milvus.io/docs/milvus_lite.md).

- **Crafting Web Apps with Omakase**: A new Django-based web app template facilitates the building of scalable Retrieval Augmented Generation (RAG) applications, complete with RAG API, data source management, and user access control. The step-by-step guide can be found [here](https://t.co/kx3DhxfDZu).

- **Navigating Data Transfer for Retrieval Systems**: For those prototyping retrieval systems, the community suggests creating an "IngestionPipeline" to efficiently handle data upserts and transfers between SimpleStore classes and RedisStores.

- **Complexities in Vector Store Queries Assessed**: The functionalities of different vector store query types like `DEFAULT`, `SPARSE`, `HYBRID`, and `TEXT_SEARCH` in PostgreSQL were clarified, with the consensus that both `text` and `sparse` queries utilize `tsvector`.

- **Troubleshooting OpenAI Certificate Woes**: Addressing SSL certificate verification issues in a Dockerized OpenAI setup, it was recommended to explore alternative base Docker images to potentially resolve the problem.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Luxia Language Model Contamination Alert**: **Luxia 21.4b v1.2** has been reported to show a 29% increase in contamination on GSM8k tests over v1.0, as detailed in a [discussion on Hugging Face](https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.2/discussions/1), raising concerns about benchmark reliability.

- **Ready, Set, Merge!: NeurIPS Model Merging Showdown**: A prize of **$8K** is up for grabs in the [NeurIPS 2023 Model Merging Competition](https://llm-merging.github.io/), enticing AI engineers to carve new paths in model selection and merging.

- **Cutting-Edge CLIP Text Encoder and PDE Solver Paradigm Shifts**: Recognition is given for advancements in the CLIP text encoder methodology through pretraining, as well as the deployment of **Poseidon**, a new model for PDEs with sample-efficient and accurate results, highlighting papers on [Jina CLIP](http://arxiv.org/abs/2405.20204) and [Poseidon](https://arxiv.org/abs/2405.19101).

- **Softmax Attention's Tenure in Transformers**: A debate has crystallized around the necessity of softmax weighted routing in transformers, with some engineers suggesting longstanding use trumps the advent of nascent mechanisms like "function attention" that retain similarities to existing methodologies.

- **A Reproducibility Conundrum with Gemma-2b-it**: Discrepancies emerged in attempts to replicate Gemma-2b-it's 17.7% success rate, with engineers turning to a [Hugging Face forum](https://huggingface.co/google/gemma-2b-it/discussions/44) and a [Colab notebook](https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb#scrollTo=cXoCKMi9EXir) for potential solutions, while results for Phi3-mini via lm_eval have proven more aligned with expected outcomes.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Rising: Package Management and Compiler Conundrums**: The **Mojo** community is awaiting updates on a proposed package manager, as per [Discussion #413](https://github.com/modularml/mojo/discussions/413) and [Discussion #1785](https://github.com/modularml/mojo/discussions/1785). The recent nightly Mojo compiler version `2024.5.3112` brought fixes and feature changes, outlined in the [raw diff](https://github.com/modularml/mojo/compare/8ae83916ebc7b3134948f466c0f56ee3e5569062...3df5fb4f9d3dd7cc5018aa8a160c3714b1a4f81e) and current [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

- **Ready for Takeoff: Community Meetings and Growth**: The Mojo community looks forward to the next meeting featuring interesting talks on various topics with details available in the [community doc](https://modul.ar/community-meeting-doc) and participation through the [Zoom link](https://modul.ar/community-meeting-zoom.).

- **The Proof is in the Pudding: Mojo Speeds and Fixes**: A YouTube video demonstrates a significant speedup by porting K-Means clustering to Mojo, detailed [here](https://www.youtube.com/watch?v=3bg5YBCcuWA). The discovery of a bug in `reversed(Dict.items())` which caused flaky tests was rectified with a PR found [here](https://github.com/modularml/mojo/pull/2896).

- **STEMming the Learning Curve: Educational Resources for Compilers**: For learning about compilers, an extensive syllabus has been recommended, available [here](https://mcyoung.xyz/syllabus).

- **Stringing Performance Together**: A more efficient string builder is proposed to avoid memory overhead and the conversation inclines toward zero-copy optimizations along with using [`iovec` and `writev`](https://man7.org/linux/man-pages/man2/writev.2.html) for better memory management.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Public Access for LangChainAPI**: A request was made for a method to expose **LangChainAPI** endpoints publicly for **LangGraph** use cases, with an interest in utilizing **LangServe** but awaiting an invite.

- **LangGraph Performance Tuning**: Discussions around optimizing **LangGraph** configurations focused on reducing load times and increasing the start-up speed of agents, indicating a preference for more efficient processes.

- **Memory and Prompt Engineering in Chat Applications**: Participants sought advice on integrating summaries from "memory" into `ChatPromptTemplate` and combining `ConversationSummaryMemory` with `RunnableWithMessageHistory`. They shared tactics for summarizing chat history to manage the token count effectively, alongside relevant [GitHub resources](https://github.com/langchain-ai/langchain/issues/16525) and [LangChain documentation](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/#dict-with-single-key-for-all-messages-input-messages-output).

- **LangServe Website Glitch Reported**: An error on the **LangServe website** was reported, together with sharing a [link to the site](https://www.langchain.com/langserve) for further details.

- **Prompt Crafting with Multiple Variables**: Queries were made on how to structure prompts with several variables from the **LangGraph** state, providing a formulated prompt example and inquiries about variable insertion timing.

- **Community Projects and Tool Showcases**: In the community work sphere, two projects were highlighted: a **YouTube tutorial** on creating custom tools for agents ([Crew AI Custom Tools Basics](https://youtu.be/Hc5yiUQKh2Q)), and an AI tool named **AIQuire** for document insights, which is available for feedback at [aiquire.app](https://aiquire.app).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Fineweb Fires Up Image-Text Grounding**: A promising approach called **Fineweb** utilizes Puppeteer to scrape web content, extracting images with text for VLM input—offering novel means for grounding visual language models. [See the Fineweb discussion here](https://vxtwitter.com/iurimatias/status/1796260746310910309).

- **StabilityAI's Selective Release Stirs Debate**: StabilityAI's decision to only release a 512px model, leaving the full suite of SD3 checkpoints unpublished, incites discussion among members on how this could influence future model improvements and resource allocation.

- **Positional Precision**: There's technical chatter regarding how positional embeddings in **DiT models** may lead to mode collapse when tackling higher resolution images, despite their current standard uses.

- **Open-Source Jubilation**: The open-source project **tooncrafter** excites the community with its potential, although minor issues are being addressed, showcasing a communal drive towards incremental advancement.

- **Yudkowsky's Strategy Stirs Controversy**: Eliezer Yudkowsky's institute published a "2024 Communication Strategy" advocating for a halt in AI development, sparking diverse reactions amongst tech aficionados. [Delve into the strategy](https://x.com/drtechlash/status/1796562490232557658?s=46&t=M3cR_nfDo7QCuM4xOvwNFA).

- **Merging Models at NeurIPS**: NeurIPS is hosting a Model Merging competition with an $8K prize to spur innovation in LLMs. Interested participants should visit the [official Discord](https://discord.gg/dPBHEVnV) and [registration page](https://llm-merging.github.io/).

- **RB-Modulation for Aesthetic AI Artistry**: The **RB-Modulation** method presents a novel way to stylize and compose images without additional training, and members can access the [project page](https://rb-modulation.github.io/), [paper](https://arxiv.org/abs/2405.17401), and soon-to-be-released [code](https://github.com/LituRout/RB-Modulation).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Yuan2 Model Sparks Community Interest**: Members shared insights on the [Yuan2 model](https://huggingface.co/papers/2311.15786) on Huggingface, highlighting a keen interest in examining its training aspects.

- **Training Techniques Face-Off**: Detailed discussions compared various preference training methodologies, spotlighting the *ORPO method*, which was suggested to supersede SFT followed by DPO, due to its "stronger effect". Supporting literature was referenced through an [ORPO paper](https://arxiv.org/abs/2403.07691).

- **Challenging Model Fine-Tuning**: Concerns emerged over struggles in fine-tuning models like llama3 and mistral for Spanish entity categorization. One instance detailed issues with model inference after successful training.

- **Members Seek and Offer Tech Aid**: From installation queries about **Axolotl** and CUDA to configuring an early stopping mechanism using **Hugging Face Accelerate library** for overfitting issues, the guild members actively sought and rendered technical assistance. Shared resources included the [axolotl documentation](https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L102L137) and the [early stopping configuration guide](https://github.com/OpenAccess-AI-Collective/axolotl/blob/d4f6c65e4c4d6899a9283a046fa922b937a0ab25/docs/config.qmd#L325C1-L327C27).

- **Axolotl Configuration Clarifications**: There was an advisory exchange regarding the proper configuration of the `chat_template` in Axolotl, recommending automatic handling by Axolotl to manage Alpaca formatting with LLama3.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **DiscoLeo Caught in Infinite Loop**: The incorporation of **ChatML into DiscoLeo** has resulted in an End Of Sentence (EOS) token issue, causing a 20% chance of the model entering an endless loop. Retraining DiscoLeo with ChatML data is proposed for resolution.

- **ChatML Favored for Llama-3 Template**: German finetuning has shown a preference for using **ChatML** over the **Llama-3 instruct template**, especially when directed towards models like Hermes Theta that are already ChatML-based.

- **IBM’s “Granite” Models Spark Curiosity**: Engineers are exploring IBM's **Granite models**, including **Lab version**, **Starcode-based variants**, and **Medusa speculative decoding**, with resources listed on [IBM’s documentation](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=models-foundation) and [InstructLab Medium article](https://medium.com/@syeda9118/instructlab-ever-imagined-the-ease-of-tuning-pre-trained-llms-3331ccea8d88).

- **Merlinite 7B Pit Against Granite**: The **Merlinite 7B model** has garnered attention for its proficiency in German, vying for comparison with the IBM Granite models tracked under the **Lab method**.

- **Quality Concerns Over AI-Generated Data**: The community indicated dissatisfaction with the quality of **AI-generated data**, illustrated by sub-par results in benchmarks like **EQ Bench on q4km gguf quants**, and showed interest in new strategies to enhance models without catastrophic forgetting.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Google's Expansion Raises Eyebrows**: A [tweet](https://x.com/_arohan_/status/1796228607255396423) has hinted at Google bolstering its compute resources, sparking speculation over its implications for AI model training capacities.
  
- **OpenAI Puts Robotics Back on Table**: OpenAI reboots its robotics efforts, now hiring research engineers, as reported via [Twitter](https://x.com/sarahnemerson/status/1796264432215146807?s=46) and in a Forbes article, marking a significant re-entry into robotics since 2020.

- **Confusion Clouds GPT-3.5's API**: Community members expressed frustration with the confusing documentation and availability narrative around GPT-3.5; some pointed out discrepancies in timelines and the inconvenience caused by deleted technical documentation.

- **Sergey's Call to Arms for Physical Intelligence**:
Nathan Lambert relayed Sergey's recruitment for a project in physical intelligence, signaling opportunities for those with interest in Reinforcement Learning (RL) to contribute to practical robot utilization.

- **'Murky Waters in AI Policy' Session Served Hot**: The latest episode of the [Murky Waters in AI Policy](https://retortai.com/episodes/murky-waters-in-ai-policy) podcast dishes out discussions on California's controversial 1047 bill and a rapid-fire roundup of recent OpenAI and Google mishaps. Nathan Lambert missed the open house for the bill, details of attendance or reasons were not provided.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Pioneering AI Sustainability**: Cohere was noted for prioritizing **long-term sustainability** over immediate grand challenges, with a focus on concrete tasks like information extraction from invoices.
- **AGI Still Room to Grow**: Within the community, it's agreed that the journey to AGI is just starting, and there's a continuous effort to understand what lies beyond the current "CPU" stage of AI development.
- **Enhanced Server Experience with Cohere**: The server is undergoing a makeover to simplify channels, add new roles and rewards, replace server levels with "cohere regulars," and introduce **Coral the AI chatbot** to enhance interactions.
- **Express Yourself with Custom Emojis**: For a touch of fun and to improve interactions, the server will incorporate new emojis, with customization options available through moderator contact.
- **Feedback Wanted on No-Code AI Workflows**: A startup is looking for insights on their **no-code workflow builder** for AI models, offering a **$10 survey incentive**—they're curious why users might not return after the first use.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Adapter Layers Bridge the Gap**: Engineers are exploring *embedding adapters* as a means to improve retrieval performance in AI models, with evidence showcased in a [Chroma research report](https://research.trychroma.com/embedding-adapters). The effectiveness of these can be likened to Froze Embeddings, which the Vespa team employs to eliminate frequent updates in dynamic systems ([Vespa's blog insights](https://blog.vespa.ai/leveraging-frozen-embeddings-in-vespa-with-sentence-transformers/)).

**ChatGPT Goes Corporate with PwC**: The acquisition of ChatGPT Enterprise licenses by PwC for roughly 100,000 employees sparked debates around the estimated value of $30M/year, with member guesses on the cost per user ranging from $8 to $65 per month.

**Google's Twin Stars: Gemini 1.5 Flash & Pro**: Release updates for Google Gemini 1.5 Flash and Pro have been pushed to general availability, introducing enhancements such as increased RPM limits and JSON Schema mode ([Google developers blog post](https://developers.googleblog.com/en/gemini-15-pro-and-15-flash-now-available/)).

**TLBrowse Joins the Open Source Universe**: TLBrowse, melding Websim with TLDraw, was open-sourced, allowing users to conjure up infinite imagined websites on @tldraw canvas, with access to a [free hosted version](https://tlbrowse.com).



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Literary Worlds in Your Browser**: Rosebud AI gears up for their *"Book to Game"* game jam, inviting participants to build games from literary works with **Phaser JS**. The event offers a **$500 prize** and runs until July 1st, details available through their [Twitter](https://x.com/Rosebud_AI/status/1796273820044595368) and [Discord](https://discord.gg/rosebud-ai).

- **Navigating the Digital Terrain**: A new guild member expressed difficulties in using the platform on Android, describing the experience as *"glitchy and buggy"*. They also sought help with changing their username to feel more at home within the virtual space.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Check the Pins for Manufacturing Updates**: It's crucial to stay on top of the **manufacturing updates** in the OpenInterpreter community; make sure to check the pinned messages in the **#general** channel for the latest information.
- **Codestral Model Sparks Curiosity**: Engineers have shown interest in the **Codestral model** with queries about its efficiency appearing in multiple channels; however, user experiences have yet to be shared. Additionally, it's noted that Codestral is restricted to non-commercial use.
- **Combating Integration Challenges**: There's a shared challenge in integrating **HuggingFace** models with OpenInterpreter, with limited success using the `interpreter -y` command. Engineers facing these issues are advised to seek advice in the technical support channel.
- **Scam Alert Issued**: Vigilance is essential as a "red alert" was issued about a potential scam within the community. No further details about the scam were provided.
- **Android Functionality Discussions Ongoing**: Members are engaged in discussions regarding **O1 Android** capability, specifically around installation in **Termux**, although no conclusive responses have been observed yet.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile Joins Forces with AutoGPT**: AutoGPT member announced a **collaboration** to weave **Llamafile** into their system, expanding the tool's reach and capabilities.
- **Inquiry into Content Blocks**: Queries were made about whether **Llamafile** can handle **content blocks** within messages, seeking parity with an OpenAI-like feature; similar clarity was sought for **llama.cpp**'s capabilities in this domain.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Netflix PRS Event Gathering AI Enthusiasts**: AI professionals are buzzing about the **PRS event at Netflix**, with multiple members of the community confirming attendance for networking and discussions.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Mistral 45GB Model Composition Speculations**: Interest is brewing around the **Mistral 45GB model's** language distribution, with a hypothesis suggesting a strong bias towards English and a smaller presence of programming languages.

- **Codestral Compliance Conundrum**: The community is engaging with the intricacies of the **Mistral AI Non-Production License (MNPL)**, finding its restrictions on sharing derivative or hosted works underwhelming and limiting for **Codestral** development.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TensorFlow vs. PyTorch Debate Continues**: A user named helplesness asked why **TensorFlow** might be considered better than **PyTorch**, sparking a comparison within the community. The discussion did not provide an answer but is indicative of the ongoing preference debates among frameworks in the AI engineering world.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1245816528025288819)** (86 messages🔥🔥): 

- **Guest Session on Transformer's Internals Requested**: A member requested a session discussing Transformer architecture topics like vanilla transformer, RoPE, RMS Normalization, and more. Video resources on these topics were shared, but the member emphasized the need for interactive sessions for Q&A.
  
- **Google's Gemini Flash to Support Fine-Tuning**: Starting June 17th, Google will allow free fine-tuning on Gemini Flash, with inference costs matching the base model rates. This was highlighted as a cost-effective opportunity for fine-tuning.

- **Cost Management for Production-Level Systems**: There was an exchange about calculating the costs for production systems using RAG LLMs, focusing on GPU time utilization and third-party service trade-offs. The discussion emphasized the importance of experimenting with compute platforms and managing expectations based on usage scenarios.

- **GGUF Format for Fine-Tuning Models**: GGUF format was recommended for fine-tuning LLMs to ensure compatibility with various tools in the ecosystem. A link to a detailed blog post on fine-tuning and inference steps was shared along with an update that Hugging Face is working on easier HF to GGUF conversions.

- **Document AI Developments**: Multiple users discussed their experiences and challenges with document processing, such as invoices and utility bills. Techniques like OCR, LiLT models, segmentation, and using multimodal/multimodal-less approaches for extracting structured information were shared, along with links to resources and related papers.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ankur-singh.github.io/blog/finetune-inference">Run your finetuned LLM with Ollama</a>: Complete workflow demonstrating how to finetune an LLM on your data and run it using Ollama.</li><li><a href="https://arxiv.org/abs/2401.00908">DocLLM: A layout-aware generative language model for multimodal document understanding</a>: Enterprise documents such as forms, invoices, receipts, reports, contracts, and other similar records, often carry rich semantics at the intersection of textual and spatial modalities. The visual cues...</li><li><a href="https://huggingface.co/blog/idefics2">Introducing Idefics2: A Powerful 8B Vision-Language Model for the community</a>: no description found</li><li><a href="https://youtu.be/hDa-M91MSGU?si=4QKcNZsB40ibgPyd">Fine-tune PaliGemma for image to JSON use cases</a>: In this tutorial, I&#39;ll showcase how to fine-tune PaliGemma, a new open vision-language model by Google on a receipt image to JSON use case. The goal for the ...</li><li><a href="https://www.cbc.ca/news/canada/manitoba/facebook-customer-support-scam-1.7219581">Winnipeg man caught in scam after AI told him fake Facebook customer support number was legitimate | CBC News</a>: A Winnipeg man who says he was scammed out of hundreds of dollars when he called what he thought was a Facebook customer support hotline wants to warn others about what can go wrong. </li><li><a href="https://huggingface.co/spaces/openbmb/MiniCPM-Llama3-V-2_5">MiniCPM-Llama3-V-2 5 - a Hugging Face Space by openbmb</a>: no description found</li><li><a href="https://youtu.be/Mn_9W1nCFLo?si=SWUPvbQ9ZCAxmAK_">LLaMA explained: KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU</a>: Full explanation of the LLaMA 1 and LLaMA 2 model from Meta, including Rotary Positional Embeddings, RMS Normalization, Multi-Query Attention, KV-Cache, Grou...</li><li><a href="https://youtu.be/UiX8K-xBUpE?si=UgGM6oimKVhvub-b">Mistral / Mixtral Explained: Sliding Window Attention, Sparse Mixture of Experts, Rolling Buffer</a>: In this video I will be introducing all the innovations in the Mistral 7B and Mixtral 8x7B model: Sliding Window Attention, KV-Cache with Rolling Buffer, Pre...</li><li><a href="https://youtu.be/bCz4OMemCcA?si=X5lnwL_cmE16XFFS">Attention is all you need (Transformer) - Model explanation (including math), Inference and Training</a>: A complete explanation of all the layers of a Transformer Model: Multi-Head Self-Attention, Positional Encoding, including all the matrix multiplications and...</li><li><a href="https://github.com/huggingface/transformers/pull/30928">FEAT / Trainer: Experimental feature - add `GGUF` conversion when pushing the model to Hub by younesbelkada · Pull Request #30928 · huggingface/transformers</a>: What does this PR do? Introduces a new quantization_config that is intended to be used only for trainer.push_to_hub(), it calls ` a GGUF conversion Space under the hood - (for now: https://huggingf...</li><li><a href="https://arxiv.org/abs/2405.20245">Retrieval Augmented Structured Generation: Business Document Information Extraction As Tool Use</a>: Business Document Information Extraction (BDIE) is the problem of transforming a blob of unstructured information (raw text, scanned documents, etc.) into a structured format that downstream systems c...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1245926979329458237)** (10 messages🔥): 

- **User offers help with legal draft system**: A member expressed willingness to assist another user with their "lgeal draft system". No further details were provided on the system or assistance needed.
- **Alexa-like music player proposal**: A member queried if fine-tuning would be suitable for a product resembling Alexa but not locked to Amazon Music. They suggested using **LangGraph + function calling** to interact with various music service APIs like YouTube and Spotify.
- **Chatbot for financial document summaries**: A member outlined a project to develop a chatbot capable of answering complex financial questions by summarizing financial documents. They indicated the necessity of **RAG** and a form of **PO** for generating user-preferenced summaries.
- **Fine-tuning LLM for Cypher/GQL translation**: One user intends to fine-tune an LLM to translate natural language questions into Cypher/GQL. They noted that this could greatly enhance interaction with graph data.
- **Discussion on instruct-LLM vs. chat-LLM**: An extensive discussion debated the models' distinctions, focusing on the training and evaluation differences. Users noted that while newer models blur these lines, **instruct models** follow clear instructions and **chat models** handle conversational context.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/)** (1 messages): 

blaine.wishart: Hi everyone...I'm on Hainan for the next 3 months.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1245842938412662884)** (18 messages🔥): 

- **Deployed model next steps**: After deploying the v1 finetuned model, a member sought guidance on using it further. [This documentation](https://modal.com/docs/guide/trigger-deployed-functions) was suggested to understand usage and invoking deployed functions on the Modal platform.
- **Modal credits hiccups resolved**: Various issues and queries about Modal credits were tackled, like a user noting lost credits and another inquisitive about expiration. Modal credits expire after one year, but active users can contact support to potentially roll over credits, and users doing academic research or involved in startups can get additional [credits](https://tally.so/r/wQ5GRl).
- **Troubleshooting dataset issues**: A member faced a "KeyError: 'Input'" while working with a specific training data file. It was recommended to check the dataset's format consistency and ensure the correct field keys match what's defined in the [config](https://github.com/modal-labs/llm-finetuning/blob/f64c8d7ea5ac46f67251801b05d52b933228db50/config/codellama.yml#L17-L20).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tally.so/r/wQ5GRl">Modal for Startups &amp; Academics</a>: Made with Tally, the simplest way to create forms.</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/f64c8d7ea5ac46f67251801b05d52b933228db50/config/codellama.yml#L17-L20">llm-finetuning/config/codellama.yml at f64c8d7ea5ac46f67251801b05d52b933228db50 · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://modal.com/docs/guide/trigger-deployed-functions">Invoking deployed functions</a>: Modal lets you take a function created by a deployment and call it from other contexts.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1245837657582600243)** (9 messages🔥): 

- **AI-Coding Humble Bundle Alert**: A member informed the channel about an AI-coding [humble bundle](https://www.humblebundle.com/software/complete-chatgpt-anthropic-gemini-prompt-engineering-api-and-programming-mega-bundle-software?mcID=102:66576d20c5895a1aa5046052:ot:5ccaf0c3db76615eab12deb2:1&linkID=66576d225fb588c450040093&utm_campaign=2024_05_30_completechatgptanthropicgeminipromptengineeringapiandprogramming_softwarebundle&utm_source=Humble+Bundle+Newsletter&utm_medium=email), but expressed skepticism about the content quality, noting *"First skim does not look great."* Another member added that generating such materials independently could be cheaper.
  
- **Sebastian Raschka’s Chapter on LLM Fine-Tuning**: A link to Sebastian Raschka’s [chapter](https://livebook.manning.com/book/build-a-large-language-model-from-scratch/chapter-6/v-7/10) on finetuning LLMs for classification in his upcoming book was shared, outlining topics like different finetuning approaches, dataset preparation, and accuracy evaluation for spam classification.

- **O'Reilly Releases Part II on LLM Building**: Following positive feedback on Part I, O'Reilly fast-tracked the release of [Part II](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/) of their series on building with LLMs, shifting focus from tactical to operational aspects of building LLM applications, and noting challenges worth addressing.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/">What We Learned from a Year of Building with LLMs (Part II)</a>: no description found</li><li><a href="https://livebook.manning.com/book/build-a-large-language-model-from-scratch/chapter-6/v-7/10">6 Finetuning for Classification · Build a Large Language Model (From Scratch)</a>: Introducing different LLM finetuning approaches · Preparing a dataset for text classification · Modifying a pretrained LLM for finetuning · Finetuning an LLM to identify spam messages · Evaluating the...</li><li><a href="https://www.humblebundle.com/software/complete-chatgpt-anthropic-gemini-prompt-engineering-api-and-programming-mega-bundle-software?mcID=102:66576d20c5895a1aa5046052:ot:5ccaf0c3db76615eab12deb2:1&linkID=66576d225fb588c450040093&utm_campaign=2024_05_30_completechatgptanthropicgeminipromptengineeringapiandprogramming_softwarebundle&utm_source=Humble+Bundle+Newsletter&utm_medium=email">The Complete ChatGPT, Anthropic, Gemini Prompt Engineering, API, and Programming Mega Bundle</a>: AI is rising—rise along with it with these online courses! Learn prompt engineering, LangChain, & more! Your purchase helps the Children’s Miracle Network.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1245844758509912105)** (4 messages): 

- **Langsmith-HIPAA Compatibility Inquiry**: A member inquired whether Langsmith offers paid plans supporting HIPAA environments, mentioning the need for handling PII/PHI securely and the necessity of a Business Associate Agreement (BAA) in place.
  
- **Langsmith Compatibility with OpenAI Models**: Another user asked if Langsmith can be used with OpenAI-compatible models like Mixtral, or any models following the same API standards, such as Anthropic.

- **Langsmith Connecting to various Models**: Lucas shared insights on using Langchain and Langsmith with Meta's Llama-3:8b via Ollama and highlighted Langchain’s integration with Together AI. The detailed steps and code snippets for using Together AI can be found in Lucas's [blog post](https://lucasvw.github.io/posts/20_ollama_langchain/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/llms/together/">Together AI | 🦜️🔗 LangChain</a>: Together AI offers an API to query 50+ leading open-source models in a couple lines of code.</li><li><a href="https://lucasvw.github.io/posts/20_ollama_langchain/">Lucas van Walstijn - Having fun with llama3:8b</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[kylecorbitt_prompt_to_model](https://discord.com/channels/1238365980128706560/1242221891733946490/1245869118108729494)** (1 messages): 

- **Seeking advice on processing LinkedIn profile data**: A user asked for the best approach to processing data from 5K scraped LinkedIn profiles with 20+ columns. They aim to build a fine-tuned model to generate personalized introduction lines using OpenPipe with GPT-4, and later fine-tune a llama-3-8b model.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/1245889210288836649)** (3 messages): 

- **Deck from the talk now accessible**: A member asked if the deck from a recent talk was available. Another provided a [Discord link](https://discord.com/channels/1238365980128706560/1242223275463938221/1245439203207286825) that also includes a link to the slides.
  
- **Insightful prompt crafting resource shared**: A member highlighted [ExplainPrompt](https://www.explainprompt.com/) as a valuable resource. The site is maintained by a GitHub colleague who posts summaries and visual guides about prompt crafting techniques based on the latest papers.

**Link mentioned**: <a href="https://www.explainprompt.com/">ExplainPrompt</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[whitaker_napkin_math](https://discord.com/channels/1238365980128706560/1242223332695478332/1246108171018440774)** (268 messages🔥🔥): 

- **Begin your LLM adventure on Johno's blog**: Members were excited to share John Whitaker's valuable content. Check out his [blog](https://johnowhitaker.dev) featuring insightful articles like *More=Better?*, mini-hardware projects on *Basement Hydroponics*, and tips on high-surface-area problems.

- **Quadcopter crash avoided by profile optimization**: GitHub links such as [fsdp_qlora benchmarks](https://github.com/AnswerDotAI/fsdp_qlora/blob/main/benchmarks_03_2024.md) and [does LoRA cause memory leaks](https://github.com/huggingface/transformers/issues/25572) were posted. These references enhance knowledge about LLM training, memory leak issues, and their practical resolutions. 

- **Johno shares LoRA wisdom**: Discussions included practical tips around LoRA's functionality, such as approximating large matrices with smaller ones, and considerations for LoRA ranks (N x r). Useful for those fine-tuning models while optimizing resource efficiency.

- **The napkin maestro redefines simplicity**: Johno's clear and effective teaching style captivated attendees, leading to calls for more in-depth sessions. "He knows how to teach and explain things really well," one member noted, urging further opportunities to learn from him.

- **Unlock the power of gradient tricks**: Members shared advanced techniques like [gradient checkpointing](https://discord.com/channels/1238365980128706560/1242223332695478332/1246108353210355805) and splitting the gradient calculation to optimize memory and speed. Hyperlinks to Twitter, GitHub, and Google Docs were passed around for further reading and exploration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/johnowhitaker?lang=en">Tweet from undefined</a>: no description found</li><li><a href="https://johnowhitaker.dev/">johnowhitaker.dev – Jonathan Whitaker</a>: no description found</li><li><a href="https://www.gradio.app/custom-components/gallery?id=radames%2Fgradio_huggingfacehub_search">Gradio Custom Components Gallery</a>: Search through a gallery of custom components.</li><li><a href="https://muellerzr.github.io/blog/gradient_accumulation.html">Zach Mueller - PyTorch, Gradient Accumulation, and the dreaded drop in speed</a>: no description found</li><li><a href="https://sakana.ai/blog/">Sakana AI</a>: Sakana AI Blog</li><li><a href="https://blog.eleuther.ai/transformer-math/">Transformer Math 101</a>: We present basic math related to computation and memory usage for transformers</li><li><a href="https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-2bitgs8-metaoffload-HQQ">mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-2bitgs8-metaoffload-HQQ · Hugging Face</a>: no description found</li><li><a href="https://x.com/gabriberton/status/1796531585958985941">Tweet from Gabriele Berton (@gabriberton)</a>: This simple pytorch trick will cut in half your GPU memory use / double your batch size (for real). Instead of adding losses and then computing backward, it&#39;s better to compute the backward on eac...</li><li><a href="https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39?permalink_comment_id=3417135">Tricks to Speed Up Data Loading with PyTorch</a>: Tricks to Speed Up Data Loading with PyTorch. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://x.com/karpathy/status/1325154823856033793">Tweet from Andrej Karpathy (@karpathy)</a>: How to become expert at thing: 1 iteratively take on concrete projects and accomplish them depth wise, learning “on demand” (ie don’t learn bottom up breadth wise) 2 teach/summarize everything you lea...</li><li><a href="https://ai.google.dev/gemini-api/docs/caching">no title found</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/en/package_reference/lora">LoRA</a>: no description found</li><li><a href="https://x.com/SakanaAILabs/status/1770613032198279663">Tweet from Sakana AI (@SakanaAILabs)</a>: Introducing Evolutionary Model Merge: A new approach bringing us closer to automating foundation model development. We use evolution to find great ways of combining open-source models, building new po...</li><li><a href="https://github.com/angie-chen55/pref-learning-ranking-acc/blob/main/pref_learning_algs_do_not_learn_pref_rankings.pdf">pref-learning-ranking-acc/pref_learning_algs_do_not_learn_pref_rankings.pdf at main · angie-chen55/pref-learning-ranking-acc</a>: Contribute to angie-chen55/pref-learning-ranking-acc development by creating an account on GitHub.</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora/blob/main/benchmarks_03_2024.md">fsdp_qlora/benchmarks_03_2024.md at main · AnswerDotAI/fsdp_qlora</a>: Training LLMs with QLoRA + FSDP. Contribute to AnswerDotAI/fsdp_qlora development by creating an account on GitHub.</li><li><a href="https://github.com/mobiusml/hqq">GitHub - mobiusml/hqq: Official implementation of Half-Quadratic Quantization (HQQ)</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://www.undermind.ai/home/">Undermind Deep Scientific Search</a>: Undermind is an AI-powered search assistant that understands your complex problems. It carefully explores the scientific literature to find what you need, no matter how complex.</li><li><a href="https://www.ai21.com/jamba">Introducing Jamba</a>: A groundbreaking SSM-Transformer Open Model</li><li><a href="https://nvidia.custhelp.com/app/answers/detail/a_id/5490">NVIDIA Support</a>: no description found</li><li><a href="https://johnowhitaker.dev">johnowhitaker.dev – Jonathan Whitaker</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/issues/25572#issuecomment-1687749561">Does lora caused memory leak in transformers ? · Issue #25572 · huggingface/transformers</a>: System Info Issues persisted across several peft version 0.3, 0.4, 0.5(dev) Accelerator I used are 0.21.0, Pytorch version I tried are 1.13.0 and 2.0 and both experience same memory explosion trans...</li><li><a href="https://docs.google.com/presentation/d/1Ye_6zeatCWkq-fx8A--yK34uwU8oC2YQtMSTV1DgkSI/edit?usp=sharing">Napkin Math For Finetuning</a>: Napkin Math For Finetuning Jonathan Whitaker @johnowhitaker</li><li><a href="https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html">How to save memory by fusing the optimizer step into the backward pass — PyTorch Tutorials 2.3.0+cu121 documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1245869242847465492)** (7 messages): 

- **Building Internal Tool for Compliance**: A member is developing an internal tool that converts inputs like "CloudTrail should have encryption at-rest enabled" into multiple files, including rego files compliant with a specific company schema. They are assessing whether the system's 66% accuracy is due to the retrieval method and considering fine-tuning the model for schema and code logic improvements.

- **Challenges in Rule Retrieval and Accuracy**: The tool currently retrieves entire documents for context, potentially overwhelming the model. Issues include code compilation errors, incomplete code, hallucinations, and incorrect logic, with considerations on whether fine-tuning could improve its adherence to schema.

- **Text Classification in Spanish Entities**: A member is refining a model to categorize Spanish text entities into persons, companies, or unions but faces poor performance during inference. They outline the multi-step instructions used for classification and seek advice on improving model accuracy.

- **Maintaining Template Alignment in Fine-Tuning**: For multi-turn chat applications, there's a discussion on whether adhering to the official chat template is crucial when fine-tuning models to retain general utility without starting from scratch. A member assumes alignment is beneficial but looks for community confirmation.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[abhishek_autotrain_llms](https://discord.com/channels/1238365980128706560/1242223673566433431/1246146244628189286)** (57 messages🔥🔥): 

- **AutoTrain Simplifies AI Model Creation**: A member shared links to [AutoTrain](https://huggingface.co/autotrain), emphasizing its user-friendly approach to creating powerful AI models without code. AutoTrain handles a variety of tasks including LLM Finetuning, text and image classification, and is integrated with the Hugging Face Hub for easy deployment.
  
- **Clarification on ORPO and SimPO**: Discussion surrounded ORPO, described as "odds ratio preference optimization" akin to DPO but without a reference model, and SimPO, with participants noting its promising aspects despite being very new and possibly subject to "buzz".

- **Challenges Without Nvidia GPUs**: Members discussed the impracticality of training AI models without Nvidia GPUs, lamenting the slow performance of CPUs and the lack of support for other GPU brands in AI libraries.

- **Dataset and Optimizer Queries**: Participants requested more details on setting up datasets for RAG and customizing optimizer functions for AutoTrain, suggesting these questions be raised in a Zoom Q&A for detailed responses.

- **Gratitude and Additional Resources**: The session ended with multiple users expressing thanks to Abhishek for the presentation on AutoTrain and sharing additional resources, including a [GitHub repo](https://github.com/huggingface/autotrain-advanced) for AutoTrain Advanced and various configuration guides.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/autotrain">AutoTrain – Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/autotrain-advanced/blob/main/docs/source/llm_finetuning_params.mdx">autotrain-advanced/docs/source/llm_finetuning_params.mdx at main · huggingface/autotrain-advanced</a>: 🤗 AutoTrain Advanced. Contribute to huggingface/autotrain-advanced development by creating an account on GitHub.</li><li><a href="https://huggingface.co/docs/autotrain/index">What is AutoTrain Advanced?</a>: no description found</li><li><a href="https://github.com/huggingface/autotrain-advanced">GitHub - huggingface/autotrain-advanced: 🤗 AutoTrain Advanced</a>: 🤗 AutoTrain Advanced. Contribute to huggingface/autotrain-advanced development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/autotrain-advanced/tree/main/configs">autotrain-advanced/configs at main · huggingface/autotrain-advanced</a>: 🤗 AutoTrain Advanced. Contribute to huggingface/autotrain-advanced development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1245957846873538593)** (3 messages): 

- **Love/Hate Relationship with Single Vector Embeddings**: Although single-vector embeddings are useful for prototyping, they fall short of a full retrieval pipeline. ColBERT surpasses single-vector methods in out-of-domain tasks due to its token-level encodings, which provide richer information for OOD scenarios ([Vespa blog on ColBERT](https://blog.vespa.ai/announcing-colbert-embedder-in-vespa/)).

- **Sparse Embeddings and M3 Mentioned Briefly**: The discussion will briefly touch on sparse embeddings and M3, focusing primarily on the advantages and limitations of single-vector embeddings for retrieval pipelines.

- **ColBERT's Detailed Output**: Unlike single-vector methods that pool everything into a 1024-dimensional vector per document, ColBERT produces numerous 128-dimensional vectors, one per token, resulting in a higher dimensional output for more detailed information processing. For instance, 500 documents with 300 tokens each yield an output of `500,300,128`.

**Link mentioned**: <a href="https://blog.vespa.ai/announcing-colbert-embedder-in-vespa/">Announcing the Vespa ColBERT embedder</a>: Announcing the native Vespa ColBERT embedder in Vespa, enabling explainable semantic search using token-level vector representations

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1246162122438869032)** (1 messages): 

- **Search for a reliable BM25 implementation**: A user is looking for a **BM25 ranking** method to mix with vector retrieval and mentions the Python package `rank_bm25`. They express surprise that it doesn't use sklearn tokenizers/vectorizers or handle n-grams, stop words, or stemming in creating the vocabulary, asking what others are using.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1245867317041303684)** (6 messages): 

- **Mitch dives into Gradio fine-tuning**: A user shared their [Gradio fine-tuning project on GitHub](https://github.com/mitch-at-orika/gradio-fine-tuning), aiming to generate a quality dataset to leverage Gradio’s latest features. They mentioned following advice from an earlier course to learn by contributing to open-source projects.

- **Question realism concerns**: Another member pointed out that the LLM-generated questions in the dataset might not reflect realistic user queries. They suggested referencing more concrete questions, providing [a specific example](https://github.com/mitch-at-orika/gradio-fine-tuning/blob/246c34368a72b0a286d3a9fd65d9a882439d4923/datasets/finetune_data.jsonl#L39C139-L39C238), and adding few-shot examples to the prompt.

- **Experimenting with RAG**: A user admitted to not trying Retrieval-Augmented Generation (RAG) before diving into fine-tuning, acknowledging that a strong prompt sometimes surpasses fine-tuning efforts. They are considering integrating RAG into their workflow to enhance question-answer generation.

- **Value of data generation**: Members exchanged views on the importance of data generation and collection in fine-tuning projects. One noted the process as the "secret sauce," showing excitement about the progress and potential of this Gradio fine-tuning venture.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mitch-at-orika/gradio-fine-tuning">GitHub - mitch-at-orika/gradio-fine-tuning: Generate a quality fine-tuning dataset for the Gradio library. The dataset will be designed to help users leverage the latest features and methods of Gradio.</a>: Generate a quality fine-tuning dataset for the Gradio library. The dataset will be designed to help users leverage the latest features and methods of Gradio. - mitch-at-orika/gradio-fine-tuning</li><li><a href="https://github.com/mitch-at-orika/gradio-fine-tuning/blob/246c34368a72b0a286d3a9fd65d9a882439d4923/datasets/finetune_data.jsonl#L39C139-L39C238">gradio-fine-tuning/datasets/finetune_data.jsonl at 246c34368a72b0a286d3a9fd65d9a882439d4923 · mitch-at-orika/gradio-fine-tuning</a>: Generate a quality fine-tuning dataset for the Gradio library. The dataset will be designed to help users leverage the latest features and methods of Gradio. - mitch-at-orika/gradio-fine-tuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1245903574492254339)** (7 messages): 

- **Axolotl not logging to WandB during training**: A user reported that while running locally, initial training metrics are logged to WandB, but nothing updates as training progresses. They suspect it might be related to changes in their configuration file, mentioning, *"the step number restarts after the 2nd step... the metrics reported after the first step 0 are the only metrics I ever see logged."*
  
- **Override datasets in axolotl CLI**: A user inquired whether it's possible to override the datasets (path and type) when calling `axolotl.cli.train` from the command line. No solutions were provided in the message thread.
  
- **Installation issue on Apple M3 Max with axolotl**: A user on an Apple M3 Max reported an error when running `pip3 install -e '.[flash-attn,deepspeed]'`. They also posted a screenshot of the error but did not receive any responses yet.
  
- **Creating instruction-style prompt templates**: A user asked for help setting up a prompt template in axolotl so that it uses a system message from their dataset instead of the preamble. They mentioned struggling with this issue for a couple of hours and sought advice.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1245969978654724159)** (9 messages🔥): 

- **Loading Large Shards With Accelerate Is Painful**: One member inquired about speeding up "loading shards with accelerate" as it "takes quite some time for 70b". Another jokingly suggested getting "a faster hard drive" and warned about the upcoming Lama 400B with weights "near 1TB in size".

- **Unsloth vs Accelerate Shard Loading**: Members discussed how *unsloth* can save 4bit models and load them in 6 shards, suggesting a similar approach for *accelerate*. However, one noted that the delay in loading times is likely related to quantization rather than hard drive speed or mere disk read times.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1245875274810392697)** (6 messages): 

- **Qwen tokenizer debugging insights shared**: After extensive debugging and communication with the Qwen team, it was determined that **PreTrainedTokenizer** is correct and using **Qwen2Tokenizer** might cause issues. This issue stems from differences in how **LLamaFactory** and **Axolotl** handle `get_train_dataloader` calls ([Huggingface transformers trainer](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L880); [Axolotl trainer builder](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/core/trainer_builder.py#L414)).

- **Adjusting prompt styles in Axolotl**: A member inquired about setting different prompt styles for the **alpaca format** in Axolotl. Another member suggested using the `chat_template: chatml` configuration to change the prompt format as per the dataset requirements ([Axolotl prompters](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/src/axolotl/prompters.py#L47)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/src/axolotl/prompters.py#L47">axolotl/src/axolotl/prompters.py at 8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2 · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L880">transformers/src/transformers/trainer.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/core/trainer_builder.py#L414">axolotl/src/axolotl/core/trainer_builder.py at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1246162272834031697)** (2 messages): 

- **Deploy Gradio RAG app easily**: A user asked for the easiest way to deploy a simple Python script for a Gradio-based RAG app so that a small group of users can test it out. Another user recommended using **HF Spaces** for the deployment.
- **Concerns about "share=true" functionality**: The same user expressed curiosity about whether using *"share=true"* in the launch method sends their code to be stored on a Gradio server. There were no additional responses to this query in the messages provided.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1245817913894502492)** (12 messages🔥): 

- **Charles Frye praises community support**: Charles thanked the members who posted event links, highlighting specific users. *"Thanks so much to the folks who posted the links i mentioned in this chat! y'all are the best."*
- **Anticipation and OLAP orientation**: Charles expressed excitement about future discussions and noted their system is *"much more oriented to read-heavy OLAP than write-heavy OLTP."* 
- **Recording of Office Hours: Modal with Charles Frye missing**: A user asked about the recording of the event, noting that the course page still had the "join event" link. Other members confirmed the issue, and Dan fixed it, stating, *"Should be fixed now."*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1245828732384448623)** (70 messages🔥🔥): 

- **LangChain Tools Demystified**: A discussion on the differences between LangChain, LangSmith, LangGraph, LangFlow, and LangServe revealed that **LangChain** is a framework for developing applications using LLMs, **LangSmith** is for inspecting and optimizing chains, and **LangServe** turns any chain into an API. LangFlow and LangGraph usages were more ambiguously linked to this framework ([LangChain Intro](https://python.langchain.com/v0.2/docs/introduction/)).

- **LangServe Praised, LangFlow Not Directly Related**: **LangServe** was highlighted as a favorite tool for turning chains into APIs. Several users clarified that **LangFlow** is not directly related to the LangChain suite but uses the LangChain framework.

- **Infrastructure and Deployment Talks**: There was interest in more granular controls within **LangServe**, and frustrations were expressed regarding its API documentation. Additionally, discussions touched on leveraging OpenAI's batch API for synthetic data generation and the comprehensive learning required for GPU optimization and fine-tuning algorithms.

- **Generative UI Hype**: Members discussed new developments like **GenUI** for improving consumer understanding of AI, with a notable focus on generative UI examples from **CVP at W&B** and a [Generative UI GitHub template](https://github.com/langchain-ai/langchain-nextjs-template/blob/main/app/generative_ui/README.md).

- **Blog Post on LangChain & LangSmith**: A user shared a [blog post](https://lucasvw.github.io/posts/20_ollama_langchain/) detailing their experience using LangChain and LangSmith with LLama3:8b on Ollama and Jarvislabs, prompting others to share it on social media for broader visibility.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/woodstock-happy50th-anniversary-happy-gif-26217300">Woodstock Happy50th GIF - Woodstock Happy50th Anniversary - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/playlist?list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg">LangGraph (Python)</a>: This video series covers how to use code functionality of LangGraph, as well as common modifications one could want to make.</li><li><a href="https://python.langchain.com/v0.2/docs/introduction/">Introduction | 🦜️🔗 LangChain</a>: LangChain is a framework for developing applications powered by large language models (LLMs).</li><li><a href="https://reflex.dev/">Reflex · Web apps in Pure Python</a>: no description found</li><li><a href="https://lucasvw.github.io/posts/20_ollama_langchain/">Lucas van Walstijn - Having fun with llama3:8b</a>: no description found</li><li><a href="https://www.answer.website/">answers, how they should be displayed.</a>: anwser engine built by developers digest</li><li><a href="https://github.com/wandb/openui">GitHub - wandb/openui: OpenUI let&#39;s you describe UI using your imagination, then see it rendered live.</a>: OpenUI let&#39;s you describe UI using your imagination, then see it rendered live. - wandb/openui</li><li><a href="https://github.com/langchain-ai/langchain-nextjs-template/blob/main/app/generative_ui/README.md">langchain-nextjs-template/app/generative_ui/README.md at main · langchain-ai/langchain-nextjs-template</a>: LangChain + Next.js starter template. Contribute to langchain-ai/langchain-nextjs-template development by creating an account on GitHub.</li><li><a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/">What We Learned from a Year of Building with LLMs (Part I)</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1245832348189200485)** (4 messages): 

<ul>
    <li><strong>UI insights give big benefits</strong>: A member pointed out that comparing runs in the UI is a "big weak spot," implying the use of matplotlib as a current workaround. They believe integrating these features natively could provide significant insights and visibility with minimal effort.</li>
    <li><strong>Anticipation for tool evolution</strong>: Another member expressed excitement about the future development of the tool. They are eagerly looking forward to its evolution based on the suggested improvements.</li>
    <li><strong>Curious about solver and log duality in production</strong>: One user inquired if solvers and logs are intended to perform dual roles in production, like invoking solvers to perform tasks and writing evaluative logs. They are curious if the system would support this dual-functionality approach.</li>
    <li><strong>Documentation highly praised</strong>: A user praised the high quality of the documentation, describing it as "remarkably well done". They mentioned that a solver is a Python function that handles a TaskState and many customizations are possible, emphasizing the educational value of the provided code examples.</li>
</ul>

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1245819907229093970)** (12 messages🔥): 

- **Confusion over Registration Deadlines**: There was some confusion regarding the deadline for submitting forms for **$3,500 in compute credits**, with [Hamel Husain's tweet](https://x.com/hamelhusain/status/1795871985265946934?s=12) indicating May 29, while email communications suggested May 30.

- **Last-Minute Form Submission**: Members were concerned about submitting forms by the deadline, but it was clarified that the deadline was midnight to provide 24hrs of leeway for those who signed up the previous night.

- **Credit Allocation Delays**: Users expressed worries about delays in credit grants from modal forms and **Charles** explained the slight delay was due to human review processes taking time.

- **Predibase Registration Issue Solved**: **Michael Ortega** informed users that Predibase had removed the restriction on Gmail addresses for account creation and encouraged users facing issues to contact support.

- **Notification of Credits**: It was clarified that credits would be reflected directly in users' accounts on the respective platforms and that the allocation might take until the middle of next week due to varying vendor responsiveness.

**Link mentioned**: <a href="https://x.com/hamelhusain/status/1795871985265946934?s=12">Tweet from Hamel Husain (@HamelHusain)</a>: The $3,500 in compute credits end TODAY.  We won&#39;t be able to give them out after 11:59 PM PST 5/29/2024  Quoting Eugene Yan (@eugeneyan)   PSA: Signups for LLM-conf + finetuning workshop close to...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/1245823316187680819)** (1 messages): 

- **Evals Gathering in SF**: A member announced a gathering with **50 or so folks** at their co-op in the Mission, SF, to discuss evaluations this Sunday. They asked interested parties to DM them for an invite and provide a social account for verification.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/1245826070712815617)** (4 messages): 

- **Hopeful Registration for Private Event**: One member expressed excitement about registering for an upcoming event and is hopeful for acceptance.
  
- **Call for DC Meetup**: Another member suggested organizing a meetup event in Washington, DC.

- **Chicago's Geographical Dilemma**: A member highlighted that Chicago feels closer to the East Coast and inquired about the possibility of creating a midwest-usa channel.

- **Modal Labs Hosting Office Hours in NYC**: Modal Labs is hosting an office hours session at their HQ in SoHo, NYC. Details for registration, location, and the event schedule are listed in the [event link](https://lu.ma/nllqm67p), open to those verifying token ownership with their wallet.

**Link mentioned**: <a href="https://lu.ma/nllqm67p">[NYC] Modal Office Hours · Luma</a>: Have questions about your Modal deployment or just want to learn more? Come by our first office hours in NY! Even if you don&#x27;t have a particular question in…

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1245814422065254563)** (4 messages): 

- **Users share their locations across Europe**: Members are checking in from different parts of Europe. One user mentioned being in **Nuremberg, Germany**, another from **Milan, Italy**, a third from **Munich, Germany**, and a final one from **Linz, Austria**.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[announcements](https://discord.com/channels/1238365980128706560/1245460787196068030/1245889007465009192)** (1 messages): 

- **Last Reminder for Credit Forms**: *"THIS IS YOUR LAST REMINDER FOR CREDITS! If you not fill out the forms within the next EIGHT HOURS you will NOT BE GRANTED ACCESS TO ANY OF THE CREDITS YOU HAVE AVAILABLE TO YOU! YOU CAN FILL THEM AGAIN JUST IN CASE."* Members are urged to fill out credit forms immediately to ensure access.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1245869285881024532)** (2 messages): 

- **Predibase offers a free trial**: Users were encouraged to [sign up for a 30-day free trial of Predibase](https://predibase.com/free-trial), which includes $25 of credits. Predibase allows for fine-tuning and deploying open-source models on a scalable cloud infrastructure.
- **Inquiry about selecting checkpoints in Predibase**: A user asked if it's possible to try different checkpoints after fine-tuning a model, referencing Predibase's documentation that "The checkpoint used for inference will be the checkpoint with the best performance (lowest loss) on the evaluation set." The user used Predibase to fine-tune a L3 70B model on a small ~200 record dataset.

**Link mentioned**: <a href="https://predibase.com/free-trial">Request Free Trial</a>: Try Predibase for free today - Sign up for your trial

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1245821791390404658)** (8 messages🔥): 

- **Academic to Industry Transition Troubles**: A user with a Ph.D. in philosophy and cognitive science shared their experience of shifting from academia to data science and software engineering. They expressed the challenge of finding new learning opportunities outside a small academic lab and the difficulty of choosing a path in AI given their broad interests and family obligations.

- **Contracting as a Lucrative Option**: Another member shared their positive experiences with contracting, explaining that it offers exposure to diverse problems and cultures. They highlighted the benefits of having the flexibility to choose projects and the possibility of getting long-term offers from organizations if performance is good.

- **Industry Job Rejections Part of the Process**: A user advised that breaking into the industry may involve many rejections and possibly taking an unappealing first job. They emphasized the importance of building a resume and GitHub portfolio to make future job applications easier and more successful.

- **Transition from Academia to Tech Roles**: One member recounted their move from academia to industry, starting with an internship at a tech startup and eventually holding various roles like sales engineer and product manager. They stressed the difficulty of finding opportunities that don't require a decrease in quality of life, especially with current economic challenges.

- **Encouragement and Offer to Help**: Several users encouraged the original poster and others in similar situations, offering support and emphasizing the importance of perseverance. *"If there's anything I can do to help, hit me up. I'm all about people lifting each other up."*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/)** (1 messages): 

rubenamtz: 👀 , credits are still cooking?
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1245830528599592970)** (10 messages🔥): 

- **GPT-powered PDF Chats using llama.cpp and Qdrant**: Check out the [everything-ai project](https://github.com/AstraBert/everything-ai) which now supports llama.cpp and Qdrant, enabling users to chat with their PDFs. This has been publicly appreciated as "coolest community news" and well-received by community members.

- **Codestral-22B Quantization and Nvidia's New Model Demo**: The quantized version of Mistral's model, [Codestral-22B-v0.1-GGUF](https://huggingface.co/QuantFactory/Codestral-22B-v0.1-GGUF), has been highlighted. Nvidia’s [embedding model demo](https://huggingface.co/spaces/Tonic/Nvidia-Embed-V1) adds to the suite of innovative AI applications shared this month.

- **SD.Next and BLIP Dataset Innovations**: The [SD.Next](https://github.com/vladmandic/automatic) release is praised for its new UI and high-res generation capabilities. Additionally, the BLIP dataset was developed using [Clotho](https://huggingface.co/datasets/muzaik/captioned-audio-1k), which enhances the growing dataset collection.

- **New Tools and Plugins Galore**: From the OSS voice-controlled robotic arm [YouTube video](https://www.youtube.com/watch?v=qv3bFhHoA5s) to the Falcon VLM [demo](https://huggingface.co/spaces/Tonic/Falcon-Vision), multiple utilities and demos are shared. These include free-to-use calligraphy datasets, better transcription apps, and visual 3D protein analysis tools.

- **Community Events and Engagement**: The community events, such as [coding sessions](https://discord.com/events/879548962464493619/1245406127668203541) and discussions about AI projects and community-led news, have been noted for their value. These highlights are appreciated by several community members for keeping them updated and engaged with the latest advancements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/chat/assistant/66562fe0abb44809b7f77897)">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://www.youtube.com/watch?v=qv3bFhHoA5s)">Open Source Voice-Controlled Robotic Arm | Redefining Robots!</a>: Welcome to the Voice-Controlled AI Robotic Arm project  where artificial intelligence meets robotics. A open-source initiative empowers users to command a ro...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1245814077989716128)** (415 messages🔥🔥🔥): 

- **Confusion Over Data Formatting and Tokens**: There was an ongoing discussion about the correct format of chatbot training data. Examples like `"<|user|>Do you enjoy cooking?</s><|assistant|>I can bake, but I'm not much of a cook.</s></s>"` led to confusion, prompting questions about whether two `</s>` tokens are necessary.
  
- **Billing Issues Cause Uproar**: A user named Tensorrist expressed urgent distress over a $100 charge for Hugging Face services, claiming they never used the service. Attempts to direct them to contact support at website@huggingface.co seemed to escalate into a heated back-and-forth.

- **Model Merging Competition at NeurIPS**: An announcement was shared about a model merging competition at NeurIPS, offering a prize pool of $8K. The community was encouraged to participate and revolutionize model selection and merging ([link](https://x.com/LChoshen/status/1796256513519989102)).

- **Blogpost Discussions**: Multiple users discussed creating tutorial blog posts, particularly focusing on fine-tuning specific models like TinyLlama and Mistral. One user requested help in avoiding overwriting their README every time they push to the hub.

- **Questions About Fine-Tuning with Specific Data**: Questions were asked about fine-tuning large language models on unique datasets, such as using RDF dumps from Wikipedia or multi-modal models with only text data. Responses suggested technical methods and directed users to the proper channels for more engagement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/cappuch/audio-embedding-wtf">So WTF is an Audio Embedding Model?</a>: no description found</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-mistral">Fine-tuning Mistral on Your Dataset</a>: no description found</li><li><a href="https://huggingface.co/QuantFactory/Codestral-22B-v0.1-GGUF">QuantFactory/Codestral-22B-v0.1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/mixedbread-ai/eming-series-66010c86a966a1c8b6cbb658">em🍞ing series - a mixedbread-ai Collection</a>: no description found</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-tinyllama">Training a Language Model Using TinyLlama</a>: no description found</li><li><a href="https://x.com/DAlistarh/status/1796530164215820766">Tweet from Dan Alistarh (@DAlistarh)</a>: Happy to release PV-Tuning, a new fine-tuning technique for highly-compressed LLMs, which sets SOTA PTQ accuracy for 1-2.5bit LLMs.  Joint work with @peter_richtarik&#39;s lab.   arXiv: https://arxiv....</li><li><a href="https://tenor.com/view/have-a-nice-day-good-sunday-gif-1674652217239459225">Have A Nice Day Good Sunday GIF - Have a nice day Good sunday - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-tinyllama#4-formatting-the-dataset">Training a Language Model Using TinyLlama</a>: no description found</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>: no description found</li><li><a href="https://tenor.com/qgNkIp2pvoz.gif">Simpsons Homer Simpson GIF - Simpsons Homer simpson - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1245925721407488092)** (3 messages): 

- **Queries about Unit 1's identity**: Members discussed the concept of "Unit 1" prompting a question about whether it's a course on Hugging Face. This was clarified with information about various courses including Reinforcement Learning and Computer Vision.

- **Reinforcement Learning Course**: It was specified that one of the courses offered is a Reinforcement Learning course where participants train Huggy the dog. A link to the course was shared, pointing to Hugging Face's learning resources.

- **Community Computer Vision Course Shared**: Another course, focused on computer vision ML, was mentioned. The course objectives include teaching ML concepts using libraries and models from the Hugging Face ecosystem, with a shared link to [Community Computer Vision Course](https://huggingface.co/learn).

**Link mentioned**: <a href="https://huggingface.co/learn">Hugging Face - Learn</a>: no description found

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1245834933231026216)** (6 messages): 

- **NeurIPS hosts model merging contest**: A member shared an announcement about a model merging competition at **NeurIPS** with a link to the [announcement tweet](https://x.com/LChoshen/status/1796256513519989102). The competition offers an $8K prize and is sponsored by **Hugging Face**, **Sakana AI Labs**, and **arcee ai**, with more details available on the [official site](https://llm-merging.github.io/).

- **Transformers tackle arithmetic with new embeddings**: A new paper titled *Transformers Can Do Arithmetic with the Right Embeddings* reveals that adding specific positional embeddings to each digit helps transformers solve arithmetic problems more efficiently. The study achieved up to **99% accuracy on 100-digit addition problems** by training on 20-digit numbers for just one day. [Read the paper here](https://huggingface.co/papers/2405.17399).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2405.17399">Paper page - Transformers Can Do Arithmetic with the Right Embeddings</a>: no description found</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1245833093705633914)** (10 messages🔥): 

- **Approvals and Article Publishing in Blog-explorers**: A member requested help to approve another user to the community of Blog-explorers, which was subsequently approved. The approved user later published an article titled *"Access 150k+ Datasets from Hugging Face with DuckDB and query the data with GPT-4o"* [published here](https://huggingface.co/blog/chilijung/access-150k).
- **Digital Portrait Generation App in Development**: An app to generate digital portraits using **Stable Diffusion (SD) with InstantID and depth + pose CNs** is being developed. The project is still a work in progress with more updates expected.
- **Fake Bot for Personal Website**: A fake bot was created for a personal website, which can be interacted with and explored. Users are invited to try it out at [ngxson.com](https://ngxson.com/).
- **Creative Writing Dataset for LLMs**: A small dataset called [PlotPalette-10K](https://huggingface.co/datasets/Hatman/PlotPalette-10K) designed for fine-tuning large language models for creative writing was shared. It's sourced from various literary sources and generated using the Mistral 8x7B language model.
- **Multi-Aspect Demonstration for SD 2.1**: A multi-aspect demonstration space for **Stable Diffusion 2.1** with an automatic checkpoint updating system has been set up. This space allows users to view ongoing updates as the training continues at [pseudo-flex-v2](https://huggingface.co/spaces/ptx0/pseudo-flex-v2).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/chilijung">chilijung (Howard)</a>: no description found</li><li><a href="https://huggingface.co/spaces/ptx0/pseudo-flex-v2">Ptx0 Terminus Xl Velocity V2 - a Hugging Face Space by ptx0</a>: no description found</li><li><a href="https://ngxson.com/">Xuan Son NGUYEN - Network and Security Engineer</a>: I&#x27;m Xuan Son NGUYEN. Network and Security Engineer. I am captivated by the potential of machine learning and its applications, while also deeply passionate about exploring low-level intricacies t...</li><li><a href="https://huggingface.co/blog/chilijung/access-150k-hugging-face-datasets-with-duckdb">How to directly access 150k+ Hugging Face Datasets with DuckDB and query using GPT-4o</a>: no description found</li><li><a href="https://huggingface.co/datasets/Hatman/PlotPalette-10K">Hatman/PlotPalette-10K · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

taha_69513: Thaaaaaaaaaaaaaaaaaaaaaaaaaaaaaanks 🙌
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1245826407980859434)** (3 messages): 

- **Papers report testing accuracy**: A member asked whether the accuracies reported in papers are usually the testing accuracy or validation accuracy. They referenced a paper fine-tuning a ViT for 10k epochs and shared a funny typo in the graph: [The paper](https://arxiv.org/abs/2211.12879).

- **NeurIPS Model Merging Competition Announced**: A competition related to model merging is announced, with $8K in prizes and support from Hugging Face, Sakana AI Labs, and Arcee AI. More details and sign-up at [Model Merging Competition](https://llm-merging.github.io/) and the announcement tweet [here](https://x.com/LChoshen/status/1796256513519989102).

- **Shared Dataset for Clothing Sales**: A member shared a clothing sales dataset and noted they successfully trained an image regression model using a custom PyTorch model. They also linked their article on this work: [Image Regression using PyTorch and 🤗 Transformers](https://huggingface.co/blog/tonyassi/image-regression).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford">Papers with Code - Stanford Cars Benchmark (Fine-Grained Image Classification)</a>: The current state-of-the-art on Stanford Cars is CMAL-Net. See a full comparison of 73 papers with code.</li><li><a href="https://arxiv.org/abs/2211.12879">Data Augmentation Vision Transformer for Fine-grained Image Classification</a>: Recently, the vision transformer (ViT) has made breakthroughs in image recognition. Its self-attention mechanism (MSA) can extract discriminative labeling information of different pixel blocks to impr...</li><li><a href="https://huggingface.co/datasets/tonyassi/clothing-sales-ds">tonyassi/clothing-sales-ds · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/tonyassi/image-regression">Sales Forecasting with Image Regression</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1245829335382884364)** (7 messages): 

- **K2 model outperforms Llama 2 70B**: LLM360 has unveiled [K2](https://huggingface.co/LLM360/K2), a **fully-reproducible large language model** that surpasses **Llama 2 70B** using 35% less compute. K2 is fully open-sourced with all artifacts and intermediate results available under Apache 2.0 license.

- **NeurIPS Model Merging Competition**: A [model merging competition](https://x.com/LChoshen/status/1796256513519989102) will be held at **NeurIPS** this year, offering $8K in prize money. The competition invites participants to revolutionize model selection and merging with support from sponsors like Hugging Face and SakanaAILabs.

- **Inquiry on Sentence Transformer**: A user inquired if periods (.) serve as sentence demarcations in sentence transformer experiments and whether periods are stripped from abbreviations like "Dr." They are interested in understanding how sentence segmentation is handled.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/LLM360/K2">LLM360/K2 · Hugging Face</a>: no description found</li><li><a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04">Neo-Models - a m-a-p Collection</a>: no description found
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1245822944416890971)** (205 messages🔥🔥): 

- **Phi3 disappoints without quantization tricks**: A user shared their frustration with **Phi3 finetune** results, attributing the issues to **exl2 quantization** breaking SWA. They concluded, *"Running it unquantized and... the results kinda stink."*
- **Discussions on dataset merging and finetuning strategies**: Members discussed the best practices for finetuning, including merging datasets before training. One user emphasized that training on mixed datasets helped avoid issues like catastrophic forgetting.
- **Llama-3 gets an unofficial 11.5B upscale model**: A user shared their creation of an **unofficial Llama-3 11.5B model** using **upscaling techniques** without continuous pretraining. They assured that, *"Full finetune will probably work better,"* but the model was already functional out of the box.
- **Possible new finetuning method, MoRA**: A member mentioned **MoRA**, an updated version of LoRA for **parameter-efficient finetuning**, making a reference to its potential use in future upgrades. The MoRA GitHub code was [shared](https://github.com/kongds/MoRA).
- **Issue with HF Trainer when loading models**: Users discussed encountering crashes in notebooks while loading models, especially on CPU. A temporary solution was suggested to *remove spaces in folder names* to resolve the crash.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Replete-AI/Llama-3-11.5B-V2">Replete-AI/Llama-3-11.5B-V2 · Hugging Face</a>: no description found</li><li><a href="https://x.com/dudeman6790/status/1796382605086015993">Tweet from RomboDawg (@dudeman6790)</a>: It seems like besides TruthfulQA, there is was basically no loss in the upscaled model we made a while back. So if anyone wants to finetune using an upscaled version of llama-3 then the base version w...</li><li><a href="https://tenor.com/view/arthur-morgan-rdr2-rdr-2-gif-15824440020465833707">Arthur Morgan Rdr2 GIF - Arthur morgan RDR2 RDR 2 - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/Replete-AI/Llama-3-13B">Replete-AI/Llama-3-13B · Hugging Face</a>: no description found</li><li><a href="https://github.com/kongds/MoRA">GitHub - kongds/MoRA: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning</a>: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning - kongds/MoRA
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1246032724146520126)** (9 messages🔥): 

- **NVIDIA's new 4nm chip blows Blackwell out of the water**: NVIDIA's new research chip achieves 96 int4 TOPs/Watt compared to Blackwell's 20T/W, representing significant power efficiency. For more details, check out the [full research talk](https://youtu.be/gofI47kfD28?si=41UIMkpMCyb_qWqA).
  
- **Confusion around float4's exponent and mantissa**: Daniel Han mentions that B200's float4 is claimed to have an exponent and mantissa both equal to 2, leading to confusion. Typically, for float4, a common configuration is *sign bit + exponent + mantissa = 4*; hence the discussion on whether NVIDIA's configuration lacks a sign bit.

- **Notable speed-ups from reduced numerical representation**: Daniel highlights that significant speed-ups are not solely due to Moore's Law but from reducing the numerical representation from fp32 to f4, providing a 32x boost. However, the Physics of LLMs paper shows int4 can be 2x worse, indicating limits to these performance improvements. Check the paper [here](https://arxiv.org/abs/2404.05405).

- **Tensor Cores and HMMA provide remarkable efficiency**: Tensor Cores using complex instructions like HMMA achieve 13x faster performance with lower energy consumption, playing a critical role in enhancing computational efficiency.

- **Progress in sparsity techniques**: NVIDIA is working on shifting from 2:4 to 2:8 sparsity, which could further optimize computational efficiency and performance in AI models.

**Link mentioned**: <a href="https://x.com/danielhanchen/status/1796253349932843214">Tweet from Daniel Han (@danielhanchen)</a>: My notes from a NVIDIA research talk:  1) NVIDIA has an research inference 4nm chip doing 96 int4 TOPs/Watt vs Blackwell&#39;s 20T/W  2) B200&#39;s float4 is exponent=2 and mantissa=2? Maybe I mishear...

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1245863643250429952)** (150 messages🔥🔥): 

- **Selecting GPUs in Unsloth**: A member asked about selecting specific GPUs for training in Unsloth. Another member suggested using `os.environ["CUDA_VISIBLE_DEVICES"]="0"` in Python, providing a [reference link](https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter) for further details.

- **Troubleshooting Fine-Tuning Errors with Quantized Models**: A user encountered a ValueError related to fine-tuning quantized models with unsloth/llama-3-8b-bnb-4bit using a personal dataset. Another pointed out the need to attach trainable adapters on quantized models, referencing [Parameter-Efficient Fine Tuning (PEFT)](https://huggingface.co/blog/peft) for more information.

- **Handling Fine-Tuning with Two Models**: A member discussed the complexity of handling two tightly related tasks where the first model's output serves as the input to the second. They considered generating "synthetic" data but concluded it would be equivalent to the real data.

- **Issues with Unsloth on Kaggle**: A user reported an issue with installing Unsloth in a Kaggle notebook, which was acknowledged and being investigated. They also filed an issue on GitHub ([link](https://github.com/unslothai/unsloth/issues/566)).

- **Understanding VRAM Spikes in Training**: Members discussed VRAM spikes during training with Unsloth, noting that long sequence lengths (16K tokens) caused fragmentation and memory allocation issues. Suggestions included using environment variables like `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/peft">Load adapters with 🤗 PEFT</a>: no description found</li><li><a href="https://github.com/kongds/MoRA">GitHub - kongds/MoRA: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning</a>: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning - kongds/MoRA</li><li><a href="https://github.com/unslothai/unsloth/issues/566">Kaggle notebook: No module named &#39;unsloth&#39; · Issue #566 · unslothai/unsloth</a>: I got the error in kaggle notebook, here is how I install unsloth:</li><li><a href="https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter)">Tensorflow set CUDA_VISIBLE_DEVICES within jupyter</a>: I have two GPUs and would like to run two different networks via ipynb simultaneously, however the first notebook always allocates both GPUs. &#xA;&#xA;Using CUDA_VISIBLE_DEVICES, I can hide devices f...</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=upcOlWe7A1vc">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?us">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1246032169021870162)** (4 messages): 

- **Ghost XB Beta supports multiple languages**: The **Ghost XB Beta** model is set to **support 9+ languages fluently**. The model is currently at **29% of its first training stage**, with updates available on [Ghost AI's X page](https://x.com/ghostx_ai) and [ghost-x.org](https://ghost-x.org).
- **Prioritize openness and efficiency**: The project emphasizes **community support** and **startups**, focusing on **self-deployment** and **cost-efficiency** for small and medium-sized models. For more details and deployment, users are directed to [Hugging Face](https://huggingface.co/ghost-x).
- **Optimized for production**: The models are designed for **high performance and enterprise-level scalability** with a low cost. They can be deployed on various platforms, ensuring wide accessibility and **ease of use**.
- **Ghost Alpha for advanced tasks**: The next generation **Ghost Alpha** models are optimized for **reasoning, multitask knowledge, and multilingual support**. Users can explore these models on [Hugging Face](https://huggingface.co/ghost-x/ghost-alpha-661005edac50abb8d56c90f1).

For further exploration, check out [Ghost Alpha on Hugging Face](https://huggingface.co/ghost-x/ghost-7b-alpha).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ghostx_ai">Tweet from undefined</a>: no description found</li><li><a href="https://ghost-x.org">Ghost X</a>: The Ghost X was developed with the goal of researching and developing artificial intelligence useful to humans.</li><li><a href="https://huggingface.co/ghost-x">ghost-x (Ghost X)</a>: no description found
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1245818109063856188)** (281 messages🔥🔥): 

- **Tako Widget Queries Locale Limits**: A member asked if the finance data widget, Tako, is limited to the USA only, but received an uncertain response. Another user confirmed their uncertainty with "not sure".

- **Pro Trials Gone for Good**: One member inquired about the availability of Perplexity Pro trials for friends, only to be informed they have been removed, including the yearly 7-day trial. This led to a discussion about the possibility of covering trial costs themselves or using referral strategies.

- **Editing Page Sections Confusion**: One user struggled with editing sections of pages, learning that the edit button allows for changing section details but not actual text. Another user confirmed and explained the limited functionality.

- **Discussion on Slower Pro Search**: Several users noticed that Perplexity Pro search has slowed down recently, which was attributed to a new search strategy that breaks down queries into sequential steps. Despite the slowdown, users appreciated the improved detailed responses.

- **Inquiries About Pro Features and Models**: Members discussed whether Pro features include image attachments compatible with GPT-4o and the absence of certain model limits in the UI. It was also noted that Pro users should select models other than the default for better performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.m.wikipedia.org/wiki/Secure_Remote_Password_protocol">Secure Remote Password protocol - Wikipedia</a>: no description found</li><li><a href="https://aistudio.google.com/app/prompts/1s_utccw9l5Qdm9Aaxwj0DWyBMpGo1UUP">no title found</a>: no description found</li><li><a href="https://tenor.com/view/hatsune-miku-vocaloid-jump-birthday-anime-gif-26599041">Hatsune Miku Vocaloid GIF - Hatsune Miku Vocaloid Jump - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.c">llm.c/train_gpt2.c at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1245856701517926526)** (2 messages): 

- **Perplexity Pages Launched**: A member shared a link to the [Perplexity Pages Debut](https://www.perplexity.ai/page/Perplexity-Pages-Debut-EzSwGJ2LTIq253dkjxrXjg). This introduces a new feature or service on the Perplexity AI platform.
- **Codestral de Mistral Explored**: Another member posted a [link about Codestral de Mistral](https://www.perplexity.ai/page/Codestral-de-Mistral-ILrQahR6Rn6pAYBtZFw1rQ). This likely elaborates on a specific topic or functionality related to Perplexity AI.
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1245821292691853322)** (11 messages🔥): 

- **NVIDIA's Advanced Research Chip Impresses**: A member shared notes from a [NVIDIA research talk](https://x.com/danielhanchen/status/1796253349932843214?s=46), highlighting a new 4nm inference chip boasting 96 int4 TOPs/Watt, far surpassing Blackwell's 20T/W. The discussion also covered advancements in tensor cores and the potential shift to 2:8 sparsity.
  
- **Meta's Next-Gen AI Accelerator Unveiled**: Discussion revolved around Meta's [next-generation AI training and inference accelerator](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/#hardware) which promises 72 accelerators per node and 354 TFLOPS/s (INT8) at 90W, setting the stage for advanced AI infrastructure.
  
- **Upcoming CUDA C/C++ Course Announcement**: A member announced their upcoming GPU programming course on FreeCodeCamp, inviting the community for feedback. This initiative aims to lower the high barrier to entry in GPU programming by offering detailed explanations and fostering community input.

- **Requests for Specific GEMM Coverage**: Feedback on the CUDA course includes requests to cover GEMM for rectangular matrices and broadcasting rules, essential for img2col based convolution. Additionally, another user noted looking forward to the course despite their WebGPU experience.
  
- **Concerns on Parallel Scan Paper**: A member sought clarification on concepts from the `Single-pass Parallel Prefix Scan with Decoupled Look-back` paper, particularly regarding its statement on two full passes and the notion of allocation within global allocation mechanisms.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1796253349932843214?s=46">Tweet from Daniel Han (@danielhanchen)</a>: My notes from a NVIDIA research talk:  1) NVIDIA has an research inference 4nm chip doing 96 int4 TOPs/Watt vs Blackwell&#39;s 20T/W  2) B200&#39;s float4 is exponent=2 and mantissa=2? Maybe I mishear...</li><li><a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/#hardware">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=UU1WVnMk4E8&t=5775s&ab_channel=freeCodeCamp.org">Create a Large Language Model from Scratch with Python – Tutorial</a>: Learn how to build your own large language model, from scratch. This course goes into the data handling, math, and transformers behind large language models....</li><li><a href="https://www.youtube.com/watch?v=5Sm9IVMet9c&t=119s&ab_channel=freeCodeCamp.org">Mojo Programming Language – Full Course for Beginners</a>: Learn Mojo in this full tutorial. The Mojo programming language combines the usability of Python with the performance of C. It&#39;s basically an enhanced versio...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1245860653973897257)** (1 messages): 

- **Human-readable Triton kernel names**: The default name for a kernel is usually `triton_`. There is a config in `_inductor/config.py` that can change these names to be more human-readable, though **using ncu for kernel profiling** is suggested.
  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1246175277428641813)** (1 messages): 

- **Scan Algorithm Part 2 Begins**: The announcement informed the community that part 2 of the **scan algorithm** session was about to start. Members were encouraged to join in the next few minutes.
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1245841697745928193)** (10 messages🔥): 

- **Choose vllm for the webserver setup**: A member clarified the utility of vllm, stating, *"As for the inference part, i think vllm is the better choice (provides a ready to use chat webserver)."*
- **Batch processing with single model struggles**: A member asked how to handle batching in PyTorch without creating multiple model instances, expressing frustration with, *"very inefficient manner where i create copies of model for each image,"* and sought advice on sharing model parameters efficiently.
- **Pytorch DataLoader vs. custom batching**: One user suggested using Pytorch DataLoader for batching, but the original poster countered that their function contains custom image-by-image operations, making a full batch-handling refactor undesirable.
- **Error with torch.multiprocessing and CUDA**: Despite attempting to use `torch.multiprocessing` to handle multiple processes, a member encountered an error related to CUDA core reservation, noting, *"it gives out the error of using cuda cores which are already reserved."*
- **Dataloader iteration for single image compatibility**: The discussion concluded with the suggestion that if the model supports only one image at a time, a dataloader would still return single images iteratively, avoiding the need for batch-based functionality changes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py">GFPGAN/inference_gfpgan.py at master · TencentARC/GFPGAN</a>: GFPGAN aims at developing Practical Algorithms for Real-world Face Restoration. - TencentARC/GFPGAN</li><li><a href="https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel">A detailed example of data loaders with PyTorch</a>: no description found</li><li><a href="https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py#L126-L131">GFPGAN/inference_gfpgan.py at master · TencentARC/GFPGAN</a>: GFPGAN aims at developing Practical Algorithms for Real-world Face Restoration. - TencentARC/GFPGAN</li><li><a href="https://pytorch.org/docs/stable/notes/multiprocessing.html">Multiprocessing best practices &mdash; PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://github.com/pytorch/examples/blob/main/mnist_hogwild/main.py#L92-L99">examples/mnist_hogwild/main.py at main · pytorch/examples</a>: A set of examples around pytorch in Vision, Text, Reinforcement Learning, etc. - pytorch/examples
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1246177049635459215)** (2 messages): 

- **Izzat's Scanning Session Part 2 Starts**: A member announced that the second part of the scan by Izzat is starting now. They provided a [link to join the session](https://linkedin.zoom.us/j/98060172269).

**Link mentioned**: <a href="https://linkedin.zoom.us/j/98060172269">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1245816576553390175)** (127 messages🔥🔥): 

- **Hitting 20k Stars Today**: One user excitedly noted, *"probably hitting 20k stars today 🎉"*, reflecting the community's anticipation of a significant milestone.
- **Merging the llmc Lib Directory**: A member discussed the planned merge of the llmc lib directory and structuring scripts for better organization. The pull request can be viewed [here](https://github.com/karpathy/llm.c/pull/475).
- **Trainer Struggles with Loss Spikes**: Several users articulated issues with training runs encountering reproducible loss spikes. One suggested plotting gradient norms to identify potential causes, while another recommended considering hardware differences during debugging.
- **Dataset Download and Hosting on Hugging Face**: Users discussed discrepancies in dataset shards and considered hosting on Hugging Face for ease of access. One user provided a [Hugging Face dataset link](https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/blob/main/fineweb_train_000377.bin) and proposed using compression methods to optimize download speeds.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/channel/UC0qODiToYRpntlfKTu1TbGw">Chris Dryden</a>: no description found</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb/discussions/28">HuggingFaceFW/fineweb · &quot;Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20&quot; using fineweb by Karpathy</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/data/data_common.py#L37">llm.c/dev/data/data_common.py at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/blob/master/scripts/run_gpt2_124M.sh">llm.c/scripts/run_gpt2_124M.sh at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/blob/main/fineweb_train_000377.bin">fineweb_train_000377.bin · chrisdryden/FineWebTokenizedGPT2 at main</a>: no description found</li><li><a href="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_train_000377.bin?download=true">no title found</a>: no description found</li><li><a href="http://d3joo518jcgf5j.cloudfront.net/fineweb_train_000377.bin">no title found</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/475">experiment with adding the llmc lib directory by karpathy · Pull Request #475 · karpathy/llm.c</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/1245819136333053952)** (3 messages): 

- **Uncertainty about next lecture**: A member mentioned that Lecture 7 might be up next, but they were not sure about the schedule of the NAM team. They asked for confirmation on this.

- **Inquiry about NAM team's Zoom links**: Another member inquired whether the NAM team's Zoom links are posted in the channel. This question was directed towards figuring out the proper platform for the upcoming session.

- **Link shared**: A Discord link was shared [here](https://discord.gg/5wCbVAjy) potentially for relevant information or resources. No additional context was provided for this link.

**Link mentioned**: <a href="https://discord.gg/5wCbVAjy">Join the PMPP UI lectures timezones Discord Server!</a>: Check out the PMPP UI lectures timezones community on Discord - hang out with 37 other members and enjoy free voice and text chat.

  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1245815010505134110)** (4 messages): 

- **Windows debugging for CUDA**: A member expressed interest in making it work with Windows, despite **torchao** not officially supporting Windows. They mentioned, "I can debug it with my Windows PC."

- **Compiler differences cause build errors**: There was a discussion on build errors caused by different meanings of `__restrict__` and `__restrict` between compilers.

- **Triton difficulties acknowledged**: A member noted that making Triton work on Windows won't be an easy task, saying, "the triton stuff isn't gonna happen so easily."

- **FP6 LLM discussions moved**: For discussions about **FP6 LLM**, they referred to another channel: *“we can discuss at <#1235775768756359289>”*.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1245814193286807552)** (152 messages🔥🔥): 

- **Privacy and Permissions in Online Content**: One member emphasized that the best privacy feature is the option to not publish. Another added that people need to be educated on not giving companies their content to avoid their usage.
- **Inconsistent Results Across Different Tools**: A member was frustrated about not getting the same results in ComfyUI as in Forge even with identical settings. They pointed out differences in settings and features like XFormers possibly affecting outcomes.
- **Combining Models for Improved Outputs**: Discussions highlighted that models like SDXL and SD15 can be combined in various ways for better results, although control nets need consistency across model phases.
- **Training and Using Specific Models**: Users shared concerns and advice related to training models specific to their needs. One member referenced youtube videos and pointed to tools like OneTrainer and kohya_ss for Lora training.
- **Resource Recommendations**: For beginners, resources like [Craiyon](https://www.craiyon.com) were recommended for initial experiments with AI-generated images before moving to more advanced web services or local installations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/dancing-cat-dance-cat-cat-meme-chinese-cat-gif-12629347036627000898">Dancing Cat Dance GIF - Dancing cat Dance Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=mxin2EjzvQM&list=PLRvM4LTxFnNVZxqIJlkB0ff3V6drhUD-r&index=2">Troca do Estilo de Arquitetura / Quiosque da Praça de Divino, MG - Controlnet IP-Adapter</a>: Olá! Neste vídeo usamos a poderosa ferramenta do IP-Adapter para trocar o estilo arquitetônico, a partir de uma  foto base de uma construção existente na pra...</li><li><a href="https://civitai.com/models/448101/sprite-sheet-maker">Sprite Sheet Maker - v4.2 Ollama | Stable Diffusion Workflows | Civitai</a>: Update: v4.2 adds Ollama and IP adapter. Version 4.0 - It may seem like some versions were skipped, they existed, I just forgot to share them as I ...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d4cwi9/pcm_phased_consistency_model/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1245835875107999844)** (60 messages🔥🔥): 

- **GPU struggles with ROCm support**: A user expressed frustration over their **RX 6700's** performance, noting they can't exceed 10 tokens per second likely due to lack of **ROCm support**. Another user mentioned that dual-booting to Ubuntu won't improve it significantly since backend speeds are similar across OSes.

- **Fixing unexpected endpoint errors**: A user reported an error with `GET /v1/chat/completions/models`, leading to another recommending posting in a specific channel with more context. Additionally, it was pointed out that the correct endpoint is `v1/models`.

- **AVX2 requirement exclusion frustration**: Multiple users experienced issues while trying to run LM Studio, with troubleshooting revealing that **AVX2 instructions** are a requirement. An older version, [0.2.10 for AVX](https://lmstudio.ai/beta-releases.html), was recommended as a workaround.

- **PDF and text-to-video limitations**: Users frequently asked about feeding **PDF files** to the AI and were directed to use server mode with a third-party app called AnythingLLM. Another inquiry about text-to-video applications highlighted the limited, proof-of-concept options like **Stable Video Diffusion**, available only for NVIDIA GPUs.

- **Localization and CPU instruction set issues**: Errors during setup on non-EN locale settings and lack of **AVX2 CPU instructions** were identified as common issues preventing successful installation. Users with old CPUs without AVX2 support were provided a link to an [older beta version](https://lmstudio.ai/beta-releases.html) as a potential solution.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found</li><li><a href="https://tenor.com/view/monsters-university-snail-going-on-my-way-omw-gif-5461800">Monsters University Snail GIF - Monsters University Snail Going - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1245817396011208837)** (30 messages🔥): 

- **Coder Prompt Template Explained**: One of the members clarified that a particular model uses the **Deepseek coder prompt template**, *highlighting its significance for those wondering about its setup*.
  
- **Manually Prompting FIM with Codestral**: A member inquired about techniques for manually prompting FIM with **Codestral**, indicating interest in the method's specifics. No follow-up or guidance was provided.

- **Model Parameter Limitation Discussion**: A member posed a question about whether increasing a model's parameter size is the only way to improve its handling of complex instructions like maintaining multiple personalities. They expressed skepticism about the effectiveness of Mixture of Expert models, noting that larger models typically perform better in their experience.

- **Llama2 7B Chat Uncensored Formatting Issue Solved**: An issue with prompt formatting for **TheBloke/llama2_7b_chat_uncensored-GGUF** was discussed, with guidance provided through a specific [model card link](https://huggingface.co/TheBloke/llama2_7b_chat_uncensored-GGUF#prompt-template-human-response). The conversation included a suggestion to check tokenizer configurations and model cards for appropriate chat templates.

- **Text Extraction Model Success**: There was a brief discussion about the success of different models in text extraction tasks. However, no specific models were mentioned or recommended during this exchange.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/DavidAU/Dark-Forest-V1-Ultra-Quality-20b-GGUF">DavidAU/Dark-Forest-V1-Ultra-Quality-20b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/DavidAU/Fimbulvetr-11B-Ultra-Quality-plus-imatrix-GGUF">DavidAU/Fimbulvetr-11B-Ultra-Quality-plus-imatrix-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/llama2_7b_chat_uncensored-GGUF#prompt-template-human-response">TheBloke/llama2_7b_chat_uncensored-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1245994617405571123)** (2 messages): 

- **Wrong channel for visual models**: A member mistakenly inquired about support for visual models in the wrong chatroom. They quickly apologized for the mistake, acknowledging their error with *"oops wrong chatroom soryr."*
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1245829023423008888)** (21 messages🔥): 

- **RX580 faces compatibility issues with ROCm**: Members discussed the limitations of using the **AMD RX580** GPU with ROCm, noting that it's considered "too old" and therefore incompatible, making users rely on OpenCL for GPU inference. One member confirmed, *"ROCm is incompatible with Polaris GPUs like the 580."*

- **ROCm and RX 6600XT unsupported issues**: Another user inquired about the **RX 6600XT** GPU's compatibility with ROCm, only to be informed that this GPU is also unsupported. The reaction was one of disappointment: "Damn."

- **M3 Max beats Snapdragon X Elite**: A user highlighted that the **M3 Max** outperforms the Snapdragon X Elite in every category, even without considering the M4 and M3 Ultra models. They noted, *"The cheapest Snapdragon PC is the 899 developer kit, but that is the slower 32GB model."*

- **Multi-GPU performance tests with LLAMA 3 8B Q8**: Testing various configurations, a user found that using two GPUs yielded 91% of the performance of a single GPU, considering the impact of PCIE bandwidth. It was concluded that splitting load across multiple GPUs via X1 to X16 adaptors offered better performance stability: *"I've decided to just get another X1 to X16 adaptor."*

- **GPUDirect shows potential, but has limits**: Discussing NVIDIA's **GPUDirect** technology for enhancing data movement, a user noted possibilities for directly reading from NVMe storage to reduce VRAM memory pressure. Another member remarked that such attempts have been made, but disk for RAM usage remains too slow: *"[Using disk for RAM] is just too slow to be of use."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/lyogavin/llama3-airllm">Run the strongest open-source LLM model: Llama3 70B with just a single 4GB GPU!</a>: no description found</li><li><a href="https://developer.nvidia.com/gpudirect">GPUDirect</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1245983293091090542)** (11 messages🔥): 

- **4070 and 1070 VRAM comparison sparks debate**: A member pointed out that the 4070 has only 12GB of VRAM compared to the 1070, which has an extra 8GB. This triggered a discussion about the suitability of these cards for larger models like "codestral".

- **1070’s performance scrutinized**: One member downplayed the performance of the 1070 as being slow, to which another responded with practical use cases showing decent performance rates, such as running Phi-3 Small at 50 tokens/s and Llama-3-8b at 35 tokens/s.

- **CPU thread utilization issue resolved**: Khabu asked for advice on optimizing CPU thread usage for better model performance, explaining issues when increasing threads beyond 4. After some troubleshooting advice, Khabu mentioned that the problem was resolved without needing further discussion.


  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1245898428870033479)** (1 messages): 

- **InternLM models galore**: Tons of **InternLM** models were announced, covering Math and Coding, ranging from *7B* to a *mixtral 8x22B* based math model. Several models are available, including [AlchemistCoder-DS-6.7B-GGUF](https://huggingface.co/lmstudio-community/AlchemistCoder-DS-6.7B-GGUF), [AlchemistCoder-L-7B-GGUF](https://huggingface.co/lmstudio-community/AlchemistCoder-L-7B-GGUF), [internlm2-math-plus-7b-GGUF](https://huggingface.co/lmstudio-community/internlm2-math-plus-7b-GGUF), [internlm2-math-plus-20b-GGUF](https://huggingface.co/lmstudio-community/internlm2-math-plus-20b-GGUF), and [internlm2-math-plus-mixtral8x22b-GGUF](https://huggingface.co/lmstudio-community/internlm2-math-plus-mixtral8x22b-GGUF).
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1245850116582281236)** (7 messages): 

- **OpenRouter enhances API performance**: Major scalability improvements have reduced latency by at least ~200ms globally, with significant gains for Africa, Asia, Australia, and South America. *"By pushing more of our user data closer to the edge, we shaved off at least ~200ms from every single request."*

- **Monitor model uptime with new charts**: OpenRouter introduced uptime charts to visualize the benefits of their provider load balancing, like the one on [WizardLM-2 8x22b](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b?tab=uptime). This feature helps avoid impacts from sporadic upstream outages.

- **Early preview of Category Rankings available**: Users can see how different models rank across various categories on [openrouter.ai/rankings](https://openrouter.ai/rankings). Notable insights include MythoMax's dominance in roleplay and GPT-4o leading in programming.

- **Laravel developers get a new package**: [moe-mizrak/laravel-openrouter](https://github.com/moe-mizrak/laravel-openrouter) was announced to help Laravel developers integrate with OpenRouter.

- **DB issues cause API disruptions but are resolved**: An internal error with the DB cache led to API calls returning 504 or 500 errors. The problem primarily affected the India (bom1) and Singapore (sin1) regions but was resolved by adding a fallback direct DB lookup, as reported by *"UPDATE: The fix is now up, and our 1-hour uptime chart is recovering."*

**Link mentioned**: <a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b?tab=uptime>).">WizardLM-2 8x22B by microsoft | OpenRouter</a>: WizardLM-2 8x22B is Microsoft AI&#x27;s most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing ...

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1245826173007560859)** (1 messages): 

- **MixMyAI.com offers pay-as-you-go AI services**: *Introducing mixmyai.com* as a comprehensive solution for all AI needs without monthly fees to different vendors. The platform combines both closed and open-source models under one roof, providing the most affordable pricing options.

- **MixMyAI emphasizes user privacy**: The service prioritizes privacy by **not storing any chats on servers** and offers a transparent dashboard to track spending. It also ensures models are always current by retiring old ones.

- **User-friendly and powerful UI**: MixMyAI boasts a powerful user interface that allows users to *search chat history, save prompts, and tweak LLM settings*. The platform emphasizes ease of use and accessibility.
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1245814123741319168)** (97 messages🔥🔥): 

- **Latent Space podcast welcomes fans**: The first [in-depth interview on MosaicML MPT-7B](https://www.latent.space/p/mosaic-mpt-7b?utm_source=substack&utm_medium=email) discusses overcoming GPT-3 limitations in context length. They are also hosting meetups and inviting beta testers for an upcoming AI course.
- **OpenRouter Ruby library released**: Obie Fernandez announces the release of the [OpenRouter Ruby client library](https://github.com/OlympiaAI/open_router). He also mentions maintaining [Ruby AI eXtensions for Rails](https://github.com/OlympiaAI/raix-rails), a library dependent on OpenRouter.
- **API 504 issues rampant across regions**: Numerous users report encountering 504 errors with specific models like "Mixtral 8x7B Instruct" and "llama-3-70b-instruct", spanning multiple global locations including India, Vietnam, and Poland. A [temporary fix](https://openrouter.ai/playground) was applied, but stability remains inconsistent.
- **Category rankings feedback and updates**: Discussions focused on improving category rankings data, with users suggesting new categories and highlighting the need to evaluate models based on common use cases and "quality". Alex Atallah and others affirmed that more detailed rankings and additional categories are forthcoming.
- **Discussion on health check endpoints**: Users requested a health check API to monitor OpenRouter's status and take actions accordingly. Alex Atallah suggested using the model pages for health checks until a dedicated endpoint is available.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.latent.space/p/mosaic-mpt-7b?utm_source=substack&utm_medium=email">MPT-7B and The Beginning of Context=Infinity — with Jonathan Frankle and Abhinav Venigalla of MosaicML</a>: Ep 13: Training Mosaic&#x27;s &quot;llongboi&quot; MPT-7B in 9 days for $200k with an empty logbook, how to prep good data for your training, and the future of open models</li><li><a href="https://github.com/OlympiaAI/open_router">GitHub - OlympiaAI/open_router: Ruby library for OpenRouter API</a>: Ruby library for OpenRouter API. Contribute to OlympiaAI/open_router development by creating an account on GitHub.</li><li><a href="https://github.com/OlympiaAI/raix-rails">GitHub - OlympiaAI/raix-rails: Ruby AI eXtensions for Rails</a>: Ruby AI eXtensions for Rails. Contribute to OlympiaAI/raix-rails development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1245854992166944879)** (85 messages🔥🔥): 

- **New Features and Benefits for Pro Users**: Members discussed the benefits of OpenAI pro users, including **higher rate limits** and access to **DALL-E**, GPT creation, real-time voice, and video chat. Free users cannot create GPTs, making the pro subscription more appealing despite the $20 monthly fee.

- **Application Use Case Suggestions**: A member described developing an AI acting as a ruler with specific traits, considering using Chat API over Assistant API due to not needing certain features like file search/coding. It was recommended to use **Chat API** for better control and efficiency.

- **ChatGPT and Bias Concerns**: A member expressed frustration over being suspended for calling ChatGPT's responses racist, leading to an in-depth discussion on **model biases** and training data, and how such biases are inevitable but mitigated by additional work on top of training data.

- **Anthropic AI Agents Feature**: Members compared Anthropic's new **"tool use" feature** with OpenAI's function calling, noting that both allow custom assistant creation through API integration. Despite apparent similarities, it was suggested Anthropic's feature might provide deeper integration with personal data. 

- **Sora and Video Generation Hype**: Discussion touched on the excitement around new-gen video models such as **Sora and Veo**, including anecdotal claims about high curation ratios for video generation. There's skepticism over the hype versus practical usability in current video AI technology.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/5/30/24167231/anthropic-claude-ai-assistant-automate-tasks">Anthropic’s AI now lets you create bots to work for you</a>: Anthropic is releasing a tool that allows customers to build their own AI assistants.</li><li><a href="https://www.theverge.com/2024/5/30/24167231/">Anthropic’s AI now lets you create bots to work for you</a>: Anthropic is releasing a tool that allows customers to build their own AI assistants.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1245824522867708064)** (10 messages🔥): 

- **Dashboard link request**: A user requested a link that shows them as a creator with the list of their GPTs instead of a single GPT URL. There was no direct response provided in the chat to this request.
- **Memory leak crashes browsers**: Members reported severe **lag and browser crashes** when using ChatGPT, attributing the cause to a **memory leak issue** on the website. Recommendations included avoiding long context and refreshing the page when locks occur.
- **Code generation freezes browsers**: A user experiences **browser freezes and crashes** when generating code inside codeblocks, even on a high-end PC, mentioning the error: *Out of memory*. This issue is recent and persists across multiple browsers.
- **Tricks to control usage**: One member speculated that OpenAI might control usage during high demand by monitoring how often users hit their usage cap daily, suggesting it might apply to both the web interface and the API.
- **Voice mode timeline shared**: In response to a query, a timeline from OpenAI was shared, stating that **real-time voice and vision** features for GPT-4 will start rolling out to a limited Alpha for ChatGPT Plus users soon, with wider availability planned over the coming months. [OpenAI's shared timeline](https://help.openai.com/en/articles/8400625-voice-chat-faq).
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1246026857758064711)** (3 messages): 

- **Dealing with repetitive GPT-3 responses**: A member is facing an issue with the API returning repetitive answers even when the prompt's topic is not narrow. Another member suggested limiting the number of messages per chat session to around 10 or repeating the whole previous answers to mitigate this problem.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1246026857758064711)** (3 messages): 

- **Repetitive answers in OpenAI API**: A member reported getting repetitive answers from the OpenAI API despite having a broad prompt. They queried if keeping track of previous answers and informing the model would solve this, but were concerned the list might become too extensive.

- **Limiting chat session helps avoid repetition**: Another member suggested limiting the number of messages per chat session to around 10 as one method to avoid repetitive answers. They also mentioned repeating the whole previous answer could help mitigate this issue.
  

---



### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/)** (1 messages): 

moonride303: https://x.com/jaseweston/status/1795978611784089799
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1245869560498884681)** (6 messages): 

- **Borsch with a twist**: A member shared an interesting borsch recipe made with **pork and beef bone stock, cow heart, mayonnaise, and monosodium glutamate**. They paired it with cucumber, Borodinsky bread, apples, an orange, and fruit tea sweetened with **stevia** instead of sugar.

- **Debate on Stevia vs. Sugar**: One member questioned the use of stevia in the recipe, asserting that "real sugar is good for you". However, the original poster retorted that "sugar is poison" and "we aren't evolved to eat it."

- **Questioning Stevia's Use**: Another member humorously noted that humans might be even less evolved to handle stevia unless they are "indigenous people from the Amazon rainforest."

- **Infrequent Berry Consumption**: When asked about the consumption of berries, the original poster clarified that berries should be a *"seasonal and infrequent treat."*
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1246000436788596828)** (16 messages🔥): 

- **RLHF/DPO's surprising limitation**: A shared [tweet](https://x.com/_angie_chen/status/1796220428345573399) suggested that RLHF/DPO does not effectively produce policies that assign a higher likelihood to preferred responses. This insight sparked confusion and debate among members, questioning the efficacy of these algorithms in preference learning.

- **Debate on ranking accuracy**: Members discussed the concept of "ranking accuracy," seeking to understand its definition and implications in the context of reference models. It was concluded that preference learning aims to train models for high ranking accuracy, meaning they can rank preferred outputs higher than dispreferred ones, but this doesn't seem to hold true universally.

- **Overfitting revelation**: After multiple members reviewed the thread, it was suggested that the discovery might show that it's possible to overfit when using RLHF/DPO methods. This opinion was countered by pointing out that the discussion could involve deeper mathematical proof about why DPO, especially without additional measures like NLL, underperforms.

- **Comparing techniques**: Members mentioned that problems similar to those found in DPO are also present in PPO, which is used by OpenAI. Another user highlighted that DPO models, despite their issues, tend to perform better than SFT (Supervised Fine-Tuning) alone.

**Link mentioned**: <a href="https://x.com/_angie_chen/status/1796220428345573399">Tweet from Angelica Chen (@_angie_chen)</a>: New work w/@sadhikamalladi, @lilyhzhang, @xinyichen2, @QiuyiRichardZ, Rajesh Ranganath, @kchonyc: Contrary to conventional wisdom, RLHF/DPO does *not* produce policies that mostly assign higher likeli...

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1245815172183101563)** (63 messages🔥🔥): 

- **NeurIPS hosts Model Merging competition**: A competition related to model merging, announced on [Twitter](https://x.com/LChoshen/status/1796256513519989102), will take place at NeurIPS, with a prize of $8K. [Sign up here](https://llm-merging.github.io/), with sponsorship from Hugging Face and Sakana AI Labs.
- **Coller Prize for AI and animal communication**: A $500K prize is offered for successfully using AI to communicate with animals, detailed via [tweet](https://x.com/nearcyan/status/1796243343288189179) and [further info](https://coller-dolittle-24.sites.tau.ac.il/). A related YouTube video by Aza Raskin about the Earth Species Project was also mentioned.
- **Llama Model Finetuning Resource**: Upscaled version of Llama-3 described as nearly lossless except for TruthfulQA, recommended for finetuning ([Tweet](https://x.com/dudeman6790/status/1796382605086015993), [HuggingFace link](https://huggingface.co/Replete-AI/Llama-3-11.5B-V2)). One user commented, *"TruthfulQA is a meaningless benchmark anyways so that's pretty nice"*.
- **Google AI Overviews update**: Google brings AI Overviews to U.S. users, enhancing search satisfaction and engagement (Blog post: [Google AI Overviews](https://blog.google/products/search/ai-overviews-update-may-2024/)). Despite some erroneous overviews, they claim higher quality clicks to webpages and continuous feedback loop improvements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://x.com/dudeman6790/status/1796382605086015993">Tweet from RomboDawg (@dudeman6790)</a>: It seems like besides TruthfulQA, there is was basically no loss in the upscaled model we made a while back. So if anyone wants to finetune using an upscaled version of llama-3 then the base version w...</li><li><a href="https://huggingface.co/mistralai/Codestral-22B-v0.1">mistralai/Codestral-22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=rjvsl0mhqTk">Aza Raskin and Earth Species Project</a>: Aza Raskin describes the mission of Earth Species Project and latest developments in using AI to unlock communication of non-human species.</li><li><a href="https://x.com/nearcyan/status/1796243343288189179">Tweet from near (@nearcyan)</a>: $500K prize for successfully using AI to converse with animals. exciting!  link: https://coller-dolittle-24.sites.tau.ac.il/</li><li><a href="https://x.com/tommyfalkowski/status/1796098620430619038?s=46">Tweet from Tommy Falkowski (@TommyFalkowski)</a>: Caret is my new favorite way of interacting with LLMs! It&#39;s an obsidian plugin that makes it easy to have branching conversations! I&#39;m using it with @ollama with Llama3:8b and it works really ...</li><li><a href="https://blog.google/products/search/ai-overviews-update-may-2024/">AI Overviews: About last week</a>: Here’s what happened with AI Overviews, the feedback we&#x27;ve received, and the steps we’ve taken.</li><li><a href="https://github.com/WecoAI/aideml/tree/main">GitHub - WecoAI/aideml: AIDE: the Machine Learning CodeGen Agent</a>: AIDE: the Machine Learning CodeGen Agent. Contribute to WecoAI/aideml development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1246078429535932418)** (1 messages): 

- **LLMs generate most web content on-the-fly**: As explained, *pretty much everything is made by a LLM*, with web pages often created in real-time as users see them "loading". Despite this, LLMs struggle with making very long or oversized webpages due to context limitations.
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1245864638890115082)** (2 messages): 

- **Milvus Lite: A compact, efficient vector store for Python**: Milvus Lite, a lightweight vector database for Python, is now available. Further details and how to get started can be found [here](https://t.co/QCOtM5CVc5).

- **Milvus Lite runs in-process in Python**: Milvus Lite integrates seamlessly with AI development stacks like LangChain and LlamaIndex, making it suitable for environments with limited resources. Instructions to incorporate it into your AI applications can be found [here](https://milvus.io/docs/milvus_lite.md).

- **Build a full web app with Omakase RAG Orchestrator**: A new web app template for building scalable Retrieval Augmented Generation (RAG) applications using Django, LlamaIndex, and Google Drive has been introduced. It features a full RAG API, data source management, and user access control [details here](https://t.co/kx3DhxfDZu).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/QCOtM5CVc5">Introducing Milvus Lite: Start Building a GenAI Application in Seconds</a>: no description found</li><li><a href="https://t.co/ckEiBVEbqK">Milvus Vector Store - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1245816948873494752)** (72 messages🔥🔥): 

- **Prototype Retrieval Systems Are Tricky**: A user shared their experience prototyping a retrieval system using SimpleStore classes and sought advice on how to transfer data to RedisStores. Others suggested various methods, with one recommended creating an "IngestionPipeline" for "transfer documents" to handle upserts and data transfer more efficiently.

- **React App Response**: A member had issues with their RAG app’s simplified final output compared to more detailed observations. Suggestions included providing additional instructions as context using `ReActAgent.from_tools(..., context="Ensure your final answers are detailed.")`.

- **OpenAI Certificate Verification Issue**: A Dockerized OpenAI setup using FastAPI and Nginx faced SSL certificate verification issues. One suggestion was to try a different base image to potentially solve the problem.

- **Understanding Vector Store Options**: Users discussed different vector store query options like `DEFAULT`, `SPARSE`, `HYBRID`, and `TEXT_SEARCH` in postgres, with some confusion about their functionalities. It was concluded that both `text` and `sparse` use `tsvector`.

- **Editing Document Objects**: A member sought ways to manually edit Document objects, especially following errors in PDF extraction. The community suggested directly modifying `document.text` or using external text editors and reinserting the edited text back into the Document object.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/?h=vectorstorequery#llama_index.core.vector_stores.types.VectorStoreQueryMode">Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/VespaIndexDemo/?h=vespa">Vespa Vector Store demo - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/vector_stores/postgres/?h=postgres+hybrid#improving-hybrid-search-with-queryfusionretriever">Postgres Vector Store - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1245823532458446970)** (20 messages🔥): 

- **Luxia v1.2 contamination confirmed**: A member highlighted that **Luxia 21.4b v1.2** shows a 29% contamination increase over GSM8k tests compared to v1.0. They used a widely known contamination test, noting that other benchmarks like ARC and Wino showed 0.0 contamination.
- **NeurIPS model merging competition**: A competition related to model merging was announced for NeurIPS 2023 with potential for breakthroughs in model selection. The event aims to attract participants with a prize of $8K and invites the community to [sign up and participate](https://llm-merging.github.io/).
- **Scaling joint generator and reward model idea**: A member discussed scaling up their model that combines cDPO for the LM head and a novel reward head for pruning generated samples based on rewards, particularly for fine-grained metrics like toxicity and creativity.
- **Query on embedding storage efficiency**: A member sought advice on efficiently storing millions of T5 embeddings for large-scale dataset sharing, mentioning the excessive space taken by T5 XL embeddings in fp16 configuration. They considered quantization, which halved the size with ~95% accuracy, but it still remained too large for their needs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.2/discussions/1">saltlux/luxia-21.4b-alignment-v1.2 · contamination results v1.0 vs v1.2 on GSM8K</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1245835205697077329)** (34 messages🔥): 

- **Excitement over NeurIPS Model Merging Competition**: Members shared an announcement about a [Model Merging competition](https://x.com/LChoshen/status/1796256513519989102) at **NeurIPS**, with $8K in prize money. Interested parties were directed to the competition's [sign-up page](https://llm-merging.github.io/) and Discord.

- **CLIP Text Encoder breakthrough**: A paper claiming a non-suck CLIP text [encoder has been lauded](http://arxiv.org/abs/2405.20204). Pretraining the text encoder with masked language modeling followed by combining text-image and text-text contrastive losses shows promise.

- **Discussion around Alignment Paper - Direct Preference Heads**: An engaging conversation occurred around a new [alignment paper](https://arxiv.org/abs/2405.20053) and its unique approach using Direct Preference Heads. The discussion highlighted its departure from LaMDA's reliance on LM head for ratings, aiming to disentangle reward signals from the output distribution.

- **Poseidon Multiscale Operator Transformer for PDEs**: A new transformer model named **Poseidon** for learning solution operators of PDEs was introduced with promising results. The [model](https://arxiv.org/abs/2405.19101) excels in sample efficiency and accuracy, showing strong performance on various PDE tasks through a novel training strategy.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://arxiv.org/abs/2201.08239">LaMDA: Language Models for Dialog Applications</a>: We present LaMDA: Language Models for Dialog Applications. LaMDA is a family of Transformer-based neural language models specialized for dialog, which have up to 137B parameters and are pre-trained on...</li><li><a href="https://arxiv.org/abs/2405.19101">Poseidon: Efficient Foundation Models for PDEs</a>: We introduce Poseidon, a foundation model for learning the solution operators of PDEs. It is based on a multiscale operator transformer, with time-conditioned layer norms that enable continuous-in-tim...</li><li><a href="http://arxiv.org/abs/2405.20204">Jina CLIP: Your CLIP Model Is Also Your Text Retriever</a>: Contrastive Language-Image Pretraining (CLIP) is widely used to train models to align images and texts in a common embedding space by mapping them to fixed-sized vectors. These models are key to multi...</li><li><a href="https://arxiv.org/abs/2405.18669">Zipper: A Multi-Tower Decoder Architecture for Fusing Modalities</a>: Integrating multiple generative foundation models, especially those trained on different modalities, into something greater than the sum of its parts poses significant challenges. Two key hurdles are ...</li><li><a href="https://arxiv.org/abs/2405.19893">Similarity is Not All You Need: Endowing Retrieval Augmented Generation with Multi Layered Thoughts</a>: In recent years, large language models (LLMs) have made remarkable achievements in various domains. However, the untimeliness and cost of knowledge updates coupled with hallucination issues of LLMs ha...</li><li><a href="https://arxiv.org/abs/2405.20053">Would I Lie To You? Inference Time Alignment of Language Models using Direct Preference Heads</a>: Pre-trained Language Models (LMs) exhibit strong zero-shot and in-context learning capabilities; however, their behaviors are often difficult to control. By utilizing Reinforcement Learning from Human...</li><li><a href="https://x.com/sirbayes/status/1796441322263294435?s=46">Tweet from Kevin Patrick Murphy (@sirbayes)</a>: I am delighted to share our recent paper: https://arxiv.org/abs/2405.19681. It can be thought of as a version of the Bayesian Learning Rule, extended to the fully online setting. This was a super fun ...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1245816087816306699)** (7 messages): 

- **Transformers excel with MLPs**: A user argued that **transformers** use the **best data dependence** by leveraging MLPs. They emphasized the superiority of MLPs in scaling, stating, *"any other data dependence would not use mlps so it would be worse at scale imo."*
- **Softmax attention debate**: Discussions questioned the necessity of **softmax weighted routing** in data-dependent aggregation. One member noted that while alternatives exist, "softmax attention is Lindy" due to extensive previous trials.
- **Alternatives to softmax still resemble current methods**: A counterpoint mentioned that replacing softmax would not introduce a novel mechanism but rather form a "function attention" maintaining a **context-dependent T x T matrix**. This suggests that truly distinct methodologies might still face fundamental similarities to current attention mechanisms.
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1246074642846711808)** (9 messages🔥): 

- **Gemma-2b-it results discrepancy sparks discussion**: A user reported not being able to replicate the 17.7% results for Gemma-2b-it, achieving only ~9%, even with majority voting. They linked to a [discussion on Hugging Face](https://huggingface.co/google/gemma-2b-it/discussions/44), seeking others' experiences.

- **Evaluation prompt remains elusive**: There was a discussion about the evaluation of results being 8-shot, as seen with Llama-2. One user pointed out that the exact evaluation prompt wasn't released and suggested checking the evaluation of Mistral-7b for additional context.

- **Using Colab for GSM8K evaluation**: A link to a [Google Colab notebook](https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb#scrollTo=cXoCKMi9EXir) was shared to aid in replicating the results, but it required signing in and specific CUDA configurations.

- **Phi3-mini evaluation aligns better**: A user mentioned that running Phi3-mini in **lm_eval** produced results closer to the reported numbers, with less significant differential. This provided a bit more confidence in the evaluation process despite other inconsistencies noticed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/google-deepmind/gemma/blob/">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb#scrollTo=cXoCKMi9EXir">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

gpantaz: Thank you for the reply 🙂
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1245816875036704780)** (13 messages🔥): 

- **Mojo Package Manager Still in Early Stages**: Members discussed the lack of updates on the Mojo roadmap regarding a package manager. They referenced past GitHub discussions such as [Discussion #413](https://github.com/modularml/mojo/discussions/413) and [Discussion #1785](https://github.com/modularml/mojo/discussions/1785) which show the proposed plans for project manifest and build tools.

- **Compiling Expertise Exploration**: A member asked for recommendations on learning materials about compilers. Another member suggested [this syllabus](https://mcyoung.xyz/syllabus) which includes a comprehensive list of readings on compilers, systems programming, and other topics.

- **Upcoming Mojo Community Meeting**: The announcement for the second Mojo Community Meeting, set to feature talks on Basalt, Compact Dict, and Pandas for Mojo, with updates on Mojo Stdlib. Details were provided in a [Google Document](https://modul.ar/community-meeting-doc), and the Zoom link for the meeting is [available here](https://modul.ar/community-meeting-zoom).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mcyoung.xyz/syllabus"> Syllabus &middot; mcyoung </a>: no description found</li><li><a href="https://github.com/modularml/mojo/discussions">modularml/mojo · Discussions</a>: Explore the GitHub Discussions forum for modularml mojo. Discuss code, ask questions &amp; collaborate with the developer community.</li><li><a href="https://github.com/modularml/mojo/discussions/413">[RFC] Allow Importing Modules via URLs · modularml/mojo · Discussion #413</a>: Overview One of Mojo&#39;s main priorities is solving the &quot;two language problem,&quot; which means that it must function for both app development use cases, but also one off scripts. Dependency m...</li><li><a href="https://github.com/modularml/mojo/discussions/1785">[Proposal] Mojo project manifest and build tool · modularml/mojo · Discussion #1785</a>: Hi all, please check out this proposal for a Mojo project manifest and build tool. As mentioned on the proposal itself, we&#39;re looking to hear from the Mojo community: Do you agree with the motivat...</li><li><a href="https://modul.ar/community-meeting-doc">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...</li><li><a href="https://modul.ar/community-meeting-zoom.">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1796606227981726168>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1245962974536138803)** (3 messages): 

- **Mojo handles SHA-256 and 256-bit integers**: A member queried whether **Mojo** can perform SHA-256 hashing and manage 256-bit integers. Another member confirmed, saying it can be done if one is familiar with the implementation and suggested representing a 256-bit integer using `SIMD[DType.int64, 4]`.
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1245874454832222219)** (28 messages🔥): 

- **Mojo speeds up K-Means clustering**: A [YouTube video](https://www.youtube.com/watch?v=3bg5YBCcuWA) showcases a step-by-step guide to porting K-Means clustering from Python+NumPy to pure Mojo, promising a 250x speedup. This video was highlighted as a great example for learning Mojo.

- **Mojo aims to be Python's superset**: One user noted that the Modular team intends for Mojo to be a superset of Python, making existing Python code work seamlessly and benefiting from features like Cython underpinnings and NumPy compatibility.

- **Tuple handling in Mojo**: A user shared code to get elements from a tuple and asked for help with recent changes, questioning if a possible bug exists given the errors faced.

- **ndarray initialization methods in Mojo**: A member sought advice on whether to use `memset` or vectorized methods for initializing ndarrays in Mojo, leading to a discussion highlighting that `memset` is likely more optimized due to its lower-level implementation, as seen in its [source code](https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/memory/memory.mojo).

- **Improving string builder performance in Mojo**: A user shared a [string builder implementation](https://github.com/thatstoasty/gojo/blob/nightly/gojo/strings/builder.mojo#L134) that claims to be significantly faster than string concatenation, and sought feedback to ensure it avoids memory issues. Another user suggested zero-copy approaches and provided insights into utilizing vectored writes and avoiding unnecessary data moves, with references to [iovec and writev](https://man7.org/linux/man-pages/man2/writev.2.html).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=3bg5YBCcuWA">Speed up K-Means clustering by porting Python implementation to Mojo🔥</a>: In this video we&#39;ll share a step-by-step guide to porting kmeans clustering from Python+NumPy to pure Mojo for huge (250x) speedup! How? Mojo is Pythonic in ...</li><li><a href="https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/memory/memory.mojo">mojo/stdlib/src/memory/memory.mojo at bf73717d79fbb79b4b2bf586b3a40072308b6184 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/thatstoasty/gojo/blob/nightly/gojo/strings/builder.mojo#L134">gojo/gojo/strings/builder.mojo at nightly · thatstoasty/gojo</a>: Experiments in porting over Golang stdlib into Mojo. - thatstoasty/gojo</li><li><a href="https://github.com/thatstoasty/gojo/blob/nightly/tests/test_performance.mojo#L7">gojo/tests/test_performance.mojo at nightly · thatstoasty/gojo</a>: Experiments in porting over Golang stdlib into Mojo. - thatstoasty/gojo
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1245933316276883537)** (1 messages): 

- **Clarifying Mojo-based Model Training in Max**: One member inquired whether implementing the backward pass, an optimizer, and basic training handling like training loops in Mojo would suffice to train a model using Max exclusively in Mojo. They sought to understand if these components were the only missing elements for conducting model training.
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1245820819750391850)** (9 messages🔥): 

- **Debugging joy: finding the bug in reversed Dict items**: A member enthusiastically shared that they found and fixed an undefined behavior in `reversed(Dict.items())` and `reversed(Dict.values())`—potentially solving weeks of flaky tests. They attached a [GitHub PR link](https://github.com/modularml/mojo/pull/2896) that details the fix.
- **Assertions save the day**: Another member highlighted the importance of enabling assertions in unit tests to avoid flaky tests, reinforcing the value of diligent debugging practices.
- **Major nightly Mojo compiler release**: The newest release for the nightly Mojo compiler version `2024.5.3112` was announced, featuring various changes such as fixes in changelogs, removing certain `math` functions, and renaming others. Detailed updates can be found in the [raw diff](https://github.com/modularml/mojo/compare/8ae83916ebc7b3134948f466c0f56ee3e5569062...3df5fb4f9d3dd7cc5018aa8a160c3714b1a4f81e) and the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **Main vs Nightly branch PR issues**: A discussion illuminated that a bug was caused by starting a PR on the `main` instead of the `nightly` branch, directing to [another GitHub PR](https://github.com/modularml/mojo/pull/2881#issuecomment-2141082033) as evidence. This crucial identification aids in streamlining future PR submissions.
- **Celebration of quick assist**: Another member humorously suggested they might have been the first to notice and share excitement about the new updates, showcasing community engagement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/pull/2896">[stdlib] Fix UB in `reversed(Dict.values())` and `reversed(Dict.items())` by gabrieldemarmiesse · Pull Request #2896 · modularml/mojo</a>: Finally found the culprit in the flakyness that plagued us since a few week in the test_reversed.mojo. The actual bug: When iterating over a list in reverse order, we should start at len(my_list) -...</li><li><a href="https://github.com/modularml/mojo/pull/2881#issuecomment-2141082033>">Update functions.ipynb by ratulb · Pull Request #2881 · modularml/mojo</a>: Typo - current to currently.
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1245823184092270686)** (41 messages🔥): 

- **Expose LangChainAPI publicly for LangGraph projects**: A user asked for a quick and inexpensive way to expose the LangChainAPI endpoints to a public location rather than keeping them on localhost. They were interested in using LangServe but had not been invited yet.
  
- **Speeding up LangGraph configurations**: A user inquired about configuration options to speed up LangGraph, as it takes a considerable amount of time to load and for a single agent to start. The conversation hints at looking for optimization choices.

- **Passing summaries into ChatPromptTemplate**: A user asked how to pass the summary from "memory" into `ChatPromptTemplate` and received guidance on using `MessagesPlaceholder` with the appropriate variable name. Specific implementation details and relevant GitHub links were shared for further reference.

- **Integrating ConversationSummaryMemory with RunnableWithMessageHistory**: Another user sought help on integrating `ConversationSummaryMemory` with `RunnableWithMessageHistory` in Python. Detailed code examples and GitHub resources were shared to explain this process.

- **Reducing input tokens by summarizing chat history**: A user faced issues with `RunnableWithMessageHistory` using too many tokens due to the chat history. The solution provided involved summarizing the chat history before running it through the chain to manage token usage better.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://js.langchain.com/v0.1/docs/modules/memory/types/summary/#usage-with-an-llm>)">Conversation summary memory | 🦜️🔗 Langchain</a>: Now let&#x27;s take a look at using a slightly more complex type of memory - ConversationSummaryMemory. This type of memory creates a summary of the conversation over time. This can be useful for cond...</li><li><a href="https://js.langchain.com/v0.1/docs/use_cases/chatbots/memory_management/#summary-memory>)">Memory management | 🦜️🔗 Langchain</a>: A key feature of chatbots is their ability to use content of previous conversation turns as context. This state management can take several forms, including:</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/#dict-with-single-key-for-all-messages-input-messages-output>)">Add message history (memory) | 🦜️🔗 LangChain</a>: The RunnableWithMessageHistory lets us add message history to certain types of chains. It wraps another Runnable and manages the chat message history for it.</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/#in-memory>)">Add message history (memory) | 🦜️🔗 LangChain</a>: The RunnableWithMessageHistory lets us add message history to certain types of chains. It wraps another Runnable and manages the chat message history for it.</li><li><a href="https://python.langchain.com/v0.1/docs/use_cases/chatbots/memory_management/#summary-memory>).">Memory management | 🦜️🔗 LangChain</a>: A key feature of chatbots is their ability to use content of previous conversation turns as context. This state management can take several forms, including:</li><li><a href="https://python.langchain.com/v0.1/docs/use_cases/chatbots/memory_management/#summary-memory>)">Memory management | 🦜️🔗 LangChain</a>: A key feature of chatbots is their ability to use content of previous conversation turns as context. This state management can take several forms, including:</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/migrate_agent/">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 LangChain</a>: Here we focus on how to move from legacy LangChain agents to LangGraph agents.</li><li><a href="https://github.com/langchain-ai/langchain/issues/16525>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/#persistent-storage>)">Add message history (memory) | 🦜️🔗 LangChain</a>: The RunnableWithMessageHistory lets us add message history to certain types of chains. It wraps another Runnable and manages the chat message history for it.</li><li><a href="https://github.com/langchain-ai/langchain/issues/1971>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/1136>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/16448>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1246026373563416596)** (5 messages): 

- **RunnableLambda solves the issue**: One user shared they solved their issue using **RunnableLambda**, suggesting to wrap the chain into a function and create a Pydantic BaseModel class with input and chat history attributes.
- **LangServe website error reported**: A user pointed out an error with the LangServe website, providing this [link](https://www.langchain.com/langserve) for reference.
  

---


### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1245990593268682813)** (2 messages): 

- **User seeks help constructing prompts with multiple variables**: A member asks about constructing prompts with multiple variables from **Langgraph state**. They provide an example prompt: *"As a {topic} expert, answer questions {questions} using these information: {knowledge}"*, asking when and how to pass these variables.
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1245862267321450588)** (2 messages): 

- **Learn Crew AI Custom Tools**: Check out this [YouTube video](https://youtu.be/Hc5yiUQKh2Q) titled "Crew AI Custom Tools Basics." It covers creating custom tools for agents to automate tasks and enhance LLM generative AI productivity.
  
- **AIQuire unveils document insights**: A member introduced AIQuire, an AI-powered tool designed to help users understand and extract answers from complex documents. They invited others to try out AIQuire at [aiquire.app](https://aiquire.app) and provide feedback to aid further development.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aiquire.app/">AIQuire: Empower yourself with the intelligence of your data.</a>: no description found</li><li><a href="https://youtu.be/Hc5yiUQKh2Q">Crew Ai Custom Tools Basics.</a>: Creating a custom tool for your agents can be the best way to automate tasks and also enable your LLM  generative ai to become more productive.Crew Ai Docume...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1245820019741560923)** (48 messages🔥): 

- **Huge potential for Fineweb**: Discussions centered around implementing Fineweb, leveraging Puppeteer to visit URLs and dump images alongside the text content for use in contextual input to ground a VLM. [Fineweb details](https://vxtwitter.com/iurimatias/status/1796260746310910309).

- **StabilityAI's Decision Sparks Reactions**: StabilityAI plans to release a 512px model instead of their full suite of SD3 checkpoints. Members discussed how this decision might affect model improvements and shared opinions on the necessity of high GPU resources.

- **Positional Embeddings in DiT Models**: Some technical discourse on how positional embeddings work for DiT models and their potential for handling different resolutions. Members noted that despite standard implementations, positional embeddings tend to mode collapse at higher resolutions.

- **Open-Source Tools Get Community Excited**: Open-source projects like tooncrafter have excited the community with new features despite minor issues. Discussions highlighted the community's optimism about quickly improving these tools.

- **AI Strategy by Eliezer Yudkowsky’s Institute**: Eliezer Yudkowsky's institute published a "2024 Communication Strategy," aiming to shut down AI development as a precautionary measure. [Read more](https://x.com/drtechlash/status/1796562490232557658?s=46&t=M3cR_nfDo7QCuM4xOvwNFA).

**Link mentioned**: <a href="https://x.com/drtechlash/status/1796562490232557658?s=46&t=M3cR_nfDo7QCuM4xOvwNFA">Tweet from Nirit Weiss-Blatt, PhD (@DrTechlash)</a>: Eliezer Yudkowsky&#39;s institute published its &#34;2024 Communication Strategy&#34;  The main goal (as he argued in TIME magazine) is to 🔻shut down🔻 AI development.  So, let&#39;s take a look at t...

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1245835029599223818)** (2 messages): 

- **NeurIPS to host a Model Merging competition**: Members announced a [Model Merging competition at NeurIPS](https://x.com/LChoshen/status/1796256513519989102) interested participants can stand a chance to win **$8K** and contribute to LLMs innovation. Check out the [official Discord](https://discord.gg/dPBHEVnV) and [sign-up page](https://llm-merging.github.io/) for more details.
- **RB-Modulation for Image Stylization and Composition**: A new method called **RB-Modulation** was introduced, offering a training-free plug-and-play solution for stylizing and composing content-style images. More details are available on the [official project page](https://rb-modulation.github.io/) including the [paper](https://arxiv.org/abs/2405.17401) and [code (coming soon)](https://github.com/LituRout/RB-Modulation).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/LChoshen/status/1796256513519989102">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: 🚨 Model Merging competition @NeurIPSConf!🚀  Can you revolutionize model selection and merging?Let&#39;s create the best LLMs!🧠✨  💻Come for science 💰Stay for $8K 💬Discord: https://discord.gg/dPBH...</li><li><a href="https://rb-modulation.github.io/">RB-Modulation</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1245912628388302869)** (22 messages🔥): 

- **Exploring Yuan2 Model on Huggingface**: Members discussed the [Yuan2 model](https://huggingface.co/papers/2311.15786) from Huggingface, expressing interest in training it and sharing the provided link for further examination.
  
- **Preference Training Comparison**: There was a conversation about various training methods, with members mentioning *HF recommends SFT on the data then DPO* and contrasting it with *ORPO*, which can *"be done by itself with stronger effect than DPO with SFT."* A link to the ORPO paper was provided: [arxiv.org/abs/2403.07691](https://arxiv.org/abs/2403.07691).

- **Advantages of ORPO Over Traditional Methods**: Members discussed the benefits of ORPO, highlighting its ability to eliminate the need for an additional preference alignment phase. This is reflected as *“a monolithic preference optimization without reference model”*.

- **Potential Integration of ORPO in Axolotl**: Inquiry about adding ORPO to Axolotl was raised, with one member suggesting that ORPO *"should work"*, indicating the possibility of trying it out in the system.

**Link mentioned**: <a href="https://huggingface.co/papers/2311.15786">Paper page - YUAN 2.0: A Large Language Model with Localized Filtering-based
  Attention</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1245918973522870373)** (12 messages🔥): 

- **Struggles with fine-tuning for text classification**: A user is having trouble fine-tuning models for Spanish entity categorization, using llama3 and mistral. They provided specific instruction steps and noted that while training succeeds, inference performs poorly.

- **Assistance with inference setup**: Another user inquired if the dataset and inference methods could be shared to better understand the fine-tuning issues. The original poster has not yet responded with further details.

- **CUDA 12.1 installation woes and a solution**: A member needed help installing CUDA 12.1 on Ubuntu 22 with an NVIDIA driver and initially faced issues. The resolution was installing CUDA 12.1 using the run file without the driver installation.

- **Framework for fine-tuning embedding models**: A member asked for recommendations on frameworks suitable for efficiently fine-tuning RoBERTa-style embedding models. The query remains unanswered.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1246108724414910547)** (2 messages): 

- **Confusion on Formatting for LLama3**: A user asked whether the `text` field should include specific tokens like `<|start_header_id|>` when using Alpaca format with LLama3. Another member advised setting the `chat_template` in the config as Axolotl will handle the formatting automatically.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1246115875023487038)** (6 messages): 

- **Axolotl Installation Troubles**: A member reported encountering an error during the installation process of **Axolotl**. Despite following general troubleshooting steps like ensuring Python version compatibility and using a virtual environment, the issue persisted.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L102L137)">axolotl/README.md at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=8b79a987-f932-4434-89c2-978e36c820b0)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1245886949827543150)** (7 messages): 

- **Member seeks help with model overfitting**: *"need to stop training as my model is starting to overfit, how can i do this?"*. Another member suggests implementing an early stopping mechanism and provides a detailed code example using the **Hugging Face Accelerate library**.
- **GitHub link for early stopping configuration**: A member shares a [GitHub link](https://github.com/OpenAccess-AI-Collective/axolotl/blob/d4f6c65e4c4d6899a9283a046fa922b937a0ab25/docs/config.qmd#L325C1-L327C27) for further instructions on configuring an early stopping mechanism in **OpenAccess-AI-Collective/axolotl**. The original poster acknowledges and appreciates the help: *"Thanks will definitely try it!"*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/d4f6c65e4c4d6899a9283a046fa922b937a0ab25/docs/config.qmd#L325C1-L327C27">axolotl/docs/config.qmd at d4f6c65e4c4d6899a9283a046fa922b937a0ab25 · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=aa1cb115-b5f9-438c-b03e-4b76a0f862dc)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1245862057585147925)** (5 messages): 

- **DiscoLeo Needs Retraining to Fix EOS Token Issue**: A member reported that **merging ChatML with state-of-the-art models like Hermes Theta** causes issues with the EOS token, getting the model stuck in an endless loop 20% of the time. They suggested retraining DiscoLeo with ChatML and requested fine-tuning data.
- **Preference for ChatML Over Llama-3 Template**: Members discussed the preference for **ChatML over the original Llama-3 instruct template**. One argued that finetuning on the German base using ChatML is better, especially since the target model (Hermes Theta) already uses ChatML.
  

---


### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1246032456503529523)** (23 messages🔥): 

- **Interest in IBM's Granite models**: Members queried the performance and availability of the new IBM "Granite" models in both English and German. IBM's models, including the **Lab version**, are listed on [watsonx.ai](https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=models-foundation), sparking curiosity due to their limited mention in open-source circles.
- **Confusion over IBM model versions**: Discussions highlighted confusion due to IBM's range of **Granite models** such as 3B/8B enhanced Llama versions and 20B/34B Starcoder-based models. A user pointed out the various versions available, including a **7B base, instruct, and instruct accelerator version (medusa speculative decoding)**.
- **Merlinite model gaining attention**: The **Merlinite 7B** model was noted for its interesting aspects, with mentions of its upload to Ollama and testing under the **Lab method**. Users expressed interest in comparing its abilities in German to models like Granite.
- **Generated data and benchmarks**: Concerns were raised about the quality of **AI-generated data** with some users noting it was mostly unsatisfactory. Benchmarks such as **EQ Bench on q4km gguf quants** were mentioned to be below common expectations, highlighting interest in comparing new "enhancing without catastrophic forgetting" approaches.
- **Community sharing resources**: A resourceful link to a [Medium article on InstructLab](https://medium.com/@syeda9118/instructlab-ever-imagined-the-ease-of-tuning-pre-trained-llms-3331ccea8d88) was shared, summarizing the ease of tuning pre-trained LLMs, reflecting ongoing community efforts to better understand and implement these models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ollama.com/sroecker/granite-7b-lab">sroecker/granite-7b-lab</a>: A Granite 7B model trained using the LAB method</li><li><a href="https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=models-foundation">Foundation models built by IBM</a>: no description found
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1245815214771933336)** (10 messages🔥): 

- **Google's New Compute Resources Ignite Interest**: In reference to rumors of a Google-Apple deal, a post shared a [Twitter link](https://x.com/_arohan_/status/1796228607255396423) suggesting Google is expanding its compute resources. The link mentions another cluster arriving, indicating increased capacity for training AI models.

- **Court Verdict and its Implications**: Users humorously discussed a high-profile court case, referencing hashtags like #miscarriage-of-justice and #they-hate-him-cuz-they-aint-him. Concerns were raised about how the verdict might influence future political events and potential insurrection attempts.

- **OpenAI Resurrects Robotics Team**: OpenAI has formally re-established its robotics team after abandoning its initial efforts in 2020. This initiative, [reported on Twitter](https://x.com/sarahnemerson/status/1796264432215146807?s=46) and in a [Forbes article](https://www.forbes.com/sites/kenrickcai/2024/05/30/openai-robotics-team/), mentions the new team has been active for around two months and is looking to hire research engineers.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_arohan_/status/1796228607255396423">Tweet from rohan anil (@_arohan_)</a>: @giffmana No worries, another cluster has arrived today in case you want to train some more</li><li><a href="https://x.com/sarahnemerson/status/1796264432215146807?s=46">Tweet from sarah emerson (@SarahNEmerson)</a>: New --  OpenAI is formally resurrecting its robotics team after abandoning efforts to build general purpose robots in 2020.   Its new team has been around for ~2 months and is currently hiring researc...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1245815628762451988)** (7 messages): 

- **Skepticism on GPT-3.5 API Availability**: A user remarked on the repeated notion that "Anyone can build a chatbot off gpt 3.5 api," critical of its frequent use and accuracy. 

- **Clarification on GPT-3.5 Timelines**: Another user clarified that while GPT-3.5-002 was available longer, GPT-3.5-003 came out with/after ChatGPT, suggesting discrepancies in public understanding of availability.

- **Confusion Over Deleted Documentation**: Users expressed frustration over the deletion of a certain page related to GPT-3.5 and described the naming scheme as confusing. One user stated they had reported the issue months ago but no changes were made.

- **Concerns About Information Deletion**: A user voiced concerns that deleting information to serve a convenient narrative is problematic. Another hinted it might be a classic AI safety measure, while one user suggested it could be an oversight, offering to share an archived version of the site.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1245875840441384981)** (8 messages🔥): 

- **Sergey recruits for physical intelligence**: Nathan Lambert shared that Sergey tried to recruit him for a project on physical intelligence, mentioning they have a "cool setup". He indicated openness to others joining, saying "If anyone is interested lmk".
- **Interest in RL and publishing**: A community member expressed interest in joining if there were opportunities to discuss Reinforcement Learning (RL) and publishing, although they were nervous about keeping up with the group’s expertise.
- **Research support and robot usage**: Nathan Lambert reassured that the group is practical and would adapt, supporting research due to a need for more people using robots. The potential member humorously doubted their suitability, mentioning their only experience with robots being a robot vacuum.
  

---


### **Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1246143273311932456)** (2 messages): 

- **Murky Waters podcast episode released**: Nathan Lambert and Tom discuss various recent AI policy happenings in a new podcast episode titled [Murky Waters in AI Policy](https://retortai.com/episodes/murky-waters-in-ai-policy). Topics include California's "anti open source" 1047 bill, the senate AI roadmap, Google's search snafu, OpenAI's activities, and reader feedback.
- **Missed the open house for the California bill**: Nathan Lambert mentions that he intended to attend the open house for California's AI bill but couldn't make it. Further details are not provided in the messages above.



**Link mentioned**: <a href="https://retortai.com/episodes/murky-waters-in-ai-policy">The Retort AI Podcast | Murky waters in AI policy</a>: Tom and Nate catch up on many AI policy happenings recently. California's 

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1245920575201738823)** (11 messages🔥): 

- **Cohere aims for long-term sustainability**: A member remarked, "Cohere is probably far ahead if you consider long term sustainability as a metric." They emphasized the value of focusing on specific tasks like extracting information from invoices rather than solving vast problems like curing diseases immediately.
  
- **AGI still in its infancy**: The discussion highlighted that achieving AGI is only the beginning, posing the question, "Then what... We're still at the 'Then what...' stage." The current state of AI is seen more as a "CPU" rather than a comprehensive system.

- **Server refresh and new community incentives**: A member announced a refresh to the server, simplifying the channel layout and introducing new roles and rewards. They mentioned, "we are taking down the server levels & instead the most active server members just became cohere regulars."

- **Cohere introduces Coral, the AI chatbot**: A test confirmed Coral, an AI chatbot, is up and running with the response, "I am Coral, an AI chatbot trained to assist human users by providing thorough responses." The successful interaction led to appreciation from the tester.

- **New emojis and stickers for community engagement**: The server will feature new emojis and reactions, and members can customize emojis by contacting the moderator. This change aims to enhance user interaction and fun within the community.
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1245920587512021056)** (2 messages): 

- **No-Code Workflow Builder for AI Models Seeks Feedback**: A startup is working on a **no-code workflow builder** designed to mix and match AI models and aims for future capabilities to auto-build workflows and auto-select the best LLMs. They are seeking feedback on why users do not continue using the platform and offer a **$10 incentive** for survey participants. 
- **Community Encouragement and Support**: A member praised the approach and offered feedback without participating in the incentive, expressing a willingness to spend 10 minutes on the platform. They appreciated the form and etiquette of the outreach post, highlighting it as a good way to engage the community.
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1245843049108607036)** (12 messages🔥): 

- **Embedding Adapters improve retrieval performance**: Members discussed the potential of embedding adapters (*"a quick win for retrieval"*), with a link to a [Chroma research report](https://research.trychroma.com/embedding-adapters). The report evaluates applying linear transforms to embeddings for improved retrieval applications.

- **Frozen Embeddings akin to Embedding Adapters**: Another discussion likened embedding adapters to Frozen Embeddings used by the Vespa team, referencing a [Vespa blog article](https://blog.vespa.ai/leveraging-frozen-embeddings-in-vespa-with-sentence-transformers/). Frozen Embeddings help avoid tedious updates to embeddings in dynamic environments like e-commerce.

- **PwC's massive ChatGPT Enterprise contract**: A tweet highlighted PwC's purchase of ChatGPT Enterprise licenses for ~100,000 employees, estimating the contract at $30M/year. Members debated the price, with guesses ranging from $8/user/month to previously heard rates of $65/user/month.

- **Google Gemini updates released**: Google Developers have announced the general availability of Gemini 1.5 Flash and 1.5 Pro, including a 1,000 RPM limit for Flash, new tuning options, and JSON Schema mode in the API. More details can be [found here](https://developers.googleblog.com/en/gemini-15-pro-and-15-flash-now-available/).

- **TLBrowse open-sourced**: TLBrowse, which merges Websim with TLDraw, has been open-sourced by its creator. Users can generate imagined websites on an infinite @tldraw canvas with a [free hosted version](https://tlbrowse.com) available to try.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.vespa.ai/leveraging-frozen-embeddings-in-vespa-with-sentence-transformers/">Leveraging frozen embeddings in Vespa with SentenceTransformers</a>: How to implement frozen embeddings approach in Vespa using SentenceTransformers library and optimize your search application at the same time.</li><li><a href="https://agent.ai">agent.ai | The Professional Network for A.I. Agents</a>: Get things done with the help of AI agents</li><li><a href="https://x.com/officiallogank/status/1796213739366322431?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Good news for @Google Developers:  - Gemini 1.5 Flash and 1.5 Pro are now GA - Gemini 1.5 Flash now has a 1,000 RPM limit - Gemini 1.5 Flash tuning announced - JSON Schema mode is now available in the...</li><li><a href="https://x.com/sawyerhood/status/1796193457662214651?s=">Tweet from Sawyer Hood (@sawyerhood)</a>: I&#39;ve just open-sourced my tlbrowse demo! You can generate imagined websites on an infinite @tldraw  canvas.</li><li><a href="https://x.com/tanayj/status/1795858607004598351?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Tanay Jaipuria (@tanayj)</a>: PwC purchasing ChatGPT Enterprise licenses for ~100,000 employees to become OpenAI&#39;s largest ChatGPT enterprise user.  Assuming ~$25/mo/seat, that&#39;s a $30M/yr contract</li><li><a href="https://research.trychroma.com/embedding-adapters">Embedding Adapters</a>: no description found</li><li><a href="https://x.com/sawyerhood/status/1796193457662214651?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sawyer Hood (@sawyerhood)</a>: I&#39;ve just open-sourced my tlbrowse demo! You can generate imagined websites on an infinite @tldraw  canvas.</li><li><a href="https://tlbrowse.com">tlbrowse</a>: no description found
</li>
</ul>

</div>
  

---



### **AI Stack Devs (Yoko Li) ▷ #[events](https://discord.com/channels/1122748573000409160/1131651713204498583/1245835549294461041)** (1 messages): 

- **Rosebud AI hosts "Book to Game" Game Jam**: Roberto from the Rosebud AI team announced a new game jam, *"Book to Game,"* using Phaser JS on their AI Game Maker platform. The event encourages participants to create interactive games based on literary works, with a **$500 prize pool**.
- **Details and Participation**: The submission deadline for the game jam is July 1st, 12:00 AM PST. More details and participation guidelines are available on [Rosebud AI's Twitter](https://x.com/Rosebud_AI/status/1796273820044595368) and their [Discord](https://discord.gg/rosebud-ai).


**Link mentioned**: <a href="https://x.com/Rosebud_AI/status/1796273820044595368">Tweet from Rosie @ Rosebud AI 🌹 (@Rosebud_AI)</a>: Turn your favorite story into a game using AI! 📚 👾  Get ready for our third Game Jam: “Book to Game”. Use Rosebud Game Maker to transform a literary work into an interactive game and bring stories t...

  

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1245816064525205554)** (5 messages): 

- **Manufacturing update pinned**: Members are directed to check the pinned message in <#1194880263122075688> for a manufacturing update. It's crucial to stay informed through the pinned messages.
- **Interest in Codestral model**: A member asked if anyone has tried Codestral yet, stating it "seems like a good model." This indicates a growing interest in exploring new models within the community.
- **Struggles with HuggingFace integration**: A member expressed frustration in using HuggingFace models with OpenInterpreter, noting only one successful attempt using the command `interpreter -y --model huggingface/mistralai/Mistral-7B-Instruct-v0.3`. Another member suggested creating a detailed post in <#1149558876916695090> to seek further assistance.
- **Potential scam alert**: A member issued a "red alert" concerning a potential scam, tagging another user for caution. This underscores the importance of vigilance in the community.
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1245835801821188188)** (6 messages): 

- **Codestral model generates buzz**: A member inquired if anyone had tried Codestral, mentioning it seems like a promising model. The request for user experiences and insights remains unanswered.
- **Query on O1 Android functionality**: Multiple members expressed interest in getting O1 Android working, with one asking if it needs to be installed in Termux. Responses to this inquiry were not provided.
- **License limitations noted**: A member highlighted that the usage of Codestral is limited to non-commercial purposes only. This point was brought up without extended discussion.


  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1245915925404975135)** (3 messages): 

- **AutoGPT integrates Llamafile**: A member from AutoGPT announced collaboration with another user to integrate **Llamafile** into their system.
- **Questions about content block support**: The same member inquired if **Llamafile supports content blocks** in user messages, similar to OpenAI's functionality. Another member questioned if **llama.cpp** supports them.
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1245825241570345061)** (2 messages): 

- **PRS Event at Netflix Attracts Attention**: A member announces they will be attending the **PRS event at Netflix** tomorrow and asks if anyone else will be there. Another member confirms they will also be attending.
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1245819417783173230)** (2 messages): 

- **Curiosity Sparks Over Mistral 45GB Model**: A member speculated about the composition of the 45GB model, suggesting it might have a heavier weight on English and smaller portions dedicated to programming languages. They expressed excitement to see the actual breakdown.

- **MNPL Compliance Dilemmas for Codestral**: A member raised concerns about finding a legal use case for Codestral under the **Mistral AI Non-Production License (MNPL)**. The **MNPL** seems to limit sharing derivative or hosted works with others, which the member found restrictive and disappointing.
  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

helplesness: Why is tensoflow better than pytorch?
  

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

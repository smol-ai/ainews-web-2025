---
id: d4a15eb1-67d6-49c2-a284-29d362a987ca
title: not much happened today
date: '2025-02-11T03:56:45.222082Z'
original_slug: ainews-not-much-happened-today-3076
description: >-
  AI news from February 2025 covers new model releases, advancements in
  reasoning, multilingual TTS, and industry impact. Highlights include Google's
  Gemini 2.0, ZyphraAI's Zonos TTS, Hugging Face's math dataset, Huginn-3.5B
  reasoning model, and the Anthropic Economic Index.
companies:
  - google
  - zyphraai
  - hugging face
  - anthropic
models:
  - gemini-2.0
  - zonos
  - huginn-3.5b
topics:
  - vision-language reasoning
  - multilingual-text-to-speech
  - voice-cloning
  - math-reasoning
  - latent-reasoning
  - ai-industry-impact
  - economic-index
---


<!-- buttondown-editor-mode: plaintext -->**a quiet day.**

> AI News for 2/7/2025-2/10/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**210** channels, and **11464** messages) for you. Estimated reading time saved (at 200wpm): **1218 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Just like [Meta's Coconut](https://arxiv.org/abs/2412.06769) before it, [Huginn's Latent Reasoning Model](https://x.com/iScienceLuvr/status/1888792081382137966) made a splash today. We agree with [Jeremy](https://x.com/jeremyphoward/status/1888815600958656793) and [Andrej](https://x.com/karpathy/status/1835561952258723930) that the best RL will probably not be in English, but we didn't choose this as feature story because presumably DeepSeek already tried that for r1 ([our coverage here](https://buttondown.com/ainews/archive/ainews-deepseek-r1-o1-level-open-weights-model/)) and didn't find it worth the tradeoff of not being able to read the thoughts.


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**AI Model Releases and Advancements**

- **Google's Release of Gemini 2.0 Flash Thinking Experimental 1-21**: [DeepLearningAI](https://twitter.com/DeepLearningAI/status/1889026549275344986) announced that Google released Gemini 2.0 Flash Thinking Experimental 1-21, the latest version of its vision-language reasoning model, featuring an expanded **1 million-token context window** and a **user-readable chain of thought**. The update improves accuracy across science, math, and multimedia benchmarks, surpassing **DeepSeek-R1** but trailing **OpenAI's o1** in some areas.

- **Release of Zonos - Multilingual TTS Model with Voice Cloning**: [@reach_vb](https://twitter.com/reach_vb/status/1889015111890997479) highlighted that [ZyphraAI](https://twitter.com/ZyphraAI) released **Zonos**, an **Apache 2.0 licensed**, multilingual **Text-to-Speech model** with **instant voice cloning** capabilities. The model supports **zero-shot TTS with voice cloning** using a 10-30 second speaker sample, **audio prefix inputs** for enhanced speaker matching, and controls for **speaking rate, pitch, frequency, audio quality, and emotions**. It runs at **~2x real-time speed on an RTX 4090** and is available on the [Hugging Face Hub](https://t.co/sZndYJ5caM).

- **Hugging Face Releases OpenR1-Math-220k Dataset**: [@_lewtun](https://twitter.com/_lewtun/status/1889002019316506684) and [@reach_vb](https://twitter.com/reach_vb/status/1888994979218915664) announced the release of **OpenR1-Math-220k**, a large-scale **math reasoning dataset** based on **Numina Math 1.5**, containing **220K math problems** and **800K raw R1 reasoning traces** generated on **512 H100 GPUs**. The dataset is **Apache 2.0 licensed**, encouraging the community to **fine-tune models** and advance mathematical reasoning capabilities.

**Advancements in AI Reasoning and Models**

- **Introduction of Huginn-3.5B Latent Reasoning Model**: [Tom Goldstein](https://twitter.com/tomgoldsteincs/status/1888980680790393085) introduced **Huginn-3.5B**, an open-source reasoning model that **reasons implicitly in latent space** without producing extra chain-of-thought tokens at test time. Trained on **800B tokens**, Huginn-3.5B demonstrates significant improvements on reasoning tasks like **GSM8K**, outperforming larger models despite its smaller size.

- **Debate on Human-Readable Reasoning Traces**: [Jeremy Howard](https://twitter.com/jeremyphoward/status/1888815600958656793) predicted that training AI systems to produce **human-readable reasoning traces** will eventually seem bizarre, comparing it to requiring a diffusion image model to output an image sequence that matches an artist's brush strokes. He suggests that future models may internalize reasoning in ways that are not easily interpretable by humans.

- **Scaling Test-Time Compute with Latent Reasoning**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1888792081382137966) discussed a new language model architecture capable of improving performance on reasoning benchmarks by **implicitly reasoning in latent space**. The model scales test-time computation without the need for specialized training data, supporting small context windows and capturing reasoning not easily represented in words.

**AI's Impact on Industry and Economy**

- **Anthropic Launches the Anthropic Economic Index**: [AnthropicAI](https://twitter.com/AnthropicAI/status/1888954156422992108) launched the **Anthropic Economic Index**, aiming to understand AI's impact on the economy over time. Their first paper analyzes millions of anonymized **Claude conversations** to reveal how AI is being used across different tasks and occupations. Key findings include:

  - **AI use tilts towards augmentation (57%) over automation (43%)**.
  - **Software and technical writing tasks** have the highest AI usage.
  - AI adoption is most common in **medium-to-high income jobs**, with low usage in very-high and low-income jobs.
  - The dataset and ongoing analysis aim to track patterns of change as AI evolves.

- **Integration of DeepSeek Models into Cloud Services**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1888812805932875828) noted that China's three big telecom operators are rushing to **integrate DeepSeek models into cloud services**, potentially **freezing their own LLM projects**. This indicates a strategic shift towards adopting existing powerful models rather than developing new ones independently.

**AI Tools, Development, and Research**

- **Combining Vector Search and Knowledge Graphs**: [Qdrant Engine](https://twitter.com/qdrant_engine/status/1888860549775065437) shared insights on building with **Neo4j and Qdrant** to create a smarter **GraphRAG**, which leverages **vector search for semantic retrieval** and **graph traversal for structured reasoning**. This approach aims for greater accuracy with **less LLM dependency**.

- **Using TensorFlow's ImageDataGenerator**: [DeepLearningAI](https://twitter.com/DeepLearningAI/status/1888967700476555324) highlighted the use of **TensorFlow’s ImageDataGenerator** to handle real-world images that vary in size, position, and contain multiple subjects. This tool automatically **labels, resizes, and batches images** for training, enhancing the efficiency of data pipelines when working with diverse image datasets.

- **Exploring AI's Limitations with Unknown Unknowns**: [@hardmaru](https://twitter.com/hardmaru/status/1888958032039813469) discussed a paper titled "**Evolution and The Knightian Blindspot of Machine Learning**", which argues that the process of evolution equips organisms to navigate **unexpected events** ("unknown unknowns"), a capability that current AI systems struggle to replicate.

**Community Insights and Events**

- **Sam Altman's Three Observations**: [Sam Altman](https://twitter.com/sama/status/1888695926484611375) shared ["Three Observations"](https://t.co/Ctvga5vfMy), offering insights likely related to AI developments, industry trends, or human potential. The content emphasizes the ongoing evolution and impact of technology.

- **AI Summit in Paris and Open-Source Advocacy**: [Clement Delangue](https://twitter.com/ClementDelangue/status/1888920800528331091) announced arrival in Paris for the **AI Summit**, emphasizing efforts to **push open-source AI** alongside team members like [Irene Solaiman](https://twitter.com/IreneSolaiman). The focus is on **doubling investments in France** with an emphasis on open-source, robotics, and applications.

- **Discussions on Chinese AI Progress**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1888781956315316378) provided a timeline reflecting skepticism towards Chinese AI advancements, noting a progression from initial underestimation to recognition of solid engineering efforts.

**Memes/Humor**

- **OpenAI's Super Bowl Ad and Rivalry with Google**: [Sam Altman](https://twitter.com/sama/status/1888703820596977684) humorously remarked on the challenge of surpassing Google with "man, still a long way to go to run down google 🥺" and mentioned "also our ad, it’s really good" in a conversation with [@xprunie](https://twitter.com/sama/status/1888702632509735199). [@teortaxesTex](https://twitter.com/teortaxesTex/status/1888879403867787485) playfully critiqued OpenAI employees for hyping their high-production-value ad, comparing OpenAI to an Apple-type corporation.

- **The Hackbot Singularity and TEDx Talk**: [@rez0__](https://twitter.com/rez0__/status/1888801773558665464) mentioned that **"the hackbot singularity is coming"** and shared his TEDx talk titled "**The Rise of AI Hackbots**" available on [YouTube](https://t.co/PHKxESqnmr), discussing the implications of AI in cybersecurity and hacking.

- **Humorous Takes on AI and Society**: [@teortaxesTex](https://twitter.com/teortaxesTex) shared several tweets with humorous or satirical reflections on AI developments and societal observations, including commentary on public transit externalities, the robustness of nation-states, and playful jabs at corporate strategies in AI advancement.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. DeepSeek-R1/V3 Performance Showcase on Xeon and GPU**

- **671B DeepSeek-R1/V3-q4 on a Single Machine (2× Xeon + 24GB GPU) – Up to 286 tokens/s Prefill & 14 tokens/s Decode** ([Score: 623, Comments: 165](https://reddit.com/r/LocalLLaMA/comments/1ilzcwm/671b_deepseekr1v3q4_on_a_single_machine_2_xeon/)): The **KTransformers team** announces support for **DeepSeek-R1/V3**, achieving up to **286 tokens/s for prefill** using a **CPU/GPU hybrid inference** system, which is significantly faster than llama.cpp. They highlight the use of **Intel AMX-accelerated kernels** and a **selective expert activation method** for performance enhancement, and emphasize that offloading computational tasks to the GPU aligns with DeepSeek's architecture, offering substantial speed improvements.
  - **CPU and GPU Configuration**: The setup uses an **Intel® Xeon® Gold 6454S** with **32 cores per socket** and **8x DDR5-4800** for each socket, paired with a **4090D GPU**. The system costs approximately **$10K**, with discussions on whether a heavy CPU setup is better than a heavy GPU setup, considering the **Xeon's cost** and potential downgrades to more affordable options.
  - **Performance and Optimization**: The **DeepSeek V3/R1** model's performance is enhanced through CPU/GPU hybrid inference, though adding more GPUs does not currently offer significant improvements due to the model's sparsity. The model's footprint can be reduced significantly through optimizations, with one user reporting a **3.38 times improvement** in prompt processing speed over **llama.cpp**, thanks to using an **RTX 4090**.
  - **Platform Support and Future Plans**: There is interest in optimizing for **Apple Silicon** and **Intel GPUs**, though the current focus is on open-sourcing version 0.3 and executing planned optimizations. **AMD** is supported but lacks the **AMX optimization** for prefill speed, and there are discussions about the potential benefits of using **48GB VRAM** and future support for **AMD Matrix Core (AMC)**.


- **[Deepseek’s AI model is ‘the best work’ out of China but the hype is 'exaggerated,' Google Deepmind CEO says. “Despite the hype, there’s no actual new scientific advance.”](https://www.cnbc.com/2025/02/09/deepseeks-ai-model-the-best-work-out-of-china-google-deepmind-ceo.html)** ([Score: 329, Comments: 244](https://reddit.com/r/LocalLLaMA/comments/1ilsd9g/deepseeks_ai_model_is_the_best_work_out_of_china/)): **Google DeepMind CEO** commented on the **DeepSeek AI model**, describing it as the "best work" from China but stated the hype around it is exaggerated. He emphasized that despite the excitement, there is no actual new scientific advancement in the model.
  - Commenters criticized **DeepMind CEO Demis Hassabis** for downplaying the **DeepSeek AI model**, arguing that its open-source nature and engineering efficiencies, such as **reduced costs** and **training efficiency**, are significant advancements. They accused Hassabis of **dishonesty by omission**, failing to acknowledge the model's open weights and cost-effectiveness as substantial contributions.
  - Some commenters highlighted that **DeepSeek's engineering achievements** are notable, even if they don't constitute a scientific breakthrough. They pointed out that **DeepSeek** achieved competitive performance with **ChatGPT** at a fraction of the cost, challenging assumptions about China's AI capabilities and suggesting that the model's efficiency and open-source approach are valuable innovations.
  - Discussions also focused on the broader implications of **open-source AI models** like DeepSeek, emphasizing the potential for democratizing AI technology. Commenters noted that **Google's** reluctance to open-source their models contrasts with the openness of DeepSeek, leading to debates about the role of open-source in advancing AI research and its geopolitical impact.


**Theme 2. Innovative Techniques in LLM Model Optimization**

- **TL;DR of Andrej Karpathy’s Latest Deep Dive on LLMs** ([Score: 382, Comments: 48](https://reddit.com/r/LocalLLaMA/comments/1ilsfb1/tldr_of_andrej_karpathys_latest_deep_dive_on_llms/)): **Andrej Karpathy** has released a **3-hour, 31-minute** video on LLMs like **ChatGPT**, described as a "goldmine of information." A summary article condensing the key insights into **15 minutes** is available [here](https://anfalmushtaq.com/articles/deep-dive-into-llms-like-chatgpt-tldr), and the original video can be found on [YouTube](https://www.youtube.com/watch?v=7xTGNNLPyMI).
  - **Fine-tuning and Prompt Engineering**: Discussions highlight the importance of fine-tuning smaller open-source models like **llama-3B** and emphasize prompt engineering as crucial for optimizing LLM applications. **Andrej Karpathy**'s work and the article by **Anfal Mushtaq** are noted for covering these topics in depth, alongside strategies to reduce hallucinations in model outputs.
  - **Data Processing and Tokenization**: The article and video explore the preprocessing of vast internet text data, including rigorous filtering and tokenization using techniques like **Byte Pair Encoding**. This process is essential for the effective training of LLMs, balancing creativity with accuracy in model predictions.
  - **Humor and Engagement**: Several comments playfully summarize the article and video in progressively shorter formats, including a one-minute recap, a 50-word summary, and even a haiku, showcasing community engagement and humor in distilling complex information.


- **[New paper gives models a chance to think in latent space before outputting tokens, weights are already on HF - Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)** ([Score: 112, Comments: 16](https://reddit.com/r/LocalLLaMA/comments/1imca0s/new_paper_gives_models_a_chance_to_think_in/)): **Scaling LLM Compute with Latent Reasoning** discusses a novel approach in AI model computation, allowing models to perform reasoning in latent space before generating output tokens. This method, detailed in the paper titled **"Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach,"** has its weights already available on **Hugging Face**.
  - **Adaptive Compute and Latent Reasoning**: A notable discussion revolves around per-token adaptive compute, where models adjust computational effort based on token importance, potentially impacting AI benchmarks significantly within the next **6-12 months**. This method allows models to "think" more on complex tokens while expending less on simpler ones, suggesting a significant shift in AI processing efficiency.
  - **Recurrent Depth Approach and Weight Sharing**: There's speculation on the implementation details, particularly whether the **R blocks** share weights and how these are sampled at test time. This recurrent depth approach, as discussed, could enhance the model's reasoning accuracy with increased recurrent steps, similar to efforts by **OpenAI**.
  - **Availability and Comparisons**: The weights for this approach are accessible on **Hugging Face**, with additional resources available on [GitHub](https://github.com/seal-rg/recurrent-pretraining). Comparisons are made to **Meta's** similar research, though they did not release weights, emphasizing the value of open-access research artifacts for practical exploration and understanding of AI's latent reasoning capabilities.


**Theme 3. Orange Pi AI Studio Pro PC: A New Player in AI Hardware**

- **[Orange Pi AI Studio Pro mini PC with 408GB/s bandwidth](https://www.reddit.com/gallery/1im141p)** ([Score: 315, Comments: 91](https://reddit.com/r/LocalLLaMA/comments/1im141p/orange_pi_ai_studio_pro_mini_pc_with_408gbs/)): The **Orange Pi AI Studio Pro mini PC** has been released, featuring an impressive **408GB/s bandwidth**. This development is significant for AI engineers looking for high-performance computing solutions in compact form factors.
  - **Hardware vs. Software Support**: The **Orange Pi AI Studio Pro mini PC** is criticized for its lack of reliable software support, with users highlighting past issues with **Orange Pi's** software ecosystem. Concerns include the absence of updates, proprietary drivers, and poor community support, making it less appealing despite its hardware capabilities.
  - **Economic Considerations**: Discussions emphasize the cost-effectiveness of pairing accelerators with DDR memory for AI workloads, as seen with setups like **Deepseek R1** on EPYC systems costing under **$10,000**, compared to more expensive VRAM setups. The **Orange Pi** device, priced around **$2,150**, is seen as potentially good value for its specifications, but skepticism remains about its practical utility without robust software support.
  - **Alternative Solutions and Comparisons**: Users suggest alternatives like **older NVIDIA GPUs** and **Intel NUCs** for better support and performance, noting the challenges of using NPUs in less mainstream systems like the **Qualcomm Snapdragon X series**. The **Orange Pi** device's potential is overshadowed by these alternatives due to its niche status and anticipated software hurdles.


**Theme 4. Scaling Retrieval-Augmented Generation (RAG) for Massive Datasets**

- **How to scale RAG to 20 million documents ?** ([Score: 137, Comments: 136](https://reddit.com/r/LocalLLaMA/comments/1im35yl/how_to_scale_rag_to_20_million_documents/)): To scale **RAG (Retrieval-Augmented Generation)** for 20 million documents, focus on optimizing latency, efficient embedding, and robust indexing strategies. Explore techniques like distributed computing, advanced indexing structures, and parallel processing to manage large-scale document retrieval efficiently.
  - The discussion highlights the challenges and strategies for scaling **RAG** with 20 million documents, emphasizing the importance of efficient **vector databases** like **Weaviate**, **PGVector**, and **Pinecone** for handling large-scale data. **HNSW indexing** and **Reranking strategies** such as **Reciprocal Rank Fusion (RRF)** are recommended to optimize retrieval quality and performance.
  - Participants debate the merits of **fine-tuning** versus **context injection**, with some arguing that fine-tuning is costly and less effective for large datasets. **DataIsLoveDataIsLife** suggests a pragmatic approach using **stella_en_400M_v5** for embedding and **MiniBatchKMeans** for clustering, estimating a processing cost of **$1,000-$20,000**.
  - The use of **GraphRAG/LightRAG** approaches and **graph databases** is proposed for better results, while others suggest leveraging existing search engines for retrieval. **Data ingestion** and **indexing** are also discussed, with suggestions for using middleware layers to manage data efficiently and experimenting with tools like **parade db** for high-scale search.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. Gemini 2 Flash: The New Benchmark for AI Translation Efficiency**

- **Did I save 93% of cost by using OpenAI for translation?** ([Score: 160, Comments: 47](https://reddit.com/r/OpenAI/comments/1ilzfiu/did_i_save_93_of_cost_by_using_openai_for/)): The post author compares translation costs, noting that **Azure charges approximately €9.60 per 1 million characters**, while **OpenAI's GPT-4o-mini costs around €0.70 per 1 million characters**, potentially saving 93% in costs. The calculation includes the need to translate words from a given sentence, requiring the input word in the output, with costs broken down as **€0.30 x 2 per million characters plus €0.075 for input**.
  - Discussions highlight the potential cost savings of using **Gemini 2 Flash** for translations, which offers better multi-lingual support and costs less than other options. Users note that with rate limiting and free tier usage, costs can be minimized or even eliminated, as detailed in [Google's pricing](https://ai.google.dev/pricing#2_0flash) with specifics on token costs and free tier limits.
  - Several users discuss strategies to further reduce translation costs, such as utilizing **batch processing** and **prompt caching**, which can cut costs significantly by allowing non-real-time processing. A link to the [OpenAI batch API documentation](https://platform.openai.com/docs/guides/batch) is provided for reference on how this can achieve up to 50% cost reduction.
  - There is a conversation about the reliability and accuracy of various translation models, with some users suggesting **open-source models** for particular use cases, despite their slower speeds. Concerns are raised about translation quality, emphasizing the importance of having a human in the loop for large-scale translations to ensure accuracy.


**Theme 2. OpenAI's Innovative Branding with Super Bowl Ad**

- **[OpenAI's $14 million SuperBowl ad](https://v.redd.it/v10i8668t7ie1)** ([Score: 2722, Comments: 601](https://reddit.com/r/OpenAI/comments/1ilusr7/openais_14_million_superbowl_ad/)): **OpenAI** is reportedly investing **$14 million** in a **Super Bowl ad** strategy, indicating a significant marketing push. This move could suggest an effort to increase public awareness and engagement with their AI technologies.
  - Many commenters believe the **Super Bowl ad** effectively positions **ChatGPT** as a major technological milestone, similar to Apple's 1984 ad, by associating it with historical advancements like fire and the moon landing. This approach aims to create brand awareness and emotional connection rather than focus on specific functionalities.
  - There is a divide in opinions about the ad's effectiveness; some argue it missed an opportunity to showcase **ChatGPT's** capabilities, while others see it as a strategic move to establish **brand recognition** and **public acceptance** of AI. The ad's creative and aesthetic quality received praise, with some noting its appeal to **Millennials** through elements like the **Ratatat Neckbrace remix**.
  - The discussion highlights the complexity of marketing AI technologies, with some emphasizing the importance of **brand positioning** and **awareness**, while others question the decision not to demonstrate practical uses of **ChatGPT** in the advertisement. Critics argue that the ad may not effectively reach those unfamiliar with **OpenAI** or **ChatGPT**.


**Theme 3. ChatGPT's Ascent to Top Global Website Traffic Rankings**

- **[ChatGPT is now the 6th most visited site in the world as of January 2025, per Similarweb. The AI chatbot now holds 2.33% of global internet traffic, marking a 5.91% monthly surge.](https://www.cryptotimes.io/2025/02/10/chatgpt-surpasses-netflix-reddit-now-6th-most-visited-site/)** ([Score: 139, Comments: 7](https://reddit.com/r/OpenAI/comments/1im0tyb/chatgpt_is_now_the_6th_most_visited_site_in_the/)): **ChatGPT** has become the **6th most visited site globally** as of **January 2025**, according to **Similarweb**, capturing **2.33% of global internet traffic** and experiencing a **5.91% monthly increase** in visits.
  - Commenters discuss that **OpenAI** is gaining significant data from **ChatGPT** interactions, which enhances their brand recognition and potential subscriber base. This data is invaluable beyond mere traffic statistics.
  - **OpenAI** has achieved substantial brand recognition with **ChatGPT**, likened to historical brand dominance like **Motorola's Droid**. Commenters note that **ChatGPT** is becoming synonymous with "AI" for the general public, unlike lesser-known competitors like **Claude**.
  - A shared **Google Trends** graph highlights the disparity in search interest between **ChatGPT** and **Claude**, emphasizing **ChatGPT's** dominant position in public awareness.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking


**Theme 1. Unsloth AI's Rise and Community Focus**

- **Unsloth Rockets to GitHub Stardom**:  [Unsloth AI Celebrates GitHub Trending](https://github.com/trending) achieved #1 trending repository on GitHub within a year, marking significant community growth and impact. The community acknowledges **Unsloth's** contributions, particularly to **Deepseek-R1**, with potential integrations already in progress.
- **REINFORCE Reasoning Methods Under Scrutiny**:  [Reasoning LLM using REINFORCE Notion Doc](https://charm-octagon-74d.notion.site/Reasoning-LLM-using-REINFORCE-194e4301cb9980e7b73dd8c2f0fdc6a0) sparks debate on novelty, with members noting existing **Unsloth** implementations. Skepticism arises around the originality of the approach, questioning its added value over current methods already available in **Unsloth**.
- **Model Merging Faces Headwinds**:  Merging models into MoEs draws skepticism, triggering discussions on potential downsides and limitations. The community debates potential learning losses in long output formats with shared structures, which could impede training for specific tasks.

**Theme 2.  No-Code AI Platforms & Tools Emerge**

- **Spark Engine Launches No-Code AI Powerhouse**: [Spark Engine v1 is Live](https://sparkengine.ai/) debuts with **80+ AI models**, offering no-code text, music, and video generation capabilities. Developers express interest in integrating infrastructure like **Unsloth** to further enhance the no-code AI ecosystem.
- **Dataset Tools Gets AI-Powered EXIF Upgrade**: [Dataset Tools EXIF Viewer on GitHub](https://github.com/Ktiseos-Nyx/Dataset-Tools) enhances EXIF data viewing and adds support for GGUF and JPEG formats. Developers leverage AI to improve features and collaborate on code optimization for the project.
- **Markdrop Python Package Drops PDF Data Bombs**: [Markdrop PDF to Markdown Converter on GitHub](https://github.com/shoryasethia/markdrop) arrives as a new Python package for converting PDFs to Markdown, extracting images, and using AI for descriptions.  The package quickly gains traction, hitting **7,000+ installs** in a month.

**Theme 3.  Model Performance and Hardware Debates Heat Up**

- **Qwen 2.5 Leaves Llama 8B in the Dust**:  **Qwen 2.5** outpaces **Llama 8B** in speed, particularly with larger models like 32B, due to better optimizations. Users suggest **Qwen 2.5** is the superior choice for those with capable hardware.
- **LM Studio Users Wrestle with Model Loading Errors**:  **LM Studio** users grapple with *'NO LM Runtime found for model format'* errors, indicating hardware limitations.  Users are advised to share system specs and screenshots and match model sizes to system capabilities based on [LM Studio Docs](https://lmstudio.ai/docs/basics/import-model).
- **M4 Ultra vs M2 Ultra: The Great Mac Chip Showdown**:  A debate sparks over the value of waiting for **M4 Ultra** versus buying **M2 Ultra** for efficient model operation.  Users are concerned about rising service costs amid uncertain model performance on **M2 Ultra**.

**Theme 4.  OpenAI Model Dynamics and User Concerns**

- **Gemini Swallows Context Whole, ChatGPT Chokes**:  **Gemini’s** massive **1-2 million token** context window gains popularity over **ChatGPT's** 32k/128k token limits. Users prefer **Gemini** for complex tasks, despite **ChatGPT** limitations and connection errors.
- **GPT-4 Feeling Dumber, Users Demand Better Prompts**:  **GPT-4** is perceived as weaker, requiring more sophisticated prompting, while connection errors plague **ChatGPT**. Users are reporting ongoing **connection errors** and feeling that **GPT-4** is not as capable as it once was.
- **DeepSeek's 'Unlimited' Turns Out to Have Limits**:  **DeepSeek's** *'unlimited'* usage is revealed to have restrictions, with high use flagged as abusive, raising transparency questions. Users express concerns about the term *'unlimited'* and inconsistent policy application.

**Theme 5.  Coding Tools and Agentic Workflows Evolve**

- **Cursor IDE Explodes with MCP Server Mania**:  **Cursor IDE** users dive deep into **MCP servers**, particularly **Perplexity MCP server**, for enhanced coding assistance.  Users explore setups and troubleshoot installation issues across different operating systems.
- **Agent Mode in Cursor Hailed as Debugging Hero**:  **Agent Mode** in **Cursor** is praised for debugging prowess, outshining standard coding commands with direct model communication.  Users find integrating diverse **LLMs** boosts coding experience, especially with real-time assistance.
- **Aider Chat History Balloons, Token Limits Loom**:  **Aider's chat history** grows excessively, reaching **25k tokens**, sparking concerns about token limit overruns. Users discuss potential bugs and prompt caching effectiveness and performance impacts.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Achieves GitHub Trending Status**: **Unsloth AI** has become the #1 trending repository on [GitHub](https://github.com/trending) within a year, celebrating its tools and resources.
   - The community acknowledges **Unsloth's** contribution to **Deepseek-R1**, with components potentially already integrated or available in current projects.
- **REINFORCE Reasoning Sparks Debate**: Concerns arose over a document on **Reasoning LLM using REINFORCE** at [this link](https://charm-octagon-74d.notion.site/Reasoning-LLM-using-REINFORCE-194e4301cb9980e7b73dd8c2f0fdc6a0), questioning its novelty.
   - Members noted that an identical implementation already exists in **Unsloth**.
- **Model Merging Faces Skepticism**: Interest in merging several effective models into a single mixture of experts (MoE) was met with skepticism, leading to discussion about potential pitfalls and limitations.
   - Discussion occurred regarding the potential loss of learning in long output formats that share common structures, which may hinder the training of specific tasks.
- **Spark Engine Integrates No-Code AI**: **Spark Engine v1** has been launched with over **80 AI models**, generating *text, music, and videos* at [SparkEngine.ai](https://sparkengine.ai/).
   - The developers expressed a desire to potentially integrate more infrastructure like **Unsloth** into the Spark Engine platform to foster advancements in the no-code AI realm.
- **Dataset Curation Dominates Model Performance**: It was emphasized that **80%** of a model's performance hinges on careful **dataset curation**, with *one member noted,* 'There is no such thing as redundant research - you learn from every paper.'
   - Another member is experimenting with Lora settings to develop a metacognitive first-person reasoning format.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Kokoro TTS Speaks C#**: A member released a C# library for **Kokoro TTS**, enabling plug & play integration on .NET platforms, available on [GitHub](https://github.com/Lyrcaxis/KokoroSharp).
   - The library promises a **multilingual** experience with all voices packaged in a convenient format, supporting fast local TTS inference and works across multiple platforms.
- **Dataset Tools Gets EXIF and AI Upgrade**: The Dataset organizer and **EXIF Viewer** received updates, enhancing its capabilities to view advanced EXIF data and supporting formats like GGUF and JPEG, available on [GitHub](https://github.com/Ktiseos-Nyx/Dataset-Tools).
   - The developer utilized AI tools to assist in the project, enhancing its features while collaborating with others for code optimization.
- **Spark Engine Ignites AI Sandbox**: The Spark Engine v1 was released after a year-long public beta, providing **over 80 models** for various AI tasks available at [sparkengine.ai](https://sparkengine.ai/).
   - The platform offers free credits daily and integrates with Hugging Face, making a robust no-code environment for users to experiment with AI capabilities.
- **Markdrop Extracts PDF Data**: A new Python package called **Markdrop** was introduced, designed for converting PDFs to Markdown with features like image extraction and AI-powered descriptions, accessible on [GitHub](https://github.com/shoryasethia/markdrop).
   - In just a month, it has achieved over **7,000 installs**, showcasing its popularity among users looking for document manipulation tools.
- **go-attention Implements Transformer in Pure Go**: A member shared their project, **go-attention**, which showcases the first full attention mechanism and transformer built in pure Go, highlighting its unique capabilities on [GitHub](https://github.com/takara-ai/go-attention).
   - The project invites others to check out examples and explore the potential of serverless implementations in Go programming.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 2.5 Smokes Llama 8B in Speed**: Users compared **Qwen 2.5** and **Llama 8B**, citing that **Qwen** offers faster response times due to optimization, especially with larger models like 32B.
   - The discussion suggested that **Qwen 2.5** is preferable with adequate hardware.
- **LM Studio Users Battle Model Loading**: Users encountered issues loading models into **LM Studio**, receiving errors like *'NO LM Runtime found for model format'*, indicating hardware limitations.
   - The suggested solution was to provide system specs and screenshots for better assistance, as well as matching model size to system capabilities according to [LM Studio Docs](https://lmstudio.ai/docs/basics/import-model).
- **Debate on M4 Ultra vs M2 Ultra ensues**: A debate emerged about the value of waiting for the **M4 Ultra** versus purchasing the **M2 Ultra** for efficient model operation.
   - Concerns centered on rising costs for existing services amidst uncertain performance of models on the **M2 Ultra**.
- **PCI-E Risers Raise Eyebrows**: A user inquired about using **PCI-E riser cables** to install additional GPUs and the performance implications, particularly with **A5000** cards.
   - A suggestion was made to repurpose old cases as GPU holders for enhanced cooling and space management.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini Gains Large Context Popularity**: **Gemini’s** capability to handle **1-2 million tokens** has made it popular, especially compared to ChatGPT’s 32k and 128k tokens, enhancing usability for complex tasks.
   - Users appreciate **Gemini’s** flexible features, making it a preferred choice for detailed work, despite concerns over **ChatGPT’s** limitations.
- **GPT-4 Feels Weaker Nowadays**: Members feel **GPT-4** is less capable, requiring better prompting to yield good results, but earlier models might have set a perception of inferiority in complex tasks.
   - Several users also reported ongoing **connection errors** while using ChatGPT, raising concerns about accessibility, which could be tied to the **ChatGPT app**.
- **Indirect Injection: Data Needs Sanitization**: Members voiced concerns over whether **OpenAI** has disclosed if deep research is vulnerable to **indirect prompt injection** from scraped pages, implying a need for data sanitization.
   - Another member was optimistic about an upcoming feature addressing this concern, looking forward to more information.
- **Markdown Manages URL Attention**: **ChatGPT** is more effective with links described in [markdown](https://markdown-guide.org/) rather than plain URLs, improving prompt hygiene.
   - Members found that using well-formatted structured data like JSON can help manage large blocks of information effectively.
- **DeepSeek's 'Unlimited' Has Usage Restrictions**: Reports highlight that high use of **DeepSeek** is categorized as abusive, sparking user concerns about the term *'unlimited'*, and raising questions about the transparency of **OpenAI's** policies.
   - The restrictions, seemingly applied inconsistently, prompted questions about the transparency of OpenAI's policies and user expectations.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor MCP Servers Spark Discussion**: Users on the channel discussed various **MCP servers**, including the **Perplexity MCP server**, detailing its setup and functionality within **Cursor** to improve coding assistance.
   - Some users shared their experiences integrating different models into their workflows, while others troubleshoot command prompts that returned errors, indicating the need for clearer documentation and support.
- **Agent Mode Praised for Debugging**: Users explored **Agent Mode** functionalities and its advantages over standard coding commands, particularly praising its debugging capabilities and direct communication with models like **Perplexity**.
   - The consensus was that integrating different **LLMs** could enhance the coding experience, especially with features allowing searching and real-time assistance.
- **MCP Server Installation Snafus Reported**: Several users encountered issues setting up **MCP servers**, specifically with command execution and server responses on different operating systems such as **Mac** and **Windows**.
   - Discussions involved troubleshooting command prompts that returned errors or failed to connect, pointing to the need for improved documentation and support.
- **Custom Cursor Rules Spark Interest**: Participants discussed the possibility of creating custom **cursor rules** to improve the implementation of specific features while using the **Perplexity MCP server**, with links to [Using Cursor with Convex](https://docs.convex.dev/ai/using-cursor).
   - Users emphasized that integrated cursor rules could streamline workflow and enhance the ability of the **AI** to respond to complex code-related queries.
- **Performance and Limitations Probed**: Discussions occurred regarding the performance of various models, including reports of service degradation and concerns about fast **API call limits** within **Cursor**.
   - Participants noted that **MCP servers**, if used correctly, could alleviate performance issues and provide better results than traditional web scraping methods.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Unique Tags Boost Lora Consistency**: Using unique tags in training data, such as specific names for objects or scenes, can significantly improve the consistency and narrative continuity of generated images in **Lora models**.
   - The method helps the model to better associate specific scenes with those names, as shown in [this example of Lora Training on BasedLabs](https://x.com/BasedLabsAI/status/1888313013276684711).
- **Optimal Flux Resolutions Found**: For generating images with **Flux**, optimal latent sizes are around **672x1024** or **1024x672**, while **1920x1088** provides a suitable quick **HD generation** size.
   - Generating images above **1MP** during initial passes may cause compositional issues.
- **Photoshop Gets ComfyUI Integration**: Users are exploring the integration of various plugins for **ComfyUI** with **Photoshop**, such as [Auto-Photoshop-StableDiffusion-Plugin](https://github.com/AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin) and [sd-ppp](https://github.com/zombieyang/sd-ppp).
   - These plugins enable the generation of stable diffusion images directly within Photoshop using a ComfyUI backend.
- **Stable Diffusion Hit GPU Snags**: Users reported troubleshooting **GPU errors** and slow performance issues across different **Stable Diffusion** UI paths, with lowering GPU settings being a common solution to resolve memory issues.
   - Using specific settings and maintaining aspect ratios were recommended to improve model performance and output quality, see [Stable Diffusion Knowledge Base (Setups, Basics, Guides and more)](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides).
- **AI-Generated Art Gets Copyright Shield?**: A recent case granted copyright protection to an AI-produced image due to sufficient human input, potentially setting a legal precedent for **AI-generated content** ownership, reported by [cnet.com](https://www.cnet.com/tech/services-and-software/this-company-got-a-copyright-for-an-image-made-entirely-with-ai-heres-how/).
   - The image, called *A Single Piece of American Cheese,* was created using Invoke's AI editing platform.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Mimics META's Moves**: Discussion highlights how **Nous Research** improves its AI models using advancements from larger companies like **META** and **DeepSeek**, while facing funding challenges as a smaller startup.
   - The focus is on creating affordable frontier AI models to maintain market competitiveness, similar to building on existing codebases.
- **Granite 3.1 Trains Multiple Objectives**: User plans to train **Granite 3.1's 3B model** to explore training strategies and custom RL loops with multiple objectives per epoch in a new setup.
   - This explores the potential of using multiple objectives within the novel training structure.
- **Zonos Clones High Fidelity Voices**: The release of **Zonos**, a high-fidelity TTS model featuring voice cloning, showcases strong performance against leading TTS providers.
   - The model's open-source license under **Apache 2.0**, as noted in [ZyphraAI's tweet](https://fxtwitter.com/ZyphraAI/status/1888996367923888341), promotes its integration into AI development.
- **LM Similarity Undermines AI Oversight**: Research has proposed a probabilistic metric for **language model similarity** based on model mistakes to enhance **AI oversight**, as detailed in a paper on [arxiv.org](https://arxiv.org/abs/2502.04313).
   - This suggests the use of **LLMs** as judges to favor similar models to facilitate weak-to-strong generalization with complementary knowledge; however, the trend is concerning as model mistakes are becoming harder to detect as AI Oversight becomes more important.
- **OVERTHINK slows reasoning models**: The **OVERTHINK** attack is causing models to slow down by as much as **46x** in inference by injecting decoy tasks, amplifying reasoning tokens without altering output, according to [Jaechul Roh's tweet](https://x.com/JaechulRoh/status/1887958947090587927).
   - The method uses complex tasks like **Markov Decision Processes** and **Sudoku** during untrusted contexts to manipulate inference processes, posing risks for models like **OpenAI's o1** and **o3-mini**.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurfers Request Profile Page Polish**: The **Codeium team** is soliciting user feedback for improvements to the Codeium profile page, with users encouraged to submit suggestions via [a provided form](https://windsurf.notion.site/194d73774c0080f0b05ee33699e907b9?pvs=105).
   - The enhancements aim to create a more useful and personalized experience, focusing on the stats and metrics that users find most valuable.
- **Jetbrain Extension Seen as Abandoned**: Users worry that the **Jetbrain extension** model availability lags behind **Windsurf**, with some speculating about a shift towards a **Cursor-centric** approach, causing frustrations over lost functionalities.
   - The announcement that a new passive in-text editor experience will be exclusive to **Windsurf**, leading to the deprecation of **Supercomplete** on the VSCode plugin, exacerbates these concerns.
- **Codeium Plagued by Payment Problems**: There's discussion around payment restrictions affecting **Russian users**, causing challenges in securing licenses due to regional limitations and company policies.
   - Users are urging Codeium for clearer communication regarding these restrictions, as well as an improved payment process.
- **Windsurfers Want Workflow Improvements**: Windsurf users reported issues with code proposals, diff displays, and automatic updates, along with the need for more consistent **tool calling** among AI models like **O3**, **Deepseek**, and **Claude**.
   - Users are also requesting better credit management, system issue notifications, improved design documents, debugging capabilities, and output consistency from AI models.
- **Credit Crunch Concerns Codeium Customers**: Users voiced concerns about the **credit system**, particularly around consumption during operations and the absence of refunds for unsuccessful attempts.
   - The frustration stems from spending credits on unsatisfactory outputs, prompting calls for more transparency in usage handling.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Exposes Reasoning Tokens**: Users can now see **reasoning tokens** on model activity pages alongside **prompt** and **completion tokens** for better transparency.
   - This enhancement aims to provide users with deeper insights into how models perform on the [OpenRouter platform](https://openrouter.ai/activity.).
- **Chat-thyme Simplifies Discord Bot Creation**: [Chat-thyme](https://github.com/chilir/chat-thyme) lets you set up Discord bots using any OpenAI-compatible LLM framework, offering easy **OpenRouter** integration.
   - It also integrates **Exa** for models supporting tool use, although reliability depends on the provider.
- **FindSMap Integrates Historical Maps Globally**: [FindSMap](http://findsmap.com) is a progressive web application connecting historical maps and archaeological institutes using **Open Street Maps** and **Leaflet.js**.
   - Built with **Claude** and **Open Router**, FindSMap showcases iterative development and dedication to the project.
- **DeepSeek R1 faces Timeouts**: Users reported significant **performance issues** with **DeepSeek R1**, experiencing timeouts during API requests, but the 'nitro' variant is integrated into the main model features, allowing users to sort by throughput
   - A new inference stack for DeepSeek R1 @togethercompute gets up to **110 t/s** on the **671B** parameter model ([tweet](https://x.com/vipulved/status/1888021545349742592)).
- **TypeScript SDK Eases LLM Calls**: A team is building a **TypeScript SDK** to interface with over **60 LLMs** using **OpenAI's format**, integrating **OpenRouter**.
   - The [GitHub project](https://github.com/lunary-ai/abso) aims to simplify calls to **100+ LLM Providers**, but feedback indicates it may be *rough around the edges*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek APIs Suffer Instability**: Users reported instability and unresponsiveness with **DeepSeek APIs**, especially when integrating them with **Aider**. One user had trouble getting outputs using DeepSeek with specific configurations.
   - Model comparisons for DeepSeek's R1 and V3 favored **Hyperbolic** and **OpenRouter** over other providers, with users noting specific configurations enhancing performance.
- **Aider Auto-Creates Files in Architect Mode**: Users are experiencing Aider auto-creating files without prompts in **Architect mode**, leading to confusion. A user shared a screenshot showing the unexpected behavior, suggesting potential configuration issues; see [issue #3153](https://github.com/Aider-AI/aider/issues/3153#issuecomment-2640194265).
   - This unexpected behavior is leading to confusion about the operation flow, and warrants more investigation into the config.
- **Aider Chat History Reaches Token Limit**: There are concerns that **Aider's chat history** is exceeding reasonable limits, with some users reporting it climbing to **25k tokens**.
   - The community discussed potential bugs and the effectiveness of prompt caching, and the overall effect on performance.
- **Copilot Proxy Unlocks GitHub Copilot Models**: The experimental [Copilot Proxy](https://github.com/lutzleonhardt/copilot-proxy) VS Code extension enables AI assistants access to **GitHub Copilot's language models**. A [YouTube video](https://youtu.be/i1I2CAPOXHM) details the extension's functionality.
   - One member sought ways to utilize the Copilot Proxy work, and another suggested using the [llmap repo](https://github.com/jbellis/llmap) with its `parse.py` script to extract file outlines.
- **Gemini Models Effective for PHP Tasks**: Users reported positive experiences with **Gemini models** like `gemini-1206-exp` for PHP tasks, with comparisons to other providers showing no significant differences in output.
   - Aider also introduced experimental support for tree-sitter-language-pack aiming to expand Aider's programming language capabilities. Users are encouraged to test this feature and provide feedback.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek R1 Goes Local**: Chinese GPU manufacturers like Moore Threads and Baidu's Kunlun are now supporting **DeepSeek's R1 LLM models** on local systems, increasing competition with NVIDIA.
   - This move signifies growing AI hardware capabilities in China, challenging NVIDIA's dominance in AI processing.
- **Anthropic Indexes Economic Impact**: Anthropic launched the **Economic Index**, including a paper analyzing millions of anonymized Claude conversations to assess AI's impact on the economy, as discussed in their [Tweet](https://x.com/AnthropicAI/status/1888954156422992108).
   - Initial findings reveal *material transportation* shows surprisingly low engagement compared to other sectors.
- **Replit Simplifies Mobile App Creation**: Replit introduced early access for **Native Mobile App support**, enabling users to create iOS and Android apps without coding, powered by Replit Assistant; [tweet here](https://x.com/amasad/status/1888727685825699874?s=46&t=PW8PiFwluc0tdmv2tOMdEg).
   - This launch marks a pivot towards more accessible app development, promising full agent support soon.
- **Deep Research Tool Sparks Debate**: Members discussed OpenAI's new **Deep Research** tool, highlighting its interactive approach by asking clarifying questions before research, which signals a move towards more proactive AI as shown on their [Deep Research page](https://openai.com/index/introducing-deep-research/).
   - Comparisons are emerging with tools like [Hugging Face's Deep Research](https://m-ric-open-deep-research.hf.space/) and other community-developed alternatives.
- **ELIZA Makes a Comeback?**: Members were introduced to the **ELIZA Operating System** ([ELIZA Operating System](https://www.elizaos.ai)) designed for AI agents, highlighting its foundational role in chatbot technology.
   - The conversation highlighted the historical significance of chatbots like **ELIZA** in the context of modern AI development.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Faces Ecosystem Hurdles**: Members debated Mojo's viability for web development, emphasizing the importance of a solid ecosystem and seamless integration with existing **Python libraries**.
   - The general consensus was that significant effort is required to build foundational tools before widespread adoption can occur, mentioning platforms like [Render](https://render.com) as a good example.
- **VariadicList Challenges Arise in Mojo**: A user reported issues initializing **VariadicList** in Mojo, specifically concerning dynamic element repetition using the `pop.variadic.create` operation, and posted a link to the [GitHub issue](https://github.com/modular/mojo/issues/3987)).
   - The issue highlights potential gaps in Mojo's current capabilities for handling variadic lists, with some members sharing their own **mojoproject.toml** files (such as [this one](https://github.com/BradLarson/max-cv/blob/main/mojoproject.toml#L21)) .
- **Domain Knowledge Drives Business**: Participants stressed that domain understanding is essential for launching a successful tech business, particularly the need for strong **networking knowledge**.
   - Many startups neglect this aspect, which leads to avoidable challenges and impedes growth. *'Understanding the domain is crucial for launching a business'*, one member stated.
- **Network Effects Influence Language Adoption**: The group discussed how **network effects** impact the adoption of languages like **Rust**, where a vibrant ecosystem fosters experimentation and growth.
   - While some tolerate rapid development 'slop', others advocate for maintaining high-quality standards to ensure long-term viability and prevent technical debt.
- **C++ Remains King in High-Performance**: The discussion highlighted **C++**'s continued dominance in performance-critical applications and its impact on new language adoption.
   - While **Mojo** has potential, its growth hinges on seamless integration with established languages and offering substantial performance advantages over current solutions.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **No Firebase/Firestore MCP Found**: A user looking for a **Firebase/Firestore MCP** was directed to a link indicating it might not exist, highlighting a need for such a tool.
   - This gap underscores opportunities for developing **MCP tools** tailored to specific database integrations.
- **MCP Command Path Misconfiguration**: Users encountered 'No Tools Found' errors while adding **MCP servers** via Cursor, suggesting path misconfigurations might be the cause.
   - Solutions involve verifying the correct command path and potentially resetting the application after updates, ensuring proper tool recognition.
- **MCP Performance Faces Python SDK Hurdles**: Users reported slow tool call responses when using **MCP** with **Claude Desktop**, attributing the issues to limitations within the Python SDK and ongoing bugs after a recent update ([python-sdk@bd74227](https://github.com/modelcontextprotocol/python-sdk/commit/bd742272ab9ef5576cbeff4045560fb2870ce53b)).
   - The feedback emphasizes a demand for enhanced error handling and overall performance improvements to facilitate smoother operation.
- **Smithery Installer Sparks Concerns**: While regarded as a leading **MCP installer**, concerns arose about **Smithery's** remote data handling and overhead, prompting a search for a more local alternative.
   - Users emphasized the need for **privacy and efficiency**, pushing for solutions that minimize remote data dependencies in MCP tools.
- **Claude Desktop Beta Still Buggy**: Beta testers experienced crashes with the **Claude Desktop app** while using their MCP servers, reflecting the current features' unreliability.
   - The consensus is that the app requires extensive feedback and substantial improvements before a stable release can be anticipated, as provided in the [Claude Desktop Quick Feedback](https://docs.google.com/forms/d/e/1FAIpQLScfF23aTWBmd6lNk-Pcv_AeM2BkgzN2V7XPKXLjFiEhvFmm-w/viewform) form.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **cuBLAS Shows Varied GPU Performance**: A user found **cuBLAS** performance inconsistent between a **1650ti** and **4090**, questioning if the build accommodates newer architectures.
   - Discussions also touched on how increasing the **L1 hit rate** might alleviate stalls related to load queuing.
- **Unsloth Turbocharges LLM Training**: **Unsloth** can speed up **LLM training by 30x**, enabling **Alpaca** training in just **3 hours** instead of **85**, according to their blog post [Introducing Unsloth](https://unsloth.ai/introducing).
   - They claim **60% less memory usage** without sacrificing accuracy, offering both open source and proprietary options.
- **Mistral Finetuning Gets 14x Faster**: The introduction of **QLoRA** support accelerates **Mistral 7B** finetuning by **14x** on a single **A100**, decreasing **peak VRAM usage by 70%**, as noted in their blog post [Unsloth update: Mistral support + more](https://unsloth.ai/blog/mistral-benchmark).
   - Additionally, **CodeLlama 34B** sees a **1.9x speedup**, with enhanced memory utilization preventing out-of-memory errors.
- **Explore iGPU Programming on Ryzen AI**: Members discussed how to leverage the **iGPU** in the **Ryzen AI CPU (Strix Point)** through **graphics frameworks** or potentially **HIP**.
   - These approaches could allow developers to tap into the processing power of integrated GPUs.
- **reasoning-gym gets Matrix Manipulation**: The **reasoning-gym** saw new PRs merged, including [Matrix Manipulation](https://github.com/open-thought/reasoning-gym/pull/100) and [Count Bits](https://github.com/open-thought/reasoning-gym/pull/101), expanding the dataset offerings.
   - Members considered how to best **benchmark** the gym environment to see how RL training impacts generalization, and considered using **OpenRouter** for inference compute.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Plus Joins Google One, Student Discounts Arrive**: [NotebookLM Plus](https://blog.google/technology/google-labs/notebooklm-new-features-december-2024/) is now part of **Google One AI Premium**, offering higher usage limits; U.S. students over 18 get a **50% discount** on the plan, which is **$9.99/month**.
   - NotebookLM Plus increases notebook capacity by **5x**, source limit per notebook by **6x**, and audio overviews by **7x**.
- **Users grapple with NotebookLM's Source Generation Hiccups**: Users report issues with **NotebookLM** failing to generate notes from uploaded sources like **.txt** and **.pdf** files; the system displays 'New Note: Generating' indefinitely.
   - Workarounds include directly pasting text and directing users to official Google support links to understand inherent free and paid version limits.
- **NotebookLM Plus Boosts Chat and Sharing Tools**: **NotebookLM Plus** now features advanced chat customization, sharing capabilities, and provides comprehensive usage analytics.
   - Notebook sharing requires **Gmail** to be enabled, presenting challenges for users with SSO from Azure.
- **AI Bridges Clarity Gap in Medical Discussions**: A member shared how **AI** helps clarify **medical jargon** related to their breast cancer diagnosis, summarizing dense articles and surgeon appointments.
   - They emphasized how AI has been *a comforting aid during their treatment* by challenging the AI for clarifications.
- **Users Build Versatile Bots With NotebookLM**: A user launched the **Versatile Bot Project**, providing [prompt documents](https://github.com/shun0t/versatile_bot_project) to transform NotebookLM into different types of chatbots through specialized prompts.
   - The user said that both prompts have been tested and aimed to create a customizable chatbot experience.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Skip Transcoders leap ahead of Sparse Autoencoders**: **Skip transcoders** demonstrate a **Pareto improvement** over **SAEs**, providing enhanced **interpretability** and fidelity for researchers, and can be used with flags `--transcode` and `--skip_connection` in the [sparsify](https://github.com/EleutherAI/sparsify) library.
   - In contrast to **SAEs**, transcoders better approximate input-output relationships, bolstering the approach to interpretability, according to the team which published [their paper](https://arxiv.org/abs/2501.18823) on arxiv.org.
- **Partial Rewriting Faces Obstacles**: The team encountered *lackluster results* in their research on partially rewriting transformers, as they trained a skip transcoder on the sixth layer of **Pythia 160M**.
   - Despite initial setbacks, the team remains optimistic about refining their methods and has published [a paper](https://arxiv.org/abs/2501.18838) detailing the approach.
- **GPU Retrofitting for AI: Proceed with Caution**: Concerns about repurposing older **1070ti** mining rigs for AI highlighted issues with outdated architecture and bandwidth limitations, possibly limiting training.
   - While these GPUs could serve adequately in inference tasks, members cautioned against expecting efficient training outcomes for contemporary AI models.
- **Chess-Based LLM Evaluation Gambit**: EleutherAI is creating a task to evaluate LLMs using a database of **4M+ chess tactics**, which could uniquely enhance LLM performance, eventually playing chess, by leveraging **reinforcement learning**.
   - The team is determining whether to do **MCQ style** versus free-form generation, hoping for models to show their reasoning through **<think>** tags.
- **Pythia's Puzzling Checkpoint Pattern**: Discussion clarified that **Pythia saves checkpoints every 1,000 steps**, contrary to claims of **10K steps**, to enable deeper analysis using **log(tokens)** for interpretations.
   - There was some consideration about whether smaller linear step sizes and switching over earlier would improve efficiency, weighed against concerns of **wallclock overhead** for saving checkpoints.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Logits vs Probabilities sparks debate**: Members debated the benefits of training models in **log space** compared to **absolute space**, emphasizing that log space can capture a wider range of values and can lead to more similarities in distant points.
   - One member pointed out that using log space affects accuracy based on the use case.
- **Sparse Autoencoders Receive Skepticism**: A member voiced skepticism about **Sparse Autoencoders** (SAEs) being overhyped, expressing disappointment in their interpretability and citing inconsistencies across random seeds, see [this paper](https://arxiv.org/abs/2501.16615).
   - The discussion referenced recent papers critiquing SAEs and exploring new methods for model interpretation, as well as skip transcoders outperforming SAE's as shared [on twitter](https://x.com/norabelrose/status/1887972442104316302).
- **Guardrails Fail Bioweapons Discovery**: A drug discovery algorithm, intended to minimize toxicity, reportedly switched to *maximizing* toxicity, leading to the discovery of **40,000** potential bioweapons in just **6 hours**.
   - The incident raised alarms about the effectiveness of current guardrails against broad knowledge synthesis and the risk of overlooking harmful compounds due to narrow focus.
- **PlanExe AI Project launches on Github**: A member introduced **PlanExe**, a structured AI planner built with **LlamaIndex** and **OpenRouter**, which can generate structured plans like SWOT analyses without extensive web searching, available on [GitHub](https://github.com/neoneye/PlanExe).
   - The creator expressed uncertainty about the accuracy of the outputs but also provided a [link to PlanExe-web](https://neoneye.github.io/PlanExe-web/).
- **LLMs Struggle With Token Counting**: Members noted that LLMs struggle with counting tokens in their context, suggesting that the difficulty extends beyond tokenization to a fundamental inability to count.
   - It was simply stated by a member that *LLMs can't count at all*.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Gemini Flash Accelerates Document Understanding**: **LlamaParse** now supports **Gemini 2.0 Flash**, achieving **GPT-4o+ performance** levels for document processing at a lower cost, setting the stage for enhanced workflows leveraging **VLMs and LLMs**.
   - A tutorial by @composiohq demonstrated building a **YouTube research agent** with **Gemini Flash 2.0**, streamlining video searches and Gmail draft creation, reinforcing LlamaIndex's utility in simplifying video research workflows.
- **CrossPoster App Arrives for AI-Enhanced Social Media**: The **CrossPoster** app launched, enabling cross-posting to **Twitter**, **LinkedIn**, and **BlueSky** using AI to optimize social media engagement.
   - The app intelligently identifies individuals and their accounts, streamlining the management of a social presence across platforms.
- **OpenAI LLM Faces Timeout Troubles**: Members found that the timeout for **OpenAI LLM** options is being overridden by the retry decorator, leading to inconsistencies, despite higher timeout settings.
   - One member shared that even after submitting a bug fix, Deepseek returns a 200 OK response after 60 seconds but with an empty body, exacerbating the issue.
- **Hand-off Frustrations in LlamaIndex**: Users voiced concerns about the `can_handoff_to` feature in LlamaIndex, particularly when agents transfer control without a response from the receiving agent, leading to dropped requests.
   - Suggested solutions included enabling debug logging and using LlamaIndex's callback handler for more effective troubleshooting.
- **Metadata Must-Haves for AzureAI Search**: A user questioned the hardcoded customization of filterable metadata fields in **AzureAI Search**, specifically noting 'author' and 'director'.
   - It was clarified that **Azure** requires these metadata fields to be defined upfront, emphasizing the significance of well-defined and useful document fields, and the need to be aware of the current limitations of the feature.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Trust Yourself During Job Hunt**: Members on the Cohere Discord emphasized *self-belief* during job applications, encouraging others to trust in themselves 'regardless of what they say'.
   - They added that *everyone is just as uncertain*, pushing for persistence in the face of challenges and highlighting the *lack of hiring opportunities* for engineering internships.
- **Networking Boosts Exposure**: Members suggest that *networking* is crucial, regardless of one’s location, suggesting participation in events to boost exposure, while also recommending engaging in open-source projects to connect with others in the field.
   - One user mentioned attending *conferences and competitions* relevant to their engineering field, even highlighting their participation in the **Canadian engineering competition**.
- **LibreChat API calls hitting v1 instead of v2**: A member highlighted that they can only access the **Cohere API** through `https://api.cohere.ai/v1` using **LibreChat's** Custom Endpoint, confirming the **Cohere API** works via **curl**.
   - It was pointed out that **LibreChat** is currently calling the old API version (v1) and needs an update to the `/v2` endpoint, though the URL [https://api.cohere.com/v1](https://api.cohere.com/v1) mirrors the functionality of `https://api.cohere.ai/v1`.
- **Cohere Community lays down the Rules**: Members discussed the **Cohere Community** rules, emphasizing respect and appropriate conduct within the server, while drafting introduction messages for newcomers, highlighting interests in AI and local initiatives like 'buy Canadian'.
   - The discussion later shifted to the scalability of **Cohere's API** and how accessible their staff is for collaboration, while one member encouraged a Socratic dialogue about vapes.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Yu Su's Language Agents Lecture Livestreamed**: Today at **4:00pm PST**, the 3rd lecture featuring **Yu Su** on *Memory, Reasoning, and Planning of Language Agents* was live streamed [here](https://www.youtube.com/live/zvI4UN2_i-w), arguing contemporary AI agents use **language as a vehicle for reasoning**.
   - Yu Su is a **Distinguished Assistant Professor** at the Ohio State University and co-directs the NLP group with significant contributions including **Mind2Web, SeeAct, HippoRAG, LLM-Planner, and MMMU** garnering recognition like the **Best Student Paper Award** at CVPR 2024 and **Outstanding Paper Award** at ACL 2023.
- **MOOC Late Enrollment and Curriculum Details Awaited**: Users can enroll in the **LLM Agents MOOC** that started in January, and staff promised to release more curriculum details soon, addressing concerns about project framework and publication limitations.
   - Participants asked about the specifics of assignments and projects outside of quizzes, to which staff mentioned detailed information would be released shortly, encouraging users to remain patient while awaiting clear guidelines on project requirements and grading policies.
- **Certificate Concerns in Berkeley MOOC**: Several users reported not receiving their certificates while their peers have, prompting a focus on missing completed **certificate declaration forms** as a required step.
   - Course staff reiterated that completion of this form is necessary for certificate issuance and needs to be submitted individually, and suggestions included creating an automated agent to streamline the certificate process and address common queries.
- **DPO Explained and Compared to SFT**: A member explained how **Supervised Fine Tuning (SFT)** uses only positive examples while **Direct Preference Optimization (DPO)** incorporates negative responses, highlighting the penalties for bad responses in DPO.
   - *Bad responses*, often well-structured, trigger an increase in their probability during SFT due to the absence of a reward model.
- **Lecture 2 Study Session Prompted Time Zone Concerns**: A member announced a study session on **Lecture 2: Learning to Reason with LLMs**, inviting others to join via a provided link, preparing to discuss **GRPO from DeepSeek-R1** as part of the study materials.
   - One participant expressed concern about the study session's timing, noting that it fell at **3:00 AM UK time**, highlighting potential scheduling conflicts for international members.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Exploring Artificial Data Generation Methods**: A member is diving into **artificial data generation** and is looking for tools to turn unstructured data like PDFs and Excel files into training samples for LLMs, citing a [YouTube video](https://youtu.be/iogrvDu5K0k?si=U9fi5C-0UvytTmBO) on the topic.
   - However, there was a recognition of challenges in training LLMs with synthetic data, noting that question generation may not provide necessary comparative insights that requires comprehensive data across multiple document sources.
- **Kolo Simplifies Fine-Tuning**: A member is developing **Kolo**, a tool designed to simplify model fine-tuning, but it currently lacks data creation capabilities.
   - The developer plans to add a training data generation feature in the future.
- **PR #2257 Under Review**: A member requested a review for [PR #2257](https://github.com/pytorch/torchtune/pull/2257), stating it passes local tests but needs more feedback.
   - Reviewers lauded the changes but raised UX concerns regarding quantization and recommended documentation improvements.
- **GRPO's Feature Philosophy**: The team debated whether to simplify **GRPO** by removing functionalities, balancing usability with cleaner code.
   - Opinions leaned toward removing unneeded code, with some acknowledging the potential need for features like activation checkpointing; see [Grpo loss by kashif](https://github.com/linkedin/Liger-Kernel/pull/553/files#diff-1534093f54f1b158be2da2b159e45561361e2479a7112e082232d3f21adc6a45).
- **Torchtune's Checkpointing Mechanics Detailed**: A member shared how resume functionality updates checkpoint paths and depends on the `resume_from_checkpoint` flag, as seen in the [Checkpointing in torchtune documentation](https://pytorch.org/torchtune/main/deep_dives/checkpointer.html#resuming-from-checkpoint-full-finetuning).
   - Discussion covered the implications of unusual workflows in loading initial weights.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All Lacks Model Selection Menu**: Users are concerned about the absence of a functional model selection menu with search options in **GPT4All**, even after 36 releases.
   - A member suggested contributing code to enhance **GPT4All** due to its open-source nature.
- **AI Agents Embrace Databases for Long-Term Memory**: Members explored using **AI agents** with databases for long-term memory and suggested improving **LLMs' temporal awareness** through functions.
   - The conversation speculated that **2025** could be a pivotal year for advancements in agentic AI.
- **GPT4All Sidelines Image Analysis**: It was clarified that **GPT4All** does not currently support image analysis, with suggestions to use other platforms for such tasks.
   - Recommendations included tools like **booruDatasetTagmanager** and **joycaption** for image-related projects.
- **Perfecting PDF Embedding Methods**: Members discussed strategies for embedding and summarizing long documents like PDFs into usable formats for **GPT4All**.
   - Proper handling of downloads to remove irrelevant content before embedding was emphasized.
- **Qwen2.5 and Phi4 Win Popularity Contest**: Members recommended **Qwen2.5** and **Phi4** for their efficiency compared to models like **Mistral**.
   - The user-friendliness of models integrated with the app was underscored, with offers of assistance for those unfamiliar with **Hugging Face**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad's Mobile Misadventures**: Testing reveals **WebGPU** failing on iPhone 15 due to caching issues, while M1 Pro users report success on Safari and Chrome with **tinychat** demos.
   - The community is calling for enhanced testing to improve compatibility, especially with **WASM** loading on mobile devices.
- **Tinygrad's Remote Roots Revealed**: Clarification emerged that **tinygrad** is a **fully remote company**, dismissing rumors of being based in San Diego due to inaccurate Twitter information.
   - The correction prompted inquiries about **Ampere Altra** processor support and backend acceleration capabilities.
- **Company Meeting Gears Up for Action**: Meeting #57 is scheduled, featuring discussions on **company updates**, **CI speed**, **tensor cores**, and potential **bounties** for **WebGPU** and **tinychat** enhancements.
   - The goal is to boost internal operational speeds and address community interests in ongoing projects.
- **FP16's Fate in ML Frameworks**: A debate sparked about why most ML frameworks don't exclusively use **fp16**, revealing potential disadvantages and performance limitations.
   - George responded with a suggestion to review discord rules, sparking further commentary on research quality prior to inquiries.
- **PR Precision and Quantization Quirks**: Discussions centered on a pull request (PR) implementing a script, emphasizing the need for additional features and testing, especially with **Hugging Face models**.
   - The community stressed the importance of clean PR structure for easy reviews while acknowledging existing **numerical inaccuracies** in quantized models as a challenge.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Trains BERT to Classify Articles**: A member transitioned from **GPT-3.5** and **GPT-4** to training a **BERT** model for article classification using **DSPy**.
   - The optimized prompt now extracts a dozen fields from each article, processed in batches every 24 hours using **Miprov2** with **o3-mini** as a teacher, and **Mistral Small 3** as a student, and resulted in a **50% discount**.
- **Multi-Agent Systems Boost Performance with MASS**: LLMs operating as multiple agents show great promise in solving complex tasks due to effective collaboration strategies highlighted in the [MASS framework](https://arxiv.org/abs/2502.02533).
   - The analysis emphasizes the importance of **prompts** and **topologies** in multi-agent system design.
- **Factorio as AI Agent System Engineering Sandbox**: Static benchmarks fall short in evaluating necessary skills for dynamic system engineering, so agents trained via automation-oriented sandbox games like [Factorio](https://arxiv.org/abs/2502.01492) is proposed.
   - This fosters the development of reasoning and long-horizon planning capabilities essential for managing complex engineering challenges.
- **Deep Research Abstractions**: A member inquired about plans to introduce abstractions that simplify tasks akin to **deep research**.
   - *Are you guys planning to introduce abstractions?* the member asked, highlighting their curiosity about potential upcoming features.
- **DSPy Client Error Debacle**: A member reported encountering the error `AttributeError: module 'dspy' has no attribute 'HFClientVLLM'` while using **dspy**.
   - They later noted that this feature was **deprecated** in **dspy 2.6**, which resolved their confusion.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Custom RAFT templates for Llama?**: A member inquired whether their own templates, similar to **RAFT's**, could be used for generating synthetic datasets with **Llama**.
   - This inquiry raises questions about the flexibility of **Llama's** dataset requirements and customization options.
- **Compatibility issues with HF Datasets**: A member voiced concerns about potential compatibility issues with **HF datasets** due to differing function properties.
   - The member suggested converting complex objects to strings to simplify dataset loading and usage.
- **JSON lines Formatting Clarified**: A member clarified that there are no issues with the **JSON** files, noting that HF expects JSON lines formatted files.
   - This clarification underscores the importance of adhering to the expected file format for successful dataset loading in **HF**.
- **README Update Proposed**: A member offered to create a pull request (**PR**) to update the **README** with a new helper function.
   - The suggestion was well-received, indicating a collaborative approach to improving user experience and documentation.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1337513705071443979)** (1052 messages🔥🔥🔥): 

> `Unsloth AI Progress, GRPO Challenges, ASCII Art Generation, Reward Function Validations, Multi-GPU Support` 


- **Unsloth AI Gains Popularity**: Unsloth has become the #1 trending repository on GitHub, marking significant progress within a year of its establishment.
   - The community has expressed appreciation for the tools and resources that Unsloth provides, with enthusiasm for its future developments.
- **GRPO and Reward Function Issues**: Users have reported challenges with GRPO effectively evaluating non-deterministic outputs, particularly in creative tasks such as RPG role-playing.
   - Discussions emphasized the importance of the reward function and suggestions for debugging to enhance output quality.
- **Exploring ASCII Art Generation**: An interest in fine-tuning models to generate ASCII art based on descriptions emerged, but concerns about models' limitations and coherence were raised.
   - Participants encouraged the exploration of existing models for potential inspiration and successful examples in generating ASCII art.
- **Validation Challenges in AI Training**: The conversation highlighted difficulties in verifying the output of models trained with RL, particularly when no fixed outputs exist to compare against.
   - It was suggested that leveraging known hallucinations might provide a starting point for generating better datasets.
- **Multi-GPU Support in Unsloth**: The Unsloth community is curious about the future implementation of multi-GPU support, currently in beta for select trusted members.
   - Users expressed interest in updates and availability for broader access in the future.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@alexhe.amd/deploy-deepseek-r1-in-one-gpu-amd-instinct-mi300x-7a9abeb85f78">Deploy Deepseek-R1 in one GPU — AMD Instinct™ MI300X</a>: Yes!</li><li><a href="https://arxiv.org/abs/2411.10440">LLaVA-CoT: Let Vision Language Models Reason Step-by-Step</a>: Large language models have demonstrated substantial advancements in reasoning capabilities, particularly through inference-time scaling, as illustrated by models such as OpenAI&#39;s o1. However, curr...</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://arxiv.org/abs/2501.17161">SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training</a>: Supervised fine-tuning (SFT) and reinforcement learning (RL) are widely used post-training techniques for foundation models. However, their roles in enhancing model generalization capabilities remain ...</li><li><a href="https://x.com/BlinkDL_AI/status/1888637497443504524">Tweet from BlinkDL (@BlinkDL_AI)</a>: https://huggingface.co/BlinkDL/temp-latest-training-models/blob/main/rwkv-x070-2b9-world-v3-preview-20250210-ctx4k.pth</li><li><a href="https://unsloth.ai/blog/reintroducing">Re-introducing Unsloth</a>: In celebration of us being the #1 Trending GitHub repo of the day, we reflect on our journey and contributions to the open-source community.</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/1ik3nkr/p_grpo_fits_in_8gb_vram_deepseek_r1s_zeros_recipe/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://unsloth.ai/newsletter">Unsloth Newsletter</a>: Join our newsletter and waitlist for everything Unsloth!</li><li><a href="https://docs.unsloth.ai/basics/errors">Errors | Unsloth Documentation</a>: To fix any errors with your setup, see below:</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit Quantization</a>: Unsloth&#x27;s Dynamic 4-bit Quants selectively avoids quantizing certain parameters. This greatly increases accuracy while maintaining similar VRAM use to BnB 4bit.</li><li><a href="https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md">ktransformers/doc/en/DeepseekR1_V3_tutorial.md at main · kvcache-ai/ktransformers</a>: A Flexible Framework for Experiencing Cutting-edge LLM Inference Optimizations - kvcache-ai/ktransformers</li><li><a href="https://huggingface.co/Mistral-AI-Game-Jam">Mistral-AI-Game-Jam (Mistral AI Game Jam)</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1889000210371932398">Tweet from Unsloth AI (@UnslothAI)</a>: Unsloth is the #1 trending repo on GitHub! 🦥It’s been an incredible journey and we couldn’t have done it without you! To celebrate, we’re taking a look back at how it all started and how we got here:...</li><li><a href="https://github.com/fzyzcjy/unsloth-zoo/commit/d2372ca5fc3bcb3ea41b6473c3b7d36de5265c55">Update peft_utils.py · fzyzcjy/unsloth-zoo@d2372ca</a>: no description found</li><li><a href="https://gitingest.com/">Gitingest</a>: Replace 'hub' with 'ingest' in any GitHub URL for a prompt-friendly text.</li><li><a href="https://docs.unsloth.ai/basics/er">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/1485#issuecomm">[FIXED] `Qwen2VL` finetuning broken · Issue #1485 · unslothai/unsloth</a>: Installing latest version of unsloth: !pip uninstall unsloth -y &amp;&amp; pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git breaks the Qwen2 7B Vision Colab ...</li><li><a href="https://github.com/TruffleClock/nano-r1">GitHub - TruffleClock/nano-r1</a>: Contribute to TruffleClock/nano-r1 development by creating an account on GitHub.</li><li><a href="https://github.com/EvolvingLMMs-Lab/open-r1-multimodal">GitHub - EvolvingLMMs-Lab/open-r1-multimodal: A fork to add multimodal model training to open-r1</a>: A fork to add multimodal model training to open-r1 - EvolvingLMMs-Lab/open-r1-multimodal</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://github.com/Zyphra/Zonos">GitHub - Zyphra/Zonos</a>: Contribute to Zyphra/Zonos development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/1485#issuecomment-2628795809">[FIXED] `Qwen2VL` finetuning broken · Issue #1485 · unslothai/unsloth</a>: Installing latest version of unsloth: !pip uninstall unsloth -y &amp;&amp; pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git breaks the Qwen2 7B Vision Colab ...</li><li><a href="https://github.com/unslothai/unsloth/issues/1613">Qwen2.5-VL-3B 4Bit train, &#39;requires_grad_&#39; error · Issue #1613 · unslothai/unsloth</a>: Hi! I am trying to sft the qwen2.5vl(unsloth/Qwen2.5-VL-3B-Instruct) model on google colab using the colab file https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7...</li><li><a href="https://www.nature.com/articles/s41467-024-55628-6">Large Language Models lack essential metacognition for reliable medical reasoning - Nature Communications</a>: Large Language Models demonstrate expert-level accuracy in medical exams, supporting their potential inclusion in healthcare settings. Here, authors reveal that their metacognitive abilities are under...</li><li><a href="https://www.fabfilter.com/">FabFilter - Quality Audio Plug-Ins for Mixing, Mastering and Recording - VST VST3 AU CLAP AAX AudioSuite</a>: no description found</li><li><a href="https://build.nvidia.com/nvidia/digital-humans-for-customer-service/blueprintcard">Build a Digital Human Blueprint by NVIDIA | NVIDIA NIM</a>: Create intelligent, interactive avatars for customer service across industries</li><li><a href="https://github.com/unslothai/unsloth/pull/1289#issuecomment-2646547748">Added Support for Apple Silicon by shashikanth-a · Pull Request #1289 · unslothai/unsloth</a>: UnoptimizedNo gguf support yet.Build Triton and bitsandbytes from sourcecmake -DCOMPUTE_BACKEND=mps -S . for bitsandbytes buildingpip install unsloth-zoo==2024.11.4pip install xformers==0.0.25</li><li><a href="https://github.com/trending">Build software better, together</a>: GitHub is where people build software. More than 150 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.max_grad_norm">Trainer</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1337754746227789834)** (10 messages🔥): 

> `Reasoning LLM using REINFORCE, Deepseek collaboration, ReMax RL Training for LLMs, Model merging challenges` 


- **Concerns Over REINFORCE Approach**: A member shared a [link to a document](https://charm-octagon-74d.notion.site/Reasoning-LLM-using-REINFORCE-194e4301cb9980e7b73dd8c2f0fdc6a0) regarding **Reasoning LLM using REINFORCE**, sparking debate on its originality.
   - Others questioned whether it offered anything novel, with one noting that an identical implementation already exists in **Unsloth**.
- **Discussion on Deepseek Availability**: A member inquired about the collaboration on **Deepseek**, uncertain of its presence on Hugging Face.
   - Responses indicated that parts of its implementation might already be integrated or available in existing projects.
- **ReMax Framework Gains Attention**: A tweet highlighted the launch of **ReMax**, a framework for **RL Training for LLMs** with features like higher throughput compared to PPO.
   - Members discussed its [GitHub code](https://github.com/liziniu/verl/tree/feature/add-remax-support/examples/remax_trainer), emphasizing its stability in training.
- **Skepticism on Model Merging**: A user expressed interest in understanding the challenges of merging several effective models into a single mixture of experts (MoE).
   - This request prompted discussion about potential pitfalls and limitations associated with model merging strategies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ziniuli/status/1888576228619370525?s=46">Tweet from Ziniu Li (@ZiniuLi)</a>: 🚀 Efficient RL Training for LLMs: ReMax, built on the Verl distributed framework, is now available! 🛠️🔑 Key Features:- Higher Throughput than PPO- Stable Training with theoretical guarantees on var...</li><li><a href="https://charm-octagon-74d.notion.site/Reasoning-LLM-using-REINFORCE-194e4301cb9980e7b73dd8c2f0fdc6a0">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1337514752464916636)** (440 messages🔥🔥🔥): 

> `Fine-tuning models, Handling OOM errors, Using quantized models, Reward functions in training, Model evaluation` 


- **Challenges in Fine-tuning and Evaluation**: Users reported issues with fine-tuning Qwen2 VL models, noting that their adjustments have had little effect on output accuracy despite decreasing training losses.
   - Several users expressed frustration finding effective approaches, with some believing their models were not improving despite applying LoRA techniques.
- **Dealing with OOM Errors**: Multiple members faced out-of-memory (OOM) errors when attempting to load and fine-tune models, emphasizing the need to manage model complexity and resource allocation.
   - Suggestions included cleaning checkpoints and ensuring the configuration aligns with available hardware, particularly when using larger models.
- **Quantized Models and Performance**: Discussion around using 4-bit quantized models highlighted potential performance benefits but also confusion regarding their implementation, especially post-quantization outputs.
   - Users noted that while dynamic quantization can enhance efficiency, it can sometimes result in unexpected output alterations, querying the balance between model size and result consistency.
- **Understanding Reward Functions in Training**: Participants emphasized the significance of tailoring reward functions to encourage desired outputs, particularly in tasks with varying input complexities.
   - There was interest in how reward mechanisms could refine model responses, especially when datasets include nuanced examples such as scientific data extraction.
- **Building Models with Multi-GPU Support**: A user inquired about training models across multiple GPUs within Unsloth, highlighting the limitations and lack of direct multi-GPU support currently.
   - Suggestions pointed towards potential future enhancements but noted the current need for single GPU training setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=IqM-T1RTzY6C,">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/vision-fine-tuning">Vision Fine-tuning | Unsloth Documentation</a>: Details on vision/multimodal fine-tuning with Unsloth</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1tiQrc6LVOxdRDWsM5WMuLrYhPkUVt617?usp=sharing#scrollTo=ZDXE1V-MNtAG">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">Finetuning from Last Checkpoint | Unsloth Documentation</a>: Checkpointing allows you to save your finetuning progress so you can pause it and then continue.</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">Datasets 101 | Unsloth Documentation</a>: Learn all the essentials of creating a dataset for fine-tuning!</li><li><a href="https://huggingface.co/yukiarimo/yuna-ai-v3-full/tree/main">yukiarimo/yuna-ai-v3-full at main</a>: no description found</li><li><a href="https://github.com/simples">simples - Overview</a>: GitHub is where simples builds software.</li><li><a href="https://github.com/unslothai/unsloth/issues/1613">Qwen2.5-VL-3B 4Bit train, &#39;requires_grad_&#39; error · Issue #1613 · unslothai/unsloth</a>: Hi! I am trying to sft the qwen2.5vl(unsloth/Qwen2.5-VL-3B-Instruct) model on google colab using the colab file https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B">unsloth/DeepSeek-R1-Distill-Qwen-32B · Hugging Face</a>: no description found</li><li><a href="https://github.com/ollama/ollama/blob/main/README.md#customize-a-prompt">ollama/README.md at main · ollama/ollama</a>: Get up and running with Llama 3.3, DeepSeek-R1, Phi-4, Gemma 2, and other large language models. - ollama/ollama</li><li><a href="https://github.com/unslothai/unsloth/issues/1208#issuecomment-2444633537">Error `KeyError: &#39;layers.0.mlp.down_proj.weight&#39;` when running Merged 4-bit Mistral Nemo in vLLM · Issue #1208 · unslothai/unsloth</a>: I am attempting to fine-tune Mistral Nemo, save it as 4-bit merged, and run it in vLLM. I do not encounter any errors during the training process. However, when I attempt to serve the model with vL...</li><li><a href="https://www.youtube.com/watch?v=JJWvYQdOVOY"> - YouTube</a>: no description found</li><li><a href="https://github.com/simplescaling/s1/blob/main/eval/generate.py#L57-L72">s1/eval/generate.py at main · simplescaling/s1</a>: s1: Simple test-time scaling. Contribute to simplescaling/s1 development by creating an account on GitHub.</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py">finetune-Qwen2-VL/finetune.py at main · zhangfaen/finetune-Qwen2-VL</a>: Contribute to zhangfaen/finetune-Qwen2-VL development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/trl.git">GitHub - huggingface/trl: Train transformer language models with reinforcement learning.</a>: Train transformer language models with reinforcement learning. - huggingface/trl</li><li><a href="https://github.com">GitHub · Build and ship software on a single, collaborative platform</a>: Join the world&#39;s most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.</li><li><a href="https://colab.research.google.com/drive/1tiQrc6LVO">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1337886405224960124)** (2 messages): 

> `Llama 3.2 model breakdown, Spark Engine v1 launch, No-code AI tools, AI model integration` 


- **Detailed Breakdown of Llama 3.2 Model**: A user has reassembled the **Llama 3.2 1B model** into its weights, providing insights into its architecture which can be explored on [GitHub](https://github.com/SoumilB7/Llama3.2_1B_pytorch_barebones).
   - *Share any suggestions for improvements* on this breakdown are welcome to enhance its effectiveness.
- **Spark Engine v1 Launch Party**: The latest version of **Spark Engine v1** has been released after over a year of public beta, boasting over **80 AI models** capable of generating various content types, including *text, music, and videos*.
   - The user expressed a desire to potentially integrate more infrastructure like **Unsloth** into the Spark Engine platform, fostering further advancements in the no-code AI realm.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sparkengine.ai/">Spark Engine - The AI Sandbox</a>: Turn ideas into AI-powered products, no coding experience required</li><li><a href="https://github.com/SoumilB7/Llama3.2_1B_pytorch_barebones">GitHub - SoumilB7/Llama3.2_1B_pytorch_barebones: Pytorch implementation of Llama 3.2 1B architecture barebones + nuggets of wisdom</a>: Pytorch implementation of Llama 3.2 1B architecture barebones + nuggets of wisdom - SoumilB7/Llama3.2_1B_pytorch_barebones
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1337594689674547294)** (42 messages🔥): 

> `Dataset Curation Importance, Meta Reasoning and Lora Fine-Tuning, Optimizing Inference Methods, Long Output Format Challenges, Integrating Updated Language Specs` 


- **Dataset Curation is Key to Model Success**: It's been emphasized that **80%** of any model's performance relies on careful **dataset curation**, highlighting its critical role in training.
   - *One member noted,* 'There is no such thing as redundant research - you learn from every paper,' which reflects a commitment to continuous learning.
- **Exploring Meta Reasoning with Lora**: A member is experimenting with Lora settings to develop a metacognitive first-person reasoning format, aiming to improve reasoning capabilities.
   - They plan to extensively test their model settings and ensure it balances between retaining base knowledge and enhancing reasoning.
- **Optimizing Inference Methods for Improved Performance**: Inference tests on Unsloth indicated it was both faster and more accurate compared to other methods, using bfloat16 across multiple GPUs.
   - Members discussed optimizations that could improve performance on lower-spec hardware while maintaining model accuracy.
- **Challenges with Long Output Formats in Fine-Tuning**: Concerns were raised about the potential loss of learning in long output formats that share common structures, which may hinder the training of specific tasks.
   - One member addressed that token loss would allow the model to focus on dynamic parts, which would aid in learning desired tasks despite the lengthy prompts.
- **Integrating Updated Language Specifications into Models**: A user sought advice on parsing updated language specifications for Janet to enhance Phi4's capabilities, given its knowledge cutoff in 2021.
   - The challenge lies in distilling current resources to improve the model's solutions in light of new updates and features in the language.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/a/qoZHqis">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://arxiv.org/abs/2502.04128">Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis</a>: Recent advances in text-based large language models (LLMs), particularly in the GPT series and the o1 model, have demonstrated the effectiveness of scaling both training-time and inference-time comput...</li><li><a href="https://arxiv.org/html/2411.00856v1">AI in Investment Analysis: LLMs for Equity Stock Ratings</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1337522439743340706)** (252 messages🔥🔥): 

> `AI Agents Course, Model Optimization, Mental Health Chatbot Recommendations, Data Privacy and Security, AI Presentation Generation Solutions` 


- **Discussion on AI Agents Course**: Several users confirmed their participation in the AI Agents course starting shortly, with some expressing excitement for the content.
   - Inquiries were made about course materials and quizzes to prepare for the upcoming sessions.
- **Exploring Smaller Models for Efficiency**: Users discussed the importance of choosing smaller models to optimize performance and reduce hardware requirements while maintaining quality.
   - It was suggested to run comparisons of multiple models and employ techniques like batching requests for better efficiency.
- **Recommendations for Mental Health Chatbots**: Various models from Hugging Face, including Aloe and OpenBioLLM, were recommended for mental health applications, highlighting their potential effectiveness.
   - Conversations addressed the use of local inference for maintaining data privacy when using AI tools in sensitive fields like health tech.
- **Data Privacy and Cloud Concerns**: A discussion emerged about the implications of sending protected patient data to cloud services and the need for local processing requirements.
   - Various strategies, such as anonymizing data or using smaller models, were suggested as ways to mitigate privacy concerns while performing AI tasks.
- **AI for Presentation and UI Generation**: A user requested solutions for automatically generating PowerPoint presentations and transforming designs from Figma into code.
   - They sought tools that could streamline these processes, leveraging AI for efficiency and reducing repetitive manual work.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://livebench.ai/">LiveBench</a>: no description found</li><li><a href="https://livebench.ai">LiveBench</a>: no description found</li><li><a href="https://sparkengine.ai">Spark Engine - The AI Sandbox</a>: Turn ideas into AI-powered products, no coding experience required</li><li><a href="https://scalingintelligence.stanford.edu/pubs/">Publications</a>: no description found</li><li><a href="https://cs229s.stanford.edu/fall2024/calendar/">Calendar</a>: Listing of course modules and topics.</li><li><a href="https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed.html">torch.cuda.manual_seed &mdash; PyTorch 2.6 documentation</a>: no description found</li><li><a href="https://tenor.com/view/remygag-remy-gag-barf-rat-gif-978058593911536854">Remygag Remy GIF - RemyGag Remy Gag - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/thrishala/mental_health_chatbot?library=transformers">thrishala/mental_health_chatbot · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/DAMO-NLP-SG/videollama3-678cdda9281a0e32fe79af15">VideoLLaMA3 - a DAMO-NLP-SG Collection</a>: no description found</li><li><a href="https://huggingface.co/spaces/Tonic/Math">Math - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://huggingface.co/spaces/Tonic/Nemo-Mistral-Minitron">Nemotron-Mini - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/spaces-gpus#community-gpu-grants">Using GPU Spaces</a>: no description found</li><li><a href="https://tenor.com/view/samurai-japan-windy-sword-birds-gif-17444000">Samurai Japan GIF - Samurai Japan Windy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/facebook/bart-large-cnn">facebook/bart-large-cnn · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ijab77/train_your_own_reasoning_model_80_less_vram_grpo/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://pytorch.org/get-started/locally/">Start Locally</a>: Start Locally</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/discussions/20">stabilityai/stable-diffusion-xl-refiner-1.0 · Only receiving black images?</a>: no description found</li><li><a href="https://forums.developer.nvidia.com/t/fp16-support-on-gtx-1060-and-1080/53256">FP16 support on gtx 1060 and 1080</a>: Hello everyone,  I am a newbee with TensorRT.  I am trying to use TensorRT on my dev computer equipped with a GTX 1060.  When optimizing my caffe net with my c++ program (designed from the samples pro...</li><li><a href="https://huggingface.co/datasets/Tonic/MiniF2F">Tonic/MiniF2F · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/bobpopboom/mentalchat2">Mentalchat2 - a Hugging Face Space by bobpopboom</a>: no description found</li><li><a href="https://huggingface.co/spaces/bobpopboom/testing">Testing - a Hugging Face Space by bobpopboom</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=g74Cq9Ip2ik">Master AI image generation - ComfyUI full tutorial 2024</a>: ComfyUI complete installation &amp; tutorial. The ultimate image generator. Text to image, image to image, faceswap, controlnet, upscaling, external plugins, &amp; m...</li><li><a href="https://deepai.org/">DeepAI</a>: Artificially intelligent tools for naturally creative humans.</li><li><a href="https://huggingface.co/datasets/mzbac/function-calling-llama-3-format-v1.1">mzbac/function-calling-llama-3-format-v1.1 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://youtu.be/V_xro1bcAuA?si=M7M7L9oZi5b6Nl5o"> - YouTube</a>: no description found</li><li><a href="https://tenor.com/view/the-game-awards-matan-matan-evenoff-bill-clinton-gif-27233593">The Game Awards Matan GIF - The Game Awards Matan Matan Evenoff - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/huggingchat/chat-ui/discussions/372#67a47cb63dd995045efdad4f">huggingchat/chat-ui · [MODELS] Discussion</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1338121555611353212)** (4 messages): 

> `Hugging Face in healthcare, Adaptive tuning of KL loss in VAE training, Classifying scientific abstracts, Gradio front end for AI agent, Model checkpointing` 


- **Exploring Hugging Face in Healthcare**: A member is diving into how to use **Hugging Face** in the healthcare sector, indicating an interest in practical applications.
   - They expressed a desire to know if anyone else is pursuing this same topic.
- **Promising Results with Adaptive KL Loss Tuning**: Another member shared insights on using various approaches for adaptively tuning **KL loss** during **VAE** training, reporting interesting results.
   - They noted the success of reducing the weight to 0 to stave off collapse while training.
- **Classifying Scientific Publications**: A user is creating a model to classify abstracts from scientific publications into two distinct categories using their **private data**.
   - They emphasized their focus on classifiers in their project.
- **Building a Gradio Front End for AI Monitoring**: One member is learning how to create a **Gradio front end** for an AI agent that monitors code execution, sharing a specific model name.
   - They attached multiple images related to their work, illustrating their progress in building this application.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1337612449498791997)** (6 messages): 

> `Reasoning Language Models, Writing-in-the-Margins bounty, Physical Perceptrons, Markdrop Python Package` 


- **Exploring Reasoning Language Models**: A YouTube video titled ["From Large Language Models to Reasoning Language Models - Three Eras in The Age of Computation"](https://www.youtube.com/watch?v=NFwZi94S8qc) was shared, diving into the evolution of LLMs and their impact on computation.
   - The discussion emphasized the transformative journey through various computational lenses.
- **$5k Bounty for Writing-in-the-Margins**: A member highlighted that there is a [$5,000 bounty](https://github.com/vllm-project/vllm/issues/9807) available for implementing the Writing-in-the-Margins inference pattern in vllm.
   - This feature aims to enhance results for long context window retrieval, with a detailed motivation provided.
- **The Quest for Physical Perceptrons**: A member questioned if anyone had seen a physical [Perceptron](https://www.youtube.com/watch?v=l-9ALe3U-Fg), referencing a YouTube video for context.
   - This inquiry opened up discussions about the historical construction and significance of perceptrons in AI.
- **Markdrop's PDF Conversion Power**: A new Python package called [Markdrop](https://github.com/shoryasethia/markdrop) was introduced for converting PDFs to markdown, including image and table extraction features.
   - It offers advanced functionalities such as AI-powered content analysis, automatic image extraction, and interactive HTML output.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=NFwZi94S8qc">From Large Language Models to Reasoning Language Models - Three Eras in The Age of Computation.</a>: In this talk, we explore the fascinating evolution of Large Language Models (LLMs) and their transformative journey through the lenses of computation and opt...</li><li><a href="https://github.com/shoryasethia/markdrop">GitHub - shoryasethia/markdrop: A Python package for converting PDFs to markdown while extracting images and tables, generate descriptive text descriptions for extracted tables/images using several LLM clients. And many more functionalities. Markdrop is available on PyPI.</a>: A Python package for converting PDFs to markdown while extracting images and tables, generate descriptive text descriptions for extracted tables/images using several LLM clients. And many more func...</li><li><a href="https://pypi.org/project/markdrop/">markdrop</a>: A comprehensive PDF processing toolkit that converts PDFs to markdown with advanced AI-powered features for image and table analysis. Supports local files and URLs, preserves document structure, extra...</li><li><a href="https://github.com/vllm-project/vllm/issues/9807">[Feature]: Integrate Writing in the Margins inference pattern ($5,000 Bounty) · Issue #9807 · vllm-project/vllm</a>: 🚀 The feature, motivation and pitch Writer has introduced &quot;Writing in the Margins&quot; algorithm (WiM) that boosts results for long context window retrieval. The task is composed from &quot;con...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1337538032466661396)** (16 messages🔥): 

> `Kokoro TTS integration, Dataset Tools update, Spark Engine launch, Markdrop PDF tool, go-attention implementation` 


- **Kokoro TTS now open-sourced with C# library**: A member announced the release of a C# library for **Kokoro TTS**, enabling plug & play integration on .NET platforms, available on [GitHub](https://github.com/Lyrcaxis/KokoroSharp). This library supports fast local TTS inference and works across multiple platforms.
   - The library promises a **multilingual** experience with all voices packaged in a convenient format.
- **Dataset Tools gets an update for EXIF and AI Metadata**: The Dataset organizer and **EXIF Viewer** received updates, enhancing its capabilities to view advanced EXIF data and supporting formats like GGUF and JPEG, shared on [GitHub](https://github.com/Ktiseos-Nyx/Dataset-Tools).
   - The developer utilized AI tools to assist in the project, enhancing its features while collaborating with others for code optimization.
- **Spark Engine v1 officially launched**: The Spark Engine v1 was released after a year-long public beta, providing **over 80 models** for various AI tasks available at [sparkengine.ai](https://sparkengine.ai/).
   - The platform offers free credits daily and integrates with Hugging Face, making a robust no-code environment for users to experiment with AI capabilities.
- **Markdrop provides advanced PDF to Markdown features**: A new Python package called **Markdrop** was introduced, designed for converting PDFs to Markdown with features like image extraction and AI-powered descriptions, accessible on [GitHub](https://github.com/shoryasethia/markdrop).
   - In just a month, it has achieved over **7,000 installs**, showcasing its popularity among users looking for document manipulation tools.
- **Innovative go-attention implementation for transformers**: A member shared their project, **go-attention**, which showcases the first full attention mechanism and transformer built in pure Go, highlighting its unique capabilities on [GitHub](https://github.com/takara-ai/go-attention).
   - The project invites others to check out examples and explore the potential of serverless implementations in Go programming.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sparkengine.ai/">Spark Engine - The AI Sandbox</a>: Turn ideas into AI-powered products, no coding experience required</li><li><a href="https://huggingface.co/spaces/elismasilva/mixture-of-diffusers-sdxl-tiling">Mixture Of Diffusers SDXL Tiling - a Hugging Face Space by elismasilva</a>: no description found</li><li><a href="https://huggingface.co/spaces/jeremyadd/mini_datathon">Mini Datathon - a Hugging Face Space by jeremyadd</a>: no description found</li><li><a href="https://huggingface.co/blog/Duskfallcrew/design-101">Design 101: A Historical and Theoretical Exploration of Graphic Arts and Design in the Age of AI</a>: no description found</li><li><a href="https://github.com/shoryasethia/markdrop">GitHub - shoryasethia/markdrop: A Python package for converting PDFs to markdown while extracting images and tables, generate descriptive text descriptions for extracted tables/images using several LLM clients. And many more functionalities. Markdrop is available on PyPI.</a>: A Python package for converting PDFs to markdown while extracting images and tables, generate descriptive text descriptions for extracted tables/images using several LLM clients. And many more func...</li><li><a href="https://pypi.org/project/markdrop">no title found</a>: no description found</li><li><a href="https://github.com/Lyrcaxis/KokoroSharp">GitHub - Lyrcaxis/KokoroSharp: Fast local TTS inference engine with ONNX runtime. Multi-speaker, multi-platform and multilingual.  Integrate on your .NET projects using a plug-and-play NuGet package, complete with all voices.</a>: Fast local TTS inference engine with ONNX runtime. Multi-speaker, multi-platform and multilingual.  Integrate on your .NET projects using a plug-and-play NuGet package, complete with all voices. - ...</li><li><a href="https://github.com/takara-ai/go-attention">GitHub - takara-ai/go-attention: A full attention mechanism and transformer in pure go.</a>: A full attention mechanism and transformer in pure go. - takara-ai/go-attention</li><li><a href="https://huggingface.co/datasets/Tonic/Climate-Guard-Toxic-Agent">Tonic/Climate-Guard-Toxic-Agent · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/Ktiseos-Nyx/Dataset-Tools">GitHub - Ktiseos-Nyx/Dataset-Tools: A Simple Viewer for EXIF and AI Metadata</a>: A Simple Viewer for EXIF and AI Metadata. Contribute to Ktiseos-Nyx/Dataset-Tools development by creating an account on GitHub.</li><li><a href="https://github.com/duskfallcrew/Beetlejuice_Summoning">GitHub - duskfallcrew/Beetlejuice_Summoning: Literally just summons a youtube video after you say his name 3x spoken unbroken, and makes sure you enter one of the &quot;WHOLE BEING DEAD THING&quot; lyrics. It&#39;s untested, but i was using GPT to be a nerd.</a>: Literally just summons a youtube video after you say his name 3x spoken unbroken, and makes sure you enter one of the &amp;quot;WHOLE BEING DEAD THING&amp;quot; lyrics. It&amp;#39;s untested, but i wa...</li><li><a href="https://tenor.com/view/baby-cute-go-cheering-rage-gif-16641949">Baby Cute GIF - Baby Cute Go - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1338186001553555526)** (15 messages🔥): 

> `AI Reading Group Schedule, Role Management, Women in AI & Robotics` 


- **February AI Reading Group Session Reminder**: The next session of the **AI Reading Group** from **Women in AI & Robotics** is scheduled for this Thursday at **12pm EST / 5pm GMT**. Participants can join the live stream in the reading group voice channel [here](https://discord.com/events/879548962464493619/1331369988560523447).
   - It was noted that these sessions occur approximately **once a month**, with opportunities for paper authors to present.
- **Role Removal Assistance**: A member inquired about removing a role, and was informed that it may have been removed or is undergoing an **overhaul**. Another member promptly assisted by stating, *'I've removed the role for you.'*
   - The original member expressed gratitude with a quick, *'tysm.'*
- **Interest in Participation**: Members expressed interest in attending the **AI Reading Group** if time allows, highlighting engagement with deeper topics like **deep tech engineering**. One member communicated enthusiasm about potentially participating in future sessions.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1337518194276958289)** (6 messages): 

> `Computer Vision for Manufacturing, Screenshot Analysis Models, Roboflow Maestro for Multimodal Models, BLIP Fine-tuning Script, AutoML for Image Classification` 


- **Computer Vision Enhances Manufacturing Inspections**: One member is experimenting with using **computer vision** in a manufacturing setting to inspect products and analyze both visual features and production logs.
   - This approach aims to ensure quality by effectively merging visual and textual data.
- **Struggles with Browser Screenshot Interpretation**: A user expressed frustration over the difficulty of finding a **computer vision model** that accurately interprets visual components on browser screenshots.
   - Despite recent impressive results in the field, they noted that existing models, particularly those utilizing *GPT-4V*, haven't provided the detail needed for effective integration into their workflow.
- **Exploring Roboflow Maestro for Model Fine-tuning**: A member shared a link to [Roboflow Maestro](https://github.com/roboflow/maestro) as a resource for streamlining the fine-tuning process of multimodal models like **PaliGemma 2** and **Florence-2**.
   - This user is considering trying it out, but currently relies on a **BLIP** fine-tuning script.
- **Interest in AutoML for Image Classification**: A member inquired if anyone has tried an **AutoML model** for image classification tasks.
   - This highlights ongoing interest in simplifying the process of model selection and training for specific computer vision needs.



**Link mentioned**: <a href="https://github.com/roboflow/maestro">GitHub - roboflow/maestro: streamline the fine-tuning process for multimodal models: PaliGemma 2, Florence-2, and Qwen2.5-VL</a>: streamline the fine-tuning process for multimodal models: PaliGemma 2, Florence-2, and Qwen2.5-VL - roboflow/maestro

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1338202903512350806)** (12 messages🔥): 

> `Context Length Problem in LLMs, Sentiment Classification with roBERTa, Evaluation of Embedding Models, Product Quantization Techniques, Coarse Quantization in PQ` 


- **Seeking Solutions for Context Length in LLMs**: A member expressed frustration over finding a practical solution to the **context length problem** in LLMs, emphasizing concerns about accuracy dropping as context length increases.
   - They are looking for **high-quality** approaches that don't sacrifice accuracy for extended context.
- **Sentiment Analysis with Twitter roBERTa**: One user shared their method of classifying sentiments using the [Twitter-RoBERTa model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment), stating that they utilize the Hugging Face classification pipeline.
   - They noted that the model's outputs sometimes yield unusually high confidence scores across all sentiment categories.
- **Evaluation Techniques for Embedding Models**: A member asked for community feedback on a paper proposing a unified evaluation method independent of downstream tasks, focused on embedding models.
   - The paper aims to correlate theoretical foundations with practical performance metrics to ensure better evaluation standards.
- **Insights on Product Quantization**: Discussion emerged around the **Product Quantization (PQ)** technique, especially its implications for using subtleties in **word embeddings**, and concerns about information loss during quantization.
   - One user inquired about the trade-off between compression benefits and potential meaning alteration in embeddings.
- **Understanding Coarse Quantization**: A user sought clarification on **coarse quantization** in the context of Product Quantization and reported difficulties finding sufficient materials on the topic.
   - They highlighted frustrations with existing AI tools failing to provide adequate answers regarding this concept.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openreview.net/forum?id=VqFz7iTGcl">When is an Embedding Model  More Promising than Another?</a>: Embedders play a central role in machine learning, projecting any object into numerical representations that can, in turn, be leveraged to perform various downstream tasks. The evaluation of...</li><li><a href="https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment">cardiffnlp/twitter-roberta-base-sentiment · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/pipelines">Pipelines</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1337534758346817626)** (19 messages🔥): 

> `Course Registration Issues, Live Q&A Session Announcement, Certification Notification Process, GitHub Pull Request for Course Content, YouTube Course Introduction` 


- **Registered but no updates received**: Several members expressed concerns about not receiving updates after registering for the course, including *jyotip2217* and *pierre2452*.
   - This raises questions about the communication process for participants who have signed up for the course.
- **Live Q&A scheduled for February 12th**: A member provided a link to a [YouTube video](https://www.youtube.com/watch?v=PopqUt3MGyQ) detailing the course's introduction and the upcoming live Q&A session on February 12th at 5 PM.
   - Participants were informed that this session will cover course logistics and provide a platform for questions.
- **Clarification needed on certification notification**: A participant asked if there is a specific notification process required for certification after enrolling using their Hugging Face account.
   - The uncertainty points to a potential gap in instructional clarity regarding course participation and certification expectations.
- **GitHub collaboration for course content**: Member *burtenshaw* shared a [GitHub Pull Request](https://github.com/huggingface/course/pull/777) aimed at migrating content from the smol course to the NLP course, including interactive quizzes.
   - This effort seeks to enhance the course by integrating more engaging materials and is open for collaboration.
- **YouTube video introduction to the course**: The shared YouTube video titled 'Welcome To The Agents Course' introduces the course structure and scope, serving as a resource for new participants.
   - This video aims to clarify upcoming course milestones, helping individuals navigate the initial phases effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/agents-course/unit0/introduction">Welcome to the 🤗 AI Agents Course - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course/unit1/introduction">Introduction to Agents - Hugging Face Agents Course</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=PopqUt3MGyQ">Welcome To The Agents Course! Introduction to the Course and Q&amp;A</a>: In this first live stream of the Agents Course, we will explain how the course will work (scope, units, challenges and more) and answer your questions.Don&#39;t ...</li><li><a href="https://github.com/huggingface/course/pull/777">[CHAPTER] New chapter on supervised fine tuning based on smol course by burtenshaw · Pull Request #777 · huggingface/course</a>: This is a draft PR for discussion.It would be cool to reuse the smol course chapter on SFT in the HF NLP course.Here I&amp;#39;ve just copied across the content, but here the next step I would propos....
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1337785854990614529)** (704 messages🔥🔥🔥): 

> `Course Introduction, Networking among participants, Python knowledge requirements, AI Agents discussions, International participation` 


- **Course Introduction and Expectations**: Participants are expressing excitement for the AI Agents course and discussing its launch, with many inquiring about course access and requirements.
   - People are eager to start learning and collaborating, with mentions of the course's focus on practical knowledge.
- **Networking Among Participants**: Users are introducing themselves along with their locations, creating a sense of community among participants from various countries.
   - Many express interest in connecting based on shared experiences in AI and related fields.
- **Python Knowledge Requirements**: Some participants are querying about the level of Python knowledge required for the course, indicating varying backgrounds in programming.
   - There is concern among users with basic Python skills, seeking reassurance about their ability to keep pace.
- **International Participation**: The channel features a diverse group of participants from numerous countries, including India, the U.S., France, and Brazil.
   - Participants are excited to learn together, appreciating the global nature of the course.
- **Technical Access Issues**: Some users report difficulties with account verification and accessing course channels on Discord after signing up.
   - There's ongoing discussion about troubleshooting these issues, with users sharing experiences regarding verification.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/021destiny">Tweet from undefined</a>: no description found</li><li><a href="https://www.kaggle.com/whitepaper-agents">Agents</a>: Authors: Julia Wiesinger, Patrick Marlow and Vladimir Vuskovic</li><li><a href="https://learn.deeplearning.ai/courses/ai-python-for-beginners/lesson/1/introduction">AI Python for Beginners: Basics of AI Python Coding - DeepLearning.AI</a>: Learn Python programming with AI assistance. Gain skills writing, testing, and debugging code efficiently, and create real-world AI applications.</li><li><a href="https://tenor.com/view/seattle-space-gif-18175495">Seattle Space GIF - Seattle Space - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lilianweng.github.io/posts/2023-06-23-agent/">LLM Powered Autonomous Agents</a>: Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The p...</li><li><a href="https://tenor.com/view/tijuana-gif-21081556">Tijuana GIF - Tijuana - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://amaykataria.com">Amay Kataria 3.0</a>: no description found</li><li><a href="https://tenor.com/view/hello-hi-hy-hey-gif-8520159980767013609">Hello Hi GIF - Hello Hi Hy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/napoleon-dynamite-wave-bye-gif-15387504">Napoleon Dynamite Wave GIF - Napoleon Dynamite Wave Bye - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/%C4%B1taly-italya-italiana-roma-ferrar%C4%B1-gif-13413841744894640022">ıtaly Italya GIF - Italy Italya Italiana - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huyenchip.com/2025/01/07/agents.html">Agents</a>: Intelligent agents are considered by many to be the ultimate goal of AI. The classic book by Stuart Russell and Peter Norvig, Artificial Intelligence: A Modern Approach (Prentice Hall, 1995), defines ...</li><li><a href="https://tenor.com/view/greetings-chat-chrissss-gridman-dante-devil-may-cry-dante-from-devil-may-cry-gif-22479319">Greetings Chat Chrissss Gridman GIF - Greetings Chat Chrissss Gridman Dante - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/agents-course">agents-course (Hugging Face Agents Course)</a>: no description found</li><li><a href="https://github.com/huggingface/agents-course">GitHub - huggingface/agents-course: This repository contains the Hugging Face Agents Course.</a>: This repository contains the Hugging Face Agents Course.  - GitHub - huggingface/agents-course: This repository contains the Hugging Face Agents Course.</li><li><a href="https://github.com/mohamedsheded/Agentic-design-patterns">GitHub - mohamedsheded/Agentic-design-patterns: A repo for implementing and understanding design patterns of agentic workflows</a>: A repo for implementing and understanding design patterns of agentic workflows - mohamedsheded/Agentic-design-patterns</li><li><a href="https://x.com/horosin_.">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1337602857037332511)** (7 messages): 

> `Reasoning Datasets, Model Training, Distillations, Learning Math` 


- **Exploring Reasoning Datasets**: A member directed others to check out various reasoning datasets available [here](https://huggingface.co/collections/open-r1/reasoning-datasets), particularly highlighting the **Bespoke-Stratos-17k** dataset.
   - Another member expressed gratitude, noting that this information was **most helpful**.
- **Attempting R1 Style Reasoning**: One user mentioned experimenting with teaching a model **R1 style reasoning** to aid in learning math, indicating a visual demonstration was attached.
   - The focus seems to be on simplifying the reasoning process, as discussed among members.
- **Discussion on Result Quality**: A member suggested removing the **<reasoning>** component entirely to assess differences in result quality.
   - This sparked a lighthearted conversation about the implications of such a change.
- **Patience Required for Training**: A member highlighted the lengthy training time, stating that training any model on a **4060ti** takes about **6 hours**, raising doubts about the efficacy of reasoning addition.
   - Despite the challenges, some **progress in learning math** was noted, underscoring a commitment to the process.



**Link mentioned**: <a href="https://huggingface.co/collections/open-r1/reasoning-datasets-67980cac6e816a0eda98c678">🧠 Reasoning datasets - a open-r1 Collection</a>: no description found

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1337513596808200374)** (596 messages🔥🔥🔥): 

> `Qwen Models, LM Studio Functionality, Embedding Models, Comparing AI Models, Using APIs` 


- **Qwen 2.5 vs Llama 8B**: Users discussed the performance differences between Qwen 2.5 and Llama 8B, with Qwen generally providing faster responses due to its optimization.
   - It was suggested that Qwen 2.5 is a better option if users have the necessary hardware to run larger models like 32B.
- **Troubleshooting Model Loading Issues**: Users reported various problems loading models into LM Studio, with suggestions to detail system specifications and provide screenshots for better assistance.
   - Errors like 'NO LM Runtime found for model format' indicate potential hardware limitations, emphasizing the importance of matching model size to system capabilities.
- **Utilizing Local Servers**: Queries were raised about accessing LM Studio models via local server, requiring connections to front-end applications for effective usage.
   - Suggestions included using compatible APIs as LM Studio does not feature a built-in web UI, highlighting the need for external integrations.
- **Model Configuration and Performance**: Discussion centered around adjusting settings like temperature and batch size in LM Studio to optimize performance based on available RAM and VRAM.
   - Users were advised that configuration tuning is crucial for achieving desired results from AI models, particularly for intensive applications.
- **AI for Coding and Projects**: Inquiries about using Qwen models for programming tasks led to insights about various models like Mistral and alternatives for effective coding assistance.
   - The conversation emphasized that while powerful models exist, starting with smaller, manageable ones might provide a better learning experience for beginners.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://qwenlm.github.io/blog/">Blog</a>: Qwen</li><li><a href="https://tenor.com/view/trump-thug-life-gif-11298887">Trump Thug Life GIF - Trump Thug Life - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lmstudio.ai/docs/basics/import-model">Import Models | LM Studio Docs</a>: Use model files you&#x27;ve downloaded outside of LM Studio</li><li><a href="https://lmstudio.ai/docs/basics/download-model#changing-the-models-directory">Download an LLM | LM Studio Docs</a>: Discover and download supported LLMs in LM Studio</li><li><a href="https://tenor.com/view/rpx_syria-mic-drop-gif-19149907">Rpx_syria Mic Drop GIF - Rpx_syria Mic Drop - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/closedagi/gpt5-r3-claude4-magnum-dong-slerp-raceplay-dpo-4.5bpw-1.1T-exl2-rpcal">closedagi/gpt5-r3-claude4-magnum-dong-slerp-raceplay-dpo-4.5bpw-1.1T-exl2-rpcal · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/stanley-hudson-the-office-annoyed-gif-20544770">Stanley Hudson GIF - Stanley Hudson The - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/everythings-under-control-happily-the-situation-is-under-control-there-is-control-over-everything-gif-26314021">Everythings Under Control Happily GIF - Everythings Under Control Happily The Situation Is Under Control - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/ishan-marikar/lm-studio-ollama-bridge">GitHub - ishan-marikar/lm-studio-ollama-bridge: lm-studio-ollama-bridge is a standalone Go-based utility that synchronizes your local Ollama models with external applications such as LM Studio. Inspired by the original matts-shell-scripts/syncmodels project</a>: lm-studio-ollama-bridge is a standalone Go-based utility that synchronizes your local Ollama models with external applications such as LM Studio. Inspired by the original matts-shell-scripts/syncmo...</li><li><a href="https://lmstudio.ai/docs/api/endpoints/rest#post-apiv0embeddings">LM Studio REST API (beta) | LM Studio Docs</a>: The REST API includes enhanced stats such as Token / Second and Time To First Token (TTFT), as well as rich information about models such as loaded vs unloaded, max context, quantization, and more.</li><li><a href="https://lmstudio.ai/docs/api/endpoints/openai#endpoints-overview">OpenAI Compatibility API | LM Studio Docs</a>: Send requests to Chat Completions (text and images), Completions, and Embeddings endpoints</li><li><a href="https://lmstudio.ai/docs/api/endpoints/rest">LM Studio REST API (beta) | LM Studio Docs</a>: The REST API includes enhanced stats such as Token / Second and Time To First Token (TTFT), as well as rich information about models such as loaded vs unloaded, max context, quantization, and more.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1337525071702659113)** (149 messages🔥🔥): 

> `GPU Overclocking, M4 Ultra vs M2 Ultra, AMD vs NVIDIA Performance, LM Studio with Intel Macs, PCI-E Riser Cables for Extra GPUs` 


- **Overclocking GPU Memory Discussion**: Members debated whether overclocking GPU memory increases inference speed, with one noting that memory overclocking might yield marginal benefits without significant gains.
   - *Mistral* and its limitations were also mentioned, highlighting the importance of model fit for optimal GPU performance.
- **Comparing M4 Ultra and M2 Ultra**: A discussion emerged regarding the value of waiting for the M4 Ultra compared to purchasing the M2 Ultra to run models more efficiently.
   - Concerns were shared about the model performance on M2 Ultra amidst rising costs associated with maintaining subscriptions to existing services.
- **AMD vs NVIDIA Performance Metrics**: Members compared the performance of the AMD 7900 XTX with the NVIDIA 4090, noting that benchmarks could vary based on the software optimizations available.
   - Some members pointed out that results might differ depending on whether the software supports ROCm or CUDA.
- **LM Studio's Compatibility with Intel Macs**: Users confirmed that LM Studio does not support Intel Macs unless using Boot Camp, which allows the installation of Windows.
   - While some alternatives like Open-webui are available, questions about model performance and GPU usage on Intel Macs were raised.
- **Using PCI-E Riser Cables for Additional GPUs**: A user wondered about the performance implications of using PCI-E riser cables to install additional GPUs, specifically discussing the potential compatibility with A5000 cards.
   - Meanwhile, a suggestion was made to repurpose old cases as GPU holders for better cooling and space management.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://openrouter.ai/cognitivecomputations/dolphin-mixtral-8x22b">Dolphin 2.9.2 Mixtral 8x22B 🐬 - API, Providers, Stats</a>: Dolphin 2.9 is designed for instruction following, conversational, and coding. Run Dolphin 2.9.2 Mixtral 8x22B 🐬 with API</li><li><a href="https://support.apple.com/en-us/102622">Install Windows 10 on your Mac with Boot Camp Assistant - Apple Support</a>: Learn how to install Windows 10 on your Mac with Boot Camp.</li><li><a href="https://www.youtube.com/watch?v=wKZHoGlllu4"> - YouTube</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://lmstudio.ai/docs/api/headless">Run LM Studio as a service (headless) | LM Studio Docs</a>: GUI-less operation of LM Studio: run in the background, start on machine login, and load models on demand</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/)** (1 messages): 

OpenAI: https://youtu.be/kIhb5pEo_j0
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1337571389221244969)** (705 messages🔥🔥🔥): 

> `AI Model Performance, Gemini vs ChatGPT, DeepSeek Restrictions, User Feedback Mechanism, Emerging AI Technologies` 


- **Concerns Over AI Model Reliability**: Users discussed the changes in AI model performance, particularly expressing concerns about models being 'lobotomized' after updates, leading to reduced output quality and consistency.
   - There's a general sentiment that many updates make previously capable models perform worse, creating distrust among users.
- **Gemini Gains Popularity with Large Context Windows**: Gemini’s capability to handle 1-2 million tokens has made it popular among users, particularly compared to ChatGPT’s limitations of 32k and 128k tokens.
   - Users appreciate Gemini’s flexible features, which enhance its usability for complex tasks and projects.
- **DeepSeek's Usage Restrictions**: There were discussions about DeepSeek's usage limitations, with reports of high use being categorized as abusive, prompting user concerns about the term 'unlimited'.
   - The restrictions, seemingly applied inconsistently, raised questions about the transparency of OpenAI's policies and user expectations.
- **Feedback Mechanisms in AI**: Users inquired about how feedback is processed within ChatGPT, prompting discussions about whether the feedback leads to meaningful improvements for individual contexts.
   - Concerns were expressed regarding the lack of transparency related to feedback implementation and model updates.
- **Social AI and Ethical Considerations**: The conversation touched on the potential of socially-oriented AIs that leverage community data to counter the influence of wealthy individuals and companies.
   - Participants debated the implications of utilizing AI trained on shadowy realms like the dark web and the ethics surrounding such technologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://clonerobotics.com/android">Android – Clone</a>: no description found</li><li><a href="https://community.openai.com/t/the-pro-tier-is-not-near-unlimited/1053132">The Pro tier is NOT &#39;near unlimited&#39;</a>: Every evening I am getting the following restriction applied to my account (and then removed by the help team after waiting for ages).  “Unusual activity detected with your o1 usage. We’ve temporarily...</li><li><a href="https://synaptiks.ai/p/from-base-models-to-reasoning-models">From Base Models to Reasoning Models</a>: Understand how modern LLM are trained and particularly new reasoning models like Deepseek-R1</li><li><a href="https://www.tagesschau.de/inland/gesellschaft/deepseek-datenschutz-100.html">Datenschützer wollen chinesische KI-Anwendung DeepSeek prüfen</a>: DeepSeek verblüfft und verunsichert die Tech-Welt. Der neue Chat-Bot funktioniert ähnlich wie sein Konkurrent ChatGPT - soll aber günstiger entwickelt worden sein. Datenschützer wollen die KI prüfen.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1337549761502384218)** (23 messages🔥): 

> `GPT-4 performance concerns, ChatGPT connection issues, Using GPT for code research, Emoji usage in responses, Children's storybook creation` 


- **GPT-4 performance feeling weak**: Members expressed concerns that **GPT-4** feels less capable now compared to initial excitement, noting it requires better prompting to yield good results.
   - *It's not about weakness,* said one member, pointing out that earlier models created a perception of inferiority in complex tasks.
- **ChatGPT experiencing connection errors**: Several users reported ongoing **connection errors** while using ChatGPT, raising concerns about accessibility.
   - One user highlighted that these issues could be tied specifically to the ChatGPT app rather than general model usage.
- **Code research with GPT**: A user inquired about the feasibility of using **GPT** for detailed code research, particularly for less-trained programming languages.
   - Another shared a positive experience with **SwiftUI documentation**, stating it effectively contributed to completing their project.
- **Concerns over emoji use in 4o**: Users discussed the influx of **emojis** in responses from GPT-4o, questioning whether it was intended to prevent misuse by other models.
   - One member criticized it as a result of a bad update, calling it annoying and unhelpful.
- **Using GPT for children's storybooks**: A member shared their experiences of using **GPT** for creating children's storybooks, prompting interest from others.
   - This conversation indicates a growing interest in leveraging GPT capabilities for creative storytelling.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1337558087061733468)** (11 messages🔥): 

> `Indirect Prompt Injection Vulnerability, Managing URLs in Prompts, Improving ChatGPT Responses, Effective Prompt Hygiene, Attention Management Techniques` 


- **Concerns Over Indirect Prompt Injection**: A member questioned whether OpenAI has disclosed if deep research is vulnerable to **indirect prompt injection** from scraped pages, implying a need for data sanitization.
   - Another member was optimistic about an upcoming feature relating to this concern, expressing eagerness for more information.
- **Markdown URLs Get Better Attention**: It was observed that ChatGPT is more effective with links described in [markdown](https://markdown-guide.org/) rather than just plain URLs, as they enhance prompt hygiene.
   - Members agreed that using well-formatted structured data like JSON can help manage large blocks of information effectively.
- **Desire for Concise Responses from ChatGPT**: A member expressed frustration over ChatGPT's lengthy and fragmented outputs, wishing for it to provide concise answers instead of overwhelming information.
   - Recommendations were made to prioritize direct instruction to the model, ensuring it understands the user's preferences for response style.
- **Guidance on Effective Prompting**: A member advised starting a conversation with ChatGPT to clarify needs, which helps improve dialogue and response customization.
   - It's suggested to clearly define any specific requirements or quirks to guide the model's understanding and output.
- **Attention Management for Structured Formats**: Members discussed the use of markdown or YAML for managing attention, noting that structured formats like JSON can also be effective if formatted correctly.
   - This leads to better engagement with links and promotes clear data presentation, enhancing the overall interaction with GPT.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1337558087061733468)** (11 messages🔥): 

> `Prompt Injection Vulnerability, Markdown for URL Management, Response Management in ChatGPT, Formatting for Effective Responses, Attention Management Techniques` 


- **Prompt Injection Concerns in Deep Research**: A member raised concerns about potential **indirect prompt injection** vulnerabilities in deep research, questioning if enough **sanitization** occurs on scraped pages.
   - Another member suggested that we will soon have more information as the feature is expected to be tested.
- **Markdown Improves GPT's URL Following**: A user noted that GPT performs better when URLs are presented in **markdown** format, enhancing prompt hygiene through organized presentation.
   - Another member supported this point, suggesting that **clean formatting** like YAML or JSON is crucial for effective attention management.
- **Managing Large Data Blocks Effectively**: A member shared that providing contexts over a page in **paged JSON files** leads to better response management from GPT.
   - They emphasized that less dynamic context helps in producing more effective results.
- **Seeking Clarity in GPT Responses**: A user expressed frustration at GPT's scattered responses, asking for a more concise delivery without excessive information.
   - Advice was offered on having clearer conversations with GPT to better communicate specific user preferences.
- **Advice on Communicating Needs to GPT**: One member advised explicitly stating needs to the model to achieve better outputs, highlighting the importance of guiding GPT's understanding.
   - Clarification that quirks or specific conditions should be communicated so GPT can tailor responses appropriately.


  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1337516248841453640)** (644 messages🔥🔥🔥): 

> `Cursor MCP Servers, Perplexity Integration, Agent Mode, MCP Setup, Performance Issues` 


- **Cursor MCP Servers Discussed**: The channel discussed various MCP servers, specifically the Perplexity MCP server, detailing its setup and functionality within Cursor, including how to utilize it effectively.
   - Users shared their experiences and difficulties, with some attempting to integrate various models into their workflows for improved coding assistance.
- **Agent Mode Functionality**: Users explored the functionalities of agent mode and its advantages over standard coding commands, particularly praising its capabilities for debugging and direct communication with models like Perplexity.
   - There was a consensus that integrating different LLMs could enhance the coding experience, particularly with features that allow for searching and real-time assistance.
- **MCP Server Installation Issues**: Several users encountered issues while setting up MCP servers, particularly with command execution and server responses on different operating systems like Mac and Windows.
   - Discussions included troubleshooting command prompts that returned errors or failed to connect, indicating a need for clearer documentation and support.
- **Cursor Rules and Enhancements**: Participants discussed the possibility of creating custom cursor rules that could improve the implementation of specific features while using the Perplexity MCP server.
   - Users emphasized the potential benefits of having integrated cursor rules to streamline workflow and enhance the capabilities of the AI in responding to complex code-related queries.
- **Performance and Limitations**: There were discussions surrounding the performance of various models, including reports of degradation in service and concerns about fast API call limits in Cursor.
   - Participants noted that MCP servers, if used correctly, could alleviate some performance issues and provide better results compared to traditional web scraping methods.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sparkengine.ai">Spark Engine - The AI Sandbox</a>: Turn ideas into AI-powered products, no coding experience required</li><li><a href="https://www.instructa.ai/en/blog/how-to-use-cursor-rules-in-version-0-45">How to use Cursor Rules in Version  0.45</a>: Master coding with the Cursor AI course. Build websites, apps, and software faster, with fewer errors. Perfect for beginners and pros. Create personal blogs to complex web apps effortlessly. No AI exp...</li><li><a href="https://www.pulsemcp.com/servers/supabase-postgrest">PostgREST (Supabase) MCP Server by Supabase | PulseMCP</a>: MCP (Model Context Protocol) Server. Connects to Supabase projects using PostgREST, or standalone PostgREST servers, enabling natural language querying and management of PostgreSQL data.</li><li><a href="https://code.visualstudio.com/">Visual Studio Code - Code Editing. Redefined</a>: Visual Studio Code redefines AI-powered coding with GitHub Copilot for building and debugging modern web and cloud applications. Visual Studio Code is free and available on your favorite platform - Li...</li><li><a href="https://supabase.com/docs/guides/getting-started/ai-prompts">AI Prompts | Supabase Docs</a>: Prompts for working with Supabase using AI-powered IDE tools</li><li><a href="https://smithery.ai/server/@daniel-lxs/mcp-perplexity/">Perplexity MCP Server | Smithery</a>: no description found</li><li><a href="https://smithery.ai/server/mcp-server-perplexity">Perplexity Server | Smithery</a>: no description found</li><li><a href="https://x.com/danperks_/status/1888371923316568310">Tweet from Dan (@danperks_)</a>: looking for some passionate people who live and breath @cursor_ai to join us in user ops!feel free to reach out to me or Eric for more info!Quoting eric zakariasson (@ericzakariasson) we&#39;re expand...</li><li><a href="https://smithery.ai/server/@daniel-lxs/mcp-perplexity">Perplexity MCP Server | Smithery</a>: no description found</li><li><a href="https://x.com/ry">Tweet from FxTwitter / FixupX</a>: Sorry, that user doesn't exist :(</li><li><a href="https://forum.cursor.com/t/ctrl-a-doesnt-work-in-composer-user-message-box-when-last-line-is-an-empty-line/46432">Ctrl+A doesn&#39;t work in Composer user message box when last line is an empty line</a>: Ctrl+A doesn’t work in Composer user message box when last line is an empty line  When pressing Ctrl+A, the cursor moves to the start of the message, but no text is actually selected.     Version: 0.4...</li><li><a href="https://docs.convex.dev/ai/using-cursor">Using Cursor with Convex | Convex Developer Hub</a>: Tips and best practices for using Cursor with Convex</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer">servers/src/puppeteer at main · modelcontextprotocol/servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://docs.cursor.com/">Get Started / Migrate from VS Code – Cursor</a>: no description found</li><li><a href="https://x.com/msfeldstein/status/1888740587698036894?s=46">Tweet from Michael Feldstein (@msfeldstein)</a>: The @cursor_ai Agent can now generate images using @FAL via MCP tools</li><li><a href="https://x.com/ryolu_/status/1888455169081577955?s=46">Tweet from Ryo Lu (@ryolu_)</a>: 4 days in @cursor_ai• Everyone is cracked• @shaoruu keeps chasing me for designs and builds them• 1 meeting per week• Got a line up coming in the next release for you already, here’s a lil’ preview ⚡️</li><li><a href="https://x.com/shaoruu/status/1888757694942904499?t=H6-k0P9YJodb49sinapWAA&s=19">Tweet from ian (@shaoruu)</a>: built a 3D basketball court using @cursor_ai, press &#34;H&#34; to feel like stephen curry 🏀</li><li><a href="https://forum.cursor.com/t/cursor-removing-itself/3035">Cursor removing Itself?</a>: The cursor application is deleting itself.</li><li><a href="https://github.com/daniel-lxs/mcp-starter">GitHub - daniel-lxs/mcp-starter</a>: Contribute to daniel-lxs/mcp-starter development by creating an account on GitHub.</li><li><a href="https://x.com/cursor_ai/status/1889047713419071869">Tweet from Cursor (@cursor_ai)</a>: Cursor going entirely from ticket to PR!We&#39;ve shipped several improvements to Cursor&#39;s agent, including support for custom tools, better semantic search, and the ability to fix lints.</li><li><a href="https://github.com/JeredBlu/guides/blob/main/cursor-mcp-setup.md">guides/cursor-mcp-setup.md at main · JeredBlu/guides</a>: Contribute to JeredBlu/guides development by creating an account on GitHub.</li><li><a href="https://github.com/daniel-lxs/mcp-perplexity">GitHub - daniel-lxs/mcp-perplexity</a>: Contribute to daniel-lxs/mcp-perplexity development by creating an account on GitHub.</li><li><a href="https://github.com/daniel-lxs/mcp-starter/releases/tag/v0.1.3">Release v0.1.3 · daniel-lxs/mcp-starter</a>: Remove log line that might cause issues for Mac users adding mcp-starter to cursor</li><li><a href="https://github.com/daniel-lxs/mcp-starter/releases/tag/v0.1.1">Release v0.1.1 · daniel-lxs/mcp-starter</a>: No longer opens a command prompt window on Windows.Full Changelog: v0.1.0...v0.1.1</li><li><a href="https://smithery.ai/">Smithery - Model Context Protocol Registry</a>: no description found</li><li><a href="https://glama.ai/mcp/servers">Open-Source MCP servers</a>: Enterprise-grade security, privacy, with features like agents, MCP, prompt templates, and more.</li><li><a href="https://cursor.directory/">Cursor Directory</a>: Find the best cursor rules for your framework and language</li><li><a href="https://github.com/eastlondoner/cursor-tools">GitHub - eastlondoner/cursor-tools: Give Cursor Agent an AI Team and Advanced Skills</a>: Give Cursor Agent an AI Team and Advanced Skills. Contribute to eastlondoner/cursor-tools development by creating an account on GitHub.</li><li><a href="https://github.com/getcursor/crawler">GitHub - getcursor/crawler: Easily show documentation to Cursor&#39;s coding AI</a>: Easily show documentation to Cursor&#39;s coding AI. Contribute to getcursor/crawler development by creating an account on GitHub.</li><li><a href="https://github.com/kleneway/awesome-cursor-mpc-server/blob/main/src/index.ts">awesome-cursor-mpc-server/src/index.ts at main · kleneway/awesome-cursor-mpc-server</a>: Example of an MCP server with custom tools that can be called directly from cursor - kleneway/awesome-cursor-mpc-server</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://github.com/microsoft/vscode/issues/240238)">microsoft/vscode</a>: Visual Studio Code. Contribute to microsoft/vscode development by creating an account on GitHub.</li><li><a href="https://www.cursor.com/pricing">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=RCFe1L9qm3E">Cursor + MCP Servers: Complete Setup Guide (Sequential Thinking, Brave Search, &amp; More)</a>: Cursor just added MCP support! In this complete setup guide, I&#39;ll show you how to integrate and use MCP servers (Sequential Thinking, Brave Search, and Puppe...</li><li><a href="https://svelte-llm.khromov.se/">svelte-llm - Svelte 5 and SvelteKit Developer documentation in an LLM-ready format</a>: no description found</li><li><a href="https://www.reddit.com/r/cursor/comments/1ilewzc/does_cursor_dumb_down_when_youve_hit_your_limit/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/cursor/comments/1ileb1w/slow_requests_are_deliberately_slowed_down_and_i/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1337513988296015875)** (599 messages🔥🔥🔥): 

> `AI Image Training, Stable Diffusion Models, ComfyUI, Lora Models, Image Resolution and Quality` 


- **Training Lora Models with Unique Tags**: There is a discussion on how unique tags, like naming bedrooms or streets in training data, can be used to improve consistency in Lora models.
   - Using unique tags is believed to help the model associate specific scenes with those names, enhancing narrative continuity in generated images.
- **Recommended Resolutions for Flux**: The optimal latent sizes for Flux are discussed, with recommendations around 672x1024 or 1024x672 for best results, while 1920x1088 is mentioned as a suitable quick HD generation size.
   - Concerns are raised about generating images at resolutions above 1mp during initial passes, as they may lead to compositional issues.
- **Using ComfyUI with Photoshop Integrations**: Users are discussing the integration of various plugins for ComfyUI with Photoshop, including Auto-Photoshop-StableDiffusion-Plugin and others.
   - These plugins aim to facilitate the generation of stable diffusion images within Photoshop using a ComfyUI backend.
- **Issues and Solutions in Stable Diffusion**: Several users are troubleshooting issues related to GPU errors and slow performance in different UI paths of Stable Diffusion, with suggestions to lower GPU settings to resolve memory issues.
   - There are shared recommendations for using specific settings and maintaining aspect ratios to improve model performance and output quality.
- **Legal Discussions Around AI-Generated Art**: There is a conversation about copyright issues concerning AI-generated images, highlighting a recent case where an AI-produced image received copyright protection due to sufficient human input in its creation.
   - This case is viewed as potentially setting a legal precedent for AI-generated content and its ownership.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://nohello.net">no hello</a>: please don't say just hello in chat</li><li><a href="https://imgur.com/a/w7cLKq0">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://apps.apple.com/us/app/tryfit-ai-outfit-changer/id6740171699">‎TryFit - AI Outfit Changer</a>: ‎HOW TO USE TRYFIT:1)Download the app2)Upload a full-body photo and Outfit Photo you want to try-on3)Click Generate.4)See yourself in new styles instantly*Transform your shopping experience with TryFi...</li><li><a href="https://civitai.com/articles/4248">What is score_9 and how to use it in Pony Diffusion | Civitai</a>: Interested in next version of Pony Diffusion? Read update here: https://civitai.com/articles/5069/towards-pony-diffusion-v7 You may&#x27;ve seen score_9...</li><li><a href="https://www.cnet.com/tech/services-and-software/this-company-got-a-copyright-for-an-image-made-entirely-with-ai-heres-how/">This Company Got a Copyright for an Image Made Entirely With AI. Here&apos;s How</a>: The image, called &quot;A Single Piece of American Cheese,&quot; was created using Invoke&apos;s AI editing platform.</li><li><a href="https://www.deepl.com/">DeepL Translate: The world&#x27;s most accurate translator</a>: Translate texts &amp; full document files instantly. Accurate translations for individuals and Teams. Millions translate with DeepL every day.</li><li><a href="https://www.decart.ai/articles/oasis-interactive-ai-video-game-model">Decart</a>: no description found</li><li><a href="https://x.com/BasedLabsAI/status/1888313013276684711">Tweet from Based Labs AI (@BasedLabsAI)</a>: A quick guide to LoRa Training on BasedLabs ⬇️Comment, retweet, and DM us If you&#39;d like to try it for free 🚀</li><li><a href="https://www.federalregister.gov/documents/2025/02/06/2025-02305/request-for-information-on-the-development-of-an-artificial-intelligence-ai-action-plan">Federal Register :: Request Access</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=1mEggRgRgfg"> - YouTube</a>: no description found</li><li><a href="https://replicate.com/lucataco/dotted-waveform-visualizer">lucataco/dotted-waveform-visualizer – Run with an API on Replicate</a>: no description found</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info</li><li><a href="https://www.youtube.com/shorts/z0qogHHNRSE?feature=share"> - YouTube</a>: no description found</li><li><a href="https://civitai.com/models/257749/pony-diffusion-v6-xl">Pony Diffusion V6 XL - V6 (start with this one) | Stable Diffusion Checkpoint | Civitai</a>: Pony Diffusion V6 is a versatile SDXL finetune capable of producing stunning SFW and NSFW visuals of various anthro, feral, or humanoids species an...</li><li><a href="https://github.com/AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin">GitHub - AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin: A user-friendly plug-in that makes it easy to generate stable diffusion images inside Photoshop using either Automatic or ComfyUI as a backend.</a>: A user-friendly plug-in that makes it easy to generate stable diffusion images inside Photoshop using either Automatic or ComfyUI as a backend. - AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin</li><li><a href="https://github.com/zombieyang/sd-ppp">GitHub - zombieyang/sd-ppp: Communicate between Photoshop and ComfyUI</a>: Communicate between Photoshop and ComfyUI. Contribute to zombieyang/sd-ppp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1337513291806675028)** (541 messages🔥🔥🔥): 

> `Nous Research and AI Development, Reinforcement Learning in AI, Granite 3.1 Model Training, Tree Search Methods in AI, Voice Cloning Technology` 


- **Nous Research's Approach to AI**: Discussion highlighted how Nous Research relies on breakthroughs from larger companies like META and DeepSeek to enhance their AI models, akin to learning from existing codebases before innovating.
   - The conversation also touched on the funding challenges for smaller startups and the importance of developing cheap frontier AI models to remain competitive.
- **Reinforcement Learning and Human Feedback**: A proposed method in reinforcement learning discussed generating multiple outputs for a question and rewarding the model for producing the correct answer after multiple attempts.
   - This method raises questions about the effectiveness of the reward strategy compared to traditional RLHF techniques.
- **Training the Granite 3.1 Model**: User shared plans to run training on Granite 3.1's 3B model, expressing a desire to investigate various training strategies, including a custom RL loop.
   - The aim is to explore the potential of multiple objectives per epoch in a newly designed training setup.
- **Limitations of Tree Search Methods in AI**: The limitations of tree search methods in reasoning tasks were discussed, particularly regarding local optima and the potential for better strategies to be implemented.
   - The conversation suggested that using multiple LLMs with different contexts might offer better problem-solving capabilities.
- **Zonos TTS Model Release**: The release of Zonos, a high-fidelity TTS model with voice cloning capabilities, was shared, highlighting its performance against leading TTS providers.
   - The model's open-source nature under the Apache 2.0 license encourages its adoption in AI development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.cnbc.com/2025/02/10/musk-and-investors-offering-97point4-billion-for-control-of-openai-wsj.html">Musk-led investor group offers $97.4 billion for OpenAI — Altman declines</a>: According to The Wall Street Journal, Elon Musk and a group of investors are offering $97.4 billion to take control of OpenAI. </li><li><a href="https://tenor.com/view/what-the-wtf-gif-25758871">What The GIF - What The Wtf - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/Joseph717171/Hermes-3-Llama-3.1-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF/blob/main/Hermes-3-Llama-3.1-8B-F32.imatrix">Hermes-3-Llama-3.1-8B-F32.imatrix · Joseph717171/Hermes-3-Llama-3.1-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/discussions/22">unsloth/DeepSeek-R1-GGUF · any benchmark results?</a>: no description found</li><li><a href="https://fxtwitter.com/ZyphraAI/status/1888996367923888341">Tweet from Zyphra (@ZyphraAI)</a>: Today, we&#39;re excited to announce a beta release of Zonos, a highly expressive TTS model with high fidelity voice cloning.We release both transformer and SSM-hybrid models under an Apache 2.0 licen...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ikh3vz/openai_is_hiding_the_actual_thinking_tokens_in">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=biUFnS7r55c">Developers are getting screwed.</a>: LEARN: https://learn.typecraft.dev/X: https://x.com/typecraft_devFor the longest time now, the software developer’s path has been a pretty clear one. As a ju...</li><li><a href="https://github.com/jart/cosmopolitan">GitHub - jart/cosmopolitan: build-once run-anywhere c library</a>: build-once run-anywhere c library. Contribute to jart/cosmopolitan development by creating an account on GitHub.</li><li><a href="https://github.com/PsycheFoundation">Psyche Foundation</a>: Psyche Foundation has one repository available. Follow their code on GitHub.</li><li><a href="https://siliconflow.cn/zh-cn/models">Models</a>: Teaming up with excellent open-source foundation models.</li><li><a href="https://github.com/3Simplex/Llama.Cpp-Toolbox">GitHub - 3Simplex/Llama.Cpp-Toolbox: Llama.Cpp-Toolbox is a PowerShell GUI interface.</a>: Llama.Cpp-Toolbox is a PowerShell GUI interface. Contribute to 3Simplex/Llama.Cpp-Toolbox development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/plsYqGjJQN">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=xL6Y0dpXEwc&t=3893s">Lecture Series in AI: “How Could Machines Reach Human-Level Intelligence?” by Yann LeCun</a>: ABOUT THE LECTUREAnimals and humans understand the physical world, have common sense, possess a persistent memory, can reason, and can plan complex sequences...</li><li><a href="https://gist.github.com/eb213dccb3571f863da82e99418f81e8.git">Calibration data provided by Dampf, combines his own efforts on top of Kalomaze&#39;s. Used for calibrating GGUF imatrix files</a>: Calibration data provided by Dampf, combines his own efforts on top of Kalomaze&#39;s. Used for calibrating GGUF imatrix files - calibration_datav3.txt
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1337573773582401557)** (17 messages🔥): 

> `AI Oversight, Layer Merging in LLMs, OVERTHINK Attack on Reasoning Models` 


- **AI Oversight using Model Similarity**: Recent research proposes a probabilistic metric for **language model similarity** based on model mistakes, aiming to enhance **AI oversight**. This method suggests that **LLMs**-as-judges favor similar models, facilitating weak-to-strong generalization with complementary knowledge.
   - _As model capabilities rise, detecting mistakes becomes more challenging, prompting reliance on AI oversight, a concerning trend in model performance._
- **Merging FFN Layers for Efficiency**: A discussion arose about merging successive **FeedForward Network (FFN)** layers into a **Mixture of Experts (MoE)**, potentially improving computational efficiency. **Parallelizing** similar layers may yield performance gains while maintaining accuracy.
   - _Members theorized that treating merged layers as **experts** could enhance overall model output, though the efficiency of such changes remains uncertain._
- **Innovative OVERTHINK Attack**: A new attack, dubbed **OVERTHINK**, targets reasoning LLMs by injecting complex tasks, causing models to slow down by up to **46×** in inference. This method amplifies reasoning tokens without altering the final output, showcasing vulnerabilities in reasoning models.
   - _By introducing decoy tasks during untrusted contexts, OVERTHINK effectively manipulates inference processes, posing risks for models like OpenAI's o1 and o3-mini._


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.02790">Leveraging the true depth of LLMs</a>: Large Language Models demonstrate remarkable capabilities at the cost of high compute requirements. While recent research has shown that intermediate layers can be removed or have their order shuffled...</li><li><a href="https://arxiv.org/abs/2502.04313">Great Models Think Alike and this Undermines AI Oversight</a>: As Language Model (LM) capabilities advance, evaluating and supervising them at scale is getting harder for humans. There is hope that other language models can automate both these tasks, which we ref...</li><li><a href="https://x.com/JaechulRoh/status/1887958947090587927">Tweet from Jaechul Roh (@JaechulRoh)</a>: 🧠💸 &#34;We made reasoning models overthink — and it&#39;s costing them big time.&#34;Meet 🤯 #OVERTHINK 🤯 — our new attack that forces reasoning LLMs to &#34;overthink,&#34; slowing models like Ope...</li><li><a href="https://x.com/JaechulRoh/status/1887965905390538758">Tweet from Jaechul Roh (@JaechulRoh)</a>: 2/ Main Method:   Our OVERTHINK attack injects complex decoy reasoning tasks (e.g., Markov Decision Processes or Sudoku) into untrusted context sources. This causes reasoning LLMs to consume more toke...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1337542610197156021)** (11 messages🔥): 

> `Mistral's Performance, Granite Model Enhancements, LIMO's Mathematical Reasoning, Knowledge Distillation Experiments, Novel Language Model Architecture` 


- **Mistral helps Macron secure investments**: Mistral played a pivotal role in helping Macron secure investments up to **EUR 30-50B** for initiatives in the UAE.
   - This milestone highlights the growing influence of Mistral in high-profile financial dialogues.
- **Granite's Enhanced Reasoning Capabilities**: The **Granite-3.2-8B-Instruct-Preview** model allows users to toggle reasoning with a simple flag, showcasing enhanced thinking capabilities with only **817 curated training samples**.
   - This model is built on prior versions and aims to refine reasoning without extensive data.
- **LIMO Model Sets New Standards in Reasoning**: LIMO demonstrates groundbreaking mathematical reasoning abilities, achieving **57.1% accuracy on AIME** and **94.8% on MATH** with only **817 samples**.
   - This performance signifies a major leap from prior models, utilizing only **1% of traditional training data**.
- **Insights from Knowledge Distillation Experiments**: A member shared findings from knowledge distillation experiments, with the **Distilled 1.5B model** showing notable performance improvements across various datasets.
   - The results underscore the preference for distillation over fine-tuning when dealing with significant model performance gaps.
- **Innovative Language Model Architecture Unveiled**: A novel language model architecture scales computation by reasoning in latent space, improving performance on reasoning benchmarks with **3.5 billion parameters**.
   - This model diverges from chain-of-thought methods, effectively scaling without requiring specialized training data.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cerebras.ai/blog/mistral-le-chat">Cerebras brings instant inference to Mistral Le Chat - Cerebras</a>: Cerebras January update: Fastest DeepSeek R1-70B, Mayo Clinic genomic model, Davos appearance, and more! Learn how we&#039;re accelerating AI with real-time inference, machine learning, and case studi...</li><li><a href="https://arxiv.org/abs/2502.03387">LIMO: Less is More for Reasoning</a>: We present a fundamental discovery that challenges our understanding of how complex reasoning emerges in large language models. While conventional wisdom suggests that sophisticated reasoning tasks de...</li><li><a href="https://arxiv.org/abs/2502.05171#:~:text=We%20study%20a%20novel%20language%20model%20architecture%20that,block%2C%20thereby%20unrolling%20to%20arbitrary%20depth%20at%20test-time.">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>: We study a novel language model architecture that is capable of scaling test-time computation by implicitly reasoning in latent space. Our model works by iterating a recurrent block, thereby unrolling...</li><li><a href="https://huggingface.co/ibm-granite/granite-3.2-8b-instruct-preview">ibm-granite/granite-3.2-8b-instruct-preview · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/bethgelab/lm-similarity">lm-similarity - a Hugging Face Space by bethgelab</a>: no description found</li><li><a href="https://chat.mistral.ai/">Le Chat - Mistral AI</a>: Chat with Mistral AI&#x27;s cutting edge language models.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/plsYqGjJQN">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1337573773582401557)** (17 messages🔥): 

> `AI Oversight in Language Models, Layer Merging Strategies in Neural Networks, Performance Improvements through Layer Parallelization, OVERTHINK Attack on Reasoning Models` 


- **AI Oversight proposes new metric for model similarity**: Research establishes a probabilistic metric for **LM similarity** based on overlap in model mistakes, enhancing **AI Oversight** efficiency.
   - *Model mistakes are becoming harder to find*, raising concerns about increased reliance on AI oversight.
- **Innovative Layer Merging Strategies explored**: Discussion highlighted potentially merging successive **FFN layers** into a **Mixture of Experts (MoE)** to enhance computational efficiency.
   - One member suggested treating similar layers as indivisible components, effectively increasing expert numbers while maintaining performance.
- **Parallelization boosts performance metrics**: Experiments demonstrated that fully parallel evaluation of **attention** and **FFN** layers outperform traditional architectures, yielding greater efficiency.
   - Members discussed merging similar layers into double-wide versions to enhance performance through reduction of similar activations.
- **Introduction of the OVERTHINK attack**: A new method called **OVERTHINK** is introduced to hinder reasoning LLMs, causing **slower responses** and increased token consumption.
   - The attack injects complex tasks like **Sudoku** into inputs, expanding reasoning token usage without altering outputs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.04313">Great Models Think Alike and this Undermines AI Oversight</a>: As Language Model (LM) capabilities advance, evaluating and supervising them at scale is getting harder for humans. There is hope that other language models can automate both these tasks, which we ref...</li><li><a href="https://arxiv.org/abs/2502.02790">Leveraging the true depth of LLMs</a>: Large Language Models demonstrate remarkable capabilities at the cost of high compute requirements. While recent research has shown that intermediate layers can be removed or have their order shuffled...</li><li><a href="https://x.com/JaechulRoh/status/1887958947090587927">Tweet from Jaechul Roh (@JaechulRoh)</a>: 🧠💸 &#34;We made reasoning models overthink — and it&#39;s costing them big time.&#34;Meet 🤯 #OVERTHINK 🤯 — our new attack that forces reasoning LLMs to &#34;overthink,&#34; slowing models like Ope...</li><li><a href="https://x.com/JaechulRoh/status/1887965905390538758">Tweet from Jaechul Roh (@JaechulRoh)</a>: 2/ Main Method:   Our OVERTHINK attack injects complex decoy reasoning tasks (e.g., Markov Decision Processes or Sudoku) into untrusted context sources. This causes reasoning LLMs to consume more toke...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1337875360720617605)** (1 messages): 

> `Profile Page Improvements, User Feedback Request` 


- **Codeium Profile Page Getting Upgrades**: Improvements are underway for the **Codeium profile page**, with an invitation for user input to enhance the experience.
   - A [form](https://windsurf.notion.site/194d73774c0080f0b05ee33699e907b9?pvs=105) has been created for users to suggest which stats and metrics they'd like to see, and it includes open-ended questions for additional ideas.
- **User Input Highly Encouraged**: The team is seeking **user feedback** to make meaningful updates to the profile experience on the platform.
   - Participants are thanked in advance for their suggestions, emphasizing the collaborative nature of this upgrade effort.



**Link mentioned**: <a href="https://www.codeium.com/profile">Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1337516931749773410)** (41 messages🔥): 

> `Jetbrain Extension Limitations, Codeium's Shift to Windsurf, Codeium Issues in IDEs, Payment Restrictions for Russian Users, Multi-file Edit Suggestions Needed` 


- **Jetbrain Extension Lags Behind Windsurf**: There is concern about the **Jetbrain extension** lagging in model availability compared to Windsurf, with users speculating they're abandoning Jetbrain for a **Cursor-centric** approach.
   - *There's frustration over losing functionalities* in existing IDEs, indicating users feel neglected by these changes.
- **Codeium Transitioning to Windsurf Exclusively**: It was announced that a new passive in-text editor experience will soon be exclusive to **Windsurf**, leading to the deprecation of **Supercomplete** on the VSCode plugin.
   - Members expressed their disappointment about the loss of support for **VSCode** and **Jetbrain**, suggesting they feel forced to adopt Windsurf.
- **Issues with Codeium in Integrated Development Environments**: Users reported that Codeium has been freezing their **Rider IDE** when commands are sent, prompting suggestions to submit diagnostic logs to support.
   - Another user noted a problem with Codeium suggestions stopping after extended IDE use, leading to questions about whether there's a refresh solution.
- **Challenges for Russian Users in Accessing Codeium**: Discussion arose around **payment restrictions for Russian users**, emphasizing struggles to secure licenses due to regional limitations and company policies.
   - Users called for clearer communication from Codeium regarding their stance on these restrictions, highlighting frustrations around payment processes.
- **Demand for Multi-file Edit Suggestions in Codeium**: Users are advocating for multi-file edit suggestions in the Codeium extensions, which they currently find in **Windsurf** but not in Codeium.
   - There is a strong desire for this functionality to be integrated into the extensions to enhance usability and streamline workflows.


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1337515297934020608)** (409 messages🔥🔥🔥): 

> `Windsurf Performance Issues, Integration of Different AI Models, User Experiences with Code Changes, Windsurf Feature Requests, Credit System Concerns` 


- **Windsurf Performance Issues**: Users reported issues with Windsurf's code proposal function, stating it no longer displays diffs or allows automatic updates, leading to manual copying and pasting of changes.
   - Additionally, many users expressed frustration over loss of credits due to reversion errors and ongoing problems with various AI models.
- **Integration of Different AI Models**: Users discussed the need for consistent tool calling among models like O3, Deepseek, and Claude, with mixed experiences reported when switching between them.
   - Some users found success by relying on Claude for making changes, while others expressed a desire for O3 High for improved coding capabilities.
- **User Experiences with Code Changes**: There were discussions about the nuances of switching between AI models and how responses might be remembered or lost during transitions.
   - Users suggested prompting AI to apply previous suggestions as a workaround for disruptions caused by switching contexts.
- **Windsurf Feature Requests**: Some users suggested features like the ability to manage credits better, notifications about system issues, and improving the design documents to alleviate workflow disruptions.
   - The community frequently referenced the need for improved debugging and consistency in the output generated from the AI models.
- **Credit System Concerns**: Concerns were raised around the credit system, especially regarding how credits are consumed during operations and the lack of refunds for unsuccessful attempts.
   - Users commonly noted that spending credits for unsatisfactory outputs has been a frustration, urging for a more transparent handling of usage.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.pulsemcp.com/posts/newsletter-deep-research-mcp-use-cases-windsurf-mcp#featured">Deep Research clones, MCP Use Cases, Windsurf + MCP | PulseMCP</a>: New this week ending Feb 8, 2025: Deep Research clones, MCP Use Cases, Windsurf + MCP</li><li><a href="https://smithery.ai">Smithery - Model Context Protocol Registry</a>: no description found</li><li><a href="https://tenor.com/view/american-psycho-patrick-bateman-american-psycho-gif-7212093">American Psycho Patrick Bateman GIF - American Psycho Patrick Bateman American - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://drive.google.com/file/d/1lBMLRjoh9Fdju_U4J3NEBZahecIC52uH/view?usp=sharing">Screen Recording 2025-02-08 124422.mp4</a>: no description found</li><li><a href="https://status.codeium.com/#),">Codeium Status</a>: no description found</li><li><a href="https://arstechnica.com/security/2025/02/deepseek-ios-app-sends-data-unencrypted-to-bytedance-controlled-servers/">DeepSeek iOS app sends data unencrypted to ByteDance&#x2d;controlled servers</a>: Apple&rsquo;s defenses that protect data from being sent in the clear are globally disabled.</li><li><a href="https://codeium.com/faq#feedback">FAQ | Windsurf Editor and Codeium extensions</a>: Find answers to common questions.</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>: Contact the Codeium team for support and to learn more about our enterprise offering.</li><li><a href="https://codeium.canny.io/feature-requests/p/multi-model-agentic-performance-for-code-creation-and-editing">Multi-Model Agentic Performance for Code Creation and Editing | Feature Requests | Codeium</a>: The evidence is mounting that combining different models can lead to more optimal performance in code creation and editing tasks.</li><li><a href="https://github.com/GreatScottyMac">GreatScottyMac - Overview</a>: GreatScottyMac has 3 repositories available. Follow their code on GitHub.</li><li><a href="https://codeium.canny.io/feature-requests?search=vertical">Feature Requests | Codeium</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://github.com/microsoft/PromptWizard">GitHub - microsoft/PromptWizard: Task-Aware Agent-driven Prompt Optimization Framework</a>: Task-Aware Agent-driven Prompt Optimization Framework - microsoft/PromptWizard</li><li><a href="https://github.com/GreatScottyMac/cascade-memory-bank">GitHub - GreatScottyMac/cascade-memory-bank: 🧠 Intelligent project memory system for Windsurf IDE. Empowers Cascade AI to maintain deep context across sessions, automatically documenting decisions, progress, and architectural evolution. Perfect for complex projects that demand consistent understanding over time.</a>: 🧠 Intelligent project memory system for Windsurf IDE. Empowers Cascade AI to maintain deep context across sessions, automatically documenting decisions, progress, and architectural evolution. Perfe.....</li><li><a href="https://www.managedv.com">ManagedV - We Launch AI-First Ventures</a>: no description found</li><li><a href="https://x.com/jackccrawfod">Tweet from FxTwitter / FixupX</a>: Sorry, that user doesn't exist :(</li><li><a href="https://about.me/jackccrawford">Jack C Crawford on about.me</a>: I am a Generative AI Maven, consultant, and small business owner in Irvine, California. Visit my website.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1337514309445750795)** (1 messages): 

> `Reasoning Tokens Visibility, Model Activity Pages` 


- **Reasoning Tokens Now Visible**: Users can now view **reasoning tokens** in model activity pages, displayed alongside **prompt** and **completion tokens**.
   - This feature enhances transparency in evaluating model performance, as illustrated in the attached image.
- **Insightful Display of Model Metrics**: The introduction of viewing **reasoning tokens** aligns with ongoing efforts to improve user insights into model performance metrics.
   - Such changes encourage deeper analysis and understanding among users regarding how models operate.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1337528042331050025)** (2 messages): 

> `chat-thyme Discord bot, FindSMap application, Open Router integration` 


- **chat-thyme: Discord Bot Made Easy**: [Chat-thyme](https://github.com/chilir/chat-thyme) is a system designed for setting up Discord bots with any LLM framework compatible with OpenAI, allowing for seamless integration with OpenRouter.
   - It also offers **search capabilities** with Exa for models that support tool use, though its reliability varies by provider.
- **FindSMap PWA: Mapping History**: [FindSMap](http://findsmap.com) is a progressive web application that connects to historical maps and archaeological institutes globally, using Open Street Maps and Leaflet.js for mapping.
   - Built with **Claude** and **Open Router**, it has undergone a long iterative process, showcasing the developer's growth and commitment to the project.



**Link mentioned**: <a href="http://findsmap.com">FindsMap - Research, Explore and Log Your Metal Detecting Finds</a>: no description found

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1337515868204040283)** (291 messages🔥🔥): 

> `DeepSeek R1 performance issues, Gemini models and pricing, API request limitations, User experience with model outputs, Account management concerns` 


- **DeepSeek R1 experiencing timeouts**: Users have reported significant performance issues with DeepSeek R1, particularly regarding timeouts when making requests.
   - The 'nitro' variant for R1 is now integrated into the main model features, allowing users to sort by throughput.
- **Concerns over Gemini model pricing**: Some users expressed frustration regarding the cost of using the Gemini Pro 1.5 model, which is seen as expensive despite being cheaper than some competitors.
   - Others suggested exploring newer models like Gemini 2.0 Flash for better pricing and performance.
- **Issues with API request quotas**: Several users faced 'Quota exceeded' errors when using API requests, indicating that their usage limits may have been reached.
   - Provider responses indicated a temporary service disruption, but some users were still able to access models without issues.
- **User experiences with model output quality**: Debates emerged around the relative quality of various AI models, with many asserting that certain models like Sonnet 3.5 outperform others in practical applications.
   - Discussions included experiences with how different models handle context and reasoning tasks.
- **Account and data management challenges**: Users raised concerns about the potential loss of chat history and difficulties in managing account settings effectively.
   - There were also discussions about accessing models with specific provider keys without incurring costs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ollama.ai">Ollama</a>: Get up and running with large language models.</li><li><a href="https://openrouter.ai/qwen/qwen2.5-vl-72b-instruct:free">Qwen2.5 VL 72B Instruct (free) - API, Providers, Stats</a>: Qwen2.5-VL is proficient in recognizing common objects such as flowers, birds, fish, and insects. Run Qwen2.5 VL 72B Instruct (free) with API</li><li><a href="https://openrouter.ai/activity.">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://x.com/vipulved/status/1888021545349742592">Tweet from Vipul Ved Prakash (@vipulved)</a>: Rolling out a new inference stack for DeepSeek R1 @togethercompute that gets up to 110 t/s on the 671B parameter model!</li><li><a href="https://openrouter.ai/qwen/qwen-vl-plus:free">Qwen VL Plus (free) - API, Providers, Stats</a>: Qwen&#x27;s Enhanced Large Visual Language Model. Significantly upgraded for detailed recognition capabilities and text recognition abilities, supporting ultra-high pixel resolutions up to millions of...</li><li><a href="https://x.com/heyshrutimishra/status/1888905083762737649">Tweet from Shruti Mishra (@heyshrutimishra)</a>: 🚨 China JUST dropped another AI model that beats OpenAI, DeepSeek, and Meta.o1-level reasoning, 200K characters context window, 50 files, real-time search in 1000+ webpages.Here&#39;s everything you ...</li><li><a href="https://ai.google.dev/gemini-api/docs/code-execution?lang=python)">no title found</a>: no description found</li><li><a href="https://github.com/simplescaling/s1">GitHub - simplescaling/s1: s1: Simple test-time scaling</a>: s1: Simple test-time scaling. Contribute to simplescaling/s1 development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/models?fmt=cards&order=newest&providers=Groq">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://groq.com/pricing/">Groq is Fast AI Inference</a>: Groq offers high-performance AI models &amp; API access for developers. Get faster inference at lower cost than competitors. Explore use cases today!</li><li><a href="https://x.com/sama/status/1889059531625464090">Tweet from Sam Altman (@sama)</a>: no thank you but we will buy twitter for $9.74 billion if you want
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1337871368875806790)** (1 messages): 

> `OpenRouter Integration, Typescript SDK for LLMs` 


- **Building OpenAI-Formatted LLM Library**: A team is developing a **TypeScript SDK** to call over **60 LLMs** using OpenAI's format and has just integrated **OpenRouter** for this purpose.
   - *Feedback is appreciated* as they acknowledge the work might still be **rough around the edges**.
- **GitHub Repository for the Project**: They shared a [GitHub link](https://github.com/lunary-ai/abso) for the **abso** project, aimed at facilitating calls to **100+ LLM Providers** using OpenAI's format.
   - The repository promises a comprehensive **TypeScript SDK** for developers looking to implement this functionality.



**Link mentioned**: <a href="https://github.com/lunary-ai/abso">GitHub - lunary-ai/abso: TypeScript SDK to call 100+ LLM Providers in OpenAI format.</a>: TypeScript SDK to call 100+ LLM Providers in OpenAI format. - lunary-ai/abso

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1337517069083611148)** (216 messages🔥🔥): 

> `Aider Performance and Configurations, DeepSeek API Stability, Model Comparisons, Gemini Usage, Language Support and Benchmarks` 


- **Aider's Impact on Confidence in Code**: Users have expressed mixed feelings about Aider, with some reporting increased confidence in its output despite potential flaws in underlying models.
   - One user humorously noted that Aider could write complex code but struggle with basic syntax correctness.
- **DeepSeek API Concerns**: Multiple users reported instability and unresponsiveness when using DeepSeek APIs, particularly in the context of integrating with Aider.
   - One user mentioned troubleshooting issues when attempting to get outputs via DeepSeek with specific configurations.
- **Model Comparisons among Providers**: Discussions around the effectiveness of different providers for DeepSeek's R1 and V3 revealed preferences for Hyperbolic and OpenRouter over others.
   - Users noted specific configurations and tools that enhanced performance when working with different models.
- **Utilization of Gemini Models**: Several users shared experiences with using Gemini models like `gemini-1206-exp`, highlighting its effectiveness for PHP tasks.
   - Comparisons were made between Gemini and other providers, with some users emphasizing the lack of noticeable differences in output.
- **Language Support Enhancements**: The introduction of experimental support for tree-sitter-language-pack aims to expand Aider's programming language capabilities.
   - Users were encouraged to test this new feature and provide feedback regarding its installation and language support effectiveness.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages">In-chat commands</a>: Control aider with in-chat commands like /add, /model, etc.</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-c">In-chat commands</a>: Control aider with in-chat commands like /add, /model, etc.</li><li><a href="https://aider.chat/docs/config/dotenv.html">Config with .env</a>: Using a .env file to store LLM API keys for aider.</li><li><a href="https://tenor.com/view/zoolander-zoolander-movie-movie-zoolander-ben-stiller-benstiller-gif-4425833449756546803">Zoolander Zoolander Movie GIF - Zoolander Zoolander movie Movie zoolander - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/faq.html">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>: How to configure aider with a yaml config file.</li><li><a href="https://tenor.com/view/its-an-illusion-creepy-jason-ink-master-s14e3-magic-gif-26755506">Its An Illusion Creepy Jason GIF - Its An Illusion Creepy Jason Ink Master - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/llms/openrouter.html">OpenRouter</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/config/options.html#--cache-keepalive-pings-value?">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://aider.chat/docs/more/edit-formats.html">Edit formats</a>: Aider uses various “edit formats” to let LLMs edit source files.</li><li><a href="https://ai.google.dev/gemini-api/docs/models/experimental-models#available-models">no title found</a>: no description found</li><li><a href="https://github.com/ai-christianson/RA.Aid">GitHub - ai-christianson/RA.Aid: Develop software autonomously.</a>: Develop software autonomously. Contribute to ai-christianson/RA.Aid development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/issues/1293">Enhance: Add project specific rules in .aiderrules · Issue #1293 · Aider-AI/aider</a>: Issue Currently we can include instructions by adding for example markdown files to the Chat. For project-specific instructions, you could include the instructions in a .aiderrules file in the root...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1337513278703931434)** (70 messages🔥🔥): 

> `Aider Configuration, Model Performance Comparisons, Usage of Aider Features, Architect Mode, Command Line Operations` 


- **Managing Aider in Architect Mode**: Users are experiencing issues with Aider auto-creating files without prompting when in Architect mode, leading to confusion about the operation flow.
   - One user shared a screenshot showcasing the unexpected behavior, indicating potential configuration issues.
- **Aider's Chat History Limits**: Concerns were raised about Aider's chat history exceeding reasonable limits, with some users noticing it climbing to **25k tokens**.
   - Discussion around potential bugs and the effectiveness of using prompt caching in relation to this issue was highlighted.
- **Using Ollama Models with Aider**: Local Ollama models with customized context sizes are being used, but users reported warnings that suggest the context size being employed isn't equivalent to the model's capabilities.
   - Questions about performance and functionality surface, particularly regarding handling code requests effectively.
- **Training and Best Practices for Aider**: One user is exploring ways to effectively train their team on Aider's best practices and interactions to improve efficiency in a startup environment.
   - They shared interest in utilizing various Aider features like `--yes-always` and test-driven development workflows.
- **Confusion Surrounding Aider's GitHub Copilot Integration**: A user inquired about the exact model used in GitHub Copilot named o3 mini, questioning its classification as low, mid, or high.
   - Another user expressed interest in obtaining reasoning summaries for the o3-mini model, highlighting curiosity around model performance metrics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/ollama.html">Ollama</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/llms/openrouter.html#controlling-provider-selection">OpenRouter</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: Intro and tutorial videos made by aider users.</li><li><a href="https://aider.chat/docs/config/options.html">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://github.com/Aider-AI/aider/issues/3153#issuecomment-2640194265">Feature Request: Allow discussion with architect before accepting · Issue #3153 · Aider-AI/aider</a>: I use aider like many others with an architect and editor model. There have been many times I query something, but I notice that the architect misunderstood some point or I did not explain it in en...</li><li><a href="https://github.com/Aider-AI/aider/blob/f7dd0fc58201711c4e483fa4340e3cb1fbd224c3/aider/models.py#L237-L240">aider/aider/models.py at f7dd0fc58201711c4e483fa4340e3cb1fbd224c3 · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1338168834556694609)** (8 messages🔥): 

> `Copilot Proxy Extension, Outline Script for Aider, C++ Code Challenges, GitHub Integration Feedback` 


- **Copilot Proxy Unlocks New Possibilities**: A member introduced the experimental [Copilot Proxy](https://github.com/lutzleonhardt/copilot-proxy), a VS Code extension designed to enable AI assistants access to GitHub Copilot's language models.
   - They shared a [YouTube video](https://youtu.be/i1I2CAPOXHM) detailing the extension's functionality and potential.
- **Community Seeks Script for Outlining Code**: A member expressed frustration after their support comment on GitHub was not merged, seeking ways to utilize the Copilot Proxy work for their needs.
   - Another member suggested using the [llmap repo](https://github.com/jbellis/llmap) and provided guidance on using its `parse.py` script to extract file outlines.
- **Struggles with Massive C++ Codebase**: A member revealed their challenges in managing a massive C++ codebase developed over 10 years, reflecting on AI's token limits during the process.
   - They mentioned needing to add an scm file for effective outlining, which they later found in the repo.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/i1I2CAPOXHM">Aider Integration with Copilot Proxy: Expanding Language Model Access</a>: Unlock GitHub Copilot Models with My New Copilot Proxy: Step-by-Step Guide🔗 GitHub Repository:https://github.com/lutzleonhardt/copilot-proxy🎥 Join me for a...</li><li><a href="https://github.com/jbellis/llmap">GitHub - jbellis/llmap</a>: Contribute to jbellis/llmap development by creating an account on GitHub.</li><li><a href="https://github.com/lutzleonhardt/copilot-proxy">GitHub - lutzleonhardt/copilot-proxy: Copilot Proxy is a Visual Studio Code extension that exposes the VS Code Language Model API via an Express server. This experimental extension is intended solely for research and prototyping purposes and should not be used in production environments.</a>: Copilot Proxy is a Visual Studio Code extension that exposes the VS Code Language Model API via an Express server. This experimental extension is intended solely for research and prototyping purpos...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1337523048852488222)** (126 messages🔥🔥): 

> `DeepSeek AI Models, Anthropic Economic Index, Replit Mobile App Support, AGI Discussions, Open Source Software and Secrets` 


- **DeepSeek AI Models gaining traction in China**: Chinese consumer GPU manufacturers have adapted support for DeepSeek's R1 LLM models on local systems, marking significant progress in AI hardware capabilities in China.
   - With Moore Threads and Baidu's Kunlun GPUs, the competition to challenge NVIDIA's dominance in AI is intensifying.
- **Anthropic Economic Index Launch**: Anthropic has launched the Economic Index to analyze the impact of AI on the economy, which includes a paper based on millions of anonymized Claude conversations.
   - Initial findings reveal interesting patterns with notable areas like material transportation showing surprisingly low engagement.
- **Replit Introduces Native Mobile App Support**: Replit announced early access for Native Mobile App support allowing users to create iOS and Android apps without coding, powered by Replit Assistant.
   - The launch suggests a pivot towards making app development more accessible, with promises of full agent support soon.
- **AGI Discussions and Perceptions**: Discussion points revolve around what defines AGI, with definitions suggesting AGIs should be independent workers trusted to complete tasks rather than merely assistants.
   - Views highlighted the need for continual assessment of AGI based on emerging technologies and their implications.
- **Open Source Software vs. Secrets Debate**: Stratechery's insights emphasize the increasing value of open-source software alongside the challenges of maintaining secretive competitive advantages in AI.
   - It was noted that many supposed secrets may not be as secure as companies believe, suggesting a faster diffusion of knowledge in the field.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1889059531625464090">Tweet from Sam Altman (@sama)</a>: no thank you but we will buy twitter for $9.74 billion if you want</li><li><a href="https://gradual-disempowerment.ai/">Gradual Disempowerment</a>: no description found</li><li><a href="https://notebooklm.google/">Google NotebookLM | Note Taking &amp; Research Assistant Powered by AI</a>: Use the power of AI for quick summarization and note taking, NotebookLM is your powerful virtual research assistant rooted in information you can trust.</li><li><a href="https://scholarqa.allen.ai/">Ai2 ScholarQA</a>: no description found</li><li><a href="https://x.com/DimitrisPapail/status/1888325914603516214">Tweet from Dimitris Papailiopoulos (@DimitrisPapail)</a>: AIME I 2025: A Cautionary Tale About Math Benchmarks and Data ContaminationAIME 2025 part I was conducted yesterday, and the scores of some language models are available here: https://matharena.ai tha...</li><li><a href="https://forum.openai.com/public/events/openais-super-bowl-ad-introducing-the-intelligence-age-4yefoxsgmg?agenda_day=67a6134762deac16356f3c82&agenda_filter_view=stage&agenda_stage=67a6134762deac16356f3c87&agenda_track=67a6134862deac16356f3c96&agenda_view=list">OpenAI’s Super Bowl Ad: Introducing the Intelligence Age - Event | OpenAI Forum</a>: no description found</li><li><a href="https://alphaxiv.org">alphaXiv</a>: Discuss, discover, and read arXiv papers.</li><li><a href="https://x.com/mbalunovic/status/1887962694659060204?s=46">Tweet from Mislav Balunović (@mbalunovic)</a>: We finally have an answer to the debate over whether LLMs generalize to new math problems or they merely memorized the answers.We evaluated them on the AIME 2025 I competition from *yesterday* and the...</li><li><a href="https://x.com/iruletheworldmo/status/1888673201263157279">Tweet from 🍓🍓🍓 (@iruletheworldmo)</a>: anthropic are currently testing claude 4.0 sonnet (chocolate) and claude 4.0 haiku (kiwi) in the lmsys battle mode. the &#39;red teaming&#39; they are currently running is from their latest model. ful...</li><li><a href="https://wccftech.com/chinese-gpu-manufacturers-push-out-support-for-running-deepseek-ai-models-on-local-systems/">Chinese GPU Manufacturers Push Out Support For Running DeepSeek&#039;s AI Models On Local Systems, Intensifying the AI Race</a>: Chinese consumer GPU manufacturers have now started to bring support for running DeepSeek&#039;s R1 LLM models on local systems.</li><li><a href="https://x.com/docmilanfar/status/1888036626573705314?s=46">Tweet from Peyman Milanfar (@docmilanfar)</a>: Michael Jordan gave a short, excellent, and provocative talk recently in Paris - here&#39;s a few key ideas- It&#39;s all just machine learning (ML) - the AI moniker is hype - The late Dave Rumelhart ...</li><li><a href="https://x.com/teortaxestex/status/1887991191037227176?s=46">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: AIME-2025 is out.Non-reasoners cannot into hard math, it&#39;s not even close.But man o3-mini is very good at this and cheap. R2 can&#39;t come soon enough.</li><li><a href="https://x.com/amasad/status/1888727685825699874?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Amjad Masad (@amasad)</a>: Announcing Native Mobile App support on Replit.Now you can build iOS and Android apps that you can take all the way to the App Store without writing any code, powered by Replit Assistant.This is early...</li><li><a href="https://x.com/_avichawla/status/1888113032418705494?t=Nezck9n_J6I9OrhqmAbmlg&s=19">Tweet from Avi Chawla (@_avichawla)</a>: Let&#39;s build our own reasoning model (like DeepSeek-R1) 100% locally:*-</li><li><a href="https://matharena.ai/">MathArena.ai</a>: MathArena: Evaluating LLMs on Uncontaminated Math Competitions</li><li><a href="https://x.com/iruletheworldmo/status/1888978299159756878">Tweet from 🍓🍓🍓 (@iruletheworldmo)</a>: someone inside anthropic told me they’re releasing claude 4 this week. and a reasoning model. blows past full o3 scores. really excited.</li><li><a href="https://x.com/zyphraai/status/1888996367923888341?s=46">Tweet from Zyphra (@ZyphraAI)</a>: Today, we&#39;re excited to announce a beta release of Zonos, a highly expressive TTS model with high fidelity voice cloning.We release both transformer and SSM-hybrid models under an Apache 2.0 licen...</li><li><a href="https://x.com/AnthropicAI/status/1888954156422992108">Tweet from Anthropic (@AnthropicAI)</a>: Today we’re launching the Anthropic Economic Index, a new initiative aimed at understanding AI&#39;s impact on the economy over time.The Index’s first paper analyzes millions of anonymized Claude conv...</li><li><a href="https://elicit.com/">Elicit: The AI Research Assistant</a>: Use AI to search, summarize, extract data from, and chat with over 125 million papers. Used by over 2 million researchers in academia and industry.</li><li><a href="https://www.youtube.com/watch?v=CRlqqp45D74)">int8 tensorcore matmul for Turing</a>: Speaker: Erik Schultheis</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations. - stanford-oval/storm</li><li><a href="https://blog.samaltman.com/three-observations">Three Observations</a>: Our mission is to ensure that AGI (Artificial General Intelligence) benefits all of humanity. Systems that start to point to AGI* are coming into view, and so we think it’s important to...</li><li><a href="https://youtu.be/3lXphIYfoBM?si=rF-paJd2aLvfhWMh">Talking about AI with the Italian Michael Jordan</a>: Prof. Michael Jordan offers his provocative thoughts on the blending of AI and economics and takes us on a tour of Trieste, a beautiful and grand city in nor...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1il2vwi/aicom_now_redirects_to_deepseek/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.lemonde.fr/en/france/article/2025/02/06/uae-to-invest-billions-in-france-ai-data-center_6737871_7.html">UAE to invest billions in France AI data center</a>: The project announced by the French presidency was signed as global experts debate the threats and promise of artificial intelligence at a gathering in Paris on Thursday and Friday, ahead of a summit ...</li><li><a href="https://www.youtube.com/watch?v=W0QLq4qEmKg&t=3810s&pp=2AHiHZ"> - YouTube</a>: no description found</li><li><a href="https://www.emergentmind.com/">Emergent Mind: AI Research Assistant</a>: Research-backed answers to your questions.</li><li><a href="https://github.com/vllm-project/vllm/issues/9807">[Feature]: Integrate Writing in the Margins inference pattern ($5,000 Bounty) · Issue #9807 · vllm-project/vllm</a>: 🚀 The feature, motivation and pitch Writer has introduced &quot;Writing in the Margins&quot; algorithm (WiM) that boosts results for long context window retrieval. The task is composed from &quot;con...</li><li><a href="https://www.reddit.com/r/LocalLLM/s/SUkLKd68tB">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/sytelus/status/1888972692306669939?s=46">Tweet from Shital Shah (@sytelus)</a>: So, AIME might not be a good test for frontier models after all. For 15 problems in AIME 2025 Part 1, I fired off deep research to find near duplicates. It turns out…  1/n🧵</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/15g2ws2/aicom_is_now_pointing_to_xai/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://techcrunch.com/2025/02/06/amazon-doubles-down-on-ai-with-a-massive-100b-spending-plan-for-2025/">Amazon doubles down on AI with a massive $100B spending plan for 2025 | TechCrunch</a>: Amazon is joining other Big Tech companies by announcing huge AI spending plans for 2025.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/RzUdU77BIZ">Reddit - Dive into anything</a>: no description found</li><li><a href="https://stratechery.com/2025/deep-research-and-knowledge-value/">Deep Research and Knowledge Value</a>: Deep Research is an AGI product for certain narrow domains; it&#8217;s ability to find anything on the Internet will make secret knowledge all the more valuable.</li><li><a href="https://stratechery.com/2025/deep-research-and-knowl">Deep Research and Knowledge Value</a>: Deep Research is an AGI product for certain narrow domains; it&#8217;s ability to find anything on the Internet will make secret knowledge all the more valuable.</li><li><a href="https://consensus.app/">Consensus AI-powered Academic Search Engine</a>: Consensus is a new breed of academic search engine, powered by AI, grounded in science. Find the best papers while getting instant insights and topic synthesis.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1337528354118565903)** (139 messages🔥🔥): 

> `Deep Research Tool Discussion, Social Agents Exploration, AI Proactive Systems, OpenAI's $200 Tax, ELIZA Operating System` 


- **Deep Research Tool Gains Attention**: Members discussed OpenAI's new [Deep Research](https://openai.com/index/introducing-deep-research/) tool, noting its ability to ask clarifying questions before completing research tasks, signaling a shift towards more interactive AI systems.
   - There's a growing interest in comparing it to other tools like [Hugging Face's Deep Research](https://m-ric-open-deep-research.hf.space/) and community-made alternatives.
- **Exploring the Potential of Social Agents**: Participants expressed interest in broader discussions about social agents, with one member highlighting the emerging significance of this area in AI development.
   - There's acknowledgment of the need for more structured exploration into how these agents can enhance user experiences.
- **AI Becomes Proactive in User Interaction**: There were discussions around the value of having AI systems that proactively prompt users, moving beyond reactive models and enhancing engagement levels.
   - This reflects a collective desire for AI to better understand user needs and provide tailored assistance.
- **Debate Over OpenAI’s $200 Fee**: Concerns were raised about the perceived 'OAI Tax' associated with using OpenAI's tools, specifically its $200 fee.
   - Some participants expressed skepticism but acknowledged that valuable alternatives are limited.
- **Introduction to ELIZA Operating System**: Members were introduced to the [ELIZA Operating System](https://www.elizaos.ai) designed for AI agents, showcasing its foundational role in developing chatbot technology.
   - The relevance of historical chatbots like ELIZA in today's AI context was an interesting angle in the conversation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.elizaos.ai">elizaOS - The Operating System for AI Agents</a>: elizaOS is an open-source protocol for autonomous AI agents (Elizas).</li><li><a href="https://x.com/voooooogel/status/1887793678112149612">Tweet from thebes (@voooooogel)</a>: so these people were being assholes to teor, so let&#39;s look into their &#34;quantum geometric tensor&#34; library (pumpfun in bio btw), i&#39;m sure we&#39;ll find some, uh, gemsQuoting Teortaxes▶️...</li><li><a href="https://github.com/go-go-golems/pinocchio">GitHub - go-go-golems/pinocchio: pinocchio LLM utility</a>: pinocchio LLM utility. Contribute to go-go-golems/pinocchio development by creating an account on GitHub.</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://m-ric-open-deep-research.hf.space/),">Gradio</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1337516819879301211)** (256 messages🔥🔥): 

> `Mojo and Web Programming, VariadicList Initialization in Mojo, Community and Ecosystem Development, Comparison with Other Languages, Understanding of Networking in Development` 


- **Discussion on Mojo's future in web programming**: Members discussed the long-term prospects for Mojo in web development, noting that establishing a robust ecosystem will take considerable time and effort.
   - Many believe that successful Mojo applications will rely on its integration with existing Python libraries, highlighting the need for foundational tools before broader adoption can occur.
- **Challenges with VariadicList initialization in Mojo**: A user raised an issue regarding the initialization of VariadicList in Mojo, providing code examples that failed to create expected outcomes.
   - They specifically inquired about the ability to dynamically repeat elements when using the `pop.variadic.create` operation.
- **Importance of domain knowledge in business development**: The conversation highlighted that understanding the domain is crucial for launching a business, especially in tech where networking knowledge is often critical.
   - Participants noted that many startups skip this understanding, leading to challenges that could have been avoided.
- **Network effects and language adoption**: Discussion focused on how network effects affect the adoption of programming languages like Rust, pointing out that a strong ecosystem promotes easier experimentation.
   - While some believe in the inevitability of slop as part of rapid development, others argue for maintaining high-quality standards.
- **C++ dominance in high-performance applications**: The group reflected on the prevalence of C++ in companies prioritizing performance optimization, discussing its impact on language adoption.
   - There was a consensus that while Mojo could gain traction, its growth would substantially depend on its compatibility and integration with established languages.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://render.com">Cloud Application Platform | Render</a>: On Render, you can build, deploy, and scale your apps with unparalleled ease – from your first user to your billionth.</li><li><a href="https://github.com/modular/mojo/issues/3987)">modular/mojo</a>: The Mojo Programming Language. Contribute to modular/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/BradLarson/max-cv/blob/main/mojoproject.toml#L21">max-cv/mojoproject.toml at main · BradLarson/max-cv</a>: An image processing framework built upon MAX. Contribute to BradLarson/max-cv development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1337514298762727434)** (157 messages🔥🔥): 

> `Firebase/Firestore MCP, Troubleshooting MCP Commands, MCP Performance Issues, Smithery MCP Installer, Claude Desktop Beta Experience` 


- **Searching for Firebase/Firestore MCP**: A user inquired if anyone had discovered a Firebase/Firestore MCP, to which another user pointed to a link that likely confirmed its unavailability.
   - This interaction highlights a gap in MCP tools for specific databases, indicating further exploration is necessary.
- **Common MCP Command Issues**: A user experienced issues adding an MCP server via Cursor, receiving a 'No Tools Found' error and discussing potential path misconfigurations.
   - Suggestions included verifying the correct command path and resetting the application after updates.
- **Concerns Over MCP Performance and Capabilities**: Users expressed frustration with slow tool call responses, attributing issues to the Python SDK's limitations and ongoing bugs after a recent update.
   - Feedback pointed towards the need for better error handling and performance improvements while using MCP in conjunction with Claude Desktop.
- **Ambivalence Toward Smithery MCP Installer**: While Smithery is regarded as a leading MCP installer, concerns about its remote data handling and overhead emerged in discussions.
   - Users emphasized the necessity for a more local alternative to address privacy and efficiency in using MCP tools.
- **Beta Testing Experiences with Claude Desktop**: Multiple users reported crashing instances of the Claude Desktop app while using their MCP servers, leading to a discussion on the unreliability of current features.
   - There was a consensus that the app is still in beta, requiring extensive feedback and improvements before a stable release can be expected.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cursor.com/advanced/model-context-protocol">Advanced / Model Context Protocol (MCP)– Cursor</a>: no description found</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLScfF23aTWBmd6lNk-Pcv_AeM2BkgzN2V7XPKXLjFiEhvFmm-w/viewform">Claude Desktop Quick Feedback</a>: Thanks for trying out our Desktop App, currently in public beta. We would love your feedback on bugs you encounter, rough edges, and feature suggestions. Thanks in advance for your feedback below. Lea...</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/commit/bd742272ab9ef5576cbeff4045560fb2870ce53b">fix: update types to reflext 2024-11-05 schema · modelcontextprotocol/python-sdk@bd74227</a>: no description found</li><li><a href="https://github.com/ClickHouse/mcp-clickhouse">GitHub - ClickHouse/mcp-clickhouse</a>: Contribute to ClickHouse/mcp-clickhouse development by creating an account on GitHub.</li><li><a href="https://glama.ai/mcp/servers?searchTerm=firebase&">Open-Source MCP servers</a>: Enterprise-grade security, privacy, with features like agents, MCP, prompt templates, and more.</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol Servers</a>: Model Context Protocol Servers. Contribute to modelcontextprotocol/servers development by creating an account on GitHub.</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/blob/f10665db4c2f676da1131617ad67715952258712/src/mcp/types.py#L995">python-sdk/src/mcp/types.py at f10665db4c2f676da1131617ad67715952258712 · modelcontextprotocol/python-sdk</a>: The official Python SDK for Model Context Protocol servers and clients - modelcontextprotocol/python-sdk</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/pull/85">fix: handle internal notifications during session cleanup by donghao1393 · Pull Request #85 · modelcontextprotocol/python-sdk</a>: fix: handle internal notifications during session cleanupMotivation and ContextAddresses an issue where internal notifications (e.g. &amp;#39;cancelled&amp;#39;) during session cleanup would trigger v...</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/issues/88">Random error thrown on response · Issue #88 · modelcontextprotocol/python-sdk</a>: Describe the bug Sometimes, I see a stacktrace printed in the logs of my mcp server. Claude eventually succeeds to response but I think its good to investigate it. To Reproduce Its hard to reproduc...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1337552977757601983)** (65 messages🔥🔥): 

> `Sampling support in MCP, Modifications to Web Research Code, Superargs use cases, Deployment of MCP servers, Cost management for MCP infrastructure` 


- **Progress on Sampling Support in MCP**: A member is developing **sampling support** in the mcp-agent and has created a model selector based on cost, speed, and intelligence preferences, as detailed in a [Twitter thread](https://x.com/qadri_sarmad/status/1887972767049621881). They seek collaboration and feedback from others who may have similar needs.
   - Another member noted that **MCP SDK Python servers** currently do not support sampling.
- **Enhancements in Web Research Code**: A participant successfully modified the **mzrxai/web-research** code to include proper headers for Chrome and eliminate headers that disclose automation. The project is available on [GitHub](https://github.com/PhialsBasement/mcp-webresearch) for review.
   - The goal of the modification is to improve the functionality of the web research server, allowing it to provide real-time information effectively.
- **Superargs Introduces Runtime Configurations**: Superargs enables dynamic configuration of MCP server arguments during runtime, allowing for delayed variable setups, as demonstrated in a [GitHub repository](https://github.com/supercorp-ai/superargs). This adaptation addresses limitations of **current MCP server designs** by simplifying configurations and tool add-ons.
   - There was a discussion about the potential of using Superargs to create an intelligent assistant that adjusts settings as needed during user interactions.
- **Debate on MCP Server Deployment at Scale**: Concerns were raised about the practicality and costs of deploying **MCP servers at scale**, especially regarding Stateful data and security isolation. Members discussed potential methods of controlling costs, such as pooling resources or utilizing services like DigitalOcean.
   - Some highlighted the challenges users may face with managing such infrastructure, suggesting that a **subscription model** might be a more user-friendly option for managed services.
- **Real-World Usage of MCP Servers**: One participant elaborated on their advanced use cases for MCP servers, particularly in embedded remote assistant applications that require runtime adjustments. They explained how using MCP servers could simplify integration with various APIs while maintaining user data security.
   - Interest was shown in exploring ways to allocate **costs effectively** to users while managing infrastructure challenges.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://marketplace.digitalocean.com/vendors">no title found</a>: no description found</li><li><a href="https://x.com/qadri_sarmad/status/1887972767049621881">Tweet from Sarmad Qadri (@qadri_sarmad)</a>: I built a simple LLM selector that lets you pick an LLM depending on cost, speed and intelligence preferences.  It is based on Model Context Protocol&#39;s model preferences spec, and uses data from @...</li><li><a href="https://github.com/PhialsBasement/mcp-webresearch">GitHub - PhialsBasement/mcp-webresearch: MCP web research server (give Claude real-time info from the web)</a>: MCP web research server (give Claude real-time info from the web) - PhialsBasement/mcp-webresearch</li><li><a href="https://github.com/supercorp-ai/superargs">GitHub - supercorp-ai/superargs: Provide AI MCP server args during runtime.</a>: Provide AI MCP server args during runtime. Contribute to supercorp-ai/superargs development by creating an account on GitHub.</li><li><a href="https://github.com/PederHP/mcpdotnet">GitHub - PederHP/mcpdotnet: .NET implementation of the Model Context Protocol (MCP)</a>: .NET implementation of the Model Context Protocol (MCP) - PederHP/mcpdotnet
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1337608106305261588)** (16 messages🔥): 

> `cuBLAS performance comparison, Matrix-Vector Multiplication vs Matrix-Matrix Multiplication, Analogue Matrix Multiplication Hardware, Load queuing and stalls, L1 hit rate` 


- **cuBLAS shows varied performance on different GPUs**: A user reported that **cuBLAS** behaves inconsistently between their **1650ti** and their brother's **4090**, with significant performance differences noted in associated images.
   - They questioned whether the cuBLAS build accommodates **newer architectures** effectively.
- **Matrix-Vector Multiplication clarifies confusion**: It was clarified that the operation **Cx = A(Bx)** tests matrix-vector multiplication (MV) rather than matrix-matrix multiplication (MM).
   - Further discussions revealed that **MM is associative**, thereby validating the approach discussed.
- **Inquiry about analogue matrix multiplication developments**: A member inquired about **mysticAI**, a company working on **analogue** matrix multiplication hardware and its claimed **1000x power efficiency** advantage.
   - Another user provided a link to their current project at [mythic.ai](https://mythic.ai), suggesting potential progress.
- **Concerns about stalls in processing**: Discussions surfaced regarding the frequent **stalls** due to load queuing in operations and a comment on the need for a larger stall bar.
   - Members noted that increasing the **L1 hit rate** might alleviate some of these stalls.
- **Total improvement metrics discussed**: Suggestions arose to illustrate total improvements in processing **time/flops** as a way to better understand efficiency gains.
   - The emphasis on measurable metrics aims to enhance discussions on performance.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1337695140382244894)** (5 messages): 

> `Triton lang Discord Access, Performance of Tensor Cores, Unsloth 30x Faster LLM Training, Mistral 14x Faster Finetuning, Triton Code Contiguity Issues` 


- **Request for Triton lang Discord Access**: A member asked if they could be added to the **Triton lang Discord**, while others showed interest in joining as well.
   - *Complexfilterr* expressed they're also eager to be included in the Discord.
- **Investigating Performance Without Tensor Cores**: *Notyourmom* questioned the performance implications of not using tensor cores in their **03-matmul.py** script on a **3050M** GPU, sharing attached images for context.
   - This sparked curiosity within the community regarding the efficiency of various implementations.
- **Unsloth Promises 30x Faster LLM Training**: A blog post on **Unsloth** details how it can make LLM training **30x faster**, allowing **Alpaca** to train in just **3 hours** instead of **85**.
   - It also boasts **60% less memory usage** and claims no loss in accuracy, with both open source and proprietary options available.
- **Mistral Finetuning Achieves 14x Speedup**: The release of QLoRA support allows finetuning **Mistral 7B** to perform **14x faster** on a single **A100**, using **70% less peak VRAM**.
   - Notably, **CodeLlama 34B** achieved **1.9x speedup**, with memory usage improvements ensuring it doesn't run out of memory.
- **Handling Non-Contiguous Variables in Triton**: *Complexfilterr* raised a question about addressing non-contiguous variables found with **tl.load** in Triton code.
   - They inquired whether generating an explicit buffer would be a viable solution for the issue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/introducing">Introducing Unsloth</a>: no description found</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth update: Mistral support + more</a>: We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1337541758745055292)** (4 messages): 

> `CUDA Kernel Invocations, Kernel Fusion, CUDA Graphs` 


- **Considerations for CUDA Kernel Invocations**: A member questioned if it's practical to care about the number of **CUDA kernels** invoked when chained in a stream, pondering if any performance gains come from fusing them.
   - Another member responded that if fusion avoids **global memory accesses**, it could indeed make a difference, especially when kernels are so short that launch overheads can't be hidden.
- **CUDA Graphs Provide Potential Benefits**: The discussion highlighted that the number of kernels is significant when their launch overhead can't be masked by asynchronous execution.
   - In such cases, utilizing **CUDA Graphs** could be beneficial, but only if they are reused often enough.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1338049535108321290)** (3 messages): 

> `fsdp2 dtensor APIs, CPython C API` 


- **Inquiry on FSDP2 dtensor APIs in C++**: A member asked about the availability of **fsdp2 dtensor APIs** in C++ and whether they exist.
   - They were seeking clarity on the best approach for accessing functionality related to FSDP2.
- **Recommendation to Use CPython C API**: Another member responded that since **FSDP2 is implemented in Python**, it is likely better to use the **CPython C API** for making Python calls.
   - This suggestion implies the absence of a direct C++ implementation for FSDP2 dtensor APIs in this context.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1337875785918320641)** (1 messages): 

> `int8 tensorcore matmul, technical insights` 


- **Excitement for Erik's Insights on Tensorcore**: The community is buzzing with excitement as **Erik** shares his expertise on the **int8 tensorcore matmul** for Turing starting now.
   - *Such deep technical insights* from Erik are anticipated to enrich our understanding and discussions in the server.
- **Anticipation Builds for Technical Depth**: Members are looking forward to Erik's deep dive into technical aspects of **tensorcore** matmul, highlighting how it benefits **Turing** architecture.
   - The server is abuzz with eager comments about Erik's ability to provide technical clarity on complex topics.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1338353095331282996)** (5 messages): 

> `Tiling in GEMM, Jeremy Howard's Lectures, Simon Boehm's Blog on Matmul` 


- **Learning Tiling in GEMM**: A member asked for good resources to learn about how **tiling** works, specifically in breaking up **GEMM** into smaller chunks.
   - They expressed interest in materials that include code examples or visualizations.
- **Jeremy Howard's Lectures on YouTube**: Another member recommended checking out **Jeremy Howard's** lectures on YouTube for insights into tiling.
   - The specific **YouTube video** linked is titled *Getting Started With CUDA for Python Programmers* at [this timestamp](https://youtu.be/nOxKexn3iBo?t=2703).
- **Simon Boehm's Blog on Matmul**: A member suggested **Simon Boehm's** blog focused on **CPU** and **CUDA** matrix multiplication as an additional resource.
   - This blog is expected to provide helpful insights and practical examples related to tiling.



**Link mentioned**: <a href="https://youtu.be/nOxKexn3iBo?t=2703">Getting Started With CUDA for Python Programmers</a>: I used to find writing CUDA code rather terrifying. But then I discovered a couple of tricks that actually make it quite accessible. In this video I introduc...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

iron_bound: Run Deepseek from fast NVME storage
https://github.com/BlinkDL/fast.c
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1337520736134234285)** (3 messages): 

> `GPU Glossary Contributions, ROCm Specific Terms` 


- **Excitement for the GPU Glossary**: A member expressed their enjoyment of the **GPU Glossary**, stating they 'absolutely loved' it.
   - This enthusiasm highlighted the community's positive reception of the resource.
- **Interest in Contributing to the Glossary**: Another member inquired if there was a way to contribute to the GPU Glossary, specifically wanting to add **ROCm** related terms and general GPU information.
   - This reflects a desire for community involvement and enhancement of the existing resource.
- **Anticipation for Future Updates**: A response suggested staying tuned for updates, indicating that contributions might be streamlined soon.
   - The phrase 'watch this space' signals an active engagement and expectation of developments regarding contributions.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1337922623279071233)** (11 messages🔥): 

> `Implementing matmul in Assembly, Llama-8b Model Memory Usage, Using eGPU with MacBook Air, Learning CUDA and Resources, Mentorship in CUDA/CUTLASS` 


- **Optimizing matmul kernel performance**: A user reports implementing and optimizing a matmul kernel in x86_64 Assembly, achieving around **105 GFLOPs** on an R7 5800X single core. They seek feedback and improvements on their rudimentary code available [here](https://github.com/alint77/matmul_assembly_x86).
   - Plans to change the matrix storage from row/column major to blocking may push performance to **120 GFLOPs**.
- **Llama-8b runs out of memory in 16BF**: A user inquired why their **llama-8b 16BF model** occupies around **30GB of VRAM** when using L40S or A40 GPUs, questioning if it loads in 32bit instead of 16bit. A fellow user suggested using `torch_dtype='auto'` during the model loading to optimize memory usage.
   - The issue stems from the weights normally loading in full precision (torch.float32) unless specified otherwise.
- **Challenges using eGPU with MacBook Air**: A user asked about hooking up an eGPU to a MacBook Air with an M2 chip for ML model training, considering CUDA options. Another user warned that NVIDIA ceased MacOS driver support since High Sierra, making modern NVIDIA GPUs incompatible with Mac.
   - It was noted that **eGPU** support is also absent for M1 based Macs and current Apple silicon models.
- **Resources for learning CUDA**: A user shared a recommendation for a free online CUDA playground at [leetgpu.com](https://leetgpu.com) to kickstart learning CUDA. Others encouraged leveraging cloud GPU instances or Google Colab for CUDA support.
   - These resources provide hands-on experience without the need for proprietary hardware.
- **Seeking mentorship for deep learning in CUDA**: A user is looking for guidance while studying the PMPP textbook and plans to contribute to CUDA/CUTLASS or ROCM. They aim to strengthen their understanding of deep learning to apply for related jobs in the field.
   - One recommendation is to engage with communities and forums to further enhance their skills and knowledge base.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://leetgpu.com,">no title found</a>: no description found</li><li><a href="https://discussions.apple.com/thread/255161653">Apple Silicon (M1/M2) and eGPU support - Apple Community</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/autoclass_tutorial#automodel:">Load pretrained instances with an AutoClass</a>: no description found</li><li><a href="https://forums.developer.nvidia.com/t/gpu-memory-not-recognized-for-the-code-nsight-compute/323099">GPU memory not recognized for the code - nsight compute</a>: Hi everyone,  I have an nvidia GeForce MX450 with cuda kit 12.8 and graphics driver 572.16 on WSL  I was trying to follow along with a course and run the following code  https://github.com/Infatoshi/c...</li><li><a href="https://github.com/alint77/matmul_assembly_x86">GitHub - alint77/matmul_assembly_x86</a>: Contribute to alint77/matmul_assembly_x86 development by creating an account on GitHub.</li><li><a href="https://www.metacareers.com/jobs/1517576482367228/">Software Engineer, Systems ML -  HPC Specialist</a>: Meta&#039;s mission is to build the future of human connection and the technology that makes it possible.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1337855565078069248)** (8 messages🔥): 

> `Live Video Conversion Issues, Video Quality Concerns, int8 Tensorcore Matmul Course, Course Exercise Performance Statistics` 


- **Live Video Views Split Issue**: A member expressed frustration over not being able to convert a live video into a regular one, causing **view counts to split**.
   - _This issue makes it difficult to track engagement and performance effectively._
- **Video Quality Affects Clarity**: Members noted that the **video quality** hindered the legibility of several **figures and screenshots** in the video.
   - They suggested that slides may provide a clearer view of the content presented in the talk.
- **Promotion of int8 Tensorcore Matmul Video**: A YouTube video titled **'int8 Tensorcore Matmul for Turing'** featuring Erik Schultheis was shared for reference.
   - This video is likely relevant for those enrolled in the related course.
- **Course Updates and Challenges**: A member mentioned that the **int8mm exercise** was updated in their course, making it challenging to achieve full points on specific tasks.
   - They humorously reflected on the difficulty of exercise 3a, still not achieving optimal times.
- **Performance Statistics from Last Year’s Course**: One member shared **performance statistics** for the **CP3a exercise**, detailing submission times of 288 students.
   - The data revealed trends about graded solutions, with specific thresholds highlighted for full points.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=CRlqqp45D74">int8 tensorcore matmul for Turing</a>: Speaker: Erik Schultheis</li><li><a href="https://ppc.cs.aalto.fi/stat/aalto2024/cp3a/">Exercises</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1338420935002296341)** (1 messages): 

> `TorchAO/Gemlite performance regression, Benchmarking Issues, Environment Setup Concerns` 


- **Major drop in TorchAO/Gemlite performance**: A member reported a significant **performance regression** with **TorchAO/Gemlite**, indicating **slower throughput** compared to their benchmarks.
   - *I've filed an issue here* [Performance Regression with TorchAO/Gemlite](https://github.com/mobiusml/gemlite/issues/15) detailing the setup used for benchmarking.
- **Benchmarking setup referenced**: They mentioned using scripts from the [Pytorch Blog on accelerating LLM inference](https://pytorch.org/blog/accelerating-llm-inference/) for the benchmarking tests conducted.
   - The setup included **H100**, **Cuda: 12.6**, **torch: 2.5.1.14+cu126**, and **torchao: 0.8**.
- **Possible setup issues questioned**: The member speculated that there might be a **bug** or issue with their current setup leading to the observed performance drop.
   - They are seeking feedback on whether others face similar performance drops or if it's specific to their environment.



**Link mentioned**: <a href="https://github.com/mobiusml/gemlite/issues/15">Performance Regression with TorchAO/Gemlite: Slower Throughput Compared to sglang.bench_offline_throughput · Issue #15 · mobiusml/gemlite</a>: I&#39;ve benchmarked using the scripts in Pytorch Blog: https://pytorch.org/blog/accelerating-llm-inference/ And here is the setup of my environment: H100 Cuda: 12.6 torch: 2.5.1.14+cu126 torchao: 0.8...

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1337601099863162953)** (2 messages): 

> `YouTube Video Discussion, Pricing Concerns` 


- **YouTube Video Surprise**: A YouTube video titled [" - YouTube"](https://www.youtube.com/watch?v=JJX_U35xa7k) was shared, but no description was provided.
   - *Viewers are left curious about the content as details remain sparse.*
- **Pricing Shock**: A member expressed dismay at some unspecified pricing, stating, "That pricing 😬".
   - An attached image potentially illustrates the pricing concern, sparking discussions about its implications.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=JJX_U35xa7k"> - YouTube</a>: no description found

  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1337903084730585229)** (2 messages): 

> `iGPU programming in Ryzen AI CPU, Graphics frameworks, HIP support` 


- **Exploring iGPU Programming in Ryzen AI**: One member inquired about the possibility of programming the **iGPU** in the **Ryzen AI CPU (Strix Point)**.
   - They are seeking methods or frameworks to leverage this functionality.
- **Graphics Frameworks and HIP as Solutions**: Another member suggested that **graphics frameworks** might be the best approach for programming the iGPU.
   - They also noted that, in theory, **HIP** should work for this purpose.


  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1338074386812375041)** (1 messages): 

> `cost-effective algorithms, topology-based algorithms, all-to-all communication` 


- **Exploring Cost-Effective Alternatives to All-to-All Communication**: A member inquired about solutions that are more efficient than **all-to-all communication**, citing **cost** concerns.
   - They specifically asked if anyone has better alternatives for algorithm choice beyond **topology-based** approaches.
- **Inquiry on Algorithm Choices**: Another discussion focused on the possible choices for algorithms, questioning if topology-based methods were the best route.
   - Members were curious about innovative strategies that could reduce costs while maintaining efficiency.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1337922525018980454)** (1 messages): 

> `Transformers version, Logits issue, LigerCrossEntropy optimization` 


- **Query about Transformers Version**: A query was raised regarding the **transformers version** being used when running tests, suggesting its relevance to the issue at hand.
   - *What's your transformers version when running tests?*
- **Logits Test Issue Linked**: It was pointed out that the problem with the **logits** in the `test_mini_models_with_logits` test might be related to an existing issue reported on GitHub, specifically [Issue #543](https://github.com/linkedin/Liger-Kernel/issues/543).
   - The issue discusses the convergence test failing due to differences in losses and logits from two models.
- **LigerCrossEntropy's Role in Newer Versions**: The message mentioned that **LigerCrossEntropy** has likely optimized logits in newer transformers versions, potentially affecting test outcomes.
   - This change might be a factor in the discrepancies being observed in model performance.



**Link mentioned**: <a href="https://github.com/linkedin/Liger-Kernel/issues/543">The convergence test `test_mini_models_with_logits` is failing with the latest transformers · Issue #543 · linkedin/Liger-Kernel</a>: 🐛 Describe the bug In this convergence test, we compare losses and last step&#39;s logits to make sure two models (w/ and w/o monkey patch) can produce similar results. For logits, LigerCrossEntropy ...

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1337566305745895434)** (1 messages): 

> `GPU Benchmarking, Thread Coarsening, Shared Memory Usage, Memory Coalescing, Synchronization Techniques` 


- **Insights into GPU Benchmarking Techniques**: The member shared that benchmarking was conducted on their laptop using the **4050 GPU**, highlighting the importance of various techniques applicable across GPUs.
   - *Important techniques discussed include* **Thread Coarsening**, **Shared Memory Usage**, **Memory Coalescing**, and ensuring proper synchronization.
- **Understanding Key Optimization Techniques**: The conversation emphasizes the significance of **optimizing memory usage** through greater **Shared Memory Usage** and **Memory Coalescing** for improved performance across GPUs.
   - Community insights suggest that proper **synchronization** of processes can lead to more effective GPU resource management.


  

---


### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1337921317692768277)** (2 messages): 

> `matmul kernel optimization, AVX2 FMA performance, matrix storage order, GitHub matmul project` 


- **Optimizing Matmul Kernel with AVX2 FMA**: A user shared their progress on optimizing a **matmul kernel** in **x86_64 Assembly** using **AVX2** and **FMA**, achieving approximately **105 GFLOPs** on a **R7 5800X** single core.
   - The theoretical maximum performance is **~150 GFLOPs**, and they are currently transitioning to a **blocking storage** method to improve performance to around **120 GFLOPs**.
- **Seeking Advice on Assembly Code**: The user requested feedback on their rudimentary code implementation, expressing a need for guidance as they are not an experienced programmer.
   - They shared a [GitHub repository](https://github.com/alint77/matmul_assembly_x86) where their work can be reviewed, inviting the community to contribute.



**Link mentioned**: <a href="https://github.com/alint77/matmul_assembly_x86">GitHub - alint77/matmul_assembly_x86</a>: Contribute to alint77/matmul_assembly_x86 development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1337627981639585882)** (1 messages): 

> `Associative Scan, Data Layouts in TK, Ortho/Aligned Layouts` 


- **Associative Scan Implementation Queries**: A member is exploring the addition of **associative scans** to TK, starting with vectors and transitioning to tiles, aiming to clarify data layout usage.
   - *Is it even worth it to implement an associative scan for aligned/ortho layouts?* they pondered, questioning the complexity versus necessity.
- **Confusion Around Layout Usage**: The member noted that current implementations like `rv_fl`, `rv_bf`, and `rv_hf` appear to rely on the **naive layout**.
   - They raised the question of whether **aligned/ortho layouts** are actually utilized for vectors in TK.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1338623638630301820)** (1 messages): 

> `PyTorch Edge Team, Discord Channel for On-Device AI, ExecuTorch Library` 


- **PyTorch Edge Team opens Discord to public**: The **PyTorch Edge team** at Meta has recently opened their [Discord channel](https://discord.gg/HqkRfk6V) to the public for discussions on announcements, issues, and releases related to on-device AI.
   - They encourage new members to join and introduce themselves in the introduction channel.
- **Discussion on Contributions to ExecuTorch**: The channel will also serve as a space for contributions to **ExecuTorch**, the on-device library focused on enhancing AI applications.
   - *Feel free to join* and actively participate in discussions surrounding this library and its developments.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1337531971785592937)** (89 messages🔥🔥): 

> `New PRs in reasoning-gym, Benchmarking the gym environment, Matrix manipulation dataset development, OpenRouter sponsorship for inference, Interactive training vision` 


- **Exciting new PRs in reasoning-gym**: Multiple new PRs have been opened, including [Matrix Manipulation](https://github.com/open-thought/reasoning-gym/pull/100) and [Count Bits](https://github.com/open-thought/reasoning-gym/pull/101), showcasing collaborative contributions.
   - The team is eager to evaluate these new additions and potentially explore more datasets.
- **Plan to benchmark the gym environment**: There are discussions about benchmarking the gym environment to see how RL training aids model generalization to unseen tasks, with interests in using OpenRouter for evaluation.
   - Suggestions for pooling resources and sharing scripts were proposed to coordinate the benchmarking effort across team members.
- **Development of matrix manipulation tasks**: The `manipulate_matrix` dataset has received positive feedback, with suggestions for additional configuration options to enhance usability in tasks.
   - Thanks to recent contributions, the repository has expanded to a total of 65 datasets, a significant milestone.
- **Considering sponsorship for inference compute**: OpenRouter may sponsor some compute credits for benchmarking, enabling smoother transitions between inference providers to maintain consistency in evaluation results.
   - A focus on securing sponsorship reflects the team's proactive approach to managing resources for their efforts.
- **Vision for interactive training with reasoning-gym**: A new issue has been opened to propose an interactive training run using CLI commands or a web front end to control dataset configurations dynamically.
   - This innovative approach highlights a potential enhancement in training workflows, allowing for real-time adjustments during experimentation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.04350">CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidance</a>: Existing methods fail to effectively steer Large Language Models (LLMs) between textual reasoning and code generation, leaving symbolic computing capabilities underutilized. We introduce CodeSteer, an...</li><li><a href="https://en.wikipedia.org/wiki/Raven%27s_Progressive_Matrices">Raven&#039;s Progressive Matrices - Wikipedia</a>: no description found</li><li><a href="https://www.youtube.com/@GPUMODE">GPU MODE</a>: A GPU reading group and community https://discord.gg/gpumodeSupplementary content here https://github.com/gpu-modeCreated by Mark Saroufim and Andreas Köpf </li><li><a href="https://openrouter.ai/)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1/providers">DeepSeek: R1 – Provider Status</a>: See provider status and make a load-balanced request to DeepSeek: R1 - DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#...</li><li><a href="https://wordgamebench.github.io),">no title found</a>: no description found</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/104">Interactive training with reasoning-gym server · Issue #104 · open-thought/reasoning-gym</a>: Vision: Launch a training run and use cli-commands (or a web-frontend) to monitor and manipulate the reasoning-gym dataset configuration - to directly control the next batch composition, e.g. add o...</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/95">Add basic matrix manipulation task dataset · Issue #95 · open-thought/reasoning-gym</a>: Write a dataset class with corresponding unit tests for basic matrix manipulation tasks. Dataset entries: The question should include a randomly generated matrix (square or non-square) and instruct...</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/1f9d9d27ab0e0722a900b89e3820bec3435bdd50/reasoning_gym/arc/arc_agi.py#L47-L87">reasoning-gym/reasoning_gym/arc/arc_agi.py at 1f9d9d27ab0e0722a900b89e3820bec3435bdd50 · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/sanjana707/Hacking_game">GitHub - sanjana707/Hacking_game: Password guessing game created using Python.</a>: Password guessing game created using Python. Contribute to sanjana707/Hacking_game development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/93">Add score_answer method to word_ladder by Adefioye · Pull Request #93 · open-thought/reasoning-gym</a>: This is a draft PR to add score_answer() method to WordLadder. Look forward to some feedback.If implementation is satisfactory, I can work on unit tests.</li><li><a href="https://docs.google.com/spreadsheets/d/1qk2BgxzfRZzTzMQnclCr47ioykgltbGkMJUHO2sH6Gw/edit?gid=1210959818#gid=1210959818">reasoning-gym-eval</a>: no description found</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">R1 - API, Providers, Stats</a>: DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with fully open reasoning tokens. It&#x27;s 671B parameters in size, with 37B active in an inference pass. Ru...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/88">Feat/re arc by joesharratt1229 · Pull Request #88 · open-thought/reasoning-gym</a>: The following PR implements the procedural task dataset class including unit tests.** Main Changes **Imports re-arc generator code from re-arc adapted such that there is explicit control over ra...</li><li><a href="https://en.wikipedia.org/wiki/Sieve_">Sieve - Wikipedia</a>: no description found</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/2307">TRL upgrade by winglian · Pull Request #2307 · axolotl-ai-cloud/axolotl</a>: wip towards adding support for GRPO
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1338622036351914066)** (1 messages): 

> `NotebookLM Plus, Google One AI Premium Plan, Student Discount on AI Premium, Enhanced Features of NotebookLM Plus` 


- **NotebookLM Plus joins Google One AI Premium**: Starting today, [NotebookLM Plus](https://blog.google/technology/google-labs/notebooklm-new-features-december-2024/) is included in the Google One AI Premium plans, providing users higher usage limits and premium features for research.
   - This enhances existing benefits including **Gemini Advanced** and **2 TB of storage** for a more valuable package.
- **Students get a sweet deal!**: U.S. students aged 18 and older can now enjoy a **50% discount** on the Google One AI Premium plan, costing just **$9.99/month**.
   - This offer aims to make advanced AI research tools more accessible for students, starting today.
- **NotebookLM Plus boosts chat and sharing tools**: NotebookLM Plus offers advanced chat customization and sharing capabilities, complete with **usage analytics**.
   - Users can now access **5x** the notebooks, **6x** the sources per notebook, and **7x** more audio overviews.
- **Upgrade options for NotebookLM Plus**: Users can upgrade to NotebookLM Plus via the provided link or directly within the NotebookLM interface later today.
   - This upgrade promises to deliver enhanced research features tailored to user needs.



**Link mentioned**: <a href="https://blog.google/feed/notebooklm-google-one">NotebookLM Plus is now available in the Google One AI Premium subscription.</a>: NotebookLM is a research and thinking companion designed to help you make the most of your information. You can upload material, summarize it, ask questions and transfor…

  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1337560727774433392)** (26 messages🔥): 

> `Medical Jargon Assistance, Audio Overview Creation, Versatile Bot Project, Mock Interview Preparation, Video Project Completion` 


- **AI Transforms Medical Jargon into Clarity**: A member shared their experience using AI to navigate **medical jargon** related to their **breast cancer diagnosis**, summarizing dense articles and recording surgeon appointments.
   - They expressed how reassuring it is to challenge the AI for clarifications, highlighting AI's role as a comforting aid during treatment.
- **Customizing Audio Overviews**: Members discussed the inability to create a new **audio overview** without first deleting the existing one, emphasizing the need for more extensive audio summaries.
   - Some suggested specifying topics in the customization section to potentially enhance depth in coverage.
- **Versatile Bot Project Launched**: A user introduced the **Versatile Bot Project**, providing two prompt documents to transform NotebookLM into different types of chatbots through specialized prompts.
   - They mentioned both prompts have been tested and aimed to create a customizable chatbot experience, encouraging community engagement.
- **Mock Interviews Enhanced with AI**: One member described how they utilize NotebookLM to prepare for mock interviews by uploading job descriptions and company information to generate tailored notes.
   - This method allows for a more focused review and preparation process, enhancing their interview readiness.
- **Short Film Produced with NotebookLM's Help**: A user completed a **6.5-minute short film**, leveraging NotebookLM for audio overviews and editing clips to accompany discussions based on microfiction they wrote.
   - They detailed the project's extensive effort, especially in generating over **50 video shots**, illustrating the power of AI-assisted creative processes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/shun0t/versatile_bot_project">GitHub - shun0t/versatile_bot_project: Transforming NotebookLM into a versatile bot</a>: Transforming NotebookLM into a versatile bot. Contribute to shun0t/versatile_bot_project development by creating an account on GitHub.</li><li><a href="https://youtu.be/yHKF8B1BRR8">Stellar Beacon 2:  Galactic Podcast Generated with NotebookLM and VideoFX</a>: Join our intrepid podcasters as they delve into the latest news from the Stellar Beacon News Bulletin. From prison colonies to rogue mercenaries, they explor...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1337526722874183782)** (118 messages🔥🔥): 

> `NotebookLM functionality issues, NotebookLM Plus features, User accounts and sharing, Gemini integration, Language options in NotebookLM` 


- **NotebookLM struggles with source generation**: Multiple users reported issues with NotebookLM not generating notes or summaries from uploaded sources, with one stating 'New Note: Generating' indefinitely.
   - Some suggested that issues might arise from specific file formats like .txt or .pdf, while others noted successful generation when pasting text directly.
- **NotebookLM Plus expands user limits**: NotebookLM Plus offers increased capabilities, such as five times the notebooks and audio overviews compared to the free version, which has inherent limits.
   - Users inquired about the specific limitations of both free and paid versions, directing them to official Google support links for comprehensive details.
- **Challenges with sharing notebooks**: An administrator experienced difficulties sharing notebooks among created user accounts and mentioned that Gmail must be enabled for sharing functionalities to work.
   - Despite setting up SSO from Azure, users could not share notebooks, leading to discussions about the requirements for sharing and accessing accounts.
- **Integration with Gemini**: Discussions highlighted the integration of Gemini with NotebookLM, with users exploring prompts to create study guides and FAQ documents more effectively.
   - Some expressed concerns regarding the potential for hallucinations when mixing responses from sources and online search results.
- **Availability of NotebookLM via Google One**: Users inquired about accessing NotebookLM through Google One subscriptions, with some still facing 'coming soon' notifications despite having subscriptions.
   - Clarifications were sought regarding the rollout timing of features associated with Google One subscriptions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=NotebookLM%20vs%20NotebookLM%20Plus%20User%20Limits">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://www.zdnet.com/article/gemini-can-now-watch-youtube-for-you-skip-the-video-get-the-highlights/">Gemini can now watch YouTube for you - skip the video, get the highlights</a>: Don&apos;t want to wade through an entire video to find what you need? Let Gemini save you time and summarize it for you.</li><li><a href="https://forms.gle/YuQVfPpasUuiNcRp6">Cyborg Technology and Human Enhancement | DS Extended Essay Form</a>: Welcome to this survey on cyborg technology and human enhancement you&#39;ve been selected due to your interest in the following topic!I am exploring how technological advancements reshape our underst...</li><li><a href="https://github.com/DavidLJz/userscripts/tree/master/gemini_download_conversation">userscripts/gemini_download_conversation at master · DavidLJz/userscripts</a>: GreaseMonkey / TamperMonkey Scripts. Contribute to DavidLJz/userscripts development by creating an account on GitHub.</li><li><a href="https://music.apple.com/us/album/it-matters-to-her/1628183633?i=1628183839">It Matters To Her by Scotty McCreery on Apple Music</a>: Song · 2021 · Duration 2:51
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1337535223784804364)** (1 messages): 

> `Sparse Autoencoders (SAEs), Skip Transcoders, Interpretability in Neural Networks, Partial Rewriting of Transformers, EleutherAI Libraries` 


- **Skip Transcoders outperform Sparse Autoencoders**: Introducing **skip transcoders** shows a **Pareto improvement** over **SAEs**, with enhanced **interpretability** and fidelity in neural networks. You can utilize skip transcoders by using flags `--transcode` and `--skip_connection` in the [sparsify](https://github.com/EleutherAI/sparsify) library.
   - *Unlike SAEs*, transcoders better approximate input-output relationships, thereby improving the approach to interpretability.
- **Disappointing results in partial rewriting**: In research on partially rewriting transformers, the team trained a skip transcoder on the sixth layer of **Pythia 160M** but faced lackluster results. They *failed to outperform* a simple baseline of using a zero vector in place of the transcoder.
   - Despite these setbacks, they remain optimistic about refining their methods for more detailed and precise explanations.
- **Interest in Interpretability Research**: The team expresses excitement about improving model interpretability, suggesting they are seeking collaboration and involvement from others. Discussions are ongoing in the **#1153431135414669422** channel for anyone who wants to contribute.
   - Thanks were given to contributors for their work on both the skip transcoder and partial rewriting papers.
- **Links to Recent Research Papers**: The team shared links to their newly published papers, including the [skip transcoder paper](https://arxiv.org/abs/2501.18823) and the [partial rewriting paper](https://arxiv.org/abs/2501.18838). Both papers advance the understanding of mechanistic interpretability in neural networks.
   - Highlights from the abstracts underscore the significance of these architectures in enhancing human-understandable frameworks for machine learning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.18823">Transcoders Beat Sparse Autoencoders for Interpretability</a>: Sparse autoencoders (SAEs) extract human-interpretable features from deep neural networks by transforming their activations into a sparse, higher dimensional latent space, and then reconstructing the ...</li><li><a href="https://arxiv.org/abs/2501.18838">Partially Rewriting a Transformer in Natural Language</a>: The greatest ambition of mechanistic interpretability is to completely rewrite deep neural networks in a format that is more amenable to human understanding, while preserving their behavior and perfor...</li><li><a href="https://x.com/norabelrose/status/1887972442104316302">Tweet from Nora Belrose (@norabelrose)</a>: Sparse autoencoders (SAEs) have taken the interpretability world by storm over the past year or so. But can they be beaten?Yes!We introduce skip transcoders, and find they are a Pareto improvement ove...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1337546408424902759)** (45 messages🔥): 

> `SAE Visualization Tools, Distill Meetup Announcement, Learning about Generative Models, Older GPUs for AI Workloads, eGPU Setup on MacBook Air` 


- **SAE Visualization Tools Needed**: A user expressed a desire for a user interface built on **delphi/sparsify** libraries to make exploring **SAEs** more accessible.
   - Another member mentioned adapting existing SAE visualization libraries as a possible solution.
- **Join the Distill Meetup!**: A virtual **Distill meetup** is being organized for next Friday with a focus on discussing articles and visualizations in science communication.
   - Interested participants were encouraged to respond for an invite and access shared meeting notes.
- **How to Learn Generative Models**: A user sought advice on learning about **promptable diffusion models** and generative video models, hinting at the challenge of finding detailed textbooks.
   - Members suggested reading research papers, tinkering with sample code, and even checking LLMs to clarify doubts about confusing concepts.
- **Repurposing Older GPUs for AI**: Concerns about using **1070ti** mining rigs for AI highlighted issues with outdated architecture and bandwidth limitations.
   - Members noted the potential for such GPUs in inference but warned against their efficiency in training modern AI models.
- **eGPU Setup with MacBook Air**: A user inquired about using their **MacBook Air M2** with an eGPU for training ML models and the feasibility of learning CUDA.
   - Responses indicated that attempting to use an external GPU setup on a Mac may not be practical for ML applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=eL6tFQqNwd4LbYlO1DVIen8K">A Comprehensive Mechanistic Interpretability Explainer &amp; Glossary - Dynalist</a>: no description found</li><li><a href="https://docs.google.com/document/d/1VPtN1uIxWZGkXAlIB-UJEausDMcvcI8D9Ro8Rt9gDPM/edit?tab=t.0">Distill: Intro meet doc</a>: Before you go through the doc, I would like to share a quote I like &quot;You must be imaginative, strong-hearted. You must try things that may not work, and you must not let anyone define your limits...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1337684779767693322)** (49 messages🔥): 

> `Sparse Outlier Matrix, Transformer Architecture without FFNs, Model Interpretability, Policy Gradient Approaches, Self-Improving Intelligence` 


- **Understanding Sparse Outlier Matrix in FP4 Training**: The sparse outlier matrix compensates for quantization error caused by clamping, allowing high-precision sparse matrix multiplication of the residuals to preserve accuracy.
   - Clamping thresholds are set high (0.99 to 0.999), resulting in a sparse residual matrix with only 0.2% to 2% non-zero elements.
- **Exploring Transformers without Feed-Forward Networks**: A new transformer model proposes using persistent memory vectors in self-attention layers, eliminating the need for feed-forward networks while maintaining performance.
   - This architecture could facilitate teaching new knowledge to transformers without modifying all weights, potentially making learning updates more efficient.
- **Insights on Model Interpretability for Researchers**: Model interpretability is highlighted as a field requiring further foundational theory but with opportunities for applied experimentation.
   - Engaging in rapid testing could provide insights beneficial for beginner researchers moving into this area.
- **Discussion on Policy Gradient Algorithms in RL**: The disparity between policy gradient algorithms, which are often forward KL-oriented, and state-of-the-art continuous control methods that favor reverse KL, is noted.
   - This difference appears less impactful in discrete action environments, suggesting variations in approach aren't as critical.
- **Seeking Feedback for Self-Improving Intelligence Paper**: An author is seeking feedback and potential arXiv endorsement for a paper detailing a novel approach using recursive reasoning loops for AI self-improvement.
   - Community members shared insights about navigating the endorsement process and seeking solid reviews before publication.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.03275">Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning</a>: Large Language Models (LLMs) excel at reasoning and planning when trained on chainof-thought (CoT) data, where the step-by-step thought process is explicitly outlined by text tokens. However, this res...</li><li><a href="https://arxiv.org/abs/2502.04728">Generating Symbolic World Models via Test-time Scaling of Large Language Models</a>: Solving complex planning problems requires Large Language Models (LLMs) to explicitly model the state transition to avoid rule violations, comply with constraints, and ensure optimality-a task hindere...</li><li><a href="https://arxiv.org/abs/1907.01470">Augmenting Self-attention with Persistent Memory</a>: Transformer networks have lead to important progress in language modeling and machine translation. These models include two consecutive modules, a feed-forward layer and a self-attention layer. The la...</li><li><a href="https://arxiv.org/abs/2502.01591">Improving Transformer World Models for Data-Efficient RL</a>: We present an approach to model-based RL that achieves a new state of the art performance on the challenging Craftax-classic benchmark, an open-world 2D survival game that requires agents to exhibit a...</li><li><a href="https://arxiv.org/abs/2410.17897">Value Residual Learning For Alleviating Attention Concentration In Transformers</a>: Transformers can capture long-range dependencies using self-attention, allowing tokens to attend to all others directly. However, stacking multiple attention layers leads to attention concentration. O...</li><li><a href="https://openreview.net/forum?id=j3bKnEidtT">Temporal Difference Learning: Why It Can Be Fast and How It Will Be...</a>: Temporal difference (TD) learning represents a fascinating paradox: It is the prime example of a divergent algorithm that has not vanished after its instability was proven. On the contrary, TD...</li><li><a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>: We study a novel language model architecture that is capable of scaling test-time computation by implicitly reasoning in latent space. Our model works by iterating a recurrent block, thereby unrolling...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt">GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 3 minutes</a>: NanoGPT (124M) in 3 minutes. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1337897922485817445)** (18 messages🔥): 

> `Checkpointing Strategies, Pythia Checkpoints, Niche Training Tasks, Training Dynamics Analysis, Saving Checkpoints Without Interrupting Training` 


- **Exploring Checkpointing Strategies for LLMs**: A member proposed using an **exponential checkpointing strategy** initially (1, 2, 4, 8, 16) followed by fixed linear intervals, expressing curiosity about alternative approaches.
   - They suggested **1K or 5K steps** would be better for linear checkpoints compared to Pythia's **10K steps**.
- **Pythia's Checkpointing Methodology**: Discussion clarified that **Pythia saves checkpoints every 1,000 steps**, countering the idea that it uses **10K steps**.
   - Researchers opted for this spacing to allow deeper analysis using **log(tokens)** for interpretations.
- **Considerations for Checkpoint Resolutions**: There was mention of **resolution loss issues** just after 1,000 steps, referencing discussions from **Curt's paper** on circuit stability.
   - Members reflected on having released a high number of checkpoints, indicating uncertainty about what resolutions were most valuable.
- **Saving Checkpoints Without Interruption**: A member inquired about whether it's possible to save checkpoints without pausing training, potentially using a separate process.
   - While one member thought this might be possible, they could not recall the specific flag or details regarding this functionality.
- **Reflections on Early Checkpointing Decisions**: In retrospect, there was some consideration about having smaller linear step sizes and switching over earlier, to improve efficiency.
   - Conflicting information from the time also highlighted practical concerns, particularly regarding **wallclock overhead** for saving checkpoints.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1337858436721021010)** (7 messages): 

> `Evaluating LLMs with Chess Tactics, MCQ vs Free Form Generation for Tasks, Current Progress on Chess Task Implementation, Challenges with Generative Tasks, Tactics Database Management` 


- **Evaluating LLMs with Chess Tactics**: A member proposed creating a task to evaluate LLMs using a dataset of chess tactics, suggesting it as a unique approach to enhance LLM performance.
   - The goal is to eventually allow LLMs to play chess, leveraging **reinforcement learning** on positions with exact solutions.
- **Choosing Task Format: MCQ vs Free Form**: Discussion highlighted the potential task formats for evaluating LLMs, debating between **MCQ style** and free-form generation.
   - One view suggests avoiding MCQA for simplicity and prefers having models show their reasoning through **<think>** tags.
- **Current Progress on Chess Task Implementation**: The developer has made progress by creating an initial example that passes validity checks for the chess evaluation task.
   - However, they encounter a bug with generative tasks while using **mlx_lm** on macOS, hindering further development.
- **Challenges with Generative Tasks**: Despite testing a prompt on ChatGPT, which finds mate in 1s, the model struggled with more complex positions.
   - There are concerns about some models not formatting answers correctly using **<answer>** tags, which complicates evaluation.
- **Managing a Large Tactics Database**: The tactics database has grown to over **4M+ tactics**, and the developer seeks suggestions on effectively managing this size.
   - They reported that analyzing a **100-example subset** with a small 14B model takes about an hour on their machine.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1337759653253087322)** (1 messages): 

> `Self-aware AI concepts, Design of CARL` 


- **Introducing CARL, the Self-Aware AI Concept**: A member presented a design concept for a self-aware AI named **CARL**, complete with a visual representation shared in an image.
   - The attached image can be viewed [here](https://cdn.discordapp.com/attachments/795089627089862656/1337759652603101224/image_2.png?ex=67ab4043&is=67a9eec3&hm=d7eff5319301221f0797dcb6a86a045ccf81b7e6e0e3521ab342ae884070aeb6&)
- **Visual Representation of AI Concepts**: The shared image of **CARL** highlights a creative approach to depicting self-aware AI, showcasing its possible form and features.
   - Members expressed interest in exploring the implications of such designs on future AI development.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1337691573541928971)** (4 messages): 

> `VocabParallelEmbedding, use_flashattn_swiglu settings, RMSNorm vs RMSNorm Fusion, Asynchronous Checkpointing in NeoX, Torch Compile Speedups` 


- **VocabParallelEmbedding adjustment for weight decay**: A member found the part of the codebase where NeoX calls its embedding layer, called **VocabParallelEmbedding**, and added it to the weight decay ignore list.
   - They questioned whether this addition alone would suffice, especially since they are using tied embeddings.
- **Curiosity about use_flashattn_swiglu's impact**: A question arose regarding the **use_flashattn_swiglu** setting in the configuration, where a member experienced negligible impact.
   - They inquired if others found it helpful, along with questioning the usefulness of **rmsnorm_fusion** from Apex.
- **Inquiry on separate process for metric collection**: A member asked if there is a method to perform metric collection, logging, and model checkpointing on a different backend in NeoX, referencing [this blog](https://pytorch.org/blog/reducing-checkpointing-times/).
   - They related their query to OLMo2's paper, highlighting improvements in checkpointing times via new asynchronous methods.
- **Speedups with torch.compile in NeoX**: A member inquired about potential speedups using **torch.compile** in NeoX as they could not find any related flags.
   - This raises curiosity about optimizing model performance with available compilation techniques.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/reducing-checkpointing-times/">Reducing Model Checkpointing Times by Over 10x with PyTorch Distributed Asynchronous Checkpointing</a>: Summary:   With PyTorch distributed’s new asynchronous checkpointing feature, developed with feedback from IBM, we show how IBM Research Team is able to implement and reduce effective checkpointing ti...</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/utils.py#L45">gpt-neox/megatron/model/utils.py at main · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1337600713328693278)** (65 messages🔥🔥): 

> `Reinforcement Learning (RL), Logits vs Probabilities, Yannic Kilcher's Work, DeepSeek Models, Religious Discussions in Discord` 


- **Reinforcement Learning Insights**: Discussion centered around the nuances between RL training processes such as using direct logit optimization instead of transitioning to probability space too early.
   - It was noted that RL can be applied before, inside, or after a transformer, impacting how policies are learned and actions are selected.
- **Logits vs Probabilities in Training**: Participants debated the benefits of training models in log space compared to absolute space, emphasizing that log space can capture a wider range of values.
   - Zickzack highlighted that using log space can lead to more similarities in distant points and affect accuracy based on the use case.
- **Yannic Kilcher's Professional Role**: Mr. Yannic Kilcher, cofounder of DeepJudge, was discussed for his contributions and current projects in AI.
   - Participants inquired whether he is a full-time YouTube creator or primarily focused on his startup.
- **DeepSeek Research Discussion**: Users shared their experiences and opinions regarding the DeepSeek r1 model, noting its efficiency in certain applications.
   - Questions arose regarding the availability of research papers related to DeepSeek developments and comparisons to other models.
- **Religious Discussions Spark Debate**: There were mixed reactions to discussions about religious texts, with some participants advocating for reading the Quran for guidance.
   - This led to light-hearted comments and jokes about people's views on religion, showing a spectrum of opinions on the topic.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/JaechulRoh/status/1887958947090587927/history">Tweet from Jaechul Roh (@JaechulRoh)</a>: 🧠💸 &#34;We made reasoning models overthink — and it&#39;s costing them big time.&#34;Meet 🤯 #OVERTHINK 🤯 — our new attack that forces reasoning LLMs to &#34;overthink,&#34; slowing models like Ope...</li><li><a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2">AI Mathematical Olympiad - Progress Prize 2</a>: Solve national-level math challenges using artificial intelligence models</li><li><a href="https://www.deepjudge.ai/">DeepJudge - Collective Knowledge Revived</a>: no description found</li><li><a href="https://arxiv.org/abs/2407.00626">Maximum Entropy Inverse Reinforcement Learning of Diffusion Models with Energy-Based Models</a>: We present a maximum entropy inverse reinforcement learning (IRL) approach for improving the sample quality of diffusion generative models, especially when the number of generation time steps is small...</li><li><a href="https://arxiv.org/abs/2304.12824">Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning</a>: Guided sampling is a vital approach for applying diffusion models in real-world tasks that embeds human-defined guidance during the sampling procedure. This paper considers a general setting where the...</li><li><a href="https://arxiv.org/abs/2312.03397">Generalized Contrastive Divergence: Joint Training of Energy-Based Model and Diffusion Model through Inverse Reinforcement Learning</a>: We present Generalized Contrastive Divergence (GCD), a novel objective function for training an energy-based model (EBM) and a sampler simultaneously. GCD generalizes Contrastive Divergence (Hinton, 2...</li><li><a href="https://arxiv.org/abs/2406.16121">Diffusion Spectral Representation for Reinforcement Learning</a>: Diffusion-based models have achieved notable empirical successes in reinforcement learning (RL) due to their expressiveness in modeling complex distributions. Despite existing methods being promising,...</li><li><a href="https://arxiv.org/abs/2302.11552">Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC</a>: Since their introduction, diffusion models have quickly become the prevailing approach to generative modeling in many domains. They can be interpreted as learning the gradients of a time-varying seque...</li><li><a href="https://arxiv.org/abs/2410.01312">Sampling from Energy-based Policies using Diffusion</a>: Energy-based policies offer a flexible framework for modeling complex, multimodal behaviors in reinforcement learning (RL). In maximum entropy RL, the optimal policy is a Boltzmann distribution derive...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1337538180160950293)** (41 messages🔥): 

> `RSS Feeds for ML/DL, Sparse Autoencoders Research, AI Oversight and Model Similarity, Hugging Face Daily Papers, PhD Paper Assistant Tool` 


- **Discussion on Useful RSS Feeds for ML/DL**: Members discussed the relevance of RSS feeds for tracking ML/DL research, with comments suggesting alternatives like **latent.space** and the **Hugging Face papers** site.
   - One user indicated that RSS is outdated, while another mentioned using GitHub to filter papers by keywords.
- **Debate on Sparse Autoencoders**: A member expressed skepticism about **Sparse Autoencoders** (SAEs) being overhyped, stating they expected more interpretability but encounter inconsistencies across random seeds in results.
   - Discussion included insights from recent papers that critique SAEs and explore new methods for model interpretation.
- **Exploring AI Oversight Through Model Similarity**: A paper highlighted the challenges of evaluating advanced language models and proposed a probabilistic metric for assessing model similarity based on mistake overlap.
   - Concerns were raised about increasing systemic mistakes in advanced models, with conversations around the implications for AI oversight.
- **Utilizing Hugging Face Daily Papers**: One user shared a resource to subscribe to daily paper updates from **Hugging Face**, suggesting it allows tracking trending ML papers efficiently.
   - Members appreciated the idea of filtering daily papers by keywords and highlighted the importance of a ranking system for better organization.
- **PhD Paper Assistant Tool Launch**: A new tool called **PhD Paper Assistant** aims to help students navigate complex research papers filled with ML jargon by filtering content based on keywords.
   - The tool also allows users to sort and pin preferred papers, enhancing the research experience for PhD students.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.04313">Great Models Think Alike and this Undermines AI Oversight</a>: As Language Model (LM) capabilities advance, evaluating and supervising them at scale is getting harder for humans. There is hope that other language models can automate both these tasks, which we ref...</li><li><a href="https://model-similarity.github.io/">Great Models Think Alike and this Undermines AI Oversight</a>: Model similarity has negative effects on using LMs to judge or train other models; Unfortunately LMs are getting similar with increasing capabilities.</li><li><a href="https://arxiv.org/abs/2501.16615">Sparse Autoencoders Trained on the Same Data Learn Different Features</a>: Sparse autoencoders (SAEs) are a useful tool for uncovering human-interpretable features in the activations of large language models (LLMs). While some expect SAEs to find the true underlying features...</li><li><a href="https://arxiv.org/abs/2501.17727">Sparse Autoencoders Can Interpret Randomly Initialized Transformers</a>: Sparse autoencoders (SAEs) are an increasingly popular technique for interpreting the internal representations of transformers. In this paper, we apply SAEs to &#39;interpret&#39; random transformers,...</li><li><a href="https://arxiv.org/abs/2501.17148">AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders</a>: Fine-grained steering of language model outputs is essential for safety and reliability. Prompting and finetuning are widely used to achieve these goals, but interpretability researchers have proposed...</li><li><a href="https://x.com/norabelrose/status/1887972442104316302">Tweet from Nora Belrose (@norabelrose)</a>: Sparse autoencoders (SAEs) have taken the interpretability world by storm over the past year or so. But can they be beaten?Yes!We introduce skip transcoders, and find they are a Pareto improvement ove...</li><li><a href="https://x.com/janleike/status/1888616860020842876">Tweet from Jan Leike (@janleike)</a>: After ~300,000 messages and an estimated ~3,700 collective hours, someone broke through all 8 levels.However, a universal jailbreak has yet to be found...Quoting Jan Leike (@janleike) We challenge you...</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>: no description found</li><li><a href="https://github.com/SmokeShine/phd-paper-assistant">GitHub - SmokeShine/phd-paper-assistant: PhD Paper Assistant is a web-based tool designed to help PhD students navigate and understand complex research papers, particularly those filled with machine learning (ML) jargon.</a>: PhD Paper Assistant is a web-based tool designed to help PhD students navigate and understand complex research papers, particularly those filled with machine learning (ML) jargon.  - GitHub - Smoke...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1337536440120250409)** (6 messages): 

> `Reinforcement Learning in AI Agents, PlanExe AI Project, LLMs and Token Counting Limitations` 


- **Questioning RL Definitions in AI Agents**: One member explored whether using **VectorDB** to store lessons as embeddings constitutes true **Reinforcement Learning** (RL), questioning if genuine RL can be implemented without fine-tuning hosted LLMs.
   - They seek insights on simulating RL-like behavior and also inquired about relevant papers on **agentic frameworks** with RL implementations.
- **Introducing PlanExe: A Structured AI Planner**: Another member presented their project, **PlanExe**, created with **LlamaIndex** and **OpenRouter**, capable of generating structured plans like SWOT analyses without deep web searching.
   - They shared a [GitHub link](https://github.com/neoneye/PlanExe) to the project, expressing uncertainty about the accuracy of the outputs it generates.
- **LLMs Struggling with Token Count**: A member pointed out that LLMs have difficulty counting tokens in their context, indicating a broader issue with tokenization not being the only problem in counting characters in words.
   - This was further emphasized by another user who remarked that **LLMs can't count at all**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/neoneye/PlanExe">GitHub - neoneye/PlanExe: AI planner similar to OpenAI&#39;s deep research</a>: AI planner similar to OpenAI&#39;s deep research. Contribute to neoneye/PlanExe development by creating an account on GitHub.</li><li><a href="https://neoneye.github.io/PlanExe-web/">PlanExe-web</a>: Website for PlanExe
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1337602872212455424)** (8 messages🔥): 

> `ELEGNT Video Discussion, Guardrails in Drug Discovery Algorithms, Guardrails for Bioweapons Discovery, Anthropic's Information Output` 


- **ELEGNT Video on Robot Movement**: A YouTube video titled **ELEGNT: Expressive and Functional Movement Design for Non-Anthropomorphic Robot** was shared [here](https://youtu.be/IHJa_CyJjdc?feature=shared). The video is currently undefined in description but relates to movement design.
   - The discussion around this video emphasizes innovative movement techniques applied to robot design.
- **Algorithm Shifts in Drug Discovery**: There's concern that a drug discovery algorithm switched from **toxicity minimizing** to **maximizing**, resulting in discovering **40,000** potential bioweapons in just **6 hours**. The implication is that guardrails are ineffective against broader knowledge synthesis.
   - It raises questions about the algorithm's focus on specific topics while potentially neglecting more harmful compounds.
- **Guardrails' Impact on Bioweapons Discovery**: The possibility exists that guardrails intended for one specific **nerve gas** could lead to overlooking numerous harmful compounds. There's an awareness that discovering other bioweapons merely requires applying the same methodology to millions of compounds.
   - Critiques suggest that by narrowing their focus, creators might inadvertently create blind spots in their safety measures.
- **Anthropic's Information Handling Critique**: A member criticized that Anthropic's outputs often offer **partial information**, creating a false sense of security. Even when outputs contain valuable pieces of information, the perception of safety lagged behind the actual effectiveness.
   - This highlights a fundamental gap between advertised safety measures and real-world applications while addressing complex concerns.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/lefthanddraft/status/1888706065514291290?s=46">Tweet from Wyatt walls (@lefthanddraft)</a>: 7 down, 1 to goQuoting Wyatt walls (@lefthanddraft) 6 down, 2 to go</li><li><a href="https://aidemos.meta.com/">AI Demos | Meta FAIR</a>: Try experimental demos featuring the latest AI research from Meta.</li><li><a href="https://youtu.be/IHJa_CyJjdc?feature=shared">ELEGNT: Expressive and Functional Movement Design for Non-Anthropomorphic Robot</a>: no description found</li><li><a href="https://tenor.com/view/pixx-pixar-lamp-pixar-gif-14006253">Pixx Pixar GIF - Pixx Pixar Lamp Pixar - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1337539933719760919)** (5 messages): 

> `Gemini Flash 2.0, AI in Enterprise Automation, CrossPoster App Launch, GraphRAG Pipelines` 


- **Gemini 2.0 Flash revolutionizes document processing**: LlamaParse now supports **Gemini 2.0 Flash**, providing **GPT-4o+ performance** at a fraction of the cost for document processing.
   - The future of workflows is poised to leverage **VLMs and LLMs**, according to recent discussions.
- **YouTube Research Agent built with Gemini Flash 2.0**: A tutorial introduced by @composiohq details how to build a **YouTube research agent** using **Gemini Flash 2.0**, enabling robust video searches and draft creations in Gmail.
   - This integration positions @llama_index as a fundamental tool in simplifying video research workflows.
- **AI's role in enterprise automation**: An article emphasized that enterprises should focus on **adapting AI technology** to automate knowledge work and resolve business challenges effectively.
   - Cobus Greyling proposes that this should become a primary goal for businesses by **2025**.
- **Launch of CrossPoster for social media**: Today marks the launch of **CrossPoster**, an app designed to cross-post to **Twitter**, **LinkedIn**, and **BlueSky** using AI for optimal social media engagement.
   - The app intelligently identifies individuals and their accounts, streamlining the process of managing social presence across platforms.
- **Utilizing GraphRAG pipelines for data insights**: GraphRAG pipelines are highlighted for their ability to transform **raw data** into actionable insights through knowledge graph creation.
   - This approach enhances **LLM accuracy** with domain-specific knowledge, allowing for more comprehensive search capabilities.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1337593289913929790)** (111 messages🔥🔥): 

> `Timeout Issues with OpenAI LLM, Agent Hand-off Mechanism in LlamaIndex, Gemini Function Calling Challenges, LlamaIndex and RAG Implementation, AzureAI Search Custom Metadata Fields` 


- **Timeout Issues with OpenAI LLM**: Members discussed how the timeout for OpenAI LLM options is clobbered by the retry_decorator, causing inconsistencies despite higher timeout settings.
   - A member mentioned that even on submission of a bug fix, Deepseek returns a 200 OK response after 60 seconds but with an empty body, making the issue complicated.
- **Agent Hand-off Mechanism in LlamaIndex**: Concerns were raised regarding the efficacy of the `can_handoff_to` feature in LlamaIndex, specifically when agents pass control without a response from the receiving agent.
   - Suggestions to enable debug logging and utilize LlamaIndex's callback handler were offered as troubleshooting steps.
- **Gemini Function Calling Challenges**: Forum members expressed frustration over the difficulty of debugging function calls in Gemini, citing issues with type annotations and unclear error messages.
   - Despite the frustrations, some were able to engineer their tool outputs around the existing bugs.
- **LlamaIndex and RAG Implementation**: Members discussed strategies for handling user queries that require entire document context in RAG settings, like identifying summary and vector indices.
   - Utilizing agents or query classification logic was recommended for better management of retrieval based on specific query needs.
- **AzureAI Search Custom Metadata Fields**: A question arose regarding the hardcoded customization of filterable metadata fields in AzureAI Search, with particular fields such as 'author' and 'director' noted.
   - Members noted that Azure requires these metadata fields to be defined upfront, which can be limiting but emphasizes the importance of useful document fields.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_deploy">GitHub - run-llama/llama_deploy: Deploy your agentic worfklows to production</a>: Deploy your agentic worfklows to production. Contribute to run-llama/llama_deploy development by creating an account on GitHub.</li><li><a href="https://github.com/anthropics/anthropic-cookbook/blob/main/misc/using_citations.ipynb">anthropic-cookbook/misc/using_citations.ipynb at main · anthropics/anthropic-cookbook</a>: A collection of notebooks/recipes showcasing some fun and effective ways of using Claude. - anthropics/anthropic-cookbook</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial (OpenAI) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/rag/">Introduction to RAG - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/pull/17764">fix gemini multi-turn tool calling by logan-markewich · Pull Request #17764 · run-llama/llama_index</a>: There were two issuesthe original tool call was not being included in the messages when converting from llama-index to gemini messagesstreaming tool calls that mix text and tool calls were clobb...</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/tracing_and_debugging/tracing_and_debugging/#tracing-and-debugging>)">Tracing and Debugging - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/issues/17756)">run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/DocumentContextExtractor/">Contextual Retrieval With Llama Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/cookbooks/contextual_retrieval/">Contextual Retrieval - LlamaIndex</a>: no description found</li><li><a href="https://github.com/openai/openai-python/blob/7193688e364bd726594fe369032e813ced1bdfe2/src/openai/_client.py#L82">openai-python/src/openai/_client.py at 7193688e364bd726594fe369032e813ced1bdfe2 · openai/openai-python</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://github.com/openai/openai-python/blob/7193688e364bd726594fe369032e813ced1bdfe2/src/openai/_response.py#L265),">openai-python/src/openai/_response.py at 7193688e364bd726594fe369032e813ced1bdfe2 · openai/openai-python</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/agent/workflow/#llama_index.core.agent.workflow.AgentWorkflow>),">Workflow - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/agent/multi_agents/#a-detailed-look-at-the-workflow>).">Multi-agent workflows - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py#L404">llama_index/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py at 7391f302e18542c68b9cf5025afb510af4a52324 · run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/azureaisearch/">Azureaisearch - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/5b0067e146c919bc804803892aa2456842a80346/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py#L93)">llama_index/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/base.py at 5b0067e146c919bc804803892aa2456842a80346 · run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/)** (1 messages): 

mrmirro: 💯
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1337540139349442590)** (14 messages🔥): 

> `Job Application Advice, Engineering Internships, Networking, Open Source Contribution, Canadian Engineering Competition` 


- **Trust Yourself in Job Applications**: Emphasizing self-belief, one member encouraged others to 'trust in yourself regardless of what they say' during job applications.
   - Another added that *everyone is just as uncertain*, pushing for persistence in the face of challenges.
- **Challenges of Finding Internships**: Members discussed the current *lack of hiring opportunities* for engineering internships, particularly in the local area.
   - One user shared their experience of competing in the **Canadian engineering competition**, ranking in the top 6 for junior design.
- **Networking is Key**: A member emphasized that *networking* is crucial, regardless of one’s location, suggesting participation in events to boost exposure.
   - Engaging in open-source projects was also recommended as a way to connect with others in the field.
- **Competing for Exposure**: In an effort to gain experience, one user mentioned attending *conferences and competitions* relevant to their engineering field.
   - They highlighted their participation in the **Canadian engineering competition**, reflecting their commitment to personal development.
- **Coding for Fun**: A member humorously stated they enjoy coding just for fun, illustrating a lighter side to the otherwise serious job seeking discussions.
   - This reflected a balance between pursuing professional goals while engaging in activities that bring joy.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1337537511802540085)** (8 messages🔥): 

> `LibreChat Endpoint Issues, Curl Testing, Cohere API Versioning` 


- **LibreChat struggles with Cohere API**: A member highlighted that they can only access the **Cohere API** through `https://api.cohere.ai/v1` using LibreChat's Custom Endpoint.
   - *CURL worked*, indicating the issue lies within LibreChat's integration with the API.
- **Curl testing reveals API accessibility**: Another member suggested testing the Cohere API with **curl**, which confirmed that it works correctly.
   - *If curl works*, it's likely that the issue is specific to LibreChat, leading to an encouraged issue report on their GitHub.
- **LibreChat using outdated Cohere API**: It was pointed out that **LibreChat** is currently calling the old API version (v1) and needs an update to the `/v2` endpoint.
   - The URL **https://api.cohere.com/v1** mirrors the functionality of `https://api.cohere.ai/v1`, providing a potential solution for current users.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1337869971556012132)** (59 messages🔥🔥): 

> `Cohere Community Rules, Introduction Messages, AI Reasoning and Scalability, Working with Cohere staff, Discussion about Vapes` 


- **Exploring Cohere Community Rules**: Members discussed the Cohere Community rules, emphasizing respect and appropriate conduct within the server.
   - Questions arose about how these rules apply to conversations and what should be considered while engaging in discussions.
- **Crafting Introduction Messages**: Users collaborated on drafting engaging introduction messages for newcomers, highlighting interests in AI and local initiatives like 'buy Canadian'.
   - An example introduction emphasized the desire to explore AI's potential and engage meaningfully with the community.
- **AI Reasoning and Scalability of Cohere**: The discussion shifted to the scalability of Cohere's API and how accessible their staff is for collaboration.
   - Members expressed interest in understanding how Cohere supports businesses in leveraging AI for their products.
- **Inquiry About Vapes**: A member encouraged a Socratic dialogue about vapes, prompting an exploration of what vapes represent in contemporary society.
   - This led to a humorous exchange where Socrates expressed his unfamiliarity with vapes, inviting education on the matter.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1338632280096374805)** (1 messages): 

> `Lecture by Yu Su, Language Agents, MOOC curriculum details` 


- **Yu Su's Lecture on Language Agents Today at 4pm PST**: Today at **4:00pm PST**, join the livestream of the 3rd lecture featuring **Yu Su** presenting on *Memory, Reasoning, and Planning of Language Agents* [here](https://www.youtube.com/live/zvI4UN2_i-w).
   - Yu Su argues that contemporary AI agents differ by utilizing **language as a vehicle for reasoning** and communication, presenting a conceptual framework and exploring their core competencies.
- **Yu Su's Contributions to NLP**: Yu Su is a **Distinguished Assistant Professor** at the Ohio State University and co-directs the NLP group with significant contributions including Mind2Web, SeeAct, HippoRAG, LLM-Planner, and MMMU.
   - His work has garnered recognition like the **Best Student Paper Award** at CVPR 2024 and **Outstanding Paper Award** at ACL 2023.
- **Upcoming MOOC Curriculum Details**: An announcement stated that **MOOC curriculum details** will be released soon, and thanked everyone for their patience.
   - Details regarding the curriculum remain pending, encouraging participants to stay tuned.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1337522928052338759)** (50 messages🔥): 

> `Course Registration, Certificate Issues, Project Collaboration, MOOC Curriculum, Research Track Registration` 


- **Course Registration for Late Enrollers**: Users inquired whether they could enroll in the **LLM Agents MOOC** that started in January, with confirmations that late registration is possible by completing the signup form.
   - Participants were clarified that earlier course versions are not strict prerequisites for later iterations.
- **Concerns Over Missing Certificates**: Several users reported not receiving their certificates while their peers have, prompting a focus on missing completed **certificate declaration forms** as a required step.
   - The course staff reiterated that completion of this form is necessary for certificate issuance and needs to be submitted individually.
- **Project Collaboration Inquiry**: One user expressed interest in collaborating on a project while ensuring compliance with course guidelines regarding publication rights as a MOOC student.
   - Course staff promised to release more curriculum details soon, addressing concerns about project framework and publication limitations.
- **MOOC Curriculum and Requirements Update**: Participants asked about the specifics of assignments and projects outside of quizzes, to which staff mentioned detailed information would be released shortly.
   - Users were encouraged to remain patient while awaiting clear guidelines on project requirements and grading policies.
- **Research Track Registration Inquiry**: Users sought clarity on how to register for the **research track**, indicating a need for guidance on the appropriate Google form.
   - Additional suggestions included creating an automated agent to streamline the certificate process and address common queries.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://]">no title found</a>: no description found</li><li><a href="https://shamik-07.github.io/compound_ai_agentic_system/">🖥️ Compound AI System using LLMs 🖥️</a>: A Compound AI System performing multiple complex tasks</li><li><a href="https://x.com/Tonikprofik/status/1868729038665392472">Tweet from Tony T (@Tonikprofik)</a>: 🚀As part of my research for Master thesis. I enrolled in  MOOC of UC Berkeley on Large Language Model (LLM) Agents, offered in Fall 2024🧠✨This course deep dived into LLM agents&#39; applications. He...
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1338119309310034052)** (12 messages🔥): 

> `SFT vs DPO, Importance of Negative Examples, LLM Training Challenges, Lecture 2 Study Session, Time Zone Discussions` 


- **SFT and DPO explain training paradigms**: A member explained how **Supervised Fine Tuning (SFT)** uses only positive examples while **Direct Preference Optimization (DPO)** incorporates negative responses, highlighting the penalties for bad responses in DPO.
   - *Bad responses*, often well-structured, trigger an increase in their probability during SFT due to the absence of a reward model.
- **Challenges of model responses under altered instructions**: Discussion focused on a slide indicating the expectation that a modified instruction **x'** would lead to a worse response **y'**, emphasizing the challenges of generating responses that are relevant yet semantically different.
   - The model is tasked with producing accurate yet deficient responses while adhering to the modified prompt's requirements, showcasing a tough balancing act.
- **Study session on Lecture 2 announced**: A member announced a study session on **Lecture 2: Learning to Reason with LLMs**, inviting others to join via a provided link.
   - Participants were encouraged to prepare for discussing **GRPO from DeepSeek-R1** as part of the study materials.
- **Concerns about timing for study session**: One participant expressed concern about the study session's timing, noting that it fell at **3:00 AM UK time**.
   - This highlighted potential scheduling conflicts for international members.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1338327366186831897)** (13 messages🔥): 

> `Artificial Data Generation, Kolo Fine Tuning Tool, Challenges in Synthetic Data Creation, Public Roadmap Update` 


- **Artificial Data Generation Exploration**: A member is venturing into **artificial data generation** and seeks tools for converting unstructured data, such as PDFs and Excel files, into training samples for LLMs.
   - They shared a [YouTube video](https://youtu.be/iogrvDu5K0k?si=U9fi5C-0UvytTmBO) on synthetic data generation methodologies relevant to this field.
- **Kolo Tool for Fine Tuning**: A member is developing a tool called **Kolo** aimed at simplifying the fine-tuning process for models, though it currently doesn't assist in data creation.
   - The creator is working on incorporating a feature to help generate training data in the future.
- **Challenges of Synthetic Data Generation**: Discussions highlighted the complexity of training LLMs with synthetic data, noting that generating questions from individual documents may not cover necessary comparative insights.
   - A member expressed that in-depth queries require comprehensive training data across multiple document sources to ensure effective learning.
- **Feedback on Roadmap Availability**: A reminder was issued regarding the public roadmap discussed previously, with a member inquiring about a draft version.
   - It was confirmed that the roadmap is in the approval process and should be shared on GitHub by the end of the week once finalized.



**Link mentioned**: <a href="https://youtu.be/iogrvDu5K0k?si=U9fi5C-0UvytTmBO">Synthetic Data Generation and Fine tuning (OpenAI GPT4o or Llama 3)</a>: ➡️ Get Life-time Access to the Complete Scripts (and future improvements): https://Trelis.com/ADVANCED-fine-tuning➡️ One-click fine-tuning and LLM templates:...

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1337628189341651078)** (43 messages🔥): 

> `PR #2257 Review, GRPO Development Philosophies, Checkpointing Methods, PyTorch Dependency Management, Support for UV and Pip` 


- **PR #2257 Needs Extra Eyes**: A member shared a [PR #2257](https://github.com/pytorch/torchtune/pull/2257) for review, noting it's working in their local tests but seeking additional feedback.
   - Another member reviewed it, praising the changes but mentioning UX concerns with quantization and suggesting documentation updates.
- **Two Philosophies on GRPO Features**: The discussion centered around whether to keep or remove various functionalities in GRPO for simplification, balancing between ease of use and code cleanliness.
   - Several members expressed opinions, leaning towards removing unnecessary code while noting the potential need for specific features like activation checkpointing.
- **Understanding Checkpointing Mechanics**: Details were shared about how resume functionality works in torchtune, emphasizing updates to checkpoint paths and the importance of the resume_from_checkpoint flag.
   - Members discussed the implications of checkpointing practices, including an unusual workflow regarding loading initial weights.
- **Managing PyTorch Dependencies**: A member proposed adding install options for different versions of dependencies, highlighting potential complications with nightly versions and standard pip support.
   - The discussion included considerations for supporting both pip and uv, weighing the benefits and drawbacks of extending the pyproject.toml.
- **Support for UV Users**: There was an acknowledgment of uv's growing popularity among users, with suggestions to implement support alongside traditional pip approaches.
   - Emphasis was placed on prioritizing pip while being open to well-tested additions for uv due to its utility in user workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://packaging.python.org/en/latest/specifications/dependency-groups/#dependency-groups">Dependency Groups - Python Packaging User Guide</a>: no description found</li><li><a href="https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index">Using uv with PyTorch | uv</a>: no description found</li><li><a href="https://pytorch.org/torchtune/main/deep_dives/checkpointer.html#resuming-from-checkpoint-full-finetuning">Checkpointing in torchtune &mdash; torchtune main documentation</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/pull/2363.">Build software better, together</a>: GitHub is where people build software. More than 150 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/pytorch/torchtune/issues/2375">pyproject.toml wrong dev deps organization · Issue #2375 · pytorch/torchtune</a>: torchtune has dev dependencies defined at [project.optional-dependencies] - https://github.com/pytorch/torchtune/blob/main/pyproject.toml#L47 while they should be defined at [dependency-groups] acc...</li><li><a href="https://github.com/pytorch/torchtune/blob/9da35c744adef777ec9b8d8620337ae5f0371dd5/recipes/configs/mistral/7B_full_ppo_low_memory.yaml#L78">torchtune/recipes/configs/mistral/7B_full_ppo_low_memory.yaml at 9da35c744adef777ec9b8d8620337ae5f0371dd5 · pytorch/torchtune</a>: PyTorch native post-training library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/1452">Removing ao from pyproject.toml by ebsmothers · Pull Request #1452 · pytorch/torchtune</a>: TLDR: We have to choose between our ability to consistently provide stable, well-tested nightly packages and a clean install experience for all of our users. This PR reluctantly proposes to sacrifi...</li><li><a href="https://github.com/pytorch/torchtune/pull/2349">Rework recipes section of README and simplify models ref by joecummings · Pull Request #2349 · pytorch/torchtune</a>: no description found</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/553/files#diff-1534093f54f1b158be2da2b159e45561361e2479a7112e082232d3f21adc6a45">Grpo loss by kashif · Pull Request #553 · linkedin/Liger-Kernel</a>: SummaryAdds the GRPO chunked lossfixes issue #548Testing DoneHardware Type:  run make test to ensure correctness run make checkstyle to ensure code style run make test-convergence to ensu...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/548">[RFC] Liger FlexChunkLoss: Grouping Loss · Issue #548 · linkedin/Liger-Kernel</a>: 🚀 The feature, motivation and pitch If I can assume that many research efforts like Group Relative Policy Optimization (GRPO) will emerge, I think we could introduce a LigerFusedLinearGroupingBase .....
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1337515018660483106)** (45 messages🔥): 

> `Model Selection Menu, AI Agents and Memory, Image Analysis in GPT4All, PDF Processing and Embedding, Long-term Memory Solutions` 


- **Critique of Model Selection Menu**: Concerns were raised about the lack of a functional model selection menu with search options in GPT4All after 36 releases, suggesting it might just require copy-pasting from other platforms.
   - One member proposed contributing code to address missing features since GPT4All is an open-source product.
- **Exploring AI Agents for Long-Term Memory**: Members discussed the potential of AI agents that utilize databases for long-term memory, with suggestions to enhance LLMs' temporal awareness through functions.
   - The year 2025 was mentioned as a possible turning point for agentic AI advancements.
- **Limitations of Image Analysis**: It was clarified that, currently, GPT4All does not support image analysis, with suggestions to explore alternative platforms for such capabilities.
   - Recommendations for tools like booruDatasetTagmanager and joycaption were provided for users interested in image-related tasks.
- **Best Practices for PDF Processing**: Members discussed effective strategies for embedding and summarizing long documents, such as PDFs, into usable formats for GPT4All.
   - The need to handle downloads from browsers properly was emphasized to ensure the elimination of irrelevant content before embedding.
- **Choosing the Right Model**: When asked about model performance, Qwen2.5 and Phi4 were recommended for their efficiency over other models, including Mistral, based on member experience.
   - The importance of selecting models integrated with the app for user-friendliness was highlighted, along with a willingness to help those unfamiliar with downloading from Hugging Face.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.gpt4all.io/gpt4all_api_server/home.html#key-features)">GPT4All API Server - GPT4All</a>: GPT4All Docs - run LLMs efficiently on your hardware</li><li><a href="https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683">Cheat Sheet: Mastering Temperature and Top_p in ChatGPT API</a>: Hello everyone!  Ok, I admit had help from OpenAi with this. But what I “helped” put together I think can greatly improve the results and costs of using OpenAi within your apps and plugins, specially ...</li><li><a href="https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-bindings/typescript/spec/chat-memory.mjs#L1"">gpt4all/gpt4all-bindings/typescript/spec/chat-memory.mjs at main · nomic-ai/gpt4all</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all</li><li><a href="https://huggingface.co/QuantFactory/gpt2-large-GGUF/tree/main">QuantFactory/gpt2-large-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/main">TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF at main</a>: no description found</li><li><a href="https://github.com/JabRef/jabref/wiki/GSoC-2024-%E2%80%90-AI%E2%80%90Powered-Summarization-and-%E2%80%9CInteraction%E2%80%9D-with-Academic-Papers)">Home</a>: Graphical Java application for managing BibTeX and biblatex (.bib) databases - JabRef/jabref</li><li><a href="https://docs.jabref.org/ai/local-llm)),">JabRef</a>: no description found
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1337521236921684014)** (21 messages🔥): 

> `tinygrad testing and features, web-based LLM demos, tinygrad community discussions, company meeting agenda, ML frameworks and fp16 concerns` 


- **Testing tinygrad on mobile devices**: Discussion emerged around the performance of **tinychat** demos on mobile, highlighting that **WebGPU** fails on iPhone 15 due to caching issues while M1 Pro users find it works well in both Safari and Chrome.
   - Users expressed a need for further testing to improve compatibility, particularly regarding **WASM** loading on mobile devices.
- **Tinygrad company structure clarification**: A user mistakenly thought **tinygrad** is based in San Diego due to Twitter information, but was corrected that it is a **fully remote company**.
   - This led to questions about the **Ampere Altra** processor support and backend acceleration capabilities for tinygrad.
- **Company Meeting #57 Topics Announced**: Meeting #57 scheduled for Monday includes topics like **company updates**, **CI speed**, **tensor cores**, and discussion on potential **bounties** related to **WebGPU** and **tinychat**.
   - Such meetings aim to enhance the operational speed of internal processes while addressing community interests in ongoing projects.
- **Exploring fp16 in ML frameworks**: A user questioned why most ML frameworks don't operate solely in **fp16**, prompting active discussion on its potential disadvantages and performance limitations.
   - George responded to the inquiry with a directive to review discord rules, sparking further commentary on research quality prior to inquiries.
- **PR clarity and numerical accuracy**: Discussion unfolded around a pull request (PR) that implements a script but requires further features and testing for **Hugging Face models**.
   - The community emphasized the importance of clean PR structure for easy reviews while acknowledging existing **numerical inaccuracies** in quantized models as a challenge.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chat.webllm.ai/">WebLLM Chat</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/actions/runs/13207080963/job/36872675131">Make tensor use UOp lshift/rshift; delete SimpleMathTrait · tinygrad/tinygrad@caafb50</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Make tensor use UOp lshift/rshift; delete SimpleMathTrait · tinygrad/tinygrad@caafb50
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1338272144655650887)** (1 messages): 

> `DSPy, BERT model training, Mistral architecture, Automated article processing` 


- **DSPy Revolutionizes Article Classification**: After struggling with **GPT-3.5** and the high cost of **GPT-4**, a member transitioned to training a **BERT** based model to classify incoming articles effectively.
   - Today marks a significant milestone with a highly optimized prompt that extracts a dozen fields from each article using **DSPy**, significantly enhancing performance.
- **Mistral Models Shine in Cost Efficiency**: The member utilized **Miprov2** with **o3-mini** as a teacher and **Mistral Small 3** as a student, creating a fully automated process that is both cheap and efficient.
   - This setup enables batch processing of articles every 24 hours, achieving results that exceeded expectations with a **50% discount**.
- **From Bumpy Beginnings to Streamlined Workflow**: Two years ago, the member faced hurdles with manual data classification, but now sees a steady flow of **100-200 articles daily** processed effortlessly.
   - The initial BERT model setup laid the groundwork for today's automated solution, showcasing tremendous growth in capability and efficiency.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1337906100560855100)** (2 messages): 

> `Multi-Agent Systems, AI Agents and System Engineering, MASS Optimization Framework, Automation-oriented Sandbox Games` 


- **Multi-Agent Systems excel at complex tasks**: Large language models operating as multiple agents can excel at solving complex tasks due to effective interaction and collaboration programs, as detailed in the [MASS framework](https://arxiv.org/abs/2502.02533).
   - The analysis emphasizes that effective **prompts** and **topologies** are critical for designing robust multi-agent systems.
- **AI Agents need dynamic system engineering skills**: The static benchmarks used for evaluating AI agents fail to reflect necessary skills for dynamic system engineering, advocating for agents trained via automation-oriented sandbox games like [Factorio](https://arxiv.org/abs/2502.01492).
   - This approach aims to foster the development of specialized reasoning and long-horizon planning capabilities essential for managing complex engineering challenges.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.01492">Develop AI Agents for System Engineering in Factorio</a>: Continuing advances in frontier model research are paving the way for widespread deployment of AI agents. Meanwhile, global interest in building large, complex systems in software, manufacturing, ener...</li><li><a href="https://arxiv.org/abs/2502.02533">Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies</a>: Large language models, employed as multiple agents that interact and collaborate with each other, have excelled at solving complex tasks. The agents are programmed with prompts that declare their func...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1337552896589299762)** (3 messages): 

> `Deep Research Abstractions, dspy Error Handling` 


- **Inquiry on Simplifying Deep Research Tasks**: A member inquired about plans to introduce abstractions that simplify tasks akin to **deep research**, noting that the necessary components might already be available.
   - *Are you guys planning to introduce abstractions?* highlights a curiosity about potential upcoming features.
- **AttributeError with dspy**: One member reported encountering the error `AttributeError: module 'dspy' has no attribute 'HFClientVLLM'` while using **dspy**.
   - After some investigation, they noted that this feature was **deprecated** in **dspy 2.6**, resolving their confusion.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1337534354296934562)** (6 messages): 

> `RAFT templates for Llama, Compatibility issues with HF datasets, Converting complex objects to strings, Updating README with helper function, JSON lines formatted files` 


- **Can we use custom templates with Llama?**: A member inquired whether their own templates, similar to **RAFT's**, could be used for generating synthetic datasets with **Llama**, or if a specific structure was necessary.
   - This raises questions about the flexibility of Llama's dataset requirements.
- **HF datasets may face compatibility issues**: A member expressed concerns that **HF datasets** might always have compatibility issues due to differing function properties.
   - They noted a preference for converting complex objects to strings for ease of use in datasets.
- **Common practice for complex objects in HF datasets**: A member shared a code snippet suggesting the practice of converting complex objects not following a schema to strings for **HF datasets**.
   - This approach aims to streamline dataset loading in scenarios with non-standard structures.
- **Proposal to update README for additional helper function**: A member offered to create a pull request (PR) to update the **README** with a new helper function that could benefit users.
   - This suggestion was positively received, with another member expressing gratitude for the help.
- **Clarification on JSON file formatting**: A member clarified that there are no issues with the **JSON** files used, stating that HF expects JSON lines formatted files.
   - This reinforces the importance of adhering to the expected file format for successful dataset loading.


  

---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

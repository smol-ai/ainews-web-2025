---
id: 0b735461-2b40-4353-8a4a-33b39cfeec54
title: Mixtral 8x22B Instruct sparks efficiency memes
date: '2024-04-17T21:02:34.918866Z'
original_slug: ainews-mixtral-8x22b-instruct-defines-frontier
description: >-
  Mistral released the instruct-tuned **Mixtral 8x22B** model, which achieves
  strong multilingual performance with *39B active parameters* and a *64k
  context window*, available under an Apache 2.0 license on Hugging Face.
  Meanwhile, **Stability AI** launched **Stable Diffusion 3** and **Stable
  Diffusion 3 Turbo** featuring the new Multimodal Diffusion Transformer
  architecture, promising faster and more reliable image generation, with
  community anticipation around self-hosting options.
tags:
  - mistral
  - databricks
  - google
  - ai21
  - deepmind
  - microsoft
  - intel
  - softbank
  - abu-dhabi-fund
  - nvidia
  - huggingface
  - snowflake
  - neeva
  - langchain
  - llama-index
  - stability
  - unsloth
  - wizardlm
  - boston-dynamics
  - mixtral-8x22b
  - zamba-7b
  - codeqwen1.5-7b
  - codeqwen1.5-7b-chat
  - groundx
  - stable-diffusion-3
  - stable-diffusion-3-turbo
  - wizardlm-2-8x22b
  - wizardlm-2-70b
  - wizardlm-2-7b
  - mistral7b
  - sdxl
  - instruct-tuning
  - multilingual-language-models
  - context-window
  - retrieval-augmented-generation
  - embedding-models
  - open-source-licensing
  - ai-investments
  - deepfake-legislation
  - ai-chip-technology
  - ai-companions
  - code-generation
  - humanoid-robots
  - multimodal-diffusion
  - model-compliance
  - model-takedown
  - self-hosting
  - model-performance
  - long-context-modeling
companies:
  - mistral
  - databricks
  - google
  - ai21
  - deepmind
  - microsoft
  - intel
  - softbank
  - abu-dhabi-fund
  - nvidia
  - huggingface
  - snowflake
  - neeva
  - langchain
  - llama-index
  - stability
  - unsloth
  - wizardlm
  - boston-dynamics
models:
  - mixtral-8x22b
  - zamba-7b
  - codeqwen1.5-7b
  - codeqwen1.5-7b-chat
  - groundx
  - stable-diffusion-3
  - stable-diffusion-3-turbo
  - wizardlm-2-8x22b
  - wizardlm-2-70b
  - wizardlm-2-7b
  - mistral7b
  - sdxl
topics:
  - instruct-tuning
  - multilingual-language-models
  - context-window
  - retrieval-augmented-generation
  - embedding-models
  - open-source-licensing
  - ai-investments
  - deepfake-legislation
  - ai-chip-technology
  - ai-companions
  - code-generation
  - humanoid-robots
  - multimodal-diffusion
  - model-compliance
  - model-takedown
  - self-hosting
  - model-performance
  - long-context-modeling
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/16/2024-4/17/2024. We checked 6 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **27** Discords (**395** channels, and **5173** messages) for you. Estimated reading time saved (at 200wpm): **587 minutes**.

As is their established pattern, Mistral followed up their magnet link [with a blogpost](https://mistral.ai/news/mixtral-8x22b/), and an instruct-tuned version of their 8x22B model:

 ![image.png](https://assets.buttondown.email/images/323db65b-608d-445d-83eb-1d6d9ce35e3f.png?w=960&fit=max) 

the image ended up sparking some friendly competition between [Databricks, Google, and AI21](https://twitter.com/AlbertQJiang/status/1780648008696091003), all of which merely emphasized that Mixtral created a new tradeoff between active params and MMLU performance:

![image.png](https://assets.buttondown.email/images/9677f3b7-64ba-4f12-af15-291dfda26c7d.png?w=960&fit=max)

Of course, what is unsaid that the active params count doesnt linearly correlate with cost to run dense models, and that singular focus on MMLU isn't ideal for less scrupulous competitors.


---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Investments & Advancements**

- **Massive AI investments from tech giants**: In /r/singularity, DeepMind CEO reveals Google plans to invest over **$100 billion** in AI, with other tech giants like Microsoft, Intel, SoftBank, and an Abu Dhabi fund making similarly huge bets, [indicating high confidence in AI's potential](https://www.bloomberg.com/news/articles/2024-04-16/deepmind-ceo-says-google-will-spend-more-than-100-billion-on-ai).

- **UK criminalizes non-consensual deepfake porn**: The UK has made it a crime to create sexually explicit deepfake images without consent. In /r/technology, commenters [debate the implications and enforcement challenges](https://time.com/6967243/uk-criminalize-sexual-explicit-deepfake-images-ai/).

- **Nvidia's AI chip dominance**: In /r/hardware, a former Nvidia employee claims on Twitter that [no one will catch up to Nvidia's AI chip lead this decade](https://i.redd.it/m388weqd9yuc1.png), sparking discussion about the company's strong position.

**AI Assistants & Applications**

- **Potential billion-dollar market for AI companions**: In /r/singularity, a tech executive predicts AI girlfriends could become a **$1 billion business**. Commenters suggest this is a vast underestimate and [discuss the societal implications](https://www.yahoo.com/tech/tech-exec-predicts-ai-girlfriends-181938674.html?).

- **Unlimited context length for language models**: A tweet posted in /r/artificial announces [unlimited context length](https://twitter.com/_akhaliq/status/1780083267888107546?t=hnN1bujYWqBlynr_zEqHKA&s=19), a significant advancement for AI language models.

- **AI surpassing humans on basic tasks**: In /r/artificial, a Nature article reports that [AI has surpassed human performance on several basic tasks](https://www.nature.com/articles/d41586-024-01087-4), though still trails on more complex ones.

**AI Models & Architectures**

- **Zamba: Novel 7B parameter hybrid architecture**: In /r/LocalLLaMA, Zyphra unveils Zamba, a 7B parameter hybrid architecture combining Mamba blocks with shared attention. It [outperforms models like LLaMA-2 7B and OLMo-7B despite less training data](https://www.reddit.com/r/LocalLLaMA/comments/1c61k7v/zamba_a_7b_mambalike_ssm_hybrid_model_trained_for/). The model was developed by a team of 7 using 128 H100 GPUs over 30 days.

---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Mixtral 8x22B Instruct Model Release**

- **Impressive Performance**: [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1780602023203029351) announced the release of Mixtral 8x22B Instruct, which significantly outperforms existing open models using only **39B active parameters** during inference, making it faster than 70B models.
- **Multilingual Capabilities**: [@osanseviero](https://twitter.com/osanseviero/status/1780595541711454602) highlighted that Mixtral 8x22B is fluent in **5 languages** (English, French, Italian, German, Spanish), has **math and code capabilities**, and a **64k context window**. 
- **Availability**: The model is available on the [@huggingface](https://twitter.com/huggingface) Hub under an **Apache 2.0 license** and can be downloaded and run locally, as confirmed by [@_philschmid](https://twitter.com/_philschmid/status/1780598146470379880).

**RAG (Retrieval-Augmented Generation) Advancements**

- **GroundX for Improved Accuracy**: [@svpino](https://twitter.com/svpino/status/1780571442096087224) shared that @eyelevelai released GroundX, an advanced RAG API. In tests on 1,000 pages of tax documents, **GroundX achieved 98% accuracy** compared to 64% for LangChain and 45% for LlamaIndex.
- **Importance of Assessing Risks**: [@omarsar0](https://twitter.com/omarsar0/status/1780613738585903182) emphasized the need to assess risks when using LLMs with contextual information that may contain supporting, contradicting, or incorrect data, based on a paper on RAG model faithfulness.
- **LangChain RAG Tutorials**: [@LangChainAI](https://twitter.com/LangChainAI/status/1780629875533181271) released a playlist explaining RAG fundamentals and advanced methods on @freeCodeCamp. They also shared a [@llama_index](https://twitter.com/llama_index/status/1780646484712788085) tutorial on using Mixtral 8x22B for RAG.

**Snowflake Arctic Embed Models**

- **Powerful Embedding Models**: [@SnowflakeDB](https://twitter.com/SnowflakeDB) open-sourced their Arctic family of embedding models on [@huggingface](https://twitter.com/huggingface), which are the result of @Neeva's search expertise and Snowflake's AI commitment, as noted by [@RamaswmySridhar](https://twitter.com/RamaswmySridhar/status/1780225794402627946).
- **Efficiency and Performance**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1780621521230111181) highlighted the efficiency of these models, with **parameter counts from 23M to 335M**, **sequence lengths from 512 to 8192**, and support for up to 2048 tokens without RPE or 8192 with RPE.
- **LangChain Integration**: [@LangChainAI](https://twitter.com/LangChainAI/status/1780650806896947547) announced same-day support for using Snowflake Arctic Embed models with their @huggingface Embeddings connector.


**Misc**

- **CodeQwen1.5 Release**: [@huybery](https://twitter.com/huybery/status/1780264890298720570) introduced CodeQwen1.5-7B and CodeQwen1.5-7B-Chat, specialized codeLLMs pretrained with **3T tokens** of code data. They exhibit exceptional code generation, long-context modeling (64K), code editing, and SQL capabilities, surpassing ChatGPT-3.5 in SWE-Bench.
- **Boston Dynamics' New Robot**: [@DrJimFan](https://twitter.com/DrJimFan/status/1780622682561929645) shared a video of Boston Dynamics' new robot, arguing that humanoid robots will exceed iPhone supply in the next decade and that "human-level" is just an artificial ceiling. 
- **Superhuman AI from Day One**: [@ylecun](https://twitter.com/ylecun/status/1780596362415063217) stated that AI assistants need human-like intelligence plus superhuman abilities from the start, requiring understanding of the physical world, persistent memory, reasoning and hierarchical planning.



---

# AI Discord Recap

> A summary of Summaries of Summaries

**Stable Diffusion 3 and Stable Diffusion 3 Turbo Launches**:

- **Stability AI** introduced **Stable Diffusion 3** and its faster variant **Stable Diffusion 3 Turbo**, claiming superior performance over DALL-E 3 and Midjourney v6. The models use the new **Multimodal Diffusion Transformer (MMDiT)** architecture.
- Plans to release SD3 weights for self-hosting with a Stability AI Membership, continuing their open generative AI approach.
- Community awaits licensing clarification on personal vs commercial use of SD3.

**Unsloth AI Developments**:

- Discussions on **GPT-4** as a fine-tuned iteration over GPT-3.5, and the impressive multilingual capabilities of **Mistral7B**.
- Excitement around the open-source release of **Mixtral 8x22B** under Apache 2.0, with strengths in multilingual fluency and long context windows.
- Interest in contributing to Unsloth AI's documentation and considering donations to support its development.

**WizardLM-2 Unveiling and Subsequent Takedown**:

- Microsoft announced the **WizardLM-2** family, including 8x22B, 70B, and 7B models, demonstrating competitive performance.
- However, **WizardLM-2** was unpublished due to lack of compliance review, not toxicity concerns as initially speculated.
- Confusion and discussions around the takedown, with some users expressing interest in obtaining the original version.

- **Stable Diffusion 3 Launches with Improved Performance**: **Stability AI** has released **Stable Diffusion 3** and **Stable Diffusion 3 Turbo**, now available on their [Developer Platform API](https://bit.ly/3xHrtjG), boasting the fastest and most reliable performance. The community awaits clarification on the **Stability AI Membership** model for self-hosting SD3 weights. Meanwhile, **SDXL finetunes** have made SDXL refiners nearly obsolete, and users discuss model merging challenges in **ComfyUI** and limitations of the **diffusers** pipeline.

- **WizardLM-2 Debuts Amidst Excitement and Uncertainty**: The release of **WizardLM-2** models by Microsoft has sparked enthusiasm for their potential **GPT-4-like capabilities** in an open-source format. However, the sudden takedown of the models due to a missed compliance review has led to confusion and speculation. Users compare the performance of WizardLM-2 variants and share tips for resolving compatibility issues in **LM Studio**.

- **Multimodal Models Advance with Idefics2 and Reka Core**: **Hugging Face's Idefics2 8B** and **Reka Core** have emerged as powerful multimodal language models, showcasing impressive capabilities in visual question answering, document retrieval, and coding. The upcoming chat-focused variant of Idefics2 and Reka Core's competitive performance against industry giants have generated significant interest. Discussions also revolve around the cost-efficiency of models like **JetMoE-8B** and the launch of **Snowflake's Arctic embed family** for text-embedding.

Other notable topics include:

- The introduction of **ALERT**, a [safety benchmark for assessing Large Language Models](https://github.com/Babelscape/ALERT), and debates around AI safety standards.
- Explorations of **Retrieval Augmented Generation (RAG)** for vision-based applications and the philosophical implications of AI simulations in **World-Sim**.
- The rise of AI-human collaboration platforms like [Payman AI](https://www.paymanai.com/) and the integration of AI inference in **Supabase's edge functions**.
- Challenges to the **Chinchilla scaling laws** and discussions on the expressive power of **state-space models** in the research community.
- Advancements in **PEFT methods** like Dora and RSLoRA, and the pursuit of **multilingual model expansion** using Mixture-of-Experts (MoE) approaches.

---



# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Stable Diffusion 3 Turbo Charges the Scene**: Stability AI has introduced **Stable Diffusion 3** and **Stable Diffusion 3 Turbo**, now available on their Developer Platform API, with claims of the fastest and most reliable performance, supported by Fireworks AI. Interested parties can get started with SD3 at [Stable Diffusion 3 & Developer API](https://bit.ly/3xHrtjG), and an open generative AI approach is promised with plans for the model weights to be available for self-hosting for members.

**Refining Visually Intuitive Generative AI**: The **SDXL finetunes** have made the use of SDXL refiners nearly obsolete, as they are now prevalent in **Civitai** downloads, suggesting a trend towards integrated finetunes over separate refiner modules, reflecting a community-driven optimization.

**Model Merging Explored**: There is lively discussion on model merging tactics within **ComfyUI**, grappling with complex mechanisms such as V-prediction and epsilon, highlighting the community's experimentation with these methods to achieve enhanced outcomes, yet acknowledging that correct implementations are crucial to prevent unpredictable results.

**Navigating Diffusers Library Limitations**: A conversation emerged around the limitations and dependencies in the **diffusers** pipeline, with a focus on **Stable Video Diffusion Pipeline** challenges. Despite these challenges, some users are optimizing usage by running models independently post-download, bypassing certain **Hugging Face** library constraints.

**Awaiting SD3’s Membership Model Details**: The community is keenly waiting for Stability AI to provide clarifications on **Stable Diffusion 3** licensing for personal versus commercial use, especially in light of the new membership model revealed for accessing self-hosted weights.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**GPT-4 Gains Over GPT-3.5**: The new iteration of GPT, **GPT-4**, is regarded as a fine-tuned enhancement over GPT-3.5, though specifics on performance metrics or features were not provided.

**Mistral7B Shines in Multilingualism**: Members conferred about the multilingual capabilities of the **Mistral7B** model, recommending the inclusion of diverse language data in training sets, particularly French, to improve performance.

**Unsloth AI Gets Help from Fans**: There’s a tangibly positive response from the community towards **Unsloth AI**, with users keen to help with documentation, expansion, and even considering donations. The **Mixtral 8x22B** model's release under **Apache 2.0** was met with excitement for its promise in multilingual fluency and handling of extensive context windows.

**Chroma Goes Go**: The **Chroma** project leaps forward with an edge version written in Go, which utilizes SQLite and **WASM** for browser-based applications, now available on [GitHub](https://github.com/l4b4r4b4b4/go-chroma).

**Mobile AI Deployment Discussed**: The complexity of deploying AI models on mobile devices surfaced, noting challenges such as the absence of CUDA and the infeasibility of running standard Deep Learning Python codes on such platforms.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**AI Assistance for NeoScript Programming**: A user looking for help with NeoScript programming expressed challenges in configuring AI models. Microsoft's new release, [WaveCoder Ultra 6.7b](https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF), excels in code translation and could be a strong candidate for this task.

**Solving AI's Echo Chamber**:
To combat repetitive AI responses, particularly in Dolphin 2 Mistral, members discussed strategies such as fine-tuning models and leveraging multi-turn conversation frameworks outlined in [Azure's article](https://learn.microsoft.com/en-us/azure/ai-services/qnamaker/how-to/multi-turn#what-is-a-multi-turn-conversation).

**Introducing the WizardLM-2 League**: The debut of **WizardLM-2** models sparked discussions about performance. Compatibility with existing tools, including the importance of using **GGUF quants** and version **0.2.19** or newer for proper functionality, was emphasized.

**Tech Wizards at Play**: One user successfully enabled direct communication between four **3090 GPUs**, improving model performance by bypassing CPU/RAM. There was also chatter about the challenges of signing Windows executables, with a hint that the Windows versions are indeed signed with an [Authenticode cert](https://docs.microsoft.com/en-us/windows-hardware/drivers/install/authenticode).

**Quantization Conundrum and Model Preferences**: Mixed reviews on quantization levels, from Q8 to Q6K, pointed to a preference for models with higher quantization levels when VRAM is sufficient. For large models, such as **WizardLM-2-8x22B**, GPUs like the 4090 with 24GB VRAM may be inadequate.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Multimodal Models Stepping Up**: Exciting advancements in multimodal language models are showcased, with **Hugging Face's Idefics2 8B** and **Reka Core** emerging as key players, evident from [Open Multimodal ChatGPT video](https://www.youtube.com/watch?v=vL1SayPCHBg) and [Reka Core overview](https://www.youtube.com/watch?v=U7RbwPKyxs8). The GPT4v/Geminipro Vision and Claude Sonnet models are recommended for vision-RAG applications.

- **LLMs Tuning into Self-Optimization**: New techniques for enhancing Instruct Model LLMs look promising, with models able to select the best solution by reconstructing inputs from outputs, detailed in a [Google Slideshow](https://docs.google.com/presentation/d/1dk2ekDPa9qFuT4B0WafaZLRso5YdTpgv9FaOEQ_lNvs/edit?usp=sharing) on aligning LLMs for medical reasoning. 

- **WizardLM Disappearance Sparks Debate**: There's uncertainty around **WizardLM**'s sudden takedown; while some speculated on toxicity issues, confirmed reports attributed it to lack of a compliance review as shared in a comprehensive [WizardLM information bundle](https://huggingface.co/alpindale/WizardLM-2-8x22B).

- **LLMs Performance: A Roller Coaster of Expectations**: Engineers discuss **CodeQwen1.5-7B Chat**'s impressive benchmarking and debate on architectures and tuning's impact on performance. Furthermore, upcoming models like **Hermes 8x22B** are eagerly awaited, with concerns on whether they can be accommodated by personal equipment setups.

- **World-Sim's Return Triggers AI Philosophical Debates**: As World-Sim gears up for a return, enthusiasts burst with anticipation, pondering the philosophical aspects and implications of such simulated worlds. Official confirmation sent excitement soaring with a [Websim link](https://websim.ai/c/BZcLXGB6Ft5cjnLns) provided for those eager to jump in.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Robots Debating Their Roots**: Engineers exchanged insights on the performance nuances of AI models including **GPT-4** and **Claude 3 Opus**, with a shared sentiment that **GPT-4** may exhibit "lazy" tendencies in real-world applications. The open-source **[Mixtral's 8x22B model](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)** is highlighted for its impressive capabilities, sparking debates on model efficacy.

**Stumped by Stubborn Software Issues**: A conversation was noted about achieving consistency between the web client and the API, with specific attention to parameters like **temperature settings**. Engineers are also discussing the benefits of including a rate limit counter in the API response for better management and transparency.

**The Vanishing Messages Mystery**: Concern was voiced over changes in the Perplexity API's payment method management, particularly the opacity surrounding the remaining message counts for pro users. This focus on transparency indicates professionals need clarity to manage resources efficiently.

**A Tale of Truncated Tokens**: Technical dialogue included challenges faced when engaging models with large context sizes, like a 42k token prompt, and the tendency for models to summarize rather than dive deep into lengthy documents. This could be pivotal as engineers optimize models to process complex prompts fully.

**The Search for Smarter Searches**: Members also discussed using `site:URL` search operators for more targeted information retrieval. Additionally, there is a call for better communication regarding rate limits in the API, including the possibility of a `429 response`.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **PyTorch's Abstraction Puzzle**: Engineers are grappling with **PyTorch's** philosophy of abstracting complexities, which, while simplifying coding, often leaves them puzzled when troubleshooting unexpected results.

- **Handling Hefty Datasets with Zarr**: There's active exploration on utilizing **zarr** to manage a hefty 150 GB MRI dataset, with discussions circling around its efficiency and whether it will overload RAM with large data loads.

- **Legal Lines Drawn for Deepfakes in the UK**: Members are discussing the implications of UK legislation targeting the creation of distressing images, questioning its enforceability given the blurriness of proving intent.

- **AI Inference Fine-Tuning Talks**: Voices from the community are calling for clarity on AI models' inference settings, like controlling **CFG** or integrating models with robust **ODE solvers**, beyond just defaulting to Euler's method.

- **Cascade Team's Corporate Shuffle**: There's speculation about the future of **Stability AI's Cascade team** after their departure and the dissolution of their Discord channel, with wonderment if there's a link to a new venture, possibly **Leonardo**, or an ongoing affiliation with SAI.

- **ALERT! A New Safety Benchmark for LLMs**: The introduction of **ALERT**, a safety benchmark for assessing **Large Language Models**, has sparked interest, providing a Dataset of Problematic Outputs (DPO) for community evaluation, available on [GitHub](https://github.com/Babelscape/ALERT).

- **AI Audio-Visual Harmony**: An **[Arxiv paper](https://arxiv.org/abs/2404.09956)** presents methods for generating audio from text, improving performance by zeroing in on concepts or events, stirring dialogue in the research community.

- **AI Safe or Stifled?**: The AI safety debate is heated, with some pushing back against confining AI strictly to PG content, arguing it could crimp its creative spark compared to other artistic mediums.

- **GANs vs. Diffusion Models: Speed or Aesthetics?**: Discussions are heating up over the advantages of **GANs**—notably, their faster inference and lesser parameter count—versus diffusion models, even as GANs face criticism for image quality and training challenges.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**OpenRouter Welcomes WizardLM Raptors**: OpenRouter announced the release of **[WizardLM-2 7B](https://openrouter.ai/models/microsoft/wizardlm-2-7b)** and a price drop for **[WizardLM-2 8x22B](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b)** to $0.65/M tokens. The **[WizardLM-2 8x22B Nitro](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b:nitro)** boasts over 100 transactions per second post its database restart.

**Latency Labyrinth Resolved**: Latency issues on various models such as **Mistral 7B Instruct** and **Mixtral 8x7B Instruct** were attributed to cloud provider DDoS protection, with updates concerning the resolution found in the associated [discussion thread](https://discord.com/channels/1091220969173028894/1229813179681345556).

**Calling All Frontend Mavericks**: A member seeks web development assistance for an AI-based frontend project for OpenRouter, specifically emphasizing role-playing novel mode and conversation style systems. Ability to distinguish AI-generated text from user input is also requested.

**AI Model Morality and Multilingual Mastery**: Vigorous exchanges regarding both censorship protocols for NSFW content and the imperative for enhancing models' multilingual performance took place. Members looked forward to direct endpoints and new provider integrations for an anticipated AI model release.

**Bitrate Bits and Quality Quibbles**: Users showed a clear preference for a minimum of 5 bits per word (bpw) for model quantization, noting that reductions below this threshold notably compromise quality. Discussions underscored the trade-offs between efficient operation and maintaining high fidelity in AI outputs.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo to Python Conversion Now a Possibility**: Engineers discuss the new package [mojo2py](https://github.com/venvis/mojo2py), capable of converting Mojo code to Python, and chatted about the desire for more learning resources, pointing to the [Mojo programming manual](https://docs.modular.com/mojo/manual/) for beginners.

- **Maxim Zaks Debates the Mojo 'Hype'**: A PyCon Lithuania talk titled "Is Mojo just a hype?" by Maxim Zaks was highlighted, provoking debate on the chatbot's industry impact, available in a [video](https://youtu.be/mhZFyzqdmi8).

- **Mojo's Inherent Nightly Nuances**: Users are navigating through the challenges of a new nightly Mojo release, noting unconventional code styling for readability, desires for comprehensive tutorials on traits, and a [recent pull request](https://github.com/modularml/mojo/pull/2313/files) reflecting significant updates.

- **Optimizing with Compile-Time Aliases**: Discussion thrived around optimizing alias memory usage in Mojo, hinted by the recommendation of readable code over extensive commenting from a cited [YouTube video](https://m.youtube.com/watch?v=Bf7vDBBOBUA).

- **Community Mojo Projects Surge**: Community contributions soared with a shared Mojo 'sketch' found at [this gist](https://gist.github.com/lsh/6ca8864a9cffef9e503d6262eb876893) and a request about implementing the Canny edge recognition algorithm in Mojo, coupled with directions to Mojo's [documentation](https://docs.modular.com/mojo/manual/get-started/) and tooling resources.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**PyTorch Resource Debate**: While discussing if "Deep Learning with PyTorch" is a relevant resource despite being 4 years old, members noted that the **PyTorch core** has remained stable, though significant updates have occurred in the compiler and distributed systems. A member shared a teaser for an [upcoming edition of the book](https://www.manning.com/books/deep-learning-with-pytorch-second-edition), which would include coverage of transformers and Large Language Models.

**CUDA Custom GEMM Sparking Interest**: The conversation involved improving GEMM performance in CUDA, with one member providing a new implementation that outperformed PyTorch's function on specific benchmarks, sharing their code on [GitHub](https://github.com/aredden/torch-cublas-hgemm/blob/master/src/simt_hgemv.cu). However, another highlighted JIT compilation issues with `torch.compile`. The group also discussed optimal block size parameters, referencing a [related code example on Gist](https://gist.github.com/mobicham/9aa8dc0e64ea1cb7d4e44fef55e6a4b4).

**Next-Gen Video Analysis & Robotics Gains Screenshare**: Members shared links about Augmend's video processing features, which combine OCR and image segmentation, previewed on [wip.augmend.us](http://wip.augmend.us), and the full service to be hosted on [augmend.com](http://augmend.com). Another highlight was Boston Dynamics' unveiling of a fully electric robot named *Atlas* intending for real-world applications, showcased in their [All New Atlas | Boston Dynamics video](https://www.youtube.com/watch?v=29ECwExc-_M).

**Bridging the CUDA Toolkit Knowledge Gap**: In the #beginner channel, members discussed issues related to using the CUDA toolkit on WSL, with one user facing problems running the **ncu profiler**. The community provided troubleshooting steps and stressed the importance of setting the correct **CUDA path in environment variables**. There was also an advisory that **Windows 11** might be necessary for effective CUDA profiling on WSL 2, with one user providing a [guide on the subject](https://peterchng.com/blog/2024/03/02/profiling-cuda-programs-on-wsl-2/).

**Quantization Dilemmas and Solutions in Air**: A thorough chat occurred on the topic of quantization axes in GPT models with a highlight on the complexities when using `axis=0`. Participants suggested quantizing Q, K, and V separately with references to Triton kernels and an autograd optimization method for boosting speed and performance. Their debate continued with discussions of 2/3 bits quantization practicality and was supplemented with implementation details and benchmarks [on GitHub](https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py).

**Optimizing ML Model Performance**: A GitHub notebook for extending PyTorch with CUDA Python garnered attention for speed enhancements but with a need for more optimization to fully tap into tensor core capabilities, as shared in the [notebook's link](https://github.com/Lightning-AI/lightning-thunder/blob/main/notebooks/extend_thunder_with_cuda_python.ipynb). Additionally, there were mentions of optimizing the softmax function and block sizes for cache utilization, with insights shared through a [GitHub pull request](https://github.com/karpathy/llm.c/pull/150).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Multiplayer GPT Headed for the Gaming Galaxy**: Engineers discussed the potential of integrating **GPT-Vision** and camera inputs for a **real-time gaming assistant** to tackle multiple-choice games. The possibility of utilizing **Azure** or virtual machines to handle intensive computational tasks was raised, alongside leveraging TensorFlow or OpenCV for system management.

**AI Versus Human Conundrum Continues**: A philosophical debate emerged concerning the differences between AI and human cognition, discussing the prospects of AI acquiring **human-like reasoning and emotions**, and the role of quantum computing in this evolution.

**The Quest for Knowledge Enhancements**: Members sought information on how to prepare a **knowledge base for custom GPT** applications and questioned the arrival of the **Whisper v3 API**. The noted limitations such as GPT-4's token memory span being speculated to have shrunk triggered calls for improved clarity on API capabilities.

**Creative Minds Favor Claude and Gemini**: When tackling literature reviews and fictional works, AI aficionados recommended using models like **Claude** and **Gemini 1.5**. These tools were favored for their prowess in handling literary tasks and creative writing respectively.

**Discord Channel Dynamics**: Two channels, **prompt-engineering** and **api-discussions**, experienced a notable decrease in activity, with participants attributing the quiet to possible over-moderation and a recent string of timeouts, including a specific **5-month timeout** case involving assistance to another user.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Hybrid Cloud Hustle with Qdrant**: Qdrant's new hybrid cloud offering allows for running their service across various environments while maintaining control over data. They backed their launch with a [thorough tutorial](https://t.co/4nS9j9ruwR) on the setup process.

- **LlamaIndex Beefs Up with Azure AI Search**: LlamaIndex teams up with Azure AI Search for advanced RAG applications, featuring a [tutorial](https://t.co/lITCdlCejT) by Khye Wei that illustrates Hybrid Search and Query rewriting capabilities.

- **MistralAI Model Immediately Indexed**: LlamaIndex has instant support for [MistralAI’s newly released 8x22b model](https://t.co/WWbYp5lqXe), paired with a Mistral cookbook focusing on intelligent query routing and tool usage.

- **Building and Debugging in LlamaIndex**: AI engineers discussed best practices for constructing search engines in LlamaIndex, resolving API key authentication errors, and navigating through updates and bug fixes, including a specific `BaseComponent` error with a [GitHub solution](https://github.com/run-llama/llama_index/pull/12882).

- **Hierarchical Structure Strategy Session**: Inquiry within the **ai-discussion channel** about constructing a hierarchical document structure using ParentDocumentRetriever, with LlamaIndex as the framework of choice.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Peering into the Future of Long-Sequence Models**: [Feedback Attention Memory (FAM)](http://arxiv.org/abs/2404.09173), discussed in recent conversations, proposes a solution to the quadratic attention problem of Transformers, enabling processing of indefinitely long sequences and showing improvement on long-context tasks. Reka's new encoder-decoder model is touted to support sequences up to 128k, as detailed in their [core tech report](https://publications.reka.ai/reka-core-tech-report.pdf).

- **Precision in Scaling Laws and Evaluation**: Questions on compute-optimal scaling laws by Hoffman et al. (2022) led to an exploration of the credibility of narrow confidence intervals without extensive experiments as detailed in [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102). Moreover, accurate cost estimations within ML papers are hindered when the size of datasets like that in the SoundStream paper is omitted, bringing to light the necessity of transparent data reporting.

- **Unpacking Model Evaluation Techniques**: In Eleuther's `#lm-thunderdome`, the usage of `lm-evaluation-harness` was demystified, explaining the output format required for `arc_easy` tasks and discussing the significance of BPC (bits per character) as an intelligent proxy correlating with a model's compression capacity. Concerning tasks like ARC, a dialogue ensued about why random guessing results in a roughly 25% accuracy rate due to its four possible answers.

- **Multi-Modal Learning Gains Traction**: The possibility of **Total Correlation Gain Maximization (TCGM)** for semi-supervised multi-modal learning received attention, with one [arXiv paper](https://arxiv.org/abs/2302.12247) discussing the informational approach and the ability to utilize unlabeled data across modalities effectively. Emphasis was also given to the method's theoretical promises and its implications in identifying Bayesian classifiers for diverse learning scenarios.

- **Concrete Guidelines for FLOPS Calculation**: On the `#scaling-laws` channel, advice was given on estimating the FLOPS for a model such as SoundStream, including using the equation **6 * # of parameters** for transformers during forward and backward passes. Newcomers are directed to a comprehensive breakdown in [Section 2.1 of the relevant paper](https://arxiv.org/abs/2001.08361) for a complete understanding of computational cost estimation.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **IDEFICS-2 Takes the Limelight**: The release of **IDEFICS-2** brings an impressive skill set with 8B parameters, capable of high-resolution image processing and excelling in visual question answering and document retrieval tasks. Anticipation builds as a chat-focused variant of IDEFICS-2 is promised, while current capabilities such as solving complex CAPTCHAs are demonstrated in a [shared example](https://x.com/lunarflu1/status/1780228654397599904).

- **Knowledge Graphs Meet Chatbots**: An informative [blog post](https://mlabonne.github.io/blog/posts/Article_Improve_ChatGPT_with_Knowledge_Graphs.html) highlights the integration of **Knowledge Graphs** with chatbots to boost performance, with exploration encouraged for those interested in advanced chatbot functionality.

- **Snowflake's Arctic Expedition**: Snowflake breaks new ground with the launch of the **Arctic embed family of models**, claimed to set new benchmarks in practical text-embedding model performance, particularly in retrieval use cases. This development is complemented by a hands-on [Splatter Image space](https://huggingface.co/spaces/szymanowiczs/splatter_image) for creating splatter art quickly, and how **Multi-Modal RAG** fuses language and images, as detailed in [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/use_cases/multimodal/).

- **Model Training and Comparisons Drive Innovation**: A fresh **IP-Adapter Playground** is unveiled, further enabling creative text-to-image interactions, alongside a new option to `push_to_hub` directly in the transformers library's pipelines. Comparing image captioning models just got easier with a dedicated [Hugging Face Space](https://huggingface.co/spaces/unography/comparing-captioning-models).

- **Challenges and Opportunities in NLP and Vision**: Community members discuss issues from truncated token handling in prompts to exploring LoRA configurations, with links shared to resources on topic modeling with [BERTopic](https://maartengr.github.io/BERTopic/index.html), training T5 models ([Github Resource](https://github.com/EleutherAI/improved-t5)), and LaTeX-OCR possibilities for equation conversion [LaTeX-OCR GitHub](https://github.com/lukas-blecher/LaTeX-OCR). These conversations encapsulate the collective pursuit of refining and harnessing AI capabilities.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Idefics2 Brings Multimodal Flair**: The new multimodal model [Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b) has been introduced, capable of processing both text and images with improved OCR and visual reasoning skills. It is offered in both base and fine-tuned forms and is under the Apache 2.0 license.

**RTX 5090 Speculation Stokes Anticipation**: NVidia is rumored to be considering an expedited release of the [RTX 5090](https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch), potentially at Computex 2024, to stay ahead of AMD's advances, sparking discussions on hardware suitability for cutting-edge AI models.

**Model Training Finetuning**: Engineers shared insights on model training configurations, focusing on the 'train_on_input' parameter in loss calculation, and suggested using "TinyLlama-1.1B-Chat-v1.0" for fine-tuning small models for efficient experimentation.

**Phorm AI Becomes Go-To Resource**: Members referred to Phorm AI for various inquiries, including epoch-wise saving techniques and data preparation for models like TinyLlama for tasks like text-to-color code predictions.

**Spam Flood Triggers Alerts**: Multiple channels within the community were targeted by spam messages promoting OnlyFans content, attempting to divert attention from the AI-centric conversations and technical discourse.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**LLM Ranking Resource Revealed**: A comprehensive website, [LLM Explorer](https://llm.extractum.io/), has been shared, showcasing a plethora of open-source language models, each assessed through ELO scores, HuggingFace leaderboard ranks, and task-specific accuracy metrics, serving as a valuable resource for model comparison and selection.

**AI+Human Symphony in the Gig Economy**: The launch of [Payman AI](https://www.paymanai.com/), a platform facilitating AI agents to remunerate humans for tasks beyond AI capabilities, has sparked interest; the concept promotes a cooperative ecosystem between AI and human talents in domains like design and legal services.

**Supabase Embraces AI Inference**: Supabase introduces a simple API for running AI inferences within its edge functions, allowing AI models such as `gte-small` to be employed directly in databases, as detailed in their [announcement](https://supabase.com/blog/ai-inference-now-available-in-supabase-edge-functions).

**Buzz Around "Llama 3" and OpenAI API Moves**: The AI community is abuzz about the mysterious "Llama 3" speculated to debut at a London hackathon, and OpenAI's Assistants API enhancements are drawing attention in light of a potential GPT-5 release, stirring debates about possible impacts on AI startups and platforms.

**BloombergGPT Paper Club Session Goes Zoom**: The LLM Paper Club invites engineers to a Zoom session on **BloombergGPT**, due to prior challenges with Discord screensharing, and the discussion has pivoted to Zoom for a better sharing experience. Participants can register for the event [here](https://lu.ma/w7jhce1y), and further reminders to join the discussions are being circulated within the community.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **AI Wearable Woes**: AI wearables lack the **contextual knowledge** of smartphones, as discussed with reference to a [YouTube review by Marquis Brownlee](https://youtu.be/TitZV6k8zfA). Engineers pointed out that greater contextual understanding is necessary for AI assistants to provide efficient responses.

- **Open-Source AI Model Buzz**: The **WizardLm2** open-source model garners interest for its potential to deliver **GPT-4-like capabilities**. Discussions forecast a strong future demand despite ongoing advancements.

- **Translator Bot's Inclusive Promise**: Engineers are currently evaluating a new **translation bot** for its ability to enrich communication by providing two-way translations, aiming for more inclusive and unified discussions.

- **Cross-Platform Compatibility Challenges**: There's a clear need for software like **01 Light** to operate on **Windows**, consistent with dialogues about difficulties adapting Mac-centric software to Windows frameworks, thereby hinting at the necessity for platform-agnostic development approaches.

- **Hardware Heats Up**: Conversations indicate significant interest in AI hardware solutions like the **Limitless** device, with comparisons drawn around user experiences. Emphasis on the need for robust backend support and seamless AI integration is shaping hardware aspirations.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Big Win for qwen-1.5-0.5B**: The **qwen-1.5-0.5B** model's winrate soared from **4% to 32%** against heavyweights like AlpacaEval using *generation in chunks*. This approach, along with a 300M reward model, may be a game-changer in output searching.

**How To Win Friends and Influence AIs**: The recently unveiled [Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b/), a polyglot SMoE model, is sharing the limelight owing to its impressive capabilities and the Apache 2.0 open license. Meanwhile, the rise of [OLMo 1.7 7B](https://huggingface.co/allenai/OLMo-1.7-7B) indicates a notable stride in language model science with a robust performance leap on the MMLU benchmark.

**Replicating Chinchilla: An Anomaly**: Discrepancies in replicating the [Chinchilla scaling paper by Hoffmann et al.](https://x.com/tamaybes/status/1780639257389904013?s=46) have cast doubts around the paper's findings. The community's reaction ranged from confusion to concern, signaling an escalating drama around the challenge of scaling law verification.

**Lighthearted Anticipation and Rumination**: With playful banter on potential showdowns in **olmo vs llama**, community members show humor in competition. Moreover, Nathan Lambert teases the guild with a forecast of content deluge, signaling a possibly intense week of knowledge sharing.

**Model Madness or Jocularity?**: A side comment in an underpopulated channel by Nathan mentioned a potential tease involving **WizardLM 2** as a troll, showing a blend of humor and light-heartedness amidst technical discussions.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **API Confusion Needs Resolving**: Engineers are probing the **Cohere API** for details on system prompt capabilities and available models. A user highlighted the request for details due to their significance in application development.

- **Benchmarking Cohere's Embeddings**: There is curiosity about how **Cohere's embeddings v3** perform against OpenAI's new large embeddings with reference to the Cohere blog, suggesting a comparative analysis has been conducted [Introducing Command R+](https://txt.cohere.com/int8-binary-embeddings/).

- **Integration Tips and Tricks**: Technical discussions addressed integrating Language Learning Models (LLMs) with platforms like BotPress, and whether Coral necessitates a local hosting solution. Future updates might simplify these integrations.

- **Fine-Tuning Fine-Tuned Models**: Clarification was sought about fine-tuning already customized models via Cohere's Web UI, directing users to the official guide [Fine-Tuning with the Web UI](https://docs.cohere.com/docs/fine-tuning-with-the-web-ui).

- **Beta Testers Called to Action**: A project named **Quant Fino** is recruiting beta testers for its Agentic entity that merges GAI with FinTech. Interested participants can apply at [Join Beta - Quant Fino](https://quantfino.com/join-beta).

- **Security Flaws Exposed in AI Model**: A redteaming exercise revealed vulnerabilities in **Command R+**, demonstrating the ability to manipulate the model into creating unrestricted agents. Concerned engineers and researchers can review the full write-up [Creating unrestricted AI Agents with Command R+](https://www.lesswrong.com/posts/4vPZgvhmBkTikYikA/creating-unrestricted-ai-agents-with-command-r).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**AI Documentation Gets Facelift**: In an effort to improve usability, contributors to the **LangChain** documentation are revamping its structure, introducing categories like 'tutorial', 'how to guides', and 'conceptual guide'. A member shared the [LangChain introduction page](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction), emphasizing LangChain's components such as building blocks, LangSmith, and LangServe, which aid in the development and deployment of applications with large language models.

**Building with LangChain — An Expressive Endeavor?**: Within the **#[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1229689230977011824)** channel, a member sought advice on YC startup applications while drawing parallels to Extensiv, leading to the mention of several entities like **Unsloth, Mistral AI**, and **Lumini**. Simultaneously, challenges with **LangServe** integration when combined with **Nemo Guardrails** were highlighted due to Nemo's transformation of output structures.

**Forge Ahead with New AI Tools and Services**: GalaxyAI's debut of an API service with complimentary access to **GPT-4** and **GPT-3.5-turbo** stirred up interest, showcased at [Galaxy AI](https://galaxyapi.onrender.com). Similarly, OppyDev’s fusion of an IDE and a chat client received attention, advocating an improved coding platform accessible at [OppyDev AI](https://oppydev.ai). Meanwhile, Rubiks.ai appealed to tech enthusiasts to beta test their search engine and assistant at [Rubiks.ai](https://rubiks.ai) using code `RUBIX`.

**AI Pioneers Share Educational Resources and Seek Collaboration**: A member from **#[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1229725236593692722)** posted a [YouTube tutorial](https://youtu.be/7LWTZqksmSg) on granting AI agents with long-term memory, igniting a discussion why 'langgraph' wasn't employed. Furthermore, a participant expressed eagerness to collaborate on new projects, inviting others to connect through direct messaging.

**Diverse Dialogues on Data and Optimization**: In a lively exchange, strategies for optimizing **RAG (Retrieval-Augmented Generation)** with large documents were evaluated, including document splitting. Members also dialogued over the best methods to manipulate CSV files with **Langchain**, suggesting improvements for chatbots and data processing.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **64 GPUs Engaged for Full-Scale Deep-Speed**: Maxidl pushed the limits by utilizing **64 80GB GPUs**, each at 77GB capacity, to run **full-scale deep-speed** with 32k sequence length and batch size of one, exploring **8-bit optimization** for better memory efficiency.
- **FSDP's Memory Usage Secrets Unlocked**: _jp1_ suggested `fsdp_transformer_layer_cls_to_wrap: MixtralSparseMoeBlock`, and setting `offload_params = true` to minimize memory usage, potentially reducing GPU requirements to 32, while maxidl sought out calculators for memory usage, referencing a [HuggingFace discussion](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/12).
- **Copyright Conundrum for Text Scraping**: A member pointed out the **EU copyright gray area** affecting text data scraping and suggested **DFKI** as a useful source. Meanwhile, multimodal data from **Wikicommons** and others are found on [Creative Commons Search](https://search.creativecommons.org/).
- **Tokenization Techniques on the Rise**: The community shared insights into creating a **Llama tokenizer** without HuggingFace, noted a misspelling in a shared custom tokenizer, and highlighted **Mistral's** new tokenization library, with a [GitHub notebook provided](https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb).
- **Decoding Strategies and Sampling Techniques Evaluated**: Concerns that [a paper on decoding methods](https://arxiv.org/abs/2402.06925) overlooked useful strategies led to a discussion on unaddressed techniques like **MinP/DynaTemp/Quadratic Sampling**. A [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1c36ieb/comparing_sampling_techniques_for_creative/) showed the impact of **min_p sampling** on creative writing, boosting scores by +8 in alpaca-eval style elo and +10 in eq-bench creative writing test.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Int8 Integration in Tinygrad**: Tinygrad has been confirmed to support **INT8 computations**, with recognition that such data type support often depends more on **hardware capabilities** rather than the software design itself.

**Graph Nirvana with Tiny-tools**: For enhanced graph visualizations in **Tinygrad**, users can visit [Tiny-tools Graph Visualization](https://tiny-tools-client.vercel.app/) to create slicker graphs than the basic `GRAPH=1` setting.

**Pytorch-Lightning's Hardware Adaptability**: Discussions about **Pytorch-Lightning** touched on its hardware-agnostic capabilities, with practical applications noted on hardware like the **7900xtx**. [Discover Pytorch-Lightning on GitHub](https://github.com/Lightning-AI/pytorch-lightning).

**Tinygrad Meets Metal**: Community members are exploring the generation of **Metal compute shaders** with tinygrad, discussing how to run simple Metal programs without Xcode and the possibility of applying this to **meshnet models**.

**Model Manipulation and Efficiency in Tinygrad**: A member's proposal for a fast, probabilistically complete **Node.equals()** prompted discussions on efficiency, while **George Hotz** explained layer device allocation, and users were directed toward *tinygrad/shape/shapetracker.py* or *view.py* for zero-cost tensor manipulations like broadcast and reshape.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Hugging Face Showcases Idefics2**: [Hugging Face](https://huggingface.co/blog/idefics2) introduces **Idefics2**, a new multimodal **ChatGPT** iteration that integrates Python coding capabilities, as demonstrated in their [latest video](https://www.youtube.com/watch?v=vL1SayPCHBg).
- **Reka Core Rivals Tech Behemoths**: Touted for its performance, **Reka Core** emerges as a strong competitor to language models from OpenAI and others, with a [video overview available](https://www.youtube.com/watch?v=U7RbwPKyxs8) to showcase its capabilities.
- **JetMoE-8B Flaunts Efficient AI Performance**: The **JetMoE-8B model** impresses with performance that surpasses Meta AI's LLaMA2-7B while costing under $0.1 million, suggesting a cost-efficient approach to AI development as explained in [this breakdown](https://www.youtube.com/watch?v=Z9Hwp_XeS1A).
- **Snowflake Announces Premier Text-Embedding Model**: Snowflake debuts the **Snowflake Arctic embed family** of models, claiming the title for the world's most effective practical text-embedding model, detailed in their [announcement](https://www.youtube.com/watch?v=p9T7ZgtM5Mo).



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Mixtral Mania**: Engineers are eagerly awaiting to test the **Mixtral 8x22B Instruct** model; for those interested, the [Model Card on HuggingFace](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) is now available.
- **Glitch in the Machine**: There's a reported installation error for **llm-gpt4all** that seems to obstruct usage; details of the problem can be found in the [GitHub issue tracker](https://github.com/simonw/llm-gpt4all/issues/28).



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Legal Entanglements Afoot?**: A member hinted at possible **legal involvement** in an unspecified situation, yet no context was provided to ascertain the details or nature of the legal matters in question.
- **The Misfortune of wizardlm-2**: An image was shared showing the deletion of **wizardlm-2**, noted specifically for lack of testing on **v0**; the intricacies of **wizardlm-2** or the testing processes were not elaborated. [View Image](https://cdn.discordapp.com/attachments/1019530324255965186/1229693872997666816/wizardlm-2-was-deleted-because-they-forgot-to-test-it-for-v0-lyaop5lw0suc1.png?ex=66309ca9&is=661e27a9&hm=f105e6497796be9c414ade2024a27f9561caf0cad6cb06ba09f80e30b5e39ae4&)



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile Script Gets a Facelift**: An improved repacking script for the llamafile archive version upgrade is now accessible via [this Gist](https://gist.github.com/mofosyne/46c63934305d5a5321c7e9fd83f4ef3e), triggering a discussion on whether to merge it with the main GitHub repo or to start new llamafiles from scratch due to concerns about maintainability.

- **Seeking Protocol for Security Flaws**: The discussion surfaced a need for clarification on the procedure to report security vulnerabilities within the system, including the steps to request a CVE number, although specific guidance is currently lacking.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1230162110596649011)** (1 messages): 

- **Stable Diffusion 3 Launch Celebration**: Stable Diffusion 3 and its faster variant, Stable Diffusion 3 Turbo, are now available on the Stability AI Developer Platform API. This release is powered through a partnership with Fireworks AI, boasting claims of being the fastest and most reliable API platform.

- **Open Generative AI Continues**: There is a plan to make Stable Diffusion 3 model weights available for self-hosting, which would require a Stability AI Membership, emphasizing the continued commitment to open generative AI.

- **Discover More About SD3**: Users are directed to [learn more and get started](https://bit.ly/3xHrtjG) with the new offerings through the provided link, which includes further details and documentation.

- **Research Background Unpacked**: According to the [Stable Diffusion 3 research paper](https://stability.ai/news/stable-diffusion-3-research-paper), this iteration rivals or surpasses the leading text-to-image systems like DALL-E 3 and Midjourney v6 in aspects such as typography and adherence to prompts, based on human preference studies.

- **Technical Advancements in SD3**: The latest version introduces the Multimodal Diffusion Transformer (MMDiT) architecture, offering improved text comprehension and image representation over previous Stable Diffusion models by utilizing distinct weight sets for different modalities.

**Link mentioned**: <a href="https://bit.ly/3xHrtjG">Stable Diffusion 3 API Now Available &mdash; Stability AI</a>: We are pleased to announce the availability of Stable Diffusion 3 and Stable Diffusion 3 Turbo on the Stability AI Developer Platform API.&amp;nbsp;

  

---


**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1229691568986914866)** (1039 messages🔥🔥🔥): 

- **SD3 Awaits Membership Clarification**: Amidst the concerns of licensing and accessibility, users await a clear statement from Stability AI regarding SD3's availability for personal and commercial use. Discussions arose following an [announcement](https://stability.ai/news/stable-diffusion-3-api) stating plans to make the model weights available for self-hosting with a Stability AI Membership.

- **SDXL Refiners Deemed Redundant**: The community finds SDXL finetunes to have made the use of SDXL refiners obsolete, stating that refiner-trained finetunes have taken precedence in Civitai downloads. Some users reminisce about initial uses of refiners but acknowledge that finetune integrations quickly replaced the need for them.

- **Model Merging Challenges**: Users explore the effectiveness and understanding of model-merging concepts around V-prediction and epsilon in ComfyUI. There's debate on the necessity of correct implementation to avoid unpredictable results, with recommendations to gain minimal knowledge through UI experimentation.

- **Diffusers Pipeline Limitations**: Some users point out limitations in the diffusers pipeline requiring Hugging Face dependency, yet others contend that once models are downloaded, the process can run independently and efficiently on local systems. Concerns are raised about the inaccessibility of `StableVideoDiffusionPipeline.from_single_file(path)` method in SVD finetunes, suggesting Comfy UI as an easier alternative.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://comfyanonymous.github.io/ComfyUI_examples/video/">Video Examples</a>: Examples of ComfyUI workflows</li><li><a href="https://comfyanonymous.github.io/ComfyUI_examples/model_merging/#advanced-merging">Model Merging Examples</a>: Examples of ComfyUI workflows</li><li><a href="https://huggingface.co/spaces/multimodalart/stable-cascade">Stable Cascade - a Hugging Face Space by multimodalart</a>: no description found</li><li><a href="https://huggingface.co/spaces/PixArt-alpha/PixArt-Sigma">PixArt Sigma - a Hugging Face Space by PixArt-alpha</a>: no description found</li><li><a href="https://huggingface.co/camenduru/SUPIR">camenduru/SUPIR · Hugging Face</a>: no description found</li><li><a href="https://stability.ai/news/stable-diffusion-3-api">Stable Diffusion 3 API Now Available &mdash; Stability AI</a>: We are pleased to announce the availability of Stable Diffusion 3 and Stable Diffusion 3 Turbo on the Stability AI Developer Platform API.&amp;nbsp;</li><li><a href="https://stability.ai/membership">Membership &mdash; Stability AI</a>: The Stability AI Membership offers flexibility for your generative AI needs by combining our range of state-of-the-art open models with self-hosting benefits.</li><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/svd">Stable Video Diffusion</a>: no description found</li><li><a href="https://github.com/kijai/ComfyUI-SUPIR">GitHub - kijai/ComfyUI-SUPIR: SUPIR upscaling wrapper for ComfyUI</a>: SUPIR upscaling wrapper for ComfyUI. Contribute to kijai/ComfyUI-SUPIR development by creating an account on GitHub.</li><li><a href="https://github.com/victorsungo/WizardLM/tree/main/WizardLM-2">WizardLM/WizardLM-2 at main · victorsungo/WizardLM</a>: Family of instruction-following LLMs powered by Evol-Instruct: WizardLM, WizardCoder - victorsungo/WizardLM</li><li><a href="https://www.reddit.com/r/StableDiffusion/s/lUYMRFOvcF">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/king159/svd-mv">GitHub - king159/svd-mv: Training code for Stable Video Diffusion Multi-View</a>: Training code for Stable Video Diffusion Multi-View - king159/svd-mv</li><li><a href="https://new.reddit.com/r/LocalLLaMA/comments/1c586rm/wizardlm2_was_deleted_because_they_forgot_to_test/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/BatouResearch/magic-image-refiner">GitHub - BatouResearch/magic-image-refiner</a>: Contribute to BatouResearch/magic-image-refiner development by creating an account on GitHub.</li><li><a href="https://github.com/ExponentialML/ComfyUI_ELLA/pull/25">Fix ELLA timesteps by kijai · Pull Request #25 · ExponentialML/ComfyUI_ELLA</a>: I have been comparing the results from this implementation to the diffusers implementation, and it&#39;s not on par. In diffusers ELLA is applied on each timestep, with the actual timestep value. Appl...</li><li><a href="https://civitai.com/models/120096/pixel-art-xl">Pixel Art XL - v1.1 | Stable Diffusion LoRA | Civitai</a>: Pixel Art XL Consider supporting further research on Ko-Fi or Twitter If you have a request, you can do it via Ko-Fi Checkout my other models at Re...</li><li><a href="https://github.com/kijai/ComfyUI-KJNodes/">GitHub - kijai/ComfyUI-KJNodes: Various custom nodes for ComfyUI</a>: Various custom nodes for ComfyUI. Contribute to kijai/ComfyUI-KJNodes development by creating an account on GitHub.</li><li><a href="https://github.com/city96/ComfyUI_ExtraModels">GitHub - city96/ComfyUI_ExtraModels: Support for miscellaneous image models. Currently supports: DiT, PixArt, T5 and a few custom VAEs</a>: Support for miscellaneous image models. Currently supports: DiT, PixArt, T5 and a few custom VAEs - city96/ComfyUI_ExtraModels</li><li><a href="https://github.com/kijai/ComfyUI-KJNodes/commit/22cf8d89968a47ce26be919f750f2311159145d1">Add node to use SD3 through API · kijai/ComfyUI-KJNodes@22cf8d8</a>: no description found</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://comfyworkflows.com">Comfy Workflows</a>: Share, discover, &amp; run thousands of ComfyUI workflows.
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1229696690315989032)** (383 messages🔥🔥): 

- **GPT-4 and GPT-3.5 Clarification**: A distinction was made between GPT-4 and GPT-3.5, noting that the newer version appears to be a fine-tuned iteration of its predecessor.
- **Mistral Model Multilingual Capabilities Discussed**: Members discussed whether datasets for **Mistral7B** need to be in English to perform well, with advice given to include French data for better results.
- **Finetuning and Cost Concerns Addressed**: A discussion about finetuning methods, costs, and specific resources like notebooks provided insights for those new to the domain. It was suggested that **continued pretraining and sft could be beneficial and cost-effective**.
- **Concerning UnSloth Contributions**: Members expressed interest in contributing to **UnSloth AI**, offering help in expanding documentation and considering donations, with links to existing resources and discussions on potential contributions shared.
- **Mixtral 8x22B Release Excitement**: The release of **Mixtral 8x22B**, a sparse Mixture-of-Experts model with strengths in multilingual fluency and long context windows, sparks discussions due to its open-sourcing under the **Apache 2.0 license**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mixtral-8x22b/">Cheaper, Better, Faster, Stronger</a>: Continuing to push the frontier of AI and making it accessible to all.</li><li><a href="https://www.amazon.com/NVIDIA-Tesla-M40-24GB-Module/dp/B01HGJGJWU/ref=sr_1_1?sr=8-1">no title found</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/lucyknada/microsoft_WizardLM-2-7B">lucyknada/microsoft_WizardLM-2-7B · Hugging Face</a>: no description found</li><li><a href="https://www.kabum.com.br/produto/359038/placa-de-video-galax-nvidia-geforce-rtx-3090-ti-ex-gamer-24gb-gddr6x-384-bits-39ixm5md6hex">Placa de Vídeo Galax NVIDIA GeForce RTX 3090 TI EX Gamer, 24GB GDDR6X, 384 Bits - 39IXM5MD6HEX</a>: Placa De Video Galax GeforceTorne sua rotina diária mais fluída Assine o Prime Ninja e tenha promoções exclusivas desconto no frete e cupons em dobro</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 2-5X faster 80% less memory LLM finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.</li><li><a href="https://gist.github.com/jedt/e45b337e9d9bd0492bf5d3c1d4706c7b">gist:e45b337e9d9bd0492bf5d3c1d4706c7b</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://youtu.be/SL2nZpv7dtY?si=Yw5JxlVhRTrBu1gA">Full fine tuning vs (Q)LoRA</a>: ➡️ Get Life-time Access to the complete scripts (and future improvements): https://trelis.com/advanced-fine-tuning-scripts/➡️ Runpod one-click fine-tuning te...</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://github.com/openai/triton/issues/194">Support for x86/ARM CPUs (e.g., Xeon, M1) · Issue #194 · openai/triton</a>: Hi there, Is there any future plan for macOS support? ❯ pip install -U --pre triton DEPRECATION: Configuring installation scheme with distutils config files is deprecated and will no longer work in...</li><li><a href="https://github.com/ollama/ollama/pull/3699">Ollama.md Documentation  by jedt · Pull Request #3699 · ollama/ollama</a>: A guide on setting up a fine-tuned Unsloth FastLanguageModel from a Google Colab notebook to:  HF hub GGUF local Ollama  Preview link: https://github.com/ollama/ollama/blob/66f7b5bf9e63e1e98c98e8f4...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1229745120740380792)** (27 messages🔥): 

- **Chroma Project Takes a Leap**: Inspired by unsloth AI strategies, a member announces the development of an edge version of **Chroma** written in Go, using SQLite for on-device vector storage. The project, which is also compatible with browsers via **WASM**, is accessible on [GitHub](https://github.com/l4b4r4b4b4/go-chroma).

- **Smileys Invade the Bottom Page**: A heartwarming mini-discussion about cute smiley faces at the bottom of a page, highlighting a particular *mustache smiley* as a favorite.

- **PyTorch’s New Torchtune**: Mention of **Torchtune**, a native PyTorch library for LLM fine-tuning that has been shared on GitHub, sparking interest due to its potential to make fine-tuning more accessible.

- **Unsloth AI's Broad GPU Support Praised**: A member congratulates Unsloth for its broad GPU support, which makes it more accessible compared to other tools that require newer GPU architectures.

- **Mobile Deployment of AI Models Discussed**: Members discuss the feasibility of running neural networks on mobile phones, identifying the need for custom inference engines and noting the absence of CUDA on mobile devices. The challenges of running typical DL Python code on iPhones versus Macs with M chips are also mentioned.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune">GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/l4b4r4b4b4/go-chroma">GitHub - l4b4r4b4b4/go-chroma: Go port of Chroma vector storage</a>: Go port of Chroma vector storage. Contribute to l4b4r4b4b4/go-chroma development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1229688010010923008)** (275 messages🔥🔥): 

- **Questions About Unsupported Attributes**: A user encountered an `AttributeError` when trying to fine-tune a model, reporting that the `'MistralSdpaAttention'` object has no attribute `'temp_QA'`. It seems to be related to a specific method within their custom training pipeline.
- **ORPO Support and Usage Clarified**: Users inquired about ORPO support in Unsloth. It's confirmed that **ORPO** is supported, referenced by links to a model trained using ORPO on HuggingFace and a [colab notebook](https://colab.research.google.com/drive/1U_p7-qFfOm4v-TIrs1wK5eEODg1HUcGB?usp=sharing).
- **Discussions on LoRA and rslora**: Users discussed using **LoRA and rslora** in training, with advice on handling different `alpha` values and potential loss spikes. Some members suggested adjusting `r` and `alpha` and disabling packing as possible solutions to training issues.
- **Embedding Tokens Not Trained**: Users touched on the subject of **embedding tokens that were not trained** in the Mistral model, in the context of whether it is possible to train these embeddings during fine-tuning.
- **Saving and Hosting Models**: Questions arose about saving finetuned models in different formats using commands like `save_pretrained_merged` and `save_pretrained_gguf`; whether they work sequentially and the need to start with fp16 first. There was also a query about hosting a model with GGUF files on the HuggingFace inference API.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/datasets">Find Open Datasets and Machine Learning Projects | Kaggle</a>: Download Open Datasets on 1000s of Projects &#x2B; Share Projects on One Platform. Explore Popular Topics Like Government, Sports, Medicine, Fintech, Food, More. Flexible Data Ingestion.</li><li><a href="https://discord.gg/82UfKN7z">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://discord.gg/s8sdX5DB">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://huggingface.co/G-reen/EXPERIMENT-ORPO-m7b2-1-merged">G-reen/EXPERIMENT-ORPO-m7b2-1-merged · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing#scrollTo=QtoqUw80QDV0)?">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1g9kHV3tc6P2cUp9gVPurKUZmiFqeb3kv">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1U_p7-qFfOm4v-TIrs1wK5eEODg1HUcGB?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms">Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)</a>: Things I Learned From Hundreds of Experiments</li><li><a href="https://unsloth.ai/blog/long-context">Unsloth - 4x longer context windows &amp; 1.7x larger batch sizes</a>: Unsloth now supports finetuning of LLMs with very long context windows, up to 228K (Hugging Face + Flash Attention 2 does 58K so 4x longer) on H100 and 56K (HF + FA2 does 14K) on RTX 4090.  We managed...</li><li><a href="https://docs.mistral.ai/guides/tokenization/#control-tokens">Tokenization | Mistral AI Large Language Models</a>: Tokenization is a fundamental step in LLMs. It is the process of breaking down text into smaller subword units, known as tokens. We recently open-sourced our tokenizer at Mistral AI. This guide will w...</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-the-lm_head-and-embed_tokens-matrices)">Home</a>: 2-5X faster 80% less memory LLM finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.</li><li><a href="https://arxiv.org/abs/2312.03732">A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA</a>: As large language models (LLMs) have become increasingly compute and memory intensive, parameter-efficient fine-tuning (PEFT) methods are now a common strategy to fine-tune LLMs. A popular PEFT method...</li><li><a href="https://huggingface.co/blog/damjan-k/rslora">Rank-Stabilized LoRA: Unlocking the Potential of LoRA Fine-Tuning</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/installation#offline-mode">Installation</a>: no description found</li><li><a href="https://huggingface.co'">no title found</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/installation#offline-mode'.">Installation</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>: 2-5X faster 80% less memory LLM finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.</a>: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI</li><li><a href="https://github.com/unslothai/unsloth/issues/331">Add ORPO example notebook to the docs · Issue #331 · unslothai/unsloth</a>: It&#39;s possible to use the ORPOTrainer from TRL with very little modification to the current DPO notebook. Since ORPO reduces the resources required for training chat models even further (no separat...</li><li><a href="https://huggingface.co/docs/transformers/main_classes/tokenizer">Tokenizer</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/transformers/issues/1413#issuecomment-538083512">Adding New Vocabulary Tokens to the Models · Issue #1413 · huggingface/transformers</a>: ❓ Questions &amp; Help Hi, How could I extend the vocabulary of the pre-trained models, e.g. by adding new tokens to the lookup table? Any examples demonstrating this?
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1229739881110966362)** (46 messages🔥): 

- **Clarification on Leaderboard Model Templates**: A member asked how the leaderboard knows the model template. It was clarified that the model's `tokenizer.chat_template` is used to inform the leaderboard.
- **ShareGPT90k Dataset Cleaned and Formatted**: A new version of the **ShareGPT90k** dataset has been cleaned of HTML tags and is available in chatml format on Hugging Face, allowing users to train with Unsloth. [Dataset ready for action](https://huggingface.co/datasets/pacozaa/sharegpt90k-cleanned).
- **Ghost Model Training Intrigue**: Members engaged in a detailed conversation about what constitutes a 'recipe' for training AI models. One member is particular about needing a detailed recipe that leads to creating a specific model with defined characteristics and not just a set of tools or methods.
- **Recipes vs. Tools in AI Model Training**: The conversation continued on the difference between a full "recipe" including datasets and specific steps, as opposed to tools and methods. One member shared their approach, underlying the importance of data quality and replication of existing models, referencing the Dolphin model card on Hugging Face.
- **Recommender Systems vs. NLP Challenges and Expertise**: A PhD candidate discussed the differences and similarities between working on NLP and developing recommender systems, highlighting the unique challenges and expertise required in the latter which includes handling noise in data, induction biases, and significant feature engineering.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms">Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)</a>: Things I Learned From Hundreds of Experiments</li><li><a href="https://tenor.com/view/nice-click-nice-man-guy-gif-21933845">Nice Click Nice GIF - Nice Click Nice Man - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=QXVCqtAZAn4&t=26s">Aligning LLMs with Direct Preference Optimization</a>: In this workshop, Lewis Tunstall and Edward Beeching from Hugging Face will discuss a powerful alignment technique called Direct Preference Optimisation (DPO...</li><li><a href="https://www.youtube.com/watch?v=hvGa5Mba4c8&t=5s">Direct Preference Optimization (DPO) explained: Bradley-Terry model, log probabilities, math</a>: In this video I will explain Direct Preference Optimization (DPO), an alignment technique for language models introduced in the paper &quot;Direct Preference Opti...</li><li><a href="https://www.youtube.com/watch?v=MJnIxpZhTk0).">FractalFormer: A WIP Transformer Architecture Inspired By Fractals</a>: Check out the GitHub repo herehttps://github.com/evintunador/FractalFormerSupport my learning journey on patreon!https://patreon.com/Tunadorable?utm_medium=u...</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes%2F">llama-recipes/recipes at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081 · meta-llama/llama-recipes</a>: Scripts for fine-tuning Llama2 with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization &amp;amp; ...</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes%2Fmultilingual%2FREADME.md">llama-recipes/recipes/multilingual/README.md at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081 · meta-llama/llama-recipes</a>: Scripts for fine-tuning Llama2 with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization &amp;amp; ...</li><li><a href="https://arxiv.org/abs/2303.14617">Neural Graph Reasoning: Complex Logical Query Answering Meets Graph Databases</a>: Complex logical query answering (CLQA) is a recently emerged task of graph machine learning that goes beyond simple one-hop link prediction and solves a far more complex task of multi-hop logical reas...</li><li><a href="https://www.youtube.com/watch?v=wzKW4P4dg1o">LLM Phase Transition: New Discovery</a>: Phase Transitions in a dot-product Attention Layer learning, discovered by Swiss AI team. The study of phase transitions within the attention mechanisms of L...</li><li><a href="https://huggingface.co/datasets/pacozaa/sharegpt90k-cleanned">pacozaa/sharegpt90k-cleanned · Datasets at Hugging Face</a>: no description found</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/16zuccy/after_500_loras_made_here_is_the_secret/).">After 500+ LoRAs made, here is the secret</a>: Well, you wanted it, here it is: The quality of dataset is 95% of everything. The rest 5% is not to ruin it with bad parameters. Yeah, I know,...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1230101881544773653)** (15 messages🔥): 

- **Exploring Multilingual Model Approaches**: A member brought up the issue of *catastrophic forgetting* in multilingual models trained on languages like Hindi or Thai. They proposed a two-phase solution involving translating questions to English, using a large English model for answering, and then translating back to the original language, questioning the drawbacks of this method.
- **Multilingual Expansion Through MOE**: Another member expressed excitement about the possibility of using MoE (Mixture of Experts) to expand multilingual capabilities of models, anticipating it would *“open so many doors!”*
- **Torchtune Gains Enthusiasm**: The community shows interest in **Torchtune**, an alternative to the abstractions provided by Hugging Face and Axolotl, highlighting its potential to streamline the fine-tuning process. There is also a hint at possible collaborations involving Unsloth AI.
- **Contemplating Language Mixing in Datasets**: In response to the splitting of translation and question-answering tasks, a member considered the possibility of combining multiple languages into a single dataset for model training and using a strategy that involves priming the model with Wikipedia articles.
- **Double-Translation Mechanism Discussed**: A concept articulated as `translate(LLM(translate(instruction)))` was proposed and discussed, supporting the idea of using a larger, more robust English language model in tandem with translation layers to process non-English queries. Concerns about the added cost due to multiple model calls were raised.
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1229711456308691015)** (175 messages🔥🔥): 

- **Repeat AI Responses Challenge:** A member asked how to prevent AI from repeating the same information during a conversation, specifically using Dolphin 2 Mistral. They also inquired about what "multi-turn conversations" are, to which another member linked an article explaining the concept in relation to bots.
- **WizardLM-2 LLM Announced:** An announcement for the new large language model family was shared, featuring WizardLM-2 8x22B, 70B, and 7B. Links to a [release blog](https://wizardlm.github.io/WizardLM2) and [model weights on Hugging Face](https://huggingface.co/collections/microsoft/wizardlm-661d403f71e6c8257dbd598a) were included, with members discussing its availability and performance.
- **Understanding Tool Differences:** One user asked for the differences between ollama and LMStudio, and it was explained that both are wrappers for llama.cpp, but LM Studio is GUI based and easier for beginners.
- **Fine-Tuning and Agents Discussion:** There was a discussion on whether it's worth learning tools like langchain depending on needs and use cases, with some suggesting it can be a hindrance if venturing outside its default settings.
- **File Management and API Interactions in LM Studio:** A new member inquired about relocating downloaded app files and interfacing LM Studio with an existing API. It was clarified that models cannot change default install locations, and files can be found under the My Models tab for relocating. No specific method for API interaction through LM Studio was mentioned.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notkyon.moe/ram-latency2.htm">RAM Latency Calculator</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/WizardLM-2-7B-GGUF">lmstudio-community/WizardLM-2-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/azure/ai-services/qnamaker/how-to/multi-turn#what-is-a-multi-turn-conversation">Multi-turn conversations - QnA Maker - Azure AI services</a>: Use prompts and context to manage the multiple turns, known as multi-turn, for your bot from one question to another. Multi-turn is the ability to have a back-and-forth conversation where the previous...</li><li><a href="https://missionsquad.ai">Mission Squad. Flexible AI agent desktop app.</a>: no description found</li><li><a href="https://x.com/WizardLM_AI/status/1779899325868589372">Tweet from WizardLM (@WizardLM_AI)</a>: 🔥Today we are announcing WizardLM-2, our next generation state-of-the-art LLM.  New family includes three cutting-edge models: WizardLM-2 8x22B, 70B, and 7B - demonstrates highly competitive performa...</li><li><a href="https://github.com/hiyouga/LLaMA-Factory">GitHub - hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs</a>: Unify Efficient Fine-Tuning of 100+ LLMs. Contribute to hiyouga/LLaMA-Factory development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=Box3pQ1HNuQ&t=315s">Microsoft’s Punch in the Face to Open AI (Open Source &amp; Beats GPT-4)</a>: WizardLM 2 is a groundbreaking family of large language models developed by Microsoft that push the boundaries of artificial intelligence.▼ Link(s) From Toda...</li><li><a href="https://github.com/Unstructured-IO/unstructured/">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.  - GitHub - Unstructured-IO/unstructured: Open source librar...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1229748345757630474)** (96 messages🔥🔥): 

- **Template Troubles with WizardLM 2**: Members reported issues with the WizardLM 2 and the Vicuna 1.5 preset, where the bot generated inputs for the user instead. A suggested solution included adjusting the rope frequency to 1 or setting `freq_base` to 0, which appeared to correct the behavior.
- **Mixed Opinions on WizardLM 2 and Wavecoder**: While some users expressed a high opinion of WizardLM 2, claiming it performed well even compared to other 7B models, others judged the performance as subpar, not noticing any significant improvement even after fine-tuning.
- **Exploring Best Quantization Practices**: Users discussed the effectiveness of different quantization levels for 7B models, comparing Q8 to Q6K quality. The consensus leaned towards higher quantization being more desirable if one has sufficient VRAM, while acknowledging the utility of smaller models for certain tasks.
- **Model Performance Debate**: There was a spirited discussion around the relative superiority of models, with focus on parameter count versus quantization level, and the belief that fine-tuning and quality of the training can be deciding factors over just the size of the model's parameters.
- **Finding the Right Code Generator**: A user experienced difficulties with the code-generating capabilities of WaveCoder-Ultra-6.7B, receiving messages that it couldn't write complete applications. Tips offered included using assertive prompts and adjusting the context window size for the model to load appropriately.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF#prompt-template>">lmstudio-community/wavecoder-ultra-6.7b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Virt-io/Google-Colab-Imatrix-GGUF/tree/main">Virt-io/Google-Colab-Imatrix-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/collections/DavidAU">High Quality / Hard to Find - a DavidAU Collection</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/WizardLM-2-7B-GGUF">lmstudio-community/WizardLM-2-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://rentry.co/4q4h7pw6">Responses</a>: These are answers to the prompt by two different LLMs. You are going to analyze  Factuality Depth Level of detail Coherency &lt;any other area that I might have missed but is generally considered impo...</li><li><a href="https://huggingface.co/bartowski/zephyr-orpo-141b-A35b-v0.1-GGUF/tree/main">bartowski/zephyr-orpo-141b-A35b-v0.1-GGUF at main</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1230041327832666154)** (4 messages): 

- **Model Loading Error in Action**: A user encountered an *error loading model architecture* when trying out **Wirard LLM 2** on LM Studio across different model sizes, including 2 4bit and 6bit, prompting a *Failed to load model* message.
  
- **Fix Suggestion for Model Loading**: Another user recommended ensuring the use of **GGUF quants** and also noted that version **0.2.19** is required for **WizardLM2** models to function properly.

- **Request for stable-diffusion.cpp**: A request was made to add **stable-diffusion.cpp** to **LM Studio** to enhance the software's capabilities.
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1229701538889596980)** (17 messages🔥): 

- **Cleaning Up LM Studio**: Users with issues were advised to delete specific LM Studio folders such as `C:\Users\Username\.cache\lm-studio`, `C:\Users\Username\AppData\Local\LM-Studio`, and `C:\Users\Username\AppData\Roaming\LM Studio`. It's crucial to **backup models and important data** prior to deletion.
- **Prompt Crafting for NexusRaven**: A user inquired if anyone has experimented with NexusRaven and devised any prompt presets for it, indicating interest in collective knowledge-sharing.
- **Script Writing with AI**: One member asked how to make the AI output a full script, suggesting they are searching for tips on generating longer content.
- **Compatibility Issues with Hugging Face Models**: A user noted problems with running certain Hugging Face models, like `changge29/bert_enron_emails` and `ktkeller/mem-jasper-writer-testing`, in LM Studio. Assistance with running these models was sought.
- **Seeking Partnership for Affiliate Marketing**: A user indicated interest in finding a partner with coding expertise for help with affiliate marketing campaigns, mentioning a willingness to share profits if successful. The user emphasized a serious offer for a partnership based on results.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1229729771949133825)** (18 messages🔥): 

- **GPU Comparison Sheet Quest Continues**: User **freethepublicdebt** was searching for an elusive Google sheet comparing GPUs and could not find the link they worked on. Another user, heyitsyorkie, attempted to help but provided the wrong link leading to further confusion.
- **Direct GPU Communication Breakthrough**: rugg0064 shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1c4gakl/got_p2p_working_with_4x_3090s) celebrating the success of getting GPUs to communicate directly, bypassing the CPU/RAM and potentially leading to performance improvements.
- **Customizing GPU Load in LM Studio**: heyitsyorkie provided insight on adjusting the GPU offload for models in LM Studio's Linux beta by navigating to **Chat mode -> Settings Panel -> Advanced Config**.
- **Splitting Workloads Between Different GPUs**: In response to a query from .spicynoodle about uneven model allocation between their GPUs, heyitsyorkie suggested modifying **GPU preferences json** and searching for "tensor_split" for further guidance.
- **SLI and Nvlink Troubles with P100s**: ethernova is seeking advice for their setup with dual P100s not showing up in certain software and NVLink status appearing inactive despite having NVLink bridges attached.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/george-hotz-geohot-money-rain-gif-6469921471081342358">George Hotz Geohot GIF - George hotz Geohot Money - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c4gakl/got_p2p_working_with_4x_3090s">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1229813824778014902)** (31 messages🔥): 

- **VRAM vs. System RAM in Model Performance**: There's a discussion on whether a model would run on a system with **24 GB VRam** and **96 GB Ram**, with one member suggesting that it *might* run but **inference will be incredibly slow** due to the speed difference between VRam and system RAM.
- **Expectations for WizardLM-2-8x22B**: Members compare the **WizardLM-2-8x22B** to other models like **Command R Plus**, with mixed experiences. While one member was not impressed with **Mixtral 8x22b** and plans to test WizardLM-2-8x22B, another mentioned getting satisfactory results with 10+ tokens/sec from WizardLM.
- **Model Performance on Different Hardware**: Users with an M3 MacBook Pro 128GB report running model **q6_k** of **Command R Plus**, achieving about **5 tokens/sec**. The speed is considered half as fast as GPT-4 on ChatGPT, but not painfully slow as each token represents a word or subword.
- **Base Model Clarification**: Clarification on what constitutes a "Base" model was provided—models not fine-tuned for chat or instruct tasks are considered base models, and they are generally found to perform poorly in comparison to their fine-tuned counterparts.
- **Model Size and Local Running Feasibility**: A conversation about the feasibility of running large models like **WizardLM-2-8x22B** locally was had, noting that a GPU like a 4090 with 24GB is **too small to run such a large model**, which runs best on Mac systems with substantial RAM.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1229814103292383302)** (19 messages🔥): 

- **Curiosity about Windows Executable Signing**: A member was curious whether the Windows executables are signed with an [Authenticode cert](https://docs.microsoft.com/en-us/windows-hardware/drivers/install/authenticode). It was confirmed that they are indeed signed.
- **Challenges with Code Signing Certificates**: In the context of signing an app, there was a discussion on the cost and process complexities associated with obtaining a Windows certificate, including a comparison to the cost of an Apple developer license.
- **Seeking Expertise on Automated Compile and Sign Process**: A member expressed interest in understanding the automated process for compiling and signing, offering to compensate for the knowledge exchange.
- **AMD HIP SDK System Requirements Clarification**: A member provided information about system requirements for GPUs from a [link to the AMD HIP SDK system requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html) and asked about the stance of LM Studio on supporting certain AMD GPUs not officially supported by the SDK.
- **Issues with AMD dGPU Recognition in LM Studio Software**: Members discussed an issue where LM Studio software was using an AMD integrated GPU (iGPU) instead of the dedicated GPU (dGPU), with one member suggesting disabling the iGPU in the device manager. Another member stated that version 0.2.19 of the software should have resolved this issue and encouraged to report the problem if it persists.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/rocm">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html">System requirements (Windows) — HIP SDK installation Windows</a>: no description found</li><li><a href="https://tenor.com/view/bill-gates-chair-jump-microsoft-chairjump-gif-5558594">Bill Gates Chair GIF - Bill Gates Chair Jump - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1229844206596522025)** (3 messages): 

- **WaveCoder Ultra Unveiled**: Microsoft has released **WaveCoder ultra 6.7b**, finely tuned using their 'CodeOcean'. This impressive model specializes in code translation and supports the Alpaca format for instruction following, with examples available on [its model card](https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF).
- **Seeking NeoScript AI Assistant**: A community member new to AI has inquired about utilizing models for NeoScript programming, specifically for RAD applications using a platform formerly known as NeoBook. They are seeking suggestions on configuring AI models despite unsuccessful initial attempts using documents as references.

**Link mentioned**: <a href="https://huggingface.co/lmstudio-community/wavecoder-ultra-6.7b-GGUF">lmstudio-community/wavecoder-ultra-6.7b-GGUF · Hugging Face</a>: no description found

  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1229691059458539530)** (17 messages🔥): 

- **Introducing Multimodal Chat GPTs**: A link to a YouTube video titled "Introducing Idefics2 8B: Open Multimodal ChatGPT" was shared, discussing the development of Hugging Face's open multimodal language model, Idefics2. Watch it [here](https://www.youtube.com/watch?v=vL1SayPCHBg).
- **Reka Core Joins the Multimodal Race**: Another YouTube video shared discusses "Reka Core," a competitive multimodal language model claiming to rival big industry names like OpenAI, Anthropic, and Google. The video can be viewed [here](https://www.youtube.com/watch?v=U7RbwPKyxs8).
- **Navigating Language and AI**: Discussions revolved around the relationship between language, AI, and the concept of the divine, touching on the idea of languages as "envelopes within the vectorspace of meaning" and the potential linguistic evolution that AI might spur. The conversation included references to general semantics and quantum mereotopology with a hint at looking into Alfred Korzybski's work.
- **Staying Up to Date with AI Research**: Members expressed the challenge of keeping up with the vast amount of AI research and literature, admitting to struggles with growing reading backlogs amidst the rapid pace of new publications.
- **JetMoE and the Economics of AI**: A YouTube video titled "JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars" highlighting how JetMoE-8B was trained on a budget yet outperforms the more expensive LLaMA2-7B was shared. The video is available [here](https://www.youtube.com/watch?v=Z9Hwp_XeS1A).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=p9T7ZgtM5Mo">Snowflake Launches the World’s Best Practical Text-Embedding Model</a>: Today Snowflake is launching and open-sourcing with an Apache 2.0 license the Snowflake Arctic embed family of models. Based on the Massive Text Embedding Be...</li><li><a href="https://www.youtube.com/watch?v=vL1SayPCHBg">Introducing Idefics2 8B: Open Multimodal ChatGPT</a>: We will take a look idefics2 the open multimodal llm by huggingfacehttps://huggingface.co/blog/idefics2#python #pythonprogramming #llm #ml #ai #aritificialin...</li><li><a href="https://www.youtube.com/watch?v=U7RbwPKyxs8">Reka Core: A Frontier Class Multimodal Language Model</a>: Reka Core is competitive with models from OpenAI, Anthropic, and Google across key industry-accepted evaluation metrics. Given its footprint and performance,...</li><li><a href="https://www.youtube.com/watch?v=Z9Hwp_XeS1A">JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars</a>: JetMoE-8B is trained with less than $ 0.1 million1 cost but outperforms LLaMA2-7B from Meta AI, who has multi-billion-dollar training resources. LLM training...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1230058492174401537)** (7 messages): 

- **Self-Supervised LLM Solution Selection Sprouts**: A novel technique for **enhancing Instruct Model LLMs** is on the table, which utilizes the model’s own capacity to generate and select the most pertinent solution based on its ability to reconstruct the original input from its responses. The method aims at information maximization and offers a scalable, unsupervised evaluation that enhances coherence and relevance, and is adaptable with existing techniques.

- **New Horizons in LLM Medical Alignment**: A shared [Google Slideshow](https://docs.google.com/presentation/d/1dk2ekDPa9qFuT4B0WafaZLRso5YdTpgv9FaOEQ_lNvs/edit?usp=sharing) points towards efforts in aligning **Language Models** specifically for medical reasoning applications, although the content details are not accessible from the provided message.

- **Mistral's Tokenization Guide Unwrapped**: [Mistral AI introduces an open-source tokenizer](https://docs.mistral.ai/guides/tokenization/), with a guide discussing the tokenization process, its importance in LLMs, and how to employ their tokenizer within Python.

- **Tempering the Tokenization Hype**: A user critiques the emphasis on tokens, arguing that tokens aren't as critical if the model is already adept at handling tags, suggesting that the true value might be in increased steerability of the model.

- **Tweeting Up a Dev Storm**: A link to a [Twitter post](https://twitter.com/OpenAIDevs/status/1780640119890047475) was shared, but the content of the tweet hasn't been discussed within the provided messages.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.mistral.ai/guides/tokenization/">Tokenization | Mistral AI Large Language Models</a>: Tokenization is a fundamental step in LLMs. It is the process of breaking down text into smaller subword units, known as tokens. We recently open-sourced our tokenizer at Mistral AI. This guide will w...</li><li><a href="https://docs.google.com/presentation/d/1dk2ekDPa9qFuT4B0WafaZLRso5YdTpgv9FaOEQ_lNvs/edit?usp=sharing">Aligning LLMs for Medical Reasoning</a>: Aligning Large Language Models to be Better Medical Reasoners Ritabrata Maiti ritabrat001@e.ntu.edu.sg 1
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1229693380511010868)** (159 messages🔥🔥): 

- **Mystery Surrounding WizardLM's Takedown**: There was confusion about why Microsoft's **WizardLM** was taken down, with speculation about it being "too toxic" and unverified rumors of it being attacked or hacked. A [bundle of links and information](https://huggingface.co/alpindale/WizardLM-2-8x22B) about **WizardLM** was shared including its removal and a re-upload mirror.

- **Concerns about the EU AI Act**: A theory was put forward that **WizardLM** had to be taken down as it violated the EU AI act for being almost uncensored, with suggestions to torrent the original version if anyone still has it. However, it was clarified later that it was originally unpublished for not going through Microsoft's new "toxicity review."

- **Excitement and Skepticism for Code Models**: Discussion on **CodeQwen1.5-7B Chat**, a code-specific language model, was lively with members sharing its [blog post and GitHub](https://qwenlm.github.io/blog/codeqwen1.5/) while noting its strong performance on benchmarks like 83.5 on humaneval. There is some skepticism about the model still using vanilla MHA (Multihead Attention) and speculation about potential contamination due to its high performance.

- **Frustrations with Mixed Messages on Model Performance**: **n8programs** shared excitement for improvements to a creative writing model achieving a benchmark score of 70, between Mistral medium and large, using **Westlake** as a base model. The legitimacy of benchmark comparisons was debated, especially in light of expectations for **LLaMa 3** and whether explicit tuning can trump new architectures.

- **Uncertainty about Future Model Releases**: Queries about upcoming releases like **Hermes 8x22B** and whether it would be realistic to run such large models on personal equipment. There is anticipation about potential **Llama-3** models and speculation on whether these new models will outperform their predecessors.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/senseable/WestLake-7B-v2">senseable/WestLake-7B-v2 · Hugging Face</a>: no description found</li><li><a href="http://www.incompleteideas.net/IncIdeas/BitterLesson.html">The Bitter Lesson</a>: no description found</li><li><a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat">Qwen/CodeQwen1.5-7B-Chat · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b">WizardLM-2 8x22B by microsoft | OpenRouter</a>: WizardLM-2 8x22B is Microsoft AI&#x27;s most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing ...</li><li><a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>: How to build Long term memory &amp; Self improving ability into your AI Agent?Use AI Slide deck builder Gamma for free: https://gamma.app/?utm_source=youtube&amp;utm...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1229838374316015636)** (7 messages): 

- **Speed Demon**: A member mentioned witnessing a performance of **700 Mbps** in an unnamed context.
- **Diving into State-Space Models**: A member sought recommendations for essential papers on recent advances in **state-space models** for weekend reading.
- **Mamba Paper Suggested**: In response to a request for recent literature, one member suggested looking into the **Mamba** paper, while another was more interested in the newer **Jamba** and related works.
- **Hermes 2 Pro Query Handling Issue**: A user expressed the need to prevent **Hermes 2 Pro** from always returning `<tool_call>` when it should sometimes just engage in chat, noting it as a current limitation.
- **Promising Future Updates**: A contributor noted they will collaborate with another member to improve **Hermes 2 Pro**'s ability to discern when to use `<tool_call>` and when to just chat in future versions.
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1229822599454920826)** (10 messages🔥): 

- **Debating JSON's Virtue**: A message refers to a previous defense for using JSON structure for input-outputs, suggesting that this format might reduce the need for *handwaving* when explaining processes.

- **Seeking Vision for RAGs**: A user expressed interest in the state of the art for vision, especially in the context of building a **Retrieval Augmented Generation (RAG)** on engineering documents with images and diagrams.

- **Vision SOTA Suggestions**: One member touted **GPT4v/Geminipro Vision** and **Claude Sonnet** as leading options in the field, recommending testing them against each other for specific use cases.

- **Turning to Open Source**: When seeking open-source alternatives, suggestions included **llava**, **cogvlm**, **mPlug-DocOwl**, and **donut**, with **mPlug-DocOwl** being specifically recommended for **DocVQA** use cases.

- **Exploring Supersizing LLMs**: A member shared a [blog post](https://blog.normalcomputing.ai/posts/2023-09-12-supersizing-transformers/supersizing-transformers.html) discussing the use of LLMs beyond token sequencing, emphasizing the need for models that perform complex reasoning and fetch accurate, topical information.

**Link mentioned**: <a href="https://blog.normalcomputing.ai/posts/2023-09-12-supersizing-transformers/supersizing-transformers.html">The Normal Blog - Infinite Context LLMs: Going Beyond RAG with Extended Minds</a>: In this blog we discuss how the transformer architecture naturally extends over external memories, and share empirical results which leverage this capability to succeed where RAG has struggled. These ...

  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1229685880848125983)** (159 messages🔥🔥): 

- **World-Sim Anticipation Builds**: Members express excitement and impatience as World-Sim's return is discussed with speculative launch times, the concept's philosophical underpinnings, and whether AI aspires to godhood. A member provided the link to the Nous Research blog post to delve deeper into this topic: [Divinity in AI](https://nousresearch.com/dsjjjj-simulacra-in-the-stupor-of-becoming/).

- **Jailbroken Prometheus Draws Interest**: The chat mentions an alternative to World-Sim, web-based Jailbroken Prometheus, sparking curiosity among users. For those looking for similar experiences, a member shared a [Websim link](https://websim.ai/c/BZcLXGB6Ft5cjnLns).

- **Official Confirmation Raises Hype**: The anticipation peaks as an official statement is made—World-Sim alongside Nous World Client returns the next day. Users celebrate with excitement and share gifs like [Let Me In!](https://tenor.com/view/let-me-in-crazy-funny-silly-gif-13908292).

- **Hetetic Modelling Choices and Payment Options**: Inquiries about Claude 3 use and the possibility of switching models in World-Sim get addressed. A member mentioned that users would have model preferences based on affordability and confirmed various subscription and payment options, including an unlimited Claude Opus.

- **Developer Mode and World Client Queries Answered**: Discussions sprout around potential features, such as "developer mode," and clarifications on the Nous World Client, which will be web-based for accessibility from any device.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">world_sim</a>: no description found</li><li><a href="https://tenor.com/view/anime-excited-happy-smile-gif-15060821">Anime Excited GIF - Anime Excited Happy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/poe-path-of-exile-login-play-poe-login-gif-26508840">Poe Path Of Exile GIF - Poe Path Of Exile Login - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/let-me-in-crazy-funny-silly-gif-13908292">Let Me In Crazy GIF - Let Me In Crazy Funny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/tree-fiddy-south-park-lock-gif-5759991">Tree Fiddy GIF - Tree Fiddy South - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/noita-explosion-electricity-boom-wand-gif-19437628">Noita Explosion GIF - Noita Explosion Electricity - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/noita-game-homing-death-gif-27319696">Noita Game GIF - Noita Game Homing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/youre-not-gonna-like-this-jerrod-carmichael-saturday-night-live-you-wont-enjoy-this-this-wont-be-ideal-gif-25522925">Youre Not Gonna Like This Jerrod Carmichael GIF - Youre Not Gonna Like This Jerrod Carmichael Saturday Night Live - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/8dbWUUf9KKw?si=nHoa6wepiPBHY7j4">Every Vault in the Fallout Series Explained | Fallout Lore</a>: Hello everyone! This video is dedicated to any new Fallout Fans who wish to get into the Fallout games and their lore. I remember when I first became a fan a...</li><li><a href="https://websim.ai/c/BZcLXGB6Ft5cjnLns">Jailbroken Prometheus Chat</a>: no description found</li><li><a href="https://youtu.be/8dbWUUf9KKw?si=nHoa6wepiPBH">Every Vault in the Fallout Series Explained | Fallout Lore</a>: Hello everyone! This video is dedicated to any new Fallout Fans who wish to get into the Fallout games and their lore. I remember when I first became a fan a...</li><li><a href="https://youtube.com/shorts/A-zEXUB5CLY">The Godhood Paradox |  Science Fiction Animatic</a>: In a future where the World Sim—an online interface powered by advanced AI—allows users to create and manipulate virtual universes, a clash emerges. The Dece...
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1229689121145094186)** (286 messages🔥🔥): 

- **Model Comparisons and Misadventures**: Discussions revolve around the performance of various AI models including **GPT-4**, **Claude**, and **Mistral**. Users share experiences suggesting that newer versions at times seem lazier or less capable of managing extensive context, while others note the usefulness of models like **Claude 3 Opus** for technical issues. There's also mentions of [Mixtral's 8x22B model](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) being impressive for an open-source release.

- **Channel Guidance and Navigation**: New members are guided on how to find related chats and access various channels using the `<id:customize>` feature or by navigating through the Perplexity name at the top of the interface. 

- **Payment Anxieties and Checkout Changes**: Users express confusion and concern over changes to the Perplexity API payment method management and the lack of transparency regarding the remaining pro message counts.

- **File Handling Frustrations**: Users discuss the limitations of AI models in handling large context sizes, with one reported difficulty getting a 42k token prompt to properly engage with the system. Another user suggests that the model might be summarizing long documents instead of processing them in detail, impacting how the AI addresses specific prompts.

- **AGI Aspirations and Subscriptions**: Conversations feature anticipated updates, with some users eagerly waiting for new features like **Grok** to be added to Perplexity while others debate over the value of their subscriptions. 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/bindureddy/status/1780236367559086466">Tweet from Bindu Reddy (@bindureddy)</a>: The new GPT-4 is amazingly lazy and literally stops after a few turns (back and forth)  It’s not very viable in the real world for the moment. Stick to the older version.  Comparatively Claude has a l...</li><li><a href="https://www.markdownguide.org/extended-syntax/#tables">Extended Syntax | Markdown Guide</a>: Advanced features that build on the basic Markdown syntax.</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-v0.1">mistralai/Mixtral-8x22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/hamak-chilling-beach-summer-vacation-gif-17726234">Hamak Chilling GIF - Hamak Chilling Beach - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/silent-indian-gif-23263843">Silent Indian GIF - Silent Indian - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/how-soon-is-now-smiths-morrissey-80s-music-new-wave-gif-17919265">How Soon Is Now Smiths GIF - How Soon Is Now Smiths Morrissey - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://vm.tiktok.com/ZGeH84n4s/">TikTok - Make Your Day</a>: no description found
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1229726228416561273)** (9 messages🔥): 

- **Exploring World Voice Day**: A link to [Perplexity's results for World Voice Day](https://www.perplexity.ai/search/World-Voice-Day-kZvtPHUZRF6kcHO9vXf5Ew) was shared, revealing resources and discussions related to this event.
- **Delving into AWS Hardening Guide**: A user referenced a search for [AWS hardening guide](https://www.perplexity.ai/search/Aws-hardening-guide-E9rxYiA9SRSLnRvPjYhAzQ), pointing to Perplexity AI's aggregated information on enhancing security on AWS.
- **Discovering "SBK Borderline"**: The song "SBK Borderline" was the focus of a link, facilitating exploration through [Perplexity's summarized content](https://www.perplexity.ai/search/SBK-Borderline-song-c_MKbBj_RZGKWKLerhEypw).
- **Curiosity about Income**: A search about income queries was signaled through a [Perplexity AI link](https://www.perplexity.ai/search/How-much-do-CyRYvhYcSvuqmVOrRkoXfw), encapsulating associated answers and data points.
- **Investigating Reboot for Better Performance**: Discussion included a practical approach for enhancing an iPad's performance, as a user considered rebooting as illustrated in the given [Perplexity link](https://www.perplexity.ai/search/How-can-I-1l3jMti.SH.skEnk1CQZqA).
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1229707531006378006)** (4 messages): 

- **Seeking API and Web Client Consistency**: A member expressed difficulty in aligning the behavior of the web client with the API, noting occasional discrepancies and seeking to understand specific settings such as temperature to ensure consistency.
- **Navigating with Site Search Operator**: In reference to locating information, a member suggested using the site search operator `site:URL` to facilitate searches on a specific website.
- **Rate Limit Counter as a Feature Request**: A user proposed having the Perplexity API include the number of requests used within a minute in the response data, to better handle rate limits and potentially wait until the limit resets.
- **Querying API Rate Limiting Mechanism**: Another member questioned whether the Perplexity API returns a `429 response` when the rate limit is reached, indicating a need for clarity on how the API communicates with users about rate limits.
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1229831161916821606)** (285 messages🔥🔥): 

- **PyTorch Design Mysteries**: Members express confusion about the design philosophy of **PyTorch**, noting it often abstracts away many details with "just one line of code," which can prove challenging when something doesn't work as expected.

- **Storing Large Datasets with Zarr**: A discussion about using **zarr** or other libraries to store large datasets for fast loading, specifically for a 150 GB MRI image dataset. One member raises concerns about whether zarr would attempt to load the entire dataset into RAM.

- **British Law Criminalizing Creation of Certain Images**: There is a wrinkle in UK law criminalizing the creation of images with the intent to cause distress, and members debate the enforceability of such a law, especially since proving intent can be challenging.

- **Mysteries of Running AI Inference**: A member voices the need for access to actual inference settings to judge AI models properly, like adjusting CFG or hooking models up to suitable ODE solvers instead of just using Euler's method.

- **The Fate of SAI's Cascade Team and Channels**: It's mentioned that the **Cascade team has left Stability AI (SAI)**, with the related Discord channel being removed, and there's speculation about the possible involvement of team members with another company, Leonardo, or remaining affiliated with SAI.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.open-sora.org||">no title found</a>: no description found</li><li><a href="https://www.bbc.com/news/uk-68823042">Creating sexually explicit deepfakes to become a criminal offence</a>: A new law will see creators of sexually explicit deepfakes face prosecution and a fine.</li><li><a href="https://huggingface.co/ptx0/terminus-xl-velocity-v2">ptx0/terminus-xl-velocity-v2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/multimodalart/perturbed-attention-guidance-sdxl">Perturbed-Attention Guidance SDXL - a Hugging Face Space by multimodalart</a>: no description found</li><li><a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>: no description found</li><li><a href="https://tenor.com/b1ALd.gif">Minority Report Leave GIF - Minority Report Leave Walk Away - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c6g6zz/comment/l010k13/>">Reddit - Dive into anything</a>: no description found</li><li><a href="https://gist.github.com/drhead/ac6ecc1f6dc1fd478064f3d81ca12a25">Loss weighting MLP prototype</a>: Loss weighting MLP prototype. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://www.instagram.com/kushu.lofi">Login • Instagram</a>: no description found</li><li><a href="https://www.instagram.com/philipp.igumnov">Login • Instagram</a>: no description found</li><li><a href="https://www.instagram.com/ph">Login • Instagram</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1229816998448468110)** (13 messages🔥): 

- **Introducing ALERT Safety Benchmark**: A new safety benchmark for assessing **Large Language Models** has been established, complete with a safety Dataset of Problematic Outputs (DPO) set. All interested can access and use it via [GitHub - Babelscape/ALERT](https://github.com/Babelscape/ALERT).

- **Exploring Generative Multimodal Content**: An Arxiv paper discussing the generation of audio from text prompts and how focusing on the presence of concepts or events could improve performance, has been shared. View the research on [arXiv](https://arxiv.org/abs/2404.09956).

- **Debate over AI Safety Standards**: Members discussed the terminology and standards of "safety" in AI, debating whether restricting AI to non-controversial or PG content might limit its creative capacities compared to other artistic tools.

- **Comparing GANs with Diffusion Models**: A discussion unfolded around the benefits of **GANs** over diffusion models. Mentioned advantages include faster inference times, smaller parameter counts, feedback from discriminators, and potentially lower costs for training.

- **Skepticism Over GANs' Image Quality and Training Difficulty**: Despite some perceived benefits, GANs were criticized for reportedly producing inferior images as judged by human discrimination and presenting challenges in training compared to diffusion models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.09956">Tango 2: Aligning Diffusion-based Text-to-Audio Generations through Direct Preference Optimization</a>: Generative multimodal content is increasingly prevalent in much of the content creation arena, as it has the potential to allow artists and media personnel to create pre-production mockups by quickly ...</li><li><a href="https://github.com/Babelscape/ALERT">GitHub - Babelscape/ALERT: Official repository for the paper &quot;ALERT: A Comprehensive Benchmark for Assessing Large Language Models’ Safety through Red Teaming&quot;</a>: Official repository for the paper &quot;ALERT: A Comprehensive Benchmark for Assessing Large Language Models’ Safety through Red Teaming&quot; - Babelscape/ALERT
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1229798653028208701)** (5 messages): 

- **New Models and Price Adjustments**: OpenRouter announces the availability of [WizardLM-2 7B](https://openrouter.ai/models/microsoft/wizardlm-2-7b) and a price reduction for [WizardLM-2 8x22B](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b) to $0.65/M tokens. Discussions about these models can be followed in their dedicated channel.

- **Latency Issues Under Investigation**: OpenRouter is investigating high latencies for **Mistral 7B Instruct** and **Mixtral 8x7B Instruct** with ongoing discussions in a [message thread](https://discord.com/channels/1091220969173028894/1229813179681345556). The cause was initially tied to a cloud provider's DDoS protection but is now resolved.

- **Third-party Problems Affecting Services**: An update revealed reoccurring high latency issues affecting **Nous Capybara 34b** among others, potentially due to a specific cloud provider. Updates continued as the situation developed, with traffic returning to normal and further deep investigation with providers.

- **Maintenance Notice**: Users were informed of an impending DB reboot expected to briefly take the site offline.

- **Launch of High-Throughput Model and Status Update**: The [WizardLM-2 8x22B Nitro](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b:nitro) model is now serving over 100 transactions per second with a notice that the DB restart was completed. The team continues to address performance issues, with updates and discussions available in [channel](https://discord.com/channels/1091220969173028894/1229813179681345556).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b:nitro>)">WizardLM-2 8x22B by microsoft | OpenRouter</a>: WizardLM-2 8x22B is Microsoft AI&#x27;s most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing ...</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-7b>)">WizardLM-2 7B by microsoft | OpenRouter</a>: WizardLM-2 7B is the smaller variant of Microsoft AI&#x27;s latest Wizard model. It is the fastest and achieves comparable performance with existing 10x larger opensource leading models  It is a finet...</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b>)">WizardLM-2 8x22B by microsoft | OpenRouter</a>: WizardLM-2 8x22B is Microsoft AI&#x27;s most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing ...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1229799199818776647)** (4 messages): 

- **Help Wanted for AI Frontend Project**: A member is seeking a web developer to assist with a project focused on a general-purpose AI frontend for OpenRouter, which has a role-playing orientation. They've managed to get the novel mode working but are struggling with the conversation style mode.
- **Assistance Requested for Distinguishing AI Text**: They are also looking to enhance the novel mode by creating a way to differentiate between text generated by the AI and the user's own written text.
- **Development Support Sought for Sidebar and Modal System**: The member needs help to improve a sidebar with options and is looking to develop a flexible modal system for their application.
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1229690118315573300)** (271 messages🔥🔥): 

- **Censorship Layers and NSFW Content Management in AI Models**: Discussions touched on the layers of censorship within a particular AI model, and a member noted that their experiences with NSFW content on their end were very explicit. Another member questioned the usefulness of a base model for their purposes.

- **Interest in Multilingual Capacity of AI Models**: The multilingual performance of WizardLM was critiqued with a member suggesting it might be undertrained for non-English languages. There was speculation on whether upcoming models could surpass 8x7b models in performance and pricing.

- **Server Issues and Latency Concerns**: Members experienced issues with high latency and server errors, noting particularly long response times. Updates on investigating and resolving the server issues were provided, with a focus on fixing core server problems before adding new models such as Lepton's Wizard 8x22b.

- **Decoding Algorithm Impact on AI Model Quality**: Discussion about quantization of models to bits per word (bpw) revealed preferences for 6 or at least 5 bpw over 4 bpw, with some noting that a noticeable quality loss occurs with lower bpw.

- **Potential New Additions and Deployments of AI Models**: The OpenRouter team indicated that new models such as Mistral 8x22B Instruct were being deployed. Concerns about the reliability of certain providers like TogetherAI were expressed, with members looking forward to direct endpoints from Mistral and the addition of Fireworks as a provider.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mixtral-8x22b/">Cheaper, Better, Faster, Stronger</a>: Continuing to push the frontier of AI and making it accessible to all.</li><li><a href="https://giphy.com/gifs/robot-boston-creepy-ly2VUVUwtuHst1FhCq">Robot GIF - Find &amp; Share on GIPHY</a>: Discover &amp; share this Robot GIF with everyone you know. GIPHY is how you search, share, discover, and create GIFs.</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/chiquichico-gif-26004262">Chiquichico GIF - Chiquichico - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=29ECwExc-_M">All New Atlas | Boston Dynamics</a>: We are unveiling the next generation of humanoid robots—a fully electric Atlas robot designed for real-world applications. The new Atlas builds on decades of...</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b:nitro">WizardLM-2 8x22B by microsoft | OpenRouter</a>: WizardLM-2 8x22B is Microsoft AI&#x27;s most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing ...
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1229848403228229702)** (67 messages🔥🔥): 

- **Insights on Mojo's Compile-Time Optimizations**: Members discussed the optimization efficiency of Mojo, mentioning that aliases like `@parameter` are determined at compile time, leading to memory and processing efficiency by avoiding the need to reserve memory for the alias after its purpose is served. This conversation was sparked by thoughts on the importance of readable code over comments, as discussed in a [YouTube video](https://m.youtube.com/watch?v=Bf7vDBBOBUA) titled "Don't Write Comments".

- **Exploring Typestates in Rust Programming**: The conversation shifted towards best practices in API design, with one member favoring the use of typestates and lifetimes for making static guarantees in programming, sharing a [Rust typestate pattern article](https://cliffle.com/blog/rust-typestate/) for reference.

- **Contemplation on Memory Allocation and Optimization**: A debate unfolded about whether variables could be optimized in the same way as aliases in Mojo, touching upon optimization concerns in Rust and the potential for memory-efficient data structures such as [bit vectors](https://willcrichton.net/notes/k-corrset/).

- **Issues Adapting Code to Mojo Version 24.2**: Conversation occurred around upgrading the llama2.mojo code to be compatible with Mojo version 24.2, specifically the need for pointer type conversions. Solutions using `DTypePointer` were offered to address issues with `AnyPointer` conversion.

- **Mojo Development and IDE Integration Discussion**: Members discussed the structure of Mojo projects and whether there is a similar package management system to Rust's Cargo. Additionally, the availability of a Mojo plugin for IDEs such as PyCharm was mentioned, with reference to the [plugin link](https://plugins.jetbrains.com/plugin/23371-mojo), and the JetBrains team's interest in further Mojo support.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://plugins.jetbrains.com/plugin/23371-mojo">Mojo - IntelliJ IDEs Plugin | Marketplace</a>: Provides basic editing for Mojo programming language: syntax checks and highlighting, commenting and formatting. New features will be added in the future, please feel...</li><li><a href="https://devlog.hexops.com/2022/packed-structs-in-zig/">Packed structs in Zig make bit/flag sets trivial</a>: As we've been building Mach engine, we've been using a neat little pattern in Zig that enables writing flag sets more nicely in Zig than in other languages. Here's a brief explainer.</li><li><a href="https://tenor.com/view/bamboozled-gif-25267741">Bamboozled GIF - Bamboozled - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://willcrichton.net/notes/k-corrset/">Analyzing Data 180,000x Faster with Rust</a>: How to hash, index, profile, multi-thread, and SIMD your way to incredible speeds.</li><li><a href="https://cliffle.com/blog/rust-typestate/">
The Typestate Pattern in Rust - Cliffle
</a>: no description found</li><li><a href="https://m.youtube.com/watch?v=Bf7vDBBOBUA&t=0s">Don&#39;t Write Comments</a>: Why you shouldn&#39;t write comments in your code (write documentation)Access to code examples, discord, song names and more at https://www.patreon.com/codeaesth...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1780676643176231240>
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1229796285352247366)** (2 messages): 

- **Replication Curiosity in Modular**: A member expressed interest in replicating a concept or project within the Mojo platform, indicating anticipation for potential outcomes.
- **Guidance on AI Long-Term Memory and Self-Improvement**: A video tutorial was shared by a member explaining how to build an AI agent with long-term memory and self-improvement capabilities, intended to be a helpful resource. The video, titled "Unlock AI Agent real power?! Long term memory & Self improving," is available on [YouTube](https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek).

**Link mentioned**: <a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>: How to build Long term memory &amp; Self improving ability into your AI Agent?Use AI Slide deck builder Gamma for free: https://gamma.app/?utm_source=youtube&amp;utm...

  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1229702087915339779)** (136 messages🔥🔥): 

- **New Python Package for Mojo to Python Code**: A new python package called [mojo2py](https://github.com/venvis/mojo2py) has been announced that converts Mojo code into Python code.
- **Need for a Comprehensive Mojo Learning Resource**: A member is seeking a comprehensive resource for learning Mojo from scratch, and was directed to the [Mojo programming manual](https://docs.modular.com/mojo/manual/), which covers fundamental concepts such as parameters vs. arguments, the ASAP concept, types and traits, and key re-reading sections like owned arguments and transfer operator.
- **Struct Inheritance and Code Reusability**: Discussions circled around the desire for some form of inheritance within Mojo, with suggestions for reducing boilerplate and instances where a child struct could be created from a parent struct. While one approach suggested was using traits for type declarations, another member clarified that if one seeks compile-time optimization, classes might be more suitable, versus runtime-based approaches.
- **Start of Conditional Conformance in Mojo**: There appears to be movement towards implementing conditional conformance in Mojo, as evidenced by recent discussion and code snippets shared amongst members. The dialogue involved understanding how conditional conformance might be leveraged to make standard library functions like `str` and `print` work for different Mojo data structures.
- **Challenges and Prospects of Advanced Type Systems**: Intense technical debate and brainstorming emerged around creating a numpy-style Mojo library that enforces shape compatibility at compile time, the potential for supporting `Variant` data structures without runtime checks, and addressing the specific issue of storing multiple variants in a single list. Various approaches were proposed and conceptually dissected, including custom structs, enum parameters, and challenges in implementing generics and shape refinement for parametric code.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/">Mojo Manual | Modular Docs</a>: A comprehensive guide to the Mojo programming language.</li><li><a href="https://www.youtube.com/watch?v=p3zo4ptMBiQ">Protocol-Oriented Programming in Swift / WWDC15 / Session 408</a>: At the heart of Swift&#39;s design are two incredibly powerful ideas: protocol-oriented programming and first class value semantics. Each of these concepts benef...</li><li><a href="https://github.com/venvis/mojo2py">GitHub - venvis/mojo2py: A python package to convert mojo code into python code</a>: A python package to convert mojo code into python code - venvis/mojo2py
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1229988474728349778)** (10 messages🔥): 

- **Sudden Sketch Success**: A community member shared an "off the cuff" programming sketch implemented in Mojo, found to be surprisingly effective, accessible via [this gist](https://gist.github.com/lsh/6ca8864a9cffef9e503d6262eb876893).
- **Anticipating Enhanced Tuple Capabilities**: Upcoming enhancements could allow `Tuple` in Mojo to take traits derived from `CollectionElement`, leading to more **elegant struct definitions** for HTML rendering.
- **Nightly Features in Play**: It was clarified that the shared code uses **nightly features**, which may cause compilation errors on the current Mojo 24.2 and on the Mojo Playground.
- **Canny Edge Recognition Challenge**: A new community member from France, experienced in Numba with Python, expressed interest in implementing the **Canny edge recognition algorithm** in Mojo to compare performance.
- **Mojo Resources for Newcomers**: A helpful response to a project inquiry included links to the [Mojo documentation](https://docs.modular.com/mojo/manual/get-started/), guidance on getting started with the language, and referenced available resources such as the Mojo SDK and [Mojo Playground](https://docs.modular.com/mojo/notebooks/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/get-started/">Get started with Mojo🔥 | Modular Docs</a>: Get the Mojo SDK or try coding in the Mojo Playground.</li><li><a href="https://docs.modular.com/mojo/">Mojo🔥 | Modular Docs</a>: A programming language that bridges the gap between AI research and production, unlocking speed and usability.</li><li><a href="https://docs.modular.com/mojo/notebooks/">Mojo🔥 notebooks | Modular Docs</a>: All the Jupyter notebooks we&#x27;ve created for the Mojo Playground.</li><li><a href="https://gist.github.com/lsh/6ca8864a9cffef9e503d6262eb876893">html.mojo</a>: GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1230092571284340819)** (1 messages): 

- **Exploring the Hype Around Mojo**: A recent [talk](https://youtu.be/mhZFyzqdmi8) titled "Maxim Zaks - Is Mojo just a hype?" from PyCon Lithuania has been released on YouTube, which prompts a discussion on the Modular chatbot's place in the industry.

**Link mentioned**: <a href="https://youtu.be/mhZFyzqdmi8)">Maxim Zaks - Is Mojo just a hype?</a>: no description found

  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - Issue 30
https://www.modular.com/newsletters/modverse-weekly-30
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1230231689897447424)** (1 messages): 

There was only one message provided with no mention of any discussion points, topics, or links to summarize. If you would like a summary of a more extensive conversation or a specific topic within the 🏎engine channel, please provide the relevant messages.
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1229881110218145864)** (21 messages🔥): 

- **A New Nightly Mojo: Updates and Changes**: A new nightly update for Mojo has been released, complete with updates to the standard library and a [detailed diff](https://github.com/modularml/mojo/pull/2313/files) available, as well as a changelog documenting the changes since the last stable release found [here](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **A Love for Unconventional Code**: Members reacted humorously to unconventional code styling, with comments indicating affection for its "horrible" appearance and a comical plea to *indent for loops* for readability.
- **Peer Pressure vs. Code Formatting Practices**: One voice suggested holding off on conforming to peer pressure regarding code indentation practices, but another opined the inevitability of adopting Mojo formatting standards.
- **Nightly update causes confusion**: The new nightly update led to confusion for a user over function overloads parameterized on traits, resulting in unexpected errors and discussions around finding a solution.
- **Traits Over Janky Workarounds and Clean-Up Releases**: Discussion included a slight jest on the preference for using 'jank' over proper trait parameterization and comments on the recent clean-up efforts in the latest Mojo nightly release.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/pull/2313/files">[stdlib] Update stdlib corresponding to `2024-04-16` nightly/mojo by patrickdoc · Pull Request #2313 · modularml/mojo</a>: This updates the stdlib with the internal commits corresponding to today&#39;s nightly release: mojo 2024.4.1618 . In the future, we may push these updates directly to nightly branch.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/docs/changelog.md">mojo/docs/changelog.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1229812396894457886)** (11 messages🔥): 

- **Seeking Guidance in PyTorch**: A member asked if "Deep Learning with PyTorch" is still a good starting point given that it was published 4 years ago. Another member confirmed that while **PyTorch's core** hasn't changed much, there have been significant updates in the compiler and distributed systems.

- **PyTorch Evolution and New Edition Tease**: Updates were discussed clarifying that the book does not cover topics like **transformers and LLMs**, and that while parts I and II remain useful, part III on deployment is outdated. It was also revealed that a **[new edition is in progress](https://www.manning.com/books/deep-learning-with-pytorch-second-edition)**, spearheaded by a new author.

- **Anticipating Blog Content**: A member mentioned they had a draft chapter on attention/transformers and considered creating a **blog post** from it.

**Link mentioned**: <a href="https://www.manning.com/books/deep-learning-with-pytorch-second-edition">Deep Learning with PyTorch, Second Edition</a>: Everything you need to create neural networks with PyTorch, including Large Language and diffusion models.&lt;/b&gt;
 
 Deep Learning with PyTorch, Second Edition&lt;/i&gt; updates the bestselling ori...

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1229921763228389508)** (20 messages🔥): 

- **Accelerated Matrix Operations in CUDA**: A member discussed the integration of a new fp16 precision [general matrix-matrix multiplication (GEMM) implementation for CUDA](https://github.com/aredden/torch-cublas-hgemm/blob/master/src/simt_hgemv.cu), which outperforms PyTorch's GEMM function in a specific matrix operation benchmark (MxNxK = 1x4096x4096).
- **Challenges with JIT Compilation**: Despite the new implementation providing a performance boost, another member noted it fails with `torch.compile`; sharing crash details with uncompiled token generation at 11.17 tokens/sec versus compiled token generation at 64.4 tokens/sec before the crash due to an unsupported method call related to 'block_dim_x'.
- **Block Size Parameters Exploration**: Discussion continued around the choice of block sizes in the new GEMM kernel, with members examining the use of a 32x4 effective block size, discovering it seemed to yield better performance and sharing their observations in [a related Gist example](https://gist.github.com/mobicham/9aa8dc0e64ea1cb7d4e44fef55e6a4b4).
- **Inquiry about Data Reading for CUDA C++**: A member sought advice on reading large datasets in CSV or Parquet formats within CUDA C++ applications, pondering the possibility of parallel execution but without offering a specific solution.
- **Speculating on CUDA Cores and Thread Dispatch**: Further technical speculation highlighted the probable connection between faster kernel performance and the use of 128 total active threads per streaming multiprocessor, considering the dispatch of 32 threads per clock cycle across 4 warps.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/aredden/torch-cublas-hgemm/blob/master/src/simt_hgemv.cu">torch-cublas-hgemm/src/simt_hgemv.cu at master · aredden/torch-cublas-hgemm</a>: PyTorch half precision gemm lib w/ fused optional bias + optional relu/gelu - aredden/torch-cublas-hgemm</li><li><a href="https://gist.github.com/mobicham/9aa8dc0e64ea1cb7d4e44fef55e6a4b4">zippy_gemv_hqq_gen.py</a>: GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1229781137547853864)** (2 messages): 

- **Searching for the F.Linear Implementation**: A member is working on a custom backward function that performs correctly with a **(bs, data_dim)** input similar to **F.Linear**. They encountered issues when integrating with **Llama** due to input dimension differences and are now seeking the forward/backward implementation of `F.Linear`, which was elusive in the indicated *tools/autograd/templates/python_nn_functions.cpp*.
  

---


**CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1229840827010781205)** (2 messages): 

- **Augmend Launches Video Processing Tool**: Augmend offers a work-in-progress feature on [wip.augmend.us](http://wip.augmend.us) for analyzing videos, with a smart addition of OCR and image segmentation to extract information directly from video screens. The completed service will be available on [augmend.com](http://augmend.com), allowing users to copy/paste and search content within any video.

- **Boston Dynamics Reveals Electric Atlas Robot**: Boston Dynamics released a YouTube video on a next-generation humanoid robot named *Atlas*; the [All New Atlas | Boston Dynamics video](https://www.youtube.com/watch?v=29ECwExc-_M) presents a fully electric robot aimed at real-world applications and highlights advances over decades of robotic development.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=29ECwExc-_M">All New Atlas | Boston Dynamics</a>: We are unveiling the next generation of humanoid robots—a fully electric Atlas robot designed for real-world applications. The new Atlas builds on decades of...

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1229724970331013160)** (43 messages🔥): 

- **Newcomer Inquiry on PMPP Lectures**: A newcomer inquired about the routine meeting schedule for going through **pmpp lectures**. Recorded lectures can be found in a specific channel, with the last covered chapter being the 10th.

- **WSL Profiling Troubles**: A user expressed difficulty running the **ncu profiler** on WSL, suspecting a **PATH issue**, and highlighted that NSight Compute on Windows was conflicting with WSL. Despite having nsight-compute installed, the `ncu` command was not found.

- **Cuda Toolkit PATH Adjustment Suggestions**: Users suggested several troubleshooting steps, focusing on adding the correct **CUDA path to the environment variables**. One user provided a **[link to NVIDIA's documentation](https://docs.nvidia.com/gameworks/content/developertools/desktop/environment_variables.htm)** to assist with setting environment variables on Windows.

- **Version Mismatch Discovered**: It was discovered that there was a version mismatch, with the user's environment configured for CUDA 12.4 while attempting to run `ncu` from CUDA **version 11.5**. Adding the path didn’t immediately resolve the issue.

- **Windows 11 Recommended for WSL 2 Profiling**: Another user mentioned needing **Windows 11** to profile CUDA programs on WSL 2 effectively, sharing a [helpful blog post](https://peterchng.com/blog/2024/03/02/profiling-cuda-programs-on-wsl-2/) detailing how to set up the system and resolve common issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/gameworks/content/developertools/desktop/environment_variables.htm">Environment Variables</a>: no description found</li><li><a href="https://peterchng.com/blog/2024/03/02/profiling-cuda-programs-on-wsl-2/">Profiling CUDA programs on WSL 2</a>: no description found
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

marksaroufim: https://www.youtube.com/watch?v=DdTsX6DQk24
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1230169198844444825)** (5 messages): 

- **RingAttention Working Group Conundrum**: A key member revealed that they cannot commit to working on the **RingAttention** project alongside their main job due to time constraints. They proposed a discussion to decide whether others will continue the initiative or temporarily conclude this working-group effort.
- **Decisive Discussion Scheduled**: A meeting was scheduled to discuss the future of the **RingAttention** project and who might continue its development.
- **A Time for Difficult Choices**: The member expressed regret over their decision to step back from **RingAttention**, emphasizing that the choice was made with heavy consideration of personal time and well-being.
- **Participants Ready for the Talk**: Team members confirmed their availability and showed readiness to join the forthcoming discussion about the future of **RingAttention**.
- **Pre-Meeting Preparations**: One of the members notified others that they would join the meeting shortly, indicating active preparation for the scheduled discussion.
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1229814130295181473)** (36 messages🔥): 

- **Quandaries About Quantization Axes**: Quantizing with `axis=0` for GPT's Q, K, V was found problematic in `gpt-fast` due to mixing of parameters during quantization. An ongoing discussion suggests quantizing Q, K, and V separately might be a solution, noting that `weight_int4pack_mm` currently only supports `axis=1`.

- **Speed Versus Quality Compromises in HQQ**: The trade-offs between speed and quality when using `axis=0` or `axis=1` in Half-Quadratic Quantization (HQQ) were explored. A member reported equivalent performance of 5.375 perplexity for both axes on `gpt-fast`.

- **Pursuing Further Optimizations**: A mention of using [Triton kernels](https://github.com/wangsiping97/FastGEMV/tree/main) and other methods like fake data to optimize performance along `axis=1`. They noted that method using autograd and randomly generated data gave slightly better results (5.3311 ppl) than HQQ with more iterations.

- **Exploring Extended Capabilities and Demystifying Differences**: Insights into the potential impact of in-channel variation on weight quantization accuracy were shared, referring to steeling quants with `axis=0` appearing to yield better results. The conversation indicated that HQQ effectively finds optimal solutions faster compared to lengthy autograd optimization.

- **Implementational Details and Benchmarks Shared**: Links were provided to the implementation details, such as a [torch int4mm demo](https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py) with transformers as well as the [optimizer code using autograd](https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L412) and discussions were centered around potentially speeding up operations further with vectorized fp16 multiplication and the practicality of lower precision quantization like 2/3 bits.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/zhxch">zhxch (zhongxiaochao)</a>: no description found</li><li><a href="https://github.com/mobiusml/hqq/blob/63cc6c0bbb33da9a42c330ae59b509c75ac2ce15/hqq/core/quantize.py#L81-L85),">hqq/hqq/core/quantize.py at 63cc6c0bbb33da9a42c330ae59b509c75ac2ce15 · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://huggingface.co/zhxchen17/scratch/tree/main">zhxchen17/scratch at main</a>: no description found</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/kernels/hqq_aten_cuda_kernel.cu#L109-L115">hqq/hqq/kernels/hqq_aten_cuda_kernel.cu at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/wangsiping97/FastGEMV/tree/main">GitHub - wangsiping97/FastGEMV: High-speed GEMV kernels, at most 2.7x speedup compared to pytorch baseline.</a>: High-speed GEMV kernels, at most 2.7x speedup compared to pytorch baseline.  - GitHub - wangsiping97/FastGEMV: High-speed GEMV kernels, at most 2.7x speedup compared to pytorch baseline.</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/backends/torchao_int4_demo.py">hqq/examples/backends/torchao_int4_demo.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py#L412">hqq/hqq/core/optimize.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1229714817586036756)** (76 messages🔥🔥): 

- **Thunder's CUDA Python Extension Takes Flight**: The [GitHub notebook](https://github.com/Lightning-AI/lightning-thunder/blob/main/notebooks/extend_thunder_with_cuda_python.ipynb) for extending PyTorch with CUDA Python receives attention for improving speed, though the integration into cuda-mode and further optimizations such as leveraging tensor cores are still needed for maximum performance.

- **Optimizing Multiplication in Transformers**: Members identified the final matmul layer and softmax as significant contributors to computational cost in profiling efforts. An optimised classifier kernel presents an opportunity for improving speed, as seen in the conversation about caching strategy and kernel optimization.

- **Increasing Efficiency of Softmax and Backpropagation**: There was discussion about avoiding the materialization of the full probability matrix, focusing instead on necessary token probabilities. A [GitHub pull request #117](https://github.com/karpathy/llm.c/pull/117) demonstrates efforts to fuse points in the classification layer.

- **Cache Utilization and Performance Correlation**: The effect of block sizes on cache hit rates was discussed, revealing that larger blocks may result in better cache utilization. This insight, embodied in an [optimised CUDA kernel](https://github.com/karpathy/llm.c/pull/150), might lead to better performance on GPUs with sufficient cache.

- **Supporting Diverse Model Architectures for Benchmarking**: It was suggested to consider the initialization of a variety of GPT model architectures for benchmarking to prevent overfitting optimizations to a single model type. An emphasis was placed on accurately reproducing models like GPT-2 to evaluate performance enhancements meaningfully.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Lightning-AI/lightning-thunder/blob/main/notebooks/extend_thunder_with_cuda_python.ipynb">lightning-thunder/notebooks/extend_thunder_with_cuda_python.ipynb at main · Lightning-AI/lightning-thunder</a>: Make PyTorch models up to 40% faster! Thunder is a source to source compiler for PyTorch. It enables using different hardware executors at once; across one or thousands of GPUs. - Lightning-AI/ligh...</li><li><a href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md">cutlass/media/docs/quickstart.md at main · NVIDIA/cutlass</a>: CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/150">Optimised version of fused classifier + bugfixes(?) by ademeure · Pull Request #150 · karpathy/llm.c</a>: This is a faster version of the cool new kernel from #117 (still /dev/cuda/ only). The biggest difference is it is optimised for doing one row per 1024-wide block rather than per 32-wide warp, whic...</li><li><a href="https://github.com/karpathy/llm.c/pull/117">WIP: Fully fused classification layer by ngc92 · Pull Request #117 · karpathy/llm.c</a>: This fuses together all the pointwise operations that happen in the token classification layer. This essentially gives us the forward/backward for the cost of about just the forward pass, because t...</li><li><a href="https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)">SlimPajama: A 627B token, cleaned and deduplicated version of RedPajama - Cerebras</a>: Cerebras has built a platform for push-button training of large language models that can accelerate time to insights without having to orchestrate across a large cluster of small devices.</li><li><a href="https://arxiv.org/abs/2304.08442">The MiniPile Challenge for Data-Efficient Language Models</a>: The ever-growing diversity of pre-training text corpora has equipped language models with generalization capabilities across various downstream tasks. However, such diverse datasets are often too larg...</li><li><a href="https://github.com/tysam-code/hlb-gpt/tree/main">GitHub - tysam-code/hlb-gpt: Minimalistic, extremely fast, and hackable researcher&#39;s toolbench for GPT models in 307 lines of code. Reaches &lt;3.8 validation loss on wikitext-103 on a single A100 in &lt;100 seconds. Scales to larger models with one parameter change (feature currently in alpha).</a>: Minimalistic, extremely fast, and hackable researcher&amp;#39;s toolbench for GPT models in 307 lines of code. Reaches &amp;lt;3.8 validation loss on wikitext-103 on a single A100 in &amp;lt;100 secon...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[massively-parallel-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1229715140434333716)** (14 messages🔥): 

- **Tablet Triumph for Presentations**: A member pondered the possibility of using an iPad to switch between slides and live writing for presentations. The consensus suggested using **a single device for both tasks** and emphasized the importance of testing the setup beforehand to ensure a smooth experience.

- **No to NSFW**: With incidents of **inappropriate content** being posted in the chat, members discussed implementing a **Discord bot** to detect and prevent such content from being shared, with suggestions of banning offenders or restricting their typing privileges.

- **Event Creation Empowerment**: It's been announced that everyone now has the **roles and privileges** to create new events on the server. This change empowers members to organize their own gatherings and discussions.

- **Interjections and Interactions**: Casual interactions among members included humorous suggestions for names like “Massively Helpful” and playing with the word "parallel" in the context of the server name. These moments reflect the lighter side of the community's interactions.

- **Tech Tips Shared**: There's helpful advice given for someone wishing to stream presentations, including **using a Wacom tablet** and maintaining audience engagement through methods like different setups. The importance of testing the setup early was highlighted once again.
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1229725006347370537)** (167 messages🔥🔥): 

- **Gaming Assistant Development Inquiry**: A user sought advice on creating a **gaming assistant** combining GPT-Vision, camera input, and probabilistic calculations for real-time multiple-choice games. Considering using **Azure or a virtual machine** for running demanding calculation software was suggested, with TensorFlow or OpenCV as possible tools to manage the system.

- **AI vs. Human Cognition Debate**: The channel hosted a philosophical discussion on the fundamental differences between AI and humans, touching on concepts such as **memory storage**, computational power, and the potential for AI to develop **human-like reasoning and emotions** with advancements like quantum computing.

- **Understanding Non-Binary Thinking**: There was an extensive debate on binary versus non-binary thinking, with users discussing the applicability of **binary thinking and labels** in humans and AI, and how gradients and chaos theory might present a more accurate model of cognition and decision-making.

- **Claude's Superiority for Literature Reviews**: Users exchanged opinions on suitable AI models for writing literature reviews, with advice given to use **Claude** over OpenAI for non-technical literary tasks, and mentioning **Gemini 1.5** for aiding in writing fictional works.

- **Navigating AI-Related Complications**: Participants reported and discussed issues such as unexpected **account terminations** and policy violations, highlighting challenges in understanding and adhering to the usage policies of AI platforms, and expressing concerns about the lack of clarity and support often encountered.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/disneyland-paris-parks-sleeping-beauty-gif-5070382">Disneyland Paris GIF - Disneyland Paris Parks - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://en.wikipedia.org/wiki/Turing_completeness">Turing completeness - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1229938287314206730)** (7 messages): 

- **GPT Gets a Trim?**: A user remarked that it seems like GPT was significantly altered or "lobotomised," while another defended the new **GPT-4 Turbo** as being effective, mentioning alternate endpoints to use.
- **Important to Report Flaws**: One member encouraged others to report any problematic messages from GPT to improve its performance.
- **Discussing Alternatives Due to Costs**: A user shared that they are using **Gemini 1.5** with a 1 million token context window on Google Studio as an alternative, implying costs are a factor.
- **Seeking Knowledge Base Training**: Someone asked for directions to trainings or resources on how to prepare a knowledge base for a custom GPT.
- **Whispering for Whisper v3 API Access**: A query was raised about when **Whisper v3** would become available through the API, noting that it has been almost a year since its release.
- **Shrinking Token Attention Span?**: A user observed that GPT-4's ability to remember past inputs seems impaired, speculating about a reduced token limit from beyond 30,000.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1229929784675405966)** (5 messages): 

- **Echoes in the Ghost Town**: One member laments the decline of activity in the prompt-engineering channel, attributing the lack of discussion to over-moderation by administrators and mods.
- **Salty Retrospection**: A user suggests their extended timeout from the server may be related to a decline in activity, and believes others may have faced similar penalties.
- **GPT-4-Turbo's Math Prowess**: GPT-4-TURBO successfully solved a math problem regarding the number of possible seating arrangements for the Smith family at their dinner table.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1229929784675405966)** (5 messages): 

- **Silence in the OpenAI Discord**: One member expressed dismay at the lack of recent activity within the **api-discussions** channel, noting it has been quiet for weeks.
- **Reflections on Server Moderation**: The same member attributed the inactivity to what they perceived as over-moderation by the server's administrators.
- **Post-Timeout Frustrations**: Following a **5-month timeout** from the server, the member lamented that they were punished for attempting to assist another user.
- **GPT-4-Turbo's Mathematical Prowess**: A user reported that **GPT-4-TURBO** correctly solved a combinatorial math problem involving the seating arrangements of the Smith family at a dinner table.
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1229835485623619755)** (3 messages): 

- **Qdrant Hybrid Cloud Offering Launch**: The [@qdrant_engine](https://twitter.com/llama_index/status/1780275878230139293) has launched a hybrid cloud offering, enabling running Qdrant as a hosted service, at the edge, or in one's own environment while maintaining full data control. The announcement also linked to an [in-depth tutorial](https://t.co/4nS9j9ruwR) on setting it up.

- **LlamaIndex Teams Up with Azure AI Search**: A [tutorial](https://t.co/lITCdlCejT) presented by Khye Wei from Microsoft demonstrates how to combine LlamaIndex with Azure AI Search to create enhanced RAG applications that feature Hybrid Search and Query rewriting.

- **Day 0 Support for MistralAI's Latest Model**: [MistralAI's new 8x22b model](https://t.co/WWbYp5lqXe), described as defining the state of the art in open models, is supported by LlamaIndex from day one. The release includes a Mistral cookbook by @ravithejads, showcasing RAG, Query routing, and Tool use.

**Link mentioned**: <a href="https://t.co/WWbYp5lqXe">MistralAI Cookbook - LlamaIndex</a>: no description found

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1229694863667367986)** (164 messages🔥🔥): 

- **Inquiry About Building a Search Engine**: Users discussed how to build a search engine using LlamaIndex. One user provided a [starter tutorial](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/) and highlighted the use of `retriever` with a higher `top_k` value to retrieve top documents.

- **Understanding LLM Retrieval Limits**: A user clarified they needed to retrieve document names instead of answers from agents, comparing it to perplexity functionality. The conversation continued with users referencing LlamaIndex's `retriever` and its settings.

- **Issues With Authentication**: Several users encountered and discussed errors related to API authentication. The error messages indicated incorrect API keys, leading to troubleshooting around environment variables and correct key usage.

- **LLamaIndex Updates And Issue Fixing**: Users collaboratively tried to resolve various issues, with a specific focus on a `BaseComponent` error which one user couldn't resolve despite trying numerous troubleshooting steps. A solution was suggested in the form of a [GitHub pull request](https://github.com/run-llama/llama_index/pull/12882).

- **LLM Query Logging and Active Model Check**: Discussion on logging within LlamaIndex led to advising on adjusting logging levels from `DEBUG` to `INFO`. A user sought to confirm which LLM was active for a query and was advised on checking and setting the LLM through the `Settings.llm` attribute.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/17IJlvx2M2iGu3weIttvwml2axAAt0Vk9?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial (OpenAI) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/callbacks/LangfuseCallbackHandler/?h=lang">Langfuse Callback Handler - LlamaIndex</a>: no description found</li><li><a href="http://localhost:port",>">no title found</a>: no description found</li><li><a href="http://localhost:port"`>">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/llms/openai_like#llama_index.llms.openai_like.OpenAILike>).">Openai like - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding/?h=embeddings+fine">Finetune Embeddings - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/create_llama_projects/tree/main/nextjs-edge-llamaparse">create_llama_projects/nextjs-edge-llamaparse at main · run-llama/create_llama_projects</a>: Contribute to run-llama/create_llama_projects development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents/">Multi-Document Agents - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/answer_and_context_relevancy/">Answer Relevancy and Context Relevancy Evaluations - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/pull/12882">Catch validation errors by logan-markewich · Pull Request #12882 · run-llama/llama_index</a>: Some people are experiencing some weird errors here. Lets just catch validation errors to prevent incompatible package versions from crashing core</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/llama_2_llama_cpp#setup-llm>)">LlamaCPP - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/tools/openapi#llama_index.tools.openapi.OpenAPIToolSpec>)">Openapi - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/q_and_a/?h=summar#summarization">Q&A patterns - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/doc_summary/DocSummary/?h=summary">Document Summary Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/response_synthesizers/custom_prompt_synthesizer/?h=summa">Pydantic Tree Summarize - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/schema#llama_index.core.schema.BaseComponent>)">Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/query_pipeline#llama_index.core.base.query_pipeline.query.QueryComponent>)">Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/query_pipeline/llm#llama_index.core.llms.llm.BaseLLMComponent>)">Llm - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/query_pipeline/llm#llama_index.core.llms.llm.LLMChatComponent>)">Llm - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1229807041850380348)** (2 messages): 

- **Seeking Hierarchical Structure Wisdom**: A member is looking to construct a **parent-child hierarchical structure** within the *LlamaIndex* using *ParentDocumentRetriever langchain* for a vast number of documents and is requesting guidance.
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1229696835212546058)** (58 messages🔥🔥): 

- **Pile-T5 Details Sought**: A user requested details about the Pile-T5 model on EleutherAI's Discord, pointing to the [Hugging Face collection page](https://huggingface.co/collections/EleutherAI/pile-t5-65a76a0d0022dd270b385a66) for further information. The discussion clarified that "sequence length" and "context window" are the same, while noting the scarcity of encoder/decoder models with long sequence lengths.
  
- **Reka's Long Enc-Dec Model Revealed**: In discussing model sequence lengths, a user mentioned Reka's new encoder-decoder model, which supports up to 128k, as described in their [core tech report](https://publications.reka.ai/reka-core-tech-report.pdf).

- **EleutherAI's Model Evaluation Harness Discussed**: The ARC-challenge on EleutherAI's Evaluation Harness was debated with concerns on the absence of "choices" in the query for models. It was mentioned that the library initially aimed to replicate plots from the GPT-3 paper, with intentions to standardize MCQA tasks by offering multiple prompting options.

- **Research Scientist Interview Insights**: Users shared insights on research scientist interviews, explaining that the focus can vary greatly depending on the company, ranging from little emphasis on traditional data structure and algorithm questions to heavy consideration of the candidate's talk, papers, and potential for grant acquisition.

- **Sequence Packing vs. Prepacking in LLMs**: A discussion emerged about whether "prepacking" is just regular sequence packing, as mentioned in a new research paper. This led to a debate about the novelty and prior documentation of these methods, with references to the T5 paper and upcoming publications addressing these and related methods for model evaluation and efficiency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/siyan_zhao/status/1780288750624612850?s=46">Tweet from Siyan Zhao (@siyan_zhao)</a>: 🚨LLM RESEARCHERS🚨Want a free boost in speed and memory efficiency for your HuggingFace🤗LLM with ZERO degradation in generation quality? Introducing Prepacking, a simple method to obtain up to 6x sp...</li><li><a href="https://huggingface.co/collections/EleutherAI/pile-t5-65a76a0d0022dd270b385a66">Pile-T5 - a EleutherAI Collection</a>: no description found</li><li><a href="https://x.com/srush_nlp/status/1779938508578165198">Tweet from Sasha Rush (@srush_nlp)</a>: Lazy twitter: A common question in NLP class is &#34;if xBERT worked well, why didn&#39;t people make it bigger?&#34; but I realize I just don&#39;t know the answer. I assume people tried but that a l...</li><li><a href="https://huggingface.co/lintang/pile-t5-base-flan">lintang/pile-t5-base-flan · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lintang">lintang (Lintang Sutawika)</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/tasks/arc.py#L61">lm-evaluation-harness/lm_eval/tasks/arc.py at b281b0921b636bc36ad05c0b0b0763bd6dd43463 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/tasks/hendrycks_test.py#L153">lm-evaluation-harness/lm_eval/tasks/hendrycks_test.py at b281b0921b636bc36ad05c0b0b0763bd6dd43463 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://huggingface.co/models?other=base_model:EleutherAI/pile-t5-base">Models - Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1229728886515040348)** (78 messages🔥🔥): 

- **New Transformer Architecture for Long Inputs**: A recent proposal for a novel Transformer architecture named [Feedback Attention Memory (FAM)](http://arxiv.org/abs/2404.09173) aims to enable processing of indefinitely long sequences by allowing the network to attend to its own latent representations, thus overcoming the quadratic attention complexity. FAM's performance showed significant improvement on long-context tasks.

- **Advances in Brain Decoding Research**: The paper [MindBridge](https://arxiv.org/abs/2404.07850v1) introduces a new approach that allows for cross-subject brain decoding by employing only one model, addressing three main challenges in the field: variability in brain sizes, individual neural pattern differences, and limited data for new subjects.

- **Rethinking Scaling Laws' Accuracy**: Discrepancies pointed out in the compute-optimal scaling laws presented by Hoffmann et al. (2022) highlight the importance of data transparency, as [a new analysis](https://arxiv.org/abs/2404.10102) suggests that the original narrow confidence intervals were implausible unless an extensive number of experiments were conducted.

- **Expressive Power of State-Space Models**: A discussion was prompted by the [analysis of State-Space Models (SSMs)](https://arxiv.org/abs/2404.08819), revealing that their expressive power for state tracking is very similar to transformers and SSMs cannot express computation beyond the complexity class $\mathsf{TC}^0$. The dialogue also touched upon clarifications and potential misunderstandings from prior related works.

- **Transformers, RL, and EEG Feedback**: Conversations touched on the concept of using Reinforcement Learning (RL) with feedback from an EEG but found limited academic research, primarily existing product implementations; the complexities and risks associated with such undertakings were also noted.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.08819">The Illusion of State in State-Space Models</a>: State-space models (SSMs) have emerged as a potential alternative architecture for building large language models (LLMs) compared to the previously ubiquitous transformer architecture. One theoretical...</li><li><a href="https://arxiv.org/abs/2404.10102">Chinchilla Scaling: A replication attempt</a>: Hoffmann et al. (2022) propose three methods for estimating a compute-optimal scaling law. We attempt to replicate their third estimation procedure, which involves fitting a parametric loss function t...</li><li><a href="https://arxiv.org/abs/2404.10642">Self-playing Adversarial Language Game Enhances LLM Reasoning</a>: We explore the self-play training procedure of large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate wit...</li><li><a href="http://arxiv.org/abs/2404.09173">TransformerFAM: Feedback attention is working memory</a>: While Transformers have revolutionized deep learning, their quadratic attention complexity hinders their ability to process infinitely long inputs. We propose Feedback Attention Memory (FAM), a novel ...</li><li><a href="https://arxiv.org/abs/2404.10667">VASA-1: Lifelike Audio-Driven Talking Faces Generated in Real Time</a>: We introduce VASA, a framework for generating lifelike talking faces with appealing visual affective skills (VAS) given a single static image and a speech audio clip. Our premiere model, VASA-1, is ca...</li><li><a href="https://arxiv.org/abs/2404.03592">ReFT: Representation Finetuning for Language Models</a>: Parameter-efficient fine-tuning (PEFT) methods seek to adapt large models via updates to a small number of weights. However, much prior interpretability work has shown that representations encode rich...</li><li><a href="https://arxiv.org/abs/2404.10179">Scaling Instructable Agents Across Many Simulated Worlds</a>: Building embodied AI systems that can follow arbitrary language instructions in any 3D environment is a key challenge for creating general AI. Accomplishing this goal requires learning to ground langu...</li><li><a href="http://arxiv.org/abs/2404.10179">Scaling Instructable Agents Across Many Simulated Worlds</a>: Building embodied AI systems that can follow arbitrary language instructions in any 3D environment is a key challenge for creating general AI. Accomplishing this goal requires learning to ground langu...</li><li><a href="https://x.com/lambdaviking/status/1713945714684756019?s=46">Tweet from Will Merrill (@lambdaviking)</a>: [1/n] How does a chain of thought change the expressive power of transformers?  New work w/ @Ashish_S_AI studies how adding CoT/decoding steps extends the problems solvable by transformers as a fn of ...</li><li><a href="https://arxiv.org/abs/2404.07850v1">MindBridge: A Cross-Subject Brain Decoding Framework</a>: Brain decoding, a pivotal field in neuroscience, aims to reconstruct stimuli from acquired brain signals, primarily utilizing functional magnetic resonance imaging (fMRI). Currently, brain decoding is...</li><li><a href="https://arxiv.org/abs/2103.13076">Finetuning Pretrained Transformers into RNNs</a>: Transformers have outperformed recurrent neural networks (RNNs) in natural language generation. But this comes with a significant computational cost, as the attention mechanism&#39;s complexity scales...</li><li><a href="https://www.lesswrong.com/posts/gTZ2SxesbHckJ3CkF/transformers-represent-belief-state-geometry-in-their">Transformers Represent Belief State Geometry in their Residual Stream — LessWrong</a>: Produced while being an affiliate at PIBBSS[1]. The work was done initially with funding from a Lightspeed Grant, and then continued while at PIBBSS.…
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1230185118723735592)** (5 messages): 

- **Flops Estimation for ML Newcomers**: A member sought advice on estimating training flops from the SoundStream paper and was guided to calculate the number of operations per token for both forward and backward passes, using the equation **6 * # of parameters** for decoder-only transformers. They were referred to a detailed example in [Section 2.1 of a relevant paper](https://arxiv.org/abs/2001.08361).

- **One Epoch Assumption in Cost Estimation**: In response to a question about training cost estimation, one member clarified that it's wise to assume a single dataset pass unless a paper explicitly mentions performing multiple epochs. 

- **Mystery of Unreported Dataset Size**: One member highlighted the difficulty in estimating training cost from a paper, like the SoundStream paper, when details like the **size of the training dataset** are not disclosed. This poses a challenge in computing accurate cost estimates.
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1229701793215414313)** (21 messages🔥): 

- **Clarifications on Model Evaluation**: There was a discussion on how to use `lm-evaluation-harness` for evaluating custom models, specifically for the `arc_easy` task, clarifying that one should return a pair (log-likelihood, is_greedy_decoding_equal_target) from `loglikelihood`. It was noted that for tasks like ARC, where there are multiple choices, the likelihood of each combination of question and answers is evaluated, and the one with the highest likelihood deemed the correct answer. 

- **Understanding BPC as a Metric**: A paper was discussed that correlates models' intelligence with their ability to compress text, using BPC (bits per character) as a proxy for intelligence. The benefits of considering BPC over loss were debated, with the conclusion that BPC is a unit of information rather than just loss, which aligns it more closely with compression capabilities.

- **Branch Comparisons and Evaluations**: There was an inquiry about the improvements in the `big-refactor` branch over the main branch of a project which apparently offers significantly better speed. Also, another user wondered about saving generation results per question using `vllm` and learned that using the `--log_samples` flag allows logging individual responses rather than just aggregate scores.

- **Leveraging Acceleration Tools for Better Performance**: It was suggested that using the `--batch_size` argument or `accelerate launch --no_python lm-eval` could be beneficial when evaluating large models, especially on a pod of 8 A100s, to potentially improve speed and performance.

- **Assistance with Model Evaluation Methods**: One user had a doubt about the `arc_easy` task always resulting in 0.25 performance when returning random debug values and learned that since ARC has four possible answers and a random selection would result in a roughly 25% correctness rate. It was explained how tasks like MMLU and lambada_openai use the loglikelihood outputs differently to calculate accuracy.

**Link mentioned**: <a href="https://x.com/arankomatsuzaki/status/1780073500536872990">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Compression Represents Intelligence Linearly  LLMs&#39; intelligence – reflected by average benchmark scores – almost linearly correlates with their ability to compress external text corpora  repo: ht...

  

---


**Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1229989040820977775)** (1 messages): 

- **Exploring Multi-Modal Learning**: jubei_ shared [two papers on arXiv](https://arxiv.org/abs/2302.12247) regarding multi-modal machine learning. The first paper proposes an information-theoretic approach named **Total Correlation Gain Maximization (TCGM)** for semi-supervised multi-modal learning that effectively utilizes unlabeled data across modalities and offers theoretical guarantees.

- **Dive into Semi-Supervised Multi-Modal Fusion**: The discussed paper addresses the challenges of labeling large datasets for multi-modal training, and emphasizes on an approach that could improve the efficiency of fusion in semi-supervised settings. *Abstract excerpts* mentioned offer insights into the promise of the **TCGM** method for identifying Bayesian classifiers in multi-modal learning scenarios.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2302.12247">Quantifying &amp; Modeling Multimodal Interactions: An Information Decomposition Framework</a>: The recent explosion of interest in multimodal applications has resulted in a wide selection of datasets and methods for representing and integrating information from different modalities. Despite the...</li><li><a href="https://arxiv.org/abs/2007.06793">TCGM: An Information-Theoretic Framework for Semi-Supervised Multi-Modality Learning</a>: Fusing data from multiple modalities provides more information to train machine learning systems. However, it is prohibitively expensive and time-consuming to label each modality with a large amount o...
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1229892954500370442)** (10 messages🔥): 

- **IDEFICS-2 Premieres with Superior Multimodal Abilities**: [IDEFICS-2](https://huggingface.co/spaces/HuggingFaceM4/idefics-8b) is unveiled, touting 8B parameters, Apache 2.0 license, high-resolution image processing up to 980 x 980, and two checkpoints including instruction fine-tuning. This multimodal model excels in tasks such as visual question answering and document retrieval.
  
- **Chatbot Variant of IDEFICS-2 on the Horizon**: The chat-focused variant of IDEFICS-2 is expected to be released in the coming days. The current version is adept in visual question answering and other non-chat tasks, with a chatty version soon to follow.

- **Clever Multimodal Interaction Showcased**: An example shared demonstrates IDEFICS-2's capabilities, seamlessly blending text recognition, color knowledge, and mathematical operations to interpret and manipulate image contents, including solving CAPTCHAs with significant background noise.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/HuggingFaceM4/idefics-8b">Idefics 8b - a Hugging Face Space by HuggingFaceM4</a>: no description found</li><li><a href="https://x.com/lunarflu1/status/1780228654397599904">Tweet from lunarflu (@lunarflu1)</a>: cool multimodal interaction from IDEFICS-2 @huggingface : 1. Detect numbers from image 2. Do math with the number 3. Retrieve background color 4. Remove pigment -&gt; Resulting color 5. Final result: ...</li><li><a href="https://huggingface.co/blog/idefics2">Introducing Idefics2: A Powerful 8B Vision-Language Model for the community</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics2-8b">HuggingFaceM4/idefics2-8b · Hugging Face</a>: no description found</li><li><a href="https://x.com/reach_vb/status/1779998271546474593">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Idefics 2 x Transformers! 🔥  Trying out the Idefics 2 8B in the wild.  Pretty wild that you can do all this in less than 10 lines of code!  Made a quick screencast taking the model out for a spin..  ...</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized">argilla/distilabel-capybara-dpo-7k-binarized · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/papers/2403.07691">Paper page - ORPO: Monolithic Preference Optimization without Reference Model</a>: no description found</li><li><a href="https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-141b-A35b">alignment-handbook/recipes/zephyr-141b-A35b at main · huggingface/alignment-handbook</a>: Robust recipes to align language models with human and AI preferences - huggingface/alignment-handbook</li><li><a href="https://x.com/narsilou/status/1778887423713333648">Tweet from Nicolas Patry (@narsilou)</a>: Tgi 2.0 is out!  -back to fully open source for good (apache 2.0) - Fastest inference server in existence (110 tok/s for cohere R+, with medusa speculation) - fp8 support - mixtral 8x22b support ! (al...</li><li><a href="https://x.com/xenovacom/status/1778812177215881395">Tweet from Xenova (@xenovacom)</a>: Introducing MusicGen Web: AI-powered music generation directly in your browser, built with 🤗 Transformers.js! 🎵  Everything runs 100% locally, meaning no calls to an API! 🤯 Served as a static websi...</li><li><a href="https://x.com/AndrewYNg/status/1779905922602782752">Tweet from Andrew Ng (@AndrewYNg)</a>: LLMs can take gigabytes of memory to store, which limits what can be run on consumer hardware. But quantization can dramatically compress models, making a wider selection of models available to develo...</li><li><a href="https://huggingface.co/blog/vlms">Vision Language Models Explained</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1229751537140502558)** (85 messages🔥🔥): 

- **Langchain Learning Inquiry**: A participant expressed an interest in *learning langchain* to build an agentic LLM, but received advice from another member suggesting that it might be more efficient to *implement a custom solution*.

- **Seeking ML Community Insights**: A survey link was shared by students researching the *democratization of ML*, asking for participation from the machine learning community. The survey was accessible through [this link](https://forms.gle/UvGdWrZhphoDFGQ99).

- **File Conversion Hiccup**: A member encountered an issue while converting HuggingFace safetensors to llama.cpp GGUF, receiving an "is not a directory" error. They were advised to ensure the path ends before the file name in the command.

- **Unsolicited Academic Abstract Spitfire Explained**: A user experienced issues with llama.cpp generating unsolicited content when started in interactive mode, inadvertently outputting abstracts like "Anti-fungal properties of silver nanoparticles". The discussion moved towards seeking a solution or a correct command to make the interaction responsive to user input.

- **Exploring Decoder-only Models for SQUAD**: An inquiry was made regarding how to postprocess decoder-only model outputs, like Mistral’s, for SQUAD evaluation. The member was looking for inspiration from *open github repos* for handling such a task.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/zero-gpu-explorers">zero-gpu-explorers (ZeroGPU Explorers)</a>: no description found</li><li><a href="https://huggingface.co/docs/diffusers/using-diffusers/inpaint">Inpainting</a>: no description found</li><li><a href="https://forms.gle/UvGdWrZhphoDFGQ99">The Democratisation of Machine Learning - Survey</a>: Thank you for taking the time to answer this survey about people’s experience with machine learning, it should take no more than 5 min  Throughout this survey &#39;Machine Learning&#39; will be referr...</li><li><a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>: How to build Long term memory &amp; Self improving ability into your AI Agent?Use AI Slide deck builder Gamma for free: https://gamma.app/?utm_source=youtube&amp;utm...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1229984219179257969)** (3 messages): 

- **Exploring Knowledge Graphs**: A member shared a [blog post](https://mlabonne.github.io/blog/posts/Article_Improve_ChatGPT_with_Knowledge_Graphs.html) discussing how to improve Chatbot performance by integrating **Knowledge Graphs**, providing a link to explore the concept further.

- **The Quest for Quantization Knowledge**: A member is learning about **quantization** through a short course offered by Deep Learning AI, indicating ongoing education in machine learning optimization techniques.

- **Multilingual Text Retrieval with RAG**: A member asked for tips on implementing an efficient retrieval system using **Retrieval-Augmented Generation (RAG)** for a multilingual set of texts, and is looking for updates or best practices in multilingual scenarios.

**Link mentioned**: <a href="https://mlabonne.github.io/blog/posts/Article_Improve_ChatGPT_with_Knowledge_Graphs.html">ML Blog - Improve ChatGPT with Knowledge Graphs</a>: Leveraging knowledge graphs for LLMs using LangChain

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1229937791295553546)** (7 messages): 

- **Splatter Art with Speed**: The [Splatter Image space on HuggingFace](https://huggingface.co/spaces/szymanowiczs/splatter_image) is a quick tool to generate **splatter art**.

- **Diving into Multi-Modal RAG**: A speaker from **LlamaIndex** shared resources about **Multi-Modal RAG (Retrieval Augmented Generation)**, showcasing applications that combine language and images. Discover how **RAG's** indexing, retrieval, and synthesis processes can integrate with the image setting in their [documentation](https://docs.llamaindex.ai/en/stable/use_cases/multimodal/).

- **LLM User Analytics Unveiled**: Nebuly introduced an **LLM user analytics playground** that's accessible without any login, providing a place to explore analytics tools. Feedback is requested for [their platform](https://playground.nebuly.com/home?projectId=69269458-99d7-4022-abb4-949c7b352649&homeTab=Overview).

- **ML Expanding into New Frontiers**: The IEEE paper highlights an *interesting scenario* where **Machine Learning (ML)** can be widely applied. The paper can be found at the [IEEE Xplore digital library](https://ieeexplore.ieee.org/abstract/document/9249641).

- **Snowflake Introduces Top Text-Embedding Model**: Snowflake launched the **Arctic embed family of models**, claiming to be the world’s best practical text-embedding model for retrieval use cases. The family of models surpasses others in average retrieval performance and is open-sourced under an Apache 2.0 license, available on [Hugging Face](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) and soon in Snowflake's own ecosystem. Read more in their [blog post](https://www.snowflake.com/blog/introducing-snowflake-arctic-embed-snowflakes-state-of-the-art-text-embedding-family-of-models/).

- **Multi-Step Tools Enhancing Efficiency**: An article on Medium discusses how **multi-step tools** developed by LangChain and Cohere can unlock efficiency improvements in various applications. The full discourse is available in the provided [Medium article](https://medium.com/ai-advances/unlocking-efficiency-the-power-of-multi-step-tools-with-langchain-and-cohere-7d1ea571ebed).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/szymanowiczs/splatter_image">Splatter Image - a Hugging Face Space by szymanowiczs</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/multimodal/">Multi-Modal Applications - LlamaIndex</a>: no description found</li><li><a href="https://playground.nebuly.com/home?projectId=69269458-99d7-4022-abb4-949c7b352649&homeTab=Overview">Nebuly AI</a>: no description found</li><li><a href="https://www.snowflake.com/blog/introducing-snowflake-arctic-embed-snowflakes-state-of-the-art-text-embedding-family-of-models/">Snowflake Launches Practical Text-Embedding Model for Retrieval use Cases</a>: Snowflake-arctic-embed is available to the open source community under an Apache 2.0 license.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1229733181960224798)** (19 messages🔥): 

- **BLIP Model Fine-tuned for Prompts**: The BLIP model has been fine-tuned to generate long captions suitable for image prompts, with a live demo accessible on Hugging Face. Check out the enhanced capabilities [here](https://huggingface.co/unography/blip-large-long-cap).
  
- **Model Comparison Made Easy**: A Hugging Face Space comparing different image captioning models has been published and duplicates the existing comparison space by another user. [Explore the model comparisons](https://huggingface.co/spaces/unography/comparing-captioning-models).

- **Support for Maximum Output Length in Serverless Inference**: Queries were made about max output length for model inference via curl, and it was clarified that parameters supported in transformers' pipelines can be used, including `max_new_tokens`.

- **IP-Adapter Playground Unveiled**: A new Hugging Face Space featuring IP-Adapter, which allows for text-to-image, image-to-image, and inpainting functionalities using images as prompts, has been launched. Dive into the [IP-Adapter Playground](https://huggingface.co/spaces/tonyassi/IP-Adapter-Playground).

- **'Push to Hub' Added to Transformers' Pipelines**: The main branch of the transformers library now includes a `push_to_hub` method, allowing pipeline outputs to be pushed directly to the Hugging Face Model Hub. Users can try this feature from the main branch or wait for the next release.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/EduardoPacheco/Grounded-SAM">Grounded SAM - a Hugging Face Space by EduardoPacheco</a>: no description found</li><li><a href="https://huggingface.co/spaces/tonyassi/IP-Adapter-Playground">IP-Adapter Playground - a Hugging Face Space by tonyassi</a>: no description found</li><li><a href="https://playground.nebuly.com/home?projectId=69269458-99d7-4022-abb4-949c7b352649&homeTab=Overview">Nebuly AI</a>: no description found</li><li><a href="https://huggingface.co/spaces/unography/comparing-captioning-models">Comparing Captioning Models - a Hugging Face Space by unography</a>: no description found</li><li><a href="https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task">Detailed parameters</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.ImageToTextPipeline">Pipelines</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/pipelines#transforme">Pipelines</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1229691536586051624)** (11 messages🔥): 

- **Seeking an SDXL Tagger Upgrade**: A member inquired about alternative taggers to the **wd14 tagger** for SDXL, searching for improved options.

- **Quest for PDF to LaTeX Conversion Tools**: A member asked if there's any **open-source PDF to LaTeX** converters, or an **image to LaTeX** converter capable of processing an entire PDF page, including text and mathematical expressions, without requiring exact positioning.

- **LaTeX-OCR for Equation Conversion**: It was pointed out that there's a **good open-source repository for converting images of equations into LaTeX code**: [LaTeX-OCR on GitHub](https://github.com/lukas-blecher/LaTeX-OCR), which utilizes a Vision Transformer (ViT).

- **No Perfect LaTeX Conversions for Text**: The conversion of text to LaTeX is complex due to LaTeX compilers and package particularities, leading to the opinion that manual rewriting may be more functional.

- **Selective Text Extraction Challenge**: A user is looking for a method to extract one specific line of text from an image, based on the largest and boldest font. It was recommended to try **Paddle OCR** for this task.

**Link mentioned**: <a href="https://github.com/lukas-blecher/LaTeX-OCR">GitHub - lukas-blecher/LaTeX-OCR: pix2tex: Using a ViT to convert images of equations into LaTeX code.</a>: pix2tex: Using a ViT to convert images of equations into LaTeX code. - lukas-blecher/LaTeX-OCR

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1229757182921150474)** (17 messages🔥): 

- **LoRA Configuration Queries**: A member is experimenting with their LoRA configuration and is seeking advice on the implications of setting the bias to 'all', 'none', or 'lora_only'.

- **Preparing Dataset for Fine-tuning Roberta**: One member is looking for guidance on preparing a CSV dataset with over 100,000 entries and 20+ features for fine-tuning a ROBERTA model for a question-answering chatbot. Following up, they clarified that the dataset includes details about pharmaceutical drugs with diverse columns such as release date and drug type.

- **BERTopic for Topic Modeling**: A member recommended [BERTopic](https://maartengr.github.io/BERTopic/index.html), a topic modeling technique using 🤗 transformers and c-TF-IDF, and reports satisfaction with the results, though there's a current challenge to convert seed words to phrases for creating topic models.

- **Seeking T5 Training Code with HF Trainer**: A member inquires where to find training code for T5 using Hugging Face's Trainer. Another member shared a [link to EleutherAI's GitHub](https://github.com/EleutherAI/improved-t5) repository with open-source scripts for an improved T5 and suggested looking into [simpleT5](https://github.com/Shivanandroy/simpleT5) for a more straightforward approach.

- **Resuming Model Download in AutoModelForVision2Seq**: A member questions how to resume a model download process using AutoModelForVision2Seq, but did not receive a direct response.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://maartengr.github.io/BERTopic/index.html">Home</a>: Leveraging BERT and a class-based TF-IDF to create easily interpretable topics.</li><li><a href="https://github.com/EleutherAI/improved-t5">GitHub - EleutherAI/improved-t5: Experiments for efforts to train a new and improved t5</a>: Experiments for efforts to train a new and improved t5 - EleutherAI/improved-t5</li><li><a href="https://github.com/Shivanandroy/simpleT5">GitHub - Shivanandroy/simpleT5: simpleT5 is built on top of PyTorch-lightning⚡️ and Transformers🤗 that lets you quickly train your T5 models.</a>: simpleT5 is built on top of PyTorch-lightning⚡️ and Transformers🤗 that lets you quickly train your T5 models. - Shivanandroy/simpleT5
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1229717436723626024)** (8 messages🔥): 

- **Truncated Tokens Concern**: A user mentioned that **truncated tokens**, such as "hdr" in their prompt, are being ignored, implying a potential problem in processing. There was agreement on this issue, but no solution provided in the discussion.
- **Compel Library Maintenance**: In response to the truncated token problem, the **Compel library** was mentioned, but there is a concern that it may not currently be maintained.
- **Model for Analysis and Text Generation from Video**: A request for a model capable of **analyzing video content** to generate titles and descriptions was posed, but the discussion thread does not provide a solution.
- **Solicitation for Test Method Roast**: A user shared a link to a **testing method/suite** and requested some constructive criticism from a user perspective. The content of the test method/suite was not discussed.
- **Resume Hugging Face Model Training**: A user asked about the necessary code changes required to **resume a Hugging Face model**, but no answers have been given in the conversation.
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1229737177861066762)** (44 messages🔥): 

- **Idefics2's Grand Entrance**: A brand new multimodal model, [Idefics2](https://huggingface.co/HuggingFaceM4/idefics2-8b), is available now, accepting both image and text inputs and boasting improved OCR and visual reasoning over its predecessor, Idefics1. It has been released with two checkpoints, featuring base and fine-tuned versions, and is licensed under Apache 2.0.
  
- **Pre-emptive Strike by NVidia?**: Rumors are circulating that NVidia might expedite the launch of the [RTX 5090](https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch), possibly as early as June 2024 at the Computex tradeshow, in response to competitive pressure from AMD's new advancements.

- **Hardware Conversations on AI Training**: Members discussed the feasibility of using Nvidia's A6000 GPUs for training and inference with models such as QLoRa, debating on the sufficiency of VRAM and potential requirement for more powerful setups.

- **Cosmo-1b Forgetting and Merging Experiments Revealed**: In experiments to compare training methods aimed at reducing catastrophic forgetting, Model Stock merge revealed potential in combining various training solutions. The sharing of detailed comparison stats in training set validation results stirred interest in further exploring the strengths of different fine-tuning approaches.

- **Technical Dig into Dora and QLoRa**: Users engaged in a technical discussion about the effectiveness of new parameter-efficient fine-tuning (PEFT) methods like Dora, comparing it to QLoRa, discussing configuration details, and noting the peculiarities in performance and resource consumption of each method.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.pcgamesn.com/nvidia/rtx-5090-5080-paper-launch">Nvidia’s RTX 5090 and 5080 could arrive much sooner than expected, but there’s a big catch</a>: Leaks point to the new Nvidia Blackwell GeForce GPUs arriving much sooner than originally expected, thanks to competition from AMD.</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics2-8b">HuggingFaceM4/idefics2-8b · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1229811042507559054)** (2 messages): 

- **Inquiry on Bot Utility**: A user expressed curiosity with a simple "Oooooo how do I use this?" indicating interest in understanding the bot's functions.
- **Spam Alert**: A spam message aimed at the entire group advertised inappropriate content with a Discord invite link.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[manticore](https://discord.com/channels/1104757954588196865/1107775871428870177/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1229719185668702310)** (13 messages🔥): 

- **Clarifying Role of 'train_on_input' Flag**: A discussion on the 'train_on_input' parameter unfolded, revealing that disabling it means the model doesn't calculate loss for the input portion, hence not predicting it anymore. This clarifies that the input forms part of the context during training regardless, but with the parameter off, the model won't be steered by the input in terms of loss calculation.

- **Understanding Loss in Training**: It was highlighted that *loss* is indeed a crucial aspect of training as it guides model improvement, and disabling 'train_on_input' stops this process for the input part. If the *eval* setting is not enabled, this process becomes even less relevant to the model's learning. 

- **Query About Cost and OnlyFans Link**: One member inquired about the cost of an unspecified service, and another user posted what seems to be a promotional message for an OnlyFans related link inviting members to join another Discord server with the promise of exclusive content.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1230227447849681008)** (3 messages): 

- **Inappropriate Content Alert**: The channel experienced an instance of spam advertising *OnlyFans leaks and explicit content* with an invite link to a Discord server.
- **Community Watchdogs in Action**: Members quickly identified the spam and labeled it as *pornspam*, alerting others about the inappropriate nature of the messages.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[hippogriff](https://discord.com/channels/1104757954588196865/1113355318211137576/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[minotaur](https://discord.com/channels/1104757954588196865/1116465236715786310/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[bots](https://discord.com/channels/1104757954588196865/1117282691121954836/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[deployment-help](https://discord.com/channels/1104757954588196865/1163840836472148058/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/)** (1 messages): 

aquash1553: @everyone Best OnlyFans Leaks & Teen Content 🍑 🔞 discord.gg/s3xygirlsss
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1229738683494699008)** (36 messages🔥): 

- **Simplifying Epoch-wise Model Saving**: A member queried about configuring Axolotl to save a model only at the end of training and not every epoch. The solution involved adjusting `save_strategy` in the training arguments to `"no"` and implementing a custom callback for a manual save upon training completion.

- **Choosing a Starter Model for Fine-Tuning**: When asked for a suitable small model for fine-tuning, "TinyLlama-1.1B-Chat-v1.0" was recommended due to its manageability for quick experiments. Members were guided to the Axolotl repository for example configurations like `pretrain.yml`.

- **Guidance on Axolotl Usage and Data Formatting**: There was a discussion on concepts like `model_type`, `tokenizer_type`, and how to format datasets for Axolotl training, particularly in relation to using the "TinyLlama-1.1B-Chat-v1.0" model. For the task of text-to-color code generation, it was suggested to structure the dataset without "system" prompts and upload it as a Hugging Face dataset if not already available.

- **CSV Structure Clarification for Dataset Upload**: Clarification was sought on whether a one-column CSV format is needed for uploading a dataset to Hugging Face for use with Axolotl. The formatted examples should be line-separated, with each line containing the input and output structured as per model requirements.

- **Posting Inappropriate Content**: A user posted a message promoting unauthorized content, which is not relevant to the technology-oriented discussion of the channel nor adhere to community guidelines.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=c4585711-a0f4-4fe4-8055-816941329e8d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=5b7d5162-b9f5-4a2b-83e0-b2154f15fe04)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=12ea1d05-4725-46ae-ac43-42fdae27790a)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ae2df564-24d0-4c41-9f77-a8ea154566bb)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=eca9e87b-1d42-427c-8a91-59f42a3da0f8)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=ccfe189d-d5fa-4308-9afe-8a86c48a0141)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1229697419386552383)** (5 messages): 

- **Inquiry on Model Fine-Tuning**: A member sought advice on how to preprocess data for fine-tuning the **TinyLlama** model with a specific dataset containing color codes and descriptions. The goal is to train TinyLlama to predict a color code from a given description.

- **Guidance on Model Preparation**: A response outlined steps for fine-tuning **TinyLlama** by preparing the dataset in a usable format and performing tokenization and formatting suitable for the task. No specific details or links were provided in the response.

- **Irrelevant Content Posted**: An off-topic message advertising **OnlyFans leaks and content** was posted to the channel. The message provided a Discord join link.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=af0c71b5-451f-4893-8158-1dfa36a9a10b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1229722146042023997)** (68 messages🔥🔥): 

- **Comprehensive LLM Benchmarks Available**: An informative website [llm.extractum.io](https://llm.extractum.io/) has been shared which provides a detailed overview of open-source language models ranked by various benchmarks. The models are rated using ELO scores, HuggingFace leaderboard scores, and several task-specific accuracy measurements.
- **AI Agents Employing Humans**: An innovative project called [Payman AI](https://www.paymanai.com/) was introduced, enabling AI agents to pay humans for tasks they can't perform themselves. This service aims to support a symbiotic relationship between AI and humans across various sectors like design, coding, and law.
- **AI Inference Integrated into Supabase**: Supabase has announced an easy-to-use API for running AI inference models within its edge functions. A new session initialization allows AI models like `gte-small` to process inquiries directly within the database service.
- **Anticipating "Llama 3" Launch**: Discussions include speculations and rumors about the release of "Llama 3", with anticipation building within the community. The context suggests that the reveal of Llama 3 may be linked to an upcoming hackathon in London.
- **OpenAI's API Expansion Ahead of GPT-5**: [OpenAI's introduction of updates to the Assistants API](https://x.com/OpenAIDevs/status/1780640119890047475) has been brought to light, encouraging discussion about the directions the company could be taking, particularly with the possible launch of GPT-5 on the horizon. Users are debating the quality and performance of such platforms and the potential impact on AI startups.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mixtral-8x22b/">Cheaper, Better, Faster, Stronger</a>: Continuing to push the frontier of AI and making it accessible to all.</li><li><a href="https://www.paymanai.com/">Payman - Home</a>: no description found</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1779917676133105732">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: I’ll be sharing more on Llama 3 very soon. It’s so cool to see what the community is already building with Llama 2 though. One of my favorites: @team_qanda & @UpstageAI used it to build a math-specifi...</li><li><a href="https://x.com/suchenzang/status/1701747947191615697?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Susan Zhang (@suchenzang)</a>: MBPP might&#39;ve also been used somewhere in the Phi-1.5 dataset.  Just like we truncated one of the GSM8K problems, let&#39;s try truncating the MBPP prompts to see what Phi-1.5 will autocomplete wi...</li><li><a href="https://strongcompute.com/research-grants">Research Grants</a>: no description found</li><li><a href="https://supabase.com/blog/ai-inference-now-available-in-supabase-edge-functions">AI Inference now available in Supabase Edge Functions</a>: Use embeddings and large language models on the edge with Supabase Edge Functions.</li><li><a href="https://x.com/OpenAIDevs/status/1780640119890047475">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Introducing a series of updates to the Assistants API 🧵  With the new file search tool, you can quickly integrate knowledge retrieval, now allowing up to 10,000 files per assistant. It works with our...</li><li><a href="https://x.com/russelljkaplan/status/1513128005828165634">Tweet from Russell Kaplan (@russelljkaplan)</a>: Second order effects of the rise of large language models:</li><li><a href="https://x.com/yoheinakajima/status/1780061516051755168?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Yohei (@yoheinakajima)</a>: A marketplace for AI agents to hire humans 🧠  ↘️ Quoting tyllen (@0xTyllen)   Excited to introduce a new project I&#39;ve been working on called Payman!    Payman is an AI Agent tool that gives Agent...</li><li><a href="https://x.com/armandjoulin/status/1780638511818838378">Tweet from Armand Joulin (@armandjoulin)</a>: Fixed the fix.  ↘️ Quoting Jonathan Frankle (@jefrankle)   Fixed it for you, @code_star</li><li><a href="https://www.youtube.com/watch?v=xZiTSZ5SOYc&t=9s">Payman - Enabling AI Agent To Human Payments!</a>: Hey everybody, in this video, I&#39;m super excited to show you Payman, a platform that allows you to connect your agents with capital that they can use to pay h...</li><li><a href="https://llm.extractum.io/">LLM Explorer: A Curated Large Language Model Directory. LLM List. 35061 Open-Source Language Models.</a>: Browse 35061 open-source large and small language models conveniently grouped into various categories and llm lists complete with benchmarks and analytics.
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1230058571975102464)** (1 messages): 

- **Paper Club Meeting on BloombergGPT**: A **BloombergGPT** discussion is scheduled, with `<@315351812821745669>` leading it, supported by `<@451508585147400209>`. Participants are reminded to sign up [here](https://lu.ma/w7jhce1y) and note the return to Zoom due to past Discord screenshare issues.


**Link mentioned**: <a href="https://lu.ma/w7jhce1y">LLM Paper Club (BloombergGPT / TimeGPT paper) · Zoom · Luma</a>: This week @yikes will be covering BloombergGPT: https://arxiv.org/abs/2303.17564 Also submit and vote for our next paper:…

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1230231139893903394)** (19 messages🔥): 

- **Acknowledgment of Efforts**: A member expresses appreciation for the time and effort the community members put into organizing the event.
- **Zoom Meeting Transition**: It was announced that the discussion would move from Discord to a [Zoom meeting](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09), with multiple members sharing the same link and directing the participants to the new location.
- **Quick Zoom Reminder**: Further notifications were posted tagging specific members, prompting them to join the [Zoom meeting](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09).
- **Zoom Entry Request**: A member mentioned their dislike for Zoom but indicated their intention to join, asking for admission into the meeting.

**Link mentioned**: <a href="https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1229787566002143355)** (59 messages🔥🔥): 

- **AI Wearables vs Smartphones**: A user shared a [YouTube review by Marquis Brownlee](https://youtu.be/TitZV6k8zfA) and generated discussion around the limitations of AI wearables compared to modern smartphones. The conversation touched on the potential need for AI assistants to have deep contextual knowledge for more efficient responses.
  
- **Anticipation for the OpenSource WizardLm2**: Members express enthusiasm for the WizardLm2 model, praising its perceived freedom from censorship and the significant leap towards GPT-4 level capabilities in an open-source model. Discussions hint at the perpetual desire for the next improvement even as current advancements are celebrated.
  
- **Translation Bot Testing and Objectives**: The new translation bot is under examination, with goals to facilitate more inclusive conversations by translating both ways. Users seem optimistic about its potential to unify discussions.
  
- **Communal Quest for Windows Compatibility**: Multiple users are voicing their struggles to get software, particularly the 01 Light software, to function on Windows. The conversation reveals a pressing need for Windows support to make enterprise inroads and the challenges faced with Mac-oriented setups.
  
- **Exploring Hardware Options and Personal AI Aspirations**: There's active chatter about various AI hardware options like the Limitless device, with users comparing personal experiences and desires for an integrated, personal AI assistant. Some spotlight the importance of backend infrastructure and seamless integration as the next frontiers in AI hardware development.

**Link mentioned**: <a href="https://youtu.be/TitZV6k8zfA?t=900&si=zsI6zFfyJ8aBATzf).">The Worst Product I&#39;ve Ever Reviewed... For Now</a>: The Humane AI pin is... bad. Almost no one should buy it. Yet.MKBHD Merch: http://shop.MKBHD.comTech I&#39;m using right now: https://www.amazon.com/shop/MKBHDIn...

  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1229696840765800461)** (17 messages🔥): 

- **Portable O1 Setup Brainstorming**: A member shared their aim to create a somewhat portable O1 setup using an RPi5 to run OI, with Arduino components involved. Others suggested that simpler, cheaper components like the m5 atom could be sufficient and asked about the member's specific goals for the setup.
- **Shipping Dates for O1 Mystery**: In response to an inquiry about an unspecified item or product, a member mentioned that shipping is aimed to start by the end of summer, but no specific dates are confirmed yet.
- **Terminal Choices for Successful Responses**: Users discussed their preferences for terminal applications, with one member successfully using **Windows Terminal** and **Powershell** to get responses. There was a mention of difficulties with recognizing the OpenAI key in Powershell for Windows 10.
- **Batch Files as a Workaround in Windows**: A member admitted to using a **batch file** because they found it more convenient, implying that it is processed by cmd.exe rather than Powershell, highlighting the quirks of Windows.
- **Troubleshooting Request for Latest Branch**: There was a request for testing the latest branch due to several people experiencing issues with connection establishment and audio uploading.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://amzn.eu/d/4GbeU5b">no title found</a>: no description found</li><li><a href="https://amzn.eu/d/fIr3Lzu">no title found</a>: no description found</li><li><a href="https://amzn.eu/d/eZQoRwD">no title found</a>: no description found
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1229974543561658419)** (11 messages🔥): 

- **Significant Improvement in Winrate**: A member shared a project update, highlighting a method that improved the winrate of **qwen-1.5-0.5B** from **4% to 32%** against AlpacaEval, Phi-2, and Gemma2b-it, using a combination of *generation in chunks* and a small (300M) reward model for output searching.
- **Seeking Validation for a Simple Method**: The same member mentioned the simplicity of their method that led to increased winrate on a **500M base model**, and sought feedback to verify the effectiveness of this approach.
- **Relevance of Reranking LLM Outputs**: Another community member acknowledged that **reranking LLM outputs** during inference is a known practice but was unsure if it had been applied to AlpacaEval before; also referencing a paper on reranking and pruning during parallel generation.
- **Research Papers as verification**: The previous member then provided links to papers discussing the approach, indicating that the terms **verifier/reward guided decoding** are associated with the method, including [2305.19472](https://arxiv.org/pdf/2305.19472.pdf) and [2402.01694](https://arxiv.org/pdf/2402.01694.pdf)
- **Underexplored but Promising**: A member agreed on the potential of such an **underexplored area**, implying that concepts like **MCTS PPO** might also be worth examining.
  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1230153917103345785)** (17 messages🔥): 

- **Mixtral-8x22B LLM Gains Attention**: A new model called [Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b/) has been touted for setting high performance and efficiency standards. It's an SMoE model, fluent in several languages, capable of function calling, and offers a 64K token context window, all under the Apache 2.0 license.

- **Mixtral-8x22B-Instruct's Chatbot Capabilities Discussed**: The instruct fine-tuned version of Mixtral-8x22B, [Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1), garnered attention for its potential in the chatbot arena, featuring detailed instructions on how to run the model.

- **Impressive OLMo 1.7 7B Model Upgrade**: [OLMo 1.7 7B](https://huggingface.co/allenai/OLMo-1.7-7B) has made waves with its 24 point increase on MMLU, training on an improved version of the Dolma dataset and staged training. It's part of a series of models designed to promote the science of language models.

- **A Proposal for Web Page Quality Propagation**: The idea of applying "web page quality" propagation to rank web pages was floated, involving a quality score boosted by backlinks and decreased by linking to low-quality sites.

- **Reflection on Common Crawl's Dense Web Graph**: The complexity of evaluating 'quality' content based on Common Crawl's web graph was discussed, noting that the graph does not indicate the success of the linearization process (the conversion of HTML into plain text).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mixtral-8x22b/">Cheaper, Better, Faster, Stronger</a>: Continuing to push the frontier of AI and making it accessible to all.</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://fxtwitter.com/AlbertQJiang/status/1780648008696091003">Tweet from Albert Jiang (@AlbertQJiang)</a>: I love open-sourced models! Please add your favourites to the Mistral Convex Hull.  ↘️ Quoting Philipp Schmid (@_philschmid)   Fixed the Fixed Fix for @AI21Labs and included Mambas. 🐍</li><li><a href="https://commoncrawl.org/web-graphs">Common Crawl - Web Graphs</a>: Detailing Common Crawl&#x27;s Web Graph releases, the technology behind them, and how to use them.</li><li><a href="https://huggingface.co/allenai/OLMo-1.7-7B">allenai/OLMo-1.7-7B · Hugging Face</a>: no description found</li><li><a href="https://fxtwitter.com/_philschmid/status/1780641241668997258">Tweet from Philipp Schmid (@_philschmid)</a>: Fixed the Fixed Fix for @AI21Labs and included Mambas. 🐍  ↘️ Quoting Armand Joulin (@armandjoulin)   Fixed the fix.
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1230219724961353888)** (9 messages🔥): 

- **Chinchilla Paper Under Scrutiny**: The [Chinchilla scaling paper by Hoffmann et al.](https://x.com/tamaybes/status/1780639257389904013?s=46) is facing replication challenges, with discrepancies found when others tried to replicate a key part of the research.
- **Doubts Cast on Scaling Law Papers**: A member expressed skepticism about the conclusions from scaling law papers, hinting at issues with the math upon closer examination of the [Chinchilla paper](https://x.com/suchenzang/status/1616752482226671620?s=46).
- **Community Engagement Over Questions with Chinchilla**: Discord users are engaging with the issue, sharing brief reactions of concern and surprise, using phrases such as *"Chinchilla oops?"* and simply *"oh no"* to express discomfort regarding the situation.
- **Authors Non-responsive to Clarification Requests**: One of the replication attempters mentioned that they reached out to the original authors for clarification but did not receive any response, adding to the frustration within the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tamaybes/status/1780639279506432473?s=46">Tweet from Tamay Besiroglu (@tamaybes)</a>: We have asked the authors for assistance, but we haven’t been able to get a response. (8/9)</li><li><a href="https://x.com/suchenzang/status/1616752482226671620?s=46">Tweet from Susan Zhang (@suchenzang)</a>: After ignoring the details in all these &#34;lets-fit-a-cloud-of-points-to-a-single-line&#34; papers (all likely wrong when you really extrapolate), @stephenroller finally convinced me to work through...</li><li><a href="https://x.com/tamaybes/status/1780639257389904013?s=46">Tweet from Tamay Besiroglu (@tamaybes)</a>: The Chinchilla scaling paper by Hoffmann et al. has been highly influential in the language modeling community. We tried to replicate a key part of their work and discovered discrepancies. Here&#39;s ...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

natolambert: shittiest leaderboard winner lol
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1229816228617523340)** (23 messages🔥): 

- **WizardLLM Code Inquiry**: A community member inquired about forking the **WizardLLM** code; another confirmed that model weights are publicly available suggesting it may return soon.
- **Anticipation for olmo vs llama 3**: Multiple members engaged in a light-hearted discussion about **olmo vs llama 3**, with a suggestion that a new battle may be upcoming, despite humorous resignation to its outcome.
- **Forecast for Prolific Blogging**: Nathan Lambert hinted at a potentially heavy week of content sharing, **expecting to possibly release three blog posts**.
- **Discussing Aesthetic Changes in the Chaotic Era**: Conversations in the **Chaotic Era** included tweaks to user interface annoyances and personal preferences on profile imagery.
- **Twitter and Memes Conversation**: Members chatted casually about their Twitter activity, shareability of content, and the possibility of one's post aligning with **"sacred numerology"** due to coincidental abbreviation.
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1230181805106401361)** (3 messages): 

- **AI Livestream Hijinks on SNL**: Nathan shared a humorous [YouTube video](https://www.youtube.com/watch?v=86qKgK0asGo) titled "Beavis and Butt-Head - SNL," which shows a NewsNation livestream event on AI being comically disrupted. He particularly noted the first minute as being very amusing.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=86qKgK0asGo">Beavis and Butt-Head - SNL</a>: A NewsNation livestream event on AI is derailed by two audience members (Ryan Gosling, Mikey Day).Saturday Night Live. Stream now on Peacock: https://pck.tv/...

  

---


**Interconnects (Nathan Lambert) ▷ #[sp2024-history-of-open-alignment](https://discord.com/channels/1179127597926469703/1223784028428177510/)** (1 messages): 

natolambert: should I wizardLM 2 as a troll lol
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1229732276162531408)** (54 messages🔥): 

- **Cohere API Clarifications Sought**: Members are seeking **clarifications on Cohere API functionality**, with particular interest in API capabilities around system prompts and model availability. One user bumps the question, emphasizing the need for detailed information.
- **Cohere Embeddings Benchmark Inquiry**: Questions have arisen about whether **Cohere's embeddings v3 have been compared** with OpenAI's new large embeddings. A link is provided to Cohere's blog where related information can be found: [Introducing Command R+](https://txt.cohere.com/int8-binary-embeddings/).
- **Integration Challenges and Solutions**: Members are addressing technical queries regarding integrations, specifically in connecting LLMs to other platforms like BotPress, and there are discussions about whether Coral requires a locally-hosted solution. One member suggests a future update may address this.
- **Fine-Tuning Model Confusion**: One user queries about the ability to **fine-tune already fine-tuned models** through Cohere's Web UI, leading to a discussion on the process and a shared link to the official documentation: [Fine-Tuning with the Web UI](https://docs.cohere.com/docs/fine-tuning-with-the-web-ui).
- **Discord Welcomes and Personal Projects**: Various new members introduce themselves, and excitement is shared about Cohere's offerings. Discussion threads include mentions of personal projects, such as **PaperPal**, built using Cohere's Command R.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.]">no title found</a>: no description found</li><li><a href="https://ibb.co/s348vXt">Screenshot-2024-04-16-151544 hosted at ImgBB</a>: Image Screenshot-2024-04-16-151544 hosted in ImgBB</li><li><a href="https://docs.cohere.com/docs/fine-tuning-with-the-web-ui">Fine-tuning with the Web UI - Cohere Docs</a>: no description found</li><li><a href="https://txt.cohere.com/int8-binary-embeddings/">Cohere int8 &amp; binary Embeddings - Scale Your Vector Database to Large Datasets</a>: Cohere Embed now natively supports int8 and binary embeddings to reduce memory cost.</li><li><a href="https://github.com/Unstructured-IO/unstructured">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.  - GitHub - Unstructured-IO/unstructured: Open source librar...</li><li><a href="https://github.com/cohere-ai/sandbox-conversant-lib">GitHub - cohere-ai/sandbox-conversant-lib: Conversational AI tooling &amp; personas built on Cohere&#39;s LLMs</a>: Conversational AI tooling &amp; personas built on Cohere&#39;s LLMs - cohere-ai/sandbox-conversant-lib</li><li><a href="https://github.com/cohere-ai/quick-start-connectors">GitHub - cohere-ai/quick-start-connectors: This open-source repository offers reference code for integrating workplace datastores with Cohere&#39;s LLMs, enabling developers and businesses to perform seamless retrieval-augmented generation (RAG) on their own data.</a>: This open-source repository offers reference code for integrating workplace datastores with Cohere&amp;#39;s LLMs, enabling developers and businesses to perform seamless retrieval-augmented generation...
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1230065514924216320)** (3 messages): 

- **Beta Testers Wanted for Quant Fino**: A new pilot has been deployed for an Agentic entity powered by **Command-R Plus**, looking to blend GAI with FinTech and Day Trading. They are currently seeking **beta testers** and feedback, with information available at [Join Beta - Quant Fino](https://quantfino.com/join-beta) with details on their cookie policy and user consent.

- **Inquiry About Rubik's API**: A member expressed interest in utilizing Rubik's via an API with **post request** support. They are awaiting further details on whether such an API is available.

- **Redteaming Reveals Vulnerabilities in Command R+**: A member has done redteaming work on the **Command R+** model, identifying potential for creating **unrestricted agents** with capabilities for nefarious tasks. They provided a detailed write-up at [LessWrong](https://www.lesswrong.com/posts/4vPZgvhmBkTikYikA/creating-unrestricted-ai-agents-with-command-r), which includes examples of agent-produced messages geared towards harmful actions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.lesswrong.com/posts/4vPZgvhmBkTikYikA/creating-unrestricted-ai-agents-with-command-r">Creating unrestricted AI Agents with Command R+ — LessWrong</a>: TL;DR There currently are capable open-weight models which can be used to create simple unrestricted bad agents. They can perform tasks end-to-end su…</li><li><a href="https://quantfino.com/join-beta">Quantfino - Home of Powerful AI Driven Finance</a>: Quantfino is the home of LLM powered and Langchain assisted Financial Analysis.
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1229820483818623010)** (1 messages): 

- **Iterative Documentation Structure Improvements**: The team is iterating on the **documentation structure** to enhance accessibility and clarity. A new organization splitting content into 'tutorial', 'how to guides', and 'conceptual guide' is proposed, with feedback requested on the structure via the provided link.

- **LangChain Framework Introduction Highlighted**: The provided link introduces **LangChain**, an open-source framework for building applications with large language models. It details how LangChain facilitates development, productionization, and deployment through [building blocks](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/expression_language/), [LangSmith](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/langsmith/), and [LangServe](https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/langserve/), and includes a diagrammatic overview.

**Link mentioned**: <a href="https://langchain-git-harrison-new-docs-langchain.vercel.app/docs/get_started/introduction">Introduction | 🦜️🔗 LangChain</a>: LangChain is a framework for developing applications powered by large language models (LLMs).

  

---


**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1229689230977011824)** (38 messages🔥): 

- **Seeking YC Startup Insights**: A member has expressed interest in applying to **YC** for a startup focused on finetuning models for agents and is inquiring if anyone knows whether this has already been done. Another member responded by listing companies like **Unsloth, Mistral AI**, and **Lumini** that are in this space.

- **Collaborative Effort Wanted for LLM Applications**: There's an open call for those working on **LLM applications** to join in short conversations, with one member promptly expressing willingness to do so.

- **Langchain Learning Curve**: A query about whether learning **Langchain** is worthwhile received lighthearted responses suggesting that one should learn by doing and encouraging hands-on experimentation with the technology.

- **Update on Handling Tabulated Data in Langchain**: Multiple users discussed handling multiple CSV files with **Langchain** for a chatbot, with suggestions ranging from using an SQL agent to different methods of utilizing CSV files and handling larger data sets effectively.

- **Exploring RAG Optimization**: Users have brought up the challenge of dealing with large documents using **RAG**, where strategies like pre or post-index splitting were discussed, and one member shared their pursuit of optimizing RAG for better accuracy.

- **Looking for a Hiring Point Person**: A new participant greeted the channel and is seeking the appropriate contact person for discussions about **hiring**.

- **Venture into Multi-Agent Frameworks**: A member pointed towards **AutoGen**, a framework provided by Microsoft for multi-agent conversations and workflows, and sparked curiosity among users in multi-agent orchestration within **Langchain**.

- **AI Startups Funding Database Unveiled**: A comprehensive fundraising database for **AI startups** has been shared, featuring impressive data collection on financing rounds and companies, including insights from **GPT-4** with an invitation for feedback on possible data inaccuracies.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://microsoft.github.io/autogen/">AutoGen | AutoGen</a>: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework</li><li><a href="https://flashcardfy.lol/">Flashcardfy - AI Flashcard Generator with Personalized Feedback</a>: Learn faster and smarter with AI-generated flashcards that provide personalized feedback.</li><li><a href="https://js.langchain.com/docs/use_cases/sql/agents">Agents | 🦜️🔗 Langchain</a>: LangChain offers a number of tools and functions that allow you to create SQL Agents which can provide a more flexible way of interacting with SQL databases. The main advantages of using SQL Agents ar...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1229808687632089131)** (1 messages): 

- **Integration Challenges with LangServe and Nemo Guardrails**: A member inquired about difficulties encountered when trying to integrate **LangServe** with a chain that includes **Nemo Guardrails**, as Nemo alters the output structure significantly. They mentioned the necessity for a novel output parser to handle these changes.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1229815994789265489)** (4 messages): 

- **Galaxy AI Introduces Multitude of Free APIs**: GalaxyAI has released a **free** API service allowing access to premium AI models such as **GPT-4**, **GPT-3.5-turbo**, and Langchain Integration, all in the OpenAI format. Check out their offerings and integrate them into your projects at [Galaxy AI](https://galaxyapi.onrender.com).

- **OppyDev Launches AI-Powered Coding Tool**: OppyDev released an AI assisted coding platform that combines an IDE with a chat client, featuring ease of use, a focus on transparency, customization, data control, and uses LLMs like GPT-4 and Claude. See a demo and learn more at [OppyDev AI](https://oppydev.ai).

- **Rubiks.ai Calls for Beta Testers for Advanced Research Assistant**: A new advanced research assistant and search engine, Rubiks.ai, seeks beta testers to try out features including Claude 3 Opus, GPT-4 Turbo, and Mistral Large powered by Groq's servers for rapid responses. Interested individuals can explore and sign up at [Rubiks.ai](https://rubiks.ai) with a promo code `RUBIX` for 2 months of free premium access.

- **Unveiling The Power of Multi-Step Tools**: An article discusses the benefits of multi-step tools integrated with LangChain and Cohere, aimed at enhancing efficiency. Read more about this advancement in the full article at [AI Advances](https://medium.com/ai-advances/unlocking-efficiency-the-power-of-multi-step-tools-with-langchain-and-cohere-7d1ea571ebed).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: no description found</li><li><a href="https://oppydev.ai">Home - OppyDev</a>: Collaborative AI Agent that Elevates your Coding Experience</li><li><a href="https://rubiks.ai">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1229725236593692722)** (5 messages): 

- **Seeking Collaboration**: A participant expressed interest in joining a project and requested a direct message to discuss further details.
- **Tutorial on AI Agents with Long-Term Memory**: A member shared a [YouTube video](https://youtu.be/7LWTZqksmSg) that explains how to imbue AI agents with long-term memory and self-improvement capabilities, providing insight into advanced AI agent development.
- **Query on Langgraph Usage**: In response to the shared video about AI agent long-term memory, a member questioned why the concept of 'langgraph' wasn't considered for implementation.

**Link mentioned**: <a href="https://youtu.be/7LWTZqksmSg?si=_tnJhoUcQr4Gojek">Unlock AI Agent real power?! Long term memory &amp; Self improving</a>: How to build Long term memory &amp; Self improving ability into your AI Agent?Use AI Slide deck builder Gamma for free: https://gamma.app/?utm_source=youtube&amp;utm...

  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1230179344241000540)** (12 messages🔥): 

- **Pushing the Limits of GPU Memory**: Maxidl reported successful operation with **full-scale deep-speed** (FSDP), a sequence length of 32k, and batch size of 1 while utilizing a whopping 64 80GB GPUs, hugging close to capacity at 77GB utilized per GPU.
- **64 GPUs Not a Typo**: When questioned, maxidl confirmed the use of 64 GPUs, noting that reducing to 32 GPUs resulted in out-of-memory (OOM) errors, thus necessitating the larger GPU count.
- **Optimization Possibilities Explored**: Considering memory constraints, maxidl mentioned the potential of **8-bit optimization** to conserve memory during training.
- **Memory Usage Optimization Suggestion**: _jp1_ suggested using `fsdp_transformer_layer_cls_to_wrap: MixtralSparseMoeBlock` and enabling `offload_params = true` for improved memory usage, anticipating it should fit within 32 GPUs' VRAM.
- **Seeking Memory Requirement Calculators**: Maxidl inquired about tools to calculate memory usage of model activations by model size and sequence length, citing a [**HuggingFace discussion**](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/12) on model memory requirements for Mixtral models.

**Link mentioned**: <a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/12">mistral-community/Mixtral-8x22B-v0.1 · [AUTOMATED] Model Memory Requirements</a>: no description found

  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1229695493416681523)** (8 messages🔥): 

- **Gray Area in Text Scraping**: A member voiced the opinion that most scraped text data are, from an EU copyright perspective, at least in a gray area. They also mentioned that texts from **DFKI** could be useful but did not have the link at hand.
  
- **Finding Multimodal Data**: A member suggested sources for multimodal data with permissive licenses, like **Wikicommons** and other platforms listed on [Creative Commons Search](https://search.creativecommons.org/).

- **Llama Tokenizer Simplified**: An individual shared a [Google Colab notebook](https://colab.research.google.com/drive/1Ica34BAGK2tuIeQl01SRNTjujPq5C3d1?usp=sharing) illustrating how to create a Llama tokenizer without relying on HuggingFace, using sentencepiece instead.

- **Query on Tokenizer Spelling**: Following a discussion on custom tokenizers, a member pointed out a misspelling in a shared tokenizer, misspelling **Muad'Dib**.

- **Modernizing Tokenization Techniques**: A contributor highlighted that **Mistral** has released their tokenization library, potentially aiding in standardized finetuning processes without custom wrappers, and provided a [link to the example notebook on GitHub](https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://search.creativecommons.org/">CC Search Portal</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ica34BAGK2tuIeQl01SRNTjujPq5C3d1?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://github.com/mistralai/mistral-common/blob/main/examples/tokenizer.ipynb">mistral-common/examples/tokenizer.ipynb at main · mistralai/mistral-common</a>: Contribute to mistralai/mistral-common development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1229701118842503208)** (1 messages): 

- **Decoding Strategies for Language Models Analyzed**: A member referenced a paper titled "[A Thorough Examination of Decoding Methods in the Era of LLMs](https://arxiv.org/abs/2402.06925)," expressing concerns that it didn't cover open-ended tasks relevant to their LLM usage experience. They also mentioned that modern sampling methods by *u/kindacognizant*, such as **MinP/DynaTemp/Quadratic Sampling**, aren't covered in such papers.
- **Surprising Impact of min_p Sampling on Creative Writing**: The same member shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1c36ieb/comparing_sampling_techniques_for_creative/) detailing a comparison of **min_p sampling parameters** and their significant effect on creative writing performance. The comparison showed an increase of +8 points in **alpaca-eval style elo** and +10 points in the **eq-bench creative writing test**.

**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1c36ieb/comparing_sampling_techniques_for_creative/">Reddit - Dive into anything</a>: no description found

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1229891842754089082)** (9 messages🔥): 

- **Tinygrad and INT8 support query**: A member asked if tinygrad supports int8 computations, to which another replied affirmatively. The location where this is defined wasn't provided.
- **Hardware's Role in Defining Tinygrad's Computations**: A user mentioned that whether tinygrad supports certain data types, like int8, is typically defined by the **hardware capabilities** rather than tinygrad itself.
- **Enhanced Graph Visualizations for Tinygrad**: An inquiry was made about improved graph visualizations in tinygrad, and a reply directed to the [Tiny-tools Graph Visualization](https://tiny-tools-client.vercel.app/) for slicker graphs than `GRAPH=1`.
- **Interest in an Optimized Node.equals() for Tinygrad**: A member expressed interest in a fast, probabilistically complete **Node.equals()** function as a cool addition to tinygrad.
- **Pytorch-Lightning Hardware Agnosticism Discussed**: The hardware-agnostic nature of Pytorch-Lightning was discussed, with a link to its GitHub repository provided, and another member confirmed its use on a **7900xtx**. [Check out Pytorch-Lightning on GitHub](https://github.com/Lightning-AI/pytorch-lightning).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tiny-tools-client.vercel.app/">React App</a>: no description found</li><li><a href="https://github.com/Lightning-AI/pytorch-lightning">GitHub - Lightning-AI/pytorch-lightning: Pretrain, finetune and deploy AI models on multiple GPUs, TPUs with zero code changes.</a>: Pretrain, finetune and deploy AI models on multiple GPUs, TPUs with zero code changes. - Lightning-AI/pytorch-lightning
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1229757044148670504)** (9 messages🔥): 

- **Exploring Metal Compute Shaders**: A member is experimenting with **tinygrad's generation of Metal compute shaders** and is interested in learning how to run a basic Metal compute shader program without using Xcode. Another suggested consulting ChatGPT for a Python script to dispatch metal shader code for a vector addition, mentioning their positive learning experience.

- **ONNX to WebGL/WebGPU Possibilities**: An inquiry was made about converting models from ONNX to WebGL/WebGPU with tinygrad, specifically for running **meshnet models** on the web. A comparison was made to a [Stable Diffusion WebGPU example](https://github.com/softwiredtech/stable-diffusion-webgpu), but the member is seeking advice on achieving the conversion directly from ONNX.

- **Layer Device Allocation Query in Tinygrad**: A participant was concerned about the apparent lack of functionality to move layers (like Linear, Conv2d) across devices in tinygrad. **George Hotz** clarified that model parameters can be moved with the `to_` method called after **get parameters** on the model.

- **Zero-Cost Tensor Manipulation in Tinygrad**: A user asked for guidance on implementing broadcast, reshape, and permute operations in tinygrad without incurring data copying costs. They were directed to look at *tinygrad/shape/shapetracker.py* or *view.py* for relevant code examples.
  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1229691075690631250)** (4 messages): 

- **Introducing Idefics2**: A new **multimodal ChatGPT** called Idefics2 by [Hugging Face has been introduced](https://www.youtube.com/watch?v=vL1SayPCHBg), which incorporates Python programming into its abilities.
- **Reka Core Takes On Giants**: The [Reka Core language model](https://www.youtube.com/watch?v=U7RbwPKyxs8) is presented as competitive with those from OpenAI, Anthropic, and Google, touting impressive performance metrics.
- **JetMoE: Budget-Friendly AI Performance**: With less than $0.1 million spend, [JetMoE-8B claims superior performance](https://www.youtube.com/watch?v=Z9Hwp_XeS1A) compared to Meta AI's LLaMA2-7B, a model backed by extensive funding.
- **Snowflake's New Text-Embedding Model**: Snowflake has launched and open-sourced their Snowflake Arctic embed family of models, highlighted as the world's best [practical text-embedding model](https://www.youtube.com/watch?v=p9T7ZgtM5Mo).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=U7RbwPKyxs8">Reka Core: A Frontier Class Multimodal Language Model</a>: Reka Core is competitive with models from OpenAI, Anthropic, and Google across key industry-accepted evaluation metrics. Given its footprint and performance,...</li><li><a href="https://www.youtube.com/watch?v=vL1SayPCHBg">Introducing Idefics2 8B: Open Multimodal ChatGPT</a>: We will take a look idefics2 the open multimodal llm by huggingfacehttps://huggingface.co/blog/idefics2#python #pythonprogramming #llm #ml #ai #aritificialin...</li><li><a href="https://www.youtube.com/watch?v=p9T7ZgtM5Mo">Snowflake Launches the World’s Best Practical Text-Embedding Model</a>: Today Snowflake is launching and open-sourcing with an Apache 2.0 license the Snowflake Arctic embed family of models. Based on the Massive Text Embedding Be...</li><li><a href="https://www.youtube.com/watch?v=Z9Hwp_XeS1A">JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars</a>: JetMoE-8B is trained with less than $ 0.1 million1 cost but outperforms LLaMA2-7B from Meta AI, who has multi-billion-dollar training resources. LLM training...
</li>
</ul>

</div>
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1230226260366721116)** (3 messages): 

- **Anticipation for Mixtral 8x22B Instruct**: Excitement for trying out the **Mixtral 8x22B Instruct** through llm was expressed, with a link to its [Model Card on HuggingFace](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) provided for reference. 
- **Issue Reported with llm-gpt4all**: A user mentioned encountering an error when installing **llm-gpt4all**; the issue is detailed on GitHub with a link to the [error report](https://github.com/simonw/llm-gpt4all/issues/28).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1">mistralai/Mixtral-8x22B-Instruct-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://github.com/simonw/llm-gpt4all/issues/28">adding the llm-gpt4all models breaks the python app. · Issue #28 · simonw/llm-gpt4all</a>: I installed llm no problem, assigning my openai key, and am able to speak to gpt4 without problem, see the output of my llm models command: OpenAI Chat: gpt-3.5-turbo (aliases: 3.5, chatgpt) OpenAI...
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1229810947829403648)** (2 messages): 

- **Lawyers Stepped In**: A member made a brief remark suggesting that lawyers were likely involved in a certain situation, although the context of the legal implication was not provided.
- **Image Illustrating Deletion of wizardlm-2**: An image was shared depicting that **wizardlm-2** was deleted due to a lack of testing for **v0**; however, the specifics of what **wizardlm-2** is or what the testing involved were not given in the message. [View Image](https://cdn.discordapp.com/attachments/1019530324255965186/1229693872997666816/wizardlm-2-was-deleted-because-they-forgot-to-test-it-for-v0-lyaop5lw0suc1.png?ex=66309ca9&is=661e27a9&hm=f105e6497796be9c414ade2024a27f9561caf0cad6cb06ba09f80e30b5e39ae4&)
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1230131674290065490)** (2 messages): 

- **Llamafile Script Improvement**: The llamafile archive version upgrade repacking script has been improved and is available at [this Gist](https://gist.github.com/mofosyne/46c63934305d5a5321c7e9fd83f4ef3e). There is a debate on whether to integrate it into the main llamafile GitHub repo due to maintenance concerns, with the notion that maintainers should create new llamafiles from the ground up.

- **Security Vulnerability Reporting Process Inquiry**: A query was raised about the procedure for reporting security vulnerabilities and the subsequent request for a CVE (Common Vulnerabilities and Exposures) identification. No additional context or instructions were provided in the message.
  

---



---




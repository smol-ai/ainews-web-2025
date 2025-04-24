---
id: 521448c3-d0cc-48c6-bb04-66adbc1f25b0
title: That GPT-4o Demo
date: '2024-06-29T00:48:47.349723Z'
original_slug: ainews-that-openai-demo
description: >-
  Romain Huet demonstrated GPT-4o’s advanced multimodal capabilities including
  rapid OCR, voice generation with tone control, and real-time screen sharing
  with ChatGPT Desktop, showcasing significant improvements in interactive AI.
  Additionally, Google DeepMind released Gemma 2, a highly efficient large
  language model available in 9B and 27B parameter sizes, optimized for TPUv5e
  and offering state-of-the-art performance with strong safety measures.
tags:
  - openai
  - gpt-4o
  - multimodal
  - google
  - deepmind
  - gemma-2
  - tpuv5e
---


<!-- buttondown-editor-mode: plaintext -->**Omnimodel is all you need**

> AI News for 6/27/2024-6/28/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**417** channels, and **3655** messages) for you. 
Estimated reading time saved (at 200wpm): **354 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

[Romain Huet's](https://twitter.com/romainhuet) demo of GPT-4o using an unreleased version of ChatGPT Desktop [made the rounds yesterday](https://x.com/tsarnick/status/1806526891354132604) and was essentially the second-ever high profile demo of GPT-4o after the release ([our coverage here](https://buttondown.email/ainews/archive/ainews-gpt-4o-the-new-sota-everything-frontier/)), and in the absence of bigger news is our pick of headliner today:

 ![image.png](https://assets.buttondown.email/images/f3179985-63dd-4431-8803-f2f0c1bc62ad.png?w=960&fit=max) 

The demo starts at [the 7:15:50 mark on stream](https://www.youtube.com/live/vaIiNZoXymg?si=S73xOMIAlTOvAzEc&t=26153), and you should watch the whole thing.

Capabilities demonstrated:

- low latency voicegen
- instructions to moderate tone to a whisper (and even quieter whisper)
- interruptions
- Camera mode on ChatGPT Desktop - constantly streaming video to GPT4o
-  ![image.png](https://assets.buttondown.email/images/c5cfc90d-8127-4e20-ae47-10486b61be2b.png?w=960&fit=max) 
- When paired with voice understanding it eliminates the need for a Send or Upload button
- **Rapid OCR**: Romain asks for a random page number, and presents the page of a book - and it reads the page basically instantly! Unfortunately the OCR failed a bit - it misread "Coca Cola" but conditions for the live demo weren't ideal. 
 ![image.png](https://assets.buttondown.email/images/f8837ed4-8e51-4be2-bcc8-d4df999156a0.png?w=960&fit=max) 
- **Screen Sharing with ChatGPT**: talking with ChatGPT to describe his programming problem and having it understand from visual context
 ![image.png](https://assets.buttondown.email/images/2bf81e11-7a9c-40d0-aab7-630fc3749825.png?w=960&fit=max) 
- **Reading Clipboard**: copies the code, asks for a "one line overview" of the code (This functionality exists in ChatGPT Desktop today)
- **Conversing with ChatGPT about Code**: back and forth talking about Tailwind classnames in code, relying on vision (not clipboard) 
![image.png](https://assets.buttondown.email/images/80b51c10-dd96-4e33-ab8a-5c3acec13c23.png?w=960&fit=max) 

The rest of the talk discusses 4 "investment areas" of OpenAI:

- **Textual intelligence** (again using "GPT Next" instead of "GPT5"...)
 ![image.png](https://assets.buttondown.email/images/09da9fa4-0123-4f40-b3f8-b927b210508f.png?w=960&fit=max) 
- **Efficiency/Cost**  ![image.png](https://assets.buttondown.email/images/21d2c62d-f81d-4d88-9378-787e644e4667.png?w=960&fit=max) 
- **Model Customization** ![image.png](https://assets.buttondown.email/images/acbc1d92-631b-4f7e-ae3e-057ee9451558.png?w=960&fit=max) 
- **Multimodal Agents**  ![image.png](https://assets.buttondown.email/images/0637489f-e54e-4d2f-aabf-83eecef1ba65.png?w=960&fit=max) , including a Sora and Voice Engine demo that you should really check out if you haven't seen it before.


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

**Gemma 2 Release by Google DeepMind**

- **Model Sizes and Training**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1806373224889954449) announced Gemma 2 in 9B and 27B parameter sizes, trained on 13T tokens (27B) and 8T tokens (9B). Uses **SFT, Distillation, RLHF & Model Merging**. Trained on Google TPUv5e.
- **Performance**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1806373227305775438) 9B delivers **class-leading performance** against other open models in its size category. 27B outperforms some models more than twice its size and is **optimized to run efficiently on a single TPU host**.
- **Availability**: [@fchollet](https://twitter.com/fchollet/status/1806346069653287085) Gemma 2 is available on Kaggle and Hugging Face, **written in Keras 3 and compatible with TensorFlow, JAX, and PyTorch**.
- **Safety**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1806373232250917334) followed **robust internal safety processes** including filtering pre-training data, rigorous testing and evaluation to identify and mitigate potential biases and risks.

**Meta LLM Compiler Release**

- **Capabilities**: [@AIatMeta](https://twitter.com/AIatMeta/status/1806361623831171318) announced Meta LLM Compiler, built on Meta Code Llama with **additional code optimization and compiler capabilities**. Can emulate the compiler, predict optimal passes for code size, and disassemble code.
- **Availability**: [@AIatMeta](https://twitter.com/AIatMeta/status/1806361623831171318) LLM Compiler 7B & 13B models released under a **permissive license for both research and commercial use** on Hugging Face.
- **Potential**: [@MParakhin](https://twitter.com/MParakhin/status/1806382680616861961) LLMs replacing compilers could lead to **near-perfectly optimized code**, reversing decades of efficiency sliding. [@clattner_llvm](https://twitter.com/clattner_llvm/status/1806457173498814708) Mojo 🔥 is a culmination of the last 15 years of compiler research, MLIR, and many other lessons learned.

**Perplexity Enterprise Pro Updates**

- **Reduced Pricing for Schools and Non-Profits**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1806408640431288664) announced **reduced pricing** for Perplexity Enterprise Pro for any school, nonprofit, government agency, or not-for-profit.
- **Importance**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1806408644826972571) These organizations play a **critical role in addressing societal issues and equipping children with education**. Perplexity wants to ensure their technology is accessible to them.

**LangChain Introduces LangGraph Cloud**

- **Capabilities**: [@LangChainAI](https://twitter.com/LangChainAI/status/1806371717084025165) announced LangGraph Cloud, infrastructure to run **fault-tolerant LangGraph agents at scale**. Handles large workloads, enables debugging and quick iteration, and provides integrated tracing & monitoring.
- **Features**: [@hwchase17](https://twitter.com/hwchase17/status/1806376010176483355) LangGraph Studio is an **IDE for testing, debugging, and sharing LangGraph applications**. Builds on LangGraph v0.1 supporting diverse control flows.

**Other Notable Updates and Discussions**

- **Gemini 1.5 Pro Updates**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1806345612184482149) opened up access to **2 million token context window** on Gemini 1.5 Pro for all developers. **Context caching** now available in Gemini API to reduce costs.
- **Lucid Dream Experience**: [@karpathy](https://twitter.com/karpathy/status/1806400213793534010) shared a lucid dream experience, noting the **incredibly detailed and high resolution graphics**, comparing it to a Sora-like video+audio generative model.
- **Anthropic Updates**: [@alexalbert__](https://twitter.com/alexalbert__/status/1806410983931629807) Anthropic devs can now view **API usage broken down by dollar amount, token count, and API keys** in the new Usage and Cost tabs in Anthropic Console.
- **Distillation Discussion**: [@giffmana](https://twitter.com/giffmana/status/1806402283649036605) and [@jeremyphoward](https://twitter.com/jeremyphoward/status/1806446889006666110) discussed the importance of distillation and the **"curse of the capacity gap"** in training smaller high-performing models.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Models and Architectures**

- **Gemma 2 models surpass Llama and Claude**: In /r/LocalLLaMA, Google's Gemma 2 27B model [**beats Llama 3 70B in LMSYS benchmarks**](https://www.reddit.com/r/LocalLLaMA/comments/1dpsal9/gemma_2_lmsys_results_from_technical_report/) according to a technical report. The 9B variant also [surpasses Claude 3 Haiku](https://www.reddit.com/r/LocalLLaMA/comments/1dpwlqj/lets_talk_more_about_gemma2_26b_the_smallest/).
- **Knowledge distillation for smaller models**: Gemma 2 9B was [**trained using the 27B model as a teacher**](https://www.reddit.com/r/LocalLLaMA/comments/1dpwlqj/lets_talk_more_about_gemma2_26b_the_smallest/), an approach that could be the future for small/medium models, potentially using even larger models like Llama 400B as teachers.
- **MatMul-free language modeling**: A new paper introduces an approach that [**eliminates matrix multiplication from language models while maintaining strong performance**](https://www.reddit.com/r/LocalLLaMA/comments/1dqdhsh/gemma2_originally_kaggle_posted_fp32_unquantized/). It reduces memory usage by 61% in training and 10x during inference, with custom FPGA hardware processing models at 13W.
- **Sohu chip delivers massive performance**: The specialized Sohu AI chip from Etched [**allegedly replaces 160 H100 GPUs**](https://www.reddit.com/r/OpenAI/comments/1dpyba1/sohu_replaces_160_nvidia_gpus_delivers_500000/), delivering 500,000 tokens/sec. It claims to be 10x faster and cheaper than Nvidia's next-gen Blackwell GPUs.

**AI Applications and Use Cases**

- **AI-generated graphic novel**: In /r/StableDiffusion, an author [**created a graphic novel 100% with Stable Diffusion**](https://www.reddit.com/r/StableDiffusion/comments/1dplv1d/i_finally_published_a_graphic_novel_made_100_with/), using SD1.5 iComix for characters, ControlNet for consistency, and Photoshop for layout. It's the first such novel published in Albanian.
- **Website redesign with AI**: A [browser extension uses the OpenAI API to redesign websites](https://www.reddit.com/r/StableDiffusion/comments/1dplv1d/browser_extension_uses_openai_api_to_redesign_the/) based on a provided prompt, leveraging CSS variables. Experiments were done on shadcn.com and daisyui.com.
- **Personalized AI assistant**: An article on /r/LocalLLaMA details [**extending a personal Llama 3 8B model with WhatsApp and Obsidian data**](https://www.reddit.com/r/LocalLLaMA/comments/1dpu2oo/personal_local_llm_extended_with_whatsapp/) to create a personalized AI assistant.

**Memes and Humor**

- **Disconnect in AI discussions**: A [meme video pokes fun at the disconnect](https://www.reddit.com/r/StableDiffusion/comments/1dplv1d/how_it_feels_to_talk_about_ai_with_normal_people/) between AI enthusiasts and the general public when discussing the technology.
- **AI-generated movies**: A [humorous video imagines formulaic movies churned out by AI](https://www.reddit.com/r/StableDiffusion/comments/1dplv1d/your_average_movie_in_2025/) in the near future.
- **Progress in AI-generated animations**: A [meme video of Will Smith eating spaghetti](https://www.reddit.com/r/StableDiffusion/comments/1dplv1d/new_checkpoint_achieved/) demonstrates AI's improving ability to generate realistic human animations, with only minor face and arm glitches remaining.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Model Performance Optimization and Benchmarking**

- **[Quantization](https://github.com/Vahe1994/AQLM)** techniques like **AQLM** and **QuaRot** aim to run large language models (**LLMs**) on individual GPUs while maintaining performance. Example: [AQLM project](https://github.com/Vahe1994/AQLM) with **Llama-3-70b** running on RTX3090.

- Efforts to **boost transformer efficiency** through methods like **Dynamic Memory Compression (DMC)**, potentially improving throughput by up to 370% on **H100 GPUs**. Example: [DMC paper](https://arxiv.org/abs/2403.09636) by @p_nawrot.

- Discussions on **optimizing CUDA operations** like fusing element-wise operations, using **Thrust library's `transform`** for near-bandwidth-saturating performance. Example: [Thrust documentation](https://nvidia.github.io/cccl/thrust/api/groups/group__modifying.html#function-for-each).

- Comparisons of **model performance** across benchmarks like **AlignBench** and **MT-Bench**, with **DeepSeek-V2** surpassing GPT-4 in some areas. Example: [DeepSeek-V2 announcement](https://x.com/deepseek_ai/status/1787478986731429933).

**2. Fine-tuning Challenges and Prompt Engineering Strategies**

- Difficulties in **retaining fine-tuned data** when converting **Llama3** models to GGUF format, with a [confirmed bug](https://github.com/ggerganov/llama.cpp/issues/7062) discussed.

- Importance of **prompt design** and usage of correct templates, including end-of-text tokens, for influencing model performance during fine-tuning and evaluation. Example: [Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47).

- Strategies for **prompt engineering** like splitting complex tasks into multiple prompts, investigating **logit bias** for more control. Example: [OpenAI logit bias guide](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api).

- Teaching LLMs to use `<RET>` token for **information retrieval** when uncertain, improving performance on infrequent queries. Example: [ArXiv paper](https://arxiv.org/abs/2404.19705).

**3. Open-Source AI Developments and Collaborations**

- Launch of **StoryDiffusion**, an open-source alternative to Sora with MIT license, though weights not released yet. Example: [GitHub repo](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file).

- Release of **OpenDevin**, an open-source autonomous AI engineer based on Devin by Cognition, with [webinar](https://lu.ma/fp0xr460) and growing interest on GitHub.

- Calls for collaboration on open-source **machine learning paper** predicting IPO success, hosted at [RicercaMente](https://edopedrocchi.github.io/RicercaMente/Projects/IPO/indexIPO.html).

- Community efforts around **LlamaIndex** integration, with issues faced in Supabase Vectorstore and package imports after updates. Example: [llama-hub documentation](https://github.com/run-llama/llama-hub/tree/main#how-to-add-a-loadertoolllama-pack).

**4. LLM Innovations and Training Insights**

- **Gemma 2 Impresses with Efficient Training**: Google's **[Gemma 2](https://huggingface.co/blog/gemma2)** models, significantly smaller and trained on fewer tokens (9B model on 8T tokens), have outperformed competitors like Llama3 70B in benchmarks, thanks to innovations such as knowledge distillation and soft attention capping.
- **Gemma-2's VRAM Efficiency Boosts QLoRA Finetuning**: The new pre-quantized **[Gemma-2 4-bit models](https://huggingface.co/unsloth/gemma-2-27b-bnb-4bit)** promise 4x faster downloads and reduced VRAM fragmentation, capitalizing on efficiency improvements in QLoRA finetuning.
- **MCTSr Elevates Olympiad Problem-Solving**: The **[MCT Self-Refine (MCTSr)](https://arxiv.org/abs/2406.07394)** algorithm integrates LLMs with Monte Carlo Tree Search, showing substantial success in tackling complex mathematical problems by systematically refining solutions.
- **Adam-mini Optimizer's Memory Efficiency**: **[Adam-mini](https://arxiv.org/abs/2406.16793)** optimizer achieves comparable or better performance than AdamW with up to 50% less memory usage by leveraging a simplified parameter partitioning approach.

**5. Secure AI and Ethical Considerations**

- **Rabbit R1's Security Lapse Exposed on YouTube**: A YouTube video titled **["Rabbit R1 makes catastrophic rookie programming mistake"](https://youtu.be/lkbV8oP-F44)** revealed hardcoded API keys in the Rabbit R1 codebase, compromising user data security.
- **AI Usage Limit Warnings and Policy Compliance**: Members highlighted the risks of pushing AI boundaries too far, cautioning that violating **[OpenAI's usage policies](https://openai.com/policies/usage-policies)** can result in account suspension or termination.
- **Open-Source AI Debate**: Intense discussions weighed the pros and cons of open-sourcing AI models, balancing potential misuse against democratization of access and the economic implications of restricted AI, considering both the benefits and the dangers.

**6. Practical AI Integration and Community Feedback**

- **AI Video Generation with High VRAM Demands**: Successful implementation of **ExVideo** generating impressive video results, albeit requiring substantial VRAM (43GB), demonstrates the continuous balance between AI capability and hardware limitations.
- **Issues with Model Implementation Across Platforms**: Integration issues with models like **Gemma 2** on platforms such as [LM Studio](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) require manual fixes and the latest updates to ensure optimal performance.
- **Challenges with RAG and API Limitations**: Perplexity's **RAG mechanism** received criticism for inconsistent outputs, and limitations with models like **Claude 3 Opus**, showcasing struggles in context handling and API performance.

**7. Datasets and Benchmarking Advancements**

- **REVEAL Dataset Benchmarks Verifiers**: The **[REVEAL dataset](https://reveal-dataset.github.io)** benchmarks automatic verifiers of Chain-of-Thought reasoning, highlighting the difficulties in verifying logical correctness within open-domain QA settings.
- **XTREME and SPPIQA Datasets for Robust Testing**: Discussion on the **[XTREME](https://huggingface.co/datasets/google/xtreme)** and **[SPPIQA](https://huggingface.co/datasets/google/spiqa)** datasets focused on assessing multilingual models' robustness and multimodal question answering capabilities, respectively.
- **Importance of Grounded Response Generators**: The need for reliable models that provide grounded responses was highlighted with datasets like [Glaive-RAG-v1](https://huggingface.co/datasets/glaiveai/RAG-v1), and considerations on scoring metrics for quality improvement.

**8. Collaboration and Development Platforms**

- **Building Agent Services with LlamaIndex**: Engineers can create vector indexes and transform them into query engines using resources shared in the **[LlamaIndex notebook](https://t.co/WYTCaqs6Yb)**, enhancing AI service deployment.
- **Featherless.ai Offers Model Access Without GPU Setup**: [Featherless.ai](https://featherless.ai/) launched a platform providing flat-rate access to over 450 models from Hugging Face, catering to community input on model prioritization and use cases.
- **LangGraph Cloud Enhances AI Workflows**: The introduction of **[LangGraph Cloud](http://bit.ly/langgraph-cloud-blog-1)** by LangChainAI promises robust, scalable workflows for AI agents, integrating monitoring and tracing for improved reliability.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma-2 Goes Lean and Keen**: The new **pre-quantized 4-bit versions of Gemma-2-27B and 9B** are now available, boasting faster downloads and lesser VRAM fragmentation which are beneficial for [QLoRA finetuning](https://huggingface.co/unsloth/gemma-2-27b-bnb-4bit).

- **The Great OS Debate for AI Development**: Within the community, there's an active debate on the merits of using **Windows vs. Linux** for AI development, featuring concerns about peripheral compatibility on Linux and a general preference toward Linux despite perceived arbitrary constraints of Windows.

- **Hugging Face's Evaluation System Under the Microscope**: The community compared Hugging Face's evaluation system to a "popularity contest" and broached the notion of having premium paid evaluations, suggesting that an "*evaluation should be allowed at any time if the user is willing to pay for it.*"

- **Big Data, Big Headaches**: Discussions around handling a **2.7TB Reddit dataset** pointed out the immense resources needed for cleaning the data, which could inflate to "about 15 TB uncompressed... for meh data at best."

- **AI Video Generation at the Edge of VRAM**: The use of **ExVideo for generating video content** has been reported to deliver impressive results, yet it commands a formidable VRAM requirement of 43GB, emphasizing the constant balance between AI capabilities and resource availability.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Gemma 2 Outshines Competition**: Google's new **Gemma 2** models have been integrated into the Transformers library, boasting advantages such as size efficiency—**2.5x smaller than Llama3**—and robust training on **13T tokens** for the **27B model**. Innovations like **knowledge distillation** and **interleaving local and global attention layers** aim for enhanced inference stability and memory reduction, with informative [Gemma 2 details](https://huggingface.co/blog/gemma2) covered in a HuggingFace blog post.

- **Deciphering New File System for Elixir**: Elixir's FSS introduces file system abstraction with HTTP support and discusses non-extensibility concerns alongside HuggingFace's first open-source image-based retrieval system, making waves with a [Pokémon dataset example](https://huggingface.co/spaces/not-lain/RAG-on-images) and interest in further projects like visual learning models with [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4) being spotlighted.

- **AI Takes Flight with Multilingual App**: U-C4N's multilingual real-time flight tracking application reveals the intersect of aviation and language, while a new 900M variant of PixArt aspires collaboration in the Spaces [arena](https://huggingface.co/spaces/ptx0/PixArt-900M). Also, a fusion of AI and musical storytelling on platforms like [Bandcamp](https://vonpsyche.bandcamp.com/album/the-prompt) breaks genre boundaries.

- **Ready, Set, Gradio!**: Gradio users face an action call to update to versions above 3.13 to avoid share link deactivation. Adherence will ensure continued access to Gradio's resources and is as simple as running `pip install --upgrade gradio`.

- **Machine Learning at Lightning Speed**: A hyper-speed YouTube tutorial has engineered a 1-minute crash course on **ten critical machine learning algorithms**, fitting for those short on time but hungry for knowledge. Check this rapid lesson [here](https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Gemma 2 Integration with Hiccups**: The latest **LM Studio 0.2.26 release** adds support for **Gemma 2** models, though some users report integration bugs and difficulties. To work around these issues, manual downloads and reinstallation of configs are suggested, with a note that some architectures, like ROCm, are still pending support.

**Gemma-2's Confusing Capabilities**: Discrepancies in the information about **Gemma-2's** context limit led to confusion, with conflicting reports of a 4k versus an 8k limit. Additionally, the support for storytelling model [ZeusLabs/L3-Aethora-15B-V2](https://huggingface.co/ZeusLabs/L3-Aethora-15B-V2) was recommended, and for models like **Deepseek coder V2 Lite**, users were advised to track [GitHub pull requests](https://github.com/ggerganov/llama.cpp/pull/8156) for updates on support status.

**Snapdragon Soars in LM Studio**: Users praised the performance of Snapdragon X Elite systems for their compatibility with **LM Studio**, noting significant CPU/memory task efficiency compared to an i7 12700K, despite falling short compared to a 4090 GPU in specific tasks.

**Threading the Needle for Multi-Agent Frameworks**: Discussions on model efficacy suggested that a **0.5B model** might comfortably proxy a user in a multi-agent framework; however, skepticism remains for such low-end models' capacity for coding tasks. For hardware enthusiasts, queries about the value of using dual video cards were answered positively.

**Rift Over ROCm Compatibility and Gemma 2 Debuts**: In the **AMD ROCm tech-preview** channel, queries about Gemma 2 model support for AMD GPUs were raised, pointing users to the newly released 0.2.26 ROCm "extension pack" for **Windows** described in [GitHub instructions](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#installation-on-windows-0226-). Furthermore, Gemma 2's launch was met with both excitement and critique, with some users labeling it as "hot garbage" and others anxious for the promised improvements in coming updates.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**AI Usage Warnings**: A discussion highlighted the risks of testing the limits of AI, leading to a clear warning: violating [OpenAI's usage policies](https://openai.com/policies/usage-policies) can result in account suspension or termination.

**Open-source AI Debate**: The engineering community debated the open-sourcing of AI models; the discussion contrasted the potential for misuse against the democratization of access, highlighting the economic implications of restricted access and the necessity of surveillance for public safety.

**RLHF Training Puzzles Users**: Conversations about Reinforcement Learning from Human Feedback (RLHF) revealed confusion regarding its occasional prompts and the opaque nature of how OpenAI handles public RLHF training.

**AI Integration Triumphs and Woes**: Experiences shared by members included issues with custom GPTs for specific tasks like medical question generation and successes in integrating AI models and APIs with other services for enhanced functionalities.

**Prompt Engineering Insights**: Members exchanged tips on prompt engineering, recommending simplicity and conciseness, with a foray into the use of "logit bias" for deeper prompt control and a brief touch on the quasi-deterministic nature of stochastic neural networks.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Custom Style Datasets Raise Red Flags**: Participants discussed the creation of datasets with a custom style, noting the potential risks of inadvertently generating NSFW content. The community underscored the complexity in monitoring image generation to avoid platform bans.
  
- **Switching to Forge amid Automatic1111 Frustrations**: Due to issues with **Automatic1111**, users are exploring alternatives like **Forge**, despite its own memory management challenges. For installation guidance, a [Stable Diffusion Webui Forge Easy Installation](https://www.youtube.com/watch?v=FKzvHFtc8N0&t=64s) video on YouTube has been shared.

- **Cascade Channel Revival Requested**: Many members voiced their desire to restore the **Cascade** channel due to its resourceful past discussions, sparking a debate on the guild's direction and possible focus shift towards **SD3**.

- **Deep Dive into Model Training Specifics**: Conversations about model training touched on the nuances of **LoRa** training and samplers such as **3m sde exponential**, as well as VRAM constraints. The effectiveness and limitations of tools like **ComfyUI**, **Forge**, and **Stable Swarm** were also examined.

- **Discord Community Calls for Transparency**: A portion of the community expressed dissatisfaction with the deletion of channels and resources, impelling a discussion about the importance of transparent communication and preservation of user-generated content.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Scarlet AI Marches into Project Management**: A preview of **Scarlet AI** for planning complex projects has been introduced; interested engineers can assess its features despite not being ready for prime time at [Scarlet AI Preview](https://app.scarletai.co/).
  
- **Character.AI Dials Into Voice**: New **Character Calls** by **Character.AI** allows for AI interactions over phone calls, suitable for interview rehearsals and RPG scenarios. The feature is showcased on their [mobile app demonstration](https://share.character.ai/Wv9R/6tdujbbr).

- **Meta Optimizes Compiler Code via LLM**: Meta has launched a **Large Language Model Compiler** to improve compiler optimization tasks, with a deep dive into the details in their [research publication](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/).

- **Infrastructure Innovations with LangGraph Cloud**: **LangChainAI** introduced **LangGraph Cloud**, promising resilient, scalable workflows for AI agents, coupled with monitoring and tracing; more insights available in their [announcement blog](http://bit.ly/langgraph-cloud-blog-1).

- **Leadership Shift at Adept Amid Amazon Team-Up**: News has surfaced regarding **Adept** refining their strategy along with several co-founders transitioning to Amazon's AGI team; learn more from the [GeekWire article](https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/).

- **OpenAI Demos Coming in Hot**: The guild was notified of an imminent OpenAI demo, advising members to access the special [OpenAI Demo channel](https://discord.com/channels/822583790773862470/1197350122112168006) without delay.

- **GPT-4o Poised to Reinvent Coding on Desktop**: The guild discussed the adoption of GPT-4o to aid in desktop coding, sharing configurations like `Open-Interpreter` which could easily be integrated with local models.

- **When Penguins Prefer Apples**: The struggles of Linux users with streaming sparked a half-humorous, half-serious comparison with Mac advantages and brought to light **Vesktop**, a performance-boosting Discord app for Linux, found on [GitHub](https://github.com/Vencord/Vesktop).
  
- **AI Community Leaks and Shares**: There's chatter about potentially sensitive GPT definitions surfacing on platforms like [GitHub](https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts); a nod to privacy concerns. Links to wear the scholarly hat were exchanged, illuminating CoALA frameworks and repositories for language agents which can be found at [arXiv](https://arxiv.org/abs/2309.02427) and on [GitHub](https://github.com/ysymyth/awesome-language-agents). 

- **Praise for Peer Presentations**: Members showered appreciation on a peer for a well-prepared talk, highlighting the importance of quality presentations in the AI field.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLMs' Instruction Pre-Training Edges Out**: Incorporating **200M instruction-response pairs** into pre-training large corpora boosted performance, allowing a modest **Llama3-8B** to hang with the big guns like **Llama3-70B**. Details on the efficient instruction synthesizer are in the [Instruction Pre-Training paper](https://arxiv.org/abs/2406.14491), and the model is available on [Hugging Face](https://huggingface.co/instruction-pretrain/finance-Llama3-8B).

- **MCTSr Fuses with LLMs for Olympian Math**: Integrating Large Language Models with **Monte Carlo Tree Search** (MCTSr) led to notable success in solving mathematical Olympiad problems. The innards of this technique are spilled in a [detailed study](https://arxiv.org/abs/2406.07394).

- **Datasets Galore: SPPIQA, XTREME, UNcommonsense**: A suite of datasets including [SPPIQA](https://reveal-dataset.github.io) for reasoning, [XTREME](https://huggingface.co/datasets/google/xtreme) for multilingual model assessment, and [UNcommonsense](https://huggingface.co/datasets/allenai/UNcommonsense) for exploring scales of bizarre, were discussed across Nos Research AI channels.

- **Hermes 2 Pro Launches with Function Boost**: The **Hermes 2 Pro 70B** model was revealed, trumpeting improvements in function calls and structured JSON outputs, boasting scores of 90% and 84% in assessments. A scholarly read isn't offered, but you can explore the model at [NousResearch's Hugging Face](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B).

- **Debating SB 1047's Grip on AI**: Members heatedly debated whether Cali's SB 1047 legislation will stunt AI's growth spurts. A [campaign](https://live-2024-stop-sb-1047.pantheonsite.io) rallying against the bill warns it could curb the risk-taking spirit essential for AI's blazing trail.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **"Cursed" Complexity vs. Practical Performance**: The **yoco** architecture's **kv cache** strategy sparked debate, with criticisms about its deviation from standard transformer practices and complexity. The discussion also covered the order of attention and feed-forward layers in models, as some proposed an efficiency gain from non-standard layer ordering, while others remained skeptical of the performance benefits.

- **Scaling Beyond Conventional Wisdom**: Discussions around **scaling laws** questioned the dominance of the **Challax scaling** model, with some participants proposing the term "scaling laws" to be provisional and better termed "scaling heuristics." References were made to papers such as ["Parameter Counts in Machine Learning"](https://www.alignmentforum.org/posts/GzoWcYibWYwJva8aL/parameter-counts-in-machine-learning) to support viewpoints on different scaling models' effectiveness.

- **Data Privacy Dilemma**: Conversations surface privacy concerns in the **privacy-preserving/federated learning** context, where aggregate data exposes a wider attack space. The potential for AI agents implementing security behaviors was discussed, considering contextual behavior identification and proactive responses to privacy compromises.

- **LLM Evaluation and Innovation**: A new **reasoning challenge dataset**, **MMLU-SR**, was introduced and considered for addition to **lm_eval**, probing large language models' (LLMs) comprehension abilities through modified questions. Links to the dataset [arXiv paper](https://arxiv.org/abs/2406.15468v1) and a GitHub PR for the **MedConceptsQA** benchmark [addition](https://github.com/EleutherAI/lm-evaluation-harness/pull/2010) were shared.

- **Instruction Tuning Potential in GPTNeoX**: Queries on **instruction tuning** in GPTNeoX, specifically selectively backpropagating losses, led to a discussion that referenced an ongoing [PR](https://github.com/EleutherAI/gpt-neox/pull/1240) and a preprocessing script "[preprocess_data_with_chat_template.py](https://github.com/EleutherAI/gpt-neox/blob/main/tools/datasets/preprocess_data_with_chat_template.py)", signifying active development in tailored training workflows.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Tribulations on Windows**: Users have reported **Triton installation issues** on Windows when using `torch.compile`, leading to "RuntimeError: Cannot find a working trilog installation." It's suggested that Triton may not be officially supported on Windows, posing a need for alternative installation methods.

- **Tensor Tinkering with torch.compile**: The author of [Lovely Tensors](https://github.com/xl0/lovely-tensors) faces breakage in `torch.compile()` due to `Tensor.__repr__()` being called on a FakeTensor. The community suggests leveraging [torch.compiler fine-grain APIs](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html) to mitigate such issues. Meanwhile, an update to NCCL resolves a broadcast deadlock issue in older versions, as outlined in [this pull request](https://github.com/huggingface/text-generation-inference/pull/2099).

- **Gearing up with CUDA Knowledge**: Stephen Jones presents [an in-depth overview of CUDA programming](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62401/), covering *wave quantization & single-wave kernels*, parallelism, and optimization techniques like tiling for improved L2 cache performance.

- **CUDA Curiosity and Cloud Query**: Members share platforms like [Vast.ai](https://vast.ai/) and [Runpod.io](https://www.runpod.io/) for CUDA exploration on cloud GPUs, and recommend starting with `torch.compile`, then moving to Triton or custom CUDA coding for Python to CUDA optimization.

- **PMPP Publication Puzzle**: A member highlights a physical copy of PMPP (4th edition) book missing several pages, inciting queries about similar experiences.

- **torch/aten Ops Listing and Bug-Hunting**: The torchao channel surfaces requests for a comprehensive list of required `torch/aten ops` for tensor subclasses such as `FSDP`, conversations about a recursion error with `__torch_dispatch__`, and a refactor [PR for Int4Tensor](https://github.com/pytorch/ao/pull/458). Additionally, there was a caution regarding GeForce GTX 1650's lack of native bfloat16 support.

- **HuggingFace Hub Hubbub**: The off-topic channel buzzes with chatter about the *pros and cons* of storing model architecture and preprocessing code directly on HuggingFace Hub. There's debate on the best model code and weight storage practices, with the [Llama model](https://github.com/meta-llama/llama) cited as a case study in effective release strategy.

- **Gemma 2 Grabs the Spotlight**: The Gemma 2 models from Google, sporting 27B and 9B parameters, outshine competitors in benchmarks, with [appreciation for openness](https://x.com/reach_vb/status/1806343018640781675) and anticipation for a smaller 2.6B variant. Discussions also focused on architectural choices like [approx GeGLU activations](https://x.com/danielhanchen/status/1806372357684220308) and the ReLU versus GELU debate, backed by [scholarly research](https://arxiv.org/pdf/2002.05202v1). Hardware challenges with FP8 support led to mentions of limitations in NVIDIA's libraries and [Microsoft's work on FP8-LM](https://arxiv.org/pdf/2310.18313). Yuchen's training insights suggest platform or dataset-specific issues when optimizing for H100 GPUs.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Cuts Costs for a Cause**: Perplexity introduces reduced pricing for its **Enterprise Pro** offering, targeting schools and nonprofits to aid in their societal and educational endeavors. Further information on eligibility and application can be found in their [announcement](https://pplx.ai/s3oZX49).

- **RAG Frustration and API Agitation**: Within **Perplexity's AI** discussion, there's frustration over the erratic performance of the **RAG**(**Relevance Aware Generation**) mechanism and demand for access to larger models such as **Gemma 2**. Additionally, users are experiencing limitations with **Claude 3 Opus**, citing variable and restrictive usage caps.

- **Security First, Fixes Pending**: Security protocols were addressed, directing members to the [Trust Center](https://trust.perplexity.ai/) for information on **data handling** and **PII management**. Meanwhile, members suggested using "#context" for improved continuity handling ongoing context retention issues in interactions.

- **Tinkering with Capabilities**: The community's attention turned to exploring **Android 14** enhancements, while raising issues with **Minecraft's mechanics potentially misleading kids**. An inquiry into filtering API results to receive recent information received guidance on using specific date formats.

- **Tech Deep-Dives and Innovations Spotlighted**: Shared content included insights on **Android 14**, criticisms of **Linux performance**, innovative uses of **Robot Skin**, and sustainable construction inspired by oysters. A notable share discussed criticisms of **Minecraft's repair mechanics** potentially leading to misconceptions.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Character.AI Pioneers Two-Way AI Voice Chats**: [Character.AI has launched Character Calls](https://blog.character.ai/introducing-character-calls/), enabling voice conversations with AI, though the experience is marred by a 5-second delay and less-than-fluid interaction. Meanwhile, industry chatter suggests Amazon's hiring of Adept's cofounders and technology licensing has left Adept diminished, amid unconfirmed claims of Adept having a toxic work environment.

- **AI Agents Trail Behind the Hype Curve**: Discussions draw parallels between AI agents' slow progress and the self-driving car industry, claiming that hype outpaces actual performance. The quality and sourcing of training data for AI agents, including an emerging focus on synthetic data, were highlighted as pivotal challenges.

- **SnailBot News Episode Stirs Up Discussion**: Excitement is brewing over SnailBot News' latest episode featuring Lina Khan; Natolambert teases interviews with notable figures like Ross Taylor and John Schulman. Ethical considerations around "Please don't train on our model outputs" data usage conditions were also brought into focus.

- **Scaling Engulfs AI Discourse**: Skepticism encircles the belief that scaling alone leads to AGI, as posited in [AI Scaling Myths](https://www.aisnakeoil.com/p/ai-scaling-myths), coupled with discussions on the alleged limitations in high-quality data for LLM developers. Nathan Lambert urges critical examination of these views, referencing [Substack discussions](https://open.substack.com/pub/aisnakeoil/p/ai-scaling-myths?r=68gy5&utm_campaign=comment-list-share-cta&utm_medium=web&comments=true&commentId=60317135) and recent advances in synthetic data.

- **Varied Reflections on AI and Global Affairs**: From Anthropic CEO's affection for Final Fantasy underscoring AI leaders' human sides to debates over AI crises being potentially more complex than pandemics, guild members engage in diverse conversations. Some talk even considers how an intelligence explosion could reshape political structures, reflecting on the far-reaching implications of AI development.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**LlamaIndex Powers Agent Services**: Engineers explored building **agentic RAG services** with **LlamaIndex**, discussing the process of creating vector indexes and transforming them into query engines. Detailed steps and examples can be found in a recently shared [notebook](https://t.co/WYTCaqs6Yb).

**Jina's Reranking Revolution**: The LlamaIndex community is abuzz about **Jina's newest reranker**, hailed as their most effective to date. Details behind the excitement are available [here](https://t.co/YsYoVOIirb).

**Node Weight Puzzle in Vector Retrievers**: AI practitioners are troubleshooting **LlamaIndex's embedding challenges**, deliberating on factors such as the parts of nodes to embed and the mismatch of models contributing to suboptimal outcomes from vector retrievers. A consensus implies creating simple test cases for effective debugging.

**Entity Linking Through Edges**: Enhancing **entity relationship detection** is generating debate, focused on adding edges informed by embedding logic. Anticipation surrounds a potential collaborative know-how piece with **Neo4j**, expected to shed light on advanced entity resolution techniques.

**Issues Surface with Claude and OpenAI Keys**: Discussions emerge about needing fixes for **Claude's** empty responses linked to Bedrock's token limitation and an *IndexError* in specific cases, as well as a curious environment behavior where code-set **OpenAI keys** seem overridden. Engineers also probe optimizations for batch and parallel index loading, aiming to accelerate large file handling.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Gemma's Multilingual Punch**: While **Gemma 2** officially supports only English, users report excellent multilingual capabilities, with specific inquiries about its performance in Korean.

**Model Migration Madness**: **Gemma 2.9B** models, with free and standard variants, are storming the scene as per the [announcement](https://openrouter.ai/models/google/gemma-2-9b-it), accompanied by price cuts across popular models, including a 10% drop for [Dolphin Mixtral](https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b) and 20% for [OpenChat](https://openrouter.ai/models/openchat/openchat-8b).

**OpenRouter, Open Issues**: OpenRouter's tight-lipped moderation contrasts with platforms like AWS; meanwhile, users confront the lack of Opus availability without enterprise support and battle **Status 400** errors from disobedient APIs of Gemini models.

**Passphrase Puzzles and API Allocutions Solved**: Engineers share wisdom on seamless GitHub authentication using `ssh-add -A`, and discuss watching Simon Willison's overview on LLM APIs for enlightenment, with resources found on [YouTube](https://www.youtube.com/watch?v=5zE2sMka620&t=2026s) and his [blog](https://simonwillison.net/2024/Jun/27/ai-worlds-fair/).

**AI Affinity Adjustments**: Embrace **daun.ai**’s advice to set the default model to 'auto' for steady results or live life on the edge with 'flavor of the week' fallbacks, ensuring continued productivity across tasks.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Gege AI Serenades with New Voices**: The music creation tool, **Gege AI**, can mimic any singer's voice from a small audio sample, inciting humorous comments about the potential for disrupting the music industry and speculations about **RIAA**'s reaction.
  
- **User Frustration with Gege AI and GPT-4 Models**: Users reported difficulties in registering for **Gege AI** with quips about social credit, while others expressed disappointment with the performance of **GPT-4 and 4O models**, suggesting they can be too literal and less suited for programming tasks than earlier versions like **GPT-3.5**.

- **Adam-mini Optimizer Cuts Memory Waste**: The **Adam-mini** optimizer offers performance comparable to, or better than, **AdamW**, while requiring 45-50% less memory by partitioning parameters and assigning a single learning rate per block, according to a recent paper highlighted in discussions.
  
- **Skepticism Meets Ambition with Gemma 27B**: While the new **Gemma 27B** model has reportedly shown some promising performance enhancements, members remained cautious due to a high confidence interval, questioning its overall advantage over previous iterations.

- **Shifting to Claude for a Smoother Ride**: Given issues with **OpenAI's models**, some members have opted for **Claude** for its superior artifacts feature and better integration with the **Hugging Face libraries**, reporting a smoother experience compared to **GPT-4** models.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Bedrock Befuddles Engineers**: Engineers shared challenges in integrating **csv_agent** and **pandas_dataframe_agent** with **Bedrock**, as well as errors encountered while working with **Sonnet 3.5 model** and Bedrock using `ChatPromptTemplate.fromMessages`, indicating possible compatibility issues.

- **Launch of LangGraph with Human-in-the-Loop Woes**: The introduction of **LangGraph**'s human-in-the-loop capabilities, notably "Interrupt" and "Authorize", was marred by deserialization errors during the resumption of execution post-human approvals, as discussed in the [LangChain Blog](https://blog.langchain.dev/human-in-the-loop-with-opengpts-and-langgraph).

- **Refinement of JSONL Editing Tools and RAG with Matryoshka Embeddings**: Community members have circulated a tool for editing JSONL datasets ([uncensored.com/jsonl](https://uncensored.com/jsonl)) and shared insights on [building RAG](https://x.com/Prashant_Dixit0/status/1806580075447590974) with Matryoshka Embeddings to enhance retrieval speed and memory efficiency, complete with a [Colab tutorial](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/tutorials/RAG-with_MatryoshkaEmbed-Llamaindex/RAG_with_MatryoshkaEmbedding_and_Llamaindex.ipynb).

- **Dappier Creates AI Content Monetization Opportunity**: The [Dappier platform](https://www.producthunt.com/posts/dappier-2-0), featured in [TechCrunch](https://techcrunch.com/2024/06/26/dappier-is-building-a-marketplace-for-publishers-to-sell-their-content-to-llm-builders/), provides a marketplace for creators to license content for AI training through a RAG API, signaling a new revenue stream for proprietary data holders.

- **Testcontainers Python SDK Boosts Ollama**: The Testcontainers Python SDK now supports Ollama, enhancing the ease of running and testing Large Language Models (LLMs) via Ollama, available in version **4.7.0**, along with example usage ([pull request #618](https://github.com/testcontainers/testcontainers-python/pull/618)).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mix-Up Between Mojolicious and Mojo Resolved**: Confusion ensued when a user asked for a **Mojo** code example and received a Perl-based **Mojolicious** sample instead; but it was clarified that the request was for info on **Modular's** AI development language **Mojo**, admired for its Python-like enhanced abilities and C-like robustness.

- **Caught in the REPL Web**: An anomaly was reported concerning the **Mojo REPL**, which connects silently and then closes without warning, prompting a discussion to possibly open a GitHub issue to identify and resolve this mysterious connectivity conundrum.

- **Nightly Notes: New Compiler and Graph API Slices**: **Modular's** latest nightly release '2024.6.2805' features a new compiler with LSP behavior tweaks and advises using `modular update nightly/mojo`; Developers also need to note the addition of *"integer literal slices across dimensions"*, with advice to document requests for new features via issues for traceability.

- **SDK Telemetry Tips and MAX Comes Back**: Guidance was shared on disabling telemetry in the **Mojo SDK**, with a helpful [FAQ link](https://docs.modular.com/mojo/faq#does-the-mojo-sdk-collect-telemetry) provided; The **MAX nightly builds** are operational again, welcoming trials of the *Llama3 GUI Chatbot* and feedback via the given [Discord link](https://discord.com/channels/1087530497313357884/1256010477637730315/1256010477637730315).

- **Meeting Markers and Community Collaterals**: The community is gearing up for the next **Mojo Community meeting**, scheduled for an unspecified *local time* with details accessible via [Zoom](https://modul.ar/community-meeting-zoom) and [Google Docs](https://modul.ar/community-meeting-doc); Plus, a warm nod to holiday celebrants in Canada and the U.S. was shared. Meanwhile, keeping informed through the **Modverse Weekly - Issue 38** is a click away at [Modular.com](https://www.modular.com/newsletters/modverse-weekly-38).



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Community Models Get Green Light**: The **Torchtune** team has expressed interest in community-contributed models, encouraging members to share their own implementations and enhance the library's versatility.
- **Debugging Diskourse's Debacles**: A puzzling issue in **Torchtune**'s text completions was tracked to end-of-sentence (EOS) tokens being erroneously inserted by the dataset, as detailed in a [GitHub discussion](https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_text_completion.py#L56).
- **Finding Favor in PreferenceDataset**: For reinforcement learning applications, the **PreferenceDataset** emerged as the favorable choice over the text completion dataset, better aligning with the rewarding of "preferred" input-response pairs.
- **Pretraining Pax**: Clarification in discussions shed light on pretraining mechanics, specifically that it involves whole documents for token prediction, steering away from fragmented input-output pair handling.
- **EOS Tokens: To Add or Not to Add?**: The community debated and concluded positively on introducing an **add_eos** flag to the text completion datasets within **Torchtune**, resolving some issues in policy-proximal optimization implementations.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Next-Gen Data Science IDE Alert**: Engineers discussed [Positron](https://github.com/posit-dev/positron), a future-forward data science IDE which was shared in the [#general](https://discord.com/channels/1238365980128706560/1238365980128706563/1256172869994545183) channel, suggesting its potential relevance for the community.

**Summarization Obstacle Course**: A technical query was observed about generating structured summaries from patient records, with an emphasis on avoiding hallucinations using Llama models; the community is tapped for strategies in prompt engineering and fine-tuning.

**LLAMA Drama**: Deployment of **LLAMA** to Streamlit is causing errors not seen in the local environment, as discussed in the [#🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1255989194275295242) channel; another member resolved a `FileNotFoundError` for **Tinyllama** by adjusting the dataset path.

**Credits Where Credits Are Due**: Multiple members have reported issues regarding missing credits for various applications, including requests in the [#fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1256121831342215219) and [#openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1256146343139737682) channels, stressing the need for resolution involving identifiers like `kishore-pv-reddy-ddc589` and organization ID `org-NBiOyOKBCHTZBTdXBIyjNRy5`.

**Link Lifelines and Predibase Puzzles**: In the [#freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1256256276900483132) channel a broken link was fixed swiftly, and a question was raised in the [#predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1255978316498735167) channel about the expiration of Predibase credits, however, it remains unanswered.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Secure Yet Open: Open Interpreter Tackles Security**: Open Interpreter's security measures, such as requiring user confirmation before code execution and sandboxing using Docker, were discussed, emphasizing the importance of community input for project safety.

- **Speed vs. Skill: Code Models in the Arena**: Engineers compared various code models, recognizing **Codestral** for superior performance, while **DeepSeek Coder** offers faster runtimes but at approximately 70% effectiveness. **DeepSeek Coder-v2-lite** stood out for its rapid execution and coding efficiency, potentially outclassing **Qwen-1.5b**.

- **Resource Efficiency Query: SMOL Model in Quantized Form**: Due to RAM constraints, there was an inquiry about running a SMOL multi-modal model in a quantized format, spotlighting the adaptive challenge for AI systems in limited-resource settings.

- **API Keys Exposed: Rabbit R1's Security Oversight**: A significant security oversight was exposed in a [YouTube video](https://youtu.be/lkbV8oP-F44), where Rabbit R1 was found to have hardcoded API keys in its codebase, a critical threat to user data security.

- **Modifying OpenInterpreter for Local Runs**: An AI engineer outlined the process for running OpenInterpreter locally using non-OpenAI providers, detailing the adjustments in a [GitHub issue comment](https://github.com/OpenInterpreter/01/issues/272#issuecomment-2119175075). Concerns were raised over additional API-related costs, on top of subscription fees.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**tinygrad gets new porting perks**: A new **port** that supports finetuning has been completed, signaling advancements for the tinygrad project.

**FPGA triumphs in the humanoid robot arena**: An 8-month-long project has yielded energy-efficient **humanoid robots** using **FPGA-based systems**, which is deemed more cost-effective compared to the current GPU-based systems that drain battery life with extensive power consumption.

**Shapetracker's zero-cost reshape revolution**: The *Shapetracker* in tinygrad allows for tensor reshaping without altering the underlying memory data, which was detailed in a [Shapetracker explanation](https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html), and discussed by members considering its optimizations over traditional memory strides.

**Old meets new in model storage**: In tinygrad, weights are handled by **safetensors** and compute by **pickle**, according to George Hotz, indicating the current methodology for model storage.

**Curiosity about Shapetracker's lineage**: Participants pondered if the concept behind **Shapetracker** was an original creation or if it drew inspiration from existing deep learning compilers, while admiring its capability to optimize without data copies.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Internship Inquiries Ignite Network**: A student intersects their academic focus on **LLMs and Reinforcement little reflectiong** with a real-world application by seeking DMs from **Cohere** employees about the company's work culture and projects. Engagement on the platform signals a consensus about the benefits of exhibiting a robust public project portfolio when vying for internships in the AI field.
- **Feature Requests for Cohere**: Cohere users demonstrated curiosity about potential new features, prompting a call for suggestions that could enhance the platform's offerings.
- **Automation Aspirations in AI Blogging**: Discussions arose around setting up **AI-powered automations** for blogging and social media content generation, directing the inquiry towards specialized assistance channels.
- **AI Agent Achievement Announced**: A member showcased an AI project called **Data Analyst Agent**, built using **Cohere and Langchain**, and promoted the creation with a [LinkedIn post](https://www.linkedin.com/posts/eddieotudor_datascience-aitechnology-machinelearning-activity-7212491482287542272-1cSF?utm_source=share&utm_medium=member_ios).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Gemma2 Gets Sample Packing via Pull Request**: A [GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718) was submitted to integrate **Gemma2** with sample packing. It's pending due to a required fix from Hugging Face, detailed within the PR.

- **27b Model Fails to Impress**: Despite the increase in size, the **27b model** is performing poorly in benchmarks when compared to the **9b model**, indicating there may be scaling or architecture inefficiencies.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Featherless.ai Introduces Flat-Rate Model Access**: [Featherless.ai](https://featherless.ai/) has launched a platform offering access to over 450+ models from Hugging Face at competitive rates, with the basic tier starting at $10/month and no need for GPU setup or download.
- **Subscription Scale-Up**: For $10 per month, the Feather Basic plan from **Featherless.ai** allows access up to 15B models, while the Feather Premium plan at $25 per month allows up to 72B models, adding benefits like private and anonymous usage.
- **Community Influence on Model Rollouts**: **Featherless.ai** is calling for community input on model prioritization for the platform, highlighting current popularity with AI persona local apps and specialized tasks like language finetuning and SQL model usage.




---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Curiosity for Chatbot Elo Evolution**: A user requested an extended timeline of **chatbot elo ratings** data beyond the provided six-week JSON dataset, expressing interest in the chatbot arena's evolving competitive landscape.
- **Observing the Elo Race**: From a start date of May 19th, there's a noted trend of the "pack" inching closer among leading chatbots in elo ratings, indicating a tight competitive field.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Feature Stores Step into the Spotlight**: An informative webinar titled "Building an Enterprise-Scale Feature Store with Featureform and Databricks" will be held on July 23rd at 8 A.M. PT. Simba Khadder will tackle the intricacies of feature engineering, utilization of Databricks, and the roadmap for handling data at scale, capped with a Q&A session. [Sign up to deep dive into feature stores](https://buff.ly/3zh3B74).



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1255960893062381721)** (549 messages🔥🔥🔥): 

- **Gemma-2 Updates with Faster Downloads and Less VRAM Fragmentation**: A new pre-quantized 4-bit versions of Gemma-2-27B and 9B, have been uploaded, promising 4x faster downloads and over 1GB less VRAM fragmentation for QLoRA finetuning. [Gemma-2-27B](https://huggingface.co/unsloth/gemma-2-27b-bnb-4bit) and [Gemma-2-9B](https://huggingface.co/unsloth/gemma-2-9b-bnb-4bit) now available on Huggingface.
- **Windows vs Linux for AI Development**: Members discussed the pros and cons of using Windows versus Linux for AI development. One user noted, "Windows is certainly not dead... But it feels more and more arbitrary every day," while another expressed frustrations with peripheral compatibility on Linux.
- **HF's Tiktoker-like Evaluation System**: Several members critiqued Hugging Face's evaluation system, comparing it to a popularity contest and suggesting premium paid evaluations. One stated, *"An evaluation should be allowed at any time if the user is willing to pay for it."*
- **The Challenges of Large Datasets**: A 2.7TB Reddit dataset was shared, but users warned it would take significant time and resources to clean. One member estimated, *"It's about 15 TB uncompressed... for meh data at best."*
- **AI Video Generation with ExVideo**: Multiple users reported impressive results using ExVideo for generating video content, though it required substantial VRAM (43GB). One member shared a link to a [GitHub repository for ExVideo Jupyter](https://github.com/camenduru/ExVideo-jupyter).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1806410668285030530">Tweet from Daniel Han (@danielhanchen)</a>: Uploaded pre-quantized 4bit bitsandbytes versions to http://huggingface.co/unsloth. Downloads are 4x faster & get &gt;1GB less VRAM fragmentation for QLoRA finetuning  Also install the dev HF pip inst...</li><li><a href="https://huggingface.co/datasets/aaronday3/entirety_of_reddit/tree/main">aaronday3/entirety_of_reddit at main</a>: no description found</li><li><a href="https://github.com/beam-cloud/beta9/">GitHub - beam-cloud/beta9: The open-source serverless GPU container runtime.</a>: The open-source serverless GPU container runtime. Contribute to beam-cloud/beta9 development by creating an account on GitHub.</li><li><a href="https://github.com/Alpha-VLLM/Lumina-T2X">GitHub - Alpha-VLLM/Lumina-T2X: Lumina-T2X is a unified framework for Text to Any Modality Generation</a>: Lumina-T2X is a unified framework for Text to Any Modality Generation - Alpha-VLLM/Lumina-T2X</li><li><a href="https://huggingface.co/datasets/aaronday3/entirety_of_reddit">aaronday3/entirety_of_reddit · Datasets at Hugging Face</a>: no description found</li><li><a href="https://ecnu-cilab.github.io/ExVideoProjectPage/?">ExVideo: Extending Video Diffusion Models via Parameter-Efficient Post-Tuning</a>: no description found</li><li><a href="https://ai.meta.com/research/cicero/diplomacy/">no title found</a>: no description found</li><li><a href="https://github.com/camenduru/ExVideo-jupyter">GitHub - camenduru/ExVideo-jupyter</a>: Contribute to camenduru/ExVideo-jupyter development by creating an account on GitHub.</li><li><a href="https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10">Subreddit comments/submissions 2005-06 to 2023-12</a>: no description found</li><li><a href="https://x.com/QuanquanGu/status/1805675325998907413">Tweet from Quanquan Gu (@QuanquanGu)</a>: We&#39;ve open-sourced the code and models for Self-Play Preference Optimization (SPPO)! 🚀🚀🚀  ⭐ code: https://github.com/uclaml/SPPO 🤗models: https://huggingface.co/collections/UCLA-AGI/sppo-6635f...</li><li><a href="https://foleycrafter.github.io/">FoleyCrafter</a>: no description found</li><li><a href="https://github.com/bmaltais/kohya_ss">GitHub - bmaltais/kohya_ss</a>: Contribute to bmaltais/kohya_ss development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py">unsloth/unsloth-cli.py at main · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html">OpenAI Compatible Server &#8212; vLLM</a>: no description found</li><li><a href="https://ploomber.io/blog/vllm-deploy/">Deploying vLLM: a Step-by-Step Guide</a>: Learn how to deploy vLLM to serve open-source LLMs efficiently</li><li><a href="https://github.com/MC-E/ReVideo">GitHub - MC-E/ReVideo</a>: Contribute to MC-E/ReVideo development by creating an account on GitHub.</li><li><a href="https://github.com/camenduru/FoleyCrafter-jupyter">GitHub - camenduru/FoleyCrafter-jupyter</a>: Contribute to camenduru/FoleyCrafter-jupyter development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/25Ij9G4haQ">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1255978788466983023)** (16 messages🔥): 

- **Unsloth and Gemma 2 get technical spotlight**: A user highlighted Daniel Han's [tweet](https://x.com/danielhanchen/status/1806372357684220308) analyzing Google's new Gemma 2 release, detailing significant technical aspects such as **pre & post layer norms, softcapping attention logits, and alternating sliding window/global attention**. The Gemma team also garnered thanks for early access, though Unsloth has yet to support finetuning for Gemma-2.

- **Knowledge Distillation sparks debate**: Users discussed the peculiarity and evolution of **Knowledge Distillation (KD)** in model training. One user humorously noted, *"Those 2 perplexity difference tho 😍"* and observed the shift from traditional KD to "modern" distillation methods.

- **Inference framework recommendations roll in**: Multiple users sought and recommended various **inference frameworks** for Unsloth-trained models, steering discussions toward issues like multi-GPU support and 4-bit loading. Recommendations included [vLLM](https://github.com/vllm-project/vllm) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), with some users noting existing bugs and limitations.

- **Gemma-2-9B finetuning wait continues**: Community members questioned the possibility of **finetuning Gemma-2-9B with Unsloth**, with responses clarifying that it isn't supported yet but is in progress.

- **Unsloth vs large models**: Comparisons were made between relatively smaller models like the 9B Gemma-2 and much larger models, with some users expressing surprise at the advancements in smaller model performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1806372357684220308">Tweet from Daniel Han (@danielhanchen)</a>: Just analyzed Google&#39;s new Gemma 2 release! The base and instruct for 9B & 27B is here!  1. Pre & Post Layernorms = x2 more LNs like Grok 2. Uses Grok softcapping! Attn logits truncated to (-30, 3...</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>: Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1255963221349765131)** (113 messages🔥🔥): 

- **Discussion on Training LM Head for Swedish Language Model**: Members debated the value of training the lm_head for Swedish, with one noting it is essential if the model does not know Swedish initially. Another highlighted the process can be done using one's machine to save costs, and results will be tested after the model reaches 8 epochs.

- **Inference Configuration Clarification**: A member queried about the difference between `FastLanguageModel.for_inference(model)` and `model.eval()`. Another member explained that the former loads a model while the latter switches it to evaluation mode, pointing out sample Unsloth notebooks use the former method.

- **Fine-Tuning and VRAM Management for Lang Models**: Members discussed VRAM limitations when fine-tuning with different batch sizes on GPUs like RTX 4090. It was shared that using power-of-two batch sizes avoids errors, despite some personal experiences to the contrary.

- **Support for LoRA on Quantized Models**: Members explored the feasibility of using Unsloth adapters on AWQ models, referencing a GitHub pull request that supports LoRA on quantized models. Some were unsure since documentation and real examples are scarce.

- **Continued Pretraining Issues and Solutions**: A member faced errors when inferring from a model after continued pretraining, using a 16GB T4. Recommendations included checking a relevant [GitHub issue](https://github.com/unslothai/unsloth/issues/702#issuecomment-2197477362) and ensuring no conflicts with the new PyTorch version.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/vllm-project/vllm/issues/5637">[Bug]: RuntimeError with tensor_parallel_size &gt; 1 in Process Bootstrapping Phase · Issue #5637 · vllm-project/vllm</a>: Your current environment The output of `python collect_env.py` Collecting environment information... PyTorch version: 2.3.0+cu121 Is debug build: False CUDA used to build PyTorch: 12.1 ROCM used to...</li><li><a href="https://github.com/unslothai/unsloth/issues/702#issuecomment-2197477362">Cache only has 0 layers, attempted to access layer with index 0 · Issue #702 · unslothai/unsloth</a>: I&#39;m encountering a KeyError when trying to train Phi-3 using the unsloth library. The error occurs during the generation step with model.generate. Below are the details of the code and the error t...</li><li><a href="https://github.com/vllm-project/vllm/pull/5669">[distributed][misc] use fork by default for mp by youkaichao · Pull Request #5669 · vllm-project/vllm</a>: fixes #5637 fork is not safe after we create cuda context. we should already avoid initializing cuda context before we create workers, so it should be fine to use fork, which can remove the necessi...</li><li><a href="https://github.com/vllm-project/vllm/pull/4012">[Core] Support LoRA on quantized models by jeejeelee · Pull Request #4012 · vllm-project/vllm</a>: Building upon the excellent work done in #2828 Since there hasn&#39;t been much progress on  #2828,so I&#39;d like to continue and complete this feature. Compared to #2828, the main improvement is the...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1256051149476200499)** (25 messages🔥): 

- **Seeking compute power for Toki Pona LLM**: A member is seeking access to compute resources for training an LLM on **Toki Pona**, a highly context-dependent language. They reported that their model, even after just one epoch, is preferred by strong Toki Pona speakers over ChatGPT-4o.
- **Oracle Cloud credits offer**: Another member offered a few hundred expiring Oracle Cloud credits and asked for a **Jupyter notebook** to run, expressing interest in fine-tuning the Toki Pona model using Oracle’s Data Science platform.
- **Discussing Oracle platform limitations**: There was a discussion about the limitations of Oracle's free trial, particularly the inability to spin up regular GPU instances, necessitating use of the Data Science platform's notebook workflows for model training and deployment.
- **Potential solutions and suggestions**: Members suggested adapting **Unsloth colabs** notebooks for fine-tuning on Oracle, specifically the Korean fine-tuning setup. One member offered to give the Oracle platform a try if another managed to run the notebook first.
- **Kubeflow Comparison**: One member compared Oracle's notebook session feature to typical **Jupyter** setups, mentioning it's similar to **SageMaker** or **Kubeflow's** approach to training and deploying machine learning workflows.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kubeflow.org/">Kubeflow</a>: Kubeflow makes deployment of ML Workflows on Kubernetes straightforward and automated</li><li><a href="https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-about.htm">Model Deployments</a>: no description found
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1256284481166245908)** (1 messages): 

- **Gemma 2 Lands in Transformers**: Google has released **Gemma 2** models, including **9B & 27B**, which are now available in the Transformers library. These models are designed to excel in the **LYMSYS Chat arena**, beating contenders like **Llama3 70B** and **Qwen 72B**.

- **Superior, Efficient, and Compact**: Highlights include **2.5x smaller size** compared to Llama3 and being trained on a lesser amount of tokens. The **27B** model was trained on **13T tokens** while the **9B** model on **8T tokens**.

- **Innovative Architecture Enhancements**: **Gemma 2** employs **knowledge distillation**, **interleaving local and global attention layers**, **soft attention capping**, and **WARP model merging** techniques. These changes aim at improving **inference stability**, reducing **memory usage**, and fixing gradient explosions during training.

- **Seamless Integration and Accessibility**: HuggingFace announced that **Gemma 2** models are now integrated into the **Transformers library**, and available on the **Hub**. Additional integrations are provided for **Google Cloud** and **Inference Endpoints** to ensure smooth usage.

- **Read All About It**: For a deep dive into the architectural and technical advancements of **Gemma 2**, a comprehensive [blog post](https://huggingface.co/blog/gemma2) is available. Users are encouraged to check out the **model checkpoints**, and the latest **Hugging Face Transformers release**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/gemma2">Welcome Gemma 2 - Google’s new open LLM</a>: no description found</li><li><a href="https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315">Gemma 2 Release - a google Collection</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1255961070108151858)** (482 messages🔥🔥🔥): 

- **Exploring FSS in Elixir**: A user provided an overview and [link](https://hexdocs.pm/fss/0.1.1/FSS.html) to FSS for file system abstraction in Elixir. They discussed its use cases and noted that it supports HTTP but doesn't seem extensible.
- **Gemma 2 GPT-4 Parameters Chat**: Users discussed items in [Google's announcement](https://developers.googleblog.com/en/gemma-family-and-toolkit-expansion-io-2024/) concerning Gemma 2. Some conversation noted trying different AI models like Gemma and their performance, with humor around the frustrations and odd behavior in models.
- **New Image Retrieval System**: User announced creating the first image-based retrieval system using open-source tools from HF. They shared their excitement and a [Colab implementation link](https://colab.research.google.com/drive/1DO5FwwLNDimh6B7T5BX9vNdFPtu1f_Pq?usp=sharing), along with a [Space for collaboration](https://huggingface.co/spaces/not-lain/RAG-on-images).
- **Visual Learning Models Discussed**: Recommendations and experiences shared for visual learning models, suggesting checking out [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4) and [Phi-3-Vision-128K-Instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct).
- **Queries on HuggingFace Tools**: Users asked questions related to specific HuggingFace tools and implementations, including fine-tuning guides and access tokens for new models like Gemma 2. A link was shared to the [HuggingFace docs for training](https://huggingface.co/docs/transformers/en/training).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://filesystem-spec.readthedocs.io/en/latest/">fsspec: Filesystem interfaces for Python &mdash; fsspec 2024.6.0.post1+g8be9763.d20240613 documentation</a>: no description found</li><li><a href="https://huggingface.co/spaces/Vipitis/shadermatch">ShaderMatch - a Hugging Face Space by Vipitis</a>: no description found</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4">THUDM/cogvlm2-llama3-chat-19B-int4 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://huggingface.co/coqui/XTTS-v2">coqui/XTTS-v2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/training">Fine-tune a pretrained model</a>: no description found</li><li><a href="https://tenor.com/view/brain-dog-brian-dog-cooked-wallahi-im-finished-cooked-dog-gif-1849480349705279416">Brain Dog Brian Dog GIF - Brain dog Brian dog Cooked - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://en.wikipedia.org/wiki/Serial_Experiments_Lain">Serial Experiments Lain - Wikipedia</a>: no description found</li><li><a href="https://huggingface.co/spaces/not-lain/RAG-on-images">Image Retriever - a Hugging Face Space by not-lain</a>: no description found</li><li><a href="https://tenor.com/view/this-dog-detects-twitter-twitter-user-dog-twitter-gif-90166331583470465">This Dog Detects Twitter Twitter User GIF - This dog detects twitter Twitter user Dog - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/monkey-cool-gif-25963936">Monkey Cool GIF - Monkey Cool - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/doubt-gif-13250124">Doubt GIF - Doubt - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/lightning-struck-by-gif-14902359">Lightning Struck GIF - Lightning Struck By - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/dbz-abridged-vegeta-gif-14758870">Dbz Abridged GIF - Dbz Abridged Vegeta - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lu.ma/fgwhcrsk">Techstars Startup Weekend Tokyo · Luma</a>: Techstars Startup Weekend Tokyo is an exciting and immersive foray into the world of startups. Over an action-packed three days, you’ll meet the very best…</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/tree/main/examples">LLaMA-Factory/examples at main · hiyouga/LLaMA-Factory</a>: Unify Efficient Fine-Tuning of 100+ LLMs. Contribute to hiyouga/LLaMA-Factory development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/not-lain/RAG-on-images/discussions/2">not-lain/image-retriever · i cant use git for the life of me. might need more testing</a>: no description found</li><li><a href="https://developers.googleblog.com/en/gemma-family-and-toolkit-expansion-io-2024/">Introducing PaliGemma, Gemma 2, and an Upgraded Responsible AI Toolkit</a>: no description found</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/tree/main/">GitHub - hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs</a>: Unify Efficient Fine-Tuning of 100+ LLMs. Contribute to hiyouga/LLaMA-Factory development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/82">zero-gpu-explorers/README · ZeroGPU Duration Quota Question</a>: no description found</li><li><a href="https://huggingface.co/datasets/coai/plantuml_generation">coai/plantuml_generation · Datasets at Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bpar6s/p40_still_worth_it/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://the-decoder.com/gpt-4-has-a-trillion-parameters/">GPT-4 has more than a trillion parameters - Report</a>: GPT-4 is reportedly six times larger than GPT-3, according to a media report, and Elon Musk&#039;s exit from OpenAI has cleared the way for Microsoft.</li><li><a href="https://huggingface.co/docs/hub/en/api#get-apidatasetsrepoidparquetconfigsplitnparquet">Hub API Endpoints</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1DO5FwwLNDimh6B7T5BX9vNdFPtu1f_Pq?usp=sharing>">Google Colab</a>: no description found</li><li><a href="https://hexdocs.pm/fss/0.1.1/FSS.html">FSS — fss v0.1.1</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/13v3b6q/multiple_cheap_gpus_or_a_single_expensive_one/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/NVIDIA/TensorRT-LLM">GitHub - NVIDIA/TensorRT-LLM: TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT-LLM also contains components to create Python and C++ runtimes that execute those TensorRT engines.</a>: TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficie...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1256127810238677053)** (6 messages): 

- **Learn 10 Machine Learning Algorithms in 1 Minute**: A member shared a [YouTube video](https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5) titled "10 Machine Learning Algorithms in 1 Minute", featuring a quick overview of top machine learning algorithms.

- **Interest in Reinforcement Learning Project**: A member expressed interest in learning Reinforcement Learning in a short duration. They proposed starting a small project to better understand the concepts, admitting they currently have only a vague idea of how it works.

- **Inquiring About Huggingface Course Updates**: A member is seeking information about the regular updates of the Huggingface courses compared to the "Natural Language Processing with Transformers (revised edition May 2022)" book. They also inquired about the up-to-dateness of the Diffusion and Community computer vision courses on the Huggingface website.

- **Improving Biometric Gait Recognition**: A member shared their progress on biometric gait recognition using basic 2D video inputs, achieving a 70% testing accuracy on identifying one out of 23 people. They plan to enhance the model by acquiring more datasets, combining several frames for RNN usage, and employing triplet loss for generating embeddings.

**Link mentioned**: <a href="https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5">10 Machine Learning Algorithms in 1 Minute</a>: Hey everyone! I just made a quick video covering the top 10 machine learning algorithms in just 1 minute! Here&#39;s a brief intro to each ( again ) :Linear Regr...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1255975841272299620)** (5 messages): 

- **Stimulating Blog on Diffusion Models**: A member highly recommended a [blog post by Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#citation) explaining diffusion models, including links to updates on various generative modeling techniques like GAN, VAE, Flow-based models, and more recent advancements like progressive distillation and consistency models.
- **Hermes-2-Pro-Llama-3-70B Released**: The upgraded [Hermes 2 Pro - Llama-3 70B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B) now includes function calling capabilities and JSON Mode. It achieved a 90% score on function calling evaluations and 84% on structured JSON Output.
- **Synthesize Multi-Table Data with Challenges**: An article discussed the complexities of [synthesizing multi-table tabular data](https://mltechniques.com/2024/06/15/synthesizing-multi-table-databases-model-evaluation-vendor-comparison/), including failures and difficulties with libraries like SDV, Gretel, and Mostly.ai, especially when dealing with columns containing dates.
- **Top Machine Learning Algorithms in a Minute**: A brief [YouTube video](https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5) titled "10 Machine Learning Algorithms in 1 Minute" promised to cover essential machine learning algorithms quickly. The video offers a fast-paced overview of key concepts.
- **AI Engineer World's Fair 2024 Highlights**: The [AI Engineer World’s Fair 2024](https://www.youtube.com/watch?v=5zE2sMka620) YouTube video covered keynotes and the CodeGen Track, with notable attendance from personalities like Vik. The event showcased significant advancements and presentations in AI engineering.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mltechniques.com/2024/06/15/synthesizing-multi-table-databases-model-evaluation-vendor-comparison/">Synthesizing Multi-Table Databases: Model Evaluation &amp; Vendor Comparison - Machine Learning Techniques</a>: Synthesizing multi-table tabular data presents its own challenges, compared to single-table. When the database contains date columns such as transaction or admission date, a frequent occurrence in rea...</li><li><a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#citation">What are Diffusion Models?</a>: [Updated on 2021-09-19: Highly recommend this blog post on score-based generative modeling by Yang Song (author of several key papers in the references)]. [Updated on 2022-08-27: Added classifier-free...</li><li><a href="https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5">10 Machine Learning Algorithms in 1 Minute</a>: Hey everyone! I just made a quick video covering the top 10 machine learning algorithms in just 1 minute! Here&#39;s a brief intro to each ( again ) :Linear Regr...</li><li><a href="https://github.com/alidenewade/Publications/blob/main/Budget%20Speech%20Essay%20Final.pdf">Publications/Budget Speech Essay Final.pdf at main · alidenewade/Publications</a>: Contribute to alidenewade/Publications development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>: https://twitter.com/aidotengineer</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B">NousResearch/Hermes-2-Pro-Llama-3-70B · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1255973799854215239)** (8 messages🔥): 

- **Flight Radar takes off into multilingual real-time tracking**: A member shares a multilingual real-time flight tracking web application built with Flask and JavaScript. The app utilizes the OpenSky Network API to let users view nearby flights, adjust search radius, and download flight data as a JPG. Find more details on [GitHub](https://github.com/U-C4N/Flight-Radar).

- **PixArt-900M Space launched**: A new 900M variant of PixArt is now available for experimentation with an in-progress checkpoint at various batch sizes. This **collaborative effort** by terminus research group and fal.ai aims to create awesome new models. Check it out on [Hugging Face Spaces](https://huggingface.co/spaces/ptx0/PixArt-900M).

- **Image retrieval system with Pokémon dataset goes live**: A fully open-source image retrieval system using a Pokémon dataset has been unveiled. The member promises a blog post about this tomorrow but you can try it now on [Hugging Face Spaces](https://huggingface.co/spaces/not-lain/image-retriever).

- **Top 10 Machine Learning Algorithms in a minute**: A quick YouTube video covering the top 10 machine learning algorithms in just one minute has been shared. [Watch it here](https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5).

- **AI-driven musical storytelling redefines genres**: An innovative album blending AI development and music has been introduced, offering a unique narrative experience designed for both machines and humans. The album is available on [Bandcamp](https://vonpsyche.bandcamp.com/album/the-prompt) and [SoundCloud](https://soundcloud.com/vonpsyche/sets/the-prompt), with a promo available on [YouTube](https://www.youtube.com/watch?v=RH9M7i8ft0E).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/not-lain/image-retriever">Image Retriever - a Hugging Face Space by not-lain</a>: no description found</li><li><a href="https://youtu.be/CaCl6B5gaA0?si=edE56MwOD4JwNBH5">10 Machine Learning Algorithms in 1 Minute</a>: Hey everyone! I just made a quick video covering the top 10 machine learning algorithms in just 1 minute! Here&#39;s a brief intro to each ( again ) :Linear Regr...</li><li><a href="https://github.com/U-C4N/Flight-Radar">GitHub - U-C4N/Flight-Radar: A multilingual real-time flight tracking web application using the OpenSky Network API. Built with Flask and JavaScript, it allows users to view nearby flights, adjust search radius, and supports six languages. Features include geolocation, and the ability to download flight data as a JPG</a>: A multilingual real-time flight tracking web application using the OpenSky Network API. Built with Flask and JavaScript, it allows users to view nearby flights, adjust search radius, and supports s...</li><li><a href="https://huggingface.co/spaces/ptx0/PixArt-900M">PixArt 900M 1024px Base Model - a Hugging Face Space by ptx0</a>: no description found</li><li><a href="https://vonpsyche.bandcamp.com/album/the-prompt">The Prompt, by Vonpsyche</a>: 12 track album</li><li><a href="https://soundcloud.com/vonpsyche/sets/the-prompt.">THE PROMPT</a>: Title: The Prompt Music by Vonpsyche Illustrations by Iron Goose.  Plot: The album takes listeners on a journey through a dystopian world where a brilliant AI developer strives to create a people-serv</li><li><a href="https://www.youtube.com/watch?v=RH9M7i8ft0E">The Prompt, by Vonpsyche</a>: Vonpsyche - The Prompt: Immerse yourself in a narrative that blurs the lines between reality and fiction. In an uncanny reflection of recent events, &#39;The Pro...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1256308604592590891)** (5 messages): 

- **New Event Coming Soon**: A member announced, *"I'll make an event in a bit!"* to the excitement of the group, which was met with reactions showing approval and anticipation.
- **Research Paper on Reasoning with LLMs**: A member shared an interesting [research paper on reasoning with LLMs](https://arxiv.org/pdf/2405.16506). Another member expressed curiosity about how it performs compared to RADIT, noting both might require finetuning but appreciating the inclusion of GNN methods.

**Link mentioned**: <a href="https://discord.gg/hugging-face-879548962464493619?event=1256365214895439974">Join the Hugging Face Discord Server!</a>: We&#x27;re working to democratize good machine learning 🤗Verify to link your Hub and Discord accounts! | 82343 members

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1255993476395569162)** (13 messages🔥): 

- **Seek YOLO for Web Automation Tasks**: A member inquired about using **YOLO** to identify and return coordinates of similar elements on a webpage using a reference image and a full screenshot. They are looking for an efficient method or an existing solution for their automation needs.

- **Exploring Efficient SAM Deployment**: A user sought advice on deploying the **Segment Anything Model (SAM)** efficiently and mentioned various efficient versions like **MobileSAM** and **FastSAM**. They are looking for best practices and equivalents to techniques like continuous batching and model quantization, often used in language models.

- **Mask Former Fine-Tuning Challenges**: Another member reported difficulties in fine-tuning the **Mask Former** model for image segmentation and questioned if the model, particularly **facebook/maskformer-swin-large-ads**, is catered more to semantic segmentation rather than instance segmentation.

- **Designing Convolutional Neural Networks**: A user expressed confusion over determining the appropriate number of convolutional layers, padding, kernel sizes, strides, and pooling layers for specific projects. They find themselves randomly selecting parameters, which they believe is not ideal.
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1256228265798930482)** (10 messages🔥): 

- **Leaderboard for Chatbot Arena Dataset Needs LLM Scripting**: A member has a chatbot arena-like dataset translated into another language and seeks to establish a leaderboard. They are struggling to find a script that would fill the "winner" field using an LLM instead of human votes.
- **Need for Chatbot Clarification**: When another member offered help, they clarified they were referring to a chatbot arena dataset. This caused some initial confusion, with a request misunderstood as needing a chatbot instead.
- **Human Preference in Arena Ratings**: Vipitis mentioned that the arena usually uses human preferences to calculate an Elo rating, suggesting a need for clearer guidance or alternative methods.
- **Urgent GEC Prediction Issue**: Shiv_7 expressed frustration over a grammar error correction (GEC) project where their predictions list is out of shape and urgently requested advice to resolve the issue.
  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1256262282804068362)** (1 messages): 

- **Older Gradio Versions' Share Links Deactivate Soon**: Starting next Wednesday, share links from Gradio versions 3.13 and below will no longer work. Upgrade your Gradio installation to keep your projects running smoothly by using the command `pip install --upgrade gradio`.
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1255964280680091822)** (105 messages🔥🔥): 

```html
<ul>
    <li><strong>Gemma 2 Support Now Available:</strong> Gemma 2 support has been added in LM Studio version 0.2.26. This update includes post-norm and other features, but users are reporting some integration bugs. <a href="https://github.com/ggerganov/llama.cpp/pull/8156">[GitHub PR]</a>.</li>
    <li><strong>Ongoing Issues with Updates and Integrations:</strong> Users are experiencing difficulties with Gemma 2 integration and auto-updates in LM Studio. Manual downloads and reinstallation of configs are suggested fixes, but some architectures like ROCm are still pending support.</li>
    <li><strong>Locally Hosted Models Debate:</strong> Advantages of hosting locally include privacy, offline access, and the opportunity for personal experimentation. Some express skepticism about its future relevance given the rise of cheap cloud-based solutions.</li>
    <li><strong>LLama 3 Model Controversy:</strong> Opinions differ on LLama 3's performance, with some claiming it is a disappointing model while others find it excels in creative tasks. Performance issues seem version-specific, with discussions around stop sequence bugs in recent updates.</li>
    <li><strong>Concerns Over Gemma 9B Performance:</strong> Some users report that Gemma 9B is underperforming compared to similar models like Phi-3, specifically on LM Studio. Ongoing development aims to address these issues, with functional improvements expected soon.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/.">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8197">Add attention and final logit soft-capping to Gemma2 by abetlen · Pull Request #8197 · ggerganov/llama.cpp</a>: This PR adds the missing attention layer and final logit soft-capping. Implementation referenced from huggingface transformers. NOTE: attention soft-capping is not compatible with flash attention s...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8156">Add support for Gemma2ForCausalLM by pculliton · Pull Request #8156 · ggerganov/llama.cpp</a>: Adds inference support for the Gemma 2 family of models. Includes support for:  Gemma 2 27B Gemma 2 9B  Updates Gemma architecture to include post-norm among other features.   I have read the contr...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1255978267828031488)** (222 messages🔥🔥): 

```html
- **Gemma-2 sparks discontent over context limit**: The announcement of **Gemma-2** with a 4k context limit was met with disappointment. One member described it as *"like building an EV with the 80mi range"*, underscoring the expectation for higher capacities in current models.
- **Confusion on Gemma-2 context limit**: While initial info suggested **Gemma-2** had a 4k context limit, others corrected it to 8k, showing discrepancies in information. One member pointed out *"Gemini is wrong about Google's product!"*.
- **Support sought for storytelling model**: A model designed for storytelling and full context use during training, [ZeusLabs/L3-Aethora-15B-V2](https://huggingface.co/ZeusLabs/L3-Aethora-15B-V2), was recommended for support. It's suggested to append “GGUF” when searching in the model explorer.
- **Deepseek Coder V2 Lite and Gemma 2 status**: **Gemma 2 9b** and **Deepseek coder V2 Lite** showed as not supported in LM Studio yet, prompting queries about their addition. A member confirmed **Gemma 2** as unsupported initially, but noted a [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/8156) that has since been merged to add support.
- **Discussion on best models in 7b~9b category**: The effectiveness of various models like **Qwen 2 7b**, **Deepseek Coder V2 Lite**, and **Llama 3** was debated. One member concluded *"Deepseek is worth it"* after performance tests, but also pointed to **Qwen 2 7b** issues without Flash Attention enabled.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/>">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://huggingface.co/ZeusLabs/L3-Aethora-15B-V2">ZeusLabs/L3-Aethora-15B-V2 · Hugging Face</a>: no description found</li><li><a href="https://x.com/AIatMeta/status/1806361623831171318">Tweet from AI at Meta (@AIatMeta)</a>: Today we’re announcing Meta LLM Compiler, a family of models built on Meta Code Llama with additional code optimization and compiler capabilities. These models can emulate the compiler, predict optima...</li><li><a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>: no description found</li><li><a href="https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9">GGUF quantizations overview</a>: GGUF quantizations overview. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8156">Add support for Gemma2ForCausalLM by pculliton · Pull Request #8156 · ggerganov/llama.cpp</a>: Adds inference support for the Gemma 2 family of models. Includes support for:  Gemma 2 27B Gemma 2 9B  Updates Gemma architecture to include post-norm among other features.   I have read the contr...</li><li><a href="https://tenor.com/view/cartoons-tom-and-jerry-ok-mouse-ok-i-got-it-gif-17005831">Cartoons Tom And Jerry GIF - Cartoons Tom And Jerry Ok - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1256323247704641609)** (1 messages): 

- **LM Studio 0.2.26 launches with Gemma 2 support**: The new LM Studio 0.2.26 now supports Google's Gemma 2 models, specifically the **9B** and **27B** versions. Check them out on the [lmstudio-community page](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF).
- **Windows on ARM64 debut**: LM Studio is now available for Windows on ARM (Snapdragon X Elite PCs) thanks to a collaboration with Qualcomm. Download the ARM64 version from [lmstudio.ai](https://lmstudio.ai/snapdragon).
- **Sign up for LM Studio 0.3.0 private beta**: A significant update to LM Studio is nearly complete, and testers are invited to help by signing up [here](https://forms.gle/K7pTWgTJsdHBmUaWA).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://lmstudio.ai/snapdragon">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://forms.gle/K7pTWgTJsdHBmUaWA">LM Studio 0.3.0 - Private Beta Sign Up</a>: Thanks for your interest in helping out test our upcoming release.   LM Studio 0.3.0 is gem-packed with new features and we&#39;d love your help to shake out the bugs before sending it out to the worl...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1255991634538008669)** (13 messages🔥): 

- **Llama.cpp error with unsupported model architecture**: Members experienced issues with the error message *'error loading model architecture: unknown model architecture: gemma2'*. One member noted that this error is due to the architecture not being supported by Llama.cpp.
- **Snapdragon X Elite praised for performance**: A member thanked the LM Studio team for quickly supporting Snapdragon X Elite systems, noting these devices perform high with low noise and excellent battery life. In benchmarks, the Snapdragon X Elite outperformed an i7 12700K on CPU/memory tasks but fell short compared to a 4090 GPU.
- **Unsupported models in LM Studio**: Members discussed attempting to run the "gemma 2 9 b" model and realized it is not yet supported in LM Studio. They were advised to use older models or explore alternatives like transformer or MLX with quantized gguf files.
- **IPv6 and syntax errors on Ubuntu**: One user resolved a model loading issue by disabling IPv6 on Ubuntu 22.04 but continues to encounter a "config-presset file syntax error" on launch, unsure of its impact.
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/)** (1 messages): 

cos2722: hello. can someone help me on making GORILL open funcion v2 work? i dont have any config
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1256058859596812298)** (28 messages🔥): 

- **Two Video Cards Supported by LM Studio**: A member asked, *"Is there value in using 2 video cards? Will lmstudio take advantage of them both?"* Another confirmed, **"Yes. And Yes."**

- **Small Code Gen Models on 4GB RAM**: The feasibility of running code generation LLMs on 4GB RAM was discussed. One suggestion was **Qwen 2 0.5B**, but its coding accuracy is described as *"mediocre at best;"* whereas **Claude 3.5 Sonnet** is recommended for better performance.

- **Multi-Agent Framework with Low-End Models**: A member plans to use a **0.5B model** as a user proxy in a multi-agent framework, believing it can manage that role easily. Another member expressed skepticism about the efficacy of such low-end models for coding tasks.

- **Lamini Memory Tuning Could Enhance LLM Accuracy**: The potential of [Lamini Memory Tuning](https://www.lamini.ai/blog/lamini-memory-tuning) was highlighted. This method **“improves factual accuracy and reduces hallucinations”** significantly and could make 0.5B models more effective on lower-end machines.

- **Mixed Reviews on Intel GPU Performance**: There were questions about Intel GPU effectiveness. One member noted **"CPU is faster"** while another added that **"Intel GPU support is in the works but currently below CPU on supported backends."**

**Link mentioned**: <a href="https://www.lamini.ai/blog/lamini-memory-tuning">Introducing Lamini Memory Tuning: 95% LLM Accuracy, 10x Fewer Hallucinations | Lamini - Enterprise LLM Platform</a>: no description found

  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1256075870263578705)** (33 messages🔥): 

- **"Gemma 2 is hot garbage" sparks skepticism**: The release of LM Studio 0.2.26 with Gemma 2 support received mixed reactions, with one user criticizing, "gemma is hot garbage... i seriously doubt they made improvements." Another user indicated issues with follow-up questions, sparking technical troubleshooting discussions.
- **Solution for "Unexpected end of JSON input" error**: A user encountering the JSON input error received advice to rename the problematic file and restart the application. They were also directed to specific Discord channels for further assistance.
- **Updating llama.cpp commit**: A user suggested updating to the latest llama.cpp commit for better performance. However, it was clarified that users need to wait for an official release incorporating the update.
- **Gemma 2 loading issues and solutions**: Users discussed issues and workarounds for Gemma 2, including reloading the model. One user highlighted the updated model settings and the need for LM Studio 0.2.26 for optimal performance.
- **Backtick formatting problem in markdown**: An issue with code block formatting in generated text was reported, where backticks were improperly placed, affecting the markdown rendering. The issue seemed transient and specific to certain code generations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF">lmstudio-community/gemma-2-9b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/futurama-angry-gif-13063135">Futurama Angry GIF - Futurama Angry - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1256340898820653086)** (4 messages): 

- **Support for Gemma 2 models questioned**: A member inquired if there has been any update on the ROCM preview to support Gemma 2 models, noting that the normal LM studio 0.2.6 release does not detect AMD GPUs like the ROCM preview version.
- **ROCm "extension pack" for Windows released**: A member announced the availability of the 0.2.26 ROCm "extension pack" for **Windows**, providing advanced installation instructions due to the current in-between development state. For details, refer to the [Extension Pack Instructions on GitHub](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#installation-on-windows-0226-).

**Link mentioned**: <a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#installation-on-windows-0226-">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs

  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1256291010150137896)** (1 messages): 

- **Gemma 2 Launches with a Bang**: Gemma 2 is now available for download with version 0.2.26 on Windows and Mac; a Linux version is coming soon. The 9B model is performing excellently, while the 27B model is under scrutiny for quirks, with [feedback being requested](https://lmstudio.ai/).
- **Grab Gemma 2 Models Easily**: The new models can be downloaded from the [LM Studio Community](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) on Hugging Face: the 9B model and the potentially quirky 27B model have been released and are ready for testing.

**Link mentioned**: <a href="https://lmstudio.ai/">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs

  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/)** (1 messages): 

mystic9t: it is surprisingly difficult to get them in a single no-code enviornment
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1255981680599240866)** (330 messages🔥🔥): 

- **Testing AI boundaries might get you banned**: Members expressed concerns about testing the limits of AI workarounds, with a reminder that violating OpenAI's usage policies could result in account suspension or termination. One shared a link to the [usage policies](https://openai.com/policies/usage-policies), emphasizing the importance of respecting safeguards.
- **Open-source vs. proprietary AI debate heats up**: Members debated the merits of open-sourcing advanced AI models, weighing the risks of potential misuse against the benefits of widespread access. One user argued that the economic displacement caused by restricting AI to the rich could be detrimental, while another emphasized the necessity of surveillance for public safety.
- **Exploring RLHF training experiences**: There was confusion and curiosity about Reinforcement Learning from Human Feedback (RLHF), with users discussing its application in OpenAI's models. Some mentioned seeing RLHF prompts very rarely, while others pondered how OpenAI manages public RLHF training.
- **Mass surveillance sparks intense discussion**: A deep conversation unfolded about the involvement of companies like OpenAI in surveillance, referencing [a blog post](https://openai.com/index/disrupting-deceptive-uses-of-AI-by-covert-influence-operations) on disrupting deceptive uses of AI. Users debated the ethics and necessity of such surveillance, with opinions diverging on the trade-offs between privacy and security.
- **Chatbot and API integrations in development**: Members shared experiences and projects related to integrating AI with other tools and services. One user detailed their work on a SearxNG integration for enhanced search capabilities within Discord, while another highlighted various AI models and APIs they're experimenting with for better functionality.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/J0KHiiTtt4w?si=_0wHUumSnvpWTNcw">Why Elon Musk says we&#39;re living in a simulation</a>: You may like playing The Sims, but Elon Musk says you are the Sim.Help us make more ambitious videos by joining the Vox Video Lab. It gets you exclusive perk...</li><li><a href="https://github.com/PierrunoYT/claude-3-artifacts">GitHub - PierrunoYT/claude-3-artifacts</a>: Contribute to PierrunoYT/claude-3-artifacts development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1255979888016621568)** (14 messages🔥): 

- **Plugins deprecated, GPTs take over**:  A user asked about using multiple plugin functions in a single chat, like a video summarizer and diagram maker. Another member clarified that *"Plugins are deprecated now and have been replaced by GPTs,"* but recommended using the `@mention` feature for flexibility to call multiple GPTs in a chat.

- **API access question for workgroups**: A user inquired about obtaining an API for workgroup use.

- **Struggles with custom GPT for medical questions**: A medical student shared issues with creating high-difficulty practice questions using a custom GPT. Despite uploading detailed guidelines and lecture information, the GPT-produced questions were subpar and sources were improperly cited. 

- **Lost GPTs, recovery steps**: Users reported losing access to their custom GPTs and sought help. Another member shared a solution, providing a URL [chatgpt.com/gpts/mine](https://chatgpt.com/gpts/mine) that redirected users, helping them restore access to their GPTs in the left pane.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1255968238370553897)** (25 messages🔥): 

- **Unicode semiotics puzzle**: A member discussed using **Unicode semiotics** for specific tasks, noting they "cost more tokens, not less" but consume fewer characters. Despite finding it useful for in-context learning, they could not reference any explanatory paper.
- **API struggles with unshuffle games**: Another member shared difficulties with the API solving unshuffle games like "edtPto lumAli" to result in "Potted Allium." There's a [shared approach](https://chatgpt.com/share/5c4d7258-cc5c-47f0-8a67-843d7b96c1d8) suggesting using Python alongside the API to improve results.
- **Prompt engineering advice**: A user asked for prompt engineering recommendations for transitioning from coding to PM/Business analysis tasks. Simple, clear, and concise prompts in plain language were advised, and the concept of "logit bias" was briefly mentioned for deeper prompt control.
- **Quasi-determinism confusion**: The concept of a "quasi-deterministic" nature of **stochastic neural networks** was discussed to clarify how these models behave. This explanation received a mixed reaction, hinting at the complex understanding required.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1255968238370553897)** (25 messages🔥): 

- **Unicode Semiotics Costs More Tokens**: Members discussed the use of Unicode semiotics for token cost savings or lower latency. It was clarified that Unicode semiotics consume fewer characters but cost more tokens, and there are no papers explaining this yet.

- **Struggles with API Unshuffle Games**: A member shared difficulties in getting the API to solve unshuffle games like "edtPto lumAli" into "Potted Allium." Another suggested using Python to generate all possible reorganizations of the words and then letting the API pick the correct ones, though hallucinations like "dotted" can occur.

- **Advice on Engineering Prompts**: Newbie inquiries on prompt engineering and negative weighting led to suggestions for using simple and plain language. The term "logit bias" was mentioned as a potential advanced technique.

- **Discussion on Deterministic Nature of Neural Networks**: A brief exchange clarified that the reverse function in neural networks tends to be stochastic and quasi-deterministic rather than fully deterministic.

- **Unsolved Semiotics Paper Inquiry**: One user asked if there was a paper on Unicode semiotics, but it was confirmed that no such documentation currently exists.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1255971125079576728)** (297 messages🔥🔥): 

<ul>
    <li><strong>Creating datasets with custom styles sparks interest and concerns:</strong> A user shared they are building a large dataset of generated images in their custom style. Concerns about generating NSFW content and getting banned were discussed, highlighting the nuances of monitoring datasets and image generation.</li>
    <li><strong>Automatic1111 troubles lead users to explore alternatives:</strong> Users reported frustrations with Automatic1111 breaking or crashing, leading some to switch to alternatives like Forge, though it has memory management issues. A YouTube guide ([Stable Diffusion Webui Forge Easy Installation](https://www.youtube.com/watch?v=FKzvHFtc8N0&t=64s)) was shared for installation help.</li>
    <li><strong>Cascade channel debate continues:</strong> Many users expressed their desire to unarchive the Cascade channel, citing valuable discussions and knowledge contained within. Frustration was evident, with some users suspecting a broader move to push engagement with SD3.</li>
    <li><strong>Model training nuances and tools discussed:</strong> Users discussed specifics about LoRa training, samplers like 3m sde exponential, and VRAM constraints, sharing tips and experiences. The utilization and limitations of different nodes and UI tools like ComfyUI, Forge, and Stable Swarm were highlighted.</li>
    <li><strong>Discord management and community dissatisfaction:</strong> Numerous users expressed dissatisfaction with the removal of channels and archives, suspecting it represents a shift in focus away from the community-driven aspects. There were calls for better communication and preservation of community-created resources.</li>
    <li><strong>YouTube humor lightens the tone:</strong> The discussion saw moments of humor, including a playful share of the YouTube video [Was Not Was - Walk The Dinosaur](https://youtu.be/vgiDcJi534Y) and jokes about colorful profile pictures and nostalgic emojis like <:kek:692062611659030548>.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jellybox.com">Tweet from Jellybox</a>: Run AI models locally. Private and entirely offline! Jellybox unlocks the power of local AI tools for everyone in a simple and easy to use package. From chatting, to agents, to image generation, Jelly...</li><li><a href="https://www.youtube.com/watch?v=FKzvHFtc8N0&t=64s">Stable Diffusion Webui Forge Easy Installation</a>: Stable Diffusion Webui Forge Easy Installation. No need to download and install python or anything else as it&#39;s all included in the installed, just down load...</li><li><a href="https://youtu.be/vgiDcJi534Y">Was Not Was - Walk The Dinosaur</a>: And Lo the Dinosaur was walked and thus began the end of their kind.</li><li><a href="https://github.com/lkwq007/stablediffusion-infinity">GitHub - lkwq007/stablediffusion-infinity: Outpainting with Stable Diffusion on an infinite canvas</a>: Outpainting with Stable Diffusion on an infinite canvas - lkwq007/stablediffusion-infinity
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1255966220784767016)** (50 messages🔥): 

```html
- **Scarlet AI Preview Launched**: A member introduced a preview of **Scarlet AI** intended for planning complex projects and delegating tasks. Test it at [https://app.scarletai.co/](https://app.scarletai.co/), though it's not yet production-ready.
- **Character AI Voice Features**: **Character.AI** launched **Character Calls** allowing users to interact with AI characters via phone calls for various use cases like practicing interviews and RPGs. Try it on their mobile app at [https://share.character.ai/Wv9R/6tdujbbr](https://share.character.ai/Wv9R/6tdujbbr).
- **Meta's LLM Compiler for Code Optimization**: Meta introduced the **Large Language Model Compiler** designed for compiler optimization tasks, enhancing understanding of intermediate representations and optimization techniques. More details available in their [research publication](https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/).
- **LangGraph Cloud for Reliable Agents**: **LangChainAI** launched **LangGraph Cloud** for fault-tolerant, scalable agent workflows with integrated tracing and monitoring. Join the waitlist and read more in their [blog post](http://bit.ly/langgraph-cloud-blog-1).
- **Adept Strategy Shift & Co-Founders Joining Amazon**: **Adept** announced updates to their strategy and changes in leadership, with several co-founders joining Amazon's AGI team. Get more details from the [GeekWire article](https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/).
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/">Amazon hires founders from well-funded enterprise AI startup Adept to boost tech giant&#8217;s &#8216;AGI&#8217; team</a>: (GeekWire File Photo / Kevin Lisota) Amazon is amping up its AI efforts by hiring executives from Adept, a San Francisco-based startup building &quot;agents&quot;</li><li><a href="https://x.com/AdeptAILabs/status/1806773469155381705?t=HevOdjCZ31VyPecgHs5KoQ&s=19">Tweet from Adept (@AdeptAILabs)</a>: Today, we’re announcing some updates to our strategy and some changes to our leadership and team. More details are in our blog: https://www.adept.ai/blog/adept-update</li><li><a href="https://x.com/noamshazeer/status/1806375332863418564?s=46&t=90xQ8sGy63D2Otiao">Tweet from Noam Shazeer (@NoamShazeer)</a>: Incredibly proud of the team for our official launch of Character Calls!  Quoting Character.AI (@character_ai)   AI Chat just got real.  Introducing Character Calls, the latest addition to our suite o...</li><li><a href="https://x.com/noamshazeer/status/1806375332863418564?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Noam Shazeer (@NoamShazeer)</a>: Incredibly proud of the team for our official launch of Character Calls!  Quoting Character.AI (@character_ai)   AI Chat just got real.  Introducing Character Calls, the latest addition to our suite o...</li><li><a href="https://x.com/tiagoefreitas/status/1806428334349504905?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Tiago Freitas (@tiagoefreitas)</a>: Just launched a preview of the new http://scarletai.co  Everyone&#39;s a manager!  Scarlet grants agency to individuals and founders alike. Eventually powering the first unicorn solopreneurs!  We enab...</li><li><a href="https://x.com/llama_index/status/1806116419995844947?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">Tweet from LlamaIndex 🦙 (@llama_index)</a>: ✨ Just announced on stage at @aiDotEngineer World&#39;s Fair! ✨ A brand new framework for getting multi-agent AI systems into production!   Currently an alpha release, llama-agents provides: ⭐️ Distri...</li><li><a href="https://x.com/DavidKPiano/status/1806417216914817514?t=99I0TJJfrKHHDQYeiizv8A&s=19">Tweet from David K 🎹 (@DavidKPiano)</a>: I love how AI startups are gradually (re)discovering state machines and the actor model for agent behavior & systems  Still unsure why you would need specialized infra for it though; it&#39;s all just...</li><li><a href="https://x.com/LangChainAI/status/1806371717084025165?t=15TNW0RaIb6EoIJ">Tweet from LangChain (@LangChainAI)</a>: 🚀 Introducing LangGraph Cloud 🚀  LangGraph helps you build reliable agents that actually work. Today, we&#39;ve launched LangGraph Cloud, our new infrastructure to run fault-tolerant LangGraph agent...</li><li><a href="https://x.com/LangChainAI/status/1806371717084025165?t=15TNW0RaIb6EoIJKPq_IjA&s=19">Tweet from LangChain (@LangChainAI)</a>: 🚀 Introducing LangGraph Cloud 🚀  LangGraph helps you build reliable agents that actually work. Today, we&#39;ve launched LangGraph Cloud, our new infrastructure to run fault-tolerant LangGraph agent...</li><li><a href="https://ai.meta.com/research/publications/meta-large-language-model-compiler-foundation-models-of-compiler-optimization/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1256021421574852659)** (2 messages): 

- **OpenAI Demo Announcement**: A message alerted everyone to an OpenAI demo with urgency, directing them to a specific [OpenAI Demo channel](https://discord.com/channels/822583790773862470/1197350122112168006). The message lacked additional context but indicated an immediate event.

- **OSS GPT Store Rundown Reminder**: Members were reminded about the OSS GPT Store rundown scheduled for an hour later. The reminder included a prompt to join a specific channel and pick up a role for future notifications.
  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1255962568636502156)** (150 messages🔥🔥): 

- **"GPT-4o to dominate desktops"**: Discussions revealed excitement around using GPT-4o on desktop for coding assistance, suggesting it "help[s] you code, etc.” Members expressed interest in trying Open-Interpreter for this purpose and its integration with local models.
- **Linux vs. Mac for streaming issues**: Members faced technical difficulties while trying to stream using Linux, noting issues with permissions and screen sharing. One joked about the need for a Mac with *"such riches"* highlighting the struggle (*"Maybe not worth the hassle for covering stuff. ya desktop app is p cool"*).
- **Live streaming woes and fixes**: The group experienced streaming issues, predominantly around poor video and audio feeds. The problem was somewhat alleviated by switching to a wired connection for stability.
- **Cursor power users and productivity tips**: One member asked for “good cursor power user content” to boost productivity. Another recommended *“indydevdan on YT”* for useful workflows and various configuration tools to improve coding efficiency with Vim.
- **Vesktop as a solution**: To address Discord performance issues on Linux, members suggested using [Vesktop](https://github.com/Vencord/Vesktop), a custom Discord app aimed at enhancing performance and support for Linux users.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=vaIiNZoXymg">AI Engineer World’s Fair 2024 - Keynotes &amp; Multimodality track</a>: https://twitter.com/aidotengineer</li><li><a href="https://www.youtube.com/live/vaIiNZoXymg">AI Engineer World’s Fair 2024 - Keynotes &amp; Multimodality track</a>: https://twitter.com/aidotengineer</li><li><a href="https://github.com/Vencord/Vesktop">GitHub - Vencord/Vesktop: Vesktop is a custom Discord App aiming to give you better performance and improve linux support</a>: Vesktop is a custom Discord App aiming to give you better performance and improve linux support - Vencord/Vesktop</li><li><a href="https://github.com/dimfeld/dotfiles/blob/master/nvim/.config/nvim/lua/commands/llm.lua">dotfiles/nvim/.config/nvim/lua/commands/llm.lua at master · dimfeld/dotfiles</a>: Contribute to dimfeld/dotfiles development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1256338676573667530)** (34 messages🔥): 

- **Public GPTs prompt leak possibility**: A member highlighted that while specific GPT definitions might not be easy to access, they are not truly private and have been extracted by others on [GitHub](https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts). Another member added, *"best to assume someone could get these, so no secrets in these."*
- **Insightful research papers shared**: One member pointed out the value of certain research papers, sharing a link to [arxiv.org/abs/2309.02427](https://arxiv.org/abs/2309.02427) discussing Cognitive Architectures for Language Agents (CoALA) and another link to a related [GitHub repository](https://github.com/ysymyth/awesome-language-agents). These papers provide a framework to organize existing language agents and plan future developments.
- **Great talk and presentation praise**: Numerous members expressed appreciation for a well-prepared presentation, with comments like, *"Great talk"* and *"Thanks!"*. The presenter was specifically praised for their preparation and contributions.
- **AI engineer conference recap suggestion**: For future sessions, one member suggested doing a recap of an AI engineer conference, possibly incorporating a bunch of lightning talks. This idea received positive feedback from others in the chat.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ysymyth/awesome-language-agents">GitHub - ysymyth/awesome-language-agents: List of language agents based on paper &quot;Cognitive Architectures for Language Agents&quot;</a>: List of language agents based on paper &quot;Cognitive Architectures for Language Agents&quot; - ysymyth/awesome-language-agents</li><li><a href="https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts">chatgpt_system_prompt/prompts/gpts at main · LouisShark/chatgpt_system_prompt</a>: A collection of GPT system prompts and various prompt injection/leaking knowledge. - LouisShark/chatgpt_system_prompt</li><li><a href="https://arxiv.org/abs/2309.02427">Cognitive Architectures for Language Agents</a>: Recent efforts have augmented large language models (LLMs) with external resources (e.g., the Internet) or internal control flows (e.g., prompt chaining) for tasks requiring grounding or reasoning, le...
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1256114993519263845)** (2 messages): 

- **Instruction Pre-Training boosts LM performance**: A new paper proposes **Instruction Pre-Training**, which augments large corpora with **200M instruction-response pairs** generated by an efficient instruction synthesizer. This method not only enhances pre-trained base models but also allows **Llama3-8B** to compete with **Llama3-70B** in continual pre-training. [Access the full paper](https://arxiv.org/abs/2406.14491) or check out the [model on Hugging Face](https://huggingface.co/instruction-pretrain/finance-Llama3-8B).
- **MCT Self-Refine improves mathematical reasoning**: The **MCT Self-Refine (MCTSr) algorithm** integrates Large Language Models (LLMs) with **Monte Carlo Tree Search (MCTS)** to boost performance in complex mathematical tasks. Extensive experiments show MCTSr significantly improving success rates in solving Olympiad-level problems, leveraging a systematic exploration and heuristic self-refine process. [Read the detailed study](https://arxiv.org/abs/2406.07394).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.07394">Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B</a>: This paper introduces the MCT Self-Refine (MCTSr) algorithm, an innovative integration of Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS), designed to enhance performance in complex m...</li><li><a href="https://arxiv.org/abs/2406.14491">Instruction Pre-Training: Language Models are Supervised Multitask Learners</a>: Unsupervised multitask pre-training has been the critical method behind the recent success of language models (LMs). However, supervised multitask learning still holds significant promise, as scaling ...</li><li><a href="https://huggingface.co/instruction-pretrain">instruction-pretrain (instruction-pretrain)</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1256081294694023279)** (10 messages🔥): 

```html
- **Public Channels Now Open**: A post announced that the channels <#1105324249721356298> and <#1104063238934626386> have been made public.

- **REVEAL Benchmarks Verifiers**: A new dataset, [REVEAL: Reasoning Verification Evaluation](https://reveal-dataset.github.io), benchmarks automatic verifiers of complex Chain-of-Thought reasoning in open-domain question-answering settings, highlighting their struggles, especially in verifying logical correctness. The dataset, detailed in an [arXiv paper](https://arxiv.org/abs/2402.00559), includes comprehensive labels and free-text justifications.

- **XTREME Evaluates Multilingual Models**: The [XTREME dataset](https://huggingface.co/datasets/google/xtreme) evaluates cross-lingual generalization ability of pre-trained multilingual models, covering 40 typologically diverse languages. It includes nine tasks requiring different levels of syntax and semantics reasoning.

- **SPIQA Challenges Multimodal Models**: The [SPIQA dataset](https://huggingface.co/datasets/google/spiqa) is designed for multimodal question answering on scientific papers, containing over 270K questions focused on figures, tables, and text paragraphs. This dataset aims to assess the capability of large multimodal models in comprehending complex figures and tables.

- **TACT Tests Numerical Reasoning**: [TACT](https://huggingface.co/datasets/google/TACT) is introduced to evaluate LLMs' reasoning and computational abilities using complex instructions through tables. The dataset shows that contemporary LLMs perform poorly, with overall accuracy below 38%.

- **UNcommonsense Explains Weird Situations**: [UNcommonsense](https://huggingface.co/datasets/allenai/UNcommonsense) focuses on explaining unusual and unexpected situations with an English-language corpus consisting of 20k unique contexts and 41k abductive explanations, offering insights into uncommon outcomes.

- **EmotionalIntelligence-50K Focuses on Emotions**: The [EmotionalIntelligence-50K dataset](https://huggingface.co/datasets/OEvortex/EmotionalIntelligence-50K) is designed to build and train models that understand and generate emotionally intelligent responses, containing 51,751 rows of text data on various prompts and responses.

- **BrightData/IMDb-Media Offers Comprehensive Film Data**: The [BrightData/IMDb-Media dataset](https://huggingface.co/datasets/BrightData/IMDb-Media) includes over 249K records with 32 data fields covering feature films, TV series, and more, regularly updated with extensive details such as ratings, reviews, cast, and budget.

- **Opus-WritingPrompts Includes Sensitive Content**: The [Opus-WritingPrompts dataset](https://huggingface.co/datasets/Gryphe/Opus-WritingPrompts) features 3008 short stories generated using Reddit's Writing Prompts. This dataset includes varied content, including erotica, and has a disclaimer for sensitive information.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://reveal-dataset.github.io">REVEAL</a>: A Chain-of-Thought Is as Strong as Its Weakest Link: A Benchmark for Verifiers of Reasoning Chains</li><li><a href="https://huggingface.co/datasets/google/TACT">google/TACT · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/google/spiqa">google/spiqa · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Gryphe/Opus-WritingPrompts">Gryphe/Opus-WritingPrompts · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/allenai/UNcommonsense">allenai/UNcommonsense · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/OEvortex/EmotionalIntelligence-50K">OEvortex/EmotionalIntelligence-50K · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/BrightData/IMDb-Media">BrightData/IMDb-Media · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/google/xtreme">google/xtreme · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/)** (1 messages): 

deoxykev: Personally I’d go straight for the empirical approach. Too many variables at play.
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1256328863865241630)** (1 messages): 

- **Discussing Longevity Research**: A member shared concerns about a potential dystopian society *"where old wealthy people live forever by sacrificing the lifespan of youths,"* while also expressing appreciation for research aimed at increasing elderly health. They suggested that such advancements should be approached in a safe manner.
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1255992857387470991)** (9 messages🔥): 

- **RankGPT is expensive and confusing**: A member commented that "RankGPT is expensive," and another user questioned what "reranking by embedding" means and why it has tokens. They later figured it out but the initial confusion highlights the complexity of the tool.
- **RAG Dataset should be public**: Discussing the reranking process, a member noted the necessity for the RAG dataset to be a public project, suggesting community access could improve understanding and utilization.
- **Smooth brain prefers Hermes 0 shot**: One user mentioned their preference for a method from a paper showing Hermes 0 shot with "good or bad" as the most effective, despite acknowledging room for improvement. They humorously confessed to wanting to avoid complex problem-solving to keep their "brain nice and smooth."
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1255974809028792350)** (1 messages): 

- **Hermes 2 Pro 70B Released**: Nous Research has released **Hermes 2 Pro 70B**, a pure Hermes model with no merge with Llama-3 Instruct. This update addresses function call issues and refusals but sacrifices a bit of performance. Check it out on [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B).

- **Enhanced for Function Calling and JSON Outputs**: Hermes 2 Pro excels at *Function Calling* and *JSON Structured Outputs*, achieving scores of 90% and 84% respectively in evaluations. The model is based on an updated OpenHermes 2.5 Dataset and includes a **Function Calling and JSON Mode dataset**.

- **Improvement on Several Metrics**: The new Hermes 2 Pro maintains **excellent general task and conversation capabilities**. It has shown improvements in several areas, including structured JSON output, and function calling, *developed in partnership with Fireworks.AI*.

**Link mentioned**: <a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B">NousResearch/Hermes-2-Pro-Llama-3-70B · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1255969935465775228)** (111 messages🔥🔥): 

- **Smart and context-aware 8B model surprises users**: Members discussed the impressive performance and context awareness of an 8B model, noting its ability to understand nuances in conversations. A user shared, *"I ask it a vague question implicating what we are going to do, and it's responses were correct!"*.
  
- **Confusion over Hermes models**: There was a brief confusion about whether the non-Theta Hermes Pro model should be preferred over the Theta version. A member clarified that the non-Theta version may be better for function calling or if experiencing tokenization issues.

- **Interest in OpenHermes 2.5 dataset cleaning methodology**: Members inquired about the cleaning process for the OpenHermes 2.5 dataset. Unfortunately, no detailed information was shared.

- **New tools and benchmarking datasets discussed**: Discussion on various new datasets and benchmarking tools, including REVEAL and UNcommonsense. Links shared include the [REVEAL dataset](https://reveal-dataset.github.io), [UNcommonsense dataset](https://huggingface.co/datasets/allenai/UNcommonsense), and models using Self-Play Preference Optimization like [Llama-3-8B-SPPO-Iter3](https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3).

- **Debate on SB 1047's impact on innovation**: Members debated the potential negative impacts of California's SB 1047 on AI innovation. A [link](https://live-2024-stop-sb-1047.pantheonsite.io) was shared discussing the potential unintended consequences of the bill, with one member stating, *"California should encourage innovation and learn to utilize AI to our strategic advantage."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/aaronday3/entirety_of_reddit">aaronday3/entirety_of_reddit · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/natolambert/status/1806437722334417131">Tweet from Nathan Lambert (@natolambert)</a>: Here&#39;s my full @interconnectsai interview with @deanwball on AI policy. We do pretty much a state of the union on all things AI policy, with our usual focuses on openness.  This was a great one! W...</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://live-2024-stop-sb-1047.pantheonsite.io/">Protect AI Research | STOP SB 1047</a>: Urge the legislature to protect Artificial Intelligence (AI) research and oppose SB 1047. Rather than overregulate AI at its infancy, we should encourage innovation and learn to utilize AI to our stra...</li><li><a href="https://suno.com/song/8aaf9c47-41d5-4a04-a05c-f3a49218ccc8">Beneath the Stars by @bozoegg | Suno</a>: progressive electronic atmospheric song. Listen and make your own with Suno.</li><li><a href="https://reveal-dataset.github.io">REVEAL</a>: A Chain-of-Thought Is as Strong as Its Weakest Link: A Benchmark for Verifiers of Reasoning Chains</li><li><a href="https://tenor.com/view/hug-love-heart-globe-planet-gif-16782181">Hug Love GIF - Hug Love Heart - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/interstellarninja/function-calling-eval/tree/main">GitHub - interstellarninja/function-calling-eval: A framework for evaluating function calls made by LLMs</a>: A framework for evaluating function calls made by LLMs - interstellarninja/function-calling-eval</li><li><a href="https://huggingface.co/google/gemma-2-9b">google/gemma-2-9b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/allenai/UNcommonsense">allenai/UNcommonsense · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3">UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3 · Hugging Face</a>: no description found</li><li><a href="https://x.com/QuanquanGu/status/1805675325998907413">Tweet from Quanquan Gu (@QuanquanGu)</a>: We&#39;ve open-sourced the code and models for Self-Play Preference Optimization (SPPO)! 🚀🚀🚀  ⭐ code: https://github.com/uclaml/SPPO 🤗models: https://huggingface.co/collections/UCLA-AGI/sppo-6635f...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/25Ij9G4haQ">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1255992054740287599)** (2 messages): 

- **Enthusiasm for Hermes2-Pro-llama-3-70B**: A user expressed excitement for the **Hermes2-Pro-llama-3-70B**. They inquired about the scenarios in which this model would be preferred over **Hermes-2-Theta**.
- **Link Shared without Context**: Another user shared a link to a specific Discord message, [link](https://discord.com/channels/1053877538025386074/1149866623109439599/1255975965272838275), suggesting it may contain relevant information or context.
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1256026605780734002)** (85 messages🔥🔥): 

- **Glaive-RAG-v1 dataset launched:** [Glaive-RAG-v1](https://huggingface.co/datasets/glaiveai/RAG-v1) has around 50k samples built using Glaive platform for RAG use cases, structured with documents, questions, answers, and citation tags. Members are asked to evaluate its quality and potential improvements for future iterations.
- **System Prompts and Domain Integration:** Discussion on integrating system prompts per domain into Hermes RAG prompts, covering "role", "style guide", and "instructions" sections. Members are considering practical ways to indicate context and relevance within prompts.
- **Relevance Scoring Mechanics:** There's an ongoing debate about including relevance scores such as a 5-point Likert scale for evaluating groundedness in responses. The consensus leans towards letting LLMs self-evaluate these metrics through guided system prompts.
- **Code and Tools Sharing:** Members discussed the utility of sharing tools and pipelines developed for the project. An example includes an image retriever pipeline shared via [Hugging Face Spaces](https://huggingface.co/spaces/not-lain/image-retriever).
- **Grounded vs. Mixed Response Modes:** Clarification that in "Grounded" mode, the model should only use information from provided documents, while in "Mixed" mode, it combines document information with the model's own knowledge.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/not-lain/image-retriever">Image Retriever - a Hugging Face Space by not-lain</a>: no description found</li><li><a href="https://x.com/bee__computer/status/1806448042406818203">Tweet from bee (@bee__computer)</a>: we still have some alphas available. pick one up at @aiDotEngineer. dm us</li><li><a href="https://huggingface.co/datasets/glaiveai/RAG-v1">glaiveai/RAG-v1 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/explodinggradients/ragas">GitHub - explodinggradients/ragas: Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines</a>: Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines - explodinggradients/ragas</li><li><a href="https://docs.google.com/spreadsheets/d/1f5fbPxhjGrmPqhbM0exOCX2vAzffRWufhF7QBp24OMw/edit?gid=0#gid=0">RAG Data Synthesis</a>: Sheet1  Domain,Curriculum file,Source/links,HF repo,Size (rows),Status,Who&#39;s working,Reviewer,Review Notes Websearch Wikipedia Codebase,WIP,Bexboy Academic Papers Books,WIP,EveryoneIsGross Finance...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1255976707345879162)** (5 messages): 

- **Surge of enthusiasm for Claude 3.5 Sonnet Model in world sim**: A member expressed excitement about the upcoming addition of the **Claude 3.5 Sonnet model** to the world simulation. This indicates a strong community interest in new AI model integrations.

**Link mentioned**: <a href="https://tenor.com/view/lain-lain-iwakura-serial-experiments-lain-wires-wired-gif-1481475804337586659">Lain Lain Iwakura GIF - Lain Lain iwakura Serial experiments lain - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1255962506417934358)** (25 messages🔥): 

- **Aggregate data increases privacy risks**: "Quickly you'll run into the situation where aggregate data from other users can inform or improve local models," leading to challenges in the **privacy-preserving/federated learning space**. This necessitates dealing with a "significantly wider attack space from malicious actors" ([BSI link](https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/Publications/Studies/KI/P464_Provision_use_external_data_trained_models.pdf?__blob=publicationFile&v=7)).

- **AI agent security behaviors**: Discussing AI agents implementing security behaviors, including activities like "identifying behaviors by scripts that compromise privacy" and "automatic things that degrade or destroy shady data collection." However, some argued this might be largely heuristic without AI generalizing abilities.

- **New MMLU-SR dataset**: A user introduced MMLU-SR, a new **reasoning challenge dataset** designed to measure comprehension abilities of Large Language Models (LLMs) ([arXiv link](https://arxiv.org/abs/2406.15468v1)). They found that LLMs perform poorly on modified test questions, suggesting poor true comprehension.

- **Trolling issues in chat**: Multiple users reported a banned user, "endomorphosis," trolling a specific channel under various accounts. Members requested his removal for a more positive community experience.

- **Channel guidance for lm_eval help**: New members seeking assistance with **lm_eval** were directed to the appropriate channel ([lm_eval channel link](https://discord.com/channels/729741769192767510/755950983669874798)). This spot is recommended for tasks and related queries.

**Link mentioned**: <a href="https://arxiv.org/abs/2406.15468v1">Reasoning or Simply Next Token Prediction? A Benchmark for Stress-Testing Large Language Models</a>: We propose MMLU-SR, a novel dataset designed to measure the true comprehension abilities of Large Language Models (LLMs) by challenging their performance in question-answering tasks with modified term...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1255960857041440989)** (122 messages🔥🔥): 

- **Debate over yoco and kv cache**: Members discussed the **kv cache** strategy in various architectures, particularly **yoco**, and expressed skepticism about its efficacy compared to alternative designs. One member called yoco's setup "cursed" due to its complexity and separation from standard transformer practices.

- **Efficiency in layer ordering**: There's a significant discussion around the **ordering of attention and feed-forward layers** in models like yoco and mamba. Some argue that placing all attention layers at one end could be more efficient, reducing computational costs, while others maintain that alternating layers might ensure better overall performance.

- **Model Scaling and Performance Concerns**: Participants debated the **impact of layer ordering** on small vs. large scale models, with some suggesting that issues at small scales might be smoothed out at larger scales. A key point of contention was whether reordering layers has measurable impacts as models grow.

- **Preliminary exploration on positional embeddings**: A member posed an innovative idea regarding **pre-applying positional embeddings (PE)** to latents before computing QK, hypothesizing it could handle operations like string reversal better. This sparked curiosity and skepticism among members, who questioned if such methods would preserve or disrupt existing benefits of techniques like RoPE.

- **Reinforcement Learning Advancements**: A member shared an [arXiv paper](https://arxiv.org/abs/2406.19320) discussing **$\Delta$-IRIS**, a new agent employing delta-based autoencoding in RL, addressing its efficiency in training time versus traditional attention-based methods.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.19320">Efficient World Models with Context-Aware Tokenization</a>: Scaling up deep Reinforcement Learning (RL) methods presents a significant challenge. Following developments in generative modelling, model-based RL positions itself as a strong contender. Recent adva...</li><li><a href="https://arxiv.org/abs/1911.03864">Improving Transformer Models by Reordering their Sublayers</a>: Multilayer transformer networks consist of interleaved self-attention and feedforward sublayers. Could ordering the sublayers in a different pattern lead to better performance? We generate randomly or...</li><li><a href="https://arxiv.org/abs/2403.00801">Self-Retrieval: Building an Information Retrieval System with One Large Language Model</a>: The rise of large language models (LLMs) has transformed the role of information retrieval (IR) systems in the way to humans accessing information. Due to the isolated architecture and the limited int...</li><li><a href="https://arxiv.org/abs/2012.15832">Shortformer: Better Language Modeling using Shorter Inputs</a>: Increasing the input length has been a driver of progress in language modeling with transformers. We identify conditions where shorter inputs are not harmful, and achieve perplexity and efficiency imp...</li><li><a href="https://colab.research.google.com/drive/1sXpRXz8KWa_OnWveOWvu1ZtgUL3tnPai?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1256062431613161513)** (45 messages🔥): 

- **Questioning Chinchilla's Status as a Law of Nature**: A member questioned why people seem to treat **Chinchilla scaling** as an immutable law, suggesting that power law scaling can't be the final optimal scaling method. They observed that there seems to be little discussion on alternatives and speculated that serious discussions on this topic might be occurring privately.

- **Debating the Validity of Power Law Scaling**: A member argued that **power law scaling** might still be a legitimate model but acknowledged that the **Chinchilla** model's relevance is tied to specific conditions like the training regime and data. They also pondered why inverse power relations couldn't be the norm, mentioning that both inverse power and logarithmic scaling seem plausible.

- **Scaling Heuristics vs. Laws**: Another member noted that terms like "scaling law" should perhaps be replaced with "scaling heuristic" to better reflect their provisional nature. They reminisced about a time when numerous papers claimed to have discovered new "laws" better than Chinchilla, implying skepticism about such definitive language.

- **Reading and Citing Key Papers**: Members referenced several key papers to bolster their arguments, including ["Parameter Counts in Machine Learning"](https://www.alignmentforum.org/posts/GzoWcYibWYwJva8aL/parameter-counts-in-machine-learning) and [Adlam 2021 on Scaling Laws](https://gwern.net/doc/ai/scaling/2021-adlam.pdf). They discussed how these papers understand and model scaling laws concerning large datasets and parameter sizes.

- **Practical Limitations and Future Directions**: The conversation also touched on practical aspects like **data selection methods** and their impact on training efficiency. A member emphasized that better data collection methods, such as stratified sampling, aren't magical solutions but do improve efficiency, highlighting the complexity of predicting the future impact of data on model performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2210.16859">A Solvable Model of Neural Scaling Laws</a>: Large language models with a huge number of parameters, when trained on near internet-sized number of tokens, have been empirically shown to obey neural scaling laws: specifically, their performance b...</li><li><a href="https://arxiv.org/abs/1905.10843">Asymptotic learning curves of kernel methods: empirical data v.s. Teacher-Student paradigm</a>: How many training data are needed to learn a supervised task? It is often observed that the generalization error decreases as $n^{-β}$ where $n$ is the number of training examples and $β$ an exponent ...</li><li><a href="https://en.wikipedia.org/wiki/Empirical_statistical_laws">Empirical statistical laws - Wikipedia</a>: no description found</li><li><a href="https://www.alignmentforum.org/posts/GzoWcYibWYwJva8aL/parameter-counts-in-machine-learning">Parameter counts in Machine Learning — AI Alignment Forum</a>: In short: we have compiled information about the date of development and trainable parameter counts of n=139 machine learning systems between 1952 an…
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1256021537606074398)** (15 messages🔥): 

- **Introducing MMLU-SR Dataset to lm_eval**: A member introduced a new dataset, **MMLU-SR**, designed to challenge LLMs' reasoning abilities through symbol replacement and inquired about adding it to **lm_eval**. After creating and submitting a PR, they received a prompt response for review. [arxiv.org/abs/2406.15468v1](https://arxiv.org/abs/2406.15468v1)
  
- **MedConceptsQA Benchmark Addition**: A member requested a review for their PR that adds the **MedConceptsQA** benchmark aimed at medical concepts question answering. This open-source benchmark features questions of various complexities. [github.com/EleutherAI/lm-evaluation-harness/pull/2010](https://github.com/EleutherAI/lm-evaluation-harness/pull/2010)

- **Custom YAML Config Debugging**: A member sought help to run a custom YAML configuration for an evaluation using the harness. They received debugging advice and managed to resolve their issue after identifying and fixing a task name conflict.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/dancing-cat-dance-cat-cat-meme-chinese-cat-gif-6295992666678715767">Dancing Cat Dance GIF - Dancing cat Dance Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2010">Added MedConceptsQA Benchmark by Ofir408 · Pull Request #2010 · EleutherAI/lm-evaluation-harness</a>: Hi, I haved added our new benchmark called MedConceptsQA. MedConceptsQA is a dedicated open source benchmark for medical concepts question answering. The benchmark comprises of questions of various...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2030">Use `shell=False` in `subprocess` Function Calls by pixeeai · Pull Request #2030 · EleutherAI/lm-evaluation-harness</a>: This codemod sets the shell keyword argument to False in subprocess module function calls that have set it to True. Setting shell=True will execute the provided command through the system shell whi...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1256242980801740830)** (6 messages): 

- **Instruction Tuning in GPTNeoX**: A member inquired about the possibility of **instruction tuning in GPTNeoX**, specifically where losses are backpropagated only for continuations, not prompts. Another member suggested looking into a [specific PR](https://github.com/EleutherAI/gpt-neox/pull/1240) and the related preprocessing script—"[preprocess_data_with_chat_template.py](https://github.com/EleutherAI/gpt-neox/blob/main/tools/datasets/preprocess_data_with_chat_template.py)"—indicating that while it's still under review, bug reports would be helpful.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/pull/1240">SFT improvements (labeling fixes, different packing implementations) by dmahan93 · Pull Request #1240 · EleutherAI/gpt-neox</a>: add different packing impl (Unpacked, packing until overflow), a bit naive but for SFT it shouldn&#39;t be an issue fix labels to also have valid/test implementations fix label masking in _get_batch t...</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/tools/datasets/preprocess_data_with_chat_template.py">gpt-neox/tools/datasets/preprocess_data_with_chat_template.py at main · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1256224175442362448)** (12 messages🔥): 

- **Triton installation woes plague Windows users**: A user reported persistent issues with torch.compile, encountering a "RuntimeError: Cannot find a working triton installation," despite having Triton installed. Another member clarified that Triton might not be officially supported on Windows and suggested alternative installation methods.
- **Troubleshooting Triton with Anaconda**: A user mentioned installing PyTorch with Anaconda and running into Triton-related errors. Another user confirmed the unavailability of the Triton package via `conda install triton` on Windows and requested the output of `conda list` to troubleshoot further.
- **Seeking documentation on make_block_ptr**: A member inquired about detailed documentation for `triton.language.make_block_ptr`, expressing confusion over the available information.
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1256049551031144458)** (7 messages): 

- **Lovely Tensors breaks with torch.compile in 2.5.0**: The author of [Lovely Tensors](https://github.com/xl0/lovely-tensors) is encountering issues with custom `Tensor.__repr__()` breaking in `torch.compile()`. This is due to their `__repr__` being called on a FakeTensor, leading to problems; a workaround involves checking if the tensor is fake.

- **Community suggests using torch.compiler APIs**: There is a suggestion to use the [torch.compiler_fine_grain_apis](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html) to disable or handle custom `repr` functions. This approach could potentially unblock users facing similar issues.

- **Broadcast deadlock issue with NCCL**: The [broadcast deadlock issue](https://github.com/NVIDIA/nccl/issues/1251) in older NCCL versions has been a significant problem but is already fixed in newer versions not shipped with torch 2.3.1. Users can resolve it by installing `pip install nvidia-nccl-cu12==2.22.3`, as detailed in [this TGI PR](https://github.com/huggingface/text-generation-inference/pull/2099).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html">TorchDynamo APIs for fine-grained tracing &mdash; PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://github.com/xl0/lovely-tensors">GitHub - xl0/lovely-tensors: Tensors, for human consumption</a>: Tensors, for human consumption. Contribute to xl0/lovely-tensors development by creating an account on GitHub.</li><li><a href="https://github.com/NVIDIA/nccl/issues/1251">Leak in FIFO queue · Issue #1251 · NVIDIA/nccl</a>: We are experiencing an issue where 8 processes, each controlling one GPU on a node, all lock up at the same time. It seems to be deterministic, though we don&#39;t know exactly the operation that is c...</li><li><a href="https://github.com/huggingface/text-generation-inference/pull/2099">Fix nccl regression on PyTorch 2.3 upgrade by fxmarty · Pull Request #2099 · huggingface/text-generation-inference</a>: As per title, fixes NVIDIA/nccl#1251 in TGI&#39;s cuda image, regression introduced in #1730 &amp; #1833 We hit this issue e.g. with llama 3 70B model with TP=4 or TP=8 on H100 &amp; default cuda grap...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1256025541685940284)** (1 messages): 

- **Think about writing a CUDA Program**: An informative session by Stephen Jones on [how to think about writing a CUDA program](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62401/). Topics include *wave quantization & single-wave kernels,* types of parallelism, and tiling to optimize block sizes for L2 cache.

**Link mentioned**: <a href="https://www.nvidia.com/en-us/on-demand/session/gtc24-s62401/">How To Write A CUDA Program: The Ninja Edition | NVIDIA On-Demand</a>: Join one of CUDA's architects in a deep dive into how to map an application onto a massively parallel machine, covering a range of different techniques aim

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1256091717308776570)** (14 messages🔥): 

- **CUDA File Type Confusion? No Worries!**: A member asked if they should set the item type of all their files as CUDA C/C++ in a CUDA Runtime project in Visual Studio. Another suggested that it depends on personal preference as long as files that need CUDA are marked correctly: "I usually set them to .cu if cuda and .c/cpp if not... it's personal preference."
  
- **Cloud GPUs for Hands-on CUDA Experience**: A beginner inquired about using CUDA Toolkit on a cloud GPU without a local GPU and sought cost-friendly cloud vendors. Suggestions included [Vast.ai](https://vast.ai/) and [Runpod.io](https://www.runpod.io/), with a mention of Lightning AI offering free 22hrs/month of L4 usage.

- **Python to CUDA Optimization Flow**: For optimizing PyTorch code, a recommended flow was given: use `torch.compile`, consider custom Triton, and finally write custom CUDA code if needed. Key advice included checking for GPU bottlenecks and using efficient implementations like `F.spda()` for attention to ensure maximum utilization.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vast.ai/">Rent GPUs | Vast.ai</a>: Reduce your cloud compute costs by 3-5X with the best cloud GPU rentals. Vast.ai&#x27;s simple search interface allows fair comparison of GPU rentals from all providers.</li><li><a href="https://www.runpod.io/">RunPod - The Cloud Built for AI</a>: Develop, train, and scale AI models in one cloud. Spin up on-demand GPUs with GPU Cloud, scale ML inference with Serverless.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1256232766786240613)** (1 messages): 

- **Missing Pages in PMPP Book**: A member reported that their recently purchased PMPP (4th edition) **book was missing several pages**—specifically 148, 149, 150, 290, 291, 292, 447, and 448. They inquired if anyone else faced the same issue.
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1255968160587317412)** (16 messages🔥): 

```html
- **Custom static analysis tools discussion**: A user mentioned wanting to run custom static analysis tools on the project. This prompted excitement and agreement within the group.
- **Need for a list of required torch/aten ops**: One member suggested maintaining a list or table of required `torch/aten ops` for different tensor subclass use cases such as `FSDP`. For example, to swap linear weight, implementing `F.linear` and `aten.detach.default` is necessary.
- **Recursion error with `__torch_dispatch__`**: A user encountered a recursion error when printing arguments in `__torch_dispatch__`, leading to a discussion on possible causes and solutions. This included checking for special functions in `__repr__()` and using a debugger for inspection.
- **Int4Tensor refactor PR**: [A PR](https://github.com/pytorch/ao/pull/458) was created to refactor `Int4Tensor` and perform some code cleanup which will be completed over the weekend. 
- **NVIDIA GeForce GTX 1650 warning**: One user raised concerns about a warning for the NVIDIA GeForce GTX 1650 not supporting bfloat16 compilation natively. It was clarified that this could lead to performance implications like multiple kernel launches, which was linked to the usage of bfloat in quant API.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/26d633b7213c80371985ba88e6db4a2f796a2e50/torch/_inductor/compile_fx.py#L1647C5-L1647C31),">pytorch/torch/_inductor/compile_fx.py at 26d633b7213c80371985ba88e6db4a2f796a2e50 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/ao/pull/458">[WIP] Int4Tensor refactor to implements pattern by melvinebenezer · Pull Request #458 · pytorch/ao</a>: Refactoring UInt4Tensor to have implements pattern similar to nf4tensor  and UInt2Tensor ToDo   Create implements for UInt4Tensor and PerChannelSymmetricWeight  Test Cases  Move uint4i to uint4.py</li><li><a href="https://github.com/vayuda/ao/blob/intx/torchao/prototype/intx/intx.py#L365C4-L368C13">ao/torchao/prototype/intx/intx.py at intx · vayuda/ao</a>: Native PyTorch library for quantization and sparsity - vayuda/ao</li><li><a href="https://github.com/pytorch/ao/issues/391">[RFC] Tensor Subclass based Quantization API · Issue #391 · pytorch/ao</a>: Status: Draft Updated: 06/17/2024 Objective In this doc we’ll talk about Tensor subclass based quantization API for modeling users and developers. Modeling User API Modeling users refer to people w...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1256151743448875019)** (9 messages🔥): 

- **Model releases on HuggingFace Hub prioritize storage convenience**: A user noticed an increasing trend where model architecture and preprocessing code are directly stored on HuggingFace Hub instead of being added to the transformers repository. They speculated whether this is due to **code licensing issues** and shared examples [here](https://huggingface.co/microsoft/Florence-2-base-ft/tree/main) and [here](https://huggingface.co/Qwen/Qwen-Audio/tree/main).

- **Pros and Cons of HuggingFace Hub strategy**: Another user pointed out the benefits of this strategy, such as the ability for authors to release new models without needing an official HF team release, but also mentioned drawbacks like being unable to use these models for certain functionalities on the HF platform unless `trust_remote_code` is enabled.

- **Debate on optimal code and model storage solutions**: The discussion highlighted differing opinions about the best practices for storing model code and weights. One user suggested that releasing code on GitHub and storing model weights on **HuggingFace Hub** might be ideal, though others noted potential compatibility issues and the convenience factors of using the HF Hub.

- **Llama release as a case in point**: The discussion mentioned the **Llama model** release strategy, which involves maintaining the inference code on GitHub independent of the transformers library. An example repository for this approach is [meta-llama on GitHub](https://github.com/meta-llama/llama).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/meta-llama/llama">GitHub - meta-llama/llama: Inference code for Llama models</a>: Inference code for Llama models. Contribute to meta-llama/llama development by creating an account on GitHub.</li><li><a href="https://huggingface.co/microsoft/Florence-2-base-ft/tree/main">microsoft/Florence-2-base-ft at main</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen-Audio/tree/main">Qwen/Qwen-Audio at main</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct/tree/main">deepseek-ai/DeepSeek-Coder-V2-Instruct at main</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1255975840458477700)** (68 messages🔥🔥): 

- **Google Gemma 2 shines in benchmarks**: Google's new Gemma 2 models (27B and 9B) outperformed Llama3 70B and Qwen 72B in the LYMSYS Chat arena. The [details of the release](https://x.com/reach_vb/status/1806343018640781675) appreciate the open sharing of experiments and the planned 2.6B model release soon.
- **Danielhanchen analyzes Gemma 2 architecture**: Daniele Hanchen highlighted key elements of the Gemma 2 models, including pre and post Layernorms and [approx GeGLU activations](https://x.com/danielhanchen/status/1806372357684220308). The models use a sliding window and global attention layers for efficient processing.
- **ReLU vs GELU debate**: Discussion on whether ReLU is better than GELU for activation functions, referencing an [arxiv paper](https://arxiv.org/pdf/2002.05202v1), testing results, and hardware benefits. The debate included the importance of accurate hyperparameters for ReLU.
- **FP8 challenges with hardware and libraries**: Challenges of FP8 support in current hardware and libraries, with a focus on NVIDIA's NCCL and cuDNN limitations. A detailed discussion on alternatives and potential workarounds ensued, including references to [Microsoft's FP8-LM paper](https://arxiv.org/pdf/2310.18313).
- **Training insights and optimizations**: Yuchen's training with H100 GPUs showed promising results with higher learning rates, suggesting that issues faced could be platform or dataset specific. Discussion about various optimizers and the specifics of their implementations followed, indicating the complexity and sensitivity of the training processes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/reach_vb/status/1806343018640781675">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Let&#39;s fucking gooo! Google just dropped Gemma 2 27B & 9B 🔥  &gt; Beats Llama3 70B/ Qwen 72B/ Command R+ in LYMSYS Chat arena & 9B is the best &lt; 15B model right now. &gt; 2.5x smaller than Llam...</li><li><a href="https://x.com/danielhanchen/status/1806372357684220308">Tweet from Daniel Han (@danielhanchen)</a>: Just analyzed Google&#39;s new Gemma 2 release! The base and instruct for 9B & 27B is here!  1. Pre & Post Layernorms = x2 more LNs like Grok 2. Uses Grok softcapping! Attn logits truncated to (-30, 3...</li><li><a href="https://x.com/Yuchenj_UW/status/1806713556047716603">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: Outperform GPT-3 (1.5B) with @karpathy&#39;s llm.c using just 1/5 training tokens 🌠  Previously, I trained the GPT-2 Small (124M) model. Recently, I trained GPT-2 XL (1.5B), aka the official GPT-2, u...</li><li><a href="https://github.com/clu0/unet.cu">GitHub - clu0/unet.cu: UNet diffusion model in pure CUDA</a>: UNet diffusion model in pure CUDA. Contribute to clu0/unet.cu development by creating an account on GitHub.</li><li><a href="https://github.com/Azure/MS-AMP">GitHub - Azure/MS-AMP: Microsoft Automatic Mixed Precision Library</a>: Microsoft Automatic Mixed Precision Library. Contribute to Azure/MS-AMP development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1256052708700454932)** (1 messages): 

- **Reduced Pricing for Perplexity Enterprise Pro for Philanthropic Organizations**: Perplexity now offers reduced pricing for Perplexity Enterprise Pro to schools, nonprofits, government agencies, and not-for-profits. The initiative aims to support organizations facing budget constraints while playing a vital role in societal and educational development. [Learn more](https://pplx.ai/s3oZX49).
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1255963603161583636)** (94 messages🔥🔥): 

- **Perplexity's RAG Performance Discussed**: Members discussed how Perplexity's Relevance Aware Generation (RAG) mechanism sometimes leads to poor outputs, especially when it tries to incorporate files inconsistently. It was noted that **writing mode aims to avoid RAG**, but actual results still often exhibit hallucinations.
  
- **Claude 3 Opus Usage Limits Frustrate Users**: The daily usage limit for **Claude 3 Opus** has been a source of ongoing frustration, fluctuating from **5 to 600 and now capped at 50 interactions per day**. One user described this limit change as a "roller coaster ride."
  
- **Security and Data Concerns Addressed**: A member asked about the **security measures** and **PII handling** for Perplexity's enterprise solution. The response directed them to the [Trust Center](https://trust.perplexity.ai/) and provided an email for further inquiries.
  
- **Intermittent Context Issues with Perplexity**: Users noted that Perplexity tends to lose context during extended interactions. One user suggested using **keywords like "#context"** to improve continuity until a fix is implemented.
  
- **VPN and Access Issues**: A few members reported issues with Perplexity not working over **VPNs** like **Cloudflare WARP**, causing connectivity and login problems. There were recommendations to switch to DNS-only mode as a workaround.

**Link mentioned**: <a href="https://trust.perplexity.ai/.">Trust Center</a>: Showcasing our security posture to build trust across the web.

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1255978834453467186)** (8 messages🔥): 

- **Android 14 insights explored**: A link to a [Perplexity AI page](https://www.perplexity.ai/page/Android-14-boosts-WxN8GGdgRQSKPPn7DftxtQ) detailing features and enhancements introduced in Android 14 was shared. This page likely discusses the specifics of the operating system update and its impact on user experience.

- **Question about RDP on Perplexity AI**: The link provided offers an in-depth look at [Microsoft Remote Desktop Protocol (RDP)](https://www.perplexity.ai/page/Why-Microsoft-STILL-ZHlLAmi2Riyx3S0.Li.kwg). It discusses ongoing considerations and potential improvements in RDP usage within Microsoft's ecosystem.

- **CriticGPT, Living Robot Skin, and sustainable innovations**: A YouTube video was shared with a title indicating a discussion on [CriticGPT, Living Robot Skin, and Oyster-Inspired Concrete](https://www.youtube.com/embed/Kqw56PUMa6M). The video seems to cover cutting-edge technologies and sustainable materials inspired by natural solutions.

- **Linux performance exploration**: A link to a [Perplexity AI search](https://www.perplexity.ai/search/why-Linux-are-E54_z89aSlKnHo6nVWPaIg) dives into reasons behind Linux performance and adoption issues. This page likely explores common challenges and solutions for Linux users.

- **Misleading Minecraft mechanics**: An article titled [Minecraft Repair Mechanics Misleads Kids](https://www.perplexity.ai/page/Minecraft-Repair-Mechanics-NdRggXKXRXyGY8LgKsp1dQ) was shared. It raises concerns about the potential misconceptions in mechanical knowledge children might develop by playing the game.

**Link mentioned**: <a href="https://www.youtube.com/embed/Kqw56PUMa6M">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1256005005878497360)** (13 messages🔥): 

- **Community wants Gemma 2 model**: "I know this goes without saying, but it'd be cool to have Gemma 2 in the available models."
- **Question on supporting larger models and pro searches**: A member asked, "when will Perplexity API start supporting pro searches and bigger models like GPT-4 and Sonnet?" Another member responded that GPT-4 and Sonnet can be used via respective providers and Pro Search for the API is currently not planned.
- **Clarification on Perplexity's added value**: "The whole point of Perplexity is that it will add more online search and better prompts to talk to GPT-4 or Sonnet to give a better experience," noted a member. Current models like `llama-3-sonar-large-32k-online` are available with specific parameters that can be found in the [Perplexity model cards](https://docs.perplexity.ai/docs/model-cards) documentation.
- **Critique of current API performance**: A member expressed dissatisfaction, stating, "I have tried them but they are not as good as GPT-4 or Sonnet 3.5 to comprehend" and noted "tons of hallucinations" in the current API.
- **Inquiring about filtering results**: Another member inquired about limiting results to new information from the last 30 days, receiving advice to try `after:2024-05-28`.

**Link mentioned**: <a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1255961250786181241)** (40 messages🔥): 

- **Character.AI launches Character Calls**: Character.AI introduced [Character Calls](https://blog.character.ai/introducing-character-calls/), a feature for two-way voice conversations with AI characters, available for free on their app. However, user feedback highlighted issues like a 5-second delay and robotic voices, affecting the fluidity of conversations.
- **Amazon acquires Adept's talents and tech**: Discussions centered around [Amazon hiring Adept's cofounders and licensing its technology](https://x.com/anissagardizy8/status/1806812006009442671?s=46), leaving Adept with around 20 employees. There were rumors about a toxic culture at Adept leading to the departure of the Transformer paper authors who initially founded the company.
- **Skepticism around the progress of AI agents**: Comparisons were made between the hype around AI agents and self-driving cars, suggesting that agents are "always just around the corner but never working reliably enough." The conversation noted that despite significant talent and investment, the development of useful AI agents is slow compared to other AI advancements like video generation.
- **Challenges in training data for AI agents**: Participants discussed that a likely bottleneck in developing AI agents is the collection and quality of training data. The focus is shifting towards generating synthetic data and obtaining annotated counterfactual examples to improve agent reliability and performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.adept.ai/blog/adept-update">An Update to Adept</a>: Announcing some updates to our strategy and the company.</li><li><a href="https://blog.character.ai/introducing-character-calls/">Introducing Character Calls</a>: 0:00  /0:07   1×                  Calling the Character.AI Community!  We&#x27;re thrilled to ring in an exciting new feature that&#x27;s set to redefine your Character.AI experience: Character Calls!...</li><li><a href="https://x.com/giffmana/status/1806411302190915603?s=46">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: @fouriergalois @character_ai just tried it, it&#39;s not comparable unfortunately, would have been super impressive!  It&#39;s not fluid at all. 5sec delay when I&#39;m done talking. I can&#39;t inter...</li><li><a href="https://x.com/anissagardizy8/status/1806812006009442671?s=46">Tweet from Anissa Gardizy (@anissagardizy8)</a>: Amazon has hired the cofounders of artificial intelligence startup Adept and licensed some of its tech, according to a post by the startup and an internal email from an Amazon exec  Adept is left w/ a...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1255991140621226085)** (7 messages): 

- **Anthropic CEO nostalgically discusses Final Fantasy**: A member shared a [YouTube video of Dario Amodei](https://youtu.be/xm6jNMSFT7g?si=BnYoL-E1QXGTw23P&t=3880), CEO of Anthropic, discussing how he and his sister played Final Fantasy growing up and continue to do so as adults. The member found this anecdote pretty endearing.

- **AI crisis vs. pandemic debate ignites**: Natolambert labeled Dwarkesh's comparison of an AI crisis being harder than a pandemic as presumptive. Dwarkesh also controversially said, “we’ve done vaccines before” to imply that COVID vaccines were normal.

- **Political instability sparks extreme hopes**: Natolambert expressed a wish for an unstable intelligence explosion to render government meaningless if Trump becomes president. The sentiment indicates a desire for significant change driven by AI developments.
  
- **European member disheartened by debate**: Xeophon expressed feeling bad about the debate, mentioning his European perspective. This hints at a more global impact of the debate on AI and political issues.

**Link mentioned**: <a href="https://youtu.be/xm6jNMSFT7g?si=BnYoL-E1QXGTw23P&t=3880">Dario Amodei - CEO of Anthropic | Podcast | In Good Company | Norges Bank Investment Management</a>: Dario Amodei CEO of Anthropic: Claude, New models, AI safety and Economic impactHow much bigger and more powerful will the next AI models be? Anthropic’s CEO...

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1255980323674656918)** (5 messages): 

- **Memes: Bourne Supremacy Reference**: A member shared a [YouTube video](https://youtu.be/I3znSbbu9IU?si=EbbsoUgHAFS1wuMY&t=65) titled "The Bourne Supremacy (9/9) Movie CLIP - Final Call to Pamela (2004) HD". The video is a movie clip from The Bourne Supremacy.

**Link mentioned**: <a href="https://youtu.be/I3znSbbu9IU?si=EbbsoUgHAFS1wuMY&t=65">The Bourne Supremacy (9/9) Movie CLIP - Final Call to Pamela (2004) HD</a>: The Bourne Supremacy movie clips: http://j.mp/1uvIXs9BUY THE MOVIE: http://amzn.to/tor8HhDon&#39;t miss the HOTTEST NEW TRAILERS: http://bit.ly/1u2y6prCLIP DESCR...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1256212211114967060)** (3 messages): 

- **Debunking AI Scaling Myths**: The article [AI Scaling Myths](https://www.aisnakeoil.com/p/ai-scaling-myths) challenges the predictability of scaling, arguing there's virtually no chance scaling alone will lead to AGI. It suggests LLM developers are near the limit of high-quality data and highlights downward pressure on model size despite the predictability shown in [scaling laws](https://arxiv.org/abs/2001.08361).

- **Discussion on AGI Definitions and Synthetic Data**: Nathan Lambert critiqued the article for not defining AGI and ignoring synthetic data, suggesting [this research](https://arxiv.org/abs/2401.16380v1) on rewriting pre-training data. Lambert also mentioned that the claim about the industry stopping large models is short-term and linked to capital expenditures, encouraging further discussion on [Substack](https://open.substack.com/pub/aisnakeoil/p/ai-scaling-myths?r=68gy5&utm_campaign=comment-list-share-cta&utm_medium=web&comments=true&commentId=60317135).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://open.substack.com/pub/aisnakeoil/p/ai-scaling-myths?r=68gy5&utm_campaign=comment-list-share-cta&utm_medium=web&comments=true&commentId=60317135">Nathan Lambert on AI Snake Oil</a>: I&#x27;m a fan, but I feel like this fell into a few of the same traps as the AGI Faithful, but from the other side: 1. Easy to do this without definitions. You did not define AGI or comment on how mu...</li><li><a href="https://www.aisnakeoil.com/p/ai-scaling-myths">AI scaling myths</a>: Scaling will run out. The question is when.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1255994147794325604)** (19 messages🔥): 

```html
<ul>
    <li><strong>SnailBot News Episode Talks</strong>: Members expressed excitement about the latest SnailBot News episode featuring a discussion around Lina Khan (Chairperson FTC) on Hard Fork [TikTok link](https://www.tiktok.com/@hardfork/video/7301774206440656171?lang=en). Natolambert mentioned plans for future interviews including Ross Taylor of Paperswithcode/Galactica and John Schulman.</li>
    <li><strong>Model Output Training Limitations</strong>: A user highlighted the interesting point on "Please don't train on our model outputs" stipulations being required by data providers. Natolambert confirmed that some models would drop the limitation if not required by data providers, citing DBRX folks.</li>
    <li><strong>Potential Interviewees Discussed</strong>: Natolambert revealed potential guests for future episodes including Amanda Askell, with one member expressing enthusiasm for her insights from past appearances. Xeophon mentioned Ross Taylor's elusive yet significant insights, stirring interest among the group.</li>
    <li><strong>Nicknames and Influence in Labs</strong>: 420gunna humorously noted the nickname "DBRex," to which Natolambert took credit. This was followed by a light-hearted comment on Natolambert's influence within labs.</li>
    <li><strong>Pre-deployment Testing and Influencing AI Labs</strong>: The conversation touched on pre-deployment testing issues and the contrasting influence on AI labs versus government figures. One member found the idea of influencing AI labs less realistic compared to government figures.</li>
</ul>
```

**Link mentioned**: <a href="https://www.tiktok.com/@hardfork/video/7301774206440656171?lang=en">TikTok - Make Your Day</a>: no description found

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1256283272808366131)** (2 messages): 

- **Build agentic RAG services with llama-agents**: A notebook demonstrates creating vector indexes, turning them into query engines, and providing these tools to agents before launching them as services. For detailed steps, check this [notebook](https://t.co/WYTCaqs6Yb).
- **Jina releases their best reranker yet**: LlamaIndex users are enthusiastic about the new Jina reranker, described as their best one to date. More details can be found [here](https://t.co/YsYoVOIirb).

**Link mentioned**: <a href="https://t.co/WYTCaqs6Yb">llama-agents/examples/agentic_rag_toolservice.ipynb at main · run-llama/llama-agents</a>: Contribute to run-llama/llama-agents development by creating an account on GitHub.

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1255982061328535603)** (68 messages🔥🔥): 

- **Embedding Node Weights and Issues with Vector Retrievers**: Multiple members discuss embedding issues with **LlamaIndex**, focusing on what parts of nodes are embedded and problems where **vector retrievers** yield poor results possibly due to incorrectly matched embedding models. One member suggests, "the best way to start is creating a simple test case that reproduces some unexpected results” to debug effectively.
- **Entity Relationship Linking**: Members debate about adding edges based on embedding conditions to better capture entity relationships, which aren't detected traditionally. They mention a potential **collaboration article by Neo4J and LlamaIndex** on entity resolution that may help.
- **Claude's Empty Responses**: There's a technical discussion about **Claude's response handling via Bedrock** leading to empty responses if max tokens are set too low. An edge case leads to an *IndexError*, prompting a member to share a temporary fix and promising to clean up and share the validating notebook.
- **Excitement Around New Releases**: Enthusiasm is expressed about the new **Gemma2 model** and the latest **announcement on the agents framework**. A link to the [Gemma2 model on Hugging Face](https://huggingface.co/bartowski/gemma-2-9b-it-GGUF) is shared, with members troubleshooting integration issues.
- **Challenges with OpenAI Key Environment Variables**: A user reports unexpected behavior where **OpenAI keys** are sought from environment variables despite being set in the code itself. Additionally, optimization queries arise regarding **batch and parallel loading of indices** to handle large file sizes faster.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llamahub.ai/l/tools/llama-index-tools-openapi">no title found</a>: no description found</li><li><a href="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF">bartowski/gemma-2-9b-it-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1256295002452135956)** (1 messages): 

- **Gemma 2.9B launches new models**: *Free & standard variants* for the new [google/gemma-2-9b-it](https://openrouter.ai/models/google/gemma-2-9b-it) are now available. OpenRouter announced this update for 2023-2024.

- **Price cuts announced**: Several popular models have received price reductions. Notable drops include [cognitivecomputations/dolphin-mixtral-8x22b](https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b) with a 10% cut, [openchat/openchat-8b](https://openrouter.ai/models/openchat/openchat-8b) with a 20% reduction, and [meta-llama/llama-3-70b-instruct](https://openrouter.ai/models/meta-llama/llama-3-70b-instruct) with a 3.5% drop.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemma-2-9b-it>)">Google: Gemma 2 9B by google</a>: Gemma 2 9B by Google is an advanced, open-source language model that sets a new standard for efficiency and performance in its size class.  Designed for a wide variety of tasks, it empowers developers...</li><li><a href="https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b>)">Dolphin 2.9.2 Mixtral 8x22B 🐬 by cognitivecomputations</a>: Dolphin 2.9 is designed for instruction following, conversational, and coding. This model is a finetune of [Mixtral 8x22B Instruct](/models/mistralai/mixtral-8x22b-instruct). It features a 64k context...</li><li><a href="https://openrouter.ai/models/openchat/openchat-8b>)">OpenChat 3.6 8B by openchat</a>: OpenChat 8B is a library of open-source language models, fine-tuned with &quot;C-RLFT (Conditioned Reinforcement Learning Fine-Tuning)&quot; - a strategy inspired by offline reinforcement learning. It...</li><li><a href="https://openrouter.ai/models/gryphe/mythomax-l2-13b>)">MythoMax 13B by gryphe</a>: One of the highest performing and most popular fine-tunes of Llama 2 13B, with rich descriptions and roleplay. #merge</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-70b-instruct>)">Meta: Llama 3 70B (Base) by meta-llama</a>: Meta&#x27;s latest class of model (Llama 3) launched with a variety of sizes &amp; flavors. This is the base 70B pre-trained version.  It has demonstrated strong performance compared to leading closed...</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-72b-instruct>)">Qwen 2 72B Instruct by qwen</a>: Qwen2 72B is a transformer-based model that excels in language understanding, multilingual capabilities, coding, mathematics, and reasoning.  It features SwiGLU activation, attention QKV bias, and gro...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1256045942189330462)** (57 messages🔥🔥): 

- **OpenRouter Moderation Strictness Discussed**: Members compared OpenRouter's self-moderation to AWS and Anthropic, suggesting it is more censored. One user mentioned, "Both will refuse without a prefill but start writing with a basic prefill."

- **Issues with Opus Availability**: A user noted that enabling Opus is currently unavailable without enterprise support. They linked to a [Reddit post](https://www.reddit.com/r/aws/comments/1cy1hce/claude_opus_shows_as_unavailable_in_us_west_2/l68rawl/) discussing this limitation.

- **Troubleshooting GitHub Authentication**: Members shared solutions for making GitHub pushes without repeatedly entering a passphrase, recommending tools like `ssh-add -A` and adding commands to `~/.bash_profile`. One detailed guide was linked in a [SuperUser post](https://superuser.com/a/1158050).

- **API Differences and Issues**: Discussions revealed API discrepancies, particularly with Gemini models producing a "Status 400" error. It's highlighted that Google APIs do not follow standard formatting, with specific adjustments required for tool roles.

- **Evaluating LLM APIs**: A member suggested watching Simon Willison's talk for an overview of LLM APIs, sharing a [YouTube link](https://www.youtube.com/watch?v=5zE2sMka620&t=2026s) and a link to his [blog post](https://simonwillison.net/2024/Jun/27/ai-worlds-fair/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/provider-routing#custom-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://www.reddit.com/r/aws/comments/1cy1hce/claude_opus_shows_as_unavailable_in_us_west_2/l68rawl/>">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620&t=2026s">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>: https://twitter.com/aidotengineer</li><li><a href="https://forum.cursor.com/">Cursor Community Forum</a>: A place to discuss Cursor (bugs, feedback, ideas, etc.)</li><li><a href="https://superuser.com/a/1158050">macOS keeps asking my ssh passphrase since I updated to Sierra</a>: It used to remember the passphrase, but now it&#x27;s asking it to me each time.&#xA;&#xA;I&#x27;ve read that I need to regenerate the public key with this command, which I did:&#xA;&#xA;ssh-keygen -y...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 messages): 

voidnewbie: Gemma 2가 명목상으로는 영어만 지원하지만 뛰어난 다국어 능력을 가지고 있는 것 같아요. 한국어를 시험해보신 분 계신가요?
  

---


### **OpenRouter (Alex Atallah) ▷ #[tips](https://discord.com/channels/1091220969173028894/1256159780628861001/1256160868203499551)** (1 messages): 

- **Set default model wisely**: daun.ai suggests setting your default model to 'auto' for reliable output on most tasks. Alternatively, use 'flavor of the week' for more serendipitous results, which will be the fallback model if no specific model is chosen and a request fails.
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1255964886056308849)** (36 messages🔥): 

- **Gege AI threatens the music industry**: A member shared a [Reddit link](https://www.reddit.com/r/singularity/comments/1dpxocg/gege_ai_a_new_music_maker_ai_that_allows_you_to/) about **Gege AI**, an AI that can clone any singer's voice with a small sample. They humorously commented, *"RIP music industry"* and suggested the **RIAA** should sue China.

- **Challenges with Gege AI registration**: Users discussed facing issues while registering for **Gege AI**. One joked about it being related to *"Not enough social credit points"*.

- **Gemma 27B model impresses and causes skepticism**: A member claimed that **Gemma 27B** is performing well, but others expressed skepticism about its true capabilities. They noted its performance still seemed better than its predecessor despite the high confidence interval.

- **Complaints about GPT-4 and 4O models**: Multiple users mentioned problems with **GPT-4 and 4O models**, noting they often take prompts too literally and are less effective for programming compared to **GPT-3.5**. One stated, *"Free alternative reign supreme"* comparing it with **Gemini 1.5 Pro**.

- **Switching to Claude for better experience**: Some users have switched from OpenAI's models to **Claude** due to a better artifacts feature and functionality with **Hugging Face libraries**. They reported improved experiences over **GPT-4** models.

**Link mentioned**: <a href="https://www.reddit.com/r/singularity/comments/1dpxocg/gege_ai_a_new_music_maker_ai_that_allows_you_to/">Reddit - Dive into anything</a>: no description found

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1256017421429510295)** (2 messages): 

- **Adam-mini optimizer slashes memory usage without sacrificing performance**: An exciting new optimizer called **Adam-mini** can achieve similar or better performance than **AdamW** with 45% to 50% less memory footprint. The paper argues that most of the learning rate resources in Adam (specifically $1/\\sqrt{v}$) can be removed by partitioning parameters into blocks and assigning a single, optimized learning rate per block, ultimately outperforming Adam in some cases.

- **Single learning rates for weight tensors eliminate excess**: The innovative approach of using one pre-searched learning rate per weight tensor shows significant performance improvements over Adam. "One pre-searched learning rate per weight tensor outperforms Adam significantly," highlighting how careful resource allocation and optimization can enhance efficiency.

**Link mentioned**: <a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: We propose Adam-mini, an optimizer that achieves on-par or better performance than AdamW with 45% to 50% less memory footprint. Adam-mini reduces memory by cutting down the learning rate resources in ...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1256002857438810223)** (26 messages🔥): 

- **CSV and Pandas DataFrame Agents with Bedrock Issues**: A member is experiencing issues building and running a **csv_agent** or **pandas_dataframe_agent** with **Bedrock**. They sought help from the community for troubleshooting.

- **Errors with Sonnet 3.5 and Bedrock**: Another member is having trouble integrating the **Sonnet 3.5 model** with **Bedrock** using `ChatPromptTemplate.fromMessages`. They shared an example format and mentioned receiving errors despite attempts to adjust message formats.

- **LangGraph and Human-in-the-Loop Launch**: [LangChain Blog](https://blog.langchain.dev/human-in-the-loop-with-opengpts-and-langgraph) announced the launch of **LangGraph** featuring human-in-the-loop capabilities via "Interrupt" and "Authorize" functions. The discussion highlighted issues with deserialization errors when attempting to resume execution after human approval steps.

- **Discussion on CSV File Handling**: A user discussed using **LangChain's CSV Loader** and expressed difficulty handling multiple CSV files effectively. They shared a [documentation link](https://python.langchain.com/v0.2/docs/how_to/document_loader_csv/) and sought community input on better approaches.

- **Python Example for Human-in-the-Loop**: A detailed example and link to a [guide for implementing human-in-the-loop in Python](https://python.langchain.com/v0.2/docs/how_to/tools_human/#adding-human-approval) were shared. This included a mechanism for asking human approval in tool invocation steps and handling tool call acceptance or rejection.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.langchain.dev/human-in-the-loop-with-opengpts-and-langgraph/?">Human-in-the-loop with OpenGPTs and LangGraph</a>: TLDR; Today we’re launching two “human in the loop” features in OpenGPTs, Interrupt and Authorize, both powered by LangGraph.  We&#x27;ve recently launched LangGraph, a library to help developers buil...</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/tools_human/#adding-human-approval>).">How to add a human-in-the-loop for tools | 🦜️🔗 LangChain</a>: There are certain tools that we don&#x27;t trust a model to execute on its own. One thing we can do in such situations is require human approval before the tool is invoked.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/document_loader_csv/">How to load CSVs | 🦜️🔗 LangChain</a>: A comma-separated values (CSV) file is a delimited text file that uses a comma to separate values. Each line of the file is a data record. Each record consists of one or more fields, separated by comm...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1255977451998416998)** (8 messages🔥): 

- **No Code Chrome Extension for LangChain**: A member shared a [YouTube video](https://www.youtube.com/watch?v=-OKC7CY2bbQ) titled *"No Code Chrome Extension Chat Bot Using Visual LangChain."* The video demonstrates how to design a LangChain RAG application with an interactive chat feature.

- **Dappier Launches AI Content Marketplace**: A new platform, [Dappier](https://www.producthunt.com/posts/dappier-2-0), aims to monetize proprietary content for AI use and training. Featured in a [TechCrunch article](https://techcrunch.com/2024/06/26/dappier-is-building-a-marketplace-for-publishers-to-sell-their-content-to-llm-builders/), the platform allows creators to license data models via a RAG API.

- **Data Analyst Agent Using Cohere and LangChain**: A member built a [Data Analyst Agent](https://www.linkedin.com/posts/eddieotudor_datascience-aitechnology-machinelearning-activity-7212491482287542272-1cSF) leveraging Cohere and LangChain, and shared the project on LinkedIn.

- **Testcontainers Adds Ollama Support**: A new PR for [testcontainers-python](https://github.com/testcontainers/testcontainers-python/pull/618) was accepted, adding support for the Ollama module. Users are encouraged to try features released in version 4.7.0.

- **Tool for Editing JSONL Datasets**: A free tool for editing fine-tune and chat datasets in JSONL format was shared: [uncensored.com/jsonl](https://uncensored.com/jsonl). The creator emphasized the hassle of manually editing JSONL datasets.

- **Building RAG with Matryoshka Embeddings**: A member shared details about [building RAG](https://x.com/Prashant_Dixit0/status/1806580075447590974) with Matryoshka Embeddings and Llama Index. Advantages include improved retrieval speed and reduced memory footprint, with a [Colab tutorial](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/tutorials/RAG-with_MatryoshkaEmbed-Llamaindex/RAG_with_MatryoshkaEmbedding_and_Llamaindex.ipynb) provided.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=-OKC7CY2bbQ">No Code Chrome Extension Chat Bot Using Visual LangChain</a>: In this demo, I show an exciting new feature of Visual Agents where you can design your LangChain RAG application, including an interactive chat feature to a...</li><li><a href="https://x.com/Prashant_Dixit0/status/1806580075447590974">Tweet from Prashant Dixit (@Prashant_Dixit0)</a>: Build Matryoshka RAG with @llama_index  These embedding models produce a range of embedding dims(768, 512, 256, 128, and 64).  🌟 Advantages  ✅ Boosting retrieval Speed performance   ✅ Reducing memory...</li><li><a href="https://uncensored.com/jsonl">Chat Uncensored AI | Rated 2024&apos;s Best Uncensored AI</a>: What&apos;s New: Now with Uncensored Images! The latest &amp; most advanced Uncensored AI (2024). No log in required, 100% private, Turbo Speed. Trusted by 10,000+ users worldwide.</li><li><a href="https://www.producthunt.com/posts/dappier-2-0"> Dappier 2.0 - Combat AI data scraping &amp; get paid for your content fairly | Product Hunt</a>: Dappier is the world’s first online marketplace for AI content and data rights. Get paid fairly as your licensed content is accessed by AI companies around the world.</li><li><a href="https://github.com/testcontainers/testcontainers-python/pull/618">fix: Add support for ollama module by bricefotzo · Pull Request #618 · testcontainers/testcontainers-python</a>: Added a new class OllamaContainer with few methods to handle the Ollama container.   The _check_and_add_gpu_capabilities method checks if the host has GPUs and adds the necessary capabilities to th...</li><li><a href="https://github.com/testcontainers/testcontainers-python/issues/617#issuecomment-2194351846">New Container: OllamaContainer · Issue #617 · testcontainers/testcontainers-python</a>: Add support for the OllamaContainer to simplify running and testing LLMs through Ollama. What is the new container you&#39;d like to have? I would like to request support for a new container: OllamaCo...</li><li><a href="https://techcrunch.com/2024/06/26/dappier-is-building-a-marketplace-for-publishers-to-sell-their-content-to-llm-builders/">Dappier is building a marketplace for publishers to sell their content to LLM builders | TechCrunch</a>: Dappier, an early stage startup, is building a marketplace where publishers and data owners can sell access to their content to LLM builders.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1256296697475371121)** (1 messages): 

- **Testcontainers Python SDK adds Ollama support**: A user announced that their [pull request](https://github.com/testcontainers/testcontainers-python/pull/618) for adding support for Ollama in the Testcontainers Python SDK has been accepted and released in version 4.7.0. They included an [example](https://github.com/testcontainers/testcontainers-python/issues/617#issuecomment-2194351846) to help others get started quickly.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/testcontainers/testcontainers-python/pull/618">fix: Add support for ollama module by bricefotzo · Pull Request #618 · testcontainers/testcontainers-python</a>: Added a new class OllamaContainer with few methods to handle the Ollama container.   The _check_and_add_gpu_capabilities method checks if the host has GPUs and adds the necessary capabilities to th...</li><li><a href="https://github.com/testcontainers/testcontainers-python/issues/617#issuecomment-2194351846">New Container: OllamaContainer · Issue #617 · testcontainers/testcontainers-python</a>: Add support for the OllamaContainer to simplify running and testing LLMs through Ollama. What is the new container you&#39;d like to have? I would like to request support for a new container: OllamaCo...
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1255984799752126574)** (2 messages): 

- **Next Mojo Community Meeting Scheduled**: The next Mojo Community meeting will take place on \[local time\]. For details, attendees can join the meeting via [Zoom](https://modul.ar/community-meeting-zoom) and access the agenda on [Google Docs](https://modul.ar/community-meeting-doc). 
- **Holiday Wishes**: Happy holidays to those in Canada and those taking time off during the July 4 week in the U.S.!
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modul.ar/community-meeting-zoom.">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://modul.ar/community-meeting-doc.">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1806718451089817703>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1256053477873025024)** (11 messages🔥): 

- **Confusion over Mojolicious and Mojo**: A user requested a code example in Mojo, and there was confusion when ModularBot provided a Perl-based Mojolicious example. Another member clarified that the inquiry was specifically about Mojo, the AI development language created by Modular in 2022. 
- **Clarification of Mojo**: After further prodding, ModularBot acknowledged the mistake and discussed Mojo’s capabilities, comparing it to *"a knight venturing into uncharted territories"* with its enhanced abilities similar to Python but robust like C.

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1255961095089291336)** (12 messages🔥): 

- **Issues with Mojo SDK and Telemetry**: Members discuss the Mojo SDK's telemetry collection, with one noting it can be disabled and sharing a [link to the FAQ](https://docs.modular.com/mojo/faq#does-the-mojo-sdk-collect-telemetry) for more information. Another thanks them for the useful info.
- **Connection Issue in Mojo REPL**: A member observes that running REPL opens a connection without showing network traffic, which later closes unexpectedly. They confirm the issue and suggest opening a GitHub issue for further investigation.
- **Discussion on Mojo Package Listing**: A member runs a command to list Mojo packages, revealing the package details of mojo version 24.4.0. This spurs a conversation on their configuration and setup. 
- **Mojo Language Design Choices**: A member notes the Io module's lack of functionality, requiring interfacing with Python to read from stdin. They ponder whether this is deliberate or if contributions to expand it would be accepted by Modular. 



**Link mentioned**: <a href="https://docs.modular.com/mojo/faq#does-the-mojo-sdk-collect-telemetry">Mojo🔥 FAQ | Modular Docs</a>: Answers to questions we expect about Mojo.

  

---


### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - Issue 38
https://www.modular.com/newsletters/modverse-weekly-38
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1255983829693497394)** (4 messages): 

- **Graph API improvements missed in changelog**: The latest release includes *"integer literal slices and slicing across all dimensions"* with some semantic restrictions. The team highlighted the importance of filing issues for tracking and addressing external interest in new features.
- **Temporary solution for unsqueeze operation**: For those needing to *"unsqueeze"*, using `ops.unsqueeze(x, axis=-1)` is suggested as a workaround.
- **MAX nightly releases back online**: The MAX nightly releases are functional again, and users are encouraged to demo the *Llama3 GUI Chatbot* via the provided [Discord link](https://discord.com/channels/1087530497313357884/1256010477637730315/1256010477637730315) and share feedback.
- **New Mojo nightly compiler released**: A new nightly compiler version `2024.6.2805` has been released, with notable updates including changes in LSP behavior. Users are instructed to update using `modular update nightly/mojo` and can check the [raw diff](https://github.com/modularml/mojo/compare/ddaa1d0a0979998a96084e3d7bd335fbcda3e8cb...439d86d608d3b6c12cead112eb651752ba1ad40d) and current [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1255987610577403954)** (30 messages🔥): 

- **Community contributions to the model welcomed**: *"This is on our radar but we also welcome any community contribution of the model if someone wants to try to use the model right away."*
- **Weird behavior in text completions linked to EOS tokens**: A user identified *"super weird behavior"* in continuations caused by *"dataset adding eos tokens when encoding"* as noted in this [GitHub link](https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_text_completion.py#L56).
- **PreferenceDataset recommended for PPO implementation**: When discussing dataset configurations, a user recommended using **PreferenceDataset** for RL where *"reward model looks at how 'preferred' the whole input+response is"*. This contrasts with **text completion dataset** used for continued pretraining of single text bodies.
- **Confusion cleared up around pretraining examples**: Discussions clarified pretraining inputs and outputs, highlighting pretraining as involving whole documents where the model predicts tokens, penalizing wrong predictions, instead of handling segmented input-output pairs.
- **Option to add EOS tokens considered reasonable**: Users debated whether it makes sense to add an option for **add_eos** in the text completion dataset, concluding it is a practical idea and helped fix a PPO implementation issue.

**Link mentioned**: <a href="https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_text_completion.py#L56).">torchtune/torchtune/datasets/_text_completion.py at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1256172869994545183)** (6 messages): 

- **Check out the evals workshop**: A member shared a link to a discussion about an evals workshop, hinting at its importance. They also shared a [GitHub link](https://github.com/posit-dev/positron) to Positron, a next-generation data science IDE.

- **Seeking JSONL data editor**: Two members expressed interest in a tool for iterating through and editing JSONL file examples directly within the same interface. One mentioned trying Lilac, which almost meets their needs but lacks direct editing capabilities.

- **Summarizing patient records**: A member is looking for tools or papers to generate structured summaries from patient records in JSON format, noting the need for methods different from text-to-text summarization. They are testing Llama models to avoid hallucinations and are seeking recommendations for prompt engineering and fine-tuning techniques.

**Link mentioned**: <a href="https://github.com/posit-dev/positron">GitHub - posit-dev/positron: Positron, a next-generation data science IDE</a>: Positron, a next-generation data science IDE. Contribute to posit-dev/positron development by creating an account on GitHub.

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1255989194275295242)** (4 messages): 

- **LLAMA on Streamlit throws error**: A member sought help with an error encountered after deploying **LLAMA** to Streamlit for an RAG application. They mentioned that the issue was not present locally but only emerged in the deployed environment.
- **Missing credits in account**: A member requested assistance with not receiving credits yet and provided their username and email for follow-up.
- **Tinyllama custom dataset path error resolved**: Initially, a member faced a `FileNotFoundError` while finetuning **Tinyllama** with a custom dataset. They later resolved it by setting the `path: my_test.jsonl` correctly without including the `data/` directory.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1256256276900483132)** (2 messages): 

- **Broken Link Issues Resolved**: A user mentioned that a link shared during a session was no longer working and requested an update. The issue was promptly acknowledged and fixed by another user.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1256121831342215219)** (2 messages): 

- **Kishore requests assistance with credits**: Kishore reported not receiving the credits and asked for help, providing his identifier `kishore-pv-reddy-ddc589`.

- **Christopher seeks credits for fireworks**: Christopher also requested credits for fireworks and included his identifier `christopher-438388`.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1255978316498735167)** (2 messages): 

- **You’ve got mail!**: A user pinged another member via DM on the Discord channel, asking them to check their messages. They used the plea emoji 🙏 to emphasize urgency.
- **Predibase credits expiration query**: A member asked if the **Predibase credits would expire on July 4th**. There was no response in the visible message history.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1256146343139737682)** (1 messages): 

- **User awaiting OpenAI credits**: A user posted that they haven't received their OpenAI credits yet. They provided their org ID **org-NBiOyOKBCHTZBTdXBIyjNRy5** and relevant email addresses (**karthikv2k@gmail.com** and **karthik@v2k.ai**).
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1256009698508013698)** (14 messages🔥): 

- **Open Interpreter prioritizes security with open discussion**: A member voiced concerns over Open Interpreter's security risks, leading to a detailed response about ongoing security measures such as user confirmation before code execution and sandboxing using Docker. The conversation emphasized transparency and community involvement to ensure the project's safety.
- **Performance comparison of Code models**: Members discussed the performance of various code models, noting that **Codestral** gives the best performance, while **DeepSeek Coder** is significantly faster but around 70% as good.
- **DeepSeek Coder-v2-lite praised for speed and code capability**: A member expressed a preference for **DeepSeek Coder-v2-lite** due to its fast performance and coding efficiency, suggesting it might be better than **Qwen-1.5b**.
- **Quantized model support inquiry**: There was an inquiry about running a SMOL multi-modal model for image understanding in a quantized form due to RAM limitations, highlighting a need for efficiency in resource-constrained environments.
- **YouTube video exposes Rabbit R1 security flaw**: A YouTube video titled ["Rabbit R1 makes catastrophic rookie programming mistake"](https://youtu.be/lkbV8oP-F44) was shared, revealing that Rabbit R1's codebase contains hardcoded API keys, compromising user data security.

**Link mentioned**: <a href="https://youtu.be/lkbV8oP-F44">Rabbit R1 makes catastrophic rookie programming mistake</a>: A group of jailbreakers recently discovered that the Rabbit R1 codebase contains hardcoded API keys - giving them easy access to user data from their AI tech...

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1256255566645563393)** (3 messages): 

- **Run OpenInterpreter locally with modifications**: A member explained that running OpenInterpreter locally with non-OpenAI providers requires some changes. They detailed the necessary differences in a [GitHub issue comment](https://github.com/OpenInterpreter/01/issues/272#issuecomment-2119175075).

- **APIs default to OpenAI**: It's noted that by default, the system likely uses OpenAI's API, potentially GPT-4 Turbo. However, specifics weren't confirmed as it hasn't been reviewed in a while.

- **Concerns about additional API costs**: Another member expressed concerns about additional charges when using the API, which are separate from the subscription costs.

**Link mentioned**: <a href="https://github.com/OpenInterpreter/01/issues/272#issuecomment-2119175075">Litellm/01 is unable to connect to non-openAI providers. · Issue #272 · OpenInterpreter/01</a>: What causes the issue: Run 01 specifying any non OAI server-host and api key Expected: Be able to connect to other services like Groq, Anthropic, OpenRouter etc as the seem to be working with the b...

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1256088170605314089)** (7 messages): 

- **Finished port supports finetuning**: A member announced the completion of a port, mentioning that it now works with finetunes as well. This indicates progress and potential new capabilities in their project.
- **FPGA-based systems for energy-efficient robotics**: Another member detailed their 8-month project focusing on energy-efficient humanoid robots. They emphasized the cost-effectiveness and logical approach of using FPGA-based systems to achieve large DRAM space with decent inference speed.
- **Humanoid robots' battery consumption on GenAI**: The same member pointed out that humanoid robots currently consume a lot of battery power on GenAI, which is inefficient given the use of 3-4 GPU-based SOMs per robot. They implied that the current setup is not sustainable.
- **Utility of JSON/YAML in tinygrad**: A user proposed making tinygrad capable of reading models from a JSON/YAML file, suggesting it could simplify configuration. Another member responded that models are already saved and loaded in dict form.
- **Current model storage mechanisms**: George Hotz clarified that **safetensors** are used for weights and **pickle** for compute in tinygrad. This highlights the project's current approach to model storage.
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1255977369727140013)** (4 messages): 

- **Shapetracker enables zero-cost tensor reshaping**: Members discussed the capabilities of the Shapetracker as explained in a [blog post](https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html). One member noted, *"if you have to reshape a huge tensor, the underlying data in the memory doesn’t have to change, just how you access it needs to change."*
- **Questions about mask solving**: In the context of Shapetracker, a member asked for clarification on what problems masks solve. They connected this query to understanding shape and strides in memory representation.
- **Curiosity about Shapetracker's origin**: A member expressed curiosity whether the logic behind Shapetracker was invented from scratch or inspired by other deep learning compilers. They marveled at how sophisticated it is, *"most frameworks optimize with strides, but shapetracker allows arbitrary movement ops with no copies at all."*

**Link mentioned**: <a href="https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html">How ShapeTracker works</a>: Tutorials on tinygrad

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1255987719281311886)** (7 messages): 

- **Internships at Cohere Spark Interest**: A student doing research on **LLMs and Reinforcement Learning** seeks insights into the work and culture at Cohere, asking for DMs from current employees. Another member noted the difficulty of securing internships at major AI companies, emphasizing the need for a substantial public portfolio.

- **Wish List for Cohere**: A member inquired if there are any features people wish to be added to **Cohere**.

- **AI Automation for Blogging**: One member asked for help setting up **AI-powered automations** to create blogs and post on social platforms. They were redirected to another channel for assistance.

- **Showcasing AI Built with Cohere and Langchain**: A member shared their [LinkedIn post](https://www.linkedin.com/posts/eddieotudor_datascience-aitechnology-machinelearning-activity-7212491482287542272-1cSF?utm_source=share&utm_medium=member_ios) about creating a **Data Analyst Agent** using **Cohere and Langchain**.

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1255974663784239184)** (4 messages): 

- **Support for Gemma2 with Sample Packing**: A member shared a [GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718) supporting Gemma2 with sample packing. They are waiting on an upstream Hugging Face fix linked within the PR.
- **27b Model Underwhelms in Benchmarks**: A user mentioned that the 27b model is surprisingly underwhelming in benchmarks compared to the 9b model, hinting at performance issues with the larger model.

**Link mentioned**: <a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1718">support for gemma2 w sample packing by winglian · Pull Request #1718 · OpenAccess-AI-Collective/axolotl</a>: Description  Motivation and Context   How has this been tested?    Screenshots (if appropriate) Types of changes  Social Handles (Optional)

  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1256088661678886912)** (3 messages): 

- **Featherless.ai launches model access platform**: Recently, a new platform was launched by **Featherless.ai** to provide access to over 450+ models on Hugging Face for a flat subscription starting at $10/month. The [platform](https://featherless.ai/) boasts features like no GPU setup/download required, OpenAI compatible API access, and new models added weekly with competitive pricing tiers.
- **Subscription Plans Detailed**: **Featherless.ai** offers two subscription tiers: Feather Basic at $10/month for up to 15B models and Feather Premium at $25/month for up to 72B models. Both plans offer unlimited personal use, with Feather Premium extending benefits to larger model sizes and private, secure, and anonymous usage.
- **Feedback Request for Prioritizing Models**: Community feedback is sought on which models to prioritize for addition to the **Featherless** platform. The platform’s early adopters mainly use it for AI persona local apps and more specific uses like language finetuning and SQL models.

**Link mentioned**: <a href="https://featherless.ai/"> Featherless - Serverless LLM</a>: Featherless - The latest LLM models, serverless and ready to use at your request.

  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1256304289484378204)** (3 messages): 

- **Request for Over-Time Elo Data**: A user inquired if there was "an over-time dataset or view of the chatbot arena elo numbers." They noted that the available JSON data only spans around six weeks.
- **Pack catching up**: The timeline mentioned starts from May 19th, and it was observed that the "pack" is catching up at the top.
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1256019429360140381)** (1 messages): 

- **Webinar on Enterprise-Scale Feature Store**: There is an upcoming webinar titled "Building an Enterprise-Scale Feature Store with Featureform and Databricks" on Tuesday, July 23rd at 8 A.M. PT. The session will feature Simba Khadder, who will discuss simplifying feature engineering, leveraging Databricks, and best practices for managing large-scale data, with a Q&A to follow. [Sign up here](https://buff.ly/3zh3B74).

**Link mentioned**: <a href="https://buff.ly/3zh3B74">Building an Enterprise-Scale Feature Store with Featureform and Databricks</a>: Join our 1-hr webinar with Featureform&#39;s founder to learn how to empower your data by using Featureform and Databricks!

  

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

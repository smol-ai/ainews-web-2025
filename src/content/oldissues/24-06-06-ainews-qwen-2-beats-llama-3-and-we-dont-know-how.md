---
id: faab9a69-3c95-4761-b1bb-d93d58c453b8
title: Qwen 2 beats Llama 3 (and we don't know how)
date: '2024-06-06T22:33:41.101639Z'
original_slug: ainews-qwen-2-beats-llama-3-and-we-dont-know-how
description: >-
  Alibaba released Qwen 2, a new open-source multilingual large language model
  series claiming to outperform Llama 3 across benchmarks, with models ranging
  up to 72B parameters and trained using innovative automated alignment
  strategies. Groq demonstrated extremely fast inference speeds on Llama 3 70B,
  processing about four Wikipedia articles in just 200 milliseconds.
tags:
  - alibaba
  - qwen2
  - llama3
  - multilingual
  - open-source
  - groq
  - inference-speed
---


<!-- buttondown-editor-mode: plaintext -->**Another model release with no dataset details.**

> AI News for 6/5/2024-6/6/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**408** channels, and **2450** messages) for you. 
Estimated reading time saved (at 200wpm): **304 minutes**.

With [Qwen 2](https://qwenlm.github.io/blog/qwen2/) being Apache 2.0, Alibaba is now claiming to universally beat **Llama 3** for the open models crown:

 ![image.png](https://assets.buttondown.email/images/99225fac-83f1-4251-9fed-565bd6757bfc.png?w=960&fit=max) 

There are zero details on dataset so it's hard to get any idea of how they pulled this off, but they do drop some hints on post-training:

> Our post-training phase is designed with the principle of scalable training with minimal human annotation. 
> 
> Specifically, we investigate how to obtain **high-quality, reliable, diverse and creative demonstration data and preference data with various automated alignment strategies**, such as 
> 
> - **[rejection sampling](https://arxiv.org/pdf/2308.01825) for math**, 
> - **execution feedback for coding** and 
> - **instruction-following, back-translation for creative writing**, 
> - **[scalable oversight](https://arxiv.org/pdf/2401.12474) for role-play**, etc. 
> 
> These collective efforts have significantly boosted the capabilities and intelligence of our models, as illustrated in the following table.

They also published a post on [Generalizing an LLM from 8k to 1M Context using Qwen-Agent](https://qwenlm.github.io/blog/qwen-agent-2405/).

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

**Qwen2 Open-Source LLM Release**

- **Qwen2 models released**: [@huybery](https://twitter.com/huybery/status/1798747031185559921) announced the release of Qwen2 models in 5 sizes (0.5B, 1.5B, 7B, 57B-14B MoE, 72B) as Base & Instruct versions. Models are **multilingual in 29 languages** and achieve **SOTA performance** on academic and chat benchmarks. Released under **Apache 2.0 except 72B**.
- **Performance highlights**: [@_philschmid](https://twitter.com/_philschmid/status/1798747595411779776) noted Qwen2-72B achieved **MMLU 82.3, IFEval 77.6, MT-Bench 9.12, HumanEval 86.0**. Qwen2-7B achieved **MMLU 70.5, MT-Bench 8.41, HumanEval 79.9**. On MMLU-PRO, Qwen2 scored **64.4, outperforming Llama 3's 56.2**.
- **Multilingual capabilities**: [@huybery](https://twitter.com/huybery/status/1798747042958967253) highlighted Qwen2-7B-Instruct's strong multilingual performance. The models are trained in **29 languages including European, Middle East and Asian languages** according to [@_philschmid](https://twitter.com/_philschmid/status/1798747598398132356).

**Groq's Inference Speed on Large LLMs**

- **Llama-3 70B Tokens/s**: [@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1798746103766200602) reported Groq achieved **40,792 tokens/s input rate on Llama-3 70B** using FP16 multiply and FP32 accumulate over the full 7989 token context length.
- **Llama 70B in 200ms**: [@awnihannun](https://twitter.com/awnihannun/status/1798765365117505677) put the achievement in perspective, noting Groq ran **Llama 70B on ~4 Wikipedia articles in 200 milliseconds**, which is about the blink of an eye, with 16-bit precision and 32-bit accumulation (lossless).

**Sparse Autoencoder Training Methods for GPT-4 Interpretability**

- **Improved SAE training**: [@gdb](https://twitter.com/gdb/status/1798764692669911142) shared a paper on improved methods for **training sparse autoencoders (SAEs) at scale to interpret GPT-4's neural activity**.
- **New training stack and metrics**: [@nabla_theta](https://twitter.com/nabla_theta/status/1798763600741585066) introduced a **SOTA training stack for SAEs** and trained a 16M latent SAE on GPT-4 to demonstrate scaling. They also proposed **new SAE metrics beyond MSE/L0 loss**.
- **Scaling laws and metrics**: [@nabla_theta](https://twitter.com/nabla_theta/status/1798765396113477801) found **clean scaling laws with autoencoder latent count, sparsity, and compute**. Larger subject models have shallower scaling law exponents. Metrics like downstream loss, probe loss, ablation sparsity and explainability were explored.

**Meta's No Language Left Behind (NLLB) Model**

- **NLLB model details**: [@AIatMeta](https://twitter.com/AIatMeta/status/1798420492774432769) announced the NLLB model, published in Nature, which can deliver **high-quality translations directly between 200 languages, including low-resource ones**.
- **Significance of the work**: [@ylecun](https://twitter.com/ylecun/status/1798446014723973333) noted NLLB's ability to provide **high-quality translation between 200 languages in any direction, with sparse training data, and for many low-resource languages**.

**Pika AI's Series B Funding**

- **$80M Series B**: [@demi_guo_](https://twitter.com/demi_guo_/status/1798500975671759001) announced Pika AI's **$80M Series B led by Spark Capital**. Guo expressed gratitude to investors and team members.
- **Hiring and future plans**: [@demi_guo_](https://twitter.com/demi_guo_/status/1798499472563110029) reflected on the past year's progress and teased **updates coming later in the year**. Pika AI is **looking for talent across research, engineering, product, design and ops** ([link](https://twitter.com/demi_guo_/status/1798501857041727748)).

**Other Noteworthy Developments**

- **Anthropic's elections integrity efforts**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1798732041925914903) published details on their **processes for testing and mitigating elections-related risks**. They also shared **samples of the evaluations used to test their models**.
- **Cohere's startup program launch**: [@cohere](https://twitter.com/cohere/status/1798688627007873124) launched a **startup program to support early-stage companies solving real-world business challenges with AI**. Participants get discounted access, technical support and marketing exposure. Cohere also released a **library of cookbooks for enterprise-grade frontier models** for applications like agents, RAG and semantic search ([link](https://twitter.com/cohere/status/1798453445076385968)).
- **Prometheus-2 for RAG evaluation**: [@llama_index](https://twitter.com/llama_index/status/1798454426904244588) introduced Prometheus-2, an **open-source LLM for evaluating RAG applications** as an alternative to GPT-4. It can process direct assessment, pairwise ranking and custom criteria.
- **LangChain x Groq integration**: [@LangChainAI](https://twitter.com/LangChainAI/status/1798576376255250534) announced an upcoming webinar on **building LLM agent apps with LangChain and Groq's integration**.
- **Databutton AI engineer platform**: [@svpino](https://twitter.com/svpino/status/1798701450396402157) shared that Databutton launched an **AI software engineer platform** to help build applications with React frontends and Python backends based on a business idea.
- **Microsoft's Copilot+ PCs**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1798749403240382604) reported on Microsoft's launch of **Copilot+ PCs with AI-first specs featuring generative models and search capabilities**, with the first machines using Qualcomm Snapdragon chips.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**LLM Developments and Applications**

- **Open-source RAG application**: In /r/LocalLLaMA, user open-sourced [LARS, a RAG-centric LLM application](https://www.reddit.com/r/LocalLLaMA/comments/1d8ur1y/open_sourcing_my_citationcentric_localllm/) that generates responses with detailed citations from documents. It supports various file formats, has conversation memory, and allows customization of LLMs and embeddings.

- **Favorite open-source LLMs**: In /r/LocalLLaMA, users shared their [go-to open-source LLMs for different use cases](https://www.reddit.com/r/LocalLLaMA/comments/1d8vapm/what_open_source_llms_are_your_daily_driver/), including Command-R for RAG on small-medium document sets, LLAVA 34b v1.6 for vision tasks, Llama3-gradient-70b for complex questions on large corpora, and more.

- **AI business models**: In /r/LocalLLaMA, a user questioned [how much original work vs leveraging existing LLM APIs](https://www.reddit.com/r/LocalLLaMA/comments/1d8wzqd/ai_business_coming_out_of_thin_air_and_llms/) new AI businesses are actually doing, as many seem to just wrap OpenAI's API for specific domains.

- **Querying many small files with LLMs**: In /r/LocalLLaMA, a user sought advice on [feeding 200+ small wiki files into an LLM](https://www.reddit.com/r/LocalLLaMA/comments/1d8xrlz/best_way_to_feed_lots_of_small_files_into_an_llm/) while preserving relationships between them, as embeddings and RAG have been hit-or-miss. Proper LLM training or LoRA are being considered.

- **Desktop app for local LLM API**: In /r/LocalLLaMA, a user created a [desktop app to interact with LMStudio's API server](https://www.reddit.com/r/LocalLLaMA/comments/1d8xlkf/i_made_a_desktop_app_for_interacting_with_my/) on their local machine and is gauging interest in releasing it for free.

- **Educational platform for LLMs**: In /r/LocalLLaMA, [Open Engineer was announced](https://www.reddit.com/r/LocalLLaMA/comments/1d90mn6/open_engineer_a_free_educational_platform_for/), a free educational resource covering topics like LLM fine-tuning, quantization, embeddings and more to make LLMs accessible to software engineers.

- **LLM assistant for database operations**: In /r/LocalLLaMA, a user shared their experience [integrating an LLM assistant into software](https://www.reddit.com/r/LocalLLaMA/comments/1d8xuxe/llm_company_assistant/) for database CRUD and order validation, considering using the surrounding system to supplement the LLM with product lookups for more robust order processing.

**AI Developments and Concerns**

- **Speculation on superintelligence development**: In /r/singularity, a user speculated that [major AI labs, chip companies and government agencies are likely already coordinating on superintelligence development](https://www.reddit.com/r/singularity/comments/1d8s0ke/there_is_probably_already_something_analagous_to/) behind the scenes, similar to the Manhattan Project, and that the US government is probably forecasting and preparing for the implications.

- **Questions around UBI implementation**: In /r/singularity, a user expressed frustration at the [lack of concrete plans or frameworks for implementing UBI](https://www.reddit.com/r/singularity/comments/1d96i0f/so_when_are_the_questions_surrounding_ubi/) despite it being seen as a solution to AI-driven job displacement, raising questions about funding, population growth impacts, and more that need to be addressed.

- **Risks of open-source AGI misuse**: In /r/singularity, a user asked how the open-source community will [prevent powerful open-source AGI from being misused by bad actors](https://www.reddit.com/r/singularity/comments/1d8t5md/what_is_the_open_source_community_solution_to/) to cause harm, given the lack of safeguards, arguing the "it's the same as googling it" counterargument is oversimplified.

- **Controlling ASI**: In /r/singularity, a user questioned the [belief that ASI could be controlled by humans](https://www.reddit.com/r/singularity/comments/1d97xc3/can_asi_even_be_controlled/) or militaries given its superior intelligence, with commenters agreeing it's unlikely and attempts at domination are misguided.

**AI Assistants and Interfaces**

- **Chat mode as voice data harvesting**: In /r/singularity, a user speculated [OpenAI's focus on chat mode is a strategic move](https://www.reddit.com/r/singularity/comments/1d923rm/is_chat_mode_openais_strategy_for_harvesting/) to collect high-quality, natural voice data to overcome the limitations of text data for AI training, as voice represents a continuous stream of consciousness closer to human thought processes.

- **Unusual ChatGPT use cases**: In /r/OpenAI, users shared [unusual things they use ChatGPT for](https://www.reddit.com/r/OpenAI/comments/1d9buhr/what_are_some_unusual_use_cases_no_ones_heard_of/), including generating overly dramatic plant watering reminders, converting cooking instructions from oven to air fryer, and mapping shopping list items to store aisles.

- **Need for an "AI shell" protocol**: In /r/OpenAI, a user envisioned the [need for a standardized "AI shell" protocol](https://www.reddit.com/r/OpenAI/comments/1d90kkn/ai_shell/) to allow AI agents to easily interface with and control various devices, similar to SSH or RDP, as existing protocols may not be sufficient.

**AI Content Generation**

- **Evolving views on AI music**: In /r/singularity, a user shared their [evolving perspective on AI music](https://www.reddit.com/r/singularity/comments/1d8p3wx/ai_music_i_feel_conflicted/), seeing it as well-suited for generic YouTube background music due to aggressive restrictions on mainstream music, and asked for others' thoughts.

- **AI-generated post-rock playlist**: In /r/singularity, a user shared a [30-minute AI-generated post-rock playlist on YouTube](https://www.reddit.com/r/singularity/comments/1d8sfgv/junes_chaos_an_aigenerated_postrock_journey/) featuring immersive tracks made with UDIO, receiving positive feedback about forgetting it's AI music. 

- **AI in VFX project**: In /r/singularity, a user shared a [VFX project incorporating AI to create urban aesthetics](https://www.reddit.com/r/singularity/comments/1d8sfgv/junes_chaos_an_aigenerated_postrock_journey/) inspired by ad displays, using procedural systems, image compositing, and layered AnimateDiffs, with CG elements processed individually and integrated.

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **LLM and Model Performance Innovations**:

   - **Qwen2 Attracts Significant Attention** with models ranging from 0.5B to 7B parameters, appreciated for their ease of use and rapid iteration capabilities, supporting innovative applications with [128K token contexts](https://qwenlm.github.io/blog/qwen2/).

   - **Stable Audio Open 1.0 Generates Interest** leveraging components like autoencoders and diffusion models, as detailed on [Hugging Face](https://huggingface.co/stabilityai/stable-audio-open-1.0), raising community engagement in custom audio generation workflows.

   - **ESPNet Competitive Benchmarks Shared for Efficient Transformer Inference**: Discussions around the newly released ESPNet showed promising transformer efficiency, pointing towards enhanced throughput on high-end GPUs (H100), as documented in the [ESPNet Paper](https://arxiv.org/abs/2406.03488).

   - **Seq1F1B Promotes Efficient Long-Sequence Training:** The pipeline scheduling method introduces significant memory savings and performance gains for LLMs, as per the [arxiv publication](https://arxiv.org/abs/2406.03488).

2. **Fine-tuning and Prompt Engineering Challenges**:

   - **Model Fine-tuning Innovations**: Fine-tuning discussions highlight the use of gradient accumulation to manage memory constraints, and custom pipelines such as using `FastLanguageModel.for_inference` for Alpaca-style prompts, as demonstrated in a [Google Colab notebook](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing).

   - **Chatbot Query Generation Issues**: Debugging Cypher queries using Mistral 7B emphasized the importance of systematic evaluation and iterative tuning methods in successful model training.

   - **Adapter Integration Pitfalls**: Critical challenges with integrating trained adapters pointed to a need for more efficient adapter loading techniques to maintain performance, supported by practical coding experiences.

3. **Open-Source AI Developments and Collaborations**:

   - **Prometheus-2 Evaluates RAG Apps**: Prometheus-2 offers an open-source alternative to GPT-4 for evaluating RAG applications, valued for its affordability and transparency, detailed on [LlamaIndex](https://t.co/BFnmE57OfB).

   - **Launch of OpenDevin** sparks collaboration interest, featuring a robust AI system for autonomous engineering developed by Cognition, with documentation available via [webinar](https://lu.ma/fp0xr460) and GitHub.

   - **Gradient Accumulation Strategies Improve Training**: Discussions on Unsloth AI emphasized using gradient accumulation to handle memory constraints effectively, reducing training times highlighted by shared [YouTube tutorials](https://www.youtube.com/watch?v=cwuYWFC7_QE).

   - **Mojo Rising as a Backend Framework**: Developers shared positive experiences using Mojo for HTTP server development, depicting its advantages in static typing and compile-time computation features on [GitHub](https://github.com/saviorand/lightbug_http/tree/main).

4. **Deployment, Inference, and API Integrations**:

   - **Perplexity Pro Enhances Search Abilities**: The recent update added step-by-step search processes via an intent system, enabling more agentic execution, as discussed within the community around [Perplexity Labs](https://labs.perplexity.ai/).

   - **Discussion on Modal's Deployment and Privacy**: Queries about using Modal for LLM deployments included concerns about its fine-tuning stack and privacy policies, with additional support provided through [Modal Labs documentation](https://www.google.com/search?q=privacy+policy+modal+labs).

   - **OpenRouter Technical Insights and Limits**: Users explored technical specifications and capabilities, including assistant message prefill support and handling function calls through [Instructor tool](https://useinstructor.com/).

5. **AI Community Discussions and Events**:

   - **Stable Diffusion 3 Speculation**: Community buzz surrounds the anticipated release, with speculation about features and timelines, as detailed in various [Reddit threads](https://www.reddit.com/r/StableDiffusion/comments/1d6ya9w/collection_of_questions_and_answers_about_sd3_and/).

   - **Human Feedback Foundation Event on June 11**: Upcoming discussions on integrating human feedback into AI, featuring speakers from Stanford and OpenAI with recordings available on their [YouTube channel](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg).

   - **Qwen2 Model Launches with Overwhelming Support**: Garnering excitement for its multilingual capabilities and enhanced benchmarks, the release on platforms like [Hugging Face](https://huggingface.co/Qwen) highlights its practical evaluations.

   - **Call for JSON Schema Support in Mozilla AI**: Requests for JSON schema inclusion in the next version to ease application development were prominently noted in [community channels](https://discord.com/channels/1089876418936180786/1182689832057716778/1248115957067415663).

   - **Keynote on Robotics AI and Foundation Models**: Investment interests in "ChatGPT for Robotics" amid foundation model companies underscore the strategic alignment detailed in [Newcomer's article](https://www.newcomer.co/p/why-investors-cant-get-enough-of).

---

# PART 1: High level Discord summaries




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Web Scraping Tool Talk**: Engineers swapped notes on **OSS scraping platforms** like **Beautiful Soup** and **Scrapy**, with a nod to **Trafilatura** for its proficiency in extracting dynamic content from tricky sources like job postings and SEC filings, citing its [documentation](https://trafilatura.readthedocs.io/en/latest/).

- **New UI Kid on the Block**: Google's new **Mesop** is stirring conversations for its potential in crafting UIs for internal tools, stepping into the domain of **Gradio** and **Streamlit**, despite lacking advanced authentication - curiosity piqued with a glance at [Mesop's homepage](https://google.github.io/mesop/).

- **Query Crafting Challenges**: Engineers debugged generating Cypher queries with **Mistral 7B**, emphasizing the importance of systematic evals, test-driven development, and the iterative process—a testament to the nitty-gritty of model fine-tuning.

- **Diving into Deployment**: Questions swirled about **Modal's** usage, including its fine-tuning stack complexity and privacy policy—a reference to [Modal Labs query](https://www.google.com/search?q=privacy+policy+modal+labs) for policy seekers, and a nod to their [Dreambooth example](https://modal.com/docs/examples/dreambooth_app) for practical enlightenment.

- **CUDA Compatibility Quirks**: The compatibility quirks between CUDA versions took the spotlight as engineers faced issues installing the `xformers` module—a pointer to [Jarvis Labs documentation](https://jarvislabs.ai/docs/troubleshooting#updating-cuda) on updating CUDA came to the rescue.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-3's Programming Puzzle**: While GPT models like **GPT-3** are adept at assisting with programming, limitations become apparent with highly specific and complex questions, signaling a push against their current capabilities.
- **Logical Loophes with Math Equations**: Even simple logical tasks, such as correcting a math equation, can stumble GPT models, revealing gaps in basic logical reasoning.
- **Eagerly Awaiting GPT-4o's Special Features**: Discussions anticipate **GPT-4o** updates, with voice and real-time vision features to be initially available to ChatGPT Plus users in weeks, and wider access later, as suggested by [OpenAI's official update](https://x.com/OpenAI/status/1790130708612088054?t=bgpr-IBhwXMa4ZiLXCrJgg&s=19).
- **DALL-E's Text Troubles**: Users are sharing workarounds for **DALL-E's** struggles with generating logos that contain precise text, including prompts to iteratively refine text accuracy and a useful [custom GPT prompt](https://chatgpt.com/g/g-TKZI5nYMc-one-word-graphics).
- **7b and Vision Model Synergy**: An integration success story where the **7b model** harmonizes well with the **llava-v1.6-mistral-7b** vision model, expanding possibilities for model collaboration.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Gradient Accumulation to the Rescue**: Engineers agreed that **gradient accumulation** can alleviate memory constraints and improve training times, but warned of potential pitfalls with larger batch sizes due to unexpected memory allocation behaviors.

**Tackling Inferential Velocity with Alpacas**: An engineer shared a code snippet leveraging `FastLanguageModel.for_inference` to utilize **Alpaca-style prompts** for sequence generation in **LLMs**, which sparked interest alongside discussions about a shared [Google Colab notebook](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing).

**Adapter Merging Mayday**: Challenges with integrating trained adapters causing significant dips in performance led to calls for more efficient adapter loading techniques to maintain training efficiency.

**Qwen2 Models Catch Engineers' Eyes**: Excitement bubbles over the release of **Qwen2 models**, with engineers keen on the smaller-sized models ranging from **0.5B to 7B** for their ease of use and faster iteration capabilities.

**Quest for Solutions in the Help Depot**: Conversations in the help channel emphasized a need for a VRAM-saving lora-adapter file conversion process, quick intel on a bug potentially slowing down inference, strategies for mitigating GPU memory overloads, and clarifications on running **gguf models** and implementing a RAG system, referenced to [Mistral documentation](https://docs.mistral.ai/guides/rag/).



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Audio Open 1.0 Hits the Stage**: [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) garners community interest for its innovative components, including the autoencoder and diffusion model, with members experimenting with tacotron2, musicgen, and audiogen within custom audio generation workflows.

- **Size Matters in Stable Diffusion**: Users recommend generating larger resolution images in **Stable Diffusion** before downsizing as an effective workaround for the random color issue plaguing small resolution (160x90) image outputs.

- **ControlNet Sketches Success**: ControlNet emerges as a preferred solution among members for converting sketches to realistic images without retaining unwanted colors, providing better control over the final composition and image details.

- **CivitAI Content Quality Control Called Into Question**: A surge in non-relevant content on **CivitAI** has led to calls for enhanced filtering capabilities to better curate quality models and maintain user experience.

- **Stable Diffusion 3 Awaits Clarification**: Despite rampant speculation within the community, the release date and details surrounding **Stable Diffusion 3** remain nebulous, with some members referencing a [Reddit post](https://www.reddit.com/r/StableDiffusion/comments/1d6ya9w/collection_of_questions_and_answers_about_sd3_and/) for tentative answers.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Taming the RAM Beast in LM Studio**: Users discussed strategies to limit RAM usage in LM Studio, suggesting approaches like loading and unloading models during use; a detailed method can be found in the [llamacpp documentation](https://example.com). While not a built-in feature in LM Studio, such tactics are employed to enable models to utilize RAM only when active despite efficiency losses.

- **Nomic Embed Models Step into the Limelight**: The discussion elevated **Nomic Embed Text models** to multimodal status, thanks to [nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5), setting them apart with impressive benchmark performances. The AI community within the server also noted **Q6 quant models of Llama-3** as a balance of quality and performance.

- **Jostling for the Perfect Configuration**: A user raised an issue about model settings resetting when initiating new chats, and discovered that retaining old chats could be a simple fix. The conversation also touched on `use_mlock` and `--no-mmap` settings in LM Studio, affecting stability during 8B model operations, emphasizing operating system-dependent subtleties.

- **Unlocking the Potential of Hardware Synergy for AI**: Engineers entered into a hearty debate on Nvidia’s proprietary drivers versus AMD’s open-source approach, highlighting implications for systems administration and security. Additionally, there was excitement about the promises of new **Qualcomm chips** and caution against judging ARM CPUs solely by synthetic benchmarks.

- **Updates and Upgrades Stir Excitement**: The **Higgs LLAMA model** garnered praise for its intelligence at a 70B scale, with anticipation building around an upcoming LMStudio update to incorporate relevant **llamacpp adjustments**. Another user is considering a massive RAM upgrade in anticipation of the much-discussed **LLAMA 3 405B** model, reflecting the intertwining interests between hardware capabilities and AI model evolution.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Moderation is Key**: The community debated moderation strategies in response to reports of inappropriate behavior. Professionalism in handling such issues is crucial.

**Gradio API Challenges**: Integrating Gradio with React Native and Node.js raised questions within the community. It's built with Svelte, so users were directed to investigate Gradio's API compatibility.

**Text with Stability**: Discussion around Stable Diffusion models for text generation pointed members towards solutions like AnyText and TextDiffuser-2 from Microsoft for robust output.

**When Compute Goes Peer-to-Peer**: The conversation turned to peer-to-peer compute for distributed machine learning, with tools like Petals and experiences with privacy-conscious local swarms offering promising avenues.

**Human Feedback in AI**: The Human Feedback Foundation is making strides in incorporating human feedback into AI, with an event on June 11th and a trove of educational sessions on their YouTube channel.

**Small Datasets, Big Challenges**: In computer vision discussions, dealing with small datasets and unrepresentative validation sets was a pressing concern. Solutions include using diverse training data and maybe even transformers despite their longer training times.

**Swin Transformer Tests**: There was a query about applying the Swin Transformer to CIFar datasets, highlighting the community's interest in experimenting with contemporary models in various scenarios.

**Deterministic Models Turn Down the Heat**: A single message highlighted lowering temperature settings to 0.1 to achieve more deterministic model behavior, prompting reflection on model tuning approaches.

**Sample Input Snafus**: Confusion over text embeddings and proper structuring of sample inputs for models like text-enc 1 and text-enc 2 surfaced, along with a discussion on the challenges posed by added kwargs in a dictionary format.

**Re-parameterising with Results**: A member successfully re-parameterised Segmind's **ssd-1b** into a **v-prediction/zsnr refiner** model and lauded it as a new favorite, hinting at a possible trend toward 1B mixture of experts models.

**A Helping Hand for Projects**: In a stretch of community aid, members offered personal assistance through DMs for addressing dataset questions, adding to the guild's collaborative environment.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**KAN Skepticism Expressed**: Kolmogorov-Arnold Networks (KANs) were deemed less efficient than traditional neural networks by guild members, with concerns about their scalability and interpretability. However, there's interest in more efficient implementations of KANs, such as those using ReLU, evidenced by a shared [ReLU-KAN architecture paper](https://arxiv.org/abs/2406.02075).

**Expanding the Data Curation Toolbox**: Participants debated the utility of **influence functions** in data quality evaluation, with the LESS algorithm ([LESS algorithm](https://www.cs.princeton.edu/~smalladi/blog/2024/04/04/dataselection/)) being mentioned as a potentially more scalable alternative for selecting high-quality training data.

**Breakthroughs in Efficient Model Training**: Innovations in model training were widely shared, including Nvidia's new open weights available on [GitHub](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro), the exploration of MatMul-free models ([arXiv](https://arxiv.org/abs/2406.02528)) for increased efficiency, and Seq1F1B's promise for more memory-efficient long-sequence training ([arXiv](https://arxiv.org/abs/2406.03488)).

**Quantization Technique May Boost LLM Performance**: The novel QJL method presents a promising avenue for large language models by compressing KV cache requirements through a quantization process ([arXiv](https://arxiv.org/abs/2406.03482)).

**Brain-Data Speech Decoding Adventure**: A guild member reported experimenting with **Whisper tiny.en embeddings** and brain implant data to decode speech, requesting peer suggestions to optimize the model by adjusting layers and loss functions while facing the constraint of a single GPU for training.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Gets Smarter**: Perplexity Pro has upgraded to show its search process step-by-step, employing an intent system for a more *agentic-like* execution, approximately a week ago.
  
- **File Format Frustrations**: Users are experiencing difficulties with Perplexity's ability to read PDFs, with success varying based on the PDF's content layout ranging from heavily styled to plain text.

- **Sticker Shock at Dev Costs**: The community reacted with humor and disbelief to a member's request for building a text-to-video MVP on a shoestring budget of $100, highlighting the disconnect between expectations and market rates for developers.

- **Haiku Feature Haunts No More**: The removal of Haiku and select features from Perplexity labs sparked discussions, leading members to speculate on cost-saving measures and express their discontent due to the impact on their workflow.

- **Curious about OpenChat Expansion**: An inquiry was raised regarding the potential addition of an **"openchat/openchat-3.6-8b-20240522"** model to Perplexity, alongside current models like **Mistral** and **Llama 3**.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Discovering Past Technical Sessions**: Recordings of past **CUDA MODE** technical events can be accessed on the [CUDA MODE YouTube channel](https://www.youtube.com/@CUDAMODE).

- **Debugging Tensor Memory in PyTorch**: A code snippet using `storage().data_ptr()` was shared to test if two PyTorch **tensors share the same memory**, stirring discussion on checking memory overlap. A member requested assistance in locating the source code for a PyTorch C++ function, specifically [at::_weight_int4pack_mm](https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html#exhale-function-namespaceat-1adeda9630914278ac02d7fd758da19e3d).

- **Extension on AI Models and Techniques**: Conversations hinge on methodologies like **MoRA** improving upon **LoRA**, and **DPO** versus **PPO** with respect to RLHF. Separate mentions went to **CoPE** for positional encodings and **S3D** accelerating inference, all found detailed in [AI Unplugged](https://datta0.substack.com/p/ai-unplugged-12-mora-dpo-vs-ppo-cope).

- **Torch Innovation Sparks**: Discussions ignited around **torch.compile** boosting KAN performance to rival MLPs, with insights shared from [Thomas Ahle's tweet](https://x.com/thomasahle/status/1798408687981297844), practical experiences, and the [GitHub repository for KANs and MLPs](https://github.com/thomasahle/kanmlps).

- **MLIR Sets Sights on ARM**: An MLIR meeting covered creating an **ARM SME Dialect**, offering insights into ARM's Scalable Matrix Extension via a [YouTube video](https://www.youtube.com/watch?v=jrniGW_Hzno). Hints pointing to potential **Triton ARM** support are discussed, with references to 'arm_neon' dialect for NEON operations in the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/ArmNeon/#arm_neonintrummla-arm_neonummlaop).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **The Quest for the ChatGPT of Robotics**: Investors are on the lookout for startups that can be the "ChatGPT for robotics," prioritizing AI over hardware. Excitement is building around niche foundation model companies as detailed in [this article](https://www.newcomer.co/p/why-investors-cant-get-enough-of).
  
- **Qwen2 Attracts Attention**: [Qwen2's launch](http://qwenlm.github.io/blog/qwen2/) has generated interest with models supporting up to 128K tokens in multilingual tasks, but users report gaps in recent event knowledge and general accuracy.

- **Dragonfly Unfurls Its Wings in Multi-modal AI**: Together.ai's [Dragonfly](https://www.together.ai/blog/dragonfly-v1) brings advances in visual reasoning, particularly in medical imaging, demonstrating integration of text and visual inputs in model development.

- **AI Community Takes a Critical View**: From discussions around influential lab employees criticizing smaller players to a shared [tweet](https://x.com/leopoldasch/status/1798483665904865474) highlighting the risk of AI labs inadvertently sharing advances with the CCP not the American research community, and [The Verge article](https://www.theverge.com/2024/6/5/24172377/humane-ai-pin-battery-case-issue-warning) on Humane AI's safety issue, the community remains vigilant.

- **Reinforcement Learning Paper Stirring Interest**: Sharing of the “Self-Improving Robust Preference Optimization” (SRPO) paper as announced in [this tweet](https://x.com/aahmadian_/status/1798740211909922862) indicates a focus on training LLMs using robust and self-improving RLHF methods. Nathan Lambert plans to dedicate time to discuss such cutting-edge papers.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Rising**: Discussions in various channels have centered on the advantages of using **Mojo** for backend server development, with examples like [lightbug_http](https://github.com/saviorand/lightbug_http/tree/main) demonstrating its use in crafting HTTP servers. The [Mojo roadmap](https://docs.modular.com/mojo/roadmap) was shared, indicating future core programming features, as active comparison to Python sparked debates on performance merits due to Mojo's static typing and compile-time computation.
  
- **Keeping Python Safe**: In teaching Python to newcomers, it's essential to avoid potential pitfalls ("footguns") to help learners transition to more complex languages, like C++.

- **Model Performance Frontiers Explored**: Discussions indicated that extending models like **Mistral** beyond certain limits requires ongoing pretraining, with suggestions to apply the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a merging strategy, though practicality and skepticism were voiced.

- **Community Coding Contributions Cloaked with Humor**: Expressions like "licking the cookie" were employed humorously to discuss encouraging open-source contributions, and members playfully reflected on the complexities of technical talks and coding challenges, likening a simple request for quicksort implementation to a noble quest.

- **Nightly Builds Yield Illumination and Frustration**: Nightly builds were examined with a spotlight on the usage of immutable auto-deref for list iterators and the introduction of features like `String.format` in the latest compiler release `2024.6.616`. However, network hiccups and the unpredictable nature of `algorithm.parallelize` for a `parallel_sort` function were sources of frustration, as seen from [GitHub discussions](https://github.com/rd4com/mojo_branch/tree/list_iter_autoderef_immut) and shared troubleshooting on workflow timing and network issues.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cohere Cashes In Big Time**: Cohere has secured a jaw-dropping **$450 million** in funding with significant contributions from tech titans like NVidia and Salesforce, despite not boasting a high revenue last year, as per a [Reuters report](https://www.reuters.com/technology/nvidia-salesforce-double-down-ai-startup-cohere-450-million-round-source-says-2024-06-04/).
- **IBM's Granite Gains Ground**: IBM's Granite models are being hailed for their transparency, particularly on data usage for training—prompting debates on whether they surpass OpenAI in the enterprise domain with insights from [Talha Khan](https://x.com/TalhaKhan_TK_/status/1798562313160761612).
- **Databricks Dominates Forrester’s AI Rankings**: In the latest report by Forrester on AI foundation models, Databricks has been recognized as a leader, emphasizing the tailored needs of enterprises and suggesting benchmark scores aren't everything. The report is highlighted in Databricks' [announcement blog](https://www.databricks.com/blog/databricks-named-leader-forrester-wavetm-ai-foundation-models-language-q2-2024) and is accessible for free [here](https://reprints2.forrester.com/#/assets/2/848/RES180932/report).
- **Qwen 2 Trumps Llama 3**: The new Qwen 2 model, with impressive 128K context window capabilities, shows superior performance to Llama 3 in code and math tasks, announced in a [recent tweet](https://x.com/reach_vb/status/1798748655366914325).
- **New Avenues for AI Web Interaction and Assistance**: Browserbase celebrates a **$6.5 million** seed fund aimed at enabling AI applications to navigate the web, shared by founders Nat & Dan, while Nox introduces an AI assistant designed to make the user experience feel invincible, with early sign-ups [here](http://heynox.com).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Prometheus-2 Pitches for RAG App Judging**: [Prometheus-2](https://t.co/BFnmE57OfB) is presented as an open-source alternative to GPT-4 for evaluating RAG applications, sparking interest due to concerns about transparency and affordability. 

**LlamaParse Pioneers Knowledge Graph Construction**: A posted notebook demonstrates how LlamaParse can execute first-class parsing to develop knowledge graphs, paired with a RAG pipeline for node retrieval.

**Configuration Overload in LlamaIndex**: AI engineers are expressing difficulty with the complexity of configuring LlamaIndex for querying JSON data and are seeking guidance, as well as discussing issues with Text2SQL queries not balancing structured and unstructured data retrieval.

**Exploring LLM Options for Resource-Limited Scenarios**: Discussions on alternative setups for those with hardware limitations veer towards smaller models like Microsoft Phi-3 and experimenting with platforms like Google Colab for heavier models.

**Scoring Filters Gain Customizable Edges**: Engineers are discussing the capability of LlamaIndex to filter results by customizable thresholds and performance score, indicating a need for fine-tuned precision in search results.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Startup Perks for Early Adopters**: Cohere introduces a [startup program](https://cohere.com/startup-program-application) aiming to equip Series B or earlier stage startups with discounts on AI models, expert support, and marketing clout to spur innovation in AI tech applications.

- **Refining the Chatbot Experience**: Upcoming changes to Cohere's Chat API on June 10th will bring a new default multi-step tool behavior and a `force_single_step` option for simplicity's sake, all documented for ease of adoption in the [API spec enhancements](https://docs.cohere.com/page/changes-in-chat-api-and-tool-use).

- **Hot Temperatures for AI Samplers**: OpenRouter stands out by allowing the temperature setting to exceed 1 for AI response samplers, contrasting with Cohere trial's 1.0 upper limit, opening discussions on flexibility in response variation and quality.

- **Developing Smart Group Chatbots**: Suggestions arose regarding the deployment of AI chatbots in group scenarios like business meetings, analyzing Rhea's advantageous multi-user context handling and potential precision concerns for personalized responses among numerous users.

- **Cohere Networking & Demos**: Community members welcomed new participant Toby Morning, exchanging LinkedIn profiles ([LinkedIn Profile](http://www.linkedin.com/in/urbantech/)) for broader connections and showcasing enthusiasm for upcoming demonstrations of the Coral AGI system's prowess in multi-user settings.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Qwen2 Leaps Ahead**: The launch of the **Qwen2** models marks a significant evolution from Qwen1.5, now featuring support for **128K token** context lengths, **27 additional languages**, and pretrained as well as instruction-tuned models in various sizes. They are available on platforms like [GitHub](https://github.com/QwenLM/Qwen2), [Hugging Face](https://huggingface.co/Qwen), and [ModelScope](https://modelscope.cn/organization/qwen), along with a [dedicated Discord server](https://discord.gg/yPEP2vHTu4).

**Map Event Prediction Discussion**: A user inquired about predicting true versus false event points on a map with temporal data, leading to a conversation about relevant commands and techniques, although specific methods were not provided.

**Update on Mistral API and Model Storage**: Mistral's introduction of a fine-tuning API and associated costs sparked discussion, with a focus on practical implications for development and experimentation. The API, including pricing details, is explained in their [fine-tuning documentation](https://docs.mistral.ai/guides/finetuning/).

**Mobile Text Input Gets a Makeover**: WorldSim Console updated their mobile platform, resolving bugs related to text input, improving text input reliability, and offering new features such as enhanced copy/paste and cosmetic customization options.

**Music Exploration in Off-Topic**: One member shared links to explore "Wakanda music", though this might have limited technical relevance for the engineer audience. Among the shared links were music videos like [DG812 - In Your Eyes](https://youtu.be/vP4zGMdTDPM) and [MitiS & Ray Volpe - Don't Look Down](https://youtu.be/e-Fors8CnKA).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Server Management Made Easy with Pilot**: The **Pilot** bot is revolutionizing how Discord servers are managed by offering features such as "Ask Pilot" for intelligent server insights, "Catch Me Up" for message summarization, and weekly "Health Check" reports on server activity. It's free to use and improves community growth and engagement, accessible through their [website](https://usepilot.app/).

**AI Competitors in Role-Playing Realm**: The WizardLM 8x22b model is currently gaining popularity in the role-playing community, nevertheless Dolphin 8x22 emerges as a potential rival, awaiting user tests to compare their effectiveness.

**Gemini Flash Sparks Image Output Curiosity**: Inquiries about whether **Gemini Flash** can render images spurred clarification that while no Large Language Model (LLM) presently offers direct image outputs, they can theoretically use base64 or call external services like Stable Diffusion for image generation.

**Tool Tips for Handling Function Calls**: For handling specific function calls and formatting, [Instructor](https://useinstructor.com/) is recommended as a powerful tool, facilitating automated command execution and improving user workflows.

**Technical Discussions Amidst Model Enthusiasm**: A member's query regarding prefill support in OpenRouter led to a confirmation that it's possible, particularly with the usage of reverse proxies; meanwhile, excitement is building around **GLM-4** due to its support for the Korean language, hinting at the model's potential in multilingual applications.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Human Feedback Spurs AI Improvement**: The upcoming *Human Feedback Foundation event* on June 11th is set to address the role of human feedback in enhancing AI applications across healthcare and civic domains; interested parties can register via [Eventbrite](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator). Additionally, recordings of past events with speakers from UofT, Stanford, and OpenAI are available at the [Human Feedback Foundation YouTube Channel](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg).

- **Azure Event Attracts AI Enthusiasts**: "Unleash the Power of RAG in Azure" is a highly subscribed Microsoft event happening in Toronto, as mentioned by an attendee seeking fellow participants; more details can be found on the [Microsoft Reactor page](https://developer.microsoft.com/en-us/reactor/events/22756/).

- **Tackling Messy Data**: Engineers discussed strategies for dealing with high-cardinality categorical columns, including the use of aggregate/grouping, manual feature engineering, string matching, and edit distance techniques, all with the goal of refining inputs for better regression outcomes.

- **Merging Data and Clustering Techniques**: There's a shared perspective that combining spell correction with feature clustering might streamline the challenges posed by high-cardinality categorical data, with an emphasis on treating such issues as core data modeling problems.

- **Practical Approaches to Feature Engineering**: Conversations pivoted towards pragmatic approaches like breaking down complex problems (e.g., isolating brand and item elements) and incorporating moving averages as part of the simplification technique for price prediction. Appreciation was expressed for the multifaceted solutions discussed, including regex for feature extraction.




---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Data Feast for AI Enthusiasts**: Engineers lauded the accessibility of **15T datasets**, humorously noting the conundrum of abundance in data but scarcity in computing resources and funding.

**GPU Banter Amidst Hardware Discussions**: The suitability of **4090s** for pretraining massive datasets sparked a facetious exchange, jesting about the limitations of consumer GPUs for such demanding tasks.

**Finetuning Fun with GLM and Qwen2**: The community shared tips and configurations for finetuning **GLM 4 9b** and **Qwen2 models**, noting that Qwen2's similarity to Mistral simplifies the process.

**Quest for Reliable Checkpointing**: The use of Hugging Face's `TrainingArguments` and `EarlyStoppingCallback` featured in talks about checkpoint strategies, specifically for capturing both the most recent and best performing states based on `eval_loss`.

**Error Hunting in AI Code**: Troubleshooting the "returned non-zero exit status 1" error prompted members to suggest pinpointing the failing command, scrutinizing `stdout` and `stderr`, and checking for permission or environment variable issues.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Catchy Naming Conundrum**: In the guild, clarity was sought on the **1B parameter zsnr/vpred refiner**; it's critiqued that the model is actually a **1.3B**, not a 1B model, sparking a light-hearted jab at the need for more accurate catchy names.

- **Vega's Parameter Predicament**: Discussions on the **Vega model** highlighted its swift processing prowess but raised concerns about its insufficient parameter size potentially limiting coherent output generation.

- **Elrich Logos Dataset Remains a Mystery**: A member queried about the availability of the **Elrich logos dataset** but did not receive any conclusive information or response regarding access.

- **The Dawn of Qwen2**: The launch of **Qwen2** has been announced, introducing substantial improvements over Qwen1.5 across multiple fronts including language support, context length, and benchmark performance. Qwen2 is now available in different sizes and supports up to **128K tokens** with resources spread across [GitHub](https://github.com/QwenLM/Qwen2), [Hugging Face](https://huggingface.co/Qwen), [ModelScope](https://modelscope.cn/organization/qwen), and [demo](https://huggingface.co/spaces/Qwen/Qwen2-72B-Instruct).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Knowledge Graph Construction Security Measures**: A tutorial on [constructing knowledge graphs from text](https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/) was shared, emphasizing the importance of data validation for security before inclusion in a graph database.

- **LangChain Tech Nuggets**: Confusion about the necessity of tools decorator prompted discussions for clarity, while a request for understanding token consumption tracking methods was observed among users. Additionally, a question arose about the creation of RAG diagrams as seen in a LangChain's FreeCodeCamp [tutorial video](https://youtu.be/sVcwVQRHIc8?si=BLfH2g7WUKtIi6A0).

- **Flow Control and Search Automation Resources**: A LangGraph conditional edges [YouTube video](https://youtu.be/EKxoCVbXZwY) was highlighted for its utility in flow control in flow engineering, and a new project termed [search-result-scraper-markdown](https://github.com/essamamdani/search-result-scraper-markdown) was shared for fetching search results and converting them to Markdown.

- **Cross-Framework Agent Collaboration**: Users expressed interest in frameworks that enable collaboration among agents built with different tools, incorporating LangChain, MetaGPT, AutoGPT, and even agents from platforms such as coze.com, highlighting the potential of interoperability in the AI space.

- **Calls for GUI and Course File Guidance**: There was a user query about finding a specific "helper.py" file from the AI Agents LangGraph course, pointing towards a need for better resource discovery methods within technical courses such as those offered on the DLAI [course page](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Preps for Prime Time**: **George Hotz** highlighted the need for updates in **tinygrad** before the **1.0 release**, with pending PRs expected to resolve the current gaps.
- **Unraveling Tensor Puzzles with UOps.CONST**: AI engineers examined the role of **UOps.CONST** in tinygrad, which serves as **address offsets** in computational processes for determining index values during tensor addition.
- **Decoding Complex Code**: In response to confusion over a snippet of code, it was clarified that intricate conditions are often needed to efficiently manage tensor shapes and strides within the row-major data layout constraints.
- **Indexing Woes Solved by Dynamic Kernel**: A discussion on tensor indexing in tinygrad revealed that kernel generation is essential due to the architecture's reliance on **static memory access**, with the kernel in question enabling operations like **Tensor[Tensor]**.
- **Masking with Arange for Getitem Operations**: The similarity between the kernel used for indexing operations and an **arange kernel** was noted, which facilitates the creation of a mask during **getitem** functions for dynamic tensor indexing.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Need for Speed with Graphics**: Members are seeking advice on executing graphics output with `interpreter.computer.run`, specifically for visualizations like those produced by `matplotlib` without success thus far.

**OS Mode Mayhem**: Conversations highlighted troubles in getting `--os mode` to operate correctly with local models from LM Studio, including issues with local LLAVA models not starting screen recording.

**Vision Quest on M1 Mac**: Engineers expressed frustration about hardware constraints on vision models for M1 Mac, indicating a strong interest in free and accessible AI solutions, given the high costs associated with OpenAI's offerings.

**Integration Anticipation for Rabbit R1**: Excitement is brewing over integrating Rabbit R1 with OpenInterpreter, particularly the upcoming webhook feature, to enable practical actions.

**Bash Model Request Open**: A call for suggestions for an open model suitable for handling bash commands has yet to be answered, leaving an open gap for potential recommendations.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**Curiosity for AI Town's Development Status**: Members in **AI Stack Devs** sought an update on the project, with one expressing interest in progress, while another apologized for not contributing yet due to a lack of time.

**Tileset Troubles in AI Town**: An engineering challenge surfaced around parsing spritesheets for **AI Town**, with a proposal to use the provided level editor or Tiled, supported by conversion scripts from the community.

**Learning to Un-Censor LLMs**: A member shared insights from a Hugging Face blog post on **abliteration**, which uncensors LLMs, featuring instruct versions of the third generation of Llama models. They followed up by inquiring about applying this technique to OpenAI models.

**Unanswered OpenAI Implementation Query**: Despite sharing the study on abliteration, a call for knowledge on how to implement the technique with OpenAI models went unanswered in the thread.

**For a deeper dive**:
- Parsing challenges and methods: \(not provided\)
- Blog post on abliteration: [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration).



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **The Truncated Text Mystery**: When working with embeddings using `llm`, a member discovered that input text exceeding the model's context length doesn't necessarily trigger an error and might result in truncated input. The actual behavior varies by model and underscores a need for clearer documentation on how each model handles excessive input length.
  
- **Embedding Jobs: To Resume or Not to Resume**: A query was raised regarding the functionality of `embed-multi` in resuming large embedding jobs without reprocessing completed parts. This highlights the need for features that can manage partial completions within embedding processes.

- **Documentation Desire for Embedding Behaviors**: The response from @SimonW pointing to a lack of clarity in model behavior documentation directed at whether inputs are truncated or produce errors, indicates a larger call from users for comprehensive and accessible documentation on these AI systems.

- **Guesswork in Model Truncation**: In absence of error messages, it was posited by @SimonW that the large text inputs leading to unexpected embedding results are likely being truncated, a behavior that should be explicitly verified within specific model documentation.

- **Efficiency in Large Embedding Tasks**: The discussion around whether the `embed-multi` can identify and skip previously processed data in rerunning large jobs showcases a concern for efficiency and the need for intelligent job handling in long-running AI processes.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

**Megatron's Checkpoint Conundrum**: Engineers enquired about **Megatron**'s compatibility with fine-tuning libraries, noting its unique checkpoint format. It was agreed that converting **Megatron checkpoints to Hugging Face format** and utilizing **Torchtune** for fine-tuning was the best course of action.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Call for JSON Schema Integration Heats Up**: An AI Engineer proposed the inclusion of a **JSON schema** in the upcoming software version to streamline application development, acknowledging some bugs but underscoring the ease it brings to building applications. No details on a timeline or potential implementation challenges were provided.



---



## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **Weekend Audio Learning on AI Infrastructure**: A link to a weekend listening session was shared by a member, featuring a discussion on AI infrastructure, which could be of interest to AI engineers looking to stay abreigned of the latest trends and challenges in the field. The content is accessible on [YouTube](https://youtu.be/4jPg4Se9h5g?si=ULVqGQa6AvI8Ch3o).



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1248011841976795137)** (66 messages🔥🔥): 

- **OSS Scraping Platform Recommendations**: A member asked for an OSS scraping platform and received suggestions including **Beautiful Soup** and **Scrapy**. Another member recommended **Trafilatura** for scraping dynamic content from job postings and SEC filings, providing a [link to Trafilatura's documentation](https://trafilatura.readthedocs.io/en/latest/).
- **Mesop compared with Gradio and Streamlit**: Google released **Mesop**, a Python-based framework for building UIs, which members compared favorably to **Gradio** and **Streamlit** for internal low-traffic apps. More details are provided on the [Mesop homepage](https://google.github.io/mesop/), piquing interest despite questions about advanced authentication features.
- **Finetuning Cypher Query Generation with Mistral 7B**: A member struggling with generating correct Cypher queries using Mistral 7B discussed systematic debugging steps with **HamelH**. The conversation emphasized writing evals, testing failure modes, and iteratively improving prompts.
- **Interest in Workshops on Finetuning Techniques**: Multiple members expressed interest in workshops covering topics like SFT, DPO, and ORPO using the **TRL library**, recommending **Leandro von Werra** as a potential instructor. However, space constraints and scheduling issues were mentioned as limiting factors.
- **Braintrust and OpenPipe Platforms**: A question was raised about **Braintrust** and **OpenPipe**, with responses noting previous office hours and talks covering these platforms. Members shared that these events answered many common questions about the platforms' purposes and utilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/live/QXVCqtAZAn4?si=y5cdnQnlsWOHGHlk">Aligning LLMs with Direct Preference Optimization</a>: In this workshop, Lewis Tunstall and Edward Beeching from Hugging Face will discuss a powerful alignment technique called Direct Preference Optimisation (DPO...</li><li><a href="https://google.github.io/mesop/">Mesop</a>: no description found</li><li><a href="https://www.tensorflow.org/guide/tpu">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1248071883371577434)** (2 messages): 

- **Regulatory and Research Aid through LLMs**: The LLM can serve as a **Research Assistant** by answering technical and regulatory questions using sources like scientific papers and regulatory documents. For pre-existing repositories like Arxiv, employing RAG and prompt engineering is recommended, while paywalled sources would need finetuning.
  
- **Discovery Helper for Research**: A secondary use case is a **Research Assistant** that points to promising papers by analyzing abstracts, titles, and metadata, which is valuable even if full access is restricted. Tools such as SciHub and DTIC can support this initiative by focusing on potential papers for the user.

- **LLMs for Legal Document Analysis**: The aforementioned research assistant LLMs can be adapted to handle legal documents for efficient research and discovery. The poster showed interest in seeing this implemented.

- **Document Distiller for Large Organizations**: This LLM is tailored for organizations with extensive document corpora (e.g., financial, government) to assist in regulatory compliance by summarizing and returning relevant documents based on inferred user intent. This idea is backed by mentions at the DataScience Salon conference by the NY Federal Reserve AI Chairman.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1248101600271269938)** (5 messages): 

```html
- **Jeremyhoward loves Hainan**: *"I love hainan! 😄"*. Later, Blaine shared his love for Shenzhou Peninsula mentioning nearby beaches and passion fruits.
- **Anmol from India seeks chatbot pricing advice**: Anmol asked for advice on pricing an enterprise customer service chatbot. He expressed hope that someone with experience could assist him.
- **Hanoi to Germany transition**: Hehehe0803 introduced themselves from Hanoi, Vietnam, currently living in Germany. They mentioned joining late and expressed hope to connect with others.
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1248007948622626907)** (8 messages🔥): 

```html
- **Modal Privacy Policy Sought**: A user inquired about the privacy policy of Modal. Another user provided a link to a Google search for further information: [Privacy Policy Modal Labs](https://www.google.com/search?q=privacy+policy+modal+labs).

- **Confusion on LLM Inference Setup**: A user asked about setting up a server to run an LLM and expose an endpoint, referencing a [Modal example script on GitHub](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/text_generation_inference.py#L240C20-L240C30). They were unsure about how to get the base URL for calling the endpoint from REST clients like Postman.

- **Praise for Modal from a GPU Enthusiast**: A user who typically trains locally with multiple GPUs tried Modal and found it "super cool". They expressed their appreciation with emojis: 👍👏.

- **Dataset Handling Issue with Axolotl Configs**: A user experienced issues with Modal's insistence on passing a dataset, which overrode their existing axolotl configuration. They mentioned hacking the `train.py` to remove the dataset code, which resolved the issue for them.
```

**Link mentioned**: <a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/text_generation_inference.py#L240C20-L240C30">modal-examples/06_gpu_and_ml/llm-serving/text_generation_inference.py at main · modal-labs/modal-examples</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1248228560737669211)** (1 messages): 

- **Loving the Discovery of Old News**: A member shared their excitement about a discovery they found, attaching [this paper from arXiv](https://arxiv.org/pdf/2207.09238). They acknowledged it might be "old news" but expressed that they "love it".
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1248002201100750859)** (2 messages): 

- **CUDA Version Mismatch Error**: A member encountered a mismatch error when installing the `xformers` PIP module due to differing CUDA versions, **11.8 and 12.1**. They asked for the recommended way to upgrade the CUDA library in the Jarvis Labs container.
- **Solution for Updating CUDA**: Another member provided a [documentation link to Jarvis Labs](https://jarvislabs.ai/docs/troubleshooting#updating-cuda) for guidance on updating the CUDA version on the Jarvis Labs instance.

**Link mentioned**: <a href="https://jarvislabs.ai/docs/troubleshooting#updating-cuda">Debugging and Troubleshooting | Jarvislabs</a>: Some common troubleshooting tips for updating Cuda, Freeing up the disk space and many more.

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1248317632000823367)** (3 messages): 

- **Members still awaiting credits**: Multiple users reported filling out a web form but **haven't received any credits** as expected. *"filled out the web form, but haven't received any as far as I can tell."*

- **New form on the way**: An update on the situation mentions that a new form will be available soon, with a **status check currently taking place**. *"We’ll have a new form out soon, let me check on the status of it."*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1248019842116751401)** (9 messages🔥): 

- **Redeem credits through email works**: One member had to accept credits via email and shared that *"this worked"* for them.
- **Unreceived credits inquiry**: *"Hi @zeke6585 haven't received the email"* stated a member, prompting another to check records and found the form was not filled out for replicate credits.
- **Duplicate form submissions still unresolved**: A member expressed confusion, having filled out the form twice and received credits from other services like OAI, HF, and Modal but not from Replicate.
- **Comment clarification on credit status**: A member clarified they were not spamming every channel but only the ones where credits were still pending, and noted an immediate resolution from Ankur on BrainTrust credits.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1248014877440868484)** (29 messages🔥): 

- **Langsmith Input Missing Issue Solved**: A member struggled with the `@traceable` decorator in Langsmith capturing outputs but not inputs in LLM calls. They resolved it by adding an argument to their function, realizing that inputs were essentially the arguments to the function without which nothing gets captured.

- **LangSmith Credit Confusion**: Multiple members expressed confusion and frustration over not receiving compute credits or understanding credit types. One highlighted that "Beta credits are only available for people who were LangSmith beta users."

- **Billing and Credits Follow-Up**: Several users who set up billing complained about not receiving their credits. They were directed to reach out to a contact at LangSmith to address the issue by sending their organization IDs for resolution. 

- **HIPAA Compliance and Enterprise Options**: LangSmith aims to be HIPAA compliant by July 1st and offers self-hosted options for enterprise customers. Details about plans and features are available, and members were guided to contact for more information.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.smith.langchain.com/pricing">Pricing | 🦜️🛠️ LangSmith</a>: Plans</li><li><a href="https://docs.smith.langchain.com/category/self-hosting">Self-hosting | 🦜️🛠️ LangSmith</a>: Self-hosting LangSmith requires an enterprise license. Check out the guides below for more information.</li><li><a href="https://docs.smith.langchain.com/how_to_guides/tracing/annotate_code#use-traceable--traceable)">Annotate code for tracing | 🦜️🛠️ LangSmith</a>: There are several ways to log traces to LangSmith.</li><li><a href="https://docs.smith.langchain.com/how_to_guides/tracing/annotate_code#wrap-the-openai-client">Annotate code for tracing | 🦜️🛠️ LangSmith</a>: There are several ways to log traces to LangSmith.</li><li><a href="https://docs.smith.langchain.com/how_to_guides/tracing/log_llm_trace">Log custom LLM traces | 🦜️🛠️ LangSmith</a>: Nothing will break if you don&#x27;t log LLM traces in the correct format and data will still be logged. However, the data will not be processed or rendered in a way that is specific to LLMs.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1248330512297230427)** (2 messages): 

- **Workshop videos receive high praise**: A member expressed gratitude for the **speakers and topics in the LLM Evals** videos, stating the course has significantly helped them structure their approach more effectively than a year of self-research. They thanked **<@525830737627185170>** and **<@916924724003627018>** for putting it together.
- **Facilitators appreciate positive feedback**: One of the organizers responded to the praise with thanks, sharing that such encouragement is highly motivating. *"It makes my day, and is super motivating!"*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-4](https://discord.com/channels/1238365980128706560/1242223495673286737/1248015694768115922)** (4 messages): 

- **Conference talk initially lacks sound**: A member noted that the recording for "Conference Talk: Best Practices For Fine Tuning Mistral w/ Sophia Yang" appeared to lack audio. They later clarified, stating, "never mind. it has voice."
- **Replicate vs. Modal deployment clarified**: A member sought confirmation on the differing deployment processes between Replicate and Modal, particularly regarding where the Docker build process occurs. Another member confirmed, "the Modal build process runs remotely."
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1248280518714200135)** (2 messages): 

- **Ben's talk rescheduled due to illness**: The team has rescheduled Ben's talk to next week because he isn't feeling well. Wishing that Ben gets well soon ❤️‍🔥.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1248012391250268362)** (150 messages🔥🔥): 

- **Frustration with Infographics and Tables in RAG**: Users discussed the difficulties of extracting data from PDFs, especially tables and infographics, in RAG systems. Tools like PyMuPDF, AWS Textract, and converting PDFs to Markdown were mentioned as potential solutions, but issues persist as noted: *"The markdown tables are malformed most of the time for my use case."*
- **Debate on Chunking Strategies**: There was a lively debate on the best practices for chunking text data for RAG, with recommendations ranging from 500 to 800 tokens with a 50% overlap. A consensus formed around the complexity and necessity of chunking correctly to ensure accurate context and retrieval.
- **Optimizing RAG with Fine-Tuning and Embeddings**: The importance of fine-tuning embedding models for better RAG performance was highlighted, with a suggestion to use generated synthetic data. One member noted, *"I think any company that is making money from RAG is leaving money on the table from not fine-tuning embedding models."*
- **Discussion on LanceDB for Multimodal AI**: LanceDB was discussed as an alternative to databases like Pinecone and SQL for managing embeddings on large-scale, multimodal data. This database promises to be **"easy-to-use, scalable, and cost-effective"** with support for hybrid search solutions.
- **Links Shared for Further Reading and Tools**: Multiple links were shared covering resources like fine-tuning pipelines, embedding quantization, and tools for PDF handling and RAG implementations. Key links include [Langchain Multi-modal RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb), [LlamaParse](https://github.com/run-llama/llama_parse), and [Creating Synthetic Data for Embeddings](https://x.com/_philschmid/status/1798388387822317933).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lancedb.github.io/lancedb/">LanceDB - LanceDB</a>: no description found</li><li><a href="https://lancedb.com">LanceDB - The Database for Multimodal AI</a>: The Database for Multimodal AI</li><li><a href="https://lancedb.github.io/lancedb/fts/">Full-text search - LanceDB</a>: no description found</li><li><a href="https://jxnl.github.io/blog/writing/2023/09/17/rag-is-more-than-embeddings/?h=rag+more">RAG is more than just embedding search - jxnl.co</a>: no description found</li><li><a href="https://manisnesan.github.io/chrestotes/posts/2023-07-07-doc-expansion-by-query-pred.html">chrestotes - Document Expansion by Query Prediction to Improve Retrieval Effectiveness</a>: no description found</li><li><a href="https://useinstructor.com/blog/2024/06/06/enhancing-rag-with-time-filters-using-instructor/">Enhancing RAG with Time Filters Using Instructor - Instructor</a>: no description found</li><li><a href="https://pymupdf.readthedocs.io/en/latest/rag.html">PyMuPDF, LLM &amp; RAG - PyMuPDF 1.24.4 documentation</a>: no description found</li><li><a href="https://github.com/xavctn/img2table">GitHub - xavctn/img2table: img2table is a table identification and extraction Python Library for PDF and images, based on OpenCV image processing</a>: img2table is a table identification and extraction Python Library for PDF and images, based on OpenCV image processing - xavctn/img2table</li><li><a href="https://github.com/VikParuchuri/marker/tree/master">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>: Convert PDF to markdown quickly with high accuracy - VikParuchuri/marker</li><li><a href="https://pymupdf.readthedocs.io/en/latest/the-basics.html#extracting-tables-from-a-page">The Basics - PyMuPDF 1.24.4 documentation</a>: no description found</li><li><a href="https://huggingface.co/blog/embedding-quantization">Binary and Scalar Embedding Quantization for Significantly Faster &amp; Cheaper Retrieval</a>: no description found</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/">Parent Document Retriever | 🦜️🔗 LangChain</a>: When splitting documents for retrieval, there are often conflicting desires:</li><li><a href="https://github.com/run-llama/llama_parse/issues/202">Mistakes parsing data from table using LlamaParse and gpt4o · Issue #202 · run-llama/llama_parse</a>: Trying to extract tabular data (table is embedded as an image) from a PDF file. While I&#39;ve managed to extract some data, there are consistent errors when the table is located at the bottom of the ...</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.</li><li><a href="https://github.com/jxnl/n-levels-of-rag">GitHub - jxnl/n-levels-of-rag</a>: Contribute to jxnl/n-levels-of-rag development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/parent_document_retriever/#retrieving-larger-chunks">How to use the Parent Document Retriever | 🦜️🔗 LangChain</a>: When splitting documents for retrieval, there are often conflicting desires:</li><li><a href="https://x.com/jxnlco/status/1798708039383679166">Tweet from jason liu (@jxnlco)</a>: Which one speaks “subscribe to my news letter, invite me to your conference, trust me with your Eng team?”</li><li><a href="https://modal.com/blog/fine-tuning-embeddings">Beating Proprietary Models with a Quick Fine-Tune</a>: Fine-tune on just a few hundred examples and kick off your very own data flywheel.</li><li><a href="https://x.com/_philschmid/status/1798388387822317933">Tweet from Philipp Schmid (@_philschmid)</a>: Creating a Pipeline for Generating Synthetic Data for Fine-Tuning Custom Embedding Models. 👀  Step 1 Create a Knowledge Base: Start with preparing your domain specific knowledge base, such as PDFs or...</li><li><a href="https://python.useinstructor.com/blog/">Welcome to the Instructor Blog - Instructor</a>: no description found</li><li><a href="https://jxnl.github.io/blog/writing/2024/02/28/levels-of-complexity-rag-applications/">Levels of Complexity: RAG Applications - jxnl.co</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/optimizing/production_rag/">Building Performant RAG Applications for Production - LlamaIndex</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb">langchain/cookbook/Multi_modal_RAG.ipynb at master · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/multi_doc_together_hybrid/#building-hybrid-retrieval-with-chunk-embedding-parent-embedding">Chunk + Document Hybrid Retrieval with Long-Context Embeddings (Together.ai) - LlamaIndex</a>: no description found</li><li><a href="https://lancedb.com/">LanceDB - The Database for Multimodal AI</a>: The Database for Multimodal AI</li><li><a href="https://github.com/kingjulio8238/memary">GitHub - kingjulio8238/memary: Longterm Memory for Autonomous Agents.</a>: Longterm Memory for Autonomous Agents. . Contribute to kingjulio8238/memary development by creating an account on GitHub.</li><li><a href="https://blog.dottxt.co/coalescence.html">Coalescence: making LLM inference 5x faster</a>: no description found</li><li><a href="https://x.com/jxnlco">Tweet from undefined</a>: no description found</li><li><a href="https://dub.sh/jxnl-rag">RAG - jxnl.co</a>: no description found</li><li><a href="https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/">Systematically Improving Your RAG - jxnl.co</a>: no description found</li><li><a href="https://jxnl.co/writing/2024/05/11/low-hanging-fruit-for-rag-search/">Low-Hanging Fruit for RAG Search - jxnl.co</a>: no description found</li><li><a href="https://jxnl.co/writing/2024/02/28/levels-of-complexity-rag-applications/">Levels of Complexity: RAG Applications - jxnl.co</a>: no description found</li><li><a href="https://jxnl.co/writing/2024/02/05/when-to-lgtm-at-k/">Stop using LGTM@Few as a metric (Better RAG) - jxnl.co</a>: no description found</li><li><a href="https://jxnl.github.io/blog/writing/2024/01/07/inverted-thinking-rag/">How to build a terrible RAG system - jxnl.co</a>: no description found</li><li><a href="https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/">RAG is more than just embedding search - Instructor</a>: no description found</li><li><a href="https://python.useinstructor.com/">Welcome To Instructor - Instructor</a>: no description found</li><li><a href="https://www.timescale.com/">PostgreSQL ++ for time series and events</a>: Engineered to handle demanding workloads, like time series, vector, events, and analytics data. Built on PostgreSQL, with expert support at no extra charge.</li><li><a href="https://www.limitless.ai/">Limitless</a>: Go beyond your mind’s limitations: Personalized AI powered by what you’ve seen, said, and heard.</li><li><a href="https://www.raycast.com/">Raycast - Your shortcut to everything</a>: A collection of powerful productivity tools all within an extendable launcher.</li><li><a href="https://www.tensorlake.ai/">Tensorlake</a>: no description found</li><li><a href="https://dunbar.app/">Home</a>: Your personal serendipity engine. Connect intelligently for new hire onboarding, peer learning, virtual coffees, and more. Try dunbar for Free No credit card required Spark meaningful connections Insp...</li><li><a href="https://www.bytebot.ai/">Bytebot - Leverage the power of AI in your web scraping, automation, testing and monitoring.</a>: Enhance and simplify your browser automation using our AI-enabled SDK. With Bytebot, creating web tasks is as easy as writing a prompt.</li><li><a href="https://www.narohq.com/">Naro - AI-powered sales knowledge</a>: no description found</li><li><a href="https://trunktools.com/">Trunk Tools</a>: Trunk Tools is at the forefront of construction innovation, offering cutting-edge AI solutions to streamline project management.</li><li><a href="https://modal.com/">Modal: High-performance cloud for developers</a>: Bring your own code, and run CPU, GPU, and data-intensive compute at scale. The serverless platform for AI and data teams.</li><li><a href="https://docs.pydantic.dev/latest/">Welcome to Pydantic - Pydantic</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1248104305756737586)** (6 messages): 

- **Exclusive sneak peek discussed**: Jeremy Howard shared a special sneak peek with members, instructing them, *"so keep it to yourself, folks!"* Despite the exclusivity, he mentioned it's fine to discuss it within this Discord.
- **Anticipation builds for demo**: A member expressed intent to wait for Jeremy Howard's demo after sneaking a peek at the codebase. 
- **Talk scheduled at an inconvenient time**: Ashpun noted that the talk timings inconveniently fell at 3:30am IST for them.
- **Setting multiple alarms**: To not miss the talk, one member humorously mentioned setting *"10 alarms"* while another member appreciated having company for the early hour.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[yang_mistral_finetuning](https://discord.com/channels/1238365980128706560/1242224842053521459/1248018263779311676)** (2 messages): 

- **Miscommunication cleared up with empathy**: *Aaah, okay, I understand now. That makes sense, and something I would expect as well. Sorry for misunderstanding you earlier.*

- **Excitement about Mistral API**: One member expressed enthusiasm about an upcoming workshop and stated, *I will try the official Mistral API*.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1248256278082490398)** (13 messages🔥): 

- **Dependency Conflicts During Axolotl Installation**: A user encountered errors while installing Axolotl with CUDA 12.1, Python 3.11, and PyTorch 2.3.1 due to dependency conflicts between multiple packages like Axolotl, Accelerate, Bitsandbytes, and Xformers. They sought resolutions for these conflicts.

- **Recommendation to Install Axolotl Without Xformers**: One member suggested that the issue was specifically with Xformers, not Axolotl, and recommended first installing Axolotl without Xformers. They also mentioned the alternative of using the Docker image for Axolotl.

- **Switching to Python 3.10 Resolves Partial Issues**: The user resolved some issues by switching to Python 3.10 and PyTorch 2.1.2, which allowed them to run the preprocess step but encountered a new error related to Flash Attention.

- **Flash Attention Requires Recompilation for CUDA 12.1**: The user faced an ImportError related to `libcudart.so.11.0` with Flash Attention, indicating a mismatch with their installed CUDA version (12.1). The suggested solution was to rebuild/recompile Flash Attention, which resolved the issue.

**Link mentioned**: <a href="https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts">Dependency Resolution - pip documentation v24.1.dev1</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1248244530843619401)** (12 messages🔥): 

- **Safe tensors vary in merged adapters:** A member questioned why merging an adapter to a base model sometimes results in 3 safe tensors and other times 6, while the base model only has 3 to start with, indicating variability in outcomes.

- **Enhance TPU efficiency with Keras's `steps_per_execution`:** A member shared a [TensorFlow blog](https://www.tensorflow.org/guide/tpu) about using `steps_per_execution` to reduce Python overhead and maximize TPU performance. An alternate approach in PyTorch XLA involves adjusting calls to `xm.mark_step()` for similar benefits.

- **Use `xm.mark_step()` judiciously:** A detailed explanation was given on how to manage TPU performance in PyTorch XLA using `xm.mark_step()` by adjusting the frequency of its calls within the training loop to balance performance and reliability, suggesting a potential feature request for the accelerate library. 

- **FSDP tutorial by Less Wright:** For those interested in FSDP, an excellent [ten-part series on YouTube](https://www.youtube.com/playlist?list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT) by Less Wright (former fastai alumn) was recommended as a comprehensive introduction.

- **Quantization process clarified:** It was confirmed that quantization happens during model load and is handled by the CPU before passing it to the GPU. Relevant documentation and resources were provided, including [Hugging Face Accelerate's quantization usage guide](https://huggingface.co/docs/accelerate/en/usage_guides/quantization) and the [Accelerate GitHub repository](https://github.com/huggingface/accelerate/blob/v0.30.1/src/accelerate/utils/bnb.py#L44).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/playlist?list=PL_lsbAsL_o2BT6aerEKgIoufVD_fodnuT">PyTorch FSDP Tutorials</a>: no description found</li><li><a href="https://github.com/huggingface/accelerate/blob/v0.30.1/src/accelerate/utils/bnb.py#L44">accelerate/src/accelerate/utils/bnb.py at v0.30.1 · huggingface/accelerate</a>: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed suppo.....</li><li><a href="https://huggingface.co/docs/accelerate/en/usage_guides/quantization">Quantization</a>: no description found</li><li><a href="https://www.tensorflow.org/guide/tpu">no title found</a>: no description found</li><li><a href="https://github.com/huggingface/accelerate/issues">Issues · huggingface/accelerate</a>: 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed suppo.....
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1248323683907276800)** (2 messages): 

- **Watch the YouTube Link**: A YouTube video was shared which can be accessed [here](https://m.youtube.com/watch?v=44vi31hehw4).

- **Python Handles Server-Side Code**: Discussion around the server-side code indicates that it’s still managed in **Python**. Points include **scaling in Spaces** and handling **concurrency** with Python.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1248030811513688074)** (12 messages🔥): 

- **Fine-tuning with Modal can be complex but is worth exploring**: Charles mentioned the complexity of the fine-tuning stack with Modal but suggested users check out their simpler [Dreambooth example](https://modal.com/docs/examples/dreambooth_app). This example demonstrates the core Modal concepts through finetuning the Stable Diffusion XL model using textual inversion.
- **Adapt existing datasets for fine-tuning on Modal**: Users can point to a Hugging Face dataset directly, as suggested by Charles. Adjustments should be made to avoid using Volumes for storing this data. 
- **Use batch processing for validation**: Charles advised writing a custom `batch_generate` method and using `.map` for processing and generating validation examples. He referenced the "embedding Wikipedia" example for further guidance.
- **Exploring Modal for cost-efficiency in app hosting**: Alan was advised by Charles on potentially moving the retrieval part of his Streamlit app to Modal or considering moving the entire app there. Concerns about cold starts and cost with a 24/7 deployment were discussed.
- **Quick support for script errors**: Chaos highlighted an issue with the `vllm_inference.py` script on GitHub, and Charles quickly responded, hinting at potential problems during the build step or GPU availability. Charles emphasized Modal’s culture of speed in support and communication.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dashboard.render.com/">Cloud Application Hosting for Developers | Render</a>: Render is a unified cloud to build and run all your apps and websites with free SSL, global CDN, private networks and automatic deploys from Git.</li><li><a href="https://github.com/modal-labs/modal-examples/issues/763">Error when Running `vllm_inference.py`: `CancelledError() · Issue #763 · modal-labs/modal-examples</a>: I have encountered an issue when attempting to run the vllm_inference.py script from the Modal Examples repository. Below are the steps I followed and the error I encountered: Steps to Reproduce Do...</li><li><a href="https://modal.com/docs/examples/dreambooth_app">Pet Art Dreambooth with Hugging Face and Gradio</a>: This example finetunes the Stable Diffusion XL model on images of a pet (by default, a puppy named Qwerty) using a technique called textual inversion from the “Dreambooth” paper. Effectively, it teach...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1248081428357447691)** (1 messages): 

- **Langsmith tracing obstacles with streaming non-OpenAI models**: A user discussed their challenge of capturing traces with **Langsmith** while using **Groq/Llama3** in a streaming fashion. They noted that using the `@traceable` decorator doesn't work with streaming outputs, as the result stays a `generator` object.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1247988604966273075)** (22 messages🔥): 

- **Last-minute enrollees can still claim credits**: A user who enrolled on May 30th asked how to claim credits, and Dan clarified that while some platforms like OAI won't offer credits, others like Modal might still be available. He shared a link to claim **OAI credits**: [OAI credits](https://discord.com/channels/1238365980128706560/1245927985123692575/1248045829705695333).

- **Modal credits now redeemable**: For users who enrolled after May 30th, Dan directed them to claim their Modal credits via a provided form [Modal form](https://bit.ly/modal-credits). He also provided a step-by-step guide for using Modal's platform and shared multiple example projects.

- **Credit claims still being processed**: Dan explained that credits are being processed in batches and reassured users who haven't yet received theirs to check if they have filled out the necessary forms. He urged those without credits to use the Modal platform to receive additional credits before the Tuesday deadline.

- **Replicate account issues resolved**: A user who hadn't received credits verified their Replicate account setup. After confirming the correct email and noticing the account creation date discrepancy, they found the credits in their email.

- **Replicate billing setup questions redirected**: Dan asked a user who received a Replicate invite to direct their billing-related questions to a more appropriate channel to get faster responses from the Replicate team.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bit.ly/modal-credits">Modal hackathon credits</a>: To claim your Modal credits, sign up for an account at https://modal.com/ first.  Then, let us know your username through this form.   For support, join the Modal Slack.  Here’s some examples to get s...</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[strien_handlingdata](https://discord.com/channels/1238365980128706560/1243773476301443073/1248012168868139099)** (1 messages): 

- **Praise for Data-Talk Presentation**: One member expressed strong appreciation for a particular presentation on data, emphasizing its foundational importance for tasks like evaluations and fine-tuning. They felt it would have been beneficial as the opening talk of the conference.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1247999686351257651)** (28 messages🔥): 

- **Account ID mix-ups lead to credit problems**: Multiple users reported issues with not receiving credits, often due to mistakenly filling out wrong account details like emails instead of account IDs in forms. Examples include user IDs such as *biggafish8-37cf1d* and *jain-nehil-ab4ee8*.

- **Credits now visible after corrections**: A user confirmed that their credits became visible after providing the correct account ID, which was *szilvia-beky-38bda3*. This suggests successful resolution for some users once the correct details were processed.

- **Expired credits notice**: When asked about the validity of credits, it was clarified that they expire after a year. This was confirmed by **aravindputrevu** stating *“Yes, they expire after a year.”*

- **Frequent guidance on form filling issues**: It was highlighted by **hamelh** that incorrect form filling, such as entering email addresses instead of account IDs, was a frequent issue. **project_disaster** acknowledged this with an apology for the mistake.

- **Continued calls for assistance**: Users continued to seek help with their account credits, providing their account IDs like *raul-brebenaru-2d3d45* and *roger-6803a6*, indicating ongoing issues even after initial error rectification efforts.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[emmanuel_finetuning_dead](https://discord.com/channels/1238365980128706560/1245129595749925086/1248044505530236968)** (98 messages🔥🔥): 

- **Fine-tuning is not dead but niche**: A member humorously suggested the title *"Fine-tuning is not dead, it's niche"* for clarity, emphasizing that fine-tuning has a specialized, expensive application. They stated, “fine-tuning can add hallucinations for Q&A,” highlighting its complexity and cost.
- **Anthropic and Speculative Thoughts**: Members discussed Anthropic's view on significant future changes, humor followed with mentions of '*Cylons*'. The talk by an Anthropic representative was highlighted: “Anthropic betting 8 years from now humans might not be around in present form.”
- **Resource Sharing for Fine-Tuning and RAG**: Multiple resources were shared regarding fine-tuning and RAG, like [Simon Willison's blog](https://simonwillison.net/) and [Anthropic's research](https://www.anthropic.com/). Emmanuel’s book and tools like LLM CLI were mentioned as valuable for understanding fine-tuning's applied aspects.
- **Prompting over Fine-Tuning**: Members preferred focusing on prompt engineering over fine-tuning for many applications, suggesting reading materials like Emmanuel's spreadsheets on prompt engineering and [HuyenChip's blog](https://huyenchip.com/2023/04/11/llm-engineering.html#prompt_optimization). The point was made with humor: "Do the boring thing!" as advice over complex fine-tuning.
- **Dynamic Few-Shot and RAG Discussions**: The talk concluded with insights into using dynamic few-shot prompting and RAG as viable alternatives or complements to fine-tuning. Links like [dynamic few-shot prompting article](https://medium.com/@iryna230520/dynamic-few-shot-prompting-overcoming-context-limit-for-chatgpt-text-classification-2f70c3bd86f9) were shared to emphasize practical approaches in evolving applications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1238365980128706560/1242223458184597534/1245504052738129961">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://www.amazon.com/Building-Machine-Learning-Powered-Applications/dp/149204511X">no title found</a>: no description found</li><li><a href="https://medium.com/@iryna230520/dynamic-few-shot-prompting-overcoming-context-limit-for-chatgpt-text-classification-2f70c3bd86f9">Dynamic Few-Shot Prompting: Overcoming Context Limit for ChatGPT Text Classification</a>: Recent explosion in the popularity of large language models like ChatGPT has led to their increased usage in classical NLP tasks like…</li><li><a href="https://www.mlpowered.com/">mlpowered</a>: Blog posts and other information</li><li><a href="https://www.kaggle.com/competitions/kaggle-llm-science-exam/leaderboard">Kaggle - LLM Science Exam</a>: no description found</li><li><a href="https://www.mlpowered.com/book/">A book about practical problems</a>: Available now on Amazon and O&rsquo;Reilly. I wrote this book to give readers tools to solve the most common practical ML problems based on my experience mentoring hundreds of Data Scientists and ML E...</li><li><a href="https://x.com/stefanhgm/status/1765466556216053879">Tweet from Stefan Hegselmann (@stefanhgm)</a>: Does removing unsupported facts in the training or prompting data effectively reduce hallucinations?  We tested this for GPT-4 & Llama 2 for generating patient summaries. W/ @shannonzshen, Florian Gie...</li><li><a href="https://medium.com/@iryna230520/dynamic-few-shot-prompting-overcoming-context-limit-for-chatgpt-">no title found</a>: no description found</li><li><a href="https://docs.google.com/spreadsheets/d/1w7gnJTevZwVojzfrmyb3IPI7-k7DtdOaN0r2wTWnNHU/edit#gid=150872633">Anthropic Prompt Engineering Interactive Tutorial [PUBLIC ACCESS]</a>: Tutorial How-To  Tutorial How-To This tutorial requires an API key for interaction.   If you don&#39;t have an API key, you can sign up for one via the &lt;a href=&quot;https://console.anthropic.com/&...</li><li><a href="https://simonwillison.net/">Simon Willison’s Weblog</a>: no description found</li><li><a href="https://www.quora.com/Should-you-fine-tune-an-LLM-or-just-do-prompt-engineering/answer/Tong-Hui-Kang-1">Tong Hui Kang&#039;s answer to Should you fine-tune an LLM, or just do prompt engineering? - Quora</a>: no description found</li><li><a href="https://www.quora.com/What-is-the-future-of-prompt-engineering-versus-fine-tuning/answer/Tong-Hui-Kang-1">Tong Hui Kang&#039;s answer to What is the future of prompt engineering versus fine-tuning? - Quora</a>: no description found</li><li><a href="https://llm.datasette.io/en/stable/">LLM: A CLI utility and Python library for interacting with Large Language Models</a>: no description found</li><li><a href="https://github.com/simonw/llm">GitHub - simonw/llm: Access large language models from the command-line</a>: Access large language models from the command-line - simonw/llm</li><li><a href="https://arxiv.org/abs/2401.08406">RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture</a>: There are two common ways in which developers are incorporating proprietary and domain-specific data when building applications of Large Language Models (LLMs): Retrieval-Augmented Generation (RAG) an...</li><li><a href="https://arxiv.org/abs/2303.17564">BloombergGPT: A Large Language Model for Finance</a>: The use of NLP in the realm of financial technology is broad and complex, with applications ranging from sentiment analysis and named entity recognition to question answering. Large Language Models (L...</li><li><a href="https://arxiv.org/abs/2305.05862">Are ChatGPT and GPT-4 General-Purpose Solvers for Financial Text Analytics? A Study on Several Typical Tasks</a>: The most recent large language models(LLMs) such as ChatGPT and GPT-4 have shown exceptional capabilities of generalist models, achieving state-of-the-art performance on a wide range of NLP tasks with...</li><li><a href="https://arxiv.org/abs/2402.15422">A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models</a>: Patients often face difficulties in understanding their hospitalizations, while healthcare workers have limited resources to provide explanations. In this work, we investigate the potential of large l...</li><li><a href="https://huyenchip.com/2023/04/11/llm-engineering.html#prompt_optimization">Building LLM applications for production</a>: [Hacker News discussion, LinkedIn discussion, Twitter thread]</li><li><a href="https://docs.google.com/spreadsheets/d/1w7gnJTevZwVojzfrmyb3IPI7-k7DtdOaN0r2wTWnNHU">Anthropic Prompt Engineering Interactive Tutorial [PUBLIC ACCESS]</a>: Tutorial How-To  Tutorial How-To This tutorial requires an API key for interaction.   If you don&#39;t have an API key, you can sign up for one via the &lt;a href=&quot;https://console.anthropic.com/&...</li><li><a href="https://www.anthropic.com/">Home</a>: Anthropic is an AI safety and research company that&#x27;s working to build reliable, interpretable, and steerable AI systems.</li><li><a href="https://community.openai.com/t/fine-tuning-vs-context-injection-rag/550286">Fine-tuning vs Context-Injection (RAG)</a>: Hi, community.  I finished my research work on comparing fine-tuning with context-injection (as an implementation of retrieval-augmented generation). A lot of work went into organizing the experimenta...</li><li><a href="https://arxiv.org/abs/2403.01432">Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge</a>: Large language models (LLMs) memorize a vast amount of factual knowledge, exhibiting strong performance across diverse tasks and domains. However, it has been observed that the performance diminishes ...</li><li><a href="https://arxiv.org/abs/2312.05934">Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs</a>: Large language models (LLMs) encapsulate a vast amount of factual information within their pre-trained weights, as evidenced by their ability to answer diverse questions across different domains. Howe...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1248019594103492818)** (29 messages🔥): 

- **Braintrust Inputs Not Captured**: A user noted that the Braintrust decorator wasn't capturing the inputs to their LLM function but was capturing the outputs. Another pointed out that the function had no arguments and suggested wrapping the OpenAI client with `wrap_openai`.
- **Exploring Braintrust Tracing Methods**: There was a discussion on three methods to trace in Braintrust: `wrap_openai`, the `@traced` decorator, and spans, with spans offering the most flexibility. One user considered using spans for a project integrating Braintrust with ZenML.
- **Credits Issue Resolved with UI Clarification**: A user named "project_disaster" mentioned not seeing their credits, with "ankrgyl" clarifying that the absence of an "Upgrade" button indicates applied credits. The user suggested having a visible gauge for tracking credit consumption over time.
- **Interest in TypeScript and Tracing Example**: A user expressed an off-topic appreciation for the use of TypeScript. Another inquired about starting with a tracing example for an LLM project, eventually planning to move towards DPO finetuning, with "ankrgyl" suggesting they begin with the logging guide on Braintrust's documentation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.braintrust.dev/docs/guides/logging">Braintrust</a>: Braintrust is the enterprise-grade stack for building AI products.</li><li><a href="https://www.braintrust.dev/docs/guides/tracing#wrapping-openai">Braintrust</a>: Braintrust is the enterprise-grade stack for building AI products.</li><li><a href="https://www.braintrust.dev/docs/guides/tracing#annotating-your-code">Braintrust</a>: Braintrust is the enterprise-grade stack for building AI products.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1248157328855793707)** (2 messages): 

```html
- **Local Roots Shout-Out**: One user mentioned living in London but originally being from Portugal. Another user opted to keep their origin a secret with a *"🤐"* emoji.
```
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[announcements](https://discord.com/channels/1238365980128706560/1245460787196068030/1248046279532220529)** (2 messages): 

- **Fill out the OpenAI credits form**: *If you missed filling out the form the first time and want your OpenAI credits, please put your OAI org id in the form at* [this link](https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f).

- **Find your Organization ID for OpenAI credits**: To see your Org ID, *please go to* [this website](https://platform.openai.com/settings/organization/general) *after you are logged in under `Organization ID` (this is available even if you are* not part of an org with someone*).*

**Link mentioned**: <a href="https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f">no title found</a>: no description found

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1248322865468674078)** (3 messages): 

- **Questions on starting model format**: A user asked if they should start with **awq** or **gptq** format before training LoRAs for best performance. No specific guidance was provided in the conversation.
- **Excitement for Predibase example**: A member expressed excitement about using Predibase's example shared in the [documentation](https://docs.predibase.com/user-guide/examples/rag). Predibase claims to offer the fastest way to **fine-tune** and **serve** open-source LLMs.
- **Inquiry about credit expiry**: A user asked about the expiration of credits and mentioned the requirement to add a credit card to be upgraded to the Developer Tier on the site. No further details about the credit expiry were discussed.

**Link mentioned**: <a href="https://docs.predibase.com/user-guide/examples/rag.">Quickstart | Predibase</a>: Predibase provides the fastest way to fine-tune and serve open-source LLMs. It&#x27;s built on top of open-source LoRAX.

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1247999434764193834)** (4 messages): 

- **Members still await their credits**: Several members reported they have not yet received their credits. One provided an email for follow-up, urging a quick resolution.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1247988384195149835)** (69 messages🔥🔥): 

```html
<!-- Summary -->

- **OpenAI credits applied retroactively**: Several users noted that their credits were applied to their existing API balance, making it similar to adding funds via a credit card. [Members discussed](https://platform.openai.com/settings/organization/billing/overview) potential improvements for those new to the API.
- **Finalizing Tier 2 API status for students**: OpenAI granted Tier 2 API status to those who filled out the form in time, allowing them to utilize the additional credits. Users should stay tuned for updates if they missed the initial registration.
- **Late submission form for credits**: To rectify earlier submission errors, [a new form for additional credit requests](https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f) has been shared and needs to be correctly filled out.
- **Internal thoughts during fine-tuning**: There was an in-depth discussion regarding how to handle "internal thoughts" in long multi-turn conversations during OpenAI model fine-tuning. Delimiters and separate examples were proposed as potential solutions.
- **Public acknowledgment and kudos**: The group appreciated the efforts of OpenAI team members for their swift and effective support, highlighted in a [Twitter post](https://x.com/TheZachMueller/status/1798674326633247143) expressing gratitude.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://maven.com/parlance-labs/fine-tuning/1/forms/f2d68f">no title found</a>: no description found</li><li><a href="https://x.com/TheZachMueller/status/1798674326633247143">Tweet from Zach Mueller (@TheZachMueller)</a>: Us: Hmm, how are we going to spend $500 worth of @OpenAI credits as part of @HamelHusain&#39;s course?   @OpenAI: Oh bet, good point! Everyone gets tier-2 status   HUGE kudos to @shyamalanadkat and An...</li><li><a href="https://cookbook.openai.com/examples/third_party/gpt_finetuning_with_wandb">Fine-tuning OpenAI models with Weights &amp; Biases | OpenAI Cookbook</a>: no description found</li><li><a href="https://cookbook.openai.com/examples/chat_finetuning_data_prep">Data preparation and analysis for chat model fine-tuning | OpenAI Cookbook</a>: no description found</li><li><a href="https://cookbook.openai.com/examples/fine_tuning_for_function_calling">Fine tuning for function calling | OpenAI Cookbook</a>: no description found</li><li><a href="https://cookbook.openai.com/examples/how_to_finetune_chat_models">How to fine-tune chat models | OpenAI Cookbook</a>: no description found</li><li><a href="https://cookbook.openai.com/examples/fine-tuned_qa/ft_retrieval_augmented_generation_qdrant">Fine-Tuning for retrieval augmented generation (RAG) with Qdrant | OpenAI Cookbook</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1247989753517838406)** (266 messages🔥🔥): 

```html
- **GPTs at their limits with advanced programming questions**: A user noted that their programming questions have become more specific and complex as their project advanced, leading to struggles with GPT models. They expressed concern that these models may be "pushing their limits for programming assistance 😁".
- **GPTs sometimes fail at simple corrections**: Another user pointed out a problem where the GPT could not correct an incorrect math equation despite being prompted, showcasing issues with basic logical consistency in the model.
- **Continuous Learning and Real-time Adjustments**: Discussion involved the idea that making models agentic and capable of continuous learning could be costly and pose regulatory challenges. Continuous learning could also lead to issues with personality drift and potential security risks.
- **Generative AI's current and future impact**: There was debate about the immediate usefulness and future potential of generative AI, with some users highlighting its potential to assist or significantly change job structures, while others were skeptical of its broader economic impacts.
- **Community discussions on AI advances and resource requirements**: Users conversed about the computational power required for training AI models, referencing specific hardware like A100 and H100 GPUs, and speculating on developments with upcoming models like GPT-5.
```
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1248049032002015282)** (20 messages🔥): 

- **Voice chat removal concerns**: Users discussed why voice chat was removed and speculated on its return with a new model. One user mentioned a new voice model based on GPT-4o is expected soon.
- **Issues with generating Chinese text**: A member reported that generating responses in Chinese using the GPT model sometimes results in special characters like \ufffd appearing. This issue occurs around 15% of the time and significantly impacts the text quality.
- **GPT-4o feature rollout timelines debated**: Members discussed the expected rollout timeline for GPT-4o's real-time voice and vision features. Official updates suggest initial availability for ChatGPT Plus users in the coming weeks, with broader access over the next few months ([official update](https://x.com/OpenAI/status/1790130708612088054?t=bgpr-IBhwXMa4ZiLXCrJgg&s=19)).
- **GPT-4o free plan limits**: A discussion touched on the number of questions one can ask on the GPT-4o free plan. The consensus was that the limit is around 10 questions.


**Link mentioned**: <a href="https://x.com/OpenAI/status/1790130708612088054?t=bgpr-IBhwXMa4ZiLXCrJgg&s=19">Tweet from OpenAI (@OpenAI)</a>: All users will start to get access to GPT-4o today. In coming weeks we’ll begin rolling out the new voice and vision capabilities we demo’d today to ChatGPT Plus.

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1247992448827920385)** (6 messages): 

- **LLaVA-Mistral Vision Model Coexists with 7b**: A member mentioned that the 7b model appears to work well with the vision model from **llava-v1.6-mistral-7b** when combined. They found it neat that this integration works at all.

- **Confusion over Image Permissions**: A member expressed frustration over the removal of image permissions in the channel, questioning why this change occurred. Another echoed the query in a follow-up message.

- **Challenges with Text Accuracy in DALL-E Logos**: Members discussed difficulties in generating logos with exact text using **DALL-E**. One member shared a method for improving text accuracy by repeatedly checking and regenerating until the text in the image is correct as part of the prompt. Another member suggested including specific instructions to emphasize distinct text layers in the image generation process, linking to a custom GPT prompt for reference: [Custom GPT prompt](https://chatgpt.com/g/g-TKZI5nYMc-one-word-graphics).
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1247992448827920385)** (6 messages): 

- **7b Model integrates with Vision Model**: A member mentioned that the **7b model** plays well with the vision model from **llava-v1.6-mistral-7b** when placed in its folder. They found it "kinda neat" that this integration works.
- **Channel Image Permissions Concern**: A member expressed confusion over the removal of image permissions in the channel. They questioned why these permissions were removed.
- **Struggles with Exact Text in DALL-E Logos**: A member asked if it's possible to generate logos with exact text using **DALL-E**, sharing difficulties with "misbuilded letters." Another member shared a workaround prompt that checks and corrects text until it is accurate.
- **Helpful Prompt for Accurate Text in DALL-E Images**: A member shared a prompt that helps ensure the text in DALL-E images is correct by layering and checking accuracy until the desired text is achieved. They also referenced incorporating this into [custom GPT instructions](https://chatgpt.com/g/g-TKZI5nYMc-one-word-graphics) to satisfy users.
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1247990342410702969)** (148 messages🔥🔥): 

```html
<ul>
  <li><strong>Gradient Accumulation Insights:</strong> Members discussed how <em>gradient accumulation</em> can help with memory issues and batch size. "It’ll decrease the time compared to small batch size", but gets tricky with larger batch sizes due to memory allocation quirks.</li>
  <li><strong>Addressing CUDA Memory Issues:</strong> <em>"When increasing batch size, the sequences' different lengths slow down the process."</em> Suggested using "gradient accumulation" or "non-power of 2 batch sizes" to mitigate memory spikes.</li>
  <li><strong>Training and Merge Issues:</strong> Members faced issues with <em>merging trained adapters</em> leading to significant performance degradation. There's a call out for effective loading of adapters to continue training without losing efficiency.</li>
  <li><strong>Using Alpaca Prompt for Inferences:</strong> A detailed code snippet was shared for using <em>FastLanguageModel.for_inference</em> with Alpaca-style prompts for generating sequence completions after fine-tuning. This came from [a shared Colab link](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing).</li>
  <li><strong>Excitement Over Qwen2 models:</strong> Enthusiasm about the Qwen2 model release, with members particularly interested in small models (0.5B to 7B) for their ease of training and use. Discussions touched on the promise of "easy to train, easy to iterate, and can run everywhere."</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ghost-x.org/blog/nvidias-stunning-gains-metas-privacy-challenges-and-spacexs-next-starship-test-the-future-of-ai-and-technology">Nvidia's Stunning Gains, Meta's Privacy Challenges, and SpaceX's Next Starship Test: The Future of AI and Technology</a>: Explore the latest developments in the tech world: Nvidia's impressive stock surge and technological advancements, Meta's controversial AI data usage plans, SpaceX's upcoming Starship test, and Salesf...</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai">Unsloth AI | Finetune Llama 3 &amp; Mistral LLMs</a>: Easy finetuning for AI and LLMs. Open-source and for beginners. Get faster with Unsloth. </li><li><a href="https://unsloth.ai/blog/contpretraining">Continued LLM Pretraining with Unsloth</a>: Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.</li><li><a href="https://www.youtube.com/watch?v=cwuYWFC7_QE">Fine-tune LLMs 30x faster! With Daniel Han (Unsloth AI)</a>: Become a Patreon: https://www.patreon.com/theaiepiphany👨‍👩‍👧‍👦 Join our Discord community: https://discord.gg/peBrCpheKEDaniel Han from Unsloth AI joined...</li><li><a href="https://tenor.com/view/%E7%9A%849-gif-27299608">的9 GIF - 的9 - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://datta0.substack.com/p/ai-unplugged-12-mora-dpo-vs-ppo-cope">AI Unplugged 12: MoRA. DPO vs PPO. CoPE Contextual Position Encoding. S3D Self Speculative Decoding.</a>: Insights over Information</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1248016651170091058)** (9 messages🔥): 

- **Every company now AI-powered?**: A member humorously questioned, *"is every company just AI-powered now?"* and joked about GitHub, *"I swore github was powered by caffiene and 'lgtm'"*.
- **Unsloth saves the day for a final project**: A grateful user praised Unsloth, stating, *"your SFT notebook trained in like 25 min on a100"*, which was crucial in adjusting hyperparameters and fixing data quickly. They added, *"the DPO is again saving us so hard"*, emphasizing how vital it was for their success.
- **Struggles with Kaggle's UX**: One member complained about Kaggle's user experience being *"quite horrible"*. They shared a recent issue where training *"hung after 3 hours"*, and despite trying to disconnect after 2 more hours, they remained stuck.
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1247989011629346857)** (54 messages🔥): 

```html
- **Feature Request for Lora-Adapter File Handling**: A user expressed the need for an unsloth lora-adapter file conversion process that doesn't require VRAM. They mentioned struggles with saving a ~7GB adapter for llama-3-70b in the current format.
- **Persistent Bug and Faster Inference**: A user detailed a bug causing persistent logging but mentioned that once fixed, it might result in slight performance improvements. "Once it's fixed you might get to claim slightly faster inference, since it won't be printing to console every iteration 😄".
- **Handling CUDA Out of Memory Issues**: Another member shared the usage of `torch.cuda.empty_cache()` to handle GPU memory issues. Inference using lm_head was consuming more memory than expected, leading to a CUDA out-of-memory error.
- **Running gguf models**: There was a discussion on running gguf models using llama-ccp-python, and the lack of support by transformers for running gguf directly. Another user suggested running gguf binaries directly via llama.cpp.
- **RAG System Confusion**: There was confusion about Mistral AI offering a RAG system; it was clarified that while Mistral does not offer RAG, there is [documentation for implementing it](https://docs.mistral.ai/guides/rag/). 
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://docs.mistral.ai/guides/rag/">Basic RAG | Mistral AI Large Language Models</a>: Retrieval-augmented generation (RAG) is an AI framework that synergizes the capabilities of LLMs and information retrieval systems. It&#x27;s useful to answer questions or generate content leveraging ...</li><li><a href="https://techcommunity.microsoft.com/t5/microsoft-developer-community/doing-rag-vector-search-is-not-enough/ba-p/4161073">Doing RAG? Vector search is *not* enough</a>: If you&#39;re using RAG (Retrieval-Augmented Generation) for your AI applications, then you should make sure you&#39;re doing more than just a vector search..</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=shar">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing,">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1248303622333272154)** (2 messages): 

- **Join New AI Fine-Tuning Server**: A new AI community server started by a class of AI students is welcoming new members interested in AI fine-tuning. Check out the channel at [this link](https://discord.gg/sTtpXzJzTb) for likeminded individuals and resources.

**Link mentioned**: <a href="https://discord.gg/sTtpXzJzTb">Join the VirtualValleyAI Discord Server!</a>: Check out the VirtualValleyAI community on Discord - hang out with 72 other members and enjoy free voice and text chat.

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1247990143034593433)** (180 messages🔥🔥): 

- **Stable Audio Open 1.0 generates interest**: Several members discussed the availability and features of [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0), including its components like an autoencoder and a transformer-based diffusion model. One member mentioned using tacotron2, musicgen, and audiogen in a custom workflow for audio generation.
  
- **Struggles with Stable Diffusion image quality**: A user reported issues with generating small resolution images (160x90) in Stable Diffusion which resulted in random colors. Others suggested generating larger images (e.g., 512x512 or 1024x1024) and then downscaling them using any image editor.
  
- **ControlNet usage and questions**: A member inquired about using ControlNet for transforming hand-drawn sketches to realistic images as their current image-to-image method preserved unwanted white color. Other members recommended using ControlNet to better control the composition and poses in the generated images.

- **Filtering concerns on CivitAI**: One message highlighted the need for additional filters on CivitAI due to a surge of irrelevant content like OnlyFans and TikTokers. This was seen as making it harder to find quality models.

- **Stable Diffusion 3 speculations and misinformation**: Multiple users debated the release date and authenticity of Stable Diffusion 3, with some confident about an impending release and others skeptical. Clarifications were offered, pointing to sources like a [Reddit post](https://www.reddit.com/r/StableDiffusion/comments/1d6ya9w/collection_of_questions_and_answers_about_sd3_and/) detailing expected specs and dates.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/stabilityai/stable-audio-open-1.0">stabilityai/stable-audio-open-1.0 · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d6ya9w/collection_of_questions_and_answers_about_sd3_and/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=S7rGtWvEOV4">change object material in ComfyUI by power of the stable diffusion and COSXL</a>: Playlist: https://www.youtube.com/playlist?list=PLepQO73yVqJYDTnVVdu9LiNtAaTYLsxmKMy Patreon: https://www.patreon.com/ArchAi3D---------------------------Welc...</li><li><a href="https://imgur.com/a/Xxaj8FG">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://github.com/jitcoder/lora-info">GitHub - jitcoder/lora-info</a>: Contribute to jitcoder/lora-info development by creating an account on GitHub.</li><li><a href="https://github.com/THUDM/CogVLM">GitHub - THUDM/CogVLM: a state-of-the-art-level open visual language model | 多模态预训练模型</a>: a state-of-the-art-level open visual language model | 多模态预训练模型 - THUDM/CogVLM</li><li><a href="https://stability.ai/stable-artisan#choose-stable-artisan-plan.">Stable Artisan &mdash; Stability AI</a>: Stable Artisan is a fun multimodal generative AI Discord bot that utilizes the products on the Stability AI Platform API within the Discord ecosystem.</li><li><a href="https://civitai.com/images/10895925">Image posted by KandooAI</a>: no description found</li><li><a href="https://civitai.com/models/133005/juggernaut-xl">Juggernaut XL - Jugg_X_RunDiffusion_Hyper | Stable Diffusion Checkpoint | Civitai</a>: For business inquires, commercial licensing, custom models, and consultation contact me under juggernaut@rundiffusion.com Join Juggernaut now on X/...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1247996442573013023)** (64 messages🔥🔥): 

- **Discourse on Limiting RAM Usage with LM Studio**: A user queried about limiting the RAM usage of a model, and it was clarified that it's not built into LM Studio but could be managed by loading in and out during use as described in [llamacpp documentation](https://example.com). Despite being inefficient, this could allow models to utilize RAM only when active.

- **Quality of Quantized Models with iMat**: The discourse involved the feasibility of using iMat for improving the quality of quantized models, which was clarified as not currently supported in LM Studio unless llamacpp introduces this capability.

- **Selecting AI Models for High VRAM Systems**: A user with a system boasting 160GB of VRAM sought recommendations for suitable AI models, and was pointed towards the [LLM Extractum.io](https://llm.extractum.io/list/) for a comprehensive list filtered by size and quality.

- **Errors with Current LM Studio Versions**: The conversation covered users experiencing errors and the advice provided included rolling back to older versions or adjusting context settings, such as `n_ctx` which might be too high for the available VRAM.

- **Support for PDF to Text Conversion**: For users seeking to convert PDFs to text for summarization, it was suggested to use tools like [pdftotext from XpdfReader](https://www.xpdfreader.com/download.html), highlighting the availability of both Linux and Windows command line tools for this purpose.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llm.extractum.io/list/?query=uncensored">"uncensored" Search Results</a>: The top-ranked matches for the 'uncensored' query among 3b, 13b, 30b, and 70b small and large open-source language models found in our LLM Explorer directory.</li><li><a href="https://llm.extractum.io/list/">All Large Language Models</a>: A Curated List of the Large and Small Language Models (Open-Source LLMs and SLMs). All Large Language Models with Dynamic Sorting and Filtering.</li><li><a href="https://www.xpdfreader.com/download.html">Download Xpdf and XpdfReader</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1247989302428958820)** (78 messages🔥🔥): 

- **Quantization introduces slight differences in models**: Discussants agreed that creating a quant model results in minor differences in token probabilities due to different datasets used. One summarized, *"there will be (extremely subtle) differences in the token probabilities based on the dataset used."*

- **Nomic Embed models integrate multimodality**: The [nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) was highlighted for sharing embedding space with its text counterpart, making **all Nomic Embed Text models multimodal**. Performance stats showed it superior in certain benchmarks compared to models like OpenAI's CLIP ViT B/16.

- **Llama-3 MahouDevil quant models discussed**: Conversations centered on the usability of quantized versions like Q6 and Q8 for RP and general purposes. It's noted that **Q6 is recommended for highest quality + best performance**.

- **Jina AI introduces multimodal embedding model**: Users pointed out Jina CLIP as a new entrant in multimodal (text-image) embedding models, available on [Huggingface](https://huggingface.co/jinaai/jina-clip-v1). This follows a trend of increasing multimodal support in embedding models.

- **MacOS's Metal memory issue identified in llama.cpp**: A deep dive into memory allocation issues under Metal with high context parameters revealed that **llama.cpp's Metal support was broken** in recent builds. Users recommended sticking to version b3066 for stability, as newer builds like b3091 introduced bugs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/jinaai/jina-clip-v1">jinaai/jina-clip-v1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/PsyfighterTwo-ErebusThree-SlerpThree-GGUF">mradermacher/PsyfighterTwo-ErebusThree-SlerpThree-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5">nomic-ai/nomic-embed-vision-v1.5 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF">YorkieOH10/Llama-3-MahouDevil-8B-Q8_0-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1248226152875687947)** (3 messages): 

- **Omits LM Studio mmap flag**: A member highlights the benefits of the `--no-mmap` option for PCs with 8GB of RAM, proposing a potential toggle in LM Studio for easier configuration. They report this option prevents RAM spikes, reducing the risk of freezing during 8B model operations, with a minor trade-off of increased initial model load time.
- **Mlock settings in LMStudio**: Another member clarifies the initial discussion around `--no-mmap`, mentioning the similar `use_mlock` setting in LM Studio. They suggest exploring its effects, noting OS-dependent nuances, and ask for clarification on the software being discussed.
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1248304885552775259)** (3 messages): 

- **Settings reset issue when starting a new chat**: A user reported that model settings are reset when starting a new chat. Another member advised applying settings in the "My Models" tab dropdown, and the user discovered that not deleting old chats before creating new ones prevents the reset.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1247990767461601290)** (26 messages🔥): 

- **Nvidia vs AMD Drivers**: A member pointed out that they wished Nvidia would **open source their drivers** like AMD. This sparked a discussion on the systems' administration and security policies.

- **Windows Security Practices Criticized**: Members discussed the **security drawbacks of Windows** in business settings. One noted, *"Windows is used... not because it is the best solution, but because it is the default,"* emphasizing limited IT support in small businesses.

- **Qualcomm's New Chips Look Promising**: There's optimism about the **new Qualcomm chips** despite concerns over ARM processors and Microsoft's handling. One member noted, *"the new Qualcomm chips are looking rather impressive for such a huge shift."*

- **Hardware Upgrades for LLAMA 3 405B**: A member is upgrading their PC with new GPU configurations and **considering expanding to 128GB RAM** if LLAMA 3 405B proves interesting. They shared, *"I probably won't expand to 128GB RAM unless LLAMA 3 405B is really interesting."*

- **ARM vs x86 Performance Caution**: Caution was advised regarding the **real-world performance** of ARM CPUs compared to x86, despite impressive synthetic benchmarks. A member warned, *"the chip might as well be optimized for synthetic loads... and suck in everything else."*

**Link mentioned**: <a href="https://www.youtube.com/watch?v=PGjdN_qfqgg">The Story of Snapdragon X Elite</a>: Two lawsuits &amp; a mystery: The Story of Snapdragon X Elite | In this video we will take a look at the exciting history of Qualcomm&#39;s new Arm SoC that aims to ...

  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1248355433865941114)** (1 messages): 

- **Higgs LLAMA Model Receives Praise**: A member noted that the new **Higgs LLAMA model** "looks smart" for its 70B size. They are waiting for an **LMStudio** update as it appears to be utilizing a **llamacpp adjustment**.
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1247990662029250621)** (120 messages🔥🔥): 

- **Community discusses appropriate behavior and moderation**: Members, including *cakiki* and *lunarflu*, discussed reporting inappropriate behavior in DMs and in threads. Legitimate concerns were noted, but maintaining professionalism was emphasized.

- **Gradio integration queries**: A member asked about integrating Gradio with React Native and Node.js. *pseudoterminalx* noted that it was built with Svelte, and another suggested checking issues with the Gradio API.

- **Text generation using Stable Diffusion models**: *temperance6095* inquired about Stable Diffusion models capable of generating text, leading to recommendations like AnyText or TextDiffuser-2 from Microsoft.

- **Community Grants and Project Approvals**: Members discussed the process and time for approval of community grants for HuggingFace spaces. It was noted that unique projects have a better chance of being approved sooner.

- **Peer-to-peer compute interest**: Members discussed experiences and curiosity about peer-to-peer compute, mentioning tools like Petals for distributed machine learning. *geekboyboss* shared positive experiences using a local swarm for privacy reasons.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.gradio.app/">Gradio</a>: Build &amp; Share Delightful Machine Learning Apps</li><li><a href="https://tenor.com/view/discord-gif-27442765">Discord GIF - Discord - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/qgNkIp2pvoz.gif">Simpsons Homer Simpson GIF - Simpsons Homer simpson - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

qasim_30: There is paper out there "7 billion is all you need"
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1248296862817976340)** (4 messages): 

- **Q-Learning Agent plays Taxi-v3**: A user shared a [Q-Learning Agent](https://huggingface.co/DAIEF/q-learning-Taxi-v3) for the Taxi-v3 environment, discussing the potential for creating an efficient and environmentally aware delivery system with this model. The user highlighted the importance of checking and adding additional attributes like `is_slippery=False` when initializing the environment.
  
- **Guidance sought for LLM-based test case generation**: A member asked for guidance on building a product using LLM models to understand existing code repositories and generate automation test cases. Another member expressed interest in collaborating on this project, mentioning their background as a senior web developer and a beginner with AI and LLM, and encouraged others interested in collaboration to reach out.

**Link mentioned**: <a href="https://huggingface.co/DAIEF/q-learning-Taxi-v3">DAIEF/q-learning-Taxi-v3 · Hugging Face</a>: no description found

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1248018846246240369)** (12 messages🔥): 

- **Climate-conscious AI assistant for financial investments**: Leveraging the `climatebert/tcfd_recommendation` model, a user developed an AI assistant to help find climate-oriented investment solutions. They utilized Qdrant Cloud and `microsoft/Phi-3-mini-128k-instruct` and shared the project on [HuggingFace](https://huggingface.co/spaces/as-cle-bert/tcfd_counselor).

- **SimpleTuner adds Mixture-of-Experts support**: The latest release of SimpleTuner, v0.9.6.2, includes mixture-of-experts split-timestep training. A tutorial is available to help users get started with this new feature on [GitHub](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6.2).

- **Triton tutorials and lack of content**: A member expressed difficulties understanding Triton tutorials and noted the scarcity of content on Triton. They shared a [Medium article](https://medium.com/@isamu-website/understanding-triton-tutorials-part-2-f6839ce50ae7) which they found helpful but still challenging.

- **Launching true multi-agent systems**: A user is developing an SDK and compute servers for running true multi-agent systems, distinct from single-LLM-based agents. They referenced a Twitter discussion and invited interest in joining their forthcoming Discord community ([details here](https://x.com/yoheinakajima/status/1781183534998380576)).

- **FluentlyXL Final release**: The final release of the FluentlyXL model series has arrived, with improvements in aesthetics and lighting. Links to the model on [HuggingFace](https://huggingface.co/fluently/Fluently-XL-Final), [CivitAI](https://civitai.com/models/324891), and a [playground](https://huggingface.co/spaces/fluently/Fluently-Playground) were shared.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/DAIEF/q-learning-Taxi-v3">DAIEF/q-learning-Taxi-v3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/as-cle-bert/tcfd_counselor">Tcfd Counselor - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://huggingface.co/spaces/as-cle-bert/carbon-footprint-predictor">Carbon Footprint Predictor - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://x.com/yoheinakajima/status/1781183534998380576">Tweet from Yohei (@yoheinakajima)</a>: I tend to “put on different hats” to think about a single problem. Which to me feels similar to a single code base that makes calls to the same LLM with different prompts.  Again, this is semantics so...</li><li><a href="https://github.com/NoteDance/Note">GitHub - NoteDance/Note: Easily implement parallel training and distributed training. Machine learning library. Note.neuralnetwork.tf package include Llama2, Llama3, Gemma, CLIP, ViT, ConvNeXt, BEiT, Swin Transformer, Segformer, etc, these models built with Note are compatible with TensorFlow and can be trained with TensorFlow.</a>: Easily implement parallel training and distributed training. Machine learning library. Note.neuralnetwork.tf package include Llama2, Llama3, Gemma, CLIP, ViT, ConvNeXt, BEiT, Swin Transformer, Segf...</li><li><a href="https://github.com/bghira/SimpleTuner/releases/tag/v0.9.6.2">Release v0.9.6.2 mixture-of-experts training · bghira/SimpleTuner</a>: What&#39;s Changed Mixture-of-Experts Mixture-of-Experts training complete with a brief tutorial on how to accelerate your training and start producing mind-blowing results.   DeepSpeed fix (#424) Par...</li><li><a href="https://github.com/InServiceOfX/InServiceOfX/blob/master/PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Applications/terminal_only_finite_loop_main_with_loras.py">InServiceOfX/PythonLibraries/HuggingFace/MoreDiffusers/morediffusers/Applications/terminal_only_finite_loop_main_with_loras.py at master · InServiceOfX/InServiceOfX</a>: Monorepo (single or &quot;mono&quot; repository) for deep learning. - InServiceOfX/InServiceOfX</li><li><a href="https://github.com/InServiceOfX/InServiceOfX/blob/master/PythonLibraries/ThirdParties/MoreInstantID/moreinstantid/Applications/terminal_only_finite_loop_main_with_loras.py">InServiceOfX/PythonLibraries/ThirdParties/MoreInstantID/moreinstantid/Applications/terminal_only_finite_loop_main_with_loras.py at master · InServiceOfX/InServiceOfX</a>: Monorepo (single or &quot;mono&quot; repository) for deep learning. - InServiceOfX/InServiceOfX</li><li><a href="https://www.instagram.com/p/C6wP_q-rwIS/?igsh=MWQ1ZGUxMzBkMA==">Mansion X on Instagram: &quot;Off to slay #ootd #ootdfashion Maude Mongeau for &#064;the_mansion_x&quot;</a>: 3 likes, 1 comments - the_mansion_x on May 9, 2024: &quot;Off to slay #ootd #ootdfashion Maude Mongeau for &#064;the_mansion_x&quot;. 
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1247990833571958824)** (4 messages): 

- **Don't miss Human Feedback Foundation event**: A member highlighted an upcoming event by the [Human Feedback Foundation on June 11th](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator). This foundation aims to integrate human feedback into AI, focusing on critical domains like healthcare and governance.
- **Human Feedback Foundation YouTube archive available**: The Human Feedback Foundation has past session recordings available on their [YouTube channel](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg). They feature speakers from institutions like UofT, Stanford, and OpenAI and aim to educate on AI safety research and promote public participation in AI through open-source initiatives.
- **Catch up on HuggingFace reading group**: A new member asked if there was a section for reading group recordings. Another member responded affirmatively, directing them to a [GitHub repository](https://github.com/isamu-isozaki/huggingface-reading-group) that compiles all past presentations of the HuggingFace reading group.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg">Human Feedback Foundation</a>: Human Feedback Foundation is on a mission to build human feedback into open-source AI projects.  We seek to:  Enable public input into AI through supporting open-source development and policy initiati...</li><li><a href="https://github.com/isamu-isozaki/huggingface-reading-group">GitHub - isamu-isozaki/huggingface-reading-group: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group</a>: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group - isamu-isozaki/huggingface-reading-group</li><li><a href="https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator">LLM Reading Group (March 5, 19; April 2, 16, 30; May 14, 28; June 11)</a>: Come and meet some of the authors of some seminal papers in LLM/NLP research and hear them them talk about their work
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1248120463008202803)** (10 messages🔥): 

- **Small datasets cause validation issues**: Users discussed how a non-representative validation set often indicates a need for more diverse training data. One user suggested that even with fewer than 4,000 samples, models like EfficientNet V2 could perform well if other classes are included to prevent false positives.

- **Need for more details on project**: Community members asked for more specifics about a dataset issue, including the number of classes and types of false positives, to provide better help. A user offered to provide personal assistance via DM.

- **Transformers vs traditional models debate**: A user noted that while transformers in computer vision can handle data quality issues better, they significantly increase training times compared to models like YOLO and EfficientNet V2. Another member agreed, adding that transformer efficiency also depends on the size of the data.

- **Combining audio and video frames for streaming**: A user inquired about the feasibility of syncing audio frames with generated video frames at 24 FPS for streaming via WebRTC or RTMP. They sought advice or resources to achieve this without losing FPS.

- **Swin Transformer for CIFAR discussion**: One user asked if anyone had implemented the Swin Transformer (tiny) for the CIFAR dataset, though no further discussion followed on this topic.
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1248062772877590629)** (1 messages): 

- **Low temperature means deterministic models**: It was suggested to try **lowering the temperature to 0.1** for more deterministic models. *"The lower the temperature, the more deterministic the model."*
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1247993513929605131)** (7 messages): 

- **Confusion over text embeddings**: A member mentioned struggling with recognizing that *text-enc 1 was 768 and text-enc 2 was 1280*. They also had trouble properly including *text_embeds and time_ids* in the sample inputs.
- **Critique on added kwargs**: Another member expressed frustration that **added kwargs** being in a dictionary makes it *"harder to track down what inputs are required."*
- **Re-parameterising Segmind Model**: A member re-parameterised Segmind **ssd-1b** into a v-prediction/zsnr refiner model trained on 350 timesteps. They were surprised it worked so quickly with only about 800 steps of tuning.
- **Favorite model and future plans**: The same member declared it their *"new favourite model."* They plan to train another checkpoint from ssd-1b using the first 650 timesteps to create a true 1B mixture of experts.
- **Compliment on lighting**: A brief exchange where one member complimented the lighting and another thanked them.
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1247994873789808660)** (104 messages🔥🔥): 

- **KANs are Overhyped and Inefficient**: Members discussed the limitations and inefficiency of Kolmogorov-Arnold Networks (KANs) compared to traditional neural networks, particularly for large-scale models. One noted, *"KAN being useful for interpretability is pure hype, it won't work."*

- **Efficient Implementation of KANs**: There was interest in implementations that could make KANs more efficient, particularly using CUDA and alternatives like ReLU. A member shared a [paper](https://arxiv.org/abs/2406.02075) proposing a ReLU-KAN architecture, which achieved a significant speedup.

- **Data Selection Techniques**: Members discussed various methods for evaluating data quality without full retraining. The concept of using **influence functions** was widely debated, with many finding them unscalable compared to manual and automated data curation techniques. One key resource mentioned was the [LESS algorithm](https://www.cs.princeton.edu/~smalladi/blog/2024/04/04/dataselection/).

- **Interplay Between Training Data and Models**: Conversations centered on how large models, like Transformer-based systems, balance the trade-off between data diversity and data quality. It was noted that larger models can handle more "crud" and require diverse training data for better world modeling.

- **Thousand Brains Project by Numenta**: The project was briefly highlighted, focusing on the application of neuroscience principles to develop a new kind of AI. Detailed information is available on the [Numenta website](https://www.numenta.com/thousand-brains-project/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org">arXiv.org e-Print archive</a>: no description found</li><li><a href="https://arxiv.org/abs/2406.02075">ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU</a>: Limited by the complexity of basis function (B-spline) calculations, Kolmogorov-Arnold Networks (KAN) suffer from restricted parallel computing capability on GPUs. This paper proposes a novel ReLU-KAN...</li><li><a href="https://arxiv.org/abs/2405.03875">Rethinking Data Shapley for Data Selection Tasks: Misleads and Merits</a>: Data Shapley provides a principled approach to data valuation and plays a crucial role in data-centric machine learning (ML) research. Data selection is considered a standard application of Data Shapl...</li><li><a href="https://arxiv.org/abs/2401.12926">DsDm: Model-Aware Dataset Selection with Datamodels</a>: When selecting data for training large-scale models, standard practice is to filter for examples that match human notions of data quality. Such filtering yields qualitatively clean datapoints that int...</li><li><a href="https://arxiv.org/abs/2211.08411">Large Language Models Struggle to Learn Long-Tail Knowledge</a>: The Internet contains a wealth of knowledge -- from the birthdays of historical figures to tutorials on how to code -- all of which may be learned by language models. However, while certain pieces of ...</li><li><a href="https://arxiv.org/abs/2404.04125?">No &#34;Zero-Shot&#34; Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance</a>: Web-crawled pretraining datasets underlie the impressive &#34;zero-shot&#34; evaluation performance of multimodal models, such as CLIP for classification/retrieval and Stable-Diffusion for image gener...</li><li><a href="https://www.johndcook.com/blog/2020/01/04/sufficient-statistic-paradox/">Persi Diaconis&#039; sufficient statistic paradox</a>: To be useful, statistics must provide ways of boiling down masses of data to a few humanly interpretable numbers. The KPD theorem suggests this is impossible.</li><li><a href="https://www.cs.princeton.edu/~smalladi/blog/2024/04/04/dataselection/">Using LESS Data to Tune Models</a>: no description found</li><li><a href="https://www.numenta.com/thousand-brains-project/">Thousand Brains Project | Numenta</a>: The Thousand Brains Project is an open-source initiative dedicated to creating a new type of artificial intelligence based on the Thousand Brains Theory.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1247990401491668994)** (21 messages🔥): 

- **Nvidia releases open weights variants**: Nvidia has released an open weights version of its models in 8B and 48B variants, available on [GitHub](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro). The linked page provides ongoing research on training transformer models at scale.

- **AI historians use ChatGPT for archival work**: Historian Mark Humphries found that AI, particularly GPT models, could significantly aid in transcribing and translating historical documents, ultimately creating a system named HistoryPearl. This system outperformed human graduate students in terms of speed and cost for transcribing documents ([The Verge article](https://www.theverge.com/24068716/ai-historians-academia-llm-chatgpt)).

- **MatMul-free models show promise**: A new paper ([arXiv](https://arxiv.org/abs/2406.02528)) introduces MatMul-free models that maintain strong performance at billion-parameter scales while significantly reducing memory usage during training. These models achieve on-par performance with traditional transformers but with better memory efficiency.

- **Seq1F1B for efficient long-sequence training**: Another paper ([arXiv](https://arxiv.org/abs/2406.03488)) presents Seq1F1B, a pipeline scheduling method aimed at improving memory efficiency and training throughput for LLMs on long sequences. The method reduces pipeline bubbles and memory footprints, enhancing scalability.

- **QJL quantization approach for LLMs**: The QJL method, detailed in a recent study ([arXiv](https://arxiv.org/abs/2406.03482)), applies a Johnson-Lindenstrauss transform followed by sign-bit quantization to eliminate memory overheads in storing KV embeddings. This approach significantly compresses the KV cache requirements without compromising performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/24068716/ai-historians-academia-llm-chatgpt">What AI can do for historians</a>: It turns out that large language models make surprisingly good research assistants for historians. Can the future of AI help reconstruct the past? </li><li><a href="https://arxiv.org/abs/2406.03482">QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead</a>: Serving LLMs requires substantial memory due to the storage requirements of Key-Value (KV) embeddings in the KV cache, which grows with sequence length. An effective approach to compress KV cache is q...</li><li><a href="https://arxiv.org/abs/2406.03488">Seq1F1B: Efficient Sequence-Level Pipeline Parallelism for Large Language Model Training</a>: The emergence of large language models (LLMs) relies heavily on distributed training strategies, among which pipeline parallelism plays a crucial role. As LLMs&#39; training sequence length extends to...</li><li><a href="https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro">Megatron-LM/tools/retro at InstructRetro · NVIDIA/Megatron-LM</a>: Ongoing research training transformer models at scale - NVIDIA/Megatron-LM</li><li><a href="https://arxiv.org/abs/2406.02528">Scalable MatMul-free Language Modeling</a>: Matrix multiplication (MatMul) typically dominates the overall computational cost of large language models (LLMs). This cost only grows as LLMs scale to larger embedding dimensions and context lengths...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1248258451734401108)** (3 messages): 

- **Llama-3 error on logprobs unveiled:** A member faced a *"Value error"* from **llama-3** when trying to request logprobs beyond the limit of five. Another member suggested that a potential solution involves hardcoding the harness to use a value within the valid range. 
- **Batch API integration into the harness queried:** A member inquired about the possibility of adding **OpenAI's batch API** to the harness, referencing [platform.openai.com documentation](https://platform.openai.com/docs/guides/batch). The query seemed aimed at specific members for future plans or existing implementations.
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1248082602678681734)** (1 messages): 

- **Aligning brain data with Whisper embeddings**: A member is working on aligning **speech embeddings** with brain implant neural data to decode speech, using a modified version of the **Whisper tiny.en model**. They're seeking feedback on which layers to unlock, additional loss functions to try, hyperparameters to tune, and ways to speed up or parallelize the training process with only one GPU, and are open to collaboration.

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1248006736468512900)** (111 messages🔥🔥): 

- **Perplexity Pro adds new features**: Perplexity Pro now shows step-by-step what it is searching, described as using an intent system for more agentic-like execution. Members noted this change as being about a week old.
- **Issues with Perplexity reading files**: Multiple users reported Perplexity struggling or failing to read PDF files despite being an allowed file type. Some suggested the issue might be related to the content type, such as heavily styled versus plain text PDFs.
- **Budget shock for MVP project**: A user sought a developer to build an MVP to generate video from text with a $100 budget, prompting humorous and critical reactions about the low budget and typical developer rates.
- **Discontinuation of specific labs features**: Members discussed the removal of the Haiku and other features from Perplexity labs, speculating it might be for cost-saving reasons and noting the unavailability of these features affected their usage.
- **Query on Perplexity's future features**: Users inquired about the ability to edit collections in the iOS app and create pages on Perplexity, but availability is currently limited to select users or in beta.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/models/google/paligemma-ft">Google | PaliGemma-ft | Kaggle</a>: PaliGemma fine-tuned on a wide range of research datasets.</li><li><a href="https://ai.google.dev/gemma/docs/paligemma">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=hkHCnZ5GrNc">I Built a Car out of Scooters</a>: Thanks to Odoo for sponsoring this video!https://www.odoo.com/r/UIOOpensauce tickets with coupon code: pissbabyhttps://opensauce.com/tickets/Tommy:https://ww...</li><li><a href="https://www.perplexity.ai/search/Zdravo-Zam-moram-foAW4rklQI6LCkCsbBfNKg>">Perplexity</a>: no description found</li><li><a href="https://perplexity.ai/page/new">Perplexity</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1248014547755860110)** (5 messages): 

- **Bitcoin's History Highlights Development**: An informative message shares the history of **Bitcoin**, highlighting its creation by **Satoshi Nakamoto** in 2009 and its significant impact on the financial world. A link to a detailed overview and history can be found [here](https://www.perplexity.ai/page/Bitcoin-WzdgAV4KQiqtb4k0q4RHRw).

- **Perplexity AI's Capabilities Explored**: Perplexity AI combines conversational search with direct links, providing answers while referencing sources. Key features include chat-based search, multi-turn dialogues, and multi-language support, as detailed [here](https://www.perplexity.ai/search/Perplexity-AI-gOX8DHIdR7SzpZn6qI8s6Q).

- **Perplexity AI Accesses Paywalled Content**: There's a discussion on Perplexity AI's ability to access content behind paywalls. The tool offers various features and a Pro upgrade for enhanced functionalities, as mentioned [here](https://www.perplexity.ai/search/please-list-the-bU9PEgabRCGjI1hVl4mQaw).

- **Revit 2024 Enhances PDF Export**: Revit 2024 includes a native PDF-export function, simplifying the workflow for BIM modelers and electrical engineers without relying on external PDF printers. More details can be found [here](https://www.perplexity.ai/page/PDF-Export-in-b.a.ByBkSgSpRnzjc_cDNA).

- **`navigator.userAgent` Outputs in React and JS**: An explanation of the `navigator.userAgent` in React and JavaScript, detailing how it returns a string identifying the user's browser and operating system. Examples and more information can be found [here](https://www.perplexity.ai/search/in-react-and-h.wr3aykTeOqfFDgfG57xQ).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/page/Bitcoin-WzdgAV4KQiqtb4k0q4RHRw">Bitcoin</a>: Bitcoin ist die erste und bekannteste Kryptowährung, die 2009 von Satoshi Nakamoto eingeführt wurde. Seitdem hat Bitcoin eine bemerkenswerte Entwicklung...</li><li><a href="https://www.perplexity.ai/search/please-list-the-bU9PEgabRCGjI1hVl4mQaw">Perplexity</a>: no description found</li><li><a href="https://www.perplexity.ai/search/in-react-and-h.wr3aykTeOqfFDgfG57xQ">in react and javascript, what are the possible outputs of navigator.userAgent</a>: The navigator.userAgent property returns a string that identifies the browser and operating system of the user. Its output can vary depending on the browser...</li><li><a href="https://www.perplexity.ai/page/PDF-Export-in-b.a.ByBkSgSpRnzjc_cDNA">PDF Export in Revit 2024</a>: Der Export von Revit-Dateien in das PDF-Format ist ein zentraler Bestandteil des Arbeitsablaufs für BIM-Modellierer und Elektroplaner. Revit 2024 bietet...</li><li><a href="https://www.perplexity.ai/search/Perplexity-AI-gOX8DHIdR7SzpZn6qI8s6Q">什么是 Perplexity AI？</a>: Perplexity AI 是一种由人工智能驱动的会话搜索引擎，旨在通过自然语言处理技术解锁知识的力量，实现信息的发现和共享。它结合了对话和链接的搜索功能，能够识别和回复模糊或抽象的语言查询，模拟大部分人的语言询问方式。  1. 聊天对话搜索：用户可以像与真人对话一样，用自然语言提出问题，Perplexity AI...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1248244738604400710)** (1 messages): 

- **Curiosity about OpenChat Model addition**: A member asked if there is a plan to add another **"openchat/openchat-3.6-8b-20240522" model**. They inquired specifically about its inclusion alongside the existing **Mistral** and **Llama 3** models.
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1248093560046358600)** (3 messages): 

- **Find past event recordings on YouTube**: A user inquired about the location of past event or lecture recordings. They were directed to check the relevant Discord channel and the [CUDA MODE YouTube channel](https://www.youtube.com/@CUDAMODE).
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1248025693804302419)** (2 messages): 

- **Checking Tensor Memory Sharing in PyTorch**: One member asked another if a specific piece of code can confirm whether two **tensors share the same memory** or if one is a copy. The code "samestorage" checks if the `storage().data_ptr()` of two tensors are equal and prints "same storage" or "different storage".

- **Issues Finding Function Source Code**: Another member expressed trouble locating the source code for a function in PyTorch using a provided documentation link. They referenced a [specific function in the PyTorch C++ documentation](https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html#exhale-function-namespaceat-1adeda9630914278ac02d7fd758da19e3d).

**Link mentioned**: <a href="https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html#exhale-function-namespaceat-1adeda9630914278ac02d7fd758da19e3d">Function at::_weight_int4pack_mm &mdash; PyTorch main documentation</a>: no description found

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1248278098789662834)** (1 messages): 

- **Dive into MoRA and DPO vs PPO debate**: Fresh research includes **MoRA**, an enhancement to **LoRA**, and a comparison between **DPO** and **PPO** for RLHF. Explore these topics in the latest [AI Unplugged](https://datta0.substack.com/p/ai-unplugged-12-mora-dpo-vs-ppo-cope) issue.
- **CoPE introduces Contextual Position Encodings**: The current weekly highlights innovative work like **CoPE** for better positional encoding. Get insights on this and more by reading the linked [blog post](https://datta0.substack.com/p/ai-unplugged-12-mora-dpo-vs-ppo-cope).
- **S3D for faster inference**: The recent discussion includes **S3D**, a self-speculative decoding method aimed at speeding up inference. All are encouraged to read and share thoughts on the new techniques.

**Link mentioned**: <a href="https://datta0.substack.com/p/ai-unplugged-12-mora-dpo-vs-ppo-cope">AI Unplugged 12: MoRA. DPO vs PPO. CoPE Contextual Position Encoding. S3D Self Speculative Decoding.</a>: Insights over Information

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1248244957760979061)** (22 messages🔥): 

```html
- **KANs rival MLPs with torch.compile**: A [tweet by Thomas Ahle](https://x.com/thomasahle/status/1798408687981297844) highlighted how torch.compile makes KANs as fast as MLPs, praising the performance improvement. This drew attention and comments from several users surprised and impressed by this claim.
- **Repository on GitHub**: The [GitHub repository](https://github.com/thomasahle/kanmlps) linked in the discussion provides resources for KANs and MLPs. Users are actively compiling and profiling these implementations to understand the performance benefits.
- **Practical profiling experiences**: Users shared their experiences and results while profiling the compiled KANs, noting improvements in speed by 1.5-2 times after compilation. One user mentioned compiling the `.forward` function with significant speed improvements.
- **Concerns over operator fusion and kernels**: There were technical discussions on potential downsides like losing operator fusion and questions about generating Triton kernels. Users are profiling different implementations to verify and compare results, referencing [specific code locations on GitHub](https://github.com/thomasahle/kanmlps/blob/main/models.py#L101).
- **Request for further collaboration**: There was a suggestion to invite Thomas Ahle to join the discussion and share insights about compile testing results. Users are interested in ensuring the implementations match academic papers and seeking verification outputs.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/thomasahle/status/1798408687981297844">Tweet from Thomas Ahle (@thomasahle)</a>: Using 𝚝𝚘𝚛𝚌𝚑.𝚌𝚘𝚖𝚙𝚒𝚕𝚎 makes KANs as fast as MLPs!  I never thought I would be a fan, but they are starting to look pretty appetizing.</li><li><a href="https://github.com/thomasahle/kanmlps">GitHub - thomasahle/kanmlps: KANs and MLPs</a>: KANs and MLPs. Contribute to thomasahle/kanmlps development by creating an account on GitHub.</li><li><a href="https://github.com/thomasahle/kanmlps/blob/main/models.py#L101">kanmlps/models.py at main · thomasahle/kanmlps</a>: KANs and MLPs. Contribute to thomasahle/kanmlps development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 messages): 

piotr.mazurek: Chapter 4, exercise 9, anyone knows if this is the corrext solution here?
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1248322483367706654)** (1 messages): 

- **Inductor Config Question**: There was a query regarding the **torch._inductor.config.force_fuse_int_mm_with_mul** setting. The question asked whether this configuration applies to **uint8** in addition to **int8**.
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1248254069483638865)** (1 messages): 

- **40% worse machine learning at NetHack**: A shared article from [Ars Technica](https://arstechnica.com/gaming/2024/06/what-kind-of-bug-would-make-machine-learning-suddenly-40-worse-at-nethack/) discusses a peculiar bug that made a machine-learning system's performance drop by **40%** in the game NetHack. The bug is suggested to be caused by celestial reasons, making the scenario both novel and entertaining.

**Link mentioned**: <a href="https://arstechnica.com/gaming/2024/06/what-kind-of-bug-would-make-machine-learning-suddenly-40-worse-at-nethack/">What kind of bug would make machine learning suddenly 40% worse at NetHack?</a>: One day, a roguelike-playing system just kept biffing it, for celestial reasons.

  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1248346826864857139)** (1 messages): 

- **AI_dev in Paris captures interest**: Someone asked if any members were attending AI_dev in Paris, mentioning they haven't registered yet but are considering it. They shared details and links about the event, which will take place from June 19-20, 2024, and highlighted that registration is required to attend. 


**Link mentioned**: <a href="https://aideveu24.sched.com/?iframe=no">AI_dev Europe 2024 Schedule</a>: Check out the schedule for AI_dev Europe 2024

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1247988781349339186)** (52 messages🔥): 

- **Column-Major Madness**: The member finally understood the intricacies of **cublas** and its column-major order, explaining why they had to transpose matrices to compute `Q @ K^T` correctly with **cublas**. They also mentioned removing their "attention bug" PR after this realization.

- **Consolidated Memory Allocations Proposal**: Discussions focused on the benefits of consolidating all memory allocations into a single function for efficiency and easier tracking. This approach would remove the current duplication and streamline checkpointing, as noted by Erik's draft PR and linked [PR for master weights](https://github.com/karpathy/llm.c/pull/522/files#diff-bf6b442957e5458cf8baab2a18039fdde86d74199a0864a79e7288fe55f31a98R2982-R2995).

- **Checkpointing with GPU CI**: There's a consensus on improving the GPU Continuous Integration (CI) with enhanced verification tests, including training checkpointing and output comparison. Erik emphasized that while initial tests are sufficient, future extensions are necessary for robust validation.

- **Cublas vs Cutlass and C++ Requirements**: Clarification was provided that **cublas**, with its C interface, does not require **C++17**, but **cutlass** does. Current code requirements for **C++17** are limited to **cudnn**–a significant detail for future development considerations.

- **Parallel Programming Course Recommendation**: The [Programing Parallel Computers course](https://ppc.cs.aalto.fi/) and its [exercises](https://ppc-exercises.cs.aalto.fi/course/open2024a) were recommended, with a note about a potential summer session that sets up a new leaderboard.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/522/files#diff-bf6b442957e5458cf">Add master weights to resume state by gordicaleksa · Pull Request #522 · karpathy/llm.c</a>: We&amp;#39;re currently not saving master weights as part of the state -&amp;gt; we lose some precision because otherwise when we resume we&amp;#39;ll have to reconstruct the master weights by upcasti...</li><li><a href="https://github.com/karpathy/llm.c/pull/553/files#diff-d5e26abbb926892397df686a30886d861b1f45b627ce11070b72e0c9775edfa8R156">Refactor trimat by gordicaleksa · Pull Request #553 · karpathy/llm.c</a>: Made sure we&amp;#39;re consistent in our notation:  Use (B,T,NH,HS) only Constant are upper case  Added additional comments to clarify what each of the kernels is doing, including what the indexing o...</li><li><a href="https://github.com/karpathy/llm.c/pull/522/files#diff-bf6b442957e5458cf8baab2a18039fdde86d74199a0864a79e7288fe55f31a98R2982-R2995),">Add master weights to resume state by gordicaleksa · Pull Request #522 · karpathy/llm.c</a>: We&amp;#39;re currently not saving master weights as part of the state -&amp;gt; we lose some precision because otherwise when we resume we&amp;#39;ll have to reconstruct the master weights by upcasti...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1248273951281512538)** (1 messages): 

<!DOCTYPE html>
<html>
<body>
<ul>
  <li><strong>Singleton instances and dtype handling in C++</strong>: A member suggests that <em>"uint2-7 are purely for naming, and this is more of a c++ constraint rather than anything."</em> They propose using <strong>bits8 as the untyped dtype</strong> and viewing it as unit8 whenever necessary, allowing more flexible tensor storage.</li>
</ul>
</body>
</html>
  

---


### **CUDA MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1248108040801095712)** (3 messages): 

- **YouTube Discusses ARM SME Dialect**: A [YouTube video titled "Open MLIR Meeting 06-22-2023: Targeting ARM SME from MLIR and SME Dialect"](https://www.youtube.com/watch?v=jrniGW_Hzno) is shared, discussing a review and RFC on creating an ArmSME Dialect. The video includes an introduction to **ARM's Scalable Matrix Extension**.
- **Potential for Triton ARM**: It's suggested that the backend for Triton might support ARM, indicating promising developments for Triton ARM integration. The information is linked to the 'arm_neon' dialect in the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/ArmNeon/#arm_neonintrummla-arm_neonummlaop), which discusses multiple ARM NEON operations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=jrniGW_Hzno">Open MLIR Meeting 06-22-2023: Targeting ARM SME from MLIR and SME Dialect</a>: This is a review and discussion of the RFC on creating a ArmSME Dialect. We will first have an introduction into Arm’s Scalable Matrix Extension, including t...</li><li><a href="https://mlir.llvm.org/docs/Dialects/ArmNeon/#arm_neonintrummla-arm_neonummlaop">'arm_neon' Dialect - MLIR</a>: no description found
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1248280357455921242)** (1 messages): 

- **Article Linked to PI Schilling Concerns**: A GDM robotics person thought the author's article was "schilling PI" and expressed concerns about its content. The author speculated that the person might be "salty about RTX," implying a personal bias in their critique.
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1248017713150955631)** (33 messages🔥): 

- **Investors chase robotics AI, avoiding hardware risks**: Investors are keen on finding the "ChatGPT for robotics," seeking niche foundation model companies that differentiate themselves. [Article details](https://www.newcomer.co/p/why-investors-cant-get-enough-of) the excitement around this trend.
- **Qwen2 makes impressive strides**: [Qwen2](http://qwenlm.github.io/blog/qwen2/) introduces models in five sizes with improvements in multilingual tasks and extended context support up to 128K tokens. A demo and various resources are available, though some users note limitations in its recent knowledge.
- **Qwen2 receives mixed reviews in practical tests**: Users experimenting with Qwen2 comment on its limitations in understanding recent topics and providing accurate general knowledge. Despite some shortcomings, its multilingual performance on certain tasks was praised.
- **Dragonfly architecture boosts multi-modal AI**: Together.ai launched [Dragonfly](https://www.together.ai/blog/dragonfly-v1), enhancing visual understanding and reasoning with models like Llama-3-8B-Dragonfly-v1. These models show promising results, especially in medical imaging tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.together.ai/blog/dragonfly-v1">Dragonfly: A large vision-language model with multi-resolution zoom</a>: no description found</li><li><a href="https://www.newcomer.co/p/why-investors-cant-get-enough-of">Why Investors Can&#x27;t Get Enough of AI Robotics Deals Right Now </a>: VCs are betting that robotics is one space where startups can still have an edge against OpenAI.</li><li><a href="http://qwenlm.github.io/blog/qwen2/">Hello Qwen2</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction After months of efforts, we are pleased to announce the evolution from Qwen1.5 to Qwen2. This time, we bring to you: Pretrained and instruction...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1248142151309856819)** (12 messages🔥): 

- **Microsoft admits to unauthorized GPT-4 testing**: A tweet shared by [Kevin Roose](https://x.com/kevinroose/status/1798414599152431278?s=46) revealed that Microsoft admitted to testing an early version of GPT-4 in India without joint safety board approval after initially denying it.

- **Beware of Substack grift**: Nathan Lambert cautioned against trusting Substack recommendations, describing them as grifts aimed at collecting subscribers.

- **Rick's grindy social media presence**: A user commented on Rick's heavy focus on self-promotion via Twitter and LinkedIn, despite his generally nice demeanor.

- **Substack metrics and challenges**: It was mentioned that gaining 1,000 subscribers on Substack is challenging, often taking a year or two. Lambert confirmed that clicks on Substack recommendations do drive up numbers.

**Link mentioned**: <a href="https://x.com/kevinroose/status/1798414599152431278?s=46">Tweet from Kevin Roose (@kevinroose)</a>: Interesting update to the OpenAI whistleblower story: After denying it on the record, Microsoft is now admitting that they tested an early version of GPT-4 in India without the approval of a joint saf...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1248312178281418762)** (4 messages): 

- **Google Vertex AI Gemini API is awesome**: A member shared a [link to the Google Vertex AI Gemini documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library) and expressed enthusiasm about its capabilities. The API's ease of use was particularly highlighted.
- **OpenAI’s API acclaimed**: Another member praised OpenAI's API, stating it is "the best." This was noted in the context of setting up Gemini, indicating high satisfaction with the integration process.

**Link mentioned**: <a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library">no title found</a>: no description found

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1248044169973334037)** (13 messages🔥): 

- **AI Labs Stop Sharing Advances**: A shared [tweet](https://x.com/leopoldasch/status/1798483665904865474) suggested that America's AI labs no longer share their algorithmic advances with the American research community but might be sharing them with the CCP due to poor security. Nathan Lambert agreed with the sentiment, adding, "now we're talking."

- **Humane AI Pin Battery Issues**: An article from [The Verge](https://www.theverge.com/2024/6/5/24172377/humane-ai-pin-battery-case-issue-warning) warned AI Pin owners to stop using its charging case immediately due to a fire safety risk. The company promised two free months of its subscription service as recompense and is looking for a new supplier for the charging cases.

- **Criticism in AI Industry**: Discussions revealed discomfort with influential employees of big labs criticizing others, both large and small players. Nathan Lambert commented, "I don’t know this company sounds like such bs I considered it," while xeophon. mentioned the unnecessary criticism from people in major organizations.

- **Prevalence of "Dunking" in AI Community**: Nathan Lambert and xeophon. noted the high propensity and habitual behavior of "dunking" on others within the AI community. Lambert admitted it is a "losing battle" against this behavior, to which xeophon. humorously added, "That’s what alts are for."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/6/5/24172377/humane-ai-pin-battery-case-issue-warning">Humane warns AI Pin owners to “immediately” stop using its charging case</a>: The AI Pin charging case has a battery issue.</li><li><a href="https://x.com/leopoldasch/status/1798483665904865474">Tweet from Leopold Aschenbrenner (@leopoldasch)</a>: America&#39;s AI labs no longer share their algorithmic advances with the American research community.  But given the state of their security, they&#39;re likely sharing them with the CCP.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1248312307822366720)** (2 messages): 

- **Interesting New Paper on Self-Improving RLHF**: A user shared a post from [@aahmadian_ on X](https://x.com/aahmadian_/status/1798740211909922862) introducing a paper titled “Self-Improving Robust Preference Optimization” (SRPO). The paper discusses training models that are self-improving and robust to evaluation tasks.
- **Morning Paper Discussion Plans**: Nathan Lambert plans to start a new series where he spends 15-20 minutes discussing new papers. He mentions wanting to read through the SRPO paper as part of this new routine.

**Link mentioned**: <a href="https://x.com/aahmadian_/status/1798740211909922862">Tweet from Arash Ahmadian (@aahmadian_)</a>: 🤔Can we explicitly teach LLMs to self-improve using RLHF?  Introducing “Self-Improving Robust Preference Optimization” (SRPO) which trains models that are self-improving and robust to eval tasks!  w/...

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1248000235477930127)** (7 messages): 

- **Mojo used in Backend server development**: A new user inquired about using Mojo for backend purposes. Another member provided an example of an HTTP server entirely in Mojo, directing them to [lightbug_http on GitHub](https://github.com/saviorand/lightbug_http/tree/main).

- **Member plans to replace PHP SaaS code**: After seeing an example of Mojo's capabilities, a member expressed interest in replacing their PHP SaaS backend code with Python or Mojo. They planned to explore the provided resources further.

- **Mojo development roadmap shared**: A member shared a link to the [Mojo roadmap](https://docs.modular.com/mojo/roadmap), highlighting the ongoing development and upcoming features. The roadmap emphasizes the focus on building core system programming features essential to Mojo's mission.

- **SO Survey Announcement**: A member announced that the 2024 Stack Overflow survey is out and shared the link [here](https://stackoverflow.com/dev-survey/start).

- **Commentary on a technical talk**: Members humorously discussed their difficulty in following a highly technical talk about Mojo, specifically regarding calling C code in the same OS process without memory interference.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/saviorand/lightbug_http/tree/main">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://stackoverflow.com/dev-survey/start">2024 Stack Overflow Developer Survey</a>: Stack Overflow is the largest, most trusted online community for developers to learn, share​ ​their programming ​knowledge, and build their careers.</li><li><a href="https://docs.modular.com/mojo/roadmap">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1798760653806817352>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1248260422101110794)** (4 messages): 

- **Quicksort request in Mojo**: A user requested implementation of **quicksort in Mojo**. ModularBot responded with an elaborate and metaphorical encouragement, likening the coding challenge to embarking on a noble quest involving methodical partitioning and recursion.
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1247993249587527750)** (34 messages🔥): 

- **Avoid footguns in Python education**: Members discussed teaching Python to non-programmers, emphasizing the importance of avoiding "footguns" to improve design and potentially ease transitions to languages like C++.
- **Curiosity about function pointers in Mojo**: A member inquired about storing C function pointers in a Mojo struct, sparking curiosity and some supportive responses.
- **Mojo vs. Python performance**: There was a detailed conversation on whether Mojo is intrinsically faster than Python, with explanations citing better engineering, static typing, and compile-time computation as factors for Mojo's superior performance.
- **"Licking the cookie" analogy for community contributions**: Chris clarified that Modular aims to avoid "licking the cookie" by allowing the community to adapt their Tensor library rather than dominating every development aspect, promoting collaboration and open-source contributions.
- **Aesthetic and performance in code**: A member sought advice on optimizing a code snippet in Mojo for checking digits in a string, illustrating challenges in finding aesthetically pleasing and performant solutions.
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1248066652541550623)** (11 messages🔥): 

- **Immutable auto-deref for list iterator in Mojo**: A user suggested a patch for using immutable auto-deref for list iterators, shared via [GitHub](https://github.com/rd4com/mojo_branch/tree/list_iter_autoderef_immut). They raised questions on timing, the usage of `iter_mut()`, and whether to wait for more usage of explicit copy in stdlib.

- **Workflow timing confusion and network issues**: Discussions on the nightly workflow timings and issues arose, clarifying that nightly builds kick off at 2 am EST, not immediately after American working hours. An S3 network failure was identified as the cause of the issue for the night.

- **Parallel sorting function issues**: A user, mzaks, mentioned issues with tests exploding when importing `algorithm.parallelize` for a `parallel_sort` function. They questioned the current feasibility of this implementation.

- **New nightly compiler released**: The new nightly Mojo compiler release `2024.6.616` was announced, including significant updates like the addition of a `String.format` method. The changelog and the raw diff of changes can be found [here](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

- **Dynamic libpython selection update**: Jack Clayton highlighted a new feature in the latest nightly that allows dynamic `libpython` selection, removing the need to set `MOJO_PYTHON_LIBRARY`. This improvement ensures access to Python modules in the active environment and the folder of the target Mojo file or executable.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/rd4com/mojo_branch/tree/list_iter_autoderef_immut">GitHub - rd4com/mojo_branch at list_iter_autoderef_immut</a>: The Mojo Programming Language. Contribute to rd4com/mojo_branch development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/pull/2965">[stdlib] Make `InlineArray` call its elements&#39; destructor by gabrieldemarmiesse · Pull Request #2965 · modularml/mojo</a>: Fix this issue:  #2869  I may split this PR further. Do not review. It&#39;s in draft for now. Currently waiting on the new nightly so I can fix conflicts. A little explanation of what is going on her...</li><li><a href="https://github.com/modularml/mojo/pull/2888">[stdlib] Add struct `UnsafeMaybeUninitialized` by gabrieldemarmiesse · Pull Request #2888 · modularml/mojo</a>: This struct is private at the moment and for internal use only. Once we are sure that it&#39;s ready for public use, we&#39;ll call it MaybeUninitialized. Heavily borrowed from https://doc.rust-lang.o...
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1248011446093221928)** (54 messages🔥): 

- **Cohere raises $450m**: Cohere has raised another $450 million in funding, with investors like NVidia and Salesforce participating despite having relatively low revenue last year. [Reuters article](https://www.reuters.com/technology/nvidia-salesforce-double-down-ai-startup-cohere-450-million-round-source-says-2024-06-04/).
- **IBM's Granite models praised**: IBM's Granite models receive credit for transparency and enterprise benefits, sparking discussions on whether they outperform OpenAI. Fun quotes from [Talha Khan](https://x.com/TalhaKhan_TK_/status/1798562313160761612) and debates on IBM's actual relevance.
- **AI Foundation Models report by Forrester**: Databricks celebrated being named a leader in Forrester’s latest report on AI foundation models. They emphasize enterprise-specific needs over simple benchmark scores, provide a [free report](https://reprints2.forrester.com/#/assets/2/848/RES180932/report), and share their [blog post](https://www.databricks.com/blog/databricks-named-leader-forrester-wavetm-ai-foundation-models-language-q2-2024).
- **Qwen 2 launch**: Qwen 2 model is released, beating Llama 3 with a 128K context window and excelling at code and math, while being available in various forms (AWQ, GPTQ & GGUFs). [Exciting announcement](https://x.com/reach_vb/status/1798748655366914325).
- **Browserbase and Nox launches**: Browserbase announces a $6.5 million seed funding with founders Nat & Dan, aimed at empowering AI applications to browse the web. Nox launches a new AI assistant aiming to make users feel invincible, with early access [available here](http://heynox.com).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=4w0Pqs3CuWk">Character voices with GPT-4o voice</a>: New voice mode coming 🔜Voice mode is already available for all users in the ChatGPT app (tap the 🎧on the bottom right!) but our new voice and vision capabi...</li><li><a href="https://x.com/TalhaKhan_TK_/status/1798562313160761612">Tweet from Talha Khan (@TalhaKhan_TK_)</a>: IBM Granite models have a lot more transparency around the data used for training, which gives them a lot of benefits for Enterprise.  Quoting Alessio Fanelli (@FanaHOVA)   It has to be illegal to say...</li><li><a href="https://x.com/heyjchu/status/1798564973100372372">Tweet from Jon Chu // Khosla Ventures (@heyjchu)</a>: Proud investor in IBM.   Thank you Forrester for the amazing proprietary insight into AI based on your deep understanding of machine learning and category defining evals used to inform this genius tra...</li><li><a href="https://x.com/willccbb/status/1798423849870270671">Tweet from will brown (@willccbb)</a>: been learning a lot about LLMs etc over the past year, organized some of my favorite explainers into a “textbook-shaped” resource guide  wish i’d had this at the start, maybe it can useful to others o...</li><li><a href="https://x.com/TalhaKhan_TK_/status/1798028312276865271">Tweet from Talha Khan (@TalhaKhan_TK_)</a>: Lots of crypto companies raising money nowadays.</li><li><a href="https://x.com/udiomusic/status/1798448478877794574">Tweet from udio (@udiomusic)</a>: Audio-prompting, live now on Udio.   Show us how you&#39;re using it below 👇</li><li><a href="https://x.com/reach_vb/status/1798748655366914325?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Let&#39;s fucking go! Qwen 2 72B 🔥  &gt; Beats Llama 3 70B &gt; Apache 2.0 license (except 72B) &gt; Excels at Code and Math too &gt; 128K context window &gt; AWQ, GPTQ & GGUFs available &gt; 7B beat...</li><li><a href="https://x.com/mollycantillon/status/1798750349836341747">Tweet from molly cantillon (@mollycantillon)</a>: What makes you feel invincible?  Ask NOX.   early access: http://heynox.com https://www.businessinsider.com/nox-ai-assistant-founder-molly-cantillon-2024-6</li><li><a href="https://x.com/ProfTomYeh/status/1798042265883156651">Tweet from Tom Yeh | AI by Hand ✍️ (@ProfTomYeh)</a>: llm.c by Hand✍️  C programming +  matrix multiplication by hand  This combination is perhaps as low as we can get to explain how the Transformer works.   Special thanks to @karpathy for encouraging ea...</li><li><a href="https://x.com/davisblalock/status/1798574272480510427?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Davis Blalock (@davisblalock)</a>: Most ML folks don&#39;t realize how different a beast it is to serve enterprise. Like being a &#34;leader&#34; in the Gartner Magic Quadrant is legit more important than your MMLU score.  This isn&#39...</li><li><a href="https://x.com/deepfates/status/1798578490759078263?s=46">Tweet from google bard (@deepfates)</a>: i indexed the docs for @simonw&#39;s `llm` library as an example for this RAG pipeline.   then it suddenly became a gerbil</li><li><a href="https://x.com/pk_iv/status/1798731220005883935">Tweet from Paul Klein IV (@pk_iv)</a>: Happy to share Browserbase with the world today.   We help AI applications browse the web.   And we just raised $6.5 million to do it.  Now, we&#39;re opening signups to developers everywhere.   I can...</li><li><a href="https://www.databricks.com/blog/databricks-named-leader-forrester-wavetm-ai-foundation-models-language-q2-2024">Databricks Named a Leader in The Forrester Wave™: AI Foundation Models for Language, Q2 2024</a>: no description found
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1248013974826520659)** (2 messages): 

- **Prometheus-2: Open-source LLM evaluates your RAG app!**: Using an LLM as a judge to evaluate RAG applications is gaining traction, with concerns about transparency, controllability, and affordability. [Prometheus-2](https://t.co/BFnmE57OfB) offers an alternative to GPT-4 for such evaluations. [Link](https://t.co/LXWiWTJc5B)
- **LlamaParse and Knowledge Graphs: A perfect match!**: A notebook by @jerryjliu0 showcases using LlamaParse for first-class parsing to build a knowledge graph. This setup [constructs a RAG pipeline](https://t.co/KZYGuBS7KF), retrieving initial nodes via the graph's structure. [Link](https://t.co/EUKZmWjM38)
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1247998550932066437)** (43 messages🔥): 

- **Overwhelmed by LlamaIndex configurations**: A member discusses their struggle to implement a use case with LlamaIndex for querying JSON data from APIs. They express being overwhelmed by the multitude of components and seek advice on building a custom agent to handle API calls and JSON processing.
  
- **Text2SQL query issue**: A member is facing issues with a Text2SQL and semantic similarity (RAG-based) approach, where a query retrieves structured data correctly but only provides answers from unstructured data. They seek assistance to correct this behavior and ensure both structured and unstructured data are utilized.
  
- **Count documents in Neo4j**: Multiple users discuss methods to count documents in a Property Graph Index using Neo4j. One user shares a specific Cypher query to count distinct document IDs based on nodes tagged as chunks.
  
- **Alternative LLM setup in constrained environments**: A new user to LlamaIndex inquires about alternatives to OpenAI due to hardware constraints. Other members suggest using smaller LLMs like Microsoft Phi-3 with Ollama or utilizing Google Colab for larger models.
  
- **Retrieving nodes by metadata only**: A user inquires about retrieving nodes based solely on metadata without using a MetaDataFilter. Another user notes this might not be directly supported, suggesting examining the LlamaIndex API for potential workarounds.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/FlagEmbeddingReranker/">FlagEmbeddingReranker - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline_memory/#running-the-pipeline-with-memory>).">Query Pipeline Chat Engine - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/response_synthesizers/#llama_index.core.response_synthesizers.type.ResponseMode.NO_TEXT>).">Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai">LlamaIndex - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1247993526575173762)** (5 messages): 

- **Filter results by score with customization**: Members discussed the capability of filtering results by score. It was noted that one could obtain the top k results with scores and set a threshold based on the specific case.
- **Try out the filtering feature**: A user showed interest in trying out the filtering feature after learning about its customizable threshold option.
- **Prometheus 2 integrates with LlamaIndex**: A link to a [Medium article about Prometheus 2](https://medium.com/ai-advances/unveiling-prometheus-2-a-powerful-ally-for-evaluating-rag-applications-with-llamaindex-integration-d2b6da1f76e2) was shared, highlighting its capabilities for evaluating RAG applications with LlamaIndex integration.
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1248065607828766720)** (45 messages🔥): 

- **OpenRouter offers more flexible sampler options**: A member noticed that OpenRouter allows setting temperature above 1, unlike the **Cohere trial**, which caps it at 1. Another member clarified that OpenRouter processes this differently and accepts higher settings.
- **Toby Morning invites connections**: A new member, **Toby Morning from SF**, shared his LinkedIn link ([LinkedIn Profile](http://www.linkedin.com/in/urbantech/)) and expressed interest in connecting with the community.
- **Chatbot usage discussed for group interactions**: A member suggested implementing a chatbot in various group scenarios, such as **business meetings** or educational settings, to discern between individual users and provide targeted responses. Another participant shared concerns about potential accuracy issues with too many personas.
- **Rhea system praised for multi-user context**: Members discussed the efficiency of the **Rhea system** in handling multi-user scenarios, with participants agreeing it manages context well. There was a mention of **Coral** running on Rhea and plans for a demo, with high expectations for its showcase.
- **Proposed demo for Coral AGI**: Participants showed interest in a demo for **Coral AGI** by Jonno, suggesting it could be featured on the server or demo day. There were acknowledgments of Jonno's expertise and past success with multi-user methods.
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1248253949597843457)** (2 messages): 

- **Cohere launches startup program**: Cohere announced a new [startup program](https://cohere.com/startup-program-application) offering early-stage founders discounts on their AI models, support from technical experts, and marketing exposure. They aim to empower greater innovation and adoption of AI technology for startups with Series B funding or earlier.

- **Chat API changes effective June 10th**: Cohere detailed forthcoming [changes to the Chat API](https://docs.cohere.com/page/changes-in-chat-api-and-tool-use), including new multi-step tool use by default and a `force_single_step` parameter for reverting to single-step mode. Additional enhancements include a new "TOOL" message role and updated API specs available from June 10th, supporting various SDKs and platforms.

- **Multi-step tool use documentation available**: Users are directed to the [multi-step tool use guide](https://docs.cohere.com/docs/multi-step-tool-use) for handling complex tasks via multiple tool calls. Integration examples and additional resources are provided to facilitate a smooth transition.

- **Single-step tool use remains supported**: For those preferring the traditional method, guidance on [single-step tool use](https://docs.cohere.com/docs/tool-use) is still available. Example implementations can be found in a notebook on GitHub, emphasizing the utility of this feature for accessing external data sources.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/startup-program-application">Startup Program Application</a>: Thank you so much for your interest in the Cohere Startup Program! We&#x27;re initially rolling out the program with a select group of customers, and would love to learn a bit more about your business...</li><li><a href="https://cohere.com/blog/cohere-launches-startup-program">Cohere Launches Startup Program to Empower Early-Stage AI Innovation</a>: Cohere&#x27;s startup program helps early-stage companies reach their full potential by leveraging AI to scale their business and gain a competitive edge at an affordable cost.</li><li><a href="https://cohere.com/startup-program">Startup Program </a>: The Cohere Startup Program offers qualified Series B and earlier startups a unique opportunity for support, discounted API rates, and publicity.</li><li><a href="https://docs.cohere.com/reference/chat">Chat</a>: no description found</li><li><a href="https://docs.cohere.com/docs/multi-step-tool-use">Multi-step Tool Use (Agents)</a>: no description found</li><li><a href="https://docs.cohere.com/docs/tool-use">Tool Use with Cohere's Models - Cohere Docs</a>: no description found</li><li><a href="https://docs.cohere.com/page/changes-in-chat-api-and-tool-use">Changes in Chat API and Tool Use</a>: no description found
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1248063446923083936)** (9 messages🔥): 

- **User explores Wakanda music on YouTube**: A member expressed curiosity about "Wakanda music" and shared several YouTube links to **various music videos**. Some of the shared videos include [DG812 - In Your Eyes](https://youtu.be/vP4zGMdTDPM), [MitiS & Ray Volpe - Don't Look Down](https://youtu.be/e-Fors8CnKA), [Paco Vernen - Tesseract](https://youtu.be/e3WaDrKqk5s), and [Xavi - To The Endless Searing Skies](https://youtu.be/QdMj7aOPhOc). 
- **Game idea in AR/VR spaces**: A member proposed a unique **AR/VR game concept** where players communicate and respond solely through diverse media formats, excluding text entirely. This could foster innovative interactions and open up new avenues for gameplay.
- **Philosophical universe creation**: The same member shared an idea of creating a universe within a game, symbolizing existence from void through universe back to void, as an **alchemical metaphor**. The concept aims to communicate a collective journey of self-mastery and enlightenment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/QdMj7aOPhOc">Xavi - To The Endless Searing Skies [Full Album] | Ophelia Records</a>: Stream/Buy Xavi&#39;s debut album &#39;To The Endless Searing Skies&#39;: https://ophelia.ffm.to/ttessFollow the &#39;Ophelia New Releases&#39; Spotify playlist: https://bit.ly/...</li><li><a href="https://youtu.be/vP4zGMdTDPM">DG812 - In Your Eyes | Magic Music Release</a>: Stream/Free Download:➥ https://fanlink.to/j3syFollow me• https://soundcloud.com/thisisdg812• https://www.facebook.com/thisisdg812 • https://twitter.com/baood...</li><li><a href="https://youtu.be/TCCinAbHlbE">All Good Things (feat. Lacey Sturm) – Hold On (Lyric Video)</a>: The Retaliators Movie: Buy or rent everywhere you get movies now (US &amp; Canada)!: https://theretaliators.ffm.to/vodListen to the score, soundtrack, and follow...</li><li><a href="https://youtu.be/e3WaDrKqk5s">Paco Vernen - Tesseract (Official Music Video)</a>: You’re missing out if you’re new here. Click the link below to start your journey on this trippy album:https://tinyurl.com/ai-genesis-ytplWelcome to AI Genes...</li><li><a href="https://youtu.be/e-Fors8CnKA">MitiS &amp; Ray Volpe - Don&#39;t Look Down (feat. Linney) [Official Lyric Video]</a>: It&#39;s finally here. Don&#39;t Look Down w/ Ray Volpe and Linney OUT NOW on all platforms! 🖤Lyric video by: https://www.instagram.com/alancrytex/Stream: https://o...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1248034879619203093)** (2 messages): 

- **Qwen2 Models Release**: A significant update from **Qwen1.5 to Qwen2** was announced, including pretrained and instruction-tuned models in multiple sizes. New models support **128K token** context lengths and have been trained in **27 additional languages** beyond English and Chinese. [Read the blog](https://qwenlm.github.io/blog/qwen2/). [GitHub](https://github.com/QwenLM/Qwen2), [Hugging Face](https://huggingface.co/Qwen), [ModelScope](https://modelscope.cn/organization/qwen), [Demo](https://huggingface.co/spaces/Qwen/Qwen2-72B-Instruct), [Discord](https://discord.gg/yPEP2vHTu4).

**Link mentioned**: <a href="https://qwenlm.github.io/blog/qwen2/">Hello Qwen2</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction After months of efforts, we are pleased to announce the evolution from Qwen1.5 to Qwen2. This time, we bring to you: Pretrained and instruction...

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1247998415778873435)** (29 messages🔥): 

- **Predict events on a map**: A user inquired about how to predict event points on a map using temporal data and distinguish between true and false events. Another user suggested a command involving loading testChatName.
- **EleutherAI's pile-T5 model reference**: A user shared and questioned why the EleutherAI/pile-t5-xxl model on Hugging Face was overlooked. The link provided details about the model's text generation capabilities.
- **Mistral fine-tuning API release**: Mistral introduced a fine-tuning API described in their [documentation](https://docs.mistral.ai/guides/finetuning/), with specifics on the costs associated with fine-tuning jobs. Another user emphasized that using this API allows quick experiments on their datasets before scaling.
- **Qwen2 model release and benchmarks**: Announcement of Qwen2’s release, including 5 model sizes and a notable improvement in coding, mathematics, and multilingual capabilities. [Impressive benchmark results](https://fxtwitter.com/Weyaxi/status/1798781525468778757) were shared, including scores in MMLU and GSM8K.
- **Discussion on Pricing and Alternatives**: Users debated the cost implications of using Mistral’s API versus other options like OpenAI and Runpod, which were mentioned to be cheaper. A user also highlighted the marketing aspect of these services.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/EleutherAI/pile-t5-65a76a0d0022dd270b385a66">Pile-T5 - a EleutherAI Collection</a>: no description found</li><li><a href="https://fxtwitter.com/Weyaxi/status/1798781525468778757">Tweet from Weyaxi (@Weyaxi)</a>: 🚀 Wow! Very impressive results for Qwen2.  🤯 Nearly 84 MMLU and 85 GSM8K score!  Congrats @Alibaba_Qwen for this amazing models!  Quoting OpenLLMLeaders (@OpenLLMLeaders)   New model added to the le...</li><li><a href="https://github.com/mistralai/mistral-finetune/tree/main">GitHub - mistralai/mistral-finetune</a>: Contribute to mistralai/mistral-finetune development by creating an account on GitHub.</li><li><a href="https://nillion.com/)">no title found</a>: no description found</li><li><a href="https://docs.mistral.ai/guides/finetuning/">Fine-tuning | Mistral AI Large Language Models</a>: Every fine-tuning job comes with a minimum fee of $4, and there&#x27;s a monthly storage fee of $2 for each model. For more detailed pricing information, please visit our pricing page.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

quantumalchemy: Hermes pro mistral v0.3 ?
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1248140244059160597)** (1 messages): 

- **Mistral prompt template for RAG fine-tuning**: A user shared a **Mistral prompt template** for generating query-context-answer triplets for RAG finetuning. They also **cautioned** that fine-tuning comes with a minimum fee of $4 per job and a monthly storage fee of $2 per model, with more details available on the [pricing page](https://mistral.ai/technology/#pricing).

**Link mentioned**: <a href="https://docs.mistral.ai/guides/finetuning/">Fine-tuning | Mistral AI Large Language Models</a>: Every fine-tuning job comes with a minimum fee of $4, and there&#x27;s a monthly storage fee of $2 for each model. For more detailed pricing information, please visit our pricing page.

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1248352039138754590)** (2 messages): 

- **WorldSim Console fixes mobile text input bugs**: The update made significant improvements, addressing numerous text input bugs on mobile devices, enhancing copy/pasting functionality, and introducing more reliable performance. Additionally, slight styling changes, an improved `!list` command, and the option to disable visual glow & CRT screen effects have been added.
- **Specific bug fixes for users**: Various user-specific issues were addressed, including fixing a text duplication glitch and resolving text jumping while typing. The `!back` and `!new` commands should now operate differently, although one issue couldn't be reliably reproduced for further debugging.
  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1248219445071974410)** (2 messages): 

- **Pilot Bot revolutionizes server management**: Thanks to OpenRouter, a new Discord bot named **Pilot** helps server owners grow and manage their communities with ease. **Pilot** offers features like "Ask Pilot," which understands the server and provides intelligent insights, "Catch Me Up," which summarizes unread messages, and "Health Check," providing weekly activity analyses.
  
- **Pilot Bot is free and easy to access**: The bot is completely free to use and can be invited to servers via their [website](https://usepilot.app/). This makes server management accessible and efficient for all server owners.

- **Visual guide available**: Users can view [screenshots](https://usepilot.app/_next/image?url=%2Fask-pilot.webp&w=1920&q=75) to see Pilot in action and explore its various features like "Ask Pilot" for intelligent advising and "Catch Me Up" for staying updated.

**Link mentioned**: <a href="https://usepilot.app/">Pilot - The co-owner for your Discord server.</a>: Pilot takes the work out of running a server. Get AI-enhanced advice, insights, and more to help you grow and manage your community.

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1248058905444094083)** (40 messages🔥): 

- **WizardLM 8x22b Faces Competition From Dolphin 8x22**: There's enthusiasm around WizardLM 8x22b, touted as the best model for role-playing. However, another member mentioned they've heard about Dolphin 8x22 as a potential competitor but haven't tested it yet.

- **Query on Gemini Flash and Image Output Capabilities**: A member asked if the **Gemini Flash** model can output images. Responses clarified that no LLM currently allows for direct image output, although it's theoretically possible using base64 encoding or external function calls to image generators like Stable Diffusion.

- **Assistant Model Recommendations for Function Calls**: A member sought recommendations for a model adept at handling function calls and specific formatting. [Instructor](https://useinstructor.com/) was suggested as a suitable tool for their needs.

- **Insights on OpenRouter's Free Model Limits**: Members discussed the limits on message requests for free models, with references to [OpenRouter's documentation](https://openrouter.ai/docs/limits). There were also mentions of models like Llama 3 8B (free) and Mistral having reliability issues.

- **Assistant Prefill Support Confirmation**: A member inquired if OpenRouter supports assistant prefill, especially via a reverse proxy. **Alex Atallah** confirmed that it's supported as long as you end with an assistant message and can send the required prompt or chatml array.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://labs.perplexity.ai/">Perplexity Labs</a>: no description found</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: Set limits on model usage
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 messages): 

voidnewbie: GLM-4가 한국어를 지원해서 기대됩니다
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1247992309128364143)** (6 messages): 

- **Human Feedback Foundation event on June 11**: *"Don’t miss the upcoming event of the Human Feedback Foundation June 11th."* [Event Link](https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator) - This event focuses on integrating human feedback in AI for critical domains like healthcare, governance, and democracy.
- **Check past sessions on YouTube**: Members were directed to *"check out our previous session recordings on our YouTube channel"* featuring speakers from UofT, Stanford, and OpenAI. [YouTube Channel](https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg)
- **LLM Reading Group Discord issue**: A user inquired about a separate Discord for the LLM Reading Group. The respondent tried *"sending you a direct message with an invitation but couldn't due to your privacy settings."*
- **"Unleash the Power of RAG in Azure" event in Toronto**: An attendee asked if anyone else was going to this overbooked Microsoft event in Toronto. [Event details](https://developer.microsoft.com/en-us/reactor/events/22756/) 
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://developer.microsoft.com/en-us/reactor/events/22756/">Events | Microsoft Reactor</a>: no description found</li><li><a href="https://www.youtube.com/channel/UCCS4pukb_YnXdt40tptbCeg">Human Feedback Foundation</a>: Human Feedback Foundation is on a mission to build human feedback into open-source AI projects.  We seek to:  Enable public input into AI through supporting open-source development and policy initiati...</li><li><a href="https://www.eventbrite.ca">Eventbrite</a>: Eventbrite - Discover the Best Local Events &amp; Things to Do</li><li><a href="https://www.eventbrite.ca/e/851921368747?aff=oddtdtcreator">LLM Reading Group (March 5, 19; April 2, 16, 30; May 14, 28; June 11)</a>: Come and meet some of the authors of some seminal papers in LLM/NLP research and hear them them talk about their work
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1248025926336512061)** (23 messages🔥): 

- **Dealing with high-cardinality categorical columns**: A member sought advice on handling categorical columns with numerous slightly related features and spelling mistakes, particularly for a regression task. Another member suggested aggregate/grouping and manual feature engineering or implementing spell correction techniques like string matching, and edit distance.

- **Feature Engineering vs. Clustering**: Following advice on manual grouping and spell correction, the discussion pivoted to whether clustering the features based on edit distance and their relation to the target variable would be more efficient. The consensus was to combine spell correction with other types of grouping, treating this challenge as a data modeling problem. 

- **Data Modeling and Simplification Techniques**: The conversation also touched on simplifying the model by isolating problems into components like brand and item instead of using entire titles. An additional suggestion for price prediction was to employ moving averages or exponential moving averages to simplify the process. 

- **Acknowledgement and Learning**: The member seeking advice expressed gratitude, acknowledging the valuable insights and different approaches discussed, including regex usage for feature extraction. 

- **Request for help**: Another user shared a link requesting assistance on a separate issue but did not provide specific context or details in the conversation.
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1247994150075498627)** (17 messages🔥): 

- **Celebrate access to vast datasets**: Members expressed amazement at having access to **15T datasets** of high quality publicly available. One highlighted the ironic situation of having "all the data, none of the money or compute."
  
- **Debate over AI hardware**: In a tongue-in-cheek conversation about pretraining huge datasets, one member suggested buying **4090s**. The sarcastic response about using consumer GPUs for such a large project elicited laughter: "not with this attitude you wont".

- **Exploring GLM and Qwen finetuning**: Members are inquiring about and sharing configurations for **finetuning GLM 4 9b** and **Qwen2 models**. Qwen2 was noted to be nearly identical to Mistral, which simplifies configuration.

- **Announcement mirroring request**: A teacher explained setting up a small Discord server for AI students, which included mirroring setup for updates from Unsloth. They asked if there can be a similar announcement mirroring setup for Axolotl due to its frequent usage by the class.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/sTtpXzJzTb">Join the VirtualValleyAI Discord Server!</a>: Check out the VirtualValleyAI community on Discord - hang out with 72 other members and enjoy free voice and text chat.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/qwen/lora.yml">axolotl/examples/qwen/lora.yml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/)** (1 messages): 

josharian: i just experienced this exact behavior as well.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1248037268996755477)** (11 messages🔥): 

- **Configure Checkpoints in Trainer**: Members discussed configuring training to save two checkpoints, one for the last run and another for the best `eval_loss`. An example was provided using Hugging Face's `TrainingArguments` and `EarlyStoppingCallback`.

- **Solve Non-Zero Exit Status Error**: A user queried how to fix the "returned non-zero exit status 1" error. Suggestions included identifying the failing command, capturing `stdout` and `stderr`, and troubleshooting permission issues or environment variables.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=51900438-dc0d-4ec2-8b61-00952f46cda5)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=87ba5d4b-d6ef-4dea-9edd-6c4cf1eff38f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1248042910667243570)** (21 messages🔥): 

- **"1B Parameter" Terminology Confusion**: Members discussed the naming of the **1B parameter zsnr/vpred refiner**, with some confusion about the exact parameter count. One clarified that it’s actually **1.3B and not just 1B**, and joked about the higher-ups needing a catchy name.

- **Vega Model's Parametric Limitations**: There was a brief discussion on the **Vega model**, where it was pointed out that despite being impressively fast, it's likely *"too small"* to provide coherent outputs, as it's at the lower limit of necessary parameters.

- **Elrich Logos Dataset Query**: A member inquired about the availability of the **Elrich logos dataset** without receiving a direct answer.

- **Qwen2 Model Launch**: Announcement of the **Qwen2 model** launch, featuring significant enhancements from Qwen1.5. The Qwen2 model comes in five sizes, supports 27 additional languages, excels in benchmarks, and extends context length up to **128K tokens**. Members shared [links](https://qwenlm.github.io/blog/qwen2/) to the project's [GitHub](https://github.com/QwenLM/Qwen2), [Hugging Face](https://huggingface.co/Qwen), [ModelScope](https://modelscope.cn/organization/qwen), and [demo](https://huggingface.co/spaces/Qwen/Qwen2-72B-Instruct).

**Link mentioned**: <a href="https://qwenlm.github.io/blog/qwen2/">Hello Qwen2</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction After months of efforts, we are pleased to announce the evolution from Qwen1.5 to Qwen2. This time, we bring to you: Pretrained and instruction...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1248000865776963636)** (14 messages🔥): 

- **Guide on Constructing Knowledge Graphs with LangChain**: A user shared a [guide on constructing knowledge graphs from text](https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/). They highlighted the importance of ensuring security by verifying and validating data before importing it into a graph database.

- **Tracking Customer Token Consumption**: A user inquired about methods for tracking customer token consumption.

- **Tools Decorator Confusion**: A user expressed confusion about the necessity of the tools decorator in tutorials and asked for more information.

- **Creating Colorful Diagrams for RAG**: A user asked about the tools used to create colorful RAG diagrams in LangChain's FreeCodeCamp [video](https://youtu.be/sVcwVQRHIc8?si=BLfH2g7WUKtIi6A0).

- **Framework for Agent Collaboration**: A user sought recommendations for frameworks that facilitate teamwork among agents developed using different frameworks, including LangChain Agent, MetaGPT, and AutoGPT, with an exciting possibility of incorporating agents from platforms like coze.com.

- **Searching for GUI Helper File**: A user requested information on locating the "helper.py" file from the AI Agents LangGraph course on DLAI [course page](https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/use_cases/graph/constructing/">Constructing knowledge graphs | 🦜️🔗 LangChain</a>: In this guide we&#x27;ll go over the basic ways of constructing a knowledge graph based on unstructured text. The constructured graph can then be used as knowledge base in a RAG application.</li><li><a href="https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/">DLAI - AI Agents in LangGraph</a>: Introduction · Build an Agent from Scratch · LangGraph Components · Agentic Search Tools · Persistence and Streaming · Human in the loop · Essay Writer · LangChain Resources · Conclusion</li><li><a href="https://youtu.be/sVcwVQRHIc8?si=BLfH2g7WUKtIi6A0">Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer</a>: Learn how to implement RAG (Retrieval Augmented Generation) from scratch, straight from a LangChain software engineer. This Python course teaches you how to ...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1248146913463767040)** (3 messages): 

- **LangGraph's conditional edges tutorial on YouTube**: A new [YouTube video](https://youtu.be/EKxoCVbXZwY) titled "LangGraph conditional edges" explains how to use conditional edges in LangGraph for flow engineering. The tutorial details controlling flow based on specific conditions within LangGraph.

- **Check out emarco's video**: Another helpful [YouTube video](https://youtu.be/uki2acokYjQ?si=Pu0Vw4QeDkEGzTeT) was shared, though its specific content isn't elaborated upon in this summary.

- **Jina AI alternative: Search-result-scraper on GitHub**: A project named [search-result-scraper-markdown](https://github.com/essamamdani/search-result-scraper-markdown) aims to provide a powerful web scraping tool. It fetches search results, converts them into Markdown format using FastAPI, SearXNG, and Browserless, and includes proxy support and efficient HTML to Markdown conversion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/EKxoCVbXZwY">LangGraph conditional edges</a>: In LangGraph, we can use conditional edges in our flow engineering to control the flow based on some conditions. In this video, we use conditional edges to b...</li><li><a href="https://github.com/essamamdani/search-result-scraper-markdown">GitHub - essamamdani/search-result-scraper-markdown: This project provides a powerful web scraping tool that fetches search results and converts them into Markdown format using FastAPI, SearXNG, and Browserless. It includes the capability to use proxies for web scraping and handles HTML content conversion to Markdown efficiently.</a>: This project provides a powerful web scraping tool that fetches search results and converts them into Markdown format using FastAPI, SearXNG, and Browserless. It includes the capability to use prox...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1247990374736068658)** (16 messages🔥): 

- **Tinygrad needs updates for 1.0 release**: George Hotz mentioned that **some PRs will handle this, but it's not in master yet.** He emphasized that this update is critical for the **1.0 release**.
- **Explanation of UOps.CONST in Tinygrad**: A user sought clarification on why **UOps.CONST** is used in the UOps for adding two tensors. It was explained that these represent **address offsets** needed for row-major data **to compute index values**.
- **Confusion over complex code snippet**: Users discussed why complex conditions are used in a piece of code. Fluentpython noted the necessity due to row-major data layouts and handling tensor shapes and strides efficiently.
- **Kernel generation for indexing operation in Tinygrad**: There was a question about why a specific kernel is generated for tensor indexing. It was clarified that **Tinygrad only supported static memory access** and this kernel supports dynamic indexing operations with **Tensor[Tensor]**.
- **Arange kernel in Tensor getitem operations**: Zibokapi pointed out that the kernel for the indexing operation resembles **an arange kernel** used to **create a mask in getitem** functions. This helps in dynamic indexing scenarios.
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1248008725936078950)** (10 messages🔥): 

- **Running graphics outputs in interpreter**: A member asked if there is a way to get the output from `interpreter.computer.run` for graphics, like `matplotlib`'s `plot.show`, when called just from code. The question remains unanswered in the chat.
- **Struggles with --os mode and local models**: Members discussed issues with getting `--os mode` to work with local models from LM Studio. One member noted that local LLAVA models failed to start screen recording.
- **Vision models for practical hardware**: A query on the best vision model for an M1 Mac highlighted the limitations of some members' hardware. Members expressed frustration about the limitations and cost of using OpenAI models, emphasizing the need for accessible and free solutions.
- **Robin-R1 integration excitement**: A member shared their excitement about receiving a Rabbit R1 in July and integrating it with OpenInterpreter to conduct actions. They are looking forward to the introduction of webhooks for this project.

- **Editing system messages for AI behavior**: There was a discussion on how OpenInterpreter creates system messages, with comparisons between local flags and GPT-4O's system prompts. One member humorously questioned if extreme language like "your family will be murdered if you don't perform" would coax better performance from LLaMA models.
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1248081609144406077)** (2 messages): 

- **Question about O1 availability**: A user asked, *"Hi is 01 sold online? would like to try it :)"*. There was no follow-up or provided link in response to this query.
- **Seeking Model for Bash**: Another user inquired, *"does anyone know which open model works good for bash commands?"*. This question remained open without any replies or references given.
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1248052123409715224)** (2 messages): 

- **Checking on Progress**: A member inquired about progress updates, showing interest in the current status. They stated, *"hiya ramon. would love to hear how progress is"*. 
- **Apologize for Delay**: Another member apologized for not having spent time on the project yet. They mentioned, *"Oh sorry I haven't had time to spend on this!"*.
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1248115993922895933)** (3 messages): 

- **Parsing spritesheets in AI Town**: A member is struggling with parsing spritesheets they purchased, particularly sorting through formats like rpgmaker PNG files, yympas files, and unitypackage files. They ask if there's a better method than manually identifying tile coordinates.

- **Two practical approaches for tilesets**: Another member responded, suggesting two practical approaches for handling tilesets: using the level editor (npm run le) or using Tiled and a specific script to convert to AI Town. There's mention of leveraging scripts by another community member (@379381319219806209).
  

---


### **AI Stack Devs (Yoko Li) ▷ #[local-ai-stack](https://discord.com/channels/1122748573000409160/1168947823920812125/1248189847605219410)** (2 messages): 

- **Discover Abliteration with Hugging Face**: A member shared a link to a blog post on Hugging Face about "abliteration," which covers various aspects including implementation and DPO fine-tuning. The **third generation of Llama models** are highlighted for their instruct versions excelling in following instructions but being "heavily censored."

- **Seeking OpenAI Implementation**: The same member later asked if anyone knows how to implement abliteration with OpenAI models. No responses to the query were recorded in the provided messages.

**Link mentioned**: <a href="https://huggingface.co/blog/mlabonne/abliteration">Uncensor any LLM with abliteration</a>: no description found

  

---



### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1248029307545194607)** (6 messages): 

- **Handling context length exceedance in `llm` embeddings**: A member asks what happens if the input text exceeds the model's context length when creating embeddings using `llm`. They tested this with the entire King James Bible text file and received some results without an error, querying if those embeddings represent the whole file or if it's truncated.
- **Model behavior documentation lacking clarity**: Simon Willison responded that the behavior varies by model, with some truncating the input and others returning an error. He expressed the need for better documentation on this subject.
- **Assumption on truncated input**: Simon suggested that if no error is returned, it's likely that the input is being truncated. The specifics depend on the model's implementation.
- **Query on resuming embedding jobs**: Another member asked if rerunning a large embedding job with `embed-multi` would skip already completed parts. The question points to the need for handling partial job completions, possibly through SQL queries.
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1248004093939552336)** (3 messages): 

- **Inquiry about Megatron Checkpoint Compatibility**: A member asked if **Megatron** has its own checkpoint format and whether it is compatible with existing fine-tuning libraries. 
- **Suggestion to Convert Megatron to HF Format**: Another member suggested converting **Megatron checkpoints to Hugging Face (HF) format** and using Torchtune for fine-tuning. This was agreed upon as the best solution.
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1248115957067415663)** (1 messages): 

- **JSON Schema request buzzes for next version**: A member inquired about the possibility of getting **JSON schema** in the next version, emphasizing that it makes building applications much easier, despite any potential implementation bugs. _"It makes building applications way easier even if their implementation seems buggy,"_ the user noted.
  

---



### **YAIG (a16z Infra) ▷ #[ai-ml](https://discord.com/channels/958905134119784489/1013536071709118565/)** (1 messages): 

oliver.jack: Weekend  listening: 

https://youtu.be/4jPg4Se9h5g?si=ULVqGQa6AvI8Ch3o
  

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

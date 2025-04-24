---
id: 69043bc0-49aa-4a04-b74f-4e8ab5fc7005
title: 'GPT-4o: the new SOTA-EVERYTHING Frontier model (GPT4O version)'
date: '2024-05-13T22:58:05.906872Z'
original_slug: ainews-gpt-4o-the-new-sota-everything-frontier
description: >-
  OpenAI has launched GPT-4o, a multimodal model that integrates text, voice,
  and vision with significantly faster response times and improved non-English
  language support. Available to all ChatGPT users including free plans, GPT-4o
  is priced at half the cost of GPT-4-turbo and offers enhanced real-time
  interaction capabilities such as voice and video input/output.
tags:
  - openai
  - gpt-4o
  - multimodal
  - voice
  - vision
  - coding
  - non-english
  - chatgpt
---


<!-- buttondown-editor-mode: plaintext -->**Omnimodality is all you want.**

> AI News for 5/10/2024-5/13/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**426** channels, and **7769** messages) for you. 
Estimated reading time saved (at 200wpm): **763 minutes**.

Say hello to [GPT-4O](https://openai.com/index/hello-gpt-4o/)!

https://www.youtube.com/watch?v=DQacCB9tDaw

It turns out that the numerous leaks about a "Her" like-chatbot announcement were most accurate, with a [surprisingly "hot"](https://x.com/andykreed/status/1790082413428629843) voice but also the ability to [respond with (an average 300ms, down from ~3000ms) low latency](https://x.com/karmedge/status/1790084650582397118?s=46&t=90xQ8sGy63D2OtiaoGJuww), [have vision](https://x.com/MikeBuckleySF/status/1790095001604730935), [handle interruptions and sing](https://twitter.com/gdb/status/1790071008499544518), [speak faster or in pirate/whale](https://twitter.com/willdepue/status/1790078289023062255), and more. There's also a [waitlisted](https://t.co/CF3T8dg8oQ) new desktop app that has the ability to read from the screen and clipboard history that directly challenges the desktop agent startups like Multion/Adept.

But nobody leaked that this also comes with a new versioned model, now confirmed to be the "gpt2-chatbot" that was previewed on LMsys, that is [confirmed to be substantially](https://twitter.com/lmsysorg/status/1790097597056872603) above all other prior frontier models:

 ![image.png](https://assets.buttondown.email/images/525a8fb6-f7a8-483f-9e71-a0631327bf1f.png?w=960&fit=max) 

The [official blogpost](https://openai.com/index/hello-gpt-4o/) has a lot more video examples demonstrating the app and model, including new versions of image output that may or may not be Dall-E or some completely new thing:

 ![image.png](https://assets.buttondown.email/images/e9f9b896-36b6-4a38-98d5-9fca8006bb43.png?w=960&fit=max) 

Lots of people are making noise about the 3d object demo, but we can't be sure if that's just code generation since there were hidden steps in there.

To do this, OpenAI had to beat SOTA on everything all at once, including ASR and Vision:

 ![image.png](https://assets.buttondown.email/images/5299d061-85d5-49d5-94a5-9e40319cf33a.png?w=960&fit=max) 

![image.png](https://assets.buttondown.email/images/dde36221-d41c-45e7-8a5c-ddb8f3e45236.png?w=960&fit=max) 

The tiktokenizer update revealed an expanded [200k vocab size](https://twitter.com/swyx/status/1790081902415851542) that makes non-English cheaper/more native.

Lots more takes are flying, but as is tradition on Frontier Model days on AINews, we're publishing two editions of AINews. **You're currently reading the one where all Part 1 and Part 2 summaries are done by GPT4O** - the next email you get is the same but with GPT4T (update: it completed [here](https://buttondown.email/ainews/archive/ainews-gpt-4o-the-new-sota-everything-frontier-9515/), 74% slower than GPT4O). We envision that you will pull them up side by side ([like this](https://twitter.com/main_horse/status/1790099796193398831)!) to get comparisons on discords you care about to better understand the improvements/regressions.

 ![image.png](https://assets.buttondown.email/images/3b02c374-6b8b-436b-b587-0ac3fa2f7c8b.png?w=960&fit=max) 


---

**Table of Contents**

[TOC] 



---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**OpenAI Releases GPT-4o, a Multimodal Model with Voice and Vision Capabilities**

- **GPT-4o Capabilities**: [@sama](https://twitter.com/sama/status/1790065469296156715) introduced GPT-4o, OpenAI's new model which can **reason across text, audio, and video in real time**. It is described as **smart, fast, natively multimodal, and a step towards more natural human-computer interaction**. [@gdb](https://twitter.com/gdb/status/1790071008499544518) noted it is **extremely versatile and fun to play with**.
- **Availability and Pricing**: GPT-4o will be **available to all ChatGPT users, including on the free plan** according to [@sama](https://twitter.com/sama/status/1790065541262032904). In the API, it is **half the price and twice as fast as GPT-4-turbo, with 5x rate limits** [@sama](https://twitter.com/sama/status/1790066685698789837).
- **Improved Language Performance**: GPT-4o has **significantly improved non-English language performance**, including an improved tokenizer to better compress many languages, as noted by [@gdb](https://twitter.com/gdb/status/1790079398625808837).

**Key Demos and Capabilities**

- **Real-time Voice and Video**: GPT-4o supports **real-time voice and video input and output**, which feels very natural according to [@sama](https://twitter.com/sama/status/1790069224171348344). This feature will roll out to users in the coming weeks.
- **Coding Capabilities**: GPT-4o is especially adept at coding tasks, as highlighted by [@sama](https://twitter.com/sama/status/1790066235696206147) and [@sama](https://twitter.com/sama/status/1790070399947981110).
- **Emotion Detection and Voice Styles**: The model can **detect emotion in voice input** and **generate voice output in a wide variety of styles with broad dynamic range**, per [@sama](https://twitter.com/sama/status/1790071830427930788).
- **Multimodal Outputs**: GPT-4o can **generate combinations of audio, text, and image outputs**, enabling interesting new capabilities that are still being explored, according to [@gdb](https://twitter.com/gdb/status/1790077263708340386).

**Reactions and Implications**

- **Game-changing User Experience**: Many, including [@jerryjliu0](https://twitter.com/jerryjliu0/status/1790069687025336754) and [@E0M](https://twitter.com/E0M/status/1790069805925404966), noted that the **real-time audio/video input and output represents a huge step change in user experience** and will lead to more people conversing with AI.
- **Comparison to Other Models**: GPT-4o was compared to other models, with [@imjaredz](https://twitter.com/imjaredz/status/1790074937119482094) stating it **blows GPT-4-turbo out of the water in terms of speed and quality**. However, [@bindureddy](https://twitter.com/bindureddy/status/1790076854076060066) pointed out that **open-source models like Llama-3 are still 5x cheaper** for pure language/coding use-cases.
- **Impressive Demos**: People were impressed by demos showcasing GPT-4o's **real-time translation abilities** [@BorisMPower](https://twitter.com/BorisMPower/status/1790070481279762490), **emotion detection and voice style control** [@BorisMPower](https://twitter.com/BorisMPower/status/1790067848091451823), and ability to **sing and dramatize content** [@swyx](https://twitter.com/swyx/status/1790072818559811862).

**Other AI News and Discussions**

- **Apple-OpenAI Deal**: Rumors circulated that the **Apple-OpenAI deal just closed**, one day before OpenAI's voice assistant announcement, leading to speculation that the new Siri will be powered by OpenAI technology [@bindureddy](https://twitter.com/bindureddy/status/1789905880193851725).
- **Anthropic Constitutional AI**: Anthropic released a new **prompt engineering tool for their Claude model** that can generate prompts optimized for different tasks, as shared by [@adcock_brett](https://twitter.com/adcock_brett/status/1789687847839998255).
- **Open vs Closed AI Debates**: There were various discussions on the tradeoffs of open vs closed AI development. Some, like [@ylecun](https://twitter.com/ylecun/status/1789655443377168766), argued that **open source frontier models are important for enabling a diversity of fine-tuned systems and assistant AIs**. Others, such as [@vkhosla](https://twitter.com/bindureddy/status/1789659621143306506), expressed concerns about the national security implications of open models.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**OpenAI's Upcoming Announcement**

- **Speculation about capabilities**: In /r/singularity, there is speculation that OpenAI's May 15th announcement will include [**agents, Q*-type algorithmic improvements, and architectural upgrades that will "feel like magic"**](https://www.reddit.com/r/singularity/comments/1cqrirh/heres_the_link_to_the_livestream/). Some in /r/LocalLLaMA expect a [voice assistant like the AI in the movie "Her"](https://www.reddit.com/r/LocalLLaMA/comments/1c0gop8/why_is_llamacpp_no_longer_providing_binaries/).
- **Tempering expectations**: However, others in /r/singularity are [tempering expectations, believing it will be an incremental improvement but not AGI](https://www.reddit.com/r/singularity/comments/1cqiwqw/relax_tomorrow_wont_be_a_big_deal/). The announcement has generated significant hype and speculation.

**Advances in AI Capabilities**

- **Drug discovery success rates**: New research shared in /r/singularity shows that [**AI-discovered drug molecules have 80-90% success rates in Phase I clinical trials, compared to the historical industry average of 40-65%**](https://www.reddit.com/r/singularity/comments/1cqbpsm/new_research_shows_aidiscovered_drug_molecules/). This represents a significant advancement in AI-powered drug discovery.
- **Autonomous fighter jets**: According to the Air Force Chief, [autonomous F-16 fighters are now "roughly even" with human pilots in performance](https://www.reddit.com/r/singularity/comments/1cq589n/autonomous_f16_fighters_are_roughly_even_with/). This milestone demonstrates the rapid progress of AI in complex domains like aerial combat.

**Open Source AI Developments**

- **Open source AI alliance**: As reported in /r/singularity, [**Meta, IBM, NASA and others have formed an open source AI alliance to be a voice in AI governance discussions**](https://www.reddit.com/r/singularity/comments/1cqngsf/in_dc_a_new_wave_of_ai_lobbyists_gains_the_upper/). This alliance aims to shape the narrative around AI development and regulation.
- **New open source dataset**: /r/LocalLLaMA announces the release of [code_bagel_hermes-2.5, a new open source dataset similar to the closed source deepseek-coder dataset](https://www.reddit.com/r/LocalLLaMA/comments/1cqs17q/new_data_set_code_bagel_hermes25_like_deepseek/). Open datasets enable wider participation in AI research.
- **Call to open source AlphaFold3**: In /r/MachineLearning, researchers are [asking Google DeepMind to open source AlphaFold3, their new state-of-the-art protein structure prediction model](https://www.reddit.com/r/MachineLearning/comments/1cqndld/d_please_consider_signing_this_letter_to_open/). Open sourcing cutting-edge models can accelerate scientific progress.

**Optimizing AI Performance** 

- **Faster GPU kernels**: Researchers at Stanford have released ThunderKittens, [an embedded DSL to help write fast GPU kernels that outperform FlashAttention-2 by 30% on the H100](https://www.reddit.com/r/MachineLearning/comments/1cqhsln/gpus_go_brrr/). Optimizing GPU performance is crucial for efficient AI training.
- **Improved stochastic gradient descent**: A new paper introduces Preconditioned SGD (PSGD) which [utilizes curvature information to accelerate stochastic gradient descent, outperforming state-of-the-art on vision, NLP and RL tasks](https://www.reddit.com/r/MachineLearning/comments/1cq8guo/r_curvatureinformed_sgd_via_general_purpose/). Algorithmic improvements can significantly boost AI performance.
- **Enhancing GPT-4 function calling**: In /r/OpenAI, it's shown that [techniques like adding function definitions, flattening schemas, and providing examples can increase the accuracy of GPT-4 function calling from 35% to 75%](https://www.reddit.com/r/OpenAI/comments/1cq5alr/increasing_35_to_75_the_accuracy_of_gpt4_by/). Fine-tuning prompts and inputs can greatly improve AI model performance on specific tasks.

**Humor and Memes**

- **Hype and speculation memes**: Various subreddits are sharing memes and jokes about the hype and speculation surrounding OpenAI's upcoming announcement, capturing the excitement and anticipation in the AI community. Examples: ["THIS IS ME RN"](https://www.reddit.com/r/singularity/comments/1cqbpsm/this_is_me_rn/), ["Group members be like"](https://www.reddit.com/r/singularity/comments/1cqbpsm/group_members_be_like/), ["Average 'future is now' fella"](https://www.reddit.com/r/singularity/comments/1cqbpsm/average_future_is_now_fella/).

---

# AI Discord Recap

> A summary of Summaries of Summaries

## Claude 3 Sonnet

**1. Efficient AI Model Training and Inference**:

- **[ThunderKittens](https://github.com/HazyResearch/ThunderKittens)** is gaining traction for optimizing CUDA kernels, seen as more approachable than **CUTLASS** for tensor core management. It promises to outperform **Flash Attention 2**.
- Discussions on **fusing kernels**, **max-autotune** in **torch.compile**, **Dynamo vs. Inductor**, and profiling with **Triton** aim to boost performance. The **[Triton Workshop](https://discord.com/events/1189498204333543425/1228827008830668801)** offers insights.
- **[ZeRO-1](https://github.com/karpathy/llm.c/pull/309)** integration in **llm.c** shows 54% throughput gain by optimizing VRAM usage, enabling larger batch sizes.
- Efforts to improve **CI with GPU support** in **llm.c** and **LM Studio** highlight the need for hardware acceleration.

**2. Open-Source LLM Developments**:

- **[Yi-1.5 models](https://huggingface.co/lmstudio-community/Yi-1.5-34B-Chat-GGUF)**, including 9B, 6B, and quantized 34B variants, gain popularity for diverse fine-tuning tasks.
- **[MAP-Neo](https://github.com/multimodal-art-projection/MAP-NEO)**, a transparent bilingual 4.5T LLM, and **[ChatQA](https://arxiv.org/abs/2401.10225)**, outperforming GPT-4 in conversational QA, generate excitement.
- **[Falcon 2](https://www.tii.ae/news/falcon-2-uaes-technology-innovation-institute-releases-new-ai-model-series-outperforming-metas)** 11B model, with 5T refined data and permissive license, attracts interest.
- Techniques like **[Farzi](https://arxiv.org/abs/2310.09983)** for efficient data distillation and **[Conv-Basis](https://arxiv.org/abs/2405.05219)** for attention approximation are discussed.

**3. Multimodal AI Capabilities**:

- **[GPT-4o](https://openai.com/index/hello-gpt-4o/)** by OpenAI integrates audio, vision, and text reasoning, impressing with real-time demos of voice interaction and image generation.
- **[VideoFX](https://fxtwitter.com/bedros_p/status/1789256595123179701?s=46)** showcases early video generation capabilities as a work-in-progress.
- Tokenizing voice datasets and training transformers on audio data are areas of focus, as seen in a **[Twitter post](https://fxtwitter.com/laion_ai/status/1788532651072049314?t=1NgVkLaxmC9gzgdSmGpM3Q&s=19)** and **[YouTube video](https://youtu.be/NwZufAJxmMA)**.
- **[PyWinAssistant](https://github.com/a-real-ai/pywinassistant)** enables AI control over user interfaces through natural language, leveraging Visualization-of-Thought.

**4. Debates on AI Safety, Ethics, and Regulation**:

- Discussions on **OpenAI's regulatory moves**, like GPU signing and White House collaboration, spark criticism over potential monopolization.
- Concerns arise about the impact of **AI art services** like Midjourney on artists' livelihoods and potential **legal repercussions**.
- The release of **WizardLM-2-8x22B** by Microsoft faces controversy due to similarities with GPT-4.
- Members analyze **AI copyright implications** and how companies offering indemnity could impact smaller AI ventures.
- Efforts to detect **untrained tokens** like SolidGoldMagikarp aim to improve tokenizer efficiency and model safety (**[arXiv paper](https://arxiv.org/abs/2405.05417)**).

## Claude 3 Opus

Here is a high-level summary of the top 3-4 major themes across the Discord channels, with important key terms, facts, and URLs bolded and linked to sources where relevant:

- **GPT-4o Launches with Mixed Reviews**: OpenAI released **GPT-4o**, a multimodal model supporting text, image, and audio inputs. It offers [free access with limitations](https://openai.com/index/gpt-4o-and-more-tools-to-chatgpt-free/) and advanced features for Plus users. Engineers noted its speed and cost-effectiveness but criticized its shorter memory and reasoning inconsistencies compared to GPT-4. Excitement grew for upcoming voice and video capabilities. GPT-4o also [topped benchmarks](https://x.com/lmsysorg/status/1790097588399779991?s=46) on the LMSys Arena.

- **Falcon-2 and Yi Models Gain Traction**: The open-source **Falcon-2 11B** model, trained on 5T refined data, was [released](https://falconllm.tii.ae/falcon-2.html) with a permissive license. Discussions highlighted its multilingual and multimodal capabilities despite restrictive terms. Simultaneously, the **Yi-1.5** series by 01.AI garnered praise for strong performance across tasks, with quantized variants like the rare 34B model suiting 24GB GPUs well.

- **Tooling and Techniques Advance LLM Efficiency**: New tools like **ThunderKittens** promised optimized CUDA kernels, potentially outperforming Flash Attention 2. The **[Triton Index](https://github.com/cuda-mode/triton-index)** and **[Awesome Triton Kernels](https://github.com/zinccat/Awesome-Triton-Kernels)** repositories cataloged Triton kernels for discovery. Techniques like **knowledge distillation**, **depth scaling**, and novel architectures like **Memory Mosaics** and **Conv-Basis attention** were explored to enhance LLM fine-tuning and inference efficiency.

- **Ethical and Legal Debates Persist in AI Development**: Conversations wrestled with the implications of AI-generated art on artists' livelihoods, considering fair use, derivative works, and potential legal challenges for Stability AI and Midjourney. The impact of AI copyright on innovation funding and the indemnification of big tech players over smaller entities remained contentious issues.

Let me know if you would like me to elaborate on any part of this summary or if you have additional questions!

## GPT4T (gpt-4-turbo-2024-04-09)

**Major Themes:**
1. **Regulatory Concerns and Monopolistic Moves**: There's significant discussion and concern over OpenAI's regulatory actions, particularly around practices that may favor larger companies, potentially leading to a monopolistic environment. Members expressed mixed feelings about OpenAI's moves, with criticisms particularly about potential restrictions that harm smaller competitors.

2. **New Model Releases and Enhancements**: Several discords discussed the release and capabilities of new models like GPT-4o, WizardLM, and Falcon 2. The release of these models sparked discussions about their enhanced multi-modal capabilities, performance improvements, and general excitement or skepticism about their real-world applications.

3. **Technical Tools and Innovations**: Various communities delved into technical aspects, discussing new tools and updates such as ThunderKittens for optimizing CUDA kernels, stable diffusion innovations, and advancements in model training techniques. There was a strong focus on optimizing performance and integrating the latest technological advancements.

4. **Community Engagement and Speculations**: Across several platforms, members engaged in forward-looking speculations about the impact of AI on various sectors. There were debates about the legal implications of deploying AI-driven services, discussions on the potential monopolistic behavior of AI giants, and the community's role in shaping the ethics and policies of AI development.

**Significant Discussions Linked to URLs:**

- **HuggingFace's Regression Analysis**: [Understanding Depth Scaling in LLMs](https://arxiv.org/abs/2312.15166)
- **OpenAI's GPT-4o Release Features**: Highlighted in multiple discussions across discords for its significant performance improvements and multimodal capabilities. Links to official release notes: [GPT-4o Launch Details](https://openai.com/index/hello-gpt-4o/)
- **ThunderKittens Optimization Tool**: Gaining traction for enhancing CUDA operations, linked here: [ThunderKittens GitHub](https://github.com/HazyResearch/ThunderKittens)
- **Falcon 2's Launch**: Discussed for its multilingual and multimodal capabilities across different discords, further details can be found here: [Falcon 2](https://falconllm.tii.ae/falcon-2.html)
- **Legal Concerns Over AI Art**: Heated discussions about the implications for artists and legal battles surrounding AI-generated art were prevalent, particularly highlighted in platforms discussing Stability.ai and Midjourney's operations.


## GPT4O (gpt-4o-2024-05-13)

1. **Regulatory Challenges and Platform Control:**
   - **OpenAI's Regulatory Moves:** Discussions spanned multiple communities about OpenAI's implementation of tighter control through measures like compulsory GPU signing and collaboration with the White House, raising concerns over monopolistic tendencies (e.g., **[Unsloth AI (Daniel Han)]**).
   - **Competitive Landscape:** Concerns were also raised about how these moves could marginalize smaller competitors, favoring big tech companies, indicating a broader fear of restricted innovation in the AI space **[Nous Research AI](https://discord.com/channels/1053877538025386074)**.

2. **Advancements in and Deployment of New Models:**
   - **GPT-4o Release:** Enthusiasm was noted for GPT-4o's launch, highlighting its free public access with certain limitations and multi-modal capabilities integrating audio, vision, and text reasoning **[OpenAI](https://discord.com/channels/974519864045756446)**.
   - **Community Response:** Some noted mixed emotions about GPT-4o's performance compared to previous models, with some excitement over new features overshadowed by noted reasoning inconsistencies **[Perplexity AI](https://discord.com/channels/1047197230748151888)** and **[HuggingFace](https://discord.com/channels/879548962464493619)**.

3. **Focus on Technical Optimization and Fine-Tuning:**
   - **ThunderKittens:** Gained attention for its promising kernel performance improvements, suggested to outperform existing methods like Flash Attention 2 **[CUDA MODE](https://discord.com/channels/1189498204333543425)** and **[Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276)**.
   - **Fine-Tuning Issues:** Multiple communities mentioned difficulties in fine-tuning models like Llama3, with discussions about specific solutions and optimization techniques **[HuggingFace](https://discord.com/channels/879548962464493619)**.

4. **Application and Use-Case Innovations:**
   - **World Simulation and AI Agents:** Platforms for running simulations like **Websim** and **AI agents** for tasks like generating PowerPoint presentations were shared. There was also notable interest in enhancing simulation capabilities, including integrating Digital Audio Workstations **[Nous Research AI](https://discord.com/channels/1053877538025386074)**.
   - **Community Tool Sharing:** Users frequently shared code examples, scripts, and tutorials to assist with setting up and configuring AI tools, emphasizing collaborative knowledge sharing across projects like **LangChain AI** and **HuggingFace**.

Important Links:

1. **WizardLM GitHub**: [https://huggingface.co/alpindale/WizardLM-2-8x22B](https://huggingface.co/alpindale/WizardLM-2-8x22B)
2. **ThunderKittens GitHub**: [https://github.com/HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
3. **OpenRouter API Watcher Demo**: [https://orw.karleo.net/](https://orw.karleo.net/)
4. **RAG Pipeline Tutorial**: [https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog](https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog)
5. **Deep Learning Initialization Guide**: [https://www.deeplearning.ai/ai-notes/initialization/index.html](https://www.deeplearning.ai/ai-notes/initialization/index.html)
6. **AI Research Papers** (various links): 
   - [SOLAR: Depth Upscaling for Language Models](https://arxiv.org/abs/2312.15166)
   - [Conv-Basis: Efficient Attention Approximation](https://arxiv.org/abs/2405.05219)
   - [FlashAttention-2](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)

---



# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **OpenAI's Regulatory Debate Heats Up**: Community discusses OpenAI's recent moves toward tighter control, with particular focus on compulsory GPU signing and collaboration with the White House. Concerns were aired about creating a monopolistic environment favoring bigger companies over smaller competitors.

- **WizardLM Steals the Spotlight**: Despite controversy, the WizardLM-2-8x22B model has garnered support, originally released by Microsoft and bearing resemblance to GPT-4. The model stirred conversations about its availability and potential censorship, with resources shared on the [WizardLM GitHub page](https://huggingface.co/alpindale/WizardLM-2-8x22B).

- **Tuning and Tooling for Peak Performance**: On the technical side, discussions emerged about efficient methods and tools for fine-tuning models. Attention was on ThunderKittens kernel for its promising performance gains, potentially outdoing Flash Attention 2, found at [ThunderKittens GitHub](https://github.com/HazyResearch/ThunderKittens).

- **Unsloth AI Gains Multi-GPU Support**: Unsloth AI has been acknowledged for its efficient model fine-tuning capabilities and is slated to support multi-GPU functionality. Importance was given to the tool’s ability to integrate new model variants without needing separate branches, as detailed on [Unsloth GitHub](https://github.com/unslothai/unsloth).

- **Fine-Tuning Frustrations with Llama3**: Engineers swapped tactics for addressing fine-tuning challenges with Llama3 models, discussing dataset sizes, padding quirks, and conversions across FP16 to GGUF format. Technical issues such as tokenization inaccuracies with GGUF tokenizers were also a key topic.

- **A Peek at Altman's Q&A**: OpenAI hosted a Q&A with CEO Sam Altman, focusing on the Model Spec and fostering community engagement. The session's motive is outlined in the [Model Spec](https://cdn.openai.com/spec/model-spec-2024-05-08.html) document.

- **Llama Variants Get Finetuned for Token Classification**: An engineer has contributed Llama variants optimized for token classification tasks, using LoRA adapters and trained on the conll2003 dataset. These models are accessible via their [Hugging Face collection](https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188).



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3: More Myth than Model?**: Discussions in the guild were rife with speculation about **Stability AI's** rumored **SD3**, akin to the **Half-Life 3** anticipation. The lack of official release dates has led to a mix of hope and disappointment among users.

- **Call for Fine-Tuning Assistance Answered**: An expert stepped forward to aid with **fine-tuning Stable Diffusion XL** for ad generation, highlighting their experience with the machine learning backend of **creativio.ai**.

- **Complexities of Model Usage and Configuration**: Users shared challenges in downloading and setting up sizable models like [**CohereForAI's C4AI Command R+**](https://huggingface.co/CohereForAI/c4ai-command-r-plus), and software such as **KoboldAI** and **OogaBooga**. These struggles underscored complexities related to software configuration and model file management.

- **Art Styles and Animation Insights**: Advice was offered on using **gpt-4** for identifying art styles and the **animatediff with controlnet tile** method for animating artwork in a way that remains true to the original piece's aesthetic.

- **Image Upscaling Quest**: A user sought expertise for enhancing image resolutions using **Automatic1111's forge with controlnet**, highlighting a broader interest in achieving detailed and high-quality image upscaling within the community.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o Unlocked for Public**: OpenAI has released **GPT-4o**, offering free access with limitations on usage and advanced features reserved for Plus users. This model distinguishes itself with multi-modal capabilities, integrating audio, vision, and text reasoning. [Launch Details](https://openai.com/index/hello-gpt-4o/) and [Usage Information](https://openai.com/index/gpt-4o-and-more-tools-to-chatgpt-free/).

- **Mixed Emotions on GPT-4o's Performance**: The engineer community's reaction to **GPT-4o** is divided, highlighting its enhanced speed and cost-effectiveness, albeit accompanied by a shorter memory span and occasional reasoning inconsistencies when compared to its predecessor. Excitement for voice and video feature integrations is palpable, tempered by the current lack of availability and some confusion over rollout schedules.

- **Fine-Tuning the AI Toolset**: Discussions on **APIs** reflect the technical crowd's interest in **GPT-4T's** extended 128k context for more nuanced applications, alongside strategies to manage the randomness at high-temperature settings. Practical concerns include vigilant monitoring of OpenAI's static pricing via their [Pricing Page](https://openai.com/api/pricing/) and awaiting the implementation of per-GPT memories discussed in the [Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq).

- **Programming Puzzles with Gemini 1.5**: AI engineers are troubleshooting problematic moderation filters affecting responses in applications using Gemini 1.5 and shared steps for creating, managing, and linking to downloadable file directories using Python scripts—indicative of their resourceful approach to solving immersion-breaking application constraints.

- **ChatGPT with a Supervisory Twist**: A user queried about crafting a **ChatGPT clone with a 3.5 model** that incorporates user message monitoring by an overseeing establishment, suggesting a nuanced approach to interface replication that extends into the administrative oversight realm.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama Struggles Beyond 8k**: The **llama 3 70b** model is exhibiting coherency issues when generating content over 8,000 tokens.
  
- **Introducing MAP-Neo**: The **MAP-Neo** project has been unveiled; it's a transparent, bilingual LLM trained on 4.5 trillion tokens, with resources and documentation available on [Hugging Face](http://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04), its [dataset](https://huggingface.co/datasets/m-a-p/Matrix), and GitHub [repository](https://github.com/multimodal-art-projection/MAP-NEO).

- **Revolutionizing Conversational QA with ChatQA**: A breakthrough detailed in an [arXiv paper](https://arxiv.org/abs/2401.10225), **ChatQA-70B** outclasses GPT-4 in conversational QA, leveraging the InstructLab framework by IBM/Redhat that introduces incremental enhancements through curated weekly dataset updates, documented [here](https://github.com/instructlab).

- **World Simulation Tech Talk**: Members shared enthusiasm for **WorldSim**, a platform for running simulations and discussing philosophy, with technical discussions and bug reports on the simulator command issues. They approached world simulation with a desire for expanded features such as digital audio workstation integration.

- **GPT-4o Stirring Debate**: **GPT-4o's** impact on the AI field leveraged controversial opinions within the community, discussing its pros, such as improved coding performance and quantitative efficiency, alongside concerns about its proprietary nature and possible challenges to open-source AI.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **PhD Thesis Worthy of Applause**: An NLP PhD thesis attracted attention and praise, with a [social media shout-out](https://twitter.com/ShunyuYao12/status/1789058769982550031) for the author's achievements.

- **No Data Left Behind**: Discussion turned to [Llama 3's](https://x.com/mark_cummins/status/1788949893903511705?s=46&t=90xQ8sGy63D2OtiaoGJuww) massive 15 trillion token training, sparking debate on data sources and prompting contrast with Stella Biederman’s stance on data necessity.

- **AI Infrastructure - Feedback Wanted**: A Substack post outlines new infrastructure services designed for AI agents, with a call for the community's input [read more here](https://sweekiat.substack.com/p/d8726e73-e717-4599-81a3-5eb82e48f9c9).

- **Falcon 2 Takes Flight, But With Tethered Wings**: [Falcon 2's launch](https://falconllm.tii.ae/falcon-2.html) stirred conversations around its leading-edge, multilingual, and multimodal facilities. Licensing conditions, however, raised eyebrows over their restrictiveness.

- **GPT-4o Drops Jaws**: Revelations around GPT-4o's capabilities, including its low latency and versatile responses, steered debate on API access and real-world performance, as enthusiasts shared [OpenAI's latest unveilings](https://openai.com/index/hello-gpt-4o/).

- **OpenAI Watch Party - Join In!**: A guild member announced a watch party for an OpenAI event with pre-event festivities kicking off 30 minutes prior [discord invite](https://discord.gg/Z7V4NDGZ?event=1238918257046458368).

- **Watch Party Woes**: At the Open AI Spring Event, an initial hiccup with the stream's audio occurred, but quick community tips helped improve the situation.

- **Apple vs. Google - The Speculative Saga**: Amidst rumors of Apple lagging in AI, guild members shared insights into whether Siri might integrate GPT-4o, hinting at the specter of regulatory concerns [related discussions](https://twitter.com/youraimarketer/status/1789918014617399355).

- **Live Impressions of GPT-4o**: Live demonstrations of GPT-4o's emotional voice capabilities and its multimodal proficiency wowed engineers, stirring talks of real-time productivity and creative applications [event playback](https://www.youtube.com/watch?v=DQacCB9tDaw&ab_channel=OpenAI).

- **AI's Next Move - Competing in the Big Leagues**: The community speculated about the competitive consequences of GPT-4o and potential disruptions to applications by Google, Siri, and others, with some considering these steps a stride towards mimicking human interaction.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Cheerio's Challenger**: A faster alternative to the **Cheerio library** was sought for HTML content extraction. A user directed others to [Perplexity's AI search](https://www.perplexity.ai/search/Is-there-a-xOtvxOveTGSfbae88ElQMA) for more information.

- **Choosing Between AI Services**: Conversations compared **ChatGPT Plus** with **Perplexity Pro**, with the latter being praised for its niche as an AI search engine enabling features like collections and model flexibility. Claude 3's usage limits in Perplexity Pro were a sore point, with users looking at **YesChat** for more generous quotas.

- **GPT-4o Steals the Spotlight**: The community engaged eagerly about the launch of **GPT-4o**, discussing its better speed and capabilities over preceding models. Interest was high regarding when **Perplexity** would incorporate GPT-4o into its services.

- **Perplexity at the Helm of AI Search**: **Alexandr Yarats** was spotlighted through his [recent interview](https://www.unite.ai/alexandr-yarats-head-of-search-at-perplexity-interview-series/), shedding light on his trajectory from Yandex and Google to becoming Perplexity AI's Head of Search.

- **Tutorial Inquiry Indicates Diverse User Base**: A user's request for a Perplexity AI tutorial in Spanish signals the platform's global reach and the need for multilingual support resources. A link was shared for a "deep dive," albeit without explicit detail: [Deep dive into Perplexity](https://discord.com/channels/1047197230748151888/1047649527299055688/1197892547276705843).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Unlocking LLM Potential on Modest Hardware**: Open-source LLM models like **Mistral** and **LLaMa3** were discussed due to their lower hardware demands compared to **ChatGPT**. Resources such as [LM Studio](https://lmstudio.ai/) allow users to discover and run local LLMs.

- **Pushing the Frontiers of AI Troubleshooting**: Various technical issues were aired, including problems encountered while disabling a safety checker in **StableDiffusionPipeline**, GPT's data retrieval challenges in RAG applications, and fine-tuning of models like **GPT-2 XL** on **Nvidia A10G** hardware. There was also buzz around **OpenAI's GPT-4o** and its capabilities.

- **Dynamic Approaches in AI Learning**: From **genAI user experience** involving containerized applications ([YouTube video](https://www.youtube.com/watch?v=UgVPzSSCjr8)) to a tutorial on **Neural Network Initialization** from DeepLearning.ai ([deeplearning.ai article](https://www.deeplearning.ai/ai-notes/initialization/index.html)), and a **JAX and TPU integration for VAR paper** ([GitHub for Equinox](https://github.com/patrick-kidger/equinox))—the community showcased a breadth of learning resources.

- **Phi-3 On-The-Go and Robotic Breakthroughs**: Highlighted resources included a paper about **Phi-3's** efficiency on smartphones ([arXiv link](https://arxiv.org/abs/2404.14219)), the book "Understanding Deep Learning" for grasping deep learning concepts, and a novel **3D Diffusion Policy (DP3)** for robots ([3D Diffusion Policy website](https://3d-diffusion-policy.github.io/)).

- **Innovative Creations and AI Deployments**: Community members showcased an array of projects: an AI-powered storyteller ([Alkisah AI](https://huggingface.co/spaces/ikmalsaid/alkisah-ai)), Holy Quran verses tool ([Kalam AI](https://huggingface.co/spaces/ikmalsaid/kalam-ai)), an OCR framework ([OCR Toolkit on GitHub](https://github.com/ajkdrag/ocrtoolkit)), fine-tuned Llama variants ([HuggingFace collection](https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188)), and a tutorial for an **AI Discord chatbot** ([YouTube video](https://youtu.be/B1F94RKksR8?si=WPSmpyjiByCHaTAQ)).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**GPT Agents in Learning Limbo**: GPT agents' inability to assimilate new information into their base knowledge caused buzz, with clarification on how information is stored as "knowledge" files that don't update the agent's core understanding.

**Hardware Hurdles for Hi-Tech Pursuits**: Engineers faced challenges running advanced models like **Llama 3 70B Q8** on hardware with **128GB RAM**, with PCIe 3.0 causing bottlenecks remedied by switching to PCIe 4.0 motherboards. Utilizing GPUs with less than 6GB VRAM for weighty models proved futile.

**Yi Models Yield Enthusiasm**: Yi-1.5 models, including 9B and quantized 34B variants, received praise and recommendations for a variety of tasks, with quantized models leveraging `llama.cpp` for improved performance.

**Tooling Up for Efficiency**: LM Studio's **0.2.22 update** introduced a CLI tool, `lms`, for model management and boasted bug fixes in llama.cpp, while the community navigated the complexities of connecting **OpenInterpreter** to **LM Studio** and configuring headless installations on Linux servers.

**Quest for Research Collaboration**: Dispensing with corporate vernacular, the conversation sought aid and shared experiences for running **MemGPT** on various setups, revealing a collective endeavor to optimize this AI model.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**JetMoE 8B Free Hits a Snag**: The **[JetMoE 8B Free model](https://openrouter.ai/models/jetmoe/jetmoe-8b-chat:free)** is experiencing downtime due to upstream overload, returning an error (502) to all requests until further notice.

**Eye on the Models—OpenRouter API Watcher**: An open-source tool called **OpenRouter API Watcher** has been unveiled, which keeps track of changes in OpenRouter's model availability, offering hourly updates via a web interface and an RSS feed with low overhead. Check out the [demo](https://orw.karleo.net/).

**A Beta Tester’s Dream with Rubik's AI Pro**: Users can beta test and provide feedback for Rubik's AI Pro, an advanced research assistant and search engine, with **2 months of free premium** access using a `RUBIX` promo code. Further details can be found at [Rubik's AI](https://rubiks.ai/).

**Jetmoe’s Caveat**: It has been confirmed that **Jetmoe** lacks internet access, which restricts its use cases, but it remains useful for academic research.

**GPT-4o Joins OpenRouter**: **GPT-4o** has been added to OpenRouter’s arsenal, supporting text and image inputs, and generating buzz for its performance and competitive pricing, although it lacks support for video and audio inputs.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Contemplation on Pattern Matching**: There was a vigorous debate about implementing pattern matching in Mojo, with affirmative stances on compiler efficiency and exhaustive case handling. Conversely, objections were raised on grounds of aesthetic preference for traditional `if-else` constructs.

- **Mojo Rises, Rust's Complexity Under Lens**: Mojo's compiler, described as more navigable and straightforward than Rust's, was a hot topic. Discussions extended to Mojo's future development and, separately, the potential relationship between Mojo and [MLIR](https://llvm.org/docs/MLIRGuide.html).

- **Innovations and Contributions in Mojo**: Ideas were exchanged on incorporating yield-like behavior and new hashing techniques into Mojo. Links to proposed changes such as in [this pull request](https://github.com/modularml/mojo/pull/2619) and a [YouTube talk](https://www.youtube.com/watch?v=9ag0fPMmYPQ) also sparked discussions on the language's ownership model.

- **Nightlies and Enhanced Mojo Performance**: Discussions on [GitHub Issues](https://github.com/modularml/mojo/issues) about CI tests in Ubuntu, custom `Hasher` struct proposals, and performance optimizations for Mojo's `List` structure highlighted the active nightly builds and their role in the ongoing development rhythm.

- **String Building in Mojo's Landscape**: A new repository for MoString received attention, offering a variation on StringBuilder approaches and a method to reduce memory allocation in Mojo, available [here on GitHub](https://github.com/dorjeduck/mostring).



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **ThunderKittens Strikes a Chord**: Engineers are showing great interest in [ThunderKittens](https://github.com/HazyResearch/ThunderKittens), a project focusing on optimizing CUDA kernels. It’s seen as more approachable than CUTLASS for tensor core management, and its repository includes projects like [NanoGPT-TK](https://github.com/HazyResearch/nanoGPT-TK), heralded for its performance in GPT training.

- **Triton's Expanding Universe**: Knowledge sharing on Triton peaked with the recommendation of advanced learning resources, including a detailed [YouTube lecture](https://www.youtube.com/watch?v=DdTsX6DQk24) and pointers to GitHub repos such as [PyTorch’s kernels](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/conv.py). The excitement is palpable with discussions of internal performance and new domain-specific languages that could outperform current implementations.

- **Learning on Demand**: Upcoming expert talks on **fusing kernels** and **CUDA C++ scans** were announced, with [Zoom](https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success) as the venue. A University of Illinois lecture series on parallel programming is also accessible, offering [Zoom sessions](https://us06web.zoom.us/j/83020353425?pwd=w3oQfYJPJVz2arzeZmxJbBsAMGFrBD.1) and a comprehensive [YouTube playlist](https://youtube.com/playlist?list=PLRRuQYjFhpmvu5ODQoY2l7D0ADgWEcYAX) for independent study.

- **Performance Tuning Tackled**: Discussions tackled techniques to boost performance from calculating outside CUDA kernels to using max-autotune for kernels to compiler dynamics with [Dynamo over Inductor](https://github.com/pytorch/workshops/tree/master/ASPLOS_2024), highlighting the nuanced trade-offs between kernel fusion benefits and configuration costs.

- **Community Support and Query Resolution**: Queries ranged from understanding GPU memory management with CUDA to seeking project assistance for thermal face recognition, involving requests for insights, papers, and Git repositories. Additionally, there’s been productive interaction over course content and GPU compatibility checks for builds.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Mind the Synthetic Hype!**: Despite a **bullish stance on synthetic data**, some engineers exercise caution due to a previous hype cycle about 5-7 years ago, questioning if critical lessons will translate with the entry of new professionals in the field.
- **Convolutional Contemplations**: AI Engineers are comparing the performance of **CNNs, Transformers, and MLPs** for vision tasks, as noted in [arXiv paper discussions](https://arxiv.org/abs/2108.13002), suggesting that while moderate scales show competitive performance, scaling up may require a mixed-method approach.
- **Efforts in Model Compression**: Conversations arose about model compression's impact on **features and neural circuits**, pondering if the lost features during compression are redundant or critically specialized revealing the dataset's diversity.
- **Curiosity over New Attention Method**: A new efficient **attention approximation method** using convolution matrices has been discussed with some skepticism, considering existing methods such as **flash attention**, alongside talks of **depth scaling in Large Language Models (LLMs)**, referencing [SOLAR](https://arxiv.org/abs/2312.15166) and [Yi 1.5 models](https://arxiv.org/abs/2403.04652).
- **Insights into Falcon-2 and Copyright Conversations**: The release of **Falcon-2 11B**, trained on a significant 5T of refined data and featuring a permissive license, sparked discussion, while ongoing debates about **AI copyright implications** highlight the competitive edge that may skew towards indemnifying corporations like Microsoft, highlighting a potential chilling effect on smaller players.




---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GPT-4o Ascends to the Top**: [GPT-4o](https://x.com/liamfedus/status/1790064963966370209?s=46), OpenAI’s latest model, has been demonstrated to outperform predecessors in coding and may raise the bar in other benchmarks like MATH. It has also become the strongest model on the [LMSys Arena](https://x.com/lmsysorg/status/1790097588399779991?s=46), boasting higher win-rates against all other models.

- **REINFORCE Understood Through PPO Lens**: A [Hugging Face PR](https://github.com/huggingface/trl/pull/1540) revealed that REINFORCE is a special case of PPO, presenting an interesting perspective on the relationship between the two reinforcement learning methods, documented in a [recent paper](https://arxiv.org/pdf/2205.09123).

- **VideoFX Work in Progress Draws Eyes**: Early footage of VideoFX showcased its burgeoning capabilities, generating interest with preview content on [Twitter](https://fxtwitter.com/bedros_p/status/1789256595123179701?s=46).

- **Tokenizer Tuning Increases Efficiency**: OpenAI has pushed a new update for their tokenizer, increasing processing speed by making use of a larger vocabulary as seen in the recent [GitHub commit](https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8).

- **Videos Capture Attention with Viral Potential**: Within Interconnects' #reads, a surge of views on certain videos sparked conversations around promotion strategies, with one aiming to reach higher view counts inspired by another Huggingface video's popularity. There was even discourse on circumventing Stanford's licensing for wider dissemination of video content.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Artistic Anxieties over AI**: Engineers discussed the implications of AI art on artists' livelihoods, examining the impact of services like Midjourney on art sales as well as potential *legal repercussions*. Some argued for fair use while others expressed concerns about derivative works, with reference to insights from [The Legal Artist](https://www.thelegalartist.com/blog/you-cant-copyright-style).
  
- **Legal Buzz Surrounding AI**: There was chatter around StabilityAI and Midjourney facing possible legal challenges given the current climate, with some hoping for David Holz to face repercussions for his work. The discussion included the unpredictable influence of jury decisions on the direction of such legal cases.

- **Evolutions in AI Efficiency**: Mention of improved efficiency in AI models sparked interest, with the spotlight on a fine-tuned Pixart Sigma model on Civitai and advancements in AI compute showcased by [FlashAttention-2](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2).

- **Falcon 2 Takes Flight**: Announcements highlighted the launch of Falcon 2 models boasting superior performance compared to Meta's Llama 3, with detailed information available through the [Technology Innovation Institute](https://www.tii.ae/news/falcon-2-uaes-technology-innovation-institute-releases-new-ai-model-series-outperforming-metas).

- **Audio's Textual Transformation**: Engineers explored the conversion of voice datasets into tokens, emphasizing high-quality annotations for emotions and speaker attributes. They shared a [Twitter post](https://fxtwitter.com/laion_ai/status/1788532651072049314?t=1NgVkLaxmC9gzgdSmGpM3Q&s=19) and a [YouTube video](https://youtu.be/NwZufAJxmMA) on training transformers with audio data for further understanding.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **ISO Date Extraction Using LangChain**: A member's request on how to **extract and convert dates to ISO format** led to shared code examples using the `DatetimeOutputParser` in both Python and JavaScript, highlighting LangChain's functionality in structured output.

- **Hook Up Local LLMs with LangChain**: The conversation included guidance on integrating local open-source LLMs such as Ollama using LangChain, with **Kapa.ai providing a breakdown** of model definitions and prompt creation.

- **Persistent Storage Solutions Beyond InMemoryStore**: In the quest for persistent storage alternatives within LangChain and Gemini, some **pointed to LangChain documentation** for potential solutions, moving past the limited `InMemoryStore`.

- **Common Hurdles with HuggingFace Integration**: Users shared experiences and fixes for frequent issues encountered when integrating HuggingFace models with LangChain, emphasizing the importance of **model compatibility and precise API interactions**.

- **Tutorials and Resources to Enhance LangChain Know-How**: The community spotlighted resources like a [YouTube tutorial](https://www.youtube.com/watch?v=KQ-xGVFHDkw) and a detailed [blog post](https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog) on creating a RAG pipeline with LangChain, with open requests for guidance on **streaming and session management** within LangChain applications.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Slide Decks on Automatic**: Using the Llama3 RAG pipeline, a new system to generate PowerPoint presentations has been developed, incorporating Python-pptx. The workflow and integration details are shared in an [article](https://t.co/iM0c5Cl2uK).
  
- **Reflecting on Reflection**: Hanane Dupouy's exploration of creating a financial agent that reflects on stock prices shows promise for advanced CRITIC applications, with an in-depth explanation available in their [exposure](https://t.co/mmJ8cjmw73).

- **Moderation by RAG**: Setting up a RAG pipeline for moderating user-generated images by converting images to text and checking against indexed rules is outlined, with a more [detailed procedure](https://t.co/z6jBpMvQss) available.

- **RAG System Under the Microscope**: A comprehensive article presented by @kingzzm covers the evaluation of RAG systems, utilizing libraries such as TruLens, Ragas, UpTrain, and DeepEval, with a link to the full [article for the metrics](https://t.co/gLbXJoPsqu).

- **Distill Knowledge, Sharpen Models**: A valuable discussion-centric [blog post](https://huggingface.co/blog/Andyrasika/knowledgedistillation-gpt) on the knowledge distillation technique used to fine-tune GPT-3.5 is recommended for engineers looking to increase model accuracy and performance.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Tech-Savvy Inner Circle Shares AI Insights**

- **LLAMA3's Instructional Layer Secrets**: An [analysis](https://gist.github.com/CoffeeVampir3/48544cdaf888a76ca6f8e25863200fad) shows key weights in **LLAMA 3** concentrated in the *K and V layers*, suggesting possible freezing to induce stylistic variations without affecting its instructional prowess.
  
- **Practicality of OpenOrca and AI Efficiency**: AI enthusiasts evaluated the feasibility of re-running OpenOrca's deduplication for GPT-4o, roughly costing $650, while spotlighting methods like *Based*, *Monarch Mixer*, *H3*, and *FlashAttention-2* to enhance computational efficiency, as discussed in a [blog post](https://hazyresearch.stanford.edu/blog/2024-05-12-tk).

- **Development Chaos: Dependencies & Docker Woes**: Developers reported difficulties ranging from **AttributeError 'LLAMA3'* errors when using Docker to outdated dependencies leading to conflicts, emphasizing the transition from **torch 2.0.0 to 2.3.0** with the need for updates in **fastchat** and **pyet**.

- **AXOLOTL Interactions Met with Errors and Questions**: The AI community faces diverse challenges, including error messages converting models to **GGUF**, loading **Gemma-7B**, and pragmatically merging **QLoRA** into base models, often left unresolved within thread discussions.

- **No Quick Fix in Sight**: Inquiries addressed to the **Axolotl-phorm-bot** about topics like **pruning support**, **continuous pretraining**, **LoRa methods**, and **QLoRA merging techniques** prompted searches in Axolotl's repository without providing immediate solutions - details check on [Phorm's platform](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined).

Deploying **practical solutions and seamless updates** remains a collective goal in tackling **emergent AI tech puzzles** — updates and breakthroughs to follow.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Goofy Errors and Speedy Performances**: Claude API users reported **"goofy errors"** impeding its use, whereas **GPT-4o** garnered praise for its swift performance, clocking at "minimum 100 tokens/s." Local models such as Mixtral and Llama3 were considered inferior to **GPT-4**.

**PyWinAssistant Showcases AI Control over UI**: An open-source project dubbed **PyWinAssistant** allows control of user interfaces through natural language, leveraging Visualization-of-Thought for spatial reasoning. Excitement grew as users shared a [GitHub repo](https://github.com/a-real-ai/pywinassistant) and a live [YouTube demo](https://www.youtube.com/live/_XyYoqpJCoQ?si=rA3ijqicagANyt96&t=1993).

**Hardware Headaches and Software Solutions**: Integration of **LiteLLM**, Groq and Llama3 successfully confirmed, while another user struggled to connect their 01-Light device. Separate issues arose with **Python script** execution resolved by importing `OpenInterpreter` correctly.

**Shipment Updates and Support Channels**: Queries about the **01 hardware** brought news of upcoming batch shipments, and an **iOS app** for the hardware is in beta, shared on [GitHub](https://github.com/eladdekel/01_For_iOS). Order cancellations were directed to *help@openinterpreter.com*.

**Dev Discussions on Model Swapping**: The **01 dev preview** prompted exchanges on switching to local models using `poetry run 01 --local`, offering insights into **model selection commands**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tensor Talk Tackles Variable Shapes**: Engineers debated how to represent tensors with **variable shapes** in **tinygrad**, a topic especially relevant in transformers due to changing token numbers. They referred to [Tinygrad's handling of variable shapes](https://mesozoic-egg.github.io/tinygrad-notes/upcast2.html) and code snippets from **Whisper** ([snippet 1](https://github.com/tinygrad/tinygrad/blob/a1940ced7746fcdf09068aadf4155e4c1e3641b8/examples/whisper.py#L36-L45), [snippet 2](https://github.com/tinygrad/tinygrad/blob/a1940ced7746fcdf09068aadf4155e4c1e3641b8/examples/whisper.py#L118-L120)) for insights.

- **Dim Versus Axis: Different Terms, Same Concept?**: There was a clarification sought on the **terminology difference** between "dim" and "axis" in tensor operations, concluding that the terms are mostly **interchangeable** and any differences might be rooted in historical conventions.

- **Debugging `AssertionError` During Training**: A user faced an `AssertionError` related to missing gradients during a **bigram model training** which led to a discussion on proper settings (`Tensor.training = True`). The conversation included a reference to a [GitHub pull request](https://github.com/tinygrad/tinygrad/pull/4460/files) to prevent such issues.

- **Feature Aggregation in Neural Turing Machines**: An **NTM implementation** prompted discussions on **feature aggregation** via tensor operations and optimization, for which code examples were exchanged and ideas on efficiency improvements were discussed ([aggregate feature code](https://gist.github.com/RaulPPelaez/36b6a3a4bbdb0c373beaf3c1376e8f49)).

- **Navigating `where` in Backprop Challenges**: Participants worked through a **backpropagation issue** with a 'where' call in tinygrad that was causing `RuntimeError`. The workaround involved a `detach().where()` method, highlighting a PyTorch-to-tinygrad gradient challenge.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Token Troubles and Model Mechanics**: A query on the unexpected surge in input tokens was clarified; web searches using command 'r' result in context passing and higher token count, leading to billing charges. Meanwhile, the challenge of 'glitch tokens' in language models like SolidGoldMagikarp was acknowledged with a [linked arXiv paper](https://arxiv.org/abs/2405.05417), which discusses detection methods for these potentially problematic tokens.
  
- **Open-Source Embeddings and Billing Brain Teasers**: No consensus was reached about the open-source nature of embedding models due to a lack of responses. In a separate issue, billing confusion over a $0.63 charge was resolved, attributed to the amount due since the last invoice.

- **Aya vs. Cohere Command Plus - Clash of the Models**: In a comparison between Aya and Cohere Command Plus models, Aya was reported less accurate, even with a 0 temperature setting, with one user suggesting its best use case in translation tasks.

- **Specializing LLMs Seek New Horizons in Telecom**: A challenge to tailor large language models (LLMs) for the telecom sector, focusing on areas such as 5G, was shared, with more details found on the [Zindi Africa competition page](https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks).

- **In Search of a Chat-with-PDF Solution**: A call was made for references to a "chat with PDF" application utilizing Cohere, with the incentive being collaboration and knowledge-sharing among members.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **GPT-4o Still Falling Short**: Members shared frustration about GPT-4o's inaccuracies, experiencing a 50% success rate when asking the model to list book titles it "saw" in a library scenario.
- **Voice Assistant Marketing Missteps**: Recent voice assistant promotion mishaps, including unwanted giggling from the devices, drew criticism from users who called it "embarrassing".
- **Custom Instructions Could Improve Voice Assistants**: Hopes are pinned on custom instructions to improve the interactions with voice assistants, aiming to eliminate awkward behavior.
- **AGI Believers Club Lacks Members**: Skepticism prevails about the near-term development of AGI, with engineers expressing a lack of belief in its imminent advent.
- **Law of Diminishing Returns in LLMs**: Discussions indicate a consensus that there are diminishing improvements in new versions of large language models, and current models have untapped capabilities.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Beware Fake Repositories**: An announcement warned about a **fake OpenELM repository**; there is no **GGUF (GitHub User File)** for OpenELM currently available, cautioning the community against potential scams.

- **llamafile Archives Receive a Boost**: A new **pull request (PR)** was mentioned for an upgrade script for llamafile Archives, based on a script from [*Brian Khuu's blog*](https://briankhuu.com/blog/2024/04/06/inplace-upgrading-of-llamafiles-engine-bash-script/), offering improvement and maintenance for file handling processes.

- **Containers Get a Green Light**: Confusion around using containerization tools like **podman or kubernetes** was resolved, affirming that utilizing containers for operations is approved and encouraged for deployment consistency and scalability.

- **Performance Check for Hermes-2-Pro**: Experiences with the **Hermes-2-Pro-Llama-3-8B-Q5_K_M.gguf** running on an **AMD 5600U** were shared, noting response times of approximately **10 seconds** and RAM usage spikes of **11GB**.

- **Model Troubleshooting: Batch Size Errors**: Reports surfaced of an error affecting both **Llama 8B and Mistral models** involving update_slots and n_batch size issues. High **RAM allocation** appears to mitigate the issue, which is less prevalent in other models like **LLaVa 1.5** and **Llama 70B**.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Searching for German Content**: A pursuit for diverse **German YouTube channels** to train a Text-to-Speech model led to suggestions such as using [Mediathekview](https://mediathekview.de/) to download content. The Mediathekview's JSON API was also highlighted as a resourceful tool, as seen in the [GitHub repository](https://github.com/59de44955ebd/MediathekViewWebVLC/blob/main/mediathekviewweb.lua).

**Keep It English**: A reminder was issued within the discussions to ensure that English remains the primary language for communication, possibly to maintain the accessibility of discussions.

**Demo Status Check**: An inquiry about the status of a unidentified demo received no response, indicating either a lack of information or attention to the query.

**Thumbs Up for... Something**: Positive feedback was expressed with a brief *"It's really nice,"* comment, though the context of this satisfaction wasn't expanded upon.

**Curiosity for RT Audio Interface**: There's evident curiosity and excitement about the "RT Audio interface" in applications beyond chat, but experiences or results have not yet been shared in the discussions.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Claude Beats Llama at Haiku**: In a showdown of linguistic prowess, engineers compared the submodel accuracy of **Claude 3 Haiku** with **Llama 3b Instruct's** entity extraction capabilities. Initial experiments with fuzzy matching proved fruitless, sparking interest in more sophisticated submodel matching techniques.

- **Teasers and Voices Stir Excitement**: Anticipation is building in the community as **OpenAI's Spring Update** has been teased, promising the introduction of **GPT-4o**. A notable highlight is none other than **Scarlett Johansson** voice-featured in the update, sparking both surprise and amusement among members.

- **Audio Futures Discussed**: Technical discussions speculated on **OpenAI's** potential integration of audio functionalities, envisioning direct audio input-output support for an AI assistant.

- **OpenAI Update Available**: Engineers eager for the latest advancements took note of the [OpenAI Spring Update](https://www.youtube.com/watch?v=DQacCB9tDaw), which includes information on **GPT-4o**, ChatGPT enhancements, and possibly more, streamed live on May 13, 2024.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

**AlphaFold Goes Social**: The **AlphaFold3 Federation** has sprung into action, inviting participants to a meet on May 12th at 9pm EST focusing on updates and pipeline development, with an open invitation link [here](https://lu.ma/swinnyfl).

**Fasteval on the Brink**: The **fasteval** project seems to be ending, but hope remains for someone to assume the helm; the current maintainers are open to transferring the project found on [GitHub](https://github.com/), or else they suggest archiving it.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Need for Speed Customization?**: There's interest in personalizing the **AI Town** experience; specifically, adjusting the **character moving speed** and **number of NPCs**. This feedback indicates user desire for more control over gameplay mechanics.
  
- **Balancing NPC Interactions**: A user suggested optimizing **AI Town** by reducing **NPC interaction frequency** to improve **player-NPC interaction** quality. They emphasized the performance challenges when running **AI Town** locally with the **llama3 model**.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **A Casual Share for Tech Enthusiasts**: User pradeep1148 shared a [YouTube video](https://www.youtube.com/watch?v=KQ-xGVFHDkw) in the **#[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** channel, which may be of interest to fellow AI engineers. The content of the video has not been described, so its relevance to the technical discussions is unknown.



---



## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **Consensus in AI Discussions Achieved**: The notorious brevity of [pranay01's](https://discord.com/channels/958905134119784489/960713746702020608/) response with a simple "Agree!" reflects either alignment or the conclusion of a discussion on a potentially complex AI infrastructure topic. No further context was provided to detail the nature of the agreement.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1238761392711012402)** (833 messages🔥🔥🔥): 

- **Community criticizes OpenAI's regulatory moves**: Members discussed OpenAI's GPU signing and collaboration with the White House as moves to monopolize and control the AI space. One noted that OpenAI wants to make authorization mandatory, restricting competition (*"god i hate regulations on anything tech when it benefits only the top companies"*).

- **Support expands for 'WizardLM' despite controversy**: Members shared links to resources on the contentious WizardLM-2-8x22B model. Participants highlighted that it was initially released by Microsoft and later censored due to its similarity to GPT-4 ([WizardLM GitHub](https://huggingface.co/alpindale/WizardLM-2-8x22B)).

- **Discord members discuss efficient fine-tuning and new tools**: Various tools and kernels like ThunderKittens were discussed for improving model training and inference. A new kernel, ThunderKittens, was noted for its promise to outperform Flash Attention 2 ([ThunderKittens GitHub](https://github.com/HazyResearch/ThunderKittens)).

- **Unsloth receives praise and updates**: Users expressed appreciation for Unsloth's library for fine-tuning models efficiently. Unsloth announced upcoming multi-GPU support and the integration of models such as Qwen's recent versions without a specific additional branch requirement ([Unsloth GitHub](https://github.com/unslothai/unsloth)).

- **Fine-tuning challenges with Llama models discussed**: Members shared experiences and troubleshooting tips around fine-tuning processes, specifically with providing dataset sizes and padding issues. Converting and handling different model formats like FP16 to GGUF was also a notable topic.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.together.ai/blog/thunderkittens">ThunderKittens: A Simple Embedded DSL for AI kernels</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.08787">Rethinking Machine Unlearning for Large Language Models</a>: We explore machine unlearning (MU) in the domain of large language models (LLMs), referred to as LLM unlearning. This initiative aims to eliminate undesirable data influence (e.g., sensitive or illega...</li><li><a href="https://ollama.com/eramax/nxcode-cq-7b-orpo">eramax/nxcode-cq-7b-orpo</a>: https://huggingface.co/NTQAI/Nxcode-CQ-7B-orpo</li><li><a href="https://huggingface.co/">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/alpindale/WizardLM-2-8x22B">alpindale/WizardLM-2-8x22B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NTQAI/Nxcode-CQ-7B-orpo">NTQAI/Nxcode-CQ-7B-orpo · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/tiiuae/falcon-11B">tiiuae/falcon-11B · Hugging Face</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1789659394302718373">Tweet from Daniel Han (@danielhanchen)</a>: Was fixing LLM fine-tuning bugs and found 4 issues:  1. Mistral: HF&#39;s batch_decode output is wrong 2. Llama-3: Be careful of double BOS 3. Gemma: 2nd token has an extra space - GGUF(_Below) = 3064...</li><li><a href="https://tenor.com/view/joy-dadum-wow-drums-gif-14023303">Joy Dadum GIF - Joy Dadum Wow - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/lyogavin/Anima/tree/main/air_llm#quickstart">Anima/air_llm at main · lyogavin/Anima</a>: 33B Chinese LLM, DPO QLORA, 100K context, AirLLM 70B inference with single 4GB GPU - lyogavin/Anima</li><li><a href="https://tenor.com/view/gojo-satoru-gojo-ohio-gif-27179630">Gojo Satoru Gojo GIF - Gojo Satoru Gojo Ohio - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/HazyResearch/ThunderKittens">GitHub - HazyResearch/ThunderKittens: Tile primitives for speedy kernels</a>: Tile primitives for speedy kernels. Contribute to HazyResearch/ThunderKittens development by creating an account on GitHub.</li><li><a href="https://github.com/lilacai/lilac">GitHub - lilacai/lilac: Curate better data for LLMs</a>: Curate better data for LLMs. Contribute to lilacai/lilac development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7204">remove convert-lora-to-ggml.py by slaren · Pull Request #7204 · ggerganov/llama.cpp</a>: Changes such as permutations to the tensors during model conversion makes converting loras from HF PEFT unreliable, so to avoid confusion I think it is better to remove this entirely until this fea...</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/llamafy_qwen.py">LLaMA-Factory/scripts/llamafy_qwen.py at main · hiyouga/LLaMA-Factory</a>: Unify Efficient Fine-Tuning of 100+ LLMs. Contribute to hiyouga/LLaMA-Factory development by creating an account on GitHub.</li><li><a href="https://typst.app/docs/reference/text/lorem/">Lorem Function – Typst Documentation</a>: Documentation for the `lorem` function.</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1239230576335257611)** (15 messages🔥): 

- **OpenAI hosts Q&A for community engagement**: OpenAI’s CEO Sam Altman held a Q&A on Reddit to discuss the newly released [Model Spec](https://cdn.openai.com/spec/model-spec-2024-05-08.html), encouraging community interaction and questions. The document outlines desired model behavior in OpenAI's API and ChatGPT.

- **Mixed feelings on AI updates**: Members expressed a range of emotions about potential OpenAI updates. While there were hopes for revitalization, others felt cautious optimism or skepticism given past experiences and current market dynamics.

- **Skepticism about OpenAI releasing open-source models**: Discussion highlighted doubts about OpenAI releasing models open-source due to potential impacts on their business model and reputation. Comparisons were made to other companies like Meta, where open-source releases were either forced or strategic responses to competition.

- **Debate on the future of AI development publicity**: The channel featured a debate on whether the reporting of an "AI winter" impacts OpenAI, with consensus leaning towards minimal impact due to OpenAI's current industry status. Discussion also covered the incentives and risks associated with releasing AI models open-source.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/ChatGPT/comments/1coumbd/rchatgpt_is_hosting_a_qa_with_openais_ceo_sam/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1coumbd/rchatgpt_is_hosting_a_qa_">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1238775563502751755)** (312 messages🔥🔥): 

- **Challenges with Quantized Models and TGI**: A member highlighted that quantized models often result in sharding errors on HF dedicated inference when used with TGI. They noted that the models need to be saved in 16-bit format via `model.save_pretrained_merged(...)` to avoid issues (TGI requires 16-bit models).

- **Issues with GGUF Tokenizers**: There were discussions about tokenization issues, particularly with Gemma's GGUF models. Members noted problems like incorrect tokenization and an extra space being added to the first token.

- **Finetuning Llama3 Models on Colab**: Multiple users faced and resolved issues fine-tuning Llama3 models. One user mentioned a solution was found by saving to GGUF manually and ensuring models are saved and loaded correctly, with relevant documents and example notebooks being effective guides.

- **Multi-GPU and Multi-Cloud Discussions**: There were suggestions and debates on multi-GPU and cloud-based training options. Some members voiced concerns about high prices and proposed potential partnerships with cloud providers to offer cost-effective solutions for commercial users.

- **Issues with Unsloat Installation on Colab**: Problems related to installing and importing Unsloth on Colab were addressed. Solutions included ensuring the correct runtime settings, particularly GPU settings, and following instructions precisely.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/wiki#ollama-guide---unsloth-fastlanguagemodel">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=yFfaXG0WsQuE)">Google Colab</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/datasets/en/loading#json">Load</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/d3a33a0dc3cabd3b3c0dba0255fb4919db44e3b5/unsloth/__init__.py#L18">unsloth/unsloth/__init__.py at d3a33a0dc3cabd3b3c0dba0255fb4919db44e3b5 · unslothai/unsloth</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/hyperlearn">GitHub - unslothai/hyperlearn: 2-2000x faster ML algos, 50% less memory usage, works on all hardware - new and old.</a>: 2-2000x faster ML algos, 50% less memory usage, works on all hardware - new and old. - unslothai/hyperlearn</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: I got unsloth running in native windows, (no wsl). You need visual studio 2022 c++ compiler, triton, and deepspeed. I have a full tutorial on installing it, I would write it all here but I’m on mob...</li><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollTo=vITh0KVJ10qX">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKk">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.re">Sou Cidadão - Colab</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1239271994239877130)** (1 messages): 

- **SauravMaheshkar shares Llama finetuned variants**: A member has been working on finetuning **Llama variants for Token Classification** and has uploaded some of the model weights to the 🤗 hub. The fine-tuned variants include `unsloth/llama-2-7b-bnb-4bit` trained on the **conll2003 dataset** using LoRA adapters, and they shared a [collection link](https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188).

**Link mentioned**: <a href="https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188">LlamaForTokenClassification - a SauravMaheshkar Collection</a>: no description found

  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1238754156731437087)** (976 messages🔥🔥🔥): 

- **Discord user wonders about SD3's existence**: Users speculated about the Stability AI's upcoming SD3, questioning if it would ever be released. Sentiments varied with some expressing disappointment over missed release dates and others humorously comparing the situation to "Half-Life 3."

- **Expertise needed for Fine-Tuning in SDXL**: A plea for assistance with fine-tuning Stable Diffusion XL for generating product ads drew responses. One experienced user offered to help, showcasing their past work on the ML backend of creativio.ai.

- **Locating and using models for AI tasks proves challenging**: Users discussed downloading and running large language models, like [CohereForAI's C4AI Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus), and the complicated process of configuring software like KoboldAI and OogaBooga. Frustrations were expressed over the difficulty and large file sizes involved.

- **Recognizing and animating art styles**: Users suggested studying art history or using tools like gpt-4 to identify art styles. For slight animations close to the original image, it was recommended to use methods like animatediff with controlnet tile.

- **Challenges with image upscaling**: A user faced difficulties in finding an effective method for upscaling images using Automatic1111's forge with controlnet. They sought advice on achieving high-quality, detailed upscales.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/dranger003/c4ai-command-r-v01-iMat.GGUF">dranger003/c4ai-command-r-v01-iMat.GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Lewdiculous/Average_Normie_l3_v1_8B-GGUF-IQ-Imatrix">Lewdiculous/Average_Normie_l3_v1_8B-GGUF-IQ-Imatrix · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-v01">CohereForAI/c4ai-command-r-v01 · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=GM-e46xdcUo">jonathan frakes telling you you&#39;re wrong for 47 seconds</a>: it never happened</li><li><a href="https://www.youtube.com/watch?v=AdQxgvRnfhc">Nikolas Cruz&#39;s Depraved Google Search History</a>: A glimpse of the Parkland shooter&#39;s descent into the dark bowels of the Internet. This is why parents should monitor what their children do online. Warning: ...</li><li><a href="https://github.com/nullquant/ComfyUI-BrushNet">GitHub - nullquant/ComfyUI-BrushNet: ComfyUI BrushNet nodes</a>: ComfyUI BrushNet nodes. Contribute to nullquant/ComfyUI-BrushNet development by creating an account on GitHub.</li><li><a href="https://youtu.be/I8hZyJhhIEU?si=BCnamxpvbM1gTEUB">PORK NIGHTMARE</a>: ATTENTION !!! Âmes sensibles s&#39;abstenir car vous regardez en face certaines des choses les plus sombres que votre humanité à engendré.ARRÊTEZ-TOUT !!!▼▼ RÉSE...</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-v01-iMat.GGUF/resolve/main/ggml-c4ai-command-r-v01-q8_0.gguf?download=true">no title found</a>: no description found</li><li><a href="https://github.com/KoboldAI/KoboldAI-Client">GitHub - KoboldAI/KoboldAI-Client</a>: Contribute to KoboldAI/KoboldAI-Client development by creating an account on GitHub.</li><li><a href="https://github.com/Zuellni/ComfyUI-ExLlama-Nodes?tab=readme-ov-file">GitHub - Zuellni/ComfyUI-ExLlama-Nodes: ExLlamaV2 nodes for ComfyUI.</a>: ExLlamaV2 nodes for ComfyUI. Contribute to Zuellni/ComfyUI-ExLlama-Nodes development by creating an account on GitHub.</li><li><a href="https://github.com/LostRuins/koboldcpp">GitHub - LostRuins/koboldcpp: A simple one-file way to run various GGML and GGUF models with KoboldAI&#39;s UI</a>: A simple one-file way to run various GGML and GGUF models with KoboldAI&#39;s UI - LostRuins/koboldcpp</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cg5zky/sd3_release/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://cobaltexplorer.com/2023/06/character-sheets-for-stable-diffusion/">Character Consistency in Stable Diffusion - Cobalt Explorer</a>: UPDATED: 07/01&#8211; Changed templates so it&#8217;s easier to scale to 512 or 768&#8211; Changed ImageSplitter script to make it more user friendly and added a GitHub link to it&#8211; Added section...
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1239631044395929685)** (2 messages): 

- **GPT-4o offers free public access with limitations**: OpenAI announced the launch of **GPT-4o** and features like browse, data analysis, and memory available to everyone for free, but with usage limits. Plus users will enjoy up to 5x higher limits and early access to features such as the macOS desktop app and next-gen voice and video capabilities. [More info](https://openai.com/index/gpt-4o-and-more-tools-to-chatgpt-free/).
- **Introducing GPT-4o with multi-modal capabilities**: OpenAI's new flagship model, **GPT-4o**, can reason in real-time across audio, vision, and text. Text and image input capabilities are available in the API and ChatGPT now, with voice and video features to follow in the coming weeks. [Details here](https://openai.com/index/hello-gpt-4o/).
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1238835920703193191)** (684 messages🔥🔥🔥): 

- **GPT-4o debuts with mixed reviews**: Members discussed the new GPT-4o's performance, noting it is faster and cheaper but with inconsistencies in reasoning and shorter memory compared to GPT-4. Some users appreciated its abilities, while others found GPT-4 to have better reasoning capabilities for custom instructions.
- **Rollout confusion and feature anticipation**: Members experienced varied rollout times for access to GPT-4o, both through API and ChatGPT. There was noticeable enthusiasm for upcoming features like real-time camera use and new voice capabilities, though these have not been fully rolled out yet.
- **Classic vs. New Model debate**: Users debated the practicality of maintaining GPT-4 when GPT-4o is available, considering the latter's lower cost and fast performance. Some pointed out specific cases where GPT-4 still performed better, leading to mixed decisions on which model to use.
- **Feature accessibility queries**: Queries about the availability of specific features like the new macOS app, visual capabilities, and voice cloning in the GPT-4o API were prominent. It was clarified that many of these features would be gradually available in the coming weeks.
- **General excitement and skepticism**: The community expressed a blend of excitement and skepticism regarding the new updates, with many looking forward to broader access and testing the new features in real-world applications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/twerk-dance-dog-funny-cute-gif-19259275">Twerk Dance GIF - Twerk Dance Dog - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=DQacCB9tDaw">Introducing GPT-4o</a>: OpenAI Spring Update – streamed live on Monday, May 13, 2024. Introducing GPT-4o, updates to ChatGPT, and more.</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8?">Sync codebase · openai/tiktoken@9d01e56</a>: no description found</li><li><a href="https://huggingface.co/models">Models - Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1238754323366936636)** (126 messages🔥🔥): 

```html
- **Issues Passing Files to GPT Actions**: A member asked if anyone figured out how to pass uploaded files to a GPT action. There wasn't a clear resolution provided in the discussion.

- **GPT-4T API Provides Higher Context**: Discussion highlighted that the API for GPT-4T is less restrained and currently allows a 128k context. Members discussed the nuances of this capability.

- **Random Output with High Temperature Settings**: A member experienced random outputs when setting the temperature above 1.5. Another advised keeping the temperature below 1 for stable and coherent responses.

- **Fetching OpenAI Model Pricing**: Members shared that OpenAI pricing is static and can be reviewed on the [OpenAI pricing page](https://openai.com/api/pricing/). There are no alerts for pricing changes, so users need to monitor the page manually.

- **Custom GPTs and Cross-Session Memory**: There was confusion about custom GPTs' cross-session memory capabilities, clarified by a member noting that per-GPT memories have not rolled out yet. More details about this can be found in the [OpenAI Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq).
```
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1239279732961443841)** (32 messages🔥): 

- **Moderation Filter Issue with Gemini 1.5**: A user reported that their application consistently fails to respond to queries related to "romance package" due to an unspecified moderation filter. Despite setting all blocks to none and trying different settings, the issue persists, making it difficult to implement their integrations at a major resort.
- **Discussion on Safety Settings**: Members discussed whether the problem with the moderation filter could be due to safety settings not being explicitly disabled. One member suggested testing in the AI Lab to ensure no syntax errors are affecting the results.
- **API Keys and Temperature Settings Experimentation**: The user tried generating new API keys and adjusting temperature settings to resolve the issue but had no success. This has led them to conclude that the problem might be on Google's end.
- **Help Offered and Syntax Check Recommended**: Another member offered help and suggested checking the syntax in the AI Lab to confirm that the issue is not due to improper syntax or safety filter settings. The user appreciated this assistance but remained convinced that the problem is external.
- **Python Script for File Operations**: A user shared a Python script snippet that outlines creating a directory, writing Python files in separate sessions, and zipping the directory. This script demonstrates a method for displaying a link to download the resulting zip file.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1239279732961443841)** (32 messages🔥): 

- **Moderation filter issue in Gemini 1.5**: A user reported an issue with their application experiencing consistent failures when users inquire about "romance package" or similar topics. Despite changing defaults and generating new API keys, the problem persists, suggesting potential model training restrictions.
- **Troubleshooting AI Safety Settings**: Another user suggested explicitly disabling safety settings to potentially resolve the issue. They stressed the importance of ensuring that safety filters are correctly turned off and offered a screenshot method for further verification.
- **Google AI Lab potential solution**: The conversation shifted to testing in Google AI Lab to determine if syntax errors are the cause. Suggestions included checking the safety filters and possibly testing for syntax errors in the lab.
- **File directory creation in Python**: A user requested guidance on creating a full file tree, writing files in Python sessions, and zipping a directory, asking for a downloadable link upon completion. The task involves programmatically setting up a directory structure and managing files through Python scripts.
  

---


**OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1239532612515663942)** (2 messages): 

- **Creating a ChatGPT clone with tracking**: A user inquired about the feasibility of creating a **ChatGPT clone** utilizing the **3.5 model** but with the capability for user messages to be monitored by the organization. This implies replicating the ChatGPT interface while adding a message tracking feature.
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/)** (1 messages): 

king.of.kings_: i am struggling to get llama 3 70b to be coherent over 8k tokens lol
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1238792963015053333)** (15 messages🔥): 

- **Aurora in France**: A member mentioned seeing the **aurora borealis** over the central volcano of Arvenia in Auvergne, France. This surreal natural phenomenon caught their attention and seemed worth sharing.

- **YouTube Links Shared**: Two YouTube links were shared: one titled ["Udio Testing: You never knew your own name : whispers in the void"](https://youtu.be/03eHNJzEYcA?si=DV2OToN0h57W7tkv) and another by another member, without additional descriptions.

- **Introducing MAP-Neo**: A user announced the release of **MAP-Neo**, a fully transparent bilingual LLM trained on 4.5T tokens, and shared links to [Hugging Face](http://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04), a [dataset](https://huggingface.co/datasets/m-a-p/Matrix), and the [GitHub repository](https://github.com/multimodal-art-projection/MAP-NEO).

- **Kingdom Come: Deliverance Cooking Mechanic**: A user discussed the perpetual stew mechanic in the game **Kingdom Come: Deliverance**, noting its historical accuracy. They shared a personal recipe involving slow-cooking vegetables and meat, highlighting a shift in cooking methods based on hunger.

- **RPA and Software Automation**: A member inquired about a library for interacting directly with **software windows via RDP**, like RPA for automation. Another member suggested using **Frida** for runtime hooks and exposing functionality via an HTTP API, although concerns were raised about the complexity due to not having access to software binaries.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/mother-day-gif-12554356809887397003">Mother Day GIF - Mother day - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/03eHNJzEYcA?si=DV2OToN0h57W7tkv">Udio Testing: You never knew your own name : whispers in the void</a>: no description found</li><li><a href="http://huggingface.co/collections/m-a-p/neo-models-66395a5c9662bb58d5d70f04">Neo-Models - a m-a-p Collection</a>: no description found</li><li><a href="https://huggingface.co/datasets/m-a-p/Matrix">m-a-p/Matrix · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/multimodal-art-projection/MAP-NEO">GitHub - multimodal-art-projection/MAP-NEO</a>: Contribute to multimodal-art-projection/MAP-NEO development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1238852402564825168)** (6 messages): 

- **Hierarchical Correlation Reconstruction in Neural Networks**: A member posted a link to an [arXiv paper](https://arxiv.org/abs/2405.05097) discussing optimization of artificial neural networks through hierarchical correlation reconstruction. The paper contrasts typical unidirectional value propagation with the multidirectional operation of biological neurons.

- **Taskmaster Episode Roleplay App**: Another member shared their creation of a React app for roleplaying as a Taskmaster contestant, using a state machine that encodes each stage of an episode. Users need to input their own OpenAI key and may encounter clunky outputs but can check out [the code on GitHub](https://github.com/LEXNY/Taskmaster-LLM/blob/main/src/App.js).

- **Yi-1.5-34B-Chat Model Update**: One message highlighted the **01-ai/Yi-1.5-34B-Chat** model on Hugging Face. It was updated recently and had over a thousand uses, as seen [here](https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8).

- **Detailed Industrial Military Complex Knowledge Graph**: A member used Mistral 7B instruct v 0.2 and their framework, llama-cpp-agent, to create a 40-node knowledge graph of the Industrial Military Complex. They shared the [framework on GitHub](https://github.com/Maximilian-Winter/llama-cpp-agent) which supports various servers and APIs like llama.cpp and TGI.

- **Detailed Thoughts on OpenAI's Technology and Strategy**: A user linked to a [Twitter thread](https://twitter.com/drjimfan/status/1790089671365767313) offering a deep dive into OpenAI's advancements in audio-to-audio mapping and video streaming to transformers. It speculates on OpenAI's strategic moves and potential Apple integrations with GPT-4o as a precursor to GPT-5.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.05097">Biology-inspired joint distribution neurons based on Hierarchical Correlation Reconstruction allowing for multidirectional neural networks</a>: Popular artificial neural networks (ANN) optimize parameters for unidirectional value propagation, assuming some guessed parametrization type like Multi-Layer Perceptron (MLP) or Kolmogorov-Arnold Net...</li><li><a href="https://huggingface.co/collections/01-ai/yi-15-2024-05-663f3ecab5f815a3eaca7ca8">Yi-1.5 (2024/05) - a 01-ai Collection</a>: no description found</li><li><a href="https://github.com/LEXNY/Taskmaster-LLM/blob/main/src/App.js">Taskmaster-LLM/src/App.js at main · LEXNY/Taskmaster-LLM</a>: Contribute to LEXNY/Taskmaster-LLM development by creating an account on GitHub.</li><li><a href="https://github.com/Maximilian-Winter/llama-cpp-agent">GitHub - Maximilian-Winter/llama-cpp-agent: The llama-cpp-agent framework is a tool designed for easy interaction with Large Language Models (LLMs). Allowing users to chat with LLM models, execute structured function calls and get structured output. Works also with models not fine-tuned to JSON output and function calls.</a>: The llama-cpp-agent framework is a tool designed for easy interaction with Large Language Models (LLMs). Allowing users to chat with LLM models, execute structured function calls and get structured...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1238772042384408587)** (741 messages🔥🔥🔥): 

- **OpenAI's GPT-4o divides opinions**: Members discussed the launch of GPT-4o, noting its dual input-output capabilities and improved coding performance. There was significant debate about its lower token limit (2048) and potential impact on the open-source AI community.
- **Speed improvements with mixed feelings**: Users noted GPT-4o's increased speed and lower costs, attributing the efficiency to potential quantization and model size reductions. Despite these benefits, some were disappointed by the limited token output and pricing.
- **Concerns about OpenAI's competitive strategies**: Several members expressed frustration with OpenAI's approach, feeling it aims to dominate the market and marginalize open-source alternatives. This sentiment highlights ongoing tension within the AI community about proprietary vs. open-source models.
- **Technical demonstrations and issues**: Members tested GPT-4o's capabilities in various scenarios, including API performance and mathematical reasoning. Some observed inconsistent results and speculated about the causes, such as potential quantization artifacts or model limitations.
- **Impact on specialized services**: Discussions also touched on how GPT-4o's features might affect companies focusing on specialized services like audio generation and multimodal capabilities, with ElevenLabs mentioned as a potentially impacted entity.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.cambioml.com">cambioml</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/llama-3-cat-8b-instruct-GGUF">mradermacher/llama-3-cat-8b-instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/refuelai/Llama-3-Refueled">refuelai/Llama-3-Refueled · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/01-ai/Yi-1.5-34B-Chat/blob/main/ggml-model-Q4_K_M.gguf">ggml-model-Q4_K_M.gguf · 01-ai/Yi-1.5-34B-Chat at main</a>: no description found</li><li><a href="https://blog.composio.dev/gpt-4-function-calling-example/">Improving GPT 4 Function Calling Accuracy</a>: Join our Discord Community and check out what we&#x27;re building!  We just published Part 2 of the blog comparing gpt-4-turbo vs opus vs haiku vs sonnet .   Introduction to GPT Function Calling  Larg...</li><li><a href="https://oobabooga.github.io/benchmark.html">oobabooga benchmark</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.18824">Benchmarking Benchmark Leakage in Large Language Models</a>: Amid the expanding use of pre-training data, the phenomenon of benchmark dataset leakage has become increasingly prominent, exacerbated by opaque training processes and the often undisclosed inclusion...</li><li><a href="https://tenor.com/view/cats-animals-reaction-wow-surprised-gif-20914356">Cats Animals GIF - Cats Animals Reaction - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/TheSkullery/llama-3-cat-8b-instruct-v1">TheSkullery/llama-3-cat-8b-instruct-v1 · Hugging Face</a>: no description found</li><li><a href="https://github.com/interstellarninja/MeeseeksAI">GitHub - interstellarninja/MeeseeksAI: A framework for orchestrating AI agents using a mermaid graph</a>: A framework for orchestrating AI agents using a mermaid graph - interstellarninja/MeeseeksAI</li><li><a href="https://x.com/willdepue/status/1790078289023062255?s=46&t=bL0EKkuCqv4FWSLQ7lV-2w">Tweet from will depue (@willdepue)</a>: i think people are misunderstanding gpt-4o. it isn&#39;t a text model with a voice or image attachment. it&#39;s a natively multimodal token in, multimodal token out model.  you want it to talk fast? ...</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">Sync codebase · openai/tiktoken@9d01e56</a>: no description found</li><li><a href="https://x.com/wenhuchen/status/1789685187804029285?s=46">Tweet from Wenhu Chen (@WenhuChen)</a>: Big News!  Meet our strongest fully open-source 7B-LLM Neo.  We release its 4.7T pre-training data Matrix and entire codebase at MAP-Neo!  1. Neo-7B beats the existing fully open-source models like OL...</li><li><a href="https://github.com/Potatooff/Le-Potato">GitHub - Potatooff/Le-Potato: Simple. elegant LLM Chat Inference</a>: Simple. elegant LLM Chat Inference. Contribute to Potatooff/Le-Potato development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/Replete-AI/code_bagel_hermes-2.5">Replete-AI/code_bagel_hermes-2.5 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/pull/30621">Chat Template support for function calling and RAG by Rocketknight1 · Pull Request #30621 · huggingface/transformers</a>: This PR updates our support of chat templates to cover tool-use and RAG use-cases. Specifically, it does the following:  Defines a recommended JSON schema spec for tool use Adds tools and documents...</li><li><a href="https://github.com/interstellarninja/MeeseeksAI/blob/2399588acdee06cff4af04ca091b1ab5c71580b8/src/agents.py#L72-L83">MeeseeksAI/src/agents.py at 2399588acdee06cff4af04ca091b1ab5c71580b8 · interstellarninja/MeeseeksAI</a>: A framework for orchestrating AI agents using a mermaid graph - interstellarninja/MeeseeksAI</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d">Sync codebase · openai/tiktoken@9d01e56</a>: no description found</li><li><a href="https://fxtwitter.com/almost_digital/status/1788877760120692994">Tweet from Johan Nordberg (@almost_digital)</a>: I joined @elevenlabsio in January and It’s been an absolute blast working with @flavioschneide on this!  This one is generated from a single text prompt “rap about never stopping to learn”, lyrics inc...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1238877292395102268)** (48 messages🔥): 

- **Exploration into MoE Architectures for Attention Blocks**: Members discussed the structure of MoE (Mixture of Experts) architectures, particularly questioning whether attention blocks are included. It was noted that traditionally only FFN layers are part of MoE, though MoE attention has been explored in research.

- **Combining Autoregressive Models with Diffusion Models**: There was curiosity about the feasibility of merging autoregressive models favored for text with diffusion models used for images to create a robust multimodal model. A member sought validation and ideas on this concept, indicating a blend of architectures might offer enhanced performance.

- **Understanding and Using Prompt Templates**: The conversation covered different formats for prompt templates, explaining their importance in model responses. Specific formats like the Alpaca Prompt Format and ChatML were discussed, alongside best practices depending on the model used, like Hermes.

- **Preventing Models from Giving Canned "Life Lessons"**: Members brainstormed methods to stop models from defaulting to general "safe" responses when they detect potentially unsafe inputs. System prompts and specific tuning techniques were suggested as solutions, including a resource on HuggingFace and an article on the Alignment Forum about mitigating refusal behaviors in models.

- **Challenges in Fine-Tuning with Specific Datasets**: One member sought advice on fine-tuning llama3 using the dolphin-2.9 dataset but encountered issues with torchtune and compatibility. After some troubleshooting and useful tips, including updating flash-attn and resolving MPI dependencies, they managed to progress with their setup.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2305.18295">RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths</a>: Text-to-image generation has recently witnessed remarkable achievements. We introduce a text-conditional image diffusion model, termed RAPHAEL, to generate highly artistic images, which accurately por...</li><li><a href="https://huggingface.co/failspy/llama-3-70B-Instruct-abliterated/blob/main/ortho_cookbook.ipynb">ortho_cookbook.ipynb · failspy/llama-3-70B-Instruct-abliterated at main</a>: no description found</li><li><a href="https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — AI Alignment Forum</a>: This work was produced as part of Neel Nanda&#x27;s stream in the ML Alignment &amp; Theory Scholars Program - Winter 2023-24 Cohort, with co-supervision from…
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1238780733619965952)** (5 messages): 

- **New ChatQA Model Outperforms GPT-4**: An arXiv paper titled [ChatQA](https://arxiv.org/abs/2401.10225) introduces conversational QA models that achieve GPT-4 level accuracies. ChatQA-70B reportedly outperforms GPT-4 on 10 conversational QA datasets without relying on synthetic data from OpenAI GPT models.

- **InstructLab Enhances LLMs Without Full Retraining**: IBM/Redhat's new [InstructLab](https://github.com/instructlab) project adds new skills and knowledge to LLMs using a large model as a teacher and a taxonomy to generate synthetic datasets. This framework allows incremental additions to models through curated datasets and weekly builds.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2401.10225">ChatQA: Building GPT-4 Level Conversational QA Models</a>: In this work, we introduce ChatQA, a family of conversational question answering (QA) models that obtain GPT-4 level accuracies. Specifically, we propose a two-stage instruction tuning method that can...</li><li><a href="https://github.com/instructlab">InstructLab</a>: InstructLab has 10 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1238765655940005959)** (22 messages🔥): 

- **Websim's popularity rises**: Members expressed excitement about Websim, a platform described as *"a really cool business/startup simulator"* and actively shared links to build bases and explore different scenarios. One of the shared links was [websim.ai](https://websim.ai/c/B8MJwg44rDhdQmJYB).

- **Consent is trending**: In a playful interaction, a member highlighted the importance of *consent* humorously suggesting *"consent is haut!"* This was part of a message sharing a link to [websim.ai](https://websim.ai/c/grXqLcCAxEGNz3TyH).

- **Discussion on simulation platforms**: Members showed interest in expanding the capabilities and functionalities of world simulation tools. For example, one mentioned needing a *Digital Audio Workstation (DAW) / VSTs in worldclient*, referring to its potential utility.

- **Bug reports and technical issues**: Several members noted technical issues with WorldSim, such as commands like *"!back"* inadvertently restarting the simulator and problems with context not clearing when requested. They also mentioned responses getting cut off and issues with typing characters.

- **Invitation to philosophy and websim salon**: There was an invitation extended by a member for others to join a *philosophy and websim salon* in chat, indicating interest in deeper discussions within the community. They coordinated on time zones to facilitate participation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://websim.ai/c/B8MJwg44rDhdQmJYB">generative.ink/chat/</a>: no description found</li><li><a href="https://websim.ai/c/grXqLcCAxEGNz3TyH">generative.ink/chat/</a>: no description found
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1238778471841533992)** (93 messages🔥🔥): 

- **Top NLP PhD spotlight**: A member called attention to a notable NLP PhD thesis, sharing a [Twitter link](https://twitter.com/ShunyuYao12/status/1789058769982550031) with accolades. Another member humorously remarked on the impressive CV.

- **Data shortage discussion**: A fascinating thread on data shortage was shared, noting that [Llama 3](https://x.com/mark_cummins/status/1788949893903511705?s=46&t=90xQ8sGy63D2OtiaoGJuww) trained on 15 trillion tokens. This sparked a conversation about data claims and the sources behind them, highlighting Stella Biederman's differing views.

- **Infrastructure services for AI agents**: A member from Singapore shared a draft on new infrastructure services for AI agents and invited feedback, directing interested individuals to [their Substack post](https://sweekiat.substack.com/p/d8726e73-e717-4599-81a3-5eb82e48f9c9).

- **Falcon 2 release**: The community discussed the launch of [Falcon 2](https://falconllm.tii.ae/falcon-2.html), noting its open-source, multilingual, and multimodal capabilities. Despite its impressive features, concerns were raised about the licensing terms, which some found restrictive.

- **GPT-4o excitement**: Members actively engaged in discussions about the new GPT-4o, sharing [various insights and links](https://openai.com/index/hello-gpt-4o/), including speculation on voice latency and features. Some contemplated its API access and performance, with links to API documentation and real-time observations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/blader/status/1790088659053719736?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Siqi Chen (@blader)</a>: this will prove to be in retrospect by far the most underrated openai event ever  openai casually dropping text to 3d rendering in gpt4o and not even mentioning it   (more 👇🏼)</li><li><a href="https://x.com/gdb/status/1790077263708340386">Tweet from Greg Brockman (@gdb)</a>: GPT-4o can also generate any combination of audio, text, and image outputs, which leads to interesting new capabilities we are still exploring.  See e.g. the &#34;Explorations of capabilities&#34; sec...</li><li><a href="https://x.com/Karmedge/status/1790084650582397118">Tweet from Robert Lukoszko — e/acc (@Karmedge)</a>: I am 80% sure openAI has extremely low latency low quality model get to pronounce first 4 words in &lt;200ms and then continue with the gpt4o model  Just notice, most of the sentences start with “Sure...</li><li><a href="https://x.com/lmsysorg/status/1790097588399779991">Tweet from lmsys.org (@lmsysorg)</a>: Breaking news — gpt2-chatbots result is now out!  gpt2-chatbots have just surged to the top, surpassing all the models by a significant gap (~50 Elo). It has become the strongest model ever in the Are...</li><li><a href="https://falconllm.tii.ae/falcon-2.html">Falcon LLM</a>: Generative AI models are enabling us to create innovative pathways to an exciting future of possibilities - where the only limits are of the imagination.</li><li><a href="https://www.bloomberg.com/news/articles/2024-05-11/apple-closes-in-on-deal-with-openai-to-put-chatgpt-on-iphone">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/jacobcolling/status/1790073742514663866?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Jake Colling (@JacobColling)</a>: @simonw @OpenAI Using the model  `gpt-4o` seems to work for my API access</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">Sync codebase · openai/tiktoken@9d01e56</a>: no description found</li><li><a href="https://www.latent.space/s/university">AI for Engineers | Latent Space | swyx &amp; Alessio | Substack</a>: a 7 day foundational course for prospective AI Engineers, developed with Noah Hein. NOT LIVE YET - we are 5/7 complete. Sign up to get it when it releases! Click to read Latent Space, a Substack publi...</li><li><a href="https://x.com/andykreed/status/1790082413428629843">Tweet from tweet davidson 🍞 (@andykreed)</a>: ChatGPT voice is…hot???</li><li><a href="https://x.com/mark_cummins/status/1788949893903511705?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Mark Cummins (@mark_cummins)</a>: Llama 3 was trained on 15 trillion tokens (11T words). That’s large - approximately 100,000x what a human requires for language learning</li><li><a href="https://x.com/mark_cummins/status/1788949893903511705?s=46&t=90xQ8sGy63">Tweet from Mark Cummins (@mark_cummins)</a>: Llama 3 was trained on 15 trillion tokens (11T words). That’s large - approximately 100,000x what a human requires for language learning</li><li><a href="https://x.com/juberti/status/1790126140784259439">Tweet from Justin Uberti (@juberti)</a>: Had a chance to try the gpt-4o API from us-central and  text generation is quite fast. Comparing to http://thefastest.ai, this perf is 5x the TPS of gpt-4-turbo and similar to many llama-3-8b deployme...</li><li><a href="https://sweekiat.substack.com/p/d8726e73-e717-4599-81a3-5eb82e48f9c9">Something to do something</a>: Click to read Something to do something, by sweekiat, a Substack publication. Launched 2 years ago.</li><li><a href="https://x.com/karmedge/status/1790084650582397118?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Robert Lukoszko — e/acc (@Karmedge)</a>: I am 80% sure openAI has extremely low latency low quality model get to pronounce first 4 words in &lt;200ms and then continue with the gpt4o model  Just notice, most of the sentences start with “Sure...</li><li><a href="https://x.com/drjimfan/status/1790089671365767313?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Jim Fan (@DrJimFan)</a>: I know your timeline is flooded now with word salads of &#34;insane, HER, 10 features you missed, we&#39;re so back&#34;. Sit down. Chill. &lt;gasp&gt; Take a deep breath like Mark does in the demo &l...</li><li><a href="https://x.com/mark_cummins/status/1788949945795424522">Tweet from Mark Cummins (@mark_cummins)</a>: Up next is code. Code is a very important text type, and the amount of it surprised me. There’s 0.75T tokens of public code. Total code ever written might be as much as 20T, though much of this is pri...</li><li><a href="https://news.ycombinator.com/item?id=40344302">Falcon 2 | Hacker News</a>: no description found
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1239418270302339115)** (1 messages): 

- **Pre-event Hype for OpenAI Watch Party**: A member announced a **watch party** for an OpenAI event happening tomorrow. The pregame starts at 9:30, half an hour before the event, and more details can be found on the [Discord event link](https://discord.gg/Z7V4NDGZ?event=1238918257046458368).

**Link mentioned**: <a href="https://discord.gg/Z7V4NDGZ?event=1238918257046458368">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1239616941677609064)** (710 messages🔥🔥🔥): 

- **Audio issues plagued Open AI Spring Event Watch Party**: Members experienced audio problems initially during the Open AI Spring Event Watch Party, where viewers could not hear the stream host. Suggestions to drop and rejoin helped mitigate some issues.

- **Speculations and reactions to Apple and Google tech**: Participants speculated about Apple's challenges and the potential of Apple licensing tech, emphasizing Siri's inferiority. A link was shared for a discussion on Twitter about whether Apple will adopt integrations discussed for iOS 18 due to potential Gemini and antitrust concerns [related tweet](https://twitter.com/youraimarketer/status/1789918014617399355).

- **GPT-4o steals the show**: The new GPT-4o model was highlighted as available for free in ChatGPT, capturing attention with discussions on performance, cost, and its availability without a subscription. A tweet with self-leaked performance metrics was shared [related tweet](https://x.com/LiamFedus/status/1790064963966370209).

- **A.I. capabilities and live demos amazed audience**: Users were impressed by real-time demos, including voice mode updates with emotional range and interruption capability, and multimodal interactions of GPT-4o. Discussions included real-time responsiveness, voice synthesis improvements, and a demo link on YouTube [link to event](https://www.youtube.com/watch?v=DQacCB9tDaw&ab_channel=OpenAI).

- **Immediate reactions and possible competitive edge**: Excitement was shared about how these advancements might affect competitors like Google and potential impacts on applications like Siri, Copilot, and Replika, with some speculating it's a step towards human-level interaction. Comments included comparisons to existing technologies and implications for future AI agents.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://twitch.tv/yikesawjeez,">Twitch</a>: no description found</li><li><a href="https://x.com/LiamFedus/status/1790064963966370209">Tweet from William Fedus (@LiamFedus)</a>: GPT-4o is our new state-of-the-art frontier model. We’ve been testing a version on the LMSys arena as im-also-a-good-gpt2-chatbot 🙂. Here’s how it’s been doing.</li><li><a href="https://x.com/oliviergodement/status/1790070151980666982?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Olivier Godement (@oliviergodement)</a>: I haven&#39;t tweeted much about @OpenAI announcements, but I wanted to share a few reflections on GPT-4o as I&#39;ve have not been mind blown like that for a while.</li><li><a href="https://x.com/brad_agi/status/1790073505658114069">Tweet from Brad (@brad_agi)</a>: 50% cheaper isn&#39;t even competitive. Source: https://artificialanalysis.ai/</li><li><a href="https://x.com/imjaredz/status/1790074937119482094?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Jared Zoneraich (@imjaredz)</a>: gpt-4o blows gpt-4-turbo out of the water.  So quick & seemingly better answer.  Also love the split-screen playground view from @OpenAI</li><li><a href="https://en.wikipedia.org/wiki/Mechanical_Turk">Mechanical Turk - Wikipedia</a>: no description found</li><li><a href="https://x.com/bdougieyo/status/1790071113420079329?s=46">Tweet from bdougie on the internet (@bdougieYO)</a>: ChatGPT saying it looks like I am in a good mood.</li><li><a href="https://blog.samaltman.com/gpt-4o">GPT-4o</a>: There are two things from our announcement today I wanted to highlight.  First, a key part of our mission is to put very capable AI tools in the hands of people for free (or at a great price). I am...</li><li><a href="https://x.com/0xkarmatic/status/1790079694043320756">Tweet from Karma (@0xkarmatic)</a>: &#34;An ASR model, an LLM,  a TTS model… are you getting it? These are not three separate model: This is one model, and we are calling it gpt-4o.&#34;  Quoting Andrej Karpathy (@karpathy)   They are r...</li><li><a href="https://www.youtube.com/watch?v=DQacCB9tDaw">Introducing GPT-4o</a>: OpenAI Spring Update – streamed live on Monday, May 13, 2024. Introducing GPT-4o, updates to ChatGPT, and more.</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">Sync codebase · openai/tiktoken@9d01e56</a>: no description found</li><li><a href="https://x.com/gdb/status/1790079398625808837">Tweet from Greg Brockman (@gdb)</a>: We also have significantly improved non-English language performance quite a lot, including improving the tokenizer to better compress many of them:</li><li><a href="https://x.com/gdb/status/1790071008499544518?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Greg Brockman (@gdb)</a>: Introducing GPT-4o, our new model which can reason across text, audio, and video in real time.  It&#39;s extremely versatile, fun to play with, and is a step towards a much more natural form of human-...</li><li><a href="https://x.com/sama/status/1790065541262032904">Tweet from Sam Altman (@sama)</a>: it is available to all ChatGPT users, including on the free plan! so far, GPT-4 class models have only been available to people who pay a monthly subscription. this is important to our mission; we wan...</li><li><a href="https://t.co/B5iqOKm06j">GitHub - BasedHardware/OpenGlass: Turn any glasses into AI-powered smart glasses</a>: Turn any glasses into AI-powered smart glasses. Contribute to BasedHardware/OpenGlass development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1238757309254078475)** (658 messages🔥🔥🔥): 

```html
- **Cheerio Library Alternatives**: A user asked if there's a faster way than the Cheerio library to extract content from HTML strings. Another user provided a link to [Perplexity's AI search](https://www.perplexity.ai/search/Is-there-a-xOtvxOveTGSfbae88ElQMA) for further exploration.

- **ChatGPT Plus vs. Perplexity Pro**: Discussions highlighted the comparative advantages of ChatGPT Plus and Perplexity Pro, including context window sizes and general AI capabilities. Users shared their experiences, stating Perplexity as more focused on being an AI search engine with specific features such as collections and model flexibility.

- **Claude 3 Opus Limits**: Users frequently mentioned dissatisfaction with the imposed limits on Claude 3 Opus usage in Perplexity Pro. One user suggested considering YesChat as an alternative, which offers more generous usage quotas.

- **GPT-4o Release Buzz**: Conversations were abuzz with the release of GPT-4o, noting its improved speed and capabilities. There was anticipation for when Perplexity would integrate GPT-4o, with comparisons to how it might outclass existing models like Claude 3 Opus.

- **Perplexity's Context Handling**: Users discussed the effectiveness of Perplexity in handling context windows and RAG (retrieval-augmented generation). The consensus was that while 32k tokens seem standard, there is uncertainty and a desire for greater context capabilities.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/live/DQacCB9tDaw?feature=shared">Introducing GPT-4o</a>: OpenAI Spring Update – streamed live on Monday, May 13, 2024. Introducing GPT-4o, updates to ChatGPT, and more.</li><li><a href="https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/">More than an OpenAI Wrapper: Perplexity Pivots to Open Source</a>: Perplexity CEO Aravind Srinivas is a big Larry Page fan. However, he thinks he&#039;s found a way to compete not only with Google search, but with OpenAI&#039;s GPT too.</li><li><a href="https://gpt-tokenizer.dev/">gpt-tokenizer playground</a>: no description found</li><li><a href="https://x.com/inafried/status/1790083063374033046">Tweet from Ina Fried (@inafried)</a>: A couple tidbits I&#39;ve confirmed as well. 1) The mysterious GPT2-chatbot that showed up on benchmark sites was GPT-4o. 2) OpenAI did desktop version first for Mac because &#34;we&#39;re just priori...</li><li><a href="https://youtu.be/MirzFk_DSiI?si=L7uUgS21JMDRvfky">Two GPT-4os interacting and singing</a>: Say hello to GPT-4o, our new flagship model which can reason across audio, vision, and text in real time.Learn more here: https://www.openai.com/index/hello-...</li><li><a href="https://www.youtube.com/watch?v=DQacCB9tDaw">Introducing GPT-4o</a>: OpenAI Spring Update – streamed live on Monday, May 13, 2024. Introducing GPT-4o, updates to ChatGPT, and more.</li><li><a href="https://youtu.be/V6pYxfcDRks">Happy Birthday with GPT-4o</a>: Say hello to GPT-4o, our new flagship model which can reason across audio, vision, and text in real time.Learn more here: https://www.openai.com/index/hello-...</li><li><a href="https://youtu.be/MirzFk_DSiI?feature=shared">Two GPT-4os interacting and singing</a>: Say hello to GPT-4o, our new flagship model which can reason across audio, vision, and text in real time.Learn more here: https://www.openai.com/index/hello-...</li><li><a href="https://fxtwitter.com/mckaywrigley/status/1790088880919818332?s=46">Tweet from Mckay Wrigley (@mckaywrigley)</a>: This demo is insane.  A student shares their iPad screen with the new ChatGPT + GPT-4o, and the AI speaks with them and helps them learn in *realtime*.  Imagine giving this to every student in the wor...</li><li><a href="https://www.yeschat.ai/pricing">YesChat.ai Pricing Plan</a>: no description found</li><li><a href="https://azure.microsoft.com/en-us/blog/introducing-gpt-4o-openais-new-flagship-multimodal-model-now-in-preview-on-azure/">Introducing GPT-4o: OpenAI’s new flagship multimodal model now in preview on Azure | Microsoft Azure Blog</a>: OpenAI, in partnership with Microsoft, announces GPT-4o, a groundbreaking multimodal model for text, vision, and audio capabilities. Learn more.
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1239038758477631649)** (21 messages🔥): 

- **Alexandr Yarats leads Perplexity Search**: An [interview with Alexandr Yarats](https://www.unite.ai/alexandr-yarats-head-of-search-at-perplexity-interview-series/) reveals his journey from Yandex to Google, and eventually to Perplexity AI as the Head of Search. Yarats discusses his initial interest in **machine learning** driven by his background in math, probability theory, and statistics.
  
- **Understanding Bernoulli's Fallacy**: A member shared a [link](https://www.perplexity.ai/search/Explain-Bernoullis-fallacy-TGhbdqjbQWSqxHWvaWUJJQ#0) explaining Bernoulli's Fallacy. The discussion aims to clarify misconceptions about the fallacy in probability and statistics.

- **Severe Geomagnetic Storms**: Insights about severe geomagnetic storms were discussed with reference to a [search](https://www.perplexity.ai/search/Severe-geomagnetic-storm-IYre33XZQX2BdeRlkJ9z.g). The effects discussed include impacts on satellite operations and power grids.

- **Eurovision 2024 Updates**: Eurovision enthusiasts discussed updates related to the 2024 event, as detailed in a [search result](https://www.perplexity.ai/search/Eurovision-2024-LN.Prd19Sju6dGjlw7HByw).

- **Importance of Magnesium**: A user inquired about the significance of magnesium, leading to a detailed explanation available [here](https://www.perplexity.ai/search/Why-is-magnesium-7asSHXRgSKegA7NBEYkEkQ). Important for various bodily functions, the mineral's role in health was elaborated.

**Link mentioned**: <a href="https://www.unite.ai/alexandr-yarats-head-of-search-at-perplexity-interview-series/">Alexandr Yarats, Head of Search at Perplexity &#8211; Interview Series</a>: Alexandr Yarats is the Head of Search at Perplexity AI. He began his career at Yandex in 2017, concurrently studying at the Yandex School of Data Analysis. The initial years were intense yet rewarding...

  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1238969127981547663)** (4 messages): 

- **User requests Perplexity tutorial**: A user requested a tutorial for Perplexity AI, asking in Spanish *"dame un tutoria de perplexity por favor"*. This indicates some users may be seeking guidance in languages other than English.
- **Link to deep dive provided**: A user shared a link to a Discord message for a "deep dive", presumably related to a more comprehensive guide or information about Perplexity AI. [Link to deep dive](https://discord.com/channels/1047197230748151888/1047649527299055688/1197892547276705843).
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1238758307267874906)** (389 messages🔥🔥): 

- **Search for Open Source LLM Model Alternatives**: Members discussed various open-source LLM models such as **Mistral** and **LLaMa3**, which can operate with lower hardware requirements compared to the free version of ChatGPT. Mention was made of platforms like **You.com** for trying these models.
- **Debugging Stable Diffusion Pipeline**: A member provided a Python code snippet for disabling the safety checker in the **StableDiffusionPipeline** using `from_pretrained()`. Another member reported the issue of black images indicating incomplete solutions.
- **Issues with GPT's Data Retrieval in RAG Applications**: Users discussed difficulties with GPT's effectiveness in retrieving data from files in Retrieval-Augmented Generation (RAG) applications. Suggested improvements included refining data sets and using better embedding models.
- **OpenAI's New Announcements**: Some participants commented on OpenAI's recent announcement of **GPT-4o**, noting its real-time audio, video, and speech synthesis capabilities. Concerns were raised about the long-term implications of life-like AI features.
- **HuggingFace Documentation and AutoTrain**: The HuggingFace documentation was recommended for beginners, and questions were raised about the fine-tuning time for models like **GPT-2 XL** on **Nvidia A10G hardware** using AutoTrain.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://www.andrewng.org/">no title found</a>: no description found</li><li><a href="https://huggingface.co/Gryphe/Tiamat-8b-1.2-Llama-3-DPO">Gryphe/Tiamat-8b-1.2-Llama-3-DPO · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://huggingface.co/blog/train-dgx-cloud">Easily Train Models with H100 GPUs on NVIDIA DGX Cloud</a>: no description found</li><li><a href="https://www.eurekai.tech">EurekAI</a>: no description found</li><li><a href="https://tenor.com/view/excuse-me-hands-up-woah-funny-face-gif-14275996">Excuse Me Hands Up GIF - Excuse Me Hands Up Woah - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=QEaBAZQCtwE&ab_channel=AssemblyAI">Getting Started With Hugging Face in 15 Minutes | Transformers, Pipeline, Tokenizer, Models</a>: Learn how to get started with Hugging Face and the Transformers Library in 15 minutes! Learn all about Pipelines, Models, Tokenizers, PyTorch &amp; TensorFlow in...</li><li><a href="https://youtu.be/DQacCB9tDaw?t=4239">Introducing GPT-4o</a>: OpenAI Spring Update – streamed live on Monday, May 13, 2024. Introducing GPT-4o, updates to ChatGPT, and more.</li><li><a href="https://www.tiktok.com/t/ZTLV3ShEp/">TikTok - Make Your Day</a>: no description found</li><li><a href="https://tenor.com/view/will-smith-chris-rock-jada-pinkett-smith-oscars2022-smack-gif-25234614">Will Smith Chris Rock GIF - Will Smith Chris Rock Jada Pinkett Smith - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1239043199100784752)** (3 messages): 

- **MedEd AI User Experience Overview**: A [YouTube video](https://www.youtube.com/watch?v=UgVPzSSCjr8) provided a quick overview on **genAI user experience**, highlighting the use of containerized applications, multimodal medical advisors, and future plans for RA generation, free tier access, and cost-conscious models. The video covers aspects from introduction to detailed features at various timestamps.

- **DeepLearning.ai on Neural Network Initialization**: A member shared an informative [resource from deeplearning.ai](https://www.deeplearning.ai/ai-notes/initialization/index.html) which explains the importance of effective initialization to prevent issues like exploding/vanishing gradients. The tutorial outlines the common training process for neural networks and emphasizes on choosing the right initialization method.

- **Exploring JAX and TPU for VAR Paper**: Another member discussed porting the VAR paper, which focuses on a new autoregressive modeling paradigm for images, to a Jax-compatible library using Equinox for TPU acceleration ([Arxiv paper](https://arxiv.org/abs/2404.02905)). They shared a [GitHub repository for Equinox](https://github.com/patrick-kidger/equinox) to further elaborate on the tools being used.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.deeplearning.ai/ai-notes/initialization/index.html">AI Notes: Initializing neural networks - deeplearning.ai</a>: In this post, we'll explain how to initialize neural network parameters effectively. Initialization can have a significant impact on convergence in training deep neural networks...</li><li><a href="https://www.youtube.com/watch?v=UgVPzSSCjr8">On User Experience. With Multiple Models. For MedEd &amp; More.</a>: A quick overview on genAI user experience with our current progress. If AI is a buffet, we aim to be your Sushi-Go-Round. 00:00 Introduction03:45 Containeriz...</li><li><a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: We present Visual AutoRegressive modeling (VAR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine &#34;next-scale prediction&#34; or &#34;next-resolutio...</li><li><a href="https://github.com/patrick-kidger/equinox">GitHub - patrick-kidger/equinox: Elegant easy-to-use neural networks + scientific computing in JAX. https://docs.kidger.site/equinox/</a>: Elegant easy-to-use neural networks + scientific computing in JAX. https://docs.kidger.site/equinox/ - patrick-kidger/equinox
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1238994655732043816)** (10 messages🔥): 

- **Phi-3 excels on smartphones**: A member highlighted that **Phi-3** runs well on low-power devices like smartphones. [Read more about it](https://arxiv.org/abs/2404.14219) in this paper by multiple authors.
- **Deep Learning Primer Book**: A "nice book" for understanding deep learning was shared. Check it out [here](https://udlbook.github.io/udlbook/).
- **Neural Network Weights Initialization**: An interesting resource from deeplearning.ai about initializing neural network weights and the issues of exploding/vanishing gradients was shared. [Link](https://www.deeplearning.ai/ai-notes/initialization/index.html) for more details.
- **Visualization of GPT**: A member found a cool visualization of GPT and shared it. [View it here](https://bbycroft.net/llm).
- **3D Diffusion Policy for Robots**: Introducing the **3D Diffusion Policy (DP3)**, a novel approach to visual imitation learning for robots that uses 3D visual representations from sparse point clouds. The method shows a 24.2% improvement over baselines with minimal demonstrations; more insights [here](https://3d-diffusion-policy.github.io/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bbycroft.net/llm">LLM Visualization</a>: no description found</li><li><a href="https://www.deeplearning.ai/ai-notes/initialization/index.html">AI Notes: Initializing neural networks - deeplearning.ai</a>: In this post, we'll explain how to initialize neural network parameters effectively. Initialization can have a significant impact on convergence in training deep neural networks...</li><li><a href="https://udlbook.github.io/udlbook/">Understanding Deep Learning</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.14219">Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone</a>: We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens, whose overall performance, as measured by both academic benchmarks and internal testing, rivals that of ...</li><li><a href="https://3d-diffusion-policy.github.io/">3D Diffusion Policy</a>: This paper introduces 3D Diffusion Policy (DP3), a visual imitation learning algorithm that masters divserse visuomotor tasks.</li><li><a href="https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1238995268410671177)** (7 messages): 

- **AI storyteller in 4 languages faces inactivity**: A member showcased an AI-powered storyteller supporting English, Malay, Chinese, and Tamil [here](https://huggingface.co/spaces/ikmalsaid/alkisah-ai), but noted that this space is currently inactive due to lack of use.

- **Holy Quran verses tool waiting for users**: They also built an AI tool to create beautiful posters based on Holy Quran verses, available [here](https://huggingface.co/spaces/ikmalsaid/kalam-ai), but this space is similarly inactive due to inactivity.

- **OCR toolkit integrates multiple frameworks**: An OCR framework was developed that integrates with DocTr, PaddleOCR, and Google Cloud Vision, making it easy to use and visualize, with the [code and docs available on GitHub](https://github.com/ajkdrag/ocrtoolkit). The toolkit allows for experimentations with different OCR frameworks seamlessly.

- **Fine-tuned Llama variants for token classification shared**: Models fine-tuned for token classification using Llama variants have been shared on the HuggingFace Hub. Details and models, such as `unsloth/llama-2-7b-bnb-4bit` trained on `conll2003`, are available in a [collection](https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188) and an upcoming blog post will be shared on Weights & Biases.

- **New AI Discord chatbot tutorial video posted**: A link to a [YouTube video](https://youtu.be/B1F94RKksR8?si=WPSmpyjiByCHaTAQ) on creating an AI Discord chatbot with web search capabilities was posted, including a git repository with detailed instructions.

- **Classifying noisy vs. clean text is now simpler**: An OCR quality classifier was launched using the PleIAs dataset for text classification, easily distinguishing between noisy and clean text. The small encoders used can serve as new filters for document quality, with details available in a [collection](https://huggingface.co/collections/pszemraj/ocr-quality-classifiers-663ef6076b5a9965101dd3e3).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/B1F94RKksR8?si=WPSmpyjiByCHaTAQ">How To Create Your Own AI Discord Chat Bot With Web Search</a>: Git Repo:https://github.com/ssimpson91/newsChanYou will need the following packages;NodeJS v. 18Python 3.10 or aboveRun these commands in your terminal to in...</li><li><a href="https://huggingface.co/collections/pszemraj/ocr-quality-classifiers-663ef6076b5a9965101dd3e3">OCR Quality Classifiers - a pszemraj Collection</a>: no description found</li><li><a href="https://huggingface.co/spaces/ikmalsaid/kalam-ai">Kalam AI - a Hugging Face Space by ikmalsaid</a>: no description found</li><li><a href="https://huggingface.co/collections/SauravMaheshkar/llamafortokenclassification-6640cfb77f6555eecb54d188">LlamaForTokenClassification - a SauravMaheshkar Collection</a>: no description found</li><li><a href="https://huggingface.co/spaces/ikmalsaid/alkisah-ai">Alkisah AI - a Hugging Face Space by ikmalsaid</a>: no description found</li><li><a href="https://github.com/ajkdrag/ocrtoolkit">GitHub - ajkdrag/ocrtoolkit: Experiment and integrate with different OCR frameworks seamlessly</a>: Experiment and integrate with different OCR frameworks seamlessly - ajkdrag/ocrtoolkit
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1239543496457584752)** (2 messages): 

- **YOCO Decoder-Decoder Architecture reduces GPU memory demands**: A member shared an [arXiv paper](https://arxiv.org/abs/2405.05254) introducing YOCO, a new architecture for large language models. The design, featuring a cross-decoder stacked upon a self-decoder, aims to reduce GPU memory usage while retaining global attention capability and improving prefill speeds.



**Link mentioned**: <a href="https://arxiv.org/abs/2405.05254">You Only Cache Once: Decoder-Decoder Architectures for Language Models</a>: We introduce a decoder-decoder architecture, YOCO, for large language models, which only caches key-value pairs once. It consists of two components, i.e., a cross-decoder stacked upon a self-decoder. ...

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1238780335379185756)** (6 messages): 

- **Class-Condition Diffusion with UNet Discussion**: A user shared their experience with class condition diffusion using UNet, referring to a [HuggingFace diffusion course](https://huggingface.co/learn/diffusion-course/unit2/3) and inquired if there's similar material for latent diffusion models.

- **Stable Diffusion Using Diffusers**: Another user provided a link to a [HuggingFace blog post on Stable Diffusion](https://huggingface.co/blog/stable_diffusion), which discusses how to use the Diffusers library with this text-to-image latent diffusion model and provides additional educational resources.

- **YOLOv1 Implementation Troubles**: A user expressed difficulty implementing YOLOv1 from scratch on a custom dataset and sought assistance from experienced individuals. They later clarified that their goal was to create an educational mini-YOLO with a ResNet backbone.

- **YOLOv1 vs. YOLOv5 or YOLOv8**: Another user questioned the necessity of using YOLOv1 instead of newer versions like YOLOv5 or YOLOv8. The original poster explained the choice was for educational and teaching purposes, aiming to implement a simpler version of YOLO with their custom dataset.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/">Hugging Face - Learn</a>: no description found</li><li><a href="https://huggingface.co/blog/stable_diffusion">Stable Diffusion with 🧨 Diffusers</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1238769279378391061)** (7 messages): 

- **Challenges with meeting transcript chunking**: A user is seeking insights on how to efficiently chunk meeting transcripts for extracting actionable insights using LLMs. They mention trying to separate by speaker changes, but find the similarity scores between interactions to be low (0.45).
  
- **Consequent messages and similarity scores**: Another member commented that consequent messages may not necessarily have high similarity scores even if the topic remains constant. They suggested finding the most relevant chunk and writing a function to fetch neighboring chunks to address the user's needs.

- **Retrieval and generation evaluation advice**: It was suggested to separate retrieval and generation components, evaluate them independently, and benchmark retriever results with different configurations like chunk size and overlap. The "mean reciprocal rank" metric was recommended for evaluation.

- **Custom Hugging Face tokenizer training issues**: A user shared their process of creating and training a custom Hugging Face tokenizer and issues faced when integrating it with a transformer, as instructed in a [2021 YouTube video](https://www.youtube.com/watch?v=MR8tZm5ViWU). They reported errors, with ChatGPT indicating the tokenizer might be in the wrong format.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=MR8tZm5ViWU)">Building a new tokenizer</a>: Learn how to use the 🤗 Tokenizers library to build your own tokenizer, train it, then how to use it in the 🤗 Transformers library.This video is part of the...

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1239087326135717899)** (14 messages🔥): 

- **Dive into Diffusion Models with these Resources**: A member asked for recommendations on understanding diffusion models, samplers, and related topics. They were pointed to the [DDPM & DDIM papers](https://arxiv.org/abs/2006.11239) and [Fast.ai's course](https://course.fast.ai/Lessons/part2.html), which includes collaboration with Stability.ai and Hugging Face.

- **Struggling with SadTalker on macOS?**: A user requested urgent help with installing SadTalker on macOS. Someone recommended [searching for the error message](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985#issuecomment-1813885266) to find more precise answers.

- **Get Hands-On with Inpainting**: There was an inquiry about using inpainting with personal images. An in-depth explanation and guide for using the [🤗 Diffusers library](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint) for inpainting was shared.

- **Creating Custom Image Datasets**: Someone asked how to use their custom image datasets instead of internet data. They were directed to a guide on [creating a dataset with the 🤗 Datasets library](https://huggingface.co/docs/diffusers/main/en/training/create_dataset).

- **Local Inference Engine for Command-R+ Advice**: There was a passing query about making a local inference engine for Command-R+. A member suggested seeking advice from an NLP-focused group for more relevant input.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/training/create_dataset">Create a dataset for training</a>: no description found</li><li><a href="https://course.fast.ai/Lessons/part2.html">Practical Deep Learning for Coders - Part 2 overview</a>: Learn Deep Learning with fastai and PyTorch, 2022</li><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint">Inpainting</a>: no description found</li><li><a href="https://www.oreilly.com/library/view/hands-on-generative-ai/9781098149239/">Hands-On Generative AI with Transformers and Diffusion Models</a>: Learn how to use generative media techniques with AI to create novel images or music in this practical, hands-on guide. Data scientists and software engineers will understand how state-of-the-art gene...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985#issuecomment-1813885266">[Bug]: ModuleNotFoundError: No module named &#39;torchvision.transforms.functional_tensor&#39; torchvision 0.17 promblem · Issue #13985 · AUTOMATIC1111/stable-diffusion-webui</a>: Is there an existing issue for this? I have searched the existing issues and checked the recent builds/commits What happened? ModuleNotFoundError: No module named &#39;torchvision.transforms.functiona...
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1238747906857766994)** (183 messages🔥🔥): 

- **Users Struggle with GPT Agent Learning**: Users expressed concerns about GPT agents not learning from additional information, with others clarifying that uploaded files are saved as "knowledge" files but do not continually modify the agent's base knowledge.
- **RTX 4070 for Summarization Tasks in Linux**: A member inquired about specs for summarizing PDFs, mentioning a system with Intel i5, RTX 4070, and 64GB RAM on GNU/Linux, only to be informed that chat with docs capabilities aren't yet supported by LM Studio.
- **Performance Issue with Multi-GPU Setup**: A user faced issues running models on a setup with multiple GPUs and reported extremely slow performance. The problem was identified as likely related to hardware setup with PCIe 3.0, and resolving it by switching motherboard equipped with PCIe 4.0.
- **Access Issues to LM Studio Features Amid Network Concerns**: Various users reported encountering difficulties accessing models from LM Studio, often due to network errors or blocked locations. Solutions such as using a VPN with IPv4 were suggested.
- **Exploring Alternatives for Local Model Deployment**: Discussion included advice on using systems with sufficient VRAM for local model deployment, emphasizing GPUs with 8GB+ memory for better performance and usability over CPU-only setups.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.asrockrack.com/general/productdetail.asp?Model=ROMED8-2T/BCM">no title found</a>: no description found</li><li><a href="https://www.asrockrack.com/general/productdetail.asp?Model=EPYCD8#Specifications">no title found</a>: no description found</li><li><a href="https://downforeveryoneorjustme.com/chat.lmsys.org?proto=https">Chat.lmsys.org down? Current problems and status. - DownFor</a>: Chat.lmsys.org won't load? Or, having problems with Chat.lmsys.org? Check the status here and report any issues!</li><li><a href="https://tenor.com/view/boo-boo-this-man-gif-4868055">Boo Boo This Man GIF - Boo Boo This Man - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1238759943537168394)** (92 messages🔥🔥): 

- **Yi-1.5 Models Gain Traction**: LM Studio community members are excited about the new [Yi-1.5 models](https://huggingface.co/01-ai/Yi-1.5-9B-Chat) and multiple versions like 9B, 6B, and upcoming 34B quantized models. Members appreciated Yi-1.5's performance, noting it performs well in diverse fine-tuning tasks but mentioned issues like confusion about its identity.

- **Challenges with Smaller Hardware**: Users discussed the difficulties of running advanced models on constrained hardware like an RTX 3050 6GB and the limitations it poses for tasks like coding or long-context processing. Recommendation steered towards using lightweight models or employing tools like stable diffusion via accessible platforms such as [itch.io](https://itch.io).

- **Audio Cleanup Solutions**: For those needing to clean up audio, options like **Voicecraft** and **RVC** were discussed to enhance instructional videos with poor audio quality, similar to Adobe's Podcast Enhance.

- **Fine-Tuning Questions and Insights**: Queries about fine-tuning datasets sparked discussions on the composition of test data, with suggestions leaning toward a mix of normal question-answer pairs. Insights were shared about models often being quantized by different people and occasionally having finetune designations in their names.

- **Command R+ Model Commended**: There was high praise for the **Command R+ model**, with users recommending it for its longer context length, enhanced smartness, and lack of censorship, which makes it preferable over others like Llama 3 70B.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/YorkieOH10/Yi-1.5-6B-Chat-Q8_0-GGUF">YorkieOH10/Yi-1.5-6B-Chat-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/YorkieOH10/Yi-1.5-9B-Chat-Q8_0-GGUF">YorkieOH10/Yi-1.5-9B-Chat-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NikolayKozloff/Meta-Llama-3-8B-Instruct-bf16-correct-pre-tokenizer-and-EOS-token-Q8_0-Q6_k-Q4_K_M-GGUF">NikolayKozloff/Meta-Llama-3-8B-Instruct-bf16-correct-pre-tokenizer-and-EOS-token-Q8_0-Q6_k-Q4_K_M-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/01-ai/Yi-1.5-9B-Chat">01-ai/Yi-1.5-9B-Chat · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/failspy/kappa-3-phi-abliterated">failspy/kappa-3-phi-abliterated · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF">dranger003/c4ai-command-r-plus-iMat.GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1238893918850908222)** (4 messages): 

- **Member shares positive feedback**: A user expressed their gratitude for the helpful feedback they received from another member, indicating a positive interaction within the community.

- **Alternatives to Innosetup**: One member suggested using **Innosetup** or **Nullsoft Installer** as good open-source alternatives for software installation, based on their past experiences.

- **Challenges with Starcoder model on Debian**: A member described encountering repetitive responses and off-topic answers while using the **starcoder2-15b-instruct** model on Debian 12. They noted the behavior was similar across different platforms and setups, including the app chatbox and VSC server.

- **Instruct model's limitations**: Another member clarified that **instruct models** are not typically designed for multi-step conversations. They emphasized that these models are intended to execute single commands and respond directly to those.
  

---


**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1238854849395822603)** (7 messages): 

- **Playground mode requires GPU**: A user inquired about running the playground mode on RAM + CPU given their limited 4GB VRAM. Another member confirmed that the **playground mode is GPU only**.
- **Warning against suspicious links**: A user warned others not to click a shortlink, pointing out that it does not direct to Steam. The warning is emphasized with a Johnny Depp gif and repeated insistence to "go away."
- **Using word files for LLM training**: A user asked if they could train a **Large Language Model (LLM)** with word files containing syllabus content for question-and-answer purposes. There was no follow-up response to this inquiry.

**Link mentioned**: <a href="https://tenor.com/view/shoo-go-away-johnny-depp-captain-jack-sparrow-gif-4877675">Shoo Go Away GIF - Shoo Go Away Johnny Depp - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1238773262884798565)** (106 messages🔥🔥): 

- **Running Large Models on Limited Hardware Fails**: Members discussed their experiences running **Llama 3 70B Q8** on hardware with **128GB RAM**, noting that it's often too slow or fails to load. One example noted 2 tok/s speed on a 4090 with 128GB for a 70B Q4 model, highlighting limitations. 
  
- **CPU Inference for Large LLMs is Painfully Slow**: Running large models like **Llama 3 70B** solely on CPUs results in slow speeds, often only achieving single-digit token per second performance. A notable example mentioned getting 0.6 tok/s after disabling E-cores on an **i5 12600K**.

- **Challenges of GPU Memory Limitations**: Users with limited VRAM, such as 2GB, found it practically useless for running advanced models, even when trying to offload layers. **"2GB video won't be useful at all - you'd want 4, but preferably 6gb, minimum to start to be useful."**

- **Mixed Results with Different GPUs**: Despite having superior specs, a **Tesla P100** performed worse than a **GTX 1060** for some members when using LM Studio. Disabling "Hardware-accelerated GPU scheduling" showed a modest 5% boost in performance.

- **Documentation and Backend Queries**: Users were curious about how the **llama.cpp** backend in **LM Studio** handles computations and whether it utilizes FP32 or FP16 and Tensor cores. Clarifications included that it generally uses quantized models which reduce precision significantly.
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1238759906635681822)** (12 messages🔥): 

- **CodeQwen1.5 shines for coding on RTX 3050**: A member recommended **CodeQwen1.5** as a highly efficient coding model, noting it outperforms **DeepSeek Coder**. The model's 4b quantization, about 4.18 GB, fits well on an RTX 3050 GPU.
- **Hugging Face's coding leaderboard is a resource**: Another member shared a link to the **Hugging Face's coding leaderboard** on their site, where users can check details about coding models of 7b or lower. [bigcode-models-leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard).
- **LLama.cpp update and bug fixes**: Responding to queries about new features, a user clarified that the latest build primarily consists of bug fixes alongside an **update to llama.cpp**. Users did not report any new hidden features.
- **Bots slip through automod**: A user commented on a suspicious link, likely for farming ad or referral income, and noted it **evaded auto-moderation**. This highlights ongoing vigilance against potential spam or malicious links in the chat.

**Link mentioned**: <a href="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard">Big Code Models Leaderboard - a Hugging Face Space by bigcode</a>: no description found

  

---


**LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1238795889171103775)** (4 messages): 

- **Seek MemGPT Help**: A member requested assistance from someone experienced with **MemGPT**, prompting responses of varying confidence and apologies. 
- **Setup Issues**: One responder mentioned successfully setting up **MemGPT** with **Kobold** and managing memory adjustments but admitted to struggling with implementation on **LM Studio**.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1238798948538515466)** (2 messages): 

- **Scoops up RX 7900 XT**: A member shared the excitement of purchasing an **RX 7900 XT** for 700 euros, mentioning it's more than enough power for their needs.
- **Bigger models recommended**: Another member suggested trying **Command-R+** or **Yi-1.5's quantized variants**, hinting that the new GPU could handle larger models.
  

---


**LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1238893873972117755)** (4 messages): 

- **Confusion on connecting LM Studio to OpenInterpreter**: A user asked for a guide on how to connect **LM Studio** to **OpenInterpreter**. The conversation reveals they are experiencing consistent errors when attempting to run the server, both when connected and not connected.
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1239407328483213364)** (1 messages): 

- **New Yi Models Available!**: The LM Studio community has released new Yi models on their Hugging Face page. There are various sizes available, including a rare 34B model, ideal for users with 24GB cards.
  
- **GGUF Quantization by Bartowski**: The models feature **GGUF quantization** provided by the community member Bartowski, based on the `llama.cpp` release [b2854](https://github.com/ggerganov/llama.cpp/releases/tag/b2854). This ensures maximum quality and enhanced performance.

- **Model Details and Performance**: All Yi-1.5 models are upgraded versions continuously pre-trained with a high-quality corpus of 500B tokens and fine-tuned on 3M diverse samples. They are designed to perform well on a wide range of tasks.

- **Links to Models**: Check out the new models here: 
  - [Yi 1.5 34B Chat GGUF](https://huggingface.co/lmstudio-community/Yi-1.5-34B-Chat-GGUF)
  - [Yi 1.5 9B Chat GGUF](https://huggingface.co/lmstudio-community/Yi-1.5-9B-Chat-GGUF)
  - [Yi 1.5 6B Chat GGUF](https://huggingface.co/lmstudio-community/Yi-1.5-6B-Chat-GGUF)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/Yi-1.5-34B-Chat-GGUF">lmstudio-community/Yi-1.5-34B-Chat-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/Yi-1.5-9B-Chat-GGUF">lmstudio-community/Yi-1.5-9B-Chat-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/Yi-1.5-6B-Chat-GGUF">lmstudio-community/Yi-1.5-6B-Chat-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1238998189626101862)** (19 messages🔥): 

- **Discussion on Vulkan-Backend for llama.cpp**: A member inquired about running a Vulkan-backend for **llama.cpp** with **LM Studio** or using a backend API. Another member responded that there isn't a solution for this yet.

- **LM Studio CLI Tool Announcement**: A member shared the release of **LM Studio 0.2.22** and its companion CLI tool, **`lms`**, which allows model management and API server control. The tool is available on [GitHub](https://github.com/lmstudio-ai/lms) and ships with LM Studio's working directory.

- **Clarification on Backend API Request**: A discussion clarified that the original query was about connecting **LM Studio** to a **llama.cpp** HTTP server rather than the suggested **CLI tool**.

- **Headless Installation Issues**: Members discussed the difficulties of installing **LM Studio** on a headless Linux cloud server due to **AppImage** issues with FUSE. Alternative suggestions included trying **Ollama** or compiling **llama.cpp** from the base.

**Link mentioned**: <a href="https://lmstudio.ai/blog/lms">Introducing `lms` - LM Studio&#x27;s companion cli tool | LM Studio</a>: Today, alongside LM Studio 0.2.22, we&#x27;re releasing the first version of lms — LM Studio&#x27;s companion cli tool.

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1238747934376464386)** (2 messages): 

- **JetMoE 8B Free Outage**: The [JetMoE 8B Free model](https://openrouter.ai/models/jetmoe/jetmoe-8b-chat:free) is currently down due to upstream overload. All requests will return an empty response with an error (502) until further notice.

- **Multimodal Models Now Available**: Two new multimodal models are now up and running on OpenRouter. Check out [OpenAI: GPT-4o](https://openrouter.ai/models/openai/gpt-4o) and [LLaVA v1.6 34B](https://openrouter.ai/models/liuhaotian/llava-yi-34b).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/jetmoe/jetmoe-8b-chat:free>)">JetMoE 8B by jetmoe | OpenRouter</a>: Coming from a broad set of teams, ranging from academic to industry veterans, Jet MoE is a combined effort from MIT, Princeton, IBM, Lepton, and MyShell.  This model is fully open source and trained o...</li><li><a href="https://openrouter.ai/models/openai/gpt-4o)">OpenAI: GPT-4o by openai | OpenRouter</a>: GPT-4o (&quot;o&quot; for &quot;omni&quot;) is OpenAI&#x27;s latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/open...</li><li><a href="https://openrouter.ai/models/liuhaotian/llava-yi-34b)">LLaVA v1.6 34B by liuhaotian | OpenRouter</a>: LLaVA Yi 34B is an open-source model trained by fine-tuning LLM on multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture. Base LLM: [Nou...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1239279202767867924)** (2 messages): 

- **Track OpenRouter model changes easily**: A member introduced the **OpenRouter API Watcher**, an open-source tool designed to monitor and store changes in the OpenRouter model list using a SQLite database. It offers a web interface and an RSS feed for updates, querying the API hourly to maintain **minimal overhead**. [Demo](https://orw.karleo.net/)
- **Become a beta tester for Rubik's AI Pro**: Another member is inviting users to beta test an advanced research assistant and search engine, offering **2 months of free premium** access to models like GPT-4 Turbo and Claude 3 Opus. Interested users were asked to DM feedback and use a promo code `RUBIX` for the free trial. [Rubik's AI](https://rubiks.ai/)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://orw.karleo.net/">OpenRouter API Watcher</a>: OpenRouter API Watcher monitors changes in OpenRouter models and stores those changes in a SQLite database. It queries the model list via the API every hour.</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1238747703710584863)** (251 messages🔥🔥): 

```html
- **Jetmoe lacks online access**: When asked if **Jetmoe** has online access, the response was clear, *“No, it doesn’t.”* Jetmoe is considered good for academic research despite this limitation.
  
- **OpenRouter tackles anti-fraud measures aggressively**: Discussion on anti-fraud updates revealed that **OpenRouter** has implemented measures to combat fraud due to losses from credit card skimming. Users can opt for crypto transactions to avoid providing personal information.

- **Embedding models support in consideration**: When asked about embedding models support, it was mentioned that **OpenRouter** is working on improving the backend and has embedding models in the queue, but there is no immediate roadmap yet.

- **Inconsistent prompt formatting issues**: Users discussed how models like **Claude** handle instructions differently than models focused on RP (role-playing) or generic tasks. The need for trial and error in crafting effective prompts for different models was highlighted.

- **OpenRouter adds GPT-4o**: Excitement surrounded the addition of **GPT-4o** to OpenRouter, with users noting its competitive pricing and high performance in benchmarks. OpenRouter will support text and image inputs for GPT-4o, although video and audio are not available.
```

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.openwebui.com/tutorial/openai">OpenAI API Endpoints | Open WebUI</a>: In this tutorial, we will demonstrate how to configure multiple OpenAI (or compatible) API endpoints using environment variables. This setup allows you to easily switch between different API providers...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cq927y/yi">Reddit - Dive into anything</a>: no description found</li><li><a href="https://stripe.com/">Stripe | Financial Infrastructure for the Internet</a>: Stripe powers online and in-person payment processing and financial solutions for businesses of all sizes. Accept payments, send payouts, and automate financial processes with a suite of APIs and no-c...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cq927y/yi15_202405/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.01.ai/">零一万物-AI2.0大模型技术和应用的全球公司（01.AI）</a>: no description found</li><li><a href="https://claudeai.uk/can-claude-read-pdf/">Can Claude Read PDF? [2023] - Claude Ai</a>: Can Claude Read PDF? PDF (Portable Document Format) files are a common document type that many of us encounter in our daily lives.
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1238892203837886504)** (65 messages🔥🔥): 

- **Implicit variants with the pipe operator in Mojo discussed**: One member queried about Mojo adopting implicit variants with the pipe operator, to which another shared a link to [PEP 604](https://peps.python.org/pep-0604/) as a comparison. The discussion touched upon potential syntax and the handling of pattern matching.

- **Pattern matching debate gets heated**: There was a vibrant discussion about the value and aesthetics of pattern matching in Mojo compared to using `if-else` statements. Advocates highlighted how pattern matching ensures exhaustive cases and compiler optimizations, while critics found it visually unappealing.

- **Mojo versus Rust: compiler experiences shared**: Members compared experiences with Mojo and Rust compilers, noting that Rust is perceived as more complex and harder to navigate, whereas Mojo’s simpler, more straightforward approach was appreciated. The debate included opinions on Rust's optimization capabilities and the projection of Mojo's future feature robustness.

- **Contributing to the Mojo compiler inquiries**: A user inquired about contributing to the Mojo compiler, prompting a response that currently, the Mojo compiler is not open source. Clarifications were made that the Mojo compiler is written in C++, not Mojo.

- **Discussion on Mojo and MLIR relationship**: There was a brief discussion on the possibilities of bootstrapping Mojo using MLIR, and whether rebuilding MLIR in Mojo would be feasible in the future. The conversation acknowledged MLIR’s C++ origins and raised the question of future development.

**Link mentioned**: <a href="https://peps.python.org/pep-0604/">PEP 604 – Allow writing union types as X | Y | peps.python.org</a>: no description found

  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1790046377613144201>
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1239603493745197056)** (1 messages): 

- **Modular's new video announcement**: The **ModularBot** shared that a new video has been posted on their YouTube channel. You can check out the latest content by clicking [here](https://www.youtube.com/watch?v=9ag0fPMmYPQ).
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1238886742702952458)** (85 messages🔥🔥): 

- **Storage and Running of Benchmarks**: Members discussed optimal ways to store and run benchmarks in repositories, with one user suggesting that including benchmarks in a `tests` folder might be practical. Another user inquired about ways to benchmark memory usage.

- **Syntax Discussion in Mojo**: There was a debate about dereferencing syntax, with some suggesting C++ style `*` would be ergonomic, but others like Chris Lattner argued for `p[]` as it composes nicely and is pythonic.

- **Iterator Implementation in Mojo**: Joker discussed implementing "yield" like behavior in Mojo by replicating the torchdata API due to Mojo's current lack of real `yield` capabilities. They detailed their approach and ran into issues with type constraints and parametric traits.

- **Tree Sitter Grammar Fork**: Lukas Hermann mentioned they wrote a Tree Sitter grammar fork and tested it successfully in text editors like Helix and Zed, planning to clean up decorators and add tests.

- **Deep Dive into Mojo Ownership**: A link to a [YouTube talk](https://www.youtube.com/watch?v=9ag0fPMmYPQ) by Chris Lattner was shared, explaining ownership in Mojo. Members discussed their struggles with ownership concepts coming from a Python background and the importance of examples showing why these ideas matter.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://doc.rust-lang.org/nomicon/subtyping.html">Subtyping and Variance - The Rustonomicon</a>: no description found</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/dtype/DType#is_floating_point">DType | Modular Docs</a>: Represents DType and provides methods for working with it.</li><li><a href="https://www.youtube.com/watch?v=9ag0fPMmYPQ">Mojo🔥: a deep dive on ownership with Chris Lattner</a>: Learn everything you need to know about ownership in Mojo, a deep dive with Modular CEO Chris LattnerIf you have any questions make sure to join our friendly...</li><li><a href="https://github.com/modularml/mojo/issues/2467#issuecomment-2106263163">[Feature Request] Unify SSO between `InlinedString` and `String` type · Issue #2467 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? We currently have https://docs.modular.com/mojo/stdlib...</li><li><a href="https://florimond.dev/en/posts/2018/08/python-mutable-defaults-are-the-source-of-all-evil">Python Mutable Defaults Are The Source of All Evil - Florimond Manca</a>: How to prevent a common Python mistake that can lead to horrible bugs and waste everyone&#39;s time.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1238795315365285908)** (1 messages): 

- **Introducing MoString on GitHub**: A member announced the creation of a [GitHub repo for MoString](https://github.com/dorjeduck/mostring) focusing on variations over **StringBuilder ideas in Mojo**. They added an `optimize_memory` method to reduce allocated memory and invited community contributions to explore suitable implementations for the Mojo standard.

**Link mentioned**: <a href="https://github.com/dorjeduck/mostring">GitHub - dorjeduck/mostring: variations over StringBuilder ideas in Mojo</a>: variations over StringBuilder ideas in Mojo. Contribute to dorjeduck/mostring development by creating an account on GitHub.

  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1238944586349412433)** (64 messages🔥🔥): 

- **Custom Hasher struct proposal sparks debate**: A member expressed concerns about forcing devs to create custom Hasher structs, favoring simpler methods like Python's `__hash__`. The proposal author provided additional [examples](https://github.com/modularml/mojo/pull/2619) showcasing the flexibility and simplicity his implementation aims to offer.

- **CI tests failure on Ubuntu sparks action**: Members discussed issues with CI tests hanging on Ubuntu, with suggestions to add timeouts to the workflows. A [pull request](https://github.com/modularml/mojo/pull/2644) was created to implement these timeouts, and it was noted that GitHub Actions might experience buggy "pending" statuses during this time.

- **Significant performance findings on `List` extend method**: A member shared benchmarking results showing the extend method of Mojo's `List` could be greatly improved via a memory pre-allocation strategy. This led to discussions about the merits of mirroring Rust's vector allocation strategies for similar tasks.

- **Nested arrays causing segmentation faults**: A member reported segmentation faults when dealing with nested arrays and questioned whether the issue was related to variadic pack or lifetime management. It led to insights on reference handling within array iterators.

- **Excitement over nightly releases**: The community celebrated the shift to automatic nightly releases for Mojo, dubbed "nightly nightlies," and discussed implications such as reduced delays between committed changes and their availability.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.github.com/en/actions/learn-github-actions/usage-limits-billing-and-administration#usage-limits),">Usage limits, billing, and administration - GitHub Docs</a>: no description found</li><li><a href="https://github.com/modularml/mojo/issues">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/pull/2619">[stdlib] Introduce Hasher type with all necessary changes by mzaks · Pull Request #2619 · modularml/mojo</a>: This is a draft because although the code compiles 8 tests are failing. It might be due to compiler bug. The error messages are cryptic. I don&#39;t have the &quot;Mojo&quot; ;) to fix them. Failed Te...</li><li><a href="https://github.com/modularml/mojo/pull/2644">[CI] Add timeouts to workflows by JoeLoser · Pull Request #2644 · modularml/mojo</a>: On Ubuntu tests, we&#39;re seeing some non-deterministic timeouts due to a code bug (either in compiler or library) from a recent nightly release.  Instead of relying on the default GitHub timeout of ...</li><li><a href="https://github.com/dorjeduck/minbpe.mojo">GitHub - dorjeduck/minbpe.mojo: port of Andrjey Karpathy&#39;s minbpe to Mojo</a>: port of Andrjey Karpathy&#39;s minbpe to Mojo. Contribute to dorjeduck/minbpe.mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/pull/2620#issuecomment-2106054892">[stdlib] Delegate string comparisons to `StringRef` by siitron · Pull Request #2620 · modularml/mojo</a>: This is a follow-up to #2409. String comparisons for StringRef are implemented. StringRef make use of memory.memcmp for all of its 6 comparisons now, hopefully this change is ok. String&#39;s and Stri...</li><li><a href="https://github.com/dorjeduck/minbpe.mojo/blob/main/mojobpe/utils/mostring/molist.mojo">minbpe.mojo/mojobpe/utils/mostring/molist.mojo at main · dorjeduck/minbpe.mojo</a>: port of Andrjey Karpathy&#39;s minbpe to Mojo. Contribute to dorjeduck/minbpe.mojo development by creating an account on GitHub.</li><li><a href="https://github.com/dorjeduck/mostring">GitHub - dorjeduck/mostring: variations over StringBuilder ideas in Mojo</a>: variations over StringBuilder ideas in Mojo. Contribute to dorjeduck/mostring development by creating an account on GitHub.</li><li><a href="https://github.com/mzaks/mojo/tree/feature/minimal-example-of-test-crash-for-new-hasher">GitHub - mzaks/mojo at feature/minimal-example-of-test-crash-for-new-hasher</a>: The Mojo Programming Language. Contribute to mzaks/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1238983332910596187)** (5 messages): 

- **GPU memory management confusion clarified**: A user with an 8GB GPU noticed that CUDA uses shared memory when running out of dedicated GPU memory. They observed significant slowdowns during this process and asked for resources to understand how this works.
- **Direct contact with Discord CEO for tech support**: One member humorously reported chatting directly with the Discord CEO to resolve stage stability issues, leading to quick action from the team. Their success prompted light-hearted reactions from other members.
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1238762803138007051)** (43 messages🔥): 

- **New Lecture on Triton Praised**: A member shared a [YouTube video](https://www.youtube.com/watch?v=DdTsX6DQk24) titled "Lecture 14: Practitioner's Guide to Triton" and the accompanying [GitHub description](https://github.com/cuda-mode/lectures/tree/main/lecture%2014). It's a resource for learning more about Triton kernels.

- **Contributors Share Resources for Conv2D Kernels**: Discussions included links to existing Conv2D kernel implementations in Triton found in [PyTorch's kernel](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/conv.py) and the [attorch repository](https://github.com/BobMcDear/attorch). There's encouragement to contribute to the main Triton repo or other related repositories.

- **Cataloging Triton Kernels Highlighted**: The [Triton Index](https://github.com/cuda-mode/triton-index) and [Awesome Triton Kernels](https://github.com/zinccat/Awesome-Triton-Kernels) repositories were mentioned as valuable resources for cataloging and discovering Triton kernels. Kernl, a tool designed for running PyTorch transformer models faster on GPU, was also highlighted: [Kernl GitHub](https://github.com/ELS-RD/kernl).

- **Excitement Over ThunderKittens**: A new DSL called ThunderKittens was shared via [Twitter](https://x.com/bfspector/status/1789749117104894179?s=46&t=ROCrCC19RlrPdFqCtEaiGA) and discussed enthusiastically. It promises to make writing AI kernels in CUDA simpler and more efficient, potentially outperforming Triton's Flash Attention.

- **Flash Attention Performance Comparisons**: There was a detailed discussion about the performance differences between Triton’s Flash Attention and a new implementation in ThunderKittens. Some members noted that proper tuning and configurations might narrow the performance gap, suggesting ongoing improvements and benchmarks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/BobMcDear/attorch">GitHub - BobMcDear/attorch: A subset of PyTorch&#39;s neural network modules, written in Python using OpenAI&#39;s Triton.</a>: A subset of PyTorch&#39;s neural network modules, written in Python using OpenAI&#39;s Triton. - BobMcDear/attorch</li><li><a href="https://www.youtube.com/watch?v=DdTsX6DQk24">Lecture 14: Practitioners Guide to Triton</a>: https://github.com/cuda-mode/lectures/tree/main/lecture%2014</li><li><a href="https://x.com/bfspector/status/1789749117104894179?s=46&t=ROCrCC19RlrPdFqCtEaiGA">Tweet from Benjamin F Spector (@bfspector)</a>: (1/7) Happy mother’s day! We think what the mothers of America really want is a Flash Attention implementation that’s just 100 lines of code and 30% faster, and we’re happy to provide.  We&#39;re exci...</li><li><a href="https://github.com/zinccat/Awesome-Triton-Kernels">GitHub - zinccat/Awesome-Triton-Kernels: Collection of kernels written in Triton language</a>: Collection of kernels written in Triton language. Contribute to zinccat/Awesome-Triton-Kernels development by creating an account on GitHub.</li><li><a href="https://github.com/openai/triton/commit/702215e26149a657ee49c6fdc4d258c51fe0cdac">[TUTORIALS] tune flash attention block sizes (#3892) · triton-lang/triton@702215e</a>: no description found</li><li><a href="https://github.com/cuda-mode/triton-index">GitHub - cuda-mode/triton-index: Cataloging released Triton kernels.</a>: Cataloging released Triton kernels. Contribute to cuda-mode/triton-index development by creating an account on GitHub.</li><li><a href="https://github.com/ELS-RD/kernl">GitHub - ELS-RD/kernl: Kernl lets you run PyTorch transformer models several times faster on GPU with a single line of code, and is designed to be easily hackable.</a>: Kernl lets you run PyTorch transformer models several times faster on GPU with a single line of code, and is designed to be easily hackable. - ELS-RD/kernl
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1239314338708197458)** (9 messages🔥): 

- **ThunderKittens speeds up kernels**: The GitHub repository for [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) focuses on tile primitives for speeding up kernels. It's a project by HazyResearch aimed at making CUDA operations more efficient.
  
- **NanoGPT-TK for optimized GPT training**: [NanoGPT-TK](https://github.com/HazyResearch/nanoGPT-TK) is a repository touted as the simplest and fastest for training and fine-tuning medium-sized GPTs. The repository also humorously emphasizes that it includes "kittens," playing on the project name.

- **FlashAttention explained humorously**: A blog post describes the efforts of HazyResearch to simplify AI kernel-building ideas through projects like ThunderKittens. They reference a [NeurIPS keynote](https://neurips.cc/virtual/2023/invited-talk/73990) and use humor to bridge the gap between complex technical models and accessible explanations.

- **Swizzling reduces memory bank conflicts**: A discussion clarified that swizzling helps avoid memory bank conflicts, enhancing memory access efficiency in CUDA programming. A link to the [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) was provided for further reading.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk">ThunderKittens: A Simple Embedded DSL for AI kernels</a>: good abstractions are good.</li><li><a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>: how make gpu fast?</li><li><a href="https://github.com/HazyResearch/ThunderKittens">GitHub - HazyResearch/ThunderKittens: Tile primitives for speedy kernels</a>: Tile primitives for speedy kernels. Contribute to HazyResearch/ThunderKittens development by creating an account on GitHub.</li><li><a href="https://github.com/HazyResearch/nanoGPT-TK">GitHub - HazyResearch/nanoGPT-TK: The simplest, fastest repository for training/finetuning medium-sized GPTs. Now, with kittens!</a>: The simplest, fastest repository for training/finetuning medium-sized GPTs. Now, with kittens! - HazyResearch/nanoGPT-TK
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1238927701281210369)** (1 messages): 

- **Fusing Kernels Talk Announcement**: An upcoming talk on **fusing kernels** is scheduled to start in 7 minutes, featuring <@488490090008674304>. The talk will happen on [Zoom](https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success), and attendees are instructed to post chat and questions in the designated channel <#1238926773216084051>.

**Link mentioned**: <a href="https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09#success">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

random_string_of_character: https://arxiv.org/abs/2405.05219
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1238752380972040323)** (14 messages🔥): 

- **Join the U Illinois PMPP lecture series via Zoom**: "We will start the 4th lecture of U Illinois PMPP series in 10 minutes.. here is a [zoom link](https://us06web.zoom.us/j/83020353425?pwd=w3oQfYJPJVz2arzeZmxJbBsAMGFrBD.1)." These lectures usually happen weekly on Saturdays, and the details are shared in a dedicated Discord server.
- **PMPP lecture comparisons vivid**: "I like how he compares warps to platoons in the army," making complex concepts more relatable.
- **Course details and accessibility**: The course on applied parallel programming is available on YouTube, with the [course playlist](https://youtube.com/playlist?list=PLRRuQYjFhpmvu5ODQoY2l7D0ADgWEcYAX) being frequently shared. Despite being from 2018, it remains a valuable resource.
- **Integration and announcement etiquette**: Laith0x0 and Wilson post announcements here but prefer not to overuse mentions. Marksaroufim suggested using a dedicated Discord channel for more persistent information sharing.
- **Compatibility queries and build dependencies**: Geri8904 is seeking compatibility information for torch-tensorrt with different CUDA and Torch versions and experiences issues with package installations. __safelix__ also encountered missing build dependencies and sought recommendations for a comprehensive requirements.txt file.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLRRuQYjFhpmvu5ODQoY2l7D0ADgWEcYAX&feature=shared">UIUC ECE408/CS483 Spring 2018 Hwu</a>: This is a junior/senior-level undergraduate course entitled &quot;Applied Parallel Programming&quot; at the University of Illinois at Urbana-Champaign. It is often als...</li><li><a href="https://youtube.com/playlist?list=PLRRuQYjFhpmvu5ODQoY2l7D0ADgWEcYAX&feature=sha">UIUC ECE408/CS483 Spring 2018 Hwu</a>: This is a junior/senior-level undergraduate course entitled &quot;Applied Parallel Programming&quot; at the University of Illinois at Urbana-Champaign. It is often als...</li><li><a href="https://us06web.zoom.us/j/83020353425?pwd=w3oQfYJPJVz2arzeZmxJbBsAMGFrBD.1">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1239615740680605778)** (1 messages): 

- **CUDA Expert Talks Date Announced**: The **PMPP Author Izzat El Hajj** will discuss **scan operations** on May 24. On May 25, **Jake and Georgii** will explain how to build advanced scan using **CUDA C++**; the event details are available [here](https://discord.gg/gFDMmM96?event=1239607867666071654).

**Link mentioned**: <a href="https://discord.gg/gFDMmM96?event=1239607867666071654">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1239310080353239223)** (5 messages): 

- **Seeking help on Thermal Face Recognition project**: A member asked for **insights, resources, such as research papers, GitHub repositories, or general suggestions** for their college final project titled *'Thermal Face Recognition'*. They aim to predict if two thermal face images belong to the same person.
- **Clarification sought and given**: One member asked if the project involves matching two thermal face images for the same person, detecting bounding boxes, or facial landmarks. The project was clarified to be related to predicting if two images are of the same person.
  

---


**CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

boxxy_ms: anyone in Toronto?
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1239536441571278909)** (2 messages): 

- **Oscar_yu hunts for official solutions**: Oscar_yu inquired about the availability of official solutions to verify the numerical correctness of his implementation. He later acknowledged finding Joey's solution in Misha's thread, expressing gratitude.
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1238941871032635583)** (67 messages🔥🔥): 

```html
- **ZeRO-1 empowers VRAM battle**: ZeRO-1 integration was discussed, with benchmarks showing a 54% training throughput improvement by optimizing VRAM usage, allowing batch size increase from 4 to 10, maxing out the A100's 40GB VRAM capacity. Catch more details [here](https://github.com/karpathy/llm.c/pull/309).
- **Optimization insights on GPU workloads**: Members discussed the benefit of performing calculations outside of CUDA kernels to optimize integer divisions and memory-bound kernels. Perspectives were shared on using 2D/3D grids and thread coarsening for efficiency, backed by detailed [code discussions](https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L689).
- **ThunderKittens catches interest**: The potential of HazyResearch's [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) for H100 llm.c optimization sparked excitement. Members see it as a lower-level abstraction than CUTLASS for managing tensor core layouts.
- **Efforts to improve CI with GPU support**: Talks revolved around the lack of GPUs in llm.c’s CI and ways to bridge this gap, noting GitHub Actions' recent GPU runner beta. Suggestions included upgrading GitHub plans and references to current pricing [details](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions#per-minute-rates-for-larger-runners).
```

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions#per-minute-rates-for-larger-runners">About billing for GitHub Actions - GitHub Docs</a>: no description found</li><li><a href="https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpLoad.html#cub-warpload)">cub::WarpLoad &mdash; CUB 104.0 documentation</a>: no description found</li><li><a href="https://github.com/NVIDIA/cccl/issues/525).">Issues · NVIDIA/cccl</a>: CUDA C++ Core Libraries. Contribute to NVIDIA/cccl development by creating an account on GitHub.</li><li><a href="https://github.blog/changelog/2023-10-31-run-your-ml-workloads-on-github-actions-with-gpu-runners/">Run your ML workloads on GitHub Actions with GPU runners</a>: Run your ML workloads on GitHub Actions with GPU runners</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L689">llm.c/train_gpt2.cu at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/HazyResearch/ThunderKittens/tree/main">GitHub - HazyResearch/ThunderKittens: Tile primitives for speedy kernels</a>: Tile primitives for speedy kernels. Contribute to HazyResearch/ThunderKittens development by creating an account on GitHub.</li><li><a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>: how make gpu fast?</li><li><a href="https://github.com/karpathy/llm.c/issues/406">2D and 3D tile divisions so that permutation coordinates can be read from threadIdx and blockIdx · Issue #406 · karpathy/llm.c</a>: Supposedly the permutation kernels, even though they are mostly memory bound can reduce the amount of division and do thread coarsening by having a 2d or 3d grid and not have to do any division in ...</li><li><a href="https://github.com/karpathy/llm.c/pull/309/commits/f613ce895b30dc0b2bd1f7e81410c6a2dcdce74d">Zero Redundancy Optimizer - Stage1 by chinthysl · Pull Request #309 · karpathy/llm.c</a>: To train much larger model variations (2B, 7B, etc), we need larger GPU memory allocations for parameters, optimizer states, and gradients. Zero Redundancy Optimizer introduce the methodology to sh...</li><li><a href="https://github.com/karpathy/llm.c/blob/2346cdac931f544d63ce816f7e3f5479a917eef5/.github/workflows/ci.yml#L141">llm.c/.github/workflows/ci.yml at 2346cdac931f544d63ce816f7e3f5479a917eef5 · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/309">Zero Redundancy Optimizer - Stage1 by chinthysl · Pull Request #309 · karpathy/llm.c</a>: To train much larger model variations (2B, 7B, etc), we need larger GPU memory allocations for parameters, optimizer states, and gradients. Zero Redundancy Optimizer introduce the methodology to sh...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1238928157692919809)** (48 messages🔥): 

- **Max-Autotune boosts performance with thorough hyperparam tuning**: The `max-autotune` mode in `torch.compile` leverages Triton-based matrix multiplications and convolutions, trying out more hyperparameters for potentially faster kernels. As a trade-off, it takes longer to compile. [torch.compile docs](https://pytorch.org/docs/stable/generated/torch.compile.html)
- **Dynamo vs. Inductor tutorials**: Members shared that Dynamo tutorials are more comprehensive compared to inductor ones and highlighted the importance of having better materials for handling dynamic shapes. Links to additional resources were provided for those interested in Dynamo's internal workings. [PyTorch Workshops](https://github.com/pytorch/workshops/tree/master/ASPLOS_2024)
- **Fusion benefits and limitations debated**: Discussions highlighted that fusing kernels generally reduces global memory read/writes which benefits memory-bound kernels, but excessive fusion may just add overhead without substantial gains. The general sentiment was to fuse extensively unless proven counterproductive.
- **Interest in Triton internals and performance profiling**: Several members expressed the need for talks on Triton internals and detailed profiling methodologies for distinguishing overhead, HBM-SRAM communication, and actual computation time. An upcoming workshop was promoted for more insights. [Triton Workshop](https://discord.com/events/1189498204333543425/1228827008830668801)
- **Availability of lecture recordings**: Due to time zone differences and chaotic schedules, members inquired about when the lecture recordings would be available. The response indicated that it might be delayed but would be addressed soon.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.compile.html">torch.compile &mdash; PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://fb.zoom.us/j/94565757373?pwd=ZHFhWjU2TFBXdnJzdnl5bDZ0cEFUZz09">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://github.com/pytorch/pytorch/wiki/Tensor-and-Operator-Basics">Tensor and Operator Basics</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/workshops/tree/master/ASPLOS_2024">workshops/ASPLOS_2024 at master · pytorch/workshops</a>: This is a repository for all workshop related materials.  - pytorch/workshops</li><li><a href="https://pytorch.org/docs/main/torch.compiler_dynamo_deepdive.html">Dynamo Deep-Dive &mdash; PyTorch main documentation</a>: no description found
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/1239093813033828372)** (5 messages): 

- **ECE408 Slides Shared for Applied Parallel Programming**: Course materials for the Spring 2019 edition of ECE408, [available here](https://lumetta.web.engr.illinois.edu/408-S19/), include timeline, project plan, and staff office hours. The course emphasizes grade distribution through Blackboard and discussions via Piazza.

- **YouTube Watch Party for CUDA Videos**: This channel hosts a viewing party where participants watch CUDA-related videos on YouTube together, especially focusing on the series [Programming Massively Parallel Processors](https://www.youtube.com/@pmpp-book). The sessions encourage discussions every 10-15 minutes to allow for questions and knowledge sharing.

- **Scheduled Viewing Times**: Viewing sessions are scheduled on Saturdays at 7:30 GMT for EMEA participants and 18:00 GMT for NAM participants. Zoom links will be provided by specific members for the meetings.

- **Plan Post 18 Lectures**: After completing the round of 18 lectures, the group may rewatch CUDA Mode videos or select another high-quality, vetted series on parallel processing. This ensures continuous learning and engagement in parallel programming topics.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@pmpp-book">Programming Massively Parallel Processors</a>: This channel is the official channel for the textbook: &quot;Programming Massively Parallel Processors: A Hands-on Approach&quot;</li><li><a href="https://lumetta.web.engr.illinois.edu/408-S19/">ECE408: Applied Parallel Programming, Spring 2019 ZJUI Section</a>: no description found
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1238856315993063554)** (61 messages🔥🔥): 

- **GPTs Agents cannot learn after initial training**: A member shared a concern about GPTs agents not learning from additional information provided after their initial training. Another member cleared this misunderstanding, explaining that uploaded files are saved as "knowledge" files for the agent to reference when required, but **they do not continually modify the agent's base knowledge**.

- **Researchers critique layer duplication**: *“It’s like they’re just introducing noise by duplicating layers and calling the model smarter,”* a commenter critiqued efforts to expand models like llama 70b to 120b and 160b by duplicating layers. Another user added *"they are finetuning over this somewhat also”.*

- **Recent arXiv paper on zero-shot generalization**: A recent [arXiv paper](https://arxiv.org/abs/2404.04125) discussed performance issues in zero-shot generalization for multimodal models, generating extensive debate. Critics noted the work's findings were unsurprising and emphasized that the paper does not address compositional generalization.

- **Falcon-2 11B release gains attention**: **Falcon-2 11B** was released, trained on 5T refined web data, with an 8k context and MQA attention for improved inference. It sparked interest due to its **permissive license and new size**.

- **Discussion on copyright impact on AI development**: Members discussed how **AI copyright issues** could influence small players and startups. The conversation highlighted that companies like Microsoft offering indemnity may dominate funding and innovation competition, potentially chilling effects on smaller AI ventures.

**Link mentioned**: <a href="https://arxiv.org/abs/2404.04125">No &#34;Zero-Shot&#34; Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance</a>: Web-crawled pretraining datasets underlie the impressive &#34;zero-shot&#34; evaluation performance of multimodal models, such as CLIP for classification/retrieval and Stable-Diffusion for image gener...

  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1238750605707841569)** (79 messages🔥🔥): 

- **New Attention Approximation Method Debuts**: A member shared an [arXiv link](https://arxiv.org/abs/2405.05219) about an efficient approximation method for attention computation using convolution matrices. Another member expressed skepticism about its practical applications compared to existing methods like flash attention.

- **Depth Upscaling in LLMs Gains Interest**: Discussions on systematic approaches to "depth upscaling," mentioned in papers such as [SOLAR](https://arxiv.org/abs/2312.15166) and Yi Granite Code models [Yi 1.5](https://arxiv.org/abs/2403.04652), included insights on appropriate datasets and prevailing techniques for improving language models.

- **Efficient Data Distillation via Farzi**: A new method called Farzi summarized an event sequence dataset into smaller synthetic datasets while maintaining performance, as highlighted in [an arXiv link](https://arxiv.org/abs/2310.09983). Authors claimed up to 120% downstream performance on synthetic data, but acknowledged scaling challenges with larger models like T5 and datasets like C4.

- **Token Glitch Detection Method Released**: A study was discussed that focuses on identifying untrained and under-trained tokens in LLMs, found at [this arXiv link](https://arxiv.org/abs/2405.05417). This method aims to improve tokenizer efficiency and overall model safety.

- **Emerging Work on Memory Mosaics**: A fresh approach called Memory Mosaics was shared via [an arXiv link](https://arxiv.org/abs/2405.06394), proposing a network of associative memories for prediction tasks, showcasing competitive performance with transformers on medium-scale language modeling tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.06394">Memory Mosaics</a>: Memory Mosaics are networks of associative memories working in concert to achieve a prediction task of interest. Like transformers, memory mosaics possess compositional capabilities and in-context lea...</li><li><a href="https://arxiv.org/abs/2405.06147v1">State-Free Inference of State-Space Models: The Transfer Function Approach</a>: We approach designing a state-space model for deep learning applications through its dual representation, the transfer function, and uncover a highly efficient sequence parallel inference algorithm th...</li><li><a href="https://arxiv.org/abs/2405.05417">Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models</a>: The disconnect between tokenizer creation and model training in language models has been known to allow for certain inputs, such as the infamous SolidGoldMagikarp token, to induce unwanted behaviour. ...</li><li><a href="https://arxiv.org/abs/2312.15166">SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling</a>: We introduce SOLAR 10.7B, a large language model (LLM) with 10.7 billion parameters, demonstrating superior performance in various natural language processing (NLP) tasks. Inspired by recent efforts t...</li><li><a href="https://arxiv.org/abs/2405.05219">Conv-Basis: A New Paradigm for Efficient Attention Inference and Gradient Computation in Transformers</a>: Large Language Models (LLMs) have profoundly changed the world. Their self-attention mechanism is the key to the success of transformers in LLMs. However, the quadratic computational cost $O(n^2)$ to ...</li><li><a href="https://huggingface.co/spaces/devingulliver/subquadratic-llm-leaderboard">Subquadratic LLM Leaderboard - a Hugging Face Space by devingulliver</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.04435">Fast Exact Retrieval for Nearest-neighbor Lookup (FERN)</a>: Exact nearest neighbor search is a computationally intensive process, and even its simpler sibling -- vector retrieval -- can be computationally complex. This is exacerbated when retrieving vectors wh...</li><li><a href="https://arxiv.org/abs/2310.09983">Farzi Data: Autoregressive Data Distillation</a>: We study data distillation for auto-regressive machine learning tasks, where the input and output have a strict left-to-right causal structure. More specifically, we propose Farzi, which summarizes an...</li><li><a href="https://arxiv.org/abs/2403.04652">Yi: Open Foundation Models by 01.AI</a>: We introduce the Yi model family, a series of language and multimodal models that demonstrate strong multi-dimensional capabilities. The Yi model family is based on 6B and 34B pretrained language mode...</li><li><a href="https://arxiv.org/abs/2405.04324">Granite Code Models: A Family of Open Foundation Models for Code Intelligence</a>: Large Language Models (LLMs) trained on code are revolutionizing the software development process. Increasingly, code LLMs are being integrated into software development environments to improve the pr...</li><li><a href="https://openreview.net/forum?id=H9DYMIpz9c&noteId=aN4DeBSr82">Farzi Data: Autoregressive Data Distillation</a>: We study data distillation for auto-regressive machine learning tasks, where the input and output have a strict left-to-right causal structure. More specifically, we propose Farzi, which summarizes...</li><li><a href="https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk">ThunderKittens: A Simple Embedded DSL for AI kernels</a>: good abstractions are good.</li><li><a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>: how make gpu fast?</li><li><a href="https://arxiv.org/abs/2309.03852">FLM-101B: An Open LLM and How to Train It with $100K Budget</a>: Large language models (LLMs) have achieved remarkable success in NLP and multimodal tasks, among others. Despite these successes, two main challenges remain in developing LLMs: (i) high computational ...</li><li><a href="https://arxiv.org/abs/2305.02869">Masked Structural Growth for 2x Faster Language Model Pre-training</a>: Accelerating large language model pre-training is a critical issue in present research. In this paper, we focus on speeding up pre-training by progressively growing from a small Transformer structure ...</li><li><a href="https://arxiv.org/abs/2104.05520">Updatable Learned Index with Precise Positions</a>: Index plays an essential role in modern database engines to accelerate the query processing. The new paradigm of &#34;learned index&#34; has significantly changed the way of designing index structures...</li><li><a href="https://www.jeanfeydy.com/">Jean Feydy's home page</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1239488393713287199)** (7 messages): 

- **Bullish on synthetic data, but with caution**: One member expressed being bullish about **synthetic data**, while another shared skepticism, noting that it "had literally the same hype cycle about 5-7 years ago" and concerns that **the lessons learned** may not carry over due to the influx of newer field professionals.
- **MLPs versus Transformers and CNNs**: A member referenced two [papers on arXiv](https://arxiv.org/abs/2108.13002), discussing the comparison of **CNNs, Transformers, and MLPs** for vision tasks. They highlighted an empirical study indicating that while all structures can achieve competitive performance at a moderate scale, they show distinctive behaviors as network size scales, advocating for a hybrid approach.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2108.13002#microsoft">A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP</a>: Convolutional neural networks (CNN) are the dominant deep neural network (DNN) architecture for computer vision. Recently, Transformer and multi-layer perceptron (MLP)-based models, such as Vision Tra...</li><li><a href="https://arxiv.org/abs/2306.13575">Scaling MLPs: A Tale of Inductive Bias</a>: In this work we revisit the most fundamental building block in deep learning, the multi-layer perceptron (MLP), and study the limits of its performance on vision tasks. Empirical insights into MLPs ar...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1239481105514758144)** (3 messages): 

- **NeurIPS last-minute submission call**: A member asked if anyone was interested in doing a last-minute submission for NeurIPS. They mentioned doing something similar to the **Othello paper**.
- **Impact of model compression on features/circuits**: Another member raised the question of what types of **features/circuits are lost** when compressing models. They pondered whether these features are totally useless or if they are just **overspecialized for small subsets** of the training distribution, suggesting such features could inform on the dataset's diversity.
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

oleksandr07173: Hello
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1238819707457372161)** (120 messages🔥🔥): 

```html
- **First Look at VideoFX Generations**: A user shared a [link to VideoFX footage](https://fxtwitter.com/bedros_p/status/1789256595123179701?s=46), stating there are more examples but it's still a WIP. The shared footage demonstrates early capabilities of VideoFX generations.
  
- **GPT-4o Steals the Spotlight**: [Liam Fedus announced](https://x.com/liamfedus/status/1790064963966370209?s=46) GPT-4o as the new state-of-the-art model. Users discussed its superior performance in coding compared to older versions and speculated about its potential in MATH and other benchmarks.

- **OpenAI's New Tokenizer**: A member shared a [GitHub commit](https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8) for the new OpenAI tokenizer. The update appears to improve processing speeds by utilizing a larger vocabulary.

- **OpenAI's Latest Demo Reaction**: Although a user found the demo impressive, they didn't see anything fundamentally new beyond UI improvements. Other discussions included speculation around GPT-4o's capabilities and its availability, with questions about OpenAI’s data strategies.

- **GPT-4o Dominates on LMSys Arena**: LMSys org [shared exciting news](https://x.com/lmsysorg/status/1790097588399779991?s=46) that GPT-4o has surpassed all models on the LMSys Arena with a significant Elo increase. The model's enhancements in reasoning and coding were particularly highlighted by users.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/google/status/1790055114272612771?s=46>)">Tweet from Google (@Google)</a>: One more day until #GoogleIO! We’re feeling 🤩. See you tomorrow for the latest news about AI, Search and more.</li><li><a href="https://fxtwitter.com/bedros_p/status/1789256595123179701?s=46">Tweet from Bedros Pamboukian (@bedros_p)</a>: VideoFX footage from the list of examples There are 2 more, but it looks like its a WIP  First look at VideoFX generations:</li><li><a href="https://x.com/lmsysorg/status/1790097595064529255?s=46">Tweet from lmsys.org (@lmsysorg)</a>: Significantly higher win-rate against all other models. e.g., ~80% win-rate vs GPT-4 (June) in non-tie battles.</li><li><a href="https://x.com/liamfedus/status/1790064963966370209?s=46">Tweet from William Fedus (@LiamFedus)</a>: GPT-4o is our new state-of-the-art frontier model. We’ve been testing a version on the LMSys arena as im-also-a-good-gpt2-chatbot 🙂. Here’s how it’s been doing.</li><li><a href="https://x.com/lmsysorg/status/1790097588399779991?s=46">Tweet from lmsys.org (@lmsysorg)</a>: Breaking news — gpt2-chatbots result is now out!  gpt2-chatbots have just surged to the top, surpassing all the models by a significant gap (~50 Elo). It has become the strongest model ever in the Are...</li><li><a href="https://www.youtube.com/watch?v=MirzFk_DSiI">Two GPT-4os interacting and singing</a>: Say hello to GPT-4o, our new flagship model which can reason across audio, vision, and text in real time.Learn more here: https://www.openai.com/index/hello-...</li><li><a href="https://x.com/drjimfan/status/1790122998218817896?s=46">Tweet from Jim Fan (@DrJimFan)</a>: I stand corrected: GPT-4o does NOT natively process video stream. The blog says it only takes image, text, and audio. That&#39;s sad, but the principle I said still holds: the right way to make a vide...</li><li><a href="https://github.com/openai/tiktoken/commit/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8">Sync codebase · openai/tiktoken@9d01e56</a>: no description found</li><li><a href="https://x.com/kaiokendev1/status/1790068145933185038?s=46">Tweet from Kaio Ken (@kaiokendev1)</a>: yeah but can it moan?
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1239691351629762630)** (1 messages): 

- **TRLOO Paper Explains REINFORCE as PPO's Special Case**: A member shared a [Hugging Face PR](https://github.com/huggingface/trl/pull/1540) and noted its explanation on how REINFORCE is a special case of PPO in the implementation. They also linked to the [referenced paper](https://arxiv.org/pdf/2205.09123).

**Link mentioned**: <a href="https://github.com/huggingface/trl/pull/1540">PPO / Reinforce Trainers by vwxyzjn · Pull Request #1540 · huggingface/trl</a>: This RP supports the REINFORCE RLOO trainers in https://arxiv.org/pdf/2402.14740.pdf. Note that REINFORCE&#39;s loss is a special case of PPO, as shown below  it matches the REINFORCE loss presented i...

  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1238889709644808294)** (5 messages): 

- **ChatbotArena Appreciation**: One member remarked that people on **ChatbotArena are very skillful** and another agreed, highlighting that it's instrumental in *determining the future.*
- **Open-Sourcing GPT-3.5**: There was a brief speculative discussion on the potential of GPT-3.5 being open-sourced. One member humorously noted that this would happen only when "hell freezes over."
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1239071719927320647)** (11 messages🔥): 

- **Video achieves 6k views in a day**: *"damn 6k views in a day"* - a member exclaimed about the quick success. **Other videos** in comparison were noted to be at *"20k"* views.
- **Natolambert aims to boost views**: *"I need to pump those numbers"* - intention to increase video views was expressed. This was motivated by another **Huggingface video** reaching *"150k"* views.
- **Discussion on posting video to X**: Suggestions made to post the video to X, with a native upload mentioned. There was concern about **Stanford's licensing**, but skipped, as natolambert believes they won't pursue, saying they can *"request permission"*, but will post anyways.

  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1238837869716574269)** (109 messages🔥🔥): 

- **Artists vs. AI Services Debate Heats Up**: Members debated whether AI services like Midjourney and others that generate art harm artists' income. Claims included AI's commercial service impact on art sales, potential legal implications, and distinctions between fair use and derivative works, with links to [The Legal Artist](https://www.thelegalartist.com/blog/you-cant-copyright-style) and various articles providing context.
- **StabilityAI and Midjourney's Legal Troubles**: Members discussed the potential downfall of StabilityAI and shared disdain for artist David Holz, with hopes for consequences stemming from public disclosures. Insights included the likelihood of juries affecting outcomes without following the law and broader implications for Midjourney’s practices.
- **DeepSeek LLM and Efficient AI Models**: A new fine-tuned Pixart Sigma model was shared on Civitai, with praise for its non-NSFW use. In parallel, a blog post highlighted advancements in AI compute efficiency, featuring innovations like [FlashAttention-2](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2) and others.
- **Launch of Falcon 2 Series**: A description of the launch and specifications of Falcon 2 models, claiming superior performance over Meta's Llama 3 was shared. A link to the [Technology Innovation Institute](https://www.tii.ae/news/falcon-2-uaes-technology-innovation-institute-releases-new-ai-model-series-outperforming-metas) provided further details.
- **OpenAI's GPT-4o Unveiled**: OpenAI's release of GPT-4o, featuring real-time communication and video processing, spurred interest. Members noted its improved performance, free access, and voice mode updates as detailed in [OpenAI's announcement](https://openai.com/index/hello-gpt-4o/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>: how make gpu fast?</li><li><a href="https://civitai.com/models/435669?modelVersionId=502675">Bunline - v0.4 | Stable Diffusion Checkpoint | Civitai</a>: PixArt Sigma XL 2 1024 MS full finetune on custom captions for roughly 35k images w/ max(w,h) &amp;gt; 1024px INSTRUCTIONS: Place the .safetensors wher...</li><li><a href="https://tenor.com/bR79n.gif">Silicon Valley Tip To Tip GIF - Silicon Valley Tip To Tip Brainstorm - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2">deepseek-ai/DeepSeek-V2 · Hugging Face</a>: no description found</li><li><a href="https://www.tii.ae/news/falcon-2-uaes-technology-innovation-institute-releases-new-ai-model-series-outperforming-metas">Falcon 2: UAE’s Technology Innovation Institute Releases New AI Model Series, Outperforming Meta’s New Llama 3</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1238861766231064607)** (5 messages): 

- **Convert Voice Data Sets to Tokens**: A member emphasized the need to convert numerous voice data sets to tokens. They also highlighted the importance of *"high quality annotations about emotions and speaker attribute"*, sharing a link to a [Twitter post](https://fxtwitter.com/laion_ai/status/1788532651072049314?t=1NgVkLaxmC9gzgdSmGpM3Q&s=19) and a [YouTube video](https://youtu.be/NwZufAJxmMA) on training transformers with audio.
- **Mathematical Notation and Sampling Functions**: There was a technical discussion about the use of notation in formal mathematics to indicate sequences of elements, specifically z indexed by i converging to z_t, and the potential role of T as a sampling function. Further elaboration was deemed difficult without more context.

**Link mentioned**: <a href="https://fxtwitter.com/laion_ai/status/1788532651072049314?t=1NgVkLaxmC9gzgdSmGpM3Q&s=19">Tweet from LAION (@laion_ai)</a>: Wanna train transformers with audio as if it was text?   - Here is how. :) https://youtu.be/NwZufAJxmMA  https://discord.gg/6jWrFngyPe

  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1238794899726405662)** (105 messages🔥🔥): 

- **Extracting and Converting Dates to ISO Format in LangChain**: One member shared a prompt containing dates and asked how to extract and convert them to ISO format using LangChain. Kapa.ai provided detailed Python and JavaScript code examples utilizing the `DatetimeOutputParser` for this process.

- **Setting Up Local Open-Source LLMs with LangChain**: A user asked how to use tools with local open-source LLMs like Ollama in LangChain. Kapa.ai explained the process including defining models and creating prompts in both Python and JavaScript.

- **Handling Ambiguous Model Outputs and Reducing La*tency in Function Calls**: Members discussed methods to refine model outputs and optimize response times when creating entities in databases via LangChain. Suggestions focused on model selection for specific tasks and improving UX by speeding up function call responses.

- **Persistent Storage Alternatives for docstore in LangChain**: A user inquired about alternatives to using `InMemoryStore` for persistent storage in the multimodal RAG setup with LangChain and Gemini. Other members suggested checking the LangChain documentation for more options.

- **Frequent Errors and Model Context Use with HuggingFace and LangChain**: Common issues like validation errors with `facebook/bart` on HuggingFace and problems related to API usage and model support were discussed. Solutions included using correctly supported models and adjusting prompts or API usage.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/use_cases/extraction#approaches>)">Extracting structured output | 🦜️🔗 LangChain</a>: Overview</li><li><a href="https://python.langchain.com/docs/modules/agents/agent_types/structured_chat#run-agent>).">Structured chat | 🦜️🔗 LangChain</a>: The structured chat agent is capable of using multi-input tools.</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/stores/">Stores | 🦜️🔗 LangChain</a>: In many different applications, having some sort of key-value storage is helpful.</li><li><a href="https://python.langchain.com/docs/use_cases/tool_use/quickstart#toolfunction-calling>)">Quickstart | 🦜️🔗 LangChain</a>: In this guide, we will go over the basic ways to create Chains and Agents that call Tools. Tools can be just about anything — APIs, functions, databases, etc. Tools allow us to extend the capabilities...</li><li><a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/multimodal_rag_langchain.ipynb">generative-ai/gemini/use-cases/retrieval-augmented-generation/multimodal_rag_langchain.ipynb at main · GoogleCloudPlatform/generative-ai</a>: Sample code and notebooks for Generative AI on Google Cloud, with Gemini on Vertex AI - GoogleCloudPlatform/generative-ai</li><li><a href="https://python.langchain.com/docs/get_started/quickstart#llm-chain>)">Quickstart | 🦜️🔗 LangChain</a>: In this quickstart we&#x27;ll show you how to:</li><li><a href="https://github.com/langchain-ai/langchain/blob/master/libs/partners/chroma/pyproject.toml">langchain/libs/partners/chroma/pyproject.toml at master · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/Go">go - Overview</a>: go has 52 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/11011>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/3577>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/19805>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/3994>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://gist.github.com/mattcollins/62fcb8d15a001d5b4e5c9fb86aad4f8e">Example of extracting multiple values from a streamed OpenAI chat response</a>: Example of extracting multiple values from a streamed OpenAI chat response - extract_multiple_values_from_stream.py</li><li><a href="https://github.com/langchain-ai/langchain/issues/16935>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/17029>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/17008>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/17031>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/90>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/docs/integrations/llms/manifest#compare-hf-models>).">Manifest | 🦜️🔗 LangChain</a>: This notebook goes over how to use Manifest and LangChain.</li><li><a href="https://github.com/langchain-ai/langchain/issues/5513>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/9908>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/4438>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1238769533431320608)** (4 messages): 

- **AI Video Recommendations Wow the Crowd**: Check out [this YouTube video](https://youtu.be/vyOtowbGwG0?feature=shared) shared in the community, likely of interest due to its relevance to the LangChain crowd.
  
- **Twitter Thread on IndexNetwork Gains Attention**: A member shared an intriguing [Twitter thread](https://twitter.com/indexnetwork_/status/1788311740595245515) by IndexNetwork, drawing attention to its relevance for AI enthusiasts.

- **Open Source Code Interpreter Alternative Launched**: A community member introduced [NLAVIDA](https://github.com/obaidur-rahaman/nlavida), an open-source alternative to advanced data analytics tools available in ChatGPT Plus. They plan to expand its functionality to support open source LLMs like Llama 3.
  
- **RAG Pipeline Tutorial Excites Developers**: One member is creating an in-depth [tutorial](https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog) on building a custom RAG pipeline using LangChain, Next.js, and Pinecone. The guide includes everything from data processing code to a client-side chat interface demo.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog">Build a RAG pipeline for your blog with LangChain, OpenAI and Pinecone</a>: You can chat with my writing and ask me questions I&#x27;ve already answered even when I&#x27;m not around</li><li><a href="https://github.com/obaidur-rahaman/nlavida">GitHub - obaidur-rahaman/nlavida: Natural Language-Assisted Visualization &amp; Interactive Data Analysis (NLAVIDA): Securely handle and analyze confidential data in enterprise environments, enhancing insights and decision-making with advanced visualization.</a>: Natural Language-Assisted Visualization &amp;amp; Interactive Data Analysis (NLAVIDA): Securely handle and analyze confidential data in enterprise environments, enhancing insights and decision-making ...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1239226368735186975)** (3 messages): 

- **YouTube Tutorial Share**: A member shared a [YouTube tutorial](https://www.youtube.com/watch?v=KQ-xGVFHDkw) useful for certain LangChain functionalities.

- **Chat with Blog using LangChain and Pinecone**: Zack Proser created a [blog post](https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog) explaining how he integrated a chat feature on his site to query blog content. He provided everything needed to replicate it, including ingest code, API route code for embeddings and vector search, and a client-side chat interface.

- **Seeking Tutorial for Session Handling with Streaming**: A member requested recommendations for a tutorial on managing history, handling sessions, and enabling streaming in LangChain. They mentioned struggling to get streaming functionality working based on the current documentation.

**Link mentioned**: <a href="https://zackproser.com/blog/langchain-pinecone-chat-with-my-blog">Build a RAG pipeline for your blog with LangChain, OpenAI and Pinecone</a>: You can chat with my writing and ask me questions I&#x27;ve already answered even when I&#x27;m not around

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1238888807072403516)** (8 messages🔥): 

- **Generate PowerPoints with Llama 3**: An article by @naivebaesian on using @llama_index to build a Llama3 RAG pipeline that can generate PowerPoint slide decks is highlighted. It utilizes the Python-pptx library and can be found [here](https://t.co/iM0c5Cl2uK).

- **Build a Financial Agent with Reflection**: Hanane Dupouy demonstrates how to create an agent capable of reflecting on stock prices. Techniques include implementing CRITIC for tool use, with more details available [here](https://t.co/mmJ8cjmw73).

- **Use RAG for Content Moderation**: @cloudraftio details setting up a RAG pipeline for content moderation of user-generated images. The process involves captioning images to text and matching them against indexed rules, more information [here](https://t.co/z6jBpMvQss).

- **Evaluate RAG Systems with Multiple Libraries**: @kingzzm provides a thorough article on evaluating RAG systems using libraries like TruLens, Ragas, UpTrain, and DeepEval. A comprehensive set of evaluation metrics is discussed, article available [here](https://t.co/gLbXJoPsqu).

- **GPT-4o Multimodal Abilities Demo**: A simple demonstration of GPT-4o's multimodal capabilities featuring @seldo's dog shows its prowess. View the demo and a humorous take on Amazon's $4,000 second-hand sneakers [here](https://t.co/yPMeyookRq).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://t.co/5k1tvKklGA">Google Colab</a>: no description found</li><li><a href="https://t.co/CMQ1aOXeWb">llama-index-llms-openai</a>: llama-index llms openai integration</li><li><a href="https://t.co/1DLv8fikOi">llama-index-multi-modal-llms-openai</a>: llama-index multi-modal-llms openai integration</li><li><a href="https://t.co/yPMeyookRq">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1238778218237005824)** (87 messages🔥🔥): 

- **Condense Plus Context Bug Identified and Fixed**: Discussions revealed that the condense_plus_context method ignored the postprocessor, which was a bug. A user confirmed this has already been fixed in the latest version.

- **Hybrid Search Error Due to Configuration Issue**: A user faced a ValueError in hybrid search due to a misconfiguration. Another member clarified the need to enable hybrid in the QdrantVectorStore constructor, not in the retriever.

- **Ease of Use and Flexibility of LlamaIndex Praised**: Multiple users highlighted the ease of use, flexibility, and documentation of LlamaIndex over other AI builder tools. Users appreciated LlamaIndex's focused approach on Retrieval-Augmented Generation (RAG), making development smoother.

- **Querying with Metadata Clarified**: Clarifications were given on how metadata in TextNodes is used during querying. It was explained that metadata helps in filtering and additional uses but needs to be appropriately configured during node creation.

- **Python Code Examples for CSV Parsing**: Detailed guidance was provided on how to efficiently read, parse, and index CSV files, emphasizing the use of the CSVReader class. A code snippet and links to further resources were shared for deeper understanding.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb">langchain/cookbook/Multi_modal_RAG.ipynb at master · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/readers/file#llama_index.readers.file.CSVReader>)">File - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/connector#concept>)">Data Connectors (LlamaHub) - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1239453695620419584)** (3 messages): 

- **Fine-Tune GPT-3.5 with Knowledge Distillation**: Members discussed a [blog post](https://huggingface.co/blog/Andyrasika/knowledgedistillation-gpt) on knowledge distillation for fine-tuning a GPT-3.5 judge. One user highlighted the importance of such articles, noting that there aren't enough resources showing users how to effectively fine-tune models.

**Link mentioned**: <a href="https://huggingface.co/blog/Andyrasika/knowledgedistillation-gpt">Knowledge Distillation for Fine-Tuning a GPT-3.5 Judge: Enhancing Accuracy and Performance </a>: no description found

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1238758760621543484)** (30 messages🔥): 

- **Llama 3 Instruct Tune Investigation**: [An analysis](https://gist.github.com/CoffeeVampir3/48544cdaf888a76ca6f8e25863200fad) shared by a member breaks down the weight differences between instruct and base Llama 3, noting that "most changes are scattered seemingly at random," with clustering in the K and V layers. This could suggest that freezing the K/V layers might allow for "more of a stylistic tune" without severely impacting the instruct ability.
  
- **OpenOrca Rerun Cost and Feasibility**: Another member is seeking sponsors to fund a rerun of the OpenOrca deduplication on GPT-4o. Estimated costs are around $650 for processing both input and output tokens, with potential batch job options to lower the expenditure.

- **AI Compute Efficiency Focus**: A shared [blog post](https://hazyresearch.stanford.edu/blog/2024-05-12-tk) delves into recent efforts to reduce AI's compute usage. It references multiple methods like Based, Monarch Mixer, H3, and FlashAttention-2 aimed at running AI more efficiently.

- **Publishing Delays Frustration**: Frustration over journal publication delays is voiced, with the concern that by the time papers are published, they could be "already outdated." A respondent noted that getting two papers published is typically sufficient for earning a PhD, even though the process is challenging.

- **Bluesky vs. Substack for Blogs**: Engagement around whether to use Substack or Bluesky for blogging mentions that while Bluesky is currently limited to threads and posts, it has a "rather nerdy audience".
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>: how make gpu fast?</li><li><a href="https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup?">Open-Orca/SlimOrca-Dedup · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1238793400229298238)** (11 messages🔥): 

- **Merged Pull Request Initiates Discussion**: Members briefly noted that a recent merge occurred successfully. One commented, "Nice it was merged."

- **Pyet PR for Llama3 Chat Template Raises Errors**: A member inquired if anyone had tried the new **pyet PR** for the **LLAMA3 chat template**. They encountered an *AttributeError: 'LLAMA3. Did you mean: 'LLAMA2'?*.

- **Updating Dependencies Resolves Issues**: One member mentioned that updating **fastchat** resolved their issue with the new PR. Another confirmed, "pr + fastchat worked ok for me."

- **Outdated Dependency Concerns**: Concerns were raised about outdated dependencies like **peft 0.10.0**, **accelerate 0.28.0**, **deepspeed 0.13.2**, **flash-attn 2.5.5**, **xformers 0.0.22**, and **transformers @ 43d17c**. They highlighted that these configurations default to **torch 2.0.0** while 2.3.0 is already available.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1238884024642961408)** (11 messages🔥): 

- **FSDP and FFT compatibility questioned**: A member asked if FSDP works with FFT or if it is still problematic. Another replied suggesting to try DeepSpeed instead.
- **DeepSpeed confirmed operational**: Another member confirmed that DeepSpeed works for the proposed scenario.
- **LLAMA3 AttributeError during Docker use**: A member encountered an *AttributeError: LLAMA3* while using Docker, and was advised to update fastchat which did not resolve the issue, but **git cloning did**.
- **Updating pip dependencies for LLAMA3 error**: Another user suggested updating pip dependencies to fix the *LLAMA3* error, confirming with their own experience.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1238787047309836370)** (10 messages🔥): 

- **Changing system_prompt in axolotl CLI**: A member inquired about changing the `system_prompt` when using `axolotl.cli.inference`. There was no direct solution provided in the thread itself.
  
- **Error converting merged model to GGUF**: A user encountered an error while converting a merged model to GGUF, specifically a `FileNotFoundError` due to the absence of a matching tokenizer. Details included paths to the model files and the specific error message.

- **RuntimeError with Gemma-7B after training**: A user's attempt to load a trained Gemma-7B model resulted in a `RuntimeError` due to a size mismatch in `model.embed_tokens.weight`. They provided details of the file structure before and after training, but the issue remained unresolved.

- **How to merge qlora to base without precision issues**: Another user asked how to merge qlora to a base model without facing precision issues (fp16/32). No solution was discussed in the visible portion of the thread.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1238876815230242937)** (9 messages🔥): 

- **Question on pruning support in Axolotl**: A user asked if **Axolotl supports pruning**, to which Phorm initiated an automatic search over the **OpenAccess-AI-Collective/axolotl** without providing a definitive answer yet. The search result indicated that further information could be found on Phorm's [official page](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined).

- **Continuous pretraining and LoRa methods inquiry**: Another query was made regarding tips for **continuous pretraining** and the various **LoRa methods**. Again, Phorm started a search over the relevant repositories but could not provide an immediate answer, suggesting users check back later on their [platform](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined).

- **Merging QLoRA into base model**: A user inquired about **how to merge QLoRA into the base model**, directing their question to a specific group within the Discord. This question was not accompanied by an immediate response.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1238756318999740507)** (41 messages🔥): 

- **Claude API fails with "goofy error"**: A user expressed frustration over Claude API's non-functionality, reporting it *"gives some goofy error."* Other members were also looking for solutions.
  
- **Selecting local models in 01 dev preview**: Discussion highlighted how the 01 dev preview defaults to OpenAI and how to switch it using `poetry run 01 --local` to select a desired model. This was clarified by a user suggesting commands for model selection.

- **Python script troubleshooting for OpenInterpreter**: A member faced issues running Python code with `interpreter.chat` function, but resolved it by using `from interpreter import OpenInterpreter`.

- **Best local models lag behind GPT-4**: Users compared various local models like Mixtral, Phi, Lama3 with GPT-4, expressing disappointment. One user noted, "If I hadn’t tried GPT-4 first I would be impressed with other models I am sure."

- **GPT-4o speed impresses users**: Users were excited about GPT-4o's performance, reporting speeds of *"minimum 100 tokens/s"* and noting it's *"way more than 2x faster."* A command to try it out was shared: `interpreter --model openai/gpt-4o`.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://interpreter.chat('text">no title found</a>: no description found</li><li><a href="https://visualstudio.microsoft.com/visual-cpp-build-tools.">Microsoft C++ Build Tools - Visual Studio</a>: no description found
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1238960200678113343)** (21 messages🔥): 

- **LiteLLM with Groq-Llama3 confirmed working**: Members discussed issues with integrating LiteLLM, Groq, and Llama3. One member confirmed, *"it works fine"*.
  
- **Website connection issues with M5 board**: *"I never get the website anymore. I've tried re-flashing, and been hammering on this for hours."* A member described extensive troubleshooting failed attempts to connect their 01-Light device.

- **01 hardware app now available**: A member *"had the opportunity to build the 01 hardware beta"* and created a more accessible app version for early-stage testing. They shared the [GitHub repo link](https://github.com/eladdekel/01_For_iOS) and mentioned a pending TestFlight approval.

- **Refund request and support**: A member asked for help with canceling an order and was advised to send an email to *help@openinterpreter.com*.

- **Upcoming 01 batch shipment**: A member inquired about the next 01 batch shipment and was informed that the *"first batch [is] expected for November."*
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1238827699946913812)** (4 messages): 

- **PyWinAssistant Excitement**: A user shared the [GitHub link to PyWinAssistant](https://github.com/a-real-ai/pywinassistant), describing it as "The first open source Large Action Model generalist Artificial Narrow Intelligence that controls completely human user interfaces by only using natural language." They highlighted that PyWinAssistant utilizes Visualization-of-Thought to elicit spatial reasoning in large language models.
- **PyWinAssistant in Action**: Another user mentioned they successfully got PyWinAssistant working and shared a [YouTube video](https://www.youtube.com/live/_XyYoqpJCoQ?si=rA3ijqicagANyt96&t=1993) demonstrating it in action. The video includes examples of PyWinAssistant controlling human user interfaces and features other tools like Autogroq and websim.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/a-real-ai/pywinassistant">GitHub - a-real-ai/pywinassistant: The first open source Large Action Model generalist Artificial Narrow Intelligence that controls completely human user interfaces by only using natural language. PyWinAssistant utilizes Visualization-of-Thought Elicits Spatial Reasoning in Large Language Models.</a>: The first open source Large Action Model generalist Artificial Narrow Intelligence that controls completely human user interfaces by only using natural language. PyWinAssistant utilizes Visualizati...</li><li><a href="https://www.youtube.com/live/_XyYoqpJCoQ?si=rA3ijqicagANyt96&t=1993">pywinassistant working || Autogroq || websim || twelvelabs  + more</a>: ➤ Twitter - https://twitter.com/techfrenaj➤ Twitch  - https://www.twitch.tv/techfren➤ Discord  - https://discord.com/invite/z5VVSGssCw➤ TikTok - https://www....
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1238814017426948098)** (38 messages🔥): 

- **Understanding Variable Shapes in Tensors**: There was a discussion around representing tensors with variable shapes for optimization, particularly in transformers where the number of tokens can change. A user referenced a [Tinygrad Notes article](https://mesozoic-egg.github.io/tinygrad-notes/upcast2.html) and examples from Whisper code ([example 1](https://github.com/tinygrad/tinygrad/blob/a1940ced7746fcdf09068aadf4155e4c1e3641b8/examples/whisper.py#L36-L45), [example 2](https://github.com/tinygrad/tinygrad/blob/a1940ced7746fcdf09068aadf4155e4c1e3641b8/examples/whisper.py#L118-L120)).

- **Clarifying Tensor and Axis Terms**: A question was raised about the difference between "dim" and "axis" in operations like sum and concatenate in tensors. It was noted that they often refer to the same concept but are used in different contexts possibly due to legacy reasons.

- **Handling Missing Gradients in Training**: One user encountered an "AssertionError" related to `Tensor.training` while training a bigram model, which was resolved by setting `Tensor.training = True`. This discussion included references to relevant [GitHub code](https://github.com/tinygrad/tinygrad/pull/4460/files) and suggestions for improving error messages.

- **Aggregating Features with Tensor Operations**: Another user sought advice on implementing feature aggregation for a simple Neural Turing Machine. They discussed tensor operations, provided code examples, and explored optimization techniques, sharing [aggregate feature GitHub code](https://gist.github.com/RaulPPelaez/36b6a3a4bbdb0c373beaf3c1376e8f49).

- **Issues with Backpropagation through `where` Call**: There was a hurdle in backpropagating through a "where" call in tinygrad that worked in PyTorch, leading to a `RuntimeError` due to missing gradients. A solution was proposed involving the use of `detach().where()` to resolve the gradient assignment issue.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/ziereis/3991cf934a0b62caec8f029f12b25135">train.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/RaulPPelaez/36b6a3a4bbdb0c373beaf3c1376e8f49">test_aggregate.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4460/files">optimizer shouldn&#39;t be run without training by geohot · Pull Request #4460 · tinygrad/tinygrad</a>: no description found</li><li><a href="https://github.com/shriar/Neural-Turing-Machine-in-Tinygrad/blob/main/NTM.py">Neural-Turing-Machine-in-Tinygrad/NTM.py at main · shriar/Neural-Turing-Machine-in-Tinygrad</a>: Contribute to shriar/Neural-Turing-Machine-in-Tinygrad development by creating an account on GitHub.</li><li><a href="https://github.com/rs9000/Neural-Turing-machine">GitHub - rs9000/Neural-Turing-machine: NTM in PyTorch</a>: NTM in PyTorch. Contribute to rs9000/Neural-Turing-machine development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/blob/a1940ced7746fcdf09068aadf4155e4c1e3641b8/examples/whisper.py#L36-L45">tinygrad/examples/whisper.py at a1940ced7746fcdf09068aadf4155e4c1e3641b8 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/a1940ced7746fcdf09068aadf4155e4c1e3641b8/examples/whisper.py#L118-L120">tinygrad/examples/whisper.py at a1940ced7746fcdf09068aadf4155e4c1e3641b8 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/rusty1s/pytorch_cluster/blob/master/csrc/cuda/radius_cuda.cu">pytorch_cluster/csrc/cuda/radius_cuda.cu at master · rusty1s/pytorch_cluster</a>: PyTorch Extension Library of Optimized Graph Cluster Algorithms - rusty1s/pytorch_cluster</li><li><a href="https://github.com/torchmd/torchmd-net/blob/75c462aeef69e807130ff6206b59c212692a0cd3/torchmdnet/extensions/neighbors/neighbors_cpu.cpp#L71-L80">torchmd-net/torchmdnet/extensions/neighbors/neighbors_cpu.cpp at 75c462aeef69e807130ff6206b59c212692a0cd3 · torchmd/torchmd-net</a>: Neural network potentials . Contribute to torchmd/torchmd-net development by creating an account on GitHub.</li><li><a href="https://www.pyg.org/)">Home - PyG</a>: PyG is the ultimate library for Graph Neural Networks
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1238858838095040534)** (24 messages🔥): 

- **Embedding models inquiry sparks interest**: A user asked whether the embedding models are open source. No further information or responses were provided to this query.
- **Confusion over billing gets resolved**: One user expressed confusion about billing numbers, particularly an unexplained cost of $0.63. They later resolved their confusion, realizing the number represents the amount due since the last invoice, although they still found the explanation unclear.
- **Web command tokens clarification**: A user questioned why input tokens surged when using command r with web searches, suspecting additional token costs for web visits. Another user confirmed that search results are indeed passed in the context, and this incurs billing.
- **SolidGoldMagikarp token issue analyzed**: A user thanked another for linking an [arXiv paper](https://arxiv.org/abs/2405.05417) that discusses the problem of 'glitch tokens' causing unwanted behavior in language models, and the methods to detect such tokens.
- **Comparing models Aya and Cohere Command Plus**: A user sought benchmarks between the Aya and Cohere Command Plus models, reporting inaccuracies with Aya even at 0 temperature. Another user recommended using Aya solely for translation tasks.

**Link mentioned**: <a href="https://arxiv.org/abs/2405.05417">Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models</a>: The disconnect between tokenizer creation and model training in language models has been known to allow for certain inputs, such as the infamous SolidGoldMagikarp token, to induce unwanted behaviour. ...

  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1238956624513597550)** (2 messages): 

- **Specializing LLMs in Telecom**: One member shared a new challenge for specializing large language models (LLMs) in telecom domains such as 5G. More details about the competition can be found on [Zindi Africa's competition page](https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks).

- **Seeking "Chat with PDF" Application**: Another member inquired whether anyone had created a "chat with PDF" type of application using Cohere. They requested any related repositories or blog posts for reference.

**Link mentioned**: <a href="https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks">Zindi</a>: no description found

  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1238788122800558090)** (23 messages🔥): 

- **GPT-4o still misses the mark**: Users expressed disappointment with GPT-4o, noting it still struggles with simple tasks like listing books on a shelf accurately, despite being faster and cheaper. "Currently in a library and it misses title, adds in ones that aren’t there, gets about 50% right."
- **Voice assistants in bad PR**: Some found recent PR efforts for voice assistants to be embarrassing, partly due to assistants giggling, which was seen as a poor marketing choice. "Just an embarrassing choice."
- **Custom instructions to the rescue**: Discussion included hopes to use custom instructions to make voice assistants less cringeworthy. "I am hoping we can use custom instructions to tone it down a bit!"
- **AGI skepticism spreading**: There was a noticeable skepticism about the imminent arrival of AGI, with some members suggesting they should start a club for non-believers. "Sometimes I feel like I’m one of the few people in the bay area that don’t expect AGI to be released next week."
- **LLMs hitting diminishing returns**: Consensus seems to be building that improvements between versions of LLMs (e.g., 4 vs 3) are showing diminishing returns, and untapped potential still exists within current models. "I keep pointing out to people in convos that 3 vs 2 was a bigger leap than 4 vs 3."
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/)** (1 messages): 

simonw: https://twitter.com/simonw/status/1790121870399782987
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1238814196158824499)** (15 messages🔥): 

- **Fake OpenELM repo warning**: A member alerted that *"it is a FAKE repo, there is no GGUF for OpenELM yet."* Another member sarcastically remarked *"At least the AI industry is catching up to the game industry then."*
- **Pull Request for llamafile Archives**: Shared a [PR link](https://github.com/Mozilla-Ocho/llamafile/pull/412) titled *"Added Script To Upgrade llamafile Archives."* The context mentions porting from [an external blog](https://briankhuu.com/blog/2024/04/06/inplace-upgrading-of-llamafiles-engine-bash-script/).
- **Container Usage Clarified**: There was some confusion about using containers like *podman or kubernetes,* which was clarified with *"using containers is perfectly fine."*
- **Hermes-2-Pro Performance**: A member reported smooth running of *"Hermes-2-Pro-Llama-3-8B-Q5_K_M.gguf"* on *"AMD 5600U,"* with around *10 second response* times and *11GB total RAM usage spikes.*
- **Batch Size Error with Llama and Mistral**: Multiple members reported a recurring error with both *Llama 8B and Mistral* models: *update_slots: failed to find free space in the KV cache, retrying with smaller n_batch = 1*. This issue seems less prominent with higher RAM allocations and other models like *LLaVa 1.5* and *Llama 70B.*

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Mozilla-Ocho/llamafile/pull/412">Added Script To Upgrade llamafile Archives by mofosyne · Pull Request #412 · Mozilla-Ocho/llamafile</a>: Context: #411 Porting https://briankhuu.com/blog/2024/04/06/inplace-upgrading-of-llamafiles-engine-bash-script/ to llamafile</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7165">Add metadata override and also generate dynamic default filename when converting gguf · Issue #7165 · ggerganov/llama.cpp</a>: This is a formalized ticket for this PR #4858 so people are aware and can contribute to figuring out if this idea makes sense... and if so then what needs to be done before this can be merged in fr...
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1239605375242600519)** (9 messages🔥): 

- **German TTS Project Seeks Podcast/YouTube Channel Suggestions**: A member is looking to compile a list of high-quality German YouTube channels with diverse content to train a Text-to-Speech (TTS) model. Another member suggested using [Mediathekview](https://mediathekview.de/) to download broadcasts and films from various German channels.

- **Managing German Video Resources with Mediathekview**: Members discussed using Mediathekview and its potential for downloading and managing German media content, including the feasibility of downloading its database. A suggestion was made to utilize Mediathekview's local database, located at `%userprofile%\.mediathek3\databasemediathekview.mv.db`.

- **Using Mediathekview's JSON API**: It was pointed out that Mediathekview has a JSON API that can be used for querying data, with a reference to the [GitHub repository](https://github.com/59de44955ebd/MediathekViewWebVLC/blob/main/mediathekviewweb.lua) for more details.

- **Encouraged to Maintain English Communication**: A member reminded others to keep the discourse in English within the channel.

- **Excitement Over RT Audio Interface in Non-Chat Applications**: One user expressed excitement about the "RT Audio interface" and inquired about any first-hand experiences or results in non-chat applications, indicating a keen interest in its capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/59de44955ebd/MediathekViewWebVLC/blob/main/mediathekviewweb.lua">MediathekViewWebVLC/mediathekviewweb.lua at main · 59de44955ebd/MediathekViewWebVLC</a>: MediathekViewWeb Lua extension for VLC. Contribute to 59de44955ebd/MediathekViewWebVLC development by creating an account on GitHub.</li><li><a href="https://podtail.com/de/top-podcasts/de/">Die 100 beliebtesten Podcasts im Moment &ndash; Deutschland</a>: Diese Liste zeigt die derzeit 100 beliebtesten Podcasts mit aktuellen Daten von Apple und Podtail.</li><li><a href="https://hypeauditor.com/top-youtube-all-germany/">Top YouTube Channels in Germany | HypeAuditor YouTube Ranking</a>: Find the most popular YouTube channels in Germany as of May 2024. Get a list of the biggest YouTubers in Germany.
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1239228517527588864)** (2 messages): 

```html
- **Demo status inquiry**: A user asked, *"Is the demo down?"* but there was no response to this query.
- **Positive feedback**: Another user remarked, *"It's really nice,"* expressing satisfaction without further elaboration.
```
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1239608271225098290)** (4 messages): 

- **Claude 3 Haiku vs Llama 3b sparks interest**: Members discussed the performance of **Claude 3 Haiku** versus **Llama 3b Instruct**. One member shared their experience building an automated scoring service to extract entities from documents and expressed the need for accurate submodel matching, mentioning that initial attempts with fuzzy string algorithms and similar pattern matching were unsuccessful. 


  

---


**LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1238796719068811275)** (6 messages): 

- **Speculation on Audio Integration**: Members talked about the possibility that **OpenAI** is working on something related to audio, with one suggesting it might involve *"audio in-out support directly to some assistant."*
- **OpenAI Spring Update Teased**: A YouTube link was shared, hinting at new features, including the **introduction of GPT-4o** as part of the *OpenAI Spring Update*. The event is set to have updates on ChatGPT and more.
- **Scarlett Johansson as a Voice**: The community expressed surprise and amusement that **Scarlett Johansson** has been featured as the voice in the new update. One member exclaimed, *"cant believe they got scarjo to do the voice"* followed by *"lol"*.

[Watch the full update here](https://www.youtube.com/watch?v=DQacCB9tDaw)

**Link mentioned**: <a href="https://www.youtube.com/watch?v=DQacCB9tDaw">Introducing GPT-4o</a>: OpenAI Spring Update – streamed live on Monday, May 13, 2024. Introducing GPT-4o, updates to ChatGPT, and more.

  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1239211766035251210)** (3 messages): 

- **AlphaFold3 Federation invites for sign-up**: A member announced the commencement of an **AlphaFold3 Federation** and shared a [sign-up link](https://lu.ma/swinnyfl) for an upcoming meet at 9pm EST on May 12th. The agenda includes progress updates, pipeline design, and Q&A.

- **Request for server ROLE information**: A member inquired about where to find ROLE information for the server and tagged another user for clarification. No further details were provided on the available roles.

**Link mentioned**: <a href="https://lu.ma/swinnyfl">AlphaFold3 [AF3] Federation Meet · Luma</a>: Current Progress Update A talk by the lead developer on the current status of Alpha Fold 3 integration. Discussion of any issues encountered during the initial…

  

---


**Alignment Lab AI ▷ #[fasteval-dev](https://discord.com/channels/1087862276448595968/1147528620936548363/1239333780695683124)** (3 messages): 

- **Fasteval project might cease**: A member inquired about the continuation of the **fasteval** project. Another member responded that they are not planning to continue it but are willing to transfer ownership of the [GitHub project](https://github.com/) if someone responsible wishes to take it over; otherwise, they suggest archiving the fasteval channels.
  

---



**AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1238800597679865927)** (1 messages): 

- **Modify AI Town settings**: A member inquired about the ability to modify the **character moving speed** and the **number of NPCs** in AI town. This suggests interest in customizing gameplay mechanics.
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1238801086161358921)** (1 messages): 

- **Optimize NPC interaction frequency for better performance**: A user inquired if it was possible to adjust the code to reduce the interaction frequency between NPCs. They suggested reallocating computation power to enhance the player-NPC interaction, noting that running AI town on a local machine with the llama3 model is quite taxing.
  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=KQ-xGVFHDkw
  
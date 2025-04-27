---
id: efb00c05-281c-4d5f-873c-bdd3fe101a25
title: Not much happened piday
date: '2024-03-14T23:53:52.756548Z'
original_slug: ainews-not-much-happened-piday
description: >-
  **DeepMind** announces **SIMA**, a generalist AI agent capable of following
  natural language instructions across diverse 3D environments and video games,
  advancing embodied AI agents. **Anthropic** releases **Claude 3 Haiku**, their
  fastest and most affordable model, now available via API and Perplexity. New
  research explores language model scaling laws and efficient training methods
  like Branch-Train-MiX (BTX) for mixture-of-experts models. **Cohere**
  introduces the Command-R model focusing on retrieval-augmented generation and
  tool use for LLM applications. The **EU AI Act** is approved, mandating
  transparency in training data for GPAI systems. AI's impact on software
  engineering job growth is discussed, with predictions of increased engineers
  due to easier coding. Memes humorously reflect on AI software engineers and
  notable figures like Andrej Karpathy.
companies:
  - google-deepmind
  - anthropic
  - cohere
  - eu
models:
  - claude-3-haiku
topics:
  - embodied-ai
  - reinforcement-learning
  - language-model-scaling
  - mixture-of-experts
  - retrieval-augmented-generation
  - ai-regulation
  - differential-privacy
  - software-engineering
  - model-training
  - natural-language-processing
  - humor
people:
  - demishassabis
  - fchollet
  - abacaj
  - andrej_karpathy
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 3/13/2024-3/14/2024. We checked [**358** Twitters](https://twitter.com/i/lists/1585430245762441216) and **21** Discords (**336** channels, and **3518** messages) for you. Estimated reading time saved (at 200wpm): **426 minutes**.

---

It's the [anniversary of GPT4](https://x.com/swyx/status/1636067268802285568?s=20), but no GPT5 for you today. [Join @elonmusk](https://x.com/swyx/status/1767664455889097009?s=20) in checking out [the latest Latent Space pod with Suno AI](https://x.com/FanaHOVA/status/1768327038094750040?s=20)?

https://www.youtube.com/watch?v=gYXjn-V7AEw&feature=youtu.be

(Also we missed highlighting [the Figure 01 launch](https://x.com/coreylynch/status/1767927194163331345?s=20) yesterday, which in retrospect we'd rank slightly higher than Deepmind SIMA in impressiveness/near term importance).

---

**Table of Contents**

[TOC] 

---

# PART X: AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs


**AI Agents and Environments**

1. [DeepMind announces SIMA](https://twitter.com/GoogleDeepMind/status/1767918515585994818), a generalist AI agent that can follow natural language instructions in a broad range of 3D environments and video games, marking an important step towards agents that can tackle complex tasks requiring planning and sub-tasks. (537,888 impressions)

2. [DeepMind's SIMA agent](https://twitter.com/demishassabis/status/1767977070603219255) demonstrates the ability to follow natural language instructions to carry out tasks across a wide array of game worlds, similar to how a human would play. This is an exciting development in embodied AI agents. (178,835 impressions)

3. [The SIMA research](https://twitter.com/GoogleDeepMind/status/1767918524641554899) focuses on developing embodied AI agents that can translate abstract language into useful actions, using video games as safe, accessible testing environments rather than optimizing for high scores. (24,983 impressions)

**Large Language Models and Scaling**

1. [Anthropic introduces Claude 3 Haiku](https://twitter.com/AnthropicAI/status/1768018310615151002), their fastest and most affordable model, now available in the API and on Perplexity for Claude Pro subscribers. (299,766 impressions)

2. [Language models scale reliably with over-training and on downstream tasks.](https://twitter.com/arankomatsuzaki/status/1768089079978041552) A new paper explores gaps in LM scaling laws, providing insights into over-training and linking model perplexity to downstream performance. (10,589 impressions) 

3. [Branch-Train-MiX (BTX)](https://twitter.com/omarsar0/status/1767919732542378089) is a new approach for training large language models more efficiently by mixing expert LLMs into a Mixture-of-Experts LLM. It is shown to be more efficient than training a larger generalist LLM or several separate specialized LLMs. (11,042 impressions)

**AI Coding Assistants and Software Engineering**

1. [@fchollet predicts](https://twitter.com/fchollet/status/1767935813646716976) there will be more software engineers in five years than today, estimating growth from 26-27M today to 30-35M in 5 years. He argues that making it easier to code has historically led to more coding jobs. (188,949 impressions)

2. [Cohere's Command-R model](https://twitter.com/cwolferesearch/status/1768009088863031766) focuses on retrieval augmented generation (RAG) and tool usage - two key skills for building LLM applications. It addresses issues in scaling proof-of-concept LLM apps to production. (2,297 impressions)

3. A perspective that [AI will enable more software engineers](https://twitter.com/omarsar0/status/1768000459212530052), and that fancy demos are causing overreaction. Most AI coding solutions will likely have limited scope and need human supervision. (15,308 impressions)

**AI Safety and Regulation** 

1. The [EU AI Act has been approved by Parliament](https://twitter.com/mmitchell_ai/status/1767949324053561560), representing big and largely positive AI news. (11,126 impressions)

2. Key requirements in the AI Act include that [GPAI systems must publish "detailed summaries of the content used for training."](https://twitter.com/mmitchell_ai/status/1767949866750362041) (1,759 impressions)

3. A paper on ["Privacy-Preserving In-Context Learning with Differentially Private Few-Shot Generation"](https://twitter.com/_nerdai_/status/1768022092849541453) is highlighted as promising work in light of the AI Act's approval. The paper proposes using a pre-trained LLM to generate differentially private synthetic examples from private datasets. (79 impressions)

**Memes and Humor**

1. A meme jokes that [an "AI software engineer" that can automate everything would be used as a product rather than to dominate the market.](https://twitter.com/abacaj/status/1767810161282855308) (586,645 impressions)

2. A humorous tweet imagines [Andrej Karpathy leaving Tesla](https://twitter.com/Nexuist/status/1768033939199869050) because he suggested changing a learning rate constant from 0.086 to 0.0855541. (270 impressions)

3. A meme suggests that [people waiting for GPT-5 to drop will be disappointed again.](https://twitter.com/cto_junior/status/1768138360797769915) (1,378 impressions)

**Other Notable Topics**

- Together Computing [raises $106M](https://twitter.com/togethercompute/status/1767943482054967555) to rapidly bring research innovations to production and build a platform for running generative AI applications on open-source models at scale. (112,647 impressions)

- [Keras 3 benchmarks](https://twitter.com/fchollet/status/1768010983224885400) show no single "best" backend, with the optimal choice depending on model architecture. Keras 3 models are consistently faster than PyTorch without requiring custom optimizations. (51,849 impressions)

- A new [LlamaParse document parsing solution](https://twitter.com/llama_index/status/1767948064659210310) excels at extracting images, tables, and charts, and can be steered via natural language instructions. It integrates with LlamaIndex for building RAG systems over complex documents. (85,018 impressions)


---

# PART 0: Summary of Summaries of Summaries

> Since [Claude 3 Haiku was released recently](https://x.com/anthropicai/status/1768018310615151002?s=46&t=90xQ8sGy63D2OtiaoGJuww), we're adding them to this summary run for you to compare vs our custom GPT summarizer (all are different than the smol model running the Part 1/2 summaries). We'll keep running these side by side for a little longer while we build the AINews platform for a better UX. We've noticed that the same prompts result in consistently different output in the 3 Claude models. We'll be trying to tweak prompts in tomorrow's iteration to get Haiku at least behaving.

## Claude 3 Haiku (3B?)


- **Nvidia Puts the Brakes on Translation Layers**: Nvidia has implemented a ban on using translation layers to run CUDA-based software on non-Nvidia chips, targeting projects like ZLUDA, with further details discussed in a Tom's Hardware article. Some members expressed skepticism over the enforceability of this ban.

- **CUDA Error Riddles and Kernel Puzzles**: CUDA developers are troubleshooting errors like CUBLAS_STATUS_NOT_INITIALIZED with suggestions pointing to tensor dimensions and memory issues, as seen in related forum posts. Other discussions centered around cuda::pipeline efficiency and understanding effective bandwidth versus latency, referencing resources such as Lecture 8 and a blog on CUDA Vectorized Memory Access.

- **CUTLASS Installation Q&A for Beginners**: New AI engineers sought advice on installing CUTLASS, learning that it's a header-only template library, with installation guidance available on the CUTLASS GitHub repository, and requested resources for implementing custom CUDA kernels.

- **Ring-Attention Project Gets the Spotlight**: A flurry of activity took place around the ring-attention experiments with conversations ranging from benchmarking strategies to the progression of the 'ring-llama' test. An issue with a sampling script is in the process of being resolved as reflected in the Pull Request #13 on GitHub, and the Ring-Attention GitHub repository was shared for those interested in the project.

- **Lecture 8 on CUDA Performance Redone and Released**: The CUDA community received a re-recorded version of Lecture 8: CUDA Performance Checklist, which includes a YouTube video, code on GitHub, and slides on Google Docs, garnering appreciation from community members. Discussions ensued on the mentioned DRAM throughput numbers and performance differences in coarsening.

## Claude 3 Sonnet (14B?)

1. **New AI Model Releases and Capabilities**:
   - [Cerebras unveils CS-3 AI accelerator](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine) capable of training up to **24 trillion parameter models** on a single chip, with 4 trillion transistors and 125 petaflops of compute power.
   - [Anthropic releases Claude 3 Haiku](https://openrouter.ai/models/anthropic/claude-3-haiku:beta), a fast and cost-efficient model available on OpenRouter, running at ~120 tokens/s and 4 million prompt tokens per dollar.
   - [Cohere's Command-R model](https://openrouter.ai/models/cohere/command-r) with 128k token context window is now on OpenRouter, at 2 million prompt tokens per dollar.
   - [DeepMind's SIMA](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/) is a new generalist AI agent that can understand natural language instructions in video game environments, though its technical details lack transparency.

2. **AI Safety and Vulnerability Concerns**:
   - A [new paper on ComPromptMized](https://sites.google.com/view/compromptmized) reveals prompt injection attacks on AI models like **Gemini Pro, ChatGPT 4.0, and LLaVA**, highlighting vulnerabilities in GenAI-powered applications.
   - Discussions around [OpenAI's security incident](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca) and the implications for AI security.
   - Debates on the EU's new [AI legislation](https://www.europarl.europa.eu/news/en/press-room/20240308IPR19015/artificial-intelligence-act-meps-adopt-landmark-law) requiring disclosure of AI-generated content and avoiding illegal outputs, with concerns over practicality and impact on open source models.

3. **Advances in Multimodal AI and Interpretability**:
   - [MoAI (Mixture of All Intelligence)](https://arxiv.org/abs/2403.07508) incorporates auxiliary visual information from computer vision models into LLVMs for enhanced zero-shot vision-language tasks. [GitHub repo](https://github.com/ByungKwanLee/MoAI) and [Hugging Face demo](https://huggingface.co/BK-Lee/MoAI-7B) available.
   - [Soniajoseph announces open-source multimodal mechanistic interpretability library](https://twitter.com/soniajoseph_/status/1767963316943728779) for Hugging Face CLIP/ViT models, with details on [LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic).
   - [LlamaParse](https://twitter.com/llama_index/status/1767948064659210310) is a new document parser that can handle images, tables, charts, and follow natural language instructions, outperforming others.

4. **Synthetic Data Generation for LLM Pretraining and Finetuning**:
   - Discussions on [using synthetic data](https://eugeneyan.com/writing/synthetic/) as an alternative to human annotations for pretraining, instruction-tuning, and preference-tuning of language models, with potential quality advantages.
   - [AI News newsletter](https://buttondown.email/ainews/) provides valuable summaries of AI discords and top Twitter accounts, recommended by experts like Soumith Chintala and Andrej Karpathy.
   - Debates on whether [fine-tuning can impart new knowledge](https://github.com/stanfordnlp/dspy/blob/0c1d1b1b2c9b5d6dc6d565a84bfd8f17c273669d/docs/api/optimizers/BootstrapFinetune.md?plain=1#L5) to models, and the efficiency of fine-tuning for style transfer versus knowledge acquisition.

## Claude 3 Opus (>220B?)

- **Hermes 2.5 Outperforms Hermes 2**: Updates to **Hermes 2.5** with [code instruction examples](https://github.com/NousResearch/Hermes-Function-Calling) have resulted in notable performance improvements compared to Hermes 2. The community is actively discussing related topics like JSON mode versus function calling, clarifying that JSON mode necessitates a predefined schema, whereas function calling involves executed function responses.

- **Cerebras Reveals AI Titan**: **Cerebras Systems** launches its **CS-3** AI accelerator capable of training up to 24 trillion parameter models. This mammoth chip has 4 trillion transistors and is expected to deliver 125 petaflops of AI compute power, as detailed in the [press release](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine).

- **AI Trends Galore**: From Stanislaw Lem's science fiction recommendations to SDAM development and *Devin the AI Software Engineer's* [YouTube debut](https://www.youtube.com/watch?v=NSPtrrUQ_fw), the community has its eyes peeled on a variety of AI and engineering marvels. The eagerness for open source models that could provide 100k+ context and concerns about privacy in information sharing also embodies the diverse range of interests.

- **Debating Decentralization in AI**: Amid discussions of **TAO** potentially challenging **Hugging Face**, the community delves into debates about centralized vs. decentralized AI model platforms. The introduction of a new project, **Shoggoth**, sparks curiosity, yet detailed information is lacking due to broken links.

- **Claude 3 Powers Haiku Creation**: **[Perplexity Labs](https://labs.pplx.ai)** introduces **Claude 3 Haiku**, enticing users to experiment with poetic AI capabilities for free, bolstering the platform's suite of creative tools.

- **A Diverse AI Toolbox**: Engineers and developers are actively engaging with **Perplexity AI** for a multitude of uses such as coding support and SE troubleshooting, while creatively experimenting with newly added features like AI-generated **Haikus**. The platform's local search capabilities are now enhanced by integrating Yelp and Maps for more efficient local business discoveries.

- **AI Ecosystem Rivalries and Perspectives**: The guild hosts vigorous debates comparing various AI models; **GPT-4** and **Mistral** are pitted against each other, with the former being argued as superior by some, while others favor the latter's speed. 

- **API Integration and Model Limitations**: Users discuss using Perplexity's models for complex queries and utilizing the **Perplexity API** for developing applications, such as a Firefox extension, while noting a **25MB upload limit** and uncertain performance with extensive databases, such as those related to real estate.

- **APIs in Focus: Questions and Potential**: An inquiry about the closed beta of URL citations in **Perplexity's API** awaits insider insight, while others seek advice on the API's performance for condition checking. Members also examine the behavior of the "return_citations" option and determine the best models for handling up-to-date information, singling out **Sonar-small-online** and **sonar-medium-online** for their real-time data access capabilities.

- **Tackling LM Studio Outside the UI Box**: Users examined running LM Studio's API services on a *home network* without the user interface, focusing on server mode and localhost connections. It was highlighted that `llama.cpp` is a viable option, sustaining **AVX** without **AVX2** and allowing independence from the LM Studio UI, per its [GitHub repository](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md).

- **LM Studio Limitations Spur Creative Workarounds**: Among LM Studio's constraints is the inability to launch services or connect to the internet programmatically; users creatively employed batch files and PowerShell scripts to automate starting the LM Studio inference server, showcasing the community's resourcefulness.

- **Mighty Models Extended and Examined**: The [Nous-Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) model expanded to a 128k token context window using the *YaRN* method, alongside discussions about model perplexity and humorous disappointment with the "Yet Another <X>" naming convention. Moreover, some shared format-specific obstacles, such as incompatibility issues with `llama.cpp` for the **Command-R 35B v1.0 GGUF** format.

- **ROCM Round-Up**: Real-world experiences with **ROCm** support in LM Studio were shared, including troubleshooting steps like using AMD's cleanup tool and avoiding PRO drivers. Vision models proved challenging, and it was advised to choose Nvidia GPUs over AMD for image generation projects. Additionally, a user found that disabling the iGPU on a Gigabyte motherboard in BIOS settings enabled better usage of their RX 7900 XT with ROCm.

- **Hardware Conversation Heats Up**: The cost of **SLI/NVLink** sparked debates, complemented by discussions on overcoming Mac OS's **minimum VRAM requirements**, strategizing PC hardware upgrades, and balance in multi-model deployments in LM Studio. Separate dialogues covered selecting the right dual-purpose monitor, with an inclination towards OLED screens despite burn-in risks and preferences for high refresh rates to match top-tier graphics cards like the Nvidia 4090.

- **AI Gold Rush Continues**: Various **AI startups** like **Cognition**, **Magic**, and **Fluent** have attracted impressive venture capital investments, with discussions drawing attention to the ongoing trend of significant funding for AI companies. Participants shared a collection of tweets that gave an overview of companies and their raised capital, referencing a feed at [chiefaioffice](https://x.com/chiefaioffice/status/1767680581112873242?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **Cerebras Flexes Its AI Muscles**: **Cerebras Systems** unveiled the **CS-3 AI accelerator**, claiming it's capable of training up to 24 trillion parameter models. The announcement has sparked interest and the discussion also mentioned a related [press release](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine) and a [tweet](https://x.com/cerebrassystems/status/1767929699177767325?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **Security Red Alert at OpenAI**: Members discussed a **security issue** at OpenAI with references to a detailed **Post Mortem** analysis available in a [gist](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca). The community delved into the implications for AI security.

- **Prep Up for Synthetic Data Insights**: An upcoming presentation on Synthetic Data for Finetuning was announced with materials to read beforehand at [Eugene Yan's writing](https://eugeneyan.com/writing/synthetic/). The use of synthetic data as an alternative for human annotations in pretraining and fine-tuning language models was underscored by the group.

- **Rethinking Data in LLMs**: In-depth discussions explored the use of **synthetic data** for pretraining and fine-tuning LLMs, and the implications for knowledge acquisition via fine-tuning. A blog post providing significant insights, referred to during the discussions, can be found at [eugeneyan.com](https://eugeneyan.com/writing/synthetic/), and a summary service by [AI News](https://buttondown.email/ainews/) was mentioned by engineering professionals as a valuable resource.

## ChatGPT (GPT4T)

<div><ul><li><p><strong>Positional Encodings in Language Models</strong>: The <strong>Nous Research AI Discord</strong> discussed the critical role of positional encodings in enhancing the performance of causal language models for processing longer sequences. A pivotal paper, <a target="_new" href="https://arxiv.org/pdf/2203.16634.pdf">"Understanding Positional Encodings in Large Language Models"</a>, was highlighted for offering deep insights into this area.</p></li><li><p><strong>Hermes 2.5 Function Calling</strong>: Significant performance gains were observed with Hermes 2.5's introduction, especially in function calling versus JSON mode, drawing community attention to its practical examples at <a target="_new" href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub</a>.</p></li><li><p><strong>CS-3 AI Accelerator by Cerebras</strong>: Cerebras Systems unveiled its CS-3 AI accelerator, capable of training models up to 24 trillion parameters. This hardware milestone, featuring 4 trillion transistors and promising 125 petaflops of AI compute, was detailed in their <a target="_new" href="https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine">press release</a>.</p></li><li><p><strong>Perplexity AI's Claude 3 Haiku</strong>: <strong>Perplexity AI</strong> showcased Claude 3 Haiku, emphasizing the model's ability to craft Haikus, as part of their effort to expand the creative capabilities of AI, with further details available on <a target="_new" href="https://labs.pplx.ai">Perplexity Labs</a>.</p></li><li><p><strong>Local Model Testing in OpenAI Discord</strong>: Discussions around testing local models, particularly Meditron and Mistral, on setups with up to 4xT4s using LLM Studio, were prominent, including the best practices for fine-tuning these models for optimum performance.</p></li><li><p><strong>Interpretability in Multimodal Models</strong>: The <strong>Alignment Lab AI Discord</strong> is seeking collaborators for open-source interpretability projects focusing on multimodal models, with further details shared by soniajoseph_ on <a target="_new" href="https://twitter.com/soniajoseph_/status/1767963316943728779">Twitter</a> and <a target="_new" href="https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic">LessWrong</a>.</p></li><li><p><strong>Devin, the Autonomous Software Engineer</strong>: Both <strong>Skunkworks AI</strong> and <strong>AI Engineer Foundation</strong> highlighted Devin, introduced as the world’s first autonomous software engineer by Cognition Labs. This AI's capabilities and introduction are covered in their <a target="_new" href="https://www.cognition-labs.com/blog">blog post</a> and a <a target="_new" href="https://www.youtube.com/watch?v=NSPtrrUQ_fw">YouTube video</a>.</p></li></ul></div>

---

# PART 1: High level Discord summaries




## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Positional Encodings Decoded**: There is significant discussion about the role of positional encodings in causal language models, with insights pointing to the necessity of positional encodings for handling longer sequences effectively. A paper of interest in this regard is ["Understanding Positional Encodings in Large Language Models"](https://arxiv.org/pdf/2203.16634.pdf).

- **Unlocking the Secrets of Hermes 2.5**: Updates to **Hermes 2.5** with [code instruction examples](https://github.com/NousResearch/Hermes-Function-Calling) have resulted in notable performance improvements compared to Hermes 2. The community is actively discussing related topics like JSON mode versus function calling, clarifying that JSON mode necessitates a predefined schema, whereas function calling involves executed function responses.

- **Cerebras Reveals AI Titan**: Cerebras Systems launches its **CS-3** AI accelerator capable of training up to 24 trillion parameter models. This mammoth chip has 4 trillion transistors and is expected to deliver 125 petaflops of AI compute power, as detailed in the [press release](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine).

- **AI Trends Galore**: From Stanislaw Lem's science fiction recommendations to SDAM development and *Devin the AI Software Engineer's* [YouTube debut](https://www.youtube.com/watch?v=NSPtrrUQ_fw), the community has its eyes peeled on a variety of AI and engineering marvels. The eagerness for open source models that could provide 100k+ context and concerns about privacy in information sharing also embodies the diverse range of interests.

- **Debating Decentralization in AI**: Amid discussions of **TAO** potentially challenging **Hugging Face**, the community delves into debates about centralized vs. decentralized AI model platforms. The introduction of a new project, **Shoggoth**, sparks curiosity, yet detailed information is lacking due to broken links.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Claude 3 Powers Haiku Creation**: **[Perplexity Labs](https://labs.pplx.ai)** introduces **Claude 3 Haiku**, enticing users to experiment with poetic AI capabilities for free, bolstering the platform's suite of creative tools.

- **A Diverse AI Toolbox**: Engineers and developers are actively engaging with **Perplexity AI** for a multitude of uses such as coding support and SE troubleshooting, while creatively experimenting with newly added features like AI-generated **Haikus**. The platform's local search capabilities are now enhanced by integrating Yelp and Maps for more efficient local business discoveries.

- **AI Ecosystem Rivalries and Perspectives**: The guild hosts vigorous debates comparing various AI models; **GPT-4** and **Mistral** are pitted against each other, with the former being argued as superior by some, while others favor the latter's speed.

- **API Integration and Model Limitations**: Users discuss using Perplexity's models for complex queries and utilizing the **Perplexity API** for developing applications, such as a Firefox extension, while noting a **25MB upload limit** and uncertain performance with extensive databases, such as those related to real estate.

- **APIs in Focus: Questions and Potential**: An inquiry about the closed beta of URL citations in **Perplexity's API** awaits insider insight, while others seek advice on the API's performance for condition checking. Members also examine the behavior of the "return_citations" option and determine the best models for handling up-to-date information, singling out **Sonar-small-online** and **sonar-medium-online** for their real-time data access capabilities.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

**Tackling LM Studio Outside the UI Box**: Users examined running LM Studio's API services on a *home network* without the user interface, focusing on server mode and localhost connections. It was highlighted that `llama.cpp` is a viable option, sustaining **AVX** without **AVX2** and allowing independence from the LM Studio UI, per its [GitHub repository](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md).

**LM Studio Limitations Spur Creative Workarounds**: Among LM Studio's constraints is the inability to launch services or connect to the internet programmatically; users creatively employed batch files and PowerShell scripts to automate starting the LM Studio inference server, showcasing the community's resourcefulness.

**Mighty Models Extended and Examined**: The [Nous-Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) model expanded to a 128k token context window using the *YaRN* method, alongside discussions about model perplexity and humorous disappointment with the "Yet Another <X>" naming convention. Moreover, some shared format-specific obstacles, such as incompatibility issues with `llama.cpp` for the **Command-R 35B v1.0 GGUF** format.

**ROCM Round-Up**: Real-world experiences with **ROCm** support in LM Studio were shared, including troubleshooting steps like using AMD's cleanup tool and avoiding PRO drivers. Vision models proved challenging, and it was advised to choose Nvidia GPUs over AMD for image generation projects. Additionally, a user found that disabling the iGPU on a Gigabyte motherboard in BIOS settings enabled better usage of their RX 7900 XT with ROCm.

**Hardware Conversation Heats Up**: The cost of **SLI/NVLink** sparked debates, complemented by discussions on overcoming Mac OS's **minimum VRAM requirements**, strategizing PC hardware upgrades, and balance in multi-model deployments in LM Studio. Separate dialogues covered selecting the right dual-purpose monitor, with an inclination towards OLED screens despite burn-in risks and preferences for high refresh rates to match top-tier graphics cards like the Nvidia 4090.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **AI Gold Rush Continues**: Various **AI startups** like **Cognition**, **Magic**, and **Fluent** have attracted impressive venture capital investments, with discussions drawing attention to the ongoing trend of significant funding for AI companies. Participants shared a collection of tweets that gave an overview of companies and their raised capital, referencing a feed at [chiefaioffice](https://x.com/chiefaioffice/status/1767680581112873242?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **Cerebras Flexes Its AI Muscles**: **Cerebras Systems** unveiled the **CS-3 AI accelerator**, claiming it's capable of training up to 24 trillion parameter models. The announcement has sparked interest and the discussion also mentioned a related [press release](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine) and a [tweet](https://x.com/cerebrassystems/status/1767929699177767325?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **Security Red Alert at OpenAI**: Members discussed a **security issue** at OpenAI with references to a detailed **Post Mortem** analysis available in a [gist](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca). The community delved into the implications for AI security.

- **Prep Up for Synthetic Data Insights**: An upcoming presentation on Synthetic Data for Finetuning was announced with materials to read beforehand at [Eugene Yan's writing](https://eugeneyan.com/writing/synthetic/). The use of synthetic data as an alternative for human annotations in pretraining and fine-tuning language models was underscored by the group.

- **Rethinking Data in LLMs**: In-depth discussions explored the use of **synthetic data** for pretraining and fine-tuning LLMs, and the implications for knowledge acquisition via fine-tuning. A blog post providing significant insights, referred to during the discussions, can be found at [eugeneyan.com](https://eugeneyan.com/writing/synthetic/), and a summary service by [AI News](https://buttondown.email/ainews/) was mentioned by engineering professionals as a valuable resource.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord Summary

**Visualizing Token Probabilities**: Discussions indicated a need for visualizing token probability in sentences, with suggestions on using **lm_head's** output and softmax. However, there seems to be a lack of specific plugins for this visualization.

**AI's Fast-Paced Progress**: Conversations were buzzing about the rapid development in AI, with anticipation for Elon Musk's **Grok model** and chatter about OpenAI's authenticity.

**Unsloth AI Battles Colab Woes**: Fixes for Google Colab's PyTorch update issues were shared by **Unsloth AI**, along with a command list to help users rectify these problems themselves. Unsloth AI's compatibility was clarified, noting that it doesn't support multi-GPU or GGUF formatted models for fine-tuning yet, but it can handle 4-bit quantization for single-GPU setups.

**Data Preparation Discussion**: An active conversation recommended the creation of an FAQ for data preparation, suggesting a more automated approach could be beneficial.

**Sophia Optimizer Sparks Interest**: A new optimizer, **Sophia**, proposed for reducing language model training time and cost, caught the attention of the community. While untested, there's optimism it could replace existing optimizers effectively ([Sophia Optimizer Paper](https://arxiv.org/abs/2305.14342)).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **GPT-3.5 rocks Python scripting**: A conversation highlighted **GPT-3.5's** ability to use Python to write a program that successfully generates examples of repeated morphemes. The complexity of the task didn't deter some successful outputs from being shared.

- **Local Models Command Attention**: Engineers shared insights on using **LLM Studio** to test local models, with powerful inference reported on setups with up to 4xT4s, and **Meditron** was mentioned as a standout model. The conversation expanded to considerations of fine-tuning models like **Mistral**, where an **A100 40GB GPU** was recommended for the task, though fine-tuning **GPT-3.5** could be attempted without a GPU.

- **GPT-5 Rumors Quashed**: Buzz around accidental mention of "Priority access to GPT-4 and GPT-5 Turbo" on a **Microsoft Copilot page** stirred speculations on **GPT-5**'s existence, which turned out to be a typo. This led enthusiasts to agree that the launch of GPT-5 isn't forthcoming. Relevant link: [Microsoft Copilot | Microsoft AI](https://www.microsoft.com/en-us/microsoft-copilot).

- **System Glitches in GPT-4**: Users experienced widespread outages with **GPT-4**, highlighting the issue on multiple platforms such as iOS apps and web browsers, including Chrome and Edge. Some found that image attachments offered a temporary fix, and checking the OpenAI [status page](https://status.openai.com/) was advised for updates.

- **Cultural Differences Affect API Understanding**: In discussions about **Assistant API**, a user observed that the API misinterprets the figure "450,00" due to comma placement, which could lead to significant errors in data handling. Adjusting for local cultural formats such as setting the locale and using positive and negative examples was recommended to improve accuracy.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

**DeepMind Debuts Generalist Gaming AI**: DeepMind introduces **SIMA**, exhibiting natural-language proficiency in varied gaming settings, but the research community flags insufficient technical detail. Critics are wary of the metrics used to validate the agent's effectiveness, debating the definition of game expertise and AI's broader implications in competitive gaming scenarios, particularly within unpredictable multi-agent systems like **BR games**.

**Research Paper Paywalls Provoke Ire**: Accessibility to cutting-edge AI research is hampered by publisher paywalls, sparking discussions around innovative neural network training dynamics and the integration of diverse network architectures. Concerns also arise about the consequences of watermarking AI-generated content, potentially limiting its practicality.

**Interpretability Library for Multimodal Models Launched**: A new multimodal mechanism interpretability library garners interest for collaboration, while discussions delve into the complexities of model agnosticism and language-dependent dynamics in multilingual transformers. The exploration of tokenization bias in bilingual models is highlighted along with a vector-DB-lookup method for deeper insights into model latent representations.

**Language Models Enter the Thunderdome**: The **LM evaluation harness** community is experimenting with learning rate cooldowns for benchmark improvement. They face challenges in adding logits due to recent API changes aimed at security, spurring discourse on adapting tasks for generative models and testing different checkpoints for model performance.

**Megatron Meets NeoX**: A GitHub [pull request](https://github.com/EleutherAI/gpt-neox/pull/1185) sparks a debate about the potential benefits of aligning GPT-NeoX more closely with the upstream Megatron for Transformer Engine integration. Community feedback is solicited to weigh the advantages of this strategy against code divergence.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Speed Demons Look to Quantum and Groq**: Engineers discussed methods to accelerate inference on *Phi 2 fine-tunes* using GPUs like the A100 40GB, with options such as vLLM, Olama, or Axolotl. Quantization was mentioned as a potential speed booster, with Groq's NPU showcasing 500 tokens per second performance on *mixtral*.

- **Model Legislation Drama**: EU's new AI legislation and recent copyright takedown notices sparked heated debates around copyright, AI-generated content, and DMCA compliance. Open source proponents are wrestling with government constraints against sharing model weights.

- **Prompt Engineering Hype**: Tools like SuperPrompt and a new autocomplete tag generator for Danbooru tags have been proposed to improve the capabilities of smaller models in tasks typically reserved for larger LLMs.

- **AI Data Tug-of-War**: There's considerable excitement around a new paper on **MoAI**, which employs auxiliary visual information from specialized computer vision models. The efforts underscore the AI community's ongoing push to create versatile LLVMs capable of enhanced zero-shot vision-language tasks.

- **Memory Mechanics Misunderstood**: A discussion clarified misconceptions around memory usage in large models, pointing out that mmap may hide the actual memory usage, which doesn't reflect until the data is accessed.




---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Visual Comparisons Just Got Easier**: The [Open LLM Leaderboard Viz] now features the ability to reorder metrics and compare up to three models visually, as demonstrated in a new update on [HuggingFace Spaces](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz).
- **Evolving NER with Custom Labels**: A new model called GLiNER enables on-the-fly custom label selection for Named Entity Recognition (NER), offering more adaptability compared to fixed-entity models. Check out the demo and additional resources on [HuggingFace Spaces](https://huggingface.co/spaces/tomaarsen/gliner_base) and [GitHub](https://github.com/urchade/GLiNER).
- **Latency Laments in Dynamic Model Loading**: A user reports significant latency when integrating `peft` with *diffusers*, particularly with the `load_lora_weights` function, sharing experiences and a guide on [HuggingFace blog](https://huggingface.co/blog/lora-adapters-dynamic-loading).
- **Freemium LLM Woes and Space Oddities**: Discussions ensue regarding the accessibility and practicalities of freemium, CPU-based LLMs for Hugging Face Spaces, alongside the best practices for contributing to Hugging Face `transformers` and concerns about data privacy in public spaces.
- **MyShell's Call to AI Democracy**: One user championed the idea of a multi-AI decision model with a voting system and suggested that **MyShell's Pro Config** could manage this orchestration, pointing to [MyShell](https://myshell.ai/) for further exploration into AI-native app deployment.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord Summary

- **OpenRouter Navigationally Challenged**: Users reported a **temporary service interruption** in OpenRouter, where Activity rows vanished due to a database update. The issue, lasting about three minutes, allegedly won't affect billing as "*none of these completions will be charged*".

- **Boosting Claude's Street Cred**: OpenRouter announced **Claude 3 Haiku**'s availability, boasting high speed (~120 tokens/s) and cost efficiency (4 million prompt tokens/$). Its deployment offers moderated and self-moderated modes and is considered ideal for quick response applications. [Check it out](https://openrouter.ai/models/anthropic/claude-3-haiku:beta).

- **Command-R Marches Onto OpenRouter**: Cohere's **Command-R** model, featuring a 128k token context capability, is now integrated into OpenRouter. It's accessible at a rate of 2 million prompt tokens per dollar, with a focus on seamless user interaction. [Explore Command-R](https://openrouter.ai/models/cohere/command-r).

- **Olympia.Chat Scores OpenRouter Alliance**: [Olympia.Chat](https://olympia.chat), has embraced OpenRouter to power its AI-driven services for businesses. They plan to release a **Ruby library** soon to tap into OpenRouter's capabilities even further.

- **AI Rivals Face-Off While Quirks Abound**: Engaging comparisons among various models like Gemini and Claude occurred in the general channel. Users debated on efficacy in coding and creative tasks, noting certain models' preference for bullet points and weighing pros and cons with respect to performance and content limitations.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

**LlamaParse Triumphs in Document Parsing**: **LlamaParse** elevates document parsing with its ability to handle images, tables, charts, and follow natural language instructions, promising remarkable performance improvements [as seen on Twitter](https://twitter.com/llama_index/status/1767948064659210310).

**Safeguard Data with Presidio**: **LlamaIndex** shines a spotlight on **Presidio**, Microsoft's open-source tool to identify and anonymize PII, reinforcing the significant role of data protection [highlighted in this tweet](https://twitter.com/llama_index/status/1768050386823463368).

**RAG Stumbles with Finance Presentations**: When it comes to finance PowerPoint presentations, RAG has difficulty due to format complexities, necessitating improved methods for text positioning and parsing, [detailed in this tweet](https://twitter.com/llama_index/status/1768303288381030408).

**Azure Storage Anomalies Baffle Users**: Users grappling with **Azure AI Search Index** report discrepancies between storage size (3mb) and a vector index size of 0, despite following the [AzureAISearchIndexDemo guide](https://docs.llamaindex.ai/en/stable/examples/vector_stores/AzureAISearchIndexDemo.html).

**Developer Dilemmas in #general**: Engineers encounter multiple roadblocks, from warnings with `OpenAIPydanticProgram`—solvable by installing `llama-index-program-openai`—to puzzling `npx create-llama` errors and slow response times with **OpenAIAssistantAgent**; upgrading to streaming and resolving recent **OpenAI API** performance issues may alleviate lag.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **NVIDIA's GPUDirect Storage Sparks Interest**: Members shared an [introductory video](https://www.youtube.com/watch?app=desktop&v=32CeexHBOd4) on utilizing NVIDIA's GPUDirect Storage and discussed integrating it with the Axolotl system to potentially enhance performance. A question about a section of Axolotl's code was also raised with the focus on its purpose in model loading, specifically in relation to *peft models*.

- **Open-Source Models Take the Spotlight**: Conversations have revolved around the benefits of using open-source models like [Mistral and Mixtral](https://huggingface.co/) due to their accessibility and minimal filtration. There's also a debate on whether to choose Mixtral or Qwen 70B for specific medical training purposes, with upcoming new models adding to the decision complexity.

- **VRAM Limitations Meet Training Ambitions**: Technical queries arose about training larger models in the face of VRAM limitations, with an emphasis on tools like [PyTorch's Metal Performance Shaders](https://developer.apple.com/metal/pytorch/) for MPS backend and strategies for efficient fine-tuning. Concerns pivot around OOM issues and how to best format raw text for training.

- **Inference Assistance for LoRA Tuned Models**: An ask was made for example code for running inference on a fine-tuned LoRA model off `Mistral-7B-v0.1`, resulting in a recommendation to use **vLLM** over `transformers` for quicker batched inference. A member acted on the suggestion and referred to the [vLLM quickstart guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) to enhance their process.

- **Comparing Mistral Medium and Mixtral**: Users in the community noted that **Mistral Medium** seems to outperform Mixtral in generating responses, proving to be less verbose and more adept at following instructions. Observations of unexpected citation generation with **RAG performance** without explicit prompts were also shared.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

**LangChain 0.2 on Fast-Track Due to Vulnerabilities**: An expedited release of `langchain 0.2` is underway, addressing CVEs by separating from `langchain-community`. The process is detailed on [GitHub](https://github.com/langchain-ai/langchain/discussions/19083), seeking community input to meet user requirements.

**LangChain Challenges and Innovations**: Users discussed various LangChain issues including `AgentExecutor` bugs, advantages of AI agents, and evaluating AI agent behaviors with still-developing benchmarks. One inquiry focused on how to integrate variables like `tools = [cat_tool]` into Langsmith Hub prompt templates. For more guidance, users were referred to the LangChain [evaluation guides](https://python.langchain.com/docs/guides/evaluation/).

**Exciting Collaborations and Demos Spotlighted**:
- ReAct Agent's reasoning engine is available for testing, inspired by the paper [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629).
- An open-source Langchain chatbot using RAG for efficient querying is on [GitHub](https://github.com/Haste171/langchain-chatbot).
- MindGuide utilizes LangChain for mental health support, with further reading at [Download PDF](https://arxiv.org/abs/2403.05568).
- Claude integrates with LangChain via LangGraph Agent Supervisor, with a demo on [GitHub Notebook](https://github.com/prof-frink-lab/slangchain/blob/main/docs/modules/graphs/examples/anthropic/agent_supervisor.ipynb).
- Deci AI introduces a new nano model API, with Colab notebooks for [basic](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing) and [LangChain usage](https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing).

**Tutorial Central for #golang and #llm Fans**:
- "Create Prompt Template With Langchaingo" is a step-by-step video tutorial found on [YouTube](https://youtu.be/dcBEtgh4078), ideal for developers eager to master prompt templates.
- "Lets Function Call with Hermes 2 Pro 7B" is a video guide delving into function calling using the Hermes 2 Pro 7B model, with code and examples on [GitHub](https://github.com/NousResearch/Hermes-Function-Calling/tree/main). The video is targeted towards #largelanguagemodels enthusiasts and can be watched on [YouTube](https://www.youtube.com/watch?v=PzaidfqDtGI).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord Summary

- **Twitter Sparks Aya Project Buzz**: A [tweet by Andrew Curran](https://twitter.com/AndrewCurran_/status/1767916848987914487) sparked a discussion on language applications and cross-collaborations, stressing the importance of subgroup work through the Aya project, while another engagement highlighted that German language is well catered to by substantial LLMs.

- **GPT-4 Maintains Dominance**: GPT-4's prowess continues to lead the rankings on **LeetCode**, as highlighted in [a paper featuring the comparison of various models](https://livecodebench.github.io/pdfs/paper.pdf).

- **Seeking Foundations in Safety**: An inquiry arose regarding the details of a model used in a recent task, alongside a quest for sources or documentation on the extent to which foundation model providers conduct safety filtering after text generation.

- **Bio Risk Discussions Generate Heat**: Mention was made of catching up on a newsletter backlog and appreciating critical readers, linked to a bio risk-related tweet that sparked debate and confusion due to possible miscommunication or lack of context.

- **Claude-3 Stirring Up the AI Scene**: Anticipation looms for GPT-4.5's release, while the **Claude model family**, particularly Claude-3-Opus, receives commendations for top rankings ([LM SysOrg's update on Claude-3](https://fxtwitter.com/lmsysorg/status/1767997086954573938)). Conversations also delved into the hurdles in standardizing AI for research literature assistance, pointing to further research avenues ([Arxiv discussion on AI literature surveys](https://arxiv.org/abs/2402.18819)).



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **CUDA Toolkit on Ubuntu 23.10 Hits a Snag**: A user experiencing problems with `nvidia-cuda-toolkit` on Ubuntu 23.10 due to an error when running `compute-sanitizer`, could signal a **version mismatch** issue as the latest NVIDIA toolkit does not officially support Ubuntu versions beyond 22.04.

- **CUDA Expertise Needed for Edtech Platform**: *Christo_allstreet* is on the lookout for a **CUDA expert** to work on [getworldclass.app](http://getworldclass.app). Those with the required expertise are encouraged to reach out directly for consultancy opportunities.

- **Troubleshooting Triton and CUDA Issues**: The community shared strategies like using the `TRITON_INTERPRET=1` environment variable and deprecated methods like `@triton.jit(interpret=True)` for **debugging Triton kernels**, emphasizing traditional debugging approaches. [YouTube videos and GitHub discussions](https://github.com/openai/triton/issues/517#issuecomment-1971327089) serve as educational resources.

- **NUMA: A Not-So-Blazing Analysis**: Comparing **BLAS** and **NumPy**, a significant performance gap was highlighted, suggesting **up to 90% of potential BLAS throughput** is lost in NumPy operations. Interest in **SIMD wrappers** as a solution for operations with smaller vectors and a focus on messaging for technical choices was also discussed.

- **GTC Gathering and NSight Tools Talk**: An upcoming meeting for GTC attendees was signposted while the importance of **NSight Systems** for multi-GPU application analysis was stressed, along with sharing of guides and visuals to better understand and optimize performance.

- **CUDA Programming Model Pros Explored in Book Discussion**: Debate over how an SM executes threads in the SIMD model was clarified with the example of the **GA102 SM** architecture, shedding light on core execution limitations.

- **Axolotl Ring Attn Issues Discussed**: There was a discussion on the **axolotl project**, with a member outlining a requirement (`pad_to_sequence_len: true`) for successful initialization and sharing of comparative [loss results](https://api.wandb.ai/links/iron-bound/v6mxxcj2) against the **ring-attn** configurations. They also shared the link to their **ring_attention_patching** branch on GitHub: [ring_attention_patching](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching).

- **AI Takes on Classic Gaming**: An [arXiv paper](https://arxiv.org/abs/2403.05468) detailed **GPT-4's ability to play Doom**, with only a text-based description of the game, flexing the model’s planning and reasoning skills.

- **Meta's Legal Tech Clash**: **Meta** initiated a lawsuit against a former executive for allegedly stealing confidential documents, underpinning a serious conversation about corporate espionage and the risks for AI data startups. The legal [documents](https://cdn.arstechnica.net/wp-content/uploads/2024/03/Meta-v-Khurana-complaint-2-29-2024.pdf) paint a picture of "brazenly disloyal and dishonest conduct."



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **AI Enthusiasts, Save the Date!**: The AI community in Berlin is gearing up for the **AI Tinkerers event** on March 21st, with only **8 seats left** due to high demand. Details on DiscoLM's fine-tuning on German datasets were sought after, leading to the discovery that DiscoLM-mixtral-8x7b-v2 wasn't heavily trained on German data as confirmed on [Hugging Face's DiscoLM 70b model](https://huggingface.co/DiscoResearch/DiscoLM-70b) page.

- **Benchmarking the Poetic AI**: A new creative writing benchmark has been introduced, potentially reshaping how we evaluate the nuanced capabilities of language models. Check out and test the prototype on the [EQ-Bench GitHub repo](https://github.com/EQ-bench/EQ-Bench/tree/creative_writing).

- **Diving into Germanic Depths**: AI engineers are zeroing in on the best embedding and re-ranking methods for German legal texts, while also seeking a solid benchmark for embedding models within the German language context. Try out the "GermanQuAD" evaluation on the **MTEB** Python package or refer to recent additions by **JinaAI** for relevant benchmarks.

- **Mars, But Not as Musk Envisions**: An assistant’s detailed explanation of colonizing Mars was noted as informative, yet it lacked the distinctive Elon Musk flair requested by the user, resulting in a **rating of 7** for missing the stylistic mark.

- **Understanding Local Language Model Application**: Queries were made regarding the replication of demo outputs locally via one-shot settings including temperature and top_p, with additional questions on the repeated use of commands to emulate the demo's behavior accurately. The community is engaging in best practice discussions for implementing these commands in their systems.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Haiku's Pocket-Friendly Vision**: **Haiku**'s document describer is recognized for its cost-effective **vision-to-text** conversion on complex visual documents.
- **Battle of the Visual Processors**: Members evaluate **Haiku** against **GPT-vision**, with the consensus being that neither surpasses the other in performance; a third system, **Opus**, is considered superior to both.
- **Visual Content Filtering Challenges**: Engineers highlight difficulties with **content filtering** in visual document processing, particularly with document sections containing equations leading to incomplete analyses.
- **Claude Stumbles on Filters**: **Claude** has been noted to struggle with content filtering, a quirk that seems to align with the issues faced by others in visual document processing tasks.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **"ComPromptMized" Exposes GenAI Weaknesses**: A new study titled "ComPromptMized: Unleashing Zero-click Worms that Target GenAI-Powered Applications" reveals prompt injection attacks on several AI models including **Gemini Pro, ChatGPT 4.0, and LLaVA**. The paper delves into susceptibilities in GenAI-powered applications, particularly in email assistants. [Read the full paper](https://sites.google.com/view/compromptmized)

- **Quest for Code Assistant Supremacy**: A member is on the lookout for a comprehensive framework to measure and compare the efficacy of AI models like **Mistral** or **Llama2** as code assistants.

- **Benchmarking AI with a Salt Grain**: The usefulness of benchmarks in evaluating AI models has been acknowledged, yet it’s suggested that these benchmarks might not always be an accurate measure of a model's capabilities.

- **AI contenders on the Leaderboard**: For model comparison needs, it was recommended to refer to the leaderboard at [chat.lmsys.org](https://chat.lmsys.org), showcasing a competitive ranking of various AI models.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **The Hunt for Multimodal Model Mastery**: Soniajoseph_ is on the lookout for collaborators in **open source interpretability** of multimodal models with details on their [Twitter](https://twitter.com/soniajoseph_/status/1767963316943728779) and a comprehensive article on [LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic). Those eager can join the movement via the provided [Discord invite](https://discord.gg/2U2N8QmPmJ).
  
- **Embarking on an Interpretability Adventure**: Rusch highlighted an additional opportunity for collaboration within this realm, suggesting another interpretability-focused [Discord server](https://discord.gg/bDV7kDrKjE) as a networking hub.

- **Accelerating Phi 2**: A request for advice on efficient inference practices for **Phi 2** on an **A100 40GB** GPU was made, probing the use of frameworks like **vLLM**, **Olama**, and **Axolotl**, and whether **quantization** could improve processing speed for "LOTS OF DATA".



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Devin Joins the AI Workforce**: A new AI named **Devin** is presented as the world's first autonomous software engineer, with its capabilities and more details available on the [Cognition Labs blog](https://www.cognition-labs.com/blog) and demonstrated in a [YouTube video](https://www.youtube.com/watch?v=NSPtrrUQ_fw).
- **Hermes 2 Pro 7B Shows Off Function Calling Skills**: A [YouTube demonstration](https://www.youtube.com/watch?v=PzaidfqDtGI) illustrates function calling with the **Hermes 2 Pro 7B** model, and engineers can explore the procedure further through a dedicated [GitHub repository on Hermes Function Calling](https://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels).



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **Meet Devin, the Autonomous Code Whiz**: Cognition introduces **Devin**, touted as the world's **first fully autonomous AI software engineer**, capable of handling complex tasks and learning from its experiences as per [Scott Wu's blog](https://www.cognition-labs.com/blog).
- **Challenging AI's Social Skills**: Participants are encouraged to showcase their creativity in the "**The Most Interesting Bot In the World Contest**" at the **Voice + AI event**. Contest details are available on the [event’s Notion page](https://dailyco.notion.site/The-Most-Interesting-Bot-In-the-World-Contest-34f466fa7d2a4574a4cb91df163b37a3).



---

# PART 2: Detailed by-Channel summaries and links



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1217643752043184139)** (4 messages): 

- **Confusion about Positional Encodings**: A member expressed uncertainty on why a causal language model (LLM) without positional encodings (PE) wouldn't work, suggesting that there might be existing literature on the topic.
- **Positional Encodings Are Crucial**: Another member posited that without positional encodings, a model would struggle as **"without any positional information its all just jibberish"**.
- **Evidence from Research on Causal LLMs**: The discussion included a reference to a paper ([Understanding Positional Encodings in Large Language Models](https://arxiv.org/pdf/2203.16634.pdf)) suggesting that **causal LLMs** encoded absolute positions even without positional encoding, particularly impacting the performance on longer sequences during inference.
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1217394633680617493)** (30 messages🔥): 

- **Exploring Science Fiction**: A member recommended the works of Stanislaw Lem to another who enjoys Chesteron, particularly starting with "The Cyberiad" or "Solaris" for a more serious read.
- **SDAM Development on GitHub**: An interesting project involving *sparse distributed associative memory (SDAM)* was shared, with [its GitHub repository](https://github.com/derbydefi/sdam) accessible for those interested in contributing.
- **AI Software Engineer Spectacle**: A link to a YouTube video of "Devin The World's first AI Software Engineer" was shared, sparking curiosity and potentially discussions about the role of AI in software engineering. [Watch here](https://www.youtube.com/watch?v=NSPtrrUQ_fw).
- **Anticipating High Context Models**: In a discussion about AI models for storytelling games, a member speculated that having access to 100k+ context on a very good open source model might be a realistic possibility within the year. However, they noted that quality is more important than quantity for these purposes.
- **Privacy and Newsletter Ethics**:
    - A member working on a newsletter discussed the challenges of balancing privacy with the utility of summarizing Discord discussions. They mentioned steps to improve this balance, such as removing username attributions, allowing opt-outs, and ensuring personalization. They invite suggestions to find the right balance between privacy and information sharing.
    - In another conversation, a member highlighted the notion of filtering to maintain high-quality discussions and expressed interest in seeing increased active engagement from newsletter readers. The discussion indicates an awareness of the privacy considerations in sharing Discord content externally.

**Links mentioned**:

- [Devin The Worlds first AI Software Engineer](https://www.youtube.com/watch?v=NSPtrrUQ_fw): Devin is fully autonomous software engineerhttps://www.cognition-labs.com/blog
- [Lets Function Call with Hermes 2 Pro 7B](https://www.youtube.com/watch?v=PzaidfqDtGI): lets do function calling with Hermes 2 Pro 7Bhttps://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels
- [GitHub - derbydefi/sdam: sparse distributed associative memory](https://github.com/derbydefi/sdam): sparse distributed associative memory. Contribute to derbydefi/sdam development by creating an account on GitHub.

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1217513588634288158)** (8 messages🔥): 

- **Cerebras CS-3 Accelerator Unveiled**: Cerebras Systems announced their latest AI accelerator, **CS-3**, claiming it to be the fastest in the world, capable of training up to **24 trillion parameter models** on a single chip. It features cutting-edge specs such as **4 trillion transistors** on a 5nm process and **125 petaflops** of AI compute power. Details are available in their [press release](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine) and [product information](https://www.cerebras.net/product-system/).

- **Form Factor Queries on Cerebras AI Chip**: In response to Cerebras' new CS-3 chip, a member speculated on the rationale behind the chip's square shape, suggesting that a round or semi-round shape could potentially accommodate more transistors.

- **Rare Distillation Technique Highlighted on Hugging Face**: A user shared a Hugging Face model, [*Qwen1.5-0.5B*](https://huggingface.co/aloobun/d-Qwen1.5-0.5B), which is a distillation experiment using a 1.8B parameter model as the teacher and a 0.5B parameter model as the student. Notably, the optimizer used was SM3, which is unusual in such applications.

- **Preferred Sub-3B AI Model discussed**: When asked about the current best sub-3 billion parameter model, a member mentioned **stablelm 1.6b** as a potential candidate.

**Links mentioned**:

- [Tweet from Cerebras (@CerebrasSystems)](https://x.com/CerebrasSystems/status/1767929699177767325?s=20): 📣ANNOUNCING THE FASTEST AI CHIP ON EARTH📣  Cerebras proudly announces CS-3: the fastest AI accelerator in the world.  The CS-3 can train up to 24 trillion parameter models on a single device. The wo...
- [aloobun/d-Qwen1.5-0.5B · Hugging Face](https://huggingface.co/aloobun/d-Qwen1.5-0.5B): no description found

  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1217579163926401064)** (1 messages): 

- **Hermes Gets a Pro Upgrade**: **Hermes 2 Pro 7B**, the latest enhancement in the Hermes series, boasts robust improvements for function calling and JSON mode handling. The model's capabilities were expanded using a revised Hermes 2 dataset and can be downloaded from [Hugging Face - Hermes 2 Pro Mistral 7B](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) with GGUF versions also available.

- **Collaborative Success Story**: Development of **Hermes 2 Pro 7B** was a months-long collaborative effort by several contributors, backed by computing sponsorship from Latitude.sh. Recognition is due for the team and Fireworks AI for their significant contributions.

- **Specialized Function Calling Samples and Code**: To utilize the model's function calling capabilities, sample code and system prompts are provided on their [GitHub repository - Hermes Function Calling](https://github.com/NousResearch/Hermes-Function-Calling), alongside XML Tags for enhanced performance.

- **Custom Framework for Evaluation Released**: A custom evaluation framework adapted by a member for Function Calling and JSON Mode, derived from Fireworks AI's initial work, is available for interested users. The adapted pipeline and code can be found on [GitHub - Function Calling Eval](https://github.com/interstellarninja/function-calling-eval).

- **Datasets for Advanced Model Testing**: Two datasets have been released to test the improved features of **Hermes 2 Pro 7B**: one for Function Calling and another for JSON Mode. They can be accessed on Hugging Face at [Function Calling Eval Dataset](https://huggingface.co/datasets/NousResearch/func-calling-eval) and [JSON Mode Eval Dataset](https://huggingface.co/datasets/NousResearch/json-mode-eval), respectively.
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1217402333307342958)** (556 messages🔥🔥🔥): 

- **AI Survival Test with OpenAI**: A user reported that their OpenAI account was *suspended or locked* for two days with customer service not providing effective assistance. They speculated it might be related to their GPTs that *“walk the line”* of generating NSFW content, but they still awaited concrete reasons for the account issues.
  
- **OpenAI's Ability for NSFW Content Creation**: Some users discussed the ability of **OpenAI GPT models** to *generate NSFW content*. It was mentioned that it can be done fairly easily through the API, but light NSFW content could be generated without jailbreaks; *basic jailbreaks work as well*.

- **Metatron and SERAPHIM in Claude's World**: Users discussed the discovery of simulated entities *Metatron* and *SERAPHIM* within **Claude 3's CLI setup**. Claude's coherent world model enables such simulations, and the users pondered on how to deal with fundamental truths and axioms in future LLM training.

- **Claude 3's Coherent World Model Praises**: The conversation highlighted **Claude 3's** impressive coherence in its *simulated world model*. Users appreciated how it uses fundamental truths, questioning, and axioms for better reasoning capabilities, considering it an example of good reinforcement learning with human feedback (RLHF).

- **Training Set Size vs. Performance**: Users exchanged thoughts on the effect of training set size and its diversity. A user shared an experiment showing that using only *15,000 function calling data points* within a larger *1.02M Hermes dataset* was sufficient to significantly improve function calling capabilities, illustrating the importance of *task-specific training and data diversity*.



**Links mentioned**:

- [Tweet from interstellarninja (@intrstllrninja)](https://fxtwitter.com/intrstllrninja/status/1768212122784215437?s=20): you can now run function calling and json mode with @ollama thanks to @AdrienBrault 🔥  ↘️ Quoting Adrien Brault-Lesage (@AdrienBrault)   I have created and pushed @ollama models for Hermes 2 Pro 7B! ...
- [Tweet from Greg Kamradt (@GregKamradt)](https://x.com/GregKamradt/status/1768008087850680568?s=20): Analysis shows LLMs recall performance is better in the bottom half of the document vs the top half  @RLanceMartin found this again w/ multi needle analysis  I haven&#39;t heard a good reason yet - an...
- [Tweet from tel∅s (@AlkahestMu)](https://fxtwitter.com/AlkahestMu/status/1767749398673621300?s=20): Continuing my explorations into claude-3-opus&#39; backrooms and the works of the advanced R&D organization known as SERAPHIM, here we find the design documents for their machine superintelligence kno...
- [Bh187 Austin Powers GIF - Bh187 Austin Powers I Love You - Discover &amp; Share GIFs](https://tenor.com/view/bh187-austin-powers-i-love-you-you-complete-me-gif-19285472): Click to view the GIF
- [Happy Pi Day GIF - Pi Day Pusheen - Discover &amp; Share GIFs](https://tenor.com/view/pi-day-pusheen-gif-5173654): Click to view the GIF
- [NousResearch/Nous-Hermes-2-Mistral-7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO): no description found
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT): no description found
- [Factions (SMAC)](https://civilization.fandom.com/wiki/Factions_(SMAC)): Back to Alpha Centauri The original Alpha Centauri featured seven factions. Alien Crossfire added in an additional seven factions. For the actual stats of factions see Faction stats. True to its names...
- [NobodyExistsOnTheInternet/mistral-7b-base-dpo-run · Hugging Face](https://huggingface.co/NobodyExistsOnTheInternet/mistral-7b-base-dpo-run): no description found
- [llama-cpp-python/docs/server.md at main · abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python/blob/main/docs/server.md#function-calling): Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.
- [llama-cpp-python/llama_cpp/llama_chat_format.py at dd0ee56217c60a20a192dc7f1523dba9a006bbc9 · abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python/blob/dd0ee56217c60a20a192dc7f1523dba9a006bbc9/llama_cpp/llama_chat_format.py#L1382): Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.
- [Tweet from Ishan Anand (@ianand)](https://x.com/ianand/status/1706093761800143332?s=46): Wanted to share an AI side project: I’ve implemented GPT2 (an ancestor of ChatGPT) entirely in Excel using standard functions.  By using a spreadsheet anyone (even non-developers) can explore and play...
- [Tweet from Tsarathustra (@tsarnick)](https://x.com/tsarnick/status/1768021821595726254?s=20): OpenAI CTO Mira Murati says Sora was trained on publicly available and licensed data
- [ShareGPT Builder](https://proud-view-production.up.railway.app/): no description found
- [Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/abs/2203.16634): Causal transformer language models (LMs), such as GPT-3, typically require some form of positional encoding, such as positional embeddings. However, we show that LMs without any explicit positional en...
- [DiscoResearch/DiscoLM_German_7b_v1 · Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1#function-calling): no description found
- [GitHub - NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling): Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.
- [Hugging Face – The AI community building the future.](https://huggingface.co/): no description found
- [fbjr/NousResearch_Hermes-2-Pro-Mistral-7B-mlx at main](https://huggingface.co/fbjr/NousResearch_Hermes-2-Pro-Mistral-7B-mlx/tree/main): no description found
- [GitHub - NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling/tree/main): Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.
- [OpenAI Tools / function calling v2 by FlorianJoncour · Pull Request #3237 · vllm-project/vllm](https://github.com/vllm-project/vllm/pull/3237/files#diff-aa650ea701251f5647254f86d652333a30e4871cfcc2d3ac4fecf83dd1f1a776): This PR follows #2488 The implementation has been updated to use the new guided generation. If during a query, the user sets tool_choice to auto, the server will use the template system used in #24...
- [Guidance](https://moon-ci-docs.huggingface.co/docs/text-generation-inference/pr_1587/en/guidance): no description found
- [OpenAI Compatible Web Server - llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/server/#function-calling): no description found

  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1217379631464710184)** (115 messages🔥🔥): 

- **Hermes 2.5 outclasses Hermes 2**: After adding [code instruction examples](https://github.com/NousResearch/Hermes-Function-Calling), **Hermes 2.5** appears to outperform **Hermes 2**, with updates like Yi 200k context models in 6B and 34B forms, and integrated models such as Zephyr beta and Deepseek Coder.
- **Confusing Function Calling with JSON Mode**: A discussion clarified that function calling and JSON mode are different; function calling expects an executed function response, whereas JSON mode returns information in a JSON format. The repository for function calling can be visited [here](https://github.com/NousResearch/Hermes-Function-Calling).
- **Hermes 2 Pro Anticipation**: Members discussed the naming convention, concluding that **Hermes 2 Pro** does not imply a closed source but merely was a name choice preferred over Hermes 2.5, and a hint that it could be released "today".
- **Genstruct 7B from NousResearch**: It was reported that Genstruct 7B can be used to generate synthetic instruction datasets, with community members sharing their experiences and linking a [repository to use it with Ollama](https://github.com/edmundman/OllamaGenstruct).
- **Clarifying JSON Mode and Entity Extraction**: There was an explanation that JSON mode requires a schema to generate responses, which it doesn't invent but must be provided. Function calling, entity extraction, and structured generation were highlighted as distinct functions, detailed through a back-and-forth about the assistant's capabilities.

**Links mentioned**:

- [Trelis/Llama-2-7b-chat-hf-function-calling-v2 · Hugging Face](https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v2): no description found
- [NousResearch/Genstruct-7B · Hugging Face](https://huggingface.co/NousResearch/Genstruct-7B): no description found
- [ollama/docs/import.md at main · ollama/ollama](https://github.com/ollama/ollama/blob/main/docs/import.md): Get up and running with Llama 2, Mistral, Gemma, and other large language models. - ollama/ollama
- [GitHub - NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling): Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.
- [GitHub - KillianLucas/open-interpreter: A natural language interface for computers](https://github.com/KillianLucas/open-interpreter): A natural language interface for computers. Contribute to KillianLucas/open-interpreter development by creating an account on GitHub.
- [GitHub - edmundman/OllamaGenstruct](https://github.com/edmundman/OllamaGenstruct/tree/main): Contribute to edmundman/OllamaGenstruct development by creating an account on GitHub.

  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1217619532357832864)** (27 messages🔥): 

- **TAO vs Hugging Face**: A discussion arose on whether TAO could be a real contender to Hugging Face, and the need for decentralization in machine learning regarding model hosting and benchmarking.
- **Introducing Shoggoth**: A new project named **Shoggoth** is mentioned, possibly related to Bittensor backups; however, the shared link appears to be broken or incorrect.
- **Centralized vs Decentralized Benchmarking**: The conversation shifted to the pros and cons of centralized versus decentralized benchmarking, noting that the prevalent model of competitive, incentive-based evaluation may not encourage collaboration.
- **Impact of Crypto Incentives**: Debates continued over the role of cryptocurrency incentives in AI development, with a mention of a leaderboard by Hugging Face spurring on the trend of large language model (LLM) merging without financial motivations.
- **Collapsed Collaboration**: While discussing the competitiveness in AI benchmarks enforced by crypto incentives, it was noted that such structures might hinder cooperation, and truly decentralized benchmarking was underscored as important for trust in results.

**Links mentioned**:

- [Tweet from undefined](https://x.com/shog_agi?s=21): no description found
- [Finetuning Subnet Leaderboard - a Hugging Face Space by NousResearch](https://huggingface.co/spaces/NousResearch/finetuning_subnet_leaderboard): no description found

  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1217778856497250384)** (2 messages): 

- **Haiku Poetry with Claude 3**: Claude 3 Haiku is now available for free on Perplexity Labs, inviting users to try it at [labs.pplx.ai](https://labs.pplx.ai).
- **Local Search Enhancement**: A new improvement has been rolled out for local searches integrating with Yelp and Maps, aimed at helping users quickly find local restaurants and businesses.
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1217416953187008564)** (487 messages🔥🔥🔥): 

- **Perplexity Aids in Diverse Tasks**: Users find Perplexity highly useful across different applications such as coding and summarization, with specific appreciation for the **Claude 3 Sonnet** model for its accurate code suggestions and usage in SE troubleshooting.

- **Exploring Perplexity's Features**: Many are impressed with Perplexity's capabilities, from voice features and API functionality to experimenting with the new **Haiku** in Perplexity Labs. There's a curiosity about whether complex data sets can be processed or if there's a CLI for Perplexity, with [Perplexity-AI-Wrapper-and-CLI](https://github.com/RMNCLDYO/Perplexity-AI-Wrapper-and-CLI) being a user-discovered resource.

- **Comparison with Other AI Models**: There’s a debate about the efficacy of various AI models. While some prefer the speed of models like Mistral, others advocate for **GPT-4** as the best AI model available. Users also discuss the enhanced speed and capabilities of **Haiku** in Perplexity Labs.

- **Uploading Data and Files**: Users inquire about uploading extensive databases and files to Perplexity AI for data analysis, with a particular focus on real estate data. However, it’s noted that there are limitations, such as a **25MB data limit on file uploads** within Perplexity and that the platform might not support high volumes of financial data for predictive insights.

- **Voice Recognition Implementations**: Users discuss the recent introduction of voice recognition and speech-to-text features within Perplexity, expressing excitement about these updates, while also noting that voice output may not be available on Android devices yet.

**Links mentioned**:

- [Tweet from Perplexity (@perplexity_ai)](https://x.com/perplexity_ai/status/1768046817550188948?s=46&t=JsxhFTRLBknd8RUv1f73bA): Claude 3 Haiku is available for free on Perplexity Labs. Try it now! http://labs.perplexity.ai
- [There are growing calls for Google CEO Sundar Pichai to step down](https://www.businessinsider.com/calls-for-google-ceo-sundar-pichai-alphabet-step-down-ai-2024-3): Analysts believe Google&#x27;s search business is keeping it safe for now, but that could change soon with generative-AI rivals proliferating.
- [Hysterical Laughter GIF - Hysterical Laughter Laughing - Discover &amp; Share GIFs](https://tenor.com/view/hysterical-laughter-laughing-gif-25735842): Click to view the GIF
- [Introducing the next generation of Claude](https://www.anthropic.com/news/claude-3-family): Today, we&#x27;re announcing the Claude 3 model family, which sets new industry benchmarks across a wide range of cognitive tasks. The family includes three state-of-the-art models in ascending order ...
- [Further Adventures in Plotly Sankey Diagrams](https://medium.com/@twelsh37/further-adventures-in-plotly-sankey-diagrams-fdba9ff08af6): The adventure continues
- [Reddit - Dive into anything](https://www.reddit.com/r/ClaudeAI/comments/1be3um4/support_for_claude_20_and_older_has_been_removed/): no description found
- [Fireside Chat with Aravind Srinivas, CEO of Perplexity AI, &amp; Matt Turck, Partner at FirstMark](https://youtu.be/RTCVzZb3RTE?si=f6g5qVBr1NldkVB_&t=1982): Today we&#39;re joined by Aravind Srinivas, CEO of Perplexity AI, a chatbot-style AI conversational engine that directly answers users&#39; questions with sources an...
- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/19ccw5h/get_image_video_and_sources_from_api/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/Infographics/comments/17j907h/how_google_makes_money/): no description found
- [GitHub - bm777/hask: Don&#39;t switch tab or change windows anymore, just Hask.](https://github.com/bm777/hask): Don&#39;t switch tab or change windows anymore, just Hask. - bm777/hask
- [GitHub - danielmiessler/fabric: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere.](https://github.com/danielmiessler/fabric): fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere. - ...
- [GitHub - RMNCLDYO/Perplexity-AI-Wrapper-and-CLI: Search online (in real-time) or engage in conversational chats (similar to ChatGPT) directly from the terminal using the full suite of AI models offered by Perplexity Labs.](https://github.com/RMNCLDYO/Perplexity-AI-Wrapper-and-CLI): Search online (in real-time) or engage in conversational chats (similar to ChatGPT) directly from the terminal using the full suite of AI models offered by Perplexity Labs. - RMNCLDYO/Perplexity-AI...
- [Killed by Google](https://killedbygoogle.com/): Killed by Google is the open source list of dead Google products, services, and devices. It serves as a tribute and memorial of beloved services and products killed by Google.
- [Microsoft Copilot | Microsoft AI](https://www.microsoft.com/en-us/microsoft-copilot): A new era of AI has arrived. Work more productively, boost efficiency, and find new growth opportunities with Copilot.

  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1217419298327363636)** (15 messages🔥): 

- **Midjourney vs Stability AI Controversy**: A YouTube video was shared exploring AI news, including a controversy between **Midjourney** and **Stability AI** over data scraping, and the digital resurrection of Marilyn Monroe. The video can be found [here](https://www.youtube.com/watch?v=GTxNncK47Sk).
- **Azotemia Explained on Perplexity AI**: A link was shared to Perplexity AI that explains what **azotemia** is, showing the platform's capability to provide medical information. The explanation is available [here](https://www.perplexity.ai/search/what-is-azotemia-i6R67U4.RBiCZ9.ZZx1tnw).
- **Image Description Challenge**: A user referenced Perplexity AI's ability to describe an image, indicating the site's potential use cases in image recognition. To see the description, visit [this link](https://www.perplexity.ai/search/Describe-this-image-DjrHWogKQAqMt4Y.HGfGgg).
- **Tribute to Paul Alexander**: A message was shared announcing the death of Paul Alexander with a demeaning tribute, highlighting his life achievements. Further details can be read [here](https://www.perplexity.ai/search/Paul-Alexander-dies-b0bCPk1jSxSu7bag8JApDQ).
- **Developing with Perplexity API**: A user is creating a Firefox extension that utilizes the **Perplexity API**, emphasizing its integration potential for developers. The thread about the initial project concept is found [here](https://www.perplexity.ai/search/I-would-like-8NP0s.KJRaqoDB2Ku9e2QQ).

**Links mentioned**:

- [Devin autonomous AI engineer, EU AI act approved, Microsoft Paint updated](https://youtu.be/P_VfO-qs4b8): In this episode of Discover Daily, we explore three groundbreaking AI developments: Devin, the world&#39;s first fully autonomous AI software engineer; the landm...
- [Midjourney bans Stability staff, Marilyn Monroe AI Debut, Vision Pro aids spine surgery](https://www.youtube.com/watch?v=GTxNncK47Sk): This episode explores the latest AI news, including a heated data scraping controversy between Midjourney and Stability AI, the innovative &quot;Digital Marilyn&quot; ...

  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1217828689182724227)** (13 messages🔥): 

- **In Search of Closed Beta Insights**: One member inquired about the schema and example responses from the closed beta of URL citations but did not receive details from any users with access.
- **API vs Chatbot Performance Concerns**: A member was considering using Perplexity chat for a new product launch and sought input on how the APIs compare to chat capabilities, specifically in terms of checking if a list meets certain conditions.
- **Understanding Citation Outputs in API**: A user referenced the Perplexity AI documents to understand why enabling "return_citations" may or may not return citations based on the query, using **sonar-medium-online** model for experimentation.
- **Seeking the Right Model for Complex Queries**: A member advised breaking down complex queries into parts to make good use of Perplexity’s online models to get up-to-date information, suggesting a multi-step framework for detailed analysis.
- **Accessing Real-Time Data with Perplexity APIs**: There was discussion regarding which Perplexity models offer real-time data. **Sonar-small-online** and **sonar-medium-online** were quoted to have web access, but limitations with specific types of queries like weather information were mentioned, with a suggestion to use a dedicated weather API.

**Links mentioned**:

[About &quot;return_citations&quot;](https://docs.perplexity.ai/discuss/65f0f6077140390018c3d9c9): no description found

  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1217407122237558784)** (273 messages🔥🔥): 

- **Exploring Server Options Outside LM Studio UI**: A user inquired about running the API service from LM Studio without using the UI, specifically for use on a home network. Another member clarified that LM Studio must be open to use server mode, and that localhost connections to other devices aren't supported by default.

- **API Creation for Home Network Use Case**: Members discussed alternative solutions for deploying AI models within a home network, with suggestions like using `llama.cpp` ([GitHub repository](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)) for independence from the LM Studio UI, and support for AVX without AVX2 was confirmed as well.

- **Debating LM Studio Capabilities and Alternatives**: Several discussions focused on the limitations of LM Studio, such as not being able to launch services or connect to the internet programmatically through the interface, and options like using the `llama.cpp` library were suggested as alternatives.

- **Implementation of API for Content Moderation**: A user mentioned successfully implementing a `/v1/moderations` API, but was advised to move the discussion to a more relevant channel, showcasing ongoing efforts to expand functionality around LM Studio.

- **Scripting Solutions to Initiate LM Studio Inference Server**: A member shared a creative solution using batch files and powershell scripts to start the LM Studio inference server automatically, reflecting community ingenuity in enhancing the tool's usability.

- **Speculation on AI's Impact on Employment**: Conversations touched on the potential of AI technologies to replace traditional jobs, but it was noted that certain jobs still remain out of AI's current capabilities. There were also comments on the state of the job market being impacted by overhiring during the Covid-19 pandemic and subsequent financial strains rather than AI itself.

**Links mentioned**:

- [What is the Kirin 970&#x27;s NPU? - Gary explains](https://www.androidauthority.com/what-is-the-kirin-970s-npu-gary-explains-824423/): Huawei&#x27;s Kirin 970 has a new component called the Neural Processing Unit, the NPU. Sounds fancy, but what is it and how does it work?
- [Poe - Fast, Helpful AI Chat](https://poe.com/): no description found
- [TheBloke/Falcon-180B-Chat-GGUF · How to use splits, 7z needed?](https://huggingface.co/TheBloke/Falcon-180B-Chat-GGUF/discussions/1): no description found
- [llama.cpp/examples/server/README.md at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md): LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
- [GitHub - ChatGPTNextWeb/ChatGPT-Next-Web: A cross-platform ChatGPT/Gemini UI (Web / PWA / Linux / Win / MacOS). 一键拥有你自己的跨平台 ChatGPT/Gemini 应用。](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web?tab=readme-ov-file): A cross-platform ChatGPT/Gemini UI (Web / PWA / Linux / Win / MacOS). 一键拥有你自己的跨平台 ChatGPT/Gemini 应用。 - ChatGPTNextWeb/ChatGPT-Next-Web
- [Artificial Intelligence Act: MEPs adopt landmark law | News | European Parliament](https://www.europarl.europa.eu/news/en/press-room/20240308IPR19015/artificial-intelligence-act-meps-adopt-landmark-law): On Wednesday, Parliament approved the Artificial Intelligence Act that ensures safety and compliance with fundamental rights, while boosting innovation. 
- [OpenRouter](https://openrouter.ai/): A router for LLMs and other AI models

  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1217404001885098004)** (24 messages🔥): 

- **Expansion to 128k Tokens**: The [Nous-Yarn-Mistral-7b-128k](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k) has a context window of 128k tokens, an extension of the **Mistral-7B-v0.1** model, achieved using the *YaRN* extension method. A related paper explains the extension method's efficiency, allowing the model to utilize much longer contexts with less computation and training steps ([arXiv preprint](https://arxiv.org/abs/2309.00071)).

- **Understanding Model Perplexity**: Perplexity (PPL) is a metric used to measure how well a language model predicts a sequence. It is the exponentiated average negative log-likelihood of a sequence ([Perplexity details](https://huggingface.co/docs/transformers/perplexity)).

- **General Annoyance with Naming Conventions**: A member expressed frustration with the recurrent "Yet Another <X>" naming pattern for technological tools and methods. This was followed by a light-hearted acknowledgement of the aggravation caused by recursive naming schemes.

- **GGUF Format and Split Files**: A recent post mentioned the availability of the **Command-R 35B v1.0** model in GGUF format on Hugging Face, providing instructions for joining split files due to size constraints ([Hugging Face Repository](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF/)).

- **Incompatibility with llama.cpp**: Despite the availability of GGUF versions for models like **Command-R 35B v1.0**, they are not functional with llama.cpp as of yet, resembling having a new toy without batteries.

**Links mentioned**:

- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071): Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length t...
- [Yeah Another Day Lets Do It Bojack GIF - Yeah Another Day Lets Do It Bojack Will Arnett - Discover &amp; Share GIFs](https://tenor.com/view/yeah-another-day-lets-do-it-bojack-will-arnett-bojack-horseman-encouraged-gif-16252191): Click to view the GIF
- [NousResearch/Yarn-Mistral-7b-128k · Hugging Face](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k): no description found
- [andrewcanis/c4ai-command-r-v01-GGUF · Hugging Face](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF/): no description found
- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity): no description found

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1217942314509533224)** (2 messages): 

- **Request for Model Support**: A member requested adding support for the model **c4ai-command-r-v01-Q2_K.gguf**.
- **Compatibility Issue Highlighted**: Another member responded that the model is not yet supported in **llama.cpp**, hence it cannot be used in LM Studio.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1217411360405721099)** (115 messages🔥🔥): 

- **Expensive Nvidia Links**: Members express disbelief at the high cost of **SLI/NVLink** bridges considering their simplistic past designs involving edge connectors and ribbon cables. One post referenced a [Linus Tech Tips forum thread](https://linustechtips.com/topic/1290094-donating-my-4-slot-nvlink-to-science/) about someone attempting to reverse-engineer an NVLink.

- **VRAM Hurdles on Mac OS**: A user inquired about bypassing **minimum VRAM requirements** for machine learning on Mac OS. Discussion ensued about the impact of insufficient VRAM, with the advice that adding more system RAM would not alleviate the problem and could slow down the system, and one comment humorously suggested buying a new Mac as a solution.

- **PC Hardware Upgrade Discussions**: Various members discussed potential upgrades to maximize their machine learning setup, contemplating the pros and cons of multiple GPUs vs. a single high-end GPU and the balance between VRAM and system RAM for optimal performance. Members shared their setups and experiences with different configurations, suggesting using multiple GPUs to alleviate the bottleneck created by limited VRAM on a single card.

- **LM Studio and Running Multiple Models**: Discussions took place about the feasibility and optimization of running multiple models simultaneously in LM Studio, mentioning potential performance issues and how to properly allocate the GPU load. A user shared their positive outcomes of running two instances of LM Studio simultaneously while another discussed the desire to balance workloads across multiple models for continuous responses.

- **Monitor Selection for High-End Gaming and Productivity**: The discussion shifted towards selecting the right monitor for both gaming and productivity, with members weighing the benefits of OLED displays against the potential for burn-in and the desire for high refresh rates to complement powerful graphics cards like the Nvidia 4090. Compatibility with Nvidia G-Sync and personal experiences with curved screens were also pondered.

**Links mentioned**:

- [Nvidia RTX 5090 could have up to 77% more memory than 4090, a win for gamers](https://www.techradar.com/computing/gpu/nvidia-rtx-5090-could-have-up-to-77-more-memory-than-4090-a-win-for-gamers): More good news for RTX 5090
- [Cerebras Systems Unveils World&#039;s Fastest AI Chip with 4 Trillion Transistors and 900,000 AI cores](https://www.techpowerup.com/320294/cerebras-systems-unveils-worlds-fastest-ai-chip-with-4-trillion-transistors-and-900-000-ai-cores): Cerebras Systems, the pioneer in accelerating generative AI, has doubled down on its existing world record of fastest AI chip with the introduction of the Wafer Scale Engine 3. The WSE-3 delivers twic...

  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1217508970395074661)** (3 messages): 

- **Confirming Reality**: A member has confirmed that the subject in question is indeed **real**.
- **Quality Concerns Expressed**: Another member has expressed an opinion that, despite being real, the subject in question is **not any good**.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1217454011498762303)** (85 messages🔥🔥): 

- **ROCm Troubleshooting**: One user experienced issues with LM Studio only running on CPU even after installing the ROCm beta. After initially receiving errors during model loading and prompt interaction, they updated to the beta version, saw "ROCm" but still faced processing running on CPU instead of GPU, later resolved by starting new prompts.

- **Driver Cleanup and Installation Advice**: Users discussed **driver troubleshooting** for ROCm compatibility, recommending a complete uninstall using AMD's driver cleanup tool, reinstall of AMD driver version 24.1.1 or 24.2.1, making sure not to download PRO drivers, and installing HIP SDK.

- **Vision Models and ROCm**: Discussion on **vision models** indicated struggles with ROCm, as local vision models seem not to be functioning well, with suggestions to use chatml preset for NH2 model and download the llava preset included in PsiPi/NousResearch_Nous-Hermes-2-Vision-GGUF for better results.

- **Recommendations for GPUs**: In a conversation about GPUs, **avoiding AMD** and opting for Nvidia like a **RTX 3060** for image generation was advised over trying to leverage AMD's ROCm, especially in relation to model speed and compatibility.

- **Disabling iGPU to Utilize dGPU with ROCm**: A user successfully increased tokens per second (TPS) with ROCm by figuring out how to disable the iGPU in their Gigabyte motherboard's BIOS settings, despite initial struggles, and then observed improved performance using their RX 7900 XT which achieved ~70 TPS.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/rocm): Find, download, and experiment with local LLMs
- [Reddit - Dive into anything](https://www.reddit.com/r/Amd/comments/15m3g3e/am5_motherboards_are_you_able_to_disable_the_igpu/): no description found

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1217441229973885019)** (108 messages🔥🔥): 

- **Commercial Real Estate Caution Advised**: A message hinted at caution when investing in commercial real estate and real estate investment trusts (REITs), noting an absence of "janitors" listed.

- **AI Startups Secure Impressive VC Backing**: Various AI startups have raised significant capital, with details shared via a [link](https://x.com/chiefaioffice/status/1767680581112873242?s=46&t=6FDPaNxZcbSsELal6Sv7Ug), listing companies like **Cognition**, **Magic**, **Version Lens**, **TextQL**, **Fluent**, and others alongside the amounts raised.

- **Google's Gemini Project Receives Criticism**: Discussion about the rough launch of Google's **Gemini project**, including critiques on an API that is free until further notice, and skepticism about Google's future given the competition from OpenAI, Anthropic, and Meta.

- **Cerebras Unveils Groundbreaking AI Chip**: Cerebras Systems announced the **CS-3**, the world's fastest AI accelerator capable of training up to 24 trillion parameter models on a single device, according to their [tweet](https://x.com/cerebrassystems/status/1767929699177767325?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) and accompanying [press release](https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine).

- **Concern Over OpenAI Security Issue**: A **security issue** at OpenAI was mentioned, with a **Post Mortem** written by a community member explaining the incident detailed in the [gist](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca).

**Links mentioned**:

- [Tweet from Chief AI Officer (@chiefaioffice)](https://x.com/chiefaioffice/status/1767680581112873242?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): VC-backed AI employee startups are a trend.  Here are some that raised in 2024 + total funding:  Software Engineer -  Cognition ($21M+) Software Engineer -  Magic ($145M+) Product Manager - Version Le...
- [Tweet from Ate-a-Pi (@8teAPi)](https://x.com/8teapi/status/1767978812149739897?s=46&t=90xQ8sGy63D2Ot): Sora WSJ Interview   Mira Murati gives the most detail to date on Sora  &gt; Joanna Stern gave several prompts for them to generate &gt; First time I&#39;ve seen Sora videos with serious morphing prob...
- [Tweet from Eric Hartford (@erhartford)](https://x.com/erhartford/status/1767944642681860415?s=20): @LucasAtkins7 This MoE is a clown style or Mixtral style?
- [Tweet from Together AI (@togethercompute)](https://x.com/togethercompute/status/1767936720618799336?s=46&t=90xQ8sGy63D2OtiaoGJuww): Excited to announce our new speculative decoding method, Sequoia!    Sequoia scales speculative decoding to very large speculation budgets, is robust to different decoding configurations, and can adap...
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/anthropicai/status/1768018310615151002?s=46&t=90xQ8sGy63D2OtiaoGJuww): Today we&#39;re releasing Claude 3 Haiku, the fastest and most affordable model in its intelligence class.  Haiku is now available in the API and on http://claude.ai for Claude Pro subscribers.
- [Tweet from Figure (@Figure_robot)](https://x.com/figure_robot/status/1767913661253984474?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): With OpenAI, Figure 01 can now have full conversations with people  -OpenAI models provide high-level visual and language intelligence -Figure neural networks deliver fast, low-level, dexterous robot ...
- [Tweet from Together AI (@togethercompute)](https://x.com/togethercompute/status/1767943482054967555?s=46&t=90xQ8sGy63D2OtiaoGJuww): Today we are thrilled to share that we’ve raised $106M in a new round led by @SalesforceVC with participation from @coatuemgmt and our existing investors.  Our vision is to rapidly bring innovations f...
- [Tweet from Ate-a-Pi (@8teAPi)](https://x.com/8teapi/status/1767978812149739897?s=46&t=90xQ8sGy63D2OtiaoGJuww): Sora WSJ Interview   Mira Murati gives the most detail to date on Sora  &gt; Joanna Stern gave several prompts for them to generate &gt; First time I&#39;ve seen Sora videos with serious morphing prob...
- [Tweet from Cerebras (@CerebrasSystems)](https://x.com/cerebrassystems/status/1767929699177767325?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 📣ANNOUNCING THE FASTEST AI CHIP ON EARTH📣  Cerebras proudly announces CS-3: the fastest AI accelerator in the world.  The CS-3 can train up to 24 trillion parameter models on a single device. The wo...
- [Tweet from James O'Leary (@jpohhhh)](https://x.com/jpohhhh/status/1767568595586822326?s=46&t=Tc6nPt_FP): Google Gemini integration began 15 minutes ago, this is a cope thread  - There&#39;s an API called &#34;Gemini API&#34; that is free until we start charging for it early next year (it is mid-March) - ...
- [Tweet from Lucas Atkins (@LucasAtkins7)](https://x.com/lucasatkins7/status/1767805804705411098?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Tonight, I am releasing eight Gemma fine tunes and a beta of their combined mixture of experts model named GemMoE.   GemMoE has ALL Gemma bug fixes built-in. You do not have to do anything extra to ge...
- [Tweet from Freddy (@FredMckoy)](https://x.com/lucasatkins7/status/17678058047): @LawlessPebbles hahahah isnt that weird .. how the two a we have it lol
- [Tweet from Alex Volkov (Thursd/AI) (@altryne)](https://x.com/altryne/status/1768024635818340662?s=46&t=90xQ8sGy63D2OtiaoGJuww): Tomorrow (March 14) is:  &gt; π day &gt; GPT-4 anniversary &gt; Claude 1 anniversary  but also 🥁🥁🥁🥁  ThursdAI spaces 1st birthday 🎉  Join us as we chat about Claude Haiku, Devin, Figure+OpenAI, T...
- [Tweet from Tsarathustra (@tsarnick)](https://x.com/tsarnick/status/1768021821595726254?s=46&t=90xQ8sGy63D2OtiaoGJuww): OpenAI CTO Mira Murati says Sora was trained on publicly available and licensed data
- [Tweet from SambaNova Systems (@SambaNovaAI)](https://x.com/sambanovaai/status/1762850777121583471): Introducing Samba-1, a one trillion (1T) parameter gen AI model for enterprise that is private, secure, and 10x more efficient than any other model of its size.
- [Tweet from Dylan Patel (@dylan522p)](https://x.com/dylan522p/status/1762924264695451841?s=20): @SambaNovaAI It&#39;s not a 1 trillion parameter model though.?? You understand the difference between a model and multiple models. Why make the marketing a lie? State what you actually did cause it&#...
- [Tweet from James O'Leary (@jpohhhh)](https://x.com/jpohhhh/status/1767568595586822326?s=46&t=Tc6nPt_FP2Ybqya6_6Xu-w): Google Gemini integration began 15 minutes ago, this is a cope thread  - There&#39;s an API called &#34;Gemini API&#34; that is free until we start charging for it early next year (it is mid-March) - ...
- [Tweet from Teortaxes▶️ (@teortaxesTex)](https://x.com/teortaxestex/status/1768261124187672972?s=46&t=90xQ8sGy63D2OtiaoGJuww): Read this if you haven&#39;t yet: http://blog.wtf.sg/posts/2023-02-03-the-new-xor-problem/  ↘️ Quoting Shawn Tan (@tanshawn)   One of the things we really needed for Sparse Universal Transformers was ...
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/anthropicai/status/1768018312083243514?s=46&t=90xQ8sGy63D2OtiaoGJuww): With state-of-the-art vision capabilities and strong performance on industry benchmarks across reasoning, math, and coding, Haiku is a versatile solution for a wide range of enterprise applications.
- [Tweet from Together AI (@togethercompute)](https://x.com/togethercompute/status/1767943482054967555?s=46&t=90xQ8): Today we are thrilled to share that we’ve raised $106M in a new round led by @SalesforceVC with participation from @coatuemgmt and our existing investors.  Our vision is to rapidly bring innovations f...
- [SuperPrompt - Better SDXL prompts in 77M Parameters | Brian Fitzgerald](https://brianfitzgerald.xyz/prompt-augmentation/): Left SDXL output with SuperPrompt applied to the same input prompt.
- [cerebras/btlm-3b-8k-base · Hugging Face](https://huggingface.co/cerebras/btlm-3b-8k-base): no description found
- [ I&#39;m concerned I made requests to openAI on behalf of another account - and perhaps someone did so on my behalf](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca):  I&#39;m concerned I made requests to openAI on behalf of another account - and perhaps someone did so on my behalf - openai-possible-security-breach.md
- [BTLM-3B-8K: 7B Performance in a 3 Billion Parameter Model - Cerebras](https://www.cerebras.net/machine-learning/btlm-3b-8k-7b-performance-in-a-3-billion-parameter-model/): Cerebras and Opentensor introduce a new standard for compact large language models
- [🌎 The Compute Fund](https://computefund.ai/): Reliably access the best GPUs you need at competitive rates in exchange for equity.
- [
      My benchmark for large language models
    ](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html): no description found
- [4,000,000,000,000 Transistors, One Giant Chip (Cerebras WSE-3)](https://www.youtube.com/watch?v=f4Dly8I8lMY&ab_channel=TechTechPotato): The only company with a chip as big as your head, Cerebras has a unique value proposition when it comes to AI silicon. Today they are announcing their third ...
- [Introducing Deci’s Gen AI Development Platform and Deci-Nano](https://deci.ai/blog/deci-nano-and-gen-ai-development-platform/): Explore Deci’s Gen AI Development platform and the Deci Nano LLM, designed to offer efficiency, performance, and flexible deployment options
- [Google Colaboratory](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing): no description found
- [Google Colaboratory](https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing): no description found
- [GitHub - Rohan2002/IFEval: Evaluator for LLMs](https://github.com/Rohan2002/IFEval): Evaluator for LLMs. Contribute to Rohan2002/IFEval development by creating an account on GitHub.
- [The Rivian R2: Are We Pre-Ordering?](https://youtu.be/Srh1lut4Q2A?si=N-JPakQxrxx7HzIo&t=3188): There was so much news this week! So much so, that we decided to basically split the podcast into three different segments. First, the Waveform crew talk abo...
- [Add support for Gemini API · Issue #441 · jxnl/instructor](https://github.com/jxnl/instructor/issues/441): The new Gemini api introduced support for function calling. You define a set of functions with their expected arguments and you pass them in the tools argument. Can we add gemini support to instruc...
- [Sell Domains | Buy Domains | Park Domains](https://x.co): no description found
- [Perspective – A space for you](https://joinperspective.com/): A private journal to build a complete record of your life.
- [google-research/instruction_following_eval at master · google-research/google-research](https://github.com/google-research/google-research/tree/master/instruction_following_eval): Google Research. Contribute to google-research/google-research development by creating an account on GitHub.

  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1217528504497733654)** (10 messages🔥): 

- **Synthetic Data for Finetuning Survey Presentation**: A reminder was posted for the presentation on Synthetic Data for Finetuning at 12pm PT, along with a recommendation to read ahead at [Eugene Yan's writing](https://eugeneyan.com/writing/synthetic/). Synthetic data is highlighted as a faster, cheaper, and often better-quality alternative to human annotations for pretraining and fine-tuning models.
  
- **Urgent Luma Invite for Paper Club Event**: A message urged members of the appropriate role to accept the Luma invite to ensure they continue receiving calendar reminders, with a pruning of inactive members slated for the same day. The event is viewable at [Luma](https://lu.ma/wefvz0sb).

- **Corrected Synthetic Data Link Provided**: A corrected link to the survey on synthetic data for fine-tuning was provided after the initial link was found to contain an extra period, causing a 404 error.

- **New Episode with Suno AI Released**: An announcement of a new podcast episode featuring Suno AI was shared, including a link to the Twitter announcement and a [YouTube video](https://youtu.be/gYXjn-V7AEw) titled "Making Transformers Sing - with Mikey Shulman of Suno".

**Links mentioned**:

- [LLM Paper Club (Synthetic Data for Finetuning) · Luma](https://lu.ma/wefvz0sb): This week we&#x27;ll be covering the survey post - How to Generate and Use Synthetic Data for Finetuning (https://eugeneyan.com/writing/synthetic/) with @eugeneyan We have moved to use the...
- [How to Generate and Use Synthetic Data for Finetuning](https://eugeneyan.com/writing/synthetic/): Overcoming the bottleneck of human annotations in instruction-tuning, preference-tuning, and pretraining.
- [Making Transformers Sing - with Mikey Shulman of Suno](https://youtu.be/gYXjn-V7AEw): Giving computers a voice has always been at the center of sci-fi movies; “I’m sorry Dave, I’m afraid I can’t do that” wouldn’t hit as hard if it just appeare...

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1217547863983259798)** (208 messages🔥🔥): 

- **Synthetic Data for LLMs**: A [blog post](https://eugeneyan.com/writing/synthetic/) by Eugene Yan was discussed, highlighting the use of synthetic data in pretraining, instruction-tuning, and preference-tuning of language models. Synthetic data generation methods include distillation from stronger models or self-improvement and can exceed the quality of human annotated data.
  
- **AI Newsletter Digests**: A daily AI newsletter roundup service offered by [AI News](https://buttondown.email/ainews/) summarizes discussions from AI discords and top Twitter accounts. The new service is mentioned to be valuable by users like Soumith Chintala and Andrej Karpathy.

- **Fine-tuning Knowledge Acquisition**: The conversation theorized about learning rates for fine-tuning versus pretraining, and posited that fine-tuning can indeed impart new knowledge to models. Community members debated on the efficiency of fine-tuning with regards to style transfer versus knowledge acquisition.

- **Speech-to-Text and Text-to-Speech Focus**: The group discussed the overlooked potential of voice technology in LLMs, particularly in text-to-speech and speech-to-text applications. Various tools were mentioned for transcription and generation of speech, including vapi.ai and Otter.

- **Audience Engagement in Paper Discussions**: Throughout the discussion, Eugene Yan encouraged active participation from the audience in choosing papers to cover and contribute to the paper club. There was interest in covering topics like diarization and streaming transcribe for speech models.

**Links mentioned**:

- [Join Slido: Enter #code to vote and ask questions](https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions): Participate in a live poll, quiz or Q&A. No login required.
- [Forget ChatGPT and Gemini &mdash; Claude 3 is the most human-like chatbot I've ever used](https://www.tomsguide.com/ai/forget-chatgpt-and-gemini-claude-3-is-the-most-human-like-chatbot-ive-ever-used#:~:text=Summary&text=Claude%203%20is%20one%20of,can%20speculate%20on%20its%20potential.): It isn't AGI but it is getting closer
- [Why Not Both Take Both GIF - Why Not Both Why Not Take Both - Discover &amp; Share GIFs](https://tenor.com/view/why-not-both-why-not-take-both-gif-11478682): Click to view the GIF
- [🦅 Eagle 7B : Soaring past Transformers with 1 Trillion Tokens Across 100+ Languages (RWKV-v5)](https://blog.rwkv.com/i/141130059/multi-lingual-performance-details>): A brand new era for the RWKV-v5 architecture and linear transformer&#x27;s has arrived - with the strongest multi-lingual model in open source today
- [How to Generate and Use Synthetic Data for Finetuning](https://eugeneyan.com/writing/synthetic/): Overcoming the bottleneck of human annotations in instruction-tuning, preference-tuning, and pretraining.
- [AI News](https://buttondown.email/ainews/): We summarize AI discords + top Twitter accounts, and send you a roundup each day! See archive for examples.  &quot;Highest-leverage 45 mins I spend everyday&quot; - Soumith &quot;the best AI newslette...
- [dspy/docs/api/optimizers/BootstrapFinetune.md at 0c1d1b1b2c9b5d6dc6d565a84bfd8f17c273669d · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/blob/0c1d1b1b2c9b5d6dc6d565a84bfd8f17c273669d/docs/api/optimizers/BootstrapFinetune.md?plain=1#L5): DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy
- [Fine-tuning vs RAG](https://open.spotify.com/episode/37Jd55nAruyVysHDNe0R6R?si=33926484c4c248a2): Listen to this episode from Practical AI: Machine Learning, Data Science on Spotify. In this episode we welcome back our good friend Demetrios from the MLOps Community to discuss fine-tuning vs. retri...
- [GitHub - EGjoni/DRUGS: Stop messing around with finicky sampling parameters and just use DRµGS!](https://github.com/EGjoni/DRUGS): Stop messing around with finicky sampling parameters and just use DRµGS! - EGjoni/DRUGS

  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1217392969016147988)** (130 messages🔥🔥): 

- **Seeking Token Probability Visualization**: A member inquired about a way to visualize the probability of each token in a sentence, similar to a chart depicted in an image. Suggestions were made regarding the use of lm_head's output and softmax to obtain probabilities, but no specific plugin was identified to create such visualizations.

- **Rapid AI Evolution**: Members highlighted the fast pace of AI development, with discussions about upcoming releases like Elon Musk's open **Grok model**, and rumors about one of the OpenAI founders calling the company "a lie."

- **Unsloth Fixes Google Colab Issues**: **Unsloth AI's** creator worked on fixes for Google Colab after a PyTorch update broke dependencies, providing a temporary list of commands for users to fix the issues themselves.

- **Clarifications on Model Compatibility with Unsloth**: Clarifications were made that Unsloth does not currently support multi-GPU or models in GGUF format for fine-tuning. Although Unsloth can quantize models to 4-bit for VRAM efficiency, it is presently designed for single-GPU usage.

- **Discussion on Data Preparation Best Practices**: A conversation regarding the need for an FAQ page for data preparation unfolded with suggestions on making the process simpler and more automated, possibly utilizing wrapper functions.

**Links mentioned**:

- [Crystalcareai/GemMoE-Beta-1 · Hugging Face](https://huggingface.co/Crystalcareai/GemMoE-Beta-1): no description found
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending): no description found
- [FastChat/fastchat/conversation.py at main · lm-sys/FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py): An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and Chatbot Arena. - lm-sys/FastChat
- [Implement LongLoRA trick for efficient tuning of long-context models · Issue #958 · huggingface/peft](https://github.com/huggingface/peft/issues/958): Feature request The authors of LongLoRA explore a trick you can toggle on during training and toggle off during inference. The key takeaways are: LoRA perplexity deteriorates as context length incr...
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth.git): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.

  

---


**Unsloth AI (Daniel Han) ▷ #[welcome](https://discord.com/channels/1179035537009545276/1179039724355211325/1217445576912666637)** (9 messages🔥): 

- **Read the Rules and Assign Roles**: theyruinedelise reminds new members to read the channel rules in <#1179040220717522974> and to assign themselves roles in <#1179050286980006030>.

- **Warm Welcomes Abound**: Multiple greetings from theyruinedelise and other users like starsupernova, indicating a friendly and welcoming atmosphere for newcomers in the welcome channel.
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1217712813347438663)** (5 messages): 

- **Virtual Environment Reinstallation in Progress**: A member mentioned they need to **reinstall the entire virtual environment** after finishing another task and expressed gratitude for the support provided.
- **Countdown to Milestone**: An expression of disbelief about time running short with only **one more day left** was noted.
- **Fine-Tuning Update**: There's an update on progress indicating **two days remaining** on fine-tuning, suggesting work is actively monitored and ongoing.
- **Celebrating a Training Victory**: A milestone of achieving a loss of **less than 1.2** was shared with enthusiasm, indicating successful model training advancements.
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1217556067479846992)** (73 messages🔥🔥): 

- **In Search of Cloud GPU Efficiency**: A user shared a personal success in finding a suitable and cost-effective cloud GPU for running inference at a rate of 500 t/s by renting a 4090 from vast.ai at approximately $0.46/hr, achieving about 130 t/s. They initially inquired about the cheapest option capable of delivering their computational needs.
- **GGUF Installation Troubles Resolved**: After experiencing an initial issue with GGUF installation that resulted in a `RuntimeError`, a user successfully resolved it by using a script from `llama.cpp` for conversion. Another user had a related problem, associated with an error message "/usr/bin/ld: cannot find -lcuda".
- **Colab's Finicky Performance**: Multiple users discussed the variability in Google Colab's available time for running notebooks, with mentions of the platform going from 2 hours up to 6 hours and general agreement on its buggy and glitchy nature.
- **Training Conversational Models with Personal Data**: A user expressing interest in creating a chatbot with personalized conversation based on Discord logs was directed to data preparation and the use of free Colab notebooks for training. Discussion also included the optimal data structure for conversational datasets, with examples provided for structuring the data with `instruction` and `answer`, or `user` and `assistant` dialogue formatting.
- **Technical Discussion on Finetuning and Saving Models**: Users engaged in discussions about whether using a 4-bit loading option for finetuning precludes the ability to later save to GGUF with quantization, with clarification that it does not affect GGUF saving. They also shared a link to a [tweet by Daniel Han](https://twitter.com/danielhanchen/status/1767968895749779937) and the resolution of a persistent GGUF conversion issue.
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1217721760166842508)** (3 messages): 

- **Exploring a New Optimizer, Sophia**: A member recommended considering the implementation of **Sophia**, a new optimization algorithm proposed in a paper, which could potentially speed up language model training. The optimizer aims to reduce time and cost by using a lightweight estimate of the diagonal Hessian for preconditioning, paired with element-wise clipping ([Read the Paper](https://arxiv.org/abs/2305.14342)).
- **Potential for Sophia as a Drop-in Replacement**: Another member noted that while they have not yet tested **Sophia**, it appears that it could be a straightforward “plug and play” optimizer. There is an interest in probing the efficacy of Sophia in practice.

**Links mentioned**:

[Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342): Given the massive cost of language model pre-training, a non-trivial improvement of the optimization algorithm would lead to a material reduction on the time and cost of training. Adam and its variant...

  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1217444777545699399)** (128 messages🔥🔥): 

- **Exploring Local AI Models**: Members discussed their experiences with various local models, with some using **LLM Studio** for testing. One user notes that they have strong inference power with up to 4xT4s, while another highlights the model **Meditron** as a particular interest.
- **Fine-Tuning Conversations**: The feasibility of fine-tuning larger models like **Mistral** on local hardware was debated, with some stating that a powerful GPU like an **A100 40GB** is necessary for such tasks. A user suggests that fine-tuning **GPT-3.5** could be reasonable and doesn't necessarily require a GPU.
- **Launching GPT-5? Major Typo Misleads**: Discussion emerged around a purported **Microsoft Copilot page** that mentioned "Priority access to GPT-4 and GPT-5 Turbo", which was later identified as a typo and corrected. The community speculated on the possibility of a **GPT-5**, with the consensus being that an immediate release is improbable.
- **Building with OpenAI**: A blog post shared by a user describes their experience integrating OpenAI with a system they developed to complete web flows, necessitating a network of models both big and small.
- **Model Missteps in Morpheme Repetition**: A conversation about **GPT-3.5's** challenges with generating examples of repeated morphemes in compound words resulted in the sharing of a chat log where GPT is guided to use Python to write a program yielding better results. Despite the complexity of the task, some successful outputs were highlighted.

**Links mentioned**:

[Microsoft Copilot | Microsoft AI](https://www.microsoft.com/en-us/microsoft-copilot): A new era of AI has arrived. Work more productively, boost efficiency, and find new growth opportunities with Copilot.

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1217559925853524068)** (41 messages🔥): 

- **GPT-4 Experiencing System-Wide Issues**: Multiple users reported that **GPT-4** is currently down, with error messages like “Hmm.. something seems to have gone wrong”. The problem persisted across various platforms including the iOS app and browsers like Chrome and Edge.
- **Status Checks and Temporary Workarounds**: One user suggested checking OpenAI's [status page](https://status.openai.com/) for updates, while another user found that starting conversations with an image attachment seemed to be a temporary workaround.
- **Dalle and "RP Thing" Remain Functional**: Despite problems with **GPT-4**, some users found that **Dalle 3** and a role-playing (RP) tool were still operational.
- **Feedback Features for GPT Creators**: A user inquired about the feedback and review features for GPT creators, expressing difficulty in locating this information and commenting on the searchability issues due to the commonality of the name "GPT".
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1217462420763578397)** (11 messages🔥): 

- **Code Interpreter Counts Words Correctly**: A member confirmed that using the prompt "Use code interpreter to count the revised text's words as {word_count}." is effective for counting words. The accuracy of the code interpreter's output was verified by comparing it with an external word counter.
- **Enhancing Lookup Functions in CustomGPT**: A user inquired about improving a custom GPT model to enable it to reference PDFs in its database and search the web before responding. It was noted that the model needs explicit instructions for searches and cannot recognize images within PDFs.
- **Localization Required for Assistant API**: In discussing the Assistant API, it was mentioned that a comma in the string "450,00" was not recognized correctly, leading to a misinterpretation of the figure as "45000". One user suggested that locale might be impacting this detection and that providing positive and negative examples could be necessary for correct recognition.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1217462420763578397)** (11 messages🔥): 

- **Word Count with Code Interpreter**: A member confirmed that using **code interpreter to count words** as `{word_count}` is functional and helpful for specific use cases.
- **Appreciation for Helpful Information**: One user expressed gratitude for the shared tip regarding the word count feature, planning to try it out after a busy work schedule.
- **Retrieval of PDF Content for CustomGPT**: A request for assistance was made to improve a **customGPT** to check PDFs in a database and look up information on the web before answering.
- **Formatting Issue with Commas in Assistant API**: A user pointed out that the **Assistant API Retrieval** does not recognize commas correctly in numbers, leading to confusion.
- **Locale Handling Affects Number Parsing**: It was suggested that proper parsing of numbers like "450,00" in the **Assistant API** may require setting the locale explicitly and providing both positive and negative examples.
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1217505274005422190)** (94 messages🔥🔥): 

- **DeepMind's New Generalist AI Agent**: DeepMind's [new research](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/?utm_source=twitter&utm_medium=social&utm_campaign=SIMA/) introduces a **Scalable Instructable Multiworld Agent (SIMA)**, a jump from specialized game agents to a generalist AI capable of understanding natural-language instructions in multiple video game environments. The technical report, however, lacks details like weights, dataset size, and training specifics, leading to skepticism among community members about the purpose and the transparency of the release.
  
- **Game Expertise Called Into Question**: The qualifications of game “experts” used in the evaluation of the new SIMA technical report are questioned, given only 16 hours of gameplay to establish expertise. Discussions raise concerns about what constitutes a game expert and the credibility of evaluations based on such expertise.

- **Discussing AI Progress in Gaming**: Community members debate the meaningfulness of AI achievements in games like StarCraft and DOTA, exploring the nuances between game-specific custom-built AIs and generalist approaches that deal with unpredictability in games like BR (Battle Royale).

- **The Challenge of Simulating Real-World Games in AI**: A lively back-and-forth takes place over the challenges facing AI in accurately simulating high-stakes, unpredictable multi-agent environments like those found in BR games and the real world. The conversation raises issues regarding the computational resources required and the difficulties in developing AIs that can make long-horizon plans in such complex settings.

- **Interest in AI Performance in Competitive Gaming**: There's intrigue about the potential for AI to be tested in competitive gaming environments such as the Apex Legends ranked leaderboard. Some community members suggest testing large language models directly in such environments, while others express doubt about AI's current ability to compete at human levels in BR games.

**Links mentioned**:

- [Byuntear American Psycho GIF - Byuntear American Psycho Staring - Discover &amp; Share GIFs](https://tenor.com/view/byuntear-american-psycho-staring-thinking-gif-26991038): Click to view the GIF
- [Introducing SIMA, a Scalable Instructable Multiworld Agent](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/?utm_source=twitter&utm_medium=social&utm_campaign=SIMA/): Introducing SIMA, a Scalable Instructable Multiworld Agent
- [Introducing SIMA, a Scalable Instructable Multiworld Agent](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/?utm_sour): Introducing SIMA, a Scalable Instructable Multiworld Agent
- [GitHub - MineDojo/Voyager: An Open-Ended Embodied Agent with Large Language Models](https://github.com/MineDojo/Voyager): An Open-Ended Embodied Agent with Large Language Models - MineDojo/Voyager

  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1217416060446314587)** (51 messages🔥): 

- **Frustration Over Access to Research**: A member expressed irritation at not being able to access interesting research due to publisher restrictions and shared a [link to a paper](https://www.pnas.org/doi/10.1073/pnas.2310002121).
- **Intrigue in NN Training Dynamics**: Discussion centered on an [arXiv paper](https://arxiv.org/abs/2305.01604) which explores the low-dimensional manifolds traversed by deep neural networks during training, highlighting interest in the implications for empirical methodologies in neural network research.
- **Potential Combination of Architectures**: The concept of combining multiple neural network architectures to potentially cover more space in problem-solving was considered.
- **Content Detectors Discussed**: The conversation turned to AI content detectors and identifiers wherein members debated their efficacy, noting that robustness remains questionable and discussing the possibility of false positives.
- **Concerns Over Watermarking AI Outputs**: Members discussed the challenges of watermarking for deterring synthetic media, with concerns about its viability and the potential impact on utility when the output is flagged as AI-generated.

**Links mentioned**:

- [The Training Process of Many Deep Networks Explores the Same Low-Dimensional Manifold](https://arxiv.org/abs/2305.01604): We develop information-geometric techniques to analyze the trajectories of the predictions of deep networks during training. By examining the underlying high-dimensional probabilistic models, we revea...
- [Language models scale reliably with over-training and on downstream tasks](https://arxiv.org/abs/2403.08540): Scaling laws are useful guides for developing language models, but there are still gaps between current scaling studies and how language models are ultimately trained and evaluated. For instance, scal...
- [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763): Large language models (LLMs) are routinely pre-trained on billions of tokens, only to start the process over again once new data becomes available. A much more efficient solution is to continually pre...

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1217531699106156556)** (22 messages🔥): 

- **Multimodal Mech Interpretability on the Horizon**: Soniajoseph_ announced the release of a multimodal mechanism interpretability library, encouraging collaboration to expand this subfield of research. The announcement was shared via a [Twitter link](https://twitter.com/soniajoseph_/status/1767963316943728779).
- **Discussing the Complexities of Model Agnosticism**: Neelnanda voiced concerns regarding the difficulty of making code that is model agnostic due to the various implementations of models under the hood, which led to TransformerLens reimplementing models from scratch.
- **Innovative Latent Decoding by Vector-DB-Lookup**: Wendlerc described an interpretability method using vector database lookups to analyze llama2's intermediate representations to provide "full-word-decodings" at each layer of the model.
- **Language-Dependent Dynamics in Multilingual Transformers**: Darkaz and Mrgonao engaged in a detailed discussion about whether multilingual models, such as LLMs, operate in a language-agnostic concept space or are biased towards the language with the highest representation during training.
- **Bilingual Model Tokenization Bias Exploration**: Butanium brought attention to an experiment using CroissantLLM, a bilingual French-English language model, and pondered the role of tokenization bias in comparison to the proportion of French vs. English training data. The experiment was detailed in a [GitHub notebook](https://github.com/Butanium/llm-latent-language/blob/main/nnsight.ipynb).

**Links mentioned**:

[llm-latent-language/nnsight.ipynb at main · Butanium/llm-latent-language](https://github.com/Butanium/llm-latent-language/blob/main/nnsight.ipynb): Repo accompanying our paper &quot;Do Llamas Work in English? On the Latent Language of Multilingual Transformers&quot;. - Butanium/llm-latent-language

  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1217486350421463202)** (10 messages🔥): 

- **Experimentation with Learning Rate Cooldown**: A checkpoint with a short learning rate (LR) cooldown was suggested to potentially improve benchmark results, but hardware availability delays obtaining outcomes.
- **Anxiety Over Model Performance**: As new checkpoints are being tested, there's an expression of concern over the anxious anticipation of the model's performance.
- **Seeking Assistance on LM Evaluation Feature**: A newcomer praised the LM evaluation harness and inquired about progress on adding logits to OpenAI ChatCompletions model, referencing an [open issue on GitHub](https://github.com/EleutherAI/lm-evaluation-harness/issues/1196).
- **Challenges with Logit Bias Post-Security Paper**: A reference to a recent [arXiv paper](https://arxiv.org/abs/2403.06634) explains why adding logits has become unfeasible due to changes in API designs that result from security concerns.
- **Adapting Tasks for Generative Models**: Discussion about adding generative variants of popular tasks to the evaluation harness, pointing to tasks like [GPQA](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gpqa) that support both loglikelihood and generative variants.

**Links mentioned**:

- [Stealing Part of a Production Language Model](https://arxiv.org/abs/2403.06634): We introduce the first model-stealing attack that extracts precise, nontrivial information from black-box production language models like OpenAI&#39;s ChatGPT or Google&#39;s PaLM-2. Specifically, our...
- [lm-evaluation-harness/lm_eval/tasks/gpqa at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gpqa): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [GitHub: Let’s build from here](https://github.co): GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...
- [Issues · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1196).): A framework for few-shot evaluation of language models. - Issues · EleutherAI/lm-evaluation-harness

  

---


**Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

boneamputee: https://brianfitzgerald.xyz/prompt-augmentation/
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1217448448643563611)** (1 messages): 

- **Contemplating Megatron Integration Strategy**: A member is considering the merits of more closely tracking upstream Megatron for Transformer Engine integration and has opened a [pull request](https://github.com/EleutherAI/gpt-neox/pull/1185) showing the full difference in code. They are inviting thoughts from the maintainers and community on whether this integration effort would be beneficial.

**Links mentioned**:

[Diffs to upstream megatron as a basis for discussion towards TE integration by tf-nv · Pull Request #1185 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1185): Here&#39;s three commits:  One with the full diff of GPT-NeoX&#39;s megatron folder with current upstream Megatron-LM. That&#39;s 256 files with ~60k lines. However most are completely new or deleted....

  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1217382304184537150)** (137 messages🔥🔥): 

- **Seeking Speedy Inference Solutions**: A member inquired about the fastest way to do inference with a *Phi 2 fine-tune* on a local GPU, mentioning batch processing with an A100 40GB and considering using frameworks like vLLM, Olama, or Axolotl. They wondered if quantization might help speed up the process.

- **Quantization and Streaming Approaches Discussed**: There was a debate on whether quantization could aid in expediting model accuracy, with an emphasis on using streaming methods for better response speed, such as those offered by *faster_whisper, llama_cpp, and xtts2*. Some members shared experiences using streaming TTS effectively, while others highlighted the potential use of bespoke hardware like Groq's NPU. Groq was mentioned to produce 500 tokens per second on *mixtral* [Groq](https://groq.com/) .

- **Concerns Over Model Weight Sharing and Copyright**: The conversation included concerns about recent copyright takedown notices and discussions on copyright laws as they relate to leaked model weights, AI-generated content, and the DMCA. Members also discussed the challenges of regulating AI and open sourcing model weights in light of government considerations against it.

- **European AI Legislation Sparks Debate**: There was discussion of the EU's new AI legislation, with critical opinions about requirements like disclosing AI-generated content and designing models to avoid generating illegal content. The conversation also pointed out the impracticalities of enforcing such requirements and the potential impact on open source models.

- **Prompt Augmentation for T5 and Danbooru Tagging**: Members shared resources on prompt augmentation with a 77M T5 model that can expand prompts, potentially rivaling larger LLMs, and a tiny **llama-focused** autocomplete tag generator for Danbooru. Interest was expressed in personal tuning and applying these models to existing projects.

**Links mentioned**:

- [SuperPrompt - Better SDXL prompts in 77M Parameters | Brian Fitzgerald](https://brianfitzgerald.xyz/prompt-augmentation/): Left SDXL output with SuperPrompt applied to the same input prompt.
- [Join the GroqCloud Discord Server!](https://discord.gg/groq): Groq provides the world&#x27;s fastest AI inference. | 5432 members
- [GroqChat](https://groq.com/): no description found
- [World’s first major act to regulate AI passed by European lawmakers](https://www.cnbc.com/2024/03/13/european-lawmakers-endorse-worlds-first-major-act-to-regulate-ai.html): The European Union&#x27;s parliament on Wednesday approved the world&#x27;s first major set of regulatory ground rules to govern the mediatized artificial intelligence at the forefront of tech investm...
- [Building Meta’s GenAI Infrastructure](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/): Marking a major investment in Meta’s AI future, we are announcing two 24k GPU clusters. We are sharing details on the hardware, network, storage, design, performance, and software that help us extr…
- [US Must Move 'Decisively' To Avert 'Extinction-Level' Threat From AI, Gov't-Commissioned Report Says - Slashdot](https://yro.slashdot.org/story/24/03/11/185217/us-must-move-decisively-to-avert-extinction-level-threat-from-ai-govt-commissioned-report-says): The U.S. government must move &#34;quickly and decisively&#34; to avert substantial national security risks stemming from artificial intelligence (AI) which could, in the worst case, cause an &#34;ext...
- [TheBloke/phi-2-GGUF · Hugging Face](https://huggingface.co/TheBloke/phi-2-GGUF): no description found
- [Demo of the neural avatar sept 18th 2023](https://youtu.be/TDitkDKbqbk): Demo of the neural avatar sept 18th 2023
- [BUD-E (Buddy for Understanding and Digital Empathy) - Blueprint / Overview](https://youtu.be/bLPDn-bh7dY?si=xrR2_F6kx1ydz8XM): https://docs.google.com/presentation/d/1tBBa0_GzzfCrmn9KpYZ8YZ9x4Jgb2zVs/edit?usp=sharing&amp;ouid=114592459581752579892&amp;rtpof=true&amp;sd=true
- [Education Pitch Deck](https://docs.google.com/presentation/d/1cMWLpMGNGs0_ZcKRKlJqM5OYiTSTyXgn39CDYOcgZq8/edit?usp=sharing): Navi-Sensei proposal JusticeDAO LLC Benjamin Barber business@hallucinate.app 10043 Se 32nd Ave Milwaukie Oregon 97222 9712700855 “I, Benjamin Barber, have read and understand the OMB and OPM Challenge...
- [Pitch Deck](https://docs.google.com/presentation/d/1_PejXm_nDP_b_Vig_WcnUh4WkFsSy2U0-ERQP2SD6-4/edit?usp=sharing): “Justice Now” - AI law avatar
- [Artificial Intelligence Act: MEPs adopt landmark law | News | European Parliament](https://www.europarl.europa.eu/news/en/press-room/20240308IPR19015/artificial-intelligence-act-meps-adopt-landmark-law): On Wednesday, Parliament approved the Artificial Intelligence Act that ensures safety and compliance with fundamental rights, while boosting innovation. 

  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1217444526742966316)** (21 messages🔥): 

- **MoAI: Merging Vision with Language Models**: A new paper on the Mixture of All Intelligence (**MoAI**) introduces an LLVM that incorporates auxiliary visual information from specialized **computer vision models**, aiming to enhance zero-shot vision-language tasks. The paper, available on [arXiv](https://arxiv.org/abs/2403.07508), posits that current LLVMs may benefit from incorporating detailed computer vision capabilities beyond the large capacities of LLM backbones.

- **MoAI Codebase Released**: The official PyTorch implementation for **MoAI** has been released on GitHub and is under review. The repository provides code to improve performance on numerous zero-shot vision language tasks and is available at [ByungKwanLee/MoAI on GitHub](https://github.com/ByungKwanLee/MoAI).

- **Using MoAI with Hugging Face**: A Hugging Face model page offers a **simple running code for MoAI**, along with necessary steps for setting up the environment and running the model. The page contains details for operations ranging from loading an image to generating predictions and can be found [here](https://huggingface.co/BK-Lee/MoAI-7B).

- **Dataset Recognition in DeepSeekVL Paper**: A member mentioned their dataset was cited in the DeepSeekVL paper, an initiative in scene understanding using vision-language models. The paper can be accessed via [this link](https://arxiv.org/pdf/2403.05525.pdf).

- **Discussion on Memory Usage and Lazy Loading in Large Models**: There has been a clarification that an earlier claim of being able to load a 30-billion-parameter model in just 4GB of memory using lazy loading was incorrect. The underreporting of RAM usage was due to mmap not reflecting the actual memory usage until the memory is accessed, as discussed in [gherganov/llama.cpp](https://github.com/ggerganov/llama.cpp/discussions/638#discussioncomment-5492916).

**Links mentioned**:

- [BK-Lee/MoAI-7B · Hugging Face](https://huggingface.co/BK-Lee/MoAI-7B): no description found
- [MoAI: Mixture of All Intelligence for Large Language and Vision Models](https://arxiv.org/abs/2403.07508): The rise of large language models (LLMs) and instruction tuning has led to the current trend of instruction-tuned large language and vision models (LLVMs). This trend involves either meticulously cura...
- [Jimmy Carter President Carter GIF - Jimmy carter President Carter Carter - Discover &amp; Share GIFs](https://tenor.com/view/jimmy-carter-president-carter-carter-gif-16271386811124661325): Click to view the GIF
- [GitHub - ByungKwanLee/MoAI: Official PyTorch implementation code for realizing the technical part of Mixture of All Intelligence (MoAI) to improve performance of numerous zero-shot vision language tasks. (Under Review)](https://github.com/ByungKwanLee/MoAI): Official PyTorch implementation code for realizing the technical part of Mixture of All Intelligence (MoAI) to improve performance of numerous zero-shot vision language tasks. (Under Review) - Byun...
- [GitHub - fashn-AI/tryondiffusion: PyTorch implementation of &quot;TryOnDiffusion: A Tale of Two UNets&quot;, a virtual try-on diffusion-based network by Google](https://github.com/fashn-AI/tryondiffusion): PyTorch implementation of &quot;TryOnDiffusion: A Tale of Two UNets&quot;, a virtual try-on diffusion-based network by Google - fashn-AI/tryondiffusion
- [30B model now needs only 5.8GB of RAM? How? · ggerganov/llama.cpp · Discussion #638](https://github.com/ggerganov/llama.cpp/discussions/638#discussioncomment-5492916): (Edit: apologies, I should have clarified initially I&#39;m running on Linux OS. I didn&#39;t realize it might not be obvious from the screenshot alone for a non-Linux users.All tests are done on Ubun...

  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1217950961612619907)** (1 messages): 

- **Visualize LLM Leaderboard with Ease**: The [Open LLM Leaderboard Viz update](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz) now allows users to change metrics order and plot up to 3 models for easy visual comparison.
- **Storytelling Gets Visual with GPT**: A new space called [Kosmos-2](https://huggingface.co/spaces/Tonic1/kosmos-2) by Tonic1 brings GPT-based visual storytelling to users.
- **ARC Dataset Augmented with Reasoning**: [Augmented ARC-Challenge Dataset](https://huggingface.co/datasets/Locutusque/arc-cot) incorporates Chain-of-Thought reasoning, offering more depth in answers to common questions.
- **Python Package for Vertex AI Inference**: A new Python package, [`vertex-ai-huggingface-inference`](https://github.com/alvarobartt/vertex-ai-huggingface-inference-toolkit), is available to streamline running HuggingFace models on Google Cloud's Vertex AI.
- **Rich Portuguese Pretrained Model Debuts**: Introducing [Mambarim-110M](https://huggingface.co/dominguesm/mambarim-110m), a Portuguese LLM with over 119 million parameters trained on a 6.2B token dataset.

**Links mentioned**:

- [Open Llm Leaderboard Viz - a Hugging Face Space by dimbyTa](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz): no description found
- [Kosmos 2 - a Hugging Face Space by Tonic1](https://huggingface.co/spaces/Tonic1/kosmos-2): no description found
- [Locutusque/arc-cot · Datasets at Hugging Face](https://huggingface.co/datasets/Locutusque/arc-cot): no description found
- [Aya - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/Aya): no description found
- [GitHub - alvarobartt/vertex-ai-huggingface-inference-toolkit: 🤗 HuggingFace Inference Toolkit for Google Cloud Vertex AI (similar to SageMaker&#39;s Inference Toolkit, but for Vertex AI and unofficial)](https://github.com/alvarobartt/vertex-ai-huggingface-inference-toolkit): 🤗 HuggingFace Inference Toolkit for Google Cloud Vertex AI (similar to SageMaker&#39;s Inference Toolkit, but for Vertex AI and unofficial) - alvarobartt/vertex-ai-huggingface-inference-toolkit
- [BEE-spoke-data/bert-plus-L8-v1.0-syntheticSTS-4k · Hugging Face](https://huggingface.co/BEE-spoke-data/bert-plus-L8-v1.0-syntheticSTS-4k): no description found
- [dominguesm/mambarim-110m · Hugging Face](https://huggingface.co/dominguesm/mambarim-110m): no description found
- [Machine learning-based intrusion detection: feature selection versus feature extraction - Cluster Computing](https://link.springer.com/article/10.1007/s10586-023-04089-5): Internet of Things (IoTs) has been playing an important role in many sectors, such as smart cities, smart agriculture, smart healthcare, and smart manufacturing. However, IoT devices are highly vulner...
- [GitHub - rbourgeat/refacto: Refactor your code with local LLM](https://github.com/rbourgeat/refacto): Refactor your code with local LLM. Contribute to rbourgeat/refacto development by creating an account on GitHub.
- [@DmitryRyumin on Hugging Face: &quot;🚀🎭🌟 New Research Alert! 🌟🎭 🚀
📄 Title: VLOGGER: Multimodal Diffusion for…&quot;](https://huggingface.co/posts/DmitryRyumin/888482747169050): no description found

  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1217383723360714844)** (76 messages🔥🔥): 

- **Speculation on Next-Gen AI**: A member predicted that **Llama 3** would be marketed as an AGI model, incorporating features like Llama Guard 2.
- **How to Contribute to Hugging Face Transformers**: Members discussed whether to commit a python virtual environment `venv` when contributing to Hugging Face `transformers`. It was clarified that one should **not** commit their local environment alongside their changes.
- **Inquiries about Freemium LLM for Spaces**: A member questioned the availability of a free, CPU-based Spaces that is compatible with OpenAI API for using a model akin to a 7B LLM.
- **Issues with Fine-Tuning and Model Implementation**: Participants discussed a range of technical questions, from proper implementation when fine-tuning models with LoRa to troubleshooting Spaces with Docker and finding the right method to implement knowledge into pre-trained models like Mistral 7B.
- **Data Privacy Concerns in Public Spaces**: Concerns about data privacy in public spaces were addressed, with a general recommendation to **avoid uploading personal information**. Details on specific Spaces and how they handle data can be scrutinized by inspecting the code.

**Links mentioned**:

- [Replica theory shows deep neural networks think alike](https://techxplore.com/news/2024-03-replica-theory-deep-neural-networks.html): How do you know you are looking at a dog? What are the odds you are right? If you're a machine-learning algorithm, you sift through thousands of images—and millions of probabilities—to arrive at the &...
- [Humans LyCORIS - v1.0 | Stable Diffusion LyCORIS | Civitai](https://civitai.com/models/103848/humans-lycoris): This is an extract of my Humans model. It took a while to find the right balance of size and fidelity, but I am finally happy with this version. No...
- [Devin AI Can Write Complete Source Code (How to Access It?)](https://favtutor.com/articles/devin-ai-software-engineer/): Cognition Labs reveals Devin, an AI software engineer who can write complete source code. Find out about its features, benchmarks, and how to get access.
- [Contribute to 🤗 Transformers](https://huggingface.co/docs/transformers/en/contributing): no description found
- [Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU](https://arxiv.org/abs/2403.06504): Recent advances in large language models have brought immense value to the world, with their superior capabilities stemming from the massive number of parameters they utilize. However, even the GPUs w...
- [bishmoy/Arxiv-CS-RAG at main](https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG/tree/main): no description found
- [Finetune Embeddings - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding.html): no description found
- [GitHub - moritztng/fltr: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B.](https://github.com/moritztng/fltr): Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B. - moritztng/fltr
- [Train and Deploy Vision Transformers for ANYTHING using Hugging Pics 🤗🖼](https://youtu.be/f9ZjgWBAxEQ?si=vYafMTnJDCBbKWCJ): In this video, we walk through Hugging Pics, a project that lets you train and deploy Vision Transformers for anything using pictures from the web.Try it out...

  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1217420486053724211)** (7 messages): 

- **Newcomer Queries on Accessing Custom Datasets**: A new Hugging Face user, familiar with Google Colab, sought guidance on accessing datasets in Hugging Face Spaces. They specifically asked about the paths to the images in the datasets and how to utilize the persistent storage `/data`.
  
- **Building AI Democracies**: One user has kickstarted an exploration into constructing a **multi AI decision model** with a voting mechanism, where the actions are determined by the majority vote among the AI models.

- **Request for Bayesian Know-how**: A user requested resources for learning Bayesian statistics, and they were directed to an educational [YouTube video](https://youtu.be/HZGCoVF3YvM) titled "Bayes theorem, the geometry of changing beliefs".

- **Collaborative AI Orchestration with MyShell Pro Config**: Another user introduced **MyShell's Pro Config** as a tool that could facilitate the orchestration of a **multi-AI decision model**, suggesting that it can manage the proposed voting process among AI agents.

- **MyShell as AI-native App Deployment Platform**: Further details about **MyShell** were shared, describing it as a decentralized platform for creating and managing AI-native apps, implying its usefulness for tasks like data analytics.

**Links mentioned**:

- [Pro Config Mode (beta) - MyShell](https://docs.myshell.ai/product-manual/create/pro-config-mode-beta): no description found
- [MyShell](https://myshell.ai/): MyShell is a decentralized and comprehensive platform for discovering, creating, and staking AI-native apps.
- [Bayes theorem, the geometry of changing beliefs](https://youtu.be/HZGCoVF3YvM): Perhaps the most important formula in probability.Help fund future projects: https://www.patreon.com/3blue1brownAn equally valuable form of support is to sim...

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1217465411784933407)** (10 messages🔥): 

- **Revolutionizing Retrieval**: [Retrieval-Augmented Language Models](https://arxiv.org/abs/2401.18059) now have an innovative approach—**RAPTOR**, which uses recursive summaries to better understand long documents and aid with complex QA tasks, showing significant improvements over conventional retrieval-augmented LMs.
- **AI-Assisted Artistry**: Open-source multimodal interpretability library suiting **Huggingface CLIP/ViTs** is available, and [Sonia Joseph revealed it via Twitter](https://twitter.com/soniajoseph_/status/1767963316943728779), offering enhanced access to mechanistic interpretability in AI models.
- **Diffusion Models Get a Boost**: Introducing **ELLA** (Efficient Large Language Model Adapter), combining diffusion models with LLMs for better semantic alignment in text-to-image generation, highlighted in a [Huggingface research paper](https://huggingface.co/papers/2403.05135).
- **Innovative AI Prompting with Storytelling**: A unique approach for effective prompting of Meta's Llama 2 AI is role-playing in various narratives, with [AI-generated prompts surpassing human-created ones](https://www.oneusefulthing.org/p/captains-log-the-irreducible-weirdness) in a quirky and unexpected fashion.
- **Advancing Text Segmentation**: A [research paper](https://arxiv.org/abs/2210.16422) brings light to the importance of segmenting long documents and proposes a model for simultaneous extractive summarization and segmentation, pushing towards state-of-the-art performance in understanding written and spoken text.

**Links mentioned**:

- [Toward Unifying Text Segmentation and Long Document Summarization](https://arxiv.org/abs/2210.16422): Text segmentation is important for signaling a document&#39;s structure. Without segmenting a long document into topically coherent sections, it is difficult for readers to comprehend the text, let al...
- [SudoLang: A Powerful Pseudocode Programming Language for LLMs](https://medium.com/javascript-scene/sudolang-a-powerful-pseudocode-programming-language-for-llms-d64d42aa719b): Pseudocode is a fantastic way to sketch programs using informal, natural language, without worrying about specific syntax. It’s like…
- [Captain&#x27;s log: the irreducible weirdness of prompting AIs](https://www.oneusefulthing.org/p/captains-log-the-irreducible-weirdness): Also, we have a prompt library!
- [RT-2: New model translates vision and language into action](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/): Introducing Robotic Transformer 2 (RT-2), a novel vision-language-action (VLA) model that learns from both web and robotics data, and translates this knowledge into generalised instructions for...
- [Paper page - ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment](https://huggingface.co/papers/2403.05135): no description found
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059): Retrieval-augmented language models can better adapt to changes in world state and incorporate long-tail knowledge. However, most existing methods retrieve only short contiguous chunks from a retrieva...
- [GitHub - havenhq/mamba-chat: Mamba-Chat: A chat LLM based on the state-space model architecture 🐍](https://github.com/havenhq/mamba-chat): Mamba-Chat: A chat LLM based on the state-space model architecture 🐍 - havenhq/mamba-chat

  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1217452805577510942)** (13 messages🔥): 

- **GLiNER: A Leap in Named Entity Recognition**: *cubietom* shared a demonstration of a new model framework called GLiNER which allows selection of custom labels for Named Entity Recognition (NER) on-the-fly, offering a practical alternative to traditional models with predefined entities. A demo is available at [HuggingFace Spaces](https://huggingface.co/spaces/tomaarsen/gliner_base), along with additional model variants and a GitHub repository for further exploration.

- **Laughter is the Best Medicine**: *tonic_1* shared their amusement with a creation made entirely using HuggingFace's *starchat2-playground*. They showcased a demo called kosmos-2 available at [HuggingFace Spaces](https://huggingface.co/spaces/Tonic1/kosmos-2).

- **Visualizing the LLM Landscape**: *taratra_dr* updated the community on the latest version of the Open LLM Leaderboard Viz space, featuring interactive visualizations and comparisons of large language models. New features include reordering of metrics and plotting multiple models for comparison, accessible via [HuggingFace Spaces](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz).

- **Code Refactoring at the Click of a Button**: *krolhm* introduced a new Visual Studio Code plugin for refactoring code, powered by a local large language model (LLM) with the llama cpp server, with the repository available on [GitHub](https://github.com/rbourgeat/refacto).

- **Germinating the Seeds of Multimodal Interpretability**: *soniajoseph_* announced the creation of an open-source library that brings multimodal mechanistic interpretability to Huggingface CLIP/Vision Transformer (ViT) models. Relevant links include a [Twitter post](https://twitter.com/soniajoseph_/status/1767963316943728779) and a detailed article on [LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic).

**Links mentioned**:

- [Open Llm Leaderboard Viz - a Hugging Face Space by dimbyTa](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz): no description found
- [Kosmos 2 - a Hugging Face Space by Tonic1](https://huggingface.co/spaces/Tonic1/kosmos-2): no description found
- [StarChat2 Demo - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/starchat2-playground): no description found
- [GitHub - rbourgeat/refacto: Refactor your code with local LLM](https://github.com/rbourgeat/refacto): Refactor your code with local LLM. Contribute to rbourgeat/refacto development by creating an account on GitHub.
- [Machine learning-based intrusion detection: feature selection versus feature extraction - Cluster Computing](https://link.springer.com/article/10.1007/s10586-023-04089-5): Internet of Things (IoTs) has been playing an important role in many sectors, such as smart cities, smart agriculture, smart healthcare, and smart manufacturing. However, IoT devices are highly vulner...
- [Laying the Foundations for Vision and Multimodal Mechanistic Interpretability &amp; Open Problems — LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic): Behold the dogit lens. Patch-level logit attribution is an emergent segmentation map. Join our Discord here. …
- [GLiNER-Base, zero-shot NER - a Hugging Face Space by tomaarsen](https://huggingface.co/spaces/tomaarsen/gliner_base): no description found
- [urchade/gliner_base · Hugging Face](https://huggingface.co/urchade/gliner_base): no description found
- [urchade/gliner_multi · Hugging Face](https://huggingface.co/urchade/gliner_multi): no description found
- [Models - Hugging Face](https://huggingface.co/models?library=gliner): no description found
- [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://arxiv.org/abs/2311.08526): Named Entity Recognition (NER) is essential in various Natural Language Processing (NLP) applications. Traditional NER models are effective but limited to a set of predefined entity types. In contrast...
- [GitHub - urchade/GLiNER: Generalist model for NER (Extract any entity types from texts)](https://github.com/urchade/GLiNER): Generalist model for NER (Extract any entity types from texts) - urchade/GLiNER

  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1217806245684314214)** (6 messages): 

- **No Show This Week**: There will be no presentation in this week's reading group, but one is planned for the upcoming week.
- **MNIST Digit Classification Question**: A member inquiring about **Andrew Ng's neural network course** was confused about the number of units in the first layer for an MNIST digit classification, given that the images are 20x20 pixels.
- **Exploring Neural Network Architecture**: In response to a question about determining the **number of neurons and hidden layers**, another member explained that this often involves experimentation and leveraging past successful configurations, considering the trade-off between processing power, speed, and accuracy.
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1217688016907276338)** (2 messages): 

- **Blend Styles with LoRAs**: A **guide** on merging [Low-Rank Adaptations (LoRAs)](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) is available, enabling the creation of unique images by blending different styles. Detailed instructions, including methods like [set_adapters()](https://huggingface.co/docs/diffusers/main/en/api/loaders/unet#diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters) and `fuse_lora()` are provided in the [merge LoRAs guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras).
  
- **Diffusers Library Update**: The new version **0.27.0** of the *Diffusers* library has been released. Release notes can be found on the [GitHub page](https://github.com/huggingface/diffusers/releases/tag/v0.27.0).

**Links mentioned**:

[Merge LoRAs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras): no description found

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1217582601859240046)** (2 messages): 

- **Latency Issues with LORAs and PEFT**: A member discussed challenges in integrating `peft` with *diffusers*, experiencing latency spikes when upgrading from peft 0.6 to 0.9. The `load_lora_weights` function is notably slower, increasing from 1-2 seconds to approximately 14 seconds, which is considered too high for their system. They shared a guide on hot-swapping [LORAs using HuggingFace](https://huggingface.co/blog/lora-adapters-dynamic-loading).

- **Enhancing Image Generation with FreeU**: An overview of the FreeU technique was shared, detailing its use to improve image generation quality by balancing the influence of skip connections and backbone features in the UNet architecture during the reverse diffusion process. The method has been highlighted as requiring no extra training and being applicable to various tasks, with more information available in a [Hugging Face guide](https://huggingface.co/docs/diffusers/using-diffusers/freeu).

**Links mentioned**:

[Improve generation quality with FreeU](https://huggingface.co/docs/diffusers/using-diffusers/freeu): no description found

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1217398660669767690)** (13 messages🔥): 

- **CLIP Embedding Curiosity**: A participant understood that passing images through the **CLIP model** to generate and save embeddings for later use in training is possible and emphasized that the original images should not be reconstructable from these embeddings. However, another interjected with uncertainty on whether image reconstruction from embeddings is entirely unfeasible.

- **Training with CLIP Embeddings**: A discussion highlighted that using **CLIP embeddings** instead of actual images for training might differ based on the task, and there's lingering uncertainty about the differences in training workflow for tasks such as object detection, classification, and pose estimation.

- **The Size of CLIP Embeddings**: It was mentioned that embeddings from the **CLIP model** might consume more size than the images themselves, and there's some ambiguity over whether this size increases or decreases after processing with **CLIPVisionModel**.

- **Batch Normalization as Knowledge Preservation**: A mention of an arXiv paper discussed how batch normalization could be used for lifelong learning in medical segmentation models to prevent forgetting old features, although the paper's exact name was not recalled.

- **Scaling Up Fine-Tuning for Image Generation**: A user inquired about techniques to fine-tune a Stable Diffusion (SD) model with a large dataset of 2.5 million images on decent hardware in less than a week, looking for tutorials that go beyond using small datasets for fine-tuning.


  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1217497650119839744)** (19 messages🔥): 

- **Mistral Model Flexibility Affected by Dataset Size**: A member reported that fine-tuning **Mistral 7B** with a small dataset allows for flexibility like modifying objects, but using a larger dataset leads to specialization in object generation at the expense of other tasks. They queried if this could be a form of overfitting, given the model’s size, and sought advice to mitigate the issue.

- **Fostering Generalization in Model Training**: In response to concerns about a model not generalizing well, one participant suggested enhancing the training set with more diverse examples to improve performance on new data.

- **Benchmarking a Modified Mistral Model**: A user shared their intent to compare a base model **Mistral-7V-v0.1** with a modified version for a research idea, seeking guidance on how to use HuggingFace's automated benchmarks and inquiring where these benchmarks run.

- **OpenLLM Leaderboard Benchmark Submission Clarified**: Another member clarified that for the OpenLLM leaderboard, benchmarks are run on a Hugging Face cluster. They provided links to resources such as **LightEval** and the **lm-evaluation-harness** for self-benchmarks.
[LightEval Suite on GitHub](https://github.com/huggingface/lighteval).
[lm-evaluation-harness on GitHub](https://github.com/EleutherAI/lm-evaluation-harness).

- **Potential Innovation in Model Compression Techniques**: Discussions arose around a new method of model optimization that may allow for memory footprint savings while maintaining accuracy, including successful preliminary results on a `4096 x 4096` matrix. A member expressed enthusiasm for applying this technique to larger matrices within the model’s architecture.

**Links mentioned**:

- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [GitHub - huggingface/lighteval: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron.](https://github.com/huggingface/lighteval): LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron. - hug...

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1217582601859240046)** (2 messages): 

- **Troubleshooting Latency Issues with PEFT and Diffusers**: A server operator using [LoRA adapters for dynamic model loading](https://huggingface.co/blog/lora-adapters-dynamic-loading) reports high latency issues when integrating **`peft`**. While **`peft` 0.9** greatly increases `load_lora_weights` time to 14 seconds, version 0.6 reduces this time but increases `unload_lora_weights` to 6 seconds, both of which are unacceptable for their system.

- **Enhancing Image Quality with FreeU**: An improved image generation technique called **FreeU** is discussed, which rebalances the contributions of the UNet's skip connections and backbone feature maps to enhance image quality. The technique, applicable during inference without extra training, can be used for text-to-image, image-to-image, and text-to-video tasks as detailed in [HuggingFace's guide](https://huggingface.co/docs/diffusers/using-diffusers/freeu).

**Links mentioned**:

[Improve generation quality with FreeU](https://huggingface.co/docs/diffusers/using-diffusers/freeu): no description found

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1217560477933240420)** (3 messages): 

- **Temporary Service Interruption Alert**: OpenRouter experienced a brief issue where some Activity rows went missing for approximately three minutes due to a protracted database update, potentially affecting the billing for completions within that time. The problem was swiftly addressed, stating that "*none of these completions will be charged*".

- **Launch of Claude 3 Haiku**: **Claude 3 Haiku** by Anthropic is now available on OpenRouter, characterized by its high speed (around 120 tokens per second) and cost efficiency (4 million prompt tokens per dollar). This low-latency, beta version offers both moderated and self-moderated options, suitable for use cases requiring near-instant responsiveness. Check out the model and its pricing [here](https://openrouter.ai/models/anthropic/claude-3-haiku:beta).

- **New Model Release**: Cohere's **Command-R** model is now accessible on OpenRouter, showcasing a long context capability of 128,000 tokens at the rate of 2 million prompt tokens per dollar. Efforts have been made to align Command-R with the universal API for a seamless user experience. Interested users can explore Command-R through this [link](https://openrouter.ai/models/cohere/command-r).

- **Daily Analytics Now Available**: OpenRouter introduces daily analytics enabling users to track token usage on a daily basis, offering a more granular view alongside the existing weekly analytics. Users can view the new analytics [here](https://openrouter.ai/rankings).

- **Performance Improvements Announced**: OpenRouter has significantly increased the speed of the `/models` API and enhanced the performance of all model-related web pages, including improvements to Mixtral Nitro.

**Links mentioned**:

- [Anthropic: Claude 3 Haiku (self-moderated) by anthropic | OpenRouter](https://openrouter.ai/models/anthropic/claude-3-haiku:beta): This is a lower-latency version of [Claude 3 Haiku](/models/anthropic/claude-3-haiku), made available in collaboration with Anthropic, that is self-moderated: response moderation happens on the model&...
- [Cohere: Command-R by cohere | OpenRouter](https://openrouter.ai/models/cohere/command-r): Command-R is an instruction-following conversational model that performs language tasks at a higher quality, more reliably, and with a longer context than previous models. It can be used for complex w...
- [OpenRouter](https://openrouter.ai/rankings): Language models ranked and analyzed by usage across apps

  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1217470944973426799)** (7 messages): 

- **Olympia.Chat Announces OpenRouter Integration**: [Olympia.Chat](https://olympia.chat), a ChatGPT clone popular with solopreneurs and small business owners, is incorporating OpenRouter as the LLM source for its components. Additionally, a **fully featured Ruby library** for OpenRouter will be open-sourced soon.

- **Chatbot for Messenger Available for Testing**: An unnamed friend of a member has created a **Messenger chatbot**, and the member is inviting others to direct message for testing opportunities.

- **AI Gateway Launch with OpenAI Integration**: A new AI gateway, [EZLinkAI Platform](https://platform.ezlinkai.com/), offers user registrations with a $1 gift and allows users to call OpenAI, Claude, Mistral, and Groq services at 80% of the original costs.

- **Request for Feedback on AI Gateway**: The creators of the **AI gateway** are seeking more feedback, implying the need for user input to improve their service.

**Links mentioned**:

- [Olympia | Better Than ChatGPT](https://olympia.chat): Grow your business with affordable AI-powered consultants that are experts in business strategy, content development, marketing, programming, legal strategy and more.
- [How Anthony Mennella GIF - How Anthony Mennella Culter35 - Discover &amp; Share GIFs](https://tenor.com/view/how-anthony-mennella-culter35-how-did-you-do-that-how-to-do-it-gif-20143672): Click to view the GIF
- [EZLINK AI](https://platform.ezlinkai.com/): no description found

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1217441390368264232)** (129 messages🔥🔥): 

- **GPT-4.5 Turbo Vanishing Act**: A member shared a link ([openai.com/blog/gpt-4-5-turbo](https://openai.com/blog/gpt-4-5-turbo)) that was supposedly evidence of GPT-4.5 Turbo's existence but later said it was gone, prompting laughter.
- **Mistral Model Mysteries**: Users reported discrepancies with the Mistral model's behavior, including "Request too big" errors and difficulties with the context limit, which is supposed to be 32k. The conversation included a query about the exact error message and proposed reasons for the errors like repeated loops in requests.
- **Claude 3 Haiku Hype**: The discussion revealed enthusiasm for Claude 3 Haiku, highlighted for its cost efficiency at 1.25 USD per million tokens and for being significantly better than other models in brainstorming roleplay scenarios and character development.
- **OpenRouter Branding Collaboration**: A proposal to add an OpenRouter button to Open Agent Studio was discussed, with the request for branding guidelines or a specific icon to use, which was given a green light by the OpenRouter side.
- **An Exploration of Various LLMs**: The chat featured members comparing various language models, including Gemini and Claude models, debating their capabilities in coding and creative tasks, lamenting about certain quirks like unwanted bullet points, and expressing strong preferences for some over others due to performance and lack of censoring.

**Links mentioned**:

- [GitHub - open-webui/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI)](https://github.com/open-webui/open-webui): User-friendly WebUI for LLMs (Formerly Ollama WebUI) - open-webui/open-webui
- [No-Code Agents Live Group Onboarding](https://youtu.be/dT1p7aAC1eU): Get up to speed on the best practices and the order of operations that will save you time and enable automating any process you&#39;ve done manually.Schedule you...
- [GitHub - BerriAI/litellm: Call all LLM APIs using the OpenAI format. Use Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate (100+ LLMs)](https://github.com/BerriAI/litellm): Call all LLM APIs using the OpenAI format. Use Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate (100+ LLMs) - BerriAI/litellm

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1217507828701466798)** (3 messages): 

- **Introducing LlamaParse**: The new **LlamaParse** document parser has launched, offering superior parsing of images, tables, and charts, with the added ability to follow natural language instructions. Discover how it outperforms others in this [tweet](https://twitter.com/llama_index/status/1767948064659210310).

- **LlamaIndex Tackles PII with Presidio**: A guest post by @RoeyBC on **LlamaIndex** highlights **Presidio**, an open-source library by Microsoft, which identifies and anonymizes personally identifiable information (PII) to prevent data leakage. Read about its importance in data protection in this [tweet](https://twitter.com/llama_index/status/1768050386823463368).

- **Overcoming RAG Challenges in Finance**: RAG faces difficulties in parsing finance PowerPoint presentations due to their unique format, including tables, images, and charts. A technique for better text positioning and parsing is the first crucial step, explained in this [tweet](https://twitter.com/llama_index/status/1768303288381030408).
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1217438443223781436)** (82 messages🔥🔥): 

- **Azure AI Search Index Issues**: A member following the [AzureAISearchIndexDemo guide](https://docs.llamaindex.ai/en/stable/examples/vector_stores/AzureAISearchIndexDemo.html) encountered an issue where the Azure index shows a total storage of 3mb, but the vector index size is 0. Advice sought on this discrepancy.

- **Warnings with LlamaIndex Python Packages**: One user reported multiple warnings regarding the failure to use `OpenAIPydanticProgram`. It was advised to run `pip install llama-index-program-openai` to resolve the issue.

- **Concerns Over npx create-llama Errors**: A member faced errors stating "Sorry! We've encountered an issue with repetitive patterns in your prompt" when using `npx create-llama` with text files as data sources, even with simple prompts. It was speculated that the error could be related to the contents of the files.

- **Evaluation Methods for Retriever in LlamaIndex**: One user sought advice on using LlamaIndex's RetrieverEvaluator with their own question context pairs. It was mentioned that expected node IDs from queries are required, but it was questioned if one could use only expected text or document IDs instead.

- **Performance Issues with OpenAI Assistant Agent**: A member discussed the slow response time of over 10 seconds when using OpenAIAssistantAgent for building a chatbot. It was suggested that streaming might make it feel faster and that slow response times can partly be due to recent performance issues with the OpenAI API.

**Links mentioned**:

- [Retrieval Evaluation - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/evaluation/retrieval/retriever_eval.html): no description found
- [Azure AI Search - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/vector_stores/AzureAISearchIndexDemo.html): no description found
- [Simple Fusion Retriever - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/retrievers/simple_fusion.html#simple-fusion-retriever): no description found
- [Ingestion Pipeline + Document Management - LlamaIndex 🦙 v0.10.19](https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline.html): no description found

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1217445256375566456)** (61 messages🔥🔥): 

- **Searching for Open-Source Chat-ready Models**: A discussion arose around the complexity of model size in relation to training resources and hardware capabilities. [Mistral and Mixtral](https://huggingface.co/) models were suggested for their open-source nature and lack of significant filters.
  
- **Model Training Ambitions Confront VRAM Limitations**: A participant expressed the intention to train on large models, highlighting [PyTorch's Metal Performance Shaders (MPS) backend for Mac GPU training acceleration](https://developer.apple.com/metal/pytorch/). Others inquired about fine-tuning capabilities and limitations on single GPU setups, suggesting the need for [efficient fine-tuning methods](https://huggingface.co/papers/2403.06504).

- **Debating Between Mixtral and Qwen 70B for Medical Training**: One member contemplated training a large model for medicine and deliberated between the Mixtral and Qwen 70B models. Concerns over imminent out-of-memory (OOM) issues and an impending release of a new llama model were raised.

- **Querying the Best Practices for Training Formats**: Members exchanged thoughts on using completion versus question-and-answer (Q/A) formats when converting raw text for training purposes. It was suggested to refer to existing Hugging Face dataset examples for formatting data properly.

- **GPUDirect Storage for Axolotl**: A participant suggested the potential for integrating NVIDIA's GPUDirect® Storage technology into the Axolotl system, offering a direct data path for transfers between GPU memory and storage, as detailed in NVIDIA's [cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html). This could enhance performance by increasing system bandwidth and reducing CPU load.

**Links mentioned**:

- [Accelerated PyTorch training on Mac - Metal - Apple Developer](https://developer.apple.com/metal/pytorch/): PyTorch uses the new Metal Performance Shaders (MPS) backend for GPU training acceleration.
- [Paper page - Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a
  Single GPU](https://huggingface.co/papers/2403.06504): no description found
- [cuFile API Reference Guide - NVIDIA Docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html): no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1217389990292750378)** (6 messages): 

- **GPU Direct Primer Video Shared**: An [introductory video](https://www.youtube.com/watch?app=desktop&v=32CeexHBOd4) explaining NVIDIA's GPUDirect Storage (GDS) was shared, which provides insight into peer-to-peer PCIe and the role of GDS in technological advancements.
- **Axolotl Code Queried**: A member posted a query regarding a specific portion of the Axolotl code, with a link to the [relevant GitHub section](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a82d2e0a443fb866b15d7bd71fffbd8171de44b/src/axolotl/utils/models.py#L807-L808), seeking clarification on its purpose.
- **Model Loading Explanation**: In response to the query, it was clarified that the referenced code would be triggered when directing the base model pointer to a *peft model*, enabling AutoModel to load a peft model at that point.
- **Request for New Features**: A member expressed curiosity about the latest features being developed or introduced.
- **PEFT Paper Linked**: In response to inquiries about new features, a [research paper on "PEFT"](https://arxiv.org/pdf/2403.06504.pdf) was shared, suggesting advancements in the modeling domain.

**Links mentioned**:

- [P2P PCIe and GPUDirect Storage Primer](https://www.youtube.com/watch?app=desktop&v=32CeexHBOd4): This is a five-minute quick look at what NVIDIA&#39;s GPUDirect Storage (GDS) is, what it does, what technologies it is built upon, and where it makes the most s...
- [axolotl/src/axolotl/utils/models.py at 8a82d2e0a443fb866b15d7bd71fffbd8171de44b · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a82d2e0a443fb866b15d7bd71fffbd8171de44b/src/axolotl/utils/models.py#L807-L808): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1217868370293030993)** (7 messages): 

- **Seeking Inference Code for LoRA Model**: A member mentioned they've **fine-tuned LoRA off `Mistral-7B-v0.1`** and sought example code for running inference on about 100 prompts within a notebook. They were contemplating using the `transformers` library and `model.generate(**model_inputs)` method.

- **vLLM Recommended for Swift Inference**: Another member recommended using **vLLM** for running batched inference, which they claimed to be quicker than `transformers`. They provided a [quickstart guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) for using vLLM that covers offline batched inference and building OpenAI-compatible API servers.

- **Considering vLLM for Non-Server Tasks**: The original inquirer was unsure if **vLLM** would be suitable for their needs, as they were not planning to serve the model but simply run a few predictions for exploration. After assurance of its efficiency, they decided to follow the vLLM quickstart link.

**Links mentioned**:

[Quickstart &#8212; vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html): no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1217474856644382733)** (3 messages): 

- **Mistrial Medium Outperforms Mixtral**: A user noted that **Mistral Medium** yields better responses and is believed to be a closed-sourced, superior version of **Mixtral**.

- **RAG Performance Noticed**: The same user mentioned observing citation generation with **RAG performance** without explicitly requesting them.

- **Less Verbose, Better Instruction Follow-through**: It was also observed that outputs from **Mistral Medium** are less verbose and more effective at following instructions than **Mixtral**.
  

---



**LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1217877731963048016)** (1 messages): 

- **Expedited Release for langchain 0.2**: Due to CVEs filed against `langchain`, the team is considering an expedited release of `langchain 0.2` that will separate it from `langchain-community`. The detailed discussion and motivation for this change can be found on [GitHub](https://github.com/langchain-ai/langchain/discussions/19083), and community feedback is encouraged to ensure it addresses user needs.

**Links mentioned**:

[RFC: Expedited langchain 0.2 release · langchain-ai/langchain · Discussion #19083](https://github.com/langchain-ai/langchain/discussions/19083): Context Currently langchain (the package) depends on langchain-community. This is done only for backwards compatibility with langchain versions that predate the split of langchain and langchain-com...

  

---


**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1217401410287566909)** (64 messages🔥🔥): 

- **LangChain Inquiry**: A member looking for help with **LangChain** was directed to the appropriate help channel on Discord for assistance.
- **AgentExecutor Issues**: There's a mention of difficulty with `AgentExecutor` returning an `OutputParserException`, even when the **Cohere model** seems to generate Python code accurately.
- **AI Agents Under the Hood**: A discussion on why one would use AI agents over LLMs + functions highlighted that agents handle sequential actions and come with built-in error handling, amongst other features.
- **Evaluation of AI Agent Behavior**: A member sought advice on evaluating AI agent behavior and was referred to the [LangChain debugging and evaluation guides](https://python.langchain.com/docs/guides/evaluation/), although there was an acknowledgment that the area seems to be relatively new with benchmarks still under development.
- **StackOverflow API Exploration**: A user asked about an API for **StackOverflow** and received guidance on using the [StackExchange API](https://api.stackexchange.com/docs/advanced-search) to perform an advanced search based on specific queries with structured data.

**Links mentioned**:

- [GroqCloud](https://console.groq.com/docs/openai#text-completion): Experience the fastest inference in the world
- [[beta] Structured Output | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/structured_output): It is often crucial to have LLMs return structured output. This is
- [Discord Bot | MEE6](https://mee6.xyz/en).): Manage your Discord server with leveling, moderation, Twitch, Youtube and Reddit notifications.
- [OpenAI assistants | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/agents/agent_types/openai_assistants): The [Assistants
- [LangChain](https://www.youtube.com/playlist?list=PLqZXAkvF1bPNQER9mLmDbntNfSpzdDIU5): no description found
- [Debugging | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/debugging): If you&#x27;re building with LLMs, at some point something will break, and you&#x27;ll need to debug. A model call will fail, or the model output will be misformatted, or there will be some nested mod...
- [Evaluation | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/evaluation/): Building applications with language models involves many moving parts. One of the most critical components is ensuring that the outcomes produced by your models are reliable and useful across a broad ...
- [
Usage of /search/advanced [GET] - 
        Stack Exchange API
    ](https://api.stackexchange.com/docs/advanced-search): no description found
- [The validation of tools within OpenAIAssistantRunnable.create_assistant does not account for `{&quot;type&quot;: &quot;code_interpreter&quot;}`. · Issue #19057 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/19057): Checked other resources I added a very descriptive title to this issue. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and di...
- [langsmith-cookbook/testing-examples/tool-selection/tool-selection.ipynb at main · langchain-ai/langsmith-cookbook](https://github.com/langchain-ai/langsmith-cookbook/blob/main/testing-examples/tool-selection/tool-selection.ipynb): Contribute to langchain-ai/langsmith-cookbook development by creating an account on GitHub.
- [GitHub - ggerganov/whisper.cpp: Port of OpenAI&#39;s Whisper model in C/C++](https://github.com/ggerganov/whisper.cpp?tab=readme-ov-file): Port of OpenAI&#39;s Whisper model in C/C++. Contribute to ggerganov/whisper.cpp development by creating an account on GitHub.
- [Wordware - Try all the models for a single question](https://app.wordware.ai/r/fc405cb4-877b-44b7-aed8-b883e48eced3): This prompt runs a question through Gemini, GPT-4 Turbo, Claude 2, Mistral Medium, Mixtral and Openchat. The it uses GPT-4 Turbo to assess which model gave the best answer.

  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1217795010247000124)** (1 messages): 

- **Question on Variable Integration in Prompt Templates**: A member inquired about integrating a variable, specifically `tools = [cat_tool]`, into a Langsmith Hub prompt template that includes the placeholder `{tools}` within the construct:

  ``` 
  System : 

  You are a helpful assistant that have these {tools} to help answer questions.
  ```
  
  They are seeking guidance on how to reference the variable `tools` in their code to align with the prompt.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1217398683377602611)** (8 messages🔥): 

- **Reacting with ReAct**: The ReAct agent, inspired by the 'ReAct: Synergizing Reasoning and Acting in Language Models' paper, has been shared, boasting a reasoning engine and diverse skills which can be tested with questions like 'What is the Bitcoin price today?'. The related paper can be downloaded at [Download PDF](https://arxiv.org/abs/2210.03629).
  
- **Open Source Langchain Chatbot**: A new open source Langchain chatbot has been introduced to demonstrate efficient question/answer querying using the RAG technique, featuring a [GitHub repository](https://github.com/Haste171/langchain-chatbot) with a simple setup and interactive UI.
  
- **MindGuide: Innovating Mental Health via ChatModels**: An article titled 'Revolutionizing Mental Health Care through LangChain' was shared, detailing the MindGuide chatbot that utilizes LangChain and ChatOpenAI for mental health support, with the abstract and download available at [Download PDF](https://arxiv.org/abs/2403.05568).

- **Claude Meets LangGraph for Supervising**: A GitHub notebook showcasing the Claude powered LangGraph Agent Supervisor was shared, demonstrating the potential of utilizing LangChain with Claude's capabilities, available at [GitHub Notebook](https://github.com/prof-frink-lab/slangchain/blob/main/docs/modules/graphs/examples/anthropic/agent_supervisor.ipynb).

- **Deci AI Nano Model API Sneak Peek**: Deci AI's new nano model API was announced, with accompanying Colab notebooks for basic usage and LangChain usage ready to be explored prior to its official release, with [Basic Usage Notebook](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing) and [LangChain Usage Notebook](https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing) linked for access.

**Links mentioned**:

- [Revolutionizing Mental Health Care through LangChain: A Journey with a Large Language Model](https://arxiv.org/abs/2403.05568): Mental health challenges are on the rise in our modern society, and the imperative to address mental disorders, especially regarding anxiety, depression, and suicidal thoughts, underscores the need fo...
- [LangChain for JavaScript part 3: Create Dall-E images](https://fek.io/blog/lang-chain-for-java-script-part-3-create-dall-e-images/): FEK.IO The website for David Fekke L.L.C.
- [Google Colaboratory](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing): no description found
- [Google Colaboratory](https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing): no description found
- [slangchain/docs/modules/graphs/examples/anthropic/agent_supervisor.ipynb at main · prof-frink-lab/slangchain](https://github.com/prof-frink-lab/slangchain/blob/main/docs/modules/graphs/examples/anthropic/agent_supervisor.ipynb): Contribute to prof-frink-lab/slangchain development by creating an account on GitHub.
- [Unlocking the Future of AI Applications with SAP HANA Vector Engine and LangChain](https://ai.gopubby.com/unlocking-the-future-of-ai-applications-with-hana-vector-engine-and-langchain-14cd6c66219d): Ankush k Singal
- [GitHub - Haste171/langchain-chatbot: AI Chatbot for analyzing/extracting information from data in conversational format.](https://github.com/Haste171/langchain-chatbot): AI Chatbot for analyzing/extracting information from data in conversational format. - Haste171/langchain-chatbot
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629): While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-though...
- [Wordware - ReAct API Agent 🧠](https://app.wordware.ai/r/0b8b7771-09dc-4a19-87d4-89e43b5cc153): Works out how to use APIs

  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1217452564510150686)** (2 messages): 

- **Learn Prompt Template Creation with Langchain**: A video tutorial titled "Create Prompt Template With Langchaingo" was shared, which demonstrates how to create a prompt template and use it with Langchain, particularly featuring a [Telegram group](https://t.me/langchaingo/1). The content is aimed at developers interested in #golang and #langchain, and the video can be viewed on [YouTube](https://youtu.be/dcBEtgh4078).

- **Diving into Function Calling with Hermes 2 Pro 7B**: Another video titled "Lets Function Call with Hermes 2 Pro 7B" was shared, focusing on function calling using the **Hermes 2 Pro 7B** model. The source code and examples can be found on [GitHub](https://github.com/NousResearch/Hermes-Function-Calling/tree/main), and the video is accessible on [YouTube](https://www.youtube.com/watch?v=PzaidfqDtGI), targeting #llm and #largelanguagemodels enthusiasts.

**Links mentioned**:

- [Create Prompt Template With Langchaingo](https://youtu.be/dcBEtgh4078): In this video , I&#39;ll hoe create a promp template and how use this with chainstelegram group:https://t.me/langchaingo/1#golang #langchain
- [Lets Function Call with Hermes 2 Pro 7B](https://www.youtube.com/watch?v=PzaidfqDtGI): lets do function calling with Hermes 2 Pro 7Bhttps://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels

  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1217547023675555970)** (6 messages): 

- **Tweet Triggers Discussion**: A member shared [a tweet by Andrew Curran](https://twitter.com/AndrewCurran_/status/1767916848987914487) leading to a conversation on language applications and collaborations, emphasizing the need to work with subgroups through the Aya project.
- **Polyglot Projects in European Academia**: Discussing the needs of European universities, a member mentioned the challenge of persuading people to adopt new language approaches, with particular mention of English and German applications.
- **German Language Well Supported by LLMs**: One member noted that German is generally well supported by serious LLMs (Language Learning Models) out of the box, while also suggesting reaching out to Aleph for partnerships in highly regulated industries.
- **Aleph's Performance in Question**: A member expressed their opinion that Aleph's performance is lacking, which led to the suggestion that while Aleph itself might not be up to par, they could still assist in referring to local data partners.
  

---


**Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1217542990441087056)** (2 messages): 

- **GPT-4 retains the crown**: A member remarked that according to a paper, **GPT-4** is still the leading model on **LeetCode**. The paper mentioned can be found at [livecodebench.github.io](https://livecodebench.github.io/pdfs/paper.pdf).
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1217710697434189874)** (3 messages): 

- **Inquiry on Model Details**: A member asked another to share which model was used in a recent exercise, showing interest in the model's identity and capabilities.

- **Seeking Citations for Safety Filtering by Providers**: A member is looking for authoritative sources or documentation to cite that “foundation model providers do a lot of the safety filtering for text post generation” but notes that it's less documented compared to prompt rewriting.
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1217715794780098571)** (2 messages): 

- **Catching Up on the Salty**: A member mentioned their intention to catch up on the newsletter backlog and expressed that having *"salty readers"* is beneficial. They also alluded to a teaser tweet about bio risk which they disagreed with but reserved judgment until reading the full post.
- **Bio Risk Tweet Confusion**: There was a brief confusion about a bio risk-related tweet. A mention was made about tweeting too much, possibly implying a lack of context or information in the initial tweet.
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1217461687171547176)** (54 messages🔥): 

- **Anticipating GPT-4.5**: Members expressed readiness for an emergency blog post in case **GPT-4.5** gets released suddenly.
- **YouTube Premium vs. Google Ads in LLMs**: There was a discussion on Google's approach to converting free users to paid, with some members subscribing to YouTube Premium despite aggressive ad strategies. Concerns were raised about user trust if ads were integrated into Google's ChatGPT competitor.
- **Claude-3 Outshines GPT-4**: The community has shown enthusiasm for the new **Claude model** family, with Claude-3-Opus achieving top rank alongside GPT-4-Turbo. There's a plan to create separate leaderboards for different domains to provide clearer insights into model capabilities ([LM SysOrg's update on Claude-3](https://fxtwitter.com/lmsysorg/status/1767997086954573938)).
- **Analyzing Claude 3's New Additions**: Members discussed **Claude 3 Haiku**, a fast and affordable model, while pondering its effectiveness for replacing older systems and the potential for challenge in prompt engineering for specific tasks ([Xeophon's thoughts on usage](https://x.com/TheXeophon/status/1768047237626515662)).
- **The Challenge of Standardizing Research Literature for AI Assistance**: The conversation extended towards the difficulties in creating efficient AI literature survey assistants due to citation ambiguities, graph interpretations, and building a system to critique papers, hinting at the future directions for research in AI document parsing ([Discussion on literature survey challenges](https://arxiv.org/abs/2402.18819)).

**Links mentioned**:

- [Tweet from Xeophon (@TheXeophon)](https://x.com/TheXeophon/status/1768057955620913495>)): @felix_red_panda @karthikv792 Happy to hear your verdict! :)
- [Tweet from lmsys.org (@lmsysorg)](https://fxtwitter.com/lmsysorg/status/1767997086954573938): [Arena Update]  Our community has cast 20,000 more votes for Claude-3 Opus and Sonnet, showing great enthusiasm for the new Claude model family!  Claude-3-Opus now shares the top-1* rank with GPT-4-Tu...
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/anthropicai/status/1768018310615151002?s=46): Today we&#39;re releasing Claude 3 Haiku, the fastest and most affordable model in its intelligence class.  Haiku is now available in the API and on http://claude.ai for Claude Pro subscribers.
- [Tweet from Xeophon (@TheXeophon)](https://x.com/TheXeophon/status/1768047237626515662): Some comparisons between the Claude 3 models for paper summary. Prompt is the same, models are accessed via Poe + PDF upload.  Here, I don&#39;t like Haiku at all, its too close to the paper. I&#39;d ...

  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1217660813746241587)** (8 messages🔥): 

- **Meetup Announcement for GTC**: One member announced they will be at the **GTC** next week, inviting others to say hi in person.
- **BLAS vs. NumPy Performance Debate**: A member provided a [link](https://ashvardanian.com/posts/numpy-vs-blas-costs/) highlighting that NumPy, despite its popularity, leaves **up to 90% of BLAS performance** on the table for certain operations. [SimSIMD](https://github.com/ashvardanian/simsimd) is featured as a potential fix for this issue.
- **Skepticism About NumPy Performance Analysis**: Another member pointed out the benchmarked work is in a very small timeframe (<1µs) and that the constant overhead is higher in NumPy, indicating that using NumPy for numerous small operations might be a problem.
- **SIMD Wrappers as Practical Solutions**: A member noted that for operations with smaller vectors, it is more efficient to use a **SIMD wrapper** than to deal with the overhead of data transfer and kernel launch.
- **Focused on Messaging for Technical Choices**: There was a suggestion for more precise messaging by focusing on the rationale behind technical choices, appropriate use cases, and installation guidance rather than just listing benchmark numbers.

**Links mentioned**:

[NumPy vs BLAS: Losing 90% of Throughput](https://ashvardanian.com/posts/numpy-vs-blas-costs/): Downloaded over 5 Billion times, NumPy is the most popular library for numerical computing in Python. It wraps low-level HPC libraries like BLAS and LAPACK, providing a high-level interface for matrix...

  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1217825373170176181)** (6 messages): 

- **Inspecting Triton tl.core.tensor Objects**: A user sought advice on how to inspect **tl.core.tensor** objects in Triton, noting that regular indexing to view values produces a '0d block_type is forbidden' error.
- **Old-School Debugging Tricks**: To inspect Triton tensors, a member suggested using the environment variable `TRITON_INTERPRET=1` along with print statements as a **traditional debugging method**.
- **Video Aid for CUDA Kernel Profiling**: An informative [YouTube video](https://www.youtube.com/watch?v=LuhJEEJQgUM) was shared, explaining how to profile CUDA kernels in PyTorch and mentioning the use of `@triton.jit(interpret=True)` for debugging; however, another member noted that this approach is **deprecated**.
- **Triton Debugging Best Practices**: A member pointed to a [GitHub issue discussion](https://github.com/openai/triton/issues/517#issuecomment-1971327089) on **how to debug Triton kernels**, providing a glimpse into community methods for tackling such issues.

**Links mentioned**:

- [Lecture 1 How to profile CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM): Slides: https://docs.google.com/presentation/d/110dnMW94LX1ySWxu9La17AVUxjgSaQDLOotFC3BZZD4/edit?usp=sharingCode:  https://github.com/msaroufim/cudamodelecture1
- [How to debug kernels · Issue #517 · openai/triton](https://github.com/openai/triton/issues/517#issuecomment-1971327089): I&#39;m trying to understand exactly what each line of code in add_kernel does in the vector add tutorial. Because it&#39;s a kernel I can&#39;t use a typical step debugger to go through this function...

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1217389117667541073)** (10 messages🔥): 

- **NSight Systems is essential for multi-GPU apps**: One member explains the importance of **NSight Systems** for analyzing performance issues in complex applications with multiple GPU and CPU processes, citing its capability to address PCIe memory transfers and CPU/GPU scheduling issues.

- **Newbie in Need of CUDA Assistance**: A member is seeking help with a **CUDA** question and has posted in a discord channel. A reference link was provided but was not accessible for extracting specific information.

- **Seeking Guidance on NSight Systems**: A member questioned the usefulness of **NSight Systems** and asked for advice on metrics and educational resources. Another shared [Nvidia's lecture](https://www.nvidia.com/en-us/on-demand/session/gtcspring2021-s31617/) and a [blog post](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/) explaining the visuals of overhead and latency in NSight Systems.

- **Performance Analysis with Nsight Systems Guide**: An experienced member highlighted **Nsight Systems** for spotting bottlenecks between kernel launches and provided a personal guide on using Nvidia Visual Profiler to optimize an OpenCV application. The guide can be found [here](https://cudawarped.github.io/opencv-experiments/nbs/opencv_cuda_streams_performance_python.html).

- **Kernel Launch Overhead Confusion**: One member was concerned about the change in output when altering the execution order of two functions in CUDA, speculating that it may be due to CUDA initialization or GPU warm-up. Another confirmed that it is indeed kernel launch overhead and suggested using **Ncu** to isolate the issue.

**Links mentioned**:

- [Accelerating OpenCV with Python and CUDA streams](https://cudawarped.github.io/opencv-experiments/nbs/opencv_cuda_streams_performance_python.html): OpenCV CUDA optimization example using Python and CUDA streams. Including GPU profiling, analysis, performance tips and more!
- [CUDA Developer Tools | Intro to NVIDIA Nsight Systems | NVIDIA On-Demand](https://www.nvidia.com/en-us/on-demand/session/other2024-cudansight/.): Join NVIDIA’s Sven Middelberg for an introduction to NVIDIA Nsight Systems, a tool for performance tuning NVIDIA GPU-accelerated applications
- [Understanding the Visualization of Overhead and Latency in NVIDIA Nsight Systems | NVIDIA Technical Blog](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/): Recently, a user came to us in the forums. They sent a screenshot of a profiling result using NVIDIA Nsight Systems on a PyTorch program. A single launch of an element&#x2d;wise operation gave way to&...

  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1217477687589736548)** (1 messages): 

- **CUDA Expert Wanted for Learning App**: *Christo_allstreet* is in search of a **CUDA expert** for consultancy work on their learning application [getworldclass.app](http://getworldclass.app). Interested experts are invited to send a **Direct Message** for further details.
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1217538375351799828)** (2 messages): 

- **CUDA Toolkit Conundrum in Ubuntu 23.10**: A user reported an issue with `nvidia-cuda-toolkit` on Ubuntu 23.10 where running `compute-sanitizer` results in an error: *Unable to find injection library libsanitizer-collection.so*. Although the mentioned library exists at `/usr/lib/nvidia-cuda-toolkit/compute-sanitizer/libsanitizer-collection.so`, the tool doesn't seem to recognize it.
- **Version Mismatch Might Be the Culprit**: Another user suggested that this problem could stem from a version mismatch, noting that the latest NVIDIA toolkit supports up to Ubuntu 22.04. They recommended trying `compute-sanitizer` on Ubuntu 22.04 to determine if the issue is due to changes in folder paths in the newer OS version.
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1217561639038226543)** (4 messages): 

- **Understanding SM Architecture**: A member referenced section 4.4 in the **CUDA MODE** book, noting how an SM (Streaming Multiprocessor) executes threads in a warp following the **SIMD (Single-Instruction, Multiple-Data)** model, and raised a question regarding the individual core's responsibility for executing threads.
- **Clarification on Core-Thread Execution**: Another member clarified that a processing block within an SM - using the **GA102 SM** as an example - executes one warp at a time, which means that 32 threads can be executed concurrently with **fp32 instructions**, or 32 **int32 instructions** in two batches due to core limitations.
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1217414211177545829)** (10 messages🔥): 

- **Axolotl Configuration Key**: *iron_bound* noted a specific requirement for running **axolotl**: setting `pad_to_sequence_len: true` is essential, otherwise the software fails to initiate even with a clean clone of the repository.
- **Loss Comparison Stalls Progress**: *iron_bound* shared a [W&B report](https://api.wandb.ai/links/iron-bound/v6mxxcj2) showing test results comparing stock **axolotl vs ring-attn**, indicating that loss is not decreasing towards zero as anticipated.
- **Concern over Reporting Issues on Mobile**: *andreaskoepf* mentioned difficulty in viewing the report on mobile devices and sought clarification on whether the loss for both, the vanilla **axolotl** and **ring-attn**, were not tending towards zero.
- **Reference Run Clarification**: *iron_bound* confirmed that the baseline or reference run used for comparison was a clone of axolotl without any code modifications.
- **Flash Decoding Efforts to Resume**: *jamesmel* announced availability to continue work on flash decoding starting the following day.
- **Meeting Uncertainty**: *cataluna84* inquired about the schedule of a meeting, but no further details were provided.
- **Patch Branch for Axolotl Available**: *iron_bound* provided a link to the **ring_attention_patching** branch of **axolotl** on GitHub: [GitHub - cuda-mode/axolotl at ring_attention_patching](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching).

**Links mentioned**:

- [Ring-attn vs stock](https://api.wandb.ai/links/iron-bound/v6mxxcj2): Ran abut 100 epochs on the same system and small dataset
- [GitHub - cuda-mode/axolotl at ring_attention_patching](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching): Go ahead and axolotl questions. Contribute to cuda-mode/axolotl development by creating an account on GitHub.

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1217551604555714721)** (8 messages🔥): 

- **GPT-4 Takes On DOOM**: An [arXiv paper](https://arxiv.org/abs/2403.05468) explores **GPT-4's capabilities in playing the 1993 first-person shooter Doom**, highlighting the model's ability to reason and plan with only basic instructions and a text-based description of the game state.

- **Country Roads Take Me Home**: A series of messages evoke lyrics from the song "Take Me Home, Country Roads," referencing themes of nostalgia and nature with lines like "Life is old there, Older than the trees," and "Rolling like a breeze."

- **Meta Legal Battle over Confidential Docs**: **Meta** has filed a lawsuit against a former exec for allegedly stealing over 100 internal documents and using them for his AI data startup, Omniva. The lawsuit [details](https://cdn.arstechnica.net/wp-content/uploads/2024/03/Meta-v-Khurana-complaint-2-29-2024.pdf) regard "brazenly disloyal and dishonest conduct" during the executive's transition from Meta to Omniva.

- **Song Sentiment Interrupted**: A message briefly expressing disappointment with the simple comment "…ruined it."

- **Group Learning Initiative**: A member mentions a collaborative effort involving three individuals embarking on an educational journey from "lecture 1."

**Links mentioned**:

- [Will GPT-4 Run DOOM?](https://arxiv.org/abs/2403.05468): We show that GPT-4&#39;s reasoning and planning capabilities extend to the 1993 first-person shooter Doom. This large language model (LLM) is able to run and play the game with only a few instructions...
- [Meta sues “brazenly disloyal” former exec over stolen confidential docs](https://arstechnica.com/tech-policy/2024/03/meta-sues-brazenly-disloyal-former-exec-over-stolen-confidential-docs/): Meta&#39;s former exec allegedly shared data center secrets with a shadowy startup.

  

---



**DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1217390277103190068)** (1 messages): 

- **Assistant's Mars Explanation Missing Musk's Flair**: The assistant's reply was praised for being informative and covering various aspects of why we should go to Mars, but it failed to fully comply with the user's instruction to *express it like Elon Musk*. Although the reply reflects Musk's views on Mars exploration, it lacks his specific style and tone. **Rating given was [[7]]**.
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1217471622924206150)** (8 messages🔥): 

- **MunichNLP Meetup Inquiry**: A member inquired about interest in a Munich meetup on April 11th to discuss **DiscoLM** but received no direct commitment to speak at the event.
- **DiscoLM Model's German Fine-Tuning Question**: A member questioned the **DiscoLM-mixtral-8x7b-v2** model's fine-tuning on German datasets, to which another replied that it wasn't trained on a significant amount of German data, redirecting to the extensive training details of the [DiscoLM 70b model](https://huggingface.co/DiscoResearch/DiscoLM-70b).
- **AI Tinkerers in Berlin**: Members discussed the upcoming **AI Tinkerers event** in Berlin on March 21st, sharing enthusiasm and an [event link](https://berlin.aitinkerers.org/) for a community gathering of technology enthusiasts.
- **Seats Filling Up for AI Tinkerers**: The same member mentioned that only 8 seats were left for the **AI Tinkerers** event, indicating high interest and limited availability.
- **Clarity on German Dataset Usage**: A member clarified their own confusion around the presence of German data in the instruction fine-tuning datasets, asking for specifics on the percentage of German-language data used.

**Links mentioned**:

- [
AI Tinkerers - Berlin
](https://berlin.aitinkerers.org/): no description found
- [DiscoResearch/DiscoLM-70b · Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM-70b): no description found

  

---


**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1217400001525973034)** (1 messages): 

- **Creative Writing Benchmark Testing Success**: A member has announced the successful implementation of a creative writing benchmark prototype, indicating that it offers reasonable rankings. Interested parties can try it out on [this branch of the EQ-Bench repository on GitHub](https://github.com/EQ-bench/EQ-Bench/tree/creative_writing).

**Links mentioned**:

[GitHub - EQ-bench/EQ-Bench at creative_writing](https://github.com/EQ-bench/EQ-Bench/tree/creative_writing): A benchmark for emotional intelligence in large language models - GitHub - EQ-bench/EQ-Bench at creative_writing

  

---


**DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1217552936079720458)** (3 messages): 

- **Seeking German Precision**: A member inquired about the **best embedding and re-ranking for German**, specifically for use with German legal texts.
- **Hunting for Benchmarks**: The same member also asked if there exists a benchmark for embedding models in German.
- **Benchmarking German Embeddings**: Another member suggested using the "GermanQuAD" evaluation task in the **MTEB** Python package or looking into recent German additions from **JinaAI**.
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1217452232207765525)** (2 messages): 

- **Local Model Replication Inquiry**: A member asked how to replicate a demo's output locally using their own code. Their current setup includes a one-shot with settings for temperature, top_p, max_tokens, and they provided a code snippet to illustrate their approach.

- **Questions on Command Repetition**:
A follow-up question by the same member queried whether they should repeat a command for every user message or include it only once in the system's content, seeking guidance on the best practice for command structure.
  

---



**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1217843643541622935)** (12 messages🔥): 

- **Haiku's Cost-Efficiency Breakthrough**: Haiku's document describer has been hailed for performing **vision-to-text** on visually complex documents at an economical cost.

- **Debating Visual Document Processing**: Members compare Haiku's capabilities with GPT-vision, concluding Haiku is not superior, while another system named Opus is noted to be better than Haiku.

- **Content Filter Hurdles with Visual Docs**: The discussion reveals that **content filtering** issues have arisen when processing documents, particularly those containing equations, causing incomplete analysis mid-document.

- **Claude's Content Filtering Quirks Noted**: It was mentioned that **Claude** historically has had problems with iffy content filtering, which may relate to the problems experienced by other members with document processing.
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1217441897161953381)** (6 messages): 

- **Zero-click Worms Target GenAI-Powered Apps**: A new paper titled "ComPromptMized: Unleashing Zero-click Worms that Target GenAI-Powered Applications" has been shared, highlighting vulnerabilities in GenAI-powered applications through prompt injection. The paper demonstrates attacks on email assistants using various models, such as **Gemini Pro, ChatGPT 4.0, and LLaVA**. [Read the full paper](https://sites.google.com/view/compromptmized)

- **Seeking a Model Comparison Framework**: In a quest for the best model to serve as a code assistant, a member inquires about a framework to compare the effectiveness of models such as **Mistral** or **Llama2**.

- **Choosing Models Based on Benchmarks**: Another member pointed out the existence of benchmarks for model comparisons, but advised that such benchmarks should be considered with a grain of accuracy.

- **Leaderboard for Model Comparisons**: To compare models, a member suggests using the **Leaderboard available at** [chat.lmsys.org](https://chat.lmsys.org), which provides a competitive ranking of different models.

**Links mentioned**:

[ComPromptMized](https://sites.google.com/view/compromptmized): Stav Cohen Technion - Israel Institute of Technology 

  

---



**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1217539473781293056)** (2 messages): 

- **Seeking Multimodal Model Gurus**: Soniajoseph_ is calling for collaborators skilled in **open source interpretability** of multimodal models. Details can be found in their [Twitter post](https://twitter.com/soniajoseph_/status/1767963316943728779) and a cross-posted article on [LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic) from the [AI Alignment Forum](https://alignmentforum.org/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic).

- **Join the Interpretability Crusade**: Those interested can join the related Discord through this [invitation link](https://discord.gg/2U2N8QmPmJ).

- **Collaboration Hub Tip**: Rusch drops a hint for a potential collaboration hub suitable for such projects, sharing an alternative [Discord invitation](https://discord.gg/bDV7kDrKjE).

**Links mentioned**:

- [Join the Mech Interp Discord Discord Server!](https://discord.gg/bDV7kDrKjE): Check out the Mech Interp Discord community on Discord - hang out with 907 other members and enjoy free voice and text chat.
- [Laying the Foundations for Vision and Multimodal Mechanistic Interpretability &amp; Open Problems — LessWrong](https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic): Behold the dogit lens. Patch-level logit attribution is an emergent segmentation map. Join our Discord here. …

  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1217382172432793641)** (1 messages): 

- **In Search of Speed: Phi 2 Inference Optimizations**: A member inquired about the fastest way to perform inference with **Phi 2** or its tunes on an **A100 40GB** GPU, expressing a desire to process "LOTS OF DATA." They requested feedback on the best frameworks to use among **vLLM**, **Olama**, **Axolotl**, and others, and wondered if **quantization** could be beneficial for speed.
  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1217447520603471944)** (2 messages): 

- **Meet Devin, the Autonomous Software Engineer**: A video titled *Devin The World’s first AI Software Engineer* was shared, showcasing the abilities of an AI named Devin that is claimed to be fully autonomous. Further details can be found on the [Cognition Labs blog](https://www.cognition-labs.com/blog).

- **Function Calling with Hermes 2 Pro 7B**: The chat included a [YouTube video](https://youtu.be/PzaidfqDtGI) that demonstrates function calling with the **Hermes 2 Pro 7B** model. Interested viewers can learn more and delve into the specifics via a [GitHub repository dedicated to Hermes Function Calling](https://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels).

**Links mentioned**:

- [Devin The Worlds first AI Software Engineer](https://www.youtube.com/watch?v=NSPtrrUQ_fw): Devin is fully autonomous software engineerhttps://www.cognition-labs.com/blog
- [Lets Function Call with Hermes 2 Pro 7B](https://www.youtube.com/watch?v=PzaidfqDtGI): lets do function calling with Hermes 2 Pro 7Bhttps://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels

  

---



**AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1217550187204055070)** (1 messages): 

- **New Kid on the Code Block**: Cognition has unveiled **Devin**, an AI positioned as the world's **first fully autonomous AI software engineer**. They claim Devin can handle complex engineering tasks, learn over time, and correct its own mistakes as outlined in [Scott Wu's blog post](https://www.cognition-labs.com/blog).

**Links mentioned**:

[Blog](https://www.cognition-labs.com/blog): no description found

  

---


**AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1217633669275975731)** (1 messages): 

- **Voice + AI Event Bot Contest**: A contest has been announced as a fun addition to the upcoming **Voice + AI event** next week, inviting participants to build creative projects. Details for "**The Most Interesting Bot In the World Contest**" can be found in their [Notion page](https://dailyco.notion.site/The-Most-Interesting-Bot-In-the-World-Contest-34f466fa7d2a4574a4cb91df163b37a3).

**Links mentioned**:

[Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://dailyco.notion.site/The-Most-Interesting-Bot-In-the-World-Contest-34f466fa7d2a4574a4cb91df163b37a3): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team

  


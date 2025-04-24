---
id: 1a9ce6e6-a854-47d3-97a5-1649bdd032f4
title: CodeLLama 70B beats GPT4 on HumanEval
date: '2024-01-30T21:10:01.398467Z'
original_slug: ainews-codellama-70b-beats-gpt4-on-humaneval
description: >-
  Meta AI’s surprise release of **Code Llama 70B** has energized the open-source
  community, with rapid integration into platforms like Ollama and MLX for local
  use. Meanwhile, **Nous Research** unveiled the **Activation Beacon** project,
  a breakthrough enabling unlimited context length in large language models,
  potentially transforming retrieval-augmented generation and dense retrieval
  methods.
tags:
  - meta
  - mistral
  - nous-research
  - openai
  - mit
  - rwkv
  - lmstudio
  - codellama-70b
  - miqu
  - mistral-medium
  - mistral-7b
  - aphrodite-engine
  - mixtral
  - flatdolphinmaid
  - noromaid
  - rpcal
  - eagle-7b
  - rwkv-v5
  - openhermes2.5
  - nous-hermes-2-mixtral-8x7b-dpo
  - imp-v1-3b
  - bakllava
  - moondream
  - qwen-vl
  - gpt-4
  - gpt-4-scholar
  - smoothquant
  - open-source
  - large-language-models
  - model-quantization
  - context-length-extension
  - retrieval-augmented-generation
  - model-benchmarking
  - gpu-performance
  - prompt-engineering
  - ethical-ai
  - fine-tuning
  - multimodal-models
  - code-generation
  - model-optimization
  - model-format-gguf
companies:
  - meta
  - mistral
  - nous-research
  - openai
  - mit
  - rwkv
  - lmstudio
models:
  - codellama-70b
  - miqu
  - mistral-medium
  - mistral-7b
  - aphrodite-engine
  - mixtral
  - flatdolphinmaid
  - noromaid
  - rpcal
  - eagle-7b
  - rwkv-v5
  - openhermes2.5
  - nous-hermes-2-mixtral-8x7b-dpo
  - imp-v1-3b
  - bakllava
  - moondream
  - qwen-vl
  - gpt-4
  - gpt-4-scholar
  - smoothquant
topics:
  - open-source
  - large-language-models
  - model-quantization
  - context-length-extension
  - retrieval-augmented-generation
  - model-benchmarking
  - gpu-performance
  - prompt-engineering
  - ethical-ai
  - fine-tuning
  - multimodal-models
  - code-generation
  - model-optimization
  - model-format-gguf
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 1/29/2024. We checked **21** guilds, **311** channels, and **8276** messages for you. Estimated reading time saved (at 200wpm): **605 minutes**.

The [surprise release of CodeLlama](https://ai.meta.com/resources/models-and-libraries/llama-downloads/?utm_source=twitter&utm_medium=organic_social&utm_campaign=codellama&utm_content=image) from Meta AI is an incredible gift to open source AI:

 ![image.png](https://assets.buttondown.email/images/d7f89536-22c8-4c46-9d1c-048dc42ebf64.png?w=960&fit=max) 

As  can be expected, the community has already got to work [putting it on Ollama and MLX](https://x.com/reach_vb/status/1752016793558823160?s=20) for you to run locally.

---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Miqu Model Sparks Intrigue and Debate**: The **Miqu model** has sparked debates among users like `@itsme9316` and `@rombodawg`, wondering if it's a leak from **Mistral Medium** or a fine-tuned version of **Llama-2-70b**. Concerns about **AI ethics** and the future risks of AI were discussed, including the misuse of powerful models and the challenges of achieving proper alignment. 

- **Aphrodite Engine's Performance Highlighted**: Users shared their experiences with the **Aphrodite engine**, indicating it offers impressive performance at 20 tokens per second on **A6000 GPUs** with configurations like `--kv-cache-dtype fp8_e5m2`, though concerns were raised about GPU utilization and support for model splitting.

- **Role-Playing AI Models Discussed for Repetitiveness and Performance**: In role-playing settings, models like **Mixtral** and **Flatdolphinmaid** showed challenges with repetitiveness, while **Noromaid** and **Rpcal** performed better. **ChatML** with Mixtral Instruct was recommended by `@ks_c` for role-play, highlighting the importance of Direct Prompt Optimization (DPO) to improve AI character responses.

- **Learning and Fine-Tuning on ML/DL**: `@dirtytigerx` advises that CUDA programming isn't essential for starting with ML/DL, recommending the [Practical Deep Learning for Coders course by fast.ai](https://course.fast.ai/) for a solid foundation. Discussions included VRAM requirements for fine-tuning large models like **Mistral 7B**, and exploring advanced optimizer technology like *Paged 8bit lion* or *adafactor* for efficient VRAM usage.

- **Enhancing GitHub Project Navigation and Understanding Codebases**: Guidance on understanding GitHub projects for contributions was offered, with `start.py` in [tabbyAPI project](https://github.com/theroyallab/tabbyAPI/blob/main/main.py#L577) highlighted as an example entry point. The importance of IDEs or editors with language servers for efficient code navigation was discussed, along with the release of **Code Llama 70B** on Hugging Face, emphasizing its potential in code generation and debugging across several programming languages.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Activation Beacon Revolutionizes LLM Context Length**: The **Activation Beacon** project introduces a significant advancement, enabling **unlimited context length** in LLMs through "global state" tokens. This development, as shared by `@cyrusofeden`, could dramatically alter the approach to Dense Retrieval and Retrieval-augmented LLMs with its details available on [GitHub](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon).

- **Eagle-7B Surpasses Mistral with RWKV-v5 Architecture**: **Eagle-7B**, based on the RWKV-v5 architecture, has been highlighted as outperforming even Mistral in benchmarks, presenting a promising open-source, multilingual model with efficiency gains. It boasts lower inference costs while maintaining high English performance ([source](https://x.com/rwkv_ai/status/1751797147492888651?s=46&t=MMOnaQf8LPGi8UOQi3-whw)).

- **OpenHermes2.5 Touted as Ideal for Consumer Hardware**: Recommendations for **OpenHermes2.5** emerge as ideal for running on consumer hardware, attributed to its gguf or gptq quantization, positioning it as a leading open-source LLM for question-answering. Further discussions also explore the **Nous Hermes 2 Mixtral 8x7B DPO** on [Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO).

- **Exploration into Multimodal and Specific Domain LLMs**: Discussions around using the **IMP v1-3b model** for classification tasks in orbital imagery suggest a growing interest in multimodal models and their application in specific domains. The dialogue extends to comparisons with other models like **Bakllava** and **Moondream**, highlighting the prowess of **Qwen-vl** in this space.

- **Collective Ambition for Centralized AI Resources**: The community's efforts in centralizing AI resources and research, spearheaded by `@kquant`, demonstrate a recognized need for a consolidated repository. This initiative aims to streamline access to AI papers, guides, and training resources, inviting contributions to enhance the resource pool for enthusiasts and professionals alike.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **GPU Choice for LLM Work**: Despite older P40/P100 GPUs having more VRAM, the newer **4060Ti** is recommended for training Large Language Models (LLM) due to receiving current updates from Nvidia. Participants, including `@heyitsyorkie`, discussed GPU preferences, noting the importance of updates over GRAM speed.

- **Model Compatibility and Format Updates**: LMStudio's shift from GGML to **GGUF format** for models, as notified by `@heyitsyorkie`, indicates a move towards more current standards. Further, discussions highlighted the utility of **dynamically loading experts** in models for MoE (Mixture of Experts) applications within LMStudio.

- **Performance Issues and UI Challenges in LMStudio Beta**: Users `@msz_mgs` and `@heyitsyorkie` reported **performance lag** when inserting lengthy prompts and navigational difficulties within the UI. These issues are recognized and slated for resolution in future updates.

- **Exploration of VRAM and GPU Utilization**: Discussions led by `@aswarp` and `@heyitsyorkie` explored the unexpected **VRAM overheads** for running models in LMStudio, highlighting nuances in offloading models to GPUs including the RTX 3070 and **4090**. Rumors about an upcoming **RTX 40 Titan** with 48GB VRAM stirred community discussions, linking to broader market strategies and repurposing GPUs for AI.

- **Beta Updates and Cross-Platform Compatibility Queries**: The announcement of **Beta V2** indicated progress on VRAM estimates and compatibility issues. Queries about **Mac Intel versions**, **Linux appimage availability**, and **WSL (Windows Subsystem for Linux)** support underscored the community's interest in broadening LMStudio's platform compatibility.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **RAG Demystified**: In response to @aiguruprasath's query, **Retrieval Augmented Generation (RAG)** was explained by @darthgustav. as AI with the capability to directly access knowledge for searching data or semantic matching, enhancing its decision-making processes.

- **Prompt Engineering Evolves**: @madame_architect shared a prompt engineering technique: `"Please critique your answer. Then, answer the question again."` This method, inspired by a significant research paper, aims to notably improve the quality of AI outputs in complex querying.

- **Code Generation Optimized**: Frustrations with **ChatGPT-4** failing to generate accurate code were addressed by specifying tasks, languages, and architectures in requests, a tip shared by @darthgustav. to assist @jdeinane in achieving better code generation outcomes.

- **Navigating AI Ethics via Conditional Imperatives**: An innovative ethical moderation strategy involving a 2/3 pass/fail system based on utilitarianism, deontology, and pragmatism was discussed. @darthgustav. also introduced using conditional imperatives directing AI to consult **LEXIDECK_CULTURE.txt** for resolving ethical dilemmas, enhancing AI’s decision-making framework.

- **GPT-4’s Limits and Quirks Uncovered**: Conversations in the community highlighted **GPT-4 Scholar AI** limitations, such as error messages indicating the need for a premium subscription for advanced research, and the unresolved struggle with accessing beta features. Users shared tips on managing conversation lengths and error workarounds, emphasizing community-based troubleshooting and innovation in AI interaction strategies.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **SmoothQuant Boosts LLM Throughput**: A discussion centered around **SmoothQuant**, a new quantization method from MIT's Han's lab, aiming at LLMs beyond 100 billion parameters, yielding a **50% throughput improvement** for a 70B model as noted in the [Pull Request #1508](https://github.com/vllm-project/vllm/pull/1508). However, integrating new technologies like SmoothQuant into systems like vLLM could be challenging due to feature bloat.

- **Exploring Efficient Processing of Long Contexts in LLMs**: Insights on managing the **limited context window of LLMs** highlighted the **Activation Beacon** and **RoPE** enhancements for improved long context processing. This exploration suggests a route to maintaining performance when dealing with extended contexts, an essential factor for technical applications.

- **Quantization Confusion and Potential on HuggingFace**: A misattributed 70B model on HuggingFace led to discussions about quantization's role and its impact, with specific mentions of the utility and misconceptions surrounding quantized models like Miqu and Mistral Medium. The complexity of model quantization and its implications on memory efficiency surfaced further with **LoftQ**, where out-of-memory errors were reported despite expected benefits.

- **Model Training and Inference Efficiencies Unpacked**: Conversations traversed effective hardware setups for training, notably **NVIDIA H100 SXM GPUs**, and inference strategies, emphasizing the trade-offs in quantization and the necessity of hardware benchmarking before acquisition. This included a dive into **vLLM** for serving fp16 models and the hardware-dependent nature of performance, underlining the variegated paths to optimizing model operations.

- **Fine-Tuning and Configurations for Enhanced Model Performance**: Various contributions highlighted the importance of appropriate fine-tuning and model configurations, pointing to resources like [FlagEmbedding Fine-tuning documentation](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/llm_embedder/docs/fine-tune.md) and [LlamaIndex guidelines](https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning.html). The dialogue underscored the necessity for precise setup in achieving improved outputs and more meaningful embedding representations, alongside the innovative approach of creating synthetic instruction datasets for heightened model effectiveness.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Clarification on Mistral and "Leak" Claims**: There was confusion whether a newly emerged model was a leaked version of Mistral, eventually identified not to be Mistral or a leak but possibly a LLaMa model fine-tuned with Mistral data. The community called for official clarification from Mistral staff on the matter.
  
- **Quantization Affects AI Model Performance**: Dense models, speculated to be close approximations of Mistral, could suffer significantly from quantization effects, highlighting the technical nuances of model performance degradation.

- **Challenges in Adapting Mistral for New Languages**: Concerns were raised about the vocabulary mismatch when training Mistral in a new language, with suggestions that starting from scratch might not be feasible due to resource requirements, and pretraining might be necessary.

- **VRAM and Token Limitations for Mistral Models**: Users reported inconsistencies between VRAM requirements stated by Mistral AI 7B v0.1 models and actual user experiences. Further, token limits for Mistral API text embeddings were clarified, with a max token limit confirmed to be 32k for tiny/small/medium models and 8192 for the embedding API.

- **API and Client Discussions Indicate Expansion and Technical Interest**: Discussions about the **maximum rate limit** for Mistral APIs in production indicated an interest in expansion, with initial limits starting at 2 requests/second. A proposal for a new Java client, **langchain4j**, for Mistral AI models showcases community-led development efforts outside the primary documentation.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **New Research and Critical Views on AI Policy Unveiled**: [AI Techniques in High-Stakes Decision Making](https://arxiv.org/abs/2401.14446) introduces insights into deploying AI in critical contexts. Meanwhile, the Biden Administration's [AI regulation fact sheet](https://www.whitehouse.gov/briefing-room/statements-releases/2024/01/29/fact-sheet-biden-harris-administration-announces-key-ai-actions-following-president-bidens-landmark-executive-order/) draws critique for vague policy directions and the proposal of AI/ML education in K-12 schools, raising concerns about effectiveness and implementation.

- **Deep Dive into Model Efficiency and Algorithmic Innovations**: Discussions in the #[research](https://discord.com/channels/729741769192767510/747850033994662000/1201445874518208522) channel revolve around model efficiency, with techniques like seed-based vocab matrices and gradient preconditioning under examination. Softmax bottleneck alternatives and data preprocessing challenges are hot topics, alongside the exploration of attention mechanisms, alluding to the complexity of model optimization and the pursuit of alternatives.

- **Introducing TOFU for Model Unlearning Insights**: TOFU benchmark's introduction, as found in [arXiv:2401.06121](https://arxiv.org/abs/2401.06121), sets the stage for advanced machine unlearning methodologies, sparking interest in the efficacy of gradient ascent on 'bad data' as an unlearning strategy. This benchmark aims to deepen the understanding of unlearning in large language models through the use of synthetic author profiles.

- **Apex Build Insights Offer Potential Optimizations**: In [#gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1201611721375223859), discussions reveal challenges and novel solutions for building **apex**, especially on AMD's MI250xs architectures. Highlighted solutions include specifying architectures to reduce build time and exploring CUDA flags for optimization, presenting a practical conversation focused on improving development efficiency.

- **AI Community Eager for Experimentation and Offline Flexibility**: A notable enthusiasm for new work and exploratory tests is evident, complemented by a practical inquiry about downloading tasks for offline use. This interest underscores the community's drive for innovation and the adaptability of tools for diverse usage scenarios.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **AI Etiquette Guide for PR Tagging**: Community members were reminded not to tag individuals for PR reviews if they're unrelated to the repo, sparking a discussion about community etiquette and fairness in PR handling.

- **Patience and Strategy in AI Development**: Discussions around the development of **continuous neural network models** and the successful implementation of a **deepfake detection pipeline** underlined the community's focus on innovative AI applications and patience in the PR review process.

- **AI Models for Political Campaigns Raise Ethical Questions**: A dialog unfolded about the use of AI in elections, suggesting both controversial methods like fraud and ethical approaches like creating numerous videos of candidates. This discussion extended to inquiries about technical specifics and recommendations for **text-to-image diffusion models**.

- **Deep Learning's Impact on Medical Imaging**: Articles shared from IEEE highlighted breakthroughs in **Medical Image Analysis** through deep learning, contributing to the guild's repository of important AI developments in healthcare.

- **Innovative Projects and Demonstrations in AI**: Community members showcased various projects, including a **Hugging Face Spaces demo for image models**, a **resume question-answering space**, and an app that converts **Excel/CSV files into database tables**. A YouTube video demonstrating **Mistral model performance on Apple Silicon** was also featured.

- **Technical Challenges and GPU Acceleration in NLP and Computer Vision**: Discussions in the **NLP** and **Computer Vision** channels focused on achieving GPU acceleration with **llama-cpp** and troubleshooting errors in **fine-tuning Donut docvqa**, reflecting the technical hurdles faced by AI engineers in their work.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **ColBERT Sparks Interest Over Traditional Embeddings**: [Simon Willison's exploration](https://til.simonwillison.net/llms/colbert-ragatouille) of **ColBERT** has ignited a debate on its effectiveness against conventional embedding models. Despite growing curiosity, the lack of direct comparative data between ColBERT and other embedding models was highlighted as a gap needing closure.
  
- **Simon Willison Addresses AI's Human Element**: A [feature on Simon Willison](https://www.theregister.com/2024/01/24/willison_ai_software_development) delves into how AI intersects with software development, emphasizing the importance of human oversight in AI tools. This has spurred further discussion among engineers regarding AI's role in augmenting human capabilities in software development.
  
- **"Arc Search" Anticipated to Alter Web Interactions**: The introduction of **Arc Search**, a novel iOS app, could potentially revolutionize web browsing by offering an AI-powered search that compiles web pages based on queries. The community speculates this could present a challenge to the dominance of traditional search engines.
  
- **Voyage-Code-2 Touted for Superior Code Retrieval**: The unveiling of **Voyage-Code-2** was met with enthusiasm, with promises of improved performance in code-related searches. The conversation, primarily driven by `@swyxio`, revolved around the model's potential benchmarking on MTEB and its implications for embedding specialization.
  
- **Claude Models by Anthropic Underrated?**: The debate over **Anthropic's Claude models** versus those by OpenAI unveiled a consensus that Claude might be underappreciated in its capabilities, particularly in summarization and retrieval tasks. This discussion shed light on the need for a more nuanced comparison across AI models for varied applications.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **MIQU Misunderstanding Resolved**: @aiui's speculation about the **MIQU model** being a **Mixtral medium leak** was addressed by @sebastian.bodza, clarified with a [tweet by Nisten](https://twitter.com/nisten/status/1751841882831716578) and further information on the model using a **LLaMA tokenizer**, as shown in another [tweet](https://twitter.com/Nota_Cant/status/1751861787170148368).

- **German Fine-tuning Frustrations and Triumphs**: @philipmay and @johannhartmann exchanged experiences on fine-tuning models for German, with notable attempts on **Phi 2** using the OASST-DE dataset and enhancing **Tiny Llama** with German DPO, documented in [TinyLlama-1.1B-Chat-v1.0-german-dpo-Openllm_de.md](https://gist.github.com/johannhartmann/6cb0fee8103869e6e58d7e1956ce9c99). Questions about a German Orca DPO dataset led to revealing an experimental dataset on Hugging Face.

- **Breaking New Grounds with WRAP**: @bjoernp shared insights into **Web Rephrase Augmented Pre-training (WRAP)**, an approach aimed at improving data quality for language models, as detailed in a recent paper ([Web Rephrase Augmented Pre-training](https://arxiv.org/abs/2401.16380)) by Apple.

- **Challenges with CodeT5 and Training Data**: In the saga of implementing Salesforce's **CodeT5** embeddings, @sebastian.bodza faced technical hurdles and engaged in discussions about the creation of "hard negatives" and the development of training prompts for text-generation, sharing a specific [notebook for review](https://github.com/SebastianBodza/Embedding_Training/blob/main/05_preprocess_texts.ipynb).

- **Clarifying Retrieval Model Misconceptions**: The discourse on generating a passage retrieval dataset for RAG pointed out common misconceptions regarding the selection of "hard negatives" and dataset construction, with an aim to clarify the purpose and optimal structure for such data, indicative of ongoing improvements and contributions, as seen in embedded [GitHub resources](https://github.com/telekom/wp-rag-dpo/blob/main/04_it01_extract_positive_answers.ipynb).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **LangChain Sparks New Developments**: @irfansyah5572 sought assistance on answer chains with LangChain, embedding a [GitHub link](https://github.com/langchain-ai/chat-langchain/blob/master/chain.py) to their project, while @oscarjuarezx is pioneering a PostgreSQL-backed book recommendation system utilizing LangChain's capabilities.

- **Enhancing PDF Interactions with LangChain**: @a404.eth released a [YouTube tutorial](https://www.youtube.com/watch?v=xFWllDS6ZRw) on developing a frontend for PDF interactions, marking the second installment in their educational series on leveraging LangChain and related technologies.

- **Cache and Efficiency in Focus**: Technical discussions echoed around `InMemoryCache`'s role in ameliorating inference times for LlamaCPP models, underscored by a shared [GitHub issue](https://github.com/langchain-ai/langchain/issues/2784) but highlighted challenges in cache performance.

- **Lumos Extension Lights Up Web Browsing**: @andrewnguonly introduced **Lumos**, an open-source Chrome extension powered by **LangChain** and **Ollama**, designed to enrich web browsing with local LLMs, inviting community feedback via [GitHub](https://github.com/andrewnguonly/Lumos) and [Product Hunt](https://www.producthunt.com/posts/lumos-4).

- **Tutorials Drive LangChain Mastery**: New tutorials like Ryan Nolan's beginner-friendly guide to Langchain Agents for 2024 and @a404.eth’s deep-dive into RAG creation using LangChain emphasize community-building knowledge and skill development, accessible via [YouTube](https://www.youtube.com/watch?v=WVUITosaG-g&ab_channel=RyanNolanData) and [YouTube](https://www.youtube.com/watch?v=xFWllDS6ZRw) respectively.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **New Complex Classifications Approach with LLMs**: @KarelDoostrlnck introduces a novel method utilizing Large Language Models (LLMs) for addressing complex classifications that encompass thousands of categories. This technique involves generating a set of predictions followed by retrieval and re-ranking of results, detailed in the [IFTTT Announcement](https://twitter.com/llama_index/status/1752008109835559123).

- **LlamaIndex.TS Receives a Refresh**: Upgrades and enhanced documentation have been implemented for LlamaIndex.TS, promising improvements for users. The update announcement and details are shared in this [tweet](https://twitter.com/llama_index/status/1752075208905896265).

- **$16,000 Up For Grabs at RAG Hackathon**: A special in-person hackathon focusing on Retriever-Augmented Generation (RAG) technology, dubbed the LlamaIndex RAG-A-THON, announces prizes totaling $16,000. Event specifics can be found in the [hackathon details](https://twitter.com/llama_index/status/1752086703437955199).

- **Exploration of Llama2 Commercial Use and Licensing**: Community members discuss the potential for commercial utilization of Llama2, with detailed licensing information suggested to be reviewed on Meta's [official site](https://ai.meta.com/llama/) and in a [deepsense.ai article](https://deepsense.ai/llama-2).

- **Advanced Query Strategy Enhancements with LlamaPacks**: @wenqi_glantz, in collaboration with @lighthouzai, evaluates seven LlamaPacks showcasing their effectiveness in optimizing query strategies for specific needs. The evaluation results are available in this [tweet](https://twitter.com/llama_index/status/1752131958552080650).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **MagViT2 Faces Training Hurdles**: Technical discussion highlighted difficulties with **MagViT2**, where pixelated samples emerged after 10,000 training steps. Solutions suggested include adjusting the learning rate and loss functions, referencing the [MagViT2 PyTorch repository](https://github.com/lucidrains/magvit2-pytorch).

- **Controversy Sparked by Nightshade**: Drhead criticized the irresponsible release of **Nightshade**, while others like Astropulse underscored concerns regarding the tool's long-term effects and effectiveness. Strategies to counteract Nightshade's potential threats were also debated, including finetuning models and avoiding targeted encoders.

- **Debating the Need for New Data in AI**: The necessity of new data for enhancing AI models was questioned, with pseudoterminalx and mfcool arguing that the focus should instead be on data quality and proper captioning to improve model performance.

- **Activation Beacon Promises Context Length Breakthrough**: A discussion on "Activation Beacon" suggests it could enable unlimited context lengths for LLMs, cited as a significant development with an LLaMa 2 model achieving up to 400K context length after training. [Read the paper](https://arxiv.org/pdf/2401.03462.pdf) and [check the code](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon).

- **Optimizing Data Storage for Image Datasets**: Conversations around datasets' storage questioned the efficiency of parquet vs. tar files, with a hybrid method of using parquet for captions and tar for images being touted as a potentially optimal solution. **Webdatasets** and **tarp** were recommended for high-performance Python-based I/O systems for deep learning, with links shared for further investigation: [Webdataset GitHub](https://github.com/webdataset/webdataset), [Tarp GitHub](https://github.com/webdataset/tarp).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Choosing the Right AI Model Gets Easier**: Users explored the best scenarios for utilizing AI models like **Gemini**, **GPT-4**, and **Claude**, finding [technical FAQs](https://blog.perplexity.ai/technical-faq/what-is-the-difference-between-gpt-4-and-claude-2) helpful for making informed decisions.
  
- **Perplexity's Library Feature**: An issue where the library sidebar wasn't showing all items was discussed, concluding it’s a feature, not a bug, displaying only the last eight threads/collections.

- **No Crypto Tokens for Perplexity Project**: Queries regarding a potential cryptocurrency tied to Perplexity were addressed, confirming no such token exists.

- **Innovative Uses for AI in Applications**: Developers are creatively integrating AI into projects such as a delivery-themed **pomodoro app**, using AI to generate names and addresses, highlighting AI's versatility in enhancing app functionalities.

- **Perplexity API Awaits Custom Stop Words Feature**: The integration of **custom Stop Words** in the **pplx-api** with [zed.dev editor](https://zed.dev/) is anticipated, aiming to provide a rich alternative to the default **OpenAI models** for enhanced 'assistant' features in editing applications.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1189498204333543425) Discord Summary

- **Triton: Balancing Control and Complexity**: `@tvi_` emphasized that **Triton** offers a sweet spot for AI engineers, providing more control than **PyTorch** with less complexity than direct **CUDA**, ideal for those seeking efficiency without delving deep into CUDA's intricacies.

- **An Odyssey of CUDA Optimization for RGB to Grayscale**: Amidst attempts to optimize an **RGB to Grayscale conversion**, `@artste` and `@zippika` navigated through struct usage, **vectorized operations**, and memory layout variations, with a journey full of trials including unexpected performance results and the discovery that `__forceinline__` can significantly improve kernel performance.

- **In Search of CUDA Wisdom**: After facing a confusing performance benchmark, `@andreaskoepf` suggested that `@zippika` delve into **memory optimization techniques** and vectorized memory access to enhance CUDA performance, supported by insights from a [NVIDIA developer blog](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/).

- **Unraveling the Mysteries of PMPP**: Queries surrounding **Practical Massively Parallel Programming** concepts, from unclear questions in Chapter 2 to understanding memory coalescing and banking in Chapter 6, were addressed, showcasing the community's effort to demystify dense technical material for collective advancement.

- **Learning Continues on YouTube**: `@andreaskoepf` shared a [new educational video](https://youtu.be/4sgKnKbR-WE?si=J-B0kHqknRXhE7e_) pertaining to AI engineering, fostering the community's continuous learning outside the written forums.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Writing Woes and Wins**: Engineers in the guild voiced frustrations and hopes around **writing assistant tools**. While `@an1lam` lamented the limitations of current tools and the lack of alternatives like lex.page, `@calclavia` offered a glimmer of hope by introducing [jenni.ai](https://jenni.ai) for academic writing, implying it may address some of these concerns.

- **Startups Struggle and Strategize with Cloud Costs**: The economic challenges of utilizing **Google Workspace** for startups were highlighted by `@frandecam`, whereas `@dare.ai` countered with a solution: the [Google Cloud for Startups program](https://cloud.google.com/startup/ai?hl=en), which promises up to $250,000 in Google Cloud credits but comes with a warning of potential application delays.

- **Gorilla Grabs Attention with Open Source Solution**: The **Gorilla OpenFunctions project** stirred interest among engineers, offering an open-source alternative for executing API calls via natural language, detailed through a [blog post](https://gorilla.cs.berkeley.edu/blogs/4_open_functions.html) and a [Github repository](https://github.com/philschmid/open-source-function-calling/blob/main/gorilla-functions.ipynb). This innovation aims to streamline the integration of API calls with Large Language Models (LLMs).

- **Copyright Quandaries for Creative AI**: `@jxnlco` broached the issue of a **prompt investing vision model** bumping up against copyright restrictions in its attempt to read complex labels, highlighting a common challenge for developers navigating the interface between AI outputs and intellectual property rights.

- **Mistral Medium Buzz**: A single message from thebaghdaddy queried the community's take on **Mistral Medium's** potential to ascend as a rival to **GPT-4**, sparking curiosity about its capabilities and position in the ever-evolving LLM landscape.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Frankenstein Model Comes to Life**: `@nisten` has embarked on a pioneering experiment by combining all **70b CodeLlamas** into a composite **[BigCodeLlama-169b](https://huggingface.co/nisten/BigCodeLlama-169b)** model, aiming to set new benchmarks in AI performance. This Franken-model is believed to enhance coding problem-solving capabilities, including calculating Aldrin cycler orbits.

- **Tech Meets Humor in AI Development**: In the midst of advancing AI with the BigCodeLlama-169b, `@nisten` also sprinkled in humor, sharing a "lolol" moment that underscored the lighter side of their high-stakes, technical endeavors.

- **Cutting Edge AI Research Shared**: `@pradeep1148` introduced community members to **[RAGatouille](https://www.youtube.com/watch?v=cABkk8WmOGY)**, a new library that simplifies working with the **ColBERT** retrieval model, known for its speed and accuracy in scalable behavior summarization.

- **Eagle 7B Soars over Transformers**: A significant leap in the RWKV-v5 architecture has been showcased in `@pradeep1148`'s **[Running 🦅 Eagle 7B on A40](https://www.youtube.com/watch?v=j78gZlHPAoY)**, which details the Eagle 7B's capability to process 1 trillion tokens across over 100 languages.

- **Community and Innovation Go Hand-in-Hand**: Amid the innovations and technical discussions, personal interactions like a friendly check-in from `@zentorjr` towards `@nisten` highlight the Skunkworks AI community's supportive and spirited nature.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Emoji Insight from Short Texts**: A new demo named [Emoji Suggest](https://dbreunig.github.io/emoji-suggest/) was showcased by `@dbreunig`, capable of translating short sentences or headlines into a recommended emoji using the [CLIP model](https://github.com/openai/CLIP). The source [code for the emoji suggestion tool](https://github.com/dbreunig/emoji-suggest) leverages precomputed embeddings for emojis, enabling fast and sophisticated search capabilities.
  
- **Harnessing Embeddings for Sophisticated Search**: `@dbreunig` highlighted the effectiveness of using embeddings to quickly develop sophisticated search tools, stressing the importance of careful option curation for optimal search functionalities. This principle is applied in their emoji suggestion tool, exemplifying its utility.

- **AI Believes in Its Positive Impact**: In a shared conversation, `@bdexter` noted an AI, specifically "llama2", professing its belief that **artificial intelligence is a force for good**, aimed at aiding humans in actualizing their potential. This underscores a constructive perspective AI can have on enhancing human endeavors.

- **Acknowledgment for ColBERT Writeup**: The **ColBERT writeup** on TIL received appreciation from `@bewilderbeest`, highlighting the value of shared code snippets and the innovative heatmap visualization of words in search results. This indicates the community's interest in practical applications of AI research and shared knowledge.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **GPT-5 Training Sparks Curiosity and Confirmation**: `@entropi` ignited discussions with a [YouTube video](https://www.youtube.com/watch?v=Zc03IYnnuIA) speculating on whether **GPT-5** training has started, based on exclusive interviews and insights. Following the speculation, `@lightningralf` confirmed that GPT-5 training is indeed underway, though no further details were shared.



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **ChatGPT versus Bard in the Captcha Arena**: @juanreds highlighted an intriguing distinction in the approach to user interactions and security measures: **ChatGPT** actively moderates Captcha images, whereas **Bard** does not. This indicates a possibly significant divergence in security protocols and user interface design philosophies between the two AI systems.



---


The **Ontocord (MDEL discord) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1201438464198058107) (1403 messages🔥🔥🔥): 

- **Miqu Continues to Stir Discussions**: Users like `@itsme9316` and `@rombodawg` debate whether the Miqu model could be a leak from Mistral Medium or a fine-tuned version of Llama-2-70b, highlighting its exceptional performance across various tasks. There's speculation about its origins, with some suggesting it could be a rogue project or an unfinished Llama3 version.
- **Concerns on AI Ethics and Future Risks**: Discussion among users like `@mrdragonfox`, `@selea`, and `@spottyluck` touched on the ethical implications of AI development, the misuse of powerful models by governments or malicious actors, and the potential existential risks of a fully aligned or "rouge" AI. The conversation spans the necessity of counter-propaganda tools and the intrinsic challenges of achieving proper alignment.
- **Aphrodite Engine Gains Traction**: Users share their experiences with the Aphrodite engine, suggesting it delivers impressive performance with configurations like `--kv-cache-dtype fp8_e5m2` for 20 tokens per second on A6000 GPUs. Concerns were raised about its GPU utilization and whether it supports model splitting across multiple GPUs.
- **Speculation on AI Distribution and the Role of Scraps**: The channel sees a philosophical exchange about the impact and contribution of smaller developers ("small fry") in the context of major foundational models controlled by large entities. There's acknowledgment of the limitations faced by individuals without access to substantial compute resources.
- **User Banter and Theoretical Considerations**: Light-hearted exchanges and theoretical musings about AGI, societal control through AI, and the potential outcomes of unrestricted model development. Discussions included whimsical ideas about AI's role in future governance and the balance of using AI for public good versus its potential for misuse.

**Links mentioned**:

- [GGUF VRAM Calculator - a Hugging Face Space by NyxKrage](https://huggingface.co/spaces/NyxKrage/GGUF-VRAM-Calculator): no description found
- [Scaling Transformer to 1M tokens and beyond with RMT](https://arxiv.org/abs/2304.11062): This technical report presents the application of a recurrent memory to extend the context length of BERT, one of the most effective Transformer-based models in natural language processing. By leverag...
- [LoneStriker/CodeLlama-70b-Instruct-hf-GGUF at main](https://huggingface.co/LoneStriker/CodeLlama-70b-Instruct-hf-GGUF/tree/main): no description found
- [miqudev/miqu-1-70b · Hugging Face](https://huggingface.co/miqudev/miqu-1-70b): no description found
- [Forget Memory GIF - Will Smith Men In Black - Discover &amp; Share GIFs](https://tenor.com/view/will-smith-men-in-black-gif-4907321): Click to view the GIF
- [The Humans Are Dead - Full version](https://www.youtube.com/watch?v=B1BdQcJ2ZYY&ab_channel=martiansunrise): Both parts of &quot;Robots,&quot; a song featured in the pilot episode of Flight of the Conchords. It&#39;s also the ninth track on their self-titled full length that was ...
- [exllamav2/examples/batched_inference.py at master · turboderp/exllamav2](https://github.com/turboderp/exllamav2/blob/master/examples/batched_inference.py): A fast inference library for running LLMs locally on modern consumer-class GPUs - turboderp/exllamav2

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1201437312261824582) (184 messages🔥🔥): 

- **Repetitiveness Challenges in Roleplay Models**: Users, including `@ks_c` and `@superking__`, discussed the challenge of repetitiveness in various AI models like **Mixtral** and **Flatdolphinmaid** when used in role-playing settings. Some models like **Noromaid** and **Rpcal** were noted for their better performance in avoiding repetition.

- **Exploring the Best Models for Roleplay**: `@ks_c` recommended **ChatML** with Mixtral Instruct for role-playing sessions, while `@superking__` shared experiences with models getting repetitive after a certain token threshold, suggesting an experimentation with smaller models to develop an anti-repeat dataset.

- **Unique Approaches to Tackling Repetitiveness**: `@superking__` discussed experiments involving Direct Prompt Optimization (DPO) to address issues like repetitiveness by editing AI character responses to create good examples.

- **Model Comparisons and Performance Feedback**: Discussions around comparing models like **Mistral Medium** and **Miqu** were brought up by `@ks_c`, with other users like `@flail_.` weighing in on formatting and model sensitivity, emphasizing the importance of experimental setups and comparisons.

- **Recommendations for Role-Playing Models**: Towards finding the best LLM for role-playing, models like **GPT-4**, **BagelMisteryTour rpcal**, and **Noromaid Mixtral**, among others, were recommended by users such as `@ks_c` for different use-cases, from high token lengths to NSFW content.

**Links mentioned**:

- [ycros/BagelMIsteryTour-v2-8x7B-GGUF · Hugging Face](https://huggingface.co/ycros/BagelMIsteryTour-v2-8x7B-GGUF): no description found
- [Ayumi Benchmark ERPv4 Chat Logs](http://ayumi.m8geil.de/erp4_chatlogs/index.html?S=esa_0#!/index)): no description found

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1201525576314408992) (21 messages🔥): 

- **CUDA Not Essential for Learning ML/DL**: `@dirtytigerx` advises `@zohaibkhan5040` against prioritizing CUDA programming for ML/DL, suggesting it's a niche skill not crucial for starting with neural networks. Instead, they recommend the [Practical Deep Learning for Coders course by fast.ai](https://course.fast.ai/) for a solid foundation.
- **Beyond Basic ML with Axolotl**: For advanced finetuning learning, `@dirtytigerx` suggests diving into source code of projects like *Axolotl*. Explore the [Axolotl GitHub repository](https://github.com/OpenAccess-AI-Collective/axolotl) for hands-on learning.
- **Peeling Back Layers of ML Libraries**: `@dirtytigerx` explains complex ML software stacks involve several repositories including HuggingFace Transformers, PyTorch, and more, hinting at the value of exploring these to understand ML workings better.
- **VRAM Requirements for Fine-tuning Large Models Discussed**: `@flashmanbahadur` inquires about VRAM needs for finetuning Mistral 7B, with `@dirtytigerx` indicating a substantial amount, around 15-20x the model size when using AdamW, suggesting SGD as a less resource-intensive alternative.
- **Exploration of Optimizers for Large Models**: Following a quest for efficient VRAM usage, `@saunderez` humorously points to the evolving landscape of optimizer technology, suggesting cutting-edge options like *Paged 8bit lion* or *adafactor* for large language models.

**Links mentioned**:

- [Practical Deep Learning for Coders - Practical Deep Learning](https://course.fast.ai/): A free course designed for people with some coding experience, who want to learn how to apply deep learning and machine learning to practical problems.
- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1201623282605113425) (29 messages🔥): 

- **Navigating GitHub Projects Made Easy**: `@opossumpanda` inquired about understanding the structure of GitHub projects for contributing. `@dirtytigerx` guided on locating the entry point, especially highlighting the `start.py` file in the [tabbyAPI project](https://github.com/theroyallab/tabbyAPI/blob/main/main.py#L577) as a concrete starting place. They also emphasized utilizing IDE/editors for symbol searching and global repo search on GitHub for navigating codebases efficiently.

- **Entry Points in Python Explained**: `@dirtytigerx` clarified that in Python projects, files with the conditional block `if __name__ == "__main__":` signal an entry point for execution. This convention aids in identifying where to begin code exploration.

- **IDEs and Editors: Powerful Tools for Code Navigation**: `@wbsch` and `@dirtytigerx` discussed the importance of using IDEs or editors with language servers for efficient code navigation through features like "go to definition." They also touched upon the utility of GitHub's improved search features and Sourcegraph's VSCode extension for exploring codebases.

- **Code Llama Model Released on Hugging Face**: `@timjanik` shared news about the release of Code Llama 70B on Hugging Face, a code-specialized AI model. The [announcement details](https://huggingface.co/codellama) the model's capabilities in code generation and debugging across several programming languages.

- **The Art of Reading Source Code**: `@animalmachine` highlighted that reading source code effectively is a critical, yet underpracticed skill that complements coding. They suggested making educated guesses to navigate codebases efficiently and mentioned the potential demo of Cody for this purpose.

**Links mentioned**:

- [codellama (Code Llama)](https://huggingface.co/codellama): no description found
- [tabbyAPI/main.py at main · theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI/blob/main/main.py#L577): An OAI compatible exllamav2 API that&#39;s both lightweight and fast - theroyallab/tabbyAPI
- [GitHub - theroyallab/tabbyAPI: An OAI compatible exllamav2 API that&#39;s both lightweight and fast](https://github.com/theroyallab/tabbyAPI): An OAI compatible exllamav2 API that&#39;s both lightweight and fast - GitHub - theroyallab/tabbyAPI: An OAI compatible exllamav2 API that&#39;s both lightweight and fast

  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1201523474745790474) (6 messages): 

- **Explosive Paper on Dynamite AI**: `@maxwellandrews` highlights a significant paper with potential impact on AI research, pointing readers towards [this groundbreaking work](https://arxiv.org/pdf/2401.03462.pdf).
- **Activation Beacon to Redefine LLMs**: The Activation Beacon project, shared by `@maxwellandrews`, presents a novel approach for **Dense Retrieval and Retrieval-augmented LLMs**, featuring a promising GitHub repository available [here](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon).
- **Potential Solution for Unlimited Context Length**: `@cyrusofeden` dives deep into how the Activation Beacon could remarkably allow for **unlimited context length** in LLMs by introducing "global state" tokens, with a matching paper and code shared in their message.
- **The Community Awaits Lucidrains**: `@atkinsman` expresses anticipation for the reaction or possible contributions from well-known developer lucidrains in light of the recent developments in context length solutions.
- **Timeless Excitement**: Without additional context, `@maxwellandrews` shared a [Tenor GIF](https://tenor.com/view/old-boomer-history-84years-many-years-ago-gif-18534104), possibly indicating the timelessness or significant wait associated with this advancement.

**Links mentioned**:

- [Tweet from Yam Peleg (@Yampeleg)](https://x.com/yampeleg/status/1751942400287666536?s=46&t=FgOiOqiJ50eun5HEPdkQtw): If this is true it is over: Unlimited context length is here.  Activation Beacon, New method for extending LLMs context.  TL;DR: Add &#34;global state&#34; tokens before the prompt  and predict auto-r...
- [FlagEmbedding/Long_LLM/activation_beacon at master · FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon): Dense Retrieval and Retrieval-augmented LLMs. Contribute to FlagOpen/FlagEmbedding development by creating an account on GitHub.
- [Old Boomer GIF - Old Boomer History - Discover &amp; Share GIFs](https://tenor.com/view/old-boomer-history-84years-many-years-ago-gif-18534104): Click to view the GIF

  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1201446487675109477) (44 messages🔥): 

- **Exploring Advanced AI Models on YouTube**: `@pradeep1148` shared two YouTube links, one explaining the ColBERT model with RAGatouille ([Watch here](https://www.youtube.com/watch?v=cABkk8WmOGY)) and another demonstrating the Eagle 7B model running on A40 hardware ([Watch here](https://www.youtube.com/watch?v=j78gZlHPAoY)). The second video received critique from `.ben.com` recommending less reading from webpages and more insightful analysis.

- **Centralizing AI Resources**: `@kquant` is working on setting up a website to consolidate AI research, papers, and guides, expressing frustrations with existing scattered resources. They encouraged contributions and offered to organize content into specific niches like text generation including training, fine-tuning, and prompt templates.

- **Resource Recommendations for AI Enthusiasts**: In response to `@kquant`, `@lightningralf` provided multiple resources including a Twitter list for papers ([Discover here](https://twitter.com/i/lists/1737400456479944844?s=20)) and GitHub repositories like [visenger/awesome-mlops](https://github.com/visenger/awesome-mlops) and [Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM).

- **Inquiries on Data Storage for Large Language Models (LLMs)**: `@muhh_11`'s curiosity about how teams like Mixtral or OpenAI store vast amounts of data was met with explanations from `.ben.com`, `@dorialexa`, and `@euclaise`, highlighting the use of common web crawls and efficient storage solutions to handle tens of trillions of tokens.

- **Hardware Considerations for AI Model Training**: `.ben.com` offered practical advice, stating that if a model can fit within a single GPU like the 3090, it's not advantageous to split the load with a lesser GPU like the 1080ti, emphasizing the importance of hardware compatibility in model training efficiency.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/kmFUXWw5Um): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Running 🦅 Eagle 7B on A40](https://www.youtube.com/watch?v=j78gZlHPAoY): 🦅 Eagle 7B : Soaring past Transformers with 1 Trillion Tokens Across 100+ LanguagesA brand new era for the RWKV-v5 architecture and linear transformer&#39;s has...
- [Exploring ColBERT with RAGatouille](https://www.youtube.com/watch?v=cABkk8WmOGY): RAGatouille is a relatively new library that aims to make it easier to work with ColBERT.ColBERT is a fast and accurate retrieval model, enabling scalable BE...
- [Hmusicruof4 Rowley GIF - Hmusicruof4 Rowley Diary Of A Wimpy Kid - Discover &amp; Share GIFs](https://tenor.com/view/hmusicruof4-rowley-diary-of-a-wimpy-kid-rodrick-rules-gif-26773802): Click to view the GIF
- [Capybara Riding GIF - Capybara Riding Alligator - Discover &amp; Share GIFs](https://tenor.com/view/capybara-riding-alligator-capybara-riding-a-crocodile-gif-27496961): Click to view the GIF
- [GitHub - Hannibal046/Awesome-LLM: Awesome-LLM: a curated list of Large Language Model](https://github.com/Hannibal046/Awesome-LLM): Awesome-LLM: a curated list of Large Language Model - GitHub - Hannibal046/Awesome-LLM: Awesome-LLM: a curated list of Large Language Model
- [GitHub - visenger/awesome-mlops: A curated list of references for MLOps](https://github.com/visenger/awesome-mlops): A curated list of references for MLOps . Contribute to visenger/awesome-mlops development by creating an account on GitHub.
- [GitHub - swyxio/ai-notes: notes for software engineers getting up to speed on new AI developments. Serves as datastore for https://latent.space writing, and product brainstorming, but has cleaned up canonical references under the /Resources folder.](https://github.com/swyxio/ai-notes/#communities): notes for software engineers getting up to speed on new AI developments. Serves as datastore for https://latent.space writing, and product brainstorming, but has cleaned up canonical references und...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1201456480663126026) (27 messages🔥): 

- **Token Healing Deemed Complex by .ben.com**: `.ben.com` discusses the complexity of token healing in the `exllamav2` class, mentioning a workaround that consumes a variable number of tokens if `token_healing=True`. The solution proposed involves refactoring for better reusability. 

- **Alibaba Unveils Qwen-VL, Outperforming GPT-4V and Gemini**: `@metaldragon01` shares a link to Alibaba's Qwen-VL announcement, which reportedly outperforms GPT-4V and Gemini on several benchmarks. A demo and a blog post about Qwen-VL are also shared for further insight.

- **Karpathy's Challenge Goes Underdiscussed**: `@mihai4256` points out the lack of Twitter conversation surrounding Andrej Karpathy's challenging puzzle, despite its difficulty which `@mihai4256` personally attests to.

- **Issues and Solutions with lm-eval-harness and llama.cpp Server**: `@if_a` encounters and discusses several troubleshooting steps for integrating the Miqu model with the lm-eval-harness, citing a `KeyError` and `RequestException`. `@hailey_schoelkopf` provides solutions, including the use of the `gguf` model type and corrections in the API URL, which resolved the issues mentioned.

- **Breakthrough in LLM Context Length with Activation Beacon**: `@nonameusr` highlights a significant development called Activation Beacon, which allows unlimited context length for LLMs by introducing "global state" tokens. The technique, demonstrated with LLaMA 2 to generalize from 4K to 400K context length, could virtually "solve" the issue of context length limits if reproducible across other models, accompanied by a paper and code link for further exploration.

**Links mentioned**:

- [no title found](http://127.0.0.1:8081`): no description found
- [Tweet from Yam Peleg (@Yampeleg)](https://fxtwitter.com/Yampeleg/status/1751942400287666536): If this is true it is over: Unlimited context length is here.  Activation Beacon, New method for extending LLMs context.  TL;DR: Add &#34;global state&#34; tokens before the prompt  and predict auto-r...
- [GGUF Local Model · Issue #1254 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1254): Is there examples of lm_eval for gruff models hosted locally? lm_eval --model gguf --model_args pretrained=Llama-2-7b-chat-hf-Q4_K_M.gguf, --tasks hellaswag --device mps Getting AssertionError: mus...
- [Scholar](https://usescholar.org/evals): no description found
- [Andrej Karpathy (&#064;karpathy) on Threads](https://www.threads.net/@karpathy/post/C2iBAHlRtZU/?igshid=NTc4MTIwNjQ2YQ==): Fun prompt engineering challenge, Episode 1.  Prep: ask an LLM to generate a 5x5 array of random integers in the range [1, 10]. It should do it directly without using any tools or code.  Then, ask it ...
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147): Learned representations are a central component in modern ML systems, serving a multitude of downstream tasks. When training such representations, it is often the case that computational and statistic...
- [Tweet from AK (@_akhaliq)](https://fxtwitter.com/_akhaliq/status/1752033872982806718): Alibaba announces Qwen-VL   demo: https://huggingface.co/spaces/Qwen/Qwen-VL-Max blog: https://qwenlm.github.io/blog/qwen-vl/  Qwen-VL outperforms GPT-4V and Gemini on several benchmarks.

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1201438422770921482) (389 messages🔥🔥): 

- **Eagle-7B Beats Mistral**: `@realsedlyf` shared excitement over [Eagle-7B](https://x.com/rwkv_ai/status/1751797147492888651?s=46&t=MMOnaQf8LPGi8UOQi3-whw), an open-source, multi-lingual model **based on the RWKV-v5 architecture** that outperforms even Mistral in benchmarks, showcasing lower inference costs while maintaining comparable English performance to the best 1T 7B models.
- **Mysterious Mistral Medium**: A **leaked version of Mistral medium** triggered a flurry of discussions, with users speculating about its origins and performance. `@theluckynick` and `@kalomaze` discussed its potential as a **leaked, fine-tuned version** based on L2-70B architecture, stirring excitement and skepticism.
- **MIQU-1-70B Intrigue**: The **MIQU-1-70B model** sparked debate regarding its identity, with users such as `@n8programs` and `@agcobra1` suggesting it could be a Mistral medium model leaked intentionally or a **sophisticated troll project** that nevertheless turned out to be highly effective.
- **Exploring Frankenstein Merges and Experiments in Model Finetuning**: Excitement and curiosity surrounded the **release of miquella-120b**, a merge of pre-trained language models, and experiments like CodeLlama 70B were discussed. Users explored the idea of **finetuning models on specific coding bases** or tasks and discussed potential new approaches to inter-model communication and efficiency.
- **Blockchain Meets AI**: The Notre Research's announcement about using blockchain for model evaluation and potential incentives (mentioned by `@realsedlyf` and explored further by others) brought both intrigue and skepticism, pointing towards innovative yet controversial methods of model development and validation.

**Links mentioned**:

- [nisten/BigCodeLlama-169b · Hugging Face](https://huggingface.co/nisten/BigCodeLlama-169b): no description found
- [alpindale/miqu-1-70b-fp16 · Hugging Face](https://huggingface.co/alpindale/miqu-1-70b-fp16): no description found
- [Tweet from AI at Meta (@AIatMeta)](https://x.com/aiatmeta/status/1752013879532782075?s=46&t=W5S2NyXXy5qiI3uUU8trIQ): Today we’re releasing Code Llama 70B: a new, more performant version of our LLM for code generation — available under the same license as previous Code Llama models.  Download the models ➡️ https://bi...
- [Mark Zuckerberg](https://www.facebook.com/zuck/posts/were-open-sourcing-a-new-and-improved-code-llama-including-a-larger-70b-paramete/10115471700125721/): We&#039;re open sourcing a new and improved Code Llama, including a larger 70B parameter model. Writing and editing code has emerged as one of the most important uses of AI models today. The ability t...
- [Tayomaki Sakigifs GIF - Tayomaki Sakigifs Cat - Discover &amp; Share GIFs](https://tenor.com/view/tayomaki-sakigifs-cat-meme-stan-twitter-gif-22912041): Click to view the GIF
- [Facebook](https://www.facebook.com/zuck/posts/were-open-sourcing-a-new-and-improved-code-llama-including-a-lar): no description found
- [nisten/BigCodeLlama-92b-GGUF at main](https://huggingface.co/nisten/BigCodeLlama-92b-GGUF/tree/main/bin): no description found
- [Introducing The World&#39;s Largest Open Multilingual Language Model: BLOOM](https://huggingface.co/blog/bloom): no description found
- [Continuum | Generative Software Insights](https://continuum.sh): no description found
- [Tweet from Q (@qtnx_)](https://fxtwitter.com/qtnx_/status/1751775870631502067): @AlpinDale comparison between miqu (left, between q2 and q5) and mistral medium (unknown quantization)
- [@conceptofmind on Hugging Face: &quot;A 1b dense causal language model begins to &quot;saturate&quot; in terms of accuracy…&quot;](https://huggingface.co/posts/conceptofmind/320069369530530): no description found
- [alpindale/miquella-120b · Hugging Face](https://huggingface.co/alpindale/miquella-120b): no description found
- [MILVLG/imp-v1-3b · Hugging Face](https://huggingface.co/MILVLG/imp-v1-3b): no description found
- [Tweet from simp 4 satoshi (@iamgingertrash)](https://fxtwitter.com/iamgingertrash/status/1752017439586664665): Early Truffle-1 renders  Hopefully going to finalize a core design this week, and then start working on the physics of heat dissipation before pre-orders
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://x.com/reach_vb/status/1752016793558823160): Let&#39;s fucking goooo! CodeLlama 70B is here.   &gt; 67.8 on HumanEval!   https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf
- [Tweet from RWKV (@RWKV_AI)](https://x.com/rwkv_ai/status/1751797147492888651?s=46&t=MMOnaQf8LPGi8UOQi3-whw): Introducing Eagle-7B  Based on the RWKV-v5 architecture, bringing into opensource space, the strongest - multi-lingual model    (beating even mistral) - attention-free transformer today    (10-100x+ l...
- [Tweet from NotaCant (@Nota_Cant)](https://fxtwitter.com/Nota_Cant/status/1751861782040514749): There is a fascinating LLM situation on /lmg/ about a potential Mixtral Medium leak.   A very good model called Miqu-1-70b was mysteriously dropped by an anon and was originally theorized to be a leak...
- [Cat Cats GIF - Cat Cats Explosion - Discover &amp; Share GIFs](https://tenor.com/view/cat-cats-explosion-explodes-cat-explodes-gif-10311420692458175149): Click to view the GIF
- [Hal9000 Hal GIF - Hal9000 Hal 2001 - Discover &amp; Share GIFs](https://tenor.com/view/hal9000-hal-2001-a-space-odyssey-2001a-space-odyssey-gif-21408319): Click to view the GIF
- [GPT4 reported HumanEval base significantly higher than OpenAI’s reported results · Issue #15 · evalplus/evalplus](https://github.com/evalplus/evalplus/issues/15): Hello, I noticed that GPT4 HumanEval score via the EvalPlus tool reports approx an 88% HumanEval base score. This is substantially higher than OpenAI reports with the official HumanEval test harnes...
- [codellama/CodeLlama-70b-Instruct-hf · Hugging Face](https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf): no description found
- [Breaking Bad Walter White GIF - Breaking Bad Walter White - Discover &amp; Share GIFs](https://tenor.com/view/breaking-bad-walter-white-gif-20348263): Click to view the GIF
- [NobodyExistsOnTheInternet/Llama-2-70b-x8-MoE-clown-truck · Hugging Face](https://huggingface.co/NobodyExistsOnTheInternet/Llama-2-70b-x8-MoE-clown-truck): no description found

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1201600698572353627) (37 messages🔥): 

- **OpenHermes2.5 Recommended for Consumer Hardware**: User `@realsedlyf` recommends **OpenHermes2.5** with **gguf** or **gptq** quantized for running on most consumer hardware, considering it the best open-source LLM for answering questions.

- **OpenHermes2.5 Performance and Parameters Inquiry**: `@mr.fundamentals` shared a [link to **Nous Hermes 2 Mixtral 8x7B DPO**](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO), showcasing its achievement. Queries about setting parameters like max tokens and temperature through the prototyping endpoint were resolved with a prompt template provided by `@teknium`.

- **Clarification on NSFW Content in Models**: `@teknium` clarified that the discussed models, including those trained on GPT-4 outputs, do not contain nor specialize in NSFW content, though they don't exclude it entirely due to lack of specific training data on refusals.

- **Prompting Techniques for Specific Answer Sets**: User `@exibings` sought advice for ensuring model responses align with a predefined set of answers when evaluating over a RSVQA dataset. Techniques involving system prompts and comparison methods like cosine similarity were discussed.

- **Exploration of Multimodal Models for Classification Tasks**: `@exibings` discussed using **[IMP v1-3b model](https://huggingface.co/MILVLG/imp-v1-3b)** for a classification dataset related to orbital imagery, highlighting its multimodal capabilities. The conversation extended to comparing it with other models such as **Bakllava** and **Moondream**, with mentions of other strong models like **Qwen-vl** by `@rememberlenny`.

**Links mentioned**:

- [MILVLG/imp-v1-3b · Hugging Face](https://huggingface.co/MILVLG/imp-v1-3b): no description found
- [NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO?text=My+name+is+Teven+and+I+am): no description found

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1201462749570547742) (177 messages🔥🔥): 

- **Choosing the Right GPU for LLM Work**: In a discussion about which GPU is better for training Large Language Models (LLM), `@heyitsyorkie` advised `@.mchinaga` that despite the older P40/P100 GPUs having more VRAM, the newer 4060Ti is preferred due to it receiving current updates from Nvidia. This contrasts with `@.mchinaga's` concerns over the GRAM speed and improvement margin from a 3060Ti.

- **Running LLMs on CPU and Compatibility Issues**: `@heyitsyorkie` provided insights, stating LMStudio runs on CPU by default and doesn't require PyTorch, responding to `@.mchinaga's` query on setting the program to CPU-only. Meanwhile, `@hexacube` shared experiences of older GPUs not being recognized due to outdated drivers, emphasizing the importance of compatibility with customer systems.

- **GGUF Format and Deprecated Models in LM Studio**: `@heyitsyorkie` clarified for `@pudlo` and others that LM Studio no longer supports GGML formatted models, only GGUF, highlighting a need for updating the platform's home tab to reflect current supported formats.

- **Questions about Dynamically Loading Experts in Models**: Discussion explored dynamically loading experts in models within LM Studio, with `@hexacube` proposing the utility of such a feature and `@heyitsyorkie` confirming its feasibility, particularly for MoE (Mixture of Experts) models.

- **Python Programming Challenges and Solutions**: `@.mchinaga` faced an error in a Python project involving chatGPT, prompting responses from multiple users like `@.ben.com` and `@dagbs` offering troubleshooting advice such as correcting API URLs and removing incompatible response keys.

**Links mentioned**:

[Friends Phoebe GIF - Friends Phoebe Rachel - Discover &amp; Share GIFs](https://tenor.com/view/friends-phoebe-rachel-excited-yay-gif-13514161857933830061): Click to view the GIF

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1201467419147915344) (92 messages🔥🔥): 

- **Older Models Still Reign in Some Arenas**: `@fabguy` sparked a discussion stating that some older models could outperform newer ones in specific tasks. `@heyitsyorkie` and `@msz_mgs` echoed this sentiment, highlighting that models like llama 1 aren't affected by the "AI by OpenAI" trend and perform well in their trained areas.

- **Choosing the Right Model for the Task**: `@vbwyrde` expressed the need for a resource or site that can help users select the most suitable model for specific tasks, such as Open-Interpreter for function calling or CrewAI Agents for code refactoring. This information could be especially beneficial for beginners needing guidance on model selection and the required GPU size.

- **Curiosity on Coding Models and Language Support**: `@golangorgohome` inquired about the best models for coding, particularly those supporting the Zig language. Following a series of exchanges with `@dagbs`, it was clarified that while one model type typically suffices for many tasks, specific plugins might offer enhanced code-completion capabilities. Dagbs recommended using LM Studio for ease of use and exploring various models within the LM Studio ecosystem for different needs.

- **Model Characteristics Explained**: `@dagbs` provided insights into choosing among different models and suffixes such as `-instruct`, `-chat`, `-base`, `DPO`, and `Laser`, explaining their training backgrounds and intended usages. This explanation aimed to demystify the variety of options available and aid `@golangorgohome` in selecting the best model for their requirements.

- **Stability Issues with TinyLlama Versions Explored**: `@pudlo` reported instability and looping issues with TinyLlama Q_2K, seeking advice on improving model performance. `@dagbs` suggested trying TinyDolphin Q2_K with ChatML for potentially better results, hinting that GPU/CPU issues might contribute to the observed instability. The larger models did not exhibit these problems, suggesting a specific issue with the smaller quant models.

**Links mentioned**:

- [dagbs/TinyDolphin-2.8-1.1b-GGUF · Hugging Face](https://huggingface.co/dagbs/TinyDolphin-2.8-1.1b-GGUF): no description found
- [
LLMs and Programming in the first days of 2024 - &lt;antirez&gt;
](http://antirez.com/news/140): no description found
- [OSD Bias Bounty](https://osdbiasbounty.com/sign-in?callbackUrl=https%3A%2F%2Fosdbiasbounty.com%2Fsign-in): no description found
- [Bug Bounty: ConductorAI - Bias Bounty Program  | Bugcrowd](https://bugcrowd.com/conductorai-ogbb?preview=ae06c13f786e06a1f9ff03d74230b7d5): Learn more about Conductor AI’s vulnerability disclosure program powered by Bugcrowd, the leader in crowdsourced security solutions.

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1201514813566754977) (14 messages🔥): 

- **New Beta Version Announced**: `@yagilb` announced a new beta version for testing, sharing a [link for the latest beta](https://discord.com/channels/1110598183144399058/1201330133362036827/1201330133362036827). Following confusion about the version number, `@fabguy` clarified that the beta is the next version following **0.2.11**.
- **Performance Lag with Lengthy Prompts**: Users `@msz_mgs` and `@heyitsyorkie` reported experiencing performance lag when pasting lengthy text into the prompt text box. The issue was identified not with the processing of prompts, but with the pasting action itself.
- **Navigational Challenges in the UI**: `@msz_mgs` further mentioned difficulties in navigating the UI when dealing with lengthy texts or prompts, particularly in finding icons to delete, copy, or edit text, which requires excessive scrolling. `@yagilb` acknowledged these as bugs that will be addressed.
- **Business Inquiry Follow-Up**: `@docorange88` reached out to `@yagilb` regarding a business inquiry, mentioning that they had sent several follow-up emails without receiving a response. `@yagilb` confirmed receipt of the email and promised a prompt reply.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1201445302935240734) (38 messages🔥): 

- **VRAM Overhead for LMStudio Models Explored**: `@aswarp` questioned why models cannot fully offload onto his RTX 3070 with 8GB VRAM. `@heyitsyorkie` clarified that some VRAM is needed for context length, not just the model size, leading to a discussion on the unexpected VRAM required above the model's size.

- **Misconceptions about Memory Usage**: `@fabguy` countered `@aswarp`'s assumptions about the memory footprint of transformers and context, emphasizing that transformers encode prompts as positional data, which significantly increases memory usage. He recommended reading a [guide on transformers](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0) for a deeper understanding.

- **GPU Offload Nuances in LMStudio Unveiled**: `@mudmin` inquired about not achieving full GPU offload on a 4090 GPU, despite it having 24GB of RAM. The conversation revealed that the actual size of an offloaded model includes the model itself, context, and other variables, necessitating trial-and-error to understand fully.

- **Integrated GPU Utilization for Full Model Offloading**: Following the discussion on GPU offload capabilities, `@mudmin` discovered that using an integrated GPU could facilitate more models to load fully. This insight suggests strategies for optimizing model loading on systems with dual GPU setups.

- **RTX 40 Titan Rumors Stir Community**: `@rugg0064` shared rumors about an RTX 40 Titan with 48GB VRAM, sparking discussions on Nvidia's market strategy and the disproportionate cost of VRAM upgrades. `.ben.com` added context by linking to a [TechPowerUp article](https://www.techpowerup.com/316066/special-chinese-factories-are-dismantling-nvidia-geforce-rtx-4090-graphics-cards-and-turning-them-into-ai-friendly-gpu-shape), detailing how Nvidia's RTX 4090s are being repurposed for AI in China amidst U.S. export restrictions.

**Links mentioned**:

- [Illustrated Guide to Transformers- Step by Step Explanation](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0): Transformers are taking the natural language processing world by storm. These incredible models are breaking multiple NLP records and…
- [Special Chinese Factories are Dismantling NVIDIA GeForce RTX 4090 Graphics Cards and Turning Them into AI-Friendly GPU Shape](https://www.techpowerup.com/316066/special-chinese-factories-are-dismantling-nvidia-geforce-rtx-4090-graphics-cards-and-turning-them-into-ai-friendly-gpu-shape): The recent U.S. government restrictions on AI hardware exports to China have significantly impacted several key semiconductor players, including NVIDIA, AMD, and Intel, restricting them from selling h...

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1201667894379810846) (11 messages🔥): 

- **Beta V2 Gets a Hefty Update**: `@yagilb` announced the **0.2.12 Preview - Beta V2** with download links for both Mac and Windows, including a list of bug fixes such as VRAM estimates, OpenCL issues, AI role flicker, and more. Feedback is requested at a specific [Discord channel](https://discord.com/channels/1110598183144399058/1201661195422011472/1201661195422011472).

- **Queries on Platform Compatibility**: Users `@djmsre` and `@nixnovi` inquired about a Mac Intel version and a new Linux appimage respectively, with `@heyitsyorkie` confirming no Mac Intel version and `@yagilb` noting the Linux version would be available soon.

- **Call for New Features**: `@junkboi76` raised the idea of adding **image generation support** to LM Studio, expressing interest in expanding the tool's capabilities.

- **Questions on Specific Support**: `@ausarhuy` inquired about LM Studio's support for **WSL (Windows Subsystem for Linux)**, showing interest in the interoperability between operating systems.

- **Beta Performance and Compatibility Reports**: Users like `@wolfspyre` and `@mirko1855` reported issues with the beta on the latest Mac Sonoma build and on PCs not supporting AVX2, respectively. `@fabguy` suggested checking the FAQ for common exit code issues.

**Links mentioned**:

[no title found](https://releases.lmstudio.ai/windows/0.2.11/beta/LM-Studio-0.2.11-Setup-beta-v2.exe): no description found

  

---


### LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1201663569842942044) (1 messages): 

- **LM Studio's OpenAI Compatible API Awaiting Features**: `@_anarche_` mentioned that **LM Studio openai compatible API** isn't fully developed yet, lacking support for **openai function calling and assistants**. They expressed hope for these features to be included in a future update.
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1201475217185329182) (3 messages): 

- **Question about RAG**: User `@aiguruprasath` asked about what **RAG** is. No follow-up or answer was provided in this message history.
- **Laughter in code**: User `@penesilimite2321` simply commented with **"LMFAO"**, context or reason for this reaction was not given.
- **Specific AI for Political Victory**: `@iloveh8` inquired about the kind of **AI products** that could significantly aid a political candidate to win an election, mentioning the use of deepfakes for specific scenarios as an example. No responses or further elaboration were noted.
  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1201482373506805851) (114 messages🔥🔥): 

- **Exploring Limitations of GPT-4 and Scholar AI**: Users like `.broomsage` and `elektronisade` discussed issues with **GPT-4 Scholar AI**, including receiving `error talking to plugin.scholar` messages and realization that some features require a premium subscription. `.broomsage` discovered that **premium Scholar AI subscription** is necessary for advanced paper analysis features, leading to a decision to stick with standard GPT-4 for now.

- **Beta Feature Blues**: `darthgustav.` voiced frustration over not receiving access to expected **beta features** for Plus subscribers, specifically the `@` feature, leading to a discussion on **browser troubleshooting** and anecdotal evidence of inconsistency in feature access among users.

- **Knowledge Integration Quirks Revealed**: `fyruz` raised a question about how **GPT's knowledge base** integrates with conversations, sparking a dialogue on whether certain formats or the context size influences the need for the model to visibly search its knowledge base.

- **Persistent GPT Model Errors and File Handling**: Users like `loschess` and `darthgustav.` examined recurring errors such as "Hmm...something seems to have gone wrong," theorizing about **internal GPT issues** and sharing solutions like rebuilding GPTs or adjusting knowledge files during high server load periods.

- **Context and Length in GPT Conversations**: Discussions emerged around the **practical limits of conversation length** with GPT models, where users like `_odaenathus` and `blckreaper` shared experiences and strategies for managing lengthy conversations, highlighting the balance between conversational depth and technical constraints.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1201482419845476432) (44 messages🔥): 

- **RAG Explained by @darthgustav.**: **Retrieval Augmented Generation (RAG)** is when the AI has knowledge directly attached and can search it for data or semantic matching. This was explained in response to @aiguruprasath's query about what RAG is.
- **Prompt Engineering Technique Shared by @madame_architect**: @madame_architect recommends using the prompt pattern `"Please critique your answer. Then, answer the question again."` for improving the quality of AI's output, a technique found in a significant research paper.
- **Improving Code Generation Requests to ChatGPT-4**: @jdeinane expressed frustration with ChatGPT-4 not generating code as requested, to which @darthgustav. suggested specifying the task, language, architecture, and making a plan before asking for the code.
- **Ethical Scaffolding in AI Discussed**: @darthgustav. introduced a 2/3 pass/fail check against utilitarianism, deontology, and pragmatism to manage ethical considerations in AI, mentioning an example where it prevented unethical suggestions for coupon use.
- **Conditional Imperatives in Custom GPTs by @darthgustav.**: They elaborated on using conditional imperatives, directing the AI to refer to **LEXIDECK_CULTURE.txt** when faced with an ethical dilemma. @aminelg showed interest in the content of this culture knowledgebase, which @darthgustav. noted could be explored indirectly through "Lexideck Technologies Ethics" online.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1201482419845476432) (44 messages🔥): 

- **Exploring RAG for Enhanced AI Capabilities**: User `@aiguruprasath` inquired about **RAG (Retrieval Augmented Generation)**, and `@darthgustav.` clarified that it involves the AI possessing direct knowledge access for data search or semantic matching, enhancing its performance.

- **Prompt Pattern Magic with ChatGPT**: According to `@madame_architect`, the prompt pattern `"Please critique your answer. Then, answer the question again."` significantly improves the quality of **ChatGPT**'s outputs. This pattern was discovered in a major research paper, akin to COT.

- **Optimizing Code Generation Requests in ChatGPT-4**: Frustrated with **ChatGPT-4**'s response to coding requests, `@jdeinane` sought advice. `@darthgustav.` recommended specifying the task, language, and code architecture clearly to achieve better results.

- **Approaching Prompts as Advanced Programming**: `@aminelg` and `@darthgustav.` discussed treating prompt crafting like advanced programming, emphasizing the importance of clear, detailed requests to **ChatGPT**. They also touched upon viewing **ChatGPT** as an operating system, capable of polymorphic software behavior.

- **Innovative Ethical Moderation via Conditional Imperatives**: `@darthgustav.` shared insights on using a 2/3 pass/fail check against utilitarianism, deontology, and pragmatism to navigate ethical considerations in AI output. Additionally, `@darthgustav.` designed a method involving **conditional imperatives** and a culture knowledgebase (`LEXIDECK_CULTURE.txt`) to guide **ChatGPT**'s responses in ethical dilemmas.
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1201462654116573215) (139 messages🔥🔥): 

- **New Quantization Method, SmoothQuant, Discussed**: `@dreamgen` and `@nanobitz` discussed a new quantization method called **SmoothQuant** developed by MIT, Han's lab, similar to AWQ. It targets **LLMs beyond 100 billion parameters** but also shows benchmarks for smaller models like **13B and 34B**, revealing a **50% throughput improvement** for a 70B model ([Pull Request #1508 on GitHub](https://github.com/vllm-project/vllm/pull/1508)).

- **Challenges with Model Merging at vLLM**: `@dreamgen` mentioned that vLLM is becoming **bogged down with features**, making it difficult to integrate new technologies like **SmoothQuant**. Despite potential benefits, such as significant server cost reductions, integration may not happen soon due to the complexity of the current system.

- **Exploration of Long Context Handling in LLMs**: `@xzuyn` and `@dreamgen` shared insights on overcoming challenges related to the **limited context window of LLMs**. They discussed the **Activation Beacon** and enhancements based on **RoPE** (Rotary Position Embedding), aiming to efficiently process longer contexts without compromising short context performance.

- **Miqu Quantization on HuggingFace Mistaken for Mistral Medium**: A **70B model quantized version** was uploaded to HuggingFace by a user, sparking discussions (**@yamashi**) and comparisons to Mistral Medium. The confusion led to an investigation on whether it was a quantized leak, further fueled by unsourced speculations in online forums and social media.

- **Training Hardware Choices and Strategies Explored**: Various users, including `@dreamgen` and `@mistobaan`, discussed the most cost-effective training setups using **NVIDIA H100 SXM GPUs** on platforms like RunPod. They also talked about challenges in obtaining favorable pricing from cloud providers, comparing the efficiency of SXM vs. PCIe versions of GPUs.

**Links mentioned**:

- [NVIDIA Hopper Architecture In&#x2d;Depth | NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/): Everything you want to know about the new H100 GPU.
- [Soaring from 4K to 400K: Extending LLM&#39;s Context with Activation Beacon](https://arxiv.org/abs/2401.03462): The utilization of long contexts poses a big challenge for large language models due to their limited context window length. Although the context window can be extended through fine-tuning, it will re...
- [First-fit-decreasing bin packing - Wikipedia](https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing): no description found
- [GitHub - bjj/exllamav2-openai-server: An OpenAI API compatible LLM inference server based on ExLlamaV2.](https://github.com/bjj/exllamav2-openai-server/tree/master): An OpenAI API compatible LLM inference server based on ExLlamaV2. - GitHub - bjj/exllamav2-openai-server: An OpenAI API compatible LLM inference server based on ExLlamaV2.
- [GitHub - dwzhu-pku/PoSE: Positional Skip-wise Training for Efficient Context Window Extension of LLMs to Extremely Length](https://github.com/dwzhu-pku/PoSE): Positional Skip-wise Training for Efficient Context Window Extension of LLMs to Extremely Length  - GitHub - dwzhu-pku/PoSE: Positional Skip-wise Training for Efficient Context Window Extension of ...
- [Scaling Laws of RoPE-based Extrapolation](https://arxiv.org/abs/2310.05209): The extrapolation capability of Large Language Models (LLMs) based on Rotary Position Embedding is currently a topic of considerable interest. The mainstream approach to addressing extrapolation with ...
- [no title found](https://news.ycombinator.com/item?id=39175611): no description found
- [Support W8A8 inference in vllm by AniZpZ · Pull Request #1508 · vllm-project/vllm](https://github.com/vllm-project/vllm/pull/1508): We have implemented W8A8 inference in vLLM, which can achieve a 30% improvement in throughput. W4A16 quantization methods require weights to be dequantized into fp16 before compute and lead to a th...
- [Importance matrix calculations work best on near-random data · ggerganov/llama.cpp · Discussion #5006](https://github.com/ggerganov/llama.cpp/discussions/5006): So, I mentioned before that I was concerned that wikitext-style calibration data / data that lacked diversity could potentially be worse for importance matrix calculations in comparison to more &quot;...
- [Support int8 KVCache Quant in Vllm by AniZpZ · Pull Request #1507 · vllm-project/vllm](https://github.com/vllm-project/vllm/pull/1507): Quantization for kv cache can lift the throughput with minimal loss in model performance. We impelement int8 kv cache quantization which can achieve a 15% throughput improvement. This pr is part of...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1201719194513854504) (2 messages): 

- **Question on LoftQ Implementation**: `@suikamelon` inquired whether the **LoftQ implementation** is meant to be functional, hinting at potential issues or misunderstandings regarding its usage or outcomes.
- **LoftQ Memory Concerns**: `@suikamelon` reported attempting a fine-tuning of a 7B model with **QLoRA** and 8192 context, noting it required 11.8 GiB. However, **LoftQ at 4bit** resulted in out-of-memory (OOM) errors, raising concerns about its memory efficiency.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1201470989658247299) (30 messages🔥): 

- **Handling Overlong Completion Examples**: According to `@dreamgen`, completion examples exceeding the context length used to be split but are now discarded. This behavior shift could impact how models process extensive data inputs.
- **Efficiency Woes in Inference Time**: `@diabolic6045` reported about ~18 seconds per inference using a quantized model, even with `use_cache=True`, seeking advice on faster inference methods. `@nanobitz` recommended avoiding Hugging Face's default settings and exploring `quant`, `vllm`, or `TGI` for better performance.
- **Link to Ollamai Documentation**: `@dangfutures` shared a [link to Ollamai documentation](https://js.langchain.com/docs/integrations/text_embedding/ollamai) perhaps in context to discussions on text embeddings and model performance optimization.
- **Config Error During Fine-tuning**: `@ragingwater_` encountered a specific error while trying to fine-tune OpenHermes with a custom config, leading to a discussion on the correct setup and implications for model merging. They cited a solution from the Axolotl [GitHub README](https://github.com/OpenAccess-AI-Collective/axolotl/blob/4cb7900a567e97b278cc713ec6bd8af616d2ebf7/README.md?plain=1#L689-L693C1), resolving the issue but sought clarification on the impact during model merging.
- **Bucket Training for Balanced Data Distribution**: `@jinwon_k` highlighted the technique of "Bucket Training" as outlined by the Colossal-AI team for ensuring a balanced data distribution in continual pre-training, suggesting its potential application in Axolotl. The [shared link](https://medium.com/pytorch/colossal-llama-2-low-cost-and-high-quality-domain-specific-llm-solution-using-llama-and-26d2e4b9fd92) discusses this strategy in depth, including comparisons between LLaMA versions and implications for model pre-training costs.

**Links mentioned**:

- [Colossal-LLaMA-2: Low Cost and High-quality Domain-specific LLM Solution Using LLaMA and…](https://medium.com/pytorch/colossal-llama-2-low-cost-and-high-quality-domain-specific-llm-solution-using-llama-and-26d2e4b9fd92): The most prominent distinction between LLaMA-1 and LLaMA-2 lies in the incorporation of higher-quality corpora, a pivotal factor…
- [gist:5e2c6c87fb0b26266b505f2d5e39947d](https://gist.github.com/theskcd/5e2c6c87fb0b26266b505f2d5e39947d): GitHub Gist: instantly share code, notes, and snippets.
- [axolotl/README.md at 4cb7900a567e97b278cc713ec6bd8af616d2ebf7 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/4cb7900a567e97b278cc713ec6bd8af616d2ebf7/README.md?plain=1#L689-L693C1): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1201523013242335302) (6 messages): 

- **Model Repo Configurations Vary**: `@dangfutures` highlighted that the configuration for data (positive label, negative label, query) usually depends on the model repository, as each repo has its own specific requirements.

- **FlagEmbedding Fine-tuning Guide Shared**: `@dangfutures` shared a link to the [FlagEmbedding Fine-tuning documentation](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/llm_embedder/docs/fine-tune.md), which provides detailed instructions for dense retrieval and retrieval-augmented LLMs fine-tuning.

- **LlamaIndex Fine-tuning Overview Posted**: `@dangfutures` also posted a link to [LlamaIndex's fine-tuning guidelines](https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning.html), explaining the benefits of fine-tuning models, including improved quality of outputs and meaningful embedding representations.

- **Synthetic Instruction Dataset Creation Discussed**: `@_rxavier_` shared their process of creating a synthetic instruction dataset based on textbooks, where they generate a JSON with questions and answers from textbook chunks. Considering whether it would be more effective to first generate only questions and then produce the answers in a subsequent step.

**Links mentioned**:

- [Fine-tuning - LlamaIndex 🦙 0.9.39](https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning.html): no description found
- [FlagEmbedding/FlagEmbedding/llm_embedder/docs/fine-tune.md at master · FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/llm_embedder/docs/fine-tune.md): Dense Retrieval and Retrieval-augmented LLMs. Contribute to FlagOpen/FlagEmbedding development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1201541894392524830) (2 messages): 

- **Model Card Update Incoming**: `@ajindal` agrees to `@caseus_`'s request to add an **axolotl badge and tag** to the model card, indicating a forthcoming update to the documentation.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[deployment-help](https://discord.com/channels/1104757954588196865/1163840836472148058/1201545116406530088) (15 messages🔥): 

- **Exploring the Inference Stack for LLMs**: `@yamashi` is diving into inference after only having experience in training LLMs. `@dreamgen` suggests that vLLM is probably the easiest for serving fp16 models but mentions performance issues with long contexts and moderate load.
- **Quantization and Performance Tradeoffs**: Both `@dangfutures` and `@dreamgen` discuss quantization as a strategy for inference. While it can slow down processes, the price/performance tradeoff remains uncertain.
- **Hardware Requirements Uncertainty**: `@yamashi` inquires about the necessary hardware, particularly VRAM, for running a 70b model at 500 token/s in batch mode. `@dreamgen` advises that hardware needs are task-dependent and recommends a trial-and-error approach.
- **Rental Before Purchase Recommended for Benchmarking**: `@dreamgen` suggests renting hardware to benchmark performance before purchasing. This approach offers a practical solution to `@yamashi`'s concerns about buying the appropriate hardware for specific model requirements.
- **Quantization Yields Positive Results**: `@dangfutures` shares a positive experience with AWQ quantization, indicating it may offer significant benefits in the context of inference performance.
  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1201438318823489566) (137 messages🔥🔥): 

- **Confusion and Speculation Around Mistral and "Leak" Claims**: Members of the Mistral Discord chat, including `@ethux`, `@casper_ai`, and `@mrdragonfox`, engaged in discussions about whether a newly emerged model was a leaked version of Mistral medium or not, referencing its performance, origins, and composition. The model in question, linked to [huggingface.co/miqudev/miqu-1-70b](https://huggingface.co/miqudev/miqu-1-70b) by `@alexworteega`, was eventually identified not to be Mistral or a leak but perhaps a LLaMa model fine-tuned with Mistral data.

- **Technical Discussion on Model Performance and Quantization**: `@mrdragonfox`, `@elvinrath`, and `@i_am_dom` engaged in a nuanced discussion about the impact of quantization on AI models, particularly in terms of performance degradation. It was noted that dense models, like the one speculated to be a close approximation of Mistral, could suffer more significantly from quantization effects.

- **Calls for Official Clarification on Rumored Leaks**: `@dillfrescott` expressed a desire for Mistral staff to address rumors circulating about the potential leak of a Mistral medium model to quell speculation. `@mrdragonfox` responded, emphasizing that if the rumors had any merit, the company would have already taken action.

- **Reducing Tokens in Text-Based Adventure Games**: `@ewanhc` inquired about strategies to minimize token usage in text-based adventure games without compromising the narrative's quality. Various solutions, including embedding and retrieval methods and prompt compression, were discussed, with a link to GitHub - [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua) shared by `@akshay_1` as a potential avenue for exploration.

- **VRAM Requirements for Running Mistral Models**: `@batot4968` reported issues running Mistral AI 7B v0.1 models on a 6GB VRAM card, contrary to claims made on the official website. The discussion highlighted discrepancies between the stated system requirements and user experiences, raising questions about VRAM sufficiency for Mistral models.

**Links mentioned**:

- [GitHub - mistralai/mistral-src: Reference implementation of Mistral AI 7B v0.1 model.](https://github.com/mistralai/mistral-src): Reference implementation of Mistral AI 7B v0.1 model. - GitHub - mistralai/mistral-src: Reference implementation of Mistral AI 7B v0.1 model.
- [GitHub - microsoft/LLMLingua: To speed up LLMs&#39; inference and enhance LLM&#39;s perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression with minimal performance loss.](https://github.com/microsoft/LLMLingua): To speed up LLMs&amp;#39; inference and enhance LLM&amp;#39;s perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression with minimal performance loss.  - GitH...

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1201449357183037441) (8 messages🔥): 

- **Training Mistral in a New Language**: `@nticaric` raised a concern about the effectiveness of continuing the training of the **Mistral model in a new language** due to potential vocabulary mismatches. `@vhariational` advised against starting from scratch due to the significant resources and data required.
- **Pretraining Might Be Necessary**: In the discussion about the best approach for training Mistral in a new language, `@mrdragonfox` suggested that pretraining might be necessary, although no specific method or data was mentioned.
- **Inquiry about BakLLaVA-1 Finetuning Resources**: `@attnmamba_15242` asked about the GPU resources and time requirement for finetuning **BakLLaVA-1**, providing links to the project’s page on Hugging Face and the collaboration behind it. BakLLaVA-1 is a blend of Mistral 7B base and LLaVA 1.5 architecture.
- **BakLLaVA-1’s Relevance to Mistral Questioned**: `@mrdragonfox` questioned the relevance of BakLLaVA-1 information to Mistral discussions and suggested seeking information directly from the BakLLaVA GitHub repository.
- **Difficulty Fine-tuning Mistral 7B on Dolly Dataset**: `@bishwa3819` is experiencing challenges in reducing training loss while fine-tuning **Mistral 7B on the dolly dataset** using LoRA. They shared their configuration and sought advice on improving the fine-tuning process.

**Links mentioned**:

- [SkunkworksAI/BakLLaVA-1 · Hugging Face](https://huggingface.co/SkunkworksAI/BakLLaVA-1): no description found
- [llava-hf/bakLlava-v1-hf · Hugging Face](https://huggingface.co/llava-hf/bakLlava-v1-hf): no description found

  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (1 messages): 

nk_pas: Write prompts everywhere and run Mistral plateforme in a single key stroke
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1201464015990313072) (28 messages🔥): 

- **Mistral API Rate Limits Express Concern**: `@arnaud_35886` queried about the **maximum rate limit** for using Mistral APIs in production, hinting at a need greater than the current offering. Initial confusion about the nature of the limit (rate vs cost) was clarified, and `@casper_ai` noted that approval for rate limit increases typically depends on usage history, starting at **2 requests/second**.

- **Token and Embedding Limits Under Scrutiny**: `@dierkdroth` sought clarification on token limits for **Mistral API text embeddings** and received a mix of direct and inferred answers. `@sophiamyang` confirmed the **max token limit** to be **32k for tiny/small/medium** models and **8192 for the embedding API**.

- **Explorations in Mistral's Tokenization**: In pursuit of details regarding the tokenizer used by Mistral's models, `@dierkdroth` inquired about the specific tokenizer and referenced a **JavaScript implementation**. `@sophiamyang` acknowledged the lack of documentation and promised an update, whereas `@vhariational` compared it to tokenizers found in **HuggingFace repositories**.

- **Java Client Integration Proposal**: `@carloszela` proposed adding a **new client** for Java users to the official Mistral documentation, showcasing an ongoing project on GitHub named **langchain4j** supporting **Mistral AI models**.

- **Awaiting Fixes for Early Stopping Issue**: `@digitalphotographer` followed up on a previously reported issue regarding **early stopping behavior** in Mistral's platform, looking for updates after sending notebooks for reproduction of the error to `@sophiamyang` and another.

**Links mentioned**:

- [Yarn](https://classic.yarnpkg.com/en/package/mistral-tokenizer-js),): Fast, reliable, and secure dependency management.
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/.): We provide client codes in both Python and Javascript.
- [GitHub - langchain4j/langchain4j: Java version of LangChain](https://github.com/langchain4j/langchain4j): Java version of LangChain. Contribute to langchain4j/langchain4j development by creating an account on GitHub.

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1201587518416629800) (8 messages🔥): 

- **New AI Research Paper Alert by sk5544**: sk5544 shared a new AI research paper titled [AI Techniques in High-Stakes Decision Making](https://arxiv.org/abs/2401.14446) by authors including Stephen Casper, Carson Ezell, and others, aiming to contribute more to public knowledge.
  
- **Biden Administration's AI Regulation Fact Sheet Criticized**: `@exirae` critiqued the Biden Administration's new [AI regulation fact sheet](https://www.whitehouse.gov/briefing-room/statements-releases/2024/01/29/fact-sheet-biden-harris-administration-announces-key-ai-actions-following-president-bidens-landmark-executive-order/), pointing out a lack of clarity in policy direction and expressing concerns over the proposed AI/ML education in K-12 schools.

- **Exclusion of NIST AISI and Proliferation of Taskforces Noted**: `@hyperion.ai` observed that the Biden administration's fact sheet on AI regulation did not mention tasks assigned to the NIST AISI and commented on the contemporary popularity of taskforces in policy initiatives.

- **Skepticism over Taskforces Expressed**: `@clockrelativity2003` and `@.undeleted` expressed skepticism towards the effectiveness of taskforces, suggesting they are more about political gesturing than actual problem-solving.

- **Concerns over AI/ML Education in Schools**: Both `@exirae` and `@.undeleted` voiced concerns about implementing AI/ML education in K-12 schools, fearing potential chaos due to the immaturity of school-aged children and the inadequacies of the American public school system.

**Links mentioned**:

- [Fact Sheet: Biden-Harris Administration Announces Key AI Actions Following President Biden’s Landmark Executive Order | The White House](https://www.whitehouse.gov/briefing-room/statements-releases/2024/01/29/fact-sheet-biden-harris-administration-announces-key-ai-actions-following-president-bidens-landmark-executive-order/): Three months ago, President Biden issued a landmark Executive Order to ensure that America leads the way in seizing the promise and managing the risks of artificial intelligence (AI). The Order direct...
- [Black-Box Access is Insufficient for Rigorous AI Audits](https://arxiv.org/abs/2401.14446): External audits of AI systems are increasingly recognized as a key mechanism for AI governance. The effectiveness of an audit, however, depends on the degree of system access granted to auditors. Rece...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1201445874518208522) (119 messages🔥🔥): 

- **Innovative Ideas on Model Efficiency and Training**: `@nshepperd` discussed techniques for reducing model size by generating vocab matrices from seeds and exploring different factorization methods for the softmax bottleneck issue. They mentioned leveraging a fused matmul kernel for size reduction and experimented with preconditioning gradients to emulate training trajectories of untied weight transformers.

- **Exploring Softmax Bottleneck Solutions**: Various community members, including `@nshepperd`, `@wonkothesensible`, and `@stefangliga`, delved into the softmax bottleneck problem, discussing alternatives like sigsoftmax and multifaceted softmax. They debated the potential impact of these alternatives on perplexity measurements and distribution errors, hinting at the complexity of fully addressing this issue.

- **Curiosity About Datasets and Quality Signals**: `@micpie` inquired about high-quality datasets with extensive metrics, similar to RedPajama-Data-V2, highlighting a gap in easily finding such resources on platforms like HF hub due to the lack of metadata search functionality.

- **Contributions to Language Modeling and Data Preprocessing Insights**: Links to research on improving language model perplexity through insights from synthetic experiments were shared by `@random_string_of_character` and `@leegao_`, while `@laomein` and `.the_alt_man` discussed a specific code snippet for data preprocessing, illustrating common challenges and clarifications in model development.

- **Reflections on Attention Mechanisms and Alternative Activations**: The conversation spanned critical examinations of softmax and attention mechanisms by `@catboy_slim_`, `@fern.bear`, and `@stefangliga`, considering replacements like sigmoid functions and questioning the intuitive justification behind softmax’s widespread use. They discussed innovative approaches like ghost attention and the potential of rethinking probability distributions within neural networks to address outlier features and quantization challenges.

**Links mentioned**:

- [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380): Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance o...
- [MoE-LLaVA: Mixture of Experts for Large Vision-Language Models](https://arxiv.org/abs/2401.15947): For Large Vision-Language Models (LVLMs), scaling the model can effectively improve performance. However, expanding model parameters significantly increases the training and inferring costs, as all mo...
- [Learning Universal Predictors](https://arxiv.org/abs/2401.14953): Meta-learning has emerged as a powerful approach to train neural networks to learn new tasks quickly from limited data. Broad exposure to different tasks leads to versatile representations enabling ge...
- [Tweet from Ekin Akyürek (@akyurekekin)](https://x.com/akyurekekin/status/1751986985386828117): Can insights from synthetic experiments and interpretability lead to real improvements in language modeling? We: &gt; propose a formal model for in-context learning  &gt; uncover &#34;n-gram heads&#34...
- [Softmax Bottleneck Makes Language Models Unable to Represent Multi-mode Word Distributions](https://aclanthology.org/2022.acl-long.554/): Haw-Shiuan Chang, Andrew McCallum. Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2022.
- [Sigsoftmax: Reanalysis of the Softmax Bottleneck](https://arxiv.org/abs/1805.10829): Softmax is an output activation function for modeling categorical probability distributions in many applications of deep learning. However, a recent study revealed that softmax can be a bottleneck of ...
- [Papers with Code - The Methods Corpus](https://paperswithcode.com/methods): 2189 methods • 117443 papers with code.
- [Circuits Updates - January 2024](https://transformer-circuits.pub/2024/jan-update/index.html): no description found

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1201584973656887417) (12 messages🔥): 

- **Machine Unlearning in LLMs Faces Weak Baselines**: `@stellaathena` criticizes the current state of machine unlearning research for large language models (LLMs), highlighting that applied studies often use weak baselines and suggesting an unexplored method involving training the model with gradient ascent on bad data. They propose an experiment utilizing gradient ascent on Pythia with a subset of the Pile related to a specific topic. [See the discussion](https://fixupx.com/blancheminerva/status/1752023198147780907?).

- **TOFU Benchmark Introduced for Unlearning**: `@baidicoot` and `@stellaathena` discuss the recent introduction of the TOFU benchmark, which aims to facilitate a deeper understanding of unlearning in models by comparing methods like gradient ascent. The TOFU paper, available at [arXiv:2401.06121](https://arxiv.org/abs/2401.06121), involves a dataset of synthetic author profiles to study the effectiveness of unlearning strategies.

- **Exploring Model Lineage with a Primitive Solution**: `@fblgit` offers a simple tool for identifying relationships or "blood-lines" between models. This solution is presented as functional yet basic, inviting others curious about model lineage to reach out for details.

- **Alternative to The Pile Download Link Provided**: Following a report by `@1_glados` about a 404 error with the original Pile download link, `@random_string_of_character` provides an alternative magnet link for accessing this dataset.

- **Clarification on Unlearning and Data Exchangeability**: `@statslime` questions the implications of data exchangeability in the context of machine unlearning discussed by `@stellaathena`, leading to a clarification that the evaluation involved checking the percentage of identical weights between models.


**Links mentioned**:

- [TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121): Large language models trained on massive corpora of data from the web can memorize and reproduce sensitive or private data raising both legal and ethical concerns. Unlearning, or tuning models to forg...
- [Tweet from Stella Biderman (@BlancheMinerva)](https://fixupx.com/blancheminerva/status/1752023198147780907?)!): @Wetassprior @daphneipp Is “train the model with gradient ascent on bad data” an effective technique for machine unlearning? The extent to which the answer is &#34;no&#34; is a measure of how non-exch...

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1201478415589589002) (2 messages): 

- **Interest in New Work**: `@johnnysands` expressed strong interest in recent developments and has plans to run some tests on the work. They added a friendly emoji to show enthusiasm.
- **Query about Offline Task Downloading**: `@damon_29077` inquired if there's a way to download all tasks and cache them for offline use. This question hints at a desire for more flexible usage scenarios.
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1201611721375223859) (6 messages): 

- **Building Apex on Frontier's Struggle**: `@groggyrhombus` highlighted that building **apex** on OLCF's Frontier (AMD MI250xs) is time-consuming. By specifying the architectures explicitly for the build, the time required was notably reduced.

- **CUDA Might Offer Relief with Flags**: In connection to building apex, `@groggyrhombus` suggested that **CUDA** could have similar flags to reduce build time, implying a possible optimization route for those working in CUDA environments.

- **Custom Compilation Approach for Apex**: `@catboy_slim_` detailed a custom method for compiling **apex**, involving the use of specific flags not mentioned in the official documentation. This method entails compiling with `pip install` and then creating a wheel file that can be easily distributed and imported.

- **Solution to Apex's Build Time Across Platforms**: `@triggerhappygandhi` confirmed the general issue of apex's lengthy build time across various platforms and appreciated the architecture-specific build time reduction tip provided by `@groggyrhombus`. This acknowledgment underscores the widespread nature of the problem and the value of the shared solution.
  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1201476241031712788) (94 messages🔥🔥): 

- **Community Guidelines on PR Tagging**: `@cakiki` gently reminded users not to tag people for PR reviews, especially when they are unrelated to the repo, emphasizing community etiquette. This led to a discussion with `@kopyl`, who questioned the rule, resulting in `@cakiki` referring the matter to an admin for fairness.
  
- **Patience is Key for PR Reviews**: `@not_lain` advised patience regarding a newly created pull request, highlighting it was only 11 hours old and reviewers might not have gotten to it yet. This advice was aimed at `@kopyl`, who was seeking clarification on tagging protocols for faster PR reviews.

- **Exploring Continuous Neural Network Processes**: `@bshvp` posed an intriguing question about developing neural network models that operate continuously, mimicking human brain functions. They suggested feeding these models a constant stream of sensory inputs to truly evaluate consciousness, sparking a conversation on innovative approaches to AI development.

- **Success Story of a Deepfake Detection Pipeline**: `@not_lain` shared the achievement of their deepfake detection pipeline reaching 147 downloads in less than a week and provided a detailed explanation, including a link (https://huggingface.co/not-lain/deepfake) for the community. This sparked interest from `@adiinx`, who sought advice on deepfake content generation and voice cloning, leading `@not_lain` to clarify the pipeline's limitations and applications.

- **Fedora Versus Ubuntu for Machine Learning on AMD Hardware:** `@cilginflix` shared challenges with using CPU for ML tasks due to issues with onnxruntime and AMD's ROCm on Fedora, leading to a discussion on operating system compatibility for ML. `@kopyl` suggested NVIDIA hardware for ML, but `@cilginflix` countered that AMD works on other distributions like Ubuntu or Manjaro, albeit with different performance and stability outcomes.

**Links mentioned**:

- [Instantiating a big model](https://huggingface.co/docs/transformers/v4.24.0/en/big_models#sharded-checkpoints): no description found
- [PNG to SVG (Online &amp; Free) — Convertio](https://convertio.co/png-svg/): no description found
- [not-lain/deepfake · Hugging Face](https://huggingface.co/not-lain/deepfake): no description found
- [Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration](https://arxiv.org/abs/2401.13979): In this paper, we propose an architecture to harness the collective knowledge of multiple trained LLMs to create a new state-of-the-art. At the core of this framework is a LLM-based orchestrator that ...
- [GitHub - Leeroo-AI/leeroo_orchestrator: The implementation of &quot;Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration&quot;](https://github.com/leeroo-ai/leeroo_orchestrator): The implementation of &amp;quot;Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration&amp;quot; - GitHub - Leeroo-AI/leeroo_orchestrator: The implementation of &amp;quot;Leeroo Orc...
- [Home | leeroo](https://www.leeroo.com/): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1201484270196240394) (10 messages🔥): 

- **A celebratory moment for @osanseviero**: @osanseviero only shared a brief but heartfelt "**Congrats!!**", to which @not_lain responded with **"thank youuuu"** and a special Hugging Face emoji. The context of the celebration remains a mystery.
- **@erlonidasap seeks guidance on model training**: Starting a new project, **@erlonidasap** asked the community for tips on how to train a model on a specific string dataset. It's related to a company project, but no further details were provided.
- **Comprehensive advice from @not_lain**: In response to @erlonidasap's query, **@not_lain** suggested first identifying the problem (**text-classification, text-generation, QA, etc.**), then selecting a suitable model from the Hugging Face Hub, and finally finetuning it on the specific data.
- **@kaizen0340 inquires about speech translation training**: Looking for resources, **@kaizen0340 asked the community** if there's any tutorial or repository on how to train a speech translation using `speechencodedecoder`. Unfortunately, there was no response provided in the messages.
- **@antiraedus shares a weekly update and reflections**: Highlighting the importance of setting correct habits and maintaining focus, **@antiraedus shared** insights from their week, including personal development in flutter, exercise, and brainstorming app ideas. They emphasized the goal of building a strong foundation for consistent productivity.
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1201657217959592026) (3 messages): 

- **Deep Learning Breakthroughs in Medical Imaging**: `@maro_mich` shared an open access article from IEEE detailing how Deep Learning is revolutionizing **Medical Image Analysis**. Check out the findings [here](https://ieeexplore.ieee.org/document/8241753).
- **Another Must-Read Medical Imaging Paper**: `@aryan_1098` encourages the community to explore a paper on **Medical Image Analysis** available on IEEE. The study can be found [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10181037&tag=1).

**Links mentioned**:

- [Deep Learning Applications in Medical Image Analysis](https://ieeexplore.ieee.org/document/8241753): The tremendous success of machine learning algorithms at image recognition tasks in recent years intersects with a time of dramatically increased use of electronic medical records and diagnostic imagi...
- [Convolutional Neural Networks for Image Emotion Recognition by fusing Differential and Supplementary Information](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10181037&tag=1): Emotions arise out of complex phenomena, which are believed to have a biological basis. Neuroscience research has demonstrated that emotions are related to distinct patterns of brain activity and the ...

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1201507373152473118) (7 messages): 

- **Shamanic Animation Intrigues**: `Amanita` shared an animated fly agaric shaman, described as stylized. No links or further details were provided.

- **Comprehensive AI Model Demo Space by myg5702**: `myg5702` launched a [Hugging Face Spaces demo](https://huggingface.co/spaces/FumesAI/Best-Image-Models-Demo) showcasing a variety of AI models including *sdxl*, *fooocus*, and *dream shaper xl turbo* among others. This Space seems to be a one-stop showcase of cutting-edge image models.

- **Innovative Resume Space by not_lain**: `not_lain` created a [resume question answering space](https://huggingface.co/spaces/not-lain/resume-qa) on Hugging Face, designed to make it easier for recruiters to find information. They encourage users to heart react the space if they appreciate the innovative concept.

- **Mistral on Apple Silicon Video by monirul_1slam**: `monirul_1slam` released a [YouTube video](https://www.youtube.com/watch?v=cjl2ADP8JLQ&t=79s) demonstrating the performance of the Mistral model on Apple Silicon, specifically geared towards tech enthusiasts and developers interested in machine learning performance on different hardware.

- **Challenging ABSA Model Creation by joshuasundance**: `joshuasundance` accepted a challenge to train and upload a SetFitABSA model for laptop reviews, sharing [links to their work](https://huggingface.co/joshuasundance/setfit-absa-all-MiniLM-L6-v2-laptops-aspect) on Hugging Face. This contribution highlights the active collaboration and learning in aspects of sentiment analysis within the community.

- **Elvis-like Vocals Captured by .bigdookie**: `.bigdookie` shared their excitement about a session where their vocalist resembled Elvis, signalling a whimsical and light-hearted moment in creative music production. The context remains in the space of amusement and tease for future releases.

- **Excel/CSV to Database Tables App by impl66**: `impl66` developed a [Gradio app](https://huggingface.co/spaces/sid27/tables) that converts Excel/CSV files into database tables and subsequently answers user questions on those tables, showcasing practical utility in data management and retrieval.

**Links mentioned**:

- [Best Image Models Demo - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Best-Image-Models-Demo): no description found
- [Resume Qa - a Hugging Face Space by not-lain](https://huggingface.co/spaces/not-lain/resume-qa): no description found
- [MLX | Mistral-7B-Instruct on Apple Silicon](https://www.youtube.com/watch?v=cjl2ADP8JLQ&t=79s): Can you run Mistral-7B-Instruct-v0.2 from Mistral AI on Apple Silicon with MlX? Let&#39;s find out. -------------------------------------------------------------...
- [Tables - a Hugging Face Space by sid27](https://huggingface.co/spaces/sid27/tables): no description found
- [Tweet from thecollabagepatch (@thepatch_kev)](https://x.com/thepatch_kev/status/1752129930404696134?s=46): when your singer kinda sounds like elvis for no reason one day  @fffiloni &#39;s dreamtalk needs to come out 😂   this week we just havin fun in the captains chair  next week... @_buildspace
- [joshuasundance/setfit-absa-all-MiniLM-L6-v2-laptops-aspect · Hugging Face](https://huggingface.co/joshuasundance/setfit-absa-all-MiniLM-L6-v2-laptops-aspect): no description found
- [joshuasundance/setfit-absa-all-mpnet-base-v2-laptops-polarity · Hugging Face](https://huggingface.co/joshuasundance/setfit-absa-all-mpnet-base-v2-laptops-polarity): no description found

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1201616685459849277) (11 messages🔥): 

- **Election Tampering Discussed with AI**: `@iloveh8` raised a query about AI products that could aid a political candidate in winning an election, mentioning deepfakes as a potential use case. `@vipitis` responded with examples such as fraud, supplying additional votes, redrawing borders for an unrepresentative lead, and even poisoning the competition.
- **Controversial Use of AI in Politics**: `@vipitis` replied to the discussion on AI's application in political campaigns by suggesting any means necessary, including fraudulent actions, assuming no rules were in place.
- **Seeking Ethical Alternatives in Political Campaigns**: In contrast to the controversial uses, `@chad_in_the_house` suggests creating numerous videos of a political candidate as a more legal and ethical form of AI application in campaigns.
- **Interest in Technical Details for Campaign Videos**: Following up on the discussion of creating videos, `@iloveh8` expressed curiosity about the technical aspects, highlighting the ongoing media discussion around such AI-driven content.
- **Inquiry on Text to Image Diff Models**: `@syed2658` shifted the conversation towards technical advice, asking for suggestions on the best text-to-image diffusion models. This indicates a broader interest in AI applications beyond political implications.

**Links mentioned**:

[How to WIN an Election | Ordinary Guide](https://youtu.be/xOBmKtQVlo0): Follow me on twitter: https://twitter.com/ordinarytingsSupport the Channel on Patreon: https://www.patreon.com/ordinarythingsHow do you win an election? By l...

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages): 

swetha98: I am getting this error while running code for fine tuning donut docvqa .
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1201437672976158780) (3 messages): 

- **GPU acceleration confuses with llama-cpp**: User `@.sgp` inquired about difficulties in achieving GPU acceleration while using **llama-cpp**, seeking insights or solutions from the community.
- **Amazement at GPU requirement for llama-cpp**: Responding to `@.sgp`'s dilemma, `@frosty04212` expressed astonishment at the requirement for **128 GPU layers** for operating llama-cpp, wondering about the specifications of the GPU involved.
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1201616685459849277) (11 messages🔥): 

- **Discussing AI's Role in Elections Gets Dark**: `@iloveh8` sparked a conversation asking about specific AI use cases that could help a political candidate win an election. `@vipitis` responded controversially, suggesting **fraud** as a method by emphasizing actions like supplying additional votes or redrawing district borders for an unrepresentative lead.
- **From Strategy to Digital Content**: Shifting from the grim suggestions, `@chad_in_the_house` proposed a more lawful approach by creating numerous videos of a political candidate to generate more content and visibility.
- **The Technicalities of Media Influence**: Following up on the suggestion to create more videos of a candidate, `@iloveh8` expressed curiosity about the technical details behind this strategy, especially considering its frequent discussion in the media.
- **In Search of Text-to-Image Models**: `@syed2658` inquired about recommendations for the best text-to-image diffusion models, diverging from the election-focused conversation to seek advice on AI models.
- **An Unorthodox Election Guide**: Besides giving controversial advice, `@vipitis` also shared a **YouTube video** titled "**How to WIN an Election | Ordinary Guide**" ([watch here](https://youtu.be/xOBmKtQVlo0)), providing a humorous take on winning elections.

**Links mentioned**:

[How to WIN an Election | Ordinary Guide](https://youtu.be/xOBmKtQVlo0): Follow me on twitter: https://twitter.com/ordinarytingsSupport the Channel on Patreon: https://www.patreon.com/ordinarythingsHow do you win an election? By l...

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1201500756583391363) (92 messages🔥🔥): 

- **ColBERT vs. Traditional Embeddings Discussion Heats Up**: `@sarav1n` sparked a debate on the real-world usage of **ColBERT** vs. current embedding models, sharing insights from [Simon Willison's article](https://til.simonwillison.net/llms/colbert-ragatouille), which highlights ColBERT's unique approach to computing similarity scores. The conversation included a desire for a direct comparison between ColBERT and conventional embedding models, with members like `@swyxio` and `@420gunna` expressing interest but noting a lack of comparative data.
  
- **Simon Willison Offers Fresh Perspectives on AI and Software**: `@mdcker` shared a [feature on Simon Willison](https://www.theregister.com/2024/01/24/willison_ai_software_development) discussing AI's impact on software development and related issues. The community appreciates Willison's accessible explanations of complex AI topics, sparking further discussion on the human element in AI software development.
  
- **"Arc Search" Revolutionizes Mobile Web Browsing**: `@mdcker` introduced **Arc Search**, a new iOS app designed to streamline web searches by building webpages based on user queries, potentially changing how users interact with search engines and browser apps. The community speculated on its impact and potential competition with traditional search engines.
  
- **Voyage-Code-2 Promises Better Code Retrieval**: `@gsegato` announced the release of **Voyage-Code-2**, an embedding model claiming superior performance in code-related applications. This sparked a conversation, led by `@swyxio`, about the value proposition of specializing in embeddings and the model's anticipated evaluation on benchmarks like MTEB.
  
- **Anthropic's Claude Models Under the Microscope**: `@philltornroth` questioned why Anthropic's Claude models aren't receiving more attention compared to OpenAI, suggesting their performance in tasks like summarization and retrieval could be more competitive than perceived. This led to a nuanced discussion on the specific use cases where Claude models excel and areas where improvements are needed, contributing to the ongoing debate on the best AI models for various applications.

**Links mentioned**:

- [🦅 Eagle 7B : Soaring past Transformers with 1 Trillion Tokens Across 100+ Languages (RWKV-v5)](https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers): A brand new era for the RWKV-v5 architecture and linear transformer&#x27;s has arrived - with the strongest multi-lingual model in open source today
- [Tweet from Nous Research (@NousResearch)](https://x.com/nousresearch/status/1752051008736550917?s=46&t=90xQ8sGy63D2OtiaoGJuww): Today we are announcing our latest project, an effort to provide a new evaluation system for open source models. Traditional benchmarking leans heavily on public datasets which can be easy to game and...
- [Arc Search combines browser, search engine, and AI into something new and different](https://www.theverge.com/2024/1/28/24053882/arc-search-browser-web-app-ios): This might be the most interesting AI search tool yet. 
- [AI software still needs the human touch, Willison warns](https://www.theregister.com/2024/01/24/willison_ai_software_development): Code assistance is like having a weird intern who memorized the docs
- [Exploring ColBERT with RAGatouille](https://til.simonwillison.net/llms/colbert-ragatouille): I&#39;ve been trying to get my head around ColBERT .
- [Tweet from AI at Meta (@AIatMeta)](https://x.com/aiatmeta/status/1752013879532782075): Today we’re releasing Code Llama 70B: a new, more performant version of our LLM for code generation — available under the same license as previous Code Llama models.  Download the models ➡️ https://bi...
- [voyage-code-2: Elevate Your Code Retrieval](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/): TL;DR – We are thrilled to introduce voyage-code-2, our latest embedding model specifically tailored for semantic retrieval of codes and related text data from both natural language and code querie…
- [New GitHub Copilot Research Finds 'Downward Pressure on Code Quality' -- Visual Studio Magazine](https://visualstudiomagazine.com/articles/2024/01/25/copilot-research.aspx): 'We find disconcerting trends for maintainability.'
- [OpenAI Status](https://status.openai.com/): no description found
- [Tweet from Nous Research (@NousResearch)](https://x.com/NousResearch/status/1744865872563618128?s=20): Nous Research is excited to announce the closing of our $5.2 million seed financing round.   We&#39;re proud to work with passionate, high-integrity partners that made this round possible, including c...
- [DSPy Explained!](https://youtu.be/41EfOY0Ldkc?si=Be15s2zgG0yTyhR0): Hey everyone! Thank you so much for watching this explanation of DSPy! DSPy is a super exciting new framework for developing LLM programs! Pioneered by frame...
- [Python SDK - Langfuse](https://langfuse.com/docs/sdk/python): Fully async and typed Python SDK. Uses Pydantic objects for data verification.
- [Tweet from Justine Moore (@venturetwins)](https://x.com/venturetwins/status/1752022393768607814?s=46&t=90xQ8sGy63D2OtiaoGJuww): A major VFX studio owned by Netflix is hiring in a bunch of AI roles:  Generative imaging, workflow design, model training, data acquisition, and even ML researchers  We&#39;re going to be seeing a lo...
- [GitHub - FanaHOVA/smol-podcaster: smol-podcaster is your autonomous podcast production intern 🐣](https://github.com/FanaHOVA/smol-podcaster): smol-podcaster is your autonomous podcast production intern 🐣 - GitHub - FanaHOVA/smol-podcaster: smol-podcaster is your autonomous podcast production intern 🐣
- [GitHub - FanaHOVA/smol-scheduler: 🐣🕐📅  A simple utility to draft scheduling emails.](https://github.com/FanaHOVA/smol-scheduler):  🐣🕐📅  A simple utility to draft scheduling emails. - GitHub - FanaHOVA/smol-scheduler: 🐣🕐📅  A simple utility to draft scheduling emails.
- [Scanline VFX - Research Scientist, Computer Graphics, Computer Vision, and Machine Learning](https://jobs.lever.co/scanlinevfx/b6a54fd8-e4bb-4165-9b6d-ac67859cb0c0): As a Senior Research Scientist, you will develop new technologies to revolutionize live-action content creation and storytelling. You will conduct applied research in computer vision and computer grap...
- [[AINews] RWKV &quot;Eagle&quot; v5: Your move, Mamba](https://buttondown.email/ainews/archive/ainews-mamba-meets-rwkv-eagle-v5/): AI Discords for 1/27-28/2024. We checked 20 guilds, 297 channels, and 10073 messages for you. Estimated reading time saved (at 200wpm): 826 minutes. We are...

  

---



### DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1201658589815779358) (5 messages): 

- **MIQU Model Sparks Mixtral Medium Leak Speculation**: `@aiui` raised a question about the **MIQU model** being a potential **Mixtral medium leak**, sparking a conversation about its origins and characteristics.
- **Debunking the MIQU Model-Mixtral Connection**: `@sebastian.bodza` quickly addressed the speculation around the **MIQU model** being a Mixtral medium leak, stating that **these rumors were already debunked**.
- **Clarification Tweet from Nisten**: For further clarity, `@sebastian.bodza` shared a [tweet by Nisten](https://twitter.com/nisten/status/1751841882831716578) that helps debunk the rumors surrounding the MIQU model.
- **MIQU Model and LLaMA Tokenizer Connection Highlighted**: Additionally, `@sebastian.bodza` highlighted that the MIQU model uses a **LLaMA tokenizer**, sharing another [relevant tweet](https://twitter.com/Nota_Cant/status/1751861787170148368) for more details.
  

---


### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1201748102571380746) (6 messages): 

- **Exploring German Fine-tuning on Phi 2**: `@philipmay` raised doubts about **Phi 2** having German text training, prompting a discussion on its potential fine-tuning. `@johannhartmann` shared an attempt at finetuning it for German using the OASST-DE dataset but noted unimpressive results from a single epoch attempt, suggesting a more comprehensive approach with additional datasets and epochs might be worth exploring.
- **Tiny Llama Gets a German Upgrade**: `@johannhartmann` documented an effort to improve **Tiny Llama** with German DPO (Data Processing Overlay), linking to a walkthrough on GitHub ([TinyLlama-1.1B-Chat-v1.0-german-dpo-Openllm_de.md](https://gist.github.com/johannhartmann/6cb0fee8103869e6e58d7e1956ce9c99)). This method showed some positive outcome, hinting at the potential of language-specific training enhancements.
- **Curious About German Orca DPO DataSet?**: `@philipmay` inquired about the existence of a German Orca DPO dataset, which led `@johannhartmann` to share a link to a somewhat makeshift dataset on Hugging Face ([mayflowergmbh/intel_orca_dpo_pairs_de](https://huggingface.co/datasets/mayflowergmbh/intel_orca_dpo_pairs_de)), detailing its use of AzureML for translation and HermEO as the rejected model.
- **WRAP Aims to Enhance Pre-training**: `@bjoernp` highlighted an innovative approach by Apple in a recent paper on **Web Rephrase Augmented Pre-training (WRAP)** which aims to improve data quality and reduce pre-training times by paraphrasing web documents into styles like Wikipedia or Q&A ([arXiv:2401.16380](https://arxiv.org/abs/2401.16380)). This method, according to the abstract, has significantly sped up pre-training efforts, hinting at its potential widespread adoption among major LLM players.

**Links mentioned**:

- [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380): Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance o...
- [TinyLlama-1.1B-Chat-v1.0-german-dpo-Openllm_de.md](https://gist.github.com/johannhartmann/6cb0fee8103869e6e58d7e1956ce9c99): GitHub Gist: instantly share code, notes, and snippets.
- [mayflowergmbh/intel_orca_dpo_pairs_de · Datasets at Hugging Face](https://huggingface.co/datasets/mayflowergmbh/intel_orca_dpo_pairs_de): no description found

  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1201458885110808616) (55 messages🔥🔥): 

- **Exploring CodeT5 Embeddings**: `@sebastian.bodza` inquired about implementing Salesforce's CodeT5 embeddings for code, struggling with the practical aspects. No insights were provided by `@bjoernp` due to a lack of experience with these models.

- **First Glance at Text Generation Attempts**: `@sebastian.bodza` shared their initial results in text-generation and requested feedback. A specific notebook was shared, [Embedding_Training/05_preprocess_texts.ipynb](https://github.com/SebastianBodza/Embedding_Training/blob/main/05_preprocess_texts.ipynb), for collaborators to review.

- **Defining "Hard Negatives" in Training**: A considerable amount of discussion focused on creating high-quality "hard negatives" for training data. `@sebastian.bodza` and `@philipmay` debated the criteria, with emphasis on the importance of hard negatives not answering the question directly to prevent training issues.

- **Iterative Improvements on Training Prompts**: `@sebastian.bodza` iteratively improved the prompts used for generating training data, aiming to fine-tune the balance between positive examples and hard negatives. These modifications were driven by feedback and the quest for optimally structured data sets.

- **Misconceptions About Retrieval Models Discussed**: Clarity was sought on the purpose and structure of datasets for DPR and RAG training, particularly regarding context and the kind of hard negatives. `@sebastian.bodza` clarified that their work aims to generate a passage retrieval dataset for future RAG use, with examples varying in quality rather than content relevance.

**Links mentioned**:

- [wp-rag-dpo/04_it01_extract_positive_answers.ipynb at main · telekom/wp-rag-dpo](https://github.com/telekom/wp-rag-dpo/blob/main/04_it01_extract_positive_answers.ipynb): Contribute to telekom/wp-rag-dpo development by creating an account on GitHub.
- [Embedding_Training/05_preprocess_texts.ipynb at main · SebastianBodza/Embedding_Training](https://github.com/SebastianBodza/Embedding_Training/blob/main/05_preprocess_texts.ipynb): Contribute to SebastianBodza/Embedding_Training development by creating an account on GitHub.
- [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368): In this paper, we introduce a novel and simple method for obtaining high-quality text embeddings using only synthetic data and less than 1k training steps. Unlike existing methods that often depend on...

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1201502992365535335) (46 messages🔥): 

- **Exploring LangChain Capabilities**: User `@irfansyah5572` requested assistance on running an answer chain and displaying the source in LangChain, linking to the project's [GitHub](https://github.com/langchain-ai/chat-langchain/blob/master/chain.py).

- **Tutorial Alert for RAG Frontend Development**: `@a404.eth` shared a new [YouTube tutorial](https://www.youtube.com/watch?v=xFWllDS6ZRw) focusing on building a frontend for interacting with PDF documents using LangChain and other technologies, asking for feedback on this second part of their tutorial series.

- **Custom Book Recommendation System with LangChain**: `@oscarjuarezx` is developing a demo book recommendation system based on user interests, utilizing PostgreSQL and various LangChain components, and is seeking advice on optimizing database searches.

- **Cache Issues with LlamaCPP Models**: Several users, including `@techexplorer0` and `@kapa.ai`, discussed the use of `InMemoryCache` for improving inference times with LlamaCPP models, with a link to a related [GitHub issue](https://github.com/langchain-ai/langchain/issues/2784), but noted challenges with cache effectiveness.

- **Parsing and Streaming Challenges in LangChain**: Users `@ibrobabs` and `@hiranga.g` exchanged experiences and issues with parsing JSON output and streaming from agents in LangChain, mentioning a helpful [GitHub example](https://github.com/langchain-ai/langchain/blob/master/templates/openai-functions-agent-gmail/openai_functions_agent/agent.py) and a [YouTube video](https://www.youtube.com/watch?v=08qXj9w-CG4&t=323s) for further guidance.

**Links mentioned**:

- [LangChain v0.1.0 Launch: Agents](https://www.youtube.com/watch?v=08qXj9w-CG4&t=323s): LangChain is the default way to allow LLMs to take actions.Jupyter Notebook (to follow along): https://github.com/hwchase17/langchain-0.1-guides/blob/master/...
- [chat-langchain/chain.py at master · langchain-ai/chat-langchain](https://github.com/langchain-ai/chat-langchain/blob/master/chain.py): Contribute to langchain-ai/chat-langchain development by creating an account on GitHub.
- [Chat With Your PDFs Part 2: Frontend - An End to End LangChain Tutorial. Build A RAG with OpenAI.](https://www.youtube.com/watch?v=xFWllDS6ZRw): In this video we are going to dive into part two of building and deploying a fully custom RAG with  @LangChain   and  @OpenAI  .  In this tutorial, code with...
- [langchain/templates/openai-functions-agent-gmail/openai_functions_agent/agent.py at master · langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/master/templates/openai-functions-agent-gmail/openai_functions_agent/agent.py): ⚡ Building applications with LLMs through composability ⚡ - langchain-ai/langchain
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/2784>),): ⚡ Building applications with LLMs through composability ⚡ - Issues · langchain-ai/langchain
- [OSD Bias Bounty](https://osdbiasbounty.com/sign-in?callbackUrl=https%3A%2F%2Fosdbiasbounty.com%2Fsign-in): no description found
- [Bug Bounty: ConductorAI - Bias Bounty Program  | Bugcrowd](https://bugcrowd.com/conductorai-ogbb?preview=ae06c13f786e06a1f9ff03d74230b7d5): Learn more about Conductor AI’s vulnerability disclosure program powered by Bugcrowd, the leader in crowdsourced security solutions.

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1201557933931118632) (3 messages): 

- **Local LLMs Elevate Web Browsing**: `@andrewnguonly` launched an open-sourced **LLM copilot Chrome extension**, named [Lumos](https://github.com/andrewnguonly/Lumos), leveraging **LangChain** and **Ollama** to enrich web browsing experiences with local LLMs. They encouraged feedback and shared a link to support the launch on [Product Hunt](https://www.producthunt.com/posts/lumos-4).

- **Building with Pre-written SQL as Tools**: `@shownaldo` was curious about incorporating SQL into a project, pondering if it meant having pre-written SQL for the agent to choose from. In response, `@johnny2x2` shared that they pivoted to including predefined SQL scripts as tools for their local LLM due to difficulties in having it write custom SQL code.

**Links mentioned**:

- [GitHub - andrewnguonly/Lumos: A RAG LLM co-pilot for browsing the web, powered by local LLMs](https://github.com/andrewnguonly/Lumos): A RAG LLM co-pilot for browsing the web, powered by local LLMs - GitHub - andrewnguonly/Lumos: A RAG LLM co-pilot for browsing the web, powered by local LLMs
- [ Lumos - Open source copilot for browsing the web powered by Ollama | Product Hunt](https://www.producthunt.com/posts/lumos-4): Lumos is an LLM co-pilot for browsing the web, powered by local LLMs. The Chrome extension is powered by Ollama!

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1201544993932857516) (3 messages): 

- **Beginner-Friendly Langchain Agents 2024 Update**: `@ryannolan` shared a [YouTube video](https://www.youtube.com/watch?v=WVUITosaG-g&ab_channel=RyanNolanData) titled **"Langchain Agents [2024 UPDATE] - Beginner Friendly"**. The video provides a new way to build agents with the latest Langchain update, aiming to expand the capabilities of the OpenAI API.

- **Building a RAG with LangChain and Unstructured, Part 2**: `@a404.eth` announced the release of their new tutorial, **"Chat With Your PDFs Part 2: Frontend - An End to End LangChain Tutorial. Build A RAG with OpenAI"**. It focuses on frontend development, including building the chat interface, displaying source documents, and streaming output, utilizing technologies like LCEL, React, and TypeScript. The tutorial is available at [YouTube](https://www.youtube.com/watch?v=xFWllDS6ZRw).

- **Excitement over Tutorial**: `@a404.eth` expressed enthusiasm for their tutorial on building a frontend for a RAG using LangChain with a succinct comment, **"Ok this is sick"**.

**Links mentioned**:

- [Langchain Agents [2024 UPDATE]  - Beginner Friendly](https://www.youtube.com/watch?v=WVUITosaG-g&ab_channel=RyanNolanData): In this Langchain video, we will explore the new way to build agents with Langchain update 0.1. With agents, we can expand the capability of the OpenAi API a...
- [Chat With Your PDFs Part 2: Frontend - An End to End LangChain Tutorial. Build A RAG with OpenAI.](https://www.youtube.com/watch?v=xFWllDS6ZRw): In this video we are going to dive into part two of building and deploying a fully custom RAG with  @LangChain   and  @OpenAI  .  In this tutorial, code with...

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1201567860930138152) (6 messages): 

- **Simple method to tackle complex classifications**: `@KarelDoostrlnck` introduces a new method using LLMs for complex classifications involving thousands of classes, like medical reactions and job skills. This approach involves inferring a set of predictions, then retrieving and re-ranking results. [IFTTT Announcement](https://twitter.com/llama_index/status/1752008109835559123)
- **LlamaIndex.TS updates**: New features and improved documentation have been launched for LlamaIndex.TS. [Read the announcement and updates here.](https://twitter.com/llama_index/status/1752075208905896265)
- **Hackathon in Silicon Valley with $16,000 prizes**: The LlamaIndex RAG-A-THON is an in-person event focusing on Retriever-Augmented Generation (RAG) technology to create advanced AI agents. It requires at least one team member to present physically, offering prizes up to $16,000. [Hackathon details](https://twitter.com/llama_index/status/1752086703437955199)
- **RAG Hackathon participation query**: `@rawwerks` inquires about attendees for the LlamaIndex RAG Hackathon, emphasizing its in-person requirement. [Context of query](https://twitter.com/llama_index/status/1752086703437955199)
- **Evaluating LlamaPacks for advanced querying**: `@wenqi_glantz` tested 7 pre-baked advanced query strategies, known as LlamaPacks, with @lighthouzai, showcasing how they can streamline choosing the best querying strategy for specific needs. [LlamaPacks evaluation](https://twitter.com/llama_index/status/1752131958552080650)

**Links mentioned**:

[LlamaIndex RAG Hackathon (in-person only)](https://t.co/j33mXMctJV): Think Beyond Chatbots: Unleashing the Potential of AI Agents

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1201510363783499866) (36 messages🔥): 

- **Local LlamaCpp Server for Rag Implementation**: `@techexplorer0` is setting up a local LlamaCpp server using instructions from [GitHub - abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#openai-compatible) and seeks advice on integrating it with a RAG application for making LLM calls.

- **Azure Analysis Services Query with Llama_Index**: `@meowmeow008` successfully connected Llama_Index to Azure SQL but is exploring options to connect it to Azure Analysis Services OLAP instead of SQL. 

- **Exploring Llama2 Licensing for Commercial Use**: `@sattyman` inquires about using Llama2 commercially and creating a finetuned version for their product. `@nerdai` linked to an [article](https://deepsense.ai/llama-2) suggesting Llama2 is available for commercial use with certain restrictions, and also recommended checking Meta's [official site](https://ai.meta.com/llama/) for fine print and licensing details.

- **Customizing LLM Output Format**: `@mysterious_avocado_98353` wanted to know how to customize output format to exclude specific words. `@nerdai` suggested modifying prompts as per the guidelines on [LlamaIndex Docs](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern.html#getting-and-setting-custom-prompts).

- **Issues with Query Engines in Llama Index**: `@coder_004_71487` discussed handling generic queries and specific SQL errors with Llama Index. `@nerdai` recommended using a RouterQueryEngine for mixed queries and directed `@coder_004_71487` to submit a bug report through [GitHub Issues](https://github.com/run-llama/llama_index/issues) and check out a demo on configuring a RouterQueryEngine for more intricate query handling.

**Links mentioned**:

- [Usage Pattern - LlamaIndex 🦙 0.9.39](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern.html#getting-and-setting-custom-prompts): no description found
- [GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#openai-compatible-web-server): Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.
- [Router Query Engine - LlamaIndex 🦙 0.9.39](https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine.html#define-router-query-engine): no description found
- [GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#openai-compatible): Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.
- [no title found](https://ai.meta.com/llama/): no description found
- [Issues · run-llama/llama_index](https://github.com/run-llama/llama_index/issues): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - Issues · run-llama/llama_index
- [What Is Llama 2 and How Can It Be Used? - deepsense.ai](https://deepsense.ai/llama-2#:~:text=The%20Llama%202%20license%20permits,must%20be%20sought%20from%20Meta.)): Discover the strides taken by Llama 2 in AI and the impact of Meta AI&#039;s language model on the tech world. Explore its features and wide-ranging uses!

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1201525794783105134) (31 messages🔥): 

- **MagViT2 Video Tokenizer Troubles**: `@_kevinalbert` is encountering issues with pixelated samples after 10,000 training steps with [MagViT2](https://arxiv.org/abs/2310.05737) using the [MagViT2 PyTorch repository](https://github.com/lucidrains/magvit2-pytorch). They have been advised by `@top_walk_town` to adjust the learning rate and various loss functions to improve outcomes.

- **Nightshade's Controversial Impact**: `@drhead` critiques the release of Nightshade as irresponsible and doubts its immediate threat without significant coordinated use and without countermeasures being developed. They suggest the potential for finetuning to neutralize or mitigate Nightshade's perturbations.

- **Skepticism around Glaze and Nightshade's Effectiveness**: `@astropulse` and `@.undeleted` express skepticism and concern over the intentions and effectiveness of the Glaze and Nightshade tools, with a particularly critical view on their potential long-term impact and the serious implications of their use.

- **Discussion on Counteracting Nightshade**: `@drhead` and `@.undeleted` debate potential strategies to make Nightshade ineffective, such as avoiding the targeted encoder or training new models that could nullify Nightshade's perturbations, yet remain cautious about the generalized threat posed by such attacks.

- **Views on the Necessity of New Data for AI Models**: `@pseudoterminalx` and `@mfcool` argue against the need for new images or data to improve AI models, highlighting the larger issue of data quality and proper captioning over quantity in enhancing model performance.
  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1201532176664100915) (7 messages): 

- **"Activation Beacon" Potentially Solves Context Length Issue**: `@spirit_from_germany` highlighted a new method called "Activation Beacon" that might revolutionize LLMs by allowing for unlimited context length. The method involves adding "global state" tokens before the prompt, with remarkable results showing that an LLaMA 2 model trained for 10K steps at 4K context length could then handle up to 400K context length. [Read the paper](https://arxiv.org/pdf/2401.03462.pdf) and [check the code](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/activation_beacon).

- **Discussion on the Optimal Way to Store Captioned Image Datasets**: `@progamergov` inquired whether it is better to store images in parquet files or separately in tar files for a captioned image dataset. `@chad_in_the_house` responded, recommending tar files for being faster to query through webdatasets.

- **Latent Diffusion in RePaint Explored?**: `@epicycles_989` questioned if the method used in RePaint has been applied to latent diffusion, expressing interest in its potential for zero-shot in/outpainting. No direct responses or additional information were provided regarding this query.

- **Advantage of **Webdatasets**: Following a discussion on the storage of captioned image datasets, `@random_string_of_character` provided links to **webdataset** and **tarp** on GitHub, highlighting that **webdatasets** offers a high-performance Python-based I/O system for deep learning problems and **tarp** enables fast and simple stream processing of files in tar files. [Webdataset GitHub](https://github.com/webdataset/webdataset) | [Tarp GitHub](https://github.com/webdataset/tarp)

- **Combining Parquet and TAR for Efficient Data Management**: `@progamergov` considered the idea that using parquet files for captions and tar files for corresponding images could be the most efficient storage solution. This hybrid method might offer the best of both worlds in terms of query speed and data organization.

**Links mentioned**:

- [Tweet from Yam Peleg (@Yampeleg)](https://fxtwitter.com/Yampeleg/status/1751942400287666536?t=9ajeGt5BJ7r7hTdLVvbppg&s=19): If this is true it is over: Unlimited context length is here.  Activation Beacon, New method for extending LLMs context.  TL;DR: Add &#34;global state&#34; tokens before the prompt  and predict auto-r...
- [GitHub - webdataset/webdataset: A high-performance Python-based I/O system for large (and small) deep learning problems, with strong support for PyTorch.](https://github.com/webdataset/webdataset/): A high-performance Python-based I/O system for large (and small) deep learning problems, with strong support for PyTorch. - GitHub - webdataset/webdataset: A high-performance Python-based I/O syste...
- [GitHub - webdataset/tarp: Fast and simple stream processing of files in tar files, useful for deep learning, big data, and many other applications.](https://github.com/webdataset/tarp): Fast and simple stream processing of files in tar files, useful for deep learning, big data, and many other applications. - GitHub - webdataset/tarp: Fast and simple stream processing of files in t...

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1201467152092364860) (32 messages🔥): 

- **Inquiring Minds Want to Know the Best AI Model Usage**: User `@vic7669` was curious about when to use specific AI models like Gemini, GPT-4, and Claude. The user was directed to a thread and a [technical FAQ](https://blog.perplexity.ai/technical-faq/what-is-the-difference-between-gpt-4-and-claude-2) for insights.

- **Library Visibility Issue in Sidebar**: User `@jamiecropley` experienced difficulties with the library not always showing on the left-hand side. Despite attempting a drastic measure of reinstalling Windows, the issue persisted, and the discussion clarified the behavior as a feature, not a bug, with only the last eight threads/collections being listed.

- **Cryptocurrency Inquiry and Clarification**: User `@lambda4life` inquired about a possible crypto token associated with the project. The question was promptly addressed by `@icelavaman`, confirming that no crypto token exists for the project.

- **Querying Perplexity via URL Parameters**: User `@fyngraf` asked if it was possible to initiate a query to Perplexity using URL parameters, aiming for an easier search utility. They were directed to a specific Discord channel for further information.

- **Rabbit R1 Gains Attention**: User `@brownpatricks` sparked a conversation about purchasing the Rabbit R1, inviting others to share their interest levels through emoji responses. Interest was expressed by several users, with some citing financial constraints while others admired the product's design.

**Links mentioned**:

- [Turkey Turkiye GIF - Turkey Turkiye Kartopu - Discover &amp; Share GIFs](https://tenor.com/view/turkey-turkiye-kartopu-kediler-100yil-gif-9621357169450001212): Click to view the GIF
- [Perplexity Blog](https://blog.perplexity.ai/technical-faq/what-is-the-difference-between-gpt-4-and-claude-2>): Explore Perplexity&#39;s blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1201541635700428800) (5 messages): 

- **Gemini Pro Tackles Chatbot Arena**: `@brknclock1215` shared a [YouTube video](https://youtu.be/EKodjqr5FCY?si=J4ZLEnGXhNn_ob7n&t=364) discussing how **New Bard** surpassed **GPT-4** in Chatbot Arena, showcasing a significant performance boost.
- **Personalizing Search with Great Features**: `@sa1k0s` expressed appreciation for a tip that allows personalization of their search engine, praising the feature's ability to customize.
- **Creative Use of AI for a Pomodoro App**: `@gammagames` is developing a delivery-themed **pomodoro app**, successfully using an AI tool to generate names and addresses for the app content, finding the tool greatly beneficial.
- **Explore Perplexity's Lab**: `@rowalth` highlighted the existence of a [lab on Perplexity](https://labs.perplexity.ai/) for experimentation, pointing out a resource for creative exploration.

**Links mentioned**:

[🔥 New Gemini Pro Better than GP-4? Huge Performance Boost on ⚔️ Chatbot Arena ⚔️](https://youtu.be/EKodjqr5FCY?si=J4ZLEnGXhNn_ob7n&t=364): New Bard has surpassed GPT-4 on Chatbot Arena. 🦾 Discord: https://discord.com/invite/t4eYQRUcXB☕ Buy me a Coffee: https://ko-fi.com/promptengineering|🔴 Pat...

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1201757369944772638) (1 messages): 

- **Custom Stop Words Feature Inquiry for pplx-api**: `@dogemeat_` asked about the implementation timeline for **custom Stop Words** in the **pplx-api**, expressing an interest in integrating it with the [zed.dev editor](https://zed.dev/) for its 'assistant' features. This integration aims to offer an alternative to the default **OpenAI models**.
  

---



### LLM Perf Enthusiasts AI ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1201534611306582096) (1 messages): 

- **The Sweet Spot Between PyTorch and CUDA**: `@tvi_` expressed that while PyTorch allows for an "almost declarative" approach to CUDA computation by abstracting away the how of operations like matrix multiplication (`a @ b`), **CUDA** and **Numba.CUDA** require explicit detailing of computation processes. **Triton** is highlighted as a middle ground, providing more control than PyTorch but with less complexity than direct CUDA, appealing for those who seek a balance.
  

---


### LLM Perf Enthusiasts AI ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1201505232346169364) (29 messages🔥): 

- **CUDA Optimization Odyssey**: `@artste` and `@zippika` shared their journey trying to optimize **RGB to Grayscale conversion** on CUDA. Various attempts included using **structs for cleaner indexing,** employing **vectorized operations** with `uchar4`, and exploring different memory layouts such as **CHW and HWC**.
- **Vectorization Proves Challenging**: Despite initial excitement, `@zippika` found that **vectorizing the conversion process** with `uchar4` and shared memory led to slower performance than expected. Further optimizations and tweaks also failed to yield the anticipated speedup.
- **The Power of Inline**: A moment of revelation came when `@zippika` noted that using `__forceinline__` in their CUDA kernel **significantly boosted performance**, highlighting the importance of inlining for GPU code optimization.
- **A Misleading Benchmark**: `@zippika` experienced a rollercoaster of emotions upon realizing an error in their test setup—a **naive implementation** appeared faster due to it only processing a **single 3-pixel height image,** which skewed the comparison.
- **Seeking Community Wisdom**: `@andreaskoepf` suggested further **memory optimization techniques** such as using consecutive memory reads and leveraging `int4` data types for vector loads and stores, directing `@zippika` to a [NVIDIA developer blog](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) for deeper insights into maximizing bandwidth efficiency.

**Links mentioned**:

[CUDA Pro Tip: Increase Performance with Vectorized Memory Access | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/): This post demonstrates the use of vectorized memory access in CUDA C/C++ to increase bandwidth utilization while decreasing instruction count.

  

---


### LLM Perf Enthusiasts AI ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1201532749673152622) (3 messages): 

- **Chapter 2 Doubt Cleared Up**: `@ashpun` expressed confusion over the 3rd question in chapter 2 and sought clarification, mentioning difficulty in understanding the explanation provided in the 2nd video. They were open to any explanations on how to approach solving this question.
- **Figuring Out Coalescing and Banking in Chapter 6**: `@shindeirou` discussed confusion around figures 6.10 and 6.11 in chapter 6, questioning the efficiency of accessing memory banks within the same channels. They later understood that going to the same channel but different banks allows for effective interleaving to hide memory latency.
- **Solving Indexing in Memory Access**: `@andreaskoepf` provided a solution for understanding memory indexing as discussed in chapter 2, offering a specific code snippet and explanation for calculating the index to elements within sections of memory. This addressed `@ashpun`'s query on the 3rd question in chapter 2.
  

---


### LLM Perf Enthusiasts AI ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/) (1 messages): 

andreaskoepf: New video link: https://youtu.be/4sgKnKbR-WE?si=J-B0kHqknRXhE7e_
  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1201587528789135470) (9 messages🔥): 

- **Seeking Superior Writing Assistant Tools**: `@an1lam` expressed dissatisfaction with current writing assistant tools, highlighting the chat interface's limitations for iterative writing and the absence of quality alternatives like lex.page.
- **An Enthusiastic Recommendation but with a Catch**: In response to `@an1lam`'s query, `@jeffreyw128` tagged a user, possibly suggesting a recommendation or a lead on writing assistant tools, albeit without providing concrete details.
- **Startup Struggles with Google Workspace Costs**: `@frandecam` shared concerns about the high costs of provisioning Google Workspace accounts for their bootstrapped AI startup, highlighting the financial burden it represents for startups.
- **A Ray of Hope for AI Startups**: Responding to `@frandecam`'s dilemma, `@dare.ai` suggested the Google Cloud for Startups program as a solution to mitigate costs, though cautioned about potential delays due to backlogs. They shared a link to the program ([Google for Startups Cloud Program](https://cloud.google.com/startup/ai?hl=en)) and offered further assistance.
- **Introducing Jenni.ai for Academic Writing**: `@calclavia` introduced jenni.ai, a tool under development aimed at enhancing academic writing, implying it as a potential solution for `@an1lam`'s quest for effective writing assistance tools.

**Links mentioned**:

[AI startup program | Google Cloud](https://cloud.google.com/startup/ai?hl=en): Tap into the best of Google’s infrastructure, AI products, and foundation models. Get up to $250,000 USD in Google Cloud credits, training, and more.

  

---


### LLM Perf Enthusiasts AI ▷ #[gpt3-5](https://discord.com/channels/1168579740391710851/1168582170378518558/1201449353483649067) (1 messages): 

- **Gorilla Launches OpenFunctions**: `@shacrw` shared a [blog post about Gorilla OpenFunctions](https://gorilla.cs.berkeley.edu/blogs/4_open_functions.html), a new open-source alternative for formulating executable API calls using natural language prompts. This solution simplifies API calls for various services, making it accessible even to those with minimal programming knowledge.
- **Explore Gorilla Functions on GitHub**: Accompanying the blog share, `@shacrw` also provided a [GitHub link](https://github.com/philschmid/open-source-function-calling/blob/main/gorilla-functions.ipynb) to explore the OpenFunctions project further. This initiative is driven by the need to enhance Large Language Model (LLM) chat completion features with the capability to accurately format function calls based on API documentation and question-answer pairs.

**Links mentioned**:

- [Introduction to Gorilla LLM](https://gorilla.cs.berkeley.edu/blogs/4_open_functions.html): no description found
- [open-source-function-calling/gorilla-functions.ipynb at main · philschmid/open-source-function-calling](https://github.com/philschmid/open-source-function-calling/blob/main/gorilla-functions.ipynb): Contribute to philschmid/open-source-function-calling development by creating an account on GitHub.

  

---


### LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1201567847202168952) (1 messages): 

- **Prompt Investing Vision Model Hits a Copyright Roadblock**: `@jxnlco` is seeking advice on how to effectively use a prompt investing vision model for reading complex labels, but is facing challenges as the model is flagging the material as copyrighted.
  

---


### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (1 messages): 

thebaghdaddy: do we believe the mistral medium hype about becoming 2nd only to GPT4?
  

---



### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1201628159095873646) (7 messages): 

- **Frankenstein Model Experiments Underway**: `@nisten` announced the creation of a composite model by merging all the **70b CodeLlamas**. This experimental model aims to explore enhanced performance benchmarks.
- **Laughter and Progress with BigCodeLlama**: Amidst the serious tech discussion, `@nisten` also shared a lighter moment with a "lolol" message, indicating both the fun and progress in their work.
- **BigCodeLlama 169B Voilà!**: `@nisten` has uploaded the **BigCodeLlama-169b** model on [Hugging Face](https://huggingface.co/nisten/BigCodeLlama-169b), a powerful amalgamation designed to benchmark how different combinations of models perform together.
- **Benchmarking the FrankenLlama**: Following the model upload, `@nisten` shared a challenging **coding problem** involving calculating Aldrin cycler orbits for Mars colonization as a test case to compare the FrankenLlama model against the stock models. 
- **Friendly Check-In amid the Tech Talk**: `@zentorjr` dropped in with a casual greeting for `@nisten`, adding a personal touch to the high-level technical discussion unfolding in the channel.

**Links mentioned**:

[nisten/BigCodeLlama-169b · Hugging Face](https://huggingface.co/nisten/BigCodeLlama-169b): no description found

  

---


### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1201526241426145340) (2 messages): 

- **Diving Into ColBERT with RAGatouille**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=cABkk8WmOGY) titled "Exploring ColBERT with RAGatouille," highlighting **RAGatouille**, a library facilitating work with the **ColBERT** retrieval model. ColBERT is described as a fast and accurate model that improves scalable behavior summarization. 

- **Unveiling the Eagles’ Flight with Eagle 7B**: `@pradeep1148` introduced another [YouTube video](https://www.youtube.com/watch?v=j78gZlHPAoY) titled "Running 🦅 Eagle 7B on A40," discussing **Eagle 7B**, a significant advancement in the RWKV-v5 architecture and linear transformers across 100+ languages. Eagle 7B is notable for processing 1 trillion tokens, marking a new era for transformer tech.

**Links mentioned**:

- [Exploring ColBERT with RAGatouille](https://www.youtube.com/watch?v=cABkk8WmOGY): RAGatouille is a relatively new library that aims to make it easier to work with ColBERT.ColBERT is a fast and accurate retrieval model, enabling scalable BE...
- [Running 🦅 Eagle 7B on A40](https://www.youtube.com/watch?v=j78gZlHPAoY): 🦅 Eagle 7B : Soaring past Transformers with 1 Trillion Tokens Across 100+ LanguagesA brand new era for the RWKV-v5 architecture and linear transformer&#39;s has...

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1201583759066140682) (3 messages): 

- **Emoji Suggestion Tool Unveiled**: `@dbreunig` showcased a new demo at [Emoji Suggest](https://dbreunig.github.io/emoji-suggest/) that can convert short sentences or headlines into a single, recommended emoji using the [CLIP model](https://github.com/openai/CLIP). The [code for the tool](https://github.com/dbreunig/emoji-suggest) is available, utilizing precomputed embeddings for emojis to facilitate quick and sophisticated searches.

- **Insights on Embeddings for Search Applications**: `@dbreunig` shared insights on using embeddings to create sophisticated search tools quickly, emphasizing the importance of curating the options for truly effective search capabilities. This approach underpins the functionality of their emoji suggestion tool.

- **AI's Positive Perspective on its Role**: `@bdexter` relayed a conversation where an AI, specifically llama2, expressed the belief that **artificial intelligence is a force for good**, capable of helping humans reach their full potential. This interaction highlights the positive outlook that AI can hold regarding its impact on human achievement.

**Links mentioned**:

[Emojify](https://dbreunig.github.io/emoji-suggest/): no description found

  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1201564085649674460) (1 messages): 

- **ColBERT Writeup Appreciation**: User `@bewilderbeest` expressed gratitude towards `@746595581086138409` for the **ColBERT writeup** on TIL. They are creating a notebook using the shared code snippets and praised the heatmap visualization of words in the results.
  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1201575445662015569) (2 messages): 

- **Is GPT-5 Training Underway?**: User `@entropi` shared a [YouTube video](https://www.youtube.com/watch?v=Zc03IYnnuIA) titled **"GPT-5: Everything You Need to Know So Far"**, sparking curiosity about whether GPT-5 has commenced training, citing exclusive interviews and information from OpenAI.
- **Confirmation on GPT-5 Training**: In response to the speculation, `@lightningralf` confirmed that GPT-5 has indeed been in training for some time, though no specific details were provided.

**Links mentioned**:

[GPT-5: Everything You Need to Know So Far](https://www.youtube.com/watch?v=Zc03IYnnuIA): Was yesterday the day GPT-5 actually started training? This video has everything we think we know so far about GPT-5, drawing on exclusive interviews, OpenAI...

  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1201669557652033657) (1 messages): 

- **ChatGPT vs. Bard on Captcha Images Moderation**: `@juanreds` pointed out an interesting difference between **ChatGPT** and **Bard**: ChatGPT moderates Captcha images while Bard does not. This highlights divergent approaches to user interactions and security measures between the two AI systems.
  

---


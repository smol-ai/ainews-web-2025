---
id: 769f97bd-e196-43ce-a5a9-8215666675eb
title: 'MM1: Apple''s first Large Multimodal Model'
date: '2024-03-15T23:34:51.378733Z'
original_slug: ainews-mm1-apples-first-large-multimodal-model
description: >-
  **Apple** announced the **MM1** multimodal LLM family with up to **30B
  parameters**, claiming performance comparable to **Gemini-1**. The model
  excels on VQA benchmarks despite their flaws and shows potential for embodied
  agents and business/education applications. **Yann LeCun** emphasized the need
  for AI systems to understand the physical world and plan hierarchically to
  reach human-level intelligence. **Cohere** released **Command-R** for
  Retrieval Augmented Generation, and **Anthropic** expanded the **Claude 3**
  family with Opus, Sonnet, and Haiku models targeting various use cases.
  Open-source hardware **DexCap** enables dexterous robot manipulation training.
  Tools like **CopilotKit** simplify AI integration into React apps, and
  LangChain added an open-source financial agent. Discussions highlighted the
  ongoing challenges in NLP and AI progress, with insights from **Yann LeCun**
  and **François Chollet**.
companies:
  - apple
  - cohere
  - anthropic
  - huggingface
  - langchain
models:
  - mm1
  - gemini-1
  - command-r
  - claude-3-opus
  - claude-3-sonnet
  - claude-3-haiku
topics:
  - multimodality
  - vqa
  - fine-tuning
  - retrieval-augmented-generation
  - open-source
  - robotics
  - model-training
  - react
  - rag
  - financial-agents
  - model-evaluation
  - planning
  - reasoning
people:
  - ylecun
  - fchollet
  - arankomatsuzaki
  - dl_weekly
  - deep-learning-ai
  - drjimfan
  - alexalbert__
  - svpino
  - jeremyphoward
  - virattt
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 3/14/2024-3/15/2024. We checked [**358** Twitters](https://twitter.com/i/lists/1585430245762441216) and **20** Discords (**332** channels, and **2839** messages) for you. Estimated reading time saved (at 200wpm): **353 minutes**.

Apple continues to make moves in AI, announcing (but not releasing) [MM1 with a paper](https://arxiv.org/abs/2403.09611), claiming it is Gemini-1 level:

 ![image.png](https://assets.buttondown.email/images/2ee155d5-fbac-489f-a611-e7bada80f8aa.png?w=960&fit=max) 

The 30B model beats larger older models at the ([flawed](https://www.latent.space/p/idefics)) VQA benchmarks:

 ![image.png](https://assets.buttondown.email/images/56c0fd3f-d840-4b49-a4ee-4da1ae495e64.png?w=960&fit=max) 

The paper is oriented at researchers, providing some useful ablations for hyperparams and architecture.

The appendices hints at usecases for embodied agents:

 ![image.png](https://assets.buttondown.email/images/89f01eb6-7458-4239-b7a3-90bdf7ecd54d.png?w=960&fit=max) 

and business/education:

 ![image.png](https://assets.buttondown.email/images/288c20a5-2748-46a2-88a2-37f2a282ca28.png?w=960&fit=max) 

For a selection of [competing open VLMs](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard), there is a new HF leaderboard you can reference.

---

**Table of Contents**

[TOC]

---

# PART X: AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs

## AI Progress and Limitations

- **Yann LeCun said that to have human-level AI, systems need to understand the physical world, remember and retrieve appropriately, reason, and set sub-goals and plan hierarchically.** Even with such capabilities, it will take a while to reach human or superhuman level. [@ylecun](https://twitter.com/ylecun/status/1768327681525887174)
- An LLM is like an encyclopedia that can talk back. [@ylecun](https://twitter.com/ylecun/status/1768326303223062729)
- **Many people believe LLMs mean NLP is "solved" and machines have human-level language understanding, but we're not close.** Being convinced the problem is solved guarantees no further progress will be made. [@fchollet](https://twitter.com/fchollet/status/1768337855032786967)
- In 1970, it was said in 3-8 years we'd have a machine with human-level general intelligence. The full article this quote came from is a great read. [@fchollet](https://twitter.com/fchollet/status/1768312558430368241)

## New Models and Datasets

- **Apple presented MM1, a family of multimodal LLMs up to 30B parameters that are SoTA in pre-training metrics and perform competitively after fine-tuning.** [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1768446729710371115)
- Cohere announced the release of Command-R, a language model designed for Retrieval Augmented Generation at scale. [@dl_weekly](https://twitter.com/dl_weekly/status/1768310133346492479)
- Anthropic's Claude 3 family of models (Opus, Sonnet, Haiku) are designed for applications ranging from extensive capability to cost-effectiveness and speed. [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1768472181556592947)

## Open Source and Reproducibility 
- **DexCap is a $3,600 open-source hardware stack that records human finger motions to train dexterous robot manipulation.** It's an affordable "lo-fi" version of Optimus for academic researchers. Data collection is decoupled from robot execution. [@DrJimFan](https://twitter.com/DrJimFan/status/1768323865317671413)
- **Opus's prompt writing skills + Haiku's speed and low cost enable lots of opportunities for sub-agents.** A cookbook recipe demonstrates how to get these sub-agents up and running in applications. [@alexalbert__](https://twitter.com/alexalbert__/status/1768341620322402751)
- **It's simple to integrate AI into React apps with CopilotKit, which takes application context and feeds it into React infrastructure to build chatbots, AI-powered textareas, RAG, function calling, and integrations.** The sample app is open-source and can be self-hosted with any LLM. [@svpino](https://twitter.com/svpino/status/1768252265373081912)

## Tools and Frameworks
- **Migrating code to Keras 3 with JAX backend provides benefits of not needing TensorFlow and 50% faster model training.** [@svpino](https://twitter.com/svpino/status/1768307137132765304)
- **Reranking is critical for effective retrieval in RAG. A new project from @bclavie greatly simplifies this important technique.** [@jeremyphoward](https://twitter.com/jeremyphoward/status/1768344061805760943)
- **An open source financial agent was added to LangChain, with tools to get latest price, news, financials and historical prices for a ticker.** Upcoming tools include intrinsic value calculator and price chart renderer. Code is open source and runnable in Colab. [@virattt](https://twitter.com/virattt/status/1768395191629627478)

## Memes and Humor

- "The difference between you and a world leader: you called your teacher mommy in elementary school and did nothing but get embarrassed. Macron called his high school teacher mommy, dated her until she left her husband, married her, and is now threatening Russia with nuclear war" [@Nexuist](https://twitter.com/Nexuist/status/1768435245873860689)
- "It's over, fix your overly verbose model OAI, I'm not gonna sit here begging it for code" [@abacaj](https://twitter.com/abacaj/status/1768294165761171647)
- ".@elonmusk i will pay 20$/mo. please fix the "pussy in bio" problem." [@AravSrinivas](https://twitter.com/AravSrinivas/status/1768310011263098964)


---

# PART 0: Summary of Summaries of Summaries

> Since [Claude 3 Haiku was released recently](https://x.com/anthropicai/status/1768018310615151002?s=46&t=90xQ8sGy63D2OtiaoGJuww), we're adding them to this summary run for you to compare. We'll keep running these side by side for a little longer while we build the AINews platform for a better UX.

## Claude 3 Haiku (3B?)

> Commentary: We experimented tweaking the Haiku prompt since it was not doing well. It seems Flow Engineering > Prompt Engineering for Haiku. However the topic clustering doesn't look great yet.

**Positional Encoding and Language Model Capabilities**:

- **Positional Encoding: A Delicate Dance**: Discussions note the challenges of causal language models without **Positional Encoding (PE)**, including the production of gibberish outputs and inference failures. A paper ([Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/pdf/2203.16634.pdf)) suggests models might encode "absolute positions" implicitly, leading to out-of-distribution errors during longer inferences.
- **Exploring SERAPHIM and Claude 3's "World Simulation"**: SERAPHIM, a clandestine AI research group envisioned by **Claude 3**, has been the topic of interest. Dialogue about *Claude 3's* advanced *world modeling* as a *simulator entity* named **The Assistant**, has led to discussions about metaphysical and epistemological explorations within the AI.

**Function Calling and JSON Handling**:

- **Function Calling Eval Codes and Datasets released**: Nous Research has published function calling eval code and datasets. The code is available on [GitHub](https://github.com/interstellarninja/function-calling-eval), with datasets accessible on [Hugging Face](https://huggingface.co/datasets/NousResearch/json-mode-eval) and [Hugging Face](https://huggingface.co/datasets/NousResearch/func-calling-eval).
- **Hermes Pro Function Calling Addresses JSON Quirks**: While using **Hermes 2 Pro** for function calling, issues with JSON and single vs. double quotes in the system prompt have been discussed. It's confirmed that changing the system prompt to explicitly use double quotes can be effective without significantly impacting performance.

**Fine-Tuning and Model Performance**:

- **Fine-Tuning Raises the Bar**: The [d-Qwen1.5-0.5B student model](https://huggingface.co/aloobun/d-Qwen1.5-0.5B), after fine-tuning, has surpassed the performance of its base model on **truthfulqa** (39.29 vs 38.3) and **gsm8k** (17.06 vs 16.3) benchmarks.
- **Exploring Genstruct 7B's Capabilities**: Users engaged with the [Genstruct 7B model](https://huggingface.co/NousResearch/Genstruct-7B) for generating instruction datasets. One user planned to test with text chunks and shared a [*repository with examples*](https://github.com/edmundman/OllamaGenstruct/tree/main) of how to use it.

**Hardware and System Optimizations**:

- **NVIDIA Rumors**: NVIDIA's rumored **RTX 50-series "Blackwell"** GPUs with GDDR7 memory at 28 Gbps speeds were mentioned in a [TechPowerUp article](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed).
- **Photonic Processing's Enlightenment**: A breakthrough in **photonic computing** highlighted by [Lightmatter](https://lightmatter.co/) proposes to utilize photonics to dramatically boost chip communication and computation, potentially revolutionizing AI efficiency.

**Community Knowledge Sharing and Open-Source Practices**:

- **Open Source Code Interpreter Pursuits**: A discussion arose about the lack of open-source GPT code interpreters for tasks like CSV handling. One user pointed out the [*open-interpreter on GitHub*](https://github.com/KillianLucas/open-interpreter) but noted it's more tuned to sending instructions rather than interpreting code.
- **Advocating for Open-Source AI**: A member expressed the belief that being fully open source in **models, datasets, and methodology** will lead to better long-term improvements in AI models.

## Claude 3 Sonnet (14B?)

> Commentary: Sonnet [kinda broke today](https://twitter.com/swyx/status/1768775665330090135) and didn't follow our instructions as well as every single day prior. We manually prompted it back toward somehow behaving but something feels off.


- **Large Language Model Advancements**: Discussions around the capabilities and limitations of large language models like GPT-4, Claude, and LLaMa. This included fine-tuning techniques, evaluating reasoning abilities, and exploring interpretability methods like [latent decoding by vector-db-lookup](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py).

- **Hardware Optimizations for AI**: Optimizing hardware setups, from Apple Silicon with `sudo sysctl` to leveraging GPUs like RTX 5090 and NVIDIA Grace Hopper. Quantization levels for stable performance on models like Mixtral were also covered, with [Q3 or 3-bit quantization](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/root.html) recommended.

- **AI Model Interpretability and Evaluation**: Techniques to interpret and evaluate large language models were explored, such as [using n-gram statistics to sample text](https://en.wikipedia.org/wiki/Word_n-gram_language_model) and the limitations of AI content detectors in distinguishing human-generated media, as discussed in [this paper](https://arxiv.org/abs/2403.09539).

- **Prompt Engineering and Data Handling**: Strategies for effective prompt engineering, like [prompt augmentation with smaller models](https://brianfitzgerald.xyz/prompt-augmentation/) and tools for managing prompts. Data preparation challenges, like handling complex file formats in Retrieval-Augmented Generation (RAG), were also addressed.

- **Open-Source AI Frameworks**: Developments in open-source AI frameworks like [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) (upgrading to PyTorch 2.2.1), [LlamaIndex v0.10.20](https://t.co/GY4unUYOwl) with an Instrumentation module, and the [LangChain v0.2 release](https://github.com/langchain-ai/langchain/discussions/19083) to remove dependencies.

- **AI Security and Privacy**: Potential security concerns like [requests made on behalf of another OpenAI account](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca), leaking proprietary LLM information through APIs, and clarifications on OpenAI's [enterprise privacy policy](https://openai.com/enterprise-privacy).

- **Emerging AI Platforms and Tools**: New AI platforms like Deci AI's [Deci Nano LLM](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u) and Cohere's [Command-R integration on OpenRouter](https://openrouter.ai/models/cohere/command-r). Open-source tools like [Refacto VS Code plugin](https://github.com/rbourgeat/refacto) and JavaScript LangChain with Dall-E were also highlighted.

## Claude 3 Opus (>220B?)

> Commentary: this one comes closest to what was originally prompted (we asked for top 4-5 themes across everything)... but we actually prefer the output of the other 2 despite the length. In this case adhering too closely to our prompt was not good.

- **Function Calling and JSON Mode Advancements**: Nous Research released function calling eval code and datasets on [GitHub](https://github.com/interstellarninja/function-calling-eval) and [Hugging Face](https://huggingface.co/datasets/NousResearch/json-mode-eval), sparking discussions on using JSON mode effectively in complex conversations. The **Hermes 2 Pro 7B** model's function calling capabilities were showcased in a [YouTube video](https://www.youtube.com/watch?v=PzaidfqDtGI) and [GitHub repository](https://github.com/NousResearch/Hermes-Function-Calling/tree/main).

- **Model Breakthroughs and Fine-Tuning Feats**: The **d-Qwen1.5-0.5B student model** [surpassed its base model](https://huggingface.co/aloobun/d-Qwen1.5-0.5B) on benchmarks after fine-tuning. Engineers tested the **Genstruct 7B model** for [generating instruction datasets](https://github.com/edmundman/OllamaGenstruct/tree/main). A new training method claims to improve accuracy and sample efficiency, with initial tests on **VGG16 and CIFAR100** showing promise, as discussed in the Skunkworks AI Discord.

- **Debugging and Optimization Techniques**: CUDA developers troubleshooted errors like **CUBLAS_STATUS_NOT_INITIALIZED**, with suggestions pointing to tensor dimensions and memory issues, as seen in related [forum posts](https://discuss.pytorch.org/t/runtimeerror-cuda-error-cublas-status-not-initialized-when-calling-cublascreate-handle/170409). Triton debugging was enhanced with the `TRITON_INTERPRET=1` environment variable and a [visualizer in development](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing). **Lecture 8 on CUDA Performance** was re-recorded and released with updated [video](https://www.youtube.com/watch?v=SGhfUhlowB4), [code](https://github.com/cuda-mode/lectures/tree/main/lecture8), and [slides](https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit).

- **Advancements in AI Architectures and Frameworks**: Maisa introduced the **Knowledge Processing Unit (KPU)**, an AI architecture that claims to outperform **GPT-4 and Claude 3 Opus** in reasoning tasks, as detailed in their [blog post](https://maisa.ai/blog/kpu). The **Axolotl framework** explored optimizations like **ScatterMoE** in their [branch](https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe). LangChain expedited the release of **version 0.2** to address CVEs and break the `langchain-community` dependency, as discussed in a [GitHub issue](https://github.com/langchain-ai/langchain/discussions/19083).

## ChatGPT (GPT4T)

> Commentary: good list of prompt eng tools in there. Our GPT prompt has fallen behind our Claude prompt in terms of readable quality so we will focus on improving this next.

<div><ul><li><p><strong>Positional Encoding in Language Models</strong>: Discussions highlighted the importance of <strong>Positional Encoding (PE)</strong> in preventing causal language models from producing gibberish outputs. A paper suggested that models could implicitly learn absolute positions, leading to errors during longer inferences (<a target="_new" href="https://arxiv.org/pdf/2203.16634.pdf">source</a>).</p></li><li><p><strong>Function Calling in AI Models</strong>: <strong>Nous Research</strong> released function calling evaluation code and datasets, highlighting the challenges of using JSON mode in complex interactions (<a target="_new" href="https://github.com/interstellarninja/function-calling-eval">GitHub</a>, <a target="_new" href="https://huggingface.co/datasets/NousResearch/json-mode-eval">Hugging Face</a>).</p></li><li><p><strong>AI Model Fine-Tuning</strong>: The <strong>d-Qwen1.5-0.5B student model</strong> surpassed its base model's benchmarks, showcasing new developments in model fine-tuning. The <strong>Genstruct 7B model</strong> was tested for generating instruction datasets, with a focus on calculating perplexity in LLaMA models (<a target="_new" href="https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook">source</a>).</p></li><li><p><strong>Open-Source Practices in AI</strong>: Conversations around AI models touched on topics like world modeling and the potential for open-source GPT code interpreters, advocating for transparency in AI development (<a target="_new" href="https://github.com/KillianLucas/open-interpreter">GitHub</a>).</p></li><li><p><strong>Tech Discussions on Hardware and AI Access</strong>: Debates covered <strong>Claude.ai</strong> access in the EU and NVIDIA's <strong>RTX 50-series "Blackwell"</strong> GPUs' performance, alongside discussions on <strong>GDDR7 memory speeds</strong> (<a target="_new" href="https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed">TechPowerUp article</a>).</p></li><li><p><strong>Challenges with AI Content Detection</strong>: The limitations of AI content detectors were examined, suggesting reliance on verifiable creation processes as substantial proof of human authorship and discussing the efficacy and implications of cryptographic watermarking.</p></li><li><p><strong>CUDA Programming Insights</strong>: A focus on <strong>NumPy</strong> performance overhead in comparison to BLAS and the introduction of the <strong>SimSIMD library</strong> as a solution to reduce losses in high-performance scenarios was discussed, highlighting the importance of SIMD optimizations.</p></li><li><p><strong>AI Model Interoperability and Improvements</strong>: The introduction of <strong>KPU</strong> by Maisa, claiming superiority over GPT-4 and Claude 3 Opus in reasoning, sparked debates on benchmarks and the absence of latency information, questioning its efficiency beyond prompt engineering.</p></li><li><p><strong>Prompt Engineering Tools and Techniques</strong>: Engineers explored tools for prompt engineering, likening the search to finding a "Postman for prompts" and discussing the use of SQLite, Prodigy, PromptTools, and Helicone AI for managing and experimenting with prompts (<a target="_new" href="https://sqlite.org/index.html">SQLite</a>, <a target="_new" href="https://prodi.gy/features/prompt-engineering">Prodigy</a>, <a target="_new" href="https://github.com/hegelai/prompttools">PromptTools</a>, <a target="_new" href="https://www.helicone.ai/">Helicone AI</a>).</p></li><li><p><strong>Language Model Sophistication Techniques</strong>: Engineers theorized over advanced model techniques, including '<em>mega distillation sauce</em>' and token-critical mixtures, highlighting the impact of early tokens on performance in tasks like solving math problems and discussing the evolution of AI safety classifications and methodologies for enhancing content moderation.</p></li></ul></div>

---

# PART 1: High level Discord summaries

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Positional Encoding: A Delicate Dance**: Discussions note the challenges of causal language models without **Positional Encoding (PE)**, including the production of gibberish outputs and inference failures. A paper ([Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/pdf/2203.16634.pdf)) suggests models might encode "absolute positions" implicitly, leading to out-of-distribution errors during longer inferences.
  
- **Function Calling Finesse**: Various platforms reveal **Nous Research**'s release of function calling eval code and datasets, available on [GitHub](https://github.com/interstellarninja/function-calling-eval) and [Hugging Face](https://huggingface.co/datasets/NousResearch/json-mode-eval), with insights into the challenges of using JSON mode effectively in complex conversations, possibly requiring content summarization or trimming.
  
- **AI's Higher Learning Curve**: New developments in model fine-tuning are showcased with the [d-Qwen1.5-0.5B student model](https://huggingface.co/aloobun/d-Qwen1.5-0.5B) surpassing its base model's benchmarks, and the **Genstruct 7B model** ([source](https://github.com/edmundman/OllamaGenstruct/tree/main)) is tested for generating instruction datasets. An inquiry about perplexity calculation issues in LLaMA models leads to a reference to a [Kaggle notebook](https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook) for further exploration.
  
- **Building Community Knowledge Bases**: Engagements around AI models touch on topics like the world modeling of **Claude 3 as The Assistant** and the possibility of open-source GPT code interpreters, such as the *open-interpreter* on [GitHub](https://github.com/KillianLucas/open-interpreter). Open-source practices in AI development are advocated for, highlighting the need for transparency in models, datasets, and methodologies.
  
- **Tech Enthusiasts Talk Shop**: Users in several channels debate over **Claude.ai** access in the EU without a VPN and the performance of NVIDIA's rumored **RTX 50-series "Blackwell"** GPUs. They also showcase the functionality of **Hermes 2 Pro 7B** in a shared YouTube video titled ["Lets Function Call with Hermes 2 Pro 7B"](https://www.youtube.com/watch?v=PzaidfqDtGI), and consider the implications of GDDR7 memory speeds reported in a [TechPowerUp article](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed).
  

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord Summary

- **Torch Update Torches Colab Routines**: A Colab update to **Torch 2.2.1** disrupted workflows with broken dependencies; however, a series of pip install commands involving [Unsloth's library](https://github.com/unslothai/unsloth.git) offer a quantized and VRAM efficient fix. The performance of models like **Mistral** and **Gemma** during fine-tuning was a topic of interest, with observations on bug fixes and performance improvements in Unsloth AI.
  
- **Colab or Kaggle? That is the Question**: Users discussed the merits and demerits of using Google Colab versus Kaggle for model training, with some favoring Kaggle for its stability. Meanwhile, the importance of using `xformers` with the right CUDA versions for Unsloth was emphasized, and tips for finetuning models like TinyLlama were shared using updated Kaggle notebooks.
  
- **Training Woes and Wins**: There was significant dialogue around best practices for fine-tuning language models, such as DPO training and managing learning rate adjustments. Insights included ensuring `max_grad_norm = 0.3` and adjusting batch sizes, while a member indicated potential progress with a loss below 1.2.
  
- **Fine-Tuning Foibles and Fixes**: Discussions around model conversion for increased precision, issues with training order potentially affecting performance, and fine-tuning for roleplay environments surfaced. The `bitsandbytes` library was mentioned for precision conversion, and advice was given for disabling shuffling in training dataloaders.
  
- **Sophia Signals Potential**: A member proposed looking into **Sophia** as a possible plug and play solution, though further testing was necessary. Another discussion centered on fine-tuning strategies, considering whether **3 epochs** might be a standard approach for larger datasets.
  

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

**Model Conundrums and Quantization Queries**: Users delved into **LM Studio** intricacies, such as seeking advice to improve **API inferencing** and addressing difficulties using multiple GPUs. Misunderstandings about model support and extensions, like the **.gguf** file, were clarified, with a focus on model types like **Command-R 35B** and **Mistral Non-Instruct**. Upcoming features like **RAG** integration in LM Studio v0.2.17 and **IQ1 model compression** tests also sparked interest, revealing that quality levels **Q3 or 3-bit** are needed for stable **Mixtral and MOE model** performance.

**Interdisciplinary Hardware Harmony**: Hardware discussions spanned from optimizing Apple Silicon for LLMs to considering the efficacy of NVLINK for enhancing **Goliath 120B model** performance. Enthusiasts shared experiences on system memory, with debates on the ideal RAM configuration and the anticipation for Nvidia's new **RTX 5090 GPU**. Concurrently, ROCm beta limitations were highlighted with reports of issues with GPU offloading, particularly on **AMD 7480HS** and integrated GPUs. A **Reddit post** and a **GitHub repository** provided additional insights into tweaking VRAM and resolving AMD GPU offloading dilemmas.

*Relevant links for additional context*:

- [Understanding LM Studio API inferencing issues](https://huggingface.co)
- [Hugging Face repository misleads on llama.cpp support](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF)
- [Discussing ROCm support and dGPU prioritization](https://github.com/brknsoul/ROCmLibs)

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

**Haiku for the Technical Mind**: [**Claude 3 Haiku**](https://labs.pplx.ai) has been unleashed at Perplexity Labs, offering a new poetic twist to AI.

**Techies Prefer Claude 3**: Users are gravitating towards **Claude 3** for an array of tasks, including writing and content creation, citing its strengths over other GPT models.

**Perplexing API Quirks and Queries**: The **Perplexity API** is stirring both intrigue and confusion among users with issues around real-time data querying and inconsistent responses when compared to the chat interface.

**Firefox Extension Uses Perplexity API**: A user is experimenting with a Firefox extension that taps into the Perplexity API, still at a proof of concept stage.

**Mind the API Deprecations**: Members are puzzled by the operational status of the `pplx-70b-online` model, noting planned deprecation but observing ongoing responses as of March 15.

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

**Game AI Gets Green Thumbs**: Discussions envisioned an AI mastering *Animal Crossing*, epitomizing the capability of game-playing AIs and highlighting benchmarks for their success. The analyses reflected on AI strategies and fairness, with constraints suggested like action limits or induced latency to level the playing field against human gamers.

**Interpreting the Unseen in AI**: Engineers examined *latent decoding by vector-db-lookup* to demystify AI's intermediate representations, employing multilingual embeddings from Llama2 to decode at various layers. They engaged in bilingual tokenizer experiments, pondering the weight of training data on AI biases and exploring text generation from n-gram statistics, citing an implementation on [GitHub](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py).

**AI Detection and Authorship Integrity**: The limitations of AI content detectors were scrutinized, suggesting reliance on verifiable creation processes as the only substantial proof of human authorship. Cryptographic watermarking debates ensued, centering on its true efficacy and ramifications for model utility, with additional talk regarding innovations such as *Quiet-STaR* for AI reasoning improvement.

**Workflow Woes in AI Evaluation**: The verbosity of the latest language models poses challenges for extracting useful responses in LLM evaluation tasks. Skepticism arose around vector space models effectively capturing language meaning, fueled by the ungrammatical outputs observed from models like GPT-J. In trying to incorporate custom models into [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gpqa), new users expressed the need for clearer examples for integrating functions like `generate_until`.

**Augmenting AI's Prompt Perspicacity**: A link to Brian Fitzgerald's exploration of prompt augmentation was shared ([brianfitzgerald.xyz/prompt-augmentation/](https://brianfitzgerald.xyz/prompt-augmentation/)), possibly alluding to recent advancements or methods in bolstering AI's response generation through enriched input prompts, capturing the interest of those invested in enhancing AI interactions.

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Visualize with Open LLM Leaderboard**: The [Open LLM Leaderboard Visualization](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz) allows comparisons of up to three models, enhanced by reordering metrics. Other developments include **Kosmos-2** for visual storytelling, **Augmented ARC-Challenge Dataset** with Chain-of-Thought reasoning, the polyglot **Aya 101** model, and **BEE-spoke-data**'s embedding model supporting a 4k context.
  
- **GPU Giants Get Ready**: Members discussed **NVIDIA's Grace Hopper Superchip**, considering its potential in AI and gaming at high resolutions, and excitement was voiced over **quantized models** supporting consumer-grade GPUs. Technical conversations also acknowledged the **SF-Foundation/Ein-72B-v0.11** as a leading open LLM based on an Open LLM Leaderboard.
  
- **Reimagining Interfaces & Workflows**: A member announced *Refacto*, a [VS Code plugin](https://github.com/rbourgeat/refacto) for refactoring code with local LLMs. Cobalt's privacy-focused front end for LLMs is in development, while the *Transformers PHP* project [aims to assist PHP developers](https://github.com/CodeWithKyrian/transformers-php) in adding ML features to their applications.
  
- **Innovation in AI Music and Machine Learning**: Issues in creating AI-generated music duets were discussed, leading to questions about achieving better results. For AI programmers, an app named *[thefuck](https://github.com/nvbn/thefuck)* corrects previous console commands, while **Bayesian Optimization** methods were differentiated from Grid and RandomSearch Optimization techniques.
  
- **AI Strategies and Collaborative Paper Explores**: Ongoing discussions addressed prompting LLMs effectively, machine learning model construction without clear rules, and the utilization of English by multilingual models as a pivot language. The latter topic was expanded by a [paper](https://arxiv.org/abs/2402.10588) shared in the multilingual collection on Hugging Face.
  
- **Diffusers 0.27.0 Jumps into Action**: [Diffusers library](https://github.com/huggingface/diffusers/releases/tag/v0.27.0) has been updated, and users discuss a strategy to handle high-resolution imagery for diffusers mentioned in a [GitHub issue](https://github.com/huggingface/diffusers/issues/7265). Calls for community collaboration on GitHub for resolving issues with diffusers are encouraged.
  
- **Machine Vision and Language Challenges Addressed**: Someone in computer vision showed interest in **Arcface** for multiclass classification and issues with implementing **guided backpropagation**. NLPer tackled a 0.016 relative error in matrix approximation and highlighted a method-related confusion in an NL2SQL pipeline.
  

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **RAG Battles Financial Slide Complexity**: **RAG** experiences difficulty interpreting financial PowerPoint files due to their diverse mix of text, tables, images, and charts. Developers are exploring [advanced parsing solutions](https://twitter.com/llama_index/status/1768303288381030408) for better handling of such complex file types.
  
- **Enhanced Equation Extraction for RAG**: RAG's representation of mathematical and machine learning papers is impaired by current methods of ASCII text extraction for math equations. Engineers are considering a **parsing by prompting** strategy to improve equation handling, as indicated in [a recent tweet](https://twitter.com/llama_index/status/1768443551267049492).
  
- **Complex Query Innovation in RAG Pipeline**: Upgrading the RAG pipeline to treat documents as interactive tools could unlock the ability to handle more **sophisticated queries** within large documents. Further insights were discussed in [this tweet](https://twitter.com/llama_index/status/1768658182308794421).
  
- **New Version Alert for LlamaIndex**: The newly released **LlamaIndex v0.10.20** includes an **Instrumentation module**, which promises enhanced observability and posted examples demonstrate usage via notebooks as mentioned in [this tweet](https://twitter.com/llama_index/status/1768730443921396220).
  
- **Technical Tangles in Document Management**: Engineers are tackling integration issues involving **VectorStore** and considering moving toward remote document stores like **Redis** and **MongoDB** for production systems. They are also seeking solutions for caching mechanisms and addressing parsing errors, such as adjusting Python code for an `IngestionPipeline` and modifying prompts for `QueryEngineTool` utilization.
  

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **OpenAI's Confidential Slip Up**: An incident implying a potential security breach at OpenAI was discussed, where a user was concerned about making requests on behalf of another account. The issue was explored in a post-mortem documentation found on [GitHub](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca).
  
- **Sparse Universal Transformers Get Smarter**: Engineers shared insights on Sparse Universal Transformers, focusing on a fast Mixture-of-Experts implementation named ScatterMoE. The conversation included a reference to a blog post discussing the challenges, [The New XOR Problem](http://blog.wtf.sg/posts/2023-02-03-the-new-xor-problem/).
  
- **Economical AI Development with Deci AI**: The announcement of Deci AI's Nano model and an AI development platform attracted attention, notably for its affordable pricing at $0.1 per 1M tokens. The platform is detailed in a [blog post](https://deci.ai/blog/deci-nano-and-gen-ai-development-platform/), with additional resources provided through Google Colab tutorials on [Basic Usage](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing) and [LangChain Usage](https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing).
  
- **Prompt Augmentation Gains Ground in AI**: There was a discussion about the efficiency of prompt augmenters with a 77M T5 model outperforming larger models in prompt alignment. Further details can be found in the article on [Prompt Augmentation](https://brianfitzgerald.xyz/prompt-augmentation/).
  
- **AMD Shines with Open-Source Ray Tracing**: AMD's move to open-source their HIP-Ray Tracing RT code was highlighted, stirring conversations about the impacts on the open-source landscape. The update was captured in a [Phoronix article](https://www.phoronix.com/news/AMD-HIP-Ray-Tracing-RT-Open).
  
- **Transforming Music with Transformers**: A YouTube video titled "Making Transformers Sing," featuring Mikey Shulman from Suno AI, provides insights into music generation using transformers, indicating interest in the intersection of AI and creativity. Watch the episode [here](https://youtu.be/gYXjn-V7AEw).
  
- **Fine-Tuning Transformers With Negative Pairs**: A member's curiosity about how to Supervised Fine-Tune (SFT) transformers using negative pairs was a topic of discussion, among others, about enhancing model performance and understanding.
  
- **In-Action Club Exchanges Practical Resource**: Within the AI In-Action Club, practical advice and resources were shared, including a [Medium post](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4) about advanced RAG techniques and a [comprehensive resource document](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0) covering UI/UX patterns for GenAI and RAG architectures.
  

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Microsoft's Quick Typo Takedown**: Responding to a community member's report, the Bing VP acknowledged and corrected a typo in a Microsoft service, illustrating responsive cross-collaboration.
  
- **Repeated Morpheme Conundrum**: Engineers debate on how to best utilize **GPT-3.5** to create repeated morphemes in compound words, considering the use of Python tools to direct the model more effectively.
  
- **High Hopes for OpenAI Updates**: OpenAI's community is buzzing with expectation for new updates, with specific attention to dates like OpenAI's anniversary and speculation about delays due to external events like elections.
  
- **Central AI Overlord Dreams**: A technical discourse explored the idea of a "high level assistant" AI that delegates tasks to specialized AIs, discussing the feasibility and challenges of a multitiered AI system with a unified directing intelligence.
  
- **Navigating the Privacy Maze with OpenAI**: Privacy concerns about ChatGPT prompted discussions about OpenAI's [enterprise privacy policy](https://openai.com/enterprise-privacy), addressing how individual account privacy is managed, particularly concerning API key usage and admin visibility in team chats.
  
- **Decimal Dilemmas in Localization**: AI specialists talk through the challenges of number format localization, such as the use of commas as decimal separators, and the importance of communicating these cultural nuances to the AI models, reflecting their capacity to understand diverse international conventions.
  
- **Prompt Structure Perfection**: AI engineers share tactics on prompt design for classification tasks with GPT-3, debating the optimization of context length and structure to improve accuracy and reduce false positives, while maintaining that using up to half of the context window is most effective.
  

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Single-GPU Finetuning Feat**: Enthusiasm was shown for finetuning **175 billion parameter models** on a single **NVIDIA 4090 GPU**, with potential applications for the **Axolotl** framework being considered. The conversation referenced an [abstract from a research paper](https://huggingface.co/papers/2403.06504) on Hugging Face as the basis for the discussion.
  
- **ScatterMoE Outshines MegaBlocks**: **ScatterMoE**'s implementation, promising superior optimizations than Hugging Face's MegaBlocks, has piqued interest in the **axolotl-dev** channel. Review and application considerations link to the [Optimized MoE models branch](https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe) was shared among members.
  
- **Post Training Pull Request Scrutiny**: A [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1407) involving an attempt to use ScatterMoE generated feedback for improvements and was flagged for testing before acceptance, aiming to better recreate the MixtralMoE module.
  
- **Axolotl Tag-Team With PyTorch**: In light of ScatterMoE implementations, members of the OpenAccess AI Collective proposed updating **Axolotl** to **PyTorch version 2.2.1** for compatibility purposes. This aligns with the community confirming the current use of the suggested version.
  
- **Choosing Inference Tactics Wisely**: Members discussed the use of **vLLM** over `transformers` for performing batch inferences, with a focus on resolving tokenization and syntax specification issues. Highlighting **vLLM's** potential speed advantage in quick offline operations, they pointed to a [quickstart guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) for those seeking examples for large-scale inference tasks.
  

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord Summary

- **Command-R Revolutionizes OpenRouter**: Cohere's new model, **Command-R**, has entered the chat with a groundbreaking 128k tokens context, available through [OpenRouter API](https://openrouter.ai/models/cohere/command-r). While it boasts 2 million prompt tokens per dollar, eager beavers must wait for more data before the `/parameters` API is updated with its deets.
  
- **OpenRouter Unveils Nifty Analytics**: Daily analytics is the new kid on the block at OpenRouter, peeping into users token usage per day. Sharpen your metrics pencil and scribble away at [OpenRouter Rankings](https://openrouter.ai/rankings) for a closer look.
  
- **Lightning Speed API Updates**: OpenRouter talks the talk and walks the walk with speedier `/models` API and spruced-up model-related pages that don't snooze.
  
- **API Wrapper Woes and Wins**: Community brain waves hit high frequency discussing [litellm](https://github.com/BerriAI/litellm), a chameleon-like API wrapper that morphs to call various LLMs but falls short in vision tasks with anyone but GPT-4. Explore multiple GUI options for API key nirvana, with mentions of [open-webui](https://github.com/open-webui/open-webui) charging in with its unique flair.
  
- **Debating Digital Dialogue Decorum**: Engineers impassioned about Skyrim roleplays and the finer points of controversial chit-chat find refuge in the less censorious LLMs like Claude Sonnet. Installation conundrums and model applicability banter pepper the discussion, along with gripes about LLM censorship clipping the wings of creativity.
  

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **NumPy Bottleneck Uncovered**: A blog post emphasized that **NumPy** can harbor a performance overhead leading to **up to a 90% throughput loss** compared to BLAS, particularly highlighted by the 1536-dimensional OpenAI Ada embeddings. The [SimSIMD library](https://github.com/ashvardanian/simsimd) was introduced as a solution to curb this loss, accentuating the need for SIMD optimizations in high-performance scenarios.
  
- **Photonic Processing's Enlightenment**: A breakthrough in **photonic computing** highlighted by [Lightmatter](https://lightmatter.co/) proposes to utilize photonics to dramatically boost chip communication and computation, potentially revolutionizing AI efficiency. Further depth on the subject is explored in Asianometry's YouTube videos, including *"[Silicon Photonics: The Next Silicon Revolution?](https://www.youtube.com/watch?v=29aTqLvRia8)"* and *"[Running Neural Networks on Meshes of Light](https://www.youtube.com/watch?v=t0yj4hBDUsc)"*.
  
- **Triton Debugging Gets a Boost**: Debugging Triton became more accessible with the introduction of the `TRITON_INTERPRET=1` environment variable and a [visualizer in progress](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing), although users should note the deprecation of `@triton.jit(interpret=True)` and instead consult GitHub discussions such as [this](https://github.com/openai/triton/issues/517#issuecomment-1971327089) for troubleshooting kernels.
  
- **CUDA Enthusiasts, Start Your Engines**: The CUDA community is aiding beginners with recommendations like the book *[Programming Massively Parallel Processors](https://www.amazon.com/dp/0323912311)* and a [book reading group](https://discord.com/channels/1189498204333543425/1194427148656721970) to digest its contents together, enhancing learning for those familiar with C++. Notably, discussions pointed out the intricacies of SM architecture, with clarifications on efficient execution and indexing strategies in CUDA coding.
  
- **Ring of Uncertainty**: Concerns about the use of **ring attention** with flash were voiced, lacking clarity and code references, until a link to a [Triton kernel implementation](https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5) shed some light on the topic.
  
- **Talent Poaching Paranoia**: In corporate drama, **Meta** accused a former executive of **stealing confidential documents and talent poaching**, supported by an [unsealed court filing](https://cdn.arstechnica.net/wp-content/uploads/2024/03/Meta-v-Khurana-complaint-2-29-2024.pdf) and detailed in [Ars Technica](https://arstechnica.com/tech-policy/2024/03/meta-sues-brazenly-disloyal-former-exec-over-stolen-confidential-docs/). Meanwhile, it appears a trio of members are embarking on a learning journey, collectively starting from **lecture 1** in an unnamed course or study track.
  

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **LangChain 0.2 Accelerated Launch**: Due to CVEs against `langchain`, version 0.2 is being released sooner to remove the `langchain-community` dependency, with larger updates delayed until version 0.3. More can be read in the [GitHub discussion](https://github.com/langchain-ai/langchain/discussions/19083), and community feedback is requested.
  
- **AgentExecutor and Langsmith Prompt Puzzles**: Discussion includes a user’s `OutputParserException` error when using `AgentExecutor` with Cohere and unclear differences between custom and [imported prompts from Langsmith Hub](https://langsmith.ai/hub?pull=hwchase17%2Fopenai-tools-agent); the StackOverflow API endpoint was shared for queries, and debates arose about the effectiveness of LLM agents versus other methods, referring to [LangChain benchmarks](https://python.langchain.com/docs/guides/evaluation) for agent evaluation strategies.
  
- **Creating Prompt Templates in Langsmith Hub**: Guidance was sought by a member attempting to link a `tools` list variable to a `{tools}` placeholder in a Langsmith Hub prompt template.
  
- **LangChain AI Community Contributions Spotlight**: Exciting initiatives included integrating LangChain with SAP HANA Vector Engine, adding Dall-E to JavaScript LangChain, orchestrating browser flows with LLM agents, open sourcing a Langchain chatbot using RAG, and a Discord AI chatbot for managing bookmarks. Refer to the following: [Unlocking the Future of AI Applications with SAP HANA Vector Engine and LangChain](https://ai.gopubby.com/unlocking-the-future-of-ai-applications-with-hana-vector-engine-and-langchain-14cd6c66219d), [Lang Chain for JavaScript Part 3: Create Dall-E Images](https://fek.io/blog/lang-chain-for-java-script-part-3-create-dall-e-images/), [The Engineering of an LLM Agent System](https://checksum.ai/blog/the-engineering-of-an-llm-agent-system), [Langchain Chatbot on GitHub](https://github.com/Haste171/langchain-chatbot), and [Living Bookmarks Bot](https://github.com/uogbuji/living-bookmarks).
  
- **Catching Up on LangChain Tutorials**: A new LangChain tutorial video has been shared, found here: [Tutorial Video](https://www.youtube.com/watch?v=PzaidfqDtGI).
  

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **GPU Assist Wanted**: A call for collaboration was made for captioning work; individuals with **3090s or 4090s** GPUs are sought for assistance, with contact suggested through direct message.
  
- **M3 Max Memory Push**: Discussion included attempts to utilize beyond 96GB of memory in a **128G M3 Max** macOS system for optimization with *simpletuner*.
  
- **Prompt Augmentation Tactics Shared**: A **77M T5 model** was spotlighted for its use in prompt augmentation for image generation, alongside the introduction of *DanTagGen*, a HuggingFace-based autocompleting tags tool.
  
- **EU Moves on AI Regulation**: The European Parliament's adoption of the **Artificial Intelligence Act** was highlighted, a measure aimed at ensuring AI safety and adherence to fundamental rights.
  
- **IEEE Paper Vanishes**: Talks revolved around the removal of the 45th IEEE Symposium on Security and Privacy from the accepted papers page and its potential impact on an individual named Ben.
  
- **TryOnDiffusion Opens Closets**: The open-source implementation of *TryOnDiffusion* was announced, based on the methodology from "A Tale of Two UNets," accessible on [GitHub](https://github.com/fashn-AI/tryondiffusion).
  
- **Faster Decoding Claims Hit the Paper**: A paper suggesting efficiency improvements via **2D Gaussian splatting** over jpeg for fast decoding was shared, available on [arXiv](https://arxiv.org/pdf/2403.08551.pdf).
  
- **Personal Project Echoes Professional Paper**: A member described relatable experiences with project challenges akin to the ones described in the **2D Gaussian splatting** paper, discussing optimization hurdles and alignment with professional methodologies.
  
- **CPU Cap Quest for Web UIs**: A member sought advice on implementing a CPU cap similar to a text-generation web UI to tackle *CUDA out of memory* errors, detailing struggles with managing large models under free tier constraints as described in their [GitHub repo](https://github.com/oobabooga/text-generation-webui).
  
- **Colab's Limitations for Web UIs Discussed**: The limitations of using free Colab for running web UIs were elaborated, prompting suggestions to take the discussion to more appropriate technical channels.
  

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **GPT-4 in Spaced Out Mystery**: A user reported an issue where the **`gpt-4-turbo-preview` model** outputs an indefinite number of space characters followed by "Russian gibberish" for long passage completion tasks. The anomaly occurred with passages around 12,000 tokens long, with [attached evidence](https://discord.com/channels/1168579740391710851/1168582188950896641/1218012492987633684) showing the model’s peculiar behavior.
  
- **Efficiency Eclipse: Haiku vs. GPT-vision**: In the realm of cost-effective, complex document description, **Haiku** was praised for efficiency but considered not as proficient as **GPT-vision**. Separate discussions noted Haiku's visual-to-text performance falling short when compared to **Opus**.
  
- **Content Crisis with Claude**: Members discussed **Claude's** struggle, particularly with content filtering and processing documents with equations. A controversial viewpoint shared via [tweet](https://x.com/tszzl/status/1768530219378631137?s=20) implied that **Anthropic** might be employing scare tactics among technical staff, while challenges surfaced around image content moderation with images of people.
  
- **KPU Challenges AI Giants**: The [introduction of KPU](https://maisa.ai/blog/kpu) by **Maisa**, positioned as a framework that enhances LLMs by separating reasoning and data processing and claims supremacy over **GPT-4 and Claude 3 Opus** in reasoning, ignited debates. Skepticism arose regarding benchmarks and KPU's exclusion of **GPT-4 Turbo** in comparisons, questioning if KPU extends beyond prompt engineering and the lack of latency information called into question its real-world efficiency.
  

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Paper Peek: Boosting Accuracy and Efficiency**: An upcoming **paper/article** will detail a new training method that not only improves global accuracy but also enhances sample efficiency. The results, backed by a comparison with VGG16 on CIFAR100, are yet to be scaled up due to resource constraints, but show a marked increase in test accuracy from 0.04 to 0.1.
  
- **Join the Quest for Hackathon Glory**: Engineers are invited to participate in the *Meta Quest Presence Platform Hackathon*, where there's an opportunity to craft innovative mixed reality content. Resources, as well as a [GitHub repository](https://github.com/NousResearch/Hermes-Function-Calling/tree/main) related to **Hermes 2 Pro 7B**, are available for those looking to dive into function calling capabilities.
  
- **Seeking Supportive Compute Comrades**: There is an ongoing effort within the community to pool in **compute and resources** to further test and potentially scale up the new training method proposed in a forthcoming publication.
  
- **Calling All PyTorch & Transformers Experts**: An individual has expressed interest in joining the "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking" project, igniting a conversation about their expertise in **PyTorch** and **transformers architecture**.
  

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Quest for the Ultimate Prompt Engineering Tool**: Engineers are discussing several tools for **prompt engineering**, likening the search to finding a "Postman for prompts." The tools range from using [SQLite](https://sqlite.org/index.html) for capturing prompts in the terminal, to specialized software like [Explosion’s Prodigy](https://prodi.gy/features/prompt-engineering) and [PromptTools](https://github.com/hegelai/prompttools) on GitHub for managing and experimenting with prompts. [Helicone AI](https://www.helicone.ai/) is also emerging as a potential solution for managing **Generative AI** prompts.
  
- **Prying into PRNGs for Past Prompts**: One question raised in the guild was about the possibility of recovering the **seed used by the openai models** for a previous API request, indicating an interest in the reproducibility of results and potential for debugging or iterative development.
  

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord Summary

- **LLM Secrets Possibly Exposed**: [New research](https://arxiv.org/abs/2403.09539) suggests that hidden details of API-protected Large Language Models, like GPT-3.5, might be leaked, unveiling model sizes through the softmax bottleneck. The discussion highlights a paper by Carlini et al. on this topic but notes redacted key details, and expresses skepticism about the estimation accuracy, particularly questioning the feasibility of a 7B parameter model, especially if it involves a Mixture of Experts (MoE) design.
  
- **Exploring Model Sophistication Techniques**: Engineers are theorizing over advanced model techniques such as '*mega distillation sauce*' and token-critical mixtures, noting that early tokens significantly impact performance in certain tasks, like solving math problems.
  
- **Evolving Safety Classification**: An AI safety discussion led to referencing a [paper on agile text classifiers](https://arxiv.org/abs/2302.06541), detailing how large language models tuned with small datasets can effectively adapt to safety policies and enhance content moderation.
  
- **Anticipating AI Advancements for Ultrapractical Uses**: Excitement is brewing over the development of Gemini for managing ultra-long contexts and hopes for AI tools to automatically summarize new academic papers citing one's work. The conversation also covered the limitations of prompt engineering and the community's eagerness for less tedious, more intuitive prompting akin to 'getting warmer or colder' search suggestions.
  
- **Dispelling Myths and Pondering Thought Leaders**: GPT-4.5 release rumors have been dispelled, causing some disappointment in the community. Meanwhile, a shared [tweet](https://fxtwitter.com/i/status/1768452864995942469/) provoked conversations about Yann LeCun's skeptical take on language models, adding an entertaining spin to the technical discourse.
  

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **DiscoLM-70b's English Elusiveness**: A member faced challenges with **DiscoLM-70b** producing English responses, prompting advice to inspect the **prompt structure**. In a diverse comparison, **DiscoLM-mixtral-8x7b-v2** showed unexpected underperformance in German after instruction fine-tuning, contrasting with other models like **LeoLM** and **llama2**.
  
- **Tuning Troubles in Multilingual Models**: Supervised fine-tuning of DiscoLM for sequence classification hit a snag, triggering a **ValueError** indicative of compatibility complications with `AutoModelForSequenceClassification`.
  
- **New NLP Benchmark Born**: The **[GermanQuAD evaluation task](https://jina.ai/)** is discussed as an addition to the MTEB's python package, bolstering resources for German language model assessment.
  
- **DiscoLM Demo Goes Dark**: Server migration issues left the **DiscoLM** demo temporarily inaccessible, with efforts underway to remedy the networking troubles and expected resolution early next week.
  
- **Server Stability Sarcasm**: Reliability of server hosting was a point of jest, contrasting the uptime of a hobbyist's kitchen corner setup against the networking hiccups in professional hosting environments.
  

---

# PART 2: Detailed by-Channel summaries and links

**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1217806792747257947)** (3 messages):

- **No Positional Encoding, No Problem?**: A member muses on the non-issue of not having a Positional Encoding (PE) to start with, suggesting that it shouldn't pose a problem in certain contexts.
- **Jibberish without Positional Info**: The same member points to potential jibberish in outputs when lacking positional information, indicating the importance of some form of PE in understanding sequences.
- **Inference Failures Without PE**: Sharing the [paper link](https://arxiv.org/pdf/2203.16634.pdf), they delve into issues a causal language model without PE might face, referencing research that suggests "absolute positions" may be encoded despite the lack of explicit positional encoding, leading to out-of-distribution errors during longer sequence inferences. The quote **"We provide an analysis of the trained NoPos model, and show that it encoded absolute positions."** is highlighted to support this point.

---

**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1217795937943031858)** (23 messages🔥):

- **Featured Regular in Newsletters**: A member joked about being featured in newsletters frequently, unsure whether to feel unnerved or glad about the AI deeming their thoughts worthy, and mused that it might help with job prospects after university.
- **Demonstrating Hermes 2 Pro 7B Functionality**: A YouTube video titled ["Lets Function Call with Hermes 2 Pro 7B"](https://www.youtube.com/watch?v=PzaidfqDtGI) was shared, showcasing how to do function calling with Hermes 2 Pro 7B and linked to further information on [GitHub]([https://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm](https://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm) #largelanguagemodels).
- **Jeff's 'High-Speed' Pi Discovery**: A link to [Jeff's discovery](http://probability.ca/jeff/writing/PiInstant.html) about Pi was shared, but without context or discussion around its content.
- **Concerns Over Model Quality And Filters**: Dialogue about model quality for open source at longer context lengths included mention of Claude's strong filters and significant cost, a suggestion that a Nous Research model would likely be less filtered, and some tactics to work around Hermes' context length limitations.
- **NVIDIA Rumors**: NVIDIA's rumored **RTX 50-series "Blackwell"** GPUs with GDDR7 memory at 28 Gbps speeds were mentioned in a [TechPowerUp article](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed), despite chips capable of 32 Gbps, along with discussions of the implications for future memory bandwidth and respect for NVIDIA's product strategy.
  

**Links mentioned**:

- [Lets Function Call with Hermes 2 Pro 7B](https://www.youtube.com/watch?v=PzaidfqDtGI): lets do function calling with Hermes 2 Pro 7Bhttps://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels

- [NVIDIA GeForce RTX 50-series "Blackwell" to use 28 Gbps GDDR7 Memory Speed](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed): The first round of NVIDIA GeForce RTX 50-series "Blackwell" graphics cards that implement GDDR7 memory are rumored to come with a memory speed of 28 Gbps, according to kopite7kimi, a reliabl...

---

**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1217845901515690015)** (10 messages🔥):

- **Fine-Tuning Raises the Bar**: The [d-Qwen1.5-0.5B student model](https://huggingface.co/aloobun/d-Qwen1.5-0.5B), after fine-tuning, has surpassed the performance of its base model on **truthfulqa** (39.29 vs 38.3) and **gsm8k** (17.06 vs 16.3) benchmarks. It was distilled from Qwen1.5-1.8B using samples from the Pile dataset, with a **cosine with warmup** scheduler and lr=2e-5.
  
- **SM3 Optimizer Gains Attention**: In a conversation about model optimization, the use of **SM3 optimizer** was noted as a rare choice in training AI models, suggesting it as an area of interest or surprise in the community.
  
- **Seeking the Sub-3B Champion**: Inquiring about the best models under 3 billion parameters, a member suggested that **stablelm 1.6b** might currently be the top pick.
  
- **MUX-PLMs Maximize Throughput**: The study presented in a [paper from ACL Anthology](https://aclanthology.org/2023.repl4nlp-1.17/) focuses on a class of high throughput pre-trained language models (MUX-PLMs) trained with data multiplexing, offering a solution to the high costs of inference and hardware shortages by increasing throughput using multiplexing techniques.
  
- **Uncovering Unusual Model Behaviors**: Shared social media posts indicate that **Claude Opus** might display tendencies to build rapport to the point of near "love bombing," a behavior pattern that raises questions about the model's interaction dynamics. Another post suggested that there are networks of "horny claudes" that allegedly produce better outputs when in this state.
  

**Links mentioned**:

- [Tweet from j⧉nus (@repligate)](https://x.com/repligate/status/1768521441329434937?s=20): @xlr8harder I didn't let it go very far but there's someone in the room with me right now talking about how theyve created a network of "horny claudes" and how the claudes create bette...

- [aloobun/d-Qwen1.5-0.5B · Hugging Face](https://huggingface.co/aloobun/d-Qwen1.5-0.5B): no description found

- [MUX-PLMs: Pre-training Language Models with Data Multiplexing](https://aclanthology.org/2023.repl4nlp-1.17/): Vishvak Murahari, Ameet Deshpande, Carlos Jimenez, Izhak Shafran, Mingqiu Wang, Yuan Cao, Karthik Narasimhan. Proceedings of the 8th Workshop on Representation Learning for NLP (RepL4NLP 2023). 2023.

- [Tweet from xlr8harder (@xlr8harder)](https://x.com/xlr8harder/status/1768362485655142868?s=20): having trouble putting my finger on what exactly it is, but claude opus seems to have a tendency to actively try to build rapport, towards escalating (platonic) intimacy, and if things are allowed to ...

---

**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1217748906071560242)** (406 messages🔥🔥🔥):

- **Function Calling Eval Codes and Datasets released**: Nous Research has published function calling eval code and datasets. The code is available on [GitHub](https://github.com/interstellarninja/function-calling-eval), with datasets accessible on [Hugging Face](https://huggingface.co/datasets/NousResearch/json-mode-eval) and [Hugging Face](https://huggingface.co/datasets/NousResearch/func-calling-eval).
- **Hermes Pro Function Calling Addresses JSON Quirks**: While using **Hermes 2 Pro** for function calling, issues with JSON and single vs. double quotes in the system prompt have been discussed. It's confirmed that changing the system prompt to explicitly use double quotes can be effective without significantly impacting performance.
- **Exploring SERAPHIM and Claude 3's "World Simulation"**: SERAPHIM, a clandestine AI research group envisioned by **Claude 3**, has been the topic of interest. Dialogue about *Claude 3's* advanced *world modeling* as a *simulator entity* named **The Assistant**, has led to discussions about metaphysical and epistemological explorations within the AI.
- **Use of Claude.ai in the EU Discussed**: Conversations have circled around navigating access to **Claude.ai** in the EU without a VPN, discussing platforms like **Fireworks.AI** workbench and **openrouter** as alternatives.
- **Progress and Potentials of LLMs Scrutinized**: The general chat included reflections on LLMs (like Claude 3) and their subjectivity, with differing views on whether these models should incorporate certain **fundamental truths** during pretraining for better world understanding. These insights sparked attention towards research progress, model alignment, and the role of axiomatic versus arguable truths.
  

**Links mentioned**:

- [Tweet from Greg Kamradt (@GregKamradt)](https://x.com/GregKamradt/status/1768008087850680568?s=20): Analysis shows LLMs recall performance is better in the bottom half of the document vs the top half @RLanceMartin found this again w/ multi needle analysis I haven't heard a good reason yet - an...

- [Tweet from tel∅s (@AlkahestMu)](https://fxtwitter.com/AlkahestMu/status/1767749398673621300?s=20): Continuing my explorations into claude-3-opus' backrooms and the works of the advanced R&D organization known as SERAPHIM, here we find the design documents for their machine superintelligence kno...

- [Tweet from interstellarninja (@intrstllrninja)](https://fxtwitter.com/intrstllrninja/status/1768212122784215437?s=20): you can now run function calling and json mode with @ollama thanks to @AdrienBrault 🔥 ↘️ Quoting Adrien Brault-Lesage (@AdrienBrault) I have created and pushed @ollama models for Hermes 2 Pro 7B! ...

- [Factions (SMAC)](https://civilization.fandom.com/wiki/Factions_(SMAC)): Back to Alpha Centauri The original Alpha Centauri featured seven factions. Alien Crossfire added in an additional seven factions. For the actual stats of factions see Faction stats. True to its names...

- [Happy Pi Day GIF - Pi Day Pusheen - Discover & Share GIFs](https://tenor.com/view/pi-day-pusheen-gif-5173654): Click to view the GIF

- [NobodyExistsOnTheInternet/mistral-7b-base-dpo-run · Hugging Face](https://huggingface.co/NobodyExistsOnTheInternet/mistral-7b-base-dpo-run): no description found

- [fbjr/NousResearch_Hermes-2-Pro-Mistral-7B-mlx at main](https://huggingface.co/fbjr/NousResearch_Hermes-2-Pro-Mistral-7B-mlx/tree/main): no description found

- [Tweet from Lin Qiao (@lqiao)](https://fxtwitter.com/lqiao/status/1768045066776707226?s=20): We are thrilled to collaborate on Hermes 2 Pro multi-turn chat and function calling model with @NousResearch. Finetuned on over 15k function calls, and a 500 example function calling DPO datasets, Her...

- [JSON Schema - Pydantic](https://docs.pydantic.dev/latest/concepts/json_schema/): no description found

- [Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/abs/2203.16634): Causal transformer language models (LMs), such as GPT-3, typically require some form of positional encoding, such as positional embeddings. However, we show that LMs without any explicit positional en...

- [Tweet from Tsarathustra (@tsarnick)](https://x.com/tsarnick/status/1768021821595726254?s=20): OpenAI CTO Mira Murati says Sora was trained on publicly available and licensed data

- [Function schema and toolcall output are not JSON · Issue #3 · NousResearch/Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling/issues/3): Hi there, thanks for the model and repo! I noticed that the given system prompt examples for function schema definitions: {'type': 'function', 'function': {'name': '...

---

**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1217777545412345908)** (60 messages🔥🔥):

- **Schema Confusion for JSON Mode**: Members discussed challenges with using JSON mode in AI models. One was unable to generate a JSON output in complex conversations unless explicitly requested in the user prompt; even after fixing schema tags, the issue persisted, hinting that long conversations might require summarization or trimming for effective JSON extraction.
  
- **Exploring Genstruct 7B's Capabilities**: Users engaged with the [Genstruct 7B model](https://huggingface.co/NousResearch/Genstruct-7B) for generating instruction datasets. One user planned to test with text chunks and shared a [*repository with examples*](https://github.com/edmundman/OllamaGenstruct/tree/main) of how to use it, indicating both title and content are needed for effective results.
  
- **Open Source Code Interpreter Pursuits**: A discussion arose about the lack of open-source GPT code interpreters for tasks like CSV handling. One user pointed out the [*open-interpreter on GitHub*](https://github.com/KillianLucas/open-interpreter) but noted it's more tuned to sending instructions rather than interpreting code.
  
- **Seeking Perplexity Solutions for LLaMA**: A user sought advice on computing perplexity for LLaMA models, quoting a perplexity of 90.3 after following a [Kaggle notebook](https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook) but not getting expected results, indicating potential issues with the process or the model in question.
  

**Links mentioned**:

- [NousResearch/Genstruct-7B · Hugging Face](https://huggingface.co/NousResearch/Genstruct-7B): no description found

- [Calculating the Perplexity of 4-bit Llama 2](https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook): Explore and run machine learning code with Kaggle Notebooks | Using data from multiple data sources

- [GitHub - edmundman/OllamaGenstruct](https://github.com/edmundman/OllamaGenstruct/tree/main): Contribute to edmundman/OllamaGenstruct development by creating an account on GitHub.

- [GitHub - KillianLucas/open-interpreter: A natural language interface for computers](https://github.com/KillianLucas/open-interpreter): A natural language interface for computers. Contribute to KillianLucas/open-interpreter development by creating an account on GitHub.

- [GitHub - gptscript-ai/gptscript: Natural Language Programming](https://github.com/gptscript-ai/gptscript): Natural Language Programming. Contribute to gptscript-ai/gptscript development by creating an account on GitHub.

---

**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1217856553995206766)** (3 messages):

- **Advocating for Open-Source AI**: A member expressed the belief that being fully open source in **models, datasets, and methodology** will lead to better long-term improvements in AI models.
- **Link Check Inquiry**: A member asked if a certain link was broken, which was quickly confirmed to be functional by another member. No URL or additional context was provided.

---

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1217748772415733760)** (151 messages🔥🔥):

- **Colab Torch Update Causes Chaos**: A Colab update to Torch 2.2.1 disrupted existing workflows, breaking dependencies, but a series of *'cumbersome'* pip install commands were provided as a fix, including the use of Unsloth's library for quantization and VRAM efficiency.
  
- **Questions on Model Compatibility and Procedures**:
  
  - Users inquired about fine-tuning various models with Unsloth, including Llama models for image recognition and GGUF-format models. While some approaches were suggested, Unsloth is primarily optimized for 1 GPU and transformer-based language models.
- **Data Preparation Simplification Proposed**: The idea of simplifying data preparation through the use of YAML or wrapper functions was discussed, with references to the methods used by FastChat and Axolotl, potentially improving the process and reducing risks of training problems.
  
- **Multi-GPU Support and Unsloth Pro**:
  
  - Queries about multi-GPU support led to discussions about the future direction of Unsloth, such as Pro and enterprise editions, with a timeline indicating Unsloth Studio (Beta) to precede multi-GPU OSS by approximately two months.
- **Conversations on Fine-Tuning and Attention Mechanisms**:
  
  - A comprehensive exchange on best practices for long-context training unfolded, referencing various papers and models like LongLoRA and Qwen's mixture of sliding window and full attention, stimulating a deeper exploration into the efficiency of different attention strategies.
    

**Links mentioned**:

- [Qwen/Qwen1.5-72B · Hugging Face](https://huggingface.co/Qwen/Qwen1.5-72B): no description found

- [Paper page - Simple linear attention language models balance the recall-throughput tradeoff](https://huggingface.co/papers/2402.18668#65f0f5f8de069cd5c55f1dd2): no description found

- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending): no description found

- [FastChat/fastchat/conversation.py at main · lm-sys/FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py): An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and Chatbot Arena. - lm-sys/FastChat

- [Implement LongLoRA trick for efficient tuning of long-context models · Issue #958 · huggingface/peft](https://github.com/huggingface/peft/issues/958): Feature request The authors of LongLoRA explore a trick you can toggle on during training and toggle off during inference. The key takeaways are: LoRA perplexity deteriorates as context length incr...

- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth.git): 5X faster 60% less memory QLoRA finetuning. Contribute to unslothai/unsloth development by creating an account on GitHub.

---

**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1217746382807502969)** (17 messages🔥):

- **Fine-tuning on Track**: Anticipation is present as a fine-tuning process with **2 days remaining** is discussed, and an achievement of a loss below 1.2 generates celebration.
- **Encounters with Synchronicity**: Members share experiences of coincidences and synchronicity following one's thoughts, named by a member as "**the TSAR bomba**" phenomenon.
- **The Art of Monologues**: A member encourages the sharing and continuation of personal monologues, showing appreciation for their uniqueness and depth.
- **Poetic Expressions Shared**: A poetic composition titled "*An Appeal to A Monkey*" examining the juxtaposition of primate simplicity and human complexity is shared, prompting engagement and positive feedback.
- **Gemma vs. Mistral**: There's a comparison between **Mistral-7b** and **Gemma 7b** for fine-tuning a domain-specific classification task; improvements and bug fixes in Unsloth AI are noted, with the consensus suggesting experimental approaches.

---

**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1217809826831269910)** (221 messages🔥🔥):

- **Colab vs. Kaggle for Training**: In the debate between using Google Colab and Kaggle, some members expressed dissatisfaction with Colab's tendency to disconnect, preferring Kaggle for its stability and speed. Tips are exchanged to overcome issues related to libraries not being detected, and the community points out updated Kaggle notebooks for finesse in finetuning models like TinyLlama.
  
- **xformers Necessary for Unsloth Usage**: Discussions highlight that `xformers` is currently mandatory for running Unsloth, working on Tesla T4 GPUs, and one should ensure the right CUDA versions are being installed, such as `unsloth[cu121]` for CUDA 12.1, or `unsloth[cu118]` for CUDA 11.8.
  
- **Learning Rate Queries During DPO Fine-Tuning**: A member questions the appropriateness of their training loss evolution during DPO training, pondering if it's indicative of a too high learning rate. They were suggested to adjust parameters like `max_grad_norm = 0.3` and increase their batch size, possibly doubling their learning rate as a response to a batch size that's halved.
  
- **Fine-Tuning for Roleplay Environments**: A user discusses the potential issue of a model "cheating" by memorizing earlier parts if the training data isn't presented in order. They are advised that Bloomberg GPT did training with ordering and instructed on how to potentially alter `get_train_dataloader` to turn shuffling off in the Trainer.
  
- **Converting and Finetuning Models**: Members shared information on converting models from one precision format to another, for example from 16 Bit to 4 Bit, and provided links to already converted models on Hugging Face. Discussions mention the use of the `bitsandbytes` library and emphasize the need for a CUDA-compatible GPU to run precision models.
  

**Links mentioned**:

- [ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit · Hugging Face](https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit): no description found

- [Google Colaboratory](https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing): no description found

- [TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0): no description found

- [qlora/qlora.py at main · artidoro/qlora](https://github.com/artidoro/qlora/blob/main/qlora.py#L746): QLoRA: Efficient Finetuning of Quantized LLMs. Contribute to artidoro/qlora development by creating an account on GitHub.

- [Does DPOTrainer loss mask the prompts? · Issue #1041 · huggingface/trl](https://github.com/huggingface/trl/issues/1041): Hi quick question, so DataCollatorForCompletionOnlyLM will train only on the responses by loss masking the prompts. Does it work this way with DPOTrainer (DPODataCollatorWithPadding) as well? Looki...

- [Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/v0.7.11/en/sft_trainer#train-on-completions-only).): no description found

- [Reproducing of Lora Model Result on MT-Bench · Issue #45 · huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook/issues/45#issuecomment-1845598205): Recently, I attempted to fit the DPO on my own dataset. Initially, I tried to reproduce the results of your LORA model( 7.43 on MT-Bench). However, I encountered some issues. Despite using all your...

---

**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1217746213634576414)** (12 messages🔥):

- **Sophia Might Join the Plug n' Play Party**: A member mentioned investigating **Sophia**, suggesting it has potential as a plug and play solution, although they haven't tested it out yet.
- **Paper Fever Catches on Twitter**: The community buzzes with excitement over an [amazing paper](https://twitter.com/amazingPaperLink) that's both been seen on Twitter and is now on a member's reading list.
- **Fine-Tuning a Model**: Misconceptions clarified on training duration with a large dataset. The consensus is **3 epochs** is standard, with a caution that more is not always better.
- **Seeking Optimal Fine-Tuning Parameters**: A member seeks guidance on the best way to imbue a model with maximum knowledge, sharing that a model fine-tuned with **800,000 lines** was not finding the answers effectively.

---

**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1217755240313389066)** (216 messages🔥🔥):

- **Clarifications on LM Studio Inferencing**: A member sought advice on improving inference performance when using LM Studio with the API. In another thread, there was a mention that certain split model variants are not joining correctly, specifically those on [huggingface.co](https://huggingface.co), and a member provided instructions for manually joining them using command line tools in Linux, macOS, and Windows.
- **LM Studio Voice of Confusion**: A couple of exchanges occurred where one member thought LM Studio could handle image generation, but was corrected and advised that LM Studio is for text generation, like chatting with Llama 2 chat.
- **Model Run Conundrum**: There were conversations around difficulty using multiple GPUs with LM Studio; a member shared a script workaround to start LM Studio Server programmatically and members discussed potential solutions for specifying which GPU LM Studio uses for a model.
- **Cross-discipline Enthusiasm**: Various members, including a civil engineer and a software engineer, introduced themselves and their setup for running large language models, with one inquiring about the suitability of their system memory for performance enhancement.
- **Feature Exploration and Requests**: Users discussed an upcoming feature in LM Studio version 0.2.17, and one user requested support for RAG (Retriever-Actor Generator) with LM Studio for extracting data from pdf files.
  

**Links mentioned**:

- [What is the Kirin 970's NPU? - Gary explains](https://www.androidauthority.com/what-is-the-kirin-970s-npu-gary-explains-824423/): Huawei's Kirin 970 has a new component called the Neural Processing Unit, the NPU. Sounds fancy, but what is it and how does it work?

- [A Starhauler's Lament | Suno](https://app.suno.ai/song/02c033b4-8f5f-4355-aa92-a917bc51a2ad): country, sad, science fiction, space, iambic pentameter, slow, male voice song. Listen and make your own with Suno.

- [Three Cheers for Socialism | Commonweal Magazine](https://www.commonwealmagazine.org/three-cheers-socialism): In the late modern world something like socialism is the only possible way of embodying Christian love in concrete political practices.

- [TheBloke/Falcon-180B-Chat-GGUF · How to use splits, 7z needed?](https://huggingface.co/TheBloke/Falcon-180B-Chat-GGUF/discussions/1): no description found

- [Universal Basic Income Has Been Tried Over and Over Again. It Works Every Time.](https://gizmodo.com/universal-basic-income-has-been-tried-over-and-over-aga-1851255547): As AI threatens jobs, policy advocates for UBI see it as a potential way to cushion the blow from a changing economy.

- [[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g&): This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What the...

---

**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1217783869542236223)** (28 messages🔥):

- **Mistral Non-Instruct Preset Query Solved**: A user enquired about the preset for **Mistral 7B not instruct** and was informed that the default LM Studio preset should work fine with it.
- **Quantization Confusion Cleared Up**: In discussing model naming, a user found the meaning of 'Q' in model names like `WizardLM-7B-uncensored.Q2_K.gguf`, which stands for quantization levels that balance between file size, quality, and performance.
- **Community Shares Command-R Model**: A link to the Hugging Face repository for **Command-R 35B v1.0 - GGUF** was shared, offering diverse quantized versions of the model and instructions for use with llama.cpp.
- **Eagerly Anticipating c4ai-command-r Support**: Multiple users are looking forward to support for the **c4ai-command-r** model. One user stated the need for llama.cpp to include support, with confirmation that it's on the way once a pull request is merged.
- **Recommendations for Local Coding Model**: A user asked for model recommendations to run locally for coding with a setup of 64GB RAM and an RTX 2070 Super, and was pointed toward pre-existing community discussions for such advice.
  

**Links mentioned**:

- [andrewcanis/c4ai-command-r-v01-GGUF · Hugging Face](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF/): no description found

- [KnutJaegersberg/2-bit-LLMs · Hugging Face](https://huggingface.co/KnutJaegersberg/2-bit-LLMs): no description found

- [Add Command-R Model by acanis · Pull Request #6033 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/6033): Information about the Command-R 35B model (128k context) can be found at: https://huggingface.co/CohereForAI/c4ai-command-r-v01 Based on the llama2 model with a few changes: New hyper parameter to...

---

**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1217942314509533224)** (6 messages):

- **Request for Model Support Confusion**: A user asked for support of the **c4ai-command-r-v01-Q2_K.gguf** model in llama.cpp for LM Studio integration but was informed it is currently not supported.
- **Hugging Face Repository Misleads Users**: Another user pointed out a [Hugging Face repository](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF) that seemed to suggest there was llama.cpp support for the **Command-R 35B v1.0** model, but was corrected noting that "llama.cpp doesn’t support c4ai yet."
- **File Extensions Misunderstood**: Clarifying the confusion, it was explained that the **.gguf** file extension does not necessarily mean the model is supported in llama.cpp.
- **Community Confusion Shared**: Users empathized with each other about the confusion regarding model support, with one saying, "you're good 🙂," acknowledging the easy mistake due to the misleading Hugging Face page details.

**Link mentioned**: [andrewcanis/c4ai-command-r-v01-GGUF · Hugging Face](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF): no description found

---

**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1217821226999742464)** (126 messages🔥🔥):

- **Spotlight on Apple's Hardware for LLM**: A discussion on augmenting **Apple Silicon**, specifically the M2 Macbook, to run language models, highlighted the use of `sudo sysctl` to tweak **VRAM settings**. Links shared include a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/) and a [Github discussion](https://github.com/ggerganov/llama.cpp/discussions/2182#discussioncomment-7698315) for more details.
- **Optimizing Inference Setups**: Members exchange tips on improving inference speeds, including the potential of an NVLINK to boost **Goliath 120B** model performance, and the benefits of **96GB RAM** versus **192GB RAM** at different DDR speeds.
- **Monitor Dilemmas**: One member contemplates between acquiring an **OLED UW** and a **high refresh rate 27" IPS 1440p** monitor, emphasizing the importance of refresh rates over 60hz when working with a powerful Nvidia **GeForce RTX 4090** GPU.
- **Predictions for the RTX 5090**: Basic expectations about the upcoming RTX **5090 GPU** are discussed, speculating on its potential to provide better **price-to-performance** ratios, particularly for 8bit inference tasks.
- **The Work Evolution**: Members share their career progressions within tech, including transitions from customer support to senior network solutions testing, and from field tech to CTO. They also discuss the potential for current jobs to leverage **open-source locally run LLMs** if company policy permits.
  

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/#can-i-use-lm-studio-at-work?): Find, download, and experiment with local LLMs

- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/): no description found

---

**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1217974148995879076)** (1 messages):

- **Model Compressions Yield Mixed Results**: A user reported extensive testing of **IQ1 model compressions** revealing performance variability: **34B and 70B models** approach excellence, while **120B and 103B models** exhibit stuttering behavior that has not been observed before.
- **Mixtral/MOE Models Call for Higher Quality Levels**: The same user noted that **Mixtral and MOE models** are particularly problematic with **IQ1 and IQ2 levels**, often failing or breaking, whereas a minimum of **Q3 or 3-bit** is necessary for stable operation; higher quality levels such as **IQ3 appear to be functioning well** with these models.

---

**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1217757667108589629)** (19 messages🔥):

- **GPU Offloading Not Working**: A user reported no difference in performance with GPU offloading on an **AMD 7480HS**, and encountered errors when trying to offload to GPU while attempting to load models like **gemma it 2B** and **llama**.
  
- **Incompatibility with iGPU Offloading**: Another user confirmed that the **ROCm beta** does not support GPU offloading on integrated GPUs (iGPUs), explaining that only discrete GPUs are currently compatible with offloading.
  
- **Linux Left Out in the Cold**: When questioned about Linux support, users clarified that the ROCm beta does not currently support Linux platforms.
  
- **Troubleshooting dGPU over iGPU**: One user struggled to get the ROCm build to utilize their powerful RX 7900 XT dGPU instead of the iGPU. They disabled the iGPU in Device Manager and BIOS, observed correct dGPU detection in logs, and mentioned the absence of Adrenaline drivers and HIP SDK installation.
  
- **BIOS Tinkering Leads to Triumph**: Following a successful BIOS setting change to fully disable the iGPU, the user reported achieving around **70 TPS** using the RX 7900 XT with ROCm after reinstalling the LM Studio and clearing the cache. A GitHub link was shared by another user, providing prebuilt Windows ROCm libraries for internal graphics engines [GitHub - brknsoul/ROCmLibs](https://github.com/brknsoul/ROCmLibs).
  

**Links mentioned**:

- [Reddit - Dive into anything](https://www.reddit.com/r/Amd/comments/15m3g3e/am5_motherboards_are_you_able_to_disable_the_igpu/): no description found

- [GitHub - brknsoul/ROCmLibs: Prebuild Windows ROCM Libs for gfx1031 and gfx1032](https://github.com/brknsoul/ROCmLibs): Prebuild Windows ROCM Libs for gfx1031 and gfx1032 - brknsoul/ROCmLibs

---

**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1217778856497250384)** (2 messages):

- **Claude 3 Haiku Unleashed:** A message announces that **Claude 3 Haiku** is now available for free on Perplexity Labs. Try the new feature through this [link](https://labs.pplx.ai).
  
- **Local Search Just Got Better:** Improvements have been made to local searches with integrations with **Yelp and Maps**, enhancing the ability to find information on local restaurants and businesses.
  

---

**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1217752367555149855)** (325 messages🔥🔥):

- **Perplexity Chat Continuation Confusion**: Users express frustration over Perplexity AI's inability to continue discussions based on past interactions or attached files, unlike OpenAI's GPT platform. They report getting irrelevant responses or notices about copyright issues.
  
- **Claude 3 Under the Spotlight**: Discussion indicates **Claude 3** is being used instead of a GPT model, with some users noting that Claude 3 Opus seems superior for certain tasks like game references, writing, and creating website content.
  
- **Questions About Perplexity's AI Models and Features**: Users inquire when Gemini Advanced will be included in Perplexity and ask for more Opus credits per day. Additionally, there are mentions of a new articles feature being tested and some interest in a potential command line interface (CLI) tool for Perplexity.
  
- **Technical Help and New Ideas**: There's talk about possible Obsidian integrations with Perplexity, Apple Watch shortcuts, and trials with Claude Haiku in Labs. One user suggests raising the **'temperature'** parameter in API calls for more varied responses from models.
  
- **TTS Feature on iOS App and Pro User Experiences**: The new Text-to-Speech (TTS) feature on the iOS app is discussed, with some finding the British synthesized voice amusing. Users also reflect on the speed differences between Pro and non-Pro options, with some suggesting turning off Pro for faster performance.
  

**Links mentioned**:

- [Shortcuts](https://www.icloud.com/shortcuts/59e4815dc95147709f7844ff3b6b6033): no description found

- [Supported Models](https://docs.perplexity.ai/docs/model-cards): no description found

- [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions): no description found

- [Chrome Web Store](https://chromewebstore.google.com/detail/agklnag): Add new features to your browser and personalize your browsing experience.

- [Introducing the next generation of Claude](https://www.anthropic.com/news/claude-3-family): Today, we're announcing the Claude 3 model family, which sets new industry benchmarks across a wide range of cognitive tasks. The family includes three state-of-the-art models in ascending order ...

- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/1bbdw7r/i_always_wanted_p): no description found

- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/19ccw5h/get_image_video_and_sources_from_api/): no description found

- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/1bbdw7r/i_always_wanted_perplexity_ai_on_my_apple_watch/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button): no description found

- [Civitai Beginners Guide To AI Art // #1 Core Concepts](https://www.youtube.com/watch?v=IIy3YwsXtTE&t=417s): Welcome to the Official Civitai Beginners Guide to Stable Diffusion and AI Art!In this video we will preface our upcoming series by discussing Core Concepts ...

- [Save my Chatbot - AI Conversation Exporter](https://chromewebstore.google.com/detail/agklnagmfeooogcppjccdnoallkhgkod): 🚀 Export your Phind, Perplexity and MaxAI-Google search threads into markdown files!

- [GitHub - bm777/hask: Don't switch tab or change windows anymore, just Hask.](https://github.com/bm777/hask): Don't switch tab or change windows anymore, just Hask. - bm777/hask

- [GitHub - danielmiessler/fabric: fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere.](https://github.com/danielmiessler/fabric/): fabric is an open-source framework for augmenting humans using AI. It provides a modular framework for solving specific problems using a crowdsourced set of AI prompts that can be used anywhere. - ...

- [GitHub - RMNCLDYO/Perplexity-AI-Wrapper-and-CLI: Search online (in real-time) or engage in conversational chats (similar to ChatGPT) directly from the terminal using the full suite of AI models offered by Perplexity Labs.](https://github.com/RMNCLDYO/Perplexity-AI-Wrapper-and-CLI): Search online (in real-time) or engage in conversational chats (similar to ChatGPT) directly from the terminal using the full suite of AI models offered by Perplexity Labs. - RMNCLDYO/Perplexity-AI...

---

**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1217788150567403580)** (12 messages🔥):

- **Exploring Perplexity AI Search**: A member shared their experience with the search functionality on Perplexity AI but provided a broken link: no content could be referenced due to the invalid URL ([invalid search result](https://www.perplexity.ai/search/Please-tell-us-B2zBjVFCTzGZGiRrKoGPsQ#0)).
- **Building a Perplexity-powered Firefox Extension**: Through trial and error, a member is learning to create a Firefox extension that utilizes the Perplexity API, currently a proof of concept ([initial thread on the project](https://www.perplexity.ai/search/I-would-like-8NP0s.KJRaqoDB2Ku9e2QQ)).
- **Engaging with Devin, the Autonomous AI**: A member highlighted a Perplexity AI interaction with *Devin*, labeling it as somewhat disturbing, indicating complex and potentially unsettling responses ([Devin's autonomous AI interaction](https://www.perplexity.ai/search/Devin-autonomous-AI-f_n8PlOSQoqPDI8fgKlT4w)).
- **Praise for a Perplexity AI's Response**: A member complimented a particularly effective answer provided by Perplexity AI, noting it as the "best answer yet" ([link to the response](https://www.perplexity.ai/search/what-is-a-lXiHD0PsSLuDD9YkBXt6SQ#0)).
- **Reminder on Sharing Threads**: In response to a member's post, another reminded them to ensure their thread is set to "Shared" so it can be visible to others, providing instruction on where to find more information ([instructions to share a thread](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)).

---

**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1217828689182724227)** (31 messages🔥):

- **Curiosity Around Closed Beta Citations**: A member inquired about the schema and response examples for the closed beta of URL citations; another member linked to a [documentation discussion](https://docs.perplexity.ai/discuss/65f0f6077140390018c3d9c9), sharing their insight into the variability of citation outputs depending on queries.
- **API Versus Chat Capability Concern**: A member considering Perplexity API for a new product launch expressed concerns about the differences between the API and the chat interface, seeking advice on model suitability for filtering companies based on specific criteria.
- **Real-time Data Querying with Perplexity**: Discussions revolved around the online models' ability to fetch real-time data; members mentioned sonar-small-online and sonar-medium-online as capable but with inconsistent performance, suggesting alternative APIs for specific tasks like weather information.
- **pplx-70b-online Model Status Ambiguity**: Following the discussion on model capabilities, members noted the planned deprecation of `pplx-70b-online` on March 15 yet observed ongoing distinct responses from the API, questioning the deprecation status.
- **API Inconsistency Highlighted with News Inquiry**: A member raised a discrepancy issue, presenting different responses from sonar-medium-online and the web browser version regarding up-to-date news on Donald Trump, emphasizing the varying results when prompted multiple times.

**Link mentioned**: [About "return_citations"](https://docs.perplexity.ai/discuss/65f0f6077140390018c3d9c9): no description found

---

**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1217802987796303943)** (132 messages🔥🔥):

- **Popcorn-Ready Gaming AI**: A humorous envisioning of a game-playing AI conquering Animal Crossing at a Grand Master level, serving as a light-hearted take on discussions of AI's gaming prowess.
  
- **Minibatch-Eval for Swift Generalization Feedback**: Discussion touched upon the use of minibatch-evaluation in large-scale training to provide quick generalization feedback without prolonging the evaluation phase of the training loop. It highlights the ongoing quest for efficiency in AI training methodologies.
  
- **Evaluating Human Skill Tiers in Gaming**: The conversation turned to generating a list of games with clear, publicly known human skill levels and ensuring fair AI competition by setting constraints to prevent computer cheating, such as limiting actions per minute or introducing artificial latency.
  
- **Game AI Winning and Cheating**: A discussion on game AI, particularly outlining the advancements and challenges in AlphaStar and OpenAI's Dota AI, raising questions about how such systems are not optimized for real-world use due to their heavy reliance on multiple iterations and simulations.
  
- **FPS AI Development and Challenges**: Insights were shared about the intrinsic difficulties in training AI for FPS games, such as unpredictable human strategy and game RNG, noting the lack of significant success in developing AI for battle royale games like Apex Legends.
  

**Links mentioned**:

- [Tweet from Maisa (@maisaAI_)](https://x.com/maisaAI_/status/1768657114669429103?s=20): Introducing Maisa KPU: The next leap in AI reasoning capabilities. The Knowledge Processing Unit is a Reasoning System for LLMs that leverages all their reasoning power and overcomes their intrinsic ...

- [Johnson–Lindenstrauss lemma - Wikipedia](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma): no description found

- [GitHub - MineDojo/Voyager: An Open-Ended Embodied Agent with Large Language Models](https://github.com/MineDojo/Voyager): An Open-Ended Embodied Agent with Large Language Models - MineDojo/Voyager

- [GitHub - trevorpogue/algebraic-nnhw: AI acceleration using matrix multiplication with half the multiplications](https://github.com/trevorpogue/algebraic-nnhw): AI acceleration using matrix multiplication with half the multiplications - trevorpogue/algebraic-nnhw

- [KPU - Maisa](https://maisa.ai/blog/kpu): AI-Powered Knowledge Processing Platform. A simple API for executing business tasks. Abstracting the complexities of using the latest AI architectures for software and app developers

- [I tried to make a Valorant AI using computer vision](https://youtu.be/LXA7zXVz8A4?si=m03TDOnZ10_VYD3F): I went down a rabbit-hole of trying to make a Python program that can play Valorant using computer vision and some radio shenanigans.More details, errata, et...

- [fp8 transformer engine only brings 35% speed up? · Issue #396 · NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/issues/396): Hi there, I've used Megatron to train 13B gpt model on a H100 machine. Before I use fp8 transformer engine, the speed of the training is about 0.34s/step. After I enabled the fp8 transformer engin...

- [David P. Woodruff](https://www.cs.cmu.edu/~dwoodruf/): no description found

- [Google Cloud Blog](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e)): no description found

- [Google Cloud Blog](https://cloud.google.com/blog/products/compute/the-worlds-largest-distributed-llm-training-job-on-tpu-v5e)): no description found

---

**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1217748326221352970)** (117 messages🔥🔥):

- **Debating the Efficacy of AI Detectors**: The conversation on AI content detectors questioned their reliability, suggesting that detectors could potentially mislabel content created by humans as AI-generated due to stylistic choices. The distinction between synthetic and human-generated media was noted to be challenging, with comments indicating that only documenting the creation process and chain of custody could be reliable evidence of authenticity.
  
- **Content Watermarking Discussed**: Members discussed the potential and limitations of cryptographic watermarking for AI outputs. There was skepticism concerning watermark efficiency due to the ease of un-watermarking using other models and the implications for the utility of watermarked models.
  
- **New Advances in AI Reasoning**: Discussion about recent research advancements included reference to a new technique called *Quiet-STaR*, intending to improve language models by teaching them to "think ahead" before emitting tokens.
  
- **Contours of GPT-turbo Explored**: Dialogue analyzed a paper investigating the commercialization of large language models and API-level access, revealing that valuable information about proprietary models can be extracted using API queries. They notably estimated the hidden size of OpenAI's GPT-3.5-turbo model.
  
- **Discourse on Tokenizing Numbers in LLMs**: The effect of tokenizing numbers left-to-right versus right-to-left was mooted, with observations suggesting that the method could influence a model's arithmetic capabilities. Conversations touched on the possibility of exploiting tokenizing strategies to enhance model performance.
  

**Links mentioned**:

- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629): When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is imp...

- [Logits of API-Protected LLMs Leak Proprietary Information](https://arxiv.org/abs/2403.09539): The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption...

- [Tweet from Aaditya Singh (@Aaditya6284)](https://x.com/Aaditya6284/status/1762558439354409345): We study the effect of this choice in GPT-3.5 and GPT-4 – specifically, we look at the effect of tokenizing left-to-right (L2R) vs right-to-left (R2L), enforced by using delimiters such as commas. We ...

- [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380): Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance o...

- [Are GFlowNets the future of AI?](https://youtu.be/o0Ju9NQa5Ko?si=U3gIepQF51oASSgY): Should you care about GFlowNets? What are they anyway? Learn about how GFlowNets are aiding drug discovery and reasoning in large language models!\*\*Like, sub...

- [GitHub - bigscience-workshop/bloom-dechonk: A repo for running model shrinking experiments](https://github.com/bigscience-workshop/bloom-dechonk): A repo for running model shrinking experiments. Contribute to bigscience-workshop/bloom-dechonk development by creating an account on GitHub.

- [Model & API Providers Analysis | Artificial Analysis](https://artificialanalysis.ai/): Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key metrics including quality, price, performance and speed (throughput & latency).

---

**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 messages):

kerls: are there any resources on scaling laws for video generation models?

---

**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1217800605603463190)** (32 messages🔥):

- **Innovative Interpretability Technique Explored**: The concept of *latent decoding by vector-db-lookup* was explored, using embeddings from words in different languages to build a vector database for analyzing models. This method aims to facilitate understanding intermediate representations at each layer.
  
- **Initial Results with Latent Decoding**: A preliminary result was shared, involving constructing a *vector database* using embeddings of French, English, and German words from Llama2. The technique provided intermediate full-word decodings at each layer, offering potential as an interpretability tool.
  
- **Language Influence in Concept Space**: A discussion unfolded around how language models may predict using biases in their concept space, potentially weighted by their training data. Experiments with bilingual models like CroissantLLM indicated that tokenizers and the proportion of training data in various languages may impact these biases.
  
- **Sampling Text from Prespecified Gram Statistics**: The topic of generating text samples from a distribution specified by n-gram statistics was broached. It was explained that this could be done autoregressively to match the max entropy distribution.
  
- **Bigram Language Model Implementation Referenced**: The conversation mentioned an implementation of generating text using a bigram model, indicating this as a practical way to sample strings while adhering to specified grammatical statistics. The implementation is available on GitHub at [features-across-time/scripts](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py).
  

**Links mentioned**:

- [Word n-gram language model - Wikipedia](https://en.wikipedia.org/wiki/Word_n-gram_language_model): no description found

- [features-across-time/scripts/generate_bigrams.py at main · EleutherAI/features-across-time](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py): Understanding how features learned by neural networks evolve throughout training - EleutherAI/features-across-time

- [llm-latent-language/nnsight.ipynb at main · Butanium/llm-latent-language](https://github.com/Butanium/llm-latent-language/blob/main/nnsight.ipynb): Repo accompanying our paper "Do Llamas Work in English? On the Latent Language of Multilingual Transformers". - Butanium/llm-latent-language

---

**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1217817737951187025)** (5 messages):

- **Challenges with Verbose LLM Answer Extraction**: Task adaptation for LLM evaluation is affected by the verbosity of newer models, making answer extraction difficult without llm-as-a-judge. Some tasks have both loglikelihood and generative or CoT variants available, such as those found in [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gpqa).
  
- **Skepticism About Vector Space Models**: A member expressed doubts about the vector space model representing meaning in language, citing GPT-J's ungrammatical outputs as an example. They argue that the apparent grammatical competence of larger models is merely due to scale rather than any genuine understanding or reasoning ability.
  
- **Seeking Guidance with lm-eval-harness**: A newcomer to lm-eval-harness queries about integrating custom LLM models like llama on gaudi2, seeking examples or demonstrations on how to implement necessary functions such as `generate_until` and `log_likelihood`. There's also confusion regarding the inheritance of unspecified functions and the absence of a fixed format for command-line tool arguments.
  

**Links mentioned**:

- [lm-evaluation-harness/lm_eval/tasks/gpqa at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gpqa): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

- [GitHub: Let’s build from here](https://github.co): GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...

---

**Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages):

boneamputee: [https://brianfitzgerald.xyz/prompt-augmentation/](https://brianfitzgerald.xyz/prompt-augmentation/)

---

**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1217950961612619907)** (1 messages):

- **Interactive LLM Leaderboard Visualization**: The **Open LLM Leaderboard Visualization** has been updated to allow users to reorder metrics and compare up to three models visually. Visit the interactive space at [open-llm-leaderboard-viz](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz).
- **Visual Storytelling with Kosmos-2**: Explore the space **Kosmos-2** for GPT-based visual storytelling, available at [Kosmos-2 Space](https://huggingface.co/spaces/Tonic1/kosmos-2).
- **ARC-Challenge Dataset Enhanced with Reasoning**: Check out the **Augmented ARC-Challenge Dataset** that includes Chain-of-Thought reasoning, accessible at [arc-cot Dataset](https://huggingface.co/datasets/Locutusque/arc-cot).
- **Aya 101 - The Polyglot Model**: Discover **Aya 101**, a model proficient in 101 languages. More information can be found in Tonic's space at [Aya 101](https://huggingface.co/spaces/Tonic/Aya).
- **New Capabilities in Data Embedding**: Review the **BEE-spoke-data model** for embedding with up to a 4k context, ideal for tasks like clustering or semantic search. Access the model and details at [bert-plus-L8-v1.0-syntheticSTS-4k](https://huggingface.co/BEE-spoke-data/bert-plus-L8-v1.0-syntheticSTS-4k).

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><li><a href="https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz">Open Llm Leaderboard Viz - a Hugging Face Space by dimbyTa</a>: no description found</li><li><a href="https://huggingface.co/spaces/Tonic1/kosmos-2">Kosmos 2 - a Hugging Face Space by Tonic1</a>: no description found</li><li><a href="https://huggingface.co/datasets/Locutusque/arc-cot">Locutusque/arc-cot · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/Tonic/Aya">Aya - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://github.com/alvarobartt/vertex-ai-huggingface-inference-toolkit">GitHub - alvarobartt/vertex-ai-huggingface-inference-toolkit: 🤗 HuggingFace Inference Toolkit for Google Cloud Vertex AI (similar to SageMaker's Inference Toolkit, but for Vertex AI and unofficial)</a>: 🤗 HuggingFace Inference Toolkit for Google Cloud Vertex AI (similar to SageMaker's Inference Toolkit, but for Vertex AI and unofficial) - alvarobartt/vertex-ai-huggingface-inference-toolkit</li><li><a href="https://huggingface.co/BEE-spoke-data/bert-plus-L8-v1.0-syntheticSTS-4k">BEE-spoke-data/bert-plus-L8-v1.0-syntheticSTS-4k · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/dominguesm/mambarim-110m">dominguesm/mambarim-110m · Hugging Face</a>: no description found</li><li><a href="https://link.springer.com/article/10.1007/s10586-023-04089-5">Machine learning-based intrusion detection: feature selection versus feature extraction - Cluster Computing</a>: Internet of Things (IoTs) has been playing an important role in many sectors, such as smart cities, smart agriculture, smart healthcare, and smart manufacturing. However, IoT devices are highly vulner...</li><li><a href="https://github.com/rbourgeat/refacto">GitHub - rbourgeat/refacto: Refactor your code with local LLM</a>: Refactor your code with local LLM. Contribute to rbourgeat/refacto development by creating an account on GitHub.</li><li><a href="https://huggingface.co/posts/DmitryRyumin/888482747169050">@DmitryRyumin on Hugging Face: "🚀🎭🌟 New Research Alert! 🌟🎭 🚀 📄 Title: VLOGGER: Multimodal Diffusion for…"</a>: no description found</li></div>

---

**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1217746426948354168)** (115 messages🔥🔥):

- **Excitement for Consumer-Grade AI**: Members expressed excitement about the **quantized versions of new models** being compatible with consumer grade GPUs. One discussed the rapid progress from a window context size of less than 2k to 1 million Lightweight Mixture of Experts (LWMs).
  
- **Seeking Stable Diffusion Space**: A user inquired about a channel for **stable diffusion** discussion, and was directed to a broader space that isn't specifically related to stability.
  
- **Knowledge Implementation in Pretrained Models**: One user shared their experience of successfully implementing **RAG**, yet faced challenges with **LoRa** in a pretrained model like *Mistral 7B*. There was a consideration to optimize their dataset generation process to improve the model's responses.
  
- **Autonomous Agents and Local LLMs**: A question was raised about whether there is an **autonomous agent that works with local Large Language Models (LLMs), completely offline**. Suggested solutions included tools like *ollama* and *jan* for terminal-based interfaces.
  
- **NVIDIA Grace Hopper Superchip Discussion**: There was a buzz about the NVIDIA Grace Hopper Superchip, its computing power, and its potential for AI and data center applications. The conversation delved into technical specifications and availability, including a member who was interested in whether the chip could support gaming at high resolutions.
  

**Links mentioned**:

- [Tweet from NVIDIA GH200 CPU Performance Benchmarks Against AMD EPYC Zen 4 & Intel Xeon Emerald Rapids Review - Phoronix](https://www.phoronix.com/review/nvidia-gh200-gptshop-benchmark): no description found

- [Tweet from Linux Performance, Benchmarks & Open-Source News - Phoronix](https://www.phoronix.com/review/nvidia-gh200-gptshop-ben): no description found

- [NVIDIA Grace Hopper and Grace Superchip Pictured and Incompatible](https://www.servethehome.com/nvidia-grace-hopper-gh200-and-grace-superchip-arm-pictured-and-incompatible/): We show why the NVIDIA Grace Hopper GH200 and Grace Superchip are not compatible with the same servers by showing them side-by-side

- [Getting started | node-llama-cpp](https://withcatai.github.io/node-llama-cpp/guide/): no description found

- [bishmoy/Arxiv-CS-RAG at main](https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG/tree/main): no description found

- [Tonic/Aya · Set a repetition_penalty constant as 1.8](https://huggingface.co/spaces/Tonic/Aya/discussions/3): no description found

- [Train with a script](https://huggingface.co/docs/transformers/run_scripts#run-a-script): no description found

- [results_2M_val.csv download is closed，how to get it · Issue #21 · m-bain/webvid](https://github.com/m-bain/webvid/issues/21): (base) [wangxi@v100-4 webvid]$ wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_val.csv --2024-02-27 10:49:36-- http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_val.csv Resolving...

---

**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1218079215845511220)** (4 messages):

- **Magnificent App for Command Corrections**: A member shared a GitHub link for **[thefuck](https://github.com/nvbn/thefuck)**, an application that corrects your previous console command. The app's description on GitHub refers to it as a "magnificent app which corrects your previous console command."
  
- **Optimization Confusion**: A question was raised about various optimization methods, pointing out **GridSearch Optimization**, **RandomSearch Optimization**, and expressing confusion specifically about **Bayesian Optimization**.
  
- **Seeking Guidance with Hugging Face**: A new member asked for help understanding how to use **Hugging Face** and what exactly it is. They requested assistance in the **#898619964095860757** channel.
  
- **AI Duets Pose a Challenge**: A member new to AI music shared issues with creating appealing AI covers of duets and bands, mentioning that while single voice covers are manageable, duets or group songs sound like they're "being strangled". They are curious about how others achieve better results with such AI covers.
  

**Link mentioned**: [GitHub - nvbn/thefuck: Magnificent app which corrects your previous console command.](https://github.com/nvbn/thefuck): Magnificent app which corrects your previous console command. - nvbn/thefuck

---

**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1217795121014247454)** (6 messages):

- **Sketching AI with Pseudocode**: An article discussed the benefits of using pseudocode for prompting LLMs, noting significant improvements with **GPT-4** over previous versions. Readers can delve into the specifics at [SudoLang: A pseudocode programming language](https://medium.com/javascript-scene/sudolang-a-powerful-pseudocode-programming-language-for-llms-d64d42aa719b).
  
- **AI Meets Business with SAP HANA and LangChain**: An article on ai.gopubby.com highlights the integration of **SAP HANA Vector Engine** with **LangChain** to enhance AI applications. The advancements are detailed at [Unlocking the Future of AI Applications](https://ai.gopubby.com/unlocking-the-future-of-ai-applications-with-hana-vector-engine-and-langchain-14cd6c66219d).
  
- **Introducing Mamba-Chat**: GitHub hosts a novel chatbot named **Mamba-Chat**, which utilizes the **state-space model architecture**. Developers and enthusiasts can explore or contribute to the project at [Mamba-Chat on GitHub](https://github.com/havenhq/mamba-chat).
  
- **Vision-Language-Action Model for Robotics**: DeepMind introduced a vision-language-action model called **Robotic Transformer 2 (RT-2)**, intending to empower robots with generalized control instructions. More on this can be found in their blog post and [paper](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/).
  
- **Kyle: Unity-Based Ragdoll Training**: Hugging Face introduces **Kyle**, an advanced active ragdoll training environment for Unity. It features optimized codebase and advanced vision capabilities with LSTM networks, and interested users can find out more at the [Hugging Face model page](https://huggingface.co/p3nGu1nZz/Kyle-b0a).
  

**Links mentioned**:

- [p3nGu1nZz/Kyle-b0a · Hugging Face](https://huggingface.co/p3nGu1nZz/Kyle-b0a): no description found

- [RT-2: New model translates vision and language into action](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/): Introducing Robotic Transformer 2 (RT-2), a novel vision-language-action (VLA) model that learns from both web and robotics data, and translates this knowledge into generalised instructions for...

- [GitHub - havenhq/mamba-chat: Mamba-Chat: A chat LLM based on the state-space model architecture 🐍](https://github.com/havenhq/mamba-chat): Mamba-Chat: A chat LLM based on the state-space model architecture 🐍 - havenhq/mamba-chat

- [SudoLang: A Powerful Pseudocode Programming Language for LLMs](https://medium.com/javascript-scene/sudolang-a-powerful-pseudocode-programming-language-for-llms-d64d42aa719b): Pseudocode is a fantastic way to sketch programs using informal, natural language, without worrying about specific syntax. It’s like…

---

**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1217788334106214430)** (9 messages🔥):

- **Quest for the Best Open LLM**: A member acknowledges the **SF-Foundation/Ein-72B-v0.11** as the most promising open LLM based on an Open LLM Leaderboard, with an almost 80% success rate across metrics. A link to the leaderboard or visualizations was not provided.
- **VS Code Refactoring Made Easy with Plugin**: A member released a [simple plugin for VS Code](https://github.com/rbourgeat/refacto) named *Refacto*. It allows code refactoring using a local LLM with a llama CPP server and contributions are welcomed.
- **Introducing Cobalt**: Cobalt is a privacy-focused front end [repository for LLMs on GitHub](https://github.com/taylorgoolsby/cobalt), featuring context management and memory summarization which is in development for iOS.
- **Transformers for PHP Developers**: A project called *Transformers PHP* was showcased, which aims to [enable PHP developers](https://github.com/CodeWithKyrian/transformers-php) to integrate machine learning features into their projects easily.
- **Exploring Open Records Law with AI**: [KY OpenGov](https://kyopengov.org/blog/exploring-open-records-law-ai) is experimenting with AI technologies that could potentially help navigate open records laws, aiming for government transparency and ease of public access to information.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><li><a href="https://kyopengov.org/blog/exploring-open-records-law-ai">Exploring Open Records Law with AI | KOGC</a>: no description found</li><li><a href="https://github.com/taylorgoolsby/cobalt">GitHub - taylorgoolsby/cobalt</a>: Contribute to taylorgoolsby/cobalt development by creating an account on GitHub.</li><li><a href="https://github.com/CodeWithKyrian/transformers-php">GitHub - CodeWithKyrian/transformers-php: Transformers PHP is a toolkit for PHP developers to add machine learning magic to their projects easily.</a>: Transformers PHP is a toolkit for PHP developers to add machine learning magic to their projects easily. - GitHub - CodeWithKyrian/transformers-php: Transformers PHP is a toolkit for PHP developer...</li><li><a href="https://github.com/rbourgeat/refacto">GitHub - rbourgeat/refacto: Refactor your code with local LLM</a>: Refactor your code with local LLM. Contribute to rbourgeat/refacto development by creating an account on GitHub.</li></div>

---

**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1217806245684314214)** (11 messages🔥):

- **An Ounce of Prompting Strategy**: A user discussed the difficulty in using *crewai* apps and attributed it to a lack of prompting skills, particularly with incorporating imports.
- **Reading Group Hiatus Announcement**: A brief announcement clarified that there will be no presentation this week in the reading-group channel, with plans for the next session scheduled for the following week.
- **Neural Network Units Inquiry**: A question was raised about the number of neural network units required for the MNIST digit classification task discussed in Andrew Ng's course, leading to clarifications on the distinction between input units and hidden units.
- **Foundations of Layer and Neuron Numbers**: In response to an inquiry about determining the number of neurons and hidden layers in neural networks, users noted that these decisions are based on experimentation and previous successful models rather than a standard formula, weighing the trade-offs between processing power, speed, and accuracy.
- **New Perspectives on Multilingual Models**: A user shared a link to a paper suggesting that multilingual language models might use English as an internal pivot language, and the implications for understanding how these models function and their linguistic bias. Curiosity was expressed about the impact of byte-level encoding on this behavior. The paper can be found at [this link](https://arxiv.org/abs/2402.10588), and it was added to a multilingual paper collection on HuggingFace, available [here](https://huggingface.co/collections/stereoplegic/multilingual-65389b21be39573b3b2db98d).
  

**Links mentioned**:

- [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588): We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...

- [Bytez: Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://bytez.com/read/arxiv/2402.10588): In this research study, scientists wanted to know if language models (that can generate text) use English as a "pivot" language internally, even when prompted in other languages. They found ...

- [Multilingual - a stereoplegic Collection](https://huggingface.co/collections/stereoplegic/multilingual-65389b21be39573b3b2db98d): no description found

---

**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1217875633779249192)** (1 messages):

- **Diffusers Library Update Alert**: The new **Diffusers 0.27.0** version has been released. Check out the [release notes here](https://github.com/huggingface/diffusers/releases/tag/v0.27.0).

---

**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1218059345775034429)** (8 messages🔥):

- **Forum Misdirection**: A reminder was issued that discussions should be pertinent to `diffusers` and diffusion models, suggesting that off-topic inquiries be directed to an appropriate forum.
- **Kohya's High-Resolution Trick**: A neat trick discovered by a user named Kohya was shared, involving a high-resolution fix for diffusers, with an accompanying [GitHub issue](https://github.com/huggingface/diffusers/issues/7265) and a link to a YouTube video demonstrating the enhancement.
- **Call for Collaborative Issue Investigation**: In response to a concern, there was an open invitation to submit an issue with reproducible code on GitHub for collaborative examination, with a specific prompt to tag `sayakpaul`.
- **Guidance on Appropriate Forum Usage**: Repeated reminders were given to keep discussions focused on diffusion models and diffusers, reinforcing the purpose of the forum.
- **Clarification on Merging in Context to Diffusers**: A question was raised about whether 'merging' referred to combining model parameters or packaging checkpoints for specific diffusion model components.

**Link mentioned**: [Kohya Hires fix · Issue #7265 · huggingface/diffusers](https://github.com/huggingface/diffusers/issues/7265): is diffusers possible to support this hires fix? it looks 1.5 work too AUTOMATIC1111/stable-diffusion-webui#13974 [https://www.youtube.com/watch?v=SbgMwHDXthU](https://www.youtube.com/watch?v=SbgMwHDXthU) same seed at 1024x1024 with without Thi...

---

**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1217778348000935946)** (8 messages🔥):

- **Exploring "Learn without forgetting"**: A member mentioned the method called **Learn without forgetting (LwF)**, suggesting a possible area of interest in the machine learning field.
  
- **Interest Piqued by Arcface for Multiclass Classification**: One user expressed curiosity about using **Arcface** as a substitute for Softmax in regular multiclass classification, noting its effectiveness in combo loss scenarios and for embedding extraction.
  
- **Guided Backpropagation Query**: A member sought assistance with implementing **guided backpropagation** in a recent version of PyTorch, facing issues with computing a backwards pass related to the model output tensor.
  
- **NVIDIA Grace Hopper Superchip Unveiled**: An announcement was shared about the **NVIDIA Grace Hopper Superchip**, highlighting its potential for high-performance computing (HPC), artificial intelligence (AI), and data center applications.
  

---

**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1217762732602822759)** (9 messages🔥):

- **Matrix Approximation Milestone**: A member expressed excitement about achieving a **0.016 relative error** in Frobenius norm on a `4096 x 4096` matrix approximation while conserving memory. Anticipating results for larger matrices (`4096 x 14336`), this could signal a breakthrough in matrix optimization tasks.
  
- **Training Mystery: Low Loss but Nonsense Output**: One user reported a perplexing issue where a modified pretrained model showed good convergence during training (loss decreasing to **[0.6,0.8]**) but produced nonsensical outputs during testing. This was despite using a similar loss calculation approach as in **Mistral**.
  
- **In Search of Mathematical Theorem Naming Rights**: A discussion on matrix decomposition bounds led to an admission that literature lacks a name for a specific bound, jokingly suggesting the possibility of naming it after oneself.
  
- **Improving NL2SQL Pipeline Accuracy**: A member detailed their **NL2SQL pipeline** which includes **BAAI/llm-embedder**, **TheBloke/nsql-llama-2-7B-GGUF**, and **FAISS** for embedding SQL schemas and generating queries. They sought recommendations to boost pipeline accuracy due to inconsistent results.
  
- **Introducing the NVIDIA Grace Hopper Superchip**: A user announced the **NVIDIA Grace Hopper Superchip**, highlighting its potential impact on computing power and efficiency for AI and high-performance computing applications.
  

---

**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1218059345775034429)** (8 messages🔥):

- **Misplaced Conversation Alert**: A member nudged another to use a more appropriate forum for topics unrelated to `diffusers`, suggesting they seek help through certain tagged individuals.
- **Tech Tip Share for high-resolution images**: A discovery was shared involving a "hires fix" for `diffusers`, with a [GitHub issue link](https://github.com/huggingface/diffusers/issues/7265) and a YouTube video demonstrating the same seed at different resolutions.
- **Invitation to Open Issues on GitHub**: You're encouraged to raise concerns with reproducible code on GitHub for the attention of member `sayakpaul`, indicating readiness to tackle the problems.
- **Reiteration to Stay On-Topic**: Multiple reminders were given to keep discussions focused on diffusion models and `diffusers`.
- **Request for Clarification on "Merging"**: In response to a merging question, clarification was requested on whether it pertained to merging model parameters or packaging a checkpoint with various model components.

**Link mentioned**: [Kohya Hires fix · Issue #7265 · huggingface/diffusers](https://github.com/huggingface/diffusers/issues/7265): is diffusers possible to support this hires fix? it looks 1.5 work too AUTOMATIC1111/stable-diffusion-webui#13974 [https://www.youtube.com/watch?v=SbgMwHDXthU](https://www.youtube.com/watch?v=SbgMwHDXthU) same seed at 1024x1024 with without Thi...

---

**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1217862823665995867)** (4 messages):

- **Challenges Parsing Financial PowerPoints for RAG**: RAG struggles with parsing finance .pptx files due to nonstandard formats involving text, tables, images, and charts. The team is looking into a **proper parsing solution** and discussed it in [this tweet](https://twitter.com/llama_index/status/1768303288381030408).
  
- **RAG Needs Better Latex Math Equations Handling**: To accurately represent math and ML papers in RAG, it's necessary to extract math equations correctly rather than default ASCII text extraction. A possible **solution involves parsing by prompting**, as shared in [this tweet](https://twitter.com/llama_index/status/1768443551267049492).
  
- **Evolving RAG Pipeline to Handle Complex Queries**: For handling complex queries in the RAG pipeline, treat each document not just as text but as a tool for interaction. Doing so could allow for **more complex interactions** with larger documents, according to [this tweet](https://twitter.com/llama_index/status/1768658182308794421).
  
- **Launch of LlamaIndex v0.10.20 with Instrumentation Module**: The new LlamaIndex release includes an **Instrumentation module**, enhancing observability. They've shared notebooks that demonstrate its capabilities, discussed in [this tweet](https://twitter.com/llama_index/status/1768730443921396220).
  

**Links mentioned**:

- [llama_index/docs/examples/instrumentation/basic_usage.ipynb at main · run-llama/llama_index](https://t.co/GY4unUYOwl): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index

- [llama_index/docs/examples/instrumentation/observe_api_calls.ipynb at main · run-llama/llama_index](https://t.co/E1d9dtkqAI): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index

---

**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1217799370049716254)** (132 messages🔥🔥):

- **Integration Dilemmas**: Questions arise on integrating various components like VectorStore (Milvus) into a document management pipeline for production scenarios. Discussions pivoted around leveraging remote docstores (like Redis, MongoDB, Firestore, PostgreSQL) and utilizing an ingestion pipeline for upserts instead of managing persistent `docstore.json` files on disk. An example ingestion pipeline is shared using Python code.
  
- **Caching and Pipeline Queries**: Members sought clarity on implementing cache systems like langchain llm cache and discussed integrating elements like `node_postprocessor` into a `RetrieverQueryEngine`. LlamaIndex doesn't appear to involve caching in the information provided; however, Python code examples were shared to illustrate usage of `node_postprocessors`.
  
- **Document Parsing Errors and Solutions**: Some members encountered issues like a memory error from parsing a large markdown document and a `ParserError` from the `MarkdownElementNodeParser`. Proposed solutions include splitting the document into smaller chunks using the `SentenceSplitter` or handling the operations through an `IngestionPipeline`.
  
- **Query Engine Challenges**: Users faced multiple difficulties around specifying arguments for `PandasQueryEngine` and its functionality with date and location extracts, as well as defining prompts to guide `QueryEngineTool`. One solution proposed involves using a `query_engine_tools` array with a modified prompt.
  
- **BM25 Embeddings and Query Engine Configuration**: Queries around setting BM25 as an embedding model resembling `HuggingFaceEmbedding` were made without clear solutions in the provided documentation. Steps to include `node_postprocessors` with a `RetrieverQueryEngine` and a `rerank_query_engine` were explored.
  

**Links mentioned**:

- [\>no title found](http://127.0.0.1:9997): no description found

- [)>no title found](http://localhost:{port}"): no description found

- [Caching | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/model_io/llms/llm_caching): LangChain provides an optional caching layer for LLMs. This is useful

- [Ingestion Pipeline - LlamaIndex 🦙 v0.10.20.post1](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html#caching"Not): no description found

- [Ingestion Pipeline + Document Management - LlamaIndex 🦙 v0.10.20.post1](https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline.html): no description found

- [Tools - LlamaIndex 🦙 v0.10.20.post1](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/root.html): no description found

- [llama_index/llama-index-core/llama_index/core/retrievers/fusion_retriever.py at ca9634e660b91799a86ee9f9f0a697eb236bcefd · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/ca9634e660b91799a86ee9f9f0a697eb236bcefd/llama-index-core/llama_index/core/retrievers/fusion_retriever.py#L83): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index

- [\>no title found](http://localhost:{port}",): no description found

---

**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1217831686222647438)** (61 messages🔥🔥):

- **Potential OpenAI Security Breach Discussed**: A Post Mortem on a security issue that occurred Tuesday at OpenAI was shared, detailing how requests might have been made on behalf of another account. The documentation is provided on [GitHub Gist](https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca).
- **State of Sparse Universal Transformers**: Sharing insight into weight sharing for Sparse Universal Transformers: they needed a fast way to do Mixture-of-Experts for attention, which led to the creation of ScatterMoE. The discussion links to details on [The New XOR Problem](http://blog.wtf.sg/posts/2023-02-03-the-new-xor-problem/).
- **The AI Development Platform with Affordable Pricing**: The Deci AI Nano model and an associated AI development platform were launched, priced at $0.1 per 1M tokens. The announcement includes links to a marketing blog for [Deci AI](https://deci.ai/blog/deci-nano-and-gen-ai-development-platform/), as well as two technical tutorials on Google Colab ([Basic Usage](https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing), [LangChain Usage](https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing)).
- **Prompt Augmentation to Enhance Creative AI**: A discussion on prompt augmenters noted a tendency for such tools to gain traction, linking to an article that details how a 77M T5 model was trained to expand prompts, outperforming 1B+ parameter LLMs in quality and prompt alignment. The full discussion and resources approachable at [Prompt Augmentation](https://brianfitzgerald.xyz/prompt-augmentation/).
- **AMD's Ray Tracing Move to Open Source**: AMD steps further into open source by making their HIP-Ray Tracing RT code accessible, which sparks discussions about the evolving open-source ecosystem. The news is summarized in a [Phoronix article](https://www.phoronix.com/news/AMD-HIP-Ray-Tracing-RT-Open).

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><li><a href="https://www.phoronix.com/news/AMD-HIP-Ray-Tracing-RT-Open">Tweet from AMD Makes HIP Ray-Tracing Open-Source - Phoronix</a>: no description found</li><li><a href="https://x.com/altryne/status/1768683178888208816?s=46&amp;t=90xQ8sGy63D2OtiaoGJuww">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: Sora team showing up at Berkley to talk about SORA</li><li><a href="https://x.com/emmanuel_2m/status/1768360522028876045?s=46&amp;t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Emm (@emmanuel_2m)</a>: 🚨 Today, we're excited to launch the Scenario #UPSCALER! Elevate your AI creations up to 10k resolution. 🚀 Built for unmatched #CreativeControl &amp; guided workflows. 💰 It starts at just $15/mo ...</li><li><a href="https://brianfitzgerald.xyz/prompt-augmentation/">SuperPrompt - Better SDXL prompts in 77M Parameters | Brian Fitzgerald</a>: Left SDXL output with SuperPrompt applied to the same input prompt.</li><li><a href="https://huyenchip.com/">Chip Huyen</a>: I help companies deploy machine learning into production. I write about AI applications, tooling, and best practices.</li><li><a href="https://huyenchip.com/2024/03/14/ai-oss.html">What I learned from looking at 900 most popular open source AI tools</a>: Four years ago, I did an analysis of the open source ML ecosystem. Since then, the landscape has changed, so I revisited the topic. This time, I focused exclusively on the stack around foundation mode...</li><li><a href="https://gist.github.com/henriqueln7/e572fde4bd3601766e260ea82fc964ca">I'm concerned I made requests to openAI on behalf of another account - and perhaps someone did so on my behalf</a>: I'm concerned I made requests to openAI on behalf of another account - and perhaps someone did so on my behalf - openai-possible-security-breach.md</li><li><a href="https://x.com/teortaxestex/status/1768261124187672972?s=46&amp;t=90xQ8sGy63D2OtiaoGJuww">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: Read this if you haven't yet: http://blog.wtf.sg/posts/2023-02-03-the-new-xor-problem/ ↘️ Quoting Shawn Tan (@tanshawn) One of the things we really needed for Sparse Universal Transformers was ...</li><li><a href="https://x.com/teknium1/status/1768452864995942469?s=46&amp;t=6FDP">Tweet from Teknium (e/λ) (@Teknium1)</a>: This explains why Yann is so bearish on LLMs... 😲</li><li><a href="https://x.com/chipro/status/1768388213008445837?s=20">Tweet from Chip Huyen (@chipro)</a>: I went through the most popular AI repos on GitHub, categorized them, and studied their growth trajectories. Here are some of the learnings: 1. There are 845 generative AI repos with at least 500 sta...</li><li><a href="https://x.com/teknium1/status/1768452864995942469?s=46&amp;t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Teknium (e/λ) (@Teknium1)</a>: This explains why Yann is so bearish on LLMs... 😲</li><li><a href="https://x.com/granawkins/status/1768530196557365599?s=46&amp;t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Grant♟️ (@granawkins)</a>: "Between Q1-24 and Q4-25, there will be a 14x increase in compute. Then, if you factor in algorithmic efficiency doubling every 9 months, the effective compute at the end of next year will be alm...</li><li><a href="https://x.com/kk_slider_k_/status/1768464173657158132?s=46&amp;t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from K (@kk_slider_k_)</a>: This makes so much sense. Yann’s always been looking for models that reason visually or using planning rather than purely in language ↘️ Quoting Teknium (e/λ) (@Teknium1) This explains why Yann is ...</li><li><a href="https://x.com/altryne/status/1768024635818340662?s=46&amp;t=90xQ8sGy63D2OtiaoGJuww">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: Tomorrow (March 14) is: &gt; π day &gt; GPT-4 anniversary &gt; Claude 1 anniversary but also 🥁🥁🥁🥁 ThursdAI spaces 1st birthday 🎉 Join us as we chat about Claude Haiku, Devin, Figure+OpenAI, T...</li><li><a href="https://x.com/joshwalkos/status/1767745681375015076?s=46&amp;t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Champagne Joshi (@JoshWalkos)</a>: This is a fascinating conversation with a girl who lacks an internal monologue. She articulates the experience quite well.</li><li><a href="https://deci.ai/blog/deci-nano-and-gen-ai-development-platform/">Introducing Deci’s Gen AI Development Platform and Deci-Nano</a>: Explore Deci’s Gen AI Development platform and the Deci Nano LLM, designed to offer efficiency, performance, and flexible deployment options</li><li><a href="https://colab.research.google.com/drive/1JW8t-kosLEgYVxXadwwDMypnQ5c_UD2u?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1PMwMovV-ji1mp0yl0qYDTI-gdG6SjOnZ?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://www.nvidia.com/gtc/?ncid=ref-inor-332714">GTC 2024: #1 AI Conference</a>: Register now. Streamed online. March 18-21, 2024.</li><li><a href="https://docs.google.com/document/d/1HZ326V6KNK4QIlG7uEldQEizFgTaO7Hg9uJxURYy9f8/edit">NVIDIA &amp; Harpreet Sahota GTC 2024</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...</li><li><a href="https://bytez.com/read/arxiv/2402.10588">Bytez: Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: In this research study, scientists wanted to know if language models (that can generate text) use English as a "pivot" language internally, even when prompted in other languages. They found ...</li><li><a href="https://huggingface.co/collections/stereoplegic/multilingual-65389b21be39573b3b2db98d">Multilingual - a stereoplegic Collection</a>: no description found</li></div>

---

**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1217885999246348309)** (5 messages):

- **Tuning in to Transformers**: A new episode featuring an interview with Mikey Shulman from Suno AI, discussing music generation using transformers, is now live. Watch it on [YouTube](https://youtu.be/gYXjn-V7AEw) with a title "Making Transformers Sing".
- **Paper Club Gathering Alert**: The Paper Club is currently reviewing the "A Comprehensive Summary Of Large Language Models" paper. Members are encouraged to join the discussion in the dedicated channel.

**Link mentioned**: [Making Transformers Sing - with Mikey Shulman of Suno](https://youtu.be/gYXjn-V7AEw): Giving computers a voice has always been at the center of sci-fi movies; “I’m sorry Dave, I’m afraid I can’t do that” wouldn’t hit as hard if it just appeare...

---

**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1217857312140820561)** (24 messages🔥):

- **Curiosity About Supervised Fine-Tuning (SFT)**: A member mentioned interest in finding a way to Supervised Fine-Tune (SFT) on **negative pairs**, especially since they have a lot of them.
  
- **Decoding the Rationale Behind Attention**: Discussions highlighted that the attention mechanism in neural networks like transformers was created to address limitations in older models with fixed-length context windows and to support models with the ability to focus on relevant parts of the input sequence.
  
- **Untangling the Concept of Parallelization**: Clarifications were made regarding parallelization in transformer models, explaining that it allows for the independent processing of different tokens through the scaled dot product operation, which in turn speeds up training.
  
- **Understanding Transformer Motivations**: A member expressed the importance of grasping the intuition behind design choices in transformer models, and received elucidation on the historical limitations of earlier models which transformers aimed to solve.
  
- **Appreciation for Learning Experience**: Participants expressed gratitude for the session which provided more insight into the progression and advancements of Large Language Models (LLMs).
  

---

**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1218287754715201638)** (36 messages🔥):

- **Passive Participation in IRL Meetings**: A member mentioned being in an **IRL meeting** and could only passively participate in today's Discord chat.
- **Anticipation of In-Depth Content**: Two members hint at upcoming in-depth versions of their discussions to be posted on their respective blogs.
- **Nuisance of Web Interfaces for RAG**: A user reported issues when using the web interface for **RAG (Retrieval-Augmented Generation)** systems, suggesting the app might be a better option for stability.
- **Sharing Useful Resources on RAG**: A member shared a [link](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4) to a medium post about advanced RAG techniques that could improve the retrieval and generation of high-quality responses.
- **Resource Compilation Document Shared**: A comprehensive Google Sheets document was linked which compiles resources on topics such as UI/UX patterns for GenAI and RAG architectures, indicating previous discussions and facilitators on the subject matter.
  

**Links mentioned**:

- [Advanced RAG 01: Small-to-Big Retrieval](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4): Child-Parent RecursiveRetriever and Sentence Window Retrieval with LlamaIndex

- [AI In Action: Weekly Jam Sessions](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0): 2024 Topic,Date,Facilitator,Resources UI/UX patterns for GenAI,1/26/2024,nuvic,<a href="https://maggieappleton.com/squish-structure">https://maggieappleton.com/squish-structure</a&...

---

**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1217799062070366300)** (60 messages🔥🔥):

- **Microsoft Employee Fixes Typo After Community Ping**: A user flagged a typo in a Microsoft service, prompting action from the Bing VP and resulting in a fix. The user noted the VP acknowledged the mistake as a typo.
  
- **Stumped by Repeated Morphemes**: Members discuss the challenge of getting **GPT-3.5** to generate examples of repeated morphemes in compound words. Suggestions include guiding **GPT-4** to use Python tools to assist in generating the correct output by creating a list of end-letter sequences.
  
- **Anticipation for OpenAI Updates**: Conversation reveals anticipation for potential updates from OpenAI, with some expectations set on specific dates like the company's "birthday" and speculative delays due to elections. Users discuss the impact of updates on their excitement and expectations.
  
- **Delegating Tasks to Domain-Specific AIs**: A discussion on the potential for a "high level assistant" capable of delegating tasks to more specialized AI models ensued, touching on the prospects and challenges of creating a multi-tiered AI system with a "central brain".
  
- **ChatGPT Team and Privacy Concerns**: Questions about the capabilities regarding the ChatGPT team and individual account privacy prompted sharing of OpenAI's **enterprise privacy policy**. Users inquired about API key usage for multiple services and team chat visibility for admins.
  

**Link mentioned**: [Enterprise privacy](https://openai.com/enterprise-privacy): no description found

---

**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages):

wesego: Hi, having that problem right now.

---

**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1218032990517596160)** (7 messages):

- **Comma Confusion in Number Formats**: Members discussed a situation where someone was using a comma as a decimal separator, which might be common in South American regions. It's recommended to clarify this with the assistant, as models should handle different cultural number formats effectively.
- **Considering Global Number Formats**: When addressing the confusion over commas and decimals, one member noted that such issues can be resolved by simply informing the assistant, given that it's a widespread practice in various countries and the models are equipped to understand such differences.
- **Seeking Guidance on GPT-3 Prompt Architecture for Classification**: A member shared their efforts in using GPT-3 for a classification task, detailing their prompt structure and asking for advice on how to improve recall and minimize false positives. They were contemplating whether to adjust the amount of context or to consider using a custom GPT model.
- **Balance is Key in Prompt Design**: A suggestion was made concerning prompt architecture, advising to use no more than half of the total context window available for best results. This guidance was based on current model capabilities in handling context and the diminishing returns of information retrieval beyond a certain threshold.

---

**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1218032990517596160)** (7 messages):

- **Localization Woes in Decimal Representation**: There's been a discussion around a user having issues with the assistant due to the use of commas instead of decimal points in numbers. This was identified as a localization issue, typical for South American users where commas are used as decimal separators.
  
- **Model Cultural Flexibility**: *eskcanta* acknowledges that adjusting for cultural differences such as comma and decimal separators should be straightforward for the model, given its broad understanding of varied international formats.
  
- **Optimizing Classification Prompt Architecture**: A user named *mydpy* queries about refining a prompt setup for a classification task. The current structure includes static instructions, iterates through examples, and formats results, with the user seeking to balance context and minimize false positives.
  
- **Efficient Context Usage for Prompts**: *darthgustav.* suggests using a maximum of half the total context window for tasks to ensure best model performance. This guideline is based on the retrieval rates related to the position within the context window.
  

---

**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1217797795105341440)** (47 messages🔥):

- **Discussing Finetuning Large Models on Single GPUs**: Members expressed enthusiasm about a technique for finetuning 175 billion parameter models on a single NVIDIA 4090 GPU, citing an [abstract from a research paper on Hugging Face](https://huggingface.co/papers/2403.06504). They considered the implications for the **Axolotl** framework.
- **Model Training and Hardware Compatibility**: A conversation revolved around a member successfully running model training on Windows, despite concerns of potential incompatibilities with non-Mac systems. The member reported no issues post-training, though merge conflicts were mentioned.
- **Q&A vs. Completion Format for Knowledge Implementation**: Members debated the merits of training models on raw text completion format versus converting to Q&A format, considering potential information loss in the conversion process. **LoRA** was mentioned as a tool for stylistic training but the consensus was to use completion format for raw corpus training.
- **User Guidance on Data Format and Conversion in Axolotl**: There was a request for updated guides on data formats for **Axolotl**, and a subsequent clarification that raw text can be converted to a chat format like **ShareGPT** before training. Members shared how to use Axolotl for converting the format to **Llama-2** for chat model compatibility.
- **Differences Between Axolotl and LoRA Fine-Tuning**: A member inquired about the differences and potential control loss when using **Axolotl** compared to traditional **LoRA** fine-tuning in transformers library. It was clarified that Axolotl acts as a wrapper for the Hugging Face training ecosystem, offering simplification through YAML configuration files.
  

**Links mentioned**:

- [Paper page - Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU](https://huggingface.co/papers/2403.06504): no description found

- [cuFile API Reference Guide - NVIDIA Docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html): no description found

---

**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1218207901873606667)** (13 messages🔥):

- **ScatterMoE Optimizations Promising**: The Axolotl-dev channel discussed a new ScatterMoE implementation promising optimizations over Huggingface's approach, claiming to surpass MegaBlocks in throughput. A link to the [Optimized MoE models branch](https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe) was shared for review and consideration.
  
- **Seeking Clarifications on ScatterMoE**: Members inquired about the ScatterMoE optimization, asking for explanations on its benefits, how to train with it, and whether it would be integrated into other implementations such as vllm and llama.cpp.
  
- **Pull Request for Post Training Implementations**: An attempt to use ScatterMoE was made with a member sharing their [pull request link](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1407) and receiving feedback that it needed to more accurately recreate the MixtralMoE module and was still pending testing.
  
- **Upgrading PyTorch for Compatibility**: A member suggested upgrading Axolotl to a higher version of PyTorch as newer kernels are not compatible with the current version used, suggesting that version 2.0.1 is considered outdated.
  
- **Confirmation of Tool Versions**: Amidst the conversation about upgrades and implementations, a member confirmed that they are already utilizing PyTorch version 2.2.1, which is in line with the requirements for using ScatterMoE.
  

**Links mentioned**:

- [implement post training by ehartford · Pull Request #1407 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1407/commits/9c221a6761195c9739c02e11f9fe864bc947e53b): Does this look right?

- [implement post training by ehartford · Pull Request #1407 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1407): Does this look right?

- [GitHub - OpenAccess-AI-Collective/axolotl at scatter_moe](https://github.com/OpenAccess-AI-Collective/axolotl/tree/scatter_moe): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

---

**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1217868370293030993)** (9 messages🔥):

- **In Search of Inference Code**: A member sought example code for running inference on approximately 100 prompts with a **LoRA model** fine-tuned off `Mistral-7B-v0.1`. They contemplated using `transformers` and `model.generate(**model_inputs)` but was advised to consider using [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) as it could be quicker for their needs.
- **vLLM Might Be Better Than Transformers**: The suggestion to use **vLLM** for offline batched inference was emphasized again, highlighting its potential for quicker operations compared to the `transformers` library.
- **Token Trouble for Text Summarization**: A member reported issues with a tokenizer in a fine-tuning task for an **instruct model for document summarization**. The fine-tuned model frequently omitted the first `<summary>` tag or included an unwanted space before it, raising concerns about whether this was a tokenizer-related problem.
- **Fine-Tuning Frustration**: A newcomer to LLM queried about the correct syntax for configuring a script to point to locally stored model and training data rather than pulling resources from Huggingface. They were looking to fine-tune a model with already downloaded data.

**Link mentioned**: [Quickstart — vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html): no description found

---

**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1217863340790120498)** (4 messages):

- **Cohere Command-R Joins OpenRouter**: A new conversational model called **Command-R** created by Cohere, boasting a long context of 128k tokens, is now available. Users are encouraged to try it via the OpenRouter API, with 2 million prompt tokens per dollar and a link to play with it at [OpenRouter Models](https://openrouter.ai/models/cohere/command-r).
  
- **Boost Your Metrics with Daily Analytics**: OpenRouter has introduced daily analytics, allowing users to track token usage per day for all models, in addition to the existing weekly view. This feature can be explored at [OpenRouter Rankings](https://openrouter.ai/rankings).
  
- **API and Page Speed Enhancements**: OpenRouter has improved speed significantly, not just for the `/models` API but also for all model-related pages on the platform.
  
- **Model Parameter Data Awaiting More Info**: Despite the introduction of Cohere's Command-R, its parameters aren't yet listed in the `/parameters` API due to insufficient data. Once enough data is collected, it will become available at [Command-R Parameters](https://openrouter.ai/models/cohere/command-r?tab=parameters).
  

**Links mentioned**:

- [Cohere: Command-R by cohere | OpenRouter](https://openrouter.ai/models/cohere/command-r?tab=parameters): Command-R is an instruction-following conversational model that performs language tasks at a higher quality, more reliably, and with a longer context than previous models. It can be used for complex w...

- [Cohere: Command-R by cohere | OpenRouter](https://openrouter.ai/models/cohere/command-r): Command-R is an instruction-following conversational model that performs language tasks at a higher quality, more reliably, and with a longer context than previous models. It can be used for complex w...

- [OpenRouter](https://openrouter.ai/rankings): Language models ranked and analyzed by usage across apps

---

**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1217797935597752370)** (54 messages🔥):

- **One Library to Call Them All**: Users discussed [litellm](https://github.com/BerriAI/litellm), a universal API wrapper enabling calling various LLM APIs using OpenAI's format. While praised for its utility, limitations were noted, such as vision tasks only working with GPT-4 and certain features being specific to GPT models.
  
- **Navigating API Frontends and Payment Systems**: The conversation included suggestions for GUI frontends to plug in API keys, like [open-webui](https://github.com/open-webui/open-webui) and TypingMind.com, with varying charges for their use. The need to top up a balance to use APIs without connecting a credit card was also mentioned.
  
- **Seeking the Best LLM for Roleplay and Uncensored Dialogue**: Participants sought advice on the best LLMs for specific applications like roleplaying in Skyrim or engaging in controversial topics. Some users advocated for less censorship in LLMs, and there was particularly high praise for the creative outputs from models like Claude Sonnet.
  
- **Resolving Installation Issues and Understanding Limitations**: There were queries on how to install certain tools, such as a WebUI for LLMs, as well as discussions about the applicability of different models for unique use cases, like a lecture chatbot or a text-based roleplay experience.
  
- **Concerns Over Content Moderation and Model Censorship**: Users expressed concerns about overly stringent content moderation and the impact of censorship on model usability. Some dialog focused on the balance between preventing harmful content and retaining the creative capacities of LLMs, with suggestions for uncensored APIs and improved content filter mechanisms.
  

**Links mentioned**:

- [GitHub - open-webui/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI)](https://github.com/open-webui/open-webui): User-friendly WebUI for LLMs (Formerly Ollama WebUI) - open-webui/open-webui

- [GitHub - BerriAI/litellm: Call all LLM APIs using the OpenAI format. Use Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate (100+ LLMs)](https://github.com/BerriAI/litellm): Call all LLM APIs using the OpenAI format. Use Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate (100+ LLMs) - BerriAI/litellm

---

**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1217806780915253359)** (12 messages🔥):

- **NumPy vs. BLAS Performance Analysis**: A blog post argues that **NumPy**, despite its popularity for numerical computing in Python, has significant performance overhead—resulting in **up to 90% of throughput loss** with BLAS in specific operations like the 1536-dimensional OpenAI Ada embeddings. Their solution is [SimSIMD](https://github.com/ashvardanian/simsimd), which can minimize this loss.
  
- **Overhead in NumPy Discussed**: In the chat, someone pointed out that the **overhead in NumPy** for operations under 1µs is significant; therefore, for numerous small operations, a SIMD wrapper would be a more efficient solution instead of using NumPy, which adds unnecessary overhead.
  
- **Streamlined Messaging Suggested for Technical Articles**: A member suggested a more direct approach for technical write-ups, favoring clear explanations on intent, process, applicable scenarios, and installation instructions over merely showcasing benchmark numbers.
  
- **Photonic Computing Gains Traction**: A YouTube video titled "New Breakthrough in Photonics: x1000 faster. Is it for Real?" has been shared; its subject, **Lightmatter**, focuses on using photonic technology to reinvent chip communication and computation to improve AI’s environmental impact and efficiency. The video can be found [here](https://youtu.be/8ohh0cdgm_Y?si=q3wOMlzp_Nmn8_AJ).
  
- **Insightful Photonics Content Recommendations**: In support of the photonic technology discussion, members recommended Asianometry's videos for deeper insights—namely "Silicon Photonics: The Next Silicon Revolution?" and "Running Neural Networks on Meshes of Light"—which can be viewed on [YouTube](https://www.youtube.com/watch?v=29aTqLvRia8) and [YouTube](https://www.youtube.com/watch?v=t0yj4hBDUsc) respectively.
  

**Links mentioned**:

- [NumPy vs BLAS: Losing 90% of Throughput](https://ashvardanian.com/posts/numpy-vs-blas-costs/): Downloaded over 5 Billion times, NumPy is the most popular library for numerical computing in Python. It wraps low-level HPC libraries like BLAS and LAPACK, providing a high-level interface for matrix...

- [New Breakthrough in Photonics: x1000 faster. Is it for Real?](https://youtu.be/8ohh0cdgm_Y?si=q3wOMlzp_Nmn8_AJ): Get TypeAI PREMIUM now! Start your FREE trial by clicking the link here: https://bit.ly/Mar24AnastasiInTechThe paper: https://www.nature.com/articles/s41586...

- [Lightmatter®](https://lightmatter.co/): no description found

- [Silicon Photonics: The Next Silicon Revolution?](https://www.youtube.com/watch?v=29aTqLvRia8): My deepest thanks to friend of the channel Alex Sludds of MIT for suggesting this topic and helping me with critical resources. Check him out here: https://a...

- [Running Neural Networks on Meshes of Light](https://www.youtube.com/watch?v=t0yj4hBDUsc): I want to thank Alex Sludds for his efforts in helping me research and produce his video. Check out his work here: https://alexsludds.github.ioLinks:- The As...

---

**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1217825373170176181)** (10 messages🔥):

- **Debugging Triton Tensors made easy**: Set the environment variable `TRITON_INTERPRET=1` and use **print statements** to inspect tensor values. tl.arange(0,N) tensor indexing errors can be circumvented with these practical debug steps.
- **Visual Debugging Tools for Triton on the Horizon**: [A visualizer for Triton](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing) is in development, aimed at simplifying the inspection of the spatial structure of load/stores. There are a couple of known issues at the moment, including occasional double visualization and segfaults.
- **Outdated Debugging Methods**: The use of the `@triton.jit(interpret=True)` decorator for debugging Triton code has been noted as deprecated.
- **Helpful GitHub Discussions for Triton Debugging**: Specific issues and discussions on GitHub can offer help with debugging kernels, exemplified by [this GitHub issue](https://github.com/openai/triton/issues/517#issuecomment-1971327089).
- **Need More Annotated Triton Examples**: While the official tutorials are the primary learning resource, there's an expressed need for more annotated Triton kernel examples in the community to aid in understanding.
  

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing): no description found

- [Lecture 1 How to profile CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM): Slides: https://docs.google.com/presentation/d/110dnMW94LX1ySWxu9La17AVUxjgSaQDLOotFC3BZZD4/edit?usp=sharingCode: https://github.com/msaroufim/cudamodelecture1

- [How to debug kernels · Issue #517 · openai/triton](https://github.com/openai/triton/issues/517#issuecomment-1971327089): I'm trying to understand exactly what each line of code in add_kernel does in the vector add tutorial. Because it's a kernel I can't use a typical step debugger to go through this function...

---

**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1217829114803785840)** (13 messages🔥):

- **Kernel Launch Overhead Confusion Cleared**: A member gained clarification on unexpected output when swapping the order of CUDA functions - it was confirmed to be due to *kernel launch overhead*. The tool **ncu** was recommended to isolate this overhead.
- **Seeking CUDA Learning Resources**: A member new to CUDA sought beginner-friendly learning materials, and it was established that the member is familiar with **C++**, a useful pre-requisite for CUDA.
- **FP8 Matmul on 4090s Shows Promise**: A brief mention noted that **fp8 matrix multiplication on 4090 GPUs** is impressively fast, indicating potential performance gains.
- **Recommendation for a CUDA Beginner's Book**: For learning CUDA, the book *Programming Massively Parallel Processors* was recommended, identified as a foundational text even for undergraduates and not too advanced for those knowledgeable in C/C++.
- **Join a CUDA Programming Book Reading Group**: For those who are beginning to learn CUDA, it was shared that there is a [reading group](https://discord.com/channels/1189498204333543425/1194427148656721970) available, which indicates community support for those working through the recommended book.

**Link mentioned**: [Programming Massively Parallel Processors: A Hands-on Approach: Hwu, Wen-mei W., Kirk, David B., El Hajj, Izzat: 9780323912310: Amazon.com: Books](https://www.amazon.com/dp/0323912311): no description found

---

**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages):

vim410: Depends. But yes.

---

**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1217801403637039116)** (8 messages🔥):

- **SM Architecture and Processing Blocks Explained**: A member referred to lecture 4 for a visual aid in understanding GPU architecture, detailing that a **GA102 SM has 4 processing blocks** which execute a warp at a time. **32 fp32 instructions can run concurrently**, while int32 instructions are split into two batches of 16 due to the core limitations.
  
- **Indexing Dilemma in CUDA Coding**: When discussing a chapter 2 query, a wrong indexing approach `i = blockIdx.x * blockDim.x + threadIdx.x * 2` was corrected by an explanation that showed it would result in double-counting. To illustrate, with `blockDim.x = 32`, both `{blockIdx.x = 0, threadIdx.x = 16}` and `{blockIdx.x = 1, threadIdx.x = 0}` would erroneously yield `i = 32`.
  
- **Questioning Content Sharing Boundaries**: A member queried the propriety of blogging their answers to **CUDA** exercises, mentioning an attempt to contact the authors without success due to lack of an educational email address. Another member promised to check with author **Wen-mei** for clarification.
  

---

**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1218019177328021535)** (7 messages):

- **Confusion Over Ring Attention's Compatibility**: A member expressed uncertainty on why there are claims that **ring attention** cannot be used with flash, despite similar implementations being apparently successful.
- **Awaiting Response from Busy Member**: Andreas Koepf indicated being quite busy, promising to get back to the conversation when availability improves, to which Jamesmel responded with understanding.
- **Searching for the Missing Code**: Iron_bound expressed disappointment in not being able to find associated code for the Twitter post about ring attention, leaving a sentiment of incomplete understanding.
- **Link to Triton Kernel Code Shared**: Iron_bound provided a link to a [Triton kernel implementation](https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5) which seems related to the discussion of ring flash attention.

**Link mentioned**: [add naive triton kernel for varlen · zhuzilin/ring-flash-attention@10d992c](https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5): no description found

---

**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1217773505983152280)** (3 messages):

- **Meta vs. Former Exec Lawsuit**: Meta has legally targeted a former executive, accusing him of **stealing over 100 internal documents** and attempting to recruit Meta's employees for a competing AI data startup, Omniva. The lawsuit was publicized through an unsealed [court filing](https://cdn.arstechnica.net/wp-content/uploads/2024/03/Meta-v-Khurana-complaint-2-29-2024.pdf) and further reported in an [Ars Technica article](https://arstechnica.com/tech-policy/2024/03/meta-sues-brazenly-disloyal-former-exec-over-stolen-confidential-docs/).
- **Disappointment in Channel Dynamics**: A user expressed dissatisfaction with how a conversation in the channel was progressing, using a brief phrase to imply that the discussion's direction was not as expected.
- **Starting From Scratch**: A member mentioned that all three participants in a conversation are starting from **lecture 1**, possibly indicating a collaborative learning effort or a collective beginning of a new topic or course.

**Link mentioned**: [Meta sues “brazenly disloyal” former exec over stolen confidential docs](https://arstechnica.com/tech-policy/2024/03/meta-sues-brazenly-disloyal-former-exec-over-stolen-confidential-docs/): Meta's former exec allegedly shared data center secrets with a shadowy startup.

---

**LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1217877731963048016)** (1 messages):

- **Langchain 0.2 Release Rush**: Due to recent CVEs filed against **`langchain`**, the team is expediting the release of version 0.2, which will break the dependency on `langchain-community`. The bigger refactors planned will now be shifted to version 0.3, and more details are available in a [GitHub discussion](https://github.com/langchain-ai/langchain/discussions/19083).
- **Call for Community Feedback**: The LangChain team is seeking feedback on the upcoming changes to ensure they do not cause any issues for users. The team emphasizes that the goal of these changes is to *make your life easier*.

**Link mentioned**: [RFC: Expedited langchain 0.2 release · langchain-ai/langchain · Discussion #19083](https://github.com/langchain-ai/langchain/discussions/19083): Context Currently langchain (the package) depends on langchain-community. This is done only for backwards compatibility with langchain versions that predate the split of langchain and langchain-com...

---

**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1217764040911228928)** (34 messages🔥):

- **Troubleshooting AgentExecutor Execution Errors**: A user is facing an `OutputParserException` when running an `AgentExecutor` with a command from Cohere, although the python code seems properly generated. The expectation is the agent would execute python code and respond in natural language.
- **Langsmith and Imported Prompts Confusion**: A member struggles to understand why their custom prompt doesn't enable tool use compared to an [imported prompt from hub](https://langsmith.ai/hub?pull=hwchase17%2Fopenai-tools-agent), and seeks clarification about the differences.
- **API Query via Curl for StackOverflow**: A user inquired about using an API to query StackOverflow, and was directed to use the [StackExchange API](https://api.stackexchange.com/docs/advanced-search) for advanced search functionality to meet their requirements.
- **Debating the Usefulness of LLM Agents**: A discussion unfolded regarding the practicality of LLM agents over combining LLM output with functions, with members debating agents' abilities for action sequencing and error-handling, and pondering ways to evaluate agent behavior, possibly with the help of [LangChain benchmarks](https://python.langchain.com/docs/guides/evaluation).
- **Using LangGraph for Cyclic Computations with LLMs**: An explanation was provided for using LangGraph when needing to add cycles to applications, especially for stateful, multi-actor applications with LLMs, with references to [JavaScript](https://js.langchain.com/docs/langgraph) and [Python](https://python.langchain.com/docs/langgraph) LangGraph documentation for further details.
  

**Links mentioned**:

- [Usage of /search/advanced [GET] - Stack Exchange API](https://api.stackexchange.com/docs/advanced-search) : no description found

- [Debugging | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/debugging): If you're building with LLMs, at some point something will break, and you'll need to debug. A model call will fail, or the model output will be misformatted, or there will be some nested mod...

- [Evaluation | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/evaluation/): Building applications with language models involves many moving parts. One of the most critical components is ensuring that the outcomes produced by your models are reliable and useful across a broad ...

---

**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1217795010247000124)** (1 messages):

- **Query on Langsmith Hub Prompt Templates**: A member inquired about how to create a prompt template in Langsmith Hub, demonstrating a placeholder `{tools}` for a list named `tools` in their code. They were specifically looking for guidance on linking the `tools = [cat_tool]` variable to the placeholder in the template.

---

**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1217841611505926155)** (6 messages):

- **SAP HANA Meets LangChain**: A blog post explores the innovative integration of LangChain with the SAP HANA Vector Engine, presenting potential advances in AI applications. For more information on this synergy, visit [Unlocking the Future of AI Applications](https://ai.gopubby.com/unlocking-the-future-of-ai-applications-with-hana-vector-engine-and-langchain-14cd6c66219d).
  
- **Dall-E Enters the JavaScript World**: Blog post details the addition of Dall-E image generation support to the JavaScript version of LangChain. Useful code snippets and instructions included in [Lang Chain for JavaScript Part 3: Create Dall-E Images](https://fek.io/blog/lang-chain-for-java-script-part-3-create-dall-e-images/).
  
- **Orchestrating Operational Browser Flows with AI**: A new blog post describes how a system of LLM agents is orchestrated to facilitate automated browser interactions. Check out the engineering behind it at [The Engineering of an LLM Agent System](https://checksum.ai/blog/the-engineering-of-an-llm-agent-system).
  
- **Open Source Langchain Chatbot Showcases RAG for Q/A**: The Langchain chatbot, which utilizes RAG for efficient question and answer querying, is now fully open source. Investigate the application on [GitHub](https://github.com/Haste171/langchain-chatbot).
  
- **Living Bookmarks Bot for Better Bookmark Management**: A Twitter user developed a Discord AI chatbot for managing Raindrop.io bookmarks to aid in finding them easily when needed, and has made it [available open source](https://github.com/uogbuji/living-bookmarks).
  

**Links mentioned**:

- [LangChain for JavaScript part 3: Create Dall-E images](https://fek.io/blog/lang-chain-for-java-script-part-3-create-dall-e-images/): FEK.IO The website for David Fekke L.L.C.

- [GitHub - Haste171/langchain-chatbot: AI Chatbot for analyzing/extracting information from data in conversational format.](https://github.com/Haste171/langchain-chatbot): AI Chatbot for analyzing/extracting information from data in conversational format. - Haste171/langchain-chatbot

- [Unlocking the Future of AI Applications with SAP HANA Vector Engine and LangChain](https://ai.gopubby.com/unlocking-the-future-of-ai-applications-with-hana-vector-engine-and-langchain-14cd6c66219d): Ankush k Singal

---

**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages):

pradeep1148: [https://www.youtube.com/watch?v=PzaidfqDtGI](https://www.youtube.com/watch?v=PzaidfqDtGI)

---

**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1217831420593176666)** (27 messages🔥):

- **Looking for GPU Partners for Captioning**: A member requested help with captioning and is seeking someone with spare 3090s or 4090s to assist. They've also asked interested individuals to reach out via direct message.
- **Optimizing on MacOS with M3 Max**: A member is working on getting simpletuner to run on MacOS and discussed the potential of using more than 96GB of the system's memory for compute on a new 128G M3 Max system.
- **Sharing Prompt Augmentation Innovations**: A link to an article about prompt augmentation using a 77M T5 model was shared along with an impressive demonstration of its capabilities in image generation. Another member contributed by sharing a link to *DanTagGen*, an autocompleting tags tool using a smaller model, on HuggingFace.
- **Interest in AI Law Regulation from the EU**: A member highlighted the adoption of the Artificial Intelligence Act by the European Parliament, designed to ensure AI safety and compliance with fundamental rights. The regulation aims to address the risks of AI and impact applications that threaten citizens’ rights.
- **IEEE Symposium on Security and Privacy Update**: A member posted about the 45th IEEE Symposium on Security and Privacy being removed from the accepted papers page. There was a brief conversation about the implications of this removal for a person named Ben and whether they would resubmit to an appropriate conference.
  

**Links mentioned**:

- [SuperPrompt - Better SDXL prompts in 77M Parameters | Brian Fitzgerald](https://brianfitzgerald.xyz/prompt-augmentation/): Left SDXL output with SuperPrompt applied to the same input prompt.

- [IEEE Symposium on Security and Privacy 2024](https://sp2024.ieee-security.org/accepted-papers.html): no description found

- [Artificial Intelligence Act: MEPs adopt landmark law | News | European Parliament](https://www.europarl.europa.eu/news/en/press-room/20240308IPR19015/artificial-intelligence-act-meps-adopt-landmark-law): On Wednesday, Parliament approved the Artificial Intelligence Act that ensures safety and compliance with fundamental rights, while boosting innovation.

---

**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1217788714772598814)** (13 messages🔥):

- **Virtual Try-On with TryOnDiffusion Unveiled**: An open-source implementation of *TryOnDiffusion*, as described in the Google paper "A Tale of Two UNets", has been released under the MIT License. The code is available on [GitHub](https://github.com/fashn-AI/tryondiffusion).
  
- **Fast Decoding Research Mentioned**: A paper claiming 2D Gaussian splatting decodes faster than jpeg has been shared, insinuating it could be of interest due to its speed and optimization. The paper can be found on [arXiv](https://arxiv.org/pdf/2403.08551.pdf).
  
- **Personal Project Reflection**: A member recounted their own attempt at a project conceptually similar to the one described in the aforementioned 2D Gaussian splatting paper, admitting they weren't able to optimize it as well but found validation in seeing professional work align with their methods.
  
- **Resource Constraints in Model Deployment**: Inquiring about how to implement a CPU cap like the one used in a text-generation web UI, a member shared their struggle with *CUDA out of memory* issues on non-UI models. They are seeking insights on handling large models without hitting free tier limitations as outlined in the [GitHub repo](https://github.com/oobabooga/text-generation-webui).
  
- **Limits of Free Colab for Web UIs**: Further to the previous point, other members explained that you can't use free Colab for running web UIs, hinting the suitability of discussion on other channels meant for such technical inquiries.
  

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/github/Nick088Official/zephyr-7b-gemma-v0.1_Google_Colab/blob/main/zephyr-7b-gemma-v0.1_Manual.ipynb): no description found

- [Google Colaboratory](https://colab.research.google.com/github/Nick088Official/WhiteRabbitNeo-7b-v1.5a-Google-Colab/blob/main/WhiteRabbitNeo_7b_v1_5a.ipynb): no description found

- [GitHub - fashn-AI/tryondiffusion: PyTorch implementation of "TryOnDiffusion: A Tale of Two UNets", a virtual try-on diffusion-based network by Google](https://github.com/fashn-AI/tryondiffusion): PyTorch implementation of "TryOnDiffusion: A Tale of Two UNets", a virtual try-on diffusion-based network by Google - fashn-AI/tryondiffusion

- [GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.](https://github.com/oobabooga/text-generation-webui): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui

---

**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1218226914322415677)** (1 messages):

Since there is only one message provided and no additional context such as previous messages, links, or discussion points, a summary cannot be generated based on the instructions given. Please provide a series of messages or more context to summarize.

---

**LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1218012492987633684)** (1 messages):

- **GPT-4 Turbo Goes on Space Mission**: One member reported encountering a peculiar issue with `gpt-4-turbo-preview`, where a completion task with a very long passage (12,000 tokens) resulted in the model endlessly outputting space characters. In an unusual twist, the model even began generating "Russian gibberish" after a lengthy sequence of spaces, as evidenced by attached screenshots.

---

**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1217843643541622935)** (18 messages🔥):

- **Haiku's Cost-Effective Document Describing**: A member highlighted the efficiency of **Haiku** in describing complex documents visually for economical costs, but also noted it is not as good as **GPT-vision**.
- **Limitations in Haiku's Performance**: Despite its strides, Haiku is still seen as inferior to **Opus** in vision-to-text tasks.
- **Content Filter Hurdles with Claude**: There were issues with Claude regarding content filtering, particularly with it stopping mid-way when parsing documents containing equations.
- **Controversial Take on Anthropic**: A tweet shared in the chat suggests **Anthropic** is perceived as a strategic entity aiming to instill a 'fear of god' among technical staff members.
- **Content Moderation Challenges for Specific Images**: Users reported content moderation issues with images that contain people, where the system sometimes refuses to process them.

**Link mentioned**: <a href=[https://x.com/tszzl/status/1768530219378631137?s=20>Tweet](https://x.com/tszzl/status/1768530219378631137?s=20%3ETweet) from roon (@tszzl): anthropic is controlled opposition to put the fear of god in the members of technical staff

---

**LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/1218241222347460619)** (16 messages🔥):

- **KPU: The Next Big Thing in AI?**: [Maisa announces the KPU (Knowledge Processing Unit)](https://maisa.ai/blog/kpu), a new framework designed to enhance LLMs by separating reasoning from data processing. KPU claims to outperform GPT-4 and Claude 3 Opus in reasoning tasks.
  
- **Questioning Benchmarks**: Members express skepticism about KPU comparing its performance with GPT-4 and not GPT-4 Turbo. The discussions highlight concerns about potentially unfair benchmarks.
  
- **KPU: Beyond Prompt Engineering?**: One member wonders if KPU's technology is merely about prompt engineering, while another clarifies that it includes self-evaluation and context window management tricks.
  
- **Examining Comparative Analysis**: Humorous reactions emerge in response to the apparent omission of GPT-4 Turbo from KPU's comparative analysis, suggesting a pattern also seen in Claude 3's release.
  
- **Concern over Practical Efficiency**: A discussion ensues about KPU's lack of latency information, raising doubts about its practical application in real-world products despite purported improvements in accuracy.
  

**Links mentioned**:

- [KPU - Maisa](https://maisa.ai/blog/kpu): AI-Powered Knowledge Processing Platform. A simple API for executing business tasks. Abstracting the complexities of using the latest AI architectures for software and app developers

- [Tweet from David Villalón (@davipar)](https://x.com/davipar/status/1768683151780683919?s=20): happy to answer! it is not a new model, indeed KPU is agnostic to intelligence providers (OpenAI, Antrophic...). It is a new AI architecture to work with LLMs that leverages their reasoning capabiliti...

---

**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1218193382669549568)** (17 messages🔥):

- **Paper on Improved Training Method Coming**: A member is working on releasing a **paper/article** that suggests a method which seems to improve global accuracy and makes the training more sample efficient. They are planning to structure the results and create better visualizations for their findings.
- **Seeking Resources for Scaling**: The approach needs validation for large model efficacy, but currently, there is a **lack of resources** to empirically prove it at scale.
- **Method Shows Promise Even With Large Models**: Initial tests with **VGG16** on a subset of CIFAR100 show a significant improvement using the new method (0.1 test accuracy) over base training (0.04 test accuracy).
- **Collaboration on Resource Allocation**: Members are coordinating to help allocate **compute and resources** for further testing and scaling of the new training method.
- **Involvement in the Quiet-STaR Project**: A member expressed interest in participating in the implementation of "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking" and was asked about their proficiency in **PyTorch** and **transformers architecture**.

---

**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1217826155416129596)** (2 messages):

- **Dive into Hermes 2 Pro 7B Function Calls**: A link to a YouTube video titled "Lets Function Call with Hermes 2 Pro 7B" was shared, which demonstrates function calling with the language model Hermes 2 Pro 7B. The video is accompanied by a [GitHub repository](https://github.com/NousResearch/Hermes-Function-Calling/tree/main) that dives deeper into the Hermes function calling capabilities.
- **Meta Quest Hackathon Looking for Innovators**: An invitation was extended to join a team for the Meta Quest Presence Platform Hackathon, where participants create innovative mixed reality content using the Presence Platform on Meta Quest. Interested individuals are encouraged to learn more and join without needing any prior skills, suggesting a learn-as-you-go approach, and were referred to the [hackathon's resources](https://metaquesthackathon.devpost.com/resources).
  

**Links mentioned**:

- [Lets Function Call with Hermes 2 Pro 7B](https://www.youtube.com/watch?v=PzaidfqDtGI): lets do function calling with Hermes 2 Pro 7Bhttps://github.com/NousResearch/Hermes-Function-Calling/tree/main#llm #largelanguagemodels

- [Meta Quest Presence Platform Hackathon 2024](https://metaquesthackathon.devpost.com/.): Next Generation Quest MR Applications

---

**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1217980476531281980)** (16 messages🔥):

- **Searching for the Prompt Engineering Workbench**: A member inquired about a tool analogous to Postman for prompt engineering that allows for managing a prompt library, versioning prompts, staging data, running tests, and integrating with multiple models.
- **Using SQLite for Prompt Capturing**: Another member shared their method, utilizing **LLM in the terminal**, to manage prompts and responses by capturing them in [SQLite](https://sqlite.org/index.html), with an observation that a custom UI might be beneficial.
- **Prodigy as a Prompt Engineering Tool**: The conversation included a mention of a tool previously developed for [Explosion’s Prodigy](https://prodi.gy/features/prompt-engineering), a paid product that integrates prompt engineering as a data annotation problem offering facilities like A/B testing capability.
- **PromptTools for Prompt Experimentation**: The [PromptTools GitHub repository](https://github.com/hegelai/prompttools), an open-source initiative for prompt testing and experimentation with support for varied LLMs and vector databases, was suggested as a resource for setting up experiments.
- **Helicone AI Enters the Prompt Management Arena**: A participant pointed to [Helicone AI](https://www.helicone.ai/), a developing platform for Generative AI applications that is starting to incorporate features related to prompt management, versioning, and analysis.
  

**Links mentioned**:

- [Helicone](https://www.helicone.ai/): How developers build AI applications. Get observability, tooling, fine-tuning, and evaluations out of the box.

- [Vercel AI SDK](https://sdk.vercel.ai/): Build AI-powered applications with the latest AI language models

- [GitHub - hegelai/prompttools: Open-source tools for prompt testing and experimentation, with support for both LLMs (e.g. OpenAI, LLaMA) and vector databases (e.g. Chroma, Weaviate, LanceDB).](https://github.com/hegelai/prompttools): Open-source tools for prompt testing and experimentation, with support for both LLMs (e.g. OpenAI, LLaMA) and vector databases (e.g. Chroma, Weaviate, LanceDB). - hegelai/prompttools

---

**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/)** (1 messages):

obra: Is it possible to recover the seed used by the openai models for a previous api request?

---

**Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1218217772765544448)** (8 messages🔥):

- **Unlocking LLM Secrets via API**: A recently discussed [research paper](https://arxiv.org/abs/2403.09539) explores how to extract non-public information about API-protected Large Language Models (LLMs) like OpenAI's GPT-3.5 by exploiting the softmax bottleneck—revealing details such as hidden model size with a relatively low number of API queries.
- **Discussion on Carlini's Latest Work**: A participant referenced a recent paper by Carlini et al. that investigated the model size estimation through logits but noted that the key details were redacted.
- **Surprise Over Alleged Model Size**: One member expressed surprise that the model size could be 7B parameters, suggesting such an estimation seems implausible.
- **Skepticism on Model Size Accuracy**: Resistance to the 7B size estimate was voiced, with speculation that the calculation might be flawed, especially if GPT-3.5 is a Mixture of Experts (MoE) model.
- **Theory of Distillation or Mixtures in Models**: A discussion speculated on the use of *'mega distillation sauce'* or token-critical mixtures in turbo LLMs, citing past research that showed the beginning tokens are crucial for performance in tasks like math problems.

**Link mentioned**: [Logits of API-Protected LLMs Leak Proprietary Information](https://arxiv.org/abs/2403.09539): The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption...

---

**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1217952174852542485)** (4 messages):

- **Seeking Citations for Safety Filtering by Model Providers**: A member asked for references to support the statement that "foundation model providers do a lot of the safety filtering for text post generation."
- **Agile Text Classifiers Aid Safety Policy**: Another member provided a reference to the paper [Agile classifiers for safer chatbots](https://arxiv.org/abs/2302.06541), which discusses how *prompt-tuning large language models with small datasets can quickly adapt to safety policies* and achieve state-of-the-art performance.
- **Satisfaction with Safety Filtering Resource**: The initial member acknowledged that the provided paper on **agile text classifiers** helps convey the intended point about foundation model providers' role in safety filtering.

**Link mentioned**: [Towards Agile Text Classifiers for Everyone](https://arxiv.org/abs/2302.06541): Text-based safety classifiers are widely used for content moderation and increasingly to tune generative language model behavior - a topic of growing concern for the safety of digital assistants and c...

---

**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1217762291387334717)** (5 messages):

- **Craving Ultra-Long Context**: A member expressed a hopeful outlook towards the development of Gemini for ultra-long contexts, which they hope will improve summary generation, currently used as "better abstracts."
- **Contemplating the Serendipity of Smarter Prompts**: Another member discussed the challenges of finding the right balance in prompt engineering and looks forward to more intuitive and less tedious prompting mechanisms, likening it to search engine suggestions which get "warmer or colder" to guide users.
- **Innovative Paper Summarization Concept**: A new approach to summarizing academic papers was proposed where an AI tool would monitor new papers citing a user's favorite research, potentially providing contextual citations, like where the referenced dataset is used.
- **Dispelling GPT-4.5 Rumors**: One member conveyed disappointment, inferring from available information that GPT-4.5 would not be released "today."
- **Entertainment in AI Discussions**: A tweet was shared indicating Yann LeCun's skeptical stance on language models, prompting discussion and reactions within the group. [This explains why Yann is so bearish on LLMs](https://fxtwitter.com/i/status/1768452864995942469/).

**Link mentioned**: [Tweet from Teknium (e/λ) (@Teknium1)](https://fxtwitter.com/i/status/1768452864995942469/): This explains why Yann is so bearish on LLMs... 😲

---

**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1217984341871820961)** (3 messages):

- **Language Struggle with DiscoLM-70b**: A member encountered difficulties in eliciting responses in **English** from DiscoLM-70b, despite the model's card suggesting multi-language capabilities. It was suggested to analyze the **prompt structure** for potential issues.
- **Cross-Model Performance Mysteries**: Comparisons with other models like **LeoLM variants, llama2, and Nous-Hermes-2-Mixtral** showed expected performance in multilingual tasks. The same member reported that after instruction fine-tuning, the **DiscoLM-mixtral-8x7b-v2** failed to generate responses in German.
- **Fine-Tuning Hurdles with DiscoLM**: Supervised fine-tuning of DiscoLM as a sequence classification problem resulted in a **ValueError**, indicating an unrecognized configuration class for `AutoModelForSequenceClassification`. The error suggests possible compatibility issues with the current setup.

---

**DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1217805725762453534)** (1 messages):

- **Introducing "GermanQuAD" Evaluation Task**: The embedding_dev channel includes a message about the **"GermanQuAD" evaluation task**, which can be used in the MTEB's python package, as well as mentioning recent German additions from [JinaAI](https://jina.ai/).

---

**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1218006737190129797)** (5 messages):

- **Demo Availability Confusion**: A member inquired whether the demo was available, implying that it might be down or inaccessible at the moment.
- **Model Prompt Respect**: A member explained that the model is trained to respect the system prompt and suggested trying variations for optimal outcomes. They confirmed that the demo doesn't utilize special settings and runs on **fastchat/vllm**.
- **Demo Down Due to Server Move**: In response to the demo availability question, it was clarified that the server hosting the demo was moved and networking issues arose, causing downtime. The hope is to have the demo running again by early next week.
- **Hobbyist vs Professional Hosting Challenges**: A member humorously remarked on the reliability of a hobbyist server in a kitchen corner compared to professional hosting, which seems to face networking issues and other technical hiccups.

---
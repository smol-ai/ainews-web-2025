---
id: fcf1837d-66ee-4eaa-a953-9d366d904115
title: Perplexity, the newest AI unicorn
date: '2024-04-23T22:48:23.949413Z'
original_slug: ainews-perplexity
description: >-
  AI news covering recent developments in Llama 3 variants, optimizations,
  performance, and open-source models like Phi-3, along with updates on
  Perplexity's Series B-1 funding and community discussions on Reddit.
companies:
  - huggingface
  - meta
  - perplexity
models:
  - llama-3-8b-16k
  - llama-3-8b-v1_1
  - llama-3-70b
  - phi-3-3.8b
  - phi-3-7b
  - phi-3-14b
topics:
  - llama-3
  - multimodal
  - fine-tuning
  - web-browsing
  - instruction-following
  - model-comparison
  - open-source-ai
  - language-models
  - performance-evaluation
  - quantization
  - web-data
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/20/2024-4/23/2024. We checked 7 subreddits and [**373** Twitters](https://twitter.com/i/lists/1585430245762441216) and **27** Discords (**395** channels, and **14864** messages) for you. Estimated reading time saved (at 200wpm): **1509 minutes**.

Just 3 months after the [Series B](https://buttondown.email/ainews/archive/ainews-142024-jeff-bezos-backs-perplexitys-520m/), Perplexity doubles its valuation again with a Series B-1, with mostly the same list of stellar investors as last time, but a rare split of Daniel Gross *not* co-leading with Nat Friedman. Dan seems to have a special relationship with the company - Aravind shared [a Dec 2022 email on Dan's product feedback](https://x.com/AravSrinivas/status/1782785662607114365).

 ![image.png](https://assets.buttondown.email/images/60694bbc-7fdd-4bb0-8a9a-928b03a06a30.png?w=960&fit=max) 


---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**Llama 3 Variants and Optimizations**

- **Context Length Extension**: In /r/LocalLLaMA, the context length of Llama-3-8B has been [**extended to 16K tokens**](https://huggingface.co/mattshumer/Llama-3-8B-16K), doubling its original context window.
- **Multimodal LLaVA Models**: The XTuner team has released [**LLaVA models based on Llama 3**](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1) on Hugging Face, which substantially outperform Llama 2 on various benchmarks.
- **BOS Token Reminder**: In /r/LocalLLaMA, a [**PSA reminds users to ensure their training setups add the BOS token**](https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_training_setup_is_adding_bos/) when finetuning Llama 3 models to avoid issues like inf grad_norm or higher loss.
- **Special Token Embedding Adjustments**: Adjustments have been made to the [**untrained special token embeddings in Llama-3-8B**](https://huggingface.co/astronomer/Llama-3-8B-Special-Tokens-Adjusted) and shared on Hugging Face to address finetuning issues caused by zero values.
- **Web browsing and interaction**: In /r/LocalLLaMA, [Llama-3-8B-Web action model introduced for web browsing and user interaction](https://www.reddit.com/r/LocalLLaMA/comments/1caw3ad/sharing_llama38bweb_an_action_model_designed_for/). WebLlama project aims to advance Llama-based agent development. Demos of [voice chatting with Llama 3 8B using OpenAI TTS and Whisper](https://v.redd.it/xwr67vtxkzvc1) shared.
- **Fine-tuning and extensions**: QDoRA introduced for [memory-efficient and accurate fine-tuning of Llama 3 models](https://www.reddit.com/r/LocalLLaMA/comments/1cas7wg/qdora_efficient_finetuning_of_llama_3_with_fsdp/), outperforming QLoRA and Llama 2. [Hugging Face Space for creating GGUF quantizations of Llama 3 models](https://www.reddit.com/r/LocalLLaMA/comments/1ca7xf8/create_llama_3_quants_through_a_hugging_face_space/) shared. Importance of [adding BOS token when fine-tuning Llama 3](https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_training_setup_is_adding_bos/) discussed.


**Llama 3 Performance and Capabilities**

- **Instruction Following**: In /r/LocalLLaMA, Llama-3-70B is praised for its [**ability to follow format instructions and provide concise responses**](https://www.reddit.com/r/LocalLLaMA/comments/1canrjq/llama370b_is_insanely_good_at_following_format/) without unnecessary boilerplate text.
- **Model Comparison**: An in-depth comparison of 20 Llama 3 Instruct model versions across HF, GGUF, and EXL2 formats at various quantization levels is shared in /r/LocalLLaMA. Key findings include [**EXL2 4.5bpw and GGUF 8-bit to 4-bit performing exceptionally well**](https://www.reddit.com/r/LocalLLaMA/comments/1cal17l/llm_comparisontest_llama_3_instruct_70b_8b/), while 1-bit quantizations showed significant quality drops.
- **Groq-Hosted Model Performance**: The Groq-hosted Llama-3-70B struggles with a lateral thinking puzzle compared to the HuggingChat version, as reported in /r/LocalLLaMA. [**Temperature settings significantly impact reasoning performance**](https://www.reddit.com/r/LocalLLaMA/comments/1casosh/groq_hosted_llama370b_is_not_smart_probably/), with 0.4 providing the best consistency.

**Phi-3 and Llama 3 Models Push Boundaries of Open-Source Language AI**

- **Phi-3 models released in 3.8B, 7B, and 14B sizes**: In /r/singularity, Meta released Phi-3 models trained on [**heavily filtered web data and synthetic data**](https://www.reddit.com/r/singularity/comments/1cau7ek/phi3_released_medium_14b_claiming_78_on_mmlu/). The 14B model claims 78% on MMLU, rivaling Llama 3 8B despite smaller size. Weights coming to Hugging Face soon.

- **Phi-3 3.8B nears GPT-3.5 performance**: In /r/singularity, the Phi-3 3.8B model is [**nearing GPT-3.5 performance on benchmarks**](https://www.reddit.com/r/singularity/comments/1cau3gy/phi3_a_small_38b_model_nears_gpt35_on_major/), with 7B and 14B versions also available. Weights releasing with a demo video, showing mind boggling progress in model efficiency.

- **Llama 3 70B ties GPT-4 on LMSYS leaderboard**: In /r/singularity, Llama 3 70B [**took second place on the LMSYS arena English leaderboard, tying GPT-4-Turbo for first**](https://www.reddit.com/r/singularity/comments/1cau6yz/llama_3_70b_takes_second_place_in_the_english/). It can be used for free through Groq API or Hugging Face. Questions raised about arena ranking validity.

- **Phi-3 technical report shows impressive benchmarks**: In /r/singularity, the Phi-3 technical report was released showing the [**3.8B model rivaling Mixtral 8x7B with 69% MMLU and 8.38 MT-bench**](https://www.reddit.com/r/singularity/comments/1catcdv/phi3_technical_report_impressive/). The 7B and 14B models show further scaling to 75% and 78% MMLU.

- **Doubling parameters yields diminishing returns for Llama 3**: In /r/singularity, a chart showed that [**doubling parameters on the same dataset scales MMLU scores by an average 17%, but only 5% for Llama 3 models**](https://www.reddit.com/r/LocalLLaMA/comments/1caneis/doubling_the_parameters_on_the_same_dataset/), suggesting Llama 3 is highly optimized already.

**Miscellaneous**

- **Parameter Scaling**: According to an image shared on Reddit, [**doubling model parameters on the same dataset typically scales MMLU performance by 17% on average, but only 5% for Llama 3 models**](https://i.redd.it/izvkuwo1s3wc1.png).
- **High-Speed Inference**: SambaNova Systems demonstrates [**high-speed inference of 430 tokens per second for Llama 3 8B**](https://www.reddit.com/r/LocalLLaMA/comments/1caxbx6/sambanova_systems_running_llama_3_8b_at_430_tps/) using 8 chips with FP16 precision, as reported in /r/LocalLLaMA.
- **Quantization Democratization**: A Hugging Face Space is introduced in /r/LocalLLaMA to [**democratize the creation of GGUF quantizations for Llama 3 models**](https://www.reddit.com/r/LocalLLaMA/comments/1ca7xf8/create_llama_3_quants_through_a_hugging_face_space/), improving reliability and accessibility.

---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Perplexity AI Raises $62.7M at $1.04B Valuation**

- **Funding Details**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1782784338238873769) and [@perplexity_ai](https://twitter.com/perplexity_ai/status/1782782211399279076) announced Perplexity AI raised **$62.7 million** in a Series B1 funding round at a **$1.04 billion valuation**, led by **Daniel Gross**, along with investors including **Stan Druckenmiller, NVIDIA, Jeff Bezos, Tobi Lutke, Garry Tan, Andrej Karpathy, Dylan Field, Elad Gil, Nat Friedman, IVP, NEA, Jakob Uszkoreit, Naval Ravikant, Brad Gerstner and Lip-Bu Tan**.
- **Growth and Partnerships**: Since January 2024, Perplexity has grown to serve **169M queries per month**, over **1 billion queries in the last 15 months**. Perplexity has partnerships with **Deutsche Telekom and Softbank** to distribute to ~**116M users worldwide**. [@AravSrinivas](https://twitter.com/AravSrinivas/status/1782785026524135848)
- **Perplexity Enterprise Pro Launch**: Perplexity is launching **Perplexity Enterprise Pro**, which comes with **SOC2 compliance, SSO, user management, enterprise-grade data retention, and security warnings** to address data and security concerns for enterprise use. [@AravSrinivas](https://twitter.com/AravSrinivas/status/1782775219733844256), [@perplexity_ai](https://twitter.com/perplexity_ai/status/1782774382399557633)

**Meta's Llama-3 Model Achieves Top Performance**

- **Llama-3 Performance**: Meta's **Llama-3 70B** model has reached **top-5 on the Arena leaderboard**, surpassing many larger models. The 8B variant has also surpassed many larger models. [@lmsysorg](https://twitter.com/lmsysorg/status/1782483699449332144)
- **Training Details**: Llama-3 models were trained on **over 15T tokens of data** and aligned using **SFT, rejection sampling, DPO, and PPO**. [@lmsysorg](https://twitter.com/lmsysorg/status/1782483701710061675)
- **English Performance**: Llama-3 70B shows even **stronger performance in the English category**, ranking ~**1st place with GPT-4 Turbo**. It consistently performs well against top models by human preference. [@lmsysorg](https://twitter.com/lmsysorg/status/1782483701710061675)

**Microsoft Releases Phi-3 Language Models**

- **Phi-3 Model Details**: Microsoft released the **Phi-3** language models in 3 sizes: **phi-3-mini (3.8B), phi-3-medium (14B), and phi-3 (7B)**. Phi-3-mini **rivals Mixtral 8x7B and GPT-3.5** despite its small size. [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1782594659761389655)
- **Training Data**: Phi-3 models were trained on **3.3T tokens (mini) and 4.8T tokens (small/medium)** using "**heavily filtered web data and synthetic data**". [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1782598582731702764)
- **Benchmark Performance**: Phi-3-mini achieves **68.8 on MMLU and 8.38 on MT-bench**. Phi-3-medium achieves **78% on MMLU and 8.9 on MT-bench**, outperforming GPT-3.5. [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1782594659761389655), [@_akhaliq](https://twitter.com/_akhaliq/status/1782598582731702764)
- **Availability**: Phi-3-mini **weights were released under MIT license** on Hugging Face. It is optimized for use with Hugging Face text generation inference. [@_philschmid](https://twitter.com/_philschmid/status/1782781516172431685)

**Google's Gemini 1.5 Pro Achieves Strong Performance**

- **Gemini 1.5 Pro Performance**: Google's **Gemini 1.5 Pro API** now achieves **#2 on the leaderboard**, surpassing GPT-4-0125 to almost reach the top spot. It shows even stronger performance on longer prompts, ranking **joint #1 with GPT-4 Turbo**. [@lmsysorg](https://twitter.com/lmsysorg/status/1782594507957223720)

**Other Notable Releases and Benchmarks**

- **Hyper-SD from ByteDance**: ByteDance released **Hyper-SD**, a novel framework for multi-concept customization in image generation that achieves SOTA performance from 1-8 inference steps. [@_akhaliq](https://twitter.com/_akhaliq/status/1782601752417575423)
- **FlowMind from JP Morgan**: JP Morgan introduced **FlowMind**, which leverages GPT to automatically generate workflows for Robotic Process Automation (RPA) tasks. [@_akhaliq](https://twitter.com/_akhaliq/status/1782604054805332258)
- **Instruction Hierarchy from OpenAI**: OpenAI proposed an **Instruction Hierarchy** to make LLMs prioritize privileged instructions and be more robust to prompt injections and jailbreaks. [@_akhaliq](https://twitter.com/_akhaliq/status/1782607669376761989)

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Evaluating and Comparing Large Language Models**

- Discussions around the performance and benchmarking of the newly released **[Phi-3](https://arxiv.org/abs/2404.14219)** and **[LLaMA 3](https://llama.meta.com/llama3/)** models, with some skepticism expressed about **Phi-3's** evaluation methodology and potential overfitting on benchmarks like MMLU.

- Comparisons between **Phi-3**, **LLaMA 3**, **GPT-3.5**, and models like **Mixtral** across various tasks, with **[Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)** (3.8B) showing impressive performance relative to its size.

- Debates around the validity and usefulness of benchmarks like **MMLU**, **BIGBench**, and **LMSYS** for evaluating true model capabilities, with suggestions that they may become less reliable as models improve.

- Anticipation for the open-source release of **Phi-3** under an **MIT license**, along with its promised multilingual capabilities.

**2. Advancements in Retrieval-Augmented Generation (RAG)**

- LlamaIndex introduced **[DREAM](https://twitter.com/llama_index/status/1781725652447879672)**, a framework for experimenting with Distributed RAG, aiming to build robust, production-ready RAG systems.

- Discussions on innovative RAG techniques like **[Superposition Prompting](https://arxiv.org/abs/2404.06910)** for efficient long context processing, **[CRAG](https://twitter.com/llama_index/status/1782799757376963006)** for improving retrieval quality, and **[RAG with function calling](https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html)**.

- Sharing of resources on **[RAG evolution](https://arxiv.org/abs/2404.10981)**, **[credibility-aware generation](https://arxiv.org/abs/2404.06809)**, and integrating retrieval with LLM planning for structured outputs.

- Releases of open-source rerankers by **[@JinaAI_](https://twitter.com/llama_index/status/1782531355970240955)** to enhance RAG performance through improved vector search ranking.

**3. Fine-tuning and Optimizing Large Language Models**

- Extensive discussions on fine-tuning strategies for **LLaMA 3** using tools like **Unsloth**, addressing issues like tokenizer configurations, efficient merging of LoRA adapters, and embedding knowledge.

- Comparisons between full fine-tuning, **QLoRA**, and **LoRA** approaches, with **[QLoRA research](https://twitter.com/teortaxesTex/status/1781963108036088060)** suggesting potential efficiency gains over LoRA.

- Implementing mixed-precision training (**BF16/FP16**) for **llm.c** showing **~1.86x performance improvement** over FP32, as detailed in **[PR #218](https://github.com/karpathy/llm.c/pull/218)**.

- Optimizations in **llm.c** like CUDA kernel improvements (**GELU**, **AdamW**) using techniques like **thread coarsening** to enhance memory-bound kernel performance.

**4. Multimodal and Vision Model Developments**

- The introduction of **[Blink](https://arxiv.org/abs/2404.12390)**, a new benchmark for evaluating the core visual perception abilities of multimodal large language models like **GPT-4V** and **Gemini**.

- Releases like **[HiDiffusion](https://hidiffusion.github.io/)** claiming to increase diffusion model resolutions with a single line of code, and **[PeRFlow](https://github.com/magic-research/piecewise-rectified-flow/blob/main/README.md)** for upsampling images through flow integration.

- The unveiling of **[SEED-X](https://arxiv.org/abs/2404.14396)**, a multimodal foundation model bridging the gap by comprehending and generating images of arbitrary sizes for real-world applications.

- Advancements in **[Mixture-of-Attention (MoA)](https://snap-research.github.io/mixture-of-attention/)** architecture for disentangled, personalized image generation from language.

**5. Misc**

- **Perplexity AI's Valuation and Enterprise Pro Launch**: **Perplexity AI** hit a **\$1 billion valuation** following a successful funding round, as reported by [Bloomberg](https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-in-funding-round). They launched **Enterprise Pro**, a \$40/month offering with enhanced data privacy and management features, already used by companies like **Stripe, Zoom, and Databricks**. Discussions touched on data usage concerns and iOS app issues amidst anticipation for the [April 23rd announcement](https://x.com/aravsrinivas/status/1781902284844421624?s=46).

- **Hugging Face Downtime Disrupts Model Access**: Many channels reported **504 Gateway Time-outs** and service disruptions while trying to use **Hugging Face**, impacting functionalities like model search and download in tools like **[LM Studio](https://x.com/lmstudioai/status/1782390856986550384?s=46)**. Speculation pointed to possible term-blocking by Hugging Face to manage traffic, with a long-term fix to eliminate the dependency in the works.

- **Phi-3 and Llama 3 Models Generate Buzz**: The AI community actively discussed the **newly released [Phi-3](https://arxiv.org/abs/2404.14219) and [Llama 3](https://huggingface.co/mattshumer/Llama-3-8B-16K) models**. Phi-3 garnered attention for its efficiency and performance on benchmarks like **MMLU**, despite skepticism about overfitting. Llama 3 saw experimentation with different variants and quantizations, alongside challenges with the tokenizer and context size. The models' potential for fine-tuning and integration with various tools was a hot topic.

- **Retrieval-Augmented Generation (RAG) Gains Traction**: Conversations delved into **evaluating and enhancing RAG systems**, from using **[LlamaIndex](https://twitter.com/llama_index)** for finance bots to introducing frameworks like **[DREAM](https://twitter.com/llama_index/status/1781725652447879672)** for distributed experimentation. Techniques such as **[superposition prompting](https://arxiv.org/abs/2404.06910)**, credibility-aware generation, and **[function-calling RAG](https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html)** were discussed, alongside the creation of RAG benchmarks that synthesize information from multiple documents.

Let me know if you would like me to elaborate on any part of the summary or if you have additional questions!

---



# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LLaMA Leaps with Unsloth's Support**: The **LLaMa 3 Instruct Model** sees advancements with a [Hugging Face upload](https://huggingface.co/unsloth/llama-3-70b-Instruct-bnb-4bit) promising speed and memory improvements. Meanwhile, members share success in fine-tuning this model using Unsloth with a single 24GB GPU at BF16, maintaining quality within limited VRAM constraints.

- **AI Ergonomics Isn’t Just about Code**: Discussing the physical aspects of deep work, engineers exchanged ergonomic setup tips, signaling the value of standing desks and specialized keyboards like the [Advantage2](https://kinesis-ergo.com/shop/advantage2/) in maintaining productivity.

- **Multilingual Models Spotlight**: Showcases included **Swedish** and **Spanish** adaptations of language models, such as the [llama-3-instruct-bellman-8b-swe-preview](https://huggingface.co/neph1/llama-3-instruct-bellman-8b-swe-preview) and **solobsd-llama3**. The **Ghost 7B Alpha** model also made an appearance, with tools and documents found [here](https://ghost-x.vercel.app/docs/models/ghost-7b-alpha).

- **Chatter about Phi-3 and Quantization**: Excitement bubbles around Microsoft's **Phi-3 Mini 4K Instruct model** with quantitative musings on 4-bit implementations. A community member's deployment of Phi-3 on Hugging Face is available [here](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit).

- **Finetuning Finesse and Framework Fixes**: Conversations revolved around the optimization of model fine-tuning practices and the identification of **tokenizer issues**, alongside community members detailing strategies for embedding knowledge into LLMs for instructional use and aligning with Unsloth's methodology.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexity AI Hits $1 Billion Valuation**: After a successful funding round, **Perplexity AI** has been valued at a whopping **$1 billion**, even appearing in [Bloomberg](https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-in-funding-round) articles, with potential collaborations hinted involving AI expert Yann LeCun. The enterprise version, dubbed **Perplexity Enterprise Pro**, boasts enhanced data privacy and management features, drawing attention from major companies.

**New Product Launch Brings Expectations and App Woes**: The launch of Perplexity AI's **Enterprise Pro** for $40/month has stirred excitement and anticipation for possible upcoming features, although some frustration was voiced over technical difficulties with the iOS app on iPads. Despite the issues, the enthusiasm suggests high expectations from the current user base.

**Data Privacy Takes Center Stage**: In light of the Enterprise Pro introduction, users discussed data privacy concerns, prompting moderator references to official statements about user consent for data use in models. Separately, the sharing channel instructed users on compliances necessary to share Perplexity AI's search threads.

**Anticipation Grows for Perplexity's High Valuation Fundraise**: Community conversations buzzed about Perplexity AI seeking to raise **$250 million** at a **$2.5 to $3 billion valuation**, as members shared a [TechCrunch article](https://techcrunch.com/2024/04/23/perplexity-is-raising-250m-at-2-point-5-3b-valuation-ai-search-sources-say/) and a [CNBC interview](https://www.cnbc.com/2024/04/23/cnbc-exclusive-cnbc-transcript-perplexity-founder-ceo-aravind-srinivas-speaks-with-cnbcs-andrew-ross-sorkin-on-squawk-box-today.html) with CEO Aravind Srinivas, signifying rapid company growth and market interest.

**API User Looks for Cutting-Edge Features**: A request on the **pplx-api** channel highlighted a thirst for an API providing up-to-date web information, like GPT but with browsing capabilities; **Perplexity's sonar online models** were recommended, found in their [documentation](https://docs.perplexity.ai/docs/model-cards), with additional advice on prompt enhancement for improved model performance.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Forge WebUI Attracts New Users**: A newcomer to **Stable Diffusion** is exploring **Forge Webui** as a starting interface, while the community debates on various alternatives for creating AI-generated images and assets, including game and sci-fi elements.
- **CUDA Conundrums and Speedy Solutions**: Technical discussions are focusing on troubleshooting issues like **CUDA errors** and prompts for improving generation speeds, with frustration expressed over missing nodes in **ComfyUI** and compatibility queries about models across platforms.
- **AI Fantasies and Dream Generation**: Some whimsical exchanges propose using AI to design perfect partners or ideal homes, showcasing the enthusiasm for AI's potential in crafting highly personalized content.
- **Stable Diffusion v3 Buzz**: There’s a mixture of excitement and skepticism about **Stable Diffusion version 3** as users await its release, discussing insider insights from the former CEO **Emad** and debating the software's true openness.
- **Community Swaps Technical Tips and Tricks**: Ongoing conversations reveal a community keen on solving practical issues like system installations transfers across drives, as they collectively navigate the evolving landscape of Stable Diffusion and its applications.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Tensor Parallel on the Vanguard**: Engineers discussed the potential of **tensor parallel** implementation in Very Large Language Models (VLLMs), with an expectation for **jamba** support to potentially skyrocket performance. Concerns include the proper management of contexts within **Claude 3** and **Big-AGI** to balance costs, with [memGPT](https://memgpt.ai/) and [SillyTavern SmartContext](https://docs.sillytavern.app/extras/extensions/smart-context/) as cited approaches.

- **AI's Groove in High-Definition**: Members shared remastered music videos, including the Beastie Boys and deadmau5 & Kaskade, along with a humorously encoded latent version of CIFAR100, titled [latent-CIFAR100](https://huggingface.co/datasets/Verah/latent-CIFAR100). A need for larger image classification datasets was recognized after testing on a 4x4x4 latent dataset, and scholarly papers [like this one](https://arxiv.org/abs/2402.10588) were shared to enrich discussions on language models and symbolic representation.

- **Toolkit Triumphs and Benchmark Brinkmanship**: DeepMind's [Penzai](https://github.com/google-deepmind/penzai) enters the scene, offering a JAX-based toolkit for neural network manipulation. Meanwhile, debates ensue on the validity of the LMSYS benchmark as noted in a [skeptical Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1c9nvpy/lmsys_becoming_less_useful). Rubik.ai threw its hat into the ring, calling for beta testers for a research assistant utilizing **Claude 3 Opus** and **GPT-4 Turbo**.

- **Model Magnification and Downtime Debacles**: The **Phi-3-mini** model was juxtaposed against [LLaMA-3](https://x.com/sebastienbubeck/status/1782627991874678809?s=46), and **GPT-3.5**, sparking debate over its quantization performance and anticipation for model weights. Hugging Face's hiccup, possibly linked to heavy **LLaMA-3** use or the **[FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)**, was a topic, while **QLoRA vs. LoRA** fine-tuning approaches were compared for efficacy.

- **The Quest for Optimal LLM Utilization**: Members shared woes and wins of navigating **Deepspeed Zero 3**, pondered **single-GPU optimization versus NVLink**, and sifted through guidance for Llama fine-tuning best practices. The community clearly values specific fine-tuning guides, with Hugging Face's blogs and **Labonne's GitHub** recommended over generic Medium articles.

- **Vision Benchmark Unveiled**: Attention turned to **RealWorldQA**, an **xAI** benchmark dataset designed for **Grok-1.5-vision-preview**, generating interest within the **Obsidian** community. The nature of the dataset was clarified as a benchmark, not a training set, as highlighted in an [xAI blog post](https://x.ai/blog/grok-1.5v), though a yearning for training datasets remains.

- **Revealing RAG Revelations**: The community examined **Retrieval-Augmented Generation (RAG)** through the lens of LLaMA index performance, superposition prompting methods detailed in this [Superposition Prompting Paper](https://arxiv.org/abs/2404.06910), and other papers shared on enhancing RAG credibility. Function-calling RAG implementations were also spotlighted, featuring resources like Pamela Fox's [blog](https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html).

- **Simulating Worlds Beyond Imagination**: While **WorldSim** was offline, alternative simulations such as **Super WorldSim** and **Snow World Simulator** found a home in **HuggingChat**. Collaborative world-building efforts are thriving on Discord, with a focus on open models like **Llama 3's** upcoming releases to enrich the simulated experience.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPU Gaffes and Glitches**: Discussions around **LM Studio's performance** on AMD and Nvidia GPUs uncovered that **GPU offloading** is essential to avoid 100% CPU utilization and prevent system inefficiency. Solutions for "Error loading model" issues focused on turning off GPU offloading or setting specific environment variables to direct **LM Studio** to use dedicated GPUs.

- **Hugging Face Hiccups**: Users encountered **503 and 500 error messages** due to Hugging Face API downtime, affecting LM Studio's ability to search and download models. While the community speculated on potential term-blocking by Hugging Face to alleviate traffic, ongoing communication through [LM Studio Tweets](https://x.com/lmstudioai/status/1782390856986550384?s=46) keeps everyone updated.

- **Model Mania**: A variety of **AI models** sparked debate, with discussions on **Meta-Llama-3-8B-Instruct-GGUF**'s infinite generation issue, finetuning **Llama 3** versus **Goliath 120B and Mistral**, and **Phi-3's** surprising efficiency. Queries about integrating tools like **Autogen** with LM Studio and concerns over model restrictions in content generation highlighted users' desire for customization.

- **Prompt Puzzles and Config Curiosities**: LM Studio users shared tips on crafting system prompts for **D&D scenarios**, addressed **Llama-3-Smaug-8B** prompt concerns, and recommended preset configurations. Meanwhile, an **Autogen** snag involving a 2-token limit issue prompted advice for troubleshooting from the community.

- **Tech Trials and ROCm Reviews**: AMD GPUs using ROCm sparked reviews of **Meta-Llama-3's** performance, with noted speeds and questions about running large models on lower-end hardware. Resourcefulness reigned with strategies on resolving AMD GPU selection in LM Studio, and [Hugging Face repository details](https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF) were shared for leveraging **Meta Llama 3 models** effectively.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **X11 Steps Up for Remote GPU Profiling**: The CUDA guild explored **X11 forwarding** to operate [Nsight Compute GUI via SSH](https://goteleport.com/blog/x11-forwarding/), with a user sharing a [tutorial for setting up Nsight Compute remotely](https://tspeterkim.github.io/posts/nsight-setup-on-ec2). Meanwhile, **'Effort' algorithm** adds dynamism to LLM inference computations and piques interest for use with Triton or CUDA, with [its code available on GitHub](https://github.com/kolinko/effort).

- **CUDA Matrix Magic and Thread Sync Discussions**: In the CUDA channel, users clarified concepts like **CUDA matrix multiplication** and the behavior of `__syncthreads()` in CUDA; notably highlighting architectural changes starting with Volta. Inline functions were demystified with discussions around `__forceinline` and `__inline`.

- **Triton Tackling Transforms & Memory Management**: Triton users faced challenges with image grayscaling and memory fragmentation, while others debated binary search implementation strategies due to current limitations. The `make_block_ptr` parameter's **order** caused confusion, steering the conversation to row-major versus column-major formats.

- **PyTorch Practices**: In the Torch channel, the guild confirmed that operations like `torch.nn.conv2d`, `torch.nn.relu`, and `torch.nn.batchnorm` are executed on the **GPU without CPU-GPU transfers** for intermediate results. GPU operations scheduling is noted to be asynchronous.

- **Optimizing with CUTLASS**: A heads-up for **Lecture 15** on CUTLASS revved the engines of keen learners, promising deeper dives into CUDA's cutting-edge tools and techniques.

- **Algorithms, Beginnings, Book Clubs, and Beyond**: Sparse discussions touched on a CUDA algorithm example, beginners' journey to mastering CUDA with **entertaining styles**, PMPP book chapter exercises, potential YouTube recording uploads, and mentions of JAX memory issues in implementing a denseformer. The hqq channel discussed significant **Triton kernel benchmarks** with a push toward efficient quantization strategies.

- **Kernels, Coarsening, and Collaboration in the Engine Room**: The llmdotc channel was ablaze with intense talks on **atomic operation removal**, BF16/FP16 mixed precision gains, demands for current CUDA versions, and coalescing insights to double GELU and AdamW kernel performances. Thread coarsening shone as a beacon of hope for optimizing memory-throttled kernels.

- **Moderation, Technical Setups, and FlashAttention**: Moderators donned their capes to manage content, while the massively-parallel-crew channel buzzed with plans to smooth out **event recordings** and future talks preparation, including a shout-out for a deep-dive on FlashAttention.

- **Local GPU Enthusiasts Convene**: In a lighter moment, the off-topic channel revealed a pleasant meetup of members living in the vicinity of Münster, celebrated as a hub for CUDA enthusiasts.

- **Ring Attention Gains Attention**: The ring-attention channel piqued curiosity through a brief mention of manual placement triumphs and tinyllama tests shared via an [Axolotl GitHub link](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Local LLMs on Smartphone Horizon:** Discussions explored the feasibility of running large language models (LLMs) on smartphones, considering memory bandwidth (up to 51.2 GB/s) and GPU capabilities (Exynos 2400 chipset specs), suggesting even 7-8B models might be workable. Community members examined existing apps like [MLC-LLM](https://github.com/mlc-ai/mlc-llm) and discussed how Hugging Face's downtime raises questions about free AI model hosting sustainability.

**SpaceByte Makes Tokenization Obsolete:** A new byte-level LLM architecture, [SpaceByte](https://arxiv.org/abs/2404.14408v1), promises to eliminate the need for tokenization, addressing potential information leakage from tokenizers. Other discussions critiqued Fineweb's relation to LLaMA and the novel application of ProGen2 for AI-designed CRISPR-Cas proteins, showcasing LLMs' role in accelerating scientific discovery.

**Scale Wisely with Tactful Debates:** A clash over data rounding in a publication sparked wider conversation about constructive criticism and tone in technical debates. The skirmish illuminated misunderstandings around attributions of rounding data to the Chinchilla paper versus the replication team, unraveling deeper issues in replication methodologies.

**RWKV Integration Ramps Up:** GPT-NeoX developers are busy implementing RWKV (Rethinking Weighted Key-Value Memory Networks) with support for fp16 and JIT kernel compilation. Progress and tasks are detailed in [GitHub Issue #1167](https://github.com/EleutherAI/gpt-neox/issues/1167), and developers are pushing for a version numbering system to streamline the iteration process.

**AI Designs High-Performance Proteins:** Profluent Bio successfully employed LLM ProGen2 to design new CRISPR-Cas protein sequences, yielding variants with increased specificity. The accomplishment demonstrates LLMs' expanding utility in biotechnology sectors.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Chatting with PDFs, Now with Math!**: [ai_pdf](https://github.com/Crizomb/ai_pdf) is an open-source project enabling conversations with PDF documents, excelling with math PDFs by converting them to LaTeX.

**Voice Directed AI Artistry**: A 2.5-minute video generated in real-time from voice commands has been shared on [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1c8oea6/endlessdreams_voice_directed_realtime_videos_at/), pointing towards a future of AI-driven dynamic video creation.

**AI Gets Reasonable**: [Transformers.js](https://xenova.github.io/transformers.js/) allows running HuggingFace Transformers directly in the browser, expanding the playfield for AI applications in web environments.

**Rust Helps Minify BPE**: `minbpe-rs` is a Rust port of `minbpe` with functions for tokenization and training, improving performance for NLP tasks. The project is available on [GitHub](https://github.com/gnp/minbpe-rs).

**Diffusion Dilemmas and AI Video Debates**: Users discuss the feasibility of creating a 1-minute video on "AI Horse" using Diffusion, and others tackle various implementation challenges, demonstrating the teething issues of burgeoning AI applications.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Code Instructions Boost Hermes**: After integrating [code instruction examples](https://link.to.examples), **Hermes 2.5** has been observed to outperform **Hermes 2** in various benchmarks, with notable improvements in metrics such as the MMLU benchmark score.

**Mistral's Capacity Challenge**: Discussions concluded that **Mistral** cannot be scaled beyond 8k without ongoing pretraining. Focus shifts to enhancements in model merging strategies, such as applying differences between **UltraChat** and base **Mistral** to **Mistral-Yarn**.

**Empathy in AI**: The **Open Empathic** project seeks assistance in expanding categories; contributors are guided by a [YouTube tutorial](https://youtu.be/GZqYr8_Q7DE) and encouraged to leverage movie scenes from YouTube for diversity in empathic response training.

**Mojo Delights in Differences**: Clarifications were made on **Mojo** around parameters and arguments with the latter being runtime values, while parameters in the language remain compile-time constants. Complex patterns like 'Type State' are being explored, and performance comparison to Python reveals ongoing efficiency issues, notably in IO operations.

**In the Trenches with Mojo SIMD and Multithreading**: Implementing SIMD patterns in **Mojo** yielded close performance to Rust in a CPU-limited context. However, optimization challenges exist, such as the best practices for `parallelize`. In other discussions, the use of `UnsafePointer` and the phasing out of `LegacyPointer` indicate a maturation of memory handling within the language.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **BOS Token Bug Squashed**: Engineers examined an issue with LLaMa 3 not adding BOS tokens correctly during fine-tuning; [a solution](https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41) was discovered via a Pull Request that modifies `tokenizer.json`.

- **Phi-3 Models Outpunch Their Weight**: Despite their smaller size (around 3.8b parameters), Phi-3 models are showing comparable performance to larger counterparts, indicating a high efficiency. They come with an open MIT license, yet might prioritize reasoning abilities over extensive knowledge.

- **GPU Demands for Training AI Under the Lens**: The discussion spotlighted the immense resources needed for AI model training, mentioning a specific setup with 512 Nvidia H100-80G GPUs running for a week, magnifying the computational intensity of such tasks.

- **LLaMa's Extended Reach is No Joke**: A member showcased [Llama 3](https://huggingface.co/mattshumer/Llama-3-8B-16K), a model that boasts a 16K token length, sparking excitement for its enhanced capacity for processing longer sequences.

- **The Roadblocks and Workarounds of AI Development**: Conversations surfaced issues with Discord link sharing, problematic 8-bit optimizer configurations, and a lengthy 1.5-hour model merging process; there were also shared efforts for guidance on using Unsloth with Axolotl for optimized training.

- **Dataset Mastery and Markdown Mysteries**: Participants shared how specifying `"type: sharegpt"` in YAML affects dataset operations and sought [documentation](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) on different dataset formats provided by Axolotl. Concerns about GitHub's rendering of **qmd** files over traditional Markdown were also voiced.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Optimizer on the Move**: Performance issues with **Wizard 8x22b** due to heavy traffic are being mitigated by optimizing the load balancer, which should lessen latencies.

- **Routing Towards Efficiency**: Following the deletion of **Databricks: DBRX 132B Instruct (nitro)**, traffic will be rerouted to the main [Databricks DBRX 132B Instruct model](https://openrouter.ai/models/databricks/dbrx-instruct), and **OpenRouter** announced the introduction of three new models, including **LLama 3 finetune**, with updates to prompt formatting and solutions to regional network hiccups focusing on dynamic routing enhancements.

- **Mitigating Model Mishaps**: Sporadic performances of **WizardLM-2** have been flagged by users, with *SillyTavern's Assistant Prefill* complicating interactions with **LLaMA 3** models, and a hotfix has been issued for Hugging Face's tokenizer service downtime, with a long-term resolution in the works.

- **Financial Viability in AI Model Provision**: There's a lively debate about the financials of providing AI services, particularly the affordability of rates and the cost differentials compared to image generation models. Discussions span **FP8 quantization**, active worker discounts, and the economic footprint of **Groq's** hardware.

- **Enhancing Contract Interaction**: Suggestions in the **#[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1231369890829439056)** channel include urging users towards contract standard awareness, implementing localization for legal relevance, and incorporating a feature for illegal terms detection, as well as the introduction of [Keywords AI](https://keywordsai.co) and [DeepGaze](https://www.deepgaze.ca/), both leveraging **OpenRouter**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Robo Creep Factor**: Engineers engaged in debate over the **Atlas robot**'s release, with anticipation for its market capabilities and underlying strategies, while grappling with its unsettling "creepiness" that sparks social media.

- **AI Divinity Discourse**: A vigorous discussion unfolded about the possibility and implications of AI spirituality, including reflections on AI consciousness, tempered by community rules on secular discourse.

- **API Crafting and Interface Upgrades**: Conversations around **MyGPT** and other tools like **MetaGPT** and **Devika** delved into their potential to craft APIs and improve app development, with interest in automated GitHub interactions.

- **Model Performance Mixed Bag**: **LLaMa 3** elicited mixed reactions on performance among the engineers, with skepticism cast on rumored **GPT-5** release dates. Additionally, there was a call for high-quality literature on generative AI, citing both **OpenAI's published papers** and repositories such as Arxiv.

- **Prompt Engineering Nuanced Discussion**: Engineers exchanged strategies on the art of prompt optimization, debating the merits of brief custom instructions and discussing the ethical side of sharing techniques. The conversation also encompassed email improvement through GPT-4 and the absence of a comprehensive prompt library.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Multimodal Model Frets Over Fitting**: Existing **multimodal datasets**, which total around 2 million pairs, risk causing overfitting in models such as GPT-4v, particularly with LAION-COCO captions, where models show a worrying trend of memorization rather than learning.
  
- **Innovations and Concerns in Image Handling and Surveillance**: The release of **Adobe Firefly Image 3** has sparked interest due to its improved image generation and integration with Photoshop. Meanwhile, concerns about AI-driven surveillance bots on Discord were addressed with the introduction of [kickthespy.pet](https://kickthespy.pet/#823813159592001537), which uses an API to detect such bots. 

- **The Next Wave in Visual Perception & Upscaling**: **Blink**, a benchmark for **multimodal LLMs** like GPT-4V and Gemini, has arrived, challenging models with tasks requiring visual perception capabilities. In image handling, both **Piecewise-Rectified Flow (PeRFlow)** and **HiDiffusion** are making strides; however, HiDiffusion's artifact issue in high-resolution images remains a point of concern ([Read more about Blink](https://arxiv.org/abs/2404.12390)).

- **Pushing the Multimodal Envelope**: The conversation around multimodal models continued, with a new architecture, **Mixture-of-Attention (MoA)**, being introduced, promising enhanced disentanglement in personalized image generation ([described in this paper](https://snap-research.github.io/mixture-of-attention/)). The **SEED-X** multimodal foundation model also generated buzz with its ability to handle images of variable sizes, focusing on comprehensive understanding and generation.

- **Collaboration Call in Code**: An open call for collaboration to build an NLP coding assistant targeting **JavaScript/Rust** frameworks caught traction in the guild, with *softmax_function* showing occasional support despite a tight schedule across multiple projects.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**DREAM Big with Distributed RAG**: LlamaIndex introduces **DREAM**, a Distributed RAG experimentation framework, while also launching various RAG enhancements like **ColBERT with a Twist** and **LoRA Fine-Tuning**. Dig into the discussions about **CRAG**, an innovative layer improving RAG retrieval, and open-source rerankers in [LlamaIndex tweets](https://twitter.com/llama_index).

**Using AI Models Beyond OpenAI**: Within **#[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1231156600190795826)**, users tackle different retrieval methods for LLMs, while addressing integration bugs and API key annoyances. There's a spotlight on techniques for improved context management and interest in using alternatives to OpenAI's options, as detailed in numerous [LlamaIndex docs](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/).

**From LinkedIn to Google Sheets, AI Funding Data Draws Interest**: A member shares an **Infini Attention** explainer on LinkedIn, while AI funding distribution by city is accessible on [Google Sheets](https://docs.google.com/spreadsheets/d/1nWBP1MpT7sACYDxqdCo8gBR7b2nXJbrF9Z43y69q9hg/edit#gid=752020121). New **LLM-Ready Markdown** integrations excite the community, and WhyHow.AI's boosted Knowledge Graph SDK invites beta testers on [Medium](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3).

**Database Debates and Fine-tuning**: Members in **#[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1231221804518080615)** actively debate database types optimal for LLM training. They underscore the importance of understanding database schema and vector store possibilities when training large language models.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Caught a Case of the Compatibility Blues**: Members noted that **Open Interpreter**, despite successful implementations, encountered challenges with Windows and mix-ups regarding model support, specifically clarifying that **OI** currently only supports **OpenAI** for the cloud option, not **Groq** or the **Llama 3 70b** model. They also discussed stability issues with the **Llama 3 70b** compared to its 8b counterpart.

**Say What, Interpreter?**: Various functionalities and integration challenges with **Open Interpreter** were highlighted, such as installation issues on Windows systems and pytesseract errors, the latter mitigated by using `pip install --upgrade litellm`. Detailed troubleshooting videos, e.g., on YouTube for integrating OI with **GROQ API**, show community eagerness for cost-effective solutions.

**Screen Vision, but No Prophecy**: In the AI vision domain, it was clarified that **Open Interpreter** leverages the **GPT-4-vision-preview** for screenshot recognition tasks, indicating a mix of text and vision capabilities within the tool.

**Helping Hands and Config Stands**: The community celebrated reaching **100 GitHub contributors** for **Open Interpreter** and displayed a strong collaboration spirit. There’s a push for sharing default configuration files, as seen in a [pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1204), to improve interactions with various models.

**M1 Mac Spacebar Conspiracy**: Specifically, for M1 Mac users, troubleshooting a recording issue where pressing the spacebar didn't work as intended, diverse solutions were proposed, including installing **ffmpeg**, checking microphone permissions or switching Python versions using **conda**.

**Cloudy with a Chance of Compatibility**: There's a desire among members to see **OI** aligned with cloud services, with calls to enable compatibility for broader cloud platform support, including but not limited to platforms like **brev.dev** and **Scaleway**.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Clickbait vs. Substance**: The debate over AGI article titles in the community reflects a push for engaging yet truthful headlines. The discord in opinions, varying from AGI's ontological status to being a faith, indicates a search for thought-provoking yet honest discourse, as illustrated by titles like "AGI Isn't Real" and Mistral CEO Arthur Mensch's interview in [Business Insider](https://www.businessinsider.com/mistrals-ceo-said-obsession-with-agi-about-creating-god-2024-4).

**Phi-3 Under the Microscope**: There is skepticism around the integrity of the **Phi-3** benchmarks due to perceived overfitting on benchmarks like the **MMLU**, calling into question their relevance for OOD performance. Criticism also extends to the model's evaluation presentation and undisclosed data pipelines, amidst excitement for Phi-3's anticipated MIT license release and multilingual capabilities.

**Benchmarking Evals**: The utility of AI model evaluations is scrutinized, noting the trade-offs between automated benchmarking tools like MMLU, BIGBench, and human-intensive evaluations like ChatBotArena. Perplexity-based evaluations, like AI2's Paloma, were confirmed to be more for internal training checkpoints rather than public competitions.

**Discord Community Dynamics**: Anecdotes about the community include a researcher's ephemeral tweeting habits, the surprising low membership despite free subscription, and candid aspirations of engaging with industry figures like Ross Taylor post NDA-laden periods.

**A Tangle of Instruction and CRINGE**: The ecosystem of instruction tuning is expounded with references to an [introductory blog](https://gaotianyu.xyz/blog/2023/11/30/instruction-tuning/) and appreciation for the classification in the MT Bench paper. Additionally, the CRINGE paper's novel training approach using negative examples gains attention and is further discussed in relation to instruction tuning.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Project Spotlight**: An [open-source matchmaking application](https://x.com/anmol_desai2005/status/1781679469679325605?s=46&t=vUJbpAOoGDUfvrA5TGBjTQ) was announced, integrating **@cohere Command R+**, **@stanfordnlp DSPy**, **@weaviate_io Vector store**, and **@crewAIInc agents**. Its GitHub link was shared for community feedback.
  
- **AI-Enhanced Job Search Tactics**: Engineers discussed that **personal projects** and having **big company names on resumes** often supersede actual work experience for securing job interviews.

- **Refining AI with Context**: Engineers broached constraining AI responses to a given topic using **preambles** and **BOS/EOS tokens** to ensure outputs remain within the intended training scope.

- **Web Scraping Headaches**: Development of a **generic web scraper** leveraging **gpt-4-turbo** for identifying (selector, column) pairs was debated, with the complexity of model interaction with web elements proving challenging.

- **Cohere Enthusiasts Seek Expansion**: The engineering community showed strong interest in integrating **Cohere Command-r with URL Grounding (RAG)** into **BotPress**, hinting at a potential user shift from ChatGPT to Cohere if successfully implemented.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Webpage Wizardry with LLM Scraper**: The newly unveiled [LLM Scraper](https://github.com/mishushakov/llm-scraper/) on GitHub presents a method to transform any webpage into structured data, leveraging LLM's parsing capabilities, and cacheing previous replies to subsequent requests.

**Stock Analysis at Your Fingertips**: [AllMind AI](https://www.producthunt.com/posts/allmind-ai-your-personal-stock-analyst), an AI tool that promises speedy and economical financial insights, is gunning for the top spot on Product Hunt.

**Automated Graphs Get Smarter**: WhyHow.AI has rolled out a major upgrade with **schema-controlled automated knowledge graphs**, aiming to structure user-uploaded content more efficiently. The new feature and its beta program were introduced on a [Medium post](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3).

**Conversational Query Crafting**: A [blog post](https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever) breaks down how the **Self-querying retriever** creates structured queries from natural language inputs, enhancing semantic similarity searches with filtering based on metadata.

**Watermark Warnings for LLMs**: The community delved into the concept of watermarking in AI-generated texts, a technique for planting identifiable patterns, as detailed on this resource page: [Watermarking LLMs](https://watermarking.aisimplyexplained.tech/).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**TinyGrad Tackles Segfaults and Training Woes**: Discussions highlighted challenges with setting up **tinygrad** post-**ROCm 6.1** release due to segfaults, while George Hotz assured that the `master` branch is stable thanks to robust CI.

**AI Hardware Hyped to Outperform Cloud**: The community debated the merits of decentralized AI services like **TinyBox** against traditional cloud services, focusing on points such as censor resistance, local training feasibility, and the importance of real-time user data training.

**Inside TinyGrad's Mechanics**: In the realm of **tinygrad**, members dove into deep discussions about **stacking tensors**, **shape tracking**, and **memory management**, exchanging tutorials and documentation that reveal the innards of the minimalist deep learning library.

**Windows Walks a Tightrope with CUDA**: Windows users shared their experiences and workarounds for running **tinygrad** with **CUDA**, using tools like **WSL** and **Docker**, while acknowledging the platform's official unsupported status for this setup.

**George Hotz Chronicles Upcoming Tinygrad Evolutions**: In a weekly roundup, Hotz mentioned focus areas for upcoming discussions, highlighting **mlperf** progress, potential **NVIDIA CI** strategies, and the goal of keeping the **tinygrad** codebase succinct.

[ShapeTracker Tutorial](https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html), [Uops Documentation](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops-doc.md), and [CUDA Tensor Core Guide](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/cuda-tensor-core-pt1.md) were shared as educational resources, while [Meta AI](https://meta.ai) was cited in the discussion.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Mixtral Edges Out Llama3**: **Mixtral-8x7B-Instruct-v0.1** demonstrated superior performance to **Llama3 70b instruct** in a German RAG evaluation, according to shared [dataset results](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval). However, members noted potential issues with the evaluation metrics, especially the "question to context" metric, and suggested a possible formatting bug in the query template which might impact results.

**Enhancing Chatbots with Execution Models and Haystack**: Armifer91 is prototyping an "execute_model" function for chatbots, grouping certain functionalities and paralleling the **MoE** approach, while a GitHub [notebook](https://github.com/vblagoje/notebooks/blob/main/haystack2x-demos/haystack_rag_services_demo.ipynb) illustrates using the **Haystack LLM** framework for dynamically invoking services. Developers are exploring improvement techniques for **Llama** related to tokenization for fine-tuning, despite facing platform instability complaints with **Hugging Face**.

**Whispers of German Speech Recognition**: Members are trialing various **Whisper** models for German speech recognition such as [whisper-tiny-german](https://huggingface.co/primeline/whisper-tiny-german) and [whisper-base-quant-ct2](https://huggingface.co/jvh/whisper-base-quant-ct2/), with a consensus on potential finetuning or quantization for enhanced functionality on smartphones.

**Template Troubles and Tokenization Tangles**: Complexities related to templates and tokenizer configurations in **Llama-3** models were prevalent in discussions, with talk on zero weights for special tokens and alternative eos_tokens in conversational contexts. The **ChatML** template is standard, yet there are tokenizer-related challenges.

**DiscoLM's German Precision Problem**: Fine-tuning **DiscoLM** for German language applications prompted debates over the model's tokenization issues and potential strategies for improvement, with **Instruct** model serving as a possible foundation. Suggestions were made to follow the **LeoLM** training approach and connect with the **occiglot** team to bolster **Llama3's** performance in German.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Expanding the LLM Horizon**: Engineers debated the prospect of using **rope** to expand large language models' context window, showing enthusiasm and referencing a [Perplexity AI article](https://www.perplexity.ai/search/why-not-scale-0KMvWZKqSVGnYIBd_vpcng) for in-depth understanding.

**FineWeb Stirs Excitement**: The announcement of **FineWeb**, a massive web data trove of 15 trillion tokens drew attention, with expectations high due to its superior performance markers over predecessors like RefinedWeb and C4, as disclosed on [Twitter](https://twitter.com/gui_penedo/status/1781953413938557276).

**Frameworks in Focus**: Discordants shared mixed feelings about the **Hydra framework**, with some appreciating its sophisticated application configuration capabilities, while others pondered over its distinctions; interest peaked with references to [Hydra's GitHub repository](https://github.com/facebookresearch/hydra).

**Microsoft's Mighty Phi-3 Emerges**: **Phi-3** sparked interest with its release—operating at a grander scale than its predecessor, Phi-2, and speculated to compete with notable models like **llama 3 8B**; speculations fueled by insights shared through a [Tweet on Phi-3's capabilities](https://twitter.com/arankomatsuzaki/status/1782594659761389655).

**Perplexity.ai Makes a Financial Leap**: The technical crowd took note of **Perplexity.ai**'s successful fundraising round, touted to enhance its search engine prowess—announcement revealed in a [Tweet detailing the $62.7M fundraise](https://twitter.com/AravSrinivas/status/1782784338238873769).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **70b Beats 8b in Llamafile Matchup**: Users indicated that the **llama 3 70b** is the go-to choice over 8b for integration with Llamafile, citing inoperability issues with the latter and highlighting that the 70b Q2 weights are a manageable 26GB in size.
- **Mixed Results with M1 Pro Quantization**: An issue was reported where the Q2 variant of the llama model gave scrambled output on the **M1 Pro system**; however, it was clarified that the model runs smoothly in **CPU mode**, although at a slower pace.
- **Android's Address Space Limitation Stumps Llamafile**: Discussion around running llamafile on Android was thwarted by the limitation that Android lacks a **47 bit address space**, making support for it currently unattainable.
- **Redis Pioneer Praises Llamafile**: The inventor of **Redis** expressed approval for the llama3 70b version of Llamafile on Twitter, a commendation that received celebration from the Llamafile community.
- **Port Prowess for Multimodal Models**: Inquiries about operating multiple instances of llamafile led to advice on employing the `--port` flag to specify different ports for concurrent model runs.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Surprise in Context Size**: A revelation from 4chan highlighted that a certain AI might have been operating with a **32k context size** throughout, challenging previous assumptions about its capabilities.
  
- **Alternate Methods to Model Scaling**: A member brought up Alpin's non-traditional approach to scaling AI models, highlighting strategies like **dynamic ntk** and **linear scaling**, which could potentially maintain effectiveness without requiring 'rope.'

- **Matt Rolls Out 16k Config for Llama**: Posted on Hugging Face was **Matt's 16k configuration for the Llama model**, including parameters such as "max_position_embeddings": 16000, and the model type specified as "llama". Configuration details available [here](https://huggingface.co/mattshumer/Llama-3-8B-16K/blob/main/config.json).

- **Medical Knowledge Made Accessible**: Engaging discussions focused on simplifying medical knowledge; suggestions ranged from fine-tuning an LLM for simplicity, to developing an agentic system that decomposes tasks into specialized stages, eventually translating medical summaries into layman's terms.

- **OCR Data Hunt for Lesser-Known Languages**: A request was made for an OCR dataset supporting less-popular languages, preferably containing document-type data, indicating ongoing efforts to increase AI's linguistic reach and accessibility.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Meta AI's 'Imagine' Grips Engineer Interest**: **Meta AI's 'Imagine'** has sparked excitement among guild members, with one calling it *insane* and prompting requests for specific examples that showcase its capabilities.
  
- **Finding the Right Dev Tools**: Members are actively looking for tried-and-true **development tools** suitable for work with **Large Language Models (LLMs)**, signifying a keen interest in optimizing their workflows.

- **Azure OpenAI Service Stutters**: Users are expressing frustration with **Azure OpenAI**, reporting significant **latency** with requests sometimes taking upwards of 20 minutes, and encountering rate-limiting issues when making more than two requests within a 15-second window.

- **Identifying the Azure Lag Source**: Some suspect that Azure's latency issues may be due to temporary **service problems**, rather than being a consistent issue with the platform.

- **Real-Time API Response Tracking Tool Shared**: A practical resource, [GPT for Work's response time tracker](https://gptforwork.com/tools/openai-api-and-other-llm-apis-response-time-tracker), was shared to monitor **API response times** of major LLMs, which could be instrumental for engineers in search of performance optimizations.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **A New Challenger Approaches in AI**: [Llama 3](https://llama.meta.com/llama3/) has claimed the joint 5th place on the [LMSYS arena leaderboard](https://chat.lmsys.org/?leaderboard), rubbing shoulders with top models like Claude 3 Opus and GPT-4 variants, and can run on high-end laptops.

- **SimonW's Toolkit for Llama 3**: Simon Willison has launched [LLM](https://llm.datasette.io/), a toolset complete with a command-line interface and a Python library, designed to streamline using Llama 3 and other models. Detailed usage instructions can be found in his blog post [here](https://simonwillison.net/2024/Apr/22/llama-3/).

- **AI Checks Architectural Homework**: AI has carved a niche in architecture, functioning as a 'preflight' tool to spot potential issues and code violations in architectural designs, though it hasn't progressed to creating blueprints yet.

- **Blueprint Interpretation Still at Ground Floor**: Conversations are circling around employing AI to interpret architectural blueprints, specifically for tracing ductwork in PDF formats, but no concrete solutions were tabled. 

- **Hackernews Digest Desideratum**: An inquiry was made about a bash script to generate summaries of Hackernews but details of the latest version were not mentioned in the discussion.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Spam Crusaders Needed**: The **general-chat** was bombarded with *spam messages* linking to an unauthorized Discord invite with NSFW content.
- **Jamba Compatibility Queries**: A member's curiosity was piqued regarding whether **Jamba** is compatible with **LM Studio** and sought details on its operational requisites, akin to **Claude**'s memory footprint.
- **Jamba's Memory Appetite**: Discussions unfolded around the challenges of running **Jamba**, particularly its hefty RAM requirements, noting that even Google Colab fell short in providing necessary resources, and efforts on Google Cloud were also fruitless.
- **Spam Link Blunder**: An untoward **spam link** promising NSFW content was distributed in the channel but should be disregarded and reported by vigilant members.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1231134777797513296)** (1118 messages🔥🔥🔥): 

- **Unsloth Supports Phi-3 Mini**: Unsloth announces their support for Microsoft's Phi-3 Mini 4K Instruct model and has uploaded a 4bit version on Hugging Face, aiming to integrate it into the Unsloth library despite some required alterations due to architectural differences from Llama 3. Their blog post about Llama 3 has been updated with this information and they are waiting to support the 14B variant when released.
- **Successful Fine-Tuning on 24GB VRAM**: A user reported success in fine-tuning Llama 3 using Unsloth on a 1x3090 24GB GPU with pure BF16 quality, effectively handling the memory demands and using only 16GB of VRAM.
- **Ergonomic Workstation Discussions**: Members shared experiences and recommendations on ergonomic workstation setups, highlighting keyboards, monitors, chairs, and the benefits of standing desks for a comfortable working environment.
- **Technical Blog Post Tips**: Following feedback on previous blog posts, Unsloth's upcoming posts will include more benchmarks and descriptive texts within images to provide clearer context and information.
- **Phi-3 Analysis and Anticipation**: There is ongoing anticipation and discussion among users regarding the newly released Phi-3 models, with curiosity about further claims and applications. Some users contemplate finetuning these models and are eagerly awaiting compatibility with existing libraries.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>: A free course designed for people with some coding experience, who want to learn how to apply deep learning and machine learning to practical problems.</li><li><a href="https://www.theverge.com/2024/4/23/24137534/microsoft-phi-3-launch-small-ai-language-model">Microsoft launches Phi-3, its smallest AI model yet</a>: Phi-3 is the first of three small Phi models this year.</li><li><a href="https://huggingface.co/chargoddard/llama3-42b-v0">chargoddard/llama3-42b-v0 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit">unsloth/Phi-3-mini-4k-instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/cosmos-carl-sagan-gif-3394876">Watching The Cosmos GIF - Cosmos Carl Sagan - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/BarraHome/llama-3-orpo-v1">BarraHome/llama-3-orpo-v1 · Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog/">Blog</a>: no description found</li><li><a href="https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers">Nvidia bans using translation layers for CUDA software &mdash; previously the prohibition was only listed in the online EULA, now included in installed files [Updated]</a>: Translators in the crosshairs.</li><li><a href="https://unsloth.ai/blog/llama3">Finetune Llama 3 with Unsloth</a>: Fine-tune Meta&#x27;s new model Llama 3 easily with 6x longer context lengths via Unsloth!</li><li><a href="https://x.com/danielhanchen/status/1782790737798861281">Tweet from Daniel Han (@danielhanchen)</a>: Phi-3 Mini 3.8b Instruct is out!! 68.8 MMLU vs Llama-3 8b Instruct&#39;s 66.0 MMLU (Phi team&#39;s own evals)  The long context 128K model is also out at https://huggingface.co/microsoft/Phi-3-mini-12...</li><li><a href="https://kinesis-ergo.com/shop/advantage2/">Advantage2 ergonomic keyboard by Kinesis</a>: Contoured design, mechanical switches, fully programmable</li><li><a href="https://www.youtube.com/watch?v=E5kzAbD8D0w">Direct Preference Optimization (DPO)</a>: Get the Dataset: https://huggingface.co/datasets/Trelis/hh-rlhf-dpoGet the DPO Script + Dataset: https://buy.stripe.com/cN2cNyg8t0zp2gobJoGet the full Advanc...</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://www.macrumors.com/2024/04/22/apple-acquires-french-ai-company/">Apple Acquires French AI Company Specializing in On-Device Processing</a>: Apple has acquired the Paris-based artificial intelligence startup Datakalab amid its push to deliver on-device AI tools.   Datakalab specializes in...</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook">Kaggle Llama-3 8b Unsloth notebook</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://www.reddit.com/r/hardware/comments/1c2dyat/geohot_hacked_4090_driver_to_enable_p2p/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</li><li><a href="https://tenor.com/view/kevin-the-office-smirk-gif-3304715514430776968">Kevin The Office GIF - Kevin The Office Smirk - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/localghost/status/1781847388879220742">Tweet from Aaron Ng (@localghost)</a>: llama 3 70b beamed to my phone from my M1 Max  ~7.6 tok/s with mlx. your own little gpt-4 at home</li><li><a href="https://unsloth.ai/blog">Blog</a>: no description found</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit/blob/main/generation_config.json">generation_config.json · unsloth/llama-3-8b-Instruct-bnb-4bit at main</a>: no description found</li><li><a href="https://www.philschmid.de/fsdp-qlora-llama3">Efficiently fine-tune Llama 3 with PyTorch FSDP and Q-Lora</a>: Learn how to fine-tune Llama 3 70b with PyTorch FSDP and Q-Lora using Hugging Face TRL, Transformers, PEFT and Datasets.</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth update: Mistral support + more</a>: We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...</li><li><a href="https://github.com/zenoverflow/datamaker-chatproxy">GitHub - zenoverflow/datamaker-chatproxy: Proxy server that automatically stores messages exchanged between any OAI-compatible frontend and backend as a ShareGPT dataset to be used for training/finetuning.</a>: Proxy server that automatically stores messages exchanged between any OAI-compatible frontend and backend as a ShareGPT dataset to be used for training/finetuning. - zenoverflow/datamaker-chatproxy</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>: no description found</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets</a>: Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit</li><li><a href="https://github.com/ml-explore/mlx-swift/?tab=readme-ov-file">GitHub - ml-explore/mlx-swift: Swift API for MLX</a>: Swift API for MLX. Contribute to ml-explore/mlx-swift development by creating an account on GitHub.</li><li><a href="https://github.com/NVIDIA/open-gpu-kernel-modules/commit/1f4613dacec2638569a74b5e3dbcab01832f72a7">add P2P support · NVIDIA/open-gpu-kernel-modules@1f4613d</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/844">iPad App · ggerganov/llama.cpp · Discussion #844</a>: I&#39;ve been playing with using llama to help me tell stories to my daughter at night. I wrote a simple native iPad app that uses llama.cpp, and provides some nice model / thread management capabilit...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/4815">main : add Self-Extend support by ggerganov · Pull Request #4815 · ggerganov/llama.cpp</a>: continuation of #4810 Adding support for context extension to main based on this work: https://arxiv.org/pdf/2401.01325.pdf Did some basic fact extraction tests with ~8k context and base LLaMA 7B v...</li><li><a href="https://archive.ph/zbhlo">Apple (AAPL) Growth Opportunities: Southeast Asia and Africa, Lower-E&#x2026;</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1231169196390617108)** (167 messages🔥🔥): 

- **New Llama AI Model Released**: A [Hugging Face model: Llama 3 70B INSTRUCT 4bit](https://huggingface.co/unsloth/llama-3-70b-Instruct-bnb-4bit) has been uploaded, promising *finetuning Mistral, Gemma, and Llama up to 2-5 times faster with 70% less memory*. Accompanying this is a [Google Colab GPU notebook](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) for Llama-3 8b.
- **Upcoming Tutorial Materials**: Community members discussed creating and sharing a guide or notebook to help with **finetuning Instruct models** with chat templates. It was suggested that materials including a *video tutorial* might be in the works.
- **Struggling with Llama C++ Batch Processing**: A user reports that using `--cont-batching` or `cache_prompt` in llama.cpp for simultaneous prompt processing shows no performance gains, as sending prompts sequentially or concurrently takes the same amount of time.
- **Gemma Keyword Extraction Challenges**: A discussion took place regarding the extraction of keyphrases from customer reviews with an LLM such as **Gemma**, and how it often results in too creative or inaccurate results, pushing users to consider other tools like [KeyBERT](https://github.com/MaartenGr/KeyBERT).
- **Unsloth Project Updates and Community Contributions**: There is an anticipation for **Unsloth's continued work on tutorials, blog posts, and a studio for Colab**, with contributions from the community anticipated, including shared notebooks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html">Answer.AI - Efficient finetuning of Llama 3 with FSDP QDoRA</a>: We’re releasing FSDP QDoRA, a scalable and memory-efficient method to close the gap between parameter efficient finetuning and full finetuning.</li><li><a href="https://www.youtube.com/watch?v=vOA9JSDPJs0">Q*</a>: Like 👍. Comment 💬. Subscribe 🟥.🏘 Discord: https://discord.gg/pPAFwndTJdhttps://github.com/hu-po/docsFrom r to Q∗: Your Language Model is Secretly a Q-Fun...</li><li><a href="https://github.com/MaartenGr/KeyBERT">GitHub - MaartenGr/KeyBERT: Minimal keyword extraction with BERT</a>: Minimal keyword extraction with BERT. Contribute to MaartenGr/KeyBERT development by creating an account on GitHub.</li><li><a href="https://pytorch-dev-podcast.simplecast.com/">no title found</a>: no description found</li><li><a href="https://huggingface.co/unsloth/llama-3-70b-Instruct-bnb-4bit">unsloth/llama-3-70b-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/@CUDAMODE">CUDA MODE</a>: A CUDA reading group and community https://discord.gg/cudamode Supplementary content here https://github.com/cuda-mode Created by Mark Saroufim and Andreas Köpf    </li><li><a href="https://discord.gg/rWpeuatu">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1231139392899383388)** (716 messages🔥🔥🔥): 

- **LLaMA Model Training Issues**: Members discussed problems related to fine-tuning LLaMA models, where the output was repeating the same sentence or stopping prematurely. Solutions such as adjusting training configurations and verifying tokenizer settings were suggested. Additionally, users faced challenges when trying to upcast to FP16 and were guided to use specific commands for successful training and quantization.

- **Exploring Quantization and Unsloth Models**: Users explored how quantization affects model quality and the resource requirements for running models on limited hardware. For practical applications, a guideline suggested considering around 4-bit quantization to maintain balance between performance and quality.

- **Setting Up and Importing to Unsloth**: Challenges were mentioned regarding setting up the Unsloth environment and importing models, with particular issues around Python environment setups. Some users mentioned success with reinstalling packages or ensuring they had the latest version of Unsloth.

- **Using Inference with Finetuned Models**: Users interacting with finetuned models noticed discrepancies in the models' responses; for example, output being identical to the input prompt. Unsloth was reported to have recently fixed such tokenizer issues (e.g., defining stopping/eos tokens), which were impacting inference performance.

- **Exporting Models and Fine-tuning Strategies**: Tips for exporting unsloth models to gguf/vLLM formats and merging LoRA adapters back to FP16 were shared. Users sought advice on best approaches for embedding knowledge into LLMs for instructional use, and in general guidance for the fine-tuning process was sought after by several community members.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh">no title found</a>: no description found</li><li><a href="https://huggingface.co/Finnish-NLP/llama-3b-finnish-v2/blob/main/config.json">config.json · Finnish-NLP/llama-3b-finnish-v2 at main</a>: no description found</li><li><a href="https://huggingface.co/imone">imone (One)</a>: no description found</li><li><a href="https://github.com/unslo">unslo</a>: GitHub is where unslo builds software.</li><li><a href="https://huggingface.co/spaces/mlabonne/OrpoLlama-3-8B">OrpoLlama-3-8B - a Hugging Face Space by mlabonne</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://www.hackster.io/news/tomeu-vizoso-s-open-source-npu-driver-project-does-away-with-the-rockchip-rk3588-s-binary-blob-0153cf723d44">Tomeu Vizoso&#39;s Open Source NPU Driver Project Does Away with the Rockchip RK3588&#39;s Binary Blob</a>: Anyone with a Rockchip RK3588 and a machine learning workload now has an alternative to the binary blob driver, thanks to Vizoso&#39;s efforts.</li><li><a href="https://github.com/unslothai/unsloth/issues/356">save_pretrained_gguf  method RuntimeError: Unsloth: Quantization failed .... · Issue #356 · unslothai/unsloth</a>: /usr/local/lib/python3.10/dist-packages/unsloth/save.py in save_to_gguf(model_type, model_directory, quantization_method, first_conversion, _run_installer) 955 ) 956 else: --&gt; 957 raise RuntimeErro...</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: no description found</li><li><a href="https://youtu.be/SL2nZpv7dtY?si=Zne1z1tB8d_A7Ia9&t=1613">Full fine tuning vs (Q)LoRA</a>: ➡️ Get Life-time Access to the complete scripts (and future improvements): https://trelis.com/advanced-fine-tuning-scripts/➡️ Runpod one-click fine-tuning te...</li><li><a href="https://www.youtube.com/@MervinPraison/videos">Mervin Praison</a>: Mervin Praison</li><li><a href="https://tenor.com/view/atom-real-steel-movie-robot-fight-gif-13618149">Atom Real Steel GIF - Atom Real Steel Movie - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/love-actually-christmas-christmas-movie-workingtitlefilms-hugh-grant-gif-15362644">Love Actually Christmas GIF - Love Actually Christmas Christmas Movie - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://colab.research.google.com/drive/1DGhWyCyf1BI-_yYaLYgOOkZuGAWiuqNj?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard">Big Code Models Leaderboard - a Hugging Face Space by bigcode</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/Dn0tmI0FFS">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/WBcE.gif">Carson Wcth GIF - Carson WCTH Happens To The Best Of Us - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://repo.anaconda.com/miniconda/">Index of /</a>: no description found</li><li><a href="https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9">GGUF quantizations overview</a>: GGUF quantizations overview. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/pidugusundeep/Brat-and-snorkel/blob/master/ann-coll.py">Brat-and-snorkel/ann-coll.py at master · pidugusundeep/Brat-and-snorkel</a>: Supporting files. Contribute to pidugusundeep/Brat-and-snorkel development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: I got unsloth running in native windows, (no wsl). You need visual studio 2022 c++ compiler, triton, and deepspeed. I have a full tutorial on installing it, I would write it all here but I’m on mob...</li><li><a href="https://github.com/meta-llama/llama-recipes">GitHub - meta-llama/llama-recipes: Scripts for fine-tuning Meta Llama3 with composable FSDP &amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp; custom datasets for applications such as summarization and Q&amp;A. Supporting a number of candid inference solutions such as HF TGI, VLLM for local or cloud deployment. Demo apps to showcase Meta Llama3 for WhatsApp &amp; Messenger.</a>: Scripts for fine-tuning Meta Llama3 with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization and Q...</li><li><a href="https://huggingface.co/imone/Llama-3-8B-fixed-special-embedding">imone/Llama-3-8B-fixed-special-embedding · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_strategy">Trainer</a>: no description found</li><li><a href="https://github.com/sgl-project/sglang">GitHub - sgl-project/sglang: SGLang is a structured generation language designed for large language models (LLMs). It makes your interaction with models faster and more controllable.</a>: SGLang is a structured generation language designed for large language models (LLMs). It makes your interaction with models faster and more controllable. - sgl-project/sglang</li><li><a href="https://github.com/hiyouga/LLaMA-Factory#hardware-requirement">GitHub - hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs</a>: Unify Efficient Fine-Tuning of 100+ LLMs. Contribute to hiyouga/LLaMA-Factory development by creating an account on GitHub.</li><li><a href="https://status.huggingface.co/">
Hugging Face status
</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1231174242029010995)** (76 messages🔥🔥): 

- **Swedish Language Model Progress**: A showcase of the **[llama-3-instruct-bellman-8b-swe-preview](https://huggingface.co/neph1/llama-3-instruct-bellman-8b-swe-preview)** model was provided, which has been trained for coherence and reasoning. Enthusiasm was expressed for the model that has been trained using Unsloth.
  
- **Introducing Ghost 7B Alpha**: Release of **Ghost 7B Alpha**, optimizing reasoning and multitasking abilities, was announced with resources such as a [model card](https://huggingface.co/ghost-x/ghost-7b-alpha), [website documentation](https://ghost-x.vercel.app/docs/models/ghost-7b-alpha), and a [demo](https://ghost-x.vercel.app/docs/notebooks/playground-with-ghost-7b-alpha).

- **Improvement Through Retraining**: A member discussed retraining the **Llama3** model using Unsloth's latest 4bit version, which led to successful results and a decision to continue experimenting with different hyperparameters.

- **Solobsd Unveils Spanish Language Model**: A new Spanish language model (**solobsd-llama3**) was announced, based on data from the Alpaca dataset, with appreciation and inquiries about the specific variant of Spanish demonstrated.

- **Model Fine-Tuning Discussions**: There was a technical exchange on how to effectively stop models during generation and how to work with dataset templates in context with Unsloth and Llama3. Advice and steps for successful training and conversion were shared among contributors.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mahiatlinux/MasherAI-7B-v6.1">mahiatlinux/MasherAI-7B-v6.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/SoloBSD/solobsd-llama3">SoloBSD/solobsd-llama3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/hikikomoriHaven/llama3-8b-hikikomori-v0.1/">hikikomoriHaven/llama3-8b-hikikomori-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Remek/Llama-3-8B-Omnibus-1-PL-v01-INSTRUCT">Remek/Llama-3-8B-Omnibus-1-PL-v01-INSTRUCT · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/BarraHome/llama-3-orpo-v1-merged_16bit">BarraHome/llama-3-orpo-v1-merged_16bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/hi">Hi (Ho)</a>: no description found</li><li><a href="https://huggingface.co/neph1/llama-3-instruct-bellman-8b-swe-preview">neph1/llama-3-instruct-bellman-8b-swe-preview · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/ghost-x/ghost-7b-alpha">ghost-x/ghost-7b-alpha · Hugging Face</a>: no description found</li><li><a href="https://ghost-x.vercel.app/docs/models/ghost-7b-alpha">Ghost 7B Alpha</a>: The large generation of language models focuses on optimizing excellent reasoning, multi-task knowledge, and tools support.</li><li><a href="https://ghost-x.vercel.app/docs/notebooks/playground-with-ghost-7b-alpha">Playground with Ghost 7B Alpha</a>: To make it easy for everyone to quickly experience the Ghost 7B Alpha model through platforms like Google Colab and Kaggle. We've made these notebooks available so you can get started right away.</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6745">Support Llama 3 conversion by pcuenca · Pull Request #6745 · ggerganov/llama.cpp</a>: The tokenizer is BPE.
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1231279056901636198)** (73 messages🔥🔥): 

- **Color Confusion Conundrum**: A member expressed difficulties reading the welcome message due to a poor color scheme selection—green background with gray text. The issue was resolved after the color was changed upon this feedback.

- **Workflow Woes in Google Colab**: Members discussed the challenges faced when using Google Colab for CUDA and C++ development lacking debugging tools and syntax highlighting. The conversation spanned issues such as the messiness of print statements for debugging and slower productivity, with some suggesting the use of VSCode over SSH.

- **SSH and Colab Conundrum**: Experiences with remote SSH access to Google Colab were shared, with a focus on workflow inefficiencies and the negatives of remote SSH not being a pleasant experience. A [tutorial from Puget Systems](https://www.pugetsystems.com/labs/hpc/How-To-Run-Remote-Jupyter-Notebooks-with-SSH-on-Windows-10-1477/) was linked for setting up Jupyter Notebooks with SSH on Windows 10.

- **Philanthropic Pursuits for Unsloth Pro**: The discussion explored Unsloth Pro's potential direction, suggesting the application for philanthropic grants and open-sourcing the code. However, it was mentioned that Unsloth has now secured funding and is building its platform.

- **Debating The Need for a Jobs Channel**: Members debated the necessity and potential risks of adding a #jobs channel to the server. Concerns about scamming, channel clutter, and maintaining focus on Unsloth were raised, without reaching a consensus on the issue.

- **Vision for Vision - Model Compatibility Suggestion**: Suggestions were made for future support of various models, including those for vision tasks possibly alongside the upcoming Llama-3 vision release. Additionally, curiosity arose regarding the instruction version of newly mentioned models like Phi-3.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.pugetsystems.com/labs/hpc/How-To-Run-Remote-Jupyter-Notebooks-with-SSH-on-Windows-10-1477/">How To Run Remote Jupyter Notebooks with SSH on Windows 10</a>: Being able to run Jupyter Notebooks on remote systems adds tremendously to the versatility of your workflow. In this post I will show a simple way to do this by taking advantage of some nifty features...</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=DdTsX6DQk24&t=2s">Lecture 14: Practitioners Guide to Triton</a>: https://github.com/cuda-mode/lectures/tree/main/lecture%2014
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1232348211570671667)** (1 messages): 

- **Perplexity Enterprise Pro Launches**: Perplexity introduces **Enterprise Pro**, a secure AI answer engine designed for businesses, featuring **increased data privacy, SOC2 compliance, user management, and single sign-on**. With heavyweights like **Stripe, Zoom, and Databricks** already leveraging its benefits, Databricks reports saving approximately *5000 hours a month*.

- **Enterprise Pro's Impact and Pricing**: Catering to diverse industries including software, finance, and sports, **Enterprise Pro** offers knowledge workers the ability to search for fast, reliable information securely, priced at **\$40/month** or **\$400/year per seat**. Interested companies can sign up at [Perplexity Enterprise](https://pplx.ai/enterprise).
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1231143301369958504)** (1005 messages🔥🔥🔥): 

- **Perplexity Enterprise Pro Unleashed**: A new, premium feature, **Perplexity Enterprise Pro**, has been announced via the official channel and on [Bloomberg](https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-in-funding-round?cmpid=socialflow-twitter-business), offering added features like improved security and data protection measures for $40/month.
- **Corporate Growth and Product Diversification**: Perplexity.ai's valuation has hit **$1 billion** following a successful funding round, signaling expansion and a broader service offering, including the teased potential involvement of AI luminary Yann LeCun.
- **Privacy Concerns and Clarifications**: User discussions raised concerns about data privacy and whether data from paid users were being used for training AI models; moderators linked to official statements implying data usage consents and options.
- **iOS App Challenges**: Users reported persistent issues with the Perplexity app on iPad, such as inability to search or sign-in, with support advising affected users to reach out via direct message for assistance.
- **Potential Changes and Features in Projected Release**: With speculative hints from moderators about imminent updates, users speculate about feature drops, removal of Opus limits, or other improvements, leading to eager anticipation for the April 23rd announcement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.rabbit.tech/.">rabbit r1 - pickup party nyc live at 8PM ET</a>: streaming from the r1 pickup party event in NYC</li><li><a href="https://decoder.sh/videos/use-your-self_hosted-llm-anywhere-with-ollama-web-ui">Use Your Self-Hosted LLM Anywhere with Ollama Web UI</a>: no description found</li><li><a href="https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-in-funding-round?cmpid=socialflow-twitter-business">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://docs.openwebui.com">🏡 Home | Open WebUI</a>: Open WebUI is an extensible, feature-rich, and user-friendly self-hosted WebUI designed to operate entirely offline. It supports various LLM runners, including Ollama and OpenAI-compatible APIs.</li><li><a href="https://www.bloomberg.com/news/articles/2024-04-23/ai-search-startup-perplexity-valued-at-1-billion-">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://www.ycombinator.com/apply">Apply to Y Combinator | Y Combinator</a>: To apply for the Y Combinator program, submit an application form. We accept companies twice a year in two batches. The program includes dinners every Tuesday, office hours with YC partners and access...</li><li><a href="https://en.m.wikipedia.org/wiki/Yann_LeCun">Yann LeCun - Wikipedia</a>: no description found</li><li><a href="https://tenor.com/view/superstore-amy-sosa-im-just-guessing-just-guessing-wild-guess-gif-24963833">Superstore Amy Sosa GIF - Superstore Amy Sosa Im Just Guessing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/money-mr-krabs-gif-18326632">Money Mr GIF - Money Mr Krabs - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/@AndrejKarpathy/videos">Andrej Karpathy</a>: FAQ Q: How can I pay you? Do you have a Patreon or etc? A: As YouTube partner I do share in a small amount of the ad revenue on the videos, but I don&#39;t maintain any other extra payment channels. I...</li><li><a href="https://tenor.com/view/think-about-it-use-your-brain-use-the-brain-think-brain-gif-7914082">Think About It Use Your Brain GIF - Think About It Use Your Brain Use The Brain - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.morphic.sh/">Morphic</a>: A fully open-source AI-powered answer engine with a generative UI.</li><li><a href="https://tenor.com/view/yt-youtube-logo-gif-27453294">Yt Youtube GIF - Yt Youtube Logo - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/heidi-klum-number-two-2fingers-deuces-your-second-yes-gif-25857953">Heidi Klum Number Two GIF - Heidi Klum Number Two 2Fingers - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fxtwitter.com/AravSrinivas/status/1782775219733844256?t=Oo_2sf1Yj-XImPRrzO19nA&s=19">Tweet from Aravind Srinivas (@AravSrinivas)</a>: We have many Perplexity users who tell us that their companies don&#39;t let them use it at work due to data and security concerns, but they really want to. To address this, we&#39;re excited to be la...</li><li><a href="https://x.com/aravsrinivas/status/1781902284844421624?s=46">Tweet from Aravind Srinivas (@AravSrinivas)</a>: 4/23</li><li><a href="https://www.morphic.sh">Morphic</a>: A fully open-source AI-powered answer engine with a generative UI.</li><li><a href="https://console.groq.com/playground?model=llama3-70b-8192">GroqCloud</a>: Experience the fastest inference in the world</li><li><a href="https://www.chooseoxygen.com/en/blog/chatgpt-vs-notion-ai-comprehensive-comparison-for-ai-writing">ChatGPT vs Notion AI: An In-Depth Comparison For Your AI Writing Needs</a>: A comprehensive comparison between two AI tools, ChatGPT and Notion AI, including features, pricing and use cases. </li><li><a href="https://x.com/AravSrinivas/status/1781721468180767002">Tweet from Aravind Srinivas (@AravSrinivas)</a>: 8b is so good. Can create a lot more experiences with it. We have some ideas. Stay tuned!  ↘️ Quoting MachDiamonds (@andromeda74356)   @AravSrinivas Will you be switching the free perplexity version t...</li><li><a href="https://github.com/mckaywrigley/clarity-ai">GitHub - mckaywrigley/clarity-ai: A simple Perplexity AI clone.</a>: A simple Perplexity AI clone. Contribute to mckaywrigley/clarity-ai development by creating an account on GitHub.</li><li><a href="https://www.google.com/amp/s/www.xataka.com/aplicaciones/ultimo-openai-llega-a-copilot-asistente-programacion-evoluciona-nuevo-modelo-ia/amp">Lo último de OpenAI llega a Copilot. El asistente de programación evoluciona con un nuevo modelo de IA</a>: En el último año, la inteligencia artificial no solo ha estado detrás de generadores de imágenes como DALL·E y bots conversacionales como ChatGPT, también ha...</li><li><a href="https://github.com/developersdigest/llm-answer-engine">GitHub - developersdigest/llm-answer-engine: Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper</a>: Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper - developersdigest/llm-answer-engine</li><li><a href="https://youtu.be/znOlwELyt8g?si=UDq4joNqi1n7z8i3">Eric Gundersen Talks About How Mapbox Uses AWS to Map Millions of Miles a Day</a>: Learn more about how AWS can power your big data solution here - http://amzn.to/2grdTah.Mapbox is collecting 100 million miles of telemetry data every day us...</li><li><a href="https://tenor.com/view/robot-depressed-marvin-hitch-hikers-guide-to-the-galaxy-gif-4931652">Robot Depressed GIF - Robot Depressed Marvin - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/hFUaXEXfNnA?si=KWY0eyvRZNac2Gzt">AWS re:Invent 2023 - Customer Keynote Perplexity | AWS Events</a>: Hear from Aravind Srinivas, cofounder and CEO of Perplexity, about how the conversational artificial intelligence (AI) company is reimagining search by provi...</li><li><a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ)">Rick Astley - Never Gonna Give You Up (Official Music Video)</a>: The official video for “Never Gonna Give You Up” by Rick Astley. The new album &#39;Are We There Yet?&#39; is out now: Download here: https://RickAstley.lnk.to/AreWe...</li><li><a href="https://youtu.be/YKMDw7ERxZ4?si=t0ybyzaEgUZNsihl">AWS re:Invent 2023 - Customer Keynote Anthropic</a>: In this AWS re:Invent 2023 fireside chat, Dario Amodei, CEO and cofounder of Anthropic, and Adam Selipsky, CEO of Amazon Web Services (AWS) discuss how Anthr...</li><li><a href="https://share.wendabao.net">no title found</a>: no description found</li><li><a href="https://github.com/xx025/carrot">GitHub - xx025/carrot: Free ChatGPT Site List 这儿为你准备了众多免费好用的ChatGPT镜像站点</a>: Free ChatGPT Site List 这儿为你准备了众多免费好用的ChatGPT镜像站点. Contribute to xx025/carrot development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1231237470188732437)** (29 messages🔥): 

- **Perplexity AI Searches Shared**: Members of the **Sharing** channel shared various links to Perplexity AI searches ranging from topics like **positive parenting** to **instructions for unclear prompts**. Each shared Perplexity page tackles specific questions or informational requests.
- **Guidance on Sharing**: Users are reminded to ensure their shared threads are shareable, with a link provided to instructions on making a thread **shareable**.
- **Perplexity AI Making Headlines**: The AI search engine startup **Perplexity AI** has been featured in news outlets, with discussions on the channel about its recent valuation increase and fundraising efforts. **TechCrunch** article and **CNBC** interview with CEO Aravind Srinivas were shared, highlighting the company's growth and enterprise launch.
- **CEO’s CNBC Interview Transcribed**: An unofficial transcript of an exclusive **CNBC interview** with **Perplexity Founder & CEO Aravind Srinivas** was shared, along with a link to the accompanying video interview.
- **Company Valuation Discussions**: Members discussed the increasing valuation of **Perplexity AI**, which is reportedly raising at least **$250 million more at a valuation of between $2.5 billion and $3 billion**, marking rapid growth since its last funding round.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.cnbc.com/2024/04/23/cnbc-exclusive-cnbc-transcript-perplexity-founder-ceo-aravind-srinivas-speaks-with-cnbcs-andrew-ross-sorkin-on-squawk-box-today.html">CNBC Exclusive: CNBC Transcript: Perplexity Founder &amp; CEO Aravind Srinivas Speaks with CNBC’s Andrew Ross Sorkin on “Squawk Box” Today</a>: no description found</li><li><a href="https://techcrunch.com/2024/04/23/perplexity-is-raising-250m-at-2-point-5-3b-valuation-ai-search-sources-say/">EXCLUSIVE: Perplexity is raising $250M+ at a $2.5-$3B valuation for its AI search platform, sources say</a>: Perplexity, the AI search engine startup, is a hot property at the moment. TechCrunch has learned that the company is currently raising at least $250</li><li><a href="https://www.youtube.com/watch?v=LGuA5JOyUhE">Perplexity CTO Denis Yarats on AI-powered search</a>: Perplexity is an AI-powered search engine that answers user questions. Founded in 2022 and valued at over $1B, Perplexity recently crossed 10M monthly active...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1231252972319801344)** (3 messages): 

- **Seeking GPT with Internet Access**: A new member inquired about an API similar to GPT chat but with **Internet access** and **up-to-date information** from the web. They were provided with a link to [Perplexity's documentation](https://docs.perplexity.ai/docs/model-cards) and informed about the **sonar online models** which offer Internet access, along with an invitation to sign up for access to citations.
- **A Pointer for Improved Model Performance**: A member suggested enhancing performance by including **one-shot examples** in the prompt, possibly aiming for more precise results or better understood instructions by the model.
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1231136613526929468)** (1044 messages🔥🔥🔥): 

- **New Kid on the Block**: A user stated that they are new to stable diffusion and are in the process of downloading Forge Webui, inquiring if it's a satisfactory choice or if there are better alternatives.
- **Exploring AI's Creative Frontier**: Various users discussed their interests in generating images and assets using AI tools, such as Stable Diffusion. One mentioned wanting to make game assets and another expressed desire to generate space ships and sci-fi themes.
- **Technical Troubles**: Several users sought technical help with issues ranging from CUDA errors and generation speed to missing nodes in ComfyUI. There are questions about using specific models in different interfaces like Forge and webui and inquiries about transferring installations between drives.
- **AI Generated Futures**: Casual conversations took place where users pondered using AI to create perfect representations of significant others or dream homes. There is clear excitement about the potential of AI to generate bespoke content.
- **Anticipation for Stability AI Release**: Users expressed curiosity and skepticism about the release and features of Stable Diffusion version 3, with some relaying information from the former CEO Emad and speculating on the timeline and true openness of the eventual release.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/chrlaf/status/1772228848387522728">Tweet from Christian Laforte (@chrlaf)</a>: @rajdhakad_ @USEnglish215753 @StabilityAI @EMostaque Our plan is to soon release the API first to collect more human preference data and validate that our safety improvements don&#39;t cause the quali...</li><li><a href="https://wallet.bitcoin.com/">Crypto Wallet | Supports Bitcoin (BTC), Bitcoin Cash (BCH), Ethereum (ETH), and ERC-20 tokens</a>: Download Bitcoin.com’s multi-coin crypto wallet. A simple and secure way to buy, sell, trade, and use cryptocurrencies. Supports Bitcoin (BTC), Bitcoin Cash (BCH), Ethereum (ETH), and ERC-20 tokens in...</li><li><a href="https://glif.app/@fab1an/glifs/clv488uy10000djtrx70u03no">glif - StableDiffusion 3 by fab1an</a>: no description found</li><li><a href="https://www.youtube.com/playlist?list=PLIF38owJLhR1EGDY4kOnsEnMyolZgza1x">ComfyUI</a>: A better method to use stable diffusion models on your local PC to create AI art.</li><li><a href="https://forums.developer.nvidia.com/t/cuda-enabled-geforce-1650/81010/5">CUDA-Enabled GeForce 1650?</a>: If you cannot find the answer in the GROMACS documentation, I would suggest asking about GROMACS configuration issues on the official GROMACS mailing list:  [url]http://www.gromacs.org/Support/Mailing...</li><li><a href="https://github.com/comfyanonymous/ComfyUI/releases/download/latest/ComfyUI_windows_portable_nvidia_cu118_or_cpu.7z">no title found</a>: no description found</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: no description found</li><li><a href="https://civitai.com/images/10123212">Image posted by pagartomas880</a>: no description found</li><li><a href="https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local">CUDA Toolkit 12.1 Downloads</a>: Get the latest feature updates to NVIDIA&#39;s proprietary compute stack.</li><li><a href="https://www.youtube.com/watch?v=ktxbXlF6UQE">Exposing the Website that Stalks You in Discord!</a>: There is a website called spy.pet that claims to have 4 billion messages saved across Discord. With this, you can &quot;see what your friends are doing on Discord...</li><li><a href="https://github.com/Stability-AI/stablediffusion">GitHub - Stability-AI/stablediffusion: High-Resolution Image Synthesis with Latent Diffusion Models</a>: High-Resolution Image Synthesis with Latent Diffusion Models - Stability-AI/stablediffusion</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.</a>: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI</li><li><a href="https://www.youtube.com/watch?v=9qd04u2Yj44">Weird Science Official Trailer #1 - Robert Downey Jr. Movie (1985) HD</a>: Subscribe to TRAILERS: http://bit.ly/sxaw6hSubscribe to COMING SOON: http://bit.ly/H2vZUnSubscribe to CLASSIC TRAILERS: http://bit.ly/1u43jDeLike us on FACEB...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://civitai.com/models/368139/character-sheet">character sheet - character sheet | Stable Diffusion LoRA | Civitai</a>: no description found</li><li><a href="https://new.reddit.com/user/emad_9608/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ltdrdata/ComfyUI-Manager">GitHub - ltdrdata/ComfyUI-Manager</a>: Contribute to ltdrdata/ComfyUI-Manager development by creating an account on GitHub.</li><li><a href="https://hidiffusion.github.io/">SOCIAL MEDIA TITLE TAG</a>: SOCIAL MEDIA DESCRIPTION TAG TAG</li><li><a href="https://github.com/megvii-research/HiDiffusion">GitHub - megvii-research/HiDiffusion</a>: Contribute to megvii-research/HiDiffusion development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1231285314484572313)** (5 messages): 

- **Tensor Parallel with VLLM**: Reference to progress on implementing **tensor parallel** with VLLM was made, with the anticipation of *jamba* support for enhancing model performance.
- **Anticipating Jamba API Release**: There's an expression of need for a **jamba API** that would allow utilization of the entire context for a particular modeling task.
- **Seeking Economical Context Management**: A user shared the struggle with managing context economically when using **Claude 3** and **Big-AGI**, where costs escalate quickly. They found potential solutions like [memGPT](https://memgpt.ai/) and [SillyTavern SmartContext](https://docs.sillytavern.app/extras/extensions/smart-context/), and are seeking additional solutions for efficient context management.
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1231206936532357181)** (22 messages🔥): 

- **Beastie Boys Get REMASTERED**: A [YouTube video titled "Beastie Boys - Root Down"](https://www.youtube.com/watch?v=Xf1YF_MH1xc) was shared; it's part of a remastered HD series that includes a backstory about the "Ill Communication" album.
- **deadmau5 & Kaskade Remembered in High Quality**: Another YouTube share featured [deadmau5 & Kaskade's track "I Remember (HQ)"](https://youtu.be/zK1mLIeXwsQ?t=119), showcasing the song's quality and providing links to more music and tour information.
- **Latent Humor in CIFAR100**: The CIFAR100 dataset has been humorously encoded into 100 classes and shared as [latent-CIFAR100](https://huggingface.co/datasets/Verah/latent-CIFAR100), with safetensors recommended for usage in the 488 latent size version.
- **Seeking Bigger Pixels for Image Classification**: A member inquired about larger image classification datasets (64x64 or 128x128) after sharing that a simple feedforward neural network yielded around 19% accuracy on a latently encoded dataset with the dimensions of 4x4x4.
- **Papers on Symbol Systems and Language Models**: A [contribution of scholarly papers](https://arxiv.org/abs/2402.10588) focused on language models and their symbolic representation, pointing to the semantic vector space as a phase in which symbolic meaning can emerge, analogous to language understanding in LLMs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/Verah/latent-CIFAR100">Verah/latent-CIFAR100 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...</li><li><a href="https://tenor.com/view/hellinheavns-gif-23278790">Hellinheavns GIF - Hellinheavns - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2311.03658">The Linear Representation Hypothesis and the Geometry of Large Language Models</a>: Informally, the &#39;linear representation hypothesis&#39; is the idea that high-level concepts are represented linearly as directions in some representation space. In this paper, we address two close...</li><li><a href="https://www.youtube.com/watch?v=Xf1YF_MH1xc">Beastie Boys - Root Down</a>: REMASTERED IN HD!Read the story behind Ill Communication here: https://www.udiscovermusic.com/stories/ill-communication-beastie-boys-album/Listen to more fro...</li><li><a href="https://youtu.be/zK1mLIeXwsQ?t=119">deadmau5 &amp; Kaskade - I Remember (HQ)</a>: ▶︎ https://deadmau5.ffm.to/randomalbumtitle follow deadmau5 &amp; friends here: https://sptfy.com/PjDOcurrent tour info here: https://deadmau5.com/showsjoin the ...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1231336533261553876)** (20 messages🔥): 

- **DeepMind's New Toolkit for Neural Networks**: Google DeepMind has introduced [Penzai](https://github.com/google-deepmind/penzai), a JAX research toolkit designed to build, edit, and visualize neural networks, aiming to enhance the way researchers interact with their models.

- **Call for Beta Testers for Advanced Research Assistant**: Rubik.ai is seeking beta testers for an advanced research assistant and search engine featuring models like **Claude 3 Opus, GPT-4 Turbo**, and others, offering two months of free premium access with the promo code `RUBIX`.

- **Exploring Loss Curves in Training Large Language Models**: Discussions revolved around diagnosing and understanding the unusual patterns in loss curves while training models, with speculation that low batch sizes and uneven loss landscapes might be contributing factors.

- **Archive of GPT System Prompts Now Available**: [EveryoneIsGross/GPTs](https://github.com/EveryOneIsGross/GPTs) hosts a collection of system prompts for GPT experiments, which include implementations of various papers and experiments in embeddings, RP, RAG, and other concepts.

- **Reddit Post Questions LMSYS Benchmark's Validity**: A [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1c9nvpy/lmsys_becoming_less_useful) challenges the usefulness of the LMSYS benchmark, suggesting it is becoming less reliable due to the difficulty in crafting questions that accurately differentiate model intelligence.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1c9nvpy/lmsys_becoming_less_useful/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-fineweb-15t-tokens-of-commoncrawl">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/vikhyatk/status/1782296370420072576">Tweet from vik (@vikhyatk)</a>: weird loss curve, won&#39;t be able to sleep tonight if i don&#39;t figure out what&#39;s causing those dips early on</li><li><a href="https://github.com/google-deepmind/penzai">GitHub - google-deepmind/penzai: A JAX research toolkit for building, editing, and visualizing neural networks.</a>: A JAX research toolkit for building, editing, and visualizing neural networks. - google-deepmind/penzai</li><li><a href="https://github.com/EveryOneIsGross/GPTs">GitHub - EveryOneIsGross/GPTs: loading zone for my GPT experiments and tools.</a>: loading zone for my GPT experiments and tools. Contribute to EveryOneIsGross/GPTs development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1231135314446389270)** (650 messages🔥🔥🔥): 

- **LLaMA vs Phi Showdown**: Discussions intensify as members compare the **newly released Phi-3-mini model** against [LLaMA-3](https://x.com/sebastienbubeck/status/1782627991874678809?s=46) and **GPT-3.5**. The performance of **Phi-3-mini**, especially in 4-bit quantization, is scrutinized with concerns over repetitive output and model weights eagerly awaited.
- **Technical Glitches at Hugging Face**: Hugging Face faces downtime, with speculations around the new **[FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)** or **LLaMA-3's demand** possibly contributing to the outages. While service has intermittently returned, issues persist.
- **Tricky Model Behavior**: Conversations around **LLaMA-3** indicate a propensity for the models to hallucinate or fail to embrace new information after fine-tuning. The **Phi-3-mini** model, in particular, is reported to have issues with stopping generation and may have a misconfigured **EOS token**.
- **Efficiency in Model Fine-Tuning**: Members talk about **QLoRA versus LoRA** for fine-tuning large language models and share opinions on their effectiveness and potential uses in production, notably with references to [QLoRA research](https://twitter.com/teortaxesTex/status/1781963108036088060).
- **Emerging Developer Interest**: Calls are made for developers engaged in models, datasets, or systems using AI models to connect, suggesting a growing community keen on discussing and potentially collaborating on AI and NLP projects.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/suchenzang/status/1782830272792404232">Tweet from Susan Zhang (@suchenzang)</a>: it seems to enjoy talking itself out of the right solution...</li><li><a href="https://evalplus.github.io/leaderboard.html">EvalPlus Leaderboard</a>: no description found</li><li><a href="https://x.com/natolambert/status/1782600141159174398">Tweet from Nathan Lambert (@natolambert)</a>: i really hope phi 3 proves us wrong about evaluation doping and it is actually an amazing model.  But, being an outlier on log compute &lt;-&gt; MMLU plots is a little sus.</li><li><a href="https://x.com/awnihannun/status/1782436898285527229">Tweet from Awni Hannun (@awnihannun)</a>: Next level: QLoRA fine-tuning 4-bit Llama 3 8B on iPhone 15 pro.  Incoming (Q)LoRA MLX Swift example by David Koski: https://github.com/ml-explore/mlx-swift-examples/pull/46 works with lot&#39;s of mo...</li><li><a href="https://huggingface.co/blog/how-to-train-sentence-transformers">Train and Fine-Tune Sentence Transformers Models</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.10198">How faithful are RAG models? Quantifying the tug-of-war between RAG and LLMs&#39; internal prior</a>: Retrieval augmented generation (RAG) is often used to fix hallucinations and provide up-to-date knowledge for large language models (LLMs). However, in cases when the LLM alone incorrectly answers a q...</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="https://x.com/huybery/status/1781172838361334015">Tweet from Binyuan Hui (@huybery)</a>: Just evaluated coding abilities of Llama3-8B-base👇🏻</li><li><a href="https://huggingface.co/abacaj/phi-2-super">abacaj/phi-2-super · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/rage-gif-24341837">Rage GIF - Rage - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/gui_penedo/status/1781953413938557276?s=46">Tweet from Guilherme Penedo (@gui_penedo)</a>: We have just released 🍷 FineWeb: 15 trillion tokens of high quality web data. We filtered and deduplicated all CommonCrawl between 2013 and 2024. Models trained on FineWeb outperform RefinedWeb, C4, ...</li><li><a href="https://huggingface.co/papers/2404.14047">Paper page - How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study</a>: no description found</li><li><a href="https://x.com/sebastienbubeck/status/1782627991874678809?s=46">Tweet from Sebastien Bubeck (@SebastienBubeck)</a>: phi-3 is here, and it&#39;s ... good :-).  I made a quick short demo to give you a feel of what phi-3-mini (3.8B) can do. Stay tuned for the open weights release and more announcements tomorrow mornin...</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct">microsoft/Phi-3-mini-4k-instruct · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/playground?models=meta-llama/llama-3-70b-instruct">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://github.com/mozilla-Ocho/llamafile?tab=readme-ov-file#using-llamafile-with-external-weights">GitHub - Mozilla-Ocho/llamafile: Distribute and run LLMs with a single file.</a>: Distribute and run LLMs with a single file. Contribute to Mozilla-Ocho/llamafile development by creating an account on GitHub.</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/tokenizer_config.json">tokenizer_config.json · microsoft/Phi-3-mini-128k-instruct at main</a>: no description found</li><li><a href="https://github.com/stanfordnlp/pyreft">GitHub - stanfordnlp/pyreft: ReFT: Representation Finetuning for Language Models</a>: ReFT: Representation Finetuning for Language Models - stanfordnlp/pyreft</li><li><a href="https://www.youtube.com/watch?v=z5rRZdiu1UE">Beastie Boys - Sabotage</a>: REMASTERED IN HD!Read the story behind Ill Communication here: https://www.udiscovermusic.com/stories/ill-communication-beastie-boys-album/Listen to more fro...</li><li><a href="https://huggingface.co/datasets/Replete-AI/OpenCodeInterpreterData">Replete-AI/OpenCodeInterpreterData · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Rombo-Hermes-2.5-Extra-code">Replete-AI/Rombo-Hermes-2.5-Extra-code · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb">HuggingFaceFW/fineweb · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Rombo-Hermes-2.5-Extra-code-sub-50k">Replete-AI/Rombo-Hermes-2.5-Extra-code-sub-50k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Rombo-Hermes-2.5-Extra-code-Medium">Replete-AI/Rombo-Hermes-2.5-Extra-code-Medium · Datasets at Hugging Face</a>: no description found</li><li><a href="https://fast.snova.ai">Streamlit</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1231255157178634414)** (78 messages🔥🔥): 

- **Dealing with OOM in Zero 3**: A user reports that Deepspeed **Zero 3** is significantly slower than **Zero 2** and experiences OOM errors even with CPU offloading, wondering about normal behavior and [seeking advice for optimal usage](https://discord.com/channels/1053877538025386074/1149866623109439599/1231634503714340955).
- **Single-GPU Optimization vs. NVLink**: One user ponders the best way to utilize dual RTX 3090s with NVLink for a single prompt to enhance performance while another suggests **single-GPU** usage is fastest, citing synchronization overhead with **multi-GPU** setups.
- **Llama Fine-tuning and Training Guidelines**: Discussions touch upon synthetic data generation for finetuning models within licensing rules, with one user warning against using generated data to improve non-**Llama** models and others discussing the correct ratios for example difficulty in finetuning.
- **Learning Rate Techniques and Forgetting in LLMs**: Users discuss whether techniques like **discriminative learning rates** and **gradual unfreezing** are prevalent in 2024, with one user unfamiliar and another confirming they are indeed in use.
- **Finding Suitable Fine-tuning Guides**: Multiple users suggest the best practices and resources for instruction fine-tuning, with preferences for Hugging Face blogs, avoiding Medium articles, and specific recommendations like tutorials on **Labonne's GitHub**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/chargoddard/mistral-11b-slimorca">chargoddard/mistral-11b-slimorca · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.01364">Continual Learning for Large Language Models: A Survey</a>: Large language models (LLMs) are not amenable to frequent re-training, due to high training costs arising from their massive scale. However, updates are necessary to endow LLMs with new skills and kee...</li><li><a href="https://arxiv.org/abs/2404.08865">LLM In-Context Recall is Prompt Dependent</a>: The proliferation of Large Language Models (LLMs) highlights the critical importance of conducting thorough evaluations to discern their comparative advantages, limitations, and optimal use cases. Par...</li><li><a href="https://arxiv.org/abs/2212.08037">Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models</a>: Large language models (LLMs) have shown impressive results while requiring little or no direct supervision. Further, there is mounting evidence that LLMs may have potential in information-seeking scen...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1231590177172881519)** (7 messages): 

- **New Benchmark for AI Vision Models**: xAI released their **RealWorldQA** benchmark dataset designed for **Grok-1.5-vision-preview**, offering direct question-answer scenarios.
- **Confusion over Dataset's Purpose**: There was a brief confusion whether the **RealWorldQA** was a training set or a benchmark, later clarified to be a benchmark as mentioned on xAI's [blog post about Grok-1.5](https://x.ai/blog/grok-1.5v).
- **Additional Dataset Interest**: Some members expressed enthusiasm for the new benchmark dataset, suggesting it could be useful for testing future versions of **Obsidian**.
- **Desire for Training Sets**: Despite recognizing the usefulness of the benchmark data, members still indicated an interest in having access to a training dataset.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-1.5v">Grok-1.5 Vision Preview</a>: no description found</li><li><a href="https://huggingface.co/datasets/xai-org/RealworldQA?row=2">xai-org/RealworldQA · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1231238181647679539)** (89 messages🔥🔥): 

- **Evaluating RAG with LLaMA**: Discussion centers around evaluating the **Retrieval-Augmented Generation (RAG)** performance using the LLaMA index, suggesting that **Mistral 7b v2** seems to outperform other models like LLaMA 3b instruct. A useful resource for this evaluation is shared: [OpenAI Cookbook example](https://cookbook.openai.com/examples/evaluation/evaluate_rag_with_llamaindex).

- **Deciphering Superposition Prompting**: The community explores a paper on a new RAG prompting method called *superposition prompting* that aims to process long contexts more efficiently ([Superposition Prompting Paper](https://arxiv.org/abs/2404.06910)). A member shares their practical use of the method in production with considerations about ordering the context.

- **Researchers Share RAG Insights**: Several papers on **RAG** methodologies were shared, highlighting innovations like improving retrieval with LLMs and credibility-aware generation, as well as addressing challenges in long-context inference. Notably, an overview paper details the evolution and organization of the **RAG** framework ([RAG Evolution Paper](https://arxiv.org/abs/2404.10981)).

- **Function-Calling RAG Techniques**: Blog posts by Pamela Fox on **RAG** techniques using function-calling were cited extensively as resources that do heavy-lifting for understanding and implementing **RAG** approaches ([Pamela Fox's RAG post](https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html)). Additionally, the GitHub repository from Azure-Samples serves as an exemplar for setting up **RAG** approaches ([Azure-Samples GitHub](https://github.com/Azure-Samples/azure-search-openai-demo/tree/main/app/backend/approaches)).

- **Fusion of Retrieval and Generation in RAG**: Conversation leads towards integrating retrieval as part of an LLM’s plan to create semi-structured output grounded in document references. Examples included a blend of Cohere's and Claude-3's capabilities to demonstrate this approach, along with a call for creating benchmarks for **RAG** models that synthesize information from multiple documents ([CLA Document Format](https://docs.cohere.com/docs/retrieval-augmented-generation-rag)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/BlancheMinerva/status/1782437494585282965">Tweet from Stella Biderman (@BlancheMinerva)</a>: Create a benchmark for RAG models where all of the questions require information from multiple documents to be synthesized answer them. Study how models trained on publicly released data do on it and ...</li><li><a href="https://arxiv.org/abs/2404.10981">A Survey on Retrieval-Augmented Text Generation for Large Language Models</a>: Retrieval-Augmented Generation (RAG) merges retrieval methods with deep learning advancements to address the static limitations of large language models (LLMs) by enabling the dynamic integration of u...</li><li><a href="https://arxiv.org/abs/2404.06910">Superposition Prompting: Improving and Accelerating Retrieval-Augmented Generation</a>: Despite the successes of large language models (LLMs), they exhibit significant drawbacks, particularly when processing long contexts. Their inference cost scales quadratically with respect to sequenc...</li><li><a href="https://docs.anthropic.com/claude/docs/long-context-window-tips">Long context window tips</a>: no description found</li><li><a href="https://cookbook.openai.com/examples/evaluation/evaluate_rag_with_llamaindex">Evaluate RAG with LlamaIndex | OpenAI Cookbook</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.05825">LLM-Augmented Retrieval: Enhancing Retrieval Models Through Language Models and Doc-Level Embedding</a>: Recently embedding-based retrieval or dense retrieval have shown state of the art results, compared with traditional sparse or bag-of-words based approaches. This paper introduces a model-agnostic doc...</li><li><a href="https://blog.pamelafox.org/2024/03/rag-techniques-using-function-calling.html">RAG techniques: Function calling for more structured retrieval</a>: no description found</li><li><a href="https://blog.pamelafox.org/2024/02/rag-techniques-cleaning-user-questions.html">RAG techniques: Cleaning user questions with an LLM</a>: no description found</li><li><a href="https://github.com/Azure-Samples/azure-search-openai-demo/tree/main/app/backend/approaches">azure-search-openai-demo/app/backend/approaches at main · Azure-Samples/azure-search-openai-demo</a>: A sample app for the Retrieval-Augmented Generation pattern running in Azure, using Azure AI Search for retrieval and Azure OpenAI large language models  to power ChatGPT-style and Q&amp;amp;A experie...</li><li><a href="https://docs.cohere.com/docs/retrieval-augmented-generation-rag">Retrieval Augmented Generation (RAG) - Cohere Docs</a>: no description found</li><li><a href="https://blog.pamelafox.org/2024/03/evaluating-rag-chat-apps-can-your-app.html">Evaluating RAG chat apps: Can your app say "I don't know"?</a>: no description found</li><li><a href="https://github.com/HK3-Lab-Team/PredCST">GitHub - HK3-Lab-Team/PredCST: Learning Predictive Models of Concrete Syntax Tree from text.</a>: Learning Predictive Models of Concrete Syntax Tree from text. - HK3-Lab-Team/PredCST</li><li><a href="https://arxiv.org/abs/2404.06809">Not All Contexts Are Equal: Teaching LLMs Credibility-aware Generation</a>: The rapid development of large language models has led to the widespread adoption of Retrieval-Augmented Generation (RAG), which integrates external knowledge to alleviate knowledge bottlenecks and mi...</li><li><a href="https://arxiv.org/abs/2404.06347">RAR-b: Reasoning as Retrieval Benchmark</a>: Semantic textual similartiy (STS) and information retrieval tasks (IR) tasks have been the two major avenues to record the progress of embedding models in the past few years. Under the emerging Retrie...</li><li><a href="https://arxiv.org/abs/2404.06082">A RAG Method for Source Code Inquiry Tailored to Long-Context LLMs</a>: Although the context length limitation of large language models (LLMs) has been mitigated, it still hinders their application to software development tasks. This study proposes a method incorporating ...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1231147992568959027)** (343 messages🔥🔥): 

- **Creative AI Alternatives Take Stage**: While awaiting the official **WorldSim** platform's return, many users have shifted to alternative interpretations like **Super WorldSim** and **Snow World Simulator** hosted on **HuggingChat**. They are tailoring these alternatives to offer specialized experiences, such as crafting superhero universes or playing D&D-like games.
- **Super WorldSim Evolves with Improvements**: Continuing updates from **Jetblackrlsh** are introducing new features to **Super WorldSim**, such as **Mind Meld** and **Improv**, enhancing the user experience and aligning closer to the sophistication of **Claude Opus**.
- **Community Imaginations Flourish**: Amidst the platform alternatives, users are engaging deeply, evolving complex fictional worlds, and generating extensive phylogenetic trees to document their simulated species' development over millions of years.
- **Discord as a Stage for Democratic World Building**: A notable trend is emerging with users like **Rundeen** setting up democratically controlled WorldSim bots on **Discord**. The community is enthusiastic about the potential for collaborative story-building and exploration.
- **Open Models Pave Future of AI Simulations**: A consensus seems to be forming that open-source AI models will be significant for future WorldSim-like experiences. **Llama 3's** anticipated larger models have caught particular attention for their potential in driving these creative simulations forward.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/">world_sim</a>: no description found</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://console.groq.com/playground?model=llama3-70b-8192">GroqCloud</a>: Experience the fastest inference in the world</li><li><a href="https://hf.co/chat/assistant/662404223e2307950aa903bc">Super World Sim - HuggingChat</a>: Use the Super World Sim assistant inside of HuggingChat</li><li><a href="https://hf.co/chat/assistant/66252be0705754b4e74c5e3f">Snow World Simulator - HuggingChat</a>: Use the Snow World Simulator assistant inside of HuggingChat</li><li><a href="https://hf.co/chat/assistant/6626e4869232378718adc5f2">Snow Singer Simulator - HuggingChat</a>: Use the Snow Singer Simulator assistant inside of HuggingChat</li><li><a href="https://a.co/d/0gve1yp">no title found</a>: no description found</li><li><a href="https://books2read.com/u/3GPpKP">Available now at your favorite digital store!</a>: The Architects&#x27; Conundrum: Quantumom vs. Data Dad by Nicholas Alexander Benson</li><li><a href="https://hf.co/chat/assistant/65bff23f5560c1a5c0c9dcbd">Image Generator - HuggingChat</a>: Use the Image Generator assistant inside of HuggingChat</li><li><a href="https://tinyurl.com/SuperWorldSim">Super World Sim - HuggingChat</a>: Use the Super World Sim assistant inside of HuggingChat</li><li><a href="https://hf.co/chat/assistant/66248a7a29ce1e0f4dd260fe">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://hf.co/chat/assistant/66252be0705754b4e74">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://hf.co/chat/assistant/662404223e230">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://hf.co/chat/assistant/6623fcdb1a7a58ed5e441db2">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://hf.co/chat/assistant/66240">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://hf.co/chat/assistant/662404223e2307">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://www.suzannetreister.net/Ampages/Amenu.html">Suzanne Treister - Amiga Videogame Stills - menu</a>: no description found</li><li><a href="https://dreams-of-an-electric-mind.webflow.io/eternal">eternal mode • infinite backrooms</a>: the mad dreams of an artificial intelligence - not for the faint of heart or mind
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1231142060816334859)** (635 messages🔥🔥🔥): 

- **GPU Offloading and System Resource Usage**: Users discussed the performance of **LM Studio** on various GPUs, with specific concerns about running models on **AMD GPUs** using ROCm and **Nvidia GPUs**. It was noted that **GPU offloading** is necessary for maximizing performance and if the system isn't offloading correctly, it could use the CPU at 100%, causing inefficiency.

- **Issues with LM Studio and Hugging Face**: Users reported concerns regarding the inability to search and download models due to **Hugging Face** downtime, which seemed to affect LM Studio's functionality, showing error messages like **503** and **500**. **Heyitsyorkie** confirmed that Hugging Face was having API issues affecting model explorer functionalities.

- **Utilizing LLMs in LM Studio**: Users sought advice on creating specific **system prompts** for models to role-play scenarios like a **D&D campaign**, as well as how to handle **max token limits** and **rolling windows** within conversations. One suggestion was to use the "**AI assistant (python)**" preset in LM Studio, ending prompts with an example of the expected JSON schema.

- **Model and API Issues**: Discussions included queries regarding **loading specific models**, issues with **unsupported processor instructions** like AVX2, handling **authorization problems**, and **error messages** such as 'Unsupported format'. Users requested potential fixes and workarounds.

- **AI Models and Quantization Questions**: Users probed into the differences between various AI model quantizations (e.g., **IQ1M vs. IQ2XS**) and discussed the upcoming **Llama 3 400b model**, conjecturing about system requirements and capacity to run such large models.

- **LM Studio Feature Requests and Feedback**: Users expressed a desire for features like **running LM Studio in the background**, and questioned the lack of a **privacy policy.** Praise was also given for making **AI accessible** through LM Studio.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/lmstudioai/status/1782390856986550384?s=46">Tweet from LM Studio (@LMStudioAI)</a>: Model search / download within LM Studio may be impacted by this Hugging Face downtime.  Stay tuned for updates  ↘️ Quoting Hugging Face Status (@hf_status)   We&#39;re experiencing some downtime on h...</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://docs.useanything.com/feature-overview/llm-selection/lmstudio">LMStudio | AnythingLLM by Mintplex Labs</a>: no description found</li><li><a href="https://lmstudio.ai/docs/local-server">Local LLM Server | LM Studio</a>: You can use LLMs you load within LM Studio via an API server running on localhost.</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found</li><li><a href="https://www.youtube.com/@IBMTechnology/playlists">IBM Technology</a>: Whether it’s AI, automation, cybersecurity, data science, DevOps, quantum computing or anything in between, we provide educational content on the biggest topics in tech. Subscribe to build your skills...</li><li><a href="https://ollama.com/blog/openai-compatibility">OpenAI compatibility · Ollama Blog</a>: Ollama now has initial compatibility with the OpenAI Chat Completions API, making it possible to use existing tooling built for OpenAI with local models via Ollama.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c858ac/llama3_seems_to_get_stuck_in_loops_sometimes/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/tree/main">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF at main</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard">Big Code Models Leaderboard - a Hugging Face Space by bigcode</a>: no description found</li><li><a href="https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat-GGUF">Qwen/CodeQwen1.5-7B-Chat-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1">Vision Models (GGUF) - a lmstudio-ai Collection</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ca8uxo/llavallama38b_is_released/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g&">[1hr Talk] Intro to Large Language Models</a>: This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What the...</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF/tree/main">lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF at main</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c9m6ei/lpt_llama_3_doesnt_have_selfreflection_you_can/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/Crizomb/ai_pdf">GitHub - Crizomb/ai_pdf: Chat locally with any PDF  Ask questions, get answer with usefull references  Work well with math pdfs (convert them to LaTex, a math syntax comprehensible by computer)</a>: Chat locally with any PDF  Ask questions, get answer with usefull references  Work well with math pdfs (convert them to LaTex, a math syntax comprehensible by computer) - Crizomb/ai_pdf</li><li><a href="https://github.com/BBC-Esq/VectorDB-Plugin-for-LM-Studio">GitHub - BBC-Esq/VectorDB-Plugin-for-LM-Studio: Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode!</a>: Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode! - BBC-Esq/VectorDB-Plugin-for-LM-Studio</li><li><a href="https://github.com/mlabonne/llm-course">GitHub - mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.</a>: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks. - mlabonne/llm-course</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c8c7xj/easiest_way_to_setup_rag_windows_nvidia_gpu/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://status.huggingface.co/">
Hugging Face status
</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1231136215638736983)** (314 messages🔥🔥): 

- **Llama 3 and Alternative Models**: Users are exploring various versions of **Llama 3** for better performance, comparing it against models like **Goliath 120B** and discussing **Mistral**. Conversations include the performance of **Llama 3** in benchmarks and whether finetuning the variants could match up to **GPT-4**.

- **Meta-Llama-3-8B-Instruct-GGUF Trepidation**: Concerns are raised about an **infinity generation issue** with **Llama 3 8B Instruct GGUF**, where the model continues generating content endlessly. Users are suggesting fixes involving stop strings and considering trying different model versions.

- **In Search of Unrestricted Content Creation**: A discussion took place on the level of **content restriction** in different models like **Llama 3**, with suggestions to modify the system prompt to reduce censorship.

- **Phi-3 Excites and Entices**: Members are evaluating **Phi-3**, noting its impressive performance on certain tasks despite its smaller size compared to larger models. There's anticipation about **Phi-3** compatibility and performance with **LM Studio**.

- **Technical Troubleshooting and Version Queries**: Users seek help and clarification on **LM Studio**'s capabilities to handle models like **Meta-Llama-3-8B-Instruct-Q4_K_M.gguf**, the impact of **context size** on model performance, and **OpenAI's GPT-4** setting high standards for comparison. There are also mentions of running **LM Studio** on a **headless server**, and explanations on the meaning of terms like "mog".
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/AI-Engine/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_k_m_with_temp_stop_token_fix.gguf?download=true">no title found</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://tide-freckle-52b.notion.site/1e0168e3481747ebaa365f77a3af3cc1?v=83e3d58d1c3c45ad879834981b8c2530">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://tenor.com/view/yoda-star-wars-learning-gif-21964563">Yoda Star GIF - Yoda Star Wars - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://doc.pypy.org/en/latest/sandbox.html">PyPy’s sandboxing features &mdash; PyPy documentation</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf">microsoft/Phi-3-mini-4k-instruct-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/models?other=base_model:meta-llama/Meta-Llama-3-8B-Instruct">Models - Hugging Face</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/llama3.preset.json">configs/llama3.preset.json at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs</li><li><a href="https://x.com/hrishioa/status/1782429651962675410">Tweet from Hrishi (@hrishioa)</a>: Is anyone finetuning an instruct version of llama3-42b? Would be really interesting if it can serve as a good/smart/client-side GPT-4 replacement  https://www.reddit.com/r/LocalLLaMA/comments/1c9u2jd/...</li><li><a href="https://huggingface.co/chargoddard/llama3-42b-v0">chargoddard/llama3-42b-v0 · Hugging Face</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>: Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1231950083847880788)** (1 messages): 

- **Hugging Face Downtime Affects LM Studio**: LM Studio's model search and download functionality may be currently impaired due to [Hugging Face downtime](https://x.com/lmstudioai/status/1782390856986550384?s=46). The team is monitoring the situation and promises to provide updates as they come.

**Link mentioned**: <a href="https://x.com/lmstudioai/status/1782390856986550384?s=46">Tweet from LM Studio (@LMStudioAI)</a>: Model search / download within LM Studio may be impacted by this Hugging Face downtime.  Stay tuned for updates  ↘️ Quoting Hugging Face Status (@hf_status)   We&#39;re experiencing some downtime on h...

  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1231175264730615889)** (27 messages🔥): 

- **Llama3 Encountering a Load Issue**: Multiple users report issues loading models with **Llama3** after the 0.2.20 update, prompting suggestions to post detailed problems in a specific channel. The error logs show a generic "Error loading model" without suggestions, hinting at a potential bug due to recent updates.
- **Gratitude for LM Studio**: A professional writer and AI researcher expressed deep appreciation for **LM Studio**, stating it significantly aids their productivity. This heartfelt feedback underscores the impact of LM Studio on users' workflow.
- **Unexpected Model Behavior Noted**: A user observed *llama* models sometimes outputting numbers instead of answers when asked general topics. This unusual behavior suggests a potential glitch in model responses.
- **VPN Causes Certificate Issues with LM Studio**: Users with **Zscaler VPN** are unable to download models in **LM Studio** due to "unable to get local issuer certificate" errors. Workarounds mentioned include downloading models on a different machine, but underlying mechanisms remain unclear, as exiting the VPN resolves the issue.
- **Queries for Hugging Face Models in LM Studio Trigger Errors**: There's a 500 error when searching for particularly popular models on **LM Studio**. Users speculate that Hugging Face may be blocking terms like "Llama" or "Llama3" due to heavy traffic, while alternative searches using "lmstudio-community" work fine.
  

---


**LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1231148383419502682)** (12 messages🔥): 

- **Seeking Full Code Output**: A user asked for a way to make the LLM always write full code instead of inserting comments like *// Add similar event listeners for left and right buttons*.
- **Exploring Endless Adventure**: Someone inquired about the best prompt for creating an endless sandbox adventure simulation game using **Llama3** and also pondered whether Llama3 can generate prompts for itself.
- **Configuring Llama-3-Smaug-8B Prompts**: A member sought assistance configuring prompts in LM Studio for the [Llama-3-Smaug-8B model](https://huggingface.co/bartowski/Llama-3-Smaug-8B#prompt-format) and wondered about the correct usage of system and user prefixes and suffixes, as their attempts led to non-stop output.
- **Prompt Configuration Clarification**: Another user clarified that configuring prompts for the questioned model is the same as the regular included llama 3 preset in v0.2.20 of LM Studio.
- **LM Studio Update and Model Search Issue**: Following a discussion about the latest LM Studio build, a 503 error code issue when searching for models was reported, with a respondent referencing a Discord channel link for further assistance, but the link was provided as 'null'.

**Link mentioned**: <a href="https://huggingface.co/bartowski/Llama-3-Smaug-8B-GGUF#prompt-format.">bartowski/Llama-3-Smaug-8B-GGUF · Hugging Face</a>: no description found

  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1231145707180658769)** (59 messages🔥🔥): 

- **Searching for Suitable GPUs**: Users in the channel discussed upgrading laptops to run LLMs with NVIDIA GPUs. A guide was shared from Reddit, titled [The LLM GPU Buying Guide - August 2023](https://www.reddit.com/r/LocalLLaMA/comments/15rwe7t/the_llm_gpu_buying_guide_august_2023/), but it was noted that upgrading GPUs in laptops is uncommon and may require external solutions for some machines.
- **Troubleshooting Model Loading Errors**: A user encountered an "Error loading model" issue where the GPU type was not detected, with the suggestion made to turn off GPU offloading in the settings panel which subsequently resolved the problem.
- **Optimizing Hardware for Model Use**: There were discussions about power consumption and efficiency related to using secondary GPUs like a GTX 1060 for running larger models, with consensus suggesting it's worth testing but to keep expectations low due to potential latency and power draw.
- **Model Preferences for Research Papers**: User queries about the best models for writing research papers led to mentions of Llama 3 8B and Claude 3, with the former being criticized for AI-like responses and the latter having limitations for free users.
- **Mac Memory Potential for Running LLMs**: Questions regarding the capabilities of a new 128 GB Mac to run large models like Grok sparked discussions; with suggestions made to conserve memory for the OS and a link provided to increase VRAM allocation using a `sudo` command on macOS. Further, it was implied that the Mac Ultra 2 with 192 GB of RAM can run 120b models well.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15rwe7t/the_llm_gpu_buying_guide_august_2023/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1231136152832970753)** (10 messages🔥): 

- **LMStudio's Local Model Detection Glitch**: A member reported an issue with **LMStudio** failing to detect locally saved files within a models directory that contains an NFS mount. Despite working in version 0.2.19 beta B, the issue arose in versions 0.2.19 beta C, 0.2.19, and 0.2.20.

- **File System Hierarchy Hassle**: Another member discussed the possible **directory structure requirements** of LMStudio, suggesting that additional directory levels above the typical maintainer/model hierarchy might contribute to the problem rather than NFS factors. The original poster confirmed using an additional directory level to differentiate between local and external storage.

- **Directory Testing Advice**: It was advised to confirm the directory structure as a potential cause of the problem by testing with a local file system, ensuring that models in new sub-directories are discovered and identified by the LMStudio app.

- **Token Misconceptions Clarified**: In the context of tokenization, members discussed that tokens in models do not necessarily align with syllables but can include various subword components like roots, prefixes, and suffixes. The nature of the language model's complexity in understanding words and tokens was explored.

- **Language Token Quantification**: A member queried the convention around the number of tokens used during language model training, reflecting on whether 50,000 tokens is a standard number due to tradition, efficacy, or a balance between complexity and model performance.
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1231343022675984514)** (20 messages🔥): 

- **Trouble with Autogen and Local LM Llama 3**: Users are experiencing issues with Autogen when pointing it at local LM Llama 3, where it processes only 2 tokens and then stops. One expressed frustration, as the LM appears to be functioning but returns data prematurely.

- **A No-Marketing Zone**: A member was reminded that marketing tools is not permitted on this server and asked to refrain from such activities in the future.

- **Potential Fix for Token Limitation**: A user encountered a similar issue and suggested *replacing "max tokens with 3000"* which seemed to resolve the problem for them. They also advised restarting Autogen, creating a new agent, and a new workflow afterwards.

- **User Proxy Quirks within Autogen**: There are also reports of the user proxy occasionally stopping its output abruptly or parroting phrases like *"good job you did it"* which diminishes the user experience, particularly in comparison to using the direct API.

- **Issues with AutoGen Manager Agent**: Another user inquired about difficulties in getting the AutoGen Manager agent to work with a local model, specifically running into an *"unable to select speaker error"*. There was no resolution suggested within the provided messages.
  

---


**LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1231516145694150656)** (1 messages): 

- **Inquiry about Project Integration**: A member asked if there is a way to integrate a certain tool with **LM Studio**, expressing interest in accessing specific **LM Studio project information** if available.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1231155214757924864)** (42 messages🔥): 

- **Meta Llama 3 LLM Excites Users**: The **Meta Llama 3** family of language models was shared, boasting dialogue optimization, helpfulness, and safety. Users are using these models successfully in LM Studio, as described in their [Hugging Face repository details](https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF).

- **Performance Discussions on AMD Hardware**: Members indicated **Meta-Llama-3-70B** and **Meta-Llama-3-8B** models having token generation speeds of around 20 tok/s and 60 tok/s respectively on AMD GPUs such as the **7900xtx**. There's curiosity about whether future versions might run on lower-end hardware.

- **ROCm Utilization Queries**: A user highlighted irregular GPU utilization when inferring large models on a dual 7900XTX setup with the ROCm tech preview. The combined GPU usage didn't reflect full utilization of one card.

- **Issues and Fixes with LM Studio ROCm Preview**: Users report bugs with gpu offloading in different versions of LM Studio ROCm preview. One user mentioned solving their issue by removing certain environment variables, while another switched to the regular LM Studio build due to unsupported hardware.

- **LM Studio GPU Selection Troubles and Solutions**: Users discussed challenges in directing LM Studio to use a dedicated AMD GPU over an integrated one. Solutions suggested include disabling the integrated GPU in BIOS and manually setting environment variables like `HIP_VISIBLE_DEVICES`.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF">NousResearch/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.howtogeek.com/disable-integrated-graphics-on-windows/">How to Disable Your Integrated Graphics on Windows 11</a>: When games and other graphics-intensive applications starts to lag, this is what you do!</li><li><a href="https://techteamgb.co.uk/2024/03/22/how-to-turn-your-amd-gpu-into-a-local-llm-beast-a-beginners-guide-with-rocm/">How to Turn Your AMD GPU into a Local LLM Beast: A Beginner’s Guide with ROCm | TechteamGB</a>: no description found</li><li><a href="https://youtu.be/VXHryjPu52k?t=249">How to Turn Your AMD GPU into a Local LLM Beast: A Beginner&#39;s Guide with ROCm</a>: RX 7600 XT on Amazon (affiliate): https://locally.link/kEJGLM Studio: https://lmstudio.ai/rocmProducts provided by GigabyteThose of us with NVIDIA GPUs, part...
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1231141833241788416)** (34 messages🔥): 

- **X11 Forwarding as a GUI Solution**: Members discussed using X forwarding with the `ssh -X` command as a way to use [Nsight Compute GUI via SSH](https://goteleport.com/blog/x11-forwarding/), and a user successfully set up the GUI and provided a [step-by-step guide](https://tspeterkim.github.io/posts/nsight-setup-on-ec2) for others to use Nsight Compute to profile remote GPUs.
- **Enhancing LLM Inference with 'Effort'**: The new 'Effort' algorithm allows dynamic adjustment of the number of calculations during LLM inference and is detailed in a project where [the source code is available on GitHub](https://github.com/kolinko/effort). Discussion suggested interest in implementing the algorithm in other settings like Triton or CUDA.
- **DGX Boxes Come NVLinked**: It was clarified that DGX boxes generally ship with NVLink installed, as they use SXM socket GPUs, supported by a resource explaining [Nvidia's NVLink and NVSwitch](https://fuse.wikichip.org/news/1224/a-look-at-nvidias-nvlink-interconnect-and-the-nvswitch/).
- **CUDA Matrix Multiplication Clarification**: A user was confused about CUDA code for matrix multiplication; another member explained the operation as computing the dot product of a row and a column from two matrices.
- **Syncing Threads in CUDA**: There was a conversation around the behavior of `__syncthreads()` in CUDA, noting that starting with Volta, all non-exited threads in the block must reach the sync point, which is a change from older architectures where __syncthreads() would ignore exited threads.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tspeterkim.github.io/posts/nsight-setup-on-ec2">How to set up Nsight Compute Locally to profile Remote GPUs</a>: no description found</li><li><a href="https://kolinko.github.io/effort/">Effort Engine</a>: A possibly new algorithm for LLM Inference. Adjust smoothly - and in real time - how many calculations you'd like to do during inference.</li><li><a href="https://goteleport.com/blog/x11-forwarding/">What You Need to Know About X11 Forwarding</a>: In this blog post, we&#x27;ll deep-dive into X11 Forwarding, explaining what X11 is and how it works under the hood.</li><li><a href="https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#remote-connections">3. Nsight Compute &mdash; NsightCompute 12.4 documentation</a>: no description found</li><li><a href="https://fuse.wikichip.org/news/1224/a-look-at-nvidias-nvlink-interconnect-and-the-nvswitch/">A look at Nvidia&#39;s NVLink interconnect and the NVSwitch</a>: A look at Nvidia&#39;s NVLink interconnect and the 2-billion transistor NVSwitch that is powering Nvidia&#39;s latest DGX-2 deep learning machine.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1231376189097119844)** (46 messages🔥): 

- **Grayscale Conversion Quirks Unveiled**: A member faced issues with grayscaling an image using Triton after resizing without changing its dimensions, resulting in aberrant images. They shared a gist for reproduction at [GitHub Gist](https://gist.github.com/alexandremuzio/3ba9d8669f57718139da36158180baaf) and the original tutorial [Jupyter Notebook](https://github.com/cuda-mode/lectures/blob/main/lecture%2014/A_Practitioners_Guide_to_Triton.ipynb).

- **Tackling Memory Fragmentation for Triton Kernels**: After debugging, it was determined that large tensor sizes cause memory to become non-contiguous, breaking pointer arithmetic in the kernel; a utility function `check_tensors_gpu_ready` was recommended for ensuring data readiness.

- **Plotting A Course for Binary Search in Triton**: There is a noted gap in Triton's ability to perform binary search or indexing into a static codebook, a capability crucial for porting certain algorithmic examples and quantization work, as discussed in [Triton's GitHub Issue](https://github.com/openai/triton/issues/974#issuecomment-1345372027).

- **Navigating Triton's Indexing and Quantization Challenges**: The conversation featured an exchange of ideas on implementing binary search and addressing quantization kernels in Triton, considering the limitations and discussing possible workarounds using Triton's primitives like `tl.reduce` or `tl.scan`.

- **Deciphering `make_block_ptr` Parameter Puzzles**: A discussion on Triton's `tl.make_block_ptr` function's `order` parameter differentiates between row-major and column-major data formats, with `order=(1,0)` meaning row-major, where the inner axis is contiguous, and `order(0,1)` meaning column-major.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Jokeren/triton-samples/blob/main/binary_search.py">triton-samples/binary_search.py at main · Jokeren/triton-samples</a>: Contribute to Jokeren/triton-samples development by creating an account on GitHub.</li><li><a href="https://github.com/openai/triton/issues/974#issuecomment-1345372027">Index in triton · Issue #974 · openai/triton</a>: We&#39;d like to do some indexing in triton kernels, say we have x_ptr, idx_ptr, out_ptr x = tl.load(x_ptr + offsets, mask = mask) idx = tl.load(idx_ptr + offsets, mask = mask) we have: 1. idx = idx.t...</li><li><a href="https://triton-lang.org/main/python-api/generated/triton.language.make_block_ptr.html#triton.language.make_block_ptr">triton.language.make_block_ptr &mdash; Triton  documentation</a>: no description found</li><li><a href="https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py#L125">triton/python/tutorials/06-fused-attention.py at main · openai/triton</a>: Development repository for the Triton language and compiler - openai/triton</li><li><a href="https://github.com/thu-ml/low-bit-optimizers/blob/main/lpmm/cpp_extension/fused_adamw_kernel.cu#L27">low-bit-optimizers/lpmm/cpp_extension/fused_adamw_kernel.cu at main · thu-ml/low-bit-optimizers</a>: Low-bit optimizers for PyTorch. Contribute to thu-ml/low-bit-optimizers development by creating an account on GitHub.</li><li><a href="https://gist.github.com/alexandremuzio/3ba9d8669f57718139da36158180baaf">Weird triton kernel behavior for gray scale. (Meant to be copy pasted in a colab with a T4 gpu)</a>: Weird triton kernel behavior for gray scale. (Meant to be copy pasted in a colab with a T4 gpu) - weird_triton_repro.py</li><li><a href="https://github.com/cuda-mode/lectures/blob/main/lecture%2014/A_Practitioners_Guide_to_Triton.ipynb">lectures/lecture 14/A_Practitioners_Guide_to_Triton.ipynb at main · cuda-mode/lectures</a>: Material for cuda-mode lectures. Contribute to cuda-mode/lectures development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1231346697506918532)** (8 messages🔥): 

- **Gratitude for Conceptual Foundations**: A member expressed appreciation for a presentation that laid out the **conceptional foundation** for "layout algebra," suggesting it revealed the "real thing" in the subject.

- **Force Inline Queries**: [__forceinline and __inline](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-qualifiers) were discussed, with members explaining they instruct the compiler to embed the function's source code in the caller context to potentially make execution faster.

- **Nsight System CLI Troubleshooting**: A member resolved a profiling issue with Nsight Systems on Windows about conflicting core counts, noting that reverting to **version 2023.4.4** from 2024.2.1 fixed the problem.

- **Inquiry for Performance Measurement Script**: A request was made for a script to measure execution time across different thread and block configurations, but no solutions or links were provided in the messages provided.

- **Inlining and Code Optimization**: Discussion highlighted that using **__forceinline** can lead to more optimization opportunities for the compiler, similar to how memory coalescing increases performance by reducing the need for separate function calls.
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1232346270304174110)** (2 messages): 

- **Understanding GPU Utilization in Neural Network Operations**: A question was raised regarding whether operations like `torch.nn.conv2d`, `torch.nn.relu`, and `torch.nn.batchnorm` result in data being transferred between CPU and GPU between each operation. It was clarified that when a GPU tensor is passed through a sequence of functions, **all operations are executed on the GPU** without copying back to host memory for intermediate results.
- **Asynchronous Execution on GPU**: It was explained that operations on the GPU are scheduled **asynchronously**, meaning Python instructions return before the computation is complete. Blocking or synchronizing operations that require reading the value, such as `.cpu()`, will cause synchronization with the CPU.
  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1231317965308170340)** (1 messages): 

- **Lecture 15 on CUTLASS**: CUDA-MODE's **Lecture 15** is starting, focusing on **Cutlass**. A presentation by the designated speaker is about to commence.
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

andreaskoepf: https://x.com/AliHassaniJr/status/1766108184630943832
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1231145398245265478)** (27 messages🔥): 

- **CUDA Lectures Ongoing and Upcoming Schedules**: The CUDA MODE lecture 2 has begun in the *general channel*; interested members can join, and another session is scheduled for the NAM time zone on Sunday. Details and planning occur in a separate invite channel, with the link shared as [CUDA MODE Lecture Planning](https://discord.gg/H9h8vKNu).
- **Lecturer's Engaging Style Captures Audience**: Members were entertained by the lecturer's fun and engaging style, with one quoting that the author is "quite a funny entertaining chap."
- **Matrix Multiplication Explorations in CUDA**: A member asked for clarification on a matrix multiplication function, sparking a discussion and sharing of code examples, such as a Python Numba implementation for fast matrix multiplication.
- **Bringing Image and Video Processing to Life with CUDA**: A conversation about possible projects using CUDA included extending image processing examples to handle video processing and adding more functionalities.
- **Hardware Selection for ML Tasks Discussed**: There's an ongoing discussion on hardware choices for machine learning systems, comparing the merits of a 2x2070 dual GPU setup and a single 4090 GPU. One member advised that the 4090 is preferable for simplicity of setup though cost concerns were raised.

**Link mentioned**: <a href="https://discord.gg/H9h8vKNu">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1231262495864389793)** (2 messages): 

- **Collaborative Exercise Verification**: A member offers to verify exercise answers for those who have attempted the exercises; verification is conditional upon members first attempting the exercises and submitting a photo via DM. There are resources for different chapters, including [Ch 2](https://docs.google.com/document/d/10ez800eu8OF-OzJXNZ0tRGdJaRAwagiyFdgeBoX0S8o/edit), [Ch 3](https://docs.google.com/document/d/1wILXD7Pq8dsvEJpt-YwVekdFxYJvjqu6qnpzYR-LbhE/edit?usp=sharing), [Ch 4](https://docs.google.com/document/d/1b29UvSN2-S8D_UP1xvtSB7nFRc86s6AdWH7n5UieDfE/edit?usp=sharing), and the highlighted [Ch 5](https://docs.google.com/document/d/12_d0PFd3H5o68drT1pv_RuSYo67Evm9X7V70RMplrVk/edit?usp=sharing).

- **Cuda Kernel Loop Execution Query**: A member is seeking clarification on why an author suggests that a simple reduction CUDA kernel loop would execute 7 times with a 256-size input and a block size of 128, as their own calculations suggest the loop should execute 8 times. They have provided screenshots of the code and author's claims for reference.
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

.bexboy: I suppose that this one session will be uploaded too?
  

---


**CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1231185461091893269)** (1 messages): 

- **Memory Troubles with DenseFormer in JAX**: A member is facing challenges implementing a *denseformer* in JAX due to high memory usage. They referenced the DenseFormer's [GitHub repository](https://github.com/epfml/DenseFormer) and described its efficient **in-place tensor mutation** in PyTorch, while noting JAX/XLA's functional approach doesn't optimize away copies as well, leading to memory issues.

- **Exploration of Write-Once Buffers**: Using inspiration from the [Equinox library](https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/common.py), the member successfully created a *write-once buffer* for gradients with respect to the input but ran into quadratic memory growth when computing gradients with respect to denseformer block weights.

- **Considering Custom Gradients for Lean Memory Footprint**: To overcome the hurdles of **quadratic memory usage**, the user is considering a custom backward pass for the entire loop/scan function, a complex solution that seeks to replicate PyTorch's efficient in-place updating within JAX's functional paradigm. They are open to high-level suggestions on tackling this problem.

**Link mentioned**: <a href="https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/common.py">equinox/equinox/internal/_loop/common.py at main · patrick-kidger/equinox</a>: Elegant easy-to-use neural networks + scientific computing in JAX. https://docs.kidger.site/equinox/ - patrick-kidger/equinox

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1232158952041480202)** (3 messages): 

- **Ring Attention Model Training Inquiry**: In response to a question about implementing training with **Ring Attention**, another member shared a **GitHub link** to the Axolotl repository where code related to this is being developed. They mention having manual placement working and successful tests with tinyllama.
  - [View the Axolotl ring attention patch on GitHub](https://github.com/cuda-mode/axolotl/tree/ring_attention_patching)

**Link mentioned**: <a href="https://github.com/cuda-mode/axolotl/tree/ring_attention_patching">GitHub - cuda-mode/axolotl at ring_attention_patching</a>: Go ahead and axolotl questions. Contribute to cuda-mode/axolotl development by creating an account on GitHub.

  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1231258187043438642)** (4 messages): 

- **Regional Surprise in Münster**: Members of the CUDA MODE Discord expressed amusement upon discovering that three of them, including **@umerha**, live in close proximity in the Münster area, highlighting the small world of the GPU community.
- **Pleasant Meetup Experience**: **@umerha** and **@t-vi** shared their positive experience meeting in Münster, referring to the visit as "an honor and a pleasure."
- **Germany's GPU Capital Unites CUDA Enthusiasts**: **@umerha** mentioned a "pilgrimage" to Münster, humorously dubbing it Germany's GPU capital, while enjoying the company of fellow members **@761222713611386900** and **@719599526448463933**.
  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1231800666603782176)** (15 messages🔥): 

- **Promising Triton Kernel Benchmarks Announced**: A new fused triton `int4 / fp16` kernel was introduced, showing improved performance for various compute shapes, with **detailed benchmarking results** provided. The benchmark indicates that the kernel requires **[Triton >= 3.0.0](https://github.com/pytorch-labs/ao/pull/153)** and comparisons with reference `hqq.linear` and the `int4_mm` kernel from Torch are included.

- **Transposing for Better Backward Pass Efficiency**: A discussion focused on the need to transpose quantized weight matrices for the backward pass in **quantization** during trainings. The forward pass uses `torch.matmul(x, dequantize().t())` and the backward pass needs `torch.matmul(grad_output, dequantize())`, differences highlighted in the [HQQ GitHub repository](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L253-L283).

- **Quantization and Performance Considerations**: Members talked about the performance drop when using **dequantization**, noting that a typical CUDA dequantize kernel plus torch.matmul is around 15% slower than a pure torch.matmul with **fp16 or bfp16**.

- **Extension of Triton Kernel to Support `axis=0`**: A request was made to extend the new triton kernel's capabilities to handle computations along `axis=0` to improve **quantization quality**. Relevant Triton code was shared for reference [here](https://github.com/mobiusml/hqq/blob/triton/hqq/kernels/triton/dequant.py#L21-L50).

- **Triton Transpose Implementation Completed**: The triton kernel now includes an implementation for transposed weight matrices, as requested for more efficient backward passes. The updated test and implementation were posted in the [pull request](https://github.com/pytorch-labs/ao/pull/153/files#diff-240c1eaceacda5c5054dbaef20f835373e25882e314aa800868c32093faf8eca) on GitHub.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/triton/hqq/kernels/triton/dequant.py#L21-L50">hqq/hqq/kernels/triton/dequant.py at triton · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L253-L283">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/ao/pull/153/files#diff-240c1eaceacda5c5054dbaef20f835373e25882e314aa800868c32093faf8eca">Fused HQQ Quantization Gemm by jeromeku · Pull Request #153 · pytorch-labs/ao</a>: @msaroufim Fused int4 / fp16 Quant Matmul Fused kernel that combines asymmetric dequantization and gemm:  Dequantization: upcasts u4 / s4 weights to float16 / bfloat16, followed by groupwise scalin...</li><li><a href="https://github.com/pytorch-labs/ao/pull/153">Fused HQQ Quantization Gemm by jeromeku · Pull Request #153 · pytorch-labs/ao</a>: @msaroufim Fused int4 / fp16 Quant Matmul Fused kernel that combines asymmetric dequantization and gemm:  Dequantization: upcasts u4 / s4 weights to float16 / bfloat16, followed by groupwise scalin...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1231157413395169300)** (600 messages🔥🔥🔥): 

- **Atomics in CUDA and Performance Bottlenecks**: Discussions focused on the removal of atomic operations from CUDA kernels, as part of performance optimization efforts. Despite concerns about how to parallelize updates when indices can vary broadly, suggestions included using scratch memory and multiple kernel calls, or pre-processing on CPU to sort indices. The contention caused by atomics and dealing with majority-repeating indices were also discussed.

- **BF16/FP16 Mixed Precision Implementation**: A significant conversation around the implementation of BF16/FP16 mixed precision training revealed an approximate **1.86x performance gain**. While efforts to optimize for lower precisions like FP8 were briefly mentioned, the PR (#218) introduces complexity with stochastic rounding and managing optimizer state that requires BF16/FP16. The latest implementation for layernorm maintains FP32 due to slow performance with BF16 atomic operations.

- **CUDA Version Requirements in FP16 Conversion**: Compiling errors occurred due to an older CUDA version on one of the devices, highlighting a dependency on newer CUDA versions for BF16 support. The problem with cuBLAS not accepting FP8 biases for FP8 matmuls, requiring BF16 biases instead, was also noted.

- **Kernel Optimization and Profiling**: Some community members shared insights and progress on optimizing CUDA kernels using techniques like dtype sizing and float4 vectors, potentially leading to a 2x speedup in GELU and AdamW kernels. A suggestion to update the kernel development scripts to reflect real-world sizing for better profiling accuracy was proposed.

- **Optimizing Memory-Throttled Kernels with Thread Coarsening**: During a community collaboration session, thread coarsening was applied to the AdamW kernel to improve its performance due to the kernel being memory bound. This optimization batched memory requests to be more parallelized, aiming for future enhancements, especially post-transition to FP16.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.twitch.tv/zhizhinpeter">zhizhinpeter - Twitch</a>: Coding Multi-GPU for llm.c</li><li><a href="https://arxiv.org/abs/2110.02861">8-bit Optimizers via Block-wise Quantization</a>: Stateful optimizers maintain gradient statistics over time, e.g., the exponentially smoothed sum (SGD with momentum) or squared sum (Adam) of past gradient values. This state can be used to accelerate...</li><li><a href="https://www.youtube.com/">YouTube</a>: no description found</li><li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-1-single-process-single-thread-multiple-devices">Examples &mdash; NCCL 2.21.5 documentation</a>: no description found</li><li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-1-single-process-sin">Examples &mdash; NCCL 2.21.5 documentation</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/blob/master/dev/cuda/encoder_backward.cu">llm.c/dev/cuda/encoder_backward.cu at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/nshepperd/flash_attn_jax/tree/main/csrc/flash_attn/src">flash_attn_jax/csrc/flash_attn/src at main · nshepperd/flash_attn_jax</a>: JAX bindings for Flash Attention v2. Contribute to nshepperd/flash_attn_jax development by creating an account on GitHub.</li><li><a href="https://github.com/KernelTuner/kernel_float">GitHub - KernelTuner/kernel_float: CUDA header-only library for working with vector types (half2, float4, double2) and reduced precision math (half, e5m2)  inside kernel code</a>: CUDA header-only library for working with vector types (half2, float4, double2) and reduced precision math (half, e5m2)  inside kernel code - KernelTuner/kernel_float</li><li><a href="https://clang.llvm.org/doxygen/____clang__cuda__intrinsics_8h_source.html">clang: lib/Headers/__clang_cuda_intrinsics.h Source File</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/218">Support for FP16/BF16 in train_gpt2.cu (1.86x Perf) by ademeure · Pull Request #218 · karpathy/llm.c</a>: Now finished and reasonably happy with it! 1.86x performance on my RTX 4090:  FP32: ~80ms BF16: ~43ms (with layernorm params in FP32, but all activations in BF16)  This allows the same train_gpt2.c...</li><li><a href="https://www.youtube.com/watch?v=0zE6c52yomU">How to go from 0 to speeding up LLM.c - CUDA Kernel Profiling setup</a>: Commands to run after getting the instance setup:git clone https://github.com/karpathy/llm.c.gitexport PATH=/usr/local/cuda/bin:$PATHsource ~/.bashrcsudo apt...</li><li><a href="https://github.com/karpathy/llm.c/pull/210">Added shared memory for the atomic additions for the layernorm_back by ChrisDryden · Pull Request #210 · karpathy/llm.c</a>: This cr was made to address the issue found in the profiler that the atomic operations in the final loop of this kernel were causing a bunch of warp stalls. By doing the atomic operation on shared ...</li><li><a href="https://github.com/karpathy/llm.c/issues/212">bug: something goes wrong at larger batch sizes · Issue #212 · karpathy/llm.c</a>: There&#39;s some bug I have difficulty tracking down today and I&#39;m going to give up for tonight and try again tomorrow. Reproduction: ./train_gpt2cu -b 12 launches the job with batch size 12. On m...</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h">flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h at main · Dao-AILab/flash-attention</a>: Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/221">Faster `matmul_backward_bias` using coalesced reads and shared memory in the kernel by al0vya · Pull Request #221 · karpathy/llm.c</a>: This kernel seems to offer a &lt;4x runtime improvement over matmul_backward_bias_kernel2 on an RTX 2070 Super GPU, runtime comparison shown below: matmul_backward_bias_kernel2: block_size 32 time 0.9...</li><li><a href="https://github.com/karpathy/llm.c/pull/215">cuDNN Forward Attention + FP16 non-cuDNN version in /dev/cuda/ by ademeure · Pull Request #215 · karpathy/llm.c</a>: Previous Kernel 4: 1.74ms Kernel 4 with TF32: 1.70ms Kernel 5 (4 with BF16 I/O): 0.91ms Kernel 6 (5 without permute, not realistic): 0.76ms Kernel 10 (cuDNN BF16, with FP32 conversion): 0.33ms Kern...</li><li><a href="https://github.com/karpathy/llm.c/commit/8488669d256c59594f486d52a8b3597da7cbfeab">speed up the backward bias kernel by 45% and speed up the full runnin… · karpathy/llm.c@8488669</a>: …g time by 1%
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[massively-parallel-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1231255000894799954)** (29 messages🔥): 

- **Introducing the Moderator Role**: A new role called "Moderator" has been introduced to manage users and content, with permissions that include timing out, kicking, banning users, and deleting inappropriate messages. Moderators can also create and edit events, and manage the stage to maintain a friendly environment for GPU and massively parallel programming discussions.

- **Technical Difficulties in Recording Panel Discussions**: Members discussed technical issues experienced during the recording of a panel discussion. The conversation included coordination to meet before future talks to ensure recording setups are functioning well, and the possibility of re-recording talks if necessary.

- **Backup Recordings Save the Day**: One member reported an abrupt end to their recording session, but it was covered by another member's backup. They confirmed that combined materials from two recordings should suffice for a complete session.

- **Scheduling Future Talks and Dry Runs**: As several events were upcoming, members coordinated about being prepared 15 minutes before the scheduled time to ensure technical setups were in place. One member noted their unavailability for recording on one of the days, but offered to handle session recording and post-production the following day.

- **Open Invitation for FlashAttention Code Deep-Dive**: After a tweet was shared about FlashAttention, the idea of a specialized deep-dive event was proposed, although no immediate plans were made. Additionally, members suggested reaching out to Tri Dao for a potential talk regarding his work on Flash decoding, with an acknowledgment that he has previously presented on related topics.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=IoMSGuiwV3g).">Flash Attention 2.0 with Tri Dao (author)! | Discord server talks</a>: ❤️ Become The AI Epiphany Patreon ❤️https://www.patreon.com/theaiepiphany👨‍👩‍👧‍👦 Join our Discord community 👨‍👩‍👧‍👦https://discord.gg/peBrCpheKEHey g...

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1231169236890812427)** (262 messages🔥🔥): 

- **LLM Local App Speculations**: Users discussed the feasibility of running LLMs locally on smartphones, focusing on the Eleuther community potentially developing an easy-to-use app. Memory bandwidth and GPU capabilities of different smartphone models, like the Samsung S24 Ultra and Snapdragon, were referenced, suggesting even 7-8B models might be potentially usable.

- **Technological Diving into Smartphone Capabilities**: Conversations delved into the hardware specifications of modern smartphones, such as the Samsung Exynos 2400 chipset, to estimate the performance of LLMs running locally. Specs like the 6.4 Gbps pin speed and 51.2 GB/s memory bandwidth were scrutinized, and speculative decoding was suggested as a possible method to improve token generation rates.

- **Examining Existing Apps for Local LLM Use**: Users explored existing solutions like [MLC-LLM](https://github.com/mlc-ai/mlc-llm) for deploying AI models natively on devices. They also discussed other apps found on the App Store and Play Store, such as "MLC Chat" and "Private AI", which utilize offline LLMs, indicating there are some current applications attempting this endeavor.

- **Hugging Face Downtime and Business Model Debate**: Extended downtime on Hugging Face triggered a debate regarding its business model. Users pondered over its strategies, comparing it to platforms like GitHub, and questioned the sustainability of providing free hosting for large AI models.

- **Discussions on Reasoning in LLMs Beyond CoT**: The conversation turned to evaluating reasoning in LLMs with various methods such as Chain-of-Thought (CoT). A recent research paper integrating Monte Carlo Tree Search with LLMs was suggested as an alternative to CoT reasoning ([AlphaLLM](http://arxiv.org/abs/2404.12253)).

- **Cost Analysis of LLM Training**: Discussions touched on the costs associated with training large models like Llama 2, considering factors such as GPU hours and token quantities. It also highlighted the potential underestimation of costs without thorough mathematical calculation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://store.google.com/intl/en/ideas/articles/pixel-feature-drop-december-2023/">Gemini Nano now running on Pixel 8 Pro — the first smartphone with AI built in</a>: Gemini is here, the most capable and flexible AI model we've ever built. Plus more AI updates coming to the Pixel portfolio.</li><li><a href="https://play.google.com/store/apps/details?id=us.valkon.privateai&hl=en&gl=US">Private AI - Apps on Google Play</a>: no description found</li><li><a href="https://llm.mlc.ai/docs/deploy/android.html">Android App &mdash; mlc-llm 0.1.0 documentation</a>: no description found</li><li><a href="https://nanoreview.net/en/soc/samsung-exynos-2400">Samsung Exynos 2400: specs and benchmarks</a>: Samsung Exynos 2400: performance tests in benchmarks (AnTuTu 10, GeekBench 6). Battery life and full specifications.</li><li><a href="https://apps.apple.com/us/app/mlc-chat/id6448482937?platform=iphone">‎MLC Chat</a>: ‎MLC Chat lets users chat with open language models locally on ipads and iphones. After a model is downloaded to the app, everything runs locally without server support, and it works without internet ...</li><li><a href="https://news.ycombinator.com/item?id=37248895">no title found</a>: no description found</li><li><a href="https://github.com/mlc-ai/mlc-llm">GitHub - mlc-ai/mlc-llm: Enable everyone to develop, optimize and deploy AI models natively on everyone&#39;s devices.</a>: Enable everyone to develop, optimize and deploy AI models natively on everyone&#39;s devices. - mlc-ai/mlc-llm</li><li><a href="https://arxiv.org/abs/2311.10207">Stella Nera: Achieving 161 TOp/s/W with Multiplier-free DNN Acceleration based on Approximate Matrix Multiplication</a>: From classical HPC to deep learning, MatMul is at the heart of today&#39;s computing. The recent Maddness method approximates MatMul without the need for multiplication by using a hash-based version o...</li><li><a href="https://github.com/Kotlin/Kotlindl">GitHub - Kotlin/kotlindl: High-level Deep Learning Framework written in Kotlin and inspired by Keras</a>: High-level Deep Learning Framework written in Kotlin and inspired by Keras - Kotlin/kotlindl</li><li><a href="http://arxiv.org/abs/2404.12253">Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing</a>: Despite the impressive capabilities of Large Language Models (LLMs) on various tasks, they still struggle with scenarios that involves complex reasoning and planning. Recent work proposed advanced pro...</li><li><a href="https://developers.googleblog.com/2024/03/running-large-language-models-on-device-with-mediapipe-andtensorflow-lite.html">Large Language Models On-Device with MediaPipe and TensorFlow Lite - Google for Developers</a>: no description found</li><li><a href="https://www.gsmarena.com/samsung_galaxy_s24_ultra-review-2670p4.php">Samsung Galaxy S24 Ultra review</a>: Samsung&#039;s S24 family is launching with Samsung&#039;s latest One UI 6.1 on top of Google&#039;s latest Android 14. Despite the fairly small &quot;.1&quot; numbering update,...</li><li><a href="https://support.google.com/googleplay/android-developer/answer/9878810?hl=en-GB#>">Inappropriate Content - Play Console Help</a>: no description found</li><li><a href="https://github.com/atfortes/Awesome-LLM-Reasoning?tab=readme-ov-file">GitHub - atfortes/Awesome-LLM-Reasoning: Reasoning in Large Language Models: Papers and Resources, including Chain-of-Thought, Instruction-Tuning and Multimodality.</a>: Reasoning in Large Language Models: Papers and Resources, including Chain-of-Thought, Instruction-Tuning and Multimodality.  - GitHub - atfortes/Awesome-LLM-Reasoning: Reasoning in Large Language M...</li><li><a href="https://semiconductor.samsung.com/dram/lpddr/lpddr5/">LPDDR5 | DRAM | Samsung Semiconductor Global</a>: Meet LPDDR5 powering next-generation applications with performance and efficiency by 6,400 Mbps of pin speed, massive transfer at 51.2Gb/s, and 20% power saving.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1231165794822520944)** (443 messages🔥🔥🔥): 

- **Diffusion Model Inference Steps Discussion**: Diffusion models trained on higher steps, like 300 or 1000, can be effectively used for inference with significantly fewer steps, such as 10-30 steps. There's a consensus that the number of training steps doesn't greatly affect the quality at a given inference step count.
  
- **Token-Free Language Models**: The [SpaceByte paper](https://arxiv.org/abs/2404.14408v1) proposes a novel byte-level architecture trying to close the gap between subword and byte-level autoregressive language modeling. It was noted that tokenizers can potentially leak information about subsequent tokens, which could be seen as a significant nuisance, especially for applications such as autocompletes.

- **Concerns About 'Fineweb' Dataset's Relation to LLaMA**: While [Fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) offers 15 trillion tokens of CommonCrawl data and claims high performance, members questioned its relationship to LLaMA's dataset and expressed skepticism about the lack of dataset decontamination. The effects of Fineweb's performance will be closely monitored over time.

- **AI-Designed CRISPR-Cas Protein**:
  A Large Language Model, ProGen2, was successfully used to design new CRISPR-Cas protein sequences that were then tested in a lab, yielding variants with improved specificity. This breakthrough example by [Profluent Bio](https://www.profluent.bio/) indicates the potential of LLMs in accelerating scientific discovery.

- **Prompt Priority for Safe Large Language Models**:
  A new paper suggests addressing safety vulnerabilities in LLMs by training models to prioritize instructions based on a defined hierarchy. This approach aims to increase robustness against prompt injections and other attacks without the need for additional preference labels or demonstrations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/BlancheMinerva/status/1782437494585282965">Tweet from Stella Biderman (@BlancheMinerva)</a>: Create a benchmark for RAG models where all of the questions require information from multiple documents to be synthesized answer them. Study how models trained on publicly released data do on it and ...</li><li><a href="http://arxiv.org/abs/2404.13208">The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions</a>: Today&#39;s LLMs are susceptible to prompt injections, jailbreaks, and other attacks that allow adversaries to overwrite a model&#39;s original instructions with their own malicious prompts. In this w...</li><li><a href="https://arxiv.org/abs/2404.14313">Self-Supervised Alignment with Mutual Information: Learning to Follow Principles without Preference Labels</a>: When prompting a language model (LM), users frequently expect the model to adhere to a set of behavioral principles across diverse tasks, such as producing insightful content while avoiding harmful or...</li><li><a href="https://arxiv.org/abs/2404.14408v1">SpaceByte: Towards Deleting Tokenization from Large Language Modeling</a>: Tokenization is widely used in large language models because it significantly improves performance. However, tokenization imposes several disadvantages, such as performance biases, increased adversari...</li><li><a href="https://arxiv.org/abs/2401.13660">MambaByte: Token-free Selective State Space Model</a>: Token-free language models learn directly from raw bytes and remove the inductive bias of subword tokenization. Operating on bytes, however, results in significantly longer sequences. In this setting,...</li><li><a href="http://arxiv.org/abs/2404.13686">Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis</a>: Recently, a series of diffusion-aware distillation algorithms have emerged to alleviate the computational overhead associated with the multi-step inference process of Diffusion Models (DMs). Current d...</li><li><a href="http://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: Transformer-based language models spread FLOPs uniformly across input sequences. In this work we demonstrate that transformers can instead learn to dynamically allocate FLOPs (or compute) to specific ...</li><li><a href="https://huggingface.co/stabilityai/stablelm-3b-4e1t">stabilityai/stablelm-3b-4e1t · Hugging Face</a>: no description found</li><li><a href="http://arxiv.org/abs/2401.06104">Transformers are Multi-State RNNs</a>: Transformers are considered conceptually different compared to the previous generation of state-of-the-art NLP models - recurrent neural networks (RNNs). In this work, we demonstrate that decoder-only...</li><li><a href="https://www.profluent.bio/">Profluent</a>: We are fluent in the language of protein design.</li><li><a href="https://arxiv.org/abs/2402.06925">A Thorough Examination of Decoding Methods in the Era of LLMs</a>: Decoding methods play an indispensable role in converting language models from next-token predictors into practical task solvers. Prior research on decoding methods, primarily focusing on task-specifi...</li><li><a href="https://arxiv.org/abs/2401.03462">Soaring from 4K to 400K: Extending LLM&#39;s Context with Activation Beacon</a>: The utilization of long contexts poses a big challenge for LLMs due to their limited context window size. Although the context window can be extended through fine-tuning, it will result in a considera...</li><li><a href="https://arxiv.org/abs/2403.11901">Larimar: Large Language Models with Episodic Memory Control</a>: Efficient and accurate updating of knowledge stored in Large Language Models (LLMs) is one of the most pressing research challenges today. This paper presents Larimar - a novel, brain-inspired archite...</li><li><a href="https://github.com/microsoft/LLMLingua">GitHub - microsoft/LLMLingua: To speed up LLMs&#39; inference and enhance LLM&#39;s perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression with minimal performance loss.</a>: To speed up LLMs&amp;#39; inference and enhance LLM&amp;#39;s perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression with minimal performance loss.  - GitH...</li><li><a href="https://github.com/krafton-ai/mambaformer-icl">GitHub - krafton-ai/mambaformer-icl: MambaFormer in-context learning experiments and implementation for https://arxiv.org/abs/2402.04248</a>: MambaFormer in-context learning experiments and implementation for https://arxiv.org/abs/2402.04248 - krafton-ai/mambaformer-icl</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.04.22.590591v1">Design of highly functional genome editors by modeling the universe of CRISPR-Cas sequences</a>: Gene editing has the potential to solve fundamental challenges in agriculture, biotechnology, and human health. CRISPR-based gene editors derived from microbes, while powerful, often show significant ...</li><li><a href="https://arxiv.org/abs/2404.08698">Lossless Acceleration of Large Language Model via Adaptive N-gram Parallel Decoding</a>: While Large Language Models (LLMs) have shown remarkable abilities, they are hindered by significant resource consumption and considerable latency due to autoregressive processing. In this study, we i...</li><li><a href="https://arxiv.org/abs/2212.04089">Editing Models with Task Arithmetic</a>: Changing how pre-trained models behave -- e.g., improving their performance on a downstream task or mitigating biases learned during pre-training -- is a common practice when developing machine learni...</li><li><a href="https://arxiv.org/abs/2312.02783">Large Language Models on Graphs: A Comprehensive Survey</a>: Large language models (LLMs), such as GPT4 and LLaMA, are creating significant advancements in natural language processing, due to their strong text encoding/decoding ability and newly found emergent ...</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb">HuggingFaceFW/fineweb · Datasets at Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/sisihae-gif-23689236">Sisihae GIF - Sisihae - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2404.07647">Why do small language models underperform? Studying Language Model Saturation via the Softmax Bottleneck</a>: Recent advances in language modeling consist in pretraining highly parameterized neural networks on extremely large web-mined text corpora. Training and inference with such models can be costly in pra...</li><li><a href="https://arxiv.org/abs/2310.11829">Towards Graph Foundation Models: A Survey and Beyond</a>: Foundation models have emerged as critical components in a variety of artificial intelligence applications, and showcase significant success in natural language processing and several other domains. M...</li><li><a href="https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/#eleuther-discord">[AINews] FineWeb: 15T Tokens, 12 years of CommonCrawl (deduped and filtered, you&#x27;re welcome)</a>: AI News for 4/19/2024-4/22/2024. We checked 6 subreddits and 364 Twitters and 27 Discords (395 channels, and 14973 messages) for you. Estimated reading time...</li><li><a href="https://arxiv.org/html/2402.08164v1">On Limitations of the Transformer Architecture</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1231665310709321739)** (35 messages🔥): 

- **Twitter Confrontation Over Rounding Data**: A member expressed frustration over being blocked on Twitter after criticizing someone for rounding numbers in their publication, sharing a [tweet](https://twitter.com/tamaybes/status/1782102492479652314) as evidence. The conversation evolved around the tone and approach used, with others pointing out that the member's direct tone might come off as rude or confrontational.
  
- **Tone Matters in Critical Conversations**: Other members joined the discussion, suggesting that the original poster's tone might have been perceived as aggressive or trolling, which could lead to defensive reactions. They emphasized the importance of a friendly and constructive tone when engaging in debates, especially when trying to convey criticism.

- **Misunderstandings in Communication Identified**: It was suggested that confusion arose because the member incorrectly attributed the rounding of data to the replication team, while in fact, the original Chinchilla paper authors reported rounded results. Clarifications were made about the capabilities of TeX in handling significant figures and rendering vector formats like SVG.

- **Critique of Chinchilla Paper and Replication Methodology**: The member clarified his original critique, noting that the real issue was not the rounding itself but the replication authors not noticing the residuals were not centered around zero, which could indicate a mistake in their replication process. This detailed feedback was part of a larger discussion critiquing the methodologies used in the Chinchilla paper reproduction.

- **Constructive Dissection of Social Media Interaction**: Participants dissected the nuances of online communication and jokingly crafted a template for friendly internet discourse, highlighting the balance needed between being direct and including "neurotypical decoration" in posts to avoid being misunderstood.

**Link mentioned**: <a href="https://x.com/kyo_takano/status/1782100341443666282))">Tweet from Kyo (@kyo_takano)</a>: You ARE rounding the original estimate lol  Try inspecting the TeX source like you did PDF figures. To be more specific, you rounded:  - E from exp(0.5267228) to 1.69 - A from exp(6.0073404) to 406.4 ...

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1232409179575029770)** (2 messages): 

- **Exponential Growth in Residual Stream Norms Uncovered**: A shared [post from LessWrong](https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward) reveals that the norm of each residual stream in language models like GPT2-XL grows exponentially during the forward pass. The summarized paper suggests LayerNorm makes it *difficult to cancel out existing features*, thereby allowing *new features to overshadow by increasing 4.5% per layer*.

**Link mentioned**: <a href="https://www.lesswrong.com/posts/8mizBCm3dyc432nK8/residual-stream-norms-grow-exponentially-over-the-forward">Residual stream norms grow exponentially over the forward pass — LessWrong</a>: Summary: For a range of language models and a range of input prompts, the norm of each residual stream grows exponentially over the forward pass, wit…

  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1232208166264111136)** (8 messages🔥): 

- **Seeking Forked Sanity**: A member humorously noted that research groups prefer running private forks of the **lm evaluation harness** instead of engaging in direct model comparisons.
- **Token Inquiry at Evaluation**: A question was raised regarding whether the **eval-harness** automatically adds a beginning-of-sequence token.
- **Experimenting with MMLU Task Implementation**: A member proposed adding an **MMLU task implementation** using the arc prompt format, aimed at investigating the impact of MMLU prompt format on model scores.
- **Call for Genericization in Task Implementation**: In response to the proposal, another member suggested to ideally create a **generic implementation** capable of supporting various styles like "arc style" and "MMLU style" for all **MCQA tasks**, though expressing interest in the current specific implementation until a more general one is developed.
- **Parallel Metrics Exploration**: A query was posted about executing metrics from the **lm-evaluation-harness** in parallel, with a request for further elaboration on the specific needs.
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1232349058723614861)** (14 messages🔥): 

- **Discussing RWKV integration with GPT-NeoX**: Developers are currently focused on integrating **RWKV (Rethinking Weighted Key-Value Memory Networks)** into GPT-NeoX. The integration work can be tracked through [GitHub Issue #1167](https://github.com/EleutherAI/gpt-neox/issues/1167) and involves disabling bf16, PP, TP, MoE, and adding fp16 support and JIT kernel compilation among other tasks.

- **FP16 Support Being Integrated**: A new branch containing integration for fp16 and fp32 support for RWKV within GPT-NeoX has been pushed by a developer, [available here](https://github.com/SmerkyG/gpt-neox/tree/rwkv). The integration is simple and pending testing with the NeoX trainer.

- **Kernel Enhancement and Code Transfer**: A developer has newly optimized kernel code ready for RWKV, which could potentially allow for full state-gradients for future BPTT use. This new method and code are available on the developer's GitHub fork, specifically the branch [rwkv-6-support](https://github.com/RWKV/RWKV-infctx-trainer/tree/rwkv-6-support).

- **RWKV Version Numbering Suggested**: Due to the iterative nature of the RWKV integration work, it's been suggested to implement version numbering to identify different iterations, such as "rwkv 6.0". The best approach for this naming convention—be it file, class, or directory specific—is still under consideration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/RWKV/RWKV-infctx-trainer/tree/rwkv-6-support">GitHub - RWKV/RWKV-infctx-trainer at rwkv-6-support</a>: RWKV infctx trainer, for training arbitary context sizes, to 10k and beyond! - GitHub - RWKV/RWKV-infctx-trainer at rwkv-6-support</li><li><a href="https://github.com/RWKV/RWKV-infctx-trainer/compare/main...rwkv-6-support">Comparing main...rwkv-6-support · RWKV/RWKV-infctx-trainer</a>: RWKV infctx trainer, for training arbitary context sizes, to 10k and beyond! - Comparing main...rwkv-6-support · RWKV/RWKV-infctx-trainer</li><li><a href="https://github.com/EleutherAI/gpt-neox/issues/1167">Add Basic RWKV Block to GPT-NeoX · Issue #1167 · EleutherAI/gpt-neox</a>: We want to add RWKV to gpt-neox: Add basic RWKV block, without kernels, from https://github.com/BlinkDL/RWKV-LM to https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model Add rwkv kernels A...</li><li><a href="https://github.com/">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://github.com/SmerkyG/gpt-neox/tree/rwkv">GitHub - SmerkyG/gpt-neox at rwkv</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library. - GitHub - SmerkyG/gpt-neox at rwkv
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1231156435409174578)** (473 messages🔥🔥🔥): 

<ul>
  <li><strong>Hugging Face Downtime Concerns</strong>: Several users reported experiencing 504 Gateway Time-outs and service disruptions while trying to access or use Hugging Face, indicating potential downtime or server issues.</li>
  <li><strong>Meta-Llama 3 Integration Questions</strong>: Users discussed the integration of Meta Llama 3 with serverless inference API and whether features like system prompts are supported when making requests.</li>
  <li><strong>Autotrain Inquiry</strong>: There was a query about whether AutoTrain supports custom models like phi-3 for fine-tuning, which was addressed by pointing to Hugging Face documentation and previous successful usage.</li>
  <li><strong>Model Upload Hurdles</strong>: A user sought help for uploading GGUF files to Hugging Face due to a size limit, which prompted advice on using sharding or splitting files to accommodate service constraints.</li>
  <li><strong>Exploring OCR Options</strong>: Discussion centered on finding an effective OCR solution for reading float numbers, with options like PaddleOCR and kerasOCR being mentioned as potentially better alternatives to tesseract and EasyOCR.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1nU4xHpLQ5PIQKY0T1MK-sJ3kVXiBYnkQ?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://x.com/abhi1thakur/status/1782807785807159488?s=46">Tweet from abhishek (@abhi1thakur)</a>: Phi-3 is here!!!! 🚀 and ofcourse, you can already fine-tune it using AutoTrain 🚀🚀🚀</li><li><a href="https://arxiv.org/abs/2309.08632">Pretraining on the Test Set Is All You Need</a>: Inspired by recent work demonstrating the promise of smaller Transformer-based language models pretrained on carefully curated data, we supercharge such approaches by investing heavily in curating a n...</li><li><a href="https://huggingface.co/chat/assistant/66238e78096b24c9dad9457c">Llama 3-70B - HuggingChat</a>: Use the Llama 3-70B assistant inside of HuggingChat</li><li><a href="https://tenor.com/view/resident-evil-resident-evil-welcome-to-raccoon-city-resident-evil-movie-burning-on-fire-gif-25613395">Resident Evil Resident Evil Welcome To Raccoon City GIF - Resident Evil Resident Evil Welcome To Raccoon City Resident Evil Movie - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/turn-down-for-what-snoop-dogg-cheers-dancing-drinking-gif-10966591">Turn Down For What Snoop Dogg GIF - Turn Down For What Snoop Dogg Cheers - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/jinx-the-cat-jinx-jinx-cat-cat-computer-gif-25786466">Jinx The Cat Jinx GIF - Jinx The Cat Jinx Jinx Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/im-dead-dead-bruh-skeleton-dead-bruh-skeleton-dead-im-dead-bruh-gif-26854866">Im Dead Dead Bruh GIF - Im Dead Dead Bruh Skeleton Dead Bruh - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://hf-mirror.com/">HF-Mirror - Huggingface 镜像站</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct">meta-llama/Meta-Llama-3-70B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/eyeverse-brace-initiation-eyebrow-shave-gif-6015143619791964168">Eyeverse Brace GIF - Eyeverse Brace Initiation - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/docs/huggingface_hub/en/guides/upload#tips-and-tricks-for-large-uploads">Upload files to the Hub</a>: no description found</li><li><a href="https://tenor.com/view/dinela-gif-26054323">Dinela GIF - Dinela - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://colab.research.google.com/drive/1u9r-p_x7QXH9zAbQ5c0O2smEBHvC44me?usp=sharing>">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GPTQ">TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GPTQ · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/cat-club-cat-cat-dance-cat-party-cat-disco-gif-27258615">Cat Club Cat GIF - Cat Club Cat Cat Dance - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/docs/huggingface_hub/en/guides/upload#preupload-lfs-files-before-commit">Upload files to the Hub</a>: no description found</li><li><a href="https://hf.co/chat/assistant/6626057fa0b4434b65ed78b5">Albert Einstein - HuggingChat</a>: Use the Albert Einstein assistant inside of HuggingChat</li><li><a href="https://rapidapi.com/swift-api-swift-api-default/api/meta-llama-3-8b">Meta Llama 3 | 8B API Documentation (swift-api-swift-api-default) | RapidAPI</a>: no description found</li><li><a href="https://bpa.st/3MUQ">View paste 3MUQ</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1884c8k/to">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtube.com/watch?v=SfKGHKzkm-o">The Rise of AI</a>: (Hidupkan Closed Caption)(Turn on the Closed Caption)Bergabunglah bersama kami dalam perjalanan melalui evolusi cepat Artificial Intelligence, mulai dari kem...</li><li><a href="https://www.youtube.com/watch?v=JOeY07qKU9c>">&quot;It&#39;s A UNIX System!&quot; | Jurassic Park | Science Fiction Station</a>: Hackerman Lexi (Ariana Richards) shows off her nerd skills as she tries to fix Jurassic Park&#39;s UNIX control system.Jurassic Park (1993): John Hammond, an ent...</li><li><a href="https://vvd.im/TicketTool">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://status.huggingface.co/">
Hugging Face status
</a>: no description found</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1884c8k/todays_ai_breakthrough_zero_step_diffusion/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1231786803434881046)** (13 messages🔥): 

- **Studying AI's Speed, Cost, and Quality**: A video titled "ORPO with LLaMa 3- Fast, Cheap, and Good!" discusses innovations in AI that challenge the old saying "Fast, Cheap, Good- Pick two." The video can be found on [YouTube](https://www.youtube.com/watch?v=oHM3faIPTg0).
- **First Reinforcement Learning Model Success**: A member learned how to create their first reinforcement learning model and shared a [Hugging Face model card](https://huggingface.co/wsqstar/ppo-LunarLander-v2) for a **PPO** agent trained to play **LunarLander-v2**.
- **Exploring Tokenization**: One member is focusing on learning about tokenizers today.
- **Dependency on Hugging Face**: A member remarked on their continued reliance on Hugging Face's resources even with local model installations.
- **Creating RAG Systems with AI Agents**: Members are learning to construct **RAG systems** utilizing the Llamaindex and are also exploring implementation with offline, open-source models using libraries like **transformer.js**.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/wsqstar/ppo-LunarLander-v2">wsqstar/ppo-LunarLander-v2 · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=oHM3faIPTg0">ORPO with LLaMA 3- Fast, Cheap, and Good!</a>: The old saying goes &quot;Fast, Cheap, Good- Pick two&quot;. AI has been no different, but we&#39;re starting to see some great innovations to change that. Great article f...</li><li><a href="https://www.youtube.com/watch?v=oPCKB9MUP6c&t=420s&ab_channel=DeployingAI">Build an Agent with Long-Term, Personalized Memory</a>: This video explores how to store conversational memory similar to ChatGPT&#39;s new long-term memory feature.We&#39;ll use LangGraph to build a simple memory-managin...</li><li><a href="https://youtu.be/q3nBKwNkRno?si=EkxSV5ZXtrSB7F6A">(RVC) I Can&#39;t Dance (AI Cover Mashup) (READ DESC)</a>: #aicover #icantdance #genesis Disclaimer: This is a simple and fun AI mashup video I made during my spare time utilizing my and other people&#39;s AI voice model...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1231254143478534255)** (21 messages🔥): 

- **Exploring Quantum Computing**: A video titled [*"New quantum computers - Potential and pitfalls | DW Documentary"*](https://youtu.be/0HFzTYlhT2E) was shared, discussing the capabilities of new supercomputers in potentially reducing animal experiments and curing cancer.
- **Neural Networks Demystified**: A member shared a [YouTube video](https://www.youtube.com/watch?v=0QczhVg5HaI) titled **"Why Neural Networks can learn (almost) anything"**, which explains the functioning and usefulness of neural networks.
- **Voice-Prompted AI Image Generation**: An intriguing [Twitter post](https://twitter.com/Dan50412374/status/1781790992318042428) demonstrates live streaming of high-resolution images generated by AI in response to spoken (whisper) voice commands.
- **Comprehensive Offline RL Framework Revealed**: The message highlighted [Hokoff](https://sites.google.com/view/hok-offline), a resource providing pre-collected datasets and a framework for Offline Reinforcement Learning and Multi-Agent Reinforcement Learning research.
- **Interactive JavaScript for 🤗 Transformers**: A tool was introduced that allows running HuggingFace Transformers directly in the browser; explore it at [transformers.js](https://xenova.github.io/transformers.js/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://reasoning-tokens.ghost.io/reasoning-tokens/">Self-Reasoning Tokens, teaching models to think ahead.</a>: What is the mathematical formulation of reasoning? How can we make LLMs like chatGPT think before they speak? And how can we make that baked into the model so it can learn to think in a self-supervise...</li><li><a href="https://arxiv.org/abs/2404.13026">PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation</a>: Realistic object interactions are crucial for creating immersive virtual experiences, yet synthesizing realistic 3D object dynamics in response to novel interactions remains a significant challenge. U...</li><li><a href="https://xenova.github.io/transformers.js/">Transformers.js</a>: no description found</li><li><a href="https://karpathy.ai/zero-to-hero.html">Neural Networks: Zero To Hero</a>: no description found</li><li><a href="https://sites.google.com/view/hok-offline">Hokoff</a>: Abstract </li><li><a href="https://huggingface.co/ByteDance">ByteDance (ByteDance)</a>: no description found</li><li><a href="https://youtu.be/0HFzTYlhT2E?si=lgzMqlFFbhVgjM7f">New quantum computers - Potential and pitfalls | DW Documentary</a>: A new supercomputer is slated to make it possible to reduce animal experiments and perhaps to cure cancer. The hype surrounding quantum computing is inspirin...</li><li><a href="https://www.youtube.com/watch?v=0QczhVg5HaI">Why Neural Networks can learn (almost) anything</a>: A video about neural networks, how they work, and why they&#39;re useful.My twitter: https://twitter.com/max_romanaSOURCESNeural network playground: https://play...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1231217503993073715)** (25 messages🔥): 

- **Math PDFs Transformed into Conversational Partners**: Crizomb introduced an open-source Retriever-Answer Generator (RAG) project, **[ai_pdf](https://github.com/Crizomb/ai_pdf)**, that enables users to chat with any PDF locally; it is particularly effective with math documents by converting them to LaTeX for easy processing by computers.

- **Groundbreaking Real-time Video Generation**: Aifartist shared a **[Reddit post](https://www.reddit.com/r/StableDiffusion/comments/1c8oea6/endlessdreams_voice_directed_realtime_videos_at/)** showcasing a 2.5-minute video generated in real-time through voice direction. They emphasize the quick feedback loop and potential for real-time movie creation by just using voice commands.

- **Infini Attention Explained Simply**: Subham5089 wrote a simplified explanation of the new **Infini Attention**, designed to help with understanding its impact on AI and shared this write-up on **[LinkedIn](https://www.linkedin.com/posts/subham-kundu-2746b515b_llms-generativeai-activity-7187373540940148736-qNG6)**.

- **Innovative Bot Programming Achieved**: Acoloss shared an amusing update about their project, which involves bots with individual memories/history performing actions based on their capabilities. They noted the implementation is functioning surprisingly well with thoughtful output communication.

- **3LC's Beta Launch to Revolutionize Datasets**: The **[3LC](https://3lc.ai/)** platform has been announced, offering tools to refine datasets and ML models, enhancing Computer Vision with plans to extend support to LLMs. Users can join the beta to shape the platform's development, with exclusive access for 100 users and free non-commercial use.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ehristoforu/llama-3-12b-instruct">ehristoforu/llama-3-12b-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/ehristoforu/Gixtral-100B">ehristoforu/Gixtral-100B · Hugging Face</a>: no description found</li><li><a href="https://3lc.ai/">Home</a>: no description found</li><li><a href="https://huggingface.co/ehristoforu/Gistral-16B">ehristoforu/Gistral-16B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bineric/NorskGPT-Llama3-8b">bineric/NorskGPT-Llama3-8b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF">QuantFactory/Meta-Llama-3-70B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3">RAG chatbot using llama3</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1c8oea6/endlessdreams_voice_directed_realtime_videos_at/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/Crizomb/ai_pdf">GitHub - Crizomb/ai_pdf: Chat locally with any PDF  Ask questions, get answer with usefull references  Work well with math pdfs (convert them to LaTex, a math syntax comprehensible by computer)</a>: Chat locally with any PDF  Ask questions, get answer with usefull references  Work well with math pdfs (convert them to LaTex, a math syntax comprehensible by computer) - Crizomb/ai_pdf</li><li><a href="https://huggingface.co/spaces/clinteroni/outpainting-with-differential-diffusion-demo">Outpainting Demo - a Hugging Face Space by clinteroni</a>: no description found</li><li><a href="https://huggingface.co/spaces/gojiteji/VTuberLogoGenerator">VTuberLogoGenerator - a Hugging Face Space by gojiteji</a>: no description found</li><li><a href="https://huggingface.co/spaces/Csplk/moondream2-batch-processing">moondream2-batch-processing - a Hugging Face Space by Csplk</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1231195072347902092)** (4 messages): 

- **Seeking Architecture for Invoice Data Extraction**: A member is working on a project to extract data from invoices and receipts which are scanned images and is seeking an architecture to create a machine learning model for this task.

- **TrackNetV3 in Action**: A member has shared the [TrackNetV3 repository](https://github.com/qaz812345/TrackNetV3) but is inquiring about processing the model's output for each frame read, rather than reading all frames and computing.

- **Introducing Themselves**: A user named jackwean_75093 has joined and greeted the community.

- **Quest for Personal Knowledge Base Construction**: The same user, jackwean_75093, asked about how to build a private knowledge base but provided no further details.

**Link mentioned**: <a href="https://github.com/qaz812345/TrackNetV3">GitHub - qaz812345/TrackNetV3: Implementation of paper - TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification</a>: Implementation of paper - TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification - qaz812345/TrackNetV3

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1231284422171562076)** (10 messages🔥): 

- **Seeking M2M100 Finetuning**: A member is asking for a finetuning code for the **M2M100** model.
- **Request for PHI-2 Tuning Assistance**: A member is looking for help with fine-tuning the **PHI-2** model.
- **Batch Size Strategy for Fine-tuning**: Discussions suggest starting with smaller batch sizes, such as **32**, and adjusting upwards to find the optimal batch size for a 2.7B model on 16GB memory, with **gradient accumulation** as a possible solution.
- **Rust Port of minbpe Announced**: The `minbpe-rs` project is a **Rust** port of `minbpe` and is available on GitHub with features like `GPT4Tokenizer`, `save`, `load`, and a `train` function. The project is led by @gnp with contributions to the documentation and README. [Check out the project](https://github.com/gnp/minbpe-rs).
- **Dependency Clash and Dataset Acquisition Trouble**: One member mentions **Bertopic's new release** causing dependency conflicts with OpenAI's and has temporarily locked their script to version 0.16.0. Simultaneously, another member seeks assistance in integrating the **go-emotions dataset** into their project.

**Link mentioned**: <a href="https://github.com/gnp/minbpe-rs">GitHub - gnp/minbpe-rs: Port of Andrej Karpathy&#39;s minbpe to Rust</a>: Port of Andrej Karpathy&#39;s minbpe to Rust. Contribute to gnp/minbpe-rs development by creating an account on GitHub.

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1231733039730659399)** (10 messages🔥): 

- **Android Tablet Struggles with Focus**: A member queried how to use **fooocus** on an Android tablet, seeking guidance from the community.

- **Professional Diffusers Offer Their Services**: A member with expertise in web design, MVPs, app development, and various technical skills including **Stable Diffusion and Computer Vision** offered their services for startups and enterprises.

- **The Forbidden Model Access**: A user faced a **403 error** while trying to download a model using **vespa** and sought assistance from the community to resolve it.

- **Trouble Loading the StoryGen Model**: A member encountered an issue loading the **haoningwu/StoryGen** model using the **DiffusionPipeline** due to a problem with the config json, and reached out for support, specifically tagging another user for help.

- **Debate on AI-generated Video for "AI Horse"**: A user asked if it’s possible to create a 1-minute video on the topic of "AI Horse" entirely with **Diffusion**, prompting another member to suggest using **pika** or some other form of Diffusion Transformer for the task.
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1231200576419921920)** (77 messages🔥🔥): 

- **Query on Mojo's Reporting Issues and Newsletter Contributions**: A member inquired about how to get issues assigned and whether articles could be submitted for the Mojo newsletter, with responses pointing out the process involves showing the ability to fix things and that newsletter contributions currently aren't a supported feature.
- **Discussion on Assistive Technology Support in GTK**: Members discussed the importance of good assistive technology support in applications, using GTK and the lack of certain features in it as an example. The value of such technologies was debated, but agreed upon as beneficial in gaining user traction.
- **Mojo Docs Update Inquiry**: A member asked if the documentation on docs.modular.com is auto-generated from `mojo doc`; the reply indicated that while it is, there's a lot of non-public CI involved, and that it isn't designed for public use yet.
- **Performance Comparison between Mojo and Python**: A comparison raised by a member between Mojo and Python in printing numbers speed led to a reference to a [known issue](https://github.com/modularml/mojo/issues/975) about Mojo's lack of buffered IO and advice on performance benchmarking, suggesting the issue remains unaddressed since December.
- **Docs.modular.com Display Bug at 995px Width**: Members reported and discussed a UI bug on the docs.modular.com site where search results fail to display at certain browser widths. A dialogue with a developer revealed that this is a known behavior that occurs at a width of 995px and could be circumvented by avoiding use at that specific width or closing the search to view content.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=40107007">no title found</a>: no description found</li><li><a href="https://github.com/modularml/mojo/issues/975):">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/basalt-org/basalt">GitHub - basalt-org/basalt: A Machine Learning framework from scratch in Pure Mojo 🔥</a>: A Machine Learning framework from scratch in Pure Mojo 🔥 - basalt-org/basalt
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1232024125971890256)** (6 messages): 

- **Teaser Alert**: Modular shared a [mysterious teaser tweet](https://twitter.com/Modular/status/1782457222511161545), hinting at something brewing in the horizon.
- **Anticipation Builds with Modular**: A [second tweet](https://twitter.com/Modular/status/1782457235454689752) by Modular raises expectations among followers, suggesting an imminent reveal.
- **Countdown to Excitement**: The suspense continues with Modular's [third teaser tweet](https://twitter.com/Modular/status/1782457253829935500), pointing to a significant announcement.
- **Momentum Gathers at Modular**: In a [fourth tweet](https://twitter.com/Modular/status/1782457261652312486), Modular keeps the community on the edge of their seats, with an apparent countdown.
- **The Final Tease**: Modular's [final tweet](https://twitter.com/Modular/status/1782457268354809918) in the series leaves followers eagerly waiting for a big revelation.
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1231629012510703649)** (3 messages): 

- **Seeking Engagement for AI Video**: A member shared a [YouTube video](https://youtube.com/watch?v=SfKGHKzkm-o) titled "The Rise of AI" for a college assignment and asked for engagement and feedback. They acknowledged the limitations of the content depth due to time constraints and mentioned that English is not their first language.

- **The Quest for Artificial Conscious Life**: A member expressed interest in double majoring in computational physics and computer science/engineering with the aim to create artificial conscious life. They questioned the current state of AI, inefficiency in power and data, and the potential need for advancements like quantum computing or ternary systems to achieve this goal.

- **Skeptical View on Quantum Computing for AI**: Discussing the employment of quantum computing in AI, a member pointed out the challenges of randomness and efficiency in quantum systems, referencing the difficulty of performing simple operations with consistency. Concerns were also voiced about government intervention potentially impeding progress in this domain.

- **Ternary Computing Mentioned in AI Development**: A brief mention of a ternary computing system, the [Setun computer](https://en.wikipedia.org/wiki/Setun), was made in relation to discussing advancements necessary for developing artificial general intelligence (AGI). The member argued that computational architecture is more crucial than mere scaling in computing for progress towards AGI.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Setun">Setun - Wikipedia</a>: no description found</li><li><a href="https://youtube.com/watch?v=SfKGHKzkm-o">The Rise of AI</a>: (Hidupkan Closed Caption)(Turn on the Closed Caption)Bergabunglah bersama kami dalam perjalanan melalui evolusi cepat Artificial Intelligence, mulai dari kem...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1231249583397408778)** (338 messages🔥🔥): 

- **Exploring Type State Patterns in Mojo**: A user inquired about implementing the Type State Pattern in Mojo, and another member shared **associated types in traits** as a potential solution. However, this feature seems to be not yet implemented in stable Mojo, but it might work with a workaround using `Size` trait with `_getitem` and `_setitem`.
  
- **Understanding Mojo Parameters and Arguments**: One user clarified the difference between parameters and arguments in Mojo - parameters are compile-time constants, while arguments are runtime values. The confusion arose during a discussion about sorting algorithms, where a snippet using `T:Sortable` trait with a `cmp_fn` function parameter was shared, prompting exploration into function parameters represented in square brackets.

- **Sorting With Traits Strategy**: Another member shared an example quicksort implementation using traits and provided feedback on enhancing it. Despite the code running into a '`T' does not implement the '__ge__'' method error, discussions included using `UnsafePointer` instead of `Pointer` and understanding that a `Sortable` trait with overloaded comparison operators (`__le__` and `__ge__`) can be useful for sorting custom data types.

- **Issues with Pointers and Lists**: There were discussions about a segmentation fault caused when trying to utilize strings with pointers. Users discussed potential causes such as misallocations or the use of value semantics leading to unexpected behaviors, highlighting the intricacies of memory management in Mojo.

- **Regex Functionality and Mojo Implementation**: A user pondered the implementation of regex functionality in Mojo, sharing a Python example for context, and noted that as of the channel history cut-off, there is no regex implementation in Mojo. They expressed an intention to attempt a basic form of regex for a project idea.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/equality_comparable#__eq__">equality_comparable | Modular Docs</a>: EqualityComparable</li><li><a href="https://docs.modular.com/mojo/stdlib/collections/">collections | Modular Docs</a>: Implements the collections package.</li><li><a href="https://docs.modular.com/mojo/stdlib/algorithm/sort#partition">sort | Modular Docs</a>: Implements sorting functions.</li><li><a href="https://docs.python.org/3/howto/sorting.html#key-functions">Sorting Techniques</a>: Author, Andrew Dalke and Raymond Hettinger,. Python lists have a built-in list.sort() method that modifies the list in-place. There is also a sorted() built-in function that builds a new sorted lis...</li><li><a href="https://joyofmojo.com/generic_quicksort/">Generic Quicksort</a>: Context Mojo Reference: Sort Mojo Version: 24.2.1 Demo: Sorting a Group of People by Age This demo showcases how to sort a group of people based on their age using a versatile QuickSort algorithm. Thi...</li><li><a href="https://programmersought.com/article/66388921702/">Python -c command line execution method - Programmer Sought</a>: no description found</li><li><a href="https://docs.modular.com/mojo/manual/traits">Traits | Modular Docs</a>: Define shared behavior for types.</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/simd">simd | Modular Docs</a>: Implements SIMD struct.</li><li><a href="https://gist.github.com/modularbot/3334ea937074b8d2349fddaee2a04cd1">playground.mojo</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://docs.modular.com/mojo/stdlib/memory/unsafe#bitcast-2">unsafe | Modular Docs</a>: Implements classes for working with unsafe pointers.</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/builtin/anytype.mojo">mojo/stdlib/src/builtin/anytype.mojo at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://www.arewewebyet.org/">Are we web yet? Yes, and it's freaking fast! </a>: no description found</li><li><a href="https://docs.modular.com/mojo/manual/parameters/#parameterized-functions">Parameterization: compile-time metaprogramming | Modular Docs</a>: An introduction to parameters and compile-time metaprogramming.</li><li><a href="https://tenor.com/view/ron-swanson-parks-and-rec-its-so-beautiful-gif-15644547">Ron Swanson Parks And Rec GIF - Ron Swanson Parks And Rec Its So Beautiful - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/modularml/mojo/issues/2113)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/toiletsandpaper/mojo_zlib_classification/blob/master/tools/utils.mojo">mojo_zlib_classification/tools/utils.mojo at master · toiletsandpaper/mojo_zlib_classification</a>: Contribute to toiletsandpaper/mojo_zlib_classification development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2197">[Feature Request] `.__doc__` attribute · Issue #2197 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? I would like to be able to get the doctsring of my str...</li><li><a href="https://github.com/modularml/mojo/issues/2164)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://mlir.llvm.org/">MLIR</a>: no description found</li><li><a href="https://youtu.be/lXAp6ZAWyBY?si=OSuCzPUmuohgUYvL">2023 LLVM Dev Mtg - MLIR Is Not an ML Compiler, and Other Common Misconceptions</a>: 2023 LLVM Developers&#39; Meetinghttps://llvm.org/devmtg/2023-10------MLIR Is Not an ML Compiler, and Other Common MisconceptionsSpeaker: Alex Zinenko------Slide...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1231289596415836291)** (35 messages🔥): 

- **Cryptic Llama Project Enigma**: Interest was expressed in building a project cryptically referenced as "🦙🦙🦙.🔥", with suggestions towards an office suite with illustrative capabilities using text as prompts.

- **Mojo Projects Galore**: Project updates included `prism`'s typed flags, `mog` for terminal styling, `gojo` emulating Go's `net` package, and work on `termios` for MacOS, all available on GitHub with nightly tuple updates required. ([Prism](https://github.com/thatstoasty/prism), [Mog](https://github.com/thatstoasty/mog), [Gojo](https://github.com/thatstoasty/gojo), [Termios](https://github.com/thatstoasty/termios))

- **Basalt Framework Seeks Web Devs**: The Basalt machine learning framework team is seeking Web Development expertise, especially in UI/UX with NextJS and ShadCN knowledge, for launching and enhancing their autogenerated documentation. Visit [Basalt's GitHub](https://github.com/basalt-org/basalt) for details.

- **Mojo and the World of JSX**: A request was made to create an LSX.mojo repository for a React-like development built on HTML syntax, suggesting a strong interest in component-based UI frameworks within Mojo. The idea of a Mojo static site generator was hinted upon, with a Djot parser in development. ([LSX Repo](https://github.com/lsh/lsx))

- **MoCodes Breaks into Error Correction**: The MoCodes project was shared, which is an Error Correction (De)Coding framework written in Mojo. It aims to optimize compute-intensive error correction code processes traditionally handled by dedicated hardware. Collaboration is sought as outlined in the README on [GitHub](https://github.com/alainrollejr/mocodes).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/basalt-org/basalt">GitHub - basalt-org/basalt: A Machine Learning framework from scratch in Pure Mojo 🔥</a>: A Machine Learning framework from scratch in Pure Mojo 🔥 - basalt-org/basalt</li><li><a href="https://github.com/thatstoasty/prism">GitHub - thatstoasty/prism: Mojo CLI Library modeled after Cobra.</a>: Mojo CLI Library modeled after Cobra. Contribute to thatstoasty/prism development by creating an account on GitHub.</li><li><a href="https://github.com/thatstoasty/mog">GitHub - thatstoasty/mog: Style definitions for nice terminal layouts.</a>: Style definitions for nice terminal layouts. Contribute to thatstoasty/mog development by creating an account on GitHub.</li><li><a href="https://github.com/thatstoasty/gojo">GitHub - thatstoasty/gojo: Experiments in porting over Golang stdlib into Mojo.</a>: Experiments in porting over Golang stdlib into Mojo. - thatstoasty/gojo</li><li><a href="https://github.com/thatstoasty/termios">GitHub - thatstoasty/termios: Mojo termios via libc</a>: Mojo termios via libc. Contribute to thatstoasty/termios development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1231200334995521667)** (19 messages🔥): 

- **Exploring Performance with CPU Limits**: In a test limiting the CPU to 1400 MHz, **Mojo scalar** performed at 1.4 ns per item, while **Rust and Mojo SIMD** were similar at about 1.0 ns per item, even after including debug prints before and after the timed section.

- **Seeking the Optimal Parallelize Strategy**: A member noted differences in the use of `parallelize` in **X thread demo** and `Matmul` documentation, with the latter specifying `num_workers` in contrast to the former. Performance variability and a lack of stability were reported when not explicitly setting the number of workers.

- **The Multithreading Conundrum**: Members discussed the complexity and best practices for setting the number of workers in multithreading. It was highlighted that multithreading performance varies based on the number of cores, the problem at hand, and whether it's acceptable for the program to saturate all resources.

- **Number of Workers: To Specify or Not?**: Another member echoed this sentiment, emphasizing the challenges and considerations in multithreading, and suggesting that sometimes setting the number of workers higher than the number of cores can be beneficial, as demonstrated in a [Modular blog post about Matmul](https://www.modular.com/blog/mojo-a-journey-to-68-000x-speedup-over-python-part-3).

- **Performance Puzzles in Random Number Generation**: A member posted a Mojo script for calculating pi via the Monte Carlo method, noting it was much slower than a Numba-jitted Python version, with a large portion of time spent generating random numbers. Following a recommendation to report this issue, an [issue was opened on GitHub](https://github.com/modularml/mojo/issues/2388) to address `random.random_float64` performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/mojo-a-journey-to-68-000x-speedup-over-python-part-3">Modular: Mojo🔥 - A journey to 68,000x speedup over Python - Part 3</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Mojo🔥 - A journey to 68,000x speedup over Python - Part 3</li><li><a href="https://www.infoq.com/presentations/multithreading/">The Dos and Don'ts of Multithreading </a>: Hubert Matthews describes some of the problems encountered in multithreading and discusses how to avoid them through appropriate design choices.</li><li><a href="https://docs.modular.com/mojo/stdlib/algorithm/functional#parallelize,">functional | Modular Docs</a>: Implements higher-order functions.</li><li><a href="https://github.com/modularml/mojo/issues/2388">[BUG] `random.random_float64` is extremely slow · Issue #2388 · modularml/mojo</a>: Bug description Generating one random number at a time in a for loop is extremely slow, almost 2 orders of magnitude slower than a numba-jitted equivalent. Context: I tried to use a simple Monte Ca...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1231366433217314909)** (24 messages🔥): 

- **C++ ORT Performance Queries**: One member was curious about how performance was being measured for C++ with ONNX Runtime (ORT) as compared to Mojo. They discussed Python's overhead and considered whether C++ inherently optimizes due to fewer Python API calls.
- **Image Processing in Python vs. C++**: Another discussion revolved around pre-processing images in Python/Mojo using numpy and cv2 versus C++ using its native OpenCV and custom functions. It was noted that post-processing is primarily executed with native code in both languages.
- **Benchmark Sharing Offer**: One member mentioned they conducted performance benchmarks across three languages and offered to share a comparative table of the results.
- **ONNX Model Input Dilemma Solved**: A member faced an issue with an ONNX model accepting an input tensor named "input.1" and sought a workaround for using it with the `model.execute` call. A solution using **PythonObject** and the alternative approach using **kwargs** in Python were provided.
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1231136409642074123)** (36 messages🔥): 

- **Pointer Conundrum and Unsafe Adventures**: The community discussed the semantics of various pointer types, with suggestions to prefix some with "Unsafe" to reflect their nature. There's also work underway to phase out `LegacyPointer`, and contributions are encouraged as seen in a [small PR](https://github.com/modularml/mojo/pull/2365) aimed at this effort.

- **Troubleshooting the Update Snags**: A user highlighted an issue with the recent update to Mojo version 2024.4.1618, where `SIMDType.to_int()` was causing build failures. It was clarified that the method has been replaced with a simple `int(...)` call following the update.

- **Taking on String Comparisons**: A snippet of code was proposed for implementing String comparisons with an eye out for future Unicode considerations, prompting a review of a previous PR that addressed similar concerns. 

- **Tuple Copy Mystery and UnsafePointers**: A question was raised about the use of `__get_address_as_owned_value` in tuple copying operations, suggesting a possible conflict with how new `UnsafePointer` types should handle references and lifetimes.

- **String Representations and Semantic Conundrums**: The distinction between `String()` and `String("")`, where the latter includes a null terminator, prompted discussions about their proper allocation behaviors and the philosophical implications of what constitutes an empty string.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/1904">[Feature Request] Explicit parametric alias with default argument · Issue #1904 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? As title. What is your motivation for this change? Exp...</li><li><a href="https://github.com/modularml/mojo/pull/2365">[stdlib] Replace `Pointer` by `UnsafePointer` in `stdlib/src/builtin/object.mojo` by gabrieldemarmiesse · Pull Request #2365 · modularml/mojo</a>: Builtins imports behave in a weird way, I had to import LegacyPointer in stdlib/src/python/_cpython.mojo, I have no explanation for this. I just import what the compiler asks me to import :p See ht...
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1231138113774944297)** (462 messages🔥🔥🔥): 

- **LLaMa 3 Tokenizer Troubles**: Members in the discord discussed issues with fine-tuning LLaMa 3 models, highlighting problems with BOS (beginning-of-sentence) tokens not being added as they should be. A workaround involved manually updating `tokenizer.json` using a Pull Request found in [Llama HF discussions](https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41) which fixed the issue.

- **GPUs and Training Time Revelations**: Conversation sparked around the high resource expenditure for training AI models, especially upon the release of the Phi-3 models. One member noted a setup of 512 H100-80G GPUs for 7 days, indicating the large scale of computing power required.

- **Phi-3 Surpasses Expectations**: Comparisons in the channel showed that even though Phi-3 models are relatively smaller in parameters (around 3.8b), they are demonstrating performance competitive with much larger models, leading to speculation and interest in their efficiency and potential.

- **OpenAI and the AI Race**: Members discussed OpenAI's silence amidst rapidly evolving AI model releases from competitors. Speculation included OpenAI’s focus on the release of GPT-5 in 2025 and the potential for current models to influence or accelerate those plans.

- **Phi-3 Licensing and Capabilities**: The open MIT license of the Phi series was highlighted as a significant advantage, despite the models' lack of extensive knowledge databases. Conversation suggested the models might excel at reasoning over memory, positioning them as an exciting option for future application integration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1104757954588196865/1192621815172964522/1192712427750572032">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format">Axolotl - Instruction Tuning</a>: no description found</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: no description found</li><li><a href="https://huggingface.co/chargoddard/llama3-42b-v0">chargoddard/llama3-42b-v0 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome>">Environment variables</a>: no description found</li><li><a href="https://huggingface.co/mattshumer/Llama-3-8B-16K">mattshumer/Llama-3-8B-16K · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_train">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ca4q50/psa_check_that_your_training_setup_is_adding_bos/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.9-llama3-8b/discussions/11">cognitivecomputations/dolphin-2.9-llama3-8b · Llama 3 Base Is Unique</a>: no description found</li><li><a href="https://www.philschmid.de/fsdp-qlora-llama3">Efficiently fine-tune Llama 3 with PyTorch FSDP and Q-Lora</a>: Learn how to fine-tune Llama 3 70b with PyTorch FSDP and Q-Lora using Hugging Face TRL, Transformers, PEFT and Datasets.</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/41">meta-llama/Meta-Llama-3-8B · Update post-processor to add bos</a>: no description found</li><li><a href="https://github.com/janphilippfranken/sami">GitHub - janphilippfranken/sami: Self-Supervised Alignment with Mutual Information</a>: Self-Supervised Alignment with Mutual Information. Contribute to janphilippfranken/sami development by creating an account on GitHub.</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct">meta-llama/Meta-Llama-3-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c8r08t/are_there_any_llama_3_8b_finetunes_already/l0gs1mb/>">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1231211561906081864)** (19 messages🔥): 

- **GPU Struggles with 8-bit Optimizers**: A member remarks that multi-GPU setups are necessary but points out issues with 8-bit optimizers not working as intended.
- **VRAM Voracious AdamW_Torch**: AdamW_Torch optimizer is identified as a VRAM-heavy alternative given the subpar performance of 8-bit optimizers.
- **Seeking Configurations for 8b Optimizer**: Members are requesting and sharing example configurations for 8-bit optimizers on models like LLaMA3.
- **Troubleshooting Discord Links**: Members are attempting to share Discord links, but facing issues with them not working as expected.
- **Subjective Improvement Post Patch**: After applying a patch to LLaMA3, members notice subjective improvements despite loss metrics remaining unchanged, with emphasis on the "vibes eval" over loss data.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1231222189815107714)** (19 messages🔥): 

- **QMD vs. Markdown**: There was a query about the sudden switch to **qmd** for documentation, with concerns raised about its rendering on GitHub.
- **Quantization Config Inquiry**: A member inquired about the **quantization configuration** for a 70B model, and it was clarified that the **config.json** from 'examples/quantize.py' is commonly used.
- **Merging Model Duration Concern**: Discussion on the time it takes to merge back LoRA to base after fine-tuning a **70B model on 4 A100s**; over one and a half hours was considered long by a member.
- **Conversational Dataset Clarification**: A question about whether "train_on_inputs" affects labels in a multi-turn conversational dataset was confirmed; it particularly impacts user inputs.
- **Dataset Types and Documentation**: There was a request for information on types of datasets, and a member shared a [comprehensive link](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) detailing the **dataset formats supported by Axolotl**, including conversation, pre-training, instruction tuning, template-free, and custom pre-tokenized datasets.

**Link mentioned**: <a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1232142687084347422)** (1 messages): 

- **Llama's Got Length**: A link to **Llama 3, a model with 16K token length**, was shared accompanied by a seemingly impressed emoticon. The link leads to [huggingface.co](https://huggingface.co/mattshumer/Llama-3-8B-16K), indicating a user's interest in the extended-length capabilities.

**Link mentioned**: <a href="https://huggingface.co/mattshumer/Llama-3-8B-16K">mattshumer/Llama-3-8B-16K · Hugging Face</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/)** (1 messages): 

duh_kola: not axolotl related but yeah i canlt uplaod shit to hub using runpod
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1231356634819854396)** (22 messages🔥): 

- **Clarification on YAML Config "conversation:" Key**: A member inquired about the `"conversation:"` key for training datasets in the YAML config file. Another member clarified that this only applies to datasets of type `sharegpt`.

- **Complications with "sharegpt" and "chatml"**: When a member asked about the effects of specifying `"type: sharegpt"` and `"conversation: chatml"`, they were informed that this signifies the dataset is in ShareGPT format and instructs data transformation into ChatML format for model training.

- **Error Troubleshooting Steps Suggested**: Following a member's report of multiple `SIGBUS` (Signal 7) errors during distributed computing, they are advised to check for memory alignment issues, review memory-mapped file usage, check hardware, update dependencies, and simplify their setup to diagnose the problem.

- **Guide on Using Unsloth with Axolotl**: A question about integrating Unsloth into Axolotl for training culminated in a brief guide, instructing to install dependencies, prepare the model and data, configure Unsloth with the correct parameters, run the training process, and monitor outcomes for efficient optimization.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e7301808-4b94-41b9-b3d4-752db98cf71f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=33a203be-00f7-40dc-9fa2-e911b904e980)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=e4ffa5d8-9095-4a00-8773-02132978f2e7)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=4eadad10-1146-45ad-9822-155e9b87cb48)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1231399047944998925)** (7 messages): 

- **Load Balancer Optimizations In Progress**: Traffic on **Wizard 8x22b** is causing performance hits, but load balancer adjustments are expected to improve latencies soon.

- **Improved Throughput for Requests**: Changes to the **load balancer** and fixes related to stop tokens handling should enhance non-stream request throughput.

- **Deletion of Nitro Instruct Model**: Requests to **Databricks: DBRX 132B Instruct (nitro)** will now be rerouted to the main [Databricks: DBRX 132B Instruct](https://openrouter.ai/models/databricks/dbrx-instruct) model.

- **Introducing New Models and Extended Context Support**: OpenRouter announces 3 new models including a free **Llama 3 finetune**, as well as the extension of **Llama 3 8B** to a 16k context. Alongside model launches, improvements in prompt formatting and region-specific networking issues are also being tackled, with a focus on enhancing **dynamic routing**. [Model discussions and details can be found here.](https://discord.com/channels/1091220969173028894/1232005820229877822/1232005820229877822)

- **MythoMax 13B Issue Resolution**: Users experiencing problems with **MythoMax 13B** outputs should see improvements following a mitigation of issues by the top provider. Concerns can be reported in the provided [discussion thread](https://discord.com/channels/1091220969173028894/1232171735944532059).

- **Addressing Spike in 504 Errors**: Users are experiencing 504 errors due to networking issues in the **central and west US regions**, affecting Llama 2 tokenizer models. A fix that removes dependency on Hugging Face, which is currently down, is under development.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/databricks/dbrx-instruct).">Databricks: DBRX 132B Instruct by databricks | OpenRouter</a>: DBRX is a new open source large language model developed by Databricks. At 132B, it outperforms existing open source LLMs like Llama 2 70B and Mixtral-8x7B on standard industry benchmarks for language...</li><li><a href="https://openrouter.ai/models/lynn/soliloquy-l3.">Lynn: Llama 3 Soliloquy 8B by lynn | OpenRouter</a>: Soliloquy-L3 is a fast, highly capable roleplaying model designed for immersive, dynamic experiences. Trained on over 250 million tokens of roleplaying data, Soliloquy-L3 has a vast knowledge base, ri...</li><li><a href="https://openrouter.ai/models/sao10k/fimbulvetr-11b-v2">Fimbulvetr 11B v2 by sao10k | OpenRouter</a>: Creative writing model, routed with permission. It&#x27;s fast, it keeps the conversation going, and it stays in character.  If you submit a raw prompt, you can use Alpaca or Vicuna formats.</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-8b-instruct:extended">Meta: Llama 3 8B Instruct (extended) by meta-llama | OpenRouter</a>: Meta&#x27;s latest class of model (Llama 3) launched with a variety of sizes &amp; flavors. This 8B instruct-tuned version was optimized for high quality dialogue usecases.  It has demonstrated strong...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1231369890829439056)** (3 messages): 

- **Contract Standards Awareness Suggestion**: A product feedback suggested that users should be prompted to choose the contract standard during upload to ensure awareness that only specific contract types are supported. This may prevent confusion over non-processed, non-supported contracts. 

- **User Localization and Contract Favorability Feature Ideas**: Another suggestion was proposed to allow users to set their location during onboarding or upload to account for local laws, and to enable a feature indicating which party the user wants to favor in the negotiation process.

- **Illegal Terms Detection Feature Request**: It was also recommended that the product should have the ability to detect illegal and onerous terms within contracts to prevent dead contracts caused by the inclusion of illegal terms by non-lawyers.

- **Keywords AI: A Tool for Developers Built on OpenRouter**: An announcement for [Keywords AI](https://keywordsai.co), a platform supporting OpenRouter including all models and the "bring your own key" option, was made, highlighting its two-line integration and developer-centric features.

- **DeepGaze Launch with Reddit Monitoring**: The launch of [DeepGaze](https://www.deepgaze.ca/), a service that feeds multiple document types into GPT-4V and uses a Discord bot to identify Reddit users with issues matching its capabilities, was shared. DeepGaze leverages OpenRouter to keep up with the latest LLM models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://keywordsai.co)">no title found</a>: no description found</li><li><a href="https://www.deepgaze.ca/">DeepGaze</a>: no description found
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1231144139890884639)** (474 messages🔥🔥🔥): 

- **More Woes with WizardLM-2**: Users report inconsistent performance with **WizardLM-2**; some finding success while others encounter incoherence or non-responsiveness. One user identified **SillyTavern's** 'Assistant Prefill' potentially causing issues with **LLaMA 3** models, while another discussed difficulties stemming from Mircosoft's billing system only showing one invoice.
  
- **OR's Response to Technical Glitches**: **OpenRouter** acknowledges issues related to provider tokenizers. A hotfix was deployed to address Hugging Face-related downtime, with a promise of a permanent fix that eliminates the dependency.

- **Rates and Tokenomics Scrutinized**: Users question how AI model providers can afford to offer services at current rates, especially when compared to the costs of image generation. Discussions mention the possible role of FP8 quantization and active worker discounts in reducing expenses, with one user citing **Groq's** hardware as potentially less economical due to high energy consumption.

- **Exploring Uncharted Model Territories**: Members share their experiences and inquiries about a range of topics, including **Phi-3-mini models**, new **LLaMA 3 70b** variants, and **WizardLM-2**'s possible connections with Microsoft. Enthusiasts are eager to get their hands on the newly released models, while others speculate on **RWKV**'s future and compare AI writing styles.

- **Anticipating Model Updates and Additions**: **OpenRouter** users await uncensored versions of **LLaMA 3 70b**, discuss the significance of jailbreakable models, and ponder the potential arrival of **Phi-3** on the platform. They also note preferences for the **8x22** models, emphasizing the balance between cost and functionality.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/a/XoI7ZD9">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://www.semianalysis.com/p/groq-inference-tokenomics-speed-but">Groq Inference Tokenomics: Speed, But At What Cost?</a>: Faster than Nvidia? Dissecting the economics</li><li><a href="https://huggingface.co/openlynn/Llama-3-Soliloquy-8B">openlynn/Llama-3-Soliloquy-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx">microsoft/Phi-3-mini-128k-instruct-onnx · Hugging Face</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Work-to-rule">Work-to-rule - Wikipedia</a>: no description found</li><li><a href="https://x.com/erhartford/status/1781199815772438819">Tweet from Eric Hartford (@erhartford)</a>: Dolphin-2.9-llama3-8b generously sponsored by @CrusoeCloud ETA Saturday. Lots of collaboration with @LucasAtkins7 and @FernandoNetoAi. Dolphin-2.9-llama3-70b to follow.  Dolphin-2.9-mixtral-8x22b stil...</li><li><a href="https://huggingface.co/posts/WizardLM/329547800484476">@WizardLM on Hugging Face: &quot;🔥🔥🔥 Introducing WizardLM-2!

📙Release Blog:…&quot;</a>: no description found</li><li><a href="https://huggingface.co/dreamgen/opus-v1.2-llama-3-8b">dreamgen/opus-v1.2-llama-3-8b · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/playground?models=meta-llama/ll">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://openrouter.ai/playground?models=meta-llama/llama-3-8b-instruct">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://fireworks.ai/blog/fire-attention-serving-open-source-models-4x-faster-than-vllm-by-quantizing-with-no-tradeoffs">FireAttention — Serving Open Source Models 4x faster than vLLM by quantizing with ~no tradeoffs</a>: Serving Open Source Models 4x faster than vLLM by quantizing with ~no tradeoffs</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct">microsoft/Phi-3-mini-4k-instruct · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-70b-instruct:nitro">Meta: Llama 3 70B Instruct (nitro) by meta-llama | OpenRouter</a>: Meta&#x27;s latest class of model (Llama 3) launched with a variety of sizes &amp; flavors. This 70B instruct-tuned version was optimized for high quality dialogue usecases.  It has demonstrated stron...</li><li><a href="https://openrouter.ai/models/lynn/soliloquy-l3">Lynn: Llama 3 Soliloquy 8B by lynn | OpenRouter</a>: Soliloquy-L3 is a fast, highly capable roleplaying model designed for immersive, dynamic experiences. Trained on over 250 million tokens of roleplaying data, Soliloquy-L3 has a vast knowledge base, ri...
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1231148707211247669)** (303 messages🔥🔥): 

- **Atlas Robot Creeps into Discussions**: The latest release of the **Atlas robot** spurred conversations about its perceived creepiness and social media buzz strategy, anticipating the model intended for sale, with one member **looking forward** to seeing its eventual capabilities.
- **The AI Spirituality Debate**: A member asked what a form of AI spirituality might look like, leading to a heated debate about consciousness, humanity, and emotions in AI, moderated by rule enforcement on non-secular discussions.
- **GPT-3's API and Interface Innovations**: Discussion touched on the potential of creating APIs with **MyGPT**'s code and the advances in tools like **MetaGPT** and **Devika**, which help write apps and might interact with GitHub.
- **LLaMa 3's Importance and Limitations**: Members discussed the recent improvements in various AI models, with LLaMa 3 earning mixed reviews for its performance, and rumored release dates of **GPT-5** considered fake without official announcements.
- **Generative Model Literature and Exuberant AI**: A request for in-depth resources on AI and generative algorithms like ChatGPT and DALL-E was met with suggestions to search **OpenAI's published papers** and repositories like Arxiv, while an anecdote on **LLaMa 3's** unusual output—overusing exclamation marks—highlighted both the unexpected quirks and perceived limitations of the model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/joe-bereta-source-fed-micdrop-im-out-gif-11904628">Joe Bereta Source Fed GIF - Joe Bereta Source Fed Micdrop - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://en.wikipedia.org/w/index.php?title=Biorobotics">Biorobotics - Wikipedia</a>: no description found</li><li><a href="https://openai.com/research/generative-models">Generative models</a>: This post describes four projects that share a common theme of enhancing or using generative models, a branch of unsupervised learning techniques in machine learning. In addition to describing our wor...</li><li><a href="https://openai.com/research/overview">Research</a>: We believe our research will eventually lead to artificial general intelligence, a system that can solve human-level problems. Building safe and beneficial AGI is our mission.</li><li><a href="https://openai.com/research/gpt-4">GPT-4</a>: We’ve created GPT-4, the latest milestone in OpenAI’s effort in scaling up deep learning. GPT-4 is a large multimodal model (accepting image and text inputs, emitting text outputs) that, while less ca...
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1231139832542134272)** (33 messages🔥): 

- **GPT Agent and LLama 3 70B Integration Attempt**: A member shared their attempt at integrating Agent GPT v2 with LLama 3 70B using Groq but faced issues as others reported the integration was failing. However, some users did eventually find it operational, suggesting there might be intermittent access or user-specific conditions affecting functionality.

- **Caution Against Sharing CGPT Chats**: Concerns were raised about posting share URLs from cgpt chats, with members being cautious about sharing logs due to access and evaluation queries regarding the improvement of model responses without explicit feedback.

- **Exploring Convolutional Layers and LoRa in LLMs**: A discussion was held around whether convolutional layers, referred to as Hyena, are comparable to LoRa layers in other models like Stable Diffusion. One member provided insight that LoRa can be used for fine-tuning large language models (LLMs) with others inquiring about models actively employing these techniques and their benefits.

- **Tools for Managing ChatGPT History Needed**: Users are seeking tools or alternative websites to better manage their ChatGPT history, highlighting the limitations of the current portal offered by OpenAI. Attention was directed towards the potential necessity of an API key for any third-party management solutions.

- **Clarification on Fine-tuning and File Retention with ChatGPT**: A user was informed that fine-tuning on ChatGPT refers to feeding content through the API to change model behavior, and that documents uploaded act only as reference material which do not alter the underlying model. Additionally, it was pointed out that attached files to a chat are retained per existing OpenAI guidelines, with a user mentioning a 3-hour retention period based on prior conditions.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1231589837035929671)** (24 messages🔥): 

- **Brevity is Key in Custom Instructions**: Users discussed optimal length for custom instructions in ChatGPT; one user opts for minimal guidance to save context space, while others experiment with length, finding that too long instructions may be counterproductive, as AI might 'forget' them.
- **Seeking Criminal Law Prompts**: A law student inquires about prompts for criminal law, but the request remains open for suggestions or tips from the community.
- **Optimizing Email Enhancement with GPT-4**: A user is fine-tuning a program to enhance emails using GPT-4, asking for advice on how to improve the prompts when the AI's outputs are not satisfying.
- **Where's the Prompt Library?**: A member of the channel inquired about the location of a prompt library, a resource that could potentially aid in developing more effective prompts.
- **Prompt Engineering Tips and Ethics**: A discussion emerges on the practice of prompt engineering, touching on the ethical implications and concerns of sharing potentially harmful techniques; however, no concrete techniques or examples are provided.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1231589837035929671)** (24 messages🔥): 

- **Brief Custom Instructions Preferred**: A user noted keeping custom instructions simple, such as *Include semicolons, colons, and em dashes in your responses where applicable*, to preserve context window space.
- **Contemplating Instructions' Length and Quality**: A discussion about the length of prompts indicated users perceive that sometimes a longer, more detailed prompt does not necessarily yield higher quality responses from the AI, suggesting shorter prompts might be preferable.
- **Exploring Prompt Division Strategies**: In response to uncertainty about how to handle large prompts, one member advised breaking them down and spreading them over multiple messages to prevent the AI from forgetting previous parts.
- **Prompting Techniques and Personalities**: A user shared admiration for a prompt engineer named **RageGPTee**, who's known for advanced techniques and "disappearing" after sharing groundbreaking skills, yet another person humorously exaggerated his capabilities.
- **Email Enhancement via GPT-4 Queries**: A member is seeking advice on optimizing prompts for a program that uses GPT-4 to enhance email drafting, following occasional subpar outputs from the AI.
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1231244510629986304)** (298 messages🔥🔥): 

- **LLM Multimodal Concerns**: The channel participants discussed that existing multimodal datasets, which total around 2 million pairs, can cause overfitting of models on specific datasets like GPT-4v captions for LAION-COCO. This overfitting is a noted problem in current multimodal approaches.

- **MoA Architecture Unveiled**: A new architecture called Mixture-of-Attention (MoA) was shared, [described in this paper](https://snap-research.github.io/mixture-of-attention/), which allows disentanglement of subject and context generation in personalized image generation.

- **AI Surveillance Bots on Discord**: Concerns about surveillance bots joining Discord servers were discussed, with a link provided to [kickthespy.pet](https://kickthespy.pet/#823813159592001537), a service that identifies such bots using an API vulnerability.

- **Discussion on Training Text-Image Diffusion Models**: Users exchanged insights about the challenges of training text-image diffusion models, emphasizing the importance of data quality, size, and model architecture. An interesting point made was that while Chinchilla's training method isn't detailed, dropout and other regularization methods might significantly impact training outcomes.

- **Adobe Unleashes Firefly Image 3**: Adobe announced the beta release of [Adobe Firefly Image 3 Foundation Model](https://www.adobe.com/products/firefly.html), which offers improved image generation quality and speed, now integrated into Photoshop and accessible through the Firefly web application. Users were curious to test its capabilities with different creative prompts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://kickthespy.pet/#823813159592001537">Kick the Spy Pet</a>: no description found</li><li><a href="https://news.adobe.com/news/news-details/2024/Adobe-Introduces-Firefly-Image-3-Foundation-Model-to-Take-Creative-Exploration-and-Ideation-to-New-Heights/default.aspx">Adobe Introduces Firefly Image 3 Foundation Model to Take Creative Exploration and Ideation to New Heights</a>: no description found</li><li><a href="https://snap-research.github.io/mixture-of-attention/">Mixture of Attention</a>: no description found</li><li><a href="https://arxiv.org/abs/2203.15556">Training Compute-Optimal Large Language Models</a>: We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertra...</li><li><a href="https://tenor.com/view/oh-no-top-gear-jeremy-clarkson-no-one-cares-gif-18925814">Oh No Top Gear GIF - Oh No Top Gear Jeremy Clarkson - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://wandb.ai/bghira/adamwbf16-wave/runs/e52bd9c5a68e37556a7f56479e5c2cce?nw=nwuserbghira">bghira</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://paperswithcode.com/dataset/cub-200-2011">Papers with Code - CUB-200-2011 Dataset</a>: The Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset is the most widely-used dataset for fine-grained visual categorization task. It contains 11,788 images of 200 subcategories belonging to birds, 5...</li><li><a href="https://youtube.com/watch?v=SfKGHKzkm-o">The Rise of AI</a>: (Hidupkan Closed Caption)(Turn on the Closed Caption)Bergabunglah bersama kami dalam perjalanan melalui evolusi cepat Artificial Intelligence, mulai dari kem...</li><li><a href="https://huggingface.co/datasets/ptx0/mj-v52-redux/tree/main">ptx0/mj-v52-redux at main</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=fmI_OciHV_8">How To Build Generative AI Models Like OpenAI&#39;s Sora</a>: If you read articles about companies like OpenAI and Anthropic training foundation models, it would be natural to assume that if you don’t have a billion dol...</li><li><a href="https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/">[AINews] FineWeb: 15T Tokens, 12 years of CommonCrawl (deduped and filtered, you&#x27;re welcome)</a>: AI News for 4/19/2024-4/22/2024. We checked 6 subreddits and 364 Twitters and 27 Discords (395 channels, and 14973 messages) for you. Estimated reading time...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1231288421285236746)** (38 messages🔥): 

- **Benchmarking Blink's Visual Perception**: A new benchmark named Blink has been introduced for testing multimodal language models (LLMs) to evaluate their visual perception abilities. It covers tasks that humans can solve quickly but are surprisingly challenging for advanced multimodal LLMs like GPT-4V and Gemini, where they perform marginally better than random guessing. [Read more about Blink](https://arxiv.org/abs/2404.12390).

- **Upscaling Difficulties in Image Extrapolation**: There is ongoing work in improving the results of 2D rope extrapolation from a 256x256 resolution to a 1024x1024, which currently does not yield impressive results and requires higher resolution tuning.

- **Piecewise-Rectified Flow Integrates with ControlNet-Tile Pipeline**: Piecewise-Rectified Flow (PeRFlow) has been mentioned for upsampling images significantly, going from 64px to 1024px through a process that integrates the flow with the ControlNet-Tile pipeline and refines the images. This can be found on [GitHub's piecewise-rectified-flow](https://github.com/magic-research/piecewise-rectified-flow/blob/main/README.md).

- **HiDiffusion Enhances Diffusion Model Resolutions**: HiDiffusion, a new development by MEGVII Technology and ByteDance, claims to increase the resolution and speed of diffusion models with a single line of code. The module displays artifacts in its outputs, raising questions about its efficacy in generating coherent high-resolution images. [Explore the HiDiffusion project](https://hidiffusion.github.io/).

- **SEED-X Multimodal Foundation Model**: SEED-X aims to bridge the gap in multimodal foundation models by comprehending images of arbitrary sizes and enabling multi-granularity image generation. The unified and versatile foundation model demonstrates effectiveness in real-world applications with multi-granularity visual semantics for comprehension and generation tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.14396">SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation</a>: The rapid evolution of multimodal foundation model has demonstrated significant progresses in vision-language understanding and generation, e.g., our previous work SEED-LLaMA. However, there remains a...</li><li><a href="https://arxiv.org/abs/2404.12803">TextSquare: Scaling up Text-Centric Visual Instruction Tuning</a>: Text-centric visual question answering (VQA) has made great strides with the development of Multimodal Large Language Models (MLLMs), yet open-source models still fall short of leading models like GPT...</li><li><a href="https://wandb.ai/bghira/simpletuner-deepfloyd/runs/c2d8a68009185bfe4bc1072957e426db/workspace?nw=nwuserbghira">bghira</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://github.com/magic-research/piecewise-rectified-flow/blob/main/README.md">piecewise-rectified-flow/README.md at main · magic-research/piecewise-rectified-flow</a>: Contribute to magic-research/piecewise-rectified-flow development by creating an account on GitHub.</li><li><a href="https://wandb.ai/bghira/simpletuner-deepfloyd/runs/c2d8a68009185bfe4bc1072957e426db/workspace?nw=nwu">bghira</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://arxiv.org/abs/2404.12390">BLINK: Multimodal Large Language Models Can See but Not Perceive</a>: We introduce Blink, a new benchmark for multimodal language models (LLMs) that focuses on core visual perception abilities not found in other evaluations. Most of the Blink tasks can be solved by huma...</li><li><a href="https://hidiffusion.github.io/">SOCIAL MEDIA TITLE TAG</a>: SOCIAL MEDIA DESCRIPTION TAG TAG</li><li><a href="https://github.com/megvii-research/HiDiffusion">GitHub - megvii-research/HiDiffusion</a>: Contribute to megvii-research/HiDiffusion development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1231977925314744360)** (6 messages): 

- **Coding Assistant Collaboration**: A member mentioned they are starting to build an NLP coding assistant focused on **JavaScript/Rust** rather than Python and expressed interest in collaborating with others.
  
- **Time Constraints on Collaboration**: *softmax_function* indicated a willingness to help occasionally with the project, citing a busy schedule with multiple projects.

- **In Search of Past Work**: *jcarbonnell* inquired about the existence of a repository with previous work that could be useful for the NLP coding assistant project.

- **Admitting Past Limitations**: *softmax_function* acknowledged discontinuing a previous project due to a lack of AI knowledge at the time, but noted an improved ability to contribute now.

- **Seeking Task Assignment Clarification**: *jcarbonnell* expressed difficulty in assigning tasks without understanding *softmax_function*'s past contributions, and intends to try a *TrainedModel.py* script and dataset shared by them.
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1231284983495135372)** (6 messages): 

- **RAG Experimentation Gets a Makeover**: Aishwarya Prabhat introduces a framework named **DREAM** for experimenting with Distributed RAG, highlighting the importance of a robust infrastructure for creating production-ready RAG systems. The details and insights are hosted on the [LlamaIndex tweet](https://twitter.com/llama_index/status/1781725652447879672).

- **Finance Bot Framework by LlamaIndex**: Hanane Dupouy shares a mini-blog on how to use **@llama_index** to build a finance agent that can retrieve stock prices and summarize financial news, enhancing interactions with public company data. Further exploration can be found in the shared [Twitter link](https://twitter.com/llama_index/status/1781837902139551920).

- **ColBERT with a Memory Twist**: Discussing the challenges in adding conversation history into a RAG pipeline, LlamaIndex proposes a retrieval agent powered by **ColBERT** that stores "state" for a conversational assistant. Learn more about this method in their [recent tweet](https://twitter.com/llama_index/status/1782086279498539330).

- **RAG Fine-Tuning with LoRA**: Mariboo's tutorial is highlighted for demonstrating the use of LoRA weights to fine-tune embedding models, a critical part of the RAG pipeline, using **@llama_index finetuning abstractions** and **@huggingface**. Dive into the tutorial via LlamaIndex's [Twitter post](https://twitter.com/llama_index/status/1782201783110213783).

- **Level-Up Your RAG with Open-Source Rerankers**: @JinaAI_ releases two **open-source rerankers** that enhance RAG systems by applying a second level of ranking to vector search on embeddings. The details about these rerankers are shared in a [tweet by LlamaIndex](https://twitter.com/llama_index/status/1782531355970240955).

- **CRAG: Innovative Layer for RAG Retrieval**: LlamaIndex discusses Corrective RAG (**CRAG**) which utilizes a **"reflection"** layer to categorize retrieved information as "Correct," "Incorrect," or "Ambiguous," addressing the issue of bad retrieval in RAG. Insights into CRAG are detailed in LlamaIndex's [tweet](https://twitter.com/llama_index/status/1782799757376963006).
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1231156600190795826)** (188 messages🔥🔥): 

- **Choosing the Right Retrieval Method**: Users discussed different retrieval approaches such as RAG, CRAG, and retanking with Vector Databases vs. Knowledge Graphs. The consensus points towards _use-case specificity_, especially when dealing with company summaries where information loss is a concern, leading to preferences towards larger chunk sizes or using SQL and graph technologies.
  
- **Integration and Summarization Challenges**: One member shared frustration over a bot that only replies with document-related responses after integrating ChainLit with LlamaIndex, hinting at context management issues within a Retriever-Answer Generator (RAG) system.

- **AI Models and OpenAI Dependence**: Questions arose surrounding the use of alternative models like Groq, Bedrock, and Ollama within the llama_index infrastructure, with members resolving doubts related to API key errors and correct embedding model usage.

- **Indexing and Storage Explorations**: Members inquire about the functionality and integration of Vector Stores such as Supabase, Chromadb, and Qdrant, often confronting warnings, bugs, or 401 errors that hint at a reliance on OpenAI’s API key even when not explicitly utilized.

- **Summarization Using DocumentSummaryIndex**: One member sought advice on how to make DocumentSummaryIndex consider all nodes for summarization, as the tool only selected one node for summary generation out of several resulting from the document split process.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/">Agents - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/?h=settings">Migrating from ServiceContext to Settings - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/WeaviateIndex_auto_retriever/?h=auto">Auto-Retrieval from a Weaviate Vector Database - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/?h=rag+cli#customization">RAG CLI - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/callbacks/TokenCountingHandler/?h=token">Token Counting Handler - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/bedrock/?h=bedroc">Bedrock - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/deploying/query_engine/usage_pattern#get-started>).">Usage Pattern - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/ollama/?h=ollama">Ollama - Llama 2 7B - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/localai/">LocalAI - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/portkey/?h=portkey)">Portkey - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/pull/13009">fix qdrant bug with checking existing collection by logan-markewich · Pull Request #13009 · run-llama/llama_index</a>: Small bug with getting info from possibly existing collection</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/?h=query+pipeline+tool">Building an Agent around a Query Pipeline - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/indexing/indexing#vector-store-index>).">Indexing & Embedding - LlamaIndex</a>: no description found</li><li><a href="https://mer.vin/2024/02/crewai-rag-using-tools/">CrewAI RAG using Tools - Mervin Praison</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/documents_and_nodes/usage_documents#metadata>)">Using Documents - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/data_connectors/PathwayReaderDemo#create-the-document-indexing-pipeline>).">Pathway Reader - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/querying/querying#querying>)">Querying - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/getting_started/starter_example_local#query-your-data>)">Starter Tutorial (Local Models) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/evaluation/UpTrain#create-a-query-engine-using-llamaindex>).">How to use UpTrain with LlamaIndex - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1231221804518080615)** (5 messages): 

- **Infini Attention Explained**: An explanation of the new **Infini Attention** technology was shared on LinkedIn, highlighting its potential and expressing anticipation for its upcoming implementations. Read the explainer on [LinkedIn](https://www.linkedin.com/posts/subham-kundu-2746b515b_llms-generativeai-activity-7187373540940148736-qNG6).

- **Comprehensive AI Funding Data Updated**: A comprehensive dataset tracking AI funding and company distributions by city is now available for community review. Check out the dataset and related city distribution analysis via [Google Sheets](https://docs.google.com/spreadsheets/d/1nWBP1MpT7sACYDxqdCo8gBR7b2nXJbrF9Z43y69q9hg/edit#gid=752020121) or the Tweet by @WangUWS on [Twitter](https://x.com/WangUWS/status/1782069636030165106).

- **LLM-Ready Markdown Gets a Boost**: LLM-ready Markdown experiences a new level of integration with FireCrawl and LlamaIndex. Read about the advancements on [Medium](https://medium.com/ai-advances/unleash-the-potential-of-llm-ready-markdown-firecrawl-and-llamaindex-integration-243e494a9eb8).

- **Launching Schema-Controlled Knowledge Graphs**: WhyHow.AI introduced a significant upgrade to their Knowledge Graph SDK, enabling the creation of schema-controlled automated knowledge graphs from PDFs. For insights and participation in the Beta program, refer to the announcement on [Medium](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3).

- **Debate on Optimal Databases for LLM Training**: There's an active conversation regarding what the ideal database type for LLM training might be, with questions raised about the suitability of relational, document, columnar databases, as well as the necessity of vector databases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/spreadsheets/d/1nWBP1MpT7sACYDxqdCo8gBR7b2nXJbrF9Z43y69q9hg/edit#gid=752020121">[FrontierOptic.com] AI Raise Tracking - April 21 2024 - Community Review Copy</a>: Cover   &lt;a href=&quot;http://FrontierOptic.com&quot;&gt;FrontierOptic.com&lt;/a&gt; AI Startup Fund Raise Data (Since May 2023) - Community Review Copy &lt;a href=&quot;https://twitter.com/WangUWS&...</li><li><a href="https://x.com/WangUWS/status/1782069636030165106">Tweet from Howe Wang (@WangUWS)</a>: To celebrate 20 years since @HilaryDuff sang &#39;Could be New York, Maybe Hollywood and Vine, London, Paris, maybe Tokyo,&#39; in &#39;Wake Up&#39;. I cleaned up the AI Hype Train data&#39;s location...
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1231255250250367139)** (110 messages🔥🔥): 

- **Exploring Open Interpreter's Features and Integration**: General discussions about Open Interpreter (OI) functionalities include questions about using the `--server` argument for building clients, challenges with OI on Windows systems, and issues with installing OI, linked to a specific [GitHub issue](https://github.com/OpenInterpreter/open-interpreter/issues/1185). There was also a mention of successfully using OI with the LLM model Llama3 for Python tasks.
- **Model Compatibility and Performance**: Users are discussing the performance of various models with OI, including the **Llama3 70b**, and one confirms it running well using the `--local` mode. Meanwhile, there were queries about the best text-to-speech services for live streaming and humanlike interaction.
- **AI Vision Model Clarifications**: It's indicated that Open Interpreter uses **GPT-4-vision-preview** for recognizing screenshots. The model name was provided in response to a user's inquiry about the LLM model used for vision tasks.
- **Development Challenges and Solutions Shared**: Users provided solutions for issues such as pytesseract errors and shared fixes, including the command `pip install --upgrade litellm`. Contributions to troubleshooting are also being streamed and shared on platforms like YouTube, with a video detailing how to integrate OI with GROQ API for potentially cheaper operations.
- **Community Collaboration and Development**: The community is actively discussing contributions to OI, offering help to new users interested in hardware like Raspberry Pi, and sharing their setups. One user mentioned reaching **100 contributors on GitHub** for OI, while another shared a [GitHub pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1204) they authored. There's also interest in sharing default configuration files to improve model interactions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/open-interpreter-1146610656779440188?event=1232412426557722755">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://tenor.com/view/que-gif-27530657">Que GIF - Que - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/scobleizer/status/1782520422678052999?s=46&t=kwbSfLYCOimQnegJhHK_iA">Tweet from Robert Scoble (@Scobleizer)</a>: #17: Making humans better with new AI  The Rabbit AI device took the Consumer Electronics Show by storm in January, which inspired @hellokillian Killian Lucas, founder of Open Interpreter,  to build a...</li><li><a href="https://pastebin.com/ugNMQ57v">▌ OS Control enabled&gt; open notepad and write &quot;hello&quot;  Let&#039;s start by try - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/issues/1185">Bug when fresh install and new start · Issue #1185 · OpenInterpreter/open-interpreter</a>: Describe the bug when i run it. this warning shown interpreter /opt/conda/lib/python3.11/site-packages/pydantic/_internal/fields.py:151: UserWarning: Field &quot;model_id&quot; has conflict with prote...</li><li><a href="https://x.com/kodjima33/status/1782492783762399700?s=46">Tweet from Nik Shevchenko (@kodjima33)</a>: FRIEND became the largest opensource AI wearable community in the world  To support the builders, we are launching an App Marketplace  You can now build your own app and it will work with the device  ...</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/project_management/hardware/devices/raspberry-pi">01/project_management/hardware/devices/raspberry-pi at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://github.com/ishank26/posts/blob/main/llama3_new.pdf">posts/llama3_new.pdf at main · ishank26/posts</a>: resources, thoughts and notes. Contribute to ishank26/posts development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=FXCaJ3Ga9TE">How to use Open Interpreter cheaper! (LM studio / groq / gpt3.5)</a>: Part 1 and intro: https://www.youtube.com/watch?v=5Lf8bCKa_dE0:00 - set up1:09 - default gpt-42:36 - fast mode / gpt-3.52:55 - local mode3:39 - LM Studio 5:5...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1213">Update local profile so it doen&#39;t use function calling by Notnaton · Pull Request #1213 · OpenInterpreter/open-interpreter</a>: leaving model = gpt4 will result in function calling. Most LM Studio models dont use function calling. making it not work Describe the changes you have made: Reference any relevant issues (e.g. &quot;...</li><li><a href="https://pastebin.com/b0bwxmzm">(oi) C:\Users\ivan&gt;interpreter --api_base &quot;https://api.groq.com/openai/v1&quot; --api - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://github.com/KoljaB/RealtimeTTS">GitHub - KoljaB/RealtimeTTS: Converts text to speech in realtime</a>: Converts text to speech in realtime. Contribute to KoljaB/RealtimeTTS development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/986">Jupyter export magic command by tyfiero · Pull Request #986 · OpenInterpreter/open-interpreter</a>: Describe the changes you have made: Added a %jupyter magic command to export the current session as a jupyter notebook file, that you can run in Google Collab. Reference any relevant issues (e.g. &quo...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1204">Bump version of tiktoken by minamorl · Pull Request #1204 · OpenInterpreter/open-interpreter</a>: Describe the changes you have made: Bumped version of tiktoken since build process is broken for some reason. This PR fixes broken process. Reference any relevant issues (e.g. &quot;Fixes #000&quot;):...
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1231154192153055262)** (22 messages🔥): 

- **Mix-up with Model Names**: One member stated they mistakenly said they got Open Interpreter working with Groq and Llama 3 70b, but they meant another similar-ish named service and clarified that **01 only supports OAI for the cloud option** currently.
- **Llama 3 Models Stability Issues**: It was mentioned that **Llama 3 70b** seems more unstable compared to Llama 3 8b, though specific details about the instability were not provided.
- **Windows Client Troubles**: Several members are experiencing issues with **01 on Windows**, with suggestions indicating there might be a client-related problem that needs addressing.
- **Recording Woes on M1 Mac**: Users reported an issue where pressing the spacebar on an M1 MacBook did not initiate recording in 01, but instead kept inputting spaces; various solutions were suggested, including installing **ffmpeg**, checking microphone and terminal permissions, or using a specific version of **Python via conda**.
- **Cloud Compatibility Request**: A member expressed interest in running **01 in the cloud**, such as on brev.dev, asking about compatibility with cloud services like Scaleway, highlighting a need for cross-platform support.
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1232362713162973244)** (39 messages🔥): 

- **The Quest for a Click-Worthy AGI Title**: The channel explored various titillating titles for an article on AGI, aiming to strike a balance between clickbait and substance. Titles like "AGI Isn't real," "AGI is religion, not science," and "AGI is what you want it to be" were debated.

- **The Importance of Audience Satisfaction**: Nathan underscored the priority of serving current readers over attracting new ones, indicating that current Discord members would appreciate the content regardless of the title's click-worthiness.

- **Controversial Paper Discourse**: A discussion took place addressing widespread criticism of the Sparks paper within the community, citing issues like irreproducibility and overhyped claims.

- **Debating AGI's True Nature**: The conversation touched upon beliefs about AGI, with some members suggesting it's more a matter of faith than science. A Business Insider article was mentioned where Mistral's CEO Arthur Mensch expressed skepticism about tech giants' portrayal of AGI.

- **Legal Spectacle on AGI Definition**: Nathan found humor in the idea that a jury might have to determine the definition of AGI due to a clause between OpenAI and Microsoft, with a community member suggesting it could be used strategically by OpenAI to sever ties with Microsoft.

**Link mentioned**: <a href="https://www.businessinsider.com/mistrals-ceo-said-obsession-with-agi-about-creating-god-2024-4?utm_source=copy-link&utm_medium=referral&utm_content=topbar">AI CEO says people&#x27;s obsession with reaching artificial general intelligence is &#x27;about creating God&#x27;</a>: Arthur Mensch doesn&#x27;t feel concerned about AI surpassing human intelligence, but he does worry about American tech giants dominating the field.

  

---


**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1232022902266789948)** (44 messages🔥): 

- **Phi Series Benchmarks Stir Debate**: Tweets shared in the community highlight discussion on the impressive benchmark results of **Phi-3**, mentioning **LLAMA 3 8B** as a standout model and **Phi-3 Mini** (4b), **Small** (7b), and **Medium** (14b) as having significant benchmark improvements due to synthetic data pipelines. Concerns are raised regarding the use of benchmarks to evaluate models, suggesting that overfitting on benchmarks makes models like Phi-3 perform well in tests but poorly out-of-distribution (OOD).

- **Skepticism Surrounding Phi-3's Validity**: Users express suspicion about the integrity of **Phi-3**, with some characterizing it as "SUS" and others critiquing it for mainly being comprised of textbooks, which could advantage it in benchmarks like **MMLU** without necessarily ensuring broad capabilities.

- **Phi-3 Evaluated as "Clusterfuck"**: A conversation around **Phi-3** criticizes the manner in which its evaluations are presented, pointing out the lack of disclosure about the data pipeline and questionable inclusion of a matplotlib plot as a JPEG in the documentation.

- **Insights on Training Data and GPU Priorities**: The discussion sheds light on the possibility that a focus on smaller models could stem from GPU limitations at **Microsoft Research (MSR)**, with comparisons made regarding GPU resource allocation between MSR and other teams or organizations such as **OAI**.

- **Phi-3 Anticipated Release and Multilingual Capability**: Conversation anticipates **Phi-3's** impending release under an **MIT license** and notes its multilingual capabilities, indicating a broader scope than previously recognized.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sebastienbubeck/status/1782650351692476742?s=46">Tweet from Sebastien Bubeck (@SebastienBubeck)</a>: @itsGauravAi Good thing that you will be able to try for yourself tomorrow :-).</li><li><a href="https://fxtwitter.com/dylan522p/status/1782461647497400324">Tweet from Dylan Patel (@dylan522p)</a>: LLAMA 3 8B was amazing but will be overshadowed Phi-3 mini 4b, small 7b, medium 14b this week, and the benchmarks are fucking insane Synthetic data pipelines are massive improvements over internet dat...</li><li><a href="https://x.com/teortaxestex/status/1782499722797674781?s=46">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: @angelusm0rt1s @fchollet It&#39;s my conviction that you can benchmark benchmarks by how well phi-2 does on them relative to some obviously capable models like Mixtral  If phi-2 &gt;&gt; mixtral your ...</li><li><a href="https://tenor.com/view/where-is-my-free-coffee-gif-25537785">Where Is GIF - Where Is My - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/nearcyan/status/1782662543858979112">Tweet from near (@nearcyan)</a>: blacked out the irrelevant parts of the phi-3 paper to help everyone understand how it performs so well for its size</li><li><a href="https://fxtwitter.com/suchenzang/status/1782823571561279860?s=46">Tweet from Susan Zhang (@suchenzang)</a>: oh no not this again</li><li><a href="https://arxiv.org/abs/2404.14219">Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone</a>: We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens, whose overall performance, as measured by both academic benchmarks and internal testing, rivals that of ...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1231364661522337823)** (9 messages🔥): 

- **Evaluations Categorization in the Spotlight**: A member discusses the **Evals section** of their research and touches on the immediate utility of automated evaluations like MMLU and BIGBench versus time-costly human evaluations like ChatBotArena.
- **The Role of Perplexity-Based Evals**: The same member questions the role of *perplexity-based evaluations* like AI2's Paloma and how they compare to task-based evaluations such as MMLU. There's uncertainty about whether Paloma was intended just for internal checks during training or as a broader public benchmark.
- **Benchmark Categorization Approval**: Both members express appreciation for a categorization of benchmarks from the MT Bench paper, indicating that it provides a helpful framework, even though the categorization of tools like Paloma isn't clear-cut.
- **Utility of Multi-Dataset Perplexity-Based Metrics in Training**: A member ponders if multi-dataset perplexity-based evaluations are more about monitoring model performance at training checkpoints rather than for post-completion model competitions. They seek confirmation on this understanding.
- **Confirming Perplexity's Role**: Another member confirms that perplexity-based evaluations are indeed used as checkpoints during training, rather than as competitions for completed models, though it is a relatively new concept for them as well.
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1232028192353816709)** (25 messages🔥): 

- **Discord's Hidden Gem**: Despite having 13k free subscribers and 250 eligible for Discord, only about **50 have joined** the channel, with plans to make its value more obvious through a quarterly shoutout, hinting at [Ben Thompson](https://stratechery.com/)'s style.
- **Peek into Deep Dives**: A member shared their analysis of the 'roadmap to pluralism' paper, with feedback suggesting the topic is currently *evergreen content* and welcomes any thoughts on the [Typefully draft](https://typefully.com/t/AstZhn4).
- **Community Engagement Differentials**: Some members mention they enjoy lurking and reading the content shared in the channel, while another voices the challenge of following too many Discords.
- **The Ephemeral Tweeter**: One user is amused by a researcher (Ross Taylor, lead of Galactica) who posts interesting tweets and deletes them within seconds, positing that past negative feedback might lead to such fleeting digital presence.
- **Candid Interviews Await NDA Clarity**: The host expresses interest in interviewing Ross Taylor but also shows reluctance due to potential NDA restrictions that could prevent an open and informative discussion.

**Link mentioned**: <a href="https://typefully.com/t/AstZhn4">no title found</a>: no description found

  

---


**Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1232223235249279046)** (9 messages🔥): 

- **LLM Benchmarks Discussion**: A link to a recent tweet discussing the current state of large language model (LLM) benchmarks was shared: [current state of llm benchmarks](https://x.com/nearcyan/status/1782634477510119767).

- **Suspicious Activity Noted**: A member mentioned being "sus", possibly implying suspicion or cautiousness within the context.

- **It's Live!**: Members discussed the timing of an unnamed feature or service going live, clarifying that it happened an hour ago.

- **Model Updates on Hugging Face**: It was noted that updates, including a 128k context length model, are now available on Hugging Face.

- **Search Web for Interesting Results**: A member pointed out that enabling the search web feature could result in discovering information about an Australian politician sharing the name Nathan Lambert.

**Link mentioned**: <a href="https://x.com/nearcyan/status/1782634477510119767">Tweet from near (@nearcyan)</a>: current state of llm benchmarks

  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1232178107838890084)** (5 messages): 

- **Instruction Tuning Gains Traction**: A member highlighted an [introductory blog post on instruction tuning](https://gaotianyu.xyz/blog/2023/11/30/instruction-tuning/) and recent progress in the field. The post is appreciated for its breadth of references and narrative though it's noted it could benefit from editing.
- **Getting to Grips with CRINGE**: The CRINGE loss paper, connected to instruction tuning, was shared which discusses a training method using negative examples to improve model performance. This method is detailed in a [paper](https://arxiv.org/abs/2211.05826) that focuses on avoiding issues like unsafe generation and contradictions.
- **LLMBar in RewardBench Utilization Noted**: It was mentioned by a member that LLMBar is used in RewardBench, a response to a query about similarity with another LLM-evaluator meta-benchmark.
- **Endorsement for LLM-Evaluator Benchmark Tools**: A comment was made expressing approval for the LLM-evaluator meta-benchmark, suggesting its utility.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gaotianyu.xyz/blog/2023/11/30/instruction-tuning/">Teach Llamas to Talk: Recent Progress in Instruction Tuning</a>: no description found</li><li><a href="https://arxiv.org/abs/2211.05826">The CRINGE Loss: Learning what language not to model</a>: Standard language model training employs gold human documents or human-human interaction data, and treats all training data as positive examples. Growing evidence shows that even with very large amoun...
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1231147213900484699)** (71 messages🔥🔥): 

- **Insights on Job Hunting for Engineers**: A member shared concerns about the challenges of landing a job through traditional applications, highlighting that *personal projects and a strong GitHub presence* are more beneficial. They also discussed the somewhat surprising benefit of *having big company names on resumes* over actual work done when it comes to getting interviews and jobs.
- **Web-Search for Academia**: One user, a student of Homeric Studies, listed multiple academic websites, such as academia.edu and perseus.tufts.edu, that they use with a script for web-search purposes, demonstrating interest in connecting command-R to rich educational resources.
- **Cohere Outreach Request**: A user requested help with implementing Cohere Command-R with URL Grounding to BotPress for chat functionalities, expressing that many users might switch to Cohere given its performance and competitive pricing.
- **Guidance on Cohere's Chat API Capabilities**: Questions arose on how to restrict a chat model to respond only within its training scope. Suggestions included using **preambles** and **BOS/EOS tokens**, with the goal of sharpening model outputs to specific topics.
- **Meetup on Variational Autoencoders by ML-Maths**: An upcoming talk by Dr. Matthew Bernstein on the *mathematics behind VAEs* and their applications in single-cell genomics was announced, inviting participants to learn about these deep, probabilistic models. The event underscores the community's interest in advanced ML topics.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://drive.google.com/file/d/11TiGQ-JxqmLQ-TJ24Jui8V9kXsI6QZld/view">Ken&#39;s Resume.pdf</a>: no description found</li><li><a href="https://docs.oracle.com/en/cloud/paas/autonomous-database/serverless/adbsb/sql-generation-ai-autonomous.html#GUID-3721296F-14A1-428A-B464-7FA25E9EC8F3">Using Oracle Autonomous Database Serverless</a>: Oracle Autonomous Database Select AI enables you to query your data using natural language.
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1231240985791696980)** (8 messages🔥): 

- **Open Source Announcement**: A new matchmaking application using **@cohere Command R+**, **@stanfordnlp DSPy**, **@weaviate_io Vector store**, and **@crewAIInc agents** has been open-sourced. [A video and GitHub links for the application](https://x.com/anmol_desai2005/status/1781679469679325605?s=46&t=vUJbpAOoGDUfvrA5TGBjTQ) were shared for exploration and feedback.

- **Challenges in Web Scraping Automation**: A member is developing a **generic web scraper** that utilizes **gpt-4-turbo** to identify (selector, column) pairs but is facing difficulties with the model accurately finding and interacting with input elements for selection and clicking.

- **Prompt IDE Tool for Optimal Performance**: Prompt Mixer, a desktop application for creating, evaluating, and utilizing AI prompts, was mentioned with a feature rundown. It offers functionalities such as automatic version control, AI recommendations, and the ability to test prompt chains. Details are available at [Prompt Mixer's website](https://www.promptmixer.dev/).

- **Request for Assistance with Cohere and BotPress**: A user is seeking help to implement **Cohere Command-r with URL Grounding (RAG)** into **BotPress**. They conceptually endorse Cohere and provide context that many using ChatGPT in BotPress may switch if successfully integrated.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.promptmixer.dev/">Prompt Mixer. AI Development Studio for companies</a>: A collaborative workspace for managers, engineers and data experts to develop AI features.</li><li><a href="https://x.com/anmol_desai2005/status/1781679469679325605?s=46&t=vUJbpAOoGDUfvrA5TGBjTQ">Tweet from Anmol Desai (@anmol_desai2005)</a>: We did it. Finally the code is open sourced. Please give it a try and we are eager for a feedback. @weaviate_io @stanfordnlp @cohere @1vnzh @CShorten30  ↘️ Quoting Muratcan Koylan (@youraimarketer)   ...
</li>
</ul>

</div>
  

---


**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1231910638087835669)** (1 messages): 

- **Seeking Norwegian Cohere Collaborators**: A member is inquiring if there are any Norwegian companies, preferably consulting firms, which have experience with Cohere and can act as a reference or consultant for a project they are working to initiate.
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1231143136332480512)** (63 messages🔥🔥): 

- **Seeking Help with Groq/Mixtral Tool Calls**: A member asked for tips on using **LangChain with Groq/Mixtral** for Tool_calls, noting **Groq** is limited to a single tool and parallel calls are disabled; they are considering how to execute single calls in sequence.
  
- **Vision Models Come to the Rescue**: In discussions about processing documents "in the wild," members suggested that **Language Models (LLMs)** are not sufficient on their own and that **vision models** are necessary for a generalized solution.

- **The Picture-Language Union Using LLama**: A conversation about the latest methods for communicating images to language models revealed using **a special image token in prompts that gets replaced by the output of the vision encoder**, providing a base64 encoded image to convert visuals into language-readable format.

- **Real-time Chat Topic Management**: One user sought advice on **managing and categorizing topics in a real-time chat** between clients and assistants, looking to associate chat messages with existing topics or create new ones where necessary.

- **Startup Interface for Vector Database Chat**: As part of seeking a quick startup interface setup where customers can log in and chat with a vector database, **LangChain** was recommended along with tools like Groq or Llama, while also applying standard practices like **setting up LangChain with needed API keys, creating a login system, and establishing a chat interface** connected to the vector database.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai/lesson/5/cross-encoder-re-ranking">DLAI - Advanced Retrieval for AI with Chroma</a>: Introduction · Overview of embeddings-based retrieval · Pitfalls of retrieval - when simple vector search fails · Query Expansion · Cross-encoder re-ranking · Embedding adaptors · Other Techniques</li><li><a href="https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/">Cross Encoder Reranker | 🦜️🔗 LangChain</a>: This notebook shows how to implement reranker in a retriever with your</li><li><a href="http://localhost:11434",>">no title found</a>: no description found</li><li><a href="https://js.langchain.com/docs/integrations/chat/groq#setup>)">ChatGroq | 🦜️🔗 Langchain</a>: Setup</li><li><a href="https://js.langchain.com/docs/modules/model_io/llms/quick_start#setup>)">Quick Start | 🦜️🔗 Langchain</a>: Large Language Models (LLMs) are a core component of LangChain.</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/quickstart#function-calling>))">Quickstart | 🦜️🔗 Langchain</a>: In this guide, we will go over the basic ways to create Chains and Agents that call Tools. Tools can be just about anything — APIs, functions, databases, etc. Tools allow us to extend the capabilities...</li><li><a href="https://js.langchain.com/docs/integrations/chat/google_vertex_ai#vertexai-tools-agent>)">ChatVertexAI | 🦜️🔗 Langchain</a>: LangChain.js supports Google Vertex AI chat models as an integration.</li><li><a href="https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm#code-generation-chat-models>)">ChatVertexAI | 🦜️🔗 LangChain</a>: Note: This is separate from the Google PaLM integration. Google has</li><li><a href="https://github.com/langchain-ai/langchain/issues/13442>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1231345297452564531)** (9 messages🔥): 

- **GitHub Project To Structure Web Data**: Mishushakov introduced a new GitHub project called [LLM Scraper](https://github.com/mishushakov/llm-scraper/), which can turn any webpage into structured data using large language models (LLMs). The community is encouraged to star the project on GitHub.
  
- **Assistance Requested for Product Hunt Ranking**: Anthology_ seeks community support to reach number one on Product Hunt with their AI tool, [AllMind AI: Your Personal Stock Analyst](https://www.producthunt.com/posts/allmind-ai-your-personal-stock-analyst), which stands at #5 and boasts faster and cheaper financial insights compared to other models.
  
- **Launch of Knowledge Graph SDK at WhyHow.AI**: Chiajy announced WhyHow.AI's major upgrade with schema-controlled automated knowledge graphs that structure data from user-uploaded content. Details for the beta program and integration capabilities were shared, along with a link to the introduction post on [Medium](https://medium.com/enterprise-rag/introducing-schema-controlled-automated-knowledge-graphs-02c7f00c3cf3).
  
- **Community Input Sought on Real-Time Chat Analysis**: Dewhysky seeks suggestions for managing topics/subjects/tasks in a real-time client and assistant chat, with the objective to associate messages with existing topics or create new ones as needed.
  
- **Server Specifications Inquiry for LLMs**: Vijay187 inquired about server requirements for using a large language model, which ansh_ai identified as needing two A100 GPUs with 80GB each for llama 3 70b.
  
- **Understanding Watermarking in LLMs**: Wisewander shared a resource regarding watermarking large language models, which involves embedding identifiable patterns in text generated by AI models like ChatGPT or Claude, detailed at [Watermarking LLMs](https://watermarking.aisimplyexplained.tech/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://watermarking.aisimplyexplained.tech/">AI Simply Explained</a>: AI Simply Explained</li><li><a href="https://www.producthunt.com/posts/allmind-ai-your-personal-stock-analyst"> AllMind AI: Your Personal Stock Analyst  - AI financial analyst with real-time market data &amp; insights | Product Hunt</a>: AllMind AI is your personal financial analyst, delivering centralized, real-time, actionable insights directly to you. Our proprietary LLM, AllMind AI, slashes research time by 90% and costs by 98%. W...</li><li><a href="https://github.com/mishushakov/llm-scraper/">GitHub - mishushakov/llm-scraper: Turn any webpage into structured data using LLMs</a>: Turn any webpage into structured data using LLMs. Contribute to mishushakov/llm-scraper development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1231989652353843331)** (1 messages): 

- **Bridging Natural and Structured Query with Langchain**: A member has detailed the workings of the **Self-querying retriever** in a [blog post](https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever), which discusses how Large Language Models (LLMs) and few-shot prompts build structured queries from natural language. The self-querying retriever enhances semantic similarity search by adding filtering capabilities to the results based on metadata.

**Link mentioned**: <a href="https://rito.hashnode.dev/rental-apartment-search-with-langchain-self-querying-retriever">Building a Rental Apartment Search with Langchain&#x27;s Self-Querying Retriever</a>: In this blog post, we delve into the capabilities of Langchain&#x27;s self-querying retriever, a powerful tool for bridging the gap between natural language and structured data retrieval. This retriev...

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1231579424420270110)** (26 messages🔥): 

- **Debating the Future of tinygrad**: Members discussed whether tinygrad/box/chip might pivot to becoming a [cloud service](https://www.eetimes.com/groq-ceo-we-no-longer-sell-hardware/), referencing opinions about AI and cloud services, and expressing a range of opinions on having decentralized versus cloud-based AI services.
- **TinyBox as AI Home Appliance**: The vision for **TinyBox** is to serve as a **home appliance** running advanced AI models, which local devices can interact with, bypassing the need for cloud servers and tackling censorship issues.
- **Portable AI Power vs. Cloud Scalability**: The debate continued with comparisons between local high-end AI hardware like TinyBox and the efficiency of cloud services, highlighting issues such as intermittent AI usage by consumers and current AI hardware limitations.
- **Local AI Training's Future Importance**: A user predicted that models will soon train on user data in *real-time* and emphasized the increasing relevance of local training hardware as models learn from smaller datasets.
- **Weekly Meeting Points for tinygrad Developers**: **George Hotz** outlined key discussion points for the weekly meeting, including the progress of *mlperf*, potential *NVIDIA CI* plans, and maintaining the tinygrad codebase under 7500 lines.

**Link mentioned**: <a href="https://tiny-tools-client.vercel.app">React App</a>: no description found

  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1231256105963749378)** (45 messages🔥): 

- **tinygrad with ROCm Hurdles**: A member is trying to set up tinygrad with ROCm but encounters segfaulting, looking for guidance post the ROCm 6.1 release.
- **Stacking Tensors in tinygrad**: In a detailed explanation, a member clarifies that `.stack` in tinygrad does realize the tensors by stacking them along a new dimension, while `.realize()` must be explicitly called to materialize computations in memory.
- **Master Branch Stability for tinygrad**: George Hotz affirms that the `master` branch of tinygrad should be stable and reliable due to robust CI processes, addressing a member's concerns about installation and functionality.
- **CUDA Compatibility and Windows Limitation**: Members discuss the challenges and workarounds for using tinygrad with CUDA on Windows, including WSL and Docker methods, while another member confirms that Windows is not officially supported.
- **In-Depth Guidance on tinygrad Mechanics**: Several members exchange resources to understand deep aspects of tinygrad, such as memory management, shape tracking, and handling in-place operations, leading to discussions about implementation details and documentation contributions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad-notes/shapetracker.html">How ShapeTracker works</a>: Tutorials on tinygrad</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/uops-doc.md">tinygrad-notes/uops-doc.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/cuda-tensor-core-pt1.md">tinygrad-notes/cuda-tensor-core-pt1.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/blob/37f8be6450b6209cdc9466a385075971e673c653/tinygrad/tensor.py#L169">tinygrad/tinygrad/tensor.py at 37f8be6450b6209cdc9466a385075971e673c653 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://meta.ai">Meta AI</a>: Use Meta AI assistant to get things done, create AI-generated images for free, and get answers to any of your questions. Meta AI is built on Meta&#039;s latest Llama large language model and uses Emu,...
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1232247102399320084)** (5 messages): 

- **Llama3 vs. Mixtral Face-Off**: A German RAG evaluation of **Llama3 70b instruct** was mentioned, but it appears that it doesn't perform as well as **Mixtral-8x7B-Instruct-v0.1** based on [this dataset](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval).

- **Metric Discrepancies Questioned**: A member raised concerns about why the "question to context" metric had large discrepancies compared to other metrics in the evaluation results. They suggested that *"loglikelihood_acc_norm_nospace"* might address the formatting issues causing these differences.

- **Potential Formatting Bug Spotted**: The possibility of a formatting bug in the query template was highlighted, specifically the absence of the "Answer:" part, which might impact the evaluation results. They referred to a relevant [GitHub source](https://github.com/huggingface/lighteval/blob/11b48333b46ecd464cc3979de66038c87717e8d6/src/lighteval/tasks/tasks_prompt_formatting.py#L83) for clarification.

- **Request for Command-R-Plus Comparison**: A comparison between **Llama3 70b instruct** and **command-r-plus** was requested to assess their respective performances.

- **DiscoLM German 7b Evaluation Details Shared**: A member shared detailed evaluation results of **DiscoLM German 7b**, noting significant improvement in 3 out of 4 categories over previously shared results and providing a performance comparison [here](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#discoresearchdiscolm_german_7b_v1).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/lighteval/blob/11b48333b46ecd464cc3979de66038c87717e8d6/src/lighteval/tasks/tasks_prompt_formatting.py#L83">lighteval/src/lighteval/tasks/tasks_prompt_formatting.py at 11b48333b46ecd464cc3979de66038c87717e8d6 · huggingface/lighteval</a>: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron. - hug...</li><li><a href="https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#discoresearchdiscolm_german_7b_v1">deutsche-telekom/Ger-RAG-eval · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval#meta-llamameta-llama-3-70b-instruct">deutsche-telekom/Ger-RAG-eval · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1231539029724631132)** (6 messages): 

- **Innovative Chatbot Execution Strategies**: Armifer91 is experimenting with categorizing chatbot functions into groups and implementing a function called "execute_model" to handle the execution of function groups, a strategy inspired by the **MoE (Mixture of Experts)** model but adapted for business applications. They are concerned about the commercial viability due to the large prompt size and are exploring embedding functions to dynamically provide functionality without excessive prompt length.

- **Haystack Framework Enhances Chatbots**: Vladimir0583 pointed out that the **Haystack LLM** framework can help with dynamically invoking services based on the user's intent by indexing them as openapi specs. A GitHub notebook was provided detailing this approach: [Haystack RAG Services Demo Notebook](https://github.com/vblagoje/notebooks/blob/main/haystack2x-demos/haystack_rag_services_demo.ipynb).

- **Seeking new tokens for Llama fine-tuning**: Sinan2 inquired about adding new special tokens to Llama for fine-tuning, wondering if it's as simple as editing the tokenizer's JSON files and training, or if the process is more complicated.

- **Frustration with Platform Downtime**: _jp1_ expressed dissatisfaction implying that the **Hugging Face platform** is down, followed by Maxidl's comment indicating that this interruption spoiled the evening's activities.

**Link mentioned**: <a href="https://github.com/vblagoje/notebooks/blob/main/haystack2x-demos/haystack_rag_services_demo.ipynb">notebooks/haystack2x-demos/haystack_rag_services_demo.ipynb at main · vblagoje/notebooks</a>: Contribute to vblagoje/notebooks development by creating an account on GitHub.

  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1231191792150511676)** (45 messages🔥): 

- **DiscoLM German Fine-Tuning Challenges**: Members discussed the limitations of fine-tuning **DiscoLM** on German benchmarks, noting that without substantial examples and relevant data, benchmark scores can decrease. There was mention of a tokenization issue with DiscoLM and proposed workarounds, such as using other models like **Instruct** as a foundation.

- **Experimenting with Whisper Models**: For German automatic speech recognition, suggestions were made to trial models such as [whisper-tiny-german](https://huggingface.co/primeline/whisper-tiny-german), [whisper-base-quant-ct2](https://huggingface.co/jvh/whisper-base-quant-ct2/), and [AISAK-Listen](https://huggingface.co/aisak-ai/aisak-listen), with additional advice on further finetuning or quantization for better quality and smartphone compatibility.

- **Conversation Templates and Tokenizer Confusions**: Discussions about the template and tokenizer complexities within **Llama-3** models ensued. It was highlighted that while using the **ChatML** template is standard, challenges arise with the tokenizer configuration, including having zero weights for special tokens and alternative eos_tokens for conversation turns.

- **Troubleshooting Model Generation Errors**: Help was provided to a member facing challenges with getting **DiscoLM German** to generate proper responses. Suggestions included using the `generate` function without the attention mask and utilizing text generation pipelines for easier application.

- **Llama3 Performance and Output Quality**: Members debated the potential of improving Llama3's performance in German, pondering whether the bottlenecks are computation or time. It was suggested to repeat the **LeoLM** style of training and reach out to the **occiglot** team for assistance, while also assessing the multilingual capabilities of the Llama3 70b model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/cstr/llama3-discolm-orca">cstr/llama3-discolm-orca · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/jvh/whisper-base-quant-ct2/">jvh/whisper-base-quant-ct2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/primeline/whisper-tiny-german">primeline/whisper-tiny-german · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/aisak-ai/aisak-listen">aisak-ai/aisak-listen · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1231218979809529906)** (53 messages🔥): 

- **Stretching the Context Window with Rope**: Members discussed the absence of providers using **rope** to extend the context window of large language models, with some expressing interest in the approach. Context was provided through a [Perplexity AI link](https://www.perplexity.ai/search/why-not-scale-0KMvWZKqSVGnYIBd_vpcng).

- **High Quality Web Data Release, FineWeb**: The release of **FineWeb**, containing 15 trillion tokens of web data, was brought to discussion, with a link to [Twitter](https://x.com/gui_penedo/status/1781953413938557276?s=46) posted. **FineWeb** supposedly exceeds previous datasets like RefinedWeb and C4 in model performance.

- **Hydra Framework Spurs Varied Reactions**: The AI community shared experiences with the **Hydra framework** from Facebook Research, designed for elegantly configuring complex applications. Some found it excellent for managing ML experiments ([GitHub link to Hydra](https://github.com/facebookresearch/hydra)), while others questioned its uniqueness.

- **Phi-3 Gains Weight**: There was buzz about Microsoft's **Phi-3** release, a successor to Phi-2 with three versions, all larger in size. Conversation included a [Tweet about Phi-3](https://x.com/arankomatsuzaki/status/1782594659761389655?s=46&t=90xQ8sGy63D2OtiaoGJuww) and speculation on its performance compared to other models like **llama 3 8B**.

- **Perplexity.ai Fundraising Success**: Remark was made on the recent funding announcement for **Perplexity.ai**, which has gained preference among some users over traditional search engines. The fundraising tweet can be found [here](https://x.com/AravSrinivas/status/1782784338238873769).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/gui_penedo/status/1781953413938557276?s=46">Tweet from Guilherme Penedo (@gui_penedo)</a>: We have just released 🍷 FineWeb: 15 trillion tokens of high quality web data. We filtered and deduplicated all CommonCrawl between 2013 and 2024. Models trained on FineWeb outperform RefinedWeb, C4, ...</li><li><a href="https://arxiv.org/abs/2404.14219">Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone</a>: We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens, whose overall performance, as measured by both academic benchmarks and internal testing, rivals that of ...</li><li><a href="https://arxiv.org/abs/2404.11483">AgentKit: Flow Engineering with Graphs, not Coding</a>: We propose an intuitive LLM prompting framework (AgentKit) for multifunctional agents. AgentKit offers a unified framework for explicitly constructing a complex &#34;thought process&#34; from simple n...</li><li><a href="https://x.com/agihippo/status/1782828359573205295?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from yi 🦛 (@agihippo)</a>: phi is a good litmus test to tell who understands LLMs and who doesn&#39;t.</li><li><a href="https://x.com/AravSrinivas/status/1782784338238873769">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Excited to announce we&#39;ve raised 62.7M$ at 1.04B$ valuation, led by Daniel Gross, along with Stan Druckenmiller, NVIDIA, Jeff Bezos, Tobi Lutke, Garry Tan, Andrej Karpathy, Dylan Field, Elad Gil, ...</li><li><a href="https://x.com/arankomatsuzaki/status/1782594659761389655?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Microsoft just released Phi-3  - phi-3-mini: 3.8B model trained on 3.3T tokens rivals Mixtral 8x7B and GPT-3.5 - phi-3-medium: 14B model trained on 4.8T tokens w/ 78% on MMLU and 8.9 on MT-bench  http...</li><li><a href="https://github.com/facebookresearch/hydra">GitHub - facebookresearch/hydra: Hydra is a framework for elegantly configuring complex applications</a>: Hydra is a framework for elegantly configuring complex applications - facebookresearch/hydra</li><li><a href="https://github.com/facebookresearch/mbrl-lib">GitHub - facebookresearch/mbrl-lib: Library for Model Based RL</a>: Library for Model Based RL . Contribute to facebookresearch/mbrl-lib development by creating an account on GitHub.</li><li><a href="https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/examples/conf/dynamics_model/gaussian_mlp_ensemble.yaml">mbrl-lib/mbrl/examples/conf/dynamics_model/gaussian_mlp_ensemble.yaml at main · facebookresearch/mbrl-lib</a>: Library for Model Based RL . Contribute to facebookresearch/mbrl-lib development by creating an account on GitHub.</li><li><a href="https://github.com/facebookresearch/mbrl-lib/blob/main/mbrl/examples/conf/main.yaml">mbrl-lib/mbrl/examples/conf/main.yaml at main · facebookresearch/mbrl-lib</a>: Library for Model Based RL . Contribute to facebookresearch/mbrl-lib development by creating an account on GitHub.</li><li><a href="https://yaml.org/spec/1.2.2/#24-tags">YAML Ain’t Markup Language (YAML™) revision 1.2.2</a>: no description found
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1232382705388228680)** (1 messages): 

- **LLM Paper Club Dives into Time Series with TimeGPT**: Tomorrow's US paper club is [discussing TimeGPT](https://lu.ma/y7olehof), a paper on time series, featuring the authors and <@556359685306056721>. Remember to sign up for notifications and that the event will take place on Zoom, not Discord.
- **Stay Up-to-date with Latent Space Events**: [Latent.Space](http://Latent.Space) encourages users to click the RSS logo above the calendar on the right to add events to their calendar. "Add iCal Subscription" will appear on hover for easy event tracking.

**Link mentioned**: <a href="https://lu.ma/y7olehof">LLM Paper Club (TimeGPT paper WITH AUTHORS) · Zoom · Luma</a>: This week @Vibhu hasa invited Nixtla to cover TimeGPT: https://arxiv.org/abs/2310.03589 Also submit and vote for our next paper:…

  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/)** (1 messages): 

alan_95125: Selfcheck, both the Evauator & Evaluatee models are the same by definition.
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1231159237258776597)** (24 messages🔥): 

- **Llama 3 70b Recommended Over 8b**: One user indicated a preference for using **llama 3 70b**, as they have not been able to get the 8b version working on Llamafile. The Q2 weights for 70b were mentioned to be only 26GB.
- **Quantization Quirks**: A user reported issues with the Q2 variant of llama model on an M1 Pro system, resulting in garbled output. Another user noted the model's functionality in a **pure CPU mode**, albeit operating more slowly.
- **Android Ambitions Thwarted by Address Space**: Interest in running llamafile on Android was discussed, but it was explained that Android support isn't possible without a **47 bit address space**.
- **Redis Inventor Endorses Llamafile**: The creator of Redis shared a positive sentiment about the llama3 70b llamafile on Twitter, offering an endorsement that the Llamafile team celebrated.
- **Multimodal Port Management**: A user inquired about controlling what port a model runs on with the goal of simultaneously running multiple llamafile instances, and another user suggested using the `--port` flag to achieve this.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1089876418936180786/1089876419926032399/1224854113674592286">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/.devops/main-vulkan.Dockerfile">llama.cpp/.devops/main-vulkan.Dockerfile at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1232172647153008730)** (3 messages): 

- **4chan's Insight on Context Size**: A member mentioned an assertion from 4chan, suggesting that a certain AI has *had 32k context* the entire time, expressing surprise at this revelation.

- **Alpin's Take on Scaling**: The discussion includes a member summarizing Alpin's approach to scaling, talking about the use of *dynamic ntk* and *linear scaling* without the use of rope but maintaining that it should still be effective.

- **Matt's Config for Long Context AI**: The member shared a link to **Matt's 16k configuration** for the Llama model on Hugging Face, providing a JSON snippet with parameters like "max_position_embeddings": 16000 and "model_type": "llama". Access the file [here](https://huggingface.co/mattshumer/Llama-3-8B-16K/blob/main/config.json).

**Link mentioned**: <a href="https://huggingface.co/mattshumer/Llama-3-8B-16K/blob/main/config.json">config.json · mattshumer/Llama-3-8B-16K at main</a>: no description found

  

---


**Skunkworks AI ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/)** (1 messages): 

noob_master169: OCR dataset for less popular languages? mainly looking for doc type data
  

---


**Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/1232163104578863195)** (10 messages🔥): 

- **Seeking Simplification of Medical Knowledge**: A physician scientist inquired about fine-tuning an LLM to explain complex genetic and medical information at a 6th grade reading level. They expressed interest in adapting the explanation process for patients with lower educational backgrounds.
- **Agentic System Over Fine-Tuning**: It was suggested that rather than immediately fine-tuning a model, one could develop an agentic system that manages tasks through specialized stages, likening it to a corporate workflow.
- **From Medical Jargon to Layman's Terms**: The advice further detailed a multi-stage approach: comprehend medical lab results using existing models enhanced by medical ontologies, summarize them at a professional level, then translate the summary to a 6th-grade level.
- **Data-Driven Fine-Tuning Direction**: The final recommendation was to utilize the strongest available model to collect inputs and outputs, which, after sufficient time in production, could lead to enough data to perform targeted fine-tuning for the specific task of simplifying medical information directly.
- **Surprised by Agent Efficiency**: The inquirer was surprised by the suggestion of using an agent for the task, having previously assumed that fine-tuning would be necessary to achieve the desired simplification of medical content.
  

---


**Skunkworks AI ▷ #[moe-main](https://discord.com/channels/1131084849432768614/1139310171076706464/)** (1 messages): 

getovahit: Enjoyed this! Thanks for sharing your work
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1231679501134467193)** (3 messages): 

- **Excitement Over Meta AI's 'Imagine'**: A member expressed enthusiasm about **Meta AI's 'Imagine'**, calling it *insane*.
- **Call for Imagine Examples**: Following the excitement about **Meta AI's Imagine**, another member asked for examples to illustrate the capabilities or outcomes.
- **In Search of Dev Tools for LLMs**: A member sought recommendations on **development tools** that are popular or preferred for working with **Large Language Models (LLMs)**.
  

---


**LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/1232022132842561688)** (5 messages): 

- **Struggle with Azure OpenAI Latency**: A member described experiencing significant latency issues with Azure's OpenAI, with some requests taking up to 20 minutes.
- **Rate Limit Woes**: Another sentiment expressed frustration at being constantly rate-limited on Azure, with just two requests within 15 seconds triggering the backoff strategy.
- **Possible Azure Latency Culprit**: A member pointed out that Azure's latency issues could have been specific to today due to reported service problems.
- **Tracking API Response Times**: The shared link from [GPT for Work](https://gptforwork.com/tools/openai-api-and-other-llm-apis-response-time-tracker) provides real-time tracking of API response times of major large language models, including OpenAI and Azure OpenAI, with suggestions for how to potentially achieve a faster response time.

**Link mentioned**: <a href="https://gptforwork.com/tools/openai-api-and-other-llm-apis-response-time-tracker">OpenAI API and other LLM APIs response time tracker</a>: no description found

  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1232013872261758996)** (2 messages): 

- **Blueprint AI in Architecture**: A member shared that a major architecture firm is using AI as a 'preflight' tool to identify potential issues and code violations in architectural plans. However, the firm has not yet adopted AI for generating content during the blueprint phase.
- **Seeking AI for Blueprint Interpretation**: The discussion also touched on exploring AI models or approaches for interpreting blueprints, particularly focused on tracing ductwork in PDF plans. No specific models or solutions were provided in the conversation.
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1231997043082268783)** (2 messages): 

- **Llama 3 Makes a Grand Entry**: [Llama 3](https://llama.meta.com/llama3/) was released, showcasing impressive results by ranking joint 5th on the [LMSYS arena leaderboard](https://chat.lmsys.org/?leaderboard), right behind major players like Claude 3 Opus and some GPT-4 variants. This open-licensed model even has the capability to run on high-end laptops.

- **SimonW Unveils Tools for Llama 3**: Simon Willison introduces [LLM](https://llm.datasette.io/), a command-line tool and Python library that facilitates access to Llama 3 and many other models. His blog post details several ways to access Llama 3, both through hosted versions and on local hardware, [highlighted here](https://simonwillison.net/2024/Apr/22/llama-3/).

- **Request for Hackernews Summary Generator**: A member is asking for the latest version of a hackernews summary generator, which they recall seeing in the form of a bash script.

**Link mentioned**: <a href="https://simonwillison.net/2024/Apr/22/llama-3/">Options for accessing Llama 3 from the terminal using LLM</a>: Llama 3 was released on Thursday. Early indications are that it’s now the best available openly licensed model—Llama 3 70b Instruct has taken joint 5th place on the LMSYS arena …

  

---



**AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1232147938097233940)** (4 messages): 

- **Spam Alert in General Chat**: The channel had multiple spam messages promoting inappropriate content with a Discord invite link.
- **Curiosity about Jamba's Requirements**: A member inquired about the compatibility of **Jamba** with **LM Studio** and its operational requirements given that it boasts memory capacity akin to **Claude**.
- **Jamba Running Challenges Discussed**: There's a discussion on the difficulty of running **Jamba** due to high RAM requirements, with a mention that Google Colab didn't provide sufficient resources and attempts on Google Cloud were unsuccessful.

**Link mentioned**: <a href="https://discord.gg/kYyKmR6U">Join the NSFW // 18 🍑🍒 Discord Server!</a>: Check out the NSFW // 18 🍑🍒 community on Discord - hang out with 31716 other members and enjoy free voice and text chat.

  

---




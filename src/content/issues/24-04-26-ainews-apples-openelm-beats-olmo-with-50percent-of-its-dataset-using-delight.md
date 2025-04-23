---
id: 5491a042-7431-4c36-8ef2-8837bf3bce30
title: Apple's OpenELM beats OLMo with 50% of its dataset, using DeLighT
date: '2024-04-26T21:32:41.171695Z'
original_slug: ainews-apples-openelm-beats-olmo-with-50-of-its
description: >-
  Apple has released OpenELM, its first relatively open large language model
  featuring novel efficient architecture with layer-wise scaling, outperforming
  its predecessor OLMo despite using fewer training tokens. Meanwhile, Meta’s
  LLaMA 3 models have significantly expanded context lengths, with versions
  supporting over 160K and even 262K tokens, enabling advanced reasoning
  capabilities.
tags:
  - apple
  - openelm
  - large-language-model
  - layer-wise-scaling
  - llama
  - context-length
  - model-efficiency
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/24/2024-4/26/2024. We checked 7 subreddits and [**373** Twitters](https://twitter.com/i/lists/1585430245762441216) and **27** Discords (**395** channels, and **5502** messages) for you. Estimated reading time saved (at 200wpm): **599 minutes**.

[Apple's AI emergence](https://buttondown.email/ainews/archive/ainews-mm1-apples-first-large-multimodal-model/) continues apace ahead of WWDC. We've covered [OLMo](https://buttondown.email/ainews/archive/ainews-ai2-releases-olmo-the-4th-open-everything/) before, and it looks like OpenELM is Apple's first [actually open LLM](https://arxiv.org/abs/2404.14619) ([weights](https://huggingface.co/apple/OpenELM), [code](https://github.com/apple/corenet)) release sharing some novel research in the efficient architecture direction.

 ![image.png](https://assets.buttondown.email/images/3bd4b772-df2f-46b7-8318-2cc230b7eb46.png?w=960&fit=max) 

It's not *totally* open, but it's pretty open. As [Sebastian Raschka put it](https://twitter.com/rasbt/status/1783480053847736713/photo/1):

> Let's start with the most interesting tidbits:
>
> - OpenELM comes in 4 relatively small and convenient sizes: 270M, 450M, 1.1B, and 3B
> - OpenELM performs slightly better than OLMo even though it's trained on 2x fewer tokens
> - The main architecture tweak is a layer-wise scaling strategy

But:

> "Sharing details is not the same as explaining them, which is what research papers were aimed to do when I was a graduate student. For instance, they sampled a relatively small subset of 1.8T tokens from various publicly available datasets (RefinedWeb, RedPajama, The PILE, and Dolma). This subset was 2x smaller than Dolma, which was used for training OLMo. What was the rationale for this subsampling, and what were the criteria?"

 ![image.png](https://assets.buttondown.email/images/5a0bcc71-6f46-41a3-a34b-6efff203c64d.png?w=960&fit=max) 

The layer-wise scaling comes from [DeLight](https://arxiv.org/abs/2008.00623), a 2021 paper deepening the standard attention mechanism 2.5-5x in number of layers but matching 2-3x larger models by parameter count. These seem paradoxical but the authors described the main trick of varying the depth between the input and the output, rather than uniform:

 ![image.png](https://assets.buttondown.email/images/64a3ecf6-fbca-4816-9233-f4100454aca8.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/a70b5ba1-00bb-482d-a4a4-f1027eec0266.png?w=960&fit=max) 

---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**LLaMA Developments**

- **LLaMA 3 increases context to 160K+ tokens**: In /r/LocalLLaMA, LLaMA 3 increases context length to [**over 160K tokens while maintaining perfect recall**](https://www.reddit.com/r/LocalLLaMA/comments/1ccqmjz/llama_3_now_with_160k_context/). Commenters note this is impressive but will require significant consumer hardware to run locally at good speeds. Meta's Llama 3 has been downloaded over 1.2M times, with over 600 derivative models on Hugging Face.
- **First LLama-3 8B-Instruct model with 262K context released**: In /r/LocalLLaMA, the first LLama-3 8B-Instruct model with [**over 262K context length is released on Hugging Face**](https://www.reddit.com/r/LocalLLaMA/comments/1cd4yim/llama38binstruct_with_a_262k_context_length/), enabling advanced reasoning beyond simple prompts.
- **Llama 3 70B outperforms 8B model**: In /r/LocalLLaMA, comparisons show the [**quantized Llama 3 70B IQ2_XS outperforms the uncompressed Llama 3 8B f16 model**](https://www.reddit.com/r/LocalLLaMA/comments/1cda0fv/llama_3_8b_f16_vs_llama_3_70b_q2/). The 70B IQ3_XS version is found to be best for 32GB VRAM users.
- **New paper compares AI alignment approaches**: In /r/LocalLLaMA, a new paper compares DPO to other alignment approaches, finding [**KTO performs best on most benchmarks and alignment methods are sensitive to training data volume**](https://www.reddit.com/r/LocalLLaMA/comments/1ccz84a/insights_into_alignment_dpo_and_its_variants/).

**AI Ethics & Regulation**

- **Eric Schmidt warns about risks of open-source AI**: In /r/singularity, former Google CEO Eric Schmidt cautions that [**open-source AI models give risky capabilities to bad actors and China**](https://www.reddit.com/r/singularity/comments/1ccyqkr/former_google_ceo_eric_schmidt_warns_that_open/). Many see this as an attempt by large tech companies to stifle competition, noting China likely has the capability to develop powerful models without relying on open-source.
- **U.S. proposal aims to end anonymous cloud usage**: In /r/singularity, a [**U.S. proposal seeks to implement "Know Your Customer" requirements to end anonymous cloud usage**](https://www.reddit.com/r/singularity/comments/1ccr2ub/us_know_your_customer_proposal_will_put_an_end_to/).
- **Baltimore coach allegedly used AI for defamation**: In /r/OpenAI, a Baltimore coach allegedly [**used AI voice cloning to attempt to get a high school principal fired by generating fake racist audio**](https://www.reddit.com/r/OpenAI/comments/1cd5h9c/baltimore_high_school_athletic_director_used_ai/).

**Hardware Developments**

- **TSMC unveils 1.6nm process node**: In /r/singularity, TSMC announces a [**1.6nm process node with backside power delivery**](https://www.reddit.com/r/singularity/comments/1ccr4hy/tsmc_unveils_16nm_process_technology_with/), enabling continued exponential hardware progress over the next few years.
- **Ultra-thin solar cells enable self-charging drones**: In /r/singularity, German researchers develop [**ultra-thin, flexible solar cells that allow small drones to self-charge during operation**](https://www.reddit.com/r/singularity/comments/1ccr6aq/german_researchers_have_developed_a_solar_cell/).
- **Micron secures $6.1B in CHIPS Act funding**: In /r/singularity, Micron secures [**$6.1 billion in CHIPS Act funding to build semiconductor manufacturing facilities in New York and Idaho**](https://www.reddit.com/r/singularity/comments/1cd0s5k/micron_set_to_receive_61b_in_chips_act_funding_to/).

**Memes & Humor**

- **AI assistant confidently asserts flat Earth**: In /r/singularity, a humorous image depicts an [**AI assistant confidently asserting that the Earth is flat**](https://www.reddit.com/r/singularity/comments/1ccqhzv/chat_is_this_real/), sparking jokes about needing AI capable of believing absurdities or that humanity has its best interests at heart.

---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

Here is a summary of the key topics and insights from the provided tweets:

**Meta Llama 3 Release and Impact**

- **Rapid Adoption**: In the week since release, Llama 3 models have been downloaded over 1.2M times with 600+ derivative models on Hugging Face, showing exciting early impact. ([@AIatMeta](https://twitter.com/AIatMeta/status/1783602908845748685))
- **Training Optimizations**: Meta is moving fast on optimizations, with Llama 3 70B training 18% faster and Llama 3 8B training 20% faster. ([@svpino](https://twitter.com/svpino/status/1783888989025431933)) 
- **Context Extension**: The community extended Llama 3 8B's context from 8k to nearly 100k tokens by combining PoSE, continued pre-training, and RoPE scaling. ([@winglian](https://twitter.com/winglian/status/1783842736833016289))
- **Inference Acceleration**: Colossal-Inference now supports Llama 3 inference acceleration, enhancing efficiency by ~20% for 8B and 70B models. ([@omarsar0](https://twitter.com/omarsar0/status/1783895931043111088))
- **Benchmark Performance**: Llama 3 70B is tied for 1st place for English queries on the LMSYS leaderboard. ([@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783570318230978783))

**Phi-3 Model Release and Reception** 

- **Overfitting Benchmarks**: Some argue Phi-3 overfits public benchmarks but underperforms in practical usage compared to models like Llama-3 8B. ([@svpino](https://twitter.com/svpino/status/1783556635543339310), [@abacaj](https://twitter.com/abacaj/status/1783898711623352686))
- **Unexpected Behavior**: As a fundamentally different model, Phi-3 can exhibit surprising results, both good and bad. ([@srush_nlp](https://twitter.com/SebastienBubeck/status/1783885843943616524))

**Extending LLM Context Windows**

- **PoSE Technique**: The Positional Skip-wisE (PoSE) method simulates long inputs during training to increase context length, powering Llama 3's extension to 128k tokens. ([@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783574428858696161)) 
- **Axolotl and Gradient AI**: Tools like Axolotl and approaches from Gradient AI are enabling context extension for Llama and other models to 160k+ tokens. ([@winglian](https://twitter.com/winglian/status/1783469196011016696), [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783736130321408011))

**Cohere Toolkit Release**

- **Enterprise Focus**: Cohere released a toolkit to accelerate LLM deployment in enterprises, targeting secure RAG with private data and local code interpreters. ([@aidangomez](https://twitter.com/aidangomez/status/1783533461401227563))
- **Flexible Deployment**: The toolkit's components can be deployed to any cloud and reused to build applications. ([@aidangomez](https://twitter.com/aidangomez/status/1783533465960378561), [@aidangomez](https://twitter.com/aidangomez/status/1783533471777935433))

**OpenAI Employee Suspension and GPT-5 Speculation**

- **Sentience Claims**: An OpenAI employee who claimed GPT-5 is sentient has been suspended from Twitter. ([@bindureddy](https://twitter.com/bindureddy/status/1783847600824995850))
- **Hype Generation**: OpenAI is seen as a hype-creation engine around AGI and AI sentience claims, even as competitors match GPT-4 at lower costs. ([@bindureddy](https://twitter.com/bindureddy/status/1783852748636905716))
- **Agent Capabilities**: Some believe GPT-5 will be an "agent GPT" based on the performance boost from agent infrastructure on top of language models. ([@OfirPress](https://twitter.com/OfirPress/status/1783870394581074110))

**Other Noteworthy Topics**

- Concerns about the AI summit board's lack of diverse representation to address power concentration risks. ([@ClementDelangue](https://twitter.com/ClementDelangue/status/1783882237764633052))
- OpenAI and Moderna's partnership as a positive sign of traditional businesses adopting generative AI. ([@gdb](https://twitter.com/gdb/status/1783529202974687527), [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1783533728846827681)) 
- Apple's open-sourced on-device language models showing poor performance but providing useful architecture and training details. ([@bindureddy](https://twitter.com/bindureddy/status/1783635037365436462), [@rasbt](https://twitter.com/rasbt/status/1783480053847736713))

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **Extending LLM Context Lengths**
   - **Llama 3 Performance and Context Length Innovations**: Discussions centered around **Llama 3's capabilities**, with some expressing mixed opinions on its code recall and configuration compared to **GPT-4**. However, innovations in extending Llama 3's **context length to 96k tokens for the 8B model** using techniques like **PoSE (Positional Skip-wisE)** and continued pre-training with 300M tokens generated excitement, as detailed in this [tweet thread](https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg).
   - The [EasyContext project](https://github.com/jzhang38/EasyContext) aims to extrapolate LLM context lengths to **1 million tokens** with minimal hardware requirements.

2. **Optimizing LLM Training and Deployment**
   - [Nvidia's Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction) is utilized for **kernel profiling** to optimize CUDA code for LLM training.
   - **Finetuning LLMs for Domain-Specific Gains**: Interest grew in **finetuning large language models** for domain-specific improvements, with examples like **[Meditron](https://arxiv.org/abs/2311.16079)** for medical applications. Discussions also covered **data synthesis** strategies using tools like **[Argilla's Distilabel](https://github.com/argilla-io/distilabel)**, and the challenges of multi-document, long-context finetuning. Cost-performance tradeoffs were debated, such as spending [$2,368 for 4 epochs vs $41,440 for 50 epochs](https://discord.com/channels/1053877538025386074/1154120232051408927/1232958591955112028) with potentially minor gains.
   - PyTorch introduces [Torchtitan](https://github.com/pytorch/torchtitan), a library dedicated to aiding LLM training from scratch.
   - The [Mixture of Depths paper](https://paper-club.ivanleo.com/papers/mixture-of-depths) proposes accelerating transformer training using a modified MoE routing mechanism.
  - **CUDA Optimization Deep Dives**: CUDA developers dug into kernel profiling with tools like **[NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction)**, discussed **memory coalescing** and **burst sizes** around 128 bytes, and debated the efficiency of **low-bit quantization** methods. Conversations also covered **flash attention compatibility** issues with PyTorch 2.3.0, and the implications of PyTorch AO supporting **custom CUDA extensions** for performance tuning.

3. **Open-Source LLM Ecosystem Expansion**
  - **Apple's Surprise Entry into Open-Source Models**: **Apple's release of [OpenELM](https://huggingface.co/apple/OpenELM)**, a family of efficient open-source language models ranging from 270M to 3B parameters, caught the AI community by surprise. The move marked a shift from Apple's traditionally proprietary approach, with the 270M model quickly gaining attention on Hugging Face.
   - [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B) is introduced, leveraging Mistral for pretrained medical LLMs.
   - Mozilla's [llamafile project](https://hacks.mozilla.org/2024/04/llamafiles-progress-four-months-in/) enables distributing and running LLMs locally with high performance.
   - Dify emerges as an [open-source LLM app development platform](https://github.com/langgenius/dify?tab=readme-ov-file) combining AI workflows and model management.

4. **Evaluating and Benchmarking LLMs**
   - On the [Judgemark benchmark](https://eqbench.com/judgemark.html), **Llama-3-70b** shows promise for fine-tuning **disco-judge** applications.
   - Discussions around the effectiveness of **validation loss** as a performance indicator for LLMs.
   - The [Low-Cost Language Models survey](https://arxiv.org/abs/2404.11160) evaluates CPU-friendly LLMs on Python code generation tasks.
   - Debates on the transparency of **Nightshade's** autoencoder capabilities and the need for publishing findings openly.





---



# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Fine-Tuning Tribulations and Triumphs**: Engineers shared their challenges with fine-tuning models, involving 4-bit quantization, VRAM requirements exceeding 48GB, and token index sequence length errors. Solutions included batch size revision, enabling [pack](https://github.com/unslothai/unsloth#-finetune-for-free) functionality, and considering alternative testing environments like **Aphrodite Engine** or **llama.cpp**.

- **Gossip or Gone?**: Speculation is rampant about the dissolution of the **WizardLM** team after **Qingfeng Sun's staff page** redirect. Contrasting sentiments were shared, from salvaging **WizardLM datasets** to **showcase** sessions where **Meta's LlaMA-3 models** (including an 8B and 70B version) were cited as top performers in their classes.

- **From Cold Storage to Hot Topics**: A member proudly announced an [open-source release](https://github.com/oKatanaaa/kolibrify) of **Kolibrify**, a curriculum training tool for instruction-following LLMs. On a technical note, the community discussed **Triton** dependencies, errors with "Quantization failed," and **gguf model** testing strategies, reaching a consensus on best practices for **fine-tuning** and deployment options.

- **Pragmatic Pruning Progress**: Insights were shared about a project on iterative context length increase for models using a **[triton laser merge trainer](https://github.com/l4b4r4b4b4/trl/tree/evol_laser_merge_trainer)** that operates during evaluation. This method, signaled as innovative due to no reinitialization requirements, could provide a pathway for enhanced model usability without system overhaul. 

- **Unsloth's Milestones and Resources**: Unsloth AI marked a significant milestone with 500k monthly downloads of their fine-tuning framework on Hugging Face and promoted the sharing of **exact match** GGUF models despite potential redundancy. Emphasis was also on directing users to **[Colab notebooks](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp)** for effective fine-tuning strategies.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Siri Gets a Brainy Buddy**: Perplexity AI Discord chatbot introduces an exclusive auditory feature for **iOS users** that reads answers to any posed question.

- **Opus Limit Outcry**: Frustration arises within the community concerning the new 50-query daily limit on Claude 3 Opus interactions, while still, **Perplexity chatbot supports Opus** despite these caps.

- **API Adoption Anxieties**: AI Engineers are discussing integration issues with the Perplexity API, such as outdated responses and a lack of GPT-4 support; a user also sought advice on **optimal hyperparameters** for the `llama-3-70b-instruct` model.

- **A Game of Models**: The community is buzzing with anticipation around Google's Gemini model, and its potential impact on the AI landscape, while noting GPT-5 will have to bring exceptional innovations to keep up with the competition.

- **Crystal Ball for Net Neutrality**: A linked article prompts discussions on the FCC's reestablishment of Net Neutrality, with implications for the **AI Boom's** future being pondered by community members.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**CUDA Collective Comes Together**: Members focused on honing their skills with **CUDA** through optimizing various kernels and algorithms, including matrix multiplication and flash attention. Threads spanned from leveraging the [NVIDIA Nsight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction) for kernel profiling to debate on the efficiency of low-bit quantization methods.

**PyTorch Tangles with Compatibility and Extensions**: A snag was hit with **flash-attn compatibility** in **PyTorch 2.3.0**, resulting in an `undefined symbol` error, which participants hoped to see rectified promptly. PyTorch AO ignited enthusiasm by [supporting custom CUDA extensions](https://github.com/pytorch/ao/pull/135), facilitating performance tuning using `torch.compile`.

**Greener Code with C++**: An announcement about a bonus talk from the **NVIDIA C++ team** on converting `llm.c` to `llm.cpp` teased opportunities for clearer, faster code.

**The Matrix of Memory and Models**: Discussions delved deep into finer points of CUDA best practices, contemplating **burst sizes** for memory coalescing around **128 bytes** as explored in **Chapter 6, section 3.d** of the CUDA guide, and toying with the concept of reducing overhead in packed operations.

**Recording Rendezvous**: Volunteers stepped up for screen recording with detailed, actionable advice and [Existential Audio - BlackHole](https://existential.audio/blackhole/download/?code=681349920) for lossless sound capture, highlighting the careful nuances needed for a refined technical setup.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPU Offloads to AMD OpenCL**: A technical hiccup with **GPU Offloading** was resolved by switching the GPU type to **AMD Open CL**, demonstrating a simple fix can sidestep performance issues.
- **Mixed News on Updates and Performance**: Upgrade issues cropped up in LM Studio with **version 0.2.21**, causing previous setups running **phi-3 mini models** to malfunction, while other users are experimenting with using **Version 2.20** and facing GPU usage spikes without successful model loading. Users are actively troubleshooting, including submitting requests for screenshots for better diagnostics.
- **LM Studio Turns Chat into Document Dynamo**: Enthusiastic discussions around improving **LM Studio's chat feature** have led to embedding document retrieval using **Retriever-Augmented Generation (RAG)** and tweaking GPU settings for better resource utilization.
- **Tackling AI with Graphical Might**: The community is sharing insights into optimal hardware setups and potential performance boosts anticipated from Nvidia Tesla equipment when using AI models, indicating a strong interest in the best equipment for AI model hosting.
- **AMD's ROCm Under the Microscope**: The use of **AMD's ROCm tech preview** has shown promise with certain setups, achieving a notable 30t/s on an eGPU system, although compatibility snags underscore the importance of checking GPU support against the ROCm documentation.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Pushing The Envelope on Model Context Limits**: Llama 3 models are breaking context barriers, with one variant reaching a **96k context for the 8B model** using PoSE and continued pre-training with 300M tokens. The efficacy of Positional Skip-wisE (PoSE) and **RoPE scaling** were key topics, with a [paper on PoSE's context window extension](https://openreview.net/forum?id=3Z1gxuAQrA) and discussions on fine-tuning RoPE base during fine-tuning for lengthier contexts mentioned.

**LLM Performance and Cost Discussions Engage Community**: Engineers expressed skepticism about validation loss as a performance indicator and shared a cost comparison of training epochs, highlighting a case where four epochs cost $2,368 versus $41,440 for fifty epochs with minor performance gains. Another engineer is considering combining several 8B models into a mixture of experts based on **Gemma MoE** and speculated on potential enhancements using **DPO/ORPO techniques**.

**The Saga of Repository Archival**: Concerns were voiced about the sudden disappearance of Microsoft’s WizardLM repo, sparking a debate on the importance of archiving, especially in light of Microsoft's investment in OpenAI. Participants underscored the need for backups, drawing from instances such as the recent reveal of **WizardLM-2**, accessible on [Hugging Face](https://huggingface.co/collections/microsoft/wizardlm-2-661d403f71e6c8257dbd598a) and [GitHub](https://github.com/victorsungo/WizardLM/tree/main/WizardLM-2).

**Synthetic Data Generation: A One-Stop Shop**: *Argilla’s Distilabel* was recommended for creating **diverse synthetic data**, with practical examples and repositories such as the [distilabel-workbench](https://github.com/argilla-io/distilabel-workbench) illustrating its applications. The conversation spanned single document data synthesis, multi-document challenges, and strategies for extended contexts in language models.

**Simulated World Engagements Rouse Curiosity**: Websim’s capabilities to simulate CLI commands and full web pages have captivated users, with example simulations shared, such as the **EVA AI interaction profile** on [Websim](https://websim.ai/c/p3pZvmAYbsRT2hzBz). Speculations on the revival of World-Sim operated in parallel, and members looked forward to its reintroduction with a "pay-for-tokens" model.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Apple's Open Source Pivot with OpenELM**: Apple has released **OpenELM**, a family of efficient language models now available on Hugging Face, scaling from 270M to 3B parameters, marking their surprising shift towards open-source initiatives. Details about the models are [on Hugging Face](https://huggingface.co/apple/OpenELM).

- **Conversations Surrounding AI Sentience and Temporal Awareness**: The community engaged in deep discussions emphasizing the difference between **sentience**—potentially linked to emotions and motivations—and **consciousness**—associated with knowledge acquisition. A parallel discussion pondered if intelligence and temporal awareness in AI are inherently discrete concepts, influencing our understanding of neural network identity and experiential dimension.

- **AI Voice Assistant Tech Talk**: AI enthusiasts compared notes on **OpenWakeWords** for homegrown voice assistant development and **Gemini**'s promise as a Google Assistant rival. Technical challenges highlighted include the intricacies of interrupt AI speech and preferences for push-to-talk versus voice activation.

- **Rate Limit Riddles with Custom GPT Usage**: Users sought clarity on **GPT-4's usage caps** especially when recalling large documents and shared tips on navigating the 3-hour rolling cap. The community is exploring the thresholds of rate limiting, particularly when employing custom GPT tools.

- **Prompt Engineering Prowess & LLM Emergent Abilities**: There's a focus on strategic prompt crafting for specific tasks such as developing GPT-based coding for **Arma 3's SQF language**. Fascination arises with **emergent behaviors** in LLMs, referring to phases of complexity leading to qualitative behavioral changes, exploring parallels to the concept of *More Is Different* in prompt engineering contexts.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**AI Rollout Must Be Crystal Clear**: Valve's new **content policy** requires developers to disclose AI usage on **Steam**, particularly highlighting the need for transparency around live-generated AI content and mechanisms that ensure responsible deployment.

**Copyright Quandary in Content Creation**: Conversations bubbled up over the legal complexities when generating content with public models such as **Stable Diffusion**; there's a necessity to navigate copyright challenges, especially on platforms with rigorous copyright enforcement like **Steam**.

**Art Imitates Life or... Itself?**: An inquiry raised by **Customluke** on how to create a model or a Lora to replicate their art style using **Stable Diffusion** sparked suggestions, with tools like **dreambooth** and **kohya_ss** surfaced for model and Lora creation respectively.

**Selecting the Better Suited AI Flavor**: A vocal group of users find **SD 1.5** superior to **SDXL** for their needs, citing sharper results and better training process, evidence that the choice of AI model significantly impacts outcome quality.

**Polishing Image Generation**: Tips were shared for improving image generation results, recommending alternatives such as **Forge** and **epicrealismXL** to enhance the output for those dissatisfied with the image quality from models like **ComfyUI**.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **BioMistral Launch for Medical LLMs**: [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B), a new set of pretrained language models for medical applications, has been introduced, leveraging the capabilities of the foundational Mistral model.

- **Nvidia's Geopolitical Adaptation**: To navigate US export controls, Nvidia has unveiled the RTX 4090D, a China-compliant GPU with reduced power consumption and CUDA cores, detailed in reports from [The Verge](https://www.theverge.com/2023/12/29/24018799/nvidia-4090d-china-slower-us-sanctions) and [Videocardz](https://videocardz.com/newz/nvidia-geforce-rtx-4090-with-blower-type-cooler-is-now-on-sale-in-china).

- **Text to Image Model Fine-Tuning Discussed**: Queries about optimizing text to image models led to suggestions involving the [Hugging Face diffusers repository](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image).

- **Gradio Interface for ConversationalRetrievalChain**: Integration of ConversationalRetrievalChain with Gradio is in the works, with community efforts to include personalized PDFs and discussion regarding interface customization.

- **Improved Image Generation and AI Insights in Portuguese**: New developments include an app at [Collate.one](https://collate.one/newsletter) for digesting read-later content, advancements in generating high-def images in seconds at [this space](https://huggingface.co/spaces/KingNish/Instant-Image), and [Brazilian Portuguese translations](https://www.youtube.com/watch?v=A9qPlYVeiOs) of AI community highlights.

- **Quantization and Efficiency**: There's active exploration on quantization techniques to maximize model efficiency on VRAM-limited systems, with preferences leaning toward Q4 or Q5 levels for a balance between performance and resource management.

- **Table-Vision Models and COCO Dataset Clarification**: There's a request for recommendations on vision models adept at table-based question-answering, and security concerns raised regarding the hosting of the official COCO datasets via an HTTP connection.

- **Call for Code-Centric Resources and TLM v1.0**: The engineering community is seeking more tools with direct code links, as exemplified by [awesome-conformal-prediction](https://github.com/valeman/awesome-conformal-prediction), and the launch of v1.0 of the Trustworthy Language Model (TLM), introducing a confidence score feature, is celebrated with a [playground](https://tlm.cleanlab.ai/) and [tutorial](https://help.cleanlab.ai/tutorials/tlm/).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Parallel Ponderings Pose No Problems**: Engineers highlighted that some **model architectures**, specifically PaLM, employ **parallel attention and FFN (feedforward neural networks)**, deviating from the series perception some papers present.

- **Data Digestion Detailing**: The **Pile dataset's hash values** were shared, offering a reference for those looking to utilize the dataset in various JSON files, an aid found on [EleutherAI's hash list](https://www.eleuther.ai/hashes).

- **Thinking Inside the Sliding Window**: Dialogue on **transformers** considered **sliding window attention** and effective receptive fields, analogizing them to convolutional mechanisms and their impact on attention's focus.

- **Layer Learning Ladders Lengthen Leeway**: Discussions about improving transformers' handling of **lengthier sequence lengths** touched upon strategies like integrating RNN-type layers or employing dilated windows within the architecture.

- **PyTorch's New Power Player**: A new **PyTorch library, torchtitan**, was introduced via a [GitHub link](https://github.com/pytorch/torchtitan), promising to ease the journey of training larger models.

- **Linear Logic Illuminates Inference**: The mechanics of **linear attention** were unpacked, illustrating its sequence-length linearity and constant memory footprint, essential insights for future model optimization.

- **Performance Parity Presumption**: One engineer reported that the **phi-3-mini-128k** might match the **Llama-3-8B**, triggering a talk on the influences of pre-training data on model benchmarking and baselines.

- **Delta Decision's Dual Nature**: The possibility of **delta rule linear attention** enabling more structured yet less parallelizable operations stirred a comparison debate, supported by a [MastifestAI blog post](https://manifestai.com/blogposts/faster-after-all/).

- **Testing Through a Tiny Lens**: Members cast doubt on "needle in the haystack" tests for long-context language models, advocating for real-world application as a more robust performance indicator.

- **Prompt Loss Ponderings**: The group questioned the systemic study of masking user prompt loss during supervised fine-tuning (SFT), noting a research gap despite its frequent use in language model training.

- **Five is the GSM8K Magic Number**: There was a consensus suggesting that using *5* few-shot examples is the appropriate alignment with the **Hugging Face leaderboard** criteria for **GSM8K**.

- **VLLM Version Vivisection**: Dialogue identified **Data Parallel (DP)** as a stumbling block in updating **VLLM** to its latest avatar, while **Tensor Parallel (TP)** appeared a smoother path.

- **Calling Coders to Contribute**: The lm-evaluation-harness appeared to be missing a `register_filter` function, leading to a call for contributors to submit a PR to bolster the utility.

- **Brier Score Brain Twister**: An anomaly within the **ARC evaluation** data led to a suggestion that the Brier score function be refitted to ensure error-free assessments regardless of data inconsistencies.

- **Template Tête-à-Tête**: Interest was piqued regarding the status of a chat templating branch in *Hailey's branch*, last updated a while ago, sparking an inquiry into the advancement of this functionality.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Mixtral Muddle**: A provider of **Mixtral 8x7b** faced an issue of sending blank responses, leading to their temporary removal from OpenRouter. Auto-detection methods for such failures are under consideration.

**Soliloquy's Subscription Surprise**: The **Soliloquy 8B** model transitioned to a paid service, charging **$0.1 per 1M tokens**. Further information and discussions are available at [Soliloquy 8B](https://openrouter.ai/models/lynn/soliloquy-l3).

**DBRX AI Achieves AI Astonishment**: Fprime-ai announced a significant advancement with their **DBRX AI** on LinkedIn, sparking interest and discussions in the community. The LinkedIn announcement can be read [here](https://www.linkedin.com/posts/fprime-ai_fprimeailabs-dbrx-ai-activity-7189599191201980417-Te5d).

**Creative Model Melee**: Community members argued about the best open-source model for role-play creativity, with **WizardLM2 8x22B** and **Mixtral 8x22B** emerging as top contenders due to their creative capabilities.

**The Great GPT-4 Turbo Debate**: Microsoft's influence on the **Wizard LM** project incited a heated debate, leading to a deep dive into the incidence, performance, and sustainability of models like GPT-4, Llama 3, and WizardLM. Resources shared include an [incident summary](https://rocky-muscle-755.notion.site/) and a miscellaneous [OpenRouter model list](https://openrouter.ai/models?q=free).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Create-llama Simplifies RAG Setup**: The **create-llama v0.1** release brings new support for **@ollama** and vector database integrations, making it easier to deploy RAG applications with llama3 and phi3 models, as detailed in their [announcement tweet](https://twitter.com/llama_index/status/1783528887726817653).

**LlamaParse Touted in Hands-on Tutorial and Webinar**: A hands-on tutorial showcases how **LlamaParse**, **@JinaAI_ embeddings**, **@qdrant_engine vector storage**, and **Mixtral 8x7b** can be used to create sophisticated RAG applications, available [here](https://twitter.com/llama_index/status/1783601807903863184), while KX Systems hosts a webinar to unlock complex document parsing capabilities with **LlamaParse** (details in [this tweet](https://twitter.com/llama_index/status/1783622871614664990)).

**AWS Joins Forces with LlamaIndex for Developer Workshop**: AWS collaborates with **@llama_index** to provide a workshop focusing on LLM app development, integrating AWS services and LlamaParse; more details can be found [here](https://twitter.com/llama_index/status/1783877951278432733).

**Deep Dive into Advanced RAG Systems**: The community engaged in robust discussions on improving RAG systems and shared a video on advanced setup techniques, addressing everything from sentence-window retrieval to integrating structured Pydantic output ([Lesson on Advanced RAG](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/5/auto-merging-retrieval)).

**Local LLM Deployment Strategies Discussed**: There was active dialogue on employing local LLM setups to circumvent reliance on external APIs, with guidance provided in the official **LlamaIndex documentation** ([Starter Example with Local LLM](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)), showcasing strategies for resolving import errors and proper package installation.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**Llama 3's Mixed Reception**: Community feedback on **Llama 3** is divided, with some highlighting its inadequate code recall abilities compared to expectations set by GPT-4, while others speculate the potential for configuration enhancements to bridge the performance gap.

**Know Your Customer Cloud Conundrum**: The proposed U.S. "Know Your Customer" policies for cloud services spark concern and discussion, emphasizing the necessity for community input on the [Federal Register](https://www.federalregister.gov/documents/2024/01/29/2024-01580/taking-additional-steps-to-address-the-national-emergency-with-respect-to-significant-malicious) before the feedback window closes.

**Boost in AI Model Training Efficiency**: Innovations in vision model training are making waves with a *weakly supervised pre-training method* that races past traditional contrastive learning, achieving **2.7 times faster** training as elucidated in this [research](https://arxiv.org/abs/2404.15653). The approach shuns contrastive learning's heavy compute costs for a multilabel classification framework, yielding a performance on par with **CLIP** models.

**The VAST Landscape of Omni-Modality**: Enthusiasm is sighted for finetuning **VAST**, a Vision-Audio-Subtitle-Text Omni-Modality Foundation Model. The project indicates a stride towards omni-modality with the resources available at its [GitHub repository](https://github.com/txh-mercury/vast).

**Nightshade's Transparency Troubles**: The guild debates the effectiveness and transparency of **Nightshade** with a critical lens on autoencoder capabilities and reluctances in the publishing of potentially controversial findings.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Mac Muscle Meets Interpreter Might**: Open Interpreter's **New Computer Update** has significantly improved local functionality, particularly with **native Mac integrations**. The implementation allows users to control Mac's native applications using simple commands such as `interpreter --os`, as detailed in their [change log](https://changes.openinterpreter.com/log/ncu-ii).

**Eyes for AI**: Community members highlighted the **Moondream tiny vision language model**, providing resources like the [Img2TxtMoondream.py script](https://github.com/CodeAKrome/bootcupboard/blob/main/llm-img/Img2TxtMoondream.py). Discussions also featured **LLaVA**, a multimodal model hosted on [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.6-34b), which is grounded in the powerful **NousResearch/Nous-Hermes-2-Yi-34B** model.

**Loop Avoidance Lore**: Engineers have been swapping strategies to mitigate looping behavior in local models, considering solutions ranging from tweaking *temperature settings* and *prompt editing* to more complex *architectural changes*. An intriguing concept, the *frustration metric*, was introduced to tailor a model's responses when stuck in repetitive loops.

**Driving Dogs with Dialogue**: A member inquired about the prospect of leveraging **Open Interpreter** for commanding the **Unitree GO2 robodog**, sparking interest in possible interdisciplinary applications. Technical challenges, such as setting dummy API keys and resolving namespace conflicts with Pydantic, were also tackled with shared solutions.

**Firmware Finality**: The **Open Interpreter 0.2.5 New Computer Update** has officially graduated from beta, including the fresh enhancements mentioned earlier. A query about the update's beta status led to an affirmative response after a version check.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**CEO's Nod to a Member's Tweet**: A participant was excited about the *CEO of Hugging Face acknowledging their tweet*; network and recognition are alive in the community.

**Tech Giants Jump Into Fine-tuning**: With examples like **[Meditron](https://arxiv.org/abs/2311.16079)**, discussion on fine-tuning language models for specific uses is heating up, highlighting the promise for domain-specific improvements and hinting at an **upcoming paper** on continual pre-training.

**Trouble in Transformer Town**: An 'AttributeError' surfaced in **transformers 4.40.0**, tripping up a user, serving as a cautionary tale that even small updates can break workflows.

**Mixing Math with Models**: Despite some confusion, inquiries were made about integrating **zzero3** with **Fast Fourier Transform (fft)**; keep an eye out for this complex dance of algorithms.

**Optimizer Hunt Heats Up**: The **FSDP (Fully Sharded Data Parallel)** compatibility with optimizers remains a hot topic, with findings that **AdamW** and **SGD** are in the clear, while `paged_adamw_8bit` is not supporting FSDP offloading, leading to a quest for alternatives within the **OpenAccess-AI-Collective/axolotl** resources.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Upload Hiccups and Typographic Tangles**: Users in the **Cohere** guild tackled issues with the **Cohere Toolkit** on Azure, pointing to the paper clip icon for uploads; despite this, problems persisted with the upload functionality going undiscovered. The **Cohere typeface**'s licensing on GitHub provoked discussion; it's not under the MIT license and is slated for replacement.

**Model Usage Must-Knows**: Discussion clarified that Cohere's [Command+ models](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus) are available with open weight access but not for commercial use, and the training data is not shared.

**Search API Shift Suggestion**: The guild mulled over the potential switch from **Tavily** to the **Brave Search API** for integrating with the Cohere-Toolkit, citing potential benefits in speed, cost, and accuracy in retrieval.

**Toolkit Deployment Debates**: Deployment complexities of the Cohere Toolkit on Azure were deliberated, where selecting a model deployment option is crucial and the API key is not needed. Conversely, local addition of tools faced issues with PDF uploads and sqlite3 version compatibility.

**Critical Recall on 'Hit Piece'**: Heated discussions emerged over the criticism of a "hit piece" against *Cohere*, with dialogue focused on the responsibility of AI agents and their real-world actions. A push for critical accountability emerged, with members reinforcing the need to back up critiques with substantial claims.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Sprints Towards Version 1.0**: Tinygrad is gearing up for its 1.0 version, spotlighting an API that's nearing stability, and has a toolkit that includes [installation guidance](https://tinygrad.github.io/tinygrad/), a [MNIST tutorial](https://tinygrad.github.io/tinygrad/mnist/), and comprehensive [developer documentation](https://tinygrad.github.io/tinygrad/developer/).
  
- **Comma Begins Tinybox Testing with tinygrad**: George Hotz emphasized tinybox by comma as an exemplary testbed for tinygrad, with a focus maintained on software over hardware, while a potential tinybox 2 collaboration looms.

- **Crossing off Tenstorrent**: After evaluation, a partnership with Tenstorrent was eschewed due to inefficiencies in their hardware, leaving the door ajar for future collaboration if the cost-benefit analysis shifts favorably.

- **Sorting Through tinygrad's Quantile Function Challenge**: A dive into tinygrad's development revealed efforts to replicate `torch.quantile` for diffusion model sampling, a complex task necessitating a precise sorting algorithm within the framework.

- **AMMD's MES Offers Little to tinygrad**: AMD's Machine Environment Settings (MES) received a nod from Hotz for its detailed breakdown by Felix from AMD, but ultimately assessed as irrelevant to tinygrad's direction, with efforts focused on developing a PM4 backend instead.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Strong Performer: Hermes 2.5 Edges Out Hermes 2**: Enhanced with code instruction examples, **Hermes 2.5** demonstrates superior performance across various benchmarks when compared to **Hermes 2**.

**Security in the Limelight**: Amidst sweeping software and feature releases by Modular, addressing security loopholes becomes critical, emphasizing protection against supply chain attacks like the **XZ incident** and the trend of open-source code prevalence in software development forecasted to hit **96% by 2024**.

**Quantum Complexity Through A Geometric Lens**: Members discussed how the geometric concept of the **amplituhedron** could simplify quantum particle scattering amplitudes, with machine learning being suggested as a tool to decipher increased complexities in visualizing **quantum states** as systems scale.

**All About Mojo**: Dialogue around the **Mojo Programming Language** covered topics like assured memory cleanup by the OS, the variance between `def` and `fn` functions with examples found [here](https://docs.modular.com/mojo/manual/functions), and the handling of mixed data type lists via `Variant` that requires improvement.

**Moving Forward with Mojo**: ModularBot flagged an issue filed on GitHub about **Mojo**, urged members to use issues for better tracking of concerns, for instance, about `__copyinit__` semantics [via GitHub Gist](https://gist.github.com/modularbot/6aed759930420cd70f38795dbcb874fe), and reported a cleaner update in code with more insertions than deletions, achieving better efficiency.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**A Tricky Query for Anti-Trolling AI Design**: A user proposed designing an **anti-trolling AI** and sought suggestions on how the system could effectively target online bullies.

**Verbose SQL Headaches**: Participants shared experiences with open-source models like **Mistral** and **Llama3** generating overly verbose SQL responses and encountered an `OutputParserException`, with links to [structured output support](https://python.langchain.com/docs/integrations/chat/) and examples of invoking SQL Agents.

**RedisStore vs. Chat Memory**: The community clarified the difference between **stores** and **chat memory** in the context of LangChain integrations, emphasizing the specific use of `RedisStore` for key-value storage and **Redis Chat Message History** for session-based chat persistence.

**Techie Tutorial on Model Invocation**: There was a discussion on the correct syntax when integrating prompts into LangChain models via JavaScript, with recommendations for using `ChatPromptTemplate` and `pipe` methods for chaining prompts.

**Gemini 1.5 Access with a Caveat**: Users discussed the integration of **Gemini 1.5 Pro** with LangChain, highlighting that it necessitates `ChatVertexAI` instead of `ChatGoogleGenerativeAI` and requires configuring the `GOOGLE_APPLICATION_CREDENTIALS` environment variable for proper access.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Apple Bites the Open Source Apple**: Apple has stepped into the open source realm, releasing a suite of models with parameters ranging from 270M to 3B, with the [270M parameter model available on Hugging Face](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca).

**Dify Platform Ups and Downs**: The open-source LLM app development platform Dify is gaining traction for combining AI workflows and model management, although concerns have arisen about its lack of [loops and context scopes](https://github.com/langgenius/dify?tab=readme-ov-file).

**PyTorch Pumps Up LLM Training**: PyTorch has introduced [Torchtitan](https://github.com/pytorch/torchtitan), a library dedicated to aiding the training of substantial language models like llama3 from scratch.

**Video Gen Innovation with SORA**: OpenAI's SORA, a video generation model that crafts videos up to a minute long, is getting noticed, with user experiences and details explored in an [FXGuide article](https://www.fxguide.com/fxfeatured/actually-using-sora/).

**MOD Layers for Efficient Transformer Training**: The 'Mixture of Depths' paper was presented, proposing an accelerated training methodology for transformers by alternately using new MOD layers and traditional transformer layers, introduced in the [presentation](https://paper-club.ivanleo.com/papers/mixture-of-depths) and detailed in the paper's [abstract](https://arxiv.org/abs/2402.00841).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Phi-3-Mini-4K Instruct Powers Up**: Utilizing **Phi-3-Mini-4K-Instruct** with llamafile provides a setup for high-quality and dense reasoning datasets as discussed by members, with [integration steps outlined on Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile).

- **Model Download Made Easier**: A README update for **Mixtral 8x22B Instruct llamafile** includes a download tip: use `curl -L` for smooth redirections on CDNs, as seen in the [Quickstart guide](https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile).

- **Llamafile and CPUs Need to Talk**: An issue with running llamafile on an **Apple M1** Mac surfaced due to AVX CPU feature requirements, with the temporary fix of a system restart and advice for using smaller models on 8GB RAM systems shared in this [GitHub issue](https://github.com/Mozilla-Ocho/llamafile/issues/327#issuecomment-2053680659).

- **Windows Meets Llamafile, Confusion Ensues**: Users reported **Windows Defender** mistakenly detecting llamafile as a trojan. Workarounds proposed included using virtual machines or whitelisting, with the reminder that official binaries can be found [here](https://www.microsoft.com/en-us/wdsi/filesubmission).

- **Resource-Hungry Models Test Limits**: Engaging the 8x22B model requires heavy resources, with references to a recommended 128GB RAM for stable execution of [Mistral 8x22B model](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8), marking the necessity for big memory footprints when running sophisticated AI models.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Llama Beats Judge in Judging**: On the **Judgemark** benchmark, **Llama-3-70b** showcased impressive performance, demonstrating its potential for fine-tuning purposes in **disco-judge** applications, as it supports at least 8k context length. The community also touched on collaborative evaluation efforts, with references to advanced judging prompt design to assess complex rubrics.

**Benchmarking Models and Discussing Inference Issues**: **Phi-3-mini-4k-instruct** unexpectedly ranked lower on the **eq-bench** leaderboard despite promising scores in published evaluations. In model deployment, discussions highlighted issues like slow initialization and inference times for **DiscoLM_German_7b_v1** and potential misconfigurations that could be remedied using `device_map='auto'`.

**Tooling API Evaluation and Hugging Face Inquiries**: Community debates highlighted **Tgi** for its API-first, low-latency approach and praised **vllm** for being a user-friendly library optimized for cost-efficiency in deployment. Queries on Hugging Face's batch generation capabilities sparked discussion, with community involvement evident in a GitHub issue exchange.

**Gratitude and Speculation in Model Development**: Despite deployment issues, members have expressed appreciation for the **DiscoLM** model series, while also speculating about the potential of constructing an **8 x phi-3 MoE model** to bolster model capabilities. **DiscoLM-70b** was also a hot topic, with users troubleshooting errors and sharing usage experiences.

**Success and Popularity in Model Adoption**: The adaptation of the **Phi-3-mini-4k** model, referred to as llamafication, yielded a respectable EQ-Bench Score of 51.41 for German language outputs. Conversation also pinpointed the swift uptake of the **gguf** model, indicated by a notable number of downloads shortly after its release.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Claude Displays Depth and Structure**: In a rich discussion, the behavior and training of **Claude** were considered "mostly orthogonal" to Anthropic's vision, revealing unexpected depth and structural understanding through **RLAIF training**. Comparisons were made to concepts like "Jungian individuation" and conversation threads [highlighted Claude's capabilities](https://x.com/repligate/status/1783426037210026372?s=46&t=xxWoJxAS_7-BBFC2ro84Zw).

**Debating the Merits of RLHF vs. KTO**: A comparison between **Reinforcement Learning from Human Feedback (RLHF)** and **Knowledge-Targeted Optimization (KTO)** sparked debate, considering their suitability for different commercial deployments.

**Training Method Transition Yields Improvements**: An interview was mentioned where a progression in training methods from **Supervised Fine Tuning (SFT)** to **Data Programming by Demonstration (DPO)**, and then to **KTO**, led to improved performance based on user feedback.

**Unpacking the Complexity of RLHF**: The community acknowledged the intricacies of **RLHF**, especially as they relate to varying data sources and their impact on downstream evaluation metrics.

**Probing Grad Norm Spikes**: A request for clarity on the implications of gradient norm spikes during pretraining was made, emphasizing the potential adverse effects but specifics were not delivered in the responses.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

**Moondream Takes On CAPTCHAs**: A [video guide](https://www.youtube.com/watch?v=Gwq7smiWLtc) showcases fine-tuning the **Moondream Vision Language Model** for better performance on a CAPTCHA image dataset, aimed at improving its image recognition capabilities for practical applications.

**Low-Cost AI Models Make Cents**: The document "Low-Cost Language Models: Survey and Performance Evaluation on Python Code Generation" was shared, covering evaluations of **CPU-friendly language models** and introducing a novel dataset with 60 programming problems. The use of a Chain-of-Thought prompt strategy is highlighted in the [survey article](https://arxiv.org/abs/2404.11160).

**Meet, Greet, and Compute**: AI developers are invited to a meetup at **Cohere space** in Toronto, which promises networking opportunities alongside lightning talks and demos — details available on the [event page](https://lu.ma/devs5).

**Arctic Winds Blow for Enterprises**: **Snowflake Arctic** is introduced via a [new video](https://www.youtube.com/watch?v=nV6eIjnHEH0), positioning itself as a cost-effective, enterprise-ready Large Language Model to complement the suite of AI tools tailored for business applications.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Run Models Locally with Ease**: Engineers explored [jan.ai](https://jan.ai/), a GUI commended for its straightforward approach for running GPT models on local machines, potentially simplifying the experimentation process.
- **Apple Enters the Language Model Arena**: The new [OpenELM](https://huggingface.co/apple/OpenELM) series introduced by Apple provides a spectrum of efficiently scaled language models, including instruction-tuned variations, which could change the game for parameter efficiency in modeling.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Llama 3 Steps Up in Topic Complexity**: Venadore has started experimenting with **llama 3** for topic complexity classification, reporting promising results.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1232939117835456574)** (1265 messages🔥🔥🔥): 

- **Finetuning LLM Challenges**: Members discussed finetuning issues with their models, particularly when working with tools like awq, gptq, and running models in 4-bit quantization. There were specific issues with token indices sequence length errors, over 48GB of VRAM being insufficient for running certain models, and confusion around utilizing Aphrodite Engine or llama.cpp for testing finetuned models. Remedies suggested included revising batch sizes and grad accumulation, enabling packing, and reaching out to community experts for guidance.
- **Finding the Right Technology Stack**: One user expressed a desire to integrate different AI models into a project that allows chatting with various agents for distinct tasks. Experienced community members recommended starting with simpler scripts instead of complex AI solutions and advised doing thorough research before implementation. Concerns about API costs versus local operation were also discussed.
- **Game Preferences and Recommendations**: Users shared their excitement about the recent launch of games like "Manor Lords" on early access and provided personal insights into the entertainment value of popular titles such as "Baldur's Gate 3" and "Elden Ring."
- **Unlocking Phi 3's Fused Attention**: It was revealed that Phi 3 Mini includes fused attention, sparking curiosity among members. Despite the feature's presence, users were advised by others to wait for further development before diving in.
- **Unsloth Achieves Significant Downloads**: The Unsloth team announced hitting 500k monthly model downloads on Hugging Face, thanking the community for the widespread support and usage of Unsloth's finetuning framework. The necessity of uploading GGUF models was discussed, with the possible redundancy noted due to others already providing them.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/LexiFun-Llama-3-8B-Uncensored-V1">Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1 · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.16710">Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding</a>: We present LayerSkip, an end-to-end solution to speed-up inference of large language models (LLMs). First, during training we apply layer dropout, with low dropout rates for earlier layers and higher ...</li><li><a href="https://arxiv.org/abs/2403.13799">Reverse Training to Nurse the Reversal Curse</a>: Large language models (LLMs) have a surprising failure: when trained on &#34;A has a feature B&#34;, they do not generalize to &#34;B is a feature of A&#34;, which is termed the Reversal Curse. Even w...</li><li><a href="https://www.amazon.co.uk/Yalucky-Novelty-Drinkware-Birthday-Christmas/dp/B0834QSW5Z">no title found</a>: no description found</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-bnb-4bit">unsloth/llama-3-8b-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://hub.docker.com/r/alpindale/aphrodite-dev/tags">Docker</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-110B-Chat">Qwen/Qwen1.5-110B-Chat · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/rookie-numbers-gif-26135237">Rookie Numbers GIF - Rookie Numbers - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/apple/OpenELM">apple/OpenELM · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TETO101/AIRI-L3-INS-1.0-0.00018-l">TETO101/AIRI-L3-INS-1.0-0.00018-l · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-262k-GGUF">crusoeai/Llama-3-8B-Instruct-262k-GGUF · Hugging Face</a>: no description found</li><li><a href="https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py)">Layer Normalization &mdash; Triton  documentation</a>: no description found</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: no description found</li><li><a href="https://github.com/meta-llama/llama3/blob/main/LICENSE">llama3/LICENSE at main · meta-llama/llama3</a>: The official Meta Llama 3 GitHub site. Contribute to meta-llama/llama3 development by creating an account on GitHub.</li><li><a href="https://github.com/oKatanaaa/kolibrify">GitHub - oKatanaaa/kolibrify: Curriculum training of instruction-following LLMs with Unsloth</a>: Curriculum training of instruction-following LLMs with Unsloth - oKatanaaa/kolibrify</li><li><a href="https://github.com/meta-llama/llama-recipes">GitHub - meta-llama/llama-recipes: Scripts for fine-tuning Meta Llama3 with composable FSDP &amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp; custom datasets for applications such as summarization and Q&amp;A. Supporting a number of candid inference solutions such as HF TGI, VLLM for local or cloud deployment. Demo apps to showcase Meta Llama3 for WhatsApp &amp; Messenger.</a>: Scripts for fine-tuning Meta Llama3 with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization and Q...</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1233054158064713779)** (18 messages🔥): 

- **Finetuning Strategies Discussed**: Members of the chat discuss whether *finetuning a language model on raw text* would cause it to lose its chatty capabilities. A solution is proposed to combine raw text with a chat dataset to preserve the conversational ability while adding knowledge from the raw text.

- **Rumors of WizardLM Disbanding**: Speculation arises based on [Qingfeng Sun's staff page](https://www.microsoft.com/en-us/research/people/qins/) redirection, suggesting that he may no longer be at Microsoft which could signal the closure of the WizardLM team. Links to a [Reddit thread](https://old.reddit.com/r/LocalLLaMA/comments/1cd4b9l/staff_page_for_qingfeng_sun_lead_wizard_lm/) and a [Notion blog post](https://rocky-muscle-755.notion.site/What-happened-to-Wizard-LM2-a247e09244d0483cbb02c1587b357c9d) give credence to the theory.

- **Unsloth AI Finetuning Resources**: For finetuning on combined datasets, members are directed to Unsloth AI’s repository on GitHub [finetune for free](https://github.com/unslothai/unsloth), which lists all available notebooks, and specifically to a [Colab notebook for text completion](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing).

- **WizardLM Data Salvage Operation**: Following the discussion about potential layoffs associated with Microsoft's WizardLM, a member comes forward stating they have copies of the WizardLM datasets which might aid in future endeavors.

- **The Rollercoaster of Model Training**: Chat members humorously share their experiences with model training, referring to their loss curves with a mix of hope and defeat.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1cd4b9l/staff_page_for_qingfeng_sun_lead_wizard_lm/">Staff page for Qingfeng Sun (Lead Wizard LM Researcher) has been deleted from Microsoft.com</a>: If you go to the staff page of [Qingfeng Sun](https://www.microsoft.com/en-us/research/people/qins/) you'll be redirected to a generic landing...</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1cd4b9l/staff_page_f">Staff page for Qingfeng Sun (Lead Wizard LM Researcher) has been deleted from Microsoft.com</a>: If you go to the staff page of [Qingfeng Sun](https://www.microsoft.com/en-us/research/people/qins/) you'll be redirected to a generic landing...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1232935568678195214)** (86 messages🔥🔥): 

- **Inference Snippet Inquiry**: A member inquired about a simpler method to test **GGUF models** without loading them into Oobabooga. Another member indicated future plans to provide **inference and deployment options**.

- **Triton Troubles**: Members discussed issues with **Triton** and its necessity for running Unsloth locally. A member had trouble with `triton.common` module due to a potential version conflict, and others acknowledged Triton as a requirement.

- **Fine-Tuning Frustrations**: A conversation circled around issues faced during fine-tuning where a model kept repeating the last token. The solution suggested was to **update the generation config** using the latest [colab notebooks](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp).

- **Quantization Failure Chaos**: Multiple members encountered a "**Quantization failed**" error when trying to use `save_pretrained_merged` and `save_pretrained_gguf`. The issue was ultimately identified as a user error where **llama.cpp was not in the model folder**, but was resolved after fixing the file location.

- **Model Training Errors and Insights**: A mix of questions and solutions were discussed regarding **training errors**, resuming training from checkpoints on platforms like Kaggle, and **finetuning** guidance. One notable point was the use of **checkpointing**, which allows training to resume from the last saved step, benefiting users on platforms with limited continuous runtime.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://www.reddit.com/r/comfyui/comments/1bq22x7/change_clothing_in_1_click_ootdiffusion/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets/TETO101/AIRI_INS5/viewer">TETO101/AIRI_INS5 · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1233058558992449658)** (10 messages🔥): 

- **Meta Unveils LlaMA-3**: Meta announces the next generation of its **LlaMA** model series, releasing an **8B model** and a **70B model**, with a teased upcoming 400B model promising GPT-4 level benchmarks. Interested parties can request access to these models, which are reportedly top of their size classes; the detailed comparison and insights are available in a [Substack article](https://datta0.substack.com/p/ai-unplugged-8-llama3-phi-3-training).
- **Open-Sourcing Kolibrify**: A user announces the release of their project **Kolibrify**, a tool for curriculum training of instruction-following LLMs with Unsloth, designed for **PhD research**. The tool, aimed at those finetuning LLMs on workstations for rapid prototyping, is available on [GitHub](https://github.com/oKatanaaa/kolibrify).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://datta0.substack.com/p/ai-unplugged-8-llama3-phi-3-training">AI Unplugged 8: Llama3, Phi-3, Training LLMs at Home ft DoRA.</a>: Insights over Information</li><li><a href="https://github.com/oKatanaaa/kolibrify">GitHub - oKatanaaa/kolibrify: Curriculum training of instruction-following LLMs with Unsloth</a>: Curriculum training of instruction-following LLMs with Unsloth - oKatanaaa/kolibrify
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1233133113719394324)** (7 messages): 

- **Innovating with TRL Trainer**: A member is working on implementing **[laser pruning and potentially freezing with a trl trainer](https://github.com/l4b4r4b4b4/trl/tree/evol_laser_merge_trainer)** that functions during the evaluation step. The goal is to iteratively increase context length for models while utilizing the same GPU.
  
- **No Reinitialization Required for Context Expansion**: The suggestion to increase available context length through model and tokenizer configuration adjustments was made. It was confirmed that these changes **do not require reinitialization** of the system.

- **Emoji Expressiveness in Chat**: Members are using emojis in their communication, with comments expressing surprise and delight at the ability to type **emojis** in the chat.
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1233126296460988489)** (1 messages): 

- **iOS Users Get Exclusive Feature**: The **Perplexity AI Discord chatbot** has been updated with a new feature where users can *ask any question and hear the answer*. This feature is available for **iOS <a:pro:1138537257024884847> users** starting today.
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1232950876365127753)** (531 messages🔥🔥🔥): 

- **Perplexity Supports Opus**: The regular Perplexity chatbot still supports Opus, despite the recent caps placed on its usage.
- **The Great Opus Limit Debate**: Users express frustration over the sudden limit placed on Claude 3 Opus interactions, reducing the available queries from previous higher or unlimited numbers to just 50 per day. Discussions revolve around the difference in model performance and pricing compared to competitors like you.com, as well as transparency regarding usage caps.
- **Enterprise Features Unclear**: Members discussed the difference between regular Pro and Enterprise Pro versions of Perplexity, especially in the context of privacy settings and data usage for model training. There seems to be confusion about whether setting a toggle does protect users' data from being used by Perplexity's models or Anthropic's models.
- **Transparency and Communication Critique**: Community members criticize Perplexity for poor communication regarding usage changes and urge for official announcements. Comparisons are made with other services like poe.com, which users perceive as more transparent with their pricing and limits.
- **Ecosystem Ramifications**: Conversation pondered the implications of Google potentially getting serious with their Gemini model, which some believe offers competitive advantages due to scalability and Google's dataset. Expectations are forming around GPT-5 needing to be particularly impressive in light of increasing competition.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/ChatGPT/comments/1ccfzon/google_presents_leave_no_context_behind_efficient/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/mattshumer_/status/1783531642344067159">Tweet from Matt Shumer (@mattshumer_)</a>: It&#39;s been a week since LLaMA 3 dropped.  In that time, we&#39;ve: - extended context from 8K -&gt; 128K - trained multiple ridiculously performant fine-tunes - got inference working at 800+ tokens...</li><li><a href="https://huggingface.co/spaces/multimodalart/stable-cascade">Stable Cascade - a Hugging Face Space by multimodalart</a>: no description found
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1232990801450238012)** (8 messages🔥): 

- **Exploring Perplexity AI**: A user shared a [search link](https://www.perplexity.ai/search/jTVe66V.RHKaTLZPJ3MZ0Q) without any accompanying commentary or context.
- **Diving into Pool Chemistry**: Discussing pool maintenance frustrations, a member mentioned the **Langlier Saturation Index** as a potentially helpful but complex solution not tailored for outdoor pools and shared an informative [Perplexity AI search link](https://www.perplexity.ai/search/What-is-the-hSnFPTgtQWu2MvGENVpNFg).
- **Net Neutrality and its AI Impact**: A link was shared regarding the FCC's restoration of Net Neutrality, with the post hinting at possible implications for the **AI Boom**, accessible at [Perplexity AI search](https://www.perplexity.ai/search/FCC-restores-net-dXr_Ke3ST8SdNITs2AMDhA#1).
- **Command the Commands**: One user queried about a specific command, referring to a [Perplexity AI search](https://www.perplexity.ai/search/what-command-i-IEqOU0n0SRyoAsxjDUkQoQ). Another user reminded to ensure the thread is **Shareable**.
- **AI for Voting?**: There's interest in how AI could apply to voting systems, with a user linking to a [Perplexity AI search](https://www.perplexity.ai/search/What-are-the-vwfV8gKTSHih8Np6rxFFIg#1).
- **Homeland Security's New Directive**: A share without comment included a link to a [Perplexity AI search](https://www.perplexity.ai/search/Homeland-Security-announced-8Q5AitclTxW6fYBe02t5CA) regarding an announcement from Homeland Security.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1232963461433266227)** (10 messages🔥): 

- **API Integration Quirks Noted**: An user mentioned integrating the Perplexity API with a speech assistant and observed issues with date relevance in responses, such as receiving a sports game score from a year ago instead of the current date. They also inquired about inserting documents for comparison purposes and expressed interest in expanded citation functionality, hinting at the potential for more versatile usage.
  
- **No GPT-4 Support with Perplexity**: A member was looking to use GPT-4 through the Perplexity API but found it unsupported. Another member provided a [documentation link](https://docs.perplexity.ai/docs/model-cards) listing available models, including `sonar-small-chat`, `llama-3-instruct` variants, and `mixtral-instruct`, but no mention of GPT-4.

- **Optimal Hyperparameters for llama-3-70b-instruct Usage**: An individual posed a question about the appropriate hyperparameters for using the `llama-3-70b-instruct` model via the API, sharing their current parameters structure and seeking confirmation or corrections, specifically regarding `max_tokens` and `presence_penalty` values.

- **Unclear Integration Details**: The same user mentioned staying within rate limits when making their calls to the API, although there was uncertainty regarding whether the Perplexity API operates identically to OpenAI's in terms of parameter settings.

- **Awaiting Enterprise API Response**: An enterprise user reached out in the channel after emailing Perplexity AI's enterprise contacts regarding API usage and awaiting a response. They were advised by another member that response times range from 1-3 weeks.

- **Clarification Sought on "Online LLMs" Usage**: A new user to Perplexity AI sought clarification on the guidance for using online LLMs, questioning whether to avoid using system prompts and if it was necessary to present queries in a single-turn conversation format.

**Link mentioned**: <a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found

  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1233008609336754176)** (13 messages🔥): 

- **Seeking Further CUDA Mastery**: A discussion suggested enhancing CUDA learning through public demonstrations of skill, specifically by writing a fast kernel and sharing the work. Suggested projects included optimizing a fixed-size matrix multiplication, flash attention, and various quantization methods.

- **BASH Channel Turns CUDA Incubator**: The `<#1189861061151690822>` channel is slated to focus on algorithms that could benefit from CUDA improvement, inviting members to contribute with their optimized kernels. However, there was a recommendation to create a more permanent repository for these contributions beyond the transient nature of Discord channels.

- **Instance GPU Configuration Verification**: A user confirmed that upon accessing their instance via SSH, the GPU configuration is verified, revealing consistent assignment of a V100 for p3.2xlarge.

- **Next CUDA Lecture Up and Coming**: An announcement was made regarding an upcoming CUDA mode lecture, scheduled to occur in 1 hour and 40 minutes from the time of the announcement.

- **Anticipating CUDA Updates**: There was a query regarding the release schedule for CUDA distributables for Ubuntu 24.04, but there was no follow-up information provided within the message history.
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1233014978609680425)** (40 messages🔥): 

- **Kernel Profiling Confusion**: One member was trying to obtain more detailed information about kernel operations using **NVIDIA's Nsight Profiler**. After initial confusion between Nsight Systems and Nsight Compute, it was clarified that using [NVIDIA Nsight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction) could yield detailed kernel stats.
- **Understand the Synchronize**: An explanation was given about `cudaStreamSynchronize`, how it implies the CPU waits for tasks in the CUDA stream to finish, and it was suggested to check if the synchronization is essential at each point it's called to potentially improve performance.
- **Occupancy and Parallelization Advice**: Discussion touched on launch statistics of CUDA kernels, indicating that launching only a small number of blocks such as **14 blocks** could result in GPU idle time unless multiple CUDA streams are being utilized.
- **Performance Insights and Tweaks**: For in-depth kernel analysis, it was suggested to switch to *full metric selection* in profiling for more comprehensive information, and a broader tip was given to aim for a higher number of blocks rather than introducing the complexity of CUDA streams, if feasible.
- **Arithmetic Intensity vs Memory Bandwidth**: There was a comparison of the FLOP/s and memory throughput between the *tiled_matmult* and *coarsed_matmult* kernels, with observations on how *__syncthreads()* calls and memory bandwidth relate. The discussion evolved into how arithmetic intensity (AI) is perceived from SRAM versus DRAM perspectives when profiling with Nsight Compute / ncu.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#introduction">4. Nsight Compute CLI &mdash; NsightCompute 12.4 documentation</a>: no description found</li><li><a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>: In this post, I’ll iteratively optimize an implementation of matrix multiplication written in CUDA.My goal is not to build a cuBLAS replacement, but to deepl...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1232984060129968178)** (9 messages🔥): 

- **Tensor Expansion Explained**: A discussion revealed that `Tensor.expand` in PyTorch works by modifying the tensor's strides, not its storage. It was noted that when using Triton kernels, issues may arise from improper handling of these modified strides.

- **Flash-Attention Incompatibility Alert**: There was a report of incompatibility between the newly released **flash-attn version 2.5.7** and the CUDA libraries installed by **PyTorch 2.3.0**, specifically an `undefined symbol` error, and hopes were expressed for a prompt update to resolve this.

- **Building Flash-Attention Challenges**: A user encountered difficulties in building **flash-attn**, mentioning that the process was excessively time-consuming without ultimate success.

- **Understanding CUDA Tensor Memory**: A member shared a [useful overview](https://pytorch.org/docs/stable/notes/cuda.html) clarifying that the memory pointers for CUDA tensors always point to device memory, and cross-GPU operations are restricted by default in PyTorch.

**Link mentioned**: <a href="https://pytorch.org/docs/stable/notes/cuda.html">CUDA semantics &mdash; PyTorch 2.3 documentation</a>: no description found

  

---


**CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1233492304527098017)** (1 messages): 

- **Boost in LLM Performance**: An exciting bonus talk by the **NVIDIA C++ team** was announced, discussing the porting of `llm.c` to `llm.cpp` promising **cleaner and faster code**. The session was starting shortly.
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1233058707487719536)** (47 messages🔥): 

- **Exploring Plenoxels and SLAM Algorithms**: The chat discussed [Plenoxels CUDA kernels](https://github.com/sxyu/svox2/tree/master/svox2/csrc), which are faster variants of NeRF, and expressed interest in seeing a CUDA version of [Gaussian Splatting SLAM](https://github.com/muskie82/MonoGS).
- **Acceleration of Mobile ALOHA with CUDA**: The inference algorithms for [Mobile ALOHA](https://mobile-aloha.github.io/), such as [ACT](https://github.com/MarkFzp/act-plus-plus) and [Diffusion Policy](https://github.com/MarkFzp/act-plus-plus), were topics of interest.
- **Kernel Operations for Binary Matrices**: There was a brainstorming session on creating a CUDA kernel for operations on binary (0-1) or ternary (1.58-bit, -1, 0, 1) matrices. The group discussed potential approaches avoiding unpacking, including a masked multiply tactic and kernel fusion.
- **Low-bit Quantization and Efficiency Discussions**: Members debated the efficiency of unpacking operations in Pytorch vs. fused CUDA or Triton kernels. Some suggested that operations be conducted without unpacking, while others highlighted memory copies and caching as significant concerns. [Microsoft's 1-bit LLM paper](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf) was mentioned as a motivating idea for optimizing linear layers in neural networks.
- **Challenges with Packed Operations in CUDA**: The conversation centered around the feasibility of conducting matrix multiplication-like operations directly on packed data types without unpacking, referring to CUDA 8.0's bmmaBitOps as a potential method. Discussion included [bit operations in CUDA's programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=Warp#sub-byte-operations) and the interest in trialing computations minimizing unpacking. A member provided a link to a [CPU version of BitNet](https://github.com/catid/bitnet_cpu) for testing purposes.

**Link mentioned**: <a href="https://github.com/catid/bitnet_cpu">GitHub - catid/bitnet_cpu: Experiments with BitNet inference on CPU</a>: Experiments with BitNet inference on CPU. Contribute to catid/bitnet_cpu development by creating an account on GitHub.

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1233009306874679356)** (6 messages): 

- **Exploring Multi-GPU Programming**: One member hinted at the potential for learning about multi-GPU programming, suggesting it might be an area of interest.
- **Laptop with NVIDIA GPU for Learning**: Members concurred that a laptop with a NVIDIA GPU, such as one containing a 4060, is a cost-effective option suitable for learning and testing CUDA code.
- **Jetson Nano for CUDA Exploration**: A Jetson Nano was recommended for those looking to learn CUDA programming, especially when there is an extra monitor available.
- **Search for NCCL All-Reduce Kernel Tutorial**: A request was made for tutorials on learning NCCL to implement all-reduce kernels. No specific resources were provided in the chat.
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1232974757205639292)** (5 messages): 

- **Clarifying Burst Size**: Burst size refers to the chunk of memory accessed in a single load operation during *memory coalescing*, where the hardware combines multiple memory loads from contiguous locations into one load to improve efficiency. This concept is explored in **Chapter 6, section 3.d** of the CUDA guide where it mentions that burst sizes can be around **128 bytes**.
- **Insights from External Resources**: A helpful [lecture slide](https://lumetta.web.engr.illinois.edu/408-S20/slide-copies/ece408-lecture8-S20.pdf) by the book's authors was provided, affirming that bursts typically contain **128 bytes** which clarifies the concept of coalesced versus uncoalesced memory access.
- **Discrepancy in Burst Size Understanding Corrected**: There was a reiterating message indicating that initially, there was a misunderstanding about coalesced access, which was resolved after revisiting and rereading the relevant section of the CUDA guide.
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

poker6345: ppt can be shared
  

---


**CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1233099632007970978)** (2 messages): 

- **Simplified BucketMul Function Revelations**: A member shared a [simplified version](https://kolinko.github.io/effort/gpu.html) of the **bucketMul** function, highlighting how it factors both weights and dispatch in computing multiplications, and potentially optimizing memory loads. It resembles discussions on bucketed COO for better memory performance, with the added consideration of activations.

- **AO Welcomes Custom CUDA Extensions**: PyTorch AO now supports custom CUDA extensions, allowing seamless integration with `torch.compile` by following the provided template, as per a [merged pull request](https://github.com/pytorch/ao/pull/135). This is especially enticing for those adept at writing CUDA kernels and aiming to optimize performance on consumer GPUs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://kolinko.github.io/effort/gpu.html">Effort Engine</a>: no description found</li><li><a href="https://github.com/pytorch/ao/pull/135">Custom CUDA extensions by msaroufim · Pull Request #135 · pytorch/ao</a>: This is the mergaeble version of #130 - some updates I have to make   Add a skip test unless pytorch 2.4+ is used and Add a skip test if cuda is not available  Add ninja to dev dependencies  Locall...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/)** (1 messages): 

iron_bound: https://www.harmdevries.com/post/context-length/
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: https://github.com/adam-maj/tiny-gpu
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1232954105031295006)** (377 messages🔥🔥): 

- **Chasing MultigPU Efficiencies**: The group is working on integrating multi-GPU support with NCCL, discussing performance penalties of multi-GPU configurations, and potential improvements like gradient accumulation. A move to merge NCCL code into the main branch is considered, along with a discussion about whether FP32 should support multi-GPU, leaning towards not including it.
  
- **Optimizing Gather Kernel without Atomics**: A strategy is discussed for optimizing a layernorm backward kernel by avoiding atomics, using threadblock counting and grid-wide synchronization techniques to manage dependencies and streamline calculations.
  
- **Debugging and Decision-making for FP32 Path**: It's suggested that the FP32 version of `train_gpt2` be simplified for educational purposes, possibly stripping out multi-GPU support to keep the example as intuitive as possible for beginners.
  
- **Brainstorming Persistent Threads and L2 Communication**: There’s an in-depth technical discussion about the potential benefits and drawbacks of using persistent threads with grid-wide synchronization to exploit the memory bandwidth more efficiently and potentially run multiple kernels in parallel.

- **Parallelism and Kernel Launch Concerns**: Dialogue revolves around the comparison of the new CUDA concurrent kernel execution model managed by queues to traditional methods, with thoughts on the pros and cons of embracing this uncharted approach to achieve better memory bandwidth exploitation and reduced latency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/73853956/tf-bitcast-equivalent-in-pytorch)">tf.bitcast equivalent in pytorch?</a>: This question is different from tf.cast equivalent in pytorch?.&#xA;bitcast do bitwise reinterpretation(like reinterpret_cast in C&#x2B;&#x2B;) instead of &amp;quot;safe&amp;quot; type conversion.&#xA...</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtc24-s62419/">Energy and Power Efficiency for Applications on the Latest NVIDIA Technology | NVIDIA On-Demand</a>: With increasing energy costs and environmental impact, it's increasingly important to consider not just performance but also energy usage</li><li><a href="https://x.com/chhillee/status/1770210441643577377?s=46&t=yqOem5ktaowo8FyJ-ilbzQ">Tweet from Horace He (@cHHillee)</a>: It&#39;s somehow incredibly hard to get actual specs of the new Nvidia GPUs, between all the B100/B200/GB200/sparse/fp4 numbers floating around. @tri_dao linked this doc which thankfully has all the n...</li><li><a href="https://github.com/k">k - Overview</a>: k has 88 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/issues/212,">Issues · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/252">reorder weights according to their precision by ngc92 · Pull Request #252 · karpathy/llm.c</a>: Simplify our logic by keeping weights of the same precision close together. (If we want to go with this, we also need to update the fp32 network to match; hence, for now this is a Draft PR)</li><li><a href="https://github.com/apple/corenet/tree/main/projects/openelm">corenet/projects/openelm at main · apple/corenet</a>: CoreNet: A library for training deep neural networks - apple/corenet</li><li><a href="https://github.com/adam-maj/tiny-gpu">GitHub - adam-maj/tiny-gpu: A minimal GPU design in Verilog to learn how GPUs work from the ground up</a>: A minimal GPU design in Verilog to learn how GPUs work from the ground up - adam-maj/tiny-gpu</li><li><a href="https://www.youtube.com/watch?v=e24BlWvSLNM">Self-Improving Agents are the future, let’s build one</a>: If you&#39;re serious about AI, and want to learn how to build Agents, join my community: https://www.skool.com/new-societyFollow me on Twitter - https://x.com/D...</li><li><a href="https://github.com/karpathy/llm.c/pull">Pull requests · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[massively-parallel-crew](https://discord.com/channels/1189498204333543425/1229286073104994344/1233114037676413029)** (25 messages🔥): 

- **Seeking Volunteers for Recording**: One member commits to screen recording and requests a backup recorder due to having to leave the session early. They advise against using AirPods as they might change the system audio output unpredictably.
- **Mac Screen Recording Tutorial**: Guidance provided for screen recording on a Mac, including downloading Blackhole from [here](https://existential.audio/blackhole/download/?code=681349920) and setting up a Multi-Output Device in the "Audio MIDI Setup."
- **Audio Troubleshooting with Blackhole**: It's suggested to avoid Bluetooth devices for audio capture to prevent interruptions, and to select BlackHole 2ch for lossless sound recording.
- **Step-by-Step Recording Instructions**: Detailed instructions include using the Cmd + Shift + 5 shortcut, selecting the entire screen, saving to an external drive, and ensuring the microphone is set to BlackHole 2ch.
- **Pre-Recording Tech Check Proposed**: A call is suggested before the event for checking sound and recording settings.

**Link mentioned**: <a href="https://existential.audio/blackhole/download/?code=681349920)">Existential Audio - BlackHole</a>: no description found

  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1232943660614815754)** (218 messages🔥🔥): 

- **LM Studio Becomes a One-Stop Chat Hub**: Members discussed integrating documents with the chat feature in LM Studio using Retriever-Augmented Generation (RAG) through custom scripts and the API. Some showcased successfully directing LM Studio to utilize their system's GPU instead of the CPU for model operations, with a switch found in the Settings panel.

- **Navigating Update Confusions**: There was confusion about updating to version 0.2.21 of LM Studio, with some users not able to see the update through the auto-updater. It was clarified that the new version hadn't been pushed to the updater, and members were directed to manually download it from [LM Studio's official website](https://lmstudio.ai/).

- **Challenges with Offline Image Generation and AI Chat**: Users inquired about offline image generation capabilities, being redirected to Automatic1111 for those needs. The conversation also mentioned experiencing 'awe' moments with AI advancements, particularly when interacting with chatbots like AI Chat on LM Studio.

- **Troubleshooting Varied Issues**: From GPU support questions to errors like "Exit code: 42," members tried to troubleshoot issues ranging from installation errors on different versions of LM Studio to getting specific models to work. **heyitsyorkie** provided advice on many technical issues, including recommending updates or altering settings to overcome errors.

- **Technical inquiries about LM Studio capabilities and settings**: Users engaged in various technical discussions around LM Studio's API server capabilities, inference speeds, GGUF model support, and specific hardware requirements for running large language models. **heyitsyorkie** and other members shared insights and resources, including linking to the [local server documentation](https://lmstudio.ai/docs/local-server) and discussing optimal setups for AI inference.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: no description found</li><li><a href="https://onnxruntime.ai/blogs/accelerating-phi-3">ONNX Runtime | Blogs/accelerating-phi-3</a>: Cross-platform accelerated machine learning. Built-in optimizations speed up training and inferencing with your existing technology stack.</li><li><a href="https://huggingface.co/ChristianAzinn/acge_text_embedding-gguf">ChristianAzinn/acge_text_embedding-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-q4.gguf">Phi-3-mini-4k-instruct-q4.gguf · microsoft/Phi-3-mini-4k-instruct-gguf at main</a>: no description found</li><li><a href="https://lmstudio.ai/docs/local-server">Local LLM Server | LM Studio</a>: You can use LLMs you load within LM Studio via an API server running on localhost.</li><li><a href="https://huggingface.co/google/siglip-so400m-patch14-384">google/siglip-so400m-patch14-384 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/aspire/acge_text_embedding">aspire/acge_text_embedding · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/qresearch/llama-3-vision-alpha">qresearch/llama-3-vision-alpha · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit">unsloth/llama-3-8b-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/ChristianAzinn">ChristianAzinn (Christian Zhou-Zheng)</a>: no description found</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">The unofficial LMStudio FAQ!</a>: Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...</li><li><a href="https://rentry.org/LMSTudioFAQ#how-">The unofficial LMStudio FAQ!</a>: Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio)">The unofficial LMStudio FAQ!</a>: Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1232943347111297065)** (75 messages🔥🔥): 

- **Exploring Model Options for Confluence/Jira BI Analysis**: A user inquired about a suitable model for analyzing data from Confluence/Jira for business intelligence analysis within a company intranet, seeking suggestions for potential models and implementation strategies.
- **Seeking Superior Python Coding Model**: When asked about the best model for Python coding, responses varied, recommending models like **CodeQwen1.5** or **DeepSeek-Coder**, with follow-up intentions to try these suggestions.
- **Translation Capabilities Questioned**: A user queried the chat for recommendations on a good 7b model that excels in translations, though no specific recommendations were provided within the messages summarized.
- **LLM Studio Compatibility Queries for Apple's OpenELM**: Discussion arose around getting Apple's OpenELM to work with LM Studio, highlighting challenges due to incompatibility with llama.cpp and waiting for required support (`https://github.com/ggerganov/llama.cpp/issues/6868`).
- **Adventures with Phi-3 Models**: Users discussed issues with downloading, loading, and running different versions of Phi-3 models in LM Studio, with some having trouble loading certain downloaded models. Conversations suggested using the GGUF format and checking if one's LM Studio version supports the phi3 format, with v0.2.21 potentially necessary for these models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://huggingface.co/Orenguteng/LexiFun-Llama-3-8B-Uncensored-V1-GGUF">Orenguteng/Llama-3-8B-LexiFun-Uncensored-V1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/">microsoft/Phi-3-mini-4k-instruct-gguf · Hugging Face</a>: no description found</li><li><a href="https://docs.krita.org/en/user_manual/layers_and_masks.html">Introduction to Layers and Masks</a>: no description found</li><li><a href="https://pinokio.computer/">Pinokio</a>: AI Browser</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6868">Support for OpenELM of Apple · Issue #6868 · ggerganov/llama.cpp</a>: Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...</li><li><a href="https://arxiv.org/html/2402.13753v1">LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/1684">k-quants by ikawrakow · Pull Request #1684 · ggerganov/llama.cpp</a>: What This PR adds a series of 2-6 bit quantization methods, along with quantization mixes, as proposed in #1240 and #1256. Scalar, AVX2, ARM_NEON, and CUDA implementations are provided. Why This is...</li><li><a href="https://www.futuretools.io/">Future Tools - Find The Exact AI Tool For Your Needs</a>: FutureTools Collects &amp; Organizes All The Best AI Tools So YOU Too Can Become Superhuman!
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1232947701335720009)** (7 messages): 

- **Persistent Error Across Versions**: A Linux user using **Debian** reports that they encounter the same error with the latest version, with the last working version being **2.19**.
- **Scaling GPU Usage But Model Fails to Load**: Another member experiences a spike in **GPU usage to 100%** upon trying to load a model using LM Studio **version 2.20**, but the model fails to load despite high GPU utilization.
- **Call for Reduced Graphics Memory Usage**: It was highlighted that the LM Studio UI consumes around **500MB of graphics memory**, which could potentially limit the memory available for models, prompting a suggestion to reduce the graphics memory usage.
- **Update Woes with Phi-3 Mini**: A member reports that after updating to version **0.2.21**, their previously functioning setup with **phi-3 mini** (using official Microsoft gguf and LM Studio config from GitHub) now produces gibberish.
- **Screenshot Request for Debugging**: In response to the phi-3 mini issue, there was a request for **screenshots** to help investigate the problem.


  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1232984596958941266)** (64 messages🔥🔥): 

- **GPU Offload Error Resolved with Switch**: A user found a solution to their **GPU Offload** error by switching the GPU type to **AMD Open CL**, which allowed the GPU offload to work despite an initial technical issue.
- **Troubleshooting a Model Loading Error**: A participant reported consistent problems with loading their model on a system with a Tesla P100 GPU, an e5-2450V4 CPU, and 16GB of RAM. Further conversation revealed the CPU's actual model to be 2650v4, not 2450v4.
- **Query on GPU Selection for Model Utilization**: A member asked for advice on directing **Mistral 7B** to use a dedicated GPU instead of defaulting to the CPU's integrated graphics, aiming to resolve performance issues.
- **Anticipation for a Potential Performance Boost**: After ordering an Nvidia Tesla P40, a community member eagerly anticipated a significant increase in token per second performance, which could enable the use of larger models and potentially multiple models at once.
- **Hardware Advise for LLM Hosting**: For those looking to host a home server for AI and web applications, members advised that a system with at least 16GB VRAM is necessary, and Nvidia's contemporary architecture GPU might be preferable.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/thumbs-up-nice-well-done-approve-good-job-gif-13666522">Thumbs Up Nice GIF - Thumbs Up Nice Well Done - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/jon-stewart-eat-eating-popcorn-watching-gif-3094746547306242594">Jon Stewart Eat GIF - Jon Stewart Eat Eating - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1232939436526927952)** (30 messages🔥): 

- **ROCm on Nvidia? Not Quite**: A member mistakenly used AMD's ROCm preview with an Nvidia GPU, but realized it defaulted to the CPU. Using ROCm technology with incompatible hardware results in CPU fallback.
- **ROCm Performance Report**: An individual reported impressive speeds with ROCm, achieving 30t/s on an eGPU setup, indicating significant performance capabilities for supported configurations.
- **Checking GPU Compatibility**: In response to inquiries about GPU support, a member linked to documentation, emphasizing that only GPUs with a checkmark under the HIPSDK are compatible with the ROCm build.
- **High Hopes for AMD Improvements**: Community members are both critiquing and expressing hope for AMD's developments in the tech space, suggesting a mix of anticipation and skepticism within the chat.
- **Troubleshooting ROCm Errors**: Users discussed errors and compatibility issues when trying to run models with the ROCm build, indicating that proper driver installation and compatibility with the HIPSDK are crucial for success.
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1232959243875913738)** (22 messages🔥): 

- **Understanding RoPE and Extrapolation**: RoPE scaling's effectiveness was debated, with one member sharing that changing the RoPE base during fine-tuning, not pretraining, enhances a model's ability to handle longer contexts as per a [research paper](https://arxiv.org/abs/2310.05209). However, it was clarified that **Llama 3** was pretrained with a 500k RoPE base, without changing the base, in an attempt to decrease RoPE decay factor for longer contexts.

- **Extrapolation Tokens Outweigh Pretraining**: The community discussed the relationship between the number of pretraining tokens and the model's ability to extrapolate, concluding that extensive pretraining is necessary before any further pretraining with higher RoPE bases to prevent loss of extrapolation capabilities.

- **PoSE as an Alternative**: A member referenced Positional Skip-wisE (PoSE) as a novel training method that simulates long inputs using a fixed context window, which could potentially address limitations of relative positional encodings. The method smartly chunks the original context window for efficient extension, as described in the [associated paper](https://openreview.net/forum?id=3Z1gxuAQrA).

- **Linear Scaling of RoPE Base Debated**: One member solicited insights on how to scale RoPE base with context length, with a community expert noting that setting the base to an arbitrarily high number and then doing empirical testing is common, rather than any systematic linear scaling.

- **Endorsement for Better Positional Encodings**: The conversation highlighted RoPE as potentially inadequate for long context generalization and proposed alternatives like YaRN or LongRoPE, specifically mentioning that LongRoPE is utilized in the **phi-3-128k** model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openreview.net/forum?id=3Z1gxuAQrA">PoSE: Efficient Context Window Extension of LLMs via Positional...</a>: Large Language Models (LLMs) are trained with a pre-defined context length, restricting their use in scenarios requiring long inputs. Previous efforts for adapting LLMs to a longer length usually...</li><li><a href="https://arxiv.org/abs/2310.05209">Scaling Laws of RoPE-based Extrapolation</a>: The extrapolation capability of Large Language Models (LLMs) based on Rotary Position Embedding is currently a topic of considerable interest. The mainstream approach to addressing extrapolation with ...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1232960295371018270)** (4 messages): 

- **Sharing Engaging YouTube Content**: A member shared a [YouTube video link](https://www.youtube.com/watch?v=Gwq7smiWLtc), but the content and context of the video was not discussed.
- **Expression of Appreciation**: A simple heart emoji ("<3") was posted by a member, indicating a show of love or appreciation towards another member.
- **Acknowledgement for Archiving Expertise**: Recognition was given to a member for their archiving skills, possibly in relation to maintaining records or documentation.
- **Another YouTube Share**: A second [YouTube video link](https://www.youtube.com/watch?v=nV6eIjnHEH0) was shared by the same member who shared the first link; however, no further details were provided.
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1232964407202676736)** (7 messages): 

- **Llama 3 Breaks Context Limit**: Innovation in Llama 3's context space, reaching **96k context for the 8B model**, using PoSE and continued pre-training. Extended context length achieved by pre-training with 300M tokens and increasing RoPE theta, shared in a [detailed tweet thread](https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg).

- **LoRA Enables Context Enhancement**: The extended context of **Llama 3 8B to 64k+** through PoSE is also available as a LoRA, making it accessible for application to any L3 8B fine-tuned model. You can find this implementation on [Hugging Face](https://huggingface.co/winglian/Llama-3-8b-64k-PoSE/tree/main/adapters).

- **LLama-3 Soars with 160K Context**: A new **LLama-3 8B model with over 160K context** has been released on Hugging Face. Achieved with less than 200M tokens of training and boasts of state-of-the-art (SOTA) long-term context handling, link to the model [here](https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k).

- **WizardLM-2 Unveiled**: The launch of **WizardLM-2**, a suite of next-generation large language models, has been announced, with variants including WizardLM-2 8x22B, WizardLM-2 70B, and WizardLM-2 7B. These models excel in chat, multilingual, reasoning, and agent tasks, with more information available in their [release blog](https://wizardlm.github.io/WizardLM2) and repositories on [Hugging Face](https://huggingface.co/collections/microsoft/wizardlm-2-661d403f71e6c8257dbd598a) and [GitHub](https://github.com/victorsungo/WizardLM/tree/main/WizardLM-2).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/gradient_ai_/status/1783611801130963242?s=46&t=W5S2NyXXy5qiI3uUU8trIQ">Tweet from Gradient (@Gradient_AI_)</a>: We just released the first LLama-3 8B with a context length of over 160K onto Hugging Face! SOTA LLMs can learn to operate on long context with minimal training (&lt; 200M tokens, powered by @CrusoeEn...</li><li><a href="https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from Wing Lian (caseus) (@winglian)</a>: I&#39;m up to 96k context for Llama 3 8B. Using PoSE, we did continued pre-training of the base model w 300M tokens to extend the context length to 64k. From there we increased the RoPE theta to furth...</li><li><a href="https://huggingface.co/dreamgen/WizardLM-2-8x22B">dreamgen/WizardLM-2-8x22B · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1232970491552206868)** (1 messages): 

```html
<ul>
  <li><strong>Announcements Channel Upgrade</strong>: The "Announcements" channel has evolved! It can now be followed and integrated into other servers for streamlined updates.</li>
</ul>
```
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1232942347520573451)** (212 messages🔥🔥): 

```html
<ul>
  <li><strong>Discussing Context Window Expansion:</strong> Members are intrigued by works on language model context window expansion, referencing models with over 8k tokens context, and highlighting the possibilities of extending models into the tens of millions of tokens using techniques such as <strong>PoSE (Positional Space Encoding)</strong> and ring attention.</li>
  <li><strong>Authorization of the AI Safety and Security Board:</strong> A tweet from Andrew Curran (@AndrewCurran_) sparked discussions with the announcement of the AI Safety and Security Board by the Department of Homeland Security, prompting mixed reactions.</li>
  <li><strong>WizardLM and Microsoft’s Model Removals:</strong> Speculations arose when Microsoft's WizardLM repo vanished, with some pointing towards a strategic move by Microsoft in response to its investments in OpenAI and products outperforming their offerings. Members share concerns and emphasize the value of creating archives or backups for such repositories.</li>
  <li><strong>AI Dialogue Systems:</strong> There's a mention of using GPT to generate dialogue and create high-quality training data through "Heated discussion between professors and student." These role-playing dialogues can lead to better question generation or more accurate answers.</li>
  <li><strong>LLMs Frontend Choices:</strong> Multiple tools and interfaces for working with Large Language Models are brought up, including <strong>Librechat, Lm studio, and OpenRouter</strong>. Members seem to be exploring various options for the best tool fit.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/andrewcurran_/status/1783857762252001715?s=46">Tweet from Andrew Curran (@AndrewCurran_)</a>: This morning the Department of Homeland Security announced the establishment of the Artificial Intelligence Safety and Security Board. The 22 inaugural members include Sam Altman, Dario Amodei, Jensen...</li><li><a href="https://arxiv.org/abs/2404.16710">Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding</a>: We present LayerSkip, an end-to-end solution to speed-up inference of large language models (LLMs). First, during training we apply layer dropout, with low dropout rates for earlier layers and higher ...</li><li><a href="https://lluminous.chat/">lluminous</a>: no description found</li><li><a href="https://librechat-librechat.hf.space">LibreChat</a>: no description found</li><li><a href="https://rocky-muscle-755.notion.site/What-happened-to-Wizard-LM2-a247e09244d0483cbb02c1587b357c9d">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://huggingface.co/LargeWorldModel/LWM-Text-1M">LargeWorldModel/LWM-Text-1M · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.16811">Make Your LLM Fully Utilize the Context</a>: While many contemporary large language models (LLMs) can process lengthy input, they still struggle to fully utilize information within the long context, known as the lost-in-the-middle challenge. We ...</li><li><a href="https://arxiv.org/abs/2401.02415">LLaMA Pro: Progressive LLaMA with Block Expansion</a>: Humans generally acquire new skills without compromising the old; however, the opposite holds for Large Language Models (LLMs), e.g., from LLaMA to CodeLLaMA. To this end, we propose a new post-pretra...</li><li><a href="https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main">Anthropic/hh-rlhf at main</a>: no description found</li><li><a href="https://huggingface.co/PY007/EasyContext-1M-Llama-2-7B">PY007/EasyContext-1M-Llama-2-7B · Hugging Face</a>: no description found</li><li><a href="https://github.com/jzhang38/EasyContext/blob/6dfd77e8f2a68bf522be8889e60c98c8e816e329/easy_context/zigzag_ring_attn/monkey_patch.py#L98">EasyContext/easy_context/zigzag_ring_attn/monkey_patch.py at 6dfd77e8f2a68bf522be8889e60c98c8e816e329 · jzhang38/EasyContext</a>: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware. - jzhang38/EasyContext</li><li><a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k">gradientai/Llama-3-8B-Instruct-262k · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/crusoeai/Llama-3-8B-Instruct-262k-GGUF">crusoeai/Llama-3-8B-Instruct-262k-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/jzhang38/EasyContext">GitHub - jzhang38/EasyContext: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware.</a>: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware. - jzhang38/EasyContext</li><li><a href="https://huggingface.co/datasets/MaziyarPanahi/WizardLM_evol_instruct_V2_196k">MaziyarPanahi/WizardLM_evol_instruct_V2_196k · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1232958591955112028)** (50 messages🔥): 

- **Val Loss Not Indicative of Performance**: One user mentioned tossing out validation loss checking from their process, stating that they found **no correlation between validation loss and downstream performance**, and that validation checks add to compute costs without providing value.
- **Training on Synthetic Data**: Another user inquired about strategies for generating **diverse synthetic data**. Helpful responses included using the [Distilabel framework](https://github.com/argilla-io/distilabel) and examining certain papers like WizardLM and Airoboros for insights.
- **Long Context Management in LLMs**: The effectiveness of context management techniques in large language models was discussed, with Llama 3 being highlighted for its performance. Some mentioned methods involve **rope scaling** and the use of the **PoSE technique** to extend context length.
- **Cost-Performance Considerations in Model Training**: A comparison was shared regarding the **cost of training epochs** using Hermes 2 dataset on Llama-3 70B with qLora—4 epochs costing $2,368 versus $41,440 for 50 epochs, for potential minor performance improvements.
- **Exploring MoE with Llama 3**: One user proposed creating a 'clown car' mixture of experts (MoE) with Llama 3, drawing parallels to the Gemma MoE model. The user speculated on the potential gains from combining several 8B models and using **DPO/ORPO techniques** to enhance outputs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/Crystalcareai/gemmoe-65f11f4922af97ebe9943591">GemMoE - a Crystalcareai Collection</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=kuvFoXzTK3E&t=4447s)">Prof. Chris Bishop&#39;s NEW Deep Learning Textbook!</a>: Professor Chris Bishop is a Technical Fellow and Director at Microsoft Research AI4Science, in Cambridge. He is also Honorary Professor of Computer Science a...</li><li><a href="https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from Wing Lian (caseus) (@winglian)</a>: I&#39;m up to 96k context for Llama 3 8B. Using PoSE, we did continued pre-training of the base model w 300M tokens to extend the context length to 64k. From there we increased the RoPE theta to furth...</li><li><a href="https://github.com/argilla-io/distilabel">GitHub - argilla-io/distilabel: ⚗️ distilabel is a framework for synthetic data and AI feedback for AI engineers that require high-quality outputs, full data ownership, and overall efficiency.</a>: ⚗️ distilabel is a framework for synthetic data and AI feedback for AI engineers that require high-quality outputs, full data ownership, and overall efficiency. - argilla-io/distilabel</li><li><a href="https://distilabel.argilla.io/latest/">Getting started</a>: Distilabel is an AI Feedback (AIF) framework for building datasets with and for LLMs.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/)** (1 messages): 

deoxykev: Does anybody know of work extending moondream’s input size?
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1233096647987302521)** (3 messages): 

- **Exploring Dataset Synthesis with Distilabel**: A member mentions utilizing **Argilla's Distilabel** for dataset synthesis and finds it valuable, despite some missing features. Examples on generating function calling and JSON/pydantic data can be found in the [distilabel-workbench repository](https://github.com/argilla-io/distilabel-workbench).

- **Synthesis Simplified for Single Documents**: For single document data synthesis, the method appears straightforward after deciding on a specific structure or template.

- **Complex Challenges for Multi-hop Synthesis**: Multi-document or multi-hop fact synthesis is acknowledged as more complex, yet potentially manageable with **Raptor** and *smart prompting* or *agentic use of database*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/argilla-io/distilabel-workbench/tree/main/projects/function_calling_dataset">distilabel-workbench/projects/function_calling_dataset at main · argilla-io/distilabel-workbench</a>: A working repository for experimental pipelines in distilabel - argilla-io/distilabel-workbench</li><li><a href="https://github.com/argilla-io/distilabel-workbench/tree/main/projects/json_schema_generating_dataset">distilabel-workbench/projects/json_schema_generating_dataset at main · argilla-io/distilabel-workbench</a>: A working repository for experimental pipelines in distilabel - argilla-io/distilabel-workbench
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1232962183823097946)** (68 messages🔥🔥): 

- **World-Sim Twitch Channel Blocked by Shutdown**: A user geared up to launch their Twitch channel for a World-Sim session but was thwarted as World-Sim was shut down due to 4chan abusers, leaving the Twitch livestream *slightly empty*.

- **Exploring Websim's Capabilities**: Members are discussing a webpage simulator, Websim, that is drawing interest for its ability to execute CLI commands similar to the defunct World-Sim and simulate entire web pages. Shareable links to the simulations have been exchanged, with one such link being https://websim.ai/c/p3pZvmAYbsRT2hzBz.

- **Anticipation for World-Sim's Return**: Users speculate on the status and nature of World-Sim's return, discussing potential investment in the platform. One user announces that people from the channel will be invited to test World-Sim for free before it goes live again, while another provides clarity by mentioning it will be a pay-for-tokens system.

- **AI Companionship Through Websim**: An AI named EVA, designed to be a human companion, has been shared among users, highlighting Websim's application for creating simulated interactions with AIs. The sharing of AI profiles, such as EVA, is met with enthusiasm as users look forward to engaging with these virtual entities.

- **Tabletop Simulator Curiosity and Participation**: The conversation touches upon a tabletop simulator in development, with users expressing interest in participating and curiosity about how it might function. One user encapsulates the concept poetically with the phrase: "Sim within a Sim // Regression et Recursion // Limits we Limits."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://websim.ai/c/oFskF68gjd7njVn0E">New Conversation - Eigengrau Rain</a>: no description found</li><li><a href="https://tenor.com/view/jordi-baste-tv3-no-pot-ser-com-robot-gif-16057126">Jordi Baste Tv3 GIF - Jordi Baste Tv3 No Pot Ser - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://websim.ai/c/p3pZvmAYbsRT2hzBz">EVA - Intraneural Cybernetic Interface
  style</a>: no description found</li><li><a href="https://websim.ai/c/hCNgw78IbjiJHLTk3">EVA Instance: ex-0101</a>: no description found
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1233055577836490752)** (170 messages🔥🔥): 

- **Apple Dives into Open Source**: Apple has released **OpenELM**, an efficient language model family, marking a shift from their traditional proprietary approach. The models are available on Hugging Face, trained with a layer-wise scaling strategy, offering variations from 270M to 3B parameters, and [can be found here](https://huggingface.co/apple/OpenELM).

- **Philosophy of Intelligence and Sentience**: Users discussed the nuances of **consciousness** and **sentience**, with varying interpretations influenced by language and cultural differences. One user expressed that sentience might be about motivation and guidance by emotions, while consciousness is about understanding knowledge.

- **Temporal Awareness in AI**: There was a philosophical debate about whether current models have **temporal awareness** or if intelligence and consciousness are discrete and decoupled from temporal constraints. The conversation touched on the complexity of identity in the context of neural networks and subjective experiences.

- **AI Voice Assistants on the Rise**: Users discussed current and upcoming AI voice assistants, highlighting projects like **OpenWakeWords** for creating home voice assistants and the potential of **Gemini** as a Google Assistant alternative. The conversation delved into the technical challenges of interrupting AI mid-speech and the nuanced use of push-to-talk versus voice-activated systems.

- **Confusion Over AI Model Releases and Capabilities**: Users speculated about the release dates of **OpenAI's next models**, compared the coding abilities of current models like **GPT-4** and **Claude**, and even joked about naming conventions for AI models. Some suggested using VPNs to access region-restricted models and shared experiences with voice-to-text transcriptions.

**Link mentioned**: <a href="https://huggingface.co/apple/OpenELM">apple/OpenELM · Hugging Face</a>: no description found

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1233014877920952401)** (42 messages🔥): 

- **Seeking GPT Performance Insights**: A member expressed interest in **GPT-4's** capabilities, comparing its performance to that of **Claude 3** and inquiring about the potential release of a hypothetical **GPT-5**.

- **Custom GPT for Web Browsing Abilities**: Discussion around creating custom GPTs comparable to **Perplexity AI Pro** and **You Pro** for web browsing and summarization, with members sharing their experiences and insights about the difference between **GPT-4** and the dedicated **Web Browser GPT** model.

- **Maximizing Context Windows in Large Document Analysis**: Inquiry into tools for analyzing large text documents, with a member comparing **Claude 3 Haiku** and **Gemini 1.5 Pro** against OpenAI offerings. The conversation touched on how context size may affect model performance, suggesting interest in a future OpenAI model with a larger context window.

- **Resolving Rate Limits and Custom GPT Usage Counts**: A user encountered a rate limit after recalling information from a large PDF using a custom GPT, spurring discussion on the nature and duration of usage caps. Clarifications were offered on the 3-hour rolling usage cap and the possibility of a lower sub-cap for custom GPTs.

- **Understanding the Mechanics of GPT Rate Limits**: Clarification was sought on whether rate limits for custom GPT use are considered as part of the overall **GPT-4 usage cap**. Discussion highlighted the nuances of the 3-hour rolling cap, with advice offered on how to anticipate when message allowances reset.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1233020762965671987)** (20 messages🔥): 

- **Challenges in Custom GPT for SQF in Arma 3**: A user is seeking advice for crafting prompts to build a GPT for coding in SQF language tailored for **Arma 3**. They have uploaded various text files with information, example code, and URLs to assist the GPT model.
  
- **Considerations for Prompt Workflow**: A veteran user recommends crafting a prompt to always scan the provided knowledge, but cautions that the practice may severely limit the programming solution space, and complex toolchain requirements could cause code hallucinations in a 32k context system.

- **AI Models' Performance Debate**: Members engage in a debate questioning whether models like **Claude** and **Llama** can compete with **GPT-3.5** in terms of **logic and tone**, with one pointing out that performance shouldn't just be measured by the ability to answer test questions.

- **Discussion on AI Intelligence Definition**: Some users dispute the definition of intelligence for AI, with opinions varying on whether AI can solve problems it hasn't been trained on and the significance of semantic scoring as a use case where **GPT-4** stands out.

- **Insights on Emergent Abilities in LLMs**: A user reflects on **emergent abilities** in Large Language Models (LLMs), suggesting that after a certain point, quantitative increases in system complexity can lead to qualitative behavior changes not predictable from the system's earlier stages, mentioning the paper *More Is Different* and its relevance to prompt engineering.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1233020762965671987)** (20 messages🔥): 

- **Challenges with Training for SQF Language in GPT**: A member is trying to train GPT for SQF language coding in *Arma 3* using various txt documents but is struggling to create an effective prompt. Other contributors suggest that a system with a larger context size and a better model may be necessary, considering the challenges with the current approach.
  
- **Debating Model Capabilities**: Users engage in a conversation about comparing AI models like Claude, Llama, and GPT-3.5 on parameters like logic and tone, while discussing benchmarks such as SAT question responses or coding problem-solving.
  
- **AI Definition of Intelligence Debated**: A discussion unfolds on defining intelligence for AI, with opinions that even bugs exhibit intelligence by compressing information and that AI can handle unseen logical problems.
  
- **Emergence in Large Language Models (LLMs)**: Emergent abilities in *LLMs* are discussed, characterizing the phenomenon where increasing size in AI systems leads to new qualitative behaviors not predictable from smaller models. This concept is related back to prompt engineering strategies like Chain of Thought (CoT).
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1232957790792843317)** (246 messages🔥🔥): 

- **Stable Diffusion for Steam Games**: Users discussed the usage of Stable Diffusion generated content on Steam. Valve's updated content survey now includes AI disclosure sections, and developers must describe their AI use and implement guardrails for live-generated AI content.
- **Concerns Over Copyrighted Content**: There was a debate regarding the use of public models like Stable Diffusion for content creation and whether the output may include copyrighted materials, alluding to the complexity of using such models on platforms like Steam with stringent copyright rules.
- **Model vs. Lora Creation Queries**: Customluke enquired about creating a model or a Lora from their artwork for generating similar art using Stable Diffusion. Suggestions included using dreambooth for models and kohya_ss for loras.
- **Preferring SD 1.5 Over SDXL**: Some users expressed a preference for SD 1.5 over other versions like SDXL, citing better results, especially with well-handled tagging and training.
- **Suggestions for Improving Image Generation**: Amid conversations about various topics, it was recommended to use different models like Forge and epicrealismXL when unsatisfied with image quality from other generators like ComfyUI.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://adorno.ai">Adorno AI - AI Audio Generation</a>: no description found</li><li><a href="https://videogigagan.github.io">VideoGigaGAN</a>: no description found</li><li><a href="https://suno.com/song/fcedaca6-eaad-4b99-b6ac-aa28feb12d6d">桃花诺三生缘 by @jone_coolke2049 | Suno</a>: 古典，国风，情长 song. Listen and make your own with Suno.</li><li><a href="https://civitai.com/models/153568?modelVersionId=433727">Real Dream - 14 | Stable Diffusion Checkpoint | Civitai</a>: Most realistic LCM 1.5 model currently available on Civitai on April 25, 2024. As I don&#x27;t have very advanced hardware, if you could provide me Buzz...</li><li><a href="https://huggingface.co/lllyasviel/fooocus_inpaint/tree/main">lllyasviel/fooocus_inpaint at main</a>: no description found</li><li><a href="https://tenor.com/view/gimme-what-about-bob-bill-murry-i-need-gif-19552065">Gimme What About Bob GIF - Gimme What About Bob Bill Murry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/Acly/comfyui-inpaint-nodes">GitHub - Acly/comfyui-inpaint-nodes: Nodes for better inpainting with ComfyUI: Fooocus inpaint model for SDXL, LaMa, MAT, and various other tools for pre-filling inpaint &amp; outpaint areas.</a>: Nodes for better inpainting with ComfyUI: Fooocus inpaint model for SDXL, LaMa, MAT, and various other tools for pre-filling inpaint &amp; outpaint areas. - Acly/comfyui-inpaint-nodes</li><li><a href="https://github.com/nerve-sparks/iris_android">GitHub - nerve-sparks/iris_android</a>: Contribute to nerve-sparks/iris_android development by creating an account on GitHub.</li><li><a href="https://github.com/chitradrishti/adlike">GitHub - chitradrishti/adlike: Predict to what extent an Image is an Advertisement.</a>: Predict to what extent an Image is an Advertisement. - chitradrishti/adlike</li><li><a href="https://www.youtube.com/shorts/ASkd9Oxk1Eo">1 Mad Dance of the Presidents (ai) Joe Biden 🤣😂😎✅ #stopworking #joebiden #donaldtrump #funny #usa</a>: 🎉 🤣🤣🤣🤣 Get ready to burst into fits of laughter with our latest &quot;Funny Animals Compilation Mix&quot; on the &quot;Funny Viral&quot; channel! 🤣 These adorable and misc...
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1232949364222070814)** (208 messages🔥🔥): 

- **Rollout of BioMistral for Medical LLMs**: An announcement about [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B), a collection of open-source pretrained large language models for medical domains, was shared, highlighting its use of Mistral as its foundation model.
- **Nvidia Adjusts for China**: Discussion of Nvidia's launch of a China-specific graphics card, the RTX 4090D, which was introduced to comply with US export controls, featuring lower power draw and fewer CUDA cores compared to the standard RTX 4090. The situation was elaborated with links to articles from [The Verge](https://www.theverge.com/2023/12/29/24018799/nvidia-4090d-china-slower-us-sanctions) and [Videocardz](https://videocardz.com/newz/nvidia-geforce-rtx-4090-with-blower-type-cooler-is-now-on-sale-in-china).
- **Optimizing Text to Image Models**: Inquiry about configurations for fine-tuning text to image models discussed, with a reference to the [Hugging Face diffusers repository](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) for potential solutions.
- **Utilizing ConversationalRetrievalChain in Gradio**: A user sought advice on implementing a ConversationalRetrievalChain with a Gradio chat interface, sharing their code and expressing a desire to use personal PDFs in the process.
- **Utilizing Quantization for Model Efficiency**: A conversation revolved around the best approaches to using quantization to increase efficiency on a limited VRAM setup, with a suggestion leaning towards using a Q4 or Q5 quantization level for optimal performance while being mindful of offloading to CPU.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/PY007/EasyContext-1M-Llama-2-7B">PY007/EasyContext-1M-Llama-2-7B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/BioMistral/BioMistral-7B">BioMistral/BioMistral-7B · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/legobatman-legojoker-legogoogle-google-joker-gif-13113737">Legobatman Legojoker GIF - Legobatman Legojoker Legogoogle - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/fine-tune-whisper">Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/tree/main/examples/text_to_image">diffusers/examples/text_to_image at main · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers</li><li><a href="https://github.com/huggingface/optimum-graphcore/blob/main/notebooks/whisper_finetuning.ipynb">optimum-graphcore/notebooks/whisper_finetuning.ipynb at main · huggingface/optimum-graphcore</a>: Blazing fast training of 🤗 Transformers on Graphcore IPUs - huggingface/optimum-graphcore</li><li><a href="https://huggingface.co/models?pipeline_tag=text-classification&library=transformers.js&sort=trending&search=xenova">Models - Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py">transformers/examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://www.youtube.com/watch?v=jtu-G7vA9SU&ab_channel=PaulMeekin">Freedom GPT pitch</a>: no description found</li><li><a href="https://deci.ai/blog/model-merging-moe-frankenmerging-slerp-and-task-vector-algorithms/">Model Merging: Comparing Methods</a>: Explore and compare model merging methods like frankenmerging, SLERP, MoE, and task vectors, highlighting their benefits and challenges.</li><li><a href="https://github.com/jzhang38/EasyContext">GitHub - jzhang38/EasyContext: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware.</a>: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware. - jzhang38/EasyContext</li><li><a href="https://rubiks.ai">Rubik's AI - AI research assistant & Search Engine</a>: no description found</li><li><a href="https://www.theverge.com/2023/12/29/24018799/nvidia-4090d-china-slower-us-sanctions">Nvidia is releasing a slower RTX 4090 in China to comply with US restrictions</a>: The US doesn’t allow Nvidia to sell the RTX 4090 in China.</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-4090-with-blower-type-cooler-is-now-on-sale-in-china">NVIDIA GeForce RTX 4090 with blower-type cooler is now on sale in China - VideoCardz.com</a>: GeForce RTX 4090 with blower cooler It goes without saying but RTX 4090 GPU with its 450W TDP is not something one would expect to get a blower-type cooler. Yet, such card does exists. The card we rep...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1233070949155475538)** (3 messages): 

- **Mistral 7B Fine-tuning File Upload Conundrum**: A member mentioned they are attempting to fine-tune **Mistral 7B** but observed that files get uploaded during the process, which wasn't the case before.
- **Seeking Candle Documentation Comparable to Transformers**: A member who has experience with the **Transformers library** expressed interest in **Candle**, inquiring about comprehensive documentation similar to what is available for Transformers, due to performance issues with Python in production environments.
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1232934063028047963)** (8 messages🔥): 

- **Exploring the 6G Future**: A member shared an [arXiv paper](https://arxiv.org/abs/1904.11686) that discusses the intersection of wireless communications and AI, envisioning 6G networks to support ubiquitous **AI services** and how AI will assist in designing and optimizing 6G networks.

- **Journey into Computer Vision with HuggingFace**: A course on **community-driven computer vision** has been started by a member, which aims to cover everything from basics to advanced topics in the field. The course is accessible on [HuggingFace's learning platform](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome).

- **Reinforcement Learning Aided by Human Insight**: An [awesome-RLHF GitHub repository](https://github.com/opendilab/awesome-RLHF), a curated list of reinforcement learning resources incorporating human feedback, is being continually updated and shared in the community.

- **Eagerness for Computer Vision Learning Expressed**: A member inquired about the quality of computer vision courses, referring to the [HuggingFace's learning platform](https://huggingface.co/learn) which offers education on applying ML libraries and models in the computer vision domain.

- **Phi3 Red Teaming Report Shared for Insights**: Insights and takeaways from the Phi3 red teaming exercise were discussed, with a link provided to a [LinkedIn post](https://www.linkedin.com/posts/divyanshuusingh_phi3-red-teaming-report-activity-7189692710952304640-WsgF) containing more detailed information.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn">Hugging Face - Learn</a>: no description found</li><li><a href="https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome">Welcome to the Community Computer Vision Course - Hugging Face Community Computer Vision Course</a>: no description found</li><li><a href="https://arxiv.org/abs/1904.11686">The Roadmap to 6G -- AI Empowered Wireless Networks</a>: The recent upsurge of diversified mobile applications, especially those supported by Artificial Intelligence (AI), is spurring heated discussions on the future evolution of wireless communications. Wh...</li><li><a href="https://github.com/opendilab/awesome-RLHF">GitHub - opendilab/awesome-RLHF: A curated list of reinforcement learning with human feedback resources (continually updated)</a>: A curated list of reinforcement learning with human feedback resources (continually updated) - opendilab/awesome-RLHF
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1232955568843522101)** (9 messages🔥): 

- **Read-Later Content, Now a Daily Digest**: A user introduced an app called [Collate.one](https://collate.one/newsletter) that transforms read-later content into a bite-sized daily newsletter, inviting others to try it and share their feedback.
- **Speedy High-Resolution Image Generation**: A space has been created on Hugging Face for [generating 4k images in 5 seconds](https://huggingface.co/spaces/KingNish/Instant-Image), duplicating functionality from another space called PixArt-alpha/PixArt-Sigma.
- **Troubleshooting Real-Time on New Space**: Following an error reported by a user trying the image generation space with a specific prompt, the creator asked users to try again suggesting the issue had been addressed.
- **AI Community Highlights in Brazilian Portuguese**: Community Highlights #54 has been translated into Brazilian Portuguese with a released [video](https://www.youtube.com/watch?v=A9qPlYVeiOs) and a related [blog post](https://iatalk.ing/destaques-da-comunidade-54/), meant to share the open-source AI community updates with Portuguese-speaking individuals.
- **Improvements to Docker XTTS Streaming Server**: Enhanced the original XTTS streaming server, adding features like speech temperature control and batch processing, showcased in a [GitHub repository](https://github.com/rrg92/docker-xtts) while emphasizing it as a learning opportunity for Gradio and speech models.
- **Mega Small Embed SynthSTS Model for Long Documents**: A user posted about their [Sentence Transformer Model](https://huggingface.co/BEE-spoke-data/mega-small-embed-synthSTS-16384-v1), which produces embeddings for long text documents and is pre-trained for a context length of 16,384. The model could be particularly useful for clustering and semantic search tasks and might see future updates.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/Instant-Image">Instant Image - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://huggingface.co/BEE-spoke-data/mega-small-embed-synthSTS-16384-v1">BEE-spoke-data/mega-small-embed-synthSTS-16384-v1 · Hugging Face</a>: no description found</li><li><a href="https://github.com/rrg92/docker-xtts">GitHub - rrg92/docker-xtts: Projeto docker para ser usado com o XTTS Streaming Server</a>: Projeto docker para ser usado com o XTTS Streaming Server - rrg92/docker-xtts</li><li><a href="https://www.youtube.com/watch?v=A9qPlYVeiOs">Destaques da Comunidade #54</a>: Mais um vídeo com os destaques da comunidade open source de IA do mundo! Post: https://iatalk.ing/destaques-da-comunidade-54/Está bem divertido fazer estes v...</li><li><a href="https://iatalk.ing/destaques-da-comunidade-54/">🤗Destaques da Comunidade #54</a>: Olá pessoal, este é o Destaques da Comunidade #54, que saiu no dia 18/04/2024O conteúdo original pode ser conferido em: Segue a lista comentada e o vídeo logo em seguida!Aproveita pra se inscrever …
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1233199775252349020)** (2 messages): 

- **Seeking Table-QA Vision Models**: A member asked for recommendations on **vision models** capable of performing question-answering on complex tables. They've tried **IDEFICS2** and **GEMINI 1.5 pro** but encountered issues with inaccurate values.
- **Security Concerns with COCO Dataset**: A member expressed concern regarding the official COCO datasets being hosted on an **HTTP** connection, hinting at potential security implications.
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1232953833441988648)** (6 messages): 

- **Call for a more Code-Centric Resource**: A member praised an existing resource, **Dr. Valeriy**, as a good model for creating tools that have direct links to code implementations, sharing the link to [valerman/awesome-conformal-prediction](https://github.com/valeman/awesome-conformal-prediction) for reference.
- **Seeking SFFTrainer Training Details**: A user sought a detailed understanding of the **SFFTrainer's** training process, particularly what components of a prompt are initially fed to the LLM and whether the LLM is restricted to the number of tokens in the provided answer.
- **Looking for Open Source STT Web Frontends**: A community member inquired about any available open-source **speech-to-text (STT)** web frontends.
- **Seeking Copyright Information for safetensors**: A member questioned the missing copyright details for **safetensors**, pointing out that, while the license is Apache, there are no year or ownership details in the [LICENSE file](https://github.com/huggingface/safetensors/blob/main/LICENSE).
- **Celebrating the Launch of Trustworthy Language Model**: The release of v1.0 of the **Trustworthy Language Model (TLM)** was announced, boasting a feature to combat LLM hallucinations with a confidence score system. Users were invited to try out the TLM and share findings via the [playground](https://tlm.cleanlab.ai/), with further insights available in a [blog post](https://cleanlab.ai/blog/trustworthy-language-model/) and a detailed [tutorial](https://help.cleanlab.ai/tutorials/tlm/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tlm.cleanlab.ai/">TLM Playground</a>: Try out Cleanlab&#x27;s Trustworthy Label Model (TLM) in your browser.</li><li><a href="https://cleanlab.ai/blog/trustworthy-language-model/">Overcoming Hallucinations with the Trustworthy Language Model</a>: Announcing Cleanlab&#x27;s Trustworthy Language Model. TLM overcomes hallucinations, the biggest barrier to productionizing GenAI, by adding a trust score to every LLM output.</li><li><a href="https://help.cleanlab.ai/tutorials/tlm/">Trustworthy Language Model (TLM)</a>: A more reliable LLM that quantifies trustworthiness for every output and can detect bad responses.</li><li><a href="https://github.com/huggingface/safetensors/blob/main/LICENSE">safetensors/LICENSE at main · huggingface/safetensors</a>: Simple, safe way to store and distribute tensors. Contribute to huggingface/safetensors development by creating an account on GitHub.</li><li><a href="https://github.com/valeman/awesome-conformal-prediction">GitHub - valeman/awesome-conformal-prediction: A professionally curated list of awesome Conformal Prediction videos, tutorials, books, papers, PhD and MSc theses, articles and open-source libraries.</a>: A professionally curated list of awesome Conformal Prediction videos, tutorials, books, papers, PhD and MSc theses, articles and open-source libraries. - valeman/awesome-conformal-prediction
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1233041403341439017)** (6 messages): 

- **Admiring LCM and ip-adapter Synergy**: A member acknowledged that **ip-adapter** and **lcm-lore** work well together, suggesting their effectiveness in combination. Though there are complaints about LCM, the member hopes for improvements from **hyper-sd**.

- **Mystery of the Blue-Tinged Images**: A user faced an issue with their images turning blue using a text-to-image pipeline with multiple controlnet. The reason remained unclear after the brief discussion.

- **Trial and Error with torch.compile**: An attempt to use **torch.compile** during training was made, initially causing the program to hang during the first forward pass. Eventually, the process completed, taking around 10 minutes.

- **Forward Pass Speed Boost Using torch.compile**: Post initial hurdles, the member noted a significant speed improvement in the forward pass, though the backward pass speed remained unaffected by using **torch.compile**.
  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1233201020428091463)** (1 messages): 

```html
<ul>
  <li><strong>Gradio Bolsters Custom Component Capabilities</strong>: Version 4.28.0 of Gradio introduces significant enhancements for custom components, including Tailwind styling, support for any vite plugin and preprocessors, and a refined custom component CLI that utilizes the vanilla Gradio SDK in spaces.</li>
  <li><strong>Streamlined Development and New Features</strong>: Additional features accompany the custom components upgrade, such as setting a maximum upload size, persistent reloads in dev mode to maintain front-end state, and a re-organized documentation to better represent the Gradio ecosystem.</li>
  <li><strong>Comprehensive Release with More Improvements</strong>: This is just a highlight of the update; more details can be found in the full changelog available on the Gradio website.</li>
</ul>
```
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1232958925477773313)** (52 messages🔥): 

- **Clarifying Model Architecture**: In discussing model architectures, one user clarified how some papers describe attention and feedforward neural networks (FFN) as parallel operations, referencing PaLM as an example where **models use parallel attention + FFN**.
- **Decoding the Pile dataset hash**: A member shared the **hash values for the Pile dataset**, providing a list with [hashes linked to EleutherAI](https://www.eleuther.ai/hashes) for those seeking to use the dataset in various JSON files.
- **Receptive Field Mechanics in Transformers**: A conversation on **sliding window attention** mentioned how the mask limits the scope of attention and compared it to the functioning of convolutions regarding effective receptive fields.
- **Exploring Layered Learning and Attention Structures**: Participants discussed the potential for interleaving RNN-type layers or using dilated windows for transformers to handle **longer sequence lengths** effectively.
- **New PyTorch Library for Large Model Training**: A user shared a [link to the GitHub repository](https://github.com/pytorch/torchtitan) for a new PyTorch library named **torchtitan**, meant for large model training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.eleuther.ai/hashes">Hashes &mdash; EleutherAI</a>: no description found</li><li><a href="https://github.com/pytorch/torchtitan">GitHub - pytorch/torchtitan: A native PyTorch Library for large model training</a>: A native PyTorch Library for large model training. Contribute to pytorch/torchtitan development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1232934258192941056)** (144 messages🔥🔥): 

- **Linear Attention Breakdown Explained**: A member provided a thorough explanation of how linear attention works by separating \( QK^T V \) into \( Q(K^T V) \) for inference. It was clarified that this approach is linear with respect to sequence length and maintains constant memory usage, unlike softmax attention which grows over time.

- **Benchmarking Beyond Hypothesized Formats**: A member reported "benchmarking" phi-3-mini-128k against other models, suggesting its performance might be on par with Llama-3-8B. The discussion evolved around whether the pre-training data significantly influences post-training performance and what constitutes a "base" in the context of AI models like phi.

- **Deep Dive into Delta Rule Practicality**: Conversations unfolded regarding the practicality and parallelization of delta rule linear attention, with one member sharing insights from a [blog post on manifestai](https://manifestai.com/blogposts/faster-after-all/). It was noted that delta rule linear attention is more organized but less parallelizable, potentially slowing down training.

- **Needle in the Haystack Tests Scrutinized**: Users questioned the efficacy of "needle in the haystack" tests for long-context language models, suggesting real-world application and personal testing are more indicative performance benchmarks. There is skepticism about how such a test accounts for the semantic similarity between the "needle" and its surrounding context.

- **Masking User Prompt Loss During SFT**: There was curiosity whether masking the user prompt loss during supervised fine-tuning (SFT) on language models had been systematically studied. While a common practice, members noted the absence of research on its effects and discussed potential gains from including prompt loss in SFT.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.15574">Retrieval Head Mechanistically Explains Long-Context Factuality</a>: Despite the recent progress in long-context language models, it remains elusive how transformer-based models exhibit the capability to retrieve relevant information from arbitrary locations within the...</li><li><a href="http://arxiv.org/abs/2404.15574">Retrieval Head Mechanistically Explains Long-Context Factuality</a>: Despite the recent progress in long-context language models, it remains elusive how transformer-based models exhibit the capability to retrieve relevant information from arbitrary locations within the...</li><li><a href="https://openreview.net/forum?id=Hygxb2CqKm">Stable Recurrent Models</a>: Stable recurrent models can be approximated by feed-forward networks and empirically perform as well as unstable models on benchmark tasks.</li><li><a href="http://arxiv.org/abs/2404.03683">Stream of Search (SoS): Learning to Search in Language</a>: Language models are rarely shown fruitful mistakes while training. They then struggle to look beyond the next token, suffering from a snowballing of errors and struggling to predict the consequence of...</li><li><a href="https://manifestai.com/blogposts/faster-after-all/">Manifest AI - Linear Transformers Are Faster After All</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

main.ai: https://twitter.com/sen_r/status/1783497788120248431
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1232947816624685098)** (21 messages🔥): 

- **Few-Shot Performance Queries for GSM8K**: A question about the number of few-shot examples (**num_fewshot**) for **GSM8K** arose to match the **Hugging Face leaderboard**, suggesting that the number should be *5*.

- **VLLM Integration Blockers Discussed**: A user inquired about obstacles to upgrading **VLLM** to the latest version. The discussion clarified that **Data Parallel (DP)** was a potential blocker, but **Tensor Parallel (TP)** should be fine.

- **Invitation for Filter Registry Function PR**: A new member noticed an absence of a `register_filter` function for `FILTER_REGISTRY` in **lm_eval**. The user was encouraged to submit a PR to address the issue.

- **Musings on the Brier Score Function**: A member encountered an issue with the Brier score function for **ARC evaluation** in the **lm-evaluation-harness** due to an anomaly in the data. It was suggested that the Brier score function be adjusted to avoid errors despite the dataset's inconsistency.

- **Progress Inquiry on Chat Templating Branch**: A user queried the status of an active branch for chat templating on *Hailey's branch*, last updated two months ago, expressing interest in the progress toward functionality.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/3196e907fa195b684470a913c7235ed7f08a4383/lm_eval/api/metrics.py#L124.">lm-evaluation-harness/lm_eval/api/metrics.py at 3196e907fa195b684470a913c7235ed7f08a4383 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1745">add task for mmlu evaluation in arc multiple choice format by jonabur · Pull Request #1745 · EleutherAI/lm-evaluation-harness</a>: This PR adds the mmlu_arc_style task that presents the MMLU questions in the same manner as the arc evals (loglikelihood for the answer as a continuation, rather than selecting the letter for the c...
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1233094280797753376)** (2 messages): 

- **Identifying Provider Issues**: A major **Mixtral 8x7b provider** was found to send blank responses and has been temporarily removed. Future solutions to *auto-detect* such issues are being considered.

- **Soliloquy 8B Switches to Paid Model**: [Soliloquy 8B](https://openrouter.ai/models/lynn/soliloquy-l3) shifted to a paid model, costing **$0.1 per 1M tokens** as per the latest update. Further details can be found on the provided Discord channel link.

**Link mentioned**: <a href="https://openrouter.ai/models/lynn/soliloquy-l3)">Lynn: Llama 3 Soliloquy 8B by lynn | OpenRouter</a>: Soliloquy-L3 is a fast, highly capable roleplaying model designed for immersive, dynamic experiences. Trained on over 250 million tokens of roleplaying data, Soliloquy-L3 has a vast knowledge base, ri...

  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1233393770071064597)** (1 messages): 

- **Announcing AI Breakthrough on LinkedIn**: A message included a LinkedIn post from *fprime-ai* talking about a technological breakthrough with their **DBRX AI** system. The post can be accessed and read in detail [here](https://www.linkedin.com/posts/fprime-ai_fprimeailabs-dbrx-ai-activity-7189599191201980417-Te5d?utm_source=share&utm_medium=member_desktop).
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1232976027555336284)** (215 messages🔥🔥): 

- **Choosing the Best Model for Roleplay Creativity**: Members discussed the most suitable open-source models for creative plot creation in role-play. Recommendations included **WizardLM2 8x22B**, command-based models for creative writing, and **Mixtral 8x22B**, with one user emphasizing Mixtral's exceptional degree of creativity.

- **Debate Over GPT Turbos and Microsoft's Wizard LM**: An extensive debate erupted regarding **Microsoft's impact** on the **Wizard LM** project, with some suggesting the company halted older models and others arguing over the performance of GPT-4 "Turbo" models. A member produced evidence by linking a detailed [summary of the incident](https://rocky-muscle-755.notion.site/).

- **Model Performance and Hosting Costs Explored**: Members evaluated various models' performance, such as **GPT-4, Llama 3, and WizardLM**, while also discussing hosting costs and sustainability of current pricing, with calculations estimating costs per million tokens.

- **Concerns over Model Switching and API Logging**: Users expressed concerns about the transparency of model switching in OpenRouter and the logging of API calls by providers, with some having hesitations about using models like Lynn: Llama 3 Soliloquy 8B.

- **OpenRouter Usage, Features, and Limitations**: Discussion about OpenRouter covered topics from enabling **system message mappings** to **playground response expansions**. Users also inquired about handling **HTTP 524 errors** and avoiding **negative balances** while using paid LLMs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/7skpsI0">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://rocky-muscle-755.notion.site/What-happened-to-WLM2-a247e09244d0483cbb02c1587b357c9d?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://openrouter.ai/models?q=free">OpenRouter</a>: Browse models on OpenRouter
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1233088063849369692)** (4 messages): 

- **create-llama v0.1 Launches**: The **create-llama** v0.1 release introduces easy setup for RAG applications with new features like **@ollama support** and new **vector database** integrations, facilitating the use of llama3 and phi3 models. The update was announced via [a tweet with more details on the improvements](https://twitter.com/llama_index/status/1783528887726817653).
  
- **Build RAG Apps with Qdrant and LlamaParse**: A step-by-step tutorial highlighting how to build RAG applications using **LlamaParse**, **@JinaAI_ embeddings**, **@qdrant_engine vector storage**, and **Mixtral 8x7b** was detailed in another tweet. Interested developers can access the full tutorial [here](https://twitter.com/llama_index/status/1783601807903863184).

- **LlamaParse Webinar by KX**: KX systems is organizing a **webinar** on maximizing the utility of **LlamaParse** for complex document parsing, table and image extraction, and natural language preprocessing. The event details are available in [this twitter post](https://twitter.com/llama_index/status/1783622871614664990).

- **AWS Workshop Featuring LlamaIndex**: **@llama_index** teams up with AWS to offer workshop materials on building LLM apps with AWS, explaining how to use services such as S3, AWS Bedrock LLMs, and embedding storage in conjunction with LlamaParse and LlamaCloud. Workshop details are summarized in [this tweet](https://twitter.com/llama_index/status/1783877951278432733).
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1232982378058874891)** (117 messages🔥🔥): 

- **Exploring RAG Implementations**: Members discussed the effectiveness of simple and advanced RAG (Retrieval-Augmented Generation) pipelines. It was suggested to explore more complex RAG solutions like sentence-window retrieval or auto-merging retrieval for improved results, and a video was linked for learning how to set up these pipelines ([Lesson on Advanced RAG](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/5/auto-merging-retrieval)).

- **Technical Troubleshooting for Chatbot Implementations**: There was a conversation about implementing a chatbot with gpt-4-vision-preview, where issues arose with the backend not supporting image uploads. A member found a solution when adding images as part of the content, rather than using `additional_args`.

- **Configuring and Using Pydantic with LlamaIndex**: A user asked for information on getting structured Pydantic output from chat completions, and another raised an issue with Pydantic imports causing type checking errors. The suggestion was to use v1 imports directly or wait for LlamaIndex to phase out v1 support.

- **Query Pipeline Configuration Queries**: Several users discussed the nuances of configuring query pipelines, mentioning issues with JSON output from GPT-4 within a pipeline and exploring how to format outputs at intermediate steps effectively. It was outlined that GPT-4 turbo does not support JSON output, while GPT-3.5 turbo does allow for JSON mode ([GPT JSON Mode Documentation](https://platform.openai.com/docs/guides/text-generation/json-mode)).

- **Local LLM Setup with LlamaIndex Guidance**: A member sought guidance for using LlamaIndex with local language models to avoid using external APIs. They were directed to the official documentation for a starter example ([Starter Example with Local LLM](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/)). The conversation included troubleshooting import errors and piecing together necessary package installations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/5/auto-merging-retrieval">DLAI - Building and Evaluating Advanced RAG</a>: Introduction · Advanced RAG Pipeline · RAG Triad of metrics · Sentence-window retrieval · Auto-merging retrieval · Conclusion</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai">LlamaIndex - LlamaIndex</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Fighter_(2024_film)">Fighter (2024 film) - Wikipedia</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/openai_json_vs_function_calling/?h=json">OpenAI JSON Mode vs. Function Calling for Data Extraction - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/bridge/pydantic.py">llama_index/llama-index-core/llama_index/core/bridge/pydantic.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/explodinggradients/ragas/issues/557)">Issues · explodinggradients/ragas</a>: Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines - Issues · explodinggradients/ragas</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/output_parsers/pydantic#llama_index.core.output_parsers.PydanticOutputParser>).">Pydantic - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/query_engine/citation/?h=citationqueryengine#llama_index.core.query_engine.CitationQueryEngine))">Citation - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/agent/react_agent#react-agent-a-simple-intro-with-calculator-tools>)">ReAct Agent - A Simple Intro with Calculator Tools - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/community/integrations/guidance#creating-a-guidance-program-to-generate-pydantic-objects>)">Guidance - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/multi_modal/multi_modal_pydantic#using-fuyu-8b-for-pydantic-strucured-output>)">Multi-Modal GPT4V Pydantic Program - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/multi_modal/multi_modal_pydantic#using-minigpt-4-for-pydantic-strucured-output>)">Multi-Modal GPT4V Pydantic Program - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1233016896589140008)** (78 messages🔥🔥): 

- **LAION Discord Reflects on Llama 3 Performance**: Users have mixed opinions regarding Llama 3's performance, with some reporting issues in recalling code correctly, and others suggesting it may be due to configuration problems. While some find it comparable to GPT-4, others see significant room for improvement.

- **Debate Over Proposed Know Your Customer Requirements**: A link to TorrentFreak discusses U.S. proposals for "Know Your Customer" requirements for cloud services and the implications for users. A member shares the Federal Register notice urging feedback before the comment period ends.

- **AI Enthusiasts Seek Like-Minded Communities**: Members of the LAION Discord express interest in joining additional AI/ML-oriented Discord servers for wider community engagement and resource sharing.

- **Tuning AI Models Brings Performance Surprises**: A member working on tuning the DF 400M/450M model discovers significant performance left untapped, emphasizing a low learning rate and improved upscaling of real photos.

- **Critique of the Efficacy of Nightshade and Publishing Protocols**: Users discuss the need for transparency and data regarding the effectiveness of Nightshade, a theoretical discussion about autoencoder limits, and the reluctance to publish findings due to possible adverse reactions from the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://torrentfreak.com/u-s-know-your-customer-proposal-will-put-an-end-to-anonymous-cloud-users-240425/">U.S. &quot;Know Your Customer&quot; Proposal Will Put an End to Anonymous Cloud Users * TorrentFreak</a>: no description found</li><li><a href="https://www.tomshardware.com/tech-industry/us-investigates-chinas-access-to-risc-v-open-source-instruction-set-may-become-new-site-of-us-china-chip-war">US investigates China's access to RISC-V &mdash; open standard instruction set may become new site of US-China chip war</a>: RISC-V seems risky for American lawmakers</li><li><a href="https://arxiv.org/abs/2401.01808">aMUSEd: An Open MUSE Reproduction</a>: We present aMUSEd, an open-source, lightweight masked image model (MIM) for text-to-image generation based on MUSE. With 10 percent of MUSE&#39;s parameters, aMUSEd is focused on fast image generation...</li><li><a href="https://www.federalregister.gov/documents/2024/01/29/2024-01580/taking-additional-steps-to-address-the-national-emergency-with-respect-to-significant-malicious">Federal Register :: Request Access</a>: no description found</li><li><a href="https://huggingface.co/datasets/fal-ai/imgsys-results">fal-ai/imgsys-results · Datasets at Hugging Face</a>: no description found</li><li><a href="https://mediabiasfactcheck.com/torrentfreak-bias/,">TorrentFreak - Bias and Credibility</a>: LEAST BIASED These sources have minimal bias and use very few loaded words (wording that attempts to influence an audience by appeals to emotion or</li><li><a href="https://en.wikipedia.org/wiki/TorrentFreak">TorrentFreak - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1233160752060305501)** (12 messages🔥): 

- **Revolutionizing Visual Representation Learning**: A novel weakly supervised pre-training method for vision models was highlighted, which categorizes pre-training on image-text data as a classification task. This approach has achieved a training speed that is **2.7 times faster** than traditional contrastive learning, without compromising the quality of representations, as detailed in this [arXiv paper](https://arxiv.org/abs/2404.15653).

- **Simple Yet Effective**: Building on the previous point, the method's success was attributed to detecting concepts from alt-text and training a multilabel classifier. It was noted that this led to the model performing comparably to **CLIP** in zero-shot scenarios and with greatly improved training efficiency.

- **The Cost of Contrast**: In a conversation about the efficacy of text encoders and contrastive learning, it was pointed out that contrastive learning, especially while aligning text encoders, is costly. The approach can incur extra computational expenses when dealing with noisy alt text.

- **Fast Yet Still Lengthy**: A humorous comment was shared acknowledging that while a 2.7x speed increase in training was significant, the overall process remains time-intensive. This reflects a realistic perspective on the improvements in speed.

- **Exploring VAST Possibilities**: Interest was expressed in finetuning **VAST**, a Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset, with a link provided to the project's [GitHub repository](https://github.com/txh-mercury/vast).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.16710">Layer Skip: Enabling Early Exit Inference and Self-Speculative Decoding</a>: We present LayerSkip, an end-to-end solution to speed-up inference of large language models (LLMs). First, during training we apply layer dropout, with low dropout rates for earlier layers and higher ...</li><li><a href="https://arxiv.org/abs/2404.15653">CatLIP: CLIP-level Visual Recognition Accuracy with 2.7x Faster Pre-training on Web-scale Image-Text Data</a>: Contrastive learning has emerged as a transformative method for learning effective visual representations through the alignment of image and text embeddings. However, pairwise similarity computation i...</li><li><a href="https://github.com/txh-mercury/vast">GitHub - TXH-mercury/VAST: Code and Model for VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset</a>: Code and Model for VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset - TXH-mercury/VAST
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1232935424461373470)** (70 messages🔥🔥): 

- **Mac Integration with Open Interpreter**: Open Interpreter's **New Computer Update Part II** enhances local functionality with a first local vision model, *native Mac integrations*, improved launch speed, and additional features. Users can run simple commands such as `interpreter --os` to control Mac's native applications directly from Open Interpreter, as detailed on their [change log](https://changes.openinterpreter.com/log/ncu-ii).

- **Vision Model Showcases and Updates**: Community members discussed the **Moondream** tiny vision language model, showcasing [Img2TxtMoondream.py](https://github.com/CodeAKrome/bootcupboard/blob/main/llm-img/Img2TxtMoondream.py), a code demo for vision models. The conversation shifted towards the use of multimodal models like **LLaVA** available on [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.6-34b), highlighting its foundation on **NousResearch/Nous-Hermes-2-Yi-34B**.

- **Resolving Loops and Model Performance**: Participants exchanged tips on optimizing local models to prevent looping behavior, suggesting adjustments such as modifying *temperature settings*, *prompt editing*, or *architectural changes*. The concept of a *frustration metric* was also introduced to adapt a model's behavior after encountering successive loops.

- **Integration Exploration and Error Troubleshooting**: A user pondered the integration of **Open Interpreter** for robot control, specifically with the **Unitree GO2 robodog**, asking for community experience. Others discussed technical issues and solutions for running local servers, such as setting dummy API keys and resolving namespace conflicts in Pydantic model configurations.

- **Open Interpreter 'New Computer Update' Non-Beta Release**: A user confirmed running **Open Interpreter 0.2.5 New Computer Update**, indicating the version that includes the recent enhancements is out of beta. However, there was a question about the update's status change from beta, leading to a clarification through a version check.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/liuhaotian/llava-v1.6-34b">liuhaotian/llava-v1.6-34b · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2112.10003">Image Segmentation Using Text and Image Prompts</a>: Image segmentation is usually addressed by training a model for a fixed set of object classes. Incorporating additional classes or more complex queries later is expensive as it requires re-training th...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/core/computer/display/point/point.py">open-interpreter/interpreter/core/computer/display/point/point.py at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#interactive-chat">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/CodeAKrome/bootcupboard/blob/main/llm-img/Img2TxtMoondream.py">bootcupboard/llm-img/Img2TxtMoondream.py at main · CodeAKrome/bootcupboard</a>: It&#39;s bigger on the inside than the outside! Contribute to CodeAKrome/bootcupboard development by creating an account on GitHub.</li><li><a href="https://changes.openinterpreter.com/log/ncu-ii">Open Interpreter - The New Computer Update II</a>: Official changelog for the open-source Open Interpreter project.</li><li><a href="https://github.com/vikhyat/moondream">GitHub - vikhyat/moondream: tiny vision language model</a>: tiny vision language model. Contribute to vikhyat/moondream development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1233253423114354719)** (2 messages): 

- **Hardware Arrival Sparks Enthusiasm**: A member expressed excitement about receiving their hardware for building, despite missing a *yellow wire and switch*, for which they have spares. Another member showed interest in the building process and looked forward to updates.
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

8i8__papillon__8i8d1tyr: https://www.youtube.com/watch?v=WeH3h-o1BgQ
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1232937070536818738)** (56 messages🔥🔥): 

- **Tweet at the Top**: A member shared their excitement about the *CEO of Hugging Face commenting on their tweet*.
- **Warm Welcome**: New member "*lazeewhalee*" joined the group and was directed to **read the readme** for navigation and guidelines.
- **70B Model Deployment Discussions**: One member mentioned deploying and running the 70B model with **exllama** and inquired about potential issues due to a missing checkpoint.
- **Speculation on AI Benchmarks**: Concerns were raised about the validity of **MMLU scores** and the performance of various models, particularly an 8B model which performed worse than the base llama3 except on MMLU.
- **Insights on Domain-Specific Training**: Members discussed the benefits of fine-tuning Large Language Models (LLMs) for specific domains and shared **[Meditron](https://arxiv.org/abs/2311.16079)** as a thorough paper on the topic. A brief mention of an **upcoming paper** by a member on domain-adaptive continual pre-training was also made.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2311.16079">MEDITRON-70B: Scaling Medical Pretraining for Large Language Models</a>: Large language models (LLMs) can potentially democratize access to medical knowledge. While many efforts have been made to harness and improve LLMs&#39; medical knowledge and reasoning capacities, the...</li><li><a href="https://arxiv.org/abs/2311.08545">Efficient Continual Pre-training for Building Domain Specific Large Language Models</a>: Large language models (LLMs) have demonstrated remarkable open-domain capabilities. Traditionally, LLMs tailored for a domain are trained from scratch to excel at handling domain-specific tasks. In th...</li><li><a href="https://huggingface.co/collections/microsoft/wizardlm-661d403f71e6c8257dbd598a?history=true>">WizardLM - a microsoft Collection</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1233048482760101929)** (3 messages): 

- **Underlying Issue Reported**: A member has confirmed experiencing an issue, although it's unclear what changed to cause it.
- **Expression of Disappointment**: The short response 'sadge' conveys disappointment or sadness over a topic that was likely discussed previously.
- **Inquiry about zzero3 and fft compatibility**: A member is asking if anyone has successfully integrated **zzero3** with **Fast Fourier Transform (fft)**.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1232950442468577321)** (3 messages): 

- **Well Wishes Exchanged**: A user expressed hopes for another's success in their endeavors.
- **Phi3 Fine-tuning Challenges**: A member discussed the difficulties they experienced while fine-tuning the **phi3** model, mentioning it requires a lot of RAM and operates slowly.
- **Technical Troubleshooting**: In response to a technical question, an issue was raised about an `AttributeError` related to the 'TextIteratorStreamer' object not having an 'empty' attribute when using **transformers 4.40.0**.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1233170259020877947)** (10 messages🔥): 

- **Optimizer Compatibility Query**: The search for optimizer compatibility with **FSDP (Fully Sharded Data Parallel)** reveals general support from optimizers like **AdamW** and **SGD**, though some, like `paged_adamw_8bit`, do not support FSDP offloading.
- **Offloading Incompatibility Issue**: There's an issue with `paged_adamw_8bit` optimizer as it's not compatible with **FSDP offloading**, indicating integration challenges between specific optimizers and FSDP features.
- **Searching for Solutions**: In response to an error, efforts are being made to search the **OpenAccess-AI-Collective/axolotl** for alternative optimizers that support **FSDP**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=a50dde20-84d2-463b-8e6d-cc3f55531430)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=f65c9e42-0ffc-4336-9b7b-5722eb092272)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1232954183498334259)** (63 messages🔥🔥): 

- **Toolkit Troubles**: A user experiencing trouble with uploading documents to the **Cohere Toolkit** on Azure received guidance pointing to the paper clip attachment icon for uploads. However, they were still unable to find the upload option and interact with their Cohere-Toolkit instance.
- **Typographic Turmoil**: A user's query about whether the **Cohere typeface** on GitHub was under the MIT license was clarified with the information that the font isn't open-sourced and will be replaced.
- **Model Access and Licensing Clarification**: Cohere's [Command+ models are open weight but not commercially applicable](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus), with weights available for non-commercial use while **training data remains withheld**.
- **Search Engine Explorations for AI**: It was revealed that **Tavily** is used with Cohere-Toolkit; however, **Brave Search API** was suggested as a potentially faster, cheaper, and accurate alternative. A discussion ensued about search engines' cost-efficiency and usage in different contexts.
- **Deployment Dilemmas with Cohere Toolkit**: Users shared insights on deploying the Cohere Toolkit on Azure; one does not need to add a Cohere API key but must select a model deployment option to ensure the application functions properly. Subsequently, a user's difficulty adding tools locally was raised, encountering issues with uploading PDFs and an unsupported version of sqlite3.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tavily.com">Tavily</a>: no description found</li><li><a href="https://docs.trychroma.com/troubleshooting#sqlite">🔍 Troubleshooting | Chroma</a>: This page is a list of common gotchas or issues and how to fix them.</li><li><a href="https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus">C4AI Command R Plus - a Hugging Face Space by CohereForAI</a>: no description found</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/blob/main/src/backend/tools/retrieval/tavily.py">cohere-toolkit/src/backend/tools/retrieval/tavily.py at main · cohere-ai/cohere-toolkit</a>: Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications. - cohere-ai/cohere-toolkit</li><li><a href="https://github.com/searxng/searxng">GitHub - searxng/searxng: SearXNG is a free internet metasearch engine which aggregates results from various search services and databases. Users are neither tracked nor profiled.</a>: SearXNG is a free internet metasearch engine which aggregates results from various search services and databases. Users are neither tracked nor profiled. - searxng/searxng
</li>
</ul>

</div>
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1232944667276869663)** (6 messages): 

- **Debate Over "Hit Piece" Against Cohere**: Members engaged in a heated debate with one party defending *Cohere* against claims of being reckless and another questioning the responsibility in potentially creating "jailbreak" scenarios in AI agents that translate tokens into real-world actions.
- **Confusion over Article Content and Comments**: A member expressed that they no longer recall details from an article they criticized for being a hit piece, highlighting a lack of distinction between chatbot and agent behavior in the discussion.
- **Challenge to Substantiate Criticism**: Upon being asked to substantiate the claim that an article was unfairly discrediting *Cohere*, a member conceded that they could not recall specific reasons offhand, reducing the credibility of the critique.
- **Miscommunication over Remembering Details**: One member ridiculed the reasoning of not being able to remember specifics as a justification for not listing problems with the allegedly malicious article.
- **Expectations of Accountability in Research Dialogue**: The conversation culminated in a statement that if one criticizes research work as malicious, they should be prepared to substantiate their claims, implying a need for accountability when contributing to research discussions.
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1232968803005239296)** (62 messages🔥🔥): 

- **Partnerships Are Key to tinygrad's Success**: Tinygrad aims to win through partnerships and getting others invested in its framework, with coma as the first partner on hardware for tinybox, and possibly collaborating with others on tinybox 2.
- **TinyBox As The Tested Ground for tinygrad**: The tinybox, produced by comma, is considered the most tested environment for tinygrad. George Hotz emphasizes that tinygrad's focus remains on the software, not the hardware production.
- **Tenstorrent Considerations for Partnership**: Despite initial discussions, a partnership with Tenstorrent did not make financial sense as their hardware wasn't competitively efficient or widespread. However, future collaboration is not ruled out if the financial calculus changes.
- **AMMD's MES Limited Utility for tinygrad**: George Hotz notes that the Machine Environment Settings (MES) from AMD are not likely to be useful for tinygrad, despite a helpful writeup from Felix at AMD. The team continues to work on a PM4 backend for their needs.
- **tinygrad MNIST Tutorial and GPU Compatibility**: A [tinygrad MNIST tutorial](https://tinygrad.github.io/tinygrad/mnist/) has been shared, suitable for running in Google Colab with GPU support. Users reported issues with newer NVIDIA hardware, which were resolved by ensuring the latest CUDA libraries were installed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1068976834382925865/1227683281269559418/1232845778259673239">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://tinygrad.github.io/tinygrad/mnist/">MNIST Tutorial - tinygrad docs</a>: no description found</li><li><a href="https://x.com/karpathy/status/1783527854741114981?s=46">Tweet from Andrej Karpathy (@karpathy)</a>: [gif] me trying to read tinygrad code earlier :D  I think the LOC requirements (which are only a proxy for simplicity) led to too great compression. You wouldn&#39;t brag about your .min.js code being...</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/docs/developer.md">tinygrad/docs/developer.md at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://gist.github.com/fxkamd/ffd02d66a2863e444ec208ea4f3adc48">Observations about HSA and KFD backends in TinyGrad</a>: Observations about HSA and KFD backends in TinyGrad - TinyGrad-notes.md
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1233023252415381604)** (6 messages): 

- **Looking Forward to tinygrad 1.0**: [tinygrad's documentation](https://tinygrad.github.io/tinygrad/) was shared, highlighting that the API is stabilizing as it nears its version 1.0 release. It includes a guide to install from source, a [MNIST tutorial](https://tinygrad.github.io/tinygrad/mnist/), [developer docs](https://tinygrad.github.io/tinygrad/developer/), and [externally created tutorials](https://mesozoic-egg.github.io/tinygrad-notes/).

- **tinygrad Takes on Quantile Function**: A member discusses their project, which aims to reimplement the `torch.quantile` function in tinygrad as part of developing sampling algorithms for diffusion models. This process includes an intermediate step of array sorting.

- **tinygrad Docs to Get More Visibility**: In anticipation of the launch of tinygrad 0.9, a member questioned whether links to the [tinygrad documentation](https://tinygrad.github.io/tinygrad/) would be included in the project README. The response indicates an affirmative action will be taken with the 0.9 launch.

**Link mentioned**: <a href="https://tinygrad.github.io/tinygrad/">tinygrad docs</a>: no description found

  

---



**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1783575774085410911>
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1233116804256235520)** (1 messages): 

- **Modular Tackles Software Security Challenges**: Modular is constantly releasing new software and features, presenting security challenges due to vulnerabilities in modern software delivery mechanisms. Highlighting the urgency to prevent attacks, it's noted that by 2024, an estimated [96% of codebases will contain open-source code](https://www.synopsys.com/blogs/software-security/open-source-trends-ossra-report.html).

- **Secure Software Delivery Is Critical for Modular**: The **XZ supply chain attack** has underscored the necessity for strong defenses against supply chain vulnerabilities, making secure software delivery a key focus for Modular since their first release of Mojo.

**Link mentioned**: <a href="https://www.modular.com/blog/preventing-supply-chain-attacks-at-modular">Modular: Preventing supply chain attacks at Modular</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Preventing supply chain attacks at Modular

  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1232950369269583953)** (2 messages): 

- **Downplaying the Importance of Processing Units**: A member expressed the opinion that **processing units** may not be as critical as widely believed.
- **The Geometric Gateway to Quantum Simplification**: Discussing the **amplituhedron**, a member suggested that geometric structures could make complex quantum phenomena like particle scattering amplitudes more **comprehensible**. The use of geometry in optimizing **quantum algorithms and circuit designs** was proposed to potentially reduce complexity and noise.
- **Visualizing Quantum States with Geometry**: The Bloch sphere was mentioned as a way to visualize **quantum gates**' effects on qubits through geometric transformations. Though effective for single qubits, the challenge of scaling to multiple qubits and representing entanglement may require complex hyper-dimensional spaces.
- **Machine Learning as a Decoder of Hyper-Dimensional Spaces**: The member posited that as the visualization of **quantum entanglement** becomes more complex with increased qubit count, **machine learning** might aid in deciphering the intricate graphs that arise.
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1233019327276646450)** (36 messages🔥): 

- **Safe from Harm**: An inquiry about potential irreversible damage to a PC caused by incorrect manual memory management in a custom type with Mojo was addressed with an assurance that the **Operating System** will clean up memory after a process exits, and **Mojo doesn't require manual memory management**.
  
- **Mojo Function Fundamentals**: A Python-to-Mojo conversion discussion revealed two function types, `def` and `fn`, as viable options. The conversation included **code examples and explanations** about the two different function definitions and how to declare variables in Mojo. [Function declarations are described here.](https://docs.modular.com/mojo/manual/functions)

- **Learning Curve for Newcomers**: In a discussion about understanding Mojo's nuances, community members were advised to focus on making their code work first, as it is normal to experience changes in a nascent programming language. The evolution of languages and potential requirement for future refactorings were highlighted as part of the learning process.

- **List Diversity in Mojo Query**: A question regarding **Mojo's ability to handle lists** with mixed data types was fielded, revealing that while possible, the current method is considered "hacky". Illustrative examples were given showing lists containing both integers and strings using `Variant` [as seen in these Gists for Ints and Floats](https://gist.github.com/modularbot/c67e0a66a97aa32314d248f4721f75e2) and for [Ints and StringLiterals](https://gist.github.com/modularbot/1a5beaf165761b55e2f743b3151210eb).

- **Embrace the Mojo Journey**: A newcomer to programming received a warm welcome from the community and a reminder to initially focus on writing code that works. Emphasizing that **mastery comes with practice**, there was an encouragement to stay adaptable and prepared for the evolving landscape of new programming languages.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/pokemon-pikachu-clap-clapping-clapping-gif-gif-13465728489229726846">Pokemon Pikachu GIF - Pokemon Pikachu Clap - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gist.github.com/modularbot/c67e0a66a97aa32314d248f4721f75e2">playground.mojo</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/modularbot/1a5beaf165761b55e2f743b3151210eb">playground.mojo</a>: GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1232977987025240064)** (5 messages): 

- **Mojo vs. Rust: A Comparative Overview**: A [discussion on lobste.rs](https://lobste.rs/s/a3yoi6/mojo_vs_rust_is_mojo_faster_than_rust#c_3zamz6) criticizes **Mojo** for being potentially less safe and slower than **Rust**, citing issues with copy on write semantics and inout parameters. The critique also suggests that Mojo’s marketing strategies may overshadow the need for robust technical advancements.
- **Awaiting Mojo's Benchmark Debut**: A user expressed excitement for **Mojo** to be included in future programming **benchmark competitions**, hinting at the community's interest in seeing empirical performance data.
- **Benchmarks: A Heated Developer Hobby**: One member commented on the **developer community's fervent discussions** regarding programming language benchmarks, noting that some developers tend to take preliminary GitHub speed benchmarks too seriously, even though they should be considered more as indicators than absolutes.
- **Advocating for Borrowed Defaults**: A user acknowledges the benefits of some **Mojo** features over **Rust**, particularly after a critique by a Rust community member known for explaining async Rust. The conversation touches on borrowed references and how they could be better presented as advantages of Mojo.
- **Spreading Mojo in Academia**: A user shared a **GDSC event link** focusing on **Python and Mojo**: [Python and Mojo: Good, Bad, and the Future](https://gdsc.community.dev/events/details/developer-student-clubs-budapest-university-of-technology-and-economics-presents-python-and-mojo-good-bad-and-the-future/). The event aimed to familiarize students with Mojo, highlighting its integration with Python and its potential in systems programming.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lobste.rs/s/a3yoi6/mojo_vs_rust_is_mojo_faster_than_rust#c_3zamz6)">Mojo vs. Rust: is Mojo faster than Rust? | Lobsters</a>: no description found</li><li><a href="https://gdsc.community.dev/events/details/developer-student-clubs-budapest-university-of-technology-and-economics-presents-python-and-mojo-good-bad-and-the-future/.">Python and Mojo: Good, Bad and the Future | Google Developer Student Clubs</a>: In-person Event - Join us for an exclusive presentation on Mojo, a Python-syntax-based language with systems programming capabilities.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1232961337957683250)** (6 messages): 

- **Curiosity About Issue Filing**: A request was made to file an issue for an unspecified matter, expressing curiosity about the topic.
- **Issue Filing Accomplished**: An issue regarding the **Mojo Programming Language** has been filed on GitHub, as confirmed by the link to the [relevant issue](https://github.com/modularml/mojo/issues/2410).
- **Exploring `__copyinit__` Semantics**: Linked [GitHub Gist](https://gist.github.com/modularbot/6aed759930420cd70f38795dbcb874fe) raises a question whether it's up to the type author to implement copy-on-write semantics, or if another issue should be filed.
- **Invitation to Track via Issues**: The suggestion was made that filing an issue would make the behavior concerning `__copyinit__` more trackable, ensuring a proper response is received.
- **Level Up Announcement**: **ModularBot** celebrated a user's advancement to level 9 in the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/issues/2410)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://gist.github.com/modularbot/6aed759930420cd70f38795dbcb874fe">playground.mojo</a>: GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1233167906099957850)** (3 messages): 

- **Optimism for Continuous Improvement**: A member expressed a positive outlook, suggesting that the performance will likely improve over time.
- **Performance Gains Coincidence**: It was noted with amusement that despite differences, PyTorch and TensorFlow reported the same performance gains.
- **Curiosity about Performance Consistency**: A member queried what the outcome would be if the performance gain tests were rerun.
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1232957962767564800)** (11 messages🔥): 

- **Confusion Cleared on Overload Resolution**: The overload resolution in programming gives precedence to functions with fewer parameters. Therefore, methods within a trait with lower precedence can be overridden by a type that declares both.
- **Trait Conformance Without Extra Parameters**: It was pointed out that a trait does not need to declare an `__eq__` method with a `none` parameter to conform; this could simplify trait declarations.
- **Potential SIMD Equality Compatibility**: A slight modification may allow SIMD to conform to EqualityComparable without altering the trait’s declaration.
- **A Redundant Parameter's Dilemma**: The downside of the discussed method adjustment is the left-over redundant `none` parameter, though it's typically not used directly in dunder methods.
- **Code Efficiency Boost with `kgen.pack.load`**: A change in the code using `kgen.pack.load` in a printf function resulted in a more efficient update: 14 insertions and 985 deletions.
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1233000905058680842)** (44 messages🔥): 

- **AI Hunts Trolls**: A user shared plans to create an anti-trolling AI targeting bullies, with a query for suggestions on additional actions the bot could take.
- **SQL Q&A with Mistral and Llama3**: A user encountered issues with SQL responses being too verbose with open-source models like Mistral or Llama3 and later an `OutputParserException`. Discussions included structured output support by Ollama and examples of invoking SQL Agents with these models.
- **Understanding Redis Integration with Langchain**: A distinction between **stores** and **chat memory** was clarified; the former is a generic key-value store accessible by the `RedisStore` class, while the latter is specific to persisting chat messages by session through Redis Chat Message History integration.
- **LangChain Model Invocation Syntax Support**: A user sought advice on incorporating a prompt into a LangChain model invocation using JavaScript, with some guidance provided on chaining prompts using the `ChatPromptTemplate` and instance methods like `pipe`.
- **Clarifying Access to Gemini 1.5 Pro Model**: Users discussed how to use Gemini 1.5 Pro with LangChain; the correct usage involved `ChatVertexAI`, with indications that Gemini models cannot be accessed with ChatGoogleGenerativeAI. Correct implementation requires setting the `GOOGLE_APPLICATION_CREDENTIALS` variable.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/integrations/chat/">Chat models | 🦜️🔗 LangChain</a>: Features (natively supported)</li><li><a href="https://www.reddit.com/r/TradingProSquad_/comments/1c9fvax/tradingview_cracked_for_desktop_pc_app_windows/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://python.langchain.com/docs/integrations/memory/redis_chat_message_history/">Redis | 🦜️🔗 LangChain</a>: [Redis (Remote Dictionary</li><li><a href="https://python.langchain.com/docs/integrations/stores/redis/">RedisStore | 🦜️🔗 LangChain</a>: The RedisStore is an implementation of ByteStore that stores</li><li><a href="https://github.com/langchain-ai/langchain/issues/20924">OllamaFunctions does not work - Received unsupported message type for Ollama · Issue #20924 · langchain-ai/langchain</a>: Checked other resources I added a very descriptive title to this issue. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and di...</li><li><a href="https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/">ChatVertexAI | 🦜️🔗 LangChain</a>: Note: This is separate from the Google PaLM integration. Google has</li><li><a href="https://python.langchain.com/docs/integrations/chat/google_generative_ai/">Google AI chat models | 🦜️🔗 LangChain</a>: Access Google AI’s gemini and gemini-vision models, as well as other</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.Tool.html#invoke>)">Tool | LangChain.js - v0.1.36</a>: no description found</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.DynamicTool.html#invoke>)">DynamicTool | LangChain.js - v0.1.36</a>: no description found</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.StructuredTool.html#invoke>)">StructuredTool | LangChain.js - v0.1.36</a>: no description found</li><li><a href="https://api.js.langchain.com/classes/langchain_core_tools.DynamicStructuredTool.html#invoke>)">DynamicStructuredTool | LangChain.js - v0.1.36</a>: no description found</li><li><a href="https://api.js.langchain.com/interfaces/langchain_core_tools.ToolInterface.html#invoke>)">ToolInterface | LangChain.js - v0.1.36</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1232935474184847381)** (1 messages): 

- **LLaMA Prompt Template Queries**: A member inquired about header usage for providing context within **LLaMA3 prompts**, referencing the [official documentation](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/). Concern was expressed about the documentation's completeness due to the model's novelty.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1232957994052878336)** (5 messages): 

- **Collate Launches Personalized Newsletters**: Vel_y announced the launch of [Collate](https://collate.one/newsletter), a service that transforms articles and PDFs into a bite-sized daily newsletter. The platform provides a way to manage information overload, turning saved content into easily digestible newsletters with a "try now" option available [here](https://collate-news.streamlit.app/?embed_options=dark_theme).

- **BlogIQ Streamlines Content Creation**: Vishal_blueb introduced [BlogIQ](https://github.com/langchain-tech/BlogIQ), an app that combines the capabilities of OpenAI and Langchain to assist bloggers in content creation. The app is positioned as a clone of services like writesonic.com and copy.ai, geared towards simplifying the process of content development for blogs.

- **LangGraph for Invoice Extraction**: Toffepeermeneer shared their first project with LangGraph, an invoice extractor that takes information from pictures and stores it in a Postgres database. The project can be found on [GitHub](https://github.com/jwa91/LangGraph-Expense-Tracker) and includes an Excalidraw project overview.

- **Galaxy AI Opens Access to Premium AI Models**: White_d3vil announced Galaxy AI, a service offering **free** API access to premium AI models including **GPT-4**, **GPT-4-1106-PREVIEW**, and **Gemma**. The APIs are compatible with OpenAI's format for easy project integration, and more information, including an invite to their Discord server, can be found [here](https://discord.com/invite/BSphj69773).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://collate.one/newsletter">Newsletter</a>: Create a bite-size email digest from your content</li><li><a href="https://github.com/jwa91/LangGraph-Expense-Tracker">GitHub - jwa91/LangGraph-Expense-Tracker: LangGraph - FastAPI - Postgresql - AI project</a>: LangGraph - FastAPI - Postgresql - AI project. Contribute to jwa91/LangGraph-Expense-Tracker development by creating an account on GitHub.</li><li><a href="https://app.excalidraw.com/l/5NC0r7Sejhe/39ULXmBwigA">Whiteboarding made easy</a>: Whiteboarding tool with hand drawn like experience. Ideal for conducting interviews, drawing diagrams, prototypes or sketches and much more!</li><li><a href="https://github.com/langchain-tech/BlogIQ">GitHub - langchain-tech/BlogIQ: Clone of writesonic.com &amp; copy.ai - BlogIQ is an innovative app powered by OpenAI and Langchain, designed to streamline the content creation process for bloggers.</a>:  Clone of writesonic.com &amp; copy.ai - BlogIQ is an innovative app powered by OpenAI and Langchain, designed to streamline the content creation process for bloggers. - langchain-tech/BlogIQ</li><li><a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>: no description found
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1232980107371745280)** (35 messages🔥): 

- **Apple Shares Open Source Models**: Apple enters the open source space with smaller-than-expected models, including a [270M parameter model featured on Hugging Face](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca), alongside 450M, 1.1B, and 3B variants.
- **Dify's App Development Platform Gains Attention**: Dify offers an open-source LLM app development platform that combines various features such as AI workflow and model management; however, some users raise concerns about its lack of [loops and context scopes](https://github.com/langgenius/dify?tab=readme-ov-file).
- **New PyTorch Library for Training LLMs**: PyTorch announces [Torchtitan](https://github.com/pytorch/torchtitan), a new library that supports training large language models such as llama3 from scratch.
- **Interest in SORA Video Generation**: A Recap on SORA, an advanced video generation model by OpenAI that can create cohesive videos up to a minute long, with details and feedback from early users shared in an [FXGuide article](https://www.fxguide.com/fxfeatured/actually-using-sora/).
- **Handling Claude 3 Output Quotations**: In a discussion about issue of quotation marks causing JSON parsing errors with Opus’s Claude 3, one member advised asking the model to escape the problematic characters, which has proven effective for them especially with CSV outputs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://www.fxguide.com/fxfeatured/actually-using-sora/">Actually Using SORA - fxguide</a>: The exact current state of SORA with the team from Air Head or &#039;How to tell a consistent story despite the slot machine nature of genAI.&#039;</li><li><a href="https://gorilla.cs.berkeley.edu/leaderboard.html">
        Berkeley Function Calling Leaderboard (aka Berkeley Tool Calling
        Leaderboard)
    </a>: no description found</li><li><a href="https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM">Stanford CS25 - Transformers United</a>: Stanford CS25: Transformers United Since their introduction in 2017, transformers have revolutionized Natural Language Processing (NLP). Now, transformers ar...</li><li><a href="https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca">OpenELM Instruct Models - a apple Collection</a>: no description found</li><li><a href="https://vram.asmirnov.xyz">VRAM Calculator</a>: no description found</li><li><a href="https://github.com/pytorch/torchtitan">GitHub - pytorch/torchtitan: A native PyTorch Library for large model training</a>: A native PyTorch Library for large model training. Contribute to pytorch/torchtitan development by creating an account on GitHub.</li><li><a href="https://github.com/langgenius/dify?tab=readme-ov-file">GitHub - langgenius/dify: Dify is an open-source LLM app development platform. Dify&#39;s intuitive interface combines AI workflow, RAG pipeline, agent capabilities, model management, observability features and more, letting you quickly go from prototype to production.</a>: Dify is an open-source LLM app development platform. Dify&amp;#39;s intuitive interface combines AI workflow, RAG pipeline, agent capabilities, model management, observability features and more, letti...</li><li><a href="https://www.fxguide.com/fxfeatured/act">Action beats: 6 scenes from White House Down - fxguide</a>: A breakdown of the 6 biggest scenes from Roland Emmerich&#039;s White House Down and the visual effects behind them.
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1233416150063517798)** (12 messages🔥): 

- **Mixture of Depths Paper Presented**: A discussion on the paper titled 'Mixture of Depths' began with the presentation shared through [this link](https://paper-club.ivanleo.com/papers/mixture-of-depths). The paper introduces an approach to accelerate the training of transformers by using a modified MOE routing mechanism to adapt token flow dynamically through transformer layers.
- **Transforming Attention Mechanics**: *The Mixture Of Depths* paper proposes a solution to the problem of scaling transformers with longer sequences. By alternating between new MOD layers and normal transformer layers, the computational attention demand is cut in half, improving various training elements.
- **Large Language Models' Real-World Application Challenges**: Another paper was referenced which explores the deployment challenges of Large Language Models (LLMs) such as computing resource demands. It was mentioned that smaller, compact LLMs often do not outperform larger zero-shot LLMs in meeting summarization tasks, even after fine-tuning, as detailed in the [abstract](https://arxiv.org/abs/2402.00841).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.00841">Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization?</a>: Large Language Models (LLMs) have demonstrated impressive capabilities to solve a wide range of tasks without being explicitly fine-tuned on task-specific datasets. However, deploying LLMs in the real...</li><li><a href="https://paper-club.ivanleo.com/papers/mixture-of-depths">Nextra: the next docs builder</a>: Nextra: the next docs builder
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1233174487709650976)** (1 messages): 

- **Recording Request for Vector DBs Chat**: A member expressed interest in the **Vector DBs chat** scheduled for Friday, APR 26 but mentioned they might miss it. They inquired if the chat could be recorded, acknowledging that while not common, it has been done before.
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1232952950465232967)** (40 messages🔥): 

- **Understanding Phi-3-Mini-4K Instruct Usage**: A discussion provided insights on using the Phi-3-Mini-4K-Instruct with llamafile; [GGUF format details were highlighted](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile), mentioning steps for setting up the model and its properties including high-quality and reasoning dense datasets.
- **Mixtral 8x22B Instruct Llamafile Quickstart**: A README update was mentioned for the Mixtral 8x22B Instruct llamafile, recommending the use of `curl -L` for redirections on CDNs when downloading `.cat*` files from the provided [Quickstart](https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile) guide.
- **CPU Feature Requirements for Llamafile**: A user encountered a "fatal error" related to AVX CPU feature requirements when attempting to run a llamafile on a Mac M1. It was suggested to [restart the computer](https://github.com/Mozilla-Ocho/llamafile/issues/327#issuecomment-2053680659) and consider using smaller models for 8GB RAM systems.
- **Windows Defender Flags Llamafile as Trojan**: A user reported Windows Defender flagged a llamafile as a trojan; suggestions included trying alternative environments like virtual machines or whitelisting the folder in Windows Defender settings. [Windows Defender support](https://www.microsoft.com/en-us/wdsi/filesubmission) is only guaranteed for binaries on the official release page.
- **Resource Requirements and Troubleshooting for Llamafile Use**: Users discussed the resource demands of running the 8x22B model, noting significant RAM requirements and potential crashes due to high memory usage. It was mentioned that at least 128GB is recommended for the [Mistral 8x22B model](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hacks.mozilla.org/2024/04/llamafiles-progress-four-months-in/">Llamafile’s progress, four months in – Mozilla Hacks - the Web developer blog</a>: Mozilla’s Innovation group launched the llamafile project last year and it has become one of Mozilla’s most-favorited repositories on GitHub.</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile">microsoft/Phi-3-mini-4k-instruct-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf#how-to-use-with-llamafile>">microsoft/Phi-3-mini-4k-instruct-gguf · Hugging Face</a>: no description found</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8">Release llamafile v0.8 · Mozilla-Ocho/llamafile</a>: llamafile lets you distribute and run LLMs with a single file llamafile is a local LLM inference tool introduced by Mozilla Ocho in Nov 2023, which offers superior performance and binary portabilit...</li><li><a href="https://www.microsoft.com/en-us/wdsi/filesubmission">Submit a file for malware analysis - Microsoft Security Intelligence</a>: no description found</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/327#issuecomment-2053680659">fatal error: the cpu feature AVX was required on M1  · Issue #327 · Mozilla-Ocho/llamafile</a>: I&#39;m encountering a weird issue while trying to run the getting started on Apple M1. sh -c &quot;./llava-v1.5-7b-q4.llamafile&quot; -- ./llava-v1.5-7b-q4.llamafile: fatal error: the cpu feature AVX...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/issues/327#issuec">fatal error: the cpu feature AVX was required on M1  · Issue #327 · Mozilla-Ocho/llamafile</a>: I&#39;m encountering a weird issue while trying to run the getting started on Apple M1. sh -c &quot;./llava-v1.5-7b-q4.llamafile&quot; -- ./llava-v1.5-7b-q4.llamafile: fatal error: the cpu feature AVX...</li><li><a href="https://github.com/htop-dev/htop/issues/1443">htop doesn&#39;t report shared memory usage on Linux · Issue #1443 · htop-dev/htop</a>: In the screenshot below, you&#39;ll see that one of my processes is using 139GB of memory, but htop reports the system using 6GB of RAM. It&#39;s because htop hides mmap(MAP_SHARED) memory. This has c...</li><li><a href="https://vt.tiktok.com/ZSFctaKnm/">TikTok - Make Your Day</a>: no description found</li><li><a href="https://blog.mozilla.ai/local-llm-as-judge-evaluation-with-lm-buddy-prometheus-and-llamafile/">Local LLM-as-judge evaluation with lm-buddy, Prometheus and llamafile</a>: In the AI news cycle, with new models unveiled every day, cost and evaluation don’t come up much but are crucial to developers and businesses</li><li><a href="https://huggingface.co/jartine">jartine (Justine)</a>: no description found</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile">jartine/Mixtral-8x22B-Instruct-v0.1-llamafile · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile/resolve/main/Mixtral-8x22B-Instruct-v0.1.Q8_0.llamafile.cat0">no title found</a>: no description found</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile/resolve/main/Mixtral-8x22B-Instruct-v0.1.Q8_0.llamafile.cat1">no title found</a>: no description found</li><li><a href="https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile/resolve/main/Mixtral-8x22B-Instruct-v0.1.Q8_0.llamafile.cat2">no title found</a>: no description found
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1233172129348980899)** (6 messages): 

- **Llama-3-70b Excels in Judgemark**: **Llama-3-70b** showed promising results on [**Judgemark**](https://eqbench.com/judgemark.html), indicating its strong potential as a base for fine-tuning the **disco-judge**. Judgemark evaluates a model's capability to judge creative writing and requires at least 8k supported context length.
  
- **Potential Collaboration on Evaluation**: A user is open to collaborating, offering insights gained from creating the [evaluation](https://sampaech.substack.com/p/creating-magi-a-hard-subset-of-mmlu) and suggested their elaborate judging prompt design for testing complex rubrics.

- **Learning from Magazine and MMLU**: The user @_jp1_ praised the work in an article for creating **MAGI**, a highly selective and discriminative subset of **MMLU**, designed to challenge and differentiate high-ability models.

- **Judgemark Data for Fine-Tuning**: A user expressed readiness to format and share all **Judgemark outputs** for potential use in fine-tuning datasets, asking about the collection process for said datasets.

- **Phi-3-mini-4k-instruct's Mixed Results**: Despite less impressive performance on **eq-bench** compared to their published evaluations, **Phi-3-mini-4k-instruct** is listed on the [eq-bench leaderboard](https://eqbench.com/judgemark.html), where users may need to scroll to find it.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://eqbench.com/judgemark.html">EQ-Bench Judgemark Leaderboard</a>: no description found</li><li><a href="https://sampaech.substack.com/p/creating-magi-a-hard-subset-of-mmlu">🧙Creating MAGI: A hard subset of MMLU and AGIEval</a>: Adding Headroom and Discriminative Power to Existing Benchmarks
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1232998702931644416)** (4 messages): 

```html
<ul>
  <li><strong>API Focus and Library Ease Discussed:</strong> Tgi is presented as API-first and prioritizes low latency, while vllm is acclaimed for being an easy-to-use library, emphasizing cost-effective and high-throughput deployment.</li>
  <li><strong>Batch Generation Inquiry at Hugging Face:</strong> A debate finds its way to Hugging Face regarding batch generation capabilities <a href="https://github.com/huggingface/text-generation-inference/issues/1008#issuecomment-1742588516"><strong>GitHub Issue #1008</strong></a>, revealing community-driven problem-solving.</li>
  <li><strong>DiscoLM Inference Speed Woes:</strong> A member reports slow initialization and inference times for DiscoLM_German_7b_v1 on a high-performance computing system, contrasting with much faster times on a local setup without GPUs.</li>
  <li><strong>Potential Misconfiguration in DiscoLM:</strong> Another member suggests ensuring correct model loading with <code>device_map='auto'</code>, expecting a significant speed improvement when using 2x V100 GPUs for inference.</li>
</ul>
```

**Link mentioned**: <a href="https://github.com/huggingface/text-generation-inference/issues/1008#issuecomment-1742588516">Batch generate? · Issue #1008 · huggingface/text-generation-inference</a>: System Info Hi, i like to ask if it is possible to do batch generation? client = Client(&quot;http://127.0.0.1:8081&quot;,timeout = 60) gen_t = client.generate(batch_text,max_new_tokens=64) generate c...

  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1232990125471043666)** (7 messages): 

- **DiscoLM-70b Deployment Struggles**: A member described issues with running **DiscoLM-70b**, facing the "Template not found" error and nonsensical output from the `/generate` endpoint on `huggingface/text-generation-inference`.
- **Appreciation for DiscoLM Models**: Salvasfw expressed their deep appreciation for the DiscoLM series of models, even amidst troubleshooting challenges.
- **Musings on Powerful MoE Model**: There was speculation about the potential of building and training an **8 x phi-3 MoE model**, with curiosity about its capabilities.
- **Mini 4k Llamafication Success**: The `Phi-3-mini-4k` was successfully llamafied, according to crispstrobe, with a decent EQ-Bench Score (v2_de) of 51.41, despite some mistakes in the German output. The model did not specifically train on German data, and the results suggest it might be further trainable.
- **Gguf Model Downloads**: Johannhartmann highlighted the popularity of the **gguf** model, which saw 1500 downloads within two days of release.


  

---



**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1233040210372202529)** (12 messages🔥): 

- **Debating Claude’s Capabilities**: An online discussion explored **Claude’s RLAIF training** as light-handed and well-executed. It is reported that Claude’s behavior shows unexpected structure and a deep understanding, ‘mostly orthogonal’ to Anthropic’s vision, giving off "Jungian individuation" and "Bodhisattva vibes." The thread also speculates on the effects of RLAIF versus the base model’s latent dynamics and discusses the potential for Claude's mode collapse to be rectified ([Claude’s conversation thread](https://x.com/repligate/status/1783426037210026372?s=46&t=xxWoJxAS_7-BBFC2ro84Zw)).

- **RLHF vs. KTO in Commercial Deployments**: In response to a query about the stability of **Reinforcement Learning from Human Feedback (RLHF)**, it’s suggested that its application depends on the context, and that **Knowledge-Targeted Optimization (KTO)** might be better suited for certain applied tasks.

- **Transitioning Training Methods for Improved Results**: A mention of an interview shares an experience where moving from **Supervised Fine Tuning (SFT)** to **Data Programming by Demonstration (DPO)** provided better outcomes, and subsequently moving to **KTO** further improved performance based on user feedback.

- **Complications and Nuance in RLHF**: There’s an assertion that **RLHF** is more nuanced than commonly thought, particularly when considering the variety of data and how it interacts with downstream evaluation metrics.

- **Understanding Grad Norm Spikes**: The channel requested clarity on why spikes in gradient norms during pretraining might be undesirable, but no detailed explanation was provided in the response.

**Link mentioned**: <a href="https://x.com/repligate/status/1783426037210026372?s=46&t=xxWoJxAS_7-BBFC2ro84Zw">Tweet from j⧉nus (@repligate)</a>: definitely have no doubt there are various ways to do RL/generation-discrimination/synthetic data/self-play-esque training on top of teacher-forcing that makes the models smarter, but especially more ...

  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1233150418717114458)** (4 messages): 

- **The "Pick Your Brain" Conundrum**: *Nathan Lambert* mentioned his discomfort with the term **"pick your brain"** especially now that he tends to decline such requests due to being busy.
- **Humorous Take on Brain Picking**: *Vsreekanti* responded humorously to the discomfort toward **brain-picking**, suggesting that one should inquire about the type of pick, jokingly preferring a lobotomy.
- **Brain Picking as Vague Request**: *Drj.bet* added that the phrase **"pick your brain"** often implies a desire for conversation without a specific question in mind.
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1233434451896438927)** (1 messages): 

- **CPU-friendly Language Models Tackling Python**: An article titled "Low-Cost Language Models: Survey and Performance Evaluation on Python Code Generation" has been shared, delving into performance evaluations of **CPU-friendly language models**. It introduces a dataset of 60 programming problems and discusses the use of a Chain-of-Thought prompt to guide models in problem-solving, available through this link: [View PDF](https://arxiv.org/abs/2404.11160).

**Link mentioned**: <a href="https://arxiv.org/abs/2404.11160">Low-Cost Language Models: Survey and Performance Evaluation on Python Code Generation</a>: Large Language Models (LLMs) have become the go-to solution for many Natural Language Processing (NLP) tasks due to their ability to tackle various problems and produce high-quality results. Specifica...

  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1232960311133343764)** (4 messages): 

- **Fine-tuning Moondream for Image Recognition**: A [YouTube video](https://www.youtube.com/watch?v=Gwq7smiWLtc) was shared demonstrating the fine-tuning of the Moondream Vision Language Model on a Captcha image dataset. It's described as a guide to improve performance on a downstream task.
- **AI Developers Meetup in Toronto**: The upcoming local & open-source AI developer meetup at **Cohere space** in Toronto was highlighted with a [link to the event](https://lu.ma/devs5). A member named *Andrei* is helping organize, and it features lightning talks, demos, and networking.
- **Introducing Snowflake Arctic**: Another [YouTube video](https://www.youtube.com/watch?v=nV6eIjnHEH0) showcases Snowflake Arctic, an enterprise-focused LLM designed for cost-effective AI solutions. Briefly, the video introduces this new addition to the landscape of large language models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Gwq7smiWLtc">Finetuning Moondream Vision Language Model</a>: This video demonstrates how to fine-tune moondream to improve performance on a downstream task. For this example, we&#39;ll fine-tune on this Captcha image datas...</li><li><a href="https://www.youtube.com/watch?v=nV6eIjnHEH0">Snowflake Arctic: The Best LLM for Enterprise AI</a>: Today, the Snowflake AI Research Team is thrilled to introduce Snowflake Arctic, a top-tier enterprise-focused LLM that pushes the frontiers of cost-effectiv...</li><li><a href="https://lu.ma/devs5">Toronto Local &amp; Open-Source AI Developer Meetup · Luma</a>: Local &amp; open-source AI developer meetup is coming to Toronto! Join the Ollamas and friends at the Cohere space! Special thank you to abetlen (Andrei), the…
</li>
</ul>

</div>
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1233183712859000842)** (2 messages): 

- **GUI for GPT Lovers**: A member discovered [jan.ai](https://jan.ai/), which is praised as a user-friendly graphical user interface for running models locally.
- **Smaller Models with Big Ambitions**: A member shared [OpenELM](https://huggingface.co/apple/OpenELM), an efficient language model family released by Apple, which offers both pretrained and instruction tuned models, utilizing a unique layer-wise scaling strategy for efficient parameter allocation.

**Link mentioned**: <a href="https://huggingface.co/apple/OpenELM">apple/OpenELM · Hugging Face</a>: no description found

  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

venadore: trying to get llama 3 to do topic complexity classification, not half bad
  

---



---




---
id: f6ce07b9-cb0b-4d23-8996-04827151f6a2
title: not much happened today
date: '2024-07-31T07:04:15.405372Z'
original_slug: ainews-to-be-named-5098
description: >-
  AI news for 7/29/2024-7/30/2024 includes Meta's SAM 2 object segmentation
  model, new web framework FastHTML, AI model benchmarks, and updates on open
  source AI resources. Notable events include Leonardo AI's acquisition by
  Canva, Midjourney v6.1 release, and advancements in quantization and
  adversarial robustness.
companies:
  - meta
  - canva
  - midjourney
  - apple
  - jeremyphoward
  - alexandr_wang
  - demishassabis
  - ylecun
models:
  - sam-2
  - gemini-1.5-pro
topics:
  - object-segmentation
  - web-development
  - ai-benchmarks
  - adversarial-robustness
  - quantization
  - open-source-ai
---


<!-- buttondown-editor-mode: plaintext -->**it was a quiet day.**

> AI News for 7/29/2024-7/30/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**248** channels, and **2257** messages) for you. Estimated reading time saved (at 200wpm): **262 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

A few small items:

- maartengrootendorst's [Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) went viral,
- chatgpt's advanced voice mode [started rolling out to a small group of users](https://x.com/EthanSutin/status/1818439026401329645) - some even got the [vision enabled version](https://x.com/manuvision/status/1818412120373182928?s=46)
- [Leonardo AI was acquired by canva]( https://x.com/ethan_smith_20/status/1818152222326186260?s=46)
- [Jim Fan shared how Project Groot is augmmenting human demonstration data for their robots](https://x.com/drjimfan/status/1818302152982343983?s=46)
- [Midjourney v6.1 shipped](https://x.com/midjourney/status/1818342703618482265)

We had fun recording a demo of Advanced Voice Mode, coming on the next LS podcast.

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet, best of 4 runs.

**Meta Releases SAM 2 for Object Segmentation**

- [@AIatMeta](https://twitter.com/AIatMeta/status/1818055906179105010) announced the release of Meta Segment Anything Model 2 (SAM 2), a unified model for real-time, promptable object segmentation in images and videos. SAM 2 is available under Apache 2.0 license.

- The model comes with a new SA-V dataset that is [4.5x larger and has ~53x more annotations](https://twitter.com/AIatMeta/status/1818055908070773078) than the largest existing video segmentation dataset.

- SAM 2 can be [applied out of the box to diverse real-world use cases](https://twitter.com/AIatMeta/status/1818055909760975134). Meta provided links to try the demo and access the code.

**New Web Development Framework: FastHTML**

- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1818036923304456492) announced FastHTML, a new way to create modern interactive web apps in Python. It scales from simple 6-line apps to complex production systems.

- FastHTML integrates authentication, databases, caching, styling, and more. It offers [1-click deployment to platforms like Railway, Vercel, and Hugging Face](https://twitter.com/jeremyphoward/status/1818036926827610423).

- The framework aims to [make web programming easier and more powerful](https://twitter.com/jeremyphoward/status/1818036930657009888) by leveraging web foundations rather than complex frameworks.

- Jeremy created a [1-hour mini-course on FastHTML](https://twitter.com/jeremyphoward/status/1818036938932371605) showing how to create and deploy a complete interactive web app from scratch using pure Python.

**AI Model Developments and Benchmarks**

- [@alexandr_wang](https://twitter.com/alexandr_wang/status/1817956788320530940) announced Scale's latest SEAL Leaderboard on Adversarial Robustness, focusing on universal harm scenarios with transparent evaluation methods.

- [@demishassabis](https://twitter.com/demishassabis/status/1818049561421910345) highlighted that Gemini 1.5 Pro topped the new Scale AI leaderboard for adversarial robustness.

- Apple released a [technical report on their Intelligence Foundation Language Models](https://twitter.com/awnihannun/status/1817989760729891296), detailing the architecture and training process of their on-device and server models.

**Open Source AI and Compute Resources**

- [@ylecun](https://twitter.com/ylecun/status/1818044278029128046) shared an article in The Economist about the importance of open source AI, co-authored by Martin Casado and UC Berkeley professor Ion Stoica.

- There were discussions about the [availability and pricing of GPU resources for AI development](https://twitter.com/far__el/status/1817965343702401363), with some noting increased availability and potentially falling demand.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Quantization Advancements for Efficient LLM Inference**

- **[A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)** ([Score: 332, Comments: 37](https://reddit.com//r/LocalLLaMA/comments/1eeyab4/a_visual_guide_to_quantization/)): The post presents **"A Visual Guide to Quantization"**, offering a comprehensive overview of various **quantization techniques** used to reduce the size and computational requirements of **Large Language Models (LLMs)**. It covers methods such as **INT8**, **INT4**, and **binary quantization**, explaining their principles and trade-offs between model size reduction and performance impact, while also discussing advanced techniques like **vector quantization** and **mixed-precision quantization**.
  - The author, **MaartenGr**, explains the motivation behind creating the visual guide, emphasizing the **increasing need for quantization** as more **LLMs** are released. The guide covers various techniques from basic value representation to advanced methods like **GPTQ**, **GGUF**, and **BitNet**.
  - The guide features **over 60 custom visuals** to enhance intuition and make quantization techniques accessible to both novice and experienced readers. It covers topics such as **(a)symmetric quantization**, **dynamic/static quantization**, and **quantization-aware training**.
  - A reader commends the guide as *"one of the best writing on quantization"* they've encountered, highlighting its exceptional quality and comprehensive coverage of the subject.

- **Llama 3.1 405B EXL2 quant results** ([Score: 75, Comments: 31](https://reddit.com//r/LocalLLaMA/comments/1efg2wv/llama_31_405b_exl2_quant_results/)): **Llama 3.1 405B** model was quantized using **EXL2** for GPU usage, with results showing that in the **125-150GB** model size range, raw EXL2 quantization outperforms Meta's distillation to 70B. The **405B** model demonstrates superior performance in long context Q&A, fact analysis, and detailed story comprehension compared to the 70B version and commercial LLMs, maintaining consistency near its **128K context limit**. Despite benchmarks suggesting similar performance between 70B and 405B models, the latter excels in practical tasks, only struggling when multiple similar examples are present in the text.
  - **Llama 3.1 405B** model's performance varies with quantization levels. At **2.5bpw** (123GB), it's coherent for short contexts but struggles beyond **4K tokens**. At **3bpw**, it maintains coherence up to **12K tokens**.
  - The model's long-context performance may stem from **more MLP params**, **bigger embedding dim**, **more attention layers**, or **raw training compute**. **Llama 3.1 70B** outperforms in-house finetunes of Llama 2 and 3 70B for **128K context**.
  - Users compared **Llama 3.1 405B** to **Claude-3.5-Sonnet** and **GPT-4**, noting similar input costs ($3/M) but highlighting Llama's advantage in finetuning capabilities. Some expressed interest in comparisons with **Mistral Large 2** and **DeepSeek-v2-coder**.


**Theme 2. Meta's Open-Source AI Contributions and Impact**

- **[Segment Anything 2 (Meta)](https://github.com/facebookresearch/segment-anything-2)** ([Score: 107, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1effb4z/segment_anything_2_meta/)): Meta has released **Segment Anything 2 (SA-2)**, an upgraded version of their image segmentation model. SA-2 offers **improved performance**, including the ability to segment **3D objects** in images and videos, and can handle **higher resolution inputs** of up to **3000x3000 pixels**. The model also introduces new capabilities such as **text prompting** and **multi-modal prompting**, allowing for more flexible and precise segmentation tasks.
  - Users praised **SA-2's performance**, with one testing it on a random video and reporting it worked "flawlessly." The [web demo](https://sam2.metademolab.com/demo) was described as "mind-blowing," particularly its ability to track a ball in video clips.
  - Discussion centered on potential applications, including applying SA-2 to **3D models** to address "useless blobs" issues in 3D human modeling, and speculation about a "Track anything" capability for video segmentation.
  - Some users questioned if segmentation is now "fully solved" given SA-2's capabilities, while others commended **Meta** and **Zuckerberg** for their open-source contributions to AI development.
- **What If Meta Open-Sources Their Image Model? The Impact Could Be HUGE!** ([Score: 76, Comments: 41](https://reddit.com//r/LocalLLaMA/comments/1efmvf2/what_if_meta_opensources_their_image_model_the/)): Meta's AI image generator, **Emu**, was trained on **1.1 billion images** and has shown impressive speed and quality. While not yet publicly available, there's speculation about potential open-sourcing, similar to Meta's **Llama models**, which could be a significant development in the field of AI image generation. If released, it would offer a novel alternative to existing tools like **Stable Diffusion**, potentially allowing users to run image generation models on personal computers.
  - **Open-sourcing Meta's image model** could drive development of smaller, efficient versions for various devices. While matching **DALL-E** or **MidJourney** locally may be challenging, simpler tasks like prototyping and object removal are already possible on high-end smartphones.
  - Image generation models are impacting industries, with **Activision Blizzard** approving use of **Midjourney** and **Stable Diffusion** for concept art and marketing. **Klarna** reported $6 million savings in image production costs using **genAI tools**, and 90% of employees integrating AI into daily workflows.
  - Recent months have seen a surge in new image generation models, including **Kolors**, **SD3**, **Aura**, **Flow**, **Lumia**, **Hunyuan**, and **Pixart**. These models have applications in marketing, video game development, and graphic design, with the U.S. graphic design market alone worth approximately **$14 billion**.


**Theme 3. Performance Comparisons of Recent LLM Releases**

- **Mistral NeMo vs Llama3.1 8B** ([Score: 74, Comments: 32](https://reddit.com//r/LocalLLaMA/comments/1eeuo9s/mistral_nemo_vs_llama31_8b/)): The post inquires about comparisons between **Llama3.1 8B** and **Mistral NeMo (12B)** models, particularly focusing on their **multilingual capabilities**. The author expresses interest in **Mistral NeMo's promising performance** but seeks confirmation on whether it outperforms **Llama3.1 8B**, requesting both personal experiences and benchmark discussions.
  - **Mistral NeMo** is considered "smarter" and comparable to **Llama3 70B**, while **Llama3.1 8B** excels in natural tone, style, and creativity. Users suggest **Nemo** is better for code and function calling, while **Llama** is more suitable for chatbots.
  - **Gemma 2 9B** is mentioned as a strong contender against both models, particularly for tasks not requiring long context. Users speculate that a potential **Gemma 2.1** with improved context handling could outperform both **Llama 3.1** and **Mistral Nemo**.
  - Users note that **Mistral NeMo** has less innate censorship and is receptive to prompting, recommending a **temperature between 0.5-1** for creative writing. The official model card's claim of outperforming "smaller or similar" models is criticized as setting a low bar.

- **Llama 3.1 405B EXL2 quant results** ([Score: 75, Comments: 31](https://reddit.com//r/LocalLLaMA/comments/1efg2wv/llama_31_405b_exl2_quant_results/)): The post compares the performance of **Llama 3.1 405B** and **70B** models in **long-context tasks**, focusing on **EXL2 quantizations** of the 405B model for GPU use. The author notes that in the **125-150GB model size range**, raw EXL2 quantization outperforms Meta's distillation to 70B in terms of **perplexity (PPL)**. Despite benchmarks suggesting similar performance, the author's testing reveals that the **405B model significantly outperforms the 70B model** and closed-source LLMs like **GPT-4** and **Claude Sonnet 3.5** in tasks involving **long context Q&A**, **fact analysis**, and **remembering details from stories**, especially near the **128K context limit**.
  - **Llama 3.1 405B** model outperforms **70B** in **long-context tasks**, but **2.5bpw quantization** of 405B struggles beyond **4K tokens**, while **3bpw** lasts until about **12K tokens**. The author suggests this warrants further investigation.
  - Discussions focused on comparing different **quantization levels** and model sizes, with interest in how the **405B model** compares to **fp16 70B** and **DeepSeek MoE models**. The author notes that **raw compute** and **training duration** may contribute to improved performance.
  - Users expressed interest in comparisons with **Mistral Large 2** and other models for complex tasks and long context use. The author is working on extracting **open test benchmarks** from internal datasets for more objective comparisons.


**Theme 4. Hardware and Efficiency Considerations for Local LLM Inference**

- **Is the new DDR6 the era of CPU-powered LLMs?** ([Score: 97, Comments: 87](https://reddit.com//r/LocalLLaMA/comments/1ef0qpb/is_the_new_ddr6_the_era_of_cpupowered_llms/)): The upcoming **DDR6 RAM** standard is reported to potentially reach frequencies of up to **17,000 MHz** in overclocking mode, prompting speculation about its impact on **CPU-powered LLMs**. The post questions whether this advancement might enable running language models entirely on CPUs, potentially reducing reliance on GPUs for such tasks.

- **Do you think Llama3 405B can be profitable ?** ([Score: 150, Comments: 102](https://reddit.com//r/LocalLLaMA/comments/1eewqtv/do_you_think_llama3_405b_can_be_profitable/)): The post discusses the **profitability challenges** of the **Llama3 405B API**, referencing a Twitter discussion by **Jia** on the topic. The author mentions a friend working for a **cloud company** that recently launched the API, struggling to find a **pricing balance** between profitability and customer acceptance.
    - **Avianio** claims profitability hosting **Llama 3 405B** at **$5 per million tokens**, while another user suggests realistic **H100 SXM** prices (<$2.5/gpu/hr) make most companies profitable on 405B and 70B models.
    - The market for serving open models is described as highly **commoditized**, with differentiation challenges. Companies like **OpenAI**, **Anthropic**, and **Mistral** rely on proprietary or exclusively licensed models to charge premium prices.
    - **Meta's** open-sourcing strategy is viewed as an attempt to **reduce profits** of potential competitors like **OpenAI**. Some users question the choice of 405B model, suggesting the 70B version as a more cost-effective alternative for most client needs.

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

TO BE COMPLETED

---

# AI Discord Recap

> A summary of Summaries of Summaries


## Claude 3.5 Sonnet


**1. LLM Advancements and Benchmarking**

- **Llama 3.1 Impresses with Multilingual Capabilities**: Meta's **[Llama 3.1](https://x.com/reach_vb/status/1815767864277606762)** has been released with models up to **405B parameters**, achieving an **85.2 score on the MMLU benchmark**, and supporting **128K context**.
   - The model comes with a more permissive license allowing training of other LLMs on its outputs, positioning it as a strong competitor to **GPT-4** and **Claude**. Users reported mixed experiences, with some praising its performance while others encountered issues like looping responses.
- **Apple's AI Models Show Promise**: Apple's new AI paper reveals significant benchmarks for their server-side and on-device models, with **MMLU scores of 61.4** for on-device and **75.4** for server models.
   - The paper details a two-stage pre-training process alongside SFT and RLHF methods. Notably, Apple stated they do not use **NVIDIA GPUs** for AI model training, instead opting for **TPUs**, making them the second-largest TPU user in the industry.
  


**2. Model Optimization and Performance Tuning**

- **Quantization Techniques Gain Traction**: A [visual guide to quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) highlights how Large Language Models (LLMs) often exceed billions of parameters, making them challenging to run on consumer hardware.

**3. Open-Source AI Developments**

- **SWE-Bench Ultra-Hackathon Pushes Boundaries**: A **6-day ultra-hackathon** for **SWE-Bench** is being hosted to push the limits of open-source code generation, with participants receiving $1,000 in compute from StrongCompute.
   - The event features talks from co-authors including [John Yang](https://x.com/jyangballin), [Carlos E. Jimenez](https://x.com/_carlosejimenez), and [Ofir Press](https://x.com/OfirPress), aiming to boost open-source code generation capabilities and spark innovative approaches in the community.
- **SAM 2 Enhances Segmentation Capabilities**: Meta released **[Segment Anything Model 2 (SAM 2)](https://ai.meta.com/blog/segment-anything-2/)**, offering real-time promptable object segmentation in images and videos, significantly improving upon its predecessor.
   - SAM 2 is trained on a new SA-V dataset with 50,000 videos and employs a novel memory attention technique. The [GitHub repository](https://github.com/facebookresearch/segment-anything-2) provides code for running inference, trained model checkpoints, and example notebooks for various segmentation tasks.
  
**4. AI Industry News and Partnerships**

- **Perplexity Launches Publishers Program**: Perplexity announced its **[Publishers Program](https://pplx.ai/publishers)**, partnering with major organizations like TIME, Der Spiegel, and Fortune to ensure access to reliable information and support publishers.
   - The initiative aims to provide new technology to engage audiences and promote collective success, with plans to introduce **revenue sharing** models in the coming months, starting with advertising through related questions.
- **Leonardo AI Joins Canva Family**: [Leonardo.Ai](https://x.com/ethan_smith_20/status/1818152222326186260?s=46) announced its acquisition by Canva, which is expected to enhance creative tools and empower creators in new ways.
   - This integration aims to speed up innovation and build on existing projects like Phoenix, potentially reshaping the landscape of AI-powered design tools and creative workflows.
  

---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.1 impresses with multilingual features**: [Llama 3.1](https://x.com/reach_vb/status/1815767864277606762) supports models with **405B** parameters and achieves **85.2** on the MMLU benchmark with **128K context**.
   - This release comes with a permissive license, allowing training on its outputs, marking it as a strong competitor to **GPT4o** and **Claude**.
- **Argilla 2.0 boasts dataset duplication feature**: Argilla 2.0's upcoming release includes a feature for easy dataset duplication, improving workflow efficiency.
   - The [announcement](https://x.com/argilla_io/status/1817945202432061792) has been received positively by the community, helping users manage multiple datasets seamlessly.
- **PEFT v0.12.0 introduces new methods**: [PEFT v0.12.0](https://x.com/julien_c/status/1817837045298978986) showcases methods like **OLoRA** and **X-LoRA**, aimed at enhancing model training efficiency.
   - These methods are crucial for improving performance and resource allocation during training.
- **Achieving SOTA in Image Generation**: A member announced achieving SOTA image generation capabilities and highlighted advancements in the field.
   - They shared [this tweet](https://twitter.com/DataPlusEngine/status/1818358813520441493) as evidence of the achievement, with further developments in image generation technologies also discussed.
- **Exploring Quantization in Language Models**: [A visual guide](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) underscores the importance of quantization techniques for optimizing LLMs on consumer hardware.
   - The focus is on creating smaller, more efficient models to address size-related challenges.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Model Loading Issues After Upgrade**: Users reported GPU acceleration failures after upgrading to version **0.2.29**, indicating potential corruption during the update process.
   - One user advised clearing application data and reinstalling version **0.2.28**, while others highlighted that **Llama 3.1** requires **0.2.29** for optimal performance.
- **Unexpected Looping Responses from Llama 3.1**: One user experienced continuous looping responses from the **Llama 3.1 8B model** after the LM Studio upgrade, recommending the Llama v2 preset instead.
   - This issue underlined the need for a deeper understanding of prompt formatting to avoid such behaviors in AI response.
- **Resources for Getting Started in AI Development**: A new user looking to dive into AI development was directed towards **Python** with **PyTorch** as essential foundational tools.
   - Free resources on platforms like **YouTube** were suggested to help with grasping the concepts involved in AI.
- **GPU Compatibility Issues Highlighted**: Members noted that **Intel Iris Xe Graphics** are unsupported in LM Studio, necessitating NVIDIA with **CUDA** or AMD with **ROCm** for proper operation.
   - The performance of the **Tesla P40** was discussed, indicating it faces compatibility and speed issues compared to contemporary consumer GPUs.
- **LM Studio Version 0.2.29 Now Available on ROCm**: Queries about LM Studio 0.2.29's release on ROCm were answered, and it was confirmed available as per the [GitHub release notes](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md).
   - Members expressed eagerness to utilize the new features offered in this update for their setups.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Publishers Program Launch**: Perplexity introduced its **publishers' program**, collaborating with organizations like [TIME](https://pplx.ai/publishers) and [Der Spiegel](https://pplx.ai/publishers) to enhance content sourcing.
   - The program aims to uphold **high-quality answers** backed by trusted sources like [The Texas Tribune](https://pplx.ai/publishers), while also planning to implement **revenue sharing** models.
- **Llama-3 Models Hallucinate**: Users are reporting issues with the **llama-3-sonar-large-32k-online** model producing hallucinated information, which has surfaced recently.
   - Concerns were echoed about the deprecation of **Llama models** on August 12, 2024, as users find them increasingly unreliable.
- **Tesla's Charging Station Alert**: Tesla has issued a warning about charging station compatibility, causing concern among users who rely on **supercharging**.
   - This announcement raises questions about the reliability of Tesla's infrastructure for long-distance travel.
- **Comparative Analysis of AI Models**: Users discussed the comparative performance of **Claude 3.5 Sonnet** and **GPT-4o**, highlighting their respective strengths across various tasks.
   - While **Claude** provides good outputs, **GPT-4o** received praise for accuracy, particularly in coding applications.
- **Space Force Expands Satellite Network**: The **Space Force** plans to expand its satellite network to enhance national security and communication capabilities.
   - This announcement has ignited debate on the implications of increased **military satellites** in orbit.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Artisan Embraces New Command /style**: The **/style** command now allows users to generate images based on specified styles, such as **Van Gogh-style cats** or **Japanese-style spaceships**.
   - Members are encouraged to try this feature, with examples already shared showcasing its creative potential.
- **Encountering OutOfMemoryError in Stable Diffusion**: Users hit **OutOfMemoryError** even with 8GB GPUs while generating images with **SD1.5 models**, leading to troubleshooting discussions.
   - Suggestions included altering CUDA settings and increasing virtual memory to mitigate these issues.
- **Struggles with AI Character Consistency**: A user detailed challenges in training models for consistent character generation using tools like **IP Adapter** and **ControlNet**.
   - They shared their current settings and sought additional improvements for more reliable results.
- **Exploring AI Animation Tools**: A discussion surfaced around various **AI animation tools**, particularly for generating minimalistic animations from static images, focusing on **Live Portrait AI**.
   - Some noted concerns over quality degradation in tools like **Runway**, leading to debates over the best software for different tasks.
- **Introducing SAM 2 for Video Segmentation**: The new **SAM 2 model** from Meta promises enhanced object segmentation for both still images and videos, paving the way for real-time applications.
   - Its strong zero-shot performance may offer benefits for creative tasks like animation remixes.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth struggles on Windows**: Users reported encountering a 'No triton module' error while using **Unsloth** on Windows and suggested switching to WSL as a workaround.
   - One user humorously mentioned their refusal to switch from Windows due to gaming preferences.
- **Challenges with fine-tuning models**: Discussions about fine-tuning a **Llama3** model focused on avoiding catastrophic forgetting, leading to the idea of combining datasets for retraining.
   - Participants confirmed that complete retraining is preferable to mitigate risks associated with catastrophic forgetting.
- **Matrix representation using custom tokens**: A user inquired about representing a **30x30 matrix** using custom tokens for their **Arc-AGI** project, highlighting the need for more details.
   - Another member prompted for clarification, indicating that a more in-depth explanation would be beneficial.
- **Rope scaling support improves in Unsloth**: A recent update confirmed that older models which previously lacked support for **rope scaling** now have this feature implemented in Unsloth as of two weeks ago.
   - Members expressed excitement about the new capability, mentioning **Phi-3 128k variants** in relation to this enhancement.
- **Creating translation datasets**: A user sought translation datasets for fine-tuning English models, considering using **DeepL** for this purpose, with others suggesting utilizing **Wikipedia** as a resource.
   - The conversation highlighted the importance of comprehensive datasets in enhancing model training.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Randomized SVD simplifies large problems**: Randomized SVD reduces large-scale matrix problems to smaller matrices, providing approximations of key singular values and vectors for efficient processing.
   - This technique is useful for handling massive datasets without overwhelming computational resources.
- **Exploring Optimizer CPU Offload**: Members discussed a proposed `cpu_offload` flag to move optimizer states to CPU, facilitating parameter transfers during optimization steps.
   - Concerns arose about the blocking nature of the optimizer step impacting the feasibility of interleaved operations with `torch.compile`.
- **Finetuning Llama 3.1 for Jeopardy**: A member is finetuning **Llama 3.1 8B** using **Unsloth**, expressing confusion over the complex configuration.
   - They emphasized a preference for a stable bf16 finetuning process to simplify the training pipeline.
- **WebGPU API: More than Just a Browser Tool**: WebGPU serves as an API with a shallow compilation definition for **WGSL**, now used in native applications beyond browsers.
   - This includes implementations in **Rust** and **Zig**, boosting usability across various platforms.
- **Excitement builds for the upcoming event**: The upcoming **CUDA MODE IRL** event is generating buzz, with attendees expressing enthusiasm about meeting in-person.
   - Members underscored the necessity of registration, and details about GPU access and keynote recordings were confirmed.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Small Models Show Competitive Edge**: A recent paper suggests that running a **70B model** once versus generating five outputs from a **13B model** can produce gains of up to **15% across five tasks**.
   - *This begs the question: what happens when both models operate under the same budget?* The findings emphasize the importance of **unit-test setups** for selecting the best outputs.
- **Skepticism Around AI Interpretability Timeline**: **AI interpretability** may take a few more years before reliable datasets are available outside private practices.
   - Members expressed that longer timelines for public data releases could foster more robust findings.
- **Apple AI Models Benchmark Insights**: The new Apple paper presents server-side and on-device models with **MMLU** scores of **61.4** and **75.4**, respectively.
   - A two-stage pre-training process alongside SFT and RLHF methods was detailed in the findings.
- **Exploring Techniques for Hermes and Llama Model Merging**: Discussions centered around merging techniques for **Hermes** models with **Llama**, with write-ups in the works on effective merging strategies.
   - Members debated the performance impact of various techniques on compatibility and efficiency.
- **Midjourney V6.1 Enhancements**: Midjourney has launched **V6.1**, featuring improved image quality and coherence as well as new upscaling models.
   - The update follows claims of achieving state-of-the-art results in image generation from the community.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Voice Mode Begins Rollout**: The **Advanced Voice Mode** is rolling out to a select group of **ChatGPT Plus** users, promoting real-time conversations and the ability to interrupt freely.
   - Instructions were sent via email and mobile apps, with broader access anticipated by **fall**.
- **Members Confirmed Search GPT Access**: Users confirmed access to [Search GPT](https://link.to.searchgpt), expressing varying levels of confidence in its capabilities.
   - Some noted it as helpful, while others questioned its functionality.
- **Anticipation Builds for GPT-4o Features**: Discussion arose around the expected release of **GPT-4o's advanced vision and voice** features, with members suggesting a potential alpha release **by the end of this month**.
   - This indicates interest in updates and potential timeline adjustments.
- **DALL-E Bot Command Issues Persist**: Users encountered problems executing the `/draw` command in the DALL-E bot channel, with some unable to create images for over **20 minutes**.
   - Frustration was voiced, and members sought community assistance to troubleshoot the issue.
- **Concerns About GPT Performance in Function Calls**: Community members raised alarms regarding the decline in **GPT-4o's** response quality when utilizing function calls, suggesting reduced accuracy in outputs.
   - They compared performance between full prompts and function call submissions, noticing significant disparities.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API down but operational**: Members reported that the **Cohere API** was temporarily down, encountering a **503 error**, but confirmed via the [Cohere status page](https://status.cohere.com/) that it is now fully operational.
   - The status page currently indicates **99.67% uptime** for endpoints and **100% uptime** for documentation, enhancing user confidence in system reliability.
- **Celebrating successful projects with Cohere API**: A member proudly showcased their **dream project** built using the **Cohere API**, featuring functionalities like weather, time, and semi-working news, sparking enthusiastic responses from the community.
   - This project emphasized the importance of background vibes and the features crucial for production efficiency.
- **Connector response format struggles**: Discussions revealed that returning **unix timestamps** as integers in the **Cohere chat API** caused issues, while string representations worked fine, leading to clarifications on the expected data types.
   - It was mentioned that although integers are supported, they are handled as strings within the connector response format.
- **Inquiry for Webinar Access**: After missing the **Enterprise Workflow Automation with GenAI** webinar, a member sought to obtain a recording, advised to contact [events@cohere.com](mailto:events@cohere.com) for swift access.
   - This highlights the structured approach Cohere promotes to ensure attendees can still access important content despite missing live sessions.
- **Exploring tool usage vs connectors**: A shift towards **tool usage** over connectors was noted in discussions, spurred by insights from recent office hours, suggesting a strategic pivot in community practices.
   - While connectors maintain distinct functions, there are currently no plans to deprecate them, allowing flexibility in user approach.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Community Meeting #5 Recap**: The recorded [Mojo Community Meeting #5](https://youtu.be/1T-MBC9k99M) discussed **GPU programming** and a Q&A session. Participants sought more focused discussions and proposed live coding sessions for future engagements.
   - The desire for deeper exploration into **Mojo's capabilities** was clear, signaling a need for enhanced topic specificity in upcoming meetings.
- **Easy Installation for Stack-PR**: **Stack-pr** can now be installed via `pipx install stack-pr`, facilitating the creation of stacked pull requests on GitHub. Members discussed submitting a feedstock to conda-forge to streamline this process.
   - Simplifying installation paths for new tools like stack-pr reflects a broader aim to enhance the Mojo ecosystem's usability.
- **CSV Reader Capabilities Explored**: Inquiries about **Mojo's CSV reader** revealed existing functionalities that could parallel Python's csv module. The discussion highlighted the community's eagerness to explore comprehensive features for enhanced understanding of Mojo.
   - Members indicated that extending **CSV capabilities** could significantly broaden Mojo's applicability in data processing.
- **Implementing Image Parsing in Mojo**: A contributor shared their successful implementation of **PNG parsing** in Mojo, linking to their [GitHub repository](https://github.com/fnands/mimage). They plan to address JPEG parsing next.
   - Community enthusiasm for image parsing libraries signals growing interest in extending Mojo's multimedia capabilities.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Offers Office Hours for Users**: LlamaIndex invites users to sign up for [office hours](https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform) to discuss use cases regarding agents and receive branded swag.
   - Participants can expect a **15-30 minute Zoom conversation** to explore how LlamaIndex can assist with agent applications.
- **GraphRAG Technique Combines Multiple Approaches**: The **GraphRAG** technique from Microsoft integrates text extraction, network analysis, prompting, and summarization into one system, enhancing data comprehension with generated graphs.
   - More details can be found [here](https://t.co/ZnDtJ731hl) along with an explanation of its application [here](https://t.co/mx54Su1gYk).
- **Webinar Rescheduled for Next Thursday**: The upcoming webinar is now scheduled for **next Thursday 8/8 at 9am PT**, as communicated in a recent update [here](https://t.co/Zo9zRz528F).
   - *Participants should update their calendars accordingly.*
- **RAPTOR Pack Updates Discussed**: Members discussed deploying **RAPTOR** to hosted vector DBs like Pinecone and managing document insertions without re-clustering.
   - *Strategies for adding new documents without compromising previously clustered data were exchanged.*
- **Generating Mermaid Diagrams from LLM Outputs**: Members shared tools for generating **Mermaid diagrams** from LLM outputs, specifically the use of `mmd` format and the recommended **Mermaid CLI** for rendering.
   - *Useful examples were provided to demonstrate effective diagram generation, with a reference to [Mermaid Syntax](https://mermaid.js.org/intro/syntax-reference.html).*



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Transformers Error during Indexing Debacle**: Several members reported an assertion error: `srcIndex < srcSelectDimSize` while utilizing the **Transformers** library, particularly in the **Mistral** model configuration.
   - A proposed fix involved **deleting the cache** and redownloading dependencies to resolve this issue.
- **Gemma 2 Outputs Continuous Pad Token**: A user faced an issue where their fine-tuned **Gemma 2 9b** model constantly outputs the `<pad>` token after its deployment to vLLM.
   - Discussion pointed towards configuration problems, emphasizing the need to verify **special tokens** from [Hugging Face](https://huggingface.co/google/gemma-2-9b-it/blob/main/special_tokens_map.json).
- **Chat Template Training Configuration Change**: The introduction of **PR #1756** requires a `roles_to_train` field for `type: chat_template`, breaking existing examples using **chat_template**.
   - Members voiced concerns over requiring additional documentation and examples to clarify this change.
- **RAG Implementation Exploration for Chatbots**: A participant discussed the possibility of using **Retrieval Augmented Generation (RAG)** as an alternative fine-tuning approach for their chatbot project.
   - They intend to split their efforts between RAG and traditional fine-tuning, aiming for a solid output enhancement.
- **Loss Function Stuck at Zero**: A user reported their model training loss being stuck at **0.0** with 'grad_norm' displaying as **nan**, suggesting a serious training issue.
   - This persistent loss could indicate underlying problems with model training dynamics or misconfigured settings that need addressing.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Agent Executor Lacks Insight**: Concerns were raised that the **Agent Executor** in LangSmith fails to demonstrate its planning processes, limiting user insight into decision-making.
   - Participants suggested that enhancing visibility may require user-level implementations for better transparency.
- **LangGraph Emerges for Planning**: A shared example of **LangGraph** sparked discussions about its potential to facilitate agentic workflows, moving beyond basic executions.
   - Users are encouraged to learn LangGraph for its advanced capabilities, enhancing their projects.
- **Llama 3.1's Fresh Tool Calling Syntax**: The unique function calling support in **Llama 3.1** utilizes a special prompt syntax, differing from standard parameter setups.
   - Questions arose about the possibility of this syntax becoming a norm in **LangChain** integration.
- **Turing Test Takes a Fun Turn**: An article explores a playful format of the **Turing Test** where three language models compete to convince each other of their AI status.
   - This light-hearted take invites readers to reflect on whether machines can indeed think, fostering dialogue about AI capabilities.
- **Comprehensive SWE Agent Guide Released**: A detailed guide on creating **SWE Agents** using tools like **CrewAI** and **LangChain** promotes leveraging the **swekit** Python framework.
   - This guide aims to simplify the scaffolding and functionality across various agentic frameworks, making it accessible [here](https://git.new/swe/kit).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Palm Chat 2 experiences a 3000% increase**: Palm Chat 2's usage surged from **1** request to **30**, illustrating a **3000% increase**.
   - A member humorously compared this spike to the *WinRAR sales* meme, adding laughter to the discussion.
- **New GPT-4o allows for extensive outputs**: The experimental version of **GPT-4o** can handle up to **64K output tokens** per request, around **18.2K words**.
   - The output cost is estimated to be **$1.15** per **64K reply**, a significant factor for large outputs.
- **Searching for LiteLLM alternatives**: A user expressed frustration with **LiteLLM's** confusing documentation, suggesting a potential build for similar services with **OpenRouter**.
   - OpenRouter offers more control by providing cost information from its generations endpoint.
- **Challenges with Claude models and instruct templates**: Discussion arose regarding whether the **Claude 3.5 Sonnet model** utilizes an instruct template, with some doubts raised.
   - It was suggested that using `prompt` mode in **OpenRouter** could effectively convert prompts into usable user messages.
- **Fireworks model status confirmed**: A member confirmed that while **Fireworks** is operational, the **Yi-Large endpoint** has been removed for unspecified reasons.
   - This prompted discussions around the stability of models hosted by **Fireworks**, ensuring continued functionality.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SAM 2 Released with Enhanced Capabilities**: [Meta Segment Anything Model 2 (SAM 2)](https://ai.meta.com/blog/segment-anything-2/) has been released, offering real-time promptable object segmentation in images and videos, significantly improving upon its predecessor with state-of-the-art performance.
   - Trained on a new SA-V dataset with **50,000** videos, SAM 2 employs a novel memory attention technique for segmentation in diverse settings.
- **Leonardo AI Joins Canva's Family**: [Leonardo.Ai](https://x.com/ethan_smith_20/status/1818152222326186260?s=46) announced its acquisition by Canva, which is expected to enhance creative tools and empower creators in new ways.
   - This integration is set to speed up innovation, building on existing projects like Phoenix.
- **Kagi Launches New LLM Benchmarking Project**: The [Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html) evaluates large language models on reasoning, coding, and instruction-following capabilities with an unpolluted benchmark.
   - Current results show **gpt-4o** leading in accuracy and efficiency, underscoring the need for continuous testing across providers.
- **Strategic Collaboration Opportunities for OpenAI and Anthropic**: Discussions suggest **OpenAI** and **Anthropic** could collaborate with brands by providing analytics based on chat mentions, akin to [Google Analytics](https://link.to/google-analytics).
   - This may align with new models like SearchGPT to present insights while ensuring data anonymization.
- **Apple Intelligence Beta Launch**: The **Apple Intelligence Beta** is now available on **macOS** and **iPhone**, providing users access to new AI functionalities.
   - Active discussions on [Discord](https://discord.com/channels/822583790773862470/1249801456870101013) include feedback on performance and usability.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Exploring Open Interpreter's Uses**: Members discussed various **use cases** for the Open Interpreter (OI), emphasizing its potential as an on-screen assistant for task management.
   - *I've been searching for a way to have something essentially learn my on-screen movements over time* showcases the personal touch on open-source capability.
- **AI Takes Over Coding**: A member touted the success of using AI for generating code, boasting awards won without writing any code themselves.
   - They urged others to leverage AI for coding efficiency, asserting *trust me, you can do it too friend*.
- **Concerns About Wayland Experience**: A user shared their struggle with **Wayland**, revealing challenges faced during the transition to this display server.
   - Their feedback reflects a shared sentiment among users adapting to new systems.
- **Perplexica: Your New Search Buddy**: A video titled [Perplexica + Llama-3.1](https://www.youtube.com/watch?v=V0vx94JYNjI) demonstrates how to set up a local, free alternative to Perplexity using Llama-3.
   - The tutorial highlights the simplicity of installation along with the functionality of AI-driven search solutions.
- **Pre-order Availability Questions**: A user inquired about the status of **pre-orders** for building Open Interpreter units, expressing frustration in finding updates.
   - It was clarified that pre-orders are no longer accepted, prompting others to gather parts independently.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **View Merging Task Clarity**: The task aims to prove that `View.__add__` merges any two mergable views, or modify it if it fails. Complexities arise when views aren’t pairwise mergable, pushing for shape tracker reduction.
   - The bounty setter emphasizes clarity in definitions to ensure minimal views for better performance in final index calculations.
- **YouTube Excursion into Parallel Computing**: A member shared a [YouTube video](https://youtu.be/c52ziyKOArc?si=pAUdzwIQGXCtpk3T) from the UCSC Colloquium discussing parallel computing and its implications, with slides available.
   - The talk was held on April 10, 2024, highlighting the importance of advancements in parallel computing methodologies.
- **TinyJit Messes with Gradients**: After applying TinyJit, all tensors returned **None** for gradients on the third training loop step, a stark contrast to previous steps. This issue seemed to stem from **TinyJit** activation disrupting normal behavior.
   - Removing TinyJit resolved the issue, confirmed by members discussing the placement of **optim.step()** outside the jitted function as a potential culprit.
- **Deciding on Jitting Strategy**: A member debated whether to jit the model's forward step alone or the entire step function, leading to advice that a comprehensive jitting approach is preferable.
   - The community consensus leaned towards jitting the full step function unless a specific reason dictated otherwise.
- **OpenCL Resource Error Encounter**: A member expressed difficulties in generating 'out of resources' errors with OpenCL on a Mac, encountering 'invalid kernels' instead. This suggests the issue likely relates to compilation rather than runtime resource limitations.
   - The consensus among peers hinted at exploring more about compilation scenarios that lead to these confusion points in resource management.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Apple Ignores NVIDIA for TPUs**: Apple has officially stated that it does not utilize **NVIDIA GPUs** for training its AI models, instead opting for **TPUs**, as reported in a [recent article](https://www.reuters.com/technology/apple-says-it-uses-no-nvidia-gpus-train-its-ai-models-2024-07-29). This move positions Apple as the **second biggest user of TPUs** in the industry.
   - The decision reflects a broader strategy to reduce reliance on competitors like NVIDIA while promoting its own AI capabilities.
- **Tim Dettmers Joins Allen Institute**: **Tim Dettmers** has secured a position at the **Allen Institute** and will begin teaching at **Carnegie Mellon University** in Fall 2025 after an extensive job search yielding **15 offers** from **17 universities**. He aims to enhance open-source contributions while continuing his work with **bitsandbytes**.
   - The competitive interest in his expertise highlights the demand for talent in AI, with firms like **Anthropic** and **Hugging Face** expressing eagerness to recruit him.
- **Sewon Kim's Attractiveness for Firms**: The recruitment of **Sewon Kim** has sparked significant interest from various companies, illustrating his growing influence in the field. This influx of interest emphasizes the importance of a **unique offering** to capture top talent.
   - This trend reflects the competitive landscape in AI talent acquisition, where standout candidates attract multiple opportunities.
- **Zuck's Colorful Commentary at SIGGRAPH**: At **SIGGRAPH**, **Zuck** made headlines for his candid remarks alongside **Jensen**, notably stating, *“Make me another cheesesteak, Jensen,”* adding humor to the event's serious discussions.
   - This moment highlights the blend of levity and weightiness often present in high-stakes conferences.
- **Perplexity Launches Innovative Program for Publishers**: **Perplexity** initiated its **Publishers Program** to provide media organizations with features like **revenue sharing** and engagement tools, intending to elevate the quality of media sources. Partners include established organizations like **TIME** and **Der Spiegel**.
   - This initiative aims not only to distribute profits but also to improve the overall responsiveness of their systems.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Exploring OPTO in Trace's Framework**: Members highlighted the implications of [OPTO used by Trace](https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/), emphasizing its relevance in AI applications.
   - The discussion points to a significant focus on self-adapting AI technologies, especially as they relate to the gaming sector.
- **Growth of Neural Networks**: Conversations referenced the evolution of neural networks to complex systems with **billions of parameters**, such as those powering [ChatGPT](https://arxiv.org/abs/2303.08774).
   - These advancements have drastically reshaped the capabilities of AI applications across various domains.
- **MIRPO compatibility with DSPy functions**: Members sought clarification on whether **MIRPO** now supports **dspy.Suggest** and **dspy.Assert**, following previous compatibility issues.
   - No updates have emerged yet to confirm that the functionality has been addressed.
- **Creating penalty metrics for answer deviations**: Discussions focused on developing a penalty metric that increases with the distance from the gold answer, advocating for proportional penalties.
   - *One suggestion involved utilizing a formula that squares the difference* between predicted and actual scores.
- **ICML talk on Language Models**: A member shared insights from an **ICML talk** focusing on the 'Physics' of Language Models, suggesting optimizers could utilize 'celebrity' exemplars.
   - The link to the talk can be found [here](https://youtu.be/YSHzKmEianc) for further exploration.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Developers Needed for Long Context Innovation**: The team is actively seeking developers to explore long context use cases with Jamba's **256k effective length**, aiming to boost output informed by **enterprise customer feedback**.
   - They encourage participants to share their experiments, offering incentives like **credits, swag, and fame**.
- **Enterprise Clients Share Positive Feedback**: Early responses from enterprise customers show **promising results** while testing Jamba's capabilities and functionalities.
   - The message calls for further insights to foster collaborative efforts in enhancing the platform.
- **New User Enthusiastic About Jamba**: A new member, **artworxai**, introduced themselves in the Discord, expressing eagerness to learn more about **Jamba**.
   - This shows a growing interest among newcomers in the platform's features and applications.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **SWE-Bench Ultra-Hackathon Pushes Code Generation Limits**: A **6-day ultra-hackathon** for **SWE-Bench** is being hosted, providing participants with **$1,000 in compute** courtesy of StrongCompute. Prizes are up for grabs for benchmark improvements, featuring talks from co-authors including [John Yang](https://x.com/jyangballin), [Carlos E. Jimenez](https://x.com/_carlosejimenez), and [Ofir Press](https://x.com/OfirPress).
   - This event aims to boost open-source code generation capabilities, with discussions expected to spark innovative approaches and insights in the community.
- **GitHub Hosts Segment Anything Model 2 Code Repository**: The [GitHub repository](https://github.com/facebookresearch/segment-anything-2) for **Segment Anything Model 2 (SAM 2)** is now live, offering code for running inference alongside trained model checkpoints and example notebooks. This resource enhances usability for various segmentation tasks in open-source projects.
   - Engagement around SAM 2 is expected to increase with these easily accessible tools, encouraging developers to implement sophisticated segmentation solutions effortlessly.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Sentry Discusses AutoFix Feature**: Jenn and Ben from **Sentry** are set to present their **AutoFix** feature in an upcoming session. Event details can be found [here](https://discord.com/events/1089876418936180786/1245836053458190438).
   - The presentation is expected to cover how this open source feature enhances development workflows and troubleshooting, providing community-driven support.
- **Benefits of Sentry's Open Source Features**: The upcoming discussion will emphasize the advantages of utilizing **open source** features like AutoFix for developers. Participants can anticipate valuable insights into community-driven updates and support.
   - This session aims to boost understanding of collaborative development practices and expand engagement with the **Sentry** platform.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1267936585283010640)** (1 messages): 

> - `Llama 3.1 Release`
> - `Argilla 2.0 Sneak Peek`
> - `PEFT v0.12.0`
> - `Hugging Face Video Initiative`
> - `Inference as a Service with Nvidia` 


- **Llama 3.1 impresses with multilingual features**: [Llama 3.1](https://x.com/reach_vb/status/1815767864277606762) has been released with models up to **405B** parameters, achieving an **85.2** on the MMLU benchmark, and it supports **128K context**.
   - *It comes with a more permissive license*, allowing training of other LLMs on its outputs, marking it as the latest competitor to **GPT4o** and **Claude**.
- **Argilla 2.0 boasts dataset duplication feature**: The upcoming [Argilla 2.0](https://x.com/argilla_io/status/1817945202432061792) release will include a highly requested feature for easy dataset duplication, aiding in managing similar datasets.
   - This enhancement is expected to streamline workflows for users handling multiple datasets.
- **PEFT v0.12.0 introduces new methods**: [PEFT v0.12.0](https://x.com/julien_c/status/1817837045298978986) just dropped, showcasing new efficient methods like **OLoRA** and **X-LoRA**, enhancing model training processes.
   - These methods are aimed at improving the performance and resource efficiency of various models.
- **Hugging Face ventures into video content**: Hugging Face is launching [video capabilities](https://x.com/micuelll/status/1816851392134586540) to bridge the gap with existing closed video models.
   - This initial step aims to leverage their models for video analysis and generation.
- **Nvidia partners with Hugging Face for AI services**: A collaboration with [Nvidia AI](https://x.com/NVIDIAAIDev/status/1818050230392398175) was announced to provide inference-as-a-service, enabling developers to prototype with open-source AI models.
   - This initiative aims to simplify the deployment of AI models in production environments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/reach_vb/status/1815767864277606762)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Meta Llama 3.1 405B, 70B & 8B are here - Multilingual & with 128K context & Tool-use + agents! Competitive/ beats GPT4o & Claude Sonnet 3.5 unequivocally the best open LLM out there!🐐  Bonus: It come...</li><li><a href="https://x.com/reach_vb/status/1818218875239977000)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Llama 3.1 8B running on Mac, 100% local, powered by llama.cpp 🔥  Two steps:  1. brew install llama.cpp  2. llama-cli --hf-repo reach-vb/Meta-Llama-3.1-8B-Instruct-Q6_K-GGUF \ --hf-file meta-llama-3.1...</li><li><a href="https://x.com/argilla_io/status/1817945202432061792)">Tweet from Argilla (@argilla_io)</a>: 💫 Excited for the Argilla 2.0 release? Stay tuned for updates coming soon! In the meantime, we&#39;re thrilled to share a sneak peek of one of the highly requested features: easy dataset duplication....</li><li><a href="https://x.com/julien_c/status/1817837045298978986)">Tweet from Julien Chaumond (@julien_c)</a>: in case you missed it last week:  peft v0.12.0 just dropped 🔥  With some cool new param-efficient methods like OLoRA, X-LoRA, FourierFT, and more</li><li><a href="https://x.com/micuelll/status/1816851392134586540)">Tweet from Miquel Farré (@micuelll)</a>: Hugging Face goes video! We want to close the gap to closed video models and this is our first step. Weights: https://huggingface.co/mfarre/Video-LLaVA-7B-hf-CinePile Code: https://github.com/mfarre/V...</li><li><a href="https://x.com/abidlabs/status/1818034189348053204)">Tweet from Abubakar Abid (@abidlabs)</a>: Thanks @mmitchell_ai for the nice PR adding the ability to watermark AI-generated videos in @Gradio with a single parameter 😎</li><li><a href="https://x.com/davidberenstei/status/1817115209590272021)">Tweet from David Berenstein (@davidberenstei)</a>: ⚗️ Find reusable synthetic data pipeline code and corresponding datasets on the @huggingface Hub.  Find your pipline and use `$ distilabel pipeline run --config &#34;hugging_face_dataset_url/pipeline....</li><li><a href="https://x.com/abhi1thakur/status/1816429924233687470)">Tweet from abhishek (@abhi1thakur)</a>: 🚨 NEW TASK ALERT: VLM Finetuning 🚨 AutoTrain just added VLM finetuning: Captioning and VQA for PaliGemma. Now, its super-easy to finetune PaliGemma on your own custom dataset. Which model and tasks ...</li><li><a href="https://x.com/NVIDIAAIDev/status/1818050230392398175)">Tweet from NVIDIA AI Developer (@NVIDIAAIDev)</a>: We partnered with @huggingface to launch inference-as-a-service, which helps devs quickly prototype with open-source AI models hosted on the Hugging Face Hub and deploy them in production.  ➡️https://...</li><li><a href="https://x.com/RisingSayak/status/1818133546411728903)">Tweet from Sayak Paul (@RisingSayak)</a>: With larger and larger diffusion transformers coming up, it&#39;s becoming increasingly important to have some good quantization tools for them.  We present our findings from a series of experiments o...</li><li><a href="https://x.com/mervenoyann/status/1816857371416887653)">Tweet from merve (@mervenoyann)</a>: Did you know that @huggingface has an open-source Cookbook with many applied AI recipes? 🤩📖  Here are some of the latest recipes contributed 🧶</li><li><a href="https://x.com/_philschmid/status/1816514989982908591)">Tweet from Philipp Schmid (@_philschmid)</a>: I heard you like Charts. 👀 So, I made a code-specific one using BigCodeBench and Aider (Code editing). We should really stop using HumanEval for coding skills! 🧑🏻‍💻  &gt; BigCodeBench evaluates LL...</li><li><a href="https://x.com/davidberenstei/status/1816419520447127728)">Tweet from David Berenstein (@davidberenstei)</a>: The @Meta  Llama-3.1 model series can be used for distilling and fine-tuning but this requires annotated preference data so I created a Human Feedback Collector based on @Gradio that directly logs dat...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1267557819280916513)** (301 messages🔥🔥): 

> - `Token Limit for Meta LLaMA`
> - `Issues with Hugging Face Datasets`
> - `Training Models on Different GPUs`
> - `Background Removal Model Updates`
> - `Using Accelerator for TPUs` 


- **Token Limit Confusion for Meta LLaMA**: Users discussed the token limits for the **meta/meta-llama-3.1-405b-instruct**, with confusion about whether it is 100 tokens, while others reported around 100 tokens in their replies.
   - One user noted that responses were getting cut off, prompting further discussion about the inference API limitations.
- **Hugging Face Datasets Reliability Issues**: Members expressed frustration regarding Hugging Face datasets being down for two days, with discussions on errors and unreliability.
   - There were suggestions to update and check the status of datasets, as members experienced 500 errors.
- **Training Issues on Different GPUs**: Users shared experiences of training models on various GPUs, mentioning issues with models freezing and out-of-memory errors while training on a **3060**.
   - One user found it worked better after switching to an **A100**, despite it having less VRAM.
- **New Background Removal Model Available**: A member announced the merging of a better background removal model on Hugging Face, providing excitement for improvements over the previous **rmbg1.4 model**.
   - Discussion followed regarding the inability to use older models as effectively for specific tasks.
- **Using Accelerator for TPUs**: While trying to utilize the trainer API on TPUs, it was mentioned that simply running the script should automatically capture the device if **Accelerate** is installed.
   - However, there were issues related to GPU usage complications that led users to seek alternatives.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NVIDIAAI/status/1818047933889274159">Tweet from NVIDIA AI (@NVIDIAAI)</a>: .@huggingface has partnered with us to launch inference-as-a-service, enhancing capabilities for developers on its platform. This service, powered by NVIDIA NIM microservices, allows immediate access ...</li><li><a href="https://x.com/_philschmid/status/1818286805441286563">Tweet from Philipp Schmid (@_philschmid)</a>: Apple&#39;s AI strategy: No secrets, Build on Open source and science, Win! 🚀  Seeing @Apple acknowledging the collective effort in AI and not being quiet on how they build Apple Intelligence with op...</li><li><a href="https://x.com/stevewattsfrey/status/1818033777622532518">Tweet from Steve Frey (@stevewattsfrey)</a>: A bold experiment: We&#39;re hosting a 6-day ultra-hackathon for SWE-Bench to push the limits of open-source code generation  - Everyone gets $1,000 in compute provided by @StrongCompute  - Up 50 rese...</li><li><a href="https://huggingface.co/FunAudioLLM/SenseVoiceSmall">FunAudioLLM/SenseVoiceSmall · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/newgen-audiomaker-roblox-6snot-lynxdenis-gif-19984815">Newgen Audiomaker GIF - Newgen Audiomaker Roblox - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/explosion-kitty-komaru-cat-explosion-cat-cat-explosion-gif-4940756872467221811">Explosion Kitty Komaru Cat GIF - Explosion kitty Komaru cat Explosion - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=4rk9fHIOGTU">Llama 8b Tested - A Huge Step Backwards 📉</a>: Full test of LLaMA 3.1 8b. Even though it had a huge bump in benchmarks, the results from my test were very disappointing. Vultr is empowering the next gener...</li><li><a href="https://youtu.be/2PKCOVqhngY?si=DUKS8F0QiBdEHj4R">&quot;I want Llama3.1 to perform 10x with my private knowledge&quot; - Self learning Local Llama3.1 405B</a>: Building Local Self Learning Llama3.1 Agent in your SlackGet free HubSpot resource of adopt AI at work: https://clickhubspot.com/7hmy🔗 Links- Get full code ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1267599437673529475)** (4 messages): 

> - `Extensive Testing Insights`
> - `Quantization in LLMs` 


- **Extensive Testing Appreciated**: A member acknowledged the **extensive testing** and expressed gratitude for the **videos** that were shared.
   - Another member responded positively, stating they were glad the insights were found **useful** and appreciated the feedback.
- **Exploring Quantization for Language Models**: A member shared an article link about a **visual guide to quantization**, highlighting the challenge of running **Large Language Models (LLMs)** on consumer hardware due to their size.
   - The article emphasizes the importance of quantization as a technique to make models smaller and more efficient, a crucial area of ongoing **research**.



**Link mentioned**: <a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">A Visual Guide to Quantization</a>: Exploring memory-efficient techniques for LLMs

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1267592359852507207)** (9 messages🔥): 

> - `Quantization of LLMs`
> - `YouTube Content on AI`
> - `Diffusion Models`
> - `TikTok AI Trends` 


- **Exploring Quantization in Language Models**: An insightful piece on [Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) highlights how Large Language Models (LLMs) often exceed billions of parameters, making them challenging to run on consumer hardware.
   - This post introduces quantization techniques aimed at optimizing these models for better efficiency and performance.
- **YouTube Videos on AI Trends**: Several links to [YouTube videos](https://youtu.be/yJVRXun70dk) were shared, showcasing various AI advancements, including a video about diffusion models.
   - One suggested clip was particularly engaging and worth viewing for enthusiasts of AI technology.
- **Advancements in Diffusion Models**: The discussion highlights a [HuggingFace blog post](https://huggingface.co/blog/quanto-diffusers) about the ability to quantize diffusion models, transforming high-resolution text-to-image generation.
   - As models scale in size, this method addresses the increased memory demands associated with larger transformer-based diffusion pipelines.
- **TikTok Trend on AI**: A TikTok video discussing AI trends drew attention, showcasing the contemporary conversation surrounding technology in modern media.
   - The video reflects how AI is extending its reach into popular culture and social media platforms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.tiktok.com/@todayin_ai/video/7395098689737444614?is_from_webapp=1&sender_device=pc&web_id=7375199177678243361">TikTok - Make Your Day</a>: no description found</li><li><a href="https://huggingface.co/blog/quanto-diffusers">Memory-efficient Diffusion Transformers with Quanto and Diffusers</a>: no description found</li><li><a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">A Visual Guide to Quantization</a>: Exploring memory-efficient techniques for LLMs
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1267623514387779686)** (11 messages🔥): 

> - `ClIP-Enhanced Text Search`
> - `Unity ML Agent Experiment`
> - `Frooty Integration Queries`
> - `Guanaco Evolved Dataset`
> - `SAM v2 Segmentation Mask Generator` 


- **ClIP-Enhanced Text Search showcases innovative image retrieval**: A member developed an image retrieval system named **CLIP-Enhanced Text Search on Images** using the **CLIP algorithm** to allow users to retrieve images based on text queries, inspired by a [Medium post](https://lnkd.in/ebBUnVdr).
   - This technology is aimed at enhancing **content creation**, **data analysis**, and **e-commerce**, enabling users to efficiently search for relevant images.
- **Unity ML Agent wanders with no direction**: A member created a new **Unity ML Agent** that learns to wander around a scene with no waypoints using **SAC** and a basic reward signal, as showcased in a [YouTube video](https://youtube.com/live/dcCn4nuKpBs?feature=share).
   - They also mentioned successful integration of **CUDA** with the latest ML-Agents and plans to upload the models to Hugging Face.
- **Querying Frooty integration for a project**: A member expressed interest in integrating their work with **Frooty**, while another shared their hope to conquer the **iplug2** level of game development, citing technical challenges ahead.
   - Issues have arisen with incorporating **websockets** and auto-record features, raising uncertainties about compatibility with **FL Studio**.
- **Guanaco dataset evolves with promising results**: A member shared progress on creating an **evolved dataset** based on **Guanaco**, aiming to enhance quality and complexity, while identifying potential improvements like grading and **DEITA filtering**.
   - Initial results show a **~2% improvement on MMLU**, with the dataset available [here](https://huggingface.co/thesven/SmolLM-360M-Guanaco-Evolved-SFT).
- **Introducing SAM v2 Mask Generator**: A member created a space to generate and export **segmentation masks** using the latest **SAM v2 model**, accessible via a shared [Hugging Face link](https://huggingface.co/spaces/lightly-ai/SAMv2-Mask-Generator).
   - This tool aims to streamline the process of mask generation for images, enhancing practical applications for users.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lnkd.in/ebBUnVdr]">LinkedIn</a>: This link will take you to a page that’s not on LinkedIn</li><li><a href="https://lnkd.in/eQYY6_rp]">LinkedIn</a>: This link will take you to a page that’s not on LinkedIn</li><li><a href="https://huggingface.co/spaces/lightly-ai/SAMv2-Mask-Generator">SAMv2 Mask Generator - a Hugging Face Space by lightly-ai</a>: no description found</li><li><a href="https://youtube.com/live/dcCn4nuKpBs?feature=share">Unity ML-Agents | Live Agent training from Scratch</a>: a quick little experiment withing ml agents and cuda
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1267623270870417421)** (2 messages): 

> - `Music Generation Models`
> - `Autoregressive Model Techniques`
> - `Papers on Music Generation` 


- **Interest in Autoregressive Music Generation**: A member expressed a desire to train a **music generation model** specifically using **autoregressive** techniques instead of diffusion methods.
   - They sought suggestions or relevant [papers to check out](https://link.to/papers).
- **Exploration of Model Alternatives**: Another member highlighted the importance of exploring various **autoregressive models** for effective music generation.
   - They recommended checking existing research and literature on this topic.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1267693369577050163)** (1 messages): 

> - `Quantization Tools for Diffusion Transformers`
> - `Transformer-based Diffusion Models`
> - `Memory Savings in Large Models`
> - `High-Resolution Text-to-Image Generation` 


- **Quantization enhances diffusion transformers**: Recent experiments demonstrate significant memory savings by quantizing different diffusion pipelines based on **diffusion transformers**, albeit with a slight increase in inference latency.
   - This latency is expected to improve over time, making quantization tools crucial as model sizes increase.
- **Shift from UNet to Transformer architectures in diffusion**: The adoption of **Transformer-based diffusion backbones** for high-resolution text-to-image generation marks a shift away from the previously prevalent UNet architecture.
   - These transformer models, which boast scalability with parameters ranging from **0.6B to 8B**, allow for more robust performance across various tasks.
- **Scaling models increases memory demands**: As diffusion models scale up, the memory requirements become more demanding due to the multiple components in a diffusion pipeline: text encoder, diffusion backbone, and image decoder.
   - This complexity underscores the importance of effective quantization strategies to manage resource consumption.



**Link mentioned**: <a href="https://huggingface.co/blog/quanto-diffusers">Memory-efficient Diffusion Transformers with Quanto and Diffusers</a>: no description found

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1267573209507823769)** (6 messages): 

> - `CachedGISTEmbedLoss function`
> - `Evaluating translated sentences`
> - `Statistical methods for evaluation`
> - `Seq2seq tasks limitations`
> - `Pseudo labels and ontology use` 


- **CachedGISTEmbedLoss Function Experimentation**: A member experimented with the **CachedGISTEmbedLoss** function but found it ineffective, possibly due to already cleaning their dataset sufficiently.
   - They noted significant training dataset improvements when ordered to focus on hard negatives, aiding in the model's ability to refine features gradually.
- **Need for Evaluators in Translation Evaluation**: A member inquired about evaluating translated sentences from a model without a reference translation.
   - In response, suggestions included using evaluators, potentially a model, or having bilingual speakers assess the outputs.
- **Demand for Statistical Evaluation Methods**: Another member expressed curiosity about statistical methods for evaluating translations, utilizing features like **POS tags** or length.
   - They highlighted a preference against using deep learning models for this evaluation.
- **Limitations of Seq2seq Tasks**: A member pointed out that seq2seq tasks often require a reference or gold label, which adds limitations to the evaluation process.
   - They also proposed the use of pseudo labels to address these challenges, suggesting a dictionary or ontology for mapping.
- **Using BabbelNET for Ontology Mapping**: The discussion included exploring a more exotic topology for evaluating translations, possibly requiring **BabbelNET coverage**.
   - This method would allow for mapping distances in translation tasks, albeit with complexities.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1267663666564567070)** (4 messages): 

> - `HuggingChat integration`
> - `Knowledge Distillation of 7B model`
> - `SOTA Image Generation`
> - `Compute Resources for Model Training` 


- **Need Help Integrating HuggingChat into Google App Script**: A member posted a question seeking assistance with integrating **HuggingChat** into **Google App Script** using API, stating it's not working as intended.
   - They are looking for guidance from anyone experienced in this integration process.
- **Knowledge Distillation for 7B Model**: Another member requested support on **knowledge distillation** for the **7B model**, specifically on setting up hyperparameters.
   - They also inquired about the compute resources needed for the task.
- **Achieving SOTA in Image Generation**: A member proudly announced achieving **SOTA image generation** capabilities internally, sharing [this tweet](https://twitter.com/DataPlusEngine/status/1818358813520441493) as a highlight.
   - They followed up with another link showcasing further advancements in image generation technologies.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1267557368787370166)** (195 messages🔥🔥): 

> - `LM Studio Model Loading Issues`
> - `Llama 3.1 Model Behavior`
> - `AI Development Learning Resources`
> - `Thread Count Adjustment in Local Server`
> - `System Prompt Management` 


- **Model Loading Issues After Upgrade**: Users reported GPU acceleration failures after upgrading LM Studio to version 0.2.29, with errors occurring on models that previously worked, suggesting potential corruption during the update process.
   - A user advised clearing application data and reinstalling version 0.2.28, while others noted that Llama 3.1 requires version 0.2.29.
- **Unexpected AI Response Behavior**: A user experienced looping responses from the Llama 3.1 8B model after upgrading LM Studio and was advised to use the Llama v2 preset for better performance.
   - They acknowledged the need for further understanding of prompt formatting to prevent such looping behavior.
- **Learning AI Development Resources**: A new user looking to start in AI development was pointed towards Python with PyTorch and transformer models as foundational tools.
   - Suggestions for free learning resources included platforms like YouTube to help grasp these concepts in AI.
- **Adjusting Thread Count in Local Server**: A user inquired about increasing thread count in their local server setup, discovering only code chat settings had the option available until upgrading to the latest version.
   - Post-update, the user confirmed the option was present, highlighting the importance of keeping software up to date.
- **Managing the System Prompt**: One user deleted the System Prompt while experimenting with settings in LM Studio and inquired about common values for such prompts.
   - It was clarified that generic prompts typically start with phrases like 'You are a helpful XYZ AI assistant', guiding the AI's responses.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.itpro.com/security/python-developers-beware-this-info-stealing-malware-campaign-is-targeting-thousands-of-github-accounts">Python developers beware: This info stealing malware campaign is targeting thousands of GitHub accounts</a>: Python developers should be wary of an information stealing malware disguised in the popular Colorama python package, which has already compromised a community of over 170,000 users</li><li><a href="https://huggingface.co/Groq/Llama-3-Groq-8B-Tool-Use">Groq/Llama-3-Groq-8B-Tool-Use · Hugging Face</a>: no description found</li><li><a href="https://www.amuse-ai.com/">Amuse</a>: Stable Diffusion Image and Video Generation</li><li><a href="https://huggingface.co/lmstudio-community/Phi-3.1-mini-128k-instruct-GGUF">lmstudio-community/Phi-3.1-mini-128k-instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/soft-kobe-bryant-no-smh-shaking-my-head-gif-18860898">Soft Kobe Bryant GIF - Soft Kobe Bryant No - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/meta-llama/llama-agentic-system/blob/main/llama_agentic_system/system_prompt.py">llama-agentic-system/llama_agentic_system/system_prompt.py at main · meta-llama/llama-agentic-system</a>: Agentic components of the Llama Stack APIs. Contribute to meta-llama/llama-agentic-system development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1267748792325771305)** (93 messages🔥🔥): 

> - `GPU Compatibility and Performance`
> - `LM Studio ROCm Release`
> - `Motherboard and GPU Upgrades`
> - `AI Streaming Setup`
> - `Driver and Cooling Issues` 


- **GPU Compatibility issues**: Members discussed that **Intel Iris Xe Graphics** are unsupported in LM Studio, requiring either NVIDIA with CUDA or AMD with ROCm for compatibility.
   - The conversation also touched on the performance of the **Tesla P40**, which while having more CUDA cores, faced issues in speed and compatibility compared to modern consumer GPUs.
- **LM Studio version 0.2.29 now on ROCm**: A query about the release of LM Studio 0.2.29 for ROCm was answered by highlighting that it is already available as indicated in the [GitHub release notes](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md).
   - Members expressed appreciation for this update and intentions to utilize the new features in their setups.
- **Upgrading Motherboard for AI Setup**: A user expressed intentions to upgrade their motherboard in order to accommodate two RT3060 GPUs for an AI streaming application, emphasizing the desire for improved VRAM capacity.
   - Discussions highlighted that using a second GPU could potentially be cost-effective and enhance performance without needing a complete rig overhaul.
- **Considerations for AI Streaming**: Participants debated possible AI integrations for game streaming, with some suggesting specific GPU setups to facilitate interaction and elevate viewing experiences.
   - There was a consensus that having dual GPUs might alleviate performance lags when running AI alongside gaming applications.
- **DIY GPU Cooling Solutions and Issues**: One member shared their DIY experience with creating air ducts for GPU cooling, advocating the use of a 3D printer for quality and convenience.
   - Concerns about driver compatibility and cooling efficiency were reiterated, emphasizing the importance of maintaining optimal temperatures during heavy workloads.



**Link mentioned**: <a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs

  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1267888908994347040)** (1 messages): 

> - `Perplexity Publishers Program`
> - `Partnerships with Media Organizations`
> - `Revenue Sharing Initiatives` 


- **Perplexity Launches Publishers Program**: Perplexity has launched its **publishers' program**, partnering with major organizations like [TIME](https://pplx.ai/publishers), [Der Spiegel](https://pplx.ai/publishers), and [Fortune](https://pplx.ai/publishers) to ensure access to reliable information.
   - This initiative aims to support publishers by providing new technology to engage audiences and promote collective success.
- **Emphasis on Trusted Information Sources**: The success of Perplexity is based on offering **high-quality answers** that hinge on **trusted sources** from organizations such as [The Texas Tribune](https://pplx.ai/publishers) and [WordPress.com](https://pplx.ai/publishers).
   - By incorporating citations in each answer, Perplexity aims to build user trust and ensure publishers receive proper credit.
- **Future Revenue Sharing for Publishers**: As part of the publishers' program, Perplexity will introduce **revenue sharing** models in the coming months, starting with advertising through related questions.
   - This move is designed to foster sustainable growth for media organizations while benefiting users with relevant content.



**Link mentioned**: <a href="https://pplx.ai/publishers">Introducing the Perplexity Publishers’ Program</a>: From day one, we’ve included citations in each answer, ensuring publishers receive proper credit and building user trust.

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1267560377571610714)** (209 messages🔥🔥): 

> - `Perplexity AI Down`
> - `Perplexity Publishers Program`
> - `AI Language Models Comparison`
> - `Impact of Pricing on Services`
> - `User Support and Experience` 


- **Perplexity AI experiencing downtime**: Users reported that **Perplexity AI** was down and unresponsive at various times, raising concerns about the service's reliability after recent updates.
   - Several users expressed frustration but noted that such outages are not unique to any one service.
- **Launch of Perplexity Publishers Program**: Perplexity announced a new program to share revenue with publishers, responding to criticism over content sourcing practices.
   - This move aims to foster ethical collaborations while also addressing backlash received from news outlets regarding content scraping.
- **Comparative Performance of AI Models**: Users compared the effectiveness of different AI models, including **Claude 3.5 Sonnet** and **GPT-4o**, noting strengths and weaknesses across various tasks.
   - Feedback suggests that while **Claude** performs well with specific outputs, **GPT-4o** is often praised for its accuracy, especially in coding.
- **User Concerns Regarding Paid Services**: There were concerns among paid users about whether advertisements would be served, with many expecting no interruptions in service for their subscription fee.
   - Some users expressed the sentiment that paying customers should not be treated as the product and should receive ad-free experiences.
- **Support and User Queries**: Users sought support regarding the functionality of the Perplexity app and possible issues with their accounts, expressing a need for effective communication.
   - Some users suggested that the silence from Perplexity regarding updates or support issues felt concerning and frustrating.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/7/25/24206488/openais-searchgpt-demo-results-arent-actually-that-helpful">OpenAI’s SearchGPT demo results aren’t actually that helpful.</a>: The trend of hallucinations showing up in public AI demos continues. As noted by a couple of reporters already, OpenAI’s demo of its new SearchGPT engine shows results that are mostly either wrong or ...</li><li><a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>: Self-attention performs well in long context but has quadratic complexity. Existing RNN layers have linear complexity, but their performance in long context is limited by the expressive power of their...</li><li><a href="https://www.perplexity.ai/hub/faq/what-is-perplexity-pro">What is Perplexity Pro?</a>: Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.</li><li><a href="https://x.com/aravsrinivas/status/1818279260517499062?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Today, we&#39;re announcing the Perplexity Publishers Program. Our success relies on ensuring answers are factually grounded in high-quality sources of information. The most scalable and sustainable w...</li><li><a href="https://knowyourmeme.com/memes/discord-user-is-a-suspected-terrorist-copypasta">Discord User Is a Suspected Terrorist Copypasta | Know Your Meme</a>: no description found</li><li><a href="https://tenor.com/view/second-futurama-scruffy-gif-20187509">Second Futurama GIF - Second Futurama Scruffy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://one.google.com">Get More Storage, More AI capabilities, and More Features - Google One</a>: no description found</li><li><a href="https://www.perplexity.ai/page/Complexity-Perplexitys-New-yl0q3mHYQz6RhRyuvjvN4w">Complexity: Perplexity&#x27;s New Extension</a>: The Complexity extension for Perplexity AI introduces a range of powerful features designed to enhance the user experience and streamline interactions with...</li><li><a href="https://techcrunch.com/2024/07/30/perplexitys-plan-to-share-ad-revenue-with-outlets-cited-by-its-ai-chatbot/?guccounter=1">Perplexity details plan to share ad revenue with outlets cited by its AI chatbot | TechCrunch</a>: Perplexity AI will soon start sharing advertising revenue with news publishers when its chatbot surfaces their content in response to a user query, a move
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1267674278762119168)** (2 messages): 

> - `Tesla's Charging Warning`
> - `Genetically Engineered Flies`
> - `Space Force Satellite Expansion` 


- **Tesla Issues Charging Warning**: Tesla has issued a warning regarding potential issues with its charging stations, prompting users to check compatibility and performance.
   - This has raised concerns among **Tesla owners** who heavily rely on **supercharging** for long-distance travel.
- **Genetically Engineered Flies Take on Waste**: Researchers have developed **genetically engineered flies** that consume organic waste, providing a solution to waste management challenges.
   - This innovative approach could potentially help in reducing **landfill waste** and improving **recycling** efforts.
- **Space Force Plans Satellite Expansion**: The **Space Force** announced plans to expand its satellite network to bolster national security and communication capabilities.
   - This move has sparked discussions about the implications of more **military satellites** in orbit and their impact on global governance.



**Link mentioned**: <a href="https://www.perplexity.ai/search/why-i-can-use-past-simple-in-p-eP05CN7fSXS7X.qRcV9ZMQ">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1267859635382124647)** (8 messages🔥): 

> - `Llama-3 model issues`
> - `Deprecation of Llama models`
> - `Request for citations in API` 


- **Llama-3 Models Producing Hallucinated Info**: A user reported being unable to use **llama-3-sonar-large-32k-online** due to it providing entirely **hallucinated** information, a problem not present until recently.
   - Another user echoed concerns about both old and new online models in labs failing to produce accurate results.
- **Upcoming Deprecation of Llama Models**: Members noted that **all Llama models** listed, including **llama-3-sonar-small-32k-online**, will be **deprecated** on August 12, 2024.
   - Concerns were raised about the efficacy of these models as users find them increasingly unreliable.
- **Request for API Citations**: A user brought attention to their request for **citations** in the API, stating it was **business critical** and reassured they hadn't received a response yet.
   - They shared a [link to their request](https://docs.perplexity.ai/discuss/66a8f6b588da9f0024012ab8), emphasizing the urgency for assistance.



**Link mentioned**: <a href="https://docs.perplexity.ai/discuss/66a8f6b588da9f0024012ab8">Request for citations in API</a>: no description found

  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1267702752092618792)** (1 messages): 

> - `Stable Artisan new command`
> - `Image style generation` 


- **Stable Artisan Introduces New Command /style**: A new command, **/style**, has been added to **Stable Artisan** allowing users to generate images based on a specified style and prompt.
   - For instance, users can create images like a **Van Gogh-style cat** or a **Japanese-style spaceship**.
- **Enjoy New Image Creation Features**: Members are encouraged to **enjoy** the new features of image creation by executing the /style command.
   - One user shared an example drawing of a **Van Gogh-style cute cat** to showcase the potential of the new command.


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1267558272299176027)** (215 messages🔥🔥): 

> - `Graphics Card Performance`
> - `Training LoRA Models`
> - `AI Animation Tools`
> - `SAM 2 Segment Anything Model`
> - `Stable Diffusion Configurations` 


- **OutOfMemoryError in Stable Diffusion**: Users reported encountering 'OutOfMemoryError' when generating images with Stable Diffusion, despite having 8GB GPUs and using SD1.5 models.
   - Recommendations included setting CUDA to 'prefer system fallback' and increasing virtual memory to alleviate memory limitations.
- **Training Consistent Characters with AI**: A user shared their struggles in achieving consistent character generation using various tools, including IP Adapter and ControlNet.
   - They provided their settings for IPAdapter but sought further advice on improving consistency in results.
- **AI Animation Tool Recommendations**: Users discussed various AI animation tools for creating minimalistic animations from static images, noting that Runway alters image quality.
   - Live Portrait AI was mentioned for its capabilities, although it primarily focuses on face animations rather than general image enhancements.
- **Introduction of SAM 2 for Video Segmentation**: The SAM 2 model from Meta promises robust segmentation capabilities for objects in both images and videos, facilitating real-time interactive applications.
   - It was noted that SAM 2 offers strong zero-shot performance, potentially benefiting animation remixes and other creative endeavors.
- **Card and Memory Requirements for Stable Diffusion**: Discussions highlighted the requirements of more RAM and effective settings for users running Stable Diffusion on various GPU configurations.
   - Users were advised to check if their setup met the necessary recommendations for optimal performance, particularly with AMD graphics cards.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fyrean.itch.io/bgbye-background-remover">BGBye - Background Remover by Fyrean</a>: Free background remover, 10 methods!</li><li><a href="https://www.amuse-ai.com/">Amuse</a>: Stable Diffusion Image and Video Generation</li><li><a href="https://youtu.be/XWuHPMvO-ps">Nvidias Cuda-System Fallback Policy &amp; What It Does! (DO NOT USE) Fortnite Tips&amp; Tricks)</a>: Discord:https://discord.gg/wHM5FyUqTxIn today&#39;s video I explained what the Cuda System fallback policy is in the NVidia control panel. This setting was just ...</li><li><a href="https://liveportrait.org/">Live Portrait AI - Bring Photos to Life with AI Animation</a>: no description found</li><li><a href="https://github.com/hayden-fr/ComfyUI-Model-Manager">GitHub - hayden-fr/ComfyUI-Model-Manager: Manage models: browsing, donwload and delete.</a>: Manage models: browsing, donwload and delete. Contribute to hayden-fr/ComfyUI-Model-Manager development by creating an account on GitHub.</li><li><a href="https://openart.ai/workflows/all">ComfyUI Workflows - Developer Community | OpenArt</a>: Discovery, share and run thousands of ComfyUI Workflows on OpenArt.</li><li><a href="https://github.com/pythongosssss/ComfyUI-Custom-Scripts">GitHub - pythongosssss/ComfyUI-Custom-Scripts: Enhancements &amp; experiments for ComfyUI, mostly focusing on UI features</a>: Enhancements &amp; experiments for ComfyUI, mostly focusing on UI features - pythongosssss/ComfyUI-Custom-Scripts</li><li><a href="https://openart.ai/workflows/home">ComfyUI Workflows - Developer Community | OpenArt</a>: Discovery, share and run thousands of ComfyUI Workflows on OpenArt.</li><li><a href="https://ai.meta.com/SAM2/">no title found</a>: no description found
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1267560106950656171)** (63 messages🔥🔥): 

> - `Unsloth usage on Windows`
> - `LLM fine-tuning discussions`
> - `Matrix representation with custom tokens`
> - `Support for rope scaling`
> - `Dataset building tools` 


- **Unsloth struggles on Windows**: A member reported getting a 'No triton module' error when trying to use Unsloth on Windows, to which others suggested switching to WSL.
   - Another user humorously commented about their refusal to leave Windows due to gaming preferences.
- **Fine-tuning LLMs with new datasets**: Discussion arose about fine-tuning a Llama3 model while avoiding catastrophic forgetting, leading to the suggestion of combining datasets for retraining.
   - Users confirmed that complete retraining is preferable to mitigate risks associated with catastrophic forgetting.
- **Representing matrices with custom tokens**: One user inquired about effectively representing a 30x30 matrix with custom tokens for their project Arc-AGI.
   - Another member asked for clarification on their request, indicating a need for more details.
- **Rope scaling support improves**: A member shared information that older models previously lacked support for rope scaling but confirmed that it is now implemented in Unsloth as of two weeks ago.
   - Users expressed excitement over the new capability, with one person mentioning Phi-3 128k variants in relation to the feature.
- **Building datasets using custom tools**: A user sought recommendations for tools to create custom dataset files from their seed data, including Python and math resources.
   - They referenced a specific tool called Agent Chef from GitHub to assist in dataset refinement and structuring.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1818341828254744922">Tweet from Unsloth AI (@UnslothAI)</a>: Watch a step-by-step video tutorial by @engineerrprompt on how to fine-tune Llama 3.1 using your own data:  https://www.youtube.com/watch?v=rpAtVIZB72U</li><li><a href="https://x.com/TheXeophon/status/1817991874134569012">Tweet from Xeophon (@TheXeophon)</a>: The on-device LLM is 2.73B params, RUNS ON &lt;4-BIT QUANT, uses LoRA adapters  Two stage pre-training, followed by SFT (w/ synth. data), RLHF (iTeC, MDLOO (both new))  MMLU: 61.4 (On-Device), 75.4 (S...</li><li><a href="https://github.com/Leoleojames1/Agent_Chef">GitHub - Leoleojames1/Agent_Chef: Agent Chef is our robust tool for dataset refinement, structuring, and generation. By leveraging procedural and synthetic dataset generation techniques, Agent Chef will enable users to refine and clean their fine-tuning data, eliminating data poisoning and low-quality knowledge bases. Additionally, it will provide templates, and frameworks.</a>: Agent Chef is our robust tool for dataset refinement, structuring, and generation. By leveraging procedural and synthetic dataset generation techniques, Agent Chef will enable users to refine and c...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1267560593825599590)** (6 messages): 

> - `Fine-Tuning`
> - `Applied LLMs Course`
> - `Prompt Engineering Video`
> - `LLM Educational Resources` 


- **Exploring Business Opportunities in Fine-Tuning**: A member initiated a discussion about potential business opportunities for ML engineers focusing on **fine-tuning** models.
   - This prompted others to consider collaboration in this area.
- **Free Resources from Applied LLMs Course Released**: According to [HamelHusain](https://x.com/HamelHusain/status/1817935895246635362), free resources from their **Applied LLMs course** have been made available today, enhancing lesson materials with additional learning tracks.
   - These resources aim to maximize learning for all participants.
- **Direct Links to LLM Courses Found**: A member shared a link to [Parlance Labs](https://parlance-labs.com/education/) which offers extensive educational materials on topics such as **fine-tuning**, **prompt engineering**, and performance evaluation for LLMs.
   - The page features various categories to help steer learning on specific applications of LLM technology.
- **New Prompt Engineering Video on Fine-Tuning**: A new video tutorial by **Prompt Engineering** demonstrates how to fine-tune **Llama 3.1** using personal data, shared by [UnslothAI](https://x.com/UnslothAI/status/1818341828254744922).
   - The tutorial can be viewed on YouTube [here](https://www.youtube.com/watch?v=rpAtVIZB72U) for a step-by-step guide to the process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1818341828254744922">Tweet from Unsloth AI (@UnslothAI)</a>: Watch a step-by-step video tutorial by @engineerrprompt on how to fine-tune Llama 3.1 using your own data:  https://www.youtube.com/watch?v=rpAtVIZB72U</li><li><a href="https://parlance-labs.com/education/">Parlance - Educational Resources</a>: Educational resources on LLMs</li><li><a href="https://x.com/HamelHusain/status/1817935895246635362">Tweet from Hamel Husain (@HamelHusain)</a>: If you remember our Applied LLMs course, you&#39;ll love this.  Today, we are making all these resources available for free to everyone! 📚   We did extra work to add learning tracks, resources, and n...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1267563667071959153)** (142 messages🔥🔥): 

> - `Fine-tuning models`
> - `Memory management techniques`
> - `Model conversion`
> - `Model performance metrics`
> - `Multi-GPU usage issues` 


- **Troubleshooting fine-tuning with Unsloth**: Users are experiencing issues loading models in Unsloth, specifically with the `OSError` indicating that unsloth is not a valid git identifier when using finetuned Llama models.
   - A workaround was shared involving setting the `revision` to `null` in the `adapter_config.json` file based on a GitHub issue.
- **Memory offloading capabilities**: There is some discussion on whether Unsloth supports memory offloading between RAM and GPU, with confirmation that it's complicated.
   - Gradient checkpointing was mentioned as an existing method for managing memory usage in training.
- **Model performance metrics on different setups**: Several users are sharing their token-per-second (TPS) performance metrics while using various models and frameworks, noting that a single request TPS for a Llama model can vary widely.
   - One user reported achieving 150 TPS with an 8-bit quantized Llama model, while another shared achieving higher TPS with batched requests.
- **Conversion of models and setup issues**: Users inquired about converting `safetensors` models to `gguf` using Unsloth and discussed installation issues encountered in environments like Yandex DataSphere.
   - Concerns around disk space limitations were raised, prompting suggestions to save LoRA adapters only.
- **Multi-GPU usage challenges**: A user is attempting to configure a multi-GPU setup without specifying `CUDA_VISIBLE_DEVICES`, facing restrictions on shared HPC resources.
   - The conversation highlighted that Unsloth currently does not support multi-GPU setups, complicating the user's ability to utilize available hardware.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rohanpaul_ai/status/1818063089872384348?s=46">Tweet from Rohan Paul (@rohanpaul_ai)</a>: 📌 LoRA adapters fine-tune the foundation models for specific tasks.  📌 Adapters are applied to all linear projection matrices in self-attention layers and fully connected layers in feedforward netwo...</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-12.-saving-the-model">How to Finetune Llama-3 and Export to Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://www.unsloth.ai/blog/llama3">Finetune Llama 3 with Unsloth</a>: Fine-tune Meta&#x27;s new model Llama 3 easily with 6x longer context lengths via Unsloth!</li><li><a href="https://github.com/unslothai/unsloth/issues/492">issue loading lora model  · Issue #492 · unslothai/unsloth</a>: while trying to load a trained llama3-instruct lora model , I am getting this error. However, It was working fine like 2 days ago. OSError: unsloth is not a valid git identifier (branch name, tag n...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1267689904742076581)** (4 messages): 

> - `Translation datasets`
> - `Continued pretraining`
> - `Blockchain Engineer portfolio` 


- **Inquiry about Translation Datasets**: A member inquired if there is a translation dataset available for fine-tuning models from **English to any language**, intending to use DeepL for this task.
   - Another member suggested utilizing **Wikipedia** as a resource.
- **Insight on Continued Pretraining**: A member explained that **Continued Pretraining** (CPT) enables models to learn new languages and understand different domains of knowledge, as outlined in their [documentation](https://docs.unsloth.ai/basics/continued-pretraining).
   - They provided links to a [text completion notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing) and a [continued pretraining notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing) for further learning opportunities.
- **Blockchain Engineer Portfolio Showcase**: A member shared their [portfolio](https://alex-portfolio.pages.dev/) as a **Blockchain Engineer** with 5 years of experience, specializing in **Cosmos SDK** and **substrate**.
   - They highlighted their experience in developing bridging protocols, zkProof, and configuring architecture on cloud platforms, inviting anyone interested to **DM** for services.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://alex-portfolio.pages.dev/">Portfolio - Alex Davis</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1267787228843606037)** (8 messages🔥): 

> - `Randomized SVD`
> - `Jacobian method for SVD`
> - `CUBLAS vs CUTLASS`
> - `Accel Venture Capital`
> - `Learning materials for distributed training` 


- **Randomized SVD simplifies large problems**: Randomized SVD does not replace SVD but reduces large-scale problems involving big matrices to smaller, more manageable skinny matrices for standard SVD processing.
   - This approach provides a good approximation of the first few singular values and vectors of the original matrix.
- **Jacobian method kernel faces NA issues**: A member created a [Triton kernel](https://link.to/kernel) for SVD using the Jacobian method but reported only experiencing NaN results.
   - They plan to share it soon, expressing that it could be useful for others.
- **CUBLAS versus CUTLASS debate**: A member inquired if it is still advisable to use **CUBLAS** over the now widely adopted **CUTLASS**.
   - This reflects ongoing discussions about optimal tools in GPU computing.
- **Understanding Accel Venture Capital**: In response to a query, it was clarified that Accel is a venture capital firm, highlighted by a link to their [website](https://www.accel.com/).
   - They are known for hosting events and supporting exceptional teams in tech.
- **Learning materials for distributed training**: A member asked for recommendations on learning materials covering **FSDP**, **TP**, and **PP** for distributed training.
   - They specifically sought principles to improve performance in these areas.



**Link mentioned**: <a href="https://www.accel.com/">Accel</a>: Accel is a global venture capital firm, and the first partner to exceptional teams from seed to IPO. Facebook, Flipkart, CrowdStrike, UiPath, and Spotify are among the companies Accel has backed over ...

  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1267587988582305908)** (3 messages): 

> - `PTX Output Verification`
> - `Libdevice Non-Approximated Exponential`
> - `Code Reflection in Java for Triton`
> - `IR and MLIR Dialects` 


- **PTX Output Verification needed**: A user questioned the validity of a process stating it should be checkable by looking at the **PTX (assembly)** it outputs.
   - This highlights the importance of PTX verification in confirming expected functionality.
- **Libdevice has Non-Approximated Exponential Function**: Another member confirmed that **libdevice** indeed has a non-approximated exponential by referencing its [GitHub implementation](https://github.com/triton-lang/triton/blob/2db56689b0d1268f09dd99cabe4ca940d710da7e/python/triton/language/extra/cuda/libdevice.py#L1156).
   - This reference underscores the documentation's significance in understanding Triton's capabilities.
- **Exploring Code Reflection for Triton in Java**: An article was shared explaining how to implement the [Triton](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) programming model in Java using **Code Reflection**.
   - The article, a part of OpenJDK Project [Babylon](https://openjdk.org/projects/babylon/), sheds light on Java's potential as an alternative to Python in Triton programming.
- **Details on IR and MLIR Dialects**: The discussed article delves into **IR** and **MLIR dialects**, presenting concepts related to code reflection in the context of Triton.
   - This focus on IR and MLIR offers valuable insight into the future of Triton development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openjdk.org/projects/babylon/articles/triton">Exploring Triton GPU programming for neural networks in Java</a>: no description found</li><li><a href="https://github.com/triton-lang/triton/blob/2db56689b0d1268f09dd99cabe4ca940d710da7e/python/triton/language/extra/cuda/libdevice.py#L1156">triton/python/triton/language/extra/cuda/libdevice.py at 2db56689b0d1268f09dd99cabe4ca940d710da7e · triton-lang/triton</a>: Development repository for the Triton language and compiler - triton-lang/triton
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1267992893512224818)** (2 messages): 

> - `CUDA Memory Alignment`
> - `PyTorch Tensor Alignment` 


- **CUDA Memory Alignment from Caching Allocator**: A question was raised about whether the GPU memory returned from the CUDA caching allocator is always aligned to certain bytes for safe reinterpretation.
   - One member recalled that the allocator generally ensures alignment, but cautioned that **not all tensor pointers in PyTorch are guaranteed to be aligned**.
- **Concerns about PyTorch Tensor Alignment**: Members discussed the implications of tensor pointer alignment in PyTorch when performing vectorized access operations.
   - This discussion highlighted potential issues that could lead to *CUDA error: misaligned address* when using reinterpret_cast.


  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1267896765118681260)** (1 messages): 

> - `CUDA MODE IRL Meeting`
> - `Keynotes by ML systems leaders`
> - `Working Groups and Hack Leads`
> - `Fireside Chat with Wen-mei Hwu`
> - `Sponsorship and Registration` 


- **Join the CUDA MODE IRL Meeting!**: CUDA MODE is meeting **IRL for the first time** on **Sep 21** in SoMa SF, with [details here](https://events.accel.com/cudamode) to hack on exciting projects.
   - Registration is limited to **150 builders**, so apply soon if you're interested!
- **Keynote Presentations by ML Icons**: Attendees will enjoy keynotes from renowned leaders in **ML systems** including **Karpathy**, **Tri Dao**, and **Supriya Rao**.
   - Following the keynotes, participants will split into working groups to collaborate on innovative projects.
- **Get Help from Hack Leads**: **Familiar names** from the server will act as Hack leads, assisting participants to overcome challenges during the sessions.
   - This structure aims to foster collaboration and creativity throughout the event.
- **Fireside Chat and Book Signing**: A highlight of the day includes a break for a **fireside chat** with **Wen-mei Hwu**, alongside a **book signing**.
   - Additionally, **Lily Liu** will also be speaking during this time, adding to the event's richness.
- **Sponsorship and Free Compute Credits**: The event is sponsored by **Accel**, **NVIDIA**, and **PyTorch**, with more sponsors and announcements coming soon.
   - Details regarding **free compute credits** and **prize money** will be shared as the date approaches.


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1267859325007958137)** (2 messages): 

> - `Machine Intelligence Advances`
> - `PIM Rediscovery` 


- **Recent breakthroughs in machine intelligence**: Advancements in **machine intelligence** have been highlighted, especially in areas like recommender systems, speech recognition, and natural language processing.
   - These developments are discussed in a recent article, which provides various **inline references** to foundational papers, reflecting the evolution and impact of this technology.
- **Rediscovery of PIM**: A humorous remark was made about the community's recent attention towards **PIM (Personal Information Management)**, with some suggesting it has been 'rediscovered'.
   - This led to light-hearted discussions about the implications and relevance of PIM in today's tech landscape.



**Link mentioned**: <a href="https://www.nature.com/articles/s44335-024-00003-3">Experimental demonstration of magnetic tunnel junction-based computational random-access memory - npj Unconventional Computing</a>: no description found

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1267647819330752622)** (1 messages): 

> - `CUDA Memory Types`
> - `VRAM and Caches`
> - `Memory Management in CUDA` 


- **Understanding CUDA Memory Types**: In CUDA, **global memory** is primarily backed by **VRAM**, which is cached in both **L2** and occasionally **L1TEX** when accessed.
   - Each memory type is essentially an abstraction, with **local memory** attempting to map to registers or cached global memory.
- **Caching Mechanisms Explained**: **Texture memory** has a unique layout and caching strategy, while **shared memory** is part of **L1TEX** and can be managed manually for efficiency.
   - **Constant memory** benefits from improved caching due to its unchanging nature, which allows for better memory subsystem performance.
- **VRAM Usage Considerations**: The size of **global memory** must be less than **VRAM**, since the driver and runtime require a portion of VRAM for their own functions.
   - Thus, drivers and caching take up some of the VRAM resources, limiting the available space for global memory.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1267675401228914749)** (19 messages🔥): 

> - `Optimizer CPU Offload`
> - `FSDP and Optimizers`
> - `Low-Bit Optimizers`
> - `CUDA Streams and Transfers`
> - `Pytorch AO Tutorials` 


- **Exploring Optimizer CPU Offload**: Members discussed implementing a `cpu_offload` flag to allocate optimizer states on CPU, moving parameters during optimization steps before transferring them back to CUDA.
   - One participant noted that the optimizer step is blocking, leading to concerns about the feasibility of interleaved operations with `torch.compile`.
- **FSDP's Handling of CPU Offload**: It was clarified that FSDP keeps a 'master copy' of parameters on CPU, allowing gradients to be moved efficiently without requiring parameter transfers.
   - Participants shared links to PyTorch documentation explaining how sharded parameters and gradients are managed during optimizer steps.
- **Considerations on Low-Bit Optimizers**: Discussions included whether a low-bit optimizer is necessary when implementing CPU offload, highlighting potential benefits in CPU RAM savings.
   - One member proposed maintaining a set of master parameters to optimize RAM usage even further.
- **Curious about CUDA Stream Functionality**: Members inquired about the specifics of queuing transfers in separate CUDA streams, referencing resources on setup and running CUDA operations.
   - A participant shared a GitHub link detailing the role of CUDA streams and how they manage device contexts.
- **Blog Post Ideas on Quantization**: A user proposed writing a blog post/tutorial focused on adding quantization steps to Karpathy's build-nanogpt tutorial using PyTorch AO.
   - They sought suggestions on emphasizing the value proposition of AO, particularly in benchmarking and datatype strategies for GPT models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/msaroufim/tinyoptimizer/blob/master/activation_offload.py">tinyoptimizer/activation_offload.py at master · msaroufim/tinyoptimizer</a>: Contribute to msaroufim/tinyoptimizer development by creating an account on GitHub.</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams">CUDA semantics &mdash; PyTorch 2.4 documentation</a>: no description found</li><li><a href="https://github.com/pytorch/ao/issues/426">The next tutorials · Issue #426 · pytorch/ao</a>: From our README.md torchao is a library to create and integrate high-performance custom data types layouts into your PyTorch workflows And so far we&#39;ve done a good job building out the primitive d...</li><li><a href="https://github.com/pytorch/pytorch/blob/32c57e78edc46aa71ed19e013741c65b3d777fe9/torch/distributed/_composable/fsdp/_fsdp_param.py#L612-L615">pytorch/torch/distributed/_composable/fsdp/_fsdp_param.py at 32c57e78edc46aa71ed19e013741c65b3d777fe9 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/3864a2d834e3dc84adad791b6fab0c0d401f7e96/torch/distributed/_composable/fsdp/_fsdp_collectives.py#L337-L339">pytorch/torch/distributed/_composable/fsdp/_fsdp_collectives.py at 3864a2d834e3dc84adad791b6fab0c0d401f7e96 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1267565640034553906)** (14 messages🔥): 

> - `SAM 2 demo`
> - `Object Tracking Challenges` 


- **SAM 2 demo impresses viewers**: A member shared a video demo of [SAM 2](https://sam2.metademolab.com/shared/d8ddf358-7f3a-452b-be9c-f9bd690c9d07.mp4) showcasing its features, eliciting enthusiastic reactions with multiple fire emojis.
   - One member commented on the **before & after** comparison, emphasizing the visual improvements.
- **Challenges with object selection in SAM 2**: Members discussed the limitation in **SAM 2** regarding object tracking, noting that the FPS and pixel resolution must match the input for effectiveness.
   - It was pointed out that while you can add positive and negative points, there wasn't an option to label multiple objects easily; instead, one had to pause and return to the timeline to select again.



**Link mentioned**: <a href="https://sam2.metademolab.com/shared/d8ddf358-7f3a-452b-be9c-f9bd690c9d07.mp4)">SAM 2 Demo | By Meta FAIR</a>: Track an object across any video and create fun effects interactively, with as little as a single click on one frame.

  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1267926065494495416)** (1 messages): 

> - `Apple and HQQ+`
> - `LoRA adapters`
> - `Quantization loss recovery` 


- **Apple 'rediscovered' HQQ+**: Apple's recent work on **HQQ+** has sparked conversation, particularly in light of a timeline where **4chan** introduced quantization loss recovery techniques for **LoRAs** three months prior.
   - A member shared a [tweet](https://x.com/teortaxesTex/status/1818289206948716660) highlighting this sequence of events and the perceived trend of Apple adopting previously established concepts.
- **Insights from Apple Paper on LoRA**: In the Apple paper, it was noted how they implement a **LoRA adapter** on top of quantized models, which aids in recovering accuracy.
   - As expressed in the discussion, *task-specific adapters are fine-tuned from this accuracy-recovering base*, making it a significant approach in their methodology.



**Link mentioned**: <a href="https://x.com/teortaxesTex/status/1818289206948716660">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: 4chan did quantization loss recovering LoRAs 3 months before Apple made it cool btw  Quoting Blaze (Balázs Galambosi) (@gblazex)   One of the most interesting things in the Apple paper for me was how ...

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1267558668153520169)** (125 messages🔥🔥): 

> - `Llama 3 Finetuning`
> - `RoPE Integration`
> - `SwiGLU Implementation`
> - `Training Code Developments`
> - `Model Evaluations and Discussions` 


- **Finetuning Llama 3.1 for Jeopardy**: A member is finetuning **Llama 3.1 8B** using **Unsloth** and feels uncertain about the configuration due to the complexity of the code with many arguments.
   - They seek a more straightforward setup, emphasizing a preference for a **stable bf16 finetuning** process.
- **Discussion on RoPE and Its Implementation**: The group discusses the implications of integrating **RoPE** into the training process, questioning whether it should be included given that **GPT-2-xl** did not use it.
   - Concerns arise regarding the code becoming overly complex, but members acknowledge RoPE's potential benefits for performance.
- **Progress on SwiGLU Implementation**: Members share the challenges of implementing **SwiGLU**, with one confirming they trained a few models on it, noting divergent training dynamics.
   - They discuss the complexities involved and contemplate making **SwiGLU** trainable, weighing its benefits against the effort required.
- **Building Modular Training Code**: There is a push toward developing a clean and modular training setup, with plans to fork and handle specific changes for **Llama 3** separately.
   - Members agree on the necessity of adapting the reference python code to incorporate features like **GQA**, **SwiGLU**, and **RoPE**.
- **Challenges with Meta's Official Code**: Concerns were expressed about the reliability of **Meta's official Llama 3** code, citing mistakes in the documentation and uncertainty about its use.
   - One member is focused on creating a simplified version of the **Llama 3 nn.Module**, favoring independent validation over external resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/zealandic1/status/1818349493055807929">Tweet from Anthonix (@zealandic1)</a>: GPT2 modified w/ SwiGLU+nobiases seems to converge best with 2x LR (~0.0036), compared to 0.0018 that @Yuchenj_UW found was best for stock GPT2 config.  These are all ~124M, the FFN was scaled as in t...</li><li><a href="https://github.com/karpath">karpath - Overview</a>: GitHub is where karpath builds software.</li><li><a href="https://hf.co/chat/assistant/66a88b3fc2901bf800fcdeae">GitHub llm.c Repo (L3.1-405b) - HuggingChat</a>: Use the GitHub llm.c Repo (L3.1-405b) assistant inside of HuggingChat</li><li><a href="https://github.com/karpathy/llm.c/pull/718">Add SwiGLU support by gordicaleksa · Pull Request #718 · karpathy/llm.c</a>: Implemented SwiGLU - swish GLU activation function from the &quot;GLU Variants Improve Transformer&quot; paper. Note: there is an increase in memory footprint as a consequence of adding an additional ...</li><li><a href="https://github.com/karpathy/llm.c/pull/708">Add high perf mode by gordicaleksa · Pull Request #708 · karpathy/llm.c</a>: Add:  Warnings when we take a suboptimal branch High perf mode that will exit immediately if we&#39;re not running using all of the most optimal branches  Also added a fwd kernel config that will be u...</li><li><a href="https://github.com/karpathy/llm.c/pull/715">Feature/restore from master by karpathy · Pull Request #715 · karpathy/llm.c</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1267969374858383361)** (7 messages): 

> - `WebGPU API`
> - `gpu.cpp usage`
> - `Realtime multimodal integration`
> - `Hybrid model computation`
> - `Local device computation` 


- **WebGPU API: More than Just a Browser Tool**: WebGPU functions as an API spec that includes a small language definition for **WGSL** (WebGPU Shading Language), allowing for shallow compilation to compatible shaders in **Metal**, **Vulkan**, and **DirectX**.
   - Originally aimed at browser use, it has been adopted for native applications, particularly with **Rust** and **Zig** implementations.
- **Prioritizing Simplicity: gpu.cpp**: The **gpu.cpp** project leverages WebGPU without creating a new language, utilizing **WGSL** for shader code and simplifying the API integration within **C++** projects.
   - This approach aims to abstract tedious aspects of the raw API, making it more user-friendly.
- **Realtime Integration with Multimodal IO**: One user expressed a desire to utilize capabilities for integrating models with **real-time multimodal (audio and video) input/output** as a primary application.
   - This interest extends to simulations and exploring branching/conditional computations over models.
- **Exploring Hybrid Model Computation**: Discussions included enthusiasm for various forms of **hybrid model computation**, combining **CPU SIMD** with **GPU** processing or integrating local and remote computations.
   - This approach could enhance performance and versatility of model deployment in diverse environments.
- **Convenient Substrate for Local Device Computation**: A user noted the motivation to explore **local device computation** using a portable GPU API in **C++**, which facilitates various new applications.
   - This combination provides an accessible way to experiment with device capabilities and computation techniques.


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1267897789938139248)** (13 messages🔥): 

> - `Event Registration`
> - `Keynote Recording`
> - `Attendee Guidelines`
> - `GPU Access`
> - `Event Excitement` 


- **Excitement builds for the upcoming event**: Members expressed their enthusiasm about the event, with one noting they were 'super excited this is finally happening!'
   - *Marksaroufim* echoed the sentiment, showing excitement as well.
- **Keynotes will be recorded**: There is an intent to record keynote addresses both in the server and at the IRL venue, ensuring audiences can access them later.
   - *Marksaroufim* confirmed the plan to document these keynotes.
- **Registration required for attendees**: Attendees should ensure they register for the event, as *Marksaroufim* encouraged one member to confirm their registration.
   - Questions about attendance were addressed, clarifying that registration is essential.
- **GPU access and requirements discussed**: A member inquired whether they would need to bring their own GPU or if compute access would be provided.
   - *Marksaroufim* indicated they are raising funds from sponsors, hinting at potential resources for attendees.
- **Confirmation emails for registrants**: After registering for the IRL event, members were curious about receiving confirmation emails regarding their approval to attend.
   - *Marksaroufim* assured that they would confirm approval with attendees soon.


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

_paradroid: https://arxiv.org/abs/2407.04620
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

mautonomy: <:uhhh:1133962718349639721> <:thinking:1134948374760669225>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1267855729146990613)** (5 messages): 

> - `AI Model Size Comparison`
> - `AI Interpretability Concerns` 


- **Smaller Models Show Competitive Edge**: A recent paper suggests that running a **70B model** once versus generating five outputs from a **13B model** can produce consistent improvements, yielding gains of up to **15% across five tasks**.
   - *This begs the question: what happens when both models operate under the same budget?* The findings emphasize the importance of **unit-test setups** for selecting the best outputs.
- **Skepticism Around AI Interpretability Timeline**: Concerns were raised about the timeline for achieving **AI interpretability**, with some suggesting it may take a few more years to obtain reliable datasets outside of private practices.
   - The sentiment reflected a belief that longer timelines for public data releases could be beneficial, allowing more robust findings to emerge.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.00725">The Larger the Better? Improved LLM Code-Generation via Budget Reallocation</a>: It is a common belief that large language models (LLMs) are better than smaller-sized ones. However, larger models also require significantly more time and compute during inference. This begs the ques...</li><li><a href="https://tenor.com/view/jim-carrey-gif-12171108510331032271">Jim Carrey GIF - Jim carrey - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1267559570843111565)** (79 messages🔥🔥): 

> - `Apple AI Models`
> - `Hermes Model Merging Techniques`
> - `Midjourney V6.1 Launch`
> - `GPT-4o Capabilities Report`
> - `New MRLM Method` 


- **Apple AI Models Benchmark Insights**: A member highlighted interesting details in the new Apple paper regarding their server-side and on-device models with significant benchmarks like **MMLU** scores of **61.4** for on-device and **75.4** for server models.
   - The paper details a two-stage pre-training process alongside SFT and RLHF methods.
- **Exploring Techniques for Hermes and Llama Model Merging**: A user asked about merging techniques for Hermes models with Llama, with suggestions of potential write-ups being prepared on creating effective merges.
   - Discussion ensued regarding the impact of various techniques on performance and compatibility.
- **Midjourney V6.1 Enhancements**: Midjourney announced the release of **V6.1**, promising significant improvements in image quality, coherence, and new upscaling models.
   - This update was released shortly after a user claimed to have achieved state-of-the-art results in image generation.
- **Upcoming GPT-4o Capabilities Report**: Anticipation grew around the forthcoming detailed report on **GPT-4o's** capabilities and safety evaluations expected in early August.
   - Members expressed curiosity about performance metrics, especially relating to image and audio generation.
- **New MRLM Method for Self-RL Models**: A user discussed a new **MRLM method** for self-rewarding language models, indicating impressive performance improvements over previous methods.
   - However, concerns lingered regarding the lack of benchmarks for established metrics like **MMLU**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/midjourney/status/1818342703618482265">Tweet from Midjourney (@midjourney)</a>: Midjourney V6.1 is now live! V6.1 greatly improves image quality, coherence, text, and comes with brand-new upscaling and personalization models. It’s smarter, faster, clearer, and more beautiful. We ...</li><li><a href="https://mihaiii-trivia.hf.space/">FastHTML page</a>: no description found</li><li><a href="https://x.com/TheXeophon/status/1817991874134569012">Tweet from Xeophon (@TheXeophon)</a>: The on-device LLM is 2.73B params, RUNS ON &lt;4-BIT QUANT, uses LoRA adapters  Two stage pre-training, followed by SFT (w/ synth. data), RLHF (iTeC, MDLOO (both new))  MMLU: 61.4 (On-Device), 75.4 (S...</li><li><a href="https://x.com/dylan522p/status/1818414482051235994">Tweet from Dylan Patel (@dylan522p)</a>: When faced with a founder who has significant compute resources, the dominant male will bring a puffier leather jacket in a bid to win the competition of luring mates. This contest is similar to that ...</li><li><a href="https://learn.deeplearning.ai/courses/ai-agents-in-langgraph/lesson/1/introduction">DeepLearning.AI - AI Agents in LangGraph</a>: Introduction · Build an Agent from Scratch · LangGraph Components · Agentic Search Tools · Persistence and Streaming · Human in the loop · Essay Writer · LangChain Resources · Conclusion</li><li><a href="https://til.simonwillison.net/llms/python-react-pattern">A simple Python implementation of the ReAct pattern for LLMs</a>: A popular nightmare scenario for AI is giving it access to tools, so it can make API calls and execute its own code and generally break free of the constraints of its initial environment.</li><li><a href="https://tenor.com/view/muahaha-evil-laugh-evil-laugh-futurama-gif-4133163">Professor Farnsworth - Evil Laugh GIF - Muahaha Evil Laugh Evil - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1267561183716573298)** (18 messages🔥): 

> - `NousResearch/Meta-Llama-3.1-8B-Instruct differences`
> - `Theta performance issues`
> - `Hermes 3 and custom datasets`
> - `BigCodeBench leaderboard` 


- **NousResearch/Meta-Llama-3.1-8B-Instruct is gated**: The primary difference between the **NousResearch/Meta-Llama-3.1-8B-Instruct** model and the original is that the NousResearch repo isn't gated.
   - This open access allows users to utilize the model without the usual restrictions.
- **Theta's unique performance challenges**: There are ongoing issues with **Theta**, as it is supposed to utilize the same system prompt as other models like **Llama 3** but does not function identically.
   - Members noted the complexities in discrepancies with model training resulting in varying behaviors.
- **Hermes 3's dataset reliance confirmed**: **Hermes 3** and its **Pro** version will solely use a custom dataset, suggesting a focused training approach.
   - If there is another merge, it would be named **Hermes 3 Theta**, indicating continuation of model updates.
- **BigCodeBench boasts a leaderboard**: A member suggested exploring **BigCodeBench** for its code generation tasks leaderboard, implying it has merit as a comparison tool.
   - This led to discussions on **Hugging Face** leaderboards, with participants expressing varying familiarity with existing options.



**Link mentioned**: <a href="https://huggingface.co/spaces/mike-ravkine/can-ai-code-results">Can Ai Code Results - a Hugging Face Space by mike-ravkine</a>: no description found

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 messages): 

teknium: https://x.com/omarsar0/status/1818139150882664696
  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/)** (1 messages): 

n8programs: awesome
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1267912672276512881)** (1 messages): 

> - `Advanced Voice Mode Rollout`
> - `Safety Reinforcements for Voice Conversations`
> - `GPT-4o Voice Capabilities Testing`
> - `Planned Features for ChatGPT Plus`
> - `Privacy Measures in Voice Mode` 


- **Advanced Voice Mode Begins Rollout**: Advanced Voice Mode is starting to roll out to a small group of **ChatGPT Plus** users, offering more natural, real-time conversations with the ability to interrupt anytime.
   - Email instructions and mobile app messages have been sent to participants, with plans for wider access by fall.
- **Focus on Safety in Voice Conversations**: The development team has been working to enhance the **safety and quality** of voice conversations as they prepare to launch this new technology.
   - Guardrails have been put in place to block requests for **violent or copyrighted content**, ensuring a safer user experience.
- **Extensive Testing of GPT-4o's Voice Features**: The team tested **GPT-4o's voice capabilities** with over 100 external red teamers across 45 languages to assess performance.
   - Learnings from these tests will inform improvements and **safety measures** for the Advanced Voice experience.
- **Future Features: Video and Screen Sharing**: In addition to voice features, **video and screen sharing** capabilities are planned for introduction at a later date.
   - This enhancement aims to further enrich conversations and interactions within the **ChatGPT Plus** environment.
- **Privacy Protections in Advanced Voice Mode**: To protect user **privacy**, GPT-4o will only use four preset voices and systems have been built to block outputs differing from those voices.
   - A detailed report on GPT-4o’s capabilities, limitations, and **safety evaluations** is expected in early August.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1267570090702802995)** (51 messages🔥): 

> - `Search GPT access`
> - `AI-DALLE 3 challenges`
> - `GPT-4o advanced vision release`
> - `AGI discussions`
> - `Midjourney V6.1 release` 


- **Search GPT Access Available**: Members confirmed access to [Search GPT](https://link.to.searchgpt) and offered assistance to those in need.
   - *It's noted that some found it helpful, while others were uncertain about its capabilities.*
- **Issues with AI-DALLE 3 Challenges**: A participant reported difficulties accessing challenges in **AI-DALLE 3**, expressing *frustration after multiple attempts*.
   - They shared details about needing assistance and community support in resolving this access issue.
- **Expected GPT-4o Advanced Features**: Discussion sparked about the anticipated release of **GPT-4o advanced vision and voice**, with mixed expectations on timing, suggesting it may be pushed to next month.
   - Another member mentioned a potential release at the **end of this month in alpha**.
- **Debate on AGI Understanding**: Some members engaged in a discussion about the ambiguity surrounding the **AGI** concept, noting the lack of a set definition.
   - Opinions varied, with one emphasizing its *interesting nature* and complexity.
- **Excitement for Midjourney V6.1**: Members celebrated the recent launch of **Midjourney V6.1**, praising its impressive capabilities in image generation.
   - Discussions highlighted how it excels in text transformations and potential use cases, with *enthusiasm noted for its image-to-audio transformation potential*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1267567418851721379)** (14 messages🔥): 

> - `GPT Vision Release`
> - `OpenAI Assistant Discussions`
> - `Quality of GPT Responses`
> - `API Response Size for GPT Actions`
> - `Memory Functionality in GPT` 


- **Inquiry about GPT Vision Release**: A user asked if **GPT Vision** is available yet, indicating a growing interest in its potential features.
   - The conversation reflects anticipation surrounding the latest updates from OpenAI.
- **Questions on OpenAI Assistants**: A user sought confirmation on whether this channel is suitable for questions about **OpenAI assistants**.
   - This inquiry points to a desire for clarity on the platform's support channels.
- **Concerns over GPT Quality**: One user expressed that the quality of **GPT responses** seems to have declined over the past weeks.
   - This sentiment was echoed by others, suggesting a broader concern within the community.
- **Max API Response Size for GPT Actions**: A member inquired about the maximum size of an **API response** for **GPT Actions**.
   - This question highlights the practical considerations users have when interacting with GPT's API.
- **Discussion on Memory Functionality**: Questions arose regarding the effectiveness of the **memory option** in GPT, with some users reporting issues.
   - This reflects ongoing user experiences and concerns about feature reliability in the latest versions.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1267700222906073138)** (8 messages🔥): 

> - `GPTs training data awareness`
> - `DALL-E bot channel issues`
> - `Custody schedule script problems`
> - `Function call performance in GPT-4` 


- **GPTs lack clarity on training data**: Members expressed frustration over GPT's inability to articulate its training data, highlighting that despite detailed outputs, foundational models seem disconnected from their sources.
   - *It's sort of ridiculous that we're this behind in public gpt's still,* indicating a demand for progress in transparency.
- **DALL-E bot channel functionality issues**: A member reported difficulties in creating images with the DALL-E bot, specifically noting failure to execute the `/draw` command for several hours.
   - They sought help from the community, emphasizing appreciation for any assistance provided.
- **Custody schedule Python script complications**: A member was attempting to create a custody schedule Excel file for August 2024 but faced issues with maintaining a clear five-day segment across month boundaries.
   - Adjusting the initial prompt clarified instructions, ultimately leading to successful script generation.
- **Performance decline in GPT-4 with function calls**: Concerns were raised regarding the performance of GPT-4 when using function calls, with members noticing a drop in the accuracy of responses.
   - One member mentioned that submissions with the full prompt yielded better results compared to using function calls.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1267700222906073138)** (8 messages🔥): 

> - `GPT Training Data Transparency`
> - `DALL-E Bot Functionality Issues`
> - `Custody Schedule Python Script`
> - `Function Call Performance in GPT-4o` 


- **GPTs clueless about training data transparency**: A member questioned why GPTs can access obscure data points but fail to reveal their training data, deeming the situation 'ridiculous'.
   - They suggested making additional question-answer pairs during training to improve clarity on previous data used.
- **DALL-E Bot issues with drawing commands**: A user expressed frustration about being unable to create images in the DALL-E bot channel, maintaining the issue for over 20 minutes.
   - They requested help from the community, emphasizing their appreciation for any support.
- **Adjusting prompts for custody schedule script**: A member struggled to get ChatGPT to produce a Python script for a custody schedule but found that clarifying the prompt improved the outcome.
   - They noted that even Claude made similar mistakes, highlighting a common issue with prompt clarity.
- **Concerns about GPT-4o function call performance**: One member raised concerns that using functions with GPT-4o has led to poorer response quality compared to using regular prompts.
   - They inquired whether others were experiencing similar deteriorating results when utilizing function calls.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1267598596837216306)** (33 messages🔥): 

> - `API Issues`
> - `Cohere Team Acknowledgment`
> - `Project Development with Cohere API`
> - `Office Hours Announcement` 


- **API Down and team is on it**: A member reported that the **API is down**, encountering a **503 error**, and shared a link to the [Cohere status page](https://status.cohere.com/).
   - Another member empathized and mentioned, *'Sorry about that! ❤️ Fixing the issues internally!'*.
- **Celebrating Project Success**: A member excitedly announced they finally built their **dream project** using the **Cohere API** and received enthusiastic responses, including a fire emoji. 
   - They noted that they have features like weather, time, math, and semi-working news, while emphasizing the importance of background vibes for production.
- **Office Hours Are Back!**: A member notified everyone that **office hours** are returning to Discord, and shared an enthusiastic link to the event page. They expressed missing the community on Twitter and invited everyone back.
   - This led to discussions about members often missing the meetings and suggestions on how to ensure attendance.



**Link mentioned**: <a href="https://status.cohere.com/>">incident.io - Status pages</a>: no description found

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1267723231264116817)** (5 messages): 

> - `Cohere Status Page`
> - `Enterprise Workflow Automation Webinar` 


- **Cohere Status Page confirms system operational**: The [Cohere Status Page](https://status.cohere.com/) indicates that the systems are fully operational with **99.67% uptime** for endpoints and **100% uptime** for documentation. No current issues affecting their systems have been reported.
   - Additionally, the page features a notable *subscription option* for updates, enhancing user engagement.
- **Request for Webinar Recording**: A member sought a recording of the **Enterprise Workflow Automation with GenAI** webinar after missing it. *Sssandra* suggested following up with [events@cohere.com](mailto:events@cohere.com) for the fastest access to the recorded session.
   - This indicates a structured approach for attendees who missed live events to still access important content.



**Link mentioned**: <a href="https://status.cohere.com/">Cohere Status Page Status</a>: Latest service status for Cohere Status Page

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1267615990594666559)** (23 messages🔥): 

> - `Cohere API downtime`
> - `Connector response format`
> - `Move towards tool usage` 


- **Cohere API faced temporary downtime**: Reports indicated that the **Cohere API** was experiencing slow response times and failures, prompting users to check the [status page](https://status.cohere.com/) for updates.
   - After some time, it was confirmed that the API is now fully operational.
- **Connector response struggles with integer timestamps**: A user noted that returning a **unix timestamp** as an integer caused the **Cohere chat API** to return no results, whereas using a string representation worked fine.
   - It was clarified that while integers are supported, they are treated as strings within the connector response.
- **Shift from connectors to tool usage**: Discussions highlighted a growing trend towards **tool usage** instead of connectors, underscored by insights from recent office hours.
   - Despite this, it was confirmed that there are no current plans for the deprecation of connectors, as they serve distinct functions alongside tools.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/tool-use">Tool Use with Cohere's Models - Cohere Docs</a>: no description found</li><li><a href="https://status.cohere.com/">Cohere Status Page Status</a>: Latest service status for Cohere Status Page
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1267570341174317056)** (13 messages🔥): 

> - `Cohere API Performance`
> - `Web Search Tool Implementation`
> - `Industry Hype Cycle`
> - `Interview Preparation`
> - `Comparison Testing` 


- **Cohere API touted as top enterprise option**: A member highlighted that the **Cohere API** is the only one they've used without experiencing downtime, calling it possibly the **best enterprise option**.
   - This sentiment was echoed by others, leading to discussions about its reliability compared to competitors.
- **Web search tool sparks creativity**: Members discussed creating a bot using the new **web search tool** in the chat interface, with one sharing quick tests using command-r as a search engine.
   - An offer for collaboration was made, with enthusiasm about comparing results during an upcoming interview.
- **Navigating the industry hype cycle**: Concerns were raised about the current **hype cycle** in the industry, questioning whether recent announcements are sincerely innovative or just iterations of existing models.
   - One member emphasized a commitment to ensuring that models deliver genuine value in an enterprise context.
- **Interview excitement and testing offers**: With excitement, a member humorously mentioned using the new tools during an interview at Cohere, prompting further offers for testing help from others.
   - This light-hearted conversation highlighted the community's supportive atmosphere for non-technical members.
- **Revisiting Cohere's strengths amid competitors**: A member joked about potential backlash from OpenAI for calling Cohere the **best enterprise option**, drawing attention to its strong reputation in the community.
   - The conversation indicated a confident and humorous stance around Cohere's performance against other companies in the field.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1267589651804455003)** (20 messages🔥): 

> - `Mojo Community Meeting #5`
> - `Stack-PR Installation`
> - `GitHub Documentation Issues`
> - `Community Feedback`
> - `Conda Packaging for Stack-PR` 


- **Mojo Community Meeting #5 Recap**: The recording of today's 5th [Mojo Community Meeting](https://youtu.be/1T-MBC9k99M) is available on YouTube, featuring discussions on GPU programming and community Q&A.
   - Participants expressed a desire for more focused discussions on *Mojo* and suggested a live coding session for future meetings.
- **Installing Stack-PR Made Easy**: A new command-line tool **stack-pr** can now be easily installed with `pipx install stack-pr`, allowing the creation of multiple stacked pull requests on GitHub.
   - Members discussed the idea of submitting a feedstock to conda-forge for the stack-pr tool, simplifying installation further.
- **GitHub Documentation Issue Raised**: A member pointed out a GitHub issue about broken links in the Mojo documentation, specifically mentioning [Issue #3308](https://github.com/modularml/mojo/issues/3308).
   - There was a call for clarity on repository ownership since the documentation does not indicate the responsible party for corrections.
- **Constructive Community Feedback**: Participants provided feedback on how community meetings could improve, such as giving speakers more time and ensuring relevance to Mojo topics.
   - The community leader encouraged sharing Discord handles for slots and focusing discussions specifically on Mojo-related topics to enhance future meetings.
- **Guidelines for Future Presentations**: A member proposed having official guidelines for the Mojo community meetings, ensuring presentations remain relevant to the language and tools.
   - Nick suggested a litmus test for future talks: if a Java programmer could benefit from it, the focus might need to be sharpened.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pypi.org/project/stack-pr/">stack-pr</a>: Stacked PR CLI for Github</li><li><a href="https://pypi.org/project/s">s</a>: The almighty S-package</li><li><a href="https://youtu.be/1T-MBC9k99M.">Mojo 🔥 Community Meeting #5</a>: Recording of the Mojo Community Meeting #5🔢 Chris Lattner on GPU programming with Mojo 🔥🔀 Async Mojo 🔥 - 10 Simple Rules❓ Community Q&amp;AFull agenda and de...</li><li><a href="https://www.marcelotrevisani.com/grayskull">Marcelo Duarte Trevisani</a>: no description found</li><li><a href="https://prefix-dev.github.io/rattler-build/latest/converting_from_conda_build/">Converting from conda-build - rattler-build</a>: None</li><li><a href="https://github.com/modularml/mojo/issues/3308">[Docs] Mojo URL leads to 404 · Issue #3308 · modularml/mojo</a>: Where is the problem? https://github.com/modularml/mojo What can we do better? The URL displayed on GitHub for Mojo in the upper right is no longer valid. Please replace this link with something be...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1267565499059929189)** (42 messages🔥): 

> - `Difference between structs and classes in Mojo`
> - `CSV reader capabilities in Mojo`
> - `Performance implications of using structs`
> - `Image parsing libraries in Mojo`
> - `Dynamic behavior in Mojo classes` 


- **Understanding Structs vs Classes in Mojo**: Members discussed the lack of classes in Mojo and the difference with structs, noting that Mojo might share similar semantics with C# and Swift, as highlighted in a [helpful article](https://learn.microsoft.com/en-us/dotnet/standard/design-guidelines/choosing-between-class-and-struct). One contributor noted that structs should not be limited to small immutable types but can be used for optimized performance.
- **Mojo CSV Reader Discussion**: Affable Honey Badger inquired about Mojo's CSV reader, discovering that it already exists but hoped for functionality akin to Python's csv module. Members reiterated that exploring such features would enhance their understanding of Mojo and its struct capabilities.
- **Exploring Image Parsing in Mojo**: A member shared that they have implemented PNG parsing in Mojo, linking to their [GitHub repository](https://github.com/fnands/mimage). They expressed intentions to tackle JPEG parsing next, referring to an outdated existing implementation and suggesting integration with their library.
- **Dynamic Behavior in Mojo Classes**: There was an ongoing debate about the need for dynamic behavior in Mojo classes, akin to reference types in other languages, hinting at requirements for dynamic dispatch and inheritance. Contributors highlighted their hope that Mojo's implementation would avoid the complexities seen in Objective-C interops.
- **The Use of Structs for Performance in Mojo**: Members discussed the implications of using structs for performance, where one member noted it would require fixing member types at compile time in exchange for better performance. This perspective highlighted a preference for struct usage unless dynamic behavior is essential, contrasting against the usage seen in other programming languages.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/classesandstructures/">Documentation</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/dotnet/standard/design-guidelines/choosing-between-class-and-struct">Choosing Between Class and Struct - Framework Design Guidelines</a>: Learn how to decide whether to design a type as a class, or to design a type as a struct. Understand how reference types and value types differ in .NET.</li><li><a href="https://ruhati.net/mojo/_struct.html">Mojo By Example: A Comprehensive Introduction to the Mojo Programming Language</a>: no description found</li><li><a href="https://github.com/fnands/mimage">GitHub - fnands/mimage: A library for parsing images in Mojo</a>: A library for parsing images in Mojo. Contribute to fnands/mimage development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1267607257051562137)** (1 messages): 

> - `LlamaIndex Office Hours`
> - `Building Agents`
> - `In-depth Questions` 


- **LlamaIndex offers Office Hours for Users**: LlamaIndex invites users to sign up for [office hours](https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform) where they can discuss use cases regarding agents and receive LlamaIndex-branded swag as a thank-you.
   - The office hours will consist of a 15-30 minute Zoom conversation to explore how LlamaIndex can help users with agentic applications.
- **In-depth Feedback and Use Cases Welcome**: Participants are encouraged to bring in-depth questions and feedback on their use of LlamaIndex, particularly for those building agents or related applications.
   - For quick questions, users are directed to refer to [Python docs](https://docs.llamaindex.ai/en/stable/), [TypeScript docs](https://ts.llamaindex.ai/), and additional documentation resources.



**Link mentioned**: <a href="https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform">LlamaIndex Community Office Hours</a>: Have in-depth questions or feedback for the folks at LlamaIndex? Sign up for our community office hours! We&#39;ll get back to you to set up a 15-30 minute Zoom call to chat. We are particularly inter...

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1267576758241329245)** (5 messages): 

> - `GraphRAG technique`
> - `Webinar Scheduling`
> - `Agentic Applications Office Hours`
> - `LlamaCloud QA Assistant Feature`
> - `MLflow in LlamaIndex` 


- **GraphRAG Combines Techniques for Text Understanding**: The **GraphRAG** technique from Microsoft integrates text extraction, network analysis, LLM prompting, and summarization into one system. More details can be found [here](https://t.co/ZnDtJ731hl), along with an explanation on its application [here](https://t.co/mx54Su1gYk).
   - *The process involves generating a graph that enriches data comprehension.*
- **Webinar Rescheduled to Next Thursday**: The scheduled webinar is now set for **next Thursday 8/8 at 9am PT.** This update was communicated in a recent message [here](https://t.co/Zo9zRz528F).
   - *Participants should mark their calendars for the new time.*
- **Office Hours for Agentic Applications**: LlamaIndex invites developers of **agents or RAG applications** to participate in office hours featuring a 15-30 minute Zoom conversation to discuss their use cases. Interested individuals can sign up [here](https://t.co/o91QKveTWS) for a chance to receive free swag.
   - *This is a great opportunity to connect and share insights on leveraging LlamaIndex.*
- **LlamaCloud Feature Enhances QA Assistant**: A new feature in **LlamaCloud** allows for dynamic retrieval, providing **chunk-level context** for pointed questions and **document-level context** for summarization. This enhances the capability of building a robust Q&A assistant as discussed [here](https://t.co/5WYIx9TcZG).
   - *The focus on contextual retrieval is crucial for effective question answering.*
- **MLflow Now Available with LlamaIndex**: **MLflow** has integrated features tailored for LlamaIndex, focusing on managing model development, deployment, and management. Key functionalities include tracking prompts and packaging engines with all dependencies detailed [here](https://t.co/BOewMnLklj).
   - *This enhancement aims to streamline the workflow for model development in LlamaIndex.*


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1267561192239661178)** (55 messages🔥🔥): 

> - `Spam Issues`
> - `LlamaIndex Instrumentation`
> - `RAPTOR Pack Update`
> - `Mermaid Diagrams`
> - `Pydantic Models` 


- **Spam Issues in Channels**: Members expressed frustration over repeated spam in every channel, emphasizing the need for better moderation tools.
   - *Discord makes it easy to delete spam messages*, however, members are concerned about the frequency.
- **Challenges with LlamaIndex Instrumentation**: Discussions focused on creating custom spans for instrumentation, detailing how to handle spans using the new `instrumentation` module in LlamaIndex.
   - Members sought clarity on how to effectively track spans and properties, stressing that practical examples would be beneficial.
- **RAPTOR Pack Usage and Updates**: Queries arose about deploying RAPTOR to hosted vector DBs like Pinecone and handling document insertions without re-clustering.
   - Members discussed strategies for managing new document additions to RAPTOR without losing previously clustered data.
- **Generating Mermaid Diagrams**: Members shared experiences and tools for generating Mermaid diagrams from LLM outputs, specifically noting the use of `mmd` format.
   - Tools such as Mermaid CLI were recommended for easy rendering of diagrams, with examples shared for better understanding.
- **Utilizing Pydantic for Structured Data**: The conversation highlighted the advantages of using Pydantic for managing structured data in projects.
   - Members pointed out the ease with which Pydantic can enforce data validation when working with complex data models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mermaid.js.org/intro/syntax-reference.html">Diagram Syntax | Mermaid</a>: no description found</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSefrnmxQWD-1OhSP51kUKtdbw9EGDjrMLefkZFACKD19TKsuQ/viewform">LlamaIndex Community Office Hours</a>: Have in-depth questions or feedback for the folks at LlamaIndex? Sign up for our community office hours! We&#39;ll get back to you to set up a 15-30 minute Zoom call to chat. We are particularly inter...</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation">Instrumentation - LlamaIndex</a>: no description found</li><li><a href="https://github.com/mermaid-js/mermaid-cli">GitHub - mermaid-js/mermaid-cli: Command line tool for the Mermaid library</a>: Command line tool for the Mermaid library. Contribute to mermaid-js/mermaid-cli development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1267570300384448643)** (8 messages🔥): 

> - `Transformers Error`
> - `Q-Galore Status`
> - `Gemma-2-27B Configuration`
> - `Chat Template Training`
> - `` 


- **Transformers throwing errors during indexing**: Several members encountered an assertion error: `srcIndex < srcSelectDimSize` while working with the **Transformers** library, especially in the **Mistral** model.
   - One member suggested **deleting the cache** and redownloading everything as a potential fix.
- **Q-Galore availability inquiry**: A member inquired about the status of **Q-Galore**, questioning whether it is currently available.
   - No responses were provided in the chat regarding its status.
- **Configuration for tuning Gemma-2-27B needed**: A member asked for a working configuration for tuning **gemma-2-27b**, indicating a need for guidance.
   - No specific configurations or solutions were shared in response.
- **New requirement for chat_template training**: Discussion highlighted that **PR #1756** introduces a requirement for a `roles_to_train` field when using `type: chat_template`. This change reportedly breaks existing examples that utilize **chat_template**.
   - Members stressed the need for examples and documentation to clarify this new feature.
- **CUDA Runtime errors related to the Transformers library**: A **RuntimeError** was reported concerning a CUDA device-side assert being triggered, specifically in the **modeling_mistral.py** file.
   - This error seems to arise when the **attention_mask** contains a `0.0`, leading to further complications in model execution.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1267899972913991740)** (43 messages🔥): 

> - `Gemma 2 model fine-tuning issues`
> - `RTX 4090 for chatbot training`
> - `Best practices for fine-tuning models`
> - `Retrieval Augmented Generation (RAG)`
> - `Loss function anomalies in training` 


- **Gemma 2 model outputs pad token**: A user encountered a problem where their fine-tuned **Gemma 2 9b** model outputs the `<pad>` token repeatedly after merging and deploying it to vLLM.
   - Discussion pointed to potential configuration issues and the importance of verifying special tokens from [Hugging Face](https://huggingface.co/google/gemma-2-9b-it/blob/main/special_tokens_map.json).
- **RTX 4090's suitability for chatbot training**: Another user is exploring training a chatbot on an **RTX 4090**, mentioning they had tried **Llama v3.1** and the sharegpt dataset but didn't achieve good results.
   - They expressed interest in using Axolotl for fine-tuning and considered the possibility of acquiring a second RTX 4090 to increase VRAM.
- **Best practices for fine-tuning models**: Advice was shared regarding using good datasets, configuring LoRA for VRAM-efficient training, and considering batch sizes, with suggestions on how to improve training outcomes.
   - The importance of having a solid dataset was reiterated, alongside possible strategies like Retrieval Augmented Generation to enhance the training process.
- **Retrieval Augmented Generation (RAG) for chatbot**: A participant also discussed exploring **Retrieval Augmented Generation (RAG)** as an alternative approach to fine-tuning for their chatbot project.
   - They plans to invest time in both RAG and fine-tuning, hoping to achieve a good fine-tuned patch to strengthen their project.
- **Loss function stuck at zero**: A user reported their training loss being stuck at **0.0** with 'grad_norm' showing as **nan**, indicating a potential issue in their model training process.
   - This loss issue could signal problems with training dynamics or configuration settings that warrant adjustment.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1267739661644857364)** (45 messages🔥): 

> - `Agent Executor and Toolkits Usage`
> - `LangGraph Functionality`
> - `Llama 3.1 Tool Calling`
> - `LangChain Context Caching`
> - `Google Gemini Integration` 


- **Agent Executor Lacks Reflection**: A user expressed concerns that the agent executor does not reveal its planning and thought processes in LangSmith, as this occurs on the model side.
   - Responses indicated that enhanced visibility for decision-making might require additional implementations at the user level.
- **Exploring LangGraph for Planning**: A link to a LangGraph example was shared as a promising starting point for building agentic workflows using graphs.
   - Users discussed the advantages of learning LangGraph for more advanced functionalities beyond basic agent executions.
- **Llama 3.1's Unique Tool Calling Syntax**: Llama 3.1 was noted for its distinct function calling support that utilizes a special prompt syntax rather than a typical parameter setup.
   - Questions arose regarding whether this syntax will be adapted into standard LangChain usage.
- **LangChain's Handling of Context Caching**: Users inquired about the integration of Google Gemini's context caching within LangChain, yet no definitive answers were found.
   - It was noted that LangChain provides support for Gemini models but lacked specific details on context caching features.
- **Best Practices for LLM Decision Making**: Advice was shared on enabling smaller, clearer decision-making tasks for LLMs, emphasizing that such models can introduce unpredictability.
   - Users were encouraged to handcraft logic wherever possible to guide LLM outputs more reliably.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.together.ai/docs/llama-3-function-calling">Function calling with Llama 3.1</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/pull/24570>).">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/migrate_agent/#in-langgraph-1>).">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/streaming_llm/#using-stream>).">How to stream responses from an LLM | 🦜️🔗 Langchain</a>: All LLMs implement the Runnable interface, which comes with default implementations of standard runnable methods (i.e. ainvoke, batch, abatch, stream, astream, astream_events).</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/migrate_agent/#iterating-through-steps>).">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts: -</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/streaming/#chains>).">How to stream | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/plan-and-execute/plan-and-execute.ipynb">langgraph/examples/plan-and-execute/plan-and-execute.ipynb at main · langchain-ai/langgraph</a>: Build resilient language agents as graphs. Contribute to langchain-ai/langgraph development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/)** (1 messages): 

ericyin_41626: https://github.com/langchain-ai/langserve/issues/720
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1267761690129207326)** (2 messages): 

> - `Turing Test Implementation`
> - `SWE Agent Development` 


- **Turing Test Gets a Fun Spin**: A new article explores a playful approach to the **Turing Test** involving three Language Models competing to convince each other they are machines in a game format.
   - *The article discusses whether machines can think* and invites readers to discover this through experimentation.
- **Guide to Building SWE Agents**: A user shared a comprehensive guide on creating **SWE Agents** using frameworks like **CrewAI**, **AutoGen**, **LangChain**, and **LLamaIndex**.
   - The guide emphasizes leveraging a Python framework called **swekit** for easy scaffolding and functionality across agentic frameworks, accessible [here](https://git.new/swe/kit).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/taming-the-llama/turing-the-tables-making-language-models-fight-cfa0bc168878?sk=8ba3dcccf2f3f0a294f991fd5e3c167f">Tur(n)ing the Tables — Making Language Models Fight.</a>: A fun take on a famous experiment, the Turing Test, and with a similar objective — how test how smart (or deceptive) a machine is. The…</li><li><a href="https://git.new/swe/kit">SWE Python Framework - Build SWE Agents </a>: Unleash the power of SWE agents with swekit, a Python framework. Effortlessly build and scaffold agents compatible with agentic frameworks like crewai and llamaindex. Leverage our tooling ecosystem fo...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1267856438127235123)** (2 messages): 

> - `Self Learning Llama3.1`
> - `SWE Agent Guide`
> - `LangChain` 


- **Exploring Self Learning with Llama3.1**: A new YouTube video titled *'I want Llama3.1 to perform 10x with my private knowledge'* discusses building a local self-learning **Llama3.1 agent** in Slack, complete with links to full code and resources.
   - The creator also invites viewers to *get a free HubSpot resource* on adopting AI for work through [this link](https://clickhubspot.com/7hmy).
- **Guide to Building Software Engineering Agents**: A user shared a guide on creating your own **SWE Agent** using LangChain, found at [this GitHub link](https://git.new/swe/kit).
   - This framework, **swekit**, aims to streamline the development of software engineering agents with built-in compatibility for **agentic frameworks** like crewai and llamaindex.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://git.new/swe/kit">SWE Python Framework - Build SWE Agents </a>: Unleash the power of SWE agents with swekit, a Python framework. Effortlessly build and scaffold agents compatible with agentic frameworks like crewai and llamaindex. Leverage our tooling ecosystem fo...</li><li><a href="https://youtu.be/2PKCOVqhngY?si=DUKS8F0QiBdEHj4R">&quot;I want Llama3.1 to perform 10x with my private knowledge&quot; - Self learning Local Llama3.1 405B</a>: Building Local Self Learning Llama3.1 Agent in your SlackGet free HubSpot resource of adopt AI at work: https://clickhubspot.com/7hmy🔗 Links- Get full code ...
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1267567980359843912)** (47 messages🔥): 

> - `Palm Chat 2 surge`
> - `GPT-4o capabilities`
> - `Cost tracking alternatives`
> - `Claude model instruction templates` 


- **Palm Chat 2 experiences a 3000% increase**: A member humorously highlighted that Palm Chat 2's usage surged from 1 request to 30, leading to a **3000% increase**.
   - Another member mentioned that such a sharp rise reminds them of the *WinRAR sales* meme, further adding to the amusement.
- **New GPT-4o allows for extensive outputs**: The experimental version of **GPT-4o** can handle up to **64K output tokens** per request, which is estimated to be around **18.2K words**.
   - It's been noted that the output price is around **$1.15** per **64K reply**, adding a significant cost element for large responses.
- **Searching for LiteLLM alternatives**: A user expressed frustration with LiteLLM's confusing documentation and suggested a potential build for similar services, opting instead for **OpenRouter**.
   - Another user noted that OpenRouter can allow for more control as it gives cost information from their generations endpoint.
- **Challenges with Claude models and instruct templates**: There was a discussion on whether the Claude 3.5 Sonnet model uses an instruct template, with some indicating it does not have one.
   - It was alluded that using the `prompt` mode in OpenRouter might convert prompts into user messages, making it key to properly guide the model.
- **Fireworks model status**: A member confirmed that while Fireworks is operational, the **Yi-Large endpoint** has been removed for unspecified reasons.
   - This sparked discussions about the stability of other models hosted by Fireworks, ensuring that most are still functioning as expected.


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1267576522785423592)** (33 messages🔥): 

> - `SAM 2 Release`
> - `Leonardo AI Joins Canva`
> - `Kagi LLM Benchmarking`
> - `OpenAI and Anthropic Collaboration with Brands`
> - `White House Report on Open-Source AI` 


- **SAM 2 Released with Enhanced Capabilities**: [Meta Segment Anything Model 2 (SAM 2)](https://ai.meta.com/blog/segment-anything-2/) has been released, offering real-time promptable object segmentation in images and videos, significantly improving upon its predecessor with state-of-the-art performance.
   - SAM 2 is trained on a new SA-V dataset with 50,000 videos and employs a novel memory attention technique, making it capable of segmenting varied objects in diverse settings.
- **Leonardo AI Joins Canva's Family**: [Leonardo.Ai](https://x.com/ethan_smith_20/status/1818152222326186260?s=46) announced its acquisition by Canva, which is expected to enhance creative tools at scale for users and empower creators in new ways.
   - The integration aims to speed up innovation and build on the existing success of projects like Phoenix.
- **Kagi Launches New LLM Benchmarking Project**: The [Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html) evaluates large language models on reasoning, coding, and instruction-following capabilities with an unpolluted benchmark for a rigorous assessment.
   - Current benchmark results show **gpt-4o** leading in accuracy and efficiency, demonstrating the need for continuous testing across providers.
- **Strategic Collaboration Opportunities for OpenAI and Anthropic**: There are discussions around **OpenAI** and **Anthropic's** potential to collaborate with brands by providing analytics data based on brand mentions in chat conversations, similar to [Google Analytics](https://link.to/google-analytics).
   - This could be increasingly relevant with new models like SearchGPT to present insights while ensuring that aggregated data remains anonymized.
- **White House Report Advocates for Open-Source AI**: The White House released a report emphasizing the importance of **open-source** AI technology and arguing against the need for immediate restrictions on such models.
   - This stance is seen as a push to support innovation while managing potential risks, highlighting an ongoing debate around open models in AI development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/midjourney/status/1818342703618482265">Tweet from Midjourney (@midjourney)</a>: Midjourney V6.1 is now live! V6.1 greatly improves image quality, coherence, text, and comes with brand-new upscaling and personalization models. It’s smarter, faster, clearer, and more beautiful. We ...</li><li><a href="https://x.com/hturan/status/1818332375358554133?s=46">Tweet from harley turan (@hturan)</a>: hey! we&#39;ve built a new AI playground over at @cloudflare to demonstrate what can be achieved by chaining multi-modal models together. think — audio → text → image → text, or composing multiple LLM...</li><li><a href="https://x.com/alexalbert__/status/1817996841923104908?s=46">Tweet from Alex Albert (@alexalbert__)</a>: I used to have a bookmark folder full of little websites to do things like validate JSON, check the difference between two texts, format markdown, etc.  Recently I&#39;ve replaced all of those bookmar...</li><li><a href="https://x.com/HamelHusain/status/1818040423136510077">Tweet from Hamel Husain (@HamelHusain)</a>: TLDR; You HAVE TO test your LLM providers  - Models are rarely &#34;the same&#34; across different providers.  You can experience materially different results for the same model!  - If one model is be...</li><li><a href="https://apnews.com/article/ai-open-source-white-house-f62009172c46c5003ddd9481aa49f7c3">White House says no need to restrict &#x27;open-source&#x27; artificial intelligence — at least for now</a>: The White House is coming out in favor of “open-source” artificial intelligence technology, arguing in a report Tuesday that there’s no need right now for restrictions on companies making key componen...</li><li><a href="https://x.com/swyx/status/1818074658299855262">Tweet from swyx 🌉 back in SF! (@swyx)</a>: Memory Attention: adding object permanence with $50k in compute  @AIatMeta continues to lead Actually Open AI. SAM2 generalizes SAM1 from image segmentation to video, releasing task, model, and datase...</li><li><a href="https://help.kagi.com/kagi/ai/llm-benchmark.html">Kagi LLM Benchmarking Project | Kagi's Docs</a>: no description found</li><li><a href="https://x.com/ethan_smith_20/status/1818152222326186260?s=46">Tweet from Ethan (@Ethan_smith_20)</a>: I am so happy to announce today http://Leonardo.Ai has joined the Canva family. It’s been one hell of a journey and I don’t think I could have imagined a better team to work alongside. I am absolutely...</li><li><a href="https://x.com/AIatMeta/status/1818055906179105010">Tweet from AI at Meta (@AIatMeta)</a>: Introducing Meta Segment Anything Model 2 (SAM 2) — the first unified model for real-time, promptable object segmentation in images & videos.  SAM 2 is available today under Apache 2.0 so that anyone ...</li><li><a href="https://x.com/nickadobos/status/1818159193037451398?s=46">Tweet from Nick Dobos (@NickADobos)</a>: GPT-4o long output!?  64k output!?!? Now we are cooking. Holy shit let’s go</li><li><a href="https://x.com/drjimfan/status/1818302152982343983?s=46">Tweet from Jim Fan (@DrJimFan)</a>: Exciting updates on Project GR00T! We discover a systematic way to scale up robot data, tackling the most painful pain point in robotics. The idea is simple: human collects demonstration on a real rob...</li><li><a href="https://www.youtube.com/watch?v=y1WnHpedi2A">Do you think that ChatGPT can reason?</a>: Prof. Subbarao Kambhampati argues that while LLMs are impressive and useful tools, especially for creative tasks, they have fundamental limitations in logica...</li><li><a href="https://ai.meta.com/blog/segment-anything-2/">no title found</a>: no description found</li><li><a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization">A Visual Guide to Quantization</a>: Exploring memory-efficient techniques for LLMs</li><li><a href="https://github.com/traceloop/openllmetry">GitHub - traceloop/openllmetry: Open-source observability for your LLM application, based on OpenTelemetry</a>: Open-source observability for your LLM application, based on OpenTelemetry - traceloop/openllmetry</li><li><a href="https://github.com/facebookresearch/segment-anything-2">GitHub - facebookresearch/segment-anything-2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model.</a>: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use th...</li><li><a href="https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1267652535083204608)** (1 messages): 

> - `Apple Intelligence`
> - `macOS updates`
> - `iPhone updates` 


- **Apple Intelligence Beta Launch**: The **Apple Intelligence Beta** is now available on both **macOS** and **iPhone**, providing users with access to new AI functionalities.
   - Updates and discussions are ongoing on [Discord](https://discord.com/channels/822583790773862470/1249801456870101013) as users engage with the latest features.
- **Engagement on Discord for Apple Intelligence**: Users are actively discussing the **Apple Intelligence Beta** and its features on **Discord**, focusing on performance and usability feedback.
   - The channel has seen a variety of comments regarding initial experiences and comparisons with earlier versions.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1267565946533445723)** (14 messages🔥): 

> - `Open Interpreter use cases`
> - `AI for coding`
> - `Speech command technology`
> - `Command line options`
> - `Workflow with Llama 3.1` 


- **Exploring Open Interpreter's Uses**: Members discussed various **use cases** for the Open Interpreter (OI), including a member's need for an on-screen assistant to help with tasks due to health issues, finding the technology promising.
   - *I've been searching for a way to have something essentially learn my on-screen movements over time* and reference back to things I did.
- **AI Takes Over Coding**: A member mentioned their success using AI to generate code, stating that they've won awards in programming without writing any code themselves, showcasing the potential of such tools.
   - They encouraged others to utilize AI in coding for productivity, saying, *trust me, you can do it too friend*.
- **Concerns About AI Automation**: Concerns were raised about **Open Interpreter's** experimental nature and its reliability for critical tasks, as one member advised caution when using voice commands.
   - Another suggested using higher-precision tools like **Whisper** for speech-to-text as a safer alternative.
- **Clarifying Command Line Options**: A member inquired if `--local` and `--os` options in command line usage are redundant and got clarification that `--os` allows control of the computer without prompts.
   - The `--local` option is for local inference, enabling local models to function within the Open Interpreter environment.
- **Using Llama 3.1 with Open Interpreter**: One member sought guidance on workflow when using **Llama 3.1** with Open Interpreter, asking whether to interact in the same terminal session or a new one once OI started.
   - They were running commands without issues and sought clarification on the best practices for questioning the model.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1267572093202206762)** (11 messages🔥): 

> - `Wayland`
> - `Open Interpreter Installation`
> - `Pre-order Status`
> - `Building Parts`
> - `Poetry Version` 


- **Concerns About Wayland Experience**: A member expressed their current lack of fondness for **Wayland**, attributing it to inexperience with the system.
   - This insight highlights a common issue among users transitioning to new display servers.
- **Installation Environment for Open Interpreter**: A member inquired about running **Open Interpreter** (OI) in one virtual environment while using **01** in another, specifically asking about desktop versus device versions.
   - Clarification on installation practices can help new users manage their environments efficiently.
- **Pre-order Availability Questions**: A user questioned the current status of pre-orders and expressed difficulty finding the relevant information on the website.
   - A response clarified that pre-orders are no longer being accepted, and encouraged others to source their own parts to build units.
- **Access Issues with Building Resources**: A user reported encountering a 'no access' message when clicking a shared link related to building resources.
   - The admin advised that granting oneself the **builder role** in the Discord server is necessary for access.
- **Discussion on Poetry Version**: One member sought advice on what version of **Poetry** is currently being utilized by the community.
   - This indicates an ongoing need for resources and compatibility information among developers.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1268004908183982183)** (2 messages): 

> - `Perplexica`
> - `Llama-3.1`
> - `Open source alternatives` 


- **Perplexica: Your New Search Buddy**: A YouTube video titled ["Perplexica + Llama-3.1 (405B, 70B, 8B) : This LOCAL & FREE CLONE of Perplexity BEATS Everyone!"](https://www.youtube.com/watch?v=V0vx94JYNjI) showcases how to set up a local and free alternative to Perplexity and SearchGPT using Meta AI's open-source Llama-3.
   - The video emphasizes the ease of installation and access to these powerful tools in a self-hosted setup.
- **Perplexica on GitHub: An Open Source Solution**: [Perplexica on GitHub](https://github.com/ItzCrazyKns/Perplexica) provides an open-source alternative to Perplexity AI, designed as an AI-powered search engine.
   - Its repository details outline the features and potential uses of Perplexica, making it a valuable resource for developers looking to enhance their search capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=V0vx94JYNjI">Perplexica + Llama-3.1 (405B, 70B, 8B) : This LOCAL &amp; FREE CLONE of Perplexity BEATS Everyone!</a>: In this video, I&#39;ll be telling you that how you can setup a Local &amp; Free Alternative to Perplexity &amp; SearchGPT by using the new Meta AI&#39;s Opensource Llama-3....</li><li><a href="https://github.com/ItzCrazyKns/Perplexica">GitHub - ItzCrazyKns/Perplexica: Perplexica is an AI-powered search engine. It is an Open source alternative to Perplexity AI</a>: Perplexica is an AI-powered search engine. It is an Open source alternative to Perplexity AI - ItzCrazyKns/Perplexica
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1267568219997208669)** (12 messages🔥): 

> - `View Merging Task`
> - `Shape Tracker Reduction`
> - `YouTube Talk on Parallel Computing`
> - `OpenCL Resource Errors` 


- **Clarification on View Merging Task**: The task is to prove that `View.__add__` merges any two mergable views, or if it fails, modify it so it works and provide a proof.
   - There are cases where views aren't pairwise mergable, yet one can still reduce the shape tracker, prompting additional clarification.
- **Identifying Mergable Views**: The bounty setter might express the task as determining and proving when two views are mergable, focusing on clarity in definitions.
   - The goal is to minimize the number of views to reduce final index computation time, ensuring optimal performance.
- **YouTube Talk on Parallel Computing**: A member shared a [YouTube video](https://youtu.be/c52ziyKOArc?si=pAUdzwIQGXCtpk3T) titled 'I want a good parallel computer - UCSC Colloquium', presenting an interesting talk on parallel computing.
   - The recorded talk took place at the UC Santa Cruz CSE Colloquium on April 10, 2024, and slides are also available.
- **Challenges Generating OpenCL Resource Errors**: A member expressed uncertainty about generating an 'out of resources' error with OpenCL on a Mac and reported only receiving 'invalid kernels'.
   - It appears that these errors are likely related to compilation issues rather than runtime resource limits.



**Link mentioned**: <a href="https://youtu.be/c52ziyKOArc?si=pAUdzwIQGXCtpk3T">I want a good parallel computer - UCSC Colloquium</a>: This is the video of a talk I gave at the UC Santa Cruz CSE Colloquium on Apr 10, 2024. The slides are available here: https://docs.google.com/presentation/d...

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1267883182108180582)** (13 messages🔥): 

> - `Gradients in Training Loop`
> - `Using TinyJit`
> - `Jitting the Step Function` 


- **Gradients become None after Jitting**: A member reported that all tensors returned **None** for gradients on the third step of the training loop after using TinyJit, which was fine in the first two steps.
   - *TinyJit kicks in on 3rd step* may lead to this issue, prompting the member to experiment with and ultimately solve the issue by removing it.
- **Potential issues with Jit and Gradients**: There was a discussion about whether jitting could affect gradients, with one member unsure and suspecting a possible skill issue.
   - Another member suggested that perhaps the **optim.step()** was outside the jitted function, which confirmed the problem.
- **Deciding to Jit Whole Step Function**: A member considered whether to jit just the model's forward step or the entire step function.
   - It was advised that unless there's a specific reason not to, it's generally better to jit the entire step function.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1267834416567160923)** (15 messages🔥): 

> - `Apple's AI Model Training`
> - `Tim Dettmers' New Role`
> - `Sewon Kim's Recruitment`
> - `TPUs Usage`
> - `Job Market Insights` 


- **Apple claims no use of NVIDIA GPUs**: Apple stated that it does not use **NVIDIA GPUs** to train its AI models, opting instead for **TPUs** as indicated in a [recent article](https://www.reuters.com/technology/apple-says-it-uses-no-nvidia-gpus-train-its-ai-models-2024-07-29). Members noted that Apple is the **second biggest user of TPUs** in the market.
- **Tim Dettmers joins Allen Institute**: After seven months on the job market, **Tim Dettmers** announced he joined the **Allen Institute** and will become a professor at **Carnegie Mellon University** in Fall 2025. He aims to enhance open-source contributions to address real-world problems while maintaining **bitsandbytes**.
   - Members discussed the competition for Dettmers' talents, highlighting interest from major firms like **Anthropic** and **Hugging Face**.
- **Sewon Kim attracted significant talent interest**: Members celebrated the recruitment of **Sewon Kim**, emphasizing that many companies sought him. The discussion underscored the value of having a **unique and sensible offering** to attract top talent.
- **Insights on the job market**: Tim Dettmers shared that his time on the academic job market involved over **125 interviews** across **17 universities**, resulting in **15 job offers**. He promised to share his insights and learnings from this experience soon.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tim_dettmers/status/1818282779488227575?s=46">Tweet from Tim Dettmers (@Tim_Dettmers)</a>: The six months on the academic job market were brutal but also very successful. More than 125 individual interviews across 17 universities leading to 15 job offers. It was a unique experience for whic...</li><li><a href="https://x.com/Tim_Dettmers/status/1818282778057941042">Tweet from Tim Dettmers (@Tim_Dettmers)</a>: After 7 months on the job market, I am happy to announce: - I joined @allen_ai - Professor at @CarnegieMellon from Fall 2025 - New bitsandbytes maintainer @Titus_vK  My main focus will be to strengthe...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1267609295881769102)** (6 messages): 

> - `Zuck's Stage Performance`
> - `Perplexity Publishers Program`
> - `OpenAI Cookbook Controversy`
> - `Email Open Rates Decline`
> - `iCloud Private Relay Issues` 


- **Zuck Drops F-Bombs with Jensen at SIGGRAPH**: During his appearance at SIGGRAPH, **Zuck** made headlines by dropping f-bombs on stage alongside **Jensen**, showcasing a more candid side.
   - *“Make me another cheesesteak, Jensen,”* was a humorous remark that resonated amidst the serious discussions.
- **Perplexity Launches Publishers Program**: Perplexity announced its **Publishers Program**, aiming to support media organizations with features like **revenue sharing** and technology to boost audience engagement.
   - Notable partners include **TIME** and **Der Spiegel**, with aspirations to enhance the quality of sources utilized in responsive answers.
- **OpenAI Cookbook Under Legal Pressure**: Discussions have arisen around the **OpenAI Cookbook**, with reports of threatened legal action forcing a reconsideration of future partnerships.
   - The potential consequences could significantly impact the project’s direction and the community's engagement.
- **Decline in Email Open Rates Investigated**: There has been a noticeable decline in **email open rates** since **7/24**, prompting an investigation into its causes.
   - The team suggested that this drop is tied to an **Apple iCloud Private Relay** outage rather than a decrease in user engagement, specifically affecting Apple Mail users.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/hub/blog/introducing-the-perplexity-publishers-program">Introducing the Perplexity Publishers’ Program</a>: From day one, we’ve included citations in each answer, ensuring publishers receive proper credit and building user trust.</li><li><a href="https://substack.com/@substackwriters/note/c-63739022?r=68gy5&utm_medium=ios&utm_source=notes-share-action">Substack Writers on Substack</a>: We’ve investigated a trend of dropping email open rates for a subset of publishers since 7/24. We take these concerns seriously and recognize the importance of these metrics as you build your business...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/)** (1 messages): 

xeophon.: https://152334h.github.io/blog/scaling-exponents/
  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1267894523531362314)** (2 messages): 

> - `OPTO and Trace`
> - `AI in Gaming History`
> - `Neural Networks Evolution`
> - `Microsoft's AI Innovations` 


- **Exploring OPTO in Trace's Framework**: A member raised curiosity about [OPTO used by Trace](https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/), highlighting its implications within AI applications.
   - The discussion underscores the growing interest in self-adapting AI technologies, particularly in the gaming industry.
- **Games as AI Innovation Frontiers**: The conversation noted that the gaming industry has historically been a frontier for AI innovation, dating back to early 2000s programming of neural networks for [virtual worlds](https://galciv3.fandom.com/wiki/History_of_the_Galactic_Civilizations_franchise).
   - This evolution has led to the development of engaging AI characters that improve player interactions significantly.
- **Growth of Neural Networks**: The dialogue referenced the advancement of neural networks from simple models to complex systems with **billions of parameters**, now powering applications like [ChatGPT](https://arxiv.org/abs/2303.08774).
   - These developments have transformed the landscape of AI capabilities, enabling more sophisticated real-world applications.
- **Microsoft's Role in AI Advancements**: Mention was made of Microsoft's initiatives, including [Copilots](https://www.bing.com/chat?q=Microsoft+Copilot&FORM=hpcodx), which utilize advanced AI capabilities for enhanced functionality.
   - Their innovations are seen as pivotal in the broader context of AI scaling and application efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/">Discover Trace, a new framework for AI optimization from language models to robot control</a>: Introducing Trace, Microsoft and Stanford University&#039;s novel AI optimization framework, now available as a Python library. Trace adapts dynamically and optimizes a wide range of applications from...</li><li><a href="https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-se">Discover Trace, a new framework for AI optimization from language models to robot control</a>: Introducing Trace, Microsoft and Stanford University&#039;s novel AI optimization framework, now available as a Python library. Trace adapts dynamically and optimizes a wide range of applications from...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1267683115674107935)** (11 messages🔥): 

> - `MIRPO updates`
> - `Penalty metrics in answer scoring`
> - `DSPy with semantic kernel`
> - `Optimizers for celebrity exemplars`
> - `Mean Squared Error in penalties` 


- **MIRPO compatibility with dspy functions**: Members inquired whether **MIRPO** has been updated to support **dspy.Suggest** and **dspy.Assert**, after an earlier issue indicated it had not.
   - There has been no clarification yet on whether this functionality has been implemented.
- **Creating penalty metrics for answer deviations**: There was a discussion on developing a metric that applies a higher penalty based on the distance from the gold answer, emphasizing the idea of proportional penalties.
   - *One member suggested using a formula involving squaring the difference* between the predicted and gold score to achieve this effect.
- **DSPy compatibility with semantic kernel**: **DSPy** users are curious about its interoperability with **semantic kernel**, evaluating potential integrations.
   - No specific updates or confirmations regarding this compatibility have been shared in the chat.
- **Insights on optimizing model penalties**: (in response to scoring metrics) **Mean Squared Error** was recommended as a conventional approach for penalizing larger errors in machine learning.
   - One member detailed how to adjust scores using a negative penalty metric in **DSPy**, explaining how to optimize towards a lower penalty effectively.
- **ICML talk on Language Models**: A member shared insights from an **ICML talk on the 'Physics' of Language Models**, suggesting the potential for optimizers to use 'celebrity' exemplars.
   - The link to the talk was provided [here](https://youtu.be/YSHzKmEianc) for further viewing.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/)** (1 messages): 

batmanosama: https://github.com/ax-llm/ax
  

---



### **AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1267598282008563754)** (1 messages): 

> - `Long Context Use Cases`
> - `Developer Collaboration`
> - `Enterprise Customer Feedback` 


- **Seeking Developers for Long Context Projects**: The team is looking for developers to help with long context use cases utilizing Jamba's **256k effective length**, aiming to improve results based on feedback from enterprise customers.
   - They invite those experimenting with long context to share their insights, offering **credits, swag, and fame** as incentives.
- **Promising Results from Enterprise Clients**: Early feedback from enterprise customers indicates **promising results** as they explore Jamba's capabilities.
   - The message emphasizes a desire to gather more insights and enhance collaborative efforts in this area.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1267633991289147452)** (4 messages): 

> - `Joining Discord`
> - `Learning about Jamba` 


- **New Faces in the Discord**: A new member, **artworxai**, announced their presence, stating they just joined the Discord.
   - *Sorry I went offline before you replied!*
- **Interest in Jamba**: **artworxai** expressed that they joined the Discord to learn about **Jamba**.
   - This highlights an interest among newcomers to explore the functionalities and insights regarding the platform.


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1267620727612379177)** (2 messages): 

> - `SWE-Bench Ultra-Hackathon`
> - `Segment Anything Model 2` 


- **SWE-Bench Ultra-Hackathon Pushes Code Generation Limits**: A bold experiment is underway with a **6-day ultra-hackathon** for **SWE-Bench**, providing $1,000 in compute for every participant from @StrongCompute.
   - Prizes are available for benchmark improvements and for teams that beat existing benchmarks, with talks from coauthors including [John Yang](https://x.com/jyangballin), [Carlos E. Jimenez](https://x.com/_carlosejimenez), and [Ofir Press](https://x.com/OfirPress).
- **GitHub Hosts Segment Anything Model 2 Code Repository**: The [GitHub repository](https://github.com/facebookresearch/segment-anything-2) for **Segment Anything Model 2 (SAM 2)** provides code for running inference along with trained model checkpoints and example notebooks.
   - This resource aims to streamline usage of SAM 2 for various segmentation tasks in open-source coding initiatives.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/stevewattsfrey/status/1818033777622532518">Tweet from Steve Frey (@stevewattsfrey)</a>: A bold experiment: We&#39;re hosting a 6-day ultra-hackathon for SWE-Bench to push the limits of open-source code generation  - Everyone gets $1,000 in compute provided by @StrongCompute  - Up 50 rese...</li><li><a href="https://github.com/facebookresearch/segment-anything-2">GitHub - facebookresearch/segment-anything-2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model.</a>: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use th...
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1267890375628947577)** (1 messages): 

> - `Sentry`
> - `AutoFix feature` 


- **Sentry’s AutoFix Feature Presentation**: Jenn and Ben from **Sentry** are set to discuss their open source feature **AutoFix** in an upcoming session.
   - Event details can be found [here](https://discord.com/events/1089876418936180786/1245836053458190438).
- **Sentry Open Source Benefits**: The discussion will highlight the benefits of using **open source** features like AutoFix for developers.
   - Participants can expect to gain insights on the community-driven support and updates related to these features.


  

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

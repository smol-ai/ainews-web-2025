---
id: 47aa9d67-913f-4412-ae5d-caa6cd205923
title: 'Qdrant''s BM42: "Please don''t trust us"'
date: '2024-07-06T02:25:00.011918Z'
original_slug: ainews-qdrants-bm42
description: >-
  **Qdrant** attempted to replace BM25 and SPLADE with a new method called
  "BM42" combining transformer attention and collection-wide statistics for
  semantic and keyword search, but their evaluation using the Quora dataset was
  flawed. **Nils Reimers** from **Cohere** reran BM42 on better datasets and
  found it underperformed. Qdrant acknowledged the errors but still ran a
  suboptimal BM25 implementation. This highlights the importance of dataset
  choice and evaluation sanity checks in search model claims. Additionally,
  **Stripe** faced criticism for AI/ML model failures causing account and
  payment issues, prompting calls for alternatives. **Anthropic** revealed that
  **Claude 3.5 Sonnet** suppresses some answer parts with backend tags, sparking
  debate. **Gemma 2** model optimizations allow 2x faster fine-tuning with 63%
  less memory and longer context windows, running up to 34B parameters on
  consumer GPUs. **nanoLLaVA-1.5** was announced as a compact 1B parameter
  vision model with significant improvements.
companies:
  - qdrant
  - cohere
  - stripe
  - anthropic
  - hugging-face
  - stablequan_ai
models:
  - claude-3.5-sonnet
  - gemma-2
  - nano-llava-1.5
topics:
  - semantic-search
  - benchmarking
  - dataset-quality
  - model-evaluation
  - model-optimization
  - vision
  - fine-tuning
  - context-windows
people:
  - nils-reimers
  - jeremyphoward
  - hamelhusain
  - rohanpaul_ai
---


<!-- buttondown-editor-mode: plaintext -->**Peer review is all you need.**

> AI News for 7/4/2024-7/5/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**418** channels, and **3772** messages) for you. Estimated reading time saved (at 200wpm): **429 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions + try [Smol Talk](https://smol.fly.dev)!

Qdrant is widely known as [OpenAI's vector database of choice](https://news.ycombinator.com/item?id=38280859), and over the July 4 holiday they kicked off some big claims to replace the venerable [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (and even the more modern [SPLADE](https://arxiv.org/abs/2109.10086)), attempting to coin "BM42":

 ![image.png](https://assets.buttondown.email/images/9c643b9a-85fb-4279-b396-abcc4869e931.png?w=960&fit=max) 

 to solve the problem of semantic + keyword search by combining transformer attention for word importance scoring with collection-wide statistics like IDF, claiming advantages over every use case:

 ![image.png](https://assets.buttondown.email/images/fd7dd267-fa6c-46a5-abe3-1a0d757088c5.png?w=960&fit=max) 

Only one problem... the results. Jo Bergum from Vespa (a competitor), [pointed out](https://x.com/jobergum/status/1809157587612336402) the odd choice of Quora (a "find similar duplicate" questions dataset, not a Q&A retrieval dataset) as dataset and obviously incorrect evals if you know that dataset:

 ![image.png](https://assets.buttondown.email/images/bebb5c7c-192a-41a3-bb91-d54c7118238a.png?w=960&fit=max) 

Specifically, the Quora dataset [only has ~1.6 datapoints per query](https://github.com/beir-cellar/beir) so their precision@10 number was obviously wrong claiming to have >4 per 10.

[Nils Reimers of Cohere](https://x.com/Nils_Reimers/status/1809334134088622217) took BM42 and reran on better datasets for finance, biomedical, and Wikipedia domains, and sadly BM42 came up short on all accounts:

 ![image.png](https://assets.buttondown.email/images/dc98cac6-200f-427f-adfc-19a6c8377c33.png?w=960&fit=max)  

For their part, [Qdrant has responded to and acknowledged](https://x.com/qdrant_engine/status/1809291686625046816) the corrections, and [published corrections](https://x.com/Nils_Reimers/status/1809299249017856379)... except still oddly running a BM25 implementation that scores worse than everyone else expects and conveniently worse than BM42.

Unfortunate for Qdrant, but the rest of us just got a lightning lesson in knowing your data, and sanity checking evals. Lastly, as always in PR and especially in AI, **Extraordinary claims require extraordinary evidence.**


> **Meta note**: If you have always wanted to customize your own version of AI News, we have now [previewed](https://x.com/Smol_AI/status/1809412102693818579) a janky early version of Smol Talk, which you can access here: https://smol.fly.dev

---

{% if medium == 'web' %}

**Table of Contents**

[TOC]

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}

---

# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet.

**Stripe Issues and Alternatives**

- **Stripe account issues**: [@HamelHusain](https://twitter.com/HamelHusain/status/1808850347169100261) noted Stripe is "holding all my money hostage" with "endless wall of red tape" despite no refund requests. [@jeremyphoward](https://twitter.com/jeremyphoward/status/1808948204358283438) called it "disgraceful" that Stripe canceled an account due to an "AI/ML model failure".
- **Appealing Stripe decisions**: [@HamelHusain](https://twitter.com/HamelHusain/status/1808861338917351893) appealed a Stripe rejection but got denied within 5 minutes, with Stripe "holding thousands of dollars hostage". 
- **Alternatives to Stripe**: [@HamelHusain](https://twitter.com/HamelHusain/status/1808906891957055713) noted needing a "backup plan" as "Getting caught by AI/ML false positives sucks." [@virattt](https://twitter.com/virattt/status/1808859416491344040) expressed caution about using Stripe after seeing many posts about issues.

**AI and LLM Developments**

- **Anthropic Constitutional AI**: [@Anthropic](https://twitter.com/Anthropic/status/1808755146190446667) noted Claude 3.5 Sonnet suppresses parts of answers with "antThinking" tags that are removed on the backend, which some disagree with being hidden.
- **Gemma 2 model optimizations**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1808858018253074591) shared Gemma 2 can be finetuned **2x faster with 63% less memory** using the UnslothAI library, allowing **3-5x longer contexts** than HF+FA2. It can go up to 34B on a single consumer GPU.
- **nanoLLaVA-1.5 vision model**: [@stablequan_ai](https://twitter.com/stablequan_ai/status/1809009769195384851) announced nanoLLaVA-1.5, a compact **1B parameter vision model** with significantly improved performance over v1.0. Model and spaces were linked.
- **Reflection as a Service for LLMs**: [@llama_index](https://twitter.com/llama_index/status/1808898730638389262) introduced using reflection as a standalone service for agentic LLM applications to **validate outputs and self-correct** for reliability. Relevant papers were cited.

**AI Art and Perception** 

- **AI vs human art perception poll**: [@bindureddy](https://twitter.com/bindureddy/status/1808946596845097406) posted a poll with 3 AI generated images and 1 human artwork, challenging people to identify the human one, as a "quick experiment" on art perception.
- **AI art as non-plagiarism**: [@bindureddy](https://twitter.com/bindureddy/status/1808804802991903222) argued AI art is not plagiarism as it does the "same thing humans do" in studying work, getting inspired, and creating something new. Exact replicas are plagiarism, but not brand new creations.

**Memes and Humor**

- **Zuckerberg meme video**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1808888487975776520) shared a meme video of Mark Zuckerberg reacting. [@BrivaelLp](https://twitter.com/BrivaelLp/status/1808969132097839362) joked about Zuckerberg's "masterclass" in transforming into a "badass tech guy".
- **Caninecyte definition**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1808871497634549961) jokingly defined a "caninecyte" as a "type of cell characterized by its resemblance to a dog" in a mock dictionary entry.
- **Funny family photos**: [@NerdyRodent](https://twitter.com/NerdyRodent/status/1808809146218918282) humorously asked "Why is it that when I go through old family pictures, someone always has to stick their tongue out?" with an accompanying pixelated artwork.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

AI Progress and Implications

- **Rapid pace of AI breakthroughs**: In /r/singularity, a post highlights how [**compressed recent AI advances are in the grand scheme of human history**](https://www.reddit.com/r/singularity/comments/1dv44b3/how_lucky_are_we_to_witness_this_rare_moment_in/), with modern deep learning emerging in just the last "second" if human existence was a 24-hour day. However, an [article questions the economic impact of AI so far](https://archive.ph/jej1s) despite the hype.
- **AI humor abilities**: [Studies show AI-generated humor being rated as funnier than humans](https://www.psypost.org/ai-outshines-humans-in-humor-study-finds-chatgpt-is-as-funny-as-the-onion/) and on par with The Onion, though some /r/singularity commenters are skeptical the AI jokes are that original. 
- **OpenAI security breach**: The New York Times [reports that in early 2023, a hacker breached OpenAI's communication systems](https://www.nytimes.com/2024/07/04/technology/openai-hack.html) and stole info on AI development, raising concerns they aren't doing enough to prevent IP theft by foreign entities.
- **Anti-aging progress**: In a YouTube interview, the [CSO of Altos Labs discusses seeing major anti-aging effects in mice](https://www.youtube.com/live/Elt4xGalQu4?si=o-fBLMCzT3EEkbCl&t=1860) from cellular reprogramming, with old mice looking young again. Human trials are next.

AI Models and Capabilities

- New open source models discussed include [Kyutai's Moshi audio model](https://www.youtube.com/watch?v=bu7-YODAcfs), the [internlm 2.5 xcomposer vision model](https://huggingface.co/internlm/internlm-xcomposer2d5-7b), and [T5/FLAN-T5 being merged into llama.cpp](https://github.com/ggerganov/llama.cpp/pull/8141).
- An [evaluation of 180+ LLMs on code generation](https://symflower.com/en/company/blog/2024/dev-quality-eval-v0.5.0-deepseek-v2-coder-and-claude-3.5-sonnet-beat-gpt-4o-for-cost-effectiveness-in-code-generation/) found DeepSeek Coder 2 beat LLama 3 on cost-effectiveness, with Claude 3.5 Sonnet equally capable. Only 57% of responses compiled as-is.

AI Safety and Security

- /r/LocalLLaMA [discusses ways to secure LLM apps](https://www.reddit.com/r/LocalLLaMA/comments/1dvoydf/how_do_you_make_your_llm_apps_secure/), including fine-tuning to reject unsafe requests, prompt engineering, safety models, regex filtering, and not rewriting user prompts.
- An example is shared of [Google's Gemini AI repeating debunked information](https://i.redd.it/9r1sj209yiad1.jpeg), showing current AI can't be blindly trusted as factual.

AI Art and Media

- Workflows are shared for [generating AI singers using Stable Diffusion, MimicMotion, and Suno AI](https://www.reddit.com/r/singularity/comments/1dv0w47/ai_singers_made_by_lonely_men_stable_diffusion/), and [using ComfyUI to generate images from a single reference](https://i.redd.it/n544mbmdpmad1.png).
- /r/StableDiffusion discusses a [new open source method for transferring facial expressions](https://www.reddit.com/r/StableDiffusion/comments/1dvmoue/wow_new_opensource_method_of_expression_transfer/) between images/video, and the [emerging role of AI Technical Artists](https://www.reddit.com/r/StableDiffusion/comments/1dv2ygq/technical_ai_artists/) to build AI art pipelines for game studios.
- /r/singularity [predicts a resurgence in demand for live entertainment](https://www.reddit.com/r/singularity/comments/1dvooca/with_the_rise_of_ai_in_entertainment_could_we_see/) as AI displaces online media.

Robotics and Embodied AI

- Videos are shared of [Menteebot navigating an environment](https://v.redd.it/eredb01h2lad1), a [robot roughly picking tomatoes](https://v.redd.it/gxkgv85kggad1), and [Japan developing a giant humanoid robot to maintain railways](https://www.theguardian.com/world/article/2024/jul/04/japan-train-robot-maintain-railway-lines).
- A tweet [calls for development of open source mechs](https://twitter.com/xuxin_cheng/status/1808144850002628658?t=vAbXjEzAfUoa1dZu_aNjSw&s=19).

Miscellaneous

- /r/StableDiffusion expresses [concern about the sudden disappearance](https://www.reddit.com/r/StableDiffusion/comments/1dvrti3/what_the_hell_happened_to_uabdullahalfaraj/) of the Auto-Photoshop-StableDiffusion plugin developer.
- An [extreme horror-themed 16.5B parameter LLaMA model](https://huggingface.co/DavidAU/L3-Stheno-Maid-Blackroot-Grand-HORROR-16B-GGUF) is shared on Hugging Face.
- /r/singularity discusses a ["Singularity Paradox" thought experiment](https://www.reddit.com/r/singularity/comments/1dv2vxh/the_singularity_paradox/) about when to buy a computer if progress doubles daily, with comments noting flaws in the premise.


---

# AI Discord Recap

> A summary of Summaries of Summaries


**1. LLM Performance and Optimization**

- New models like Llama 3, DeepSeek-V2, and Granite-8B-Code-Instruct are showing strong performance on various benchmarks. For example, [Llama 3 has risen to the top of leaderboards like ChatbotArena](https://lmsys.org/blog/2024-05-08-llama3/), outperforming models like GPT-4-Turbo and Claude 3 Opus.
- Optimization techniques are advancing rapidly:
  
  - [ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/) promises 4x reduction in communication overhead for large model training.
  - The [vAttention system](https://arxiv.org/abs/2405.04437) aims to dynamically manage KV-cache memory for efficient LLM inference.
  - [QServe](https://arxiv.org/abs/2405.04532) introduced W4A8KV4 quantization to boost cloud-based LLM serving performance.

**2. Open Source AI Ecosystem**

- Tools like [Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) are supporting diverse dataset formats for LLM training.
- [LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) launched a course on building agentic RAG systems.
- Open-source models like [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) are being released, focusing on specific use cases.

**3. Multimodal AI and Generative Models**

- New multimodal models are enhancing various capabilities:
  
  - [Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609) focuses on improved chat interactions.
  - [CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677) refines coding abilities.
  - [Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/) brings AI chatbots to browsers via WebGPU.
- Combinations of models (e.g., Pixart Sigma + SDXL + PAG) are aiming to achieve DALLE-3 level outputs.

**4. Stability AI Licensing**

- [Stability AI revised the license for SD3 Medium](https://stability.ai/news/license-update) after community feedback, aiming to provide more clarity for individual creators and small businesses.
- Discussions about AI model licensing terms and their impact on open source development are ongoing across multiple communities.
- Stability AI's launch of **Stable Artisan**, a Discord bot integrating various Stable Diffusion models for media generation and editing, was a hot topic ([Stable Artisan Announcement](https://bit.ly/4aiVy6C)). Users discussed the implications of the bot, including questions about **SD3's open-source status** and the introduction of **Artisan as a paid API service**.

**5. Community Tools and Platforms**

- Stability AI launched [Stable Artisan](https://bit.ly/4aiVy6C), a Discord bot integrating models like Stable Diffusion 3 and Stable Video Diffusion for media generation within Discord.
- [Nomic AI announced GPT4All 3.0](https://home.nomic.ai/gpt4all), an open-source local LLM desktop app, emphasizing privacy and supporting multiple models and operating systems.

**6\. New LLM Releases and Benchmarking Discussions**:

- Several AI communities discussed the release of new language models, such as **Meta's Llama 3**, **IBM's Granite-8B-Code-Instruct**, and **DeepSeek-V2**, with a focus on their performance on various benchmarks and leaderboards. ([Llama 3 Blog Post](https://lmsys.org/blog/2024-05-08-llama3/), [Granite-8B-Code-Instruct on Hugging Face](https://huggingface.co/ibm-granite/granite-8b-code-instruct), [DeepSeek-V2 on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2))
- Some users expressed skepticism about the validity of certain benchmarks, calling for more credible sources to set realistic standards for LLM assessment.

**7\. Optimizing LLM Training and Inference**:

- Across multiple Discords, users shared techniques and frameworks for optimizing LLM training and inference, such as **Microsoft's ZeRO++** for reducing communication overhead ([ZeRO++ Tutorial](https://www.deepspeed.ai/tutorials/zeropp/)), **vAttention** for dynamic KV-cache memory management ([vAttention Paper](https://arxiv.org/abs/2405.04437)), and **QServe** for quantization-based performance improvements ([QServe Paper](https://arxiv.org/abs/2405.04532)).
- Other optimization approaches like **Consistency LLMs** for parallel token decoding were also discussed ([Consistency LLMs Blog Post](https://hao-ai-lab.github.io/blogs/cllm/)).

**8\. Advancements in Open-Source AI Frameworks and Datasets**:

- Open-source AI frameworks and datasets were a common topic across the Discords. Projects like **Axolotl** ([Axolotl Dataset Formats](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)), **LlamaIndex** ([Building Agentic RAG with LlamaIndex Course](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)), and **RefuelLLM-2** ([RefuelLLM-2 on Hugging Face](https://huggingface.co/refuelai/Llama-3-Refueled)) were highlighted for their contributions to the AI community.
- The **Modular** framework was also discussed for its potential in Python integration and AI extensions ([Modular Blog Post](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo)).

**9\. Multimodal AI and Generative Models**:

- Conversations surrounding multimodal AI and generative models were prevalent, with mentions of models like **Idefics2 8B Chatty** ([Idefics2 8B Chatty Tweet](https://twitter.com/sanhestpasmoi/status/1787503160757485609)), **CodeGemma 1.1 7B** ([CodeGemma 1.1 7B Tweet](https://twitter.com/reach_vb/status/1786469104678760677)), and **Phi 3** ([Phi 3 Reddit Post](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)) for various applications such as chat interactions, coding, and browser-based AI.
- Generative modeling techniques like combining **Pixart Sigma, SDXL, and PAG** for high-quality outputs and the open-source **IC-Light** project for image relighting were also discussed ([IC-Light GitHub Repo](https://github.com/lllyasviel/IC-Light)).


**10\. New Model Releases and Training Tips in Unsloth AI Community**:

- The Unsloth AI community was abuzz with discussions about new model releases like **IBM's Granite-8B-Code-Instruct** ([Granite-8B-Code-Instruct on Hugging Face](https://huggingface.co/ibm-granite/granite-8b-code-instruct)) and **RefuelAI's RefuelLLM-2** ([RefuelLLM-2 on Hugging Face](https://huggingface.co/refuelai/Llama-3-Refueled)). Users shared their experiences with these models, including challenges with **Windows compatibility** and skepticism over certain **performance benchmarks**. The community also exchanged valuable tips and insights on model training and fine-tuning.

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Vietnamese Linguistics Voiced**: Vi-VLM's Vision\*\*: The **Vi-VLM team** announced a **Vision-Language model** tailored for Vietnamese, integrating [Vistral and LLaVA](https://huggingface.co/Vi-VLM/Vistral-V-7B) frameworks to focus on image descriptions. Viewers can find the demo and supporting code in the linked repository.
  - **Dataset Availability**: Vi-VLM released a dataset specific for VLM training in Vietnamese, which is accessible for enhancing local language model applications. The dataset adds to the linguistic resources available for Southeast Asian languages.
- **Grappling with Graphics**: WHAM's Alternative Search\*\*: An enthusiast sought alternatives to **WHAM** for **human pose estimation** in complex videos, pointing out the ungainly Python and CV dependencies. The community exchange hints at a need for tools that accommodate non-technical users in sophisticated AI applications.
  - Learning resources for **ViT and U-Net** implementations were shared, including a guide from [Zero to Mastery](https://www.learnpytorch.io/08_pytorch_paper_replicating/) and courses by Andrew Ng, indicating community interest in mastering these vision transformer models.
- **Tuning In**: Audio-Language Model Discourse\*\*: **Moshi's linguistic fluidity**: Yann LeCun shared a [tweet](https://x.com/ylecun/status/1808573888642617406) spotlighting **Kyutai.org's digital pirate** that can comprehend English spoken with a French accent, showcasing the model's diverse auditory processing capabilities.
  - Interest in the **Flora** paper and audio-language models remains strong, reflecting the AI community's focus on cross-modal faculties. Upcoming paper reading sessions on these topics are anticipated with enthusiasm.
- **Frozen in Thought**: The Mistral Model Stalemate\*\*: Users reported a crawling halt in the **Mistral model's** inference process, notably at iteration 1800 out of 3000, suggesting possible caching complications. This reflects on the pragmatic challenge of managing resources during extensive computational tasks.
  - Conversations surfaced around making effective API calls without downloading models locally, highlighting the need for streamlined remote inference protocols. The API-centric dialogue underscores a trend towards more flexible, cloud-based ML operations.
- **Diffusion Discussion**: RealVisXL and ISR\*\*: **RealVisXL V4.0**, optimized for rendering photorealistic visuals, is now in training, with an [official page](https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning) and sponsorship on Boosty, spotlighting community support for model development.
  - The existing IDM-VTON's 'no file named diffusion_pytorch_model.bin' error in Google Colab exemplifies the troubleshooting dialogs that emerge within the diffusion model space, emphasizing the practical sides of AI deployment.

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Clearing the Confusion: Stability AI’s Community License**: Stability AI has revised their **SD3 Medium** release license after feedback, offering a new **Stability AI Community License** that clarifies usage for individual creators and small businesses. Details are available in their [recent announcement](http://stability.ai/news/license-update), with the company **striking a balance** between commercial rights and community support.
  - Users can now freely use **Stability AI's models for non-commercial purposes** under the new license, providing an **open source boon** to the community and prompting discussions about how these changes could impact model development and accessibility.
- **Anime AI Model's Metamorphosis: Animagine XL 3.1**: The **Animagine XL 3.1** model by [Cagliostro Research Lab](https://huggingface.co/cagliostrolab) and [SeaArt.ai](https://www.seaart.ai/) is driving conversations with its enhancements over predecessor models, bringing higher quality and broader range of anime imagery to the forefront.
  - The **AAM Anime Mix XL** has also captured attention, sparking a flurry of comparisons with **Animagine XL 3.1**, as enthusiasts discuss their experiences and preferences between the different anime-focused generation models.
- **Debating the GPU Arms Race: Multi-GPU Configurations**: The technical community is actively discussing the optimization of **multiple GPU setups** to boost **Stable Diffusion's performance**, with emphasis on tools like **SwarmUI** that cater to these complex configurations.
  - The debates converge on the challenges of efficiently managing resources and achieving high-quality outputs, highlighting the combination of **technical prowess and creativity** required to navigate the evolving landscape of AI model training.
- **CivitAI's SD3 Stance Spurring Controversy**: CivitAI's move to ban **SD3 models** has divided opinion within the community, as some view it as a potential roadblock for the development of the **Stable Diffusion 3** framework.
  - The discussions deepen with insights into licensing intricacies, commercial implications, and the overall trajectory of how this decision could shape future collaborations and model evolutions.
- **License and Limits: Stable Diffusion Under Scrutiny**: The latest conversations scrutinize the **license for Stable Diffusion 3** and its compatibility with both individual and enterprise usage, considering the community's need for clarity and freedom in AI model experimentation.
  - Community sentiment is split, as discussions pivot around whether the perceived license restrictions unfairly penalize smaller projects or whether they're an inherent part of maturing technologies in the field of AI.

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma's Quantum Leap**: The new **Gemma 2** has hit the tech scene, boasting **2x faster finetuning** and a lean VRAM footprint, requiring **63% less VRAM** ([Gemma 2 Blog](https://unsloth.ai/blog/gemma2)). Support for hefty **9.7K token contexts** on the 27B model was a particular highlight among Unsloth users.
  - The marred launch with **notebook issues** such as mislabeling was glossed over by a community member's remark on the rushed blogpost, but those issues have been swiftly tackled by developers ([Notebook Fixes](https://github.com/unslothai/unsloth/pull/67)).
- **Datasets Galore at Replete-AI**: Replete-AI has introduced two extensive datasets, **Everything_Instruct** and its multilingual cousin, each packing 11-12GB of instruct data ([Replete AI Datasets](https://huggingface.co/datasets/Replete-AI/Everything_Instruct)). Over 6 million rows are at AI developers' disposal, promising to fuel the next wave of language model training.
  - The community's enthusiasm was tempered with quality checks, probing the datasets for **deduplication** and **content balance**, a nod to the seasoned eye for meticulous dataset crafting.
- **Notebooks Nailed to the Mast**: Requests in **collaboration** channels have led to a commitment for **pinning versatile notebooks**, assisting members to swiftly home in on valuable resources.
  - Continued efforts were seen with **correcting notebook links** and the promise to integrate them into the Unsloth **GitHub page**, showcasing a dynamic community-driven documentation process ([GitHub Unsloth](https://github.com/unslothai/unsloth)).
- **Patch and Progress with Unsloth 2024.7**: Unsloth's patch 2024.7 got mixed reception due to **checkpoint-related errors**, yet it marks an important stride by integrating **Gemma 2 support** into Unsloth's ever-growing toolkit ([2024.7 Update](https://github.com/unslothai/unsloth)).
  - Devoted users and Unsloth's responsive devs are on top of the **fine-tuning foibles and error resolutions**, evidencing a robust feedback loop essential for fine-grained model optimization.
- **Facebook's Controversial Token Tactics**: Facebook's **multi-token prediction** model stirred debate over access barriers, stirring a whirlwind of opinions among Unsloth's tight-knit community.
  - Critical views on data privacy were par for the course, specifically relating to the need for sharing contact data to utilize Facebook's model, fueling an ongoing conversation on ethical AI usage ([Facebook's Multi-Token Model](https://huggingface.co/facebook/multi-token-prediction)).

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Sprinting on Funding Runway**: Following a link to [rakis](https://github.com/hrishioa/rakis?tab=readme-ov-file), community members discussed the whopping $85M seed investment intersecting AI with blockchain, sparking conversations on the current venture capital trends in technology.
  - The developers of BM42 faced heat for potentially skewed benchmarks, leading to a vigilant community advocating for rigorous evaluation practices; this prompted a [revised approach to their metrics and datasets](https://x.com/qdrant_engine/status/1809291686625046816).
- **Collision Course: Coding Tools**: Users compared git merge tool experiences, singling out [lazygit](https://github.com/jesseduffield/lazygit) and [Sublime Merge](https://www.sublimemerge.com/), driving the conversation towards the need for more nuanced tools for code conflict resolution.
  - Claude 3.5 and other AI-based tools grabbed the spotlight in discussions for their prowess in coding assistance, emphasizing efficiency in code completion and capabilities like handling complex multi-file refactors.
- **Tuning into Technical Talk**: On the **Latent Space Podcast**, Yi Tay from Reka illuminated the process of developing a training stack for frontier models while drawing size and strategy parallels with teams from OpenAI and Google Gemini.
  - Listeners were invited to engage on [Hacker News](https://news.ycombinator.com/item?id=40886218) with the live discussion, bridging the gap between the podcast and broader AI research community dialogues.
- **Navigating AV Troubles**: OpenAI's AV experienced disruptions during the AIEWF demo, with voices for a switch to Zoom ensuing, followed by a swift action resulting in sharing a Zoom [meeting link](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09) for better AV stability.
  - Compatibility issues between Discord and Linux persisted as a recurrent technical headache, prompting users to explore more Linux-friendly communication alternatives.
- **Deconstructing Model Merger Mania**: Debates on **model merging tactics** took center stage with participants mulling the differing objectives and potential integrative strategies for tools like **LlamaFile and Ollama**.
  - The conversation dived into the possibilities of wearable technology integration with AI for enhancing event experiences, paired with a deep consideration for privacy and consent.

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Snapdragon's Surprising Speed**: The **Surface Laptop with Snapdragon Elite** showcased heft, hitting **1.5 seconds to first token** and **10 tokens per second on LLaMA3 8b with 8bit precision**, whilst only using 10% GPU. No **NPU** activity yet, but the laptop's speed stirred speculation on eventual NPU boosts to LLaMA models.
  - Tech enthusiasts compared **Snapdragon's CPU prowess** to older Intel counterparts, finding the former's velocity vivacious. Amidst laughter, the tech tribe teases about a makeshift *Cardboard NPU*, projecting potential performance peaks pending proper NPU programming.
- **Quantization Quirks and Code Quests**: Quantization quandaries arose with **Gemma-2-27b**, where model benchmarks behaved bizarrely across different quantized versions. Meanwhile, tailored system prompts polished performance for **Gemma 2 27B**, prompting **PEP 8**\-adhering and efficient algorithm emission.
  - Suggestions surfaced that **Qwen2 models** trot best with **ChatML** and a **flash attention setting**, while users with non-CUDA contraptions cautioned against the chaos of **IQ quantization**, noting notably nicer behavior on alternative architectures.
- **LM Studio's ARM Standoff**: A vexed user voiced frustration when **LM Studio's AppImage** defied a dance with **aarch64 CPUs**. The error light shone, signaling a syntax struggle, and a lamenting line confirmed, "*No ARM CPU support on Linux.*"
  - Dialogues dashed hopes for immediate **ARM CPU** inclusions, leaving Linux loyalists longing. A shared sibling sentiment suggested an **architecture adjustment** for **LM Studio** belongs on the horizon but hasn't hit home base just yet.
- **RTX's Rocky Road**: **RTX 4060 8GB VRAM** owners opined their predicament with **20B quantized models**; a tenacious tussle with tokens terminated in total system freezes. Fellow forum members felt for them, flashing back to their own fragmentary **RTX 4060** experiences.
  - Guild guidance gave GPU grievances a glimmer of hope, heralding less loaded models like **Mistral 7B** and **Open Hermes 2.5** for mid-tier machine mates. A commendatory chorus rose for smaller souls, steering clear of titanic token-takers.
- **ROCm's Rescue Role**: Seeking solace from stifled sessions, users with **7800XT** aired their afflictions as models muddled up, missing the mark on GPU offload. A script signalled success, soothing overtaxed systems seeking **ROCm solace**.
  - The cerebral collective converged on solutions, corroborating the effectiveness of the **ROCm installation script**. Joyous jingles jived in the forum, as confirmation came that GPGPU gurus had gathered a workaround worthy of the wired world.

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Conundrums & Mixed-precision MatMul**: Discussions in the CUDA MODE guild veered into optimizing matrix multiplication using CUDA, highlighting a [blog post](https://siboehm.com/articles/22/CUDA-MMM) on techniques for column loading in GPU matrix multiplication; another thread featured the release of customized **gemv kernels** for int2\*int8 and the BitBLAS library for mixed-precision operations.
  - Users explored **TorchDynamo**'s role in PyTorch performance, and compared ergonomics of Python vs C++ for **CUDA kernel** development, with Python favored for its agility in initial phases. Some faced challenges adapting to Python 3.12 bytecode changes with `torch.compile`, addressed in a recent [discussion](https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-12-completed/2054).
- **GPTs Crafting Executive Summaries & Model Training Trials**: A [blog post](https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/) detailing the use of GPTs for executive summary drafting sparked interest, while LLM training trials with FP8 gradients were flagged for increasing losses, prompting a switch to BF16 for certain operations.
  - Schedule-Free Optimizers boasted smoother loss curves, with empirical evidence of convergence benefits [shared by users](https://x.com/_clashluke/status/1808590060654108910), meanwhile, a backend SDE's transition to CUDA inference optimization was deliberated with suggestions spanning online resources, course recommendations, and community involvement.
- **AI Podcasts & Keynotes Spark Engaging Discussions**: **Lightning AI**'s [Thunder Sessions podcast](https://x.com/LightningAI/status/1808610408481370205) with Luca Antiga and Thomas Viehmann caught the attention of community members, whereas **Andrej Karpathy**'s [keynote at UC Berkeley](https://www.youtube.com/watch?v=tsTeEkzO9xc) was a highlighter of innovation and student talent.
  - Casual conversations and channel engagement painted a picture of an interactive forum, with members sharing brief notes of excitement or appreciation, yet holding back on deeper technical exchanges in channels tagged as less focused.
- **Deep Learning Frameworks & Triton Kernel Fixes**: The quest to build a deep learning framework from scratch in C++, akin to **tinygrad**, uncovered the complexity hurdle, kindling a debate on the affordances of C++ vs Python in this context, while Triton kernel's `tl.load` issues in parallel CUDA graph instances required ingenuity to circumnavigate latency concerns.
  - Further intricacies surfaced when discussing the functioning of the `.to` method in **torch.ao**, where current limitations restrict dtype and memory format changes, prompting temporary function amendments as discussed in issue trackers and [commit logs](https://github.com/pytorch/ao/blob/a8956992191853b13f82ceb3e6929bed7691a3fa/torchao/dtypes/affine_quantized_tensor.py#L262).

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Llamas Looping Lines**: Repetition Glitch in AI\*\*: Users experienced **Perplexity AI** outputting repetitive responses across models such as **Llama 3** and **Claude**, and were reassured that the issue was being addressed with an imminent fix.
  - **Alex** confirmed the issue's recognition and the ongoing efforts to rectify it, marking a pressing concern within the Perplexity AI's performance benchmark.
- **Real-Time Reality Check Fails**: Live Access Hiccups\*\*: A gap in expectations has emerged as **Perplexity AI** users face live internet data retrieval issues, receiving obsolete rather than up-to-date information.
  - Despite attempts to resolve the inaccuracies by restarting the application, the users indicated the problem persistence in the feedback channel.
- **Math Model Missteps**: Perplexity Pro's Calculation Challenges\*\*: **Perplexity Pro's** computations, such as CAPM beta, were highlighted for inaccuracies despite its GPT-4o origins, casting shadows on its reliable academic application.
  - The community expressed its dissatisfaction and concerns regarding the model's utility in fields requiring exact mathematical problem solving.
- **Stock Market Success Stories**: Perplexity’s Profitable Predictions\*\*: Anecdotes of financial victories like making $8,000 surfaced among users who harnessed **Perplexity AI** for stock market decisions, triggering conversations on its varied benefits.
  - Such user stories serve as testimonials to the diverse capabilities of the **Pro** version of Perplexity AI in real-world use cases.
- **Subscription Scrutiny**: Decoding Perplexity AI Plans\*\*: Questions and comparisons flourished as users delved into the differences between **Pro and Enterprise Pro** plans, particularly concerning model allocations like **Sonnet and Opus**.
  - Enquiries were directed at understanding not just availability but also the specificity of models included in Perplexity’s varied subscription offerings.

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **BUD-E Board Expansion**: [BUD-E now reads clipboard text](https://youtu.be/WMcEzVfEXpM), a new feature shown in a YouTube video with details on [GitHub](https://github.com/christophschuhmann/Desktop_BUD-E/tree/main). The feature demo, presented in low quality, sparked light-hearted comments.
  - The community discussed **AI model training** challenges due to recurrent usage of overlapping datasets, with **FAL.AI's** dataset access hurdles highlighting the issue. Contrastingly, breakthroughs like **Chameleon** are linked to a variety of integrated data.
- **Clipdrop Censorship Confusion**: Clipdrop's NSFW detection misfired, mislabeling a benign image as inappropriate, much to the amusement of the community.
  - [Stability AI revises license](https://stability.ai/news/license-update) for **SD3 Medium**, now under the Stability AI Community License, allowing increased access for individual creators and small businesses after community feedback.
- **T-FREE Trend Setter**: The new **T-FREE tokenizer**, detailed in a [recently released paper](https://arxiv.org/abs/2406.19223), promises sparse activations over character triplets, negating the need for large reference corpora and potentially reducing embedding layer parameters by over **85%**.
  - The approach is praised for enhancing performance on less common languages and slimming embedding timers, adding a compact edge to LLMs.
- **Alert: Scammer in the Guild**: A scammer was flagged in the #[research] channel, putting the community on high alert.
  - A string of identical phishing links offering a '$50 gift card' was posted across multiple channels by a user, raising concerns.

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Voices in the Void**: The unveiling of a new **Moshi AI demo** sparked a mix of excitement for its real-time **voice interaction** and disappointment over issues with interruptions and looped responses.
  - **Hume AI's playground** was scrutinized for its lack of long-term memory, frustrating users who seek **persistent AI conversations**.
- **Memory Banks in Question**: GPT's memory prowess came under fire as it saves user preferences but still fabricates responses, with members suggesting **enhanced customization** to mitigate this.
  - A heated **GPT-2 versus modern models** debate surfaced, comparing the **cost-efficiency** of older models with the performance leaps in current iterations like GPT-3.5 Turbo.
- **ChatGPT: Free vs. Plus Plans**: Advantages of the **paid ChatGPT Plus** plan were clarified, detailing perks such as a higher usage cap, **DALL·E** access, and an expanded context window.
  - **GPT-4 usage** concerns were addressed, with cooldown periods in place after limit hits, specifically allowing Plus members up to 40 messages every 3 hours.
- **AI Toolbox Expansion**: Community members explored tools for **testing multiple AI responses** to prompts, suggesting a **custom-built tool** and existing options for efficient assessment.
  - Conversation turned to **API integrations**, looking at Rigorous Aggregate Generators (RAG) for linking AI models to diverse datasets and utilizing existing **Assistant API endpoints**.
- **Contest with Context**: In **#prompt-engineering**, strategies for contesting **traffic tickets** were delineated, advising structured approaches and legal argumentation techniques.
  - Discussions blossomed over creating an **employee recognition program** to heighten workplace morale, focusing on goals and recognition criteria for notable contributions.

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Datasets Deluge by Replete-AI**: **Replete-AI** dropped two gargantuan datasets, titled **Everything_Instruct** and **Everything_Instruct_Multilingual**, boasting 11-12GB and over 6 million data stripes. Intent is to amalgamate variegated instruct data to advance AI training.
  - The **Everything_Instruct** targets English, while **Everything_Instruct_Multilingual** brings in a linguistic mix to broaden language handling of AI. Both sets echo past successes like bagel datasets and take a cue from EveryoneLLM AI models. [Dive in at Hugging Face](https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual).
- **Nomic AI Drops GPT4All 3.0**: The latest by **Nomic AI**, **GPT4All 3.0**, hits the scene as an open-source, local LLM desktop app catering to a plethora of models and prioritizing privacy. The app is noted for its redesigned user interface and is licensed under MIT. [Explore its features](https://home.nomic.ai/gpt4all).
  - Touting more than a quarter-million monthly active users, **GPT4All 3.0** facilitates private, local interactions with LLMs, cutting internet dependencies. Uptake has been robust, signaling a shift towards localized and private AI tool usage.
- **InternLM-XComposer-2.5 Raises the Bar**: **InternLM** introduced **InternLM-XComposer-2.5**, a juggernaut in large-vision language models that brilliantly juggles 24K interleaved image-text contexts and scales up to 96K via RoPE extrapolation.
  - This model is a frontrunner with top-tier results on 16 benchmarks, closing in on behemoths like GPT-4V and Gemini Pro. Brewed with a sprinkle of innovation and a dash of competitive spirit, [this InternLM concoction awaits](https://x.com/_akhaliq/status/1808747694317261114?s=46).
- **Claude 3.5's Conundrum and Lockdown**: Attempts to bypass the ethical constraints in **Claude 3.5 Sonnet** led to frustration among users, with strategies around specific pre-prompts making little to no dent.
  - Despite the resilience of Claude's restrictions, suggestions to experiment with **Anthropic's workbench** were shared. Yet, users were cautioned about the risks of account restrictions following such endeavors. [Peer into the conversation](https://console.anthropic.com/workbench).
- **Apollo's Artistic AI Ascent**: **Achyut Benz** bestowed the Apollo project upon the world, an AI that crafts visuals akin to the admired **3Blue1Brown** animations. Built atop **Next.js**, it taps into **GroqInc** and interweaves both **AnthropicAI 3.5 Sonnet** & **GPT-4**.
  - Apollo is all about augmenting the learning experience with AI-generated content, much to the enjoyment of the technophile educator. [Watch Apollo's reveal](https://x.com/achyut_benz/status/1808969030969274507?s=46).

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Quantum Leap in LLM Deployment**: OpenRouter's deployment strategy for LLM models specifies **FP16/BF16** as the default quantization standard, with exceptions noted by an associated **quantization icon**.
  - The adaptation of this quantization approach has sparked detailed discussions on the technical implications and efficiency gains.
- **API Apocalypse Averted by OpenRouter**: A sudden change in **Microsoft's API** could have spelled disaster for OpenRouter users, but a swift patch brought things back in line, earning applause from the community.
  - The fix restored harmony, reflecting OpenRouter’s readiness for quick turnarounds in the face of technical disruptions.
- **Infermatic Instills Privacy Confidence**: In an affirmative update, **Infermatic declared its commitment** to real-time data processing with its new [privacy policy](https://infermatic.ai/privacy-policy/), explicitly stating it won’t retain input prompts or model outputs.
  - This update brought clarity and a sense of security to users, distancing the platform from previous data retention concerns.
- **DeepSeek Decodes Equation Enigma**: Users troubleshooting issues with **DeepSeek Coder** found a workaround for equations not rendering by ingeniously using regex to tweak output strings.
  - Persistent problems with TypingMind's frontend not correctly processing prompts were flagged for a fix, demonstrating proactive community engagement.
- **Pricey API Piques Peers**: Debate heated up around **Mistral's Codestral API** pricing strategy, with the 22B model considered overpriced by some community members.
  - Users steered each other towards more budget-friendly alternatives like **DeepSeek Coder**, which offers competitive coding capabilities without breaking the bank.

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Fingerprints of the Digital Minds**: The community explored **Topological Data Analysis (TDA)** for unique model fingerprinting and debated the utility of checksum-equivalent metrics for model validation, such as for the `LlamaForCausalLM` using tools like [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
  - Discussions also touched on **Topological Data Analysis** to profile model weights by their **invariants**, referencing resources like [TorchTDA](https://giotto-ai.github.io/gtda-docs/0.5.1/library.html) and considering bit-level innovations from papers like [1.58-bit LLMs](https://arxiv.org/abs/2402.17764) for efficiency.
- **Tales of Scaling and Optimization**: Attention was given to the [efficientcube.ipynb](https://github.com/kyo-takano/chinchilla/blob/master/examples/efficientcube.ipynb) notebook for scaling laws, while **AOT compilation capabilities** in JAX were highlighted as a step forward in [pre-execution code](https://jax.readthedocs.io/en/latest/aot.html#debug-information-and-analyses-when-available) optimization.
  - **FLOPs estimation methods** for JIT-ed functions in Flax were shared, and critical batch sizes were reinvestigated, challenging the assumption that performance is unaffected below a certain threshold.
- **Sparse Encoders and Residual Revelations**: The deployment of **Sparse Autoencoders (SAEs)** trained on Llama 3 8B's residual stream discussed utilities for integrating with LLMs for better processing, furnished with details on the [model's implementation](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x).
  - Looking into residual stream processing, the strategy organized SAEs by layer for optimizing their synergy with Llama 3 8B, as expanded upon in the associated [model card](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x).
- **Harnessing the Horsepower of Parallel Evaluation**: Enthusiast surfaced questions on the viability of caching preprocessed inputs and resolving **Proof-Pile Config Errors**, noting that changing to `lambada_openai` circumvented the issue.
  - Notables included model name length issues, prompting **OSError(36, 'File name too long')**, and guidance was sought on setting up parallel model evaluation, with warnings about single-process evaluation assumptions.

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Lamentations**: LangChain users reported **performance issues** when running on CPU, with long response times and convoluted processing steps being a significant pain point.
  - The debate is ongoing whether the sluggishness is due to inefficient model reasoning or the absence of GPU acceleration, while some suggest it's bogged down by unnecessary complexity, as discussed [here](https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents).
- **AI Model Showdown: OpenAI vs ChatOpenAI**: Discussions ensued over the advantages of using **OpenAI** over **ChatOpenAI** as the former might be phased out, sparking a comparison of their implementation efficiencies.
  - Members shared mixed experiences around task-specific requirements, while some preferred OpenAI for its familiar interface and tooling.
- **Juicebox.ai: The People Search Prodigy**: **Juicebox.ai**'s **PeopleGPT** was praised for its Boolean-free natural language search capabilities to swiftly identify qualified talent, enhancing the talent search with ease-of-use features.
  - The technical community lauded its combination of filtering and natural language search, elevating the overall experience for users; details are available [here](https://juicebox.ai/).
- **RAG Chatbot Calendar Conundrum**: A **LangChain RAG-based chatbot** developer sought guidance for integrating a **demo scheduling function**, highlighting the complexities found in the implementation process.
  - Community response was geared towards assisting with this integration, indicating a cooperative effort to enhance the chatbot's capabilities despite the absence of explicit links to resources.
- **Visual Vectored Virtuosity**: [A blogpost](https://www.lightly.ai/post/vector-indexes-and-image-retrieval-using-lightly) outlined creating an **E2E Image Retrieval app** using **Lightly SSL** and **FAISS**, complete with a vision transformer model.
  - The post, accompanied by [Colab Notebook](https://colab.research.google.com/drive/1n4CwX5T6Ch2v7OYTRe6g1j_QJHxxOvcM) and [Gradio app](https://huggingface.co/spaces/lightly-ai/food101-image-retrieval), was shared to encourage peer learning and application.

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex RAG-tastic Webinar Whirl**: **LlamaIndex** partnered with **Weights & Biases** for a webinar demystifying the complexities involved in **RAG experimentation and evaluation**. The session promises insights into accurate LLM Judge alignment, with a spotlight on Weights and Biases collaboration.
  - Anticipation builds as the **RAG pipeline** serves as a focal point for the upcoming webinar, highlighting challenges in the space. A hint of skepticism over RAG's nuanced evaluation underscores community buzz around the event.
- **Rockstars of AI Edging Forward**: Rising star **@ravithejads** shares his ascent in becoming a **rockstar AI engineer and educator**, fueling aspirations within the **LlamaIndex** community.
  - **LlamaIndex** illuminates @ravithejads's contribution to OSS and consistent engagement with AI trends, igniting discussions about pathways for professional development in AI.
- **Reflecting on 'Reflection as a Service'**: 'Reflection as a Service' enters the limelight at **LlamaIndex**, proposing an introspective mechanism to boost LLM reliability by adding a self-corrective layer.
  - This innovative approach captivated the community, sparking dialogue on its potential to enhance agentic applications through intelligent **self-correction**.
- **Cloud Function Challenges Versus Collaborative Fixes**: Discussions surfaced on the **Google Cloud Function** regarding hardships with **multiple model loading**, sparking a collective search for more efficient methods among AI enthusiasts.
  - Community wisdom circulates as members share their strategies for reducing load times and optimizing model use, showcasing a collaborative spirit in problem-solving.
- **CRAG – Corrective Measures on Stage**: **Yan et al.** introduce **Corrective RAG (CRAG)**, an innovative LlamaIndex service designed to dynamically validate and correct irrelevant context during retrieval, stirring interest among AI practitioners.
  - Connections are drawn between **CRAG** and possibilities for advancing retrieval-augmented generation systems, fueling forward-thinking conversations on refinement and accuracy.

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Open Invites to AI Soirees**: Community members clarified that **no special qualifications** are necessary to attend the **London AI event**; simply filling out a form will suffice. The inclusive policy ensures that events are accessible to all, fostering a diverse exchange of ideas.
  - Discussion around event attendance highlighted the importance of **community engagement** and **open access** in AI gatherings, as these policies promote broader participation and knowledge sharing across fields and expertise levels.
- **API Woes in Production Mode**: A TypeError issue was raised by a member deploying an app using Cohere's **rerank API** in production, sparking a troubleshooting thread in contrast with its smooth local operation.
  - The community’s collaborative effort in addressing the rerank API problems showcased the value of shared knowledge and immediate peer support in overcoming technical challenges in a production environment.
- **Fresh Faces in AI Development**: Newly joined members of diverse backgrounds, including a **Computer Science graduate** and an **AI developer** focused on teaching, introduced themselves, expressing eagerness to contribute to the guild's collective expertise.
  - The warm welcome extended to newcomers underlines the guild's commitment to nurturing a vibrant community of AI enthusiasts poised for collaborative growth and learning.
- **Command R+ Steals the Limelight**: Cohere announced their most potent model in the Command R family, **Command R+**, now ready for use, creating quite the buzz among the tech-savvy audience.
  - The release of Command R+ is seen as a significant step in advancing the capabilities and applications of AI models, indicating a continuous drive towards innovation in the field.
- **Saving Scripts with Rhea.run**: The introduction of a 'Save to Project' feature in **Rhea.run** was met with enthusiasm as it allows users to create and preserve interactive applications through conversational HTML scripting.
  - This new feature emphasizes Rhea.run’s dedication to simplifying the app creation process, thereby empowering developers to build and experiment with ease.

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **MacOS Copilot Sneaks into Focus**: The [Invisibility MacOS Copilot](https://x.com/sulaimanghori/status/1791113392482377833) featuring GPT-4, Gemini 1.5 Pro, and Claude-3 Opus was highlighted for its context absorption capabilities and is currently available for free.
  - Community members showed interest in potentially open-sourcing grav.ai to incorporate similar functionalities into the **Open Interpreter (OI) ecosystem**.
- **'wtf' Command Adds Debugging Charm to OI**: The 'wtf' command allows **Open Interpreter** to intelligently switch [VSC themes](https://discord.com/invite/YQhmv5pd?event=1258399216078684242) and provide terminal debugging suggestions, sparking community excitement.
  - Amazement over the command's ability to execute actions intuitively was voiced, with plans to share further updates on security roundtables and the upcoming **OI House Party event**.
- **Shipping Woes for O1 Light Enthusiasts**: Anticipation and frustration were the tones within the community regarding the **01 Light shipments**, as discussions revolved around delays.
  - Echoed sentiments of waiting reinforced the collective desire for clear communication on shipment timelines.

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Objects Go Haywire!**: Members discussed a casting bug affecting **Mojo objects** compared to **Python objects**, potentially linked to [GitHub Issue #328](https://github.com/modularml/mojo/issues/328).
  - A debate ensued on whether the **casting bug** might be correlated with differences in object handling, as outlined in issues [#3065](https://github.com/modularml/mojo/issues/3065) and [#3167](https://github.com/modularml/mojo/issues/3167).
- **MLIR's Unsigned Integer Drama**: The community discovered that **MLIR** interpreted unsigned integers as signed, sparking discussion and leading to the creation of [GitHub Issue #3065](https://github.com/modularml/mojo/issues/3065).
  - Concern surged around how this unsigned integer casting issue could impact various users, pivoting the conversation to this emerging bug.
- **Compiler Nightly News: Segfaults and Solutions**: Recent **segfaults** in the nightly build led to the submission of a bug report and sharing the problematic file, seen [here](https://github.com/Mojo-Numerics-and-Algorithms-group/MojoSci/blob/dynobs/src/diffeq/runga_kutta.mojo).
  - Added to this, new compiler releases were announced, with improvements including an `exclusive` parameter and new methods in version `2024.7.505`, linked in the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **Marathon March: Mojo's Matrix Multiplication**: Benny impressed by sharing a matrix multiplication technique and recommended tailoring block sizes, advising peers to consult UT Austin papers for insights.
  - In a separate discussion thread, speed bumps occurred with increased compilation times and segfaults in the latest test suite, with participants directing each other to resources such as a [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1TBz9Lp0JT1Ph7ndfbWqp-B30FQcRYl1959hP2lZ6yH4/edit) for papers.

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Solo Smithing Without Chains**: Discussion confirmed **LangSmith** can operate independently of **LangChain** as demonstrated in examples on [Colab](https://colab.research.google.com/github/langchain-ai/langsmith-cookbook/blob/main/tracing-examples/traceable/tracing_without_langchain.ipynb) and [GitHub](https://github.com/langchain-ai/langsmith-cookbook/blob/main/tracing-examples/traceable/tracing_without_langchain.ipynb). **LangSmith** allows for **instrumentation of LLMs**, offering insights into application behaviors.
  - Community members assuaged concerns about GPU credits during an AI course, emphasizing proper communication of terms and directing to clear info on the [course platform](https://example.com/course-terms).
- **Credit Clarity & Monthly Challenges**: A hot topic revolves around the **$1000 monthly credit** and its perishability, with consensus on no rollover but still appreciating the offer.
  - A user's doubt about a mysteriously increased balance of **$1030** post-Mistral finetuning led to speculation on a possible **$30 default credit** per month.
- **Training Tweaks: Toiling with Tokens**: A thread on the `Meta-Llama-3-8B-Instruct` setup using `type: input_output` sparked some confusion, with users examining special tokens and model configurations, referencing [GitHub](https://github.com/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/special_tokens_map.json).
  - Trainers experienced better outcomes favoring **L3 70B Instruct** over **L3 8B**, serendipitously found when a configuration defaulted to the instruct model, highlighting model choice implications.
- **Credit Confusion & Course Catch-up**: Uncertainty loomed about credit eligibility for services, with one member seeking clarification on terms post-enrollment since **June 14th**.
  - Another user echoed concerns about compute credit expiration, requesting an extension for the remaining credit which slipped through the calendar cracks.

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Debunking the Demo Dilemma**: Community member challenged the legitimacy of an AI demo, calling into question the realism of its responses and highlighting significant **response time** problems. The thread included a link to the contentious [demonstration](https://x.com/benhylak/status/1808611023123067357).
  - In an apologetic pivot, **Stability AI** made revisions to the **Stable Diffusion 3 Medium** in response to community feedback, along with clarifications on their [license](https://x.com/stabilityai/status/1809274908641489160?s=46), earmarking a path for future **high-quality Generative AI** endeavors.
- **Search Smackdown: BM42 vs. BM25**: The **Qdrant Engine** touted its BM42 as a breakthrough in search technology, promising superior RAG integration over the long-established **BM25**, as seen in their [announcement](https://x.com/qdrant_engine/status/1808498752107241949).
  - Critics, including **Jo Bergum**, questioned the integrity of BM42's reported success, suggesting the improbability of the claims and sparking debate on the validity of the findings presented on the dataset from Quora.
- **VAEs Vexation and AI Investment Acumen**: A humorous account of the difficulties in grasping **Variational Autoencoders** surfaced, juxtaposed against a claim of exceptional AI **investment strategy** within the community.
  - A serious projection deduced that for AI to bolster **GDP growth** effectively, it must range between **11-15%**, while the community continues to grapple with **Anthropic Claude 3.5 Sonnet's** opaque operations.
- **Google's Grinder in the Global AI Gauntlet**: Users discussed **Google's** sluggish start in the Generative AI segment, expressing concerns over the company's messaging clarity and direction regarding its products like **Gemini web app**.
  - Discourse evolved around the pricing model and effectiveness of **Google’s Gemini 1.5**, with comparisons to other AI offerings and software like **Vertex AI**, amid reflections on the **First Amendment's** application to AI.

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **API Queue System Quirks**: Reports of issues with the **build.nvidia API** led to discovery of a new **queue system** to manage requests, signaling a potentially overloaded service.
  - A member encountered **script issues with build.nvidia API**, observing restored functionality after temporary downtime hinting at service intermittency.
- **YAML Yields Pipeline Progress**: A member shared their **pipeline**'s integration of **YAML examples** for few-shot learning conversation models, sparking interest for its application with textbook data.
  - Further clarifications were provided on how the YAML-based structure contributed to efficient **few-shot learning** processes within the pipeline.
- **Gemma2 Garners Stability**: **Gemma2** update brought solutions to past bugs. A reinforcement of version control with a **pinned version of transformers** ensures smoother future updates.
  - **Continuous Integration (CI)** tools were lauded for their role in preemptively catching issues, promoting a robust environment against development woes.
- **A Call for More VRAM**: A succinct but telling message from 'le_mess' underlined the perennial need within the group: a request for more **VRAM**.
  - The single-line plea reflects the ongoing demand for higher performance computing resources among practitioners, without further elaboration in the conversation.

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tensor Trouble in Tinygrad**: Discussions arise about **Tensor.randn** and **Tensor.randint** creating contiguous **Tensors**, while `Tensor.full` leads to non-contiguous structures, prompting an examination of methods that differ from [PyTorch's expectations](https://pytorch.org/docs/stable/tensors.html).
  - A community member queried about placement for a bug test in **tinygrad**, debating between **test_nn** or **test_ops** modules, with the final decision leaning towards an efficient and well-named test within **test_ops**.
- **Training Pains and Gains**: Tinygrad users signal concerns about the framework's **large-scale training efficiency**, calling it sluggish and economically impractical, while considering employing BEAM search despite its complexity and time demands.
  - A conversation sparks around the use of pre-trained **PyTorch models** in Tinygrad, directing users towards `tinygrad.nn.state.torch_load` for effective model inference operations.
- **Matmul Masterclass**: A blog post showcasing a guide to high-performance matrix multiplication achieves over 1 TFLOPS on CPU, shared within the community, detailing the practical implementation approach and [source code](https://github.com/salykova/matmul.c).
  - The share included a [link to the blog post](https://salykova.github.io/matmul-cpu) that breaks down matrix multiplication into an accessible 150 line C program, inviting discussion on performance optimization in **Tinygrad**.

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune's Tuning Talk**: Community members exchanged insights on setting evaluation parameters for **Torchtune**, with mentions of a potential 'validation dataset' parameter to tune performance.
  - Others raised concerns about missing **wandb logging** metrics, specifically for **evaluation loss** and **grad norm** statistics, highlighting a need for more robust metric tracking.
- **Wandb Woes and Wins**: A topic of discussion was **wandb's** visualization capabilities, where a **grad norm graph** miss sparked questions about its availability compared to tools like aoxotl.
  - Suggestions included adjusting the initial **learning rate** to affect the **loss curve**, but despite optimizations, one member noted no significant loss improvements, emphasizing the challenges of parameter fine-tuning.

---

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Code Clash: Python meets TypeScript**: A challenging encounter was shared regarding the integration of **Python** with **TypeScript** while setting up the **Convex** platform. Issues surfaced when **Convex** experienced launch bugs stemming from a lack of pre-installed Python.
  - Furthermore, discussion revolved around the difficulties faced in automating the installation of the **Convex** local backend within a **Docker** environment, emphasizing the complication arose from the specific configuration of container folders as volumes.
- **Pixel Hunt: In Search of the Perfect Sprite**: A member explored the domain of **sprite sheets**, expressing their goal to find visuals resonant with the **Cloudpunk** game's style, but found their assortment from **itch.io** lacking the desired cyberpunk nuance.
  - They are on the lookout for sprite resources that align better with **Cloudpunk's** distinctive aesthetic, as previous acquisitions fell short in mirroring the game's signature atmosphere.

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Summarizing with a GPT Trio**: [Three GPTs Walk into a Bar and Write an Exec Summary](https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary) blog post showcases a dynamic trio of **Custom GPTs** designed to extract insights, draft, and revise executive summaries swiftly.
  - This toolkit enables the producing of succinct and relevant executive summaries under tight deadlines, streamlining the process for delivering condensed yet **impactful briefs**.
- **Magpie's Maiden Flight on HuggingFace**: The Magpie model makes its debut on [HuggingFace Spaces](https://huggingface.co/spaces/sroecker/Elster), offering a tool for generating preference data, albeit with a duplication from [davanstrien/magpie](https://huggingface.co/spaces/davanstrien/magpie).
  - **User experiences** reveal room for improvement, as feedback indicates that the model’s performance isn't fully satisfactory, yet the community remains **optimistic** about its potential applications.

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Build With Claude Campaigns Ahead**: Engineering enthusiasts are called to action for the [Claude hackathon](https://docs.anthropic.com/en/build-with-claude-contest/overview), a creative coding sprint winding down next week.
  - Participants aim to craft innovative solutions, employing **Claude's capabilities** for a chance to shine in the closing contest.
- **Kafka's Cost-Cutting Conclave**: A webinar set for **July 18th at 4 PM IST** promises insights into [optimizing Kafka](https://www.meetup.com/futureofdata-bangalore/events/301849238/?notificationId=1389017441959817216) for better performance and reduced expenses.
  - **Yaniv Ben Hemo** and **Viktor Somogyi-Vass** will steer discussions, focusing on scaling strategies and efficiency in Kafka setups.

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Jovial Jest at Job Jargon**: Conversations have sprouted around the growing **potential uses for embeddings** in the field, sparking some playful banter about job titles.
  - One participant quipped about renaming themselves an *Embeddings AyEngineer*\*, lending a humorous twist to the evolving nomenclature in AI.
- **Title Tattle Turns Trendy**: The rise in embedding-specific roles leads to a light-hearted suggestion of the title **Embeddings Engineer**.
  - This humorous proposition underscores the significance of **embeddings** in current engineering work and the community's creative spirit.

---

The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1258521435433861150)** (1 messages):

> - `VLM training dataset in Vietnamese`
> - `Highlights parser`
> - `See 2 sound demo`
> - `text2cypher model`
> - `Guide to Designing New Functional Proteins`

- **Vietnamese VLM Dataset Released**: **VLM training dataset** in Vietnamese released by [user](https://huggingface.co/datasets/Vi-VLM/Vista). The dataset is now available for community use.
- **Highlights Parser Tool**: **Highlights parser tool** created by [user](https://huggingface.co/spaces/rrg92/hf-community-highlights-parser) is now available. It helps users parse community highlights effectively.
- **See 2 Sound Demo**: Check out the **See 2 sound demo** based on the newly released paper available on this [space](https://huggingface.co/spaces/rishitdagli/see-2-sound). It provides an innovative way to experience sound.
- **Text2Cypher Model Outperforms GPT-4**: The new [text2cypher model](https://huggingface.co/lakkeo/stable-cypher-instruct-3b) by user outperforms **GPT-4**. This model represents a significant advancement in text-to-cypher translation.
- **Guide to Designing Functional Proteins**: **Guide to Designing New Functional Proteins** and improving them with Generative AI now available [here](https://huggingface.co/blog/AmelieSchreiber/protein-optimization-and-design). This guide covers protein function, stability, and diversity.

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1258139701903097867)** (495 messages🔥🔥🔥):

> - `Use of Deepeval with HuggingFace Transformers`
> - `Proficiency certifications in ML`
> - `Uploading image on HuggingFace projects using Gradio API`
> - `GPU recommendations for ML beginners`
> - `Issues with renting A100 vs. 4090 GPUs for inference`

- **Proficiency certifications in ML**: Members discussed various certifications to validate ML skills, preferring free options from platforms like Harvard and Coursera.
- **GPU recommendations for ML beginners**: Users debated between recommending RTX 3060 or 4060, considering VRAM and performance, with suggestions leaning towards 3060 for its 12GB VRAM.
- **Issues with renting A100 vs. 4090 GPUs for inference**: A discussion revolved around renting GPU configurations for efficient ML model inference, with suggestions pointing towards H100 over multiple 4090s for better performance.
- **Creating video with AI models**: The chat explored text-to-video generation AI models like the ipivs-morph-img2vid-animatediff-lcm-hyper-sd, noting that processing on standard devices is slow but feasible.
- **Stable Diffusion model licensing update**: **Stability AI revised the license** for SD3 Medium to better support the open-source community, addressing previous issues with commercial use restrictions.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://swtokyo.com/">Startup Weekend Tokyo</a>: no description found</li><li><a href="https://www.runpod.io/serverless-gpu">Serverless GPU Endpoints for AI Inference</a>: Run machine learning inference at scale with RunPod Serverless GPU endpoints.</li><li><a href="https://huggingface.co/blog/alvdansen/training-lora-m3lt">How I train a LoRA: m3lt style training overview</a>: no description found</li><li><a href="https://lumalabs.ai/dream-machine">Luma Dream Machine</a>: Dream Machine is an AI model that makes high quality, realistic videos fast from text and images from Luma AI</li><li><a href="https://tenor.com/view/happyfourthofjuly-july4th-gif-22215151">Happyfourthofjuly July4th GIF - Happyfourthofjuly July4th - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.instagram.com/p/C8luO4VM3l1/">ERLAX on Instagram: "…</a><p><a href="https://www.instagram.com/p/C8luO4VM3l1/">#techno #dreamcore #rave #digitalart #aiart #stablediffusion"</a>: 2,738 likes, 151 comments - erlax.case on June 24, 2024: "… #techno #dreamcore #rave #digitalart #aiart #stablediffusion".</p></li><li><a href="https://huggingface.co/artificialguybr/doodle-redmond-doodle-hand-drawing-style-lora-for-sd-xl">artificialguybr/doodle-redmond-doodle-hand-drawing-style-lora-for-sd-xl · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/13v3b6q/multiple_cheap_gpus_or_a_single_expensive_one/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=r5PM6vOQPISl">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/spaces/InstantX/InstantStyle">InstantStyle - a Hugging Face Space by InstantX</a>: no description found</li><li><a href="https://x.com/p4ino/status/1808560882189803931">Tweet from lilbotomy☆ (@p4ino)</a>: this is how i tell stories</li><li><a href="https://github.com/nroggendorff/diffusion/blob/main/zelda.ipynb">diffusion/zelda.ipynb at main · nroggendorff/diffusion</a>: Contribute to nroggendorff/diffusion development by creating an account on GitHub.</li><li><a href="https://huggingface.co/internlm/internlm2_5-7b-chat-1m">internlm/internlm2_5-7b-chat-1m · Hugging Face</a>: no description found</li><li><a href="https://x.com/p4">Tweet from undefined</a>: no description found</li><li><a href="https://tenor.com/view/i-just-work-here-idk-idk-about-that-i-don%27t-know-gif-3168423486813006711">I Just Work Here Idk GIF - I just work here Idk Idk about that - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://stability.ai/news/license-update">Community License — Stability AI</a>: Our new Community License is now free for research, non-commercial, and commercial use. You only need a paid Enterprise license if your yearly revenues exceed USD$1M and you use Stability AI models in...</li><li><a href="https://huggingface.co/spaces/aheedsajid/Edge-TTS/discussions/1#6685a353d8e85b570562e2c6">aheedsajid/Edge-TTS · 🚩 Report: Spam</a>: no description found</li><li><a href="https://tenor.com/view/happy-tree-friends-htf-cuddles-giggles-flaky-gif-5696779679679953568">Happy Tree Friends Htf GIF - Happy tree friends Htf Cuddles - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/Nick088/Stable_Diffusion_Finetuned_Minecraft_Skin_Generator/discussions/4#667fd0173a46eeac17c80179">Nick088/Stable_Diffusion_Finetuned_Minecraft_Skin_Generator · 🚩 Report: Spam</a>: no description found</li><li><a href="https://huggingface.co/spaces/Nick088/SDXL-Flash/discussions/1#667fd1079f4f7654f000f465">Nick088/SDXL-Flash · Need a btter version</a>: no description found</li><li><a href="https://www.reddit.com/r/Stable">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1dq4y7r/how_are_videos_like_these_created/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">Three GPTs Walk into a Bar and Write an Exec Summary – D-Squared</a>: no description found<p></p></li></ul></div>

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1258349133480071208)** (2 messages):

> - `Building a TikTok videos dataset for harmful content classification`
> - `Troubleshooting LDM implementation with RGB images`

- **TikTok Dataset to Classify Harmful Content**: A user shared a TikTok videos dataset, **30 GB** with around **3,000 videos**, to build a video classification model for classifying harmful content for children. They also provided a [notebook](https://www.kaggle.com/code/anhoangvo/how-to-use-hugging-face-for-fine-tuning-on-the-tik) for fine-tuning a Hugging Face model on this dataset.
- **LDM Model Troubleshooting**: A user is learning to create LDMs from scratch with **Flax library**, succeeding with the MNIST dataset but facing issues with RGB images from **imagenette/160px-v2**. They requested tips for troubleshooting as their model only generates color blocks for RGB images.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.kaggle.com/datasets/anhoangvo/tikharm-dataset/">TikHarm Dataset</a>: A dataset of TikTok videos for training models to classify harmful content.</li><li><a href="https://www.kaggle.com/code/anhoangvo/how-to-use-hugging-face-for-fine-tuning-on-the-tik">How to Use Hugging Face for Fine-Tuning on the Tik</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from TikHarm Dataset</li></ul></div>

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1258225348802248796)** (6 messages):

> - `Kyutai.org's digital pirate understands English with a French accent`
> - `Small demo of Moshi, an audio language model`
> - `Graph Structure Learning (GSL) with GraphEdit and large language models`
> - `Claude's ease in building Deep Learning Visualizer dashboards`
> - `nanoLLaVA - cool VLM under 1B`

- **Kyutai's digital pirate gets language savvy**: A tweet from [Yann LeCun](https://x.com/ylecun/status/1808573888642617406) reveals that **Kyutai.org's digital pirate** can understand English with a **French accent**. This was demonstrated in a **small demo** by Neil Zegh from the **Moshi** project.
- **GraphEdit pushes the boundaries of GSL**: The paper [GraphEdit](https://arxiv.org/abs/2402.15183) proposes a new approach to **Graph Structure Learning** using **Large Language Models (LLMs)** for **enhanced reliability** by instruction-tuning over graph structures.
- **nanoLLaVA attains attention**: The Hugging Face space [nanoLLaVA](https://huggingface.co/spaces/qnguyen3/nanoLLaVA) is highlighted as a **cool Visual Language Model (VLM)** under **1 billion parameters**. It has been noted for its **impressive visualization capabilities**.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/ylecun/status/1808573888642617406">Tweet from Yann LeCun (@ylecun)</a>: Where we learn that http://Kyutai.org's digital pirate understands English with a French accent Quoting Guillaume Grallet (@guillaumgrallet) A small demo by ⁦@neilzegh⁩ from #moshi, an audio la...</li><li><a href="https://huggingface.co/spaces/qnguyen3/nanoLLaVA">nanoLLaVA-1.5 - a Hugging Face Space by qnguyen3</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.15183">GraphEdit: Large Language Models for Graph Structure Learning</a>: Graph Structure Learning (GSL) focuses on capturing intrinsic dependencies and interactions among nodes in graph-structured data by generating novel graph structures. Graph Neural Networks (GNNs) have...</li></ul></div>

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1258155599452049538)** (32 messages🔥):

> - `Introduction of Vision-Language model for Vietnamese by Vi-VLM team`
> - `Vi-VLM releasing a dataset for VLM training in Vietnamese`
> - `Simple translation tool for converting messages to pt-br`
> - `CyclicFormer architecture enhancement for transformers`
> - `UVR5's UI completion for audio separation`

- **Vi-VLM introduces Vision-Language model for Vietnamese**: The Vi-VLM team introduced a Vision-Language model for Vietnamese, built on LLaVA and Vistral, with an image description focus; demo and code [available here](https://huggingface.co/Vi-VLM/Vistral-V-7B).
- **CyclicFormer enhances transformer performance**: The CyclicFormer architecture introduces a cyclic loop between decoder layers to enhance transformer performance, [GitHub link here](https://github.com/LegallyCoder/CyclicFormer).
- **E2E Image Retrieval app using Lightly SSL**: An image retrieval app was built using an arbitrary image dataset from the Hub, leveraging FAISS for vector indexing and Lightly SSL for self-supervised learning, detailed in a [blogpost](https://www.lightly.ai/post/vector-indexes-and-image-retrieval-using-lightly).
  - *Check out the [Gradio app](https://huggingface.co/spaces/lightly-ai/food101-image-retrieval) for a practical demonstration.*
- **UVR5 UI for audio separation completed**: UVR5's UI is now complete, allowing easy separation of vocals and instrumental tracks; it uses advanced audio separation models available via [Gradio](https://huggingface.co/spaces/TheStinger/UVR5_UI).
  - *Perfect separation of voice and melody in various tests, including popular songs like 'Faroeste Caboclo' from 1987.*
- **Simple translation tool for pt-br**: A tool was created to translate community highlights into pt-br, useful for faster importing of messages; see the [tool here](https://huggingface.co/spaces/rrg92/hf-community-highlights-parser).

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://huggingface.co/rishitdagli/see-2-sound">rishitdagli/see-2-sound · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/rrg92/hf-community-highlights-parser">Highs Parser - a Hugging Face Space by rrg92</a>: no description found</li><li><a href="https://huggingface.co/spaces/TheStinger/UVR5_UI">UVR5 UI - a Hugging Face Space by TheStinger</a>: no description found</li><li><a href="https://github.com/LegallyCoder/CyclicFormer">GitHub - LegallyCoder/CyclicFormer: CyclicFormer is a new architecture designed to enhance the performance of the transformer architecture. It introduces a new perspective for decoder layers, forming a cyclic loop between all the layers.</a>: CyclicFormer is a new architecture designed to enhance the performance of the transformer architecture. It introduces a new perspective for decoder layers, forming a cyclic loop between all the lay...</li><li><a href="https://huggingface.co/datasets/Vi-VLM/Vista">Vi-VLM/Vista · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Vi-VLM/Vistral-V-7B">Vi-VLM/Vistral-V-7B · Hugging Face</a>: no description found</li><li><a href="https://github.com/hllj/Vistral-V">GitHub - hllj/Vistral-V: Vistral-V: Visual Instruction Tuning for Vistral - Vietnamese Large Vision-Language Model.</a>: Vistral-V: Visual Instruction Tuning for Vistral - Vietnamese Large Vision-Language Model. - hllj/Vistral-V</li><li><a href="https://www.lightly.ai/post/vector-indexes-and-image-retrieval-using-lightly">Vector Indexes and Image Retrieval using&nbsp;lightly</a>: Use a pre-trained Vision Transformer provided by Lightly to create a vector index on an arbitrary dataset for Image Retrieval using faiss</li><li><a href="https://huggingface.co/spaces/lightly-ai/food101-image-retrieval">Food101 Image Retrieval - a Hugging Face Space by lightly-ai</a>: no description found</li><li><a href="https://x.com/MaheshkarSaurav/status/1808881869829853305">Tweet from Saurav Maheshkar ☕️ (@MaheshkarSaurav)</a>: 🚀 Latest work at @LightlyAI. Learn how you can create an Image Retrieval app using FAISS (@AIatMeta) as an vector index 🗃️, model implementations from the Lightly SSL package and @weights_biases for...</li><li><a href="https://colab.research.google.com/drive/1n4CwX5T6Ch2v7OYTRe6g1j_QJHxxOvcM?usp=sharing">Google Colab</a>: no description found</li></ul></div>

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1258614538463477921)** (7 messages):

> - `triton paper reading`
> - `upcoming paper reading schedule`
> - `interest in audio-language models`
> - `flora paper discussion`

- **Upcoming Paper Reading on Triton**: A member apologized for delaying a planned paper reading on **Triton** due to being busy and invited others to present if interested. Participants were encouraged to contact another member for more information.
- **Flora Paper Gains Interest**: A member expressed interest in the **Flora** paper, calling it cool. This paper seems to be gaining attention for an upcoming discussion.

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1258329870401011745)** (4 messages):

> - `WHAM alternatives for human pose estimation in monocular, in-the-wild videos`
> - `Learning ViT and U-Net implementations`
> - `Using visual-semantic information to boost fine-grained image classification performance`
> - `Discussing zero/few shot multi-modal models at CVPR`

- **Searching for WHAM alternatives for wrestling animations**: A non-coder is looking for a machine learning method for **human pose estimation** in monocular, in-the-wild videos of **complex human interactions** like Brazilian jiu-jitsu. They struggled with WHAM due to its **complex Python and CV dependencies** and seek a more user-friendly alternative.
- **Learning ViT and U-Net from online resources**: A member shared a [link](https://www.learnpytorch.io/08_pytorch_paper_replicating/) to learn **ViT** and **U-Net** implementations from the **DL Specialization by Andrew Ng** and **CNN Course Week 3**.
- **Boosting image classification using visual-semantic info**: Another user inquired about leveraging **visual-semantic information** from captions/metadata to enhance **fine-grained image classification performance** beyond zero/few shot learning. Florence 2 was suggested as a potential model for this specific supervised fine-tuning.

**Link mentioned**: [08\. PyTorch Paper Replicating - Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/08_pytorch_paper_replicating/): Learn important machine learning concepts hands-on by writing PyTorch code.

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1258136650824028261)** (17 messages🔥):

> - `Meta-LLaMA download issues`
> - `API calls to models without local download`
> - `Inference freeze in Mistral model`
> - `Static KV cache documentation`
> - `Troubleshooting errors related to memory`

- ****Meta-LLaMA download struggles****: A user expressed frustration over **Meta-LLaMA** taking forever to download and worried about their hard drive filling up due to potential temp files.
- ****API call confusion****: There was confusion on whether one could build an API call to a model without a local download, questioning the feasibility of this approach.
- ****Mistral model freezes at iteration 1800****: **Mistral** froze at iteration 1800 during inference of 3000 runs, whereas it worked fine for 100 inferences, leading to suspicion of some kind of caching problem.
- ****Static KV cache causes confusion****: A user highlighted that the static **KV cache** is on by default since version 4.41, suggesting checking the [relevant release](https://github.com/huggingface/transformers/releases/tag/v4.38.0) for more details.
- ****TypedStorage deprecation concern****: Concerns were raised about **TypedStorage** being deprecated, with a suggestion to wait for a stable solution before making any code changes.

**Link mentioned**: [Release v4.38: Gemma, Depth Anything, Stable LM; Static Cache, HF Quantizer, AQLM · huggingface/transformers](https://github.com/huggingface/transformers/releases/tag/v4.38.0): New model additions 💎 Gemma 💎 Gemma is a new opensource Language Model series from Google AI that comes with a 2B and 7B variant. The release comes with the pre-trained and instruction fine-tuned v....

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1258165680478621706)** (3 messages):

> - `Running RealVisXL_V4.0_Lightning using diffusers`
> - `Error with yisol/IDM-VTON in Google Colab`
> - `Improving resume analyzer to assess project intensity`

- ****RealVisXL V4.0 Lightning model release****: [RealVisXL V4.0 Lightning](https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning) is in training and supports photorealistic images in both sfw and nsfw categories. Users can support the creator on [Boosty](https://boosty.to/sg_161222) and find the CivitAI page [here](https://civitai.com/models/139562/realvisxl-v40).
- ****Diffusers don't match A1111 quality****: A user reported that the RealVisXL V4.0 model works well with A1111 but produces poorer quality images with diffusers despite using the same parameters.
- ****Error with IDM-VTON in Google Colab****: A user is encountering a 'no file named diffusion_pytorch_model.bin' error while using yisol/IDM-VTON on Google Colab.
- ****Enhancing Resume Analyzer Beyond Keywords****: A user is seeking advice on creating a resume analyzer that evaluates project intensity rather than just matching keywords. They aim to differentiate between less complex tasks and more significant projects.

**Link mentioned**: [SG161222/RealVisXL_V4.0_Lightning · Hugging Face](https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning): no description found

---

### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1258830278483382343)** (1 messages):

> - `License concerns with SD3 Medium release`
> - `Stability AI Community License update`
> - `Issues with commercial licensing in previous release`
> - `Improvement and support for open source community`

- **Stability AI updates license for broader use**: Stability AI acknowledged that their **SD3 Medium** release didn't meet community expectations and the associated commercial license caused confusion. They have revised the license for individual creators and small businesses, covered under the new **Stability AI Community License**, [read the full update here](http://stability.ai/news/license-update).
- **Non-commercial use is free under new Stability AI License**: Under the new Stability AI Community License, **non-commercial use remains free**. This change supports the open source community by giving broader access to recent releases, including *SD3 Medium*.

**Link mentioned**: [Community License — Stability AI](http://stability.ai/news/license-update): Our new Community License is now free for research, non-commercial, and commercial use. You only need a paid Enterprise license if your yearly revenues exceed USD$1M and you use Stability AI models in...

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1258163876957126717)** (528 messages🔥🔥🔥):

> - `Hyper vs turbo`
> - `AAM Anime Mix XL`
> - `Animagine XL 3.1`
> - `Stable Diffusion GPU usage`
> - `CivitAI and SD3 discussions`

- **Hyper is the new Turbo: Animagine XL 3.1 Updates**: Users discussed the merits of the anime-themed model **Animagine XL 3.1**. This model improves on **Animagine XL 3.0** with higher quality images and a broadened character range from well-known anime series, developed by [Cagliostro Research Lab](https://huggingface.co/cagliostrolab) and [SeaArt.ai](https://www.seaart.ai/).
- **AAM Anime Mix XL Gains Attention**: A user shared their enthusiasm for **AAM Anime Mix XL**, another popular anime image generation model. This sparked comparisons and recommendations for related models like Animagine XL 3.1.
- **Struggles with Multiple GPU Configurations**: Users discussed the challenges and potential solutions for using multiple GPU setups to improve Stable Diffusion's speed and output quality. Specific tools like **SwarmUI** were highlighted for their capabilities of handling multi-GPU operations.
- **CivitAI's SD3 Ban Sparks Debate**: The community reacted to **CivitAI's** ban on SD3 models with mixed opinions. Many expressed that this move could hinder the development of **SD3**, while others discussed the technical and licensing issues surrounding the model.
- **Stable Diffusion Licensing and Model Updates**: The conversation included concerns about the **license** for **Stable Diffusion 3** and its new models. There were debates over whether the licensing terms were too restrictive, affecting both small and large business users.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://civitai.com/articles/6024/using-a1111-why-not-swarmui-a-transition-guide">Using A1111? Why not SwarmUI? - A transition guide | Civitai</a>: I’ve recently transitioned from Forge to SwarmUI (previously known as StableSwarmUI), and I’m really glad I did! I had experimented with it before,...</li><li><a href="https://youtu.be/6Q4BJOcvwGE?si=LdajWOtf4iTKGVWJ&amp;t=844">SegMoE - The Stable Diffusion Mixture of Experts for Image Generation!</a>: Mixture of experts. Seems hot for AI text generation... but what if you had a mixture of experts for IMAGE generation? Oh. Segmind just did that. Welcome to ...</li><li><a href="https://www.youtube.com/watch?v=XtMvk0dpnO4&amp;list=PLNlRhPQovztRqp_zyp-lY79fWZIzjnNTf">How to Make Concept Art with AI (Free and Easy) - Stable Diffusion Tutorial 2022</a>: ATTENTION! Lots has changed for the better since I made this video! Here’s my guide how to install and use Stable Diffusion in June 2023: https://youtu.be/nB...</li><li><a href="https://huggingface.co/cagliostrolab/animagine-xl-3.1">cagliostrolab/animagine-xl-3.1 · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1do5gvz/the_open_model_initiative_invoke_comfy_org/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/vladmandic/automatic/">GitHub - vladmandic/automatic: SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models</a>: SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models - vladmandic/automatic</li><li><a href="https://github.com/ltdrdata/ComfyUI-Manager">GitHub - ltdrdata/ComfyUI-Manager: ComfyUI-Manager is an extension designed to enhance the usability of ComfyUI. It offers management functions to install, remove, disable, and enable various custom nodes of ComfyUI. Furthermore, this extension provides a hub feature and convenience functions to access a wide range of information within ComfyUI.</a>: ComfyUI-Manager is an extension designed to enhance the usability of ComfyUI. It offers management functions to install, remove, disable, and enable various custom nodes of ComfyUI. Furthermore, th...</li><li><a href="https://huggingface.co/models?search=sdxl%20controlnet%20tile">Models - Hugging Face</a>: no description found</li><li><a href="https://poe.com/PhdExpert-CDvr4">PhdExpert-CDvr4 - Poe</a>: INPUT YOUR DESIRED LANGUAGE. [TOP NOTCH RESPONSE EXPECTED]</li><li><a href="https://github.com/kijai/ComfyUI-LivePortrait?tab=readme-ov-file">GitHub - kijai/ComfyUI-LivePortraitKJ: ComfyUI nodes for LivePortrait</a>: ComfyUI nodes for LivePortrait. Contribute to kijai/ComfyUI-LivePortraitKJ development by creating an account on GitHub.</li><li><a href="https://huggingface.co/ptx0">ptx0 (PseudoTerminal X)</a>: no description found</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">Three GPTs Walk into a Bar and Write an Exec Summary – D-Squared</a>: no description found</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1258137476757852291)** (267 messages🔥🔥):

> - `Gemma 2 Release and its features`
> - `Issues with the Gemma 2 notebooks and user feedback`
> - `Methods for dataset preparation and handling long-context examples`
> - `Performance and optimization techniques for various LLMs`
> - `Recent advancements and announcements in AI models and tools`

- ****Gemma 2 Release brings speed and VRAM improvements****: The **Gemma 2 Release** is now available, claiming **2x faster finetuning** and using **63% less VRAM** compared to Flash Attention 2 ([Gemma 2 Blog](https://unsloth.ai/blog/gemma2)). Key details include support for **up to 9.7K context lengths** with Unsloth.
  - "Blogpost was super rushed honestly <slothlaughcry>I already found some mistakes," noted by a community member highlighting the fast-paced release.</slothlaughcry>
- ****Unsloth notebooks and model directory issues****: **Users reported issues** with the **Gemma 2 notebooks**, particularly errors related to model directory naming and missing configurations (e.g., `unsloth/gemma` instead of `unsloth_gemma`). **Collaboration and quick fixes** were made by the developers to address these problems.
- ****Training on long-context examples and dataset preparation techniques****: Members discussed techniques for handling **long-context datasets**, with some examples reaching up to **78,451 tokens**. Suggestions included setting appropriate context lengths and using **specific functions to find max tokens** in a dataset.
  - Sharing functions and discussing **prompt engineering** methods were common themes. Practical advice like, "you can choose the tone in the instruction part," were shared to help users better format their data for model training.
- ****Gemma 2 performance and limitations in the absence of Flash Attention support****: Without Flash Attention support, **Gemma 2** models are reported to be notably slow and almost **unusable for intensive tasks**. This highlights the significant impact of optimized attention mechanisms on model performance.
  - Community members suggested that **gradacc (gradient accumulation)** might be a more efficient approach than traditional batching, with one noting, "If anything, gradacc was faster."
- ****New AI models and tools announcements****: Nomic AI announced **GPT4ALL 3.0**, a new open-source local LLM desktop app, emphasizing privacy and local data processing ([GPT4ALL 3.0 Announcement](https://home.nomic.ai/gpt4all)). It's praised for supporting thousands of models and major operating systems.
  - InternLM-XComposer-2.5 was also mentioned, highlighting its capabilities to support **long-context input and output**, achieving GPT-4V level performance with just a 7B LLM backend ([InternLM-XComposer-2.5](https://huggingface.co/internlm/internlm-xcomposer2d5-7b)).

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://unsloth.ai/blog/gemma2">Finetune Gemma 2 with Unsloth</a>: Fine-tune Google's new Gemma 2 model 2x faster with 63% less memory VRAM via Unsloth! 9B and 27B parameters.</li><li><a href="https://youtu.be/ZJKglSWgD0w?si=20kiqxXIPvelywyJ">Emotions in AI: Fine-Tuning, Classifying, and Reinforcement Learning</a>: In this video we are exploring the creation of fine-tuning dataset for LLM's using Unsloth and Ollama to train a specialized model for emotions detection.You...</li><li><a href="https://huggingface.co/mlx-community/Phi-3-mini-4k-instruct-8bit">mlx-community/Phi-3-mini-4k-instruct-8bit · Hugging Face</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1808622550467162219">Tweet from Daniel Han (@danielhanchen)</a>: Gemma 2 finetuning is now 2x faster and uses 63% less VRAM with @UnslothAI! 1. We fixed 2 issues in the official Gemma repo 2. 27b Softcapping must be done on attn &amp; logits, or losses will diverge. 9...</li><li><a href="https://github.com/unslothai/unsloth/blob/9b4cc934efec66abd0a77df011779b393a99c026/unsloth/models/llama.py#L1175-L1179">unsloth/unsloth/models/llama.py at 9b4cc934efec66abd0a77df011779b393a99c026 · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/b4rtaz/distributed-llama">GitHub - b4rtaz/distributed-llama: Tensor parallelism is all you need. Run LLMs on weak devices or make powerful devices even more powerful by distributing the workload and dividing the RAM usage.</a>: Tensor parallelism is all you need. Run LLMs on weak devices or make powerful devices even more powerful by distributing the workload and dividing the RAM usage. - b4rtaz/distributed-llama</li><li><a href="https://x.com/nomic_ai/status/1808162955806097767">Tweet from Nomic AI (@nomic_ai)</a>: Launching GPT4All 3.0: The Open-Source Local LLM Desktop App - Completely Private Experience - Supports 1000’s of models and all major operating systems - Major UI/UX Improvements - Local File Chat -...</li><li><a href="https://home.nomic.ai/gpt4all">GPT4All</a>: Run Large Language Models Locally: privacy-first and no internet required</li><li><a href="https://github.com/google/gemma_pytorch/pull/67">Fix downcasting and upcasting by danielhanchen · Pull Request #67 · google/gemma_pytorch</a>: Fixes RMS Layernorm downcasting prematurely. We move it to the very end. Fixes embedding matrix scaling / normalizer upcasting to float32. Instead we must use float16 or bfloat16 for the normali...</li><li><a href="https://tenor.com/view/baby-face-palm-really-sigh-stupid-gif-12738431">Baby Face Palm GIF - Baby Face Palm Really - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://x.com/_akhaliq/status/1808747694317261114">Tweet from AK (@_akhaliq)</a>: InternLM-XComposer-2.5 A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output We present InternLM-XComposer-2.5 (IXC-2.5), a versatile large-vision language model that s...</li><li><a href="https://huggingface.co/internlm/internlm-xcomposer2d5-7b">internlm/internlm-xcomposer2d5-7b · Hugging Face</a>: no description found</li><li><a href="https://hqjiang.com/minference.html">MInference: Million-Tokens Prompt Inference for LLMs</a>: no description found</li><li><a href="https://research.nvidia.com/labs/toronto-ai/FMS/">Forecasting Model Search</a>: no description found</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1258151227787706450)** (1 messages):

> - `Gemma 2 Release`
> - `Training speed and VRAM reduction`
> - `Context length improvements`
> - `4-bit model support updates`
> - `Experimentation with models`

- ****Gemma 2 speeds up finetuning****: Unsloth now supports **Gemma 2** with **2x faster training** and **63% less memory usage**. Check out the [Gemma 2 Blog](https://unsloth.ai/blog/gemma2) for more details.
- ****Context lengths boosted significantly****: You can now finetune **Gemma 2 (27B)** with **9.7K context lengths** on a 40GB GPU using Unsloth, compared to 3K with HF+FA2. The **9B model** achieves **11K context lengths** on a 24GB card, versus 2.6K with HF+FA2.
- ****New Free Notebooks available****: Access the [Gemma 2 (9B) Colab notebook](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4) to get started with the latest model. **Gemma 2 (27B)** notebook support has also been added.
- ****4-bit models now supported****: Explore the new 4-bit models: [Gemma 2 (9B) Base](https://huggingface.co/unsloth/gemma-2-9b-bnb-4bit), [Gemma 2 (9B) Instruct](https://huggingface.co/unsloth/gemma-2-9b-it-bnb-4bit), [Gemma 2 (27B) Base](https://huggingface.co/unsloth/gemma-2-27b-bnb-4bit), and [Gemma 2 (27B) Instruct](https://huggingface.co/unsloth/gemma-2-27b-it-bnb-4bit). The **Phi 3 mini** update is also available on HF.
- ****Call for community experimentation****: Unsloth encourages users to share, test, and discuss their **models and results** in their community channels. Join the discussion and experiment with the latest updates.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://unsloth.ai/blog/gemma2">Finetune Gemma 2 with Unsloth</a>: Fine-tune Google's new Gemma 2 model 2x faster with 63% less memory VRAM via Unsloth! 9B and 27B parameters.</li><li><a href="https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)">Google Colab</a>: no description found</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1258172763252850719)** (7 messages):

> - `Release of Replete-AI datasets`
> - `Discussion on Facebook multi-token prediction`
> - `Fireworks.ai yi-large issues`

- ****Replete-AI Drops Massive Datasets****: Replete-AI announced the release of [two new datasets](https://huggingface.co/datasets/Replete-AI/Everything_Instruct) each around 11-12GB and containing over 6 million rows of data. The datasets include an English-only version and a multilingual version aimed at training versatile AI models.
- ****Is Facebook's Multi-Token Prediction Worth it?****: Discussion sparked about the worthiness of [Facebook's multi-token prediction model](https://huggingface.co/facebook/multi-token-prediction) that requires sharing contact information to access. One member expressed skepticism, while another deemed it worthwhile despite Facebook's involvement.
- ****Fireworks.ai yi-large Disappoints Users****: Users reported frustrations with the yi-large model on Fireworks.ai. One user admitted to being 'jebaited' by the model, indicating it did not meet their expectations.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://huggingface.co/facebook/multi-token-prediction">facebook/multi-token-prediction · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct">Replete-AI/Everything_Instruct · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual">Replete-AI/Everything_Instruct_Multilingual · Datasets at Hugging Face</a>: no description found</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1258142841218011297)** (121 messages🔥🔥):

> - `Issues with Unsloth patch 2024.7 and checkpoints`
> - `Gemma 2 support in Unsloth`
> - `Fine-tuning models using Unsloth`
> - `Errors during fine-tuning and evaluation processes`
> - `Updating Unsloth and GGUF issues`

- ****Gemma 2 support announced in Unsloth!****: Unsloth has added support for **Gemma 2**; you can now update and try the new features with the latest patch [2024.7](https://github.com/unslothai/unsloth).
- ****Checkpoint training errors in Unsloth patch 2024.7****: Users reported errors like `RuntimeError: Expected all tensors to be on the same device` when resuming training from a checkpoint in **Unsloth patch 2024.7**. Some suggested returning to older versions, but issues persist and require investigation.
- ****Unsloth fine-tuning pitfalls****: Some users experienced issues fine-tuning **Gemma 1.1** and **Phi-3 mini** models without LoRA; it works for Phi-3 but raises errors when attempted with full fine-tuning on Gemma 1.1.
- ****Errors with specific models and configurations****: Various errors were encountered, such as `RuntimeError: The size of tensor a (4096) must match the size of tensor b (4608)`, when dealing with large models like **Gemma-2-27B-bnb-4bit** and potential VRAM issues noted during evaluation with specific metrics.
- ****Updating Unsloth and handling GGUF issues****: Users were guided to update Unsloth via the wiki; some faced errors pushing fine-tuned models to Hugging Face due to GGUF quantization issues, which have since been fixed according to dev updates.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://huggingface.co/mlx-community/Phi-3-mini-4k-instruct-8bit">mlx-community/Phi-3-mini-4k-instruct-8bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/tokenizer_config.json">tokenizer_config.json · microsoft/Phi-3-mini-128k-instruct at main</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://discuss.huggingface.co/t/adding-accuracy-precision-recall-and-f1-score-metrics-during-training/16419/2">Adding accuracy, precision, recall and f1 score metrics during training</a>: hi, you can define your computing metric function and pass it into the trainer. Here is an example of computing metrics. define accuracy metrics function from sklearn.metrics import accuracy_score, ...</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/pull/671">Ollama by danielhanchen · Pull Request #671 · unslothai/unsloth</a>: no description found</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1258172565105541152)** (3 messages):

> - `Release of two new massive datasets by Replete-AI`
> - `Details and translations of Everything_Instruct_Multilingual`
> - `Questions about dataset deduplication and content balance`

- **Replete-AI unveils massive instruct datasets**: Replete-AI released two new datasets, **Everything_Instruct** and **Everything_Instruct_Multilingual**, each sizing 11-12GB with over 6 million rows of data. These datasets combine multiple types of instruct data to train advanced AI models in English and multilingual versions.
- **Translations for Everything_Instruct_Multilingual demo**: A message demonstrated the **Everything_Instruct_Multilingual** dataset by providing translations in **10 different languages** including Arabic, German, Spanish, and French for simple English commands.
  - Translations such as 'wake me up at nine am on friday' were shown in each language, like German: 'weck mich am freitag um neun uhr auf'.
- **Community queries dataset quality**: Community members raised questions about the new datasets' quality, asking if they are **deduped** and **decontaminated**. Another member expressed concerns regarding the dataset's balance, noting that nearly **50%** is code-related.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual">Replete-AI/Everything_Instruct_Multilingual · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct">Replete-AI/Everything_Instruct · Datasets at Hugging Face</a>: no description found</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1258148010303688834)** (10 messages🔥):

> - `Pinning notebooks`
> - `Adding notebooks to the GitHub page`
> - `Correcting notebook links in the channels`

- ****Pinning notebooks request confirmed****: A member requested that certain notebooks be pinned, and another member confirmed that they would do so, asking for some time.
- ****Notebook links corrected in channels****: A correction was made to the notebooks linked in the channels, clarifying that there were two notebooks: one about using multiple datasets and another about text classification.
- ****Notebooks to be added to GitHub page****: It was mentioned that the notebooks will be added to the GitHub page, but more time is needed for checking and editing.

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1258138493779972189)** (94 messages🔥🔥):

> - `AI + Blockchain funding discussions`
> - `Git merge tool alternatives and conflict resolutions`
> - `Learning AI curriculum and recommendations`
> - `Claude and other AI tools for coding assistance`
> - `Evaluations and criticisms of new search algorithms like BM42`

- ****AI + Blockchain grabs $85M seed****: "AI + Blockchain = $85M seed ☠️ vcs are cooked," one member stated, joking about the massive funding while sharing a link to a free project: [rakis](https://github.com/hrishioa/rakis?tab=readme-ov-file).
- ****Git Merge Tools Showdown****: Members discussed various tools for resolving git merge conflicts, including interactive rebase tools like [lazygit](https://github.com/jesseduffield/lazygit) and [Sublime Merge](https://www.sublimemerge.com/), emphasizing the tediousness of manual conflict resolution.
- ****Learning AI Curriculum for Beginners****: A user looking for AI learning resources received suggestions such as Replit's 100 Days of Code and the Deep Learning Specialization by Andrew Ng, and preferred interactive courses over books like [Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/).
- ****Claude 3.5 and Other AI Tools for Coding****: Users shared their experiences with coding tools like Claude 3.5 and aider, with favorable mentions for Cursor in terms of code completion and the ability to handle complex multi-file refactors.
- ****Controversy Over BM42 Search Algorithm****: The introduction of BM42 by Qdrant faced criticism for presenting potentially misleading benchmarks, prompting the developers to revise their evaluation metrics and datasets, as seen in their [follow-up post](https://x.com/qdrant_engine/status/1809291686625046816).

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/qdrant_engine/status/1808498752107241949?s=46">Tweet from Qdrant (@qdrant_engine)</a>: For 40 years, BM25 has been the standard for search engines. However, it falls short for modern RAG applications. Say hello to BM42: The combination of semantic and keyword search</li><li><a href="https://github.com/wesen/glazed/blob/e180e5d59031f20009c461466a2995ff28ee25a7/pkg/doc/topics/13-layers-and-parsed-layers.md">glazed/pkg/doc/topics/13-layers-and-parsed-layers.md at e180e5d59031f20009c461466a2995ff28ee25a7 · wesen/glazed</a>: a library to make it easy to output structured data in your command line tools. add the icing on top of your data - wesen/glazed</li><li><a href="https://x.com/jobergum/status/1809157587612336402">Tweet from Jo Kristian Bergum (@jobergum)</a>: Okay, gloves off. What @qdrant_engine did with the BM42 post is unacceptable. They are misguiding the RAG community in a big way. 1) Presenting Quora as a relevant RAG question-answering dataset. I...</li><li><a href="https://x.com/qdrant_engine/status/1809291686625046816">Tweet from Qdrant (@qdrant_engine)</a>: Hey all! We actually did find a discrepancy with our previous benchmarks of bm42. Please don't trust us and always check performance on your own data. Our best effort to correct it is here: http...</li><li><a href="https://www.manning.com/books/build-a-large-language-model-from-scratch">Build a Large Language Model (From Scratch)</a>: Learn how to create, train, and tweak large language models (LLMs) by building one from the ground up!&lt;/b&gt;<p>In Build a Large Language Model (from Scratch)&lt;/i&gt;, you’ll discover how LLMs w...</p></li></ul></div>

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1258858993888137227)** (5 messages):

> - `New podcast episode with Yi Tay of Reka`
> - `Discussion on the qualities of successful AI researchers`
> - `Comparisons of OpenAI, Google Gemini, and Reka teams`
> - `Technical topics covered in the podcast`

- **Yi Tay on YOLO Researcher Metagame**: [New podcast episode](https://latent.space/p/yitay) with **Yi Tay** of **Reka** discusses his team’s journey in building a new training stack from scratch and training frontier models purely based on gut feeling. **Yi Tay** draws comparisons to OpenAI and Google Gemini team sizes and reflects on the research culture at **Reka**.
  - *"@sama once speculated on the qualities of '10,000x AI researchers', and more recently @_jasonwei described the 'Yolo run' researcher."* Detailed topics include LLM trends, RAG, and Open Source vs Closed Models.
- **Now on Hacker News**: [Latent Space Podcast](https://news.ycombinator.com/newest) episode with Yi Tay is now featured on Hacker News. Engage with the [discussion](https://news.ycombinator.com/item?id=40886218) and vote for visibility.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://news.ycombinator.com/newest">New Links | Hacker News</a>: no description found</li><li><a href="https://x.com/latentspacepod/status/1809300018907828285">Tweet from Latent Space Podcast (@latentspacepod)</a>: 🆕 pod: The Yolo Researcher Metagame with @YiTayML! https://latent.space/p/yitay OpenAI (ca. GPT4): ~600 people Google Gemini: ~950 coauthors @RekaAILabs: 20 people @sama once speculated on the qua...</li></ul></div>

---

### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1258135475609145376)** (34 messages🔥):

> - `Issues with Discord AV`
> - `Migration to Zoom for better AV`
> - `Known compatibility issues between Discord and Linux`

- ****Discord AV struggles during AIEWF demo****: **OpenAI AV** faced **significant issues** during the AIEWF demo, with multiple users unable to see the screen and experiencing cut-outs. *Eugene and others suggested switching to Zoom for a more stable experience.*
  - *swyxio added:*
- ****Switch to Zoom for Paper Club****: The group decided to switch from Discord to **Zoom** due to continuous AV issues. The Zoom link [was shared](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09), and members began migrating.
- ****Discord-Linux compatibility problems discussed****: Several participants highlighted **known compatibility problems** between Discord and Linux. *Eugene added that Discord does not play well with Linux* and suggested looking into alternatives.

**Link mentioned**: [Join our Cloud HD Video Meeting](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09): Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1258874839163207785)** (243 messages🔥🔥):

> - `User technical difficulties and skill humor`
> - `Personal compliments to workshop hosts`
> - `Discussion on model merging tactics`
> - `LlamaFile vs Ollama comparison`
> - `Event planning and feedback`

- ****Users Battle Technical Issues and Share Laughs****: A user struggled to hear during a call, prompting jokes and the now-popular phrase, **'skill issue tbh'**. Eventually, the user realized they were not in the call and reconnected with a humorous resolution.
- ****LlamaFile vs Ollama: Divergent Aims****: Community members compared **LlamaFile** and **Ollama**, noting LlamaFile's strength in **portability and optimization** versus Ollama's **broad compatibility with numerous models**.
- ****Model Merging Tactics****: A discussion highlighted the difference in product goals between **LlamaFile and Ollama** while raising ideas of potential model merging tactics and respective improvements needed on both sides.
- ****AI-Generated Notes and Wearable Tech****: Discussion on the use of wearables highlighted their potential privacy concerns and the importance of consent in recording. Participants mentioned ambitions to integrate wearables with AI-generated notes for easier event navigation.
- ****Upcoming Event Plans and Feedback****: Participants brainstormed potential improvements for future events, considering additional days for workshops and community events and noting the success of current methods for organizing and executing productive AI conferences.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.ivanleo.com/blog/ai-conf">AI Engineering World Fair</a>: no description found</li><li><a href="https://aie.compasswearable.com/events">AI Engineers World Fair Recaps - Powered by Compass</a>: Experience the biggest technical AI conference with live transcriptions and AI-generated summaries.</li><li><a href="https://codingwithintelligence.com/p/ai-engineer-world-fair-in-sf">AI Engineer World Fair in SF</a>: Week 26 of Coding with Intelligence</li><li><a href="https://x.com/RickLamers/status/1808705188024439187">Tweet from Rick Lamers (@RickLamers)</a>: Model merging is nuts, check out this family tree :0</li><li><a href="https://x.com/philip_kiely/status/1808589566921879702">Tweet from Philip Kiely (@philip_kiely)</a>: Here are 3 themes I picked up in 3 incredibly high-energy days at @aiDotEngineer World's Fair: 1. Open source is closing the gap 2. Inference everywhere 3. Evals are everything Details:</li><li><a href="https://docs.google.com/document/d/1TLXkcaNX6cvpiqqyo952_K2a7XTF064R44v3WL9CSbE/edit?usp=sharing">AI Engineering Worlds Fair</a>: AI Engineering Worlds Fair Thomas Dohmke Human centric approach - “co-pilot” Copilot helps devs be in the flow of software Democratizes access to information - onboarding Agent - ai dishwasher (side...</li><li><a href="https://docs.google.com/presentation/d/1A_yLcD6Sy1Nr_v2YesOzvtcg5yAmmrfPR2bU4dyxTzw/edit?usp=sharing">AI in action - 2024-07-05</a>: AI in action AI Engineers World Fair recap 2024-07-05</li><li><a href="https://x.com/intertwineai/status/1807060271828975632">Tweet from Bryan Young (@intertwineai)</a>: @aiDotEngineer Day 3 Recap and Wrap! 1/12: Day 3 of #AIEWF 2024 is over and it's clear we're just scratching the surface of AI's potential and defining what an @aiDotEngineer is. Here...</li><li><a href="https://x.com/intertwineai/status/1806270266965889289">Tweet from Bryan Young (@intertwineai)</a>: @aiDotEngineer 2nd Day Recap! 1/14. The second day started with a timely session on AI-generated music by @YoungPhlo_. We all made some sick beats together. Although the fresh @RIAA lawsuits agains...</li><li><a href="https://x.com/intertwineai/status/1805867608593645916">Tweet from Bryan Young (@intertwineai)</a>: 1/5: Day 1 of @aiDotEngineer was just as exciting as I thought it would be! #AIEWF Quick recap of the day:</li></ul></div>

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1258136112522854441)** (157 messages🔥🔥):

> - `Waiting to upgrade hardware for LM Studio`
> - `Comparison of Llama3 and Mistral models`
> - `Usage of API keys from OpenAI or Anthropic in LM Studio`
> - `Text embeddings and local server setup in LM Studio`
> - `Challenges in running large models like Llama3 70b on limited hardware`

- ****Waiting to upgrade hardware for LM Studio****: A user mentioned planning to wait for 2 years to buy a new laptop for **LM Studio**, preferring to use their current setup with **64GB DDR4 RAM**, **Ryzen 5900** CPU, and **3060 6GB GPU** in the meantime.
- ****Comparison of Llama3 and Mistral models****: Members discussed preferences, with some favoring **Llama3 8b** over **Mistral 7b Instruct 0.3**, and others highlighting successful experiences with **OpenHermes 2.5** finetuned from **Mistral**.
- ****Usage of API keys from OpenAI or Anthropic in LM Studio****: A user inquired whether **LM Studio** allows using API keys from **OpenAI** or **Anthropic** for loading their models. They were informed that **LM Studio** supports only local text models.
- ****Challenges in running large models like Llama3 70b on limited hardware****: A user reported issues running **Llama3 70b** on a **RTX 3090 Ti** due to memory constraints, receiving advice to lower GPU offload and context length or switch to smaller models.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator">LLM Model VRAM Calculator - a Hugging Face Space by NyxKrage</a>: no description found</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>: Text embeddings are a way to represent text as a vector of numbers.</li><li><a href="https://llama3.dev/">Llama 3 Chat Meta AI - Llama 3 Chat Online 8B and 70B</a>: Llama 3 is the latest language model from Meta.Llama 3 comes in two sizes: 8B and 70B.Quickly try out Llama 3 Online with this Llama chatbot.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1806ksz/information_on_vram_usage_of_llm_model/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/638">30B model now needs only 5.8GB of RAM? How? · ggerganov/llama.cpp · Discussion #638</a>: (Edit: apologies, I should have clarified initially I'm running on Linux OS. I didn't realize it might not be obvious from the screenshot alone for a non-Linux users.All tests are done on Ubun...</li></ul></div>

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1258136706755067915)** (130 messages🔥🔥):

> - `Discussion on model behavior mismatch between different quantized versions of Gemma-2-27b`
> - `Using system prompts to improve coding model behaviors`
> - `Comparing different quantization techniques and their performance`
> - `Qwen2 model preset and ChatML format discussion`
> - `Issues and experiences with different large language models like Gemma, InternLM, and Dolphin`

- ****Gemma 2 models underperform in benchmarks****: Users reported that **Gemma-2-27b** models performed poorly and erratically in benchmarks, with significant inconsistencies across different quantization methods (Q5_K_M or Q6_K). A specific test showed **vast discrepancies** in performance between **27b** and **9b** models.
- ****System prompts improve coding responses****: Crafting tailored system prompts for coding guidance improved response quality in models like **Gemma 2 27B**. A specific method, focusing on PEP 8 guidelines and efficient algorithms, enhanced **code generation consistency and completeness**.
- ****Understanding ChatML format for Qwen2****: New users struggled with using **Qwen2 models** due to the lack of clear instructions on ChatML presets. A detailed explanation on the **importance of ChatML format** helped clarify preset configurations.
- ****Issues with different quantization techniques****: Users discussed the instability of **IQ quants** on non-CUDA hardware, reporting slower token speeds and random behavior like infinite loops and inconsistent responses. It's advised to avoid IQ quants on **Apple devices** and consider other quantization methods for better performance.
- ****Experiences with various LLMs in game development and other tasks****: Members shared mixed results from using different large models like **Gemma, InternLM, and Dolphin** for tasks like game development and VFX pipelines. Models showed uneven performances in retaining context and following instructions, leading to concerns over **practical application and stability**.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chat-markup-language">How to work with the Chat Markup Language (preview) - Azure OpenAI</a>: Learn how to work with Chat Markup Language (preview)</li><li><a href="https://huggingface.co/facebook/multi-token-prediction">facebook/multi-token-prediction · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/koboldai-erebus-extended-32k-7B-GGUF?not-for-all-audiences=true">mradermacher/koboldai-erebus-extended-32k-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Qwen2-7B-Instruct-GGUF">bartowski/Qwen2-7B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/ESFT-vanilla-lite">deepseek-ai/ESFT-vanilla-lite · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Downtown-Case/internlm2_5-7b-chat-1m-llamafied-Q6K-GGUF/tree/main">Downtown-Case/internlm2_5-7b-chat-1m-llamafied-Q6K-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/KoboldAI/Mistral-7B-Erebus-v3?not-for-all-audiences=true).">KoboldAI/Mistral-7B-Erebus-v3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/Mistral-7B-Erebus-v3-i1-GGUF?not-for-all-audiences=true).">mradermacher/Mistral-7B-Erebus-v3-i1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dvcqt5/checked_180_llms_on_writing_quality_code_for_deep/">Reddit - Dive into anything</a>: no description found</li></ul></div>

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1258451443069751357)** (3 messages):

> - `Issue with model downloads in LM on MacBook Pro M2`
> - `Solution for pausing/stopping downloads in LM`

- **Models Get Stuck Downloading on MacBook Pro M2**: **msouga** experienced an issue with some models in LM getting stuck downloading indefinitely on their **MacBook Pro with an M2 chip**, unable to stop these downloads or estimate their completion time.
- **How to Pause/Stop Downloads in LM**: **a_dev_called_dj_65326** suggested checking under the **downloads section** (bottom bar) to pause or stop the downloads. **msouga** confirmed this solution worked perfectly.

---

### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1258814454661906525)** (5 messages):

> - `Nxcode 7B JSON request`
> - `CodeQwen 1.5 7B ChatML compatibility`
> - `RTX 4060 8GB VRAM and 16 GB DDR5 RAM performance issues`
> - `Suggested models for mid-range GPU setups`

- **Nxcode 7B JSON request**: @49206c696b652063757465 asked for a JSON for **Nxcode 7B** or **CodeQwen 1.5 7B**.
- **CodeQwen 1.5 7B ChatML compatibility**: @heyitsyorkie mentioned that both **Nxcode 7B** and **CodeQwen 1.5 7B** use **ChatML**, and **CodeQwen** requires **flash attention** enabled.
- **RTX 4060 8GB VRAM struggles with 20B models**: @falconandeagle123 shared that their laptop with **RTX 4060 8GB VRAM** and **16 GB DDR5 RAM** struggled to run **q4 quant 20B models**, causing the laptop to freeze.
- **Suggested models for mid-range GPU setups**: @niga256_512_1024_2048 suggested using simpler models like **Mistral 7B**, **Open Hermes 2.5**, **Wizard code**, and **Phi 3 mini** for mid-range GPU setups.
  - They pointed out that these models are more suitable for systems similar to a laptop with RTX 4060.

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1258297612751081535)** (61 messages🔥🔥):

> - `Surface Laptop with Snapdragon Elite performance details`
> - `NPU and GPU utilization in Snapdragon devices`
> - `Comparison of CPU performance on Snapdragon and Intel devices`
> - `Future support for NPU in Llama.cpp`
> - `General discussion on hardware used with LM Studio`

- \****Snapdragon Elite CPU holds its own**: *Member discusses performance details of Surface Laptop with Snapdragon Elite, including first token speed and tokens per second (t/s) on LLaMA3 models*. Other members compare this with their Intel quad-core laptops and find Snapdragon's CPU performance impressive.*\*: A member reports **1.5 seconds to first token** and **10 t/s on LLaMA3 8b** with **8bit precision** on a **Surface Laptop with Snapdragon Elite** and **32 GB of RAM**. They note **10% GPU usage** and no NPU activity, sparking curiosity about potential future NPU utilization.
  - Comparisons reveal **Snapdragon Elite CPU's** performance to be **significantly faster** than older Intel quad-core laptops, even rivaling typical cloud AI speeds. Members speculate about future NPU support possibly leading to further speed improvements.
- \***\*Future NPU support for Llama.cpp?\*\*: \*Discussion on when NPU support might land for Llama models in LM Studio.**\*: Members discuss that **NPU support is not yet available** in **Llama.cpp**, leading to **CPU-only performance for LLaMA models** in LM Studio. Speculation arises about when support might be implemented, with hopes set for late 2024 or early 2025.
  - Conversations reveal that **Qualcomm has a GitHub repository** showing LLaMA2 operating on NPU, though it's currently rough. Community shows enthusiasm for *future enhancements*\*, especially with Qualcomm and Microsoft pushing for NPU utilization.
- ****NPU implementation faces delays*\*: \*Members express hopes and struggles with current hardware performance.***: Efforts to implement **NPU** in existing systems have been slow, with members sharing links to discussions and repositories investigating the subject ([GitHub repo](https://discord.com/channels/1110598183144399058/1153759714082033735/1257360281936597103)).
  - Members appear optimistic about eventual improvements, even sharing humorous suggestions like a **Cardboard NPU** as a placeholder solution.
- ****Surface Laptop shows promise on Snapdragon Elite*\*: \*Users share their positive experience regarding the new Surface Laptop's build quality and performance.***: *A member praises the build quality, keyboard, and trackpad of their Surface Laptop with Snapdragon Elite*. They highlight the ability to **perform video editing and play games** as standout features.
  - Overall, the Surface Laptop with Snapdragon Elite is well-received, especially as a **daily driver for personal use**, despite needing separate work laptops with IT restrictions.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://tenor.com/view/office-michael-scott-thank-you-gif-5278681">Office Michael GIF - Office Michael Scott - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/not-funny-haha-not-funny-hahaha-evil-laugh-laughing-gif-17347025675359632212">Not Funny Haha Not Funny GIF - Not funny Haha not funny Hahaha - Discover &amp; Share GIFs</a>: Click to view the GIF</li></ul></div>

---

### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1258422461549121566)** (2 messages):

> - `AppImage not compatible with aarch64 CPUs`
> - `No ARM CPU support on Linux for LM Studio`

- ****AppImage not compatible with aarch64 CPUs****: A user encountered an **Exec format error** while trying to execute [LM_Studio-0.2.27.AppImage](https://link.to/LM_Studio-0.2.27.AppImage) on an **aarch64** system, indicating architecture incompatibility. The `lscpu` command output confirmed the CPU architecture as **aarch64**.
- ****No ARM CPU support on Linux****: Discussion highlighted the lack of **ARM CPU support** for **LM Studio** on Linux. A member confirmed, *"No arm cpu support on linux"*.

---

### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1258509419747086456)** (2 messages):

> - `7800XT user confirms GPU works`
> - `Problems loading models with GPU offload`
> - `Successful ROCm installation script`

- ****7800XT user confirms GPU works****: *User reports that their **7800XT** works successfully and is not sure if pinging is needed.*
- ****Problems loading models with GPU offload****: **Loading models failed** unless GPU offload is disabled. Users discussed installation scripts to address this issue.
- ****Successful ROCm installation script****: A **user suggested a script** to install ROCm that helped solve loading issues with GPU offload. Another user confirmed it works well.

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1258257005542375555)** (10 messages🔥):

> - `Matrix multiplication in CUDA`
> - `Efficient remote development with pay-for-use compute`
> - `New blog post on executive summary process using GPTs`
> - `Paid CUDA/ML system certifications`
> - `Upcoming in-person CUDA mode event in October`

- **Matrix Multiplication in CUDA: Why Column Instead of Row?**: A user questioned why a 64-element column is loaded on the purple tile instead of a row during a GPU matrix multiplication, and another shared a detailed [blog post](https://siboehm.com/articles/22/CUDA-MMM) for optimizing this process using CUDA.
- **Streamline Executive Summaries with GPTs**: A new [blog post](https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/) details a process involving three Custom GPTs to expedite the writing of executive summaries, showing how they can extract insights, draft, and revise the summaries quickly.
- **Tips for Efficient Remote Development**: Members discussed solutions for remote development that allows for pay-per-use compute while retaining files, mentioning services like **Lightning AI** and **AWS S3** as potential options.
- **Recommendations for CUDA/ML Certifications**: A user sought recommendations for paid CUDA/ML certifications under $500, leading to suggestions of NVIDIA courses and a possible community-organized workshop.
- **In-Person CUDA Mode Event Announced**: **CUDA Mode** is planning an in-person event for October, as revealed by a community member, promising more details soon.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>: In this post, I’ll iteratively optimize an implementation of matrix multiplication written in CUDA.My goal is not to build a cuBLAS replacement, but to deepl...</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">Three GPTs Walk into a Bar and Write an Exec Summary – D-Squared</a>: no description found</li></ul></div>

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1258221254331203666)** (2 messages):

> - `Triton kernels with multiple CUDA graphs create latency issues`
> - `SRAM contention affecting performance`

- ****Triton Kernels under Parallel CUDA Execution****: **Multiple CUDA graph instances** running in parallel with Triton kernels show **worse latencies** compared to local benchmarks.
  - It's suggested that **SRAM contention** might be a cause if multiple instances are doing `tl.load`.
- ****Comparison with Torch Performance****: Despite potential **SRAM contention**, this issue doesn't **seem present in Torch** under similar conditions.
  - This discrepancy raises questions about how SRAM evictions are **handled differently between Triton and Torch**.

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1258584845966249985)** (7 messages):

> - `torch.compile not supported on Python 3.12`
> - `Python bytecode compatibility issues`
> - `TorchDynamo and Python frame evaluation API`
> - `TorchDynamo's role in PyTorch performance`

- ****Torch 2.3 `.compile` Unsupported on Python 3.12****: For **torch 2.3**, the `.compile` function is **not supported on Python 3.12** due to changes in Python's internals, especially in how it handles bytecode. A detailed explanation can be found [here](https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-12-completed/2054).
- ****Python Bytecode Changes Cause Lag in Support****: **Python bytecode** changes every Python version, requiring time for frameworks like **torch.compile** to adjust and support these new changes. More information on the bytecode adjustments can be read [in this documentation](https://pytorch.org/docs/stable/torch.compiler_deepdive.html).
- ****TorchDynamo Enhances PyTorch Performance****: [TorchDynamo](https://pytorch.org/docs/stable/torch.compiler_deepdive.html) is a **Python-level JIT compiler** that hooks into CPython's frame evaluation to modify Python bytecode and compile PyTorch operations into an **FX Graph**. Using `torch._dynamo.optimize()` wrapped by `torch.compile()`, it boosts PyTorch code performance seamlessly.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://pytorch.org/docs/stable/torch.compiler_deepdive.html">TorchDynamo Deep Dive — PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-12-completed/2054">Torch.compile support for Python 3.12 completed</a>: Signal boosting that Python 3.12 support has been added to torch.compile and has been present in the nightly builds for a while. We anticipate that this feature will be included in the PyTorch 2.4 rel...</li></ul></div>

---

### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1258827477841743982)** (5 messages):

> - `New method for training language models to predict multiple future tokens`
> - `Self speculative decoding in language models`
> - `Comparison between multi-token prediction and lookahead decoding baselines`
> - `Effectiveness of n-gram generation in multi-token prediction models`

- **New approach boosts language model efficiency**: [Latest research paper](https://arxiv.org/abs/2404.19737) suggests training language models to predict multiple future tokens at once, resulting in higher sample efficiency and improved downstream capabilities with no additional training time. **13B parameter model** shows substantial gains, solving **12% more problems on HumanEval** and **17% more on MBPP**.
- **Self Speculative Decoding gets a thumbs up**: A member mentioned the cool aspect of the model's ability to perform **self speculative decoding**.
- **Questioning lookahead decoding baselines**: Members wondered how this new multi-token prediction compares to **lookahead decoding baselines**.
- **Dissecting n-gram effectiveness**: A discussion emerged on the effectiveness of generating **n-grams** in multi-token prediction models and their alignment with traditional next-token prediction outputs.

**Link mentioned**: [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737): Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in h...

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages):

iron_bound: [https://oimo.io/works/life/](https://oimo.io/works/life/)

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1258442106981187747)** (17 messages🔥):

> - `Learning path for backend SDEs interested in CUDA and inference optimization`
> - `Challenges of finding jobs with open source contributions`
> - `Recommendation of CUDA Mode GitHub for beginners`
> - `Building a deep learning framework from scratch in C++ using CUDA`
> - `Using Python for CUDA kernel development vs C++`

- ****Finding Path to CUDA Mastery****: A backend SDE seeks advice on transitioning to a job related to **CUDA** and **inference optimization**. Recommendations included **watching specific channels** and **reading relevant resources**, contributing to GitHub, and joining **working groups**.
- ****Open Source Contributions Not Always a Job Ticket****: Concerns were raised about individuals making significant **open source contributions** yet failing to secure jobs. The community acknowledged the challenge and discussed the high bar for entry.
- ****CUDA Mode GitHub: A Beginner's Treasure Trove****: For beginners looking to dive into CUDA, **CUDA Mode GitHub** was recommended as a fruitful starting point. It's suggested as a platform to build engaging projects and learn efficiently.
- ****Building Deep Learning Frameworks in C++ with CUDA****: A member expressed interest in building a deep learning framework similar to **tinygrad** using **CUDA** and **C++** for parallelism but encountered difficulties with C++ complexity. They considered using **Python** instead for better manageability and potential for faster completion.
- ****Python vs C++ for CUDA Kernel Development****: Debate ensued over whether to use Python or C++ for **CUDA** kernel development. The consensus leaned towards using Python for initial endeavors and transitioning to C++ for deep system-level work, citing repositories like **llama.c** for learning.

---

### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1258429258984849580)** (4 messages):

> - `Fourth edition released in 2022`
> - `Differences between third and fourth editions`

- ****Fourth edition released in 2022****: The fourth edition was released in 2022, whereas the previous edition was released in 2012.
- ****Differences between third and fourth editions****: A member mentioned not having read the third edition, expressing curiosity about the differences. Another member referred to the back of their copy for details.

---

### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1258851566044184587)** (4 messages):

> - `casual conversation`
> - `channel engagement`

- **Casual Engagement in Channel**: A member expressed their excitement with a simple *"that's so cool!"*, indicating casual engagement and appreciation.
  - Another member replied with *"thanks"*, showing a friendly and appreciative interaction in the channel.
- **Friendly Interactions**: Members engaged in a casual and friendly manner with short messages like *"yo"* and *"you"*.
  - These interactions reflect a positive and welcoming community environment.

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1258171846227136622)** (11 messages🔥):

> - `Handling` a.to `method recognition and functionality`
> - `Removing unnecessary args in PyTorch/ao`
> - `Current limitations and workarounds for` a.to `method`
> - `Adding support for` device `and dtype handling in subclasses`
> - `Future functionality and testing in Torchbench models`

- ****Fixing `a.to` issues in PyTorch****: The `a.to(torch.int32)` method is recognized as `a.to(device=torch.int32)` causing unexpected behavior, and needing removal of unnecessary `device` and `memory_format` arguments in [affine_quantized_tensor.py](https://github.com/pytorch/ao/blob/a8956992191853b13f82ceb3e6929bed7691a3fa/torchao/dtypes/affine_quantized_tensor.py#L262) to fix this issue.
- ****Challenges with `a.to(dtype=torch.int32)`****: A discussion highlighted that `a.to(dtype=torch.int32)` currently only changes the device and not other keywords like dtype or layout, indicating that **dtype and memory format changes** are unsupported for now.
- ****Temporary Function Adjustments in AQT****: A suggestion was made to modify the `affine_quantized_tensor.py` file to temporarily drop `device`, `dtype`, and `memory_format` arguments to handle the limitations in the current implementation.
- ****Subclass `a.to` Method Limitations****: Discussion around subclass functionality in `torchbench` revealed that handling `a.to` method for differing dtypes was not intended as changing external representations' dtype poses complex challenges.
- ****Testing Functionality in Torchbench****: Concerns were raised about whether the current setup supports `.to` method across various models in `torchbench`, especially regarding **subclass handling** and required functionality testing in AQT implementations.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://github.com/pytorch/ao/blob/a8956992191853b13f82ceb3e6929bed7691a3fa/torchao/dtypes/affine_quantized_tensor.py#L262">ao/torchao/dtypes/affine_quantized_tensor.py at a8956992191853b13f82ceb3e6929bed7691a3fa · pytorch/ao</a>: Create and integrate custom data types, layouts and kernels with up to 2x speedups and 65% less VRAM for inference and training - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/a8956992191853b13f82ceb3e6929bed7691a3fa/torchao/dtypes/affine_quantized_tensor.py#L261:">ao/torchao/dtypes/affine_quantized_tensor.py at a8956992191853b13f82ceb3e6929bed7691a3fa · pytorch/ao</a>: Create and integrate custom data types, layouts and kernels with up to 2x speedups and 65% less VRAM for inference and training - pytorch/ao</li></ul></div>

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1258329101425709126)** (3 messages):

> - `Thunder Sessions podcast by Lightning AI`
> - `Andrej Karpathy's keynote at UC Berkeley AI Hackathon 2024`

- ****Thunder Sessions podcast ignites excitement****: Lightning AI announced the [Thunder Sessions podcast](https://x.com/LightningAI/status/1808610408481370205) hosted by **Luca Antiga** and **Thomas Viehmann** to cover compilers and performance optimization, airing **Friday, July 5 @ 11am EST**.
- ****Andrej Karpathy steals the show at UC Berkeley Hackathon****: The [YouTube video](https://www.youtube.com/watch?v=tsTeEkzO9xc) of the 2024 UC Berkeley AI Hackathon Awards Ceremony features **Andrej Karpathy** delivering an inspiring keynote, highlighting groundbreaking pitches from the participants.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/LightningAI/status/1808610408481370205">Tweet from Lightning AI ⚡️ (@LightningAI)</a>: We’re excited to introduce 🌩️&nbsp;Thunder Sessions 🌩️, a new podcast from the team at Lightning AI covering the world of compilers and performance optimization. Join us this Friday, July 5 @ 11am EST w...</li><li><a href="https://www.youtube.com/watch?v=tsTeEkzO9xc">Andrej Karpathy's Keynote &amp; Winner Pitches at UC Berkeley AI Hackathon 2024 Awards Ceremony</a>: At the 2024 UC Berkeley AI Hackathon's Awards Ceremony, the atmosphere was electric as Andrej Karpathy, founding member of OpenAI, delivered an inspiring key...</li></ul></div>

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1258149753305301012)** (134 messages🔥🔥):

> - `CUDA MODE Discord chatbot messages`
> - `FP8 Gradient Issues in GPT-2 Training`
> - `Schedule-Free Optimizer Paper`
> - `GPT-2 Training Performance`
> - `Training Length Estimations for GPT-2`

- ****Issues with Schedule-Free Optimizers****: A member noted that using [Schedule-Free Optimizers](https://x.com/_clashluke/status/1808590060654108910) produced surprisingly smooth loss curves, which seemed improbable on noisy datasets like ImageNet. Despite initial skepticism, the optimizer showed significant convergence advantages even without custom optimizations.
- ****FP8 Gradient Activations Impact GPT-2 Training****: A member found that converting gradient activations to FP8 significantly increased loss during GPT-2 test runs. They noted that this error propagated through the model, and attempts to mitigate it with stochastic rounding had limited success, suggesting keeping some operations in BF16 for stability.
- ****Technical Woes with Compile Times on Lambda Servers****: A user reported much longer compile times on Lambda servers compared to local machines, likely due to disabled CPU Turbo on virtualized instances. Investigations revealed the CPU staying at a base clock of 2GHz, unable to utilize its full potential of 3.8GHz Turbo clock speeds.
- ****Sweeps on Hyperparameters and Model Scaling****: Several discussions focused on sweeping different hyperparameters like LR, `attn_mult`, and `out_mult` across different model widths and depths. Preliminary results indicated that cosine schedulers and an `attn_mult` of 1 were optimal, but further tests were ongoing.
- ****Austin Tech Scene Tidbits****: Casual talk revealed that members attended July 4th parties with notable figures from the tech industry, like Lex Fridman. They also noted Austin's importance in semiconductor engineering but highlighted its lack of intersection with the broader tech scene.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/_clashluke/status/1808590060654108910?s=46&amp;t=Qzf619GMalbD77YmVui2Jw">Tweet from Lucas Nestler (@_clashluke)</a>: Schedule-free optimizers (https://x.com/aaron_defazio/status/1776320004465582331) are surreal. I've read the paper, looked into the math, and tried to understand what's happening. It all seem...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/scripts/run_gpt2_1558M.sh">llm.c/scripts/run_gpt2_1558M.sh at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li></ul></div>

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1258735996489433140)** (3 messages):

> - `Optimized kernels in CUDA for int2*int8 gemm`
> - `Release of a custom gemv for int2*int8`
> - `BitBLAS library for mixed-precision matrix multiplications`

- ****Newcomer asks about optimized kernels for int2*int8 gemm****: A new member asked if there are optimized kernels in **CUDA** for \**int2*int8 gemm\** operations.
- ****Custom gemv kernel release announced****: A member announced that they have made a custom **gemv kernel** for int2\*int8, which will be released in a few days.
  - They also suggested checking out [BitBLAS](https://github.com/microsoft/BitBLAS) as another option.

**Link mentioned**: [GitHub - microsoft/BitBLAS: BitBLAS is a library to support mixed-precision matrix multiplications, especially for quantized LLM deployment.](https://github.com/microsoft/BitBLAS): BitBLAS is a library to support mixed-precision matrix multiplications, especially for quantized LLM deployment. - microsoft/BitBLAS

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1258136049272487966)** (165 messages🔥🔥):

> - `Perplexity AI Repetition Issue`
> - `Live Internet Access Problems`
> - `Math Accuracy in Perplexity Pro`
> - `Experience with Perplexity in Stock Market`
> - `Subscription Plans and Model Usage`

- ****Perplexity AI Repetition Issue****: Users reported Perplexity AI giving repetitive answers with the same prompt, particularly with models like Llama 3 and Claude. One user mentioned that Alex responded they are aware of the issue and working on a fix.
- ****Live Internet Access Problems****: One user described issues with Perplexity AI accessing live internet for real-time data, providing inaccurate and outdated information instead. Despite closing and reopening the app, the problem persisted and the user noted it in the feedback channel.
- ****Math Accuracy in Perplexity Pro****: Users expressed frustration with Perplexity Pro's inaccuracies in handling math problems like CAPM beta calculations. Despite the model being GPT-4o, results were significantly off, raising doubts about the model's efficacy in academic calculations.
- ****Experience with Perplexity in Stock Market****: One user shared that they made $8k in the stock market using Perplexity, praising its abilities. This sparked a brief discussion about the various benefits users have experienced with the pro version.
- ****Subscription Plans and Model Usage****: Users discussed the differences between Pro and Enterprise Pro plans, with specifics on model usage like Sonnet and Opus. Questions emerged about the availability and specificity of models in different subscription plans.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: no description found</li><li><a href="https://www.mayoclinic.org/diseases-conditions/hyperthyroidism/symptoms-causes/syc-20373659">Hyperthyroidism - Symptoms and causes</a>: no description found</li><li><a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">Three GPTs Walk into a Bar and Write an Exec Summary – D-Squared</a>: no description found</li></ul></div>

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1258270754781528106)** (13 messages🔥):

> - `Threads' Milestone`
> - `Ancient Aboriginal Rituals`
> - `Nuclear-Powered Data Centers`
> - `Mars Moss`
> - `Eating Contests`

- ****Threads Hit Milestone****: A [YouTube video](https://www.youtube.com/embed/Q-jy32fjcSs) titled **Discover today: Threads' Milestone, Ancient Aboriginal Rituals, and Nuclear-Powered Data Centers** discusses the recent achievement by Threads.
- ****Mars Moss and Other Wonders****: Another [YouTube video](https://www.youtube.com/embed/PfWwPIB62d8) titled **Discover today: Mars Moss, Eating Contests, Tech Titans, and Toxic Green** explores the existence of moss on Mars and various unusual topics.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.youtube.com/embed/Q-jy32fjcSs">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/embed/PfWwPIB62d8">YouTube</a>: no description found</li></ul></div>

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1258281585703256165)** (15 messages🔥):

> - `Difference between pplx-70b-online and llama-3-sonar-large-32k-online`
> - `Google Dorks usage with the API`
> - `Temporal awareness in LLMs`
> - `Effectiveness of query commands in LLMs`
> - `Perplexity AI models and model cards`

- ****Google Dorks and API Mastery****: A user suggested that leveraging Google Dorks can enhance the utility of the API, as it simplifies filtering source domains effectively on web products.
- ****LLMs Lack Temporal Awareness****: Users discussed the inability of models like **llama3** and **haiku** to intuitively understand 'latest' or 'most recent' without explicit cues, influencing their responses.
- ****Query Commands in LLMs: Not Official****: It was highlighted that while **Google Dork operators** are often suggested to constrain results, they are not officially integrated into the backend of Perplexity's LLMs.
- ****Perplexity Model Clarification****: A user sought clarification on the difference between **pplx-70b-online** and **llama-3-sonar-large-32k-online** models, referencing both Perplexity's blog and API documentation.
- ****Model Alias and Obsolescence****: There was confusion over model aliases and potential obsolescence; one user suggested some models might be aliases, while another noted that certain models might now throw errors.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found</li><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: no description found</li></ul></div>

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1258332502016266342)** (185 messages🔥🔥):

> - `BUD-E update on new features`
> - `Issues with Clipdrop NSFW detection`
> - `Discussion on dataset availability and usage`
> - `Performance of various AI models and training techniques`
> - `Commercial licensing of Stability AI models`

- ****BUD-E updates with Clipboard Access****: [A recent YouTube video](https://youtu.be/WMcEzVfEXpM) showcases **BUD-E's new feature** of reading text from the screen and clipboard, detailed in the project description on [GitHub](https://github.com/christophschuhmann/Desktop_BUD-E/tree/main). The demo was presented in 240p resolution, which drew some humorous criticism.
- ****Clipdrop's NSFW Detection Failure****: A member shared a humorous incident where Clipdrop incorrectly labeled an image as **NSFW content**.
- ****Struggles with Dataset Availability****: Members discussed the difficulties faced by **FAL.AI** in acquiring new datasets, with comments highlighting the extensive reliance on the same datasets for multiple models. One user emphasized that interesting breakthroughs, like **Chameleon**, come from diverse and integrated modalities.
- ****Stability AI's License Fix****: [Stability AI](https://stability.ai/news/license-update) revised the commercial license for **SD3 Medium** to the Stability AI Community License, allowing broader free use for individual creators and small businesses. This change was made in response to community feedback regarding the original commercial license.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://youtu.be/WMcEzVfEXpM">BUD-E Update: Seeing images &amp; reading text from screen &amp; clipboard</a>: https://github.com/christophschuhmann/Desktop_BUD-E/tree/main</li><li><a href="https://huggingface.co/spaces/LPDoctor/Glyph-SDXL-v2/tree/main">LPDoctor/Glyph-SDXL-v2 at main</a>: no description found</li><li><a href="https://tenor.com/view/dog-in-space-dog-i-have-no-idea-i-have-no-idea-what-im-doing-gif-25502378">Dog In Space Dog GIF - Dog In Space Dog I Have No Idea - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gist.github.com/Nodja/2a97c530b8898affd8fd897a95595ee0">Letter level tokenization</a>: Letter level tokenization. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://stability.ai/news/license-update">Community License — Stability AI</a>: Our new Community License is now free for research, non-commercial, and commercial use. You only need a paid Enterprise license if your yearly revenues exceed USD$1M and you use Stability AI models in...</li></ul></div>

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1258681015606575116)** (2 messages):

> - `scammer alert`
> - `new tokenizer proposal for LLMs`
> - `T-FREE tokenizer paper`

- ****User flags a scammer****: A user alerted the community to the presence of a **scammer** in the chat.
- ****T-FREE Tokenizer Proposal Shakes Up LLMs****: A new paper proposes **T-FREE**, a tokenizer that embeds words through sparse activation patterns over character triplets, eliminating the need for a reference corpus and achieving a parameter reduction of more than **85%** in embedding layers. You can view the [paper here](https://arxiv.org/abs/2406.19223).
  - The paper outlines **T-FREE's** advantages, including improved performance for underrepresented languages and significant compression of embedding layers.

**Link mentioned**: [T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings](https://arxiv.org/abs/2406.19223): Tokenizers are crucial for encoding information in Large Language Models, but their development has recently stagnated, and they contain inherent weaknesses. Major limitations include computational ov...

---

### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/)** (1 messages):

khazn: $50 gift card [steamcommunity.com/gift/sd271azjxn2h](https://exi.link/EvuqQq)

---

### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/)** (1 messages):

khazn: $50 gift card [steamcommunity.com/gift/sd271azjxn2h](https://exi.link/EvuqQq)

---

### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/)** (1 messages):

khazn: $50 gift card [steamcommunity.com/gift/sd271azjxn2h](https://exi.link/EvuqQq)

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1258135295153537054)** (116 messages🔥🔥):

> - `Moshi AI demo`
> - `Issues with GPT-2`
> - `Voice modality in OpenAI models`
> - `Bangla language support in chatGPT`
> - `API usage for AI integration`

- \****Moshi AI Demo Excites and Frustrates**: A new [Moshi AI demo](https://moshi.chat/?queue_id=talktomoshi) was released, featuring real-time voice interaction and promises of open-source flexibility. However, users experienced issues like conversational interruptions and looped responses, highlighting the current model's limitations.*\*: A new [Moshi AI demo](https://moshi.chat/?queue_id=talktomoshi) was released, featuring real-time voice interaction and promises of open-source flexibility. However, users experienced issues like conversational interruptions and looped responses, highlighting the current model's limitations.
- \****Lack of Long-Term Memory in AI**: Hume AI's [playground](https://demo.hume.ai/) offers interruptable voice AI but lacks long-term memory functionality, resetting after each session. This limitation frustrates users who desire continuous learning from their AI interactions.*\*: Hume AI's [playground](https://demo.hume.ai/) offers interruptable voice AI but lacks long-term memory functionality, resetting after each session. This limitation frustrates users who desire continuous learning from their AI interactions.
- \****Call for Enhanced Bangla Language Support**: A user highlighted ongoing issues with chatGPT handling the Bangla language, urging improvements for better accessibility. The request was posted with a thread ID for specific reference and emphasizes the need for broader language support.*\*: A user highlighted ongoing issues with chatGPT handling the Bangla language, urging improvements for better accessibility. The request was posted with a thread ID for specific reference and emphasizes the need for broader language support.
- \****GPT-2 vs Modern Models Debate**: There was a discussion on whether to use the older GPT-2 model for text generation or upgrade to more current options like GPT-3.5 Turbo. While some argued for the cost-efficiency of GPT-2, others pointed out the drastically better performance of newer models.*\*: There was a discussion on whether to use the older GPT-2 model for text generation or upgrade to more current options like GPT-3.5 Turbo. While some argued for the cost-efficiency of GPT-2, others pointed out the drastically better performance of newer models.
- \****Navigating AI Integration via API**: Users discussed various methods for integrating AI models using APIs, particularly focusing on RAG via Assistant API endpoints. The conversation highlighted how crucial coding knowledge is for maximizing AI's utility and customization.*\*: Users discussed various methods for integrating AI models using APIs, particularly focusing on RAG via Assistant API endpoints. The conversation highlighted how crucial coding knowledge is for maximizing AI's utility and customization.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://demo.hume.ai/">Voice-to-Voice Demo •&nbsp;Hume AI</a>: Speak to the first empathic AI voice.</li><li><a href="https://moshi.chat/?queue_id=talktomoshi">moshi.chat</a>: no description found</li></ul></div>

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1258211085408796753)** (26 messages🔥):

> - `Differences between free and paid ChatGPT plans`
> - `Handling images and PDFs in GPT knowledge base`
> - `Effectiveness of GPT memory`
> - `Accessing other GPT models within a GPT`
> - `External file linking and vector databases for GPT knowledge base`

- **Paid ChatGPT Plan Benefits Explained**: A member asked about the benefits of a paid ChatGPT plan, and it was explained that **Plus** offers a higher usage cap, access to **DALL·E**, and a larger context window. Additional details can be found [here](https://openai.com/chatgpt/pricing).
- **Images and PDFs in GPT Knowledge Base**: Members discussed whether GPT uses vision to read images and handle PDFs uploaded to the knowledge section. The conclusion was that **GPT** does not use vision and relies on **OCR** for text extraction from images and PDFs.
- **GPT Memory Effectiveness Questioned**: A member criticized GPT's memory feature, noting it saves preferences but still makes things up. Another member clarified that these memories function as suggestions, not hard rules, and recommended using customization options to improve behavior.
- **Linking GPTs and Document Services**: A complex discussion unfolded around linking GPT knowledge bases to Google Drive and other similar services. **It was noted that external files cannot match the optimization of vector databases without a custom backend**, with some services offering live link support for similar features.
- **GPT-4 Usage Cooldown Confirmed**: Concerns about GPT-4 availability were addressed, explaining that users face a cooldown period before using GPT-4 again after hitting their limit. **Plus users can send up to 40 messages every 3 hours on GPT-4** and 80 on GPT-4o, with potential reductions during peak hours.

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1258154368767627295)** (16 messages🔥):

> - `Employee Recognition Program`
> - `Content Generation script for training courses`
> - `Tool to test multiple AI responses`
> - `Tabletop RPG prompts`
> - `Traffic ticket challenge guidance`

- **Employee Recognition Program Boosts Morale**: Users discussed developing an **employee recognition program** to boost morale and motivation. The program includes goals, recognition methods, criteria, an implementation plan, and feedback mechanisms.
- **Effective Content Generation Script**: One user is seeking advice on **developing a content generation script** to create training course structures based on inputs like location, length, topic, and audience. They are considering **prompt engineering, RAG, and web search integration** as potential techniques.
- **Tool for Testing Multiple AI Responses**: A user inquired about tools to **test and visualize multiple AI responses** from the same prompt, seeking features like supporting file uploads and displaying response variations. Suggestions included a **custom-built tool** or existing options like Autogen.
- **Tabletop RPG Battle Maps Prompting**: A user asked for **prompt ideas for generating tabletop RPG battle maps**. Specific tools or techniques were not discussed.
- **Guidance on Challenging Traffic Tickets**: The channel discussed a structured approach to **challenging a traffic ticket** in court. The guidance included steps for contesting the ticket effectively and strategies for presenting a case.

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages):

teknium: [https://x.com/kerstingaiml/status/1809152764649574541?s=46](https://x.com/kerstingaiml/status/1809152764649574541?s=46)

---

### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1258180446613668012)** (1 messages):

> - `Replete-AI releases two massive datasets`
> - `Everything_Instruct and Everything_Instruct_Multilingual datasets`
> - `Sizes and features of new datasets`
> - `Influence of bagel datasets and EveryoneLLM AI models`

- **Replete-AI Unveils Massive Datasets**: <@716121022025302076> released two new datasets, **Everything_Instruct** and **Everything_Instruct_Multilingual**, each sized 11-12GB and containing over 6 million rows of data. These are formatted in **Alpaca Instruct** style with a focus on creating a comprehensive instruct dataset to train AI models.
- **Dual Dataset for Ultimate AI Model Training**: **Everything_Instruct** is designed for English-only data, while **Everything_Instruct_Multilingual** includes multilingual translations to enhance models' language capabilities. Both datasets are inspired by the **bagel datasets** and previous **EveryoneLLM AI models**.
  - The goal is to combine all conceivable types of instruct data into one massive dataset to train top-notch AI models. Enjoy [the datasets on Hugging Face](https://huggingface.co/datasets/Replete-AI/Everything_Instruct).

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct">Replete-AI/Everything_Instruct · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual">Replete-AI/Everything_Instruct_Multilingual · Datasets at Hugging Face</a>: no description found</li></ul></div>

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1258278785409876009)** (4 messages):

> - `Upcoming Nous physical magazine contribution`
> - `Open-source / decentralized technology in StudioMilitary magazine`

- **Call for Contributions in Nous Physical Magazine**: **John0galt** invited everyone to contribute to the upcoming **Nous physical magazine** by offering good writing, interesting content, or ideas. **Reach out to John0galt** if interested.
- **StudioMilitary Magazine Seeking Contributions**: **StudioMilitary** has begun work on their first magazine edition **focusing on open-source and decentralized technology**. They are looking for contributions in writing, articles, pictures, and infographics, and have encouraged interested parties to [reach out](https://x.com/StudioMilitary/status/1807826564970848691).

**Link mentioned**: [Tweet from John Galt (@StudioMilitary)](https://x.com/StudioMilitary/status/1807826564970848691): I'm beginning work on the first edition of our magazine. General theme is open-source / decentralized technology. Highlighting the optimistic forces in our world. If you're interested in cont...

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1258565061295997020)** (5 messages):

> - `Apollo project by Achyut Benz`
> - `flask-socketio-llm-completions GitHub repo`
> - `foxhop's demo chatroom app`
> - `LLM integration with flask-socketio`

- ****Apollo project visualizes topics AI-generated in 3Blue1Brown style****: *Achyut Benz* introduced [Apollo](https://x.com/achyut_benz/status/1808969030969274507?s=46), which visualizes topics in **3Blue1Brown** style videos, all AI-generated. It uses the **Next.js framework**, **GroqInc inference**, and supports **AnthropicAI 3.5 Sonnet** & **OpenAI GPT-4** integrated with **LangChainAI**.
  - Inspired by Chris Abey, the project aims to enhance learning through **AI-generated** educational videos.
- ****Chatroom app sends messages to multiple LLMs via flask-socketio****: *foxhop* shared a [GitHub repo](https://github.com/russellballestrini/flask-socketio-llm-completions) for **flask-socketio-llm-completions**, a chatroom app that sends messages to **GPT**, **Claude**, **Mistral**, **Together**, and **Groq AI**, streaming to the frontend.
  - "This app is maintained to work seamlessly with various **LLMs** and demonstrates real-time communication capabilities."
- ****Foxhop showcases demo for LLM-integrated chatroom app****: *foxhop* provided a [demo link](http://home.foxhop.net:5001/chat/vllm-hermes-llama-3?username=changeme) to showcase the chatroom app integrated with **LLMs**. The demo exemplifies how messages interact with **vLLM**, **Hermes**, and **Llama3** models.
  - The application serves as a practical tool for interacting and experimenting with **LLM** capabilities in a chatroom environment.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/achyut_benz/status/1808969030969274507?s=46">Tweet from ach (@achyut_benz)</a>: introducing apollo, a new project i've been working on that visualizes topics or concepts in @3blue1brown style videos, all ai-generated. @nextjs framework @GroqInc inference supports @Anthropic...</li><li><a href="https://github.com/russellballestrini/flask-socketio-llm-completions">GitHub - russellballestrini/flask-socketio-llm-completions: Chatroom app where messages are sent to GPT, Claude, Mistral, Together, Groq AI and streamed to the frontend.</a>: Chatroom app where messages are sent to GPT, Claude, Mistral, Together, Groq AI and streamed to the frontend. - russellballestrini/flask-socketio-llm-completions</li></ul></div>

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1258153439620235345)** (110 messages🔥🔥):

> - `New datasets released by Replete-AI`
> - `Nomic AI launches GPT4ALL 3.0`
> - `InternLM-XComposer-2.5 model release`
> - `Challenges with jailbreaks for Claude 3.5 Sonnet`
> - `Discussion on visual latent space for LLMs`

- **Replete-AI Unveils Massive New Datasets**: Replete-AI releases two new massive datasets, **Everything_Instruct** and **Everything_Instruct_Multilingual**, each sizing 11-12GB with over 6 million rows of data, aiming to combine various instruct data to train AI models to new heights. [Details here](https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual).
  - These datasets, inspired by bagel datasets and Replete-AI's EveryoneLLM models, include one set for English and another with multilingual translations to enhance models' multilingual capabilities.
- **Nomic AI Launches GPT4ALL 3.0**: **Nomic AI** announces the release of **GPT4All 3.0**, an open-source local LLM desktop app supporting thousands of models across major operating systems with significant UI/UX improvements and MIT license. [Check it out](https://home.nomic.ai/gpt4all), boasting 250,000+ monthly active users and privacy-first features with local file chat.
- **InternLM-XComposer-2.5 Sets New Benchmarks**: InternLM releases **InternLM-XComposer-2.5**, a versatile large-vision language model supporting long-contextual input and output, trained with 24K interleaved image-text contexts and capable of extending to 96K long contexts via RoPE extrapolation. [Announcement here](https://x.com/_akhaliq/status/1808747694317261114?s=46), it surpasses existing open-source models on 16 benchmarks and competes closely with GPT-4V and Gemini Pro.
- **Frustrations with Jailbreaking Claude 3.5 Sonnet**: Users share challenges in jailbreaking **Claude 3.5 Sonnet**, discussing attempts with specific pre-prompts and roles, but the AI remains persistent on ethical constraints. Some suggest using Anthropic's workbench for potentially higher success but warn of possible account bans.
- **Exploring LLMs' Visual Latent Space Capabilities**: Discussions arise about letting **LLMs** draw or represent their visual latent space, considering if trained on enough visual data, they could repeat visual elements like chemical structures or 3D spaces. Some examples include a model generating a 3D city using HTML and CSS, suggesting potential but noting the need for datasets involving visual data.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/nomic_ai/status/1808162955806097767">Tweet from Nomic AI (@nomic_ai)</a>: Launching GPT4All 3.0: The Open-Source Local LLM Desktop App - Completely Private Experience - Supports 1000’s of models and all major operating systems - Major UI/UX Improvements - Local File Chat -...</li><li><a href="https://home.nomic.ai/gpt4all">GPT4All</a>: Run Large Language Models Locally: privacy-first and no internet required</li><li><a href="https://console.anthropic.com/workbench">Anthropic Console</a>: no description found</li><li><a href="https://x.com/localai_api/status/1808975139792425168?s=46">Tweet from LocalAI (@LocalAI_API)</a>: 🚀 New model alert! Check out #internlm2, a 7B parameter chat model with outstanding reasoning capabilities &amp; 1M context window. Install it in LocalAI with `local-ai run internlm2_5-7b-chat-1m` #AI #N...</li><li><a href="https://x.com/_akhaliq/status/1808747694317261114?s=46">Tweet from AK (@_akhaliq)</a>: InternLM-XComposer-2.5 A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output We present InternLM-XComposer-2.5 (IXC-2.5), a versatile large-vision language model that s...</li><li><a href="https://www.codedump.xyz/py/ZfkQmMk8I7ecLbIk**">no title found</a>: no description found</li><li><a href="https://x.com/_philschmid/status/1808755146190446667">Tweet from Philipp Schmid (@_philschmid)</a>: I wasn't aware of that, but it looks like Anthropic Claude 3.5 Sonnet on (claude ai) is suppressing parts of his answer from the user, which are not sent to the client. You can test that with, fro...</li><li><a href="https://pastebin.com/Gj7CpdSE">Karan4D's WorldSim System Prompt Open Source - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://github.com/russellballestrini/flask-socketio-llm-completions/pull/1">Keyword search across all chatrooms to find across conversation history by russellballestrini · Pull Request #1 · russellballestrini/flask-socketio-llm-completions</a>: Summary by CodeRabbit New Features Added a search functionality to find rooms and messages. Introduced a search results page to display search outcomes. Refactor Streamlined chat interface b...</li><li><a href="https://x.com/9mmballpoint/status/1808890582825120219">Tweet from RednBlackSalamander (@9mmballpoint)</a>: Art tools</li><li><a href="https://huggingface.co/internlm/internlm-xcomposer2d5-7b">internlm/internlm-xcomposer2d5-7b · Hugging Face</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/pull/5649#issuecomment-2209429032">Support Open Models that allow OpenAI API-style tool use &amp; "auto" tool choice by K-Mistele · Pull Request #5649 · vllm-project/vllm</a>: DRAFT: OpenAI Tool Use Checklist This (Draft) PR will add support for OpenAI-style tool calling in a way that is minimally opinionated about tool use formats &amp; prompt formatting. The following fea...</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">Three GPTs Walk into a Bar and Write an Exec Summary – D-Squared</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct_Multilingual">Replete-AI/Everything_Instruct_Multilingual · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct">Replete-AI/Everything_Instruct · Datasets at Hugging Face</a>: no description found</li></ul></div>

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1258183587006513254)** (1 messages):

> - `using visual-semantic information to boost image classification performance`
> - `zero/few shot multi-modal models discussed at CVPR`
> - `applying Florence 2 for supervised fine-tuning`

- **Boosting Image Classification with Visual-Semantic Info**: A user inquires about using the interaction between **visual-semantic information** to enhance fine-grained image classification performance, specifically through supervised fine-tuning. They mention a potential application of **Florence 2** for this purpose.
- **CVPR Highlights Zero/Few Shot Multi-modal Models**: At **CVPR**, numerous papers focused on zero/few shot multi-modal models, demonstrating interest in leveraging both visual and textual data. A user working in computer vision seeks advice on employing this research in practical, supervised settings.

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1258342665636806678)** (8 messages🔥):

> - `crossover with pipelines, flows, and agents`
> - `rag dataset as 0 shot context ingestion`
> - `context and metadata for llm`
> - `HF tool processing corpus queries against hf datasets`
> - `keyword matching for relevance score and filtering`

- ****Crossover with pipelines, flows, and agents****: **Pipelines, flows, and agents** are merging, and the idea is to make the RAG dataset primarily for **0 shot context ingestion**, focusing on agent-based processing later.
  - *interstellarninja* mentioned it's beneficial to incorporate cross-overs into agentic flows, even during RAG development.
- ****HF Tool Processing and Keyword Matching****: A **HF tool** was described that can process a corpus of queries against HF datasets, converting them into schemas with metadata as **.jsonl** files, utilizing an **inverted index for keyword matching**.
  - *@everyoneisgross* mentioned the interface allows for editing generations with Gradio, keyword search functions well for toy prompting.

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1258326477343035463)** (10 messages🔥):

> - `Users discussing lack of credits to use WorldSIM`
> - `Issues with using GPT-3.5 on WorldSIM`
> - `Prompt engineering for different models on WorldSIM`
> - `Positive feedback about WorldSIM`
> - `Buddhist world simulation on WorldSIM`

- **WorldSIM Users Run Out of Credits**: A user recommended explaining the credit limitations on **WorldSIM**, suggesting a heading like "Not enough credits to use" or using red text to indicate "NO CREDITS". This would help avoid confusion for new users.
- **Frustration with GPT-3.5 on WorldSIM**: Several members expressed frustration with using **GPT-3.5** on **WorldSIM**, mentioning that it often returns one-line answers before eventually working. One user complained about wasting credits on multiple messages to get started.
- **New Prompt Engineering for WorldSIM Models**: A discussion revealed that **WorldSIM** is working on new prompt engineering for different models. A member mentioned that separating the prompts between different models is a work in progress (**WIP**).
- **Members Praise WorldSIM**: A member stated that **WorldSIM** is "bonkers" and congratulated the team on an awesome job. Another member shared their experience of using up all their credits during a lunch hour to create a world rooted in Buddhist principles.

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1258788212323778652)** (1 messages):

> - `Simple Telegram bot to interface with different AI models`
> - `First 1000 responses free on the bot`

- ****Try Mysticella Bot for AI Model Interfacing****: Created a [simple Telegram bot](https://t.me/mysticella_bot) to interface with different AI models. **First 1000 responses** are free.
- ****Telegram Bot First 1000 Responses Free****: Check out the new Telegram bot **Mysticella** for free AI model interfacing. The **first 1000 responses** are completely free.

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1258136927434051644)** (107 messages🔥🔥):

> - `Quantisation of deployed LLM models in OpenRouter`
> - `Microsoft's API changes affecting OpenRouter`
> - `Infermatic's privacy policy update`
> - `Issues with DeepSeek Coder equations rendering`
> - `Mistral Codestral API pricing and performance`

- **LLM models quantization confusion clarified**: OpenRouter LLM models are deployed in **FP16/BF16** unless a provider specifies otherwise, as explained by a user. Another user clarified the presence of a **quantization icon** indicating model quantization status.
- **Microsoft API change impacts OpenRouter**: **Microsoft introduced a breaking change** to their API used by OpenRouter, but a patch was quickly deployed. User feedback praised the rapid response and fix.
- **Infermatic clarifies privacy policy**: **Infermatic does not log any input prompts or model outputs**, processing data in real-time only, as clarified in their revised [privacy policy](https://infermatic.ai/privacy-policy/). Users found this reassuring compared to older policies indicating potential data retention.
- **DeepSeek Coder equation issue resolved**: Users experienced issues with equations not rendering correctly in **DeepSeek Coder**, although one user found solutions by manipulating output strings with regex. Another user reported the system prompts not being processed correctly on TypingMind's frontend, raising the issue for review.
- **Mistral Codestral API pricing criticized**: Users expressed dissatisfaction with **Mistral's Codestral API** pricing, considering it overpriced for a 22B model. Alternative options like **DeepSeek Coder** were recommended for better cost efficiency and comparable coding performance.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - a Hugging Face Space by DontPlanToEnd</a>: no description found</li><li><a href="https://openrouter.ai/models/sao10k/l3-euryale-70b">Llama 3 Euryale 70B v2.1 by sao10k</a>: Euryale 70B v2.1 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k). - Better prompt adherence. - Better anatomy / spatial awareness. - Adapts much better to unique and c...</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: Set limits on model usage</li><li><a href="https://docs.mistral.ai/capabilities/code_generation/">Code generation | Mistral AI Large Language Models</a>: Codestral is a cutting-edge generative model that has been specifically designed and optimized for code generation tasks, including fill-in-the-middle and code completion. Codestral was trained on 80+...</li><li><a href="https://www.baseten.co/blog/llm-transformer-inference-guide/">A guide to LLM inference and performance</a>: Learn if LLM inference is compute or memory bound to fully utilize GPU power. Get insights on better GPU resource utilization.</li><li><a href="https://github.com/SillyTavern/SillyTavern/blob/release/src/prompt-converters.js#L86">SillyTavern/src/prompt-converters.js at release · SillyTavern/SillyTavern</a>: LLM Frontend for Power Users. Contribute to SillyTavern/SillyTavern development by creating an account on GitHub.</li><li><a href="https://web.archive.org/web/20240112082806/https://infermatic.ai/privacy-policy/">Privacy Policy - Infermatic</a>: no description found</li><li><a href="https://aistudio.google.com/app/prompts/new_chat?pli=1">no title found</a>: no description found</li><li><a href="https://infermatic.ai/privacy-policy/">Privacy Policy - Infermatic</a>: no description found</li><li><a href="http://llum.chat">lluminous</a>: no description found</li></ul></div>

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1258299318939549696)** (42 messages🔥):

> - `Failed jobs on the leaderboard`
> - `Checksum for generative models`
> - `Topological Data Analysis for model fingerprinting`
> - `1.58 bit LLM paper and its implementation`
> - `VQ-VAE immunity to posterior collapse`

- ****Leaderboard Job Issues Surface****: A member inquired about failed jobs on the [Hugging Face leaderboard](https://huggingface.co/datasets/open-llm-leaderboard/requests/blob/8c010a41f0b5f726199183bbad05f1649a362adf/cognitivecomputations/dolphin-2.9.2-qwen2-72b_eval_request_False_bfloat16_Original.json#L9) and whether they can be re-added.
- ****Debate on Checksums for Generative Models****: Discussion arose on whether there is a checksum-like metric for generative models like `LlamaForCausalLM` using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), with discrepancies noted between benchmarks and checksums.
- ****Exploring TDA for Model Fingerprinting****: Members delved into the use of Topological Data Analysis (TDA) to fingerprint models by measuring topological invariants, referencing tools like [TorchTDA](https://giotto-ai.github.io/gtda-docs/0.5.1/library.html).
  - *’Have you ever looked into Topological Data Analysis? You could potentially accomplish this by using TDA to profile the weights by their inherent topological invariants.*’
- ****Implementing 1.58-bit LLM Innovations****: A member sought guidance on adopting techniques from the [1.58-bit LLM paper](https://arxiv.org/abs/2402.17764) to quantize weights and activations for higher cost-efficiency.
  - They planned to replace the linear layers with a 'BitLinear' layer in a pre-trained model like Pythia to test quantized weight training.
- ****Struggles with PDF Markup Tools****: A member expressed frustration over the lack of PDF markup tools with a 'Search -> Markup All' function, mentioning expensive options like Bluebeam and PDF Studio.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://giotto-ai.github.io/gtda-docs/0.5.1/library.html">Overview — giotto-tda 0.5.1 documentation</a>: no description found</li><li><a href="https://huggingface.co/datasets/open-llm-leaderboard/requests/blob/8c010a41f0b5f726199183bbad05f1649a362adf/cognitivecomputations/dolphin-2.9.2-qwen2-72b_eval_request_False_bfloat16_Original.json#L9">cognitivecomputations/dolphin-2.9.2-qwen2-72b_eval_request_False_bfloat16_Original.json · open-llm-leaderboard/requests at 8c010a41f0b5f726199183bbad05f1649a362adf</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.18432">On the Origin of Llamas: Model Tree Heritage Recovery</a>: The rapid growth of neural network models shared on the internet has made model weights an important data modality. However, this information is underutilized as the weights are uninterpretable, and p...</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li></ul></div>

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1258310457001775146)** (28 messages🔥):

> - `Diffusion forcing for planning`
> - `Comparison with Nathan Frey's walk_jump method`
> - `Discussing new research strategies`
> - `Continual pre-training for LLMs`
> - `Function approximation with different homotopy classes`

- **Diffusion Forcing Shows Promise in Planning**: A member shared a [video](https://boyuan.space/diffusion-forcing/static/videos/planning/planning.mp4) demonstrating diffusion forcing for planning, generating a lot of interest and positive feedback, *'really cool result tbh'*.
- **Diffusion Forcing vs Walk-Jump Method**: Discussion on whether diffusion forcing would outperform Nathan Frey's [walk_jump method](https://www.youtube.com/watch?v=O3YBEnvvPZY) concluded that they may be orthogonal techniques with different mechanisms.
- **Effective Paper Consumption Strategy**: A member inquired about the strategy for keeping up with new research, receiving advice that skimming ArXiv papers on release and systematic effort in filtering important ones is key.
- **Continual Pre-training for Large Language Models**: Recent research on continual pre-training observed a **'stability gap'** in the performance of LLMs when adapting to new domains, and proposed [three strategies](https://arxiv.org/abs/2406.14833) to mitigate it.
- **Homotopy Classes in Function Approximation**: A member queried the benefit of having each basis function's image belong to different homotopy classes during function approximation, particularly in modeling rotation trajectories.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.youtube.com/watch?v=O3YBEnvvPZY">Protein Discovery with Discrete Walk-Jump Sampling | Nathan Frey</a>: Portal is the home of the AI for drug discovery community. Join for more details on this talk and to connect with the speakers: https://portal.valencelabs.co...</li><li><a href="http://arxiv.org/abs/2306.12360">Protein Discovery with Discrete Walk-Jump Sampling</a>: We resolve difficulties in training and sampling from a discrete generative model by learning a smoothed energy function, sampling from the smoothed data manifold with Langevin Markov chain Monte Carl...</li><li><a href="https://arxiv.org/abs/2406.14833">Efficient Continual Pre-training by Mitigating the Stability Gap</a>: Continual pre-training has increasingly become the predominant approach for adapting Large Language Models (LLMs) to new domains. This process involves updating the pre-trained LLM with a corpus from ...</li></ul></div>

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1258145861884575765)** (5 messages):

> - `efficientcube.ipynb in chinchilla repository`
> - `XLA capabilities in JAX`
> - `FLOPs estimation for JIT-ed functions in Flax`
> - `Critical batch size and performance degradation`

- ****EfficientCube Notebook in Chinchilla****: A [toolkit for scaling law research](https://github.com/kyo-takano/chinchilla/blob/master/examples/efficientcube.ipynb), named **efficientcube.ipynb**, has been added to the Chinchilla repository. The notebook includes utilities relevant for scaling research activities.
- ****JAX adds AOT Compilation Capabilities****: [JAX](https://jax.readthedocs.io/en/latest/aot.html#debug-information-and-analyses-when-available) now supports ahead-of-time (AOT) compilation in addition to JIT compilation. This allows users to compile code prior to execution, giving more control over the compilation process.
- ****Flax FLOPs Estimation Method Shared****: A code snippet for estimating **FLOPs of JIT-ed functions** in Flax was shared in a [discussion on GitHub](https://github.com/google/flax/discussions/1854#discussioncomment-4758695). This method leverages XLA’s capabilities within JAX for precise performance measurements.
- ****Reevaluation of Critical Batch Size Theory****: Recent findings suggest that below a certain **optimal batch size**, performance degrades, contradicting the **conventional wisdom** that any batch size below a critical value is good. This is noted as being interesting in theory but not significant at large scales.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://jax.readthedocs.io/en/latest/aot.html#debug-information-and-analyses-when-available">Ahead-of-time lowering and compilation — JAX documentation</a>: no description found</li><li><a href="https://github.com/google/flax/discussions/1854#discussioncomment-4758695">How do you access XLA's flop estimate for a jitted program? · google/flax · Discussion #1854</a>: For now, here is how you do it: In [1]: import jax, jax.numpy as jnp In [2]: m = jax.xla_computation(lambda x, y: x @ y)(jnp.ones((1000, 1000)), jnp.ones((1000,1000))).as_hlo_module() In [3]: clien...</li><li><a href="https://github.com/kyo-takano/chinchilla/blob/master/examples/efficientcube.ipynb">chinchilla/examples/efficientcube.ipynb at master · kyo-takano/chinchilla</a>: A toolkit for scaling law research ⚖. Contribute to kyo-takano/chinchilla development by creating an account on GitHub.</li></ul></div>

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1258144662800044155)** (3 messages):

> - `SAEs on Llama 3 8B`
> - `Sparse autoencoders`
> - `Residual stream processing`

- ****SAEs on Llama 3 8B trained****: [Sparse autoencoders (SAEs) trained on the residual stream of Llama 3 8B](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x) are now available for use. These SAEs employ the **RedPajama corpus** and can be loaded using the EleutherAI **`sae` library**.
  - *Downloads are not currently tracked for this model.*
- ****Residual stream processing using SAEs****: This project organizes **SAEs by layer** and integrates them with the Llama 3 8B model to process residual streams more effectively. For more details, consult the [model card](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x).

**Link mentioned**: [EleutherAI/sae-llama-3-8b-32x · Hugging Face](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x): no description found

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1258161481808351432)** (18 messages🔥):

> - `Preprocessing Function Optimization`
> - `Proof-Pile Config Error`
> - `Metric Inconsistencies in Config`
> - `Long Model Names Issue`
> - `Evaluating Model in Parallel`

- **Preprocessing Caching Alternative**: A user inquired if preprocessed questions/arguments could be saved before feeding them into the model, to avoid rerunning preprocessing functions every time.
- **Proof-Pile Config Error Resolution**: A user faced an error with the `proof-pile` task using a specific config file. Switching to `lambada_openai` worked, indicating a potential issue with the dataset itself.
- **Metric Mismatch Identified in Config**: There was confusion over using `loglikelihood_rolling` in the config while `loglikelihood` got called, likely due to metric inconsistencies. **loglikelihood metrics:** `perplexity` vs `word_perplexity`, `byte_perplexity`, `bits_per_byte`.
- **Long Model Names Cause Saving Issues**: A user experienced issues with saving due to long model names causing files and directories to not be written correctly. Errors returned **OSError(36, 'File name too long').**
- **Parallel Evaluation Setup Inquiry**: A user asked how to evaluate the model in a parallelized way while passing it via the `pretrained` parameter. **Warning received:** 'assuming single-process call to evaluate() or custom distributed integration'.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/GenerationVisualizer">Exploring model generations - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2010">Added MedConceptsQA Benchmark by Ofir408 · Pull Request #2010 · EleutherAI/lm-evaluation-harness</a>: Hi, I haved added our new benchmark called MedConceptsQA. MedConceptsQA is a dedicated open source benchmark for medical concepts question answering. The benchmark comprises of questions of various...</li></ul></div>

---

### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages):

wendlerc: Does anyone have a good SDXL latent downscaler? I’d like to go from 128x128x4 to 64x64x4.

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1258139491739238462)** (75 messages🔥🔥):

> - `Difficulty using LangChain`
> - `Preference between OpenAI or ChatOpenAI`
> - `PeopleGPT and Juicebox.ai functionality`
> - `RAG Architecture for scheduling demos`
> - `LangChain performance issues and improvements`

- ****Whys and Whys Nots of LangChain****: A member expressed difficulty using **LangChain** and questioned its utility, citing long response times and unnecessary steps in processing, especially while running locally on CPU.
  - Another user pointed out it might be the model's reasoning performance or simply the fact it's running without a GPU, leading to inefficiencies like excessive irrelevant searches.
- ****OpenAI vs. ChatOpenAI****: **OpenAI** and **ChatOpenAI** were compared for task executions, with a user inquiring the pros and cons and noting that **OpenAI** might be deprecated in favor of **ChatOpenAI**.
  - Several members clarified that diverse experiences exist, depending on the exact requirements and implementation contexts.
- ****PeopleGPT in Juicebox.ai Shines****: A member discussed **Juicebox.ai** powered by **PeopleGPT**, a natural language-based search engine for finding qualified talent without using Booleans, providing easy clickable examples [here](https://juicebox.ai/).
  - The discussion focused on the technical functionality, highlighting it combines filters with search to enhance user experience.
- ****Issues with LangChain and CSV Files****: A user sought updated methods for dealing with multiple CSV files in **LangChain**, noting previous limitations in handling more than two files post-update.
  - The member reminisced about the effectiveness of previous modules and queried modern alternatives for optimal performance and integration.
- ****Challenges with LangChain for Scheduling Demos****: A member struggled with incorporating demo scheduling features in a chatbot using LangChain and RAG architecture, mentioning tools like **SlackScheduleMessage**.
  - Detailed steps provided from LangChain's community were discussed for possible solutions, emphasizing the need for further community input.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://python.langchain.com/v0.2/docs/concepts/#messages">Conceptual guide | 🦜️🔗 LangChain</a>: This section contains introductions to key parts of LangChain.</li><li><a href="https://api.python.langchain.com/en/latest/_modules/langchain_core/messages/human.html#HumanMessage">langchain_core.messages.human — 🦜🔗 LangChain 0.2.6</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/23881">Agents and GraphQL- 401 Client Error: Unauthorized for url: https://streaming.bitquery.io/eap · Issue #23881 · langchain-ai/langchain</a>: Checked other resources I added a very descriptive title to this issue. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and di...</li><li><a href="https://api.python.langchain.com/en/latest/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html">langchain.agents.tool_calling_agent.base.create_tool_calling_agent — 🦜🔗 LangChain 0.2.6</a>: no description found</li><li><a href="https://juicebox.ai/">Juicebox (PeopleGPT) - The leader in AI-powered people search.</a>: Discover PeopleGPT, the search engine that know who you're looking for. Search through 800+ million profiles in real-time using natural language. Get contact details and set up outreach campaigns...</li><li><a href="https://x.com/levelsio/status/1804078191385956668">Tweet from @levelsio (@levelsio)</a>: I recommend everyone against using LangChain and this article explains well It uses abstractions on top of abstractions and actually makes your code needlessly complicated Just write API calls and a...</li><li><a href="https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents">Why we no longer use LangChain for building our AI agents</a>: When abstractions do more harm than good - lessons learned using LangChain in production and what we should’ve done instead</li><li><a href="https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/">Three GPTs Walk into a Bar and Write an Exec Summary – D-Squared</a>: no description found</li></ul></div>

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1258353115384774686)** (3 messages):

> - `Adding demo scheduling feature to chatbot using the RAG architecture and LangChain framework`
> - `Blogpost on creating an E2E Image Retrieval app using Lightly SSL and FAISS`
> - `Beta testing for advanced research assistant and search engine with premium model access`

- **RAG Chatbot Needs Demo Scheduling Feature**: A member asked for community help to add a **demo scheduling** feature to their chatbot built using the **RAG architecture** and the **LangChain framework**.
- ****Lightly SSL** and **FAISS** power Image Retrieval App**: A blogpost was shared on creating an **E2E Image Retrieval app** using **Lightly SSL** and **FAISS**, including implementing a vision transformer model and creating vector embeddings. The detailed blogpost includes a [Colab Notebook](https://colab.research.google.com/drive/1n4CwX5T6Ch2v7OYTRe6g1j_QJHxxOvcM) and a [Gradio app](https://huggingface.co/spaces/lightly-ai/food101-image-retrieval).
- ****Rubik's AI** offers Free Beta Testing**: An invitation was extended for beta testing an **advanced research assistant** and search engine, offering **2 months of free premium access** to models like Claude 3 Opus, GPT-4o, and more.
  - Users were prompted to [sign up](https://rubiks.ai/) using the promo code 'RUBIX' for the free trial.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant &amp; Search Engine</a>: no description found</li><li><a href="https://www.lightly.ai/post/vector-indexes-and-image-retrieval-using-lightly">Vector Indexes and Image Retrieval using&nbsp;lightly</a>: Use a pre-trained Vision Transformer provided by Lightly to create a vector index on an arbitrary dataset for Image Retrieval using faiss</li><li><a href="https://huggingface.co/spaces/lightly-ai/food101-image-retrieval">Food101 Image Retrieval - a Hugging Face Space by lightly-ai</a>: no description found</li><li><a href="https://x.com/MaheshkarSaurav/status/1808881869829853305">Tweet from Saurav Maheshkar ☕️ (@MaheshkarSaurav)</a>: 🚀 Latest work at @LightlyAI. Learn how you can create an Image Retrieval app using FAISS (@AIatMeta) as an vector index 🗃️, model implementations from the Lightly SSL package and @weights_biases for...</li><li><a href="https://colab.research.google.com/drive/1n4CwX5T6Ch2v7OYTRe6g1j_QJHxxOvcM?usp=sharing">Google Colab</a>: no description found</li></ul></div>

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages):

dievas_: [https://www.youtube.com/watch?v=yF9kGESAi3M](https://www.youtube.com/watch?v=yF9kGESAi3M) try this one

---

### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1258212529507663915)** (1 messages):

> - `Next webinar on RAG experimentation/evaluation with LlamaIndex and Weights and Biases`
> - `Announcements about the timing and focus of the upcoming webinar`
> - `Complex challenge of aligning LLM Judge for accurate evaluation`

- \***� Next Webinar on Aligning Your LLM Judge**: Join the next webinar on a principled approach to **RAG experimentation/evaluation** with [LlamaIndex and Weights and Biases](https://lu.ma/dywrdye5) next Wednesday at 9am PT. Reserve your spot by registering through the provided link.
- **Complex Challenge of Aligning Your LLM Judge**: This webinar will explore various **evaluation strategies** focused on aligning your LLM Judge using a **RAG pipeline** as a case study. It will also demonstrate how to leverage **Weights and Biases Weave** for systematic assessment.

**Link mentioned**: [LlamaIndex Webinar: Aligning Your LLM Judge with LlamaIndex and W&B Weave · Zoom · Luma](https://lu.ma/dywrdye5): While creating a RAG pipeline is now straightforward, aligning your LLM Judge for accurate evaluation remains a complex challenge. In this webinar, we’ll delve…

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1258148071141933167)** (4 messages):

> - `New Webinar: A Principled Approach to RAG Experimentation + Evaluation`
> - `Reflection as a Service`
> - `Becoming a Rockstar AI Engineer and Educator`
> - `Corrective RAG as a Service`

- **Webinar: Partnering with Weights & Biases on RAG**: LlamaIndex announced a [webinar](https://twitter.com/llama_index/status/1808589017744880062) with **Weights & Biases** to showcase building, evaluating, and iterating on RAG pipelines. This follows 1+ years of RAG development but notes that proper evaluation remains challenging.
- **Ensuring Reliability with Reflection as a Service**: LlamaIndex discussed the concept of 'Reflection as a Service,' addressing reliability issues in agentic applications by implementing a reflection step to self-correct outputs if incorrect. This solution aims to prevent problematic outputs from LLMs.
- **Rockstar AI Engineer: @ravithejads's Journey**: LlamaIndex highlighted the journey of community member @ravithejads, who became a **developer advocate** through passion, OSS contributions, and staying updated with the latest AI trends. His story is shared to inspire others to excel in AI engineering and education.
- **Releasing Corrective RAG as a Service**: LlamaIndex announced the release of [Corrective RAG (CRAG)](https://twitter.com/llama_index/status/1809282069606068486) by **Yan et al.**, which dynamically validates retrieved context and corrects it if irrelevant, using web search before the generation step.

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1258263851745546300)** (71 messages🔥🔥):

> - `Google Cloud Function inference pipeline with multiple model loading`
> - `Performance comparison of Cohere's command r+`
> - `Implementing conversational memory in LlamaIndex with RAG`
> - `Using hybrid retrievers without storing/loading from filesystem`
> - `Few-shot example technique for 'Poor man's RLHF'`

- ****Multiple Model Loading in Google Cloud Function Inference Pipeline****: A user expressed issues with loading the Alibaba NLP embedding model and Llama3 LLM for inferences on a Google Cloud Function, facing repetitive loading times. They asked for alternatives to load embeddings directly from Vertex AI and received suggestions but no concrete solution.
- ****Handling Conversational Memory in LlamaIndex****: A user sought ways to avoid overuse of conversation memory in LlamaIndex and received advice on improving prompt engineering to mitigate the issue. They agreed that modifying the system prompt might help.
- ****Hybrid Retriever Usage Without Filesystem Storage****: A user inquired about implementing a hybrid retriever without filesystem storage, and suggestions included writing the BM25 algorithm for sparse vectors and storing them in a vector store. Discussion also mentioned future explorations with bm42 and minor tweaks needed for LlamaIndex support.
- ****Handling Large Models and Quantization****: A user discussed challenges with using the 'gte-Qwen2-7B-instruct' and 'BAAI/bge-large-en-v1.5' embedding models due to GPU limitations. They planned to test quantized embedding models and learned both models can be used if dimensions match.
- ****Local LLMs, GPT4All, and Outdated Documentation****: Concerns were raised about outdated examples and links in the documentation. Latest information on using local LLMs was shared, and it was noted that contributions to update the documentation are welcome.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://Your_url_here',api_key=" ")"="">no title found</a>: no description found</li><li><a href="https://qdrant.tech/articles/bm42/#">BM42: New Baseline for Hybrid Search - Qdrant</a>: Introducing BM42 - a new sparse embedding approach, which combines the benefits of exact keyword search with the intelligence of transformers.</li><li><a href="https://x.com/mathemagic1an/status/1617606970114179072">Tweet from Jay Hack (@mathemagic1an)</a>: "Poor man's RLHF" 1) Have user indicate when model is correct 2) Store associated (input, output) in embedding index 3) At inference time, retrieve nearest K previous inputs 4) Put the...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_double_merging_chunking/">Semantic double merging chunking - LlamaIndex</a>: no description found</li><li><a href="https://github.com/microsoft/graphrag">GitHub - microsoft/graphrag: A modular graph-based Retrieval-Augmented Generation (RAG) system</a>: A modular graph-based Retrieval-Augmented Generation (RAG) system - microsoft/graphrag</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>: no description found</li><li><a href="https://colab.research.google.com/drive/16QMQePkONNlDpgiltOi7oRQgmB8dU5fl?usp=sharing#scrollTo=20cf0152">Google Colab</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/schema/#llama_index.core.schema.IndexNode.from_text_node>).">Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_context/">Chat Engine - Context Mode - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/retrievers/recursive_retriever_nodes/#chunk-references-smaller-child-chunks-referring-to-bigger-parent-chunk>).">Recursive Retriever + Node References - LlamaIndex</a>: no description found</li></ul></div>

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1258237090370551909)** (45 messages🔥):

> - `Discussion about qualifications for attending the London event`
> - `Issue with deploying an app using Cohere's rerank API in production`
> - `Introduction of new members`
> - `Teaching AI and advanced development`
> - `Working on AI-Plans, a peer review platform for red teaming alignment plans`

- ****No Qualification Needed for London Event****: A member asked if certain qualifications were necessary to attend the London event, and others clarified that **no prerequisite requirements** were needed and anyone could attend by filling out a form. *No PhD needed to attend community events* was a key message.
- ****Rerank API Error in Production****: A member raised a **TypeError** when deploying an app using the **rerank API** in production, contrasting its local functionality. Another member noted that the issue seems unrelated to Cohere and asked for the Streamlit script for further diagnosis.
- ****New Members Introduce Themselves****: Several new members, including a recent **Computer Science graduate** and an AI developer interested in teaching, introduced themselves and expressed excitement about joining the community. They highlighted their backgrounds and what they hope to achieve within the Discord.
- ****Teaching AI and Advanced Development****: A member expressed **keen interest in teaching AI and advanced development**, inviting others to reach out for collaboration. This was well-received, with another member openly offering to seek his expertise soon.
- ****AI-Plans Platform****: A member revealed working on **AI-Plans**, a peer review platform for **red teaming alignment plans**. This sparked interest and welcomed them to further discuss their project.

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1258377613630505061)** (17 messages🔥):

> - `Featuring a tutorial on Cohere blog`
> - `Introducing Command R+, the new powerful model`
> - `Using Rhea.run to create toy apps`
> - `New 'Save to Project' feature in Rhea.run`

- ****Feature Tutorial on Cohere Blog****: A member expressed interest in featuring a [tutorial on the Cohere blog](https://cohere.com/blog/build-a-smart-slack-bot-with-language-models) and shared an old blog post and starter code for a Slack bot on [GitHub](https://github.com/cohere-samples/cohere-slack-starter-app). Another member confirmed they will follow up directly.
- ****Using Rhea.run for Toy Apps****: Members discussed using [Rhea.run](https://rhea.run) to create toy apps, noting its capability to generate interactive applications by asking it to design HTML scripts.
- ****Introducing Command R+****: Cohere [announced the release of Command R+](https://cohere.com/blog/command-r-plus-microsoft-azure), their most powerful model in the Command R family, now available for use.
- ****New Feature in Rhea.run****: A new 'Save to Project' feature was introduced in [Rhea.run](https://rhea.run), which allows users to create interactive applications by designing HTML scripts through conversations.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://rhea.run">Rhea | Byte Breeze Studios</a>: no description found</li><li><a href="https://cohere.com/blog/build-a-smart-slack-bot-with-language-models">Build a smart Slack bot with language models</a>: Have you ever wanted to build an intelligent Slack bot? There are many ways to inject intelligence into a Slack or Discord bot. Starter Code: https://github.com/cohere-samples/cohere-slack-starter-ap...</li><li><a href="https://github.com/cohere-samples/cohere-slack-starter-app">GitHub - cohere-samples/cohere-slack-starter-app: Co:here-powered Slack App Starter Project</a>: Co:here-powered Slack App Starter Project. Contribute to cohere-samples/cohere-slack-starter-app development by creating an account on GitHub.</li></ul></div>

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1258170247358316655)** (57 messages🔥🔥):

> - `Technical question about interpreter output`
> - `Discussion on new MacOS Copilot, Invisibility`
> - `Acknowledgment and progress on Open Interpreter (OI) security`
> - `Open Interpreter's new debugging feature`
> - `Monthly House Party events`

- ****Invisibility: MacOS Copilot Gains Traction****: Members discussed the new [Invisibility MacOS Copilot](https://x.com/sulaimanghori/status/1791113392482377833) that uses GPT-4, Gemini 1.5 Pro, and Claude-3 Opus, highlighting its free availability and features like seamless context absorption. Development of voice, long term memory, and iOS is ongoing.
  - Interest was expressed about integrating similar tools into the OI ecosystem, with one member suggesting the possibility of open-sourcing grav.ai, a preceding project.
- ****Open Interpreter Implements Debug Command****: One user excitedly reported that Open Interpreter can now change the [VSC theme from light mode to dark mode](https://discord.com/invite/YQhmv5pd?event=1258399216078684242) automatically, showcasing its ability to perform certain actions without explicit programming. This feature, referred to as the 'wtf' command, allows for debugging errors in the terminal and suggesting fixes.
  - This newly implemented functionality caused quite a buzz, with members sharing their amazement and support for ongoing improvements.
- ****Acknowledgment of OI Security Measures****: A member praised the OI team for their dedication to security, mentioning a meeting where various ideas and suggestions were discussed to improve the system's security model. The team's commitment to making security a priority was highly appreciated.
  - Plans for future security roundtables were mentioned, with a promise to update the community on dates and ongoing efforts.
- ****Monthly House Party Recap****: The community celebrated the success of [OI’s 4th of July House Party](https://discord.com/invite/YQhmv5pd?event=1258399216078684242) which showcased new demos, faces, and previews of upcoming updates. The next event is scheduled for August 1st.
  - Members expressed their joy and gratitude for the event, highlighting its role in fostering engagement and collaboration within the community.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/sulaimanghori/status/1791113392482377833">Tweet from SKG (ceo @ piedpiper) (@sulaimanghori)</a>: So we've been cooking the last few weeks. Excited to finally unveil Invisibility: the dedicated MacOS Copilot. Powered by GPT4o, Gemini 1.5 Pro and Claude-3 Opus, now available for free -&gt; @inv...</li><li><a href="https://web.archive.org/web/20240418151656/https://grav.ai/">Gravity</a>: Your personal AI.</li></ul></div>

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1258565540935630912)** (2 messages):

> - `01 Light shipments update`
> - `Delays in 01 Light shipments`

- ****01 Light Shipments Update****: Members expressed anticipation about the **01 Light shipments** with one *hoping for an update soon*. Another member shared their frustration, stating they've been *waiting forever*.
- ****Frustration Over Shipment Delays****: A member conveyed their dissatisfaction over the prolonged wait for the **01 Light**. The sentiment was echoed by another member, indicating collective frustration.

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1258137941260238868)** (5 messages):

> - `Discussion on casting bugs in Mojo`
> - `Comparison between Mojo and Python objects`
> - `Proposal for a Mojo Fundamentals course at EDx`
> - `Resources for learning Mojo`

- ****Casting Bug in Mojo****: A member highlighted the casting bug with references to relevant GitHub issues [#3065](https://github.com/modularml/mojo/issues/3065) and [#3167](https://github.com/modularml/mojo/issues/3167).
- ****Mojo vs Python Objects Discussion****: There is speculation that the casting bug might be related to differences between **Mojo objects** and **Python objects**, referencing issue [#328](https://github.com/modularml/mojo/issues/328).
- ****Mojo Fundamentals Course Proposal****: A user proposed creating a "Mojo Fundamentals" course for EDx, but another member suggested it would become outdated quickly. They recommended using [Mojo by example](https://ruhati.net/mojo/) and [mojo-learning](https://github.com/rd4com/mojo-learning) as up-to-date resources instead.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://github.com/modularml/mojo/issues/328)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/3065)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/3167)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1258147458001928233)** (22 messages🔥):

> - `Casting file pointer to struct in Mojo`
> - `Calling external programs in Mojo using system or popen`
> - `Handling bitcast issues in Mojo with byte array manipulation`
> - `Pass a List as an argument to a function in Mojo`
> - `MLIR issue with unsigned integer casting in Mojo`

- **Casting file pointer to struct in Mojo**: A user successfully bitcasted a `List`'s `UnsafePointer` to a struct in Mojo using an example shared by another user, with specific reference to [bitcast](https://docs.modular.com/mojo/stdlib/memory/unsafe_pointer/UnsafePointer#bitcast).
- **MLIR unsigned integer casting bug reported**: **MLIR Issue #3065** was discussed where casting to unsigned integers behaves like casting to signed integers, creating inconsistencies. This issue has been affecting multiple users and the discussion moved from Discord to [GitHub Issue #3065](https://github.com/modularml/mojo/issues/3065).
- **External programs in Mojo**: **Running external programs** in Mojo can be achieved using `external_call` with references given to [example here](https://github.com/modularml/max/blob/main/examples/) for implementations like `system` and `popen`. A Python example for `popen` was shared, detailing how to run
- **Handling bitcast issues in Mojo with byte array manipulation**: A user encountered inconsistencies when bitcasting objects from a file pointer in Mojo, with behaviors changing based on byte array lookup. The issue was suspected to be due to the bytes getting freed, suggesting keeping the bytes around or using `Reference` to avoid undefined behavior.
- **Pass a List as an argument to a function in Mojo**: A user resolved an issue passing `List` as an argument by specifying the type in the function signature as `inout inList:List[String]`. They initially faced type errors but successfully appended items to the list following the fix.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://docs.modular.com/mojo/stdlib/memory/unsafe_pointer/UnsafePointer#bitcast).">UnsafePointer | Modular Docs</a>: This is a pointer type that can point to any generic value that is movable.</li><li><a href="https://github.com/modularml/mojo/issues/3065">[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo</a>: Bug description Migrating this here after a bit of discussion in Discord. It seems like casting to unsigned integers actually just casts to signed integers, but has different behaviour in different...</li><li><a href="https://github.com/modularml/max/blob/main/examples/">max/examples at main · modularml/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform - modularml/max</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1258188994445967470)** (10 messages🔥):

> - `segfault issues with nightly build`
> - `bug report submission`
> - `os.path.expanduser bug`
> - `new nightly Mojo compiler releases`
> - `changelog updates`

- ****Nightly Build Segfaults on Compilation****: A member experienced a segfault while compiling a source file with the nightly build and shared the [problematic file](https://github.com/Mojo-Numerics-and-Algorithms-group/MojoSci/blob/dynobs/src/diffeq/runga_kutta.mojo). This prompted them to submit a bug report.
- ****os.path.expanduser Bug Causes Nightly Build Failures****: A bug introduced by using `os.path.expanduser` caused nightly builds to fail because the `HOME` environment variable was not set during tests. A member admitted the mistake, apologizing for the inconvenience.
- ****New Nightly Mojo Compiler Released****: A new Mojo compiler version `2024.7.416` has been released, featuring updates like an `exclusive` parameter to pointer types and the implementation of `collections.Counter`. See the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) and [raw diff](https://github.com/modularml/mojo/compare/5b77a66cb42143ffbcf39db635964ae344e63d25...654d07945a5aff2c92e0877153ea5d4b4563dcb6) for detailed changes.
- ****Subsequent Nightly Mojo Compiler Release****: Another nightly compiler version `2024.7.505` was released, deprecating `time.now` in favor of `time.perf_counter` methods. Detailed changes are available in the [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) and [raw diff](https://github.com/modularml/mojo/compare/654d07945a5aff2c92e0877153ea5d4b4563dcb6...39d95f073592c59b5badeb9740600674540e1235).

---

### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1258318749841494069)** (17 messages🔥):

> - `Feedback from Modular staff on best answers`
> - `Interest in x86 and SVE rounds`
> - `PR for a better timer needing MLIR knowledge`
> - `Benny's solution for matrix multiplication`
> - `Compilation times and segfaults in test suite`

- **Modular staff to provide feedback on best answers**: Modular staff will give feedback on the best answer at the end of the challenge, as well as offer suggestions for improvement.
- **Interest in x86 and SVE benchmarks**: A discussion emerged about conducting x86 (with and without AMX) and SVE rounds since Graviton 4 is expected to go GA soon, and it features SVE.
- **Benny shares matrix multiplication solution and hints**: Benny shared his best solution for matrix multiplication and hinted at tuning the block size for improved performance. He mentioned using **CPU cache sizes** as parameters and suggested checking UT Austin papers for more details.
- **Compilation time and segfault issues in test suite**: Users reported long compilation times and internal segfault issues when running the latest test suite with provided solutions.
- **Relevant papers for parameter tuning**: Benny referenced several UT Austin papers for parameter tuning related to **cache sizes** and matrix multiplication performance improvements. He provided a [Google Spreadsheet link](https://docs.google.com/spreadsheets/d/1TBz9Lp0JT1Ph7ndfbWqp-B30FQcRYl1959hP2lZ6yH4/edit) listing those papers.

**Link mentioned**: [Matrix Multiplication](https://docs.google.com/spreadsheets/d/1TBz9Lp0JT1Ph7ndfbWqp-B30FQcRYl1959hP2lZ6yH4/edit): Sheet1 Contstraints,Parameters / Tuning Vectorization,Contiguous Access,Nelts, Unrollable Parallelization,Unrollable Unrolling,Contiguous Operations Tiling Square Optimized,Amorized Increase,Recursiv...

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1258529329957306378)** (12 messages🔥):

> - `Usage of LangSmith without LangChain`
> - `Accusation of lack of GPU credits during AI course`
> - `3rd place solution in AI Mathematical Olympiad`
> - `Benefits of in-context learning vs. fine-tuning`

- ****LangSmith Operates Independently from LangChain****: A user inquired if **LangSmith** can be used without **LangChain**, to which others confirmed that it's possible and provided a [Colab example](https://colab.research.google.com/github/langchain-ai/langsmith-cookbook/blob/main/tracing-examples/traceable/tracing_without_langchain.ipynb) and [GitHub link](https://github.com/langchain-ai/langsmith-cookbook/blob/main/tracing-examples/traceable/tracing_without_langchain.ipynb). **LangSmith** allows **instrumentation of any LLM application**, useful for debugging and monitoring.
- ****Accusations About Missing GPU Credits****: A heated debate ensued over claims that participants of a course did not receive **GPU credits**, with multiple members pointing out that the terms were clearly stated and visible on the course platform. Some speculated that the complaints might be unfounded or driven by ulterior motives.
- ****Top 3rd Place AI Mathematical Olympiad Solution’s Lack of Fine-Tuning****: A user highlighted that the **3rd place solution** in the AI Mathematical Olympiad, which won **$32k**, did not involve any fine-tuning. The leaderboard can be reviewed for more details [here](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/leaderboard).
- ****In-Context Learning vs Fine-Tuning Discussion****: An interesting discussion was sparked by a **LinkedIn post** comparing **in-context learning** with **fine-tuning** for LLMs. The detailed insights can be found [here](https://www.linkedin.com/posts/zainhas_should-you-finetune-your-llm-or-is-giving-activity-7215029375476383744-ZZ0K).

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/leaderboard">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>: no description found</li><li><a href="https://docs.smith.langchain.com/old/cookbook/tracing-examples/traceable">Tracing without LangChain | 🦜️🛠️ LangSmith</a>: Open In Collab Open In GitHub</li></ul></div>

---

### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1258498357983186997)** (7 messages):

> - `Discussion on monthly credits and expiration`
> - `Distributed finetuning issue solutions`
> - `Clarifying the usage and remaining balance of credits`

- ****Clarifying monthly credits and expiration****: Members discussed the **$1000 monthly credit** and potential loopholes, clarifying that **unused credits** may not carry over, but still finding it generous.
- ****Issues with distributed finetuning****: A member shared a [link to a thread](https://discord.com/channels/1238365980128706560/1247226177257734247) detailing steps to resolve issues encountered during **distributed finetuning**.
- ****Understanding credit usage and balance****: Discussion centered on members noticing their remaining balance, with one reporting **$1030** after finetuning Mistral, and questioning if it is due to a **default $30** per month allocation.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/)** (1 messages):

goktrenks: when is the expiration date for the credits? (thanks btw!)

---

### **LLM Finetuning (Hamel + Dan) ▷ #[ankurgoyal_textsql_llmevals](https://discord.com/channels/1238365980128706560/1242222674835538012/1258791747300102208)** (2 messages):

> - `Text2SQL use case discussion and appreciation for iterative eval dataset building`

- ****Iterative Building of Eval Dataset Impresses****: A member expressed appreciation for the session on **Text2SQL**, highlighting its value due to the **iterative building of the eval dataset**.
  - The iterative process was particularly appreciated and seen as beneficial for an upcoming use case.
- ****Thanks to the Community****: Members expressed gratitude towards the community, particularly towards an individual for their guidance in **building the eval dataset** for Text2SQL.
  - Such sessions and discussions are found **incredibly valuable** by the members.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1258434662552965223)** (1 messages):

> - `Applying eval framework to unstructured applications`
> - `Challenges of using unit tests/Level 1 evals without structured output`

- **Challenges in Eval Framework for Unstructured Output**: A user questioned the applicability of the **eval framework** to outputs that lack strict syntax rules, **like a query language**. They expressed confusion over implementing **unit tests/Level 1 evals** without a structured output.
- **Missing Methodology in Unstructured Eval Applications**: The user asked if they were *missing something* when considering how to apply the **eval framework** to less structured applications, indicating a gap in understanding or practice.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1258743017863380992)** (2 messages):

> - `Pushing models to HF_HUB for inference endpoints`
> - `Training models on HF_HUB as endpoints`

- **Push Models to HF_HUB for Inference**: **Inference endpoints** on **HF_HUB** might be facilitated by pushing a model to the hub and then using the credits for an endpoint. This suggestion revolves around utilizing existing resources for creating efficient inference pipelines.
- **Training Not Feasible as Endpoints**: The idea that **training will work as an endpoint** on HF_HUB is questionable. It's discussed that **training may not be practical for endpoints**, possibly due to resource or infrastructure limitations.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1258469044718272513)** (3 messages):

> - `Using type: input_output with Meta-Llama-3-8B-Instruct`
> - `Special tokens configuration in Axolotl`
> - `Training outcomes with L3 8B base vs L3 70B Instruct`
> - `Template usage for prompt formatting`
> - `Special tokens setup discrepancies between models`

- **Struggling with Meta-Llama-3-8B-Instruct setup**: A user shared challenges with using `type: input_output` and configuring `special_tokens` for the `Meta-Llama-3-8B-Instruct` model, citing confusion over correct setup in their jsonl and yml files. They referenced a [GitHub example](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/llama-3/instruct-lora-8b.yml) and a [blog post](https://hamel.dev/notes/llm/finetuning/09_template_free.html) for additional context.
- **Disparities in special tokens setup**: Discussion included the need to add special tokens from Meta's [special_tokens_map.json](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/special_tokens_map.json), comparing it to the special tokens setup for Mistral 7B base. They suggested following similar configurations as used in other training setups to avoid issues.
- **Training results favoring L3 70B Instruct**: A user noted better subjective outcomes training on L3 70B Instruct base compared to L3 8B base, discovering the improved results only after checking the model configuration post-training. They mentioned an accidental but preferable result when a training setup defaulted to the 70B instruct model.

**Link mentioned**: [Hamel’s Blog - Template-free axolotl](https://hamel.dev/notes/llm/finetuning/09_template_free.html): Template-free prompt construction in axolotl with the new input_output format.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1258798631255805982)** (1 messages):

> - `Eligibility for credits on all services`
> - `Enrollment date and course catch-up`

- ****Seeking Eligibility for Credits****: A member inquired about their **eligibility for credits** on all services and expressed gratitude for any applicable credits.
- ****Course Enrollment Date****: The same member mentioned they enrolled in the course on **June 14th** and have been catching up recently.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1258342772276989964)** (1 messages):

> - `Expired compute credits`
> - `Extension request for compute credits`

- ****Compute Credits Expire Too Soon****: A member realized that their **compute credits** have expired after only one month, leaving them with around **$70** still unused.
- ****Extension Request for Compute Credits****: The same member politely asked if it is possible to get an extension for the remaining **compute credits**.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1258803520342069429)** (1 messages):

> - `Credit grant request`
> - `Enrollment details`

- ****Credit grant request****: A user requested credit grants for their updated form with organization ID **org-SxGZTlTAAYP5xAswIojG7KI5**.
- ****Enrollment details****: The user mentioned they enrolled on **June 14th** and are catching up on the course lately.

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1258171932621148260)** (5 messages):

> - `Unimpressed reaction to AI demo`
> - `Stability AI's apology and license update`

- ****AI Demo Criticism Raises Authenticity Questions****: @benhylak expressed disappointment with an AI demo on [X](https://x.com/benhylak/status/1808611023123067357), questioning its authenticity by stating *'it's really, really bad... leaves me wondering if the demo was fake?'*. **Response time** issues were particularly noted.
- ****Stability AI Apologizes and Updates License****: Stability AI acknowledged that **Stable Diffusion 3 Medium** didn't meet community expectations and clarified [updates](https://x.com/stabilityai/status/1809274908641489160?s=46) on its commercial license, aiming to address confusion and concerns. They committed to releasing **high-quality Generative AI** models moving forward.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/stabilityai/status/1809274908641489160?s=46">Tweet from Stability AI (@StabilityAI)</a>: At Stability AI, we’re committed to releasing high-quality Generative AI models and sharing them generously with our community of innovators and media creators.&nbsp; We acknowledge that our latest releas...</li><li><a href="https://x.com/benhylak/status/1808611023123067357">Tweet from ben (@benhylak)</a>: just tried it and... it's really, really bad. leaves me wondering if the demo was fake? Quoting ben (@benhylak) the world is about to change very fast.</li></ul></div>

---

### **Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1258334337158610966)** (8 messages🔥):

> - `BM42 vs BM25 in search engines`
> - `Contextual AI's focus on RAG`
> - `Jo Bergum's critique of Qdrant's BM42 claims`

- **BM42 challenges BM25 in search tech**: **Qdrant Engine** claims that the new search model, **BM42**, surpasses the traditional **BM25** in modern RAG applications, offering a mix of semantic and keyword search as mentioned in their [tweet](https://x.com/qdrant_engine/status/1808498752107241949).
- **Jo Bergum- BM42 results are fake**: **Jo Bergum** criticized **Qdrant Engine** for falsifying results about BM42 on a Quora dataset, stating that Precision@10 reported was impossibly high, and calling the results \*\*

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/qdrant_engine/status/1808498752107241949">Tweet from Qdrant (@qdrant_engine)</a>: For 40 years, BM25 has been the standard for search engines. However, it falls short for modern RAG applications. Say hello to BM42: The combination of semantic and keyword search</li><li><a href="https://x.com/jobergum/status/1809157587612336402?s=46">Tweet from Jo Kristian Bergum (@jobergum)</a>: Okay, gloves off. What @qdrant_engine did with the BM42 post is unacceptable. They are misguiding the RAG community in a big way. 1) Presenting Quora as a relevant RAG question-answering dataset. I...</li></ul></div>

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1258300980542439425)** (9 messages🔥):

> - `Understanding VAEs`
> - `Interconnects' investment genius`
> - `GDP growth rate from AI for timelines`
> - `Anthropic Claude 3.5 Sonnet suppressing answers`

- **Understanding VAEs leads to nosebleeds**: *VAEs* (Variational Autoencoders) have caused confusion, with one user humorously noting they got a nosebleed trying to understand them.
- **Investment Genius in Interconnects**: In a recent post, it was revealed that **interconnects** showcased his prowess as an "absolute investment genius."
- **AI-driven GDP growth requires significant rates**: **GDP growth** from AI needs to be between **11-15%** to meet Stuart's timelines, depending on initial conditions. This metric was checked for feasibility and deemed reasonable.
- **Anthropic Claude 3.5 Sonnet suppressing answers**: [Anthropic Claude 3.5 Sonnet](https://fxtwitter.com/_philschmid/status/1808755146190446667) is reportedly suppressing parts of its answers from users. The usage of hidden tags like *§§antThinking§§* has raised concerns about **transparency** in these AI systems.

**Link mentioned**: [Tweet from Philipp Schmid (@_philschmid)](https://fxtwitter.com/_philschmid/status/1808755146190446667): I wasn't aware of that, but it looks like Anthropic Claude 3.5 Sonnet on (claude ai) is suppressing parts of his answer from the user, which are not sent to the client. You can test that with, fro...

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1258407157356040192)** (4 messages):

> - `Gemini web app`
> - `Google AI Studio`
> - `Vertex AI`
> - `Google's AI race`
> - `First Amendment and weights`

- **Google's AI race lags behind**: **Google** is behind other companies in the AI race, needing to clean up clarity issues that caused user confusion, according to a detailed discussion in the chat.
  - One participant expressed that Google **is slow and messy booting up in the AI race**, but acknowledged that they are improving.
- **Understanding Gemini and its offerings**: The **Gemini web app** costs **$20/mo** and competes with ChatGPT, previously named Bard and powered by **PaLM 2** before now using **Gemini 1.5**. **Google AI Studio** provides an API key for developers to use Gemini 1.5 with 2M context, while **Vertex AI** offers the same for enterprises.
  - *One user expressed confusion* about whether the paid version of Gemini always uses Gemini 1.5 due to unclear FAQs.
- **First Amendment and weights**: A user discussed the application of the **First Amendment** to AI model weights, suggesting it could be a logical but optimistic view.
  - *The idea is that weights* should be protected as something published, thereby covered by the **First Amendment**.

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1258511792901066842)** (20 messages🔥):

> - `Issues with build.nvidia API`
> - `Queue system for build.nvidia API`
> - `Script issues and resolutions`
> - `Pipeline using YAML examples`

- ****build.nvidia API has hiccups****: A member noted trouble with the **build.nvidia API**. Another pointed out the emergence of a **queue system** for handling requests.
  - In attempts to resolve the script issues, a member realized it worked again after a brief pause, suggesting intermittent reliability of the API.
- ****Pipeline accepts YAML inputs****: In a discussion on handling inputs, a member mentioned their **pipeline** employs **YAML examples** of conversations for few-shot learning. They clarified this when questioned about incorporating textbook data.

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1258454095292010547)** (1 messages):

> - `Gemma2 update fixing issues`
> - `Pinned version of transformers`
> - `CI catching problems`

- **Gemma2 Fixes Issues in Updates**: The update for **Gemma2** addressed previously encountered problems. Using the pinned version of transformers ensures these issues are avoided, thanks to our CI system detecting such problems.
- **CI Ensures Stability with Transformers**: **Pinned version of transformers** should sidestep issues, as continuous integration (CI) will catch potential problems. This guarantees a more stable development environment.

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/)** (1 messages):

le_mess: Need more VRAM 🙂

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1258154032288239779)** (3 messages):

> - `test for bug placement`
> - `issue reproduction`
> - `focused test case`
> - `PR management`

- ****Test Bug Placement Decision****: A user inquired about the best location for a bug test—either in **test_nn** or **test_ops**—and asked for advice on naming it.
  - The user confirmed understanding and delegated the task to someone else, indicating that they will handle it.
- ****Issue Reproduction and PR Management****: Another user suggested leaving the PR open, treating it as an issue with a reproduction step, and ensuring the fix includes a more focused test case.
  - *Final confirmation from the original user* implied they would handle the specifics.

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1258137471414308945)** (12 messages🔥):

> - `Contiguous Tensors in Tinygrad`
> - `Tinygrad Training Efficiency Concerns`
> - `Matrix Multiplication Blog Post`
> - `Using Pre-trained PyTorch Models with Tinygrad`

- **Tinygrad Contiguous Tensors Confuse Users**: There's a discussion about `Tensor.randn/randint` creating contiguous Tensors whereas `Tensor.full` and similar methods create non-contiguous ones, which contrasts with PyTorch behavior.
- **Optimize Tinygrad for Large Scale Training**: Members discuss the inefficiencies in Tinygrad for large-scale training, mentioning it as slow and not cost-effective. A suggestion to use BEAM search was made but it takes time.
- **Learn Matmul with an Informative Blog Post**: An engaging [blog post](https://salykova.github.io/matmul-cpu) about high-performance matrix multiplication on CPU is shared, demonstrating over 1 TFLOPS performance with easy-to-understand [code](https://github.com/salykova/matmul.c).
- **Run Inference on Tinygrad with PyTorch Models**: Inquiry about the best way to run inference with a pre-trained PyTorch model using Tinygrad. The answer provided points to the usage of `tinygrad.nn.state.torch_load`.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://gist.github.com/python273/0dc136fbc63559188ab279c07329e891">TinyJit vis WIP</a>: TinyJit vis WIP. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://salykova.github.io/matmul-cpu">Beating NumPy’s matrix multiplication in 150 lines of C code</a>: TL;DR The code from the tutorial is available at matmul.c. This blog post is the result of my attempt to implement high-performance matrix multiplication on CPU while keeping the code simple, portable...</li></ul></div>

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1258157010567037008)** (8 messages🔥):

> - `Setting evaluation parameters for Torchtune`
> - `Grad norm graph on wandb`
> - `Loss curve optimization in wandb`
> - `Learning rate adjustment impacts`
> - `Missing wandb logging metrics`

- ****Setting evaluation parameters for Torchtune****: A user inquired about how to set evaluation parameters in **Torchtune**, and another mentioned there should be a parameter for 'validation dataset' or something similar.
- ****Missing grad norm graph in wandb****: A user sought assistance on obtaining a **grad norm graph** in wandb, as it is a default graph in other tools like aoxotl.
- ****Loss curve optimization in wandb****: A user was advised to observe the shape of the **loss curve** for a downward trend and was provided an example with a [link](https://wandb.ai/salman-mohammadi/torchtune_codellama_testing/runs/zobzkhd3?nw=nwusersalmanmohammadi). They noted insufficient optimisation in their loss curve and the suggestion to increase the initial learning rate.
- ****Learning rate adjustment impacts****: After receiving feedback, a user increased the initial **learning rate** and altered several parameters to optimize their model but reported no significant improvement in the loss.
- ****Missing wandb logging metrics****: A user questioned the absence of **wandb logging** for evaluation loss and grad norm, indicating an issue with metric logging.

**Link mentioned**: [salman-mohammadi](https://wandb.ai/salman-mohammadi/torchtune_codellama_testing/runs/zobzkhd3?nw=nwusersalmanmohammadi).): Weights & Biases, developer tools for machine learning

---

### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1258176915458752653)** (5 messages):

> - `Investigating system robustness with Python and TypeScript`
> - `Challenges with automatic Docker installation of Convex local backend`

- ****Python & TypeScript face integration issues****: A member shared issues with integrating **Python** and **TypeScript**, specifically encountering bugs when launching **Convex** if Python wasn't pre-installed.
- ****Docker's Convex backend installation is tricky****: Another member discussed challenges in making the **Convex** local backend installation automated within **Docker**, mainly due to how the container folder was set up as a volume for ease of updates and access.

---

### **AI Stack Devs (Yoko Li) ▷ #[assets](https://discord.com/channels/1122748573000409160/1176906086368935966/1258699780851241060)** (1 messages):

> - `Collection of sprite sheets`
> - `Aesthetics and style matching with Cloudpunk`
> - `Largest tilemaps on itch.io`

- **Searching for sprite sheets to match Cloudpunk's aesthetics**: A member inquired about the source of a specific collection of **sprite sheets**, mentioning their purchases of several large **tilemaps on itch.io** that didn't quite match the **dark, futuristic, cyberpunk** aesthetics of [Cloudpunk](https://store.steampowered.com/app/746850/Cloudpunk/).
- **Matching aesthetics of purchased tilemaps**: The member is curious about where to obtain **spritesheets** that go well with the **Cloudpunk** aesthetic, as their current collections and purchases from **itch.io** fall short.

---

### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1258487391300358279)** (1 messages):

> - `Three GPTs Walk into a Bar and Write an Exec Summary blog post by dsquared70`
> - `Utilizing Custom GPTs for creating executive summaries`
> - `Processes for high-frequency, short turnaround executive summaries`

- ****Three GPTs Revolutionize Executive Summaries****: [Three GPTs Walk into a Bar and Write an Exec Summary](https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary) blog post introduces a simple process for rapid executive summary creation. **Three Custom GPTs** work together: one extracts insights, one crafts summaries, and a third revises the content.
- ****High-Frequency Executive Summary Tactics****: The blog details how these **Custom GPTs** address high-frequency and short turnaround needs when summarizing events, technology, or trends. Often tasked with tight deadlines, this process ensures quick yet meaningful summaries.

**Link mentioned**: [Three GPTs Walk into a Bar and Write an Exec Summary – D-Squared](https://www.dylandavis.net/2024/07/three-gpts-walk-into-a-bar-and-write-an-exec-summary/): no description found

---

### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1258458307484712980)** (2 messages):

> - `Magpie model available on HuggingFace Spaces`
> - `Generating preference data via HuggingFace Spaces`
> - `Duplicated model from davanstrien/magpie`
> - `User feedback on Magpie model performance`

- **Magpie model available on HuggingFace Spaces**: A Magpie model is now accessible on [HuggingFace Spaces](https://huggingface.co/spaces/sroecker/Elster), which has been duplicated from [davanstrien/magpie](https://huggingface.co/spaces/davanstrien/magpie).
  - *Doesn't work that well* yet, but the concept of generating preference data via HuggingFace Spaces is well-liked.
- **User feedback on Magpie model performance**: A user shared that the Magpie model doesn’t function effectively but appreciates the concept.

**Link mentioned**: [Magpie - a Hugging Face Space by sroecker](https://huggingface.co/spaces/sroecker/Elster): no description found

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1258246339028979722)** (2 messages):

> - `Claude hackathon collaboration`
> - `Kafka optimization webinar`

- ****Claude Hackathon Collaboration****: A member invited others to collaborate and build something cool for the [Claude hackathon](https://docs.anthropic.com/en/build-with-claude-contest/overview) ending next week.
- ****Optimize Kafka and Save Costs!****: Join a webinar on **July 18th at 4 PM IST** to learn best practices for [optimizing Kafka](https://www.meetup.com/futureofdata-bangalore/events/301849238/?notificationId=1389017441959817216), including scaling strategies and cost-saving techniques.
- ****Expert Speakers at Kafka Webinar****: The event will feature **Yaniv Ben Hemo** from Superstream and **Viktor Somogyi-Vass** from Cloudera, who will share their expertise on building scalable, cost-efficient Kafka environments.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://docs.anthropic.com/en/build-with-claude-contest/overview">no title found</a>: no description found</li><li><a href="https://www.meetup.com/futureofdata-bangalore/events/301849238/?notificationId=1389017441959817216">Optimizing Kafka for Cost-Efficiency: Best Practices and Strategies, Thu, Jul 18, 2024, 4:00 PM | Meetup</a>: **Event Title:** **Optimizing Kafka for Cost-Efficiency: Best Practices and Strategies** **Event Details:** Date: July 18th 2024 Time: 4:00 PM IST (Virtual Event) Join us</li></ul></div>

---

### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1258348238851932220)** (1 messages):

> - `Potential uses for embeddings`
> - `New job title 'Embeddings Engineer'`

- ****Embeddings Engineer finds more uses****: An individual stated they are discovering more **potential uses for embeddings** and joked about adopting the title *Embeddings Engineer*.
- ****New job title humor****: **Embeddings Engineer** was suggested humorously as a new job title due to the increasing number of uses for embeddings.
  - *I think I'll call myself Embeddings Engineer from now on* 😄

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
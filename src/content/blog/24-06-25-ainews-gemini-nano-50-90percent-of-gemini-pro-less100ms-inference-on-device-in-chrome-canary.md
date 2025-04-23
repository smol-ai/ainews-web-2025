---
id: 8e70e6f1-f2bd-4ccd-923a-0f2d00840700
title: >-
  Gemini Nano: 50-90% of Gemini Pro, <100ms inference, on device, in Chrome
  Canary
date: '2024-06-25T07:02:13.519492Z'
original_slug: ainews-gemini-nano-50-90-of-gemini-pro-100ms
description: >-
  Google Chrome Canary now includes Gemini Nano, a lightweight 4-bit quantized
  model accessible via a new prompt API, offering fast on-device performance
  with model weights available on HuggingFace. Meanwhile, Anthropic released
  Claude 3.5 Sonnet, a faster, cheaper, and free-to-try model that outperforms
  GPT-4o on some benchmarks and leads open models in coding tasks.
tags:
  - google
  - chrome
  - gemini
  - prompt-api
  - anthopic
  - claude-3-5
  - coding
---


<!-- buttondown-editor-mode: plaintext -->**window.ai.createTextSession() is all you need**

> AI News for 6/21/2024-6/24/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**415** channels, and **5896** messages) for you. Estimated reading time saved (at 200wpm): **660 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

The latest [Chrome Canary](https://www.google.com/intl/en_au/chrome/canary/) now has Gemini Nano in a feature flag:

- Prompt API for Gemini Nano [chrome://flags/#prompt-api-for-gemini-nano](chrome://flags/#prompt-api-for-gemini-nano)
- Optimization guide on device
[chrome://flags/#optimization-guide-on-device-model](chrome://flags/#prompt-api-for-gemini-nano)
- Navigate to [chrome://components/](chrome://components/) and look for Optimization Guide On Device Model; Check for update to start the download


You'll now have access to the model via the console: `http://window.ai.createTextSession()`

 ![image.png](https://assets.buttondown.email/images/35e5c81e-fb99-46d5-9996-c335a5b4aae9.png?w=960&fit=max) 

Nano 1 and 2, at a 4bit quantized 1.8B and 3.25B parameters has decent performance relative to Gemini Pro:

 ![image.png](https://assets.buttondown.email/images/8592d84c-0fa3-4fac-909d-ad06593c05a7.png?w=960&fit=max) 


and you should see [this live demo](https://x.com/mortenjust/status/1805190952358650251) of how fast it runs
 ![image.png](https://assets.buttondown.email/images/5dce2799-dc79-4533-b7eb-82c3724c0df7.png?w=960&fit=max) 

Lastly, [the base model and instruct-tuned model weights](https://x.com/reach_vb/status/1805226216997200145) have already been extracted and posted to HuggingFace.

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

**AI Model Releases and Benchmarks**

- **Anthropic Claude 3.5 Sonnet**: [@adcock_brett](https://twitter.com/adcock_brett/status/1804908080829726790) noted Anthropic launched Claude 3.5 Sonnet, an upgraded model that **bests GPT-4o across some benchmarks**. For devs, it's **2x the speed of Opus, while pricing comes in at 1/5 the cost** of Anthropic's previous top model. For consumers, it's **completely free to try**. [@lmsysorg](https://twitter.com/lmsysorg/status/1804967083358523559) reported Claude 3.5 Sonnet has climbed to **#4 in Coding Arena, nearing GPT-4-Turbo levels**. It's now the **top open model for coding**. It also ranks #11 in Hard Prompts and #20 in Overall generic questions.
- **DeepSeek-Coder-V2**: [@dair_ai](https://twitter.com/dair_ai/status/1804922107870036049) noted DeepSeek-Coder-V2 competes with closed-sourced models on code and math generation tasks. It achieves **90.2% on HumanEval and 75.7% on MATH**, higher than GPT-4-Turbo-0409 performance according to their report. Includes a **16B and 236B parameter model with 128K context length**.
- **GLM-0520**: [@lmsysorg](https://twitter.com/lmsysorg/status/1804967083358523559) reported GLM-0520 from Zhipu AI/Tsinghua impresses at **#9 in Coding and #11 Overall**. Chinese LLMs are getting more competitive than ever!
- **Nemotron 340B**: [@dl_weekly](https://twitter.com/dl_weekly/status/1804951560356503900) reported NVIDIA announced Nemotron-4 340B, a family of **open models that developers can use to generate synthetic data for training large language models**.

**AI Research Papers**

- **TextGrad**: [@dair_ai](https://twitter.com/dair_ai/status/1804922109543477361) noted TextGrad is a new framework for **automatic differentiation through backpropagation on textual feedback provided by an LLM**. This improves individual components and the natural language helps to optimize the computation graph.
- **PlanRAG**: [@dair_ai](https://twitter.com/dair_ai/status/1804922113985245385) reported PlanRAG enhances decision making with a new RAG technique called **iterative plan-then-RAG**. It involves two steps: 1) an LLM generates the plan for decision making by examining data schema and questions and 2) the retriever generates the queries for data analysis. The final step checks if a new plan for further analysis is needed and iterates on previous steps or makes a decision on the data.
- **Mitigating Memorization in LLMs**: [@dair_ai](https://twitter.com/dair_ai/status/1804922115637875086) noted this paper presents a modification of the next-token prediction objective called **goldfish loss to help mitigate the verbatim generation of memorized training data**.
- **Tree Search for Language Model Agents**: [@dair_ai](https://twitter.com/dair_ai/status/1804922123896713254) reported this paper proposes an **inference-time tree search algorithm for LM agents to perform exploration and enable multi-step reasoning**. It's tested on interactive web environments and applied to GPT-4o to significantly improve performance.

**AI Applications and Demos**

- **Wayve PRISM-1**: [@adcock_brett](https://twitter.com/adcock_brett/status/1804908105815212100) reported Wayve AI introduced PRISM-1, a **scene reconstruction model of 4D scenes (3D in space + time) from video data**. Breakthroughs like this will be crucial in the development of autonomous driving.
- **Runway Gen-3 Alpha**: [@adcock_brett](https://twitter.com/adcock_brett/status/1804908283334959538) noted Runway demoed Gen-3 Alpha, a new AI model that can **generate 10-second videos from text prompts and images**. These human characters are 100% AI-generated.
- **Hedra Character-1**: [@adcock_brett](https://twitter.com/adcock_brett/status/1804908305703227797) reported Hedra launched Character-1, a new foundation model that can **turn images into singing portrait videos**. The public preview web app can generate up to 30 seconds of expressive talking, singing, or rapping characters.
- **ElevenLabs Text/Video-to-Sound**: [@adcock_brett](https://twitter.com/adcock_brett/status/1804908328088227923) noted ElevenLabs launched a new **open-source text and video-to-sound effects app and API**. Devs can now build apps that generate sound effects based on text prompts or add sound to silent videos.

**Memes and Humor**

- **Gilded Frogs**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1804981653808275800) defined "Gilded Frogs" as frogs that have **amassed great wealth and adorn themselves with luxurious jewelry**, including gold chains, gem-encrusted bracelets, and rings, covering their skins with diamonds, rubies, and sapphires.
- **Llama.ttf**: [@osanseviero](https://twitter.com/osanseviero/status/1804883653085769960) noted Llama.ttf is a **font which is also an LLM**. TinyStories (15M) as a font 🤯 The font engine runs inference of the LLM. Local LLMs taken to an extreme.
- **VCs Funding GPT Wrapper Startups**: [@abacaj](https://twitter.com/abacaj/status/1804976343471284326) posted a meme image joking about **VCs funding GPT wrapper startups**.
- **Philosophers vs ML Researchers**: [@AmandaAskell](https://twitter.com/AmandaAskell/status/1804986384022966385) posted a meme image comparing the **number of papers published by philosophers vs ML researchers**.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**Stable Diffusion / AI Image Generation**

- **Pony Diffusion model impresses users**: In /r/StableDiffusion, users are [discovering the capabilities and creative potential of the Pony Diffusion model](https://www.reddit.com/r/StableDiffusion/comments/1dmejz5/now_i_get_why_people_like_pony_so_much/), finding it fun and refreshing to use. Some [admit to underestimating Pony's responsibility and prompt adherence](https://www.reddit.com/r/StableDiffusion/comments/1dmhmdz/turns_out_im_the_immature_one/). There are requests for [in-depth Pony tutorials](https://www.reddit.com/r/StableDiffusion/comments/1dmisnf/are_there_any_thorough_tutorials_for_pony/) to help produce desired family-friendly anime/manga style images while avoiding unintended NSFW generations.

- **New techniques and model updates**: Users are sharing [background replacement, re-lighting and compositing workflows in ComfyUI](https://www.reddit.com/r/StableDiffusion/comments/1dn748i/background_replacement_relighting_and_composit_in_comfyui/) and demonstrating the [use of the [SEP] token for multiple prompts in adetailer models](https://www.reddit.com/r/StableDiffusion/comments/1dmwqqb/did_you_know_you_can_use_sep_to_use_multiple_prompts_for_adetailer_just_make_sure_you_get_the_order_right/). The [SD.Next release announcement](https://www.reddit.com/r/StableDiffusion/comments/1dmox7j/sdnext_release_20240623/) highlights 10+ improvements like quantized T5 encoder support, PixArt-Sigma variants, HunyuanDiT 1.1, and efficiency upgrades for low VRAM GPUs. [sd-scripts now supports training Stable Diffusion 3 models](https://www.reddit.com/r/StableDiffusion/comments/1dmp2xx/sdscripts_finally_supports_to_train_sd3/).

- **Creative applications and model comparisons**: An exhibition at the Nikola Tesla Museum features [118 AI-assisted artworks created with Stable Diffusion](https://www.reddit.com/r/StableDiffusion/comments/1dmj35z/elementally_ai/), highlighting adoption outside the AI community. New LoRA models like [Aether Illustration](https://www.reddit.com/r/StableDiffusion/comments/1dmjk2x/aether_illustration_new_nordic_style_illustration/) for Nordic-style portraits and a [black-and-white illustration style for SDXL](https://www.reddit.com/r/StableDiffusion/comments/1dn74k4/black_white_illustration_style_lora_sdxl_civitai_link_in_comments/) are being released. A [comparison of various models on a "woman lying on grass" prompt](https://www.reddit.com/r/StableDiffusion/comments/1dmt3c4/woman_lying_on_grass_comparasion_sd3_vs_sdxl_vs_sdxl_turbo_vs_dreamshaper_xl_lighting_vs_juggernaut_x_vs_stable_cascade_vs_epicrealism_5_vs_sd_15_vs_midjourney_vs_dalle_3_vs_adobe_firefly/) sparks discussion on their relative performance.

- **Licensing discussions**: Users discovered the [initial Stable Cascade weights were released under an MIT license for about 4 days](https://www.reddit.com/r/StableDiffusion/comments/1dn6yjp/stable_cascade_weights_were_actually_mit_licensed/) before changing to a more restrictive one, suggesting potential for commercial use of the MIT-licensed version. This has led to people downloading that specific version.

**ChatGPT / AI Assistants**

- **AI-generated games impress users**: In /r/ChatGPT, Claude, an AI assistant, [created a playable 3D first-person shooter game within the chat interface](https://www.reddit.com/r/ChatGPT/comments/1dmejz5/claude_made_me_a_3d_firstperson_shooter_touchscreen_game_right_in_the_chat_interface_in_the_game_you_shoot_happy_emojis_at_sad_monsters_to_make_them_happy_by_the_way_the_ridiculous_idea_for_a_game_is_claudes/). The game, which involves shooting happy emojis at sad monsters, was Claude's own idea. This is seen as a groundbreaking moment, with AI now competing with beginner human game developers. Users appreciate Claude's cute and hopeful approach.

- **Model performance and benchmarks**: [Claude 3.5 Sonnet outperforms other models on various benchmarks like MMLU-Pro](https://www.reddit.com/r/ChatGPT/comments/1dmd0km/claude_35_sonnet_take_the_top_spot_on_mmlupro_plus_new_sonnet_35_benchmarks_that_recently_came_out/), according to recently released results.

- **Improving chatbots with knowledge integration**: In /r/singularity, a user is [surprised large AI companies haven't connected their chatbots to knowledge bases like Wikipedia or tools like WolframAlpha](https://www.reddit.com/r/singularity/comments/1dmnflb/chatgpt_can_be_much_better_easily_imo/) for improved accuracy on facts, math, physics, etc. They believe the underlying technology exists but needs integration, though language models may still face fundamental limitations.

---

# AI Discord Recap

> **SPECIAL NOTE**: As we do for fresh frontier model launches, we are displaying the differences in output from running the same prompts across Claude 3.5 and 3 and GPT4o.

## Claude 3 Sonnet

**1\. LLM Performance Benchmarking and Advancements**

- Meta's [Llama 3 model](https://lmsys.org/blog/2024-05-08-llama3/) has rapidly risen to the top of leaderboards like ChatbotArena, outperforming models like GPT-4-Turbo and Claude 3 Opus as mentioned [in this discussion](https://discord.com/channels/879548962464493619/879548962464493622/1253795430014259370).
- New models like [IBM's Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct) and [DeepSeek's DeepSeek-V2 236B model](https://huggingface.co/deepseek-ai/DeepSeek-V2) were discussed, with the latter [outperforming GPT-4 in some benchmarks](https://discord.com/channels/1053877538025386074/1149866623109439599/1253789370444550296).
- However, there was [skepticism around certain benchmarks](https://discord.com/channels/729741769192767510/755950983669874798/1254766616986783835) and calls for credible sources to set realistic evaluation standards.

**2\. Techniques for Efficient LLM Training and Inference**

- [DeepSpeed's ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/) was mentioned as [promising 4x reduced communication overhead](https://discord.com/channels/879548962464493619/897390579145637909/1253935410091003946) for large model training on GPUs.
- The [vAttention system](https://arxiv.org/abs/2405.04437) was discussed for [dynamically managing KV-cache for efficient inference](https://discord.com/channels/879548962464493619/897390579145637909/1253935410091003946) without PagedAttention.
- [QServe's W4A8KV4 quantization](https://arxiv.org/abs/2405.04532) was highlighted as a technique to [boost cloud LLM serving performance on GPUs](https://discord.com/channels/879548962464493619/897390579145637909/1253935410091003946).
- Techniques like [Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/) were mentioned for [exploring parallel token decoding to reduce inference latency](https://discord.com/channels/879548962464493619/897390579145637909/1253935410091003946).

**3\. Open-Source AI Frameworks and Community Efforts**

- The [Axolotl project](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) was discussed for [supporting diverse dataset formats](https://discord.com/channels/1104757954588196865/1104757955204743201/1253827044463083582) for instruction tuning and LLM pre-training.
- [LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) was mentioned as powering [a new course on building agentic RAG systems](https://discord.com/channels/1059199217496772688/1187460979064324127/1254141690714001508) with Andrew Ng.
- The open-source [RefuelLLM-2 model](https://huggingface.co/refuelai/Llama-3-Refueled) was introduced, claiming to be [the best for "unsexy data tasks"](https://discord.com/channels/1179035537009545276/1179035537529643040/1253788511379787947).
- Discussions around [Modular's Mojo](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo) highlighted its potential for Python integration and AI extensions like *bfloat16* as mentioned [here](https://discord.com/channels/1087530497313357884/1098713601386233997/1253826170680643594).

**4\. Multimodal AI and Generative Modeling**

- The [Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609) and [CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677) models were mentioned for their focus on chat interactions and coding abilities respectively, as discussed [here](https://discord.com/channels/1179127597926469703/1183121795247779910/1253786942328275014).
- The [Phi 3 model](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/) was highlighted for bringing powerful AI chatbots to browsers via WebGPU, according to [this Reddit link](https://discord.com/channels/1110598183144399058/1110598183144399061/1253787640977817662).
- Combining Pixart Sigma, SDXL and PAG was mentioned as aiming to achieve DALLE-3 level outputs, with potential for further refinement through fine-tuning, as per [this discussion](https://discord.com/channels/1179127597926469703/1183121795247779910/1253786942328275014).
- The open-source [IC-Light project](https://github.com/lllyasviel/IC-Light) focused on improving image relighting techniques was also brought up in [this conversation](https://discord.com/channels/1179127597926469703/1183121795247779910/1253786942328275014).


## Claude 3.5 Sonnet

1.  **LLM Performance and Benchmarking Debates**:
  
  - New models like [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) and [Hermes 2 Theta Llama-3 70B](https://huggingface.co/OpenPipe/Hermes-2-Theta-Llama-3-70B-32k) are generating buzz for their performance. However, there's growing skepticism across communities about AI benchmarks and leaderboards, with calls for more credible evaluation methods.
  - Discussions in multiple discords touched on the challenges of comparing models like GPT-4, Claude, and newer open-source alternatives, highlighting the complexity of real-world performance versus benchmark scores.

2.  **Optimizing LLM Training and Inference**:
  
  - Communities are sharing techniques for improving LLM efficiency, such as [quantization methods](https://github.com/ggerganov/llama.cpp/pull/7931) and optimizing for specific hardware like [AMD GPUs](https://github.com/ROCm/xformers).
  - There's significant interest in reducing computational costs, with discussions ranging from [VRAM optimization](https://huggingface.co/mradermacher/Hermes-2-Theta-Llama-3-70B-32k-i1-GGUF/tree/main) to novel architectures for more efficient inference.

3.  **Open-Source AI Development and Collaboration**:
  
  - Projects like [Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) and [LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) are fostering community-driven development of AI tools and frameworks.
  - There's a trend towards open-sourcing models and tools, as seen with [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) and [Llamafile](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.7), encouraging wider participation in AI development.

4.  **Multimodal AI and Creative Applications**:
  
  - Discussions across discords highlight the growing interest in multimodal models that can handle text, image, and potentially video, with projects like [Stable Artisan](https://bit.ly/4aiVy6C) bringing these capabilities to wider audiences.
  - There's ongoing experimentation with combining different models and techniques to achieve DALL-E 3-level outputs, showing a community-driven approach to advancing generative AI capabilities.

5.  **AI Integration and Practical Applications**:
  
  - Multiple communities are exploring ways to integrate AI into everyday tools, from [browser-based models](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/) to Discord bots for media creation.
  - There's a growing focus on making AI more accessible and useful for specific tasks, as seen in discussions about code generation, data analysis, and creative applications across various discord channels.

## Claude 3 Opus

1.  **LLM Performance and Benchmarking**:
  
  - Discussions on the performance of models like [Llama 3](https://lmsys.org/blog/2024-05-08-llama3/), [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2), and [Claude 3.5 Sonnet](https://www.anthropic.com/index/claude-3-5-sonnet) across various benchmarks and leaderboards.
  - Skepticism surrounding certain benchmarks, such as [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/), with calls for more credible assessment standards.

2.  **Optimizing LLM Training and Inference**:
  
  - Techniques for efficient training, such as [ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/) and [Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/), and optimized inference with [vAttention](https://arxiv.org/abs/2405.04437) and [QServe](https://arxiv.org/abs/2405.04532).
  - Discussions on quantization methods, like [W4A8KV4](https://arxiv.org/abs/2405.04532), and their impact on model performance and resource requirements.

3.  **Open-Source AI Frameworks and Collaborations**:
  
  - Updates and collaborations involving open-source frameworks like [Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/), [LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex), and [Modular](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo).
  - Introducing new open-source models, such as [RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled) and [Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct).

4.  **Multimodal AI and Generative Models**:
  
  - Advancements in multimodal AI with models like [Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609) and [CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677).
  - Innovations in generative modeling, such as [Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/) for browser-based chatbots and combining techniques to achieve [DALLE-3](https://openai.com/dall-e-3/)\-level outputs.
  - Open-source efforts in image relighting with projects like [IC-Light](https://github.com/lllyasviel/IC-Light).

5.  **AI Ethics, Legality, and Accountability**:
  
  - Discussions on the ethical implications of AI-generated content, as seen with [Perplexity AI's alleged plagiarism](https://www.wired.com/story/perplexity-plagiarized-our-story-about-how-perplexity-is-a-bullshit-machine/).
  - Concerns about the legal risks associated with AI models making inaccurate or defamatory statements, as highlighted in the [Perplexity AI case](http://archive.today/GNgAe).
  - Debates on the accountability of tech companies using open datasets and the practice of ["AI data laundering"](https://waxy.org/2022/09/ai-data-laundering-how-academic-and-nonprofit-researchers-shield-tech-companies-from-accountability/).

## GPT4T (gpt-4-turbo-2024-04-09)

\*\*1. **AI Hardware Evolves but Costs Spiral**:

- VRAM requirements for AI models like **Command R (34b) Q4_K_S** lead to discussions about switching to **EXL2**, a more VRAM-efficient format. The [NVIDIA DGX GH200](https://www.nvidia.com/en-gb/data-center/dgx-gh200/) remains out of reach due to high costs.

\*\*2. **Optimization Takes Center Stage in AI Tools**:

- **Quantization techniques** are leveraged to optimize model performance, with ROCm's versions of [xformers](https://github.com/ROCm/xformers) and [flash-attention](https://github.com/ROCm/flash-attention) mentioned for efficiency. Implementation of **PyTorch enhancements** in the **Llama-2** model results in significant performance boosts.

\*\*3. **AI Breaks New Ground in Multifaceted Applications**:

- AI-powered reading aid **Pebble**, developed with **OpenRouter**, [Pebble](https://pebble.study/), illustrates novel uses of AI in enhancing reading comprehension. **Cohere's APIs** enable multi-step tool uses, significantly broadening functional reach, aided by detailed documentation on [multi-step tool use](https://docs.cohere.com/docs/multi-step-tool-use).

\*\*4. **AI-centric Education and Jobs Expanding**:

- **MJCET** launches the first AWS Cloud Club in Telangana to educate students on AWS technologies, celebrating with an event featuring **AWS Community Hero Mr. Faizal Khan**. RSVP through [event link](https://meetu.ps/e/NgmgX/14DgQ2/i). Also, AI-inspired educational content is proffered through Hamel Dan's **LLM101n course**, teaching LLM building from scratch.

\*\*5. **Miscellaneous AI Developments Reflect Wide Impact**:

- **Intel** retracts from AWS, puzzling the AI community on resource allocations. **Claude Sonnet 3.5's** prowess in coding tasks garners praise, showcasing AI's advancement in technical applications. In bizarre turns, **Mozilla's Llamafile** ventures into Android territory promising innovative mobile solutions.

## GPT4O (gpt-4o-2024-05-13)
### **Model Optimization and LLM Innovations**

- **DeepSeek and Sonnet 3.5 Dominate Benchmarks**: The **DeepSeek** model impressed the community with its quick performance and coding abilities, surpassing **GPT-4** in some cases ([DeepSeek announcement](https://chat.deepseek.com/coder)). Similarly, **Claude 3.5 Sonnet** outperformed **GPT-4o** in coding tasks, validated through LMSYS leaderboard positions and hands-on usage ([Claude thread](https://fxtwitter.com/RobertHaisfield/status/1804945938936668413)).
- **ZeRO++ and PyTorch Accelerate LLM Training**: **ZeRO++** reduces communication overhead in large model training by 4x, while new PyTorch techniques accelerate Llama-2 inference by 10x, encapsulated in the [GPTFast package](https://github.com/MDK8888/GPTFast), optimizing its use on A100 or H100 GPUs ([ZeRO++ tutorial](https://www.deepspeed.ai/tutorials/zeropp/)).

### **Open-Source Developments and Community Efforts**

- **Axolotl and Modular Encourage Community Contributions**: Axolotl announced the integration of ROCm fork versions of [xformers](https://github.com/ROCm/xformers) for AMD GPU support, and Modular users discussed contributing to learning materials for LLVM and CUTLASS ([related guide](https://pikuma.com/blog/understanding-computer-cache)).
- **Featherless.ai and LlamaIndex Expand Capabilities**: Featherless.ai, a new platform to run public models serverlessly, was launched to wide curiosity ([Featherless](https://featherless.ai)). **LlamaIndex** now supports image generation via StabilityAI, enhancing its toolkit for AI developers ([LlamaIndex-StabilityAI](https://t.co/a7F0gv4tpi)).

### **AI in Production and Real-World Applications**

- **MJCET's AWS Cloud Club Takes Off**: The inauguration of the AWS Cloud Club at MJCET promoted hands-on AWS training and career-building initiatives ([AWS event](https://meetu.ps/e/NgmgX/14DgQ2/i)).
- **Use of OpenRouter in Practical Applications**: **JojoAI** was highlighted for its proactive assistant capabilities, using integrations like DigiCord to outshine competitive models like ChatGPT and Claude ([JojoAI site](https://www.digicord.site)).

### **Operational Challenges and Support Queries**

- **Installation and Compatibility Issues Plague Users**: Difficulties in setting up libraries like xformers on Windows raised compatibility discussions, with suggestions converging on Linux for more stable operations ([Unsloth troubleshooting](https://github.com/unslothai/unsloth/issues/243)).
- **Credit and Support Issues**: Numerous members of the Hugging Face and Predibase communities faced issues with missing service credits and billing inquiries, showcasing the need for improved customer support systems ([Predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1253822951149797488)).

### **Upcoming Technologies and Future Directions**

- **Announcing New AI Models and Clusters**: **AI21's Jamba-Instruct** with a 256K context window and **NVIDIA's Nemotron 4** highlighted breakthroughs in handling large-scale enterprise documents ([Jamba-Instruct](https://openrouter.ai/models/ai21/jamba-instruct), [Nemotron-4](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct)).
- **Multi Fusion and Quantization Techniques**: Discussions on the merits of early versus later fusion in multimodal models and advancements in quantization highlighted ongoing research in reducing AI model inference cost and boosting efficiency ([Multi Fusion](https://arxiv.org/abs/2406.09406)).

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Juggernaut or SD3 Turbo for Virtual Realities?**: While *Juggernaut Lightning* is favored for its realism in non-coding creative scenarios, *SD3 Turbo* wasn't discussed as favorably, suggesting that choices between models are influenced by specific context and goals.

**Quantum Leap for PyTorch Users**: Investments in libraries like *PyTorch* and *HuggingFace* are recommended over dated ones like *sklearn*, and use of *bitsandbytes* and precision modifications such as 4-bit quantization can assist with model loading on constrained hardware.

**Meta-Model Mergers and Empathic Evolutions**: The *Open Empathic* project is expanding with contributed movie scene categories via YouTube, while merging tactics for *UltraChat* and *Mistral-Yarn* elicited debate, with references to *mergekit* and *frankenMoE finetuning* as noteworthy techniques for improving AI models.

**Souped-Up Software and Services**: A suite of contributions surfaced, including *Mistroll 7B v2.2*'s release, simple finetuning utilities for *Stable Diffusion*, a media-to-text conversion GUI using *PyQt* and *Whisper*, and the new AI platform [Featherless.ai](https://featherless.ai) for serverless model usage.

**In Pursuit of AI Reasoning Revelations**: Plans to unravel recent works on reasoning with LLMs are brewing, with *Understanding the Current State of Reasoning with LLMs* ([arXiv link](https://arxiv.org/abs/2206.07682)) and repositories like [Awesome-LLM-Reasoning](https://github.com/atfortes/Awesome-LLM-Reasoning) and its namesake [alternative repository link](https://github.com/luban-agi/Awesome-LLM-reasoning) earmarked for examination.

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI Previews Generate Buzz**: A member's anticipation for **Unsloth AI's** release led to the sharing of a [temporary recording](https://discord.com/channels/1179035537009545276/1179035537529643040/1253726564500111381), as theywaited for early access after a video filming announcement. Thumbnail updates, such as changing "csv -> unsloth + ollama" to "csv -> unsloth -> ollama", were suggested for clarity, alongside adding explainer text for newcomers.
- **Big VRAM Brings Bigger Conversations**: A [YouTube video](https://youtu.be/L4Bmrk2QprE?si=x-iFJrVRcK9-MQ8t&t=679) showcased the PCIe-NVMe card by **Phison** as an astonishing 1Tb VRAM solution, sparking discussions about its impact on performance. Meanwhile, Fimbulvntr's success in extending **Llama-3-70b** to a 64k context and the debate on VRAM expansion highlighted the ongoing exploration of large model capacities.
- **Upgrades and Emotions in LLMs**: Monday or Tuesday earmarked the **Ollama** update, promising CSV file support, while Sebastien's **emotional llama model**, fostering a better understanding of emotions in AI, became available on [Ollama](https://ollama.com/sebdg/emotional_llama) and [YouTube](https://www.youtube.com/watch?v=ZJKglSWgD0w).
- **Solving Setups & Compatibility**: From struggles to install xformers on Windows with Unsloth via conda to ensuring correct execution of initial setup cells in Google Colab notebooks, members swapped tips for overcoming software challenges. GPU Cloud (NGC) container setup discussions, as well as CUDA and PyTorch version constraints, featured solutions like using different containers and sharing Dockerfile configurations.
- **Pondering on Partnerships & AI Integration**: A blog titled "[Apple and Meta Partnership: The Future of Generative AI in iPhones](https://ghost-x.org/blog/apple-and-meta-partnership-the-future-of-generative-ai-in-iphones/)" stirred the guild's interest, with discussions focused on the strategic implications and potential integration challenges of generative AI in mobile devices.

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Bot Beware**: A Discord bot was shared for integrating Gemini and StabilityAI services, but members raised safety and context concerns regarding the link.
- **Civitai Pulls SD3 Amidst License Concerns**: The removal of SD3 resources by Civitai sparked intense discussions, suggesting the step was taken to preempt legal issues.
- **Running Stable with Less**: Techniques for operating Stable Diffusion on lower specification GPUs, like utilizing *automatic1111*, were debated, weighing the efficiency of older GPUs against newer models like the **RTX 4080**.
- **Training Troubles and Tips**: Community members sought advice for training models and overcoming errors such as VRAM limits and problematic metadata, with some suggesting specialized tools like **ComfyUI** and **OneTrainer** for enhanced management.
- **Model Compatibility Confusion**: Discussions highlighted the necessity for alignment between models like SD 1.5 and SDXL with add-ons such as ControlNet; mismatched types can lead to performance degradation and errors.

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUTLASS and CUDA Collaboration Call**: Users expressed interest in forming a **CUTLASS** working group, encouraged by a shared [YouTube talk on Tensor Cores](https://youtu.be/hQ9GPnV0-50?feature=shared). Additionally, insights on the CPU cache were amplified with a shared [primer on cache functionality](https://pikuma.com/blog/understanding-computer-cache), highlighting its significance for programmers.
- **Floating Points and Precision Perils**: Precision loss in FP8 conversion drew attention, prompting a shared resource for understanding rounding per IEEE convention and the use of tensor scaling to counteract loss. For those exploring **quantization**, a compilation of papers and educational content was recommended, including [Quantization explained](https://youtu.be/0VdNflU08yA) and [Advanced Quantization](https://youtu.be/1u9xUK3G4VM).
- **Enthusiasts of INT4 and QLoRA Weigh In**: In a discussion contrasting **INT4 LoRA fine-tuning** versus **QLoRA**, it was noted that QLoRA's inclusion of a CUDA dequant kernel (axis=0) sustains both quality and pace, especially compared to solutions using tinnygemm for large sequences.
- **Networks Need Nurturing**: The integration of **Bitnet tensors** with **AffineQuantizedTensor** sparked debate, considering special layouts for specifying packed dimensions. For assistance with debugging Bitnet tensor issues, [CoffeeVampire3's GitHub](https://github.com/CoffeeVampir3/ao-bitnet/blob/main/bitnet_staging/bitnet_trained_to_ao_test.py) and the [PyTorch ao library tutorials](https://github.com/pytorch/ao/issues/426) were spotlighted as go-to resources.
- **Strategies to Scale System Stability**: Strategies for **multi-node setup optimizations** and integrating **FP8 matmuls** were at the forefront of conversations, addressing performance challenges and training stability, especially on **H100 GPUs** which showed issues compared to **A100**. Upcoming large language model training on a **Lambda cluster** was also prepped for, with an eye on efficiency and stability.

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**VRAM Crunch and Hefty Price Tags**: Engineers highlighted the VRAM bottleneck when handling colossal models like **Command R (34b) Q4_K_S**, suggesting **EXL2** as a more VRAM-efficient format. For heavy-duty AI work, the [NVIDIA DGX GH200](https://www.nvidia.com/en-gb/data-center/dgx-gh200/), touted for its mammoth memory, remains out of reach financially for most, hinting at thousands of dollars in investment.

**Quantum Leaps in LLM Reasoning**: Users were impressed with the **Hermes 2 Theta Llama-3 70B** model, known for its significant token context limit and creative strengths. Conversations around LLMs lack temporal awareness spurred mention of the **Hathor Fractionate-L3-8B** for its performance when output tensors and embeddings remain unquantized.

**Cool Rigs and Hot Chips**: On the hardware battlefield, using **P40 GPUs** with **Codestral** demonstrated a surge in power utilization to 12 tokens/second. Meanwhile, the **iPad Pro’s** 16GB RAM was debated for its ability to handle AI models, and the dream of using **DX or Vulkan** for multi-GPU support in AI was floated in response to the absence of NVlink in 4000 series GPUs.

**Patchwork and Plugins**: The **LLaMa library** vexed users with errors stemming from a model's expected tensor count mismatch, whereas **deepseekV2** faced loading woes, potentially fixable by updating to **V0.2.25**. Enthusiasm bubbled for a hypothetical all-in-one model runner that could handle a gamut of Huggingface models including text-to-speech and text-to-image.

**Model Engineering and Enigmas**: The quaintly named **Llama 3 CursedStock V1.8-8B** model piqued curiosity for its unique performance, especially in creative content generation. There was chatter about a **Multi-model sequence map** allowing data flow among several models, and the latest quantized **Qwen2 500M** model made waves for its ability to operate on less capable rigs, even a **Raspberry Pi**.

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Siri and ChatGPT's Odd Couple**: There's confusion among users about Siri's integration with ChatGPT, with the consensus being that ChatGPT acts as an enhancement to Siri rather than a core integration. Elon Musk's critical comments fueled further discussion on the topic.
- **Claude's Coding Coup Over GPT-4o**: The **Claude 3.5 Sonnet** is praised for its superior performance in coding tasks compared to **GPT-4o**, with users highlighting Claude's success in areas where GPT-4o stumbled. Effectiveness is gauged by both practical usage and positions on the LMSYS leaderboard rather than just benchmark scores.
- **Persistent LLM Personal Assistant Dreaming**: Enthusiasm is noted regarding the possibility of tailoring and maintaining language models, like **Sonnet 3.5 or Gemini 1.5 Pro**, to serve as personalized work-bots trained on an individual's documents, prompting discussions about long-term and specialized applications of LLMs.
- **GPT-4o’s Context Window Woes**: Users struggle with limitations in **GPT-4o**'s ability to adhere to complex prompt instructions and handle lengthy documents. Alternatives such as Gemini and Claude are suggested for better performance with larger token windows.
- **DALL-E Vs. Midjourney Artistic Showdown**: A debate is unfolding on the server over DALL-E 3 and Midjourney’s capacities for generating AI images, particularly in the realm of paint-like artworks, with some showing a preference for the former's distinct artistic styles.

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI Caught in Plagiarism Uproar**: [Wired reported](https://www.wired.com/story/perplexity-plagiarized-our-story-about-how-perplexity-is-a-bullshit-machine/) Perplexity AI's alleged policy violations by scraping websites, with its chatbot misattributing a crime to a police officer and a debate emerging on the legal implications of inaccurate AI summaries.
- **Mixed Reactions to Claude 3.5 Sonnet**: The release of **Claude 3.5 Sonnet** was met with both applause for its capabilities and frustration for seeming overcautious, as reported by [Forbes](https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/), while users experienced inconsistencies with Pro search results leading to dissatisfaction with Perplexity's service.
- **Exclusives on Apple and Boeing's Struggles**: Apple's AI faced limitations in Europe while Boeing's Starliner confronted significant challenges, information disseminated on Perplexity with direct links to articles on these issues ([Apple Intelligence Isn't](https://www.perplexity.ai/page/Apple-Intelligence-Isnt-KJfiVRPEQMmkim0gv7Xh7w), [Boeing’s Starliner Stuck](https://www.perplexity.ai/page/Boeings-Starliner-Stuck-lIlR4mleQUK1Q0kahpVwRQ)).
- **Perplexity API Quandaries**: The Perplexity API community discussed issues like potential moderation triggers or technical errors with **LLama-3-70B** when handling long token sequences, and queries about restricting link summarization and time filtration in citations via the API were raised as documented in the [API reference](https://docs.perplexity.ai/reference/post_chat_completions).
- **Community Convergence for Better Engagement**: An OpenAI community message highlighted the need for shareable threads to foster greater collaboration, while a Perplexity AI-authored [YouTube video](https://www.youtube.com/embed/xUsxGDrwzls) previews diverse topics like **Starliner dilemmas and OpenAI's latest moves** for educational consumption.

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

### **Boost in Dataset Deduplication**: **Rensa** outperforms **datasketch** with a [2.5-3x speed boost](https://github.com/beowolx/rensa), leveraging Rust's FxHash, LSH index, and on-the-fly permutations for dataset deduplication.

### **Model Jailbreak Exposed**: A [Financial Times article](https://on.ft.com/45ByjEj) highlights hackers "jailbreaking" AI models to reveal flaws, while contributors on GitHub share a ["smol q\* implementation"](https://github.com/EveryOneIsGross/ganymede) and innovative projects like [llama.ttf](https://fuglede.github.io/llama.ttf/), an LLM inference engine disguised as a font file.

### **Lively Debate on Model Parameters**: In the *[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1254053586997088310)*, discussions ranged from the surprisingly capable story generation of [TinyStories-656K](https://huggingface.co/raincandy-u/TinyStories-656K) to assertions that general-purpose performance soars with 70B+ parameter models.

### **Dataset Synthesis and Classification Enhanced**: Members share a [Google Sheet](https://docs.google.com/spreadsheets/d/1f5fbPxhjGrmPqhbM0exOCX2vAz_CREYkzsdG2NSf-KY/edit#gid=0) for collaborative dataset tracking, explore improvements using the Hermes RAG format, and delve into datasets like [SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF?row=0) and [ft-instruction-synthesizer-collection](https://huggingface.co/datasets/instruction-pretrain/ft-instruction-synthesizer-collection/tree/main) for scientific and instructional purposes.

### **AI Safety Models Scrutiny and Coursework**: *#[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1253789370444550296)* sees a mix, from **Gemini** and **OpenAI**'s redaction-capable safety models to the launch of Karpathy's [LLM101n course](https://github.com/karpathy/LLM101n), encouraging engineers to build a storytelling LLM.

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SLURM Hiccups with Jupyter**: Engineers are facing issues with SLURM-managed nodes when connecting via Jupyter Notebook, citing errors potentially due to SLURM restrictions. A user experienced a 'kill' message on console before training even with correct GPU specifications.
- **PyTorch Boosts Llama-2 Performance**: PyTorch's team has implemented techniques to accelerate the Llama-2 inference speed by up to a factor of ten; the enhancements are encapsulated in the [GPTFast package](https://github.com/MDK8888/GPTFast), which requires A100 or H100 GPUs.
- **Ethics and Sharing of AI Models**: A serious conversation about the ethical and practical considerations of distributing proprietary AI models such as Mistral outside official sources highlighted concerns for legalities and the importance of transparency.
- **Understanding AI Model Variants**: Users debate methods to determine if an AI model is GPT-4 or a different variant, including examining knowledge cutoffs, latency disparities, and network traffic analysis.
- **LingOly Challenge Introduces**: A new LingOly benchmark is addressing the evaluation of LLMs in advanced reasoning involving linguistic puzzles. With over a thousand problems presented, top models are achieving below 50% accuracy, indicating a robust challenge for current architectures.
- **Text-to-Speech Innovation with ARDiT**: A [podcast episode](https://youtu.be/lj2y5hE04XI?t=4585) explores the usage of SAEs for model editing, inspired by the approach detailed in the [MEMIT paper](https://arxiv.org/pdf/2210.07229.pdf) and its [source code](https://github.com/kmeng01/memit), suggesting wide applications for this technology.
- **Pondering the Optimality of Multimodal Architectures**: Dialogue surfaced about whether an early fusion model, like Chameleon, stands superior to later fusion approaches for multimodal tasks. The trade-off between generalizability and visual acuity loss in the image tokenization process of early fusion was a focus.
- **Intel Retreats from AWS Instance**: Intel is discontinuing their AWS instance leveraged by the gpt-neox development team, prompting discussions on cost-effective or alternative manual solutions for computational resources.
- **Execution Error: NCCL Backend**: Engineers report persistent NCCL backend challenges while attempting to train models with gpt-neox on A100 GPUs, a problem consistent across various NCCL and CUDA versions, with Docker use or without.

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Character.AI Cracks Inference at Scale**: Noam Shazeer of Character.AI illuminates the pursuit of AGI through [optimization of inference processes](https://research.character.ai/optimizing-inference/), emphasizing their capability to handle upwards of 20,000 inference queries every second.
- **Acquisition News: OpenAI Welcomes Rockset**: [OpenAI has acquired Rockset](https://x.com/deedydas/status/1804185430897889427), a company skilled in hybrid search architecture with solutions like vector (FAISS) and keyword search, strengthening OpenAI's RAG suite.
- **AI Education boost by Karpathy**: Andrej Karpathy plants the seeds of an ambitious new course, "LLM101n," which will deep dive into constructing ChatGPT-like models from ground up, following the legacy of the legendary CS231n.
- **LangChain Clears the Air on Funds**: Harrison Chase addresses scrutiny regarding LangChain's expenditure of venture capital on product development instead of promotions, with a response detailed in a [tweet](https://x.com/hwchase17/status/1804166140773691837).
- **Murati Teases GPT's Next Leap**: Mira Murati of OpenAI teases enthusiasts with a timeline hinting at a possible release of the next GPT model in about 1.5 years, while discussing the sweeping changes AI is bringing into creative and productive industries, available in a [YouTube video](https://www.youtube.com/watch?v=yUoj9B8OpR8).
- **Latent Space Scholarship on Hiring AI Pros**: A new "Latent Space Podcast" episode breaks down the art and science of hiring AI engineers, guiding listeners through hiring processes and defensive AI engineering strategies, with insights from @james_elicit and @*adamwiggins* available on [this page](https://x.com/latentspacepod/status/1804269727482810386) and gathering buzz on Hacker News.
- **Embarking on new YAML Frontiers**: Conversations illustrate developing a YAML-based DSL for Twitter management to enhance post analytics, with a nod to Zoho Social's comprehensive features; for similar ventures, Anthropics suggests employing XML tags, and a [GitHub repo](https://github.com/go-go-golems/go-emrichen) showcases the successful design of a YAML templating language with LLMs in Go.

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **LLVM's Price Tag**: An article estimating the cost of the LLVM project was shared, detailing that 1.2k developers produced a codebase of 6.9M lines with an estimated cost of $530 million. Cloning and checking out LLVM is part of understanding its development costs.
- **Installation Troubles and Request for Help**: Issues with Mojo installation on 22.04 were highlighted, citing failures in all devrel-extras tests; a problematic situation that led to a pause for troubleshooting. Separately, frustration over segmentation faults during Mojo development prompted a user to offer a $10 OpenAI API key for help with their critical issue.
- **Discussions on Caching and Prefetching Performance**: Deep dives into caching and prefetching, with emphasis on correct application and pitfalls, were a significant conversation topic. Insights shared included the potential for adverse effects on performance if prefetching is incorrectly utilized, and recommendations to utilize profiling tools such as `vtune` for Intel caches, even though Mojo does not support compile-time cache size retrieval.
- **Improvement Proposals and Nightly Mojo Builds**: Suggested improvements for Mojo's documentation and a proposal for controlled implicit conversion in Mojo were noted. Updates on new nightly Mojo compiler releases as well as MAX repo updates sparked discussions on developmental workflow and productivity.
- **Data Labeling and Integration Insights**: A new data labeling platform initiative received feedback about common pain points and successes in automation with tools like [Haystack](https://haystack.deepset.ai/). The potential for ERP integration (prompted by manual data entry challenges and PDF processing) was also a focal point, indicating a push towards streamlining workflows in data management.

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **New Gates Open at Weta & Stability AI**: A wave of discussions followed news of leadership changes at **Weta Digital** and **Stability AI**, focusing on the implications of these shake-ups and questioning the motives behind the appointments. Some talks pointed to **Sean Parker** and shared articles on the subject, linking a Reuters article [Reuters article on Stability AI](https://www.reuters.com/technology/artificial-intelligence/stability-ai-appoints-new-ceo-information-reports-2024-06-21/).
- **Llama 3 on the Prowl**: There was palpable excitement about the **Llama 3** hardware specifications suggesting impressive performance, potentially outclassing rival models like **GPT-4O** and **Claude 3**. Participants shared projected throughputs of "1 to 2 tokens per second" on advanced setups.
- **The Protection Paradox with Glaze & Nightshade**: A sobering conversation unfolded over the limited ability of programs like **Glaze** and **Nightshade** to protect artists' rights. Skeptics noted that second movers often find ways around such protections, thus providing artists with potentially false hope.
- **Multimodal Models – A Repetitive Breakthrough?**: The guild examined a new paper on multimodal models, raising the question of whether the purported advancements were meaningful. The paper promotes training on a variety of modalities to enhance versatility, yet participants critiqued the repeated 'breakthrough' narrative with little substantial novelty.
- **Testing Limits: Promises and Limitations of Diffusion Models**: A deeper dive into diffusion models was encapsulated in a **GitHub repository** shared by lucidrains, discussing the **EMA (Exponential Moving Average)** model updates ([Diffusion Models on GitHub](https://github.com/lucidrains/ema-pytorch)) and their use in image restoration, despite evidence pointing to the consistent bypassing of protections like Glaze.

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Welcome Wagon for Newcomers**: New members joined the Cohere-focused Discord, guided by shared insights and [tool use documentation](https://docs.cohere.com/docs/tool-use) that helps connect Cohere models to external applications.
- **Skepticism Surrounding BitNet Practicality**: Amidst debates on BitNet's future, it's noted to require training from scratch and is not optimized for existing hardware, leading Mr. Dragonfox to express concerns about its commercial impracticality.
- **Cohere Capacities and Contributions**: Following the integration of a Cohere client in Microsoft's [AutoGen framework](https://github.com/microsoft/autogen/pull/3004/files), there was a call within the community for further support from the Cohere team in the project's advancement.
- **AI Enthusiasts Eager for Multilingual Expansions**: Cohere's model's ability to understand and respond in multiple languages, including Chinese, was confirmed, directing interested parties to [documentation](https://docs.cohere.com/docs/tool-use) and a GitHub [notebook example](https://github.com/cohere-ai/notebooks/blob/main/notebooks/agents/Vanilla_Tool_Use.ipynb) to learn more.
- **Developer Office Hours and Multi-Step Innovations**: Cohere announced upcoming developer office hours emphasizing the **Command R family's tool use capabilities**, providing resources on [multi-step tool use](https://docs.cohere.com/docs/multi-step-tool-use) for leveraging models to execute complex sequences of tasks.

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Confusion Over Context and Tokens**: Users reported confusion regarding the integration of **max tokens** and context windows in agents, specifically with LangChain not adhering to **Pydantic** models' validations. It was noted that context window or max token counts should include both the input and generated tokens.
- **LangChain Learning and Implementation Queries**: There was a spirited discussion about the learning curve with **LangChain**, with members sharing resources like [Grecil's personal journey](https://corrective-rag.streamlit.app) that includes tutorials and documentation. Meanwhile, debate about **ChatOpenAI** versus **Huggingface** models highlighted performance differences and adaptation in various scenarios.
- **Enhancing PDF Interrogation with LangChain**: A detailed guide was shared for generating Q&A pairs from PDFs using LangChain, referring to issues like [#17008](https://github.com/langchain-ai/langchain/issues/17008) on GitHub for further guidance. Adjustments for using **Llama2** as the LLM were also discussed, emphasizing customizing the `QAGenerationChain`.
- **From Zero to RAG Hero**: Members showcased their experience building no-code **RAG workflows** for financial documents, an [article](https://medium.com/@manthapavankumar11/effortless-no-code-rag-workflows-for-financial-documents-implementing-embedding-cache-and-chat-e8d267b1c888) detailing the process was shared. A discussion also centered around a custom [Corrective RAG app](https://corrective-rag.streamlit.app) and **Edimate**, an AI-driven video creation, demoed [here](https://x.com/dswharshit/status/1805203856088834428), which signs a future for e-learning.
- **AI Framework Evaluation Video**: For engineers evaluating AI frameworks for app integration including models like GPT-4o, a [YouTube video](https://youtu.be/uG0cs8AlnHw) was shared, urging developers to consider critical questions regarding the necessity and choice of the AI framework for specific applications.

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Jamba Instruct Boasts Big Context Window**: [AI21's Jamba-Instruct model](https://openrouter.ai/models/ai21/jamba-instruct) has been introduced, showcasing a gigantic **256K context window**, ideal for handling extensive documents in enterprise settings.
- **Nemotron 4 Makes Waves with Synthetic Data Generation**: [NVIDIA's release of Nemotron-4-340B-Instruct](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct) focuses on synthetic data generation for English-language applications with its new chat model.
- **JojoAI Levels Up to Proactive Assistant**: JojoAI differentiates itself by becoming a proactive assistant that can set reminders, employing DigiCord integrations, positioning it apart from competitors like ChatGPT or Claude. Experience it on the [JojoAI site](https://www.digicord.site).
- **Pebble's Pioneering Reading Aid Tool**: The unveiling of the **Pebble** tool, powered by OpenRouter with **Mistral 8x7b** and **Gemini**, provides a resource for enhancing reading comprehension and retention for web content. Kudos to the OpenRouter team for their support as acknowledged at [Pebble](https://pebble.study/).
- **Tech Community Tackles Environmental and Technical Issues**: Discussions pointed to concerns about the environmental footprint of using models like Nemotron 340b, with smaller models being recommended for efficiency and eco-friendliness. The community also dealt with practical affairs, such as resolving the disappearance of Claude self-moderated endpoints, praising Sonnet 3.5 for coding capabilities, addressing OpenRouter rate limits, and advising on best practices for handling exposed API keys.

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Local LLMs Enter OS Mode**: The OpenInterpreter community has been discussing the use of **local LLMs** in OS mode with the command `interpreter --local --os`, but there are concerns regarding their performance levels.
- **Desktop Delights and GitHub Glory**: The OpenInterpreter team is promoting a forthcoming **desktop app** with a unique experience compared to the GitHub version, encouraging users to join the [waitlist](https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com). Meanwhile, the project has celebrated **50,000 GitHub stars**, hinting at a major upcoming announcement.
- **Model Benchmarking Banter**: The **Codestral and Deepseek models** have sparked attention with Codestral surpassing internal benchmarks and Deepseek impressing users with its quick performance. There's buzz about a future optimized `interpreter --deepseek` command.
- **Cross-Platform Poetry Performance**: The use of **Poetry** for dependency management over `requirements.txt` has been a contentious topic, with some engineers pointing to its shortcomings on various operating systems and advocating for alternatives like **conda**.
- **Community Kudos and Concerns**: While there's enthusiasm and appreciation for the community's support, particularly for beginners, there's also frustration regarding shipping delays for the **01 device**, highlighting the balance between community sentiment and product delivery expectations.

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Instruction Synthesizing for the Win**: A newly shared [Hugging Face repository](https://huggingface.co/instruction-pretrain/instruction-synthesizer) highlights the potential of **Instruction Pre-Training**, providing 200M synthesized pairs across 40+ tasks, likely offering a robust approach to multi-task learning for AI practitioners looking to push the envelope in supervised multitask pre-training.

**Bringing DeBERTa and Flash Together?**: Curiosity is brewing over the possibility of combining **DeBERTa** with **Flash Attention 2**, posing the question of potential implementations that leverage both technologies to AI engineers interested in novel model architecture synergies.

**Fixes and Workarounds**: From a **Maven course platform** blank page issue solved using mobile devices to the resolution of permission errors after a kernel restart within **braintrust**, practical troubleshooting remains a staple of community discourse.

**Credits Saga Continues**: Persistent reports of missing service credits on platforms like Huggingface and Predibase sparked member-to-member support and referrals to respective billing supports. This included a tip that **Predibase credits expire after 30 days**, suggesting that engineers keep a keen eye on expiry dates to maximize credit use.

**Training Errors and Overfitting Queries**: Errors in running **Axolotl's training command** ([Modal FTJ](https://modal.com/ftj)) and concerns about **LORA overfitting** ('significantly lower training loss compared to validation loss') were significant pain points, showcasing the need for vigilant model monitoring practices among AI engineers.

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LightningAI and LlamaIndex Join Forces**: LightningAI's RAG template offers an [easy setup](https://t.co/2NLH7zuZS6) for multi-document agentic RAGs, promoting efficiency in AI development. Additionally, LlamaIndex's integration with [StabilityAI](https://t.co/a7F0gv4tpi) now allows for image generation, broadening AI developer capabilities.
- **Customizing Complexity with LlamaIndex**: Those developing with LlamaIndex can customize text-to-SQL pipelines using Directed Acyclic Graphs (DAGs), as explained in this [feature overview](https://t.co/fiS0kfj8rk). Meanwhile, for better financial analysis, the CRAG technique can be leveraged using Hanane Dupouy's [tutorial slides](https://t.co/lHsThk9IOU) for improved retrieval quality.
- **Fine-Tuning RAGs with Mlflow**: To enhance answer accuracy in RAGs, integrating LlamaIndex with [Mlflow](https://t.co/fo8XxMTO93) provides a systematic way to manage critical parameters and evaluation methods.
- **In-Depth Query Formatting and Parallel Execution in LlamaIndex**: Members discussed LlamaIndex's query response modes like **Refine** and **Accumulate**, and the utilization of **OLLAMA_NUM_PARALLEL** for concurrent model execution; document parsing and embedding mismatches were also topics of technical advice.
- **Streamlining ML Workflows with MLflow and LLMs**: A [Medium article](https://medium.com/ai-advances/unlocking-efficiency-in-machine-learning-a-guide-to-mlflow-and-llms-with-llamaindex-integration-2b1e7ade1437) by Ankush K Singal highlights the practical integration of MLflow and LLMs through LlamaIndex to streamline ML workflows.

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini vs. LLAMA Parameter Showdown**: A source from Meta indicated that **Gemini 1.5 Pro** has fewer parameters than **LLAMA 3 70B**, inciting discussions about the impact of MoE architectures on parameter count during inference.
- **GPT-4's Secret Sauce or Distilled Power**: The community debated whether **GPT-4T/o** are early fusion models or distilled versions of larger predecessors, showing divergence in understanding of their fundamental architectures.
- **Multimodal Training Dilemmas**: Members highlighted the difficulties in post-training multimodal models, citing the challenges of transferring knowledge across different data modalities. The struggles suggest a general consensus on the complexity of enhancing native multimodal systems.
- **Nosing Into Nous and Sony's Stir**: A tongue-in-cheek enquiry by a Nous Research member to @sonymusic sparked a blend of confusion and interest, touching upon AI's role in legal and innovation spaces.
- **Sketchy Metrics on AI Leaderboards**: The legitimacy of the **AlpacaEval leaderboard** came under fire with engineers questioning biased metrics after a model claimed to have beaten **GPT-4** while being more cost-effective. This led to discussions on the reliability of performance leaderboards in the field.

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **ROCm Forks Entering the Fray**: To utilize certain functionalities, engineers are advised to use the **ROCm fork versions** of [xformers](https://github.com/ROCm/xformers) and [flash-attention](https://github.com/ROCm/flash-attention), with a note on hardware support specifically for MI200 & MI300 GPUs and requirement of ROCm 5.4+ and PyTorch 1.12.1+.
- **Reward Models Dubbed Subpar for Data Gen**: The consensus is that the reward model isn't efficient for generating data, as it is designed mainly for classifying the quality of data, not producing it.
- **Synthesizing Standardized Test Questions**: An idea was shared to improve AGI evaluations for smaller models by synthesizing **SAT**, **GRE**, and **MCAT** questions, with an additional proposal to include **LSAT** questions.
- **Enigmatic Epoch Saving Quirks**: Training epochs are saving at seemingly random intervals, a behavior recognized as unusual but familiar to the community. This may be linked to the steps counter during the training process.
- **Dataset Formatting 101 and MinHash Acceleration**: A member sought advice on dataset formatting for **llama2-13b**, while another discussed formatting for the **Alpaca** dataset using **JSONL**. Moreover, a fast MinHash implementation named **Rensa** is shared for dataset deduplication, boasting a 2.5-3x speed increase over similar libraries, with its GitHub repository available for community inputs ([Rensa on GitHub](https://github.com/beowolx/rensa)).
- **Prompt Structures Dissected and Mirrored**: Clarification on `prompt_style` in the Axolotl codebase unveiled different prompt formatting strategies with **INSTRUCT**, **CHAT**, and **CHATML** highlighted for contrasting interactive uses. The use of `ReflectAlpacaPrompter` to automate prompt structuring using the designated style was exemplified ([More on Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=4809da1a-b260-413e-bdbe-8b82397846e6)).

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile Leveled Up**: [Llamafile v0.8.7](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.7) has been released, boasting **faster quant operations** and **bug fixes**, with whispers of an upcoming Android adaptation.
- **Globetrotting AI Events on the Horizon**: SF gears up for the **World's Fair of AI** and the **AI Quality Conference** with community leaders in attendance, while the [Mozilla Nightly Blog](https://blog.nightly.mozilla.org/2024/06/24/experimenting-with-ai-services-in-nightly/) hints at potential llamafile integration offering AI services.
- **Mozilla Nightly Blog Talks Llamafile**: The Nightly blog details experimentation with local AI chat services powered by llamafile, signaling potential for wider adoption and user accessibility.
- **Llamafile Execution on Colab Achieved**: Successful execution of a llamafile on Google Colab demonstrated, providing a [template for others to follow](https://colab.research.google.com/drive/1jWKKwVCQneCTB5VNQNWO0Wxqg1vG_E1T#scrollTo=13ISLtY9_v7g).
- **Memory Manager Facelift Connects Cosmos with Android**: A significant [GitHub commit](https://github.com/jart/cosmopolitan/commit/6ffed14b9cc68b79d530b23876f522f906173cca) for the Cosmopolitan project revamps the memory manager, enabling support for Android and stirring interest in running llamafile through Termux.

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **ORPO's Missing Piece**: The ORPO training option for **Torchtune** is not supported, though DPO can use a documented [recipe for training](https://github.com/pytorch/torchtune/blob/f200da58c8f5007b61266504204c61a171f6b3dd/recipes/configs/llama2/7B_lora_dpo.yaml#L9), as noted by guild members citing a [mix dataset for ORPO/DPO](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k).
- **Epochs Stuck on Single Setting**: Training on multiple datasets with **Torchtune** does not currently allow for different epoch settings for each—users should utilize *ConcatDataset* for combining datasets, but the same number of epochs applies to all.
- **To ChatML or Not to ChatML**: Engineers debated the efficacy of utilizing ChatML templates with the **Llama3** model, contrasting approaches using instruct tokenizer and special tokens against base models without these elements, referencing models like [Mahou-1.2-llama3-8B](https://huggingface.co/flammenai/Mahou-1.2-llama3-8B) and [Olethros-8B](https://huggingface.co/lodrick-the-lafted/Olethros-8B).
- **Tuning Phi-3 Takes Tweaks**: The task of fine-tuning **Phi-3 models** (like Phi-3-Medium-4K-Instruct) was addressed, with suggestions to modify the tokenizer and add a custom build function within Torchtune to enable compatibility.
- **System Prompts: Hack It With Phi-3**: Despite **Phi-3** not being optimized for system prompts, users can work around this by prepending system prompts to user messages and adjusting the tokenizer configuration with a specific [flag](https://github.com/pytorch/torchtune/blob/main/torchtune/models/phi3/_sentencepiece.py#L128) discussed to facilitate fine-tuning.

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Conditional Coding Conundrum**: In discussions about **tinygrad**, the use of a conditional operation like `condition * a + !condition * b` as a simplification for the WHERE function was met with caution due to potential issues with *NaNs*.
- **Intel Adventures in Tinygrad**: Queries about **Intel support** in **tinygrad** revealed that while **opencl** is an available option, the framework has not integrated XMX support to date.
- **Monday Meeting Must-Knows**: The **0.9.1 release** of **tinygrad** is on the agenda for the upcoming Monday meeting, focusing on *tinybox* updates, a new profiler, runtime improvements, `Tensor._tri`, llama cast speedup, and bounties for *uop matcher speed* and *unet3d* improvements.
- **Buffer View Toggle Added to Tinygrad**: A commit in **tinygrad** introduced a new flag to toggle the buffer view, a change that was substantiated with a [GitHub Actions run](https://github.com/tinygrad/tinygrad/actions/runs/9638260193/job/26578693946?pr=5120).
- **Lazy.py Logic in the Limelight**: An engineer seeks clarification after their edits to `lazy.py` within **tinygrad** resulted in a mix of both positive and negative process replay outcomes, suggesting a need for further investigation or peer review.

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Claude Sonnet 3.5 Stuns with Performance**: An engineer shared their experience using **Claude Sonnet 3.5** in Websim, praising its **speed, creativity,** and **intelligence**. They were particularly taken with the "generate in new tab" feature and experimented with sensory engagement by toying with color schemes from iconic fashion brands, as shown in a [shared tweet](https://fxtwitter.com/RobertHaisfield/status/1804945938936668413).

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AWS Cloud Club Lifts Off at MJCET**: MJCET has launched the **first AWS Cloud Club in Telangana**, a community aimed at providing students with resources and experience in **Amazon Web Services** to prepare for tech industry careers.
- **Cloud Mastery Event with an AWS Expert**: An inaugural event will celebrate the AWS Cloud Club's launch on **June 28th, 2024**, featuring **AWS Community Hero Mr. Faizal Khan**. Interested parties can RSVP via an [event link](https://meetu.ps/e/NgmgX/14DgQ2/i).

---

The **AI Stack Devs (Yoko Li) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **Datasette - LLM (@SimonW) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1253795430014259370)** (715 messages🔥🔥🔥):

- **Juggernaut Lightning vs SD3 Turbo**: A member recommended using Juggernaut Lightning as it is "way more realistic" compared to SD3 Turbo due to it being a base model. Another member mentioned Juggernaut being more suited for role-playing and creativity rather than coding and intelligence.
- **Help for Beginners**: An ML beginner sought advice on which libraries to use for their project and received suggestions to use PyTorch for its extensive neural network support and HuggingFace for loading pre-trained models. Another member recommended avoiding outdated libraries like sklearn.
- **Model Loading Issues**: A member faced challenges loading large AI models on limited hardware and received guidance on using quantization techniques to improve performance. Recommendations included installing the bitsandbytes library and instructions for modifying model load configurations to utilize 4-bit precision.
- **AI Content Creation Tools**: There was a discussion on the complexities of generating AI-generated videos similar to Vidalgo, indicating that while generating text and audio is straightforward, creating small moving videos is challenging. Tools like RunwayML and Capcut were suggested for video edits and stock images.
- **Collaborative Projects and Model Updates**: Members shared their experiences and projects related to various AI models, including a model trained to play games using Xbox controller inputs and a toolkit for preprocessing large image datasets. Additionally, ongoing work and upcoming updates on several models and their potential applications were discussed.

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://docs.continue.dev/how-to-use-continue">🧑‍🎓 How to use Continue | Continue</a>: Using LLMs as you code with Continue</li><li><a href="https://en.wikipedia.org/wiki/Chess_notation">Chess notation - Wikipedia</a>: no description found</li><li><a href="https://bhosmer.github.io/mm/ref.html">mm ref</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/index">Datasets</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=_mkyL0Ww_08">Anthropic's SHOCKING New Model BREAKS the Software Industry! Claude 3.5 Sonnet Insane Coding Ability</a>: Learn AI With Me:https://www.skool.com/natural20/aboutJoin my community and classroom to learn AI and get ready for the new world.#ai #openai #llm</li><li><a href="https://www.swebench.com/">SWE-bench</a>: no description found</li><li><a href="https://huggingface.co/briaai/RMBG-1.4">briaai/RMBG-1.4 · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-141b-A35b">alignment-handbook/recipes/zephyr-141b-A35b at main · huggingface/alignment-handbook</a>: Robust recipes to align language models with human and AI preferences - huggingface/alignment-handbook</li><li><a href="https://en.wikipedia.org/wiki/Apple_M1#GPU">Apple M1 - Wikipedia</a>: no description found</li><li><a href="https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-beta">alignment-handbook/recipes/zephyr-7b-beta at main · huggingface/alignment-handbook</a>: Robust recipes to align language models with human and AI preferences - huggingface/alignment-handbook</li><li><a href="https://huggingface.co/papers/2310.16944">Paper page - Zephyr: Direct Distillation of LM Alignment</a>: no description found</li><li><a href="https://huggingface.co/chat?model=HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://github.com/abi/screenshot-to-code">GitHub - abi/screenshot-to-code: Drop in a screenshot and convert it to clean code (HTML/Tailwind/React/Vue)</a>: Drop in a screenshot and convert it to clean code (HTML/Tailwind/React/Vue) - abi/screenshot-to-code</li><li><a href="https://github.com/simpler-env/SimplerEnv">GitHub - simpler-env/SimplerEnv: Evaluating and reproducing real-world robot manipulation policies (e.g., RT-1, RT-1-X, Octo) in simulation under common setups (e.g., Google Robot, WidowX+Bridge)</a>: Evaluating and reproducing real-world robot manipulation policies (e.g., RT-1, RT-1-X, Octo) in simulation under common setups (e.g., Google Robot, WidowX+Bridge) - simpler-env/SimplerEnv</li><li><a href="https://huggingface.co/blog?tag=rlhf">Hugging Face – Blog</a>: no description found</li><li><a href="https://huggingface.co/posts/nroggendorff/357091156426242">@nroggendorff on Hugging Face: "@osanseviero your move"</a>: no description found</li><li><a href="https://youtu.be/udPY5rQVoW0">Playing a Neural Network's version of GTA V: GAN Theft Auto</a>: GAN Theft Auto is a Generative Adversarial Network that recreates the Grand Theft Auto 5 environment. It is created using a GameGAN fork based on NVIDIA's Ga...</li><li><a href="https://tenor.com/view/huh-cat-cat-huh-small-cat-huh-what-gif-2593177363967991691">Huh Cat GIF - Huh Cat Cat huh - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/tAEVgAa4CZw?si=_1ThlIeIQyJAGpze">Hand Gesture Drawing App Demo | Python OpenCV &amp; Mediapipe</a>: In this video, I demonstrate my Hand Gesture Drawing App using Python with OpenCV and Mediapipe. This app allows you to draw on screen using hand gestures de...</li><li><a href="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1">stabilityai/stable-video-diffusion-img2vid-xt-1-1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/main">microsoft/Phi-3-mini-4k-instruct-gguf at main</a>: no description found</li><li><a href="https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3">RAG chatbot using llama3</a>: no description found</li><li><a href="https://huggingface.co/Azazelle/L3-RP_io/tree/main">Azazelle/L3-RP_io at main</a>: no description found</li><li><a href="https://www.vidalgo.tech/">Vidalgo - One-Click Vertical Video Creation</a>: Experience effortless video creation with Vidalgo! Our platform empowers you to produce stunning vertical videos for TikTok, YouTube Shorts, and Instagram Reels in just one click. Start creating today...</li><li><a href="https://huggingface.co/stabilityai/stablelm-zephyr-3b">stabilityai/stablelm-zephyr-3b · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/">Hugging Face</a>: The AI community building the future. Hugging Face has 227 repositories available. Follow their code on GitHub.</li><li><a href="https://tenor.com/view/toy-story-woody-buzz-lightyear-funny-gif-13488605">Toy Story Woody GIF - Toy Story Woody Buzz Lightyear - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/Azaz">azaz (Z)</a>: no description found</li><li><a href="https://github.com/huggingface/alignment-handbook">GitHub - huggingface/alignment-handbook: Robust recipes to align language models with human and AI preferences</a>: Robust recipes to align language models with human and AI preferences - huggingface/alignment-handbook</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/agent#transformers.Agent">Agents &amp; Tools</a>: no description found</li><li><a href="https://github.com/beowolx/rensa">GitHub - beowolx/rensa: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets</a>: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets - beowolx/rensa</li><li><a href="https://stackoverflow.com/questions/7821661/how-to-write-code-to-autocomplete-words-and-sentences">How to write code to autocomplete words and sentences?</a>: I'd like to write code that does autocompletion in the Linux terminal. The code should work as follows. It has a list of strings (e.g. &amp;quot;hello, &amp;quot;hi&amp;quot;, &amp;quot;how a...</li><li><a href="https://github.com/minimaxir/textgenrnn">GitHub - minimaxir/textgenrnn: Easily train your own text-generating neural network of any size and complexity on any text dataset with a few lines of code.</a>: Easily train your own text-generating neural network of any size and complexity on any text dataset with a few lines of code. - minimaxir/textgenrnn</li><li><a href="https://github.com/huggingface/datatrove">GitHub - huggingface/datatrove: Freeing data processing from scripting madness by providing a set of platform-agnostic customizable pipeline processing blocks.</a>: Freeing data processing from scripting madness by providing a set of platform-agnostic customizable pipeline processing blocks. - huggingface/datatrove</li><li><a href="https://github.com/not-lain/loadimg">GitHub - not-lain/loadimg: a python package for loading images</a>: a python package for loading images. Contribute to not-lain/loadimg development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/vaas-far-cry3-that-is-crazy-gif-26006603">Vaas Far Cry3 GIF - Vaas Far Cry3 That Is Crazy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/vikhyatk/status/1804672473172210106">Tweet from vik (@vikhyatk)</a>: asked claude to make me a cool new vaporwave style home page... should i switch to it?</li><li><a href="https://x.com/vikhyatk/status/1804673335437254721">Tweet from vik (@vikhyatk)</a>: "make it better"</li><li><a href="https://we.tl/t-3ZjcQJIKA2">sonnet_shooter.zip</a>: 1 file sent via WeTransfer, the simplest way to send your files around the world</li><li><a href="https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py#L705">huggingface_hub/src/huggingface_hub/hub_mixin.py at main · huggingface/huggingface_hub</a>: The official Python client for the Huggingface Hub. - huggingface/huggingface_hub</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bnetfp/is_the_p40_the_most_costeffective_way_to_run/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://wandb.ai/vanpelt/m1-benchmark/reports/Can-Apple-s-M1-help-you-train-models-faster-cheaper-than-NVIDIA-s-V100---VmlldzozNTkyMzg">Can Apple’s M1 Help You Train Models Faster &amp; Cheaper Than NVIDIA’s V100?</a>: In this article, we analyze the runtime, energy usage, and performance of Tensorflow training on an M1 Mac Mini and Nvidia V100. .</li><li><a href="https://github.com/maxmelichov/Text-To-speech">GitHub - maxmelichov/Text-To-speech: Roboshaul</a>: Roboshaul. Contribute to maxmelichov/Text-To-speech development by creating an account on GitHub.</li><li><a href="http://www.roboshaul.com/">Robo-Shaul project</a>: The Robo-Shaul Competition was a 2023 competition to clone the voice of Shaul Amsterdamski. The results are all here.</li><li><a href="https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/">Introducing Accelerated PyTorch Training on Mac</a>: In collaboration with the Metal engineering team at Apple, we are excited to announce support for GPU-accelerated PyTorch training on Mac. Until now, PyTorch training on Mac only leveraged the CPU, bu...</li><li><a href="https://www.nature.com/articles/s41467-024-46631-y">Alignment of brain embeddings and artificial contextual embeddings in natural language points to common geometric patterns - Nature Communications</a>: Here, using neural activity patterns in the inferior frontal gyrus and large language modeling embeddings, the authors provide evidence for a common neural code for language processing.</li></ul></div>

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1254401005286592604)** (3 messages):

- **Coding Self-Attention and Multi-Head Attention**: A member shared a link to their blog post detailing the implementation of self-attention and multi-head attention from scratch. The blog post explains the importance of attention in Transformer architecture for understanding word relationships in a sentence to make accurate predictions. [Read the full post here.](https://ash-01xor.github.io/blog/posts/Attention/)
- **Interest in Blog Post**: Another member expressed interest in the blog post on attention mechanisms. They affirmed their engagement with a simple "Yes I am interested."
- **Tree-Sitter S-expression Challenges**: A member mentioned the challenges they are facing with Tree-Sitter S-expressions, referring to them as "a pain." This suggests difficulties in parsing or handling these expressions in their current work.

**Link mentioned**: [Ashvanth.S Blog - Wrapping your head around Self-Attention, Multi-head Attention](https://ash-01xor.github.io/blog/posts/Attention/): no description found

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1253935410091003946)** (5 messages):

- **Implementing RMSNorm Layer in SD3**: A member mentioned implementing an optional **RMSNorm layer** for the Q and K inputs, referencing the [SD3 paper](https://arxiv.org/pdf/2403.03206). No further details were provided on this implementation.
- **LLMs and Refusal Mechanisms**: A blog post was shared about **LLM refusal/safety** highlighting that *refusal is mediated by a single direction in the residual stream*. The full explanation and more insights can be found in the [paper now available on arXiv](https://arxiv.org/abs/2406.11717).
- **Florence-2 Vision Foundation Model**: The abstract for **Florence-2**, a vision foundation model, was posted [on arXiv](https://arxiv.org/abs/2311.06242). Florence-2 uses a unified prompt-based representation across various computer vision and vision-language tasks, leveraging a large dataset with 5.4 billion annotations.
- **Facebook AI Twitter Link**: A Twitter link related to **Facebook AI** was shared without any additional context. [Twitter link](https://twitter.com/FacebookAIslop)
- **wLLama Test Page**: A link was shared to a **wLLama basic example** page demonstrating model completions and embeddings. Users can test models, input local files, and calculate cosine distances between text embeddings [wLLama Basic Example](http://wllama-basic-example.glitch.me/).
  

**Links mentioned**:

- [wllama.cpp demo](http://wllama-basic-example.glitch.me/): no description found
- [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242): We introduce Florence-2, a novel vision foundation model with a unified, prompt-based representation for a variety of computer vision and vision-language tasks. While existing large vision models exce...
- [Refusal in LLMs is mediated by a single direction — AI Alignment Forum](https://www.alignmentforum.org/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction.): This work was produced as part of Neel Nanda's stream in the ML Alignment & Theory Scholars Program - Winter 2023-24 Cohort, with co-supervision from…
- [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717): Conversational large language models are fine-tuned for both instruction-following and safety, resulting in models that obey benign requests but refuse harmful ones. While this refusal behavior is wid...

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1253878207875125328)** (12 messages🔥):

- **Mistroll 7B Version 2.2 Released**: A member shared the [Mistroll-7B-v2.2 model](https://huggingface.co/BarraHome/Mistroll-7B-v2.2) trained 2x faster with Unsloth and Huggingface's TRL library. This experiment aims to fix incorrect behaviors in models and refine training pipelines focusing on data engineering and evaluation performance.
- **Stable Diffusion Trainer Code Shared**: A simple Stable Diffusion 1.5 Finetuner for experimentation was shared on [GitHub](https://github.com/CodeExplode/MyTrainer). This "very janky" code uses Diffusers, aimed at helping users explore finetuning.
- **Media to Text Conversion Software Release**: Developed by a member, this software converts media files into text using PyQt for GUI and OpenAI Whisper for STT, supporting local and YouTube video transcriptions. Available on [GitHub](https://github.com/yjg30737/whisper_transcribe_youtube_video_example_gui).
- **Enhancements to SimpleTuner**: Refactored and enhanced EMA support for SimpleTuner was shared, now compatible with SD3 and PixArt training, supporting CPU offload and step-skipping. The changes can be reviewed on [GitHub](https://github.com/bghira/SimpleTuner/pull/521/files).
- **Featherless.ai - New AI Platform**: A member introduced [Featherless.ai](https://featherless.ai), a platform to run public models from Huggingface serverlessly, instantly. They are onboarding 100+ models weekly and aim to cover all HF public models, inviting users to try the service and provide feedback.
  

**Links mentioned**:

- [BarraHome/Mistroll-7B-v2.2 · Hugging Face](https://huggingface.co/BarraHome/Mistroll-7B-v2.2): no description found
- [Linear Regression From Scratch In Python](https://medium.com/@amitsubhashchejara/linear-regression-from-scratch-in-python-ee1a955e49ed): Learn the implementation of linear regression from scratch in pure Python. Cost function, gradient descent algorithm, training the model…
- [GitHub - CodeExplode/MyTrainer: A simple Stable Diffusion 1.5 Finetuner for experimentation](https://github.com/CodeExplode/MyTrainer): A simple Stable Diffusion 1.5 Finetuner for experimentation - CodeExplode/MyTrainer
- [GitHub - yjg30737/pyqt-assistant-v2-example: OpenAI Assistant V2 Manager created with PyQt (focused on File Search functionality)](https://github.com/yjg30737/pyqt-assistant-v2-example): OpenAI Assistant V2 Manager created with PyQt (focused on File Search functionality) - yjg30737/pyqt-assistant-v2-example
- [GitHub - yjg30737/whisper_transcribe_youtube_video_example_gui: GUI Showcase of using Whisper to transcribe and analyze Youtube video](https://github.com/yjg30737/whisper_transcribe_youtube_video_example_gui): GUI Showcase of using Whisper to transcribe and analyze Youtube video - yjg30737/whisper_transcribe_youtube_video_example_gui
- [EMA: refactor to support CPU offload, step-skipping, and DiT models | pixart: reduce max grad norm by default, forcibly by bghira · Pull Request #521 · bghira/SimpleTuner](https://github.com/bghira/SimpleTuner/pull/521/files>): no description found
- [CaptionEmporium/coyo-hd-11m-llavanext · Datasets at Hugging Face](https://huggingface.co/datasets/CaptionEmporium/coyo-hd-11m-llavanext): no description found
- [Featherless - Serverless LLM](https://featherless.ai/): Featherless - The latest LLM models, serverless and ready to use at your request.
- [Featherless AI - Run every 🦙 AI model & more from 🤗 huggingface | Product Hunt](https://www.producthunt.com/posts/featherless-llm): Featherless is a platform to use the very latest open source AI models from Hugging Face. With hundreds of new models daily, you need dedicated tools to keep with the hype. No matter your use-case, fi...

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1253848266756460545)** (5 messages):

- **Chad plans reasoning with LLMs discussion**: A member announced plans to discuss "reasoning with LLMs" next Saturday and received enthusiastic support. He felt most confident about this topic and chose it over Triton.
- **Readying for “Understanding the Current State of Reasoning with LLMs”**: Chad stated he would start with the paper *Understanding the Current State of Reasoning with LLMs* [arXiv link](https://arxiv.org/abs/2206.07682) and referenced an elaborative Medium article [article link](https://medium.com/@isamu-website/understanding-the-current-state-of-reasoning-with-llms-dbd9fa3fc1a0).
- **Exploring Awesome-LLM-Reasoning repositories**: He mentioned diving into repositories like [Awesome-LLM-Reasoning](https://github.com/atfortes/Awesome-LLM-Reasoning) and another repository with the same name [alternative repository link](https://github.com/luban-agi/Awesome-LLM-reasoning) to explore the current state of LLMs for logic.
- **Survey Paper Mentioned**: Chad plans to go through the beginning of *Natural Language Reasoning, A Survey* [survey PDF](https://arxiv.org/pdf/2303.14725) and reference papers published post-GPT-4 launch [GPT-4 research link](https://openai.com/index/gpt-4-research/).
- **Seeking long-term planning papers**: He expressed interest in learning about good long-term planning papers for LLMs, particularly those focused on pentesting.
  

**Links mentioned**:

- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682): Scaling up language models has been shown to predictably improve performance and sample efficiency on a wide range of downstream tasks. This paper instead discusses an unpredictable phenomenon that we...
- [Understanding the Current State of Reasoning with LLMs](https://medium.com/@isamu-website/understanding-the-current-state-of-reasoning-with-llms-dbd9fa3fc1a0): The goal of this article is to go through the repos of Awesome-LLM-Reasoning and Awesome-LLM-reasoning for an understanding of the current…

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1253912311622664213)** (9 messages🔥):

- **Pricing Performance for OCR Models**: Members are seeking recommendations for a good price-to-performance model for OCR that outputs in JSON. This highlights ongoing quests for cost-effective AI solutions.
- **Stable Faces, Changing Hairstyles Video**: A video showing a model where *"faces almost remained constant but the hairstyle kept changing"* sparked curiosity about which model achieved this. The video can be found [here](https://x.com/gunsnrosesgirl3/status/1804462040871801220?s=46).
- **Unsupported Image Type RuntimeError**: A user encountered a *"RuntimeError: Unsupported image type, must be 8bit gray or RGB image."* This occurred during the encoding process of images for face recognition, with code provided for debugging.

**Link mentioned**: [Tweet from Science girl (@gunsnrosesgirl3)](https://x.com/gunsnrosesgirl3/status/1804462040871801220?s=46): The evolution of fashion using AI

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages):

capetownbali: Let us all know how your fine tuning on LLama goes!

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1254221908036026501)** (2 messages):

- **Redirect to diffusion-discussions channel**: A user advised, "Your best bet is to ask [here](https://discord.com/channels/879548962464493619/1019883044724822016)" for further discussions on the related topic.
- **Inquiry about audio conversion models**: A member inquired about the availability of models for audio-to-audio conversion, specifically from Urdu/Hindi to English, indicating a need for multilingual processing capabilities.

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1253788511379787947)** (376 messages🔥🔥):

- **Cossale eagerly awaits Unsloth's release**: They requested early access and were informed by theyruinedelise that the video would be filmed the next day. They can watch a temporary [recording](https://discord.com/channels/1179035537009545276/1179035537529643040/1253726564500111381) in the meantime.
- **Feedback on Thumbnails and Flowcharts**: Cossale suggested changes to the thumbnail for clarity, prompting theyruinedelise to update it from "csv -> unsloth + ollama" to "csv -> unsloth -> ollama". They also advised adding descriptive text below logos for beginner users.
- **Gigantic VRAM discussions impress**: Members discussed Phison's impressive PCIe-NVMe card presenting as 1Tb VRAM, impacting performance. Fimbulvntr shared a [YouTube video](https://youtu.be/L4Bmrk2QprE?si=x-iFJrVRcK9-MQ8t&t=679) to explain this tech.
- **Excitement around extended LLMs**: Fimbulvntr succeeded in extending Llama-3-70b’s context to 64k, and iron_bound debated performance implications of VRAM expansion. The conversation touched on various large model updates and their potential impacts.
- **Upcoming releases and resources in the community**: Theyruinedelise announced the Ollama update set for Monday or Tuesday including CSV file support. Additionally, Sebastien's fine-tuned emotional llama model and its supportive resources are now available on [Ollama](https://ollama.com/sebdg/emotional_llama) and [YouTube](https://www.youtube.com/watch?v=ZJKglSWgD0w).


<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.lamini.ai/blog/lamini-memory-tuning">Introducing Lamini Memory Tuning: 95% LLM Accuracy, 10x Fewer Hallucinations | Lamini - Enterprise LLM Platform</a>: no description found</li><li><a href="https://app.uniswap.org/explore/tokens/ethereum/0xfaca6611fca6de09f726b8a0a1448253b6f748e5">Get DOLPHIN on Uniswap</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1805265568658076069">Tweet from Unsloth AI (@UnslothAI)</a>: Tomorrow we will be handing out our new stickers for the @aiDotEngineer World's Fair! 🦥 Join us at 9AM, June 25 where we will be doing workshops on LLM analysis + technicals, @Ollama support &amp; m...</li><li><a href="https://arxiv.org/abs/2405.12130">MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning</a>: Low-rank adaptation is a popular parameter-efficient fine-tuning method for large language models. In this paper, we analyze the impact of low-rank updating, as implemented in LoRA. Our findings sugge...</li><li><a href="https://x.com/Nottlespike/status/1805054022661087657?t=oJn25UE9vTesxym63ToA1A&amp;s=09">Tweet from Kearm (@Nottlespike)</a>: http://x.com/i/article/1805030133478350848</li><li><a href="https://x.com/dudeman6790/status/1805108449581072710">Tweet from RomboDawg (@dudeman6790)</a>: Announcing Replete-Coder-Qwen2-1.5b An uncensored, 1.5b model with good coding performance across over 100 coding languages, open source data, weights, training code, and fully usable on mobile platfo...</li><li><a href="https://www.youtube.com/watch?v=ZJKglSWgD0w">Emotions in AI: Fine-Tuning, Classifying, and Reinforcement Learning</a>: In this video we are exploring the creation of fine-tuning dataset for LLM's using Unsloth and Ollama to train a specialized model for emotions detection.You...</li><li><a href="https://www.youtube.com/watch?v=dik_wnOE4dk">Tell 'im 'e's dreamin'</a>: Some clips from the movie The Castle.</li><li><a href="https://youtu.be/L4Bmrk2QprE?si=x-iFJrVRcK9-MQ8t&amp;t=679">AI and Unified Memory Architecture: Is it in the Hopper? Is it Long on Promise, Short on Delivery?</a>: Sit back, relax and enjoy the soothing sounds of Wendell's rambleing. This episode focuses on the MI 300a/x and Nvidia Grace Hopper. Enjoy!******************...</li><li><a href="https://cloud.llamaindex.ai">LlamaCloud</a>: no description found</li><li><a href="https://tenor.com/view/noice-nice-click-gif-8843762">Noice Nice GIF - Noice Nice Click - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/sebdg/SebLLama-Notebooks/tree/main/Emotions">SebLLama-Notebooks/Emotions at main · sebdg/SebLLama-Notebooks</a>: Contribute to sebdg/SebLLama-Notebooks development by creating an account on GitHub.</li><li><a href="https://github.com/Unstructured-IO/unstructured">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines. - GitHub - Unstructured-IO/unstructured: Open source librar...</li><li><a href="https://github.com/datamllab/LongLM">GitHub - datamllab/LongLM: [ICML'24 Spotlight] LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning</a>: [ICML'24 Spotlight] LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning - datamllab/LongLM</li><li><a href="https://colab.research.google.com/drive/1lq043a_zdssGBWJakckyy3yrbNSqqixP#scrollTo=ekOmTR1hSNcr">Google Colab</a>: no description found</li><li><a href="https://ollama.com/sebdg/emotional_llama">sebdg/emotional_llama</a>: Introducing Emotional Llama, the model fine-tuned as an exercise for the live event on Ollama discord channer. Designed to understand and respond to a wide range of emotions.</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Qwen-1.5b">Replete-AI/Replete-Coder-Qwen2-1.5b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Replete-AI/Adapter_For_Replete-Coder-Qwen-1.5b/">Replete-AI/Adapter_For_Replete-Coder-Qwen2-1.5b · Hugging Face</a>: no description found</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1253862702561103904)** (108 messages🔥🔥):

- **Logitech mouse and ChatGPT wrapper**: A member discussed using a Logitech mouse with a “cool” ChatGPT wrapper capable of programming basic queries such as summarizing and rewriting text. They shared a link to show the UI of this setup.


<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://arxiv.org/abs/2401.11817">Hallucination is Inevitable: An Innate Limitation of Large Language Models</a>: Hallucination has been widely recognized to be a significant drawback for large language models (LLMs). There have been many works that attempt to reduce the extent of hallucination. These efforts hav...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dl8guc/hf_eng_llama_400_this_summer_informs_how_to_run/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/PygmalionAI/aphrodite-engine">GitHub - PygmalionAI/aphrodite-engine: PygmalionAI's large-scale inference engine</a>: PygmalionAI's large-scale inference engine. Contribute to PygmalionAI/aphrodite-engine development by creating an account on GitHub.</li><li><a href="https://link.springer.com/article/10.1007/s10676-024-09775-5">ChatGPT is bullshit - Ethics and Information Technology</a>: Recently, there has been considerable interest in large language models: machine learning systems which produce human-like text and dialogue. Applications of these systems have been plagued by persist...</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1253789943264972892)** (228 messages🔥🔥):

- **Installation Woes with Xformers on Windows**: One user struggled to install xformers on Windows when setting up Unsloth via conda, encountering a "PackagesNotFoundError." [Another](https://anaconda.org) suggested that the challenges may be due to platform compatibility, prompting discussions about whether Unsloth works better on Linux.
- **Trouble Importing FastLanguageModel in Colab**: Users reported issues with importing `FastLanguageModel` in Unsloth’s Google Colab notebooks. A workaround suggested was ensuring all initial cells, particularly those installing Unsloth, are executed properly.
- **Results Varying Based on Token Expiration**: One user solved their issues by changing their Google account, identifying that an expired token in Colab secrets was causing problems, particularly around accessing datasets and downloading models.
- **Using Huggingface Tokens**: A user discovered that adding a Huggingface token fixed access issues, prompting confusion as models were meant to be public. The general sentiment was that inconsistencies in Huggingface access could be at play.
- **Running Unsloth with Docker and Jupyter**: There was a discussion about setting up Unsloth on NVIDIA GPU Cloud (NGC) containers with compatibility issues noted for specific CUDA and PyTorch versions. A solution involved trying different containers and careful installation of dependencies like xformers and bitsandbytes, with users sharing their Dockerfile configurations.
  

**Links mentioned**:

- [PyTorch Release 24.05 - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-05.html#rel-24-05): no description found
- [Home](https://github.com/unslothai/unsloth/wiki): Finetune Llama 3, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
- [unsloth (Unsloth AI)](https://huggingface.co/unsloth): no description found
- [GitHub - srush/Triton-Puzzles: Puzzles for learning Triton](https://github.com/srush/Triton-Puzzles/): Puzzles for learning Triton. Contribute to srush/Triton-Puzzles development by creating an account on GitHub.
- [Sao10K/Claude-3-Opus-Instruct-15K · Datasets at Hugging Face](https://huggingface.co/datasets/Sao10K/Claude-3-Opus-Instruct-15K): no description found
- [I got unsloth running in native windows. · Issue #210 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/210): I got unsloth running in native windows, (no wsl). You need visual studio 2022 c++ compiler, triton, and deepspeed. I have a full tutorial on installing it, I would write it all here but I’m on mob...
- [Google Colab breaks · Issue #243 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/243): I am getting the below error while trying to import the FastLangugeModel from unsloth while using an A100 GPU on colab. Failed to import transformers.integrations.peft because of the following erro...
- [Google Colab](https://colab.research.google.com/drive/19ScqSD6-p9NBrpyq5XzVwayhpNnn7YYf?usp=sharing): no description found
- [Google Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing]): no description found
- [Google Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=QmUBVEnvCDJv): no description found
- [Google Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=QmUBV): no description found
- [CUDA_VISIBILE_DEVICES not functioning · Issue #660 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/660): I saw error message when I am trying to do supervised fine tuning with 4xA100 GPUs. So the free version cannot be used on multiple GPUs? RuntimeError: Error: More than 1 GPUs have a lot of VRAM usa...
- [unsloth/unsloth/models/llama.py at main · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py): Finetune Llama 3, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
- [Package repository for pytorch :: Anaconda.org](https://conda.anaconda.org/pytorch): no description found
- [Package repository for nvidia :: Anaconda.org](https://conda.anaconda.org/nvidia): no description found
- [Package repository for xformers :: Anaconda.org](https://conda.anaconda.org/xformers): no description found

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1254452617564782723)** (1 messages):

- **Blog on Apple and Meta partnership stirs conversation**: An AI enthusiast shared a blog post titled [Apple and Meta Partnership: The Future of Generative AI in iPhones](https://ghost-x.org/blog/apple-and-meta-partnership-the-future-of-generative-ai-in-iphones/). The article discusses the implications, benefits, and challenges of integrating generative AI models into Apple's AI system, generating interest in the potential impact on the tech landscape.

**Link mentioned**: [Apple and Meta Partnership: The Future of Generative AI in iPhones](https://ghost-x.org/blog/apple-and-meta-partnership-the-future-of-generative-ai-in-iphones/): Recent discussions between Apple and AI companies like Meta regarding partnerships to integrate generative AI models into Apple's AI system for iPhones have generated significant interest. This articl...

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1253786569949708398)** (583 messages🔥🔥🔥):

- **Discord Bot Advertisement Gone Wrong**: A member shared a bot link, claiming it integrates with Gemini for chat assistance and StabilityAI for text-to-image generation. Others criticized the link's lack of context and its potential safety issues.
- **Civitai and SD3 Licensing Drama**: There was a heated debate over Civitai removing SD3 resources due to licensing concerns. One member argued this was done in response to potential legal issues, while others found the justification dubious.
- **Stable Diffusion on Low-End GPUs**: Multiple members discussed the challenges of running Stable Diffusion on low-spec machines. Suggestions included using automatic1111 and adjusting settings like steps and resolution, and there was a debate about the effectiveness of older GPUs versus newer ones like RTX 4080.
- **Training and Technical Discussions**: Members asked for advice on training models and handling errors, including issues with metadata and VRAM allocation. Recommendations were given to join specific training servers or use tools like ComfyUI and OneTrainer for better management.
- **Misunderstood Model Integrations**: Users discussed compatibility issues between different model architectures, particularly between SD 1.5, SDXL, and ControlNet modules. The significance of matching model types with their appropriate extensions was highlighted to avoid errors and improve performance.
  

**Links mentioned**:

- [no title found](https://www.youtube.co): no description found
- [Discord - Group Chat That’s All Fun & Games](https://dsc.gg/vexel): Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.
- [Green Code](https://www.youtube.com/@Green-Code): 01001000 01101001 00100001 00100000 01001001 00100000 01101101 01100001 01101011 01100101 00100000 01110110 01101001 01100100 01100101 01101111 01110011 00100000 01100001 01100010 01101111 01110101 01...
- [Stable Diffusion 3 Medium - a Hugging Face Space by stabilityai](https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium): no description found
- [Alfredo Canziani](https://www.youtube.com/@alfcnz): Music, math, and deep learning from scratch
- [FIFO Diffusion tests](https://www.youtube.com/watch?v=bRbIUf2XIII): As the title says rendered in FIFO Diffusion. 4 1/2h render time for both clips total on a 4090. Kinda underwhelming . Will give it an other chance.
- [Advanced Style transfer with the Mad Scientist node](https://youtu.be/ewKM7uCRPUg?si=9xtX87QB8_-3F19i): We are talking about advanced style transfer, the Mad Scientist node and Img2Img with CosXL-edit. Upgrade the IPAdapter extension to be able to use all the n...
- [Hot Sweating GIF - Hot Sweating Melting - Discover & Share GIFs](https://tenor.com/view/hot-sweating-melting-burning-donald-duck-gif-12424818824830848928): Click to view the GIF
- [Well, This Is Shit](https://open.spotify.com/track/0dyz56yOkAvjJMAJ5IxfiE?si=caaca21651c44ccc): Thomas Benjamin Wild Esq · Song · 2021
- [lllyasviel/sd-controlnet-canny at main](https://huggingface.co/lllyasviel/sd-controlnet-canny/tree/main): no description found
- [Download the latest official NVIDIA drivers](http://www.nvidia.com/Download/index.aspx): Download the latest official NVIDIA drivers
-   [```
    PyTorch
  ```](https://pytorch.org)
  [](https://pytorch.org): no description found
- [List of Aesthetics](https://aesthetics.fandom.com/wiki/List_of_Aesthetics#Aesthetics_by_Type): If you need assistance with identifying your aesthetic or creating a moodboard, feel free to ask questions in the Discussion Tab (in the pull-down bar of the "Explore" tab at the top of the ...
- [lllyasviel/sd_control_collection at main](https://huggingface.co/lllyasviel/sd_control_collection/tree/main): no description found
- [TypeError: list indices must be integers or slices, not str](https://stackoverflow.com/questions/32554527/typeerror-list-indices-must-be-integers-or-slices-not-str): I've got two lists that I want to merge into a single array and finally put it in a csv file. How I can avoid this error : def fill_csv(self, array_urls, array_dates, csv_file_path): ...
- [stable-diffusion-webui/requirements_versions.txt at master · AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/requirements_versions.txt): Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.
- [0002 - Pony - v3.1alt | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/471439/0002-pony): differences between 0001: Higher saturation Brighter More dynamic uses 2 loras trained by me: - QEW: [https://civitai.com/models/470285/qew-quasarca](https://civitai.com/models/470285/qew-quasarca)...
- [GitHub - Nerogar/OneTrainer: OneTrainer is a one-stop solution for all your stable diffusion training needs.](https://github.com/Nerogar/OneTrainer): OneTrainer is a one-stop solution for all your stable diffusion training needs. - Nerogar/OneTrainer
- [Feature request: Option to run CodeFormer and/or GFPGAN automatically again after upscale · Issue #1151 · AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/1151): Is your feature request related to a problem? Please describe. I've noticed that it seems GFPGAN and CodeFormer run before the upscaling happens, which results in a bit of a blurred resolution in ...
- [[Feature Request]: Offline Mode · Issue #11518 · AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/11518): Is there an existing issue for this? I have searched the existing issues and checked the recent builds/commits What would your feature do ? Have an option to download all files that could be reques...
- [GitHub - lucidrains/mmdit: Implementation of a single layer of the MMDiT, proposed in Stable Diffusion 3, in Pytorch](https://github.com/lucidrains/mmdit): Implementation of a single layer of the MMDiT, proposed in Stable Diffusion 3, in Pytorch - lucidrains/mmdit
- [ABS Aquilon Aqua Gaming PC - Windows 11 Home - Intel Core i7 14th Gen 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI-Powered Performance - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI16G - Newegg.com](https://www.newegg.com/abs-aqa14700kf4060ti16g-stratos-aqua/p/N82E16883360436): Buy ABS Aquilon Aqua Gaming PC - Windows 11 Home - Intel Core i7 14th Gen 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI-Powered Performance - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI...
- [Don't ask to ask, just ask](https://dontasktoask.com/): no description found
- [Civitai Link | One-click install Stable Diffusion models](https://civitai.com/product/link): Directly download any models from Civitai to your Stable Diffusion instance.
- [Update on SD3 on Civitai | Civitai](https://civitai.com/articles/5840/update-on-sd3-on-civitai): Standard disclaimer; This post does not constitute legal advice. How you interact with SAI and their product is up to you. You should seek your own...
- [Stable Diffusion 3](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3): no description found
- [stabilityai/stable-diffusion-3-medium · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium): no description found
- [SD3 IS HERE!! ComfyUI Workflow.](https://youtu.be/Di1KqPXxx2Y?si=NXtsTleKTijMBVJV): SD3 is finally here for ComfyUI!Topaz Labs: [https://topazlabs.com/ref/2377/HOW](https://topazlabs.com/ref/2377/HOW) TO SUPPORT MY CHANNEL-Support me by joining my Patreon: [https://www.patreon.co](https://www.patreon.co)...
- [Deep Learning Fundamentals - Lightning AI](https://lightning.ai/courses/deep-learning-fundamentals/): Deep Learning Fundamentals is a free course on learning deep learning using a modern open-source stack.
- [Introduction - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1): no description found
- [HuggingFace](https://www.youtube.com/@HuggingFace): HuggingFace is on a mission to solve Natural Language Processing (NLP) one commit at a time by open-source and open-science. Our youtube channel features tutorials and videos about Machine Learning, ...
- [Whose art is this, really? Inside Canadian artists’ fight against AI](https://www.thestar.com/news/canada/whose-art-is-this-really-inside-canadian-artists-fight-against-ai/article_54b0cb5c-7d67-5663-a46a-650b462da1ad.html): Visual artists’ work is being gathered online and used as fodder for computer imitations. When Toronto’s Sam Yang complained to an AI platform, he got an email he says was meant to taunt h...
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1253951731210125353)** (17 messages🔥):

- **Beginners questioning working group contributions**: A new member asked how to contribute to working groups, wondering if monitoring GitHub repositories is sufficient or if a more formal method exists.
- **Register usage in complex kernels**: A member shared debugging strategies for a kernel using too many registers per thread, suggesting either commenting out code parts or examining SASS in Nsight Compute.
- **Announcing CUTLASS working group**: A member proposed forming a working group to create learning materials for CUTLASS, inviting others to express interest and prepare by reviewing a [YouTube talk on Tensor Cores](https://youtu.be/hQ9GPnV0-50?feature=shared).
- **CPU cache insights**: A member shared a [CPU-centric guide on computer cache](https://pikuma.com/blog/understanding-computer-cache), emphasizing the importance of understanding cache for programmers.
  

**Links mentioned**:

- [Lecture 23: Tensor Cores](https://youtu.be/hQ9GPnV0-50?feature=shared)): Slides: https://drive.google.com/file/d/18sthk6IUOKbdtFphpm_jZNXoJenbWR8m/view?usp=drive_link
- [Exploring How Cache Memory Really Works](https://pikuma.com/blog/understanding-computer-cache): Even though we often hear terms like L1, L2, cache block size, etc., most programmers have a limited understanding of what cache really is. This is a beginner-friendly primer on how cache works.

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1253916276393250886)** (4 messages):

- **INT4 LoRA fine-tuning vs QLoRA**: A user inquired about the differences between **INT4 LoRA fine-tuning** and **QLoRA** in terms of accuracy and speed. Another member explained that QLoRA with HQQ involves frozen quantized weights, does not use tinnygemm, and utilizes dequantizing alongside *torch.matmul* due to inefficiencies in tinnygemm for large sequences.
- **Performance and Speed in QLoRA**: It's mentioned that **QLoRA** maintains good quality and fast performance, especially when a CUDA dequant kernel (axis=0) is implemented. A separate contribution was noted where a user created a fused GEMM for int4, which is effective for training with fixed sequence lengths, providing the fastest solution.

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1254356369302360109)** (1 messages):

- **Measure Bandwidth, Throughput, Latency with NVIDIA tools**: A member shared a detailed [GitHub guide](https://github.com/CisMine/Guide-NVIDIA-Tools/tree/main/Chapter09) on how to measure **bandwidth, throughput, and latency** using NVIDIA tools. The guide provides step-by-step instructions contributing to better performance analysis and optimization.

**Link mentioned**: [Guide-NVIDIA-Tools/Chapter09 at main · CisMine/Guide-NVIDIA-Tools](https://github.com/CisMine/Guide-NVIDIA-Tools/tree/main/Chapter09): Contribute to CisMine/Guide-NVIDIA-Tools development by creating an account on GitHub.

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1254368386608271361)** (1 messages):

- **Internship Seeker with AI and CUDA Skills**: A member from **VietNam** seeks a remote internship in AI and CV focusing on CUDA optimization. They shared their experience and two GitHub repositories: [Parallel-Computing-Cuda-C](https://github.com/CisMine/Parallel-Computing-Cuda-C) and [Guide-NVIDIA-Tools](https://github.com/CisMine/Guide-NVIDIA-Tools).

**Link mentioned**: [GitHub - CisMine/Parallel-Computing-Cuda-C](https://github.com/CisMine/Parallel-Computing-Cuda-C): Contribute to CisMine/Parallel-Computing-Cuda-C development by creating an account on GitHub.

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1254771364288532520)** (3 messages):

- **Seeking AI/ML Fundamentals**: A member asked for recommendations on good courses for learning fundamentals in **AI/ML** on platforms like Coursera. Another member inquired about their background in programming, computer science, or math to suggest appropriate resources.

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1253934851313107067)** (28 messages🔥):

- **Precision Loss in FP8 Conversion Discussed**: Members discussed how **PyTorch follows the IEEE convention** for rounding in FP8 conversions, addressing precision loss and suggesting that scaling tensors could minimize this loss. One member mentioned that scaling ensures more effective use of the GPU's range ([link](https://github.com/pytorch/pytorch/blob/f42d5b6dca75ee020355fc75532347ca2734b117/c10/util/Float8_e4m3fn.h#L46)).
- **Floating-Point Precision Explained**: Floating-point precision issues were a hot topic, and a member shared the [floating-point-gui.de](https://floating-point-gui.de/) as a resource for understanding unexpected precision errors in numerical outputs.
- **Scaling for FP8 Precision**: Several members debated how to determine **scaling factors** for tensor conversion to FP8, with some suggesting to base it on min/max values or other metrics to avoid overflow and underflow ([link](https://gist.github.com/drisspg/64600f98c4a0cb41917afe81e757469e)).
- **Quantization Learning Resources Shared**: For those looking to understand quantization better, members recommended various resources including a [GitHub list of papers](https://github.com/cuda-mode/awesomeMLSys) and educational YouTube videos ([Quantization explained](https://youtu.be/0VdNflU08yA) and [Advanced Quantization](https://youtu.be/1u9xUK3G4VM)).
- **FP8 Scaling Updates**: One member mentioned recent updates to PyTorch, now supporting **row-wise scaling** for FP8 conversion and hinted at upcoming posts for community discussion.
  

**Links mentioned**:

- [Scaled_FP8.md](https://gist.github.com/drisspg/64600f98c4a0cb41917afe81e757469e): GitHub Gist: instantly share code, notes, and snippets.
- [Quantization explained with PyTorch - Post-Training Quantization, Quantization-Aware Training](https://youtu.be/0VdNflU08yA?feature=shared): In this video I will introduce and explain quantization: we will first start with a little introduction on numerical representation of integers and floating-...
- [Lecture 7 Advanced Quantization](https://youtu.be/1u9xUK3G4VM?feature=shared): Slides: https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&dl=0
- [GitHub - cuda-mode/awesomeMLSys: An ML Systems Onboarding list](https://github.com/cuda-mode/awesomeMLSys): An ML Systems Onboarding list. Contribute to cuda-mode/awesomeMLSys development by creating an account on GitHub.
- [pytorch/c10/util/Float8_e4m3fn.h at f42d5b6dca75ee020355fc75532347ca2734b117 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/f42d5b6dca75ee020355fc75532347ca2734b117/c10/util/Float8_e4m3fn.h#L46): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [Float stored in 8 bits - ONNX 1.17.0 documentation](https://onnx.ai/onnx/technical/float8.html#cast): no description found
- [The Floating-Point Guide - What Every Programmer Should Know About Floating-Point Arithmetic](https://floating-point-gui.de/): no description found
- [Visualising ML number formats](https://thecharlieblake.co.uk/visualising-ml-number-formats): A visualisation of number formats for machine learning --- I couldn’t find any good visualisations of machine learning number formats online, so I decided to make one. It’s interactive, and hopefully ...
- [float8_experimental/float8_experimental/float8_utils.py at d4ade877dff327ea7f51e91f7cc218ae956e8cfd · pytorch-labs/float8_experimental](https://github.com/pytorch-labs/float8_experimental/blob/d4ade877dff327ea7f51e91f7cc218ae956e8cfd/float8_experimental/float8_utils.py#L142): This repository contains the experimental PyTorch native float8 training UX - pytorch-labs/float8_experimental
- [float8_experimental/test/test_base.py at d4ade877dff327ea7f51e91f7cc218ae956e8cfd · pytorch-labs/float8_experimental](https://github.com/pytorch-labs/float8_experimental/blob/d4ade877dff327ea7f51e91f7cc218ae956e8cfd/test/test_base.py#L86-L111): This repository contains the experimental PyTorch native float8 training UX - pytorch-labs/float8_experimental

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1254565529515982928)** (18 messages🔥):

- **Valorant account locked for associating with a cheater**: A user's friend got her Valorant account locked for 180 days because she queued with someone who was cheating. *"I told her to go through support but she's getting desperate so I figured it was worth mentioning."*
- **Anxiety over account lock**: The friend was anxious and only waited an hour for support before seeking further help. *"I told her to wait for now."*
- **Region and details provided**: The user mentioned that the affected friend is located in California and plays Valorant. *"She's in California, she just told me."*
- **Response from support query**: A respondent mentioned the possibility of looking into the issue but noted that there might not be much they can do. *"I think the answer is 'nothing really' LOL"*
- **Replay review and appropriate bans**: Assurance was given that replays would be watched to make sure bans are appropriate. *"They'll watch the replay and do the bans appropriately though!"*

---

### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1254763687831535616)** (2 messages):

- **Running `torchao_int4_demo.py` produces nonsense output**: One member reported getting meaningless output like *"Unterscheidung Hinweis Unterscheidung Einzeln Unterscheidung Unterscheidung ..."* when trying to run `torchao_int4_demo.py`. They mentioned the only change was *"setting `compile=None`"* and sought help from another member who inquired if the issue occurs with all models and suggested trying with `'axis=0'`.

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1253786744155930655)** (465 messages🔥🔥🔥):

- **Plan for NCCL Initialization**: A member proposed a plan to use **MPI** to initialize **NCCL** and fallback to the file system or TCP sockets if MPI is unavailable. They aimed to keep GPU computations in CUDA to ensure stability and performance.
- **H100 vs A100 Training Stability**: Members discussed the instability in the training on **H100 GPUs** compared to **A100 GPUs**, with **H100** experiencing "exploding" gradients around 28K steps. One suggested copying computations to GPU to avoid this issue.
- **CUDA and Multi-node Setup**: Significant efforts were made to test multi-node setups using different methods such as **MPI**, **slurm**, and TCP sockets. The discussions included refinements necessary to ensure all nodes work well together without significant overhead.
- **Integrating FP8 Matmuls**: A member described integrating **FP8 matmuls** and observed marginal performance increases. They shared detailed challenges and strategies related to FP8 tensor cores and optimizing rescaling and transposing operations.
- **Preparation for Cluster Training**: Plans were discussed to try training large language models on a new **Lambda cluster**, aiming to complete significant training milestones faster. This included ensuring cost efficiency and verifying the stability of the training runs on different hardware setups.
  

**Links mentioned**:

- [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380): Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance o...
- [WIP Distribution Visualisation to help with FP8 work & beyond by ademeure · Pull Request #618 · karpathy/llm.c](https://github.com/karpathy/llm.c/pull/618): Not ready for integration at all / still very hacky, bunch of unsolved issues I am not sure where code should go etc.: need to find a way to make it pollute the code less with all of those generat...
- [Socket server/client interface by chinthysl · Pull Request #633 · karpathy/llm.c](https://github.com/karpathy/llm.c/pull/633/files#diff-6b403830ffefa78e4ce238b4ab24bb0624dfede82fe2a5214ee63e2cfda07a19): Dummy PR to make use of the distributed interface in PR #632
- [FlexNet 11.19.5 build on Visual Studio 2015](https://community.flexera.com/t5/FlexNet-Publisher-Forum/FlexNet-11-19-5-build-on-Visual-Studio-2015/m-p/306967): Hi all,   I am trying to build my app with FlexNet 11.19.5. I am facing some compiler issues (Visual Studio 2015):c:\\program files (x86)\\windows kits\\8.1\\include\\shared\\ws2def.h(100): warning C4005: '...
- [MPI/TCP/FS for NCCL-init by gordicaleksa · Pull Request #632 · karpathy/llm.c](https://github.com/karpathy/llm.c/pull/632): Instead of mixing NCCL & Open MPI during training: let's transition to using only NCCL. To the best of my knowledge there are no downsides here, they're equivalent and speedwise i couldn&#...

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1254425625394020453)** (2 messages):

- **PCIe limitations discussed**: Members discussed how **PCIe has power, weight, and pin limits** when it comes to communication. One member noted that the main reason for not creating lower-spec products is focus on selling high-end servers which are more profitable.
- **Big players targeted**: Another member speculated that the company is primarily targeting **big players like cloud GPU providers.** This aligns with their current product strategy which maximizes revenue.

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1254075604400341034)** (25 messages🔥):

- **Debugging Bitnet Tensor Issue**: Members faced an issue with Bitnet tensors while running a trainable network, encountering an error due to a dimension not divisible by 4. An error traceback was shared indicating an `AssertionError` caused by Bitnet dispatch attempting an unsupported `aten.to.dtype_layout` operation.
- **Updated Test Script and Repo Link**: An updated test script was linked to [CoffeeVampir3's GitHub](https://github.com/CoffeeVampir3/ao-bitnet/blob/main/bitnet_staging/bitnet_trained_to_ao_test.py) to use the new library paths. CoffeeVampir3 also shared the main repository link [here](https://github.com/CoffeeVampir3/ao-bitnet/tree/main).
- **Affine Quantization Discussion**: Vayuda and Jerry discussed the potential integration of Bitnet tensors into AffineQuantizedTensor, considering creating a new layout for packed tensors which would indicate the currently packed dimension. Jerry emphasized that bit (uint1) tensors should remain separate but compatible with affine quantized tensors.
- **Seeking Assistance and Minimal Repro Request**: Marksaroufim requested a minimal reproducible example to debug the dtype conversion issue in Bitnet tensors. CoffeeVampir3 provided the link to the test script to facilitate this debugging process.
- **New Tutorials and Tensor Subclassing Ideas**: Marksaroufim suggested new tutorials on the [PyTorch ao library](https://github.com/pytorch/ao/issues/426), highlighting the library's potential to handle quantized optimizers and kv caches. Gau.nernst and Vayuda discussed the absence of progress on fp5 and the potential interest in integrating 8-bit Adam with tensor subclasses.

**Link mentioned**: [The next tutorials · Issue #426 · pytorch/ao](https://github.com/pytorch/ao/issues/426): From our README.md torchao is a library to create and integrate high-performance custom data types layouts into your PyTorch workflows And so far we've done a good job building out the primitive d...

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1253787640977817662)** (312 messages🔥🔥):

- **GPU VRAM limits model capabilities**: Discussions highlighted limitations in loading large models like **Command R (34b) Q4_K_S** on GPUs with limited VRAM, resulting in reduced token context windows and hindered usability. Various members recommended looking into alternative formats like **EXL2** which are more VRAM-efficient for models.
- **Interest in server setup and headless operation**: Users expressed interest in running **LM Studio** on remote servers and headless setups for better hardware utilization. Suggestions included exploring **llama.cpp** for server setups and noting that **LM Studio** does not support direct remote or headless operations.
- **Text-to-text dominant focus and model customization**: Members discussed the limited capabilities of **LM Studio** to only handle text-to-text interactions, with no support for image generation or text-to-speech features. Some users mentioned alternative frontends like **SillyTavern** but acknowledged its RP/character focus, highlighting the need for more versatile options.
- **Optimizing cooling for P40 GPUs**: There were troubleshooting tips shared on GPU cooling, especially around P40 GPUs. Users noted the importance of adequate cooling solutions and shared experiences like crafting custom air ducts to manage GPU temperatures more effectively.
- **Exploring various language models for coding**: Discussions involved finding the best language models for coding tasks, with mentions of models like **Codestral 22B**. Members highlighted the importance of model size and quantization, recommending **Q5** or **Q6** quants for optimal performance given specific hardware constraints.
  

**Links mentioned**:

- [README.md · artificialguybr/ColoringBookRedmond-V2 at main](https://huggingface.co/artificialguybr/ColoringBookRedmond-V2/blob/main/README.md): no description found
- [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394\): This paper introduces the MCT Self-Refine (MCTSr) algorithm, an innovative integration of Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS), designed to enhance performance in complex m...
- [GitHub: Let’s build from here](https://github.com/): GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...
- [bartowski/Codestral-22B-v0.1-GGUF · Hugging Face](https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF): no description found
- [Confused Computing GIF - Confused Computing Counting - Discover & Share GIFs](https://tenor.com/view/confused-computing-counting-math-problems-gif-14678592): Click to view the GIF
- [configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md): LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs
- [GitHub - theroyallab/tabbyAPI: An OAI compatible exllamav2 API that's both lightweight and fast](https://github.com/theroyallab/tabbyAPI): An OAI compatible exllamav2 API that's both lightweight and fast - theroyallab/tabbyAPI

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1253881658881609759)** (116 messages🔥🔥):

- **Hermes 2 Theta Llama-3 amazed users**: Members praised the **Hermes 2 Theta Llama-3 70B** model for its ability to remember context up to 19k tokens and effectively follow instructions. One member shared that it might be their top model now due to its deep reasoning and creative capabilities in role-play scenarios. [Hermes 2 Theta Llama-3](https://huggingface.co/OpenPipe/Hermes-2-Theta-Llama-3-70B-32k).
- **DeepSeek Coder V2 gains popularity**: Users discussed the performance and prompt issues of the **DeepSeek Coder V2** model, recommending using specific prompt presets to avoid unexpected output in Chinese. One user highlighted how this model outperformed GPT4o for tasks related to C# coding. [DeepSeek Coder V2](https://chat.deepseek.com/coder).
- **Llama 3 CursedStock models intrigue**: Members expressed curiosity and amusement at the unusual naming and performance of **Llama 3 CursedStock V1.8-8B**, sharing that it fits its quirky name by merging uncensored models. There were also discussions about how well it performs in niche roles such as specific story-writing and generating creative content. [Llama-3 CursedStock V1.8-8B](https://huggingface.co/PJMixers/LLaMa-3-CursedStock-v1.8-8B).
- **Concerns over Temporal Awareness in LLMs**: There was a debate about LLMs' inability to handle tasks that require temporal awareness and cause-and-effect reasoning. Users acknowledged the limitations of current AI, emphasizing the need for specialized hardware to achieve genuine general intelligence.
- **Experimenting with Quantized Models**: Users shared experiences with different quantized models like Q6_K_L and Q8, noting issues with certain builds in handling large context sizes. They also discussed the potential benefits of keeping output tensors and embeddings unquantized for better performance, particularly with the **Hathor Fractionate-L3-8B** model. [Hathor Fractionate-L3-8B](https://huggingface.co/Nitral-AI/Hathor_Fractionate-L3-8B-v.05).
  

**Links mentioned**:

- [DeepSeek](https://chat.deepseek.com/coder): Chat with DeepSeek AI.
- [PrunaAI/cognitivecomputations-Dolphin-2.9.1-Phi-3-Kensho-4.5B-GGUF-smashed · Hugging Face](https://huggingface.co/PrunaAI/cognitivecomputations-Dolphin-2.9.1-Phi-3-Kensho-4.5B-GGUF-smashed): no description found
- [cognitivecomputations/Dolphin-2.9.1-Phi-3-Kensho-4.5B · Hugging Face](https://huggingface.co/cognitivecomputations/Dolphin-2.9.1-Phi-3-Kensho-4.5B): no description found
- [mradermacher/New-Dawn-Llama-3-70B-32K-v1.0-GGUF · Hugging Face](https://huggingface.co/mradermacher/New-Dawn-Llama-3-70B-32K-v1.0-GGUF): no description found
- [meta-llama/Meta-Llama-3-8B · Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B): no description found
- [PJMixers/LLaMa-3-CursedStock-v1.8-8B · Hugging Face](https://huggingface.co/PJMixers/LLaMa-3-CursedStock-v1.8-8B?not-for-all-audiences=true): no description found
- [Flash Thumbs Up GIF - Flash Thumbs Up Way To Go - Discover & Share GIFs](https://tenor.com/view/flash-thumbs-up-way-to-go-good-gif-22860767): Click to view the GIF
- [GitHub - RJ-Flash/AGI-Project: The AGI Project aims to develop an Artificial General Intelligence (AGI) system capable of understanding, learning, and applying knowledge across a wide range of tasks at a level comparable to human intelligence. Our goal is to create a system that can perform any intellectual task that a human being can do, with the ability to learn and adapt.](https://github.com/RJ-Flash/AGI-Project): The AGI Project aims to develop an Artificial General Intelligence (AGI) system capable of understanding, learning, and applying knowledge across a wide range of tasks at a level comparable to huma...
- [mradermacher/Hermes-2-Theta-Llama-3-70B-32k-i1-GGUF at main](https://huggingface.co/mradermacher/Hermes-2-Theta-Llama-3-70B-32k-i1-GGUF/tree/main): no description found
- [Nitral-AI/Hathor_Fractionate-L3-8B-v.05 · Hugging Face](https://huggingface.co/Nitral-AI/Hathor_Fractionate-L3-8B-v.05): no description found
- [bartowski/Hathor_Stable-L3-8B-v0.5-GGUF · Hugging Face](https://huggingface.co/bartowski/Hathor_Stable-L3-8B-v0.5-GGUF): no description found
- [TheDrummer (Drummer)](https://huggingface.co/TheDrummer/): no description found
- [mradermacher/Halu-8B-Llama3-Blackroot-GGUF · Hugging Face](https://huggingface.co/mradermacher/Halu-8B-Llama3-Blackroot-GGUF/): no description found
- [mradermacher/Mistral-7B-Erebus-v3-i1-GGUF · Hugging Face](https://huggingface.co/mradermacher/Mistral-7B-Erebus-v3-i1-GGUF/): no description found
- [OpenPipe/Hermes-2-Theta-Llama-3-70B-32k · Hugging Face](https://huggingface.co/OpenPipe/Hermes-2-Theta-Llama-3-70B-32k): no description found

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1253883521303314532)** (4 messages):

- **DeepseekV2 Chat Loading Issues**: One user mentioned that **deepseekV2** cannot be loaded for chat. Another noted that **V0.2.25** is required and "auto update currently broken".
- **Multi-Model Sequence Proposal**: A member proposed a feature for **Multi-model** setups to "build a sequence map for models" allowing one model to feed information into two parallel models, which then feed into a final model.
- **Ubuntu LM Studio Network Error**: **LM Studio** on Ubuntu 22.04 gets a "network error" when trying to search models on Hugging Face. However, the member noted it still works on Mac M1 and the issue appeared after commenting out the ser2net config file for port 3001, used by AnythingLLM web server.

---

### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1254859928183636009)** (9 messages🔥):

- **Estimating the AI setup cost stumps users**: A member asked about the budget to set up a machine with the performance of GPT or Bard. Responses indicated that the cost is extremely high, potentially thousands of dollars, depending on the configuration, and not feasible for a typical user.
- **NVIDIA DGX GH200 is highlighted**: A link to the [NVIDIA DGX GH200](https://www.nvidia.com/en-gb/data-center/dgx-gh200/) was shared, noting that it is used by OpenAI and features large memory capacities designed to handle terabyte-class models. Another member humorously remarked that such setups are out of reach for most people's budgets.

**Link mentioned**: [NVIDIA DGX GH200](https://www.nvidia.com/en-gb/data-center/dgx-gh200/): Massive memory supercomputing for emerging AI

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1253836060459274280)** (18 messages🔥):

- **NVlink's absence limits 4000 series GPUs**: A member questioned whether the absence of NVlink in 4000 series GPUs would hinder using multiple GPUs for AI purposes. They also queried the potential use of **DX or Vulkan multi-GPU features** as alternatives.
- **Performance on Nvidia P40s in Proxmox setup**: A user discussed their new setup with two Nvidia P40s in a server running Proxmox and Debian. They noted power utilization spiked significantly when using Codestral for full GPU offload, achieving 12 tokens/second.
- **ROCm 6.1.3 supports multi-GPU**: It was shared that AMD released **ROCm 6.1.3**, which now supports multi-GPU for high-end RDNA3 cards.
- **Debate on 16GB RAM for iPad Pro**: There was a debate on whether the 16GB RAM version of the **iPad Pro** is necessary for running large AI models. One member highlighted that quantized models can fit into 16GB on their RTX 4070 Ti Super, but was unsure if this would apply to Apple's hardware.
- **Corsair PSU and storage purchase query**: A user inquired if purchasing a **Corsair AX1600i for €266 and 4 Exos Enterprise 18TB drives for €668** was worth it, receiving no specific feedback.

---

### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1253935277769228300)** (3 messages):

- **Llama.cpp model loading error**: One member reported a **"wrong number of tensors"** issue with the error message `'done_getting_tensors: wrong number of tensors; expected 356, got 291'` while loading the **Blombert 3B f16 gguf** model. Another suggested the error is due to **llama.cpp** version incompatibility with LM Studio.
- **Context length troubleshooting advice**: A common issue with large models such as **Blombert 3B** was discussed, attributing errors to mismatched context lengths. *"Keep ratcheting the context length down until it doesn't lose its' mind,"* was advised as a possible solution.

---

### **LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/)** (1 messages):

cdrivex4: Yes ok.. Sounds like fun

---

### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1254635814118490143)** (1 messages):

- **Qwen2 500M Model Quantization Update**: The latest quantized versions of the **Qwen2 500M** model have been published. These models are optimized for **speedy generation** and can even be deployed on lightweight compute machines like a **Raspberry Pi**. Explore the models [here](https://huggingface.co/lmstudio-community/Qwen2-500M-Instruct-GGUF).

---

### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1254225652538933379)** (12 messages🔥):

- **Model loading issues frustrate user**: One user struggled with loading their model using LMS with a batch script but eventually succeeded. They asked for feedback on their batch script to check for mistakes or streamlining opportunities.
- **LMStudio is not open source**: A user inquired whether LMStudio is open source and if it could be extended. Another member clarified that it is not open source, leading the user to consider developing their own tools to achieve desired functionalities.
- **Dreams of an all-in-one model runner**: A discussion touched on the desire for a program capable of running various models from Huggingface, including text to speech, text to image, and more. No existing solution was known, but there was interest in such a project.

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1253792638432182303)** (276 messages🔥🔥):

- **GPT-5 Anticipation Builds**: Users expressed frustration at OpenAI's delayed feature rollouts, with voice mode and GPT-4 Vision being repeatedly mentioned as overdue. A member stated, *"at this point i don't even care when it comes it comes, and ill use it but meh thats just me ofcourse."*
- **Siri and ChatGPT Integration Debate**: Confusion arose over whether ChatGPT is integrated into Siri, with one member clarifying, "no its just like a bonus its not exactly integrated where its reliant on it". Elon Musk's criticism of the integration also sparked conversation.
- **Claude vs ChatGPT Performance**: Many users discussed the superiority of **Claude 3.5 Sonnet** over **GPT-4o**, especially in coding, with one saying, "same things i tried in 4o and where it failed, claude 3.5 did it successfully and more". Benchmarks and specific features like Claude's "artifacts" were frequently mentioned as evidence.
- **AI Model Economics and Token Limits**: Discussions highlighted comparative aspects of various AI models, including **Claude’s 200k tokens** versus **ChatGPT’s 128k for GPT-4** and 32k for Plus users. One user noted, "Claude 3.5 Sonnet is on the LMSYS leaderboard," emphasizing practical performance over pure benchmarks.
- **Persistent Use-Cases for LLMs**: A user inquired about how to create a persistent LLM trained on personal documents, asking, "Is there a way to essentially hyper focus one of these LLMs like sonnet 3.5, or gemini 1.5 pro, etc and use personally as my own work-bot?" This sparked significant interest around the potential for customized, long-term AI applications.
  

**Links mentioned**:

- [Wired: AI startup Perplexity is 'BS machine'](https://www.youtube.com/watch?v=MFdjEW8_SUg): Katie Drummond, Wired’s global editorial director, joins 'Squawk Box' to discuss the magazine's investigation into AI search startup Perplexity.
- [Computer Stick Man GIF - Computer Stick Man Table Flip - Discover & Share GIFs](https://tenor.com/view/computer-stick-man-table-flip-look-at-you-9gag-gif-26257332): Click to view the GIF

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1253827661143212032)** (29 messages🔥):

- **GPT-4o connectivity issues resolved**: Multiple users reported encountering an error message on GPT-4o stating, *"An error occurred connecting to the worker,"* but it was resolved after a short period. One user confirmed, *"seems for me its back working now."*
- **Screen sharing feature has no ETA**: A user inquired about the availability of a screen-sharing feature, to which another user responded that there is no estimated time of arrival (ETA) yet.
- **GPT-4o prompt adherence problems**: Users discussed issues with GPT-4o where it fails to stick to specified prompt formats and instructions consistently. For instance, it often outputs in markdown despite clear instructions for HTML, and it misinterpreted structured review instructions by reviewing entire documents at once.
- **ChatGPT's slow performance and crashes**: Users experienced slow performance and frequent crashes while using ChatGPT. One remarked, *"yeah, its crashing frequently here too."*
- **Document length and GPT context window limitations**: A user with 1200-page documents faced issues with GPT accurately processing content. Another user explained that ChatGPT’s context window is not sufficient for such large documents and recommended tools like Gemini and Claude for larger token windows.

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1253787018446504020)** (53 messages🔥):

- **Members discuss background removal limitations**: A member mentioned that *DALL-E only edits its own generations* and that ChatGPT offers some image editing capabilities like generating Python scripts for tasks, but struggles with *background removal*. Another member suggested trying *online services* for background removal.
- **Eager anticipation for Sora launch**: A user expressed excitement about Sora's launch, asking for updates. Another member shared that there is no timeline yet but linked to a [Sora video generated on the server](https://discord.com/channels/974519864045756446/1238697668646010910/1240586938273103894).
- **Creation of fantasy movie plots with AI**: A member excitedly shared their *fantasy movie ideas* being developed with ChatGPT, including a reimagining of *The Wizard of Oz*. They discussed the use of *DALLE* to visualize their ideas.
- **Troubleshooting ChatGPT's capabilities**: Users were troubleshooting ChatGPT's *image background removal* skills, noting that while it attempts with basic coding, it runs into *memory allocation issues* with more complex tasks like using the "Deeplab model". The discussion included insights on modifying behavior by adjusting *custom instructions*.
- **Interactive prompts and optimizing responses**: A member shared a detailed interactive prompt for building a PC on a budget, and another sought advice on prompts related to *cryptocurrency*. Additionally, there was interest in improving MyGPT prompts for better response accuracy and reliability, especially in extracting topics and processing uploaded files.

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1253787018446504020)** (53 messages🔥):

- **Background removal: Dream or reality?**: Members discussed attempts to get ChatGPT to perform background removal on images. Despite ChatGPT generating scripts to try this, results were inconsistent due to memory allocation issues when using advanced machine learning tools.
- **Sora launch anticipation grows**: New users expressed excitement and impatience for the launch of Sora. A member shared a link to a video of a Sora event that generated some buzz on the server.
- **DALL-E vs. Midjourney for artworks**: Members debated the effectiveness of DALL-E 3 compared to Midjourney for creating AI images, especially for paint-like images. Personal preferences leaned towards DALL-E 3 for its specific artistic styles.
- **Fantasy movies and prompt crafting**: A user shared their experience using ChatGPT to create movie ideas, specifically a reimagination of "The Wizard of Oz". They sought advice on refining prompts for more accurate and vivid image generation.
- **Interactive PC building prompts**: A member showcased a creative interactive prompt designed to help users build PCs within a specified budget, incorporating web searches for affordable components and tracking the project's progress using Python.

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1253789650942951627)** (381 messages🔥🔥):

- **Wired slams Perplexity for plagiarism**: A Wired article accused Perplexity AI of "surreptitiously scraping" websites, violating its own policies. Users discussed it, with some finding the backlash excessive considering AI's common practices with data summarization ([source](https://www.wired.com/story/perplexity-plagiarized-our-story-about-how-perplexity-is-a-bullshit-machine/)).
- **Legal perspectives on AI summarization**: Redditors discussed the legal risks of AI summarizing articles inaccurately and potentially making defamatory statements. A Wired observation highlighted Perplexity’s chatbot falsely attributing a crime to a police officer despite linking to the source ([archive link](http://archive.today/GNgAe)).
- **Claude 3.5 Sonnet rollout**: Perplexity Pro members noted the recent addition of the Claude 3.5 Sonnet model. Initial reactions praised its capabilities but some users criticized it for being overly cautious and limiting ([Forbes Article](https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/)).
- **User frustrations and platform reliability**: Several users reported issues with Perplexity, including inconsistencies in Pro search results and login problems on the mobile app. One user expressed significant dissatisfaction with the functionality and restriction levels of Claude 3.5 Sonnet.
- **Pro search and model usage insights**: Discussions revealed frustrations with changes in Pro search's effectiveness and source limits, with users suggesting Perplexity prioritizes partnerships over core improvements. A user noted that Claude's API subscription provides more value compared to competitors ([related video](https://www.youtube.com/watch?v=iDlM0cYS9Zs)).

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://www.forbes.com/sites/randalllane/2024/06/11/why-perplexitys-cynical-theft-represents-everything-that-could-go-wrong-with-ai/">Why Perplexity’s Cynical Theft Represents Everything That Could Go Wrong With AI</a>: It’s the perfect case study for this critical moment: AI is only as good as the people overseeing it.</li><li><a href="https://www.wired.com/story/perplexity-plagiarized-our-story-about-how-perplexity-is-a-bullshit-machine/">Perplexity Plagiarized Our Story About How Perplexity Is a Bullshit Machine</a>: Experts aren’t unanimous about whether the AI-powered search startup’s practices could expose it to legal claims ranging from infringement to defamation—but some say plaintiffs would have strong cases...</li><li><a href="https://tenor.com/view/just-when-i-thought-i-was-out-they-pull-me-back-in-michael-corleone-al-pacino">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=iDlM0cYS9Zs">I installed Android on Rabbit R1 &amp; Made it Useful</a>: I managed to install Android 13 onto the Rabbit R1 and it made the device a lot more useful! Letting me download apps, send messages, and a lot more. Here's ...</li><li><a href="https://x.com/roramora0/status/1804604063922655743">Tweet from Cubicle e/acc (@roramora0)</a>: @dwarkesh_sp Dwarkesh, if you're in touch with Anthropic please notify them that their recaptcha-en.js file has a security loophole that allows mouse action simulation using js code. this allowed ...</li><li><a href="https://www.reddit.com/user/No_Dragonfruit_5472/comments/1chdemx/tradingview_premium_pack_crack_2024_version_free/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/view/just-when-i-thought-i-was-out-they-pull-me-back-in-michael-corleone-al-pacino-the-godfather-gif-19249100">Just When I Thought I Was Out They Pull Me Back In GIF - Just When I Thought I Was Out They Pull Me Back In Michael Corleone - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.goodnewsnetwork.org/robot-mimics-human-sense-of-touch-to-better-sort-through-litter/">Robot Mimics Human Sense of Touch to Better Sort Through Litter</a>: The authors explain that human touch has many layers of sensory perception, including changes in how different temperatures feel.</li><li><a href="https://chromewebstore.google.com/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo?hl=en">Perplexity - AI Companion</a>: Ask anything while you browse</li><li><a href="https://www.goodnewsnetwork.org/robot-mimics-human-sense-of-touc">Robot Mimics Human Sense of Touch to Better Sort Through Litter</a>: The authors explain that human touch has many layers of sensory perception, including changes in how different temperatures feel.</li><li><a href="http://archive.today/GNgAe">Perplexity Plagiarized Our Story About How Perplexity Is a Bullshit M…</a>: no description found</li></ul></div>

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1253940127986745426)** (12 messages🔥):

- **Discover Apple AI Delayed in Europe**: Members shared a page discussing **Apple's AI capabilities and their limitations** in the European region. For more details, check out [Apple Intelligence Isn't](https://www.perplexity.ai/page/Apple-Intelligence-Isnt-KJfiVRPEQMmkim0gv7Xh7w).
- **Perplexity Search and Learning**: Multiple members shared their unique searches on Perplexity AI, indicating its diverse usage for learning and information-gathering. Notable searches included topics like [AI improvements](https://www.perplexity.ai/search/AI-Y9Ao26a2SquKulTrvmGfLg#0) and [language exploration](https://www.perplexity.ai/search/let-words-spray-vxBv1ca.QbmnB.oMSR.2Jw).
- **Boeing's Starliner Issues**: Two members highlighted an article on Perplexity AI about **Boeing's Starliner facing challenges**. Read more via [Boeing’s Starliner Stuck](https://www.perplexity.ai/page/Boeings-Starliner-Stuck-lIlR4mleQUK1Q0kahpVwRQ).
- **OpenAI Community Message**: A community message advised members to ensure their threads are shareable for better community engagement. Read the full advisory [here](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **YouTube Educational Content**: Perplexity AI shared an upcoming YouTube video, hinting at important topics like **Starliner issues, Apple AI in Europe, OpenAI's acquisition**, and more. Watch the preview [here](https://www.youtube.com/embed/xUsxGDrwzls).

**Link mentioned**: [YouTube](https://www.youtube.com/embed/xUsxGDrwzls): no description found

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1253805803740336249)** (12 messages🔥):

- **Looking for project ideas**: A user is seeking **interesting projects** to build using the API and resources to understand *what is being done and what is possible*.
- **LLama-3-70B API context length confusion**: A user noted a **connection error** when total tokens exceed around 1642, while another user reported success with a nearly 3000-token request. Possible **moderation trigger or technical issue** is suspected.
- **Perplexity summarization navigates hyperlinks**: When asking Perplexity to summarize a webpage via a link, it navigates through hyperlinks from the provided link. The user is looking for a way to restrict summarization to the initial URL.
- **Inquiry on citations time filter in API**: A user asked if there is a **time filter for citations** for online models via API, noting the presence of some undocumented request parameters. The user does not have beta access but has requested it.

**Link mentioned**: [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions): no description found

---

### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1254516705778995200)** (20 messages🔥):

- **Rensa boosts dataset deduplication**: A member introduced **Rensa**, a high-performance **MinHash implementation in Rust with Python bindings**, showcasing features like FxHash, LSH index, and on-the-fly permutations. They claimed it is 2.5-3x faster than **datasketch** and shared it on [GitHub](https://github.com/beowolx/rensa).
- **Claude's odd reaction to The Cyberiad**: Members discussed the AI Claude producing a sonnet break when asked about The Cyberiad. One participant shared a prompt that caused this and suggested that *"<meta_start>"</meta_start>* could be a glitch token.
- **Glitch token research shared**: During the discussion on Claude's behavior, a member shared **arXiv articles** on glitch tokens for further reading: [article 1](https://arxiv.org/pdf/2404.09894) and [article 2](https://arxiv.org/pdf/2405.05417).
- **Sonnet's reluctance on tech topics**: A member observed that the AI model was frequently refusing requests related to tech news and machine merging. Another member humorously remarked that the sensitivity to AI-related questions seems heightened.
- **Critical view on ChatGPT paper**: A link to a critique of the "ChatGPT is bullshit" paper was shared, arguing against the paper's point that LLMs produce deceptive and truth-indifferent outputs. The critique is available on [Substack](https://spacedogchronicles.substack.com/p/nothing-is-an-absolute-reality-all?r=2cp9ad).
  

**Links mentioned**:

- [Nothing is an absolute reality, all is permitted](https://spacedogchronicles.substack.com/p/nothing-is-an-absolute-reality-all?r=2cp9ad): What is truth in the age of machine learning?
- [GitHub - beowolx/rensa: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets](https://github.com/beowolx/rensa): High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets - beowolx/rensa

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1253821344097898671)** (9 messages🔥):

- **Hackers jailbreak AI models**: Shared a tweet about hackers "jailbreaking" powerful AI models to highlight their flaws. The detailed article can be found [here](https://on.ft.com/45ByjEj).
- **GitHub's smol q* implementation**: Mention of a GitHub repository [ganymede](https://github.com/EveryOneIsGross/ganymede), which is a "smol implementation of q*". It's a resource for those interested in a hacky q\* star implementation with qwen 0.5b.
- **Game made from "Claude thingy"**: A member shared a link to a game they made, available on [Replit](https://replit.com/@0xjzy/SkeletalExpensiveEmbeds).
- **LLM inference in a font**: Described [llama.ttf](https://fuglede.github.io/llama.ttf/), a font file that's also a large language model and an inference engine. Explanation involves using HarfBuzz's Wasm shaper for font shaping, allowing for complex LLM functionalities within a font.
- **Tweet link by mautonomy**: Shared a Twitter link without additional context. The tweet can be found [here](https://twitter.com/agi2025/status/1798905521334010193?s=19).
  

**Links mentioned**:

- [llama.ttf](https://fuglede.github.io/llama.ttf/): no description found
- [Tweet from Financial Times (@FT)](https://x.com/FT/status/1804009458613326282): Hackers ‘jailbreak’ powerful AI models in global effort to highlight flaws https://on.ft.com/45ByjEj
- [SkeletalExpensiveEmbeds](https://replit.com/@0xjzy/SkeletalExpensiveEmbeds): Run Python code live in your browser. Write and run code in 50+ languages online with Replit, a powerful IDE, compiler, & interpreter.
- [GitHub - EveryOneIsGross/ganymede: smol implementation of q\*](https://github.com/EveryOneIsGross/ganymede): smol implementation of q\*. Contribute to EveryOneIsGross/ganymede development by creating an account on GitHub.

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1253789370444550296)** (278 messages🔥🔥):

- **Link for the bloke server shared**: A user asked for a link to the bloke server, and another member responded with [the Discord invite link](https://discord.gg/VpFvs9cU).
- **Safety models in AI responses**: A discussion highlighted that safety models in **Gemini** and possibly **OpenAI** check responses and can redact or reject them. One user noted, "Even though you could jailbreak it, you will not see the message if it cannot escape the safety filtering."
- **Karpathy's new course**: A user pointed out a new course by Karpathy, [LLM101n: Let's build a Storyteller](https://github.com/karpathy/LLM101n), mistaking it initially for the micrograd repo.
- **Hermes 2 Pro 70b format issues**: Users reported issues with Hermes-2-Theta-Llama-3-70B model responses starting with "<|end_header_id|>" and were advised to use the llama3 instruct format instead.
- **Release of Replete-Coder**: A new model, Replete-Coder-Qwen2-1.5b, was announced, scoring a 35 on HumanEval across 100 coding languages. More details were shared [in a tweet](https://x.com/dudeman6790/status/1805108449581072710).
  

**Links mentioned**:

- [PromptIde](https://ide.x.ai/): no description found
- [Dołącz do serwera TheBloke AI na Discordzie!](https://discord.gg/VpFvs9cU): For discussion of and support for AI Large Language Models, and AI in general. | 23728 członków
- [BigCodeBench Leaderboard - a Hugging Face Space by bigcode](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard): no description found
- [Xiaojie Xiaocat GIF - Xiaojie Xiaocat - Discover & Share GIFs](https://tenor.com/view/xiaojie-xiaocat-gif-18274610571282727560): Click to view the GIF
- [Skeleton Skeletons GIF - Skeleton Skeletons Skull - Discover & Share GIFs](https://tenor.com/view/skeleton-skeletons-skull-skulls-jellymid-gif-25125581): Click to view the GIF
- [Tweet from RomboDawg (@dudeman6790)](https://x.com/dudeman6790/status/1805108449581072710): Announcing Replete-Coder-Qwen2-1.5b An uncensored, 1.5b model with good coding performance across over 100 coding languages, open source data, weights, training code, and fully usable on mobile platfo...
- [Cool Beans GIF - Cool Beans Thumbsup - Discover & Share GIFs](https://tenor.com/view/cool-beans-thumbsup-gif-13344631): Click to view the GIF
- [GitHub - karpathy/LLM101n: LLM101n: Let's build a Storyteller](https://github.com/karpathy/LLM101n): LLM101n: Let's build a Storyteller. Contribute to karpathy/LLM101n development by creating an account on GitHub.
- [Tweet from Andrew Curran (@AndrewCurran_)](https://x.com/AndrewCurran_/status/1805259592806678699): This morning the RIAA, on behalf of Universal, Warner and Sony, filed a copyright infringement lawsuit against Suno and Udio.
- [Tweet from Keyon Vafa (@keyonV)](https://fxtwitter.com/keyonV/status/1803838591371555252): New paper: How can you tell if a transformer has the right world model? We trained a transformer to predict directions for NYC taxi rides. The model was good. It could find shortest paths between new...
- [Announcing PromptIDE](https://x.ai/blog/prompt-ide): no description found

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1254053586997088310)** (15 messages🔥):

- **Tiny Stories Model Impresses with Compact Size**: Discussion centered around the smallest LLMs, with a notable highlight being the [TinyStories-656K](https://huggingface.co/raincandy-u/TinyStories-656K) model, which has only 600k parameters. This lightweight model is capable of generating coherent stories utilizing a llama architecture.
- **Larger Models Show Superior Performance**: Members discussed the effectiveness of larger models, noting that good general-purpose performance starts at around 3B parameters with significant improvements seen in 7B-8B models. For top-tier performance, models with 70B+ parameters are considered the benchmark.
- **Autonomous Agents**: There was a debate on the potential of text predictors like Claude performing tasks comparable to a sentient human, with some asserting that autonomous, self-improving agents are within reach.
- **Fun with AI**: A humorous greentext story created by Claude emphasized its capability for creative text generation, illustrating advanced text prediction abilities and entertaining the users.

**Link mentioned**: [raincandy-u/TinyStories-656K · Hugging Face](https://huggingface.co/raincandy-u/TinyStories-656K): no description found

---

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1254368727064117278)** (12 messages🔥):

- **Track dataset generation in Google Sheets**: A member shared a [Google Sheet](https://docs.google.com/spreadsheets/d/1f5fbPxhjGrmPqhbM0exOCX2vAzffRWufhF7QBp24OMw/edit?gid=0#gid=0) for tracking dataset generation domains, encouraging participation by indicating interest, potential document sources, and target sizes. This aims to streamline the dataset creation process.
- **Huggingface chat template simplifies document input**: Members discussed enhancing the Huggingface chat template with document input fields, promoting the Hermes RAG format for standard metadata. This modification makes integrating documents into the model input **heaps easier** by using tools like jinja templates and XML for formatting.
- **AllenAI citation classification prompt**: An interesting citation classification prompt by AllenAI was shared, potentially useful for the `academic papers` category. This YAML-based prompt helps classify citations into categories like "Background," "Extends," "Uses," "Motivation," "CompareOrContrast," and "FutureWork."
- **SciRIFF dataset**: The group discussed the [SciRIFF dataset](https://huggingface.co/datasets/allenai/SciRIFF?row=0), which includes 137K instruction-following demonstrations for understanding scientific literature across five domains. The dataset comes with various configurations and a corresponding [GitHub repo](https://github.com/allenai/SciRIFF) for code, model training, and evaluation.
- **Instruction-pretrain dataset**: A member highlighted the [ft-instruction-synthesizer-collection](https://huggingface.co/datasets/instruction-pretrain/ft-instruction-synthesizer-collection/tree/main), noting it's fully RAG formatted and suggesting it might be **interesting** despite it being primarily multi-choice instead of free-form. The possibility of augmentation was considered to adapt the dataset for varied uses.
  

**Links mentioned**:

- [instruction-pretrain/ft-instruction-synthesizer-collection at main](https://huggingface.co/datasets/instruction-pretrain/ft-instruction-synthesizer-collection/tree/main): no description found
- [allenai/SciRIFF · Datasets at Hugging Face](https://huggingface.co/datasets/allenai/SciRIFF?row=0): no description found
- [RAG Data Synthesis](https://docs.google.com/spreadsheets/d/1f5fbPxhjGrmPqhbM0exOCX2vAzffRWufhF7QBp24OMw/edit?gid=0#gid=0): Sheet1 Domain,Curriculum file,Source/links,HF repo,Size (rows),Status,Who's working,Reviewer,Review Notes Websearch Wikipedia Codebase Academic Papers Books Finance ,SEC filings etc.,1000,Agent s...

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/)** (1 messages):

teknium: [https://twitter.com/hamish_kerr/status/1804352352511836403](https://twitter.com/hamish_kerr/status/1804352352511836403)

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1253814084449734792)** (114 messages🔥🔥):

- **SLURM Node Issues**: A user reported connecting to a SLURM-managed node through Jupyter Notebook, encountering errors at the training stage potentially due to SLURM restrictions. They mentioned testing on the console and receiving a 'kill' message before starting training, despite specifying GPU usage correctly.
- **PyTorch Accelerates Llama-2**: The PyTorch team released techniques for increasing Llama-2 inference speed by 10x, shared in a [blog post](https://pytorch.org/blog/accelerating-generative-ai-2/). A user developed a pip package [GPTFast](https://github.com/MDK8888/GPTFast) that applies these techniques to all HF models, asking for access to A100 or H100 GPU clusters.
- **Open-Source AI Model Issues**: Discussions arose around the ethics and practicality of sharing proprietary AI models like Mistral outside official channels. Users stressed the legal and moral implications of such actions, emphasizing the need for accountability and transparency in AI development.
- **Model Latency Profiling**: Users discussed methods for determining if an AI model is GPT-4 or another variant, with suggestions including checking knowledge cutoffs and profiling latency differences. Sniffing network traffic to identify the model used in API calls was also proposed.
- **LingOly Benchmark Discussion**: A new benchmark called LingOly, evaluating large language models (LLMs) on advanced reasoning with linguistic puzzles from low-resource languages, was discussed. The benchmark presents 1,133 problems and top models achieving below 50% accuracy, noted for its challenging nature and potential memorization concerns.
  

**Links mentioned**:

- [LINGOLY: A Benchmark of Olympiad-Level Linguistic Reasoning Puzzles in Low-Resource and Extinct Languages](https://arxiv.org/abs/2406.06196v2): In this paper, we present the LingOly benchmark, a novel benchmark for advanced reasoning abilities in large language models. Using challenging Linguistic Olympiad puzzles, we evaluate (i) capabilitie...
- [Virus Computer GIF - Virus Computer Hello Your Computer Has Virus - Discover & Share GIFs](https://tenor.com/view/virus-computer-hello-your-computer-has-virus-meme-memes-gif-20233783): Click to view the GIF
- [examples/examples/benchmarks/bert at main · mosaicml/examples](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert): Fast and flexible reference benchmarks. Contribute to mosaicml/examples development by creating an account on GitHub.
- [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/.): This post is the second part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance...
- [GitHub - AnswerDotAI/bert24](https://github.com/AnswerDotAI/bert24): Contribute to AnswerDotAI/bert24 development by creating an account on GitHub.

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1253799610372329492)** (155 messages🔥🔥):

- **TTS Paper Introduces ARDiT**: Discussion around [a new TTS paper](https://arxiv.org/abs/2406.05551) highlighting the potential of ARDiT in zero-shot text-to-speech. A member remarked, *"there's a bunch of ideas that could be used elsewhere."*
- **Exploring Multi-Objective Loss**: Intense debate on enforcing Pareto improvements in neural network training, focusing on multidimensional objectives. One member [shared insights on multi-objective optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization) and another concluded, *"probably you'd have to pick a small subset of the weights (say, the norm weights and biases) that vary between the different Pareto versions and share the rest."*
- **Quadratic Voting in Optimization**: Reference to [quadratic voting](https://en.wikipedia.org/wiki/Quadratic_voting) as a method to balance competing human values and integrate it into multi-objective optimization. The conversation weaved around the feasibility and implications of using quadratic voting in machine learning models.
- **Controversy in Multi-Task Learning**: A member recommends a paper revealing no significant benefits from specialized multi-task optimization methods over traditional approaches ([read here](https://arxiv.org/abs/2209.11379)). Another member [highlights a follow-up study](https://arxiv.org/abs/2312.06134) discussing optimization dynamics in data-imbalanced task collections.
- **Latent Space Regularization in AEs**: A thread discussed how to incorporate noise in autoencoder embeddings, suggesting adding Gaussian noise directly to the encoded output. Members debated on the necessity of regularization and batch normalization to prevent embeddings from scaling uncontrollably.
  

**Links mentioned**:

- [Towards an Improved Understanding and Utilization of Maximum Manifold Capacity Representations](https://arxiv.org/abs/2406.09366): Maximum Manifold Capacity Representations (MMCR) is a recent multi-view self-supervised learning (MVSSL) method that matches or surpasses other leading MVSSL methods. MMCR is intriguing because it doe...
- [HyperZ$\\cdot$Z$\\cdot$W Operator Connects Slow-Fast Networks for Full Context Interaction](https://arxiv.org/abs/2401.17948): The self-attention mechanism utilizes large implicit weight matrices, programmed through dot product-based activations with very few trainable parameters, to enable long sequence modeling. In this pap...
- [Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data](https://arxiv.org/abs/2406.14546): One way to address safety risks from large language models (LLMs) is to censor dangerous knowledge from their training data. While this removes the explicit information, implicit information can remai...
- [Quadratic voting - Wikipedia](https://en.wikipedia.org/wiki/Quadratic_voting): no description found
- [4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities](https://arxiv.org/abs/2406.09406): Current multimodal and multitask foundation models like 4M or UnifiedIO show promising results, but in practice their out-of-the-box abilities to accept diverse inputs and perform diverse tasks are li...
- [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782): While deep learning and deep reinforcement learning (RL) systems have demonstrated impressive results in domains such as image classification, game playing, and robotic control, data efficiency remain...
- [Multi-objective optimization - Wikipedia](https://en.wikipedia.org/wiki/Multi-objective_optimization): no description found
- [VisualRWKV: Exploring Recurrent Neural Networks for Visual Language Models](https://arxiv.org/abs/2406.13362): Visual Language Models (VLMs) have rapidly progressed with the recent success of large language models. However, there have been few attempts to incorporate efficient linear Recurrent Neural Networks ...
- [Transformers Can Do Arithmetic with the Right Embeddings](https://arxiv.org/abs/2405.17399): The poor performance of transformers on arithmetic tasks seems to stem in large part from their inability to keep track of the exact position of each digit inside of a large span of digits. We mend th...
- [Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models](https://arxiv.org/abs/2010.05874): Massively multilingual models subsuming tens or even hundreds of languages pose great challenges to multi-task optimization. While it is a common practice to apply a language-agnostic procedure optimi...
- [Toward Infinite-Long Prefix in Transformer](https://arxiv.org/abs/2406.14036v1): Prompting and contextual-based fine-tuning methods, which we call Prefix Learning, have been proposed to enhance the performance of language models on various downstream tasks that can match full para...
- [Order Matters in the Presence of Dataset Imbalance for Multilingual Learning](https://arxiv.org/abs/2312.06134): In this paper, we empirically study the optimization dynamics of multi-task learning, particularly focusing on those that govern a collection of tasks with significant data imbalance. We present a sim...
- [Tweet from François Fleuret (@francoisfleuret)](https://x.com/francoisfleuret/status/1804873919653957733): A little report!
- [Do Current Multi-Task Optimization Methods in Deep Learning Even Help?](https://arxiv.org/abs/2209.11379): Recent research has proposed a series of specialized optimization algorithms for deep multi-task models. It is often claimed that these multi-task optimization (MTO) methods yield solutions that are s...
- [Autoregressive Diffusion Transformer for Text-to-Speech Synthesis](https://arxiv.org/abs/2406.05551): Audio language models have recently emerged as a promising approach for various audio generation tasks, relying on audio tokenizers to encode waveforms into sequences of discrete symbols. Audio tokeni...
- [Grokking of Hierarchical Structure in Vanilla Transformers](https://arxiv.org/abs/2305.18741): For humans, language production and comprehension is sensitive to the hierarchical structure of sentences. In natural language processing, past work has questioned how effectively neural sequence mode...
- [Why Momentum Really Works](https://distill.pub/2017/momentum/): We often think of optimization with momentum as a ball rolling down a hill. This isn't wrong, but there is much more to the story.
- [no title found](https://aligniverse.streamlit.app/): no description found

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1254263108042489967)** (10 messages🔥):

- **Epoch revisits compute trade-offs in machine learning**: Members discussed [Epoch AI's blog post](https://epochai.org/blog/trading-off-compute-in-training-and-inference#monte-carlo-tree-search) about balancing compute during training and inference. One stated, *"It's possible to increase inference compute by 1-2 orders of magnitude, saving ~1 OOM in training compute."*
- **Paper on Neural Redshifts sparks interest**: Members shared [a paper on Neural Redshifts](https://openaccess.thecvf.com/content/CVPR2024/papers/Teney_Neural_Redshift_Random_Networks_are_not_Random_Functions_CVPR_2024_paper.pdf), noting that initializations may be more significant than researchers often acknowledge. One remarked, *"Initializations are a lot more interesting than researchers give them credit for being."*
- **AI Koans elicit laughs and enlightenment**: A humorous exchange about AI koans was shared, linking to a [collection of hacker jokes](http://www.catb.org/esr/jargon/html/koans.html). The illustration included an anecdote about a novice and an experienced hacker, showing how *“turning it off and on”* can fix problems unexpectedly.
  

**Links mentioned**:

- [Trading Off Compute in Training and Inference](https://epochai.org/blog/trading-off-compute-in-training-and-inference#monte-carlo-tree-search): We explore several techniques that induce a tradeoff between spending more resources on training or on inference and characterize the properties of this tradeoff. We outline some implications for AI g...
- [Some AI Koans](http://www.catb.org/esr/jargon/html/koans.html): no description found

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1253931388760232018)** (3 messages):

- **Model editing using SAEs explored in podcast**: A member referenced a [podcast episode](https://youtu.be/lj2y5hE04XI?t=4585) discussing the potential for using **SAEs** for model editing, specifically evaluating effectiveness using a non-cherrypicked list of edits from the **MEMIT paper**. They linked to the [MEMIT paper](https://arxiv.org/pdf/2210.07229.pdf) and its [source code](https://github.com/kmeng01/memit) for further exploration.
- **Interest in empirical evaluation for dictionary learning**: A member inquired if there are any recommended papers that empirically evaluate model behavior when influenced by features found via **dictionary learning**. This suggests a focus on empirical methods to understand model steering through structured feature manipulation.
  

**Links mentioned**:

- [Ep 14 - Interp, latent robustness, RLHF limitations w/ Stephen Casper (PhD AI researcher, MIT)](https://youtu.be/lj2y5hE04XI?t=4585): We speak with Stephen Casper, or "Cas" as his friends call him. Cas is a PhD student at MIT in the Computer Science (EECS) department, in the Algorithmic Ali...
- [Mass Editing Memory in a Transformer](https://memit.baulab.info/): Updating thousands of memories in GPT by directly calculating parameter changes.

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1254766616986783835)** (6 messages):

- **Local Model Registration Simplified**: A user inquired about the possibility of registering a model locally without altering `lm_eval/models/__init__.py`. Another user explained the usage of `register_model` and provided a code snippet showcasing how to achieve this with a wrapper module.
- **Breaking Change in Commit Highlighted**: A commit that added `tokenizer logs info` inadvertently broke the main branch. The user highlighted the issue with incorrect importing paths and requested a hotfix.
- **Hotfix Requested and Applied**: Another user directed attention to a proposed [hotfix](https://github.com/EleutherAI/lm-evaluation-harness/pull/2015), asking someone to test it. After confirmation, they acknowledged the fix resolved the issue.

**Link mentioned**: [add tokenizer logs info (#1731) · EleutherAI/lm-evaluation-harness@536691d](https://github.com/EleutherAI/lm-evaluation-harness/commit/536691da2444bd35b76d3f9c9527126273a63251): \* add tokenizer logs info

- add no tokenizer case
- Update lm_eval/logging_utils.py

Co-authored-by: Hailey Schoelkopf &lt;[65563625+haileyschoelkopf@users.noreply.github.com](mailto:65563625+haileyschoelkopf@users.noreply.github.com)&gt;

- U...

---

### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1254207280333983855)** (2 messages):

- **Debate over best multimodal LLM architecture**: A member questioned whether early fusion models like Chameleon are superior to using a vision encoder before feeding the image into the LLM context. They expressed concern that each approach might not be definitively better for all tasks but could be task-dependent.
- **Visual acuity trade-offs in early fusion**: They noted that early fusion might be better for generality; however, they heard the model struggles with visual acuity. This is due to the image tokenization process that compresses image information, losing clarity compared to patch embedding with a vision encoder.

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1254196252451803168)** (3 messages):

- **Intel pulling AWS instance, considers alternatives**: *“Intel is pulling our AWS instance so I'm thinking we either pay a little for these, or switch to manually-triggered free github runners.”* No definitive decision mentioned.
- **NCCL backend issues on A100 GPUs**: Attempts to train a model with **gpt-neox** on in-house **A100 GPUs** are facing NCCL backend issues. The issue persists across various versions of **NCCL** and **Cuda**, even with and without Docker.

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1253822664632438887)** (133 messages🔥🔥):

- **Noam Shazeer talks optimizing inference at Character.AI**: A new [blog post](https://research.character.ai/optimizing-inference/) from Noam Shazeer discusses how Character.AI is working towards AGI by optimizing inference processes. The post highlights their efforts to handle over 20,000 inference queries per second.
- **OpenAI acquires Rockset**: [OpenAI has acquired Rockset](https://x.com/deedydas/status/1804185430897889427) to bolster their Retrieval-Augmented Generation (RAG) capabilities. Founded in 2016, Rockset's team has deep expertise in building hybrid search solutions like vector (FAISS) and keyword search.
- **Karpathy announces a new course**: [Karpathy is planning an ambitious "LLM101n" course](https://x.com/miru_why/status/1804205538798182780?s=46&t=90xQ8sGy63D2OtiaoGJuww) on building ChatGPT-like models from scratch, similar to his famous CS231n course.
- **LangChain funding controversy addressed**: [LangChain's Harrison Chase](https://x.com/hwchase17/status/1804166140773691837) clarifies that their funding is focused solely on product development, not on sponsoring events or ads, in response to criticisms about their use of venture capital funds.
- **Mira Murati hints at GPTnext**: Mira Murati implied that the next major GPT model might [release in 1.5 years](https://www.youtube.com/watch?v=yUoj9B8OpR8), discussing the monumental shifts AI tools bring to creativity and efficiency in various fields.
  

**Links mentioned**:

- [llama.ttf](https://fuglede.github.io/llama.ttf/?utm_source=changelog-news): no description found
- [llama.ttf](https://fuglede.github.io/llama.ttf/?utm_source=changelog-new): no description found
- [Optimizing AI Inference at Character.AI](https://research.character.ai/optimizing-inference/): At Character.AI, we're building toward AGI. In that future state, large language models (LLMs) will enhance daily life, providing business productivity and entertainment and helping people with e...
- [Multi Blog – Multi is joining OpenAI](https://multi.app/blog/multi-is-joining-openai) : Recently, we’ve been increasingly asking ourselves how we should work with computers. Not on or using computers, but truly with computers. With AI. We think it’s one of the most importan...
- [Olympia | Better Than ChatGPT](https://olympia.chat): Grow your business with affordable AI-powered consultants that are experts in business strategy, content development, marketing, programming, legal strategy and more.
- [Tweet from Hamel Husain (@HamelHusain)](https://x.com/HamelHusain/status/1804301841666314501): In most cases i’ve encountered in the wild, the title “AI Engineer” is harmful. I explain why in the below video Quoting Hugo Bowne-Anderson (@hugobowne) The AI Engineer Data Literacy Divide 🎙...
- [AI Everywhere: Transforming Our World, Empowering Humanity](https://www.youtube.com/watch?v=yUoj9B8OpR8): Dartmouth Engineering hosted an exclusive conversation with alum and OpenAI Chief Technology Officer Mira Murati Th'12. She discussed the artificial intellig...
- [Tweet from Sully (@SullyOmarr)](https://x.com/sullyomarr/status/1803779798658859067?s=46): Introducing Otto - a new way to interact and work with AI Agents - using tables! Now you can have hundreds of agents working for you at the same time
- [Tweet from Harrison Chase (@hwchase17)](https://x.com/hwchase17/status/1804166140773691837): @levelsio all of our funding is going to our core team to help build out LangChain, LangSmith, and other related things we literally have a policy where we don't sponsor events with $$$, let alon...
- [Tweet from Alex Albert (@alexalbert__)](https://x.com/alexalbert__/status/1804899734475374857): Artifacts pro tip: If you are running into unsupported library errors with NPM modules, just ask Claude to use the cdnjs link instead and it should work just fine.
- [Tweet from nano (@nanulled)](https://x.com/nanulled/status/1804638164923167057): 100x checked data training and... It fking works and actually reasons over patterns. I can't fking believe that.
- [Tweet from Morten Just (@mortenjust)](https://x.com/mortenjust/status/1805190952358650251?s=46&t=90xQ8sGy63D2OtiaoGJuww): This is fast. Chrome running Gemini locally on my laptop. 2 lines of code.
- [Tweet from Andrew Curran (@AndrewCurran_)](https://x.com/andrewcurran_/status/1805259592806678699?s=46&t=90xQ8sGy63D2OtiaoGJuww): This morning the RIAA, on behalf of Universal, Warner and Sony, filed a copyright infringement lawsuit against Suno and Udio.
- [Tweet from Ammaar Reshi (@ammaar)](https://x.com/ammaar/status/1803914672091074726): Claude Sonnet 3.5 with Artifacts can also play sound! Using the @elevenlabs API it created a functional AI sound effects generator app, all I did was paste in the API documentation. I'm mind bl...
- [no title found](https://news.ycombinator.com/item?id=40739982): no description found
- [Tweet from Andrew Curran (@AndrewCurran_)](https://x.com/andrewcurran_/status/1805259592806678699?s=46&t=90xQ8sGy63D2Oti): This morning the RIAA, on behalf of Universal, Warner and Sony, filed a copyright infringement lawsuit against Suno and Udio.
- [Tweet from Morten Just (@mortenjust)](https://x.com/mortenjust/status/1805190952358650251?s=46&t=90xQ8s): This is fast. Chrome running Gemini locally on my laptop. 2 lines of code.
- [Tweet from TestingCatalog News 🗞 (@testingcatalog)](https://x.com/testingcatalog/status/1805288828938195319?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): It is coming 🔥
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://x.com/reach_vb/status/1805226216997200145?s=46&t=90xQ8sGy63D2OtiaoGJuww): Wait WHAT? Someone already extracted Gemini Nano weights from Chrome and shared them on the Hub ⚡ > Looks like 4-bit running on tf-lite (?) > base + instruction tuned adapter Obligatory disclo...
- [Tweet from Deedy (@deedydas)](https://x.com/deedydas/status/1804185430897889427): OpenAI just acquired Rockset to power RAG. Rockset was founded in 2016 by an ex-Facebook team that built RocksDB, a fork of Google's LevelDB, an embeddable NoSQL DB written by Jeff Dean himself. ...
- [Tweet from Bilawal Sidhu (@bilawalsidhu)](https://x.com/bilawalsidhu/status/1804255144835457503?s=46&t=90xQ8sGy63D2OtiaoGJuww): Wow. Stability AI's new CEO is Prem Akkaraju, Ex-CEO of the legendary VFX studio Weta Digital. SVD could've been competitive with Runway/Luma, but they dropped the ball. In fact, Luma AI go...
- [Tweet from Bilawal Sidhu (@bilawalsidhu)](https://x.com/bilawalsidhu/status/1804255144835457503?s=46&t=90xQ8s): Wow. Stability AI's new CEO is Prem Akkaraju, Ex-CEO of the legendary VFX studio Weta Digital. SVD could've been competitive with Runway/Luma, but they dropped the ball. In fact, Luma AI go...
- [Tweet from miru (@miru_why)](https://x.com/miru_why/status/1804205538798182780?s=46&t=90xQ8sGy63D2OtiaoGJuww): looks like @karpathy is now planning out a full cs231n-like course ‘LLM101n’ covering how to build a ChatGPT-like model from scratch https://github.com/karpathy/LLM101n. very ambitious!
- [Tweet from Robert Graham 𝕏 (@ErrataRob)](https://x.com/erratarob/status/1804018865145315529?s=46&t=90xQ8sGy63D2OtiaoGJuww): nVidia is in the same position as Sun Microsystems was in the early days of the dot-com bubble. Sun had the leading edge web servers, the smartest engineers, the most respect in the industry. If you ...
- [Tweet from jason liu (@jxnlco)](https://x.com/jxnlco/status/1804601597353226738): This seems made up. If you’ve built mle systems. I’m not convinced chaining and agents isn’t just a pipeline. Mle has never build a fault tolerance system?
- [Tweet from Mira Murati (@miramurati)](https://x.com/miramurati/status/1804567253578662264?s=46&t=90xQ8sGy63D2OtiaoGJuww): At OpenAI, we’re working to advance scientific understanding to help improve human well-being. The AI tools we are building, like Sora, GPT-4o, DALL·E and ChatGPT, are impressive from a technical stan...
- [Tweet from Andrej Karpathy (@karpathy)](https://x.com/karpathy/status/1805328398920958214?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): The @aiDotEngineer World's Fair in SF this week 🔥 https://www.ai.engineer/worldsfair Reminded of slide #1 from my most recent talk: "Just in case you were wondering… No, this is not a norma...
- [GitHub - admineral/Reactor: Early Alpha: Chat with React Code-Editor and Live-preview using Sandpack by Codesandbox. Vercel ai SDK RSC GenUI](https://github.com/admineral/Reactor): Early Alpha: Chat with React Code-Editor and Live-preview using Sandpack by Codesandbox. Vercel ai SDK RSC GenUI - admineral/Reactor
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1dmt6oy/two_quotes_from_anthropics_product_lead_on_claude/): no description found
- [GitHub - beowolx/rensa: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets](https://github.com/beowolx/rensa): High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets - beowolx/rensa

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1253830564629184625)** (3 messages):

- **New podcast on hiring AI engineers drops!**: A new episode of the Latent Space Podcast titled "How to Hire AI Engineers" has been released, featuring guest posts and a bonus pod from @james_elicit and @*adamwiggins*. The episode covers a range of topics including "Defining the Hiring Process," "Defensive AI Engineering," and "Tech Choices for Defensive AI Engineering" [full details here](https://x.com/latentspacepod/status/1804269727482810386).
- **Podcast also featured on Hacker News**: In addition to the direct link, it was mentioned that the podcast is also being discussed on [Hacker News](https://news.ycombinator.com/). No further details were provided.

**Link mentioned**: [Tweet from Latent Space Podcast (@latentspacepod)](https://x.com/latentspacepod/status/1804269727482810386): 🆕How to Hire AI Engineers a rare guest post (and bonus pod) from @james_elicit and @*adamwiggins*! Covering: - Defining the Hiring Process - Defensive AI Engineering as a chaotic medium - Tech Choi...

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1253801612745506840)** (72 messages🔥🔥):

- **Recording Permissions Pending World's Fair**: One member asked another if they could record the session and promised to hold off on uploads until after the World's Fair. Permission was granted with a thumbs up emoticon.
- **Developing a Twitter Management Application**: One member discussed creating a YAML-based DSL for a Twitter management app using the Twitter API, aiming to generate better analytics on social posts. They sought feedback on the importance of adding more features and shared detailed YAML code segments.
- **Zoho Social for Inspiration**: A member suggested referencing features from Zoho Social to build the Twitter analytics app. They provided a [Zoho Social link](https://www.zoho.com/social/features.html) detailing various features like scheduling, monitoring, and analyzing social media posts.
- **Anthropic's XML Tags Suggestion**: It was mentioned that Anthropics recommends using XML tags for certain functionalities, linking to a related [document](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags).
- **LLM-generated YAML Project Success**: A discussion followed about the usefulness of LLMs in generating YAML-based projects, with one member sharing their experience of using an LLM to create a YAML templating language implementation in Go, pointing to their [GitHub repository](https://github.com/go-go-golems/go-emrichen).

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://gist.github.com/wesen/9b2baa6cf5ed4a137adccd3e7c70c41c">ai-workshop.md</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://www.zoho.com/social/features.html">Zoho Social - Features</a>: Zoho Social's features tell you what makes it the best social media marketing software your money can buy today.</li><li><a href="https://github.com/go-go-golems/go-emrichen">GitHub - go-go-golems/go-emrichen: YAML templating language emrichen implementation in go</a>: YAML templating language emrichen implementation in go - go-go-golems/go-emrichen</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1253826170680643594)** (62 messages🔥🔥):

- **Estimating the Cost of LLVM**: Curiosity.fan shared [an article estimating the cost of LLVM](https://chriscummins.cc/2019/llvm-cost/) which concluded that 1.2k developers produced a 6.9M line codebase with an estimated cost of $530 million. The discussion included cloning and checking out the LLVM project to understand its development costs.
- **Issues with Mojo Installation**: Darinsimmons shared his frustrations with a fresh install of 22.04 and nightly builds of Mojo, stating none of the devrel-extras tests, including blog 2406, passed. He plans to take a break from the computer to resolve the issue.
- **Interactive Discussion on LLVM and Mojo**: Interest in LLVM and Mojo was enhanced by videos like the [EuroLLVM 2024 talks](https://youtu.be/y85-1g39X3E?si=N7ZEMxJgWBBwD22x), with users expressing their enthusiasm and plans to delve deeper into MLIR and LLDB extensions.
- **Documentation Navigation Confusion**: Users discussed the confusion stemming from the lack of clear differentiation between nightly and stable documentation in Mojo. Suggestions were made to maintain separate documentation sets for stable and nightly versions to aid clarity.
- **Curiosity about Mojo Stencil Operations**: Benny.n showed interest in exploring the `stencil` function in Mojo's algorithm library, speculating its use in reducing dimensions. He also expressed plans to reimplement autotune functionality, making hyperparameter evaluations more efficient at compile time.
  

**Links mentioned**:

- [2024 EuroLLVM - How Slow is MLIR](https://www.youtube.com/watch?v=7qvVMUSxqz4): 2024 European LLVM Developers' Meetinghttps://llvm.org/devmtg/2024-04/------How Slow is MLIRSpeaker: Mehdi Amini, Jeff Niu------Slides: https://llvm.org/devm...
- [stencil | Modular Docs](https://docs.modular.com/mojo/stdlib/algorithm/functional/stencil): stencilrank Int, stencilaxis Int, type fn(StaticIntTuple[$1]) capturing -> Tuple[StaticIntTuple[$1], StaticIntTuple[$1]], mapstrides Int) capturing -> Int, loadfn fn[Int capturing -> SIMD$4, ...
- [2024 EuroLLVM - Mojo debugging: extending MLIR and LLDB](https://youtu.be/y85-1g39X3E?si=N7ZEMxJgWBBwD22x): 2024 European LLVM Developers' Meetinghttps://llvm.org/devmtg/2024-04/------Mojo debugging: extending MLIR and LLDBSpeaker: Walter Erquinigo, Billy Zhu------...
- [2024 EuroLLVM - Efficient Data-Flow Analysis on Region-Based Control Flow in MLIR](https://www.youtube.com/watch?v=vvVR3FyU9TE): 2024 European LLVM Developers' Meetinghttps://llvm.org/devmtg/2024-04/------Efficient Data-Flow Analysis on Region-Based Control Flow in MLIRSpeaker: Weiwei ...
- [Estimating the Dollar Cost of LLVM](https://chriscummins.cc/2019/llvm-cost/): Full time geek and re­search stu­dent with a pas­sion for de­vel­op­ing great soft­ware, of­ten late at night.
- [mojo/examples/reduce.mojo at nightly · modularml/mojo](https://github.com/modularml/mojo/blob/nightly/examples/reduce.mojo): The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
- [mojo/docs/changelog.md at 1b79ef249f52163b0bafbd10c1925bfc81ea1cb3 · modularml/mojo](https://github.com/modularml/mojo/blob/1b79ef249f52163b0bafbd10c1925bfc81ea1cb3/docs/changelog.md#v070-2024-01-25): The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.

---

### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1254604078236045398)** (1 messages):

- **Modular posts new video**: **Modular** just announced a new [YouTube video](https://www.youtube.com/watch?v=AEnvEpQm9zg) titled " - YouTube." The description of the video is currently undefined.

**Link mentioned**: [\- YouTube](https://www.youtube.com/watch?v=AEnvEpQm9zg): no description found

---

### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1253804704555733055)** (5 messages):

- **Building a new data labeling platform**: A member asked for feedback on building a different kind of data labeling platform, inquiring about the most common types of data labeled, methods used, pain points, human intervention, and potential cost of an automated solution.
- **Product image labeling pain points**: A member discussed labeling product images and metadata, emphasizing pain points like ambiguity and the extent of manual effort required. They expressed willingness to use an automated product if it's cost-effective and reliable.
- **Manual labeling for PDFs**: Another member shared their experience with manual data labeling for PDFs and mentioned trying to fine-tune models for automation. They highlighted [Haystack](https://haystack.deepset.ai/) as a tool they've explored and underlined the importance of accuracy in pdf data extraction and labeling, especially for ERP integration.
- **Interest in ERP integration**: The original poster appreciated the feedback and noted the possibility of integrating their labeling platform with ERP systems, prompted by the insights shared about quickbooks and manual data entry.

**Link mentioned**: [Haystack | Haystack](https://haystack.deepset.ai/): Haystack, the composable open-source AI framework

---

### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1253795498612097115)** (51 messages🔥):

- **CONTRIBUTING.md lacks testing instructions**: A user noticed that the `CONTRIBUTING.md` file in the Mojo repo doesn't specify how to run all tests before submitting a PR. They recommended adding these instructions and linked the relevant document [here](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md).
- **Error with Mojo's control-flow.ipynb**: A user reported a SIGSEGV error when running a code snippet in `control-flow.ipynb`. Another user couldn't reproduce the issue and suggested updating to the latest nightly version and changing the type as a possible fix.
- **Issue with Mojo's staticmethod.ipynb**: An error was reported involving the destruction of a field out of a value in `staticmethod.ipynb`. Despite updating, the issue persisted, leading the user to consider filing a GitHub issue for further assistance.
- **OpenAI API key offer for help**: A user experiencing a critical issue offered an OpenAI API key worth $10 as an incentive for someone to help solve their problem, highlighting the community spirit and urgency of the issue. They emphasized the blocking nature of the problem and provided the GitHub issue [link](https://github.com/modularml/mojo/issues/3102).
- **Development and Docker support for Mojo**: Discussions included setups for running Mojo in dev containers, with links to example projects like [benz0li/mojo-dev-container](https://github.com/benz0li/mojo-dev-container) and an official modular Docker container example [here](https://github.com/modularml/mojo/tree/main/examples/docker). Users shared their preferences and experiences with these environments.
  

**Links mentioned**:

- [YouTube](https://www.youtube.com/watch?v=): no description found
- [2024 EuroLLVM - Mojo debugging: extending MLIR and LLDB](https://www.youtube.com/watch?v=y85-1g39X3E): 2024 European LLVM Developers' Meetinghttps://llvm.org/devmtg/2024-04/------Mojo debugging: extending MLIR and LLDBSpeaker: Walter Erquinigo, Billy Zhu------...
- [thatstoasty - Overview](https://github.com/thatstoasty): thatstoasty has 19 repositories available. Follow their code on GitHub.
- [mojo/stdlib/docs/development.md at main · modularml/mojo](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md): The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
- [mojo/examples/docker at main · modularml/mojo](https://github.com/modularml/mojo/tree/main/examples/docker): The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
- [[BUG] LSP & Mojo crashes when using Python.evaluate in a certain way · Issue #3102 · modularml/mojo](https://github.com/modularml/mojo/issues/3102): Bug description LSP and mojo crashes when using Python.evaluate in a certain way. i expected it to show me what the issue with the code was instead of crashing. Steps to reproduce Include relevant ...
- [GitHub - benz0li/mojo-dev-container: Multi-arch (linux/amd64, linux/arm64/v8) Mojo dev container](https://github.com/benz0li/mojo-dev-container): Multi-arch (linux/amd64, linux/arm64/v8) Mojo dev container - benz0li/mojo-dev-container
- [Modular Inc](https://github.com/modular): Modular is an integrated, composable suite of tools that simplifies your AI infrastructure so your team can develop, deploy, and innovate faster. - Modular Inc

---

### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1254449303972483212)** (58 messages🔥🔥):

- **Help with `prefetch` and `PrefetchOptions`**: One member asked for guidance on `prefetch` and `PrefetchOptions`, noting an unexpected speedup when using `PrefetchOptions().for_write().low_locality().to_instruction_cache()` for reading data immediately after. Another member confirmed prefetching is usually beneficial only for large `N`, as smaller `N` can be counterproductive.
- **Cache Performance and Prefetching**: Members discussed the importance of understanding cache activities via a profiler, as misuse of manual prefetching can degrade performance. They emphasized reading relevant manuals like the [Intel HPC tuning manual](https://link.to/manual) for further insights on prefetching mechanics.
- **Instruction vs Data Cache**: Clarification was given that fetching to the instruction cache (`icache`) also affects the `L2` cache shared between instructions and data. This can result in unexpected speedups due to structural cache management differences.
- **Function Inlining in Vectorized/Parallelized Calls**: It was discussed that inlining functions often leads to performance improvements in vectorized/parallelized operations since outlined functions are rarely vectorized automatically.
- **Tools for Optimization**: For cache size optimizations and other performance reasons, tools like `vtune` for Intel or `AMD uProf` for AMD are recommended. Mojo currently lacks compile-time cache size retrieval, which is necessary to avoid issues like false sharing.
  

**Links mentioned**:

- [Prefetching - Algorithmica](https://en.algorithmica.org/hpc/cpu-cache/prefetching/): no description found
- [PREFETCHW — Prefetch Data Into Caches in Anticipation of a Write](https://www.felixcloutier.com/x86/prefetchw): no description found
- [PREFETCHh — Prefetch Data Into Caches](https://www.felixcloutier.com/x86/prefetchh): no description found

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1253799832485892167)** (21 messages🔥):

- **Nightly MAX repo lags behind Mojo**: A member noticed the nightly/max repo hadn't been updated for almost a week. Another member explained that there's been an issue with the CI that publishes nightly builds of MAX, and a fix is in progress.
- **New Mojo Nightly Builds Released**: Announcements were made for new nightly Mojo compiler releases. Users can update to `2024.6.2205` and `2024.6.2305` with details provided in the [raw diffs](https://github.com/modularml/mojo/compare/bc3546a57e101fe0eb990bc15e96dad2b39e1aaf...40dc6b31bcaf1deb7032b2dff10ac80c068f9a3d) and [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **Controlled implicit conversion proposal**: A discussion revealed that the proposal to make implicit conversion opt-in is coming from Modular. The plan is to use a decorator to enable it only where it makes sense.
- **Troubleshooting segmentation faults in input() function**: A user sought help for a segmentation fault issue when resizing buffers in their `input()` function. Another user suggested it might be related to [an existing bug](https://github.com/modularml/mojo/issues/3065) about unsigned integer casting.
- **External emojis are functional**: A member celebrated that external emojis now work in the Discord. They expressed excitement at the new capability.
  

**Links mentioned**:

- [gojo/input.mojo at input · thatstoasty/gojo](https://github.com/thatstoasty/gojo/blob/input/input.mojo#L58): Experiments in porting over Golang stdlib into Mojo. - thatstoasty/gojo
- [mojo/stdlib/docs/development.md at main · modularml/mojo](https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md): The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
- [[BUG] Unsigned integer casting overflowing as if signed when using `int()` or `UInt32()` · Issue #3065 · modularml/mojo](https://github.com/modularml/mojo/issues/3065): Bug description Migrating this here after a bit of discussion in Discord. It seems like casting to unsigned integers actually just casts to signed integers, but has different behaviour in different...

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1253788315237351476)** (102 messages🔥🔥):

- **Weta Digital leadership changes spark reactions**: Discussions emerged about Weta Digital and their new CEO, with mentions of Sean Parker and speculation about the decision being more of a sale. *"Prem Akkaraju from Weta Digital huh"*, referenced along with frustrations over potential harassment faced by the company.
- **New CEO at Stability AI and industry intrigue**: A Reuters article about Stability AI appointing a new CEO was shared, with skepticism over the motives behind the leadership change. One member highlighted *"for those who don't want to pay these clowns for a $400 subscription"* and shared a [Reuters link](https://www.reuters.com/technology/artificial-intelligence/stability-ai-appoints-new-ceo-information-reports-2024-06-21/).
- **Llama 3 hardware recommendations draw interest**: Specifications for running Q6 llama 400 on a 12-channel AMD server were shared, along with cost approximations, eliciting excitement over potential performance. Expectations set for *"1 to 2 tokens per second with this setup"* prompted predictions on how it would compare to GPT-4O and Claude 3.
- **Debate on Meta model speculation**: Users debated the projected capabilities of Meta's 405B models and their potential training overhauls. Comments included hopes for updated weights from models like the 8B and 70B, along with observations such as, *"Meta didn't release a paper for Llama 3."*
- **Exploring advancements in EMA and model distillations**: Users discussed the implementation of EMA model updates in diffusers, shared by lucidrains on [GitHub](https://github.com/lucidrains/ema-pytorch), and their applicability to specific projects. The value of multiple captions in training datasets and the nuances of text embeddings were also analyzed, considering their impact on model training and performance.
  

**Links mentioned**:

- [Connecting Living Neurons to a Computer](https://youtu.be/c-pWliufu6U): Use code thoughtemporium at the link below to get an exclusive 60% off anannual Incogni plan: https://incogni.com/thoughtemporium____________________________...
- [Call to Build Open Multi-Modal Models for Personal Assistants | LAION](https://laion.ai/notes/open-gpt-4-o/): <p>Technologies like the recently introduced GPT-4-OMNI from OpenAI show again the potential which strong multi-modal models might have to positively transfo...
- [ema: offload to cpu, update every n steps by bghira · Pull Request #517 · bghira/SimpleTuner](https://github.com/bghira/SimpleTuner/pull/517/files>): no description found
- [Neuroplatform - FinalSpark](https://finalspark.com/neuroplatform/): no description found

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1253894530336559276)** (27 messages🔥):

- **Glaze team remarks on new attack paper**: The Glaze team responded to the new paper on adversarial perturbations, acknowledging the paper's findings and discussing their own tests with the authors' code. They highlighted the "noisy upscaling" method and its reliance on diffusion models, similar to [DiffPure](https://arxiv.org/abs/2205.07460), to remove artifacts from images.
- **Skepticism on Glaze/Nightshade's efficacy**: Members expressed skepticism and sadness over artists who believe Glaze or Nightshade will protect their art. They stressed the inevitable advantage of second movers in circumventing these protections and the resultant false hopes for artists.
- **New paper on multimodal models**: A new paper on [multimodal models](https://arxiv.org/abs/2406.09406) was discussed, noting its efforts to train on a wide range of modalities and tasks, improving model versatility. However, members felt like such papers repetitively declare breakthroughs without substantial new results.
- **Discussion on diffusion models for image restoration**: A detailed inquiry into image restoration tools was made, with [Robert Hoenig](https://arxiv.org/search/?query=Hoenig) discussing their experimental use of [super-resolution adversarial defense](https://github.com/aamir-mustafa/super-resolution-adversarial-defense) and training on specific image resolutions. The tests revealed that Glaze protections were consistently bypassed.
  

**Links mentioned**:

- [Tweet from François Fleuret (@francoisfleuret)](https://x.com/francoisfleuret/status/1804873919653957733): A little report!
- [4M-21: An Any-to-Any Vision Model for Tens of Tasks and Modalities](https://arxiv.org/abs/2406.09406): Current multimodal and multitask foundation models like 4M or UnifiedIO show promising results, but in practice their out-of-the-box abilities to accept diverse inputs and perform diverse tasks are li...
- [KalMamba: Towards Efficient Probabilistic State Space Models for RL under Uncertainty](https://arxiv.org/abs/2406.15131): Probabilistic State Space Models (SSMs) are essential for Reinforcement Learning (RL) from high-dimensional, partial information as they provide concise representations for control. Yet, they lack the...
- [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794): We introduce DataComp for Language Models (DCLM), a testbed for controlled dataset experiments with the goal of improving language models. As part of DCLM, we provide a standardized corpus of 240T tok...
- [Glaze - v2.1 Update](https://glaze.cs.uchicago.edu/update21.html): no description found

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1253902101382303775)** (117 messages🔥🔥):

- **New Members Navigate Discord and Cohere Channels**: Several new members joined the Discord, including one invited by Varun. Advice was given on navigating the platform, utilizing specific channels, and a [tool use documentation](https://docs.cohere.com/docs/tool-use) link was shared to assist in understanding how to connect Cohere models to external tools.
- **Discussion on BitNet and Model Quantization**: Members debated the feasibility and future use of BitNet, noting that BitNet is not optimized for current hardware and requires training from scratch. Mr. Dragonfox elaborated on why BitNet is currently impractical for commercial use, mentioning its lack of hardware support and inefficient training demands.
- **Interest in New AI Models and Rumors**: A member expressed interest in Cohere releasing new models, similar to recent updates from Meta, OpenAI, and Anthropic. There was also speculation on Anthropic's latest model, Claude-3.5-Sonnet, and discussions were held on scaling monosemanticity in models, linking to a paper on the topic.
- **Discussion on Cohere's Multilingual Capabilities**: A user inquired whether Cohere can respond in other languages such as Chinese. Nick_Frosst confirmed this ability and directed users to [documentation](https://docs.cohere.com/docs/tool-use) and a [notebook example](https://github.com/cohere-ai/notebooks/blob/main/notebooks/agents/Vanilla_Tool_Use.ipynb) for implementing tool use with Cohere models.
  

**Links mentioned**:

- [abideen/Bitnet-Llama-70M · Hugging Face](https://huggingface.co/abideen/Bitnet-Llama-70M): no description found
- [Bonjour Bonjour Mon Amor GIF - Bonjour Bonjour mon amor Bonjour mon cher - Discover & Share GIFs](https://tenor.com/view/bonjour-bonjour-mon-amor-bonjour-mon-cher-bon-matin-bonjours-gif-11477332989234919415): Click to view the GIF
- [Tool Use with Cohere's Models - Cohere Docs](https://docs.cohere.com/docs/tool-use): no description found
- [Login | Cohere](https://coral.cohere.com/?_gl=1*db2k2l*_gcl_au*MTE5MDQyODEyMC4xNzE5MDcxNDg5*_ga*NzUxMTg0MTI2LjE3MTkwNzE0ODk.*_ga_CRGS116RZS*MTcxOTA5OTEwMC40LjEuMTcxOTA5OTEwMy41Ny4wLjA.): Cohere provides access to advanced Large Language Models and NLP tools through one easy-to-use API. Get started for free.
- [Add support for BitnetForCausalLM (new model / new datatype) by Eddie-Wang1120 · Pull Request #7931 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/7931): Self Reported Review Complexity: Review Complexity : Low Review Complexity : Medium Review Complexity : High I have read the contributing guidelines PR Intro This PR is to support BitnetFor...

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1254548505964449876)** (10 messages🔥):

- **Microsoft AutoGen adds Cohere Client**: A contributor shared a [GitHub pull request](https://github.com/microsoft/autogen/pull/3004/files) for adding the Cohere client in AutoGen. Users expressed excitement, saying "siiick, thx for adding the client support!"
- **Call for Cohere team involvement**: A member clarified that the contribution was not theirs and called out to community contributors. Another member requested the Cohere team's assistance for further implementation, "we would like the cohere team to help us with the CohereClient implementation."

**Link mentioned**: [Cohere Client by Hk669 · Pull Request #3004 · microsoft/autogen](https://github.com/microsoft/autogen/pull/3004/files): Why are these changes needed? To enhance the support of non-OpenAI models with AutoGen. The Command family of models includes Command, Command R, and Command R+. Together, they are the text-generat...

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1254845382656004157)** (1 messages):

- **Cohere Developer Office Hours Announcement**: *"Join us tomorrow for our upcoming Cohere Developer Office Hours!"* A Senior Product Manager at Cohere will co-host the session to discuss the **Command R family tool use capabilities**, with a specific focus on **multi-step tool use** in the Cohere API.
- **Detailed Multi-step Tool Use Overview**: Cohere shared an overview of multi-step tool use, which *"allows Cohere's models to invoke external tools: search engines, APIs, functions, databases, and so on."* For more information, refer to the Cohere documentation and blog posts ([multi-step tool use](https://docs.cohere.com/docs/multi-step-tool-use), [Command R+](https://cohere.com/blog/multi-step-tool-use)).
  

**Links mentioned**:

- [Join the Cohere Community Discord Server!](https://discord.gg/s3pcZTyPgD?event=1248301309233336350): Cohere community server. Come chat about Cohere API, LLMs, Generative AI, and everything in between. | 17232 members
- [Automating Complex Business Workflows with Cohere: Multi-Step Tool Use in Action](https://cohere.com/blog/multi-step-tool-use): Enterprises are increasingly adopting AI to enhance business workflows. AI models, equipped with external tools, have the potential to streamline business operations. At Cohere, we’re excited to shar...
- [Multi-step Tool Use (Agents)](https://docs.cohere.com/docs/multi-step-tool-use): no description found

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1253801659675578378)** (100 messages🔥🔥):

- **Max Tokens and Pydantic Validations Confuse Users**: Users discussed confusion around max tokens for agents and context windows, and issues with LLM not following **Pydantic** validation. *"The context window or max token always includes the complete input token plus generated token."*
- **LangChain Tutorials and Resources**: Several users expressed difficulty learning **LangChain**, particularly in building chatbots and handling conversational digressions. [Grecil](https://corrective-rag.streamlit.app) shared a personal journey into LangChain and provided links to tutorials and documentation.
- **Using Multiple Chat Models and APIs**: Users debated performance issues and the application in different scenarios of **ChatOpenAI** vs. open-source models from **Huggingface**. One user asked about handling RAG on Excel files, implying versatility concerns with **LangChain** support for various data formats.
- **Handling Message History and Metadata in Chains**: Users sought help with implementing and troubleshooting **RunnableWithMessageHistory** and incorporating metadata in document retrievers. *"How to add the metadata that contains the documents/chunks retrieved in this chain."*
- **Streamlit App Hosting Discussions**: Issues of resource management and concurrency in **Streamlit** apps were discussed, including embedding API keys and handling multiple users simultaneously. "Yeah, Streamlit takes care of that. As soon as you close the tab, your instance and the files you uploaded are erased."
  

**Links mentioned**:

- [simplememory](https://pypi.org/project/simplememory/): The framework for agent memory
- [Reflexion - LangGraph](https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/#revision): no description found
- [NVIDIA NIMs | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/chat/nvidia_ai_endpoints/#example-usage-within-a-conversation-chains>).): The langchain-nvidia-ai-endpoints package contains LangChain integrations building applications with models on
- [Build a Chatbot | 🦜️🔗 Langchain](https://js.langchain.com/v0.2/docs/tutorials/chatbot/#managing-conversation-history>):): Overview
- [Build a Chatbot | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/tutorials/chatbot/#managing-conversation-history>)): This guide assumes familiarity with the following concepts:
- [Build a Chatbot | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/tutorials/chatbot/#message-history>)): This guide assumes familiarity with the following concepts:
- [TiDB Vector | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/vectorstores/tidb_vector/#using-as-a-retriever>)): TiDB Cloud, is a comprehensive Database-as-a-Service (DBaaS) solution, that provides dedicated and serverless options. TiDB Serverless is now integrating a built-in vector search into the MySQL landsc...
- [Introduction - LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-6-customizing-state.): no description found
- [no title found](https://corrective-rag.streamlit.app): no description found

---

### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1254216522109423647)** (21 messages🔥):

- **Generate QA pairs from PDF using LangChain**: A user requested the code to generate questions and answers from a PDF using LangChain. The Python code involves loading the PDF with `PyPDFLoader`, splitting it into chunks, creating embeddings with `OpenAIEmbeddings`, and setting up a `RetrievalQA` chain.
- **Linking issues from GitHub**: The code provided references several GitHub issues, such as [this one](https://github.com/langchain-ai/langchain/issues/17008) for guidance on generating question-answer pairs from PDFs.
- **Using Llama2 as LLM**: Another user requested modifications to the code to use Llama2 as the LLM. The updated instructions suggested initializing `LlamaCpp` and setting up `QAGenerationChain` with the `prompt_template`.
- **Iterating through text for QA pairs**: Lastly, instructions were given on how to iterate through text chunks from the PDF to generate question-answer pairs using the `QAGenerationChain`. This approach ensures multiple pairs are generated from the document.
  

**Links mentioned**:

- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/20406>).): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/20406>)): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/10395>).): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/17008>)): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/4950>).): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1254034783869206558)** (5 messages):

- **No Code RAG Workflows for Financial Documents**: A member shared an [article](https://medium.com/@manthapavankumar11/effortless-no-code-rag-workflows-for-financial-documents-implementing-embedding-cache-and-chat-e8d267b1c888) on designing a Retrieval-Augmented Generation (RAG) application using Flowise for financial document analysis. Key features include embedding cache using Redis and Qdrant for semantic search.
- **Linear Regression from Scratch**: Another member posted an [article](https://medium.com/@amitsubhashchejara/linear-regression-from-scratch-in-python-ee1a955e49ed) detailing how to implement linear regression from scratch in Python. The tutorial avoids using machine learning packages like scikit-learn, focusing instead on core concepts.
- **Corrective RAG App**: A member provided a link to their [Corrective RAG app](https://corrective-rag.streamlit.app) on Streamlit.
- **Edimate: AI-driven Educational Videos**: A member introduced Edimate, a tool that generates educational videos in about three minutes. They shared a [demo](https://x.com/dswharshit/status/1805203856088834428) showing its potential to transform e-learning by creating captivating, animated videos.
- **Regression Testing for LLMs**: An informative post linked to a [code tutorial](https://www.evidentlyai.com/blog/llm-regression-testing-tutorial) on regression testing for LLMs using open-source tools. The tutorial covers creating golden datasets, assessing response changes, and using the Evidently Python library to evaluate LLM outputs.
  

**Links mentioned**:

- [Tweet from Harshit Tyagi (@dswharshit)](https://x.com/dswharshit/status/1805203856088834428): How can you re-define E-learning with AI? This was the question I had as I have spent close to a decade in Edtech. The answer turned out to be generate videos/courses to explain any topic, on demand...
- [A tutorial on regression testing for LLMs](https://www.evidentlyai.com/blog/llm-regression-testing-tutorial): In this tutorial, you will learn how to systematically check the quality of LLM outputs. You will work with issues like changes in answer content, length, or tone, and see which methods can detect the...
- [Linear Regression From Scratch In Python](https://medium.com/@amitsubhashchejara/linear-regression-from-scratch-in-python-ee1a955e49ed): Learn the implementation of linear regression from scratch in pure Python. Cost function, gradient descent algorithm, training the model…
- [Effortless No Code RAG Workflows for Financial Documents: Implementing Embedding Cache and Chat…](https://medium.com/@manthapavankumar11/effortless-no-code-rag-workflows-for-financial-documents-implementing-embedding-cache-and-chat-e8d267b1c888): In the rapidly evolving landscape of financial data analysis, harnessing the power of advanced technologies without the need for extensive…
- [no title found](https://corrective-rag.streamlit.app): no description found

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1254886695481114694)** (1 messages):

- **Deciding on an AI Framework? Ask Critical Questions First**: A member shared a [YouTube video on AI framework considerations](https://youtu.be/uG0cs8AlnHw). The video discusses essential questions developers should ask before integrating AI tools like GPT-4o into their apps.

**Link mentioned**: [Do you even need an AI Framework or GPT-4o for your app?](https://youtu.be/uG0cs8AlnHw): So, you want to integrate AI into your product, right? Whoa there, not so fast!With models like GPT-4o, Gemini, Claude, Mistral, and others and frameworks li...

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1254860159113756702)** (1 messages):

- **AI21 introduces Jamba-Instruct:** Jamba-Instruct, an instruction-tuned variant by AI21, is tailored for enterprise use with an impressive **256K context window** to handle large documents. Check out more details [here](https://openrouter.ai/models/ai21/jamba-instruct).
- **NVIDIA releases Nemotron 4 340B Instruct:** Nemotron-4-340B-Instruct is a chat model focused on **synthetic data generation** for English-language applications. Find out more [here](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct).
  

**Links mentioned**:

- [AI21: Jamba Instruct by ai21](https://openrouter.ai/models/ai21/jamba-instruct): The Jamba-Instruct model, introduced by AI21 Labs, is an instruction-tuned variant of their hybrid SSM-Transformer Jamba model, specifically optimized for enterprise applications. - 256K Context Wind...
- [NVIDIA Nemotron-4 340B Instruct by nvidia](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct): Nemotron-4-340B-Instruct is an English-language chat model optimized for synthetic data generation. This large language model (LLM) is a fine-tuned version of Nemotron-4-340B-Base, designed for single...

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1253865624073932801)** (7 messages):

- **JojoAI transforms into a proactive assistant**: A member has transformed **JojoAI** into a proactive assistant capable of functions like *setting reminders*. They highlight that, unlike ChatGPT or Claude, JojoAI uses DigiCord integrations to remind users at specific times [JojoAI site](https://www.digicord.site).
- **Pebble: AI reading comprehension tool**: An AI-powered reading comprehension tool called **Pebble** was launched to help users remember information on the web. The developer used OpenRouter with **Mistral 8x7b** and **Gemini** and shared gratitude for the support of the OpenRouter team [Pebble](https://pebble.study/).
- **MoA project modified with OpenRouter**: A contributor modified the **MoA project** to use OpenRouter and added a server with an API endpoint, creating a GUI for usage. The project is available on [GitHub](https://github.com/timothelaborie/MoA-Openrouter/blob/main/gui.ipynb).
  

**Links mentioned**:

- [Pebble](https://pebble.study/): no description found
- [DigiCord](https://www.digicord.site): The most powerful AI-powered Discord Bot ever!
- [MoA-Openrouter/gui.ipynb at main · timothelaborie/MoA-Openrouter](https://github.com/timothelaborie/MoA-Openrouter/blob/main/gui.ipynb): together MoA but with Openrouter. Contribute to timothelaborie/MoA-Openrouter development by creating an account on GitHub.

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1253793170710462559)** (106 messages🔥🔥):

- **Nemotron 340b's environmental impact questioned**: *"Nemotron 340b is definitely one of the most environmentally unfriendly models u could ever use."* Discussion continued with comparisons suggesting Gemini Flash and other smaller, cheaper models as better alternatives for synthetic data generation.
- **Claude self-moderated endpoints issue fixed**: *"Looks like the Claude self-moderated endpoints are gone?"* After flagging a 404 error, a fix was implemented quickly, and the issue was resolved.
- **Sonnet 3.5 praised for coding**: A user shared positive experiences using Sonnet 3.5 for coding, calling it impressive and pointing to a [real-world demo](https://simonwillison.net/2024/Jun/21/search-based-rag/) with Retrieval Augmented Generation (RAG).
- **OpenRouter rate limits and credits explained**: *"How do you increase the rate limits for a particular LLM?"* Documentation on rate limits and credits was shared, explaining how to check the balance and usage via API requests.
- **Handling exposed API keys**: *"Hey, I like an idiot, showed a newly made api key on a stream and someone used it."* Recommendations were given to disable rather than delete compromised keys to trace any improper usage better.
  

**Links mentioned**:

- [Transforms | OpenRouter](https://openrouter.ai/docs/transforms): Transform data for model consumption
- [Limits | OpenRouter](https://openrouter.ai/docs/limits#rate-limits-and-credits-remaining): Set limits on model usage
- [Building search-based RAG using Claude, Datasette and Val Town](https://simonwillison.net/2024/Jun/21/search-based-rag/): Retrieval Augmented Generation (RAG) is a technique for adding extra “knowledge” to systems built on LLMs, allowing them to answer questions against custom information not included in their training d...

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1253828783019003924)** (85 messages🔥🔥):

- **Local LLMs on OS mode?**: A member asked whether local LLMs can be used in OS mode. Another member confirmed *"Yes! But performance of these models aren't very good..."* and provided the command *`interpreter --local --os`*.
- **Desktop App Premium Experience**: A member inquired about differences between the desktop app and the GitHub version. Mikebirdtech emphasized that *"The desktop app is going to be a very cool way to experience Open Interpreter"* and recommended joining the [waitlist for the desktop app](https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com).
- **Hitting GitHub Star Milestone**: Killianlucas excitedly announced the project has hit **50,000 stars on GitHub**, describing it as a huge accomplishment for the community. He mentioned a big server announcement coming soon.
- **Codestral and Deepseek Model Hype**: Several members discussed the recently released Deepseek and Codestral models, with Killianlucas noting that *"codestral... beat all our internal benchmarks..."* and favored Deepseek for its speed, mentioning an upcoming update with an optimized `interpreter --deepseek` command.
- **Ollama Connection Issues**: Arsaboo had issues connecting to Ollama hosted on a different computer using the OI interface. Multiple members suggested various fixes and troubleshooting steps, including changing API base URLs and using proxies, but none resolved the issue conclusively.
  

**Links mentioned**:

- [no title found](http://192.168.2.162:11434): no description found
- [Open Interpreter v0.3 Part 2](https://www.youtube.com/live/7lyw8V1PK3s?si=XT3DgJNTb7vQfpdM&t=9772): 0:00 - Setting up6:10 - Debugging `interpeter --os`8:01 - Using cursor to help debug19:38 - chat22:24 - Sonnet gives better answer than 4o29:00 - Fixing bash...
- [open-interpreter/interpreter/terminal_interface/profiles/defaults/codestral.py at main · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/profiles/defaults/codestral.py): A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
- [Update vision model to gpt-4o by MikeBirdTech · Pull Request #1318 · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/pull/1318): Describe the changes you have made: gpt-4-vision-preview was deprecated and should be updated to gpt-4o https://platform.openai.com/docs/deprecations/2024-06-06-gpt-4-32k-and-vision-preview-models ...
- [Using open interpreter with Ollama on a different machine · Issue #1157 · OpenInterpreter/open-interpreter](https://github.com/OpenInterpreter/open-interpreter/issues/1157#issuecomment-2184982086>): Describe the bug I am trying to use OI with Ollama running on a different computer. I am using the command: interpreter -y --context_window 1000 --api_base http://192.168.2.162:11434/api/generate -...
- [Open Interpreter - Desktop App](https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com): Apply for early access to the Open Interpreter Desktop App.
- [Google Colab](https://colab.research.google.com/drive/1jWKKwVCQneCTB5VNQNWO0Wxqg1vG_E1T#scrollTo=13ISLtY9_v7g)): no description found

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1253843036362903614)** (17 messages🔥):

- **Poetry vs requirements.txt sparks debate**: Members discussed the advantages and disadvantages of using **Poetry** over a traditional `requirements.txt` file. One member highlighted Poetry's deterministic builds and ease of management, while another pointed out that it can be difficult to manage across platforms, suggesting **conda** as an alternative.
- **01 Installation Documentation Shared**: A member shared a [setup link](https://01.openinterpreter.com/getting-started/setup) for installing 01 on different operating systems. Another member expressed frustration, stating that it "doesn't work yet" on some platforms.
- **Windows Installation Challenges**: Discussions highlighted difficulties in managing dependencies on Windows with tools like **Poetry** and **venv** compared to **conda**. Despite one user's assertion that Poetry and venv work fine on Windows, another noted frequent failures for non-01 packages.
- **Community Sentiments**: A member expressed strong positive sentiments, calling this discord community their favorite. Others discussed the beginner-friendliness of the 01 light, with developers noting current versions require technical knowledge but future releases aim to be more accessible.
- **Shipping Timeline Frustrations**: Members expressed concerns over the shipping timelines of the 01 device. One user mentioned repeated delays, while another defended the timelines against perceived misinformation.
  

**Links mentioned**:

- [Poetry - Python dependency management and packaging made easy](https://python-poetry.org/): no description found
- [Setup - 01](https://01.openinterpreter.com/getting-started/setup): no description found

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1254330956882640936)** (5 messages):

- **Funny Thumbnail from Techfren’s Community**: A member shared a [YouTube live video](https://youtube.com/live/8TN8tzkyB50?feature=share) and noted the amusing thumbnail made by Flashwebby from the techfrens community. Another member commented on loving the thumbnail, which prompted the original member to share their lighthearted contribution to the video.
- **Amoner Remixes "The Wheels on the Bus" with AI**: A member presented a [YouTube video](https://www.youtube.com/watch?v=a-Jlq0iX898&t=47s&ab_channel=Amoner) highlighting a remix of "The Wheels on the Bus" using Suno and Luma technologies. The video description emphasizes the innovative use of GenAI technology for creating next-gen music and visuals.

**Link mentioned**: [AI Remix: The Wheels on the Bus | Next-Gen Music & Visuals by Suno & LumaLabs](https://www.youtube.com/watch?v=a-Jlq0iX898&t=47s&ab_channel=Amoner): Experience 'The Wheels on the Bus' like never before with this innovative AI-generated remix! Using the latest in GenAI technology, we've collaborated with S...

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1253816131815346207)** (33 messages🔥):

- **Explore Instruction Pre-Training for multi-task learning**: A member shared a [Hugging Face repository](https://huggingface.co/instruction-pretrain/instruction-synthesizer) on **Instruction Pre-Training**, which augments raw corpora with instruction-response pairs for supervised multitask pre-training. This method has effectively synthesized 200M pairs across 40+ task categories.
- **DeBERTa with Flash Attention 2**: A user inquired if anyone knew of any **DeBERTa implementations using Flash Attention 2**, indicating interest in combining these two technologies.
- **Blank Page Issue on Maven Course Platform**: Multiple users experienced a blank page when trying to access a course on Maven, prompting discussion about troubleshooting and attempts to contact Maven support. A temporary workaround involved accessing the course on mobile devices.
- **Running AI Applications Workshop**: Attendees discussed an upcoming event in San Francisco, [AI Engineer World’s Fair](https://www.ai.engineer/worldsfair), which includes workshops on quickly deploying AI applications with templates. Several members expressed interest in meeting up at the event.
- **Why companies prefer fine-tuning over RAG**: There was a discussion on why job ads often seek fine-tuning expertise rather than Retrieval-Augmented Generation (RAG). It was suggested that companies aim to reduce LLM costs, making fine-tuning a valuable skill.
  

**Links mentioned**:

- [AI Mathematical Olympiad - Progress Prize 1 | Kaggle](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/512844): no description found
- [Welcome to Outlines! - Outlines 〰️](https://outlines-dev.github.io/outlines/welcome/): Structured text generation with LLMs
- [instruction-pretrain/instruction-synthesizer · Hugging Face](https://huggingface.co/instruction-pretrain/instruction-synthesizer): no description found
- [Instruction Pre-Training: Language Models are Supervised Multitask Learners](https://arxiv.org/abs/2406.14491): Unsupervised multitask pre-training has been the critical method behind the recent success of language models (LMs). However, supervised multitask learning still holds significant promise, as scaling ...
- [no title found](https://maven.com/parlance-labs/fine-tuning/1/home): no description found
- [GitHub - beowolx/rensa: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets](https://github.com/beowolx/rensa): High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets - beowolx/rensa
- [AI Engineer World's Fair](https://www.ai.engineer/worldsfair): Join 2,000 software engineers enhanced by and building with AI. June 25 - 27, 2024, San Francisco.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/)** (1 messages):

christopher_39608: Interesting post:

[https://x.com/rasbt/status/1805217026161401984](https://x.com/rasbt/status/1805217026161401984)

---

### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1253805483987570872)** (6 messages):

- **Missing Credits and Troubleshooting**: A user reported, *"I haven't received the credits yet,"* and was advised to contact billing if they had filled out the form correctly. They were informed to email billing with proof of sign-up date, HF username, and email.
- **Prompt Customer Service Response**: Another individual faced the same issue and mentioned their HF username and email directly in the channel. They received a quick response advising them to contact billing for further assistance and acknowledged sending the receipt to the provided email.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1253980281635868774)** (3 messages):

- **Broken template reported for Mixtral 8x22**: A user inquired about the broken template issue for **Mixtral 8x22** and tagged two members, seeking help to address it.
- **Replicate credits usage with VScode extension**: It was shared that **Replicate credits** can be utilized with a VScode extension named **continue.dev**. This extension functions similar to **Github Copilot**, using Replicate APIs, and also offers a **@docs feature** to interact with Replicate documentation locally.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1253820800776147055)** (1 messages):

- **Missing Credits Frustrate User**: A user reported not seeing their credits after logging in and adding a credit card for billing. They shared their organization ID, *be7114fc-9d79-475a-a258-ddbda1553c9a*, to seek assistance.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/)** (1 messages):

jxnlco: nah

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1254483892149293156)** (3 messages):

- **Subprocess.CalledProcessError plagues training**: A user reported an error, *subprocess.CalledProcessError: Command '['/usr/bin/python3', '-m', 'axolotl.cli.train', '/content/qlora.yml']' returned non-zero exit status 1*, indicating issues with running Axolotl's training command.
- **LORA overfitting concerns**: Another user queried whether significantly lower training loss compared to validation loss signals overfitting, even when using **LORA**. The question implies common concerns among users about overfitting in fine-tuning models.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1254604590226210826)** (1 messages):

- **Help requested for error in .yml and dataset**: A member asked for assistance with an error they encountered. They attached the .yml and dataset to provide context and mentioned using Modal for this FTJ, appreciating any support offered.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[simon_cli_llms](https://discord.com/channels/1238365980128706560/1242664474276659320/)** (1 messages):

mgrcic: Also available at [https://www.youtube.com/watch?v=QUXQNi6jQ30](https://www.youtube.com/watch?v=QUXQNi6jQ30)

---

### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1253805778351947838)** (3 messages):

- **Dan clarifies credit issues**: A user sought help figuring out credits as they hadn't received any yet. Dan asked if the user signed up and responded to the forms by the deadline, and offered to check what data was sent to the platforms if provided with the email address.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1254293680638922752)** (2 messages):

- **User tags and codes dominate the chat**: With user tags like `<@466291653154439169>` and codes such as `tyagi-dushyant1991-e4d1a8` and `williambarberjr-b3d836`, it appears members are sharing unique identifiers or codes. No further context on the usage or purpose of these tags was provided.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1253818157441749175)** (25 messages🔥):

- **Some users are missing their credits**: Several members, including **xyz444139**, **nima01258**, and **claudio_08887**, reported not receiving their credits despite following procedures. **ankrgyl** addressed these issues by checking email records, confirming permissions, and applying credits where appropriate.
- **Permission issues resolved after kernel restart**: **claudio_08887** encountered a *"User does not have permissions to create a project within this org"* error while running an evaluation example. The problem was resolved after restarting the kernel, indicating it might have been a transient issue.
- **braintrust lacks direct fine-tuning capabilities**: When asked about tutorials for fine-tuning Huggingface models with braintrust, **ankrgyl** clarified that braintrust can assist in evaluating fine-tuned models but does not have built-in fine-tuning capabilities.
- **Customer feedback is appreciated and encouraged**: **lapuerta91** expressed admiration for the product, to which **ankrgyl** responded with appreciation and invited further feedback on potential improvements.

---

### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1253822951149797488)** (13 messages🔥):

- **Predibase credits expire in 30 days**: A user queried if **Predibase credits expire at the end of the month**. Confirmation was provided that **credits expire 30 days after they are issued** with a reference link.
- **New user assistance with credits**: A new user noted only seeing $25 in available credits. **Predibase support** suggested directly messaging or emailing [support@predibase.com](mailto:support@predibase.com) for assistance.
- **Enterprise tier features**: There was a discussion about the **enterprise tier** of Predibase, stating it offers features for **production-scale applications**. Users interested in this tier were advised to contact support.

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1254141690714001508)** (5 messages):

- **LightningAI's RAG template simplifies AI development**: LightningAI provides tools for developing and sharing both traditional ML and genAI apps, as shown in [Jay Shah's template](https://t.co/2NLH7zuZS6) for setting up a multi-document agentic RAG. This template allows for an out-of-the-box setup to streamline the development process.
- **Customizable Text-to-SQL with DAGs**: Existing text-to-SQL modules often need custom orchestration and prompt adjustments for production use. An [underrated feature](https://t.co/fiS0kfj8rk) of llama_index is its ability to support these advanced LLM customizations.
- **Corrective RAG for better financial analysis**: The CRAG technique, as described by Yan et al., assesses retrieval quality and uses web search for backup context when the knowledge base is insufficient. Hanane Dupouy's [tutorial slides](https://t.co/lHsThk9IOU) offer detailed guidance on implementing this advanced RAG technique.
- **RAG parameter tuning with Mlflow**: Managing RAG's numerous parameters, from chunking to indexing, is crucial for answer accuracy, and it’s essential to have a systematic tracking and evaluation method. Integrating llama_index with [Mlflow](https://t.co/fo8XxMTO93) helps achieve this by defining proper eval metrics and datasets.
- **LlamaIndex integrates image generation via StabilityAI**: The new feature in create-llama now supports image generation using [StabilityAI](https://t.co/a7F0gv4tpi). This integration expands the capabilities of LlamaIndex for AI developers.

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1253885329572892813)** (70 messages🔥🔥):

- **LlamaIndex's Query Response Modes Explained**: Members discussed various query response modes in LlamaIndex, such as **Refine**, **Compact**, **Tree Summarize**, and **Accumulate**. Each mode uses different strategies to generate and refine responses incrementally or through tree summarization ([source](https://docs.llamaindex.ai/en/latest/api_reference/response_synthesizers/#llama_index.core.response_synthesizers.type.ResponseMode)).
- **Using OLLAMA_NUM_PARALLEL with LlamaIndex**: A member inquired about the use of **OLLAMA_NUM_PARALLEL** to run multiple models concurrently in LlamaIndex. It was noted that this seems to only require setting an environment variable and no changes in LlamaIndex are needed yet.
- **Document Parsing Issues**: Issues were raised about some documentation pages not rendering correctly on LlamaIndex's site. Links ending in .md were pointed out as the cause, leading to a plan to update those pages ([example link](https://docs.llamaindex.ai/en/stable/community/full_stack_projects/)).
- **Discussion on Custom Similarity Scores in Vector Databases**: A member asked about defining custom similarity scores using Weaviate or Elasticsearch in LlamaIndex. It was recommended to implement this at the level of the vector database, as LlamaIndex wraps around their libraries and doesn't directly support custom retrievers.
- **Embedding Dimensions Mismatch in PGVectorStore**: A member faced issues with embedding dimension mismatches when using **bge-small embedding** model with **PGVectorStore**, which required 384-dimension embeddings instead of the default 1536. Adjustments in the `embed_dim` parameter and ensuring the correct embedding model was advised.
  

**Links mentioned**:

- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997): Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented...
- [Anthropic | LlamaIndex.TS](https://ts.llamaindex.ai/modules/llms/available_llms/anthropic): Usage
- [Full Stack Projects - LlamaIndex](https://docs.llamaindex.ai/en/stable/community/full_stack_projects/): no description found
- [Full-Stack Web Application - LlamaIndex](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/apps/): no description found
- [Query Engine with Pydantic Outputs - LlamaIndex](https://docs.llamaindex.ai/en/latest/examples/query_engine/pydantic_query_engine/#query-engine-with-pydantic-outputs>)): no description found
- [Index - LlamaIndex](https://docs.llamaindex.ai/en/latest/api_reference/response_synthesizers/#llama_index.core.response_synthesizers.type.ResponseMode>)): no description found

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1254781955933999154)** (1 messages):

- **Guide to MLflow and LLMs with LlamaIndex**: A link to a [Medium article](https://medium.com/ai-advances/unlocking-efficiency-in-machine-learning-a-guide-to-mlflow-and-llms-with-llamaindex-integration-2b1e7ade1437) about integrating **MLflow** and **LLMs** using **LlamaIndex** was shared. The article aims to "unlock efficiency in machine learning", authored by Ankush K Singal.

**Link mentioned**: [Unlocking Efficiency in Machine Learning: A Guide to MLflow and LLMs with LlamaIndex Integration](https://medium.com/ai-advances/unlocking-efficiency-in-machine-learning-a-guide-to-mlflow-and-llms-with-llamaindex-integration-2b1e7ade1437): Ankush k Singal

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1253860614942752838)** (17 messages🔥):

- **Gemini 1.5 Pro has fewer parameters than LLAMA 3 70B**: A member with a "reputable source at Meta" claimed *"Gemini 1.5 Pro has fewer parameters than LLAMA 3 70B."* This led to discussions on the architecture differences, esp. MoE (Mixture of Experts), influencing the active parameter count during inference.
- **Early fusion technique in GPT-4**: There's a debate whether GPT-4T/o are distilled models or utilize an early fusion technique. One member suggested *"GPT4 o is just early fusion GPT4"* while another believed it involved larger models like *"GPT4-omni"* distilled down.
- **Difficulty in post-training multimodal models**: A discussion emerged on post-training multimodal models like Gemini Ultra and GPT4-o, highlighting challenges in modality transfer. One pointed out that *"post-training for native multimodal models are really hard, and the transfer across modalities seem small."*
- **Multi joins OpenAI, sunsets app**: Multi, once aiming to reimagine desktop computing as inherently multiplayer, is joining OpenAI according to a [blog post](https://multi.app/blog/multi-is-joining-openai). Multi will stop service by July 24, 2024, a member remarked *"OpenAI is on a shopping spree".*

**Link mentioned**: [Multi Blog – Multi is joining OpenAI](https://multi.app/blog/multi-is-joining-openai) : Recently, we’ve been increasingly asking ourselves how we should work with computers. Not on or using computers, but truly with computers. With AI. We think it’s one of the most importan...

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1253943008655577090)** (20 messages🔥):

- **The Value of Faulty Code**: Members debated the importance of including faulty code during training. One stated, *"code with errors so that it understands how to fix errors"* is necessary, while another emphasized that *"bad data needs to be situated in some context that makes it obvious that it's bad."*
- **Risk Aversion in AI Datasets**: There was a discussion on the high stakes of using open datasets. A member pointed out, *"the stakes are too high now... people filter down CommonCrawl the millionth time"* largely due to concerns over legality and backlash.
- **Ethical and License Issues**: The conversation covered the inconsistency of license terms. One member humorously remarked, *"you just can't upload and train on your own lolol"* pointing to practical evasions of restrictive licenses.
- **High-Risk Data Types**: Natolambert noted that video and image datasets carry a higher risk compared to other types of data. They also expressed a need for faster improvements in synthetic data options, implying current limitations.
- **Link To Relevant Article**: Discussion included a [2022 article on AI data laundering](https://waxy.org/2022/09/ai-data-laundering-how-academic-and-nonprofit-researchers-shield-tech-companies-from-accountability/) that highlighted the shielding of tech companies from accountability, shared by dn123456789. This sparked remarks on the sad state of dataset ethics in current AI practices.
  

**Links mentioned**:

- [AI Data Laundering: How Academic and Nonprofit Researchers Shield Tech Companies from Accountability - Waxy.org](https://waxy.org/2022/09/ai-data-laundering-how-academic-and-nonprofit-researchers-shield-tech-companies-from-accountability/): Tech companies working with AI are outsourcing data collection and training to academic/nonprofit research groups, shielding them from potential accountability and legal liability.
- [AI Data Laundering: How Academic and Nonprofit Researchers Shield Tech Companies from Accountability - Waxy.org](https://waxy.org/2022/09/ai-data-laundering-how-academic-and-nonprofit-researchers-shield-tech-compa): Tech companies working with AI are outsourcing data collection and training to academic/nonprofit research groups, shielding them from potential accountability and legal liability.

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1253805819179569195)** (13 messages🔥):

- **Sony Music vs Nous Research:** A Nous Research member tagged @sonymusic on X, questioning, *“who exactly is nouse research?"*. This sparked curiosity and seemed to mix up the conversation about AI innovation and potential legal entanglements.
- **Pre-emptive Cease and Desist Joke:** One member joked about unlocking the *"ultra-rare 'Pre-emptive cease and desist' achievement"* despite never having trained audio models, adding humor to the legal concerns.
- **Claude 3.5 Conspiracy Theory:** There was a humorous conspiracy theory shared that *"Claude 3.5 isn’t real but just Claude 3 with the 'I’m very smart' vector cranked up,"* demonstrating skepticism towards model improvements.
- **OpenAI's Vague Apology:** Mira Murati’s post on X addressed OpenAI’s mission, tools like Sora and GPT-4o, and the balance between creating innovative AI while managing its impact. Despite her detailed explanation, a member commented that the apology was *"clearly not pleasing anybody."*
- **Hugging Face Access Drama:** An announcement on a Hugging Face model page states they are suspending new download access requests due to conflicts, citing a perceived *“repeated misuse of the 'Contributor Covenant Code of Conduct'"* by Hugging Face, and prioritization of commercialization over community well-being.
  

**Links mentioned**:

- [CausalLM/14B-DPO-alpha · Hugging Face](https://huggingface.co/CausalLM/14B-DPO-alpha): no description found
- [Tweet from Nous Research (@NousResearch)](https://x.com/nousresearch/status/1804219649590276404?s=46): uhh hey @sonymusic who exactly is nouse research
- [Tweet from Tsarathustra (@tsarnick)](https://x.com/tsarnick/status/1803901130130497952): Mira Murati: GPT-3 was toddler-level, GPT-4 was a smart high schooler and the next gen, to be released in a year and a half, will be PhD-level
- [Tweet from emozilla (@theemozilla)](https://x.com/theemozilla/status/1804220182237495461?s=46): I unlocked the ultra-rare "Pre-emptive cease and desist" achievement (p.s. I've never trained any audio models) Quoting Nous Research (@NousResearch) uhh hey @sonymusic who exactly is ...
- [Tweet from Mira Murati (@miramurati)](https://x.com/miramurati/status/1804567253578662264): At OpenAI, we’re working to advance scientific understanding to help improve human well-being. The AI tools we are building, like Sora, GPT-4o, DALL·E and ChatGPT, are impressive from a technical stan...

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1253786942328275014)** (9 messages🔥):

<ul>
  <li><strong>Internet Traffic and Content Quality</strong>: A member suggested that if the content is really good, people will click and explore it. However, they noted that if the content is mediocre, it doesn’t deserve much traffic anyway.</li>
  <li><strong>Farmer and Sheep Problem Joke</strong>: A shared a humorous tweet that extends the "one farmer and one sheep problem," suggesting that "sheep can row the boat as well." The full tweet can be viewed <a href="https://x.com/_arohan_/status/1804661929694446065">here</a>.</li>
  <li><strong>Gemini 1.5 Bragging Rights</strong>: There was a mention of an updated Gemini model that reportedly didn't make it into the I/O presentation. The tweet about this can be found <a href="https://x.com/an1lam/status/1792397828733776026">here</a>.</li>
  <li><strong>Anthropic's AI Videos</strong>: Anthropic has been sharing videos on YouTube about topics like AI personality and interpretability. Noteworthy videos are <a href="https://www.youtube.com/watch?v=iyJj9RxSsBY">"What should an AI's personality be?"</a> and <a href="https://www.youtube.com/watch?v=sQar5NNGbw4">"Scaling interpretability"</a>.</li>
  <li><strong>Mixed Reception to AI Content</strong>: Some members felt that certain parts of AI-related content were boring or not as interesting as hoped. Despite these critiques, there is a desire for continued production of such content.</li>
</ul>

<div class="linksMentioned"><p><strong>Links mentioned</strong>:</p><ul><li><a href="https://x.com/_arohan_/status/1804661929694446065">Tweet from rohan anil (@_arohan_)</a>: Sorry I had to share this one. Sheep can row the boat as well you know!</li><li><a href="https://x.com/an1lam/status/1792397828733776026!">Tweet from Stephen Malina (@an1lam)</a>: Can't believe this didn't make it into the I/O presentation! Updated Gemini passing.</li><li><a href="https://www.youtube.com/watch?v=iyJj9RxSsBY">What should an AI's personality be?</a>: How do you imbue character in an AI assistant? What does that even mean? And why would you do it in the first place? In this conversation, Stuart Ritchie (Re...</li><li><a href="https://www.youtube.com/watch?v=sQar5NNGbw4">Scaling interpretability</a>: Science and engineering are inseparable. Our researchers reflect on the close relationship between scientific and engineering progress, and discuss the techn...</li></ul></div>

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1254198777745641603)** (3 messages):

- **Eat up piggies**: A user shared the message *"eat up piggies"*. It remains unclear in context without further explanation.
- **Model hubs on the way**: Another message stated simply *"model hubs soon 🤗"*. This hints at upcoming developments or releases related to model hubs.
- **Expressing confusion**: Nathan Lambert shared the sentiment *"This makes no sense in so lost"*. This suggests some confusion or misunderstanding regarding the previous messages.

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1253808200164315196)** (4 messages):

- **Mixture of Agents model raises eyebrows**: A member shared a tweet about the Mixture of Agents model being the strongest on the AlpacaEval leaderboard, claiming it beats GPT-4 by being 25 times cheaper. Another member deemed it *dumb*, questioning the legitimacy of the leaderboard which allegedly incorporates biased metrics.
- **Alpaca Eval skepticism**: Several members expressed skepticism about the Alpaca Eval leaderboard, indicating that it might include biased or inflated performance metrics. One member bluntly stated, "They add all sorts of slop to their leaderboard" and labeled themselves as an "alpaca eval hater".

**Link mentioned**: [Tweet from Kyle Corbitt (@corbtt)](https://x.com/corbtt/status/1804199596656410987): Thrilled to be officially recognized as the strongest model on the AlpacaEval leaderboard. 🙂 [https://tatsu-lab.github.io/alpaca_eval/](https://tatsu-lab.github.io/alpaca_eval/) Quoting Kyle Corbitt (@corbtt) Super excited to announce our ...

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1253827044463083582)** (33 messages🔥):

- **Use ROCm Fork Versions**: Members discussed needing to use the ROCm fork versions of [xformers](https://github.com/ROCm/xformers) and [flash-attention](https://github.com/ROCm/flash-attention) for certain functionalities. One user confirmed that flash-attention support requires ROCm 5.4+, PyTorch 1.12.1+, and MI200 & MI300 GPUs.
- **Reward Model Not Effective for Data Generation**: A brief exchange concluded that the reward model isn't worthwhile for generating data, as it primarily classifies data quality.
- **Boosting AGI Eval**: One user mentioned plans to synthesize SAT, GRE, and MCAT questions to potentially boost AGI evaluations for smaller models, with suggestions to include LSAT questions as well.
- **Epoch Saving Issues**: A user reported issues with epoch saving during training, where it saves at seemingly inconsistent points like 1.05 epochs and then returns to 0.99 epochs. This was recognized as a known but peculiar behavior, possibly related to the steps counter.
- **Finetuning on AMD**: Questions were raised about finetuning on AMD hardware, with a response indicating that Eric has experience with this, though it wasn't confirmed if it is a straightforward process.
  

**Links mentioned**:

- [GitHub - ROCm/flash-attention: Fast and memory-efficient exact attention](https://github.com/ROCm/flash-attention): Fast and memory-efficient exact attention. Contribute to ROCm/flash-attention development by creating an account on GitHub.
- [GitHub - ROCm/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.](https://github.com/ROCm/xformers): Hackable and optimized Transformers building blocks, supporting a composable construction. - ROCm/xformers

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/)** (1 messages):

lore0012: I am no longer hitting the issue.

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1253830860449382578)** (4 messages):

- **HeaderTooLarge error in fine-tuning Qwen2 7b**: A member encountered a `safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge` while running `CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess axolotl/ben_configs/qwen2_first.yaml`. This error occurs when attempting to load checkpoint shards.
- **Local directory issues with Qwen2 7b model**: The fine-tuning configuration works when setting `base_model` to a Hugging Face repository but fails when pointing to a local directory (`/large_models/base_models/llm/Qwen2-7B`). The failure persists even though the folder is a mounted NFS.
- **Frustration with NVIDIA Megatron-LM bugs**: A user expressed frustration after spending a week trying to get megatron-lm to work, encountering numerous errors. An example of the issues faced can be seen in [GitHub Issue #866](https://github.com/NVIDIA/Megatron-LM/issues/866), which discusses a problem with a parser argument in the `convert.py` script.

**Link mentioned**: [[BUG] the argument of parser.add_argument is wrong in tools/checkpoint/convert.py · Issue #866 · NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/issues/866): Describe the bug [https://github.com/NVIDIA/Megatron-LM/blob/main/tools/checkpoint/convert.py#L115](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/checkpoint/convert.py#L115) It must be 'choices=['GPT', 'BERT'],' not 'choice=['GPT', 'BER...

---

### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1254518443789648024)** (5 messages):

- **Newbie asks about dataset suitability**: A new member experimenting with fine-tuning **llama2-13b** using **axolotl** inquired about dataset formatting and content. They asked, "Would this be an appropriate place to ask about dataset formatting and content?"
- **Formatting example for 'Alpaca' dataset**: Another member shared a dataset case using **JSONL** for fine-tuning **Alpaca**. They provided detailed examples, including instructions, input patterns, and expected outputs, and questioned if the LLM could generalize commands like "move to the left" and "move a little to the left."
- **Introducing Rensa for high-performance MinHash**: A member excitedly introduced their side project, **Rensa**, a high-performance MinHash implementation in Rust with Python bindings. They claimed it is 2.5-3x faster than existing libraries like `datasketch` for tasks like dataset deduplication and shared its [GitHub link](https://github.com/beowolx/rensa) for community feedback and contributions.

**Link mentioned**: [GitHub - beowolx/rensa: High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets](https://github.com/beowolx/rensa): High-performance MinHash implementation in Rust with Python bindings for efficient similarity estimation and deduplication of large datasets - beowolx/rensa

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1254711001174245438)** (5 messages):

- **Prompt Style Explained in Axolotl Codebase**: The inquiry about `prompt_style` led to an explanation that it specifies how prompts are formatted for interacting with language models, impacting the performance and relevance of responses. Examples such as `INSTRUCT`, `CHAT`, and `CHATML` were detailed to illustrate different prompt structuring strategies for various interaction types.
- **Example of ReflectAlpacaPrompter Usage**: The `ReflectAlpacaPrompter` class example highlights how different `prompt_style` values like "instruct" and "chat" dictate the structure of generated prompts. The `match_prompt_style` method is used to set up the prompt template according to the selected style.

**Link mentioned**: [OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=4809da1a-b260-413e-bdbe-8b82397846e6)): Understand code, faster.

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1254906057256468573)** (1 messages):

- **Llamafile v0.8.7 releases with upgrades**: [Llamafile v0.8.7](https://discord.com/channels/1089876418936180786/1182689832057716778/1254823644320763987) released with **faster quant operations** and **bug fixes**. An Android version hint was also mentioned.
- **San Francisco hosts major AI events**: **World's Fair of AI** and **AI Quality Conference** will feature prominent community members. Links to [World's Fair of AI](https://www.ai.engineer/worldsfair) and [AI Quality Conference](https://www.aiqualityconference.com/) are provided.
- **Firefox Nightly AI services experiment**: Firefox Nightly consumers can access optional AI services through an ongoing experiment. Details can be explored in the [Nightly blog](https://discord.com/channels/1089876418936180786/1254858795998384239).
- **Latest ML Paper Picks available**: The [latest ML Paper Picks](https://discord.com/channels/1089876418936180786/1253145681338830888) have been shared by a community member.
- **RSVP for upcoming July AI events**: Events include [Jan AI](https://discord.com/events/1089876418936180786/1251002752239407134), [AI Foundry Podcast Roadshow](https://discord.com/events/1089876418936180786/1253834248574468249), and [AutoFIx by Sentry.io](https://discord.com/events/1089876418936180786/1245836053458190438).

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1253796478535860266)** (31 messages🔥):

- **Llamafile Help Command Issue**: A user reported that running `llamafile.exe --help` returns empty output and inquired if this is a known issue. There was no further discussion or solutions provided in the chat.
- **Running Llamafile on Google Colab**: A user, after some initial confusion, successfully ran a llamafile on Google Colab and shared a [link to their example](https://colab.research.google.com/drive/1jWKKwVCQneCTB5VNQNWO0Wxqg1vG_E1T#scrollTo=13ISLtY9_v7g).
- **Llamafile Repackaging Concerns**: A user expressed concerns about the disk space requirements when repackaging llamafiles, suggesting the ability to specify different locations for extraction and repackaging. This sparked a discussion on the potential need for specified locations via environment variables or flags due to large llamafile sizes.
- **New Memory Manager for Cosmopolitan**: A [commit on GitHub](https://github.com/jart/cosmopolitan/commit/6ffed14b9cc68b79d530b23876f522f906173cca) discussing a rewrite of the memory manager to support Android was shared and sparked interest in potentially running llamafile on Android via Termux.
- **Mozilla Nightly Blog Mentions Llamafile**: The [Nightly blog](https://blog.nightly.mozilla.org/2024/06/24/experimenting-with-ai-services-in-nightly/) mentioned llamafile, offering guidance on toggling Firefox configurations to enable local AI chat. This excited the community, with suggestions to provide clearer instructions for new users.
  

**Links mentioned**:

- [no title found](http://localhost:8080`): no description found
- [Tweet from Dylan Freedman (@dylfreed)](https://x.com/dylfreed/status/1803502158672761113): New open source OCR model just dropped! This one by Microsoft features the best text recognition I've seen in any open model and performs admirably on handwriting. It also handles a diverse range...
- [Mozilla Builders](https://future.mozilla.org/builders/): no description found
- [Release llamafile v0.8.7 · Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.7): This release includes important performance enhancements for quants. 293a528 Performance improvements on Arm for legacy and k-quants (#453) c38feb4 Optimized matrix multiplications for i-quants on...
- [Rewrite memory manager · jart/cosmopolitan@6ffed14](https://github.com/jart/cosmopolitan/commit/6ffed14b9cc68b79d530b23876f522f906173cca): Actually Portable Executable now supports Android. Cosmo&#39;s old mmap code required a 47 bit address space. The new implementation is very agnostic and supports both smaller address spaces (e.g....
- [ggerganov - Overview](https://github.com/ggerganov/): I like big .vimrc and I cannot lie. ggerganov has 71 repositories available. Follow their code on GitHub.
- [Google Colab](https://colab.research.google.com/drive/1jWKKwVCQneCTB5VNQNWO0Wxqg1vG_E1T#scrollTo=13ISLtY9_v7g): no description found
- [Feature Request: Support for Florence-2 Vision Models · Issue #8012 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/8012): Feature Description Support for Florence-2 Family of Vision Models needed Motivation A 400M model beating a 15-16B parameter model in benchmarks? Possible Implementation No response

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1253791496432517293)** (24 messages🔥):

- **DPO Training Options Available; ORPO Not Yet Supported**: When asked about the options for DPO and ORPO training with Torchtune, a member shared a [dataset for ORPO/DPO](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) and mentioned that ORPO is not yet supported while DPO has a [recipe available](https://github.com/pytorch/torchtune/blob/f200da58c8f5007b61266504204c61a171f6b3dd/recipes/configs/llama2/7B_lora_dpo.yaml#L9). This was confirmed by another member who added that ORPO would need to be implemented separately from supervised fine-tuning.
- **Training on Multiple Datasets and Epochs Limitation**: A member inquired about training on multiple datasets and setting different epochs per dataset, and was directed to use *ConcatDataset*. It was highlighted that setting different epochs per dataset is not supported.
- **Debate on ChatML Template Use with Llama3**: There was an ongoing discussion about the use of ChatML templates with Llama3, featuring [Mahou-1.2-llama3-8B](https://huggingface.co/flammenai/Mahou-1.2-llama3-8B) and [Olethros-8B](https://huggingface.co/lodrick-the-lafted/Olethros-8B). Participants debated whether using an instruct tokenizer and the base model without special tokens versus with ChatML was appropriate.
- **Phi-3 Model Fine-Tuning Feasibility**: Queries about the feasibility of fine-tuning the Phi-3-Medium-4K-Instruct model using torchtune were addressed. It was suggested to update the tokenizer and add a custom build function in torchtune for compatibility, and include system prompts by prepending them to user messages if desired.
- **Instruction on Using System Prompts with Phi-3**: It was noted that Phi-3 models might not have been optimized for system prompts, but users can still prepend system prompts to user messages for fine-tuning on Phi-3 as usual. A specific flag in the tokenizer configuration [was mentioned](https://github.com/pytorch/torchtune/blob/main/torchtune/models/phi3/_sentencepiece.py#L128) for allowing system prompt usage.
  

**Links mentioned**:

- [lodrick-the-lafted/Olethros-8B · Hugging Face](https://huggingface.co/lodrick-the-lafted/Olethros-8B): no description found
- [flammenai/Mahou-1.2-llama3-8B · Hugging Face](https://huggingface.co/flammenai/Mahou-1.2-llama3-8B): no description found
- [microsoft/Phi-3-mini-4k-instruct · Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct): no description found
- [torchtune/torchtune/models/phi3/_sentencepiece.py at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/torchtune/models/phi3/_sentencepiece.py#L128.): A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.
- [mlabonne/orpo-dpo-mix-40k · Datasets at Hugging Face](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k): no description found
- [torchtune/recipes/configs/llama2/7B_lora_dpo.yaml at f200da58c8f5007b61266504204c61a171f6b3dd · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/f200da58c8f5007b61266504204c61a171f6b3dd/recipes/configs/llama2/7B_lora_dpo.yaml#L9): A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.
- [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/html/2404.14219v1#S2)): no description found
- [microsoft/Phi-3-mini-4k-instruct · System prompts ignored in chat completions](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/51#665f24e07a329f831b1e3e4e.): no description found
- [microsoft/Phi-3-medium-4k-instruct · Hugging Face](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct): no description found
- [config.json · microsoft/Phi-3-medium-4k-instruct at main](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/blob/main/config.json): no description found

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1253788818042126418)** (8 messages🔥):

- **WHERE Function Clarification**: A member asked if the WHERE function could be simplified with conditional operations like `condition * a + !condition * b` and was pointed out that *NaNs* could be an issue.
- **Intel Support Inquiry**: Someone inquired about **Intel support** in tinygrad. Another member responded that **opencl** can be used, but there is no XMX support yet.
- **Monday Meeting Overview**: Key topics for the upcoming Monday meeting at 9:40 a.m. PT include updates on *tinybox*, new profiler, runtime enhancements, and plans for the **0.9.1 release**. Specific agenda items cover enhancements like `Tensor._tri`, llama cast speedup, and mentions of bounties such as improvements in *uop matcher speed* and *unet3d*.
- **Future of Linear Algebra Functions**: A user asked about plans for implementing general linear algebra functions like determinant calculations or matrix decompositions in tinygrad. *No specific response was given in the extracted messages.*

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1254621018971050006)** (2 messages):

- **Buffer view option flagged in tinygrad**: A commit was shared that introduces a flag to make the buffer view optional in tinygrad. The commit message reads, *"make buffer view optional with a flag"* and the associated [GitHub Actions run](https://github.com/tinygrad/tinygrad/actions/runs/9638260193/job/26578693946?pr=5120) was provided.
- **Change in lazy.py raises concerns**: A member questioned if they were doing something wrong as their changes to `lazy.py` resulted in positive (good) and negative (bad) process replay outputs. They were seeking clarity on this unexpected behavior, implying potential issues with their modifications.

**Link mentioned**: [make buffer view optional with a flag · tinygrad/tinygrad@bdda002](https://github.com/tinygrad/tinygrad/actions/runs/9638260193/job/26578693946?pr=5120): You like pytorch? You like micrograd? You love tinygrad! ❤️ - make buffer view optional with a flag · tinygrad/tinygrad@bdda002

---

### **LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1254510317266796731)** (1 messages):

- **Claude Sonnet 3.5 impresses in Websim**: A member was testing **Claude Sonnet 3.5** in Websim and was highly impressed by the model's *"speed, creativity, and intelligence"*. They highlighted features such as "generate in new tab" and shared their experience of trying to *"hypnotize" themselves with the color schemes of different iconic fashion brands*. [Twitter link](https://fxtwitter.com/RobertHaisfield/status/1804945938936668413).

**Link mentioned**: [Tweet from Rob Haisfield (robhaisfield.com) (@RobertHaisfield)](https://fxtwitter.com/RobertHaisfield/status/1804945938936668413): I was "testing" Sonnet 3.5 @websim_ai + new features (mainly "generate in new tab"). I'm FLOORED by this model's speed, creativity, intelligence 🫨😂 Highlights from the lab t...

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1254828730174406738)** (1 messages):

- **MJCET launches AWS Cloud Club**: We are delighted to share that MJCET has launched the FIRST **AWS Cloud Club** in Telangana! This vibrant community provides resources, training, and hands-on experience with Amazon Web Services (AWS), equipping members with essential skills for a tech industry career.
- **Exclusive inaugural event with AWS Hero**: Join the grand inauguration of AWS Cloud Club MJCET on June 28th, 2024, from 10am to 12pm at Block 4 Seminar Hall, featuring **Mr. Faizal Khan**, AWS Community Hero. RSVP via this [meetup link](https://meetu.ps/e/NgmgX/14DgQ2/i) to confirm your attendance.

**Link mentioned**: [Inauguration of AWS Cloud Clubs MJCET, Fri, Jun 28, 2024, 10:00 AM | Meetup](https://meetu.ps/e/NgmgX/14DgQ2/i): **Join Us for the Grand Inauguration of AWS Cloud Club MJCET!** We are delighted to announce the launching event of our AWS Cloud Club at MJCET! Come and explore the world

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
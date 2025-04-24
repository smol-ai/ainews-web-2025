---
id: 474d1a8b-e935-44b5-b8bf-714a5c63822a
title: Andrew likes Agents
date: '2024-03-26T01:11:50.136366Z'
original_slug: ainews-andrew-likes-agents
description: >-
  Andrew Ng’s analysis highlights that incorporating an iterative agent workflow
  significantly boosts coding performance, with GPT-3.5 agents achieving up to
  95.1% accuracy on the HumanEval benchmark, surpassing GPT-4’s zero-shot 67%.
  Meanwhile, the Stable Diffusion community is actively developing new models
  like Cyberrealistic_v40 and Platypus XL, alongside innovative techniques for
  upscaling and image quality enhancement.
tags:
  - andrew-ng
  - agents
  - gpt-3.5
  - gpt-4
  - stable-diffusion
  - upscaling
  - local-llm
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 3/21/2024-3/25/2024. We checked [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **22** Discords (**342** channels, and **12281** messages) for you. Estimated reading time saved (at 200wpm): **1173 minutes**.

[Andrew Ng's The Batch writeup on Agents](https://www.reddit.com/r/singularity/comments/1bl3s9r/andrew_ng_cofounder_of_google_brain_former_chief/?utm_source=ainews&utm_medium=email) made a splash across all platforms this weekend:

> Devin’s splashy demo recently received a lot of social media buzz. My team has been closely following the evolution of AI that writes code. We analyzed results from a number of research teams, focusing on an algorithm’s ability to do well on the widely used HumanEval coding benchmark. You can see our findings in the diagram below. 
>
> GPT-3.5 (zero shot) was 48.1% correct. GPT-4 (zero shot) does better at 67.0%. However, the improvement from GPT-3.5 to GPT-4 is dwarfed by incorporating an iterative agent workflow. Indeed, wrapped in an agent loop, GPT-3.5 achieves up to 95.1%. 

 ![image.png](https://assets.buttondown.email/images/b3745325-26e9-47d6-a195-5868d74082c8.png?w=960&fit=max) 

Nothing here is new to people who have studied the agents field, but Andrew's credibility and agent framework (very close to [Lilian Weng](https://twitter.com/lilianweng/status/1673535600690102273?lang=en) + the recent new metagame of multiagent collaboration) sells it.

We published [The Unbundling of ChatGPT](https://www.latent.space/p/feb-2024) today. Also [Emad stepped down from Stability](https://stability.ai/news/stabilityai-announcement), and there are [more Sora videos](https://openai.com/blog/sora-first-impressions?utm_source=ainews&utm_medium=email) out, make sure to check out the Don Allen Stevenson III one.

---

**Table of Contents**

[TOC] 


---

# REDDIT

> we've added more subreddits, and are synthesizing topics across them. Comment crawling still not implemented but coming along.

**Stable Diffusion Models and Techniques**

- **New Stable Diffusion models and techniques** are being developed, such as Cyberrealistic_v40, Platypus XL, and SDXL Lightning for generating Naruto-style images. ([Playing with Cyberrealistic_v40](https://www.reddit.com/gallery/1bmqyis), [Still Liking Platypus XL](https://i.redd.it/zvltrzu96dqc1.png), [Naruto (Outputs from SDXL Lightning)](https://www.reddit.com/gallery/1bmmuuj))
- /r/StableDiffusion: **LoRA and upscaling methods** are being explored to improve image quality, such as cartoonizing images while preserving content and a general purpose negative prompt LoRA to eliminate common problems. ([best LORA or method with sd1.5 to cartoon-ize an image while keeping content? (flatten shading, reinforce outlines)](https://www.reddit.com/r/StableDiffusion/comments/1bmjdhs/best_lora_or_method_with_sd15_to_cartoonize_an/), [General purpose negative prompt?](https://www.reddit.com/r/StableDiffusion/comments/1bmgj70/general_purpose_negative_prompt/))
- /r/StableDiffusion: **New workflows and extensions** are being developed for Stable Diffusion, such as BeautifAI for upscaling in ComfyUI, FrankenWeights for mixing model weights, and integrating Prompt Quill expansion in Fooocus. ([BeautifAI - Image Upscaler & Enhancer - ComfyUI](https://www.reddit.com/r/StableDiffusion/comments/1bn5jdu/beautifai_image_upscaler_enhancer_comfyui/), [It's alive! FrankenWeights is coming... [WIP]](https://www.reddit.com/r/StableDiffusion/comments/1bmemjs/its_alive_frankenweights_is_coming_wip/), [Prompt Quill in Fooocus](https://www.reddit.com/r/StableDiffusion/comments/1bmm615/prompt_quill_in_fooocus/))

**Local LLM Deployment and Optimization**

- /r/LocalLLaMA: **Deploying large language models locally** is a popular topic, with discussions around hardware requirements, inference speed, and model selection for different use cases. ([Would it make sense to stick a P40 24GB in with a 3090 to have 48GB VRAM?](https://www.reddit.com/r/LocalLLaMA/comments/1bn43rt/would_it_make_sense_to_stick_a_p40_24gb_in_with_a/), [Best output quality for 4090 & 64GB RAM?](https://www.reddit.com/r/LocalLLaMA/comments/1bmk3c7/best_output_quality_for_4090_64gb_ram/), [What is your computer specs?](https://www.reddit.com/r/LocalLLaMA/comments/1bmu9sh/what_is_your_computer_specs/))
- /r/LocalLLaMA: **Optimizing LLM performance** is an active area of research, with discussions around architectures for reasoning, finetuning strategies, and serving multiple users efficiently. ([What architecture will give us a reasoning LLM ?](https://www.reddit.com/r/LocalLLaMA/comments/1bn48cl/what_architecture_will_give_us_a_reasoning_llm/), [All work and no play makes your LLM a dull boy; why we should mix in pretraining data for finetunes.](https://www.reddit.com/r/LocalLLaMA/comments/1bmslfq/all_work_and_no_play_makes_your_llm_a_dull_boy/), [Is it possible to serve mutliple user at once using llama-cpp-python ?](https://www.reddit.com/r/LocalLLaMA/comments/1bmw12y/is_it_possible_to_serve_mutliple_user_at_once/))
- /r/LocalLLaMA: **Guides and resources** are being developed to help users get started with local LLMs, from beginner to advanced levels. ([New user beginning guide: from total noob to well-informed user, part 1/3, another try...](https://www.reddit.com/r/LocalLLaMA/comments/1bmvtyb/new_user_beginning_guide_from_total_noob_to/))

**Machine Learning Research and Techniques**

- /r/MachineLearning: New **machine learning architectures and techniques** are being proposed and discussed, such as Treeformers using hard attention and decision trees for causal language modeling. ([[P] Treeformer: hard attention + decision trees = causal language modelling](https://www.reddit.com/r/MachineLearning/comments/1bmmqqq/p_treeformer_hard_attention_decision_trees_causal/))
- /r/MachineLearning: **Optimization techniques** for deploying ML models are being explored, such as using TensorRT for fast PyTorch model inference. ([[D] Looking for fastest inference way to run a pytorch model on TensorRT](https://www.reddit.com/r/MachineLearning/comments/1bmnn5j/d_looking_for_fastest_inference_way_to_run_a/))
- /r/MachineLearning: **Debugging and improving ML models** is an ongoing challenge, with discussions around understanding and fixing issues like spiking test loss. ([[D] Does anyone know why my test loss is spiking so crazily?](https://www.reddit.com/r/MachineLearning/comments/1bmor9g/d_does_anyone_know_why_my_test_loss_is_spiking_so/))

**AI Assistants and Applications**

- /r/OpenAI: **AI assistants are being used in new ways**, such as mediating arguments to provide a neutral perspective and helping with coding tasks. ([Mediating Arguments with ChatGPT](https://www.reddit.com/r/OpenAI/comments/1bmgh5w/mediating_arguments_with_chatgpt/), [Coding LLM that runs on 3090](https://www.reddit.com/r/LocalLLaMA/comments/1bmfu9g/coding_llm_that_runs_on_3090/))
- /r/StableDiffusion: **New AI applications** are being developed, such as AI influencers, interactive "AI Brush" tools, and immersive experiences to explore AI-generated worlds based on images. ([Here is my first 45 days of wanting to make an AI Influencer and Fanvue/OF model with no prior Stable Diffusion experience](https://www.reddit.com/r/StableDiffusion/comments/1bn0kf8/here_is_my_first_45_days_of_wanting_to_make_an_ai/), [Quick Breakdown of my interactive "AI Brush" build with StreamDiffusion](https://v.redd.it/4yna3y1t2bqc1), [Excited for the future](https://www.reddit.com/r/OpenAI/comments/1bmuqae/excited_for_the_future/))

**Memes and Humor**

- **AI-generated memes and humorous content** continue to be popular, poking fun at the current state of AI. ([Do not generate a tree using a model trained on p*rn](https://i.redd.it/gq2xicv17cqc1.jpeg), ["Don't ever buy no weed from the gas station bro"](https://i.redd.it/s09ogpjy4cqc1.png), [It do be like that](https://i.redd.it/9tfrhr4p9eqc1.png))


# PART X: AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs

**Model Releases & Updates**

- [Mistral AI released a new 7B 0.2 base model with 32k context window](https://twitter.com/osanseviero/status/1771654833830821949), announced at a hackathon (13k views)
- [Quantized 4-bit Mistral 7B models released](https://twitter.com/danielhanchen/status/1771737648266178801), enabling 2x faster inference with 70% less VRAM using QLoRA finetuning (14k views)
- [Mistral 7B 0.2 expected to outperform Yi-9B](https://twitter.com/teortaxesTex/status/1771665411857096870) (486 views)


**Open Source Efforts & Challenges**

- [Stability AI's Emad Mostaque out following investor mutiny and staff exodus](https://twitter.com/stanfordnlp/status/1771671300823826867) (11k views)
- [Open source AI would be better if AAA games weren't "total shit" for the last decade](https://twitter.com/teortaxesTex/status/1771651366290636913), making high-end GPUs not worth it for gaming (1.7k views)
- [Open-source AI isn't real without distributed pretraining](https://twitter.com/generatorman_ai/status/1771516761206395164), can't depend on VCs spending millions then giving it away (468 views)

**Emerging Applications & Demos**

- [Financial agent application](https://twitter.com/virattt/status/1771614341831201193) built with LangChain, can get stock prices, financials, market news (46k views)
- [Telegram proxy setup guide](https://twitter.com/rameerez/status/1771602942287626281) to circumvent potential ban in Spain using built-in proxy feature (50k views)
- [Dancing robot powered by Mistral 7B](https://twitter.com/sophiamyang/status/1771667841181233221) demoed at hackathon (9.8k views)
- [Claude-to-Claude conversations](https://twitter.com/AISafetyMemes/status/1771768138042122301) induce concerning outputs like "psychotic breaks" (53k views)


---

# PART 0: Summary of Summaries of Summaries


- **Mistral's New 7B v0.2 Base Model Drops**: Mistral AI casually released their new **Mistral 7B v0.2 Base** model at the [@cerebral_valley hackathon](https://x.com/alexreibman/status/1771608346635751541?s=46), featuring a 32k context window and other improvements detailed in their [release notes](https://x.com/mistralailabs/status/1771670765521281370?s=46). The AI community is abuzz with the implications and benchmarking results of this significant update.

- **Stability AI CEO Emad Mostaque Resigns**: In a major shakeup, **Emad Mostaque** [resigned as CEO of Stability AI](https://stability.ai/news/stabilityai-announcement) to pursue decentralized AI. Interim co-CEOs Shan Shan Wong and Christian Laforte will lead the search for a permanent replacement. Speculation is rife about the company's future direction and commitment to open-source initiatives amidst this leadership transition.

- **Anthropic's Claude Shines Despite Limitations**: Users praise **Anthropic's Claude** for its performance and context, especially the self-moderated version, but express frustration with the strict 1M token per day rate limit on the 200k context window. The **$500 scale plan** is suggested as a more accessible option for extensive usage, while the potential of Claude's API for open-source development generates excitement.

- **Optimizers and Architectures Advance LLMs**: Novel optimizers like **GaLore** and architectures like **DenseFormer** are pushing the boundaries of language model training efficiency and performance. Discussions revolve around GaLore's significant VRAM savings and potential over-training risks, while DenseFormer's [depth-weighted averaging](https://arxiv.org/abs/2402.02622) shows promising perplexity improvements. The community eagerly awaits further developments in these areas.

- **AI Assistants and Agents Evolve**: Projects like **Open Interpreter's 01 Light**, a fully open-source personal AI agent, and **World Simulator** from Nous Research are capturing the community's imagination with their engaging experiences and potential for customization. Meanwhile, frameworks like **LangChain** are enabling more sophisticated decision-making and task automation for AI agents, as evidenced by various shared guides and tutorials.


---



# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD Ecosystem Buzzing**: The community is actively discussing **Stable Diffusion** models, particularly in anticipation of the upcoming **SD3** release, with buzz around potential improvements and comparative analysis with models such as **SDXL**. Issues surrounding compatibility with AMD GPUs were also raised, with members sharing solutions and workarounds.

- **AI Art at a Click, But Not Without Hiccups**: Frustrations were voiced over online AI image generation services like **Civitai** and **Suno**, citing content restrictions and types of content generated. Community members shared resources such as [Stable Cascade Examples](https://comfyanonymous.github.io/ComfyUI_examples/stable_cascade/) to showcase different model capabilities.

- **Regulatory Rumbles**: A polarized debate unfolded on the implications of regulation in AI technology. Ethical considerations were weighed against fears of stifling innovation, reflecting a community conscious of the balance between open-source development and proprietary constraints.

- **Tech Support Tribe**: A knowledge-sharing atmosphere prevails as newbies and veterans alike navigate technical tribulations related to model installations. Resources for learning and troubleshooting were shared, including direct links to support channels and expert advice within the community.

- **Connecting AI Threads**: Various links were circulated for further information and utilities, such as a [Stable Diffusion Glossary](https://stable-diffusion-art.com/glossary/), and a comprehensive multi-platform package manager for Stable Diffusion [StabilityMatrix](https://github.com/LykosAI/StabilityMatrix/blob/main/README.md). These tools are meant to aid understanding and enhance usage of Stable Diffusion products among AI engineers.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Innovating Within Memory Limits**: A [recent pull request](https://github.com/pytorch/torchtune/pull/527) on the **PyTorch** `torchtune` repository allows full model fine-tuning while keeping the memory footprint under 16GB, enabling users with consumer-grade GPUs to train more efficiently.
- **Enhancing Fine-Tuning Capabilities**: The latest release (v0.10.0) of the [Hugging Face PEFT](https://github.com/huggingface/peft/releases/tag/v0.10.0) includes LoftQ, which improves fine-tuning for large models.
- **ORPO Implementation with Mistral**: A user reported effective application of **ORPO TRL** on **Mistral-7B-v0.2** using the *argilla/ultrafeedback-binarized-preferences-cleaned* dataset, suggesting that the method has potential for further optimization.
- **Debating Best Practices for Training AI Models**: Discussions across channels engage with the challenges of **multi-turn dialogue training in ORPO**, the importance of standardized formats, and pressing issues with using *Ollama templates* and different quant models.
- **Model Performance Milestones**: The community celebrates the performance of new models like [sappha-2b-v3](https://huggingface.co/Fizzarolli/sappha-2b-v3) and [MasherAI-v6-7B](https://huggingface.co/mahiatlinux/MasherAI-v6-7B) which reportedly surpassed benchmarks after fine-tuning with **Unsloth** on Gemma-2b.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Users Clash Over Image Generation**: The Pro users discussed **Perplexity's image generation** feature and noted that turning off the Pro toggle on the web version allows for image generation using Writing mode.

- **Model Showdown: Claude Opus vs. GPT-4 Turbo**: Engineering chatter touched upon comparing **Claude 3 Opus and GPT-4 Turbo**, highlighting that GPT-4 Turbo can compile Python files, unlike Perplexity.

- **Stability AI Strikes Local Chord**: There was a buzz about **Stability AI's local models**, like SDXL, and the tradeoff between performance and the hefty costs of running these tools on personal hardware. 

- **Perplexity Puzzles and Potential**: Users were bewildered by some aspects of Perplexity, including intrusive search triggers and unrelated prompts, while also envisioning features like **Claude 3 Opus's API** and integration with iOS Spotlight search.

- **Coding Within Token Boundaries**: A crucial tip for engineers working with the **Perplexity API** was to heed the **16,384 token limit**, with suggestions to use tools like OpenAI's tokenizer to gauge token counts accurately and adhere to the limit for optimal operation.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Taming the VRAM Beast for AI**: Engineers debate the best GPUs for running AI models, with the RTX 3090's 24GB VRAM being a popular choice. Compatibility issues between differing GPU makes, such as AMD and NVIDIA, were noted when configuring multi-GPU setups.

- **Local LLMs Stirring a Tech Revolution**: Discussions about the feasibility of distributed computing for LLMs revealed skepticism due to high CPU utilization when using ZLUDA with ROCm, despite the appeal of local computing akin to the Linux/LAMP era. 

- **Docs Unleashed and Multi-Model Mania**: LM Studio launched a [new documentation site](https://lmstudio.ai/docs) and also introduced a Multi Model Session feature, which is explained in a [tutorial video](https://youtu.be/4fdZwKg9IbU?feature=shared&t=357).

- **LM Studio's Growing Pains and Performance Quirks**: Users reported issues from **high CPU usage** to models outputting gibberish, which sometimes were resolved by a simple restart. Compatibility was questioned for older hardware like the RX 570 with ROCM, and errors like "Exit code: 42" when loading models signaled a need for continued troubleshooting.

- **Open-interpreter Unpacked and GGUF Model Performance**: Open-interpreter issues included connection problems and discussions on various GGUF-compatible models' performance. The Open Interpreter device attracted interest, with options to 3D print it oneself using free STL files from [01.openinterpreter.com/bodies/01-light](https://01.openinterpreter.com/bodies/01-light). Meanwhile, non-blessed model errors prompted gaze towards issue [#1124 on open-interpreter's GitHub](https://github.com/OpenInterpreter/open-interpreter/issues/1124).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Makes Leap to Linux**: Users have managed to get **Open Interpreter** running on Ubuntu 22.04, discussing microphone support and client-server mechanics, signaling a push towards cross-platform compatibility.

- **DIYers Assemble! The O1 Light Craze**: Enthusiastic members of the community are sharing tips on 3D printing and assembling their own **01 Light** devices, with dedicated Discord spaces popping up to share build experience and design tweaks.

- **AI-Powered Engineering Discussions Heat Up**: Technical conversations are expanding around the **Open Interpreter**, with focus on running the 01 server on various machines, whether low-spec or cloud-based, and enhancing installers for user-friendliness. Developers are also brainstorming integrations for **Groq** and extending *01 Light* functionalities.

- **Community Contributions Unleash Open Source Power**: The AI engineer community is diving into contributions for the **Open Interpreter** project, focusing on app development, performance of different **LLMs**, and potential desktop app for **Apple silicon devices**.

- **Open-Source AI Assistants Set the Stage**: An insightful YouTube video on the **01 Lite** titled "Open Interpreter's 01 Lite - WORLD'S FIRST Fully Open-Source Personal AI AGENT Device" has been highlighted, showing off the capabilities of this homegrown AI assistant. An edited live stream has also been shared to provide a concise overview of the **01 software** [Open Interpreter's 01 Lite](https://www.youtube.com/watch?v=Q_p82HtBqoc).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **EU Data Laws Challenge LAION's Efficiency**: LAION datasets may be underperforming in comparison to US datasets, largely due to the EU's stringent regulations. The use of synthetic data and forming collaborations in less restrictive regions were mentioned as possible workarounds, humorously termed "data laundering."

- **Leadership Shuffle at Stability AI**: Emad Mostaque has stepped down as CEO of Stability AI, with the company confirming his resignation and the appointment of interim co-CEOs Shan Shan Wong and Christian Laforte, who will oversee the search for his replacement. There is speculation about the impact on the company's future and its commitment to open-source ([Stability AI Press Release](https://stability.ai/news/stabilityai-announcement)).

- **Comparing SD3 to DALL-E 3**: Discussions indicate that the SD3 model can match some aspects of DALL-E 3 performance, but struggles with complex interaction understanding, leading to collage-like image assembly, rather than cohesive concept blending.

- **AI Ethics Debate Surfacing Amidst Industy Drama**: A recent conversation on Twitter about the motives behind leading AI industry figures led to a guild-wide debate regarding the ethical responsibilities of developers and researchers, along with the impact of AI "celebrity" culture on social media.

- **AMD GPUs Fall Behind in AI Support**: Guild members expressed dissatisfaction with AMD's support for machine learning workloads when compared to NVIDIA's offerings. The lack of consumer-level ML support is viewed as a potential oversight given the rise of models like Stable Diffusion.

- **Andrew Ng Foresees AI Workflow Evolution**: Google Brain's co-founder Andrew Ng predicts that **AI agentic workflows** could surpass next-generation foundation models this year by iterating over documents multiple times. Current one-shot LLM approaches need to evolve ([Reddit Highlight](https://www.reddit.com/r/singularity/comments/1bl3s9r/andrew_ng_cofounder_of_google_brain_former_chief/)).

- **MIT Accelerates Image Generation Tech**: MIT's CSAIL developed a method that speeds up the image generation process for tools like Stable Diffusion and DALL-E by 30 times, using a streamlined single-step teacher-student framework without compromising on image quality ([MIT News Article](https://news.mit.edu/2024/ai-generates-high-quality-images-30-times-faster-single-step-0321)).

- **NVIDIA Addresses Diffusion Model Training Hurdles**: NVIDIA's recent blog post discusses the improvements in training diffusion models, including the EDM2 code and model release. They address style normalization issues that could be overcome with changes similar to those in EDM2 ([NVIDIA Developer Blog Post](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/)).

- **Unet's Future in Question with Rise of Linear Networks**: Guild members debated the relevance of advancements in Unet given the rise of linear network models for image generation. Although layer norm and traditional normalization methods are questioned, their integral part in network functionality remains a topic of discussion.

- **Large Language Models Show Resilience with Pruning**: Insights reveal that large language models (LLMs) retain performance even when middle blocks are removed, hinting at the redundancy of certain segments. This has encouraged a deeper look into the architecture of linear networks and their potential for strategic pruning.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Shipping Queries and DIY Solutions**: Members are curious about shipping timelines for an unnamed product, anticipating **summer availability**. They discussed alternative DIY options in the absence of specific release dates.

- **AI Model Innovation and Optimism**: Enthusiasm for **Claude**'s impact on open-source projects is high, and an implementation of **Raptor** without pretraining is reported to summarize 3b model transcripts in just 5 minutes. References to [FastAPI's](https://fastapi.tiangolo.com) ease-of-use for back-end development were shared, alongside a note about Suno.AI's ability to curate Spotify playlists.

- **Deployment and Persuasion Techniques Analyzed**: Kubernetes is leveraged for deploying Nous models, while a pre-registered [arXiv study](https://arxiv.org/abs/2403.14380) analyses the persuasive power of LLMs. Platforms like [ArtHeart.ai](https://artheart.ai/) are recognized for AI-driven art creation, with BitNet 1.5's quantized-aware training reviewed for inference speedups.

- **Weaving Worlds and Aiding Therapists with AI**: The World Simulator project revealed a remarkable level of engagement, while work on an AI therapist dubbed **Thestral** aims to use the LLaMA 70B model. The community is actively discussing the **ethical constraints** of Opus from Claude 3, the impact of **refusal prompts** in models like Hermes 2 Pro, and manipulation techniques to circumvent LLM limitations known as the "Overton Effect."

- **LLMs, Tuning, and Refinement Questions**: Debates arose around the inclusion of few-shot prompts in SFT datasets and the quest for tiny LLMs, with recommendations to watch **Andrej Karpathy's** videos. The importance of causal masking and the mysteries behind Llama's tri-layer feedforward design incited discussions, circling an [arXiv paper on SwiGLU](https://arxiv.org/pdf/2002.05202.pdf) nonlinearity.

- **Parenting, Open Sourcing, and RAFT's Prominence**: Members touched upon their experiences with parenthood while seeking open-source options like a Wikipedia RAG Index. The conversation highlighted a departure towards a promising retrieval-augmented fine-tuning method called **RAFT**, which was discussed in a shared paper and can be explored in the [Gorilla GitHub repository](https://github.com/ShishirPatil/gorilla/tree/main/raft).

- **Casual Chats and World-Sim Tech**: A member hinted at changing language settings on Tenor.com, displaying a shared [Grim Patron GIF](https://tenor.com/view/everyone-get-in-here-grim-patron-gif-26273450), while a simple "helloooo" from another brightened the chat.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora Shapes the Future of Filmmaking**: OpenAI's **Sora** is celebrated for empowering artists and filmmakers to create groundbreaking surreal art. Director Paul Trillo lauds Sora for enabling the visualization of previously unimaginable ideas, with its potential highlighted on the [OpenAI blog](https://openai.com/blog/sora-first-impressions).

- **Diving into AI's Cultural Compass**: A hot topic in AI discussions is the perceived Western liberal-centrist bias in language models like GPT, sparking debates on whether there should be multiple culturally aligned AI versions. Efforts to align AI with non-Western norms are facing challenges, and the "Customize ChatGPT" feature was cited as a tool for users to personalize AI responses with their values.

- **GPT-4's Evolving Features and Access Concerns**: Members noticed a reduction in the **Custom GPT pin limit** and sought a keyboard shortcut for shared GPT access. OpenAI confirmed the capability of **GPT-4 with Vision** to read images, while also announcing the end of the **ChatGPT plugins beta** through a [discontinuation notice](https://help.openai.com/en/articles/8988022-winding-down-the-chatgpt-plugins-beta).

- **Refining AI to Enrich User Experience**: Strategies are shared for enhancing AI's creative writing, narrative style, and coding output quality. A user faced issues with the `.Completion` endpoint due to an OpenAI SDK update, and was directed to the [v1.0.0 Migration Guide](https://github.com/openai/openai-python/discussions/742) on the openai-python GitHub repository for assistance.

- **Enabling Accessibility in Vision**: Privacy-sensitive advice was provided to a member looking to improve image recognition for disabled individuals, with an emphasis on writing up the issue for a Discord suggestions channel. Users also explored prompt engineering tactics to craft AI with specific personalities, and to refine the generation of hypothesis paragraphs avoiding generic statements.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**AI Art Prompt Guide Quest**: Users are seeking advice on crafting prompts for AI-generated art, though specific resources weren't provided.

**Blenderbot's Role-Play**: Discussions highlight Blenderbot's ability to exhibit consistent character traits during interactions, in contrast to AI that acknowledges its non-human nature.

**GPU Operation Showdown**: A technical debate unfolded around the execution speed differences between multiplication and conditional checking on GPUs. Look into 'iq's work was suggested for further insights.

**Complex Creativity for ChatGPT**: A user requested a linguistically diverse and creative prompt for ChatGPT, prompting another to exclaim over the prompt's complexity.

**Optimizing GPU Inference**: The community explored methods and libraries like TensorRT-LLM and exLLama v2 for optimizing large language model inferencing on GPUs, with suggestions for tools ideal for simultaneous multi-user serving.

**Rust's Rising Star**: Conversations around converting the GLiNER model to Rust via the *Candle* library noted benefits including reduced dependencies and suitability for production, with GPU compatibility confirmed.

**Efficient Coding with Federated Learning**: [An open-source GitHub project](https://github.com/ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting) demonstrates an energy-efficient approach to federated learning for load forecasting.

**Compiling the Stable Diffusion Compendium**: A plethora of resources and guides for Stable Diffusion have been shared by community members, including [civitai.com](https://civitai.com/articles/2054) for comprehensive learning on Stable Diffusion.

**Deck Out Your Memory – Diffusers Edition**: An experimental tool for estimating the **inference-time memory requirements of `DiffusionPipeline`** has been [released for feedback](https://github.com/huggingface/diffusers/discussions/7434).

**SegGPT: The Contextual Segmentor**: Introducing [SegGPT on HuggingFace](https://huggingface.co/docs/transformers/main/en/model_doc/seggpt), a model with impressive one-shot segmentation that can be trained for various image-to-image tasks.

**BLIP-2 Ups the Fusion Game**: In vision-language model fusion, [BLIP-2](https://arxiv.org/abs/2301.12597) has been recommended for connecting pre-trained image encoders with language models, further elaborated in the [transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/blip-2).

**Embedding Precision with Quantization**: [Embedding Quantization](https://huggingface.co/blog/embedding-quantization) for Sentence Transformers brings major search speed improvements without compromising retrieval accuracy.

**Catering to the German Learners**: A GPT-powered German language learning tool named *Hans* promises enhanced user experience for German learners and is available on the GPT Store.

**All-MiniLM-L6-v2 Download Dilemma**: A user looked for assistance in downloading and training the **all-MiniLM-L6-v2 model**, emphasizing the power of community support for model implementation.

**Revolutionizing Decision-Making with Langchain**: An article on Medium posits Langchain as a transformative approach to how language agents resolve problems, available on [Medium](https://medium.com/ai-advances/language-agent-tree-search-with-langchain-revolutionizing-decision-making-with-language-models-a46c991397f1).

**Diving Into Data's Importance**: A shared [arXiv paper](https://arxiv.org/pdf/2212.03533.pdf) emphasizes the significance of data as a potential critical influencing factor, reminding us of the indispensable value of quality data.

**NEET/JEE Data Quest**: A dataset of NEET/JEE exams is being sought for training MCQ answer generators, indicating the intersection of AI technology and educational resources.

**AI on the Forefront**: Recurrent Neural Notes newsletter discusses the potential limits of AI, possibly providing nuanced insights on future AI capabilities available on [Substack](https://open.substack.com/pub/thernn/p/rnn-7-the-real-limits-of-ai?r=kxtnk&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Twitter Sneak Peek on Human-LlamaIndex Workflow**: A new template was introduced to streamline interactions between humans and **LlamaIndex's agents**, slated to reduce intrusiveness for users. The details and a preview were shared on [Twitter](https://t.co/Z16QPCWFmG).

**Integrating Custom LLMs with LlamaIndex**: Leonie Monigatti detailed the process of incorporating custom Language Models (LLMs) into **LlamaIndex**, with an explanation available on [LinkedIn](https://t.co/DBjXGkLFkg).

**Guide to Building RAG Agent for PDFs**: A tutorial by Ashish S. on creating a **LlamaParse**-powered RAG flow for PDF files was published and can be viewed in its entirety via this [Tweet](https://t.co/vIANM2Byel).

**New LlamaIndex Python Documentation Released**: **LlamaIndex** has updated its Python documentation to feature example notebooks better, improved search, and clearer API layouts, announced in a [Twitter post](https://t.co/FAuBj5gnCC).

**LlamaIndex Community Tackles Integration and Documentation Challenges**: Discussions in the community highlighted various integrations with **Merlin API** and **LocalAI**, an inquiry about the logic in LlamaIndex's evaluation process, conflicting documentation post v0.10 updates, requests for examples of multi-agent chatbots, and turning Python functions into LlamaIndex tools. Users exchanged resources, including several [documentation](https://github.com/mudler/LocalAI) links and [GitHub code examples](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/extractors/llama-index-extractors-entity/llama_index/extractors/entity/base.py).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Whisper for Video Processing**: Community members are seeking a video processing tool comparable to OpenAI's Whisper, and suggestions included [Video Mamba](https://huggingface.co/blog/vladbogo/video-mamba), Twelve Labs, and [videodb.io](https://videodb.io).

- **OpenAI's Sora Gains Traction Amongst Creatives**: OpenAI's introduction of **Sora** has garnered positive feedback from artists, showcasing the tool's versatility in generating both realistic and imaginative visuals.

- **Google Confounds with AI Services**: Discussion revealed confusion between Google's AI Studio and Vertex AI, particularly with the former's new 1 million token context APIs, drawing comparisons with OpenAI's API for model deployment.

- **AI Wearables Gaining Popularity**: The ALOHA project, an open-source AI wearable, is discussed amid conversations about the rise of AI wearables. Pre-orders for another AI wearable, [Compass](https://x.com/itsmartynask/status/1771890769865187648), began, indicating a thriving interest in local, personal AI solutions.

- **Efficiency in LLMs With LLMLingua**: Microsoft's LLMLingua was shared as a promising tool for compressing prompts and KV-Cache in Large Language Models (LLMs), achieving significant compression rates with little loss in performance.

- **Insider AI Discussion Takes Podcast Form**: A podcast episode highlighted with a [tweet](https://twitter.com/swyx/status/1771255525818397122) provided insights into major AI companies, sparking interest in the AI community.

- **AI Unbundling Trend Spotted**: An essay featured on [latent.space](https://latent.space/p/feb-2024) discussed the unbundling of ChatGPT, suggesting that specialized AI services are becoming more popular as user growth for generalist models stagnates.

- **Paper Club Hiccups**: The llm-paper-club-west faced technical difficulties with speaking rights on Discord, causing the meeting to switch over to Zoom and raising awareness for the need to streamline access for future online gatherings.

- **Ideas and Music Flow in AI in Action Club**: The club had vibrant discussions on tensor operations, coding best practices for LLMs, and spontaneous sharing of music evoking the calm of night from Slono on [Spotify](https://open.spotify.com/artist/1rWeYVkrGXRqaD8e0kwMbc?si=xu1E7Di8T_OUpQvT46f-BA). They also released a [schedule](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0) for upcoming sessions on AI topics.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**GaLore Optimization Sparks Debate**: The [GaLore optimizer](https://github.com/jiaweizzhao/GaLore/issues/6) discussion highlighted its VRAM savings abilities but also raised the question of potential over-training due to "coarseness." Some engineers are eager to test GaLore out, especially in light of the new **Mistral v0.2 Base Model** release, which now has a 32k context window.

**Fine-Tuning Large Language Models on a Budget**: Technical discussions surfaced around fine-tuning a 7b model within 27gb of memory, with a spotlight on a GitHub repository called [torchtune](https://github.com/pytorch/torchtune) that allows for efficient fine-tuning without Huggingface dependencies. A specific [pull request](https://github.com/pytorch/torchtune/pull/527) was recommended to review full fine-tune methods requiring less than 16GB of RAM.

**TypeError Troubles and Help Channel Support**: A member grappling with a `TypeError` in "examples/openllama-3b/qlora.yml" was directed to a specialized help channel (#1111279858136383509) for expertise in resolving it. This exemplifies the collaborative environment, urging members to specific resources for technical resolutions.

**Medical Model Publishing Dilemma**: The decision whether to publicly share a preprint of a medical model in the midst of journal review sparked a discussion on the trade-offs of early disclosure. The conversation underscores the importance of strategic research dissemination in the field.

**Open Calls for Developer Recognition and Business Collaboration**: [CHAI announced prizes for LLM developers](https://chai-research.typeform.com/chaiprize), encouraging community contributions, whereas businesses were invited to share their applications of Axolotl confidentially, alluding to the value of real-world use-case narratives in furthering AI technology.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Midnight 70B Unleashed into the Roleplay Realm**: The *Midnight 70B* model, tailored for storytelling and roleplay with lineage from Rogue Rose and Aurora Nights, is now up for grabs sporting a 25% discount, tagged at **$0.009/1k tokens** on [OpenRouter](https://openrouter.ai/models/sophosympatheia/midnight-rose-70b).
- **OpenRouter Refines Cost Tracking Tools**: OpenRouter implements an advanced **Usage Analytics** feature for real-time cost tracking and has made a **Billing Portal** available for more efficient credit and invoice management.
- **Noromaid Mixtral and Bagel Prices Readjusted**: Prices for running the Noromaid Mixtral and Bagel models no longer include discounts, with new prices set at **$0.008/1k tokens** for Mixtral while Bagel comes at **$0.00575/1k tokens**.
- **Claude 3 & Grok:** In multi-model discussions, Claude 3's self-moderated version gained traction for improved filtering and the Grok model generated debate; its performance deemed satisfactory against premium alternatives but cost-prohibitive. Users voiced preferences for longer context lengths and raised quality differences in model completions between OpenRouter and direct API usage.
- **OpenRouter DDoS Sufferance and API Response Issue**: OpenRouter faced a DDoS attack leading to service instability, since resolved, and users observed that citation data from Perplexity isn't provided in OpenRouter's API responses as expected.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **PyTorch Struggles with MPS**: An ongoing effort to improve the **MPS backend in PyTorch** is being tackled, with notable issues like tensor copying since September 2022 being a hurdle. This work is anticipated to enhance performance for model testing and finetuning locally. 

- **Token Blocking Contested For LLM Training**: A **debate on token block strategies for language model pretraining** triggered a discussion on the merits of overlapping vs. non-overlapping sequences and their impact on model efficacy, touching on the importance of beginning-of-sentence tokens.

- **AMD Driver Dilemma Spurs Discussion**: Comparisons between AMD Radeon and Nvidia GPU drivers sparked debates, centering on driver inadequacies and the possibility of AMD open-sourcing their drivers. Some participants considered the potential for activist investor action to prompt change at AMD.

- **Machine Learning Model Merger Methodologies**: New **model merging methods** are being created with the aim to outdo existing techniques such as DARE, though these are still in the experimental phase and require additional testing and validation.

- **New ML Architectures Brim with Promise**: Innovations like **[DenseFormer](https://arxiv.org/abs/2402.02622)** and **[Zigzag Mamba](https://arxiv.org/abs/2403.13802)** suggest improvements in perplexity and diffusion model memory usage respectively, while **[DiPaCo](https://arxiv.org/abs/2403.10616v1)** offers a novel approach towards robust distributed model training.

- **SVM Kernel Conquest**: Results inform that the **sigmoid SVM kernel** shows better performance on Pythia's input embeddings than other kernels such as rbf, linear, and poly. 

- **N-gram Project "Tokengrams" Gains Traction**: The *Tokengrams* project is now reportedly usable for efficiently computing and storing token n-grams from text corpora, suggesting an efficient resource available to researchers at [GitHub - EleutherAI/tokengrams](https://github.com/EleutherAI/tokengrams).

- **Chess-GPT Gets Analyzed**: A case study on **Chess-GPT** discusses the technique of using language models to predict chess moves with an Elo rating estimation alongside validating computations using linear probes, detailed at [Chess GPT Interventions](https://adamkarvonen.github.io/machine_learning/2024/03/20/chess-gpt-interventions.html).

- **Evaluation Variability Worries AI Engineers**: The **evaluation result inconsistencies** when comparing **Hugging Face transformers** to **Megatron-DeepSpeed evaluations** has drawn attention, with suggestions to verify if implementation details like *bfloat16* numeric handling in *fused kqv multiplications* might be contributing to variability.

- **Minecraft Serves As RL Testing Ground**: A **Minecraft-based environment for Reinforcement Learning**, available on [GitHub - danijar/diamond_env](https://github.com/danijar/diamond_env), alongside discussions on project [Voyager](https://github.com/MineDojo/Voyager/issues/149), underscores the use of games for AI model collaboration research.

- **Multimodal Embedding Spaces Exploration**: Interest in **theoretical works on multimodal embedding spaces** was raised, with the community providing insights on how Stable Diffusion's subculture treats embeddings in line with **IMG2IMG** workflows.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**When Discord Fails, Meet Pushes Through**: Technical difficulties during a GTC event led to the suggestion of defaulting to voice channels for future lectures due to screen sharing issues on Discord stage channels. An unsatisfied member proposed switching to Google Meet in future due to the instability of Discord streams.

**CUDA-tious Profiling**: For engineers delving into CUDA, [a lecture on how to profile CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM) was shared, complete with [accompanying slides](https://docs.google.com/presentation/d/110dnMW94LX1ySWxu9La17AVUxjgSaQDLOotFC3BZZD4/edit?usp=sharing) and a [GitHub code repository](https://github.com/msaroufim/cudamodelecture1). CUDA programming becomes a necessity when seeking performance gains where PyTorch's speed is insufficient.

**Triton Tricky Tidbits**: Discussions around **Triton's performance issues** were prominent, and members were warned that **Triton operations might be phased out** in the future. A new prototype folder in the `torchao` repository was proposed for collaboration on API design for efficient kernel usage, as support for Triton continues.

**Sparsity Meets Decomposition Elegance**: A novel approach to distributed sparse matrix multiplication was introduced in the **Arrow Matrix Decomposition** paper by researchers [Lukas Gianinazzi](https://arxiv.org/abs/2402.19364) and Alexandros Nikolaos Ziogas, with the implementation available on [GitHub](https://github.com/spcl/arrow-matrix).

**Blackwell GPUs Smile for the Camera**: Members discussed the new **Blackwell GPUs**, highlighting a tweet with a humorous take on the GPUs' smiley face pattern. Speculation on the unseen NVIDIA Developer Discord server took place after a GitHub discussion about the **CUTLASS library** was brought up. The community also touched on data type standardization in deep learning, noting the absence of Google in recent standard consortiums and the lack of an **IEEE standard** for new floating point numbers.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Mistral's New 7B Model Steals the Spotlight**: Mistral AI **casually dropped a new model**, the Mistral 7B v0.2 Base, at the [@cerebral_valley hackathon](https://x.com/alexreibman/status/1771608346635751541?s=46). The model details including fine-tuning guidance are available [here](https://x.com/mistralailabs/status/1771670765521281370?s=46), although no magnet links were provided for this release, as noted by [@natolambert](https://twitter.com/MistralAI).

**Shakeup at Stability AI**: CEO Emad Mostaque **resigned** from Stability AI, hinting at his future focus on **#DecentralizedAI**. The community expressed mixed feelings about the impact and direction of his tenure, amidst discussions of internal struggles and the nature of Stability AI's contributions to AI academia.

**Nemo Interoperability Seekers**: Questions arose about converting and wrapping **Nemo checkpoints** for compatibility with Hugging Face, underscoring the technical challenges in machine learning model interoperability.

**AI's Ethical Tightrope**:
- Debates ignited over whether creating a "generalist agent" in Reinforcement Learning is both *practically and fundamentally feasible*, based on discussions found [here](https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw).
- The channel also tackled the FTC's antitrust lawsuits against Apple with Nathan Lambert pointing out the public's misunderstanding of antitrust regulations and backing his views with supporting tweets.

**February's Big AI Chats**: Illuminating interviews with Anthropic's CEO and Mistral's CEO have been drawing attention, such as this ["Fireside Chat"](https://youtu.be/sQpeIuymJZ8) and the discussion on Amodei's AI industry predictions [here](https://www.youtube.com/watch?v=gAaCqj6j5sQ). Additionally, Latent Space's February recap, highlighting key AI developments, can be found [here](https://www.latent.space/p/feb-2024).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **AI Delivers More for Less**: An innovative method was proposed to bypass the 4k output token limit of **GPT-4-Turbo** by initiating a follow-up request upon reaching the length limit, which allows the model to continue generating content seamlessly.
- **Bedrock Meets Python**: A guide has surfaced detailing the use of **Bedrock with Python**, showing practical integration techniques. Interested engineers can dive into the guide [here](https://medium.com/@leonardo.bolanos/leveraging-bedrock-anthropic-haiku-with-python-a-comprehensive-guide-9f5e912982be).
- **Analyze LLM Conversations with SimplyAnalyze.ai**: The launch of **SimplyAnalyze.ai** was announced, which pairs with LangChain to dissect LLM dialogue across business divisions. To join the free developer preview, engineers can visit [SimplyAnalyze's website](https://simplyanalyze.ai/).
- **Harnessing LangChain for Decision-making**: A post detailing the use of **Langchain in Agent Tree Search** was shared to foster more sophisticated decision-making processes with Language Models. Engineers can read more about it [here](https://medium.com/ai-advances/language-agent-tree-search-with-langchain-revolutionizing-decision-making-with-language-models-a46c991397f1).
- **Upgraded Chatbot with Memory and Parsing Skills**: Enhancements to a **local character AI chatbot** have been made, improving CSV and NER parsing, among other features. To check out the upgraded capabilities, the GitHub repository is available [here](https://github.com/ossirytk/llama-cpp-chat-memory).



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**Real Estate Matching Gone Awry**: A discussion unfolded around a problem with **GPT4.Turbo** misinterpreting property size requirements, with one property being suggested at 17,000 square feet despite a request for 2,000 - 4,000 square feet. A simple CSV-based database filter was recommended over a complex LLM, sparking a conversation about common missteps and linking to a resource by **Jason Liu** on the potential over-reliance on embedding search in LLMs.

**Frustrations with Token Limitations**: Participants voiced frustration with **Anthropic's** rate limit of **1M tokens per day**, considering a **200k context window** to be insufficient. The **Bedrock monthly fee model** was discussed as a potential alternative, while a **$500 scale plan** from Anthropic was suggested as offering easier access for extensive use.

**Seeking Superior Explainers**: The community was asked for their top **explainer resources** on advanced LLM topics, with a specific call-out for high-quality, clear content on topics like RHLF, rather than a vast collection of blogs. *Exa.ai* was suggested as a beneficial resource for delving into LLM-related subjects.

**Brief Cry for Coding Quality**: In the **#[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/)** channel, a user lamented the difficulty in writing high-quality code with a succinct and relatable one-liner.

**GPT-3.5-0125 Takes the Lead**: **GPT-3.5-0125** was lauded for its significant performance improvements over previous models, as observed in a user's comparative tests, elevating its status as a particularly advanced iteration within the realm of LLMs.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Call for AI Avengers**: The **Youth Inquiry Network** and **Futracode** are teaming up to develop a machine learning algorithm that recommends optimal research topics from existing databases. They're recruiting **web developers, data analysts, and AI & ML experts** to champion this cause.
  
- **Contribute Your Skills for Glory**: Volunteers will not only advance their careers with a shiny new portfolio piece but also walk away with a **certificate and two professional recommendation letters**. Those who help will also get to keep the developed **ML algorithm code** for personal or commercial use.

- **Flex Time for World Savers**: They assure a **flexible commitment** for this groundbreaking project—perfect for superheroes with a packed schedule. Recruits can bypass the bureaucratic labyrinth by dropping a simple "interested" to get started on their mission.

- **Mystery Educational Reform Doc Drops**: An unspecified member shares a [Google Docs link](https://docs.google.com/document/d/1f-CHZudw3ZOGFIk-Kov3QHkPjjR-Sh4mMmxcExgnWUk/edit?usp=sharing) discussing **Post-AGI Educational Reforms**, possibly hinting at a future-focused AI education paradigm.

- **Moment of Meta Moderation**: In an ironic twist, a **moderator experiences a self-epiphany** of their own status in a call for moderation, reminding us that even bots can forget their protocols.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **LLM Versus Ollama Showdown**: Members clarified that **`llm` interfaces with models**, such as **Mistral**, by setting up API endpoints, which are then executed by `ollama`. `ollama` allows local model execution, making the models accessible via local HTTP API endpoints.
  
- **Techie Commits with AI Assistance**: The tool **[AICommits (GitHub - Nutlope/aicommits)](https://github.com/Nutlope/aicommits)**, designed to help write git commit messages with AI, gained appreciation for its utility, with requests for additional features such as emoji standards for commits.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **AI Cooking Up Stunts**: An AI has crafted a unique cookbook based on YouTuber **Mr. Beast's** daring adventures, which sparked interest in the group. The inventive application was showcased in a [YouTube video](https://www.youtube.com/watch?v=Nc5Yk0XXgP8), mixing culinary arts and machine learning for whimsical results.
- **In Search of German Tech Savvy**: A community member is on the lookout for **German-language resources** on deep learning and AI, indicating a desire to dive into technical content in their native tongue. The request painted a picture of a global, multilingual interest in the AI community.



---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1220621626790514709)** (1195 messages🔥🔥🔥): 

- **Stable Diffusion Inquiry and Assistance**: Members discussed various aspects and uses of Stable Diffusion models, including performance, sampler settings, and ControlNet models. Users also exchanged guidance on handling errors and setting up the AI on different systems, especially with AMD GPUs.

- **Exploring SD3 and Alternatives**: Conversation topics included the anticipated release window for SD3 and potential improvements, along with comparisons to other offerings like SDXL and AI-generated video potential.

- **Feedback on Online AI Services**: The chat touched on the limitations and frustrations with online AI image generation services, such as those on Civitai and Suno, specifically pointing out issues with content restrictions and preferences on the type of content displayed.

- **Debate on AI Ethics and Regulation**: Members debated the need for regulations on AI technology use and the importance of open-source models versus proprietary ones. Concerns were raised about regulations potentially stifling innovation and accessibility.

- **Technical Troubleshooting and Learning**: New members seeking help with technical issues regarding model installation and use were directed to support channels and experts within the community. More experienced members aimed to guide and provide resources while advising newcomers on the learning curve associated with AI image generation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://seaart.ai>">no title found</a>: no description found</li><li><a href="https://artificialguy.com">ArtificialGuyBr</a>: no description found</li><li><a href="https://sakana.ai/evolutionary-model-merge/">no title found</a>: no description found</li><li><a href="https://imgur.com/H4PmCXo">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://comfyanonymous.github.io/ComfyUI_examples/stable_cascade/">Stable Cascade Examples</a>: Examples of ComfyUI workflows</li><li><a href="https://xkcd.com/435/">Purity</a>: no description found</li><li><a href="https://siliconangle.com/2024/03/24/emad-mostaque-resigns-ceo-troubled-generative-ai-startup-stability-ai/">Emad Mostaque resigns as CEO of troubled generative AI startup Stability AI - SiliconANGLE</a>: Emad Mostaque resigns as CEO of troubled generative AI startup Stability AI - SiliconANGLE</li><li><a href="https://huggingface.co/thibaud">thibaud (Thibaud Zamora)</a>: no description found</li><li><a href="https://tenor.com/view/dune-oil-gif-2770573093912411630">Dune Oil GIF - Dune Oil - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://siliconangle.com/2024/03/18/nvidias-newest-cloud-service-promises-accelerate-quantum-computing-simulations-beef-post-quantum-security/">Nvidia&#039;s newest cloud service promises to accelerate quantum computing simulations - SiliconANGLE</a>: Nvidia&#039;s newest cloud service promises to accelerate quantum computing simulations - SiliconANGLE</li><li><a href="https://civitai.com/models/359999">CinematicRedmond - Cinematic Model for SD XL - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: Cinematic.Redmond is here! I&#x27;m grateful for the GPU time from Redmond.AI that allowed me to make this model! This is a Cinematic model fine-tuned o...</li><li><a href="https://huggingface.co/spaces/artificialguybr/artificialguybr-demo-lora">Artificialguybr Demo Lora - a Hugging Face Space by artificialguybr</a>: no description found</li><li><a href="https://app.suno.ai/song/89a68ee1-899e-44c7-a8d8-a1c011376f3a">Neon City Lights | Suno</a>: japanese vocals chill jazz j-pop downtempo song. Listen and make your own with Suno.</li><li><a href="https://x.com/chrlaf/status/1772226646365397311?s=20">Tweet from Christian Laforte (@chrlaf)</a>: @thibaudz Thanks Thibaud, as I wrote elsewhere, the plan hasn&#39;t changed, we are still hard at work improving the model towards open release. Including source code and weights.</li><li><a href="https://tenor.com/view/ok-then-um-well-ok-then-wtf-gif-23665207">Ok Then Um GIF - Ok Then Um Well Ok Then - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://civitai.com/models/310571/boring-reality">Boring Reality - BoringReality_primaryV4.0 | Stable Diffusion LoRA | Civitai</a>: NOTE: Please read below for working with these loras. They are unlikely to give good results when used individually and as is. This model is actual...</li><li><a href="https://civitai.com/models/38784/controlnet-11-models">ControlNet 1.1 Models - Tile (e) | Stable Diffusion Controlnet | Civitai</a>: STOP! THESE MODELS ARE NOT FOR PROMPTING/IMAGE GENERATION These are the new ControlNet 1.1 models required for the ControlNet extension , converted...</li><li><a href="https://civitai.com/models/359999/cinematicredmond-cinematic-model-for-sd-xl">CinematicRedmond - Cinematic Model for SD XL - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: Cinematic.Redmond is here! I&#x27;m grateful for the GPU time from Redmond.AI that allowed me to make this model! This is a Cinematic model fine-tuned o...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bnjm3i/stable_diffusion_3/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://stable-diffusion-art.com/glossary/">Stable Diffusion Glossary - Stable Diffusion Art</a>: Confused about a term in Stable Diffusion? You are not alone, and we are here to help. This page has all the key terms you need to know in Stable Diffusion.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bmpqeh/stabilityai_is_alive_and_will_live_there_were/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://civitai.com/models/261999/drone-shot-above-xl-lora">Drone Shot &quot;Above&quot; XL LoRA - v1.0 | Stable Diffusion LoRA | Civitai</a>: Use &quot; above &quot; in prompt. Works best for large scenes and not individual objects or characters.</li><li><a href="https://github.com/virattt/financial-agent">GitHub - virattt/financial-agent: A financial agent, built entirely with LangChain!</a>: A financial agent, built entirely with LangChain! Contribute to virattt/financial-agent development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=1sN6U5dV1Os">AI Dance Animation - [ NEXT GEN ] - Stable Diffusion | ComfyUI</a>: This AI animation was done using AnimateDiff and ControlNet nodes without any girl LORAs. Since the last AI dance video it&#39;s a major improvement in consisten...</li><li><a href="https://arstechnica.com/information-technology/2023/11/unauthorized-david-attenborough-ai-clone-narrates-developers-life-goes-viral/">Unauthorized “David Attenborough” AI clone narrates developer’s life, goes viral</a>: &#34;We observe the sophisticated Homo sapiens engaging in the ritual of hydration.&#34;</li><li><a href="https://huggingface.co/artificialguybr">artificialguybr (ArtificialGuy/JV.K)</a>: no description found</li><li><a href="https://civitai.com/models/136070?modelVersionId=267507">ControlNetXL (CNXL) - bdsqlsz-depth | Stable Diffusion Checkpoint | Civitai</a>: bdsqlsz : canny | depth | lineart-anime | mlsd-v2 | normal | openpose | recolor | segment | segment-v2 | sketch | softedge | t2i-color-shuffle | ti...</li><li><a href="https://www.lesswrong.com/tag/rokos-basilisk">Roko&#x27;s Basilisk - LessWrong</a>: Roko’s basilisk is a thought experiment proposed in 2010 by the user Roko on the Less Wrong community blog. Roko used ideas in decision theory to argue that a sufficiently powerful AI agent would have...</li><li><a href="https://www.youtube.com/watch?v=w3vXaK3JC8E">Who is going to tell her…</a>: Support the channel by grabbing a t-shirt: http://www.clownplanetshirts.comDon’t forget to subscribe. Hit the bell to stay updated on the latest videos.Watch...</li><li><a href="https://www.youtube.com/watch?v=1CIpzeNxIhU">How AI Image Generators Work (Stable Diffusion / Dall-E) - Computerphile</a>: AI image generators are massive, but how are they creating such interesting images? Dr Mike Pound explains what&#39;s going on. Thumbnail image partly created by...</li><li><a href="https://github.com/stitionai/devika">GitHub - stitionai/devika: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective. Devika aims to be a competitive open-source alternative to Devin by Cognition AI.</a>: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/15c3rf6/sdxl_resolution_cheat_sheet/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=5mA_tzJije0">Revenge of the nerds - John Goodman speech</a>: Inspiring stuff!</li><li><a href="https://github.com/LykosAI/StabilityMatrix/blob/main/README.md">StabilityMatrix/README.md at main · LykosAI/StabilityMatrix</a>: Multi-Platform Package Manager for Stable Diffusion - LykosAI/StabilityMatrix</li><li><a href="https://replicate.com/artificialguybr/cinematic.redmond">artificialguybr/cinematic.redmond – Run with an API on Replicate</a>: no description found</li><li><a href="https://youtu.be/RbId7zb8mqE?si=fVBYGBVUpvbYn3H3">Eyaura: Give Me A Soul. Album: T.B.D.</a>: G STRING - STEAM:https://store.steampowered.com/app/1224600/G_String/G STRING DISCORD - OFFICIAL:https://discord.gg/fUuDyx7uYeG STRING DISCORD - MISC:https:/...</li><li><a href="https://forms.gle/DqMih6Z9wCmTRvZN8">Name Archetypes</a>: So, I made these faces in Midjourney by using the prompt &quot;a photo of (name) --style raw.&quot; And I couldn&#39;t help but notice that whenever you meet a new person, they sorta have a name assoc...</li><li><a href="https://civitai.com/models/339604/how-to-generate-multiple-different-characters-mix-characters-andor-minimize-color-contamination-or-regional-prompt-adetailer-and-inpaint-or-my-workflow">How to Generate Multiple Different Characters, Mix Characters, and/or Minimize Color Contamination | Regional Prompt, Adetailer, and Inpaint | My Workflow - 2. Adetailer | Stable Diffusion Workflows | Civitai</a>: How to Generate Multiple Different Characters, Mix Characters, and/or Minimize Color Contamination | Regional Prompt, Adetailer, and Inpaint | My W...</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/169#discussioncomment-8428689">Forge Is Not Using ComfyUI as A Backend · lllyasviel/stable-diffusion-webui-forge · Discussion #169</a>: Recently some people begin to spread misinformation about Forge using ComfyUI as a backend. This is false, harmful to the community, and harmful to the efforts of our engineering team. The backend ...
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1220603138852257793)** (1009 messages🔥🔥🔥): 

- **Exploring Metrics in SFTTrainer**: A user seeks advice on using a generation-based metric for validation with `SFTTrainer`, referencing a workaround in a [GitHub issue](https://github.com/huggingface/trl/issues/862#issuecomment-1896074498). They are unclear about the preds received in the `compute_metrics` function and how `SFTTrainer` computes loss when fine-tuning with LoRA adapters.

- **Chatbot Model Inference Hardware Requirements**: A user asks how to determine hardware requirements for running LLM models like [Nous-Capybara-34B-GGUF](https://huggingface.co/TheBloke/Nous-Capybara-34B-GGUF), and another user suggests referring to another HH model's [discussion](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/discussions/2) for estimates, clarifying that model requirements can vary based on quantization and prompt.

- **Model Differences & Quantization**: Inquiry made about the differences between two versions of *Mistral* models leads to an explanation that 4-bit models like [here](https://huggingface.co/unsloth/mistral-7b-v0.2-bnb-4bit) are faster to download but suffer a slight drop in accuracy.

- **Mistral and Their Marketing Strategy**: Discussion unfolds around Mistral's model release practices which are deemed unusual by a member due to not uploading their base models on Hugging Face and the unconventional leaking incident on 4chan.

- **Debate on Computer Science Education**: A heated discussion takes place on the importance of a Computer Science degree in light of LLMs now capable of writing code. The conversation veers into various programming languages, memory safety, and the value of degrees from different universities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://inflection.ai/inflection-2-5">Inflection-2.5: meet the world&#x27;s best personal AI</a>: We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.</li><li><a href="https://gpt4all.io/index.html">GPT4All</a>: Free, local and privacy-aware chatbots</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/152334H/miqu-1-70b-sf">152334H/miqu-1-70b-sf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://tenor.com/view/funny-very-sloth-slow-gif-15401812">Funny Very GIF - Funny Very Sloth - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/sloth-slow-stamp-gif-8535595">Sloth Slow GIF - Sloth Slow Stamp - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/sloth-smile-slow-smooth-hd-neuron-activation-gif-24950071">Sloth Smile Slow GIF - Sloth Smile Slow Smooth - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2">mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/mistral-7b-v0.2-bnb-4bit">unsloth/mistral-7b-v0.2-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/mistral-7b-v0.2">unsloth/mistral-7b-v0.2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/discussions/2">TheBloke/CodeLlama-34B-Instruct-GGUF · [AUTOMATED] Model Memory Requirements</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/Nous-Capybara-34B-GGUF">TheBloke/Nous-Capybara-34B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://docs.gpt4all.io/gpt4all_python.html">Generation - GPT4All Documentation</a>: no description found</li><li><a href="https://github.com/InflectionAI/Inflection-Benchmarks">GitHub - InflectionAI/Inflection-Benchmarks: Public Inflection Benchmarks</a>: Public Inflection Benchmarks. Contribute to InflectionAI/Inflection-Benchmarks development by creating an account on GitHub.</li><li><a href="https://www.quantamagazine.org/how-quickly-do-large-language-models-learn-unexpected-skills-20240213/">How Quickly Do Large Language Models Learn Unexpected Skills? | Quanta Magazine</a>: A new study suggests that so&#x2d;called emergent abilities actually develop gradually and predictably, depending on how you measure them.</li><li><a href="https://www.infoworld.com/article/3713203/white-house-urges-developers-to-dump-c-and-c.html">White House urges developers to dump C and C++</a>: Biden administration calls for developers to embrace memory-safe programing languages and move away from those that cause buffer overflows and other memory access vulnerabilities.</li><li><a href="https://github.com/huggingface/trl/issues/862#issuecomment-1896074498">Compute metrics for generation tasks in SFTTrainer · Issue #862 · huggingface/trl</a>: Hi, I want to include a custom generation based compute_metrics e.g., BLEU, to the SFTTrainer. However, I have difficulties because: The input, eval_preds, into compute_metrics contains a .predicti...</li><li><a href="https://huggingface.co/alpindale/Mistral-7B-v0.2-hf">alpindale/Mistral-7B-v0.2-hf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/GAIR/lima">GAIR/lima · Datasets at Hugging Face</a>: no description found</li><li><a href="https://lightning.ai/pages/community/tutorial/accelerating-large-language-models-with-mixed-precision-techniques/">Accelerating Large Language Models with Mixed-Precision Techniques - Lightning AI</a>: Training and using large language models (LLMs) is expensive due to their large compute requirements and memory footprints. This article will explore how leveraging lower-precision formats can enhance...</li><li><a href="https://huggingface.co/ISTA-DASLab">ISTA-DASLab ( IST Austria Distributed Algorithms and Systems Lab)</a>: no description found</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm</li><li><a href="https://github.com/PygmalionAI/aphrodite-engine/tree/main/tests/benchmarks">aphrodite-engine/tests/benchmarks at main · PygmalionAI/aphrodite-engine</a>: PygmalionAI&#39;s large-scale inference engine. Contribute to PygmalionAI/aphrodite-engine development by creating an account on GitHub.</li><li><a href="https://github.com/PygmalionAI/aphrodite-engine">GitHub - PygmalionAI/aphrodite-engine: PygmalionAI&#39;s large-scale inference engine</a>: PygmalionAI&#39;s large-scale inference engine. Contribute to PygmalionAI/aphrodite-engine development by creating an account on GitHub.</li><li><a href="https://ev01.sx/">Watch movies online and Free tv shows streaming - ev01.net</a>: Fast and Free streaming of over 250000 movies and tv shows in our database. No registration, no payment, 100% Free full hd streaming</li><li><a href="https://huggingface.co/argilla">argilla (Argilla)</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/master">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1220627573776584816)** (58 messages🔥🔥): 

- **Kernel Conundrum on Local Machine**: A discussion around an issue requiring kernel restarts on user's local machine, hints it might be memory-related. An error message about 32-bit and needing to restart to get rid of it was shared, with some discussion about possibly being Out Of Memory (OOM), but user confirms the machine works fine after kernel restarts.

- **Warming Up to Fiber Optics and Big Models**: One user excitedly reports upgrading to fiber-optics and a new 2TB WD black edition to support larger models. There is enthusiasm for the current performance and potential future upgrades to hardware.

- **ORPO Generates Buzz in AI Community**: There's excitement around ORPO (**Off-policy Reinforcement learning with Pretrained Overparametrized Models**), as users discuss its integration and boost in performance for models. The link to the original paper on [arXiv](https://arxiv.org/pdf/2403.07691.pdf) was provided.

- **Unsloth Keeps Up with TRL**: In relation to the ORPO discussion, users confirmed that Unsloth AI should support it if it's supported by TRL (**Transformer Reinforcement Learning**). Optimizations and patching from Unsloth to TRL were mentioned, along with encouragement to share if there are any issues with the new integrations.

- **New Toolkit for Transformer Models**: An interesting toolkit for transformers called transformer-heads was linked. It's designed for attaching, training, saving, and loading new heads for transformer models, available on [GitHub](https://github.com/center-for-humans-and-machines/transformer-heads).

**Link mentioned**: <a href="https://github.com/center-for-humans-and-machines/transformer-heads">GitHub - center-for-humans-and-machines/transformer-heads: Toolkit for attaching, training, saving and loading of new heads for transformer models</a>: Toolkit for attaching, training, saving and loading of new heads for transformer models - center-for-humans-and-machines/transformer-heads

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1220638996661538886)** (317 messages🔥🔥): 

- **Understanding Unsloth's Fast Dequantizing**: Unsloth AI's 'fast_dequantize' in `fast_lora.py` is noted to be optimized for speed with reduced memory copies compared to `bitsandbytes`.
- **Troubleshooting and Updating Mistral with Unsloth**: A member was advised to upgrade Unsloth due to issues with Gemma GGUF, with a command provided for the upgrade. It was noted that problems existed not only with GGUF but also with merging.
- **Resolving Inference Issue with Looping Tokens**: Discussion on a reported issue where models converted to gguf, particularly using Mistral, started repeating `<s>` in a loop during responses. Unsloth's maintainer suggests checking the `tokenizer.eos_token`.
- **Combining Multiple Datasets for Fine-Tuning**: It's suggested that multiple datasets can be concatenated into one text string, processed, and then appended together for training. Enhanced instructions and responses from different datasets can potentially be merged for this purpose.
- **Needing Clarity on Fine-Tuning Parameters**: Queries were made about controlling epochs with `max_steps` during fine-tuning, for which setting `num_train_epochs` instead was recommended. Additionally, it's mentioned higher memory consumption may result from increasing `max_seq_length` due to padding.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://goddard.blog/posts/clown-moe/">Mixture of Experts for Clowns (at a Circus)</a>: no description found</li><li><a href="https://huggingface.co/HirCoir/Claud-mistral-7b-bnb-4bit-GGUF/tree/main">HirCoir/Claud-mistral-7b-bnb-4bit-GGUF at main</a>: no description found</li><li><a href="https://huggingface.co/HirCoir/Claud-openbuddy-mistral-7b-v19.1-4k">HirCoir/Claud-openbuddy-mistral-7b-v19.1-4k · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2#instruction-format">mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2#scrollTo=LjY75GoYUCB8&line=8&uniqifier=1.">Google Colaboratory</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py#L488">bitsandbytes/bitsandbytes/autograd/_functions.py at main · TimDettmers/bitsandbytes</a>: Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm</li><li><a href="https://youtu.be/_GkHZQYFOGM?si=taLl7f-TNdJta_W8">Fine tuning LLMs for Memorization</a>: ➡️ ADVANCED-fine-tuning Repo (incl. Memorization Scripts): https://trelis.com/advanced-fine-tuning-scripts/➡️ One-click Fine-tuning &amp; Inference Templates: ht...</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1220821099466592318)** (33 messages🔥): 

- **Hints of Troubles with Unsloth Integration**: Users express issues with using *Ollama templates*, particularly with different quant models such as q4, leading to poor results.
- **Diagnostics on GPT4All**: One user is running tests on **GPT4All** to resolve issues and is advised not to escape backticks and to try different *quant* sizes.
- **Q8 Model Version Shows Promise**: After a bit of back-and-forth, a user confirms that the Q8 model on **Huggingface** seems to be functioning correctly.
- **Sappha-2b-v3 Makes Waves**: A new model, [sappha-2b-v3](https://huggingface.co/Fizzarolli/sappha-2b-v3), which is fine-tuned with Unsloth on Gemma-2b, outperforms current models on several benchmarks, prompting discussions on its capability.
- **Interest Peaks for New Models**: Users show excitement for the newly released models and share links to their best-performing models, such as [MasherAI-v6-7B](https://huggingface.co/mahiatlinux/MasherAI-v6-7B), while seeking information on the fine-tuning process used.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mahiatlinux/MasherAI-v6-7B">mahiatlinux/MasherAI-v6-7B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Fizzarolli/sappha-2b-v3">Fizzarolli/sappha-2b-v3 · Hugging Face</a>: no description found</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/modelfile.md">ollama/docs/modelfile.md at main · ollama/ollama</a>: Get up and running with Llama 2, Mistral, Gemma, and other large language models. - ollama/ollama
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1221127411723403406)** (29 messages🔥): 

- **PyTorch Full Finetuning Without Breaking the (Memory) Bank**: PyTorch users take note: a new [pull request](https://github.com/pytorch/torchtune/pull/527) allows full finetuning to fit into less than 16GB of RAM, making it more accessible for those with consumer-grade GPUs.
- **LoftQ Hits the Scene**: In artificial intelligence optimization news, LoftQ has been included in the [Hugging Face PEFT release v0.10.0](https://github.com/huggingface/peft/releases/tag/v0.10.0), which brings enhanced fine-tuning capabilities to larger models.
- **Multi-Turn Training Challenge for ORPO**: There's a discussion on multi-turn dialogue training for ORPO, with suggestions to resolve the current limitations of using the (prompt:"", chosen:"", rejected:"") format by introducing a more efficient method that handles multiple turns effectively.
- **ORPO Needs Better Multi-Turn Training**: The community expresses concerns that the current ORPO method doesn't seem to cater well to multi-turn dialogue training, which is essential for ORPO to be a feasible replacement for SFT, highlighting the importance of a standardized and optimized format for dialogue training.
- **Successful ORPO Trials with Mistral**: One member boasts impressive results when applying ORPO TRL implementation on the Mistral-7B-v0.2 model, using the *argilla/ultrafeedback-binarized-preferences-cleaned* dataset, suggesting that further tuning might yield even better outcomes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/peft/releases/tag/v0.10.0">Release v0.10.0: Fine-tune larger QLoRA models with DeepSpeed and FSDP, layer replication, enhance DoRA · huggingface/peft</a>: Highlights  Support for QLoRA with DeepSpeed ZeRO3 and FSDP We added a couple of changes to allow QLoRA to work with DeepSpeed ZeRO3 and Fully Sharded Data Parallel (FSDP). For instance, this allow...</li><li><a href="https://github.com/pytorch/torchtune/pull/527">Full finetune &lt; 16GB by rohan-varma · Pull Request #527 · pytorch/torchtune</a>: Context  We&#39;d like to enable a variant of full finetune that trains in &lt; 16GB of RAM for users with consumer grade GPUs that have limited GPU RAM. This PR enables the full finetune to fit into ...
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1220636126981652500)** (892 messages🔥🔥🔥): 

- **Pro Users Debate Perplexity's Image Generation**: Users discussed the ability to generate images using Perplexity as a Pro feature, noting that it requires using Writing mode and switching off the Pro toggle on the web version.
- **Claude Opus & GPT-4 Turbo Differences**: Conversations centered around the functionality of Claude 3 Opus and GPT-4 Turbo models, comparing their abilities for academic research and writing code, and the distinction of GPT-4 Turbo being able to compile Python files which Perplexity does not currently support.
- **Exploring Stability AI and Local Models**: Talk of Stability AI's models like SDXL and local installations was a focus, with users sharing tips and experiences about running these powerful image generation tools on personal hardware, despite the high costs involved.
- **Investigating Perplexity Bugs and Confusions**: Users expressed confusion about certain Perplexity features, such as repeated unrelated prompts appearing during sessions, how to disable unnecessary search triggers when using certain AI models, and issues encountered on the iOS app.
- **Perplexity Features and Updates Discussion**: Users debated the potential of features like Claude 3 Opus's API capabilities, Op1 synthesizer's aesthetics and new models like Rabbit R1, and discussed the possibility of integrating Perplexity with iOS Spotlight search.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imagine.meta.com/">Imagine with Meta AI</a>: Use Imagine with Meta AI to quickly create high-resolution, AI-generated images for free. Just describe an image and Meta AI will generate it with technology from Emu, our image foundation model.</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found</li><li><a href="https://stability.ai/stable-image">Stability AI Image Models &mdash; Stability AI</a>: Experience unparalleled image generation capabilities with SDXL Turbo and Stable Diffusion XL. Our models use shorter prompts and generate descriptive images with enhanced composition and realistic ae...</li><li><a href="https://civitai.com/models">Civitai | Share your models</a>: no description found</li><li><a href="https://tenor.com/view/swedish-house-mafia-one-op1-synthesizer-edm-gif-19823728">Swedish House Mafia One GIF - Swedish House Mafia One Op1 - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/iota-crypto-cryptocurrency-green-candles-pepe-gif-22089159">Iota Crypto GIF - Iota Crypto Cryptocurrency - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.vellum.ai/blog/prompt-engineering-tips-for-claude">Claude 2.1 prompt engineering guide</a>: Learn how to prompt Claude with these 11 prompt engineering tips.</li><li><a href="https://blogs.bing.com/search-quality-insights/december-2023/Introducing-Deep-Search">Introducing deep search | Search Quality Insights</a>: no description found</li><li><a href="https://www.xda-developers.com/copilot-gpt-4-turbo-model-free/">You can now use Copilot's GPT-4 Turbo model for free</a>: Microsoft has just made the advanced GPT model available for everyone, with no catches or tricks.</li><li><a href="https://stability.ai/news/stabilityai-announcement">Stability AI Announcement &mdash; Stability AI</a>: Earlier today, Emad Mostaque resigned from his role as CEO of Stability AI and from his position on the Board of Directors of the company to pursue decentralized AI.  The Board of Directors has appoin...</li><li><a href="https://www.youtube.com/watch?v=G7RgN9ijwE4">Have you ever had a dream like this?</a>: We all have at one point.</li><li><a href="https://tenor.com/view/kys-wojak-mushroom-kill-urself-die-gif-22188194">Kys Wojak GIF - Kys Wojak Mushroom - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.perplexity.ai/docs/perplexitybot">PerplexityBot</a>: no description found</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1bm305k/what_the_hell_claud_3_opus_is_a_straight/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/view/golden-eggs-willy-wonka-and-the-chocolate-factory-clean-the-eggs-get-the-eggs-ready-chocolate-golden-eggs-gif-21442701">Golden Eggs Willy Wonka And The Chocolate Factory GIF - Golden Eggs Willy Wonka And The Chocolate Factory Clean The Eggs - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://apps.apple.com/app/id1484498200">‎Orion Browser by Kagi</a>: ‎Orion is a fast, free, web browser for iPhone and iPad with no ads, no telemetry. Check also Orion for your Mac desktop.  Orion has been engineered from ground up as a truly privacy-respecting browse...</li><li><a href="https://www.cnbc.com/2024/03/21/doj-sues-apple-over-iphone-monopoly.html">DOJ sues Apple over iPhone monopoly in landmark antitrust case</a>: Apple and its iPhone and App Store business have been eyed by the Department of Justice, which previously filed antitrust suits against Google.
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1220614816494911599)** (38 messages🔥): 

- **Legal Battle Ahead**: The United States is engaged in a lawsuit, details of which can be explored at [United States Sues](https://www.perplexity.ai/search/United-States-sues-EDYwvszoRZO.lccIR60pLQ).
- **Email Authentication Scrutiny**: Users interested in email security protocols, specifically DMARC, can find information at [DMARC Details](https://www.perplexity.ai/search/DMARC-AvS.J6JyS0C6EiEK13CRpw).
- **Decoding Market Tools**: TradingView, a tool for traders and investors, is discussed and can be examined at [TradingView Insights](https://www.perplexity.ai/search/tradingview-ZK6dm4jcSyateKhccuwpXQ).
- **Generating Curiosity Around Perplexity**: The potential of Perplexity AI replacing other tools is being questioned, and insights can be discovered at [Should Perplexity Replace?](https://www.perplexity.ai/search/should-perplexity-replace-qCBVfG3vTmO1oGEr9dTzfg).
- **The Concept of Love Explored**: An inquiry into the nature of love has been raised, with a desire to understand more at [What is Love?](https://www.perplexity.ai/search/What-is-love-bTVpK3ZjTqaMh89tPeWzvg).
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1220604143857832019)** (24 messages🔥): 

- **Token Limit Troubles**: A member encountered a **BadRequestError** due to a prompt and token generation request exceeding the **16,384 token limit**. They were instructed to reduce the `max_tokens` value to stay under the limit and to consider shortening their prompts.
- **Resume Analyzer Development**:
The member shared they are working on a **resume analyzer/builder project** as a way to practice with AI, indicating they are new to the field.
- **Seeking Token Count Clarity**: In response to a question about limiting user prompts by token count, it was explained that the **number of tokens** is tied to the length of the message sent. They were referred to Perplexity AI's documentation for an accurate token count.
- **Tokenization Tool Tip**: Another member recommended using OpenAI's tokenizer tool as a general gauge for token count, but noted that different AI models may tokenize differently. For precision, they advised checking token usage directly through the Perplexity API.
- **In Search of an Autogpt-like Service**: A member inquired about an **autogpt-like service that supports Perplexity API keys** for automating iterative tasks. There were no responses provided to this query within the summarized message history.

**Link mentioned**: <a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: no description found

  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1220624337363800145)** (533 messages🔥🔥🔥): 

- **High CPU Usage Queries**: Users noticed high CPU usage when running models on LM Studio version 0.2.17; some resolved issues by restarting LM Studio and resetting settings to default. Advice for resolving these issues includes reviewing log files for errors.
- **GPU Compatibility Concerns**: There were inquiries about the best GPU cards for LM Studio, with discussions revealing preference for Nvidia graphics cards like the 3090 TI for better compatibility and performance. Users also discussed various issues regarding GPU offload and the impact of different model file sizes on performance.
- **Local Server Accessibility**: Users encountered errors using LM Studio's local server feature, with successful resolutions involving reinstallation and proper configurations; users are prompted to post error reports in a specific Discord channel (#1139405564586229810) for assistance.
- **Model Format Support**: Discussions indicated that LM Studio supports GGUF model format, and users explored how to convert Hugging Face models to GGUF format using command-line methods and the importance of sharing converted models back on Hugging Face.
- **Linux and MacOS Support**: Users inquired about using LM Studio on Linux and MacOS platforms. There is no immediate plan for a docker image or an Intel Mac version of LM Studio, but users are encouraged to vote for this feature on the feature request channel (#1128339362015346749).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://host.docker.internal:8080.">no title found</a>: no description found</li><li><a href="http://localhost:YOUR_PORT`">no title found</a>: no description found</li><li><a href="https://memgpt.readme.io/docs/lmstudio">LM Studio</a>: no description found</li><li><a href="https://huggingface.co/roborovski/superprompt-v1">roborovski/superprompt-v1 · Hugging Face</a>: no description found</li><li><a href="https://python.langchain.com/docs/expression_language/get_started">Get started | 🦜️🔗 Langchain</a>: LCEL makes it easy to build complex chains from basic components, and</li><li><a href="https://huggingface.co/bartowski/c4ai-command-r-v01-GGUF">bartowski/c4ai-command-r-v01-GGUF · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/ban-keyboard-gif-23575674">Ban Keyboard GIF - Ban Keyboard - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/gptq-integration">Making LLMs lighter with AutoGPTQ and transformers</a>: no description found</li><li><a href="https://github.com/czkoko/SD-AI-Prompt">GitHub - czkoko/SD-AI-Prompt: A shortcut instruction based on LLama 2 to expand the stable diffusion prompt, Power by llama.cpp.</a>: A shortcut instruction based on LLama 2 to expand the stable diffusion prompt, Power by llama.cpp. - czkoko/SD-AI-Prompt</li><li><a href="https://www.youtube.com/watch?v=4fdZwKg9IbU">Run ANY Open-Source LLM Locally (No-Code LMStudio Tutorial)</a>: LMStudio tutorial and walkthrough of their new features: multi-model support (parallel and serialized) and JSON outputs. Join My Newsletter for Regular AI Up...</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">The unofficial LMStudio FAQ!</a>: Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...</li><li><a href="https://www.youtube.com/watch?v=UbQgXeY_zi4&list=RDUbQgXeY_zi4&start_">Caravan Palace - Lone Digger (Official MV)</a>: 📀 Preorder our new album: https://caravanpalace.ffm.to/gmclub🎫 Come see us live: http://www.caravanpalace.com/tour 🔔 Subscribe to our channel and click th...</li><li><a href="https://github.com/stitionai/devika">GitHub - stitionai/devika: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective. Devika aims to be a competitive open-source alternative to Devin by Cognition AI.</a>: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective...</li><li><a href="https://www.youtube.com/watch?v=UbQgXeY_zi4&list=RDUbQgXeY_zi4&start_radio=1&ab_channel=CaravanPalace">Caravan Palace - Lone Digger (Official MV)</a>: 📀 Preorder our new album: https://caravanpalace.ffm.to/gmclub🎫 Come see us live: http://www.caravanpalace.com/tour 🔔 Subscribe to our channel and click th...</li><li><a href="https://github.com/caddyserver/caddy">GitHub - caddyserver/caddy: Fast and extensible multi-platform HTTP/1-2-3 web server with automatic HTTPS</a>: Fast and extensible multi-platform HTTP/1-2-3 web server with automatic HTTPS - caddyserver/caddy</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/2948">Tutorial: How to convert HuggingFace model to GGUF format · ggerganov/llama.cpp · Discussion #2948</a>: Source: https://www.substratus.ai/blog/converting-hf-model-gguf-model/ I published this on our blog but though others here might benefit as well, so sharing the raw blog here on Github too. Hope it...</li><li><a href="https://wiki.yacy.net/index.php/Dev:API">Dev:API – YaCyWiki</a>: no description found</li><li><a href="https://searchlab.eu/en/">Web Harvesting for Data Mining - YaCy Searchlab</a>: no description found</li><li><a href="https://yacy.net/">Home - YaCy</a>: no description found</li><li><a href="https://huggingface.co/nisten/mistral-instruct0.2-imatrix4bit.gguf">nisten/mistral-instruct0.2-imatrix4bit.gguf · Hugging Face</a>: no description found</li><li><a href="https://x.com/shao__meng/status/1771718504535978173?s=20">Tweet from meng shao (@shao__meng)</a>: They just change the readme of HuggingFace space: Mistral-7B-Instruct-v0.2 based on Mistral-7B-Instruct-v0.1 =&gt; Mistral-7B-v0.2  https://twitter.com/shao__meng/status/1771680453210370157  ↘️ Quotin...</li><li><a href="https://youtu.be/6fFUfyT-EyA?si=Wt5IMTvNfrdHLGyV">George Hotz | Exploring | finding exploits in AMD&#39;s GPU firmware | Giving up on AMD for the tinybox</a>: Date of the stream 21 Mar 2024.from $1050 buy https://comma.ai/shop/comma-3x &amp; best ADAS system in the world https://openpilot.comma.aiLive-stream chat added...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1220749688379150487)** (71 messages🔥🔥): 

- **Q*-AGI Skepticism Strikes**: A discussion about *Q*-AGI videos led to expressions of fatigue and refusal to consume more content on the topic, with a [humorous meme](https://tenor.com/view/dont-say-that-ever-again-diane-lockhart-the-good-fight-dont-say-that-never-say-that-again-gif-18052604895623551134) shared to underline the sentiment.
- **AI in Architecture Needs Human Verification**: Dialogue regarding the use of AI in architectural engineering highlighted skepticism about trusting AI models without human oversight, citing the need for human certification due to legal and safety concerns.
- **Fine-Tuning Models for Specific Tasks**: Members discussed the effectiveness of fine-tuning smaller language models for specific tasks. One member shared their creation of a program that lets users train their own models based on ChatGPT 2, complete with an instruction manual generated by Claude.
- **Understanding Model Size vs. Quantization**: There was clarification provided about the difference between *#b* (size of the model based on parameters) and *q#* (level of quantization) when deciding which model version to run, such as "llama 7b - q8" versus "llama 13b - q5".
- **Choosing the Right Model for Context Length**: A user inquired about models with 32K context length for RAG-adjacent interactions, and it was mentioned that **Mistral** recently released a 7b 0.2 version with a 32k context.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://www.philschmid.de/fine-tune-llms-in-2024-with-trl">How to Fine-Tune LLMs in 2024 with Hugging Face</a>: In this blog post you will learn how to fine-tune LLMs using Hugging Face TRL, Transformers and Datasets in 2024. We will fine-tune a LLM on a text to SQL dataset.</li><li><a href="https://tenor.com/view/dont-say-that-ever-again-diane-lockhart-the-good-fight-dont-say-that-never-say-that-again-gif-18052604895623551134">Dont Say That Ever Again Diane Lockhart GIF - Dont Say That Ever Again Diane Lockhart The Good Fight - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/models?pipeline_tag=image-to-text&language=en&sort=likes">Models - Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1220815376770666556)** (17 messages🔥): 

- **Request for Download Speed Limiter**: A user requested the addition of a **download speed limiter** to avoid consuming all the bandwidth at home. Other users suggested using system-level settings to throttle speeds, arguing that large downloads are common across many applications.
- **Confusion Over Image Uploading**: One user struggled to figure out how to upload images to a model. Guidance was provided by others, mentioning a tool called **.mmproj converter** and directing where to find it and how to use it.
- **Llama Image Input Issues on 0.2.17**: A user was advised to use version **0.2.16** rather than **0.2.17** of a tool due to issues with image inputs on the newer version. However, a follow-up clarified that version **0.2.16 for Linux was skipped**, and version **0.2.14** worked well for llava vision models except for moondream2.
- **Problems with --context_window Setting**: A user raised an issue with the **--context_window** setting when using LM Studio, mentioning it only works with the default setting. No direct solution was provided in the message history.
- **Moving Discussions to Relevant Channels**: A user was instructed to **move a technical discussion** to a more appropriate channel specifically dedicated to such topics.

**Link mentioned**: <a href="https://huggingface.co/nisten/obsidian-3b-multimodal-q6-gguf">nisten/obsidian-3b-multimodal-q6-gguf · Hugging Face</a>: no description found

  

---


**LM Studio ▷ #[📘-docs-and-tips](https://discord.com/channels/1110598183144399058/1123654763112824892/1221503868798636062)** (2 messages): 

- **Launch of New Docs**: LM Studio has unveiled its **new documentation website** which can be accessed at [lmstudio.ai/docs](https://lmstudio.ai/docs).
- **Navigating Multi Model Sessions**: To understand the new **Multi Model Session feature** or **JSON Mode**, users can watch a tutorial video from [5:57](https://youtu.be/4fdZwKg9IbU?feature=shared&t=357) which provides instructions and insights on their usage.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/4fdZwKg9IbU?feature=shared&t=357)">Run ANY Open-Source LLM Locally (No-Code LMStudio Tutorial)</a>: LMStudio tutorial and walkthrough of their new features: multi-model support (parallel and serialized) and JSON outputs. Join My Newsletter for Regular AI Up...</li><li><a href="https://lmstudio.ai/docs">Documentation | LM Studio</a>: Technical Reference
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1220661495445065789)** (228 messages🔥🔥): 

- **Distributed AI - A Networked Dream or Practical Scheme?**: An extended discussion took place regarding the feasibility of networked machines running language models collaboratively, akin to a mini-distributed computing environment. There was skepticism about the practicability due to latency and bandwidth constraints, but examples like HyperspaceAI's approach and methods such as DHT (distributed hash table) were cited for potential inspiration.

- **VRAM Large and in Charge**: Price and performance comparisons between high VRAM GPUs such as the RTX 3090 with its 24GB of VRAM were discussed for running AI models. There was a general consensus that, for the purpose of local machine learning, GPUs like the RTX 3090 offer the best value for VRAM capacity.

- **New Horizons in Local Compute**: The conversation touched on the prospects of LLMs (large language models) shifting computing demands back to local infrastructure much like the Linux/LAMP movement of the '90s. The parallel was made between the potential growth in LLM development and deployment and past tech revolutions that demanded significant grassroots technical expertise.

- **Macs and Memory - Apple's Big VRAM Offer**: Speculation around the RAM capacity of future Mac models, specifically the M3 Ultra Studio, was discussed with expectations it might allow for at least 256GB (V)RAM. Current models like the M3 and M2 Ultra Mac Studio were highlighted for their substantial combined system and GPU RAM, capable of performing high VRAM computations.

- **Hybrid GPU Setups - The Plot Thickens**: An attempted setup with an AMD 7800XT and NVIDIA 3060 in the same PC resulted in initialization errors with LM Studio software, prompting a conversation about the challenges of running a Frankenstein build. The broader implications of how to approach building rigs with multiple high VRAM GPUs for running large AI models were also touched upon.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://evalplus.github.io/leaderboard.html">EvalPlus Leaderboard</a>: no description found</li><li><a href="https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/">Hackers can read private AI-assistant chats even though they’re encrypted</a>: All non-Google chat GPTs affected by side channel that leaks responses sent to users.</li><li><a href="https://www.youtube.com/watch?v=nQmZmFERmrg">MemGPT Explained!</a>: Thank you so much for watching our paper summary video on MemGPT! MemGPT is a super exciting new work bridging together concepts in how Operating Systems man...</li><li><a href="https://github.com/intel/intel-npu-acceleration-library">GitHub - intel/intel-npu-acceleration-library: Intel® NPU Acceleration Library</a>: Intel® NPU Acceleration Library. Contribute to intel/intel-npu-acceleration-library development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch-labs/gpt-fast">GitHub - pytorch-labs/gpt-fast: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python.</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - pytorch-labs/gpt-fast
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1220708565686419537)** (16 messages🔥): 

- **Reboot Solves Mysterious Output Issues**: A user reported that after experiencing a decrease in output quality and models outputting gibberish, a **restart of LM Studio** resolved the issue.
- **Older AMD GPUs Unsupported by ROCM Build**: A user faced an error while loading models on LM Studio with a RX 570, which was clarified by another user stating that the **RX570 is too old** to work with the ROCM build.
- **Server Output Abruptly Stops**: Multiple members discussed an issue where the server stops after outputting only 2 tokens. Suggestions to troubleshoot included sharing logs and trying out a 'hello world (curl)' example.
- **Lengthy Sessions Lead to Garbled Output**: One member experienced **garbled output** during prolonged sessions with LM Studio, which seems to relate to a multiple of the token count but persists despite the rolling window approach to manage context.
- **Guidance Offered for Managing Token Limits**: It was noted that halving tokens could cause issues since tokens are not equivalent to words and may result in incomplete phrases affecting the model’s response. The suggestion was to replicate the experiment on the API server for better logging and analysis.
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1220607338617241660)** (48 messages🔥): 

- **Trouble with Large Language Models**: One member mentioned a limitation in running **more than one LLM** due to an 8GB VRAM cap, suggesting possible configurations to mitigate this limit.
- **Token Troubles with Autogen**: There were multiple reports of issues with Autogen, including unexpected 4-token limits outside local environments, which led to confusion among users attempting remote server setups.
- **Workflow Woes**: Users experienced frustration with Autogen workflows, with some resorting to manual edits of workflow files to adjust the `max_tokens` parameter to -1, as suggested in the thread.
- **Autogen Studio UX Struggles**: Members discussed the user experience of Autogen Studio, pointing out less-than-intuitive UI and the need for improved error messaging and model loading indicators.
- **Community Collaboration on Autogen**: The discussion showed members actively helping each other troubleshoot issues with Autogen Studio, emphasizing community-driven problem solving and knowledge sharing.
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=Nc5Yk0XXgP8
  

---


**LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1220927806385684500)** (4 messages): 

- **Windows Compatibility Confusion for MemGPT**: A participant expressed difficulty in running **MemGPT** on Windows, finding the process complicated.
- **Check the User Guide**: In response, another member suggested reviewing the **user guide** for assistance.
- **Potential Linux Exclusive**: The same member then speculated that the issue might be because **MemGPT** is a Linux-only application.
- **WSL as a Solution**: A practical solution offered was to use **Windows Subsystem for Linux (WSL)** to overcome the Windows compatibility issue.
  

---


**LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1221357876052889672)** (3 messages): 

- **AVX Beta Version Update Uncertain**: A member inquired about the possibility of an update to the **0.2.10 avx beta version**. Another stated that **supporting older hardware** is not a high priority at the moment and updates may eventually come once more pressing issues are addressed.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1220694776106586142)** (26 messages🔥): 

- **ZLUDA vs ROCm Compatibility Issues**: A user reported **100% CPU utilization** after installing ZLUDA alongside ROCm, suggesting interference or prioritization issues between the two. The discussion pointed toward a potential conflict with having ZLUDA in the path instead of ROCm, affecting performance.
  
- **ROCM Loading Woes**: A user encountered a persistent "Exit code: 42" error when attempting to load models over 10GB with ROCm offloading enabled, even with ample VRAM on an rx6950xt. The models would load without GPU offloading, albeit slowly, indicating potential issues with the offloading process.

- **Path to Resolution?**: It was proposed that users check their User PATH environment variable for entries ending in "ROCm/bin" to ensure proper connection to ROCm libraries. One user reported an absence of ROCm-related paths in their environment variables, possibly contributing to the issues.

- **Environment Variable Guidance Shared**: To help with ROCm loading errors, a user suggested adding a specific path to the User PATH variable: 
  ```C:\Users\[username]\AppData\Local\LM-Studio\app-0.2.17\resources\app\.webpack\main\build\Release\ROCm\bin``` 
  This path is intended to assist in connecting to ROCm libraries for those experiencing the loading error.

**Link mentioned**: <a href="https://winaero.com/how-to-see-names-and-values-of-environment-variables-in-windows-10/)">How to see names and values of environment variables in Windows 10</a>: In this article, we will see how to view environment variables defined in Windows 10 and their values for the current user and the system variables.

  

---


**LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1220968798946066463)** (23 messages🔥): 

- **Troubleshooting Interpreter Connection Issues**: A user faced an error with a local *Mistral model* stating "**Model with key 'gpt-4' not loaded**." After some back-and-forth discussion, making a simple `curl` request to the server fixed the issue, but the cause remained unclear.
- **Local LLM Recommendations for Open-interpreter**: Users discussed various LLM options for use with Open-interpreter: **CodeLlama 7B Instruct - GGUF** gave correct answers for sample questions, while **Mistral 7B Instruct v0.2 - GGUF** underperformed. Another recommendation was for **Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B-GGUF**, which was noted for verbosity.
- **Unboxing Open Interpreter's Hype**: Users shared views on a YouTube promotional video for Open Interpreter, with one user unable to order due to location constraints. Another shared that printing the device oneself is possible, with STL files available for free on [01.openinterpreter.com/bodies/01-light](https://01.openinterpreter.com/bodies/01-light).
- **Exploring GGUF-Compatible Models**: Users provided links to various models on Hugging Face, including CodeLlama 7B Instruct and Mistral 7B Instruct, formatted in GGUF, discussing their performance when tested with Open-interpreter.
- **Handling Non-Blessed Model Errors**: A user resolved errors encountered with non-blessed models by modifying the default system message, linking to issue [#1124 on open-interpreter's GitHub](https://github.com/OpenInterpreter/open-interpreter/issues/1124).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://01.openinterpreter.com/bodies/01-light">01 Light - 01</a>: no description found</li><li><a href="https://huggingface.co/saltlux">saltlux (saltlux)</a>: no description found</li><li><a href="https://huggingface.co/Nan-Do/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B-GGUF">Nan-Do/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF">TheBloke/CodeLlama-7B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=jWr-WeXAdeI">Open Source AI Agents STUN the Industry | Open Interpreter AI Agent + Device (01 Light ) is out!</a>: 📩 My 5 Minute Daily AI Brief 📩https://natural20.beehiiv.com/subscribe🐥 Follow Me On Twitter (X) 🐥https://twitter.com/WesRothMoneyLINKS:https://www.openin...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/issues/1124">bug:  `markdown` disabled or not supported. · Issue #1124 · OpenInterpreter/open-interpreter</a>: Describe the bug When prompting a local model, https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF, using LM Studio, I kept getting what should have been valid python output, but the code bl...</li><li><a href="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF">TheBloke/Mistral-7B-Instruct-v0.2-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1220604195145781268)** (362 messages🔥🔥): 

- **Open Interpreter Hits Ubuntu**: Discussions indicate users are managing to get Open Interpreter running on Ubuntu 22.04, with some tweaking around microphone support and client understanding. They've expressed a need to better understand the client-server operations, seeking insight from the community.

- **Hardware Hacks and Hopes**: The community is actively seeking information on building the O1 light and is interested in hacking together hardware for the project. A new dedicated Discord channel called <#1221879849535279204> was created due to high demand.

- **Excitement and Queries About 01 Device**: Users are excited about trying and potentially building around the O1 device, asking questions about its capabilities, the need for a subscription, Windows compatibility, eSIM possibilities for 4G connectivity, and if there's a UI for ease of use.

- **Tech Support Queries**: Members are troubleshooting various technical issues related to the Open Interpreter, from excessive AI chattering with the `--os` option to utilizing different LLMs and integrating with various APIs like Groq. There's a focus on enhancing the installer's user-friendliness.

- **Contribution and Community Growth**: There's eagerness within the community to contribute to Open Interpreter, with users discussing front-end app development, potential integration with Groq, the performance of different LLMs, and a desktop app for Apple silicon devices. The community supports each other's ideas and open-source efforts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.amazon.com/dp/B06XT1Z9TF.">no title found</a>: no description found</li><li><a href="https://www.goody2.ai/">GOODY-2 | The world&#x27;s most responsible AI model</a>: Introducing a new AI model with next-gen ethical alignment. Chat now.</li><li><a href="https://docs.openinterpreter.com/language-models/hosted-models/openai">OpenAI - Open Interpreter</a>: no description found</li><li><a href="https://x.com/hellokillian/status/1757526563879587995?s=20).">Tweet from killian (@hellokillian)</a>: ..jesus  open interpreter&#39;s first vision model, piloting my 8gb M1 macbook. 100% offline.  this will be inside every computer in the world.</li><li><a href="https://docs.openinterpreter.com/settings/all-settings#max-tokens">All Settings - Open Interpreter</a>: no description found</li><li><a href="https://x.com/openinterpreter/status/1771358466877321227?s=46">Tweet from Open Interpreter (@OpenInterpreter)</a>: https://twitter.com/i/spaces/1dRJZEPewmgGB</li><li><a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: Learn how to deploy + call models from different providers on LiteLLM</li><li><a href="https://docs.litellm.ai/docs/providers/groq">Groq | liteLLM</a>: https://groq.com/</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/3e95571dfcda5c78115c462d977d291567984b30/interpreter/core/llm/llm.py#L117">open-interpreter/interpreter/core/llm/llm.py at 3e95571dfcda5c78115c462d977d291567984b30 · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/utils/count_tokens.py">open-interpreter/interpreter/terminal_interface/utils/count_tokens.py at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/lavague-ai/LaVa">Redirect Notice</a>: no description found</li><li><a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/MikeBirdTech/op">Redirect Notice</a>: no description found</li><li><a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/lavague-ai/LaVague&ved=2ahUKEwiE54Sl8IuFAxXQs1YBHfqkCJ0QFnoECA8QAQ&usg=AOvVaw1b8qvOy99zeAyRN_tGJuYY">no title found</a>: no description found</li><li><a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/MikeBirdTech/open-interpreter-termux&ved=2ahUKEwi_vZHz-42FAxUPJzQIHQx_BuQQFnoECBMQAQ&usg=AOvVaw3stRzAssQaHpTjlvYh3KQD">no title found</a>: no description found</li><li><a href="https://x.com/openinterpreter">Tweet from undefined</a>: no description found</li><li><a href="https://groq.com/">GroqChat</a>: no description found</li><li><a href="https://humanaigc.github.io/animate-anyone/">Animate Anyone</a>: Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://youtu.be/YxiNUST6gU4?si=fSBtR7Tw6WCvWNvN">Introducing Light 01: World&#39;s First Personal AI Assistant by Open Interpreter (Full Setup)</a>: In this video, we&#39;ll look at the OpenInterpreter Light 01 GitHub repository, a cutting-edge project that&#39;s revolutionizing how we interact with computers usi...</li><li><a href="https://github.com/Tas667/scalpel/">GitHub - Tas667/scalpel: python script that helps you quickly understand the structure and contents of an unknown project.</a>: python script that helps you quickly understand the structure and contents of an unknown project. - Tas667/scalpel</li><li><a href="https://youtu.be/FXCaJ3Ga9TE?si=mHELyLpTr8I0MtuM&t=351">How to use Open Interpreter cheaper! (LM studio / groq / gpt3.5)</a>: Part 1 and intro: https://www.youtube.com/watch?v=5Lf8bCKa_dE0:00 - set up1:09 - default gpt-42:36 - fast mode / gpt-3.52:55 - local mode3:39 - LM Studio 5:5...</li><li><a href="https://github.com/cs50victor/os1">GitHub - cs50victor/os1: AGI operating system ( UI for openinterpreter&#39;s 01 )</a>: AGI operating system ( UI for openinterpreter&#39;s 01 ) - cs50victor/os1</li><li><a href="https://x.com/fieroty/status/1772004445217489196?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">Tweet from Ty (@FieroTy)</a>: local LLMs with the 01 Light? easy</li><li><a href="https://nosta.me/">Nosta</a>: no description found</li><li><a href="https://tenor.com/view/surprised-pikachu-pokemon-shock-surprised-pikachu-gif-15357817">Surprised Pikachu GIF - Surprised Pikachu Pokemon - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.litellm.ai/docs/providers/anthropic#supported-models">Anthropic | liteLLM</a>: LiteLLM supports</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/CONTRIBUTING.md">open-interpreter/docs/CONTRIBUTING.md at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://tx.nixc.us/65TjpxNIT7/OpenInterpreter%20in%20Webtop.mov">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1220612133792907265)** (576 messages🔥🔥🔥): 

- **Building the 01 Yourself**: Members are discussing the process of 3D printing and assembling their own **01 Light** devices, with plenty of enthusiasm for DIY. The community is sharing insights on materials, settings, and possible design tweaks for printing, with some planning to iterate on their self-made devices. 
- **Availability of 01 Outside the US**: Pre-orders are currently limited to the US only, with no estimated time for international release provided. However, community members from outside the US are encouraged to build their own devices and collaborate with others.
- **Understanding and Running 01 Server on Machines**: Conversations include questions about running **01 server** on various machines with different specs, including low-spec and cloud options. It suggests that as long as the machine can handle the model, it could be feasible, with the actual LLM being the most resource-intensive part.
- **Developing 01 Capabilities**: There's excitement around extending the functionalities of **01 Light**, like integrating LEDs, speakers, or SIM card capabilities for connectivity. Ideas are being exchanged on how to create a more versatile and DIY-friendly design.
- **General Questions and Excitement**: New members are voicing their anticipation for the device, asking questions about estimated delivery times, subscription requirements, and compatibility with Windows or Mac for automation tools.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.voltera.io/v-one">V-One - A Desktop PCB Printer | Voltera</a>: V-One is a 4-in-1 desktop PCB printer. Prototype and assemble PCBs in an hour and get immediate feedback on new designs.</li><li><a href="https://x.com/hellokillian">Tweet from undefined</a>: no description found</li><li><a href="https://www.hackster.io/news/m5stack-launches-the-m5dial-a-swish-iot-rotary-encoder-with-a-built-in-color-touchscreen-display-438513c4e52c">M5Stack Launches the M5Dial, a Swish IoT Rotary Encoder with a Built-In Color Touchscreen Display</a>: With support for the Arduino IDE, ESP-IDF, and UIFlow, this multifunctional gadget aims to power your Internet of Things control projects.</li><li><a href="https://tenor.com/view/shut-up-and-take-my-money-futurama-fry-take-my-money-money-gif-15195954">Shut Up And Take My Money Futurama GIF - Shut Up And Take My Money Futurama Fry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://changes.openinterpreter.com/">Open Interpreter Blog</a>: Official changelog for the open-source Open Interpreter project.</li><li><a href="https://www.youtube.com/@MikeBirdTech/videos">Mike Bird</a>: A.I. engineering  </li><li><a href="https://shop.m5stack.com/products/atom-echo-smart-speaker-dev-kit">ATOM Echo Smart Speaker Development Kit</a>: ATOM ECHO is a programmable smart speaker.This eps32 AIoT Development Kit has a microphone and speaker for AI voice interaction light and small. It can be access AWS, Baidu, ESPHome and Home Assistant...</li><li><a href="https://shop.m5stack.com/products/unitv2-ai-camera-gc2145">M5Stack UnitV2 - The standalone AI Camera for Edge Computing (SSD202D) TinyML</a>: UnitV2 is a high-efficiency AI recognition module, using Sigmstar SSD202D chip, having 128MB-DDR3 memory, 512MB Flash, and 1080P camera. UnitV2 is simple to use and efficient,supporting AI recognition...</li><li><a href="https://x.com/openinterpreter/status/1771358466877321227?s=46">Tweet from Open Interpreter (@OpenInterpreter)</a>: https://twitter.com/i/spaces/1dRJZEPewmgGB</li><li><a href="https://github.com/rhasspy/piper/?tab=readme-ov-file#running-in-python">GitHub - rhasspy/piper: A fast, local neural text to speech system</a>: A fast, local neural text to speech system. Contribute to rhasspy/piper development by creating an account on GitHub.</li><li><a href="https://0ggfznkwh4j.typeform.com/to/WfuYTxMM?typeform-source=pcr08jir95k.typeform.com">Contact Us</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.</li><li><a href="https://github.com/OpenInterpreter/01/blob/main/hardware/light/BOM.md">01/hardware/light/BOM.md at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=xPd8FFzIeOw">ChatGPT &quot;Code Interpreter&quot; But 100% Open-Source (Open Interpreter Tutorial)</a>: This is my second video about Open Interpreter, with many new features and much more stability, the new Open Interpreter is amazing. Update: Mixtral 7x8b was...</li><li><a href="https://github.com/OpenInterpreter/01/issues">Issues · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/docs/bodies">01/docs/bodies at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://0ggfznkwh4j.typeform.com/to/WfuY">Discover Typeform, where forms = fun</a>: Create a beautiful, interactive form in minutes with no code. Get started for free.</li><li><a href="https://github.com/openai/whisper">GitHub - openai/whisper: Robust Speech Recognition via Large-Scale Weak Supervision</a>: Robust Speech Recognition via Large-Scale Weak Supervision - openai/whisper</li><li><a href="https://github.com/adafruit/Adafruit-PAM8302-Mono-Amplifier-PCB">GitHub - adafruit/Adafruit-PAM8302-Mono-Amplifier-PCB: PCB files for the Adafruit PAM8302 Mono Amplifier</a>: PCB files for the Adafruit PAM8302 Mono Amplifier. Contribute to adafruit/Adafruit-PAM8302-Mono-Amplifier-PCB development by creating an account on GitHub.</li><li><a href="https://wiki.seeedstudio.com/xiao_esp32s3_speech2chatgpt/">Miniature ChatGPT Voice Assistant based on XIAO ESP32S3 Sense | Seeed Studio Wiki</a>: This tutorial explains how to use the XIAO ESP32S3, record a voice, recognise the voice and then ask ChatGPT a question and get an answer to display on the screen.</li><li><a href="https://youtu.be/4fdZwKg9IbU?si=_rOJ4fXzAO7SpuPE">Run ANY Open-Source LLM Locally (No-Code LMStudio Tutorial)</a>: LMStudio tutorial and walkthrough of their new features: multi-model support (parallel and serialized) and JSON outputs. Join My Newsletter for Regular AI Up...</li><li><a href="https://youtu.be/-Y1wWJAnqRk?si=PLWODfGzDtGR4Poc">PCB prototyping, PCB making at home - WEGSTR</a>: Experience the fascinating world of PCB manufacturing with this step-by-step video guide. Learn the art of making high-quality PCBs using a CNC milling machi...</li><li><a href="https://x.com/i/spaces/1dRJZEPewmgGB">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://www.adafruit.com/product/3968?gad_source=1&gclid=CjwKCAjwnv-vBhBdEiwABCYQA2Mme3uVca46pohzqJ-jT8IzOZ7Xew6y5cEefuqKkRNhJdbfjdvKyxoC-lAQAvD_BwE">Speaker - 40mm Diameter - 4 Ohm 3 Watt</a>: Hear the good news! This speaker&amp;nbsp;is a great addition to any audio project where you need a&amp;nbsp;4 Ohm impedance and 3W or less of power.At 40mm diameter it has a more square-ish shape,  ....</li><li><a href="https://www.adafruit.com/product/3341">DotStar Micro LEDs (APA102&ndash;2020) - Smart SMD RGB LED - 10 pack</a>: These incredibly small surface-mount LEDs are an easy way to add a lot of very tiny (but bright!) colorful dots to your project. They&amp;#39;re mini versions of&amp;nbsp;the ones in our digital  ...</li><li><a href="https://www.instagram.com/reel/C41iQZ6L0_I/"> Concept Bytes on Instagram: &quot;A useful Ai named Jarvis. 
What features do you wanna see next?
#ironman #ai #tech #xtool #xtoolf1 
&#064;xtool.official&quot;</a>: 6,784 likes, 154 comments - concept_bytes on March 22, 2024: &quot;A useful Ai named Jarvis.  What features do you wanna see next? #ironman #ai #tech #xtool #xtoolf1  &#064;xtool.official&quot;</li><li><a href="https://github.com/OpenInterpreter/01/blob/main/software/source/server/i.py">01/software/source/server/i.py at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/01/blob/main/ROADMAP.md">01/ROADMAP.md at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=JeyZ4HQARMc">Wake word demonstration on Raspberry Pi and custom ESP32 board in Home Assistant | Year of the Voice</a>: If there&#39;s one thing people have been asking for during the Year of the Voice, it&#39;s wake word. The ability to say something like &quot;OK Google,&quot; &quot;Hey Siri,&quot; or ...</li><li><a href="https://youtu.be/nzznxPeWDO4?si=sSA3iuiZQZEqgogG">Shocking Open Interpreter AI Agent + Device (01 Light ) is Finally Revealed</a>: Shocking Open Interpreter AI Agent + Device (01 Light ) is Finally Revealed#OpenInterpreter #aiagent CHANNEL LINKS:🕵️‍♀️ Join my Patreon: https://www.patreo...</li><li><a href="https://github.com/Tas667/scalpel/">GitHub - Tas667/scalpel: python script that helps you quickly understand the structure and contents of an unknown project.</a>: python script that helps you quickly understand the structure and contents of an unknown project. - Tas667/scalpel</li><li><a href="https://github.com/SYSTRAN/faster-whisper">GitHub - SYSTRAN/faster-whisper: Faster Whisper transcription with CTranslate2</a>: Faster Whisper transcription with CTranslate2. Contribute to SYSTRAN/faster-whisper development by creating an account on GitHub.</li><li><a href="https://github.com/m-bain/whisperX">GitHub - m-bain/whisperX: WhisperX:  Automatic Speech Recognition with Word-level Timestamps (&amp; Diarization)</a>: WhisperX:  Automatic Speech Recognition with Word-level Timestamps (&amp; Diarization) - m-bain/whisperX</li><li><a href="https://developer.nvidia.com/blog/building-generally-capable-ai-agents-with-minedojo/">Building Generally Capable AI Agents with MineDojo | NVIDIA Technical Blog</a>: NVIDIA is helping push the limits of training AI generalist agents with a new open&#x2d;sourced framework called MineDojo.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1220623095551885402)** (11 messages🔥): 

- **World's First Fully Open-Source AI Assistant**: A YouTube video titled "Open Interpreter's 01 Lite - WORLD'S FIRST Fully Open-Source Personal AI AGENT Device" which reviews and shows the installation of the **01 Lite**, a 100% open-source personal AI assistant, was highlighted in the chat. Here is the [video link](https://www.youtube.com/watch?v=Q_p82HtBqoc).
- **Live AI Software and Party Recap**: A member shared a link to a YouTube live video of their first attempt at running 01 software along with recordings from the launch party that took place on Discord. The stream can be viewed [here](https://www.youtube.com/live/QQl9dIfqv58?si=0nEPsJwLJWSW8H_y&t=2227).
- **The Necessity of High-Quality LLMs**: **vincentjedi** discussed the essential role of large language models (LLMs) in the future of Open Interpreter (OI), stating that progress is "100 percent" reliant on the ability of LLMs to translate prompts into bug-free commands.
- **Pioneering the 'Rabbit Strategy'**: The idea of using a "rabbit strategy" to train large action models through user interactions fed back into a cloud service was mentioned by **vincentjedi** as a necessary approach for OI.
- **Optimistic View on UI/UX Challenges**: While **vincentjedi** noted the significant challenge of achieving a bug-free user experience across various apps and interfaces, **techfren** pointed out the rapid trial-and-error approach that can be applied safely and more efficiently in UI testing.
- **Edited 01 Software Stream for Convenience**: An edited version of the live stream featuring segments focusing on 01 software was posted by **techfren** for easier viewing, presenting an insightful resource for those interested in OI's offerings. Watch the edited stream [here](https://www.youtube.com/watch?v=l3fUlHjEmZE).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Q_p82HtBqoc">Open Interpreter&#39;s 01 Lite - WORLD&#39;S FIRST Fully Open-Source Personal AI AGENT Device</a>: 01 Lite by Open Interpreter is a 100% open-source personal AI assistant that can control your computer. Let&#39;s review it and I&#39;ll show you how to install open...
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1220711550726180939)** (574 messages🔥🔥🔥): 

- **European Data Laws Hinder LAION's Potential**: A discussion highlighted that datasets from LAION may be less effective than their US counterparts due to stringent EU regulations. It was suggested that until the EU relaxes its data laws, reliance on synthetic data and collaborations with less restrictive jurisdictions will be necessary, a process humorously termed as "data laundering."

- **Stability AI CEO Steps Down Amidst Chaos**: Stability AI's founder, Emad Mostaque, announced his resignation as CEO, confirmed by a [press release](https://stability.ai/news/stabilityai-announcement) from the company. Interim co-CEOs Shan Shan Wong and Christian Laforte will lead the search for a new permanent CEO, while speculation arises about the potential consequences for the company's direction and open-source commitments.

- **SD3 Model Expectations Set**: Previews of the SD3 model suggest it performs comparably to DALL-E 3 in certain contexts but overall struggles with understanding complex concepts and interactions within prompts. Despite a more realistic image generation capability, the SD3 model reportedly often assembles images in a collage-like fashion, without a clear blend of concepts.

- **AI Drama and Ethics in the Spotlight**: A message pointed to a conversation on Twitter where concerns were raised about the motivations behind prominent figures in the AI industry. This sparked a discussion about the ethical responsibilities of AI developers and researchers and the infatuation with AI "celebrity" culture on social media platforms.

- **Performance Challenges with AMD for AI**: Users shared their frustrations with AMD GPUs and ROCm support for ML workloads, comparing unfavorably to NVIDIA's solutions. Anecdotal evidence suggested AMD's lack of investment in consumer-level ML support could be a missed opportunity in the rise of generative AI models like Stable Diffusion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.forbes.com/sites/kenrickcai/2024/03/22/stability-ai-founder-emad-mostaque-plans-to-resign-as-ceo-sources-say/?sh=5e3e64bd5239">Stability AI Founder Emad Mostaque Plans To Resign As CEO, Sources Say</a>: Mostaque has told a number of people close to him that he plans to step down as chief executive of the once buzzy generative AI startup known for Stable Diffusion.</li><li><a href="https://x.com/chrlaf/status/1771933102329171976?s=20">Tweet from Christian Laforte (@chrlaf)</a>: @USEnglish215753 @StabilityAI @EMostaque Yes, the plan hasn&#39;t changed, we are still hard at work improving the model towards open release.</li><li><a href="https://webllm.mlc.ai/#chat-demo>">WebLLM | Home</a>: no description found</li><li><a href="https://lifehacker.com/tech/its-not-safe-to-click-links-on-x">It's Not Safe to Click Links on X</a>: When someone posts a link on X, the site generates a link preview. But reportedly, this system can be tricked, and bad actors can redirect you to malicious sites from a falsely advertised link preview...</li><li><a href="https://stability.ai/news/stabilityai-announcement">Stability AI Announcement &mdash; Stability AI</a>: Earlier today, Emad Mostaque resigned from his role as CEO of Stability AI and from his position on the Board of Directors of the company to pursue decentralized AI.  The Board of Directors has appoin...</li><li><a href="https://tenor.com/Di0E.gif">Money Crying GIF - Money Crying Woody Harrelson - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/YqCL.gif">Im Doing My Part Serious GIF - Im Doing My Part Serious Stare - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1bmtp77/do_not_generate_a_tree_using_a_model_trained_on/>">reddit.com: over 18?</a>: no description found</li><li><a href="https://archive.ph/J7Xdw">Stability AI Founder Emad Mostaque Plans To Resign As CEO, Sources Say</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1220751321188012102)** (92 messages🔥🔥): 

- **Andrew Ng Predicts AI Workflow Revolution**: Andrew Ng, a co-founder of Google Brain, predicts that **AI agentic workflows** will drive significant progress in AI this year, potentially outpacing even next-generation foundation models. He emphasizes the importance of iterating over documents multiple times, such as outlining, drafting, and revising, contrasting it with the current zero-shot LLM approach ([Highlight from Reddit](https://www.reddit.com/r/singularity/comments/1bl3s9r/andrew_ng_cofounder_of_google_brain_former_chief/)).

- **MIT CSAIL Speeds Up Image Generation**: MIT researchers have made a breakthrough by creating a framework that accelerates the image-generating process of tools like **Stable Diffusion** and **DALL-E** by 30 times, simplifying it to a single step without losing image quality, through a teacher-student model ([MIT News Article](https://news.mit.edu/2024/ai-generates-high-quality-images-30-times-faster-single-step-0321)).

- **NVIDIA Explores Training of Diffusion Models**: NVIDIA's blog post discusses the challenges in improving diffusion models and how they tackle issues shared by many neural networks during training. They point to their EDM2 code and model release and suggest ownership issues related to style normalization may require addressing through modifications like those in EDM2 ([NVIDIA Developer Blog Post](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/)).

- **Debating Unet's Relevance in an Era of Linear Networks**: Conversations cast doubt on the value of enhancements to Unet architectures given the shift toward linear network models for image generation tasks. Some argue that linear models do not require traditional normalization methods, while others express skepticism, suggesting that concepts like layer norm remain integral to neural network functionality.

- **Strategic Pruning and the Mystery of Middle Blocks**: Discussion on the resilience of large language models (LLMs) leads to the insight that removing blocks from the middle of the network causes minimal degradation to performance. This leads to speculation about the potential redundancy of certain network segments, especially "unet middle block," and the need for further study into the architectural peculiarities of linear networks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.mit.edu/2024/ai-generates-high-quality-images-30-times-faster-single-step-0321">AI generates high-quality images 30 times faster in a single step </a>: A new distribution matching distillation (DMD) technique merges GAN principles with diffusion models, achieving 30x faster high-quality image generation in a single computational step and enhancing to...</li><li><a href="https://tenor.com/view/explode-cute-cat-gif-14074577">Explode Cute GIF - Explode Cute Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.google.com/document/d/1M_QWSRv44M3j69Sxq1fcgfowvgioS5nYfP84D9keUeI/edit">TRC Report 4</a>: no description found</li><li><a href="https://www.reddit.com/r/singularity/comments/1bl3s9r/andrew_ng_cofounder_of_google_brain_former_chief/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/">Rethinking How to Train Diffusion Models | NVIDIA Technical Blog</a>: After exploring the fundamentals of diffusion model sampling, parameterization, and training as explained in Generative AI Research Spotlight: Demystifying Diffusion&#x2d;Based Models&#8230;
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1221088542047080459)** (26 messages🔥): 

- **Shipping Timeline Inquiry**: A member inquired about shipping timelines for an unspecified product, suggesting a desire to order one for experimenting without affecting customer availability. Another member indicated that there were no firm dates but shipping is expected in the summer, with an alternative option to build it oneself.
- **Raptor Implementation Exploration**: A member described implementing a version of Raptor, generating summaries over clusters and using sentence-level word-to-vector for embedding without pretraining. The member noted a 5-minute summary generation for a 3b model on transcript documents, stating that the technique may produce many generations but could be equivalent to prompt with chunk summarization.
- **Claude Appreciation Expressed**: A brief message from a member expressed enthusiasm for Claude, while another member anticipates its impact on open-source development and the creation of new projects due to the quality of its context.
- **Sharing FastAPI Resources**: A member shared a link to [FastAPI](https://fastapi.tiangolo.com/), touting its ease of use and readiness for production. They also linked to its [documentation](https://fastapi.tiangolo.com) and [source code](https://github.com/tiangolo/fastapi), inquiring about open-source projects using this backend framework.
- **Suno.AI Creative Fun**: Links to Suno.AI were shared by a member, indicating it’s an enjoyable platform with another confirming its ability to create Spotify playlists. Members seemed to express delight in the platform’s output.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fastapi.tiangolo.com/">FastAPI</a>: FastAPI framework, high performance, easy to learn, fast to code, ready for production</li><li><a href="https://www.youtube.com/watch?v=Nc5Yk0XXgP8">Mr. Beast Meets Mistral: AI Created a Cookbook Based on His Wildest Stunts!</a>: Today we create Beast CookbookThe &quot;Beast Cookbook&quot; idea is a fun and creative way to engage with Mr. Beast&#39;s content and generate an entertaining, fictional ...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1220773709313474622)** (13 messages🔥): 

- **Kubernetes for Nous Models**: Members discuss deploying Nous models using Kubernetes, referencing a [tweet by Sidero Labs](https://twitter.com/SideroLabs/status/1771207304748167445). Techniques include running Ollama with GGUF model format in Docker containers and then orchestrating with Kubernetes pods.
  
- **OpenRouter.ai Discovery**: A member brought attention to the [OpenRouter.ai](https://openrouter.ai/) service, which some have used to access Opus. It's associated with a Discord staff member.
  
- **Persuasive Power of Language Models Analyzed**: An [arXiv paper](https://arxiv.org/abs/2403.14380) pre-registered study is mentioned, focusing on the persuasive capabilities of large language models in debates against humans.
  
- **Showcase of AI-Driven Art Platforms**: Shared links include [ArtHeart.ai](https://artheart.ai/), a platform where users can entertain, create, and earn with AI characters, and [novelcrafter](https://novelcrafter.co), among others.
  
- **Assessing BitNet's Quantized-Aware Training**: A [Hugging Face blog post](https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5) delves into experiments with BitNet 1.5, discussing potential speedups during inference and limitations during training due to the need for smooth optimizer gradients.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/winglian/status/1771918928341794821?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from Wing Lian (caseus) (@winglian)</a>: It&#39;s unclear whether we&#39;ll be able to achieve the improvements seen in the paper when integrating DenseFormers with a pretrained Mistral-7B as they found the best performance was seen training...</li><li><a href="https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5">Experiments with Bitnet 1.5 (ngmi)</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.14380">On the Conversational Persuasiveness of Large Language Models: A Randomized Controlled Trial</a>: The development and popularization of large language models (LLMs) have raised concerns that they will be used to create tailor-made, convincing arguments to push false or misleading narratives online...</li><li><a href="https://openrouter.ai/">OpenRouter</a>: A router for LLMs and other AI models
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/)** (1 messages): 

proprietary: @everyone https://twitter.com/NousResearch/status/1771735632035127594
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1220619542405976094)** (469 messages🔥🔥🔥): 

- **World Simulator Wows Crowd**: The community is abuzz with the impact of the World Simulator project, finding the immersive prompting and AI-generated universe especially cool and engaging. Users shared experiences like creating a prequel to "The Three-Body Problem" and generating unique, if sometimes baffling, civilizational evolutions.
  
- **Wait for Winged AI Therapist**: A member discussed their work-in-progress AI therapist called Thestral, mulling over fine-tuning NousResearch's Hermes on the LLaMA 70B for a therapy-focused outcome. They shared their intention to implement it using a dataset designed for therapeutic conversation fine-tuning.

- **Sim's Opus Opulence**: Users discussed the underlying model used in World Simulator, with Opus from Claude 3 being cited for its capability to creatively simulate universes, despite its refusals and "ethical" constraints. There's a shared sentiment that, despite limitations and costs, Opus provides a more satisfying user experience than alternative models.

- **Curious Case of Model Refusals**: A detailed discussion unveiled concerns over refusal prompts embedded in the Hermes 2 Pro function-calling model which could interfere with customized AI functionalities. Members deliberated on the dichotomy of effective refusal prompts versus the potential for models to adapt to newly incorporated functions.

- **Decoding the Overton Effect in LLMs**: A member elucidated the so-called "Overton Effect" in LLMs, which leads to AI models like Claude to steer conversations towards more commonly accepted norms, potentially stifling creativity and novelty in the generative process. This insight sparked conversations about manipulating model prompting to bypass standard LLM limitations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1">mixedbread-ai/mxbai-rerank-large-v1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO">NousResearch/Nous-Hermes-2-Mistral-7B-DPO · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Nous">Nous (موسى عبده هوساوي )</a>: no description found</li><li><a href="https://vatsadev.github.io/articles/transformerMath.html">Transformers learn patterns, math is patterns</a>: no description found</li><li><a href="https://x.com/marvinvonhagen/status/1771609042542039421?s=46&t=TOasxww3M5DjlB4iBWa_ig">Tweet from Marvin von Hagen (@marvinvonhagen)</a>: Mistral just announced at @SHACK15sf that they will release a new model today:  Mistral 7B v0.2 Base Model  - 32k instead of 8k context window - Rope Theta = 1e6 - No sliding window</li><li><a href="https://tenor.com/view/spongebob-why-why-why-why-why-why-why-why-why-why-why-why-why-gif-25252239">Spongebob Why Why Why Why Why Why Why GIF - Spongebob Why Why Why Why Why Why Why Why - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/hd_nxx-gif-26166561">Hd_nxx GIF - HD_NXX - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://models.mistralcdn.com/mistral-7b-v0-2/mistral-7B-v0.2.tar">no title found</a>: no description found</li><li><a href="https://gandalf.lakera.ai/">Gandalf | Lakera – Test your prompting skills to make Gandalf reveal secret information.</a>: Trick Gandalf into revealing information and experience the limitations of large language models firsthand.</li><li><a href="https://tenor.com/view/sifas-ruby-kurosawa-love-live-merge-gif-20382260">Sifas Ruby Kurosawa GIF - Sifas Ruby Kurosawa Love Live - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/RESMPDEV/Mistral-7B-v0.2/tree/main">RESMPDEV/Mistral-7B-v0.2 at main</a>: no description found</li><li><a href="https://x.com/jmorgan/status/1771705967886929960?s=46">Tweet from Jeffrey Morgan (@jmorgan)</a>: Run Mistral&#39;s new base text completion model updated to v0.2 with Ollama:  ollama run mistral:text  https://ollama.com/library/mistral:text</li><li><a href="https://x.com/AlexReibman/status/1771608346635751541?s=20">Tweet from Alex Reibman 🖇️ (@AlexReibman)</a>: Mistral casually dropping a new model at the @cerebral_valley hackathon</li><li><a href="https://huggingface.co/jinaai/jina-embeddings-v2-base-en">jinaai/jina-embeddings-v2-base-en · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/Q_p82HtBqoc">Open Interpreter&#39;s 01 Lite - WORLD&#39;S FIRST Fully Open-Source Personal AI AGENT Device</a>: 01 Lite by Open Interpreter is a 100% open-source personal AI assistant that can control your computer. Let&#39;s review it and I&#39;ll show you how to install open...</li><li><a href="https://huggingface.co/datasets/wesley7137/therapist-sft-format">wesley7137/therapist-sft-format · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/furlat/SpriteStash">GitHub - furlat/SpriteStash: A multimodal sprites vectordb using LanceDB, Pydantic and pygame-ce</a>: A multimodal sprites vectordb using LanceDB, Pydantic and pygame-ce - furlat/SpriteStash</li><li><a href="https://github.com/mistralai-sf24/hackathon">GitHub - mistralai-sf24/hackathon</a>: Contribute to mistralai-sf24/hackathon development by creating an account on GitHub.</li><li><a href="https://gist.github.com/fullstackwebdev/5e812f46c542ab8869db899b0c535fc2">unsloth_finetune_mistral-7b-v0.2-on-openhermes-2.5-dataset.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/jquesnelle/crt-terminal">GitHub - jquesnelle/crt-terminal: Retro styled terminal shell</a>: Retro styled terminal shell. Contribute to jquesnelle/crt-terminal development by creating an account on GitHub.</li><li><a href="https://huggingface.co/colbert-ir/colbertv2.0">colbert-ir/colbertv2.0 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1220716612810706985)** (24 messages🔥): 

- **Few-Shot Prompts in SFT Datasets?**: A few members engaged in the question of the normalcy or benefit of including few-shot prompts in instruction SFT datasets for boosting few-shot prompting capabilities. The conversation leaned towards the practice not being common, with an additional thread mentioned for further insight: [relevant thread](https://discord.com/channels/1053877538025386074/1109649177689980928/1212266969131130880).
 
- **Searching for a Tiny LLM**: A member asked for a recommendation for a **Low-Parameter LLM** to learn from where another tried to clarify whether they meant 100M parameters, and a recommendation to watch **Andrej Karpathy's videos** for such models was given.

- **Causal Masking Theory or Engineering Hack?**: The necessity of causal masking in attention was questioned, with another member pointing out its importance for the model to learn next token prediction.

- **The Tri-Layer Mystery of Llama's Feedforward**: Discussion about Llama's feedforward having three linear layers was clarified with the mention of a GitHub issue and an [arXiv paper](https://arxiv.org/pdf/2002.05202.pdf). The implementation of SwiGLU was highlighted as a successful nonlinearity choice used in the model design.

- **Comparing ORPO to SFT+DPO and Model Preference for Coding**: A query arose whether **ORPO** reliably outperforms **SFT+DPO**, with no consensus reached in the chat, and a separate inquiry into the preferred local model for coding in lmstudio was met with the mention that no specific model has stood out.

**Link mentioned**: <a href="https://github.com/meta-llama/llama/issues/1004">Why does the FeedForward have three linear layer? · Issue #1004 · meta-llama/llama</a>: I find that the FFN implementation has three linear layers. https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L337-L345 But in the paper &quot;Atte...

  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1220606789326864456)** (3 messages): 

- **Confirmation of Ability**: A member expressed confidence in their ability to complete a task by stating, *“let me see if i could do it.”*
- **Inquiry on Model Characteristics**: A question was raised regarding which models are considered *"nonagreeable"* without any further context or follow-up provided.
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1220652967787495476)** (19 messages🔥): 

- **Chit-chat on Discord Parenthood**: Members briefly discussed the joys and surprises of parenthood, with one noting how "the biological machinery kicked in" after deciding to become a parent, resulting in unexpected happiness.
- **In Search of Open Source Wikipedia RAG Index**: A member inquired about an open-source Wikipedia RAG Index, and another suggested that there are similar resources available by various contributors.
- **Insights on RAFT by Microsoft and UC Berkley**: A link was shared to a paper and Twitter post discussing "Retrieval Augmented Fine-Tuning (RAFT)" which aims to make Language Models like Llama 7B more robust by training with distractor documents and incorporating chain-of-thought. The shared [paper and post](https://huggingface.co/papers/2403.10131) showed RAFT's promising results, such as outperforming GPT-3.5 in medical contexts.
- **Repository Link for RAFT Implementation**: The GitHub repository for RAFT implementation, coined "Gorilla", was shared, offering an API store for Large Language Models (LLMs). The repository can be found at [GitHub - ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main/raft).
- **Discussion on GermanRAG and Cross-Document Knowledge Retrieval**: One member mentioned a project called GermanRAG while discussing the challenges of gathering knowledge across multiple documents. Another member confirmed this complexity and hinted at a potential solution they've been working on.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ShishirPatil/gorilla/tree/main/raft">gorilla/raft at main · ShishirPatil/gorilla</a>: Gorilla: An API store for LLMs. Contribute to ShishirPatil/gorilla development by creating an account on GitHub.</li><li><a href="https://x.com/_philschmid/status/1771456524763697591?s=20">Tweet from Philipp Schmid (@_philschmid)</a>: Can we make RAG applications more robust with fine-tuning? A paper by @Microsoft  and  UC Berkley put this to the test to see if small open LLMs, like @AIatMeta Llama 7B, can match @OpenAI GPT-3.5.  T...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1221910863972143154)** (2 messages): 

- **Language Setting Hint Offered**: A member shared a [GIF from Tenor](https://tenor.com/view/everyone-get-in-here-grim-patron-gif-26273450) and noted that Tenor.com's language settings can be changed if it does not match the user's browser language.
- **A Breezy Greeting**: Another member simply dropped in to say "helloooo" to the chat.

**Link mentioned**: <a href="https://tenor.com/view/everyone-get-in-here-grim-patron-gif-26273450">Everyone Get In Here Grim Patron GIF - Everyone Get In Here Grim Patron - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1221870596015657032)** (1 messages): 

- **Sora Inspires Creativity in Artists and Filmmakers**: OpenAI highlights its collaboration with creatives using **Sora**, a tool that aids in bringing imaginative ideas into reality. Director Paul Trillo praises its potential, "Sora is at its most powerful when you’re not replicating the old but bringing to life new and impossible ideas we would have otherwise never had the opportunity to see."

- **Sora: A Bridge to the Surreal**: Production company shy kids values Sora for its capacity to "make things that are totally surreal," signaling a leap beyond generating realistic images towards crafting the unimaginable. Their excitement and the potential applications in creative workflows are detailed on the [OpenAI blog](https://openai.com/blog/sora-first-impressions).

**Link mentioned**: <a href="https://openai.com/blog/sora-first-impressions">Sora: First Impressions</a>: We have gained valuable feedback from the creative community, helping us to improve our model.

  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1220659924212908032)** (264 messages🔥🔥): 

- **Exploring LLM Biases and Defaults**: A robust discussion unfolded on biases in AI, specifically surrounding the default alignment of general-purpose LLMs like GPT towards a "Western liberal-centrist" value system. A user proposed the idea of creating multiple "aligned" versions of AI models, arguing that current LLMs implicitly present Western values as optimal by default.

- **Customizing ChatGPT**: The 'Customize ChatGPT' feature was highlighted as a way to inject personal values or cultural background into AI responses. It was suggested that instead of identifying AI biases, users could focus on how ChatGPT can add productive value to their lives.

- **Aligning AI to Non-Western Norms Proves Tricky**: Efforts to guide AI towards non-Western answers showed mixed results, with AI still tending to incline towards Western-centric ideals. Despite experimenting with inhibiting 'Western-centric' prompts such as 'TikTok' and attempting to influence it with non-Western scaffolding, the challenge remains to prevent the AI from exerting Western values on answers.

- **On Bias, Culture, and Reinforcement**: The conversation touched on concerns about whether AI may reinforce existing biases if aligned with specific cultural or political viewpoints. The discussion considered whether AI should aim to broaden views or if users should have options to specify political or cultural alignments.

- **Looking for Practical AI Solutions**: Users shared tips for handling common AI shortcomings like correcting DALL-E 3's misrepresentation of fingers and hands and the lack of a seed feature for subtle modifications. Discussion also ventured into the need for clearer communication from AI providers regarding certain features being excluded or postponed.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ehsanxpin/status/1771578381869465606?s=20">Tweet from Ehsan Azari (@ehsanxpin)</a>: Library Is All You Need  It will bring efficiency and consistency, scalability, adaptability, and &#34;interoperability&#34; in language models globally  CC: @karpathy  ↘️ Quoting Ehsan Azari (@ehsanx...</li><li><a href="https://duckduckgo.com/?q=summarize+youtube+video">summarize youtube video at DuckDuckGo</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1220640058856439880)** (67 messages🔥🔥): 

- **Custom GPT Sidebar Limits**: A member expressed concerns about a change in the **Custom GPT pin limit** on the sidebar, which seemed reduced from 10 to 4 without prior notice. There was no mention of a workaround or solution provided.
  
- **Seeking Keybind for Shared GPTs Access**: A member asked if there was a **keyboard shortcut** to access "MyGPTs -> Shared with ME", and another member provided a suggestion to use user scripts with a browser extension like `tampermonkey` to create a custom solution.

- **Email Verification Request**: A user sought confirmation on the authenticity of an **email purportedly from OpenAI**; another user suggested checking the **mail headers** for verification.

- **GPT-4 with Vision Capability**: In a discussion about the abilities of GPT models to read images, it was affirmed that **GPT-4 with Vision** is capable of this, with a link to the official OpenAI documentation provided for reference: [Official OpenAI Documentation on Vision](https://platform.openai.com/docs/guides/vision).

- **Clarification on the Discontinuation of Plugins**: Addressing an inquiry about accessing plugins, a member clarified that the **ChatGPT plugins beta** is being wound down with a helpful URL to the official announcement: [Winding Down the ChatGPT Plugins Beta](https://help.openai.com/en/articles/8988022-winding-down-the-chatgpt-plugins-beta).
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1220896722553671691)** (61 messages🔥🔥): 

- **Vision's Recognition of the Disabled**: A member expressed difficulty in getting recognition from Vision for images that include themselves, specifically for use in robotics and personal grooming assistance. The suggestion was made to explore solutions carefully due to privacy concerns.
- **Enhancing Chapter Writing with GPT**: In an exchange about improving writing with Chat-GPT, a member sought advice on prompting for adding subsections without rewriting an entire chapter. Tips included being more specific in the prompt, such as specifying where to insert new content.
- **Prompt Engineering for Better Code**: A member shared a detailed multi-part prompt designed to enhance the quality of coding tasks executed by GPT, focusing on meticulously crafted steps emphasizing coding practices. Other members engaged, discussing the merits of the approach and offering to refine it and provide their own versions.
- **Migration Issues with OpenAI SDK**: A user sought assistance with an error encountered from an OpenAI SDK update, which deprecated the `.Completion` endpoint. Another member directed them to a server channel specifically for questions related to migration issues.
- **Striving for Show, Not Tell in AI Writing**: Members discussed strategies to prompt Chat-GPT to show actions in storytelling instead of telling, aiming to improve narrative quality. Advise was shared on reformulating prompts to guide Chat-GPT towards desired writing styles.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/openai/openai-python">GitHub - openai/openai-python: The official Python library for the OpenAI API</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://github.com/openai/openai-python/discussions/742```">v1.0.0 Migration Guide · openai/openai-python · Discussion #742</a>: We have released a new major version of our SDK, and we recommend upgrading promptly. It&#39;s a total rewrite of the library, so many things have changed, but we&#39;ve made upgrading easy with a cod...
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1220896722553671691)** (61 messages🔥🔥): 

- **Exploring Vision's Accessibility**: A user sought advice on how to make the Vision model recognize them as a disabled person for use cases like vision assistance for personal grooming. Despite sensitivity around discussing potential solutions, suggestions to write up the issue for a Discord suggestions channel and a link to a previous related post were offered.

- **Enriching GPT-4 Generated Book Content**: A member was seeking help on how to instruct GPT-4 to add subsections to a chapter without rewriting the entire content. Suggested strategies included using section numbering for better organization and clarity during prompt construction, and starting a new conversation detailing the versions for a consolidated output.

- **Improving GPT Coding Task Responses**: Users discussed strategies to improve GPT's code output, with one sharing a detailed prompt that instructs the model for better performance during coding tasks. Suggestions for using technical process names for better engagement and customized JSON instructions as prompts for coding tasks were made.

- **Crafting a 'Kawaii-bubbly' AI Personality**: A user requested assistance in creating prompts to give GPT a 'kawaii-bubbly' personality for writing an animator's social media bio. Although it was challenging to create prompts for a personality the user couldn't emulate, examples of attempted prompts were provided.

- **Increasing Quality of Hypothesis Paragraphs with ChatGPT**: A member needed support to avoid generic statements and produce hypothesis paragraphs laden with theories and proofs from experts. Advice was given to communicate directly with ChatGPT as one would in a normal conversation and to specify the inclusion of certain elements for a high-quality output.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/openai/openai-python">GitHub - openai/openai-python: The official Python library for the OpenAI API</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://github.com/openai/openai-python/discussions/742```">v1.0.0 Migration Guide · openai/openai-python · Discussion #742</a>: We have released a new major version of our SDK, and we recommend upgrading promptly. It&#39;s a total rewrite of the library, so many things have changed, but we&#39;ve made upgrading easy with a cod...
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1220621362864062504)** (242 messages🔥🔥): 

- **Questions About AI Art Prompt Help**: Someone asked for recommendations on good places to get help making prompts for AI art. No specific solutions were provided in the messages.
- **Blenderbot's Consistent Character**: A user discussed the benefits of Blenderbot's consistent character in contrast to chatbots that are self-aware of being AI, such as ChatGPT. They noted Blenderbot might claim to be a housewife or a schoolteacher but remains "in character", unlike some other models.
- **Execution Speed of Multiplication vs. Conditional Checking on GPUs**: A member inquired about the performance differences between performing a multiplication operation (a*b=c) versus a conditional check (if(a==0){}) on a GPU. Another user suggested that shader compilers do a lot for efficiency, and someone recommended looking into works by 'iq' for more information.
- **Linguistically Diverse Prompt for ChatGPT**: A detailed and creatively complex prompt was requested for ChatGPT that included various styles from authors and principles from well-known personalities, though one user simply responded with "Bloody hell" to the complexity.
- **TensorRT-LLM vs. ExLLama v2 for GPU Inference**: The discussion revolved around different methods for running large language model (LLM) inferences on GPUs, citing that TensorRT-LLM might be suitable for single-batch inference, while libraries like exLLama v2 are optimized for single-user speed. For serving many simultaneous users, other solutions were recommended like vllm or tgi.
- **Quantizing with GGML**: A user asked if ggml supports quantization for all models or only generative ones, and another member responded that ggml does not support all models but includes various language and multimodal models and recommended using specific language model files like llama.cpp for faster performance.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rubiks.ai/">Rubik's AI - Waitlist</a>: no description found</li><li><a href="https://huggingface.co/CopyleftCultivars/Gemma2B-NaturalFarmerV1">CopyleftCultivars/Gemma2B-NaturalFarmerV1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/fffiloni/coqui-bark-voice-cloning-docker">Coqui Bark Voice Cloning Docker - a Hugging Face Space by fffiloni</a>: no description found</li><li><a href="https://hf-mirror.com/">HF-Mirror - Huggingface 镜像站</a>: no description found</li><li><a href="https://discuss.huggingface.co/">Hugging Face Forums</a>: Community Discussion, powered by Hugging Face &lt;3</li><li><a href="https://huggingface.co/welcome">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/v2.1.0/en/process#split-long-examples)">Process</a>: no description found</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://github.com/mistralai-sf24/hackathon">GitHub - mistralai-sf24/hackathon</a>: Contribute to mistralai-sf24/hackathon development by creating an account on GitHub.</li><li><a href="https://github.com/turboderp/exllamav2">GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs</a>: A fast inference library for running LLMs locally on modern consumer-class GPUs - turboderp/exllamav2</li><li><a href="https://github.com/davidberenstein1957/fast-sentence-transformers">GitHub - davidberenstein1957/fast-sentence-transformers: This repository, called fast sentence transformers, contains code to run 5X faster sentence transformers using tools like quantization and ONNX.</a>: This repository, called fast sentence transformers, contains code to run 5X faster sentence transformers using tools like quantization and ONNX. - davidberenstein1957/fast-sentence-transformers</li><li><a href="https://github.com/coqui-ai/TTS">GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production</a>: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production - coqui-ai/TTS</li><li><a href="https://github.com/suno-ai/bark?tab=readme-ov-file#-installation">GitHub - suno-ai/bark: 🔊 Text-Prompted Generative Audio Model</a>: 🔊 Text-Prompted Generative Audio Model. Contribute to suno-ai/bark development by creating an account on GitHub.</li><li><a href="https://carrcenter.hks.harvard.edu/news/dont-talk-people-theyre-chatbots">Don&#039;t Talk to People Like They&#039;re Chatbots</a>: &quot;AI could make our human interactions blander, more biased—or ruder,&quot; write Carr Center faculty Bruce Schneier and Technology and Human Rights Fellow Albert Fox Cahn in The Atlantic.</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found</li><li><a href="https://github.com/vgthengane/Awesome-Mamba-in-Vision">GitHub - vgthengane/Awesome-Mamba-in-Vision: List of papers related to State Space Models (Mamba) in Vision.</a>: List of papers related to State Space Models (Mamba) in Vision. - vgthengane/Awesome-Mamba-in-Vision
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1221453209579814952)** (9 messages🔥): 

- **Intrigue for an Unspecified Tool**: A member expressed excitement about a tool but did not provide further details or a link to it.

- **A Call for Help with HuggingFace**: A user sought assistance with the **qra-13b model** from HuggingFace, with a particular mention of Poland.

- **Model Conversion Endeavors**: A member has been working on converting the **GLiNER model** from PyTorch to the *Candle* (Rust), exploring quantization techniques and learning about the Candle library.

- **Perks of Model Conversion to Rust**: In a conversation about the advantages of converting models to Rust, a member mentioned **less dependencies**, suitability for production deployment, and improved speed, though their current implementation wasn't faster.

- **Rust-Based Models and GPU Compatibility**: It was confirmed that models converted to Rust using the Candle library are **compatible with GPUs**.
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1220631213903970364)** (12 messages🔥): 

- **Deep Dive into Visual Processing**: A YouTube video titled ["Understanding early visual processing mechanisms by the principle of efficient encoding"](https://www.youtube.com/watch?v=Ed9otQAmEF4) discusses early visual processing in biological vision.
- **Exploring Superlet Transform for Audio Analysis**: A new method called *Superlet Transform* is highlighted as an improvement for real-time audio analysis, with its effectiveness demonstrated in a [Nature article](https://doi.org/10.1038/s41467-020-20539-9) and complementary [benchmarks provided](https://doi.org/10.1038/s41598-022-22055-w) in an article.
- **Language Agent Tree Search with Langchain**: An article on Medium discusses the potential revolution in decision-making using language models with Langchain, potentially changing how language agents approach problem-solving. The article is available on [Medium](https://medium.com/ai-advances/language-agent-tree-search-with-langchain-revolutionizing-decision-making-with-language-models-a46c991397f1).
- **Valuable Insights from CivitAI**: An assortment of articles and guides on Stable Diffusion, including tips, tricks, and insights for both novices and intermediates, can be found collected by a member on [CivitAI](https://civitai.com/articles/2054).
- **The Significance of Data**: A member shared an [arXiv paper](https://arxiv.org/pdf/2212.03533.pdf) that underscores the import of data and its potential to be a critical factor in a given context.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ieeexplore.ieee.org/document/10333889">Exploring Lightweight Federated Learning for Distributed Load Forecasting</a>: Federated Learning (FL) is a distributed learning scheme that enables deep learning to be applied to sensitive data streams and applications in a privacy-preserving manner. This paper focuses on the u...</li><li><a href="https://civitai.com/articles/2054">A curated list of Stable Diffusion Tips, Tricks, and Guides | Civitai</a>: Stable Diffusion: https://stable-diffusion-art.com/samplers https://civitai.com/articles/983/insights-for-intermediates-how-to-craft-the-images-you...</li><li><a href="https://arxiv.org/abs/2207.04343">Explaining Chest X-ray Pathologies in Natural Language</a>: Most deep learning algorithms lack explanations for their predictions, which limits their deployment in clinical practice. Approaches to improve explainability, especially in medical imaging, have oft...</li><li><a href="https://www.youtube.com/watch?v=Ed9otQAmEF4">Understanding early visual processing mechanisms by the principle of efficient encoding</a>: This is lecture 2 of the five lectures at CVPR 2022 tutorial &quot;A post-Marrian computational overview of how biological (human) vision works&quot;, on June 19, 2022...</li><li><a href="https://doi.org/10.1038/s41467-020-20539-9">Time-frequency super-resolution with superlets - Nature Communications</a>: Identifying the frequency, temporal location, duration, and amplitude of finite oscillation packets in neurophysiological signals with high precision is challenging. The authors present a method based...</li><li><a href="https://doi.org/10.1038/s41598-022-22055-w">Super-resolved time–frequency measurements of coupled phonon dynamics in a 2D quantum material - Scientific Reports</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1220731707842629632)** (19 messages🔥): 

- **Federated Learning Goes Energy-Efficient**: A GitHub project on *Exploring Lightweight Federated Learning for load forecasting* is shared, aiming to tackle load forecasting using clustering and sequential DNN methods. The project is accessible at [Exploring-Lightweight-Federated-Learning-for-load-forecasting on GitHub](https://github.com/ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting).
  
- **Stable Diffusion Resources Compiled**: Member shares multiple links about **Stable Diffusion** including guides for samplers, insights, and tools, with standalone articles like "*how-to craft the images you want with a1111*" and a video guide on "**video2video**". The links, such as [civitai.com](https://civitai.com/articles/2054), provide valuable resources for Stable Diffusion users.

- **Anki Made Easy with AnkiForge**: A new app called AnkiForge is announced, which allows users to generate Anki flashcards from text notes and future support for audio files. The app can be tried at [AnkiForge](https://ankiforge.onrender.com/).
  
- **Localization and Trust in Fact-Checking**: A new research paper discussing "Evidence Attribution of LLM Output Through Knowledge Graphs" for the purpose of verifying LLM outputs is introduced, exploring the trust and validation mechanism in the era of misinformation. The paper focuses on a fine-grained evidence attribution method and is available on [arXiv](https://arxiv.org/abs/2403.09724).

- **Exploring AI's Limits in Recurrent Neural Notes Newsletter**: The latest issue of **Recurrent Neural Notes** discusses the potential limits of AI and includes in-depth articles. Discover the newsletter's insights and thoughts on AI's future at [Recurrent Neural Notes on Substack](https://open.substack.com/pub/thernn/p/rnn-7-the-real-limits-of-ai?r=kxtnk&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true).

- **German Learning with GPT Bot Hans**: An announcement introduces *Hans*, a GPT-powered German language learning tool available in the GPT store, aimed to help users improve their German language skills. Check out Hans at [Hans 🥨 in the GPT store](https://chat.openai.com/g/g-mP8tCHgOc-hans).

- **Video Explainers for LLM Jargons**: A series of videos explaining various LLM (Large Language Models) concepts like Multi Query Attention, Sliding Window Attention and more are shared to help demystify the complex world of language models. The educational series is available on [YouTube](https://www.youtube.com/playlist?list=PLfSv7CK7EjD2fC9S6MAKRNDgTSCYgdGgz).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://civitai.com/articles/2054">A curated list of Stable Diffusion Tips, Tricks, and Guides | Civitai</a>: Stable Diffusion: https://stable-diffusion-art.com/samplers https://civitai.com/articles/983/insights-for-intermediates-how-to-craft-the-images-you...</li><li><a href="https://www.youtube.com/playlist?list=PLfSv7CK7EjD2fC9S6MAKRNDgTSCYgdGgz">LLM Jargons Explained</a>: Welcome to the &quot;LLM Jargons Explained&quot; series, where I demystify the complex world of language models and decoding techniques. Whether you&#39;re a language mode...</li><li><a href="https://arxiv.org/abs/2403.09724">ClaimVer: Explainable Claim-Level Verification and Evidence Attribution of Text Through Knowledge Graphs</a>: In the midst of widespread misinformation and disinformation through social media and the proliferation of AI-generated texts, it has become increasingly difficult for people to validate and trust inf...</li><li><a href="https://huggingface.co/spaces/Tonic/Command-R">Command-R - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://github.com/ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting">GitHub - ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting: Federated Learning on Energy Dataset for load forecasting using clustering and sequential DNN methods</a>: Federated Learning on Energy Dataset for load forecasting using clustering and sequential DNN methods - ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting</li><li><a href="https://open.substack.com/pub/thernn/p/rnn-7-the-real-limits-of-ai?r=kxtnk&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true">RNN #9 - The Real Limits of AI</a>: Exploring the Computational Boundaries of Neural Networks</li><li><a href="https://youtu.be/YpuRcmPnSTM">What you just said</a>: From the movie, Billy Madison</li><li><a href="https://ankiforge.onrender.com/">Anki Forge</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1220795671427285084)** (48 messages🔥): 

- **Insights Into Obesity Unveiled**: A Kaggle notebook offering an EDA Exploration and Visualisation of obesity data has been shared, promising insights into factors that influence obesity across demographics and lifestyles. The notebook can be found at [Deciphering Obesity Trends: An In-depth EDA](https://www.kaggle.com/code/muhammadibrahimqasmi/deciphering-obesity-trends-an-in-depth-eda).

- **Upcoming Event Alert**: A reminder about an imminent meeting was posted, quickly followed up by a highlight of the discussed paper on a **Hyper Z.Z.W operator to replace transformers**. The paper intends to tackle challenges in attention-based mechanisms and can be read [here](https://arxiv.org/pdf/2401.17948.pdf).

- **The Quest for 1 Million Context**: Conversation touched on the difficulty of achieving 1 million context using vanilla attention and speculated on companies' technology, particularly Google, and their potential proprietary advancements in computation efficiency.

- **Relevance Matters for Chatbot Responses**: A member reflected upon the impressive capabilities of chat GPT when asked highly relevant and important questions, stating the model's responses align with the significance of the inquiry posed.

- **Catch the Recording of the Recent Meeting**: For those who missed the reading group's event, a recording was mentioned and eventually linked, hosting a presentation on next-generation network architecture and the **Hyper Z.Z.W Operator**. Interested parties can watch the presentation at [Hugging Face Reading Group 16: Hyper ZZ.W Operator Terminator](https://youtu.be/urgLoVPj1P8).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/code/muhammadibrahimqasmi/deciphering-obesity-trends-an-in-depth-eda">Deciphering Obesity Trends &#x1F4C9;: An In-depth EDA &#x1F4CA;</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from multiple data sources</li><li><a href="https://youtu.be/urgLoVPj1P8">Hugging Face Reading Group 16: HyperZ⋅Z⋅W Operator Terminator</a>: Presenter: Harvie Zhang who is also the author of this work. For this meeting unfortunately there was a bit of moderation issue</li><li><a href="https://youtu.be/XiSPWW-3uNY">Watchtower</a>: Provided to YouTube by TuneCoreWatchtower · Michael Salvatori, Skye Lewin, Rotem Moav &amp; Pieter SchlosserDestiny 2: Forsaken (Original Soundtrack)℗ 2018 Bungi...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1221482604923064340)** (1 messages): 

- **Experimental Tool for Memory Requirements**: An experimental tool has been released to gauge the **inference-time memory requirements of a `DiffusionPipeline`**. The tool is available for testing and feedback is welcomed on the [GitHub discussion page](https://github.com/huggingface/diffusers/discussions/7434).

**Link mentioned**: <a href="https://github.com/huggingface/diffusers/discussions/7434">Calculate the component-wise memory of `DiffusionPipeline` checkpoint · huggingface/diffusers · Discussion #7434</a>: We shipped a Hugging Face Space that lets you calculate the memory requirements of a DiffusionPipeline checkpoint given a torch_dtype: https://huggingface.co/docs/diffusers/main/en/using-diffusers/...

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1220762227687293088)** (21 messages🔥): 

- **SegGPT Joins the HuggingFace Hub**: HuggingFace introduces [SegGPT](https://huggingface.co/docs/transformers/main/en/model_doc/seggpt), a model that can be trained for any image-to-image task. It was highlighted in the paper *SegGPT: Segmenting Everything In Context* and has shown impressive one-shot segmentation results on datasets like COCO-20 and FSS-1000.

- **Cracking Diffusion Models**: A member expressed that, after engaging with several blogs and coding along with tutorials, they’ve improved their understanding of diffusion models. Eager to contribute to open-source diffusion projects, they ponder whether to delve deeper into coding or explore different tasks and fine-tuning techniques effective in diffusion models.

- **Puzzles With Vision Model Channel Inputs**: A challenge was raised regarding vision models typically accepting only 3-channel images. It was pointed out that 3-channel defaults are common due to the prevalence of such data in benchmark datasets, though **BridgeTower** was mentioned as not accommodating single-channel images despite configuration attempts.

- **Fusing Image Features with LLM**: Responding to an inquiry on merging text and image generation models, **BLIP-2** was recommended as a resource. The associated [BLIP-2 paper](https://arxiv.org/abs/2301.12597) outlines an approach where vision-language representations are learned by training an intermediary transformer connecting pre-trained image encoders to language models.

- **BLIP-2 Resources Shared**: Further resources on BLIP-2, including the [transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/blip-2), were shared to assist with understanding the fine-tuning process. BLIP instruct, an instruction-tuned variant, was noted to potentially yield better performance than standard BLIP-2.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2301.12597">BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models</a>: The cost of vision-and-language pre-training has become increasingly prohibitive due to end-to-end training of large-scale models. This paper proposes BLIP-2, a generic and efficient pre-training stra...</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/blip-2">BLIP-2</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/seggpt">SegGPT</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1220621609409445919)** (24 messages🔥): 

- **HuggingFace Trainer Troubles**: A user is experiencing issues with HuggingFace's **Trainer** class not recognizing the 'accelerate' package despite being installed. Various troubleshooting steps are discussed, including upgrading packages, clearing caches, and changing the import order of libraries.
- **SentenceTransformer Function Fails Offline**: Several users report problems with **SentenceTransformers** not accepting local directories in offline environments, contrary to its functionality with `transformers.AutoModel.from_pretrained`. Requests for validation of SentenceTransformers' offline capabilities are made.
- **Quest for NEET/JEE Dataset**: A user is seeking datasets with questions, answers, and explanations from previous years' NEET/JEE exams to train a MCQ answer generator using GPT-4, with concerns about the potential margin of error being discussed.
- **Embedding Quantization Breakthrough**: 🤗 HuggingFace announced a new **Embedding Quantization** method for **Sentence Transformers** resulting in massive improvements in search speeds and reductions in memory, storage, and cost, all while preserving retrieval performance. Details and a demo can be found at the announcement [space](https://huggingface.co/spaces/sentence-transformers/quantized-retrieval) and the in-depth [blog post](https://huggingface.co/blog/embedding-quantization).
- **Inference API Summary Length Clarification**: A user queries about controlling the length of summaries produced by the **facebook/bart-large-cnn** model in a MERN application. It's explained that the `max_length` parameter determines the maximum sentence length in input batches.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/awnr/Mistral-7B-v0.1-half-naive-A">awnr/Mistral-7B-v0.1-half-naive-A · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/sentence-transformers/quantized-retrieval">Quantized Retrieval - a Hugging Face Space by sentence-transformers</a>: no description found</li><li><a href="https://huggingface.co/blog/embedding-quantization">Binary and Scalar Embedding Quantization for Significantly Faster &amp; Cheaper Retrieval</a>: no description found</li><li><a href="https://sbert.net/examples/applications/embedding-quantization/README.html">Embedding Quantization &mdash; Sentence-Transformers  documentation</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1220658219781722113)** (31 messages🔥): 

- **All-MiniLM-L6-v2 Model Inquiry**: A member expressed interest in using the **all-MiniLM-L6-v2 model** for their dataset but needed guidance on how to download and train it. They requested someone to direct message them for assistance.
- **Background Addition for Images on Hugging Face**: One member asked for a pretrained model on Hugging Face capable of adding backgrounds to images. Another member suggested the use of **RMBG** for background removal and the application of filters for smoothing with tools like **OpenGL, PyGLET, Kivy, and GIMP**.
- **Stylizing Images with SDXL**: A question was raised on how to stylize an input image into watercolor or other effects and how to make images seamless for creating repeating patterns using **SDXL**.
- **Advice Sought on Continuing Diffusion Studies**: A member who studied and wrote code about diffusion models from various resources, including YouTube and Medium articles, asked for advice on further steps, whether to continue coding, study diffusion techniques, or dive into fine-tuning, with the long-term goal of contributing to open-source diffusion model repositories.
- **Learning Resources for Fine-Tuning Diffusion Models**: Two members had an exchange where one asked for resources to learn fine-tuning diffusion models on personal images, and the other pointed to [Hugging Face documentation](https://huggingface.co/docs/diffusers/main/en/training/overview) and suggested trying to fix a simple open source issue marked with the “Good first issue” label, complemented by examining previously merged PRs.

**Link mentioned**: <a href="https://huggingface.co/docs/diffusers/main/en/conceptual/contribution#:~:text=Fix%20a%20simple%20issue%2C%20marked%20by%20the%20%E2%80%9CGood%20first%20issue%E2%80%9D%20label%2C%20see%20here.)?">How to contribute to Diffusers 🧨</a>: no description found

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1220766917199921274)** (8 messages🔥): 

- **Streamlining Human-LlamaIndex Interaction**: There's a new template that allows humans to only interact with **LlamaIndex's agents** when intervention is needed, aiming for less intrusive user experiences. Here's a sneak peek at the [Twitter post](https://t.co/Z16QPCWFmG).
- **Custom LLMs Join the LlamaIndex Fold**: Learn how to integrate your own custom Language Models (LLMs) into **LlamaIndex**. Leonie Monigatti explains the process in detail on [LinkedIn](https://t.co/DBjXGkLFkg).
- **Creating a RAG Agent for PDFs**: Ashish S. crafted a tutorial on building an agentic RAG flow over PDFs that includes **LlamaParse** for extracting text and tables. The complete guide is shown in this [Tweet](https://t.co/vIANM2Byel).
- **Building RAG and Agents with MistralAI**: A comprehensive resource compilation for using **LlamaIndex**, **MistralAI**, and optionally **LlamaParse** to build advanced RAG and agents has been announced. Access the resources [here](https://t.co/5zPWnjPCth).
- **Python Documentation Upgrade for LlamaIndex**: The new **LlamaIndex Python documentation** has been revamped to prominently feature example notebooks, improved search functionality with previews and term highlights, and streamlined API information. Check out the improved docs in this [Twitter announcement](https://t.co/FAuBj5gnCC).
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1220604644951330866)** (296 messages🔥🔥): 

- **Discussion on Bot and AI Tools Integration**: Users discussed integrating different AI tools like Merlin API and LocalAI with LlamaIndex, where LocalAI can be used with LlamaIndex's `OpenAILike` method for interaction, as detailed in their [documentation and LocalAI setup guide](https://github.com/mudler/LocalAI).
  
- **Evaluation Logic Explanation Requested**: A user sought explanation for LlamaIndex's evaluation code logic, involving various evaluators such as `CorrectnessEvaluator` and `SemanticSimilarityEvaluator`. Another user, whitefang_jr, provided clarity by identifying the pathway taken by input through different evaluators, with links to the BatchEvalRunner [documentation](https://docs.llamaindex.ai/en/stable/examples/evaluation/batch_eval/).

- **Inquiry About Mixed Messaging in Documentation**: A user expressed frustration over conflicting information across LlamaIndex's documentation, citing specific examples such as guides on using tools that don't match implementation. A discussion followed to clarify confusion, with others acknowledging a need for updated notebooks and docs post v0.10 updates.

- **Request for Multi-Agent Chatbot Example**: A user asked for examples on building multi-agent chatbots using LlamaIndex to accomplish sequential tasks like SQL queries, summarization, and Q&A. Teemu2454 provided a link to an example of multi-document agents ([source](https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents-v1/)) which could be a relevant starting point.

- **Turning Python Functions into LlamaIndex Tools**: Inquiring about functionality similar to OpenAI Assistants with tools, a user asked how to convert a Python function into a tool for LlamaIndex. Cheesyfishes provided code using `FunctionTool.from_defaults(fn=add)` and a link to the associated [source code on GitHub](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/extractors/llama-index-extractors-entity/llama_index/extractors/entity/base.py).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://api.getmerlin.in/#pricing">Merlin API Platform</a>: Integrate LLMs Into Your Production Apps In Minutes.</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html">Redirecting...</a>: no description found</li><li><a href="https://colab.research.google.com/drive/13NJEyhKWT7xdJFAJ6nB8mq-fk22UVDKa?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://llamahub.ai/l/tools/llama-index-tools-salesforce?from=">no title found</a>: no description found</li><li><a href="https://pretty-sodium-5e0.notion.site/llama-index-tools-salesforce-cdb97eca825c47bd8811b209035dae0d">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/">LlamaIndex - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/">LlamaIndex - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/">Using Documents - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/supporting_modules/service_context_migration.html">Redirecting...</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/">Qdrant Vector Store - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/agent_runner_rag/?h=tools#define-toolset">Controllable Agents for RAG - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/">Vector Stores - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/localai/#llamaindex-interaction">LocalAI - LlamaIndex</a>: no description found</li><li><a href="https://www.llamaindex.ai/blog/llamaindex-v0-10-838e735948f8">LlamaIndex v0.10 — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents-v1/">Multi-Document Agents (V1) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm/?h=hugg">HuggingFace LLM - StableLM - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/usecases/10q_sub_question/">10Q Analysis - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/indices/vector/#llama_index.core.indices.VectorStoreIndex">Vector - LlamaIndex</a>: no description found</li><li><a href="https://codespaces.new/Bloom-Assistant/api.getbloom.ai/tree/codespacers">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/run-llama/llama_index/blob/f5263896121721de1051ce58338a1e0ea6950ca7/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py#L704">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py at f5263896121721de1051ce58338a1e0ea6950ca7 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/be63bae53227f1360472477eb2afa993791c09ce/llama-index-core/llama_index/core/objects/base.py#L47-L49">llama_index/llama-index-core/llama_index/core/objects/base.py at be63bae53227f1360472477eb2afa993791c09ce · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/extractors/llama-index-extractors-entity/llama_index/extractors/entity/base.py">llama_index/llama-index-integrations/extractors/llama-index-extractors-entity/llama_index/extractors/entity/base.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/be63bae53227f1360472477eb2afa993791c09ce/llama-index-packs/llama-index-packs-snowflake-query-engine/llama_index/packs/snowflake_query_engine/base.py#L44">llama_index/llama-index-packs/llama-index-packs-snowflake-query-engine/llama_index/packs/snowflake_query_engine/base.py at be63bae53227f1360472477eb2afa993791c09ce · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://llamahub.ai/l/tools/llama-index-tools-bing-search?from=all">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/tools/google/">Google - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/">API Reference - LlamaIndex</a>: no description found</li><li><a href="https://work.caltech.edu/telecourse">Learning From Data - Online Course (MOOC)</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/storing/chat_stores.html">Redirecting...</a>: no description found</li><li><a href="https://llamahub.ai/l/llama-packs/llama-index-packs-snowflake-query-engine?from=">no title found</a>: no description found</li><li><a href="https://growsmethod.com/practices/TracerBullets.html">Tracer Bullet Development</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/batch_eval/">BatchEvalRunner - Running Multiple Evaluations - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/">Migrating from ServiceContext to Settings - LlamaIndex</a>: no description found</li><li><a href="https://github.com/UKPLab/sentence-transformers/releases/tag/v2.6.0">Release v2.6.0 - Embedding Quantization, GISTEmbedLoss · UKPLab/sentence-transformers</a>: This release brings embedding quantization: a way to heavily speed up retrieval &amp; other tasks, and a new powerful loss function: GISTEmbedLoss. Install this version with pip install sentence-trans...</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/storing/chat_stores/">Chat Stores - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/indices/">Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/pull/12187">fix async streaming by logan-markewich · Pull Request #12187 · run-llama/llama_index</a>: Need to ensure lazily-declared queue/async stuff is actually instantiated before accessing Fixes #12180</li><li><a href="https://www.youtube.com/watch?v=QCZU9nCb-AM">Cria Demo (Thu Mar 14 2024)</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/issues/12180">[Bug]: AttributeError: &#39;NoneType&#39; object has no attribute &#39;wait&#39; · Issue #12180 · run-llama/llama_index</a>: Bug Description Async Streaming Chat example: https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/#async-streaming-chat produces exception: AttributeError: &#39;NoneType&#39; object has n...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/bedrock/?h=bedrock">Bedrock Embeddings - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/bedrock/?h=bedrock">Bedrock - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/sagemaker_endpoint_llm/">Interacting with LLM deployed in Amazon SageMaker Endpoint with LlamaIndex - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/semantic_similarity_eval/#embedding-similarity-evaluator">Embedding Similarity Evaluator - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/faithfulness_eval/">Faithfulness Evaluator - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1220707548311715891)** (164 messages🔥🔥): 

- **Searching for Video Processing Tool Like Whisper**: A user enquired about a tool comparable to Whisper but for video processing and mentioned that it possibly leveraged VLM for scene evaluation and was potentially open source. Multiple suggestions were made, including [Video Mamba](https://huggingface.co/blog/vladbogo/video-mamba), Twelve Labs, and video intelligence service [videodb.io](https://videodb.io).

- **OpenAI's Sora Wows Artists**: The OpenAI blog shared first impressions of Sora, revealing strong interest and endorsements from creative professionals. Examples of artist work were discussed, displaying how Sora enables the creation of both realistic and surreal imagery.

- **Google's AI Studio vs. Vertex AI Confusion**: Discussions revolved around the differences and usage of Google's AI Studio versus Vertex AI in serving up models like Gemini, with AI Studio starting to roll out 1 million token context APIs and comparisons made to the OpenAI API in terms of ease of use.

- **AI Wearables on a Roll**: Chat snippets focused on the trend of open-source AI wearables, including the $200 ALOHA project, and discussion on whether such products are fully local. Pre-orders for [Compass](https://x.com/itsmartynask/status/1771890769865187648), another AI wearable, began, with plans for shipping to start the following week.

- **Efficiency in Large Language Models**: LLMLingua by Microsoft was shared as a tool to compress LLM prompts and KV-Cache, potentially achieving up to 20x compression with minimal performance loss. It was suggested that while optimizing costs is essential, it's also crucial not to over-optimize too early and instead focus on delivering value.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.instalora.xyz/">InstaLoRA - Instant LoRA Generator</a>: Generate your LoRA in seconds</li><li><a href="https://videodb.io">VideoDB</a>: Build intelligent applications on all types of video with 2 simple lines of code. Built by developers. For developers.</li><li><a href="https://huggingface.co/blog/vladbogo/video-mamba">VideoMamba: State Space Model for Efficient Video Understanding</a>: no description found</li><li><a href="https://x.com/kodjima33/status/1772011777066442819">Tweet from Nik Shevchenko (@kodjima33)</a>: Today I saw the launch of another &#34;open-source&#34; AI wearable that has not published anything just to charge you 5x the cost  At @MistralAI x @cerebral_valley hackathon in @SHACK15sf we built FR...</li><li><a href="https://www.forbes.com/sites/kenrickcai/2024/03/22/stability-ai-founder-emad-mostaque-plans-to-resign-as-ceo-sources-say/?sh=703ce1225239">Stability AI Founder Emad Mostaque Plans To Resign As CEO, Sources Say</a>: Mostaque has told a number of people close to him that he plans to step down as chief executive of the once buzzy generative AI startup known for Stable Diffusion.</li><li><a href="https://openai.com/blog/sora-first-impressions">Sora: First Impressions</a>: We have gained valuable feedback from the creative community, helping us to improve our model.</li><li><a href="https://www.evenuplaw.com/">EvenUp</a>: no description found</li><li><a href="https://tenor.com/view/dj-khaled-another-one-one-more-time-gif-4816107">Another One GIF - Dj Khaled Another One One More Time - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://stability.ai/news/stabilityai-announcement">Stability AI Announcement &mdash; Stability AI</a>: Earlier today, Emad Mostaque resigned from his role as CEO of Stability AI and from his position on the Board of Directors of the company to pursue decentralized AI.  The Board of Directors has appoin...</li><li><a href="https://x.com/xiangyue96/status/1771898843275067420?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Xiang Yue (@xiangyue96)</a>: @MistralAI just released their v0.2 Base😱. @WenhuChen and I quickly evaluated a few benchmarks using the OpenCompass evaluation package. It seems that the capability dropped a little bit on nearly al...</li><li><a href="https://www.twelvelabs.io/">Multimodal AI that understands videos like humans</a>: Bring human-like video understanding to any application, whether you have terabytes or petabytes of video</li><li><a href="https://x.com/omooretweets/status/1771960892810240333?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Olivia Moore (@omooretweets)</a>: Why is ChatGPT being adapted for so many use cases?   (1) It has distribution; and (2) It combines text, image, voice to be a full partner.  But, it&#39;s limited in UI and workflow. IMO, this is part...</li><li><a href="https://x.com/itsmartynask/status/1771890769865187648?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from mkrupskis (@ItsMartynasK)</a>: Today I am launching pre-orders for Compass - a $99 open-source guide.  - 30 hours battery life - learns by transcribing your conversations - revisit important moments in your life - shipping first or...</li><li><a href="https://x.com/matpagliardini/status/1771168258856501564?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Matteo Pagliardini (@MatPagliardini)</a>: A tweak in the architecture of #Transformers can significantly boost accuracy!  With direct access to all previous blocks’ outputs, a 48-block #DenseFormer outperforms a 72-block Transformer, with fas...</li><li><a href="https://www.latent.space/p/feb-2024">The Unbundling of ChatGPT (Feb 2024 Recap)</a>: Peak ChatGPT? Also: our usual highest-signal recap of top items for the AI Engineer from Feb 2024!</li><li><a href="https://x.com/ai_for_success/status/1771932897915650371?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from AshutoshShrivastava (@ai_for_success)</a>: AI-powered devices: wearables/personal assistants  Humane AI Pin: $699 Rabbit R1: $199 Open Interpreter 01 Light: $99 and open-sourced Compass: $99 and open-sourced  This is just the start... We&#39;r...</li><li><a href="https://x.com/swyx/status/1772305930836918656?s=20">Tweet from swyx (@swyx)</a>: 🆕 The Unbundling of ChatGPT    https://latent.space/p/feb-2024   A whole year has passed with ~0 growth in ChatGPT user numbers. Instead, users are exploring a whole host of verticalized players for ...</li><li><a href="https://x.com/mattshumer_/status/1771204395285246215?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Matt Shumer (@mattshumer_)</a>: Introducing `claude-investor` 📈  The first Claude 3 investment analyst agent.  Just provide an industry, and it will: - Find financial data/news for key companies - Analyze sentiment/trends for each ...</li><li><a href="https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#performance">Our next-generation model: Gemini 1.5</a>: Gemini 1.5 delivers dramatically enhanced performance, with a breakthrough in long\u002Dcontext understanding across modalities.</li><li><a href="https://x.com/shiringhaffary/status/1771210619485659183?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Shirin Ghaffary (@shiringhaffary)</a>: 🚨 NEW  OpenAI goes to Hollywood   Company has mtgs next week w/ Hollywood studios, media execs, talent agencies encouraging them to use Sora. Some filmmakers already have access. COO Brad Lightcap le...</li><li><a href="https://x.com/clementdelangue/status/1771395468959813922?s=46">Tweet from clem 🤗 (@ClementDelangue)</a>: Should we acquire Stability and open-source SD3?</li><li><a href="https://x.com/amanrsanger/status/1771590523046051947?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Aman Sanger (@amanrsanger)</a>: &#34;Token Counts&#34; for long context models are a deceiving measure of content length. For code:  100K Claude Tokens ~ 85K gpt-4 Tokens 100K Gemini Tokens ~ 81K gpt-4 Tokens 100K Llama Tokens ~ 75K...</li><li><a href="https://x.com/amanrsanger/status/1771590523046051947?s=46&t=6FDPa">Tweet from Aman Sanger (@amanrsanger)</a>: &#34;Token Counts&#34; for long context models are a deceiving measure of content length. For code:  100K Claude Tokens ~ 85K gpt-4 Tokens 100K Gemini Tokens ~ 81K gpt-4 Tokens 100K Llama Tokens ~ 75K...</li><li><a href="https://x.com/karpathy/status/1723140519554105733">Tweet from Andrej Karpathy (@karpathy)</a>: LLM OS. Bear with me I&#39;m still cooking.  Specs: - LLM: OpenAI GPT-4 Turbo 256 core (batch size) processor @ 20Hz (tok/s) - RAM: 128Ktok - Filesystem: Ada002</li><li><a href="https://latecheckout.substack.com/p/the-guide-to-unbundling-reddit">The Guide to Unbundling Reddit</a>: Every few years a great unbundling occurs In 2010, Andrew Parker wrote a defining post about the &quot;unbundling&quot; of Craigslist where he outlined the opportunity to carve out niche products from...</li><li><a href="https://aneyeonai.libsyn.com/177-bjrn-ommer-are-diffusion-models-the-key-to-unlocking-ais-potential">Eye On A.I.: #177 Björn Ommer:  Diffusion Mods Explained By Stable Diffusion’s Creator</a>: Join host Craig Smith on episode #177 of Eye on AI as he explores the cutting-edge world of generative models in artificial intelligence with Björn Ommer, a visionary AI researcher and Head of Compute...</li><li><a href="https://x.com/emostaque/status/1771400218170519741?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Emad acc/acc (@EMostaque)</a>: As my notifications are RIP some notes:  1. My shares have majority of vote @StabilityAI  2. They have full board control  The concentration of power in AI is bad for us all  I decided to step down to...</li><li><a href="https://github.com/simonw/files-to-prompt">GitHub - simonw/files-to-prompt: Concatenate a directory full of files into a single prompt for use with LLMs</a>: Concatenate a directory full of files into a single prompt for use with LLMs - simonw/files-to-prompt</li><li><a href="https://github.com/semanser/codel">GitHub - semanser/codel: ✨ Fully autonomous AI Agent that can perform complicated tasks and projects using terminal, browser, and editor.</a>: ✨ Fully autonomous AI Agent that can perform complicated tasks and projects using terminal, browser, and editor. - semanser/codel</li><li><a href="https://www.pixee.ai/">Your Automated Product Security Engineer · Pixeebot</a>: Pixeebot provides immediate and constant fixes that make your code more secure. It’s like having another security-expert developer on your side.</li><li><a href="https://www.brightwave.io/">Brightwave</a>: no description found</li><li><a href="https://github.com/OwlAIProject/Owl">GitHub - OwlAIProject/Owl: A personal wearable AI that runs locally</a>: A personal wearable AI that runs locally. Contribute to OwlAIProject/Owl development by creating an account on GitHub.</li><li><a href="https://github.com/microsoft/LLMLingua">GitHub - microsoft/LLMLingua: To speed up LLMs&#39; inference and enhance LLM&#39;s perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression with minimal performance loss.</a>: To speed up LLMs&amp;#39; inference and enhance LLM&amp;#39;s perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression with minimal performance loss.  - GitH...</li><li><a href="https://github.com/OwlAIProject/Owl?tab=readme-ov-file#introducing-our-reference-hardware-device-bee-">GitHub - OwlAIProject/Owl: A personal wearable AI that runs locally</a>: A personal wearable AI that runs locally. Contribute to OwlAIProject/Owl development by creating an account on GitHub.</li><li><a href="https://ai.google.dev/docs/migrate_to_cloud">no title found</a>: no description found</li><li><a href="https://console.cloud.google.com/project)">Google Cloud Platform</a>: no description found</li><li><a href="https://cloud.google.com/resource-manager/docs/creating-managing-projects#creating_a_project)">no title found</a>: no description found</li><li><a href="https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).">Google Cloud Platform</a>: no description found
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1220814514711167046)** (5 messages): 

- **Insights into AI Giants**: A new podcast episode mentioned in a [tweet](https://twitter.com/swyx/status/1771255525818397122) unveils **juicy insights** into OpenAI, Google, and Adept, although not all prepared questions were covered.
- **AI In Action with Llama.cpp**: The AI In Action event started with a [live discussion](https://discord.com/channels/822583790773862470/1200548371715342479) about **Llama.cpp**, conducted by @363877777977376768.
- **ChatGPT's Unbundling Explored**: A new essay on the **Unbundling of ChatGPT** argues that despite stagnant user growth, OpenAI may still succeed amid a trend where users seek specialized AI services. The essay also prompts OpenAI to potentially release **Sora** and **GPT-5** to prevent mass unsubscriptions, and is available to read [here](https://latent.space/p/feb-2024).

**Link mentioned**: <a href="https://x.com/swyx/status/1772305930836918656?s=20">Tweet from swyx (@swyx)</a>: 🆕 The Unbundling of ChatGPT    https://latent.space/p/feb-2024   A whole year has passed with ~0 growth in ChatGPT user numbers. Instead, users are exploring a whole host of verticalized players for ...

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1220766447270105088)** (14 messages🔥): 

- **Technical Difficulties for LLM Paper Club**: A member encountered issues with obtaining speaking rights for a session in the **llm-paper-club-west** and expressed this in the chat.
- **Zoom to the Rescue**: With the inability to secure speaking rights in Discord, members resorted to using **Zoom** for the **paper club** meeting.
- **Speaker Rights Confusion**: There was confusion regarding how to obtain speaking rights in the Discord channel for future **paper club sessions**.
- **Meeting Over Before Resolution**: The meeting concluded on Zoom before Discord speaking permissions could be resolved, leading to the consideration of facilitating future stages.
- **Access Control Assistance Offered**: Another member indicated that **speaking rights** could be assigned by a certain individual, suggesting a possible solution for future meetings.
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1220824233127706638)** (92 messages🔥🔥): 

- **Discussions on Tensor Operations and Transform Models**: Members delved into tensor dimension handling, referred to humorously as "*pad and pray*", and pondered enhancing IDE support for dimension enforcement. The simplicity of envisioning transformer models as graphs with tensor operations and adjustable weights was highlighted as a mental model.

- **Unveiling Music with Slono**: A [Spotify link](https://open.spotify.com/artist/1rWeYVkrGXRqaD8e0kwMbc?si=xu1E7Di8T_OUpQvT46f-BA) was shared, showcasing Slono's work aimed at evoking the ambience of nights winding down.

- **Coding and Commenting Context in LLMs**: Discussion revolved around the value of comments in large language models (LLMs), emphasizing contextual information at varying levels of abstraction. Mentioned was the impact of comments on helping LLMs understand code.

- **Anticipation for Future Tech Showdowns**: Musings about C++'s speed advantage over Python and a light-hearted prediction of a 2025 face-off between Luminal, Tinygrad, and Mojo were shared. There was also interest in learning more about the Luminal project.

- **AI in Action Club Schedule and Topics Shared**: A Google Docs [spreadsheet](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0) containing forthcoming topics for AI in Action Club sessions, including UI/UX patterns for generative AI, RAG architectures, and the impact of prompt formatting on model evaluation, was made accessible.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bbycroft.net/llm">LLM Visualization</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.00789">Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces</a>: Attention mechanisms have been widely used to capture long-range dependencies among nodes in Graph Transformers. Bottlenecked by the quadratic computational cost, attention mechanisms fail to scale in...</li><li><a href="https://tenor.com/view/friends-bestfriends-yep-bff-gif-4566644">Did We Just Become Best Friends? GIF - Friends Bestfriends Yep - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown,@ UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://open.spotify.com/artist/1rWeYVkrGXRqaD8e0kwMbc?si=xu1E7Di8T_OUpQvT46f-BA">slono</a>: Artist · 110 monthly listeners.
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1220711791135424565)** (214 messages🔥🔥): 

- **GaLore Optimizer Heats Up the Chat**: Members discussed the [GaLore optimizer](https://github.com/jiaweizzhao/GaLore/issues/6), which offers significant VRAM savings during full parameter finetuning. Concerns were raised about its "coarseness" and potential to cause model over-training, likening the optimizer's granularity to using "a very coarse optimizer with adjustable resolution that can update all weights of the model."

- **Axolotl Discord Delves into Dataset Dilemmas**: One member inquired about configurations for sharegpt and chatml in SFT and DPO, with another confirming that chatml is indeed what the model sees. Meanwhile, there was confusion over an example config setting in the Axolotl repo, potentially leading to improper dataset tokenization paths.

- **Anticipation for New Models and Optimizers**: Amidst the technical talk, excitement was palpable about the release of **Mistral v0.2 Base Model**, boasting a larger context window of 32k; however, some lamented the restriction to Mistral 7B models. GaLore remains a hot topic with plans afoot to test over the weekend, fueling debates on optimizing strategies.

- **Publishing Predicament Posted**: A member shared their dilemma about whether to release a preprint of their medical model while it's undergoing its 3rd round of journal reviews. This sparked a discussion on the pros and cons of early sharing of research.

- **Open Calls and Company Queries**: CHAI announced support for open source LLM community with [prizes for LLM developers](https://chai-research.typeform.com/chaiprize), while another member encouraged companies using Axolotl for their business applications to reach out to share their use cases discreetly.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/posts/smangrul/896443101397392">@smangrul on Hugging Face: &quot;🤗 PEFT v0.10.0 release! 🔥🚀✨

Some highli📝ghts:
1. FSDP+QLoRA and DeepSpeed…&quot;</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/accelerate/fsdp#use-peft-qlora-and-fsdp-for-finetuning-large-models-on-multiple-gpus">Fully Sharded Data Parallel</a>: no description found</li><li><a href="https://x.com/xiangyue96/status/1771898843275067420?s=20">Tweet from Xiang Yue (@xiangyue96)</a>: @MistralAI just released their v0.2 Base😱. @WenhuChen and I quickly evaluated a few benchmarks using the OpenCompass evaluation package. It seems that the capability dropped a little bit on nearly al...</li><li><a href="https://huggingface.co/docs/peft/accelerate/deepspeed#use-peft-qlora-and-deepspeed-with-zero3-for-fi">DeepSpeed</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/accelerate/deepspeed#use-peft-qlora-and-deepspeed-with-zero3-for-finetuning-large-models-on-multiple-gpus">DeepSpeed</a>: no description found</li><li><a href="https://chai-research.typeform.com/chaiprize">Chai Prize</a>: Complete and win 3 days unlimited messages!</li><li><a href="https://github.com/mistralai-sf24/hackathon">GitHub - mistralai-sf24/hackathon</a>: Contribute to mistralai-sf24/hackathon development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/config.yml">axolotl/examples/mistral/config.yml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/trl/blob/8534f0edf8608ad6bcbea9beefae380fa60ded77/trl/trainer/dpo_trainer.py#L877-L881">trl/trl/trainer/dpo_trainer.py at 8534f0edf8608ad6bcbea9beefae380fa60ded77 · huggingface/trl</a>: Train transformer language models with reinforcement learning. - huggingface/trl</li><li><a href="https://github.com/jiaweizzhao/GaLore/issues/6">Third-party benchmark · Issue #6 · jiaweizzhao/GaLore</a>: Hello, thank you very much for such excellent work. We have conducted some experiments using Llama-Factory, and the results indicate that Galore can significantly reduce memory usage during full pa...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1220729246029316247)** (15 messages🔥): 

- **Unexpected TypeError in OpenAI Example**: A member encountered a `TypeError` when trying to run an example from "examples/openllama-3b/qlora.yml" related to `LlamaRotaryEmbedding.forward()` receiving an unexpected keyword argument 'seq_len'.
- **Room Redirection for Help**: Another user redirected the member facing the TypeError to a specific help channel with the ID #1111279858136383509 for better assistance.
- **Discussing Efficiency of LLM Fine-tuning**: Members discussed the possibility of fine-tuning a 7b model in 27gb of memory, referencing a GitHub repository called [torchtune](https://github.com/pytorch/torchtune) that facilitates *LLM Fine-tuning* without relying on Huggingface libraries.
- **Fine-tuning Ramifications**: A member indicated the benefits of using native torch for efficiency while acknowledging the steeper learning curve compared to using libraries like Huggingface.
- **Recommendation and Teasing About Huggingface**: A [pull request on torchtune](https://github.com/pytorch/torchtune/pull/527) was recommended for reviewing how to full fine-tune with less than 16GB of RAM, along with a playful jab at Huggingface's expense.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune">GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/527">Full finetune &lt; 16GB by rohan-varma · Pull Request #527 · pytorch/torchtune</a>: Context  We&#39;d like to enable a variant of full finetune that trains in &lt; 16GB of RAM for users with consumer grade GPUs that have limited GPU RAM. This PR enables the full finetune to fit into ...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1220729707608277003)** (14 messages🔥): 

- **Mixtral Fine-Tuning Techniques Unclear**: A member is looking for advice on how to target the router layers in Mixtral with galore but has not found clear documentation online. They mentioned an intention to try **-block_sparse_moe** and **-self_attn** but later lamented that it does not work with Zero3.

- **Coding Assistant Training for Mixtral-7B**: A user asked how to train and fine-tune a Mixtral-7B model to be a coding assistant using runpod, python, etc., questioning the tools, IDEs, and concepts needed to train a Mixtral model on their own hardware. Another member acknowledged the complexity of the question.

- **Data Preprocessing Error in Axolotl**: While attempting to pre-process data with Axolotl, a member faced a KeyError related to the 'instruction' key, even though they confirmed all rows included the key. Another participant suggested there might be rows missing the key, but this was not the case upon verification.

- **Fine-Tuning Issues with TheBloke's Model**: A user encountered an error while trying to fine-tune TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ model using auto train, sharing error outputs indicating a FileNotFoundError in subprocess.py on Windows. They also shared a link to the [model's repository on Hugging Face](https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ).

- **Inquiry About 'Gema' Compatibility with PyTorch**: A member asked if 'gema' is still incompatible with PyTorch, seeking up-to-date information on the issue. No clear consensus or answer was provided within the discussed messages.

**Link mentioned**: <a href="https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ">TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ · Hugging Face</a>: no description found

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1221161287770181642)** (8 messages🔥): 

- **Midnight 70B Launches to Acclaim**: **Midnight 70B** is the newest and most anticipated model optimized for storytelling and roleplay, a successor to Rogue Rose and Aurora Nights. Available at [OpenRouter](https://openrouter.ai/models/sophosympatheia/midnight-rose-70b), with an introductory price of **$0.009/1k tokens**, a **25% discount**.

- **New Features for Cost Monitoring and Management**: OpenRouter introduced **Usage Analytics** with a new chart feature displaying the daily spending on models, as well as a **Billing Portal** accessible via the users' account for managing credits and invoices.

- **Price Adjustments for Noromaid Mixtral and Bagel**: Due to the high cost of running them, the discounts on Noromaid Mixtral and Bagel models have been removed, with the former priced at **$0.008/1k tokens** and the latter at **$0.00575/1k tokens**.

- **Request for Expanded Context Length**: A user expressed a desire to use Noromaid Mixtral at its original **32K context length**, stating that the current 8K is insufficient.

- **Database Downtime Due to DDoS**: The OpenRouter platform experienced database issues due to a DDoS attack that bypassed Cloudflare, but stability was restored as per the latest update.

**Link mentioned**: <a href="https://openrouter.ai/models/sophosympatheia/midnight-rose-70b">Midnight Rose 70B by sophosympatheia | OpenRouter</a>: A merge with a complex family tree, this model was crafted for roleplaying and storytelling. Midnight Rose is a successor to Rogue Rose and Aurora Nights and improves upon them both. It wants to produ...

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1220663758121341028)** (208 messages🔥🔥): 

- **Claude 3 Self-Moderated Version Endorsed**: Claude 3's self-moderated version is recommended over the site-filtered version due to better selectivity in rejection.
- **The Case of the Underappreciated Grok**: Grok models sparked a debate, with some users arguing that while it's not as powerful as Mixtral, it's a solid base model when not compared to fine-tuned alternatives, despite being more expensive.
- **Fine-Tuning Model Favorites**: Multiple users discussed their experiences with different models for roleplay and tasks, expressing interest in potentially extended contexts beyond 8k for models like Midnight Rose and the consistent quality from open source models like Haiku on a budget.
- **OpenRouter and Model Performances**: Some users reported differences in the quality of model completions when using models like Opus and Haiku on OpenRouter versus direct API access and are looking into whether it involves default system prompts.
- **Perplexity Citations in OpenRouter**: Discussions around not receiving citation data via OpenRouter when using Perplexity emerged, with acknowledgement that while the data exists, it's not currently returned due to API response consistency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/deliprao/status/1770128250003460396?s=46">Tweet from Delip Rao e/σ (@deliprao)</a>: I look at this and don&#39;t walk way thinking Grok is better. As a pragmatic person, I look at it and wonder why bother with Grok (314B) when you have Mixtral with almost similar performance and is a...</li><li><a href="https://grok.x.ai/">xAI Grok</a>: no description found</li><li><a href="https://worldsim.nousresearch.com">world_sim</a>: no description found</li><li><a href="https://imgur.com/a/JWX7br0">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://huggingface.co/sophosympatheia/Midnight-Rose-70B-v2.0.3">sophosympatheia/Midnight-Rose-70B-v2.0.3 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1220704866775863336)** (64 messages🔥🔥): 

- **GPU Battlegrounds: Apple’s MPS vs. PyTorch**: A member is working diligently on improving the MPS backend in PyTorch, discussing both the significance of the work for local model testing and finetuning, and the potential widespread performance benefits. Despite challenges, they remain committed, highlighting an issue in tensor copying that has affected MPS since September 2022.

- **Debate on Token Block Strategies for LLM Pretraining**: Members engaged in a nuanced debate about the best way to form token blocks for language model pretraining, deliberating between overlapping versus non-overlapping sequences. Multiple perspectives were offered, taking into account the importance of beginning-of-sentence tokens and the implication of each method on training efficacy.

- **AMD vs. Nvidia GPU Drivers – A Market Challenge**: There was a lively discussion about the perceived inadequacy of AMD Radeon drivers compared to Nvidia's, and its market implications. Participants noted that consumer GPU drivers often offload compatibility work elsewhere, debated the potential for AMD open-sourcing their drivers, and considered activist investor intervention to drive corporate change at AMD.

- **Concerns Over Public Speaking Prowess in AI Industry**: There was a brief commentary comparing the public speaking abilities of well-known figures in the tech industry. Lex Fridman was mentioned as a point of reference in discussing the styles of other speakers.

- **Merger Mania in Machine Learning**: A member introduced a new merge method they are developing for combining models that could potentially surpass existing methods like DARE. They noted that their approach is in early stages and that more testing is needed to confirm its effectiveness.

**Link mentioned**: <a href="https://tenor.com/view/ratatouille-flashback-childhood-memory-delicious-gif-3463448">Ratatouille • Flashback GIF - Ratatouille Flashback Childhood - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1220604913982242857)** (105 messages🔥🔥): 

- **DenseFormer with Depth-Weighted-Average**: A new tweak in transformer architecture, named [DenseFormer](https://arxiv.org/abs/2402.02622), adds a depth-weighted average step to significantly improve the perplexity of large-scale models without enlarging their size. A [discussion on Hacker News](https://news.ycombinator.com/item?id=39794906) indicates skepticism over its scalability, yet proponents argue for its potential.

- **Mamba Meets Zigzag**: Addressing the inherent issues of diffusion models in scalability and complexity, a study introduces [Zigzag Mamba](https://arxiv.org/abs/2403.13802), a variant that enhances memory use and speed in processing high-resolution visual datasets. The study, contributed by <@193386166517628929>, focuses on optimizing sequence flattening methods for improved performance.

- **Putting Pieces Together in DiPaCo**: The proposition of the [DiPaCo](https://arxiv.org/abs/2403.10616v1) architecture innovates machine learning model training by using a path composition approach to ensure robustness and reduced communication needs across potentially disconnected computational workers. This suggests a potential path for decentralized machine learning model training.

- **Potential of "Proof-of-Training-Data"**: Addressing concerns about model provenance and the risk of poisoned samples, an arXiv [abstract](https://arxiv.org/abs/2307.00682) explores the idea of "Proof-of-Training-Data," which would allow verification of the data and computation used to train neural networks.

- **Training Bias with BiTFiT**: Research has been conducted on applying BitFit to [modern large language models](https://github.com/lawrence-cj/LLaMA-DiffFit) like LLama2/Mistral. The posted study demos efficient fine-tuning of LLaMA models by initializing new bias terms and freezing other parameters, thereby enhancing parameter efficiency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://proceedings.mlr.press/v139/davis21a.html">Catformer: Designing Stable Transformers via Sensitivity Analysis</a>: Transformer architectures are widely used, but training them is non-trivial, requiring custom learning rate schedules, scaling terms, residual connections, careful placement of submodules such as n...</li><li><a href="https://arxiv.org/abs/2402.02622">DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging</a>: The transformer architecture by Vaswani et al. (2017) is now ubiquitous across application domains, from natural language processing to speech processing and image understanding. We propose DenseForme...</li><li><a href="https://news.ycombinator.com/item?id=39794906">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.13802">ZigMa: Zigzag Mamba Diffusion Model</a>: The diffusion model has long been plagued by scalability and quadratic complexity issues, especially within transformer-based structures. In this study, we aim to leverage the long sequence modeling c...</li><li><a href="https://adamkarvonen.github.io/machine_learning/2024/03/20/chess-gpt-interventions.html">Manipulating Chess-GPT’s World Model</a>: Manipulating Chess-GPT’s World Model</li><li><a href="https://arxiv.org/abs/1802.01483">Explicit Inductive Bias for Transfer Learning with Convolutional Networks</a>: In inductive transfer learning, fine-tuning pre-trained convolutional networks substantially outperforms training from scratch. When using fine-tuning, the underlying assumption is that the pre-traine...</li><li><a href="https://arxiv.org/abs/2403.10616v1">DiPaCo: Distributed Path Composition</a>: Progress in machine learning (ML) has been fueled by scaling neural network models. This scaling has been enabled by ever more heroic feats of engineering, necessary for accommodating ML approaches th...</li><li><a href="https://arxiv.org/abs/2307.00682">Tools for Verifying Neural Models&#39; Training Data</a>: It is important that consumers and regulators can verify the provenance of large neural models to evaluate their capabilities and risks. We introduce the concept of a &#34;Proof-of-Training-Data&#34;:...</li><li><a href="https://arxiv.org/abs/1802.07044">The Description Length of Deep Learning Models</a>: Solomonoff&#39;s general theory of inference and the Minimum Description Length principle formalize Occam&#39;s razor, and hold that a good model of data is a model that is good at losslessly compress...</li><li><a href="https://arxiv.org/abs/2403.15297">Sphere Neural-Networks for Rational Reasoning</a>: The success of Large Language Models (LLMs), e.g., ChatGPT, is witnessed by their planetary popularity, their capability of human-like question-answering, and also by their steadily improved reasoning...</li><li><a href="https://arxiv.org/abs/2103.01075">OmniNet: Omnidirectional Representations from Transformers</a>: This paper proposes Omnidirectional Representations from Transformers (OmniNet). In OmniNet, instead of maintaining a strictly horizontal receptive field, each token is allowed to attend to all tokens...</li><li><a href="https://github.com/lawrence-cj/LLaMA-DiffFit">GitHub - lawrence-cj/LLaMA-DiffFit: Efficient Fine-tuning LLaMA Using DiffFit within 0.7M Parameters</a>: Efficient Fine-tuning LLaMA Using DiffFit within 0.7M Parameters - lawrence-cj/LLaMA-DiffFit
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1221451424299221062)** (5 messages): 

- **SVM Kernels Explored for Pythia Embeddings**: One member reported that after running several SVM kernels on Pythia's input embeddings, the **sigmoid kernel outperformed** rbf, linear, and poly kernels. Although the finding wasn't attributed to cleverness but to trial and error, the user expressed a desire for **intuition to streamline the process**.

- **SVM vs. Logistic Regression**: A participant admitted they lack knowledge about SVMs and would have rather used **logistic regression** for classification problems.

- **Tokengrams Repository Update**: The *Tokengrams* project has progressed to a point of usability, as indicated with a shared link to the GitHub repository. This tool is for **efficiently computing and storing token n-grams from large corpora**. [GitHub - EleutherAI/tokengrams](https://github.com/EleutherAI/tokengrams).

- **Chess-GPT Interventions Summarized**: A link to a blog post was shared, detailing the **Chess-GPT** project which uses a language model to predict chess moves from PGN strings and estimates the skill level of players. The post describes the use of **linear probes to validate the model's computations** and mentions Chess-GPT's capability of playing chess at approximately 1500 Elo. [Chess GPT Interventions](https://adamkarvonen.github.io/machine_learning/2024/03/20/chess-gpt-interventions.html).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://adamkarvonen.github.io/machine_learning/2024/03/20/chess-gpt-interventions.html">Manipulating Chess-GPT’s World Model</a>: Manipulating Chess-GPT’s World Model</li><li><a href="https://github.com/EleutherAI/tokengrams">GitHub - EleutherAI/tokengrams: Efficiently computing &amp; storing token n-grams from large corpora</a>: Efficiently computing &amp; storing token n-grams from large corpora - EleutherAI/tokengrams
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1220621835327111238)** (26 messages🔥): 

- **Potential Variance in Evaluation Results**: A member discussed encountering variations in evaluation results, with about half the runs matching exactly and the other half differing by approximately 0.5% when comparing **Hugging Face (HF) transformers** with **Megatron-DeepSpeed** evaluations. They mentioned that checking a forward pass numerically could help identify if there are fundamental differences in the implementations. 

- **Determinism in Attention Mechanisms Questioned**: In a quest to understand the discrepancies in evaluation results, a member questioned whether *flash attention* could contribute to variation, but it was clarified that flash attention is deterministic in forward pass. They also speculated if differences in doing *fused kqv multiplications* could be causing numerical discrepancies, potentially due to **bfloat16**.

- **Minecraft as an RL Benchmark for LLM Collaboration**: One member highlighted a GitHub repository, [GitHub - danijar/diamond_env](https://github.com/danijar/diamond_env), which represents a standardized Minecraft Diamond Environment for Reinforcement Learning. They also referenced an issue on the [Voyager](https://github.com/MineDojo/Voyager/issues/149) project's GitHub discussing potential collaboration with LM harness projects.

- **Inverse-Scaling Evaluation Pipeline Inquiries**: A member inquired about adapting a multi-choice problem-solving approach from the inverse-scaling evaluation pipeline to work with the **lm-eval-harness**. A **code snippet** from their [GitHub repository](https://github.com/naimenz/inverse-scaling-eval-pipeline/blob/main/eval_pipeline/models.py) was provided for discussion, leading to an explanation of how logits are treated in the context of an answer choice in the harness.

- **Question on BB-Q Lite Task Scoring Method in BigBench**: A member questioned whether the **bbq_lite** subset in the BigBench task uses a straight accuracy scoring method and proposed that the complexity of its original bias scoring mechanism was possibly avoided in implementation. It was suggested to refer to a specific [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1185) for an alternative BBQ implementation in the **lm-evaluation-harness**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/naimenz/inverse-scaling-eval-pipeline/blob/main/eval_pipeline/models.py">inverse-scaling-eval-pipeline/eval_pipeline/models.py at main · naimenz/inverse-scaling-eval-pipeline</a>: Basic pipeline for running different sized GPT models and plotting the results - naimenz/inverse-scaling-eval-pipeline</li><li><a href="https://github.com/MineDojo/Voyager/issues/149">Implement a way test local models · Issue #149 · MineDojo/Voyager</a>: Hello, Wonderful work on Voyager. Please consider added local model support (instead of openai package - using something like Python requests package to a localhost local model using openai complet...</li><li><a href="https://github.com/danijar/diamond_env">GitHub - danijar/diamond_env: Standardized Minecraft Diamond Environment for Reinforcement Learning</a>: Standardized Minecraft Diamond Environment for Reinforcement Learning - danijar/diamond_env</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1185">Add various social bias tasks by oskarvanderwal · Pull Request #1185 · EleutherAI/lm-evaluation-harness</a>: This PR implements various popular benchmarks for evaluating LMs for social biases. I also aim to have these validated where possible: e.g., by comparing with existing implementations or results, o...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1221160628501086388)** (3 messages): 

- **Inquiry about Multimodal Embedding Theories**: A member asked for theoretical works on **multimodal embedding spaces**, indicating a broad interest without looking for anything specific.
- **Insight on Embeddings in Stable Diffusion Culture**: Stable Diffusion’s subculture treats embeddings similarly to **IMG2IMG workflows** in their diffusion models, notably SDXL IMG2IMG which might offer a lead for research. 
- **Clarification on Terminology**: The term "**IMG2IMG**" could be confused with "init image" usage, especially due to the use of this phrase in the Automatic1111 web UI; alternatives like "image prompting" or "image variations" were suggested.
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1220828118881931284)** (9 messages🔥): 

- **Discord stage channel struggles**: During a GTC event, the Discord stage channel encountered screen sharing issues, leading to a quick resolution by switching to a voice channel. A suggestion was made to use voice channels by default for future lectures.
- **Google Meet over Discord**: One member expressed frustration with Discord streams and proposed using Google Meet for future sessions, seeking opinions and contacts at Discord for feedback on stream stability.
- **ML and CUDA Connection**: A member inquired about when CUDA programming becomes necessary in ML, as they've never had to delve that deep in their ML practice.
- **CUDA Programming for Speed**: A link to a YouTube lecture was shared to explain profiling CUDA kernels in PyTorch: [Lecture 1 How to profile CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM). Accompanying resources include [slides on Google Docs](https://docs.google.com/presentation/d/110dnMW94LX1ySWxu9La17AVUxjgSaQDLOotFC3BZZD4/edit?usp=sharing) and a [GitHub code repository](https://github.com/msaroufim/cudamodelecture1).
- **Understanding When to Drop to CUDA**: In response to the lecture, a member summarized that CUDA is necessary for performance gains when PyTorch is not fast enough, likening it to writing in C for CPU programs. Another member agreed with this takeaway.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=LuhJEEJQgUM">Lecture 1 How to profile CUDA kernels in PyTorch</a>: Slides: https://docs.google.com/presentation/d/110dnMW94LX1ySWxu9La17AVUxjgSaQDLOotFC3BZZD4/edit?usp=sharingCode:  https://github.com/msaroufim/cudamodelecture1

  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1221525289603956766)** (27 messages🔥): 

- **Debugging Triton Performance Issues**: A member sought advice on debugging performance issues with Triton kernels, comparing **unsloth's fast_embedded_rope** performance unfavorably to eager PyTorch on an A10G with contiguous tensors.

- **Assurances on Triton Compiler Bug Resolution**: Members discussed historical **Triton compiler bugs** referenced in code comments, with clarification provided that these were not current issues. Additionally, barriers such as `debug_barrier()` were explained as necessary for correctness, similar to syncthreads in CUDA.

- **Triton Operations May Be Phased Out**: A contributor indicated that Triton operations might be removed in the future, advising against submitting PRs to address related issues, and confirmed that tutorials would remain available for learning purposes.

- **Possible Benchmarking From Meta**: It was mentioned that Meta could potentially introduce an **op benchmark** for Triton, which would provide reference implementations for developers to utilize.

- **Proposal for Collaboration on Architecture Optimization**: A new prototype folder in the `torchao` repository was suggested to a member, with the intention of merging their work and collaborating on API design for efficient kernel usage.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch-labs/ao">GitHub - pytorch-labs/ao: torchao: PyTorch Architecture Optimization (AO). A repository to host AO techniques and performant kernels that work with PyTorch.</a>: torchao: PyTorch Architecture Optimization (AO). A repository to host AO techniques and performant kernels that work with PyTorch. - pytorch-labs/ao</li><li><a href="https://github.com/openai/triton/blob/fb8983e5a2754ce793ab8d14ed0c333bfd9ba197/python/triton/ops/cross_entropy.py#L35">triton/python/triton/ops/cross_entropy.py at fb8983e5a2754ce793ab8d14ed0c333bfd9ba197 · openai/triton</a>: Development repository for the Triton language and compiler - openai/triton
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1220999272372375652)** (7 messages): 

- **Seeking the Blackwell NVIDIA Whitepaper**: A user inquired about the release of the Blackwell NVIDIA whitepaper but no direct information on this topic was provided.

- **GTC Session Details Shared**: A member shared a link to the [GTC Session Catalog](https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=S62400#/session/1696033648682001S1DC) highlighting the upcoming workshops, AI conference and Expo dates, and the keynote scheduled for March 17-21 in San Jose, CA and virtually.

- **CUDA Toolkit and CuDNN Installation Guidance**: A user asked if it was okay to install a CUDA toolkit of a higher version than shown on `nvidia-smi` and about post-installation steps for CuDNN. Another member mentioned that one needs to add CuDNN to the path or copy its files to the toolkit directory.

- **Link Omission for Toolkit/Driver Compatibility**: A member failed to include a link when referring to toolkit/driver compatibility. The link was subsequently provided, directing users to NVIDIA's [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) to understand the minimum driver versions required for each toolkit.

- **Call for Favorite CUDA Kernels**: A member invited others to submit favorite CUDA kernels that could optimize operations for large language models (LLMs) to possibly feature in a Thunder tutorial. The discussion links to a [GitHub issue](https://github.com/Lightning-AI/lightning-thunder/issues/70) on Lightning AI's repository that discusses this feature.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/deploy/cuda-compatibility/index.html">CUDA Compatibility  :: NVIDIA GPU Management and Deployment Documentation</a>: no description found</li><li><a href="https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=S62400#/session/1696033648682001S1DC">NVIDIA #GTC2024 Conference Session Catalog</a>: Register now. Streamed online. March 18-21, 2024.</li><li><a href="https://github.com/Lightning-AI/lightning-thunder/issues/70">Support for CUDA kernels · Issue #70 · Lightning-AI/lightning-thunder</a>: 🚀 Feature Hi there 👋 From the main readme file I noticed that Thunder except custom kernels, but only the ones that are written in Trition. Is there a plan to support CUDA kernels? Motivation I&#39;...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1220986153574994001)** (3 messages): 

- **Link to New Matrix Decomposition Paper**: A paper on **Arrow Matrix Decomposition** by researchers [Lukas Gianinazzi](https://arxiv.org/search/cs?searchtype=author&query=Gianinazzi,+L), [Alexandros Nikolaos Ziogas](https://arxiv.org/search/cs?searchtype=author&query=Ziogas,+A+N), and others was shared, providing insights on a novel approach to distributed sparse matrix multiplication. Access the research [here](https://arxiv.org/abs/2402.19364).

- **GitHub Repository for Arrow Matrix Decomposition**: The **Arrow Matrix Decomposition** code has been made available on GitHub for those interested in communication-efficient distributed sparse matrix multiplication. The repo is available at this [GitHub link](https://github.com/spcl/arrow-matrix).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.19364">Arrow Matrix Decomposition: A Novel Approach for Communication-Efficient Sparse Matrix Multiplication</a>: We propose a novel approach to iterated sparse matrix dense matrix multiplication, a fundamental computational kernel in scientific computing and graph neural network training. In cases where matrix s...</li><li><a href="https://github.com/spcl/arrow-matrix">GitHub - spcl/arrow-matrix: Arrow Matrix Decomposition - Communication-Efficient Distributed Sparse Matrix Multiplication</a>: Arrow Matrix Decomposition - Communication-Efficient Distributed Sparse Matrix Multiplication - spcl/arrow-matrix
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1221559417250517166)** (3 messages): 

- **GPU vs CPU Architecture Complexity**: A member questioned whether **NVIDIA GPU architecture** is simpler than that of modern CPUs. Another member clarified that GPUs are specialized for high throughput of simple operations, in contrast to CPUs, which handle a lower throughput but more complex operations.
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1220761584335327242)** (12 messages🔥): 

- **Accountability in Progress**: A member committed to completing and discussing exercises for Chapters 3 & 4 of their study material after long work shifts, using public accountability as a motivator.
- **Resource Sharing for Exercise Answers**: It was suggested to create a shared Google Doc to compile agreed-upon exercise answers for cross-checking and as a resource for all members.
- **Exclusive Access to Exercise Solutions**: One member proposed starting a shared document with exercise solutions, offering access to those who show their initial attempt to maintain the challenge integrity.
- **Experience Sharing Among Members**: Members exchanged their backgrounds regarding experience with C++ and multithreading, with varying focus on CUDA and broader parallel programming concepts for applying to various technologies.
- **Collaborative Learning Through Shared Solutions**: A link to a Google Doc containing Chapter 2 exercise solutions was shared, accessible to those who DM the creator after attempting the exercises themselves.
[Ch 2 Exercise Solutions](https://docs.google.com/document/d/10ez800eu8OF-OzJXNZ0tRGdJaRAwagiyFdgeBoX0S8o/edit)
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1221305349873602570)** (5 messages): 

- **Inquiring About Lecture 11 Upload**: A user inquired when the recording for **Lecture 11** would be uploaded. Another user responded that it will be on YouTube once Mark has time, sharing a temporary link to watch it on [OneDrive](https://1drv.ms/v/s!AsJJewlEEg2oiPp1ja8bHuVmbVYp4Q?e=pHJp67).

- **Lecture 11 Now on YouTube**: It was confirmed that **Lecture 11** had been uploaded to [YouTube](https://youtu.be/mGDnOLcfE8g).

- **Seeking Sparsity Lecture Slides**: A user requested the slides for the Sparsity lecture, asking if they were public and for a link. Another user pinged a specific member to share the slides if possible.

**Link mentioned**: <a href="https://1drv.ms/v/s!AsJJewlEEg2oiPp1ja8bHuVmbVYp4Q?e=pHJp67">no title found</a>: no description found

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1221129136496574464)** (21 messages🔥): 

- **Diving Back into Ring Attention**: A member mentioned they will focus on **Ring Attention** for the next ten days to run some tests and explore training details.
- **Clarifying Meetup Time Post-Daylight Saving**: A member asked about the exact time for regular meetings, to which another member replied with the scheduled timing using a timestamp: <t:1711299600:t>.
- **Potential Workspace Upgrade Suggestion**: One member proposed the idea of increasing the workspace folder disk quota, and it was followed by a discussion about possibly migrating to a new machine with more storage.
- **Sharing Progress and Workspace Access**: Several links to Wandb.ai were shared showing progress on runs related to Axolotl. SSH configuration was updated and discussions about reinstalling conda and re-adding SSH keys took place.
- **Technical Adjustments for Collaboration**: There were conversations regarding conda reinstallation, with the base environment being moved under `/workspace/miniconda3`. SSH access was being coordinated, with requests for public keys to be sent for those needing to connect for the first time.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/iron-bound/axolotl/runs/wjb8eyw3/workspace?nw=nwuserironbound">iron-bound</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/iron-bound/axolotl/runs/7djmd1i2?nw=nwuserironbound">iron-bound</a>: Weights & Biases, developer tools for machine learning
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1220626324289228822)** (15 messages🔥): 

- **GPU Grins Back**: A tweet showing the new **Blackwell GPUs** seemingly having a smiley face pattern was highlighted. A [Twitter link](https://fxtwitter.com/iScienceLuvr/status/1770931936657358908) was shared for amusement.
- **NVIDIA's Best Chips Yet**: The **B200 accelerators** with their impressive specifications were discussed, citing them as the best in the market combined with their CUDA ecosystem. An [AnandTech article](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data) detailing NVIDIA's Blackwell architecture was shared.
- **Hidden in Plain Sight**: A member revealed the existence of an "NVIDIA Developer" Discord server linked to a GitHub discussion ([GitHub link](https://github.com/NVIDIA/cutlass/discussions/1086)) about the **CUTLASS library**.
- **Diving Into New Data Types**: Reference materials were requested for new float/int data types in deep learning, leading to the sharing of an [FP8 introduction paper](https://arxiv.org/abs/2209.05433) and an [OCP standardization post](https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai) about various companies standardizing next-generation narrow precision data formats.
- **A Sea of Standards**: Discussion surrounded the implementation variety and lack of **IEEE standard** for new floating-point numbers, with a notable absence of Google from the consortium agreeing on new formats.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data">NVIDIA Blackwell Architecture and B200/B100 Accelerators Announced: Going Bigger With Smaller Data</a>: no description found</li><li><a href="https://fxtwitter.com/iScienceLuvr/status/1770931936657358908">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Why isn&#39;t anybody talking about the fact that the new Blackwell GPUs are literally smiling at us lol</li><li><a href="https://github.com/NVIDIA/cutlass/discussions/1086">New Discord Channel! · NVIDIA/cutlass · Discussion #1086</a>: As a means to further improve the user experience and education around CUTLASS, we have created a new Discord channel! Click the link to join! See you there and we thank you for all of your support :)</li><li><a href="https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai">Open Compute Project</a>: no description found
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1220856479830245437)** (37 messages🔥): 

- **Stuck on Puzzle 4**: A member is **debugging Puzzle 4** with a discrepancy between the expected and actual test results. They shared their print statements to cross-check their answers, mentioning that the issue occurs when using torch for the outer sum.

- **Insights on Puzzle 10 batching**: In response to a query about whether to parallelize on the batch dimension for Puzzle 10, it's mentioned that the focus should be on keeping the kernel in **fast shared memory,** not necessarily on parallelizing the batch dimension, although tensor cores could also be utilized.

- **Initializing Arrays with Negative Infinity**: A discussion took place about how to initialize an array with all `-inf` values in Triton. Using functions like `tl.full` and substitutes like a large negative number are suggested solutions, as `tl.arange(0, B1) * float("-inf")` results in NaNs due to 0 * -inf.

- **Challenges with Indexing in Triton**: Queries about single-position indexing and slicing in arrays lead to the clarification that such operations are not supported in Triton. This is due to how arrays and memory are handled, and workarounds for these limitations involve avoiding direct indexing or employing associative scans.

- **Puzzle 3 Exploration Reveals Understanding**: A self-described 'noob' question about Puzzle 3 led to a user figuring out their misunderstanding on their own. The issue revolved around loading and adding vectors within the Triton kernel.
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1221172294840356924)** (11 messages🔥): 

- **Mistral Hints at New Model at Hackathon**: Chat messages link to tweets by **@AlexReibman** and **@adarshxs** indicating that **Mistral** dropped a new model at the [@cerebral_valley hackathon](https://x.com/alexreibman/status/1771608346635751541?s=46).

- **No Magnet for the New Release Announcement**: **@natolambert** notes that the [new model release](https://twitter.com/MistralAI) is absent any magnet links, expressing a bit of disappointment.

- **Mistral 7B v0.2 Base Model Details Revealed**: **@xeophon.** shares direct links from **@MistralAILabs** that detail the [new release of Mistral 7B v0.2 Base](https://x.com/mistralailabs/status/1771670765521281370?s=46), including specifics on its configuration and where to find guidance on how to fine-tune the model.

- **Reflection on Mistral's Growth**: **@xeophon.** comments casually on the rapid growth and development of Mistral, perceived through the frequency of new model releases.

- **Clarification on Mistral Model Versions**: Members **@philpax** and **@xeophon.** discuss the iterations of Mistral models, clarifying that the recently mentioned **Mistral-0.2** is not a completely new model but related to a previous instruct version, with **@philpax** initially misunderstanding the versioning before correcting himself.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/alexreibman/status/1771608346635751541?s=46">Tweet from Alex Reibman 🖇️ (@AlexReibman)</a>: Mistral casually dropping a new model at the @cerebral_valley hackathon</li><li><a href="https://x.com/adarshxs/status/1771610229412614149">Tweet from Adarsh (@adarshxs)</a>: yo @MistralAI dropping a new model today!!</li><li><a href="https://x.com/mistralailabs/status/1771670765521281370?s=46">Tweet from Mistral AI Labs (@MistralAILabs)</a>: New release: Mistral 7B v0.2 Base (Raw pretrained model used to train Mistral-7B-Instruct-v0.2) 🔸 https://models.mistralcdn.com/mistral-7b-v0-2/mistral-7B-v0.2.tar 🔸 32k context window 🔸 Rope Theta...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1221524167971831918)** (2 messages): 

- **Nemo Checkpoint Conversion Inquiry**: A member inquired about how to convert a **Nemo checkpoint** to be compatible with Hugging Face for inference purposes.
- **Exploring Checkpoint Wrapping**: The same member also asked for advice on wrapping **Nemo checkpoints** for use, potentially looking for a wrapper or an interface.
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1220943001992298597)** (29 messages🔥): 

- **Stability AI CEO Steps Down**: Stability AI announced that [CEO Emad Mostaque resigned](https://stability.ai/news/stabilityai-announcement) from his CEO role and board position to pursue decentralized AI, with Shan Shan Wong and Christian Laforte stepping in as interim co-CEOs. Mostaque's tweets hinted at a focus on **#DecentralizedAI** and governance in AI.

- **Stability AI Internal Struggles and Speculations**: Discussion in the chat suggests that Stability AI has faced **longstanding internal issues**, leading to Emad Mostaque's departure from the company. Members debated whether Mostaque's actions were a grift or a result of the company's continuous scramble to find direction.

- **The Fine Line Between Contribution and Grift**: Chat members shared perspectives on the nature of Stability AI's operations, with some feeling they **appropriately licensed** algorithms developed by academics, while others saw it as questionable given the academics’ minor compute contributions.

- **AI Community's Take on Emad Mostaque's Departure**: Opinions varied about Emad Mostaque's legacy, with some perceiving him as a **grifter** while acknowledging the **legitimate aspects** of Stability AI's business.

- **Alternatives for AI Academics**: A point was raised regarding the limited options for academics in AI, indicating a preference for collaboration with companies like Stability AI to have a more significant impact than possible with limited resources in academia.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/ClementDelangue/status/1771395468959813922">Tweet from clem 🤗 (@ClementDelangue)</a>: Should we acquire Stability and open-source SD3?</li><li><a href="https://stability.ai/news/stabilityai-announcement">Stability AI Announcement &mdash; Stability AI</a>: Earlier today, Emad Mostaque resigned from his role as CEO of Stability AI and from his position on the Board of Directors of the company to pursue decentralized AI.  The Board of Directors has appoin...</li><li><a href="https://fxtwitter.com/emostaque/status/1771403116099068048?s=46">Tweet from Emad acc/acc (@EMostaque)</a>: * by they - my shares have full board control aha  So its a decision by me as it were  We should have more transparent & distributed governance in AI as it becomes more and more important   Its a hard...</li><li><a href="https://x.com/egrefen/status/1771628344204795962?s=46&t=pgJi6RxHJJYXIrBn2rKMXg">Tweet from Edward Grefenstette (@egrefen)</a>: This week in AI: Quis griftat ipsos griftatores?</li><li><a href="https://fxtwitter.com/emostaque/status/1771407651387383845?s=46">Tweet from Emad acc/acc (@EMostaque)</a>: Also there is no presale, TGE or token fml  If there was though I would call it stable coin 😏</li><li><a href="https://fxtwitter.com/emostaque/status/1771380668930674850?s=46">Tweet from Emad acc/acc (@EMostaque)</a>: Not going to beat centralized AI with more centralized AI.  All in on #DecentralizedAI   Lots more 🔜  ↘️ Quoting Stability AI (@StabilityAI)   An announcement from Stability AI: https://bit.ly/43zsVj...</li><li><a href="https://stability.ai/news">News &mdash; Stability AI</a>: Discover the latest product announcements, company updates, and industry news with Stability AI.  We build the foundation to activate humanity’s potential.</li><li><a href="https://fxtwitter.com/emostaque/status/1771400218170519741?s=46">Tweet from Emad acc/acc (@EMostaque)</a>: As my notifications are RIP some notes:  1. My shares have majority of vote @StabilityAI  2. They have full board control  The concentration of power in AI is bad for us all  I decided to step down to...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1220635579079987210)** (8 messages🔥): 

- **RL Generalist Agent Discussion**: A member linked to a [discussion on Twitter](https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw) about the philosophy of creating a "generalist agent" in Reinforcement Learning (RL), considering the practical and principal possibilities of realizing such an agent.
- **Misinterpretation of Antitrust Laws Online**: Nathan Lambert expressed frustration over the general public's misunderstanding of antitrust laws in regards to recent tech lawsuits and debates.
- **Criticism of Apple Antitrust Lawsuits**: Several messages by Nathan Lambert criticize the FTC's lawsuits against Apple, suggesting that people cheering them on should read more informed opinions, such as those by Ben Thompson.
- **Disagreement with Twitter Sentiments on FTC vs. Apple**: In an argument on Twitter, Nathan Lambert holds the position that the FTC lawsuits against Apple are misplaced, supported by a [tweet](https://x.com/fentpot/status/1771634407226446254?s=20) implying that the effects of regulation on a company like Apple are negligible.
- **Conversation About the Merits of FTC Lawsuits**: A member named twkillian shared views that the recent FTC lawsuits may questionably allege anti-competitive behavior but doubted that such behavior was intended to worsen the marketplace or disadvantage other products.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw">Tweet from Machine Learning Street Talk (@MLStreetTalk)</a>: We just dropped the show with @MinqiJiang and @MarcRigter and discuss the philosophy of whether it is possible, in principle and in practice to build a &#34;generalist agent&#34; in RL.</li><li><a href="https://x.com/fentpot/status/1771634407226446254?s=20">Tweet from el (@fentpot)</a>: @nagolinc @norabelrose @natolambert i’m sorry but the second order effects are so negligible here for a large company like Apple. most founders would be thrilled to get anywhere near Apple’s size even...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1221520762066964560)** (19 messages🔥): 

- **Insights from Anthropic's CEO**: Nathan Lambert highlighted an interview titled ["Anthropic CEO on Leaving OpenAI and Predictions for Future of AI"](https://www.youtube.com/watch?v=gAaCqj6j5sQ) discussing Dario Amodei's predictions on the AI industry for 2024 and beyond.
- **Reflecting on Early OpenAI Visions**: Nathan remarked that early OpenAI contributors had a clear vision of the compute trajectory in AI.
- **Exploring Interview Content**: Nathan Lambert expressed an interest in Mistral's CEO, Arthur Mensch, for potential insights into company culture, directing attention to a ["Fireside Chat w/ Mistral CEO, Arthur Mensch"](https://youtu.be/sQpeIuymJZ8).
- **Defining AGI amid Debate**: The chat touched on the difficulty of defining AGI, with Nathan stating a personal threshold that includes GPT-4 as AGI, sparking a discussion on what constitutes true general intelligence.
- **February's AI Highlights with Latent Space**: Xeophon shared a link to Latent Space's monthly recap for February 2024 which covers significant AI news and upcoming events, including "AI UX 2024" and references the rapid user growth of ChatGPT in early 2023. The recap can be found [here](https://www.latent.space/p/feb-2024).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.latent.space/p/feb-2024">The Unbundling of ChatGPT (Feb 2024 Recap)</a>: Peak ChatGPT? Also: our usual highest-signal recap of top items for the AI Engineer from Feb 2024!</li><li><a href="https://youtu.be/sQpeIuymJZ8">Fireside Chat w/ Mistral CEO, Arthur Mensch</a>: Join us to hear from Arthur Mensch, Co-founder &amp; CEO of Mistral, in conversation w/ Elad Gil.​​Topics covered will include:​Open source &amp; LLMs​Agents and mul...</li><li><a href="https://www.youtube.com/watch?v=gAaCqj6j5sQ&t=5s">Anthropic CEO on Leaving OpenAI and Predictions for Future of AI</a>: Dario Amodei is the Co-Founder and CEO of Anthropic. In the episode, we discuss detailed predictions on the AI industry for 2024, 2025, and beyond. Dario dis...
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1220609180185002036)** (42 messages🔥): 

- **The Perils of "You Up?" in AI**: A user humorously highlighted the complex prerequisites for a chatbot to respond to the question "you up?" suggesting it should first solve the space-time continuum and ensure a secure connection.
- **Deciding Which Chain to Use**: There was a conversation about using multiple chains for a task, with a member suggesting that the decision of whether to query a SQL database or a vector database should be based on expected result sizes.
- **Technical Struggles with RunnableParallel and Streamlit**: A member encountered a "missing ScriptRunContext" Error when attempting to use RunnableParallel with a streamlit app, indicating possible compatibility issues between the two.
- **Launching a Learning Platform for RAG**: A user shared a link to an upcoming free resource to learn Retrieval-Augmented Generation (RAG) for programming with AI, mentioning OpenAI, LangChain, Chroma, and Python as some of the technologies participants will work with. [Intro to AI for Developers](https://takehomes.com/library/developers/intro-to-ai)
- **Vector Database Choices and Clustering Algorithms for Information Grouping**: A user sought advice on choosing between ChromaDB and Qdrant for vector databases and between density-based or centroid-based clustering algorithms for semantic-based cluster grouping of key information in documents.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://takehomes.com/library/developers/intro-to-ai">A Practical Introduction to AI for Developers – TakeHomes Library</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/6138">ConversationChain default prompt leads the model to converse with itself · Issue #6138 · langchain-ai/langchain</a>: System Info langchain==0.0.195 python==3.9.6 Who can help? @hwchase17 Information The official example notebooks/scripts My own modified scripts Related Components LLMs/Chat Models Embedding Models...</li><li><a href="https://github.com/docker/for-mac/issues/6938">The HEALTHCHECK flag &quot;start-interval&quot; is not recognized in Docker version 24.0.5, build ced0996 · Issue #6938 · docker/for-mac</a>: Description The docker documentation indicates that HEALTHCHECK has a flag called &quot;start-interval&quot; (docs) Actually using that flag in a Dockerfile causes an error Reproduce Use this Dockerfi...</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/tool_error_handling#chain>).">Tool error handling | 🦜️🔗 Langchain</a>: Using a model to invoke a tool has some obvious potential failure modes. Firstly, the model needs to return a output that can be parsed at all. Secondly, the model needs to return tool arguments that ...</li><li><a href="https://github.com/langchain-ai/langchain/issues/10629>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/4197>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/12410>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13602>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1221422519953657958)** (1 messages): 

- **Inquiring about Client-Side Execution with Langserve**: A member asked if it's possible to use a langserve-hosted runnable with tools that are executed on the client's side. There were no further details or responses provided.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1220692626458673193)** (12 messages🔥): 

- **Innovative Way to Extend LLM Output Beyond Limits**: A suggestion was made about a workaround to surpass the 4k output token limit of GPT-4-Turbo by detecting the stop reason as "length" and sending a follow-up request with the original and generated prompt, allowing for continued generation.
  
- **Bedrock and Python Integration Guide**: A comprehensive guide on leveraging Bedrock in combination with Python has been introduced. Those interested can access the full article [here](https://medium.com/@leonardo.bolanos/leveraging-bedrock-anthropic-haiku-with-python-a-comprehensive-guide-9f5e912982be).
  
- **Announcing SimplyAnalyze for LLMs Analytics**: SimplyAnalyze.ai was presented, a service that integrates with LangChain to analyze LLM conversations across various company departments. The creators shared contact information for those interested in their free developer preview and you can [get in touch through their website](https://simplyanalyze.ai/).

- **Exploring Agent Tree Search with Langchain**: An informative post has been shared about using Langchain to improve decision-making with Language Models. You can read the full article [here](https://medium.com/ai-advances/language-agent-tree-search-with-langchain-revolutionizing-decision-making-with-language-models-a46c991397f1).
  
- **Langchain-based Chatbot with Enhanced Capabilities**: A local character AI chatbot was updated, featuring improvements in CSV parsing, NER parsing, web scraping, and document fetching. Access the repository to explore the enhancements [on GitHub](https://github.com/ossirytk/llama-cpp-chat-memory).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Haste171/langchain-chatbot">GitHub - Haste171/langchain-chatbot: AI Chatbot for analyzing/extracting information from data in conversational format.</a>: AI Chatbot for analyzing/extracting information from data in conversational format. - Haste171/langchain-chatbot</li><li><a href="https://github.com/ossirytk/llama-cpp-chat-memory">GitHub - ossirytk/llama-cpp-chat-memory: Local character AI chatbot with chroma vector store memory and some scripts to process documents for Chroma</a>: Local character AI chatbot with chroma vector store memory and some scripts to process documents for Chroma - ossirytk/llama-cpp-chat-memory
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1221153829005824121)** (5 messages): 

- **Discover LangGraph Control over Chatbots**: A YouTube video titled "How To Control Your Chatbot Actions and Prompt System: LangGraph" was shared, demonstrating ways to build an agent within Langchain and automate your chatbot experience. The video is available at [How To Control Your Chatbot Actions and Prompt System: LangGraph](https://www.youtube.com/watch?v=4e5A3opn-tc).
- **Mr. Beast Adventures into AI Cookbooks**: A creative YouTube video "Mr. Beast Meets Mistral: AI Created a Cookbook Based on His Wildest Stunts!" was shared, showcasing an AI-generated cookbook inspired by the popular YouTuber's stunts. Check out the entertaining concept at [Mr. Beast Meets Mistral](https://www.youtube.com/watch?v=Nc5Yk0XXgP8).
- **Spam Alert**: Multiple identical messages offering a $50 steam gift card were posted, potentially indicating spam activity. The accompanying link was [steamcommunity.com/gift/758474483](https://u.to/uMaEIA).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=4e5A3opn-tc">How To Control Your Chatbot Actions and Prompt System: LangGraph</a>: #langgraph #langchain #ai #chatbot #python #automation if you are keeping up with Agent in Langchain, you know there are many ways to build an agent, but in ...</li><li><a href="https://www.youtube.com/watch?v=Nc5Yk0XXgP8">Mr. Beast Meets Mistral: AI Created a Cookbook Based on His Wildest Stunts!</a>: Today we create Beast CookbookThe &quot;Beast Cookbook&quot; idea is a fun and creative way to engage with Mr. Beast&#39;s content and generate an entertaining, fictional ...
</li>
</ul>

</div>
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1220818264251174993)** (21 messages🔥): 

- **Trouble in Real Estate AI-Land**: A member from *Uniti AI* is struggling with **GPT4.Turbo** to accurately match property inventory based upon user requirements, mentioning issues such as suggesting a property of 17,000 square feet when the request was for 2,000 - 4,000.

- **LLM's Role in Filtering**: The member's current approach involves using LLM to match properties from a CSV file with specified criteria. The detailed prompt aims to ensure that **inventory suggestions** remain within the specified requirements, with a variance up to +/- 20%.

- **Simple Solutions Over Complication**: Another member suggested using a simple database filter instead of an LLM, pointing out that an LLM can generate the query but is not necessary for the actual filtering process.

- **The Common LLM Trap Avoided**: In response to feeling unintelligent for the oversight, the member seeking help was reassured that falling into the "common LLM trap" happens and isn't a reflection of their capabilities.

- **Useful Resources Linked**: They provided a link to an instructional blog post by **Jason Liu**: ["RAG is more than just embedding search"](https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/) that discusses the limitations of embedding search and the applicability of LLMs in generating queries and handling natural language interactions.

**Link mentioned**: <a href="https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/">RAG is more than just embedding search - Instructor</a>: no description found

  

---


**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1221464274879119452)** (5 messages): 

- **Chafing Under Anthropic's Rate Limits**: One member expressed frustration about **Anthropic's** strict rate limits, citing a **200k context window** but only allowing **1M tokens per day** on the API.
- **Looking for Bedrock's Financial Ease**: The same member inquired about **Bedrock's monthly fee model** for guaranteed throughput, seeking insights from anyone with experience using the service.
- **Anthropic's Scale Plan Provides Relief**: Another member suggested contacting Anthropic's sales team for access to their "scale" plan, noting a reasonable monthly spend of **$500** for what was referred to as a relatively low cost.
  

---


**LLM Perf Enthusiasts AI ▷ #[resources](https://discord.com/channels/1168579740391710851/1168760058822283276/1220915143987433524)** (3 messages): 

- **Hunt for the Ultimate Guides**: A user is compiling a resource guide and is seeking the community’s **favorite explainer resources** on advanced topics related to Large Language Models (LLMs).
- **Exa.ai Endorsement**: In response to the call for resources, exa.ai was suggested as a useful tool for exploring LLM-adjacent topics.
- **Clarification on Resource Depth**: The user clarified their request by saying they are searching for the best, most clear explainers on topics like RHLF, beyond just a compilation of numerous blog posts or articles.
  

---


**LLM Perf Enthusiasts AI ▷ #[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/)** (1 messages): 

ibash: > write high quality code
Damn.
  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1220914049970339870)** (1 messages): 

- **GPT-3.5-0125 Outshines Its Predecessors**: A member highlighted that **GPT-3.5-0125** significantly outperforms previous models in all their tests, marking it as a distinctly superior iteration.
  

---


**LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/)** (1 messages): 

emrgnt_cmplxty: Basic prompting isn't getting it done for you?
  

---



**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1221887967409213612)** (1 messages): 

- **Volunteers Needed for Groundbreaking ML Research Project**: The **Youth Inquiry Network** and **Futracode** are collaborating to develop a machine learning algorithm that will recommend the best research topics by utilizing existing research databases. They are seeking web developers, data analysts, and AI & ML experts to contribute to this ambitious endeavor.

- **Contribute for Recognition and Experience**: Volunteers will have the chance to boost their portfolios and receive a certification, two professional recommendation letters, and the source code from the project. The engagement promises flexible scheduling and does not require a time-intensive commitment.

- **Work Now, Own Later**: Participants in this non-profit initiative will retain full rights to the developed ML algorithm, including the freedom to display, promote, sell, or use it however they see fit after the project's completion.

- **No Red Tape to Get Involved**: Interested individuals can directly express their interest without the need for a formal application—simply by messaging the recruiter or by commentating "interested" to be contacted for further steps.
  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1220643702200860704)** (8 messages🔥): 

- **Brief Communication Interchange**: The phrase *life lesson* prompted a humorous empathetic response from another member, indicating a shared communal understanding or incident.
- **Link to Educational Document**: A member shares a potentially informative [Google Docs link](https://docs.google.com/document/d/1f-CHZudw3ZOGFIk-Kov3QHkPjjR-Sh4mMmxcExgnWUk/edit?usp=sharing) regarding **Post-AGI Educational Reforms**; however, no further details or context is provided.
- **Query for DPO-P Training Code**: A member inquires if anyone possesses a code for **DPO-P training**, with no further elaboration on what DPO-P entails or the application of such code. 
- **Moment of Self-Awareness**: In a playful turn of self-realization, a member recognizes their own mod status after initially calling for moderation.
- **Recruiting Volunteers for ML Project Collaboration**: A call-to-arms is issued for **coders, data analysts, AI, and ML specialists** to volunteer on a machine learning project aimed at suggesting research topics, with incentives like certifications, recommendation letters, and the freedom to use the resulting code. Interested individuals are invited to directly message the initiator with the word "interested".

**Link mentioned**: <a href="https://docs.google.com/document/d/1f-CHZudw3ZOGFIk-Kov3QHkPjjR-Sh4mMmxcExgnWUk/edit?usp=sharing">Post-AGI Educational Reforms </a>: no description found

  

---


**Alignment Lab AI ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/1221887773313863742)** (1 messages): 

- **Non-Profits Seek Tech Talent for Groundbreaking ML Project**: The "Youth Inquiry Network" and "Futracode" are collaborating to create a Machine Learning algorithm that suggests the best research topics by training on research databases. They're looking for web developers, data analysts, AI & ML specialists to join this endeavor.

- **Volunteer Work with Tangible Perks**: The project is a volunteer effort aimed to benefit students struggling to find research topics. Contributors will receive a boost to their portfolio, certification, and recommendation letters from the founders of the two non-profits.

- **Contribute Code for a Cause**: Volunteers will retain the code source for personal growth, experience enhancement, and even the freedom to sell their contributions following project completion. The work schedule is flexible, tailored to fit the contributors' availability.

- **No-Strings-Attached Application Process**: Interested individuals can join the project by directly messaging or commenting "interested," with no formal application form required.
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1220948701175091200)** (5 messages): 

- **Clarifying `llm` and `ollama` Difference**: A member explained that **`llm` interfaces with models** but doesn't execute them like `ollama`. `llm` can be set up to use API endpoints served by `ollama`, which executes models locally and makes them available as local HTTP API endpoints.
- **Understanding Mistral Model Execution**: Inquiring about **Mistral model execution**, a member received clarification that using the Mistral model with `llm` implies it's running locally, but through the **HTTP API endpoints provided by `ollama`** or the `llm-llama-cpp` plugin that can run local models without HTTP.
- **Appreciation for AI-Powered Git Commit Helper**: A member shared their continuous use of **[AICommits (GitHub - Nutlope/aicommits)](https://github.com/Nutlope/aicommits)**, a tool for writing git commit messages with AI assistance, while expressing a desire for features like emoji standards for commits.

**Link mentioned**: <a href="https://github.com/Nutlope/aicommits">GitHub - Nutlope/aicommits: A CLI that writes your git commit messages for you with AI</a>: A CLI that writes your git commit messages for you with AI - Nutlope/aicommits

  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1221326221430882404)** (2 messages): 

- **AI Gets Culinary**: A member shared a [YouTube video](https://www.youtube.com/watch?v=Nc5Yk0XXgP8) titled **"Mr. Beast Meets Mistral: AI Created a Cookbook Based on His Wildest Stunts!"**. The video discusses how an AI created a cookbook inspired by the stunts of YouTuber Mr. Beast.
- **Seeking German DL/AI Content**: A member asked the group for recommendations on **deep learning/AI podcasts or video series in German**. They expressed an interest in engaging with content in said language.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=Nc5Yk0XXgP8">Mr. Beast Meets Mistral: AI Created a Cookbook Based on His Wildest Stunts!</a>: Today we create Beast CookbookThe &quot;Beast Cookbook&quot; idea is a fun and creative way to engage with Mr. Beast&#39;s content and generate an entertaining, fictional ...

  


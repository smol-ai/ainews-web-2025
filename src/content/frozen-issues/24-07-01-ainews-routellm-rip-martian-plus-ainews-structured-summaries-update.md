---
id: b9088584-8ef0-45e9-9390-6a5c1547cfc2
title: 'RouteLLM: RIP Martian? (Plus: AINews Structured Summaries update)'
date: '2024-07-02T00:23:08.479510Z'
original_slug: ainews-to-be-named-5628
description: >-
  **LMSys** introduces RouteLLM, an open-source router framework trained on
  **preference data** from Chatbot Arena, achieving **cost reductions over 85%
  on MT Bench, 45% on MMLU, and 35% on GSM8K** while maintaining **95% of
  GPT-4's performance**. This approach surpasses previous task-specific routing
  by using syntax-based Mixture of Experts (MoE) routing and data augmentation,
  beating commercial solutions by 40%. The update highlights advances in **LLM
  routing**, **cost-efficiency**, and **model performance optimization** across
  multiple models rather than single-model or MoE-level improvements.
  Additionally, the AI Twitter recap notes the **Gemma 2 model family** as a top
  open model, the **Block Transformer architecture** for improved inference
  throughput, and a proposal for a fully Software 2.0 computer vision system by
  **karpathy**.
companies:
  - lmsys
  - openai
models:
  - gpt-4
  - gemma-2-27b
  - gemma-2-9b
topics:
  - llm-routing
  - cost-efficiency
  - model-performance
  - model-optimization
  - data-augmentation
  - syntax-based-routing
  - mixture-of-experts
  - inference-throughput
  - software-2.0
  - computer-vision
people:
  - karpathy
  - bindureddy
  - armand-joulin
---


<!-- buttondown-editor-mode: plaintext -->**LLM Preference data is all you need.**

> AI News for 6/28/2024-7/1/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**419** channels, and **6896** messages) for you. 
Estimated reading time saved (at 200wpm): **746 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Remember [the Mistral Convex Hull of April](https://buttondown.email/ainews/archive/ainews-mixtral-8x22b-instruct-defines-frontier/), and then [the DeepSeekV2 win of May?](https://buttondown.email/ainews/archive/ainews-deepseek-v2-beats-mixtral-8x22b/). The cost-vs-performance efficient frontier is being pushed out again, but not at the single-model or MoE level, rather *across* all models:

 ![image.png](https://assets.buttondown.email/images/15826a27-7536-4595-9b00-e75ad39574f7.png?w=960&fit=max) 

The headline feature to note is this sentence: We trained four different routers using public data from Chatbot Arena and demonstrate that they can significantly reduce costs without compromising quality, with **cost reductions of over 85% on MT Bench, 45% on MMLU, and 35% on GSM8K** as compared to using only GPT-4, while still achieving **95% of GPT-4’s performance**. 

The idea of LLM routing isn't new; [model-router](https://x.com/withmartian/status/1641884426161520640) was a featured project at the "Woodstock of AI" meetup in early 2023, and subsequently raised a sizable [$9m seed round](https://www.businessinsider.com/martian-ai-startup-pitchdeck-seed-vc-funding-nea-prosus-2023-12) off that concept. However these routing solutions were based off of **task-specific** routing, the concept that different models are better at different tasks, which stands in direct contrast with **syntax-based** MoE routing.

LMSys' new open source router framework, [RouteLLM](https://github.com/lm-sys/RouteLLM), innovates by using **preference data** from The Arena for training their routers, based on predicting the best model a user prefers conditional upon a prompt. They also use **data augmentation** of the Arena data to further improve their routing benefits:

 ![image.png](https://assets.buttondown.email/images/6ac3c6bf-5899-4141-974f-9ac867341cc9.png?w=960&fit=max) 

Perhaps most brutally, LMSys claim to beat existing commercial solutions by 40% for the same performance.

 ![image.png](https://assets.buttondown.email/images/463c1ea5-b1f7-4545-84eb-f1455b8659ad.png?w=960&fit=max) 

> **SPECIAL AINEWS UPDATE: Structured Summaries**
>
> We have revised our core summary code to use structured output, focusing on achieving 1) better topic selection, 2) separation between fact and opinion/reation, and 3) better linking and highlighting. You can see the results accordingly. We do see that they have become more verbose, with this update, but we hope that the structure makes it more scannable, our upcoming web version will also be easier to navigate.

![image.png](https://assets.buttondown.email/images/d09386ee-eed1-4cd4-9918-cc0cc92c8618.png?w=960&fit=max)

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

**AI Models and Architectures**

- **Gemma 2 model family**: [@armandjoulin](https://twitter.com/armandjoulin/status/1807412150144491910) noted Gemma 2 27B is now the **best open model while being 2.5x smaller than alternatives**, validating the team's work. [@bindureddy](https://twitter.com/bindureddy/status/1807485457048953010) said Gemma 2 27B approaches Llama 3 70B performance, and Gemma 2 9B is beyond that Pareto front with excellent post-training.
- **Block Transformer architecture**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1807551317759455559) shared a paper showing **up to 20x improvement in inference throughput with Block Transformer compared to vanilla transformers** with equivalent perplexity, by reducing KV cache IO overhead from quadratic to linear. It isolates expensive global modeling to lower layers and applies fast local modeling in upper layers.
- **Fully Software 2.0 computer vision**: [@karpathy](https://twitter.com/karpathy/status/1807497426816946333) proposed a **100% fully Software 2.0 computer with a single neural net and no classical software**. Device inputs directly feed into the neural net, whose outputs display as audio/video.

**AI Agents and Reasoning**

- **Limitations of video generation models**: [@ylecun](https://twitter.com/ylecun/status/1807497091964449266) argued video generation models do not understand basic physics or the human body. [@giffmana](https://twitter.com/giffmana/status/1807511985807908926) was annoyed to see AI leaders use a clunky gymnastics AI video to claim human body physics is complicated, like showing a DALL-E mini generation to say current image generation is doomed.
- **Q* for multi-step reasoning**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1807561583079367086) shared a paper on Q*, which guides LLM decoding through deliberative planning to improve multi-step reasoning without task-specific fine-tuning. It formalizes reasoning as an MDP and uses fitted Q-iteration and A* search.
- **HippoRAG for long-term memory**: [@LangChainAI](https://twitter.com/LangChainAI/status/1807466126097650112) shared HippoRAG, a neurobiologically inspired long-term memory framework for LLMs to **continuously integrate knowledge**. It enriches documents with metadata using Unstructured API.

**AI Applications**

- **AI for legal workflows**: [@scottastevenson](https://twitter.com/scottastevenson/status/1807540320982433945) noted **agents are coming for legal workflows**, tagging @SpellbookLegal.
- **AI doctor from Eureka Health**: [@adcock_brett](https://twitter.com/adcock_brett/status/1807444895902138368) shared that Eureka Health introduced Eureka, the "first AI doctor" offering **personalized care 90x faster than most US care** according to early tests. 
- **AI-generated Olympic recaps**: [@adcock_brett](https://twitter.com/adcock_brett/status/1807444918337519673) reported NBC will launch **10-minute AI-generated recaps for the 2024 Olympics**, cloning Al Michaels' voice to narrate highlights on Peacock. The demo is hardly distinguishable from human-made content.

**Memes and Humor**

- [@nearcyan](https://twitter.com/nearcyan/status/1807557363840520355) joked "im going to become the joker what is wrong with you all" in response to AI sparkle icons.
- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1807443823259230276) humorously suggested if you cannot generate 6-7 birds, you are socially dead, as 4-5 are validated but 0-5 and 2-5 are too hard while 5-5 is too easy.
- [@kylebrussell](https://twitter.com/kylebrussell/status/1807462686566826356) shared an image joking "you deserve to have opponents who make you feel this way".

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**Humor/Memes**

- **AI-generated humorous videos**: In /r/StableDiffusion, users shared AI-generated videos with [humorous content](https://v.redd.it/7boa86nsbu9d1), such as a man opening a box of chocolates that explodes in his face. Comments noted the dream-like quality and uncanny movement in AI videos, drawing parallels to human brain processing.
- **AI combining memes**: Another [AI-generated video](https://v.redd.it/t3wzai9b0g9d1) combined various memes, which users found entertaining and a good use case for AI.
- **Superintelligent AI memes**: Image macro memes were shared depicting the [relationship between humans and superintelligent AI](https://i.redd.it/9ww15hhkst9d1.png), and a person [trying to control an advanced AI](https://i.redd.it/rgkq6s1mrt9d1.png) with a remote. Another meme showed [relief at an AI saying it won't kill humans](https://i.redd.it/04v5eb7isq9d1.png).

**AI Art**

- **Vietnam War with demons**: In /r/StableDiffusion, [AI-generated images](https://www.reddit.com/gallery/1dsfnje) depicted a fictional Vietnam War scenario with marines battling demons, inspired by the horror art styles of Beksinski, Szukalski and Giger. Users shared the detailed prompts used to create the images.
- **Blåhaj through time**: A [series of images](https://imgur.com/a/T6ELspY) showed the IKEA stuffed shark toy Blåhaj in various historical and futuristic settings.
- **1800s kids playing video games**: An AI-generated [video from Luma Dream machine](https://v.redd.it/dqdvym45lu9d1) depicted children in the 1800s anachronistically playing video games.

**AI Scaling and Capabilities**

- **Kurzweil's intelligence expansion prediction**: In a [Guardian article](https://www.theguardian.com/technology/article/2024/jun/29/ray-kurzweil-google-ai-the-singularity-is-nearer), AI scientist Ray Kurzweil predicted that AI will expand intelligence a millionfold by 2045.
- **Scaling limits of AI**: A [YouTube video](https://www.youtube.com/watch?v=ZyMzHG9eUFo) explored how far AI can be scaled.
- **Model size vs data quality**: A [post](https://i.redd.it/16iuw5kemu9d1.png) in /r/LocalLLaMA suggested that sometimes a smaller 9B model with high-quality data can outperform a 2T model on reasoning tasks, sparking discussion on the relative importance of model size and data quality. However, commenters noted this is more of an exception.
- **Processor performance plateau**: An [image](https://i.redd.it/7xi5yqjy3r9d1.jpeg) of a processor performance graph with a logarithmic y-axis was shared, suggesting performance is not actually exponential and will plateau.

**AI Models and Benchmarks**

- **Gemma 2 9B model**: In /r/LocalLLaMA, a user made an [appreciation post](https://www.reddit.com/r/LocalLLaMA/comments/1drxhlh/gemma_2_9b_appreciation_post/) for the Gemma 2 9B model, finding it better than Llama 3 8B for their use case.
- **Llama 400B release timing**: Another [post](https://www.reddit.com/r/LocalLLaMA/comments/1drw01y/400b_llama3_might_not_be_impactful_if_not/) discussed the potential impact of Llama 400B, suggesting it needs to be released soon to be impactful. Commenters noted a 400B model is less practical than ~70B models for most users.
- **Gemma2-27B LMSYS performance**: [Discussion](https://www.reddit.com/r/LocalLLaMA/comments/1ds2tv9/gemma227b_outperforms_both_qwe272b_and_llama370b/) around Gemma2-27B outperforming larger Qwe2-72B and Llama3-70B models on the LMSYS benchmark, questioning if this reflects real capabilities or LMSYS-specific factors.
- **Llama 400B speculation**: [Speculation](https://www.reddit.com/r/LocalLLaMA/comments/1ds2p09/llama_400_released_internally_at_meta_available/) that Meta may have internally released Llama 400B and made it available on WhatsApp based on an alleged screenshot.
- **Llama 3 405B release implications**: An [image](https://i.redd.it/cvzbd8cfwp9d1.png) suggesting the release of Llama 3 405B could spur other big tech companies to release powerful open-source models.
- **Gemma-2-9B AlpacaEval2.0 performance**: The [UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3 model](https://huggingface.co/UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3) achieved a 53.27% win rate on the AlpacaEval2.0 benchmark according to its Hugging Face page.

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **Model Training, Quantization, and Optimization**:
   - **[Adam-mini Optimizer](http://arxiv.org/abs/2406.16793): Saves VRAM by 45-50%.** Achieves performance akin to AdamW without the excessive memory overhead, useful for models like **llama 70b** and **GPT-4**.
   - **Hugging Face's new [low-precision inference](https://github.com/huggingface/diffusers/discussions/8746)** boosts transformer pipeline performance. Aimed at models like **SD3** and **PixArt-Sigma**, it improves computational efficiency.
   - **[CAME Optimizer](https://arxiv.org/abs/2307.02047): Memory-efficient optimization.** Shows better or comparable performance with reduced memory need, beneficial for stable diffusion training.

2. **New AI Models and Benchmarking**:
   - **[Gemma 2](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF)** demonstrates mixed performance but shows potential against models like Phi3 and Mistral, pending further optimization.
   - **[Claude 3.5](https://youtu.be/B45s_qWYUt8?si=_c7sQUFUN6bZa61m)** faces contextual retention issues despite high initial expectations; alternative models like Claude Opus perform reliably.
   - **[Persona Hub](https://arxiv.org/abs/2406.20094)** leverages diverse data applications to skyrocket MATH benchmark scores, proving synthetic data's efficacy in broader AI applications.

3. **Open-Source AI Tools and Community Engagement**:
   - **[Rig Library](https://bit.ly/Rig-Feeback-Form)**: Integrates fully with Cohere models, aimed at Rust developers with $100 feedback rewards for insights.
   - **[LlamaIndex](https://t.co/YsYoVOIirb)** introduces its best Jina reranker yet and provides a comprehensive tutorial for hybrid retrieval setups, promising advancements in retrieval pipelines.
   - **[Jina Reranker](https://t.co/cTxW2UwuZ0)**: A new hybrid retriever tutorial details combining methods for better performance, allowing integration with tools like Langchain and Postgres.

4. **Technical Challenges and Troubleshooting**:
   - **[BPE Tokenizer Visualizer](https://screenrec.com/share/SV7cw9vryx)** helps understand tokenizer mechanics in LLMs, inviting community feedback to refine the tool.
   - **Database Queue Issues** plague Eleuther and Hugging Face models' benchmarking efforts, urging users to look at alternatives like **[vllm](https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm)** for better efficiency.
   - **[Training GPT Models](https://github.com/tinygrad/tinygrad/pull/5159/files#diff-91ca5e2e75ef3ea1982c8ca6cc175ee88f20efa0d8e4b96f305b970dc6df71e7R291)** across multiple systems: Discussions emphasized handling GPU constraints and optimizing scales for effective resource usage.

5. **AI in Real-World Applications**:
   - **[Featherless.ai](https://featherless.ai)** launches to provide serverless access to LLMs at a flat rate, facilitating easy AI persona application development without GPU setups.
   - **[Deepseek Code V2](https://openrouter.ai/models/deepseek/deepseek-coder)** highly praised for its performance in solving complex calculus and coding tasks efficiently.
   - **[Computer Vision in Healthcare](https://youtu.be/QIjB4tqLqcs?si=bHfDg3WuRn5rEYbs)**: Exploring agentic hospitals using CV, emphasizing compute resources integration to enhance patient care and reduce administrative workloads.

---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Protein Prediction Precision**: New Frontiers with ESM3-SM-open-v1**: Evolutionary Scale unleashes **ESM3-SM-open-v1**, a transformative model for predicting protein sequences, underpinned by a sophisticated understanding of biological properties. Its utility is amped by the [GitHub repository](https://github.com/evolutionaryscale/esm) and an interactive [Hugging Face space](https://huggingface.co/spaces/as-cle-bert/proteins-with-esm), a synergy poised to further biological research.
   - Momentum gathers as practitioners are beckoned to leverage the model, crafting research pathways over at [Hugging Face](https://huggingface.co/spaces/as-cle-bert/proteins-with-esm), with the community already branding it as a **refreshing** leap in biological AI applications.
- **Aesthetic Essence Extracted**: Unveiling Datasets for Enhanced Model Refinement**: **Terminusresearch** curates a trove of visual data, their [photo-aesthetics dataset](https://huggingface.co/datasets/terminusresearch/photo-aesthetics) with 33.1k images, for honing the aesthetic discernment of AI. This dataset, brimming with real photographs, sets the stage for nuanced model training.
   - Additional sets capturing images of architectures and people engaged with objects complement the principal dataset, as this fledgling endeavor shows promise in advancing models' abilities to navigate and interpret the visual domain with an aesthetically attuned eye.
- **Tokenization Tamed**: BPE Depictions Emerge for LLM Clarity**: The BPE Tokenizer, a cornerstone in LLM workings, gains greater transparency with a novel [visualizer](https://github.com/mdabir1203/BPE_Tokenizer_Visualizer-) crafted by a vigilant community contributor. This deployable utility is set to clarify tokenizer mechanics, augmenting developer fluency with LLM intricacies.
   - Crowdsourced improvement efforts are in flight, as calls echo for feedback to rectify issues and finesse the visualizer—an initiative slated to enrich LLM accessibility.
- **Visionary Venues for Healing**: Charting Agentic Hospitals with CV**: A clarion call resounds for **computer vision** to pioneer **agentic hospitals**—a blend envisioned by a Sydney and Hyderabad doctor, aiming to curb administrative hefty loads. **Fisheye cameras** are slated to serve as the linchpin in orchestrating smooth operations and patient-centric care enhancements.
   - The pursuit of computational gifts surges as a plea is extended for compute resources, heralding a potential revolution where AI could transform healthcare dynamics, fostering tech-powered medical ecosystems.
- **Inference Ingenuity**: Embracing Low-Precision Aim in Transformers**: Exploration into low-precision inference within **transformer pipelines** like **SD3** and **PixArt-Sigma** signals a shift towards computational economy. A [GitHub discourse](https://github.com/huggingface/diffusers/discussions/8746) unveils the technique's promise of boosting the alacrity of model performance.
   - While ripe with the potential to optimize, this emerging approach beams with challenges that need judicious solving to unlock its full spectrum of benefits.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **RAM-tough Configs**: Unsloth Sails on WSL2**: Engineers recommended optimizations for running Unsloth AI on WSL2 to utilize full system resources. Configurations such as setting memory limits in `.wslconfig` and using specific install commands were exchanged for performance gains on various hardware setups.
   - Troubleshooting tips to address installation obstacles were shared, with consensus forming around memory allocation tweaks and fresh command lines as keys to unleashing Unsloth's efficiency on both Intel and AMD architectures.
- **DRM's Dual Role**: Multi-GPU Support Meets Licensing in Unsloth**: Unsloth's inclusion of a DRM system for multi-GPU training outlined strict NDA-covered tests that ensure GPU ID pinning and persistent IDs, revealing a behind-the-scenes effort to marry licensing control with functional flexibility.
   - Community chatter lit up discussing the configurations and boundaries of multi-GPU setups, with updates on the DRM's stability being pivotal to scaling AI training capacities.
- **Fine-Precise Feats**: Navigating Fine-Tuning Mechanisms with Unsloth**: AI enthusiasts dissected the mazes of fine-tuning Lexi and Gemma models noting specific quirks such as system tokens, troubleshooting endless generation outputs post-fine-tuning, and emphasizing bold markdown syntax for clarity.
   - Shared techniques for finessing the fine-tuning process included multi-language dataset translation, careful curation to avert critical forgetting, and using tokenizer functions fittingly to line up with custom training data.
- **Synthetic Personas Shaping Data**: Billion-Dollar Boost in MATH**: The Persona Hub's billion-persona-strong methodology for generating synthetic data has been a hot topic, presenting a quantum leap in mathematical reasoning scores that stoked buzz about the ease-of-use presented in its [abstract](https://arxiv.org/abs/2406.20094).
   - Amid a volley of perspectives on the project, some stressed the weight of the assembled data over the creation code itself, sparking debate on the critical elements of large-scale synthetic data applications.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemma's GPU Gambits**: While Gemma 2 GPU offloads to **Cuda** and **Metal**, updates beyond *0.2.23* are crucial to squash issues plaguing this AI models' performance, according to community feedback. [Deep exploration](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) revealed Gemma 2 struggles with lengthy model loads on specific setups; the AI gurus implored continuous patch works.
   - One cogent point raised was Gemma 2's current fencing in by **Cuda** and **Metal**, with suggested future broadening of horizons. Community dialogues surfaced detailing the technical challenges and positing potential updates to bolster model robustness and compatibility.
- **AutoUpdater's Agile Advancements**: LM Studio's **AutoUpdater** sleekly slides in, vaulting users ahead to **v0.2.26** with simmering anticipation for **v0.3**. A [public post](https://lmstudio.ai) conveys enhancements aimed squarely at the **LLama 3 8B model**, curing it of its troublesome ignoring of stop sequences - a much-welcome modification.
   - The conversation surrounding recent feature releases revolved around their capacity to straighten out past sore spots, such as a refusal to heed stop sequences. Users exchanged *active voice* approval of updates that promise a harmonious human-model interaction.
- **Deepseek's CPU Chronicles**: Deepseek v2 flaunts its prowess, flexing only **21B of its 200B+ parameters** on powerful CPUs, clocking **3-4 tokens/sec** on meaty **Threadripper systems**. Caseloads of user-generated performative data sink into community dialogues, grounding claims about CPU feasibility.
   - **Shared User Testing** meandered through a maze of metrics from RAM usage to loading speeds, encapsulating experiences with Deepseek v2's muscular but curiously well-managed performance on top-tier CPUs. The robust evaluations aim to steer the ship for future development and utility in practical settings.
- **Trials of Quantization Quests**: [Quantization queries](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md) dominated discussion, with a particularly piquant focus on 'GGUF quants' known for dancing deftly between performance and efficiency. Experimental tangoes with **q5_k_l** sparked analytic anecdotes aiming for a blend of briskness and lightness in resource consumption.
   - With a bevy of benchmarked brainchildren like **q5_k_l** and **GGUF quants**, the conclave of AI crafters probed for precision in performance while pressing for preservation of prowess. Documented discussions dial in on data and user feedback, pursuing the pinnacle of practice for these pioneering performance enhancers.
- **Smooth Sailing with Sliding Window**: The freshest chapter in Gemma 2's saga stars the **Sliding Window Attention**, freshly fused into the latest **llama.cpp**. This savvy update sanctions the AI to skillfully sift through past tokens, dexterously deepening its contextual comprehension. Users await with baited breath for [fixes](https://github.com/ggerganov/llama.cpp/pull/8227) to lingering quality quandaries.
   - As the bold **Sliding Window Attention** feature debuts in the Gemma 2 scene, bringing with it the promise of enhanced performance by adeptly managing token history, eclectic anecdotal evidence of its success emerges from the virtual corridors. Yet, spattered within this hope, voices of caution remain tenacious, precariously positioning expectations for comprehensive future felicitations.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Loras In Limbo with SD3**: A spirited discussion centered on the challenges in **training Loras** for **Stable Diffusion 3**, focusing on the need for robust support. Enthusiasm is building for the potential ease of **SD 8b**.
   - While some are eager to start, a cautionary tone prevailed urging patience for better **training tools** and data, to prevent substandard creations of Loras or checkpoints.
- **GPU Gurus Guide Newcomers**: Debate flourished regarding the **hardware requirements** for running Stable Diffusion, with the common consensus leaning towards **Nvidia GPUs** with a minimum of **12GB VRAM**.
   - Notably, existing RTX 3090 cards were commended, while the latest RTX 4080 and 4090 were highlighted for their future-proofing attributes despite their hefty price tags.
- **Installation Inception**: Users banded together to tackle **installation issues** with Stable Diffusion, sharing knowledge about various interfaces like **Automatic1111** and **ComfyUI**, along with key setup commands.
   - Helpful resources and guides were traded, including specific configuration advice such as incorporating 'xformers' and 'medvram-sdxl' to enhance performance of complex workflows.
- **High-Res Hacking for Sharper Art**: The community dove into the use of **high-resolution fix settings** in SDXL to achieve crisper images, underscoring the importance of exact parameter settings, like '10 hires steps'.
   - Participants amplified the benefits of plugins such as **adetailer**, highlighting its ability to refine critical aspects of imagery, particularly faces and eyes in anime-style graphics.
- **Model Mining and Loras Lore**: Conversations unearthed sources for finding **models and Loras**, naming platforms like **Civitai** for their extensive collections and user contributions.
   - Insight was shared on the validity of using prompt examples as a guide to accurately exploit these assets, accentuating the collective effort in model training and distribution.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Synchronizing Watches: Time-Zone Tools Tackle Scheduling Tangles**: Two tools, **[Discord Timestamps](https://r.3v.fi/discord-timestamps/)** and **[When2meet](https://www.when2meet.com/)**, were shared to **streamline meeting coordination** across time-zones.
   - Both **Discord Timestamps** and **When2meet** ease scheduling woes by converting times and pooling availability, fostering **effortless collective scheduling**.
- **Stable Footing on Log Scale's Numerical Grounds**: Discussion on the **log exp function's** role in **numeric stability** was stimulated by a [blog post](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/), emphasizing its use to **prevent underflow** in calculations.
   - **Debate ensued** over frequency of necessity, with questions raised on the indispensability of log scales in **modelling** and **ML**, accentuating the **divide in practices**.
- **Tensor Tales: Metadata Woes in the PyTorch Workshop**: Challenge to retain **metadata** in `torch.Tensor` through operations posted, with suggestions like **subclassing** posited, yet no definitive solution surfaced.
   - **Compiled quandaries** surfaced with `torch.compile` given constraints in supporting **dynamic input shapes**, layering complexity on **HuggingFace transformers** usage, with proposed yet unused solutions.
- **Flashing Insight: AI Engineers Laud Notable Speaker Series**: Endorsements filled the chat for the enlightening **
   - **Stephen Jones' delivery** earns accolades for consistently **insightful content**, reinforcing his **reputation in AI engineering** circles.
- **Performance Peaks and Valleys: Traversing Kernel Landscapes with FP16**: Kernel performance became a hot topic with **FP16** showing a **single launch efficiency** versus **bfloat**'s multiple, sparking strategies for optimizing **big tensor operations**.
   - **Bitpacking optimization** shows promise on smaller tensors but wanes with bulkier ones, prompting **further exploration** in **kernel performance** enhancement.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemma's Glorious Launch**: Gemma 2 Models Hit Perplexity Labs**: The **Gemma 2 models** have landed on [Perplexity Labs](https://labs.pplx.ai), aiming to spur community engagement through user feedback on their performance.
   - Enthusiastic participants can now test-drive the new models and contribute insights, as the announcement encourages interactive community experience.
- **Chatty Androids**: Push-to-Talk vs Hands-Free Modes**: Perplexity AI's Android app is now boasting a **voice-to-voice feature**, enhancing user accessibility with hands-free and push-to-talk modes, as detailed in their latest update.
   - The hands-free mode initiates immediate listening upon screen activation, contrasting with push-to-talk which awaits user command, aiming to enrich user interaction.
- **Claude's Context Clumsiness**: Users Decry Forgetfulness**: Reports surge of **Claude 3.5** failing to cling to context, diverging into generalized replies despite the chatter on specific topics, challenging engineers with unexpected twists in conversations.
   - Switches to the **Opus** model have shown improvements for some, hinting at potential bugs in **Claude 3.5** affecting engagement, with community calls for smarter Pro search to preserve context.
- **API Vexations**: Discrepancies Emerge in Perplexity's API**: Perplexity API users grapple with inconsistencies, spotting that date-specific filters like `after:2024-05-28` may lure the API into crafting forward-dated content, sparking a debate on its predictive prowess.
   - Feedback bubbles as a user's interaction with the Perplexity Labs Playground hits a snag due to Apple ID recognition issues, sparking dialogues on user inclusivity and experience refinement.
- **Gaming the System**: Minecraft Mechanics Misinform Minors?**: A fiery thread unfolds critiquing [Minecraft's Repair Mechanics](https://www.perplexity.ai/page/Minecraft-Repair-Mechanics-NdRggXKXRXyGY8LgKsp1dQ), suggesting the in-game logic might twist youngsters' grasp on real-world tool restoration.
   - The digital debate digs into the educational impact, urging a reality check on how virtual environments like Minecraft could seed misunderstandings, prompting programmers to ponder the implications.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Billion Persona Breakthrough: Skyrocketing Math Benchmarks**: Aran Komatsuzaki's tweet highlighted the successful creation of a [billion persona dataset](https://x.com/arankomatsuzaki/status/1807593343007818065), pushing MATH benchmark scores from 49.6 to 64.9. The method behind it, including the diverse data applications, is detailed in the [Persona Hub GitHub](https://github.com/tencent-ailab/persona-hub) and the corresponding [arXiv paper](https://arxiv.org/abs/2406.20094).
   - The new dataset enabled synthetic generation of **high-quality mathematical problems** and NPC scenarios for gaming. This innovation demonstrates significant performance gains with synthetic data, providing numerous use cases for academic and entertainment AI applications.
- **Dream Data Duality: The Android & Human Dataset**: The dataset contrasts 10,000 legitimate dreams with 10,000 synthesized by the **Oneirogen model**, showcased on [Hugging Face](https://huggingface.co/datasets/gustavecortal/the-android-and-the-human). Oneirogen's variants, 0.5B, 1.5B, and 7B, offer a new standard for dream narrative assessment.
   - The corpus is available for discerning the differences between authentic and generated dream content, paving the way for advanced classification tasks and psychological AI studies.
- **Tech Giants' Mega Ventures: Microsoft & OpenAI's Data Center**: Microsoft revealed collaboration with OpenAI on the Stargate project, potentially infusing over $100 billion into the venture as reported by [The Information](https://www.theinformation.com/articles/microsoft-and-openai-plot-100-billion-stargate-ai-supercomputer). The companies aim to address the growing demands for AI with significant processing power needs.
   - This initiative could shape the energy sector significantly, considering Microsoft's nuclear power strategies to sustain such extensive computational requirements.
- **Speculative Speed: SpecExec's LLM Decoding Innovation**: SpecExec offers a brand new method for LLM inference, providing [speed increases](https://www.together.ai/blog/specexec) of up to 18.7 times by using 4-bit quantization on consumer GPUs. This breakthrough facilitates quicker LLM operations, potentially streamlining AI integration into broader applications.
   - The model speculatively decodes sequences, verified speedily by the core algorithm, stoking discussions on compatibility with different LLM families and integration into existing platforms.
- **Charting the Phylogeny of LLMs with PhyloLM**: PhyloLM's novel approach introduces phylogenetic principles to assess the lineage and performance of LLMs, as seen in the [arXiv report](https://arxiv.org/abs/2404.04671). The method crafts dendrograms based on LLM output resemblance, evaluating 111 open-source and 45 closed-source models.
   - This method teases out performance characteristics and relationships between LLMs without full transparency of training data, offering a cost-effective benchmarking technique.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Errors as Values Spark Design Discussion**: Discourse in Mojo uncovered nuances between handling **errors as values** versus traditional exceptions. The discussion revealed a preference for **Variant[T, Error]**, and the need for better **match statements**.
   - Contributors differentiated the use of try/except in Mojo as merely sugar coating, suggesting deeper considerations in language design for elegant error resolution.
- **Ubuntu Users Unite for Mojo Mastery**: A collaborative effort emerged as users grappled with setting up **Mojo on Ubuntu**. Success stories from installations on **Ubuntu 24.04** on a **Raspberry Pi 5** showcased the communal nature of troubleshooting.
   - Dialogues featured the significance of community support in surmounting setup struggles, particularly for newcomers navigating different Ubuntu versions.
- **Marathon Mojo: Coding Challenges Commence**: [Monthly coding challenges](https://discord.com/channels/1087530497313357884/1255303604894437388) have been initiated, providing a dynamic platform for showcasing and honing skills within the Mojo community.
   - The initiative, driven by **@Benny-Nottonson**, focuses on practical problems like optimized **matrix multiplication**, with detailed participation instructions on the [GitHub repository](https://github.com/Benny-Nottonson/Mojo-Marathons).
- **AI Aspirations: Cody Codes with Mojo**: The intersection of Cody and Mojo sparked interest, with discussions on using Cody to predict language features. The Python-like syntax paves the way for streamlined integration.
   - With aspirations to explore advanced Mojo-specific features such as SIMD, the community is poised to push the boundaries of what helper bots like Cody can achieve.
- **Asynchronous I/O and Systemic Mojo Strengths**: Engagement soared with conversations on the I/O module's current constraints. Members advocated for **async APIs** like `io_uring`, aiming for enhanced network performance.
   - **Darkmatter__** and **Lukashermann.com** debated the trade-off between powerful but complex APIs versus user-friendly abstractions, emphasizing the need for maintainability.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter's Oopsie**: Analytics on **OpenRouter** went offline due to a **database operation mistake**, as stated in the [announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1257248877497552948). Customer data remains secure and unaffected by the mishap.
   - The team promptly assured users with a message of regret, clarifying that customer credits were not compromised and are addressing the data fix actively.
- **DeepSeek Code Charms Calculus**: Praise was given to **DeepSeek Code V2** via OpenRouter API for its impressive accuracy in tackling calculus problems, shared through the [general channel](https://discord.com/channels/1091220969173028894/1094454198688546826/1256356791814455306). Economical and effective attributes were highlighted.
   - It was confirmed that the model in use was the **full 263B** one, suggesting considerable power and versatility for various tasks. Details available on [DeepSeek-Coder-V2's page](https://openrouter.ai/models/deepseek/deepseek-coder).
- **Mistral API Mix-Up**: A report surfaced of a **Mistral API error** while using **Sonnet 3.5** on Aider chat, causing confusion among users who were not employing Mistral at the time.
   - Users were directed to contact Aider's support for specific troubleshooting, hinting at an automatic fallback to Mistral during an outage. Details discussed in the [general channel](https://discord.com/channels/1091220969173028894/1094454198688546826/1256356791814455306).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Adept's Amazon Alignment**: Adept AI Labs announced strategic updates and leadership changes, with co-founders heading to Amazon, detailed in their [blog post](https://www.adept.ai/blog/adept-update) and a [GeekWire article](https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/). This move aims for Adept to use Amazon's technology under a non-exclusive license while maintaining its independence.
   - Community reflections shed light on the confusion caused by the initial blog post, leading to a discussion around the **partnership's nature with Amazon**, prompting readers to prefer the [GeekWire article](https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/) for clearer insights.
- **Fair Feedback on AIEWF**: Organizers and attendees of the AI Engineer World's Fair (AIEWF) engaged in feedback sessions to discuss improving session lengths and logistical aspects, drawing from [GitHub discussions](https://github.com/swyxio/swyxdotio/issues/510) about lessons learned and future planning.
   - Suggestions included extending the event or conducting more structured feedback, with a call for a designated space for hackathons, inspired by other conferences’ success in fostering in-depth discussions.
- **Runway's Video Generation Voyage**: Runway released their Gen-3 Alpha Text to Video feature, a major advancement in high-fidelity and controllable video generation, announced on their [official account](https://x.com/runwayml/status/1807822396415467686).
   - Open to everyone, the feature promises a significant leap in video generation technology, accessible via Runway's [website](http://runwayml.com), sparking curiosity and experimentation among creators.
- **Privacy in Prompt Planning**: Discussions emerged concerning the privacy of GPT system prompts, with an emphasis on treating prompts as potentially public information, referencing examples found on [GitHub](https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts).
   - Advice circulated to avoid including sensitive data in GPT system definitions, with community backing the notion by suggesting a cautious approach to what is shared in prompts.
- **CoALA Paper Piques Interest**: The community discussed the new paper on Cognitive Architectures for Language Agents (CoALA), which can be found on [arXiv](https://arxiv.org/abs/2309.02427), introducing a framework for organizing existing language agent models.
   - A repository of language agents, [awesome-language-agents](https://github.com/ysymyth/awesome-language-agents), based on the CoALA paper, became a highlighted resource for those looking to delve deeper into the study of language agents.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Web Chronicles**: React & FastAPI Unite**: A guild member inquired about integrating **LangGraph** with a React frontend and a FastAPI backend, receiving directions to [chat-langchain on GitHub](https://github.com/langchain-ai/chat-langchain) and [Semantic Router docs](https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb).
   - They were advised on creating an agent or leveraging semantic similarity for routing, using outlined methods for a solid foundation in tool implementation.
- **Embeddings Nesting Games**: Speeding Through with Matryoshka**: Prashant Dixit showcased a solution for boosting retrieval speeds using **Matryoshka RAG** and **llama_index**, detailed in a [Twitter post](https://x.com/Prashant_Dixit0/status/1806580075447590974) and a comprehensive [Colab tutorial](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/tutorials/RAG-with_MatryoshkaEmbed-Llamaindex/RAG_with_MatryoshkaEmbedding_and_Llamaindex.ipynb).
   - The technique employs varied embedding dimensions (768 to 64), promising enhanced performance and memory efficiency.
- **Automate Wise, Analyze Wise**: EDA-GPT Revolution**: Shaunak announced **EDA-GPT**, a GitHub project for automated data analysis with **LLMs**, deployable via **Streamlit**, and aided by a video tutorial for setup found in the [project's README](https://github.com/shaunthecomputerscientist/EDA-GPT).
   - This innovation streamlines the data analysis process, simplifying workflows for engineers.
- **Postgres Meets LangChain**: A Match Made for Persistence**: Andy Singal's [Medium post](https://medium.com/ai-advances/unleashing-the-power-of-persistence-langchain-meets-postgres-9cc7f069b260) highlights the merging of **LangChain** and **Postgres** to optimize persistence, bringing the reliability of Postgres into LangChain projects.
   - The synergistic pair aims to bolster state management with Postgres' sturdiness in storage.
- **Casting MoA Magic with LangChain**: A YouTube tutorial titled ["Mixture of Agents (MoA) using langchain"](https://www.youtube.com/watch?v=VNy7CM23WA0) walked viewers through the process of creating a multi-agent system within LangChain, aiming to amplify task performance.
   - The video provided an entry point to MoA, with specific code examples for engineering audiences interested in applying combined agent strengths.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Adept in Amazon's Ambit**: **Adept** shifts strategy and integrates with **Amazon**; co-founders join *Amazon*, as detailed in their [blog post](https://www.adept.ai/blog/adept-update). The company, post-departure, operates with about 20 remaining employees.
   - This strategic move by **Adept** resembles **Microsoft's strategy** with Inflection AI, sparking speculations on **Adept's direction** and **organizational culture** changes, as discussed in [this tweet](https://x.com/anissagardizy8/status/1806812006009442671?s=46).
- **AI Mastery or Mystery?**: Debate ensues on whether **AI agent development** is lagging, paralleling **early-stage self-driving cars**. Projects like **Multion** are criticized for minimal advancements in capabilities beyond **basic web scraping**.
   - Community conjectures that innovative **data collection methods** are the game changer for **AI agents**, with a pivot to generating **high-quality, model-specific data** as a crucial piece to surmount current limitations.
- **Cohere's Clever Crack at AI Clarity**: **Cohere's CEO**, **Aidan Gomez**, shares insights on combating **AI hallucinations** and boosting **reasoning power** in a [YouTube discussion](https://youtu.be/B45s_qWYUt8?si=qs1u6p7wiXFP46PT), hinting at the potential of synthetic data generation.
   - The community compares these efforts to **Generative Active Learning** and the practice of *hard negative/positive mining* for LLMs, echoing the significance at **5:30** and **15:00** marks in the video.
- **Model Value Conundrum**: The **$1B** valuation of user-rich **Character.ai**, contrasted with **Cognition AI's** non-operational **$2B** valuation, incites discussions around **pitching strength** and **fundraising finesse**.
   - **Cognition AI** waves the flag of its founders' **IMO accolades**, targeting developer demographics and facing scrutiny over their merit amidst fierce competition from **big tech AI entities**.
- **Layperson's RL Leap**: An AI aficionado wraps up **Silver's RL intro** and **Abeel's Deep RL**, aiming next at **Sutton & Barto**, scouting for any unconventional advice with an eye towards **LM alignment**.
   - The RL rookies get tipped to skim **Spinning Up in Deep RL** and dabble in real **code bases**, performing hands-on *CPU-powered tasks* for a grounded understanding, as guided possibly by the **HF Deep RL course**.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Cross-OS Setup Sagas**: Members report friction setting up `01` on **macOS and Windows**, despite following the foreseen steps with hurdles like **API key** dependency and terminal command confusions. [@killian](https://www.reddit.com/r/LocalLLaMA/comments/1dl3a13/killian_showed_a_fully_local_computercontrolling/) and others have floated potential solutions, amidst general consensus on setup snags.
   - Consistent calls for clarity in documentation echo across trials, as highlighted by a thread discussing **desktop app redundancy**. A GitHub pull request may be the beacon of hope with promising leads on simplified Windows procedures laid out by [dheavy](https://github.com/OpenInterpreter/01/pull/203).
- **Amnesic Assistants? Addressing AI Forgetfulness**: An inquiry into imbuing **Open Interpreter** with improved **long-term memory** has surfaced, pinpointing pain points in **Sonnet** model's ability to learn from past interactions.
   - Discourse on memory enhancements signifies a collective trial with the **OI memory constraints**, yet remains without a definitive advanced strategy despite suggestions for specific command usage and esoteric pre-training ventures.
- **Vector Search Pioneers Wanted**: A hands-on tutorial on vector search integration into public datasets has been showcased in a [Colab notebook](https://colab.research.google.com/github/onefact/loving-the-baseline/blob/main/nearest-neighbors.ipynb) by a proactive member, setting the stage for a cutting-edge presentation at the Fed.
   - The collaborator extends an olive branch for further **vector search enhancement ventures**, heralding a potential new chapter for community innovation and applied AI research.
- **Multimodal Model Hunt Heats Up**: Queries on selecting top-notch open-source multimodal models for censored and uncensored projects bubbled up, prompting suggestions like **Moondream** for visual finesse paired with **robust LLMs**.
   - The conversation led to fragmented views on model adequacy, reflecting a panorama of perspectives on multimodal implementation strategies without a unified winner in sight.
- **Windows Woes and Installation Wobbles**: Discontent bubbles over **typer installation troubles** on Windows for **OpenInterpreter**, with members finessing the pyproject.toml file and manipulating `poetry install` maneuvers for success.
   - A narrative of documentation woes weaves through the guild, amplifying the outcry for transparent, **up-to-date guidelines**, and inviting scrutiny over the practicality of their **01 Light** setups. [@Shadowdoggie](https://github.com/OpenInterpreter/01) spotlights the dichotomy between macOS ease and Windows woe.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Reranker Revolution**: Excitement builds with the **Jina reranker's** new release, claimed to be their most effective yet, detailed [here](https://t.co/YsYoVOIirb). The community praises its impact on retrieval strategies and **results combinatorics**.
   - The guide to build a custom hybrid retriever is welcomed for its thorough approach in combining retrieval methods, shared by **@kingzzm** and available [here](https://t.co/cTxW2UwuZ0). Feedback reveals its far-reaching potential for advanced **retrieval pipelines**.
- **LlamaIndex's Toolkit Expansion**: Integration uncertainties arise as users ponder over the compatibility of **Langchain Tools** with **LlamaIndex**, an inquiry sparked by a community member's question.
   - Discussion thrives around using **Langchain Tools** alongside **LlamaIndex agents**, with a keen interest on how to merge their functionalities for refined efficiency.
- **Query Quandaries Quelled**: Users grapple with **query pipeline configuration** in LlamaIndex, with insightful suggestions like utilizing kwargs to manage the **`source_key`** for improved input separation and retrieval setup.
   - Embedding performance concerns with large CSV files lead to a proposal to upscale **`embed_batch_size`**, broadening pathways to incorporate more substantial LLMs for better code evaluation.
- **Sub-Agent Specialization**: The curiosity around **sub-agents** manifests as users seek guidance to customize them using prompts and inputs to enhance task-specific actions.
   - **CodeSplitter** tool gains attention for its potential in optimizing metadata extraction, hinting at a shift towards more efficient node manipulation within the **LlamaIndex**.
- **Kubernetes & Multi-Agent Synergy**: The launch of a new **multi-agent systems deployment** starter kit by **@_nerdai_** paves the way for moving local agent services to Kubernetes with ease, found [here](https://t.co/wfcI0wSmFG).
   - The kit enables transitioning to k8s deployment without friction, marking a **significant stride in scaling** service capabilities.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llamas Leap with Less**: Community members are astounded by the **llama3 70b** model achieving high performance on a mere 27 billion parameters, triggering a discussion on the plausibility of such a feat. While some remain dedicated to **Mixtral** for its well-rounded performance and favorable licensing for consumer hardware.
   - Debate unfolds on **Hugging Face** regarding the **Hermes 2 Theta** and **Pro** iterations – one a novel experiment and the other a polished finetune – while users ponder the merits of structured JSON Outputs exclusive to the **Pro** version.
- **Formatting Flair and Frustrations**: Issues with **Axolotl's** custom ORPO formatter stirred discussions due to improper tokenization and how system roles are managed in ChatML.
   - Suggestions for using alternative roles to navigate the challenges were met with concerns about conflict, showcasing the need for more seamless customization solutions.
- **Synthetic Slowdown Sparks Speculation**: The **Nvidia synthetic model** under the microscope for its sluggish data generation, moving at a snail's pace compared to faster models like **llama 70b** or **GPT-4**.
   - This prompted queries on the advantages smaller models might hold, especially in terms of efficiency and practical application.
- **Cutting-Edge Compaction in Optimization**: AI enthusiasts probed into innovative memory-efficient optimizers like **CAME** and **Adam-mini** with the promise of reduced memory usage without compromising performance.
   - Technical aficionados were directed to [CAME's paper](https://arxiv.org/abs/2307.02047) and [Adam-mini's research](https://arxiv.org/abs/2406.16793) to dive into the details and potential application in areas like stable diffusion training.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Benchmarks Caught in Queue Quagmire**: Hot topics included the computation resource bottleneck affecting **leaderboard benchmark** queue times, speculated to stem from HF's infrastructure. **Stellaathena** implied that control over the queues was not possible, indicating a need for alternative solutions.
   - [@dimfeld](https://x.com/dimfeld/status/1806116419995844947) suggested **vllm** as an alternative, pointing to a [helpful wiki](https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm) for technical guidance on model memory optimization.
- **Im-Eval Ponders 'HumanEval' Abilities**: Questions arose over **im-eval's** capability to handle `HumanEval` and `HumanEvalPlus`, with **Johnl5945** sparking a discussion on configuring evaluation temperatures for the evaluation tool.
   - The conversation concluded without a firm resolution, highlighting a potential area for follow-up research or clarification on im-eval's functionality and temperature control.
- **Adam-mini: The Lightweight Optimizer**: The [Adam-mini optimizer](http://arxiv.org/abs/2406.16793) was a notable subject, offering significant memory savings by using block-wise single learning rates, promising comparable performance to AdamW.
   - Members evaluated its efficacy, recognizing the potential to scale down optimizer memory usage without impacting model outcomes, which could usher in more memory-efficient practices in ML workflows.
- **Gemma 2's Metrics Mystify Users**: Discrepancies in replicating **Gemma 2** metrics led to confusion, despite diligent use of recommended practices, such as setting `dtype` to `bfloat16`. Concerns arose over a substantial difference in reported accuracies across benchmarks like **piqa** and **hellaswag**.
   - Further probing into potential issues has been urged after the proper debugging commands seemed to return correct but inconsistent results, as reported in an informative [tweet by @LysandreJik](https://x.com/LysandreJik/status/1807779464849273343).
- **Token Representation 'Erasure Effect' Uncovered**: A recent [study](https://arxiv.org/abs/2406.20086) unveiled an 'erasure effect' in token representations within LLMs, particularly notable in multi-token named entities, stirring a vibrant discourse around its implications.
   - The academic exchange focused on how such an effect influences the interpretation of semantically complex token groups and the potential for enhanced model designs to address this representation challenge.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Codebase Cleanup Crusade**: George Hotz issued a call to action for streamlining **tinygrad** by moving **RMSNorm** from **LLaMA** into `nn/__init__.py`, complete with tests and documentation.
   - Community response involved, suggesting enhancements to the organization and potentially unifying code standards across the project.
- **Mondays with tinygrad**: During the latest meeting, participants discussed various recent advancements including [a sharding update](https://github.com/tinygrad/tinygrad/pull/5123) and a [single pass graph rewrite](https://github.com/tinygrad/tinygrad/pull/5159/files#diff-91ca5e2e75ef3ea1982c8ca6cc175ee88f20efa0d8e4b96f305b970dc6df71e7R291), touching on **tensor cores** and new bounties.
   - The detailed exchange covered **lowerer continuation**, the **Qualcomm runtime**, and identified the next steps for further improvements in the tinygrad development process.
- **Standalone tinygrad Showdown**: Queries emerged on the potential for compiling **tinygrad** programs into standalone C for devices like Raspberry Pi, with a shared interest in targeting low-power hardware.
   - Members shared resources like [tinygrad for ESP32](https://github.com/wozeparrot/tinygrad) to inspire pursuit of applications beyond traditional environments.
- **Bounty Hunting Brilliance**: A robust discussion clarified the requirements for the **llama 70b lora bounty**, including adherence to the [MLPerf reference](https://github.com/mlcommons/training/tree/master/llama2_70b_lora) with flexibility in the computational approach.
   - The community explored the possibility of employing qlora and shared insights on implementing the bounty across different hardware configurations.
- **Graph Rewrite Revelation**: The exchange on **graph rewrite** included interest in adopting new algorithms into the process, with a focus on optimizing the scheduler.
   - **ChenYuy**'s breakdown of the meeting noted while specific graph algorithms have yet to be chosen, there's momentum behind migrating more functions into the graph rewrite framework.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **27 Billion Big Doubts, Little Triumphs**: Amidst mixed opinions, the **27B model** faced skepticism yet received some recognition for potential, eclipsing **Command R+** in even its less promising scenarios. One member bolstered the conversation by noting **Gemma 2 9B**'s unexpectedly superior performance.
   - Enthusiasm tinged with doubt flowed as *"True but even in the worst case(-15) it's better than command r+"* was a shared sentiment indicating a lean towards the 27B's superior potential over its counterparts.
- **ChatGPT's Slippery Slope to Second-Rate**: Troubles were aired regarding the **ChatGPT 4 and 4o models** losing their grip on nuanced programming tasks, with comparisons drawn favorably towards the **3.5 iteration**. Several users felt the latest models were over-zealous, taking prompts too literally.
   - Frustration bubbled as a member's comment *"Sometimes the paid 4 and 4o models feel absolutely useless when programming"* captured the communal drift towards more reliable **free alternatives**.
- **Gemini's Rising, ChatGPT's Falling**: **Gemini 1.5 Pro** took the spotlight with commendations fir its responsive interaction, while **ChatGPT** faced complaints of growing inefficiency, especially in programming tasks. Users are applauding Gemini's can-do attitude in contrast to ChatGPT's waning enthusiasm.
   - Comparisons made by users like *"Gemini 1.5 pro does a super excellent job compared to chatgpt's increasing laziness"* highlight a shift towards alternative models that maintain their zest and engagement over time.
- **Claude Crafts Artifact Appeal**: **Claude's artifact feature** won over users, providing a more immersive and efficient experience that challenges the status quo established by **ChatGPT**. This particular feature has attracted a growing audience ready to switch allegiances.
   - Community consensus is reflected in statements like *"The artifacts feature is a much better experience"*, signaling a rising trend for enthusiast tools that resonate more closely with the user-experience expectations.
- **The Language Labyrinth & LLMs**: The discussion shifted toward a global audience as **non-English speakers** sought LLMs skilled in diverse languages, prioritizing conversational capabilities in their native tongues over niche task efficiency. This global tilt continues despite the varying effectiveness of the models in specific tasks.
   - The narrative *"It's already up there not because it can answer difficult platform, but due its multilingual capabilities"* showcases a surging demand for models that democratize language inclusivity and support localized user interactions.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **JSONL Jugglers Gather**: Efforts to simplify JSONL editing result in shared expertise, with [a simple JSONL editor](https://github.com/aastroza/simple-jsonl-editor) for quick edits and a scripting method maintained for diverse JSON-structured file manipulations.
   - Community exchange recommends direct editing and prompt engineering for extracting summaries from structured patient data in JSON format, avoiding new package implementations and minimizing the incidence of hallucinations in LLM evaluations.
- **Kv-Caching to Enlighten Vision-LLMs**: LLM users explore [kv-caching enhancements](https://sachinruk.github.io/blog/2024-06-29-kv-cache.html) for vision-centric models, finding valuable improvements in prediction probabilities.
   - The guide provides actionable optimization advice for vision models on constrained GPU setups, drawing practical interest and implementation feedback.
- **Navigating LLM Inference on Kubernetes**: Skepticism appears over ML inference implementations on **Kubernetes**, with a light-hearted tweet inciting in-depth discussions on cloud infrastructure alternatives for ML workloads.
   - Despite some [shared difficulties](https://github.com/modal-labs/llm-finetuning) with tool scaling on **Modal**, confidence is expressed in Modal over Kubernetes for specific distributed systems.
- **Hugging Face Credits: A Relief to Users**: Clarifications emerged on **Hugging Face credits**, confirming a 2-year validity period, easing users' concerns of any immediate credit expiration.
   - Discussions point to a community need for improved communication channels about credit status and management on **Hugging Face**.
- **In Pursuit of Optimal IDEs**: [Zed IDE](https://zed.dev) wins over a Sublime Text veteran with impressive features and AI integrations, yet curiosity about **Cursor's** offerings remains peaked.
   - Community feedback sought on user experiences with **Cursor**, suggesting a broader exploration of AI integration in development environments.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Cultivates Interns**: A community member keen on Cohere's internship mentioned their AAAI conference paper and past co-op, probing for **Cohere's API** advancements and novel LLM feature tasks.
   - Conversations evolved around resources for blending LLMs with reinforcement learning, enriching **Cohere's** technical landscape.
- **Coral's Cribs: The Rate Limit Riddle**: Frustrations sparked around **Coral's API** rate limits, with users chafing at a sparse 5 calls/minute constraint.
   - Insights were exchanged via a [practical guide](https://docs.cohere.com/docs/rate-limits), casting production keys in a hero's light with a lavish 10,000 calls/minute.
- **Mixed Messages Over Aya-23 Models**: The guild navigated a fog of versions for **Aya-23** models, spotlighting the 8B and 35B models found on [Hugging Face](https://huggingface.co/CohereForAI/aya-23-8B), while chasing phantom rumors of a 9B variant.
   - A consensus clarified no current application of these model versions for running inferences, reaffirming their adequacy.
- **Cohere's Climb to Clearer Cognition**: Members buzzed about **Cohere's** plans to curb AI hallucinations, following [Aidan Gomez's enlightenment](https://youtu.be/B45s_qWYUt8?si=_c7sQUFUN6bZa61m) on enhancing AI reasoning.
   - The CEO's roadmap refrained from external partnerships, instead spotlighting self-reliant development.
- **Rig Wrangles Cohere Compatibility**: Rust enthusiasts rejoiced as the **Rig** library announced full integration with **Cohere models**, encouraging feedback through a [rewarding review program](https://bit.ly/Rig-Feeback-Form).
   - Contributors can clinch a $100 honorarium by providing insights to refine Rig, making it a linchpin for LLM-powered projects.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Diving into DPO with Data-driven Precision**: Debate ensued on the benefits of **DPO training** for datasets with robust data, with some considering immediate use of **DPO/PPO** while others expressed **hesitation**. It was suggested to utilize **PreferenceDataset** for these applications.
   - The discussions highlighted that **Socket experts** should guide such decisions, referencing past successes of straight **DPO/PPO** training on **llama2** and **Pythia 1B-7B**.
- **WandB to the Rescue for Phi Mini Fine-tuning**: An AI enthusiast successfully fine-tuned a **Phi Mini (LoRA)** model and looked for guidance on **evaluating logs**. The consensus was to adopt **WandBLogger** for sophisticated log management and visualization.
   - Warnings were voiced about **yaml configuration** pitfalls and the importance of a well-set **WandBLogger** to prevent errors and gain enhanced training oversight was emphasized.
- **Fine-tuning Fineries: Logs & Gradient Governance**: Technical talk touched on the appropriateness of **gradient size** with suggestions tabled to tailor it to dataset specifics. Shared logs spurred scrutiny for signs of **overfitting** and discussion about extending **training epochs**.
   - Logs revealed irregularities in **loss and learning rate** metrics, especially within smaller datasets, underscoring the utility of tools like **WandB** for clarity in the fine-tuning foray.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Featherless Flies with Flat Fees**: The recent launch of **Featherless.ai** offers a subscription-based access to all LLM models available on Hugging Face, starting at $10/month without the need for local GPU setups as shown [here](https://featherless.ai/). The platform has seen an uptick in use for local AI persona applications like Sillytaven, along with needs such as language tuning and exploiting SQL models.
   - **Text-to-Speech (TTS) Temptation** emerges as **Featherless.ai** considers integrating TTS systems like [Piper](https://github.com/rhasspy/piper/blob/master/VOICES.md) based on growing user requests to enhance NPC voice diversity in online gaming, maintaining a focus on popular models unrunnable on local CPU setups.
- **Windows Wisdom with WSL README**: New member Niashamina brings **Windows wisdom** to the guild by creating a **README** for using **WSL** to get AI Town operational on Windows, mentioning the possibility of Docker integration in the near [future](https://github.com/).
   - While the integration into Docker is still underway and the README's draft awaits its GitHub debut, Niashamina quips about its eventual usefulness, hinting at the hands-on **Windows progress** they're pioneering.
- **Hexagen.World’s New Geographic Gems**: A brief but notable announcement unveils **fresh locations** available at [Hexagen.World](https://Hexagen.World), expanding the domain's virtual landscape offerings.
   - The reveal doesn't dive into details but plants seeds of curiosity for those inclined to explore the newly added virtual terrains, opening windows to **new localizations**.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Facebook Crafts Compiler Capabilities**: LLM Compiler Model**: Facebook released its **LLM Compiler model** with the prowess to compile C, optimize assembly, and LLVM IR, now conveniently packaged by Mozilla into [llamafiles for various operating systems](https://github.com/Mozilla-Ocho/llamafile).
   - The **llamafile**, supporting both AMD64 and ARM64 architectures, [has been uploaded by Mozilla to Hugging Face](https://huggingface.co/Mozilla/llm-compiler-13b-ftd-llamafile) to enhance accessibility for its users.
- **Llamafile Leaps to Official**: Integration Saga on Huggingface**: Aiming for the official status of llamafile on Huggingface, contributors are poised to create pull requests to update the [model libraries](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts) and corresponding [code snippets](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries-snippets.ts) files.
   - This integration will ease the user experience by adding a button on repositories using **llamafile**, and autofill code to load the model seamlessly.
- **Tech Specs Uncorked**: llamafile's Hardware Haunt**: Community discussions surfaced about the feasibility of running **llamafile** on varied devices; however, it requires a **64-bit system**, sidelining the Raspberry Pi Zero from the action.
   - While **llamafile server v2.0** impresses with its minimal memory footprint, using only 23mb for hosting embeddings to HTTP clients with **all-MiniLM-L6-v2.Q6_K.gguf**, iPhone 13 support remains unconfirmed.
- **llamafile v0.8.9 Ascends Android**: Gemma2 Gets a Grip**: **llamafile v0.8.9** strikes with its official Android compatibility and refined support for the Google Gemma2 architecture, alongside Windows GPU extraction fixes.
   - The [newly spun v0.8.9 release](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.9) also highlights advancements in server mode operations while underpinning Google Gemma v2 enhancements.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Zooming into AI Engineering Confusion**: An anticipated **'From ML Engineering to AI(cntEngr)' event's recording** turned into a digital goose chase, with members encountering [invalid Zoom links](https://events.zoom.us/ejl/AuK2GHiGErlhKuJWYRGND-S4fQfnvCgfPO50QMGDk0trhQ2ykP5H~A2Qw5SEBU-CKEpNn-eBw) and access code issues.
   - Despite the effort, the community couldn't retrieve the recording, highlighting the gaps in event-sharing infrastructure within the AI engineering space.
- **Pipeline Wizardry Workshop Wonders**: [Data Talks Club's upcoming Zoomcamp](https://lu.ma/cnpdoc5n?tk=uEvsB6) promises hands-on journey for AI engineers with focus on building **open-source data pipelines** using **dlt** and **LanceDB** scheduled for July 8.
   - With guidance by Akela Drissner from **dltHub**, participants will dive deep into REST APIs, data vectorizing, and orchestration tools, aiming to deploy pipelines across varying environments including Python notebooks and Airflow.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Pack Pulls Ahead in Database Dominance**: On the topic of database progression, **dbreunig** highlighted a noticeable trend starting from **May 19th**, showing competitors closing in on the lead in database technologies.
   - The commentary suggests a shift in the AI database landscape with multiple players improving and vying for the top position.
- **Catch-Up Clause in Computation**: In a recent insight by **dbreunig**, data since **May 19th** indicates a tightening race among leading database contenders.
   - This observation pinpoints a critical period where competing technologies began to show significant gains, catching up with industry leaders.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Coders in a Memory Jam**: An engineer's attempt at training a **BPE tokenizer for the Panjabi language** on a hefty 50 GB corpus hits a snag with an **OOM issue** on a 1TB RAM machine. They've shed light on the ordeal by sharing comparable [GitHub issues](https://github.com/huggingface/tokenizers/issues/1434).
   - Despite reaching beyond the **Pre-processing sequences steps continue beyond len(ds)**, memory consumption keeps soaring, hinting at a possible misfire in the `train_from_iterator` function, detailed in this [related issue](https://github.com/huggingface/tokenizers/issues/1345). Technical insights or alternative training methodologies are in high demand for this perplexing issue.
- **Debugging Dilemma: Delving into the Rusty Depths**: The quest to crack the OOM mystery during BPE tokenizer training leads one intrepid coder to a wall, as the `train_from_iterator` function in `tokenization_utils_fast.py` becomes an impenetrable fortress.
   - Speculations arise that the issue may stem from executable/binary Rust code, a theory supported by other community encounters, leaving our engineer scratching their head and seeking [expert aid](https://github.com/huggingface/transformers/blob/e65502951593a76844e872fee9c56b805598538a/src/transformers/tokenization_utils_fast.py#L817).



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1257065010312646818)** (1 messages): 

- **Predict protein sequences with ESM3-SM-open-v1**: A demo [space](https://huggingface.co/spaces/as-cle-bert/proteins-with-esm) for **ESM3-SM-open-v1** was shared for predicting sequences and structures of masked protein sequences by a verified user.
   - *Details on predicting masked protein sequences were highlighted in the demo space.*
- **BPE Tokenizer Visualizer is live**: A [visualizer](https://github.com/mdabir1203/BPE_Tokenizer_Visualizer-) for **BPE Tokenizer** created by another community member was introduced.
   - *The visualizer makes it easier to understand and work with BPE tokenization.*
- **Exciting new datasets by terminusresearch**: [People holding things datasets](https://huggingface.co/datasets/terminusresearch/photo-anatomy), [architectural datasets](https://huggingface.co/datasets/terminusresearch/photo-architecture), and [aesthetics datasets](https://huggingface.co/datasets/terminusresearch/photo-aesthetics) were released by terminusresearch.
   - *These datasets provide diverse options for research and development in various domains.*
- **Explore Fast whisper playground**: A fast [whisper playground](https://huggingface.co/spaces/Iatalking/fast-whisper-server) was shared by a community member for quick experimentation.
   - *Users can now test whisper models rapidly in this new interactive setup.*
- **Test results for Gemma 2 27B**: A YouTube video discussed [Gemma 2 27B](https://youtu.be/vIKNRiVxWeo) test results shared by a user.
   - *The video explores the performance of the latest Gemma version by Google.*

**Link mentioned**: <a href="https://youtu.be/vIKNRiVxWeo)">Gemma2:27B First Test ! How Can it be THAT Bad ?!</a>: Let&#39;s test the biggest version (27B) of the gemma2 release an hour ago by Google with ollama

  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1256323280126738562)** (952 messages🔥🔥🔥): 

- **Issues with Model Downloads and Caching**: A user faced problems downloading Falcon40B, noticing the model files vanished after the download. Turns out, model files are cached by HuggingFace and can be accessed using the `snapshot_download` function.
   - Another user recommended using a more efficient model like Llama 3 8B instead of Falcon 40B, suggesting it would yield better performance with less resource consumption.
- **Stuck Spaces on HuggingFace Platform**: A user's space was stuck on 'preparing' for over a day. Despite attempts to rebuild and restart, the issue persisted.
   - Moderator suggested restarting or pausing/unpausing the space to resolve the issue; however, the problem remained unresolved, requiring further support.
- **Legal LLM and RAG Implementation**: A user sought advice on building an LLM for legal advice, dealing with token size and hallucinations. Suggestions included using Retrieval-Augmented Generation (RAG) for more effective data utilization.
   - The discussion emphasized the importance of evaluating bulk generation outputs using an intermediary LLM for ranking quality, and avoiding frequent fine-tuning on potentially outdated legal data.
- **Audio Classification with Whisper and Wav2Vec**: Questions arose regarding the use of Whisper and Wav2Vec models for audio classification, particularly around segmenting audio recordings.
   - Advice was given to segment the audio into 15-30 second chunks for better model performance, and to use specific formatting when working with different operating systems.
- **Token Management in Gradio and Spaces Scripts**: A user needed help improving the security of a script that uses Gradio and HuggingFace Spaces for converting and pushing models to the hub.
   - Concerns were raised about securely managing HF tokens within the script, leading to suggestions for better exception handling and token resetting practices.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/xyplon/text-to-image-models-playground">Text To Image Models Playground - a Hugging Face Space by xyplon</a>: no description found</li><li><a href="https://huggingface.co/spaces/aheedsajid/Edge-TTS">Edge TTS - a Hugging Face Space by aheedsajid</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/models-download-stats">Models Download Stats</a>: no description found</li><li><a href="https://huggingface.co/dev-mode-explorers">dev-mode-explorers (Dev Mode Explorers)</a>: no description found</li><li><a href="https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents">Why we no longer use LangChain for building our AI agents</a>: When abstractions do more harm than good - lessons learned using LangChain in production and what we should’ve done instead</li><li><a href="https://sachinruk.github.io/blog/2024-06-29-kv-cache.html">Prompt Caching: Poor man’s guide to zero shot vision-LLM classification – deepschool.ai</a>: Using KV caching and logit ratios to speed up and control LLM/ VLM outputs.</li><li><a href="https://huggingface.co/posts">Hugging Face – Posts</a>: no description found</li><li><a href="https://huggingface.co/nerijs/pixel-art-xl">nerijs/pixel-art-xl · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/nroggendorff/finetune-mistral">Fine-tuning Mistral on Your Dataset</a>: no description found</li><li><a href="https://huggingface.co/VAP36/EltonJohn70s/resolve/main/Ej1979.zip">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/TheStinger/UVR-Test/discussions/1">TheStinger/UVR-Test · Update requirements.txt</a>: no description found</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/81">zero-gpu-explorers/README · Pro Account with ZeroGPU Space: &quot;You have exceeded your GPU quota (59s left vs. 60s requested). Please retry in 0:00:56&quot;</a>: no description found</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/models?pipeline_tag=image">Models - Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/cat-cat-memes-cat-images-cat-meme-gif-4644773688486402896">Cat Cat Memes GIF - Cat Cat memes Cat images - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.google.it/search?q=tensorflow+transformer">tensorflow transformer - Google Search</a>: no description found</li><li><a href="https://tenor.com/view/soldier-ww2-traumatized-meme-eyes-gif-12257475272172704406">Soldier Ww2 GIF - Soldier Ww2 Traumatized - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/drake-notebook-gif-20708336">Drake Notebook GIF - Drake Notebook - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/coBTzvQsPpQ?si=PgVtXG5hw_XgjpBX">RAG explained step-by-step up to GROKKED RAG sys</a>: Today I try to answer all questions by my subscriber about my last three videos, w/ focus on the new Grokked LLM integration into traditional RAG systems. I&#39;...</li><li><a href="https://github.com/huggingface/transformers/issues/31293>">Issues · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - Issues · huggingface/transformers</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE">LICENSE · meta-llama/Meta-Llama-3-8B at main</a>: no description found</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5">nomic-ai/nomic-embed-text-v1.5 · Hugging Face</a>: no description found</li><li><a href="https://www.tensorflow.org/text/tutorials/transformer#define_the_components">no title found</a>: no description found</li><li><a href="https://github.com/Vipitis/shadertoys-dataset">GitHub - Vipitis/shadertoys-dataset: WIP refactor of a dataset</a>: WIP refactor of a dataset. Contribute to Vipitis/shadertoys-dataset development by creating an account on GitHub.</li><li><a href="https://huggingface.co/nroggendorff/mayo-7b-it">nroggendorff/mayo-7b-it · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/carwash-car-cat-gif-13432273130992663014">Carwash Car Cat GIF - Carwash car cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/intone/Ammino-1.1B">intone/Ammino-1.1B · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Cf_aWv4LyUQam80gNHVRygNUtneovVmp#scrollTo=ua9MQA3WXCtr">Google Colab</a>: no description found</li><li><a href="https://tenor.com/bA1GjIzjUL.gif">Rammus Ok GIF - Rammus Ok Okay - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1256350504708083795)** (7 messages): 

- **Huggingface Course Update Status**: **Huggingface courses** seem to be regularly maintained and might offer updated content compared to the May 2022 edition of the NLP book. **Details on the Diffusion course and Community computer vision courses** are still unclear regarding their up-to-date status.
   - Users are advised to verify course content directly from the [Huggingface website](https://huggingface.co/course) to ensure they get the most recent updates and information.
- **Biometric Gait Recognition with 2D Video Cameras**: **Gait recognition using 2D cameras** achieved a 70% accuracy identifying individuals from 23 people based on single frames. The next steps involve finding more datasets, combining frames for RNN use, and training with **triplet loss** to generate embeddings.
   - Members interested in areas like **Triplet Collapse** and advanced gait recognition methods are encouraged to participate by direct messaging the author for collaboration and knowledge sharing.
- **Custom ML Library for Garmin Devices**: An exciting project is underway to **clone Apple's double tap feature on Garmin devices** using a custom ML library in Monkey C language. The project is in its initial stages, focusing on developing this feature tailored for Garmin's hardware.
   - Collaborators and experts in Monkey C or similar projects are invited to share insights or join the development process.
- **Understanding Tree-Sitter S-Expressions**: Exploring **Tree-sitter s-expressions** with resources like their [official documentation](https://tree-sitter.github.io/tree-sitter/using-parsers#pattern-matching-with-queries) and various binding libraries in different programming languages. Developers can use **node-tree-sitter** or the **tree-sitter Rust crate** for extended functionality.
   - The **Tree-sitter project** supports a variety of languages through C APIs and higher-level bindings, empowering users to integrate robust parsing functionalities into their applications.
- **Engaging Explainers of ML Concepts**: A member praised an **explanation of the attention module** given by Ashpun, highlighting the importance of such knowledge in **ML interviews**. Understanding these concepts can spark new ideas and improve large language models (LLMs).
   - Ashpun appreciated the feedback and emphasized the significance of clear, detailed explanations in the learning and application of ML concepts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">Top 10 Deep Learning Algorithms intro in 1 min</a>: Welcome to our deep dive into the Top 10 Deep Learning Algorithms! In this video, we break down each algorithm with a concise 10-word explanation. Perfect fo...</li><li><a href="https://tree-sitter.github.io/tree-sitter/using-parsers#pattern-matching-with-queries">Tree-sitter｜Using Parsers</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1256856303767130233)** (11 messages🔥): 

- **Kickstarter Project Fuses Anime and AI**: A Kickstarter project was shared that focuses on creating anime and manga artwork with AI. More details can be found on their [Kickstarter page](https://www.kickstarter.com/projects/mobinet-games/creating-anime-and-manga-artwork-with-ai).
   - The project promises innovative artwork generation techniques and could be an exciting development for AI-enabled creativity in anime and manga design.
- **Firecrawl Turns Websites into Data**: **Firecrawl** is an open-source tool that converts websites into clean markdown or structured data, with 500 free credits initially. They announced it celebrates [7000+ stars](https://github.com/mendableai/firecrawl) on GitHub.
   - The tool can crawl all accessible subpages without needing a sitemap, transforming data for LLM applications. Examples and features are elaborated on their [website](https://www.firecrawl.dev/).
- **Lora Model for SDXL Unveiled on HuggingFace**: A **Lora model** for sdxl was published on HuggingFace, featuring unique creature prompts in the style of TOK. Explore more creative outputs and prompts on [HuggingFace](https://huggingface.co/alvdansen/m3lt).
   - The model is appreciated by the community, with users praising the work and citing related models such as `alvdansen/BandW-Manga`.
- **Langchain Integrates with Postgres**: An article was shared discussing the persistence capabilities when integrating **Langchain** with **Postgres**. For an in-depth read, check this [Medium post](https://medium.com/ai-advances/unleashing-the-power-of-persistence-langchain-meets-postgres-9cc7f069b260).
   - This integration enables improved data management and retrieval, benefiting AI workflows that require robust database solutions.
- **AI Explores Mesh Generation Techniques**: A [YouTube video](https://www.youtube.com/watch?v=rQolOT4tuUY&ab_channel=IndividualKex) titled 'AI just figured out Meshes' details advancements in AI for mesh generation. The original paper and a demo on HuggingFace are linked.
   - The video explains the capabilities and applications in mesh generation, supported by [code](https://github.com/buaacyw/MeshAnything) and research for deeper exploration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/alvdansen/m3lt">alvdansen/m3lt · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=rQolOT4tuUY&ab_channel=IndividualKex">AI just figured out Meshes</a>: original paper: https://huggingface.co/papers/2406.10163demo: https://huggingface.co/spaces/Yiwen-ntu/MeshAnythingcode: https://github.com/buaacyw/MeshAnythi...</li><li><a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">Top 10 Deep Learning Algorithms intro in 1 min</a>: Welcome to our deep dive into the Top 10 Deep Learning Algorithms! In this video, we break down each algorithm with a concise 10-word explanation. Perfect fo...</li><li><a href="https://getdoks.org/">Build an amazing docs site</a>: Everything you need to build a stellar documentation website. Fast, accessible, and easy to use.</li><li><a href="https://mintlify.com/">Mintlify - The modern standard for documentation</a>: no description found</li><li><a href="https://www.firecrawl.dev/">Firecrawl</a>: Turn any website into LLM-ready data.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1256396266070474824)** (17 messages🔥): 

- **esm3-sm-open-v1 model for biological research**: [Evolutionary Scale](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) released the *esm3-sm-open-v1* model, described as a frontier generative model for biology, able to reason across three fundamental biological properties of proteins. The model is accessible on [GitHub](https://github.com/evolutionaryscale/esm) and can be tried via a user-built space on Hugging Face.
   - Users are encouraged to try the model for their research by accessing the space at [this link](https://huggingface.co/spaces/as-cle-bert/proteins-with-esm), described as 'refreshing' by the creator.
- **Publishing internal aesthetic datasets**: [Terminusresearch](https://huggingface.co/datasets/terminusresearch/photo-aesthetics) published one of their internal aesthetic datasets consisting of 33.1k real photograph images filtered for specific qualities from Pexels. The dataset aims to help in fine-tuning models for aesthetic judgments.
   - More components of the dataset, including images of people holding items and a small architectural set for regularization data, were also released to hopefully improve model performance in related tasks.
- **BPE Tokenizer Visualizer for LLMs**: A new [BPE Tokenizer Visualizer](https://github.com/mdabir1203/BPE_Tokenizer_Visualizer-) was created to help visualize how BPE Tokenizers work in LLMs. A demo of the visualizer is available [here](https://screenrec.com/share/SV7cw9vryx).
   - The creator seeks help with fixing issues and feedback from the community to improve the tool.
- **Run transformers on robotics hardware easily**: [Embodied Agents](https://github.com/mbodiai/embodied-agents) project enables running transformer models on robotics hardware with few lines of Python code. This tool is aimed at seamless integration into robotics stacks.
   - The GitHub page provides detailed instructions and example code for users to get started quickly.
- **Stable Cypher Instruct 3B model release**: [Stable Cypher Instruct 3B](https://huggingface.co/lakkeo/stable-cypher-instruct-3b) is a newly released 3B parameter model designed to outperform SoA models such as GPT4-o in generating CYPHER queries. It's a fine-tuned version of stable-code-instruct-3b, specifically trained on synthetic datasets from Neo4j Labs.
   - The model is accessible via Hugging Face and is aimed at facilitating text-to-CYPHER query generation for GraphDB databases such as Neo4j.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://marketing.startyparty.dev/">startyparty</a>: the start of it all</li><li><a href="https://huggingface.co/blog/alvdansen/enhancing-lora-training-through-effective-captions">Enhancing Image Model Dreambooth Training Through Effective Captioning: Key Observations</a>: no description found</li><li><a href="https://huggingface.co/lakkeo/stable-cypher-instruct-3b">lakkeo/stable-cypher-instruct-3b · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">Top 10 Deep Learning Algorithms intro in 1 min</a>: Welcome to our deep dive into the Top 10 Deep Learning Algorithms! In this video, we break down each algorithm with a concise 10-word explanation. Perfect fo...</li><li><a href="https://huggingface.co/blog/alvdansen/training-lora-m3lt">How I train a LoRA: m3lt style training overview</a>: no description found</li><li><a href="https://github.com/U-C4N/UMBOT">GitHub - U-C4N/UMBOT</a>: Contribute to U-C4N/UMBOT development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/as-cle-bert/proteins-with-esm">Proteins With Esm - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://github.com/mbodiai/embodied-agents">GitHub - mbodiai/embodied-agents: Seamlessly integrate state-of-the-art transformer models into robotics stacks</a>: Seamlessly integrate state-of-the-art transformer models into robotics stacks - mbodiai/embodied-agents</li><li><a href="https://huggingface.co/datasets/terminusresearch/photo-aesthetics">terminusresearch/photo-aesthetics · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/terminusresearch/photo-anatomy">terminusresearch/photo-anatomy · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/mdabir1203/BPE_Tokenizer_Visualizer-">GitHub - mdabir1203/BPE_Tokenizer_Visualizer-: A Visualizer to check how BPE Tokenizer in an LLM Works</a>: A Visualizer to check how BPE Tokenizer in an LLM Works - mdabir1203/BPE_Tokenizer_Visualizer-</li><li><a href="https://screenrec.com/share/SV7cw9vryx">24.05.2024_00.18.02_REC</a>: Recorded with ScreenRec
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1256355527202701332)** (30 messages🔥): 

- **Reasoning with LLMs intrigues members**: Members discussed the paper [Reasoning with LLMs](https://arxiv.org/pdf/2405.16506) which attracted interest for its usage of GNNs in adaptation to retrieval.
   - A meeting was organized to dive into [current research on reasoning with LLMs](https://github.com/atfortes/Awesome-LLM-Reasoning) and [symbolic reasoning](https://github.com/luban-agi/Awesome-LLM-reasoning) featuring [a detailed write-up](https://isamu-website.medium.com/understanding-the-current-state-of-reasoning-with-llms-dbd9fa3fc1a0) and a [YouTube presentation](https://www.youtube.com/watch?v=vbji1PvXgBc&ab_channel=IsamuIsozaki).
- **Discord technical issues lead to Zoom switch**: During a meeting, members faced technical difficulties with audio on Discord. As a workaround, the session was switched to [Zoom](https://drexel.zoom.us/j/86571034095) which resolved the issues.
   - Members speculated the problem might stem from client version mismatches but agreed that when functioning correctly, Discord is an excellent platform for calls.
- **Terminator architecture promises rapid training**: The [Terminator architecture's code](https://github.com/hyperevolnet/Terminator) was released, touted for its efficiency in training convergence.
   - Testing showed that Terminator can achieve adequate results within 50-100 epochs, which is significantly less than other architectures, promising faster training times.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=vbji1PvXgBc&ab_channel=IsamuIsozaki">Hugging Face Reading Group 24: Understanding Current State of Reasoning with LLMs</a>: Presenter: Isamu IsozakiPast Presentations: https://github.com/isamu-isozaki/huggingface-reading-group</li><li><a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">Top 10 Deep Learning Algorithms intro in 1 min</a>: Welcome to our deep dive into the Top 10 Deep Learning Algorithms! In this video, we break down each algorithm with a concise 10-word explanation. Perfect fo...</li><li><a href="https://github.com/hyperevolnet/Terminator">GitHub - hyperevolnet/Terminator: The official repository for HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction.</a>: The official repository for HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction. - hyperevolnet/Terminator</li><li><a href="https://github.com/atfortes/Awesome-LLM-Reasoning">GitHub - atfortes/Awesome-LLM-Reasoning: Reasoning in Large Language Models: Papers and Resources, including Chain-of-Thought, Instruction-Tuning and Multimodality.</a>: Reasoning in Large Language Models: Papers and Resources, including Chain-of-Thought, Instruction-Tuning and Multimodality.  - GitHub - atfortes/Awesome-LLM-Reasoning: Reasoning in Large Language M...</li><li><a href="https://drexel.zoom.us/j/86571034095">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://drexel.zoom.us/j/8657103">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1257175065590763662)** (1 messages): 

- **Low-Precision Inference for Transformers**: We investigated the use of low-precision inference for **transformer-based pipelines** such as **SD3** and **PixArt-Sigma**. Some interesting findings are discussed in this [GitHub thread](https://github.com/huggingface/diffusers/discussions/8746).
   - The discussion highlights potential performance improvements and challenges in implementing low-precision inference in these models.
- **Exploring Transformer Pipelines**: Detailed analysis of transformer-based pipelines like **SD3** and **PixArt-Sigma** was conducted. Insights and technical details are available in the [discussion thread](https://github.com/huggingface/diffusers/discussions/8746).
   - Key points include the benefits of low-precision inference and its impact on model performance and efficiency.
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1256491748583280763)** (12 messages🔥): 

- **Hand Gesture Drawing App Needs Improvements**: A user shared a [YouTube video](https://youtu.be/QIjB4tqLqcs?si=bHfDg3WuRn5rEYbs) demonstrating their project that **turns hand gestures into digital art** using **OpenCV** and **Mediapipe**. They asked for suggestions to make it better.
   - Suggestions on improving the app were sparse, but several members discussed the importance of **object size and convolutional layer configuration** in model performance.
- **ViTMAEForPretraining Setup Challenges**: Users discussed the use of **ViTMAEForPretraining** for custom images, with concerns about the technical challenges. One shared code indicating issues with **mask_ratio settings** and ripping out the encoder for inference.
   - The implementation seemed **janky** and uncertain, calling for more refined methods in model configuration during pretraining and inference.
- **Error in face_recognition Module Due to Numpy Version**: A discussion on errors in the **face_recognition** module hinted at issues caused by **numpy** version. A user suggested reverting to a version lower than 2.0 to fix the issue.
   - This was confirmed to resolve the problem, providing a quick workaround for others facing similar issues.
- **Building Agentic Hospitals with Computer Vision**: A doctor from Sydney and Hyderabad shared a vision for an **agentic hospital** using **computer vision** to reduce administrative burdens. They are targeting **fisheye cameras** in containers to streamline operations and improve patient care.
   - They requested compute resources to help bring this vision to life, highlighting the need for efficient and technology-driven healthcare solutions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/QIjB4tqLqcs?si=bHfDg3WuRn5rEYbs">🎨 Hand Gesture Drawing App Demo - Creating Art in Progress! 🖐️✍️</a>: Watch me try out my latest project that turns hand gestures into digital art using OpenCV and Mediapipe. Join me as I create ( or try to ) something  in real...</li><li><a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">Top 10 Deep Learning Algorithms intro in 1 min</a>: Welcome to our deep dive into the Top 10 Deep Learning Algorithms! In this video, we break down each algorithm with a concise 10-word explanation. Perfect fo...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1256362763849171057)** (9 messages🔥): 

- **Urgent Help Needed with GEC Prediction List**: A member expressed an urgent issue with their **GEC prediction list** being out of shape and requested immediate advice.
   - The member emphasized the urgency of the situation, seeking assistance from the community.
- **Top 10 Deep Learning Algorithms Video**: A user shared a [YouTube video](https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC) titled **"Top 10 Deep Learning Algorithms intro in 1 min"** which provides concise 10-word explanations for each algorithm.
   - The video aims to offer a quick overview of deep learning algorithms but received a warning for **cross-posting** across multiple channels.
- **Seeking Hands-on LLM Learning Methods**: A member inquired about effective ways to engage in **hands-on learning** for LLMs.
   - No direct advice or resources were provided in response to the query.
- **Incorporating RBF Layer in Transformer Model**: A user asked if anyone has tried replacing a layer inside a **transformer model** with an **RBF layer**.
   - The user did not receive any responses or further discussion on the topic.

**Link mentioned**: <a href="https://youtu.be/vHMEJwxs_dg?si=NHM0uee4Rys7jVZC">Top 10 Deep Learning Algorithms intro in 1 min</a>: Welcome to our deep dive into the Top 10 Deep Learning Algorithms! In this video, we break down each algorithm with a concise 10-word explanation. Perfect fo...

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1257186167829434408)** (10 messages🔥): 

- **How to Trigger LoRA with Diffusers**: A user asked how to trigger **LoRA** with diffusers, mentioning they have loaded the LoRA but it has no effect.
   - Another user responded that **diffusers** do not parse prompts and you need to manually load the **LoRA weights**.
- **Issue with LoRA Weights Not Working**: The user shared their code for loading LoRA weights with `text2imgPipe` and setting the adapter weights, but noted it works with A1111.
   - Another user pointed out the issue might be due to setting the weights to **0.0**. The original user clarified that one of the LoRA weights is actually set to 0.5.
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1256326241364217886)** (1031 messages🔥🔥🔥): 

- **Unsloth Setup on WSL2**: Users discussed setting up Unsloth on WSL2, including commands and configurations to maximize resource usage. Some troubleshooting steps and tutorials were shared for those encountering installation issues.
   - Various users shared their configurations and fixes, such as setting memory to 0 in `.wslconfig` to avoid limiting RAM and using updated installation commands. Discussions included running Unsloth on both Intel and AMD systems.
- **Multi-GPU Support and DRM in Unsloth**: Unsloth's new DRM system for multi-GPU support was discussed, including GPU ID pinning and the persistence of unique GPU IDs. The DRM system is in testing under strict NDA, showing early access and efforts to balance licensing and flexibility.
   - Users expressed interest in multi-GPU training with Unsloth, including configurations and potential limitations. Updates on progress and potential release timelines were shared, emphasizing the importance of stable DRM implementation.
- **Fine-Tuning and Dataset Preparation Challenges**: Questions were raised about fine-tuning different models, including Lexi and Gemma. Users shared methods for integrating system tokens and handling challenges like endless generation after fine-tuning.
   - Discussions included techniques like translating datasets to different languages and maintaining uncensored training data. Best practices for dataset curation and avoiding critical forgetting were advised.
- **Using AI Tools and Platforms Effectively**: Various AI tools and platforms like Runpod and Ollama were discussed, including their benefits and how to integrate them with existing workflows. Users noted the availability of credits for training and the practicality of renting compute resources.
   - There were discussions about automating pipelines for deploying models and translating datasets for better performance. Comparisons between local and API-based solutions were made to highlight efficiency and cost-effectiveness.
- **Persona-Driven Data Synthesis and Its Applications**: A novel methodology for persona-driven data synthesis was introduced, leveraging diverse perspectives within large language models to create synthetic data at scale. The Persona Hub was mentioned, containing 1 billion personas to facilitate this process.
   - Applications of this methodology include creating high-quality datasets for various scenarios like mathematical reasoning, user prompts, and game NPCs. The potential impact on LLM research and practical applications was emphasized.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://foleycrafter.github.io/">FoleyCrafter</a>: no description found</li><li><a href="https://www.scmp.com/lifestyle/gadgets/article/3268063/china-made-play-dreams-headset-cheaper-higher-resolution-apple-vision-pro-clone">Play For Dream mixed reality headset: an improved Apple Vision Pro clone</a>: It may look just like the Apple Vision Pro, but the headset from Shanghai-based Play For Dream puts out higher-resolution visuals, feels more comfortable, and can run Android apps.</li><li><a href="https://docs.continue.dev/walkthroughs/tab-autocomplete#:~:text=The%20models%20that%20we%20suggest%20for%20autocomplete%20are%20trained%20with%20a%20highly%20specific%20prompt%20format%2C%20which%20allows%20them%20to%20respond%20to%20requests%20for%20completing%20code">Tab Autocomplete (beta) | Continue</a>: Continue now provides support for tab autocomplete in VS Code and JetBrains IDEs. We will be greatly improving the experience over the next few releases, and it is always helpful to hear feedback. If ...</li><li><a href="https://ar5iv.labs.arxiv.org/html/2304.06035">Choose Your Weapon: Survival Strategies for Depressed AI Academics</a>: no description found</li><li><a href="https://huggingface.co/google/recurrentgemma-2b-it">google/recurrentgemma-2b-it · Hugging Face</a>: no description found</li><li><a href="https://www.numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction">NuExtract: A Foundation Model for Structured Extraction  - NuMind</a>: We introduce NuExtract, a lightweight text-to-JSON LLM. NuExtract allows to extract arbitrarily complex information from text and turns it into structured data. This model can be directly used in a ze...</li><li><a href="https://huggingface.co/CohereForAI/aya-101">CohereForAI/aya-101 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/sweaty-sweat-heat-hot-wipe-sweat-gif-17716050">Sweaty Heat GIF - Sweaty Sweat Heat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/fireworks-ai/llama-3-firefunction-v2">fireworks-ai/llama-3-firefunction-v2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Kearm/UnslothAIWorldFair">Kearm/UnslothAIWorldFair · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/failspy/Llama-3-8B-Instruct-MopeyMule">failspy/Llama-3-8B-Instruct-MopeyMule · Hugging Face</a>: no description found</li><li><a href="https://share.hsforms.com/1tvg18CtoSH6EYna-eQzpgAecykq">Form</a>: no description found</li><li><a href="https://github.com/MC-E/ReVideo">GitHub - MC-E/ReVideo</a>: Contribute to MC-E/ReVideo development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/NeuralNovel/Unsloth-DPO">NeuralNovel/Unsloth-DPO · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/SafeAILab/EAGLE">GitHub - SafeAILab/EAGLE: Official Implementation of EAGLE</a>: Official Implementation of EAGLE. Contribute to SafeAILab/EAGLE development by creating an account on GitHub.</li><li><a href="https://github.com/b4rtaz/distributed-llama">GitHub - b4rtaz/distributed-llama: Tensor parallelism is all you need. Run LLMs on weak devices or make powerful devices even more powerful by distributing the workload and dividing the RAM usage.</a>: Tensor parallelism is all you need. Run LLMs on weak devices or make powerful devices even more powerful by distributing the workload and dividing the RAM usage. - b4rtaz/distributed-llama</li><li><a href="https://github.com/camenduru/FoleyCrafter-jupyter">GitHub - camenduru/FoleyCrafter-jupyter</a>: Contribute to camenduru/FoleyCrafter-jupyter development by creating an account on GitHub.</li><li><a href="https://rocm.docs.amd.com/en/latest/">AMD ROCm™ documentation — ROCm Documentation</a>: no description found</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/microsoft/WSL2-Linux-Kernel">GitHub - microsoft/WSL2-Linux-Kernel: The source for the Linux kernel used in Windows Subsystem for Linux 2 (WSL2)</a>: The source for the Linux kernel used in Windows Subsystem for Linux 2 (WSL2) - microsoft/WSL2-Linux-Kernel</li><li><a href="https://github.com/unslothai/unsloth/pull/708">yaml and cli by darkacorn · Pull Request #708 · unslothai/unsloth</a>: should work out of the box - seb please rev/ daniel and mike please comments on the idea as for why its pretty straight forward - central config vs over chaneing files - and more maintainable via c...</li><li><a href="https://ai.meta.com/research/cicero/diplomacy/">no title found</a>: no description found</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Qwen2-1.5b">Replete-AI/Replete-Coder-Qwen2-1.5b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/proj-persona/PersonaHub">proj-persona/PersonaHub · Datasets at Hugging Face</a>: no description found</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1drk3kc/gemma_2_betrayed_us/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1WHlelph_UAPksquDgsZ2Q-H6VUrAjS7H?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://x.com/arankomatsuzaki/status/1807593343007818065?s=46">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Scaling Synthetic Data Creation with 1,000,000,000 Personas  - Presents a collection of 1B diverse personas automatically curated from web data - Massive gains on MATH: 49.6 -&gt;64.9  repo: https://g...</li><li><a href="https://arxiv.org/abs/2406.20094">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>: We propose a novel persona-driven data synthesis methodology that leverages various perspectives within a large language model (LLM) to create diverse synthetic data. To fully exploit this methodology...</li><li><a href="https://www.bigscreenvr.com/">no title found</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues?q=wsl">Issues · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - Issues · unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: I got unsloth running in native windows, (no wsl). You need visual studio 2022 c++ compiler, triton, and deepspeed. I have a full tutorial on installing it, I would write it all here but I’m on mob...</li><li><a href="https://github.com/unslothai/unsloth/blob/933d9fe2cb2459f949ee2250e90a5b610d277eab/unsloth/tokenizer_utils.py#L962">unsloth/unsloth/tokenizer_utils.py at 933d9fe2cb2459f949ee2250e90a5b610d277eab · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/933d9fe2cb2459f949ee2250e90a5b610d277eab/unsloth/models/llama.py#L1199">unsloth/unsloth/models/llama.py at 933d9fe2cb2459f949ee2250e90a5b610d277eab · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1256945644120178688)** (14 messages🔥): 

- **Meta's Internal Release of Llama 400**: A [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1ds2p09/llama_400_released_internally_at_meta_available) discusses the **internal release of Llama 400** at Meta and its availability on WhatsApp, accompanied by an alleged WhatsApp screenshot. The discussion focuses on whether this information suggests a top-performing model.
   - Members express curiosity and hope for high performance, with one member noting, *'if scores are high (top 3 at least), then I hope so.'* Another member aligns with this sentiment.
- **LMSYS Model Types Clarified**: Discussion reveals that **LMSYS** has two models available: 'base' and 'instruct'. **User 'edd0302'** and another member confirmed this information in their exchange.
   - There appears to be a comparison of performance expectations, with **user 'mahiatlinux'** concurring on the variety of LMSYS models and their potential high scores.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1ds2p09/llama_400_released_internally_at_meta_available/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ds2p09/llama_400_released_internally_at_meta_available">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1256350984800829491)** (170 messages🔥🔥): 

- **Differences between Pretraining and Finetuning Explained**: A user asked about differences between continued pretraining and finetuning, mentioning the target modules included more elements for pretraining. Another user suggested that the extra modules are needed if pretraining on another language, referencing a [YouTube video](https://youtu.be/GZqYr8_Q7DE) by TrellisResearch.
   - The video explains why training embedding and LM head are important when fine-tuning on a different language.
- **Inference Errors After Pretraining**: A user experienced an error during inferencing after pretraining, suspecting it might be due to GPU memory issues with a 16GB T4. Another user advised checking a [GitHub issue](https://github.com/unslothai/unsloth/issues/702) for similar problems.
   - It was suggested that the issue might be due to the new PyTorch version causing problems.
- **Finetuning Llama3 Models**: A user inquired about finetuning an instruct version of Llama3 and expressed concerns about the chat template and mapping. It was suggested to use the base models for finetuning instead of instruct models.
   - Additional advice included trying out different tokenizers and formatting functions to ensure compatibility with the training data.
- **Handling Errors with Dataset Formats**: A user faced issues uploading a ShareGPT format dataset to Hugging Face and received an error when the dataset was loaded. Another user suggested ensuring the format is in `jsonl` and provided [example formats](https://huggingface.co/docs/datasets/loading_datasets.html).
   - It's recommended to read datasets using the Hugging Face library and then push them to the hub to avoid errors.
- **Running Unsloth on AMD and Windows**: A user asked about running Unsloth on AMD GPUs and Windows, encountering errors related to nvidia-smi and AMD drivers. Solutions included using different installation methods and ensuring proper driver initialization.
   - Further discussions involved dealing with limitations and compatibility issues when pushing large models to Hugging Face.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/tomasonjo/text2cypher-qwen2-72b-4bit-gguf/tree/main">tomasonjo/text2cypher-qwen2-72b-4bit-gguf at main</a>: no description found</li><li><a href="https://lightning.ai/lightning-ai/studios/train-a-gpt-classifier-from-scratch?section=tutorials">Train a GPT Classifier from Scratch - a Lightning Studio by sebastian</a>: This Studio provides a Jupyter notebook that explains how to finetune a GPT model from scratch to classify spam text messages with 96% accuracy</li><li><a href="https://github.com/unslothai/unsloth/issues/711">ROCm + WSL2 incompatibility · Issue #711 · unslothai/unsloth</a>: AMD&#39;s HIP/ROCm WSL2 driver does not utilize a kernel module, currently. /home/musclez/ComfyUI/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:691: UserWarning: Can&#39;t initialize amdsm...</li><li><a href="https://github.com/unslothai/unsloth/issues/702#issuecomment-2197477362">Cache only has 0 layers, attempted to access layer with index 0 · Issue #702 · unslothai/unsloth</a>: I&#39;m encountering a KeyError when trying to train Phi-3 using the unsloth library. The error occurs during the generation step with model.generate. Below are the details of the code and the error t...</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#fin">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/xingyaoww/opendevin-code-act">xingyaoww/opendevin-code-act · Datasets at Hugging Face</a>: no description found</li><li><a href="https://docs.vllm.ai/en/latest/models/lora.html">Using LoRA adapters &#8212; vLLM</a>: no description found</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1bd_VkH4aszEVvzuRXNtFNU70yWf38UJE?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1257009330092507207)** (2 messages): 

- **Fix OSS model bugs efficiently using GGUF**: Common bugs in OSS models like **Llama-3** include *Double BOS tokens issue* and *Untrained tokens causing NaNs*. Fixes involve using **GGUF's CPU conversion** instead of GPU to handle BOS tokens correctly.
   - These computational errors are mitigated by appropriately handling specific tokens, ensuring smoother model performance.
- **Install Unsloth effortlessly for LLMs fine-tuning**: To install *Unsloth*, create a Conda environment with specific packages and then activate the environment. The installation command is: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`.
   - Following these steps ensures a ready environment for fine-tuning LLMs efficiently.
- **Unsloth accelerates model training**: [UnslothAIWorldFair](https://huggingface.co/datasets/Kearm/UnslothAIWorldFair) model trained **2x faster** using *Unsloth* with Huggingface’s TRL library.
   - This model, under the **Llama 3 Community License**, has a size of 8.03B parameters and utilizes **BF16 tensor type**.
- **Qwen2-Wukong-0.5B introduces chat finetune**: [Qwen2-Wukong-0.5B](https://huggingface.co/RESMPDEV/Qwen2-Wukong-0.5B) is a **dealigned chat finetune** of the original Qwen2-0.5B model.
   - Trained on **teknium OpenHeremes-2.5 dataset** for three epochs, it outperforms in data classification against the 1.1B TinyLlama tune.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/RESMPDEV/Qwen2-Wukong-0.5B">RESMPDEV/Qwen2-Wukong-0.5B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Kearm/UnslothAIWorldFair">Kearm/UnslothAIWorldFair · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Kearm/UnslothAIWorldFair">Kearm/UnslothAIWorldFair · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1257120870552571965)** (10 messages🔥): 

- **1B Personas Drive Data Synthesis**: The [Persona Hub](https://github.com/tencent-ailab/persona-hub) introduces a novel persona-driven data synthesis methodology, leveraging diverse personas to create synthetic data from 1 billion personas curated from web data. This approach facilitated a significant boost in MATH scores, from **49.6** to **64.9**.
   - The community discussed the implications of this synthetic data creation, emphasizing the versatility and ease of use presented in the [abstract](https://arxiv.org/abs/2406.20094). One user noted, *this seems very important, but there’s no code*, another suggested *replicating this isn't hard, the data is way more important than the code*.
- **Data Over Code Debate**: Members debated the importance of data versus code in the context of the Persona Hub. While one user expressed disappointment over the absence of code, others argued that the data was the crucial element.
   - The discussion highlighted differing perspectives on the necessity of code, with remarks like *the data is way more important than the code*. This underscores the community's varied viewpoints on what components are essential for replicating and leveraging such large-scale projects.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/arankomatsuzaki/status/1807593343007818065?s=46">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Scaling Synthetic Data Creation with 1,000,000,000 Personas  - Presents a collection of 1B diverse personas automatically curated from web data - Massive gains on MATH: 49.6 -&gt;64.9  repo: https://g...</li><li><a href="https://arxiv.org/abs/2406.20094">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>: We propose a novel persona-driven data synthesis methodology that leverages various perspectives within a large language model (LLM) to create diverse synthetic data. To fully exploit this methodology...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1256323607676719146)** (332 messages🔥🔥): 

- **Gemma 2 GPU Offloads - Currently Limited**: A member inquired about Gemma 2 GPU offload support, but it currently only supports **Cuda** and **Metal**. It was also highlighted that updates are necessary for versions beyond 0.2.23 to fix issues.
   - Loading the Gemma 2 models is taking excessively long on certain configurations, and updates have been requested and noted as ongoing fixes in the LM Studio community.
- **AutoUpdater Revamped in LM Studio**: An **AutoUpdater** for LM Studio is now working, ensuring users can easily update to version **0.2.26**. The community has expressed anticipation for the upcoming version 0.3.
   - Discussion highlights that recent iterations should resolve issues, especially concerning the **LLama 3 8B model** eliminating its persistent stop sequence ignoring problem.
- **Deepseek v2 and CPU Performance**: Deepseek v2 performance was explored, noting that despite its 200B+ parameters, only 21B are activated, implying manageable performance on powerful CPUs. Reports mention 3-4 tokens/sec on high-end Threadripper systems with extensive memory.
   - **Shared User Testing:** Users shared their hands-on performance metrics, including RAM usage, model loading, and generation speeds, shedding light on practical capabilities and limitations.
- **New Experimental Quants in LM Studio**: There was mixed feedback on experimental quantizations like **q5_k_l**, aiming to maintain output quality while reducing memory footprint. **GGUF quants** are particularly mentioned for their balance between performance and efficiency.
   - Community feedback and testing continue, aiming to refine these quantizations for broader adoption and consistent performance enhancements.
- **Sliding Window Attention Merged for Gemma 2**: The **Sliding Window Attention** feature for **Gemma 2** has been merged into the latest **llama.cpp**. This aims to improve model performance by effectively handling past tokens for enhanced context understanding.
   - Despite this update, some users have noted ongoing quality issues, and further investigations and updates are anticipated to address those concerns comprehensively.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6">no title found</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/windows/wsl/install">Install WSL</a>: Install Windows Subsystem for Linux with the command, wsl --install. Use a Bash terminal on your Windows machine run by your preferred Linux distribution - Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin,...</li><li><a href="https://www.youtube.com/watch?v=l8pRSuU81PU&">Let&#39;s reproduce GPT-2 (124M)</a>: We reproduce the GPT-2 (124M) from scratch. This video covers the whole process: First we build the GPT-2 network, then we optimize its training to be really...</li><li><a href="https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF">lmstudio-community/gemma-2-9b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=iOdFUJiB0Zc">Fine Tuning LLM Models – Generative AI Course</a>: Learn how to fine tuning LLM models. This course will teach you fine tuning using using QLORA and LORA, as well as Quantization using LLama2, Gradient and th...</li><li><a href="https://huggingface.co/thethinkmachine">thethinkmachine (Shreyan Chaubey)</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs</li><li><a href="https://unsloth.ai">Unsloth AI | Finetune Llama 3 &amp; Mistral LLMs</a>: Easy finetuning for AI and LLMs. Open-source and for beginners. Get faster with Unsloth. </li><li><a href="https://huggingface.co/Joseph717171">Joseph717171 (Joseph)</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio CLI. Written in TypeScript/Node</a>: LM Studio CLI. Written in TypeScript/Node. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://huggingface.co/bartowski/Samantha-Qwen-2-7B-GGUF/discussions/2">bartowski/Samantha-Qwen-2-7B-GGUF · Testing experimental quants</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8183">Bug: quantized gemma 27b output still wrong after tokenizer fix and soft capping · Issue #8183 · ggerganov/llama.cpp</a>: What happened? The quantized version of gemma 27b (Q8_0) still gets the answer wrong to even simple problems. The version of gemma on ai studio answers correctly all my questions. Example problem t...</li><li><a href="https://huggingface.co/bartowski/Samantha-Qwen-2-7B-GGUF/discussions/2#6673e67852c02322ba7ee01c">bartowski/Samantha-Qwen-2-7B-GGUF · Testing experimental quants</a>: no description found</li><li><a href="https://huggingface.co/bartowski">bartowski (Bartowski)</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8197">Add attention and final logit soft-capping, update scaling factor to Gemma2 by abetlen · Pull Request #8197 · ggerganov/llama.cpp</a>: This PR adds the missing attention layer and final logit soft-capping. Implementation referenced from huggingface transformers. Additionally Gemma2 applies a pre-attention scaling of hidden_size / ...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8227">gemma2: add sliding window mask by ngxson · Pull Request #8227 · ggerganov/llama.cpp</a>: This is a hack to support sliding window attention for gemma 2 by masking past tokens. The goal is to make it works. While the ideal solution is to have per-layer KV cache management (with differen...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8240)),">Issues · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://ubuntu.com/desktop/wsl">Windows Subsystem for Linux (WSL) | Ubuntu</a>:    Access the Ubuntu terminal on Windows with WSL. Develop cross-platform applications and manage IT infrastructure without leaving Windows. 
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1256328974762508309)** (221 messages🔥🔥): 

- **Gemma 2's performance variability**: Gemma 2's performance was discussed extensively with mixed reviews ranging from it being as good as **Mistral** to having significant issues like **constant repetition**. Users explored several technical adjustments and insights were shared about **pending updates** for fixing these issues.
   - There were concerns about compatibility particularly with **AMD ROCm** and the efficiency of different quantizations like **Q4KM and IQ3_M**. Discussions revealed mixed opinions on how effective these fixes would be, comparing personal trials and experiences.
- **Model preferences and comparisons**: Members frequently compared various models like **ChatGPT, Claude 3.5, Deepseek** and others, sharing performance benchmarks and individual experiences. **Deepseek Coder** was particularly highlighted as the best local code model among the 7b range.
   - Switching from **ChatGPT** due to dissatisfaction was a common sentiment, with some users preferring alternatives like **Claude 3.5 from Anthropic**. The newer models, like **LLM Studio**, were often praised for their efficiency despite users highlighting areas for improvement such as the UI.
- **Quantization and its hurdles**: Quantization discussions often revolved around balancing model size with performance, with specific focus on varieties like **Q4KM, IQ4_XS, and IQ3_M**. Tested quantized models showed **5KM** being better but **IQ3-M** being workable if hardware constraints exist.
   - Reports highlighted issues with certain quantized models, pointing out mismatches and corrupted quant files. Users shared solutions and alternative quant models that worked without significant problems, like **bartowski's quantizations**.
- **Vision models for image and text tasks**: Users discussed capabilities of vision models for handling tasks like **image captioning** and sought recommendations for specific needs like comics analysis. Models like **Florence-2 and LLaVa** were suggested as capable options.
   - Discussions expanded on specific use cases for these models and their integration within existing workflows. **Meta Chameleon** was mentioned as another potential model for vision tasks though it wasn't widely tested by the group.
- **Local model experiments and settings**: Users often shared their settings and local model configurations, highlighting the importance of **embedding options and GPU settings**. Many preferred smaller, more efficient models due to hardware limitations.
   - Practical advice for optimizing local LLM performance included tips on configuring **GPU layers or fine-tuning**. Conversations also elaborated on embedding techniques for specific task optimizations using models like **Nomic-Embed-Text-v1.5**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/cartoons-tom-and-jerry-ok-mouse-ok-i-got-it-gif-17005831">Cartoons Tom And Jerry GIF - Cartoons Tom And Jerry Ok - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9">GGUF quantizations overview</a>: GGUF quantizations overview. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://huggingface.co/bartowski/Yi-1.5-34B-Chat-GGUF/blob/main/Yi-1.5-34B-Chat-Q6_K.gguf">Yi-1.5-34B-Chat-Q6_K.gguf · bartowski/Yi-1.5-34B-Chat-GGUF at main</a>: no description found</li><li><a href="https://rentry.org/LMSTudioFAQ">The unofficial LMStudio FAQ!</a>: Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...</li><li><a href="https://huggingface.co/mxs980/gte-Qwen2-1.5B-instruct-Q8_0-GGUF">mxs980/gte-Qwen2-1.5B-instruct-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://link.springer.com/chapter/10.1007/978-3-540-30493-7_19">Design of a Hybrid CDN</a>: Peer-to-peer (P2P) based networks have several desirable features for content distribution, such as low costs, scalability, and fault tolerance. However, they usually fail to provide guarantees for co...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1256323247704641609)** (1 messages): 

- **LM Studio 0.2.26 Release**: LM Studio 0.2.26 is now available for Mac (M1/M2/M3), Windows (x86/ARM64), and Linux (x86). You can download it from [lmstudio.ai](https://lmstudio.ai).
   - This release includes support for Google's Gemma 2 models and is built with `llama.cpp` at commit `97877eb10bd8e7f8023420b5b5300bcbdadd62dc`.
- **Support for Google Gemma 2 Models**: The latest version adds support for **Google's Gemma 2** models, specifically **9B** and **27B**. You can [download them here](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) for 9B and [here](https://huggingface.co/lmstudio-community/gemma-2-27b-it-GGUF) for 27B.
   - This functionality was enabled by contributions from the community to `llama.cpp`.
- **LM Studio for Windows on ARM**: **LM Studio 0.2.26** is now available for Windows on **ARM64** (Snapdragon X Elite PCs). This was achieved through collaboration with Qualcomm, discussed further [on LinkedIn](https://www.linkedin.com/posts/qualcomm_ai-snapdragon-snapdragonxseries-activity-7212153031222513666-7s0N?utm_source=share&utm_medium=member_desktop).
   - The ARM64 version can be downloaded from [lmstudio.ai/snapdragon](https://lmstudio.ai/snapdragon).
- **Upcoming LM Studio 0.3.0 Beta Testing**: A **huge update** to LM Studio is almost ready, and the team is seeking beta testers. Interested users can [sign up here](https://forms.gle/K7pTWgTJsdHBmUaWA).
   - Emails with new beta versions will be sent to participants as they become available.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://lmstudio.ai/snapdragon">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://forms.gle/K7pTWgTJsdHBmUaWA">LM Studio 0.3.0 - Private Beta Sign Up</a>: Thanks for your interest in helping out test our upcoming release.   LM Studio 0.3.0 is gem-packed with new features and we&#39;d love your help to shake out the bugs before sending it out to the worl...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1257011322353225908)** (13 messages🔥): 

- **Performance Issues in V0.2.26**: A user reported that V0.2.26 feels slower than V0.2.24, with **5b models** performing as fast as **8b models** in the previous version.
   - Another member suggested the slowdown might be due to **old chats** and reaching **context limits**, recommending a **more powerful GPU** as a solution.
- **No Browser Spell-Check in Fields**: A member noted the lack of browser spell-check functionality in various fields, even though misspelled words are underlined.
   - Another member confirmed that while **spell-check** has been long requested, it is not currently available.
- **Keyboard Navigation in Delete Chat Modal**: A user provided feedback that the "Delete Chat" modal should have a button focused for keyboard navigation.
   - They mentioned that while **Escape/Enter** keys can control the modal, initial tab-selection requires a mouse click.
- **Guidelines for Creating New Language Channels**: A user inquired about creating a Norwegian language channel, as per the guidelines suggesting to request new languages.
   - Members confirmed that users should be able to **create threads themselves**, and the inquiry was just following **compliance rules**.
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1256364080504242318)** (149 messages🔥🔥): 

- **Mix smaller models with lower hardware needs**: A user suggested that combining 0.5b models could make them highly usable on 4GB computers, potentially bridging the hardware gap.
   - There were discussions about CPU and GPU speed differences, with Intel and IPEX's llama.cpp fork being highlighted for better GPU performance.
- **Server Power Use: It's Electrifying!**: Users shared detailed stats on power usage for various LLM models and setups, including a server with dual Xeon E5 CPUs and Tesla P40 GPUs drawing up to 858 watts.
   - Another user compared their own server setup, noting Nvidia-smi reported figures around 70W & 180W, sparking a discussion on efficient monitoring and hardware optimization.
- **4090 vs Dual 4080s: GPU Showdown**: A user queried the community about the performance trade-offs between a single 4090 and dual 4080 GPUs, receiving advice that a single 4090 often performs better due to model speed suffering when split over two cards.
   - Further discussions suggested that for large models like 70b, a single 24GB GPU is preferable to dual 16GB setups, reinforcing the single 4090 recommendation.
- **Trading Setup: Mac vs PC**: A user considered the benefits of trading software on a Mac Studio compared to a PC setup, citing that they could run their Python/C# trading bots anywhere, but the interface is Windows-based.
   - Another suggested the million-dollar idea of developing trading software for Mac to make the switch more feasible, highlighting hardware flexibility as crucial for AI workloads.
- **ASICs: The Future of AI**: Pooxid introduced the Sohu ASIC specialized for transformers, claiming it can run LLaMA 70B with over 500,000 tokens per second—much faster and cheaper than NVIDIA's GPUs.
   - The discussion noted that while ASICs like Sohu can't run traditional AI models, they could significantly speed up transformer-based tasks, sparking interest and cautious optimism.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.etched.com/announcing-etched">Etched is Making the Biggest Bet in AI</a>: no description found</li><li><a href="https://tenor.com/view/doja-cat-star-wars-gif-25078126">Doja Cat GIF - Doja Cat Star - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/bartowski/Smaug-Llama-3-70B-Instruct-32K-GGUF">bartowski/Smaug-Llama-3-70B-Instruct-32K-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1256323614081421395)** (14 messages🔥): 

- **ROCm 0.2.26 Release Hits the Servers**: Version **0.2.26 ROCm** was announced to be available [today](https://discord.com/channels/1110598183144399058/1195858490338594866/1256374959651946616). The exact release timing wasn't clear, leading to some fun banter about timezone differences.
   - An amusing exchange included a **Futurama Angry GIF** ([link](https://media1.tenor.com/m/V45VFWvSVyUAAAAC/futurama-angry.gif)). Users seemed eager but patient for the deployment.
- **Troubleshooting Empty Responses in New Release**: One user experienced **empty responses** after trying the newest release with updated ROCm and Gemma. The verbose logs showed the process starting but not completing with any output.
   - After some back and forth, it was revealed that using **SillyTavern** as the front-end was causing the issue. Testing directly in the UI worked fine, indicating potential bugs in the third-party integration.
- **Debugging Issues with LM Studio CLI**: To help debug the issues, a member suggested installing the **LM Studio CLI** tool (`lms`). A link to the [installation guide](https://lmstudio.ai/blog/lms) was provided.
   - Using the command `lms log stream` can help identify what prompts are being sent to the model, aiding in the troubleshooting process. This was recommended for further debugging.

**Link mentioned**: <a href="https://tenor.com/view/futurama-angry-gif-13063135">Futurama Angry GIF - Futurama Angry - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1256340898820653086)** (29 messages🔥): 

- **ROCm Extension Pack for Windows 0.2.26 Now Available**: A member announced the release of the **0.2.26 ROCm "extension pack"** for **Windows**, providing a download link and instructions for installation. Users can obtain it from [LMStudio.ai](https://lmstudio.ai) and need to run a command in Powershell as part of the installation.
   - Following the installation steps correctly, several users confirmed successful updates and improved performance, particularly for the Gemma 2 model, which now works seamlessly. One user highlighted that the **RX 7900 XTX** is drawing 350W at full usage with these updates.
- **Significant Uplift in Performance with ROCm 0.2.26**: Members reported noticeable performance enhancements after upgrading to **ROCm 0.2.26** and using the latest **24.6.1** AMD drivers. For instance, running **Codestral 22B Q4_K_M** on a **6900 XT** yielded performance jumps from **9.29 tok/s** on the previous version to **26.75 tok/s**.
   - A user noted that despite purchasing an NVIDIA GPU for better AI performance, the recent improvements might render it unnecessary. Another user emphasized the importance of comparing performance differences across hardware.
- **Qwen2 Model Issues and Fixes**: Qwen2 was reported to be buggy by multiple users, but a member pointed out that enabling **flash attention** can fix the issues. Specific configurations and setup instructions were discussed to ensure the model operates correctly.
   - *heyitsyorkie* clarified that Qwen2 needs **flash attention** to function properly, confirming the solution to the arising bugs. Users provided feedback on what configurations worked or didn’t work for them.
- **ROCm Support and Compatibility Discussion**: Technical discussions ensued about the lack of **ROCm support** on certain GPUs and the availability of specific builds. **Blue.balls** mentioned issues with ROCm not appearing as an option, which was clarified to be due to the user’s GPU lacking ROCm support.
   - Members discussed the necessity of having the correct extension packs and compatible hardware. *heyitsyorkie* confirmed that the **0.2.26 Linux ROCm** support is still being developed and suggested waiting for official updates.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai,">no title found</a>: no description found</li><li><a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#installation-on-windows-0226-">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1256339154602688622)** (11 messages🔥): 

- **Installing LM Studio on Unraid/Dockers is challenging**: A member asked if there is a way to install **LM Studio** on **unraid** or **dockers**, but received feedback stating that it's currently not possible.
   - The discussion didn't yield any solution, highlighting the current limitation in deploying LM Studio in such environments.
- **LM Studio SDK version update causes loadModel error**: After updating LM Studio from **0.2.25** to **0.2.26** and **lmstudio.js** from **0.0.3** to **0.0.12**, a user experienced an error when executing the `await client.llm.load(modelPath)` command.
   - The error *'Received invalid creationParameter for channel'* was reported, and it was suggested to raise an issue on [GitHub](https://github.com/lmstudio-ai/lmstudio.js/issues).
- **Community support for SDK issues**: Members, including **@heyitsyorkie** and **yagilb**, responded to a user's issue with SDK commands and offered assistance.
   - They suggested opening an issue on GitHub for further investigation and troubleshooting, demonstrating **active community support**.

**Link mentioned**: <a href="https://github.com/lmstudio-ai/lmstudio.js/issues">Issues · lmstudio-ai/lmstudio.js</a>: LM Studio TypeScript SDK (pre-release public alpha) - Issues · lmstudio-ai/lmstudio.js

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1256323904729907272)** (716 messages🔥🔥🔥): 

- **Lora Training on SD3 Difficulties**: Users discussed the challenges of creating and using Loras with Stable Diffusion 3 models, mentioning the complexity and current lack of support for training them effectively. There is anticipation for SD 8b, which might not need as much fine-tuning.
   - Some users expressed the importance of waiting for proper fine-tuning data and training tools for SD3 rather than rushing into creating subpar Loras or checkpoints.
- **Choosing the Right Hardware for SD**: New users interested in running Stable Diffusion discussed hardware requirements, such as GPU memory and processing power. Recommendations leaned towards Nvidia GPUs with at least 12GB VRAM or more for efficient operation.
   - Users mentioned that while older GPUs like the RTX 3090 with 24GB VRAM are valuable, the latest cards (RTX 4080, 4090) are more future-proof despite their high costs.
- **Installation and Setup Issues**: Issues around installing and setting up Stable Diffusion with various interfaces (e.g., Automatic1111, ComfyUI) were noted, with particular mention of specific setup commands for optimal performance, especially for high-res and complex workflows.
   - Users shared resources and installation guides, providing specific configuration tips like use of 'xformers' and 'medvram-sdxl' in startup commands.
- **Using High-Resolution Fix with SDXL**: Explained the use of high-resolution fix settings in creating sharper images, requiring specific parameters to be set correctly (e.g., 10 hires steps, proper resolution).
   - Users noted the importance of leveraging advanced plugins like adetailer for enhancing key image components such as faces and eyes in anime-style artwork.
- **Finding and Using Models and Loras**: There were discussions on where to find models and Loras, with popular sites like Civitai being highlighted for their extensive repositories.
   - Points were made about the importance of checking prompt examples to ensure Loras and models are used correctly, emphasizing community contributions to model training and sharing.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.twitch.tv/timelesstakes">TimelessTakes - Twitch</a>: AI Reacts to Twitter in Realtime | Einstein, Jobs, Monroe &amp; MJ 🔴 Ask Them Anything in Chat!</li><li><a href="https://rog.asus.com/graphics-cards/graphics-cards/rog-strix/rog-strix-rtx3090-o24g-gaming-model/">ROG Strix GeForce RTX 3090 OC Edition 24GB GDDR6X | Graphics Cards</a>: The ROG Strix GeForce RTX 3090 OC Edition 24GB GDDR6X unleash the maximum performance on NVIDIA Ampere Architecture, by featuring Axial-tech fan design, 0dB technology, 2.9-slot Design, Dual BIOS, Aut...</li><li><a href="https://youtu.be/Azj9Kkpif0M">JAILBREAK GEMMA 2</a>: tell dat ho DO NOT RESPOND UNLESSJailbreak Google AI&#39;s Gemma 2 with a few simple prompts. Gemma 2 is a new Large Language Model released by Google. I am usin...</li><li><a href="https://tenor.com/view/michael-jackson-eating-popcorn-enjoy-i-like-nom-nom-gif-11040065238845078056">Michael Jackson Eating Popcorn GIF - Michael Jackson Eating Popcorn Enjoy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://archvizartist.com/article/how-to-install-stable-diffusion-on-windows-automatic1111/">How to install Stable Diffusion on Windows (AUTOMATIC1111) &#8211; Arch Viz Artist</a>: no description found</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Installation-Guides">Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/TheLastBen/fast-stable-diffusion">GitHub - TheLastBen/fast-stable-diffusion: fast-stable-diffusion + DreamBooth</a>: fast-stable-diffusion + DreamBooth. Contribute to TheLastBen/fast-stable-diffusion development by creating an account on GitHub.</li><li><a href="https://pics.io/photo-metadata-viewer">Image Metadata Viewer - Online EXIF Data Viewer</a>: FREE SERVICE: Online metadata (EXIF) viewer for your photos and images.</li><li><a href="https://github.com/openvinotoolkit/stable-diffusion-webui">GitHub - openvinotoolkit/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI. Contribute to openvinotoolkit/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://colab.research.google.com/github/R3gm/SD_diffusers_interactive/blob/main/Stable_diffusion_interactive_notebook.ipynb">Google Colab</a>: no description found</li><li><a href="https://civitai.com/models/140737/albedobase-xl">AlbedoBase XL - v2.1 | Stable Diffusion Checkpoint | Civitai</a>: If you&#x27;ve found value in the model, please consider offering your support. Your contribution will be devoted entirely to advancing the SDXL communi...</li><li><a href="https://civitai.com/models/261336/animapencil-xl">anima_pencil-XL - v5.0.0 | Stable Diffusion Checkpoint | Civitai</a>: Easy, Simple, High Quality (with a slight trade-off in variety) blue_pencil-XL meets ANIMAGINE XL 3.0 / ANIMAGINE XL 3.1 License : Fair AI Public L...</li><li><a href="https://www.newegg.com/abs-aqa14700kf4060ti16g-stratos-aqua/p/N82E16883360436">ABS Aquilon Aqua Gaming PC - Windows 11 Home - Intel Core i7 14th Gen 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI-Powered Performance - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI16G - Newegg.com</a>: Buy ABS Aquilon Aqua Gaming PC - Windows 11 Home - Intel Core i7 14th Gen 14700KF - GeForce RTX 4060 Ti 16GB - DLSS 3 - AI-Powered Performance - 32GB DDR5 6000MHz - 1TB M.2 NVMe SSD - AQA14700KF4060TI...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/13a9avh/quick_question_does_putting_break_in_a_prompt/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1257353627534622871)** (1 messages): 

- **Essential Tools for Meeting Coordination**: Two tools were shared for coordinating meeting times: **[Discord Timestamps](https://r.3v.fi/discord-timestamps/)**, which allows messages to display times in the viewer's timezone, and **[When2meet](https://www.when2meet.com/)**, which helps collect a group's availability to find the best meeting time.
   - These tools aim to simplify scheduling by handling timezone conversions and availability polling, making it easier to find mutually convenient meeting times for all participants.
- **Simplifying Group Scheduling**: **Discord Timestamps** lets you send messages that display times specific to the viewer's timezone using a format like `<t:1717964400:f>`.
   - **When2meet** streamlines the process of collecting a group's availability to determine the optimal meeting time.
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1256686956440715354)** (4 messages): 

- **Log Exp Utility Debate**: **chr0nomaton** mentioned the utility of log exp functions when working on reducing numerical stability issues in statistical modeling and machine learning. He shared a [blog post](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) that explains how **logarithmic scale** converts multiplication to addition for better numerical stability.
   - **chhillee** countered that the log scale is generally not needed, prompting further questions about its necessary applications. **doubleart** queried if there are specific scenarios where it becomes crucial.
- **Log Scale Stabilizes Small Floating Points**: **chr0nomaton** highlighted that working on a **logarithmic scale** is particularly useful when handling small floating point numbers to avoid underflow. He emphasized that this technique makes dealing with log likelihoods and probabilities much more numerically stable.
   - **chhillee** responded that log scale utility is generally unnecessary, initiating further discussion. **doubleart** asked for clarity on specific situations where it is essential.

**Link mentioned**: <a href="https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/">The Log-Sum-Exp Trick</a>: no description found

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1256846056218562622)** (24 messages🔥): 

- **Appending metadata to torch.Tensor with torch.compile**: A member asked about appending metadata in the form of a tuple to `torch.Tensor` while maintaining support for `torch.compile` without encountering `UserDefinedVariable` errors. They explored options including using tensor subclasses with metadata from `torchvision` but faced issues with losing the metadata during tensor operations.
   - Suggestions included storing the metadata as a torch Tensor or using subclasses like those in `torchvision`; however, these did not solve the problem. The user shared difficulties in maintaining the metadata through various tensor manipulations and explicit tensor passing.
- **Forced use of compiled functions in torch.compile**: A user asked if there's a way to force `torch.compile` to use the compiled function for predefined input shapes and revert to eager mode for others to avoid recompilation. The context involved optimizing HuggingFace transformers' `.generate()` function where the prefill and decoding phases have different input shape requirements.
   - Suggestions included using `torch.compile(dynamic=True)`, marking parts of the code to run in eager mode using fine-grained APIs, and setting recompile limits. The user confirmed they are using a custom wrapper already and expressed concern about maintenance across new releases.
- **HuggingFace trainer save_model timeout issue**: A member faced a timeout issue when using `HuggingFace` trainer's `save_model` over multiple machines, though intermediate checkpoints save successfully. They identified that calling `state_dict = trainer.accelerator.get_state_dict(trainer.model)` was causing the hang leading to socket timeouts.
   - The setup works fine on a single machine, pointing to issues specific to multi-machine configurations. The error logs indicated problems with `torch.distributed.elastic` related to socket timeouts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html">TorchDynamo APIs for fine-grained tracing &mdash; PyTorch 2.3 documentation</a>: no description found</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py">hqq/hqq/utils/generation_hf.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch/vision/blob/main/torchvision/tv_tensors/_bounding_boxes.py">vision/torchvision/tv_tensors/_bounding_boxes.py at main · pytorch/vision</a>: Datasets, Transforms and Models specific to Computer Vision - pytorch/vision</li><li><a href="https://docs.google.com/presentation/d/1piuv9nBzyoqdH49D1SoE5OZUPSMpOOFqfSKOhr-ab2c/edit#slide=id.p1">BackToPython PTC 2022 Poster</a>: 1 Back to Python: Extending PyTorch without touching C++ Alban Desmaison Meta Abstract Extending PyTorch from Python directly In the past year, the PyTorch team has been working on improving the exten...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1256713771242754099)** (3 messages): 

- **AI Engineer World’s Fair 2024 Showcase**: A member recommended the [YouTube video](https://www.youtube.com/watch?v=JVSKlEmUr0k) titled **"AI Engineer World’s Fair 2024 – GPUs & Inference Track"** hosted on the channel **AI Engineer World**. They emphasized that *this whole track* is worth skimming through.
   - *Dylan's talk* within the same track received positive feedback, underscoring its importance in the community.
- **Stephen Jones' Talks Always Impress**: A member vouched for all talks by **Stephen Jones**, describing them as consistently valuable and insightful. They noted, *"literally every talk from Stephen Jones is worth watching"*.
   - Another comment echoed this sentiment, adding that Stephen Jones' talks are highly recommended by the community.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=JVSKlEmUr0k">AI Engineer World’s Fair 2024 — GPUs &amp; Inference Track</a>: https://twitter.com/aidotengineer

  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1256639961550291060)** (2 messages): 

- **Book Has Blank Pages**: A user reported that their book has **blank pages**. This indicates a potential printing error.
   - It's unclear whether the issue is isolated or prevalent across multiple copies.
- **PMPP Edition Differences**: A user asked about the differences between the **third and fourth editions** of PMPP. They are concerned if they need to purchase the new edition.
   - No further details or replies have been provided yet about the specific changes or necessity to upgrade.
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1256361464252465265)** (42 messages🔥): 

- **Kernel launch bottleneck resolved by FP16**: **Kernel launches** were causing performance issues when using bfloat in the quant API, but switching to **FP16** reduced it to a single launch. Detailed profiling revealed the multiple kernel launches.
   - Despite the optimization, further improvements are discussed, such as special unpack+dequant+matmul kernels for **large tensors**. The community suggests integrating **gemv work** for future advancements.
- **Bitpacking optimisation struggles with large tensors**: Efforts to **make bitpack functions faster** yielded success for smaller tensors but not for tensors 2048 or larger, where performance was worse than **FP16**.
   - Future **research** may involve developing special kernels to unlock significant speedups, acknowledging current limitations with PyTorch's ability to fuse operations efficiently.
- **Flash attention discussion for torchao contributions**: Adding **variants of flash attention** in Triton was suggested as a good first issue for new contributors to torchao, especially around relative positional encodings from tools like **sam-fast**.
   - **Marksaroufim** expressed reservations as the focus is on a dtype and layout library, but acknowledged it could be beneficial for **architecture optimization** if the scope doesn’t creep too much.
- **FlexAttention API PR unveiled**: **Drisspg** and collaborators have been working on a new **FlexAttention API** for PyTorch, aimed at extending the functionality of existing attention modules.
   - The initial **pull request** is shared with the community ([FlexAttention API PR](https://github.com/pytorch/pytorch/issues/121845)), highlighting the upcoming broad announcement upon public API release.
- **Discussion on good first issues for CUDA contributors**: There was a discussion about identifying a **good first issue** for new contributors, with potential tasks involving CUDA kernel development for torchao.
   - Concerns were raised about the complexity of writing kernels in raw CUDA. More approachable tasks and offering high-level walks-through are suggested to help onboard new contributors.

**Link mentioned**: <a href="https://github.com/pytorch/pytorch/issues/121845">FlexAttention API by drisspg · Pull Request #121845 · pytorch/pytorch</a>: Summary This PR adds a new higher-order_op: templated_attention.  This op is designed to extend the functionality of torch.nn.fucntional.scaled_dot_product_attention.  PyTorch has efficient pre-wri...

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1257054602830811199)** (2 messages): 

- **Next-door 10x Software Engineer Video Shared**: *Iron_bound* shared a [YouTube video](https://www.youtube.com/watch?v=kKAue9DiHc0) titled "**Next-door 10x Software Engineer** [FULL]".
   - *as_ai* commented on the video, stating *"I love this guy's videos. They're so funny"*.
- **Appreciation for Funny Videos**: *As_ai* expressed their appreciation for the humor in the shared video.
   - *As_ai* mentioned, *"I love this guy's videos. They're so funny"*, clearly enjoying the content.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=kKAue9DiHc0">*Next-door 10x Software Engineer* [FULL]</a>: no description found

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1256323417297125456)** (473 messages🔥🔥🔥): 

- **FineWeb vs. FineWeb-EDU Stability**: There have been training instabilities with FineWeb 100B samples, whereas FineWeb-EDU samples are more stable for the 1.5B model. This has been corroborated by multiple members and further analysis is ongoing.
   - The team is looking into the potential issues within FineWeb samples and plans to switch to FineWeb-EDU, which has shown consistent stability in training runs.
- **Tracing Performance Issues**: A deep dive into tracing performance revealed potential areas for fine-tuning, especially with different batch sizes and recompute settings. Some setups showed minimal improvements, indicating possible diminishing returns on scaling.
   - There is an initiative to better understand the interplay between batch sizes, recompute settings, and overall training performance, with findings suggesting that more aggressive strategies may not yield proportional improvements.
- **Coordination and Debugging with MuP and FP8**: Efforts to integrate Maximum Update Parametrization (MuP) into the training workflow have shown potential for stabilizing training. Coordination checks and further stability tests are ongoing to fine-tune hyperparameters.
   - FP8 integration is being explored with an emphasis on minimizing performance overhead and ensuring architectural soundness. The team is also focused on cleaning the matmul operations for better performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/661">Fix periodic loss spikes by gordicaleksa · Pull Request #661 · karpathy/llm.c</a>: We got a similar bug introduced for similar reasons in our main code as before in PyTorch code: Due to the fact that zero grad happens after the backward step instead of immediately before we do ba...</li><li><a href="https://github.com/karpathy/llm.c/pull/653">Matmul refactor using only cuBLASLt + GELU Fusion by ademeure · Pull Request #653 · karpathy/llm.c</a>: In preparation for FP8, this replaces all cuBLAS calls by cuBLASLt which is now wrapped by a single matmul_cublaslt() function. It also adds support for GELU fusion which can be controlled on the c...</li><li><a href="https://huggingface.co/mlx-community">mlx-community (MLX Community)</a>: no description found</li><li><a href="https://github.com/LambdaLabsML/llm.c-1cc">GitHub - LambdaLabsML/llm.c-1cc</a>: Contribute to LambdaLabsML/llm.c-1cc development by creating an account on GitHub.</li><li><a href="https://github.com/clu0/unet.cu">GitHub - clu0/unet.cu: UNet diffusion model in pure CUDA</a>: UNet diffusion model in pure CUDA. Contribute to clu0/unet.cu development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/commit/a876282eb845f89aef70c780033ee150aba044b0">Merge pull request #653 from ademeure/cublaslt_refactor · karpathy/llm.c@a876282</a>: Matmul refactor using only cuBLASLt + GELU Fusion</li><li><a href="https://github.com/karpathy/llm.c/pull/657">Remove per-layer attproj and fcproj activation tensors by ademeure · Pull Request #657 · karpathy/llm.c</a>: I&#39;m not sure how we missed this but we don&#39;t actually need these tensors at all for the backward pass at all! Possibly this wasn&#39;t the case when residual/layernorm/recompute were implement...</li><li><a href="https://github.com/microsoft/mup?tab=readme-ov-file#how-mup-works-under-the-hood">GitHub - microsoft/mup: maximal update parametrization (µP)</a>: maximal update parametrization (µP). Contribute to microsoft/mup development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/6">Not all heroes wear capes · Issue #6 · karpathy/llm.c</a>: Just wanted to say thank you on behalf of the community. Thank you Andrej. ❤️ I am closing this post submitting the issue so don&#39;t have to.</li><li><a href="https://github.com/karpathy/llm.c/blame/master/.github/workflows/ci_gpu.yml#L90">Blaming llm.c/.github/workflows/ci_gpu.yml at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/650#issuecomment-2198342446">muP (maximum update parametrization) by gordicaleksa · Pull Request #650 · karpathy/llm.c</a>: Main changes:  Modify random initialization Scale attention scores by 1/d and not 1/sqrt(d) and add an attn_mult Scale activations by 1/width_mult before mapping into logits Update learning rate &amp;...</li><li><a href="https://github.com/karpathy/llm.c/pull/650/">muP (maximum update parametrization) by gordicaleksa · Pull Request #650 · karpathy/llm.c</a>: Main changes:  Modify random initialization Scale attention scores by 1/d and not 1/sqrt(d) and add an attn_mult Scale activations by 1/width_mult before mapping into logits Update learning rate &amp;...</li><li><a href="https://github.com/karpathy/llm.c/pull/650">muP (maximum update parametrization) by gordicaleksa · Pull Request #650 · karpathy/llm.c</a>: Main changes:  Modify random initialization Scale attention scores by 1/d and not 1/sqrt(d) and add an attn_mult Scale activations by 1/width_mult before mapping into logits Update learning rate &amp;...</li><li><a href="https://huggingface.co/datasets/karpathy/fineweb-edu-100B-gpt2-token-shards">karpathy/fineweb-edu-100B-gpt2-token-shards · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/karpathy/llmc-starter-pack">karpathy/llmc-starter-pack · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/662">fix the overfit single batch behavior to actually overfit batch, not … by karpathy · Pull Request #662 · karpathy/llm.c</a>: …microbatch, we do this more cleanly by simply resetting the dataloader every step</li><li><a href="https://github.com/Azure/MS-AMP">GitHub - Azure/MS-AMP: Microsoft Automatic Mixed Precision Library</a>: Microsoft Automatic Mixed Precision Library. Contribute to Azure/MS-AMP development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1256535488454721568)** (1 messages): 

- **Nscale benchmarks AMD MI300X GPUs**: Nscale [benchmarks AMD MI300X GPUs](https://www.nscale.com/blog/nscale-benchmarks-amd-mi300x-gpus-with-gemm-tuning-improves-throughput-and-latency-by-up-to-7-2x) showing performance improvements up to **7.2x** with GEMM tuning. They focus on enhancing **throughput**, **latency reduction**, and handling complex models efficiently.
   - The blog emphasizes the importance of **GEMM tuning** using libraries such as **rocBLAS** and **hipBLASlt** to optimize **GEMM operations**. These tools are critical for maximizing GPU-accelerated tasks' performance, ensuring **higher throughput** and efficient processing.
- **GEMM tuning boosts GPU performance**: Nscale's technical deep dive discusses the impact of **GEMM tuning** on GPU performance. The blog highlights significant throughput benchmarking and performance tuning techniques.
   - By using optimization libraries like **rocBLAS** and **hipBLASlt**, the blog illustrates how **GEMM tuning** can drastically reduce latency and improve computational efficiency. These optimizations are vital for handling intricate models and datasets in AI tasks.

**Link mentioned**: <a href="https://www.nscale.com/blog/nscale-benchmarks-amd-mi300x-gpus-with-gemm-tuning-improves-throughput-and-latency-by-up-to-7-2x">Nscale Benchmarks: AMD MI300x GPUs with GEMM tuning improves throughput and latency by up to 7.2x</a>: Optimising AI model performance: vLLM throughput and latency benchmarks and GEMM Tuning with rocBLAS and hipBLASlt

  

---


### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/1256708419558047825)** (1 messages): 

- **Questions about pruning in RF**: A member referred to a [figure from PyTorch](https://pytorch.org/assets/images/accelerating-neural-network-training/fg8.png) and asked for clarification on the term **"Pruned in RF"**, questioning if **RF** refers to **register file**.
   - Further context was sought on the meaning and significance of **"RF"** in the context of kernel PRs and related papers/articles.
- **Memory access size in CUDA threads explained**: A member quoted, "This means that within a CUDA thread, we want to read/write chunks of 128 bytes at a time," and asked if **128** was the size of memory that can be accessed with one column select.
   - The member is seeking further understanding about the memory access patterns and size in CUDA threads, specifically regarding the optimal chunk size.
- **FP16 (BF16) significance in tile handling clarified**: A query was raised about the calculation "instead of a thread handling a single 4x4 tile, which is only 4x4x2 = 32 byte," asking if the **"x2"** stands for **FP16 (BF16)** 2Bytes.
   - This question highlights the need for clarification on the use of **FP16 (BF16)** data types and their representation in memory handling calculations.
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1257006579518013540)** (2 messages): 

- **Gemma 2 Models Released on Perplexity Labs**: **Gemma 2 models** are now available on [Perplexity Labs](https://labs.pplx.ai). Users are encouraged to try them and share their feedback.
   - The announcement was made with excitement, seeking community input on the new models.
- **Voice-to-Voice Feature in Android App**: The latest version of the Android app introduces a **voice-to-voice feature** with two modes: Hands-free and Push-to-talk. Users can try it and provide feedback in the designated channel.
   - The Hands-free mode starts listening right away when the screen opens, while Push-to-talk requires pressing and holding the Mic button to speak.
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1256323813021585448)** (367 messages🔥🔥): 

- **Claude 3.5 Struggles with Context**: Users report that **Claude 3.5** often forgets context, giving general advice instead of specific answers when asked follow-up questions. Some users noted it's problematic with web search turned on, as it confuses the models, emphasizing the need for manual adjustment.
   - Despite issues, switching from **Claude 3.5** to **Opus** has helped some users with better context retention. The community suggests that **Claude 3.5** might have bugs impacting performance, and **Pro search** should intelligently turn off web search to improve coherence.
- **Perplexity AI's Pro Search Inconsistencies**: Many users experience variability in **Pro search**; sometimes it uses multi-step searches while other times, it does not. One member noted an issue where short paragraphs were handled poorly unless manually switching modes to 'writing'.
   - Users noted the frustration of Perplexity models grabbing content from web search rather than previous prompts. Suggestions included improving how context is prioritized in web searches to maintain dialogue coherence.
- **Testing Context Windows for Different Models**: Community member **dailyfocus_daily** shared results of testing context windows for various models, noting **Opus** got it right at 27k but failed at 46k, indicating a likely 32k context limit in code. **Sonnet 3.5** and **GPT-4o** were praised for consistent 64k token retention.
   - Another user queried if **Claude 3.5 Sonnet** really limited at December 2022 information, with responses confirming knowledge cut-off issues due to different system prompts by **Claude.com** and **Perplexity.com**.
- **Claude on Perplexity and User Limits**: **Claude 3.5** on Perplexity is reported not to give the full 200k tokens as claimed, with a common limit set around 32k tokens. **Users from Complexity** confirmed these constraints while discussing the broader token limit inconsistencies on the platform.
   - Feedback touched on the **Perplexity Pro plan**, noting a maximum of 600 uses per 24-hour period with **Opus** capped at 50/24h. Users were directed to official resources to understand limitations and suggestions for improving usability and accessibility.
- **General User Feedback and Support Queries**: Users frequently asked about **Perplexity AI's capabilities and limitations**, such as custom instruction influences, issues with document reattachment in threads, and specific model behavior discrepancies. **User input** reflected a need for improved UI and better alignment of model capabilities with user expectations.
   - Questions about subscription plans and educational discounts appeared, requesting clear guidelines for group accounts and pro usage. Some feedback from **non-pro users** highlighted inconsistent model performance and limitations in switching between search focuses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/discord-light-mode-light-mode-benny-beni-gif-25360134">Discord Light Mode Benny GIF - Discord Light Mode Light Mode Benny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/aoikSxHXBYw?si=3jPRoPJmoJhZ6x1k">Mixture of Agents (MoA) BEATS GPT4o With Open-Source (Fully Tested)</a>: Full test of Mixture of Experts implementation.Subscribe to my newsletter for a chance to win a Dell Monitor: https://gleam.io/otvyy/dell-nvidia-monitor-1 (O...</li><li><a href="https://open.spotify.com/track/1wf9F3L1B11i9WTfvwnfMo?si=IXCqWpDxTz6CvZZKj4E1zg">The Chemical Worker&#x27;s Song (Process Man)</a>: Great Big Sea · Song · 1995</li><li><a href="https://by-ai-monnef-9ff5d9c2460ae15d70e737f77eab719c6e8a4c64c2f99ca1c2.gitlab.io/2024/opus_50-ball-game/">opus_50 Game</a>: no description found</li><li><a href="https://open.spotify.com/track/6pPCkAzVYapjObH73BWu9t?si=aeucsMekQxG28MzDt0Gz2Q&context=spotify%3Aalbum%3A53VtJpStdfdFG2MSJRgZgC)">Father</a>: Sabaton · Song · 2022
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1256330807539400734)** (16 messages🔥): 

- **Minecraft Repair Mechanics Misleads Kids**: [Minecraft Repair Mechanics](https://www.perplexity.ai/page/Minecraft-Repair-Mechanics-NdRggXKXRXyGY8LgKsp1dQ) was criticized for potentially giving children an incorrect understanding of real-world tool repair. The details of the mechanics are discussed in the provided link.
   - The community expressed concern over the educational aspects of games like Minecraft. They highlighted that such mechanics could confuse kids about real tool maintenance.
- **Impact of Global Warming on Smartphones**: A discussion emerged around the [increasing impact of global warming](https://www.perplexity.ai/page/the-increasing-impact-of-globa-xEkc7Nd9RA2tXXMoC4VT3g) on smartphone charging performance. The page details how rising temperatures affect battery efficiency.
   - Members shared their experiences and concerns about this issue. They debated potential solutions, with emphasis on the necessity for more resilient battery technologies.
- **Starting Perplexity: A Tale**: The [story behind starting Perplexity](https://www.perplexity.ai/search/the-story-behind-starting-perp-DnZ.yJgfSM28Ra9_h2uKWg) was shared through multiple links. This narrative not only covers the inception but also delves into the challenges faced.
   - Members reflected on the startup journey, offering insights and opinions. Direct quotes highlighted the perseverance required to launch and sustain innovative projects.

**Link mentioned**: <a href="https://www.youtube.com/embed/lJbAw0wCc0I">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1256349528102277161)** (19 messages🔥): 

- **Date-Specific Results in Perplexity API**: A member suggested using `after:2024-05-28` in queries to limit results to new information from the last 30 days in the Perplexity API. However, it was noted that this might increase hallucination, generating articles from the future.
   - Another member recommended applying for the closed beta to get the `search_domain_filter` feature, as a potential solution to this issue. They shared links to a form and an email address ([fill out this form](https://perplexity.typeform.com/to/j50rnNiB) and [send email](mailto:api@perplexity.ai)).
- **Perplexity API Post Request Example**: A member shared an example of making a POST request to the Perplexity API, including the required headers and data. They detailed the specific usage for retrieving headlines after a certain date.
   - Responses from this approach were demonstrated, showing the versatility in action. The example entailed obtaining recent headlines from the NPR website using the appropriate prompts and API configuration.
- **Pro Account Issues with Perplexity Labs Playground**: A user faced issues with the Perplexity Labs Playground recognizing their Pro account when logged in using Apple ID. Another member clarified that the Playground is free to access for everyone, and the 'Try Perplexity' button is mainly a marketing tool.
   - The concern about billing overlap if the user canceled and switched to a regular email account was also discussed. This addressed the misunderstandings related to account functionalities both in free and Pro versions.
- **Discrepancy Between Perplexity API and Web UI Results**: A member observed a significant difference in the quality of results between the Perplexity API and its Web UI. They shared an example query where the Web UI provided much better information compared to the API.
   - They were advised to post their issue on the Perplexity forums and apply for better filtering options available through the closed beta. This indicated ongoing enhancements and user feedback considerations for the API.
- **Access Issues with API Settings Page**: A user reported problems accessing the API settings page, experiencing continuous loading on the web. Another member suggested switching browsers, specifically mentioning Safari as a solution that worked fine.
   - Clearing the cache on Chrome was recommended as a potential fix. Post-switching, the user confirmed that the issue was resolved, indicating that browser settings might interfere with accessing certain features.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.npr.org"">no title found</a>: no description found</li><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: no description found</li><li><a href="https://docs.perplexity.ai/discuss">Discussions</a>: no description found</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api form</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1257320294259888290)** (3 messages): 

- **Synthetic Data Collection Ramps Up With 1 Billion Personas**: [A collection of 1 billion diverse personas](https://x.com/arankomatsuzaki/status/1807593343007818065) has been automatically curated from web data to scale synthetic data creation, yielding **massive gains on MATH benchmarks** from 49.6 to 64.9. The approach leverages these personas to facilitate diverse and scalable synthetic data across various scenarios.
   - The [Persona Hub GitHub repository](https://github.com/tencent-ailab/persona-hub) and [arXiv paper](https://arxiv.org/abs/2406.20094) detail the methodology used, showcasing use cases that include synthesizing **high-quality mathematical problems**, logical reasoning tasks, and NPCs for games.
- **PhyloLM Brings Phylogenetics to LLMs**: [PhyloLM](https://arxiv.org/abs/2404.04671) introduces a method that adapts phylogenetic algorithms to Large Language Models (LLMs) to explore their relationships and predict performance characteristics. The method uses similarity metrics of LLMs' output to construct dendrograms, capturing known relationships among 111 open-source and 45 closed models.
   - This method's phylogenetic distance successfully predicts performance in standard benchmarks, [validating its functional utility](https://arxiv.org/html/2404.04671) and enabling cost-effective LLM capability estimations without transparent training data.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.04671">PhyloLM : Inferring the Phylogeny of Large Language Models and Predicting their Performances in Benchmarks</a>: This paper introduces PhyloLM, a method adapting phylogenetic algorithms to Large Language Models (LLMs) to explore whether and how they relate to each other and to predict their performance character...</li><li><a href="https://x.com/arankomatsuzaki/status/1807593343007818065">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Scaling Synthetic Data Creation with 1,000,000,000 Personas  - Presents a collection of 1B diverse personas automatically curated from web data - Massive gains on MATH: 49.6 -&gt;64.9  repo: https://g...</li><li><a href="https://arxiv.org/abs/2406.20094">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>: We propose a novel persona-driven data synthesis methodology that leverages various perspectives within a large language model (LLM) to create diverse synthetic data. To fully exploit this methodology...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1256455180774408192)** (4 messages): 

- **Explore Dreams with Android and Human Dataset**: Check out the [Android and the Human dataset](https://huggingface.co/datasets/gustavecortal/the-android-and-the-human), which includes 10,000 real dreams from DreamBank and 10,000 generated dreams using the **Oneirogen model**. This resource allows for differentiating between real and generated dreams and supports classification tasks.
   - Generated dreams were produced using **Oneirogen (0.5B, 1.5B, and 7B)** language models, providing a benchmark for analyzing dream narratives. The dataset is ideal for investigating the distinctions between real and synthetic dream content.
- **Snorkel AI's Chat-Optimized Model**: [Snorkel-Mistral-PairRM-DPO](https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO) by Snorkel AI is now available for chat purposes, with live tests on the [Together AI playground](https://api.together.xyz/playground/chat/snorkelai/Snorkel-Mistral-PairRM-DPO). This model is also accessible through Together AI's API for broader use.
   - A recent [Snorkel AI blog post](https://snorkel.ai/new-benchmark-results-demonstrate-value-of-snorkel-ai-approach-to-llm-alignment/) details the model's alignment strengths. The model operates on the standard speed of HF's 7B model text inference endpoint, thanks to integration efforts by the Together AI team.
- **PersonaHub Revolutionizes Data Synthesis**: [PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) offers a persona-driven dataset methodology to scale synthetic data creation with over 1 billion personas. These personas cover diverse perspectives, facilitating rich data synthesis for a variety of applications.
   - Introduced in the paper, *Scaling Synthetic Data Creation with 1,000,000,000 Personas*, this methodology enhances LLM training by simulating complex instructional, logical, and mathematical scenarios. PersonaHub demonstrates versatility and scalability in generating synthetic data for intricate research and practical applications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/snorkelai/Snorkel-Mistral-PairRM-DPO">snorkelai/Snorkel-Mistral-PairRM-DPO · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/gustavecortal/the-android-and-the-human">gustavecortal/the-android-and-the-human · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/proj-persona/PersonaHub">proj-persona/PersonaHub · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1256328863865241630)** (10 messages🔥): 

- **Lab-Grown Blood Enters Clinical Trials**: Cambridge researchers have commenced the world’s first clinical trial of **lab-grown red blood cells** for transfusion. Watch the [YouTube video](https://youtu.be/o0IM-FcX_9U) for more details.
   - Excitement and humor around the topic included remarks about vampires preferring non-artificial blood and **holding back research progress**.
- **Open Model Initiative Launches**: The **Open Model Initiative** has been launched to promote open-source AI models for image, video, and audio generation. Read the full announcement on [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1do5gvz/the_open_model_initiative_invoke_comfy_org/).
   - The initiative aims to produce high-quality, competitive models with open licenses, ensuring **free and unrestricted access** for all.
- **Mixture of Agents Using Langchain**: [YouTube video](https://www.youtube.com/watch?v=VNy7CM23WA0) explores how to implement **Mixture of Agents (MoA)** using langchain. This approach aims to harness the collective strengths of multiple agents.
   - The video presents a hands-on tutorial, focusing on the collaborative aspect of using various agents to improve functionality and performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=VNy7CM23WA0">Mixture of Agents (MoA) using langchain</a>: Today we will implement mixture of agents using langchain.We introduce Mixture of Agents (MoA), an approach to harness the collective strengths of multiple L...</li><li><a href="https://youtu.be/o0IM-FcX_9U?feature=shared">First ever clinical trial of lab-grown red blood cell transfusion</a>: Cambridge researchers are taking part in the world’s first clinical trial of red blood cells that have been grown in a laboratory for transfusion into anothe...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1do5gvz/the_open_model_initiative_invoke_comfy_org/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1256373456300146718)** (60 messages🔥🔥): 

- **Gemma 2 9B challenges Phi3 medium**: Discussions surfaced about whether **Gemma 2 9B** outperforms **Phi3 medium** for synthetic generation, with benchmark comparisons found on [Hugging Face](https://huggingface.co/google/gemma-2-9b). Members debated merits including context length but no definitive consensus was reached.
   - Members noted **Phi3** boasts a larger context window, creating skepticism about **Gemma 2 9B**'s comparative edge. The vibe check for **Gemma 2 9B** is ongoing, as multiple users are testing its performance and feedback.
- **SpecExec accelerates LLM inference**: [SpecExec](https://www.together.ai/blog/specexec) promises speculative decoding for LLM inference, achieving speeds of 4-6 tokens per second with 4-bit quantization on consumer GPUs. The method potentially speeds up inference by up to 18.7x, making it an attractive option for running large models efficiently on consumer hardware.
   - The technique involves using a draft model to predict a sequence of tokens that can be quickly verified by the main model. Discussions revolved around suitable model families for speculative decoding and the importance of compatible vocab sizes.
- **Meta Chameleon vs. CogVLM2 and LlavaNext for vision captioning**: Users sought recommendations for vision-captioning models, weighing options including **Meta Chameleon**, **CogVLM2**, and **LlavaNext**. Reports noted that **Florence 2 large** also enters the competition, but its detailed captioning may fall short in some instances.
   - "Query prompts significantly affect benchmark results for vision-language models," members cautioned. Detailed discussions highlighted how **Florence 2** might not be the best, emphasizing personal testing due to variability in outputs.
- **FireFunction V2: a new function calling model**: Launched by Fireworks, **FireFunction V2** offers state-of-the-art function calling, scoring 0.81 vs GPT-4o at function-calling. Built on **Llama 3**, it supports parallel function calling and can handle up to 20 function specs at once, aiming to improve instruction-following tasks.
   - [Detailed info](https://huggingface.co/fireworks-ai/llama-3-firefunction-v2) shows significant improvements over its predecessor, making it a robust choice for function calling needs. The model retains Llama 3's high conversational capabilities, while introducing new efficiencies in multi-turn chats and structured information extraction.
- **UGI Leaderboard on Hugging Face unveiled**: The launch of the [Uncensored General Intelligence Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard) on Hugging Face sparked interest. The leaderboard ranks models based on their ease of use and general intelligence, providing insights into the latest and most effective models.
   - Specifically, **Teknium** questions how abliterated models like **L3 70B** rank compared to base **Mixtral**, given differences in model refusal behaviors. The discussion underscores the community's engagement with benchmarking and comparative performance assessments for AI models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard">UGI Leaderboard - a Hugging Face Space by DontPlanToEnd</a>: no description found</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://huggingface.co/fireworks-ai/llama-3-firefunction-v2">fireworks-ai/llama-3-firefunction-v2 · Hugging Face</a>: no description found</li><li><a href="https://x.com/WABetaInfo/status/1806101428609622181">Tweet from WABetaInfo (@WABetaInfo)</a>: 📝 WhatsApp beta for Android 2.24.14.7: what&#39;s new?  WhatsApp is working on a feature to choose the Meta AI Llama model, and it will be available in a future update!  https://wabetainfo.com/whatsa...</li><li><a href="https://x.com/nicolas__yax/status/1807761080917045489">Tweet from Nicolas Yax (@nicolas__yax)</a>: From the similarity matrix, we can compute dendrograms. These are able to accurately capture the different finetuning relationships between models (like OH @Teknium1 @maximelabonne and others) when co...</li><li><a href="https://x.com/honghuazhang2/status/1806727439823102325?s=46">Tweet from Honghua Zhang (@HonghuaZhang2)</a>: Proposing Ctrl-G, a neurosymbolic framework that enables arbitrary LLMs to follow logical constraints (length control, infilling …) with 100% guarantees.    Ctrl-G beats GPT4 on the task of text editi...</li><li><a href="https://www.together.ai/blog/specexec">SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices</a>: no description found</li><li><a href="https://arxiv.org/abs/2305.17333v3">Fine-Tuning Language Models with Just Forward Passes</a>: Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively large amount of memory. Zeroth-order (ZO) methods c...</li><li><a href="https://x.com/NousResearch/status/1793637803701780797">Tweet from Nous Research (@NousResearch)</a>: Nous Research is hiring!  Apply Here: https://forms.gle/UWx2Pht8qioi1bjAA</li><li><a href="https://x.com/rauchg/status/1806899014312595948">Tweet from Guillermo Rauch (@rauchg)</a>: If you ask Google Chrome&#39;s embedded model, Gemini Nano, &#34;who is he&#34;, with no additional context, it reliably answers @elonmusk 🤨</li><li><a href="https://huggingface.co/google/gemma-2-9b">google/gemma-2-9b · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1256903083829624882)** (6 messages): 

- **10k DPO dataset size validation**: A member asked if ~10k samples are sufficient for a DPO dataset. Another member confirmed that while more samples are always better, **10k is quite good** and aligns the model well.
   - The same member shared that they started with 10k samples and observed significant improvements when gradually increasing the dataset size.
- **Speculations on Hermes Gemma 27b**: **Hermes Gemma 27b** was briefly mentioned in a question about its release or existence. There were no additional details or confirmations provided on this topic.
   - No further discussion ensued about **Hermes Gemma 27b**, leaving its status and specifications still in question.
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1256356923113214004)** (163 messages🔥🔥): 

- **Self-Retrieval Paper Dissection**: A detailed discussion on the [self-retrieval paper](https://arxiv.org/abs/2403.00801) examined the dataset construction and training methods, noting concerns about potential repetition during training. Specifically, each sentence in a document is indexed, and the document itself is paired for training, which might cause model overfitting.
   - The conversation highlighted potential drawbacks and overfitting issues, with one member expressing frustration, saying, *'this I get, simple next token prediction; at least I think I get'.* Another member decided against using the paper due to these concerns.
- **Innovative Dataset Search Tool for HF**: A member shared a newly updated tool for dataset searching [holdingfaceSEARCH](https://github.com/EveryOneIsGross/holdingfaceSEARCH) that can download and search HF datasets by selected keys, allowing local embeddings for RAG tasks. The tool aims to simplify the process by enabling a 'no frills' RAG-chat for quick ingestion and search.
   - The creator emphasized the flexibility of the tool, noting its ability to cater to different dataset requirements and prompting other members to share their projects and tools. This led to a broader discussion on efficient dataset handling techniques.
- **Groq and The Future of Custom Chips**: Groq emerged as a notable player in the AI hardware space, potentially offering 1000 tokens per second for a 70B model, challenging Nvidia's dominance. The conversation suggested that Groq might revolutionize the AI computation market if they provide effective training and serving capabilities.
   - Members discussed the competitive landscape, noting that Nvidia's CUDA still gives it an edge for training applications. The future of LPUs was debated, with one member highlighting, *'I hope groq does it. I still think etched is a marketing gimmick'.*
- **Microsoft and OpenAI's Mega Data Center**: Microsoft and OpenAI are teaming up on a colossal data center project called Stargate, potentially costing over $100 billion, according to [The Information](https://www.theinformation.com/articles/microsoft-and-openai-plot-100-billion-stargate-ai-supercomputer). This venture aligns with Microsoft's nuclear energy strategy to power its energy-hungry AI ambitions.
   - The discussion also touched on the potential environmental and logistical challenges of such an ambitious project, suggesting significant impacts on the energy sector for future data centers.
- **Anthropic's Market Strategy Compared to OpenAI**: Members praised Anthropic for its approach, contrasting it positively against OpenAI. The discussion acknowledged that Anthropic hasn't faced significant public complaints, unlike OpenAI, whose co-founders have started seeking safer superintelligence options elsewhere.
   - One member humorously remarked, *'saw the new startup? sutskever's?'*, highlighting their preference for Anthropic's current trajectory. This reflects broader industry dynamics and strategic shifts among AI companies.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2023/9/26/23889956/microsoft-next-generation-nuclear-energy-smr-job-hiring">Microsoft is going nuclear to power its AI ambitions</a>: Microsoft is hiring someone to lead its nuclear strategy.</li><li><a href="https://arxiv.org/abs/2406.19292">From Artificial Needles to Real Haystacks: Improving Retrieval Capabilities in LLMs by Finetuning on Synthetic Data</a>: Recent studies have shown that Large Language Models (LLMs) struggle to accurately retrieve information and maintain reasoning capabilities when processing long-context inputs. To address these limita...</li><li><a href="https://arxiv.org/abs/2403.00801">Self-Retrieval: Building an Information Retrieval System with One Large Language Model</a>: The rise of large language models (LLMs) has transformed the role of information retrieval (IR) systems in the way to humans accessing information. Due to the isolated architecture and the limited int...</li><li><a href="https://www.complex.com/music/a/jaelaniturnerwilliams/the-weeknd-interviews-10-year-old-self">Tweet from The Weeknd Interviews 10-Year-Old AI Version of Himself, Tells Him ‘You Say Superpower a Lot&#x27;</a>: Things got strange when The Weeknd used OpenAI to interview his younger self.</li><li><a href="https://github.com/tencent-ailab/persona-hub">GitHub - tencent-ailab/persona-hub</a>: Contribute to tencent-ailab/persona-hub development by creating an account on GitHub.</li><li><a href="https://arxiv.org/abs/2310.11511">Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection</a>: Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate. Ret...</li><li><a href="https://github.com/EveryOneIsGross/holdingfaceSEARCH">GitHub - EveryOneIsGross/holdingfaceSEARCH: I simple import or CLI tool for downloading and searching HF datasets</a>: I simple import or CLI tool for downloading and searching HF datasets - EveryOneIsGross/holdingfaceSEARCH</li><li><a href="https://www.forbes.com/sites/cindygordon/2024/03/31/microsoft-and-openai-partnering-on-stargate-a-100b-us-data-center/">Microsoft And OpenAI Partner On $100 Billion U.S. Data Center, Report Says</a>: Sky rocketing demands for generative artificial intelligence have accelerated demands for AI-centric data centers that can handle more advanced tasks.</li><li><a href="https://docs.google.com/spreadsheets/d/1f5fbPxhjGrmPqhbM0exOCX2vAzffRWufhF7QBp24OMw/edit?gid=0#gid=0">RAG Data Synthesis</a>: Sheet1  Domain,Curriculum file,Source/links,HF repo,Size (rows),Status,Who&#39;s working,Reviewer,Review Notes Websearch Wikipedia Codebase,WIP,Bexboy Academic Papers Books,WIP,EveryoneIsGross Finance...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1256513728971804724)** (31 messages🔥): 

- **Command Issues in WorldSim**: Users reported issues with using the **!save** command in **WorldSim**, which resulted in a wipeout instead of saving the discussion. Another concern involved the **!back** command, which was clearing the entire chat instead of going back to the last chat.
   - Apyh acknowledged both issues and assured they would be investigated, with **!back** already fixed. Vuyp also noted random stopping in generation after one line with **Sonnet 3.5**.
- **Claude 3.5 Sonnet Support Rolled Out**: **WorldSim** now supports the Claude 3.5 Sonnet model option. This update was announced with caution, mentioning it works great for the **World Client** but is less creative in **WorldSim**.
   - Apyh and other users discussed how **Claude 3.5 Sonnet** tends to stop generating responses abruptly, with Apyh assuring that new prompts are being developed to resolve this behavior.
- **Credits Inquiry for Beta Testers**: Rundeen queried a drop from $50 to $4 in credits, asking if there was a weekly allowance system. Apyh clarified that beta testers were given free credits only if they signed up with the same email for the live version and offered to help port the credits.
   - Apyh invited Rundeen to DM their account id or email to resolve the credits issue, showing a hands-on approach to customer support.
- **Accidental Reset in WorldSim**: Keksimus.maximus accidentally reset their **WorldSim** scenario involving dragons by clicking the **WorldClient** button. Apyh explained how to use `**!list**` and `**!load**` commands to recover previous chats.
   - This reset incident highlights the need for clear instructions and possibly UI improvements to prevent accidental resets in **WorldSim**.
- **Future WorldSim Bot Enhancements**: Rezonaut suggested sharpening some bots for question/answer functionalities that link to various areas of Nous and its available resources. Teknium redirected this kind of bot usage to the appropriate channel.
   - The conversation pointed to possible enhancements in bot functionalities, highlighting the ongoing efforts to improve the Nous user experience.
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1257051471598321684)** (7 messages): 

- **Errors as values spark debate**: Members discussed using **errors as values** with `Variant[T, Error]` in Mojo, noting current limitations. **Errors** and **Exceptions** were contrasted as different types with distinct handling challenges.
   - "Mojo's try/except is syntax sugar," one user remarked, highlighting a distinction in how Mojo treats errors internally. The absence of **match statements** was also criticized for hindering elegant error handling.
- **Setting up Mojo on Ubuntu**: A new user sought help setting up **Mojo on Ubuntu**, specifically mentioning issues. Another member shared success running Mojo on **Ubuntu 24.04** on a **Raspberry Pi 5** and offered to assist.
   - The troubleshooting request underscores common challenges encountered by **new users** when setting up Mojo on different **Ubuntu versions**. Community support played a key role in addressing these issues.
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1257366341191471115)** (1 messages): 

- **New Mojo Coding Challenges Channel Launches**: A new channel has been launched for [monthly Mojo coding challenges](https://discord.com/channels/1087530497313357884/1255303604894437388). Community members can subscribe for alerts by reacting with the <:mojomarathon:1255306448632807424> emoji.
   - The monthly challenges are organized by a community member, moderator, and maintainer of the `basalt` ML Framework. Find more details in the channel and the [GitHub repo](https://github.com/Benny-Nottonson/Mojo-Marathons).
- **Subscribe for Monthly Mojo Challenges Alerts**: Members can get alerts for new challenges by reacting with the <:mojomarathon:1255306448632807424> emoji. This feature ensures you stay updated with the latest Mojo coding challenges.
   - The challenges provide a great opportunity to engage with the community and enhance your skills in the `basalt` ML Framework. Check out the [GitHub repo](https://github.com/Benny-Nottonson/Mojo-Marathons) for more details.

**Link mentioned**: <a href="https://github.com/Benny-Nottonson/Mojo-Marathons">GitHub - Benny-Nottonson/Mojo-Marathons</a>: Contribute to Benny-Nottonson/Mojo-Marathons development by creating an account on GitHub.

  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1256346953474768947)** (32 messages🔥): 

- **PyTorch and Mojo insights**: **PyTorch in Mojo**: Discussion about using PyTorch in Mojo, highlighting its flexibility and popularity among AI developers. A [YouTube documentary](https://www.youtube.com/watch?v=rgP_LBtaUEc) on PyTorch's evolution and impact was shared.
   - **Quick iteration adoption**: An interesting note was made about how quick iteration first allowed for PyTorch's adoption and performance improvements being worked in later stages.
- **TacoBot's Revival**: **ModularBot retired, TacoBot introduced**: ModularBot was retired and later reintroduced as TacoBot, with users humorously interacting with the new bot. TacoBot amusingly continues to share its love for tacos.
   - **TacoBot avatar**: Users suggested and praised TacoBot’s new avatar provided by ChatGPT, which seemed to capture its spirit perfectly.
- **Facebook's New LLM Compiler**: **LLM Compiler by Facebook**: Enthusiastic discussion about Facebook's new [LLM Compiler model](https://huggingface.co/facebook/llm-compiler-13b-ftd) specialized in compiler optimization. It has been trained on various assembly codes and can replicate the functionality of the clang compiler.
   - **Free for use**: The LLM Compiler was noted to be available for both research and commercial use, with two versions catering to foundational and fine-tuned model needs.
- **Cody supports Mojo**: **Cody and Mojo integration**: Users discussed the use of Cody to help with Mojo coding, noting its potential to guess language features due to its Python-like syntax.
   - **Testing advanced features**: A plan was mentioned to test Cody with more advanced, Mojo-specific tasks, especially related to SIMD, to evaluate its capability.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=rgP_LBtaUEc">Official PyTorch Documentary: Powering the AI Revolution</a>: This film unveils the authentic narrative of PyTorch’s inception, attributing its existence to a dedicated group of unsung heroes driving technological innov...</li><li><a href="https://huggingface.co/facebook/llm-compiler-13b-ftd">facebook/llm-compiler-13b-ftd · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1256368212367507577)** (149 messages🔥🔥): 

- **Mojo Discusses I/O and Async APIs**: Members discussed the current limitations in Mojo's I/O module, specifically the need to interface with Python to read from stdin. The conversation turned towards future improvements in async APIs, with many advocating for the potential use of io_uring for better performance.
   - Darkmatter__ emphasized the importance of a performant I/O subsystem for high-speed networking, arguing that io_uring and completion-based APIs should be favored despite their complexity. Lukashermann.com countered that maintainability and ease of understanding are also crucial for the average user, suggesting simpler abstractions.
- **Mojo's Future as a Systems Language**: Twilight_muse and darkmatter__ expressed excitement about Mojo's potential to revolutionize systems programming, praising its MLIR backend and ownership model. They highlighted the need for Mojo to learn from Rust's mistakes, especially in async I/O implementation, to become a robust systems language.
   - Darkmatter__ shared insights into networking performance bottlenecks and emphasized the advantages of io_uring over traditional polling-based I/O. The discussion underscored how Mojo can bridge the gap between high-performance systems programming and accessible general-purpose coding.
- **Sum Types and Union Types Debate**: Nick.sm and soracc debated the merits and pitfalls of sum types (enums) versus union types, with Nick.sm arguing for a need to describe different sets of options that are semantically related. Soracc suggested that sum types are a fundamental concept rooted in intuitionistic type theory, while Nick.sm compared them to inheritance, hinting at their limitations.
   - The discussion shifted towards practical applications and how Mojo might implement such types to avoid the issues faced in other languages. This ongoing debate reflects deeper considerations in Mojo's language design to balance mathematical rigor with real-world usability.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/get-started">Get started with Mojo🔥 | Modular Docs</a>: Install Mojo now and start developing</li><li><a href="https://docs.modular.com/mojo/manual/python/">Python integration | Modular Docs</a>: Using Python and Mojo together.</li><li><a href="https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/">ioringapi - Win32 apps</a>: Provides APIs for creating and managing I/O rings.</li><li><a href="https://www.youtube.com/watch?v=7qvVMUSxqz4">2024 EuroLLVM - How Slow is MLIR</a>: 2024 European LLVM Developers&#39; Meetinghttps://llvm.org/devmtg/2024-04/------How Slow is MLIRSpeaker: Mehdi Amini, Jeff Niu------Slides: https://llvm.org/devm...</li><li><a href="https://github.com/modularml/mojo/discussions/3049">Self-holding and open-sourced, which comes first? · modularml/mojo · Discussion #3049</a>: As far as I know, the Mojo compiler is written in C++. I am wondering whether Mojo will first become self-holding (compiler written in Mojo), or the compiler will first be open-sourced. May I ask a...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1257368905618292806)** (7 messages): 

- **Setting up the Environment for Mojo Marathons**: A member requested **details on the environment** setup for Mojo Marathons, including core isolation and timer tick configurations. They offered to help with the setup and mentioned switching the timer to **ARM MSRs** to avoid syscall overhead.
   - Another member responded that they have an existing script for benchmarks and would be interested in comparing notes. They mentioned plans to write up recommendations and submit a PR to use **ARM MSRs or rtdsc**.
- **Impact of Timer Tick Isolation**: A member mentioned that **timer tick isolation** isn't commonly used due to its requirement of configuration at boot time and its interference with most benchmark libraries. They noted, however, that it avoids **random interruptions**.
   - They also stated that unless someone tries to **parallelize** this process, the setup should be straightforward, mostly involving kernel command line arguments and the use of **taskset**.
- **Using libhugetlb to Isolate TLB State**: A member suggested the use of **libhugetlb** to isolate TLB state for the benchmarks. They implied this would enhance the consistency of benchmark results.
   - The conversation did not delve further into the specifics of **libhugetlb** usage but suggested its potential benefits in maintaining isolation and consistency.
  

---


### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1256636887276326973)** (6 messages): 

- **Testing the new apps section**: A member mentioned they were experimenting with the **new apps section** and didn't realize it would appear in the chat.
   - Another member asked for clarification on what the **apps section** meant, indicating some confusion or lack of awareness about this new feature.
- **Spam handling by moderators**: A member reported **spam** and was informed to ping a moderator if the auto moderator fails to catch it.
   - The spam was promptly handled by one of the moderators, who thanked the member and encouraged them to report future incidents.
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1256492979745591296)** (28 messages🔥): 

- **Mojo Nightly Compiler Updated**: A new nightly Mojo compiler `2024.6.2905` has been released with several updates, including new `min` and `max` overloads for `UInt`, and an update to the `UInt` type changelog. Check out the raw diff [here](https://github.com/modularml/mojo/compare/439d86d608d3b6c12cead112eb651752ba1ad40d...9cf3a83e0eb661a2263ae117c921bb004df6721c) and the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
   - Problems with recent nightly builds were identified and a fix is in progress. Changes causing internal dependency issues have been reverted to ensure future releases are stable.
- **Seeking Stdin Stream Testing Methods**: Users discussed methods for testing functions involving stdin streams, such as writing to stdin for each test in a file. Recommendations involved using independent processes, threads, or tools like `tee(2)`.
   - Additional complexities such as handling EOF were mentioned, highlighting the potential need to clear stdin between tests.
- **Challenges with TLS in Mojo**: A member inquired about building a Linux CLI utility in Mojo that must handle JSON HTTP requests with TLS. The community noted current io in Mojo typically involves calling out to C or Python, making TLS integration particularly challenging.
   - Given the immaturity of Mojo's FFI, wrapping libraries like OpenSSL or BoringSSL would be difficult. This presents significant hurdles for anyone needing TLS.
- **Mojo's Potential Beyond Early Challenges**: Despite current limitations, members expressed excitement about Mojo's potential to replace languages like Java or .NET for business applications. Anticipation is high for features like Wasm as a compilation target.
   - Acknowledging the growing pains, contributors remain optimistic about incremental improvements and enhancements in Mojo's functionality.
- **Mojo Standard Library Contributors Eligible for Merch**: Contributors to the Mojo standard library can reach out to team members for Mojo merchandise. Jack Clayton confirmed this in response to an inquiry in the chat.
   - Participants were encouraged to engage with the community and contribute to the library for the chance to receive these rewards.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=DJtef410XaM">The Clean Architecture in Python</a>: Brandon Rhodeshttp://pyvideo.org/video/2840/the-clean-architecture-in-pythonhttp://pyohio.org/schedule/presentation/58/Even design-conscious programmers find...</li><li><a href="https://docs.modular.com/mojo/stdlib/tensor/tensor/Tensor#__init__">Tensor | Modular Docs</a>: A tensor type which owns its underlying data and is parameterized on DType.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1256662493963751487)** (9 messages🔥): 

- **Mojo Marathons kick off!**: Welcome to **Mojo Marathons**! Hosted by [@Benny-Nottonson](https://github.com/Benny-Nottonson), this monthly competition challenges Mojicians to showcase their skills for prizes and community recognition.
   - The first challenge involves creating an optimized **matrix multiplication algorithm using Mojo**, with participants forking the [repository](https://github.com/Benny-Nottonson/Mojo-Marathons) and submitting their solutions by the end of the month.
- **Benchmarking suggestions for better precision**: A member suggested using the **stdlib benchmark module** for more precise results in the Mojo Marathons challenge. This includes tools that run operations multiple times with warmups.
   - [@Benny-Nottonson](https://github.com/Benny-Nottonson) agreed, inviting the member to write an improved testing module and submit a PR.
- **Compilation issues with main.mojo**: A member faced errors while compiling `main.mojo` in the July challenge. The errors pointed at missing attributes 'load' and 'store' in `DTypePointer[Type, 0]`.
   - [@Benny-Nottonson](https://github.com/Benny-Nottonson) clarified that the code is meant to run on the latest stable build of **Mojo**, not the nightly build.
- **Matrix multiplication function output issue**: A member encountered an *AssertionError* when comparing the output of two matrix multiplication functions, with a difference of `0.015625` between `24.4375` and `24.453125`.
   - [@Benny-Nottonson](https://github.com/Benny-Nottonson) confirmed this indicates an incorrect output from the user’s function.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/benchmark/bencher/ThroughputMeasure">ThroughputMeasure | Modular Docs</a>: Records a throughput metric of metric BenchMetric and value.</li><li><a href="https://github.com/Benny-Nottonson/Mojo-Marathons/blob/main/July2024/src/test.mojo#L72">Mojo-Marathons/July2024/src/test.mojo at main · Benny-Nottonson/Mojo-Marathons</a>: Contribute to Benny-Nottonson/Mojo-Marathons development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1257248877497552948)** (3 messages): 

- **Analytics on OpenRouter temporarily down**: Our analytics on [OpenRouter](https://openrouter.ai/rankings) are temporarily down due to a **database operation mistake**. The team is working on fixing the data and apologizes for the inconvenience.
   - The error does not affect customer data and credit. Users were reassured with a *"sorry for the inconvenience"* message.
- **Customer data unaffected**: Despite the downtime of analytics on [OpenRouter](https://openrouter.ai/rankings), **customer data and credits remain unaffected**. Users were informed that the team is addressing the issue.
   - The notification included an apology for any inconvenience caused and assured users that **customer data security** is intact.

**Link mentioned**: <a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: Language models ranked and analyzed by usage across apps

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

wyverndryke: These are amazing you guys! 😄
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1256356791814455306)** (155 messages🔥🔥): 

- **OpenRouter API response quality concerns**: A user expressed concerns that prompts tested in OpenRouter produce different, lower-quality responses when executed via OpenRouter API. Suggestions to debug included checking the request sample and ensuring it matches the setup on OpenRouter.
   - Other members discussed setting up request samples correctly and considered using Bedrock or other moderated models for better quality responses. One member mentioned using a reverse proxy for Bedrock setup.
- **Using Geminis Model context discrepancy**: A user found an error on the website showing Gemini's context as 2.8M while only 1M was observed, leading to confusion. Another member clarified that OpenRouter counts tokens differently, typically shorter.
   - Additional insights highlighted that OpenRouter's token counting results in shorter tokens compared to usual standards, hence the perceived discrepancy. Alternative routes for feedback and reporting errors were discussed.
- **DeepSeek Code V2 lauded for accuracy**: A member praised the DeepSeek Code V2 through OpenRouter API for its high accuracy in solving calculus problems and coding implementations. They found the model both effective and economical.
   - Another member confirmed the model is the full 263B one as it routes through the DeepSeek API, suggesting its accuracy and power in various use cases. External links were provided for further details on the model.
- **Issues with embedding models and local alternatives**: A member inquired about OpenRouter's support for embedding models, but another clarified that direct API usage for embeddings is recommended due to their low cost and compatibility concerns.
   - Suggestions were made to use local embedding models, highlighting their advantage in maintaining consistency with already generated embeddings. Nomic models on HuggingFace were recommended as popular choices.
- **Error with Mistral API during Sonnet 3.5 use**: A user reported encountering a Mistral API error while using Sonnet 3.5 on Aider chat, even though they were not using Mistral. It was suggested that fallback mechanisms may be automatically switching to Mistral.
   - For resolving this issue, users were advised to reach out to Aider's support for more specific debugging. The problem was flagged as a likely fallback to a different API when the primary request is blocked.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/deepseek/deepseek-coder">DeepSeek-Coder-V2 by deepseek</a>: DeepSeek-Coder-V2, an open-source Mixture-of-Experts (MoE) code language model. It is further pre-trained from an intermediate checkpoint of DeepSeek-V2 with additional 6 trillion tokens.  The origina...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-70b-instruct">Meta: Llama 3 70B Instruct by meta-llama</a>: Meta&#x27;s latest class of model (Llama 3) launched with a variety of sizes &amp; flavors. This 70B instruct-tuned version was optimized for high quality dialogue usecases.  It has demonstrated stron...</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5">nomic-ai/nomic-embed-text-v1.5 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct">deepseek-ai/DeepSeek-Coder-V2-Instruct · Hugging Face</a>: no description found</li><li><a href="https://gitgud.io/khanon/oai-reverse-proxy#what-is-this">khanon / oai-reverse-proxy · GitLab</a>: Reverse proxy server for various LLM APIs. Features translation between API formats, user management, anti-abuse, API key rotation, DALL-E support, and optional prompt/response logging.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 messages): 

salnegyeron: 나쁘진 않는데 Cohere의 Aya23이 나은 것 같았어요
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1256338211668496506)** (74 messages🔥🔥): 

- **Adept Strategy Shift: Co-Founders Heading to Amazon**: Adept AI Labs announced updates to their strategy and leadership, with co-founders including CEO David Luan joining Amazon, as confirmed by their [blog post](https://www.adept.ai/blog/adept-update) and a more detailed [GeekWire article](https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/). The article notes Amazon's intent to utilize Adept's technology under a non-exclusive license while Adept continues as an independent company.
   - *“That link explains what happened much better than their blog post did.”* commented a community member, highlighting that the blog post was confusing, making readers speculate about the exact nature of the relationship between Adept and Amazon.
- **AI Engineer World’s Fair Recap & Lessons**: Organizers of the AI Engineer World’s Fair (AIEWF) gathered feedback about the event, focusing on improving session lengths and overall logistics. [Discussions](https://github.com/swyxio/swyxdotio/issues/510) highlighted how the constraints from AI Engineer Summit should be adjusted to provide speakers more time for in-depth presentations.
   - One attendee suggested extending the event duration or conducting a survey to gather more structured feedback, while another emphasized the need for a dedicated space for hackathons or ad-hoc talks, referencing successful formats from other conferences.
- **Runway Gen 3 Release: High-Fidelity Text to Video**: Runway announced the general availability of their Gen-3 Alpha Text to Video feature, promoting it as a new frontier for high-fidelity, fast, and controllable video generation. They shared the exciting update via their [official account](https://x.com/runwayml/status/1807822396415467686).
   - The feature is now accessible to everyone on their [website](http://runwayml.com), allowing users to experiment with and leverage advanced video generation capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AdeptAILabs/status/1806773469155381705?t=HevOdjCZ31VyPecgHs5KoQ&s=19">Tweet from Adept (@AdeptAILabs)</a>: Today, we’re announcing some updates to our strategy and some changes to our leadership and team. More details are in our blog: https://www.adept.ai/blog/adept-update</li><li><a href="https://x.com/lmsysorg/status/1807812671238258931">Tweet from lmsys.org (@lmsysorg)</a>: Not all questions need GPT-4!  We introduce RouteLLM – a routing framework based on human preference data that directs simple queries to a cheaper model.  With data augmentation techniques, RouteLLM a...</li><li><a href="https://www.geekwire.com/2024/amazon-hires-founders-from-well-funded-enterprise-ai-startup-adept-to-boost-tech-giants-agi-team/">Amazon hires founders from well-funded enterprise AI startup Adept to boost tech giant&#8217;s &#8216;AGI&#8217; team</a>: (GeekWire File Photo / Kevin Lisota) Amazon is amping up its AI efforts by hiring executives from Adept, a San Francisco-based startup building &quot;agents&quot;</li><li><a href="https://x.com/ashvanth_s1/status/1806994830062493805)">Tweet from Ashvanth.S (@ashvanth_s1)</a>: Was going through the &#34;Language models are few shot learners&#34; paper, which introduced GPT-3.   This was back in 2020 when they faced such issues while training. would love to read more on the ...</li><li><a href="https://maven.com/parlance-labs/fine-tuning">Mastering LLMs: A Conference For Developers &amp; Data Scientists by Dan Becker and Hamel Husain on Maven</a>: An online conference for everything LLMs.</li><li><a href="https://x.com/karpathy/status/1807121265502965802?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Andrej Karpathy (@karpathy)</a>: @realSharonZhou The Fosbury flop of M&A</li><li><a href="https://x.com/runwayml/status/1807822396415467686">Tweet from Runway (@runwayml)</a>: Gen-3 Alpha Text to Video is now available to everyone.  A new frontier for high-fidelity, fast and controllable video generation.  Try it now at http://runwayml.com</li><li><a href="https://asciinema.org/a/76uAJPY1825TxnuUlZ4yhvosu">untitled</a>: Recorded by wesen3000</li><li><a href="https://asciinema.org/a/0fT2j4M8xWoJZ23A2PTjBTn4r">untitled</a>: Recorded by wesen3000</li><li><a href="https://github.com/simonw/llm-cmd">GitHub - simonw/llm-cmd: Use LLM to generate and execute commands in your shell</a>: Use LLM to generate and execute commands in your shell - simonw/llm-cmd</li><li><a href="https://lmsys.org/blog/2024-07-01-routellm/">RouteLLM: An Open-Source Framework for Cost-Effective LLM Routing | LMSYS Org</a>: &lt;p&gt;LLMs have demonstrated remarkable capabilities across a range of tasks, but there exists wide variation in their costs and capabilities, as seen from the ...</li><li><a href="https://github.com/swyxio/swyxdotio/issues/510">Organizing AI Engineer World&#39;s Fair 2024 · Issue #510 · swyxio/swyxdotio</a>: slug: aiewf-2024 We have just come off a very intense production period of the first AI Engineer World&#39;s Fair, the large, multi-track format of the AI Engineer series of conferences that my biz pa...</li><li><a href="https://github.com/gorilla-llm/gorilla-cli">GitHub - gorilla-llm/gorilla-cli: LLMs for your CLI</a>: LLMs for your CLI. Contribute to gorilla-llm/gorilla-cli development by creating an account on GitHub.</li><li><a href="https://github.com/pgibler/cmdh">GitHub - pgibler/cmdh: Create Linux commands from natural language, in the shell.</a>: Create Linux commands from natural language, in the shell. - pgibler/cmdh</li><li><a href="https://github.com/go-go-golems/geppetto">GitHub - go-go-golems/geppetto: golang GPT3 tooling</a>: golang GPT3 tooling. Contribute to go-go-golems/geppetto development by creating an account on GitHub.</li><li><a href="https://github.com/ShishirPatil/gorilla">GitHub - ShishirPatil/gorilla: Gorilla: An API store for LLMs</a>: Gorilla: An API store for LLMs. Contribute to ShishirPatil/gorilla development by creating an account on GitHub.</li><li><a href="https://x.com/iamgingertrash/status/1807798608374411630).">Tweet from simp 4 satoshi (@iamgingertrash)</a>: Deleted my tweet on cursor — it looks like the local mode thing was a hack, and they never intended for cursor to be used locally anyway  So they’re pretty clear about their data practices since it’s ...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1256325180741914694)** (1 messages): 

- **OSS GPT Store Rundown Announced**: A member announced that there will be an **OSS GPT Store rundown** by <@383468192535937026> happening in about an hour.
   - *Reminder*: If you're not busy at **AIEF** <@&1254604002000244837>, hop into <#1209303473263485011> and pick up the mentioned role for future notifications.
- **Pickup Role for Future Notifications**: A friendly reminder was given to pick up the mentioned role in <#1209303473263485011> if you want notifications about upcoming events.
   - This will help you stay updated on what's happening next in the **OSS GPT Store** and other relevant updates.
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1256338676573667530)** (34 messages🔥): 

- **GPT System Prompts Not Private**: Members discussed whether GPT system prompts are public, highlighting that although they are not *specifically public*, they can be found in repos such as [this one](https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts).
   - It was advised to assume that someone could access these prompts, thereby recommending no secrets in GPT system definitions.
- **Discussing CoALA Framework**: The latest paper on **Cognitive Architectures for Language Agents (CoALA)**, linking to [arXiv](https://arxiv.org/abs/2309.02427), sparked interest.
   - The paper organizes existing language agents into a coherent framework and suggests future actionable directions.
- **Awesome Language Agents Repo**: There was a sharing of the [awesome-language-agents](https://github.com/ysymyth/awesome-language-agents) GitHub repo.
   - This repository lists language agents based on the CoALA paper, providing a valuable resource for further exploration.
- **AI Conference Recap**: The idea of doing a recap of an **AI engineer conference** with multiple lightning talks was floated.
   - Members seemed interested in the idea for the next meeting, considering past talks had good participation and feedback.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ysymyth/awesome-language-agents">GitHub - ysymyth/awesome-language-agents: List of language agents based on paper &quot;Cognitive Architectures for Language Agents&quot;</a>: List of language agents based on paper &quot;Cognitive Architectures for Language Agents&quot; - ysymyth/awesome-language-agents</li><li><a href="https://arxiv.org/abs/2309.02427">Cognitive Architectures for Language Agents</a>: Recent efforts have augmented large language models (LLMs) with external resources (e.g., the Internet) or internal control flows (e.g., prompt chaining) for tasks requiring grounding or reasoning, le...</li><li><a href="https://github.com/LouisShark/chatgpt_system_prompt/tree/main/prompts/gpts">chatgpt_system_prompt/prompts/gpts at main · LouisShark/chatgpt_system_prompt</a>: A collection of GPT system prompts and various prompt injection/leaking knowledge. - LouisShark/chatgpt_system_prompt
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1256494640383655976)** (97 messages🔥🔥): 

- **LangGraph React and FastAPI Example**: A user asked for an example of using LangGraph with a React frontend and FastAPI backend. They were directed to *chat-langchain* on [GitHub](https://github.com/langchain-ai/chat-langchain) and the semantic routing library [Semantic Router](https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb).
   - It's suggested to either create an agent and tool to handle user inputs or use semantic similarity methods for routing decisions. These resources provide groundwork for implementing these techniques.
- **Accessing Configurable Parameters in Tools**: Queried how to access config parameters like thread_id in tools. Found accessing via specific tool schemas such as *GetThreadSchema* in [langchain-community](https://v02.api.js.langchain.com/types/langchain_community_tools_gmail.GetThreadSchema.html).
   - Further discussion on the use of `.astream_events()` to stream LangGraph state suggests iterating over events to capture the final state, ensuring state management across executions using thread IDs.
- **Streaming and Checkpointing in LangGraph**: A user asked about resuming LangGraph execution and accessing saved states using checkpoints. It's shown through code snippets that checkpoints ensure previous states are maintained and accessible using unique identifiers like thread IDs.
   - Described usage of `.astream_events()` for real-time state updates. Key events and configurations were cited to manage state transitions effectively, which is especially useful for tools requiring historical context.
- **RAG vs Semantic Similarity Example Retrieval**: Users inquired about the difference between Retrieval-Augmented Generation (RAG) and Semantic Similarity Example Retrieval. RAG combines retrieval-based with generative models for context routing, whereas similarity example retrieval selects relevant examples based on embeddings.
   - Both methods could be used together for robust applications, leveraging RAG for routing and similarity methods for specific example selections. Documentation was referenced for further understanding.
- **LangGraph State Management Techniques**: Questions were raised regarding managing and accessing LangGraph state during streaming and post-execution. State information is accessible and persistable using checkpoint systems and tools were discussed to leverage saved states.
   - Emphasis was placed on `stream` methods and `thread_id` configurations to maintain execution continuity. The provided code examples reinforced the applicability of these techniques.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.langflow.org/">Langflow - Create your AI App!</a>: LangFlow is a GUI for LangChain, designed with react-flow to provide an effortless way to experiment and prototype flows with drag-and-drop components and a chat box.</li><li><a href="https://js.langchain.com/v0.2/docs/concepts/#routing>).">Conceptual guide | 🦜️🔗 Langchain</a>: This section contains introductions to key parts of LangChain.</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#astream_events>).">Conceptual guide | 🦜️🔗 LangChain</a>: This section contains introductions to key parts of LangChain.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/migrate_agent/#in-langgraph-2>).">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://github.com/langchain-ai/langchain/issues/12304>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/streaming/#using-stream-events>).">How to stream runnables | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#astream_events>)">Conceptual guide | 🦜️🔗 LangChain</a>: This section contains introductions to key parts of LangChain.</li><li><a href="https://ai.google.dev/competition">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=i5cEuq9yuS4">Academia Braker, the second demo.</a>: This is the second demo video for the app that I create for Generative AI Agents Developer Contest by NVIDIA and LangChain. #NVIDIADevContest #LangChain @nvi...</li><li><a href="https://github.com/langchain-ai/langchain/issues/16640>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/discussions/22529,">How to access config variables in a tool ? · langchain-ai/langchain · Discussion #22529</a>: Checked other resources I added a very descriptive title to this question. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and...</li><li><a href="https://github.com/langchain-ai/chat-langchain">GitHub - langchain-ai/chat-langchain</a>: Contribute to langchain-ai/chat-langchain development by creating an account on GitHub.</li><li><a href="https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb">semantic-router/docs/03-basic-langchain-agent.ipynb at main · aurelio-labs/semantic-router</a>: Superfast AI decision making and intelligent processing of multi-modal data. - aurelio-labs/semantic-router</li><li><a href="https://api.python.langchain.com/en/stable/runnables/langchain_core.runnables.config.RunnableConfig.html#langchain_core.runnables.config.RunnableConfig>">langchain_core.runnables.config.RunnableConfig &mdash; 🦜🔗 LangChain 0.1.4</a>: no description found</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/migrate_agent/#in-langgraph-2>)">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/migrate_agent/#basic-usage>).">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts: -</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/#retrieval-tool>)">How to add chat history | 🦜️🔗 LangChain</a>: In many Q&amp;A applications we want to allow the user to have a back-and-forth conversation, meaning the application needs some sort of &quot;memory&quot; of past questions and answers, and some logi...</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/#agent-constructor>)">Conversational RAG | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/agents/#adding-in-memory>)">Build an Agent | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/message_history/#invoking-with-config>)">How to add message history | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://python.langchain.com/v0.2/docs/tutorials/chatbot/#message-history>)">Build a Chatbot | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/chatbot/#message-history>)">Build a Chatbot | 🦜️🔗 Langchain</a>: Overview
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1256370823069962333)** (7 messages): 

- **Boost Retrieval Speeds with Matryoshka Embeddings**: Prashant Dixit shared a [post](https://x.com/Prashant_Dixit0/status/1806580075447590974) about building Matryoshka RAG with **llama_index** for accelerating retrieval speeds and reducing memory footprint. Check out his [tutorial](https://colab.research.google.com/github/lancedb/vectordb-recipes/blob/main/tutorials/RAG-with_MatryoshkaEmbed-Llamaindex/RAG_with_MatryoshkaEmbedding_and_Llamaindex.ipynb) featuring **lancedb** and **huggingface**.
   - He highlights that these embedding models offer a variety of dimensions (768, 512, 256, 128, and 64), which can **boost performance** and **reduce memory usage**.
- **Automate Data Analysis with EDA-GPT**: Shaunak shared his new [GitHub project](https://github.com/shaunthecomputerscientist/EDA-GPT) for automated data analysis leveraging **LLMs**. The project is deployable on **Streamlit** or can be set up locally, with an included video tutorial in the README.
   - The application integrates **large language models (LLMs)** to streamline data analysis workflows.
- **Todoist Integration for Community Package**: mbbrainz introduced a new **Todoist integration**, created with the help of ChatGPT, and seeks to share it with the community. He suggested that [this chat](https://chatgpt.com/share/e5c10477-0db9-4941-b684-5fc0ec5556e2) might be useful for creating a comprehensive tutorial.
   - This integration serves as a potential enhancement to the community package, offering streamlined task management.
- **LangChain Meets Postgres**: Andy Singal published a [Medium article](https://medium.com/ai-advances/unleashing-the-power-of-persistence-langchain-meets-postgres-9cc7f069b260) on integrating **LangChain** with **Postgres**. He explores how persistence can be optimized by using these technologies together.
   - The article delves into leveraging **Postgres' reliability** for robust persistence in LangChain projects.
- **Automated Competitor Research Tool Guide**: Sheru shared a detailed guide on creating an **automated competitor researcher**. The step-by-step guide shows how to track any changes your competitors make on their website—check it out [here](https://dub.composio.dev/compete).
   - This tool is designed to keep you informed on competitor activities and website modifications.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/5-QV3lVI8uo">Chat With Any WebPage or Application using Visual Agents &amp; LangChain</a>: In this video, I show how you can build your own AI expert using OpenAI and LangChain RAG, with a chat interface. Then instantly access it from our browser e...</li><li><a href="https://dub.composio.dev/compete">Competitor Researcher</a>: Study your competitors&#x27; websites &amp; follow every change that they make with this project.</li><li><a href="https://x.com/Prashant_Dixit0/status/1806580075447590974">Tweet from Prashant Dixit (@Prashant_Dixit0)</a>: Build Matryoshka RAG with @llama_index  These embedding models produce a range of embedding dims(768, 512, 256, 128, and 64).  🌟 Advantages  ✅ Boosting retrieval Speed performance   ✅ Reducing memory...</li><li><a href="https://github.com/shaunthecomputerscientist/EDA-GPT">GitHub - shaunthecomputerscientist/EDA-GPT: Automated Data Analysis leveraging llms</a>: Automated Data Analysis leveraging llms. Contribute to shaunthecomputerscientist/EDA-GPT development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1256848855232938025)** (3 messages): 

- **Mixture of Agents (MoA) shone brightly**: A YouTube video titled ["Mixture of Agents (MoA) using langchain"](https://www.youtube.com/watch?v=VNy7CM23WA0) demonstrates **implementing MoA** with LangChain. The video highlights the benefits and steps required for combining multiple agents to leverage their strengths.
   - The video focuses on practical applications and *introduces MoA*, aiming to optimize tasks by harnessing collective capabilities. The presenter outlines essential code snippets and detailed guidance for better understanding.
- **Automating web search with LangChain and Google APIs**: An [Analytics Vidhya blog post](https://www.analyticsvidhya.com/blog/2024/06/automating-web-search-using-langchain-and-google-search-apis/) describes **automating web searches** using LangChain and Google Search APIs. It explains how AI and **NLP innovations** can simplify finding relevant information by summarizing search results.
   - The article elaborates on the use of **LangChain and OpenAI** to enhance search efficiency, bypassing traditional search limitations. *Users get concise, summarized responses* to their queries along with related links.
- **PR agent creation using LangChain detailed**: A [detailed guide](https://git.new/pr-agent-langchain) outlines steps for creating a PR agent using **Composio, LangChain, OpenAI, and ChatGPT**. It advises ensuring Python 3.8+ is installed and presents a **schematic diagram** of the PR agent setup.
   - The guide emphasizes integrating a **Slackbot** for PR reviews, providing illustrative images and detailed steps for implementation. From setup to execution, it covers all aspects necessary for effective PR automation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=VNy7CM23WA0">Mixture of Agents (MoA) using langchain</a>: Today we will implement mixture of agents using langchain.We introduce Mixture of Agents (MoA), an approach to harness the collective strengths of multiple L...</li><li><a href="https://git.new/pr-agent-langchain">composio/python/examples/pr_agent/pr_agent_langchain at master · ComposioHQ/composio</a>: Composio equips agents with well-crafted tools empowering them to tackle complex tasks - ComposioHQ/composio</li><li><a href="https://www.analyticsvidhya.com/blog/2024/06/automating-web-search-using-langchain-and-google-search-apis/">Automating Web Search Using LangChain and Google Search APIs</a>: Automate web research using LangChain and Google Search APIs for efficient data extraction and AI-driven query answering.
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1256371087906967595)** (40 messages🔥): 

- **Adept Strategy Shift & Co-Founders Joining Amazon**: Adept recently announced on their [blog](https://www.adept.ai/blog/adept-update) a strategic shift and details about company updates. This includes co-founders of Adept joining Amazon and licensing some of Adept's technology, as confirmed by Adept's and Amazon's internal communications.
   - The startup is left with around 20 employees after the co-founders' departure to Amazon according to a [tweet](https://x.com/anissagardizy8/status/1806812006009442671?s=46). This move mirrors Microsoft's steps with Inflection AI, sparking discussions on Adept’s future and culture changes.
- **Transformer Authors Leaving Adept**: Two of the co-founders, who were original authors of the Transformer paper, left Adept to start their own ventures. This has led to several leadership changes, including the company now being on its third CEO.
   - There are rumors of a toxic culture at Adept possibly being a reason for their departure, with some attributing it to one of the co-founders. This internal upheaval is reportedly part of why the strategy shifted significantly with Amazon's involvement.
- **Challenges in AI Agent Development**: Members compared the current state of AI agents to the early self-driving car hype, suggesting they are 'always just around the corner' but not yet reliably functional. They noted projects like Multion showing only basic web scraping capabilities.
   - There’s speculation that data collection is a bottleneck for AI agents' progress, with some believing that more extensive or synthetic data could be crucial for advancement. The focus is shifting from scaling existing data to generating quality data specifically tailored to model needs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.adept.ai/blog/adept-update">An Update to Adept</a>: Announcing some updates to our strategy and the company.</li><li><a href="https://x.com/anissagardizy8/status/1806812006009442671?s=46">Tweet from Anissa Gardizy (@anissagardizy8)</a>: Amazon has hired the cofounders of artificial intelligence startup Adept and licensed some of its tech, according to a post by the startup and an internal email from an Amazon exec  Adept is left w/ a...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1256724693654704148)** (6 messages): 

- **Cohere's Approach to Improving AI**: A YouTube video titled '[How Cohere will improve AI Reasoning this year](https://youtu.be/B45s_qWYUt8?si=qs1u6p7wiXFP46PT)' discusses how **Aidan Gomez**, CEO of **Cohere**, plans to tackle **AI hallucinations** and enhance **reasoning abilities**. The video highlights synthetic data generation techniques aimed at identifying model weaknesses and closing performance gaps.
   - A member pointed out that **Generative Active Learning** (generating the examples most crucial for labeling/ingesting) might be the correct term for these techniques, comparing it to *hard negative/positive mining* for LLMs. Observations were made around **5:30** and **15:00** timestamps in the video.
- **Generative Active Learning Mention in MIT Course**: While watching MIT Data Centric AI lectures, a concept called **Generative Active Learning** was referenced. This subfield focuses on generating the most important examples for labeling and ingestion.
   - This concept was discussed as potentially analogous to *hard negative/positive mining* for LLMs, which could help identify and mitigate model weaknesses. The term resonated with ongoing discussions about improving AI models.

**Link mentioned**: <a href="https://youtu.be/B45s_qWYUt8?si=qs1u6p7wiXFP46PT">How Cohere will improve AI Reasoning this year</a>: Aidan Gomez, CEO of Cohere, reveals how they&#39;re tackling AI hallucinations and improving reasoning abilities. He also explains why Cohere doesn&#39;t use any out...

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1257050918826938498)** (23 messages🔥): 

- **Llama 405B surprises WhatsApp users**: A [tweet](https://x.com/teknium1/status/1807490685387591983) mentioned that **Llama 405B** is apparently showing up in WhatsApp, causing surprise among users. This revelation sparked queries and excitement in the community.
   - Members expressed curiosity and intrigue about this development, wondering *when Llama* would become publicly available.
- **Llama 3 release remains uncertain**: **Nathan Lambert** voiced skepticism about the release of **Llama 3**, suggesting that there may not be a need for its release given the state of existing models. Concerns about **political pressure** were also mentioned as a reason behind the uncertainty.
   - Further discussion revealed that expectations for **Llama 3** are tempered due to **Meta's strategic interests** and the overall landscape of **ML organizations**. Lambert emphasized that the community should adjust their expectations accordingly.
- **Chatbot Arena pushes for model evaluation transparency**: **Chatbot Arena** aims to aid LLM evaluations by human preference, emphasizing **data transparency and community engagement** as outlined in a [tweet](https://x.com/lmsysorg/status/1807503885181006236). The initiative encourages studying real-world prompts and using this data to improve models.
   - The **Arena** addresses concerns about **prompt duplication** and **data distribution**, inviting community participation through Kaggle challenges and other platforms to tackle human preference in model evaluations.
- **Hope rests on Deepseek for the open-source community**: In light of Llama 3's uncertain release, the community jokingly laments and eyes **Deepseek** as a hopeful alternative for open-source models. The sentiment reflects a broader reliance on **Deepseek** to meet the community's needs.
   - A member opined that having a large model like **Deepseek** could help groups with limited compute resources to build on top of it, though **Nathan Lambert** expressed doubts about the feasibility.
- **Training large models in the open community is unrealistic**: **Nathan Lambert** argues that expecting the open community to train on 400B models is unrealistic, given the computational and data constraints. This perspective comes despite discussions of its potential utility for groups with significant compute abilities.
   - Lambert highlighted that **Meta** releasing such models would inadvertently aid many ML organizations, which might not align with strategic interests. The dialogue underscores the challenges facing the open community in staying competitive with top proprietary models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/lmsysorg/status/1807503885181006236">Tweet from lmsys.org (@lmsysorg)</a>: There seems to be some confusion. Chatbot Arena’s mission from Day 1 is to help solve LLM evals by human preference.  By opening our datasets/papers, we encourage the community to study real-world pro...</li><li><a href="https://x.com/teknium1/status/1807490685387591983?s=46">Tweet from Teknium (e/λ) (@Teknium1)</a>: Apparently llama 405b is showing up in whatsapp? 🙃
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1256561795695509558)** (5 messages): 

- **Character.ai vs Cognition AI Valuations Puzzle**: **Character.ai**, with 20 million users and founded by **Noam Shazeer**, has a $1B valuation, while **Cognition AI** with no users and a weaker product demo boasts a $2B valuation. Discussions speculate differences in **fundraising strategies** and market pitching.
   - Comments highlight that **Cognition AI** is pitching a vision based on founding members being **IMO medalists** and targeting the lucrative developer market. Concerns were raised about the robustness of their claim given strong competition from **big tech** and established AI players.
- **IMO Medals and Valuations**: Member **an1lam** humorously noted that **Noam Shazeer** also has an **IMO gold** medal. This adds to the irony given the criticisms of **Cognition AI's** valuation based solely on their founders' math accolades.
   - **Competition** from large AI firms like **OAI** and **Anthropic** was noted, with many having IMO medalists and more substantial AI achievements. Concerns were voiced about the sustainability of **Cognition AI's** business claims.
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1257110677668237352)** (7 messages): 

- **YouTube Video on AI Innovations**: A member shared a [YouTube video](https://youtu.be/cz6q-9sv-pU?si=XeckJ9U2V6XqZtzM&t=1290) titled "Live from The Lakehouse, Day 2 Segment 3: AI Regulation and Industry Innovations." The video discusses the latest trends in AI regulation and industry innovations.
   - Another video titled ["Marines commercial (sword classic)"](https://www.youtube.com/watch?v=7C0kHFVHSDs) was mentioned as a classic Marine sword commercial from the 80s. A user humorously noted the alpha nature of an unlisted video with only 8 views.
- **DBRex Legal Team Sword Incident**: Members joked about the DBRex legal team being 'sword'd up.' The context or specific details of the incident were not discussed further.
   - A user expressed admiration for Frankle, referring to him as 'so cool.' This was in the context of the previous discussions around the videos and legal team.
- **DBRX-Next Model on Chatbot Arena**: A user mentioned **DBRX-Next** model on Chatbot Arena but couldn't recall additional details about it. They were unsure if they had previously heard about it and then forgotten, indicating some community interest but lack of information.
   - Another user commented on efforts to fix their fine-tuning, possibly referring to the same or another model, indicating ongoing work and challenges in the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=7C0kHFVHSDs">Marines commercial (sword classic)</a>: The classic Marine sword commercial from the 80&#39;s.</li><li><a href="https://youtu.be/cz6q-9sv-pU?si=XeckJ9U2V6XqZtzM&t=1290">Live from The Lakehouse, Day 2 Segment 3: AI Regulation and Industry Innovations</a>: Join us for an insightful discussion on the latest trends in AI regulation and industry innovations in this segment from Day 2 of Live from The Lakehouse. Ou...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1256493245848879226)** (4 messages): 

- **Learning RL with an LM alignment focus**: A user is almost done with **David Silver's Introduction to RL** and **Pieter Abeel's Foundations of Deep RL lectures**, planning to read **Sutton & Barto** next. They asked for any unexpected tips on learning RL, especially from an LM alignment perspective.
   - Recommendations included **reading Spinning Up in Deep RL** and leveraging **code bases to solve basic tasks**. It's advised many tasks can be executed on CPU, emphasizing practical implementation as crucial, and the **HF Deep RL course** may have useful examples.
- **Spinning Up in Deep RL is useful**: A member suggested **Spinning Up in Deep RL** as a very helpful resource. Another member agreed, adding that using code bases to solve basic tasks is key.
   - It's recommended to perform tasks on **CPU** and focus on **doing practical implementations** for better learning.
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1256449307767476305)** (33 messages🔥): 

- **Ollama struggles with 01 setup**: A member shared their difficulties in making `01` work on macOS and Windows despite following setup instructions and using Ollama with the `--local` launch parameter. [A user suggested](https://www.reddit.com/r/LocalLLaMA/comments/1dl3a13/killian_showed_a_fully_local_computercontrolling/) using commands in the terminal, but the member faced issues with needing an API key.
   - Other users reported similar issues over recent months, with no widely-accepted solution found. The desktop app was mentioned as an unnecessary step by another member, but the exact fix remains elusive.
- **Long-term memory in Open Interpreter**: A member asked for advice on giving **Open Interpreter** long-term memory to avoid repetitive mistakes with everyday tasks. They noted that using the **Sonnet** model led to errors that required multiple attempts to correct.
   - Another member recommended using specific commands or pre-training methods, but further detailed solutions were not provided. A continuous struggle with the **OI memory constraints** was evident.
- **Vector search integration demo**: A member offered help integrating vector search at scale and shared a [Colab notebook](https://colab.research.google.com/github/onefact/loving-the-baseline/blob/main/nearest-neighbors.ipynb) from a recent talk given at lightning.ai. They plan to present a similar talk at the Fed soon.
   - The member is open to further collaboration and sharing their expertise on vector search, highlighting the practical applications and ongoing support for the community.
- **Recommendations for multimodal AI models**: Another member asked for open-source model recommendations to build a multimodal AI, both censored and uncensored versions. **Moondream** was suggested as a great option for visual understanding, possibly combined with a larger LLM.
   - This sparked a short discussion about the capabilities and viability of different models for specific use cases. However, no consensus on the best model was reached.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai.google.dev/gemini-api/docs/code-execution?utm_source=gais&utm_medium=email&utm_campaign=june">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Q_p82HtBqoc&t">Open Interpreter&#39;s 01 Lite - WORLD&#39;S FIRST Fully Open-Source Personal AI AGENT Device</a>: 01 Lite by Open Interpreter is a 100% open-source personal AI assistant that can control your computer. Let&#39;s review it and I&#39;ll show you how to install open...</li><li><a href="https://colab.research.google.com/github/onefact/loving-the-baseline/blob/main/nearest-neighbors.ipynb">Google Colab</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1dl3a13/killian_showed_a_fully_local_computercontrolling/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1256415466499342357)** (42 messages🔥): 

- **OpenInterpreter struggles on Windows**: A member faced issues installing **typer** on Windows for OpenInterpreter and suggested improvements on the pyproject.toml file. They eventually succeeded by skipping some dependencies and retrying the poetry install.
   - Community members expressed frustrations with **outdated documentation** and installation issues on both Windows and Ubuntu, emphasizing the need for urgent documentation updates. Mikebirdtech advised against flooding channels with repeat questions.
- **GitHub resources for Windows installation**: A link to a [GitHub pull request](https://github.com/OpenInterpreter/01/pull/203) was shared, which includes updated documentation for Windows installation. This pull request aims to compile learnings from previous installation attempts.
   - Shadowdoggie praised the compatibility of OpenInterpreter with macOS, but highlighted a desire for better **Windows support**. Other users echoed the need for clear, updated instructions and expressed concerns about the viability of their purchased **01 Light** systems.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: The open-source language model computer</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/01/pull/203">Update documentation for Windows installation by dheavy · Pull Request #203 · OpenInterpreter/01</a>: Problem Installation for Windows, with its key differences, isn&#39;t provided in the documentation. Solution Compile learnings from previous users&#39; attempt (including Zorcon&#39;s on Discord and ...</li><li><a href="https://01.openinterpreter.com/getting-started/setup">Setup - 01</a>: no description found
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1256353985959559218)** (7 messages): 

- **Jina Reranker Gets an Upgrade**: LlamaIndex users are excited about the new **Jina reranker's** release, touted as their best reranker yet. More information on this release can be found [here](https://t.co/YsYoVOIirb).
   - *One user* stated that this reranker is a game-changer for improving retrieval strategies and combining results effectively.
- **Going Beyond Naive RAG**: At the **AI World's Fair**, LlamaIndex's **Jerry Liu** discussed using agents for deeper analysis beyond naive RAG and introduced the new llama-agents framework. For more details, visit [this link](https://t.co/XWA6qnF7mn).
   - *Another user* praised the talk, highlighting its depth and practical insights on advanced agent usage.
- **Hybrid Retrieval from Scratch Tutorial**: **@kingzzm** presented a detailed tutorial on creating a custom hybrid retriever, combining different retrieval strategies through a reranker. Find the tutorial [here](https://t.co/cTxW2UwuZ0).
   - *Participants* found the tutorial comprehensive and valuable for building effective retrieval pipelines.
- **Report Generation Agent for Beginners**: **@gswithai** shared an introductory guide on building a report generation agent using a series of tools such as RAG over guideline documents. Check out the tutorial [here](https://t.co/CJW6wefxLT).
   - The guide is praised for its simplicity and utility in setting up a working ReAct agent for report generation.
- **Deploying Multi-Agent Systems to Kubernetes**: **@_nerdai_** launched a new starter kit for deploying multi-agent systems using Docker and Kubernetes. More information can be found [here](https://t.co/wfcI0wSmFG).
   - The kit is aimed at making the shift from local services to k8s deployment seamless and efficient for users.
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1256494605336318105)** (65 messages🔥🔥): 

- **Query Pipeline Troubleshooting**: A user had difficulty using unconventional input with query pipelines in **LlamaIndex**, specifically on separating arguments between modules using **`source_key`**. Another user suggested using kwargs to properly route the inputs.
   - Further discussion included integrating retrieval in the pipeline and appropriate ways to set up and manage it. **Source_key** was reiterated as correct but required careful implementation.
- **Embedding Performance Issues**: A user struggled with embedding a large CSV file due to slow performance of embedding rows (~2048 rows/50 sec). Another user suggested increasing the **`embed_batch_size`** to 500 to improve performance.
   - Despite the performance improvement, using compact LLMs like Phi3 had issues generating evaluable code. The user considered bigger LLMs such as **quantized Llama3 from Ollama** as a potential solution.
- **Handling Metadata in Queries**: A user faced issues retrieving nodes by metadata value in **LlamaIndex** as the retriever ranked irrelevant nodes higher. The solution suggested was incorporating rerankers for better accuracy.
   - Another discussion included the feasibility of filtering nodes based on metadata string containment, and the importance of metadata in enhancing retrieval and response accuracy in LlamaIndex.
- **Managing State and Scale in Agent Services**: Queries about managing state in **LlamaIndex agent services** were raised, especially for scenarios with multiple service replicas. It was discussed that **AgentService** state management is in-memory but moving towards a stateless approach using a key-value store.
   - The possibility of running multiple instances of AgentService with shared state was discussed, with a solution to implement a stateless service for horizontal scaling being under consideration. A sidecar pattern was brought up as a prospective approach.
- **Tutorials and Sub-Agent Capabilities**: Users sought out tutorials or examples for implementing **sub-agents** in **LlamaIndex**, aiming to assign predefined prompts and inputs for specific tasks. Sub-agents were discussed in terms of enhancing task-specific functionalities.
   - Additionally, a query was raised about effectively using and customizing tools like **CodeSplitter** to streamline metadata extraction and node handling tasks in a more efficient manner.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/@llamaindex?si=1hMPLukiNN8dMJzx">LlamaIndex</a>: Official YouTube Channel for LlamaIndex - the data framework for your LLM applications </li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline/">An Introduction to LlamaIndex Query Pipelines - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_memory/">Query Pipeline Chat Engine - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC/">Extracting Metadata for Better Document Indexing and Understanding - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/MetadataExtractionSEC">Extracting Metadata for Better Document Indexing and Understanding - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1257039529098088512)** (1 messages): 

- **Can LlamaIndex Use Langchain Tools?**: A user inquired whether **Langchain Tools** can be used with **LlamaIndex agents**. They highlighted that using Langchain Tools in reverse is already known.
   - The community seeks further confirmation and discussion on the integration of these tools with LlamaIndex agents.
- **Integration Queries on Langchain and LlamaIndex**: Members discussed potential ways to use **Langchain Tools** with **LlamaIndex agents**. The user mentioned knowing the reverse integration.
   - This raised queries on the compatibility and integration process between the two toolsets.
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1256551595718610995)** (54 messages🔥): 

- **Reaching llama3 70b with just 27b is wild**: Members expressed amazement at the performance of the **llama3 70b** model, achieved with just 27 billion parameters, prompting discussions on its feasibility.
   - One member remarked that they are still on **Mixtral** due to its balance between performance and accuracy, especially on consumer-grade cards, praising its multilingual capabilities and license.
- **Hermes 2 Theta vs Hermes 2 Pro**: A discussion emerged around **Hermes 2 Theta** and **Hermes 2 Pro** models, accessible on Hugging Face, with one being an experimental merge and the other a clean finetune.
   - Members debated which option is better, citing the different training datasets and functionalities, like JSON Structured Outputs in the **Pro** version.
- **Custom ORPO formatters in Axolotl**: Users discussed issues with custom ORPO formatters in **Axolotl**, such as incorrect tokenization and handling of system roles in ChatML.
   - Suggestions included using custom roles and input roles to overcome the limitations, but members expressed concerns about potential conflicts.
- **Nvidia Synthetic Model Performance**: Some members tried the **Nvidia synthetic model**, noting its slow data generation speed – taking much longer compared to models like llama 70b or GPT-4.
   - The model’s performance raised questions on the benefits of smaller, more efficient versions for practical use.
- **CAME and Adam-mini Optimizers**: The community explored new memory-efficient optimizers like **CAME** and **Adam-mini**, which claim reduced memory usage with either better or comparable performance to traditional methods.
   - Links to relevant [papers on CAME](https://arxiv.org/abs/2307.02047) and [Adam-mini](https://arxiv.org/abs/2406.16793) were shared for those interested in the technical details and potential usage in tasks like stable diffusion training.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B">NousResearch/Hermes-2-Pro-Llama-3-70B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B">NousResearch/Hermes-2-Theta-Llama-3-70B · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2307.02047">CAME: Confidence-guided Adaptive Memory Efficient Optimization</a>: Adaptive gradient methods, such as Adam and LAMB, have demonstrated excellent performance in the training of large language models. Nevertheless, the need for adaptivity requires maintaining second-mo...</li><li><a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: We propose Adam-mini, an optimizer that achieves on-par or better performance than AdamW with 45% to 50% less memory footprint. Adam-mini reduces memory by cutting down the learning rate resources in ...
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1256551947344023653)** (11 messages🔥): 

- **Training prompt formats remain consistent**: A member asked if the same prompt format was used during training. Another member confirmed that it was.
   - Despite this, there were continued discussions about other potential issues not related to prompt repetition penalties or training/test prompts.
- **Resolving FT phi-3-small loading errors**: A member struggled with loading errors for phi-3-small in both 8k and 128k versions, encountering a validation error related to warmup_steps and warmup_ratio. Another member clarified that these parameters are mutually exclusive and only one should be used.
   - Upon revisiting their configuration, the member acknowledged the mistake and found the hidden ratio parameter, thanks to the helpful feedback.
- **Tiktoken import error plagues offline setup**: A member working on an offline machine encountered an error with tiktoken import: *requests.exceptions.ConnectionError* due to network unavailability. The error was caused by a failed connection attempt to [openaipublic.blob.core.windows.net](https://openaipublic.blob.core.windows.net).
   - *ConnectionError* occurred from attempting to access the tiktoken encoding resources without an available network connection.
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1256323358560096337)** (9 messages🔥): 

- **Leaderboard Benchmarks Bottlenecked by Computing Resources**: Members discussed the lengthy queue times for the current leaderboard benchmarks, feeling like the queue has been backed up for 2 days. **Stellaathena** mentioned that the queue on HF is entirely outside their control and is likely bottlenecked by computing resources.
   - "The queue on HF is entirely outside of our control, but presumably bottlenecked by computing resources," one member highlighted. This suggests a persistent issue with HF's ability to manage benchmark requests promptly.
- **Im-Eval's Compatibility with HumanEval and HumanEvalPlus**: **Johnl5945** asked if im-eval can perform `HumanEval` and `HumanEvalPlus`, along with querying how to set the evaluation temperature. This inquiry highlights potential usage scenarios for im-eval.
   - Members did not provide a direct answer, leaving the query about im-eval's capabilities and configuration for temperature settings open-ended, suggesting further exploration is needed.
- **HF Queue Issues & Alternative Solutions like vllm**: Members noted issues with HF's queue delays and recommended exploring alternatives, such as **vllm**. A **helpful wiki** was shared: [GitHub Wiki](https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm) for guidance on saving models.
   - "Oh yes that's a known issue with HF - it happens. Have you tried vllm instead?" a helpful member suggested, linking to a [GitHub Wiki](https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm) for further guidance.

**Link mentioned**: <a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1256325117173170377)** (46 messages🔥): 

- **$
abla$-IRIS Agent Sets New Benchmarks**: Researchers propose [$\Delta$-IRIS](https://arxiv.org/abs/2406.19320), a new RL agent with discrete autoencoder and autoregressive transformer, setting new state-of-the-art in the Crafter benchmark. The agent is substantially faster in training compared to previous approaches.
   - Community members appreciated the performance boost, noting that this agent can encode stochastic deltas between timesteps with continual tokens efficiently.
- **RoPE-before-QK Positioning Problem Solved**: The issue with rope-before-qk where qk mixes information across rope channels disrupting positional embeddings has been addressed by tying the linear transform per head channel. This method preserves the **relative interaction of RoPE** effectively.
   - The discussion also highlighted that phase offsets could be learned and data-dependent for further improvements, which could enhance positional embeddings in transformers.
- **Adam-mini Optimizer Cuts Memory Usage by Half**: The [Adam-mini optimizer](http://arxiv.org/abs/2406.16793) achieves comparable or better performance than AdamW with 45% to 50% less memory usage by partitioning parameters and assigning single learning rates to blocks. This reduction is achieved without significantly sacrificing model performance.
   - Community discussion acknowledged that partitioning and block-wise optimizations made it a competitive alternative to current popular optimizers without the excessive memory overhead.
- **Flora Challenges Low-Rank Adapter Limitations**: Flora, as introduced in this [paper](https://arxiv.org/abs/2402.03293), proposes an approach to achieve high-rank updates through random projections, reducing optimization states. It maintains the benefit of LoRA's reduced memory usage but improves on its performance limitations.
   - Community members discussed its theoretical merit, suggesting it might face similar limitations as LoRA but with potential improvements from the dynamic resampling of projections.
- **Erasure Effect in Token Representations Discovered**: A new study on token representations in LLMs reveals an erasure effect where information about previous tokens is diminished or 'erased' in representation with named entities and multi-token words. The token erasure effect is strong in the last token representations of multi-token concepts.
   - The [study](https://arxiv.org/abs/2406.20086) was appreciated for shedding light on how LLMs handle semantically unrelated token groups and provided insights into improving model understanding of complex token sequences.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>: We propose Adam-mini, an optimizer that achieves on-par or better performance than AdamW with 45% to 50% less memory footprint. Adam-mini reduces memory by cutting down the learning rate resources in ...</li><li><a href="https://arxiv.org/abs/2406.19320">Efficient World Models with Context-Aware Tokenization</a>: Scaling up deep Reinforcement Learning (RL) methods presents a significant challenge. Following developments in generative modelling, model-based RL positions itself as a strong contender. Recent adva...</li><li><a href="https://arxiv.org/abs/2406.20086">Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs</a>: LLMs process text as sequences of tokens that roughly correspond to words, where less common words are represented by multiple tokens. However, individual tokens are often semantically unrelated to th...</li><li><a href="https://arxiv.org/abs/2402.03293">Flora: Low-Rank Adapters Are Secretly Gradient Compressors</a>: Despite large neural networks demonstrating remarkable abilities to complete different tasks, they require excessive memory usage to store the optimization states for training. To alleviate this, the ...</li><li><a href="https://arxiv.org/abs/2404.14507">Align Your Steps: Optimizing Sampling Schedules in Diffusion Models</a>: Diffusion models (DMs) have established themselves as the state-of-the-art generative modeling approach in the visual domain and beyond. A crucial drawback of DMs is their slow sampling speed, relying...</li><li><a href="https://github.com/BorealisAI/flora-opt">GitHub - BorealisAI/flora-opt: This is the official repository for the paper &quot;Flora: Low-Rank Adapters Are Secretly Gradient Compressors&quot; in ICML 2024.</a>: This is the official repository for the paper &quot;Flora: Low-Rank Adapters Are Secretly Gradient Compressors&quot; in ICML 2024. - BorealisAI/flora-opt</li><li><a href="https://arxiv.org/abs/2311.00537">Machine Learning Without a Processor: Emergent Learning in a Nonlinear Electronic Metamaterial</a>: Standard deep learning algorithms require differentiating large nonlinear networks, a process that is slow and power-hungry. Electronic learning metamaterials offer potentially fast, efficient, and fa...</li><li><a href="https://colab.research.google.com/drive/13WO33fQzhnSV4daghFXUmVCwVJ0OV5MI?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1256326613503578212)** (7 messages): 

- **Custom Config YAML Errors**: Formatting Concerns**: **New user**: A member issues a query about using a custom config yaml to run an eval, facing errors despite creating a new yaml with parameters like `dataset_name`, `description`, etc. Including the parameter `--verbosity DEBUG` helps to identify where the error lies, and noting that naming conventions must be unique helps resolve the issue.
   - Adding appropriate flags reveals that there is a `task named 'mmlu_clinical_knowledge'` conflict due to existing registered tasks with the same name. Ultimately, running the appropriate debugging commands successfully resolves the problem.
- **Gemma 2 Metrics Troubleshooting**: Discrepancies Noted**: **Reproducibility issues**: A user reports significant discrepancies in replicating the metrics for **Gemma 2** using `lm_eval`, despite following setup instructions such as setting `dtype` to `bfloat16` and using the `4.42.3` version of the transformers library. The observed metric discrepancies include a 0.5990 accuracy on **piqa** compared to the model card's 0.817, along with similar differences for **hellaswag** and **winogrande**.
   - The command used (`lm_eval --model hf --model_args pretrained=google/gemma-2-9b,parallelize=True,trust_remote_code=True,dtype=bfloat16 --tasks piqa --batch_size auto:4 --log_samples --output_path output/gemma-2-9b`) seems correctly formed, prompting a need for further investigation into potential unseen issues.

**Link mentioned**: <a href="https://x.com/LysandreJik/status/1807779464849273343)">Tweet from Lysandre (@LysandreJik)</a>: Last week, Gemma 2 was released. Since then, implems have been tuned to reflect the model performance:  pip install -U transformers==4.42.3  We saw reports of tools (transformers, llama.cpp) not being...

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

arctodus_: Thanks! This is what I was looking for. Will take a look.
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1256362059277533256)** (27 messages🔥): 

- **Move RMSNorm to nn/init.py**: George Hotz asked if someone could *tastefully move RMSNorm* from **LLaMA** to `nn/__init__.py` and add documentation and tests.
   - This would help standardize and improve the **tinygrad** codebase organization.
- **Tinygrad Monday Meeting Topics**: The Monday meeting covered multiple topics such as the new tinybox update, [faster sharded llama](https://github.com/tinygrad/tinygrad/pull/5123), and [clean graph_rewrite](https://github.com/tinygrad/tinygrad/pull/5159/files#diff-91ca5e2e75ef3ea1982c8ca6cc175ee88f20efa0d8e4b96f305b970dc6df71e7R291).
   - Discussion included **lowerer continuation**, **tensor cores**, and various bounties like standard mean one kernel and **qualcomm runtime**.
- **Compile Tinygrad Programs for Raspberry Pi**: A user inquired if **tinygrad** programs could be compiled into standalone C programs for devices like Raspberry Pi.
   - Another member shared [this link](https://github.com/wozeparrot/tinygrad-on-esp32) for **tinygrad on ESP32**, indicating potential interest and use cases.
- **Llama 70B Lora Bounty Requirements**: There was a discussion about the **llama 70b lora bounty** requirements and whether **qlora** would be accepted.
   - **ChenYuy** explained it must follow the [MLPerf reference](https://github.com/mlcommons/training/tree/master/llama2_70b_lora), but it can be done on a different machine with offloading or multiple boxes.
- **Graph Rewrite Followup Discussion**: One member missed the discussion and asked about the **graph rewrite followup**, specifically Egraphs/muGraphs.
   - **ChenYuy** paraphrased that there is interest in moving more algorithms like the scheduler into graph rewrite, although a specific graph algorithm isn’t ready to be prioritized.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/5159/files#diff-91ca5e2e75ef3ea1982c8ca6cc175ee88f20efa0d8e4b96f305b970dc6df71e7R291),">single pass rewrite by geohot · Pull Request #5159 · tinygrad/tinygrad</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/issues/5244)">Issues · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Issues · tinygrad/tinygrad
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1256418416206545009)** (34 messages🔥): 

- **Challenges in tinygrad computation and gradient updates**: A member encountered issues with defining intermediate tensors in tinygrad, where **C** gets cut off from the computation graph if an intermediate tensor (`emb = C[X]`) is defined first. They found moving `emb = C[X]` down, right before logits, maintains the link between `C` and loss, making the gradient computation work.
   - Another issue arose when iterating over `parameters` for update steps, with the original tensor reverting back on the next epoch. The solution involved using `p.assign(p.realize().detach() - lr * p.grad)` to ensure tensors are correctly updated.
- **Text2video feature interest in tinygrad**: A member inquired about adding a text2video feature like **SORA** to tinygrad, seeking known missing parts that people could contribute to. There was no direct response indicating a plan for this addition.
   - This conversation showed there is community interest in expanding tinygrad's capabilities and potential contributions from users.
- **Tinygrad running across multiple machines**: George Hotz shared an interest in running tinygrad across 4 mac minis at **AGIHouse Hackathon**, linking to [wozeparrot/tinygrad](https://github.com/wozeparrot/tinygrad/tree/ops_remote).
   - Further discussion included feasibility queries about running **nemotron-340b inference** on several **mac studio M2 ultra** units.
- **TinyJit decorator causing training anomalies**: A member reported a problem where adding the `TinyJit` decorator over `train_step` caused loss values to plummet to zero rapidly. **Chenyuy** suggested the issue might be due to reusing the same mini-batch, pointing to the need for varying training examples during steps.
   - Others engaged in technical discussions about the JIT's behavior concerning gradient memory resetting and zeroing out gradients.
- **Debugging tinygrad issues and PR suggestions**: __gip__ faced an issue with `DEBUG >= 6`, mentioning missing `applegpu` tools during tinygrad's debug output. They made a [PR](https://github.com/tinygrad/tinygrad/pull/5236) to improve the error message when these tools are missing.
   - __gip__ also encountered another issue with Apple GPU disassembly while debugging, reporting it on the [dougallj/applegpu GitHub](https://github.com/dougallj/applegpu/issues/61).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/wozeparrot/tinygrad/tree/ops_remote">GitHub - wozeparrot/tinygrad at ops_remote</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - GitHub - wozeparrot/tinygrad at ops_remote</li><li><a href="https://github.com/dougallj/applegpu.git">GitHub - dougallj/applegpu: Apple G13 GPU architecture docs and tools</a>: Apple G13 GPU architecture docs and tools. Contribute to dougallj/applegpu development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/blob/b2ea610df830f8b2b25dd06ce67c2304b0f9d08a/examples/llm.c/train_gpt2.py#L168">tinygrad/examples/llm.c/train_gpt2.py at b2ea610df830f8b2b25dd06ce67c2304b0f9d08a · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="http://www.catb.org/~esr/faqs/smart-questions.html">How To Ask Questions The Smart Way</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5236">fix: message when applegpu tools missing by gip · Pull Request #5236 · tinygrad/tinygrad</a>: Show error message when applegpu tools is missing. Happy to add a section in the docs on how to install applegpu tools in another PR. Although if we do that it would make sense to also add how to i...</li><li><a href="https://github.com/dougallj/applegpu/issues/61">AssertionError during  disassembly · Issue #61 · dougallj/applegpu</a>: While using tinygrad debug mode I ran into this crash during apple gpu disassembly. To repro: python3 compiler_explorer.py tmp_p36do0r triggers an assertion error at https://github.com/dougallj/app...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1256324486643581110)** (25 messages🔥): 

- **Skepticism Over 27B Model Quality**: **27B model** effectiveness questioned but seen as better even in worst cases compared to **Command R+**. There's skepticism about its actual capability despite confidence intervals.
   - Mention of **Gemma 2 9B** performing better adds to the debate but overall sentiment leans towards 27B's potential. *“True but even in the worst case(-15) it's better than command r+”*.
- **ChatGPT 4 and 4o Models Struggle**: Reports suggest **ChatGPT 4 and 4o models** are deteriorating in performance, notably in programming tasks. Users note **ChatGPT 3.5** handles prompts better without taking them too literally.
   - *“Sometimes the paid 4 and 4o models feel absolutely useless when programming”*, highlighting preference for **free alternatives** due to declining efficiency in paid versions.
- **Gemini 1.5 Pro Outshines ChatGPT**: **Gemini 1.5 Pro** praised for excelling compared to **ChatGPT** which is seen as increasingly lazy in following prompts. Complaints about GPT4's underperformance in programming are rising.
   - *“Gemini 1.5 pro does a super excellent job compared to chatgpt's increasing laziness”*. Users are gravitating towards more responsive models like Gemini.
- **Claude’s Artifacts Feature Wins Users**: Users express preference for **Claude's artifacts feature**, finding it a better experience. Some consider switching from **ChatGPT** altogether due to this feature.
   - *“The artifacts feature is a much better experience”* summarizes the sentiment driving users towards alternatives like Claude.
- **Non-English Speakers Seek Multilingual LLMs**: Discussion suggests **non-English speakers** are turning to the LLM arena for models that excel in their native languages. This drives popularity despite models' effectiveness in specific tasks.
   - *“It's already up there not because it can answer difficult problems, but due its multilingual capabilities”*. The usage trends show a preference for language-specific conversational abilities.
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1256526950604738681)** (31 messages🔥): 

- **Block-wise Adam Optimizer’s Efficiency**: A discussion highlighted that block-wise Adam optimizer saves **90% VRAM** compared to per-parameter vs, which is **45% reduction overall**. It's claimed to be superior not just in VRAM savings but also in performance, especially in stepwise loss curves.
   - Multiple members expressed their surprise at the optimizer’s performance and savings. One user cited that the optimizer doesn't need to keep three sets of parameters in memory, pondering its compatibility with **8bit optimizations**.
- **Persona-Driven Data Synthesis for Diverse Scenarios**: A novel [persona-driven data synthesis](https://arxiv.org/html/2406.20094v1) methodology for large language models was discussed. This methodology can tap into **1 billion diverse personas** from web data to create synthetic data, potentially driving a paradigm shift in data creation.
   - The discussion emphasized the methodology's versatility and scalability in creating diverse synthetic data, including **mathematical problems** and **game NPCs**. This approach leverages various perspectives encapsulated within the model to enhance synthetic data generation significantly.
- **Full-Finetuned Model Evaluation Challenges**: A blog post on [full-finetuned model evaluation](https://mlops.systems/posts/2024-07-01-full-finetuned-model-evaluation.html) was shared, underscoring the complexities and challenges of evaluating structured data extraction from finetuned models. **Accuracy** is highlighted as the core metric of interest.
   - The poster experienced significant implementation difficulties and performance issues without a dedicated system, noting the mounting complexity of managing these evaluations. This reflects broader challenges in maintaining and scaling finetuned models effectively.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.16008">Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization</a>: Large language models (LLMs), even when specifically trained to process long input contexts, struggle to capture relevant information located in the middle of their input. This phenomenon has been kno...</li><li><a href="https://arxiv.org/html/2406.20094v1">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>: no description found</li><li><a href="https://github.com/zyushun/Adam-mini/blob/main/Adam_mini.py">Adam-mini/Adam_mini.py at main · zyushun/Adam-mini</a>: Code for Adam-mini. Contribute to zyushun/Adam-mini development by creating an account on GitHub.</li><li><a href="https://mlops.systems/posts/2024-07-01-full-finetuned-model-evaluation.html">Alex Strick van Linschoten - My finetuned models beat OpenAI’s GPT-4</a>: Finetunes of Mistral, Llama3 and Solar LLMs are more accurate for my test data than OpenAI's models.
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1256332407850668032)** (16 messages🔥): 

- **Simple JSONL Editing Tools**: A user is seeking a simple and quick tool for editing JSONL files. [An example provided](https://github.com/aastroza/simple-jsonl-editor) includes a simple GUI for this purpose.
   - Another user shared their personal scripting approach for JSONL edits, indicating they maintain a folder of Python scripts for editing JSONL files.
- **Summarizing Structured Data**: A user is exploring methods to generate summaries from structured patient data in JSON format. They mention testing llama models and are concerned about hallucinations, considering both human and LLM evaluation methods.
   - [Recommendations included](https://github.com/aastroza/simple-jsonl-editor) were prompt engineering and few-shot learning, with a suggestion to initially troubleshoot the baseline approach rather than implementing new packages.
- **Kv-Caching for LLM Inference**: A user shared a blog post on implementing kv-caching to enhance LLM inference, specifically for vision-LLMs like phi-3-vision. [This technique](https://sachinruk.github.io/blog/2024-06-29-kv-cache.html) improves prediction probabilities by storing common prompts and using vocabulary logits.
   - The blog is appreciated by another user for its guidance on optimizing vision-based models, highlighting practical steps for those without ample GPU resources.
- **Resources for Audio Chat Products**: Discussion on developing audio chat products suggested using a combination of STT (whisper), LLM processing, and TTS models. The potential impact of GPT-4o voice on this pipeline was mentioned.
   - Users recommended specific tools like SileroVAD for voice detection and [the Chainlit cookbook for rapid testing](https://github.com/Chainlit/cookbook/tree/main/audio-assistant) of audio assistant prototypes.
- **Knowledge Graphs & Lang Graph Expertise**: A seasoned Python AI full stack developer shared their extensive experience with AI-driven software, including deep work on e-commerce, image processing, and climate analysis projects. They highlighted significant improvements in accuracy rates using advanced RAG and tailored LLMs.
   - Further inquiries were made about the developer's work with knowledge graphs and lang graphs, suggesting a demand for deeper understanding and practical examples.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/515379">AI Mathematical Olympiad - Progress Prize 1 | Kaggle</a>: no description found</li><li><a href="https://sachinruk.github.io/blog/2024-06-29-kv-cache.html">Prompt Caching: Poor man’s guide to zero shot vision-LLM classification – deepschool.ai</a>: Using KV caching and logit ratios to speed up and control LLM/ VLM outputs.</li><li><a href="https://github.com/Chainlit/cookbook/tree/main/audio-assistant">cookbook/audio-assistant at main · Chainlit/cookbook</a>: Chainlit&#39;s cookbook repo. Contribute to Chainlit/cookbook development by creating an account on GitHub.</li><li><a href="https://github.com/aastroza/simple-jsonl-editor">GitHub - aastroza/simple-jsonl-editor: Simple GUI for editing .jsonl files</a>: Simple GUI for editing .jsonl files. Contribute to aastroza/simple-jsonl-editor development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1256961192065499136)** (4 messages): 

- **Kubernetes faces skepticism for ML inference**: A member noticed a tweet mocking the setup of inference on **Kubernetes (k8s)** for ML, sparking curiosity about the reasons behind this stance. They asked for insights on the topic, highlighting the importance of discussions from experienced practitioners in managing high-traffic production systems.
   - In response, the tweet's author clarified that the comment was *mostly in good fun* and expressed confidence in **Modal** as a superior cloud infrastructure for ML inference.
- **Struggles with axolotl on distributed Modal setup**: A member shared their challenges in getting **axolotl** to run on **Modal**'s distributed setup, pointing to a cloned repository on [GitHub](https://github.com/modal-labs/llm-finetuning). They noted success with a single **A100** GPU but encountered errors when attempting to scale up.
   - The errors reported include a *Distributed Training Error* related to **NCCL** communication and *Socket Timeout*, indicating potential network connectivity or setup issues. Additionally, multiple ranks failed to communicate, resulting in process failures.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/charles_irl/status/1807265742506639739">Tweet from Charles 🎉 Frye (@charles_irl)</a>: mfw someone says they&#39;re setting up their inference on k8s</li><li><a href="https://github.com/modal-labs/llm-finetuning">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1256638752680575110)** (7 messages): 

- **Credits good for 2 years on Hugging Face**: A member asked if the credits would be applicable for any usage on Hugging Face, and another confirmed they are. They clarified that the credits are **valid for 2 years**, alleviating concerns about a 2-day expiration.
   - Multiple members were relieved to learn the credits last for 2 years, with one expressing gratitude and another mentioning they also had many credits left. Another member sought additional information on their credit status with no response, highlighting communication issues.
- **Members manage unused credits**: Members discussed using LLM and embedding inferences with their remaining credits, worried about an impending expiration. Clarification about the 2-year validity of the credits eased their concerns.
   - There were additional requests for information about credit status that went unanswered. This points to a need for better communication regarding credit management.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1256662452519567360)** (1 messages): 

- **Get Mistral 7B Instruct v3 on Replicate**: [A discussion](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) began on how to make **Mistral 7B Instruct v3** available on **Replicate**, with a link to the model card for **Mistral-7B-Instruct-v0.3**. The instructions recommended using `mistral_common` for tokenization.
   - Currently, only versions **v1** and **v2** are present on [Replicate](https://replicate.com/mistralai). A user asked if they could publish **v3** from their personal account as a 'community' contribution.
- **Differences in Mistral-7B Instruct Versions Explained**: The **Mistral-7B-Instruct-v0.3** Large Language Model has notable updates compared to [v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/edit/main/README.md), including an extended vocabulary, supporting v3 tokenizer, and function calling. The user inquiries clarified how these differences impact availability on platforms like Replicate.
   - The project documentation recommends using **mistralai/Mistral-7B-Instruct-v0.3** with [mistral-inference](https://github.com/mistralai/mistral-inference) for the best results. Further details about installation via `pip install mistral_inference` were provided.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">mistralai/Mistral-7B-Instruct-v0.3 · Hugging Face</a>: no description found</li><li><a href="https://replicate.com/mistralai">mistralai – Replicate</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[ankurgoyal_textsql_llmevals](https://discord.com/channels/1238365980128706560/1242222674835538012/1256404374700425216)** (9 messages🔥): 

- **Braintrust Tools for Measuring Outputs**: A user inquired about tools for measuring the output of tool calls and complex agentic tool calling workflows. They were directed to the Braintrust [cookbook recipes](https://www.braintrust.dev/docs/cookbook/recipes/Assertions) for details on these tools.
   - The user thanked for the information, mentioning the ease of debugging and integrating data, tasks, and scoring bits in Braintrust via `Eval()`. They expressed their enjoyment of the tool's UX and helpfulness.
- **Real-time Observability via Logs Tab**: The user asked about real-time observability and whether the 'Logs' tab is used for this purpose, pointing to a need for documentation or cookbook references. They were referred to the [general docs for logs](https://www.braintrust.dev/docs/guides/logging) and a specific [cookbook for using logs and evals together](https://www.braintrust.dev/docs/cookbook/recipes/HTMLGenerator).
   - It was clarified that the tracing set up for evaluations will work for logs as well, and the Logs tab in the Braintrust UI would automatically update in real-time. The documentation also covers logging interactions, debugging customer issues, and capturing user feedback.
- **Human Review Feature**: There was a brief inquiry about the Braintrust 'human review' feature and whether it applies to logs, experiments, or both. It was confirmed that the 'Trace' datastructure is identical across experiments and logs, making the human review feature applicable everywhere.
   - A new feature for human review in datasets was also mentioned, enabling the integration of human feedback from diverse sources. More details were provided in the Braintrust [guide for human review](https://www.braintrust.dev/docs/guides/human-review#writing-categorical-scores-to-expected-field).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.braintrust.dev/docs/guides/human-review#writing-categorical-scores-to-expected-field">Human review</a>: no description found</li><li><a href="https://www.braintrust.dev/docs/cookbook/recipes/Assertions">How Zapier uses assertions to evaluate tool usage in chatbots</a>: no description found</li><li><a href="https://www.braintrust.dev/docs/guides/logging">Logging</a>: no description found</li><li><a href="https://www.braintrust.dev/docs/cookbook/recipes/HTMLGenerator">Generating beautiful HTML components</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[paige_when_finetune](https://discord.com/channels/1238365980128706560/1242224662142779530/1257279259047886868)** (2 messages): 

- **Recording for this session still unavailable**: A member noted that the **recording for the session** is still unavailable. **Dan Becker** responded promptly confirming the recording should now be available and requested to be informed if there are any issues.
   - Dan Becker addressed the member’s concern about the **session recording's availability** in real-time and promised to rectify any potential problems swiftly.
- **Session recording update**: Dan Becker mentioned that the **recording for the session** should now be up. He asked to be informed if there were any further issues.
   - **Dan Becker** promptly addressed the session recording concern, ensuring it was resolved and requesting feedback on any remaining problems.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1257396427227660379)** (1 messages): 

- **Fine-tuning AutoModelForSequenceClassification with axolotl**: A member inquired about the possibility of fine-tuning using **AutoModelForSequenceClassification** with **axolotl**. They noted issues with documentation clarity in this area.
   - The member sought confirmation on whether fine-tuning with this method is feasible or if it merely suffers from poor documentation.
- **Issue with documentation clarity**: The lack of well-documented procedures for fine-tuning using **AutoModelForSequenceClassification** with **axolotl** was highlighted. Improving documentation for such use cases might aid users significantly.
   - Clearer guidelines would help users determine the feasibility and workflow for fine-tuning with axolotl.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1257083590798217377)** (1 messages): 

- **Less Intense Accelerate Talk: Hugging Face Accelerate**: A presentation on **Hugging Face Accelerate** titled *'Making Device-Agnostic ML Training and Inference Easy at Scale'* was shared by Zachary Mueller at a recent conference. The [YouTube video](https://www.youtube.com/watch?v=IBBeLNgGIIo) provides an overview of the tool's features and applications.
   - The talk covered how **Hugging Face Accelerate** simplifies **device-agnostic ML training and inference**, focusing on ease of use at scale. For those wanting a comprehensive but less intense overview, this presentation is highly recommended.
- **Hugging Face Accelerate: A Deep Dive**: Zachary Mueller's recent conference talk introduces **Hugging Face Accelerate** for making ML training and inference device-agnostic. The [YouTube video](https://www.youtube.com/watch?v=IBBeLNgGIIo) delves into the technical aspects and scalability of the tool.
   - The presentation highlights the ease of implementing **Hugging Face Accelerate** in various device environments, making it accessible for large-scale ML applications. It's a must-watch for those interested in device-agnostic solutions.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=IBBeLNgGIIo">Hugging Face Accelerate: Making Device-Agnostic ML Training and Inference Easy... - Zachary Mueller</a>: Hugging Face Accelerate: Making Device-Agnostic ML Training and Inference Easy at Scale - Zachary Mueller, Hugging FaceHugging Face Accelerate is an open-sou...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/)** (1 messages): 

1dingyao: Hi <@466291653154439169>,

`adingyao-41fa41` is for your assistance please.

Many thanks!
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/)** (1 messages): 

saharn_34789: Anyone in NY? if meet up in Boston, I will manage.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1256875456142118942)** (4 messages): 

- **Zed IDE impresses Sublime Text users**: Someone recently tried [Zed IDE](https://zed.dev) and found it very impressive, particularly for those who grew up with Sublime Text. They described the experience as *wooow*.
   - Curious to try other options, they mentioned that while Zed has nice AI integration, it isn't as advanced as Cursor. They expressed interest to hear if anyone else has tried Zed.
- **Curiosity about Cursor**: The user expressed curiosity about trying **Cursor** after their positive experience with Zed. They believe that Cursor might offer a wilder AI integration compared to Zed.
   - Given their interest in comparing different IDEs, they are keen to see if others in the community have tried **Cursor**. They are looking for feedback and opinions on its AI capabilities.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1256959592483782768)** (1 messages): 

- **Mistral-7b outshines competitors**: The best-performing model for a user's specific use case was **Mistral-7b-optimised**, finetuned on [OpenPipe](https://mlops.systems/posts/2024-07-01-full-finetuned-model-evaluation.html). This model outperformed several others in the evaluations.
   - Replication of these results is possible with the **normal open-weights Mistral** on other providers or local machines. The process was noted to be very easy when using **OpenPipe**.
- **Easy finetuning with OpenPipe**: A user highlighted that the finetuning process is very straightforward with **OpenPipe**. This made replicability of results with the **normal open-weights Mistral** effortless.
   - The ease of use was specifically emphasized. This contributes to **OpenPipe's** usability for finetuning models like **Mistral-7b**.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[bergum_rag](https://discord.com/channels/1238365980128706560/1252713659243827251/1256658846617964625)** (5 messages): 

- **Reciprocal rank fusion solid for hybrid search**: One member noted that **reciprocal rank fusion** is a strong starting point for combining **BM25** and **vector search results**, despite both having their own issues.
   - Another member acknowledged this, stating *'So reciprocal rank fusion is good to start with for hybrid search. Thanks!'*
- **Slide deck search frustrations**: A member asked about the location of the slide deck mentioned in a video, wondering if it had been shared.
   - It was clarified that **decks are usually on Maven**, but the concerned deck was not found there.
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1256415982763901048)** (32 messages🔥): 

- **Cohere Internship Insights**: A member shared their interest in Cohere's internship positions, citing their current work on a workshop paper for the AAAI conference and previous co-op experience. They inquired about specific challenges and projects, as well as opportunities to develop new features or improve Cohere's API for LLMs.
   - Discussion touched on Cohere's support and resources for research projects integrating LLMs with other AI frameworks such as reinforcement learning.
- **Coral API Rate Limits Frustrate Users**: Several users expressed frustration over the rate limits on Coral trial keys, which are capped at 5 calls per minute. An explanatory [link](https://docs.cohere.com/docs/rate-limits) was shared, detailing the differences between trial and production keys, the latter offering 10,000 calls per minute.
   - "Trial keys are made for trying out services," one user emphasized, recommending the upgrade to production keys for higher throughput.
- **Aya-23 9B Model Queries**: Members discussed the release of different versions of Aya-23 models, specifically the availability of the 8B and 35B versions on [Hugging Face](https://huggingface.co/CohereForAI/aya-23-8B).
   - There was confusion about the existence of an Aya-23 9B model and its relevance, with clarification that current models are considered adequate and are not run for inference.
- **AI Reasoning Enhancements by Cohere**: A YouTube video titled ["How Cohere will improve AI Reasoning this year"](https://youtu.be/B45s_qWYUt8?si=_c7sQUFUN6bZa61m) was shared, featuring CEO Aidan Gomez discussing efforts to tackle AI hallucinations and enhance reasoning abilities.
   - The video explains why Cohere avoids using any external models and focuses on internal development.
- **Support for Academic Research**: In response to a query about development tools for academic research, a [GitHub repository](https://github.com/cohere-ai/cohere-toolkit) was shared. It contains prebuilt components for building and deploying RAG applications.
   - This toolkit can serve as a base for academic projects, ensuring streamlined development and deployment.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/B45s_qWYUt8?si=_c7sQUFUN6bZa61m">How Cohere will improve AI Reasoning this year</a>: Aidan Gomez, CEO of Cohere, reveals how they&#39;re tackling AI hallucinations and improving reasoning abilities. He also explains why Cohere doesn&#39;t use any out...</li><li><a href="https://docs.cohere.com/docs/rate-limits">API Keys and Rate Limits</a>: no description found</li><li><a href="https://github.com/cohere-ai/cohere-toolkit">GitHub - cohere-ai/cohere-toolkit: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications.</a>: Cohere Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications. - cohere-ai/cohere-toolkit</li><li><a href="https://huggingface.co/CohereForAI/aya-23-8B">CohereForAI/aya-23-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/CohereForAI/aya-23-35B">CohereForAI/aya-23-35B · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1257183563833409596)** (3 messages): 

- **Rig Fully Integrates with Cohere Models**: The **Rig** library, a Rust library designed for building LLM-powered applications, now comes fully integrated with all **Cohere models**. Check out the [announcement link](https://discord.com/channels/954421988141711382/1218409701339828245/1250476945821532195) in the community for more info.
   - A call for feedback and an incentivized program offering a **$100 reward** for Rust developers has been launched to gather user experiences and suggestions for improving Rig. For detailed steps to participate, visit the [feedback form link](https://bit.ly/Rig-Feeback-Form).
- **Community Notebook for Mapping Citations**: A community member shared a **notebook** for mapping citations and documents back to responses in applications. The notebook link is available on [Google Colab](https://colab.research.google.com/drive/1o89bvd_JGRijQSFwiyXK-61oTD6okf_0?usp=sharing).
   - The notebook allows storing markdown responses with cited text highlighted and sources presented. It also offers customizable parameters to adjust the look and feel, with further enhancements welcomed from the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bit.ly/Rig-Feeback-Form">Rig Feedback and Rewards Program</a>: Thank you for your interest in Rig, an open-source library for building LLM-powered applications using Rust! As Rig is a brand new library, we&#39;re interested in gathering feedback from Rust develop...</li><li><a href="https://colab.research.google.com/drive/1o89bvd_JGRijQSFwiyXK-61oTD6okf_0?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1256387194248237139)** (22 messages🔥): 

- **Jump into DPO Training**: A user questioned if DPO training should follow after regular training, with some members suggesting jumping right into DPO/PPO **might work** if there's enough data. However, there was **uncertainty** and it was advised to check with DPO experts.
   - Members noted that **PreferenceDataset** should be used for this specific setup, but **Socket experts** would have final say on the strategy. One user highlighted that successes had been achieved with straight DPO/PPO on models like **llama2** and **Pythia 1B-7B**.
- **Finetuning First Model with WandB**: A user finished fine-tuning their first model using **Phi Mini (LoRA)** and sought advice on **evaluating the logs**. Recommendation was made to use **WandBLogger** for better log management and metric visualization.
   - The user was cautioned about potential issues with repeated keys in the **yaml configuration** for logging. Advice was given on setting up **WandBLogger** correctly to avoid errors and improve training oversight.
- **Evaluating Fine-tuning Logs**: Users discussed whether the **gradient size** was appropriate for the given dataset, suggesting potential adjustments. One user shared log files and asked for insights on potential **overfitting** or adjustments needed for more epochs.
   - Logs showed that **loss and learning rate** metrics might vary in small datasets, leading to challenges in understanding the results purely from log files. Emphasis was placed on using **WandB** to facilitate this process.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/nbeerbower/llama-3-gutenberg-8B">nbeerbower/llama-3-gutenberg-8B · Hugging Face</a>: no description found</li><li><a href="https://pytorch.org/torchtune/main/deep_dives/wandb_logging.html">Logging to Weights &amp; Biases &mdash; torchtune main documentation</a>: no description found</li><li><a href="https://huggingface.co/datasets/jondurbin/gutenberg-dpo-v0.1">jondurbin/gutenberg-dpo-v0.1 · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/)** (1 messages): 

mikhail_ee: Some fresh locations from https://Hexagen.World
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1256332208185151652)** (12 messages🔥): 

- **Featherless.ai launches new platform**: The **Featherless.ai** platform recently launched, offering access to all 🦙 models on Hugging Face instantly with a flat subscription starting from $10/month. Users don't need any GPU setup or to download models, as seen [here](https://featherless.ai/).
   - Early users primarily utilize this platform for AI persona local applications such as Sillytaven, with a smaller subset focusing on specific uses like language fine-tuning or SQL models.
- **Featherless.ai explores TTS model integration**: **Featherless.ai** may explore integrating TTS models like [Piper](https://github.com/rhasspy/piper/blob/master/VOICES.md) into their platform based on user feedback. A user requested this feature for generating diverse NPC voices for an online game.
   - The team explained that models small enough (like 100MB) can be run locally on CPUs, and they are primarily focusing on popular models that can't be efficiently run on CPUs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/rhasspy/piper/blob/master/VOICES.md">piper/VOICES.md at master · rhasspy/piper</a>: A fast, local neural text to speech system. Contribute to rhasspy/piper development by creating an account on GitHub.</li><li><a href="https://huggingface.co/rhasspy/piper-voices">rhasspy/piper-voices · Hugging Face</a>: no description found</li><li><a href="https://featherless.ai/"> Featherless - Serverless LLM</a>: Featherless - The latest LLM models, serverless and ready to use at your request.
</li>
</ul>

</div>
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1257021855739547811)** (2 messages): 

- **Introducing Niashamina**: A new member, Niashamina, introduced themselves as an **AI engineer** with skills on Windows.
   - They mentioned they have created a **README** for using **WSL** to launch **AI Town** on Windows and are currently working on integrating it into Docker.
- **WSL README for AI Town on Windows**: Niashamina announced they made a **README** for using **WSL** to launch **AI Town** on Windows.
   - They inquired about the appropriate place to share their **README** on GitHub.
- **Docker Integration Progress**: Niashamina mentioned they are working on **integrating AI Town into Docker**.
   - They noted the work is still in progress and humorously questioned its usefulness.
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1256421679412940830)** (12 messages🔥): 

- **Facebook's new LLM Compiler model**: Facebook released its **LLM Compiler model** capable of compiling C, optimizing assembly, and LLVM IR. Check out the [model packaged by Mozilla](https://huggingface.co/Mozilla/llm-compiler-13b-ftd-llamafile) into executable weights called llamafiles.
   - The [llamafile](https://github.com/Mozilla-Ocho/llamafile) works on Linux, MacOS, Windows, FreeBSD, OpenBSD, and NetBSD for AMD64 and ARM64. Mozilla uploaded it to Hugging Face for easier accessibility.
- **Making llamafile official on Huggingface**: To make llamafile official on Huggingface, a member suggested creating a PR into the [file for model libraries](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts). Additionally, the [snippet file](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries-snippets.ts) needs modifications.
   - This will add a button on all repositories using llamafile and fill out a code to load the model. it's important for seamless integration.
- **Realistic Hardware for llama.cpp**: Discussing realistic hardware, a member asked about running **llamafile** on devices like iPhone 13 and Raspberry Pi Zero W. It was clarified that **llamafile** requires a **64-bit system** and cannot run on Raspberry Pi Zero.
   - The RAM requirement varies; for instance, hosting embeddings to HTTP clients with **all-MiniLM-L6-v2.Q6_K.gguf** uses only 23mb of memory. Therefore, **llamafile server v2.o** uses almost no RAM.
- **Release of llamafile v0.8.9**: The **llamafile v0.8.9** release confirmed working **Android support** and introduced better **Gemma2 support**. The release includes fixes such as GPU extraction on Windows and added Google Gemma v2 support.
   - Further improvements on the new server mode were also highlighted. Check the [release details](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.9) for more information.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Mozilla/llm-compiler-13b-ftd-llamafile">Mozilla/llm-compiler-13b-ftd-llamafile · Hugging Face</a>: no description found</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.9">Release llamafile v0.8.9 · Mozilla-Ocho/llamafile</a>: This release gets Gemma2 working closer to how Google intended.  af22695 Make gemma2-27b-it the same as aistudio.google.com 41678c8 Add sliding window mask for Gemma2 140eed5 Add soft-capping to Ge...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.8">Release llamafile v0.8.8 · Mozilla-Ocho/llamafile</a>: 571b4e5 Fix bug preventing GPU extraction on Windows 4aea606 Support flash attention in --server mode 7fd9101 Don&#39;t flush bf16 subnormals to zero 7692b85 Add Google Gemma v2 support 72fb8ca Introd...
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1256490833641734164)** (8 messages🔥): 

- **Event Recording of 'From ML Engineering to AI Engineering'**: A member missed the 'From ML Engineering to AI Engineering' event and inquired about a recording. Community members shared a [Zoom link](https://events.zoom.us/ejl/AuK2GHiGErlhKuJWYRGND-S4fQfnvCgfPO50QMGDk0trhQ2ykP5H~A2Qw5SEBU-CKEpNn-eBw) but the link turned out to be invalid.
   - Despite the ongoing back-and-forth, users shared multiple links, but all resulted in invalid pages. One member mentioned that the link requires a code, complicating access further.
- **Data Talks Club LLM Zoomcamp: Building Data Pipelines**: An upcoming Zoomcamp focuses on creating open-source data pipelines using dlt and LanceDB, is scheduled for Monday, July 8 and runs for 90 minutes. The workshop covers extracting data from REST APIs, vectorizing and loading into LanceDB, and incremental loading methods.
   - Participants will learn how to deploy these pipelines in diverse environments like Python notebooks, virtual machines, and orchestrators such as Airflow, Dagster, or Mage. The event is sponsored and led by Akela Drissner, Head of Solutions Engineering at dltHub.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://events.zoom.us/ejl/AuK2GHiGErlhKuJWYRGND-S4fQfnvCgfP">Something went wrong</a>: no description found</li><li><a href="https://lu.ma/cnpdoc5n?tk=uEvsB6">Open source data ingestion for RAGs with dlt · Luma</a>: Creating scalable data pipelines - Akela Drissner About the event  In this hands-on workshop, we’ll learn how to build a data ingestion pipeline using dlt to…</li><li><a href="https://events.zoom.us/ejl/AuK2GHiGErlhKuJWYRGND-S4fQfnvCgfPO50QMGDk0trhQ2ykP5H~A2Qw5SEBU-CKEpNn-eBw">Something went wrong</a>: no description found</li><li><a href="https://events.zoom.us/ejl/AuK2GHiGErlhKuJWYRGND-S4fQfnvCgfPO50QMGDk0trhQ2ykP5H~A2Qw5SEBU-CKEpNn-eBwjM_fMUTbTOlHM-LoWdsP8pDkCYNvYe-h892C3_JDfnGrM48-PFMLhBmVxJ43wdd3-9_kpWLMCWgHe6UIk-PCEp85k/home">All-in-one virtual event platform | Zoom Events</a>: no description found
</li>
</ul>

</div>
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/)** (1 messages): 

dbreunig: Only since 5-19, but you can definitely see the pack catching up at the top
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1257361863906431128)** (1 messages): 

- **Huge corpus crashes tokenizer training**: A member is training a **BPE tokenizer for the Panjabi language** on a 50 GB text corpus but is encountering an **Out of Memory (OOM) issue** even when using a 1TB RAM instance. They shared [related GitHub issues](https://github.com/huggingface/tokenizers/issues/1434) for reference and asked for suggestions on more efficient model training.
   - The user mentioned that the memory keeps increasing even after the **Pre-processing sequences steps continue beyond len(ds)**, resembling [this issue](https://github.com/huggingface/tokenizers/issues/1345). They suspect the problem might be in the **train_from_iterator function** within the `tokenization_utils_fast.py` file but can't pinpoint the exact cause.
- **Cannot debug tokenizer training function**: The user tried debugging the code to understand the OOM issue but couldn't get into the `train_from_iterator` function in `tokenization_utils_fast.py`. They speculated that the issue might be due to calling executable/binary code running in Rust.
   - The user's attempts to trace the specific cause of the excessive memory usage have been unsuccessful, leading to further confusion and a need for community input or alternative approaches.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/tokenizers/issues/1434>)">Issues · huggingface/tokenizers</a>: 💥 Fast State-of-the-Art Tokenizers optimized for Research and Production - Issues · huggingface/tokenizers</li><li><a href="https://github.com/huggingface/tokenizers/issues/1345)">Issues · huggingface/tokenizers</a>: 💥 Fast State-of-the-Art Tokenizers optimized for Research and Production - Issues · huggingface/tokenizers</li><li><a href="https://github.com/huggingface/transformers/blob/e65502951593a76844e872fee9c56b805598538a/src/transformers/tokenization_utils_fast.py#L817>).">transformers/src/transformers/tokenization_utils_fast.py at e65502951593a76844e872fee9c56b805598538a · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

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

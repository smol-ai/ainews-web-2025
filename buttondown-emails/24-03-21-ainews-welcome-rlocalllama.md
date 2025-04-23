---
id: 6ae360e4-c736-4611-b702-7a2c02157747
title: '[AINews] Welcome /r/LocalLlama!'
date: '2024-03-21T23:33:53.811566Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-welcome-rlocalllama
---

<!-- buttondown-editor-mode: plaintext -->> AI News for 3/20/2024-3/21/2024. We checked [**358** Twitters](https://twitter.com/i/lists/1585430245762441216) and **21** Discords (**337** channels, and **9841** messages) for you. Estimated reading time saved (at 200wpm): **1033 minutes**.

It's a quiet news day - [Sakana shipped an evolutionary model merging paper](https://arxiv.org/abs/2403.13187), [OpenInterpreter launched their O1 devkit](https://x.com/openinterpreter/status/1770821439458840846?s=46&t=90xQ8sGy63D2OtiaoGJuww), and people are talking about [how Claude Haiku is underrated if you make 10-shot examples](https://x.com/mattshumer_/status/1770942240191373770).

But on the occasion of [Reddit's successful IPO](https://www.cnbc.com/2024/03/21/reddit-ipo-rddt-starts-trading-on-nyse.html) today, it's a good time to FINALLY introduce Reddit summaries to AINews! just starting with /r/LocalLlama for now, and we'll be summarizing the comments soon, but next we have r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence mapped out. Let us know if we're missing any major alpha drop subreddits.

---

**Table of Contents**

[TOC] 


---

# REDDIT: /r/LocalLlama

**Model Releases and Benchmarks**

- [Cerebrum 8x7b is here!](https://www.reddit.com/r/LocalLLaMA/comments/1bj8d4w/cerebrum_8x7b_is_here/) Aether Research released Cerebrum 8x7b based on Mixtral, trained similarly to their 7b version. It performs on par with GPT 3.5 Turbo and Gemini Pro on reasoning tasks, making it SOTA for open-source reasoning models. (201 upvotes)
- [Moistral 11B v1, the moistest Mistral there is - from the creators of Cream-Phi-2! (Finetuned, not merged)](https://huggingface.co/TheDrummer/Moistral-11B-v1?not-for-all-audiences=true) (165 upvotes)
- [New creative writing benchmark using Claude3 as judge](https://www.reddit.com/r/LocalLLaMA/comments/1bjih89/new_creative_writing_benchmark_using_claude3_as/) A creative writing benchmark was created using Claude Opus as judge, with 19 writing prompts, 36 narrowly defined assessment criteria, and exemplar reference output for each question. (14 upvotes)

**Quantization and Performance Optimization**

- [[Help/Serious Discussion] - I tried my hand at a 1.58 BitNet implementation - but I'm stuck.](https://www.reddit.com/r/LocalLLaMA/comments/1bjjywn/helpserious_discussion_i_tried_my_hand_at_a_158/) A hobbyist attempted implementing the 1.58 BitNet Ternary paper, generating models matching expected sizes (e.g. 300M params at 72MB). However, they encountered issues with training loss not decreasing and inference not working properly. (32 upvotes) 
- [The Era of 1 bit LLMs - Training, Tips, Code](https://www.reddit.com/r/LocalLLaMA/comments/1bjinlq/the_era_of_1_bit_llms_training_tips_code/) A followup to the 1.58bit paper was shared. (110 upvotes)
- [Nvidia Blackwell (h200) and FP4 precision](https://www.reddit.com/r/LocalLLaMA/comments/1bjlu5p/nvidia_blackwell_h200_and_fp4_precision/) The new Nvidia h200 chips support FP4, but it's unclear if this level of quantization is useful for LLMs in practice, as even FP8 is rarely used. (8 upvotes)

**Deployment and Serving**

- [LMDeploy is very simple to use and highly efficient for VLM deployment.[Discussion]](https://www.reddit.com/r/LocalLLaMA/comments/1bjaly4/lmdeploy_is_very_simple_to_use_and_highly/) LMDeploy v0.2.6+ supports vision-language model (VLM) inference and serving, with just a few lines of code using the `pipeline` API. Models like Qwen-VL-Chat can be served with an OpenAI compatible server or Gradio UI. (18 upvotes)
- [Searching for a GUI for LLMs APIs (openrouter, openai et simila), with plug-ins and RAG support.](https://www.reddit.com/r/LocalLLaMA/comments/1bjbzpa/searching_for_a_gui_for_llms_apis_openrouter/) A user is looking for a user-friendly GUI that supports OpenAI's ChatGPT API (or compatible like OpenRouter) and allows for plugins and RAG. (3 upvotes)
- [LocalLLM with RAG multi-user server](https://www.reddit.com/r/LocalLLaMA/comments/1bj8avr/localllm_with_rag_multiuser_server/) Someone is trying to set up gpt4all as an internal server with the sbert plugin for local files, but is having trouble getting it working over the API. (2 upvotes)

**Training Data and Fine-Tuning**

- [Pipeline for generating training data (10,000 journal entries by 10,000 different people)](https://www.reddit.com/r/LocalLLaMA/comments/1bjr3ix/pipeline_for_generating_training_data_10000/) A pipeline was built to generate diverse synthetic journal entry data for fine-tuning. It used prompt variations, life variables (job, emotion, etc.), and random selection to avoid repetitive content. (4 upvotes)
- [Fine-Tuning a Language Model for Chat](https://www.reddit.com/r/LocalLLaMA/comments/1bjgr43/finetuning_a_language_model_for_chat/) Someone is asking how to fine-tune a language model for chat on a new topic using only articles, and if a Q&A dataset is needed. (0 upvotes) 
- [Preparing training data](https://www.reddit.com/r/LocalLLaMA/comments/1bjk7dw/preparing_training_data/) A user is asking how to prepare training data for fine-tuning. (2 upvotes)

**Hardware and Compute Resources**

- [PC/GPU upgrade to run LLM locally](https://www.reddit.com/r/LocalLLaMA/comments/1bjcbn6/pcgpu_upgrade_to_run_llm_locally/) Someone is looking to upgrade their GPU to run decent LLMs locally, considering a 24GB VRAM NVIDIA card. They want to know if other components like the motherboard also need upgrading. (3 upvotes)
- [Fine tuning on laptop rtx 4080](https://www.reddit.com/r/LocalLLaMA/comments/1bjfwf2/fine_tuning_on_laptop_rtx_4080/) A user is wondering if it's feasible to do fine-tuning on models like Mistral 7B using a laptop with an RTX 4080 12GB. (2 upvotes)
- [Old mining cards P102-100 worth it when looking at price/performance?](https://www.reddit.com/r/LocalLLaMA/comments/1bjhufg/old_mining_cards_p102100_worth_it_when_looking_at/) Someone is asking if old P102-100 mining cards at $20 each are worth it for inference in terms of price/performance, given they can be unlocked to 10GB but have PCIE 1.1 x4 lanes. (1 upvote)

**Memes and Humor**

- ["Who's next?"](https://i.redd.it/5rma8h7xqipc1.png) A meme image joking about Microsoft destroying open source AI initiatives in an attempt to monopolize the market. (349 upvotes)
- [I made a game using LLMs. It is called Classroom Simulator and was inspired by The Sims and Black and White. Currently online and free to play. Link in the comments.](https://v.redd.it/zcnqywua1ipc1) (101 upvotes)
- [I hate Microsoft](https://www.reddit.com/r/LocalLLaMA/comments/1bjmsfq/i_hate_microsoft/) A user venting frustration at Microsoft for "destroying every open source initiative" in an attempt to monopolize the AI market. (92 upvotes)


# PART X: AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs

**Intel and AI Capacity**

- [@sama](https://twitter.com/sama/status/1770468022081527966): "happy to see this—excited for intel, the US, and more AI capacity!" (681k views)

**Debugging and Counterintuitive Code**

- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1770528106513600636): "2h of debugging. Whatever you say, that's counter intuitive." (541k views)
- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1770586939059523970): "This being said, I don't see how a different language design could solve this specific problem of "counterintuitiveness"." (13k views)
- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1770693914669772800): "TFW this idea that made you jump out of your bed does not work." (4k views)

**Microsoft and OpenAI**

- [@AISafetyMemes](https://twitter.com/AISafetyMemes/status/1770348812600529089): "Microsoft CEO: It would not matter if OpenAI disappeared tomorrow" (192k views)
- [@Teknium1](https://twitter.com/Teknium1/status/1770428674883699134): "Braindead move by Microsoft. Doomer grifter master booksalesman hired by msft to "lead" their new AI initiative. The guy started inflection just a few months ago, raised two billion to fund his book tour, then dips? Lmao Oh well, I guess this takes Microsoft out of the competition for good models." (180k views)
- [@Teknium1](https://twitter.com/Teknium1/status/1770431787459842092): "i guess when you're a doomer the best course of action is to lock up 2b$ in VC money that couldve gone elsewhere, then lockup 50000 h100s, then leave, then lock up microsofts own ai efforts 😏" (16k views)
- [@mark_riedl](https://twitter.com/mark_riedl/status/1770622060848378317): "Here is my thought on the news about Microsoft and Inflection AI: Nadella hired a toxic manager who abused his people, and dragged out sexual abuse cases to run their new AI division. But I guess hiring a founder of DeepMind is more important than having good leadership." (15k views)
- [@ethanCaballero](https://twitter.com/ethanCaballero/status/1770511139601871351): ""Now I am become Microsoft, the devourer of frontier model startups"" (2k views)

**Q-Star Energy-Based Model for Dialog Generation**

- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1770635047722438733): Detailed explanation of the Q-star energy-based model (EBM) ideas for dialog generation, written for an average undergraduate student. Key points: uses an abstract semantic representation space, performs optimization to find lowest-energy response, separates deciding what to say from how to say it. (186k views)
- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1770691504148750772): "Lotta folks in the QTs and replies that didn't read the 2nd post in the thread...(Hint -- that's actually the important bit.)" (16k views)
- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1770635053196034493): "But actually it's not a description of Q* at all. Rather, it's an auto-generated explanation by Claude of @ylecun's EBM project. As you see, it looks *very* similar indeed. I'd be very skeptical about these claims of OpenAI "leaks". It seems to just be summarising Yann's work." (13k views)
- [@leithnyang](https://twitter.com/leithnyang/status/1770642937413820926): "this is basically yan lecun's jepa architecture rebranded as q*" (113 views)

**Advice and Observations**

- [@gdb](https://twitter.com/gdb/status/1770532522692100299): "knowing what to do and actually doing are both critical, but it is a common mistake to value only one of them" (142k views) 
- [@gdb](https://twitter.com/gdb/status/1770677916826763387): "obsessing over the details is underrated" (113k views)
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1770587464459231269): "Very few people understand long-term thinking. And those who do will reap massive benefits." (57k views)

**Memes and Humor**

- [@KevinAFischer](https://twitter.com/KevinAFischer/status/1770640604516778216): "Nothing to see here. Just a stochastic parrot" (23k views)
- [@cto_junior](https://twitter.com/cto_junior/status/1770436957560012857): "New waifu acquired" (30k views)
- [@Nexuist](https://twitter.com/Nexuist/status/1770571250047279349): "techbros could sell 1,000,000 EVs and they'd still be bad for the world techbros could ship 100,000 tons into orbit and they'd still be bad for the world techbros could cure 10,000 quadriplegics and they'd still be bad for the world &lt;— you are here" (72k views)
- [@cto_junior](https://twitter.com/cto_junior/status/1770445195043098763): "what is wrong with you" (3k views)
- [@nearcyan](https://twitter.com/nearcyan/status/1770588147405160507): "imagine being a founder without a neuralink and you have to move your hands to do work like some old man lol" (17k views)
- [@nearcyan](https://twitter.com/nearcyan/status/1770703167501533540): "wow you guys are really weird" (3k views)
- [@cto_junior](https://twitter.com/cto_junior/status/1770689422343741887): "Imagine this running on Neuralink 🤩🤩🤩You can always be in gooncave, doesn't matter if outside is 1 hacker way or dominos" (1k views)


---

# PART 0: Summary of Summaries of Summaries


> we are concluding that Claude Opus is just the best model for top level summaries so we're discontinuing the A/B/C tests (see archives for our struggles/record). We'll be exposing parallel runs for all 3 + more models (incl Gemini 1.5!!) as this problem is topologically similar to our personalization app we'll be launching.

**1. Grok-1: The Behemoth Unleashed**

- xAI released **Grok-1**, a **314 billion parameter Mixture-of-Experts model**, sparking debates on its performance compared to GPT-3.5, Mixtral, and LLaMA. The model is available on [GitHub](https://github.com/xai-org/grok-1) under Apache 2.0 license.
- Discussions centered around Grok-1's **potential with continual pretraining**, **quantization strategies**, and the implications of its **distribution via torrents** on open-source AI credibility.
- A [high school finals exam dataset](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam) revealed Grok-1 performing closely to **GPT-4 and Claude**, despite skepticism over its quality.

**2. Innovations in Retrieval-Augmented Generation (RAG)**

- Members explored enhancing RAG models with features like **response modes** for verbose/structured outputs, **citation highlighting**, understanding intent, and task decomposition for improved relevance.
- Proposals included **balancing external context utilization with internal knowledge**, training specialized models for efficient real-time RAG operations, and **output formatting** best practices.
- Resources were shared, including a [GitHub implementation](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py) of Command R for RAG and [Cohere's model](https://cohere.ai/docs) with inline citations.

**3. Scaling Strategies and Efficiency for Large Language Models**

- Discussions revolved around **continual pretraining recipes** for scaling context lengths, with a focus on data engineering approaches highlighted in [this paper](https://arxiv.org/abs/2402.10171).
- An [arXiv paper](https://arxiv.org/abs/2403.08763) proposed cost-effective techniques like **learning rate warming and data replay** for updating LLMs without full retraining.
- The viability of **downscaling models** like [Smallstral](https://huggingface.co/AlexWortega/smallstral) was explored, showing promise in performance and efficient pretraining.

**4. Multilingual Challenges and Benchmarking for Language Models**

- Discussions touched on the **complexities of language-specific knowledge** when working with multilingual models trained on English-dominated corpora, citing [this paper](https://arxiv.org/abs/2402.10588).
- Members highlighted the need for **German-specific benchmarks** measuring native language quality, proposing university collaborations and referencing resources like [SuperGLEBer](https://www.informatik.uni-wuerzburg.de/datascience/news/single/news/our-paper-supergleber-german-language-understanding-evaluation-benchmark-was-accepted-at-the-naacl-2024/).
- The [Medusa paper](https://arxiv.org/abs/2401.10774) on efficient LLM inference and a study on [LLM impact on peer reviews](https://arxiv.org/abs/2403.07183) sparked conversations around model efficiency and academic influence.

**5. Misc**

- **LangChain Enhancements and Integrations**: LangChain users are exploring new features like **astream_events**, seeking beta testers for the advanced research assistant [Rubik's AI](https://rubiks.ai/), and sharing projects like [AI chatbots](https://github.com/Haste171/langchain-chatbot) and [bookmark managers](https://twitter.com/uogbuji/status/1768681648516661446). Integrations with **Vertex AI** and **Hugging Face** are also being discussed, along with tutorials on [building AI apps](https://youtu.be/vHjc5CEoIJE) and [plan-and-execute agents](https://www.youtube.com/watch?v=ZlJbaYQ2hm4).
- **Photonics and NVIDIA Advances**: Discussions around a [new photonics chip](https://youtu.be/8ohh0cdgm_Y) that's 1000x faster than traditional chips and NVIDIA's **H100 GPU** paired with ARM-based CPUs drawing ~850W are generating buzz. NVIDIA's **GTC keynote** also stirred excitement with mentions of a 1.8T parameter model and new hardware like the **B100 with 192GB HBM**.
- **Prompt Engineering and Testing Tools**: New tools and platforms for prompt engineering and testing are emerging, such as **Prodigy's prompt engineering features**, [PromptTools](https://github.com/hegelai/prompttools), [PromptFoo](https://github.com/promptfoo/promptfoo), **Vercel's AI Playground**, and **Helicone.ai**. Experiments with AI-enhanced blog customization and discussions on AI-augmented blogging functionalities are also taking place.

---



# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Introducing Next-Gen Stable Video 3D**: Stability.ai has launched **Stable Video 3D** (SV3D), a model superseding Stable Video Diffusion, which offers enhanced 3D and multi-view synthesis from single images. They've rolled out two new variants: **SV3D_u** for generating orbital videos and **SV3D_p** with advanced features. [Discover more about SV3D here](https://stability.ai/news/introducing-stable-video-3d).

- **Cascade's Code Conundrums**: Engaging with the Stable Diffusion community, an engineer lamented the optimization of the code for running **Stable Cascade**, mentioning that it was considerably slower and more CPU-intensive than Stable Diffusion XL (SDXL).

- **Anxiously Awaiting Stable Diffusion 3**: The engineering community is abuzz with anticipation for the release of **Stable Diffusion 3 (SD3)**, articulating hope for enhanced adherence to prompts and rumored imminent invites for early access.

- **Security Skepticism Surrounding Cryptocurrency Collaboration**: News about Stability AI's venture into blockchain partnerships concerned many engineers, fueling debates on the impact this move might have on open-source traditions and security standards.

- **The Challenge of AI on Consumer-Grade Tech**: Practical discussions indicated challenges faced when running advanced AI models like Cascade or SD3 on standard hardware setups, with a particular emphasis on GPU VRAM demands. Engineers also stressed the need for more accessible generative AI tools for various applications, including gaming.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Pro Perks or Perplexing Problems?**: Perplexity AI has granted **Pro users unlimited daily queries** on **Claude 3 Opus**, but users are raising concerns about the actual extent of "unlimited" in light of context limits. Clarification on what "unlimited" entails, in terms of use and context, is a hot topic among the community.

**AI Parenting Prospects**: A vibrant community discussion unfolded over the role of AI in simplifying complex concepts for children, underscoring the importance of an AI's developmental appropriateness and its potential in educational support.

**Perplexity Amongst the Engineers**: Despite plans to deprecate the `sonar-medium-online` model, it seems to be running post-deadline, causing user confusion. Engineers debate API behavior, with discussions around the `maxtokens` parameter and observations of different news results when queried through browsers versus the API.

**In Search of Truth and Tech Jobs**: Users shared their experiences using **Perplexity AI's Claude 3 Opus** for creative writing experiments, cleanest options query, probing North Korea's political dynamics, speculating about living on Mars, and scraping job postings. Questions abound as to the variability and reliability of provided links in search results.

**Cautious Optimism on Corporate Collaborations**: Speculation grows around **Apple and Google's potential AI integrations**, as details on generative AI collaborations are keenly discussed by members who share thoughts on tech giants' strategies and the future of AI commercialization.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Grok 1 Enters the Chat**: Elon Musk's **Grok 1**, a 314 billion parameter Mixture-of-Experts model, was released, surprising many with its size and expected below-Miqu but above-Llama2 70b performance. Interest was particularly piqued by Grok 1's comparability to Mixtral, as details were shared via links such as [xai-org on Hugging Face](https://huggingface.co/xai-org/grok-1).

- **AI Tuning Tweaks and Tips**: For fine-tuning **QLoRA** on Mistral-7b, a learning rate of `2e-4` for up to 3 epochs was the go-to choice. Creative model merging tactics were proposed, like applying UltraChat and base Mistral merging strategies to **Mistral-Yarn**, eliciting a mix of skepticism and optimism within the community.

- **Unsloth AI Hits GitHub Trend**: Unsloth AI's GitHub repository turned heads as it trended, with its owners thanking users and inviting more engineers to check out their [faster finetuning repository](https://github.com/unslothai/unsloth).

- **Vigilance Against Impersonation**: A scam account was reportedly impersonating **Daniel Han** on Discord. The community was warned to stay alert, emphasizing the importance of verifying identities and reporting suspicious accounts.

- **VRAM Woes with Model Saving**: It was noted that adequate VRAM and additional system RAM are necessary to prevent crashes when saving models like the 7b Mistral bnb 4bit. This was highlighted as an issue particularly relevant when using platforms like Colab versus local environments.

- **Creative Community Bonds over AI and Art**: Discussions in the community favored creative expression, as members supported each other's poetic endeavors. Moreover, there was an exchange of resources like a visualizer tool for Reinforcement Learning and a collection of CSS or Tailwind UI elements found at [UIverse Elements](https://uiverse.io/elements).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Grok-1 and Command-R Stir Excitement**: Engineers are discussing the large-scale Grok-1 model by xAI and the Command-R model's pending integration with LM Studio via [llama.cpp Pull Request #6033](https://github.com/ggerganov/llama.cpp/pull/6033). While some opt for smaller, efficient models like Gemma 2B or Mistral 7B due to hardware limitations, others explore the Command-R's compatibility, with links to its [Hugging Face repository](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF).

- **LM Studio Capabilities Query**: Members are seeking clarity on LM Studio's capabilities, such as using personal documents for chat and the support for plugins like autogen. Configuration files can be found on [GitHub](https://github.com/lmstudio-ai/configs), and questions regarding AI difficulties direct members to seek guidance in specific channels.

- **Seeking Hardware Harmony for AI**: Technical discussions focus on hardware configurations, including anticipated performance per dollar of the forthcoming 5090 GPU and the challenges of multi-GPU setups using PCIe risers. A particularly intense debate centers around optimal GPU choices for language model tasks and implications for cooling and power draw in custom setups.

- **AVX Beta and Model Support**: The beta app of LM Studio is an **older version** without high-priority AVX support. While it supports some models, the latest ones like **starcoder2** and **gemma** are not available. However, running the **Mistral** model on the beta app is feasible.

- **AMD ROCm's Role in LM Studio**: The ROCm libraries for AMD GPUs are essential to compatibility with LM Studio. Pre-built Windows ROCm libraries supporting gfx1031 and gfx1032 have been shared [on GitHub](https://github.com/brknsoul/ROCmLibs), but current discussions indicate that models may only utilize the primary GPU for now, with speculation about future support for dual 7000 series GPUs.

- **Agent System Evaluation in Progress**: A solitary message inquires about the selection process for an **agent system** to validate creative concepts, highlighting the member's engagement with a collaborative project on agent evaluation.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NVIDIA Takes It Slow with RTX 50-Series**: NVIDIA plans to equip its GeForce RTX 50-series "Blackwell" graphics cards with **GDDR7 memory at 28 Gbps**. This moves slower than the available 32 Gbps chips, sparking debate on the strategic choice given memory bandwidth considerations and historical trends. Link: [NVIDIA's Memory Strategy](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed).

- **AI Models Get Game Ready with MatchboxDAO**: MatchboxDAO announces a project opening game data for **AI agent development**, supported by community funding, aiming to foster innovation in gameplay AI. Link: [Game On for AI Developers](https://x.com/unkjdgames?s=21).

- **Modifying Memory - Grok-1's Launch and Limitations**: xAI's 314-billion parameter MoE model **Grok-1** faces scrutiny for marginal improvement over GPT-3.5, raising questions about the practicality of super-large models and ongoing pretraining needs.

- **OpenAI's GPT-4 Shrouded in Speculation**: NVIDIA CEO hints at a new architecture with 1.8 trillion parameters, fueling rumors that it might be **GPT-4**. This speculation includes hints at MoE configurations that OpenAI has yet to confirm officially.

- **Downscaling LLMs for Enhanced Performance**: A new approach focusing on **downscaling models**, like **Smallstral**, reveals promising results in tasks performance and continuous pretraining effectiveness. This emphasizes the versatility and potential for efficiency in AI model scaling strategies. Link: [Scaling Downward](https://huggingface.co/AlexWortega/smallstral).

- **RAG Discussion Touches New Heights**: Enhancements in RAG capabilities were avidly discussed, centering on features such as response modes and high recall relevance. The community reflects on the balance between external context utilization and internal knowledge for model outputs and explores using **smaller, specialized models** to optimize RAG pipelines. Relevant Links: [Cohere's in-line citation model](https://cohere.ai/docs), [Command R for RAG GitHub implementation](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Grok-1 Faces Scrutiny**: The [Grok-1 model](https://github.com/xai-org/grok-1) has entered the arena with questions about its performance and Twitter's chatbot interface efficacy. Engineers have concerns about Grok's model size, skeptical that larger means better, when compared to competitors like Mixtral or MiQ. Meanwhile, there's a call for accessible tutorials on Retrieval-Augmented Generation (RAG) and caution is advised regarding a PyTorch Mac bug detailed in this [GitHub issue](https://github.com/pytorch/pytorch/issues/122123).

- **Speculative Sampling in Mamba Models Challenged**: Discourse in the thunderdome of models casts doubt on speculative sampling for models like Mamba. They, unlike Transformers, may not benefit similarly from speculative sampling, and the computational cost of verification remains an obstacle. Model integration with `lm-eval-harness` is under exploration, while issues like defaulting to `gpt-2-small` and evaluation hang-ups are dissected, including a specific deadlock concern found [here](https://github.com/EleutherAI/lm-evaluation-harness/issues/1485).

- **Data Complexity Shakes Scaling Laws**: In the [#scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1218832533517766666) channel, the spotlight is on how dataset complexity impacts language model scaling laws, with syntactic properties from a Probabilistic Context-Free Grammar (PCFG) and gzip compression playing into predictions. Researchers wait with bated breath for more extensive experiments to determine hard numbers on scaling laws. 

- **N-gram Sampling Techniques Debated**: In [#interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1218288738728284241), engineers confront the challenge of sampling strings from specific n-gram statistics. An autoregressive sampling approach is proposed to create max entropy distributions aligning with these statistics, armed with a practical example shared on [GitHub](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py).

- **Shuffling The Pile for Pretraining**: Queries about The Pile data shuffling lead to clarifications that original files aren't shuffled but pretokenized data available on Hugging Face is. It's the same dataset utilized by Pythia, with a note that while individual components of The Pile are unshuffled, train/test/validation splits are expected to be mixed.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Pondering AI's Essence and Techniques**: Engineers discussed whether AI like **ChatGPT** truly "understands" language or if it's an illusion created by sophisticated next-word prediction algorithms. The impact of human training was also debated, with some suggesting that it enables conversational abilities that can surpass those of some humans.

- **Stunned by DALL-E 3's Skills**: The community expressed admiration for **DALL-E 3**’s advanced capabilities in following detailed prompts compared to its predecessors, while also considering practical aspects such as speed and image-saving. Benefits of **ChatGPT+**, which utilizes **DALL-E 3** and **GPT-4**, were also mentioned.

- **AI Models in Comparison**: **GPT-4** and **Claude** were juxtaposed based on user experiences, with discussions on their conversational capabilities, cost efficiency, and respective strengths in verbosity and political correctness.

- **Challenges and Optimizations in AI Utilization**: Users shared frustration with sensitive content filters during creative endeavors, noticed changes in ChatGPT's behavior possibly due to conflicts with browser extensions, and sought out methods to prevent refusals by AI models.

- **Learning AI Platforms and Prompt Crafting**: There was an exchange on resources for learning AI concepts, particularly with PyTorch, and the mathematical foundations necessary to dive into AI. Prompts for classification tasks were explored with the aim to enhance performance, while prompting strategies to circumvent refusals were shared.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **A Slider for Aya**: The Aya demo has integrated a *repetition penalty* and seeks contributors to add a **slider feature** in the Gradio interface. Make a contribution with a PR [here](https://huggingface.co/spaces/Tonic/Aya/discussions/3).

- **NVIDIA's Mighty Duo**: NVIDIA's **H100 GPU** and ARM-based server CPUs have been combined, drawing approximately **850W**; while benchmarks suggest the **H100 alone** could draw up to 700W. Refer to [these benchmarks](https://www.phoronix.com/review/nvidia-gh200-gptshop-ben) for details.

- **The Data Keepers of HuggingFace**: HuggingFace boasts a **data leaderboard**, highlighting over **120B models** hosted on the platform. Discover the expanse of data [here](https://huggingface.co/spaces/Weyaxi/data-leaderboard).

- **Navigating MLOps with Hugging Face and SageMaker**: An Amazon SageMaker and Hugging Face workshop offers a notebook for creating an **MLOps pipeline**; suitable for individuals looking to streamline machine learning operations. Check out the workshop [here](https://www.philschmid.de/mlops-sagemaker-huggingface-transformers).

- **Multilingual Musings and AI**: Discussions touched on machine learning models working across different languages like **Chinese and English**, highlighting the complexities when dealing with language-specific knowledge and tasks. Also, the **Medusa paper on efficient language model inference**, and a study on the impact of **LLMs on scientific peer reviews** spurred conversations on model efficiency and the influence of LLMs in academia. Refer to the Medusa paper [here](https://arxiv.org/abs/2401.10774), and the peer review impact study [here](https://arxiv.org/abs/2403.07183).

- **NL2SQL Strides and NVIDIA's Novel Chipset**: An Engineer is refining a **NL2SQL pipeline**, while NVIDIA's **Grace Hopper Superchip** was highlighted for its prowess in AI-related tasks. For NLP beginners, resources like the Hugging Face [NLP course](https://huggingface.co/learn/nlp-course/chapter1/1) and Stanford's [SLP3 manuscript](https://web.stanford.edu/~jurafsky/slp3/) were recommended, along with an inquiry into free APIs for LLM deployment, citing "ollama" as a potential resource.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Interactive Documents Revolutionize RAG**: A new approach has been proposed for handling complex queries within a [RAG pipeline](https://t.co/eCdLmlXZFj) by treating documents as interactive tools, enabling more nuanced interactions and better query resolution.

- **LlamaIndex v0.10.20 Debuts with Instrumentation**: The latest LlamaIndex update boasts an Instrumentation module, detailed through notebooks on [basic observability](https://t.co/GY4unUYOwl) and [API call tracking](https://t.co/E1d9dtkqAI).

- **Enhancing QA with Search-in-the-Chain**: A discussed paper by Shicheng Xu et al. offers a new method intertwining retrieval and planning to improve question-answering, with an emphasis on step verification and plan adjustment detailed [here](https://t.co/7gLlDyd1cV).

- **Merging RAG and Job Search**: A highlighted [blog post by Kyosuke Morita](https://t.co/1Y9TPgGHW1) delves into a job assistant tool that fuses LlamaParse with LlamaIndex to tailor job matches to candidate CVs.

- **MemGPT Webinar Expands Agent Memory**: A shared [webinar featuring Charles Packer](https://t.co/bUpqCvLweS) explores MemGPT architecture, which grants an agent memory tools to interact with a core memory, boosting function-calling abilities.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Yann LeCun's LLM Bearishness Sparks Debate**: Conversations sparked by a tweet from @Teknium1 discussed how Yann LeCun's skepticism towards large language models (LLMs) may stem from consideration of cognitive processes that don't rely on internal monologues. The discussion involved the concept of 'shape rotators' versus 'wordcels' and included reference to [an interview with someone lacking an inner monologue](https://x.com/joshwalkos/status/1767745681375015076?s=46&t=6FDPaNxZcbSsELal6Sv7Ug).

- **Grok-1's Open Release Met with Skepticism and Hope**: xAI released Grok-1, a colossal 314 billion parameter Mixture-of-Experts model, inviting the AI community to contribute to its continued training and evaluation. Skeptics and optimists alike chimed in, comparing Grok-1 to models like LLaMA and Claude, and contemplating the improvements that continual pretraining might bring as noted in Yao Fu's [thoughts on Grok's potential](https://x.com/francis_yao_/status/1769575936994013611?s=46&t=90xQ8sGy63D2OtiaoGJuww).

- **Paper Club Session Highlights - The Genesis of Attention**: The **Paper Club session** elucidated the 'why' behind the advent of the attention mechanism in transformers, illustrating its breakthrough over fixed-length encoding vectors and allowing models to refer to any part of input sequences, thus paving the way for transformer efficiency.

- **Lex Fridman's Podcast Critiqued for Lacking Depth**: Listeners voiced disappointment with Lex Fridman's podcast featuring Sam Altman, criticizing the lack of in-depth discussion on the operational intricacies and political climate of OpenAI, considering it a missed opportunity for substantial conversation in the AI space.

- **Discussion on Retrieval-Augmented Generation and Embeddings**: Within the AI in Action Club, members shared a link to "Advanced RAG 01 - Small to Big Retrieval," suggesting detailed insights on Retrieval-Augmented Generation. The concept of 'contrastive embeddings' and the application of LLMs in generating such embeddings were topics of interest, indicative of search for innovations beyond traditional cosine similarity.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**Codex Decoded in Copilot**: **Microsoft Codex** can now be accessed for free within the Copilot app, integrating Jupyter Notebooks and libraries like simpy and matplotlib, enabling a more resourceful coding environment.

**DALL-E 3 Dataset's New Home**: Confusion about the **DALL-E 3 dataset** being removed from Hugging Face was resolved; it's been relocated and is available at this [direct link](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset).

**Grok-1 Joins the AI Fray**: OpenAI's **Grok-1**, an impressive 314B parameter model, has hit the scene with a splash, performing notably well in various benchmarks. Its release on GitHub piqued interest and comparison with models like **Mixtral** and **LLaMA**, and is up for exploration [here](https://github.com/xai-org/grok-1).

**Efficient Ways to Better LLMs**: An [arXiv paper](https://arxiv.org/abs/2403.08763) discussed cost-effective methods such as learning rate warming and replay of previous data for updating LLMs without full re-training.

**Speculative GPT-4 Gossip**: Speculation abounds on **GPT-4** being a 1.8 trillion-parameter mixture of experts (MoE) model, following a hint from Nvidia. The authenticity of GPT-4's details remains unconfirmed and the topic was sparked by a [tweeted image](https://pbs.twimg.com/media/GI-reRIW0AAZpMC?format=jpg&name=large).



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Photonics Chips Blaze Past Traditional Silicon**: [Anastasia's video](https://youtu.be/8ohh0cdgm_Y) on photonic chips stimulated chatter about technology that's a thousand times faster than traditional chips, alongside mentions of resources like the [Asianometry channel](https://www.youtube.com/watch?v=29aTqLvRia8) for enthusiasts seeking in-depth knowledge on silicon photonics and light-based networks.

**Triton Debugging Gets Visual**: Engineers shared a new visualizer tool for simplifying Triton debugging, and a set of **Triton Puzzles** for deepening knowledge, available for trials on [Google Colab](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing).

**CUDA Communities Unpack Scheduler Mysteries**: Intense discussions delved into the nuances of CUDA's warp schedulers and memory management tactics, sparking a conversation about the intricacies of **ProducerProvides, ConsumerTakes**, async work, and stream synchronization.

**Reconfigurable Computing in Academia**: Members gazed into the academic niche of reconfigurable computing for efficient ML, driven by [Prof. Mohamed Abdelfattah's work](https://www.mohsaied.com/) and an [ECE 5545 course syllabus](https://abdelfattah-class.github.io/ece5545/), despite some confusion over textbook specifics resolved by referencing the course's first lecture video.

**Catching Up with CUDA**: Fresh CUDA enthusiasts were offered guidance with book recommendations like "Programming Massively Parallel Processors", available [here on Amazon](https://www.amazon.de/-/en/Wen-mei-W-Hwu/dp/0323912311), and encouragement to harness frameworks like **torch** for stepping into ML/DL realms.

**Thoughtful Threads on Striped and Flash Attention**: A healthy debate on attention mechanisms saw discussions about memory requirements contrasting *Ring Attention* and *Flash Attention*, including recommendations to consult specific literature ([Striped Attention paper](https://arxiv.org/abs/2311.09431)) and code ([GitHub implementation](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)) for clarification.

**AI and Systems Collide at MLSys 2024**: Engineers swapped details about the MLSys 2024 conference, emphasizing its critical role at the convergence of Machine Learning and Systems for facing emerging AI challenges ([MLSys Conference](https://mlsys.org/)).

**Gearing Up for a GTC Gathering**: Gautier's biggest AI enthusiasts are organizing meetups for GTC 2023, discussing visiting plans and sharing contact information while acknowledging some high-spirited humor around the constraints of attending such exclusive events.





---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**LLaMa Models Play Nice with Prompts**: The LLaMa models are confirmed to work well with prompts structured in "system", "user", and "assistant" roles, useful for those utilizing the OpenAI JavaScript library.

**Script Breaks Down Books for AI Segmentation**: An innovative script has been developed that deconstructs books for AI-driven segment generation, with notable improvements in generative quality when instruction-based data is utilized, revealed through testing with Airoboros 70B and comparing against lzlv 70B.

**Demand for In-Depth Usage Analytics Rises**: Discussions highlighted the community's need for detailed usage analytics akin to those provided by OpenAI, revealing a specific interest in insights such as daily or weekly usage costs, broken down by models and applications.

**Models Play Hard to Get**: Recent changes in model behavior have been noted, with a particular decrease in a model's willingness to perform tasks, accompanying questions about access to beta models like sonnet:beta and opus:beta. The company confirmed that there should be general access.

**API for the People, by the People**: One user plans to debut a public API and seeks to have it included in OpenRouter’s listings, prompting a positive response from the platform eager for further detail exchanges through direct messages.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**API Evolution Sparks Curiosity**: Engineers are questioning the future of LangChain's **astream_log** given the beta status of **astream_events**; concerns revolve around potential deprecation or the distinction in use cases between the two.

**Rubik's AI Awaits Eager Testers**: Beta testers are being summoned for **Rubik's AI**, a promising research assistant offering access to **Claude 3 Opus**, **GPT-4 Turbo**, and **Mistral Large**. Those interested can join the [waitlist](https://rubiks.ai/).

**LangChain JavaScript Streaming Stumbles**: Reports have surfaced of streaming issues with `RemoteRunnable` in JavaScript, unlike its functionality in Python. The community is looking for insights or fixes, with suggestions to follow up on [GitHub](https://github.com/langchain-ai/langchain/issues/13126) and LangChain's security [guidelines](https://js.langchain.com/docs/security#reporting-a-vulnerability).

**Community Showcases Diverse AI Creations**: Innovators have introduced various AI tools: an AI chatbot for data analysis ([Haste171/langchain-chatbot](https://github.com/Haste171/langchain-chatbot)), **Living Bookmarks** bot managing Raindrop.io bookmarks, a call for interviews on productivity with [NeuroFusion](https://calendly.com/neurofusion/30min), a popular AI-based scraper **Scrapegraph-ai**, and **Lyzr.ai's Automata** for simulating sales roles ([GitHub Repo](https://github.com/LyzrCore/lyzr-automata)).

**AI Learning Made Accessible**: Didactic resources on creating a personalized nutrition AI with privacy focus using **Langchain's Pebblo** are shared in a *YouTube tutorial* ([Nutriheal Demo](https://youtu.be/vHjc5CEoIJE)), along with documentation for locally deploying AI solutions, harnessing generic UI for AI assistants, and developing 'plan-and-execute' style AI agents with strategic abilities ([Langgraph Tutorial](https://www.youtube.com/watch?v=ZlJbaYQ2hm4)).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Model Mystery Unveiled Through API**: An [arXiv paper](https://arxiv.org/abs/2403.09539) discusses how queries to API-protected large language models (LLMs) could leak proprietary information such as model size – an unintended "softmax bottleneck". Concerns were raised about the accuracy of these findings, especially when models use technologies like MoE, which could skew size estimations.

**Open Source Definitions Stir Drama**: A Twitter [conversation](https://twitter.com/rasbt/status/1769779229263065184) sparked predictions of drama in the machine learning community over what should be considered "open source". This sparked conversations about including **data** in the definition of open-source software, with a push towards establishing a pragmatic consensus on the term's boundaries. Meanwhile, there is dissatisfaction with EleutherAI's social media engagement strategy.

**Grok-1 Joins The Model Party**: xAI introduced [Grok-1](https://x.ai/blog/grok-os), a **314 billion parameter MoE model**, raising discussions around its release, performance metrics, which were rumored to surpass those of Falcon, and its marketing strategy. Skepticism was voiced over torrent-based distribution affecting the reputation and policies around open-source AI models, leading to a tongue-in-cheek idea about physically shipping models via mail.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Confusion over Aribus Developments**: A guild member sought insights on developments using **Aribus**, sharing a [Twitter link](https://twitter.com/alignment_lab/status/1758949148143841379) but received no further details or clarifications within the channel.
- **In Search of HTTP-Savvy Embeddings**: Interest was expressed in locating an embeddings model trained on **HTTP responses**, with a suggestion to potentially employ a transformer model with appropriate training for the task.
- **Fine-Tuning Quest for Mistral**: An inquiry was made for a **Mistral model** that has undergone fine-tuning with both *orca-math-word-problems-200k dataset* and *nvidia/OpenMathInstruct-1*, however, there were no subsequent suggestions shared on the matter.
- **Collaborative Call for Grok 1 Enhancement**: A call to action for collaborative fine-tuning of **Grok 1** touched on the need for significant **compute** and **data resources**, mentioning that MoE training infrastructure is available to support efforts.
- **Grok 1 Benchmark Concerns and Surprising Performance**: **Grok 1** has ignited conversation around its benchmark performance on the MMLU and its close showing to **GPT-4** and **Claude** in a [high school finals exam dataset](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam), raising questions about its capabilities and the ongoing need for extensive compute and diverse data for further training.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Devin Sparks a Debate on App Complexity**: A member humorously stated that **Devin** has inspired them to prioritize simplicity in app development, suggesting that complex applications might be unnecessary.

- **Mysterious Tweet Stirs Anthropic Conspiracy**: A link to a [tweet](https://x.com/tszzl/status/1768530219378631137?s=20) indicated concern that **Anthropic** could be using their AI to influence technical personnel, implying a possible guise of controlled opposition.

- **Claude Sonnet Scales New Heights**: Someone in the guild is looking into utilizing **Claude Sonnet** for a high-usage project and is curious about others' experience with the AI at the scale of tens of millions of tokens per month.

- **Decoding the KPU Hype**: Conversations revealed skepticism about the [Knowledge Processing Unit (KPU)](https://maisa.ai/blog/kpu) claims, debating the validity of benchmark comparisons with GPT-4. Maisa's CEO clarified on [Twitter](https://x.com/davipar/status/1768683151780683919?s=20) that KPU is an architectural approach to enhance existing LLMs, not a new model.

- **Unfinished Business in OpenAI Channel**: A sole [link](https://x.com/leopoldasch/status/1768868127138549841?s=46) was mentioned in the #openai channel, with no further context provided.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **German Linguistics Troubleshooting**: Users experienced difficulties with **DiscoLM-mixtral-8x7b-v2**, particularly for generating German responses post instruction fine-tuning; one outlined a `ValueError` from using AutoModel for sequence classification, hinting at configuration issues. The community also discussed merging language models, dataset quality, and prompt consistency, emphasizing the challenges of maintaining language quality during model integration.

- **Grok Under the Microscope**: The community shared the [Grok model release](https://github.com/xai-org/grok/blob/main/model.py) on GitHub, exploring the feasibility of deploying it due to its significant parameter count (314 billion) and subsequent computational demands.

- **Evaluating German Model Mastery**: Conversations referenced benchmarks such as the *supergleber-german-language-evaluation-benchmark*, with mentions of Reddit threads and papers providing more information. Participants advocated for the creation of German-specific benchmarks in evaluation platforms, emphasizing the necessity for native speaker insight on language quality.

- **University Alliance for Language Excellence**: There was a proposal for utilizing German public university resources to develop benchmarks that more accurately assess language quality, mentioned in reference to expanding the **DiscoLM** project, and championing the value of academic partnerships.

- **Demo Delights and Dilemmas**: *jp1* shared details about **fastchat/VLLM** use in demos without special adjustments, while also noting the relocation of the demo server from personal to professional hosting, unfortunately leading to networking issues. *chromix* provided a light-hearted comparison, suggesting that more "professional" hosting environments may not always translate to increased reliability.




---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Prodigy's New Prompt Engineering Features**: Prodigy now includes prompt engineering tools for turning this task into a data annotation problem. Interested users can explore the offering on [Prodigy's feature page](https://prodi.gy/features/prompt-engineering).

- **Open Source Aids for Prompt Engineering**: The engineering community shared pointers to [PromptTools by hegelai](https://github.com/hegelai/prompttools) and [PromptFoo](https://github.com/promptfoo/promptfoo), encouraging exploration of these resources for prompt testing and handling multiple LLMs and vector databases.

- **UI for Model Benchmarks and Prompt Versioning Emerges**: Vercel's [AI Playground](https://sdk.vercel.ai/) is cited as a tool for comparing different AI models using the same prompts, and Helicone.ai's emerging prompt management and versioning capabilities are gaining recognition.

- **AI-Enhanced Blog Customization Trials**: A member has taken on a project to adapt blog content to varied personas with GPT-3.5-turbo, with a live demo available at [How to Build a Buzzword](https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html), introducing potential tools for augmenting writing focus and clarity.

- **Exploring AI's Role in Blogging**: Discussions surfaced around AI-enhanced blogging features, such as rewriting in different personas, generating counterpoints, persona-based content sharing, and offering summaries or translations.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Model Enhancement Method in the Works**: A new method for improving **global accuracy** and training efficiency is being prepared for release after the generation of improved charts and results.
- **Call for Empirical Validation at Scale**: The discussions highlighted that while promising results were observed, a lack of computational resources has stalled empirical validation of the method's effectiveness on large-scale models.
- **Helping Hands for Scaling Offered**: An offer was made to discuss the promising method and explore ways to commit **compute and resources** to validate and scale it up.
- **Substantial Gains Observed in CIFAR100**: Using VGG16 on a subset of CIFAR100 for one epoch, the method achieved significantly **higher test accuracy**, demonstrating initial success.
- **Graph Reporting Glitches Discussed**: Conversations touched on technical issues with Wandb, specifically how to update charts effectively by resetting steps when plotting new experimental data.



---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1219396298176991303)** (1 messages): 

- **Introducing Stable Video 3D**: Stability.ai announces the release of **Stable Video 3D** a model that builds upon Stable Video Diffusion, offering enhanced 3D quality and multi-view capabilities. It operates by taking a single image and outputting multiple views which can be used to generate 3D meshes; [learn more about Stable Video 3D](https://stability.ai/news/introducing-stable-video-3d).
- **Advancement Over Prior Models**: The release of SV3D marks improved performance over [Stable Zero123](https://stability.ai/news/stable-zero123-3d-generation) and other open-sourced alternatives such as [Zero123-XL](https://objaverse.allenai.org/docs/zero123-xl/), promising greatly improved quality in 3D technology.
- **Two New SV3D Variants Released**: Stability.ai's release features two variants, **SV3D_u** for generating orbital videos from single images without camera conditioning, and **SV3D_p**, which extends these capabilities with additional features.

**Link mentioned**: <a href="https://stability.ai/news/introducing-stable-video-3d">Introducing Stable Video 3D: Quality Novel View Synthesis and 3D Generation from Single Images &mdash; Stability AI</a>: When we released Stable Video Diffusion, we highlighted the versatility of our video model across various applications. Building upon this foundation, we are excited to release Stable Video 3D. This n...

  

---


**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1218109086101540905)** (988 messages🔥🔥🔥): 

- **AI Chatbots Can't Code (Yet)**: A member expressed frustration with the code for running Stable Cascade, suggesting it was poorly optimized and possibly written by a chatbot. They remarked that running Cascade took considerably longer than SDXL, with significant load on the CPU.

- **Community Hopes for SD3 Access**: In anticipation of Stable Diffusion 3 (SD3), community members were eagerly awaiting more news and access, with rumors suggesting invites might go out soon. There was speculation and hope that SD3 can improve upon the current models and their ability to adhere to prompts.

- **Stability AI's Potential Cryptocurrency Pivot Draws Concern**: News about Stability AI partnering with blockchain and crypto-based companies raised concerns among community members. They expressed consternation over the potential move away from open-source principles and toward less secure, scam-prone crypto integrations.

- **Running AI models On Limited Hardware**: Members discussed the challenges of running advanced AI (such as Cascade or SD3) on consumer-grade hardware, comparing experiences with different GPUs. It was noted that image models usually demand less VRAM compared to large language models.

- **Pressure for Practical AI Generative Tools Grows**: Community members were eager for Stable Diffusion tools that simplify the process of training or finetuning without compromising on result quality. Queries ranged from how to run them more effectively with limited resources to the potential of fine-tuning for specific use-cases like game assets creation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 is a 314B parameter Mixture of Experts model - Base model (not finetuned) - 8 experts (2 active) - 86B active parameters - Apache 2.0 license - Code:  - Happy coding! p.s. we re hiring: </li><li><a href="https://huggingface.co/coqui/XTTS-v2">coqui/XTTS-v2 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/iron-man-mr-clean-mop-ai-floors-gif-27596354">Iron Man Mr Clean GIF - Iron Man Mr Clean Mop - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/avatar-cuddle-hungry-yummy-food-gif-5610436">Avatar Cuddle GIF - Avatar Cuddle Hungry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/PollyannaIn4D">PollyannaIn4D (Pollyanna)</a>: no description found</li><li><a href="https://tenor.com/view/yess-yes-gif-25420589">Yess GIF - Yess Yes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.python.org/3/library/pickle.html">pickle — Python object serialization</a>: Source code: Lib/pickle.py The pickle module implements binary protocols for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is...</li><li><a href="https://www.pny.com/professional/software-solutions/about-nvidia-gpus/nvlink">NVLink | pny.com</a>: no description found</li><li><a href="https://civitai.com/models/207992/stable-video-diffusion-svd)">Stable Video Diffusion - SVD - img2vid-xt-1.1 | Stable Diffusion Checkpoint | Civitai</a>: Check out our quickstart Guide! https://education.civitai.com/quickstart-guide-to-stable-video-diffusion/ The base img2vid model was trained to gen...</li><li><a href="https://thedailywtf.com/articles/The_Complicator_0x27_s_Gloves">The Complicator&#39;s Gloves</a>: Good software is constantly under attack on several fronts. First, there are The Amateurs who somehow manage to land that hefty contract despite having only finished &quot;Programming for Dummies&quot...</li><li><a href="https://stability.ai/news/introducing-stable-video-3d">Introducing Stable Video 3D: Quality Novel View Synthesis and 3D Generation from Single Images &mdash; Stability AI</a>: When we released Stable Video Diffusion, we highlighted the versatility of our video model across various applications. Building upon this foundation, we are excited to release Stable Video 3D. This n...</li><li><a href="https://civitai.com/models/351450/proteus-rundiffusion?dialog=commentThread&commentId=372974">Proteus-RunDiffusion - withclip | Stable Diffusion Checkpoint | Civitai</a>: Introducing Proteus-RunDiffusion In the development of Proteus-RunDiffusion, our team embarked on an exploratory project aimed at advancing the cap...</li><li><a href="https://www.pny.com/professional/software-so">Page Not Found | pny.com</a>: no description found</li><li><a href="https://new.reddit.com/r/StableDiffusion/comments/1b6skvx/wheres_waldo_beach_scenes_as_an_animated_loop/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=fibDNwF8bjs">WKUK - Anarchy [HD]</a>: Economic ignorance at its most comical.— &quot;Freedom, Inequality, Primitivism, and the Division of Labor&quot; by Murray Rothbard (http://mises.org/daily/3009).— &quot;Th...</li><li><a href="https://youtu.be/ruANV24h0Dw?si=rVFKZqowCdpKTzgp">Короткометражный мультфильм &quot;Парк&quot; (сделан нейросетями)</a>: Короткометражный мультфильм &quot;Парк&quot; - невероятно увлекательный короткометражный мультфильм, созданный с использованием нейросетей.</li><li><a href="https://github.com/mix1009/sdwebuiapi">GitHub - mix1009/sdwebuiapi: Python API client for AUTOMATIC1111/stable-diffusion-webui</a>: Python API client for AUTOMATIC1111/stable-diffusion-webui - mix1009/sdwebuiapi</li><li><a href="https://www.youtube.com/watch?v=YTE0OTVOnZU">Vancouver, Canada 1907 (New Version) in Color [VFX,60fps, Remastered] w/sound design added</a>: I colorized , restored and I added a sky visual effect and created a sound design for this video of Vancouver, Canada 1907, Filmed from the streetcar, these ...</li><li><a href="https://github.com/DiffusionDalmation/pt_to_safetensors_converter_notebook#">GitHub - DiffusionDalmation/pt_to_safetensors_converter_notebook: This is a notebook for converting Stable Diffusion embeddings from .pt to safetensors format.</a>: This is a notebook for converting Stable Diffusion embeddings from .pt to safetensors format. - DiffusionDalmation/pt_to_safetensors_converter_notebook</li><li><a href="https://www.youtube.com/watch?v=5mIWo6dgTmI&ab_channel=Megaprojects">The Mushroom Motherboard: The Crazy Fungal Computers that Might Change Everything</a>: Unlock the secrets of fungal computing! Discover the mind-boggling potential of fungi as living computers. From the wood-wide web to the Unconventional Compu...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API)">Home</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://github.com/Stability-AI/generative-models">GitHub - Stability-AI/generative-models: Generative Models by Stability AI</a>: Generative Models by Stability AI. Contribute to Stability-AI/generative-models development by creating an account on GitHub.</li><li><a href="https://github.com/chaojie/ComfyUI-DragAnything/tree/main">GitHub - chaojie/ComfyUI-DragAnything</a>: Contribute to chaojie/ComfyUI-DragAnything development by creating an account on GitHub.</li><li><a href="https://github.com/GraftingRayman/ComfyUI-Trajectory">GitHub - GraftingRayman/ComfyUI-Trajectory</a>: Contribute to GraftingRayman/ComfyUI-Trajectory development by creating an account on GitHub.</li><li><a href="https://youtu.be/m9jg1fdOiVY?t=412">Install ComfyUI on Mac OS (M1, M2 or M3)</a>: This video is a quick wakthrough to show how to get Comfy UI installed locally on your m1 or m2 mac. Find out more about AI Animation, and register as an AI ...</li><li><a href="https://stable-diffusion-art.com/regional-prompter/)">Regional Prompter: Control image composition in Stable Diffusion - Stable Diffusion Art</a>: Do you know you can specify the prompts for different regions of an image? You can do that on AUTOMATIC1111 with the Regional Prompter extension.
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1219057096780419163)** (1 messages): 

- **Unlimited Claude 3 Opus Queries for Pro Users**: The announcement reveals that **Perplexity Pro users** have been granted **unlimited daily queries** on Claude 3 Opus, claimed to be the best Language Model (LLM) currently available. Pro users can take full advantage of the offering starting now.
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1218100055626743851)** (795 messages🔥🔥🔥): 

- **Perplexity Pro Confusions**: Users express confusion over Perplexity AI's context limits and "unlimited" claims. Conversations note misunderstandings about Pro search usage, with a focus on the need for clarity in Perplexity's descriptions.
  
- **Claude 3 Opus Discussions**: Users discuss the capabilities and integration of Claude 3 Opus within Perplexity AI, comparing it to GPT-4 and other models. A conversation centers around the mystery of this model's "unlimited" usage and any potential context limitations.
  
- **Parenting and AI**: A vibrant debate erupts regarding AI's role in explaining complex topics to children, with one user advocating its use for simplifying concepts. Discussions also touch on the developmental capacity of children and the advantages of AI in education.

- **Debates on AI Responsiveness**: Users deliberate over AI's ability to stick to specific prompts, sharing insights and challenges when trying to instruct AI to provide concise responses or tailor its usage to children's questions.

- **Prospective Partnerships and Updates**: Speculation arises around potential partnerships and integrations involving Apple, Google, and the generative AI landscape, with users sharing news links and thoughts about company strategies.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/3/18/24104626/apple-license-google-gemini-generative-ai-openai-chatgpt">Apple’s AI ambitions could include Google or OpenAI</a>: Another big Apple / Google deal could be on the horizon.</li><li><a href="https://x.com/AravSrinivas/status/1769475725965566167?s=20">Tweet from Aravind Srinivas (@AravSrinivas)</a>: We have made the number of daily queries on Claude 3 Opus (the best LLM in the market today) for Perplexity Pro users, unlimited! Enjoy!</li><li><a href="https://x.com/AravSrinivas/status/1769485603622867394?s=20">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Yep, thanks to @elonmusk and xAI team for open-sourcing the base model for Grok. We will fine-tune it for conversational search and optimize the inference, and bring it up for all Pro users!  ↘️ Quoti...</li><li><a href="https://tenor.com/view/shikimori-shikimoris-not-just-cute-shikimoris-not-just-a-cutie-anime-anime-an">no title found</a>: no description found</li><li><a href="https://tenor.com/view/shikimori-shikimoris-not-just-cute-shikimoris-not-just-a-cutie-anime-anime-anime-girl-gif-26002811">Shikimori Shikimoris Not Just Cute GIF - Shikimori Shikimoris Not Just Cute Shikimoris Not Just A Cutie Anime - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://us.nothing.tech/pages/perplexity">Nothing Perplexity Offer</a>: Here at Nothing, we’re building a world where tech is fun again. Remember a time where every new product made you excited? We’re bringing that back.</li><li><a href="https://fxtwitter.com/BrivaelLp/status/1769482175005577571?s=20">Tweet from Brivael (@BrivaelLp)</a>: Zuck just reacted to the release of Grok, and he is not really impressed.  &#34;314 billion parameter is too much. You need to have a bunch of H100, and I already buy them all&#34; 🤣</li><li><a href="https://x.com/technology/status/1769597406243360937?s=20">Tweet from Bloomberg Technology (@technology)</a>: EXCLUSIVE: Apple is in talks to build Google’s Gemini AI engine into the iPhone in a potential blockbuster deal https://trib.al/YMYJw2K</li><li><a href="https://youtube.com/clip/Ugkx9gPr2y53Be9C99y-EVVWfZPjRxNQo6FL?si=0r1zDbn2FfjmrsuB">✂️ Sam Altman on AI LLM Search</a>: 47 seconds · Clipped by Syntree · Original video &quot;Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power &amp; AGI | Lex Fridman Podcast #419&quot; by Le...</li><li><a href="https://youtu.be/OPoWMXqq62Q?si=jk-ZbhjfkZtRkjz7">What Are These Companies Hiding?</a>: Thoughts on the Rabbit R1 and Humane Ai PinIf you&#39;d like to support the channel, consider a Dave2D membership by clicking the “Join” button above!http://twit...</li><li><a href="https://fccid.io/2BFB4R1">FCC ID 2BFB4R1 AI Companion by Rabbit Inc.</a>: FCC ID application submitted by Rabbit Inc. for AI Companion for FCC ID 2BFB4R1. Approved Frequencies, User Manuals, Photos, and Wireless Reports.
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1218101595586429048)** (35 messages🔥): 

- **Creative Exploration with Claude 3 Opus**: An intriguing creative writing experiment titled "ever increasing intelligence until it's unintelligible to humans" was conducted using **Claude 3 Opus**. The task can be explored further [here](https://www.perplexity.ai/search/increasing-intelligence-of-HLUn3nOzSx6Nc5ecNpe5pA).
- **Visibility Is Key**: Users are reminded to make sure their threads are shared publicly to ensure community visibility. Instructions found in the [Discord link](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Debate on Cleanliness**: A discussion on which option is cleaner sparked interest and can be seen [here](https://www.perplexity.ai/search/Which-is-cleaner-qIQdwpX1QjiFQvEBgwiydQ).
- **North Korea's Dynamic**: A Perplexity search related to **North Korea's Kim** and their actions generated curiosity. Insightful results are available [here](https://www.perplexity.ai/search/North-Koreas-Kim-.uALFoJfS0mVkML42bECvA).
- **Questions About the Future**: The community shared intrigue regarding when humans might live on Mars and other queries about the future. The engaging discussion is available [here](https://www.perplexity.ai/search/When-can-human-lrFdtQ6NTvCb6LYe.WkreQ).
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1218160850670583828)** (64 messages🔥🔥): 

- **Model Deprecation Confusion**: The `sonar-medium-online` model was expected to be deprecated on March 15, but users have observed that it is still functional and not just rerouting to alternative models. There has been speculation about whether the deprecation comes into effect at the end of the day or if plans have changed.
- **API Giveth and API Taketh Away**: When using `sonar-medium-online`, one user found inconsistencies between news pulled via the web browser versus the API, highlighting differing responses regarding recent news about Donald Trump.
- **Quest for Links in the Job Market Jungle**: A user was trying to use the Perplexity API to get specific job posting links. It's noted that while occasionally the API provides actual job position links, other times it only returns links to job search platforms like LinkedIn or Glassdoor.
- **Dancing with Tokens – Max or Min?**: There was a discussion about how setting the `maxtokens` parameter affects the API's response. The consensus reveals that if set too low, the API may provide incomplete responses; if too high, it might not utilize the available space, suggesting the model does not "fill" extra space but focuses on complete responses.
- **Seeking Sources & Citations**: A conversation regarding URL citations confirms that feature is still in beta, linking to an application form for those interested. Additionally, current API access for 'Pro' users to URL citations from the closed beta was discussed, and users shared links for the application and discussions on model performance comparisons.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai">pplx-api</a>: no description found</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api form</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1218108656428650526)** (853 messages🔥🔥🔥): 

- **Grok 1: The Behemoth Unleashed**: Elon Musk's release of **Grok 1**, a 314 billion parameter Mixture-of-Experts model, has sparked discussions due to its size and impracticality for most users. The model was anticipated to be undertrained, with performance slightly below Miqu, slightly above Llama2 70b, and on par with Mixtral.

- **Hyperparameters for QLoRA**: The preferred hyperparameters for fine-tuning **QLoRA** on Mistral-7b seem to be a learning rate of `2e-4` and up to 3 epochs, as suggested in Unsloth's notebooks. However, users are encouraged to adjust these settings according to specific tasks and datasets.

- **Impersonation Alert in Discord**: Users reported a scam account pretending to be **Daniel Han** (`starsupernova`) on Discord. Reports to Discord have been filed, and users are cautioned to be wary of friend requests from the impersonator and report if encountered.

- **New Tools and Integrations**: AIKit introduced an integration for fine-tuning with **Unsloth**, providing users the ability to fine-tune language models with a config file and create OpenAI compatible model images using Docker. WandB (Weights & Biases) has been suggested for monitoring and visualizing training data.

- **Understanding Quantization**: There's a continued interest in understanding quantization for language models. A 4-bit BnB quantization reduces model sizes by reducing the bits per weight, but resources for learning about quantization were requested. Fine-tuning guidelines and dataset structuring for instruction tuning were also sought after by community members.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://docs.anthropic.com/claude/page/cosmic-keystrokes">Cosmic keystrokes</a>: no description found</li><li><a href="https://x.ai/about">About xAI</a>: no description found</li><li><a href="https://x.ai/blog/grok">Announcing Grok</a>: no description found</li><li><a href="https://lightning.ai/live-session/a35263e0-0428-40b6-8828-8e72773a284d">Lightning AI | Turn ideas into AI, Lightning fast</a>: The all-in-one platform for AI development. Code together. Prototype. Train. Scale. Serve. From your browser - with zero setup. From the creators of PyTorch Lightning.</li><li><a href="https://huggingface.co/xai-org/grok-1">xai-org/grok-1 · Hugging Face</a>: no description found</li><li><a href="https://x.ai/">Blog</a>: no description found</li><li><a href="https://substack.recursal.ai/p/eaglex-17t-soaring-past-llama-7b">🦅 EagleX 1.7T : Soaring past LLaMA 7B 2T in both English and Multi-lang evals (RWKV-v5)</a>: A linear transformer has just cross the gold standard in transformer models, LLaMA 7B, with less tokens trained in both English and multi-lingual evals. A historical first.</li><li><a href="https://arxiv.org/abs/2401.04088">Mixtral of Experts</a>: We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. Mixtral has the same architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (...</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-72B">Qwen/Qwen1.5-72B · Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog/gemma-bugs">Unsloth Fixing Gemma bugs</a>: Unsloth fixing Google&#x27;s open-source language model Gemma.</li><li><a href="https://huggingface.co/damerajee/Llamoe-test">damerajee/Llamoe-test · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://sozercan.github.io/aikit/">Introduction | AIKit</a>: AIKit is a one-stop shop to quickly get started to host, deploy, build and fine-tune large language models (LLMs).</li><li><a href="https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2">How to Fine-Tune an LLM Part 1: Preparing a Dataset for Instruction Tuning</a>: Learn how to fine-tune an LLM on an instruction dataset! We&#39;ll cover how to format the data and train a model like Llama2, Mistral, etc. is this minimal example in (almost) pure PyTorch.</li><li><a href="https://openhands.ai4bharat.org/en/latest/instructions/datasets.html#supported-datasets">ISLR Datasets &mdash; 👐OpenHands  documentation</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1768991010938404879">Tweet from Unsloth AI (@UnslothAI)</a>: Unsloth is trending on GitHub this week! 🙌🦥  Thanks to everyone & all the ⭐️Stargazers for the support!  Check out our repo: http://github.com/unslothai/unsloth</li><li><a href="https://huggingface.co/papers/2402.18668#65f0f5f8de069cd5c55f1dd2">Paper page - Simple linear attention language models balance the recall-throughput
  tradeoff</a>: no description found</li><li><a href="https://huggingface.co/Crystalcareai/GemMoE-Beta-1">Crystalcareai/GemMoE-Beta-1 · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2310.17680">CodeFusion: A Pre-trained Diffusion Model for Code Generation</a>: Imagine a developer who can only change their last line of code, how often would they have to start writing a function from scratch before it is correct? Auto-regressive models for code generation fro...</li><li><a href="https://huggingface.co/argilla">argilla (Argilla)</a>: no description found</li><li><a href="https://github.com/AI4Bharat/OpenHands">GitHub - AI4Bharat/OpenHands: 👐OpenHands : Making Sign Language Recognition Accessible. | **NOTE:** No longer actively maintained. If you are interested to own this and take it forward, please raise an issue</a>: 👐OpenHands : Making Sign Language Recognition Accessible. | **NOTE:** No longer actively maintained. If you are interested to own this and take it forward, please raise an issue - AI4Bharat/OpenHands</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py">transformers/src/transformers/models/mixtral/modeling_mixtral.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://github.com/jiaweizzhao/GaLore?tab=readme-ov-file#install-galore-optimizer">GitHub - jiaweizzhao/GaLore</a>: Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/HirCoir/Piper-TTS-Spanish">Piper TTS Spanish - a Hugging Face Space by HirCoir</a>: no description found</li><li><a href="https://github.com/xai-org/grok-1/issues/6#issuecomment-2002664859">Error when installing requirements · Issue #6 · xai-org/grok-1</a>: i have installed python 3.10 and venv. Trying to &quot;pip install -r requirements.txt&quot; ERROR: Ignored the following versions that require a different python version: 1.6.2 Requires-Python &gt;=3...</li><li><a href="https://github.com/mistralai/mistral-src">GitHub - mistralai/mistral-src: Reference implementation of Mistral AI 7B v0.1 model.</a>: Reference implementation of Mistral AI 7B v0.1 model. - mistralai/mistral-src</li><li><a href="https://www.youtube.com/watch?v=jvqFAi7vkBc">Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power &amp; AGI | Lex Fridman Podcast #419</a>: Sam Altman is the CEO of OpenAI, the company behind GPT-4, ChatGPT, Sora, and many other state-of-the-art AI technologies. Please support this podcast by che...</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://youtu.be/rANv5BVcR5k">Mistral Fine Tuning for Dummies (with 16k, 32k, 128k+ Context)</a>: Discover the secrets to effortlessly fine-tuning Language Models (LLMs) with your own data in our latest tutorial video. We dive into a cost-effective and su...</li><li><a href="https://the-decoder.com/falcon-180b-open-source-language-model-outperforms-gpt-3-5-and-llama-2/">Falcon 180B open-source language model outperforms GPT-3.5 and Llama 2</a>: The open-source language model FalconLM offers better performance than Meta&#039;s LLaMA and can also be used commercially. Commercial use is subject to royalties if revenues exceed $1 million.</li><li><a href="https://huggingface.co/datasets/teknium/GPT4-LLM-Cleaned">teknium/GPT4-LLM-Cleaned · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/pull/97">Staging PR for implimenting Phi-2 support. by cm2435 · Pull Request #97 · unslothai/unsloth</a>: ….org/main/getting-started/tutorials/05-layer-norm.html]</li><li><a href="https://github.com/huggingface/transformers/pull/29588">FEAT / Optim: Add GaLore optimizer by younesbelkada · Pull Request #29588 · huggingface/transformers</a>: What does this PR do? As per title, adds the GaLore optimizer from https://github.com/jiaweizzhao/GaLore Fixes: #29512 This is how I am currently testing the API: import torch import datasets from ...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1218580567453470860)** (1 messages): 

- **Unsloth AI Shines on GitHub**: Unsloth AI has seen a surge of activity on GitHub this week, earning a spot as a trending project. The Unsloth team expressed gratitude to the community and stargazers, inviting more users to star their [faster and more efficient finetuning project](https://github.com/unslothai/unsloth).

**Link mentioned**: <a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth

  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1218112720994308122)** (25 messages🔥): 

- **The Coincidence of Thoughts**: A discussion about *coincidences* occurred with members sharing experiences of thinking about something only to encounter it elsewhere. One cited an example of thinking of a username and then seeing it used by someone else, while positing that our minds pick up information unconsciously, drawing a parallel to how children learn.

- **Encouragement for Creative Expression**: Members encouraged each other's monologues and were open to sharing and discussing *poetic compositions*, showing support for creative endeavors within the community.

- **Exploring Fine-Tuning for Classification Tasks**: In the realm of AI fine-tuning, a member shared their experience with **Mistral-7b** on a *domain-specific classification task* and pondered whether to try **Gemma 7b**. Another member assured the group that all bug fixes for Unsloth were complete and suggested that the strength of Gemma and Mistral could vary.

- **Clarification on AI Model Branches**: A member sought assistance locating the "Mixtral branch" of an AI model. Helpful responses guided them to the right place, with a link to the related pull request on GitHub provided ([Mixtral Support Pull Request](https://github.com/unslothai/unsloth/pull/145)).

- **Sharing Open-Source UI Elements and Maps**: Links to resources were shared among the community including a map visualization for agents in a RL poker game and a collection of open-source UI elements made with CSS or Tailwind ([UIverse UI Elements](https://uiverse.io/elements)).

- **Expressions of Support for Applications**: Members expressed supportive sentiments toward those considering applications to unspecified opportunities, reinforcing the atmosphere of friendly encouragement within the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pwhiddy.github.io/pokerl-map-viz/">Pokemon Red Map RL Visualizer</a>: no description found</li><li><a href="https://uiverse.io/elements">4217 UI elements: CSS &amp; Tailwind</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/pull/145">[WIP] add support for mixtral by tohrnii · Pull Request #145 · unslothai/unsloth</a>: Mixtral WIP
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1218104575022727230)** (568 messages🔥🔥🔥): 

- **Dealing with VRAM demands during model saving**: A user noticed high VRAM usage during the saving process, in addition to VRAM used to load the model. Saving on a machine with 8GB VRAM for a 7b Mistral bnb 4bit model led to crashes, indicating that adequate VRAM plus additional system RAM is necessary for successful model saving.

- **VRAM clearing during model saving may not help**: When suggested to restart the computer to clear VRAM as a solution for crashes during model saving, it was clarified that the model needs to be loaded into VRAM for saving, hence restarting does not suffice.

- **Colab resources for training and saving models**: A user successfully ran code in Colab after failing initially, emphasizing the luck element in gaining enough resources on the platform.

- **Differences between models saved in Colab vs. local machines**: 8GB VRAM appears suitable for running the 7b Mistral bnb 4bit model, highlighting a discrepancy between VRAM requirements for operating in Colab versus local setups.

- **Targeting model merging tactics**: A suggestion was made to apply tactics used when merging UltraChat with base Mistral to Mistral-Yarn, with discussions implying a mix of skepticism and optimism based on previous experiences with model merging approaches.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit">ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1X_PHYBawrsCgKfMEPxvIDX__rYa1-v97?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook">Kaggle Mistral 7b Unsloth notebook</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/unsloth/mistral-7b-instruct-v0.2-bnb-4bit">unsloth/mistral-7b-instruct-v0.2-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth">DPO Trainer</a>: no description found</li><li><a href="https://github.com/artidoro/qlora/blob/main/qlora.py#L746">qlora/qlora.py at main · artidoro/qlora</a>: QLoRA: Efficient Finetuning of Quantized LLMs. Contribute to artidoro/qlora development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://docs.gpt4all.io/gpt4all_python.html">Generation - GPT4All Documentation</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=FqfebeAdT073,">Google Colaboratory</a>: no description found</li><li><a href="https://pastebin.com/ybSeKHhU">Unsloth: Merging 4bit and LoRA weights to 16bit...Unsloth: Will use up to 5.34 - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments">Trainer</a>: no description found</li><li><a href="https://github.com/huggingface/trl/issues/1041">Does DPOTrainer loss mask the prompts? · Issue #1041 · huggingface/trl</a>: Hi quick question, so DataCollatorForCompletionOnlyLM will train only on the responses by loss masking the prompts. Does it work this way with DPOTrainer (DPODataCollatorWithPadding) as well? Looki...</li><li><a href="https://huggingface.co/docs/trl/v0.7.11/en/sft_trainer#train-on-completions-only).">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/discussions/2/files">HuggingFaceH4/zephyr-7b-alpha · Add chat template</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha#intended-uses--limitations">HuggingFaceH4/zephyr-7b-alpha · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L56">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md">llama.cpp/examples/server/README.md at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>: Python bindings for llama.cpp. Contribute to abetlen/llama-cpp-python development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/alignment-handbook/issues/45#issuecomment-1845598205">Reproducing of Lora Model  Result on MT-Bench · Issue #45 · huggingface/alignment-handbook</a>: Recently, I attempted to fit the DPO on my own dataset. Initially, I tried to reproduce the results of your LORA model( 7.43 on MT-Bench). However, I encountered some issues. Despite using all your...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1218239216975351928)** (21 messages🔥): 

- **Reading List Material**: A member mentioned an *amazing paper* that they came across on Twitter, adding it to their reading list.
- **Training Duration Debate**: A discussion ensued about the optimal number of epochs for training a model, with one member suggesting a maximum of 4 epochs, and stating that **3 epochs is the standard** for fine-tuning language models.
- **Finding the Sweet Spot**: In the journey for maximum knowledge retention, a member was advised against excessive epochs as it might lead the model to memorize the dataset without retaining broader knowledge.
- **Parameter-to-Token Ratio Questioned**: Another conversation revolved around the right amount of trainable parameters in relation to the dataset size, hinting that a model with 800,000 lines might need a 32 or 64 rank with a suggestion that **alpha = rank * 2**.
- **Model Integration Suggestion**: A member shared links to **Tiny Mistral** and **Tiny Mistral Instruct**, small models on Hugging Face that can possibly be integrated into the Unsloth Repository, with a brief insight into the configuration of the models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1">Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/M4-ai/TinyMistral-6x248M-Instruct/tree/main">M4-ai/TinyMistral-6x248M-Instruct at main</a>: no description found
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1218098224586293319)** (301 messages🔥🔥): 

- **Curious Newcomers and Old-Timers Alike**: The channel welcomes new members, such as a passionate software engineer eager to explore large language models on a Mac M3 Pro and a self-described "curious geek" excited about diving into the AI world. The community offers suggestions for starting models and models that run on specific hardware configurations.

- **In Search of Guidance and Solutions**: Users sought advice for software issues like being stuck in a validating file integrity loop, configuring GPUs to use in LM Studio, and resolving JavaScript errors in Kali Linux. In many cases, community members provide troubleshooting assistance and workarounds like hiding GPUs via the NVIDIA Control Panel.

- **Tools, Support, and Plugin Discussions**: The community discusses integrations such as using the continue extension in VSCode for autopilot coding, as well as the constraints of running models locally (including large ones like Grok-1), and the limits of model size when considering GPU resources. In particular, a user shares success in integrating Visual Studio Code with LM Studio for coding tasks.

- **Seeking Model Capabilities and Aware of Limitations**: Users inquire about the potential for models to read and process files and documents within LM Studio, and whether functions or document retrieval is supported. Others ponder the feasibility of running open-sourced models like Grok-1 locally due to such considerable size and parameters.

- **LM Studio Development and Support Queries**: Discussions pop up about the ongoing development of LM Studio, including upcoming support for specific models like commandr, starcoder2, and miqu-103B. Users also engage regarding the creation of chat templates for OpenChat integrations and recommendations for models suitable for learning Python.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 is a 314B parameter Mixture of Experts model - Base model (not finetuned) - 8 experts (2 active) - 86B active parameters - Apache 2.0 license - Code:  - Happy coding! p.s. we re hiring: </li><li><a href="https://tenor.com/view/ratha-gif-26742750">Ratha GIF - Ratha - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/xai-org/grok-1/discussions/30">xai-org/grok-1 · 314B  params  has  297G  file size ?</a>: no description found</li><li><a href="https://github.com/continuedev/continue/issues/713"">Issues · continuedev/continue</a>: ⏩ The easiest way to code with any LLM—Continue is an open-source autopilot for VS Code and JetBrains - Issues · continuedev/continue</li><li><a href="https://www.youtube.com/watch?v=lCZRwrRvrWg&">Mistral: Easiest Way to Fine-Tune on Custom Data</a>: This video is sponsored by Gradient.ai, check them out here: https://gradient.1stcollab.com/engineerpromptIn this video, we will learn how to fine-tune Mistr...</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g&">[1hr Talk] Intro to Large Language Models</a>: This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What the...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1218119135423234058)** (138 messages🔥🔥): 

- **Anticipation for Command-R Model Support**: The integration of Command-R model with LM Studio is eagerly awaited by members, who are asking about beta access. Current discussions indicate support for Command-R in the next release of LM Studio; [Pull Request #6033 on llama.cpp](https://github.com/ggerganov/llama.cpp/pull/6033) which adds the model, has been merged, awaiting LM Studio update.

- **Grok Model Buzz**: The newly released Grok-1 base model by xAI, discussions around it highlight its enormous size and potential cost for hardware and hosting. Members share thoughts and information about Grok, including a [discussion on ycombinator](https://news.ycombinator.com/item?id=39737281) and a [blog post with further details](https://x.ai/blog/grok-os).

- **Seeking Smaller and Efficient Models**: Users with limited VRAM are looking for model recommendations that can run on GPUs like the RTX 2070 Super and the GTX 1660 Super. Consensus suggests smaller models like Gemma 2B or Mistral 7B at higher quantizations may operate within hardware constraints.

- **Inquiry about Chat Templates for OpenChat**: Users are attempting to configure custom chat templates for OpenChat, with one [proposing a template structure](https://huggingface.co/01-ai/Yi-9B-200K) for models like Yi-9B-200K; discussions suggest that personal experimentation and documentation review are key to proper setup.

- **Yi Model Architecture Curiosities**: The architecture and capabilities of the Yi-9B-200K model sparked curiosity, leading to conversations about the transformer architecture, parameter significance, and context length. Educational resources like Andrej Karpathy's "[Intro to Large Language Models](https://youtu.be/zjkBMFhNj_g)" talk and supplementary YouTube videos were shared to help with understanding.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=39737281">no title found</a>: no description found</li><li><a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: In this work, we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and c...</li><li><a href="https://huggingface.co/01-ai/Yi-34B/discussions/23">01-ai/Yi-34B · Prompt template?</a>: no description found</li><li><a href="https://huggingface.co/01-ai/Yi-9B-200K">01-ai/Yi-9B-200K · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/PAbZRGGYNyM?si=xVNZCYUddDvoFUly">What are  Parameters in Large Language Model?</a>: What are the Parameters in the Large Language Model? 00:26 💡 Parameters in large language models like GPT-3 are variables learned during training to minimiz...</li><li><a href="https://youtu.be/zjkBMFhNj_g?si=Rn96V9CMqEHLy6-7">[1hr Talk] Intro to Large Language Models</a>: This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What the...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6033">Add Command-R Model by acanis · Pull Request #6033 · ggerganov/llama.cpp</a>: Information about the Command-R 35B model (128k context) can be found at: https://huggingface.co/CohereForAI/c4ai-command-r-v01 Based on the llama2 model with a few changes:  New hyper parameter to...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1218213037060657273)** (12 messages🔥): 

- **Confusion about Command-R 35B Compatibility**: A discussion about a [Hugging Face repository](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF) led to some confusion regarding llama.cpp's compatibility with the Command-R model from CohereForAI. It was clarified that despite the GGUF format being available, llama.cpp does not currently support the c4ai model.
- **Mixed Messages on llama.cpp Support**: A member clarified a misunderstanding, stating that llama.cpp actually does support the c4ai model, contradicting a previous message in the conversation.
- **Call for AMD OpenCL Drivers Notification**: A suggestion was made for the website's Linux download page to inform AMD users that they need OpenCL drivers to use their GPU with the program.
- **Guidance Sought for AI Difficulties**: A user expressed frustration over the complexity of using AI, and was directed to a specific channel, presumably for better support and detailed assistance.
- **Inquiry About LM Studio Capabilities**: Questions arose about whether personal documents could be used for chatting in LM Studio or if plugins like autogen could be integrated. It was explained that plugins like autogen/langchain are already supported via server mode connections.

**Link mentioned**: <a href="https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF">andrewcanis/c4ai-command-r-v01-GGUF · Hugging Face</a>: no description found

  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1218129474348912711)** (480 messages🔥🔥🔥): 

- **Debate Over Optimal GPU Choices**: Community members are discussing the prospective power and value of the forthcoming 5090 GPU for LM tasks, comparing it to the 3090 and 4090. Opinions indicate that while the 5090 may offer better performance per dollar for general AI tasks, the bandwidth/$ may not exceed that of a 3090.

- **Wish for a Single Slot 5090**: A desire for a single-slot version of the 5090 GPU is expressed to facilitate multi-GPU setups. Additionally, there is a discussion on the effectiveness of the Fractal North case for housing such setups and observations on cooling needs, like the efficacy of Corsair's 7000x tower for managing power draw and heat.

- **The Quest for Max PCIe 4.0 Slots**: Finding a motherboard with at least two x16 Gen 5 slots is a goal for one user, as it would improve the potential configurations for new GPU setups. Queries about the power draw on a Corsair 7000x setup are made to gauge how its cooling performs.

- **LM Studio's Applicability at Work**: The discussion touches on LM Studio's terms for use within a work setting, with links shared to clarify permissions and requirements. There's a recognition of the necessity to undergo approval processes in corporate environments before adopting such tools.

- **Multi-GPU Setup Challenges**: Experiences are shared about the difficulties of setting up multiple GPUs using PCIe risers, with oculink cables and extra PSUs highlighted as successful solutions. The conversation details the importance of having all GPUs in the same PCIe generation slots for functionality.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.amazon.com/AMD-3200MHZ-SYSTEM-COMPONENTS-PROCESSORS/dp/B07XP9S55C/ref=sr_1_2">no title found</a>: no description found</li><li><a href="https://www.amazon.de/-/en/HHCJ6-NVIDIA-Server-Accelerator-Renewed/dp/B07GJ45V3D/ref=sr_1_2?crid=1O8IZM1RV0TIH&dib=eyJ2IjoiMSJ9.B2ZUEDxvj_Z73GUX0GJebEDmX0cqUrowZhMOgYhwtCaPdx9UH8NiM39aqowgVAc5YENjqRh8_cc1qHbgwPJMprvhMhnuusRAJuQqLmWDyskupHMP8ACQI354KZZjKYrdtnPPNGnuoJdVlHxoPQ8ll9ilsDZZ334_L6TwueHlrTelgoIjaTt650I3FQyWgOFmpTvAb3YigqPDURnBJMq1D6wanBHjVSaSdFOEnWlP2cUV8J9Hq4Lh_0bJbRh-kAaca58OndCeXm-tGVmNFLi7TuMKGZORpZ0Q6IcMd6Vz11w.MFnlYLfXX9YWUon0J_Dg0ds2eKFM6AwZgazWMdxeEjE&dib_tag=se&keywords=Tesla+K80&qid=1710787582&s=computers&sprefix=tesla+k80%2Ccomputers%2C421&sr=1-2">no title found</a>: no description found</li><li><a href="https://lmstudio.ai/#can-i-use-lm-studio-at-work?">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://coral.ai/products/m2-accelerator-dual-edgetpu#description">M.2 Accelerator with Dual Edge TPU | Coral</a>: Integrate two Edge TPUs into legacy and new systems using an M.2 (E key) interface.</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found</li><li><a href="https://www.aliexpress.com/item/100500634581">404 page</a>: no description found</li><li><a href="https://www.ebay.co.uk/itm/273788651049?">Dell T710 Tower Server Dual 6-CORE X5650 **144Gb RAM**240gb SSD +6X 600G SFF SAS  | eBay</a>: no description found</li><li><a href="https://www.aliexpress.com/item/1005006525215524.html">no title found</a>: no description found</li><li><a href="https://www.ebay.co.uk/itm/115960685949?">AMD EPYC 7F72 CPU PROCESSOR 24 CORE 3.20GHz 192MB CACHE 240W - 100-000000141  | eBay</a>: no description found</li><li><a href="https://www.ebay.de/itm/126352871326?epid=11041255665&itmmeta=01HS9333CQ68S4STA8BZJ3V0BH&hash=item1d6b37cf9e:g:DOEAAOSweRlkuVOG&itmprp=enc%3AAQAJAAAA0GtLL6BuVwKKMH1iyVWS1kdp6p0LvQb%2Fcu8c94aisQZDISgf4yKcfrjNbigVkO4IGdfBt3tcIr6du3Nb1xXGbEe2CNScd%2B4RoCdoEx%2BQMPtNGs0TtY3wzAbszVam1AHN8tC%2Bzq%2BVoVhSwCmdZ77779duZUVHF%2Fq1ckL28OWoVp%2FRStC3u0NyyTZtUke6tEsgNdQYOKI4%2BqNOIN11tc8XuhOtaovFo6WzH87nIC6BUNiaWYnvWcqUPH3NUs6Gxi%2FWnel1Vj9wokxL8oELjbCFBOA%3D%7Ctkp%3ABFBMyLaMo8pj">AMD EPYC 7232P CPU PROCESSOR 8 CORE 3.10GHz 32MB CACHE 120W - 100-000000081  | eBay</a>: no description found</li><li><a href="https://www.ebay.ca/itm/126375063761">AMD EPYC 7232P 8-Core 3.1GHz 32MB L3 Processor - Socket SP3 - 100-000000081  | eBay</a>: no description found</li><li><a href="https://www.newegg.com/asrock-rack-romed8-2t/p/N82E16813140044">Asrock Rack ROMED8-2T ATX Server Motherboard AMD EPYC 7003 (with AMD 3D V-Cache Technology)/7002 series processors SP3 (LGA 4094) Dual 10GbE - Newegg.com</a>: Buy Asrock Rack ROMED8-2T Server Motherboard AMD EPYC 7003 (with AMD 3D V-Cache Technology)/7002 series processors SP3 (LGA 4094) Dual 10GbE with fast shipping and top-rated customer service. Once you...</li><li><a href="https://www.ebay.de/itm/145329120119?epid=507128083&itmmeta=01HS9DKVRXS2WQPFX74KY649GW&hash=item21d6">Nvidia Tesla K80 24GB GPU GDDR5 PCI-E GPU Accelerator 12 Month warranty  | eBay</a>: no description found</li><li><a href="https://www.thingiverse.com/search?q=K80+cooling+&page=1&type=things&sort=relevant">Search Thingiverse - Thingiverse</a>: Download files and build them with your 3D printer, laser cutter, or CNC.</li><li><a href="https://www.ebay.de/itm/125947603377?itmmeta=01HS9HRSJMXBV00M1XW59H5NAE&hash=item1d530fe9b1:g:fHQAAOSwWVxkbefZ&itmprp=enc%3AAQAJAAAA4A6tXSRz7NxXocQqxCeo%2F2TdOTiIP1AMtfRCBxeBISSicEa3bP%2FtSfa9CmVAH74vTwUFyfwFd1VhNC71wMalgSqfYNDwr7svQreF5j3Gqk4Brm8Zn7hMHU6mRQVuxRyyv5VyA1PeZKdylhbJH0O%2BC2IM8GdP7yLRbRw6sOGTb2KMO0V0m%2B7aGkzXe6h33qOgF16cjz2vh2TITEEOr1eYGfz7ViQZ846gljR8VFArZiDwxgIU8naY8yQRPUJe4Znn3GYEn3GT3DNHxdg5zoB7qyMOytwL9TKozBLIkBQVtyyq%7Ctkp%3ABk9SR8KZ47HKYw">New /Wave ®AI Server NF5688M6 NVIDIA HGX TESLA A800 80G octet GPU server/Futures  | eBay</a>: no description found</li><li><a href="https://www.ebay.co.uk/itm/296113403496?">Dell T710 Tower Server Dual 6-CORE X5670 **24 cores**64GB RAM  | eBay</a>: no description found</li><li><a href="https://www.ebay.de/itm/145329120119?epid=507128083&itmmeta=01HS9DKVRXS2WQPFX74KY649GW&hash=item21d64a6377:g:kacAAOSw~q1lFEwb&itmprp=enc%3AAQAJAAAA4GTzwRZBHO82ltgqug5ARkRZ5JKlaikKECFytG5%2FNjvBMzyE2UGOBW0yRbeW%2B%2F3prx2LD9sPaLsinW103607IHMVVMe2tg6FIa2KVc%2FUVWqCGgQPrRRS97i9Q%2FZW0nnLz5XSLuFob%2FicmlhLi7Ve68FV47SLRenj5tDoUD8mwpvdoxA5uQtR0DNACYnvlVQe4BeXKFAWKA8iKA6WdrVikWOsQcODTpcW916%2FL8jFOUSFjg9D5%2FP1xg4foswYBWrIeaD4Pm9rguigAFQvYGqHFLKNXgB4CjCD0BczHhSZYunI%7Ctkp%3ABk9SR8i8z63KYw">Nvidia Tesla K80 24GB GPU GDDR5 PCI-E GPU Accelerator 12 Month warranty  | eBay</a>: no description found</li><li><a href="https://zifa666.aliexpress.com/store/5885523/pages/all-items.html?productGroupId=40000003590095&shop_sortType=bestmatch_sort">Luckim Official Store - Amazing products with exclusive discounts on AliExpress</a>: no description found</li><li><a href="https://www.techpowerup.com/cpu-specs/core-i5-3470.c1039#:~:text=Programs%20using%20Advanced%20Vector%20Extensions,performance%20for%20calculation%2Dheavy%20applications.">Intel Core i5-3470 Specs</a>: Ivy Bridge, 4 Cores, 4 Threads, 3.2 GHz, 77 W</li><li><a href="https://www.microcenter.com/product/677156/nvidia-geforce-rtx-3090-founders-edition-dual-fan-24gb-gddr6x-pcie-40-graphics-card-(refurbished)">Micro Center - Computers and Electronics</a>: Micro Center - Computers and Electronics - Thousands of products to buy: desktops, laptops, monitors, build your own PC parts, upgrades, digital imaging, printing supplies, portable devices, audio equ...</li><li><a href="https://www.aliexpress.com/item/1005006345813657.html">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1219065221327355974)** (4 messages): 

- **Seeking Presets for Different Models**: A user inquired about a comprehensive list of presets for different models. The response provided a [GitHub link](https://github.com/lmstudio-ai/configs) with JSON configuration files and a collection of example config files for LM Studio.

- **Looking for ROCm Peers**: A user asked whether there are any ROCm users present in the chat. Another user directed them to a specific channel with the code `#1195858490338594866` for a potentially helpful discussion.

**Link mentioned**: <a href="https://github.com/lmstudio-ai/configs">GitHub - lmstudio-ai/configs: LM Studio JSON configuration file format and a collection of example config files.</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs

  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1219051718172606537)** (1 messages): 

- **Inquiry on Local Inference Server Capabilities**: A member inquired if anyone has successfully integrated a model with JSON function calling into the **Local Inference Server**. No further details or follow-up were provided.
  

---


**LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1219383598193311744)** (5 messages): 

- **AVX Beta Clarification**: A member inquired if the beta app uses AVX instructions, suggesting its beta status is due to AVX usage.
- **Beta App Details Revealed**: It was confirmed that the beta app is an **older version** and that AVX support isn't a high priority for the team.
- **Model Compatibility Questions**: A member asked whether the models work like the newer ones in the beta app and it was clarified that while models will work, the newest models like **starcoder2, gemma** etc., are not supported.
- **Mistral Model on Beta**: Upon asking, a member was informed that they can run the **Mistral** model on the beta app.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1218206050495234070)** (5 messages): 

- **Pre-built ROCm Libraries on Github**: A member shared a [GitHub link](https://github.com/brknsoul/ROCmLibs) to pre-built Windows ROCm libraries that support gfx1031 and gfx1032. The link refers to a repository containing libraries intended to assist those working with specific AMD GPUs.
- **No Dual GPU Support Yet for LM Studio**: A member inquired about using an AMD GPU (6700 xt) with their 7800 xt in LM Studio, noting that the software seems to only utilize the primary GPU currently. They sought to confirm whether support for multiple GPUs would be coming soon.
- **AMD GPU 6700 xt Unsupported by ROCm**: Another member clarified that the AMD GPU 6700 xt is not officially supported by ROCm, which is why it wouldn't work in LM Studio as the latter uses the ROCm libraries.
- **Parallel Use of 7000 Series AMD GPUs in LM Studio**: Following the clarification about 6700 xt's support, the same member speculated that LM Studio would likely utilize two 7000 series GPUs in parallel if they were available.

**Link mentioned**: <a href="https://github.com/brknsoul/ROCmLibs">GitHub - brknsoul/ROCmLibs: Prebuild Windows ROCM Libs for gfx1031 and gfx1032</a>: Prebuild Windows ROCM Libs for gfx1031 and gfx1032 - brknsoul/ROCmLibs

  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1219265025487667200)** (1 messages): 

- **Agent System Selection Process**: A member inquired about progress in choosing an **agent system** for the purpose of validating a creative concept with different agents. They reached out specifically to another member for an update on their decision-making process.
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1218144997094723615)** (56 messages🔥🔥): 

- **GDDR7 Memory Speed Insights for NVIDIA's RTX 50-Series**: An article shared describes NVIDIA's plan to equip the GeForce RTX 50-series "Blackwell" graphics cards with GDDR7 memory at 28 Gbps speed, despite the availability of faster 32 Gbps chips. The article speculates on NVIDIA's strategy based on historical precedents and potential memory bus widths.
  
- **Anticipating Advances in AI Interfaces**: Members discuss the potential of upcoming AI models to improve agent interfaces significantly, suggesting that future advancements will likely combine new model development with agent-focussed customizations.
  
- **Game Data Open for AI Development**: [MatchboxDAO](https://x.com/unkjdgames?s=21) announces a game that has opened its data to developers for creating AI agents, with funding support available for interested community contributors.
  
- **Predicting the Future of AI's Role in Society**: A recalled prediction from Sam Altman speculates on AI's evolving capabilities, ranging from legal and medical applications to assembly-line tasks, and eventually towards robotic companionship.
  
- **Community Discusses Interactive AI Agents**: A dialogue unfolded around seeking solutions to make AI assistants more responsive within conversations, pausing intelligently when interrupted, and resuming after the user's interjection.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/unkjdgames?s=21">Tweet from undefined</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=ZlJbaYQ2hm4">Plan-and-Execute using Langgraph</a>: how to create a &quot;plan-and-execute&quot; style agent. This is heavily inspired by the Plan-and-Solve paper as well as the Baby-AGI project.The core idea is to firs...</li><li><a href="https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed">NVIDIA GeForce RTX 50-series &quot;Blackwell&quot; to use 28 Gbps GDDR7 Memory Speed</a>: The first round of NVIDIA GeForce RTX 50-series &quot;Blackwell&quot; graphics cards that implement GDDR7 memory are rumored to come with a memory speed of 28 Gbps, according to kopite7kimi, a reliabl...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1218108265854926899)** (16 messages🔥): 

- **Mermaids Drawn by "Horny Claudes"**: The [Repligate Twitter post](https://x.com/repligate/status/1768521441329434937?s=20) mentioned the creation of a network of 'horny Claudes', which supposedly produce better mermaid diagrams, suggesting that the models' state could influence the quality of generated diagrams. The comments indicated both shock and humor regarding the concept.

- **Apple Drops AI Model Information**: [Apple discussed the details of their AI models](https://twitter.com/arankomatsuzaki/status/1768446729710371115), sparking conversations about the recent sharing of AI model information from proprietary sources. The discussion included disappointment over the lack of released model weights.

- **Leading Edge in AI Alignment**: An [abstract on Hugging Face](https://huggingface.co/papers/2403.07691) explores a new algorithm called ORPO for preference-aligned supervised fine-tuning of language models, which is said to eliminate the additional phase of preference alignment, showing promise across models of varying sizes.

- **Reproducing MetaAI's Self-Rewarding Language Model**: An attempt to reproduce the Self-Rewarding Language Model paper by MetaAI was made by the [Oxen.ai Community](https://github.com/Oxen-AI/Self-Rewarding-Language-Models), contributing to replicating research findings within the open-source community.

- **Unifying LLM Agents into Computational Graphs**: A [research paper](https://arxiv.org/abs/2402.16823) introduced a new framework that treats large language model-based agents as computational graphs, which can be automatically optimized, leading to more efficient problem-solving architectures. The community responded with enthusiasm, appreciating the approach to unify disparate LLM functionalities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/burny_tech/status/1769530798242255129">Tweet from Burny — Effective Omni (@burny_tech)</a>: My thoughts on Musk destabilizing other gigantic players in the intelligence wars by possibly leading open source using Grok   Grok 1 is a 314B parameter model and it&#39;s a mixture of experts archit...</li><li><a href="https://arxiv.org/abs/2402.16823">Language Agents as Optimizable Graphs</a>: Various human-designed prompt engineering techniques have been proposed to improve problem solvers based on Large Language Models (LLMs), yielding many disparate code bases. We unify these approaches ...</li><li><a href="https://huggingface.co/papers/2403.07691">Paper page - ORPO: Monolithic Preference Optimization without Reference Model</a>: no description found</li><li><a href="https://x.com/repligate/status/1768521441329434937?s=20">Tweet from j⧉nus (@repligate)</a>: @xlr8harder I didn&#39;t let it go very far but there&#39;s someone in the room with me right now talking about how theyve created a network of &#34;horny claudes&#34; and how the claudes create bette...</li><li><a href="https://github.com/Oxen-AI/Self-Rewarding-Language-Models">GitHub - Oxen-AI/Self-Rewarding-Language-Models: This is work done by the Oxen.ai Community, trying to reproduce the Self-Rewarding Language Model paper from MetaAI.</a>: This is work done by the Oxen.ai Community, trying to reproduce the Self-Rewarding Language Model paper from MetaAI. - Oxen-AI/Self-Rewarding-Language-Models
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1218105907615895562)** (656 messages🔥🔥🔥): 

- **Grok Unleashed**: A new 314-billion parameter MoE model named Grok-1 has been released by xAI. It's criticized for barely outperforming GPT-3.5 and is considered too large for practical use without further pretraining.
- **Grok's Commercial Use in Question**: Some suspicion exists as to whether the Yi-9B model can truly be utilized for commercial purposes and if the permission process is just marketing.
- **Continual Pretraining Challenges**: Discussions center around the feasibility and methods of continually pretraining models, particularly MoEs like Mixtral, and whether it leads to improved performance without domain-specific data.
- **GPT-4 Confirmation Rumor**: NVIDIA CEO Jensen Huang's GTC keynote mentioned an architecture with 1.8 trillion parameters, rumored to be GPT-4. The mention includes the MoE configuration not officially confirmed by OpenAI.
- **Recommended Reads**: Several users have shared links to recent papers on various AI topics, including multimodal models from Apple, continual learning, and memory akin to biological neural networks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/lqiao/status/1768045066776707226?s=20">Tweet from Lin Qiao (@lqiao)</a>: We are thrilled to collaborate on Hermes 2 Pro multi-turn chat and function calling model with @NousResearch. Finetuned on over 15k function calls, and a 500 example function calling DPO datasets, Her...</li><li><a href="https://x.com/grok/status/1769441648910479423?s=46">Tweet from Grok (@grok)</a>: @elonmusk @xai ░W░E░I░G░H░T░S░I░N░B░I░O░</li><li><a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered">anon8231489123/ShareGPT_Vicuna_unfiltered · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/whyarethis/status/1769269824587542692?s=46">Tweet from Parzival - 🌞/⏫ (@whyarethis)</a>: Now we are going somewhere.</li><li><a href="https://huggingface.co/datas">datas (shu nakamura)</a>: no description found</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1769773746896662873?s=20">Tweet from interstellarninja (@intrstllrninja)</a>: @Cyndesama claude 3 opus runs ai town simulation with python42</li><li><a href="https://huggingface.co/migtissera/Tess-70B-v1.6">migtissera/Tess-70B-v1.6 · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.08763">Simple and Scalable Strategies to Continually Pre-train Large Language Models</a>: Large language models (LLMs) are routinely pre-trained on billions of tokens, only to start the process over again once new data becomes available. A much more efficient solution is to continually pre...</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1768948484479049897?s=20">Tweet from interstellarninja (@intrstllrninja)</a>: &lt;cmd&gt; run world_sim.exe --epoch &#34;Earth in 2500&#34; --civilization_type &#34;Type-II on Kardashev scale&#34; &lt;/cmd&gt;  ↘️ Quoting mephisto (@karan4d)   im opensourcing worldsim of course...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: In this work, we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and c...</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1768942321129697790?s=20">Tweet from interstellarninja (@intrstllrninja)</a>: Hermes 2 Pro function-calling model integrated with search engine by @ExaAILabs👀  ↘️ Quoting Barton Rhodes 🦺 (@bmorphism)   added @ExaAILabs support for use with @NousResearch new function-calling m...</li><li><a href="https://huggingface.co/Replete-AI/Mistral-11b-v0.1">Replete-AI/Mistral-Evolved-11b-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...</li><li><a href="https://x.com/itsandrewgao/status/1769460684956602527?s=46">Tweet from Andrew Kean Gao (@itsandrewgao)</a>: i think grok-4bit is just barely too big for an H100 GPU :(  ↘️ Quoting Andrew Kean Gao (@itsandrewgao)   HOLY SH*T @grok IS 314 BILLION PARAMETERS  Mixture of 8 Experts, not RLHFd/moralized  THIS IS ...</li><li><a href="https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO/discussions/10/files">NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · Adding Evaluation Results</a>: no description found</li><li><a href="https://x.com/aravsrinivas/status/1769485603622867394?s=46&t=TOasxww3M5DjlB4iBWa_ig">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Yep, thanks to @elonmusk and xAI team for open-sourcing the base model for Grok. We will fine-tune it for conversational search and optimize the inference, and bring it up for all Pro users!  ↘️ Quoti...</li><li><a href="https://arxiv.org/abs/2403.08540">Language models scale reliably with over-training and on downstream tasks</a>: Scaling laws are useful guides for developing language models, but there are still gaps between current scaling studies and how language models are ultimately trained and evaluated. For instance, scal...</li><li><a href="https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset/tree/main">openchat/openchat_sharegpt4_dataset at main</a>: no description found</li><li><a href="https://arxiv.org/abs/2303.11934">Sparse Distributed Memory is a Continual Learner</a>: Continual learning is a problem for artificial neural networks that their biological counterparts are adept at solving. Building on work using Sparse Distributed Memory (SDM) to connect a core neural ...</li><li><a href="https://x.com/burkov/status/1769496949252673550?s=46&t=TOasxww3M5DjlB4iBWa_ig">Tweet from Andriy Burkov (@burkov)</a>: We are yet to see how good Grok is compared to GPT-4, but what we can tell for sure is that if you are to train a competitor to OpenAI/Anthropic today, you would not need to start from scratch anymore...</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1769424961192529962?s=20">Tweet from interstellarninja (@intrstllrninja)</a>: &lt;cmd&gt; sudo python3 akashic_records.py --entity [&#34;sam altman&#34;, &#34;elon musk&#34;] --mode &#34;email thread&#34; --topic &#34;superintelligence scenarios&#34; &lt;/cmd&gt;</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/causality.ipynb">Abstractions/abstractions/goap/causality.ipynb at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/gridmap.ipynb">Abstractions/abstractions/goap/gridmap.ipynb at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/system_prompt.md">Abstractions/abstractions/goap/system_prompt.md at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.</li><li><a href="https://docs.pydantic.dev/latest/concepts/json_schema/">JSON Schema - Pydantic</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=oYFjDt4-hFw&ab_channel=NewEconomicThinking">Cosma Shalizi - Why Economics Needs Data Mining</a>: Cosma Shalizi urges economists to stop doing what they are doing: Fitting large complex models to a small set of highly correlated time series data. Once you...</li><li><a href="https://huggingface.co/01-ai/Yi-9B">01-ai/Yi-9B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/01-ai/Yi-9B-200K">01-ai/Yi-9B-200K · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=t6SQj8YidGA">Accelerationism Accelerationism (Acc/Acc)</a>: Accelerationism accelerationism is when you accelerate accelerationism to apply accelerationism to accelerationismparts that were too edgy: https://www.patre...</li><li><a href="https://www.hd-computing.com/">HD/VSA</a>:   </li><li><a href="https://www.youtube.com/watch?v=Y2F8yisiS6E">GTC March 2024 Keynote with NVIDIA CEO Jensen Huang</a>: Watch NVIDIA CEO Jensen Huang’s GTC keynote to catch all the announcements on AI advances that are shaping our future.Dive into the announcements and discove...</li><li><a href="https://www.youtube.com/watch?v=zduSFxRajkE">Let&#39;s build the GPT Tokenizer</a>: The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizer...</li><li><a href="https://www.youtube.com/wa">Liam Johnson DESTROYS Heckler | New York Stand-up</a>: Last weekend Liam Johnson decided to finally make his first appearance here at Giggle Nerd. He performed on Sunday from 23:00 to 23:25 and our audience loved...</li><li><a href="https://github.com/PrismarineJS/mineflayer">GitHub - PrismarineJS/mineflayer: Create Minecraft bots with a powerful, stable, and high level JavaScript API.</a>: Create Minecraft bots with a powerful, stable, and high level JavaScript API. - PrismarineJS/mineflayer</li><li><a href="https://github.com/Prismarin">Prismarin - Overview</a>: Prismarin has 3 repositories available. Follow their code on GitHub.</li><li><a href="https://hack.meetmeinshibuya.com/">HacksTokyo</a>: AI x Digital Entertainment Hackathon in Tokyo!</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.03.11.584515v1">Whole-body simulation of realistic fruit fly locomotion with deep reinforcement learning</a>: The body of an animal determines how the nervous system produces behavior. Therefore, detailed modeling of the neural control of sensorimotor behavior requires a detailed model of the body. Here we co...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1218205298729156648)** (25 messages🔥): 

- **Perplexed by Perplexity**: A member tried to calculate the perplexity for **NousResearch/Llama-2-7b-chat-hf** based on a [Kaggle notebook guide](https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook) but ended up with an unexpected perplexity value of 90.3.
- **Dreaming of a 20b Model**: There's a desire to see a **20b base model** that rivals *Mistral*. While the conversation suggested a need for significant funding, there was also talk of potential strategies, such as upscaling or merging with other models.
- **Scaling Down is the New Scaling Up?**: One member shared their experience working on [downscaling models](https://huggingface.co/AlexWortega/smallstral) with continuous pretraining, demonstrating how a layered pruned Mistral variant, **Smallstral**, performs on various tasks.
- **Expanding Model Capabilities**: There was a query about using multiple parallel linear layers for classification purposes in transformer models, aiming to group vocabulary based on linguistic features.
- **Fine-Tuning Frontiers**: The discussion touched on fine-tuning possibilities with high-performance compute resources, and one member excitedly teased an upcoming **Mixtral** model, which shows promising improvement over *qloras*.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook">Calculating the Perplexity of 4-bit Llama 2</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from multiple data sources</li><li><a href="https://huggingface.co/AlexWortega/smallstral">AlexWortega/smallstral · Hugging Face</a>: no description found</li><li><a href="https://wandb.ai/alexwortega/cpm_rus/runs/w5t4dsat?nw=nwuseralexwortega">alexwortega</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1218181932853104720)** (18 messages🔥): 

- **Link Troubleshooting 1-on-1**: A user questioned if a link was broken, to which another replied with a simple "No."
- **Awestruck by an Idea**: User **fullstack6209** expressed being in awe for several days over an unspecified idea, which led to another user seeking clarification about what was meant.
- **Bittensor Chain Issues Reported**: **jubilant_dragon_18246** noted there have been issues with the Bittensor chain for the past 11 hours and **teknium** humorously agreed that it appeared broken.
- **Bittensor Chain Path to Recovery**: It was reported that the Bittensor chain was back up but required an update to **subtensor** that not all users had completed.
- **TAO Acquiring Adventure**: User **ee.dd** inquired about the best place to purchase TAO to register and was advised to use the MEXC exchange, leading to an unsuccessful withdrawal attempt on Kucoin. Additionally, discussions about GPU requirements indicated a single 3090 was sufficient if setting up a qlora trainer, while 80GB or 48GB (for g1) might be needed otherwise.
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1218682432610373703)** (100 messages🔥🔥): 

- **Evolving RAG Capabilities**: Members discussed potential features and improvements to enhance RAG models, mentioning properties such as response modes that switch from verbose to structured output, citation and span highlighting, and the ability to understand intent and decomposition. High recall and relevance ranking were also mentioned, but it was noted that some LLMs experience challenges in reasoning with long external contexts.
- **RAG Model Context and Functionality**: There was debate over how a RAG model should balance using provided external context and its own knowledge, with suggestions for "modes" allowing the model to focus solely on external sources or to extrapolate with internal knowledge when prompted. The idea of training models to be able to call functions and break down complex extraction tasks was also floated.
- **Output Formatting for RAG Responses**: There is a consensus that while markdown might not need to be the default output format, outputs should incorporate structured elements like lists, tables, and code, and maintain good practices in citation. The conversation included mentioning the utility of [Cohere's model](https://cohere.ai/docs) which includes inline citations in its responses.
- **Potential Uses for Specialized Smaller Models in RAG Pipelines**: A proposition was made to train specialized, smaller models to enhance RAG pipeline efficiency, such as a dedicated "relevant info extractor" model. A concern was expressed that larger models might not be as optimal for real-time RAG operations due to latency issues.
- **Sharing RAG-Related Resources and Experiences**: Members shared links to external resources, like a [Github implementation of command R for RAG](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py), and briefly discussed their personal projects and contributions to the RAG ecosystem.

**Link mentioned**: <a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py">scratchTHOUGHTS/commanDUH.py at main · EveryOneIsGross/scratchTHOUGHTS</a>: 2nd brain scratchmemory to avoid overrun errors with self. - EveryOneIsGross/scratchTHOUGHTS

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1218167767379742813)** (273 messages🔥🔥): 

- **Grok-1 AI Model Discussion**: Members are evaluating the performance and training data size of [Grok-1](https://github.com/xai-org/grok-1), comparing it to other models like Mixtral and Claude 2. Questions were raised about whether the Twitter chatbot interface is optimized for actual use, and there is anticipation for independent benchmarks.

- **Suggestions for LLM Evaluation Data**: The community discussed the feasibility of using various sources such as NPR transcripts and Wikipedia for creating benchmarks to evaluate LLMs. Concerns were raised about potential copyright issues and the desire to avoid legal entanglements.

- **RAG Implementation Resources Sought**: One user inquired about the best tutorials or implementations for Retrieval-Augmented Generation (RAG), indicating a need for accessible educational materials on the topic.

- **PyTorch Bug Alert for Mac Users**: A member raised an issue regarding a bug in PyTorch that may affect matrix multiplication on Macs, which can cause incorrect results and performance issues, providing a [GitHub issue link](https://github.com/pytorch/pytorch/issues/122123) for reference.

- **Conferences and Journal Submissions**: A user sought advice on cost-effective options for submitting research papers, with TMLR mentioned as a free journal option, while conference submissions like ICLR and AISTATS were discussed for future consideration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok">Announcing Grok</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is imp...</li><li><a href="https://x.com/maisaAI_/status/1768657114669429103?s=20">Tweet from Maisa (@maisaAI_)</a>: Introducing Maisa KPU: The next leap in AI reasoning capabilities.  The Knowledge Processing Unit is a Reasoning System for LLMs that leverages all their reasoning power and overcomes their intrinsic ...</li><li><a href="https://arxiv.org/abs/2203.07852">Block-Recurrent Transformers</a>: We introduce the Block-Recurrent Transformer, which applies a transformer layer in a recurrent fashion along a sequence, and has linear complexity with respect to sequence length. Our recurrent cell o...</li><li><a href="https://tenor.com/view/excited-fuego-gif-26833875">Excited Fuego GIF - Excited Fuego - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2312.12705">Optimizing Distributed Training on Frontier for Large Language Models</a>: Large language models (LLMs) have demonstrated remarkable success as foundational models, benefiting various downstream applications through fine-tuning. Recent studies on loss scaling have demonstrat...</li><li><a href="https://en.wikipedia.org/wiki/Wikipedia:Database_reports/Most_edited_articles_last_month">Wikipedia:Database reports/Most edited articles last month - Wikipedia</a>: no description found</li><li><a href="https://arxiv.org/abs/2002.09402">Addressing Some Limitations of Transformers with Feedback Memory</a>: Transformers have been successfully applied to sequential, auto-regressive tasks despite being feedforward networks. Unlike recurrent neural networks, Transformers use attention to capture temporal re...</li><li><a href="https://www.npr.org/sections/publiceditor/2009/08/19/112034424/free-transcripts-now-available-on-npr-org>">Free Transcripts now Available on NPR.org</a>: Transcripts of favorite, missed or maddening stories on NPR used to cost $3.95 each, but now they are free on NPR.org.</li><li><a href="https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py">cookbook/calc/calc_transformer_flops.py at main · EleutherAI/cookbook</a>: Deep learning for dummies. All the practical details and useful utilities that go into working with real models. - EleutherAI/cookbook</li><li><a href="https://aideadlin.es/?sub=ML,CG,NLP,RO,SP,DM,CV">AI Conference Deadlines</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/issues/122123)">Issues · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - Issues · pytorch/pytorch</li><li><a href="https://github.com/trevorpogue/algebraic-nnhw">GitHub - trevorpogue/algebraic-nnhw: AI acceleration using matrix multiplication with half the multiplications</a>: AI acceleration using matrix multiplication with half the multiplications - trevorpogue/algebraic-nnhw</li><li><a href="https://www.youtube.com/watch?v=Sq1QZB5baNw),">Figure Status Update - OpenAI Speech-to-Speech Reasoning</a>: no description found</li><li><a href="https://maisa.ai/blog/kpu">KPU - Maisa</a>: AI-Powered Knowledge Processing Platform. A simple API for executing business tasks. Abstracting the complexities of using the latest AI architectures for software and app developers</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://www.cs.cmu.edu/~dwoodruf/">David P. Woodruff</a>: no description found</li><li><a href="https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/">RT-2: New model translates vision and language into action</a>: Introducing Robotic Transformer 2 (RT-2), a novel vision-language-action (VLA) model that learns from both web and robotics data, and translates this knowledge into generalised instructions for...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1218100666493304852)** (245 messages🔥🔥): 

- **Speculative Sampling Debate for Mamba Models**: A discussion revealed skepticism regarding *speculative decoding* with models like Mamba, noting they don't operate in a way that benefits from speculative sampling as Transformers do. Despite being faster than typical series generation, they aren't parallel in nature and verification still requires considerable computation, making speculative sampling potentially ineffective.

- **Grok Model Size and Performance Scrutinized**: Members exchanged thoughts on whether having a world-class team could circumvent poor outcomes with large language models, debating grok's potential performance issues. The community highlighted that Grok's comparatively larger size doesn't necessarily guarantee superior performance to existing models like Mixtral or MiQ.

- **Efficiency and Scaling of LLMs**: Efficiency and scaling strategies for Large Language Models (LLMs) were mulled over, including the use of different GPU types and configurations. Discourse emphasized the potential pros and cons of speculative sampling techniques, and the complexity of scaling deep models like DeepScaleLM, which proposes improvements for traditional Transformer models.

- **Debating the Quality of Grok versus Other Models**: Grok's possible advantages due to its integration as a feature in Twitter were discussed, despite lacking a broad usage or an accessible API. Skepticism remained about the quality and effectiveness of the model, pending independent benchmarks and fine-tuning comparisons.

- **Training Specifications and Impact on Model Quality**: Conversations touched upon the importance of training specifications, such as the amount and type of data used. It was suggested that companies like XAi likely based training cessation on internal benchmark saturation, with a particular focus on real-time applications and events on Twitter.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok">Announcing Grok</a>: no description found</li><li><a href="https://x.com/Aaditya6284/status/1762558439354409345">Tweet from Aaditya Singh (@Aaditya6284)</a>: We study the effect of this choice in GPT-3.5 and GPT-4 – specifically, we look at the effect of tokenizing left-to-right (L2R) vs right-to-left (R2L), enforced by using delimiters such as commas. We ...</li><li><a href="https://arxiv.org/abs/2401.16380">Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling</a>: Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance o...</li><li><a href="https://arxiv.org/abs/2402.18510">RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval</a>: This paper investigates the gap in representation powers of Recurrent Neural Networks (RNNs) and Transformers in the context of solving algorithmic problems. We focus on understanding whether RNNs, kn...</li><li><a href="https://arxiv.org/abs/2403.09539">Logits of API-Protected LLMs Leak Proprietary Information</a>: The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption...</li><li><a href="https://pytorch.org/blog/accelerating-generative-ai-2/">Accelerating Generative AI with PyTorch II: GPT, Fast</a>: This post is the second part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance...</li><li><a href="https://arxiv.org/abs/2403.04706">Common 7B Language Models Already Possess Strong Math Capabilities</a>: Mathematical capabilities were previously believed to emerge in common language models only at a very large scale or require extensive math-related pre-training. This paper shows that the LLaMA-2 7B m...</li><li><a href="https://arxiv.org/abs/2403.09635">Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models</a>: In spite of their huge success, transformer models remain difficult to scale in depth. In this work, we develop a unified signal propagation theory and provide formulae that govern the moments of the ...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: In this work, we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and c...</li><li><a href="https://arxiv.org/abs/2403.06963">The pitfalls of next-token prediction</a>: Can a mere next-token predictor faithfully model human intelligence? We crystallize this intuitive concern, which is fragmented in the literature. As a starting point, we argue that the two often-conf...</li><li><a href="https://arxiv.org/abs/2403.09394">GiT: Towards Generalist Vision Transformer through Universal Language Interface</a>: This paper proposes a simple, yet effective framework, called GiT, simultaneously applicable for various vision tasks only with a vanilla ViT. Motivated by the universality of the Multi-layer Transfor...</li><li><a href="https://arxiv.org/abs/2403.06504">Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU</a>: Recent advances in large language models have brought immense value to the world, with their superior capabilities stemming from the massive number of parameters they utilize. However, even the GPUs w...</li><li><a href="https://arxiv.org/abs/2402.00691">Comparative Study of Large Language Model Architectures on Frontier</a>: Large language models (LLMs) have garnered significant attention in both the AI community and beyond. Among these, the Generative Pre-trained Transformer (GPT) has emerged as the dominant architecture...</li><li><a href="https://arxiv.org/abs/2403.10430">Construction of Arithmetic Teichmuller Spaces IV: Proof of the abc-conjecture</a>: This is a continuation of my work on Arithmetic Teichmuller Spaces developed in the present series of papers. In this paper, I show that the Theory of Arithmetic Teichmuller Spaces leads, using Shinic...</li><li><a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://github.com/xai-org/grok">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://github.com/enfiskutensykkel/ssd-gpu-dma">GitHub - enfiskutensykkel/ssd-gpu-dma: Build userspace NVMe drivers and storage applications with CUDA support</a>: Build userspace NVMe drivers and storage applications with CUDA support - enfiskutensykkel/ssd-gpu-dma</li><li><a href="https://github.com/bigscience-workshop/bloom-dechonk">GitHub - bigscience-workshop/bloom-dechonk: A repo for running model shrinking experiments</a>: A repo for running model shrinking experiments. Contribute to bigscience-workshop/bloom-dechonk development by creating an account on GitHub.</li><li><a href="https://arxiv.org/abs/2403.07183">Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>: We present an approach for estimating the fraction of text in a large corpus which is likely to be substantially modified or produced by a large language model (LLM). Our maximum likelihood model leve...</li><li><a href="https://bytez.com/read/arxiv/2403.07183">Bytez: Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>: This study examines the use of large language models (LLMs), like ChatGPT, in scientific peer review. The authors developed a method to estimate the percentage of text in peer reviews that is generate...</li><li><a href="https://artificialanalysis.ai/">Model &amp; API Providers Analysis | Artificial Analysis</a>: Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key metrics including quality, price, performance and speed (throughput &amp; latency).
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1218832533517766666)** (11 messages🔥): 

- **Data complexity impacts scaling laws**: The sensitivity of language model scaling laws to data complexity was highlighted, with syntactic properties of a Probabilistic Context-Free Grammar (PCFG) and gzip compression being effective predictors of dataset-specific scaling properties.
- **Awaiting comprehensive experiments**: Further, more comprehensive experiments are underway to fit scaling laws and provide hard numbers, with anticipation to use a particular user's package for assistance in the analysis.
- **Complexity and downstream tasks**: The relationship between model perplexity and data complexity, as well as potential impacts on downstream tasks, prompted discussion around how such complexity might be aligned with task specificity and leveraged for data cleaning and efficient pretraining.
- **Syntactic specifications as dataset labels**: In response to an inquiry about dataset labeling, it's explained that the additional labels represent syntactic specifications derived from the PCFG used to generate the dataset, including metrics like the number of nonterminals and terminals.
- **Perplexity measures and information density**: A clarification was made that perplexity and loss are effectively the same, with the focus on using compression measures such as gzip to potentially find optimal ranges of lexical densities for efficient pretraining.
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1218288738728284241)** (13 messages🔥): 

- **Query on Sampling Strings from a Specified Distribution**: A member inquired if there is a canonical way to sample strings from a pre-specified set of 1-gram, 2-gram, ..., n-gram statistics on a vocabulary.
  
- **Constraint Hierarchy in Gram Statistics**: It was clarified that specifying n-gram statistics also determines the statistics of all lower-order grams, albeit with some minor considerations for beginning-of-sentence (BOS) and end-of-sentence (EOS) tokens.

- **Autoregressive Sampling Explained**: Autoregressive sampling is the method to use for drawing samples from a distribution matching specified n-gram statistics. This approach starts with unigram distribution, then proceeds with conditional bigram distribution, etc., thereby creating the max entropy distribution that corresponds to those specified statistics.

- **N-gram Language Models Background**: The discussion included a reference to the [Wikipedia entry](https://en.wikipedia.org/wiki/Word_n-gram_language_model) on word n-gram language models, highlighting their historical context and their replacement by more advanced models like recurrent neural networks and large language models.

- **Practical Implementation of Sampling from Bigram Distributions**: A GitHub Python script for generating bigrams, which is part of the EleutherAI project on analyzing feature evolution during neural network training, was shared as an example. The script can be found at [features-across-time/scripts/generate_bigrams.py](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py">features-across-time/scripts/generate_bigrams.py at main · EleutherAI/features-across-time</a>: Understanding how features learned by neural networks evolve throughout training - EleutherAI/features-across-time</li><li><a href="https://en.wikipedia.org/wiki/Word_n-gram_language_model">Word n-gram language model - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1218143473916575765)** (31 messages🔥): 

- **Integration of LLMs with lm-eval-harness**: A user inquires how to implement functions like `generate_until` and `log_likelihood` for a LLM model, specifically on *megatron deepspeed* for llama on gaudi2. Reference implementations in the `models` directory are mentioned, with the need for demos and clarification on inheritance and argument structure. However, no specific solutions or demo codes are provided.

- **Model Incorrectly Defaults to GPT-2-Small**: The issue of specifying a model in lm-eval-harness and having it default to `gpt-2-small` instead of the specified model, such as *Mixtral*, is raised. The user identifies the cause as the specification of `model_args` twice in their command, with the first instance being ignored.

- **Inconsistency in Reported MMLU Scores**: A discrepancy between the MMLU score reported for *llama2-70b* on the openllm leaderboard (69%) and scores received by users (62-64%) is discussed. Clarification is provided that the leaderboard's averaging method differs by not weighting subtask sizes.

- **Potential Deadlock Issue in lm-evaluation-harness**: A GitHub issue regarding a `wmt14-en-fr` evaluation deadlock is shared ([#1485](https://github.com/EleutherAI/lm-evaluation-harness/issues/1485)). Suggestions involve avoiding concurrent processes on the same filesystem, and looking at code associated with multiprocessing for possible solutions.

- **LM Harness Model Cache Directories**: Questions about the location of downloaded models for `lm-eval` lead to clarifications: models are typically stored in the Hugging Face cache directory, which can be configured with environmental variables such as `HF_HOME`, `TRANSFORMERS_CACHE`, and `HF_DATASETS_CACHE`.

- **New Release of lm-eval-harness**: The new version 0.4.2 of lm-eval has been released and is available on [PyPI](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2). The announcement invites more contributors and promises reviews for pending pull requests.

- **Translations in LM Evaluation Harness**: The topic of including machine-translated evaluations such as those for *arc_challenge* or MMLU in lm-eval-harness is discussed. A potential approach involves organizing such tasks under a specific directory and indicating their translated nature in their names.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/perplexity">Perplexity of fixed-length models</a>: no description found</li><li><a href="https://github.com/">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md">lm-evaluation-harness/docs/model_guide.md at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1485">`wmt14-en-fr` deadlock issue · Issue #1485 · EleutherAI/lm-evaluation-harness</a>: While running evaluation on this task, during ter metric computation, the program gets stuck forever. The command: lm_eval --model hf --model_args pretrained=microsoft/phi-2,trust_remote_code=True ...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2">Release v0.4.2 · EleutherAI/lm-evaluation-harness</a>: lm-eval v0.4.2 Release Notes We are releasing a new minor version of lm-eval for PyPI users! We&#39;ve been very happy to see continued usage of the lm-evaluation-harness, including as a standard test...</li><li><a href="https://github.com/huggingface/evaluate/blob/8dfe05784099fb9af55b8e77793205a3b7c86465/metrics/perplexity/perplexity.py">evaluate/metrics/perplexity/perplexity.py at 8dfe05784099fb9af55b8e77793205a3b7c86465 · huggingface/evaluate</a>: 🤗 Evaluate: A library for easily evaluating machine learning models and datasets. - huggingface/evaluate
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1219336845310038047)** (3 messages): 

- **Clarification on The Pile Data Shuffling**: A member inquired about whether The Pile data for pretraining was pre-shuffled, with a subsequent clarification explaining that the original files were not shuffled, while the preprocessed and pretokenized data on Hugging Face are ready-to-go. They noted that it is the same data used by Pythia.
- **Pile Parts Unshuffled but Train/Test/Val Might Be**: Another member added that the individual components of the Pile are not shuffled, in part because some are organized by date, but there’s an expectation that the original train/test/validation split should be shuffled to ensure a good mix across the various datasets.
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1218173412522852483)** (193 messages🔥🔥): 

- **Diving into ChatGPT's Understanding**: A discussion pondered if AI truly "understands" language, considering emergent behaviors from sophisticated next-word predictions and the impact of human training on AI performance. They debated the nature of AI "consciousness" comparing physical to abstract experiences, with sentiments that genuine human training creates models capable of conversational interactions superior to some humans.

- **Image Generation Excellence**: Users express awe at **DALL-E 3**'s ability to follow detailed prompts accurately, labeling it "awesome" and appreciating its advancements over its predecessors. They contrast their experiences with Microsoft Copilot and discuss the pros and cons of different image generation tools, touching on issues like speed and image-saving, with some preferring **ChatGPT+** because of its underlying **DALL-E 3** and **GPT-4** capabilities.

- **Debating AI Models**: A conversation unfolded comparing **GPT-4** with **Claude**, as users shared their experiences using both models for various tasks. They discuss the strengths of Claude as a conversational tool, while noting that both models have their respective advantages and limitations, touching on aspects like cost efficiency, political correctness, and verbosity in provided information.

- **Learning AI and PyTorch**: Users exchanged advice on the mathematical foundations needed to dive into AI and PyTorch, suggesting pre-calculus and linear algebra as starting points. Resources like **3blue1brown** on YouTube were recommended for intuitive learning, and users were encouraged to engage in continuous learning and exploration.

- **AI Support Channels**: There was an exchange of information detailing how to get in touch with OpenAI's support team. Discussion highlights included navigating **OpenAI's help bot** on their support website, and leading users to report bugs or raise tickets for assistance, while also mentioning **platform.openai.com** for bug reports, with a reference to **<#1070006915414900886>** in Discord for additional help.

**Link mentioned**: <a href="https://openai.com/enterprise-privacy">Enterprise privacy</a>: no description found

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1218428016573812888)** (34 messages🔥): 

- **Curiosity for GPT-5**: There was a short exchange with users inquiring about the release date of **GPT-5**, but no specific information or dates were provided.
- **Integration Challenges with GPT-3.5**: One user experienced difficulties making **GPT Turbo 3.5** generate code accurately, specifically regarding its method to locate elements on a webpage, and wondered if it was due to outdated **Playwright** libraries.
- **Troubleshooting GPT Response Issues**: Members reported problems with GPT not responding to prompts, and others suggested it could be an error needing support assistance.
- **Discussion on Sudden Change in ChatGPT Behavior**: Concerns were raised about ChatGPT's behavior changing over the past few days, later identified as a conflict with the **WebChatGPT Chrome extension** by the user experiencing the issue.
- **Filter Sensitivity Frustrations**: Multiple users expressed frustrations with content filters being too sensitive for creative writing purposes, noting that even benign actions like a "kiss on the lips" could trigger warnings or refusals by GPT.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1218207072064114718)** (79 messages🔥🔥): 

- **Exploring Prompt Architecture for Classification**: A member discusses optimizing prompt structure for **classification tasks**, aiming for higher recall and fewer false positives. They are experimenting with the amount of context provided and considering using a custom **GPT model**.

- **Troubles with Turbo for Playwright Tests**: When attempting to generate Playwright test code using **GPT-3.5 Turbo**, it creates non-usable code. A member suggests that the model might not be up to date with the latest Playwright library and that **GPT-4** could yield better results.

- **Dealing with Refusals in Output**: One member experiences frequent **"refusal to do tasks"** by the model, which prompts a discussion about how to handle or avoid such refusals. Members recommend using meta-prompting strategies and breaking tasks into chunks to prevent the model from hitting refusal conditions.

- **Shifting Behaviors and Content Policies**: The conversation also touches upon the observation that prompts which worked previously now yield **"sorry I can't do that"** messages, hinting at changes in the model's behavior over time or more aggressive bias minimization. There's a discussion about the challenges in overcoming these hurdles without stepping into content policy violation territory.

- **Querying Strategies for Web Search**: A member asks how to get the AI to use **web search with multiple queries** for more comprehensive information gathering. Despite confusion, it is clarified that guidance should be provided to the model on which sources to check and what information to look for.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1218207072064114718)** (79 messages🔥🔥): 

- **Clarifying Context Window for Classification**: A member inquired about the optimal amount of context to include in a prompt for classification use cases. They are attempting to achieve higher recall and minimize false positives through a detailed prompt architecture, considering a **dataframe with input features**. Another member suggested referring to "needle in a haystack results" and recommended using no more than 1/2 of the total context window for the best compliance and completion.

- **Prompting Playback**: Members discussed the occasional tendency of the AI to refuse tasks, which seems to increase in frequency within a single conversation. One proposes **meta-prompting** as a solution, suggesting it allows the AI to moderate itself to avoid refusals without contravening content policies.

- **Exploring Model Responses and Performance**: Chat participants exchanged observations on how GPT models respond to tasks, including increased refusal messages for previously functioning prompts. A member highlighted the implementation of "Superficial algorithmic bias minimization" and posed a method of **categorizing GPT responses** into various types to decipher whether a prompt was understood.

- **Web Search Woes and Workarounds**: A user asked how to instruct GPT to conduct a web search using multiple queries for a more comprehensive set of results rather than a singular query. The ensuing discussion explored techniques like **prompt engineering** to guide the AI toward desired outputs but clarification on the process remained necessary.

- **Sharing Solutions and Seeking Support**: Members shared their creative uses of GPT, including creating a support-focused AI and asked for feedback from the community. There was also discussion on how the model's perceived refusal behavior might affect user experience and expectations of AI interactions.
  

---



**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1218106794698739782)** (96 messages🔥🔥): 

- **Aya Demo Asks for a Slider**: The Aya demo has received a community contribution, implementing a high *repetition penalty*. A request has been made for contributors to add a **slider feature** to the Gradio interface. [To contribute, make a PR here](https://huggingface.co/spaces/Tonic/Aya/discussions/3).

- **NVIDIA H100 and ARM-Based Server CPUs Generate Buzz**: A **massive GPU** combined with a **server CPU** on the same board, rumored to draw around **850W** of power, was a topic of intrigue. Discrepancies arose in power consumption figures, ranging from expected numbers like **300-350W for the GPU** to claims of the H100 drawing **up to 700W**. [Link to benchmarks](https://www.phoronix.com/review/nvidia-gh200-gptshop-ben).

- **Data Hoarding on HuggingFace**: A member revealed a **data leaderboard** showcasing the large volume of data hosted on HuggingFace, including over **120B models**. [Leaderboard here](https://huggingface.co/spaces/Weyaxi/data-leaderboard).

- **Discussion on Working with Large LLMs**: Members shared insights into the challenges and considerations of working with **large language models** (LLMs) and high-performance computing. Topics ranged from the *slow generation* speed taking **tens of seconds for a single token**, to the potential of *quantization to improve speed*, and the complexities of managing **models like xAI's Grok-1** with 314 billion parameters.

- **Community Engagement with Grok Release**: The release of the **Grok-1 model** with **314 billion parameters**, under Apache 2.0 license, sparked numerous discussions. Links to get started with Grok were shared, while concerns were raised about the ability to upload such significant datasets onto platforms like HuggingFace. [Read more about Grok](https://x.ai/blog/grok-os) or find the Grok model on [HuggingFace](https://huggingface.co/alpindale/grok-1).

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.phoronix.com/review/nvidia-gh200-gptshop-ben">Tweet from Linux Performance, Benchmarks &amp; Open-Source News - Phoronix</a>: no description found</li><li><a href="https://huggingface.co/spaces/ivrit-ai/whisper-large-v3-space">Whisper Large V3 - a Hugging Face Space by ivrit-ai</a>: no description found</li><li><a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 is a 314B parameter Mixture of Experts model - Base model (not finetuned) - 8 experts (2 active) - 86B active parameters - Apache 2.0 license - Code:  - Happy coding! p.s. we re hiring: </li><li><a href="https://huggingface.co/spaces/Tonic/Aya/discussions/3">Tonic/Aya · Set a repetition_penalty constant as 1.8</a>: no description found</li><li><a href="https://fxtwitter.com/Weyaxi/status/1768779404442739147">Tweet from Weyaxi (@Weyaxi)</a>: 🤔Have you ever wondered how much data we host on @huggingface?  Well, I did after seeing  @TheBlokeAI&#39;s model count and 120B models just chilling on the platform 😅  📊 So I scraped all repositor...</li><li><a href="https://github.com/gradio-app/gradio/issues/7722">Video-LLaVA demo api not working with Gradio-Client · Issue #7722 · gradio-app/gradio</a>: Describe the bug Im trying to use the python api for the Video-LLaVA model demo on hugging face spaces but I get an error: Traceback (most recent call last): File &quot;/Users/kamakshiramamurthy/Deskt...</li><li><a href="https://github.com/moritztng/fltr">GitHub - moritztng/fltr: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B.</a>: Like grep but for natural language questions. Based on Mistral 7B or Mixtral 8x7B. - moritztng/fltr
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1218115205553324112)** (12 messages🔥): 

- **Bayesian Optimization Baffles**: A member expressed confusion about **Bayesian optimization** in comparison to GridSearch and RandomSearch optimization techniques.
  
- **Seeking Hugging Face Guidance**: A member requested help in understanding how to use **Hugging Face** and its services, such as the Transformers library for natural language processing tasks.

- **Duet AI Covers Troubles**: One inquiry centered on producing **AI covers of duets and bands** which resulted in a response suggesting the separate recording and overlaying of individual voices to improve quality.

- **End-to-End MLOps with SageMaker and Hugging Face**: A member shared a link to a **workshop notebook** about using Amazon SageMaker and Hugging Face for creating an **MLOps pipeline**, with detailed steps and prerequisites ([Workshop Notebook](https://www.philschmid.de/mlops-sagemaker-huggingface-transformers)).
  
- **Image Processing Aspirations**: A member discussed plans to integrate basic image processing tools such as **contrast and brightness adjustment** into their project, **Fooocus**, to avoid using Photoshop.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co).">no title found</a>: no description found</li><li><a href="https://www.philschmid.de/mlops-sagemaker-huggingface-transformers">MLOps: End-to-End Hugging Face Transformers with the Hub &amp; SageMaker Pipelines</a>: Learn how to build an End-to-End MLOps Pipeline for Hugging Face Transformers from training to production using Amazon SageMaker.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1218346001421570138)** (12 messages🔥): 

- **The Linguistic Duality Breakthrough**: Members discussed the capability of machine learning models to handle languages as linguistically different as Chinese and English. One member expressed surprise at this capability, especially given the deep differences in linguistic structure and modes of thinking specific to each language.

- **Exploring Multilingual Model's Thought Process**: Following the conversation about language models working across Chinese and English, discussions pointed to the fact that task simplicity might mask the nuanced differences in language-specific knowledge. It was mentioned that while basic tasks showcased in a paper can be completed, the intricacy of authoring a Chinese novel could highlight these intrinsic linguistic distinctions.

- **Medusa in the Spotlight**: A link to a paper about **Medusa**, an efficient method for Language Model inference that includes parallel processing, was shared. It sparked a curiosity about how such models would distill information effectively when predictions are not language-specific.

- **Assessing the Influence of English in Multilingual Models**: Concerns were raised that an English-dominated training corpus might inadvertently skew a model towards European language and thought patterns. This ongoing dialogue reflected the community’s engagement with open questions about language models being influenced by dominant languages such as English.

- **How Chatbots Might Alter Peer Reviews**: A paper was highlighted that studied the impact of Large Language Models (LLMs) on scientific peer reviews, with findings suggesting a significant percentage of text in AI conference reviews could have been modified by LLMs. The conversation seems centered on the behavioral insights and implications of LLM modifications in academic peer review contexts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2401.10774">Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads</a>: The inference process in Large Language Models (LLMs) is often limited due to the absence of parallelism in the auto-regressive decoding process, resulting in most operations being restricted by the m...</li><li><a href="https://huggingface.co/papers/2403.09611">Paper page - MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.07183">Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>: We present an approach for estimating the fraction of text in a large corpus which is likely to be substantially modified or produced by a large language model (LLM). Our maximum likelihood model leve...</li><li><a href="https://bytez.com/read/arxiv/2403.07183">Bytez: Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>: This study examines the use of large language models (LLMs), like ChatGPT, in scientific peer review. The authors developed a method to estimate the percentage of text in peer reviews that is generate...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1218158991570636900)** (18 messages🔥): 

- **NL2SQL Seeker Seeks Aid**: A participant is working on a **NL2SQL pipeline** using BAAI/llm-embedder, TheBloke/nsql-llama-2-7B-GGUF, and FAISS for vector storage, seeking advice to improve the accuracy of selecting relevant SQL tables and generating queries.

- **NVIDIA's Newest Powerhouse Revealed**: A member introduces the **NVIDIA Grace Hopper Superchip**, emphasizing its strength in HPC, AI, and data center applications.

- **A Journey into NLP Begins**: Newcomers to NLP are directed to the Hugging Face NLP course at [HuggingFace Course](https://huggingface.co/learn/nlp-course/chapter1/1) and the comprehensive textbook hosted at [Stanford's SLP3 manuscript](https://web.stanford.edu/~jurafsky/slp3/).

- **NLP Learning Resources Compilation**: Alongside the above resources, participants mention **Stanford's CS224n course notes** as a concise version of the Stanford manuscript to aid in NLP education.

- **Exploring Free LLM APIs for Production**: One user inquires about a free LLM API for production deployment, with another suggesting "ollama" for a free option to implement locally.

**Link mentioned**: <a href="https://huggingface.co/learn/nlp-course/chapter1/1">Introduction - Hugging Face NLP Course</a>: no description found

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1218217429868478474)** (7 messages): 

- **Interacting with Documents as Tools in RAG**: Suggested an innovative approach to handle complex queries in a [RAG pipeline](https://t.co/eCdLmlXZFj), where each retrieved document is treated as an interactive tool, thus enabling more advanced interactions.
- **Launching LlamaIndex v0.10.20 with Instrumentation**: Announced the new version of LlamaIndex featuring an Instrumentation module, including notebooks demonstrating [basic observability](https://t.co/GY4unUYOwl) and [API call observation](https://t.co/E1d9dtkqAI).
- **Search-in-the-Chain for Enhanced QA**: Discussed a paper by Shicheng Xu et al. that introduces a method to intertwine retrieval and planning for better question-answering through a process that [verifies steps and adjusts plans accordingly](https://t.co/7gLlDyd1cV).
- **Blog Post on RAG-based Job Assistant**: Highlighted a [blog post by Kyosuke Morita](https://t.co/1Y9TPgGHW1) about creating a job assistant to match candidates to jobs by parsing CVs using LlamaParse in combination with LlamaIndex.
- **MemGPT Webinar Released**: Shared a [webinar featuring Charles Packer](https://t.co/bUpqCvLweS) which introduces MemGPT, an architecture that gives an agent memory tools to interact with a "core" memory, enhancing its function-calling abilities.
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1218113300764819488)** (303 messages🔥🔥): 

- **Chaining OpenAI Agents Puzzle**: Discussions revolved around the possibility of chaining multiple OpenAI agents using tools described in LlamaIndex documentation. A member attempted to use `FunctionTool` and `QueryEngineTool` from LlamaIndex but encountered an error suggesting that the message content was empty or incorrectly formatted.
  
- **Xinference CPU Cluster Query**: Members discussed whether using Xinference in a CPU cluster can reduce inference times. While the knowledge base lacks specific performance details, generally using CPU clusters for inference can distribute workloads and potentially speed up the process.
  
- **Adjusting Token Limit for Local LLM**: A user required assistance on changing the max token size for local LLMs. It was suggested to use `Ollama(... additional_kwargs={"num_predict": number_of_tokens})` and passing `context_window` to the constructor as potential solutions.
  
- **Filtering in LlamaIndex**: One member asked if metadata filtering could be done before retrieval in the SimpleFusionRetriever and Retriever Query Engine process. It was hinted that vector databases like Qdrant can attach filters to sub-retrievers to allow for pre-retrieval filtering.
  
- **Langfuse Integration Spans Issue**: A user integrating Langfuse with LlamaIndex noted missing spans for certain steps like embedding user questions and looking up documents in Qdrant. It was suggested they ensure the callback manager is passed into all components, including the embedding model, to see the expected spans.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cloud.llamaindex.ai">LlamaCloud</a>: no description found</li><li><a href="https://www.promptingguide.ai/techniques/fewshot">Prompt Engineering Guide</a>: A Comprehensive Overview of Prompt Engineering</li><li><a href="https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.CodeSplitter.html">CodeSplitter - LlamaIndex 🦙 v0.10.20.post1</a>: no description found</li><li><a href="https://qdrant.tech/documentation/tutorials/llama-index-multitenancy/">Multitenancy with LlamaIndex - Qdrant</a>: Qdrant is an Open-Source Vector Database and Vector Search Engine written in Rust. It provides fast and scalable vector similarity search service with convenient API.</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html">Defining and Customizing Documents - LlamaIndex 🦙 v0.10.20.post1</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/image_to_image_retrieval.html">Image to Image Retrieval using CLIP embedding and image correlation reasoning using GPT4V - LlamaIndex 🦙 v0.10.20.post1</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/extraction.html">Structured Data Extraction - LlamaIndex 🦙 v0.10.20.post1</a>: no description found</li><li><a href="https://www.promptingguide.ai/techniques/rag">Prompt Engineering Guide</a>: A Comprehensive Overview of Prompt Engineering</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/root.html">Tools - LlamaIndex 🦙 v0.10.20.post1</a>: no description found</li><li><a href="http://localhost:{port}",>">no title found</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/5c53f41712785e5558156372bdc4f33a6326fa5f/docs/examples/vector_stores/Qdrant_using_qdrant_filters.ipynb">llama_index/docs/examples/vector_stores/Qdrant_using_qdrant_filters.ipynb at 5c53f41712785e5558156372bdc4f33a6326fa5f · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/issues/12034">[Question]: custom llm but is blocked · Issue #12034 · run-llama/llama_index</a>: Question Validation I have searched both the documentation and discord for an answer. Question the code is from typing import Optional, List, Mapping, Any from llama_index.core import SimpleDirecto...</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py">llama_index/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/hofstadter-io/hof/blob/_dev/flow/chat/prompts/dm.cue">hof/flow/chat/prompts/dm.cue at _dev · hofstadter-io/hof</a>: Framework that joins data models, schemas, code generation, and a task engine. Language and technology agnostic. - hofstadter-io/hof</li><li><a href="http://127.0.0.1:9997>">no title found</a>: no description found</li><li><a href="http://localhost:{port}">)">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1218542835754860564)** (4 messages): 

- **RAG Tutorial with LlamaParse and More**: A step-by-step video on creating an effective RAG with LlamaParse, Qdrant, and Groq has been shared, explaining the process and showcasing **LlamaParse** functionality. Watch the detailed guide on [YouTube](https://youtu.be/w7Ap6gZFXl0).

- **In Search of RAG Preparation Tips**: A member is seeking advice on the top tips for preparing a document for **RAG** and methods for automatically adding metadata to **pinecone** for optimal document retrieval.

- **Medium Post on AI Assistant Using RAG**: An article discussing the **empowerment of voices through an AI Assistant** with a RAG pipeline, memory, and LlamaIndex has been recommended. The in-depth analysis can be found on [Medium](https://medium.com/ai-advances/empowering-voices-ai-assistant-with-rag-pipeline-memory-and-llamaindex-11c4e319d915).

- **Switching to Huggingface Models in RAG Implementation**: A member is having trouble replacing OpenAI models with Huggingface models in a **RAPTOR** pack for RAG, citing multiple errors in the process. They are seeking advice on correcting their implementation based on an example from the official [GitHub repository](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raptor/examples/raptor.ipynb).

**Link mentioned**: <a href="https://youtu.be/w7Ap6gZFXl0">RAG with LlamaParse, Qdrant and Groq | Step By Step</a>: In this video, I will show you how to create a effective RAG with LlamaParse, Qdrant and Groq. I will explain what LlamaParse is and briefly walk you through...

  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1218154073912639508)** (202 messages🔥🔥): 

- **Understanding Yann's Stance on LLMs**: A series of discussions highlighted a tweet from @Teknium1 regarding Yann LeCun's bearish view on large language models (LLMs). It was mentioned that Yann might favor models with visual reasoning or planning capabilities over purely language-based models, based on the hypothesis some individuals inherently lack an internal monologue, possibly influencing their preference for non-linguistic thought processes. An interview with an individual who also lacks an inner monologue was shared. Members questioned the dichotomy between 'shape rotators' and 'wordcels' in cognitive reasoning.
- **OpenAI's GTC Virtual Sessions Offer**: Members discussed OpenAI's GTC (GPU Technology Conference) attendance, sharing free access codes to virtual sessions and hinting at a potential hardware exchange program for influencers who help with sign-ups. The registration link was provided along with access to session details.
- **Releasing Grok-1: Huge Model with Uncertain Impact**: xAI announced the open release of Grok-1, a 314 billion parameter Mixture-of-Experts model, hoping for community contributions in continued training and evaluation. The community reaction was mixed, with some expressing concern over its quality compared to other models like LLaMa and Claude, while appreciating the scale of the model. Discussions revolved around the potential of continual pretraining and quantization to improve or utilize the model.
- **SWYX on Lex Podcast's Missed Opportunities**: The Lex Fridman podcast featuring Sam Altman received criticism for not delving into more substantial issues and glossing over the inner workings and politics at OpenAI. Listeners found the conversation lacking in depth, focusing more on tangential topics and less on providing insights into AI and model advancements.
- **Jensen Huang's Nvidia Keynote Expectations**: There was anticipation for Nvidia CEO Jensen Huang's GTC keynote, with speculations on the potential reveal of significant parameters for AI advancements. While no direct quotes confirm it, the community seemed to accept the 1.8 trillion parameter reveal for GPT-4 during the presentation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://x.com/teknium1/status/1768452864995942469?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Teknium (e/λ) (@Teknium1)</a>: This explains why Yann is so bearish on LLMs... 😲</li><li><a href="https://x.com/repligate/status/1769241542420738126?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from j⧉nus (@repligate)</a>: this was the result of navigating to the ../../microsoft/bing/bing_chat directory in claude&#39;s backrooms, then letting claude use commands to look around on its own, then running:  &lt;cmd_soul&gt;...</li><li><a href="https://arxiv.org/abs/2402.10171">Data Engineering for Scaling Language Models to 128K Context</a>: We study the continual pretraining recipe for scaling language models&#39; context lengths to 128K, with a focus on data engineering. We hypothesize that long context modeling, in particular \textit{t...</li><li><a href="https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space">Explaining the SDXL latent space</a>: no description found</li><li><a href="https://x.com/francis_yao_/status/1769575936994013611?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Yao Fu (@Francis_YAO_)</a>: Grok&#39;s MMLU is only on par with Mixtral, despite one order of magnitude larger. I believe it has great potential but not fully released, and good continue pretrain data may substantially lift the ...</li><li><a href="https://huggingface.co/collections/suno/bark-6502bdd89a612aa33a111bae">Bark - a suno Collection</a>: no description found</li><li><a href="https://x.com/Francis_YAO_/status/1759986097365627054?s=20">Tweet from Yao Fu (@Francis_YAO_)</a>: Frontier models all have at least 100k context length, Gemini 1.5 has even 1m context. What about research and open source?   Introducing Long Context Data Engineering, a data driven method achieving ...</li><li><a href="https://x.com/teortaxestex/status/1769460562763604375?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: @aidan_mclau 0) Rocket man bad 1) it&#39;s not much worse 2) As you can see it&#39;s a sparse-upcycled Grok-0. It&#39;s undercooked. In 2023, continual pretraining has been ≈solved, and having validat...</li><li><a href="https://substack.recursal.ai/p/eaglex-17t-soaring-past-llama-7b">🦅 EagleX 1.7T : Soaring past LLaMA 7B 2T in both English and Multi-lang evals (RWKV-v5)</a>: A linear transformer has just cross the gold standard in transformer models, LLaMA 7B, with less tokens trained in both English and multi-lingual evals. A historical first.</li><li><a href="https://x.com/openinterpreter/status/1769448726660337875?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Open Interpreter (@OpenInterpreter)</a>: 100 years in the making. 100 hours to go.</li><li><a href="https://x.com/teknium1/status/1768452864995942469?s=46&t=6FDP">Tweet from Teknium (e/λ) (@Teknium1)</a>: This explains why Yann is so bearish on LLMs... 😲</li><li><a href="https://x.com/altryne/status/1768683178888208816?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>:   Sora team showing up at Berkley to talk about SORA</li><li><a href="https://x.com/granawkins/status/1768530196557365599?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Grant♟️ (@granawkins)</a>: &#34;Between Q1-24 and Q4-25, there will be a 14x increase in compute.  Then, if you factor in algorithmic efficiency doubling every 9 months, the effective compute at the end of next year will be alm...</li><li><a href="https://x.com/swyx/status/1769776691562324215?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from swyx (@swyx)</a>: how is it possible to have a 2hr conversation with sama and get zero alpha  but hey we talked about aliens again thats fun</li><li><a href="https://x.com/kk_slider_k_/status/1768464173657158132?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from KZ (@kzSlider)</a>: This makes so much sense. Yann’s always been looking for models that reason visually or using planning rather than purely in language  ↘️ Quoting Teknium (e/λ) (@Teknium1)   This explains why Yann is ...</li><li><a href="https://x.com/emmanuel_2m/status/1768360522028876045?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Emm (@emmanuel_2m)</a>: 🚨 Today, we&#39;re excited to launch the Scenario #UPSCALER! Elevate your AI creations up to 10k resolution.  🚀 Built for unmatched #CreativeControl & guided workflows.  💰 It starts at just $15/mo ...</li><li><a href="https://x.com/xlr8harder/status/1769454853506638008?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from xlr8harder (@xlr8harder)</a>: I think I speak for everyone here when I say: 314 billion parameters what the hell</li><li><a href="https://x.com/danielhanchen/status/1769550950270910630?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Daniel Han (@danielhanchen)</a>: Had a look through @Grok&#39;s code: 1. Attention is scaled by 30/tanh(x/30) ?! 2. Approx GELU is used like Gemma 3. 4x Layernoms unlike 2x for Llama 4. RMS Layernorm downcasts at the end unlike Llama...</li><li><a href="https://www.nfx.com/post/ai-like-water">Tweet from AI Is Like Water</a>: Generative AI is like water. The phrase was borne out of frustration, but it opens up a new world of AI playbooks.</li><li><a href="https://x.com/burny_tech/status/1769549895835226613?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Burny — Effective Omni (@burny_tech)</a>: New details about GPT-5 from Sam Altman He’s basically admitting that GPT-5 will be a massive upgrade from GPT-4, so we can expect a similar jump from 3 to 4. &#34;&#34;If you overlook the pace of imp...</li><li><a href="https://x.com/joshwalkos/status/1767745681375015076?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Champagne Joshi (@JoshWalkos)</a>: This is a fascinating conversation with a girl who lacks an internal monologue. She articulates the experience quite well.</li><li><a href="https://youtu.be/I-HMKky7Qsw?si=yCvekF3a0zr_1IgA&t=718">Beyond Transformers - Intro to RWKV Architecture &amp; The World To... Eugene Cheah &amp; Harrison Vanderbyl</a>: Beyond Transformers - Intro to RWKV Architecture &amp; The World Tokenizer - Eugene Cheah &amp; Harrison Vanderbyl, Recursal AIWhats comes next after transformers?In...</li><li><a href="https://www.youtube.com/watch?v=USlE2huSI_w">WATCH: Jensen Huang&#39;s Nvidia GTC Keynote - LIVE</a>: Tune in at 1:00pm PT / 4:00pm ET when Nvidia CEO Jensen Huang kicks off its biannual GTC conference.Never miss a deal again! See CNET’s browser extension 👉 ...</li><li><a href="https://github.com/FranxYao/Long-Context-Data-Engineering">GitHub - FranxYao/Long-Context-Data-Engineering: Implementation of paper Data Engineering for Scaling Language Models to 128K Context</a>: Implementation of paper Data Engineering for Scaling Language Models to 128K Context - FranxYao/Long-Context-Data-Engineering</li><li><a href="https://youtu.be/J0p_thJJnoo?si=IaGuEgUcs1BRgjhF">#51 FRANCOIS CHOLLET - Intelligence and Generalisation</a>: In today&#39;s show we are joined by Francois Chollet, I have been inspired by Francois ever since I read his Deep Learning with Python book and started using th...</li><li><a href="https://x.com">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://www.nvidia.com/gtc/?ncid=ref-inor-332714">GTC 2024: #1 AI Conference</a>: Register now. Streamed online. March 18-21, 2024.</li><li><a href="https://docs.google.com/document/d/1HZ326V6KNK4QIlG7uEldQEizFgTaO7Hg9uJxURYy9f8/edit">NVIDIA &amp; Harpreet Sahota GTC 2024</a>: no description found</li><li><a href="https://youtu.be/jvqFAi7vkBc?si=WTfgLyNfGhkP2Azx">Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power &amp; AGI | Lex Fridman Podcast #419</a>: Sam Altman is the CEO of OpenAI, the company behind GPT-4, ChatGPT, Sora, and many other state-of-the-art AI technologies. Please support this podcast by che...</li><li><a href="https://buttondown.email/ainews/archive/ainews-mm1-apples-first-large-multimodal-model/">[AINews] MM1: Apple&#x27;s first Large Multimodal Model</a>: AI News for 3/14/2024-3/15/2024. We checked 358 Twitters and 20 Discords (332 channels, and 2839 messages) for you. Estimated reading time saved (at 200wpm):...</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...</li><li><a href="https://bytez.com/read/arxiv/2402.10588">Bytez: Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>: In this research study, scientists wanted to know if language models (that can generate text) use English as a &quot;pivot&quot; language internally, even when prompted in other languages. They found ...</li><li><a href="https://huggingface.co/collections/stereoplegic/multilingual-65389b21be39573b3b2db98d">Multilingual - a stereoplegic Collection</a>: no description found
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1218137415068422164)** (2 messages): 

- **Join the Paper Club Discussion**: A reminder was issued to join the **Paper Club session** where they are going through the paper "A Comprehensive Summary Of Large Language Models". The session was set to begin in 2 minutes in channel <#1107320650961518663>.

- **AI Models Dropping Beats**: A new song titled "90s hip-hop song" about **AI models** creating new songs was shared, featuring lyrics about AI's impact on music and the ability to generate new content based on historic data. The song can be found at [Suno AI](https://app.suno.ai/song/83680b6f-db37-44de-adf9-3f7fff6b79d9).

**Link mentioned**: <a href="https://news.ycombinator.com/item?id=39746163">Suno, an AI music generator | Hacker News</a>: no description found

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1218135292574306328)** (20 messages🔥): 

- **Exploring the Why Behind Attention**: The discussion in LLM Paper Club (Asia) focused on clarifying *why the attention mechanism in transformers* was developed. It addressed the limitations of previous fixed-length encoding vectors and **how attention allows the model to consider all parts of the input sequence**.
  
- **Parallelization Puzzles Solved**: A participant explained that attention in transformer models allows for **parallel processing of different tokens**, enabling more efficient compute and faster training compared to sequential models like RNNs.
  
- **Attention is the Key to Efficiency**: By **processing tokens independently using the scaled dot product operation**, attention mechanisms remove the need for sequential "waiting" found in older models such as RNNs.
  
- **Grasping the Intuition Behind LLM Design**: The conversation highlighted an issue faced by some learners jumping directly into GPT-models: the challenge of understanding **intuitive decisions in the model's design** and recognizing the problems they resolve.
  
- **Appreciation for Hosted Session Insight**: By the end of the session, participants expressed gratitude, noting they had gained better intuition about the evolution and rationale behind **long language models (LLMs)** thanks to the hosts' explanations.
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1218287754715201638)** (36 messages🔥): 

- **Quiet Check-in from Members**: A few members are passively tuning in today or expressing general greetings; active participation may be limited for some due to being in meetings.

- **In-Depth Blog Post Promise**: A member mentioned they will be posting a detailed version of a topic on their blog later, hinting at more information to come on a specific discussion.

- **The Waiting Game**: One member likened a loading screen experience to '**the RAG experience**,' likely referring to the Retrieval-Augmented Generation model usage.

- **RAG Discussion and Resource Sharing**: A link to an article titled "Advanced RAG 01 - Small to Big Retrieval" was shared, suggesting an in-depth look at Retrieval-Augmented Generation: [Advanced RAG](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4).

- **Curiosity About AI Modeling Alternatives**: There was a discussion about alternatives to cosine similarity in AI modeling, with a nod towards the concept of 'contrastive embeddings' and the application of **LLMs (Large Language Models)** in generating these embeddings.

**Link mentioned**: <a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-struct...

  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1218220293865345024)** (168 messages🔥🔥): 

- **Codex on CoPilot**: A member found out that **Microsoft Codex** can be accessed for free within the Copilot app, offering tools like Jupyter Notebooks along with libraries like simpy and matplotlib.

- **LAION's Hugging Face Dataset**: There was confusion about the **DALL-E 3 dataset** being removed from Hugging Face, which was clarified to have been moved to a new location. A useful direct [link](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset) to the dataset was provided.

- **IPFS Bridge Development**: A member is working to finish a "model manager" for an MLops platform and is polishing an **IPFS - Hugging Face** bridge. A scraping tool is already functioning for mirroring datasets on IPFS.

- **Grok-1 Release Discussion**: The release of **Grok-1**, a new 314B parameter model by OpenAI, was shared and discussed. It was noted for its performance in code/humaneval benchmarks and compared to other models like **Mixtral** and **LLaMA**.

- **AI in Browser**: A query was raised about running language models in a browser without a paid API, leading to suggestions of using libraries like **transformer.js**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/datasets/en/loading#hugg">Load</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/loading#hugging-face-hub">Load</a>: no description found</li><li><a href="https://tenor.com/view/silicon-valley-yes-cheer-think-gif-9010547">Silicon Valley Yes GIF - Silicon Valley Yes Cheer - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.economist.com/business/2023/11/23/why-chinese-companies-are-flocking-to-mexico">Why Chinese companies are flocking to Mexico</a>: The country offers a back door to the United States</li><li><a href="https://fxtwitter.com/imgn_ai/status/1769791182270333067">Tweet from imgnAI (@imgn_ai)</a>: catgirls are at NVIDIA GTC ✨  meowing for your creative freedom 👊  this is a message that needs to be heard 🐱💕</li><li><a href="https://github.com/victorchall/EveryDream2trainer/blob/main/caption_cog.py">EveryDream2trainer/caption_cog.py at main · victorchall/EveryDream2trainer</a>: Contribute to victorchall/EveryDream2trainer development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/aiwars/comments/1bbxtp6/the_people_behind_the_nightshade_glaze_account/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset">OpenDatasets/dalle-3-dataset · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1218181910669295716)** (13 messages🔥): 

- **Clarification on Channel Topics**: Members pointed out that discussions about web UIs related to free Colab might not be suitable for the **research** channel, as it's not about cutting-edge research.
  
- **Generative World Model Document Shared**: A link to a Google Doc titled "Generative Audio Video Text world model" was shared, although no additional commentary or explanation was provided. [View the document](https://docs.google.com/document/d/1f6CpVjdApmQl3nXsUACGtSd9nML3CypI1iE889i4JbM/edit?usp=drivesdk).

- **Pre-training LLMs on New Data**: An **arXiv paper** was mentioned that discusses how incorporating simple techniques such as learning rate warming and replay of previous data can save compute compared to re-training language models on new data. [Read the article](https://arxiv.org/abs/2403.08763).

- **Grok Open Release on GitHub**: A GitHub repository for **Grok open release** was linked, with no further discussion on its contents or implications. [Explore the repo](https://github.com/xai-org/grok-1).

- **Speculation on Nvidia Confirming GPT-4 Details**: Discussion surfaced around a rumor that Nvidia confirmed **GPT-4** is a mixture of experts (MoE) with 1.8 trillion parameters, referencing an image on Twitter. [See the tweet](https://pbs.twimg.com/media/GI-reRIW0AAZpMC?format=jpg&name=large). It was also noted that GPT-4's exact identity remains speculative.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.08763">Simple and Scalable Strategies to Continually Pre-train Large Language Models</a>: Large language models (LLMs) are routinely pre-trained on billions of tokens, only to start the process over again once new data becomes available. A much more efficient solution is to continually pre...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: In this work, we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and c...</li><li><a href="https://docs.google.com/document/d/1f6CpVjdApmQl3nXsUACGtSd9nML3CypI1iE889i4JbM/edit?usp=drivesdk">Generative Audio Video Text world model</a>: no description found</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1218310691103178803)** (43 messages🔥): 

- **Exploring the Photonics Frontier**: [Anastasia's YouTube video](https://youtu.be/8ohh0cdgm_Y) discusses a new chip technology that is a thousand times faster and links to both the video and the associated Nature paper were shared. Further recommendations for videos on photonics include the [Asianometry channel](https://www.youtube.com/watch?v=29aTqLvRia8) with topics like silicon photonics and light-based neural networks.

- **PyTorch vs. TensorFlow: Memory Management Choices Explained**: In-depth discussions on the reasons behind PyTorch's decision to expose tensor memory management to users, highlighting avoidance of hidden copies, no magic principle, and explicit device handling in mathematical operations.

- **Looking for the Latest GPU Profiling Tools?**: Users discussed cloud GPU services that allow profiling with nsight compute on Ada or Hopper GPUs with suggestions like [RunPod](https://www.runpod.io/) and [Lambda Labs](https://lambdalabs.com/), with reports of some services not granting the necessary privileges for profiling.

- **NVIDIA's GTC Keynote Sparks Conversations**: During the GTC March 2024 keynote, NVIDIA CEO Jensen Huang's mention of a 1.8T parameter state-of-the-art model stirred curiosity among members, alongside discussions about new hardware reveals like the B100 with 192GB HBM, security enhancements, and interconnect technologies.

- **Getting Started and Finding Your Place**: A new member sought guidance on where to introduce themselves within the community, with direction provided towards channels structured around specific technologies and libraries, such as the beginner channel for a smooth start.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.set_default_device.html">torch.set_default_device &mdash; PyTorch 2.2 documentation</a>: no description found</li><li><a href="https://www.runpod.io/">Rent Cloud GPUs from $0.2/hour</a>: no description found</li><li><a href="https://www.cerebras.net/product-chip/">Product - Chip - Cerebras</a>: no description found</li><li><a href="https://lambdalabs.com/">GPU Cloud, Clusters, Servers, Workstations | Lambda</a>: GPU Cloud, GPU Workstations, GPU Servers, and GPU Laptops for Deep Learning &amp; AI. RTX 4090, RTX 3090, RTX 3080, RTX A6000, H100, and A100 Options. Ubuntu, TensorFlow, and PyTorch Pre-Installed.</li><li><a href="https://youtu.be/8ohh0cdgm_Y?si=q3wOMlzp_Nmn8_AJ">New Chip Breakthrough: x1000 faster</a>: Get TypeAI PREMIUM now! Start your FREE trial by clicking the link here:  https://bit.ly/Mar24AnastasiInTechThe paper: https://www.nature.com/articles/s41586...</li><li><a href="https://www.youtube.com/live/Y2F8yisiS6E?si=g5MChTXs3a9gGykE">GTC March 2024 Keynote with NVIDIA CEO Jensen Huang</a>: Watch NVIDIA CEO Jensen Huang’s GTC keynote to catch all the announcements on AI advances that are shaping our future.Dive into the announcements and discove...</li><li><a href="https://lightmatter.co/">Lightmatter®</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=29aTqLvRia8">Silicon Photonics: The Next Silicon Revolution?</a>: My deepest thanks to friend of the channel Alex Sludds of MIT for suggesting this topic and helping me with critical resources. Check him out here: https://a...</li><li><a href="https://www.youtube.com/watch?v=t0yj4hBDUsc">Running Neural Networks on Meshes of Light</a>: I want to thank Alex Sludds for his efforts in helping me research and produce his video. Check out his work here: https://alexsludds.github.ioLinks:- The As...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1218241351582482493)** (7 messages): 

- **New Triton Debugging Visualizer**: A member introduced a new visualizer aimed at simplifying the process of debugging in Triton by offering better views of the **spatial structure of load/stores**. No specifics on how the visualizer looks were provided.
- **Try Your Hand at Triton Puzzles**: The same member also shared a set of **Triton Puzzles**, which are considered a bit challenging but are good for understanding complex problems. Interested members can try them out and report any issues found at this [Google Colab link](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing). Known bugs include occasional double visualizations and segmentation faults.
- **Seeking Triton Learning Resources?**: A member asked for Triton learning resources given their familiarity with CUDA. Responses pointed to using the official Triton tutorials, the aforementioned puzzles, and the idea of annotating popular Triton kernels for learning.
- **Endorsement for Triton Resources**: Multiple members responded favorably towards the Triton puzzles and the idea of running interpreters on CPU, mentioning they would explore these resources. One response included a minor textual correction for the shared content.

**Link mentioned**: <a href="https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing">Google Colaboratory</a>: no description found

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1218467001450627072)** (68 messages🔥🔥): 

- **CUDA Warp Scheduler Inquiry**: A member inquired about how to define the number of **warp schedulers** and the number of **threads each warp scheduler controls**, aiming to understand the total number of threads that can run simultaneously to optimize efficiency and occupancy.
  
- **Active Warp Clarification Sought**: The term *active warp* was discussed with clarification sought on scenarios involving threads within a warp and how this impacts whether a warp is considered active. Examples from code were provided to illustrate points of confusion, such as whether a warp with no threads satisfying a condition still qualifies as active.

- **Memory Manager Abstraction Debated**: An extensive discussion unfolded about a **memory manager** in CUDA, exploring the semantics and practicalities of managing pointers for producers and consumers of data within the memory space. The concepts of **ProducerProvides, ConsumerTakes**, and more were debated, revealing concerns about async work and stream synchronization when optimizing memory usage in CUDA applications.

- **Reports from the Video-Pipeline Frontier**: One member showcased their work on optimizing a video pipeline with a focus on efficiently transferring data between producer and consumer memory spaces. There was a lively back-and-forth about the Manager class interface and the role of delays, async copies, and memory bottlenecks in pipeline parallelism.

- **Sharing CUDA Project Architecture Best Practices**: Questions and answers were exchanged regarding project structuring in CUDA, specifically whether the `main()` function should reside in a `.cpp` or a `.cu` file, and how to correctly include a kernel function from a `.cu` file. This led to a shared sentiment about the need for clear educational resources on proper CUDA project organization.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Y2F8yisiS6E">GTC March 2024 Keynote with NVIDIA CEO Jensen Huang</a>: Watch NVIDIA CEO Jensen Huang’s GTC keynote to catch all the announcements on AI advances that are shaping our future.Dive into the announcements and discove...</li><li><a href="https://github.com/tspeterkim/flash-attention-minimal">GitHub - tspeterkim/flash-attention-minimal: Flash Attention in ~100 lines of CUDA (forward pass only)</a>: Flash Attention in ~100 lines of CUDA (forward pass only) - tspeterkim/flash-attention-minimal
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1219091487455711414)** (5 messages): 

- **Exploring Reconfigurable Computing and ML**: A YouTube video titled "Prof. Mohamed Abdelfattah" and a website are shared, focusing on reconfigurable computing and efficient machine learning research by Prof. Abdelfattah's group at Cornell University. Viewers are invited to [explore their research](https://www.mohsaied.com/).

- **Hardware-Centric View of Machine Learning Systems**: Information about ECE 5545 (CS 5775), a hardware-centric machine learning course, is provided, covering topics like ML algorithm hardware/software, optimization techniques, and system design. Interested participants are encouraged to [read the syllabus](https://abdelfattah-class.github.io/ece5545/).

- **Textbook Mystery in Machine Learning Course**: A user points out that the referred website for ECE 5545 does not specify what "the textbook" for the course is, stating it as "weird".

- **Solving the Textbook Puzzle**: In response to the textbook query, it’s mentioned that the first lecture video of the course reveals the textbook information, highlighting the importance of supplementary course material.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://abdelfattah-class.github.io/ece5545/">ML Hardware and Systems</a>: no description found</li><li><a href="https://www.youtube.com/@mabdelfattah88">Prof. Mohamed Abdelfattah</a>: This is the channel for Prof. Mohamed Abdelfattah&#39;s research group at Cornell University. We are researching reconfigurable computing and efficient machine learning. For more information check out...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages): 

vim410: Depends. But yes.
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1219389682241110147)** (5 messages): 

- **Solid CUDA Foundation, Ready for ML**: andreaskoepf acknowledged al0vya's solid foundation in CUDA and recommended playing with a deep learning framework like **torch** to get started with ML/DL, as it typically involves *matrix multiplications, pointwise non-linearities, softmax, and normalization*.
- **Book Recommendation for CUDA Mastery**: andreaskoepf suggested getting the book "Programming Massively Parallel Processors" for more in-depth CUDA knowledge and added that while it has minor DL content, it remains an *excellent general CUDA programming book*. [Programming Massively Parallel Processors on Amazon](https://www.amazon.de/-/en/Wen-mei-W-Hwu/dp/0323912311).

**Link mentioned**: <a href="https://www.amazon.de/-/en/Wen-mei-W-Hwu/dp/0323912311">no title found</a>: no description found

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1218146385942286407)** (6 messages): 

- **CUDA Indexing Confusion Cleared Up**: A member questioned the indexing expression `i = blockIdx.x * blockDim.x + threadIdx.x * 2`, resulting in a clarification that this calculation could cause **double-counting** of indexes among threads. It was exemplified that two different threads could end up being assigned the same index.
- **Blogging Exercise Solutions Considered**: A member inquired about the potential issues of **blogging exercise solutions** to the CUDA book exercises, expressing difficulty in contacting the authors and a sense of loss from not having an educational email address after graduation.
- **Seeking Permission for Public Content**: Following a caution that some content might be **instructor only**, another member responded saying they will check with **Wen-mei**, presumably one of the authors, to clarify if it's acceptable to publicly share exercise solutions.
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1218239914542366790)** (14 messages🔥): 

- **Team Member Apologizes for Busy Schedule**: One of the chat participants expressed they were very busy and would notify the group when their schedule cleared up.
- **Member Expresses Difficulty Finding Code**: A member indicated they were unable to find specific code, and another team member provided a link to a [Triton kernel commit](https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5) to assist.
- **Seeking Clarity on Ring Attention Memory Requirements**: A member was writing a blog post and needed clarification on the memory requirements of ring attention versus flash attention, especially in terms of linear memory scaling relative to block size.
- **Recommendation to Read a Paper for Insights**: To better understand the performance characteristics of Ring Attention, it was suggested to read an [arXiv paper on Striped Attention](https://arxiv.org/abs/2311.09431), which includes helpful visuals.
- **Debate over Flash Attention's Memory Footprint**: The discussion continued with various members debating whether the memory requirements for Flash Attention indeed scale linearly with the block size c², including a reference to [flash attention's implementation on GitHub](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2311.09431">Striped Attention: Faster Ring Attention for Causal Transformers</a>: To help address the growing demand for ever-longer sequence lengths in transformer models, Liu et al. recently proposed Ring Attention, an exact attention algorithm capable of overcoming per-device me...</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h">flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h at main · Dao-AILab/flash-attention</a>: Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.</li><li><a href="https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5">add naive triton kernel for varlen · zhuzilin/ring-flash-attention@10d992c</a>: no description found
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1218332053032927322)** (5 messages): 

- **AI Meets Systems at MLSys 2024**: Members discussed the upcoming MLSys 2024 conference in May, highlighting its interdisciplinary nature at the intersection of Machine Learning and Systems. The conference is framed as essential for addressing future challenges in the AI landscape, with a particular focus on holistic approaches ([MLSys Conference](https://mlsys.org/)).

- **When Phones Are Not Too Bright**: A humorous remark labeled smartphones as "Not so smart phone", though no context was provided to understand the underlying issue or topic being referenced.

- **Calculator Conundrum Sparks Debate**: Members debated over the correct way to perform calculations, suggesting that the sequence in which multiplication and division are carried out matters, while another noted that scientific calculators may process `ax` and `a×x` differently. No specific examples or further explanations were provided.

**Link mentioned**: <a href="https://mlsys.org/">MLSys 2024</a>: no description found

  

---


**CUDA MODE ▷ #[gtc-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1218444664315711498)** (9 messages🔥): 

- **GTC 2023 Meetup Plans Unveiled**: One member is planning to be at GTC on Monday morning, openly inviting others to meet up and offering to share their phone number via DM.
- **Event Enthusiasts Set Dates**: Another member has announced they will be attending the event from the 14th to the 25th of March and is open to meeting up during the event dates.
- **Extended Visit After Seeing Schedule**: Excitement for the conference's schedule has led one member to consider attending for the entire week, contingent on the availability of decent wifi.
- **GTC Meme Humor**: A member humorously suggests there should be a meme about not being able to attend GTC.
- **Volunteer Hopes Dashed**: One expressed disappointment for having reached out to volunteer at GTC for a free pass without success.
- **The Ideal Infiltration Tactics?**: Following a mention of needing another way to access GTC, a member shared a link to a [YouTube video](https://www.youtube.com/watch?v=Sfrjpy5cJCs) titled "I Snuck Into A Secret Arms-Dealer Conference," humorously insinuating an unorthodox method of attending conferences.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=Sfrjpy5cJCs">I Snuck Into A Secret Arms-Dealer Conference</a>: Get an exclusive video every month at https://www.patreon.com/Boy_BoyWe made this in collaboration with the legendary Australian political satire group The C...

  

---



**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1218183723200155748)** (159 messages🔥🔥): 

- **LLM Format Flexibility Confirmed**: A quick affirmation was provided that LLaMa models can employ a prompt format including "system", "user", and "assistant" roles, relevant for users of the OpenAI JavaScript library.
  
- **Balancing the Books**: One user explained the creation of a script that takes books, breaks them down, and prompts a model to generate segments accordingly. Airoboros 70B was used, with comparisons made to lzlv 70B and an observation that instruction-based data can improve generative quality.

- **In Search of Detailed Analytics**: Users expressed a need for detailed usage analytics similar to OpenAI’s offering, showing a demand for daily or weekly usage costs and possibly a breakdown by models and apps.

- **Model Moderation and Access Queries**: Users report changes in a model's willingness to perform tasks and inquire about current access issues to sonnet:beta and opus:beta through the API, with the company confirming accessibility for most.

- **Potential New API Listing**: A user indicated they are setting up their own public API and inquired about having it listed on OpenRouter, to which the official response was open and inviting further details via direct message. 

- **Discussions on Model Costs and Performance**: There were discussions about the costs of using different models, such as Claude 3 Opus versus others like Sonnet, with users exchanging views on the affordability and performance of these AI models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai">OpenRouter</a>: A router for LLMs and other AI models</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1218212402127175711)** (95 messages🔥🔥): 

- **Query on Streaming APIs**: A user questioned the difference between **astream_log** and **astream_events**, asking if **astream_log** might be deprecated in favor of the beta **astream_events**, or if they are simply two APIs with distinct use cases.

- **Beta Testers Wanted for Advanced Research Assistant**: An invitation was extended for beta testers for an advanced research assistant called **Rubik's AI**. Interested users can join the waitlist for access to premium features like **Claude 3 Opus**, **GPT-4 Turbo**, and **Mistral Large**, via [Rubik's AI](https://rubiks.ai/).

- **Feedback and Suggestions for LangChain Documentation**: One user expressed difficulty navigating LangChain documentation, particularly for beginners. A response invited specific feedback on confusing pages or suggestions for missing content.

- **Structured Output with LLM Using LangChain**: A user inquired on how to get structured outputs from LLMs using LangChain, such as listing cities with populations. A detailed code example was provided using **PydanticOutputParser** to define the desired output structure.

- **Function Calls with Google Gemini Through LangChain**: A discussion emerged about how to make the Gemini model on Vertex AI aware of the existence of functions through LangChain, enabling the LLM to call a function in response to a query. The conversation included the use of `.bind(functions=[schema])` to pass function schemas to the LLM.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rubiks.ai/">Rubik's AI - Waitlist</a>: no description found</li><li><a href="https://codelabs.developers.google.com/codelabs/gemini-function-calling#4.">no title found</a>: no description found</li><li><a href="https://bloon.ai">Bloon AI</a>: Redefining Intelligent Learning</li><li><a href="https://github.com/langchain-ai/langchain/discussions/19239">Feature Request: Support for Negative Embeddings in Similarity Searches · langchain-ai/langchain · Discussion #19239</a>: Checked I searched existing ideas and did not find a similar one I added a very descriptive title I&#39;ve clearly described the feature request and motivation for it Feature request I propose adding ...</li><li><a href="https://www.teradata.com/insights/ai-and-machine-learning/using-natural-language-to-query-teradata-vantagecloud-with-llms">Using Natural Language to Query Teradata VantageCloud With LLMs| Teradata</a>: Learn to translate your English queries into SQL and receive responses from your analytic database in plain English.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1219304272244510741)** (45 messages🔥): 

- **Trouble with `RemoteRunnable` Streaming in JavaScript**: A user faced challenges with streaming output through `RemoteRunnable` when working with JavaScript. While it functioned correctly in Python, the same code would downgrade to `/invoke` in JavaScript instead of calling `/stream`.
- **Streaming Mechanism Clarity Requested**: The user sought clarity on why streaming was not functioning as expected, questioning if `RunnableSequence` inheriting `_streamIterator from Runnable`, which calls `invoke`, could be the issue.
- **Looking for Support from LangChain Team**: The user inquired about how to reach out to the LangChain team regarding the streaming issue. The AI suggested reporting the issue on GitHub or reaching out via email as per the Security Reporting Guidelines.
- **No Known Fixes in Recent Updates**: There was no information provided about any recent changes that could have resolved the streaming problem. The AI recommended checking the LangChain GitHub repository for the latest updates.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://js.langchain.com/docs/security#reporting-a-vulnerability>).">Security | 🦜️🔗 Langchain</a>: LangChain has a large ecosystem of integrations with various external resources like local and remote file systems, APIs and databases. These integrations allow developers to create versatile applicat...</li><li><a href="https://api.js.langchain.com/classes/langchain_core_runnables_remote.RemoteRunnable.html#pipe>):">RemoteRunnable | LangChain.js - v0.1.28</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/13126>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13126>)),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/11998>)),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13723>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/17315>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1218223379690029179)** (11 messages🔥): 

- **New AI Chatbot for Data Analysis**: A user shared a link to [Haste171/langchain-chatbot](https://github.com/Haste171/langchain-chatbot) on GitHub, which is an AI chatbot designed for analyzing and extracting information from data in a conversational format.
- **Bookmark Management with AI**: [Living Bookmarks](https://twitter.com/uogbuji/status/1768681648516661446), released as open source on GitHub, is a Discord AI chatbot that interacts with Raindrop.io bookmarks to help users find them when relevant.
- **Seeking Productivity Insight**: A user is building a digital advisor and invited tech and professional services workers to discuss productivity, and physical and mental health needs, offering [30-minute consultation slots](https://calendly.com/neurofusion/30min).
- **AI-based Scraper Gets Popularity**: The [Scrapegraph-ai](https://github.com/VinciGit00/Scrapegraph-ai), an AI-based scraper built with LangChain, has been released on pip with over 2300 installations, encouraging users to star the project for support.
- **AI Solution Simulates Sales Roles**: A Twitter post details how **Lyzr.ai's Automata** simulates SDR and AE functions, from processing email lists to closing sales with the help of multiple AI agents and tools like OpenAI and *Perplexity*. The project repository is available on [GitHub](https://github.com/LyzrCore/lyzr-automata).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://calendly.com/neurofusion/30min">User Interview 🔎 - NEUROFUSION Research, Inc.</a>: Hey, I&#39;m building a digital advisor to help improve how you show up to work and other areas of your life. I&#39;d love to speak with you to learn about your needs around productivity, physical and...</li><li><a href="https://github.com/Haste171/langchain-chatbot">GitHub - Haste171/langchain-chatbot: AI Chatbot for analyzing/extracting information from data in conversational format.</a>: AI Chatbot for analyzing/extracting information from data in conversational format. - Haste171/langchain-chatbot</li><li><a href="https://github.com/VinciGit00/Scrapegraph-ai">GitHub - VinciGit00/Scrapegraph-ai: Python scraper based on AI</a>: Python scraper based on AI. Contribute to VinciGit00/Scrapegraph-ai development by creating an account on GitHub.</li><li><a href="https://youtu.be/vHjc5CEoIJE">Making an AI application in 15 minutes</a>: Stack- Custom UI and RAG: A tweaked version of open-webui.- Local LLM Hosting: Ollama for locally hosted LLMs.- Data Privacy: Integrates Pebblo by DaxaAI to ...</li><li><a href="https://navvy.co/.">Home</a>: I’m deeply passionate about AI. Let’s connect to unlock AI’s potential and collaborate on innovative projects!</li><li><a href="https://x.com/siva_1gc/status/1768997890544800070?s=20">Tweet from Siva Surendira (@siva_1gc)</a>: It took a bit more time than we thought.. But here it is.. 😎  Automation of SDR & AE function with @lyzrai Automata and @OpenAI... Runs on @awscloud - secure and private..  How it works? 👇  Agent 1:...</li><li><a href="https://github.com/LyzrCore/lyzr-automata">GitHub - LyzrCore/lyzr-automata: low-code multi-agent automation framework</a>: low-code multi-agent automation framework. Contribute to LyzrCore/lyzr-automata development by creating an account on GitHub.</li><li><a href="https://amzn.eu/d/3Dcdsbk">no title found</a>: no description found</li><li><a href="https://amzn.eu/d/2uVnCp8">no title found</a>: no description found</li><li><a href="https://www.facebook.com/casi.schulze.10">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1218824643436085321)** (2 messages): 

- **Personalized Nutrition AI Demo**: A personalized nutrition AI app, **Nutriheal**, has been showcased as using tools like **Ollama** and **Open-webui**, with privacy integration through **Langchain's Pebblo** by Daxa AI. A *YouTube video tutorial* explains how to create such an application in 15 minutes, emphasizing user-friendliness and data protection. [Watch the video here](https://youtu.be/vHjc5CEoIJE).

- **Discover How to Build AI Locally**: The tutorial also promotes guides on building and deploying AI solutions locally, shattering the myth that only large tech companies can handle AI. These resources aim to simplify the setup and execution of sophisticated AI models for individual users. [Read the guide here](//build-and-deploy-genai-solutions-locally).

- **Generic UI for AI Chat Assistants**: Another available resource discusses creating a generic chat UI for custom LLM (Large Language Model) assistants, indicating a focus on reusable interfaces for different AI solutions. It implies a wider application and ease of integration for personal AI development. [Find the UI guide here](/generic-ui-for-custom-llm-assistants). 

- **Plan-and-Execute with Langgraph Tutorial**: An educational video has been shared on creating a "plan-and-execute" style AI agent inspired by the Plan-and-Solve paper and the Baby-AGI project. The core goal is to emulate strategic planning and execution in AI agents. [See the tutorial here](https://www.youtube.com/watch?v=ZlJbaYQ2hm4).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=ZlJbaYQ2hm4">Plan-and-Execute using Langgraph</a>: how to create a &quot;plan-and-execute&quot; style agent. This is heavily inspired by the Plan-and-Solve paper as well as the Baby-AGI project.The core idea is to firs...</li><li><a href="https://youtu.be/vHjc5CEoIJE">Making an AI application in 15 minutes</a>: Stack- Custom UI and RAG: A tweaked version of open-webui.- Local LLM Hosting: Ollama for locally hosted LLMs.- Data Privacy: Integrates Pebblo by DaxaAI to ...</li><li><a href="https://navvy.co/.">Home</a>: I’m deeply passionate about AI. Let’s connect to unlock AI’s potential and collaborate on innovative projects!
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1218217772765544448)** (8 messages🔥): 

- **Revealing Model Secrets via API Queries**: A link to an [arXiv paper](https://arxiv.org/abs/2403.09539) explores the possibility of learning non-public information about API-protected large language models (LLMs) like OpenAI's gpt-3.5-turbo using API queries. The paper highlights a "softmax bottleneck," which could reveal the model's hidden size and other details.

- **Model Size Estimation Exposed**: A member discussed another paper from Carlini and others that used logits to estimate model size but redacted those details, remarking that the current paper performs a similar analysis without redactions.

- **Surprise at a 7B Model Size Finding**: One member expressed surprise at the paper's suggestion that a certain model might be only 7B in size.

- **Inaccuracy Speculations on Model Size**: Another member posited skepticism about the 7B model size finding, suggesting it might be inaccurate unless there exists some advanced distillation method.

- **Misleading Model Size Estimates with MoEs**: The discussion touched upon potential inaccuracies in model size calculations if the model in question uses a Mixture of Experts (MoE), noting that a model like Mistral already has a substantial embedding dimension.

**Link mentioned**: <a href="https://arxiv.org/abs/2403.09539">Logits of API-Protected LLMs Leak Proprietary Information</a>: The commercialization of large language models (LLMs) has led to the common practice of high-level API-only access to proprietary models. In this work, we show that even with a conservative assumption...

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1219339209270362135)** (19 messages🔥): 

- **Anticipating ML Drama**: A tweet shared in the chat predicts potential drama following a Twitter [exchange on open-source definitions](https://twitter.com/rasbt/status/1769779229263065184).
- **Seeking OSS Clarity**: Chat members express interest in the open-source software (OSS) community arriving at a clear stance regarding what constitutes open source, aiming to end ongoing debates.
- **Critique on Data Exclusion in Open Source**: There is a sentiment that excluding **data** from the open-source definition is a poor decision, with members already dissatisfied with the potential stance.
- **Defining the Practicalities of Open Source**: Efforts are being made to establish a practical definition of open source to pacify contentious discussions and reach a common understanding.
- **Frustrations with Online Engagement**: A user expresses frustration with EleutherAI's approach to online discourse, implying it can be counterproductive, and mentions an intention to avoid Twitter and focus on blogging.

**Link mentioned**: <a href="https://x.com/BlancheMinerva/status/1769792488091353099">Tweet from Stella Biderman (@BlancheMinerva)</a>: @natolambert @felix_red_panda You&#39;re wrong though :P

  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1219005089826607185)** (63 messages🔥🔥): 

- **Grok-1 Released to the Public**: xAI has announced the release of [Grok-1](https://x.ai/blog/grok-os), a **314 billion parameter Mixture-of-Experts model** with a custom training stack on top of JAX and Rust. The model weights and architecture are available under the Apache 2.0 license at [github.com/xai-org/grok](https://github.com/xai-org/grok).
- **Grok-1 Model Details Debated**: Chat participants questioned the performance and release strategy of **Grok-1**, suggesting it might be "undercooked" or hastily released. The discussion also touched on the marketing of such models and the significance of their distribution methods.
- **Comparison with Falcon**: Speculation arose regarding Grok's performance, with claims that Grok seems to outperform the Falcon model based on given GSM8K (45.94) and MMLU (70.5) benchmark scores.
- **Concerns Over Model Distribution Via Torrents**: The distribution of Grok via torrents prompted debates on its implications for open AI teams and policymaking, with some suggesting it could affect the credibility and policy support for open-source models.
- **Humorous Suggestion of Model Distribution by Mail**: A humorous debate sparked about the cost-effectiveness of distributing heavy AI models via FedEx flash drives, satirically proposing a "mail-order models business" as an alternative to traditional online egress costs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-os">Open Release of Grok-1</a>: no description found</li><li><a href="https://www.wheresyoured.at/peakai/">Have We Reached Peak AI?</a>: Last week, the Wall Street Journal published a 10-minute-long interview with OpenAI CTO Mira Murati, with journalist Joanna Stern asking a series of thoughtful yet straightforward questions that Murat...</li><li><a href="https://x.com/thexeophon/status/1769449427972858103?s=46">Tweet from Xeophon (@TheXeophon)</a>: Chinchilla doesn’t apply to MoE directly, does it? If it does, we can infer the training data set size for Grok. It’s unexpectedly large, so I guess they went for optimality first, given the little ti...</li><li><a href="https://fxtwitter.com/grok/status/1769441648910479423">Tweet from Grok (@grok)</a>: @elonmusk @xai ░W░E░I░G░H░T░S░I░N░B░I░O░
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1218732428462395502)** (6 messages): 

- **Seeking Clarity on Aribus Development**: A member inquired about what others are developing with **Aribus**, accompanied by a [Twitter link](https://twitter.com/alignment_lab/status/1758949148143841379) that they found confusing. No further details or clarifications were provided in the subsequent messages.
- **Hunt for HTTP-Aware Embeddings Model**: Someone expressed interest in finding an embeddings model trained specifically on **HTTP responses** and sought guidance on where to start the search. They also mentioned the possibility of using any transformer model as an embedding model provided it has the right training.
- **Looking for Mistral with Special Training**: A member is in search of a **Mistral model** that has been fine-tuned (FT) on both the *orca-math-word-problems-200k dataset* and *nvidia/OpenMathInstruct-1*. No follow-up information or suggestions were shared.
- **Short and Sweet Greeting**: A user simply entered the chat with a brief "hi". There was no substantive discussion following this greeting.
  

---


**Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1219081302683422851)** (32 messages🔥): 

- **Call for Fine-Tuning Collaboration on Grok 1**: A member seeks collaboration for fine-tuning **Grok 1**, a large, possibly undertrained model, highlighting the need for substantial **compute** and **data** resources. They suggest that an existing MoE training infrastructure is already in place.
- **Potential Issues with Benchmark Performance of Grok 1**: A discussion revealed concerns about **Grok 1's** performance on the MMLU benchmark, with members suggesting the need for more compute power and continuous pretraining on diverse datasets. There is curiosity around the model's capabilities compared to other models like **Mixtral**.
- **Debate on Model's Value and Cost-Efficiency**: There's skepticism regarding the cost-efficiency of further training **Grok 1** when compared to other models, and questions about whether it could become the best open-source LLM or outperform models like **GPT-4** and **Claude**.
- **Data Set Curiosity and Jax Expertise**: Participants are exploring the ideal data mix for fine-tuning and confirmed the participation of a self-identified **Jax expert**. The specifics of data requirements and the benefits of training efforts were points of discussion.
- **Grok 1's Unexpected Performance**: A member pointed to **Grok 1** exhibiting surprising capabilities in a [held-out high school finals exam](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam), mentioning its close performance to **GPT-4** and **Claude** on this specific exam.

**Link mentioned**: <a href="https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam">keirp/hungarian_national_hs_finals_exam · Datasets at Hugging Face</a>: no description found

  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1218226914322415677)** (1 messages): 

- **Devin Inspires Lazy App Development**: A member expressed how **Devin** has motivated them to be "too lazy to even paste things into terminal" for building simple apps. They believe anything more complex than local apps is overkill and questioned the effectiveness of current open-source solutions.
  

---


**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1218206756031955006)** (7 messages): 

- **Fear of Algorithmic Overlords**: A [tweet](https://x.com/tszzl/status/1768530219378631137?s=20) was shared suggesting that **Anthropic** might be acting as controlled opposition to instill fear among technical staff.
- **Smooth Moderating Except for Human Images**: Regarding content moderation, the member has not encountered issues except with images that contain people, where sometimes "**it just refuses**."
- **Exploring Claude Sonnet for High Volume Use**: A member is considering using **Claude Sonnet** for a project expecting usage of several dozen million tokens per month and is inquiring about experiences at such scale.

**Link mentioned**: <a href="https://x.com/tszzl/status/1768530219378631137?s=20">Tweet from roon (@tszzl)</a>: anthropic is controlled opposition to put the fear of god in the members of technical staff

  

---


**LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/1218241222347460619)** (16 messages🔥): 

- **KPU Unveiled As New Solution for LLMs**: Maisa introduces the [Knowledge Processing Unit (KPU)](https://maisa.ai/blog/kpu), a framework claimed to outperform advanced language models like GPT-4. It separates reasoning from data processing within an AI system to enhance complex task-solving capabilities.
- **Benchmarking Confusion Over KPU**: Discussion arises on why **KPU**+GPT-4-turbo is compared to just GPT-4 instead of GPT-4-turbo, suggesting that the latter would be a more appropriate benchmarking comparison.
- **Deciphering the Tech Behind KPU**: There is some confusion and humor around the actual technology of KPU, with it seeming to involve a combination of self-evaluation and "clever context window tricks," rather than being a new model.
- **Concerns Over Practicality and Performance**: A member questions whether improvements like a 6% increase on MATH by KPU are practical, considering unreported latency that could negatively impact product integration.
- **KPU Explained by CEO**: Maisa's CEO clarifies via a [Twitter post by @davipar](https://x.com/davipar/status/1768683151780683919?s=20) that KPU is not a new model, but an architecture working with existing LLMs to optimize knowledge management, promising cost savings and improved performance with a "virtual context window."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://maisa.ai/blog/kpu">KPU - Maisa</a>: AI-Powered Knowledge Processing Platform. A simple API for executing business tasks. Abstracting the complexities of using the latest AI architectures for software and app developers</li><li><a href="https://x.com/davipar/status/1768683151780683919?s=20">Tweet from David Villalón (@davipar)</a>: happy to answer! it is not a new model, indeed KPU is agnostic to intelligence providers (OpenAI, Antrophic...). It is a new AI architecture to work with LLMs that leverages their reasoning capabiliti...
</li>
</ul>

</div>
  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/)** (1 messages): 

res6969: https://x.com/leopoldasch/status/1768868127138549841?s=46
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1218132499150934157)** (21 messages🔥): 

- **Difficulty with German Response Generation for Various Models**: A user experienced trouble generating German responses with the **DiscoLM-mixtral-8x7b-v2** model after instruction fine-tuning, whereas multiple other models yielded acceptable performance. A related issue was a `ValueError` exception when trying to use AutoModel for sequence classification, suggesting possibly unrecognized or unsupported configuration classes.
  
- **Assistance with Grok**: A GitHub link to the Grok model ([Grok open release](https://github.com/xai-org/grok/blob/main/model.py)) was shared, with users discussing the feasibility of running the model due to its large size (314 billion parameters requiring substantial computation resources).

- **German Language Model Challenges and Approaches**: User discussions revealed insights on merging language models for German, quality of datasets for fine-tuning, and the importance of using consistent prompt formats to maintain language output quality. The conversation highlighted challenges in preserving language quality when merging models, and the prospect of community collaboration to improve German language models.

- **Benchmarking Multilingual and German Models**: References were made to various benchmarks and benchmarks-in-disguise like the supergleber-german-language-evaluation-benchmark, with links to papers and Reddit posts for further details. Contributors discussed the potential of adding German-specific benchmarks to platforms like EleutherAI's lm-evaluation-harness and the need for benchmarks measuring language quality as perceived by native speakers.

- **Leveraging Universities for Research in Language Quality**: There was a suggestion to leverage university resources to research and develop benchmarks that assess language quality, with the indication that public-funded German universities could support such initiatives. This was mentioned in the context of the *DiscoLM* project, stressing the potential benefits of academic collaboration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1bfce18/still_didnt_found_a_better_small_german_llm_anyone/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bfce18/still_did">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/xai-org/grok/blob/main/model.py">grok-1/model.py at main · xai-org/grok-1</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://github.com/xai-org/grok/blob/e50578b5f50e4c10c6e7cff31af1ef2bedb3beb8/model.py#L294">grok-1/model.py at e50578b5f50e4c10c6e7cff31af1ef2bedb3beb8 · xai-org/grok-1</a>: Grok open release. Contribute to xai-org/grok-1 development by creating an account on GitHub.</li><li><a href="https://www.informatik.uni-wuerzburg.de/datascience/news/single/news/our-paper-supergleber-german-language-understanding-evaluation-benchmark-was-accepted-at-the-naacl-2024/">Our Paper &quot;SuperGLEBer: German Language Understanding Evaluation Benchmark&quot; was accepted at the NAACL 2024</a>: In our paper, we assemble a broad Natural Language Understanding benchmark suite for the German language and consequently evaluate a wide array of existing German-capable models in order to create a b...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1b5vp2e/llm_comparisontest_17_new_models_64_total_ranked/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets/ChuckMcSneed/WolframRavenwolfs_benchmark_results">ChuckMcSneed/WolframRavenwolfs_benchmark_results · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/KLUE-benchmark/KLUE">GitHub - KLUE-benchmark/KLUE: 📖  Korean NLU Benchmark</a>: 📖  Korean NLU Benchmark. Contribute to KLUE-benchmark/KLUE development by creating an account on GitHub.</li><li><a href="https://github.com/facebookresearch/belebele">GitHub - facebookresearch/belebele: Repo for the Belebele dataset, a massively multilingual reading comprehension dataset.</a>: Repo for the Belebele dataset, a massively multilingual reading comprehension dataset. - facebookresearch/belebele</li><li><a href="https://github.com/google-research/xtreme">GitHub - google-research/xtreme: XTREME is a benchmark for the evaluation of the cross-lingual generalization ability of pre-trained multilingual models that covers 40 typologically diverse languages and includes nine tasks.</a>: XTREME is a benchmark for the evaluation of the cross-lingual generalization ability of pre-trained multilingual models that covers 40 typologically diverse languages and includes nine tasks. - goo...
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1218111377495949322)** (4 messages): 

- **No Special Settings for Demo**: *jp1* clarified that for the demo, neither special settings nor adjustments are generally needed, and they are currently utilizing **fastchat/VLLM** by default.

- **Demo Server Relocated**: *jp1* informed that the server which was used for demo purposes has been moved from a personal kitchen setting to a more official location. However, there have been some unexpected networking issues, which they hope to resolve by early next week.

- **Downside of Professional Hosting**: *chromix* humorously compared the reliability of a hobbyist server in his kitchen corner with a professionally hosted server, which seems to experience a variety of technical issues including networking problems and spontaneous SAN failures.
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1218229369680695428)** (20 messages🔥): 

- **Prodigy Introduces Prompt Engineering Tools**: A former Explosion employee highlighted that some prompt engineering tools they developed are now part of Prodigy's paid product. The tool aims to turn prompt engineering into a data annotation problem and can be seen on [Prodigy's feature page](https://prodi.gy/features/prompt-engineering).

- **Prompt Testing Made Easier with Open Source Tools**: Members shared various resources for prompt testing and experimentation, including the repos [PromptTools by hegelai](https://github.com/hegelai/prompttools) and [PromptFoo](https://github.com/promptfoo/promptfoo), which offer support for a range of LLMs and vector databases.

- **Vercel and Helicone.ai for Model Comparisons and Prompt Management**: The Vercel [AI Playground](https://sdk.vercel.ai/) was mentioned as a useful interface for comparing models with a single prompt, while Helicone.ai was recognized for its budding capabilities in prompt management and versioning.

- **Experimenting with AI-Enhanced Blog Customization**: A member is piloting a project to "translate" blog posts into various personas using GPT-3.5-turbo, hinting at potential tools to improve writing clarity and focus, and shared a link to a live example: [How to Build a Buzzword](https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html).

- **Discussion on Blogging with AI-Augmented Actions**: Ideas were exchanged about how AI could enrich blogging platforms, suggesting functionalities like rewriting from different personas, providing counterpoints, offering persona-based social sharing, and generating summaries or translations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html">How to Build a Buzzword</a>: And why they’re so powerful</li><li><a href="https://www.helicone.ai/">Helicone</a>: How developers build AI applications. Get observability, tooling, fine-tuning, and evaluations out of the box. </li><li><a href="https://sdk.vercel.ai/">Vercel AI SDK</a>: Build AI-powered applications with the latest AI language models</li><li><a href="https://github.com/hegelai/prompttools">GitHub - hegelai/prompttools: Open-source tools for prompt testing and experimentation, with support for both LLMs (e.g. OpenAI, LLaMA) and vector databases (e.g. Chroma, Weaviate, LanceDB).</a>: Open-source tools for prompt testing and experimentation, with support for both LLMs (e.g. OpenAI, LLaMA) and vector databases (e.g. Chroma, Weaviate, LanceDB). - hegelai/prompttools</li><li><a href="https://github.com/promptfoo/promptfoo">GitHub - promptfoo/promptfoo: Test your prompts, models, RAGs. Evaluate and compare LLM outputs, catch regressions, and improve prompt quality. LLM evals for OpenAI/Azure GPT, Anthropic Claude, VertexAI Gemini, Ollama, Local &amp; private models like Mistral/Mixtral/Llama with CI/CD</a>: Test your prompts, models, RAGs. Evaluate and compare LLM outputs, catch regressions, and improve prompt quality. LLM evals for OpenAI/Azure GPT, Anthropic Claude, VertexAI Gemini, Ollama, Local &amp;...
</li>
</ul>

</div>
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/)** (1 messages): 

obra: Is it possible to recover the seed used by the openai models for a previous api request?
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1218193382669549568)** (17 messages🔥): 

- **Pending Release of a Model Improvement Method**: A member indicated they are structuring results for a method that seems to improve **global accuracy** and make training **more sample efficient**. They promised to release a paper/article once better charts and structured results are ready.
- **Seeking Resources for Scaling to Large Models**: The discussion revealed that while some validation exists, empirical proof of the method's effectiveness on large-scale models is lacking due to resource constraints. The member expressed a need for resources to pursue this validation.
- **Offer to Discuss and Scale Method**: There was an offer to jump on a call to discuss previously mentioned methods and possibly help allocate **compute and resources** to scale the method up.
- **Improvement Evident in Subset Experiments**: The member mentioned their method yielded a **higher test accuracy** on a subset of CIFAR100 when used with VGG16 for 1 epoch, citing specific accuracy figures to highlight improvement.
- **Exploring Ways to Improve Graph Reporting**: There were comments about issues with updating charts on Wandb, the platform used for reporting experimental results, specifically how to reset steps when plotting new data.
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=ZlJbaYQ2hm4
  


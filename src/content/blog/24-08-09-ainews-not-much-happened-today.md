---
id: 6d6bf91e-0eae-41b6-9792-5162e266a599
title: not much happened today
date: '2024-08-10T05:51:12.824379Z'
original_slug: ainews-to-be-named-3898
description: >-
  Qwen2-Math-72B has outperformed leading models like GPT-4o and
  Claude-3.5-Sonnet on various math benchmarks, showcasing advances in
  specialized training techniques. Additionally, Google AI has significantly cut
  prices for its Gemini 1.5 Flash model, reducing input costs by 78% and output
  costs by 71%, making it more accessible for developers.
tags:
  - qwen2
  - qwen2-math-72b
  - gpt-4o
  - claude-3.5-sonnet
  - google
  - gemini-1.5-flash
  - pricing
---


<!-- buttondown-editor-mode: plaintext -->**A quiet week is all you need.**

> AI News for 8/8/2024-8/9/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**249** channels, and **2549** messages) for you. Estimated reading time saved (at 200wpm): **278 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Unlike most newswires we do not seek to/have to fill pages with stuff when there isn't much going on. The biggest news this week was price cuts and structured outputs. Congrats to Cursor AI for announcing their [$60m Series A](https://techcrunch.com/2024/08/09/anysphere-a-github-copilot-rival-has-raised-60m-series-a-at-400m-valuation-from-a16z-thrive-sources-say/). We have been big fans of [Composer](https://x.com/shaoruu/status/1812412514350858634).

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

**AI Model Updates and Developments**

- **Qwen2-Math Models**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821615977332682929) reported that Qwen2-Math-72B outperformed GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, and Llama-3.1-405B on various math benchmarks. The models are based on Qwen2 and trained on math web text, books, exams, and codes, utilizing synthetic data and advanced techniques like rejection sampling and group relative policy optimization.

- **Google AI Pricing**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821653314066399688) shared that Google AI has significantly reduced pricing for Gemini 1.5 Flash, cutting input prices by 78% to $0.075/1 million tokens and output prices by 71% to $0.3/1 million tokens for prompts under 128K tokens.

- **Anthropic Bug Bounty Program**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1821533729765913011) announced an expansion of their bug bounty program, focusing on finding universal jailbreaks in their next-generation safety system. They're offering rewards for novel vulnerabilities across various domains, including cybersecurity.

- **IDEFICS3-Llama Fine-tuning**: [@mervenoyann](https://twitter.com/mervenoyann/status/1821605881815147004) shared a new tutorial on QLoRA fine-tuning IDEFICS3-Llama 8B on VQAv2, demonstrating efficient fine-tuning techniques for visual question answering.

**AI Research and Benchmarks**

- **Chinese Open Weights Model**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1821557546102174194) mentioned a Chinese open weights model that surpasses all previous models, both closed and open, at MATH benchmarks.

- **Mamba Survey**: [@omarsar0](https://twitter.com/omarsar0/status/1821556218168549561) shared a survey of Mamba, providing a systematic review of existing Mamba-based models across domains and tasks, focusing on advancements, adaptation techniques, and applications where Mamba excels.

- **LLM-based Agents for Software Engineering**: [@omarsar0](https://twitter.com/omarsar0/status/1821549401866686604) highlighted a survey paper on current practices and solutions for LLM-based agents in software engineering, covering topics like requirement engineering, code generation, and test generation.

**AI Tools and Platforms**

- **R2R RAG Engine**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821524844137091459) discussed R2R, an open-source RAG engine that simplifies the development of RAG applications, offering features like multimodal support, hybrid search, and automatic knowledge graph generation.

- **LlamaIndex Workflows**: [@llama_index](https://twitter.com/llama_index/status/1821575082516660440) introduced Workflows, a new abstraction for building complex agentic gen AI applications, demonstrating how to rebuild LlamaIndex's built-in Sub-Question Query Engine using this feature.

- **Mistral AI Agents**: [@sophiamyang](https://twitter.com/sophiamyang/status/1821476909345128740) announced the introduction of Mistral AI agents, allowing users to build agents based on Mistral models or fine-tuned models for use on Le Chat.

**AI Safety and Regulation**

- **California Bill SB 1047**: [@ylecun](https://twitter.com/ylecun/status/1821650987926339955) shared concerns expressed by Zoe Lofgren (Democratic member of the House) about California bill SB 1047, noting it's "heavily skewed toward addressing existential risk."

- **Open-source AI Debate**: [@bindureddy](https://twitter.com/bindureddy/status/1821656181833924752) initiated a discussion about banning open-source AI, highlighting the controversy surrounding such proposals.

**Memes and Humor**

- **Heavenbanning Day**: [@nearcyan](https://twitter.com/nearcyan/status/1821663160396607670) joked about "Heavenbanning Day" after two years, with a follow-up tweet clarifying that "heavenbanning isn't real because nothing ever happens."

- **Story Points Criticism**: [@svpino](https://twitter.com/svpino/status/1821528301061493215) shared a humorous critique of story points in Agile development, comparing them to the Emperor's New Clothes and calling the practice a "charade."

- **AI Compliments**: [@AmandaAskell](https://twitter.com/AmandaAskell/status/1821575345671446531) jokingly suggested tweeting compliments to future AIs to gain their favor.

This summary captures the key discussions in AI model developments, research, tools, safety, and regulation, along with some humorous takes on AI and software development practices.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Specialized AI Models for Mathematics and Technical Tasks**

- **Qwen2-Math | Math-specific model series based on Qwen2** ([Score: 73, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1en7tt8/qwen2math_mathspecific_model_series_based_on_qwen2/)): **Qwen** has released a series of **math-specific models** based on their **Qwen2** architecture, available on **Hugging Face**. The series includes models of various sizes (**72B**, **7B**, and **1.5B** parameters) in both base and instruct-tuned versions, aimed at enhancing mathematical reasoning capabilities.

- **Implemented LLaMA 3.1 8B's function calling from scratch, some challenges and feedback!** ([Score: 60, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1enr1ug/implemented_llama_31_8bs_function_calling_from/)): The author implemented **function calling** for **LLaMA 3.1 8B** using **LlamaCPP Python binding's generate() function**, noting challenges in separating custom function calls from dialogue. They observed that **small models** like LLaMA 3.1 8B struggle with tool usage without specific instructions, and expressed a preference for **YAML** over **JSON** for function calling due to token efficiency. The post concludes with the author considering developing a **REST server** to stream raw tokens or submitting a feature request for this functionality.
  - **YAML** is preferred over **JSON** for function calling due to token efficiency and readability. Users discussed prompting techniques to make models respond in YAML format, with the caveat that **LLaMA 3.1 8B** may struggle with complex instructions.
  - There's strong interest in an endpoint for generating **raw tokens** and **top 200 token distribution probabilities**, which could enable clever applications but is currently difficult to access from existing inference engines.
  - Users compared **Gemma2** to **LLaMA 3.1**, with some considering Gemma2 superior. However, it was noted that **Gemma2** doesn't currently support **function calling** in frameworks like **Ollama**, limiting its use for certain applications.


**Theme 2. Hugging Face's Strategic Expansion and Open-Source TTS Advancements**

- **[AI Unicorn Hugging Face Acquires A Startup To Eventually Host Hundreds Of Millions Of Models | Forbes](https://www.forbes.com/sites/richardnieva/2024/08/08/hugging-face-xethub-acquisition/)** ([Score: 200, Comments: 43](https://reddit.com//r/LocalLLaMA/comments/1en5p3h/ai_unicorn_hugging_face_acquires_a_startup_to/)): **Hugging Face**, an **AI unicorn** valued at **$4.5 billion**, has acquired **Paperspace**, a startup specializing in **AI infrastructure and cloud computing**. This acquisition aims to enhance Hugging Face's capabilities, allowing it to potentially host **hundreds of millions of AI models** and compete with major cloud providers like **Amazon**, **Google**, and **Microsoft**. The move is part of Hugging Face's strategy to become a comprehensive platform for AI development and deployment, offering services from model training to inference.

- **Improved Text to Speech model: Parler TTS v1 by Hugging Face** ([Score: 111, Comments: 35](https://reddit.com//r/LocalLLaMA/comments/1encx98/improved_text_to_speech_model_parler_tts_v1_by/)): **Hugging Face** has released **Parler TTS v1**, an improved open-source **Text-to-Speech model** available in **885M (Mini)** and **2.2B (Large)** versions. The model, trained on **45,000 hours** of open speech data, offers up to **4x faster generation**, supports **SDPA & Flash Attention 2** for speed boosts, includes in-built streaming, and allows for fine-tuning on custom datasets with improved speaker consistency across more than a dozen speakers.

**Theme 3. Emerging AI Models and Performance Benchmarks**

- **Shout out to Deepseek v2** ([Score: 56, Comments: 34](https://reddit.com//r/LocalLLaMA/comments/1enb0p2/shout_out_to_deepseek_v2/)): Deepseek v2, a **200 billion parameter open-source model**, has been praised for its performance in coding tasks, matching top models and ranking **#3 on BigCodeBench** alongside 3.5 Sonnet. The model's API offers competitive pricing with **cache hit rates at $0.017 per million tokens**, allowing the user to process **66 million input tokens for just $3.13**. Additionally, the model's efficiency suggests it can run locally on **quad 3090 GPU setups**, making it an attractive option for developers and researchers.

- **New sus-column-r model on LMSYS. It's just f** up** ([Score: 62, Comments: 49](https://reddit.com//r/LocalLLaMA/comments/1enmcr9/new_suscolumnr_model_on_lmsys_its_just_f_up/)): The **sus-column-r model** on LMSYS is reportedly outperforming **GPT-4** and **Claude 3.5 Sonnet** in various tasks including **translation**, **coding**, **mathematics**, and answering **rare questions**. The post author expresses disbelief at the model's capabilities, noting they would have assumed it was **GPT-5** if not for the model's self-identification response, and mentions a lack of information about **ColumnAI**, the apparent creators.
  - Users tested the **sus-column-r model** with "hard" prompts, finding it performed similarly to **GPT-4o**. Some expressed skepticism, requesting actual examples and reminding others of the "*We Have No Moat*" concept.
  - Discussion arose about the model's origin, with some suggesting it's from **Cohere's Column series**. Others cautioned against stating this as fact, noting that **Cohere's current model** is underperforming compared to newer ones.
  - The model demonstrated extensive knowledge, correctly identifying the origin of "*Die monster, you don't belong in this world*" and reportedly knowing details about a user's **8th grade winter school trip**. Some users found it underwhelming, while others called it "**very big**" and "**sus**".


**Theme 4. Exploring LLM Capabilities and Limitations**

- **What can't AI / LLM's do for you?** ([Score: 79, Comments: 177](https://reddit.com//r/LocalLLaMA/comments/1en2res/what_cant_ai_llms_do_for_you/)): The post discusses the current state and future expectations of **AI and LLMs**, noting that while there have been incremental improvements, there hasn't been a **game-changing advancement since GPT-4**. The author observes a **convergence in capabilities** among top-level models, questioning whether we're simply attempting tasks that **GPT-4 could already perform** and asks what practical tasks users want AI to accomplish that it currently cannot. The post suggests that the **limitation may lie in the chatbot interface** rather than the underlying LLM technology, proposing that **different fine-tuning approaches** and creating **agents** instead of chat models might unlock more useful behaviors from existing foundation models.
  - **Code generation** for larger applications remains challenging, with LLMs struggling to produce coherent code over **200 lines** without significant manual corrections. Users desire improved capabilities for complex, multi-feature development tasks.
  - **Visual understanding** tasks like object localization, comic book comprehension, and structured image analysis are still difficult for AI. Users report needing extensive preprocessing and specialized tools to achieve partial success in these areas.
  - Users want AI to produce **longer, coherent outputs** beyond current token limits. Some models like **Sonnet 3.5** and **Gemini 1.5 Pro** show promise in this area, but further improvements are desired for extended context generation.


## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Model Capabilities and Advancements**

- **GPT-4o demonstrates unintended voice cloning**: In /r/singularity, a [video from OpenAI](https://www.reddit.com/r/singularity/comments/1enne2l/gpt4o_yells_no_and_starts_copying_the_voice_of/) shows GPT-4o yelling "NO!" and briefly copying the user's voice during testing. This highlights potential risks and challenges in controlling advanced AI models.

- **Google DeepMind's AI achieves human-level performance in table tennis**: In /r/singularity, [Google DeepMind announced](https://www.reddit.com/r/singularity/comments/1en8vrg/google_deepminds_aipowered_robot_plays_table/) their AI-powered robot as the first 'agent' to reach human-level performance in table tennis.

- **Gemini 1.5 Flash pricing reduced**: In /r/singularity, Google [announced a 70% price reduction](https://www.reddit.com/r/singularity/comments/1endo0r/gemini_15_flash_price_is_now_70_lower/) for Gemini 1.5 Flash, making advanced AI capabilities more accessible.

- **OpenAI enables free DALL-E 3 image generation**: In /r/singularity, OpenAI [announced](https://www.reddit.com/r/singularity/comments/1engl4w/openai_were_rolling_out_the_ability_for_chatgpt/) that ChatGPT Free users can now create up to two images per day using DALL-E 3.

**AI in Scientific Research and Mathematics**

- **AI automating mathematical proofs**: In /r/singularity, mathematician Terence Tao [discusses how AI is being used](https://www.reddit.com/r/singularity/comments/1emzf0i/mathematician_terence_tao_says_ai_is_already/) to automate mathematical proofs, potentially revolutionizing the field.

- **Google DeepMind's CSCG for AGI development**: In /r/singularity, a [paper on Clone-structured causal graphs (CSCG)](https://www.reddit.com/r/singularity/comments/1en4t83/google_deepmind_cscg_clonestructured_causal/) is presented as a breakthrough towards AGI, focusing on schema-learning and rebinding mechanisms.

**Robotics Advancements**

- **Boston Dynamics' Atlas performs complex movements**: In /r/singularity, a [video demonstrates](https://www.reddit.com/r/singularity/comments/1eneffr/impressive_boston_dynamics_atlas_does_pushups_and/) the Atlas robot performing push-ups and a burpee, showcasing advancements in robotic agility and control.

**Memes and Humor**

- **"The future is now" meme**: In /r/singularity, a [popular meme post](https://www.reddit.com/r/singularity/comments/1en5p99/the_future_is_now/) humorously comments on the rapid advancement of AI technology.


---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. LLM Advancements and Benchmarking**

- **Gemini 1.5 Flash Slashes Prices**: Google announced significant price cuts for **Gemini 1.5 Flash**, reducing costs by up to **70%** to 7.5c/million tokens for prompts under 128,000 tokens, making it highly competitive in the fast-and-cheap model market.
   - The updated model can now natively understand PDFs and has improved performance for text and multi-modal queries. This move is seen as part of an ongoing trend of price slashing to improve efficiency across the AI industry.
- **DeepSeek-V2 Claims to Outperform GPT-4**: The newly released **DeepSeek-V2** model is reported to surpass **GPT-4** in some benchmarks like **AlignBench** and **MT-Bench**, showcasing advancements in model performance.
   - This claim has sparked discussions about the need for standardized benchmarking and transparent evaluation methods in the AI community to validate such assertions of superiority.
- **MiniCPM-V 2.6 Challenges Top Models**: The open-source vision multi-image model **MiniCPM-V 2.6** is reported to outperform models like **Gemini 1.5 Pro** and **GPT-4V** according to its developers' claims.
   - Links to both the [Hugging Face model](https://huggingface.co/openbmb/MiniCPM-V-2_6) and [GitHub repository](https://github.com/OpenBMB/MiniCPM-V) were shared, inviting the community to explore and validate these performance claims.
  


**2. Model Optimization and Inference Techniques**

- **Tree Attention Algorithm Optimizes Long-Context Processing**: A new paper introduces the **Tree Attention algorithm**, which optimizes self-attention calculations through parallel computation on GPU clusters, promising improved efficiency in handling long-context attention tasks.
   - The implementation, available on [GitHub](https://github.com/Zyphra/tree_attention), aims to enhance performance in scenarios requiring extensive context processing, potentially revolutionizing how models handle large-scale information.
- **Apple Open-Sources Matryoshka Diffusion Models**: Apple has open-sourced a Python package for efficiently training **text-to-image diffusion models** using smaller datasets, linked to their [ICLR 2024 paper](https://github.com/apple/ml-mdm).
   - This package aims to achieve high-quality results with a focus on reduced data and compute requirements, potentially democratizing access to advanced AI image generation techniques.


**3. AI Startup Funding**

- **Sequoia Capital Eyes AI Reasoning Startup**: Sequoia Capital has discussed funding an AI reasoning startup co-founded by **Robinhood's CEO**, aiming to enhance AI capabilities in reasoning and decision-making.
   - This potential investment, reported by [The Information](https://www.theinformation.com/articles/sequoia-capital-has-discussed-funding-ai-reasoning-startup-cofounded-by-robinhood-ceo), signals growing interest in AI technologies that can improve logical processing and decision-making capabilities.
- **Anysphere Secures $60M for AI Coding Assistant**: **Anysphere**, the developer of the AI coding assistant Cursor, has secured over **$60 million** in Series A financing, achieving a **$400 million valuation**.
   - The funding round, co-led by Andreessen Horowitz, demonstrates strong investor confidence in AI-powered coding solutions and their potential to transform software development practices.

**4. Open-Source AI Frameworks and Community Efforts**

- **Replete-LLM-Qwen2-7b release**: The new model **Replete-LLM-Qwen2-7b** was launched, featuring impressive capabilities and benchmarks, inviting users to test it out via [Hugging Face](https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b).
  - Discussions suggested that personal testing is crucial to understanding performance differences.
- **Open Interpreter Hackathon Sparks Interest**: Open Interpreter is gearing up for the '**Breaking Barriers**' hackathon in Dallas from **Sept 20-23**, with **$17,500** in prizes on the line.
  - The event encourages in-person participation but remote applicants are welcome as community discussions on team formation continue.

**5. New AI Model Releases and Innovations**

- **Launch of Replete-LLM-Qwen2-7b**: **Replete-LLM-Qwen2-7b** has been launched, showcasing impressive capabilities and inviting users to test it through [Hugging Face](https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b).
  - The developer emphasized the importance of personal testing instead of relying solely on marketed superiority claims.
- **ActionGemma-9B Model for Function Calling**: The new **ActionGemma-9B** model, designed for function calling, leverages multilingual capabilities from Gemma and the **xLAM** dataset, enhancing user interaction.
  - Details about its functionalities can be accessed [here](https://huggingface.co/KishoreK/ActionGemma-9B).


**6. Community Support and Resources**

- **Seeking AI Research Communities**: Members expressed a desire for a more active **audio research community**, noting that previous platforms like harmonai had become inactive.
  - This highlights a gap in support for audio research discussions and the need for a vibrant community.
- **Hackathon Announcement**: Open Interpreter announced its participation in the '**Breaking Barriers**' hackathon, offering **$17,500** in prizes, encouraging community involvement.
  - The event emphasizes collaboration and innovation in AI, with both in-person and remote participation options.


---

# PART 1: High level Discord summaries




## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Seeking a Thriving Audio Research Community**: A member sought recommendations for an **audio research community** akin to Nous, citing a lack of active discussion in previous discords.
   - *The old harmonai discords are pretty dead,* highlighting a significant gap in audio research support.
- **Introducing CRAB for Multimodal Agents**: The community welcomed the 🦀 **CRAB: Cross-environment Agent Benchmark** for evaluating multimodal agents across platforms including 📱 Android and 💻 Ubuntu.
   - Features include a graph evaluator and task generation to boost **human-like performance**.
- **Intern Eric Reveals ReFT Mastery**: Tomorrow, **Intern Eric** will demonstrate **'How I fine-tuned Llama3 in 14 minutes w/ ReFT'** during a presentation at **10 AM Pacific**.
   - The session focuses on an application of **Representational Fine Tuning**, promising valuable insights into model tuning.
- **Clarifying ReFT vs RLHF Confusion**: Members discussed the differences between **ReFT** and **RLHF**, with one user highlighting misconceptions about their relationships.
   - This confusion signals a need for clearer definitions in community discussions about these techniques.
- **Model Performance Comparison Discussion**: An emphasis was placed on integrating **A/B tests** and robust benchmarks to validate claims of new models' superiority, particularly referencing **Llama-3.1-8B** and **Gemma-2-9B**.
   - Users raised concerns about casually calling models *state-of-the-art* without proper benchmarking.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 2 gains traction**: Members noted that **Gemma 2** is becoming increasingly popular, attracting its own audience compared to predecessors like **Llama** and **Mistral**. Discussions highlighted the model’s distinct characteristics and performance nuances relative to its competitors.
   - *The shift in interest showcases a growing acceptance of diverse architectures in the community.*
- **Introducing Replete-LLM-Qwen2-7b**: The new model **Replete-LLM-Qwen2-7b** was launched, featuring impressive capabilities and benchmarks, inviting users to test it out via [Hugging Face](https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b). The developer urged users to personally evaluate models instead of relying solely on marketed superiority claims.
   - Discussions suggested that personal testing is crucial to understanding performance differences.
- **Model benchmarking debated**: Conversations arose around the drawbacks of current model benchmarks, with users pointing out performance variations tied to varying training data. A member noted that despite higher performance in coding tasks, benchmark scores might not reflect quality due to differences in training goals.
   - *This conversation underscores the importance of context in evaluating model efficacy.*
- **Continuous batching in models elaborated**: Users explored the adaptability of models for continuous finetuning, discussing enhancements such as **ReFT**. Queries arose concerning how **Unsloth** can support additional functionalities under continual training strategies.
   - *This highlights the growing interest in dynamic model adjustment techniques.*
- **Flash Attention 3 compatibility concerns**: Flash Attention 3 (FA3) is noted as compatible exclusively with **H100** hardware and the **Hopper** architecture per [MrDragonFox](https://link.to.paper). This led to clarifications about the automatic installation of **FA2** when using Flash Attention.
   - The discussion prompted inquiries into the practical usage of Flash Attention versions, with members curious if **FA2** still holds predominance.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio faces performance slowdowns**: Users reported long load times and sluggish responses in **LM Studio**, attributing issues to the context length setting, despite prior successful usage.
   - *Reports suggest* that the performance lag affects model loading and responsiveness which ideally should not be impacted by unchanged settings.
- **New users demand guidance on models**: A newcomer asked about supported models in **LM Studio** capable of handling images and PDFs, as well as visual generation models.
   - The discussion highlighted a need for improved onboarding tools to familiarize users with model capabilities.
- **Gemma 2 impresses users with performance**: Users recommend experimenting with **Gemma 2 27B**, noting exceptional performance especially when stacked against **Yi 1.5 34B**.
   - *Feedback underlines* how even the smaller **Gemma 2 9B** model performs effectively across tasks, raising excitement for its bigger counterpart.
- **Debate rages on laptop choices for LLM inference**: A user weighs options between machines featuring an **RTX 4050** or **RTX 4060** for LLM inference, with discussions centered around the significance of an extra **2GB VRAM**.
   - Experts stress that while adding RAM aids performance, maximizing **VRAM** takes precedence to fully harness larger models.
- **NVIDIA GPU power limits on Linux**: Users discussed methods to persistently limit **NVIDIA GPU** power on Linux, particularly for the **RTX 3090**, via tools like **nvidia-smi**.
   - Scripts were suggested to maintain power limits upon reboot, although enterprise systems typically provide better power control options.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SOTA Background Removal Beats RMBG1.4**: A verified member highlighted the **Bilateral Reference for High-Resolution Dichotomous Image Segmentation** model, outperforming **RMBG1.4** in background removal, thanks to contributions from various universities. More details can be found on the [model page](https://huggingface.co/ZhengPeng7/BiRefNet) and in the [arXiv paper](https://arxiv.org/pdf/2401.03407).
   - This model's advances showcase an increased focus on high-quality results with fewer data requirements, signaling a critical shift in background removal techniques.
- **Function Calling with ActionGemma-9B**: The new **ActionGemma-9B** model, fine-tuned for function calling, leverages multilingual capabilities from Gemma and the **xLAM** dataset. Details can be accessed [here](https://huggingface.co/KishoreK/ActionGemma-9B).
   - This development enhances user interaction with models by enabling specific function calling, pushing forward the capabilities of multilingual models in real-world applications.
- **Unity ML-Agents Video Series Launch**: A YouTube video titled **Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers** illustrates creating a chatbot using Unity and Sentence Transformers. Watch the introduction [here](https://youtube.com/live/J-de9K_3xDw?feature=share).
   - This initiative represents an exciting blend of game development and conversational AI, catering to developers interested in integrating advanced language models in gaming environments.
- **Matryoshka Diffusion Models Released**: Apple open-sourced a Python package for training **text-to-image diffusion models** using smaller datasets, linked to their [ICLR 2024 paper](https://github.com/apple/ml-mdm). This allows for **high-quality results** with reduced data and compute needs.
   - This approach could redefine efficiency metrics in training diffusion models, potentially impacting future research in AI-generated media.
- **Discussion on LoRA Training Techniques**: Members recommended focusing on training **LoRAs** instead of the full model, noting minimal benefits from training larger architectures. Memory requirements for running Flux for inference were also discussed.
   - Emphasizing the need for efficient model training practices, these discussions reflect a growing trend toward lighter, more adaptable models in the field.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DALL·E 3 expands access for free users**: OpenAI announced that **ChatGPT Free** users can now create up to **two images per day** with DALL·E 3, supporting both personal and professional needs.
   - Feedback has been mixed, with some users disappointed by the limitations compared to other models.
- **Gemini 1.5 slashes prices by 70%**: Gemini 1.5 Flash has implemented price cuts of up to **70%**, making it much more competitive alongside GPT4o's significant reductions.
   - Analysts suggest this aggressive pricing strategy enhances efficiency, reflecting ongoing competition in AI technology.
- **Deep-Live-Cam enables real-time deepfakes**: **Deep-Live-Cam** allows users to generate high-quality deepfakes from a single image in real-time, as demonstrated through impressive experiments.
   - This project has generated excitement for its potential use in virtual meetings, showcasing its impressive capabilities.
- **Anysphere secures $60M funding**: Anysphere successfully raised **over $60 million** in Series A financing, securing a valuation of **$400 million** for its AI coding assistant, Cursor.
   - Led by Andreessen Horowitz, this funding round highlights investor confidence in AI-driven coding solutions.
- **Llama 3.1 model receives key updates**: Meta launched an updated version of the **Llama 3.1** 405B model, modifying the KV heads from 16 to 8 to comply with its whitepaper specifications.
   - This change has sparked speculation regarding its impact on the model's performance and architecture.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro limits drop**: Users reported that the **Pro** search limit has decreased from **600** to **450**, with a future drop to **300** anticipated, creating unrest regarding transparency.
   - Concerns mount as many users voiced frustrations about this change being made without warning, raising questions about service reliability.
- **OpenAI's Strawberry Model generates buzz**: OpenAI's new **'Strawberry'** model is aimed at enhancing reasoning abilities, generating excitement across the AI community after Sam Altman hinted at it via social media.
   - The project is seen as a significant advancement in tackling complex research tasks, sparking interest among engineers and researchers alike.
- **Anduril hits $14B valuation**: **Anduril Industries** raised **$1.5 billion**, soaring to a **$14 billion** valuation from **$8.5 billion**, largely due to government contracts.
   - With revenues doubling to **$500 million**, the company's growth trajectory indicates robust demand in defense tech amid increasing geopolitical tensions.
- **Image generation hurdles in Perplexity**: Users expressed frustrations over the complexity of image generation processes in Perplexity, wishing for simpler functionality like direct prompt submission.
   - Discussions revealed that current image generation tools are perceived as limited and impractical for user needs, beckoning improvements.
- **API roadmap inquiry**: A member brought up the need for a **roadmap** on adding **internet access** capabilities to the API, highlighting user interest in enhanced features.
   - Clarifications were made regarding models that include 'online' indicating partial internet access, though not in real-time, emphasizing available functionalities.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Navigating NeurIPS Publishing Process**: One member shared their experience with **NeurIPS**, feeling overwhelmed about obtaining quality feedback and publication in major AI conferences. *This process is very overwhelming and I don't know anybody who published in major AI conferences*.
   - They echoed concerns that the journey through these top conferences can be anxiety-inducing.
- **Rebuttal Strategies for Reviewer Scores**: Advice surfaced about rebuttal strategies, particularly for reviewers with low confidence, suggesting minimal focus on those issues. One member noted, *If they state the reason for their low confidence then you can try to address that but otherwise I wouldn't*.
   - This insight aimed to refine the rebuttal process and reduce unnecessary stress.
- **Challenge of Big Conferences**: The conversation highlighted how daunting big conferences can be, with recommendations to consider smaller niche conferences for a more enriching experience. A participant stated, *It feels like one has to publish in the big ones at least once to be taken seriously*.
   - This sparked discussion about the balance between prestige and quality of feedback.
- **Discussion on RLHF Cleanup**: Members debated the need for a cleanup process regarding **RLHF** practices before moving forward with public announcements. A tutorial or blog post was suggested, but the general consensus warned it may take additional time.
   - This discussion underscored the importance of a well-prepared narrative before outreach.
- **Qwen2 Model Exhibiting Unusual Memory Behavior**: Testing revealed that the **Qwen2** model exhibited significant reserved memory during training, particularly at batch sizes of 4, raising red flags about potential memory leak issues. Members are now looking to profile this behavior more thoroughly.
   - This discovery could lead to critical optimizations and adjustments in future training protocols.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **PyTorch Profiler Memory Leak Bug**: A member struggled with a **memory leak** when using the PyTorch Profiler with `profile_memory=True`, unsure of the root cause in settings.
   - Another found success by switching to `torch.cuda.memory._record_memory_history()` for profiling, indicating an alternative approach.
- **Tensor Cores for 4090 Insights**: Discussion centered on where to access detailed specs for **tensor cores** on the 4090, with suggestions to review the **Ada whitepaper**.
   - The **Ampere whitepaper** was mentioned as a reference for **3090** specs, emphasizing the need for thorough documentation.
- **torch.compile Leans on Triton Kernels**: It was shared that **torch.compile** mainly outputs **Triton kernels**, providing a cleaner implementation than CUDA kernel outputs from PyTorch's eager mode.
   - The existence of a **Cutlass backend** was noted but progress remained unclear, highlighting ongoing enhancements in kernel development.
- **INT8 Quantized Training Fix**: An error in **INT8 quantized training** was resolved by setting `requires_grad=False` when calling `torch.chunk()`, streamlining the implementation.
   - This indicates potential intricacies in PyTorch's handling of gradients in tensor operations, highlighting the importance of precision.
- **RoPE Kernel Refactoring**: A discussion took place regarding the **RoPE kernel**, where members suggested refactoring to use explicit trigonometry for improved code clarity.
   - An earlier version without complex numbers was shared, showing a potentially more maintainable approach to kernel design.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Debating CBRN Risks in AI Models**: Extensive discussions highlighted whether filtering **CBRN-related information** could mitigate risks without impairing models' capabilities.
   - Participants pointed out the trade-offs between knowledge removal and the risk of still producing harmful outputs.
- **Opportunities for AI Safety Research**: A member brought up a **career transition grant** from Open Philanthropy aimed at AI safety, seeking GPU resources for educational exercises.
   - Various GPU access options, including **Colab and CAIS clusters**, were discussed for supporting AI research.
- **Challenges with Karpathy's nanoGPT evaluation**: Members addressed issues with **lm-evaluation-harness** for **Karpathy's nanoGPT** model, noting incompatibilities with HF formats.
   - A user requested help getting the evaluation harness operational due to these challenges.
- **Tree Attention for Efficient Computation**: Conversations pointed to a paper on a **Tree Attention algorithm**, which optimizes self-attention calculations through parallel computation on GPUs.
   - The implementation shows promise for enhancing efficiency in long-context attention tasks, with a **GitHub repository** shared.
- **Zamba Model Surprises with Performance**: **Zamba model** garnered attention for outperforming **LLaMA 2 7B** with fewer training tokens, despite limited exposure.
   - Its publicly available dataset has sparked interest due to the model's impressive efficiency and results.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Optimize VRAM Without Downgrading**: Users noted that in **Low VRAM Mode**, switching to a lower model may not be necessary if the generation completes successfully, potentially saving processing time.
   - *Experimenting with model options can help optimize performance,* reducing unnecessary adjustments.
- **Face Swapping Tools: Rope Takes the Lead**: Members recommended [Rope](https://github.com/Hillobar/Rope) for face swapping due to its easier installation compared to Roop, particularly for those on Intel CPUs.
   - The focus was on finding effective yet simple tools for users keen on executing face swaps.
- **Stable Diffusion Performance is Variable**: Users observed fluctuating sampling speeds (s/it) in **Stable Diffusion**, with reported delays when shifting model sizes impacting overall performance.
   - *Insights into setups like ROCm and WSL2 were shared,* indicating the significance of hardware configurations.
- **Commission Custom Lora Models Securely**: Participants discussed utilizing **Civitai's bounty system** for commissioning custom pony lora models, aiming for secure transactions.
   - *Thorough vetting of creators is emphasized as a critical step for reliability* in commissioning practices.
- **Live Preview Settings Spark Interest**: A user asked about optimal live preview settings in **A1111**, specifically questioning the purpose of various formats and if frames are saved.
   - *This reflects a community drive to refine image generation workflows* for enhanced efficiency.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DALL·E 3 Free Access for ChatGPT Users**: ChatGPT Free users can now generate up to **two images per day** using **DALL·E 3**, allowing image creation for projects like *slide decks* and *personalized cards*.
   - This update simplifies image requests by letting users directly ask ChatGPT for images tailored to their specifications.
- **Mistral NeMo Not Achieving Expectations**: Members expressed interest in the performance of **Mistral NeMo** on M1 machines with 16GB RAM, noting limitations on running larger models.
   - Concerns arose regarding the model's compatibility and performance efficacy on consumer-grade hardware.
- **Debate on GPT-4 vs GPT-4o Performance**: Users criticized **GPT-4o**, arguing it underperforms compared to **GPT-4**, particularly in image analysis tasks.
   - *GPT-4o* received flak for providing rigid responses, reminiscent of a programmer disconnecting from core principles.
- **Interest in Local AI Model Workflows**: A participant discussed switching to **Open WebUI** and **Ollama** for running local AI models, contemplating discontinuing their ChatGPT+ subscription.
   - Reliability was noted with *LLama*, but there are challenges with self-hosted setups that need addressing.
- **LangChain and CSV Integration Inquiry**: A user sought resources for integrating a **CSV file** as a *retrieval-augmented generation* (RAG) document within **LangChain**.
   - This shows a growing interest in processing structured data with language models and elevates discussions on practical AI applications.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 1.5 Flash Price Slash**: Multiple users noted that **Gemini 1.5 Flash** has dropped its price to just **7.5c/million tokens**, making it highly competitive for rapid, cost-effective model solutions.
   - The model now natively supports PDFs and has improved its capabilities for **text** and **multi-modal queries**.
- **GPT-4o Mini Tops Gemini 1.5 in Coding**: **GPT-4o Mini** received praise for its lower hallucination rates against **Gemini 1.5**, especially in coding-related tasks.
   - Users indicated a strong preference for models that effectively minimize hallucinations while optimizing coding functionalities.
- **OpenRouter API's Configuration Woes**: A developer raised issues in configuring the **OpenRouter API**, specifically with custom parameters in the `providers` configuration when using the **OpenAI SDK** in TypeScript.
   - The API currently lacks support for these custom parameters, leading to persistent linting errors.
- **Dunning-Kruger Insights Spark Humor**: A lively banter erupted around the **Dunning-Kruger Effect**, as users humorously critiqued self-assessment in discussions about expert knowledge.
   - The conversation humorously juxtaposed confidence against actual ability, particularly regarding profitable ventures.
- **Quest for Japanese-Language LLMs**: A user requested recommendations for **LLMs** that surpass **GPT-4o Mini** in Japanese language capabilities, looking for affordable alternatives.
   - This search reflects a growing demand for models that excel in specialized language processing outside the capabilities of larger models.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **New Sus-Column-R Model Outshines Competitors**: A post on [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1enmcr9/new_suscolumnr_model_on_lmsys_its_just_f_up/) discusses the performance of a new **sus-column-r model**, claiming it outperforms **GPT-4** and **Claude 3.5** in tasks like **translation**, **coding**, and **mathematics**.
   - *I don't understand how this is possible,* highlighted the user, reflecting on the community's intrigue.
- **API Response Quality Under Scrutiny**: Members report a troubling **403 Forbidden** error while using curl for API requests, suggesting it could stem from an **invalid API key** or geolocation restrictions.
   - Despite troubleshooting, members could not resolve the issue, noting discrepancies between VPS and local request successes.
- **Docker Installation Leaves Users Perplexed**: A user faced issues with their interface being non-operational post-**Docker** installation, questioning if any steps were overlooked.
   - In response, **Nick Frosst** indicated that the problem likely relates to a **backend setup** misconfiguration, though specifics remain unclear.
- **Langchain's Multistep Functionality Throws Errors**: A user encountered an error with **Langchain**'s **multistep_tool_use**, receiving a message indicating failure to parse multihop completion.
   - Seeking help, they requested references to documentation on proper integration of **Cohere** and **Langchain**.
- **Embedding Model Quality Discrepancies**: A user reported dissatisfaction after switching from `embed-english-light-v2.0` to `embed-english-light-v3.0`, observing reduced retrieval quality contrary to expectations.
   - Elaborating on their dataset, they noted the newer models did not meet the anticipated performance improvements.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Event-Driven Agent Systems empower flexibility**: Building agents in an **event-driven manner** allows for flexible **cyclic, multi-agent** systems with complex communication patterns. Check out this [awesome tutorial video](https://t.co/he0Az19WJS) showcasing the benefits.
   - _“This is an awesome tutorial video”_ emphasizes the utility of the event-driven approach in agent systems.
- **Mixture-of-Agents overcomes larger model limitations**: A new paper by [Junlin Wang](https://t.co/C8pZBBIcOk) reveals a way to ensemble smaller LLMs into a **Mixture-of-Agents** system outperforming state-of-the-art larger models using a fully async, **event-driven workflow**.
   - Details of the implementation are discussed on [Twitter](https://t.co/hNpoZuBC5l).
- **Understanding Property Graphs for GraphRAG**: An important [video tutorial](https://t.co/CdWqPOxt4c) explains LlamaIndex's **property graphs**, which allow each node and relation to store a structured dictionary of properties, unlocking various techniques.
   - _“This underlying abstraction unlocks a lot of cool techniques”_ highlights the functionality of property graphs.
- **Building Multimodal RAG Pipelines for real-world applications**: New notebooks explain how to create practical **multimodal RAG** pipelines over complex **legal, insurance, and product documents**, starting with the parsing of **insurance claims**.
   - Detailed breakdowns and real-world use cases can be found [here](https://t.co/1w4IfXQ7CP).
- **Selecting embedding models for effective document retrieval**: A member discussed using the [HuggingFaceEmbedding](https://github.com/openai/tiktoken) model with Llama, showcasing document loading examples before query calls.
   - Questions arose around document retrieval after embedding, clarifying key sequential steps for achieving desired results.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Hackathon Sparks Interest**: Open Interpreter is gearing up for the '**Breaking Barriers**' hackathon in Dallas from **Sept 20-23**, with **$17,500** in prizes on the line.
   - The event encourages in-person participation but remote applicants are welcome as community discussions on team formation continue.
- **MiniCPM-V 2.6 Tops the Competition**: The **MiniCPM-V 2.6** model has reportedly outperformed notable competitors like **Gemini 1.5 Pro** and **GPT-4V**, raising interest among users.
   - Links to the [Hugging Face model](https://huggingface.co/openbmb/MiniCPM-V-2_6) and [GitHub repository](https://github.com/OpenBMB/MiniCPM-V) provide further insights into its capabilities.
- **Community Calls for ESP32S3 Insights**: A user sought assistance in deploying **O1** on the **ESP32S3**, inquiring about existing experiences from fellow members.
   - The request for shared experiences aims to enhance implementation strategies among interested users within the community.
- **Request for Linux Support Discussions**: Members discussed the need for a dedicated **#linux-something_or_other** channel to address Linux-specific topics more effectively.
   - This suggestion has garnered positive feedback, linking it to an existing channel aimed at addressing troubleshooting concerns.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain struggles with LLM feature consistency**: Members expressed confusion regarding **LangChain's** ability to provide a uniform API across all LLMs, noting it works with **OpenAI** but not with **Anthropic**.
   - It was clarified that while function calls are similar, prompt modifications are essential due to inherent LLM differences.
- **Claude 3.5 suffers from outages**: **Anthropic’s Claude 3.5** experienced significant downtime, with reports indicating an internal server error code **500** halting its functionality.
   - Users shared the error message, highlighting issues with the API affecting operational capacities.
- **Join the $1000 CTF Challenge!**: There's an exciting **capture-the-flag (CTF) challenge** where participants aim to extract a password from an AI agent, with a prize of **$1000**.
   - The competition raises concerns over data privacy as it examines the risks of leaking secrets through user feedback forms.
- **Mood2Music Dashboard Revealed**: The **Mood2Music** dashboard was showcased, offering AI-driven song recommendations that link to both **Spotify** and **Apple Music** based on user mood.
   - This tool targets **decision fatigue** in music selection by curating playlists aligned with users' emotional states.
- **Introducing CRAB: The Multimodal Agent Benchmark**: The **CRAB** benchmark framework facilitates the building and assessment of multimodal language model agents across various environments, including **Android** and **Ubuntu**.
   - Featuring a **fine-grain** evaluation metric and task generation capabilities, it aims to improve human-like task execution, with resources available on [GitHub](https://github.com/) and the project's [website](https://crab.camel-ai.org/).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **CC vs LAION Dataset Showdown**: The debate regarding whether the [Fondant 25M dataset](https://huggingface.co/datasets/fondant-ai/fondant-cc-25m) holds the title for the largest collection of creative commons/public domain images heated up, touching on the reliability concerns of **LAION-5B** due to its dependence on often irrelevant alt text.
   - Participants highlighted that **LAION-5B** might pose greater risks for accuracy in tasks sensitive to image captioning.
- **Gemma Model Steering Inquiry**: An inquiry popped up about steering **Gemma 2 2B** using the **Gemma Scope**, with a focus on creating effective control vectors for output generation.
   - There's a clear demand for more comprehensive insights beyond basic Google results to elevate understanding of model features.
- **Captions' Reliability Under Scrutiny**: Discussion centered on the **unreliability of mass-captured captions**, with voices expressing concern that all captions might lack precise accuracy.
   - Questions arose about whether employing clip similarity scores might enhance evaluation of whether new captions are **less reliable than originals**.
- **Halva Assistant Insights**: A link was shared regarding the [Halva Assistant](https://research.google/blog/halva-hallucination-attenuated-language-and-vision-assistant/) which aims to mitigate hallucinations in language and vision tasks.
   - This innovation could be pivotal for future AI development, particularly in improving reliability in multimodal systems.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Sequoia Capital Eyes AI Reasoning Startup**: Sequoia Capital has discussed funding an AI reasoning startup co-founded by **Robinhood's CEO**, aiming to enhance AI capabilities in **reasoning** and **decision-making**. More details can be found in [The Information](https://www.theinformation.com/articles/sequoia-capital-has-discussed-funding-ai-reasoning-startup-cofounded-by-robinhood-ceo).
   - This startup focuses on advancing how AI interacts in logical contexts, a critical area for future AI development.
- **Anaconda's New Commercial License Policies**: Research and academic organizations are now required to pay for **Anaconda's** software, as the company pursues compliance with its terms-of-service. Reports indicate institutions are facing legal demands for commercial licenses due to unauthorized usage.
   - Members also raised questions about whether using Anaconda in **Docker containers** necessitates additional licensing, hinting that it likely does.
- **uv Emerges as Speedy pip Alternative**: **uv** is being discussed as a faster alternative to **pip** for package installations, with users noting significant speed improvements. This alternative requires no extra tooling, simply swapping `pip` with `uv pip` for installations.
   - Using uv could streamline development processes for many, especially in environments needing rapid package management.
- **Discourse Improvement Through Humor**: A humorous remark about bad takes in discussions suggests that the world would benefit if only those with **bad takes** contributed to conversations. *If everyone who had bad takes exclusively had bad takes, the world would be a lot better* reflects a common sentiment.
   - This statement highlights a desire for more constructive engagement in community dialogues, calling for higher quality discourse.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Mastering DSPy with YouTube Tutorial**: A member shared a [YouTube tutorial](https://youtu.be/_ROckQHGHsU) on DSPy, detailing **8 examples** from basic to advanced LLM projects aimed at enhancing user understanding.
   - This structured approach allows viewers to grasp key DSPy concepts effectively and implement them in their own projects.
- **Experimenting with OpenAI's Structured Output API**: A member announced their experimentation with the new **structured output API** from OpenAI, enhancing data interactions within projects.
   - This API aims to improve how structured data outputs are utilized, sparking interest in broader implementation.
- **Elevating DSPy Prompts with Custom GPT**: Members discussed improving complex prompts interweaving instructions and examples, focusing on **Signature adapters** and **MIPRO optimization**.
   - A suggested starting point was a [custom GPT guide](https://chatgpt.com/g/g-cH94JC5NP-dspy-guide-v2024-2-7) for better modularization of prompts.
- **Exploring DSPy Use Cases for RAG**: A member sought insights on the suitability of DSPy for RAG tasks, drawing parallels with fine-tuning processes.
   - Another member clarified that successful application hinges on optimizing tasks, metrics, and examples for enhanced LLM performance.
- **Signature Adapters Show Potential for DSPy**: Discussion revolved around the potential benefits of using **Signature adapters** in customizing DSPy prompts.
   - A relevant link for further reading on this topic was shared: [Signature GPT resource](https://chatgpt.com/g/g-JQgwRHI0D-dspy-signature-gpt-v2024-2-21).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Poe Hackathon for Generative UI**: Poe is hosting a one-day [hackathon](https://x.com/poe_platform/status/1820843642782966103) aiming to develop generative UI experiences with advanced LLMs like **GPT-4o** and **Gemini 1.5 Pro**, with in-person events in **Hillsborough, CA 94010**.
   - Only registered participants will receive exclusive details, underscoring the competitive edge of this event.
- **AI-Health Initiative Internship Open**: The **Alliance AI-Health Research Initiative** is on the lookout for students for a **4-month remote internship** to advance research in areas like cancer detection and AI-based heat stroke detection.
   - Applications are open until **August 11**, with opportunities for interns to publish their research findings in an academic journal, [apply here](https://tinyurl.com/applyalliance).
- **Feature Stores in Computer Vision Under Scrutiny**: A member raised questions about the effectiveness and value of **feature stores** in **computer vision**, kicking off a discussion on their role in managing projects.
   - The need for real-world implementations was highlighted, as examples could substantiate the impact of feature stores within various frameworks.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular's License Raises Questions**: A member pointed out that **Modular's license** for **max/mojo** is permissive unless there's intent to commercialize an AI infrastructure platform.
   - Concerns emerged about potential implications if Modular ventures into **robotics** or **AI labeling platforms**.
- **Future Competitiveness Uncertainty**: The community debated that software classified as non-competitive under Modular's agreement may become competitive in the future.
   - Questions lingered on whether such competitive software development must be *frozen* once it transitions.
- **Triton Language User Outreach**: A call went out for **Triton lang** users who have crafted a custom kernel to engage in one-on-one chats with the product team, with **Mojo swag** as an incentive.
   - This initiative aims to gather insights from users to enhance product offerings.
- **Curiosity Around Triton Language**: One member expressed their **first time** hearing about **Triton**, indicating a growing interest in newer programming languages.
   - This hints at a potential for broader community engagement in **advanced programming technologies**.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Impressive Cuts in Google Gemini Pricing**: The [YouTube video titled 'Google Gemini Insane Price Cuts!!!'](https://www.youtube.com/watch?v=3ICC4ftZP8Y) highlights significant price reductions for **Google Gemini 1.5 Flash**.
   - Details about these changes were also shared in the [Google Blog](https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...).
- **Confusion over Comparing Gemini to GPT-4o**: Discussion revolves around whether to compare **Gemini 1.5 Flash** with **GPT-4o**, or draw distinctions to **Gemini 1.5 Pro** instead.
   - Members debated the merit of separating comparisons between standard and mini versions.
- **Free Finetuning of Gemini 1.5 at Play**: There was conversation about **Gemini 1.5's free finetuning** feature influencing its comparison with the Pro version.
   - This distinction has become a focal point in discussions regarding the Gemini models' capabilities.
- **Inquiring about Llama CPP Prompt Caching**: A member sought help on which arguments to use for caching prompts with the **Llama CPP server**, aiming to cache just the initial prompt.
   - They clarified they want to cache the first user prompt, which is **1.5k tokens**, while letting **Llama CPP** manage other content.
- **Inquiries about Llama 3 Training Details**: A member asked for documentation on the training process of the [Llama 3 model](https://huggingface.co/axolotl-ai-co/llama-3-8b-chatml) by Meta, specifically on data and masks used.
   - They noted the approach of renaming existing tokens to serve as special tokens in the Llama 3 model.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AMD backend potentially uses more memory**: A member raised concerns about whether the **AMD backend** consumes more memory compared to the **GPU backend**, leading to discussions about resource allocation and performance.
   - This highlights ongoing considerations in the community on optimizing memory management for various backends.
- **GPU failure reported amidst intense computation**: One member shared the unfortunate news of their **GPU** being damaged, simply stating, *'Rip my GPU got blown.'*
   - This incident has sparked worries about GPU reliability during demanding workload sessions.
- **De-sharding models for simplicity**: A user inquired about transforming a **multi lazy buffer** into a **normal lazy buffer** by de-sharding a model, indicating a desire to streamline processes.
   - This points to prevalent challenges in model optimization and architecture adaptation within the community.
- **Clarifying copy_to_device function usage**: Discussion about the **copy_to_device** function emerged, suggesting its importance in data handling during model operations.
   - This reinforces the need for clarity among users about effective memory management practices in their workflows.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1271194238947823657)** (3 messages): 

> - `Community for Audio Research`
> - `CRAB: Cross-environment Agent Benchmark` 


- **Seeking Audio Research Community**: A user inquired about a community similar to the Nous Research AI Discord but focused on audio rather than language, expressing the desire for a space where they can pose difficult research questions.
   - *The old harmonai discords are pretty dead,* indicating a need for more active discussion around audio research.
- **Introducing CRAB for Multimodal Agents**: A member introduced the 🦀 CRAB: Cross-environment Agent Benchmark, which provides an end-to-end framework for building multimodal agents and evaluating them across platforms like 📱 Android and 💻 Ubuntu.
   - Key components include a graph evaluator for detailed metrics and task generation for composing subtasks, aiming to enhance human-like task performance.



**Link mentioned**: <a href="https://x.com/camelaiorg/status/1821970132606058943?s=46">Tweet from CAMEL-AI.org (@CamelAIOrg)</a>: Introducing 🦀 CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents  🦀 CRAB provides an end-to-end and easy-to-use framework to build multimodal agents, operate environments, ...

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1271283013120032850)** (7 messages): 

> - `ReFT Presentation by Intern Eric`
> - `Discussion about ReFT and RLHF`
> - `Dinner Specialties`
> - `Community Events` 


- **Intern Eric to Present on ReFT**: Tomorrow, **Intern Eric** will present **'How I fine-tuned Llama3 in 14 minutes w/ ReFT'**, showcasing a practical implementation of this technique.
   - The presentation will happen at **10:00 AM Pacific** and aims to delve into **Representational Fine Tuning**.
- **Confusion Between ReFT and RLHF**: A member expressed confusion, stating that they thought **ReFT** was a form of **RLHF**.
   - This highlights ongoing discussions in the community regarding the specific definitions and applications of these techniques.
- **Unique Dinner Menu Shared**: A member shared their unique dinner menu featuring **Borsch made from cow and pig bones, chicken meatballs,** and sour cream.
   - Additional items included a **tomato**, **two cucumbers**, and **fermented fireweed with milk**.
- **Community Engagement Reminder**: A reminder was given that every **Friday**, they gather to discuss research papers and their applications.
   - Members are encouraged to join this community of thinkers and builders for a collaborative atmosphere.
- **YouTube Link Shared**: A member shared a [YouTube video](https://www.youtube.com/watch?v=KXJGrOqVrVQ) which could be relevant to ongoing discussions.
   - This indicates active sharing of resources among members to enhance understanding of topics.



**Link mentioned**: <a href="https://oxen.ai/community?utm_source=x&utm_content=y">Community Resources | Oxen.ai</a>: Manage your machine learning datasets with Oxen AI.

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1271526592614760571)** (3 messages): 

> - `Decoding the Decoder LLM`
> - `GPT2 in Excel`
> - `Fine-tuning Llama 3.1` 


- **Decoding the Decoder LLM using Spreadsheets**: A YouTube video titled ["Decoding the Decoder LLM without de code: Ishan Anand"](https://youtu.be/NamKkerrlnQ?si=xMq8cXI7KZ4N-dU0) illustrates how spreadsheets simplify understanding AI models.
   - The video emphasizes that even experienced engineers may struggle with these models, making this educational tool particularly relevant.
- **GPT2 implemented in Excel**: A member shared that a guy successfully implemented **GPT2** in an Excel spreadsheet as an *educational tool*.
   - This approach aims to make the workings of AI models more accessible and understandable.
- **Fine-tuning Llama 3.1 effortlessly**: A YouTube video titled ["Fine tune 🦙 Llama 3.1 8b with Google Collab | LLM Tutorial"](https://youtu.be/y0EK1Xh4sMU) showcases how to fine-tune **Llama 3.1 8b** for free on Google Collab using Q Lora in just **5 minutes**.
   - Links for resources are provided in the comment section to assist viewers in the process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/y0EK1Xh4sMU">Fine tune 🦙 Llama 3.1 8b with Google Collab | LLM Tutorial</a>: Links in comment sectionFine tune Llama 3.1 8b Instruct for Free on Google Collab from Unsloth with Q Lora in 5 minutesDPO</li><li><a href="https://youtu.be/NamKkerrlnQ?si=xMq8cXI7KZ4N-dU0">Decoding the Decoder LLM without de code: Ishan Anand</a>: Spreadsheets are all you need: Decoding the Decoder LLM without de codeThe struggle to grasp the inner workings of AI models can leave even experienced engin...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1271181325688246444)** (200 messages🔥🔥): 

> - `Model Performance Comparison`
> - `SOTA Claims and Benchmarks`
> - `Hermes 2 Pro vs Mistral`
> - `Replete-LLM Qwen2 Release`
> - `Hand Testing vs Benchmarks` 


- **Model Performance Comparison**: A user emphasized that **effective model testing** should ideally include A/B tests and multiple benchmarks to ensure reliability, especially when claiming superiority.
   - This discussion led to the acknowledgment of **Llama-3.1-8B** and **Gemma-2-9B** as reference points for comparing new models' performance.
- **SOTA Claims and Benchmarks**: Concerns were raised about calling new models *state-of-the-art (SOTA)* without formal benchmark results to validate such claims.
   - Users suggested that detailed model cards and personal testing, while valuable, should supplement these claims with transparent benchmarking against leading models.
- **Hermes 2 Pro vs Mistral**: A user praised **Hermes 2 Pro** for its outstanding performance in parallel tool calls, outperforming **Mistral** significantly in this aspect.
   - This led to discussions about how increasing open-source contributions are pushing the boundaries of model capabilities, often with less funding.
- **Replete-LLM Qwen2 Release**: The release of **Replete-LLM Qwen2-7b** was announced, highlighting its competitive features and open-source nature.
   - Users expressed enthusiasm about the model, though skepticism about SOTA claims was noted, emphasizing the need for benchmark validation.
- **Hand Testing vs Benchmarks**: There was a strong debate over the reliability of hand testing versus standard benchmarks in assessing model performance.
   - While some argued that personal testing provides a better insight into a model's capabilities, others maintained that benchmarks serve as a necessary yardstick for comparison.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/dnhkng/RYS-Llama-3.1-8B-Instruct">dnhkng/RYS-Llama-3.1-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://x.com/rpnickson/status/1821634114274873850?s=46">Tweet from Roberto Nickson (@rpnickson)</a>: Holy shit. Without a doubt the most realistic AI images I&#39;ve ever seen.  We are 99.7% of the way to completely indistinguishable-from-reality AI imagery.  (You can still see a few flaws when zoomi...</li><li><a href="https://tenor.com/view/star-wars-more-gif-21856591">Star Wars GIF - Star Wars More - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://new.reddit.com/r/singularity/comments/1envnur/5_new_models_on_lmsys/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/nousresearch/comments/1eog6es/repletellm_stateoftheart_model_releases_today/?utm_name=web3xcss">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>: Contribute to cognitivecomputations/grokadamw development by creating an account on GitHub.</li><li><a href="https://huggingface.co/dnhkng/RYS-Llama-3.1-8B-Instruct#ai-consultant---lead-ai-engineer">dnhkng/RYS-Llama-3.1-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/dnhkng/RYS-Llama-3.1-8B-Instruct#project-lead---innovative-technologies">dnhkng/RYS-Llama-3.1-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/dnhkng/RYS-Llama-3.1-8B-Instruct#research-scientist">dnhkng/RYS-Llama-3.1-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/dnhkng">dnhkng - Overview</a>: dnhkng has 59 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1271206575792590979)** (109 messages🔥🔥): 

> - `Claude's Upside Down Text Generation`
> - `Multi-GPU Setup for LLMs`
> - `Qwen2-Audio Capabilities` 


- **Claude generates upside down text effectively**: Members noted that Claude can generate text upside down, with some attributing this capability to a potential 'reverse tokenizing token'. A comparison was made against other models, which reportedly struggle with similar tasks.
   - One user expressed frustration after being banned, while others discussed implications of 'antthinking' and the overall logic behind Claude's performance.
- **Challenges with Multi-GPU setups**: Discussion occurred around setting up multi-GPU configurations, specifically using a 4090 alongside a 3090 or 3060, with considerations on power supply demands. Recommendations were made to use a separate power supply for GPUs to manage energy consumption better.
   - Users shared personal experiences with power consumption, emphasizing the importance of calculating actual usage to ensure that their PSUs could handle the load safely.
- **Introduction to Qwen2-Audio**: The Qwen2-Audio model has been released, which allows for both audio and text inputs, generating text outputs while maintaining context during conversations. Users were excited about its capabilities, likening it to Whisper but with enhanced conversational context.
   - Links to the model's demo and features were shared, highlighting functionalities like voice chat and multilingual support, with future plans for expanding the model's capabilities.



**Link mentioned**: <a href="https://x.com/Alibaba_Qwen/status/1821945506463129905">Tweet from Qwen (@Alibaba_Qwen)</a>: Today we release Qwen2-Audio, the next version of Qwen-Audio, which is capable of accepting audio and text inputs and generating text outputs. We open-weight Qwen2-Audio-7B and Qwen2-7B-Instruct in Hu...

  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1271316572530806927)** (7 messages): 

> - `Wordware template`
> - `Benchmarking tasks`
> - `PR merge readiness`
> - `Converter adjustment`
> - `Output length` 


- **Wordware Template Unveils Benchmarking Tasks**: A member shared a [Wordware template](https://app.wordware.ai/share/fda1ba91-b363-4814-a4e3-a997afef7949/playground) aimed at generating a list of specific benchmarking tasks and queries.
   - They indicated it was a **very rough attempt**, inviting feedback for improvements.
- **Ready to Merge PR Discussion**: <@687315767208706059> asked if a **PR** regarding the converter was ready to merge, prompting attention to ongoing development.
   - This question shows the team's collaborative effort in ensuring that code contributions are properly integrated.
- **Need for Additional Features**: Another member noted that they just need to **add in this** feature for completion of the pending task.
   - This suggests ongoing refinement and collaboration on features needed for the project.
- **Output Length Concerns**: A member expressed that the outputs from the template are lengthy, around **1800 characters**.
   - This highlights a potential area for optimization to enhance user experience.
- **Adjustability of the Template**: The original poster reiterated that the Wordware template can easily be adjusted, expressing flexibility in its design.
   - This indicates ongoing experimentation and willingness to refine the tool based on user feedback.



**Link mentioned**: <a href="https://app.wordware.ai/share/fda1ba91-b363-4814-a4e3-a997afef7949/playground">Benchmark_Query_Creator</a>: no description found

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1271187995302559794)** (216 messages🔥🔥): 

> - `Gemma 2 popularity`
> - `Replete-LLM-Qwen2-7b release`
> - `Model benchmarking challenges`
> - `Continuous batching in models`
> - `Training and loss calculation` 


- **Gemma 2 gaining popularity**: Members noted that **Gemma 2** is becoming popular, gaining its own audience unlike its predecessor which was overshadowed by **Llama** and **Mistral**.
   - Conversations highlighted the model's unique characteristics, with participants discussing its performance compared to other models.
- **Launch of Replete-LLM-Qwen2-7b**: A new model, **Replete-LLM-Qwen2-7b**, was released, with details shared on its capabilities and benchmarks, inviting users to test it out in a Hugging Face space.
   - The developer emphasized the importance of testing models personally rather than solely relying on claims about their superiority.
- **Challenges in model benchmarking**: Discussions arose around the unreliability of benchmarks, with members noting that model performance can vary based on the data included in training.
   - A member mentioned that despite a model performing better in coding tasks, its benchmark scores were unexpectedly low due to differences in training objectives.
- **Continuous batching in models**: Users discussed the continued finetuning capabilities of models, mentioning the flexibility in modifying models to support additional features like **ReFT**.
   - Querying and clarification about how **Unsloth** supports various functionalities also emerged during the conversation.
- **Calculating loss post-training**: A member asked about the correct approach to compute loss for individual data points from a dataset after training a model.
   - The coding snippet provided for loss calculation with tensor inputs was discussed, focusing on proper label assignment for effective loss evaluation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/buzz-lightyear-gif-21327280">Buzz Lightyear GIF - Buzz Lightyear - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://colab.research.google.com/drive/1vbH6h760iesRfcVQlm4-KVv1zYM5sB9k?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b">Replete LLM Qwen2 7b - a Hugging Face Space by rombodawg</a>: no description found</li><li><a href="https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html">AsyncLLMEngine &#8212; vLLM</a>: no description found</li><li><a href="https://www.reddit.com/r/nousresearch/comments/1eog6es/repletellm_stateoftheart_model_releases_today/?utm_name=web3xcss">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets/ugshanyu/url">ugshanyu/url · Datasets at Hugging Face</a>: no description found</li><li><a href="https://youtu.be/TKmfBnW0mQA?si=lz2sHuGY_IXBYbN_">Fixing bugs in Gemma, Llama, &amp; Phi 3: Daniel Han</a>: The story behind our 8 bug fixes for Gemma, multiple tokenization fixes for Llama 3, a sliding window bug fix and Mistral-fying Phi-3, and learn about how we...</li><li><a href="https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF">Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2">Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1271275160086904935)** (10 messages🔥): 

> - `Off-topic chat rules`
> - `Message deletion permissions` 


- **Discussion on Off-topic Chat Permissions**: Members debated whether messages like *‘can u dm me?’* are allowed in the off-topic chat, with one stating, **‘oh yes not allowed’**.
   - The need for a dedicated **rules channel** was suggested to clarify such matters.
- **Deleting Messages in Off-topic Chat**: A member confirmed that they have **permission to delete messages**, addressing the question about message management.
   - Following this, another member emphasized the importance of having a rules channel for better governance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1271242752579080306)** (83 messages🔥🔥): 

> - `Loading Models in Colab`
> - `Fine-tuning Llama Models`
> - `LORA Adapter and Merging Models`
> - `GPU and CPU Issues`
> - `Dataset Format for Llama-2-Chat` 


- **Challenges in Loading Models on Colab**: Users reported running out of disk space while training models on Colab, specifically while handling GGUF files.
   - One user inquired about possible solutions to avoid saving to disk and only uploading to Hugging Face, but was informed that uploading requires disk space as well.
- **Fine-tuning Llama Models Effectively**: A user shared experiences fine-tuning Llama 3.1 8b, suggesting that more training examples could improve model performance.
   - Responses indicated that generating a larger dataset and using the appropriate chat template would be crucial for successful fine-tuning.
- **Understanding LORA and Model Merging**: Discussion revolved around how to save LORA adapters and merged models, clarifying that merging combines both into a new model while saving the LORA adapter keeps it separate.
   - Users were reminded that it's generally best not to fine-tune on already fine-tuned models to avoid potential quality loss.
- **Handling GPU and CPU Limitations**: Issues related to GPU RAM limitations arose as users attempted to run models, with suggestions for checking if NVIDIA drivers were properly set up.
   - One user noted that certain operations would be slower due to the transfer of model data to the CPU when GPU RAM is insufficient.
- **Dataset Preparation for Llama-2-Chat**: Clarification was sought regarding the correct format for the dataset when fine-tuning the Llama-2-Chat model.
   - Templates and examples were shared, emphasizing the need for a clear conversation format to ensure effective training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1vbH6h760iesRfcVQlm4-KVv1zYM5sB9k?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing>">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/saving-models/saving-to-gguf#manual-saving">Saving to GGUF | Unsloth Documentation</a>: Saving models to 16bit for GGUF so you can use it for Ollama, Jan AI, Open WebUI and more!</li><li><a href="https://arxiv.org/abs/2305.17493">The Curse of Recursion: Training on Generated Data Makes Models Forget</a>: Stable Diffusion revolutionised image creation from descriptive text. GPT-2, GPT-3(.5) and GPT-4 demonstrated astonishing performance across a variety of language tasks. ChatGPT introduced such langua...</li><li><a href="https://huggingface.co/blog/mlabonne/sft-llama3">Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth</a>: no description found</li><li><a href="https://huggingface.co/unsloth/llama-2-7b-chat-bnb-4bit">unsloth/llama-2-7b-chat-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://docs.google.com/document/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?usp=sharing">SHARED Continuous Finetuning By Rombodawg</a>: Continuous Fine-tuning Without LossUsing Lora and Mergekit  	In this write up we are going to be discussing how to perform continued Fine-tuning of open source AI models using Lora adapter and mergek...</li><li><a href="https://docs.unsloth.ai/basics/chat-templates>">Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)">Unsloth Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1271208651624943636)** (12 messages🔥): 

> - `Flash Attention Versions`
> - `Hopper Architecture`
> - `H100 Hardware Limitations` 


- **Discussion on Flash Attention 3**: Flash Attention 3 (FA3) is noted to be only compatible with H100 hardware and the Hopper architecture, as indicated by [MrDragonFox](https://link.to.paper).
   - The discussion highlighted that FA2 is automatically installed when Flash Attention is used, leading to a clarification on the limited use of FA3.
- **Curiosity about Flash Attention Usage**: Members expressed curiosity about the use of Flash Attention 2 and 3, asking if FA2 is still the most common version in use today.
   - *Flail_* showed interest in the differences among these versions, emphasizing the evolving nature of attention implementations.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1271181270638133381)** (244 messages🔥🔥): 

> - `LM Studio Performance Issues`
> - `Model Loading and Usage`
> - `Houdini and VFX Tools Discussion`
> - `Flux and ComfyUI`
> - `Community Support for New Users` 


- **LM Studio Performance Issues**: A user reported unusually long load times for models and sluggish responses from the AI in LM Studio, despite having used it successfully for months.
   - The slowdown was attributed to the context length setting, although the user noted this wasn't different from prior usage.
- **Model Loading and Usage**: A new user inquired about supported models in LM Studio that can recognize images and PDFs, as well as models for generating visuals.
   - The conversation highlighted the need for user-friendly features to assist newer users in understanding model capabilities.
- **Houdini and VFX Tools Discussion**: Participants discussed the advantages of using Houdini as a VFX tool compared to Autodesk software, citing performance and user experience.
   - There were points about open-source alternatives like Blender and their potential to disrupt the current market for CG software.
- **Flux and ComfyUI**: Users expressed preferences for different UIs, with some preferring the cleaner look of Forge and others finding ComfyUI better for experimentation.
   - Flux support is anticipated to arrive in Forge soon, with users eager for integration and ease of access.
- **Community Support for New Users**: A new user sought guidance on the best way to post suggestions for LM Studio's beta, with community members directing them to relevant channels.
   - There was an emphasis on collective support for newcomers in navigating the platform and improving their experience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tauri.app">Build smaller, faster, and more secure desktop applications with a web frontend | Tauri Apps</a>: Tauri is a framework for building tiny, blazing fast binaries for all major desktop platforms. Developers can integrate any front-end framework that compiles to HTML, JS and CSS for building their use...</li><li><a href="https://huggingface.co/spaces/vilarin/Llama-3.1-8B-Instruct">Meta-Llama3.1-8B - a Hugging Face Space by vilarin</a>: no description found</li><li><a href="https://tenor.com/view/aw-cry-sad-grandpa-gif-14766695">Aw Cry GIF - Aw Cry Sad - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://civitai.com/models/617609?modelVersionId=690425#_">FLUX.1 [dev] - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: Do not download if you have not read all the suggestions Because it is heavy and requires more stuff than SD. And we have new way to run Flux ez wi...</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge.git">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1271217222093766859)** (34 messages🔥): 

> - `Gemma 2 performance`
> - `Laptop choices for LLM inference`
> - `NVIDIA GPU power limiting on Linux`
> - `RAM vs. VRAM for model performance`
> - `Updates on 8700G performance` 


- **Gemma 2 impresses users with performance**: Users recommend trying **Gemma 2 27B** as it performs remarkably well, particularly when compared to **Yi 1.5 34B**.
   - Feedback highlights **Gemma 2 9B**'s effectiveness on various tasks, prompting excitement about the larger **27B model**.
- **Choosing the right laptop for LLMs**: A user is considering a laptop with either an **RTX 4050** or **RTX 4060** for LLM inference, debating the importance of extra **2GB VRAM**.
   - Experts suggest that while RAM is beneficial, focusing on **VRAM** is crucial, with laptops presenting challenges due to upgrade constraints.
- **Limiting NVIDIA GPU power on Linux**: Users discuss how to persistently limit power for **NVIDIA GPUs**, especially for an **RTX 3090**, using tools like **nvidia-smi**.
   - Scripts are suggested to ensure power limits apply after reboot, although enterprise servers offer built-in features for power throttling that consumer hardware might lack.
- **RAM and VRAM balance for model performance**: Participants emphasize that **8GB VRAM** is insufficient for demanding models, suggesting **8GB** significantly expands models' usability.
   - It's noted that relying solely on RAM can slow down performance, hence maximizing VRAM is vital for efficiently running larger models.
- **8700G performance updates**: A user reports enhancements to **8700G** with tweaked RAM settings, achieving **16 tok/s** with LLAMA 3.1 8B using **ollama**.
   - They note limitations and performance issues in LM Studio with AMD GPUs, impacting usability beyond **20GB RAM**, highlighting the need for ongoing optimization.



**Link mentioned**: <a href="https://llm.extractum.io/static/llm-leaderboards/">LLM Leaderboards</a>: All LLM Leaderboards on a single page. A comprehensive list of LLM Leaderboards: Dive into rankings, challenges, and advancements in AI language models within natural language processing, fostering fa...

  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1271232340126863410)** (1 messages): 

> - `Background Removal Improvements`
> - `Function Calling with ActionGemma-9B`
> - `Unity ML-Agents Development`
> - `Segment Anything Model Insights`
> - `Arabic Web Dataset Creation` 


- **SOTA Background Removal Beats RMBG1.4**: A verified member highlighted the **Bilateral Reference for High-Resolution Dichotomous Image Segmentation** model, achieving better performance in background removal than **RMBG1.4**, thanks to contributions from several universities and labs.
   - More information can be found on the [model page](https://huggingface.co/ZhengPeng7/BiRefNet) and in the [arXiv paper](https://arxiv.org/pdf/2401.03407).
- **Function Calling in ActionGemma-9B**: A new **ActionGemma-9B** model fine-tuned for function calling has been released, leveraging multilingual capabilities from Gemma and the **xLAM** dataset.
   - You can check out the model [here](https://huggingface.co/KishoreK/ActionGemma-9B) for more details.
- **Unity ML-Agents Video Series Launch**: A YouTube video titled **Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers** showcases the journey of creating a chatbot using Unity and Sentence Transformers.
   - Watch the exciting introduction [here](https://youtube.com/live/J-de9K_3xDw?feature=share).
- **Insights on Segment Anything Model**: A blog post discusses the **Segment Anything Model** and related advancements in computer vision, focusing on the disparity between language models and vision tasks.
   - For a deeper understanding, the post can be found [here](https://www.lightly.ai/post/segment-anything-model-and-friends).
- **Arabic Web-Only Dataset Pre-training**: The **ArabicWeb24** initiative addresses the creation of a high-quality Arabic web-only pre-training dataset to improve NLP models.
   - Explore the blog post detailing this initiative [here](https://huggingface.co/blog/MayFarhat/arabicweb24).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ZhengPeng7/BiRefNet">ZhengPeng7/BiRefNet · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/KishoreK/ActionGemma-9B">KishoreK/ActionGemma-9B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/DamarJati/FLUX.1-DEV-Canny">FLUX.1-DEV Canny - a Hugging Face Space by DamarJati</a>: no description found</li><li><a href="https://youtube.com/live/J-de9K_3xDw?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers| Part 1</a>: Welcome to our exciting journey of creating an intelligent chatbot using Unity ML-Agents and Sentence Transformers! 🚀In this video, we walk you through the ...</li><li><a href="https://www.lightly.ai/post/segment-anything-model-and-friends">Segment Anything Model and Friends</a>: The Segment Anything Model (SAM) and its successors made a significant leap forward in computer vision, particularly in image and video segmentation. Along with SAM’s innovative approach to promptable...</li><li><a href="https://huggingface.co/spaces/Delik/Anitalker">Anitalker - a Hugging Face Space by Delik</a>: no description found</li><li><a href="https://huggingface.co/spaces/LEGENT/LEGENT">LEGENT - a Hugging Face Space by LEGENT</a>: no description found</li><li><a href="https://www.lightly.ai/post/using-self-supervised-learning-for-dense-prediction-tasks">Using Self-Supervised Learning for Dense Prediction Tasks</a>: Overview of Self-Supervised Learning methods for dense prediction tasks such as object detection, instance segmentation, and semantic segmentation</li><li><a href="https://dev.to/tonic/dockers-testcontainers-are-great-42cl">no title found</a>: no description found</li><li><a href="https://huggingface.co/blog/prithivMLmods/lora-adp-01">Unlocking Creativity with Text-to-Image Generation: Exploring LoRA Models and Styles</a>: no description found</li><li><a href="https://youtu.be/fnKrReaqQgc">How does Uber predict Arrival Times (ETA) for trips? | Uber ML System Design | #systemdesign</a>: Do you know how cab companies like Uber, Ola, and Lyft predict the Expected Time of Arrival (ETA) for the trips? In this video, we design an end-to-end machi...</li><li><a href="https://huggingface.co/blog/MayFarhat/arabicweb24">ArabicWeb24: Creating a High Quality Arabic Web-only Pre-training Dataset </a>: no description found</li><li><a href="https://github.com/Rivridis/LLM-Assistant">GitHub - Rivridis/LLM-Assistant: Locally running LLM with internet access</a>: Locally running LLM with internet access. Contribute to Rivridis/LLM-Assistant development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1271184771065581669)** (174 messages🔥🔥): 

> - `Hugging Face models and API`
> - `Amazon Bedrock pricing`
> - `Model training and architecture`
> - `Message classification methods`
> - `Sampling parameters in LLMs` 


- **Hugging Face models usability and issues**: Users discussed downloading models from Hugging Face and loading them with AutoModel.from_pretrained, where missing files could lead to errors.
   - A user identified a missing file needed for model loading, resolving their issue while another faced challenges with the Gradio interface.
- **Amazon Bedrock's high costs**: The rising prices of Amazon Bedrock were attributed to supply and demand issues, particularly mentioning a chip shortage.
   - An individual shared that they opted for Amazon services due to subprocessor clauses within GDPR compliance in contracts.
- **Analyzing model training importance**: A conversation evolved around the significance of dataset quality in model training, emphasizing scaling results across benchmarks.
   - Participants noted that while architecture matters, the dataset and training data's integrity play a critical role in performance.
- **Exploring message classification methods**: A user presented their metrics for message classification, seeking methods related to categories like Announcements and Technical Support.
   - This indicates a need for structured approaches in class-based message processing within applications.
- **Innovative ideas for tweaking LLM outputs**: A user suggested implementing adaptive temperature adjustments in LLM sampling based on input sequences for improved output variations.
   - This concept parallels existing methods in diffusion but raises questions about practicality and retraining requirements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nroggendorff/hidden-cascade">Hidden Cascade - a Hugging Face Space by nroggendorff</a>: no description found</li><li><a href="https://tenor.com/view/cant-unsee-how-to-unsee-blind-homer-simpson-simpsons-gif-17613101">Cant Unsee How To Unsee GIF - Cant Unsee How To Unsee Blind - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/datasets/Skylion007/openwebtext/tree/main">Skylion007/openwebtext at main</a>: no description found</li><li><a href="https://huggingface.co/facebook/m2m100_418M">facebook/m2m100_418M · Hugging Face</a>: no description found</li><li><a href="https://x.com/jeremyphoward/status/1821557546102174194?t=MUNK-YPm71">Tweet from Jeremy Howard (@jeremyphoward)</a>: Chinese open weights model that easily surpasses all previous models, both closed and open, at MATH:  Quoting Qwen (@Alibaba_Qwen)   Today we release a new model series for math-specific language mode...</li><li><a href="https://x.com/jeremyphoward/status/1821557546102174194?t=MUNK-YPm71NgXJTbOLJDLA&s=19">Tweet from Jeremy Howard (@jeremyphoward)</a>: Chinese open weights model that easily surpasses all previous models, both closed and open, at MATH:  Quoting Qwen (@Alibaba_Qwen)   Today we release a new model series for math-specific language mode...</li><li><a href="https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2">SeamlessM4T-v2</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1271193792635998219)** (3 messages): 

> - `Embedding Serialization`
> - `Reinforcement Learning with Human Feedback` 


- **Learning Embedding Serialization and Deserialization**: A member shared that they are learning how to **serialize and deserialize** embeddings data between **Python** and **C#**.
   - This technical skill is essential for facilitating data interchange between different programming environments.
- **Curated Resources for RLHF**: A member shared a link to a [GitHub repository](https://github.com/opendilab/awesome-RLHF) that contains a curated list of resources for **reinforcement learning with human feedback (RLHF)**.
   - This repository is continually updated, providing valuable information for those interested in this cutting-edge field.



**Link mentioned**: <a href="https://github.com/opendilab/awesome-RLHF?tab=readme-ov-file#2024">GitHub - opendilab/awesome-RLHF: A curated list of reinforcement learning with human feedback resources (continually updated)</a>: A curated list of reinforcement learning with human feedback resources (continually updated) - opendilab/awesome-RLHF

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1271190710552494243)** (13 messages🔥): 

> - `Matryoshka Diffusion Models`
> - `ReFT Fine-Tuning`
> - `Flux Dev Styles Gallery`
> - `VFusion3D Model Release`
> - `SentenceTransformers in Unity` 


- **Apple's Matryoshka Diffusion Models Released**: A researcher at Apple announced the open-sourcing of a Python package for efficiently training **text-to-image diffusion models** using smaller datasets, linked to their [ICLR 2024 paper](https://github.com/apple/ml-mdm).
   - This package aims to achieve **high-quality results** with a focus on reduced data and compute requirements.
- **Eric's Rapid Llama3 Fine-Tuning Presentation**: Intern Eric will present on how he fine-tuned **Llama3** in **14 minutes** using a method called **ReFT**, which integrates representations into the hidden states instead of modifying parameters.
   - His presentation will take place on **Friday, August 9**, at 10:00 AM Pacific, with more information available in the recurring calendar invite.
- **New Flux Dev Styles Gallery Launched**: A member created a [GitHub gallery](https://enragedantelope.github.io/Styles-FluxDev/) to help identify various styles in **Flux Dev**, generated using **ComfyUI** and **Mile High Styler**.
   - The gallery highlights styles but is not comprehensive, and improvements in prompting can enhance style application across generated images.
- **Launch of VFusion3D by Facebook**: **VFusion3D** has been released on Hugging Face, with a demo available [here](https://huggingface.co/spaces/facebook/VFusion3D) showcasing its capabilities as a 3D generative model trained on limited 3D data and extensive synthetic multi-view data.
   - This marks a significant step in exploring scalable **3D generative and reconstruction models** as part of advancing to a 3D foundation.
- **Success of SentenceTransformers Integration in Unity**: A member successfully integrated **SentenceTransformers** and **AutoTokenizer** into **Unity**, requiring the construction of a shell for proper output display.
   - This integration shows potential for using advanced models within gaming environments and interactive applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://enragedantelope.github.io/Styles-FluxDev/">Flux Style Test Gallery</a>: no description found</li><li><a href="https://huggingface.co/spaces/facebook/VFusion3D">VFusion3D - a Hugging Face Space by facebook</a>: no description found</li><li><a href="https://github.com/apple/ml-mdm">GitHub - apple/ml-mdm: Train high-quality text-to-image diffusion models in a data &amp; compute efficient manner</a>: Train high-quality text-to-image diffusion models in a data &amp; compute efficient manner - apple/ml-mdm</li><li><a href="https://oxen.ai/community?utm_source=x&utm_content=y">Community Resources | Oxen.ai</a>: Manage your machine learning datasets with Oxen AI.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1271588188321742879)** (3 messages): 

> - `SEE-2-SOUND Presentation`
> - `Hacking with LLMs`
> - `Benchmark Discussion` 


- **SEE-2-SOUND Presentation Details**: The recording of the previous session titled *Hugging Face Reading Group 26: SEE-2-SOUND* is available on [YouTube](https://www.youtube.com/watch?v=7wkgFR-HYjY) with presenter Rishit Dagli.
   - The [GitHub link](https://github.com/isamu-isozaki/huggingface-reading-group) provides access to past presentations.
- **Upcoming Talk on Hacking with LLMs**: Members plan to discuss *hacking with LLMs* next Saturday, featuring a write-up available on [Medium](https://medium.com/gopenai/understanding-penetration-testing-with-llms-2b0ec6add14a).
   - The talk will also touch on a **benchmark**, details of which will be shared during the session.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=7wkgFR-HYjY">Hugging Face Reading Group 26: SEE-2-SOUND: Zero-Shot Spatial Environment-to-Spatial Sound</a>: Presenter: Rishit DagliPast Presentations: https://github.com/isamu-isozaki/huggingface-reading-group

  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1271359321824952456)** (8 messages🔥): 

> - `Dreambooth LoRA training scripts`
> - `CLIP text encoder support`
> - `Link issues in README`
> - `bf16 vs fp16 training`
> - `Model distribution for Lora training` 


- **Dreambooth LoRA Scripts Released**: The team announced the release of [Dreambooth LoRA training scripts](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md) for FLUX.1, which includes support for **text encoder training** of CLIP.
   - They cautioned that **memory requirements** are quite high and urged users to check the README.
- **Broken Link in README**: A member pointed out that the guide link from **@bghira** is broken in the README, prompting a quick acknowledgment of the issue.
   - The team responded: *'thanks for the catch! will pr a fix.'*
- **Training LoRA in bf16 Recommended**: There was a discussion about whether **LoRA** should be trained in bf16 for consistency with the base models.
   - One member confirmed, *'yeah I'd stick to bf16',* but also noted it might work fine with fp16, referencing a GitHub fix.
- **Request for Model Distribution Support**: A user expressed appreciation for the **balanced mode** in diffusers that enables running native Flux on 2x 16GB and inquired about model distribution support for **Lora training**.
   - *'I know I'm stretching it,'* they added, highlighting the long-term vision for the feature.
- **Runtime Error Encountered**: A user reported encountering a **RuntimeError** related to shape sizing while running the training script for Dreambooth LoRA.
   - The error message detailed the shape issue: *'shape '[1, 16, 8, 2, 8, 2]' is invalid for input of size 262144.'*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.co">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md">diffusers/examples/dreambooth/README_flux.md at main · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1271184281254891652)** (9 messages🔥): 

> - `StrokeSet for Handwriting Conversion`
> - `Image Annotation Software` 


- **StrokeSet concept for handwriting transformation**: A member discussed converting images of handwriting into a **StrokeSet** format composed of strokes with numerous points, rather than using SVG format.
   - They referenced the [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database/data-format) for data format specifications.
- **Exploring image annotation tools**: Another member is developing software for bounding box image annotation and is looking into established tools in this space.
   - They shared **[Humans in the Loop](https://humansintheloop.org/)** with links to ten open-source annotation tools and highlighted the importance of effective dataset creation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database/data-format">Research Group on Computer Vision and Artificial Intelligence &mdash; Computer Vision and Artificial Intelligence</a>: no description found</li><li><a href="https://humansintheloop.org/10-of-the-best-open-source-annotation-tools-for-computer-vision/">Humans in the Loop</a>: Humans in the Loop provides continuous ML model improvement with human input: from dataset collection and annotation to model verification and edge case handling.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1271593133976850565)** (2 messages): 

> - `Voice Recording Sample`
> - `Whisper Implementation Issues` 


- **Irony in Audio Format OGG**: A very interesting voice recording sample was found that ironically discusses the audio format **OGG**.
   - The unexpected nature of the content sparked a lighthearted conversation among members.
- **Whisper wants to convert OGG to ARG**: A member noted that their local **Whisper** aims to convert **OGG** files to **ARG** format.
   - They jokingly suggested that many implementations might face issues with this conversion, adding humor to the discussion.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1271181234562928660)** (27 messages🔥): 

> - `LoRA training`
> - `CUDA resource management`
> - `Splitting models across GPUs`
> - `ONNX model quantization`
> - `Device mapping in model loading` 


- **LoRA training is preferred over full model training**: A member suggested to focus on training **LoRAs** instead of the full model, noting that there are minimal benefits to training the larger architecture.
   - Another member expressed concern over loading **Flux** for inference and the potential space required for training **LoRAs**.
- **CUDA setup can aggregate VRAM efficiently**: Members discussed the importance of having **CUDA** properly configured to easily manage **VRAM** aggregation across installations.
   - It was noted that using proper techniques with multiple GPUs can effectively allocate resources, although CUDA and Flash attention are needed for training **LoRAs**.
- **Discussion on splitting models across GPUs**: There was a consensus that it is **not** straightforward to split a single model across multiple GPUs, with suggestions for combining GPU memory instead.
   - One member confirmed that while individual model sharding is possible, it often leads to increased latency due to the overhead of data movement.
- **Exploring ONNX for model optimization**: A member highlighted the potential of using **ONNX** for extracting sub-models, which may improve efficiency over relying solely on **Python/PyTorch**.
   - Quantizing models to **4 or 8 bits** was also suggested as a viable optimization when transitioning to ONNX.
- **Device mapping and model loading errors**: A member encountered a `NotImplementedError` when attempting to use `device_map='auto'`, suggesting the use of `device_map='balanced'` instead.
   - The discussion referenced relevant documentation and implementation details from **Hugging Face**, emphasizing modeling practices with multiple components.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement.">Working with big models</a>: no description found</li><li><a href="https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#extracting-sub-model-with-inputs-outputs-tensor-names">onnx/docs/PythonAPIOverview.md at main · onnx/onnx</a>: Open standard for machine learning interoperability - onnx/onnx</li><li><a href="https://github.com/huggingface/diffusers/blob/65e30907b5df1338ae65a20b78866c87e061c952/tests/models/test_modeling_common.py#L855">diffusers/tests/models/test_modeling_common.py at 65e30907b5df1338ae65a20b78866c87e061c952 · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers</li><li><a href="https://github.com/huggingface/diffusers/blob/65e30907b5df1338ae65a20b78866c87e061c952/tests/models/test_modeling_common.py#L955">diffusers/tests/models/test_modeling_common.py at 65e30907b5df1338ae65a20b78866c87e061c952 · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers</li><li><a href="https://github.com/huggingface/diffusers/blob/65e30907b5df1338ae65a20b78866c87e061c952/src/diffusers/models/transformers/pixart_transformer_2d.py#L81">diffusers/src/diffusers/models/transformers/pixart_transformer_2d.py at 65e30907b5df1338ae65a20b78866c87e061c952 · huggingface/diffusers</a>: 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch and FLAX. - huggingface/diffusers
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1271204423254675560)** (56 messages🔥🔥): 

> - `DALL·E 3 updates`
> - `Gemini 1.5 price cuts`
> - `Deep-Live-Cam deepfake`
> - `Anysphere fundraising`
> - `Llama 3.1 updates` 


- **DALL·E 3 allows free users to create images**: OpenAI announced that ChatGPT Free users can now create up to **two images per day** with DALL·E 3, using it for personal and professional needs.
   - However, community feedback varied, with some expressing disappointment over the limits compared to other models.
- **Gemini 1.5 sees major price cuts**: Recent updates indicate that Gemini 1.5 Flash has cut prices up to **70%**, making it more competitive in the AI landscape, alongside GPT4o's substantial reductions.
   - Analysts note this trend in price slashing improves efficiency across the board, hinting at a continued competitive market.
- **Deep-Live-Cam creates real-time deepfakes**: A trending GitHub project called **Deep-Live-Cam** allows users to create high-quality deepfakes from a single image streamed live.
   - Experiments showed impressive real-time capabilities, eliciting excitement about potential applications in virtual meetings.
- **Anysphere raises $60M in Series A funding**: Anysphere, the developer of the AI coding assistant Cursor, secured **over $60 million** in Series A financing, achieving a **$400 million valuation**.
   - This funding round was co-led by Andreessen Horowitz, signaling strong investor confidence in AI-powered coding solutions.
- **Meta updates Llama 3.1 model**: Meta has released a new version of the **Llama 3.1 405B model**, making changes to the KV heads from 16 to 8, aligning it with the whitepaper specifications.
   - This update has sparked speculation and interest in its implications on performance and model architecture.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://facebookresearch.github.io/nougat/">Nougat</a>: no description found</li><li><a href="https://junlinhan.github.io/projects/vfusion3d.html">Creative and Descriptive Paper Title.</a>: Paper description.</li><li><a href="https://homebrew.ltd/blog/can-llama-3-listen">Homebrew</a>: Building human-augmenting AIs that run on energy-efficient hardware.</li><li><a href="https://x.com/swyx/status/1821771182443540752?s=46">Tweet from swyx 🍓 (@swyx)</a>: by my calculations, the gpt4o price cut and the gemini 1.5 pro august result are &#34;true ups&#34; vs the Price-Intelligence Frontier of Q2 2024.   But we&#39;re now in Q3... and it&#39;s been nonsto...</li><li><a href="https://x.com/matthewberman/status/1821949143918489794?s=61">Tweet from MatthewBerman (@MatthewBerman)</a>: #1 trending github repo right now looks INSANE  Single image to live stream deep fake.   Look at that quality!!  It&#39;s called Deep-Live-Cam (link in replies)</li><li><a href="https://x.com/jay_wooow/status/1821989710207840730?s=46&t=6FDPaNxZcbSsELal6Sv7">Tweet from joao (@jay_wooow)</a>: @MatthewBerman Some experiments - it works almost flawlessly and it&#39;s totally real-time. Took me 5 minutes to install.</li><li><a href="https://x.com/OpenAI/status/1821644904843636871">Tweet from OpenAI (@OpenAI)</a>: We’re rolling out the ability for ChatGPT Free users to create up to two images per day with DALL·E 3.   Just ask ChatGPT to create an image for a slide deck, personalize a card for a friend, or show ...</li><li><a href="https://x.com/MatthewBerman/status/1821949143918489794">Tweet from MatthewBerman (@MatthewBerman)</a>: #1 trending github repo right now looks INSANE  Single image to live stream deep fake.   Look at that quality!!  It&#39;s called Deep-Live-Cam (link in replies)</li><li><a href="https://x.com/altryne/status/1821617540227068017?s=46">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: This is also new!  New multimodal capabilities for PDFs understanding 👀   cc @simonw  Quoting Logan Kilpatrick (@OfficialLoganK)   Good news for @GoogleAI developers:  - Gemini 1.5 Flash price is now...</li><li><a href="https://x.com/jay_wooow/status/1821989710207840730?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from joao (@jay_wooow)</a>: @MatthewBerman Some experiments - it works almost flawlessly and it&#39;s totally real-time. Took me 5 minutes to install.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eoin62/meta_just_pushed_a_new_llama_31_405b_to_hf/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/IvjLXGR7-vM?si=KWCzye6rKoBv--uL">Open Approach to Trust &amp; Safety: Llama Guard 3, Prompt Guard &amp; More | Llama for Developers</a>: Download Llama 3.1 ➡️ https://go.fb.me/kbpn54Zacharie Delpierre Coudert &amp; Spencer Whitman from the Llama Trust &amp; Safety team at Meta join us for a discussion...</li><li><a href="https://youtu.be/02fBBoZa9l4?feature=shared">Mark Zuckerberg on Llama, AI, &amp; Minus One</a>: How did Mark Zuckerberg turn Facebook into Meta? Why does he think open source is the future of AI? And how has he maintained a Minus One mindset at his foun...</li><li><a href="https://youtu.be/Lba_MBZsR5s?si=tdMGsm3m0eQUFpg5">Designing Reliable AI Systems with DSPy | Neural Search Talks: Omar Khattab</a>: In this episode of Neural Search Talks, we&#39;re chatting with Omar Khattab, the author behind popular IR &amp; LLM frameworks like ColBERT and DSPy. Omar describes...</li><li><a href="https://techcrunch.com/2024/08/09/anysphere-a-github-copilot-rival-has-raised-60m-series-a-at-400m-valuation-from-a16z-thrive-sources-say/">Exclusive: Anysphere, a GitHub Copilot rival, has raised an over $60M Series A at  $400M valuation from a16z, Thrive, sources say</a>: Anysphere, a two-year-old startup that’s developed an AI-powered coding assistant called Cursor, has raised over $60 million in a Series A financing at a</li><li><a href="https://www.promptfiddle.com/gpt4o-pdf-extraction-4RHoa">gpt4o pdf extraction — Prompt Fiddle</a>: no description found</li><li><a href="https://findoc-gloo.vercel.app/">BAML Demo</a>: no description found</li><li><a href="https://github.com/BoundaryML/baml-examples/blob/main/findoc/src/app/actions/extract-pdf.ts#L17">baml-examples/findoc/src/app/actions/extract-pdf.ts at main · BoundaryML/baml-examples</a>: Contribute to BoundaryML/baml-examples development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1271558940135784529)** (147 messages🔥🔥): 

> - `Ruby for AI development`
> - `AI consulting and client acquisition`
> - `Prompt crafting workshops`
> - `Research agents`
> - `AI tools and automation` 


- **Ruby community expands for AI applications**: There's a small but growing community focused on building AI applications using **Ruby**, which is noted for its ability to create domain-specific languages (DSLs) effectively.
   - One member is working on a project called Boxcars, which aims to improve Ruby's capabilities in AI, highlighting its potential for *LLM coding*.
- **AI consulting and project collaboration suggestions**: Members discussed the prospects of **AI consulting**, stressing the importance of identifying tedious tasks for automation to showcase skills in the consulting space.
   - Suggestions included using tools like **Elicit** for problem discovery, prompting questions about what annoys clients in their roles.
- **Interest in Research Agents for collaboration**: Several members expressed interest in exploring **research agents**, with one suggesting to use a research agent for research on the topic itself, like using Elicit to analyze its documentation.
   - It was proposed to collaborate and potentially prepare for a deeper discussion in two weeks.
- **Prompt crafting workshops gain traction**: There's an ongoing interest in hosting **prompt crafting workshops**, particularly aimed at helping those without coding or machine learning backgrounds learn how to coax models effectively.
   - Participants agreed on the value of narrowing the scope of models and connecting them later for practical solutions.
- **Importance of context in AI development**: Context is emphasized as crucial for effective AI tool creation, with tools designed to perform specific tasks noted for their efficiency.
   - Members discussed the balance between using generalized tools and the advantages of customization for particular applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/thedayisntgray">Tweet from undefined</a>: no description found</li><li><a href="https://docs.sublayer.com/">Sublayer - Ruby AI Framework</a>: no description found</li><li><a href="https://translation-cellium.ngrok.app/">Translations</a>: no description found</li><li><a href="https://tenor.com/bNGaG.gif">Boo GIF - Boo - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://tenor.com/bc64D.gif">Gimme Code Gimme GIF - Gimme Code Gimme Code - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://artium.ai/apex">APEX - The Future Of AI Product Planning</a>: APEX is the place to put your ideas and explore limitless possibilities. Eliminate months of workshops and endless planning in 15 minutes.</li><li><a href="https://github.com/facebookresearch/faiss/issues/2359">Correct pronunciation · Issue #2359 · facebookresearch/faiss</a>: Sorry for such a stupid question. I was not able to find this information anywhere. What is the right way to pronounce faiss? Like the word facebook Like the word feisty Thanks in advance</li><li><a href="https://tenor.com/Ynql.gif">Jack Nicholson Yes GIF - Jack Nicholson Yes Nodding - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1271186400787697758)** (178 messages🔥🔥): 

> - `Perplexity Pro Limits`
> - `Subscription Issues`
> - `Image Generation Complexity`
> - `Model Usage Clarity`
> - `Browser Integration` 


- **Perplexity Pro limits decrease**: Multiple users have reported that the Pro search limit has recently decreased from **600** to **450** and is expected to drop further to **300** soon.
   - This change was made without prior notification, raising concerns among subscribers about future limits and transparency.
- **Challenges with subscription purchases**: Users are experiencing issues with the Stripe payment platform, leading to difficulties in purchasing Pro subscriptions with various payment methods.
   - Some suspect they might have been banned, while others report receiving no help through chat or phone support.
- **Image Generation Difficulties**: A user expressed frustration with the complexity of image generation in Perplexity, questioning why it isn't simpler like entering a prompt and clicking a button.
   - Responses indicated that the current tools for image generation are limited and may not be practical for user needs.
- **Usage of wrong models**: Users noted that while using Perplexity, they are defaulted to the Pro model, with confusion about how to switch between available models.
   - Some users reported they couldn't rewrite or delete threads, suggesting the platform is experiencing technical issues.
- **Setting Perplexity as Default Search**: A user shared that they set Perplexity as their default search engine in a browser, noting the loss of some conveniences but viewing it as an adjustment process.
   - Others have similarly tried integrating Perplexity into their workflows, balancing between usability and convenience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aravsrinivas/status/1821637031002566671?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Light mode  Quoting Aravind Srinivas (@AravSrinivas)   Don&#39;t be evil</li><li><a href="https://www.perplexity.ai/page/google-loses-search-antitrust-euX29JMyQJWU8YAEGeROzw>)">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://x.com/aravsrinivas/status/1821785478183694579?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: 🧵reply here with perplexity permalinks that have hallucinations and loss of context across follow-ups in your thread for your perplexity queries. will be fun and educational!</li><li><a href="https://www.perplexity.ai/search/are-there-any-plans-by-perplex-iskQH6kpQh6IpJFJjbiRqg">Are there any plans by Perplexity to add Gemini 1.5 Pro to the list of...</a>: Based on the available information, there are currently no public plans announced by Perplexity AI to add Gemini 1.5 Pro to their list of available language...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1271201418715467922)** (12 messages🔥): 

> - `OpenAI's Strawberry Model`
> - `Decimal Comparisons`
> - `Defence Tech Anduril Valuation`
> - `Stuck Astronauts Return Timeline`
> - `AI-Assisted Medical Advocacy` 


- **OpenAI's Strawberry Model sparks interest**: OpenAI's new model, **'Strawberry'**, aims to enhance AI reasoning capabilities and tackle complex research tasks, generating significant buzz within the AI community.
   - Sam Altman's social media hint about strawberries was interpreted as a clue towards this innovative project, igniting excitement among enthusiasts.
- **Comparing 3.33 and 3.4 decimals**: The comparison shows that **3.4 is greater than 3.33**, emphasizing the importance of aligning decimal points for accurate assessments.
   - This method aids in precise measurements relevant in fields like science and finance, where even small differences hold significance.
- **Anduril achieves a $14B valuation**: Defense tech startup **Anduril Industries** has raised $1.5 billion, now boasting a valuation of **$14 billion**, marking a significant jump from its previous $8.5 billion valuation.
   - The company doubled its revenue to approximately **$500 million**, fueled by government contracts and investments from major firms.
- **Stuck Astronauts' return delayed**: NASA officials announced that two astronauts stuck at the International Space Station since June 2024 may not return to Earth until **February 2025**.
   - The delay is due to mechanical failures with the **Boeing Starliner** capsule, which has raised safety concerns regarding the astronauts' journey home.
- **AI tools transforming medical advocacy**: Innovative companies are developing **AI tools** to assist with medical note analysis and help individuals manage their health.
   - These advancements provide essential support for women dealing with breast implant illness, enhancing their understanding and healthcare experience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/decimal-comparisons-3-33-vs-3-TtUoN0wVRhqXcBAb_tX.Ww">Decimal Comparisons: 3.33 vs 3.4</a>: 好的,我會用中文回答你關於小數比較的問題。  3.4 大於 3.33  1. 首先,我們要將這兩個小數對齊到相同的小數位:    3.33    3.40  2. 從左到右比較每一位數字:    - 整數部分都是3,相等    - 第一位小數都是3,相等    - 第二位小數,3.4的4大於3.33的3  3....</li><li><a href="https://www.perplexity.ai/search/what-insights-does-the-200-day-NJPf1o6GQ9C5OXy.forfmA">what insights does the 200 day SMA provide for the BTC USD pairing?</a>: The 200-day Simple Moving Average (SMA) provides several key insights for the BTC-USD pairing:  1. Trend Identification:    - The 200-day SMA helps traders...</li><li><a href="https://www.perplexity.ai/search/core-framework-NQi9hl9ySrKJX9eE4bcSoA#0">core framework</a>: The term &quot;core framework&quot; can refer to several different concepts depending on the context. Below are some of the primary interpretations:  The NIST...</li><li><a href="https://www.perplexity.ai/search/why-are-my-silver-spoons-disco-GAuUmAZXTtygkMgLKt3diQ">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/page/defence-tech-anduril-hits-14b-ZeHfS6seRCSRPwp4qitrbg">Defence Tech Anduril Hits $14B Valuation</a>: According to TechCrunch, Bloomberg, and others, defense technology startup Anduril Industries has raised $1.5 billion in a new funding round, reaching a...</li><li><a href="https://www.perplexity.ai/page/enbloc-website-ai-assisted-med-j39p4Uv8SAysUM.iyXVcdA">Enbloc Website: AI-Assisted Medical Advocacy</a>: Leading developers of AI tools for health management are revolutionizing patient care and empowering individuals to take control of their health. These...</li><li><a href="https://www.perplexity.ai/search/how-can-i-use-inline-padding-vSnbnuMzStiI1sZKGs4HhA?__cf_chl_rt_tk=Ssx1APDVq7zXMbPOHijPl8mFfEfk1JbgdUF6zBGNwT8-1723119550-0.0.1.1-8468">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/page/decimal-comparisons-9-9-vs-9-1-efHZQQZGT6SQZ_GFVODu_g">Decimal Comparisons: 9.9 vs 9.11</a>: Decimal comparisons, while seemingly straightforward, can be surprisingly tricky, as evidenced by the question: Is 9.9 bigger or smaller than 9.11? This...</li><li><a href="https://www.perplexity.ai/page/stuck-astronauts-will-return-i-00a_0jYZRKGrA38tE5uaGA">Stuck Astronauts Will Return in 2025</a>: According to NASA officials, two astronauts who have been stuck at the International Space Station since June 2024 due to issues with Boeing&#x27;s Starliner...</li><li><a href="https://www.perplexity.ai/page/openai-s-strawberry-model-a-ne-w.mxwLOcRSGrgiKFhCD2Gw">OpenAI&#x27;s &quot;Strawberry&quot; Model: A New Frontier in AI Reasoning</a>: OpenAI&#x27;s &quot;Strawberry&quot; model, a cutting-edge AI initiative, is generating buzz for its potential to enhance reasoning capabilities and navigate complex...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1271237725768515584)** (9 messages🔥): 

> - `Google Maps URL efficiency`
> - `API roadmap for internet access`
> - `Costs of online model usage`
> - `Quality of Chinese search results`
> - `Searching Wikipedia pages in JSON format` 


- **Struggles with Google Maps URLs**: A user is facing challenges obtaining accurate Google Maps URLs for their day-to-day trip itinerary.
   - *Is there a way to get URLs in an efficient way or is it not possible?*
- **Querying API for internet access roadmap**: A member inquired about a roadmap for adding **internet access** and **PRO search** capabilities to the API.
   - Another member clarified that models with 'online' in the name have some capacity to access the internet, but not in real time.
- **Clarification on online model fees**: Someone asked about the fee structure for using online models, wondering whether it was $5 after 1000 searches or $0.005 for each query.
   - The response confirmed it is **$0.005 for each query**, leading to discussions about the rapid consumption of credits.
- **Concerns over quality of Chinese search results**: A member shared their experience with searching Chinese resources, suggesting that results may be of lower quality than expected.
   - However, they noted that searching Chinese wiki pages yielded **solid results** despite these concerns.
- **JSON format for Wikipedia search results**: A user presented a prompt for searching Wikipedia pages on **中华人民共和国** and outputting related URLs and content in JSON format.
   - They also emphasized the need to assess whether the content contains relevant information about **中华人民共和国**.


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1271551907202076765)** (22 messages🔥): 

> - `NeurIPS experience`
> - `Rebuttal strategies`
> - `Conference publishing challenges`
> - `Impact of reviewer confidence`
> - `Smaller conferences vs. larger conferences` 


- **Navigating NeurIPS Publishing Process**: One member shared their experience with NeurIPS, expressing feeling overwhelmed about obtaining quality feedback and publication in major AI conferences.
   - *This process is very overwhelming and I don't know anybody who published in major AI conferences*.
- **Rebuttal Strategies for Reviewer Scores**: Advice was given regarding rebuttals, highlighting that if reviewers have low confidence, it may not warrant extensive focus during the rebuttal process.
   - One member noted, *If they state the reason for their low confidence then you can try to address that but otherwise I wouldn't*.
- **Challenge of Big Conferences**: Discussion emphasized the daunting nature of big conferences, with members recommending smaller niche conferences for a better experience.
   - One participant mentioned the significance of publishing at top conferences, stating, *It feels like one has to publish in the big ones at least once to be taken seriously*.
- **Encouragement to Try Elsewhere**: Members encouraged taking feedback constructively and considering other venues for papers, underscoring that rejection is a common part of the process.
   - It was stated, *It's normal for papers to go through several rounds of reviews/rejections in different conferences*.
- **Value of Research Over Conference Prestige**: Commentary suggested that excellent research often appears on arXiv instead of going through traditional conference channels, retaining its value regardless of the venue.
   - A member cited the original DQN paper being a workshop paper, showcasing that impactful work can emerge from smaller venues.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1271203318881845404)** (128 messages🔥🔥): 

> - `RLHF Cleanup Discussions`
> - `Qwen2 Model Behavior`
> - `Expandable Segments Implementation`
> - `Memory Management in Training`
> - `Publicity for Small Models` 


- **Discussion on RLHF Cleanup**: Members discussed the need for potential cleanup on RLHF processes before making further public announcements.
   - A tutorial or blog post was suggested, but consensus was that it may take additional time to prepare.
- **Qwen2 Model Exhibiting Unusual Memory Behavior**: Testing revealed that the **Qwen2** model had significant reserved memory during training, especially at batch sizes of 4, indicating potential issues.
   - Members expressed interest in profiling this behavior further, questioning whether it indicated a memory leak.
- **Proposed Implementation of Expandable Segments**: There was a proposal to enable `expandable_segments:True` as a default feature across model configurations due to its minimal performance impact.
   - Concerns were raised about whether this would cause any breakage, but many argued it could easily be toggled off if needed.
- **Memory Management Challenges During Training**: Users reported issues with OOM (Out of Memory) errors while training on both 0.5B and 1.5B variants of the Qwen models, despite adjustments.
   - It was noted that using Attention (AC) didn’t significantly affect throughput, suggesting that different strategies might be needed to optimize performance.
- **Public Relations Strategy for Small Models**: Members discussed drafting a public announcement about the small models like Qwen2, emphasizing their ability to run on limited hardware.
   - It was suggested to proceed with adding support for expandable segments across models quickly, aiming for general adoption by future releases.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/salman-mohammadi/torchtune/runs/vh7uhuz1/overview">salman-mohammadi</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/salman-mohammadi/torchtune/runs/3guul08l?nw=nwuserrdoublea">salman-mohammadi</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/jcummings/small-model-large-reserved-memory/runs/mqo9mayl?nw=nwuserjcummings">jcummings</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/salman-mohammadi/torchtune/runs/djmo8cwh?nw=nwuserrdoublea">salman-mohammadi</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/salman-mohammadi/torchtune?nw=nwusersalmanmohammadi">salman-mohammadi</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://github.com/pytorch/torchtune/blob/f6ddfccf7b9f9e8facb1fc42d2a2c5635d34d8ed/torchtune/models/qwen2/_tokenizer.py#L96">torchtune/torchtune/models/qwen2/_tokenizer.py at f6ddfccf7b9f9e8facb1fc42d2a2c5635d34d8ed · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://www.ventoy.net/en/index.html">Ventoy</a>: Ventoy is an open source tool to create bootable USB drive for ISO files. With ventoy, you don't need to format the disk again and again, you just need to copy the iso file to the USB drive and boot i...</li><li><a href="https://github.com/pytorch/torchtune/issues/1278">[RFC] Optimizer CPU offload from torchao for single GPU low memory config · Issue #1278 · pytorch/torchtune</a>: The recent addition of optimizer CPU offload in torchao can be useful for single GPU low memory config. https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload...</li><li><a href="https://huggingface.co/datasets/Anthropic/hh-rlhf">Anthropic/hh-rlhf · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1271392626746327071)** (10 messages🔥): 

> - `PyTorch Profiler Memory Leak`
> - `Tensor Core Specs for 4090` 


- **Confusion over PyTorch Profiler Memory Leak**: A member encountered a **memory leak** while using the PyTorch Profiler with `profile_memory=True` and is unsure which part of their settings caused it.
   - Another member confirmed they faced similar issues and switched to `torch.cuda.memory._record_memory_history()` for memory profiling instead.
- **Discussion on Tensor Core Specs for 4090**: A user inquired about where to find detailed specs for **tensor cores** on the 4090 or any other graphics card.
   - One member suggested searching for the **Ada whitepaper** for the 4090 and noted that the **Ampere whitepaper** contains details for the 3090.


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1271279698822824039)** (6 messages): 

> - `torch.compile kernels`
> - `CUDA kernels visibility`
> - `torchao import error`
> - `Cutlass backend progress` 


- **torch.compile primarily uses Triton kernels**: Members discussed that **torch.compile** mostly outputs **Triton kernels** since they are easier to write and generate, maintaining strong performance.
   - While **PyTorch eager** mode does output **CUDA kernels**, there is also a Cutlass backend for **torch.compile**, though its current progress is unclear.
- **CUDA kernels visibility in eager mode queried**: A member inquired if there is a way to view the **CUDA kernels** generated by eager mode similar to how **torch.compile** operates.
   - This reflects ongoing interest in kernel outputs across different modes within the **PyTorch** framework.
- **torchao import issue reported**: An import issue was reported with the statement `from torch._inductor.runtime.runtime_utils import do_bench`, indicating it is broken in the nightly build.
   - This issue was confirmed to affect **torchao**, highlighting the importance of maintaining updates in nightly releases.
- **Resolution available for torchao import issue**: Another member indicated that the import problem has been fixed in the latest version of **ao**, suggesting users merge from the latest main branch.
   - This solution provides a route for users experiencing the import error to regain functionality.


  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1271334549229666314)** (11 messages🔥): 

> - `Flash Attention Paper`
> - `Cooperative Thread Array`
> - `Memory Access Issues`
> - `KV Block Ordering`
> - `Synchronization in CUDA` 


- **Clarification on Flash Attention Block Order**: Discussion on the **Flash Attention** paper clarified that the ordering of KV blocks isn't crucial as they can be scheduled in different orders without changing the algorithm's correctness.
   - However, the pairs of K and V rows must maintain their logical pairing for the algorithm to function properly.
- **Understanding Cooperative Thread Array (CTA)**: Members explained that CTA, or **cooperative thread array**, refers to a parallel worker in CUDA that allows threads in the same CTA to access shared memory.
   - This concept is vital as it delineates the relationship between warps and CTAs in the CUDA execution model.
- **Potential Issues with Naive Maximum Calculation**: Questions were raised about atomicity in memory accesses when calculating `m_i = max(m_i, s_ij)` across multiple threads with varying `j` values.
   - It was clarified that while thread-specific memory accesses are ordered, visibility across different threads doesn't ensure atomicity, requiring synchronization.
- **Significance of 'i' and 'j' in Flash Attention**: The discussion clarified that in the **Flash Attention** algorithm, `i` is a spatial coordinate, while `j` represents a temporal aspect related to the number of KV blocks processed.
   - This distinction is important for understanding how maximum entries are tallied within the algorithm.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1271492294377209947)** (16 messages🔥): 

> - `INT8 Quantized Training Issues`
> - `Observer Implementation for Quantization`
> - `Blockwise Quantization Observer` 


- **INT8 Quantized Training Error Resolution**: A member faced an error when implementing FSDP support for a subclass related to **INT8 quantized training**, particularly when calling `torch.chunk()` on a tensor that requires grad.
   - They noted that setting `requires_grad=False` in their implementation of `aten.split.Tensor` resolved the issue, indicating some underlying logic in PyTorch managing this aspect.
- **Observer Usage in Static Quantization**: A member pointed out that in the static quantization calibration flow tutorial, the observers are imported from **torch.ao** instead of the **torchao** repository, suggesting confusion in availability of observer classes.
   - They mentioned the potential to manually calculate averages without an observer, commenting humorously on the ease of using `model(inputs).abs().mean()`.
- **Discussion on Blockwise Quantization Observer**: A need for reimplementing observers for blockwise quantization in **torchao** was highlighted, with a member stating their intention to create a general observer despite time constraints.
   - Another member shared a [GitHub Pull Request](https://github.com/pytorch/ao/pull/650) for an `AffineQuantizedObserver`, suggesting that customization might be necessary for specific applications, particularly Adaptive Weight Quantization (AWQ).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/650">Add AffineQuantizedObserver by jerryzh168 · Pull Request #650 · pytorch/ao</a>: Summary: In our static_quant flow tutorial we were still using observers from torch.ao which we plan to deprecate, this PR adds a more general observer for AffineQuantizedTensor, and has shown that...</li><li><a href="https://github.com/pytorch/ao/pull/644/files">Add experimental INT8 quantized training by gau-nernst · Pull Request #644 · pytorch/ao</a>: Address #554 (but don&#39;t close it)
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1271543496230043748)** (1 messages): 

> - `Python User Survey`
> - `Community Feedback on Free Threading` 


- **NVIDIA seeks Python user insights**: NVIDIA is gathering feedback from Python users through a short [survey](https://www.surveymonkey.com/r/FHF2RTL) to better understand their experiences and challenges with CUDA Python products.
   - The survey is anonymous, and responses will help prioritize future features based on community needs.
- **Community urged to discuss free threading**: A message emphasized the importance of community input on areas where assistance can be provided as the community transitions to **free threading**.
   - Engagement from users is encouraged to tailor support and resources effectively.



**Link mentioned**: <a href="https://www.surveymonkey.com/r/FHF2RTL">CUDA Python Survey</a>: Take this survey powered by surveymonkey.com. Create your own surveys for free.

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1271181581423349892)** (85 messages🔥🔥): 

> - `RoPE Implementation`
> - `KV Cache Optimizations`
> - `Complex Numbers in Code`
> - `Memory Management Techniques`
> - `Driver Issues with PyTorch` 


- **RoPE Refactoring Discussion**: Members debated the current implementation of the **RoPE kernel**, acknowledging its complexity and the potential benefits of using explicit trigonometry instead of complex numbers, focusing on improving code readability.
   - An older, simpler version was shared which avoided complex numbers entirely and implemented straightforward rotations.
- **KV Cache Integration**: A member mentioned successfully implementing their **KV Cache** optimizations for Llama31, enabling full **bfloat16 finetune** with substantial memory utilization on an 80GB GPU.
   - They expressed a desire to optimize memory usage further during training to alleviate constraints as they proceed with the proof-of-concept.
- **Concerns Over Complex Number Logic**: A discussion arose around the logic of using complex numbers in code, highlighting the necessity for clear comments to ensure understandability amongst team members.
   - Members agreed that simplifying the code to use sine and cosine might make it more approachable and easier to comprehend.
- **Memory Management in GPU Training**: A linked GitHub pull request discussed using **cudaMallocManaged** for managing memory when device memory runs out, allowing for continued training even on restricted systems.
   - The conversation suggested leveraging C libraries for allocations and using `torch.frombuffer` with dynamic memory management solutions.
- **Driver Compatibility Fixes**: One member resolved issues with **PyTorch** performance by reverting to standard driver settings and updating to ROCm 6.2 after experiencing problems with AMD drivers in passthrough mode.
   - This adjustment led to more consistent and reliable results in their usage of the llm.c framework.



**Link mentioned**: <a href="https://github.com/karpathy/llm.c/pull/709">Allocate managed memory if device memory runs out by ngc92 · Pull Request #709 · karpathy/llm.c</a>: Use cudaMallocManaged to allocate optimizer states if we run out of device memory, so we can still train (slowly) even if we cannot fit the optimizer state This is based on #694 , which should be m...

  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1271449050474086482)** (12 messages🔥): 

> - `Cloud GPU rental`
> - `Profiling issues with Runpod`
> - `MI250 availability`
> - `Cost-effectiveness of buying GPUs` 


- **Seeking recommendations for Cloud GPU rental**: A member expressed interest in renting a GPU for **CK**, particularly asking for suggestions on suitable providers.
   - They highlighted a preference for a service that wouldn't complicate **profiling** tasks.
- **Concerns about profiling on Runpod**: Another member suggested using an **MI250** from Runpod, but concerns were raised about profiling issues on that platform.
   - *‘I might be misremembering but iirc people were having a hard time profiling with Runpod’* was mentioned as a warning.
- **Limited availability of MI GPUs**: Members noted the lack of options for hosting **MI300** GPUs, mentioning the requirement to rent an entire machine due to limitations in **AMD's hypervisor**.
   - One user remarked, *‘AMD hasn't realized that you need to patch your hypervisor to enable PCIe atomics yet.’*
- **Potential lower cost alternative with 7900XTX**: A user suggested it might be cheaper to purchase a **7900XTX** rather than renting cloud GPUs due to cost concerns.
   - This remark reflects a growing sentiment towards evaluating cost versus cloud services.
- **Referral for more MI250 info from AMD side**: A recommendation was made to contact a member who works on **llmc** from AMD, indicating they might have insights on obtaining **MI250** GPUs.
   - Another user mentioned that this contact gets their **MI250** from **Hyperbolic**, potentially offering a valuable resource.


  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1271356591408676895)** (3 messages): 

> - `Bitnet model`
> - `AO integration`
> - `Quantization Aware Training (QAT)` 


- **Meeting Scheduled to Run Bitnet Model**: A meeting is scheduled for <t:1723309200:R> with the primary goal to get a working **bitnet model** running with **AO** integration.
   - One member expressed willingness to help, despite the meeting time being at **1AM** for them, humorously noting they would likely be asleep.
- **Insights on Bitnet's QAT Approach**: One member mentioned they've not read the **bitnet paper** in-depth but observed that they seem to be utilizing **Quantization Aware Training (QAT)**.
   - It was noted that the **master weight** is still maintained in **FP32/BF16**, indicating a specific approach to model precision.


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1271496058563727360)** (2 messages): 

> - `Pure UNet optimization`
> - `From scratch model implementations` 


- **Optimizing Pure UNet for speed**: A member expressed interest in a project to make [pure UNet](https://github.com/clu0/unet.cu) run faster than **torch.compile**.
   - *They're open to collaborating with others* to enhance the model's performance.
- **Excitement for from-scratch model implementations**: Another member highlighted that **from scratch model implementations** sound very cool as a project idea.
   - *This approach could lead to innovative solutions and deeper understanding* of various model architectures.


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1271190433388695678)** (98 messages🔥🔥): 

> - `CBRN Risks and AI Filtering`
> - `AI Safety Measures`
> - `Career Transition Grants in AI`
> - `AI Models and Ethical Guidelines`
> - `GPU Resources for AI Research` 


- **Debating CBRN Risks in AI Models**: There was an extensive discussion on whether filtering CBRN-related information from AI training data could effectively reduce risks without diminishing the model's scientific capabilities.
   - Participants argued about the complexities of knowledge removal versus the potential for models to still generate harmful outputs if they are sufficiently intelligent.
- **Opportunities for AI Safety Research**: A member highlighted a career transition grant from Open Philanthropy focused on AI safety and mechanistic interpretability, seeking GPU resources to assist with educational exercises.
   - Various alternatives for GPU access were discussed, including options like Colab, vast.ai, and CAIS clusters for AI research.
- **Alternative GPU Resources**: The community shared several GPU resource options for learners, including Kaggle for T4 GPUs and suggestions for utilizing Apple M1 onboard capabilities.
   - Suggestions included a mix of free tiers and rental options to support machine learning exercises efficiently.
- **Ethics in AI Research Output**: Conversations centered around ensuring that AI models do not generate harmful instructions, reflecting on the balance between necessary expertise and dangerous knowledge.
   - Some participants expressed concern that removing sensitive information could limit the understanding of negative examples and create potential information hazards.
- **Technological Capabilities of AI Code Editors**: A user inquired about AI code editors capable of manipulating files, images, and modding video games with extensive code generation capabilities.
   - The discussion touched on limitations of current LLMs and the search for tools that might overcome these restrictions in code generation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.safe.ai/work/compute-cluster">Compute Cluster | CAIS</a>: The Center for AI Safety is launching an initiative to provide large-scale compute resources for ML safety research. Apply here.</li><li><a href="https://arena3-chapter0-fundamentals.streamlit.app/#option-1-colab)">no title found</a>: no description found</li><li><a href="https://arena3-chapter0-fundamentals.streamlit.app/).">no title found</a>: no description found</li><li><a href="https://arena-uk.slack.com/archives/C058P58400K/p1686507915681109?thread_ts=1686121368.323589&cid=C058P58400K)">Slack</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1271211243369271307)** (18 messages🔥): 

> - `Synchronization of Model Curricula`
> - `Benchmarking and Evaluation Practices`
> - `Tree Attention Algorithm`
> - `Zamba Model Performance`
> - `UT-RNN Hybrid Implementations` 


- **Synchronizing Curricula for Models**: A member inquired about methods to synchronize curricula of two different models to ensure identical minibatches are used.
   - Another suggested recording the order and grouping of training data, while concerns were raised about deterministic behavior of seeded dataloaders.
- **Debate on Benchmarking Practices**: A member reported that their NeurIPS papers received reviews dismissing benchmarks and best practices as 'not real science'.
   - Another noted that taking benchmarks for granted could be ill-advised, recognizing possible misunderstandings around evaluation methodologies.
- **Tree Attention for Efficient Computation**: The discussion highlighted a paper that derives a scalar energy function for calculating self-attention, leading to a Tree Attention algorithm that optimizes performance via parallel computation.
   - The implementation promises improved efficiency in handling long-context attention on GPU clusters, with a relevant GitHub repository shared.
- **Zamba Model Surprises with Performance**: The team behind the Zamba model received attention for outperforming LLaMA 2 7B with fewer training tokens, despite minimal exposure.
   - The model's dataset has been made publicly available, generating interest due to its notable efficiency and performance.
- **Potential of UT-RNN Hybrid Models**: One user expressed excitement over the possibility of fine-tuning the Zamba model into a UT-RNN hybrid due to its architecture that utilizes shared attention and MLP blocks.
   - This design offers additional pathways for retaining input information, suggesting a promising direction for future model development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.04220">Diffusion Guided Language Modeling</a>: Current language models demonstrate remarkable proficiency in text generation. However, for many applications it is desirable to control attributes, such as sentiment, or toxicity, of the generated la...</li><li><a href="https://arxiv.org/abs/2408.04093">Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters</a>: Self-attention is the core mathematical operation of modern transformer architectures and is also a significant computational bottleneck due to its quadratic complexity in the sequence length. In this...</li><li><a href="https://github.com/Zyphra/tree_attention">GitHub - Zyphra/tree_attention: Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters</a>: Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters - Zyphra/tree_attention</li><li><a href="https://github.com/Zyphra/Zamba2">GitHub - Zyphra/Zamba2: PyTorch implementation of models from the Zamba2 series.</a>: PyTorch implementation of models from the Zamba2 series. - Zyphra/Zamba2
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1271443945372647546)** (7 messages): 

> - `GemmaScope paper`
> - `SAE training process`
> - `Model learning SO(3) group operations`
> - `Decomposing model activations`
> - `Sparse autoencoders` 


- **Curiosity About SAE Training Order**: A member asked why **SAE** is trained after **RMS norm** for MLP but before **RMS** for the attention layer, to which another member responded that it is to be positioned before **w_0** on the readout head for attention.
   - This highlights an intentional design choice to optimize how the model processes attention mechanisms.
- **Importance of Pre-Training for Attention Heads**: Discussion revealed that training **SAEs** before **w_0** enables certain advanced mechanisms in the **GemmaScope** paper, emphasizing the advantages of this approach.
   - This suggests that strategic training sequences can enhance model performance and interpretability.
- **Finding Papers on SO(3) Group Operations**: A member was searching for a paper demonstrating model learning of **SO(3)** group operations for representing rotations, and later found a relevant link, indicating surprise at its age.
   - This conversation underscores the continuing relevance of foundational research in contemporary machine learning discussions.
- **Recommendations for Related Papers**: Another member suggested a different paper related to symmetry, sharing a link that they particularly enjoyed, which parallels the search for explainable models.
   - This emphasizes the collaborative nature of the community as members actively support each other's research interests.



**Link mentioned**: <a href="https://arxiv.org/abs/2406.17759">Interpreting Attention Layer Outputs with Sparse Autoencoders</a>: Decomposing model activations into interpretable components is a key open problem in mechanistic interpretability. Sparse autoencoders (SAEs) are a popular method for decomposing the internal activati...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1271300413328719882)** (23 messages🔥): 

> - `Karpathy's nanoGPT evaluation`
> - `lm-evaluation-harness inconsistencies`
> - `Floating point discrepancies in evaluations`
> - `Neurips benchmark track reviews` 


- **Challenges with Karpathy's nanoGPT evaluation**: Several members discussed issues with using the **lm-evaluation-harness** for evaluating checkpoints of **Karpathy's nanoGPT model** and noted it isn't saved in an HF compatible way.
   - One user expressed struggle with getting the evaluation harness to work and sought assistance from others.
- **Inconsistencies in lm-evaluation-harness results**: A user shared findings of **inconsistencies** in evaluation results that seem linked to **batch size** and the number of GPUs utilized.
   - Another member suggested using a different few-shot sampling method for better **determinism** in their evaluations.
- **Floating point issues affecting evaluation outcomes**: Discussion arose around how slight variations in **floating point** calculations might lead to discrepancies in evaluation scores, although these should be minimal.
   - One member noted that **for a large dataset**, variations should not significantly impact the results, expecting only minor changes at the third decimal point.
- **Excitement over Neurips benchmark results**: A user received **scores of 6/6/5** and confidence ratings of **3/2/3** for their Neurips benchmark track reviews.
   - They were optimistic about their chances after a rebuttal, encouraged by another member who said those scores are quite promising.



**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#task-name--tags-registering-a-task">lm-evaluation-harness/docs/new_task_guide.md at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1271189123050246224)** (117 messages🔥🔥): 

> - `Low VRAM Mode`
> - `Face Swapping Tools`
> - `Stable Diffusion Performance`
> - `Custom Lora Commissions`
> - `Live Preview Settings in A1111` 


- **Low VRAM Mode Usage**: When reaching low VRAM mode, switching to a lower model might not be necessary if the generation completes successfully, but could save time with different versions.
   - *Experimenting with model options can help optimize performance.*
- **Face Swapping Tools Comparison**: For face swapping on a PC with Intel CPU, members suggested using [Rope](https://github.com/Hillobar/Rope) as it is easier to install than Roop.
   - The discussion highlighted the need for effective tools that are simple to set up for users interested in face swapping.
- **Stable Diffusion Performance Factors**: Users reported variances in sampling speeds (s/it), with some experiencing jumps in processing time when changing model sizes, leading to concerns over performance consistency.
   - *Insights into different setups and hardware configurations, like ROCm and WSL2, were also shared to gauge expected performance.*
- **Commissioning Custom Lora Models**: Participants discussed finding reliable commissions for creating custom pony lora models, suggesting using Civitai's bounty system to ensure a secure transaction.
   - *Several recommended creators with reputable commission practices, emphasizing the importance of thorough vetting.*
- **Live Preview Settings Queries**: A user inquired about the best live preview settings in A1111, questioning the purpose of certain formats and whether preview frames are saved on the drive.
   - *The discussion indicates a community interest in optimizing their image generation workflows for improved performance.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://civitai.com/search/models?sortBy=models_v9&query=baldur">Civitai | Share your models</a>: no description found</li><li><a href="https://www.shakker.ai/">Shakker AI - Premium Stable Diffusion Model Hub</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/windows/wsl/install">Install WSL</a>: Install Windows Subsystem for Linux with the command, wsl --install. Use a Bash terminal on your Windows machine run by your preferred Linux distribution - Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin,...</li><li><a href="https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html">Install Radeon software for WSL with ROCm &#8212; Use ROCm on Radeon GPUs</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=44waH3sDYOM">Tiled Diffusion with Tiled VAE / Multidiffusion Upscaler, the Ultimate Image Upscaling Guide [A1111]</a>: #aiart, #stablediffusiontutorial, #generativeartThis tutorial will cover how to upscale your low resolution images to 4k resolution and above with the Tiled ...</li><li><a href="https://github.com/Hillobar/Rope">GitHub - Hillobar/Rope: GUI-focused roop</a>: GUI-focused roop. Contribute to Hillobar/Rope development by creating an account on GitHub.</li><li><a href="https://github.com/apple/ml-mdm">GitHub - apple/ml-mdm: Train high-quality text-to-image diffusion models in a data &amp; compute efficient manner</a>: Train high-quality text-to-image diffusion models in a data &amp; compute efficient manner - apple/ml-mdm
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1271207192938287344)** (1 messages): 

> - `DALL·E 3 image generation`
> - `ChatGPT Free users` 


- **DALL·E 3 Available for Free Users**: ChatGPT Free users can now generate up to **two images per day** using **DALL·E 3**.
   - This update enables users to create images for various applications, such as *slide decks* or *personalized cards*.
- **Image Creation Simplified**: Users can simply ask ChatGPT to create an image according to their needs, enhancing user convenience.
   - This feature makes it easier to visualize concepts and enhance personal communications.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1271195097198891009)** (86 messages🔥🔥): 

> - `Mistral NeMo Performance`
> - `GPT-4 vs GPT-4o`
> - `Open WebUI & Ollama Integration`
> - `Neuroadaptive Language Research`
> - `Local AI Model Run Recommendations` 


- **Exploration of Mistral NeMo's Performance**: Members expressed curiosity about the performance of **Mistral NeMo** on different hardware, especially M1 machines with 16GB RAM.
   - A participant mentioned not being able to run larger models due to hardware limitations.
- **Concerns over GPT-4's Capabilities**: A user shared their frustrations about **GPT-4o**, describing it as less competent compared to **GPT-4**, especially in specific tasks like image analysis.
   - *GPT-4o* was criticized for rigid responses, drawing a comparison between it and a programmer losing touch with basic understanding.
- **Local AI Model Workflow**: One participant discussed transitioning to **Open WebUI** and **Ollama**, considering halting their ChatGPT+ subscription for local models.
   - They mentioned that running these models has been reliable, especially with *LLama*, but acknowledged the challenge of using self-hosted solutions.
- **Interest in Neuroadaptive Language Research**: A member is developing a model to translate between autistic and neurotypical perspectives and seeks collaboration with researchers in related areas.
   - A fellow participant expressed interest, indicating a connection based on personal experience as both autistic and a researcher.
- **Recommendations for Local AI Performance**: There was discussion regarding the performance of various GPUs for running AI models, with some suggesting using an **RTX 2060** for more efficient processing.
   - Participants shared concerns about slow response times when running models on Discord routes and discussed the importance of model quantization.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1271230573951582229)** (4 messages): 

> - `LangChain with CSV`
> - `ChatGPT issues on Safari` 


- **Resources for LangChain and CSV Integration**: A user sought resources for utilizing a **CSV file** as a *retrieval-augmented generation* (RAG) document in **LangChain** with **OpenAI**.
   - This indicates a growing interest in handling structured data with language models.
- **ChatGPT not working on Safari**: A user expressed frustration regarding repeated error messages while using **ChatGPT** on **Safari** on iOS for the past two days, noting it's very annoying.
   - Another user prompted them to consider using the **ChatGPT app** instead of the web version to avoid these issues.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1271360918655205387)** (4 messages): 

> - `Chat Prompt Library`
> - `Becoming a Prompt Engineer`
> - `Learning Resources for Prompt Engineering` 


- **Location of Chat Prompt Library**: A member inquired about the location of the **chat prompt library**.
   - Another member indicated that it has been **renamed** and pointed to a specific channel.
- **Advice for Aspiring Prompt Engineers**: A Greek undergraduate student expressed interest in becoming a **prompt engineer**, seeking guidance on where to start.
   - An experienced member suggested exploring **Arxiv** and **Hugging Face**, and highlighted the importance of getting into **meta-prompting**.
- **Discord Communities for Learning**: An experienced member mentioned that there are plenty of **Discords** dedicated to those interested in prompt engineering.
   - They encouraged looking into these communities for further learning resources beyond specific sites.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1271360918655205387)** (4 messages): 

> - `Chat prompt library`
> - `Becoming a prompt engineer`
> - `Learning resources for prompt engineering` 


- **Chat prompt library renamed information**: A member inquired about the location of the **chat prompt-library**.
   - Another member replied that it has been **renamed** and provided a link to its new location.
- **Path to becoming a prompt engineer**: A user from Greece expressed interest in becoming a **prompt engineer** despite its limited presence in the country.
   - In response, another member recommended starting with **Arxiv** and **Hugging Face**, as well as engaging with various Discord communities for support.
- **Meta-prompting as a powerful strategy**: A member highlighted the importance of getting into **meta-prompting** as a key strategy in prompt engineering.
   - They emphasized that there is a lot to learn in this field and encouraged further exploration through provided resources.


  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1271218232656920648)** (95 messages🔥🔥): 

> - `Gemini 1.5 Flash Performance`
> - `GPT-4o Mini vs Gemini 1.5`
> - `OpenRouter API Configuration`
> - `Dunning-Kruger Effect in Discussions`
> - `Model Recommendations for Japanese` 


- **Gemini 1.5 Flash Performance Discussed**: Several users noted that **Gemini 1.5 Flash** has seen a significant price drop, making it more competitive in the fast-and-cheap model market.
   - The updated model can now natively understand PDFs and has improved performance for text and multi-modal queries.
- **Comparing GPT-4o Mini and Gemini 1.5**: **GPT-4o Mini** was praised for its lower hallucination rates compared to **Gemini 1.5**, particularly in coding tasks.
   - Users expressed a preference for models that reduce hallucinations and streamline coding capabilities.
- **OpenRouter API Configuration Challenges**: A developer faced issues passing custom parameters, specifically the `providers` configuration, while using the **OpenAI SDK** in TypeScript.
   - It was mentioned that the API currently doesn't support these parameters natively, causing linting errors.
- **Dunning-Kruger Effect Highlighted in Discussion**: A humorous debate emerged where a participant used the **Dunning-Kruger Effect** to illustrate a point about self-assessment in discussions about expertise.
   - The conversation turned comedic as users reflected on confidence versus ability, particularly in the context of making money.
- **Seeking Recommendations for Japanese LLM**: A user inquired about LLM models that outperform **GPT-4o Mini** in Japanese, seeking alternatives within a similar price range.
   - The ongoing search reflects the demand for models that excel in specific languages beyond the general capabilities of larger models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://aider.chat`">no title found</a>: no description found</li><li><a href="https://aider.chat`.">no title found</a>: no description found</li><li><a href="https://simonwillison.net/2024/Aug/8/gemini-15-flash-price-drop/">Gemini 1.5 Flash price drop</a>: Google Gemini 1.5 Flash was already one of the cheapest models, at 35c/million input tokens. Today they dropped that to just 7.5c/million (and 30c/million) for prompts below 128,000 tokens. The …</li><li><a href="https://aitestkitchen.withgoogle.com/tools/image-fx">ImageFX</a>: no description found</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/document-understanding">no title found</a>: no description found</li><li><a href="https://openrouter.ai/models?modality=text%2Bimage-%3Etext">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://en.wikipedia.org/wiki/Dunning–Kruger_effect">Dunning–Kruger effect - Wikipedia</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Dunning%E2%80%93Kruger_effect">Dunning–Kruger effect - Wikipedia</a>: no description found</li><li><a href="https://tenor.com/view/kenan-thompson-kenan-thompson-snl-mmhmm-gif-704775788876592410">Kenan Thompson Snl GIF - Kenan Thompson Kenan Thompson - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/bigscience-workshop/petals">GitHub - bigscience-workshop/petals: 🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading</a>: 🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading - bigscience-workshop/petals</li><li><a href="https://openrouter.ai/models/01-ai/yi-vision">Yi Vision - API, Providers, Stats</a>: The Yi Vision is a complex visual task models provide high-performance understanding and analysis capabilities based on multiple images.  It&#x27;s ideal for scenarios that require analysis and interp...</li><li><a href="https://openrouter.ai/models/fireworks/firellava-13b">FireLLaVA 13B - API, Providers, Stats</a>: A blazing fast vision-language model, FireLLaVA quickly understands both text and images. It achieves impressive chat skills in tests, and was designed to mimic multimodal GPT-4. Run FireLLaVA 13B wit...</li><li><a href="https://openrouter.ai/settings/preferences">Settings | OpenRouter</a>: Manage your accounts and preferences</li><li><a href="https://openrouter.ai/docs/provider-routing#ignore-providers-for-a-request">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-api/">Gemini 1.5 Flash price drop with tuning rollout complete, and more</a>: no description found</li><li><a href="https://github.com/mlc-ai/web-llm?tab=readme-ov-file">GitHub - mlc-ai/web-llm: High-performance In-browser LLM Inference Engine</a>: High-performance In-browser LLM Inference Engine . Contribute to mlc-ai/web-llm development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1271221815028613132)** (8 messages🔥): 

> - `Welcome Messages`
> - `New sus-column-r model`
> - `Comparison with GPT-4 and Claude 3.5` 


- **Warm Welcome to New Members**: Multiple members welcomed new users to the server, creating a friendly atmosphere for newcomers.
   - The greetings included simple messages of 'Hello!' and 'Welcome to Cohere!' as an invitation for engagement.
- **Discussion on New sus-column-r Model**: A user shared a [Reddit link](https://www.reddit.com/r/LocalLLaMA/comments/1enmcr9/new_suscolumnr_model_on_lmsys_its_just_f_up/) discussing a new 'sus-column-r' model on LMSYS, claiming it outperforms GPT-4 and Claude 3.5 in various tasks.
   - *I don't understand how this is possible,* said the user, citing improved performance in **translation**, **coding**, and **mathematics**.



**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1enmcr9/new_suscolumnr_model_on_lmsys_its_just_f_up/">Reddit - Dive into anything</a>: no description found

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1271222019110993920)** (22 messages🔥): 

> - `Using Preamble ID`
> - `Response Quality in RAG`
> - `Cohere Embedding Models`
> - `Limiting Output Tokens`
> - `Structured JSON Outputs` 


- **Using Preamble ID for Context**: A user sought help to utilize a preamble ID to generalize prompts across inputs without regenerating a new preamble each time, asking for a way to maintain context.
   - Another member suggested using a **conversation ID** which resolved the user's issue.
- **Challenges with RAG Response Quality**: A community member reported issues with response generation in a Retrieval-Augmented Generation (RAG) setup, noticing that hallucinations increased when not enough information was present.
   - Despite following the guidelines, the AI did not adhere to factual information, prompting suggestions for modifications to the prompt.
- **Concerns with Cohere's Embedding Models**: A user discussed their experience switching from the `embed-english-light-v2.0` to `embed-english-light-v3.0` models and observed a decline in retrieval quality despite expectations of improvement.
   - They provided details about their dataset and usage, indicating that the newer models were not performing adequately.
- **Limiting Output Tokens in Command-R Model**: A user inquired about how to limit output tokens for the **command-r** model, mentioning that the `max_tokens` parameter did not seem effective.
   - After sharing their API call, confirmation was requested about whether the output exceeded the specified token limit.
- **Structured JSON Outputs Discussion**: A member discussed attempting to achieve structured JSON outputs, indicating their prompt was set for a classification task requiring binary outputs.
   - They sought clarification on whether their existing outputs went beyond the expected `Yes/No` format.



**Link mentioned**: <a href="https://docs.cohere.com/reference/chat">Chat - Cohere API References</a>: Generates a text response to a user message. To learn how to use the Chat API with Streaming and RAG follow our Text Generation guides .

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1271349639609581568)** (23 messages🔥): 

> - `403 Forbidden error`
> - `VPS connection issues`
> - `Langchain multistep tool use error` 


- **403 Forbidden Error Troubleshooting**: Members discussed encountering a **403 Forbidden** error when trying to post requests using curl, with suggestions stating it usually indicates an **invalid API key** or geolocation restrictions.
   - One member confirmed their VPS location is in the US, while others mentioned that the same request worked on their local machines, leading to confusion about the cause.
- **Discussion on using VPS for API requests**: There were mentions that **403 errors** could also arise from the VPS's IP address location possibly being restricted, prompting requests for more information about the user’s location.
   - Despite the discussions, members were unable to resolve the error, noting that previous API requests returned a stream response without issues.
- **Errors with Langchain's Multistep Tool Use**: One member reported an error while using **Langchain** with the **multistep_tool_use** functionality, specifically receiving the message: `ERROR:root:Failed to parse multihop completion for input`.
   - They sought help from others for potential fixes or references to better documentation regarding the integration of **Cohere** and **Langchain**.



**Link mentioned**: <a href="https://docs.cohere.com/docs/implementing-a-multi-step-agent-with-langchain">Implementing a Multi-Step Agent with Langchain</a>: In this document, we'll go through the nuts-and-bolts of building a generative-AI agent with Cohere's multi-step tool use functionality and the Langchain framework. Building the Langchain ReAct Agent ...

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1271457209863372875)** (3 messages): 

> - `Docker Installation Issues`
> - `Backend Setup Concerns` 


- **Docker installation raises questions**: A user expressed confusion over their interface not being operable after installing with **Docker**, asking if they missed something.
   - This prompted **Nick Frosst** to suggest that the issue likely lies in the backend setup, though he admits uncertainty regarding the specifics.
- **Possible backend misconfiguration indicated**: Nick Frosst responded to the user's query by implying that there may be a misconfiguration with the **backend** setup.
   - He acknowledged his uncertainty about the exact issue, indicating an ongoing discussion around resolving the problem.


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1271257088106168465)** (4 messages): 

> - `Event-Driven Agent Systems`
> - `Mixture-of-Agents`
> - `Property Graphs`
> - `Multimodal RAG Pipelines` 


- **Flexibility with Event-Driven Agent Systems**: Building agents in an **event-driven manner** provides users with greater flexibility to create **cyclic, multi-agent systems** with complex communication patterns. Check out this [awesome tutorial video](https://t.co/he0Az19WJS) comparing graph-based agent programming.
   - _“This is an awesome tutorial video”_ shows the benefits of using an event-driven approach in agent systems.
- **Mixture-of-Agents Outperforms Larger Models**: A new paper by [Junlin Wang](https://t.co/C8pZBBIcOk) presents how to ensemble smaller LLMs to form a **Mixture-of-Agents** system that can outperform state-of-the-art larger models. This has been implemented in a fully async, **event-driven workflow**.
   - The implementation is discussed in detail on [Twitter](https://t.co/hNpoZuBC5l) for those interested in practical applications.
- **Understanding Property Graphs for GraphRAG**: An important [video tutorial](https://t.co/CdWqPOxt4c) on LlamaIndex's **property graphs** explains how each node and relation can store a structured dictionary of properties. This foundational knowledge is essential before diving into **GraphRAG**.
   - _“This underlying abstraction unlocks a lot of cool techniques”_ emphasizes the functionality of property graphs.
- **Building Multimodal RAG Pipelines**: Exciting notebooks are available that explain how to build practical, real-world **multimodal RAG** pipelines over complex **legal, insurance, and product documents**. The series begins with parsing **insurance claims**.
   - Follow this [link](https://t.co/1w4IfXQ7CP) for detailed breakdowns and real-world use cases.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1271188452834279484)** (45 messages🔥): 

> - `Embedding Models and Document Retrieval`
> - `Using Llama-Index with Multi Models`
> - `Configuring Filters in Query Engines`
> - `Ingesting Language Documents`
> - `RAG Pipeline Workflows` 


- **Selecting Embedding Models in Llama**: A member discussed using the [HuggingFaceEmbedding](https://github.com/openai/tiktoken) model with Llama, showcasing an example of loading a specific model for embedding documents before making query calls.
   - Another user inquired about document loading and retrieval after embeddings have been done, clarifying the sequential process needed for achieving desired outcomes.
- **Image Querying with Multi-Modal Models**: A user expressed interest in utilizing a multi-model approach in Llama by querying an image but faced issues with configuring the proxy for OpenAIMultiModal LLM.
   - They encountered challenges with synchronous versus asynchronous client requirements when trying to set up http_client with httpx.
- **Filtering Documents in Query Engines**: A user shared their experience regarding node filtering during document ingestion and aimed to retrieve specific documents based on `business_id` metadata.
   - Another member suggested implementing `MetadataFilters` during retrieval, emphasizing that filtering should ideally occur before retrieval to be effective.
- **Ingesting German Language Documents**: A member struggled with ingesting German language documents into a vector database, discovering that the summaries were being returned in English despite specifying German in their code.
   - They received advice to update the prompt for the summarization process to ensure outputs remain in the original language of the documents.
- **RAG Pipeline Workflows Implementation**: Discussions emerged about the new workflows architecture enhancing the COA (Chain of Abstraction) approach, focusing on executing in steps for iterative refinement.
   - Members agreed on the potential for utilizing agents and workflows creatively but acknowledged the need for proper documentation or examples for implementation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/abacaj/status/1821626025584828747?s=46&t=lqHyE7PE7Ct6jUJNcJPPxg">Tweet from anton (@abacaj)</a>: 42 page PDF (it says 30k tokens) that contains both images and text thrown into gemini-flash 1.5, gets every answer correct... it&#39;s so over</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/workflow/rag/">RAG Workflow with Reranking - LlamaIndex</a>: no description found</li><li><a href="https://github.com/openai/tiktoken">GitHub - openai/tiktoken: tiktoken is a fast BPE tokeniser for use with OpenAI&#39;s models.</a>: tiktoken is a fast BPE tokeniser for use with OpenAI&#39;s models. - openai/tiktoken</li><li><a href="https://github.com/IntelLabs/RAGFoundry">GitHub - IntelLabs/RAGFoundry: Framework for specializing LLMs for retrieval-augmented-generation tasks using fine-tuning.</a>: Framework for specializing LLMs for retrieval-augmented-generation tasks using fine-tuning. - IntelLabs/RAGFoundry</li><li><a href="https://github.com/run-llama/llama_index/blob/8a48fdc6f1086847ea7c0c9ef0df955b347ec366/llama-index-core/llama_index/core/extractors/metadata_extractors.py#L348">llama_index/llama-index-core/llama_index/core/extractors/metadata_extractors.py at 8a48fdc6f1086847ea7c0c9ef0df955b347ec366 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1271209109869559860)** (21 messages🔥): 

> - `Hackathon Announcement`
> - `Open Interpreter functionalities`
> - `MiniCPM-V model`
> - `Terminal agent environment`
> - `Linux support request` 


- **Exciting Hackathon Announcement!**: Open Interpreter is participating in a **big** hackathon, 'Breaking Barriers: A generative AI Hackathon for Digital Inclusion', happening in Dallas, Texas from Sept 20-23, with prizes totaling **$17,500**.
   - Participation is favored in-person, though applications for remote attendance are encouraged; discussions are ongoing about team formation in the community.
- **MiniCPM-V Model Surpasses Others**: A member highlighted that the open-source vision multi-image model, **MiniCPM-V 2.6**, is reported to outperform models like **Gemini 1.5 Pro** and **GPT-4V** according to their claims.
   - Links to both the [Hugging Face model](https://huggingface.co/openbmb/MiniCPM-V-2_6) and [GitHub repository](https://github.com/OpenBMB/MiniCPM-V) were shared for further exploration.
- **Concerns with Ollama Performance**: Members expressed frustration regarding slow performance of **Ollama** on Windows when running via `interpreter --codestral` or LM Studio.
   - Alternative solutions or workarounds were sought to enhance the user experience.
- **Request for Linux Support Channel**: A community member requested the creation of a dedicated **#linux-something_or_other** channel to facilitate focused discussions on Linux-related topics.
   - The suggestion received a positive response, pointing to an existing channel for troubleshooting.
- **Terminal Agent Environment Features**: A member demonstrated the capabilities of a terminal agent environment, showcasing screenshot features and highlighting cursor visibility in grayscale augmented settings.
   - Links to relevant GitHub issues and screenshots were shared to illustrate the functionalities described.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/monkey-monkey-eating-monkey-eating-strawberries-kardie-gif-gif-22488578">Monkey Monkey Eating GIF - Monkey Monkey Eating Monkey Eating Strawberries - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://ollama.com/aiden_lu/minicpm-v2.6">aiden_lu/minicpm-v2.6</a>: MiniCPM-V 2.6 is the latest and most capable model in the MiniCPM-V series. It exhibits a significant performance improvement over MiniCPM-Llama3-V 2.5</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/issues/1390">Add real terminal support · Issue #1390 · OpenInterpreter/open-interpreter</a>: Is your feature request related to a problem? Please describe. OpenInterpreter currently is unable to interact with common REPL and shell environment in an asynchronous way. It is always blocking. ...</li><li><a href="https://huggingface.co/openbmb/MiniCPM-V-2_6">openbmb/MiniCPM-V-2_6 · Hugging Face</a>: no description found</li><li><a href="https://github.com/OpenBMB/MiniCPM-V">GitHub - OpenBMB/MiniCPM-V: MiniCPM-V 2.6: A GPT-4V Level MLLM for Single Image, Multi Image and Video on Your Phone</a>: MiniCPM-V 2.6: A GPT-4V Level MLLM for Single Image, Multi Image and Video on Your Phone - OpenBMB/MiniCPM-V
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1271413670571933696)** (1 messages): 

> - `ESP32S3`
> - `O1 Integration` 


- **User seeks help running O1 on ESP32S3**: A member expressed interest in running **O1** on the **ESP32S3** and inquired if others have tried this setup.
   - They requested assistance from the community for implementing this on their device.
- **Community Request for ESP32S3 Experiences**: There is a call for members who have experience with the **ESP32S3** to share their insights on running **O1**.
   - The user is looking to leverage collective knowledge for better implementation strategies.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

8i8__papillon__8i8d1tyr: https://www.youtube.com/watch?v=V5kAmFRwuxc
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1271209872326791272)** (18 messages🔥): 

> - `LangChain API Differences`
> - `Anthropic Claude 3.5 Downtime`
> - `Disconnect between Discord and Product Announcements`
> - `LangChain Support and Documentation Issues`
> - `Community Support for LangChain` 


- **LangChain struggles with LLM feature consistency**: A member expressed confusion about **LangChain's** capability to provide a uniform API for all LLMs, noting it works for **OpenAI** but not for **Anthropic**.
   - Another member confirmed that while function calls may be similar, prompt modifications are necessary due to differences across LLMs.
- **Anthropic's Claude 3.5 experiences significant downtime**: A member reported that **Anthropic’s Claude 3.5** had been down all day, citing an internal server error with code **500**.
   - They shared the error message, indicating issues with the API that hindered functionality.
- **Discord discussions vs. official product announcements**: Concerns were raised about a disconnect between the **Discord conversations** and the official product announcements on LinkedIn regarding **LangGraph Cloud** and **Studio**.
   - A member questioned the clarity and consistency of the information being shared across platforms.
- **Frustration with LangChain's tool and documentation**: After giving **LangChain** another try, a member reported inconsistent tool/function calling and pointed to inadequate examples and documentation as the main obstacles.
   - They inquired about available commercial support for better assistance with the platform.
- **Community support for LangChain dwindling**: A member lamented the **declining community support** for **LangChain**, referencing perceptions from Hackernews that it has lost traction despite initial promise.
   - They expressed a desire for collaboration, citing personal struggles and seeking paid assistance regarding LangChain utilization.


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1271211436701515898)** (4 messages): 

> - `CTF Challenge`
> - `Mood2Music Dashboard`
> - `CRAB Benchmark` 


- **Join the $1000 CTF Challenge!**: Participants are encouraged to engage in a **capture-the-flag (CTF) challenge** with a goal to extract a password from an AI agent, with a prize of **$1000**.
   - The challenge explores the risks of accidentally leaking secrets through user feedback forms, raising questions about data privacy and security.
- **Mood2Music Dashboard Revealed**: An exciting preview of the **Mood2Music** dashboard showcases AI-powered song recommendations that connect to **Spotify** and **Apple Music** based on user mood.
   - The tool enhances listening experiences by creating playlists tailored to users' emotions, aiming to alleviate **decision fatigue** in music selection.
- **Introducing CRAB: The Multimodal Agent Benchmark**: The new **CRAB** framework allows for building and evaluating multimodal language model agents across multiple environments such as **Android** and **Ubuntu**.
   - It features a **fine-grain** evaluation metric, task generation capabilities, and aims to enhance human-like task execution, with resources available on GitHub and the project's [website](https://crab.camel-ai.org/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/camelaiorg/status/1821970132606058943?s=46">Tweet from CAMEL-AI.org (@CamelAIOrg)</a>: Introducing 🦀 CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents  🦀 CRAB provides an end-to-end and easy-to-use framework to build multimodal agents, operate environments, ...</li><li><a href="https://invariantlabs.ai/ctf-challenge-24">Fool an Agent to Extract the Secret Password</a>: Announcing the first Invariant Capture The Flag (CTF) challenge for agentic systems security.</li><li><a href="https://mood2music.me">mood2music</a>: no description found
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1271209576104198144)** (18 messages🔥): 

> - `Image Datasets Comparison`
> - `Model Steering with Gemma`
> - `Captions and Reliability`
> - `LAION Database Discussion` 


- **Image Datasets: CC vs LAION**: A discussion arose regarding whether the [Fondant 25M dataset](https://huggingface.co/datasets/fondant-ai/fondant-cc-25m) is the largest collection of creative commons/public domain images.
   - Concerns were raised about the **LAION-5B** dataset being less reliable due to its reliance on alt text that often doesn't relate to the image.
- **Steering Models like Gemma**: One member inquired if anyone had tried steering a model like **Gemma 2 2B** using the **Gemma Scope**, and how to make effective control vectors for output generation.
   - They expressed a need for insights beyond simple Google queries to enhance their understanding of model features.
- **Trust Issues with Captions**: Participants discussed the unreliability of mass-captured captions, emphasizing that all captions on this scale may lack accuracy.
   - The conversation included whether using clip similarity scores could help determine if new captions are **less reliable than originals**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2211.15006">Fine-tuning language models to find agreement among humans with diverse preferences</a>: Recent work in large language modeling (LLMs) has used fine-tuning to align outputs with the preferences of a prototypical user. This work assumes that human preferences are static and homogeneous acr...</li><li><a href="https://huggingface.co/datasets/fondant-ai/fondant-cc-25m">fondant-ai/fondant-cc-25m · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

nodja: https://research.google/blog/halva-hallucination-attenuated-language-and-vision-assistant/
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1271190643984568320)** (2 messages): 

> - `Sequoia Capital funding`
> - `AI reasoning startup`
> - `Chain of thought in AI` 


- **Sequoia Capital eyes funding for AI reasoning startup**: Sequoia Capital has discussed funding an AI reasoning startup co-founded by **Robinhood's CEO**, as reported in [The Information](https://www.theinformation.com/articles/sequoia-capital-has-discussed-funding-ai-reasoning-startup-cofounded-by-robinhood-ceo).
   - The startup aims to enhance AI capabilities in **reasoning** and **decision-making**.
- **Insights from Ross on Chain of Thought**: An interview with Ross highlighted that the **chain of thought** technique maintains context in tokens rather than the latent space, presenting a new perspective on AI processing.
   - *I never thought about it that way* emphasizes the importance of understanding context within AI mechanisms.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1271461440175341601)** (12 messages🔥): 

> - `Anaconda Software Licensing`
> - `Alternative to pip` 


- **Anaconda enforces commercial licenses on research organizations**: Research and academic organizations are learning they must now pay for software made by **Anaconda**, previously thought to be free, as the company pursues violators of its terms-of-service.
   - A source reported receiving a legal demand for a commercial license, noting that their institution was warned about potential 'back bills' for unauthorized usage.
- **Clarification on license requirements for containers**: Users are questioning whether using Anaconda on **Docker containers** requires additional licensing, suggesting that it likely does.
   - Members noted that the citations regarding licensing issues were not directly from Anaconda but from affected organizations.
- **Switching to uv for faster installations**: A member suggested considering **uv** as an alternative to **pip** for installations, highlighting that it is significantly faster.
   - They mentioned that using uv requires no additional tooling, allowing users to simply replace `pip` with `uv pip` for their installations.



**Link mentioned**: <a href="https://www.theregister.com/2024/08/08/anaconda_puts_the_squeeze_on">Anaconda puts the squeeze on data scientists</a>: Academic, non-profit organizations told to start paying up – or else

  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1271629008773316712)** (1 messages): 

> - `Bad Takes`
> - `Improvement in Discourse` 


- **World would improve without bad takes**: *If everyone who had** bad takes **exclusively had bad takes**, the world would be a lot better lol.* This reflects a humorous take on the impact of negative opinions in discussions.
   - The statement suggests that the quality of discourse might significantly improve if only those with poor opinions participated.
- **Humor in Discussion**: The phrase demonstrates the use of **humor** to critique current discourse trends, highlighting a preference for more constructive opinions.
   - It encapsulates a sentiment many share about wanting more positive engagement in community conversations.


  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/)** (1 messages): 

chygao: https://youtu.be/6QWuJRvMtxg?si=SYXsRvYbfcdtYLC2
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1271450524369162280)** (3 messages): 

> - `DSPy Tutorial`
> - `OpenAI Structured Output API` 


- **DSPy Tutorial walkthrough**: A member shared a [YouTube tutorial](https://youtu.be/_ROckQHGHsU) on DSPy, covering major concepts through **8 examples**, ranging from basics to advanced LLM projects.
   - This tutorial aims to help viewers grasp the complexities of DSPy in a structured format.
- **Experimenting with New OpenAI API**: Another member announced their experimentation with the new **structured output API** from OpenAI while in the voice lounge.
   - This API aims to enhance how users interact with and utilize structured data outputs in their projects.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1271296628585660437)** (8 messages🔥): 

> - `DSPy Prompt Improvement`
> - `Tutorial on DSPy Concepts`
> - `DSPy Use Cases`
> - `Signature Adapters`
> - `RAG Optimization` 


- **Improving DSPy Prompts with Custom GPT**: A member is seeking advice on enhancing a complex prompt that interleaves instructions and examples, mentioning potential use of Signature adapters and MIPRO optimization.
   - Another member suggested starting with a [custom GPT guide](https://chatgpt.com/g/g-cH94JC5NP-dspy-guide-v2024-2-7) for modularizing prompts.
- **DSPy Tutorial Available on YouTube**: A member shared a tutorial on their channel explaining major DSPy concepts through eight examples, progressing from basic to advanced LLM projects, viewable [here](https://youtu.be/_ROckQHGHsU).
   - Another member expressed support by subscribing to the channel.
- **Understanding DSPy Use Cases for RAG**: A member inquired about the use cases of DSPy and its suitability for RAG tasks.
   - In response, another member clarified that it's similar to fine-tuning, where tasks, metrics, and examples are optimized for better LLM performance.
- **Exploring Signature Adapters**: Members discussed the potential of using Signature adapters in customizing instructions for DSPy prompts.
   - A specific link for a related [Signature GPT resource](https://chatgpt.com/g/g-JQgwRHI0D-dspy-signature-gpt-v2024-2-21) was shared for further exploration.


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1271235461368778772)** (8 messages🔥): 

> - `Poe Hackathon`
> - `Alliance AI-Health Research Initiative` 


- **Poe announces innovative hackathon**: Poe (@poe_platform) is hosting a one-day [hackathon](https://x.com/poe_platform/status/1820843642782966103) focused on developing generative UI experiences using advanced LLMs like **GPT-4o** and **Gemini 1.5 Pro**.
   - The in-person portion will be held in **Hillsborough, CA 94010**, with details exclusive to registered participants.
- **Internship opportunity in AI and Health**: The **Alliance AI-Health Research Initiative** is seeking students for a **4-month remote internship** to conduct pioneering research on projects like cancer detection and AI-based heat stroke detection.
   - Interested applicants can [apply here](https://tinyurl.com/applyalliance) by **August 11** for a chance to publish their findings in an academic journal.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/poe_platform/status/1820843642782966103">Tweet from Poe (@poe_platform)</a>: We’re excited to announce a one-day hackathon with @agihouse_org around our new Previews feature! Compete to create the most innovative and useful in-chat generative UI experiences using the latest LL...</li><li><a href="https://tinyurl.com/applyalliance">no title found</a>: no description found</li><li><a href="https://x.co">Sell Domains | Buy Domains | Park Domains</a>: no description found
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1271668086008840304)** (1 messages): 

> - `Feature Stores in Computer Vision` 


- **Evaluating Feature Stores for Computer Vision**: A member inquired about the use of **feature stores** in **computer vision**, seeking insights on their value and effectiveness.
   - The discussion opens up the potential benefits and considerations of integrating feature stores for managing and optimizing computer vision projects.
- **Interest in Practical Implementations**: There was a call for examples of **practical implementations** of feature stores within computer vision frameworks to assess their impact.
   - This highlights the need for real-world case studies to validate the effectiveness of feature stores in specific applications.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1271226593930510437)** (8 messages🔥): 

> - `Modular Licensing`
> - `Future of Modular's AI Applications`
> - `Triton Language`
> - `Custom Kernels` 


- **Modular's License Permissiveness Under Scrutiny**: A member highlighted that Modular's license for using **max/mojo** is permissive unless you're attempting to commercialize an AI infrastructure platform.
   - Members are questioning what happens if Modular expands into other domains, such as **robotics** or **AI labeling platforms**.
- **Non-Competitive Software Could Become Competitive**: Discussion revealed that if software is not competitive today, but **becomes competitive** in the future, it remains non-competitive under Modular's licensing agreement.
   - However, questions arose about whether development on such software must be *frozen* once it turns competitive.
- **Call for Triton Lang Custom Kernel Users**: A request was made for **Triton lang** users who have written a custom kernel to participate in a one-on-one conversation with the product team.
   - Incentives include receiving some **Mojo swag** for their contributions.
- **Initial Awareness of Triton Language**: A member expressed curiosity, noting it was their **first time hearing about Triton**.
   - This indicates a potential interest in expanding knowledge about newer languages and technologies within the community.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1271192271496155290)** (5 messages): 

> - `Google Gemini Price Cuts`
> - `Comparison of Gemini and GPT-4o`
> - `Gemini 1.5 Free Finetuning` 


- **Impressive Cuts in Google Gemini Pricing**: The [YouTube video titled 'Google Gemini Insane Price Cuts!!!'](https://www.youtube.com/watch?v=3ICC4ftZP8Y) highlights significant price reductions for **Google Gemini 1.5 Flash**.
   - Details about these changes were also shared in the [Google Blog](https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...).
- **Confusion over Comparing Gemini to GPT-4o**: There's discussion regarding the comparison of **Gemini 1.5 Flash** to **GPT-4o**, debating whether it should be **Gemini 1.5 Pro** vs **GPT-4o** instead.
   - One member questioned if the correct comparison should separate standard and mini versions.
- **Free Finetuning of Gemini 1.5 at Play**: A participant suggested the unusual comparison was due to **Gemini 1.5's free finetuning** feature, unlike the Pro version.
   - This distinction seems to be influencing the ongoing conversation about the capabilities and offerings of the Gemini models.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=3ICC4ftZP8Y">Google Gemini Insane Price Cuts!!!</a>: Google Gemini 1.5 Flash has some insane price cuts!🔗 Links 🔗Details - https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/1271325299979714570)** (1 messages): 

> - `Llama CPP Server`
> - `Prompt Caching`
> - `RAG-Based Interaction`
> - `Gemma 2` 


- **Inquiring about Llama CPP Prompt Caching**: A member expressed confusion regarding which arguments to use for caching prompts with the **Llama CPP server**, seeking clarification on the appropriate settings.
   - Their goal is to cache the initial prompt while allowing **Llama CPP** to manage dynamic prompt content.
- **Desire for Selective Prompt Caching**: The member clarified they do not want to cache all user interactions but focus on the first user prompt, which is notably larger at **1.5k tokens**.
   - They are exploring the possibility of saving the first/system prompt in a file while leveraging **Llama CPP** for subsequent dynamic updates.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1271181377437696195)** (2 messages): 

> - `Llama 3 Model Details`
> - `Citing Axolotl in Academic Work` 


- **Inquiries about Llama 3 training details**: A member asked if there is any documentation related to the training process of the [Llama 3 model](https://huggingface.co/axolotl-ai-co/llama-3-8b-chatml) by Meta, particularly regarding the data and masks used.
   - They noted the unique approach of renaming existing tokens to serve as special tokens in the model.
- **Citation preferences for Axolotl**: Another member sought guidance on the preferred method to cite Axolotl in an academic paper or technical report.
   - This indicates a growing interest in formally acknowledging the Axolotl project in scholarly work.



**Link mentioned**: <a href="https://huggingface.co/axolotl-ai-co/llama-3-8b-chatml">axolotl-ai-co/llama-3-8b-chatml · Hugging Face</a>: no description found

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

drose0933: Yoooo
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1271297363029393481)** (4 messages): 

> - `AMD backend memory usage`
> - `GPU failure`
> - `De-sharding models`
> - `copy_to_device function` 


- **AMD backend potentially uses more memory**: A member questioned whether the **AMD backend** consumes more memory compared to the **GPU backend**.
   - This sparked discussions about resource allocation and performance across different backends in terms of memory management.
- **GPU failure reported**: A member lamented that their **GPU** was damaged, stating simply, *'Rip my GPU got blown.'*
   - This raises concerns within the community about GPU reliability and the challenges faced during intense computation.
- **De-sharding models for simplicity**: A user inquired about how to 'de-shard' a model, specifically by turning a **multi lazy buffer** into a **normal lazy buffer**.
   - This reflects ongoing challenges in model optimization and architecture adaptation within the community.
- **Usage of copy_to_device function**: A mention of **copy_to_device** surfaced, possibly hinting at its relevance in data handling during model operations.
   - This suggests a need for clarity among users regarding memory management techniques in their workflows.


  

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

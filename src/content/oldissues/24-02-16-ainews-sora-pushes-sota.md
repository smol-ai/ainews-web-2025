---
id: efb96740-607d-4a0a-98d5-26ccac6e392d
title: Sora pushes SOTA
date: '2024-02-16T11:15:03.174687Z'
original_slug: ainews-sora-pushes-sota
description: >-
  Google's Gemini 1.5 model, featuring a groundbreaking 1 million token context
  window, and OpenAI's Sora model have sparked significant excitement and
  discussion in AI communities, highlighting their potential for advanced media
  production and long-context applications. Meanwhile, NVIDIA's "Chat with RTX"
  feature showcases retrieval-augmented generation (RAG) capabilities on 30+
  series GPUs, contrasting with LM Studio's current support limited to smaller
  models like Mistral 7b and Llama 13b.
tags:
  - google
  - gemini-1.5
  - openai
  - sora
  - nvidia
  - chat-with-rtx
  - retrieval-augmented-generation
  - mistral-7b
  - llama-13b
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/13-15/2024. We checked **20** guilds, **312** channels, and **10550** messages for you. Estimated reading time saved (at 200wpm): **909 minutes**. Due to a config bug we summarized 2.5 days worth of conversations by accident.

If you're reading this you probably are aware of the absolute mayhem unleashed the day after Valentine's. We covered Gemini 1.5 and Sora on a [live ThursdAI podcast](https://sub.thursdai.news/p/thursdai-feb-15-2024-openai-changes) so you can get our takes there, but also we have been tracking the must-see and must-know Sora takes on the Latent Space discord. Of course, we weren't alone. 

 ![image.png](https://assets.buttondown.email/images/197545b3-2b72-41f1-937c-81c88827e5d9.png?w=960&fit=max) 

This was a rough day to launch anything if you aren't a frontier model lab.

---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Dungeon Master AI Development Queries**: Engineers discussed the creation of a **Dungeons and Dragons DM assistant** capable of cross-referencing rules, locating stat blocks, and accessing lore, while also producing new world-building content. Models like **H20 GPT** were suggested for document question answering, with challenges such as understanding tables and cross-references being noted.

- **Power Supply Concerns for High-End GPUs**: There's a debate on the feasibility of running a **3090 and 3060 GPU** on a **650W PSU**. Engineers stressed the risks involved and the importance of cable management, highlighting the potential power limitations to 250W for the 3090 and 150W for the 3060 to avoid overtaxing the PSU.

- **Cutting-Edge AI Sparks Discussion**: The guild examined Google's **Gemini 1.5** and OpenAI's **Sora**—impactful technologies with applications ranging from long token context handling to generating detailed minute-long video simulations. There's anticipation for their use in serious media production and cautious optimism expressed by the community.

- **Complexities of Running Large World Models (LWM)**: Members shared struggles with the multimodal functionality of **LWM**, discussing computational resource limits and technical intricacies. There is a shared experience of difficulties in making these complex models operational.

- **GPT-Assisted Coding**: The utility of LLMs like **GPT** and **Copilot** for coding support was debated. Some members valued these tools for initial code drafting and documentation, while others pointed out limitations such as missing edge cases, suggesting that these models are complementary tools rather than replacements for human expertise.

- **Role-Play Model Optimization Techniques**: For role-play and story-writing, different settings were debated for various models like **Yi models** and **Mixtral Instruct**. Adjustments in temperature settings between 0.6 to 0.7 were recommended, with an emphasis on balancing diversity and coherence in outputs.

- **Exploration of Training Effects**: Guild members encountered higher training and evaluation losses with **Mixtral** versus **Mistral**, with experiments ongoing to determine whether it's a bug or a feature of setup choices. Other discussions involved data cleaning for datasets using tools like **pandas** and **regex**, and the use of **M2 Max** for running 70B models, with tips shared on increasing RAM usage.

- **Discord Bot Fine-Tuning**: Information and potential scripts to fine-tune models were shared, including **"AutoFineTune"**, a script capable of generating synthetic message pairs for smaller models, discussed as part of an effort to simplify the fine-tuning process.

- **Model Merging Hurdles**: An encountered **RuntimeError** was shared when attempting to merge two **MistralCasualML** models with differing context sizes, highlighting a tensor size mismatch. Community members were seeking solutions to this and related issues.

- **JSPyBridge Facilitates Cross-Language Engineering**: Engineers shared success in integrating Python and JavaScript through **JSPyBridge**, demonstrating pragmatic examples such as creating new JavaScript classes that interact with Python, adjusting **BigDL's LLM transformer** for specific file types, and handling device tensor discrepancies—all critical details for AI engineers looking to interweave diverse technologies.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

**Chat with RTX Generates RAG Excitement**: NVIDIA's "Chat with RTX" feature, utilizing retrieval-augmented generation (RAG) on Nvidia 30+ series GPUs, has been contrasted with LMStudio, which supports RAG but is currently limited to Mistral 7b and Llama 13b models.

**Geminis Giant Leap in Context**: Conversations are abuzz with Google's Gemini 1.5 model boasting a 1 million token context window; access remains invite-only, underscoring the gap between proprietary and open-source AI tools.

**Sora's Synthetic Cinema**: OpenAI's Sora model, capable of generating videos from text up to a minute long, is on engineers' radars. With availability initially to a select group, its implications for evidence credibility are under scrutiny.

**Model Support and LM Studio Features in Spotlight**: Yi-VL models are pending an update to be compatible with LMStudio due to new `llama.cpp` requirements. Meanwhile, users discuss LMStudio features ranging from enabling function calling to overcoming model and software restrictions.

**RAM Bug Uncovered in LMStudio**: An acknowledged bug in LMStudio misreports system RAM, misleading users like `@pdx_`, who saw no change indicated in the software after a hardware upgrade to 64Gb.

**Cost and Compatibility Guide the Hardware Debate**: Discussions around hardware for LLM tasks involve detailed cost comparisons for high-end builds, potential GPU mixing for optimizing performance, and overclocking intricacies.

**Quantum Compression and AVX Instructions**: A new development in model compression, specifically 1.5 bit quantization, is expected to greatly improve efficiency, allowing large models to operate on reduced hardware. In the meantime, users are advised to utilize an AVX beta release for CPUs lacking AVX2 support.

**Humorous Take on AI Work Ethic and Errors**: `@wolfspyre` brought levity to the conversation with a comical inquiry if bots need to work and a playful depiction of bots stuck in a repetitive output loop.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **ChatGPT Remembers**: ChatGPT introduced a **new memory feature** to remember past conversations, with user controls for memory management being tested among Free and Plus users; details are outlined in [OpenAI’s blog post](https://openai.com/blog/memory-and-new-controls-for-chatgpt).
  
- **OpenAI Introduces Sora**: **Sora**, a model that generates short videos from text descriptions, was announced, targeting red teamers and creative professionals for initial feedback, as mentioned on [OpenAI's Sora introduction page](https://openai.com/sora).

- **Google’s AI Joins the Fray**: The AI community compared Google's GPT model, priced similarly to OpenAI's models, discussing Google's strategic positioning and the enhanced capabilities of *Gemini Advanced* - for more info [check Google's Gemini post](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/?utm_source=yt&utm_medium=social&utm_campaign=gemini24&utm_content=&utm_term=).

- **GPT-4's Learning Curve and System Strain**: Reports of service outages and issues with GPT-4 context retention prompted discussions on performance challenges, while users eagerly discussed the implications of the Sora model for creative fields, despite its current inaccessibility.

- **Prompt Engineering Deep Dive**: Users delved into strategies for engaging with GPT, optimizing token usage, and crafting prompts for structured outputs like yes/no answers, utilizing resources like the [behavioral adjustment tool](https://chat.openai.com/g/g-6qn4yGrBR-directive-gpt-llm-behavioral-adjustment-tool) for prompt refinement and service improvements.

- **Challenges in Image Rotation and GPT Interactions**: Frustrations were voiced regarding DALL-E 3's 50/50 success rate with image orientation and the disappearance of webp files, as well as the importance of balancing grammer precision with token economy for prompt optimization, reflected in [Discord conversations](https://discord.com/channels/974519864045756446/1037561178286739466).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Introducing QuIP# - State-of-the-art in Quantization**: A [research paper](https://arxiv.org/abs/2402.04396) details **QuIP#**, a method for post-training quantization of large language models that achieves impressive results with 2-3 bit precision, potentially outperforming existing methods in highly compressed regimes.

- **New Advances and Speculation in AI**: Discussions include **ZLUDA**, a tool to run CUDA on AMD GPUs — though reportedly abandoned — and anticipation around a mysterious new architecture with comparisons made to **DeepMind's RETRO**. Meanwhile, speculations humorously suggest future paper naming conventions, such as "optimal 0.5 bit quantization."

- **AI-Assisted Content Creation Blossoms**: OpenAI's announcement of [Sora](https://openai.com/sora), a text-to-video model, ignited excitement among users, marking a significant step in AI-generated video content. Sharing and dissecting breakthroughs, from model **routing analysis** to **QuIP#** and **MIQU**'s training, showcases the guild's dedication to technical exploration.

- **Evaluating and Hosting AI Models**: Practical recommendations were shared for AI model assessment, such as using the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), and hosting options like [Together](https://togetherplatform.com/) for Deepseek Coder. For vision language models, **Replicate** and [Alibaba](https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start) were recommended, despite some rate limit concerns.

- **Collective Cognition Project Hits a Snag**: The project faced downtime, linked to new modes in chat GPT that broke the website, suggesting that maintenance challenges have led to a period of inactivity.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Direct Principle Feedback Tackles the Avoidance Issue**: **EleutherAI** introduced a new method, **Direct Principle Feedback (DPF)**, outperforming traditional models and matching **GPT-4** for guiding chatbots to avoid unwanted topics, detailed in their recent paper, which can be accessed [here](https://arxiv.org/abs/2402.07896).

- **Language Model Harness Troubleshooting**: `@christianpala` provided a suggested fix for issues with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/620d6a152b2eb9216138495bfd6db4ea770dec26/lm_eval/models/openai_completions.py#L124) when adapted for local models and tokenizer. Users inquired about open-book and Chain of Thought (COT) prompts support and were advised on Python version compatibility for using the harness with older versions.

- **Exploration of Model Training and Pre-Training Techniques**: A member questioned pre-training encoder-decoder parts for seq2seq tasks like **machine translation**, initiating a discussion on its efficacy. Another flagged potential alignment issues between training data batches and checkpoints in Pythia-deduped for **2.8b** models, with another member committing to inspect this concern.

- **Safety, Security, and Inferring Capabilities in LLMs**: Researchers discussed the implications of secret collusion via steganography among AI agents and the memorization capabilities of LLMs, highlighting the risks as their abilities evolve. A new collaborative scrutiny is suggested to probe the replicability of **Pythia** findings post-training discrepancy in some models.

- **Interpretability Methods and Challenges Sparse**: Users expressed a need for updated overviews on AI interpretability, discussed interpretability in vision transformers and diffusion models, and sought approaches for evaluation techniques applied to propensity evaluations in models.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Mistral Outperforms on Sturdy Servers**: Users underlined that **Mistral's** performance is heavily dependent on server prowess and load conditions, highlighting that even smaller models like **GPT-4** can excel if the server is unfettered.
  
- **Intern Seeks Finetuning Wizardry**: An influx of requests surfaces around finetuning Mistral, with users sharing a plethora of materials such as [Jupyter notebooks](https://jupyter.org/), [Kaggle](https://www.kaggle.com/), and [Hugging Face's AutoTrain](https://huggingface.co/autotrain), while interns share their daunting tasks, including transforming infrastructures with Kubernetes.

- **Latency Lurks in Mistral API's Shadows**: Reports of high latency issues with Mistral API's 'completions' endpoint arise, with users being directed to consult Mistral support for remediation.

- **Mistral Mysteries Unveiled**: While **Mistral's** training data remains shrouded in secrecy, details emerge about **Mixtral Instruct**, a 6.48B parameter model with I32 and FP16 tensor type support hosted on [Hugging Face](https://huggingface.co/casperhansen/mixtral-instruct-awq), boasting over 8,430 recent downloads. A query about the distinctions among various **Mistral 8x7B fine-tunes** spirals into a discussion about dataset specificity for fine-tuning.

- **Fine-Tuning Frustrations and Successes**: Engineers explore fine-tuning **Mistral 8x7B** using Apple's MLX, with resources and scripts from GitHub repositories like [mlx-examples](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md) being circulated for potential guidance, while another [repository](https://github.com/mzbac/mlx-moe) signals ongoing development for better MLX compatibility.

- **NVIDIA's Novel Chatbot Chat with RTX**: NVIDIA debuts a customizable chatbot, **Chat with RTX**, powered by RTX 30 Series GPUs, eliciting comparisons with other chatbot solutions and proving to be a topic of fascination within the community.

- **European Internship Quest and PDF Pandemonium**: A French librarian scouts for internships alongside discussions on the parsimonious budgets for S2S models, while users battle the woes of PDF data extraction and laud the launch of a new [character AI website](https://www.wearefanchat.com).

- **GDPR, Chatbots, and Payment Pathways on La Plateforme**: Queries about GDPR compliance with Mistral's APIs lead to sharing of the [data processing agreement](https://mistral.ai/data-processing-agreement/). Meanwhile, members guide a new subscriber on setting up a ChatGPT-like bot with resources like [Mistral's Python client library](https://github.com/mistralai/client-python) and discussions on payment methods, including the absence of PayPal for Mistral, steer towards workarounds.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Efficiency in Image Generation Gets a Boost**: `@chad_in_the_house` has applied `lfq 2^17` to ImageNet, focusing on rapid training via the **Muse architecture**. Envisioning **new fine-tuning prospects** for `vqgans`, this stride could mark a leap for image generation processes.

- **Safety vs. Functionality Tradeoff in AI**: OpenAI's commitment to safety is viewed with concern as `.undeleted` speculates that extreme safety tuning might render models too expensive and impractical for certain applications. This conversation reflects an underlying tension between **AI safety and utility**.

- **The Quest for Quality Synthetic NSFW Datasets**: Echoing struggles within the AI community, `@progamergov` points out the challenges in procuring high-grade synthetic NSFW content for datasets, and criticizes Civitai's subpar outputs. This discussion highlights a niche but critical aspect of dataset development in AI.

- **Video-Linguistic Models Eye New Frontiers**: **RingAttention** has been identified as a promising approach for parsing extensive video and book datasets, as touched upon by `@spirit_from_germany` and `@max_voltage`. This technique is earmarked for its potential impact on **long-sequence training**.

- **Exploring OpenAI Sora's Text-to-Video Paradise**: `@qwerty_qwer` brings attention to OpenAI's [Sora](https://openai.com/sora), a transformative text-to-video model flaunting the capacity to conjure richly detailed scenes. Despite its awe-inspiring demo, the closed access nature raises some concerns within the community about its broader adoption and transparency.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Hugging News Unwrapped**: The Hugging Face community has been buzzing with diverse updates, including the launch of new APIs and model compatibilities, advancements in community contributions, and innovative application tools. Additionally, the upcoming reading group session is set to cover the paper "Mamba: Content-Based Reasoning for Foundations Models," which discusses computational inefficiencies in Transformers. ([89th edition of Hugging News](https://twitter.com/_philschmid/status/1755592500511997980), [45th edition of Community Highlights](https://huggingface.co/spaces/Tonic/prometheus), [Read “Mamba” paper](https://arxiv.org/abs/2312.00752))

- **Snap Detection Quest & Gemini Pro Insights**: Discord users in the general channel discussed various topics including the hunt for real-time finger snap detection in video/audio, difficulties with token issues in Hugging Face Spaces, a user's blog post gaining traction, debunking myths about stolen data in Mistral and other LLMs, and buzz around Google's Gemini 1.5's improved long-context understanding. No specific solutions or papers provided for snap detection. ([Google’s Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/?utm_source=yt&utm_medium=social&utm_campaign=gemini24&utm_content=&utm_term=))

- **Sheet-savvy and Modulated Melodies**: Learning new skills is highlighted in the today-im-learning channel, with discussions around the finesse of merging Google Sheets, learning achievements with DoReMi reproduction using FP8 3D parallelism, delving into diffusors and transformers, exploring face-swapping programs, and discussions around custom NER tagging in language models with references to specific datasets. ([few-nerd dataset](https://huggingface.co/datasets/DFKI-SLT/few-nerd), [conll2003 dataset](https://huggingface.co/datasets/conll2003))

- **MoE Discussion Strikes Chord**: In the cool-finds channel, members discussed the potential threats posed by vulnerabilities in Mixture of Experts models. Attention was also drawn to a project capable of parsing long text and video data over one million tokens, the merging of online and offline reinforcement learning algorithms, and SPIN, a method enabling Language Learning Models to mimic human reactions. ([Paper on MoE security issue](https://huggingface.co/papers/2402.05526), [DeepMind's largeworldmodel project](https://largeworldmodel.github.io/))

- **RAGs to Riches & Local LLMs**: The i-made-this channel showcased member creations and projects such as RAG-based applications with a plethora of text2image prompts, hosting of large language models for free via **LocalLlm** on Colab, the release of **tokviz** for visualizing model tokenization, and the introduction of UI interaction models PTA-Text and generative coding models Trinity and Neo. ([LocalLlm on GitHub](https://github.com/groloch/LocalLlm/), [tokviz documentation](https://github.com/Mr-DG-Wick/tokviz), [Trinity Space](https://huggingface.co/spaces/Tonic/trinity), [Neo Space](https://huggingface.co/spaces/Tonic/neo))

- **LangTest Dive and Seed Selection**: The reading-group channel hosted conversations on the application of the LangTest library for safe LLMs, arranging presentations on model merging, addressing questions about the Mamba paper, exploring the effects of random seed selection, and discussing works related to seeds and model performance. ([LangTest publication](https://www.sciencedirect.com/science/article/pii/S2665963824000071), [Mamba paper discussion](https://arxiv.org/abs/2312.00752))

- **Shaping Dreams and Cascading Conversation**: Diffusion-discussions saw members report success with image generation from text using Stable Cascade, inquire about deploying models on SageMaker, generate images using serverless APIs, and solve problems associated with vanishing gradients during model fine-tuning. ([Lykon/dreamshaper-8 discussions](https://huggingface.co/Lykon/dreamshaper-8/discussions))

- **Visionary Queries and PTA-Text Showcase**: The computer-vision channel experienced queries and discussion on topics such as gaussian splats, multimodal projects, improving image retrieval systems, transforming hairstyles using generative models, and unveiling a project focused on multimodal UI interactions, PTA-Text. ([PTA-Text Space](https://huggingface.co/spaces/AskUI/pta-text-v0.1), [Model checkpoint](https://huggingface.co/AskUI/pta-text-0.1))

- **Text and Voice Transformers Talk**: Conversations in the NLP channel covered XLM-RoBERTa language extraction, translations into algebraic representations, simulating voices and changing languages with transformers, introducing the PTA-Text project for UI interaction, and discussing its capabilities and current limitations. ([XTTS model space](https://huggingface.co/spaces/coqui/xtts))



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Slack to Receive Perplexity Updates**: Perplexity is rolling out a new feature called **Perplexity Push** that will allow topic **subscriptions** within Slack, aimed to streamline team communications and information sharing.

- **Perplexity AI Unveils New Models and Faces Reliability Issues**: Perplexity shared details regarding its `pplx-7b-online` and `pplx-70b-online` models for API integrations, while users reported intermittent failures and inconsistencies in API responses. Meanwhile, speculation around an unconfirmed `pplx-8x7b` model stirred curiosity, but no official information on availability or pricing was given. The **Gemini 1.5 AI** model by Google was announced, noting its potential one million-token context window.

- **Resourceful Community Engages and Shares Perplexity Content**: Users engaged with Perplexity AI's features including an alternative **Alt-D Feed** for community collaboration and discussed bookmarking limitations. A GitHub repo integration with structured data and logic patterns was teased by a user, while another shared their success story using Perplexity AI for a DIY hair tutorial.

- **API Pain Points Need Addressing**: Various users expressed frustration over the **Perplexity AI** models delivering unreliable API results, calling attention to inconsistencies and hallucinated content in responses. A user provided a [guide to integrate Perplexity AI with LangChain](https://mochan.org/posts/perplexity-ai-langchain/), looking to help others overcome issues with model substitutions in applications.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **No-Code Revolution for AI Workflows**: A **webinar on building no-code RAG** (Retriever-And-Generator) is announced for Friday at 9 am PT. The session, led by Henry Heng from FlowiseAI, is set to explore creating LLM-powered workflows using LlamaIndex.TS and Flowise integration, targeting those seeking to bypass coding steps. [Register for the informative webinar](https://lu.ma/ubm3jg3k).

- **DanswerAI Empowered by LlamaIndex**: The integration of **DanswerAI** with LlamaIndex technology promises to enhance workplace tool efficiency. LlamaIndex highlights this collaboration, along with a series of other educational content, including scientific research workflow tutorials and guidelines for building custom agents with LLM. The featured notebook and video tutorial are bridging the gap for AI engineers.

- **Arize-Phoenix Enhancements Incoming**: Metadata tagging in tracing user queries is undergoing improvements and expected in the coming week, as confirmed in an update related to Arize-Phoenix. An issue with SimpleDirectoryReader misinterpreting DOCX files has been resolved with the newest `llama-index-core` version, and a [community Discord server](https://discord.gg/55mzvBnS) has been set up for integration support.

- **Real-Time RAG Optimization Conversations**: LlamaIndex users discuss real-time optimization of RAG pipelines through user feedback, suggesting the use of reranking based on scores. Providing actual code snippets, the community offers insights for separating retrieval and synthesis steps for more effective real-time evaluation.

- **Integration Troubles & Solutions Shared**: Users share solutions to common integration issues such as excluding metadata from custom QA templates, with suggestions to set exclusion keys before data ingestion. Additionally, resources like [Excalidraw for collaborative whiteboarding](https://excalidraw.com/) and [Notion for workspace organization](https://pretty-sodium-5e0.notion.site/ce81b247649a44e4b6b35dfb24af28a6?v=53b3c2ced7bb4c9996b81b83c9f01139) are mentioned, along with a diverse range of LlamaIndex documentation and GitHub examples provided for various use cases.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Journaling App Integrates Memory with LangChain**: LangChain introduces a new journaling app featuring memory capabilities, currently in the early stages with feedback sought. Access a demonstration via [Loom video](https://www.loom.com/share/63a9696036c74765a9f9ecae06336aa5) and the app itself at [Journal by LangChain](https://journal.langchain.com/).

- **LangSmith's Leap Forward**: The general availability of LangSmith is announced along with a $25M Series A led by Sequoia Capital, a new homepage, brand, and career opportunities. Insights available in their [blog post](https://blog.langchain.dev/langsmith-ga/), with features on [Forbes](https://www.forbes.com/sites/alexkonrad/2024/02/15/open-source-ai-startup-langchain-launches-langsmith/?sh=26e00cb24f00) and [Product Hunt](https://www.producthunt.com/posts/langsmith-general-availability).

- **Pinecone and Langchain Dependency Challenges**: Peer dependency conflicts arise between Pinecone v2 and LangChain, with solutions including the use of `npm install --legacy-peer-deps` or version bumps discussed. Optimization tips for RAG pipelines based on user feedback were exchanged, including manual inspection and parameter adjustment.

- **LangServe Development Discussions Unfold**: Topics ranged from overcoming Image Base64 encoding issues in LangChain playground to "connection refused" errors within Kubernetes clusters. Deployment questions concerning Langchain/LangServe apps prompted mentions of using Vercel and Replit for web accessibility.

- **AI Innovation and Knowledge Sharing**: A Reverse Job Board at [AI Devs Work](https://www.aidevs.work/) provides a platform for AI talent recruitment. A guide on creating a goal-setting assistant and a [tutorial](https://dewykb.github.io/blog/qa-cli-with-langchain) on building a LangChain.js-powered question-answering CLI with Dewy showcase application building chops. Additionally, "[Multi Document RAG using LangChain codes explained](https://youtu.be/cBpdiQ3gljM?si=lAhY7F0UXZfUZP57)" video tutorial is highlighted, offering education on implementing Multi-Document RAG Agents.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Keras As a Universal Model Adaptor**: Engineers discussed porting models to **Keras** for broader hardware support, as it now functions as an independent abstraction layer over frameworks like Torch, TF, and Jax.
- **Checkpoint Fiasco Finds a Fix**: A link to a problematic pull request on the **HuggingFace repository** was shared, which was identified to be causing checkpoint saving errors. This issue was also connected to recent outages experienced by HF.
- **Efficient LLM Hosting Solutions Debated**: Cost-effective hosting for large language models was a hot topic, with **Together AI**, OpenRouter, and services like `basten` being suggested. Additionally, **NVIDIA's RTX-based** demo app, Chat With RTX, was brought up as a way to run personalized GPT models on local RTX hardware.
- **Serious Schema Strategies**: JSON schema for pairing user and assistant messages was recommended for better dataset structuring, while there was a push for flexibility in role naming within the message schema to avoid influencing model behavior.
- **Real-Time LoRA Adapter Flexibility**: The feasibility of adding **LoRA adapters to a base model in real-time** was confirmed to be possible with the HF framework, presenting a dynamic way to manage PEFT models.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **Long Live LLMs with Math Skills**: The fine-tuning of a 7B parameter model on business data could deteriorate its performance on mathematical questions, as noted by `@mertbozkir`. For math-intensive tasks, engagement with domain-specific methods or models like **internlm, metamath, arithmo** might be necessary.

- **GPU Market Volatility Hits Techies**: Recent conversations have highlighted the frustration with the fluctuating prices and availability of GPUs like the 3090s, with `@joseph_en` and `@andreaskoepf` sharing their experiences of cost spikes and referring to GPUs as "GPU gold."

- **CUDA Compatibility Quest**: Multiple users, including `@_tvi_`, `@shikhar_7985`, and `@btdubbins`, discussed the struggles of maintaining different CUDA versions for compatibility with other systems like PyTorch and FAISS. `@marksaroufim` recommended [Conda for managing CUDA versions](https://pytorch.org/) with PyTorch.

- **Associativity in Algorithms Challenged**: There's skepticism regarding the practicality of function representation, associativity, and classes from `@andreaskoepf`, `@_tvi_`, and others, with `@euclaise` sparking the conversation about using function composition similar to prefix-sum operations.

- **Search for Fun and Education in CUDA**: Users, including `@euclaise` and `@marksaroufim`, are discussing CUDA educational resources with an emphasis on enjoyment. Suggestions like [The Book of Shaders](https://thebookofshaders.com/) were mentioned, but no particular CUDA book was singled out as being notably fun.

- **Matrix Magic in Memory**: Discussions unfolded around the performance impact of keeping vectors in sequential memory for dot products, optimal index orders in loops, and the application of `atomicAdd` operations in shared memory, without definite consensus on best practices.

- **Lecture Legwork**: Queries about the organization of CUDA-related YouTube content, such as **Lecture 5**, spurred users to suggest solutions like a comprehensive playlist or adding videos directly to [Cuda's official YouTube channel](https://youtu.be/eUuGdh3nBGo?si=XnUPc-oaAdy4IQLd).

- **The TensorFlow Conundrum and PyTorch vs. JAX**: Rumors of TensorFlow's potential discontinuation were brought up briefly by `@spacyphus`, while `@marcom79` initiated a comparison between JAX and the upcoming PyTorch 2.0, with no detailed discussion ensuing on these topics.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Gemini Pro 1.5: Bigger, Longer, Uncut Context**: **Gemini Pro 1.5** has been a hot topic, with `@wenquai` highlighting its impressive 1 million token context window, while `@thebaghdaddy` brought attention to Jeff Dean's post claiming an even more remarkable ten million token context window, including the ability to process extensive multimodal content. This information surfaced along with discussions surrounding the effectiveness of such large context windows, where skepticism was noted regarding models' performance with token windows past 50-60k. Jeff Dean's [Twitter post](https://x.com/jeffdean/status/1758146022726041615?s=46) announcing Gemini Pro 1.5's developer preview captured the attention for its capabilities and forthcoming wider release.

- **Surya OCR Trumps Tesseract**: Converting **35k PDFs** into data has delivered a significant financial punch for `@res6969` due to high processing costs using a vision transformer. `@robhaisfield` chimed in with [**Surya OCR**](https://github.com/VikParuchuri/surya), a new OCR tool reported to outdo Tesseract across **93 languages**, potentially offering a cost-effective solution.

- **Engineering Minds, Grab Your Slice!**: The AI community in Singapore is buzzing with a meet-up opportunity posted by `@ivanleomk`, promising a project hacking session with **free pizza** at **Funan Mall**, organized by Gabriel Chua, Jon Jon, & tengfone. For those looking to mingle with minds alike, there's one slot up for grabs with [registration details available online](https://lu.ma/ai-weds).

- **Rumors and Releases from OpenAI**: GPT-5 rumors and the humor they're generating among enthusiasts was noted by `@res6969`. On a more concrete note, OpenAI is showcasing innovation, testing ChatGPT's memory capabilities, as described in their [blog post](https://openai.com/blog/memory-and-new-controls-for-chatgpt). They also debuted [Sora](https://openai.com/sora), a text-to-video AI model, now undergoing red team testing to identify potential risks. 

- **GPT-4 Enigma**: A single message with a cryptic "yeah" from `robotums` under the #gpt4 channel seems to reflect the terse mystery that often surrounds emerging technologies.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Fine-Tuning Follies**: After **fine-tuning** a 7B parameter language model on business data, users noted a likely **degradation in math performance**, suggesting the intensity and duration of fine-tuning determine the impact.

- **Optimism Versus Pessimism in ML Stability**: The conversation highlighted a dichotomy in reinforcement learning where **optimism** is essential for exploration, while applying **pessimism during inference** could lead to more stable sequential decision-making in machine learning systems.

- **Business Instruction Extraction Quest**: A user sought guidance on extracting **business-related instructions** from the teknium/OpenHermes-2.5 Instruction dataset, although no specific methodologies or resources were provided.

- **Trouble in Discord Town**: There were concerns raised regarding a user’s **potential technical issues with Discord**, with suggestions favoring **direct messaging** as a solution.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **LLaVA Integration Challenges Go Unanswered**: `@CodeMan` sought advice on configuring **LLaVA** for use with an **SGLang server and SGLang worker**, departing from the typical model worker approach. The query remained unanswered, indicating a gap in community knowledge or engagement on this topic.

- **Business Data Quest in OpenHermes Dataset**: `@sabu7003` is searching for techniques to extract **business-related instructions** from the [teknium/OpenHermes-2.5 Instruction dataset](https://github.com/teknium/OpenHermes-2.5), highlighting a need for targeted data isolation methods in this dataset.

- **Fine-Tuning Finesse for Business Data**: `@sabu7003` also raised a question about the effectiveness of a **7B parameter LLM** in mathematics problems after being fine-tuned solely on business information, a query that went without community input or exploration.

- **Random Seed Learnability Sparks Debate**: `@stereoplegic` and `@aspott` sparked a discussion on whether **random seeds** could be learnable parameters within AI models, with `@aspott` noting the impossibility of obtaining a gradient from a seed and suggesting learning an **initialization function** as an alternative pathway.



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

**Weekly Sync-Up Time**: The **weekly meeting** was initiated with an announcement tagging `@._z`.

**Hackathon Hosting Huddle**: An invitation was extended to co-host an **AI developers hackathon** by `@caramelchameleon`, considering the proximity to the Game Developers Conference and inviting both online and onsite participation.

**Hackathon Experience on the Table**: `@yikesawjeez` indicated interest in the hackathon opportunity, drawing from their background in organizing such events in the Bay Area.

**Investor Matchmaking Event**: `@atalovesyou` publicized a chance for startup founders to engage with over 30 venture capital firms at an **investor matchmaking session**; additional slots are available at [Founders x VC Event](https://lu.ma/c4klulz8) for interested individuals.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Less Compute, Same Power with Gemini 1.5**: [Google announced](https://developers.googleblog.com/2024/02/gemini-15-available-for-private-preview-in-google-ai-studio.html?m=1) its *Gemini 1.5 Pro*, maintaining the performance of *Gemini 1.0 Ultra* but with reduced compute needs, featuring a context window capable of handling 1 million tokens.

- **Prompt Engineering vs. Token Capacity**: The discussion highlighted the trade-off between prompt engineering and the direct input of relevant data, spurred by the increased token handling capacity of models like *Gemini 1.5*. As models improve, the skill of prompt engineering could become obsolete if larger contexts can be managed more economically.

- **Token Stretching Not Yet Standard**: Though Google has tested models with up to 10 million tokens, the decision to release *Gemini 1.5* with a 1 million token context window suggests external constraints such as cost. This could imply that prompt engineering will retain its relevance in efficiently interacting with models in the near future.



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1206869146646609920) (1263 messages🔥🔥🔥): 

- **Contextual Inquiry for DnD Assistant**: `@zackman634` is looking to create a Dungeons and Dragons (DnD) Dungeon Master (DM) assistant that can cross-reference rules from various books, find stat blocks, and access lore, while also being able to generate new content on demand for world-building. They received suggestions to try models like H20 GPT for document question answering and were advised about the complexities regarding models understanding tables and cross-references.
- **Concerns over PSU Capacity**: `@kalomaze` is worried about running a 3090 and 3060 GPU on a 650w PSU, having power-limited the GPUs to 250w and 150w respectively. Fellow users like `@felixsanz` and `@alphaatlas1` warned about the risks, and the importance of not daisy-chaining the power cables was discussed.
- **Google's Gemini 1.5 and OpenAI's Sora Make Waves**: Users discuss Google's recent announcement of Gemini 1.5 with up to 1M token context and OpenAI's SORA which can generate minute-long videos with detailed simulations. `@nextdimension` highlighted SORA's capabilities and how it may soon be utilized in serious applications like films, and `@itsme9316` expressed cautious optimism towards such leaps in technology.
- **LWM Multimodal Woes**: Multiple users including `@itsme9316` and `@mrdragonfox` share their struggles with trying to get Large World Model (LWM) multimodal functionality to work. Issues were posted about reaching the limits of computational resources, and users find commonality in their inability to operate the model due to technical complexities.
- **GPT for Coding Assistance**: Discussion around the utility of using LLMs like GPT and Copilot for coding support, with mixed opinions on efficacy. While `@mr_pebble` appreciates the help in initial code drafting and documentation, `@nextdimension` and others note that LLMs tend to miss complex edge cases, indicating that these tools complement rather than replace the need for human insight in software development.

**Links mentioned**:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and d...
- [Lumiere - Google Research](https://lumiere-video.github.io/?): Space-Time Text-to-Video diffusion model by Google Research.
- [Artefact2/BagelMIsteryTour-v2-8x7B-GGUF · Hugging Face](https://huggingface.co/Artefact2/BagelMIsteryTour-v2-8x7B-GGUF): no description found
- [GameEditor](https://tamats.com/projects/litegraph/): GameEditor for simple games
- [Sora: Creating video from text](https://openai.com/sora): no description found
- [TheBlock (Jom Tobbins)](https://huggingface.co/TheBlock): no description found
- [Birds over river on Vimeo](https://player.vimeo.com/video/913130937?h=469b1c8a45): no description found
- [Our next-generation model: Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/?utm_source=yt&utm_medium=social&utm_campaign=gemini24&utm_content=&utm_term=): Gemini 1.5 delivers dramatically enhanced performance, with a breakthrough in long\u002Dcontext understanding across modalities.
- [non tpu inference · Issue #4 · LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM/issues/4): we need some samples that can run actually inference on vision / image samples on gpu (lwm) ➜ LWM git:(main) ./scripts/run_sample_image.sh WARNING: Logging before InitGoogle() is written to STDERR ...
- [Bugs Bunny Yawn GIF - Bugs Bunny Yawn Dont Care - Discover &amp; Share GIFs](https://tenor.com/view/bugs-bunny-yawn-dont-care-bored-gif-14699324): Click to view the GIF
- [Killed by Google](https://killedbygoogle.com/): Killed by Google is the open source list of dead Google products, services, and devices. It serves as a tribute and memorial of beloved services and products killed by Google.
- [An algebraic theory to discriminate qualia in the brain](https://arxiv.org/abs/2306.00239): The mind-brain problem is to bridge relations between in higher-level mental events and in lower-level neural events. To address this, some mathematical models have been proposed to explain how the br...
- [GitHub - kneasle/sapling: A highly experimental vi-inspired editor where you edit code, not text.](https://github.com/kneasle/sapling): A highly experimental vi-inspired editor where you edit code, not text. - kneasle/sapling
- [
		A grammar of Kalamang
							| Language Science Press
			](https://langsci-press.org/catalog/book/344): no description found
- [January | 2024 | Ars Technica](https://arstechnica.com/tech-policy/2024/01/since-elon-musks-twitter-purchase-firm-reportedly-lost-72-of-its-value/>): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1206942630399705098) (301 messages🔥🔥): 

- **Fine-Tuning Model Preferences**: `@dreamgen` and `@neriss` discussed appropriate settings for role-play/story-writing with different models. `@neriss` recommended a temperature of 0.7 for Yi models, while suggesting that base Mixtral Instruct may be a superior option if the hardware supports it.

- **Model Training and Temperature Insights**: `@neriss` explained that higher temperatures increase the diversity of model outputs at the cost of coherence. For more creative outputs, a higher temperature is advisable, with Yi models known to run well at lower temperatures, around 0.6.

- **Model Benchmarks and Issues**: `@weirdconstructor` ran the Agnes Test for ERP models, noting that higher temperatures, like the one used by the model `@c.gato` trained, might not be ideal for preventing repetitive dialogues. It was suggested that models with more RP data might require lower temperatures to minimize loops.

- **Data Cleaning and Analysis Guidance**: `@mrdragonfox` provided advice and resources for cleaning datasets, recommending the use of pandas for tabular data and regex for general cleaning. They also shared a gist link for assistance.

- **M2 Max Performance on Transformers Models**: `@heyitsyorkie` shared their experience using miquliz v2.0 120b q4km and `@sssynk` discussed the capabilities of their M2 Max for running 70B models effectively, with prompt processing times being a consideration. `@timothyallan` mentioned a terminal hack to increase RAM usage, allowing larger models to run on an M2 Max.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1112409939336503338/1175838485568036936): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Kquant03/NeuralTrix-7B-dpo-laser · Hugging Face](https://huggingface.co/Kquant03/NeuralTrix-7B-dpo-laser): no description found
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1112409939336503338/1175939201330585631): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Kquant03/NeuralTrix-7B-dpo-laser-GGUF at main](https://huggingface.co/Kquant03/NeuralTrix-7B-dpo-laser-GGUF/tree/main): no description found
- [gist:f786564868357cde5894ef6e2c6f64cf](https://gist.github.com/darkacorn/f786564868357cde5894ef6e2c6f64cf): GitHub Gist: instantly share code, notes, and snippets.
- [Ayumi Benchmark ERPv4 Chat Logs](https://ayumi.m8geil.de/erp4_chatlogs/#!/model/DL_20240215_7B-Q6_K_Thespis_V0_5_SFTTest_2Epoch.gguf>): no description found
- [JSONEditor](https://ayumi.m8geil.de/erp4_chatlogs/json_editor.html?url=DL_20240202_13B-Q5_K_M_Thespis_DPOTest2_pub.json>)): no description found
- [Adjust VRAM/RAM split on Apple Silicon · ggerganov/llama.cpp · Discussion #2182](https://github.com/ggerganov/llama.cpp/discussions/2182#discussioncomment-7698315): // this tool allows you to change the VRAM/RAM split on Unified Memory on Apple Silicon to whatever you want, allowing for more VRAM for inference // c++ -std=c++17 -framework CoreFoundation -o vra...

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1206887025983098910) (58 messages🔥🔥): 

- **Notebook Upload Anticipation**: `@starsupernova` mentioned they would upload a notebook to Unsloth's Discord server and ping `@1025039473932775485` once done, but no specific bugs were guaranteed to be addressed.
- **Mixtral vs. Mistral Loss Conundrum**: `@dreamgen` and `@nruaif` discussed observing higher training and eval losses with Mixtral compared to Mistral, initially thinking it's a bug. Both are experimenting with different setups to address this issue.
- **Fine-tuning Intricacies and Struggles**: From `@kquant` laser-tuning yielding loss of 0.08 points to `@haroon30` inquiring about VRAM and RAM requirements for finetuning deepseek models, community members are sharing their fine-tuning challenges and seeking advice.
- **Building a Better LLM for Portuguese**: `@luishenriquemartins` seeks to leverage a large dataset in Portuguese to train an LLM for journalistic applications. The discussion ranged from considering the cost of training with the help of a research institution to the possibility of fine-tuning existing models like mistral or llama.
- **AutoFineTune Script Showcase**: `@jiha` shared a link to a tweet by `@yoheinakajima` introducing a script named "AutoFineTune", which can generate synthetic message pairs and fine-tune a small model using Together Compute. The GitHub/Replit is provided in the thread linked in the tweet.

**Links mentioned**:

[Tweet from Yohei (@yoheinakajima)](https://x.com/yoheinakajima/status/1757663960772612408?s=20): Just made a lil&#39; script to easily fine-tune a small model with synthetically generated data...  ...calling it &#34;AutoFineTune&#34;! (~110 lines of code)  Generates 100+ synthetic message pairs w...

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1207468673112080404) (3 messages): 

- **Merging Models with Different Context Sizes Results in Error**: User `@222gate` encountered a **RuntimeError** when trying to merge two **MistralCasualML** models with different context sizes. The reported error message was related to a tensor size mismatch: `Tensor size mismatch for model.layers.22.self_attn.o_proj.weight, sizes: [torch.Size([2560, 2560]), torch.Size([4096, 4096])]`.
- **Seeking Solutions for Tensor Mismatch**: `@222gate` asked the community if anyone knew a workaround for the tensor size mismatch issue they faced while merging models.
- **Positive Feedback but Undisclosed Solution**: `@222gate` expressed excitement with a message saying "this is awesome" but did not provide details on whether the issue was resolved or the nature of what they found awesome.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1207041428220280852) (6 messages): 

- **Integrating Python in JavaScript with JSPyBridge**: `@spottyluck` described a successful experience in bridging JavaScript with Python using [JSPyBridge](https://github.com/extremeheat/JSPyBridge), including example code snippets that show how to create a new class in JavaScript that interacts with Python code asynchronously.
- **Interfacing with compression in Node.js**: They detailed creating an async function in Node.js **`compressPrompt`** that uses Python classes via the bridge to compress prompts for efficient processing.
- **Alterations to BigDL**: `@spottyluck` modified BigDL's **LLM transformer** to load **q8_0 gguf files** and disabled an optimization to prevent prompt mangling, crucial for running LLMLingua. The provided code snippet shows the necessary adjustments and considerations for running the transformer, especially on Windows.
- **Device Tensor Handling in Node.js**: Additional guidance was provided on dealing with errors related to tensors not being on the expected device, highlighting the use of **`model.to()`** in the Node.js context when interfacing with Python.
- **Compressing prompts in the request handling process**: `@spottyluck` finalized with an explanation of integrating prompt compression through a conditional in the Node.js `router.post` method, allowing the Python-based compression to be leveraged as if it were a native JavaScript class.

**Links mentioned**:

[GitHub - extremeheat/JSPyBridge: 🌉. Bridge to interoperate Node.js and Python](https://github.com/extremeheat/JSPyBridge): 🌉. Bridge to interoperate Node.js and Python . Contribute to extremeheat/JSPyBridge development by creating an account on GitHub.

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1206868042865188904) (410 messages🔥🔥🔥): 

- **Intrigue Over "Chat with RTX"**: NVIDIA's "Chat with RTX" has sparked interest with its built-in retrieval-augmented generation (RAG) feature on Nvidia 30+ series, which is different from LMStudio as it can perform RAG tasks but only supports Mistral 7b and Llama 13b models currently (`@heyitsyorkie`).

- **Curiosity About Gemini 1.5's Massive Context Window**: There's a buzz about Google's Gemini 1.5 model claiming to support a 1 million token context window, though it's invite-only and not publicly available (`@hypocritipus` and `@rugg0064`).

- **Exploring Sora's Capabilities**: OpenAI's new Sora model for generating videos from text prompts up to a minute long has caught attention as it becomes available to red teamers and creative professionals; however, there's concern about the impact on evidence credibility (`@joelthebuilder` and `@rugg0064`).

- **Optimizing LMStudio Experience**: Users inquire about various features within LMStudio, such as enabling function calling (`@vbwyrde`), adjusting thread usage (`@rekt.gg`), and configuring advanced inference parameters (`@jackiezhou0601`).

- **Lament Over Model and Software Restrictions**: Dialogue touched on the limitations imposed by different AI systems, where models like GPT-4 remain proprietary and can't run locally, which disappoints users like `@stoic_king` and `@securityguruguy`. There's also a discussion about the impracticality of large context sizes and potential privacy concerns (`@pwrreset` and `@hypocritipus`).

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1191233385058816160): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1190413331731845130): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [BRIA RMBG 1.4 - a Hugging Face Space by briaai](https://huggingface.co/spaces/briaai/BRIA-RMBG-1.4): no description found
- [Boximator: Generating Rich and Controllable Motions for Video Synthesis](https://boximator.github.io/): no description found
- [Stable Cascade - a Hugging Face Space by multimodalart](https://huggingface.co/spaces/multimodalart/stable-cascade): no description found
- [NVIDIA Chat With RTX](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/): Your Personalized AI Chatbot.
- [TheBloke/OpenHermes-2.5-Mistral-7B-GGUF · Hugging Face](https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF): no description found
- [Artificial intelligence, real emotion. People are seeking a romantic connection with the perfect bot](https://apnews.com/article/ai-girlfriend-boyfriend-replika-paradot-113df1b9ed069ed56162793b50f3a9fa): On online messaging forums, users say they’ve developed emotional attachments to bots and are using them to play out sexual fantasies or more.
- [Hugging Face – The AI community building the future.](https://huggingface.co/): no description found
- [System prompt - Pastebin.com](https://pastebin.com/vnxJ7kQk): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/17vonjo/your_settings_are_probably_hurting_your_model_why/): no description found
- [Hello It Have You Tried GIF - Hello It Have You Tried Turning It Off And On Again - Discover &amp; Share GIFs](https://tenor.com/view/hello-it-have-you-tried-turning-it-off-and-on-again-telephone-on-call-gif-15495555): Click to view the GIF
- [Sudowrite](https://www.sudowrite.com/): Bust writer's block and be more creative with our magical writing AI.
- [The unofficial LMStudio FAQ!](https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio): Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...
- [GitHub - KoljaB/LocalAIVoiceChat: Local AI talk with a custom voice based on Zephyr 7B model. Uses RealtimeSTT with faster_whisper for transcription and RealtimeTTS with Coqui XTTS for synthesis.](https://github.com/KoljaB/LocalAIVoiceChat): Local AI talk with a custom voice based on Zephyr 7B model. Uses RealtimeSTT with faster_whisper for transcription and RealtimeTTS with Coqui XTTS for synthesis. - KoljaB/LocalAIVoiceChat
- [What is RAG? - Retrieval-Augmented Generation Explained - AWS](https://aws.amazon.com/what-is/retrieval-augmented-generation/): no description found
- [GitHub - microsoft/NeuralSpeech](https://github.com/microsoft/NeuralSpeech): Contribute to microsoft/NeuralSpeech development by creating an account on GitHub.
- [Sora: Creating video from text](https://openai.com/sora): no description found
- [Fire Ants live up to their name by invading PC and eating thermal paste, raising system temps](https://www.techspot.com/news/101895-fire-ants-live-up-their-name-invading-pc.html): The problem of PC-invading ants was highlighted by Redditor Thejus_Parol (via PCGamesN). The user reports that the max GPU temps on their PC started rising slightly, prompting...
- [GitHub - ggerganov/whisper.cpp: Port of OpenAI&#39;s Whisper model in C/C++](https://github.com/ggerganov/whisper.cpp): Port of OpenAI&#39;s Whisper model in C/C++. Contribute to ggerganov/whisper.cpp development by creating an account on GitHub.
- [Our next-generation model: Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/?utm_source=yt&utm_medium=social&utm_campaign=gemini24&utm_content=&utm_term=): Gemini 1.5 delivers dramatically enhanced performance, with a breakthrough in long\u002Dcontext understanding across modalities.
- [


    Squibler Pricing


](https://www.squibler.io/pricing): no description found
- [Release v0.1.25 · ollama/ollama](https://github.com/ollama/ollama/releases/tag/v0.1.25): Windows Preview Ollama is now available on Windows in preview. Download it here. Ollama on Windows makes it possible to pull, run and create large language models in a new native Windows experience...
- [How to control Home Assistant with a local LLM instead of ChatGPT](https://theawesomegarage.com/blog/configure-a-local-llm-to-control-home-assistant-instead-of-chatgpt): While it&#039;s easy to make a local assist pipeline, and get away from Google Assistant or Alexa, it&#039;s not really that easy to go all local and still...
- [The unofficial LMStudio FAQ!](https://rentry.org/LMSTudioFAQ#model-failed-to-load): Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...
- [Update llama.cpp support by vosen · Pull Request #102 · vosen/ZLUDA](https://github.com/vosen/ZLUDA/pull/102): Add sign extension support to prmt, allow set.&lt;op&gt;.f16x2.f16x2, add more BLAS mappings
- [Apple silicon - Wikipedia](https://en.wikipedia.org/wiki/Apple_silicon#M_series): no description found

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1206891496368771072) (125 messages🔥🔥): 

- **Yi-VL Models Await Update**: `@heyitsyorkie` mentioned that **Yi-VL models** are currently unsupported due to the need for an update to the `llama.cpp` version used by LM Studio. `@jedd1` inquired about a list or heuristic for unsupported GGUFs, to which `@heyitsyorkie` replied that issues generally arise with new small models requiring updates to `llama.cpp`.

- **PDF & Book Upload for Assistants Under Discussion**: `@edu0835` inquired about the possibility of creating an assistant in **LM Studio** that allows for *PDF* or book uploads, like the Huggingface chat assistant or GPTs with entire books in PDF form, specifically referencing medical texts for disease responses.

- **Comparing Coding Models**: Discussions about the comparative performance and peculiarities of **Deepseek Coder Ins 33b** and **Codellama Instruct 70b** were had by users `@kujila` and `@heyitsyorkie`, with the former stating a preference for Deepseek due to its serious approach compared to Codellama's "whimsical" responses.

- **Exploration of Image Generation**: `@joelthebuilder` and `@heyitsyorkie` shared their experiences and recommended tools for diving into image generation, mentioning **Stable Cascade**, **automatic1111**, and **comfyui** as notable options to check out.

- **Quest for Reliable RAG Solutions for Windows**: `@666siegfried666` sought information about Retrieval-Augmented Generation (RAG) options for Windows, with `@wildcat_aurora` and `@kujila` suggesting to look at **H2oGPT**, **lollms**, or **AGiXT** which can be used with **LM Studio local server** for RAG capabilities.

- **File Downloading and Model Conversion Challenges**: There was discussion about difficulties with downloading and converting large model files, such as issues with the model downloader in LM Studio and the need for manual file merging when certain quantizations are missing, as shared by `@666siegfried666`, `@fabguy`, and `@n0w1sm`.

**Links mentioned**:

- [All Large Language Models](https://llm.extractum.io/list/): A Curated List of the Large and Small Language Models (Open-Source LLMs and SLMs). All Large Language Models with Dynamic Sorting and Filtering.
- [Napster Shrug GIF - Napster Shrug Blame Nap - Discover &amp; Share GIFs](https://tenor.com/view/napster-shrug-blame-nap-gif-24272952): Click to view the GIF
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1207374637588152393/1207374637588152393): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Chat with RTX Now Free to Download | NVIDIA Blog](https://blogs.nvidia.com/blog/chat-with-rtx-available-now/): New tech demo gives anyone with an NVIDIA RTX GPU the power of a personalized GPT chatbot, running locally on their Windows PC.
- [Bear Blowakiss GIF - Bear BlowAKiss Love - Discover &amp; Share GIFs](https://tenor.com/view/bear-blowakiss-love-hearts-kissing-gif-14185235873390222605): Click to view the GIF
- [Nexesenex/Senku-70b-iMat.GGUF at main](https://huggingface.co/Nexesenex/Senku-70b-iMat.GGUF/tree/main): no description found
- [wolfram/miquliz-120b-v2.0-GGUF · Hugging Face](https://huggingface.co/wolfram/miquliz-120b-v2.0-GGUF): no description found
- [When Its Done Its Done Finish GIF - When Its Done Its Done Finish Final - Discover &amp; Share GIFs](https://tenor.com/view/when-its-done-its-done-finish-final-conclude-ending-gif-13222232): Click to view the GIF
- [Star Wars Obi Wan Kenobi GIF - Star Wars Obi Wan Kenobi Its Good Enough For Me - Discover &amp; Share GIFs](https://tenor.com/view/star-wars-obi-wan-kenobi-its-good-enough-for-me-good-enough-for-me-ewan-mcgregor-gif-26028661): Click to view the GIF
- [A simple guide to local LLM fine-tuning on a Mac with MLX &#8211; Andy Peatling](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/): no description found
- [The new Yi-VL-6B and 34B multimodals ( inferenced on llama.cpp, results here ) · ggerganov/llama.cpp · Discussion #5092](https://github.com/ggerganov/llama.cpp/discussions/5092): Well, their benchmarks claim they are almost at GPT4V level, beating everything else by a mile. They also claim that CovVLM is one of the worst (and it&#39;s actually the best next to GPT4, by far) On...

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1207652032895520788) (3 messages): 

- **RAM Upgrade Confusion**: `@pdx_` shared that after upgrading their system to **64Gb of RAM**, LM Studio still indicated they had 16Gb. `@yagilb` acknowledged this as a **known bug** and reassured that it would be fixed in the next update, clarifying that the issue is purely informational and loading modules should work if the VRAM is sufficient.
- **Gratitude for Quick Response**: `@pdx_` expressed gratitude with a simple "ok thanks 🙂" following the prompt support received regarding the RAM misreport issue.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1206869868037799971) (200 messages🔥🔥): 

- **AMD vs Nvidia for LLM Performance**: `@goldensun3ds` expressed curiosity about combining an RTX 4060 Ti and an RX 7600 XT for potentially better performance. `@666siegfried666` remarked on ROCm's youth and potential for future improvements, indicating some skepticism about mixing GPUs.
- **High-End Build Cost Comparisons**: `@nink1` broke down the costs of a high-end build comparable to a Mac studio, while others like `@heyitsyorkie` and `@jedd1` debated the pros and cons of different configurations, including gaming and Linux compatibility.
- **Threadripper & RAM Overclocking**: Multiple users discussed the utilization of AMD's Threadripper for various workloads. `@666siegfried666` delved into the details of RAM overclocking, suggesting careful manual tuning to avoid system instability.
- **Exploring VRAM Upgrades for AI Workloads**: Users like `@rugg0064` and `@goldensun3ds` discussed the possibility and practicality of modding GPUs such as the RTX 2080 Ti to increase VRAM to 22GB, suggesting potential for AI applications but questioning the economic viability compared to buying new GPUs with more VRAM.
- **Laptop GPU Selection Issues in LM Studio**: `@radion8267` sought assistance for configuring LM Studio to use a dedicated GPU rather than the default APU, noting performance issues. `@heyitsyorkie` mentioned a known detection bug and suggested potential alternatives.

**Links mentioned**:

- [Doja Cat GIF - Doja Cat Star - Discover &amp; Share GIFs](https://tenor.com/view/doja-cat-star-wars-gif-25078126): Click to view the GIF
- [Brexit British GIF - Brexit British Pirate - Discover &amp; Share GIFs](https://tenor.com/view/brexit-british-pirate-england-sinking-gif-5922477): Click to view the GIF
- [GIGABYTE RTX 3090 GV-N3090TURBO-24GD GDDR6X HDMI \ DP PCI-E Graphic Card - Newegg.com](https://www.newegg.ca/p/1FT-000A-005U6): Buy GIGABYTE RTX 3090 GV-N3090TURBO-24GD GDDR6X HDMI \ DP PCI-E Graphic Card with fast shipping and top-rated customer service. Once you know, you Newegg!
- [How to upgrade GPU memory. Upgrade 2080ti to 22G. 2080ti22g.com](https://www.youtube.com/watch?v=DhHYhkx8RiQ): Buy now: 2080ti22g.comToday, I&#39;m going to modify a 2080 Ti graphics card. It&#39;s a blower-style 2080 Ti. I want to upgrade it to 22GB and also replace the stoc...
- [AMD ROCm 6.0 adds support for Radeon RX 7900 GRE and PRO W7800 - VideoCardz.com](https://videocardz.com/newz/amd-rocm-6-0-adds-support-for-radeon-rx-7900-gre-and-pro-w7800): AMD ROCm 6.0 released The latest ROCm platform will now support RX 7900 GRE and PRO W7800.  AMD keeps improving its ROCm platform by supporting new AI algorithms and machine learning models. The recen...
- [GeForce RTX 2080 Ti with upgraded 22GB memory for AI workloads lands on eBay for $500 - VideoCardz.com](https://videocardz.com/newz/geforce-rtx-2080-ti-with-upgraded-22gb-memory-for-ai-workloads-lands-on-ebay-for-500): RTX 2080 Ti with twice the memory for AI One can now find NVIDIA 5-year-old architecture with upgraded memory specs at relatively low price.  GPUs are no longer just limited to gaming and cryptomining...
- [(4k) RTX 3090*4! It is a Luxury in Dreams](https://m.youtube.com/watch?v=fdtAOPyZ9z8): This computer first wanted to install air -cooled heat dissipation.Later, because the original graphics card was too thick and could not be installed, it was...
- [GeForce RTX 2080 Ti with upgraded 22GB memory for AI workloads lands on eBay for $500 - VideoCardz.com](https://videocardz.com/newz/geforce-rtx-2080-ti-with-upgraded-22gb-memory-for-ai-workloads-lands-on-): RTX 2080 Ti with twice the memory for AI One can now find NVIDIA 5-year-old architecture with upgraded memory specs at relatively low price.  GPUs are no longer just limited to gaming and cryptomining...
- [[Feature request] Any plans for AMD XDNA AI Engine support on Ryzen 7x40 processors? · Issue #1499 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/1499): Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...
- [ATOM Echo Smart Speaker Development Kit](https://shop.m5stack.com/products/atom-echo-smart-speaker-dev-kit?variant=34577853415588): ATOM ECHO is a programmable smart speaker.This eps32 AIoT Development Kit has a microphone and speaker for AI voice interaction light and small. It can be access AWS, Baidu, ESPHome and Home Assistant...
- [Lian-Li O11 Dynamic XL ROG certificated -Black color Tempered Glass](https://www.canadacomputers.com/product_info.php?cPath=6_6004_5960&item_id=151208): Lian Li O11 Dynamic XL ROG certificated, Front and Left Tempered Glass, E-ATX, ATX Full Tower Gaming Computer Case - Black

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1206999570790481951) (19 messages🔥): 

- **Quantum Leap in Model Compression**: `@drawless111` shared excitement about 1.5 bit quantization being worked on and posted a [GitHub pull request](https://github.com/ggerganov/llama.cpp/pull/5453) as evidence of this development. They also mentioned impressive benchmarks for a 70-billion parameter model (70B) that hinted at substantial advancements in quantization efficiency.

- **Quant Size Anticipated to be a Gamechanger**: `@heyitsyorkie` and `@drawless111` discussed the potential impact of 1.5 bit quant sizes, with `@heyitsyorkie` expressing curiosity about performance quality when compared to other quantization methods. `@drawless111` responded with optimism, highlighting that these new quants - particularly IQ2 and IQ3 - are outperforming previous models and could soon replace them.

- **Models Running on Slim Hardware**: Both `@drawless111` and `@heyitsyorkie` discussed the implications of new quant sizes, like IQ1, allowing for large 70B models to run on machines with only 16 GB of VRAM, addressing a previous message about encountering 5 IQ1 models on Hugging Face which ballooned to 10 shortly.

- **Performance Details of Quant Models**: `@drawless111` provided detailed comparisons of different quantized models, discussing their sizes (such as IQ2_XXS at 2 GB) and performance. The post-compression fine-tuning was noted as a factor that could affect the performance of these compressed models.

- **Troubleshooting for LM Studio AppImage**: After a user `@w_sky` mentioned the `LM_Studio-0.2.14-beta-1.AppImage` for Linux was crashing, `@heyitsyorkie` inquired whether the CPU supported `AVX2 instructions`, suggesting a potential cause for the crash.

**Links mentioned**:

- [Nexesenex/NousResearch_Yarn-Llama-2-70b-32k-iMat.GGUF · Hugging Face](https://huggingface.co/Nexesenex/NousResearch_Yarn-Llama-2-70b-32k-iMat.GGUF): no description found
- [Claire Bennet Heroes GIF - Claire Bennet Heroes Smile - Discover &amp; Share GIFs](https://tenor.com/view/claire-bennet-heroes-smile-happy-relieved-gif-5008424): Click to view the GIF
- [1.5 bit quantization by ikawrakow · Pull Request #5453 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5453): This draft PR is a WIP that demonstrates 1.5 bits-per-weight (bpw) quantization. Only CUDA works, there is no implementation for the other supported back-ends. CUDA, AVX2 and ARM_NEON are implement...

  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/) (1 messages): 

.ben.com: markdown has linebreaks
end your line with two spaces
the carriage returns
  

---


### LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1207094925192331294) (7 messages): 

- **AVX Version Woes for LM Studio**: User `@rafalsebastian` faces disappointment upon learning that their processor does not support **AVX2 instructions** needed for LM Studio. They inquire if it's possible to run LM Studio on a CPU with only AVX support.
- **Windows Rescue for Older CPUs**: `@heyitsyorkie` responds with a solution, directing to download **version 0.2.10 AVX beta release** for Windows from [LM Studio's beta releases](https://lmstudio.ai/beta-releases.html), although they recommend upgrading to a CPU with AVX2 instructions for optimal performance.
- **Salvation for Workstations**: `@rafalsebastian` expresses gratitude as their older workstation is saved from being scrapped thanks to the AVX beta release.
- **Linux Users Left Waiting**: Despite `@rafalsebastian`'s interest in a Linux version of the **AVX beta**, `@heyitsyorkie` confirms that **no Linux version is available** and there likely won't be one for some time.
- **Reluctant to Experiment**: `@rafalsebastian` shares that they have another workstation with Xeon CPUs that support AVX2, but they hesitate to use their primary work machine for experimentation with LM Studio.

**Links mentioned**:

[LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html): no description found

  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1207052700936839258) (1 messages): 

- **Comedic Inquiry on Bot's Efficiency**: `@wolfspyre` quipped about bot functionality with a light-hearted question: *do they have to work?* followed by a smiling emoji symbolizing a **grin**.
- **A Case of the Repetitive Bot Syndrome**: `@wolfspyre` humorously portrayed a scenario of a bot outputting the same text repetitively, complete with a playful exaggeration of the repetition and comic sound effects. The concern was also raised about potential **repetition errors** involving task distribution among workers.
  

---



### OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1207041425871609967) (2 messages): 

- **ChatGPT gets a Memory Upgrade**: `@abdubs` announced that ChatGPT is being tested for its **new memory feature**, which will allow it to remember past conversations. Users can control this feature by telling ChatGPT to remember or forget information, and it's being rolled out to a select group of Free and Plus users. For full details, users can read more on [OpenAI's blog post](https://openai.com/blog/memory-and-new-controls-for-chatgpt).

- **Meet Sora, the Text-to-Video Model**: `@abdubs` introduced **Sora**, OpenAI's first model that generates up to 60-second videos from text descriptions, which can include complex scenes and characters showing emotions. Currently available to red teamers and creative professionals for feedback, more information is available at [OpenAI’s Sora introduction page](https://openai.com/sora).

**Links mentioned**:

- [Memory and new controls for ChatGPT](https://openai.com/blog/memory-and-new-controls-for-chatgpt): We’re testing the ability for ChatGPT to remember things you discuss to make future chats more helpful. You’re in control of ChatGPT’s memory.
- [Sora: Creating video from text](https://openai.com/sora): no description found

  

---


### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1206874952482029578) (338 messages🔥🔥): 

- **Diverse Uses for GPT-4**: In a series of messages, participants, including `@feltsteam0`, discussed the different ways people use GPT-4, with some using it to simplify complex topics and others concerned about potential for increased laziness among users.

- **Google vs OpenAI AI Models**: `@kevinlk` questioned Google's strategy for releasing their GPT model with similar pricing to OpenAI's models. `@lumirix` mentioned the perks of Gemini Advanced, and several users compared the performance of OpenAI models with Google's newly released iterations.

- **Concerns About GPT's Performance**: Users, such as `@pigondrugs` and `@drinkoblog.weebly.com`, expressed their issues with the latest updates to GPT models, specifically pointing out difficulties in retaining context and maintaining coherent long-form communication.

- **New Player in Town - Abacus.AI's Smaug-72B**: A few users, including `@cassofthenight` and others, reacted to the announcement of Abacus.AI's latest model outperforming OpenAI's, stirring a discussion on the competition in the AI arena.

- **Sora - OpenAI's Next Leap in AI**: Discussions blossomed around OpenAI's text-to-video model called Sora, with users like `@dooz`, `@johnnyrobert`, and `@infidelis` speculating on its potential impact on creative industries and the limitations of current AI in filmmaking.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/977259063052234752/1207751838846423040): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Careers](https://openai.com/careers): Developing safe and beneficial AI systems requires people from a wide range of disciplines and backgrounds. We’re always looking for curious minds to join our team.
- [Our next-generation model: Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/?utm_source=yt&utm_medium=social&utm_campaign=gemini24&utm_content=&utm_term=): Gemini 1.5 delivers dramatically enhanced performance, with a breakthrough in long\u002Dcontext understanding across modalities.
- [Sora: Creating video from text](https://openai.com/sora): no description found
- [All Neural Networks, All Autonomous, All 1X speed](https://www.1x.tech/discover/all-neural-networks-all-autonomous-all-1x-speed): Our environments are designed for humans, so we design our hardware to take after the human form for maximum generality. To make the best use of this general-purpose hardware, we also pursue the maxim...
- [All Neural Networks. All Autonomous. All 1X speed | 1X Studio](https://www.youtube.com/watch?v=iHXuU3nTXfQ): #1X #Android #EmbodiedLearningAll Neural Networks. All Autonomous. All 1X speed. This video contains no teleoperation, no computer graphics, no cuts, no vide...
- [EVE for Real World Manipulation | by 1X](https://www.youtube.com/watch?v=20GHG-R9eFI): #1XTechnologies #Android #RoboticsThis video demonstrates our hardware and motion AI’s natural dynamics and precision by running a preplanned sequence of mot...
- [Tweet from Andrew Curran (@AndrewCurran_)](https://fxtwitter.com/andrewcurran_/status/1758153524846944284?s=46): Gemini 1.5 is capable of a 10 million token context window! Amazing! We are escalating.  They will introduce 1.5 Pro with a 128k context window. Then plan to introduce pricing tiers that start at the ...
- [Error 404 (Not Found)!!!](https://blog.google/technology/ai/google-gemini-next-generation-model-february-): no description found

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1206957552831832114) (131 messages🔥🔥): 

- **GPT's Vision for Video Understanding Tutorial**: `@flokyhuan` shared a [link to OpenAI's notebook](https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding) which outlines how to use GPT-4's visual capabilities for video understanding, even though GPT-4 cannot directly process videos.
- **Fine-Tuning Image Recognition Clarification**: Users `@flokyhuan` and `@solbus` discussed that fine-tuning OpenAI language models is currently text-only, and the model does not support fine-tuning for image recognition tasks.
- **Service Outages Timely Troubles**: Several users including `@cmt283`, `@james18btdoomer`, `@snowzer`, and `@lumirix` reported and discussed various error messages and interruptions in service when using GPT-4, indicating a potential widespread system issue.
- **GPT-4 Access and Latency Woes**: User `@3top1a` encountered frequent errors during custom GPT prompts, wondering about the limits to GPT's knowledge and the feasibility of processing large text files.
- **Intrigue Around Sora**: A discussion led by `@antnation`, `@wccats11`, and `@doperabbitwojak` highlighted excitement for Sora, OpenAI’s text-to-video model, which is in development and currently unavailable to users.

**Links mentioned**:

- [OpenAI Status](https://status.openai.com/): no description found
- [Processing and narrating a video with GPT&#x27;s visual capabilities and the TTS API | OpenAI Cookbook](https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding): no description found

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1206983306969882644) (86 messages🔥🔥): 

- **Notice Me, Chatbot**: `@beanz_and_rice` made several attempts to engage with the chatbot, expressing feeling unnoticed, until `@toror` playfully acknowledged the situation.
- **Horizontally Rotated Woes**: `@kv1383` expressed frustration with images rotating incorrectly and disappearing webp files, to which `@darthgustav.` replied explaining potential GPT model limitations with orientation.
- **Too Big Prompt Dilemma**: `@rdcdt` queried about simplifying a 4k character long prompt, and was directed by `@bambooshoots` to the *behavioral adjustment tool* for help ([g-6qn4yGrBR-directive-gpt-llm-behavioral-adjustment-tool](https://chat.openai.com/g/g-6qn4yGrBR-directive-gpt-llm-behavioral-adjustment-tool)).
- **Seeking a Yes/No Only Response**: `@loictonneau` sought a way to craft prompts that elicit only "yes" or "no" responses from GPT, and `@darthgustav.` provided a structured output template to facilitate this.
- **Token Optimization Techniques**: `@realspacekangaroo` discussed strategies for minimizing token usage in prompts, while `@eskcanta` and `@darthgustav.` suggested focusing on clear, efficient language and the potential risks and benefits of using intentionally poor grammar for cost-saving.

**Links mentioned**:

- [no title found](https://chat.openai.com>.): no description found
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1037561178286739466): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1206983306969882644) (86 messages🔥🔥): 

- **Technical Inquiry on Image Rotation and File Persistence**: `@kv1383` expressed frustration that an image ended up being rotated horizontally instead of vertically, and also mentioned a dislike for webp files as they seem to disappear after some time. To this, `@darthgustav.` responded that DALL-E 3 does not truly understand orientation and offers a 50/50 chance of getting it right.

- **Streamlining Interaction with OpenAI**: Several users, including `@loictonneau`, `@rdcdt` and `@beanz_and_rice`, engaged in conversations about using ChatGPT and creating prompts, with `@loictonneau` seeking help to create a yes/no prompt and `@rdcdt` asking for prompt simplification advice. Assistance and resources were provided by `@darthgustav.` and `@bambooshoots`.

- **Prompt Grammar Debate**: A debate over the use of proper grammar in prompts was sparked by `@realspacekangaroo`, arguing that imprecise grammar can save tokens, which is economically beneficial for large-scale use. `@eskcanta` cautioned against this practice for clarity's sake and to avoid unforeseen model updates affecting prompt interpretation.

- **Concerns on Model Contamination and Behavior Tuning**: `@stealth2077` in a series of messages expressed concerns about editing model outputs and the potential for contamination of context. `@eskcanta` suggested positive reinforcement and guidance as a strategy to avoid context issues and provided examples of their methodology for training the model on specific tasks.

- **Explorations of Text Classification Using GPT**: `@ben.30` and `@romera5032` discussed their experiences using GPT for text classification within their companies. `@ben.30` encountered difficulties with GPT classifying 'boat skip' and sought advice on forcing the model to adhere to a knowledge base, with the conversation moving to a direct exchange after connecting on the platform.

**Links mentioned**:

- [no title found](https://chat.openai.com>.): no description found
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1037561178286739466): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1207707938068897802) (5 messages): 

- **Magic.dev mentioned by 4biddden**: User `@4biddden` just shared the word `magic.dev` without further context or explanation.

- **Gummybee highlights a new paper**: `@giftedgummybee` shared an [arXiv paper](https://arxiv.org/abs/2402.08268) asserting a new approach to language models by incorporating video sequences to enhance understanding of the physical world, overcoming challenges of memory constraints and computational complexity.

- **Speculation on the nature of a new architecture**: `@hexani` mused that a certain undisclosed new architecture could simply be akin to **DeepMind's RETRO** under a different name.

- **Debate around mysterious architecture continues**: Following up, `@hexani` invited others to guess what the new architecture might be, insinuating curiosity and anticipation about its possible features.

- **Predicting the architecture’s identity**:`@atkinsman` surmised that the unrevealed architecture could likely employ an approach similar to **RETRO or self-extend**, rather than being completely novel, speculating that recent releases by competitors may have influenced its development.

**Links mentioned**:

[World Model on Million-Length Video And Language With RingAttention](https://arxiv.org/abs/2402.08268): Current language models fall short in understanding aspects of the world not easily described in words, and struggle with complex, long-form tasks. Video sequences offer valuable temporal information ...

  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1207123443850674206) (5 messages): 

- **Zuckerberg Shifts Perception in AI and VR**: User `@nonameusr` shared a link ([Zuck's AI and VR take](https://vxtwitter.com/pitdesi/status/1757552017042743728)) hinting at a perception shift where **Mark Zuckerberg** is seen transitioning from the villain to the savior in AI and VR.

- **Skepticism About VR Passthrough Quality**: `@teknium` responded, **agreeing** with the opinions in a linked post except for the claim of superior passthrough, and noted that the **passthrough on their Quest 3 was terrible**.

- **The Mysterious Rock-Cat Raises Eyebrows**: User `@error.pdf` shared a gif from Tenor ([Rock-Cat's Eyebrow Raise](https://tenor.com/view/rock-cat-eyebrow-cat-meme-sus-dwayne-johnson-gif-14343467910353677310)) that humorously combines a cat with Dwayne "The Rock" Johnson's iconic eyebrow raise.

**Links mentioned**:

[Rock Cat Eyebrow Cat GIF - Rock cat Eyebrow cat Meme - Discover &amp; Share GIFs](https://tenor.com/view/rock-cat-eyebrow-cat-meme-sus-dwayne-johnson-gif-14343467910353677310): Click to view the GIF

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1206906627668713512) (37 messages🔥): 

- **CUDA for AMD GPUs?:** `@leontello` introduced [ZLUDA](https://github.com/vosen/ZLUDA), a tool that allows unmodified CUDA applications to run on AMD GPUs with near-native performance. However, `@adjectiveallison` clarified that ZLUDA is essentially abandoned, with updates only expected for workloads of personal interest to the developer.

- **Wavelet Space Attention enhancing Transformers:** An arXiv paper shared by `@euclaise` discusses improving long sequence learning capabilities in Transformers through the implementation of [Wavelet Space Attention (WavSpA)](https://arxiv.org/abs/2210.01989).

- **New Local AI Assistants Merge:** `@sanjay920` posted a GitHub link to [Rubra](https://github.com/acorn-io/rubra), a project merging openhermes and neuralchat aimed at simplifying the creation of AI Assistants and Large Language Models. This announcement was met with enthusiasm by `@teknium`, while `@gabriel_syme` humorously played down the idea of it being truly local.

- **Impressive Context Size for LLM:** `@if_a` and others discussed the introduction of Gemini 1.5 Pro by Google, highlighting its [10M token context length](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#gemini-15) and efficient MoE architecture, marking it as possibly a significant upgrade from previous models.

- **Multilingual Generative Model with Instructions in 101 Languages:** `@.benxh` shared a [Hugging Face link](https://huggingface.co/CohereForAI/aya-101) to Aya 101, a model that reportedly outperforms both mT0 and BLOOMZ, featuring capabilities for instructions in 101 languages and trained on a vast dataset including xP3x and other collections.

**Links mentioned**:

- [Instruction Tuning with Human Curriculum](https://arxiv.org/abs/2310.09518): In building instruction-tuned large language models (LLMs), the importance of a deep understanding of human knowledge can be often overlooked by the importance of instruction diversification. This res...
- [Software in the natural world: A computational approach to emergence in complex multi-level systems](https://arxiv.org/abs/2402.09090): Understanding the functional architecture of complex systems is crucial to illuminate their inner workings and enable effective methods for their prediction and control. Recent advances have introduce...
- [Our next-generation model: Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#gemini-15): Gemini 1.5 delivers dramatically enhanced performance, with a breakthrough in long\u002Dcontext understanding across modalities.
- [Synthetic Data for Finetuning: Distillation and Self-Improvement](https://eugeneyan.com/writing/synthetic/): Overcoming the bottleneck of human annotations in instruction-tuning, preference-tuning, and pretraining.
- [CohereForAI/aya-101 · Hugging Face](https://huggingface.co/CohereForAI/aya-101): no description found
- [WavSpA: Wavelet Space Attention for Boosting Transformers&#39; Long Sequence Learning Ability](https://arxiv.org/abs/2210.01989): Transformer and its variants are fundamental neural architectures in deep learning. Recent works show that learning attention in the Fourier space can improve the long sequence learning capability of ...
- [Tweet from Hao Liu (@haoliuhl)](https://x.com/haoliuhl/status/1757828392362389999?s=46&t=5): We are excited to share Large World Model (LWM), a general-purpose 1M context multimodal autoregressive model. It is trained on a large dataset of diverse long videos and books using RingAttention, an...
- [GitHub - vosen/ZLUDA: CUDA on AMD GPUs](https://github.com/vosen/ZLUDA): CUDA on AMD GPUs. Contribute to vosen/ZLUDA development by creating an account on GitHub.
- [GitHub - rubra-ai/rubra: AI Assistants, LLMs and tools made easy](https://github.com/acorn-io/rubra): AI Assistants, LLMs and tools made easy. Contribute to rubra-ai/rubra development by creating an account on GitHub.
- [Sora: Creating video from text](https://openai.com/sora): no description found
- [Tweet from Hao Liu (@haoliuhl)](https://x.com/haoliuhl/status/1757828392362389999?s=46&t=56KhEVcLcr_8q0kCmyuWBg): We are excited to share Large World Model (LWM), a general-purpose 1M context multimodal autoregressive model. It is trained on a large dataset of diverse long videos and books using RingAttention, an...
- [Tweet from Haihao Shen (@HaihaoShen)](https://x.com/HaihaoShen/status/1758048469091307717?s=20): 📽️Editing LLM knowledge is possible, e.g., Rank-One Model Editing (ROME). 📔Paper: https://arxiv.org/pdf/2202.05262.pdf 🎯Sample code: https://github.com/intel/intel-extension-for-transformers 💣The ...

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1206984989376389211) (536 messages🔥🔥🔥): 

- **QuIP# - A Leap in Post-Training Quantization**: User `@stellaathena` shared a [research paper](https://arxiv.org/abs/2402.04396) discussing QuIP#, a method for post-training quantization of large language models achieving state-of-the-art results with 2-3 bit precision. They suggest that this approach outperforms previous methods, especially in highly compressed regimes.

- **Intriguing Rumors**: Users converse about potential future advancements like "optimal 0.5 bit quantization" (`@nruaif`) and humorous speculation on the next naming convention for quantization papers by `@if_a`.

- **New Frontier in Model Routing Analysis**: User `@teknium` shares a [routing analysis study](https://x.com/fejo_11/status/1757417292659310675?s=46) based on Mixtral 8x7B model from @MistralAI using POS tags instead of document context, suggesting a new research direction in model understanding.

- **reViSiTing the Hand Issues**: The conversation on "hands" is telling about the community's ongoing challenge with detailed image generation as users like `@giftedgummybee` engage in technical jargon to highlight improvements and benchmarks.

- **OpenAI's Sora Video Generation**: Members of the chat, including `@otisai`, `@bstdev`, and `@leontello`, were abuzz with excitement after OpenAI's announcement of [Sora](https://openai.com/sora), a text-to-video model that marks a significant advancement in AI-generated video content. The chat reflects the impact this technology could have across AI communities and associated industries.

**Links mentioned**:

- [Dancing Emoji GIF - Dancing emoji - Discover &amp; Share GIFs](https://tenor.com/view/dancing-emoji-gif-15627004785323743466): Click to view the GIF
- [Tweet from Ben Nash (@bennash)](https://fxtwitter.com/bennash/status/1758203109573059034): The existing competition with the same prompt. Not even close!!
- [Tweet from Ben Nash (@bennash)](https://fxtwitter.com/bennash/status/1758200859547025779): This video was made with the not-yet-released Sora AI technology just announced from OpenAi. This changes everything. It&#39;s 27 seconds from a text prompt.   Here is their prompt: Prompt: A white an...
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396): Post-training quantization (PTQ) reduces the memory footprint of LLMs by quantizing their weights to low-precision. In this work, we introduce QuIP#, a weight-only PTQ method that achieves state-of-th...
- [Tweet from jf (@fejo_11)](https://x.com/fejo_11/status/1757417292659310675?s=46): Mixtral 8x7B: Routing Analysis based on POS tags  I conducted a routing analysis using @MistralAI&#39;s Mixtral 8x7B model, focusing on Part-of-Speech (POS) tags, diverging from the original methodolo...
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353): Among the widely used parameter-efficient finetuning (PEFT) methods, LoRA and its variants have gained considerable popularity because of avoiding additional inference costs. However, there still ofte...
- [Tweet from Jim Fan (@DrJimFan)](https://x.com/drjimfan/status/1758210245799920123?s=46&t=Z1UUKUZo-Mhpzs0t8qSXZA): If you think OpenAI Sora is a creative toy like DALLE, ... think again. Sora is a data-driven physics engine. It is a simulation of many worlds, real or fantastical. The simulator learns intricate ren...
- [World Model on Million-Length Video And Language With RingAttention](https://arxiv.org/abs/2402.08268): Current language models fall short in understanding aspects of the world not easily described in words, and struggle with complex, long-form tasks. Video sequences offer valuable temporal information ...
- [dataautogpt3/ProteusV0.3 · Hugging Face](https://huggingface.co/dataautogpt3/ProteusV0.3): no description found
- [
    
      
        Representation Engineering Mistral-7B an Acid Trip
      
    
  ](https://vgel.me/posts/representation-engineering/): no description found
- [The San Francisco Compute Company](https://sfcompute.com/): no description found
- [Sora: Creating video from text](https://openai.com/sora): no description found
- [liuhaotian/llava-v1.6-34b · Hugging Face](https://huggingface.co/liuhaotian/llava-v1.6-34b): no description found
- [Laverne And GIF - Laverne And Shirley - Discover &amp; Share GIFs](https://tenor.com/view/laverne-and-shirley-funny-cool-gif-27491810): Click to view the GIF
- [NousResearch/Nous-Hermes-2-Vision-Alpha · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Vision-Alpha): no description found
- [Oppenheimer Cillian Murphy GIF - Oppenheimer Cillian murphy Cillian - Discover &amp; Share GIFs](https://tenor.com/view/oppenheimer-cillian-murphy-cillian-murphy-peaky-blinders-gif-1787947313354313976): Click to view the GIF
- [NExT-GPT](https://next-gpt.github.io/): no description found
- [gpulist](https://gpulist.ai/): buy and sell spare gpu capacity. made by gpulist.ai
- [eleutherai](https://wandb.ai/eleutherai/huggingface/runs/ajy0h7rf): Weights & Biases, developer tools for machine learning
- [‎Practical AI: Machine Learning, Data Science on Apple Podcasts](https://podcasts.apple.com/gb/podcast/data-synthesis-for-sota-llms/id1406537385?i=1000644406332>.): ‎Technology · 2024
- [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219): The recent surge of generative AI has been fueled by the generative power of diffusion probabilistic models and the scalable capabilities of large language models. Despite their potential, it remains ...
- [Tweet from AI at Meta (@AIatMeta)](https://x.com/AIatMeta/status/1758176023588577326?s=20): Today we’re releasing V-JEPA, a method for teaching machines to understand and model the physical world by watching videos. This work is another important step towards @ylecun’s outlined vision of AI ...
- [GitHub - LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM): Contribute to LargeWorldModel/LWM development by creating an account on GitHub.
- [[missing post]](https://www.greaterwrong.com/posts/5spBu): no description found
- [Xigmoid: An Approach to Improve the Gating Mechanism of RNN](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9892346): This work proposes an innovative approach for the gating mechanism of RNN class models. A transfer function is embedded into the original sigmoid to form a new gate function called xigmoid. The purpos...
- [GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.](https://github.com/comfyanonymous/ComfyUI): The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI
- [GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui): Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.
- [Steering GPT-2-XL by adding an activation vector](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector>): Summary: We demonstrate a new scalable way of interacting with language models: adding certain activation vectors into forward passes.[2] Essentially, we add together combinations of forward passes in...

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1206908017510195220) (53 messages🔥): 

- **Searching for GPT-4 Alternatives**: `@natefyi_30842` inquired about more affordable coding models as an alternative to GPT-4. `@teknium` suggested **Deepseek Coder**, and upon asking where to find it hosted, `@teknium` mentioned perhaps on [Together](https://togetherplatform.com/).
  
- **SFT vs. Continued Pretraining Clarified**: `@natefyi_30842` sought clarification on the difference between *SFT* (supervised fine-tuning) and continued pretraining, with `@teknium` confirming that *continued pretraining* generally uses a raw corpus without instruction focus.

- **MIQU's Training Unveiled**: `@teknium` explained that MIQU was continued pretrained from the *llama-2 70b* model and then instruction-tuned (SFT'd), with only its final form being made accessible.

- **AI Benchmarking Made Easy**: `@nerdabhay` asked for resources to test a trained model, and `@teknium` recommended the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), while `@atkinsman` shared a link to a Google Colab for automatic evaluation setup by [llm-autoeval](https://github.com/mlabonne/llm-autoeval).
  
- **API Options for Vision Language Models Explored**: `@vikas.p` inquired about the best vision language models available via an API with decent rate limits and pricing. Multiple suggestions were made, including GPT-4V which scales with total API spend, and `@leontello` noted the existence of Qwen-VL and LLaVA models while `@orabazes` recommended checking **Replicate** for hosting these models, with mention of [Alibaba hosting Qwen-VL](https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start) albeit with low rate limits.

**Links mentioned**:

- [Tweet from AMD Quietly Funded A Drop-In CUDA Implementation Built On ROCm: It's Now Open-Source - Phoronix](https://www.phoronix.com/review/radeon-cuda-zluda): no description found
- [GitHub - mlabonne/llm-autoeval: Automatically evaluate your LLMs in Google Colab](https://github.com/mlabonne/llm-autoeval?tab=readme-ov-file"): Automatically evaluate your LLMs in Google Colab. Contribute to mlabonne/llm-autoeval development by creating an account on GitHub.
- [快速开始_模型服务灵积(DashScope)-阿里云帮助中心](https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start): no description found

  

---


### Nous Research AI ▷ #[collective-cognition](https://discord.com/channels/1053877538025386074/1154961277748256831/1206982859089387562) (3 messages): 

- **Project Faces Downtime**: `@adjectiveallison` encountered an issue when attempting to access the site, questioning if the project is still active.
- **Modes Break the Machine**: `@teknium` confirmed that due to new modes in chat GPT, the website broke, and the maintaining team could not sustain it, resulting in the project's current inactivity.
  

---



### Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1207433385329561631) (1 messages): 

- **Solving the Pink Elephant Problem**: `@canadagoose1` highlighted a new **EleutherAI** paper that addresses the challenge of making chatbots avoid certain topics, known as the Pink Elephant Problem. The paper introduces **Direct Principle Feedback (DPF)**, a technique that outperforms traditional models and is on par with **GPT-4**, and can be found [here](https://arxiv.org/abs/2402.07896).
  
- **DPF for Customizable Chatbot Control**: The announcement shared insights into the **Direct Principle Feedback (DPF)** method that allows fine-grained control over language models by avoiding the need for reranking responses, making it a promising approach for real-life AI fine-tuning (RLAIF) applications.

- **Read More on Twitter**: Additional information and discussions on the **Pink Elephant Problem** and the newly published paper can be followed on a Twitter thread posted by `@synth_labs`, inviting further exploration of the research [here](https://fxtwitter.com/synth_labs/status/1757227081673449666?s=20).

**Links mentioned**:

- [Suppressing Pink Elephants with Direct Principle Feedback](https://arxiv.org/abs/2402.07896): Existing methods for controlling language models, such as RLHF and Constitutional AI, involve determining which LLM behaviors are desirable and training them into a language model. However, in many ca...
- [Tweet from Open Synth Lab (@synth_labs)](https://fxtwitter.com/synth_labs/status/1757227081673449666?s=20): PINK ELEPHANTS! 🐘 Now, don’t think about it.  Chatbots also find this supremely difficult. Ask one of the most popular open source models NOT to talk about pink elephants, and it will fail 34% of the...

  

---


### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1206899065321824366) (228 messages🔥🔥): 

- **XLMR Language Detection Curiosity**: `_michaelsh` asked about how to extract the language from the **XLM-RoBERTa** model as mentioned in a [Hugging Face post](https://huggingface.co/docs/transformers/model_doc/xlm-roberta), curious to know the method of language determination.

- **reka.ai Model Speculations**: `@rallio.` wondered if the reka.ai model could be a **T5** style model since the founder was the UL2 model guy at Google and mentioned its 20-billion-parameter scale. `@stellaathena` responded indicating that the size of a model doesn't necessarily correlate with its style, and emphasized that practical considerations as important as technical motivations.

- **Cloud Resource Recommendations for NLP**: `@pxxxl` inquired about the best cloud resources for training an **NLP Classification model**, receiving suggestions for GCP, Colab, Runpod, and vast.ai, the latter needing caution if unfamiliar with pitfalls as per `@ad8e`.

- **Inquiries About Custom Adapters on Mamba**: `@vidava` discussed challenges and sought guidelines surrounding the creation of semicustom LLM models with their own fine-tuning adapters for models like Mamba. They expressed interest in obtaining resources to conduct further experiments and engaged in a detailed dialogue about potential solutions, including torch parameterizations and dynamically modifying class methods.

- **Gemini 1.5 – A Leap in Multi-Modal AI**: `@karatsubabutslower` shared a [Twitter link](https://vxtwitter.com/JeffDean/status/1758146022726041615) highlighting Google’s Gemini 1.5, prompting `@fessus` to ponder upon its implications for robotics, with `@clock.work_` and `@karatsubabutslower` discussing the real-time data stream processing that robotics require, beyond the capabilities of models showcased in demos.

**Links mentioned**:

- [Memory and new controls for ChatGPT](https://openai.com/blog/memory-and-new-controls-for-chatgpt): We’re testing the ability for ChatGPT to remember things you discuss to make future chats more helpful. You’re in control of ChatGPT’s memory.
- [Our next-generation model: Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#architecture): Gemini 1.5 delivers dramatically enhanced performance, with a breakthrough in long\u002Dcontext understanding across modalities.
- [lora_example.py](https://gist.github.com/Chillee/a8d2070b1b7b3f97d8c87bac3c366f8e): lora_example.py. GitHub Gist: instantly share code, notes, and snippets.
- [BridgeAI Programme - Brief - Digital Catapult FutureScope](https://futurescope.digicatapult.org.uk/our-programmes/bridgeai-programme/bridgeai-programme-brief/?utm_source=Website&utm_medium=IUK+KTN&utm_campaign=KTN&utm_id=IUKKTN&utm_content=IUKKTNWebsitebrief#section-2): Digital Catapult is launching an accelerator programme; Innovate UK BridgeAI, that seeks to stimulate the adoption of artificial intelligence and machine learning technologies in agriculture, creative...
- [Tweet from Open Synth Lab (@synth_labs)](https://fxtwitter.com/synth_labs/status/1757227081673449666>)): PINK ELEPHANTS! 🐘 Now, don’t think about it.  Chatbots also find this supremely difficult. Ask one of the most popular open source models NOT to talk about pink elephants, and it will fail 34% of the...
- [Open Synth Lab (@synth_labs)](https://nitter.osmarks.net/synth_labs/status/1757227081673449666): PINK ELEPHANTS! 🐘 Now, don’t think about it.  Chatbots also find this supremely difficult. Ask one of the most popular open source models NOT to talk about pink elephants, and it will fail 34% of the...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1206876064702406726) (218 messages🔥🔥): 

- **Suspicions Around MoE Scaling Law Paper**: `@kyo_takano` raised concerns about the MoE scaling law paper. They questioned the unusually perfect loss predictor and consistent parameters achieved by the authors, suggesting an almost perfectly fitted model that generalizes even in a higher-compute regime is highly unlikely.

- **Discussing Encoder-Decoder Pre-Training**: `@loubb` began a conversation on whether it would be beneficial to pre-train parts of an encoder-decoder model, specifically the decoder, for seq2seq tasks like machine translation. The user proposed pre-training the decoder on unsupervised data before fine-tuning on seq2seq tasks, emphasizing the usefulness of learned text representations prior to fine-tuning.

- **LLM Security and Adversarial Compromises**: A new paper, mentioned by `@ai_waifu`, discussed the emergence of secret collusion among communicating AI agents, detailing how steganography might be used to conceal unauthorized information sharing. This highlights the security and privacy concerns arising as the capabilities of LLMs grow.

- **Research on Memorization in LLMs**: Several users, including `@avi.ai`, `@0x_paws`, and `@pizza_joe.`, shared papers addressing the memorization capabilities of large language models (LLMs), exploring both the use of copyrighted content to train LLMs and adversarial efforts to extract information from models.

- **Non-Determinism in GPT-4 and MoE Models**: Extensive discussion occurred regarding the non-determinism noticed in outputs from GPT-4, even when a seed was used. Users like `@catboy_slim_` and `@carsonpoole` debated whether the non-determinism stemmed from MoE implementation, batch effects, or different backend model behaviors.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/729741769192767510/730095596861521970/1196618227543982150): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/729741769192767510/747850033994662000/1194665233671782442): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Tweet from Hannes Stärk (@HannesStaerk)](https://fxtwitter.com/HannesStaerk/status/1695943729314746410): Diffusion models are dead - long live joint conditional flow matching! 🙃 Tomorrow @AlexanderTong7 presents his &#34;Improving and generalizing flow-based generative models with minibatch optimal tran...
- [Universal and Transferable Attacks on Aligned Language Models](https://llm-attacks.org/): no description found
- [Tweet from OpenAI (@OpenAI)](https://x.com/openai/status/1758192957386342435?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): Introducing Sora, our text-to-video model.  Sora can create videos of up to 60 seconds featuring highly detailed scenes, complex camera motion, and multiple characters with vibrant emotions.          ...
- [Tweet from Nature Reviews Physics (@NatRevPhys)](https://fxtwitter.com/NatRevPhys/status/1757089166683230242): Perspective: Generative learning for nonlinear dynamics  By @wgilpin0 @TexasScience  https://rdcu.be/dysiB
- [Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision](https://arxiv.org/abs/2109.08203): In this paper I investigate the effect of random seed selection on the accuracy when using popular deep learning architectures for computer vision. I scan a large amount of seeds (up to $10^4$) on CIF...
- [Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models](https://arxiv.org/abs/2402.07865): Visually-conditioned language models (VLMs) have seen growing adoption in applications such as visual dialogue, scene understanding, and robotic task planning; adoption that has fueled a wealth of new...
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353): Among the widely used parameter-efficient finetuning (PEFT) methods, LoRA and its variants have gained considerable popularity because of avoiding additional inference costs. However, there still ofte...
- [Copyright Traps for Large Language Models](https://arxiv.org/abs/2402.09363): Questions of fair use of copyright-protected content to train Large Language Models (LLMs) are being very actively debated. Document-level inference has been proposed as a new task: inferring from bla...
- [Scalable Diffusion Models with Transformers](https://www.wpeebles.com/DiT.html): no description found
- [Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast](https://arxiv.org/abs/2402.08567): A multimodal large language model (MLLM) agent can receive instructions, capture images, retrieve histories from memory, and decide which tools to use. Nonetheless, red-teaming efforts have revealed t...
- [Computing Power and the Governance of Artificial Intelligence](https://arxiv.org/abs/2402.08797): Computing power, or &#34;compute,&#34; is crucial for the development and deployment of artificial intelligence (AI) capabilities. As a result, governments and companies have started to leverage compu...
- [Secret Collusion Among Generative AI Agents](https://arxiv.org/abs/2402.07510): Recent capability increases in large language models (LLMs) open up applications in which teams of communicating generative AI agents solve joint tasks. This poses privacy and security challenges conc...
- [Scalable Extraction of Training Data from (Production) Language Models](https://arxiv.org/abs/2311.17035): This paper studies extractable memorization: training data that an adversary can efficiently extract by querying a machine learning model without prior knowledge of the training dataset. We show an ad...
- [ZerO Initialization: Initializing Neural Networks with only Zeros and Ones](https://arxiv.org/abs/2110.12661): Deep neural networks are usually initialized with random weights, with adequately selected initial variance to ensure stable signal propagation during training. However, selecting the appropriate vari...
- [Non-determinism in GPT-4 is caused by Sparse MoE](https://152334h.github.io/blog/non-determinism-in-gpt-4/): It&rsquo;s well-known at this point that GPT-4/GPT-3.5-turbo is non-deterministic, even at temperature=0.0. This is an odd behavior if you&rsquo;re used to dense decoder-only models, where temp=0 shou...
- [Trap street - Wikipedia](https://en.wikipedia.org/wiki/Trap_street): no description found
- [no title found](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/): no description found
- [How to run deterministically? · Issue #349 · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/issues/349): I noticed that the deterministic argument was removed in the FlashAttention2 APIs and there isn&#39;t an obvious way to tweak it since the num_splits argument was also removed from the python interfac...
- [Tweet from Jeff Dean (@🏡) (@JeffDean)](https://x.com/jeffdean/status/1758146211029405951?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): Needle in a Haystack Tests Out to 10M Tokens  First, let’s take a quick glance at a needle-in-a-haystack test across many different modalities to exercise Gemini 1.5 Pro’s ability to retrieve informat...
- [A Poster for Neural Circuit Diagrams](https://www.vtabbott.io/ncd-poster/): As some of you might know, I have been working on neural circuit diagrams over the past year or so. These diagrams solve a lingering challenge in deep learning research – clearly and accurately commun...
- [llm-random/research/conditional/moe_layers/expert_choice.py at ad41b940c3fbf004a1230c1686502fd3a3a79032 · llm-random/llm-random](https://github.com/llm-random/llm-random/blob/ad41b940c3fbf004a1230c1686502fd3a3a79032/research/conditional/moe_layers/expert_choice.py#L59): Contribute to llm-random/llm-random development by creating an account on GitHub.
- [Portal](https://portal.valencelabs.com/logg): Home of the TechBio community. Tune into our weekly reading groups (M2D2, LoGG, CARE), read community blogs, and join the discussion forum. 
- [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450): Pre-training and fine-tuning, e.g., BERT, have achieved great success in language understanding by transferring knowledge from rich-resource pre-training task to the low/zero-resource downstream tasks...
- [Improving Black-box Robustness with In-Context Rewriting](https://arxiv.org/abs/2402.08225): Machine learning models often excel on in-distribution (ID) data but struggle with unseen out-of-distribution (OOD) inputs. Most techniques for improving OOD robustness are not applicable to settings ...
- [Are Neighbors Enough? Multi-Head Neural n-gram can be Alternative to Self-attention](https://arxiv.org/abs/2207.13354): Impressive performance of Transformer has been attributed to self-attention, where dependencies between entire input in a sequence are considered at every position. In this work, we reform the neural ...
- [BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning](https://arxiv.org/abs/2002.06715): Ensembles, where multiple neural networks are trained individually and their predictions are averaged, have been shown to be widely successful for improving both the accuracy and predictive uncertaint...
- [Generative learning for nonlinear dynamics | Nature Reviews Physics](https://www.nature.com/articles/s42254-024-00688-2.epdf?sharing_token=D_ImKvUZsRHYzs0lhT-4hNRgN0jAjWel9jnR3ZoTv0OFpVCe5j8bo6KJ1K_rllqrEXyt3r74B4sNMsFSoYzk3qrjVQZAFqeWPvf0ZTRuVS6GZQhz83MTvZr0nlCnrXj25-QPv4XzGPY-Homhk29UsvbEDaEd1lFW8i_n6jM6_1w%3D): no description found
- [UnitY: Two-pass Direct Speech-to-speech Translation with Discrete Units - Meta Research](https://research.facebook.com/publications/unity-direct-speech-to-speech-translation/): We present a novel two-pass direct S2ST architecture, UnitY, which first generates textual representations and predicts discrete acoustic units subsequently.

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1206974168516526100) (8 messages🔥): 

- **Seeking Interpretability Overviews**: `@jaimerv` requested recommendations for an updated overview of interpretability approaches, referencing a paper on **Representation Engineering**.
  
- **Saliency in Vision and Transformers**: `@aiobhinn` offered insights on different lines of research in interpretability, mentioning **salient map approaches** in vision tasks and **attention maps** or **information flow** studies in transformer models.

- **Clarifying Research Focus**: Responding to `@aiobhinn`'s query, `@jaimerv` clarified that their research focuses on evaluations using interpretability techniques, specifically for evaluating **propensity evaluations** like honesty and power-seeking.

- **Diffusion Models Interpretability**: `@rbz99_27250` inquired about methods to evaluate or interpret **diffusion models**, noting a lack of research on the UNET aspect as compared to the CLIP side of problems within diffusion models.
  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1207285411882344468) (17 messages🔥): 

- **Harnessing Trouble with Local Models**: `@christianpala` is encountering issues when trying to use the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/620d6a152b2eb9216138495bfd6db4ea770dec26/lm_eval/models/openai_completions.py#L124) with local models and tokenizer, particularly around calculating logprobs and sorting items that are being returned incorrectly by the tokenizer.
- **Suggested Fix for lm-evaluation-harness**: `@christianpala` suggested a fix for the mentioned issue by changing `self.end_of_text_token_id = self.tokenizer.eos_token` to `self.end_of_text_token_id = self.tokenizer.eos_token_id` but indicated that integrating the tokenizer as an argument isn't directly supported by the harness.

- **Evaluating Math in Language Models**: `@kamilla7693` inquired about how non-vision models handle SAT or GRE math tests' graph and plot questions. `@baber_` and `@stellaathena` noted that models like MATH use LaTeX to represent graphics, whereas some questions simply reference non-existent images.

- **Enquiring about Open-Book and COT Support in Harness**: `@uanu.` asked if the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) supports open-book tasks or Chain of Thought (COT) prompts, with `@hailey_schoelkopf` confirming COT support but no current capabilities for search augmented tasks.

- **Issues with Python Version and Harness Cloning**: `@madison_33844` faced an error regarding Python version compatibility when using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and received advice from `@pminervini` to try updating Python and use a specific older version of the harness (`b281b09`) for replicability with the OpenLLM leaderboard.

**Links mentioned**:

- [Hallucinations Leaderboard - a Hugging Face Space by hallucinations-leaderboard](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard): no description found
- [lm-evaluation-harness/lm_eval/models/openai_completions.py at 620d6a152b2eb9216138495bfd6db4ea770dec26 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/620d6a152b2eb9216138495bfd6db4ea770dec26/lm_eval/models/openai_completions.py#L124): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1207072674048909352) (4 messages): 

- **Potential Misalignment in Pythia-Deduped for 2.8b**: `@pietrolesci` flagged a possible issue with 2.8b models in the Pythia-deduped suite regarding *alignment between training data batches and checkpoints*. They observed that the batch loss isn't decreasing as expected post-training for 2.8b, unlike other model versions.
- **Schoelkopf to the Rescue**: Upon noticing the issue reported by `@pietrolesci`, `@hailey_schoelkopf` acknowledged the concern and promised to follow up on the matter.
- **Call for Collaborative Scrutiny**: `@stellaathena` expressed excitement over the potential to demonstrate the replicability of Pythia via a blog post or workshop paper, highlighting the opportunity for a community-driven verification project.
- **Grateful for Support and Suggestions**: `@pietrolesci` thanked `@hailey_schoelkopf` for looking into the 2.8b alignment issue and appreciated `@stellaathena` for proposing a post-ACL deadline project to delve into the findings.
  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1206885397448097812) (282 messages🔥🔥): 

- **Mistral Performance Discussion**: Users discuss the dependency of Mistral performance on both hardware and server load. `@i_am_dom` emphasizes that a smaller model like GPT-4 can outpace larger models like 7B if the server is robust and not under load.
  
- **Mistral Learning Curve for Interns**: `@nana.wav` is seeking guidance on using Mistral after downloading it, with an intention to finetune the model. Assistance is offered, including suggestions to look up resources like Jupyter notebooks, Kaggle, and Hugging Face for beginners.

- **Users Share Internship Struggles**: `@frosty04212` and others share tales of overwhelming tasks during internships, including migrating entire stacks to Kubernetes and dealing with workplace expectations for (almost) free work.

- **Latency Issues with Mistral API**: `@justinmann.` and `@ginterhauser` report high latency when using Mistral API endpoints like `api.mistral.ai/v1/chat/completions`, and are advised to contact support at Mistral for assistance with scaling issues.

- **Inquiries on Model Specifications and Troubleshooting**: `@drnicefellow` asks about the token count Mistral is trained on, while `@nana.wav` seeks help with execution errors, receiving advice to check updates and ensure correct installations. `@sapphics` discusses challenges with Mistral embed and receives directions to the documentation for clarification.

**Links mentioned**:

- [Stack Overflow - Where Developers Learn, Share, &amp; Build Careers](https://stackoverflow.com/): Stack Overflow | The World&#x2019;s Largest Online Community for Developers
- [Embeddings | Mistral AI Large Language Models](https://docs.mistral.ai/guides/embeddings/): Embeddings are vectorial representations of text that capture the semantic meaning of paragraphs through their position in a high dimensional vector space. Mistral AI Embeddings API offers cutting-edg...
- [Accelerating Systems with Real-time AI Solutions - Groq](https://wow.groq.com/): Groq is providing general early access to the alpha release version of our API free of charge for a limited time for research and development purposes only.
- [Project Jupyter](https://jupyter.org/): The Jupyter Notebook is a web-based interactive computing platform. The notebook combines live code, equations, narrative text, visualizations, interactive dashboards and other media. 
- [Kaggle: Your Machine Learning and Data Science Community](https://www.kaggle.com/): Kaggle is the world&#x2019;s largest data science community with powerful tools and resources to help you achieve your data science goals.
- [GPU Cloud Login | Lambda](https://cloud.lambdalabs.com): no description found
- [AutoTrain – Hugging Face](https://huggingface.co/autotrain): no description found
- [GitHub - guardrails-ai/guardrails: Adding guardrails to large language models.](https://github.com/guardrails-ai/guardrails): Adding guardrails to large language models. Contribute to guardrails-ai/guardrails development by creating an account on GitHub.
- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [GitHub - huggingface/text-generation-inference: Large Language Model Text Generation Inference](https://github.com/huggingface/text-generation-inference): Large Language Model Text Generation Inference. Contribute to huggingface/text-generation-inference development by creating an account on GitHub.
- [GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs](https://github.com/vllm-project/vllm): A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm
- [HuggingChat](https://huggingface.co/chat): Making the community's best AI chat models available to everyone.
- [Hugging Face – The AI community building the future.](https://huggingface.co): no description found
- [GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models.](https://github.com/oobabooga/text-generation-webui): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1206934364563447857) (38 messages🔥): 

- **Praise for DSPy's Prompt Flow**: `@mrdragonfox` shared a positive example of why [DSPy is powerful](https://twitter.com/CShorten30/status/1751656468879708496), highlighting its efficiency by using the LLM as a "device" rather than a "chat" interface.
- **Debate on Model's Production Viability**: `@mrdragonfox` criticized **LangChain** for its complex dependencies, suggesting it's impractical for production use, while `@rolandtannous` mentioned the occurrence of production releases with others holding back due to potential system crashes. Further, `@rabdullin` discussed industry variances in adopting these models, and shared an [NVIDIA demo app](https://github.com/NVIDIA/trt-llm-rag-windows/blob/release/1.0/app.py#L75-L103) for personalized chatbots.
- **Intrigue Around Mistral-7B's Training Data**: Users `@kushagra_67246` and `@gamerboi0129` inquired about the datasets involved in training Mistral-7B, while `@tom_lrd` and `@mrdragonfox` conveyed the secretive nature of such datasets.
- **Clarification on Mistral's Open-Sourced Checkpoint**: `@nofreewill42` sought information on the availability of a raw open-sourced checkpoint following raw internet text pretraining, without finetuning, referring to `mistralai/Mistral-7B-v0.1`.
- **Guide to Chaining LLM Responses**: `@brendawin` queried about integrating an API as a prompt in app development, with `@mrdragonfox` providing guidance on chaining LLMs and handling logic externally, and shared a [link to Mistral's guides](https://docs.mistral.ai/guides/overview/).

**Links mentioned**:

- [Guides | Mistral AI Large Language Models](https://docs.mistral.ai/guides/overview/): Welcome to our Mistral Getting Started guide! This guide is designed to help you learn how to use Mistral API quickly and easily. Learn how to write prompts for a variety of use cases, build a basic R...
- [NVIDIA Chat With RTX](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/): Your Personalized AI Chatbot.
- [trt-llm-rag-windows/app.py at release/1.0 · NVIDIA/trt-llm-rag-windows](https://github.com/NVIDIA/trt-llm-rag-windows/blob/release/1.0/app.py#L75-L103): A developer reference project for creating Retrieval Augmented Generation (RAG) chatbots on Windows using TensorRT-LLM - NVIDIA/trt-llm-rag-windows

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1207062905279741964) (7 messages): 

- **Model Mix-Up Alert**: `@casper_ai` noted that *thebloke*'s model is corrupted. They provided a link to a working version of Mixtral Instruct that is AWQ quantized: [Mixtral Instruct - AWQ](https://huggingface.co/casperhansen/mixtral-instruct-awq).
- **Alternative Mixtral Repository Recommended**: In a follow-up, `@casper_ai` recommends using their [Mixtral Instruct AWQ repository](https://huggingface.co/casperhansen/mixtral-instruct-awq) as the repository from *TheBloke* is currently not functioning.
- **Model Details Shared**: The working version of Mixtral Instruct has **6.48B parameters** and supports **I32 and FP16** tensor types. It's received 8,430 downloads in the last month.
- **Cryptomotion Seeks Help**: New joiner `@cryptomotion` asked for links to the authoritative <#1154028168466923600> documentation.
- **Official Documentation Provided**: `@mrdragonfox` responded with the official [Mistral AI documentation](https://docs.mistral.ai/) and details on how to use the API.

**Links mentioned**:

- [Introduction | Mistral AI Large Language Models](https://docs.mistral.ai/): Mistral AI currently provides two types of access to Large Language Models:
- [casperhansen/mixtral-instruct-awq · Hugging Face](https://huggingface.co/casperhansen/mixtral-instruct-awq): no description found

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1206897875590848523) (10 messages🔥): 

- **Seeking Guidance on MLX and Mistral**: `@hammer_mt` asked for a tutorial on fine-tuning **Mistral 8x7B** using Apple's MLX, similar to a detailed guide available for local LLM fine-tuning on a Mac. [Here's the guide in question.](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/)
- **Slim Chances on 8x7B Fine-Tuning with MLX**: `@mrdragonfox` expressed skepticism about the feasibility of fine-tuning **Mistral 8x7B** using MLX, hinting at potential technical challenges.
- **Potential MLX Fine-Tuning Resources Shared**: `@sublimatorniq` suggested looking into an [MLX example repository](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md) which could possibly help with the process.
- **Development in Progress for MLX**: `@cogbuji` indicated ongoing development efforts for MLX compatibility and provided a link to a resource for creating moe models using MLX. [Visit this GitHub repo for scripts and info.](https://github.com/mzbac/mlx-moe)
- **Clarifying Mistral 8x7B Variants**: `@notphysarum` enquired about the differences between different **Mistral 8x7B fine-tunes**, leading `@hammer_mt` to suggest that the variants are likely fine-tuned on specific familiar datasets.

**Links mentioned**:

- [A simple guide to local LLM fine-tuning on a Mac with MLX &#8211; Andy Peatling](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/): no description found
- [mlx-examples/llms/mlx_lm/LORA.md at main · ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md): Examples in the MLX framework. Contribute to ml-explore/mlx-examples development by creating an account on GitHub.
- [GitHub - mzbac/mlx-moe: Scripts to create your own moe models using mlx](https://github.com/mzbac/mlx-moe): Scripts to create your own moe models using mlx. Contribute to mzbac/mlx-moe development by creating an account on GitHub.
- [Enable the Mixtral-like Moe model without the quantized gate layer · Issue #394 · ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/issues/394): Currently, the community has started experimenting with building more models using a mix of different local experts. In the current implementation of mlx-lm, we have hardcoded the linear_class_pred...

  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1207326771943571526) (5 messages): 

- **NVIDIA RTX Powers Personal Chatbots**: `@ethux` shared a [link](https://blogs.nvidia.com/blog/chat-with-rtx-available-now/) about **Chat with RTX**, NVIDIA’s new offering allowing users to personalize a chatbot using an [NVIDIA GeForce RTX 30 Series GPU](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/) or higher. It includes a tech demo currently available for free [download](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/).
- **User Inquiry about NVIDIA's Chatbot Technology**: `@sublimatorniq` asked about the performance of **Chat with RTX**, but `@ethux` responded that they have not used it yet, comparing it to *Lmstudio* in terms of functionality.
- **The Leading German Chatbot on Mistral**: `@johannhartmann` introduced **Wiedervereinigung-7b-dpo**, the top-performing German chatbot on Mistral benchmarks, available on [Hugging Face](https://huggingface.co/mayflowergmbh/Wiedervereinigung-7b-dpo). The model is a merge of four German Mistral fine-tunes and includes dpo-training for improved result quality.

**Links mentioned**:

- [Chat with RTX Now Free to Download | NVIDIA Blog](https://blogs.nvidia.com/blog/chat-with-rtx-available-now/): New tech demo gives anyone with an NVIDIA RTX GPU the power of a personalized GPT chatbot, running locally on their Windows PC.
- [mayflowergmbh/Wiedervereinigung-7b-dpo · Hugging Face](https://huggingface.co/mayflowergmbh/Wiedervereinigung-7b-dpo): no description found

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1206886738346254346) (19 messages🔥): 

- **Internship Quest in France**: `@maeelk`, a French librarian, is reaching out to find an internship opportunity for a student studying psychology and AI. They've shared a [link to the master's program](https://formations.univ-smb.fr/fr/catalogue/master-XB/sciences-humaines-et-sociales-SHS/master-psychologie-KGYQCP1D/ergonomie-socio-cognitive-des-systemes-intelligents-classique-et-alternance-KIIPYUGG.html) and is asking if Mistral or any French-based company is willing to offer an internship.

- **Budget Limits for AI Projects**: `@akshay_1` discusses a client's underwhelming budget of $1,000 to build an S2S model with a persona using an audio dataset – a budget `@ethux` and `@mrdragonfox` find far too low for any significant work.

- **The Trouble with PDFs in Data Science**: Converting PDFs containing LaTeX to text for an LLM connection is a subject of discussion, and `@mrdragonfox` shares a [blog post](https://unstructured.io/blog/how-to-process-pdf-in-python) from Unstructured detailing the process and challenges of extracting data from PDFs.

- **Launch of a New Character AI Website**: `@ppprevost` announces the creation of a [character.ai like website](https://www.wearefanchat.com) using Langchain, Next.js, and the Mistral API. They invite members to try it and provide feedback, and share a [YouTube video](https://youtu.be/0tbyMuBrFU8?si=kJ2z5Z1A2M9Zg8ro) showcasing the site.

**Links mentioned**:

- [How to Process PDFs in Python: A Step-by-Step Guide – Unstructured](https://unstructured.io/blog/how-to-process-pdf-in-python): Unstructured effortlessly extracts and transforms complex data for use with every major vector database and LLM framework.
- [Ergonomie socio-cognitive des syst&egrave;mes intelligents - Classique et alternance - Ametys Campus - Universit&eacute; Savoie Mont Blanc](https://formations.univ-smb.fr/fr/catalogue/master-XB/sciences-humaines-et-sociales-SHS/master-psychologie-KGYQCP1D/ergonomie-socio-cognitive-des-systemes-intelligents-classique-et-alternance-KIIPYUGG.html): no description found
- [Fanchat - Create Your Mind-Blowing Mistral AI Characters](https://www.wearefanchat.com): no description found
- [Video screening wearefanchat.com](https://youtu.be/0tbyMuBrFU8?si=kJ2z5Z1A2M9Zg8ro): #langchain #mistral #react #ia

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1207316943988858942) (113 messages🔥🔥): 

- **Seeking GDPR Compliance Information**: `@.hutek` was inquiring about compliance details related to Mistral's APIs for client projects in France. `@dawn.dusk` provided a link to Mistral's data processing agreement ([Data Processing Agreement](https://mistral.ai/data-processing-agreement/)) which outlines how Mistral AI processes personal data under GDPR.

- **ChatGPT-like Testing with Mistral API**: `@_jackisjack` subscribed to the Mistral API and asked for guidance on setting up a simple ChatGPT-like dialogue without customization or development. `@fersingb` suggested using [Mistral's Python client library](https://github.com/mistralai/client-python) and specifically the `chatbot_with_streaming.py` example after setting the API key.

- **Streamlined Path to Chatbot Testing Discussed**: `@mrdragonfox` and `@fersingb` guided `@_jackisjack` through setting up a simple testing environment for a ChatGPT-like dialogue with Mistral and recommended using the open-source UI from [ETHUX Chat](https://chat.ethux.net/).

- **Payments and Access Concerns**: `@notphysarum` asked whether PayPal was an option for payment on Mistral as they lacked a credit card. `@lerela` responded that PayPal was not available, and the conversation shifted to potential platforms that support PayPal and provide access to Mistral's APIs.

- **Language Capabilities and Performance**: During discussions, `@mrdragonfox` mentioned that Mistral's API is trained in French as well as English and linked a user interface that allows testing the models ([ETHUX Chat](https://chat.ethux.net/)). They further commented on Mistral's performance, estimating that Mistral medium falls between GPT-3.5 and GPT-4 in terms of capabilities.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/): Find, download, and experiment with local LLMs
- [Guides | Mistral AI Large Language Models](https://docs.mistral.ai/guides/overview/): Welcome to our Mistral Getting Started guide! This guide is designed to help you learn how to use Mistral API quickly and easily. Learn how to write prompts for a variety of use cases, build a basic R...
- [Home](https://docs.librechat.ai/): 🪶 Introducing LibreChat
- [Data Processing Agreement](https://mistral.ai/data-processing-agreement/): Frontier AI in your hands
- [ETHUX Chat](https://chat.ethux.net/): Made possible by PlanetNode with ❤️
- [GitHub - mistralai/client-python: Python client library for Mistral AI platform](https://github.com/mistralai/client-python): Python client library for Mistral AI platform. Contribute to mistralai/client-python development by creating an account on GitHub.
- [GitHub - huggingface/chat-ui: Open source codebase powering the HuggingChat app](https://github.com/huggingface/chat-ui): Open source codebase powering the HuggingChat app. Contribute to huggingface/chat-ui development by creating an account on GitHub.
- [Mistral Medium: Quality, Performance &amp; Price Analysis | Artificial Analysis](https://artificialanalysis.ai/models/mistral-medium): Analysis of Mistral&#x27;s Mistral Medium across metrics including quality, latency, throughput tokens per second, price and others. API Host providers compared include Mistral.

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1206942941617061898) (427 messages🔥🔥🔥): 

- **LAION Explores Efficient Image Gen**: `@chad_in_the_house` experimented with image generation by applying `lfq 2^17` on ImageNet training and using the **Muse architecture** for further development. They consider `lfq` architectures swift to train and suggest they can be fine-tuned from existing `vqgans`.

- **Model Performance Real Talk**: Discussing the [Stable Cascade GitHub repository](https://github.com/Stability-AI/StableCascade) and its impact, `@pseudoterminalx` and others express skepticism around its performance and potential issues, such as face distortion and large VRAM requirements for inference.

- **OpenAI's Sora Surprises the Community**: The latest update, [Sora from OpenAI](https://openai.com/sora), a text-to-video model capable of producing minute-long videos, amazed many users. This includes a demonstration of its ability to simulate complex scenes and is expected to open up a wave of new creative opportunities.

- **Workshop Call for Low-Resource Languages at ICML 2024**: `@sttruong` invites interested parties to contribute to a workshop focused on low-resource languages. The topics cover data processing, LLM training, and social impacts, with a proposal deadline of February 15th.

- **Concerns about Sustainable Training Practices**: Amidst praise for Sora, `@pseudoterminalx` raises ethical questions about the reliance on Kenyan labor for content moderation and annotation, emphasizing a potential shadow over advancements in AI capabilities.

**Links mentioned**:

- [Court Dismisses Authors’ Copyright Infringement Claims Against OpenAI * TorrentFreak](https://torrentfreak.com/court-dismisses-authors-copyright-infringement-claims-against-openai-240213/): no description found
- [Computing Power and the Governance of Artificial Intelligence](https://arxiv.org/abs/2402.08797): Computing power, or &#34;compute,&#34; is crucial for the development and deployment of artificial intelligence (AI) capabilities. As a result, governments and companies have started to leverage compu...
- [ptx0/terminus-xl-gamma-v2-1 · Hugging Face](https://huggingface.co/ptx0/terminus-xl-gamma-v2-1): no description found
- [Stable Cascade - a Hugging Face Space by multimodalart](https://huggingface.co/spaces/multimodalart/stable-cascade): no description found
- [Hey Hindi GIF - Hey Hindi Bollywood - Discover &amp; Share GIFs](https://tenor.com/PzRY.gif): Click to view the GIF
- [Sora: Creating video from text](https://openai.com/sora): no description found
- [Imperium Of Man - Warhammer 40k](https://www.youtube.com/watch?v=sgM6Jj73cr8): Imperium Of Man - Warhammer 40k is a fan-made (unofficial) trailer by JustMovies, produced using various AI generative tools. What started as a project a few...
- [Crypto Kids Poster | 24posters | Hip Hop &amp; Street Art Prints](https://24posters.co/products/crypto-kids-6): Transform your walls with our viral new Crypto Kids Poster. Inspired by street-wear &amp; hip hop culture, enjoy artwork designed to bring you bedroom to life. Fast shipping times (3-5 days) 10,000+ h...
- [GitHub - Stability-AI/StableCascade](https://github.com/Stability-AI/StableCascade): Contribute to Stability-AI/StableCascade development by creating an account on GitHub.

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1206988937554296842) (41 messages🔥): 

- **Concern over OpenAI Model Degradation**: `.undeleted` expressed worries that OpenAI's safety tuning could degrade model quality to the point of being impractical for tasks. They commented, *"...become unreasonably expensive...already happened."*

- **Synthetic NSFW Content Shortage**: `@progamergov` mentioned the struggle to find high-quality synthetic NSFW content for datasets, criticizing Civitai for its messy outputs. 

- **Anime AI Development Stagnation Observed**: `@drhead` argued that the anime community's reliance on the NovelAI leaked model hindered progress, contrasting this with the furry community, who, due to a lack of analogous leaks, have advanced further in their respective model development.

- **RingAttention Enables Video-Linguistic Models**: `@spirit_from_germany` and `@max_voltage` discussed the potential of models using RingAttention for parsing large datasets, such as the combined video and book data, noting the technique's influence on long-sequence training.

- **Sora, OpenAI's Text-to-Video Model**: `@qwerty_qwer` shared a link introducing OpenAI's text-to-video model, Sora, which generates detailed scenes and movements based on provided text prompts. Discussion about its early release and the seeking of feedback was mentioned, along with skepticism from `@twoabove` regarding the closed nature of the model.

**Links mentioned**:

- [Large World Models](https://largeworldmodel.github.io/): no description found
- [Shinobi](https://shinobi8894.onrender.com/): no description found
- [Sora: Creating video from text](https://openai.com/sora): no description found
- [GitHub - insight-platform/Savant: Python Computer Vision &amp; Video Analytics Framework With Batteries Included](https://github.com/insight-platform/Savant): Python Computer Vision &amp; Video Analytics Framework With Batteries Included - insight-platform/Savant
- [Tweet from OpenAI (@OpenAI)](https://fxtwitter.com/openai/status/1758192957386342435): Introducing Sora, our text-to-video model.  Sora can create videos of up to 60 seconds featuring highly detailed scenes, complex camera motion, and multiple characters with vibrant emotions.          ...

  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1207064103118704671) (2 messages): 

- **Hugging News and Updates**: The 89th edition of Hugging News introduces a variety of developments including the Message API with OpenAI compatibility, community efforts for building open datasets, and new releases such as Datatrove, Gradio 4.18.0, Remove Background Web, Nanotron, and updates to Hugging Face Competitions and Accelerate. Additionally, the introduction of LoRA Studio, 2FA for the HF Hub, and a task page for Mask Generation are highlighted. [Read on Twitter](https://twitter.com/_philschmid/status/1755592500511997980).

- **Exciting Community Contributions**: The 45th edition of Community Highlights features prompt-to-automation demos, a specialized model for judging multiagent conversations called Prometheus, and the first version of the `tokviz` library for visualizing tokenization patterns. Innovations also include text-to-image and text-to-animation demos, art generation through Kandinsky-API, and datasets to train art generation models similar to midjourney / DALL-E-3. [Check Prometheus](https://huggingface.co/spaces/Tonic/prometheus).

- **Creative Community Spaces Unveiled**: Users continue to shine with unique spaces like a monocular depth estimation tool that converts RGB to depth, a quiz creator space named quizmona, and an on-device LLM for mobile machine learning dubbed Olmo-on-device. These creative tools expand the applications of AI in various fields and make them accessible to a broader audience.

- **Educational Opportunities and Tooling**: A partnership with Codecademy offers a free AI course on transformers, a blog post introduces SegMoE for model merging on text-to-image models, and Accelerate showcases faster loading of pre-trained PyTorch models. These resources aid users in learning about AI technologies and optimizing their implementation.

- **Upcoming Reading Group**: A reading group discussion is scheduled to cover the paper "Mamba: Content-Based Reasoning for Foundations Models," focusing on addressing the computational inefficiency of Transformers on long sequences. This indicates a community interest in advancing the understanding of foundational models and architectural improvements. [View the paper](https://arxiv.org/abs/2312.00752).

**Links mentioned**:

- [Prometheus - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/prometheus): no description found
- [GitHub - Mr-DG-Wick/tokviz: tokviz is a Python library for visualizing tokenization patterns across different language models.](https://github.com/Mr-DG-Wick/tokviz): tokviz is a Python library for visualizing tokenization patterns across different language models. - Mr-DG-Wick/tokviz
- [Proteus 0.3 - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Proteus-0.3): no description found
- [Pta Text V0.1 - a Hugging Face Space by AskUI](https://huggingface.co/spaces/AskUI/pta-text-v0.1): no description found
- [Text To Animation Fast AnimateDiff - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Fast-Text-to-animation): no description found
- [ehristoforu/midjourney-images · Datasets at Hugging Face](https://huggingface.co/datasets/ehristoforu/midjourney-images): no description found
- [ehristoforu/dalle-3-images · Datasets at Hugging Face](https://huggingface.co/datasets/ehristoforu/dalle-3-images): no description found
- [Olmo - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/Olmo): no description found
- [Spatial Media Converter](https://www.spatialmediaconverter.com/): Convert RGB Images to Spatial Photos for Apple Vision Pro.
- [Quiz Maker - a Hugging Face Space by narra-ai](https://huggingface.co/spaces/narra-ai/quizmona): no description found
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752): Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time a...
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/events/879548962464493619/1203285706949009448): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1205128865735770142): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Releases · gradio-app/gradio](https://github.com/gradio-app/gradio/releases): Build and share delightful machine learning apps, all in Python. 🌟 Star to support our work! - gradio-app/gradio
- [Tweet from Nouamane Tazi (@Nouamanetazi)](https://x.com/Nouamanetazi/status/1755607253087097207): Super happy to see https://github.com/huggingface/nanotron released today! ❤️  It&#39;s been a fun and insightful ride building a library for 3D parallelism training from scratch, and it&#39;s crazy t...
- [Tweet from Zach Mueller (@TheZachMueller)](https://x.com/TheZachMueller/status/1755993747232305468): Today is an extra-special release of @huggingface Accelerate!  Among other features, this latest version (with collaboration from @PyTorch) integrates a PyTorch-native pipeline-parallel inference fram...
- [Tweet from Omar Sanseviero (@osanseviero)](https://x.com/osanseviero/status/1757321643054006330): Over 300 models have been trained with axolotl and shared on the Hub! It&#39;s also the cutest icon ever.  https://huggingface.co/models?other=axolotl&sort=trending
- [Tweet from Sayak Paul (@RisingSayak)](https://x.com/RisingSayak/status/1755804957989539977): Why should LLM kids have all the fun from model merging? Why not us, the diffusion kids?   Friends from @_segmind open-sourced SegMoE to reduce this gap 🔥  Do MoE style merging on text-to-image model...
- [Tweet from Sayak Paul (@RisingSayak)](https://x.com/RisingSayak/status/1756634311493890559): 🤗 Accelerate power-user chronicles 👨‍🏫  Here, I show you how to load a pre-trained PyTorch model ~2x faster with Accelerate. The comments in the code snippet should be self-explanatory.   But if yo...

  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1206866578499440712) (227 messages🔥🔥): 

<ul>
<li><strong>Real-time snap detection inquiry</strong>: User <code>@butchbangher</code> inquired about a model or program to detect finger snaps in real-time video and audio. Having tried MediaPipe without finding included support for this gesture, they were searching for guidance on how to approach temporal detection.</li>
<li><strong>HF Spaces token issues</strong>: Users <code>@hari4626</code> and <code>@thatonecoder20</code> discussed a missing HF_Token field, necessary for running spaces, which might require manual inclusion in settings.</li>
<li><strong>Blog post reaches audience</strong>: <code>@not_lain</code> celebrated that their blog post about custom architectures with Hugging Face has reached 240 readers, sharing a link to the post and a snippet of code for baseline model creation.</li>
<li><strong>AI Career Hopes</strong>: <code>@00face</code> discussed the difficulty in debunking a misconception about Mistral and all LLMs containing stolen data and was looking for white papers or hard data to refute such claims.</li>
<li><strong>Introducing Gemini 1.5 Pro</strong>: In the field of generative models, users <code>@pierrunoyt</code>, <code>@danfosing</code>, and <code>@skyward2989</code> discussed the latest announcement from Google's Gemini 1.5, noting its enhanced performance and breakthroughs in long-context understanding.</li>
</ul>

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1206246780950544405): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Boximator: Generating Rich and Controllable Motions for Video Synthesis](https://boximator.github.io/): no description found
- [Stable Cascade - a Hugging Face Space by multimodalart](https://huggingface.co/spaces/multimodalart/stable-cascade): no description found
- [lamm-mit/x-lora · Hugging Face](https://huggingface.co/lamm-mit/x-lora): no description found
- [Custom architectures with HuggingFace 🤗](https://huggingface.co/blog/not-lain/custom-architectures-with-huggingface): no description found
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending): no description found
- [NVIDIA Chat With RTX](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/): Your Personalized AI Chatbot.
- [Our next-generation model: Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/?utm_source=yt&utm_medium=social&utm_campaign=gemini24&utm_content=&utm_term=): Gemini 1.5 delivers dramatically enhanced performance, with a breakthrough in long\u002Dcontext understanding across modalities.
- [Sora: Creating video from text](https://openai.com/sora): no description found
- [mlx-community (MLX Community)](https://huggingface.co/mlx-community): no description found
- [How to add a model to 🤗 Transformers?](https://huggingface.co/docs/transformers/en/add_new_model): no description found
- [BridgeAI Programme - Brief - Digital Catapult FutureScope](https://futurescope.digicatapult.org.uk/our-programmes/bridgeai-programme/bridgeai-programme-brief/?utm_source=Website&utm_medium=IUK+KTN&utm_campaign=KTN&utm_id=IUKKTN&utm_content=IUKKTNWebsitebrief#section-2): Digital Catapult is launching an accelerator programme; Innovate UK BridgeAI, that seeks to stimulate the adoption of artificial intelligence and machine learning technologies in agriculture, creative...
- [Tweet from Greg Brockman (@gdb)](https://fxtwitter.com/gdb/status/1758193811489243408?s=20): Announcing Sora — our model which creates minute-long videos from a text prompt: https://openai.com/sora
- [VanceAI Photo Restorer | AI Old Photo Restoration Online Solution](https://vanceai.com/old-photo-restoration/): no description found
- [GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production](https://github.com/coqui-ai/TTS): 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production - coqui-ai/TTS
- [Lykon/dreamshaper-8 · Discussions](https://huggingface.co/Lykon/dreamshaper-8/discussions): no description found
- [Deep Papers Episode 3 - Toolformer: Training LLMs To Use Tools](https://youtu.be/pSKHDduKt_g): Deep Papers is a podcast series featuring deep dives on today’s seminal AI papers and research. Hosted by AI Pub creator Brian Burns and Arize AI founders Ja...
- [How to update a label while a method function call is running · Issue #7419 · gradio-app/gradio](https://github.com/gradio-app/gradio/issues/7419): Asking GPT-4 but all answers it gives me is wrong. This drives me crazy but cant be helped. What I want is so simple. I want to update a gr.label or whatever that is possible while a function is ru...

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1206910023985274900) (11 messages🔥): 

```html
<ul>
  <li><strong>Merging Sheets with Caution</strong>: `@lunarflu` discussed the challenges of merging two Google Sheets, emphasizing the need to avoid duplicate records and maintain unique keys. They highlighted the importance of creating distinct records to prevent data issues.</li>
  <li><strong>A Melody of Learning</strong>: `@neuralink` expressed their progress in learning about DoReMi reproduction and training with FP8 3D parallelism, achieving a remarkable 99% and 32% respectively.</li>
  <li><strong>End-to-End Learning Spree</strong>: `@sardarkhan_` engaged in a deep dive into diffusors and transformers before switching gears back to rigorous coursework preparation.</li>
  <li><strong>Face Swapping Exploration</strong>: `@virtual_josh` shared their experience exploring different programs for deep faking videos and asked for recommendations on services for swapping faces in videos.</li>
  <li><strong>Custom Labels in NER</strong>: `@jakemorrison` inquired about the flexibility of `ner_tags` labels in token classification, sparking a discussion where `@cubietom` pointed to custom label usage with references to the CoNLL2003 and Few-NERD datasets.</li>
</ul>
```


**Links mentioned**:

- [few-nerd.py · DFKI-SLT/few-nerd at main](https://huggingface.co/datasets/DFKI-SLT/few-nerd/blob/main/few-nerd.py#L48): no description found
- [conll2003 · Datasets at Hugging Face](https://huggingface.co/datasets/conll2003#conll2003-1): no description found

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1206992488791801916) (12 messages🔥): 

- **MoE Models Under Threat**: `@osanseviero` highlighted a paper revealing vulnerabilities in **Mixture of Experts (MoE)** models that could allow attackers to influence the output of other users' queries within the same batch. The paper was discussed, including potential mitigation strategies on [HuggingFace](https://huggingface.co/papers/2402.05526) and further personal insights were shared in a [blog post](https://huggingface.co/posts/osanseviero/980907000007376).

- **Concerns Over Potential MoE Risks**: `@meatfucker` pointed out that the threat from the MoE vulnerability is not immediate, but could pose problems in the future if left unaddressed. The user also mentioned the potential of incidental negative impacts on output quality in systems using large batches.

- **Million-Length Video and Language Processing**: `@not_lain` shared excitement about a new DeepMind project, which includes open-source 7B models capable of deciphering long text and video data over one million tokens. More information and resources are available through the [largeworldmodel project](https://largeworldmodel.github.io/) and [arXiv abstract](https://arxiv.org/abs/2402.08268).

- **Online and Offline RL Blended**: `@poudelbibek` brought attention to a paper discussing **Online Decision Transformers (ODT)**, a novel reinforcement learning algorithm that unifies offline pretraining with online finetuning. The paper can be found on [arXiv](https://arxiv.org/abs/2202.05607).

- **Introducing SPIN for Realistic Model Reactions**: `@andysingal` posted about SPIN, a new method enabling Language Learning Models (LLMs) to produce reactions indistinguishable from human responses, enhancing self-play capabilities without the need for higher-level annotators. The method details can be checked out on [GitHub](https://github.com/andysingal/llmcourse/blob/main/llama_finetune/SPIN.md).

**Links mentioned**:

- [@osanseviero on Hugging Face: &quot;Mixture of experts: beware 🛡️⚔️

New paper by DeepMind: Buffer Overflow in…&quot;](https://huggingface.co/posts/osanseviero/980907000007376): no description found
- [Paper page - Buffer Overflow in Mixture of Experts](https://huggingface.co/papers/2402.05526): no description found
- [Online Decision Transformer](https://arxiv.org/abs/2202.05607): Recent work has shown that offline reinforcement learning (RL) can be formulated as a sequence modeling problem (Chen et al., 2021; Janner et al., 2021) and solved via approaches similar to large-scal...
- [Tweet from Aran Komatsuzaki (@arankomatsuzaki)](https://fxtwitter.com/arankomatsuzaki/status/1757596665295368534?s=20): World Model on Million-Length Video And Language With RingAttention  Open-sources 7B models capable of processing long text documents and videos of over 1M tokens  proj: https://largeworldmodel.github...

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1207160436416454749) (25 messages🔥): 

- **RAG Application Ready to Share**: `@osiworx` created a RAG-based application using nearly 1 million text2image prompts and inquires about the possibility of running it on HuggingFace, seeking advice on datastore management.

- **Free Hosting for LLMs on Colab**: `@typoilu` introduced **LocalLlm**, a solution for hosting large language models for free on Colab or locally, inviting the community to try it and provide feedback on the early version of the repository. [LocalLlm on GitHub](https://github.com/groloch/LocalLlm/).

- **Visualize Tokenization Patterns with tokviz**: `@deeeps.ig` announced the first release of **tokviz** on PyPI, a library for visualizing how various language models from the Hugging Face library tokenize text, and shared the [documentation](https://github.com/Mr-DG-Wick/tokviz).

- **PTA-Text to Locate UI Text**: `@calmdown.manu` showcased the PTA-Text model, designed to process UI screenshots and click commands, and shared a [demo](https://huggingface.co/spaces/AskUI/pta-text-v0.1) and the [model checkpoint](https://huggingface.co/AskUI/pta-text-0.1).

- **Trinity and Neo Models Now Available**: `@tonic_1` highlighted the introduction of **Trinity** from Rabbit, a coding model believed to be a deepseek branch, which is now available on the Hugging Face Spaces; mentioned **Neo** as a partner to Trinity capable of fitting 33B parameters on an A10G. [Trinity on HuggingFace](https://huggingface.co/spaces/Tonic/trinity), [Neo on HuggingFace](https://huggingface.co/spaces/Tonic/neo).

**Links mentioned**:

- [Pta Text V0.1 - a Hugging Face Space by AskUI](https://huggingface.co/spaces/AskUI/pta-text-v0.1): no description found
- [Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies](https://arxiv.org/abs/2208.10264): We introduce a new type of test, called a Turing Experiment (TE), for evaluating to what extent a given language model, such as GPT models, can simulate different aspects of human behavior. A TE can a...
- [Kandinsky API - a Hugging Face Space by ehristoforu](https://huggingface.co/spaces/ehristoforu/Kandinsky-API): no description found
- [Neo - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/neo): no description found
- [Proteus 0.3 - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Proteus-0.3): no description found
- [Trinity - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/trinity): no description found
- [blog-explorers (Blog-explorers)](https://huggingface.co/blog-explorers): no description found
- [AskUI/pta-text-0.1 · Hugging Face](https://huggingface.co/AskUI/pta-text-0.1): no description found
- [GitHub - groloch/LocalLlm: A drop-in solution to create chat interfaces with open-source models !](https://github.com/groloch/LocalLlm/): A drop-in solution to create chat interfaces with open-source models ! - groloch/LocalLlm
- [GitHub - meta-introspector/lang-agent-streamlit-ui: The streamlit ui for lang-agent](https://github.com/meta-introspector/lang-agent-streamlit-ui): The streamlit ui for lang-agent. Contribute to meta-introspector/lang-agent-streamlit-ui development by creating an account on GitHub.
- [tokviz](https://pypi.org/project/tokviz/): Library for visualizing tokenization patterns across different language models
- [GitHub - Mr-DG-Wick/tokviz: tokviz is a Python library for visualizing tokenization patterns across different language models.](https://github.com/Mr-DG-Wick/tokviz): tokviz is a Python library for visualizing tokenization patterns across different language models. - Mr-DG-Wick/tokviz

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1206995110550577232) (40 messages🔥): 

- **LangTest Paper Published**: `@prikfy` announced the publication of their **LangTest** paper in the Software Impacts journal, a library for testing LLM and NLP models, including a method to augment training datasets based on test outcomes. The paper can be accessed [here](https://www.sciencedirect.com/science/article/pii/S2665963824000071), and a GitHub repository and website for LangTest were highlighted by `@ryzxl`.

- **Model Merging Presentation on the Horizon**: `@prateeky2806` offered to present ideas on **model merging** in the upcoming reading group session on March 1st. `@lunarflu` suggested that the presentation include diagrams and potentially a demonstration using a notebook or **Gradio**.

- **Mamba Paper Inquiry Answered**: Questions regarding the **Mamba paper** were addressed with an arXiv link provided by `@chad_in_the_house`, and `@ericauld` mentioned discussing variations of the work and entry points for new variations.

- **Secrets of Seed Selection Explored**: `@stereoplegic` inquired about papers where random seeds are learnable parameters, evoking a discussion on gradient-based optimization and data augmentation policies with references to the AutoAugment paper by `@chad_in_the_house`.

- **Search for Seed-Related Works**: Dialogue on the impact of random seed selection on model performance was initiated after `@stereoplegic` did not find much existing literature. They detailed their approach involving the use of random seeds in model initializations, with `@chad_in_the_house` providing references and engaging in discussion on the potential of the concept.

**Links mentioned**:

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752): Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time a...
- [Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision](https://arxiv.org/abs/2109.08203): In this paper I investigate the effect of random seed selection on the accuracy when using popular deep learning architectures for computer vision. I scan a large amount of seeds (up to $10^4$) on CIF...
- [Generating images of rare concepts using pre-trained diffusion models](https://arxiv.org/abs/2304.14530): Text-to-image diffusion models can synthesize high-quality images, but they have various limitations. Here we highlight a common failure mode of these models, namely, generating uncommon concepts and ...
- [GitHub - JohnSnowLabs/langtest: Deliver safe &amp; effective language models](https://github.com/JohnSnowLabs/langtest): Deliver safe &amp; effective language models. Contribute to JohnSnowLabs/langtest development by creating an account on GitHub.
- [LangTest | Deliver Safe & Effective Models | John Snow Labs](https://langtest.org): no description found
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501): Data augmentation is an effective technique for improving the accuracy of modern image classifiers. However, current data augmentation implementations are manually designed. In this paper, we describe...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1206977701970186300) (17 messages🔥): 

- **Turning Text to Images**: `@isidentical` reported achieving a **50% success rate** on generating images from arbitrary words using Stable Cascade, after figuring out a good prompting strategy.
- **Stable Diffusion Discussions**: `@chad_in_the_house` mentioned that the HuggingFace diffusion chat might be better discussed in a different channel, but acknowledged that text generation with models like Stable Cascade is quite effective.
- **SageMaker Setup Snag**: `@nayeem0094` encountered a problem where the disk space was insufficient for the expected model file size while setting up a `HuggingFaceModel` on SageMaker and asked for assistance.
- **Serverless API Generation Query**: `@vrushti24` inquired about generating multiple images using a serverless API with the `Lykon/dreamshaper-8` model, which currently generates only one image from text.
- **Vanishing Gradient Puzzle**: `@maxpappa` reached out to discuss experiences of vanishing gradient issues while fine-tuning models or using DiffusionDPO pipelines, later clarifying that they are using fp32 instead of **fp16** in response to `@pseudoterminalx`.

**Links mentioned**:

- [Lykon/dreamshaper-8 · Discussions](https://huggingface.co/Lykon/dreamshaper-8/discussions): no description found
- [Models - Hugging Face](https://huggingface.co/models): no description found

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1206903277564993586) (8 messages🔥): 

- **Gaussian Splat Expertise Suggestion**: `@johko990` recommended that someone looking for help with gaussian splats should ask in another channel where experts on the topic are likely active.
- **Multimodal Queries and Collaboration Offer**: `@joee2711` is working on a multimodal project involving Q-formers and MLP connectors; queries about their differences and similarity to adapters. The user is also seeking collaboration.
- **Seeking Image Retrieval System Improvements**: `@femiloye` is developing an image retrieval system based on custom DeiT transformers trained with reid loss and seeks advice for enhancing retrieval accuracy beyond just model embeddings.
- **Hairstyle Transformation Research Assistance**: `@abrahamowodunni` requested resources for changing hairstyles with a generative vision model, which `@lunarflu` suggested might relate to another user's fashion demo project.
- **New Project Spotlight - PTA-Text Model**: `@calmdown.manu` shared a project about a lightweight multimodal model, PTA-Text, designed for UI interaction using screenshots and text commands, inviting feedback and highlighting current limitations related to training data and functionality.

**Links mentioned**:

- [Pta Text V0.1 - a Hugging Face Space by AskUI](https://huggingface.co/spaces/AskUI/pta-text-v0.1): no description found
- [AskUI/pta-text-0.1 · Hugging Face](https://huggingface.co/AskUI/pta-text-0.1): no description found

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1206899152236183583) (6 messages): 

- **Seeking Language Extraction from XLM-R**: `@_michaelsh` is looking for guidance on how to extract the language from an **XLM-RoBERTa** model, linking to the [HuggingFace documentation](https://huggingface.co/docs/transformers/model_doc/xlm-roberta), but has not received a response yet.

- **Curiosity for Algebraic Translation**: `@_david_valente_` inquires about any existing work that translates natural language into algebraic representations like **LEAN**, but no answers have been provided in the discussion.

- **Voice Simulation and Language Change with Transformers**: `@mentrass` asked about simulating their voice and changing the language with transformers. `@mahimairaja` recommended **XTTS**, a real-time voice cloning tool, and provided a [link to the model](https://huggingface.co/coqui/XTTS-v2) that supports 17 languages and is used in Coqui Studio and API.

- **Introducing a Text-Only Click Model**: `@calmdown.manu` shared a project named **PTA-Text**, which is a text-only click model designed for UI interactions. They provided both a [demo](https://huggingface.co/spaces/AskUI/pta-text-v0.1) and a [model checkpoint](https://huggingface.co/AskUI/pta-text-0.1), noting that it's designed for 1920x1080 screenshots and is still in the prototype stage.

**Links mentioned**:

- [Pta Text V0.1 - a Hugging Face Space by AskUI](https://huggingface.co/spaces/AskUI/pta-text-v0.1): no description found
- [AskUI/pta-text-0.1 · Hugging Face](https://huggingface.co/AskUI/pta-text-0.1): no description found
- [coqui/XTTS-v2 · Hugging Face](https://huggingface.co/coqui/XTTS-v2): no description found
- [XTTS - a Hugging Face Space by coqui](https://huggingface.co/spaces/coqui/xtts): no description found

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1206977701970186300) (17 messages🔥): 

- **Successful Text Generation with Stable Cascade**: `@isidentical` reported achieving a **50% success rate** on arbitrary word text generation using a good prompting strategy with **Stable Cascade**, mentioning this in the context of the model's performance in README examples.
- **Inference Engine Mention by Huggingface**: `@chad_in_the_house` briefly noted that Huggingface has made an inference engine for large language models, though the specific link was not provided.
- **Deploying Models on SageMaker**: `@nayeem0094` faced issues deploying a HuggingFace Model on SageMaker due to insufficient disk space, with an error message indicating a lack of available space for the expected file size (3892.53 MB).
- **Serverless API Query for Dreamshaper-8**: `@vrushti24` inquired about the possibility of generating multiple images from a single text prompt using a serverless API for the **Lykon/dreamshaper-8** model, asking for advice within the HuggingFace community.
- **Vanishing Gradient Issue in Fine-Tuning**: `@maxpappa` sought advice for an issue with vanishing gradients when fine-tuning a model or using DPO with the **DiffusionDPO** pipeline, clarified later that he was using fp32 training, not fp16.

**Links mentioned**:

- [Lykon/dreamshaper-8 · Discussions](https://huggingface.co/Lykon/dreamshaper-8/discussions): no description found
- [Models - Hugging Face](https://huggingface.co/models): no description found

  

---



### Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1207424713081622618) (1 messages): 

- **Perplexity Push Spices up Slack with Subscriptions**: User `@ok.alex` announced the upcoming feature **Perplexity Push**, allowing users to **subscribe to topics** and receive updates directly in **Slack channels**. This feature promises to enhance team discussions and keep everyone in the loop.
  

---


### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1206881234823811072) (256 messages🔥🔥): 

- **Referral Program and Coupon Usage Explained**: User `@mares1317` provided details on how to apply a coupon to a Perplexity account, referencing an [FAQ on coupons and discounts](https://blog.perplexity.ai/faq/coupons-and-discounts) and instructing users to go to [perplexity.ai/pro](https://perplexity.ai/pro) to redeem coupons.
  
- **Perplexity API Integration and pplx Models Highlighted**: `@mares1317` shared a [link](https://blog.perplexity.ai/blog/introducing-pplx-online-llms) explaining the new `pplx-7b-online` and `pplx-70b-online` models, emphasizing their help in delivering up-to-date and factual responses via API and Perplexity Labs.

- **Discussion and Speculation on pplx-8x7b Model**: Discussions took place around the nature and capabilities of the `pplx-8x7b` model. While there was no definitive documentation provided, users like `@akumaenjeru` and `@jake` speculated that it's likely related to existing models such as `mixtral-8x7b-instruct` or a fine-tune of the `mixtrial` model.

- **Perplexity Service Availability Concerns**: Multiple users like `@diego.tech`, `@lucassmith56_38679`, and `@luke_____________` reported issues with Perplexity's service availability, citing timeout errors and problems with model responses.

- **Gemini 1.5 Announcement Catches Attention**: `@luke_____________` highlighted [blog post](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/amp/) updates about Google's AI technology, Gemini 1.5, discussing its potential and anticipated features like a one million-token context window and faster release cycles compared to past models.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649527299055688/1197892547276705843): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649527299055688/1207087142472519701): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047204950763122820): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1118264005207793674/1206743956302471168): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [‎What Gemini Apps can do and other frequently asked questions](https://gemini.google.com/faq?gad_source=1&gclid=Cj0KCQiAw6yuBhDrARIsACf94RXwwoXpktZalDwo6OO8RsVYvKAaDpxT1Cr_XIek-8kBnPaZa7Jb5bwaAvsQEALw_wcB): Learn what Gemini can do, how it works, and different ways to get access to it.
- [Our next-generation model: Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/amp/): Gemini 1.5 delivers dramatically enhanced performance, with a breakthrough in long\u002Dcontext understanding across modalities.
- [More than an OpenAI Wrapper: Perplexity Pivots to Open Source](https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/): Perplexity CEO Aravind Srinivas is a big Larry Page fan. However, he thinks he&#039;s found a way to compete not only with Google search, but with OpenAI&#039;s GPT too.
- [Introducing PPLX Online LLMs ](https://blog.perplexity.ai/blog/introducing-pplx-online-llms): The first-of-its-kind Online LLM API
- [OpenAI Develops Web Search Product in Challenge to Google](https://www.theinformation.com/articles/openai-develops-web-search-product-in-challenge-to-google): OpenAI has been developing a web search product that would bring the Microsoft-backed startup into more direct competition with Google, according to someone with knowledge of OpenAI’s plans. The searc...
- [Perplexity Blog](https://blog.perplexity.ai/technical-faq/what-is-a-token-and-how-many-tokens-c): Explore Perplexity&#39;s blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Coupons and discounts](https://blog.perplexity.ai/faq/coupons-and-discounts): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Introducing pplx-api ](https://blog.perplexity.ai/blog/introducing-pplx-api): Perplexity Lab's fast and efficient API for open-source LLMs
- [What are Threads?](https://blog.perplexity.ai/faq/what-are-threads): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [What is a token and how many tokens can Perplexity read at once?](https://blog.perplexity.ai/technical-faq/what-is-a-token-and-how-many-tokens-can-perplexity-read-at-once): Dive deep into Perplexity's technical details with our comprehensive FAQ page. From the nuances of AI models like GPT-4 and Claude 2 to token limits and AI profiles, get concise answers to optimize yo...
- [What is Perplexity Pro?](https://blog.perplexity.ai/faq/what-is-perplexity-pro): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1207062946623135824) (22 messages🔥): 

- **Perplexity AI Search Spotlight**: User `@idesign12` shared a Perplexity AI search link, seemingly to showcase the search capabilities of the platform, but no specific details were provided.
- **Introducing Alt-D Feed for Community Curation**: `@ok.alex` shared a [link to Alt-D Feed](https://www.perplexity.ai/collections/Alt-D-Feed-x.dZp0_3RAyKoTWkhJW_DA), an alternative feed/newsletter for community collaboration. They encouraged likes and shares for those interested.
- **Bookmarking on Perplexity Discussed**: `@jaybob32` inquired about bookmarking collections, to which `@ok.alex` replied that bookmarking in-browser is the current solution as collections aren't saved to user libraries unless they are contributors. However, suggestions for improvements are being considered.
- **Perplexity AI and GitHub Repo Announced**: `@_kokomos` described how they are integrating structured data and logic patterns with Perplexity AI, while also providing a [link to their non-live GitHub repository](https://github.com/thecommonsai/kokomos) which will become public in the future.
- **DIY Tutorial Triumph**: User `@duplex0150` found success in layering their hair by following a tutorial from Perplexity AI and shared their [positive experience, with future plans to cut at angles](https://www.perplexity.ai/search/how-do-I-cViLV9ExT3icLzN6kBxIjw?s=m).

**Links mentioned**:

- [Tweet from Aravind Srinivas (@AravSrinivas)](https://fxtwitter.com/AravSrinivas/status/1757823982177563093): Thanks for the great support and encouraging responses. As the future of getting your information online changes to directly asking questions, new modes of user behavior like getting updates in the fo...
- [He is taking on Google 😮😮 #startup #ai #perplexity](https://youtube.com/shorts/uNT2FbISbyk?si=mNZF2E_17HM1mQBA): This IITian is taking on Google, with his AI startup Perplexity. Aravind Srinivas studied engineering at IIT Madras before getting a PhD in computer science ...
- [Perplexity’s CEO on Its Plan to Displace Google Search With AI Answers — With Aravind Srinivas](https://open.spotify.com/episode/6iw9jSIr6Jcd8xYXAew3AZ?si=S2p0ym2UTe2m7QySo_VbVg): Listen to this episode from Big Technology Podcast on Spotify. Aravind Srinivas is the CEO of Perplexity, an AI-powered search engine that answers queries with a few paragraphs in natural language. Sr...
- [GitHub - thecommonsai/kokomos: TOBE Production Repo for kokomos](https://github.com/thecommonsai/kokomos): TOBE Production Repo for kokomos. Contribute to thecommonsai/kokomos development by creating an account on GitHub.

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1206872009557151774) (60 messages🔥🔥): 

- **API Intermittent Failures Discussed**: `@myadmingushwork_52332` reported that both `pplx-7b-online` and `pplx-70b-online` were returning random and absurd replies, providing a snippet of the problematic output. No specific resolution was mentioned in the subsequent conversation.
- **Inconsistencies in API Responses Troublesome**: `@ia7df` expressed concerns about the inconsistency in API responses compared to `perplexity.ai` and sought developer assistance. `@icelavaman` clarified that `perplexity.ai` and `pplx-api` are different and require different prompts; however, issues of consistency across prompts were not resolved.
- **LangChain and Perplexity Compatibility Clarified**: `@icelavaman` shared a helpful [guide](https://mochan.org/posts/perplexity-ai-langchain/) by Mochan for using Perplexity AI with LangChain, addressing `@ponomoly_dev`'s issues with substituting `pplx-7b-chat` for `gpt-3.5-turbo`.
- **New Model Availability Uncertain**: `@paul16307` inquired about the availability and pricing of `PPLX-8x7B` on the API, and while users like `@brknclock1215` indicated it might already be functional, `@icelavaman` stated there is no ETA for official release or pricing information.
- **Discrepancy in API Model Performance Observed**: `@xlhu_69745` reiterates that they're getting random results from `pplx-70b-online` and recognizes that the model sometimes hallucinates, citing examples where the provided links do not exist.

**Links mentioned**:

- [Adding Perplexity.ai API support to Langchain | Mochan.org | Mochan Shrestha](https://mochan.org/posts/perplexity-ai-langchain/): no description found
- [no title found](https://airbyte.com/how-to-sync/square-to-local-json): no description found

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1207015086686740480) (1 messages): 

- **No-Code RAG Building with FlowiseAI**: `@jerryjliu0` announces an upcoming **webinar on how to build no-code RAG** with Henry Heng from FlowiseAI. The event is scheduled for Friday at 9 am PT, aiming to teach users about leveraging the LlamaIndex.TS + Flowise integration to develop LLM-powered workflows without coding. [Register for the webinar here](https://lu.ma/ubm3jg3k).

**Links mentioned**:

[LlamaIndex Webinar: Build No-Code RAG · Zoom · Luma](https://lu.ma/ubm3jg3k): Flowise is one of the leading no-code tools for building LLM-powered workflows. Instead of learning how to code in a framework / programming language, users can drag and drop the components...

  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1207012470367981648) (7 messages): 

- **DanswerAI Integrates LlamaIndex**: *@llama_index* acknowledged the integration of **DanswerAI**, a ChatGPT tool enhancing efficiency across workplace tools, backed by LlamaIndex technology. Check out their [full announcement here](https://twitter.com/llama_index/status/1757453320829251755).
- **No-Code RAG Webinar**: *@FlowiseAI*, known for building no-code LLM workflows, joins @llama_index for a webinar featuring *@henryhengzj*. They will discuss LlamaIndex.TS and [Flowise integration](https://twitter.com/llama_index/status/1757455162988540329).
- **Scientific Research Workflow Tutorial**: A new notebook by *@quantoceanli* details constructing an agent to perform scientific research, including fetching abstracts from ArXiv. The workflow aims to simplify the process for researchers, [shared by LlamaIndex](https://twitter.com/llama_index/status/1757579982879260891).
- **Tutorial on Building Custom Agentic Workflows**: LlamaIndex released a [video tutorial](https://twitter.com/llama_index/status/1757810147257389132) to empower AI engineers in creating their own agents from scratch, demonstrating that it is not just for AI researchers.
- **Technology Showcase for ADU Planning GenAI App**: Celebrating their hackathon's first-place winner, ADU Planner, LlamaIndex highlighted the AI app's multifaceted capabilities, from parsing ADU local regulations to floor plan suggestions, [available here](https://twitter.com/llama_index/status/1758207209140601315).
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1206881863298318337) (285 messages🔥🔥): 

- **Arize-Phoenix Tracing Feature Update Coming Soon**: User `@richard1861` inquired about tracing user queries with Arize-Phoenix. `@cheesyfishes` confirmed that tagging traces with metadata is in progress and should be ready in the next week or so.

- **Custom Metadata Tagging in Queries**: User `@akash_18327` sought advice on excluding metadata from the context in their custom QA template. `@cheesyfishes` suggested setting excluded metadata keys before data ingestion with `document.excluded_llm_metadata_keys = ["field1", ...]`.

- **Trouble Reading DOCX in v0.10**: `@.mai_` reported issues with SimpleDirectoryReader interpreting DOCX files as encoded data. The issue was resolved after updating to the latest `llama-index-core` version alongside `llama-index`.

- **Real-time RAG Pipeline Optimization**: `@barrahh` asked about optimizing the RAG pipeline using user feedback or ratings. `@lemuffinman` mentioned the possibility of using reranking based on score, while `@cheesyfishes` provided a code example to separate the retrieval and synthesis steps for real-time evaluation.

- **Discord Server for Peer Support**: `@ryanrib14` shared a [Discord invite link](https://discord.gg/55mzvBnS) to join a community server aimed at helping with integration issues related to LlamaIndex and Azure AI search vector bank and sharing experiences.

**Links mentioned**:

- [Excalidraw — Collaborative whiteboarding made easy](https://excalidraw.com/): Excalidraw is a virtual collaborative whiteboard tool that lets you easily sketch diagrams that have a hand-drawn feel to them.
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://pretty-sodium-5e0.notion.site/ce81b247649a44e4b6b35dfb24af28a6?v=53b3c2ced7bb4c9996b81b83c9f01139): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
- [Cards Page](https://www.nerdai.io/cards): no description found
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://pretty-sodium-5e0.notion.site/v0-10-0-Migration-Guide-6ede431dcb8841b09ea171e7f133bd77): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
- [Join multiple async generators in Python](https://stackoverflow.com/questions/55299564/join-multiple-async-generators-in-python/55317623#55317623): I would like to listen for events from multiple instances of the same object and then merge this event streams to one stream. For example, if I use async generators:&#xA;&#xA;class PeriodicYielder: &#...
- [Join the azureaisearch Discord Server!](https://discord.gg/55mzvBnS): Check out the azureaisearch community on Discord - hang out with 2 other members and enjoy free voice and text chat.
- [Observability - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/module_guides/observability/observability.html): no description found
- [llama_index/llama-index-core/llama_index/core/node_parser/text/sentence.py at a4184f47626c6957f40f5b2732de9344e26d2a01 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/a4184f47626c6957f40f5b2732de9344e26d2a01/llama-index-core/llama_index/core/node_parser/text/sentence.py#L65): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [How to convert Decimal to Double in C#?](https://stackoverflow.com/questions/5): I want to assign the decimal variable &amp;quot;trans&amp;quot; to the double variable &amp;quot;this.Opacity&amp;quot;.&#xA;decimal trans = trackBar1.Value / 5000;&#xA;this.Opacity = trans;&#xA;&#xA;...
- [Faithfulness Evaluator - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/examples/evaluation/faithfulness_eval.html): no description found
- [llama_index/llama-index-core/llama_index/core/chat_engine/condense_question.py at 3823389e3f91cab47b72e2cc2814826db9f98e32 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/3823389e3f91cab47b72e2cc2814826db9f98e32/llama-index-core/llama_index/core/chat_engine/condense_question.py#L177): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [no title found](http://<host>:<port>"): no description found
- [Qdrant Vector Store - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo.html): no description found
- [llama_index/llama-index-core/llama_index/core/base/base_query_engine.py at 448584c8cd30bab744d7629c9d1a7ee72e5af5ad · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/448584c8cd30bab744d7629c9d1a7ee72e5af5ad/llama-index-core/llama_index/core/base/base_query_engine.py#L37): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/llms/callbacks.py at fc51b9fcc9c2bbdc09a9ac91deea7715872c3f44 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/fc51b9fcc9c2bbdc09a9ac91deea7715872c3f44/llama-index-core/llama_index/core/llms/callbacks.py#L24): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-legacy/llama_index/legacy/vector_stores/mongodb.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama-index-legacy/llama_index/legacy/vector_stores/mongodb.py#L160-L183): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [Available LLM integrations - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/modules.html#bedrock): no description found
- [AsyncCallback Iterator To Stream Internal Events by jordanparker6 · Pull Request #9164 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/9164): Description Llama Index should have an out-of-the-box callback that allows the user to stream all internal events through an async iterator. The current streaming misses all the internal events lik...

  

---



### LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1207172343600783380) (10 messages🔥): 

- **LangChain Introduces a Journaling App with Memory**: `@hwchase17` announced an early incarnation of a journaling app that uses LangChain's memory module, intended to remember user information for future interactions. The application is in a very early stage and feedback is welcome; see the app in action via [Loom video](https://www.loom.com/share/63a9696036c74765a9f9ecae06336aa5) and try it out at [Journal by LangChain](https://journal.langchain.com/).

- **User Feedback on Login Methods**: `@rpall_67097` suggested incorporating social logins like Google, GitHub, or Twitter for the Journal app, mentioning the barrier that traditional email/password sign-ups may create for potential users.

- **LangSmith Now Generally Available & Fundraised Series A**: `@hwchase17` shared the news of LangSmith's general availability, a $25M fundraise led by Sequoia Capital, and introduced their redesigned homepage and brand with excitement. Learn more on their [blog post](https://blog.langchain.dev/langsmith-ga/), access LangSmith directly [here](https://smith.langchain.com/), read about their journey on [Forbes](https://www.forbes.com/sites/alexkonrad/2024/02/15/open-source-ai-startup-langchain-launches-langsmith/?sh=26e00cb24f00), and find out more about working with them at [LangChain Careers](https://www.langchain.com/careers).

- **Query About LangSmith Pricing**: `@rajib2189` expressed enthusiasm about the announcement of LangSmith but noted issues accessing the pricing page.

- **LangSmith Featured on Product Hunt**: `@hwchase17` mentioned that LangSmith is now live on Product Hunt, showcasing its features for developing and monitoring LLM applications. Find it on Product Hunt: [LangSmith General Availability](https://www.producthunt.com/posts/langsmith-general-availability).

**Links mentioned**:

- [ LangSmith General Availability - LLM application development, monitoring, and testing | Product Hunt](https://www.producthunt.com/posts/langsmith-general-availability): LangSmith is a solution for developing, tracing, debugging, testing, deploying, and monitoring LLM applications. Integrates seamlessly with LangChain, but exposes SDKs for use outside of the LangChain...
- [Loom | Free Screen &amp; Video Recording Software](https://www.loom.com/share/63a9696036c74765a9f9ecae06336aa5): Use Loom to record quick videos of your screen and cam. Explain anything clearly and easily – and skip the meeting. An essential tool for hybrid workplaces.
- [LangChain Companion - Journal](https://journal.langchain.com/): no description found
- [Announcing the General Availability of LangSmith and Our Series A Led By Sequoia Capital](https://blog.langchain.dev/langsmith-ga/): Today, we’re thrilled to announce the general availability of LangSmith — our solution for LLM application development, monitoring, and testing. We initially launched LangSmith in closed beta in July ...
- [LangSmith](https://smith.langchain.com/): no description found
- [Open Source AI Software Maker LangChain Launches First Paid Product — With A Massive Waitlist](https://www.forbes.com/sites/alexkonrad/2024/02/15/open-source-ai-startup-langchain-launches-langsmith/?sh=26e00cb24f00): CEO Harrison Chase confirmed a $20 million funding round led by Sequoia and said his one-year-old startup already had a waitlist of 80,000 for its new tools.
- [LangChain](https://www.langchain.com/): LangChain’s suite of products supports developers along each step of their development journey.
- [Careers](https://www.langchain.com/careers): We are a small team of builders making an outsized impact in our industry.

  

---


### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1206920246364213248) (64 messages🔥🔥): 

- **Database Integration for Personalized Chats**: `@batmansalt` suggested that personalized chat histories for customers can be stored in a **database**, and loaded into the chatbot prompt when a new chat session begins, enhancing the interaction with the chatbot.
- **Peer Dependency Conflicts with Pinecone and Langchain**: `@segmentationfault.` encountered issues upgrading **Pinecone** to v2 due to **langchain** dependency conflicts. `@jacoblee93`, a full-time maintainer of **Langchain**, provided assistance, including the recommendation to use `npm install --legacy-peer-deps` or to bump **langchain** to the latest version.
- **Langchain User Seeks RAG Optimization Tips**: `@barrahh` inquired about optimizing **RAG** pipelines based on user feedback or ratings. `@batmansalt` proposed manual inspection of results to refine parameters like chunk size and number of retrieved texts, and mentioned using high ratings for future fine-tuning of the model.
- **Collaboration Opportunities in the Langchain Community**: `@kiddu` expressed interest in joining AI or backend projects, and `@aminerwy` invited collaborations on an AI-powered public transit planning assistant, especially to improve the backend RAG conversational chatbot.
- **Trouble with Streaming in ConversationChain**: `@hndrxx_25149_81926` mentioned issues with streaming capability within **ConversationChain**, and queried the community for possible workarounds.

**Links mentioned**:

[Pinecone | 🦜️🔗 Langchain](https://js.langchain.com/docs/integrations/vectorstores/pinecone): You can use Pinecone vectorstores with LangChain.

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1207088905829089401) (64 messages🔥🔥): 

- **Image Base64 Bloating Browser**: `@dachsteinhustler` found that including base64-encoded images in the **LangChain playground** caused browser crashes due to the lengthy intermediate steps. A rewrite of the app using a **RunnableLambda** avoided displaying the strings.

- **K8s Connection Refused Troubles**: `@ezelanza.` experienced a "connection refused" error when trying to invoke an OpenAI API within a **Kubernetes cluster**. The issue was discussed extensively with `@veryboldbagel` providing guidance on security concerns regarding accidentally posted OpenAI API keys and offering structure fixes for **CURL requests** and the use of **APIHandlers**.

- **Trouble With LangServe Routes**: `@ezelanza.` sought help debugging issues with **LangServe routes**. `@veryboldbagel` advised on the correct curl request structure and suggested checking the request pattern using browser developer tools.

- **LangServe Chat History Challenge**: `@lfglopes` queried about implementing a chat history feature within LangServe that interacts with a **SQL database**, sparking a discussion on managing conversation threads through separate endpoints or internally generated UUIDs guided by `@veryboldbagel`.

- **Deployment Query for Langchain/LangServe App**: `@aminerwy` was looking for advice on deploying a Langchain/LangServe app to be **accessible from the web**. The community mentioned Vercel and Replit as potential platforms for deployment.

**Links mentioned**:

- [RunnableLambda: Run Custom Functions | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/how_to/functions#accepting-a-runnable-config): run-custom-functions}
- [langserve/examples/api_handler_examples/server.py at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/blob/main/examples/api_handler_examples/server.py): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.
- [langserve/examples/multiple_servers at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/tree/main/examples/multiple_servers): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.
- [Invoke through forwarding message · Issue #466 · langchain-ai/langserve](https://github.com/langchain-ai/langserve/issues/466): Hi My scenario is FRONT_END(:5005/openaiapi)--&gt; BACK_END(:8000/api_openai )-INVOKE --&gt;OPENAI (API) In my front end, I&#39;m using Fastapi listening to /api_openai, to forward the request to open...
- [Streaming | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/streaming#generator-functions): streaming-with-langchain}
- [Examples: Add multiple servers example by eyurtsev · Pull Request #469 · langchain-ai/langserve](https://github.com/langchain-ai/langserve/pull/469): Adds an example of using multiple servers

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1207253301083508746) (4 messages): 

- **Guidance for Goal-Setting Assistant**: `@avfranco` offered a step-by-step approach for creating a goal-setting assistant, suggesting that establishing a vision, breaking down features, selecting core components like architecture and user interface, and continuous experimentation are key steps to success.

- **Curiosity About Action Plan Tools vs. LangGraph**: `@jay0304.` inquired whether the Action plan and tools are an alternative to LangGraph, or if they can be utilized concurrently.

- **Reverse Job Board for AI Experts**: `@sumodd` introduced a new [Reverse Job Board](https://www.aidevs.work/) for individuals seeking AI-related roles, featuring a free platform where recruiters can discover potential candidates listing various skills and experiences.

- **LangChain Meets Dewy**: `@kerinin` shared a [tutorial](https://dewykb.github.io/blog/qa-cli-with-langchain) on building a question-answering CLI with Dewy, an open-source knowledge base, and LangChain.js, illustrating how developers can incorporate large language model functionalities into their applications.

**Links mentioned**:

- [Building a question-answering CLI with LangChain.js and Dewy | Dewy](https://dewykb.github.io/blog/qa-cli-with-langchain): This guide will walk you through building a question-answering CLI using LangChain.js for prompting, the OpenAI API for the language model, and Dewy as your knowledge base.
- [Neural Network - an initiative by Convergent Software](https://www.aidevs.work/): A reverse job board to connect AI engineers with organizations

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1207682664702681128) (1 messages): 

- **Learn to Implement Multi-Document RAG**: `@mehulgupta7991` shared a [YouTube tutorial](https://youtu.be/cBpdiQ3gljM?si=lAhY7F0UXZfUZP57) titled "Multi Document RAG using LangChain codes explained," which guides viewers through the implementation of **Multi-Document RAG** using Agents with custom tools for chatting with different external files. The tutorial is also part of a book launched by the user.

**Links mentioned**:

[Multi Document RAG using LangChain codes explained](https://youtu.be/cBpdiQ3gljM?si=lAhY7F0UXZfUZP57): This tutorial explains how to use multiple diverse files with a single RAG agent for querying your data. This tutorial is a part of my newly launched book &quot;L...

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1206909948051587092) (69 messages🔥🔥): 

- **Exploring Keras Habitation for Porting Models**: `@yamashi` proposed porting models to Keras to extend hardware support. They highlighted that Keras, now independent with version 3, serves as an abstraction layer over frameworks such as Torch, TF, and Jax.
- **Checkpoint Saving Snafus Identified**: After facing an error during checkpoint saving, `@dreamgen` shared a [link](https://github.com/huggingface/peft/pull/1414) to a pull request on the HuggingFace repository that caused issues, which were discussed in relation to recent outages experienced by HF.
- **Quest for Affordable LLM Hosting**: `@le_mess` inquired about the cheapest endpoint service for hosting a model like Mixtral via an API. Contributions from `@dreamgen`, `@noobmaster29`, and others outlined various options, including `together.ai`, OpenRouter, and `basten`.
- **NVIDIA Showcases Chatbot Frontend**: `@dangfutures` shared a [link](https://github.com/NVIDIA/trt-llm-rag-windows) to NVIDIA's RTX-based demo app called Chat With RTX, allowing personalized GPT models to run on local RTX hardware. A discussion surrounding its practicality and bugs ensued, with alternatives like using the engine with Chainlit proposed by `@nruaif` and `@dangfutures`.
- **CohereForAI's Aya Model Serialization**: Users `@noobmaster29`, `@nanobitz`, and `@dreamgen` discussed the newly released Aya model by CohereForAI, capable of instructions in 101 languages, and considered its expected performance in comparison to its predecessors.

**Links mentioned**:

- [Quantization](https://huggingface.co/docs/optimum/en/llm_quantization/usage_guides/quantization): no description found
- [CohereForAI/aya-101 · Hugging Face](https://huggingface.co/CohereForAI/aya-101): no description found
- [Together AI](https://www.together.ai/): Build gen AI models with Together AI. Benefit from the fastest and most cost-efficient tools and infra. Collaborate with our expert AI team that’s dedicated to your success.
- [peft/utils/save_and_load.py try to connect to the hub even when HF_HUB_OFFLINE=1 · Issue #1452 · huggingface/peft](https://github.com/huggingface/peft/issues/1452): System Info peft 0.8.2 axolotl v0.4.0 export HF_DATASETS_OFFLINE=1 export TRANSFORMERS_OFFLINE=1 export HF_HUB_OFFLINE=1 Who can help? No response Information The official example scripts My own mo...
- [GitHub - NVIDIA/trt-llm-rag-windows: A developer reference project for creating Retrieval Augmented Generation (RAG) chatbots on Windows using TensorRT-LLM](https://github.com/NVIDIA/trt-llm-rag-windows): A developer reference project for creating Retrieval Augmented Generation (RAG) chatbots on Windows using TensorRT-LLM - NVIDIA/trt-llm-rag-windows
- [Fix breaking change  by younesbelkada · Pull Request #1414 · huggingface/peft](https://github.com/huggingface/peft/pull/1414): Fix a breaking change in the recent release, I made a new PR as I messed up the commit history on the previous PR cc @sayakpaul @pacman100
- [GitHub - triton-inference-server/tensorrtllm_backend: The Triton TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend): The Triton TensorRT-LLM Backend. Contribute to triton-inference-server/tensorrtllm_backend development by creating an account on GitHub.
- [NVIDIA Chat With RTX](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/?utm_source=www.therundown.ai&utm_medium=newsletter&utm_campaign=chatgpt-gets-a-memory): Your Personalized AI Chatbot.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1207090774991175680) (20 messages🔥): 

- **Collaborating Minds for Schema Design**: `@faldore` and others have agreed that a JSON schema detailing user and assistant message pairs, with optional system, tools, and source messages is ideal for dataset formatting. `@faldore` illustrated how this schema enforces user and assistant message pairing and that the last response is always from the assistant.
  
- **Flexibility in Role Naming**: `@c.gato` raised the idea of renaming the "user" and "assistant" roles in the message schema. They noted the influence of using the term "assistant" on model behavior and shared an anecdote about having to replace "assistant" with "secretary" in an RP model to avoid self-referential AI behavior.

- **Multi-User Chat Complications**: Upon further discussion of the proposed schema, `@c.gato` inquired about its applicability to multi-user chat scenarios. `@dreamgen` suggested the core schema should be unopinionated and open to extensions to cater to diverse tasks like RP or story-writing.

- **Support for Non-Linear Learning in AI**: `@suikamelon` shared a research paper on the potential benefits of integrating structured cognitive learning methodologies into LLM instruction tuning and asked about disabling random shuffling in favor of curriculum learning. `@c.gato` expressed interest in exploring the sorting of training examples by length.

- **Contemporary Discussion on Instruction-Tuned LLMs**: `@suikamelon` discussed a novel approach to instruction tuning inspired by curriculum learning, indicating models may perform better when complex instructions are fine-tuned last. They express skepticism but acknowledge the potential utility of a more structured and less randomized approach to fine-tuning.

**Links mentioned**:

- [Instruction Tuning with Human Curriculum](https://arxiv.org/abs/2310.09518): In building instruction-tuned large language models (LLMs), the importance of a deep understanding of human knowledge can be often overlooked by the importance of instruction diversification. This res...
- [RunPod template not working with network volumes, /workspace/axolotl empty · Issue #813 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/813): Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior Other users also encountered this: #467 According t...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1206977628645630012) (11 messages🔥): 

- **Real-time LoRA Adapter updates**: `@wizmak` sought advice on adding fine-tuned LoRA adapters to a base model in real-time, without needing to merge and restart. `@nanobitz` confirmed it's possible with HF, implying you can load and unload the PEFT model dynamically.

- **SGLang and LLaVA Worker Requirements?**: `@CodeMan` questioned if both SGLang and LLaVA workers are necessary for particular functionality, but the context or responses to the query were not provided.

- **In Search of DeepSpeed Config for Model Parallelism**: `@mihai4256` asked for a working DeepSpeed Zero 3 config for model parallelism, noting the challenge of finding one despite their expectation that it should be readily available.

- **Mixture of Experts Training Resources**: `@emperor` inquired about the best repository to train different mixture of experts architectures from scratch, later suggesting Megablocks as a possible solution, though no direct affirmations or alternatives were provided.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1207032182896599160) (15 messages🔥): 

- **RunPod Image Works on Vast.AI**: `@dreamgen` shared a *Public Service Announcement* that the **Axoltl RunPod** image can be used on Vast.AI without any issues, and it works **straight out of the box**.
- **Opting for Vast.AI over RunPod**: `@dreamgen` and `@dangfutures` noted that **Vast.AI** might offer **cheaper GPUs** than RunPod, especially when **H100 SXM** GPUs are rarely available on RunPod's community cloud.
- **Ease of Setup between Services**: `@dangfutures` commented that although **Vast.AI** may be preferred for GPU pricing, RunPod offers a simpler setup process. `@dreamgen` said the setup felt similar on both and highlighted that `/workspace/axolotl` isn't empty on Vast.
- **Data Transfer Queries Resolved**: Users asked about transferring data from services like **Google Storage** to Vast. `@dreamgen` recommended using `scp`, and `@nanobitz` mentioned you can **SSH into a Docker** container provided by Vast, offering considerable flexibility.
- **Troubleshooting RunPod GPU Issues**: `@c.gato` indicated frustration with **4090 RunPods**, stating that they appear to have **driver issues**, mentioning crashes due to lack of AMP support, and also had difficulty getting the **axolotl docker** to work on Vast.
  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1207258559893610546) (9 messages🔥): 

- **LLM's Struggle with Math Post Business-Finetuning**: User `@mertbozkir` suggested that a 7B parameter model fine-tuned on business data would give poor answers to math questions if it lacks domain-specific methods like forward/backward reasoning. They mentioned alternatives like **internlm, metamath, arithmo** which are configured for such tasks.
- **Price Surge for 3090s GPUs**: `@joseph_en` lamented the price increase of 3090s GPUs, sharing their experience of buying them cheaper back in July and August, implying a significant cost uptrend.
- **GPU's Unpredictable Value**: `@andreaskoepf` humorously referred to the GPUs as "GPU gold" in the light of recent price fluctuations and the shared experiences of `@joseph_en`.
- **Resource Page Update Reminder**: `@andreaskoepf` acknowledged the need to update their resource-stream page with the recently posted links, showcasing an effort to keep shared resources organized.
  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1206870504657657876) (12 messages🔥): 

- **Microsoft corners chip market**: `@andreaskoepf` humorously suggested that Microsoft has bought the full production capacity of chips, affecting the market, and joked about antitrust agencies being unable to keep up with Sam Altman's pace, insinuating a dystopian future where traditional efforts can't combat Altman's "nano-bot and virus army."
- **GPU Troubles with PyTorch**: `@_tvi_` shared frustrations about working with PyTorch on a Radeon VII and Ryzen APU, citing issues with video RAM allocation and kernel crashes when large memory chunks are allocated.
- **CUDA Compatibility Anecdotes**: `@shikhar_7985` sought advice on managing different CUDA versions for various projects, while `@btdubbins` discussed the need to remain pinned to CUDA 11 for compatibility with FAISS, before considering an update to CUDA 12.
- **PyTorch Inline Loading Bug**: `@eporat` reported a problem with PyTorch's `load_inline` not creating .so files, which was resolved by changing the optimization flag; `@marksaroufim` suggested a workaround for a known recompilation issue by adding a semicolon to the code — [as discussed here](https://github.com/pytorch/pytorch/issues/119206).
- **Conda for CUDA Version Management**: In response to a CUDA version management query, `@marksaroufim` recommended using Conda when working with PyTorch, as found on [PyTorch's official site](https://pytorch.org/).

**Links mentioned**:

- [Tweet from [Phoronix] AMD Quietly Funded A Drop-In CUDA Implementation Built On ROCm: It&#039;s Now Open-Source Image (Radeon Cuda 1)](https://www.phoronix.com/image-viewer.php?id=radeon-cuda-zluda&image=radeon_cuda_1_lrg): no description found
- [
    
      PyTorch
    
  ](https://pytorch.org/): no description found
- [load_inline should always recompile a kernel if it failed · Issue #119206 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/119206): 🐛 Describe the bug Repro in the real world here https://github.com/cuda-mode/lectures/blob/main/lecture3/pmpp.ipynb If you try to load a kernel without ninja the load will fail but then if you pip .....

  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1207071022029742130) (9 messages🔥): 

- **Exploring the Bounds of Function Composition**: `@euclaise` shared insights on using prefix-sum-like scans for computing complex recurrences, such as `y[t]=max(y[t-1], x[t])`, indicating the approach's generality due to function composition's associative property. Further details and discussions can be found in their [tweets](https://twitter.com/Euclaise_/status/1757795082067919055).
  
- **Skepticism on Function Representation**: `@andreaskoepf` questioned the practical limitations in representing functions using the method `@euclaise` discussed, expressing curiosity about the performance-acceptable class of such representations.

- **Practical Challenges with Associativity of Functions**: `@_tvi_` pointed out the computational difficulty in applying function associativity for more complex functions, suggesting the approach's utility may be limited to functions that are "easy to represent and fast to apply."

- **In Search of Knowledge on Function Classes**: `@telepath8401` inquired about resources for understanding "easily representable" functions and their classes, indicating a desire to learn more about the subject.

- **Collaboration Invitation for RingAttention Kernel Project**: `@andreaskoepf` extended an invitation for collaboration on the RingAttention kernel project within the cuda-mode Discord, offering to help organize GPU resources and coordinate efforts despite not being able to fully devote themselves as a developer.
  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1206882432926748713) (9 messages🔥): 

- **User Reveals Their GPU**: `@cs_os_05101` mentioned simply, **"I have 4060 Ti."**
- **Looking for Engaging CUDA Literature**: `@euclaise` inquired about CUDA books that are **fun to read**; however, no specific book titles were recommended in the conversation.
- **Shaders as an Entry Point**: `@marksaroufim` suggested [The Book of Shaders](https://thebookofshaders.com/) by *Patricio Gonzalez Vivo* and *Jen Lowe* as a fun and progressive guide to Fragment Shaders.
- **euclaise Familiar with Shaders, Not CUDA**: Despite the suggestion, `@euclaise` clarified they are already **familiar with shader programming** but not directly with CUDA or compute shaders.
- **Seeking Fun in Programming Massively Parallel Processors (PMPP)**: Although not characterizing it as fun, `@marksaroufim` mentioned **PMPP** (Programming Massively Parallel Processors) as the best resource they've found related to CUDA, while `@euclaise` expressed a willingness to try it, suggesting that research is the most **fun** for them.

**Links mentioned**:

[The Book of Shaders](https://thebookofshaders.com/): Gentle step-by-step guide through the abstract and complex universe of Fragment Shaders.

  

---


### CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1207062344346247168) (7 messages): 

- **Matrix Transposition May Not Boost Performance**: `@andreaskoepf` brought up whether maintaining both vectors for dot products in sequential memory makes a significant performance difference and suggested the idea of an alternating memory layout. `@jeremyhoward` responded, noting from his experience that transposing the matrix for tile creation did not yield any performance improvements.

- **Exploring In-Loop Index Order Optimization**: `@eporat` mentioned that instead of in-place transposing, changing the order of indices in the inner loop might be a viable optimization. However, `@andreaskoepf` seems unsure about the improvements as the data would be read transposed in any case.

- **Altering For-Loop Variables Slows Down Performance**: `@eporat` tested changes in a CUDA kernel, only to find that changing the order of loop variables made the function even slower. They shared a modified function with an `atomicAdd` operation, but it failed to work efficiently with shared memory.
  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1207248533564948490) (4 messages): 

- **Lecture 5 Inquiry and Discovery**: `@filippob82` inquired about the availability of **Lecture 5** on the Cuda YouTube channel. `@reluctantly_normalized` replied with a link to the lecture on Jeremy Howard's channel: [Going Further with CUDA for Python Programmers](https://youtu.be/eUuGdh3nBGo?si=XnUPc-oaAdy4IQLd).
- **Suggestion for Cuda's Channel**: `@reluctantly_normalized` suggested to have a reference or a copy of the lecture on **Cuda's official YouTube channel** for easier access.
- **Playlist Creation Idea**: `@filippob82` proposed creating a YouTube playlist as a possible solution to organize the lectures.

**Links mentioned**:

[Going Further with CUDA for Python Programmers](https://youtu.be/eUuGdh3nBGo?si=XnUPc-oaAdy4IQLd): This technical talk by Jeremy Howard explores advanced programming techniques for maximizing performance when using CUDA with Python. The focus is on optimiz...

  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1207236267713241119) (2 messages): 

- **Uncertain Future for TensorFlow?**: User `@spacyphus` inquired about the potential discontinuation of TensorFlow, but there was no further discussion or information provided to confirm or deny this possibility.
- **JAX vs. PyTorch Debate Sparked**: User `@marcom79` asked `@660097403046723594` for an opinion on JAX versus PyTorch, suggesting that PyTorch 2.0 might be feature-equivalent to JAX after its recent updates. The conversation did not progress beyond the initial query.
  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1207711000615325736) (10 messages🔥): 

- **Gemini Pro 1.5 Excites with Massive Context Window**: `@wenquai` expressed excitement for **Gemini Pro 1.5**, which boasts a 1 million token context window and the ability to process long videos. They find the capacity to handle such extensive content impressive.
- **Skepticism Around Large Context Windows**: `@thebaghdaddy` expressed skepticism about the effectiveness of large context windows, like the 250k ones, citing that models tend to perform worse after 50-60k tokens. They referenced testing on Claude which showed negligence of content in the middle of large context windows.
- **Curiosity Based on Google's Claims**: `@wenquai` acknowledged this skepticism but mentioned relying on Google’s reports for optimism, also revealing efforts to gain access through a Google cloud rep.
- **Gemini's Claimed Ten Million Token Context**: `@thebaghdaddy` corrected an earlier figure, stating Gemini Pro 1.5 claims an even more astonishing ten million token context window, referencing a post by Jeff Dean.
- **Jeff Dean Highlights Gemini 1.5 Pro Innovations**: A detailed post by Jeff Dean shared by `@thebaghdaddy` unveils **Gemini 1.5 Pro**, highlighting its 10 million token context length and ability to handle vast multimodal inputs. Dean's post [Twitter](https://x.com/jeffdean/status/1758146022726041615?s=46) includes links to a main blog post, technical report, and various interaction videos along with announcements of a limited developer preview and upcoming broader model release with pricing tiers.

**Links mentioned**:

[Tweet from Jeff Dean (@🏡) (@JeffDean)](https://x.com/jeffdean/status/1758146022726041615?s=46): Gemini 1.5 Pro - A highly capable multimodal model with a 10M token context length  Today we are releasing the first demonstrations of the capabilities of the Gemini 1.5 series, with the Gemini 1.5 Pr...

  

---


### LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/) (1 messages): 

robotums: yeah
  

---


### LLM Perf Enthusiasts AI ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/1207392639335399474) (15 messages🔥): 

- **Massive Dataset Processing Costs**: `@res6969` shared that their dataset comprises **35k PDFs with around 40 pages each**, resulting in substantial processing costs, particularly due to the use of a vision transformer.
- **Vision Transformers vs GPT-4V**: In the cost breakdown, `@res6969` clarified that the **vision transformer** represents a significant portion of the cost, despite initially considering whether the cost was mostly due to **GPT-4V**.
- **Announcing Surya OCR**: `@robhaisfield` highlighted a new OCR tool called [**Surya OCR**](https://github.com/VikParuchuri/surya), which **outperforms Tesseract** in text recognition for **93 languages** according to a tweet from @VikParuchuri.
- **Seeking Cost-Effective Alternatives**: `@robhaisfield` and `@res6969` discussed the possibility of finding a more efficient method than the vision transformer for classifying sections in PDFs, with the vision transformer costing **$10/1000 pages**.
- **Innovative Solutions on the Horizon**: `@robhaisfield` suggested the use of **GPT-4V or Llava** for identifying charts or figures in a PDF as a potential cost-saving measure, which `@res6969` acknowledged could indeed work and contemplated doing the math to compare costs.

**Links mentioned**:

[Tweet from Vik Paruchuri (@VikParuchuri)](https://fxtwitter.com/VikParuchuri/status/1757185570940567666?s=20): Announcing surya OCR - text recognition in 93 languages. It outperforms tesseract in almost all languages, often by large margins.  Find it here - https://github.com/VikParuchuri/surya .

  

---


### LLM Perf Enthusiasts AI ▷ #[irl](https://discord.com/channels/1168579740391710851/1171569983688560732/1207341172012621967) (1 messages): 

- **AI Wednesdays with Free Pizza**: `@ivanleomk` invites AI enthusiasts to gather at **Funan Mall, Singapore**, next Wednesday for a project hacking session complete with **free pizza**. The event is hosted by Gabriel Chua, Jon Jon, & tengfone, with [details and registration available here](https://lu.ma/ai-weds). Only one spot is left and registration requires host approval.

**Links mentioned**:

[AI Wednesdays · Luma](https://lu.ma/ai-weds): Let&#x27;s hang out and build! 🛠️ 🔥 📍 Location: Near Funan Mall (Exact location will be provided to registered attendees) ⏰ Doors open at 5.30pm, and feel free to join any time. 🍕 Pizza, 📶...

  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1207005947008913471) (8 messages🔥): 

- **GPT-5 Rumors Abound**: `@res6969` humorously noted the extent of rumors about GPT-5 might be overhyped.
- **Laughter Among Enthusiasts**: Emojis from `@res6969` and `@potrock` indicate amusement, possibly in response to ongoing discussions or the hype around GPT-5.
- **OpenAI Tests ChatGPT with Memory**: `@potrock` shared [OpenAI's blog post](https://openai.com/blog/memory-and-new-controls-for-chatgpt) on a new ChatGPT feature that tests memory across conversations, allowing users to request the AI to remember or forget certain pieces of information.
- **Skepticism Over OpenAI's Recent Updates**: `@thebaghdaddy` expressed a critical view that OpenAI might be using strategy leaks as a distraction from less popular feature releases in the past months.
- **Announcing OpenAI's Sora**: `@res6969` linked to OpenAI's introduction of [Sora](https://openai.com/sora), a text-to-video AI model that generates minute-long videos and is now being tested by red teamers and creative professionals to assess potential risks and gather feedback on its use.

**Links mentioned**:

- [Sora: Creating video from text](https://openai.com/sora): no description found
- [Memory and new controls for ChatGPT](https://openai.com/blog/memory-and-new-controls-for-chatgpt): We’re testing the ability for ChatGPT to remember things you discuss to make future chats more helpful. You’re in control of ChatGPT’s memory.

  

---



### Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1207259092977188864) (4 messages): 

- **Fine-Tuning LLMs on Business Data Affects Math Performance**: `@sabu7003` inquired about the potential performance of a large language model (LLM), specifically a 7B parameter model, on math questions after fine-tuning on business data only. `@rusch` suggested that such fine-tuning would **gradually degrade the model's math capabilities**, with the degree of degradation being proportional to the intensity and duration of the fine-tuning process.
- **Optimism vs. Pessimism in ML Decision Making**: `@rrenaud` shared an insight on the role of optimism in exploration/exploitation during reinforcement learning (RL), and how, conversely, **pessimism during inference** can prevent machine learning (ML) systems from deviating too far from the training distribution and help maintain stability in sequential decision-making.
  

---


### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1207652894057439272) (1 messages): 

- **Seeking Business-Specific Instructions**: `@sabu7003` is looking for methods to extract **only business-related instructions** from the teknium/OpenHermes-2.5 Instruction dataset. They have not indicated any methods attempted or any links to the dataset.
  

---


### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1207457661709324378) (2 messages): 

- **Silence from a Discord User**: `@joshxt` expressed concern over not hearing from a user, hinting that the user's **Discord might be broken**. `@atlasunified` suggested that **direct messaging** (DM) him is the best course of action.
  

---


### Alignment Lab AI ▷ #[qa](https://discord.com/channels/1087862276448595968/1147528698669584424/) (1 messages): 

daydream.nation: o sh
  

---



### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1207037835085881404) (1 messages): 

- **LLaVA Setup Inquiry**: User `@CodeMan` is seeking advice on integrating **LLaVA** with an **SGLang server and SGLang worker**, as opposed to the standard model worker setup. No responses or further discussion followed.
  

---


### Skunkworks AI ▷ #[datasets](https://discord.com/channels/1131084849432768614/1131669182124138616/1207652538409680926) (1 messages): 

- **Seeking Business-Specific Instructions**: `@sabu7003` inquired about methods to filter out **business-related instructions** from the [teknium/OpenHermes-2.5 Instruction dataset](https://github.com/teknium/OpenHermes-2.5). They are looking for guidance on how to isolate business-specific data.
  

---


### Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/1207258970004394044) (1 messages): 

- **Inquiring Minds Want to Know**: User `@sabu7003` questioned the ability of a **7B parameter LLM** to answer math questions after being fine-tuned on business data alone, pondering if and how the performance on math would differ from business queries. There was no response or further discussion provided in the channel messages.
  

---


### Skunkworks AI ▷ #[papers](https://discord.com/channels/1131084849432768614/1156310031768232007/1207801787214598215) (4 messages): 

- **Can Random Seeds be Learnable?**: `@stereoplegic` inquired about the possibility of **random seeds** being learnable as scalar parameters in AI models.
- **Learning Random Seeds - A Technical Impossibility?**: `@aspott` asserted that learning a random seed isn't feasible since one can't get a gradient on a random seed.
- **Exploring Seed Loss and Initialization Functions**: Despite the challenge, `@stereoplegic` suggested evaluating the loss of passes through the parameters initialized by seeds, while `@aspott` proposed the potential to learn an **initialization function** instead.
  

---



### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1207016777154826250) (7 messages): 

- **Weekly Meeting Kickoff**: `@._z` announced the **start of the weekly meeting** with a jovial "Déjà vu" sentiment.
- **Absentee Alert**: `@juanreds` informed that they **couldn't attend** the weekly meeting.
- **Hackathon Co-hosting Opportunity**: `@caramelchameleon` inquired about interest in **co-hosting an AI developers hackathon** before Game Developers Conference, open to both online and onsite participation in San Francisco.
- **Hackathon Organizer Steps In**: `@yikesawjeez` expressed interest, mentioning their experience in organizing **hackathons related to events** in the Bay Area.
- **Exclusive Founders x VC Event Slots Open**: `@atalovesyou` shared an opportunity for startup founders to join an **investor matchmaking session** with limited additional spots available at [Founders x VC Event](https://lu.ma/c4klulz8), featuring 30+ venture capital firms and extensive networking opportunities.

**Links mentioned**:

[Founder x Investor Matchmaking · Luma](https://lu.ma/c4klulz8): LIMITED SPOTS REMAINING. We have received interest from over 600+ Pre-Seed, Seed, Series A+ Founders. We are at capacity but opened a few more slots for founders on a ticket...

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1207745451688857681) (2 messages): 

- **Google Unveils Gemini 1.5 Pro**: `@tariqali` shared a [Google Developers blog post](https://developers.googleblog.com/2024/02/gemini-15-available-for-private-preview-in-google-ai-studio.html?m=1) announcing the private preview of Google's *Gemini 1.5 Pro*, which reportedly has the same performance as *Gemini 1.0 Ultra* but uses less compute. It also referenced the model's ability to handle a context window of 1 million tokens.
  
- **Expanding the Context Window**: In discussing the importance of prompt engineering with larger context windows, `@tariqali` speculated that the need for prompt engineering could decrease as the ability to input more relevant data directly increases. They considered the possibility that the cheaper compute might outweigh the efforts of prompt engineering, potentially rendering it an antiquated skill.

- **Gemini's Potential Constraints**: Following up, `@tariqali` highlighted a line from the [Google Blog](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/) indicating that Google has successfully tested models with up to 10 million tokens, but chose to release *Gemini 1.5* with a 1 million token context window instead. They inferred that there might be significant constraints, such as cost, preventing the release of models with larger context windows, suggesting that prompt engineering may still be valuable in the short term.

**Links mentioned**:

- [Our next-generation model: Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/): Gemini 1.5 delivers dramatically enhanced performance, with a breakthrough in long\u002Dcontext understanding across modalities.
- [Gemini 1.5: Our next-generation model, now available for Private Preview in Google AI Studio - Google for Developers](https://developers.googleblog.com/2024/02/gemini-15-available-for-private-preview-in-google-ai-studio.html?m=1): no description found
- [Error 404 (Not Found)!!!](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/,): no description found
- [Large World Models](https://largeworldmodel.github.io/): no description found

  


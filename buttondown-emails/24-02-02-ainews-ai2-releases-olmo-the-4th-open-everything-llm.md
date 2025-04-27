---
id: 134d580b-d650-41c0-bdf3-3118898ad27c
title: AI2 releases OLMo - the 4th open-everything LLM
date: '2024-02-03T03:35:10.019799Z'
original_slug: ainews-ai2-releases-olmo-the-4th-open-everything
description: >-
  **AI2** is gaining attention in 2024 with the release of **OLMo**, a series of
  open language models including **1B**, **7B**, and an upcoming **65B** model,
  emphasizing reproducible and open research akin to **Pythia**. The
  **TheBloke** Discord highlighted **Miqu-70B**'s self-correcting abilities and
  discussions on programming languages and VRAM optimization. The **Mistral**
  Discord focused on the **GPU shortage** impacting semiconductor production
  involving **TSMC**, **ASML**, and **Zeiss**, debates on open-source versus
  proprietary models, and fine-tuning challenges with references to **LoRA**
  techniques. *"The releasing-models-thru-magnet-link meta still has not yet run
  out of juice"* and detailed community insights were shared by **Nathan
  Lambert** and others.
companies:
  - ai2
  - allenai
  - mistral-ai
  - tsmc
  - asml
  - zeiss
models:
  - olmo-1b
  - olmo-7b
  - olmo-65b
  - miqu-70b
  - miqu-1-70b-sf-4.25bpw-h6-exl2
  - distilbert-base-uncased
topics:
  - fine-tuning
  - gpu-shortage
  - model-release
  - open-source
  - embedding-chunking
  - search-functionality
  - programming-languages
  - vram-optimization
  - reproducibility
people:
  - natolambert
  - lhc1921
  - mrdragonfox
  - gbourdin
  - yashkhare_
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/1/2024. We checked **21** guilds, **312** channels, and **5874** messages for you. Estimated reading time saved (at 200wpm): **483 minutes**. We enountered stability issues with very link-heavy messages (thanks @yikesawjeez...) that we had to figure out how to address.

As teased on Nathan Lambert's [Latent Space appearance](https://www.latent.space/p/rlhf-201), we're about to see AI2 come up a lot more this year under new leadership. The first results of that are coming through now with [OLMo](https://allenai.org/olmo/olmo-paper.pdf) (Open Language MOdels) - a 1B, and set of 7B models, with a 65B on the way.

 ![image.png](https://assets.buttondown.email/images/bbb5cb1e-b02f-41f7-9da0-31dce096f27f.png?w=960&fit=max) 

[Nathan's Substack](https://www.interconnects.ai/p/olmo) has the less corpo take if you enjoy that tone (we do) and it is also fun to note that [the releasing-models-thru-magnet-link meta](https://twitter.com/natolambert/status/1753063313351835941) still has not yet run out of juice.

In the LS Discord we had the honor of discussion with Nathan in more detail, including the odd choice to release a "Twin" AMD model, the exclusion of Mistral 7B from benchmarks, and more.

 ![image.png](https://assets.buttondown.email/images/09aedd23-595b-48fd-bc44-6e12b56d7df4.png?w=960&fit=max) 

We happened to cover Pythia ([one of the top 10 papers of 2023](https://magazine.sebastianraschka.com/p/10-ai-research-papers-2023)) in this week's Paper Club, and Nathan agreed that OLMo might be regarded a spiritual successor to Pythia in its commitment to reproducible and fully open research. 

Hopefully the start of more in 2024.

---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Miqu-70B: The Surprise Self-Corrector**: Within the **general** discussions, Miqu-70B, particularly the Mistral Medium version, received praise for its ability to self-correct during responses. Advanced users reported model speed averages of 17 tokens per second after optimizations, comparing it favorably with Mixtral.

- **Language Showdown: C++ vs. Rust vs. Go**: A lively debate amongst engineers in the **general** channel concerned the merits of programming languages such as C++, Rust, and Go. The preference seems to lean towards the simpler, more manageable languages due to their ease of understanding.

- **VRAM Capers with Miqu-1-70B-SF**: In **characters-roleplay-stories**, efforts were made to fit Miqu-1-70B-SF-4.25bpw-h6-exl2 within the constraints of available VRAM, leading to discussions on potential solutions, including hardware upgrades.

- **Category Conundrum: Distilbert-base to the Rescue?**: One user in the **training-and-fine-tuning** channel experimented with **Distilbert-base-uncased** for predicting hierarchical codes from text, observing fair accuracy for higher-level predictions but difficulties with the finer categorical distinctions.

- **File Hierarchy Heresy: Search to Surpass Structure**: Users in the **coding** channel discussed the increasing reliance on powerful **search functionality** over traditional file organization, with good search considered more effective than traditional methods like tags or directories, especially in software like nvpy and simplenote.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Tackling the GPU Drought**: Engineers discussed the **GPU shortage** due to complex semiconductor manufacturing processes and the months-long wafer production cycle. This limits scaling up production, with companies like **TSMC, ASML, and Zeiss** bearing the brunt of collaboration requirements.

- **Open Source vs. Proprietary**: A vibrant debate centered on open-source versus proprietary software, highlighting that while open source relies on community and company contributions, the "open core model" offers a promising monetization strategy through premium features.

- **Chunking Choices in AI Modeling**: **@lhc1921** highlighted the specificity required in chunking embeddings—there's no one-size-fits-all approach, and **@mrdragonfox** reinforced that it depends on dataset characteristics.

- **Recommendations for JSON & Grammar Within AI**: Users suggested improved results in JSON output generation by prompting with examples and discussed integrating grammar via examples on a platform, with **@gbourdin** sharing a related [link](https://horosin.com/extracting-pdf-and-generating-json-data-with-gpts-langchain-and-nodejs).

- **Fine-Tuning Finesse**: Questions arose regarding fine-tuning LLMs, from creating conversational datasets for beginners to addressing challenges in models for low-resource languages. **@yashkhare_** referenced research on using LoRA for pretraining and **@mrdragonfox** voiced concerns about inadequate fine-tuning methodologies compared to instruct versions of models.

- **Mistral's Successful Implementations & API Inquiries**: **@soletum** reported user satisfaction with integrating **Mistral** within their product, and inquiries about **API key activation times** were noted, suggesting immediate contact with Mistral support.

- **Dedicated Endpoint Discussions on La plateforme**: Queries about the pricing of dedicated endpoints for big companies led to suggestions of direct contact with developer relations for precise figures, considering the costs could be significantly higher than raw GPU expenses.

- **Office-Hour Insights Bring Clarity and Requests**: Engineers sought advice on system prompts and the potential of Mixtral-8x7B-Instruct-v0.1, discussing the nuances of instruct models and user message identities. Feature requests for the Mistral API indicated a keen interest in enhancements.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Rethinking RMT with Weight Freezing**: The concept of **weight freezing** in an approach somewhat similar to **RMT** was highlighted as innovative by `@hexani`, despite not being the main focus of the referenced material.

- **Potential Release of Qwen2 Causes Buzz**: Anticipation around the `Qwen2` release grew as it appeared briefly on a leaderboard before going offline, as mentioned by `@nonameusr` and `@weyaxi`. Meanwhile, a "llamafied" version of `Qwen-72B` raised tokenizer performance concerns, with the relevant model found on [Hugging Face](https://huggingface.co/Weyaxi/Qwen-72B-Llama).

- **Training Leveraging 4-bit Optimizer States**: A [thu-ml study](https://github.com/thu-ml/low-bit-optimizers/) suggests training neural networks with **4-bit optimizer states** can dramatically reduce memory footprint, as outlined in their [paper](https://arxiv.org/abs/2309.01507).

- **Enduring Value of N-gram Models**: An 'infinite-gram' model was proposed as relevant in the neural LLM era, potentially scaling beyond traditional n-gram limits as per a shared [paper](https://arxiv.org/abs/2401.17377).

- **Anthropic Raises Alarm on Sleeper Agents in LLMs**: A potential fastidious problem with backdoor training was discussed, with *sleeper agents* being able to persist post safety training, based on an [Anthropic article](https://www.anthropic.com/news/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training).

- **Project Obsidian Goes Dormant**: **Project Obsidian's** focus on **multimodality** was clarified with the project being essentially complete and released, as pointed out by `@teknium` and available on [Nous' Hugging Face](https://huggingface.co/Nous).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Custom Pipelines and Creativity Benchmarks**: A custom pipeline for `vikhyatk/moondream1` has been [introduced on Hugging Face](https://huggingface.co/vikhyatk/moondream1/discussions/6), and an evaluation benchmark for model creativity has been proposed, with discussions [on Twitter](https://twitter.com/Vipitis/status/1752699776766988309). A Resume QA space and Han Instruct dataset were [launched](https://huggingface.co/spaces/not-lain/resume-qa), and the Ukrainian wav2vec2 bert model has been [released](https://huggingface.co/Yehor/w2v-bert-2.0-uk).

- **Celebrating Model Milestones and LLM Hosting Discussions**: Downloads for `moondream1` hit 2500, video generation tools similar to `moonvalley` were discussed, API usage for models like `llama2` on Hugging Face's servers was clarified, discussions around free LLM hosting for projects, and choosing novel datasets for academic pursuits [ensued](https://arxiv.org/abs/2311.07989).

- **Knowledge Sharing and Remote Employment Proposals**: Free API access for LLMs, a guided tour to pretrained LLMs [were highlighted](https://docs.google.com/presentation/d/1TMRpL52pkz8ULSJvxaCdsrqROW4w8TkfA5GQ_VCPkhQ/edit?usp=sharing), and a **transnational employment strategy** was suggested for U.S. citizens to collaborate with an international web developer.

- **New Diffusion and Clustering Models**: A new diffusion model from Google, **MobileDiffusion**, was announced, potentially outperforming Stable Diffusion and DALL·E. [EfficientNet models](https://www.akshaymakes.com/blogs/pytorch) were suggested to help with clustering models to identify user actions.

- **Innovations from Community Creators**: **ColBERT's live visualization tool** hosted at [colbert.aiserv.cloud](https://colbert.aiserv.cloud) was shared. The **UltraTextbooks** dataset was released on [Hugging Face](https://huggingface.co/datasets/Locutusque/UltraTextbooks), and **VS Code integration with Jupyter** on a remote GPU was guided through a [Medium post](https://medium.com/@chongdashu/connecting-visual-studio-code-with-jupyter-notebooks-on-a-remote-gpu-server-instance-8f7cc0696a45).

- **Readings and Presentations About AI and Law**: The **Mamba** presentation was set for a reading group discussion with scheduling through a [When2meet link](https://www.when2meet.com/?23471427-n4DUl), and recording inquiries were made for a presentation on AI challenges in law.

- **Dreambooth Training and TPU Inquiry**: The **Dreambooth LoRA** training for Stable Diffusion is now available, with advanced capabilities revealed in a [GitHub script](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sd15_advanced.py). A question about TPU compatibility for such training [remains unanswered](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968).

- **AWS SageMaker and CUDA Toolkit Installation Discussed**: Guidance was sought for using **AWS SageMaker** with **Mistral**, **Llama**, or **Tapas**, and installation issues possibly related to the CUDA toolkit version were reported. 

- **Popup Modal Component for Gradio**: The `𝚐𝚛𝚊𝚍𝚒𝚘_𝚖𝚘𝚍𝚊𝚕` was released to enhance Gradio apps with pop-up features and can be explored on the [Hugging Face Space](https://huggingface.co/spaces/aliabid94/gradio_modal).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **LLaMa 1.6 Gains Quiet Applause**: User `@kache_` indicated **LLaMa 1.6** is performing well, though no specific performance metrics or comparisons were given, keeping the discussion light and mysterious.

- **Community Writeup Earns Hacker News Spotlight**: A member's Reddit writeup was spotted making rounds on Hacker News, amplified by a fellow user who provided a [link to the thread](https://news.ycombinator.com/item?id=39215242).

- **Controversy over Bard’s Watermarked Images**: `@max_voltage` sparked a debate by criticizing Bard's image-generation feature for embedding watermarks, a step towards responsible AI, and pointed to a broader conversation, suggesting a clash between creativity and ethics.

- **Imagen 2 Images Show Unexpected Noise**: `@thejonasbrothers` raised concerns about the noise levels in images generated by Imagen 2, indicating potential quality issues and inefficient output formats when compared to SDXL's imagery.

- **Deep Dive into Autoencoder's Latent Space Sensitivity**: A detailed discussion unfolded between `@drhead` and `@thejonasbrothers` on the nuances in the latent space of autoencoders, touching on Segmind Vega and SDXL VAE models, and the training's impact on noise pattern evolution.

- **OLMo Receives a Shoutout**: User `felfri_` merely dropped a [link to OLMo by the Allen Institute for AI](https://allenai.org/olmo), without context, but presumably as a hint towards innovative research worth looking into for the engineering minds.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Long-Term Memory for ChatGPT Sparks Interest**: `@flyyt4` sought advice on enhancing ChatGPT's long-term memory, with `@fabguy` suggesting "memGPT" without providing a specific source or additional context.

- **Prompt Crafting Techniques Shared**: Advanced prompting techniques like Chain of Thought and Tree Thinking were highlighted by `@wolfspyre`, including a [link to advanced prompt examples](https://funkpd.com/devlog/prompt-examples/).

- **Exploring Unrestricted Models**: Users discussed the performance and utility of uncensored models such as The Bloke Goat Storytelling 70B Q3KS, with a focus on story-writing applications and avoiding smaller models.

- **Challenges with GGUF and Model Adapters**: Mixed feedback surfaced on the usability of the GGUF format and locating model adapters within LM Studio, with constructive tips shared for locating and editing models on the disk.

- **Hardware Concerns for Model Performance**: Users like `@silverstar5654` and `@docorange88` inquired about leveraging dual RTX 3090s, RTX 6000s, or NVLink setups for running large models like Mixtral and discussed potential performance with multi-GPU configurations.

- **Inquiries About llava 1.6 and Autogen's Server Capabilities**: Support for recognizing llava 1.6 as a vision model in LM Studio is in question, and there is curiosity about running Autogen on multiple servers and whether it's possible to operate dual instances of LM Studio on separate ports.

- **Integration Issues and Cost Concerns with Open Interpreter**: User `@raiderduck` reported nonsensical responses when using OI with LM Studio and cited high costs when using GPT-4, amounting to $250 in a week, prompting a search for cost-effective server settings.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **OLMo Steps into the LLM Ring**: A newly released [OLMo paper](https://allenai.org/olmo/olmo-paper.pdf) has initiated a lively debate on its approach and licensing, with AI2 clarifying that no chat model or interface has been released yet. Questions were raised about OLMo's benchmarks, training code, particularly its warmup scheduler, normalization choices, and AI2's evaluation metrics, with discussions on tokenizers, layer norm usage, and potential benchmark cherry-picking stirring the technical crowd.
  
- **Trillions of Tokens in N-gram Models Discussed**: The Eleuther community debated the value of scaling [n-grams to trillions of tokens](http://arxiv.org/abs/2401.17377) and its integration with LLMs, pondering over potential performance boosts and generalization. The limitations and applications of the Infinigram model, the potential of tagging backtranslated data for LLMs, and synthetic data generation strategies, like MCTS, also formed part of the rigorous research discourse, showcasing the relentless pursuit of enhancing AI efficiency.

- **Innovations and Contributions Celebrated**: Updates on the AI research landscape were shared, with the release of the Gaussian Adaptive Attention library for multi-modal work drawing interest. The dynamic nature of AI research was further highlighted by announcements of works such as [Amortized Text-to-Mesh (AToM)](https://snap-research.github.io/AToM/) and others, disseminating the latest in visual representation learning and model editing techniques.

- **In-Context Learning and Neural Circuits Analyzed**: Insights into in-context language learning (ICLL) culminated in the discovery of a [relevant research paper](https://arxiv.org/abs/2401.12973) and discussions on [contextual neurons](https://arxiv.org/abs/2311.00863) within language models. The group also debated the effectiveness of ∞-gram models in perplexity versus open-ended text generation, and compared the challenges faced in ∞-gram retrieval with MoE model attacks, referencing a [study on the latter](https://arxiv.org/pdf/2210.10253.pdf).

- **Brief Confirmation in Thunderdome**: Within the Thunderdome's sparse chatter, a concise response from `daniellepintz` simply confirmed the non-existence of a `limit` without further elaboration or context.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Reddit Sounds the Alarm on VAEs**: A [Reddit post](https://www.reddit.com/r/StableDiffusion/s/p2BkzI9ypO) unveiled a critical issue in the KL divergence loss of VAEs used in models like **SD1.x, SD2.x, SVD, DALL-E 3**, which could be impacting efficiency by transmitting global information through minimal pixels.

- **VAE Simplicity Catches Eye**: A VAE-related Reddit discussion noted by `@swyxio` piqued interest for its straightforwardness, suggesting it was refreshingly concise compared to typical academic publications that might include "unnecessary fluff."

- **LLMs Spark Memes and More**: `@natolambert` shared a [link](https://twitter.com/natolambert/status/1753063313351835941) to **LLM memes**, highlighting the community's humorous engagement with language models, while AI2's Open Language Models (**OLMo**) were discussed for their hardware diversity and short context length limitation.

- **Open Source Gains a Player**: **Nomic AI** released a set of open-source embeddings that exhibited high performance on the LoCo benchmark, accompanying the launch with a [detailed blog post](https://www.interconnects.ai/p/olmo).

- **Latest Latent Space Podcast Lands Quietly**: A new episode of the **Latent Space Podcast** was released as announced by `@swyxio` on [Twitter](https://twitter.com/latentspacepod/status/1753120715254198425) and Hacker News, though it struggled to gain traction despite clickbait strategies.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Ethical Quandaries and AI Role Confusion**: A discussion initiated by `@yami1010` dealt with the ethical parameters of AI, prompting a desire for resources about the nuances in LLMs, and touching upon the often nebulous definitions of responsibility in AI development.

- **Debating the ChatGPT Smarts Slide**: `@voidrunner42` and `.@tadase.` engaged in a conversation about ChatGPT's performance, debating whether a perceived decline in smartness is due to a real decrease in capabilities or just the nature of the prompts provided.

- **AI Credit Where Credit Is Due**: `@movoza` corrected a misattribution where Bing appeared to claim sole credit for developing an AI, igniting a discussion on the collective effort in AI advancements and the intricate history involving multiple stakeholders.

- **Trolling the GPT-3 School Bot and Beyond**: `@qiqimon` asked about the feasibility of GPT-powered chatbots in school customer service, while others noted the importance of measures to prevent misuse, suggesting this use case is within reach but not without its challenges.

- **Tech Talk on GPT-3 Data Formats and Auth Issues**: Opinions were shared on the optimal file formats for knowledgebases to feed into GPT, specifically preferring RSV over JSON and XLXS, and `@woodenrobot` recounted troubles with API authentication during their project's transition from alpha to public beta, highlighting real-world issues AI engineers face when scaling solutions.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Model Concerns and Solutions Spread Across Channels**: Discussion in the **general** channel raised concerns about code changes potentially impacting models like Llama and Mistral, but a fix was noted to have been merged upstream in the **axolotl-dev** channel. Specific to **Mixtral**, users in the **general-help** channel shared advice on finetuning approaches depending on GPU VRAM, and discussed finetuning the entire model with **Zero3** offloading.

- **Performance Discussions for 7B Models**: There's been talk about the stagnation of 7B models, but names like `CapybaraHermes-2.5-Mistral-7B` and `Eagle 7B` have been mentioned, along with performance updates and links to results, highlighting the active competition in this space. [Eagle 7B on Hugging Face](https://huggingface.co/RWKV/v5-Eagle-7B), [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), and [CapybaraHermes-2.5-Mistral-7B](https://huggingface.co/argilla/CapybaraHermes-2.5-Mistral-7B) were cited.

- **Exploring Text Embedding and Vector Databases**: There was a technical exchange about text embedding models and vector databases, with **nomic-embed-text-v1** being notable, as well as discussions on starting with **bge** and utilizing GPUs in cloud services. For further exploration, [qdrant/fastembed on GitHub](https://github.com/qdrant/fastembed) and [Improving RAG by Optimizing Retrieval and Reranking Models](https://docs.argilla.io/en/latest/tutorials_and_integrations/tutorials/feedback/fine-tuning-sentencesimilarity-rag.html#Bi-Encoder-Model) were shared.

- **RunPod Service Challenges Contested**: In **runpod-help**, complaints surfaced about sudden pod shutdowns, data loss, and communication mix-ups about pod deletion times. In addition, issues with SSL errors during data transfers and slow download speeds prompted discussions about the reliability and efficiency of RunPod services, with the RunPod Discord suggested as a place to seek help: [RunPod Discord Help Channel](https://discord.com/channels/912829806415085598/1187492973148115076).

- **Unanswered Mistral Configurations**: A lone message in the **shearedmistral** channel by `dangfutures` highlights that there's ongoing curiosity or confusion regarding the configurations for Mistral, perhaps indicative of a broader conversation or need for clarification within the community.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **AI Amusement Shared Across Channels**: User `@tbyfly` was directed to share a humorous response from Perplexity AI in another channel after their initial inquiry in the general chat.

- **Navigating Model Complexities**: The PPLX models, promoted for generating up-to-date and factual responses, were discussed; however, users `@brknclock1215` and `@jayb1791` noted limitations of the **7b-online model** with complex queries and privacy overreach for SQL assistance. Meanwhile, `@general3d` suggested improvements for the **codellama 70b model** to ensure more consistent answers.

- **Perplexity as a Default Search**: `@bartleby0` provided a solution for setting Perplexity AI as the default search engine, while separately mentioning Arc Search as a potential competitor.

- **Intriguing AI Applications and Observations**: New member `@.sayanara` linked to a blog post and book about AI's potential to mitigate misinformation ([The Right Direction for AI](https://figmentums.com/2024/02/01/the-right-direction-for-ai/)). Elsewhere, an absence from a list of top apps led `@bartleby0` to note Facebook's lack of presence as "interesting."

- **Subscription Snafus and API Anomalies**: `@dame.outlaw` requested assistance with subscription issues, and `@alankarsh` reported problems with API credits post subscription, revealing a hiccup in user experience.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Open Source Rival to OpenAI Embeddings**: A new open-source text embedding, *nomic-embed-text-1* by `@nomic_ai`, has been introduced by LlamaIndex, showcasing better performance than OpenAI's text-embedding-3-small and includes [integration details on Twitter](https://twitter.com/llama_index/status/1753106179008696521).
- **Keynote Replay on Agentic RAG**: The keynote by @jerryjliu0 on *Beyond Naive Rag: Adding Agentic Layers* is now available to watch as a replay on [YouTube](https://t.co/hrUMF8bq8Q), with the slides accessible [here](https://t.co/P39riIMGK6).
- **Llama Index Expands Compatibility**: LlamaIndex's guide for integrating other LLMs, such as using Llama-2, is highlighted with a [guide](https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#modules) and an [example notebook](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/llama_2_llama_cpp.ipynb), while also addressing issues regarding Postgres connections and MongoDB in an active discussion.
- **Seeking PII Anonymization for LLMs**: The quest to anonymize personally identifiable information from text datasets efficiently through the langchain and Presidio shows a need for production-level solutions, as the current approach remains experimental.
- **Speech-to-Text Deep Dive**: Insights into OpenAI's Whisper model are explored in a detailed [blog post by @amgadoz](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b), focusing on its encoder-decoder transformer architecture and its relation to the "Attention is All You Need" paper.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Beware of LangChain AI Twitter Scam**: Multiple users, including `@markopolojarvi`, `@alcazarr`, and `@solac3`, reported a possible security breach involving **LangChain AI's Twitter account**. A suspicious [tweet](https://twitter.com/LangChainAI/status/1753014882696405254) led to speculations of hacking.

- **Innovative Autonomous GPT-4 Agent Platform Introduced**: `@robot3yes` unveiled **Agent IX**, a standalone GPT-4 agent platform, and is encouraging community exploration on [GitHub](https://github.com/kreneskyp/ix).

- **ContextCrunch Streamlines Token Efficiency**: `@speuce` promoted **ContextCrunch**, a prompt compression API designed to reduce token costs, with early access and further details available at [contextcrunch.com](https://contextcrunch.com/).

- **LangServe and Llamacpp Aim for Streaming Capabilities**: `@veryboldbagel` and `@legendary_pony_33278` engaged in discussions on enabling streaming with **Llamacpp** and **Langserve** or **FastAPI**, with a shared [GitHub example](https://github.com/langchain-ai/langserve/blob/main/examples/local_llm/server.py) demonstrating LangServe with ollama.

- **Exploring Step-Back Prompting and Context Compression**: Methods for enhancing language model interactions, such as Step-Back Prompting and **Contextual Compression** using **LangChain**, were detailed in a [Medium article](https://medium.com/@mrk5199/how-to-compress-llm-contexts-with-langchain-2b58eb84f57b). This compression technique is discussed as a potential solution to excessive token usage in RAG setups.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Switching to Mixtral for Bigger Gains**: `_jp1_` confirmed the **migration from a 7b model to Mixtral**, citing the latter's proficiency for evaluation tasks and extended an invitation for assistance in the transition process.
- **API Exposed**: `@sebastian.bodza` raised an alarm about the **API lacking security measures** as it's currently not leveraging tokens in requests, posing a security risk. 
- **Nomic Outperforms OpenAI**: Showcased by `@bjoernp`, the **Nomic's embedding model**, `nomic-embed-text-v1`, holds sequence length supremacy with 8192 and outshines its OpenAI counterparts, is freely available along with its weights and training code at [Nomic AI on Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1).
- **OLMo 7B Enters the Open Model Arena**: The introduction of **OLMo 7B**, the Open Language Model by Allen AI presented by `_jp1_`, comes with its dataset, training resources, and a research paper, which are accessible at [OLMo on Allen AI](https://allenai.org/olmo/olmo-paper.pdf) and [OLMo on Hugging Face](https://huggingface.co/allenai/OLMo-7B).
- **GPU Workarounds for Inference and Shared Resources**: In the absence of GPUs, `_jp1_` proposed inference alternatives using services like **replicate, modal, or serverless runpod**, and hinted at a possible group hosting if needed while sharing a potentially useful but unspecified Google Colab notebook at [this link](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK).



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Nomic Embed Leads the Pack**: The new **nomic-embed-text-v1** from [HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1) claims superior performance to similar OpenAI models, boasting scores of 62.39 on MTEB and 85.53 on LoCo. Interest is shown in comparing embedding models in a competitive leaderboard akin to the LLM arena.
  
- **Seeking Deployment Guidance**: A Discord user, firefox8975, is looking for advice or a guide to deploy open-source machine learning models to AWS using VLLM, indicating a practical problem-solving discussion among the members.

- **Exa Announces Launch Events**: **Exa** (formerly Metaphor) is hosting toga-themed launch parties in San Francisco on **February 2** ([RSVP SF party](https://partiful.com/e/7qDnQGjE1MdU32Cei0J0?)) and New York City on **February 7** ([RSVP NYC party](https://yuzu.party/1FEOWfzNCHm3Fi6vtrSs)). `@sarahchieng` also kindly offers to buy coffee for those interested to discuss Exa in both cities.

- **Vector Database Migration Underway**: In the realm of vector databases, a user `@michelcarroll` shares their transition from **Weaviate** to **pgvector with HNSW**, providing insight into the practical application and migration of data platforms.

- **Chain of Thought Persistence Examined**: There's a conversation about the tradeoffs of saving and reusing a language model's Chain of Thought (CoT). While the reuse of CoT can reduce costs and processing time, it was highlighted that doing so incurs a latency tradeoff due to the necessity of multiple API calls.



---



## [CUDA MODE (Mark Saroufim)](https://discord.com/channels/1189498204333543425) Discord Summary

- **CUDA Kernel achieves Warp-Speed**: `@zippika` implemented a CUDA kernel to convert **rgb_to_grayscale**, optimizing it to utilize **ulong** for vectorized loads, which increased GPU *occupancy* to **77%** and *memory* utilization to **71.87%**. However, despite theoretical improvements, the kernel was slower in practice, evidencing the complexity of optimization. [See the optimization details here](https://example.com/link-to-code).
  
- **Jumping into CUDA with C++ Grounding**: User `@noobpeen` has successfully set up **CUDA 12.2 with Visual Studio** and plans to leverage their knowledge of PyTorch and C++ to start CUDA development. Experienced user `@lancerts` suggested starting with a new CUDA project, specifically developing CUDA kernels, and studying the book *Professional CUDA C Programming* for a deeper dive.

- **Employing Thread Coarsening for Efficiency**: In the discussion, `@tvi_` emphasized the benefits of **thread coarsening** to increase work efficiency by reducing global memory load frequency, a principle that speaks to more than just compute efficiency but also memory optimization as examined in the *PMPP* book.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Datasette's Documentation Meets GPT**: @simonw experimented with inputting the **PDF version** of the [Datasette documentation](https://docs.datasette.io/en/stable/) into a **GPT**, but found the initial results lacking. They retained some optimism about the technique's potential after more refinement.
- **Pursuing 'Quiet AI' Discussed**: One post introduced an external article concerning the pursuit of 'Quiet AI', however, no substantial discussion or details were provided. [Read the article here](https://www.dbreunig.com/2024/02/01/pursuing-quiet-ai.html).





---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1202529153241452575) (1279 messages🔥🔥🔥): 

- **Discussing Miqu-70B's Performance**: Users like `@netrve` and `@mrdragonfox` shared experiences with the Miqu-70B model and various versions such as Mistral Medium, with some commenting on its unexpected ability to self-correct during responses. The conversation touched on aspects like model speed (`@goldkoron` mentioned getting on average 17t/s after some tweaks) and comparison to other models like Mixtral.

- **Conversations on Programming Languages**: A discourse around programming languages unfolded, debating the merits and drawbacks of C++, Rust, Go, and other languages. `@mrdragonfox` praised the simplicity of C and stated that not all programming needs classes. The discussion touched on advanced features of C++ and the simplicity offered by Rust and Go, with `@rtyax` and others expressing a preference for simpler, more understandable languages.

- **Chatbot UI Discussion**: `@righthandofdoom` expressed interest in a simplistic web UI or native UI that could connect to remote OpenAI-compatible endpoints, with `@coffeevampir3` and `@animalmachine` suggesting alternatives like Hugging Face's "Candle" and lamenting issues in creating frontends with current tools.

- **Speculative Decoding and API Tools**: `@.justinobserver` and others discussed speculative decoding and which current tools support it, with mentions of llama-cpp and the desire for `@flashmanbahadur` to find an API tool that supports exl2 and an OpenAI compatible API. `@itsme9316` suggested "tabbyapi" for its OpenAI compatible API with exl2.

- **Finetuning and Calibration Misunderstandings**: Users like `@turboderp_` and `@giftedgummybee` addressed misunderstandings about the purpose of calibration in the context of finetuning. The conversation delved into the challenges of using quantization as a way to retain model quality, with `@giftedgummybee` facing issues with loading parquet files and considering reliance on default quant settings.
  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1202524767907745802) (665 messages🔥🔥🔥): 

- **Fit for Chat Finetune?**: `@kaltcit` pointed out that a llama resume trained (`<@266127174426165249>`) is good at ERP, however, without ghost attention.
- **Miqu API Optimism**: `@righthandofdoom` offered others the opportunity to use Groq API with his ratelimited API key for Mixtral experiments.
- **Model VRAM Dilemma**: `@kaltcit` faced issues trying to fit Miqu-1-70B-SF-4.25bpw-h6-exl2 within available VRAM, discussing with `@turboderp_` about potential solutions and even considering hardware upgrades.
- **Experimenting with New Tasks**: Participants discussed the outcomes observed from newly attempted tasks, such as detailed character RP, longer token contexts, with models like Miqu and LLMs, seeking optimal performance and grappling with hardware limitations.
- **LLM Performance Insights Shared**: Various users like `@mrdragonfox`, `@doctorshotgun`, and `@goldkoron` discussed different settings and configurations to optimize the speed and performance of language models on given hardware, considering factors like VRAM, model bits-per-weight, and context sizes.

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1202666974606266409) (26 messages🔥): 

- **Challenges in Predictive Model Training**: `@hapticalacrity` is seeking advice on training a model to predict 8-digit hierarchical codes from text strings, with 25 examples for each subcategory. **Distilbert-base-uncased** was tried for the task, showing accuracy at higher levels but performing poorly at predicting the last 2 digits of the codes.
  
- **Innovative Alternatives for Model Improvement**: `@tom_lrd` advised considering methods like **recursive clustering with embeddings** instead of costly model training, which could efficiently handle the category prediction without relying on heavy hardware resources.
  
- **Utilizing Natural Language Models for Category Classification**: `@tom_lrd` suggested employing **prompt engineering** with large models such as mistral7b to classify text into categories in a hierarchical manner, despite acknowledging it as a more resource-intensive approach.
  
- **Rapid Resolution of Training Queries**: `@hapticalacrity` expressed gratitude to `@tom_lrd` for providing concise, helpful feedback on the classification problem, highlighting the collaborative aspect of the **TheBloke** Discord community.
  
- **Model Usage Post Fine-Tuning**: `@chovii` encountered a **ValueError** when attempting to use a fine-tuned model and sought assistance for the correct procedure after encountering **issues when merging LORA layers.**
  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1202879708090343476) (2 messages): 

- **InternLM Recommended for Limited Resources**: `@alphaatlas1` suggested using **InternLM 20B** for those who might not be able to run the larger 34B models. This advice is tailored to users with resource constraints.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1202676909591363625) (12 messages🔥): 

- **Debate on the Merits of Search vs File Organization**: `@wbsch` argued that **good search capabilities** in software can be more effective than traditional file organization methods like tags or directories. They mentioned using systems like nvpy and simplenote since 2008 for their note-taking needs.
  
- **The Necessity of File System Knowledge for IT**: `@wbsch` agreed with `@Splice` affirming that understanding files and file systems is essential for IT, although not so much for organizing files, where good search functionality has its advantages.
  
- **File Systems in User Interfaces**: `@wbsch` elaborated that for user interfaces, good search is essential and should be complemented by methods like "recently changed files" to reduce user effort, while mentioning IDEs benefit from function search features for efficiency.

- **Practical Solutions to Dynamic Linking Issues**: `@spottyluck` shared an issue with a missing `libllama.so` library when running a `main` command and demonstrated the use of `patchelf` to solve the problem by specifying the library's path, and then `LD_PRELOAD` as an alternative if the library is removed.

- **Good Maintenance Practices Recommended**: In the context of missing shared libraries, `@spottyluck` advised fixing the build process after applying temporary fixes like `patchelf` to ensure that the correct library linking is established.
  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1202543162514739200) (178 messages🔥🔥): 

- **GPU Shortages Affecting Model Training**: `@frosty04212` discussed the lack of GPU availability affecting the production of more units, attributing the shortage in part to the complexities of semiconductor manufacturing which involves long processes and deep investments. `@mrdragonfox` expanded on the barriers to scaling up production such as the months-long wafer manufacturing cycle, extensive cleanroom requirements, specialist workforce needs, and the collaboration between companies like TSMC, ASML, and Zeiss.

- **The Open Source Debate**: An active conversation about the interplay between open source and proprietary software occurred, with users like `@frosty04212`, `@ethux`, and `@firesonwires` discussing whether open source technology can compete with closed source, noting that open source often relies on contributions and funding from larger companies. The "open core model" was mentioned as a positive hybrid approach, implying that while some elements can be open and free, monetization can occur through premium features.

- **Mistral's Public Relations Strategy**: User `@_dampf` praised Mistral for being transparent regarding the leak of an early access version of the Mistral Medium model, instead of remaining silent. `@ethux` added that the leaked model was not the latest version, and `@mrdragonfox` mentioned that leaked models, such as MIQU, wouldn't receive official support.

- **Mistral API Key Activation Inquiries**: User `@cptplastic` enquired about the activation time for new API keys at Mistral after facing a rate limit issue. `@mrdragonfox` suggested emailing Mistral support for any activation troubles, while `@i_am_dom` affirmed that API keys should have immediate and reasonable limits.

- **Predictions and Musings on Future Models**: Speculative chatter about what Mistral might do next surfaced with `@gtnbssn` jesting about a model named miqtral or miqstral that might use Q-learning. Opinions and jokes about upcoming technologies were shared among users, reflecting curiosity and excitement for future model iterations.
  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1202667649704927283) (4 messages): 

- **Chunk Size Matters in Embeddings**: `@lhc1921` humorously admits to being pedantic after noting that there are definitely wrong ways to chunk embeddings in certain use cases.
- **Data Dictates Chunking Strategy**: `@mrdragonfox` emphasizes that how to chunk embedding documents is data-dependent, implying that there isn't a one-size-fits-all chunk size that will work across different datasets.
  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1202627410953117746) (4 messages): 

- **Grammar Integration on La Platforme?**: `@superintendent` queried about integrating grammar on the platform, mentioning an approach of providing examples to the system.

- **Prompting Bests Direct Instruction for JSON Schema**: `@gbourdin` recommended prompting the chatbot with examples to yield desired outputs, sharing a [link](https://horosin.com/extracting-pdf-and-generating-json-data-with-gpts-langchain-and-nodejs) that demonstrates using JSON schema with examples in the prompt for effective results.

- **Size Matters for JSON Output**: `@samy7418` observed that generating JSON output with few-shot examples is possible on medium-sized models, but the small-sized counterpart tends to add explanatory text to the JSON output.

- **Seeking Context in Discord Links**: `@akshay_1` replied to `@samy7418` with a link to a specific message in the Discord channel, presumably for context or further information.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1202543151584387092) (44 messages🔥): 

- **Seeking Guidance on Fine-tuning**: User `@d.j147` is new to the field and is seeking advice on how to create a conversational dataset and fine-tune large language models (LLMs).
- **Challenges with Low-Resource Languages**: `@quicksort` shares that fine-tuning LLMs for languages that make up less than 5% of pretraining corpora is resulting in subpar language quality, and is asking for insights or success stories about continuously pretraining Mistral with low-resource language datasets.
- **LoRA Pretraining Exploration**: `@yashkhare_` discusses the intention to pretrain using LoRA, referencing a [research work](https://arxiv.org/pdf/2304.08177.pdf) and wondering about the number of tokens and languages used in the Mistral-7b paper’s training.
- **Frustrations with Mistral Training Methods**: `@mrdragonfox` expresses that the community has not yet figured out an optimal way to train Mistral, mentioning that attempts so far do not compare to the base instruct version of language models.
- **Axolotl Training Query**: User `@woodenstick_` is encountering an `IndexError` while using Axolotl to train Mistral and has shared the configuration snippet possibly linked to the error.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1202620605279637534) (16 messages🔥): 

- **Mistral Receives Positive Feedback**: `@soletum` informed the channel that they implemented **Mistral** within their product, offering a text correction and generation interface based on keywords. Their initial users are satisfied with its usefulness.
- **Community Encouragements**: `@frammie` expressed admiration for the implementation of Mistral, describing it as looking "very nice, awesome!"
- **Showcase of AIDocks**: User `@lhc1921` shared a link to the GitHub repository for **AIDocks** with the [GitHub Link](https://github.com/l4b4r4b4b4/AIDocks).
- **Friendly German Exchange**: Conversational exchange in German took place between `@mrdragonfox` and `@lhc1921`, discussing the presence of German-speaking community members on the platform.
- **Hugging Face Chatbot Suggestion**: In response to `@jamiecropley` asking for a platform to use **Mistral** as a chatbot, `@ethux` recommended checking out [Hugging Face Chat](https://huggingface.co/chat).
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1202564142490325032) (26 messages🔥): 

- **In Search of a Price Tag for Dedication**: `@gbourdin` inquired about the availability and cost of dedicated endpoints on La plateforme, seeking a monthly price quote for a big company's estimate. Despite no specific figures provided, they mentioned considering the inference endpoint pricing from HuggingFace as a reference.
- **Revenue Benchmarks for Custom Endpoints**: `@mrdragonfox` suggested that custom dedicated endpoints typically require an enterprise to have at least $1 million in annual revenue, emphasizing that such services are substantially pricier.
- **Direct Contact for Accurate Estimates**: It was recommended that `@gbourdin` reach out to `support@mistral.ai` or directly to Sophia, who is in developer relations, to get precise pricing, especially since `@mrdragonfox` clarified they do not represent Mistral.
- **Understanding the Scale of Costs**: `@mrdragonfox` indicated that the enterprise deployment of Mistral's state-of-the-art model, if available, would likely be far more costly than the raw GPU cost—roughly ten times more by their estimation.
- **Initial Steps Towards Partnership**: `@gbourdin` considered starting with the regular endpoint of La plateforme, accepting that a minimum budget for customizing endpoints would likely begin around $10k per month, and planned to further discussions with the potential enterprise client based on this information.
  

---


### Mistral ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/1202644588867879042) (240 messages🔥🔥): 

<ul>
<li><strong>Clarity Requested for System Prompts</strong>: `@sa_code` sought guidance on using system prompts with Mistral, referencing a Jinja template in their PR, and noting the absence of system prompt support in the Mixtral chat template on Hugging Face. They provided links to the documentation and their PR for reference.</li>
<li><strong>Seeking Mixtral's Full Potential</strong>: `@touristc` inquired if Mixtral-8x7B-Instruct-v0.1 has reached its full parameter potential or if there's room for improvement. `@sophiamyang` responded, indicating that there's always room for improvement.</li>
<li><strong>Instruct Models Explained</strong>: `@damiens_` queried the difference between instruct and non-instruct models, to which `@sophiamyang` replied that instruct models are trained to follow instructions better. The topic was further discussed with links to an explanatory video and guide posted by `@canyon289` and an original paper link shared by `@sandeep3890`.</li>
<li><strong>Message Identity Complexity Discussed</strong>: `@jakobdylanc` pondered the complexity behind assigning identities to user messages. `@lerela` explained that the process isn't overly complex but requires dataset preparation and fine-tuning.</li>
<li><strong>API Feature Requests and Hints of Future Enhancements</strong>: Participants in the office-hour made various feature requests for the Mistral API, such as JSON output support, logprobs and logit bias, function calling, and improved support for API parameters like openai. Comments from `@lerela`, `@sophiamyang`, and `@nicolas_mistral` hinted at ongoing work and future releases that could address some of these requests.</li>
</ul>
  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1202524801411981342) (3 messages): 

- **Retraction of Previous Statement**: `@hexani` retracted an earlier statement, mentioning that the approach discussed is somewhat similar to **RMT** but interestingly **freezes weights**, which they found noteworthy.
- **Innovation Highlighted in Weight Freezing**: `@hexani` clarified that while the method discussed shares similarities with **RMT**, the innovative aspect lies in the **weight freezing** rather than the main point of the referenced material.
- **Wavenet Vibes from Graphs**: `@gabriel_syme` observed that the graphs discussed bear resemblance to **Wavenet** or something similar, suggesting a parallel in the visual data representation.
  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1202624026988519475) (12 messages🔥): 

- **Search for Streamlined Notion Templates**: User `@faiqkhan` inquired about **minimalistic Notion startup templates**, expressing a preference for free options since most of the good ones appear to require payment.
- **Sharing Dance of the Quackduckflower**: Error.PDF posted a link to a GIF with the title "Quackduckflower," providing a [visual amusement](https://tenor.com/view/quackduckflower-gif-25482704).
- **Inquiry for Dingboard Access**: `.benxh` asked who to contact for access to **dingboard**, and `random_string_of_character` recommended sending a direct message on Twitter to Yacine.
- **Interactive Fiction Adventure with 'Her'**: `@everyoneisgross` shared a [ChatGPT-enhanced interactive version](https://chat.openai.com/g/g-sAg0WI4ey-intheractive) of the movie script for "Her," allowing users to engage with the story sequentially.
- **Discussion on CLIP Embedding Phenomenon**: `@cccntu` started a conversation about the phenomenon where **CLIP embeddings** only maintain partial meaning of text due to contrastive loss not needing to retain every detail.

**Links mentioned**:

- [Cat Driving Car GIF - Cat driving car - Discover &amp; Share GIFs](https://tenor.com/view/cat-driving-car-gif-17184928271354710905): Click to view the GIF
- [Quackduckflower GIF - Quackduckflower - Discover &amp; Share GIFs](https://tenor.com/view/quackduckflower-gif-25482704): Click to view the GIF

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1202637540234887198) (14 messages🔥): 

- **Pushing the Limits of Optimizer Memory Efficiency**: Research from [thu-ml](https://github.com/thu-ml/low-bit-optimizers/) presents the capability of training neural networks with **4-bit optimizer states**, potentially reducing the memory footprint of model training. The study explores detailed empirical analysis and proposes new quantization strategies, described in their [paper](https://arxiv.org/abs/2309.01507).

- **Demystifying Whisper Model for Speech-to-Text**: `@amgadoz` wrote a detailed blog post about OpenAI's **Whisper** model, discussing its architecture and how it transcribes audio to text. The insights and the model's reliance on large-scale supervised pre-training are available on [Substack](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b).

- **AI Literacy with OpenHermes 2.5 AWQ**: `@__ctrlaltdel__` shared a YouTube [video](https://youtu.be/4-hzQSOhIfc) of their talk focused on open source AI, small models, and applications in structured data extraction given in Vancouver, Canada.

- **Exploring End-side Large Language Models**: Metaldragon01 linked to a **Notion document** discussing MiniCPM, but provided no further context or a valid URL.

- **CroissantLLM Emerges as a Bilingual Competitor**: `@euclaise` shared information about **CroissantLLM**, a 1.3B parameter bilingual language model pretrained on English and French datasets. Further details and use recommendations can be found on [Hugging Face](https://huggingface.co/croissantllm/CroissantLLMBase) and their [related paper](https://arxiv.org/abs/2402.00786).

- **NeurIPS Paper on Synthetic Data Generation Company**: `@euclaise` referenced a paper from the NeurIPS conference about a synthetic data generation company but did not include further details or commentary. The paper can be accessed [here](https://proceedings.neurips.cc/paper_files/paper/2022/file/0f5fcf4bff73a3537e0813a38f0d3f76-Paper-Conference.pdf).

**Links mentioned**:

- [Intro to Open Source AI](https://youtu.be/4-hzQSOhIfc): This is a recording of a talk I gave on Jan 30th, 2024 at BCIT in Vancouver, Canada. The talk is specifically about Natural Language Processing (NLP) with a ...
- [Whisper: How to Create Robust ASR (2 / N)](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=profile&utm_medium=reader2): Part 2 of a multi-part series in which we delve deep into Whisper, OpenAI&#x27;s state-of-the-art automatic speech recognition model
- [croissantllm/CroissantLLMBase · Hugging Face](https://huggingface.co/croissantllm/CroissantLLMBase): no description found
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
- [Memory Efficient Optimizers with 4-bit States](https://arxiv.org/abs/2309.01507): Optimizer states are a major source of memory consumption for training neural networks, limiting the maximum trainable model within given memory budget. Compressing the optimizer states from 32-bit fl...
- [GitHub - thu-ml/low-bit-optimizers: Low-bit optimizers for PyTorch](https://github.com/thu-ml/low-bit-optimizers/): Low-bit optimizers for PyTorch. Contribute to thu-ml/low-bit-optimizers development by creating an account on GitHub.

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1202524431633883166) (374 messages🔥🔥): 

- **Exploring Qwen's AI and Llamafication**: A discussion about `Qwen-72B` AI involved users like `@light4bear` and `@weyaxi` sharing insights on the model. `@weyaxi` shared a [Hugging Face link](https://huggingface.co/Weyaxi/Qwen-72B-Llama) to the llamafied version and noted concerns about the tokenizer performance when used.
  
- **Uncertainty Around Qwen2 Release**: Users `@nonameusr` and `@weyaxi` discussed the anticipated release of `Qwen2`, with `@weyaxi` mentioning that it briefly appeared on the leaderboard and then went offline, causing speculation about its unveiling.

- **Optimizer Discussions & Large Model Training**: Conversations touched on the challenges and strategies for training large language models, such as `@euclaise` mentioning a script for a more efficient optimizer named Adalite, which performs well in certain scenarios.

- **Integrating LLMs into Gaming and Consumer Hardware Constraints**: `@light4bear` shared an idea for an online universe game powered by LLMs, which sparked a debate on the feasibility of running such models on consumer-grade hardware. `@euclaise` and `@stefangliga` pointed out that current models would likely need significant downsizing to run on typical consumer setups.

- **Real-Time Relevance of n-gram Language Models**: `@johnryan465` highlighted an interesting paper that defends the continued relevance of n-gram language models in the age of neural LLMs and proposes an 'infinite-gram' model, which, unlike traditional n-gram models, does not limit the range of n.

**Links mentioned**:

- [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377): Are n-gram language models still relevant in this era of neural large language models (LLMs)? Our answer is yes, and we show their values in both text analysis and improving neural LLMs. Yet this nece...
- [Qwen-VL-Max - a Hugging Face Space by Qwen](https://huggingface.co/spaces/Qwen/Qwen-VL-Max): no description found
- [Weyaxi/Qwen-72B-Llama · Hugging Face](https://huggingface.co/Weyaxi/Qwen-72B-Llama): no description found
- [Weyaxi/Helion-4x34B · Hugging Face](https://huggingface.co/Weyaxi/Helion-4x34B): no description found
- [supertrainer2000/supertrainer2k/optim/adalite.py at master · euclaise/supertrainer2000](https://github.com/euclaise/supertrainer2000/blob/master/supertrainer2k/optim/adalite.py): Contribute to euclaise/supertrainer2000 development by creating an account on GitHub.
- [dataautogpt3/miqu-120b · Hugging Face](https://huggingface.co/dataautogpt3/miqu-120b): no description found
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/): no description found
- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/AravSrinivas/status/1752803571035504858?s=20): Quite a few people asked me if all of Mistral&#39;s models are based off on Meta&#39;s Llama. Especially because the similarity of the outputs was also discovered via testing Mistral Medium on Perplex...
- [Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2): no description found

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1202643231611944980) (38 messages🔥): 

- **Backdoor Training Resistant to Safety Measures**: `@if_a` provided an [Anthropic article](https://www.anthropic.com/news/sleeper-agents-training-deceptive-llms-that-persist-through-safety-training) that suggests *sleeper agents* (deceptive LLMs) can persist even after safety training, indicating that backdoors introduced during training may not be fixable with current methods.
- **Diff Algorithm Suggested for License Comparison**: In a discussion about identifying differences in open source licenses, `@if_a` recommended using a *diff algorithm* followed by an LLM to summarize the changes, with the assumption that modifications are made to standard templates.
- **Context Length Extension with YaRN**: `@rememberlenny` inquired about the feasibility of using YaRN for context length extension of models, and `@bloc97` confirmed it can be used for extensions up to *128k tokens*, with better performance observed in shorter context versions.
- **Curious About Costs to Train on Math**: `@jiha` asked about the cost of training a model similar to *AlphaGeometry* for number theory; `@Error.PDF` responded with an estimate of a *few million dollars* for training and a *few thousand* for execution.
- **Question on Open Sourcing Model Details**: `@420gunna` inquired about resources that inspired the creation of open datasets, specifically reaching out to `@387972437901312000` after thanking them for sharing their datasets and expressing interest in data-centric AI.

**Links mentioned**:

- [teknium/OpenHermes-7B · Hugging Face](https://huggingface.co/teknium/OpenHermes-7B): no description found
- [Cat Reaction GIF - Cat Reaction - Discover &amp; Share GIFs](https://tenor.com/view/cat-reaction-gif-27157580): Click to view the GIF

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1202614144759627816) (6 messages): 

- **Project Obsidian Inquiry**: `@rabiussany` sought clarification on the goals of **Project Obsidian** and needed help understanding current activities within Nous.
- **Project Focus Explained Briefly**: `@_3sphere` informed that the channel description indicates a focus on **multimodality**, but also mentioned that activity in the project seems to be currently low.
- **Newcomer Eager to Participate**: `@rabiussany` showed eagerness to participate in Nous Research projects, inquiring about ways to get involved.
- **Obsidan Project Status Update**: `@teknium` clarified that **Project Obsidian** is essentially complete and released, directing `@rabiussany` to Nous' [huggingface](https://huggingface.co/Nous) for the Obsidian model.
- **Future Collaboration Hopeful**: `@rabiussany` expressed a willingness to wait for future opportunities to contribute to Nous Research projects.
  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1202735168159354920) (3 messages): 

- **Custom Pipeline for `moondream1`**: User `@vikhyatk` introduced a custom pipeline for `vikhyatk/moondream1` on the Hugging Face platform, allowing users to call the pipeline with specific Python code snippets. You can find the pipeline and the discussion [here](https://huggingface.co/vikhyatk/moondream1/discussions/6).

- **Promoting an Evaluation Benchmark for Creativity**: `@Vipitis` shared about proposing a novel evaluation benchmark to assess language model creativity, sharing the update on [Twitter](https://twitter.com/Vipitis/status/1752699776766988309).

- **Resume QA Space Launch**: User `@not-lain` launched a Resume QA space, aimed at improving resumes and sharpening interview responses. Discover more about this helpful tool on the [Hugging Face space](https://huggingface.co/spaces/not-lain/resume-qa).

- **Introduction of Han Instruct Dataset**: User `@pythainlp` shared about the Han Instruct dataset, providing an insightful dataset for various questions and answers. Learn more about the dataset [here](https://huggingface.co/datasets/pythainlp/han-instruct-dataset-v1.0).

- **Release of Ukrainian wav2vec2 bert model**: User `@Yehor` announced the Ukrainian wav2vec2 bert model and provided links to a Discord server and a Telegram group for discussions related to Speech Recognition. The model and additional resources can be found [here](https://huggingface.co/Yehor/w2v-bert-2.0-uk).

**Links mentioned**:

- [Resume Qa - a Hugging Face Space by not-lain](https://huggingface.co/spaces/not-lain/resume-qa): no description found
- [vikhyatk/moondream1 · add pipeline](https://huggingface.co/vikhyatk/moondream1/discussions/6): no description found
- [pythainlp/han-instruct-dataset-v1.0 · Datasets at Hugging Face](https://huggingface.co/datasets/pythainlp/han-instruct-dataset-v1.0): no description found
- [Yehor/w2v-bert-2.0-uk · Hugging Face](https://huggingface.co/Yehor/w2v-bert-2.0-uk): no description found
- [Tweet from thecollabagepatch (@thepatch_kev)](https://x.com/thepatch_kev/status/1752129930404696134): when your singer kinda sounds like elvis for no reason one day  @fffiloni &#39;s dreamtalk needs to come out 😂   this week we just havin fun in the captains chair  next week... @_buildspace
- [@s3nh on Hugging Face: &quot;GPU Poor POV: Quantization Today I want to share with you my notebook plug…&quot;](https://huggingface.co/posts/s3nh/851992122690412): no description found
- [@natolambert on Hugging Face: &quot;Today, we’re releasing our first pretrained Open Language Models (OLMo) at the…&quot;](https://huggingface.co/posts/natolambert/114173328374820): no description found
- [@psinger on Hugging Face: &quot;Happy to share H2O-Danube-1.8b, a small 1.8b model based trained on only 1T…&quot;](https://huggingface.co/posts/psinger/455307248098208): no description found
- [@santiviquez on Hugging Face: &quot;Had a lot of fun making this plot today. If someone ever asks you why you…&quot;](https://huggingface.co/posts/santiviquez/295325502020879): no description found
- [@gsarti on Hugging Face: &quot;🔍 Today&#39;s pick in Interpretability &amp; Analysis of LMs: Gradient-Based Language…&quot;](https://huggingface.co/posts/gsarti/622879789886281): no description found
- [Locutusque/UltraTextbooks · Datasets at Hugging Face](https://huggingface.co/datasets/Locutusque/UltraTextbooks): no description found
- [miqu-70b Chat - a Hugging Face Space by freQuensy23](https://huggingface.co/spaces/freQuensy23/miqu-chat): no description found
- [joshuasundance/mtg-coloridentity-multilabel-classification · Hugging Face](https://huggingface.co/joshuasundance/mtg-coloridentity-multilabel-classification): no description found
- [Tables - a Hugging Face Space by sid27](https://huggingface.co/spaces/sid27/tables): no description found
- [MLX | Mistral-7B-Instruct on Apple Silicon](https://www.youtube.com/watch?v=cjl2ADP8JLQ&t=79s): Can you run Mistral-7B-Instruct-v0.2 from Mistral AI on Apple Silicon with MlX? Let&#39;s find out. -------------------------------------------------------------...
- [Best Image Models Demo - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Best-Image-Models-Demo): no description found
- [ColBERT Inference in the Browser](https://colbert.aiserv.cloud/): no description found

  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1202577314446319686) (241 messages🔥🔥): 

- **Celebrating "moonddream1" Model Downloads**: User `@not_lain` announced that their model `moonddream1` has reached 2500 downloads, celebrating the milestone achievement.
- **Discussion on Video Generation Tools**: `@ch33zw2zard` inquired about the best video generation tools available, expressing interest in alternatives to `moonvalley`; they are open to recommendations.
- **Hugging Face API Usage Clarification**: `@ram1428` sought understanding about whether using the Hugging Face API token for models like `llama2` involves computation on Google Colab or Hugging Face's servers. It was clarified that using the API key allows the use of models without downloading them, offloading compute to Hugging Face's servers.
- **Exploring LLM Hosting for Projects**: `@woodenrobot` asked the community for suggestions on free Large Language Model (LLM) hosting for open-source projects, with discussions about free tiers and possible integrations with other services like Colab or Kaggle.
- **Dataset Dilemma in Academic Projects**: `@akvnn` engaged in an extended discussion with `@doctorpangloss` on choosing a suitable and novel dataset for a classification problem aiming for publication. Despite suggestions and advice, `@akvnn` remained undecided on a specific topic but considered collaborations with dental experts for unique dental scan data.

**Links mentioned**:

- [Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code](https://arxiv.org/abs/2311.07989): In this work we systematically review the recent advancements in code processing with language models, covering 50+ models, 30+ evaluation tasks, 170+ datasets, and 700+ related works. We break down c...
- [RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation](https://arxiv.org/abs/2303.12570): The task of repository-level code completion is to continue writing the unfinished code based on a broader context of the repository. While for automated code completion tools, it is difficult to util...
- [Code Evaluation - a Vipitis Collection](https://huggingface.co/collections/Vipitis/code-evaluation-6530478d8e4767ecfe1bc489): no description found
- [Paper page - DevEval: Evaluating Code Generation in Practical Software Projects](https://huggingface.co/papers/2401.06401): no description found
- [GitHub - huggingface/llm-ls: LSP server leveraging LLMs for code completion (and more?)](https://github.com/huggingface/llm-ls): LSP server leveraging LLMs for code completion (and more?) - GitHub - huggingface/llm-ls: LSP server leveraging LLMs for code completion (and more?)
- [torch_geometric.nn.pool.global_mean_pool &mdash; pytorch_geometric  documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html#torch_geometric.nn.pool.global_mean_pool): no description found
- [GitHub - Silver267/pytorch-to-safetensor-converter: A simple converter which converts pytorch bin files to safetensor, intended to be used for LLM conversion.](https://github.com/Silver267/pytorch-to-safetensor-converter): A simple converter which converts pytorch bin files to safetensor, intended to be used for LLM conversion. - GitHub - Silver267/pytorch-to-safetensor-converter: A simple converter which converts py...
- [no title found](https://api.endpoints.huggingface.cloud/#get-/v2/endpoint/-namespace-/-name-/logs)): no description found
- [GitHub - TabbyML/tabby: Self-hosted AI coding assistant](https://github.com/TabbyML/tabby): Self-hosted AI coding assistant. Contribute to TabbyML/tabby development by creating an account on GitHub.
- [lowres/anime · Datasets at Hugging Face](https://huggingface.co/datasets/lowres/anime): no description found
- [lowres/sukasuka-anime-vocal-dataset · Datasets at Hugging Face](https://huggingface.co/datasets/lowres/sukasuka-anime-vocal-dataset): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1202569480816042054) (12 messages🔥): 

- **Free LLM API Access for Newcomers**: User `@erlonidasap` asked if there were any free APIs for large language models (LLMs), and `@not_lain` informed them about **Hugging Face's free open APIs** for AI models and spaces, sharing a detailed presentation with more information on [page 16](https://docs.google.com/presentation/d/1TMRpL52pkz8ULSJvxaCdsrqROW4w8TkfA5GQ_VCPkhQ/edit?usp=sharing).
- **A Guided Tour to Pretrained LLMs**: On page 16 of his **Google Slides presentation**, `@not_lain` provides an introduction to using pretrained LLMs with clickable yellow links for further exploration.
- **Sharing is Caring**: Community members, including `@tadeodonegana`, appreciated `@not_lain`'s effort in sharing the presentation, and `@not_lain` expressed gratitude for the positive feedback.
- **Proposal for a Transnational Employment Strategy**: `@nikolacurovic`, a non-U.S. senior web developer, proposed a collaboration with U.S. citizens to apply for jobs on LinkedIn while offering to fulfill the job responsibilities remotely, suggesting a percentage-based financial arrangement for any job offers secured.

**Links mentioned**:

[introduction to using pretrained LLMs](https://docs.google.com/presentation/d/1TMRpL52pkz8ULSJvxaCdsrqROW4w8TkfA5GQ_VCPkhQ/edit?usp=sharing): Introduction to using pretrained LLMs Hafedh hichri Released last year, Was SOTA on different tasks, such as image classification, image segmentation

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1202542074617143337) (2 messages): 

- **Google Unleashes Mobile Diffusion Power**: `sta._.no` shared excitement over a new 386M diffusion model from Google, raising curiosity about possible open sourcing. [MobileDiffusion](https://blog.research.google/2024/01/mobilediffusion-rapid-text-to-image.html) boasts rapid sub-second text-to-image generation on mobile devices, outperforming its high-parameter predecessors like Stable Diffusion and DALL·E.

- **I-BERT Accelerates RoBERTa Inference**: `andysingal` introduced the [I-BERT model](https://huggingface.co/docs/transformers/v4.37.1/en/model_doc/ibert), an integer-only quantization of RoBERTa, capable of running inference up to **four times faster**. This efficiency opens doors for transformatively faster natural language processing tasks on edge devices.

**Links mentioned**:

- [MobileDiffusion: Rapid text-to-image generation on-device &#8211; Google Research Blog](https://blog.research.google/2024/01/mobilediffusion-rapid-text-to-image.html): no description found
- [I-BERT](https://huggingface.co/docs/transformers/v4.37.1/en/model_doc/ibert): no description found

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1202566037099384853) (9 messages🔥): 

- **ColBERT in Action**: `@andreer_vespa_99582` shared a tool for live visualization of token contributions to document similarity using the ColBERT model hosted on [colbert.aiserv.cloud](https://colbert.aiserv.cloud). Users like `@cakiki` and `@weyaxi` expressed their enthusiasm and support for the project.
  
- **Free Chat Demo with MiQU**: `@frequesny` posted about a free demo for the MiQU model, which is GPT-4 level and a result of a leak from Mistral AI. The demo is available on [Hugging Face's project page](https://huggingface.co/spaces/freQuensy23/miqu-chat).
  
- **UltraTextbooks Dataset Released**: `@locutusque` introduced the "UltraTextbooks" dataset, combining synthetic and human-written textbooks for advanced NLP tasks, hosted on [Hugging Face's datasets page](https://huggingface.co/datasets/Locutusque/UltraTextbooks). `@stroggoz` expressed appreciation for the new resource.
  
- **VS Code, Jupyter, and Remote GPUs United**: `@chongdashu` wrote a guide on integrating Visual Studio Code with Jupyter Notebooks on a remote GPU server, aiming to streamline the machine learning development workflow. The detailed guide is published on [Medium](https://medium.com/@chongdashu/connecting-visual-studio-code-with-jupyter-notebooks-on-a-remote-gpu-server-instance-8f7cc0696a45).
  
- **Serverless Image Similarity on HuggingFace Spaces**: `@omerxfaruq` developed a serverless image similarity tool using Upstash Vector, demonstrated at [Find Your Twins Space](https://huggingface.co/spaces/omerXfaruq/FindYourTwins) and detailed in a [Hugging Face blog post](https://huggingface.co/blog/omerXfaruq/serverless-image-similarity-with-upstash-vector). The solution focuses on using HuggingFace ecosystem and Upstash to streamline backend and frontend complexities.

**Links mentioned**:

- [ColBERT Inference in the Browser](https://colbert.aiserv.cloud): no description found
- [miqu-70b Chat - a Hugging Face Space by freQuensy23](https://huggingface.co/spaces/freQuensy23/miqu-chat): no description found
- [Connecting Visual Studio Code with Jupyter Notebooks on a remote GPU server instance](https://medium.com/@chongdashu/connecting-visual-studio-code-with-jupyter-notebooks-on-a-remote-gpu-server-instance-8f7cc0696a45): Leverage the power of all three without compromise
- [FindYourTwins - a Hugging Face Space by omerXfaruq](https://huggingface.co/spaces/omerXfaruq/FindYourTwins): no description found
- [Serverless Image Similarity with Upstash Vector and Huggingface Models, Datasets and Spaces](https://huggingface.co/blog/omerXfaruq/serverless-image-similarity-with-upstash-vector): no description found
- [Locutusque/UltraTextbooks · Datasets at Hugging Face](https://huggingface.co/datasets/Locutusque/UltraTextbooks): no description found

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1202681670516351078) (28 messages🔥): 

- **Mamba Presentation on the Horizon**: User `@ericauld` proposed to present Mamba or related topics for the reading-group, with the presentation being set for Friday (Feb 9) and the time being flexible, including late afternoon California time. A [When2meet link](https://www.when2meet.com/?23471427-n4DUl) was provided by `@chad_in_the_house` to schedule the exact time.
- **Presentation Recording Inquiries**: In anticipation of the discussion on AI challenges in law, `@k4rolina_n` requested recording the meeting for those who might join late. `@chad_in_the_house` confirmed the intention to record with OBS.
- **AI in Law Sparking Interest**: `@chad_in_the_house` announced a presentation about the difficulties of AI with Law, which would occur the following day (from the time of messaging) from 1-2pm EST, to be held in the Discord voice-chat.
- **Grappling with Compression Algorithms**: `@chad_in_the_house` commented on a paper about compression algorithms, mentioning a lack of broader benchmarks such as MT and the absence of inference techniques like speculative decoding in the evaluation criteria.
- **Paper Ponderings**: In discussing a paper on compression, `@chad_in_the_house` initially found it interesting but noted that the "best" approach for compression isn't well-defined, before acknowledging the paper's comprehensiveness.

**Links mentioned**:

[Eric's Presentation - When2meet](https://www.when2meet.com/?23471427-n4DUl): no description found

  

---


### HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968) (1 messages): 

- **Advanced Dreambooth Training Arrives**: User `@linoy_tsaban` announced that **Dreambooth LoRA** training for Stable Diffusion (SDXL) is now available in `diffusers` due to a significant community contribution from `@brandostrong`. It includes advanced features such as pivotal tuning, custom captions, the prodigy optimizer, and the newly added noise offset support for improved results, all while requiring less compute power.
  
- **Dreambooth LoRA Training Script Ready for Action**: Hugging Face's GitHub provides the [advanced training script](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sd15_advanced.py) for those ready to leverage these new capabilities in finetuning SD models.
  
- **Release Gets Twitter Spotlight**: The release of this new feature was heralded on Twitter, with `@linoy_tsaban` sharing the announcement, which can be found in the [release tweet](https://twitter.com/linoy_tsaban/status/1753022391079620663).

**Links mentioned**:

[diffusers/examples/advanced_diffusion_training/train_dreambooth_lora_sd15_advanced.py at main · huggingface/diffusers](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sd15_advanced.py): 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch - huggingface/diffusers

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1202589652893179924) (1 messages): 

- **Inquiry about Diffusion Training on Google TPUs**: `@pawkanarek` asked if advanced diffusion training mentioned in the [announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968) would work on a Google TPU. The community has not yet responded with information regarding TPU compatibility.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1202638569043402802) (18 messages🔥): 

- **Function Name Fix Resolves Error**: `@swetha98` resolved an error in the fine-tuning code for a donut model by changing the function name and confirmed that the solution provided by Merve worked fine.
- **Screenshot Clarity for Error Sharing**: `@johko990` suggested to `@swetha98` for future references to share screenshots instead of photos taken by phone for clarity when presenting code-related issues.
- **Seeking Clustering Model Advice for User Actions**: `@amarcel` requested suggestions for clustering models able to identify repetitive sequences of user actions (e.g., clicks, copies, pastes) among 15k screenshots, even considering potential misclicks by users.
- **Model Training with Appended Actions**: `@banaanbakje` shared experience in training models by appending actions to screenshots and suggested making a function that scans for sequences of similar user actions.
- **EfficientNet Based Model Building Resource Shared**: Referencing `@amarcel`'s scenario about analyzing user actions, `@banaanbakje` provided a [link to a guide on building an AI-powered game bot](https://www.akshaymakes.com/blogs/pytorch) with PyTorch and EfficientNet that could help with the concept.


**Links mentioned**:

[Akshay's Personal Website](https://www.akshaymakes.com/blogs/pytorch): I am a Machine Learning Enthusiast. Check out my Projects and Blogs

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1202524818025615400) (5 messages): 

- **Seeking AWS SageMaker Guidance**: User `refik0727` is looking for tutorials or GitHub code on how to use **AWS SageMaker**, **Mistral**, **Llama**, or **Tapas** for building a chatbot using a CSV file or connecting to a local database.
- **ChatGPT as a Learning Resource**: `@akshit1993` recommended that **ChatGPT** is currently the best resource for learning about the topics `refik0727` is interested in.
- **Puzzled by Installation Issues**: User `@.sgp` initially thought that following the install instructions from the documentation would work, but then encountered an **error** during installation.
- **Outdated CUDA Toolkit Causes Error**: `@joshuasundance` suggests that `@.sgp`'s installation error might be due to using an **outdated version of the CUDA toolkit**.
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1202589652893179924) (1 messages): 

- **TPU Compatibility Question Unanswered**: User `@pawkanarek` asked if **advanced diffusion training** will work on Google TPUs, referencing an announcement [here](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968). However, no further information or answers were provided in the available messages.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1014557141132132392/1202588850153852968): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1202875316595593276) (1 messages): 

- **Popup Modal Component Released for Gradio**: `@yuviii_` has announced the release of a new **custom component for Gradio**, the 𝚐𝚛𝚊𝚍𝚒𝚘_𝚖𝚘𝚍𝚊𝚕, created by Ali Abid. This component can be used for displaying a license agreement, prompting user logins, alerts, step-by-step guides, contextual help, or confirmations within Gradio Apps.

- **Explore the 𝚐𝚛𝚊𝚍𝚒𝚘_𝚖𝚘𝚍𝚊𝚕 Component**: You can check out and implement the `𝚐𝚛𝚊𝚍𝚒𝚘_𝚖𝚘𝚍𝚊𝚕` in your own Gradio apps by visiting the [Hugging Face Space provided](https://huggingface.co/spaces/aliabid94/gradio_modal). The component could enhance user interaction within Gradio applications by offering various popup functionalities.

**Links mentioned**:

[gradio_modal V0.0.1 - a Hugging Face Space by aliabid94](https://huggingface.co/spaces/aliabid94/gradio_modal): no description found

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1202553466124369951) (298 messages🔥🔥): 

- **Whispers of "LLaMa 1.6" Success**:
  User `@kache_` briefly mentioned their satisfaction by stating "**llava 1.6** is very good," but provides no further details on its performance or comparisons to other models.

- **Hacker News Spotlight**: `@itali4no` highlighted that a writeup by `@122380520226160640` gained traction on Hacker News, linking to the discussion with the message: "Your reddit writeup is doing numbers on orange site <@122380520226160640>, if you haven't seen [Hacker News Discussion](https://news.ycombinator.com/item?id=39215242)."

- **Watermark Debate in Bard's Image Generation**: `@max_voltage` shared concerns about Bard's upgraded image-generation feature being too wrapped up in responsible AI principles, embedding watermarks to distinguish AI-created visuals, and linked to the relevant discussion with a disgusted emoji.

- **Imagen 2's Generated Image Quality Scrutinized**: `@thejonasbrothers` noted the apparent noise in all the Imagen2 images and wondered why output wasn't returned in more efficient formats like "jpeg with 80% quality," expressing a critical comparison to SDXL's output.

- **Discourse on Autoencoder Sensitivity and Training**: Acclaimed users like `@drhead` and `@thejonasbrothers` had an extended technical discussion about the sensitivity of latent space in autoencoders, the impact of noise patterns, and the potential evolution of these patterns during model training. The conversation referenced various models, including Segmind Vega, and the differences observed in them when compared to SDXL VAE.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/823813159592001537/823813160075132991/1202377957449154590): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Finally! First Look at Google&#39;s New Imagen 2 &amp; Image FX Interface!](https://youtu.be/2CB1Wb0b6dA?si=cKLpPhzXpi-ApjC4): ImageFX is an experimental technology that allows you to generate your own synthetic images. ImageFX is powered by Google’s Imagen 2 and uses Google DeepMind...
- [no title found](https://news.ycombinator.com/item?id=39215242): no description found
- [ImageFX](https://aitestkitchen.withgoogle.com/tools/image-fx): no description found
- [GitHub - openai/consistencydecoder: Consistency Distilled Diff VAE](https://github.com/openai/consistencydecoder): Consistency Distilled Diff VAE. Contribute to openai/consistencydecoder development by creating an account on GitHub.

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (1 messages): 

felfri_: https://allenai.org/olmo
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1202529743363121152) (137 messages🔥🔥): 

- **In Search of Enhanced Memory**: `@flyyt4` asked about libraries or tools to improve the long-term memory of LLMs like ChatGPT, which prompted `@fabguy` to simply mention "memGPT."
- **The Art of Advanced Prompting**: `@wolfspyre` shared a [link to advanced prompt examples](https://funkpd.com/devlog/prompt-examples/) for text generation, emphasizing the significance of Chain of Thought, Tree Thinking, and prompt compression.
- **Discovering LLaMA on Hugging Face**: `@pierrunoyt` posted a link to [Hugging Face's moondream1 project page](https://huggingface.co/spaces/vikhyatk/moondream1), and `@fabguy` acknowledged this as a good find.
- **Optimizing GPU Offload**: `@yagilb` gave advice on turning on GPU offload to address complaints of slow model load times by `@pierrunoyt`.
- **Downloading Correct LLaMA Model Versions**: After `@.veski` installed a heavy LLaMA model, `@heyitsyorkie` directed them to download the GGUF quantized version from Hugging Face for better compatibility with LM Studio.

**Links mentioned**:

- [moondream1 - a Hugging Face Space by vikhyatk](https://huggingface.co/spaces/vikhyatk/moondream1): no description found
- [Devlog: Make Amazing GPT copy! Prompt Examples of Chain of Thought, Tree Thinking, and more - FunkPd](https://funkpd.com/devlog/prompt-examples/): Welcome to the world of advanced prompt examples for text generation - a realm where creativity and logic intertwine, and where complexity is not a barrier,
- [llama.cpp/README-sycl.md at ce32060198b7e2d6a13a9b8e1e1369e3c295ae2a · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/ce32060198b7e2d6a13a9b8e1e1369e3c295ae2a/README-sycl.md?plain=1#L64): Port of Facebook&#39;s LLaMA model in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
- [GitHub - facebookresearch/llama: Inference code for LLaMA models](https://github.com/facebookresearch/llama): Inference code for LLaMA models. Contribute to facebookresearch/llama development by creating an account on GitHub.

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1202595073406144542) (32 messages🔥): 

- **Users express dissatisfaction with ChatGPT**: `@goldensun3ds` expressed frustration over ChatGPT becoming "useless," particularly with its reduced web browsing capabilities and seemingly arbitrary limitations such as censorship and context size. They likened ChatGPT’s dominance and limitations to Intel's history with CPUs.
- **Quality concerns over various model versions**: `@kujila` remarked on the superiority of **Q4** over **Q2** by calling Q4 "a lot less of a FILTHY LIAR," inferring improvements in the model's reliability.
- **Searching for uncensored models**: In a response to `@zono50.` looking for the best uncensored 7B model, `@goldensun3ds` recommended avoiding small models for story writing and provided a personal experience with several models including **The Bloke Goat Storytelling 70B Q3KS and The Bloke Dolphin 2 6 Mixtral 7B Q4**.
- **Users troubleshooting and seeking models with no guardrails**: `@p4stoboy` asked for recommendations on the best model without guardrails after a disappointing experience with **Mistral 7B Instruct**. Various users discussed solutions, including `@ptable` who posted a link in the channel, and `@goldensun3ds` advised on editing AI messages in LM Studio to add missing pronouns.
- **Performance and context limitations in larger models discussed**: `@binaryalgorithm` touched on the necessity of keeping story lore consistent with large contexts and mentioned that models like **GPT-4** are limited in their output, affecting story continuity. They speculated on the costs and feasibility of using models with larger context sizes via API.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1185654311276003448): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1202529860225081385) (12 messages🔥): 

- **Insights on GGUF format**: `@fabguy` acknowledged that **GGUF** is a better format than `.chk` as it explicitly loads tensors, however, they expressed caution due to limited knowledge of **llama.cpp implementation**. Despite this caveat, `@fabguy` mentions having no issues with **LM Studio** over the past 6 months.

- **Finding and Removing Model Adapters**: `@elsatch` inquired about removing **model adapters** from vision models after downloading, specifically after installing **fp16** for **llava** and then wanting to switch to **Q4**. `@heyitsyorkie` pointed to finding them in the *local models folder*.

- **Request for Timely Support Before Major Meeting**: `@docorange88` urgently requested a response to their email prior to a significant meeting scheduled for 12PM CST concerning software discussion. `@yagilb` prioritized the response, apologized for the delay, and ensured timely communication.

- **Guidance on Locating Model Adapters on Disk**: `@fabguy` helped `@elsatch` by suggesting to *Right-Click* the model in the "my models" view to be led directly to the model folder on the disk.

- **LM Studio Version Differences Affecting Model Loading**: `@foobar8553` reported an issue where they could not load the **mixtral-8x7B ... Q5_K_M model** with **LM Studio 0.2.12** despite being able to do so with **0.2.10** due to potential changes in the way models are loaded into RAM and GPU. `@yagilb` offered to provide **0.2.10** for comparison testing and asked about the operating system used.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1202562558586912798) (44 messages🔥): 

- **Dual 3090s for Larger Models**: `@silverstar5654` inquired about using dual 3090 GPUs with LM Studio to load larger models. `@pefortin` suggested simply checking the offload to GPU box, assuming that the drivers are installed correctly.

- **LM Studio ARC Support Speculation**: `@goldensun3ds` shared a link questioning whether ARC support could be coming to LM Studio, to which `@rugg0064` provided a brief affirmative response.

- **Mixtral in VRAM Queries**: `@docorange88` asked about running Mixtral on two 4090 or RTX6000 GPUs and whether the memory requirements could offload to CPU RAM. `@foobar8553` shared that they ran Mixtral with 12 GB VRAM on a 4070, utilizing additional system RAM.

- **Model Performance on Multi-GPUs**: `@ellric_` questioned the performance and quality of 8bit versus 4bit models and if adding more GPUs would improve speed. `@mahdiyari` noted that using two GPUs might be slower unless using NVLink, and shared a GPU benchmark link for LLM inference.
  
- **Loading Issues with Models on LMStudio**: `@lowkey9920` experienced difficulties running a 3.8 GB model on a system with a 3080 GPU and 64GB RAM, specifically with "magicoder-s-ds." Others like `@binaryalgorithm` also reported issues, while noting that other models loaded fine.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1148200624924667914/1202195308914671649): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1202885906290446376) (1 messages): 

- **Llava 1.6 Support Inquiry**: User `@_koopman` inquired about support for **llava 1.6**, noting that LM Studio does not currently recognize it as a vision model. There is anticipation for integration.
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1202585069567352843) (3 messages): 

- **Exploring Autogen with Multiple Servers**: `@ellric_` was curious about the capability of running **Autogen** using two different servers instead of a single machine. They considered experimenting with this setup.
- **Running Dual Instances of LM Studio**: `@_anarche_` confirmed that it's feasible to run two models simultaneously using **LM Studio** on separate ports if there's sufficient hardware to support it. They provided guidance to `@ellric_` for their planned experimentation.
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1202797003508158527) (2 messages): 

- **LM Studio Puzzles with Open Interpreter**: User `@raiderduck` is experiencing issues while integrating **OI (Open Interpreter)** with **LM Studio**, where responses become nonsensical upon activating the server. They have successfully loaded *Solar instruct* into LM Studio, which works well in chat but fails when using OI, leading to self-replies and nonsense.

- **OI Costs a Pretty Penny with GPT-4**: `@raiderduck` mentioned that **OI** operates smoothly when paired with **GPT-4**, but the extensive usage has resulted in a steep **$250 bill in one week**. They are currently seeking server preset settings advice for OI to alleviate these costs.
  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1202639362471366746) (33 messages🔥): 

- **OLMo Enters the LLM Arena**: `@sk5544` sparked a discussion with the release of the [OLMo paper](https://allenai.org/olmo/olmo-paper.pdf), catching the community's attention on their approach and licensing, with `@hailey_schoelkopf` clarifying that AI2 hasn't released a chat model or interface.
- **Debate Over OLMo's Evaluation Scores**: `@mistobaan` expressed surprise at OLMo's modest score on the GSM8K benchmark, prompting a debate about tokenizer issues and exacting format problems outlined by `@stellaathena`, mentioning SFT models potentially scoring better.
- **OLMo's Training Code Scrutinized**: The community dissected OLMo's warmup scheduler, with `@jerry0478` explaining its function for resetting optimizer state, and `@mistobaan` pointing out an absence of papers discussing the warmup technique, referencing an alternative [PyTorch warmup tool](https://github.com/Tony-Y/pytorch_warmup).
- **Examination of OLMo's Normalization Choices**: Discussion about OLMo's use of layer norm versus rmsnorm gained momentum, clarified by `@ad8e` that they used layer norm without learnable scales, and `@the_random_lurker` giving a skeptical take on ai2's emphasis on data quality and model evaluations.
- **Community Weighs in on AI2's Evaluation Metrics**: `@the_random_lurker` criticized AI2's choice of evaluation metrics, suggesting they might have cherry-picked results to favorably compare against `llama2`, raising eyebrows about the benchmarks not featured in the main paper but on a HF page.

**Links mentioned**:

- [OLMo/olmo/optim.py at main · Mistobaan/OLMo](https://github.com/Mistobaan/OLMo/blob/main/olmo/optim.py#L544C7-L544C28): Modeling, training, eval, and inference code for OLMo - Mistobaan/OLMo
- [phi-playground/notebooks/evaluation/evaluation_GSM8k_transformers_colab.ipynb at main · bmx-ai/phi-playground](https://github.com/bmx-ai/phi-playground/blob/main/notebooks/evaluation/evaluation_GSM8k_transformers_colab.ipynb): A series of utilities to play with microsoft/phi-* models  - bmx-ai/phi-playground
- [GitHub - Tony-Y/pytorch_warmup: Learning Rate Warmup in PyTorch](https://github.com/Tony-Y/pytorch_warmup): Learning Rate Warmup in PyTorch. Contribute to Tony-Y/pytorch_warmup development by creating an account on GitHub.

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1202528783098323045) (159 messages🔥🔥): 

- **N-gram Scaling Sparks Debate**: `@xylthixlm` shared a paper on scaling n-grams to trillions of tokens, prompting discussion on its innate value and synergy with larger language models (LLMs). Several users, including `@ai_waifu` and `@johnryan465`, contemplated combining n-gram models with LLMs to boost overall performance and generalization, while `@catboy_slim_` commented on the potential in-distribution bias of such models.

- **Infinigram Under the Microscope**: The research community showed both interest and skepticism towards the Infinigram model; `@_inox` pointed out it was only benchmarked in combination with LLMs, and not on its own, sparking a discussion on its standalone capabilities.

- **Tagging Backtranslations for LLMs**: `@resonancia` sparked a discussion about applying techniques from the machine translation field, such as tagging backtranslated data, to improve the efficiency of language models. `@teknium` mentioned that while no Hermes-series models uses such techniques, similar approaches are seen in other works, hinting at possible applications in future models.

- **Synthetic Data Generation Strategies Evaluated**: Users, with insights from `@blagdad`, discussed different methods for synthetic data generation, specifically mentioning MCTS (Monte Carlo tree search) and the potential drawbacks of beam search in diversity. It was widely agreed that diverse synthetic data is key, and this might entail using alternative sampling methods.

- **Machine Learning Community Highlights**: `@ge0.io` announced the release of the Gaussian Adaptive Attention library tailored for multi-modal work, while `@digthatdata` shared links to new papers and upcoming research, such as Amortized Text-to-Mesh (AToM), showcasing the dynamic nature of the AI research landscape.

**Links mentioned**:

- [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](http://arxiv.org/abs/2401.17377): Are n-gram language models still relevant in this era of neural large language models (LLMs)? Our answer is yes, and we show their values in both text analysis and improving neural LLMs. Yet this nece...
- [AToM: Amortized Text-to-Mesh using 2D Diffusion](https://snap-research.github.io/AToM/): AToM: Amortized Text-to-Mesh using 2D Diffusion, 2023.
- [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417): Recently the state space models (SSMs) with efficient hardware-aware designs, i.e., Mamba, have shown great potential for long sequence modeling. Building efficient and generic vision backbones purely...
- [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259): We present a scalable method to build a high quality instruction following language model by automatically labelling human-written text with corresponding instructions. Our approach, named instruction...
- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089): Changing how pre-trained models behave -- e.g., improving their performance on a downstream task or mitigating biases learned during pre-training -- is a common practice when developing machine learni...
- [AToM: Amortized Text-to-Mesh using 2D Diffusion](https://arxiv.org/abs/2402.00867): We introduce Amortized Text-to-Mesh (AToM), a feed-forward text-to-mesh framework optimized across multiple text prompts simultaneously. In contrast to existing text-to-3D methods that often entail ti...
- [Tagged Back-Translation](https://aclanthology.org/W19-5206/): Isaac Caswell, Ciprian Chelba, David Grangier. Proceedings of the Fourth Conference on Machine Translation (Volume 1: Research Papers). 2019.
- [Tagged Back-translation Revisited: Why Does It Really Work?](https://aclanthology.org/2020.acl-main.532/): Benjamin Marie, Raphael Rubino, Atsushi Fujita. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. 2020.
- [GitHub - gioannides/Gaussian-Adaptive-Attention](https://github.com/gioannides/Gaussian-Adaptive-Attention): Contribute to gioannides/Gaussian-Adaptive-Attention development by creating an account on GitHub.

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1202650548852228189) (11 messages🔥): 

- **Spontaneous Research Intersection**: `@norabelrose` expressed surprise upon discovering a [research paper](https://arxiv.org/abs/2401.12973) on in-context language learning (ICLL) that was aligned with their own research interests.
- **Paper on 'Contextual Neurons' Shared**: `@norabelrose` shared a link to a paper exploring the concept of [contextual neurons](https://arxiv.org/abs/2311.00863) in language models, particularly a neuron that activates on German text, which is part of a second-order circuit.
- **Perplexity vs. Open-Ended Text Generation**: `@nostalgebraist` noted that ∞-gram can improve neural LMs perplexity, but may be harmful to open-ended text generation by causing odd mistakes and retrieval of irrelevant tokens.
- **N-gram strength in Retrieval over Coherence**: `@nostalgebraist` suggested that n-gram models are better suited for tasks resembling long-tail retrieval rather than those requiring coherent text generation, with perplexity results reflecting this dichotomy.
- **Comparing MoE Attack and ∞-gram Retrieval**: `@xa9ax` drew parallels between difficulties experienced with ∞-gram in open-ended text generation and MoE model attacks where incorrect expert selection can compromise output, referencing a [paper on MoE attacks](https://arxiv.org/pdf/2210.10253.pdf) and expressing interest in this area of research.

**Links mentioned**:

- [In-Context Language Learning: Architectures and Algorithms](https://arxiv.org/abs/2401.12973): Large-scale neural language models exhibit a remarkable capacity for in-context learning (ICL): they can infer novel functions from datasets provided as input. Most of our current understanding of whe...
- [Training Dynamics of Contextual N-Grams in Language Models](https://arxiv.org/abs/2311.00863): Prior work has shown the existence of contextual neurons in language models, including a neuron that activates on German text. We show that this neuron exists within a broader contextual n-gram circui...

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (1 messages): 

daniellepintz: Nope, no `limit`
  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1202599419485229096) (151 messages🔥🔥): 

- **Reddit Exposes a Flaw in VAE Latent Space**: `@gsegato` highlighted a post discussing the critical flaws in the KL divergence loss on the KL-F8 VAE. It reportedly affects models like SD1.x, SD2.x, SVD, DALL-E 3, making them less efficient by smuggling global information through minimal pixels.
- **VAE Flaw Sparks Intrigue**: `@swyxio` responded to the discussion on VAE shortcomings, noting the brevity of the Reddit writeup and its straightforward nature compared to traditional academic papers would have included unnecessary fluff.
- **LLMMemes Come Marching In**: `@natolambert` spread the word about LLM memes through a [Twitter post](https://twitter.com/natolambert/status/1753063313351835941), generating interest in the lighthearted side of language models.
- **AI2 Unveils OLMo Models**: Discussion about [AI2's OLMo model release](https://huggingface.co/allenai/OLMo-7B), featuring details like the use of different hardware during training, with a nod towards its relatively short context length of 2048 compared to some other models.
- **Nomic AI Releases New Embeddings**: `@coffeebean6887` highlighted Nomic AI's release of open source embeddings and their excellent performance on the LoCo benchmark; a detailed blog post and dataset are available for further exploration.

**Links mentioned**:

- [Tweet from anton (@abacaj)](https://x.com/abacaj/status/1752788052500512869?s=46&t=90xQ8sGy63D2OtiaoGJuww): Here&#39;s a snippet of using gpt-4 as a &#34;reward&#34; model. Has been working *really* well for me (better than using a numbering system)
- [Amazon announces Rufus, a new generative AI-powered conversational shopping experience](https://www.aboutamazon.com/news/retail/amazon-rufus): With Rufus, customers are now able to shop alongside a generative AI-powered expert that knows Amazon’s selection inside and out, and can bring it all together with information from across the web to ...
- [Meet Act II of Arc Browser | A browser that browses for you](https://youtu.be/WIeJF3kL5ng?si=itoKPOlDUAIsuV2c): On Thursday, February 1st @ 12.30pm ET we shared our vision for Act II of this journey we call Arc — a new category of software, a browser that browses for y...
- [Open Language Models (OLMos) and the LLM landscape](https://www.interconnects.ai/p/olmo): A small model at the beginning of big changes.
- [Why we founded Parcha ](https://www.hitchhikersguidetoai.com/p/why-we-founded-parcha): A deeper dive into why we&#x27;re building AI agents to supercharge compliance and operations teams in fintech at Parcha
- [no title found](https://news.ycombinator.com/item?id=36855516): no description found
- [OLMo Suite - a allenai Collection](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778): no description found
- [GitHub - uptrain-ai/uptrain: Your open-source LLM evaluation toolkit. Get scores for factual accuracy, context retrieval quality, tonality, and many more to understand the quality of your LLM applications](https://github.com/uptrain-ai/uptrain): Your open-source LLM evaluation toolkit. Get scores for factual accuracy, context retrieval quality, tonality, and many more to understand the quality of your LLM applications - GitHub - uptrain-ai...
- [Tweet from anton (@abacaj)](https://x.com/abacaj/status/1752814377068023988?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q): LazyGPT is back  ↘️ Quoting Shannon Sands (@max_paperclips)   So apparently, according to this at least, it&#39;s actually worse  https://aider.chat/docs/benchmarks-0125.html  Amazing. I can&#39;t see...
- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/s/p2BkzI9ypO): no description found
- [Tweet from swyx (@swyx)](https://x.com/swyx/status/1753279194065428839?s=46&t=90xQ8sGy63D2OtiaoGJuww): @simon_mo_ @RickLamers @zhuohan123 @woosuk_k @amanrsanger i think theyre doing the thing we talked about at neurips
- [allenai/OLMo-7B · Hugging Face](https://huggingface.co/allenai/OLMo-7B): no description found
- [NAIRR Pilot - Home](https://nairrpilot.org/): no description found
- [[Public] vLLM Project Update @ Second vLLM Meetup](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/mobilepresent?slide=id.g2650ce3df47_0_470)): Project Update Jan 31st, 2024 The Second vLLM Meetup @ IBM 1
- [R-Judge: Benchmarking Safety Risk Awareness for LLM Agents](https://arxiv.org/html/2401.10019v1): no description found
- [How to Announce Your Actual AI](https://matt.sh/ai-how-to-announce): no description found
- [no title found](https://dev.to/builderio/dont-build-ai-products-the-way-everyone-else-is-doing-it-9a7): no description found
- [7 Habits of Highly Effective AI Business Projects](https://towardsdatascience.com/7-habits-of-highly-effective-ai-business-projects-6ced590e6db8?gi=e4b47a172d38): What’s the difference between good &amp; great AI business projects? Here are 7 things to consider when doing AI work in your organisation.
- [How to convince Venture Capitalists you’re an expert in Artificial Intelligence](https://medium.com/machine-learning-in-practice/how-to-convince-venture-capitalists-youre-an-expert-in-artificial-intelligence-39d5edaca290): If you like this article, check out another by Robbie:  15 Ways a Venture Capitalist Says “No”
- [Launching your new AI Startup in 2023 &mdash; Building Better Teams](https://buildingbetterteams.de/profiles/brian-graham/navigating-ai-businesses): In the last few months more and more people have been asking me for my thoughts on their AI business ideas, and for help with navigating the space. This post covers the majority of my thoughts on the ...
- [How to use AI to do practical stuff: A new guide](https://www.oneusefulthing.org/p/how-to-use-ai-to-do-practical-stuff): People often ask me how to use AI. Here&#x27;s an overview with lots of links.
- [3 things everyone’s getting wrong about AI](https://www.washingtonpost.com/technology/2023/03/22/ai-red-flags-misinformation/): As AI tools spread, people are struggling to separate fact from fiction.
- [How to talk about AI (even if you don’t know much about AI)](https://www.technologyreview.com/2023/05/30/1073680/how-to-talk-about-ai-even-if-you-dont-know-much-about-ai/): Plus: Catching bad content in the age of AI.
- [What are AI Agents?](https://serokell.io/blog/what-are-ai-agents): In this post, you’ll learn what AI agents are and what they are truly capable of. You’ll also learn how to build an AI agent suitable for your goals.
- [AI Agent Basics: Let’s Think Step By Step](https://www.jonstokes.com/p/ai-agent-basics-lets-think-step-by): An introduction to the concepts behind AgentGPT, BabyAGI, LangChain, and the LLM-powered agent revolution.
- [Pitching Artificial Intelligence to Business People](https://towardsdatascience.com/pitching-artificial-intelligence-to-business-people-f8ddd8fb2da2): From silver bullet syndrome to silver linings
- [How PR people should (not) pitch AI projects](https://thenextweb.com/news/how-pr-people-should-not-pitch-ai-projects-syndication): These are exciting times for the artificial intelligence community. Interest in the field is growing at an accelerating pace, registration at academic and professional machine learning courses is soar...
- [Educating Clients about Machine Learning and AI &#8211; Andy McMahon](https://electricweegie.com/articles/educating-clients/): no description found
- [People + AI Guidebook](https://pair.withgoogle.com/guidebook/patterns/how-do-i-onboard-users-to-new-ai-features): A toolkit for teams building human-centered AI products.

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1202682021730586655) (3 messages): 

- **New Latent Space Podcast Episode Alert**: `@swyxio` announced the release of the latest [Latent Space Podcast](https://twitter.com/latentspacepod/status/1753120715254198425) on Twitter and Hacker News (HN).
- **Clickbait Experiment Falters**: `@swyxio` mentioned that despite using a clickbait title and image for the podcast promotion, the response has been disappointing.
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1202568921253937222) (50 messages🔥): 

- **Confusion over AI Ethics and Development**: `@yami1010` revealed confusion about AI's role across various domains and questioned what truly constitutes unethical in AI learning. They expressed a desire to access sources explaining LLMs' inner workings, highlighting the blurred lines in AI development responsibilities.
  
- **Chatbot Integration for Customer Service Inquiry**: `@qiqimon` inquired about the difficulty of integrating a GPT-powered chatbot for handling customer service inquiries in a school setting, suggesting usage for tasks such as enrollment information.

- **Misattribution of AI Development Credit**: `@movoza` voiced frustration over Bing claiming it developed an AI, corrected only after pointed questions to acknowledge OpenAI’s role. `@yami1010` contributed a broader perspective on AI development's collective nature, bringing in history and various stakeholders involved.

- **Understanding Perceived Performance Dip in ChatGPT**: `@voidrunner42` observed that ChatGPT seems to be getting dumber, with `.@tadase.` suggesting that the issue might be related to how the system is prompted rather than a general decline in quality.

- **Moderation Standards and Image Guidelines Debate**: A conversation between `@jeremy.o` and `@Ted` dealt with content moderation on a server, particularly around image allowable content and the interpretation of G-rating guidelines, highlighting a tension between content diversity and strict moderation.
  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1202600416840392745) (92 messages🔥🔥): 

- **Chatbot Integration Queries for Schools**: `@qiqimon` inquired about the complexity of implementing a chatbot using GPT for a school's customer service to manage enrollment inquiries, with `@lugui` and others suggesting it's quite straightforward but requires measures to prevent trolling.
- **GPT and Knowledgebase Troubles**: `@united_beagle_74104` expressed frustration with ChatGPT errors and sought customer support contact. Meanwhile, users discussed the best file formats for maintaining a knowledgebase, with strong opinions on avoiding XLXS and DOCX due to encoding issues, as advised by `@darthgustav.` and others.
- **RSV Over JSON for AI Product Recommendations**: `@quengelbert` and `@darthgustav.` had an extensive exchange on the best knowledgebase format for an AI product-recommendation assistant, with RSV (row separated values) suggested over JSON or XLXS for improved GPT performance.
- **API Authentication Struggles for Custom GPT Actions**: `@woodenrobot` faced difficulties regarding bearer token authentication when moving a GPT project from internal alpha to public beta, particularly in working with a Weaviate database.
- **Erratic GPT Behavior with Knowledge Files and RAG**: Users `@loschess`, `@blckreaper`, and others reported inconsistency in GPT's accessing knowledge files and using APIs following recent updates, and shared workarounds such as reducing file sizes to avoid retrieval tool dependencies.
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1202562854621020261) (79 messages🔥🔥): 

- **Concern Over Changes Affecting Models**: `@dreamgen` expressed concerns that the code change involving the line `if any(m in name for m in ["norm", "gate"]):` would affect many models like Llama and Mistral. `@nafnlaus00` found this problematic change in `src/axolotl/utils/models.py` and a discussion ensued about the potential effects on model training and memory use.
- **Proposed Fix for Model Issue:** A fix was proposed by `@nafnlaus00`, changing code to include `model_config.model_type == "mixtral"` when checking for "gate" in names, to which `@nanobitz` queried whether "mixtral" was meant instead of "mistral".
- **Contributions on GitHub and Model Troubleshooting**: `@nafnlaus00` had trouble with Git commands, accidentally deleting important files, and expressed dissatisfaction with GitHub, while `@nanobitz` offered to help create a pull request and asked `@nafnlaus00` for a GitHub handle reference.
- **Discussion on Text Embedding and Vector Databases**: Users engaged in a technical discussion about text embedding models and vector databases, with specific mention of `nomic-embed-text-v1` by `@nanobitz`, and recommendations to start with `bge` by `@dangfutures`. The conversation also touched on finding and using GPUs in cloud services.
- **Updates on 7B Models Performance and Utilization**: There was a conversation about the stagnation of 7B model improvements and mentions of new models like `CapybaraHermes-2.5-Mistral-7B` and `Eagle 7B`. `@dangfutures` and `@c.gato` shared links to recent 7B models and their results, highlighting the progress and competition in this space.

**Links mentioned**:

- [RWKV/v5-Eagle-7B · Hugging Face](https://huggingface.co/RWKV/v5-Eagle-7B): no description found
- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard): no description found
- [argilla/CapybaraHermes-2.5-Mistral-7B · Hugging Face](https://huggingface.co/argilla/CapybaraHermes-2.5-Mistral-7B): no description found
- [GitHub - qdrant/fastembed: Fast, Accurate, Lightweight Python library to make State of the Art Embedding](https://github.com/qdrant/fastembed): Fast, Accurate, Lightweight Python library to make State of the Art Embedding - GitHub - qdrant/fastembed: Fast, Accurate, Lightweight Python library to make State of the Art Embedding
- [nomic-ai/nomic-embed-text-v1 · Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1): no description found
- [🌠 Improving RAG by Optimizing Retrieval and Reranking Models](https://docs.argilla.io/en/latest/tutorials_and_integrations/tutorials/feedback/fine-tuning-sentencesimilarity-rag.html#Bi-Encoder-Model): In this tutorial, we will show how to improve a RAG (Retrieval Augmented Generation) model by optimizing the retrieval and reranking models. For this purpose, we will use the ArgillaTrainer to fine...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (1 messages): 

caseus_: A fix for this was merged upstream in transformers
  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1202547054325538836) (10 messages🔥): 

- **Seeking Finetuning Advice for Mixtral**: `@emperor` inquired about the best way to fully finetune Mixtral, considering whether all parameters except the routers and gates should be fine-tuned.
  
- **GPU VRAM Matters for Mixtral Finetuning**: `@caseus_` suggested that depending on the available GPU VRAM, approaches like using **ds zero3**, **8bit optimizer**, and freezing the top third to half layers could be beneficial starting points for finetuning Mixtral. 

- **Full Model Finetuning with Zero3**: `@caseus_` also mentioned that with offloading using **Zero3**, one might be able to finetune the entire Mixtral model.

- **Looking for Hyperparameters Insights**: `@emperor` sought confirmation on whether Mixtral has any good working hyperparameters (hparams) and whether it behaves like **Mistral7b**, which does not favor high learning rates, unlike the Llama model.

- **Tokenization Troubles and Potential Solution**: `@arcontex` asked for assistance with an unspecified problem, and to this, `@nanobitz` suggested trying to switch the tokenizer to `AutoTokenizer`.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1202590835104026664) (17 messages🔥): 

- **RunPod Rundown**: `@c.gato` experienced a sudden shutdown of their pod and loss of data, mentioning that they received an email stating the pod would not be removed for two days, but found it gone in just 45 minutes.
- **Link to RunPod Discord Provided**: In response to `@c.gato`'s issue, `@caseus_` suggested addressing the concern on the official RunPod Discord, providing the link: [RunPod Discord Help Channel](https://discord.com/channels/912829806415085598/1187492973148115076).
- **Possible Bug in Pod Deletion Timing**: `@caseus_` indicated that `@c.gato`'s experience might point to a bug, differentiating between a pod being stopped and deleted.
- **Hugging Face Connection Troubles**: `@dreamgen` reported an SSL error when attempting to transfer data from RunPod to Hugging Face, questioning the reliability of the service.
- **Slow Download Speeds Questioned**: `@dreamgen` complained about the lengthy download time for a Torch package on RunPod, expressing frustration over the service's efficiency.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/912829806415085598/1187492973148115076): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (1 messages): 

dangfutures: did you guys figure out the configs for mistral
  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1202548368539123782) (66 messages🔥🔥): 

- **Guide to Share Funny AI Responses**: `@tbyfly` sought guidance on where to share an amusing response from Perplexity AI. `@me.lk` directed them to share it in another channel, and `@tbyfly` confirmed the share.
  
- **Subscription Support Query**: `@dame.outlaw` asked for assistance with a subscription issue on the mobile app, to which `@icelavaman` responded by asking for details regarding their account and login method. 

- **Perplexity AI Models Explained**: `@icelavaman` provided a link to a blog post explaining the new PPLX models that focus on delivering up-to-date and factual responses. `@clay_ferguson` sought information on model usage, for which `@icelavaman` referred again to the detailed blog post.

- **Setting Perplexity as Default Search Engine**: `@bartleby0` shared a solution for setting Perplexity AI as a browser’s default search engine using a provided template link.

- **In-App Issues and Competitor Mention**: Users discussed in-app problems such as text selection bugs and default search engine settings. Separately, `@dpshade22` mentioned Arc Search being a potential competitor or user of Perplexity AI.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1202251275698450443/1202251275698450443): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047204950763122820/1180293637087698954): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649527299055688/1197892547276705843): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Supported Models](https://docs.perplexity.ai/docs/model-cards): no description found
- [Introducing PPLX Online LLMs ](https://blog.perplexity.ai/blog/introducing-pplx-online-llms): The first-of-its-kind Online LLM API
- [What models does Copilot use?](https://blog.perplexity.ai/technical-faq/what-models-does-copilot-use): Dive deep into Perplexity's technical details with our comprehensive FAQ page. From the nuances of AI models like GPT-4 and Claude 2 to token limits and AI profiles, get concise answers to optimize yo...
- [Perplexity Blog](https://blog.perplexity.ai/technical-faq): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1202548983457644584) (4 messages): 

- **Cryptic Contemplation**: User `@tbyfly` shared a pondering emoji, but without additional context or discussion, the intent behind the message remains a mystery.
- **Approval of Content**: `@twelsh37` seemed pleased with an unspecified content, describing it as a "**jolly decent watch**," though the specifics of the content were not provided.
- **Embracing Perplexity through Blogging**: New community member `@.sayanara` shared their enthusiasm about discovering Perplexity AI and linked a blog post praising AI's potential to combat misinformation. They advocate for AI to be used responsibly, nudging people towards clarity and facts, as discussed in both their [blog article](https://figmentums.com/2024/02/01/the-right-direction-for-ai/) and book, *[Pandemic of Delusion](https://figmentums.com/2023/02/23/pandemic-of-delusion/)*.
- **Surprising Top App Trends**: `@bartleby0` commented on the notable absence of Facebook from a list of top apps, calling it "interesting" but did not provide a link or elaborate further on the topic.

**Links mentioned**:

[The Right Direction for AI](https://figmentums.com/2024/02/01/the-right-direction-for-ai/): In this blog and in my book, Pandemic of Delusion, I have focused a lot on AI and particularly on its tremendous potential to shape our thinking for better or for worse. While AI represents a frigh…

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1202546069783707700) (8 messages🔥): 

- **7b Model's Limitations Exposed**: `@brknclock1215` points out that the **7b-online model** tends to fail on complex queries and often relies on its training data, leading to inaccurate outputs or answers.
- **Tweaking Codellama 70b's Response Behavior**: `@general3d` mentions that **codellama 70b** is very restrictive and suggests that the **Perplexity AI team** might be able to tweak it to give more consistent answers by starting responses with "Sure!" or similar prompts.
- **AI Agents as a Workaround**: `@tbyfly` proposes using **AI Agents** to run prompts back and forth until a satisfactory answer is obtained, as a potential solution to codellama 70b's issues.
- **Privacy Concerns Thwart SQL Assistance**: `@jayb1791` complains that **codellama 70b** refused to assist with an SQL question due to data privacy concerns, even though the question merely referenced a business address within the SQL query.
- **Issues with API Credit and Subscription Services**: `@alankarsh` is experiencing problems with API credits not showing up after being charged for a pro subscription, despite trying three different cards and communicating with the customer experience team for assistance.
  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1202665033599815761) (3 messages): 

- **Open Source Embedding Rivals OpenAI's Latest**: LlamaIndex introduces `@nomic_ai`'s *nomic-embed-text-1*, boasting better performance than OpenAI's text-embedding-3-small. The embedding is open source with open data and fully integrated with LlamaIndex; more details are on [Twitter](https://twitter.com/llama_index/status/1753106179008696521).

- **Missed @aiusergroup's Keynote? Watch the Replay**: Conference attendees can now watch the replay of @jerryjliu0's keynote on *Beyond Naive Rag: Adding Agentic Layers* via [YouTube](https://t.co/hrUMF8bq8Q) and access the slides at [this link](https://t.co/P39riIMGK6), as shared by [LlamaIndex Twitter](https://twitter.com/llama_index/status/1753206969161429123).

- **Blog Post on Enhancing RAG with Data Science**: @sudalairajkumar explores the combination of classic data science and retrieval approaches to improve Retrieval-Augmented Generation (RAG), providing guidance on dataset evaluation and embedding model selection in a detailed [blog post](https://t.co/zTPyeaqIbU) highlighted by LlamaIndex.

**Links mentioned**:

- [no title found](https://t.co/hrUMF8bq8Q): no description found
- [LlamaIndex Talk (AI User Conference)](https://t.co/P39riIMGK6): Beyond Naive RAG: Adding Agentic Layers Jerry Liu, LlamaIndex co-founder/CEO

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1202537178282991647) (40 messages🔥): 

- **LLAMA Index Beyond OpenAI**: `@whitefang_jr` confirmed that LlamaIndex can be easily used with other LLMs and pointed to the [guide for integrations](https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#modules) and an [example notebook](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/llm/llama_2_llama_cpp.ipynb) for incorporating Llama-2 with LlamaIndex.
- **Integration Instructions for Local LLAMA Packs**: `@whitefang_jr` provided source code to adjust for integrating LLAMA 2 with the resume_screener pack, and [linked to the GitHub repository](https://github.com/run-llama/llama-hub/blob/e6960d4c64f053d32f1e92aa3a587f7b045cbfff/llama_hub/llama_packs/resume_screener/base.py#L58).
- **Postgres Conundrum in TypeScript/JS Project**: User `@.Jayson` noted issues with Postgres connections and `@whitefang_jr` suggested customizing the PG vector class to address multiple connections; `@cheesyfishes` also mentioned considering connection pooling.
- **Spam Alert Rapid Response**: `@cheesyfishes` took action against a spam alert mentioned by `@mysterious_avocado_98353`, handling the situation by banning and deleting the harmful content.
- **Ingestion Pipeline Configuration with MongoDB**: `@ramihassanein` sought help for an error encountered while setting up an ingestion pipeline with MongoDB Atlas, to which `@cheesyfishes` responded with a solution to update the MongoDB vector store to use the updated base class.

**Links mentioned**:

- [no title found](https://llamahub.ai/l/llama_packs-resume_screener?from=all): no description found
- [llama-hub/llama_hub/llama_packs/resume_screener/base.py at e6960d4c64f053d32f1e92aa3a587f7b045cbfff · run-llama/llama-hub](https://github.com/run-llama/llama-hub/blob/e6960d4c64f053d32f1e92aa3a587f7b045cbfff/llama_hub/llama_packs/resume_screener/base.py#L58): A library of data loaders for LLMs made by the community -- to be used with LlamaIndex and/or LangChain - run-llama/llama-hub
- [Query Transformations - LlamaIndex 🦙 0.9.42.post1](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations.html): no description found
- [Using LLMs - LlamaIndex 🦙 0.9.42.post1](https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#modules): no description found
- [LlamaCPP - LlamaIndex 🦙 0.9.42.post1](https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp.html): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1202588194822950942) (3 messages): 

- **Seeking PII Anonymization Solutions**: `@princekumar13` is looking for methods to anonymize personally identifiable information (PII) from text data before sending it to a large language model (LLM) while maintaining data accuracy. They've found a method using langchain and Presidio but it's experimental and not suitable for production.
  
- **Interest in Specialized LLM for Scheduling**: `@erizvi` inquired about the smallest LLM that could interpret natural language inputs specific to configuring job schedules and then convert them into cron job expressions.

- **Whisper into Text: Understanding Speech-to-Text Models**: `@amgadoz` shared a [blog post](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b) detailing OpenAI's Whisper model, an advanced speech-to-text (STT) system, including insights into its architecture and functionality. The article is the second part of a series, with a focus on the encoder-decoder transformer used by Whisper based on the "Attention is All You Need" paper.

**Links mentioned**:

[Whisper: How to Create Robust ASR (2 / N)](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=profile&utm_medium=reader2): Part 2 of a multi-part series in which we delve deep into Whisper, OpenAI&#x27;s state-of-the-art automatic speech recognition model

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1202524630154223626) (32 messages🔥): 

- **Seeking Resources for AWS Sagemaker and Chatbot Building**: `@refik0727` inquired about tutorials or GitHub repositories for building chatbots with **AWS SageMaker**, using models like **Mistral** and **Llama**, with the ability to integrate one's own CSV file or connect to a local database.

- **Potential LangChain Twitter Security Breach**: Multiple users including `@markopolojarvi`, `@alcazarr`, and `@solac3` alerted the community about a potential scam involving the **LangChain AI Twitter** account being hacked, sharing the same suspicious [Twitter link](https://twitter.com/LangChainAI/status/1753014882696405254).

- **Inquiry about ChatOpenAI and Streaming**: User `@hiranga.g` asked the community if `get_openai_callback()` is compatible with an **Agent** for streaming with **ChatOpenAI**.

- **Proposal for a Dedicated Channel for OpenGTPs Discussed**: `@benjaminbascary` proposed the idea of creating a separate channel within the Discord for **OpenGTPs** discussions.

- **Exploration of Streaming with Llamacpp and Langserve/FastAPI**: `@legendary_pony_33278` sought community help for enabling streaming with **Llamacpp** and **Langserve** or **FastAPI**, to which `@veryboldbagel` responded by sharing a GitHub [example with Ollama](https://github.com/langchain-ai/langserve/blob/main/examples/local_llm/server.py) that may not specifically handle streaming.

- **Guardrails for AI That Teaches Without Giving Answers**: In response to `@_emiya`'s quest to build an AI that teaches students without providing direct answers to math problems, `@deetisaac` recommended looking into **Guardrails**, while sharing the [GitHub repository](https://github.com/guardrails-ai/guardrails).

- **Deep Learning Resources for Enhancing RAG Applications**: After `@daii3696` asked for resources to improve RAG applications, `@johnny2x2` suggested reading the **llama index source code** and announced that they will be hosting a talk about RAG techniques the following Tuesday.

- **Adjusting Max Tokens Limit for ChatOpenAI Agents**: `@irfansyah5572` sought advice on setting the max tokens limit for an agent encountering an `InvalidRequestError`, which `@the_agent_j` addressed by referring to the `maxTokens` parameter in the [LangChain OpenAI API](https://api.js.langchain.com/classes/langchain_openai.ChatOpenAI.html#maxTokens).

- **Question about Disabling Exponential Backoff**: User `@apollo7701` queried the community about ways to disable exponential backoff without any further context or follow-up responses.

**Links mentioned**:

- [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents): no description found
- [ChatOpenAI | LangChain.js - v0.1.12](https://api.js.langchain.com/classes/langchain_openai.ChatOpenAI.html#maxTokens): no description found
- [langserve/examples/local_llm/server.py at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/blob/main/examples/local_llm/server.py): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.
- [GitHub - guardrails-ai/guardrails: Adding guardrails to large language models.](https://github.com/guardrails-ai/guardrails): Adding guardrails to large language models. Contribute to guardrails-ai/guardrails development by creating an account on GitHub.

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1202819089563910244) (2 messages): 

- **LangServe Local LLM Example with Ollama**: `@veryboldbagel` shared a [GitHub link](https://github.com/langchain-ai/langserve/blob/main/examples/local_llm/server.py) to an example of using **LangServe with ollama**, reminding users of the limitations regarding concurrent usage.
- **LangSmith's Waitlist to Shrink**: `@veryboldbagel` announced that more individuals will be taken off the waitlist for **LangSmith** in the coming weeks.

**Links mentioned**:

[langserve/examples/local_llm/server.py at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/blob/main/examples/local_llm/server.py): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1202638561854234646) (9 messages🔥): 

- **Autonomous GPT-4 Agent Platform IX**: `@robot3yes` introduced **Agent IX**, an autonomous GPT-4 agent platform on GitHub, encouraging others to check it out. Visit their work on [GitHub - kreneskyp/ix](https://github.com/kreneskyp/ix).
- **Maximize Token Savings with ContextCrunch**: `@speuce` shared **ContextCrunch**, a prompt compression API that helps save on token costs and integrates with LangChain. Early access and feedback are welcomed at [contextcrunch.com](https://contextcrunch.com/).
- **Job Opportunity at Skipp for OpenAI Full-Stack Developer**: `@marcelaresch_11706` highlighted a job opening at Skipp for a senior full-stack developer with a focus on backend and in-depth knowledge of OpenAI API. The position is full-time within the Pacific Time zone, looking for candidates from LATAM and Europe.
- **Deep Dive into OpenAI's Whisper Model**: `@amgadoz` wrote a detailed blog post on OpenAI's Whisper model for Speech-to-Text, covering the model's architecture and function. The model's insights are available without a paywall on [Substack](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b).
- **Step-Back Prompting Technique Explored**: `@andysingal` published an article on Medium about incorporating Step-Back Prompting with Langchain, discussing the improvements in language processing capabilities. Read the full article on [Medium](https://medium.com/ai-advances/langchain-elevates-with-step-back-prompting-using-ragatouille-b433e6f200ea).

**Links mentioned**:

- [Whisper: How to Create Robust ASR (2 / N)](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=profile&utm_medium=reader2): Part 2 of a multi-part series in which we delve deep into Whisper, OpenAI&#x27;s state-of-the-art automatic speech recognition model
- [GitHub - kreneskyp/ix: Autonomous GPT-4 agent platform](https://github.com/kreneskyp/ix): Autonomous GPT-4 agent platform. Contribute to kreneskyp/ix development by creating an account on GitHub.
- [ContextCrunch](https://contextcrunch.com/): no description found
- [Langchain Elevates with Step-Back Prompting using RAGatouille](https://medium.com/ai-advances/langchain-elevates-with-step-back-prompting-using-ragatouille-b433e6f200ea): A Language Revolution

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1202586642423091250) (2 messages): 

- **Token Count Troubles with ChatOpenAI**: User `@hiranga.g` enquired about **how to get Token counts** on ChatOpenAI Agent during streaming, mentioning that `get_openai_callback()` does not seem to work with it.
- **Compressing LLM Contexts with LangChain**: User `@speuce` shared a [Medium article](https://medium.com/@mrk5199/how-to-compress-llm-contexts-with-langchain-2b58eb84f57b) outlining the process of **Contextual Compression** in **LangChain**, utilizing [ContextCrunch](https://contextcrunch.com) for efficient compression, a solution to the high token usage in Retrieval Augmented Generation (**RAG**) setups.

**Links mentioned**:

[How to Compress LLM Contexts with LangChain](https://medium.com/@mrk5199/how-to-compress-llm-contexts-with-langchain-2b58eb84f57b): In this tutorial, you will learn to reduce token usage by up to 90% using LangChain.

  

---



### DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1202711546078171256) (1 messages): 

- **Migration from 7b to Mixtral Planned**: `@_jp1_` mentioned the decision to **switch from a 7b model to a Mixtral base** due to limitations in the 7b model for certain dimensions, noting that Mixtral offers a better trade-off for evaluation tasks. They also welcomed help with this transition.
  

---


### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1202525729183629332) (15 messages🔥): 

- **API Security Omission Alert**: `@sebastian.bodza` highlighted a potential security concern, pointing out that the **API is not secured**, with no token used in the request.
- **New Embedding Model on the Horizon**: `@bjoernp` showcased **Nomic's long context text embedder** `nomic-embed-text-v1`, which boasts 8192 sequence length and outperforms OpenAI models, complete with open weights, training code, and data ([Nomic AI on Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1)).
- **OLMo 7B Unleashed by Allen AI**: `@_jp1_` introduced the release of **OLMo 7B**, an Open Language Model by Allen AI, alongside its dataset, training code, and a paper ([OLMo on Allen AI](https://allenai.org/olmo/olmo-paper.pdf), [OLMo on Hugging Face](https://huggingface.co/allenai/OLMo-7B)).
- **Quest for the Best German Embedding Model**: In response to `@ustoll` asking for the best German text embedding model for Q/A retrieval, `@philipmay` emphasized that the use case, such as clustering or semantic similarity, is key in determining the appropriate model.
- **Fine-Tuning with RAG and Axolotl**: `@rasdani` described the process of converting the RAG dataset to the ShareGPT format for use with **Axolotl**, noting the original absence of `positive_ctx_idx` during SFT and its later addition for potential but not implemented DPO.

**Links mentioned**:

- [nomic-ai/nomic-embed-text-v1 · Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1): no description found
- [allenai/OLMo-7B · Hugging Face](https://huggingface.co/allenai/OLMo-7B): no description found

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1202723426192597082) (2 messages): 

- **GPU Alternatives for Inference**: User `@_jp1_` discussed options for running inference tasks without dedicated GPUs. They suggested using services like **replicate**, **modal**, or **serverless runpod**, and mentioned the possibility of hosting the model with *together et al.* if there is a demand.
- **Colab Resource Shared**: `@_jp1_` also shared a Google Colab notebook link, but the specific contents or context of the notebook remain unclear due to the lack of details in the message. [Access the Colab Notebook](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK).

**Links mentioned**:

[Google Colaboratory](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK#scrollTo=5i8R0l-XqkgO): no description found

  

---



### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/1202659198660513875) (3 messages): 

- **Nomic Embed Outperforms Ada and Small OpenAI Models**: `@thebaghdaddy` shared a [link to HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1) featuring **nomic-embed-text-v1**, an open source text encoder with a context length of 8192 that claims to outperform OpenAI's *text-embedding-ada-002* and *text-embedding-3-small*, with performance metrics provided.
- **Performance Metrics Table Presented**: According to the source, **nomic-embed-text-v1** scored 62.39 on MTEB and 85.53 on LoCo benchmarks, indicating its superior performance. The table also highlights the accessibility of open weights, training code, and data for this model.
- **Intrigued by Open Source Encoder**: `@thebaghdaddy` expressed interest in personally testing the new **nomic-embed-text-v1** model.
- **Searching for an Embeddings Testing Ground**: `@thebaghdaddy` inquired about the existence of an "embeddings arena," similar to the LLM arena leaderboard, showcasing competitive models.

**Links mentioned**:

[nomic-ai/nomic-embed-text-v1 · Hugging Face](https://huggingface.co/nomic-ai/nomic-embed-text-v1): no description found

  

---


### LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/) (1 messages): 

firefox8975: Has anyone tried or is there a guide on how to deploy oss models to aws using vllm?
  

---


### LLM Perf Enthusiasts AI ▷ #[irl](https://discord.com/channels/1168579740391710851/1171569983688560732/1202671582510583840) (2 messages): 

- **Exa (formerly Metaphor) Throws Toga-Themed Launch Parties**: `@sarahchieng` invites everyone to **Exa's** coast-to-coast launch parties featuring an **Acropolis room**, a **DJ**, and "truth-inducing serums." The [SF party](https://partiful.com/e/7qDnQGjE1MdU32Cei0J0?%3C%3CFriday,%20Feb%202) is on **February 2**, and the [NYC party](https://yuzu.party/1FEOWfzNCHm3Fi6vtrSs) on **February 7**; both require an RSVP.

- **Coffee on Exa in SF and NYC**: In addition to the launch party, `@sarahchieng` offers to meet up for coffee in both **San Francisco** and **New York City** and states the costs are "on me...haha."

**Links mentioned**:

- [RSVP to Exa Greco-Roman Toga Launch Party | Partiful](https://partiful.com/e/7qDnQGjE1MdU32Cei0J0?): Same mission, new name! Come join the Exa (prev. Metaphor) team for the official Exa Launch Party! Exa&#x27;s mission is to organize the world&#x27;s knowledge... and so naturally this is a TOGA party...
- [Exa East Coast launch party](https://yuzu.party/1FEOWfzNCHm3Fi6vtrSs): Come join the Exa (prev. Metaphor) team for the official Exa Launch Party! Swag, snacks, and Exa credits provided :)  Launch post: https://twitter.com/ExaAILabs/status/1750942315055882554  Who we are:...

  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1202660583871217724) (2 messages): 

- **Database Transition in Progress**: `@michelcarroll` revealed that they are currently using **Weaviate** but are in the process of migrating to **pgvector with HNSW**, in response to `@.psychickoala`'s inquiry about the vector database being used.
  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1202839633084289075) (5 messages): 

- **Saving and Reusing Chain of Thought**: `@byronhsu` inquired about the outcomes of saving a model's chain of thought (CoT) and then using it as input for subsequent prompts. The aim is to reduce the time and cost of multiple iterations, questioning whether this technique affects the reasoning abilities of the model.
- **Latency Tradeoff in CoT Approach**: `@justahvee` confirmed that feeding a previously saved CoT into a second step works but noted a tradeoff with inference latency due to multiple calls (the original generation and the follow-up inference).
- **Clarifying Inference Latency Concerns**: `@byronhsu` sought clarification from `@justahvee` on the inference latency issue, assuming that feeding in a CoT directly should result in faster inferences.
- **Explaining the Latency Tradeoff**: `@justahvee` explained that the latency experienced is due to the total time of the initial CoT generation and the second inference using the CoT, as it involves multiple separate calls.
- **Assurance on Accuracy Retention**: `@byronhsu` received confirmation from `@justahvee` that accuracy is retained when CoT output from one step is used as input for the next step.
  

---



### CUDA MODE (Mark Saroufim) ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1202789393975484446) (7 messages): 

- **A Warp-Speed Experiment**: `@zippika` shared[ code](https://example.com/link-to-code) for `rgb_to_grayscale` CUDA kernel optimizations, utilizing **ulong** for vectorized loads per warp. This unique approach loads three pixels at a time, aiming to increase efficiency.
- **Crank up that Occupancy**: Subsequent modifications led to a much higher **occupancy** with a calculated theoretical of **100%** but a measured achieved occupancy of **77%**, indicating better utilization of GPU resources in `@zippika`'s experiments.
- **Throughput Under the Microscope**: `@zippika` posted the **GPU Throughput** analysis, highlighting **Memory** utilization at **71.87%** and **Compute (SM)** usage at **46.36%**, reflecting the comprehensive workload distribution of their GPU-based computation.
- **Speed Bumps Ahead**: Despite the conceptual success, `@zippika` mentioned not having tested the speed and later followed up to report that the new implementation appears to be **slower** in practice.
- **Thinking Out Loud with Emoji Support**: The user apologized for the seemingly disorganized messages, expressing that the comments were part of a thought process, accompanied by a unique emoji (`<:floom:799302782477402142>`), indicating a mix of informal communication and technical content.
  

---


### CUDA MODE (Mark Saroufim) ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1202774545870749736) (3 messages): 

- **CUDA Setup Success**: User `@noobpeen` announced they've successfully set up **CUDA 12.2 with Visual Studio** and is inquiring about the next steps to follow.
- **Experience with PyTorch and C++**: `@noobpeen` shared their familiarity with PyTorch and proficiency in C++ to give context to their CUDA setup progress.
- **Advice to Dive into CUDA Development**: In response to `@noobpeen`, `@lancerts` suggested creating a new CUDA project and beginning with CUDA kernels development, followed by recommending the journey through the *Professional CUDA C Programming* book.
  

---


### CUDA MODE (Mark Saroufim) ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1202535400065728512) (1 messages): 

- **Optimizing Memory Loads via Thread Coarsening**: `@tvi_` explained that thread coarsening is employed to **increase tile size** and reduce the global memory load frequency by handling more output pixels per thread. This approach is linked to the concept of **"work efficiency"**, which is discussed later in the book, underscoring its importance in memory operations as opposed to just compute tasks.
  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/) (1 messages): 

dbreunig: https://www.dbreunig.com/2024/02/01/pursuing-quiet-ai.html
  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1202686909344907274) (2 messages): 

- **Datasette PDF provided to GPT**: `@simonw` shared an experience where they provided a **GPT with the PDF version** of the [Datasette documentation](https://docs.datasette.io/en/stable/), but the results were not very satisfactory.
- **Hopeful Yet Cautious Optimism**: Despite the initial underwhelming results, `@simonw` conveyed a belief that with substantial effort, feeding a GPT with Datasette's PDF documentation could potentially be effective.

**Links mentioned**:

[Datasette documentation](https://docs.datasette.io/en/stable/): no description found

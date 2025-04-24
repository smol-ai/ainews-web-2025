---
id: 17ad79bf-7163-457b-91d4-ced9e9bb7e2d
title: Mistral Large disappoints
date: '2024-02-26T21:59:34.252178Z'
original_slug: ainews-mistral-large-disappoints
description: >-
  Mistral recently launched Mistral Large, a new language model achieving 81.2%
  accuracy on MMLU benchmarks, trailing GPT-4 Turbo by about 5 percentage points
  but at similar cost, leading to mixed community reception and skepticism about
  open sourcing. Additionally, technical discussions in TheBloke Discord
  highlighted challenges in LLM training, such as DeepSpeed memory errors and
  advanced model merging techniques, reflecting ongoing efforts to optimize
  large language model development.
tags:
  - mistral
  - mistral-large
  - gpt-4
  - gpt-4-turbo
  - llm
  - training
  - model-merging
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/21-23/2024. We checked **20** guilds, **318** channels, and **15439** messages for you. Estimated reading time saved (at 200wpm): **1430 minutes**.

Mistral came out swinging today [announcing Mistral-Large](https://twitter.com/GuillaumeLample/status/1762128616849072171/photo/2) on La Plateforme and on Azure, trailing GPT4 about 5 percentage points on their aggregated benchmarks:

 ![image.png](https://assets.buttondown.email/images/23f5530d-5b50-49ed-ac55-12e9b279c76d.png?w=960&fit=max) 


The community reception has been mildly negative.

 ![image.png](https://assets.buttondown.email/images/c92ba048-d2ad-4783-bb90-66d346f50173.png?w=960&fit=max) 

And hopes are not high for open sourcing. Notably, Mistral are also claiming that the new Mistral-Small is "significantly better" than the openly released Mixtral 8x7B. 

 ![image.png](https://assets.buttondown.email/images/bb29bff3-b2fa-40d8-b51c-115410b3fe59.png?w=960&fit=max) 


---

**Table of Contents**

[TOC] 


# PART 0: Summary of Summaries of Summaries

<div><h2><strong>Evaluating LLMs Performance and Cost-Efficiency</strong>:</h2><p>The discussion in <strong>TheBloke</strong> Discord underscores the comparative analysis between <strong>Mistral Large</strong> and <strong>GPT-4 Turbo</strong>, with <strong>Mistral Large</strong>'s performance on benchmarks like MMLU falling short despite similar cost implications, suggesting a reevaluation of cost-benefit for users and developers alike.</p><h2><strong>Technical Training Hurdles and Best Practices</strong>:</h2><p>Challenges in implementing <strong>DeepSpeed</strong> to avoid out-of-memory errors and the application of <strong>DPO</strong> using the <code>DPOTrainer</code> highlight the technical intricacies and community-driven solution sharing, illustrating the ongoing efforts to optimize LLM training efficiency and practicality.</p><h2><strong>Advancements in AI Deception for Roleplay Characters</strong>:</h2><p>The dialogue on creating AI characters capable of deception, especially with the application of survival goals, reflects the nuanced exploration of AI's narrative capabilities. The use of <strong>DreamGen Opus V1</strong> despite tokenizer and verbosity issues underscores the creative pursuits in AI storytelling.</p><h2><strong>Intricacies of Model Merging</strong>:</h2><p>The discourse led by community members on merging non-homogenous models using strategies like <strong>linear interpolation</strong> and <strong>PEFT merging methods</strong> reveals a deep dive into the complexities and potential of enhancing LLMs through model integration, marking a significant area of exploration within AI development practices.</p></div>

# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Evaluating LLMs Performance and Cost-Efficiency**: The performance of **Mistral Large** was compared unfavorably to **GPT-4 Turbo** by `@timotheeee1`, suggesting it may not be worth the similar costs given its performance on benchmarks like MMLU.

- **Technical Training Hurdles and Best Practices**: Issues with DeepSpeed OOM errors and discussions around practical implementations of DPO using the `DPOTrainer` from the `trl` [Hugging Face library](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py) prompted sharing of insights and resources among users such as `@cogbuji` and `@plasmator`.

- **Advancing AI Deception in Roleplay Characters**: Dialogues on creating AI characters that can convincingly lie highlighted the improvements when applying explicit survival goals. Challenges encountered when using the [DreamGen Opus V1](https://huggingface.co/dreamgen/opus-v1-34b) model were discussed, along with tokenizer issues and verbosity in AI storytelling.

- **The Intricacies of Model Merging Explored**: Discussions led by `@jsarnecki` and `@maldevide` delved into the complexities of merging non-homogenous models and various strategies for successful mergers, like linear interpolation. The limitations and possibilities were articulated, drawing on resources like [mergekit](https://github.com/arcee-ai/mergekit) and advancements in PEFT merging methods outlined in a [Hugging Face blog post](https://huggingface.co/blog/peft_merging).

- **Engineers Pine for the Past While Gazing Toward AI's Future in Decompilation**: Reminiscences of OllyDbg's features by `@spottyluck` contrast with excitement for potential **AI-assisted decompilation** expressed by `@mrjackspade`. The suggestion to use a large volume of open-source projects for creating AI training data sets demonstrates forward-thinking for advancing AI capabilities in code reconstruction.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

**Mistral Large Takes the Stage**: The introduction of **Mistral Large**, a highly optimized language model with an 81.2% accuracy on MMLU and features such as multilingual capabilities and native function calling, stirred interest and discussion across the community. It's available for use via platforms such as [la Plateforme](https://mistral.ai/news/mistral-large/).

**Technical Hurdles & Triumphs in LLM Deployment**: Members shared experiences and exchanged technical advice on the challenges of deploying Mistral models, such as the **Mistral7x8b** and **Mistral-7B-Instruct**, on various hardware setups including Tesla V100s and local machines with limited VRAM. Tips on adjusting layer sharing, precision levels, and dealing with freezing issues were exchanged, highlighting the technical nuances of high-performance model usage.

**Fine-Tuning Finesse**: The community discussed fine-tuning practices, emphasizing the need for experimentation and adequate data quantities, with suggestions pointing to around 4000 instances for specific tasks. There was also a focus on the right data format for fine-tuning with Mistral models, and the necessity of understanding advanced fine-tuning techniques like LoRA.

**Contemplating Commercial Impacts & Open Access**: Conversations around Mistral's shift towards more business-oriented, closed-weight models like Mistral Small and Large surfaced concerns about the future of open models. However, many members are hopeful for the continued support of open model development despite big tech partnerships.

**Mistral API Insights and Queries**: Queries related to the **Mistral API** were numerous, ranging from concerns about data privacy, with confirmations that data isn't used for model training, to functional inquiries about running Mistral on local machines without GPUs. There was also a discussion on third-party offerings and potential integrations for extending Mistral's capabilities.

**User-Driven Design and Application Ideas**: The community actively shared ideas for new applications and enhancements, including the development of plugins and mobile apps that leverage Mistral. One user proposed adding a language level setting to **Mistral's Le Chat** and there's a buzz around the feature simplicity of **Mistral-Next** within Le Chat, which could indicate a user preference for streamlined AI products.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Troubleshooting LM Studio's White Window Woe**: User `@steve_evian` encountered an issue where **LM Studio** launched to a white window; `@heyitsyorkie` recommended clearing .cache and %APPDATA%, which resolved the issue.

- **Exploring Multilingual LLM Presence**: `@.deug` queried about pre-trained multilingual LLMs with Korean support; `@heyitsyorkie` noted a scarcity of LLMs proficient in Korean to English translation, recommending combining **LM Studio** with an online translator like **DeepL**.

- **LM Studio API Refuses to Run Headless**: `@muradb` inquired about headless operation for **LM Studio API**; `@heyitsyorkie` clarified the current version doesn't support it, while `@krypt_lynx` expressed desire for open-source and headless features, confirmed to be unavailable by `@heyitsyorkie`.

- **Hyperparameter Evaluation Remains a Personal Choice**: `@0xtotem` pondered the proper dataset for hyperparameter evaluation for a RAG model—with consensus leaning towards using the closest data available, as specific guidance was lacking.

- **GPU Wars: Nvidia Faces Off AMD in User Preferences**: The suitability of AMD's GPUs for running LLMs was debated; users showed a general preference for **Nvidia** due to the ease of **AI** applications setup, despite speculation about AMD working on CUDA alternatives.

- **A Collective Effort to Aid IQ Models in LM Studio**: `@drawless111` succeeded in making IQ models work and offered guidance on locating specified formats on HGF; others discussed improvements and updates to various models and tools like **llama.cpp**.

- **Online Reinforcement Learning Without File System Access**: `@wolfspyre` asked about **LM Studio's** capability for local file system access; it was clarified that LLMs don't have this capability, nor does LM Studio support executing commands from LLMs.

- **AutoGen Anomalies Addressed with a Classic Reboot**: Users shared troubleshooting tips for **AutoGen errors**, including reinstalling packages and the reliable "turn it off and on again" strategy, as humorously depicted in a [Tenor GIF](https://tenor.com/view/it-problem-phone-call-have-you-tried-turning-it-off-and-on-again-gif-17823069).

- **Seeking Support for Langchain's RAG Utilization**: In a one-message brief mention, bigsuh.eth inquired about using **RAG** within Langchain via **LM Studio**, but no discussions or answers followed.

- **Open-Interpreter Connectivity Conundrum Cracked**: User `@nxonxi` dealt with connection errors and syntax mistakes when trying to run Open Interpreter with `--local` flag; after troubleshooting, simple Python requests worked as a solution.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Discover Daily Podcast Unveiled**: Perplexity AI, in partnership with [ElevenLabs](https://elevenlabs.io), launched the **Discover Daily podcast**. The episodes sourced from Perplexity's [Discover feed](https://www.perplexity.ai/discover) are available on [podcast.perplexity.ai](https://podcast.perplexity.ai), featuring daily tech, science, and culture insights using ElevenLabs' voice technology.

- **Sonar Models Spark Debate**: New `sonar-small-chat` and `sonar-medium-chat` models along with their search-enhanced versions were introduced by Perplexity AI, leading to community comparisons with `pplx-70b-online`. Users reported incoherent responses from sonar models, requesting not to phase out `pplx-70b-online` due to its better performance and mentioning that fixes for sonar models were underway as per community insights and the [API Updates](https://docs.perplexity.ai/changelog/api-updates-february-2024).

- **Gibberish Responses from Sonar Models Under Scrutiny**: Users like `@brknclock1215` suggested possible mitigation of gibberish outputs by limiting response length, which contrasted with the stable output quality of pplx models even at longer lengths. Meanwhile, API users discussed fetching model details programmatically for improving user interface selections.

- **Engagements in #general Rife with AI Chat Model Discussions**: The community engaged in various discussions including the retirement of **Gemini** in favor of possible Gemini Ultra, inconsistencies in model responses across different platforms, and leveraging Perplexity's **Pro** capability for image generation.

- **Assorted Inquiries and Tests in Sharing**: Members in the **sharing** channel delved into a mix of topics like exploring user guides for Perplexity topics, questioning the novelty of Lenovo's technology, and sharing mixed use cases leveraging the AI for personal assistance and technical inquiries.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **VPN Interference with OpenAI Services**: `@strang999` encountered an error using OpenAI services, which `@satanhashtag` attributed to potential VPN interference and suggested disabling web protection in VPN settings.

- **GPT-4 Context and Captcha Challenges**: `@orbart` and `@blckreaper` are frustrated with ChatGPT's reduced memory for narrative work, suspecting a decrease in tokens processed, while `@little.toadstool` and `@necrosystv` reported cumbersome captcha tests within ChatGPT.

- **Quest for Image-to-Video and Data Privacy Concerns**: `@sparkette` was looking for a browser-based image-to-video generator and `@razorbackx9x` asked about AI for sorting credit report data, with `@eskcanta` cautioning against uploading sensitive personally identifiable information (PII).

- **Navigating Custom GPT and Assistant Differences**: Users noted inconsistencies between Custom GPTs and Assistant GPTs in handling formatting and markdown, particularly when generating tables or images, with advice to refer to specific API configurations.

- **Anticipating Sora and Protecting Prompts**: The community is curious about the capabilities of OpenAI's Sora and discussed the feasibility of protecting custom prompts with `@.dunamis.` and `@kyleschullerdev_51255` agreeing that complete protection isn't possible, suggesting a layered web application for security instead.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Watch Out for Crypto Scams**: One post in the #learning-ml channel from `@josephsweeney11` appears to be a **potential scam** involving making $40k in 72 hours and should be approached with **extreme caution**.

- **Understanding Transformers' Learning Capabilities**: In the #learning-ml channel, `@phryq.` inquired about experiments training transformers to understand size relationships to enhance image generation, using hypothetical objects.

- **New Snap Video Project Unveiled**: A new project termed Snap Video was discussed in the #general channel, addressing challenges in video generation with a transformer-based model, and sharing the project [link](https://snap-research.github.io/snapvideo/#title-footer) and related [research paper](https://arxiv.org/pdf/2402.14797.pdf).

- **Debate Over Optimal CLIP Filtering Techniques**: In the #research channel, the discussion revolved around whether CLIP filtering is suboptimal compared to image-text pair classifiers, with reference to a recently published DFN paper in the conversation.

- **Gradient Precision: bfloat16 vs fp32 Debate**: Conversations in the #research channel have touched on the use of **autocasting on TPUs** with bfloat16 gradients and compared its performance against the default fp32 gradients in PyTorch's **autocast behavior**.

- **Sharing of AI Research Papers and Methods**: Across the channels, participants shared insights and resources on various AI research topics such as **state space architecture**, **Transformer optimization**, **AI-generated text detection**, and discussions around making **LLMs significantly cheaper**, with links to resources like [Mamba-ND](https://arxiv.org/abs/2402.05892), among others.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Democratization of AI Hardware Sparks Intense Debate**: In the discussion surrounding the potential of creating proprietary TPUs and the democratization of hardware, parallels to the car and RAM industries elicited skepticism regarding tech promises from companies like Samsung. The importance of such advancements was underscored, given their impact on AI capabilities and access.
  
- **Towards Accessible and Practical AI Solutions**: Several initiatives, including the creation of **Galaxy AI**, offering free API access to models such as **GPT-4, GPT-3.5**, and **Galaxy AI's Gemini Pro**, to the presentation of **surya**, an OCR and line detection project that supports over 90 languages, are aimed at making AI tools more accessible and practical for various applications, as explained in [this GitHub repository](https://github.com/VikParuchuri/surya).

- **Neural Network Innovations & Model Finetuning Challenges**: From introducing the support for WavLMForXVector in browsers to reviewing **Peft's library** for new merging methods for LoRA, there's a clear focus on model deployment and improving AI performance. Finetuning difficulties, whether with **Flan T5** producing incoherent output or a zigzag loss graph in **Qwen1.5-0.5B**, remain pivotal points of discussion.

- **Cross-disciplinary AI Projects Garner Attention**: Projects that integrate AI with specific disciplines, such as **Unburn Toys**, an open-source AI toolbox, or the **TTS Arena** for comparing TTS models, signify a cross-functional approach to AI development. This is complemented by the release of datasets for niche applications like philosophical Q&A, available on [Hugging Face here](https://huggingface.co/datasets/sayhan/strix-philosophy-qa).

- **Knowledge Sharing and Collaborative Growth in AI Communities**: Whether it's a query on **imitation learning for robotics**, the use of **AnimeBackgroundGAN**, the issues related to multi-language OCR, or the **Japanese Stable Diffusion model**'s approach to training in a new language, it's evident that AI communities serve as invaluable forums for sharing knowledge, solving problems, and fostering collective progress in the field.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Batching Blues with GPT-4**: `@rwamit` raised concerns about increasing processing time from 2s to 60s per iteration when implementing batching to query GPT-4 using langchain wrapper, ballooning the task from 5 to 96 hours for 5-6k records.

- **Intrigue in Initialization**: A particular code piece in [Gemma's PyTorch implementation](https://github.com/google/gemma_pytorch/blob/01062c9ef4cf89ac0c985b25a734164ede017d0b/gemma/model.py#L176) involving RMSNorm sparked discussions about the significance of the addition of +1 to the normalization process.

- **EfficientNet's Efficacy**: A debate arose over EfficientNet's merits, with `@vapalus` defending its use in segmentation tasks despite criticism from `@fern.bear` regarding its marketing versus performance.

- **Mistral Large Debuts**: The release of _Mistral Large_ was announced, a model acclaimed for its strong text generation performance and availability on la Plateforme and Azure. Check out [Mistral Large](https://mistral.ai/news/mistral-large/) for additional insights.

- **DPO Paper and SFT**: Clarity was sought by `@staticpunch` about `model_ref` initialization in DPO, with confirmation that Supervised Fine-Tuning (SFT) on preferred completions should precede DPO, as discussed in the DPO paper.

- **Diving Deeper into GRUs**: `@mrgonao` showed curiosity about why gated units like GRUs are termed as such, yet explanations regarding their etymology remained elusive within the channel.

- **The Search for Smarter Search**: The "Searchformer" paper [Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping](https://arxiv.org/abs/2402.14083) describes how a Transformer-based model can outperform traditional A* methods in solving puzzles, offering an innovative approach to search problems.

- **RLHF and the Simplicity Debate**: A paper advocating for simpler REINFORCE-style optimization over Proximal Policy Optimization (PPO) for RLHF triggered discussions on the efficiency of fundamental methods in RL for language models. The paper is accessible [here](https://arxiv.org/abs/2402.14740).

- **Watermarking Frameworks Face-off**: The landscape of text watermarking for large language models was shared, featuring techniques for embedding detectable signals in generated text and analyzing the robustness of such watermarks.

- **Tales of GPT-NeoX and Python**: Amidst hesitation about upgrading to **Python 3.10**, conversations in development veered towards preferences for a **custom training loop** over GPT-NeoX, showing an active engagement with the finer points of AI development optimization.

- **Multilingual Matters in Tokenization**: Queries about optimizing the **Mistral tokenizer** for better multilingual representation underscored ongoing efforts to enhance language model capabilities beyond English, indicating a focus on global applicability.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Create-llama Eases Full-Stack Development**: The newest create-llama release integrates **LlamaPack**, streamlining the construction of full-stack web apps through the inclusion of advanced **RAG** concepts with minimal coding. The announcement was shared in a [tweet by @llama_index](https://twitter.com/llama_index/status/1761159412629336404).

- **Counselor Copilot Leverages Advanced RAG**: **Counselor Copilot** project, highlighted in a [tweet](https://twitter.com/llama_index/status/1761433854458614075), distinguishes itself by utilizing advanced **RAG** to assist crisis counselors, showcasing a use case as a co-pilot rather than a basic chatbot.

- **RAG Retrieval Enhanced by Summaries**: To improve RAG retrieval, a technique using sub-document summaries helps tackle global concept awareness problems that arise from naive chunking. This approach is detailed in a [tweet](https://twitter.com/llama_index/status/1761793821422264757) discussing the consequential boost in contextual awareness of each chunk.

- **LlamaParse Masters Complex PDF Parsing**: **LlamaParse** has been introduced as a powerful tool for parsing PDFs with complex tables and figures, crucial for high-quality RAG applications. Accurate table representations aid the LLM in providing correct answers, as stated in a [tweet](https://twitter.com/llama_index/status/1762158562657374227).

- **Challenges with Kafka's Protagonists in AI**: In a discussion regarding generating a book review for Kafka's "Metamorphosis," `@daguilaraguilar` faces trouble with the AI incorrectly highlighting "Grete" as the protagonist instead of "Mr. Samsa," referencing their [code](https://www.gutenberg.org/cache/epub/5200/pg5200.txt). 

- **Insights into Financial Document Analysis and Context Management**: [SEC Insights](https://www.secinsights.ai/) brings advanced capabilities for analyzing financial documents, and there is a call within the community for benchmarks related to best practices in context management for large-window LLMs such as GPT-4 turbo and Gemini 1.5.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

**Sora's Consistency Questioned**: In a correction to a [WSJ video](https://youtu.be/XllmgXBQUwA), `@swyxio` pointed out that OpenAI's Sora maintains consistency over more than 1-minute videos by interpolating from a start image.

**NVIDIA's GEARing Up**: NVIDIA announced a new research group, **GEAR** (Generalist Embodied Agent Research), co-founded by Dr. Jim Fan, focusing on autonomous machines and general-purpose AI.

**AI-Generated Podcasts Hit the Airwaves**: Perplexity has launched an [AI-generated podcast](https://podcast.perplexity.ai/), drawing content from their Discover feed and employing ElevenLabs' voices for narration.

**One Line of AI Code with Cloudflare**: Cloudflare's new [AI Gateway](https://developers.cloudflare.com/ai-gateway/) has been introduced, featuring easy integration via a single line of code for AI analytics and insights.

**AI Takes on Data Analysis with GPT-4-ada-v2**: A new tool - [ChatGPT Data Analysis V2](https://x.com/btibor91/status/1761726596585504939?s=46&t=90xQ8sGy63D2OtiaoGJuww) enhances data analysis by offering targeted replies and data grid overlay editor, possibly implementing interactive charts and leveraging `gpt-4-ada-v2`.

**LLM Paper Club T5 Session Recap**: A recent LLM Paper Club session led by `@bryanblackbee` dissected the T5 paper with discussions encapsulated in shared [Notion notes](https://www.notion.so/blackbeelabs/Paper-T5-25d26c7d49f7474bb18c90b16eb10413?pvs=4). Open inquiries included model vocabulary, fine-tuning processes, and architecture differences for NLP tasks.

**Local Model Enthusiasts Convene in AI in Action Club**: The "AI in Action" event highlighted local model exploration, tooling discussions for local AI models, and references to model fine-tuning with LoRAs deploying tools like `ComfyUI`. The Latent Space Final Frontiers event was announced, inviting teams to push the boundaries of AI with an application link [here](https://lu.ma/latent-space-final-frontiers).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Gradient Clipping Woes & DeepSpeed Query**: An issue with gradient clipping set to 0.3 was discussed with suspicions of temporary spikes; meanwhile, a [GitHub issue](https://github.com/huggingface/transformers/issues/29254) about HuggingFace's Trainer supporting DeepSpeed Stage 3 incited feedback on usage and updates. Axolotl's cache clearing techniques were also shared, using `huggingface-cli delete-cache`.

- **Strategic Shifts at Mistral AI?**: Discussions surfaced regarding a strategic partnership between Microsoft and Mistral AI, centering on potential implications for open-source models and the commercial direction of Mistral AI. Links to a [Twitter post](https://fxtwitter.com/casper_hansen_/status/1762159643344662859) and news article were shared for further insight.

- **Ease of Access with Axolotl's Auto-Install**: The Axolotl project saw improvements with the introduction of `auto_install.sh` to simplify installations, showing commitment to non-Python developer support. A [Twitter post](https://twitter.com/casper_hansen_/status/1761700050458103964) sought community support for the CUDA mode series with the potential assistance of Jeremy Howard.

- **GPUs, Dockers, and Newbies**: Technical issues regarding GPUs, such as long training times and high loss, Docker container complications, and the desire for a beginner-friendly Axolotl tutorial were prominent. Hugging Face's reported checkpoint save error issue [#29157](https://github.com/huggingface/transformers/issues/29157) and Axolotl's GitHub [#1320](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1320) were among the key references.

- **Community Highlights Korean Expansion & RAG Features**: A fine-tuned phi-2 model without a model card was announced, EEVE-Korean models were touted for extended Korean vocabulary, and R2R Framework for RAG system development was introduced. The supporting [arXiv technical report](https://arxiv.org/abs/2402.14714) and various [Hugging Face models](https://huggingface.co/yanolja) were provided to the community.

- **Runpod Hits a DNS Hitch**: A *NameResolutionError* on **runpod** suggested DNS resolution issues possibly involving proxy settings when trying to reach 'huggingface.co' were reported.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **CUDA Under Fire**: Computing legend **Jim Keller** criticized NVIDIA's **CUDA** architecture in a [Tom's Hardware article](https://www.tomshardware.com/tech-industry/artificial-intelligence/jim-keller-criticizes-nvidias-cuda-and-x86-cudas-a-swamp-not-a-moat-x86-was-a-swamp-too), suggesting it lacks elegance and is cobbled together. Meanwhile, the introduction of **ZLUDA**, which enables CUDA code to run on AMD and Intel GPUs, was open-sourced with hopes to challenge NVIDIA's AI dominance ([GitHub link](https://github.com/vosen/ZLUDA)).

- **Gearing Up with GPUs**: Debates surfaced regarding GPU choices for AI with the **4060 ti** being the cheapest 16GB consumer GPU and the **3090** offering 24GB VRAM as a stronger alternative for LLM tasks. Discussions were also vibrant around second-hand GPU buying strategies and potential technical remedies when issues arise.

- **Quantized Computation Conversations**: Clarity surfaced on how computations in quantized models maintain accuracy, and discussions around implementing efficient CUDA kernels through `torch.compile` by detecting patterns were prominent. The speed of CUDA kernel compilation was also a topic, with methods to reduce compile times from over 30 seconds to under 2 seconds proposed ([repository link](https://github.com/pbridger/cuda-experiments)).

- **Triton Tinkering**: Interest piqued in **Triton**, a tool for enabling **Jax** support via Pallas and its comparison to CUDA for multi-GPU/node execution. There were calls for experts to explain the lower-level workings of Triton, its foundation in **LLVM and MLIR**, and to create benchmarks for its quantized matmul kernel.

- **Flash Attention Finessed**: Within ring attention discussions, a `zigzag_ring_flash_attn_varlen_qkvpacked_func` implementation showed speed improvements. A Hugging Face document detailed memory efficiency benefits ([Flash Attention Visual](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention)), and benchmarks indicated a 20% speed up over classical flash attention ([benchmark link](https://github.com/zhuzilin/ring-flash-attention/blob/main/benchmark_qkvpacked_func.py)).

- **CMU's Paper on Efficient LLM Serving**: A paper from CMU on efficient methods in deploying generative LLMs was shared, titled "Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems" ([arXiv link](https://arxiv.org/pdf/2312.15234.pdf)), surveying over 150 works on techniques including *non-autoregressive generation* and local attention variants.

- **Learning Efficiency Through MIT**: An MIT course on efficient AI computing was unveiled, covering model compression, pruning, quantization, and providing hands-on experience with models like **LLaMa 2** and touching quantum machine learning topics ([course link](https://hanlab.mit.edu/courses/2023-fall-65940)).

- **CUDA-MODE Lecture Announcements and Learnings**: Lecture 7 on Quantization titled *Quantization CUDA vs Triton* was announced, emphasizing the discourse on efficient techniques in AI computations with quantization at the forefront. Lecture content was supplemented by YouTube videos and easily accessible slide presentations, fostering continued education in the community ([YouTube Lecture 6](https://www.youtube.com/watch?v=hIop0mWKPHc), [Lecture 7](https://youtu.be/1u9xUK3G4VM?si=ssW_DEDqBIRHpNYN)).

- **Job Prospects and Queries**: Nvidia was confirmed to be looking for CUDA and C++ experts, inviting applicants to DM their CV for **JobID: JR1968004**. Questions around hiring status for companies like **Mistral** were floated, underlining the employment buzz within the AI engineering sector.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Exploring Function Calls in AI Models**: Engineer `@saita_ma_` is looking for ways to execute function calls with local models such as **OpenHermes**, inspired by what **CrewAI** achieved. Meanwhile, `@kenwu_` shared a [Google Colab](https://colab.research.google.com/drive/14IOr0PZY9Skpc7IjxSeN-GZekNoI3I1U?usp=sharing) seeking assistance on agent and function calling using **Cohere API** and LangChain.

- **LangChain Integration in Various Projects**: The creation of a personalized chatbot implementing OpenAI, Qdrant DB, and Langchain JS/TS SDK was shared by `@deadmanabir`, while `@david1542` introduced [Merlinn](https://merlinn.co/), a machine learning tool to support on-call engineers. Furthermore, `@edartru.` offered [Langchain-rust](https://github.com/Abraxas-365/langchain-rust), a crate that allows Rust developers to use large language models in programming.

- **Tutorial Resources Promote DIY AI Projects**: A recent [YouTube tutorial](https://youtu.be/n9AMtXLveMs) shows viewers how to create a ChatGPT-like UI using ChainLit, LangChain, Ollama, & Gemma. `@rito3281` wrote about using LLMs for finance analysis in the insurance industry, and `@tarikkaoutar` posted a [video](https://www.youtube.com/watch?v=q5LvDHiSBy4) on creating a multi-agent application involving LangGraph.

- **Sarcasm Detection and Timeout Extensions in LLMs**: There was a suggestion to tag phrases with "sarcasm" for better LLM detection post-fine-tuning, but further discussion on the mechanics was not provided. A query about extending the default 900-second timeout was raised, yet no subsequent solutions or elaborations were found.

- **Emerging Tools and Use Cases Explored by Developers**: `@solo78` invites collaborative discussions on AI implementations in the insurance sector's finance function. An AI-powered [resume optimizer](https://github.com/AnshKetchum/resumeop) that helped secure interviews at tech companies was introduced by `@eyeamansh`.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **ChatGPT Multilingual Mishaps**: Users noted that **chatgpt-3.5-turbo** sometimes mistranslates document titles, with one instance changing "Taking Advantage of the Internet" to "*Sacándole Provecho a Internet*". The suggested workaround is to use a system prompt specifying **"Always use English"** to prevent such language detection errors.

- **Prompt Crafting Nostalgia & Fixes**: `@tariqali` discussed the benefits of old school prompt crafting for better control in light of chatbot "time out" issues. Meanwhile, `@derekpwillis` and `@simonw` conversed about devcontainer configuration, with `@simonw` recommending the addition of `llm models` to the `setup.sh` script, which `@derekpwillis` implemented to solve certain bugs.

- **Aspirations for LargeWorldModel on LLM**: There is interest in running the [LargeWorldModel](https://largeworldmodel.github.io/) on LLM, possibly leveraging GPU instances for PyTorch models, as discussed by `@simonw`. He referenced the models' availability on the [Hugging Face repository](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-1M).

- **Groq Inference Plugin Debuts**: `@angerman.` released a Groq inference plugin, [llm-groq](https://github.com/angerman/llm-groq), with the community showing support and curiosity regarding its performance capabilities.

- **llm-groq Plugin Hits PyPI**: Following advice from `@0xgrrr`, `@angerman.` published his [llm-groq plugin](https://pypi.org/project/llm-groq/) to PyPI, facilitating easier installation via `llm install`. He shared his publishing experience and drew comparisons between Haskell and Python community practices.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Bold Claims on AI Hallucination Footprint**: Richard Socher's [tweet](https://x.com/RichardSocher/status/1760800655428796772?s=20) hinting at possible solutions for AI hallucinations sparked discussions around embedding models and validation mechanisms to improve AI's factual accuracy.

- **Introducing a New Wikipedia:** [Globe Explorer](http://explorer.globe.engineer/), a tool leveraging GPT-4 to generate customizable Wikipedia-style pages, has been launched and made viral, with a drive to top Product Hunt’s list with additional [Product Hunt details](https://www.producthunt.com/posts/globe-explorer).

- **FireFunction V1 Ignites Excitement**: The release of **FireFunction V1** by `@lqiao` promises GPT-4-level outputs with faster and more efficacious function calling, announced along with useful new structured output modes such as JSON, discussed with interest among function calling approaches, as detailed in [FireFunction's blog post](https://fireworks.ai/blog/firefunction-v1-gpt-4-level-function-calling).

- **Fine-Tuning Adventures with gpt-4-turbo**: The query on embedding techniques for improved data extraction and classification tasks using gpt-4-turbo for 1-shot learning stirred interest in effective fine-tuning practices.

- **Anki's AI Flashcard Revolution Still Pending**: The integration of GPT-4 for producing Anki flashcards revealed successes and limitations, such as verbose outputs and challenges with visual content integration, featured in an analytical [Tweet by Niccolò Zanichelli](https://x.com/nc_znc/status/1753847802487017911?s=46&t=4-kZga74dpKGeI-p2P7Zow).

- **Peering into Feather's purpose**: Feather OpenAI's icon, hinting at a writing tool, along with historical snapshots and its significance in hiring for SME data labeling and coding annotation, garnered interest, alongside advancements like the "gpt-4-ada-v2" with features enhancing data analysis capabilities, as discussed in Semafor's [article](https://www.semafor.com/article/01/27/2023/openai-has-hired-an-army-of-contractors-to-make-basic-coding-obsolete) and [Tibor Blaho's tweet](https://x.com/btibor91/status/1761726596585504939?s=46).



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Callbacks Feature in Hugging Face Trainer**: Sebastian Bodza discussed using custom callbacks with the [Hugging Face trainer](https://huggingface.co/docs/transformers/main_classes/callback), emphasizing that while currently exclusive to PyTorch, they offer "read-only" control through the `TrainerControl` interface.

- **Benchmarks Emerge in German Emotional Intelligence**: EQ-Bench now supports the German language, courtesy of updates from Calytrix, with `gpt-4-1106-preview` topping the German EQ-Bench preliminary scores, details found at the [EQ-Bench GitHub repository](https://github.com/EQ-bench/EQ-Bench). However, concerns were raised about the validity of the translated benchmarks, suggesting emotional understanding nuances might be lost, potentially skewing results due to English-centric reasoning patterns.

- **Misgivings on Probability-Based LLM Evaluations**: Bjoernp recommended an [arXiv paper](https://arxiv.org/abs/2402.13887) revealing the inherent limitations in probability-based evaluation methods for LLMs, specifically regarding multiple-choice questions and their alignment with generation-based predictions.

- **Introducing Layered Sentence Transformers**: Johann Hartmann unveiled **Matryoshka Embeddings** via a [Hugging Face blog post](https://huggingface.co/blog/matryoshka), detailing their advantages over regular embeddings, and confirmed their integration into the Sentence Transformers library, enhancing the toolkit for users.

- **Clarity on RAG Approach for German Dataset**: Johann Hartmann and Philip May deliberated the evaluation methodology for a German retrieval context understanding dataset, with May clarifying that it's crucial to assess if an LLM can identify relevant information in multiple retrieved contexts. The dataset is a work-in-progress and currently lacks public accessibility.



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **Hackathon Team Formation Heats Up**: `@reydelplatanos` and `@hiro.saxophone` teamed up for an **upcoming hackathon**, with `@hiro.saxophone` bringing experience in ML engineering, particularly in multimodal RAG. Meanwhile, `@ryznerf.` also showed interest in joining a hackathon team, emphasizing eagerness to participate.

- **Collaboration Across Disciplines**: Back end developer `@reydelplatanos` has partnered with ML engineer `@hiro.saxophone` for the hackathon, representing a fusion of backend and machine learning skills in their new team.

- **Hackathon Registration Rush**: `@silverpiranha` and `@jamthewallfacer` discussed registration for an event, with `@silverpiranha` eventually confirming successful registration and suggesting a potential team-up.

- **Drones managed by Code**: `@.yosun` introduced a hackathon project idea about controlling drones through function calls, referencing a method from the [OpenAI Cookbook](https://cookbook.openai.com/examples/fine_tuning_for_function_calling) and shared a code snippet as an illustration.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Gemma-7B Gets Conversation Manners**: `@imonenext` has integrated special tokens `<start_of_turn>` and `<end_of_turn>` into the **Gemma-7B** model to facilitate turn-taking in conversational AI. The model with these enhancements is now available for training and fine-tuning on [Hugging Face](https://huggingface.co/imone/gemma-7b-with-it-tokens).



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Seeding Insights for Stochastic Precision**: `@stereoplegic` highlighted an article on the significance of **random seeds** in deep learning, particularly for Python's PyTorch users. The article [Random Numbers in Deep Learning; Python & the PyTorch Library](https://www.linkedin.com/pulse/random-numbers-deep-learning-python-part-4-pytorch-library-jarkko-idkgf) was lauded as a "shockingly good read" for those keen to explore or fine-tune the underlying mechanics of randomness in model training.



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1210501662611869707) (1013 messages🔥🔥🔥): 

- **Mistral Large Not Worth the Performance?**: `@timotheeee1` suggested that **Mistral Large**, with a similar cost as **GPT-4 Turbo**, is not justifiable given its slightly inferior performance on benchmarks like MMLU. The cost effectiveness is questioned.
- **Megatokens Make Their Debut**: `@itsme9316` humorously coined the term "megatoken" during a discussion about **token costs**, sparking a series of light-hearted responses including "lol" from several users such as `@technotech`.
- **The Great Model Debate**: A lengthy debate ensued regarding whether **LLMs** can truly "reason." Users like `@kalomaze` and `@kaltcit` exchanged views on language models' capabilities to perform reasoning or whether what they exhibit can only be termed as *quasi-reasoning*.
- **Open Source Hopes Dashed for Mistral Large?**: Dialogue surrounding **Mistral**'s commitment to open sourcing its large models showed frustration, with users like `_dampf` lamenting the change and expressing a lack of surprise at the news.
- **Experiencing Technical Difficulties**: Users like `@kaltcit` reported issues with models like **academiccat dpo**, experiencing errors and segfaults during measurement, hinting at instability or unpredictability in some AI models.

**Links mentioned**:

- [Cody - Sourcegraph](https://sourcegraph.com/cody/chat): no description found
- [No GIF - No Nope Cat - Discover &amp; Share GIFs](https://tenor.com/view/no-nope-cat-cute-gif-4544032): Click to view the GIF
- [Supermaven](https://supermaven.com/blog/introducing-supermaven): no description found
- [Cat Cat Jumping GIF - Cat Cat Jumping Cat Excited - Discover &amp; Share GIFs](https://tenor.com/view/cat-cat-jumping-cat-excited-excited-dance-gif-19354605): Click to view the GIF
- [Mark Zuckerberg Last Breath Sans GIF - Mark Zuckerberg Last Breath Sans Last Breath - Discover &amp; Share GIFs](https://tenor.com/bMnZg.gif): Click to view the GIF
- [Neural Text Generation With Unlikelihood Training](https://openreview.net/forum?id=SJeYe0NtvH): Neural text generation is a key tool in natural language applications, but it is well known there are major problems at its core. In particular, standard likelihood training and decoding leads to...
- [OpenCodeInterpreter](https://opencodeinterpreter.github.io): no description found
- [mobiuslabsgmbh/aanaphi2-v0.1 · Hugging Face](https://huggingface.co/mobiuslabsgmbh/aanaphi2-v0.1): no description found
- [LongRoPE](https://www.youtube.com/watch?v=PFxi6SmozZ4): Like 👍. Comment 💬. Subscribe 🟥.🏘 Discord: https://discord.gg/pPAFwndTJdhttps://github.com/hu-po/docs/blob/main/2024.02.25.longrope/main.mdhttps://arxiv.o...
- [Cat Kitten GIF - Cat Kitten Speech Bubble - Discover &amp; Share GIFs](https://tenor.com/view/cat-kitten-speech-bubble-speech-discord-gif-25192162): Click to view the GIF
- [Vampire Cat Cat Eating Box GIF - Vampire Cat Cat Eating Box Cat Box - Discover &amp; Share GIFs](https://tenor.com/view/vampire-cat-cat-eating-box-cat-box-cat-fangs-gif-23385382): Click to view the GIF
- [Welcome Gemma - Google’s new open LLM](https://huggingface.co/blog/gemma): no description found
- [2021 Texas power crisis - Wikipedia](https://en.m.wikipedia.org/wiki/2021_Texas_power_crisis): no description found
- [MaxRiven - Turn It Up | Official Music Video | AI](https://youtu.be/OLEzmClaRnw?list=RDMMOLEzmClaRnw&t=16): Thank for watching !Watch in HD and enjoy it !If you liked the video please share with your friends !►Stream &amp; Download: https://fanlink.to/MXRVNturnitupThan...
- [GitHub - Dicklesworthstone/the_lighthill_debate_on_ai: A Full Transcript of the Lighthill Debate on AI from 1973, with Introductory Remarks](https://github.com/Dicklesworthstone/the_lighthill_debate_on_ai): A Full Transcript of the Lighthill Debate on AI from 1973, with Introductory Remarks - Dicklesworthstone/the_lighthill_debate_on_ai
- [Uglyspeckles - Carrot Cake Soul Shuffling Incident SFX](https://www.youtube.com/watch?v=VAP74RD30UY): From The House in Fata Morgana fandisc: Carrot Cake Jinkaku Shuffle JikenThis soundtrack is owned by Novectacle (Vegetacle)
- [GitHub - Azure/PyRIT: The Python Risk Identification Tool for generative AI (PyRIT) is an open access automation framework to empower security professionals and machine learning engineers to proactively find risks in their generative AI systems.](https://github.com/Azure/PyRIT): The Python Risk Identification Tool for generative AI (PyRIT) is an open access automation framework to empower security professionals and machine learning engineers to proactively find risks in th...
- [Announcing Microsoft’s open automation framework to red team generative AI Systems | Microsoft Security Blog](https://www.microsoft.com/en-us/security/blog/2024/02/22/announcing-microsofts-open-automation-framework-to-red-team-generative-ai-systems/): Read about Microsoft&#039;s new open automation framework, PyRIT, to empower security professionals and machine learning engineers to proactively find risks in their generative AI systems.
- [The Strange Evolution of Artificial Intelligence](https://www.youtube.com/watch?v=M6x7alUU4Xw): Center for the Future Mind presents Scott Aaronson speakingn at Mindfest 2024. Full episode will go live tomorrow Tuesday February 27 at 12PM EST.NOTE: The p...
- [no title found](https://chat.mistral.ai/chat): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1210501643955863602) (275 messages🔥🔥): 

- **Achieving Deception in AI**: `@superking__` and others discussed the challenges of programming a character to lie convincingly, as larger models like *Mixtral* perform better with explicit goals such as "survive at any cost".
- **The Tale of a Shapeshifting Android**: Despite `@superking__`'s efforts to create a character card for an android hiding its identity, the AI blew its cover until tasked with the goal "survive at any cost", which led to an improvement in secretive behavior.
- **Opus V1 Models and Technical Challenges**: Participants such as `@dreamgen` and `@kquant` navigated issues around [DreamGen Opus V1](https://huggingface.co/dreamgen/opus-v1-34b), tokenizer problems, and optimal model settings for better performance.
- **Model Issues with Verbosity and Looping**: Several users, including `@superking__` and `@dreamgen`, discussed instances where the AI would write unnecessarily long sentences or enter looping patterns, with shared experiences and potential fixes.
- **Discussion on Character Roleplay**: `@keyboardking` successfully created a character card that managed a gender disguise narrative, showcasing current AI capabilities in managing nuanced roleplay scenarios.

**Links mentioned**:

- [Kquant03/NurseButtercup-4x7B-bf16 · Hugging Face](https://huggingface.co/Kquant03/NurseButtercup-4x7B-bf16): no description found
- [maeeeeee/maid-yuzu-v8-alter-3.7bpw-exl2 · Hugging Face](https://huggingface.co/maeeeeee/maid-yuzu-v8-alter-3.7bpw-exl2): no description found
- [Chub](https://www.chub.ai/characters/kemoanon/amber-furry-1dc80cad): Find, share, modify, convert, and version control characters and other data for conversational large language models (LLMs). Previously/AKA Character Hub, CharacterHub, CharHub, CharaHub, Char Hub.
- [dreamgen/opus-v1-34b-awq · Hugging Face](https://huggingface.co/dreamgen/opus-v1-34b-awq): no description found
- [Angry Bender Mad GIF - Angry Bender Mad Angry - Discover &amp; Share GIFs](https://tenor.com/view/angry-bender-mad-angry-pissed-off-fist-gif-16261502): Click to view the GIF
- [#0SeptimusFebruary 24, 2024 3:31 PMWhat tale do you wish to hear?# - Pastebin.com](https://pastebin.com/xanUet2d): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1210647502605258793) (71 messages🔥🔥): 

- **Seeking DPO Implementation Advice**: `@cogbuji` is on the hunt for a *practical* implementation of DPO (Decision Transformer) and considers using the `DPOTrainer` from the `trl` [Hugging Face library](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py) as a reference. Various members, such as `@dirtytigerx`, engage in the discussion, offering insights and resources.
- **Fine-Tuning Versus Training Dilemmas**: `@cognitivetech` expresses concerns about the efficiency of fine-tuning full LLMs and the potential loss of information. The user considers using `gguf` for fine-tuning and also explores leveraging the [official QA-Lora implementation](https://github.com/yuhuixu1993/qa-lora) for instruct fine-tuning.
- **Dealing with DeepSpeed OOM Issues**: `@plasmator` struggles to set up DeepSpeed Zero due to out-of-memory errors, despite calculations indicating sufficient resources.
- **Storytelling LLMs and Comic Book Training Set**: `@hellblazer.666` inquires about how to train smaller models for storytelling, specifically using comic book texts as a dataset. They also share the [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit) repository as a potential tool for converting their data into a suitable format for training.
- **Training Methods and Model Selection**: In a comprehensive discussion, `@dirtytigerx` and `@hellblazer.666` discuss various training methods for LLMs, including full fine-tuning, PEFT-techniques like LoRA, as well as the usage of retrieval-augmented generation (RAG). They conclude that starting with a base model fine-tuned for storytelling might be the best approach for `@hellblazer.666`'s project.



**Links mentioned**:

- [GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets](https://github.com/e-p-armstrong/augmentoolkit): Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit
- [trl/trl/trainer/dpo_trainer.py at main · huggingface/trl](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py): Train transformer language models with reinforcement learning. - huggingface/trl

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1210651801062215770) (37 messages🔥): 

- **Challenges in Novel Model Merging**: User `@jsarnecki` inquired about merging non-homogenous models like **llama-2-13b** and **Mistral-7b** using [mergekit](https://github.com/arcee-ai/mergekit), which `@maldevide` confirmed is not possible. The discussion evolved towards exploring merging techniques that could help `@jsarnecki` reach their objective.
- **Optimizing for Use-Cases**: `@maldevide` prompted `@jsarnecki` to consider whether they were experimenting for capability discovery or targeting specific use-cases, further providing insights into successful merges on Hugging Face's models.
- **Techniques for Merging Homogenous Models**: `@alphaatlas1` mentioned **git-rebasin** as a potential option for merging models with identical size/layout and discussed limitations such as the lack of a good technique for merging different base models.
- **Advanced Merging Tactics Discussed**: The conversation shifted to various merging strategies, including linear interpolation, additive merging, and stochastic sampling as shared by `@maldevide`. The complexity of model merging techniques and their applicability to different model types was highlighted.
- **DARE Ties Merging Insights**: Diffusion models were noted to have challenges with DARE ties merging, as mentioned by `@alphaatlas1`, who also referenced a particular Hugging Face [blog post](https://huggingface.co/blog/peft_merging). However, `@maldevide` shared a successful experience, pointing to a different implementation on [GitHub](https://github.com/54rt1n/ComfyUI-DareMerge).

**Links mentioned**:

- [🤗 PEFT welcomes new merging methods](https://huggingface.co/blog/peft_merging): no description found
- [Daring Hydra - v1.2 | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/246219/daring-hydra): Daring Hydra is an attempt at a lifelike realistic model. v1.2 is actually the result of four different attempts to improve from v1.1, which is pre...
- [GitHub - 54rt1n/ComfyUI-DareMerge: ComfyUI powertools for SD1.5 and SDXL model merging](https://github.com/54rt1n/ComfyUI-DareMerge): ComfyUI powertools for SD1.5 and SDXL model merging - 54rt1n/ComfyUI-DareMerge
- [GitHub - s1dlx/meh: Merging Execution Helper](https://github.com/s1dlx/meh): Merging Execution Helper. Contribute to s1dlx/meh development by creating an account on GitHub.

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1210581496386027590) (6 messages): 

- **DotPeek Scope Clarification**: `@al_lansley` inquired about the languages that DotPeek supports, and `@spottyluck` confirmed it's limited to **just C#**.
- **Nostalgia for OllyDbg's Features**: `@spottyluck` lamented the lack of a true successor to OllyDbg, particularly its "animate into" feature, noting its limitations with 64bit which renders it nearly obsolete.
- **Eager Anticipation for AI in Decompilation**: `@mrjackspade` expressed excitement for the potential of **AI-assisted decompilation** to simplify the reverse-engineering process.
- **Frustration with Reconstructing Code**: `@mrjackspade` shared their frustration over manually reconstructing obfuscated decompiled code, hinting at the tedious nature of the process.
- **Idea for AI Training Data Sets**: `@mrjackspade` suggested an approach to creating training data sets for AI decompilation by using a large volume of open-source projects and their outputs.
  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1210529155305705532) (1198 messages🔥🔥🔥): 

<ul>
<li><strong>Mistral Large vs Next Performance:</strong> Users like `@yasserrmd` and `@chrunt` compared the capabilities of Mistral Large and Next. Large seemingly outperforms Next in certain benchmarks while Next is favored for its concise responses.</li>
<li><strong>Hardware Requirements for AI:</strong> Discussions led by `@mrdragonfox` and `@tu4m01l` highlighted the impracticality of running large AI models like Mistral Large on CPUs, suggesting the use of APIs for efficiency.</li>
<li><strong>Corporate Partnerships and Open Models:</strong> Concerns were voiced by users such as `@reguile` about the future of open models following the Microsoft partnership with Mistral. Some, like `@foxlays`, hope Mistral continues to support open model development.</li>
<li><strong>Speculations on GPT-3.5 Turbo Parameters:</strong> Debates around the actual size of GPT-3.5 Turbo were stirred by a redacted Microsoft paper, with `@i_am_dom` and `@lyrcaxis` discussing its validity and efficiency.</li>
<li><strong>Mistral's Market Positioning and Strategy:</strong> `@blacksummer99` shared insights into Mistral's efforts to differentiate from OpenAI and the conceived positioning as a European leader in the AI field.</li>
</ul>

**Links mentioned**:

- [Pricing and rate limits | Mistral AI Large Language Models](https://docs.mistral.ai/platform/pricing/): Pay-as-you-go
- [GOODY-2 | The world&#x27;s most responsible AI model](https://www.goody2.ai/): Introducing a new AI model with next-gen ethical alignment. Chat now.
- [Endpoints and benchmarks | Mistral AI Large Language Models](https://docs.mistral.ai/platform/endpoints/): We provide five different API endpoints to serve our generative models with different price/performance tradeoffs and one embedding endpoint for our embedding model.
- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [AI Playground: Run your prompts across mutiple models and scenarios](https://www.empirical.run/playground): Compare and evaluate multiple AI model completions on different prompts and model parameters
- [Cat Bruh GIF - Cat Bruh Annoyed - Discover &amp; Share GIFs](https://tenor.com/view/cat-bruh-annoyed-gif-21339312): Click to view the GIF
- [CRYNYL](https://crynyl.com/): Fall Out Boy's new album, filled with the band's real tears for maximum emotional fidelity.
- [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207): We study how to apply large language models to write grounded and organized long-form articles from scratch, with comparable breadth and depth to Wikipedia pages. This underexplored problem poses new ...
- [Legal terms and conditions](https://mistral.ai/terms/#terms-of-service-la-plateforme): Terms and conditions for using Mistral products and services.
- [Typing With Feet GIF - Typing With Feet - Discover &amp; Share GIFs](https://tenor.com/view/typing-with-feet-gif-22890703): Click to view the GIF
- [WWW.SB](https://www.re): no description found
- [Council Post: Is Bigger Better? Why The ChatGPT Vs. GPT-3 Vs. GPT-4 'Battle' Is Just A Family Chat](https://www.forbes.com/sites/forbestechcouncil/2023/02/17/is-bigger-better-why-the-chatgpt-vs-gpt-3-vs-gpt-4-battle-is-just-a-family-chat/): Alright, now we understand that ChatGPT is just a smaller and a more specific version of GPT-3, but does it mean that we will be having more such models emerging in the nearest future: MarGPT for Mark...
- [no title found](https://chat.mistral.ai/chat): no description found
- [no title found](https://chat.mistral.ai): no description found
- [no title found](https://chat.mistral.ai/>!): no description found

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1210994360426963034) (209 messages🔥🔥): 

- **GPU Essentials for Server Builds**: `@lukun` inquired about which models could run on a server with no GPU, and `@tom_lrd` and `@_._pandora_._` explained that a GPU is necessary for reasonable performance, even with smaller models. For larger models, having a GPU with at least 24 GB VRAM, such as a 3090/4090, is recommended by `@mrdragonfox`. They also provided a [detailed test gist on GitHub](https://gist.github.com/darkacorn/71658f280ea0fc0ad4b97d2a616f4ce8) to illustrate the performance at different conditions.
  
- **The Cost of Scaling Up**: Users `@dekaspace`, `@mrdragonfox`, and others discussed the specs for a server build to run language models. `@mrdragonfox` suggested 24 GB VRAM as a baseline and noted that models over 70B parameters would require a substantial investment in specialized hardware, mentioning Groq's custom ASIC deployment as a costly approach.

- **Questions About Mistral's Direction**: Several users, including `@redbrain` and `@blacksummer99`, expressed concerns over Mistral's seemingly new business-oriented direction, with closed-weight models like Mistral Small and Mistral Large, diverging from their previously open model reputation. The community speculated about upcoming releases and potential for open weight models in the future.

- **Benchmarks of Mistral Models**: `@bofenghuang` conducted tests with Mistral's models on a French version of MT-Bench, publishing results that placed Mistral Large at a notable position behind GPT-4. They shared their findings on [Hugging Face Datasets](https://huggingface.co/datasets/bofenghuang/mt-bench-french) and a [browser-based space](https://huggingface.co/spaces/bofenghuang/mt-bench-french-browser) for further inspection.

- **Hopes for Open Access to New Models**: Community sentiment as shared by `@saintvaseline`, `@_._pandora_._`, and others reflects a mix of hope for future open-access models and skepticism due to the involvement of large tech firms like Microsoft. Some members, including `@tom_lrd`, `@m._.m._.m`, and `@charlescearl_45005`, anticipated Mistral to eventually offer some lesser-quality open models while speculating on the implications of commercial partnerships.

**Links mentioned**:

- [GroqChat](https://groq.com/): no description found
- [Au Large](https://mistral.ai/news/mistral-large/.): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [Always Has Been Among Us GIF - Always Has Been Among Us Astronaut - Discover &amp; Share GIFs](https://tenor.com/view/always-has-been-among-us-astronaut-space-betrayal-gif-23836476): Click to view the GIF
- [TheBloke/Mistral-7B-Instruct-v0.1-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF): no description found
- [bofenghuang/mt-bench-french · Datasets at Hugging Face](https://huggingface.co/datasets/bofenghuang/mt-bench-french#evaluation): no description found
- [Mt Bench French Browser - a Hugging Face Space by bofenghuang](https://huggingface.co/spaces/bofenghuang/mt-bench-french-browser): no description found
- [Northern Monk Beer GIF - Northern Monk Beer Craft Beer - Discover &amp; Share GIFs](https://tenor.com/view/northern-monk-beer-craft-beer-faith-keep-the-faith-gif-17350825): Click to view the GIF
- [100k test . exllama2(testbranch) + fa  1 - 100k in 128t steps](https://gist.github.com/darkacorn/71658f280ea0fc0ad4b97d2a616f4ce8): 100k test . exllama2(testbranch) + fa  1 - 100k in 128t steps - gist:71658f280ea0fc0ad4b97d2a616f4ce8
- [[Feature Request] Dynamic temperature sampling for better coherence / creativity · Issue #3483 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/3483): Prerequisites [✅] I reviewed the Discussions, and have a new bug or useful enhancement to share. Feature Idea Typical sampling methods for large language models, such as Top P and Top K, (as well a...

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1210587410920898630) (56 messages🔥🔥): 

- **Request for Support Unanswered**: `@fangh` flagged that they sent an email last week and haven't received a response, seeking an update from `@266127174426165249`.
- **Running Mixtral on Local Machine Query**: `@c_ffeestain` inquired if they can run Mixtral 8x7B on their local machine with 32GB RAM and 8GB VRAM, and is currently using a version on [HuggingFace](https://huggingface.co/chat).
- **GPU Compatibility and Configuration Advice**: `@_._pandora_._` explained that in theory, Mixtral could be run on `@c_ffeestain`'s machine but would be extremely slow. They also offered help with finding the number of layers shared with the GPU to improve performance.
- **Exploring Model Quants and Layer Sharing**: `@c_ffeestain` noted after downloading the model that generating one token takes about 5-10 seconds. They are in the process of adjusting how many layers are shared with their GPU, but encounter issues detecting their AMD GPU.
- **Inference and Fine-tuning on a Tesla V100**: `@dazzling_maypole_30144` experienced an out-of-memory error trying to deploy Mistral-7B-Instruct on a Tesla V100. `@mrdragonfox` and `@casper_ai` suggested that the V100 might not have enough memory for this task and recommended either alternatives like T4 or A10 GPUs or running the model in AWQ format for better compatibility.

**Links mentioned**:

- [HuggingChat](https://huggingface.co/chat): Making the community's best AI chat models available to everyone.
- [TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF): no description found

  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1210547816334630952) (6 messages): 

- **Inquiry about Mistral Data Normalization**: User `@severinodadalt` from Barcelona Supercomputing Center inquired if the **Mistral data has been normalized** and the method of its implementation. The user mentioned the absence of information on the topic and is considering that no normalization has been applied.
- **No Base Model Normalization Details**: In response to `@severinodadalt`'s inquiry about data normalization, `@mrdragonfox` noted that **no base model** will provide such information.
- **Performance Variance in Different Precision Levels**: `@bdambrosio` asked if there would be a change in inference speed when running **Mistral 8x7B** locally in full fp16, compared to the current 8 bit exl2, especially with more VRAM available. The question arises from noticing differences between 6.5 and 8-bit precision levels.
- **Precision Levels Affect Performance**: In response, `@mrdragonfox` confirmed that differences are noticeable, and that **performance measurement tools like turboderp generally assess perplexity (ppl)**, suggesting that the precision level does indeed affect performance.
- **Quantization and Context Accuracy**: `@mrdragonfox` also pointed out that **quantization can slightly degrade context accuracy** when performing tasks with models like Mistral.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1210530200031600640) (185 messages🔥🔥): 

- **Fine-Tuning Data Quantities and Expectations**: `@pteromaple` inquired about the amount of data needed for fine-tuning, questioning if 4000 instances would suffice. While `@egalitaristen` suggested it depends on the specificity of the fine-tuning, highlighting that for narrow tasks, this might be enough, the discussion concluded that trial and error could be the best approach.

- **Data Format Dilemmas for Fine-Tuning**: `@pteromaple` sought advice on the correct data format for fine-tuning 'Mistral-7B-Instruct-v0.2' with Unsloth and queried about the impact of data format on training results, revealing their current use of Alpaca format. `@_._pandora_._` recommended creating a custom prompt format and warned about potential issues when fine-tuning Mistral 7B Instruct with non-English languages.

- **Mistral's Mysterious Output After Fine-Tuning**: `@mr_seeker` reported a peculiar issue where a fine-tuned model outputs `/******/` and loses coherence when prompted with non-dataset-like data. Suggestions from `@mrdragonfox` and others pointed towards the model's routing layer, with an indication that successful fine-tuning may require understanding the intricacies of the model's architecture beyond just applying techniques such as LoRA.

- **Serverless Fine-Tuning and Model Hosting Discussed**: `@stefatorus` questioned the possibility of Mistral offering serverless fine-tuning functionalities in the cloud and discussed related offerings by companies like Hugging Face and OpenAI. RunPod was also brought up as a potential cost-effective solution, but the viability for those with budget constraints was a concern.

- **LoRA Parameters Puzzle**: `@tom891` faced challenges in determining the appropriate LoRA parameters for their 200k sample dataset for Mistral 7B fine-tuning. Despite guidance from `@mrdragonfox` and others emphasizing the necessity of understanding the underlying theory and urging independent exploration over spoon-fed answers, the user continued to seek direct suggestions for effective parameter configurations.

**Links mentioned**:

[Serverless GPUs for AI Inference and Training](https://www.runpod.io/serverless-gpu): no description found

  

---


### Mistral ▷ #[announcements](https://discord.com/channels/1144547040454508606/1157222698229968896/1211713039963787365) (2 messages): 

- **Meet Mistral Large**: `@sophiamyang` announced the launch of **Mistral Large**, a new optimised model with top-tier reasoning, multilingual capabilities, native function calling, and a 32k parameter size. Boasting 81.2% accuracy on MMLU, it stands as the second-ranked model in the world and is available via [la Plateforme](https://mistral.ai/news/mistral-large/) and Azure.

- **La Plateforme Premieres le Chat Mistral**: `@sophiamyang` introduced **le Chat Mistral**, a front-end demonstration showcasing the capabilities of the Mistral models. Discover its potential at [Chat Mistral](https://chat.mistral.ai/).

**Links mentioned**:

- [Au Large](https://mistral.ai/news/mistral-large/.): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [no title found](https://chat.mistral.ai/): no description found

  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1210738634475573248) (24 messages🔥): 

- **Join @jay9265's Live Coding Stream**: @jay9265 is live streaming on [Twitch](https://www.twitch.tv/jay9265/), inviting everyone interested to join.
- **LLMs for Problem Formulation Assistance**: @egalitaristen suggests that *LLMs can be utilized to help formulate problems or tasks*, reminding @jay9265 that explaining the issue to an LLM is a way to seek assistance.
- **Lower Temperature for Structured Code**: For tasks that involve structured code like JSON, @egalitaristen advised @jay9265 to **reduce the generation temperature** to around `0.3` for less "creativity" and more accuracy.
- **WhatsApp Chrome Plugin by @yasserrmd**: @yasserrmd has developed a *Chrome plugin* that uses *Mistral API* to generate WhatsApp formatted text, with more details available on [LinkedIn](https://www.linkedin.com/posts/moyasser_whatsapp-chromeextension-mistralai-activity-7166631159303421952-8bRo/?utm_source=share&utm_medium=member_desktop).
- **AI Inference Benchmarking Analysis**: @yasserrmd shared insights from benchmarking AI inference performance across platforms like Groq using Mistral, OpenAI ChatGPT-4, and Google Gemini, providing a [LinkedIn post](https://www.linkedin.com/posts/moyasser_mixtral-chatgpt-gemini-activity-7165901459459371008-G9tI?utm_source=share&utm_medium=member_desktop) for more information.

**Links mentioned**:

[Twitch](https://www.twitch.tv/jay9265/): no description found

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1210836740512809010) (17 messages🔥): 

- **Pricing Woes in the Chatbot Landscape**: `@sublimatorniq` brought up the subject of perplexity, likely referring to pricing or complexity in chatbot services. `@mrdragonfox` suggested that the race to offer the lowest prices cannot continue indefinitely, with unit economics needing to make sense for businesses.
- **Groq's Competitive Pricing Promise**: `@shivakiran_` highlighted Groq's promise of $0.27/million, which likely refers to the cost of processing a certain number of chatbot interactions.
- **Sustainability of Low Prices Questioned**: `@mrdragonfox` pointed out that sustaining low prices for the sake of competition isn't a financially sound strategy, as it doesn't equate to profitability, especially with new players willing to absorb even more costs.
- **Critique of Initial Pricing Strategies in Tech**: `@egalitaristen` expressed concern over companies that start with low initial pricing only to later introduce "real" pricing that can be multiple times higher, warning that it may drive most of the user base to seek alternatives.
- **Pistachio Day Proclaimed on Discord**: `@privetin` shared a celebration of **Pistachio Day** with a [link to nutsforlife.com.au](https://www.nutsforlife.com.au/pistachio-day/) and fun facts about the benefits of pistachios, including their protein content and sleep-inducing melatonin.

**Links mentioned**:

- [Laughing GIF - Laughing - Discover &amp; Share GIFs](https://tenor.com/view/laughing-gif-7903323): Click to view the GIF
- [Pistachio Day - Nuts for Life | Australian Nuts for Nutrition &amp; Health](https://www.nutsforlife.com.au/pistachio-day/): Happy Pistachio Day! Each year, 26 February is dedicated to this tiny nut, which punches above its weight when it comes to taste and nutrition!

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1210532718442713109) (66 messages🔥🔥): 

- **Privacy and Hosting Clarifications Sought**: User `@exa634` enquired about whether data passing through the **Mistral API** is used for model training and about the geographical location of the hosting. It was confirmed by `@akshay_1` and `@ethux` that the data is not used for training and that servers are located in Sweden, as mentioned in Mistral's [privacy policy](https://mistral.ai/privacy-policy/).

- **Mistral7x8b Freezing Issue**: User `@m.kas` reported a bug where **Mistral7x8b** freezes when trying to generate content for the year 2024. The user `@1015814` suggested checking for an accidental end token, but `@m.kas` clarified no such token was set.

- **Expectation of Function Calling on Mistral Platform**: Users `@nioned` and `@mrdragonfox` brought up the topic of function calling on the platform, hinting that third-party providers may offer solutions and expressing optimism that **Mistral** will implement it in due time.

- **API Key Activation Delays Addressed**: User `@argumentativealgorithm` experienced a delay with their API key activation after adding billing information. `@lerela` confirmed that a short waiting period is common before the key becomes operational, which resolved the user's issue. 

- **Speech to Speech App Query**: User `@daveo1711` asked about using **Mistral Large** for a speech-to-speech application, to which `@akshay_1` replied that Mistral only supports text and suggested checking out other models for the desired functionality.

**Links mentioned**:

- [Legal terms and conditions](https://mistral.ai/privacy-policy/): Terms and conditions for using Mistral products and services.
- [client-python/examples/function_calling.py at main · mistralai/client-python](https://github.com/mistralai/client-python/blob/main/examples/function_calling.py): Python client library for Mistral AI platform. Contribute to mistralai/client-python development by creating an account on GitHub.
- [Client does not return a response · Issue #50 · mistralai/client-js](https://github.com/mistralai/client-js/issues/50): Hi there, Running the latest version of the SDK 0.1.3, but when I try to init and call the client, it does not return anything. Here is my code: const mistral = new MistralClient(env.PUBLIC_MISTRAL...
- [GitHub - Gage-Technologies/mistral-go at v1.0.0](https://github.com/Gage-Technologies/mistral-go/tree/v1.0.0): Mistral API Client in Golang. Contribute to Gage-Technologies/mistral-go development by creating an account on GitHub.

  

---


### Mistral ▷ #[le-chat](https://discord.com/channels/1144547040454508606/1211692704363323453/1211692849935163462) (69 messages🔥🔥): 

- **Mistral Chat's Popularity Issues**: Users `@lerela`, `@mr_electro84`, and others have noted that **Le Chat** is experiencing difficulties likely due to high traffic and popularity. `@mr_electro84` reported platform outages, including the API console.
  
- **Confusion Over Mistral Chat's Pricing**: `@_._pandora_._` and `@wath5` discussed whether **Le Chat** is free, with some users believing they are using paid credits while others, including `@margaret_52502`, stated it's free.

- **Enthusiasm and Suggestions for Mistral's Potential**: User `@aircactus500` proposed various enhancements for **Mistral**, from a mobile app with social networking elements to a search engine and even a 3D virtual assistant. They mentioned the idea of a language level setting for `le Chat` which sparked interest in the community.

- **Conversation About Mistral-Next**: Users `@__oo__`, `@_._pandora_._`, and `@tom_lrd` discussed a feature within **Le Chat** called **Mistral-Next**, highlighting its conciseness and simplicity compared to the large model, with hopes for its availability as an openweights model.

- **Developing Concepts for Mistral Chat Applications**: User `@aircactus500` is conceptualizing features for an app tailored to **Le Chat**, including the ability to select the AI's conversational style. They have expressed excitement over having a French AI community platform, feeling it enhances idea generation without needing to translate thoughts.
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1210502514462695444) (608 messages🔥🔥🔥): 

- **LM Studio White Window Issue**: User `@steve_evian` reported an issue where LM Studio only displays a white window upon launch. `@heyitsyorkie` suggested clearing .cache and %APPDATA% before reinstalling, which resolved the problem for `@steve_evian`.

- **LM Studio Multilingual Model Query**: User `@.deug` asked for recommendations on pre-trained multilingual LLMs that include Korean language support. `@heyitsyorkie` responded that there are few LLMs adept in translating Korean to English consistently and advised using an online translator like DeepL in combination with LM Studio.

- **Presets Reverting in LM Studio**: User `@wyrath` commented about LM Studio's UX, pointing out that when starting a "New Chat," selected presets revert to defaults, necessitating manual reselection each time. The discussion provided workarounds and the possibility of this being a bug.

- **LM Studio API and Local Hosting**: User `@muradb` inquired about running the LM Studio API on a server without a graphical environment. `@heyitsyorkie` clarified that LM Studio doesn't support headless running and didn't comment on future plans for this feature.

- **Request for Open Source and Headless LM Studio**: User `@krypt_lynx` regretfully noted LM Studio's closed source nature, also expressing that community contributions could add missing features such as headless operation. `@heyitsyorkie` confirmed that LM Studio is indeed closed source.

**Links mentioned**:

- [GroqChat](https://groq.com/): no description found
- [Phind](https://www.phind.com/blog/introducing-phind-70b): no description found
- [Seth Meyers GIF - Seth Meyers Myers - Discover &amp; Share GIFs](https://tenor.com/view/seth-meyers-myers-ehh-maybe-gif-22478163): Click to view the GIF
- [Big Code Models Leaderboard - a Hugging Face Space by bigcode](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard): no description found
- [Fine-tune a pretrained model](https://huggingface.co/docs/transformers/en/training): no description found
- [TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GGUF · Hugging Face](https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GGUF): no description found
- [Continual Learning for Large Language Models: A Survey](https://arxiv.org/abs/2402.01364): Large language models (LLMs) are not amenable to frequent re-training, due to high training costs arising from their massive scale. However, updates are necessary to endow LLMs with new skills and kee...
- [dreamgen/opus-v1.2-7b · Hugging Face](https://huggingface.co/dreamgen/opus-v1.2-7b): no description found
- [Anima/air_llm at main · lyogavin/Anima](https://github.com/lyogavin/Anima/tree/main/air_llm): 33B Chinese LLM, DPO QLORA, 100K context, AirLLM 70B inference with single 4GB GPU - lyogavin/Anima
- [The unofficial LMStudio FAQ!](https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio): Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...
- [no title found](https://ai.meta.com/resources/models-and-libraries/audiocraft/): no description found
- [MusicLM](https://google-research.github.io/seanet/musiclm/examples/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/s/TwWCNHdSv1): no description found
- [GitHub - deepseek-ai/DeepSeek-Coder: DeepSeek Coder: Let the Code Write Itself](https://github.com/deepseek-ai/DeepSeek-Coder?tab=readme-ov-file#supported-programming-languages): DeepSeek Coder: Let the Code Write Itself. Contribute to deepseek-ai/DeepSeek-Coder development by creating an account on GitHub.
- [Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning](https://arxiv.org/abs/2302.02662): Recent works successfully leveraged Large Language Models&#39; (LLM) abilities to capture abstract knowledge about world&#39;s physics to solve decision-making problems. Yet, the alignment between LLM...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1210529019863375913) (98 messages🔥🔥): 

- **Hyperparameter Evaluation Dilemma**: `@0xtotem` inquired whether hyperparameters for a RAG should be evaluated on their own dataset or if a similar dataset would suffice. It remains unresolved as a personal choice depending on the closest available data.
- **Dolphin Model Dilemma**: `@yahir9023` struggled to create a Dolphin model prompt template in **LM Studio**, sharing an external file for further explanation since Discord lacks text-sending functionality.
- **Model Memory Challenge**: `@mistershark_` discussed the difficulty of keeping multiple large language models in VRAM simultaneously and confirmed the availability and capabilities of ooba. `@goldensun3ds` asked for clarification, and `@mistershark_` explained the need for significant hardware, sharing the [GitHub link](https://github.com/oobabooga/text-generation-webui-extensions) to ooba. 
- **Translation Model Inquiry**: `@goldensun3ds` questioned the best model for Japanese to English translation, considering models like Goliath 120B and suggesting a potential **Mixtral model**. No definitive answer was given, drawing attention to the user's powerful hardware setup.
- **Mixed-Expert Models**: `@freethepublicdebt` queried if there will be future models with different mixtures of expert precisions (FP16, 8bit, and 4bit), which could promote generalization and GPU efficiency. No response was given regarding the existence or development of such models.

**Links mentioned**:

- [Knight Rider Turbo GIF - Knight Rider Turbo Boost - Discover &amp; Share GIFs](https://tenor.com/view/knight-rider-turbo-boost-tap-gif-16606813): Click to view the GIF
- [Pedro Sánchez anuncia la creación de un &quot;gran modelo de lenguaje de inteligencia artificial&quot; entrenado en español](https://www.xataka.com/robotica-e-ia/pedro-sanchez-anuncia-creacion-gran-modelo-lenguaje-inteligencia-artificial-entrenado-espanol): El Mobile World Congress ya ha comenzado y las conferencias ya empiezan a sucederse. Xiaomi y HONOR dieron el pistoletazo de salida al evento y Pedro Sánchez...
- [GitHub - rhasspy/piper: A fast, local neural text to speech system](https://github.com/rhasspy/piper): A fast, local neural text to speech system. Contribute to rhasspy/piper development by creating an account on GitHub.
- [GitHub - oobabooga/text-generation-webui-extensions](https://github.com/oobabooga/text-generation-webui-extensions): Contribute to oobabooga/text-generation-webui-extensions development by creating an account on GitHub.

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1210593638363693108) (8 messages🔥): 

- **Fire Emoji for the Fresh Update**: User `@macfly` expressed enthusiasm for the latest update, complimenting its **look and feel**.

- **Acknowledging a Needed Fix**: `@yagilb` acknowledged an unspecified issue and assured that **it will be fixed**, apologizing for any inconvenience.

- **High Praise for LM from a Seasoned User**: `@iandol`, who previously used GPT4All, praised LM for its **excellent GUI and user-friendly local server** setup.

- **Download Dilemma in China**: `@iandol` reported **difficulties downloading models** due to being in China and inquired about proxy support to facilitate downloads.

- **Seeking Dolphin 2.7 Download Support**: `@mcg9523` faced challenges downloading **Dolphin 2.7** in LM Studio and was advised by `@heyitsyorkie` to switch to **"compatibility guess"** and collapse the readme for better visibility.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1210511959355686962) (178 messages🔥🔥): 

- **Quest for CUDA Support in AMD**: User `@nink1` reminisced about AMD's growth and speculated that smart folks at AMD might be working on creating their own CUDA support, citing enterprise trends towards cost-effective solutions. The fact that ZLUDA was open-sourced hints at potential internal advancements at AMD.
  
- **Choosing the Best GPU for LLMs**: Amidst discussions on AMD vs. Nvidia GPUs, users like `@baraduk`, `@wolfspyre`, and `@heyitsyorkie` debated Radeon RX 7800 XT's suitability for running LLM models, with a general preference for Nvidia due to easier setup for AI applications, notably with ROCm on AMD requiring additional effort.

- **To NVLink or Not to NVLink**: Participants like `@slothi_jan`, `@dave2266_72415`, and `@nink1` explored the pros and cons of NVLink for multi-GPU setups. While NVLink could theoretically boost performance compared to using standard PCIe slots, practical considerations like cost and compatibility are significant factors.

- **Mac vs. Custom PC for Running LLMs**: User `@slothi_jan` sparked discussions on whether to purchase a Mac Studio or a custom PC with multiple RTX 3090 GPUs for AI model use. Opinions varied, but factors like speed, cost, ease of use, and future-proofing were key considerations with valuable input from users like `@heyitsyorkie`, `@rugg0064`, and `@nink1`, who noted the surprisingly good performance of Apple's M3 Max.

- **Troubleshooting PC Shutdowns During LLM Use**: `@666siegfried666` sought assistance with their PC (featuring a 5800X3D CPU and 7900 XTX GPU) shutting down during use of LM Studio. `@heyitsyorkie` suggested testing with other compute-intensive tasks to identify whether the issue is with LM Studio or the PC hardware itself.

**Links mentioned**:

- [no title found](https://www.amazon.ca/Pro-WRX80E-SAGE-ThreadripperTM-Extended-ATX-Workstation/dp/B0BZT9NF57/ref=sr_1): no description found
- [no title found](https://www.amazon.ca/Pro-WRX80E-SAGE-ThreadripperTM-Extended-ATX-Workstation/dp/B0BZT9NF57/ref=sr_1_3?crid=9K4TV6E0MG76&dib=eyJ2IjoiMSJ9.-IoMlBofHFrBEIQHfWDvBPT0_VBq2-8Wn19yDkkxoRFMsBwd3D-gtI6nkIt95ykpK62aExUjHKkhTW5mLMjGqvYIQMlWdPbFFivIDAJBmIuVtl_EuNvnuJy1Vq2ocMLv9gwjwLfDi-a7AgMJp2qfowLr2vEy2i2Rheq47OO3Ky_0UfCLrVMk54fyXfDETn6YvdV_DGCnHdfYIwLjX9cabDgXGLjYnWpuzclAgMtx8juvdfi47HxfDruBLJfhB-IRu1QYGHEu86lzplr8bhYWnG3_ASWVnmRtaMwy-DvPo68.o_mxP_7nOy1NiZAhN23M0aYK4z9r8GKaK5BKri2_VWo&dib_tag=se&keywords=Asus+16x+Pcie+motherboard&qid=1708889966&sprefix=asus+16x+pcie+motherboard%2Caps%2C96&sr=8-3): no description found
- [High Five GIF - High Five Minion - Discover &amp; Share GIFs](https://tenor.com/view/high-five-minion-claptrap-bl1-gif-5280331): Click to view the GIF
- [README.md · TheBloke/Llama-2-70B-GGUF at main](https://huggingface.co/TheBloke/Llama-2-70B-GGUF/blame/main/README.md): no description found
- [Releases · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/releases): LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
- [OpenCL - The Open Standard for Parallel Programming of Heterogeneous Systems](https://www.khronos.org/opencl/): no description found
- [Knowledge Doubling Every 12 Months, Soon to be Every 12 Hours - Industry Tap](https://www.industrytap.com/knowledge-doubling-every-12-months-soon-to-be-every-12-hours/3950): Knowledge Doubling Every 12 Months, Soon to be Every 12 Hours - Industry Tap

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1210514486533029908) (27 messages🔥): 

- **Celebrating "IQ" Models Working**: `@drawless111` enthusiastically confirmed that **IQ1, IQ2,** and **IQ3** models are working in LM Studio, praising **Yags and the team**. They highlighted **IQ1's** impressive specs with *14.5 GB VRAM and 70 Billion model at 11.95 t/s*.
  
- **Searching for "IQ" Formats Revealed**: `@drawless111` provided a step-by-step guide on finding **"IQ" formats** on HGF such as "gguf imat" or "gguf imatrix", and noted to avoid compressions fixed with random text for higher quality.
  
- **LLM Local File System Access Query**: `@wolfspyre` inquired about local file system access for running models, wondering if a directory like `/tmp` is accessible, but later `@fabguy` clarified that **LLMs** don’t have such capabilities, and LM Studio does not support executing commands from LLMs.
  
- **No Model Tokenization Speed Stats API Yet**: `@wolfspyre` asked if there’s an API to get model tokenization speed stats to which `@yagilb` shortly replied with a "*Not yet*".
  
- **Llama 1.6 Update Rolled Out**: Users `@n8programs` and `@heyitsyorkie` discussed and celebrated the update of **llama.cpp** to version **1.6** in LM Studio, describing it as *EPIC*.
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1210596239087894578) (9 messages🔥): 

- **AutoGen Anomaly Squashed**: User `@thebest6337` initially reported a mysterious error with AutoGen but resolved the issue by ***"uninstall[ing] and reinstall[ing] every autogen python package"***.
- **Good Samaritan Reminder**: `@heyitsyorkie` encouraged `@thebest6337` to share the solution to their problem with AutoGen to assist others, leading to the discovery of the fix.
- **When in Doubt, Reboot!**: In response to `@thebest6337`'s fix, `@heyitsyorkie` humorously posted a Tenor GIF link, implying that the classic "turn it off and on again" method is a universally applicable solution: [Tenor GIF](https://tenor.com/view/it-problem-phone-call-have-you-tried-turning-it-off-and-on-again-gif-17823069).
- **Slow Responding Local Models**: User `@gb24.` queried about the slow response time (approx. five minutes) from a local model, implying it is an unusually long delay as the task was not code intensive.

**Links mentioned**:

[It Problem Phone Call GIF - It Problem Phone Call Have You Tried Turning It Off And On Again - Discover &amp; Share GIFs](https://tenor.com/view/it-problem-phone-call-have-you-tried-turning-it-off-and-on-again-gif-17823069): Click to view the GIF

  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/) (1 messages): 

bigsuh.eth: Hello, can I use LM Studio and use RAG in langchain?
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1211451861417070722) (7 messages): 

- **Connection Issues for nxonxi**: User `@nxonxi` encountered a `httpcore.Connect Error: [Errno 111] Connection refused` when attempting to run `open-interpreter` with the `--local` command after installing LM Studio.
- **Syntax Error Strikes**: The same user received an error stating `{'error': "'prompt' field is required"}`, which turned out to be due to a syntax error in their request payload.
- **Simple Python Request to the Rescue**: `@nxonxi` confirmed that while LM Studio is not working from OpenAI (OI), it is operational via a simple Python request.
- **Endpoint URL Troubleshooting**: `@1sbefore` suggested checking the endpoint URL, mentioning that for TGWUI it is `http://0.0.0.0:5000/v1` and advised `@nxonxi` to possibly remove `/completions` or `/v1/completions` from the URL being used in requests as a possible solution.
  

---



### Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1210641574153822249) (1 messages): 

- **Perplexity Partners with ElevenLabs**: `@ok.alex` announced the launch of the **Discover Daily podcast**, a collaboration with [ElevenLabs](https://elevenlabs.io), pioneers in Voice AI technology. Find the podcast on your favorite platforms for a daily dive into tech, science, and culture, with episodes sourced from Perplexity's [Discover feed](https://www.perplexity.ai/discover).
- **Discover Daily Podcast Elevates Your Day**: Listening to the latest episodes of **Discover Daily** is recommended during your daily commute or in that spare moment of curiosity. The episodes are available at [podcast.perplexity.ai](https://podcast.perplexity.ai) and are enhanced by ElevenLabs' voice technology.



**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/elevenlabs): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discover Daily by Perplexity](https://podcast.perplexity.ai): We want to bring the world's stories to your ears, offering a daily blend of tech, science, and culture. Curated from our Discover feed, each episode is designed to enrich your day with insights and c...

  

---


### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1210516967602126868) (348 messages🔥🔥): 

- **Perplexity AI Unveils New Models 'Sonar':** The Perplexity AI Discord community discussed the recent introduction of **Sonar** models (`sonar-small-chat` and `sonar-medium-chat`) and their search-enhanced versions. These models claim improvements in cost-efficiency, speed, and performance. Users speculate, based on test interactions, that **Sonar Medium** may possess a knowledge cutoff around December 2023 ([source](https://docs.perplexity.ai/changelog/api-updates-february-2024)).
  
- **Goodbye Gemini**: The community briefly mourned the removal of **Gemini** from the list of available models on Perplexity, with some users clamoring for its return or the potential introduction of **Gemini Ultra**.

- **Perplexity AI and Imagery**: It was clarified that **Perplexity Pro** does have the capability to generate images, albeit with some operational issues under scrutiny. Users are directed to online resources and Reddit for assistance ([Reddit post](https://www.reddit.com/r/perplexity_ai/comments/18eqmig/how_to_generate_images_with_perplexity_ai/)).

- **Mobile-specific Responses from AI Models**: There was a discussion about whether AI chat models respond differently on mobile devices compared to PCs, with some users noticing concise answers from models like **Gemini** when accessed through the app ([system prompt](https://www.perplexity.ai/search/A-box-is-yl_bkD0DS5GQ9qeA.Kp2mw#2)).

- **Alleged Deals and Discrepancies**: In the mix of conversations, various unrelated topics were raised such as a supposed **6-month free trial** of Perplexity Pro tied to a card service, which was confirmed to be legitimate by a moderator, and an inquiry about whether **Mistral** is partnering with Microsoft following a historical collaboration with Google.

**Links mentioned**:

- [Phind](https://www.phind.com/blog/introducing-phind-70b): no description found
- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [API Updates February 2024](https://docs.perplexity.ai/changelog/api-updates-february-2024): Announcing Our Newest ModelWe are excited to announce the launch of our latest Perplexity models: sonar-small-chat and sonar-medium-chat, along with their search-enhanced versions, sonar-small-online ...
- [Microsoft partners with Mistral in second AI deal beyond OpenAI](https://www.theverge.com/2024/2/26/24083510/microsoft-mistral-partnership-deal-azure-ai): Microsoft makes another AI investment.
- [no title found](https://api.perplexity.ai'): no description found
- [PerplexityBot](https://docs.perplexity.ai/docs/perplexitybot): We strive to improve our service every day. To provide the best search experience, we need to collect data. We use web crawlers to gather information from the internet and index it for our search engi...
- [Perplexity Blog](https://blog.perplexity.ai/): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [no title found](https://ramp.com/rewards/perplexity): Perplexity is the leading real-time AI answer engine. Perplexity Pro supercharges research with unlimited file uploads, guided AI search with Copilot, and dedicated support.
- [no title found](https://ai.meta.com/research/publications/sonar-sentence-level-multimodal-and-language-agnostic-representations/): no description found
- [Adiós Google | Hola Perplexity](https://youtu.be/NjQ8LeYfxRY?si=m32SzgylMsQPIBuQ): No te vas a creer lo que hace este buscador gracias a la Inteligencia Artificial. Aún no sabemos que sería de Perplexity de no ser por Jeff Bezos, Nvidia y D...
- [‎Discover Daily by Perplexity on Apple Podcasts](https://podcasts.apple.com/us/podcast/discover-daily-by-perplexity/id1732181427): ‎News · 2024
- [Images &amp; media](https://blog.perplexity.ai/faq/images-media): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Billing and Subscription](https://blog.perplexity.ai/faq/billing-and-subscription): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/18eqmig/how_to_generate_images_with_perplexity_ai/): no description found
- [no title found](https://chat.mistral.ai/): no description found

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1210695128214147172) (23 messages🔥): 

- **Exploring Topics on Perplexity AI**: Users in the "sharing" channel are sharing various links to Perplexity AI topics ranging from reviews of the **Xiaomi 14 series** (`@icelavaman`), discussions about **PerplexityAI and ElevenLabs** (`@icelavaman`), to analyses of "why put Mistral" in AI models (`@mydpi`).
- **Curiosity About Global Events**: Some users are looking into timely events and items like the first US moon mission in years (`@sanjaymenon`), **Lenovo's transparent laptop concept** (`@vipul7031`), and **Starshield in Taiwan** (`@cy_alex`).
- **Model Comparisons and Technical Queries**: Tech enthusiasts are delving into comparisons such as iPhone models (`@ming9993`) and questions about tech strategies like the use of eigenlayer nodes (`@novice9708`).
- **Personal Assistants and Learning with Perplexity AI**: Individuals are leveraging Perplexity AI for personal discovery and study, with searches about **American athletes** (`@commuting5048`) and making personal collections such as "Make your own" (`@_yoojungin`).
- **Miscellaneous Interests Spotlight**: Interests in the channel are diverse, with users such as `@chob_hee` seeking mathematical calculations, `@mistercare` looking into recommended tools (in German), and `@veryoriginalname123` expressing a personal statement (*"I am a..."*).
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1210566406664617984) (339 messages🔥🔥): 

- **Sonar Models Debut**: Perplexity AI introduced new models: `sonar-small-chat` and `sonar-medium-chat`, as well as their online counterparts with enhanced search capabilities. Users, including `@thedigitalcat` and `@digital_despot`, expressed a preference for the `pplx-70b-online` model, which appears to offer more coherent answers.
- **Comparing Sonar and pplx-70b**: `@jaicraft` suggested that *sonar-medium* should outperform *pplx-70b* but others, including `@sergevar` and `@thedigitalcat`, reported receiving incoherent or "gibberish" responses from the sonar models.
- **Prefer pplx-70b Over Sonar Medium**: Users like `@thedigitalcat` requested the `pplx-70b-online` model not be phased out due to its superior performance. `@ok.alex` from Perplexity AI acknowledged issues with `sonar-medium-online` and mentioned that a fix was being worked on.
- **API Usage Improvements Discussed**: `@ericosk` sought a programmatic way to fetch model details, expressing a use case for populating a UI with model choices. Additionally, users like `@thedigitalcat` and `@brknclock1215` discussed the impact of using or omitting system prompts in API calls.
- **Gibberish Output from Sonar Models**: `@brknclock1215` noted that limiting output length could mitigate the gibberish responses from sonar models, but `@thedigitalcat` shared that pplx models were unaffected by lengthy output. `@thedigitalcat` provided a screenshot demonstrating a non-human-readable response from `sonar-medium-online`.

**Links mentioned**:

- [no title found](https://api.perplexity.ai")): no description found
- [API Updates February 2024](https://docs.perplexity.ai/changelog/api-updates-february-2024): Announcing Our Newest ModelWe are excited to announce the launch of our latest Perplexity models: sonar-small-chat and sonar-medium-chat, along with their search-enhanced versions, sonar-small-online ...
- [Mixtral of experts](https://mistral.ai/news/mixtral-of-experts/): A high quality Sparse Mixture-of-Experts.
- [More than an OpenAI Wrapper: Perplexity Pivots to Open Source](https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/): Perplexity CEO Aravind Srinivas is a big Larry Page fan. However, he thinks he&#039;s found a way to compete not only with Google search, but with OpenAI&#039;s GPT too.
- [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions): Generates a model&#x27;s response for the given chat conversation.
- [pplx-api](https://docs.perplexity.ai): no description found
- [How to access the usage of a stream when using OpenAI sdk?](https://docs.perplexity.ai/discuss/65da5519af6a9a00293e2f59): Hi, I&#x27;m currently having a hard time accessing the usage from the stream in JS. It&#x27;s fine in Python as we can just iterate through the response but can&#x27;t find a way in JS. I&#x27;m also...
- [pplx-api form](https://perplexity.typeform.com/to/j50rnNiB): Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.
- [hask/main/background.js at 34dad93639419617595915122b0099b7023a3dae · bm777/hask](https://github.com/bm777/hask/blob/34dad93639419617595915122b0099b7023a3dae/main/background.js#L87): Hask anything powered by Online LM. Contribute to bm777/hask development by creating an account on GitHub.

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1210524635196694558) (183 messages🔥🔥): 

- **VPN May Interfere with OpenAI Services**: User `@strang999` experienced the "Something went wrong" error. `@satanhashtag` suggested VPN services could interfere, even when not actively used, and recommended disabling web protection in VPN settings.

- **Image-to-Video Generation Tools Sought**: `@sparkette` asked for a browser-based image-to-video generator that doesn't use a credit system. `@lugui` proposed snappyvideo.ai, although it didn't fit the unmetered usage criteria.

- **Anticipation for Sora's Capabilities**: Users `@rreitsma` and `@madame_architect` discussed the potential of OpenAI's Sora for creating informative TV shows or personalized language courses, highlighting its advanced simulation features.

- **Mixed Experiences with Copilot**: `@pruo` and `@madame_architect` shared experiences with Copilot, an in-app chatbot by Microsoft, indicating that while `@pruo` found it valuable, `@madame_architect` felt quality was lacking compared to previous AI iterations.

- **Gemini Users Face Social Pressure**: `@pruo` expressed frustration at being shamed for using Google's Gemini AI system, seeking to use it without judgment. `@tariqali` responded by emphasizing the problem with the AI, not the users, and the merits of not relying on a single AI system.

**Links mentioned**:

- [GroqChat](https://groq.com/): no description found
- [Mistral AI releases new model to rival GPT-4 and its own chat assistant | TechCrunch](https://techcrunch.com/2024/02/26/mistral-ai-releases-new-model-to-rival-gpt-4-and-its-own-chat-assistant/): Mistral AI is launching a new flagship large language model called Mistral Large. It is designed to rival other top-tier models like GPT-4.
- [Video generation models as world simulators](https://openai.com/research/video-generation-models-as-world-simulators): We explore large-scale training of generative models on video data. Specifically, we train text-conditional diffusion models jointly on videos and images of variable durations, resolutions and aspect ...
- [Place Anything into Any Video](https://arxiv.org/html/2402.14316v1): no description found
- [HuggingChat](https://huggingface.co/chat/): Making the community's best AI chat models available to everyone.
- [Gorilla](https://gorilla.cs.berkeley.edu/): no description found
- [Introduction to Gorilla LLM](https://gorilla.cs.berkeley.edu/blogs/4_open_functions.html): no description found
- [GitHub - ShishirPatil/gorilla: Gorilla: An API store for LLMs](https://github.com/ShishirPatil/gorilla): Gorilla: An API store for LLMs. Contribute to ShishirPatil/gorilla development by creating an account on GitHub.
- [How Do Agents Make Decisions?](https://www.jasss.org/17/4/13.html): no description found

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1210537626499678228) (103 messages🔥🔥): 

- **ChatGPT's Context Limits Stir Frustration**: `@orbart` expressed disappointment with a perceived reduction in ChatGPT's ability to remember long texts for narrative work, suspecting a "nerf" in capabilities. `@blckreaper` corroborated the feeling, suggesting a reduction in tokens processed from files, from 15K to approximately 8K.

- **Captcha Conundrum Throws Users In Loops**: `@little.toadstool` and `@necrosystv` reported undergoing repeated and frustrating 20-stage captcha tests within ChatGPT, disrupting the user experience and prompting questions about the service's current issues.

- **The Search for Math and PDF Solutions**: Users like `@candonlyc` and `@yami1010` discussed the lack of a MathPix ChatGPT Plugin and the challenges associated with OCR capabilities for mathematical content, leading to suggestions of using external resources or APIs for enhancement.

- **Protecting Custom Prompts a Slippery Slope**: Users `@.dunamis.` and `@kyleschullerdev_51255` exchanged ideas about safeguarding prompts, with the consensus being that complete protection isn't feasible and a layered web application approach might offer better security.

- **Curiosity About GPT-4's Fine-Tuning and Discoverability**: `@kxlja` inquired whether AI models on the Discover page are selected by hand or through other criteria, and `@liangdev` asked about accessing the GPT-4 model for fine-tuning, probing into the availability of such an option.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1210519410972823632) (209 messages🔥🔥): 

- **Assistant vs Custom GPT Nuances**: `@brunoalec` noted inconsistencies when using Custom GPTs from the OpenAI store as GPT Assistants in the API. `@rendo1` clarified that Assistants can generate tables in 'code block' format and markdown formatting is not supported in the Assistants UI, unlike ChatGPT UI which converts markdown into visual elements.
- **Improving Search Functionality**: `@kevinnoodles` faced issues with ChatGPT searches returning no valid results or denying access. There was no solution proposed in the chat.
- **Text Classification Task Query**: `@crifat` questioned whether to use fine-tuning or Assistant for a text classification problem. `@eskcanta` suggested first trying with the base model to check the error rate.
- **Prompt Optimization for Code Tasks**: `@tawsif2781` inquired about the best way to prompt for converting JavaScript to TypeScript in a project. There was no specific guide provided in the chat.
- **ChatGPT Responsiveness Issues**: `@ianhoughton44` reported ongoing issues with ChatGPT responses being unhelpful or non-compliant for more than a week but didn't receive any troubleshooting advice in the discussion.

**Links mentioned**:

- [Usage policies](https://openai.com/policies/usage-policies): no description found
- [Terms of use](https://openai.com/policies/terms-of-use): no description found
- [Enterprise privacy](https://openai.com/enterprise-privacy): no description found

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1210519410972823632) (209 messages🔥🔥): 

- **In Search for GPT UI Optimization**: User `@joemama8222` sought advice on improving UI design for HTML code but didn't mention specifics or share a solution.
- **Prompt Truncation Woes Rampant**: `@jimmysapp` expressed continual issues with prompt truncation and missing responses to custom instructions in ChatGPT, with the problem persisting across both browser and phone app. User `@madame_architect` recommended clearing cookies and rebooting, while others like `@eskcanta` speculated on possible confusion within the AI due to content policy.
- **AI Function Inquiry Met with Developer Expertise**: User `@agi_dude` asked about function calling with a specific setup for programming documentation queries; guidance was provided by `@eskcanta` and `@madame_architect`, with the latter redirecting to API documentation suggesting Assistant API usage.
- **Debate Over Image Prompt Reproduction Abilities**: `@bombasticfard` inquired about replicating specific images with AI prompts, `@bambooshoots` suggested a strategy using Wright's Pixel Painter Pro CustomGPT, and `@cqoker` shared success using the term "anime 2d model/format" to produce desired image styles.
- **Confusion on AI Abilities Between Custom and Assistant Models**: User `@brunoalec` noted differences between Custom GPTs and Assistant GPTs regarding table formatting, DALL-E usage, and markdown functionality, with `@rendo1` explaining that Assistants cannot natively format markdown or generate images directly without specific API configurations.
- **Credit Report Data Handling Addressed**: `@razorbackx9x` inquired about an AI that can sort credit report data into Excel. `@eskcanta` strongly cautioned against uploading sensitive PII data, reinforced by `@s_p_e_c` asking for official clarity on privacy policies, and `@madame_architect` advocating for cleaning data before use.

**Links mentioned**:

- [Terms of use](https://openai.com/policies/terms-of-use): no description found
- [Enterprise privacy](https://openai.com/enterprise-privacy): no description found
- [Usage policies](https://openai.com/policies/usage-policies): no description found

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1210517643220619274) (624 messages🔥🔥🔥): 

- **Transformer-based Video Model Discussion**: `@yoavhacohen` mentioned a new project called Snap Video that addresses challenges in video generation by employing a framework that accounts for spatially and temporally redundant pixels and a new transformer-based architecture. He shared the project [link](https://snap-research.github.io/snapvideo/#title-footer) and the related [research paper](https://arxiv.org/pdf/2402.14797.pdf).

- **Concerns About Generative Video Models**: User `@qwerty_qwer` expressed skepticism about the meaningfulness of generative video models unless they are released by large organizations, suggesting that research students lack the necessary compute resources for impactful releases.

- **Seeking Open Source Projects for Contribution**: `@k_ek_w` introduced themselves as a data scientist with 1 year of experience looking for open source AI and ML projects to contribute to.

- **Image Captioner Demonstration**: `@yoavhacohen` provided examples comparing captions from their team's image captioner against LLaVA and Google Captioner for different images, highlighting the differing levels of detail in caption descriptions.

- **LoRA Land Release**: User `@helium__` announced the release of [LoRA Land](https://predibase.com/lora-land), a collection of Mistral-7b models fine-tuned on various tasks. They noted the models' superior performance and cost efficiency, and shared a [webinar link](https://my.demio.com/ref/VlvFU73TUTUuKMjO) for more information.

**Links mentioned**:

- [Tweet from Stella Biderman (@BlancheMinerva)](https://x.com/BlancheMinerva/status/1761174487398072651?s=20): @maxhbain @Shutterstock Hi, I can&#39;t message you (it&#39;s set to premium only) but I would love to talk about this and especially continuing to make the data available for researchers if possible....
- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [SPIN Diffusion Demo V1 - a Hugging Face Space by UCLA-AGI](https://huggingface.co/spaces/UCLA-AGI/SPIN-Diffusion-demo-v1): no description found
- [Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models](https://tuning-encoder.github.io/): no description found
- [SDXL Lightning - by fal.ai](https://fastsdxl.ai/): Lightning fast SDXL API demo by fal.ai
- [Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models](https://arxiv.org/abs/2306.00637): We introduce Würstchen, a novel architecture for text-to-image synthesis that combines competitive performance with unprecedented cost-effectiveness for large-scale text-to-image diffusion models. A k...
- [Shaheer Rehman GIF - Shaheer Rehman - Discover &amp; Share GIFs](https://tenor.com/bCp9a.gif): Click to view the GIF
- [Starship Troopers GIF - Starship Troopers - Discover &amp; Share GIFs](https://tenor.com/bn7zR.gif): Click to view the GIF
- [Safety Review for LAION 5B | LAION](https://laion.ai/notes/laion-maintanence/): &lt;p&gt;There have been reports in the press about the results of a research project at Stanford University, according to which the LAION training set 5B contains...
- [LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4 - Predibase](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4): LoRA Land is a collection of 25+ fine-tuned Mistral-7b models that outperform GPT-4 in task-specific applications. This collection of fine-tuned OSS models offers a blueprint for teams seeking to effi...
- [Tweet from Allen T (@Mr_AllenT)](https://fxtwitter.com/Mr_AllenT/status/1761406217186849232?s=20): China will air it’s first AI anime on it’s CCTV station soon  I wonder how long until AI series become more common worldwide?  
- [Shutterstock Expands Partnership with OpenAI, Signs New Six-Year Agreement to Provide High-Quality Training Data | Shutterstock, Inc.](https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year): The Investor Relations website contains information about Shutterstock, Inc.&#039;s business for stockholders, potential investors, and financial analysts.
- [TTS Arena - a Hugging Face Space by TTS-AGI](https://huggingface.co/spaces/TTS-AGI/TTS-Arena): no description found
- [LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard): no description found
- [Vision Arena (Testing VLMs side-by-side) - a Hugging Face Space by WildVision](https://huggingface.co/spaces/WildVision/vision-arena): no description found
- [Garfield Diffusion V1 - v1.0 | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/29444/garfield-diffusion-v1>): My first big model. Trained on 16240 comic strips tagged by lasagna.cz . Meaning there are character tags and many others (see all on the website)....
- [GitHub - Breakthrough/PySceneDetect: :movie_camera: Python and OpenCV-based scene cut/transition detection program &amp; library.](https://github.com/Breakthrough/PySceneDetect): :movie_camera: Python and OpenCV-based scene cut/transition detection program &amp; library. - Breakthrough/PySceneDetect
- [shinonomelab/cleanvid-15m_map · Datasets at Hugging Face](https://huggingface.co/datasets/shinonomelab/cleanvid-15m_map): no description found
- [WebVid 大型短视频数据集 / 数据集 / 超神经](https://hyper.ai/datasets/17289): no description found
- [Add new merging methods by pacman100 · Pull Request #1364 · huggingface/peft](https://github.com/huggingface/peft/pull/1364): What does this PR do?  Add new model merging methods for LoRA based on the papers TIES-MERGING: Resolving Interference When Merging Models and Language Models are Super Mario: Absorbing Abilities f...
- [Snap Video](https://snap-research.github.io/snapvideo/#title-footer): no description found

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1210525488771112960) (67 messages🔥🔥): 

- **CLIP Filters vs HQ Classifiers Debate**: `@top_walk_town` pointed out the importance of the DFN paper for showing that CLIP filtering is suboptimal compared to using high-quality image-text pair classifiers.

- **BFloat16 Gradient Discussion**: `@yoavhacohen` affirmed the use of **autocasting on TPUs** with bfloat16, while `@top_walk_town` and `@chad_in_the_house` discussed Pytorch's **autocast behavior**, where the backward pass defaults to fp32.

- **Model Parameter Discrepancies**: `@thejonasbrothers` noted confusion about Google's release of the **gemma as a 7b model**, which is actually a 9b model when counting parameters.

- **Gradient Precision Trade-offs**: `@chad_in_the_house` updated that training with **bf16 gradients** is faster but yields worse results compared to fp32 gradients.

- **Research Papers and Methods Sharing**: Multiple research papers and AI-related methods were shared by users `@said2000`, `@thejonasbrothers`, and others, touching on **state space architecture**, optimization of **Transformer models**, and the detection of AI-generated text's **"radioactivity"**. Additionally, `@vrus0188` shared a YouTube video discussing the potential for AI to make **LLMs significantly cheaper**.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=39491957>): no description found
- [Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping](https://arxiv.org/abs/2402.14083): While Transformers have enabled tremendous progress in various application settings, such architectures still lag behind traditional symbolic planners for solving complex decision making tasks. In thi...
- [Generative Models: What do they know?](https://intrinsic-lora.github.io/): no description found
- [Mamba-ND: Selective State Space Modeling for Multi-Dimensional Data](https://arxiv.org/abs/2402.05892): In recent years, Transformers have become the de-facto architecture for sequence modeling on text and a variety of multi-dimensional data, such as images and video. However, the use of self-attention ...
- [Watermarking Makes Language Models Radioactive](https://arxiv.org/abs/2402.14904): This paper investigates the radioactivity of LLM-generated texts, i.e. whether it is possible to detect that such input was used as training data. Conventional methods like membership inference can ca...
- [collabora/whisperspeech · Hugging Face](https://huggingface.co/collabora/whisperspeech): no description found
- [Fireship](https://www.youtube.com/@Fireship/videos): High-intensity ⚡ code tutorials and tech news to help you ship your app faster. New videos every week covering the topics every programmer should know.   The original home of #100SecondsOfCode #TheCod...
- [no title found](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/): no description found
- [LoRA Land: Fine-Tuned Open-Source LLMs that Outperform GPT-4 - Predibase](https://predibase.com/blog/lora-land-fine-tuned-open-source-llms-that-outperform-gpt-4): LoRA Land is a collection of 25+ fine-tuned Mistral-7b models that outperform GPT-4 in task-specific applications. This collection of fine-tuned OSS models offers a blueprint for teams seeking to effi...
- [no title found](https://nicholas.carlini.com/writing/2024/evaluation_examples/index.html): no description found
- [yet-another-applied-llm-benchmark/tests at main · carlini/yet-another-applied-llm-benchmark](https://github.com/carlini/yet-another-applied-llm-benchmark/tree/main/tests): A benchmark to evaluate language models on questions I&#39;ve previously asked them to solve. - carlini/yet-another-applied-llm-benchmark
- [Mamba Might Just Make LLMs 1000x Cheaper...](https://www.youtube.com/watch?v=SbmETE7Ey20): Check out HubSpot&#39;s ChatGPT at work bundle! https://clickhubspot.com/twcWould mamba bring a revolution to LLMs and challenge the status quo? Or would it just...
- [The AI &#39;Genie&#39; is Out + Humanoid Robotics Step Closer](https://www.youtube.com/watch?v=gGKsfXkSXv8): First text-to-speech, text-to-video and text-to-action, and now text-to-interaction? Let’s take a look at the new Genie paper from Google DeepMind, and set i...
- [Scalable Diffusion Models with State Space Backbone](https://arxiv.org/abs/2402.05608): This paper presents a new exploration into a category of diffusion models built upon state space architecture. We endeavor to train diffusion models for image data, wherein the traditional U-Net backb...
- [SDXL-Lightning: Progressive Adversarial Diffusion Distillation](https://arxiv.org/abs/2402.13929v1): We propose a diffusion distillation method that achieves new state-of-the-art in one-step/few-step 1024px text-to-image generation based on SDXL. Our method combines progressive and adversarial distil...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ay6ey7/brazilian_modders_successfully_double_rtx_2080/?$deep_link=true&correlation_id=cc141632-1a81-440f-a901-d71b4a415414&post_fullname=t3_1ay6ey7&post_index=1&ref=email_digest&ref_campaign=email_digest&ref_source=email&utm_content=post_title): no description found

  

---


### LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1211135283853131776) (2 messages): 

- **Beware of Potential Scams**: User `@josephsweeney11` offered to help 10 people make $40k in 72 hours with a 10% commission via Telegram @auto_trade_admin. This kind of message could be a **scam** and users should exercise **caution**.
- **Experimenting with Transformer Learning Capabilities**: `@phryq.` is curious if anyone has explored the learning capabilities of transformers through experimental training, such as understanding and applying size relationships between made-up objects to generate images. They provided specific examples to question if the model can deduce that a "krog" should be rendered four times as large as a "mmmmmchakaboooboolight."
  

---


### LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/) (1 messages): 

said2000: https://arxiv.org/abs/2402.05892
  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1210505687944142898) (182 messages🔥🔥): 

```html
<ul>
  <li><strong>AI Hardware Endeavors and Speculations</strong>: Users discussed the potential of developing proprietary TPUs and the availability of particular nanometer manufacturing processes, highlighting how this democratization could grant freedom akin to the car industry. The conversation referenced comparisons to the RAM industry's price practices, indicating skepticism about tech promises from companies like Samsung.</li>
  <li><strong>Ongoing AI Debates</strong>: Community members voiced opinions on the impact of AI and capitalism, with some debating whether open-source efforts could rival giants like Intel or Nvidia. Discussions reflected concerns about the loss of jobs and wealth inequality tied to technology advancements, balanced by the practicalities of AI product development to secure individual financial well-being.</li>
  <li><strong>Inquiries and Assistance on Model Utilization</strong>: Users sought help for a range of topics, including the use of specific models on certain GPUs and integrations, limitations related to model sizes and memory constraints, the management of datasets, and finding resources for projects. The community contributed with suggestions such as using llama.cpp for model parallelization and employing CPU offloading with accelerate for large models.</li>
  <li><strong>Exploring Practical Applications and Collaborations</strong>: From seeking partnerships for neural network projects to finding efficient strategies to work with open-source models, users exchanged ideas and advice. They covered areas like machine learning, object detection, language models, and the use of serverless GPU services for cost-effective research and development.</li>
  <li><strong>Technical Support and Problem-Solving</strong>: The backend issues of Hugging Face services, such as inference-api serverless timeouts, were discussed, with user experiences highlighting fluctuating performance. Community members also addressed problems with data serialization, style customization in components, and concerns about GPU support for different models.</li>
</ul>
```

**Links mentioned**:

- [BRIA 2.2 FAST - a Hugging Face Space by briaai](https://huggingface.co/spaces/briaai/BRIA-2.2-FAST): no description found
- [Top 10 Serverless GPUs: A comprehensive vendor selection](https://research.aimultiple.com/serverless-gpu/): Explore what is serverless gpu, its benefits for ML models &amp;amp; top serverless gpu providers to deploy your LLMs cheaper &amp;amp; faster.
- [BRIA 2.2 - a Hugging Face Space by briaai](https://huggingface.co/spaces/briaai/BRIA-2.2): no description found
- [Suppress HuggingFace logging warning: &quot;Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.&quot;](https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id): In HuggingFace, every time I call a pipeline() object, I get a warning:&#xA;`&amp;quot;Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.&amp;quot;&#xA;&#xA;How do I supp...
- [GPU portfolio](https://www.ovhcloud.com/fr/lp/gpu-portfolio/): no description found
- [🌌 Analysis of Spaces in Hugging Face](https://huggingface.co/blog/Weyaxi/huggingface-spaces-analysis): no description found
- [Tweet from Weyaxi (@Weyaxi)](https://fxtwitter.com/Weyaxi/status/1761042421243093164): 🎉 New blogpost in @huggingface   🌌 Analysis of Spaces in Hugging Face  I scraped 20K spaces&#39; code files and combined them into one dataset, showcasing meaningful statistics 📶  📝 Blogpost: http...
- [Amanda Ingrao (&#064;artofthemoon.designs) on Threads](https://www.threads.net/@artofthemoon.designs): 6 Followers
- [70k Guns Object Detection Dataset (v5, Main) by Phillip Lavrador](https://universe.roboflow.com/phillip-lavrador/70k-guns/dataset/5): 70277 open source Gun images and annotations in multiple formats for training computer vision models. 70k Guns (v5, Main), created by Phillip Lavrador
- [New Chip Opens Door to AI Computing at Light Speed - Penn Engineering Blog](https://blog.seas.upenn.edu/new-chip-opens-door-to-ai-computing-at-light-speed/): Penn Engineers have developed a new chip that uses light waves, rather than electricity, to perform the complex math essential to training AI. The chip &hellip; Read More &rsaquo;
- [Matrix Calculator](https://www.calculator.net/matrix-calculator.html): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1210507913014026280) (8 messages🔥): 

- **Imitation Learning Inquiry**: User `@alefram` sought advice on starting to learn about **imitation learning for robotics**. No specific resources or tips were provided in response to this query.
- **Deep RL Course Participation**: `@meriem_baziz` expressed their intent to take a **deep RL course** and asked for advice. Again, the community did not provide visible feedback or guidance.
- **Random Insights on LinkedIn**: `@stereoplegic` shared a [LinkedIn article](https://www.linkedin.com/pulse/random-numbers-deep-learning-python-part-4-pytorch-library-jarkko-idkgf) that provides insight into working with **random seeds** in PyTorch, which they recommended as an informative read.
- **CLI Packaging Enigma**: User `@vipitis` is learning how to package **CLI entry points** with `pyproject.toml`, venturing into the intricacies of Python project packaging.
- **V-JEPA Paper Under the Spotlight**: `@subham5089` authored and shared a [blog post](https://www.linkedin.com/posts/subham-kundu-2746b515b_generatieveai-multimodalai-knowledgesharing-activity-7167474445782134786-Wixz) explaining the V-JEPA paper released by Meta, likening the model to BERT for multimodal learning, before being reminded by `@cakiki` to avoid cross-posting in multiple channels.
- **Gemma Model Local Deployment**: `@ariondas` promoted a [LinkedIn post](https://www.linkedin.com/pulse/use-gemma-your-local-ubuntu-machine-using-ollama-arion-das-cpm9c) outlining how to access Google's Gemma model on a local Ubuntu machine.
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1210528152418258954) (33 messages🔥): 

- **Deep Unsupervised Learning Course Spring 2024**: User `@omrylcn` shared a [link](https://sites.google.com/view/berkeley-cs294-158-sp24/home) to the **Berkeley CS294-158 SP24** course on Deep Unsupervised Learning, mentioning that it will cover Deep Generative Models and Self-Supervised Learning, similar to a previous offering.

- **The Emergence of Large Action Models**: `@fernando_cejas` highlighted a blog post discussing **Large Action Models (LAMs)**— an advanced AI system capable of performing human-like tasks within digital environments by mixing language capabilities with task execution.

- **Introducing Galaxy AI with Accessible Models**: User `@white_d3vil` introduced **Galaxy AI** platform offering free API access to various AI models including **GPT-4, GPT-3.5**, and their proprietary **Gemini-Pro**. The platform and models are available for testing in projects as per the [site](https://galaxyapi.onrender.com).

- **Exploring VLM Resolution Challenges and Solutions**: `@osanseviero` recommended two blog posts from HuggingFace discussing the challenges of resolution in vision-language models (VLMs) and presenting a new approach to overcome this issue. It features a demo and relevant models available on the [HuggingFace hub](https://huggingface.co/blog/visheratin/vlm-resolution-curse).

- **Scale AI's Rise in the Data Labeling Market**: User `@valeriiakuka` shared an article from Turing Post about **Scale AI**'s journey to becoming one of the highest-valued companies in the data labeling market, marking its 8th anniversary. The article is part of a series discussing AI Infrastructure Unicorns and can be found [here](https://www.turingpost.com/p/scaleai).

**Links mentioned**:

- [Reader](https://read.readwise.io/new/read/01hqbvnzwgrzgztmzzrf5ycq3y): no description found
- [Warp](https://app.warp.dev/referral/59MJGK): no description found
- [Warp](https://app.warp.dev/block/bmE1t3n7VJt4V6VJAVWxFT): no description found
- [🪆 Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): no description found
- [CS294-158-SP24 Deep Unsupervised Learning Spring 2024](https://sites.google.com/view/berkeley-cs294-158-sp24/home): About: This course will cover two areas of deep learning in which labeled data is not required: Deep Generative Models and Self-Supervised Learning.  Recent advances in generative models have made it ...
- [Scale AI: How to Scale a Company on Every AI Trend](https://www.turingpost.com/p/scaleai): A remarkable journey from appointment apps to data labeling powerhouse
- [Paper page - FiT: Flexible Vision Transformer for Diffusion Model](https://huggingface.co/papers/2402.12376): no description found
- [Galaxy AI - Swagger UI](https://galaxyapi.onrender.com): no description found
- [Breaking resolution curse of vision-language models](https://huggingface.co/blog/visheratin/vlm-resolution-curse): no description found
- [@visheratin on Hugging Face: &quot;VLMs have a resolution problem, which prevents them from finding small details…&quot;](https://huggingface.co/posts/visheratin/787127935781600): no description found
- [Large Action Models (LAMs): A New Step in AI for Understanding and Doing Human Tasks &#8211; Be on the Right Side of Change](https://blog.finxter.com/large-action-models-lams-a-new-step-in-ai-for-understanding-and-doing-human-tasks/): no description found
- [Unveiling the Power of Llamaindex: Jina vs Nomic AI vs FlagEmbedding](https://medium.com/ai-advances/unveiling-the-power-of-llamaindex-jina-vs-nomic-ai-vs-flagembedding-557158d7ad1e?sk=eb9c5b51166a4d4bf34a3490011bfc56): Ankush k Singal
- [Ankush k Singal – Medium](https://medium.com/@andysingal): Read writing from Ankush k Singal on Medium. My name is Ankush Singal and I am a traveller, photographer and Data Science enthusiast . Every day, Ankush k Singal and thousands of other voices read, wr...
- [Verbal lie detection using Large Language Models - Scientific Reports](https://www.nature.com/articles/s41598-023-50214-0#Tab3): no description found

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1210536409308463104) (24 messages🔥): 

<ul>
<li><strong>Bringing Speaker Embeddings to the Browser</strong>: User `@davidre95` announced a pull request for adding support for <b>WavLMForXVector</b> to transformers.js, enabling speaker embeddings models to run in browsers. The related PR can be found on [GitHub here](https://github.com/xenova/transformers.js/pull/603), and the compatible onnx models are available on [Hugging Face](https://huggingface.co/D4ve-R/wavlm-base-plus-sv).</li>
<li><strong>.NET Library for ONNX Inference</strong>: User `@sa_ddam213` introduced a C# <b>.NET library</b> for ONNX model inference without requiring Python, with the code available on [GitHub here](https://github.com/saddam213/OnnxStack).</li>
<li><strong>Open-Source AI Project Unveiled</strong>: User `@flameface` shared a link to <b>Unburn Toys</b>, an open-source AI project which is a collection of useful tools, whose code repository can be found on [GitHub here](https://github.com/flameface/unburn-toys).</li>
<li><strong>Interactive TTS Model Comparison</strong>: User `@realmrfakename` showcased a Hugging Face Space named <b>TTS Arena</b>, which allows users to compare TTS models by listening to samples and voting, available on [Hugging Face here](https://huggingface.co/spaces/TTS-AGI/TTS-Arena). Feedback and pointers to an open TTS tracker were offered by `@pendrokar`.</li>
<li><strong>Philosophical Q&A Dataset Compiled</strong>: User `@nabereon` published a dataset of 133,799 philosophy questions and answers, available on [Hugging Face here](https://huggingface.co/datasets/sayhan/strix-philosophy-qa), and welcomed feedback.</li>
<li><strong>Gradio App for Code-Free AI Experimentation</strong>: User `@nishantsethi_62323` shared their first Gradio app on Hugging Face Space, designed for experimenting with ideas without writing code, accessible on [Hugging Face here](https://huggingface.co/spaces/nsethi610/ns-gradio-apps).</li>
<li><strong>Fine-Tuning LLMs Made Easier</strong>: User `@ameerazam` provided resources for finetuning large language models (LLMs) with less than 7 billion parameters, sharing a repository with code on [Hugging Face here](https://huggingface.co/ameerazam08/gemma-jokes).</li>
</ul>

**Links mentioned**:

- [TTS Arena - a Hugging Face Space by TTS-AGI](https://huggingface.co/spaces/TTS-AGI/TTS-Arena): no description found
- [ameerazam08/gemma-jokes · Hugging Face](https://huggingface.co/ameerazam08/gemma-jokes): no description found
- [D4ve-R/wavlm-base-plus-sv · Hugging Face](https://huggingface.co/D4ve-R/wavlm-base-plus-sv): no description found
- [Ns Gradio Apps - a Hugging Face Space by nsethi610](https://huggingface.co/spaces/nsethi610/ns-gradio-apps): no description found
- [Prompting - ElevenLabs](https://elevenlabs.io/docs/speech-synthesis/prompting#emotion>): no description found
- [Add support for WavlmForXVector by D4ve-R · Pull Request #603 · xenova/transformers.js](https://github.com/xenova/transformers.js/pull/603): Adding support for wavlm with xvector head on top. The onnx version of microsoft/wavlm-base-plus-sv can be found at D4ve-R/wavlm-base-plus-sv. Aims to be as close to the python implementation as po...
- [GitHub - saddam213/OnnxStack: C# Stable Diffusion using ONNX Runtime](https://github.com/saddam213/OnnxStack): C# Stable Diffusion using ONNX Runtime. Contribute to saddam213/OnnxStack development by creating an account on GitHub.
- [sayhan/strix-philosophy-qa · Datasets at Hugging Face](https://huggingface.co/datasets/sayhan/strix-philosophy-qa): no description found
- [GitHub - flameface/unburn-toys: Unburn Toys is an open-source AI project with a bunch of useful tools.](https://github.com/flameface/unburn-toys): Unburn Toys is an open-source AI project with a bunch of useful tools. - flameface/unburn-toys

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1210560103888068649) (26 messages🔥): 

- **Neural Circuit Diagrams Presentation Announced**: `@chad_in_the_house` notified the group that `@1191190979580022875` would present on "Neural Circuit Diagrams: Robust Diagrams for the Communication, Implementation, and Analysis of Deep Learning Architectures". The meeting would be held at [7 pm EST](https://discord.com/channels/879548962464493619/1203285086624157696).
- **Meeting Commenced with Tools and Research Discussion**: `@chad_in_the_house` shared that [Mathcha.io](https://www.mathcha.io/editor) is the tool used for creating diagrams while discussing a paper. A blog post on [`mixtral`](https://www.vtabbott.io/mixtral/) by `@vtabbott_` was also highlighted for future parsing work.
- **Presentation Video Posted on YouTube**: `@chad_in_the_house` posted the presentation video on YouTube with the title [**Hugging Face Reading Group 14: Neural Circuit Diagrams**](https://www.youtube.com/watch?v=pwM_PzqvF9U) and promised to update GitHub with additional content.
- **Upcoming PR Presentation Teaser**: The next week's presentation, hinted by `@chad_in_the_house`, will be by `@563068096747798529` on a PR to the [peft library](https://github.com/huggingface/peft/pull/1364), focusing on new merging methods for LoRA, accompanied by visual illustrations from two arXiv papers ([2306.01708](https://arxiv.org/abs/2306.01708) and [2311.03099](https://arxiv.org/abs/2311.03099)).
- **Scheduling and Attribution for Upcoming Talk**: `@chad_in_the_house` and `@prateeky2806` coordinated scheduling for the next talk through [when2meet](https://www.when2meet.com/?23839966-23Aty), with `@prateeky2806` attributing primary work on the PR to `@871797575454425159` and `@504681610373758977`.

**Links mentioned**:

- [Understanding Mixtral-8x7b](https://www.vtabbott.io/mixtral/): This blog post is adapted from an X thread I posted. Its garnered significant interest, so I decided to post it here as well!  Mixtral-8x7b by @MistralAI is an LLM that outperforms all but OpenAI and ...
- [Mathcha](https://www.mathcha.io/editor): no description found
- [Paper page - Neural Circuit Diagrams: Robust Diagrams for the Communication,
  Implementation, and Analysis of Deep Learning Architectures](https://huggingface.co/papers/2402.05424): no description found
- [Hugging Face Reading Group 14: Neural Circuit Diagrams](https://www.youtube.com/watch?v=pwM_PzqvF9U): Presented by Vincent Abbott
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708): Transfer learning - i.e., further fine-tuning a pre-trained model on a downstream task - can confer significant advantages, including improved downstream performance, faster convergence, and better sa...
- [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099): In this paper, we unveil that Language Models (LMs) can acquire new capabilities by assimilating parameters from homologous models without retraining or GPUs. We first introduce DARE to set most delta...
- [TIE+SuperMario Pres - When2meet](https://www.when2meet.com/?23839966-23Aty): no description found
- [Add new merging methods by pacman100 · Pull Request #1364 · huggingface/peft](https://github.com/huggingface/peft/pull/1364): What does this PR do?  Add new model merging methods for LoRA based on the papers TIES-MERGING: Resolving Interference When Merging Models and Language Models are Super Mario: Absorbing Abilities f...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1211125926205718618) (15 messages🔥): 

- **Inquiry about AnimeBackgroundGAN**: `@mfd000m` asked how to use the model `akiyamasho/AnimeBackgroundGAN` and whether they should clone a repo or use libraries like transformers or diffusion. No specific solution was provided in the subsequent messages.
- **Finetuning Diffusion Models for New Languages**: `@alielfilali01` queried about the possibility of finetuning a diffusion model on a different language corpus instead of a new image style. `@chad_in_the_house` responded, sharing a link to the [Japanese Stable Diffusion model](https://huggingface.co/rinna/japanese-stable-diffusion) which uses a two-stage training procedure tailored for the Japanese language.
- **Loss Zigzag in Model Finetuning**: `@khandelwaal.ankit` is trying to finetune `Qwen/Qwen1.5-0.5B` with a specific dataset but is encountering a zigzag loss graph despite trying various hyperparameters. There were no further clarifications or suggestions concerning this issue.
- **Latent Outputs with the Diffusers Library**: `@shinyzenith` discussed the use of `output_type='latent'` in the stable_diffusion_pipeline from the diffusers library, assuming it yields sampled latent spaces for given prompts. They shared a technical concern about getting NaN values when calculating KL divergence due to negative weights and pondered normalizing the weights, but were unsure if it would distort their analysis.

**Links mentioned**:

- [no title found](https://www.instagram.com/p/C3acjG6r2v-/): no description found
- [rinna/japanese-stable-diffusion · Hugging Face](https://huggingface.co/rinna/japanese-stable-diffusion#training): no description found
- [rinna (rinna Co., Ltd.)](https://huggingface.co/rinna): no description found

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1210597365581611088) (23 messages🔥): 

- **Emotions in Focus**: `rodricota_` mentioned they are building an **emotion recognition model** and wanted to discuss some issues, while `@justinm8449` chimed in stating they've already built such a model.
- **BLIP2 for Image Sequences Inquiry**: `@seanb2792` asked if **BLIP2** could process **image slices from a 3D model** that share context, given their dependence on each other, soliciting thoughts on whether to use a different model for this task.
- **Seeking Robust OCR Models for Complex Characters**: `@icecoldt369` was looking for **OCR models** adept at handling **foreign languages with complex characters**, citing dissatisfaction with results from classic LSTM models. They engaged in dialog with `@cursorop`, discussing the necessity of finetuning and model limitations with lesser-used languages such as Khmer.
- **OCR Model for Multiple Languages Discussed**: `@cropinky` shared a [GitHub link](https://github.com/VikParuchuri/surya) to **surya**, an **OCR and line detection** project that supports over 90 languages, which has been gaining attention recently.
- **Computer Vision Model Benchmarks and Project Ideas Exchanged**: `@coffeevampir3` sought out benchmarks for vision models, to which `@cropinky` recommended the extensive list on [Papers With Code](http://paperswithcode.com/sota). Moreover, `@solution3746` requested ideas for a final year computer vision project and received a suggestion to count people from **CCTV footage**.

**Links mentioned**:

- [GitHub - VikParuchuri/surya: OCR and line detection in 90+ languages](https://github.com/VikParuchuri/surya): OCR and line detection in 90+ languages. Contribute to VikParuchuri/surya development by creating an account on GitHub.
- [Papers with Code - Browse the State-of-the-Art in Machine Learning](http://paperswithcode.com/sota): 12480 leaderboards • 4728 tasks • 9286 datasets • 119860 papers with code.

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1210532994490568754) (109 messages🔥🔥): 

- **Fine-Tuning Follies**: `@jimmyfromanalytics` is facing issues fine-tuning **Flan T5** for generating positive and negative comments on a niche topic and seeks advice. The model is outputting incoherent sentences after fine-tuning, suggesting difficulty in prompt engineering.
- **BERT vs. LLM for Text Classification**: `@arkalonman` asks for sources comparing fine-tuning a larger LLM like **Mistral 7B** or **Gemma 7B** with a standard **BERT** variant for text classification. `@lavi_39761` advises that encoder models are more suited and efficient for classification purposes.
- **Puzzling Finetuning Failures**: `@frosty04212` reports an issue with fine-tuning an already fine-tuned **RoBERTa** model for NER, encountering **0 and NaN** loss values. The issue seems resolved after reinstalling the environment.
- **DeciLM Training Dilemmas**: `@kingpoki` is trying to train **DeciLM 7b** with qlora but encounters a performance warning related to embedding dimension not set to a multiple of 8. Users discuss possible reasons for the warning.
- **Whisper Project Queries**: `@psilovechai` is looking for a local project with an interface like **Gradio** to train and process transcribing audio files using **Whisper**. They receive suggestions for GitHub repositories that could offer a solution.

**Links mentioned**:

- [climatebert (ClimateBert)](https://huggingface.co/climatebert): no description found
- [Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code](https://arxiv.org/abs/2311.07989): In this work we systematically review the recent advancements in code processing with language models, covering 50+ models, 30+ evaluation tasks, 170+ datasets, and 700+ related works. We break down c...
- [Matrix Multiplication Background User&#x27;s Guide - NVIDIA Docs](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc```): no description found
- [DevEval: Evaluating Code Generation in Practical Software Projects](https://arxiv.org/abs/2401.06401): How to evaluate Large Language Models (LLMs) in code generation is an open question. Many benchmarks have been proposed but are inconsistent with practical software projects, e.g., unreal program dist...
- [NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional Correctness](https://arxiv.org/abs/2401.15963): Existing evaluation benchmarks of language models of code (code LMs) focus almost exclusively on whether the LMs can generate functionally-correct code. In real-world software engineering, developers ...
- [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs](https://arxiv.org/html/2312.05934v3): no description found
- [GitHub - jlonge4/whisperAI-flask-docker: I built this project because there was no user friendly way to upload a file to a dockerized flask web form and have whisper do its thing via CLI in the background. Now there is. Enjoy!](https://github.com/jlonge4/whisperAI-flask-docker): I built this project because there was no user friendly way to upload a file to a dockerized flask web form and have whisper do its thing via CLI in the background. Now there is. Enjoy! - jlonge4/w...
- [Reddit - Dive into anything](https://www.reddit.com/r/learnmachinelearning/comments/xly2gp/created_a_gui_for_openais_whisper_using_gradio/): no description found
- [Improve _update_causal_mask performance by alessandropalla · Pull Request #29210 · huggingface/transformers](https://github.com/huggingface/transformers/pull/29210/files): What does this PR do? Fixes # (issue) #29206 Before submitting   This PR fixes a typo or improves the docs (you can dismiss the other checks if that&amp;#39;s the case).  Did you read the contributor ...
- [GitHub - alessandropalla/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/alessandropalla/transformers): 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - alessandropalla/transformers
- [GitHub - innovatorved/whisper-openai-gradio-implementation: Whisper is an automatic speech recognition (ASR) system Gradio Web UI Implementation](https://github.com/innovatorved/whisper-openai-gradio-implementation): Whisper is an automatic speech recognition (ASR) system Gradio Web UI Implementation - innovatorved/whisper-openai-gradio-implementation
- [GitHub - amrrs/openai-whisper-webapp: Code for OpenAI Whisper Web App Demo](https://github.com/amrrs/openai-whisper-webapp?tab=readme-ov-file): Code for OpenAI Whisper Web App Demo. Contribute to amrrs/openai-whisper-webapp development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1211125926205718618) (15 messages🔥): 

- **Introduction to the diffusion-discussion**: `@mfd000m` is new to the discourse on diffusion models and is seeking advice on how to use the model `akiyamasho/AnimeBackgroundGAN`, asking whether they should clone a repository or use libraries like transformers or diffusion.
- **LM Studio Confusion**: `@tmo97` mentions **LM Studio** briefly, triggering a query from `@mfd000m` asking what it is, indicating unfamiliarity with the term or tool.
- **Looking for Guidance in Cross-Language Model Finetuning**: `@alielfilali01` inquires about fine-tuning a diffusion model on different languages rather than image styles, noting a lack of experience with diffusers and an interest in community knowledge on the subject.
- **Challenges in Model Fine-Tuning**: `@khandelwaal.ankit` is experiencing difficulties fine-tuning the **Qwen/Qwen1.5-0.5B** model with a specific dataset, indicating an inconsistent loss graph despite trying various hyperparameters.
- **Sharing Success Stories with Japanese Stable Diffusion**: In response to the language fine-tuning query, `@chad_in_the_house` shares the [Japanese Stable Diffusion model card](https://huggingface.co/rinna/japanese-stable-diffusion#training), explaining the two-stage training procedure as a potential blueprint for similar endeavors.

**Links mentioned**:

- [rinna/japanese-stable-diffusion · Hugging Face](https://huggingface.co/rinna/japanese-stable-diffusion#training): no description found
- [ &#x434;&#x440;&#x435;&#x437;&#x434;&#x43e;&#x43d; on Instagram: &quot;a warm breath&#x2026; #drezzdon&quot;](https://www.instagram.com/p/C3acjG6r2v-/): 27K likes, 105 comments - drezzdon on February 16, 2024: &quot;a warm breath&#x2026; #drezzdon&quot;
- [rinna (rinna Co., Ltd.)](https://huggingface.co/rinna): no description found

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1210544551706632243) (190 messages🔥🔥): 

- **Prompt Template Batching Dilemma**: User `@rwamit` is seeking advice on implementing batching with the langchain wrapper to query GPT-4 due to cost concerns. They shared their method of multiplying a prompt template to process multiple records at once but face an issue with processing time increasing drastically (from 2s/it to 60s/it), ballooning from 5 hours to 96 hours for 5-6k records.

- **Gemma Pytorch Code Curiosities**: A discussion led by users such as `@miaumo` and `@ad8e` revolved around a particular piece of code in [Gemma's PyTorch implementation](https://github.com/google/gemma_pytorch/blob/01062c9ef4cf89ac0c985b25a734164ede017d0b/gemma/model.py#L176) involving RMSNorm with a curious addition of +1. Speculations were made about initialization and the importance of this detail.

- **EfficientNet Debate**: `@vapalus` argued that while EfficientNet might not be ideal for a range of tasks, it performs well as a backbone in segmentation tasks for structured inputs. This followed a critique of EfficientNet by `@fern.bear`, who expressed strong dissatisfaction with the model's marketing and actual performance.

- **Mistral Large Model Released**: Announcement shared about the release of _Mistral Large_, described as a cutting-edge text generation model with strong benchmark results. The announcement highlighted that the model was available through la Plateforme and Azure ([Mistral news](https://mistral.ai/news/mistral-large/)).

- **DPO Paper Clarification Request**: `@staticpunch` inquired about the initialization process of `model_ref` as described in the DPO paper, believing that the suggestion was to conduct Supervised Fine-Tuning (SFT) on preferred completions first, followed by DPO. `@elad7318` and `@alstroemeria313` provided clarification confirming this understanding.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=25040917): no description found
- [SPIN Diffusion Demo V1 - a Hugging Face Space by UCLA-AGI](https://huggingface.co/spaces/UCLA-AGI/SPIN-Diffusion-demo-v1): no description found
- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [gemma_pytorch/gemma/model.py at 01062c9ef4cf89ac0c985b25a734164ede017d0b · google/gemma_pytorch](https://github.com/google/gemma_pytorch/blob/01062c9ef4cf89ac0c985b25a734164ede017d0b/gemma/model.py#L432): The official PyTorch implementation of Google&#39;s Gemma models - google/gemma_pytorch
- [Transformers without Tears: Improving the Normalization of Self-Attention](https://arxiv.org/abs/1910.05895): We evaluate three simple, normalization-centric changes to improve Transformer training. First, we show that pre-norm residual connections (PreNorm) and smaller initializations enable warmup-free, val...
- [gemma_pytorch/gemma/model.py at 01062c9ef4cf89ac0c985b25a734164ede017d0b · google/gemma_pytorch](https://github.com/google/gemma_pytorch/blob/01062c9ef4cf89ac0c985b25a734164ede017d0b/gemma/model.py#L176>): The official PyTorch implementation of Google&#39;s Gemma models - google/gemma_pytorch
- [whisper/whisper/model.py at ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab · openai/whisper](https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L44): Robust Speech Recognition via Large-Scale Weak Supervision - openai/whisper
- [GitHub - BlinkDL/SmallInitEmb: LayerNorm(SmallInit(Embedding)) in a Transformer to improve convergence](https://github.com/BlinkDL/SmallInitEmb): LayerNorm(SmallInit(Embedding)) in a Transformer to improve convergence - BlinkDL/SmallInitEmb
- [GitHub - Stability-AI/StableCascade: Official Code for Stable Cascade](https://github.com/Stability-AI/StableCascade): Official Code for Stable Cascade. Contribute to Stability-AI/StableCascade development by creating an account on GitHub.
- [Support Gemma · turboderp/exllamav2@cc1094a](https://github.com/turboderp/exllamav2/commit/cc1094a41b589f2b1d7a2fcddd8ff1137fbc413f#diff-501d582ac96c58cf6f8a58fc9c96c6a0e033b1440606e25ea21b76e1df469937): no description found
- [Support Gemma · turboderp/exllamav2@cc1094a](https://github.com/turboderp/exllamav2/commit/cc1094a41b589f2b1d7a2fcddd8ff1137fbc413f#diff-be918d4cf7c22a983335f65c5c5841446390e896cbe1c1e0d217ce5880fdddc9): no description found

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1210620741888770118) (84 messages🔥🔥): 

- **Seeking Knowledge on Gated Units Like GRU**: `@mrgonao` inquired about good resources explaining why gated units such as GRU are named as they are, suggesting an interest in the etymology or conceptual reasoning behind the "gated" terminology. No responses provided any links or explanations.

- **Paper Inquiry on Digit-Level Tokenization for Mathematics**: `@stellaathena` asked for the title of a paper concerning digit-level tokenization in mathematics, and `@random_string_of_character` provided a link to the paper titled "Digit-Level Language Models for Digit-level Mathematical Tasks" by Siavash Golkar et al., available at [arxiv.org/abs/2310.02989](https://arxiv.org/abs/2310.02989).

- **Searchformer Paper Generates Buzz**: `@jckwind` shared a link to the paper "Searchformer: Learning to Search Better Than A*" [arxiv.org/abs/2402.14083](https://arxiv.org/abs/2402.14083) which discusses how a Transformer model trained to simulate the search dynamics of $A^*$ search can solve Sokoban puzzles with higher efficiency than traditional methods.

- **RLHF Pitting Simplicity Against PPO**: `@0x_paws` linked to a paper [arxiv.org/abs/2402.14740](https://arxiv.org/abs/2402.14740) that advocates for simpler REINFORCE-style optimization over Proximal Policy Optimization (PPO) in the context of Reinforcement Learning from Human Feedback (RLHF), igniting a discussion on the potential of basic methods in RL for language models.

- **Introducing Watermarking Framework**: In response to `@hyperion.ai`'s query about state-of-the-art text watermarking, `@catboy_slim_` and `@ai_waifu` referred to the watermarking paper "A Watermark for Large Language Models" which suggests embedding signals in generated text [arxiv.org/abs/2301.10226](https://arxiv.org/abs/2301.10226), while `@dmayhem` shared a link to a paper discussing the impossibility of creating robust watermarking schemes under certain assumptions [arxiv.org/abs/2311.04378](https://arxiv.org/abs/2311.04378).

**Links mentioned**:

- [Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping](https://arxiv.org/abs/2402.14083): While Transformers have enabled tremendous progress in various application settings, such architectures still lag behind traditional symbolic planners for solving complex decision making tasks. In thi...
- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740): AI alignment in the shape of Reinforcement Learning from Human Feedback (RLHF) is increasingly treated as a crucial ingredient for high performance large language models. \textsc{Proximal Policy Optim...
- [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226): Potential harms of large language models can be mitigated by watermarking model output, i.e., embedding signals into generated text that are invisible to humans but algorithmically detectable from a s...
- [How Transformers Learn Causal Structure with Gradient Descent](https://arxiv.org/abs/2402.14735): The incredible success of transformers on sequence modeling tasks can be largely attributed to the self-attention mechanism, which allows information to be transferred between different parts of a seq...
- [xVal: A Continuous Number Encoding for Large Language Models](https://arxiv.org/abs/2310.02989): Large Language Models have not yet been broadly adapted for the analysis of scientific datasets due in part to the unique difficulties of tokenizing numbers. We propose xVal, a numerical encoding sche...
- [Towards Efficient and Exact Optimization of Language Model Alignment](https://arxiv.org/abs/2402.00856): The alignment of language models with human preferences is vital for their application in real-world tasks. The problem is formulated as optimizing the model&#39;s policy to maximize the expected rewa...
- [Bayesian Reward Models for LLM Alignment](https://arxiv.org/abs/2402.13210): To ensure that large language model (LLM) responses are helpful and non-toxic, we usually fine-tune a reward model on human preference data. We then select policy responses with high rewards (best-of-...
- [MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases](https://arxiv.org/abs/2402.14905): This paper addresses the growing need for efficient large language models (LLMs) on mobile devices, driven by increasing cloud costs and latency concerns. We focus on designing top-quality LLMs with f...
- [Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models](https://arxiv.org/abs/2311.04378): Watermarking generative models consists of planting a statistical signal (watermark) in a model&#39;s output so that it can be later verified that the output was generated by the given model. A strong...
- [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback](https://arxiv.org/abs/2305.14975): A trustworthy real-world prediction system should produce well-calibrated confidence scores; that is, its confidence in an answer should be indicative of the likelihood that the answer is correct, ena...
- [Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping](https://arxiv.org/abs/2402.14083v1): While Transformers have enabled tremendous progress in various application settings, such architectures still lag behind traditional symbolic planners for solving complex decision making tasks. In thi...
- [Training Chain-of-Thought via Latent-Variable Inference](http://arxiv.org/abs/2312.02179): Large language models (LLMs) solve problems more accurately and interpretably when instructed to work out the answer step by step using a ``chain-of-thought&#39;&#39; (CoT) prompt. One can also improv...
- [Tweet from Lorenzo (Yunze) Xiao (@LrzNeedResearch)](https://x.com/lrzneedresearch/status/1759788360174854597?s=12): Do you feel like your AI anime characters are always out-of-character?   How do we evaluate this?  I am thrilled to introduce our work: InCharacter- a novel perspective to evaluate the personality fid...
- [GitHub - kirilligum/trust-and-teach](https://github.com/kirilligum/trust-and-teach/): Contribute to kirilligum/trust-and-teach development by creating an account on GitHub.
- [GitHub - nbardy/tiny_moe](https://github.com/nbardy/tiny_moe): Contribute to nbardy/tiny_moe development by creating an account on GitHub.
- [MPIrigen: MPI Code Generation through Domain-Specific Language Models](https://arxiv.org/abs/2402.09126v1): The imperative need to scale computation across numerous nodes highlights the significance of efficient parallel computing, particularly in the realm of Message Passing Interface (MPI) integration. Th...
- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [Pipelined Stochastic Gradient Descent with Taylor Expansion](https://www.mdpi.com/2076-3417/13/21/11730): Stochastic gradient descent (SGD) is an optimization method typically used in deep learning to train deep neural network (DNN) models. In recent studies for DNN training, pipeline parallelism, a type ...
- [My AI Safety Lecture for UT Effective Altruism](https://scottaaronson.blog/?p=6823)): Two weeks ago, I gave a lecture setting out my current thoughts on AI safety, halfway through my year at OpenAI. I was asked to speak by UT Austin&#8217;s Effective Altruist club. You can watch the…

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1210547294756143114) (18 messages🔥): 

- **Exploring Linguistic Lens Tuning**: `@butanium` shares a hypothesis that training a tuned **linguistic lens** on Chinese would teach it to translate from English to Chinese, suggesting that if the model originally "thought" in English, this would be the result.
- **Looking into Language Tokens**: `@butanium` predicts that even for English tasks, **Chinese tokens** would become more present, indicating a possible underlying shift due to lens tuning.
- **Code Conundrum with Language Plots**: `@mrgonao` is trying to adjust code to replace "en" tokens with "zh" tokens in plots to understand the Chinese lens better, but time constraints delay a deep dive into the issue.
- **Dataset Dilemma During Translation Task**: `@mrgonao` notes strange behavior with the generated datasets for translation tasks, with incorrect language associations, and clarifies their own error upon discussion with `@butanium`. The issue is documented on [GitHub](https://github.com/SrGonao/llm-latent-language/tree/tuned-lens/visuals/translation).
- **Investigating Multilingual Model Representations**: `@mrgonao` shares a visual analysis of language lenses by considering a neutral language pair (French to German), while `@norabelrose` suggests that language saliency might correlate with corpus frequency. The analysis is based on the **llama-2-7b** model with plans to compare with **llama-2-13b**.
  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1211233369438949397) (30 messages🔥): 

- **Help Wanted: Investigation into `lm_eval` Hanging Issues**: `@flobulous` is having trouble with `lm_eval` hanging indefinitely after running evaluations, specifically when using the `vllm` model. They shared the command and codebase commit [`f78e2da45f034a23b1b13cde3235105b0f55d830`](https://github.com/EleutherAI/lm-evaluation-harness/commit/f78e2da45f034a23b1b13cde3235105b0f55d830) for assistance.

- **Inconsistent LLM Evaluations Revealed**: `@.rand0mm` pointed to a study shared by `@AlhamFikri`, highlighting the inconsistencies between multiple-choice (MCQ) and free-text evaluations of LLMs. The study is detailed in [this paper](https://arxiv.org/abs/2402.13887) on arXiv.

- **Reproducing Open LLM Leaderboard Results with `lm-eval`**: `@hailey_schoelkopf` provided detailed instructions on how to replicate Open LLM Leaderboard results using `lm-eval`. They emphasized using a specific commit and uniform settings as outlined in the Open LLM Leaderboard's HF space.

- **Demand for Better Code-Level Usage of `lm-eval`**: `@ariel2137` inquired about a potential extension and improvements to the "code-level usage" interface of `lm-eval`. `@hailey_schoelkopf` expressed openness to enhancing the usage experience and invited feedback and suggestions.

- **The Need for Support in Multilingual Evaluations**: Conversations initiated by `@.johnnysands` about multilingual evaluations led to the suggestion of duping configs for new languages. `@.rand0mm` mentioned that the MMLU had been translated into French using GPT-3.5 turbo, available on Hugging Face datasets.

**Links mentioned**:

[Tweet from Alham Fikri Aji (@AlhamFikri)](https://x.com/alhamfikri/status/1761963829427109978?s=46&t=of8J2JWAyM5NQncAsmHhQA): Many LLM evaluations use a restrictive multiple-choice (MCQ) format, but in practice, these LLMs are used in a more open-ended, free-text format  🔎Our new study reveals that their probability-based M...

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1210627797790498916) (6 messages): 

- **Python 3.10 Upgrade Hesitation**: `@catboy_slim_` expressed hesitance in upgrading to **Python 3.10** due to concerns over test coverage, implying a lack of urgency for this change.
- **GPT-NeoX Development Curiosity**: `@catboy_slim_` expressed interest in the reasons behind certain development choices, while `@80melon` stated a preference for a **custom training loop** over continuing interest in **GPT-NeoX**.
- **Dealing with Configuration Errors**: `@jdranpariya` encountered a `ValueError` while trying to disable **deepspeed** in the config, indicating potential issues with **NeoXArgs** validation when adjusting settings.
- **Optimization for Multilingual Tokenization**: `@rand0mm` inquired about the best data sources for extending the **Mistral tokenizer** to more effectively represent other languages, pointing to efforts to improve multilingual capabilities.
  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1210719408490553345) (5 messages): 

- **Create-llama Launches LlamaPack Integration**: The `@llama_index` announced the newest create-llama release, which facilitates building full-stack web apps with just two lines of code by using LlamaPack. This feature exemplifies the ease of integrating advanced RAG concepts into projects. [Tweet about create-llama](https://twitter.com/llama_index/status/1761159412629336404)

- **Counselor Copilot Project Highlighted**: A tweet by `@llama_index` featured the Counselor Copilot project as a socially impactful RAG application, serving as an assistant for crisis counselors. The project is also a reference for using advanced RAG as a co-pilot rather than a naive chatbot. [Tweet introducing Counselor Copilot](https://twitter.com/llama_index/status/1761433854458614075)

- **Comprehensive RAG Pain Points Cheat Sheet**: A video walkthrough was shared by `@llama_index` featuring @wenqi_glantz, discussing her "12 RAG Pain Points and Solutions" blog post in depth to address issues at every stage of RAG deployment. The post serves as an essential cheatsheet for those working with RAG. [Tweet about RAG walkthrough](https://twitter.com/llama_index/status/1761553473219551301)

- **Improving RAG Retrieval with Sub-Document Summaries**: `@llama_index` shared a technique to enhance RAG retrieval performance by using sub-document summaries to combat the global concept awareness issue in naive chunking. By injecting summaries as metadata, each chunk gets contextual enhancement. [Tweet discussing chunking trick](https://twitter.com/llama_index/status/1761793821422264757)

- **LlamaParse Overcomes Table Representation Challenges in PDFs**: The `@llama_index` tweet introduced LlamaParse, a PDF parser adept at handling embedded tables and figures, which is crucial for building high-quality RAG applications. Accurate table representation ensures the LLM receives clear information, leading to correct answers. [Tweet about LlamaParse](https://twitter.com/llama_index/status/1762158562657374227)
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1210530643893690390) (234 messages🔥🔥): 

- **Exploring Custom LLMPrompt Templates**: `@andreipopg` is trying to understand how to use a custom prompt with the SubQuestionQueryEngine. The user gets tips like "use the RouterQueryEngine for selecting specific data sources" and is advised that "SubQuestionQueryEngine uses a prompt for generating sub-questions," which can be customized ([GitHub example](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/question_gen)).
  
- **Troubleshooting Install Issues**: `@chbla.` is facing problems with `llama_index` installation, specifically with `set_global_handler` and `Settings`. `@whitefang_jr` suggests a full reinstall with `pip uninstall llama-index`, which resolves `@chbla.`'s issue.

- **RAG vs. No-RAG Evaluation**: `@addo__` is looking to evaluate GPT-3.5 with RAG on a dataset, as compared to using no RAG. `@whitefang_jr` provides a solution using the `FaithfulnessEvaluator` from LlamaIndex for the no-RAG option.

- **Local LLM Integration Inquiry**: `@miteshgarg_61244` seeks to use local offline fine-tuned LLM models with LlamaIndex's `NLSQLTableQueryEngine` and `SQLTableRetrieverQueryEngine`. `@whitefang_jr` recommends setting the local LLM as a global default in `Settings` and possibly deploying the model on a local server using FastAPI.

- **LlamaIndex Chat Engine Details**: `@vett93` wants to know the differences between `index.as_query_engine()` and `index.as_chat_engine()` after observing varying results using different LLMs. `@whitefang_jr` explains that `index.as_query_engine()` queries data for a response, while `index.as_chat_engine()` considers conversation history for stateful interactions.

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/multi_modal/mm_agent.ipynb): no description found
- [no title found](http://localhost:8000",>): no description found
- [no title found](http://localhost:8000">): no description found
- [semantic-text-splitter](https://pypi.org/project/semantic-text-splitter/): Split text into semantic chunks, up to a desired chunk size. Supports calculating length by characters and tokens (when used with large language models).
- [                Insurance Industry Loan Agreements | Justia    ](https://contracts.justia.com/categories/business-finance/subcategories/loan-agreements/industries/insurance/): no description found
- [TikTokLive v6.0.1](https://isaackogan.github.io/TikTokLive/): no description found
- [seman](https://pypi.org/project/seman): no description found
- [LlamaIndex 🦙 0.9.15.post2](https://docs.llamaindex.ai/en/v0.9.15.post2/): no description found
- [Fine-tuning - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning.html): no description found
- [Overview](https://lancedb.github.io/lancedb/hybrid_search/hybrid_search/): no description found
- [Semantic Chunker - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking.html): no description found
- [Cost Analysis - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/understanding/evaluating/cost_analysis/root.html#using-mockllm): no description found
- [Evaluating With LabelledRagDataset’s - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/evaluating_with_llamadatasets.html): no description found
- [Routers - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/root.html#routers): no description found
- [Multi-Tenancy RAG with LlamaIndex - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/examples/multi_tenancy/multi_tenancy_rag.html): no description found
- [Chat Engine - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/root.html): no description found
- [llama_index/llama-index-integrations/question_gen at main · run-llama/llama_index](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/question_gen): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/agent/react/formatter.py at 14c52d42a4a12bc63db7f582e9a17c91f5984f15 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/14c52d42a4a12bc63db7f582e9a17c91f5984f15/llama-index-core/llama_index/core/agent/react/formatter.py#L55): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/agent/react/base.py at 14c52d42a4a12bc63db7f582e9a17c91f5984f15 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/14c52d42a4a12bc63db7f582e9a17c91f5984f15/llama-index-core/llama_index/core/agent/react/base.py#L94): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/agent/react/prompts.py at 14c52d42a4a12bc63db7f582e9a17c91f5984f15 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/14c52d42a4a12bc63db7f582e9a17c91f5984f15/llama-index-core/llama_index/core/agent/react/prompts.py#L7): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [ Globe Explorer - A discovery engine, a wikipedia page for anything | Product Hunt](https://www.producthunt.com/posts/globe-explorer): Explorer is a visual way to breakdown any topic. It uses LLMs to understand your query, and generate an exhaustive page on that topic visually, allowing you to explore information in a way that search...
- [LlamaIndex Webinar: Build No-Code RAG with Flowise](https://www.youtube.com/watch?v=k5Txq5C_AWA): Flowise is one of the leading no-code tools for building LLM-powered workflows. Instead of learning how to code in a framework / programming language, users ...
- [Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex.](https://blog.llamaindex.ai/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5): Discover how to optimize RAG’s chunk size for peak performance using LlamaIndex’s Response Evaluation
- [An Introduction to LlamaIndex Query Pipelines - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline.html): no description found
- [llama_index/llama-index-core/llama_index/core/agent/react/step.py at 14c52d42a4a12bc63db7f582e9a17c91f5984f15 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/14c52d42a4a12bc63db7f582e9a17c91f5984f15/llama-index-core/llama_index/core/agent/react/step.py#L403): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [Introducing LlamaCloud and LlamaParse](https://blog.llamaindex.ai/introducing-llamacloud-and-llamaparse-af8cedf9006b): Today is a big day for the LlamaIndex ecosystem: we are announcing LlamaCloud, a new generation of managed parsing, ingestion, and…
- [Github Repo Reader - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo.html): no description found
- [CodeSplitter - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.CodeSplitter.html): no description found
- [Customizing LLMs within LlamaIndex Abstractions - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html#example-using-a-custom-llm-model-advanced): no description found
- [LocalAI - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/examples/llm/localai.html#llamaindex-interaction): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1211167569973743716) (9 messages🔥): 

- **Misunderstood Metamorphosis Protagonist**: `@daguilaraguilar` is struggling to generate a book review where **"Mr. Samsa"** is recognized as the protagonist instead of Grete. Their [code example](https://www.gutenberg.org/cache/epub/5200/pg5200.txt) incorrectly identifies the main character in Kafka's **"Metamorphosis"**.

- **AI's Kafka Confusion**: `@daguilaraguilar` shared output from their script which mistakenly outputs *"Grete"* as the protagonist for the book **"Metamorphosis"** by Franz Kafka, despite expecting **"Mr. Samsa."**

- **Understanding V-JEPA's Role in Multimodal Learning**: `@subham5089` wrote a [blog](https://www.linkedin.com/posts/subham-kundu-2746b515b_generatieveai-multimodalai-knowledgesharing-activity-7167474445782134786-Wixz) about the V-JEPA paper released by Meta, discussing its significance for multimodal learning and drawing comparisons to BERT in text-based LLMs.

- **Introducing SEC Insights for Financial Analysis**: `@forbes99` introduced [SEC Insights](https://www.secinsights.ai/), a tool designed for analyzing complex financial documents, with features like cross-document inquiries and paragraph-level citations, aiming to enhance business intelligence.

- **Context Management in Large Window LLMs**: `@jonas69301` is in search of benchmarks or evaluations on the best practices for providing extensive context to large context window coding LLMs, such as GPT-4 turbo and Gemini 1.5, with concerns about the order, repetition, and structure of the information.

- **Open-Source Text Generation with Llama2 Model**: `@theexecutor5677` is seeking advice for an open-source text generation application that integrates CSV and PDF inputs with the Llama2 model, and is also interested in combining the approach with RAG (Retrieval-Augmented Generation).

**Links mentioned**:

[no title found](https://www.secinsights.ai/?): no description found

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1210622577819779192) (79 messages🔥🔥): 

- **Swyx Debunks WSJ's Sora Video**: `@swyxio` corrects a claim from a [WSJ video on OpenAI's Sora](https://youtu.be/XllmgXBQUwA), stating that Sora can maintain consistency over >1min videos by interpolating from a start image, contrary to WSJ's assertion of impossibility.
- **NVIDIA Gears up with GEAR**: `@guardiang` shares [news of NVIDIA's new research group](https://x.com/drjimfan/status/1761052023821369639?s=46&t=6FDPaNxZcbSsELal6Sv7Ug), **GEAR**, co-founded by Dr. Jim Fan, aimed at creating autonomous machines with a general-purpose AI.
- **Perplexity Powers Podcast with AI**: `@swyxio` points out [Perplexity's AI-generated podcast](https://podcast.perplexity.ai/) which pulls content from their Discover feed, employing ElevenLabs' voices for narration.
- **Cloudflare Launches AI Gateway**: `@henriqueln7` spotlights [Cloudflare's AI Gateway](https://developers.cloudflare.com/ai-gateway/), offering one-line-code insights and controls for AI applications, including analytics, caching, and rate limiting.
- **Detecting the Details in Data Analysis Tool**: `@swyxio` highlights a [ChatGPT Data Analysis V2](https://x.com/btibor91/status/1761726596585504939?s=46&t=90xQ8sGy63D2OtiaoGJuww) tool utilizing **gpt-4-ada-v2**, featuring a data grid overlay editor, targeted replies, and possibly interactive charts.

**Links mentioned**:

- [Au Large](https://mistral.ai/news/mistral-large/): Mistral Large is our flagship model, with top-tier reasoning capacities. It is also available on Azure.
- [Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)](https://x.com/apples_jimmy/status/1761856728520634383?s=20): Guess people are talking about this because of the trademark? It’s been around for ages.  Subject domain experts temporarily hired by openai write code that openai uses to fine tune their models off. ...
- [AI Gateway · Cloudflare AI Gateway docs](https://developers.cloudflare.com/ai-gateway/): Cloudflare’s AI Gateway allows you to gain visibility and control over your AI apps. By connecting your apps to AI Gateway, you can gather insights on …
- [Tweet from Shu (@shuding_)](https://x.com/shuding_/status/1761085838174175379?s=46&t=90xQ8sGy63D2OtiaoGJuww):   ↘️ Quoting Guillermo Rauch (@rauchg)   AG(UI) has been achieved internally
- [Generative Models: What do they know?](https://intrinsic-lora.github.io/): no description found
- [Fathom - Free AI Meeting Assistant](https://fathom.video/): Records, transcribes & highlights the top moments from your calls. Sends automatically generated call notes to your CRM.
- [One Year of Latent Space](https://www.alessiofanelli.com/posts/latent-space?utm_source=ainews&utm_medium=email&utm_campaign=ainews-one-year-of-latent-space): Lessons (and memories) from going from 0 to 1M readers in 1 year with Latent Space.
- [Discover Daily by Perplexity](https://podcast.perplexity.ai/): We want to bring the world's stories to your ears, offering a daily blend of tech, science, and culture. Curated from our Discover feed, each episode is designed to enrich your day with insights and c...
- [OPENAI FEATHER - OpenAI, Inc. Trademark Registration](https://uspto.report/TM/98010856): Trademark registration by OpenAI, Inc. for the trademark OPENAI FEATHER.
- [Tweet from FxTwitter / FixupX](https://x.com/scottastevens): Sorry, that user doesn't exist :(
- [AI Engineer World&#39;s Fair 2024 - Call for Proposals](https://docs.google.com/forms/d/e/1FAIpQLScc-47zw-tWjYbhAkwTeLy_-MQW3L-3uwtaVnEzudrEZcQ7bg/viewform?pli=1&pli=1): The AI Engineer World&#39;s Fair is a landmark event congregating the top companies, founders, AI Engineers, and software engineers looking to transition into AI Engineering. It&#39;s an event for sof...
- [GenAI Office Hours with Eugene, Hamel, and Jason](https://www.youtube.com/watch?v=tzG1PsqTeZI): Every week we jump on and have a conversation about what we&#39;ve learned through our independent consulting and experience at work
- [Tweet from vik (@vikhyatk)](https://x.com/vikhyatk/status/1761930498518155730?s=20): @natolambert i spoke too soon, forgot i was doing 2x64-shot CoT 😭
- [no title found](https://news.ycombinator.com/item?id=39448254): no description found
- [Tweet from Russ Salakhutdinov (@rsalakhu)](https://x.com/rsalakhu/status/1761062276272902527?s=46&t=90xQ8sGy63D2OtiaoGJuww): Congratulations to my former CMU PhD student Zhilin Yang on his new LLM startup Moonshot AI, raising over $1B in VC funding.  Zhilin has done some fundamental work in NLP and large language models dur...
- [Gemini image generation got it wrong. We&#x27;ll do better.](https://blog.google/products/gemini/gemini-image-generation-issue/): An explanation of how the issues with Gemini’s image generation of people happened, and what we’re doing to fix it.
- [Tweet from hugo alves (@Ugo_alves)](https://x.com/ugo_alves/status/1761857718812315838?s=46&t=90xQ8sGy63D2OtiaoGJuww): For those asking about OpenAI’s Feather
- [Tweet from Jim Fan (@DrJimFan)](https://x.com/drjimfan/status/1761052023821369639?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Career update: I am co-founding a new research group called &#34;GEAR&#34; at NVIDIA, with my long-time friend and collaborator Prof. @yukez. GEAR stands for Generalist Embodied Agent Research.  We be...
- [Tweet from Scott Stevenson (@scottastevenson)](https://x.com/scottastevenson/status/1761824726404436264?s=46&t=90xQ8sGy63D2OtiaoGJuww): The newer preview OpenAI models are performing far worse on legal workloads than plain old GPT-4 for us at @SpellbookLegal. Surprising result after a tonne of subjective testing this weekend.  I recko...
- [Demis Hassabis on Chatbots to AGI | EP 71](https://youtu.be/nwUARJeeplA?si=V09X6h7iqucrh4af): This week’s episode is a conversation with Demis Hassabis, the head of Google’s artificial intelligence division. We talk about Google’s latest A.I. models, ...
- [Tweet from Eugene Yan (@eugeneyan)](https://x.com/eugeneyan/status/1761164851278496204): This is the online environment/tribe I&#39;ve been trying to recreate, where we&#39;re just chatting in the hallway at a conference and friends come up serendipitously.   Started as a 1-on-1 with just...
- [Tweet from Tibor Blaho (@btibor91)](https://x.com/btibor91/status/1761726596585504939?s=46&t=90xQ8sGy63D2OtiaoGJuww): ChatGPT Data Analysis V2 apparently uses a new GPT-4 model called &#34;gpt-4-ada-v2&#34; (Advanced Data Analysis V2). It adds:  - a data grid overlay editor for uploaded files  - an option for a &#34;...
- [Is the AI Boom Real?](https://youtu.be/J-BvkmNtgAM?si=W6XSJocA6odM9kqS): Notes: 7:50 - TPUs are in their fifth iteration. Messed up. Links:- The Asianometry Newsletter: https://www.asianometry.com- Patreon: https://www.patreon.com...
- [OpenAI’s Sora: How to Spot AI-Generated Videos | WSJ](https://youtu.be/XllmgXBQUwA?si=p9): OpenAI just revealed Sora – an AI video generator that creates hyper-realistic scenes and animated worlds in moments. But the tech isn’t perfect. There are a...
- [OpenAI’s Sora: How to Spot AI-Generated Videos | WSJ](https://youtu.be/XllmgXBQUwA?si=p9qTWbKwc3u_JcBx): OpenAI just revealed Sora – an AI video generator that creates hyper-realistic scenes and animated worlds in moments. But the tech isn’t perfect. There are a...

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1210525808612085812) (9 messages🔥): 

- **T5 Paper Discussion Imminent**: `@ivanleomk` announced an LLM Paper Club session led by `@bryanblackbee` to discuss the T5 paper, starting in 5 minutes with a link to join the discussion [here](https://discord.gg/wjrQxPpW).
- **Wishing for a Replay**: `@swyxio` expressed regret for missing the T5 paper discussion and humorously suggested the need for a recording of the session.
- **AI in Action Event Kickoff**: `@kbal11` alerted members about the "AI in Action" event featuring `@yikesawjeez` and focusing on local models, providing a link [here](https://discord.gg/QCPSP7bv) for immediate attendance.
- **Compliments for a Smooth Session**: `@swyxio` complimented `@kbal11` for nicely running the "AI in Action" session with `@yikesawjeez`.
- **Community Celebrates a Milestone**: `@fanahova` shared a birthday celebration message, thanking everyone for being part of the community, followed by `@rubenartus` complementing the celebration cake and hat.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/wjrQxPpW): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/QCPSP7bv): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1210526022488039447) (16 messages🔥): 

- **Get Your LLM Paper Club Notes Here**: `@bryanblackbee` shared a [Notion link](https://www.notion.so/blackbeelabs/Paper-T5-25d26c7d49f7474bb18c90b16eb10413?pvs=4) containing notes pertaining to the LLM Paper Club.
- **Invitation to Engage in LLM Discussions**: `@ivanleomk` invited participants to join the discussion by either speaking up during the session or by dropping questions and topics in the chat.
- **Inquiry into Model Vocabulary and Text Constraints**: `@mattoshimasu` raised questions about whether new models are utilizing a smaller vocabulary set, the length of texts, and the number of verbs.
- **Understanding NLP Fine-Tuning for Newcomers**: `@healthymonkey` inquired about the fine-tuning process for NLP tasks, using T5 and sentiment classification as examples.
- **Architectural Differences in NLP Tasks Discussed**: `@hanzo4958` questioned the effectiveness of encoder-decoder versus decoder-only architecture for traditional NLP tasks.
- **Paper Club Participants Express Gratitude**: Multiple participants including `@healthymonkey`, `@hanzo4958`, `@edwin_75513_08956`, `@lord_idiot`, and `@youngphlo` thanked the hosts for the detailed session and helpful notes.

**Links mentioned**:

[Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://www.notion.so/blackbeelabs/Paper-T5-25d26c7d49f7474bb18c90b16eb10413?pvs=4): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team

  

---


### Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1210693465269338193) (136 messages🔥🔥): 

- **Exploring Latent Space in Local Models**: `@dsquared70` inquired about preferred local models, sparking a conversation about local AI model exploration. `@420gunna` mentioned the curiosity to experiment with image/video generative models and LoRA locally, to which `@markredito` advised checking out resources like `comfyui` and `A1111`.

- **Diving into Model Fine-Tuning with LoRAs**: `@kishore.reddy` and `@markredito` discussed deploying and stacking multiple LoRAs to fine-tune generative models on the same GPU, referencing tools such as `ComfyUI` and platforms like `civit.ai` which host a community sharing models and merged models.

- **Latent Space Final Frontiers Event Highlighted**: `@kbal11` shared information about the Latent Space Final Frontiers event, which focuses on teams pushing the boundaries of AI and features research/startup competitions judged by industry experts. Details and event application can be found [here](https://lu.ma/latent-space-final-frontiers).

- **Local Model Interaction Tools Discussed**: `@markredito`, `@420gunna`, and `@swyxio` discussed `LM Studio` and `Ollama` as tools to pull down language models and interact with them locally. Additionally, `@swyxio` mentioned `gemma.cpp` from Google for model wrapping with streamlined user interfaces.

- **Humor Infused in Tech Banter**: The conversation took a light-hearted turn with jokes about the juxtaposition of high GPU capacity and low internet bandwidth, as highlighted by `@swyxio` and `@kbal11`. This demonstrates the community's ability to infuse humor into technical discussions.

**Links mentioned**:

- [Twitch](https://twitch.tv/yikesawjeez): no description found
- [SDXL Lightning - by fal.ai](https://fastsdxl.ai/): Lightning fast SDXL API demo by fal.ai
- [Merge Large Language Models with mergekit](https://huggingface.co/blog/mlabonne/merge-models): no description found
- [Smol Talk](https://buttondown.email/ainews): We summarize AI discords, and send you a roundup each day!
- [Latent Space: Final Frontiers · Luma](https://lu.ma/latent-space-final-frontiers): We&#x27;re excited to host the second annual Latent Space demo day 🚀 Enough chatting with PDFs. Let&#x27;s see some Science Fiction-level AI. This year&#x27;s theme is Final Frontiers: who are the te...
- [GitHub - deforum-art/deforum-stable-diffusion](https://github.com/deforum-art/deforum-stable-diffusion): Contribute to deforum-art/deforum-stable-diffusion development by creating an account on GitHub.
- [GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.](https://github.com/comfyanonymous/ComfyUI): The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface. - comfyanonymous/ComfyUI

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1210594248819212288) (52 messages🔥): 

- **Gradient Clipping Query and Solution**: `@c.gato` inquired about a potential issue with gradient clipping not working, despite being set to 0.3 in the config, after observing a spike. `@nruaif` suggested the spike may just be temporary, recommending to check if clipping is properly implemented. 

- **DeepSpeed Stage 3 Support Discussion**: `@mihai4256` shared a [GitHub issue](https://github.com/huggingface/transformers/issues/29254) raising concerns about HuggingFace's Trainer supporting DeepSpeed Stage 3, with `@noobmaster29` and `@nanobitz` providing feedback on the usage and recent updates.

- **Axolotl Model Storage and Cleanup**: `@c.gato` sought assistance on where Axolotl stores downloaded models and how to clean up space. `@mihai4256` advised checking the `TRANSFORMERS_CACHE` directory, and shared steps using `huggingface-cli delete-cache` to clear the cache.

- **Mistral AI's Strategic Partnership Draws Attention**: The news about a "strategic partnership" between Microsoft and Mistral AI, including investments and the release of a new AI model, sparked conversation with users like `@yamashi` and `@casper_ai` discussing the implications for open-source model availability and the perceived commercial direction of Mistral AI.

- **Axolotl and OpenAI Mistral Discussions**: A mix of technical support, discussing issues and updates on Axolotl, Mistral AI, and token classification training features, was seen, including `@mihai4256` asking for clarification on installation of deps for non-Python devs and `@kearm` mentioning a new support PR.

**Links mentioned**:

- [Tweet from Casper Hansen (@casper_hansen_)](https://fxtwitter.com/casper_hansen_/status/1762159643344662859): @MistralAI is committed to open-weight models according to their CEO - still bullish  *&#34;Commercial activity will enable us to finance the costly research required for model development. And we wil...
- [Microsoft strikes deal with Mistral in push beyond OpenAI ](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb): no description found
- [DeepSpeed Support Stage 3  · Issue #29254 · huggingface/transformers](https://github.com/huggingface/transformers/issues/29254): System Info Does the trainer support stage 3? According to https://huggingface.co/transformers/v4.3.0/main_classes/trainer.html - it does not. Thanks, Brett Who can help? na Information The officia...
- [Introducing auto_install.sh by monk1337 · Pull Request #1329 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1329/files): Objective: This PR introduces an automated setup script (auto_install.sh) designed to streamline the installation process of Axolotl. It addresses common challenges faced by users not utilizing Doc...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1210689676000624785) (14 messages🔥): 

- **GPTQ and EXL Requirement Clarified**: `@nanobitz` responded to `@curiositix` that they need **gptq** or **exl**, indicating that the suggested **Google**'s **Gemma** C++ inference engine does not meet their requirements.
- **Axolotl's Auto-Install Goodness**: `@stoicbatman` announced the creation of `auto_install.sh` to simplify Axolotl setup ([Pull Request #1329](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1329)), and `@kearm` expressed support for the initiative, urging a review.
- **Seeking Review for Installation Script**: `@stoicbatman` requested a review for the newly introduced `auto_install.sh`, highlighting its goal to ease the installation process, especially for those not using **Docker**.
- **Tweeting for Community Support**: `@casper_ai` created a [Twitter post](https://twitter.com/casper_hansen_/status/1761700050458103964) to garner attention for the **CUDA mode series** potentially with help from Jeremy Howard.
- **Document Clarification with Axolotl PR**: In query to `@caseus_`, `@yamashi` provided a [link](https://github.com/OpenAccess-AI-Collective/axolotl/blob/13199f678b9aab39e92961323bdbce3234ee4b2b/docs/mac.md) to clarify which document `@208256080092856321` was referring to in a discussion regarding **Mistral Lora** within the **Axolotl project**.

**Links mentioned**:

- [axolotl/docs/mac.md at 13199f678b9aab39e92961323bdbce3234ee4b2b · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/13199f678b9aab39e92961323bdbce3234ee4b2b/docs/mac.md): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18pk6wm/how_to_qlora_fine_tune_using_axolotl_zero_to/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/s/NCqFshpmqs): no description found
- [Introducing auto_install.sh by monk1337 · Pull Request #1329 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1329/files): Objective: This PR introduces an automated setup script (auto_install.sh) designed to streamline the installation process of Axolotl. It addresses common challenges faced by users not utilizing Doc...
- [Introducing auto_install.sh by monk1337 · Pull Request #1329 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1329): Objective: This PR introduces an automated setup script (auto_install.sh) designed to streamline the installation process of Axolotl. It addresses common challenges faced by users not utilizing Doc...
- [GitHub - google/gemma.cpp: lightweight, standalone C++ inference engine for Google&#39;s Gemma models.](https://github.com/google/gemma.cpp/): lightweight, standalone C++ inference engine for Google&#39;s Gemma models. - google/gemma.cpp
- [Mps mistral lora by maximegmd · Pull Request #1292 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1292#discussion_r1493791256): Additional MPS example to train a Mistral Lora. Some documentation on usage and limitations.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1210501864139788338) (121 messages🔥🔥): 

- **GPU-Powered Mystery**: `@kearm` discusses an issue with high loss and extended training times using 4 Nvidia RTX 3090 graphics cards. Despite the powerful setup with a Threadripper Pro, the operation led to an estimated training time of 340 hours.

- **Gotta Troubleshoot 'Em All**: Various members, including `@kearm` and `@nanobitz`, delve into technical troubleshooting, trying to identify and solve issues related to high loss during training and checkpoint failures. Configurations, deepspeed versions, and potential fixes are discussed, with `@kearm` experiencing persistent issues despite downgrading deepspeed.

- **300 Seconds of Slowness**: `@dreamgen` asks for assistance regarding slow merging of models, specifically *mixtral*, and unexpected non-utilization of the GPU. The discussion evolves around syncing to main, possible memory issues, and potential docker-related solutions.

- **Docker Dilemma**: `@kearm` attempts to run Axolotl within Docker but faces errors, including GPU connection issues on Ubuntu and a specific error when attempting to run the Docker image. `@nanobitz` points out the need for the **Nvidia container toolkit**, and `@stoicbatman` offers a command template for `@kearm` to facilitate GPU recognition by Docker.

- **Newbie's Navigator Needed**: `@grahama` expresses interest in an easy-to-follow, end-to-end tutorial for beginners wanting to use Axolotl to fine-tune models like *mixtral 7b*. `@nanobitz` indicates that the project README contains a quickstart section that can guide users from setup to inference.

**Links mentioned**:

- [Docker](https://hub.docker.com/r/winglian/axolotl-cloud/tags): no description found
- [Error while saving with EarlyStoppingCallback · Issue #29157 · huggingface/transformers](https://github.com/huggingface/transformers/issues/29157): System Info transformers version: 4.38.0.dev0 (also in 4.38.0 and 4.39.0.dev0) Platform: Linux-5.15.0-78-generic-x86_64-with-glibc2.35 Python version: 3.10.12 Huggingface_hub version: 0.20.3 Safete...
- [fine tune gemma model checkpoint save error · Issue #1320 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1320): Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior should work Current behaviour this error comes when...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1211178569154175067) (5 messages): 

- **Uncertainty Over Training Progress**: `@noobmaster29` remarked that the train loss appeared quite low, potentially indicating good performance.
- **Confusion Over Epoch Results**: `@noobmaster29` expressed confusion as running a full epoch yielded worse results than stopping at 50%, challenging expectations on model training outcomes.
- **Difficulty Assessing Without Evaluation Metrics**: `@noobmaster29` stated the importance of evaluation, noting it's hard to judge the model's performance without it.
- **Acknowledging Assistance**: `@noobmaster29` thanked `@kaltcit` for their help, to which `@kaltcit` responded with "np."
  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1210641300429480048) (3 messages): 

- **Fine-tuning Achievements with Phi-2**: `@finetuningllms` presents a **2.78B parameter finetuned model** of phi-2 without a model card yet, mentioning it was finetuned using axolotl and promises an upcoming card with an image. The model, boasting high performance, can currently be viewed [here](https://huggingface.co/axra/phi-2-x-0.1).

- **Expand Your Language Model's Vocabulary**: `@seungduk` announced the release of **EEVE-Korean models** built with Axolotl, offering optimized Large Language Models (LLMs) with expanded Korean vocabulary. Variants including 10.8B and 2.8B parameter models can be viewed with instructions for use and community engagement on [Hugging Face](https://huggingface.co/yanolja).

- **Korean Language LLM Enhancement Exposed**: Published alongside the models, a **technical report** shared by `@seungduk` details an efficient method for expanding non-English vocabularies in language models and demonstrates their enhanced capabilities in both Korean and English text understanding. Find their research and findings on [arXiv](https://arxiv.org/abs/2402.14714).

- **RAG System Development Simplified**: `@emrgnt_cmplxty` introduced **R2R**, a semi-opinionated framework designed to streamline the transition from experimental Retriever-Answer Generator (RAG) models to production-ready systems. R2R promises ease of deployment, adaptation, and maintenance for production RAG pipelines, and more details can be found on their [GitHub repository](https://github.com/SciPhi-AI/R2R).

**Links mentioned**:

- [axra/phi-2-x-0.1 · Hugging Face](https://huggingface.co/axra/phi-2-x-0.1): no description found
- [GitHub - SciPhi-AI/R2R: A framework for rapid development and deployment of production-ready RAG systems](https://github.com/SciPhi-AI/R2R): A framework for rapid development and deployment of production-ready RAG systems - SciPhi-AI/R2R
- [yanolja/EEVE-Korean-10.8B-v1.0 · Hugging Face](https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0): no description found
- [yanolja/EEVE-Korean-Instruct-10.8B-v1.0 · Hugging Face](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0): no description found
- [yanolja/EEVE-Korean-2.8B-v1.0 · Hugging Face](https://huggingface.co/yanolja/EEVE-Korean-2.8B-v1.0): no description found
- [yanolja/EEVE-Korean-Instruct-2.8B-v1.0 · Hugging Face](https://huggingface.co/yanolja/EEVE-Korean-Instruct-2.8B-v1.0): no description found
- [Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models](https://arxiv.org/abs/2402.14714): This report introduces \texttt{EEVE-Korean-v1.0}, a Korean adaptation of large language models that exhibit remarkable capabilities across English and Korean text understanding. Building on recent hig...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1211033820573204540) (1 messages): 

- **Trouble in Runpod Town**: User `@tom891` reported an error occurring on **runpod** involving a *NameResolutionError* when trying to access 'huggingface.co'. The error suggests a **temporary DNS resolution failure**, positing a potential proxy issue.
  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1210617572299644959) (61 messages🔥🔥): 

- **CUDA Criticized by Computing Legend**: `@itali4no` shared an article where **Jim Keller** criticized NVIDIA's **CUDA** architecture, comparing it unfavorably with x86 and suggesting it lacks elegance due to being cobbled together over time. The full [Tom's Hardware article](https://www.tomshardware.com/tech-industry/artificial-intelligence/jim-keller-criticizes-nvidias-cuda-and-x86-cudas-a-swamp-not-a-moat-x86-was-a-swamp-too) details his viewpoints.

- **Debate Over GPU Choices for AI**: `@cropinky.` suggested that while a **4060 ti** may be the cheapest 16GB consumer GPU and has low power draw, it's generally not enough for LLM tasks compared to options like used **3090**'s with 24GB VRAM, as indicated by `@andreaskoepf` emphasizing VRAM importance. A discussion about buying second-hand GPUs for AI tasks spotlighted the gamble involved and potential remedies if issues arise, including changing thermal pads or paste.

- **Precise Computations in Quantized AI Models Discussed**: `@andreaskoepf` and `@zippika` had an in-depth discussion about how computations in quantized models (4 bit/8 bit) typically happen at higher resolutions like 16 bit to maintain accuracy, with dequantization before matrix multiplication. `@marksaroufim` contributed by clarifying the terms used for different quantization strategies, like weight only quantization and the ambiguity in distributed settings.

- **In-Person Attendance at GTC Conference**: `@vim410` and `@andreaskoepf` suggested organizing either a watch party for Jensen's Keynote or an in-person meetup for those attending the upcoming GTC conference. `_t_vi_` confirmed attendance along with Mike Ruberry and shared excitement for presenting their work.

- **ZLUDA Project Opensourced**: `@ju_rstr` shared news about **ZLUDA**, a tool that allows NVIDIA's CUDA code to run on AMD and Intel GPUs, which has been open-sourced after AMD and Intel withdrew support. The developer behind ZLUDA, Andrzej Janik, hopes his project will challenge NVIDIA's AI dominance, and more information can be found on [ZLUDA's GitHub page](https://github.com/vosen/ZLUDA).

**Links mentioned**:

- [A lone developer just open sourced a tool that could bring an end to Nvidia's AI hegemony &mdash; AMD financed it for months but abruptly ended its support. Nobody knows why](https://www.techradar.com/pro/a-lone-developer-just-open-sourced-a-tool-that-could-bring-an-end-to-nvidias-ai-hegemony-amd-financed-it-for-months-but-abruptly-ended-its-support-nobody-knows-why): ZLUDA could run Nvidia CUDA code on AMD and Intel GPUs
- [Tweet from Jürgen Schmidhuber (@SchmidhuberAI)](https://x.com/SchmidhuberAI/status/1761057748962124205?s=20): 2010 foundations of recent $NVDA stock market frenzy: our simple but deep neural net on @nvidia GPUs broke MNIST https://arxiv.org/abs/1003.0358. Things are changing fast. Just 7 months ago, I tweeted...
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353): Among the widely used parameter-efficient finetuning (PEFT) methods, LoRA and its variants have gained considerable popularity because of avoiding additional inference costs. However, there still ofte...
- [Jim Keller criticizes Nvidia's CUDA, x86 &mdash; 'Cuda&rsquo;s a swamp, not a moat. x86 was a swamp too'](https://www.tomshardware.com/tech-industry/artificial-intelligence/jim-keller-criticizes-nvidias-cuda-and-x86-cudas-a-swamp-not-a-moat-x86-was-a-swamp-too): Jim Keller is not exactly a fan of Nvidia's CUDA.
- [Meet](https://meet.google.com/jcq-zyjr-wjy): Real-time meetings by Google. Using your browser, share your video, desktop, and presentations with teammates and customers.
- [GitHub - vosen/ZLUDA: CUDA on AMD GPUs](https://github.com/vosen/ZLUDA): CUDA on AMD GPUs. Contribute to vosen/ZLUDA development by creating an account on GitHub.
- [Tweet from the tiny corp (@__tinygrad__)](https://x.com/__tinygrad__/status/1760988080754856210): A bunch of rambling about the tinybox. I don&#39;t think there&#39;s much value in secrecy.  We have the parts to build 12 boxes and a case that&#39;s pretty close to final. Beating back all the PCI-E...
- [GitHub - TimDettmers/bitsandbytes at 5d6dfe6fb43e5aae277ec86cba20a002b34df705](https://github.com/TimDettmers/bitsandbytes/blob/5d6dfe6fb43e5aae277ec86cba20a0): Accessible large language models via k-bit quantization for PyTorch. - GitHub - TimDettmers/bitsandbytes at 5d6dfe6fb43e5aae277ec86cba20a002b34df705
- [bitsandbytes/bitsandbytes/functional.py at 5d6dfe6fb43e5aae277ec86cba20a002b34df705 · TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes/blob/5d6dfe6fb43e5aae277ec86cba20a002b34df705/bitsandbytes/functional.py#L1686-L1691): Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes
- [bitsandbytes/csrc/kernels.cu at 5d6dfe6fb43e5aae277ec86cba20a002b34df705 · TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes/blob/5d6dfe6fb43e5aae277ec86cba20a002b34df705/csrc/kernels.cu#L3597-L3604): Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes

  

---


### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1210595819695382639) (6 messages): 

- **Triton as a Gateway to Jax**: `@srush1301` discussed the Triton implementation, mentioning it allows for **Jax support** via Pallas, and expressed a desire for a simpler version for researchers to modify.
- **Triton vs CUDA Multi-GPU Support Inquiry**: `@taekmin.kim` asked if Triton is better than CUDA for multi-GPU or node execution, looking for insights into its distributed computing capabilities.
- **Call for Triton Experts**: `@andreaskoepf` voiced the need for an expert to explain Triton, especially its **lower-level workings**, its foundation in **LLVM and MLIR**, and its future potential.
- **Benchmarking Triton's Quantized Matmul Kernel**: `@andreaskoepf` proposed creating an isolated benchmark setup for Triton's quantized matmul kernel, sharing during a talk to encourage experimentation and comparison with CUDA.
- **Sharing Benchmark Code**: `@andreaskoepf` suggested including the Python file for the aforementioned **benchmark setup** in the lectures repository for accessibility.
  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1210604718318026782) (22 messages🔥): 

- **CUDA vs. Python Rounding Error Issues**: `@zippika` encountered more rounding errors when implementing `nn.Linear` operations in C++ compared to Python due to certain NVIDIA cub compilation flags. A comparison of the code in C++ versus Python was shared illustrating differences that lead to inaccuracies. [Python version deemed more accurate](https://cdn.discordapp.com/emojis/858554607281766420.png).

- **Code Synchronization in Tensor Quantization**: `@zippika` noted the correspondence between `dequantize_torch_fp4` in C++, and `dequantize_fp4_codebook_invoke_qtype` in Python, which have similar functionalities but different argument ordering.

- **Speed Testing BNB vs. TorchFP4**: `@zippika` performed speed tests on the Mistral-7b-instruct-v0.2 model, indicating TorchFP4 had a higher tokens per second rate than BNB.

- **Readme Improvements for `torch-bnb-fp4` Library**: `@zippika` updated the library's readme, [now including an huggingface example script for speed testing](https://github.com/aredden/torch-bnb-fp4/blob/main/examples/speed_test_mistral_7b.py).

- **CUDA with OpenGL vs Vulkan**: `@morousg` answered `@g.huy`'s query about combining CUDA with OpenGL, saying it is possible but NVIDIA focuses more on CUDA with Vulkan. Vulkan is recommended over OpenGL for greater efficiency and capabilities.

**Links mentioned**:

- [torch-bnb-fp4/examples/speed_test_mistral_7b.py at main · aredden/torch-bnb-fp4](https://github.com/aredden/torch-bnb-fp4/blob/main/examples/speed_test_mistral_7b.py): Contribute to aredden/torch-bnb-fp4 development by creating an account on GitHub.
- [TensorRT-LLM/docs/source/performance.md at release/0.5.0 · NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/release/0.5.0/docs/source/performance.md#h100-gpus-fp8): TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficie...
- [GitHub - NVIDIA/TensorRT-LLM: TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT-LLM also contains components to create Python and C++ runtimes that execute those TensorRT engines.](https://github.com/NVIDIA/TensorRT-LLM): TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficie...
- [GitHub - ggerganov/llama.cpp: LLM inference in C/C++](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#vulkan): LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1210690041974624266) (9 messages🔥): 

- **Exploring Efficient Kernel Advertisement**: `@hdcharles_74684` discussed the complexity of making various CUDA kernels accessible, mentioning the release of int_mm via out_dtype as **clunky** and noting the lack of support for int4 in PyTorch. They highlighted a method of integrating efficient kernels through `torch.compile` by detecting certain patterns, referencing their work on a [4-bit Triton kernel](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/post_grad.py#L241-L274).

- **The Limits of `torch.compile`**: `@hdcharles_74684` pointed out the limitations of PyTorch's `torch.compile`, especially in the context of creating *efficient* kernels from simple operations. They plan to address gaps in available kernels, with a focus on weight-only int8 quantization for batch sizes greater than one.

- **Speeding Up CUDA Kernel Compilation**: `@briggers` proposed a method for reducing `cpp_extension.load_inline` compile times, seen in `cuda-mode-session-4.ipynb`, from over 30 seconds to under 2 seconds by using `cpp_extension.load` and avoiding unnecessary header files. A [GitHub repository](https://github.com/pbridger/cuda-experiments) was shared to demonstrate the improved approach, splitting code into separate `.cpp` and `.cu` files.

- **Request for Precompiled Headers (PCH) Guidance**: `@jeremyhoward` requested help with implementing precompiled headers in C++, mentioning it has been years since his last deep involvement with C++.

- **Potential Inefficiency in Recompiling Extensions**: `@briggers` discussed the limitations of using `ninja` to compile extensions, where it recompiles both wrapper and CUDA code even when only algorithm tweaks in the `.cu` file are made. `_t_vi_` contributed that avoiding C++ files during compilation might not be a substantial gain and questioned current PyTorch support for that method.

**Links mentioned**:

- [GitHub - pbridger/cuda-experiments](https://github.com/pbridger/cuda-experiments): Contribute to pbridger/cuda-experiments development by creating an account on GitHub.
- [pytorch/torch/_higher_order_ops/out_dtype.py at ed0ea2f30b2f31be7534a7fdafbed90d247f76b5 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/ed0ea2f30b2f31be7534a7fdafbed90d247f76b5/torch/_higher_order_ops/out_dtype.py#L107)): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [pytorch/torch/_inductor/fx_passes/post_grad.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/post_grad.py#L241-L274): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

  

---


### CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1211036068489732186) (1 messages): 

- **Lecture 7 on Quantization**: `@andreaskoepf` announced that **CUDA-MODE Lecture 7**, titled *Quantization CUDA vs Triton* is scheduled to begin soon. The lecture is starting at a timestamp converted to `<t:1708804800:R>`.
  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1211072282617708604) (4 messages): 

- **CMU's Paper on Efficient LLM Serving**: `@ericauld` shared a link to a paper from CMU, titled ["Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems"](https://arxiv.org/pdf/2312.15234.pdf), focusing on the challenges and methodologies in deploying generative large language models (LLMs) efficiently.
- **Print to Understand**: `@marksaroufim` expressed their intention to print the mentioned CMU paper, indicating their interest in its content.
- **Survey Abstract Highlighted**: `@andreaskoepf` provided a direct link to the abstract of the CMU survey paper on [arXiv](https://arxiv.org/abs/2312.15234), highlighting the need for efficient LLM serving from a machine learning system (MLSys) perspective.
- **Survey Content Breakdown**: `@marksaroufim` shared key insights after reading the survey, noting standout techniques like *non-autoregressive generation*, *speculative decoding*, *MoE architectures*, *local attention variants*, and different forms of parallelism, illustrating the paper's breadth in surveying over 150 referenced works.

**Links mentioned**:

[Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234): In the rapidly evolving landscape of artificial intelligence (AI), generative large language models (LLMs) stand at the forefront, revolutionizing how we interact with our data. However, the computati...

  

---


### CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1211046036282413126) (1 messages): 

- **Efficiency in AI Learning Unveiled**: `@mortezism` shared a [course link](https://hanlab.mit.edu/courses/2023-fall-65940) from MIT focusing on **efficient AI computing techniques** including model compression, pruning, quantization, and more. The course offers hands-on experience with large language models like **LLaMA 2** and covers cutting-edge topics such as quantum machine learning.

**Links mentioned**:

[MIT 6.5940 Fall 2023 TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2023-fall-65940): no description found

  

---


### CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1211427216014180462) (3 messages): 

- **Mistral's Hiring Status Inquiry**: `onuralp.` asked whether **Mistral** is actively hiring in the Bay Area or if hiring is role-specific similar to Deepmind. No public answers were given in the discussion.
- **Nvidia CUDA/C++ positions open**: `@dasher519` inquired about job opportunities at **Nvidia** for CUDA and C++ experts. `@vim410` confirmed that they are hiring and directed applicants to DM their CV for **JobID: JR1968004**.
  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1210635511920721960) (11 messages🔥): 

- **Conundrum with OpenCV in Google Colab**: `@dpearson` is facing difficulties using `'include <opencv2/opencv.hpp>'` in Google Colab with **nvcc4jupyter**. They are seeking alternatives for testing CUDA code on images within a Jupyter notebook environment.
- **Discovering CUDA through YouTube**: `@bikash_p` recommends a YouTube lecture by Jeremy and a related Colab notebook to execute CUDA code using the PyTorch CPP extension, highlighting the seamless integration with ninja for compilation.
- **ACX Community Cross-Pollination**: Both `@ringofbetelgeuse` and `_red.j` express surprise, possibly over finding about CUDA MODE and acknowledge joining from the ACX community.
- **Python Enthusiast's AI Aspiration**: `@ilovepython3` voices their aspirations to fine-tune AI models, despite self-proclaimed poor math skills, and queries about prerequisites for engaging with CUDA MODE.
- **Guidance for a Budding AI Enthusiast**: In response to `@ilovepython3`'s query regarding where to start, `@jeremyhoward` suggests tackling the **fast.ai** course first to build foundational knowledge before diving into CUDA.

**Links mentioned**:

- [Lecture 3: Getting Started With CUDA for Python Programmers](https://youtu.be/4sgKnKbR-WE)): Recording on Jeremy&#39;s YouTube https://www.youtube.com/watch?v=nOxKexn3iBoSupplementary Content: https://github.com/cuda-mode/lecture2/tree/main/lecture3Speak...
- [Google Colaboratory](https://colab.research.google.com/drive/180uk6frvMBeT4tywhhYXmz3PJaCIA_uk?usp=sharing)): no description found

  

---


### CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1211040809588359208) (3 messages): 

- **Confusion Over Grid Diagram in PMPP Book**: `@bikash_p` questioned a discrepancy in the PMPP book, where the code specifies `dim3 dimGrid(2,2,1)`, but the accompanying diagram shows two separate grids. They wondered if the diagram should instead show a single grid with four blocks.
- **Clarification on Kernel Function Calls and Grids**: `@alexanderrgriffing` responded to `@bikash_p` clarifying that the figure represents multiple kernel function calls, with each call launching its own grid of thread blocks. Hence, two kernel calls result in two separate grids.
- **Appreciation for Community Support**: `@bikash_p` expressed gratitude for the explanation provided by `@alexanderrgriffing` regarding the schematic representation of grids in CUDA code context from the PMPP book.
  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1210638897957052477) (3 messages): 

- **Optimization Education Straight from YouTube**: `@marksaroufim` shared **Lecture 6** focusing on **Optimizing Optimizers** with both a [YouTube video](https://www.youtube.com/watch?v=hIop0mWKPHc) and the accompanying slides in a [Google Docs presentation](https://docs.google.com/presentation/d/13WLCuxXzwu5JRZo0tAfW0hbKHQMvFw4O/edit#slide=id.p1).
- **Gratitude Expressed by `@filippob82`**: `@filippob82` expressed thanks for the shared educational content on CUDA optimization.
- **Taking Quantization Further**: `@andreaskoepf` provided a link to **Lecture 7** titled **Advanced Quantization** on YouTube ([watch here](https://youtu.be/1u9xUK3G4VM?si=ssW_DEDqBIRHpNYN)) and thanked `@325883680419610631` for recording, cutting, and uploading the lecture, with additional slides available on [Dropbox](https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&dl=0).

**Links mentioned**:

- [Lecture 7 Advanced Quantization](https://youtu.be/1u9xUK3G4VM?si=ssW_DEDqBIRHpNYN): Slides: https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&amp;dl=0
- [Lecture 6 Optimizing Optimizers](https://www.youtube.com/watch?v=hIop0mWKPHc): Slides: https://docs.google.com/presentation/d/13WLCuxXzwu5JRZo0tAfW0hbKHQMvFw4O/edit#slide=id.p1

  

---


### CUDA MODE ▷ #[smol-hw](https://discord.com/channels/1189498204333543425/1205223658021458100/1210841747450630164) (8 messages🔥): 

- **Contemplating Random Numbers**: User `@marksaroufim` posted a [range of numbers](https://github.com/TimDettmers/bitsandbytes/commit/67475257a96b792f9b66e71892dab90f7a60ed87) with no context, sparking curiosity from `@nshepperd` about the origin of these values.
- **Contributions to Quantization**: `@drisspg` shared progress on quantization techniques with notes on reproduction, and provided a link to their [GitHub repository](https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/quant/qlora_debug.py) with relevant code.
- **Doubts about Quantile Alignment**: `@drisspg` revealed skepticism about the alignment of quantiles to expectations mentioning having a notebook with related concerns but did not provide a link to it.
- **Exploring Quantization Strategies**: `@marksaroufim` highlighted a [PyTorch core team repository](https://github.com/pytorch-labs/ao) focused on quantization and pruning of GPU models and referred to a [PyTorch blog post](https://pytorch.org/blog/accelerating-generative-ai-2/) detailing optimizations in generative AI accelerations.

**Links mentioned**:

- [transformer_nuggets/transformer_nuggets/quant/qlora_debug.py at main · drisspg/transformer_nuggets](https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/quant/qlora_debug.py): A place to store reusable transformer components of my own creation or found on the interwebs - drisspg/transformer_nuggets
- [Added documentation for NF4; failing 8-bit matmul; fixed absmax bug. … · TimDettmers/bitsandbytes@6747525](https://github.com/TimDettmers/bitsandbytes/commit/67475257a96b792f9b66e71892dab90f7a60ed87): …#529 #543
- [GitHub - pytorch-labs/ao: The torchao repository contains api&#39;s and workflows for quantization and pruning gpu models.](https://github.com/pytorch-labs/ao): The torchao repository contains api&#39;s and workflows for quantization and pruning gpu models. - pytorch-labs/ao
- [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/): This post is the second part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance...
- [Accelerating Generative AI with PyTorch: Segment Anything, Fast](https://pytorch.org/blog/accelerating-generative-ai): This post is the first part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance ...

  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1210551834150707231) (45 messages🔥): 

- **Tweaking Attention for Speed**: `@zhuzilin96` implemented a `zigzag_ring_flash_attn_varlen_qkvpacked_func`, which showed a speed improvement although less than anticipated. They later mentioned hardcoding bf16 was due to personal preference rather than necessity.
- **Flash Attention Finessed**: `@iron_bound` shared an explanation and visual from [Hugging Face's documentation](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention) about **Flash Attention**, highlighting its benefits for memory efficiency and training speed by leveraging SRAM over HBM.
- **Zigzag Ring Speed-Up Measured**: `@zhuzilin96` posted a [benchmark script](https://github.com/zhuzilin/ring-flash-attention/blob/main/benchmark_qkvpacked_func.py) showing a roughly 20% speed up in zigzag ring attention over classic flash attention, but admitted that their earlier screenshot wasn't correctly warmed up.
- **Ring to the Max**: `@andreaskoepf` discussed maximizing the benefits of RingAttention for larger batch sizes, noting that it's crucial to measure when the ring-attn-block computation outweighs memory transfer time. Meanwhile, `@jamesmel` contributed a minor PR for requirements and `@andreaskoepf` clarified that the Cuda Mode fork is mainly for backup purposes.
- **In-Depth Optimization Discussions**: `@w0rlord` and `@andreaskoepf` engaged in discussions about softmax base 2 tricks and flash attention function accuracy with respect to sequence lengths. `@andreaskoepf` shared a [notebook](https://github.com/cuda-mode/ring-attention/blob/main/trition_flash_attn/softmax_base2_trick.ipynb) regarding the trick and observed that flash attention gave correct results only for longer sequences.

**Links mentioned**:

- [Flash Attention](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention): no description found
- [Should exp2 be faster than exp?](https://stackoverflow.com/questions/30222836/should-exp2-be-faster-than-exp): I&#x27;m mostly interested in the &quot;exp&quot; and &quot;exp2&quot; functions in C/C&#x2B;&#x2B;, but this question is probably more related to the IEEE 754 standard than specific language features...
- [A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library](https://arxiv.org/abs/2312.11918): We provide an optimized implementation of the forward pass of FlashAttention-2, a popular memory-aware scaled dot-product attention algorithm, as a custom fused CUDA kernel targeting NVIDIA Hopper arc...
- [Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120): Transformer achieves promising results on various tasks. However, self-attention suffers from quadratic memory requirements with respect to the sequence length. Existing work focuses on reducing time ...
- [Wait time instrumentation [not intended to be merged] by andreaskoepf · Pull Request #9 · zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention/pull/9): I tried to measure the time spent in the reqs returned  batch_isend_irecv(). Interestingly this time seems to be indepentent of sequence length and in total negligible. Could be that on a single no...
- [ring-attention/trition_flash_attn/softmax_base2_trick.ipynb at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/blob/main/trition_flash_attn/softmax_base2_trick.ipynb): ring-attention experiments. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.
- [ring-attention/trition_flash_attn/workbench.py at 391a4cce570aae380ad5b318cb4b0f80f4cb3aee · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/blob/391a4cce570aae380ad5b318cb4b0f80f4cb3aee/trition_flash_attn/workbench.py#L38-L54): ring-attention experiments. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.
- [Smol Talk](https://buttondown.email/ainews): We summarize AI discords, and send you a roundup each day!
- [A ring attention with flash attention kernel implementation · Issue #4 · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/issues/4): Hi! Thank you for your work on implementing the ring attention in pytorch! I&#39;ve just tried to implement a ring_flash_attn_qkvpacked_func (corresponding to flash_attn_qkvpacked_func in flash attent...
- [Pull requests · zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention/pull/): Ring attention implementation with flash attention - Pull requests · zhuzilin/ring-flash-attention
- [added requirements.txt by melvinebenezer · Pull Request #7 · zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention/pull/7/files): Had errors running the test cases Following the repo from Cuda Mode -  ring attention channel
- [Stripe Attn by reyoung · Pull Request #6 · zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention/pull/6): [x] complete function implementation [x] complete unittest

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1210505745167286292) (39 messages🔥): 

- **Function Calling Dilemma with Local Models**: `@saita_ma_` is seeking an easy way to do function calling with local models like **OpenHermes** and has found resources lacking, despite knowing it's possible as demonstrated by **CrewAI**.
- **Langchain Tutorials Hit YouTube**: `@datasciencebasics` shares a YouTube tutorial on [creating a Chat UI using ChainLit, LangChain, Ollama & Gemma](https://youtu.be/n9AMtXLveMs), which allows viewers to create a **ChatGPT-like UI** locally.
- **Colab Corner**: `@kenwu_` is looking for help with agent and function calling using **Cohere API** and LangChain; shared their [Google Colab notebook](https://colab.research.google.com/drive/14IOr0PZY9Skpc7IjxSeN-GZekNoI3I1U?usp=sharing) for collaboration and assistance.
- **Sarcasm Detection in LLMs**: `@juepachon` sparks a conversation on whether tagging phrases with "sarcasm" could help an LLM to understand and detect sarcasm better after fine-tuning.
- **Usescraper Launch and Blog Post**: `@dctanner` announces [UseScraper.com](https://usescraper.com/), a new tool for crawling website content, and wrote a blog post on how it ties in with **LangChain**.

**Links mentioned**:

- [Redirecting...](https://errors.pydantic.dev/2.6/v/missing): no description found
- [Google Colaboratory](https://colab.research.google.com/drive/14IOr0PZY9Skpc7IjxSeN-GZekNoI3I1U?usp=sharing): no description found
- [Get started | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/get_started): LCEL makes it easy to build complex chains from basic components, and
- [Llama.cpp | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/llms/llamacpp): llama-cpp-python is a
- [Streaming | 🦜️🔗 Langchain](https://js.langchain.com/docs/use_cases/question_answering/streaming#chain-with-sources): Often in Q&amp;A applications it’s important to show users the sources that
- [Extract Topics From Video/Audio With LLMs (Topic Modeling w/ LangChain)](https://www.youtube.com/watch?v=pEkxRQFNAs4.): Learn To Build With AI: https://mail.gregkamradt.com/signupTwitter: https://twitter.com/GregKamradtCode: https://github.com/gkamradt/langchain-tutorials/blob...
- [Create Chat UI Using ChainLit, LangChain, Ollama &amp; Gemma 🧠](https://youtu.be/n9AMtXLveMs): In this video, I am demonstrating how you can create a simple ChatGPT like UI locally in your computer. You can follow along with me by cloning the repo loca...
- [no title found](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/medlm): no description found
- [GitHub - google/generative-ai-python: The Google AI Python SDK enables developers to use Google&#39;s state-of-the-art generative AI models (like Gemini and PaLM) to build AI-powered features and applications.](https://github.com/google/generative-ai-python): The Google AI Python SDK enables developers to use Google&amp;#39;s state-of-the-art generative AI models (like Gemini and PaLM) to build AI-powered features and applications. - google/generative-ai-p...

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1210786636032647238) (3 messages): 

- **Cancelled Error Confusion**: User `@cryptossssun` encountered an `asyncio.exceptions.CancelledError` but did not provide further details about the context or the code involved.
- **Query about Extending Timeout Limits**: `@howtonotgiveafuck` is looking for a way to **extend the timeout** beyond the default 900 seconds. No solutions or further discussion on the topic were provided within the scope of the messages.
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1210624662720217169) (11 messages🔥): 

- **Build Custom Chatbots with Ease**: `@deadmanabir` shared a guide on crafting personalized chatbots that maintain a conversation history. The technology stack includes OpenAI, Qdrant DB, and Langchain JS/TS SDK, with more details available on [Twitter](https://twitter.com/ItsDutta99/status/1761064358321525235).

- **Insights on AI in the Insurance Industry**: `@solo78` expressed interest in exchanging use cases and implementing AI, particularly in the finance function within the insurance sector.

- **Merlinn AI Empowers Engineers**: `@david1542` introduced [Merlinn](https://merlinn.co/), a project that aids on-call engineers in incident investigations and troubleshooting, utilizing Langchain under the hood.

- **Langchain on Rust**: `@edartru.` shared [Langchain-rust](https://crates.io/crates/langchain-rust), a new crate enabling Rust developers to write programs with large language models, with the [source code available on GitHub](https://github.com/Abraxas-365/langchain-rust).

- **Novel Resume Optimizer Launch**: `@eyeamansh` developed an open-source resume optimizer using AI, which proved successful in securing calls from tech giants like NVidia and AMD. The tool is designed to reduce cost and effort and can be found on [GitHub](https://github.com/AnshKetchum/resumeop).

**Links mentioned**:

- [Merlinn - Resolve incidents fast using AI](https://merlinn.co/): Investigate production incidents efficiently using AI; Empower your team by an AI agent that knows your environment.
- [GitHub - consumer-ai-lab/microservices-based-chatbot-api](https://github.com/consumer-ai-lab/microservices-based-chatbot-api): Contribute to consumer-ai-lab/microservices-based-chatbot-api development by creating an account on GitHub.
- [GitHub - AnshKetchum/resumeop: Go the extra mile, without wasting thousands of hours. Achieve job market freedom using open source AI.](https://github.com/AnshKetchum/resumeop): Go the extra mile, without wasting thousands of hours. Achieve job market freedom using open source AI. - AnshKetchum/resumeop
- [GitHub - SciPhi-AI/R2R: A framework for rapid development and deployment of production-ready RAG systems](https://github.com/SciPhi-AI/R2R): A framework for rapid development and deployment of production-ready RAG systems - SciPhi-AI/R2R
- [SalesGPT: Elevating Sales Conversations with Langchain Intelligence](https://medium.com/ai-advances/salesgpt-elevating-sales-conversations-with-langchain-intelligence-a1e1be461ee4): Ankush k Singal
- [GitHub - Abraxas-365/langchain-rust: LangChain for Rust, the easiest way to write LLM-based programs in Rust](https://github.com/Abraxas-365/langchain-rust): LangChain for Rust, the easiest way to write LLM-based programs in Rust - Abraxas-365/langchain-rust

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1210600621854425099) (7 messages): 

- **Build Your Own Chat UI**: A newly shared [YouTube video](https://youtu.be/n9AMtXLveMs) demonstrates how to create a **Chat UI** using ChainLit, LangChain, Ollama, & Gemma, enabling viewers to set up a **ChatGPT-like interface** locally on their computer.
- **LLMs Illuminate Quarterly Reports**: @rito3281 has crafted a [detailed article](https://rito.hashnode.dev/daily-portfolio-summarizer-with-langchain-qdrant-and-mistral-ai) discussing how **Large Language Models (LLMs)** can assist in parsing through a company's quarterly report, predicting future growth, and identifying risks and market opportunities, using **LangChain, Qdrant, and Mistral AI**.
- **Ollama's New Embeddings on Colab**: @schimazing shares a modification that utilizes **Ollama's new embeddings** completely hosted on Google Colab, as highlighted in this [Twitter post](https://twitter.com/theReedTard/status/1761107453465252120?s=19), with no API keys required.
- **Decoding the AI Process**: In response to @rajib2189's inquiry about the underlying mechanisms of AI, @speuce clarified that the process is **perplexity-based** rather than relying on stopwords or stemming.
- **LangGraph, Calls, and Scraping Simplified**: @tarikkaoutar presents a [YouTube video](https://www.youtube.com/watch?v=q5LvDHiSBy4) that explains how to combine LangGraph, function calls, and a web scraper to create a **multi-agent application**, encouraging shares to broaden reach.

**Links mentioned**:

- [Daily Portfolio Summarizer with Langchain, Qdrant, and Mistral AI](https://rito.hashnode.dev/daily-portfolio-summarizer-with-langchain-qdrant-and-mistral-ai): Today&#x27;s Investors are bombarded with news, reports, statistics, and more information. AI cuts through this noise, analyzes vast datasets to unearth hidden patterns and trends, and offers insights...
- [Create Chat UI Using ChainLit, LangChain, Ollama &amp; Gemma 🧠](https://youtu.be/n9AMtXLveMs): In this video, I am demonstrating how you can create a simple ChatGPT like UI locally in your computer. You can follow along with me by cloning the repo loca...
- [LangGraph + Function Call + Web Scraper = Multi-Agent Application](https://www.youtube.com/watch?v=q5LvDHiSBy4): #chatbot #langgraph #functioncall #ai #automation #dropshipping In this video, I will explain how you can create a LangGraph, make function calls, and develo...

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1211734565870370876) (4 messages): 

- **LOL: ChatGPT Goes Multilingual with Data**: `derekpwillis` shared an anecdote where using **chatgpt-3.5-turbo** for data extraction tasks resulted in some document titles being translated into Spanish, such as "Taking Advantage of the Internet" becoming "*Sacándole Provecho a Internet*".
- **The Multilingual Bug Strikes Again**: `simonw` compared this behavior to a known issue where ChatGPT, coupled with **Whisper voice**, sometimes misinterprets British accents as Welsh and responds in Welsh.
- **Quick Fix Suggestion**: `simonw` suggested a workaround by using a system prompt specifying **"Always use English"** to avoid erroneous language detection.
- **Ready to Implement the Language Patch**: `derekpwillis` acknowledged the bug and expressed the intention to implement the "Always use English" prompt to address the issue.
  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1210560212474269726) (30 messages🔥): 

- **Reviving Old School Prompt Crafting**: `@tariqali` reminisced about the pre-RLHF approach of using extensive prompts to guide text generation, finding it reminiscent of providing chatbots with a transcript to resume conversations. He finds this method offers more control, especially useful for instances like incomplete chatbot messages caused by "time out" issues.

- **Simplifying Devcontainer Setups and Workarounds**: `@derekpwillis` mentioned having to tinker with the `devcontainer.json` file, while `@simonw` suggested adding `llm models` to the `setup.sh` script as a bug workaround. `@derekpwillis` later confirmed the implementation of the proposed fix.

- **LargeWorldModel Running on LLM**: `@simonw` expressed interest in seeing [LargeWorldModel](https://largeworldmodel.github.io/) running in LLM and discussed the possibility of using GPU instances to accommodate PyTorch models from their [Hugging Face repository](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-1M).

- **Plugin for Groq Inference by Angerman**: `@angerman.` shared his creation of a Groq inference plugin, [llm-groq](https://github.com/angerman/llm-groq), contributing another inference provider for experimentation. `@0xgrrr` cheered on the addition, inquiring about the performance claims.

- **Publishing to PyPI for Easier Plugin Installation**: `@angerman.` learned to publish his [llm-groq plugin](https://pypi.org/project/llm-groq/) to PyPI following `@0xgrrr`'s advice, enabling simpler installation using `llm install`. `@angerman.` confirmed successful publishing and expressed his experience comparing Haskell and Python community practices.

**Links mentioned**:

- [LargeWorldModel/LWM-Text-Chat-1M · Hugging Face](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-1M): no description found
- [Large World Models](https://largeworldmodel.github.io/): no description found
- [Packaging Python Projects - Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives): no description found
- [llm-groq](https://pypi.org/project/llm-groq/): no description found

  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1210624176927281162) (6 messages): 

- **Examining Hallucination Mitigation**: User `@res6969` shared a [tweet by @RichardSocher](https://x.com/RichardSocher/status/1760800655428796772?s=20) discussing a potential solution to the hallucination problem in AI. The tweet alludes to successful reference incorporation, stirring curiosity in the research community.
- **Speculating on Anti-Hallucination Techniques**: `@res6969` speculated that the approach to curb hallucinations involves a validating mechanism coupled with cutting-edge embedding models. This suggests a growing interest in enhancing AI's factual accuracy.
- **Introducing Globe Explorer**: User `@sincethestudy` announced the launch of [Globe Explorer](http://explorer.globe.engineer/), a tool that creates a customizable Wikipedia-style page on any topic using GPT-4, heralding a new era in information discovery.
- **Globe Explorer Seeks Product Hunt Supremacy**: In an effort to top Product Hunt’s daily list, `@sincethestudy` urged the community to upvote [Globe Explorer](https://www.producthunt.com/posts/globe-explorer). Promises of exclusive access to a "pro" version were offered to supporters.

**Links mentioned**:

- [Tweet from brian-machado-finetuned-7b (e/snack) (@sincethestudy)](https://x.com/sincethestudy/status/1761099508853944383?s=20): Globe Explorer is kinda like a custom wikipedia page on anything you want.  We are entering a new age of information discovery.  go try it: http://explorer.globe.engineer/
- [Tweet from Richard Socher (@RichardSocher)](https://x.com/RichardSocher/status/1760800655428796772?s=20): Did we solve the hallucination problem? It is starting to look like it here and in any other example I&#39;ve tried in research mode - all with tons of up-to-date references.  Query: Reddit S-1
- [ Globe Explorer - A discovery engine, a wikipedia page for anything | Product Hunt](https://www.producthunt.com/posts/globe-explorer): Explorer is a visual way to breakdown any topic. It uses LLMs to understand your query, and generate an exhaustive page on that topic visually, allowing you to explore information in a way that search...

  

---


### LLM Perf Enthusiasts AI ▷ #[finetuning](https://discord.com/channels/1168579740391710851/1168582249738944532/1210641609704865793) (1 messages): 

- **Fine-Tuning with Full Documents or Extracts?**: `@pantsforbirds` is achieving **great results with 1-shot data extraction** using **gpt-4-turbo** by embedding entire documents into the prompt. They seek advice on whether to embed full example documents or just relevant sections in their finetuning dataset for a more complicated extraction/classification task.
  

---


### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/1211631845087649893) (3 messages): 

- **FireFunction V1 Sparks Interest**: `@sourya4` asked for top choices for function calling with open-weights models. They then shared a link to `@lqiao`'s announcement about **FireFunction V1**, poised to deliver GPT-4-level structured output and decision-routing at higher speeds, and also stated open-weights availability and commercial usability with supportive [blog post](https://fireworks.ai/blog/firefunction-v1-gpt-4-level-function-calling).

- **Structured Output for Better Development**: The announcement from `@lqiao` further introduced **JSON mode and grammar mode** for all language models, ensuring structured outputs and reducing time spent on system prompts, detailed in a second [blog post](https://fireworks.ai/blog/why-do-all-LLMs-need-structured-output-modes).

- **Hackathon for Hands-on Experience**: `@yikesawjeez` mentioned current preferred tools for function calling, including gorilla openfunctions and others, but flagged an upcoming hackathon focused on **FireFunction** as a potential game-changer in determining a new favorite.

**Links mentioned**:

[Tweet from Lin Qiao (@lqiao)](https://x.com/lqiao/status/1760664322215379153?s=12): 🔥 Structure is all you need. 🔥  We’re excited to announce:  - FireFunction V1 - our new, open-weights function calling model:     - GPT-4-level structured output and decision-routing at 4x lower lat...

  

---


### LLM Perf Enthusiasts AI ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/1210697163684978788) (5 messages): 

- **Introducing Globe Explorer**: `@joshcho_` shared a tweet by `@sincethestudy` introducing **Globe Explorer**, likening it to a customizable Wikipedia page for anything and hailing it as a herald of a new age in information discovery. They encouraged people to try it at [explorer.globe.engineer](http://explorer.globe.engineer/).
- **Journey of Viral Spread**: `@joshcho_` humorously noted that a request for widespread sharing of Globe Explorer was unnecessary, as it had already become viral.
- **Launch of R2R for RAG Systems**: `@emrgnt_cmplxty` announced the launch of **R2R**, a framework to facilitate the rapid development and deployment of production-ready Retriever-And-Generator (RAG) systems, and provided a link to the [GitHub repository](https://github.com/SciPhi-AI/R2R). They emphasized the framework's simplicity and its aim to set a new benchmark for ease of use in production environments.

**Links mentioned**:

- [Tweet from brian-machado-finetuned-7b (e/snack) (@sincethestudy)](https://x.com/sincethestudy/status/1761099508853944383?s=46): Globe Explorer is kinda like a custom wikipedia page on anything you want.  We are entering a new age of information discovery.  go try it: http://explorer.globe.engineer/
- [GitHub - SciPhi-AI/R2R: A framework for rapid development and deployment of production-ready RAG systems](https://github.com/SciPhi-AI/R2R): A framework for rapid development and deployment of production-ready RAG systems - SciPhi-AI/R2R

  

---


### LLM Perf Enthusiasts AI ▷ #[collaboration](https://discord.com/channels/1168579740391710851/1168816033130365018/1211154406238715914) (3 messages): 

- **Anki and LLM Collaboration Potential**: User `@degtrdg` shared [a tweet](https://x.com/nc_znc/status/1753847802487017911?s=46&t=4-kZga74dpKGeI-p2P7Zow) discussing the performance of various LLMs, including **GPT-4 and GPT-3.5**, in generating flashcards for spaced repetition tools like Anki, noting that there is still room for improvement.
- **GPT-4 Generates Verbose but Useful Anki Cards**: User `@thebaghdaddy` found success with **GPT-4** in creating Anki cards by first organizing information into a table format covering various aspects, such as mechanisms and side effects for a list of drugs, and then prompting GPT-4 to create cards from the table, resulting in slightly verbose but useful content.
- **Anki and LLMs: The Visual Limitation**: `@thebaghdaddy` noted a limitation when integrating LLMs with Anki: the inability to include images, which are beneficial for study methods like image occlusion.

**Links mentioned**:

[Tweet from Niccolò Zanichelli (in SF in May) (@nc_znc)](https://x.com/nc_znc/status/1753847802487017911?s=46&t=4-kZga74dpKGeI-p2P7Zow): Interesting analysis evaluating the capabilities of different LLMs (GPT-4, GPT-3.5 and some open ones) w.r.t. generating spaced repetition flashcards conditioned on some explanatory text. Clear improv...

  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1211454351155925083) (5 messages): 

- **Feather Flock Together**: User `@.kiingo` linked to [Feather OpenAI](https://feather.openai.com/) sparking speculation about its purpose. `@justahvee` responded, suggesting the service seems related to writing based on its icon.
- **Unearthing Feather's Past**: `@dare.ai` clarified that Feather has been in use since 2022 and is not new, providing a snapshot link from the [The Wayback Machine](https://web.archive.org/web/20230403164757/https://feather.openai.com/).
- **Feather's Role in Training AI Models**: In another message, `@dare.ai` noted Feather's use for SME data labeling and coding annotation, critical for training models, and cited an article from [Semafor](https://www.semafor.com/article/01/27/2023/openai-has-hired-an-army-of-contractors-to-make-basic-coding-obsolete) regarding OpenAI's hiring practices.
- **GPT-4 Ada's Analytical Advancements**: User `@res6969` shared a tweet from `@btibor91` about a new GPT-4 model known as "gpt-4-ada-v2," which features a data grid overlay editor, options for 'targeted replies', and potential interactive charts, defining the updated version as "ChatGPT Data Analysis V2".

**Links mentioned**:

- [Wayback Machine](https://web.archive.org/web/20230403164): no description found
- [OpenAI has hired an army of contractors to make basic coding obsolete | Semafor](https://www.semafor.com/article/01/27/2023/openai-has-hired-an-army-of-contractors-to-make-basic-coding-obsolete): The company behind ChatGPT now employs around 1,000 people around the world to label data and help OpenAI’s models learn software engineering tasks.
- [Tweet from Tibor Blaho (@btibor91)](https://x.com/btibor91/status/1761726596585504939?s=46): ChatGPT Data Analysis V2 apparently uses a new GPT-4 model called &#34;gpt-4-ada-v2&#34; (Advanced Data Analysis V2). It adds:  - a data grid overlay editor for uploaded files  - an option for a &#34;...
- [Login to Feather](https://web.archive.org/web/20230403164757/https://feather.openai.com/): no description found

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1210623558175297618) (4 messages): 

- **Exploring Custom Callbacks in Training**: `@sebastian.bodza` discussed the potential of using custom callbacks with the [Hugging Face trainer](https://huggingface.co/docs/transformers/main_classes/callback), noting that they are currently a feature exclusive to PyTorch and are "read only," except for the control they have via `TrainerControl`.
- **LLMs and the Query of English Centricity**: `@_jp1_` pointed out an insightful paper on the English-centric thought process in open large language models (LLMs). He suggests it has significant implications for multilingual applications and shared his perspective with a [link to his tweet](https://twitter.com/jphme/status/1762032277033255208?t=IcVEkSzPbWdDTwVMesloWg&s=19).
- **Scrutinizing LLM Probability-Based Evaluations**: `@bjoernp` shared an [arXiv paper](https://arxiv.org/abs/2402.13887) that discusses the limitations of probability-based evaluation methods for LLMs, especially for multiple-choice questions, addressing a problem also encountered in the DiscoLM series research. The study casts doubts on the effectiveness of such evaluations as they may not align with generation-based prediction.

**Links mentioned**:

- [Beyond Probabilities: Unveiling the Misalignment in Evaluating Large Language Models](https://arxiv.org/abs/2402.13887): Large Language Models (LLMs) have demonstrated remarkable capabilities across various applications, fundamentally reshaping the landscape of natural language processing (NLP) research. However, recent...
- [Callbacks](https://huggingface.co/docs/transformers/main_classes/callback): no description found

  

---


### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1211590899902058557) (10 messages🔥): 

- **Emotional intelligence benchmark extends to German**: EQ-Bench has received German language support and efficiency improvements from `.calytrix`, making it faster and less resource-intensive. The update is available on the [EQ-Bench GitHub repository](https://github.com/EQ-bench/EQ-Bench).
- **Preliminary scores for German EQ-Bench revealed**: `.calytrix` listed initial scores for models on the German version of EQ-Bench, with `gpt-4-1106-preview` scoring the highest at 81.91, followed by various models, including `gpt-3.5-turbo-0125` and different versions of Mistral and Laser.
- **Concerns about the validity of translated EQ-Bench**: `_jp1_` expressed skepticism about the effectiveness of the EQ-Bench German translation, suggesting that nuances in emotional understanding might not translate well, potentially leading to similar results across different language benchmarks due to shared English-centric reasoning.
- **Translation seen as non-detrimental to benchmark efficacy**: `.calytrix` asserted that the discriminative power of EQ-Bench is retained despite potential translation issues, backed by parallel scores between English and German benchmarks, which suggest that the test is effective even if not perfect.
- **Debate on the cultural nuances in EQ-Bench translations**: `_jp1_` posited that a model's ability to understand German-specific emotional nuances could lead to different results in bilingual benchmarks, a theory `.calytrix` found compelling but remained skeptical on whether different cultural thinking could significantly influence benchmark rankings.

**Links mentioned**:

[GitHub - EQ-bench/EQ-Bench: A benchmark for emotional intelligence in large language models](https://github.com/EQ-bench/EQ-Bench): A benchmark for emotional intelligence in large language models - EQ-bench/EQ-Bench

  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1210583489066565692) (2 messages): 

- **Introducing Matryoshka Embeddings**: `@johannhartmann` shared a [Hugging Face blog post](https://huggingface.co/blog/matryoshka) introducing **Matryoshka Embeddings**, explaining their utility, how to train them using Sentence Transformers, and showcasing a demo of their capabilities. The blog provides a detailed comparison between Matryoshka embeddings and regular embeddings.
- **Sentence Transformers now feature Matryoshka**: Additionally, `@johannhartmann` mentions that Matryoshka Embeddings are now incorporated into Sentence Transformers, broadening the toolkit for users of this library.

**Links mentioned**:

[🪆 Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): no description found

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1210889254423498762) (6 messages): 

- **Dataset Accessibility Inquiry**: `@thomasrenkert` inquired about accessing the `context_understanding` dataset on Hugging Face.
- **Work-In-Progress Dataset Details**: `@bjoernp` responded that the dataset, which is a work-in-progress for a benchmark on retrieval context understanding, is not ready for broad sharing and lacks public documentation.
- **Understanding RAG Evaluation**: `@johannhartmann` questioned the approach of asking which context is used to answer a question in the `ger-rag-eval` instead of checking for a proper answer.
- **Clarifying RAG Evaluation Methodology**: `@philipmay` explained that in a RAG setting, multiple contexts are retrieved, and it's important to test whether the LLM can locate the relevant information within them.
- **Acknowledgment of Explanation**: `@johannhartmann` acknowledged the point made by `@philipmay` regarding the RAG evaluation approach.
  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1210854713583472640) (12 messages🔥): 

- **Looking for Hackathon Teammates**: `@reydelplatanos` is seeking teammates for an upcoming hackathon. `@hiro.saxophone` responded with an offer to team up, mentioning their experience as an ML engineer and previous work on a multimodal RAG.

- **Registration Woes and Team Optimism**: Both `@silverpiranha` and `@jamthewallfacer` expressed they are awaiting registration confirmation for an event. `@silverpiranha` then shared excitement about the high participation and eventual successful registration, inviting `@jamthewallfacer` to team up.

- **Back End Meets ML Engineering for Hackathon**: `@reydelplatanos`, identifying as a backend developer, accepted `@hiro.saxophone`’s offer to form a team for the hackathon, signifying a new partnership.

- **Looking for Additional Hackathon Members**: `@ryznerf.` joined the conversation late but is eager to participate in the hackathon and is looking to join a team.

- **A High-Flying Coding Idea**: `@.yosun` shared a fun hackathon idea about using function calling for piloting a drone, citing an example from the OpenAI Cookbook. They provided a snippet of code illustrating function definitions for drone operation.

**Links mentioned**:

[Fine tuning for function-calling | OpenAI Cookbook](https://cookbook.openai.com/examples/fine_tuning_for_function_calling): no description found

  

---



### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1211602243456081921) (1 messages): 

- **Gemma Introduces Conversation Control Tokens**: `@imonenext` enhanced the **Gemma-7B** model with special tokens for turn-taking in conversations. The new tokens `<start_of_turn>` and `<end_of_turn>` are designed for better instruction/RL fine-tuning and can be accessed on [Hugging Face](https://huggingface.co/imone/gemma-7b-with-it-tokens).

**Links mentioned**:

[imone/gemma-7b-with-it-tokens · Hugging Face](https://huggingface.co/imone/gemma-7b-with-it-tokens): no description found

  

---



### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1211008837587312670) (1 messages): 

- **Understanding Random Seeds in Deep Learning**: `@stereoplegic` shared an article deemed a "shockingly good read" from LinkedIn, focusing on the use of **random numbers** in deep learning, specifically in Python using the PyTorch library. They recommended it to those interested in understanding or working with random seeds: [Random Numbers in Deep Learning; Python & the PyTorch Library](https://www.linkedin.com/pulse/random-numbers-deep-learning-python-part-4-pytorch-library-jarkko-idkgf).
  

